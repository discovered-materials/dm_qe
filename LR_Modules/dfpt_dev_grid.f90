!
! Copyright (C) 2026 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
MODULE dfpt_dev_grid
  !-----------------------------------------------------------------------
  !! Real-space grid operations of the DFPT self-consistency on the GPU
  !! (companion of dfpt_dev): symmetrization of the response density /
  !! potential over the small group of q.
  !!
  !! The routines take and return host arrays; they are used when the FFT
  !! grid is not distributed over MPI ranks. QE_PH_DEV=0 selects the stock
  !! routines.
  !
  USE kinds, ONLY : DP
#if defined(__CUDA)
  USE cudafor
  USE cufft
#endif
  !
  IMPLICIT NONE
  PRIVATE
  SAVE
  !
  PUBLIC :: dfpt_dev_grid_ok, dfpt_dev_symdvscf, dfpt_dev_grid_reset
  !
  LOGICAL :: env_read = .FALSE., env_on = .TRUE.
  !
#if defined(__CUDA)
  !
  ! ... tables of dfpt_dev_dv_of_drho, valid for one q-point and ground state
  !
  LOGICAL :: xc_ready = .FALSE.
  LOGICAL :: xc_gga = .FALSE.
  INTEGER :: xc_nnr = 0, xc_ngm = 0, xc_ns = 0
  REAL(DP), ALLOCATABLE, DEVICE :: dmuxc_d(:,:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: grho_d(:,:,:), vrr_d(:,:,:), vsr_d(:,:,:), vss_d(:,:,:), &
                                   vs_d(:,:,:), rho1_d(:)
  REAL(DP), ALLOCATABLE, DEVICE :: qg_d(:,:), hfac_d(:)
  INTEGER, ALLOCATABLE, DEVICE :: nlp_d(:)
  !
  ! ... cuFFT plans on the dense grid: plan_p(1) for xc_nb fields, plan_p(2) for 3*xc_nb
  !
  INTEGER :: plan_p(2) = 0, xc_nb = 0, xc_dims(3) = 0
  !
#endif
  !
CONTAINS
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION dfpt_dev_grid_ok( dfft )
    !-----------------------------------------------------------------------
    !! True if the device routines of this module can serve grid dfft.
    !
    USE fft_types,     ONLY : fft_type_descriptor
#if defined(__CUDA)
    USE control_flags, ONLY : use_gpu
    USE mp_bands,      ONLY : nproc_bgrp
#endif
    !
    TYPE(fft_type_descriptor), INTENT(IN) :: dfft
#if defined(__CUDA)
    CHARACTER(LEN=16) :: val
    INTEGER :: ln, ierr
    !
    IF ( .NOT. env_read ) THEN
       CALL get_environment_variable( 'QE_PH_DEV', val, ln, ierr )
       env_on = .NOT. ( ierr == 0 .AND. ln > 0 .AND. val(1:1) == '0' )
       env_read = .TRUE.
    ENDIF
    dfpt_dev_grid_ok = env_on .AND. use_gpu .AND. nproc_bgrp == 1 .AND. &
         dfft%nnr == dfft%nr1*dfft%nr2*dfft%nr3 .AND. dfft%nr1x == dfft%nr1 .AND. &
         dfft%nr2x == dfft%nr2 .AND. dfft%nr3x == dfft%nr3
#else
    dfpt_dev_grid_ok = .FALSE.
#endif
    !
  END FUNCTION dfpt_dev_grid_ok
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dfpt_dev_grid_reset( )
    !-----------------------------------------------------------------------
    !! Forget the tables that depend on the q-point and on the ground state.
    !
#if defined(__CUDA)
    INTEGER :: istat
    !
    xc_ready = .FALSE.
    IF ( ALLOCATED(dmuxc_d) ) DEALLOCATE( dmuxc_d )
    IF ( ALLOCATED(grho_d) ) DEALLOCATE( grho_d, vrr_d, vsr_d, vss_d, vs_d, rho1_d )
    IF ( ALLOCATED(qg_d) ) DEALLOCATE( qg_d, hfac_d, nlp_d )
    IF ( plan_p(1) /= 0 ) istat = cufftDestroy( plan_p(1) )
    IF ( plan_p(2) /= 0 ) istat = cufftDestroy( plan_p(2) )
    plan_p = 0 ; xc_nb = 0 ; xc_dims = 0
#endif
    !
  END SUBROUTINE dfpt_dev_grid_reset
  !
#if defined(__CUDA)
#endif
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dfpt_dev_symdvscf( dvtosym, dfft, done )
    !-----------------------------------------------------------------------
    !! Device version of symdvscf: symmetrize the lr_npert response
    !! functions dvtosym over the small group of q and, if minus_q, over
    !! the operation that sends q into -q+G. done = .FALSE. on exit means
    !! that the case is not served and nothing was changed.
    !
    USE constants,        ONLY : tpi
    USE fft_types,        ONLY : fft_type_descriptor
    USE cell_base,        ONLY : at
    USE symm_base,        ONLY : s, ft, t_rev
    USE noncollin_module, ONLY : nspin_lsda, nspin_mag, noncolin, domag
    USE control_lr,       ONLY : lgamma
    USE lr_symm_base,     ONLY : minus_q, irotmq, nsymq, gi, gimq, lr_npert, upert, upert_mq
    !
    TYPE(fft_type_descriptor), INTENT(IN) :: dfft
    COMPLEX(DP), INTENT(INOUT) :: dvtosym(dfft%nnr, nspin_mag, lr_npert)
    LOGICAL, INTENT(OUT) :: done
    !
#if defined(__CUDA)
    INTEGER :: nr1, nr2, nr3, nnr, np, ns, nsy, isym, is, ipert, jpert, ir, i, j, k, ri, rj, rk, &
               rr, irm
    INTEGER, ALLOCATABLE :: ss_h(:,:,:), ftau_h(:,:)
    REAL(DP), ALLOCATABLE :: gf_h(:,:)
    REAL(DP) :: n(3), gfm(3), gfm1, gfm2, gfm3, arg
    COMPLEX(DP) :: acc, x, ph
    LOGICAL :: do_mq
    INTEGER, ALLOCATABLE, DEVICE :: ss_d(:,:,:), ftau_d(:,:), trev_d(:)
    REAL(DP), ALLOCATABLE, DEVICE :: gf_d(:,:)
    COMPLEX(DP), ALLOCATABLE, DEVICE :: u_d(:,:,:), umq_d(:,:), a_d(:,:,:), b_d(:,:,:)
    !
    done = .FALSE.
    IF ( nsymq == 1 .AND. .NOT. minus_q ) THEN
       done = .TRUE.
       RETURN
    ENDIF
    IF ( noncolin .AND. domag ) RETURN
    IF ( .NOT. dfpt_dev_grid_ok( dfft ) ) RETURN
    do_mq = minus_q .AND. .NOT. lgamma
    IF ( do_mq .AND. irotmq > nsymq ) RETURN
    !
    CALL start_clock( 'symdvscf_dev' )
    nr1 = dfft%nr1 ; nr2 = dfft%nr2 ; nr3 = dfft%nr3 ; nnr = dfft%nnr
    np = lr_npert ; ns = nspin_lsda ; nsy = nsymq ; irm = MAX(irotmq,1)
    !
    ALLOCATE( ss_h(3,3,nsymq), ftau_h(3,nsymq), gf_h(3,nsymq) )
    CALL scale_sym_ops( nsymq, s, ft, nr1, nr2, nr3, ss_h, ftau_h )
    n(1) = tpi / DBLE(nr1) ; n(2) = tpi / DBLE(nr2) ; n(3) = tpi / DBLE(nr3)
    DO isym = 1, nsymq
       gf_h(:,isym) = gi(1,isym) * at(1,:) * n(:) + gi(2,isym) * at(2,:) * n(:) + &
                      gi(3,isym) * at(3,:) * n(:)
    ENDDO
    gfm(:) = gimq(1) * at(1,:) * n(:) + gimq(2) * at(2,:) * n(:) + gimq(3) * at(3,:) * n(:)
    gfm1 = gfm(1) ; gfm2 = gfm(2) ; gfm3 = gfm(3)
    !
    ALLOCATE( ss_d(3,3,nsymq), ftau_d(3,nsymq), trev_d(nsymq), gf_d(3,nsymq) )
    ALLOCATE( u_d(np,np,nsymq), umq_d(np,np), a_d(nnr,ns,np), b_d(nnr,ns,np) )
    ss_d = ss_h ; ftau_d = ftau_h ; gf_d = gf_h
    trev_d = t_rev(1:nsymq)
    u_d = upert(1:np,1:np,1:nsymq)
    IF ( minus_q .AND. ALLOCATED(upert_mq) ) THEN
       umq_d = upert_mq(1:np,1:np)
    ELSEIF ( do_mq ) THEN
       CALL stop_clock( 'symdvscf_dev' )
       RETURN
    ENDIF
    a_d = dvtosym(:,1:ns,:)
    !
    ! ... S(irotmq) q = -q + G together with time reversal
    !
    IF ( minus_q ) THEN
       IF ( lgamma ) THEN
          !$cuf kernel do(3)
          DO ipert = 1, np
             DO is = 1, ns
                DO ir = 1, nnr
                   a_d(ir,is,ipert) = CMPLX( DBLE(a_d(ir,is,ipert)), 0.0_DP, KIND=DP )
                ENDDO
             ENDDO
          ENDDO
       ELSE
          !$cuf kernel do(3)
          DO ipert = 1, np
             DO is = 1, ns
                DO ir = 1, nnr
                   k = (ir-1) / (nr1*nr2)
                   j = (ir-1 - k*nr1*nr2) / nr1
                   i = ir-1 - k*nr1*nr2 - j*nr1
                   ri = ss_d(1,1,irm)*i + ss_d(2,1,irm)*j + ss_d(3,1,irm)*k - ftau_d(1,irm)
                   ri = MOD(ri,nr1) ; IF ( ri < 0 ) ri = ri + nr1
                   rj = ss_d(1,2,irm)*i + ss_d(2,2,irm)*j + ss_d(3,2,irm)*k - ftau_d(2,irm)
                   rj = MOD(rj,nr2) ; IF ( rj < 0 ) rj = rj + nr2
                   rk = ss_d(1,3,irm)*i + ss_d(2,3,irm)*j + ss_d(3,3,irm)*k - ftau_d(3,irm)
                   rk = MOD(rk,nr3) ; IF ( rk < 0 ) rk = rk + nr3
                   rr = 1 + ri + nr1*( rj + nr2*rk )
                   arg = gfm1*i + gfm2*j + gfm3*k
                   ph = CMPLX( COS(arg), SIN(arg), KIND=DP )
                   acc = (0.0_DP, 0.0_DP)
                   DO jpert = 1, np
                      acc = acc + umq_d(jpert,ipert) * a_d(rr,is,jpert) * ph
                   ENDDO
                   b_d(ir,is,ipert) = ( a_d(ir,is,ipert) + CONJG(acc) ) * 0.5_DP
                ENDDO
             ENDDO
          ENDDO
          !$cuf kernel do(3)
          DO ipert = 1, np
             DO is = 1, ns
                DO ir = 1, nnr
                   a_d(ir,is,ipert) = b_d(ir,is,ipert)
                ENDDO
             ENDDO
          ENDDO
       ENDIF
    ENDIF
    !
    ! ... the small group of q
    !
    IF ( nsymq > 1 ) THEN
       !$cuf kernel do(3)
       DO ipert = 1, np
          DO is = 1, ns
             DO ir = 1, nnr
                k = (ir-1) / (nr1*nr2)
                j = (ir-1 - k*nr1*nr2) / nr1
                i = ir-1 - k*nr1*nr2 - j*nr1
                acc = (0.0_DP, 0.0_DP)
                DO isym = 1, nsy
                   ri = ss_d(1,1,isym)*i + ss_d(2,1,isym)*j + ss_d(3,1,isym)*k - ftau_d(1,isym)
                   ri = MOD(ri,nr1) ; IF ( ri < 0 ) ri = ri + nr1
                   rj = ss_d(1,2,isym)*i + ss_d(2,2,isym)*j + ss_d(3,2,isym)*k - ftau_d(2,isym)
                   rj = MOD(rj,nr2) ; IF ( rj < 0 ) rj = rj + nr2
                   rk = ss_d(1,3,isym)*i + ss_d(2,3,isym)*j + ss_d(3,3,isym)*k - ftau_d(3,isym)
                   rk = MOD(rk,nr3) ; IF ( rk < 0 ) rk = rk + nr3
                   rr = 1 + ri + nr1*( rj + nr2*rk )
                   arg = gf_d(1,isym)*i + gf_d(2,isym)*j + gf_d(3,isym)*k
                   ph = CMPLX( COS(arg), SIN(arg), KIND=DP )
                   x = (0.0_DP, 0.0_DP)
                   DO jpert = 1, np
                      x = x + u_d(jpert,ipert,isym) * a_d(rr,is,jpert) * ph
                   ENDDO
                   IF ( trev_d(isym) == 0 ) THEN
                      acc = acc + x
                   ELSE
                      acc = acc + CONJG(x)
                   ENDIF
                ENDDO
                b_d(ir,is,ipert) = acc / DBLE(nsy)
             ENDDO
          ENDDO
       ENDDO
       dvtosym(:,1:ns,:) = b_d
    ELSE
       dvtosym(:,1:ns,:) = a_d
    ENDIF
    !
    DEALLOCATE( ss_d, ftau_d, trev_d, gf_d, u_d, umq_d, a_d, b_d )
    DEALLOCATE( ss_h, ftau_h, gf_h )
    done = .TRUE.
    CALL stop_clock( 'symdvscf_dev' )
#else
    done = .FALSE.
#endif
    !
  END SUBROUTINE dfpt_dev_symdvscf
  !
END MODULE dfpt_dev_grid

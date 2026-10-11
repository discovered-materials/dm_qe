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
  PUBLIC :: dfpt_dev_grid_ok, dfpt_dev_symdvscf, dfpt_dev_dv_of_drho, dfpt_dev_grid_reset
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
  !-----------------------------------------------------------------------
  SUBROUTINE xc_setup( )
    !-----------------------------------------------------------------------
    !! Upload the xc kernel (dmuxc, gradient-correction tables) and the
    !! G-space factors of the current q-point.
    !
    USE constants,         ONLY : e2, fpi
    USE cell_base,         ONLY : tpiba2
    USE fft_base,          ONLY : dfftp
    USE gvect,             ONLY : ngm, g
    USE noncollin_module,  ONLY : nspin_mag
    USE xc_lib,            ONLY : xclib_dft_is
    USE scf,               ONLY : rho, rho_core
    USE uspp,              ONLY : nlcc_any
    USE qpoint,            ONLY : xq
    USE control_lr,        ONLY : lnolr
    USE gc_lr,             ONLY : grho, dvxc_rr, dvxc_sr, dvxc_ss, dvxc_s
    USE eqv,               ONLY : dmuxc
    USE Coul_cut_2D,       ONLY : do_cutoff_2D
    USE Coul_cut_2D_ph,    ONLY : cutoff_2D_qg
    !
    REAL(DP), ALLOCATABLE :: qg_h(:,:), hfac_h(:), rho1_h(:)
    REAL(DP) :: qg2, g2
    INTEGER :: ig, nnr
    !
    CALL dfpt_dev_grid_reset( )
    nnr = dfftp%nnr
    xc_nnr = nnr ; xc_ngm = ngm ; xc_ns = nspin_mag
    xc_gga = xclib_dft_is('gradient')
    !
    ALLOCATE( dmuxc_d(nnr,xc_ns,xc_ns) )
    dmuxc_d = dmuxc(1:nnr,1:xc_ns,1:xc_ns)
    IF ( xc_gga ) THEN
       ALLOCATE( grho_d(3,nnr,xc_ns), vrr_d(nnr,xc_ns,xc_ns), vsr_d(nnr,xc_ns,xc_ns), &
                 vss_d(nnr,xc_ns,xc_ns), vs_d(nnr,xc_ns,xc_ns), rho1_d(nnr) )
       grho_d = grho(1:3,1:nnr,1:xc_ns)
       vrr_d = dvxc_rr(1:nnr,1:xc_ns,1:xc_ns)
       vsr_d = dvxc_sr(1:nnr,1:xc_ns,1:xc_ns)
       vss_d = dvxc_ss(1:nnr,1:xc_ns,1:xc_ns)
       vs_d = dvxc_s(1:nnr,1:xc_ns,1:xc_ns)
       ALLOCATE( rho1_h(nnr) )
       rho1_h(:) = rho%of_r(1:nnr,1)
       IF ( nlcc_any ) rho1_h(:) = rho1_h(:) + rho_core(1:nnr)
       rho1_d = rho1_h
       DEALLOCATE( rho1_h )
    ENDIF
    !
    ALLOCATE( qg_h(3,ngm), hfac_h(ngm) )
    DO ig = 1, ngm
       qg_h(:,ig) = xq(:) + g(:,ig)
       qg2 = qg_h(1,ig)**2 + qg_h(2,ig)**2 + qg_h(3,ig)**2
       g2 = g(1,ig)**2 + g(2,ig)**2 + g(3,ig)**2
       hfac_h(ig) = 0.0_DP
       IF ( qg2 > 1.d-8 ) THEN
          hfac_h(ig) = e2 * fpi / ( tpiba2 * qg2 )
          IF ( do_cutoff_2D ) THEN
             hfac_h(ig) = cutoff_2D_qg(ig) * hfac_h(ig)
          ELSEIF ( lnolr .AND. .NOT. g2 > 1.d-8 ) THEN
             hfac_h(ig) = 0.0_DP
          ENDIF
       ENDIF
    ENDDO
    ALLOCATE( qg_d(3,ngm), hfac_d(ngm), nlp_d(ngm) )
    qg_d = qg_h ; hfac_d = hfac_h
    nlp_d = dfftp%nl(1:ngm)
    DEALLOCATE( qg_h, hfac_h )
    xc_ready = .TRUE.
    !
  END SUBROUTINE xc_setup
  !
  !-----------------------------------------------------------------------
  SUBROUTINE xc_plans( nb, nr1, nr2, nr3 )
    !-----------------------------------------------------------------------
    INTEGER, INTENT(IN) :: nb, nr1, nr2, nr3
    INTEGER :: istat, fdim(3), nnr
    !
    IF ( nb == xc_nb .AND. ALL( xc_dims == (/ nr1, nr2, nr3 /) ) .AND. plan_p(1) /= 0 ) RETURN
    IF ( plan_p(1) /= 0 ) istat = cufftDestroy( plan_p(1) )
    IF ( plan_p(2) /= 0 ) istat = cufftDestroy( plan_p(2) )
    plan_p = 0
    fdim = (/ nr3, nr2, nr1 /)
    nnr = nr1*nr2*nr3
    istat = cufftPlanMany( plan_p(1), 3, fdim, fdim, 1, nnr, fdim, 1, nnr, CUFFT_Z2Z, nb )
    IF ( istat /= 0 ) CALL errore( 'dfpt_dev_grid', 'cufftPlanMany failed', ABS(istat) )
    istat = cufftPlanMany( plan_p(2), 3, fdim, fdim, 1, nnr, fdim, 1, nnr, CUFFT_Z2Z, 3*nb )
    IF ( istat /= 0 ) CALL errore( 'dfpt_dev_grid', 'cufftPlanMany failed', ABS(istat) )
    xc_nb = nb ; xc_dims = (/ nr1, nr2, nr3 /)
    !
  END SUBROUTINE xc_plans
#endif
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dfpt_dev_dv_of_drho( npert, drho, dv, done, drhoc, xc_only )
    !-----------------------------------------------------------------------
    !! Device version of dv_of_drho for npert perturbations at once:
    !! dv = response Hartree + xc potential of the response density drho
    !! (plus the response core charge drhoc in the xc part). LDA and GGA,
    !! nspin_mag = 1 and 2. done = .FALSE. on exit: case not served, dv
    !! untouched. With xc_only the Hartree term is left out (dv_of_drho_xc).
    !
    USE fft_base,          ONLY : dfftp
    USE gvect,             ONLY : ngm
    USE cell_base,         ONLY : tpiba
    USE noncollin_module,  ONLY : nspin_mag, nspin_lsda, noncolin
    USE funct,             ONLY : dft_is_nonlocc
    USE xc_lib,            ONLY : xclib_dft_is
    USE control_flags,     ONLY : gamma_only
    USE martyna_tuckerman, ONLY : do_comp_mt
    USE control_lr,        ONLY : lrpa
    !
    INTEGER, INTENT(IN) :: npert
    COMPLEX(DP), INTENT(IN) :: drho(dfftp%nnr, nspin_mag, npert)
    COMPLEX(DP), INTENT(INOUT) :: dv(dfftp%nnr, nspin_mag, npert)
    LOGICAL, INTENT(OUT) :: done
    COMPLEX(DP), INTENT(IN), OPTIONAL :: drhoc(dfftp%nnr, npert)
    LOGICAL, INTENT(IN), OPTIONAL :: xc_only
    !
#if defined(__CUDA)
    REAL(DP), PARAMETER :: epsr = 1.0d-6, epsg = 1.0d-10
    LOGICAL :: hartree
    INTEGER :: nnr, ns, np, nb, ngml, ir, is, js, ks, ls, ip, ig, ipol, istat, nrx
    REAL(DP) :: fac, scal, tpb, grho2
    COMPLEX(DP) :: s1, t, gd(3,2), hh(3,2), psl(2,2), dro(2), dvadd(2), aa, bb, cc
    LOGICAL :: add_nlcc, with_xc, gga
    COMPLEX(DP), ALLOCATABLE, DEVICE :: a_d(:,:,:), tot_d(:,:,:), aux_d(:,:,:), w_d(:,:,:), &
                                        gq_d(:,:,:,:), c_d(:,:)
    !
    done = .FALSE.
    IF ( noncolin .OR. gamma_only .OR. do_comp_mt .OR. dft_is_nonlocc() ) RETURN
    IF ( nspin_mag > 2 ) RETURN
    IF ( .NOT. dfpt_dev_grid_ok( dfftp ) ) RETURN
    !
    CALL start_clock( 'dv_of_drho' )
    nnr = dfftp%nnr ; ns = nspin_mag ; np = npert ; nb = ns*np ; ngml = ngm
    IF ( .NOT. xc_ready .OR. xc_nnr /= nnr .OR. xc_ngm /= ngm .OR. xc_ns /= ns ) CALL xc_setup( )
    CALL xc_plans( nb, dfftp%nr1, dfftp%nr2, dfftp%nr3 )
    add_nlcc = PRESENT(drhoc)
    with_xc = .NOT. lrpa
    hartree = .TRUE.
    IF ( PRESENT(xc_only) ) hartree = .NOT. xc_only
    IF ( .NOT. hartree ) with_xc = .TRUE.
    gga = with_xc .AND. xc_gga
    fac = 1.0_DP / DBLE(nspin_lsda)
    scal = 1.0_DP / DBLE(nnr)
    tpb = tpiba
    !
    ALLOCATE( a_d(nnr,ns,np), tot_d(nnr,ns,np), aux_d(nnr,ns,np), w_d(nnr,ns,np) )
    a_d = drho
    IF ( add_nlcc ) THEN
       ALLOCATE( c_d(nnr,np) )
       c_d = drhoc
       !$cuf kernel do(3)
       DO ip = 1, np
          DO is = 1, ns
             DO ir = 1, nnr
                tot_d(ir,is,ip) = a_d(ir,is,ip) + fac * c_d(ir,ip)
             ENDDO
          ENDDO
       ENDDO
    ELSE
       !$cuf kernel do(3)
       DO ip = 1, np
          DO is = 1, ns
             DO ir = 1, nnr
                tot_d(ir,is,ip) = a_d(ir,is,ip)
             ENDDO
          ENDDO
       ENDDO
    ENDIF
    !
    ! ... local part of the xc kernel
    !
    IF ( with_xc ) THEN
       !$cuf kernel do(3)
       DO ip = 1, np
          DO is = 1, ns
             DO ir = 1, nnr
                t = (0.0_DP, 0.0_DP)
                DO js = 1, ns
                   t = t + dmuxc_d(ir,is,js) * tot_d(ir,js,ip)
                ENDDO
                aux_d(ir,is,ip) = t
             ENDDO
          ENDDO
       ENDDO
    ELSE
       !$cuf kernel do(3)
       DO ip = 1, np
          DO is = 1, ns
             DO ir = 1, nnr
                aux_d(ir,is,ip) = (0.0_DP, 0.0_DP)
             ENDDO
          ENDDO
       ENDDO
    ENDIF
    !
    ! ... gradient correction (dgradcorr)
    !
    IF ( gga ) THEN
       ALLOCATE( gq_d(nnr,3,ns,np) )
       !$cuf kernel do(3)
       DO ip = 1, np
          DO is = 1, ns
             DO ir = 1, nnr
                w_d(ir,is,ip) = tot_d(ir,is,ip)
             ENDDO
          ENDDO
       ENDDO
       istat = cufftExecZ2Z( plan_p(1), w_d, w_d, CUFFT_FORWARD )
       !$cuf kernel do(4)
       DO ip = 1, np
          DO is = 1, ns
             DO ipol = 1, 3
                DO ir = 1, nnr
                   gq_d(ir,ipol,is,ip) = (0.0_DP, 0.0_DP)
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       !$cuf kernel do(4)
       DO ip = 1, np
          DO is = 1, ns
             DO ipol = 1, 3
                DO ig = 1, ngml
                   gq_d(nlp_d(ig),ipol,is,ip) = CMPLX( 0.0_DP, qg_d(ipol,ig)*scal*tpb, KIND=DP ) * &
                                                w_d(nlp_d(ig),is,ip)
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       istat = cufftExecZ2Z( plan_p(2), gq_d, gq_d, CUFFT_INVERSE )
       !
       ! ... gq = gradient of the response density; on exit of the kernel gq = h
       !
       IF ( ns == 1 ) THEN
          !$cuf kernel do(2)
          DO ip = 1, np
             DO ir = 1, nnr
                grho2 = grho_d(1,ir,1)**2 + grho_d(2,ir,1)**2 + grho_d(3,ir,1)**2
                IF ( ABS(rho1_d(ir)) > epsr .AND. grho2 > epsg ) THEN
                   s1 = grho_d(1,ir,1) * gq_d(ir,1,1,ip) + grho_d(2,ir,1) * gq_d(ir,2,1,ip) + &
                        grho_d(3,ir,1) * gq_d(ir,3,1,ip)
                   aux_d(ir,1,ip) = aux_d(ir,1,ip) + vrr_d(ir,1,1) * tot_d(ir,1,ip) + &
                                    vsr_d(ir,1,1) * s1
                   t = vsr_d(ir,1,1) * tot_d(ir,1,ip) + vss_d(ir,1,1) * s1
                   gq_d(ir,1,1,ip) = t * grho_d(1,ir,1) + vs_d(ir,1,1) * gq_d(ir,1,1,ip)
                   gq_d(ir,2,1,ip) = t * grho_d(2,ir,1) + vs_d(ir,1,1) * gq_d(ir,2,1,ip)
                   gq_d(ir,3,1,ip) = t * grho_d(3,ir,1) + vs_d(ir,1,1) * gq_d(ir,3,1,ip)
                ELSE
                   gq_d(ir,1,1,ip) = (0.0_DP, 0.0_DP)
                   gq_d(ir,2,1,ip) = (0.0_DP, 0.0_DP)
                   gq_d(ir,3,1,ip) = (0.0_DP, 0.0_DP)
                ENDIF
             ENDDO
          ENDDO
       ELSE
          !$acc parallel loop collapse(2) private(gd, hh, psl, dro, dvadd, aa, bb, cc)
          DO ip = 1, np
             DO ir = 1, nnr
                DO is = 1, 2
                   dro(is) = tot_d(ir,is,ip)
                   DO ipol = 1, 3
                      gd(ipol,is) = gq_d(ir,ipol,is,ip)
                      hh(ipol,is) = (0.0_DP, 0.0_DP)
                   ENDDO
                   dvadd(is) = (0.0_DP, 0.0_DP)
                ENDDO
                DO is = 1, 2
                   DO js = 1, 2
                      psl(is,js) = grho_d(1,ir,is) * gd(1,js) + grho_d(2,ir,is) * gd(2,js) + &
                                   grho_d(3,ir,is) * gd(3,js)
                   ENDDO
                ENDDO
                DO is = 1, 2
                   DO js = 1, 2
                      dvadd(is) = dvadd(is) + vrr_d(ir,is,js) * dro(js)
                      DO ipol = 1, 3
                         hh(ipol,is) = hh(ipol,is) + vs_d(ir,is,js) * gd(ipol,js)
                      ENDDO
                      DO ks = 1, 2
                         ! a(is,js,ks), c(is,js,ks) of dgradcorr
                         IF ( is == js .AND. js == ks ) THEN
                            aa = vsr_d(ir,is,is)
                            cc = vsr_d(ir,is,is)
                         ELSE
                            IF ( is == 1 ) THEN
                               aa = vsr_d(ir,1,2)
                            ELSE
                               aa = vsr_d(ir,2,1)
                            ENDIF
                            IF ( js == 1 ) THEN
                               cc = vsr_d(ir,1,2)
                            ELSE
                               cc = vsr_d(ir,2,1)
                            ENDIF
                         ENDIF
                         dvadd(is) = dvadd(is) + aa * psl(js,ks)
                         DO ipol = 1, 3
                            ! ps1(ipol,js,ks) = drhoout(js) * grho(ipol,ks)
                            hh(ipol,is) = hh(ipol,is) + cc * dro(js) * grho_d(ipol,ir,ks)
                         ENDDO
                         DO ls = 1, 2
                            IF ( is == js .AND. js == ks .AND. ks == ls ) THEN
                               bb = vss_d(ir,is,is)
                            ELSE
                               IF ( is == 1 ) THEN
                                  bb = vss_d(ir,1,2)
                               ELSE
                                  bb = vss_d(ir,2,1)
                               ENDIF
                            ENDIF
                            DO ipol = 1, 3
                               ! ps2(ipol,js,ks,ls) = ps(js,ks) * grho(ipol,ls)
                               hh(ipol,is) = hh(ipol,is) + bb * psl(js,ks) * grho_d(ipol,ir,ls)
                            ENDDO
                         ENDDO
                      ENDDO
                   ENDDO
                ENDDO
                DO is = 1, 2
                   aux_d(ir,is,ip) = aux_d(ir,is,ip) + dvadd(is)
                   DO ipol = 1, 3
                      gq_d(ir,ipol,is,ip) = hh(ipol,is)
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDIF
       !
       ! ... minus the divergence of h (fft_qgraddot)
       !
       istat = cufftExecZ2Z( plan_p(2), gq_d, gq_d, CUFFT_FORWARD )
       !$cuf kernel do(3)
       DO ip = 1, np
          DO is = 1, ns
             DO ir = 1, nnr
                w_d(ir,is,ip) = (0.0_DP, 0.0_DP)
             ENDDO
          ENDDO
       ENDDO
       !$cuf kernel do(3)
       DO ip = 1, np
          DO is = 1, ns
             DO ig = 1, ngml
                w_d(nlp_d(ig),is,ip) = CMPLX( 0.0_DP, scal*tpb, KIND=DP ) * &
                     ( qg_d(1,ig) * gq_d(nlp_d(ig),1,is,ip) + qg_d(2,ig) * gq_d(nlp_d(ig),2,is,ip) + &
                       qg_d(3,ig) * gq_d(nlp_d(ig),3,is,ip) )
             ENDDO
          ENDDO
       ENDDO
       istat = cufftExecZ2Z( plan_p(1), w_d, w_d, CUFFT_INVERSE )
       !$cuf kernel do(3)
       DO ip = 1, np
          DO is = 1, ns
             DO ir = 1, nnr
                aux_d(ir,is,ip) = aux_d(ir,is,ip) - w_d(ir,is,ip)
             ENDDO
          ENDDO
       ENDDO
       DEALLOCATE( gq_d )
    ENDIF
    !
    ! ... Hartree term in G-space, for the total (up + down) response density
    !
    IF ( .NOT. hartree ) THEN
       dv = aux_d
       DEALLOCATE( a_d, tot_d, aux_d, w_d )
       IF ( add_nlcc ) DEALLOCATE( c_d )
       done = .TRUE.
       CALL stop_clock( 'dv_of_drho' )
       RETURN
    ENDIF
    IF ( ns == 2 ) THEN
       !$cuf kernel do(2)
       DO ip = 1, np
          DO ir = 1, nnr
             w_d(ir,1,ip) = a_d(ir,1,ip) + a_d(ir,2,ip)
             w_d(ir,2,ip) = (0.0_DP, 0.0_DP)
          ENDDO
       ENDDO
    ELSE
       !$cuf kernel do(2)
       DO ip = 1, np
          DO ir = 1, nnr
             w_d(ir,1,ip) = a_d(ir,1,ip)
          ENDDO
       ENDDO
    ENDIF
    istat = cufftExecZ2Z( plan_p(1), w_d, w_d, CUFFT_FORWARD )
    istat = cufftExecZ2Z( plan_p(1), aux_d, aux_d, CUFFT_FORWARD )
    !$cuf kernel do(3)
    DO ip = 1, np
       DO is = 1, ns
          DO ir = 1, nnr
             aux_d(ir,is,ip) = aux_d(ir,is,ip) * scal
          ENDDO
       ENDDO
    ENDDO
    !$cuf kernel do(3)
    DO ip = 1, np
       DO is = 1, ns
          DO ig = 1, ngml
             aux_d(nlp_d(ig),is,ip) = aux_d(nlp_d(ig),is,ip) + hfac_d(ig) * scal * w_d(nlp_d(ig),1,ip)
          ENDDO
       ENDDO
    ENDDO
    istat = cufftExecZ2Z( plan_p(1), aux_d, aux_d, CUFFT_INVERSE )
    dv = aux_d
    !
    DEALLOCATE( a_d, tot_d, aux_d, w_d )
    IF ( add_nlcc ) DEALLOCATE( c_d )
    done = .TRUE.
    CALL stop_clock( 'dv_of_drho' )
#else
    done = .FALSE.
#endif
    !
  END SUBROUTINE dfpt_dev_dv_of_drho
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

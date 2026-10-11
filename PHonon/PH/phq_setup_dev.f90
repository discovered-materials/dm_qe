!
! Copyright (C) 2026 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
MODULE phq_setup_dev
  !-----------------------------------------------------------------------
  !! k-point loops of the set-up of a phonon calculation with ultrasoft
  !! pseudopotentials or PAW (phq_init -> drho) on the device.
  !!
  !! The stock compute_drhous / incdrhous build, for every k-point and every
  !! one of the 3*nat displacement modes, the change of the wavefunctions
  !! due to the orthogonality constraint with a six-fold scalar loop, then
  !! transform it to real space one band at a time on the CPU. Here, for
  !! each k-point, the modes are processed together:
  !!
  !!   ps1(i,j;nu) = -wgg(i,j) [ X_nu^T (q conj(becq)) + becp1^T (q conj(Y_nu)) ]
  !!   X_nu = sum_ipol u(na ipol,nu) alphap_ipol,   Y_nu = sum_ipol conj(u) alpq_ipol
  !!   dpsi(:,i;nu) = evq ps1(i,:;nu)^T
  !!
  !! are four matrix products for all modes, bands and projectors, followed
  !! by batched FFTs, the accumulation of dpsi(r) conj(psi(r)) and one more
  !! matrix product for <beta|dpsi> (addusdbec).
  !!
  !! The term of the dynamical matrix due to the orthogonality constraint
  !! (compute_nldyn: one small ZGEMV per atom, cartesian direction, band and
  !! second atom) is rewritten as
  !!
  !!   dyn(nu,mu) = 2 wk sum_ij wgg(i,j) A2(j,i;nu) A1(j,i;mu)
  !!
  !! with A1, A2 built by a few kernels and one matrix product per k-point;
  !! its part with the local potential (drho: (3 nat)^2 scalar products of
  !! functions of the smooth grid) is a matrix product.
  !!
  !! Served: collinear, no DFT+U, all atoms displaced (no nat_todo), FFT grids
  !! not distributed over MPI ranks. The modes are processed in chunks sized
  !! on the free device memory.
  !
  USE kinds, ONLY : DP
#if defined(__CUDA)
  USE cudafor
  USE cufft
  USE cublas_v2
#endif
  !
  IMPLICIT NONE
  PRIVATE
  SAVE
  !
  PUBLIC :: phq_setup_dev_on, drhous_dev
  !
#if defined(__CUDA)
  COMPLEX(DP), PARAMETER :: c_one = (1.0_DP, 0.0_DP), c_zero = (0.0_DP, 0.0_DP)
  INTEGER, PARAMETER :: nplan_max = 8
  INTEGER :: plan(nplan_max) = 0, plan_nb(nplan_max) = 0, plan_age(nplan_max) = 0, &
             plan_dim(3,nplan_max) = 0, plan_clock = 0
  TYPE(cublasHandle) :: hblas
  LOGICAL :: blas_ready = .FALSE.
  INTEGER :: dbg_chunk = -1
#endif
  !
CONTAINS
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION phq_setup_dev_on( )
    !-----------------------------------------------------------------------
    !! The device code of this module can be used in this run.
#if defined(__CUDA)
    USE usaug_dev,  ONLY : usaug_on
    USE fft_base,   ONLY : dffts
    USE partial,    ONLY : nat_todo_input
    !
    phq_setup_dev_on = usaug_on( ) .AND. nat_todo_input <= 0 .AND. &
         .NOT. dffts%has_task_groups .AND. dffts%nnr == dffts%nr1*dffts%nr2*dffts%nr3 .AND. &
         dffts%nr1 == dffts%nr1x .AND. dffts%nr2 == dffts%nr2x .AND. dffts%nr3 == dffts%nr3x
#else
    phq_setup_dev_on = .FALSE.
#endif
  END FUNCTION phq_setup_dev_on
  !
#if defined(__CUDA)
  !
  !-----------------------------------------------------------------------
  REAL(DP) FUNCTION free_bytes( )
    !-----------------------------------------------------------------------
    INTEGER(KIND=cuda_count_kind) :: mfree, mtot
    INTEGER :: istat
    istat = cudaMemGetInfo( mfree, mtot )
    free_bytes = REAL( mfree, DP )
  END FUNCTION free_bytes
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION ncol_fit( bytes_per_col, nwant )
    !-----------------------------------------------------------------------
    !! How many of nwant columns of bytes_per_col bytes are processed at a
    !! time: what fits in a quarter of the free device memory
    !! (QE_USAUG_CHUNK=n in the environment: at most n, for testing).
    REAL(DP), INTENT(IN) :: bytes_per_col
    INTEGER, INTENT(IN) :: nwant
    REAL(DP) :: x
    CHARACTER(LEN=16) :: val
    INTEGER :: ln, ierr, ios
    !
    IF ( dbg_chunk < 0 ) THEN
       dbg_chunk = 0
       CALL get_environment_variable( 'QE_USAUG_CHUNK', val, ln, ierr )
       IF ( ierr == 0 .AND. ln > 0 ) THEN
          READ( val, *, IOSTAT=ios ) dbg_chunk
          IF ( ios /= 0 ) dbg_chunk = 0
       ENDIF
    ENDIF
    x = MIN( REAL(nwant,DP), 0.25_DP * free_bytes( ) / bytes_per_col )
    ncol_fit = MAX( 1, INT(x) )
    IF ( dbg_chunk > 0 ) ncol_fit = MIN( ncol_fit, dbg_chunk )
  END FUNCTION ncol_fit
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION get_plan( n1, n2, n3, nb )
    !-----------------------------------------------------------------------
    !! cuFFT plan of nb transforms on a grid n1 x n2 x n3
    INTEGER, INTENT(IN) :: n1, n2, n3, nb
    INTEGER :: i, k, istat, fdim(3), nnr
    !
    plan_clock = plan_clock + 1
    DO i = 1, nplan_max
       IF ( plan_nb(i) == nb .AND. plan_dim(1,i) == n1 .AND. plan_dim(2,i) == n2 .AND. &
            plan_dim(3,i) == n3 ) THEN
          plan_age(i) = plan_clock
          get_plan = plan(i)
          RETURN
       ENDIF
    ENDDO
    k = 1
    DO i = 1, nplan_max
       IF ( plan_nb(i) == 0 ) THEN
          k = i
          EXIT
       ENDIF
       IF ( plan_age(i) < plan_age(k) ) k = i
    ENDDO
    IF ( plan_nb(k) /= 0 ) istat = cufftDestroy( plan(k) )
    fdim = (/ n3, n2, n1 /)
    nnr = n1 * n2 * n3
    istat = cufftPlanMany( plan(k), 3, fdim, fdim, 1, nnr, fdim, 1, nnr, CUFFT_Z2Z, nb )
    IF ( istat /= 0 ) CALL errore( 'phq_setup_dev', 'cufftPlanMany failed', ABS(istat) )
    plan_nb(k) = nb
    plan_dim(:,k) = (/ n1, n2, n3 /)
    plan_age(k) = plan_clock
    get_plan = plan(k)
    !
  END FUNCTION get_plan
  !
  !-----------------------------------------------------------------------
  SUBROUTINE free_plans( )
    !-----------------------------------------------------------------------
    INTEGER :: i, istat
    DO i = 1, nplan_max
       IF ( plan_nb(i) /= 0 ) istat = cufftDestroy( plan(i) )
    ENDDO
    plan = 0 ; plan_nb = 0 ; plan_age = 0 ; plan_dim = 0 ; plan_clock = 0
  END SUBROUTINE free_plans
  !
  !-----------------------------------------------------------------------
  SUBROUTINE drhous_dev( drhous, dbecsum, wgg, becq, alpq, done )
    !-----------------------------------------------------------------------
    !! Change of the charge density and of becsum due to the orthogonality
    !! constraint, for all the displacement modes: compute_drhous + incdrhous
    !! + addusdbec. done = .FALSE. on exit: nothing was computed (not enough
    !! device memory), the caller has to use the stock code.
    !
    USE ions_base,   ONLY : nat, ityp
    USE cell_base,   ONLY : omega
    USE wavefunctions, ONLY : evc
    USE buffers,     ONLY : get_buffer
    USE uspp,        ONLY : nkb, vkb, qq_nt, ofsbeta, ijtoh
    USE uspp_param,  ONLY : nhm, nh, upf
    USE lsda_mod,    ONLY : lsda, nspin, current_spin, isk
    USE klist,       ONLY : xk, wk, ngk, igk_k
    USE fft_base,    ONLY : dffts
    USE wvfct,       ONLY : nbnd, npwx
    USE qpoint,      ONLY : nksq, ikks, ikqs
    USE eqv,         ONLY : evq
    USE control_lr,  ONLY : lgamma, nbnd_occ
    USE units_lr,    ONLY : iuwfc, lrwfc
    USE becmod,      ONLY : bec_type
    USE uspp_init,   ONLY : init_us_2
    USE modes,       ONLY : u
    USE phus,        ONLY : alphap
    USE lrus,        ONLY : becp1
    !
    COMPLEX(DP), INTENT(INOUT) :: dbecsum(nhm*(nhm+1)/2,nat,nspin,3*nat)
    !! the derivative of becsum
    COMPLEX(DP), INTENT(INOUT) :: drhous(dffts%nnr,nspin,3*nat)
    !! the orthogonality term of the change of the charge density
    REAL(DP), INTENT(IN) :: wgg(nbnd,nbnd,nksq)
    !! the weights
    TYPE(bec_type), INTENT(IN) :: becq(nksq), alpq(3,nksq)
    !! <beta|psi_{k+q}> and its derivatives
    LOGICAL, INTENT(OUT) :: done
    !
    INTEGER :: nm, nnr, n1, n2, n3, nbo, ne, ik, ikk, ikq, npw, npwq, nocc, isp, na, nt, ih, jh, &
               ie, ikb, jkb, mu, nu, ig, ir, ibnd, jbnd, ipol, im, m0, nmc, nmcur, c, j, j0, nb, &
               nfb, ncols, ip, istat, iof, nkbl, nbndl, npwxl, e, ldp
    REAL(DP) :: weight, wgt, per_mode, s1
    COMPLEX(DP) :: s
    INTEGER, ALLOCATABLE :: ivec(:), e_na(:), e_ijh(:), e_ikb(:), e_jkb(:), nl_h(:)
    COMPLEX(DP), ALLOCATABLE :: ue_h(:,:), stage(:,:,:), drho_h(:,:), dbs_h(:,:)
    REAL(DP), ALLOCATABLE :: qq_h(:,:,:)
    INTEGER, ALLOCATABLE, DEVICE :: kbna_d(:), kbof_d(:), kbnh_d(:), e_ikb_d(:), e_jkb_d(:), &
                                    nlk_d(:), nlq_d(:)
    REAL(DP), ALLOCATABLE, DEVICE :: qq_d(:,:,:), wgg_d(:,:)
    COMPLEX(DP), ALLOCATABLE, DEVICE :: ue_d(:,:), vkb_d(:,:), evq_d(:,:), psir_d(:,:), &
                                        bp1_d(:,:), bq_d(:,:), alp_d(:,:,:), alq_d(:,:,:), &
                                        qbc_d(:,:), x_d(:,:), qy_d(:,:), p1_d(:,:), p2_d(:,:), &
                                        cc_d(:,:), dpsi_d(:,:), fr_d(:,:), drho_d(:,:), &
                                        dbq_d(:,:), dbs_d(:,:)
    !
    done = .FALSE.
    nm = 3 * nat
    nnr = dffts%nnr ; n1 = dffts%nr1 ; n2 = dffts%nr2 ; n3 = dffts%nr3
    nkbl = nkb ; nbndl = nbnd ; npwxl = npwx
    nbo = 0
    DO ik = 1, nksq
       nbo = MAX( nbo, nbnd_occ(ikks(ik)) )
    ENDDO
    IF ( nkb <= 0 .OR. nbo <= 0 ) RETURN
    !
    ! ... entries (atom, ih <= jh) of dbecsum
    !
    ne = 0
    DO na = 1, nat
       nt = ityp(na)
       IF ( upf(nt)%tvanp ) ne = ne + ( nh(nt) * (nh(nt)+1) ) / 2
    ENDDO
    !
    ! ... device memory: data of a k-point, then what is needed per mode
    !
    s1 = 16.0_DP * ( REAL(nnr,DP) * nbo + REAL(npwx,DP) * (nkb + nbnd) + 9.0_DP * nkb * nbnd )
    IF ( s1 > 0.3_DP * free_bytes( ) ) RETURN
    per_mode = 16.0_DP * ( REAL(nnr,DP) + REAL(npwx,DP) * nbo + 2.0_DP * nkb * nbo + &
                           REAL(nkb,DP) * nbnd + 3.0_DP * nbo * nbnd + MAX(ne,1) )
    !
    IF ( .NOT. blas_ready ) THEN
       istat = cublasCreate( hblas )
       IF ( istat /= 0 ) CALL errore( 'phq_setup_dev', 'cublasCreate failed', ABS(istat) )
       blas_ready = .TRUE.
    ENDIF
    !
    ! ... displacements: zero for the atoms that a mode does not move
    ! ... (incdrhous skips them)
    !
    ALLOCATE( ue_h(nm,nm) )
    DO nu = 1, nm
       DO na = 1, nat
          mu = 3 * (na - 1)
          IF ( ABS(u(mu+1,nu)) + ABS(u(mu+2,nu)) + ABS(u(mu+3,nu)) > 1.0E-12_DP ) THEN
             ue_h(mu+1:mu+3,nu) = u(mu+1:mu+3,nu)
          ELSE
             ue_h(mu+1:mu+3,nu) = c_zero
          ENDIF
       ENDDO
    ENDDO
    ALLOCATE( ue_d(nm,nm) )
    ue_d = ue_h
    DEALLOCATE( ue_h )
    !
    ! ... for each beta function: atom, offset of the atom, number of beta
    ! ... functions of the atom; q functions per atom
    !
    ALLOCATE( ivec(nkb), kbna_d(nkb), kbof_d(nkb), kbnh_d(nkb) )
    DO na = 1, nat
       DO ih = 1, nh(ityp(na))
          ivec(ofsbeta(na)+ih) = na
       ENDDO
    ENDDO
    kbna_d = ivec
    DO na = 1, nat
       DO ih = 1, nh(ityp(na))
          ivec(ofsbeta(na)+ih) = ofsbeta(na)
       ENDDO
    ENDDO
    kbof_d = ivec
    DO na = 1, nat
       DO ih = 1, nh(ityp(na))
          ivec(ofsbeta(na)+ih) = nh(ityp(na))
       ENDDO
    ENDDO
    kbnh_d = ivec
    DEALLOCATE( ivec )
    ALLOCATE( qq_h(nhm,nhm,nat), qq_d(nhm,nhm,nat) )
    DO na = 1, nat
       qq_h(:,:,na) = qq_nt(:,:,ityp(na))
    ENDDO
    qq_d = qq_h
    DEALLOCATE( qq_h )
    !
    ALLOCATE( e_na(MAX(ne,1)), e_ijh(MAX(ne,1)), e_ikb(MAX(ne,1)), e_jkb(MAX(ne,1)) )
    ie = 0
    DO na = 1, nat
       nt = ityp(na)
       IF ( .NOT. upf(nt)%tvanp ) CYCLE
       DO ih = 1, nh(nt)
          DO jh = ih, nh(nt)
             ie = ie + 1
             e_na(ie) = na
             e_ijh(ie) = ijtoh(ih,jh,nt)
             e_ikb(ie) = ofsbeta(na) + ih
             e_jkb(ie) = ofsbeta(na) + jh
          ENDDO
       ENDDO
    ENDDO
    ALLOCATE( e_ikb_d(MAX(ne,1)), e_jkb_d(MAX(ne,1)) )
    e_ikb_d = e_ikb ; e_jkb_d = e_jkb
    !
    ! ... data of a k-point
    !
    ALLOCATE( nlk_d(npwx), nlq_d(npwx), nl_h(npwx) )
    ALLOCATE( vkb_d(npwx,nkb), evq_d(npwx,nbnd), psir_d(nnr,nbo), wgg_d(nbnd,nbnd) )
    ALLOCATE( bp1_d(nkb,nbnd), bq_d(nkb,nbnd), alp_d(nkb,nbnd,3), alq_d(nkb,nbnd,3), qbc_d(nkb,nbnd) )
    ALLOCATE( stage(nkb,nbnd,3) )
    !
    ! ... modes at a time, then FFTs at a time
    !
    nmc = ncol_fit( per_mode, nm )
    ldp = nbo * nmc
    ALLOCATE( x_d(nkb,nbo*nmc), qy_d(nkb,nbnd*nmc), p1_d(ldp,nbnd), p2_d(nbo,nbnd*nmc), &
              cc_d(nbnd,nbo*nmc), dpsi_d(npwx,nbo*nmc), drho_d(nnr,nmc), dbq_d(nkb,nbo*nmc), &
              dbs_d(MAX(ne,1),nmc) )
    ALLOCATE( drho_h(nnr,nmc), dbs_h(MAX(ne,1),nmc) )
    nfb = ncol_fit( 32.0_DP * REAL(nnr,DP), nbo*nmc )
    ALLOCATE( fr_d(nnr,nfb) )
    !
    drhous(:,:,:) = c_zero
    dbecsum(:,:,:,:) = c_zero
    !
    DO ik = 1, nksq
       ikk = ikks(ik)
       ikq = ikqs(ik)
       npw = ngk(ikk)
       npwq = ngk(ikq)
       nocc = nbnd_occ(ikk)
       weight = wk(ikk)
       wgt = 2.0_DP * weight / omega
       IF ( lsda ) current_spin = isk(ikk)
       isp = 1
       IF ( lsda ) isp = isk(ikk)
       !
       ! ... beta functions at k+q, wavefunctions at k and k+q
       !
       CALL init_us_2( npwq, igk_k(1,ikq), xk(1,ikq), vkb )
       vkb_d = vkb
       CALL get_buffer( evc, lrwfc, iuwfc, ikk )
       IF ( .NOT. lgamma .AND. nksq > 1 ) CALL get_buffer( evq, lrwfc, iuwfc, ikq )
       evq_d = evq(1:npwx,1:nbnd)
       nl_h(:) = 1
       DO ig = 1, npw
          nl_h(ig) = dffts%nl(igk_k(ig,ikk))
       ENDDO
       nlk_d = nl_h
       nl_h(:) = 1
       DO ig = 1, npwq
          nl_h(ig) = dffts%nl(igk_k(ig,ikq))
       ENDDO
       nlq_d = nl_h
       !
       ! ... psi_k(r) of the occupied bands (dpsi_d is used as work space)
       !
       dpsi_d(:,1:nocc) = evc(1:npwx,1:nocc)
       DO j0 = 0, nocc-1, nfb
          nb = MIN( nfb, nocc-j0 )
          !$cuf kernel do(2)
          DO j = 1, nb
             DO ir = 1, nnr
                fr_d(ir,j) = (0.0_DP, 0.0_DP)
             ENDDO
          ENDDO
          !$cuf kernel do(2)
          DO j = 1, nb
             DO ig = 1, npw
                fr_d(nlk_d(ig),j) = dpsi_d(ig,j0+j)
             ENDDO
          ENDDO
          ip = get_plan( n1, n2, n3, nb )
          istat = cufftExecZ2Z( ip, fr_d, fr_d, CUFFT_INVERSE )
          IF ( istat /= 0 ) CALL errore( 'drhous_dev', 'cufftExecZ2Z failed', ABS(istat) )
          !$cuf kernel do(2)
          DO j = 1, nb
             DO ir = 1, nnr
                psir_d(ir,j0+j) = fr_d(ir,j)
             ENDDO
          ENDDO
       ENDDO
       !
       ! ... <beta|psi> and derivatives at k and k+q, weights
       !
       bp1_d = becp1(ik)%k
       bq_d = becq(ik)%k
       DO ipol = 1, 3
          stage(:,:,ipol) = alphap(ipol,ik)%k(:,:)
       ENDDO
       alp_d = stage
       DO ipol = 1, 3
          stage(:,:,ipol) = alpq(ipol,ik)%k(:,:)
       ENDDO
       alq_d = stage
       wgg_d = wgg(:,:,ik)
       !
       ! ... q conj(becq)
       !
       !$cuf kernel do(2)
       DO jbnd = 1, nbndl
          DO ikb = 1, nkbl
             na = kbna_d(ikb) ; iof = kbof_d(ikb) ; ih = ikb - iof
             s = (0.0_DP, 0.0_DP)
             DO jh = 1, kbnh_d(ikb)
                s = s + qq_d(ih,jh,na) * CONJG( bq_d(iof+jh,jbnd) )
             ENDDO
             qbc_d(ikb,jbnd) = s
          ENDDO
       ENDDO
       !
       DO m0 = 0, nm-1, nmc
          nmcur = MIN( nmc, nm-m0 )
          ncols = nocc * nmcur
          !
          ! ... X_nu and q conj(Y_nu)
          !
          !$cuf kernel do(2)
          DO c = 1, ncols
             DO ikb = 1, nkbl
                im = (c-1)/nocc + 1
                ibnd = c - (im-1)*nocc
                mu = 3 * (kbna_d(ikb) - 1)
                x_d(ikb,c) = ue_d(mu+1,m0+im) * alp_d(ikb,ibnd,1) + &
                             ue_d(mu+2,m0+im) * alp_d(ikb,ibnd,2) + &
                             ue_d(mu+3,m0+im) * alp_d(ikb,ibnd,3)
             ENDDO
          ENDDO
          !$cuf kernel do(2)
          DO c = 1, nbndl*nmcur
             DO ikb = 1, nkbl
                im = (c-1)/nbndl + 1
                jbnd = c - (im-1)*nbndl
                na = kbna_d(ikb) ; iof = kbof_d(ikb) ; ih = ikb - iof
                mu = 3 * (na - 1)
                s = (0.0_DP, 0.0_DP)
                DO jh = 1, kbnh_d(ikb)
                   s = s + qq_d(ih,jh,na) * ( ue_d(mu+1,m0+im) * CONJG( alq_d(iof+jh,jbnd,1) ) + &
                                              ue_d(mu+2,m0+im) * CONJG( alq_d(iof+jh,jbnd,2) ) + &
                                              ue_d(mu+3,m0+im) * CONJG( alq_d(iof+jh,jbnd,3) ) )
                ENDDO
                qy_d(ikb,c) = s
             ENDDO
          ENDDO
          !
          ! ... ps1, with the bands of psi_{k+q} as rows
          !
          istat = cublasZgemm( hblas, CUBLAS_OP_T, CUBLAS_OP_N, ncols, nbnd, nkb, c_one, &
                               x_d, nkb, qbc_d, nkb, c_zero, p1_d, ldp )
          IF ( istat /= 0 ) CALL errore( 'drhous_dev', 'cublasZgemm failed', ABS(istat) )
          istat = cublasZgemm( hblas, CUBLAS_OP_T, CUBLAS_OP_N, nocc, nbnd*nmcur, nkb, c_one, &
                               bp1_d, nkb, qy_d, nkb, c_zero, p2_d, nbo )
          IF ( istat /= 0 ) CALL errore( 'drhous_dev', 'cublasZgemm failed', ABS(istat) )
          !$cuf kernel do(2)
          DO c = 1, ncols
             DO jbnd = 1, nbndl
                im = (c-1)/nocc + 1
                ibnd = c - (im-1)*nocc
                cc_d(jbnd,c) = - wgg_d(ibnd,jbnd) * ( p1_d(c,jbnd) + p2_d(ibnd,(im-1)*nbndl+jbnd) )
             ENDDO
          ENDDO
          !
          ! ... dpsi = evq ps1^T, <beta|dpsi>
          !
          istat = cublasZgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwq, ncols, nbnd, c_one, &
                               evq_d, npwx, cc_d, nbnd, c_zero, dpsi_d, npwx )
          IF ( istat /= 0 ) CALL errore( 'drhous_dev', 'cublasZgemm failed', ABS(istat) )
          istat = cublasZgemm( hblas, CUBLAS_OP_C, CUBLAS_OP_N, nkb, ncols, npwq, c_one, &
                               vkb_d, npwx, dpsi_d, npwx, c_zero, dbq_d, nkb )
          IF ( istat /= 0 ) CALL errore( 'drhous_dev', 'cublasZgemm failed', ABS(istat) )
          !
          ! ... drho(r,nu) = wgt sum_i dpsi_i(r;nu) conj(psi_i(r))
          !
          !$cuf kernel do(2)
          DO im = 1, nmcur
             DO ir = 1, nnr
                drho_d(ir,im) = (0.0_DP, 0.0_DP)
             ENDDO
          ENDDO
          DO j0 = 0, ncols-1, nfb
             nb = MIN( nfb, ncols-j0 )
             !$cuf kernel do(2)
             DO j = 1, nb
                DO ir = 1, nnr
                   fr_d(ir,j) = (0.0_DP, 0.0_DP)
                ENDDO
             ENDDO
             !$cuf kernel do(2)
             DO j = 1, nb
                DO ig = 1, npwq
                   fr_d(nlq_d(ig),j) = dpsi_d(ig,j0+j)
                ENDDO
             ENDDO
             ip = get_plan( n1, n2, n3, nb )
             istat = cufftExecZ2Z( ip, fr_d, fr_d, CUFFT_INVERSE )
             IF ( istat /= 0 ) CALL errore( 'drhous_dev', 'cufftExecZ2Z failed', ABS(istat) )
             !$cuf kernel do(1)
             DO ir = 1, nnr
                DO j = 1, nb
                   c = j0 + j
                   im = (c-1)/nocc + 1
                   ibnd = c - (im-1)*nocc
                   drho_d(ir,im) = drho_d(ir,im) + wgt * fr_d(ir,j) * CONJG( psir_d(ir,ibnd) )
                ENDDO
             ENDDO
          ENDDO
          drho_h(:,1:nmcur) = drho_d(:,1:nmcur)
          !$omp parallel do private(ir)
          DO im = 1, nmcur
             DO ir = 1, nnr
                drhous(ir,isp,m0+im) = drhous(ir,isp,m0+im) + drho_h(ir,im)
             ENDDO
          ENDDO
          !$omp end parallel do
          !
          ! ... dbecsum (addusdbec)
          !
          IF ( ne > 0 ) THEN
             !$cuf kernel do(2)
             DO im = 1, nmcur
                DO e = 1, ne
                   ikb = e_ikb_d(e) ; jkb = e_jkb_d(e)
                   s = (0.0_DP, 0.0_DP)
                   IF ( ikb == jkb ) THEN
                      DO ibnd = 1, nocc
                         s = s + weight * ( CONJG( bp1_d(ikb,ibnd) ) * dbq_d(ikb,(im-1)*nocc+ibnd) )
                      ENDDO
                   ELSE
                      DO ibnd = 1, nocc
                         s = s + weight * ( CONJG( bp1_d(ikb,ibnd) ) * dbq_d(jkb,(im-1)*nocc+ibnd) + &
                                            CONJG( bp1_d(jkb,ibnd) ) * dbq_d(ikb,(im-1)*nocc+ibnd) )
                      ENDDO
                   ENDIF
                   dbs_d(e,im) = s
                ENDDO
             ENDDO
             dbs_h(:,1:nmcur) = dbs_d(:,1:nmcur)
             DO im = 1, nmcur
                DO e = 1, ne
                   dbecsum(e_ijh(e),e_na(e),isp,m0+im) = dbecsum(e_ijh(e),e_na(e),isp,m0+im) + dbs_h(e,im)
                ENDDO
             ENDDO
          ENDIF
       ENDDO
    ENDDO
    !
    DEALLOCATE( ue_d, kbna_d, kbof_d, kbnh_d, qq_d, e_ikb_d, e_jkb_d, nlk_d, nlq_d )
    DEALLOCATE( vkb_d, evq_d, psir_d, wgg_d, bp1_d, bq_d, alp_d, alq_d, qbc_d )
    DEALLOCATE( x_d, qy_d, p1_d, p2_d, cc_d, dpsi_d, drho_d, dbq_d, dbs_d, fr_d )
    DEALLOCATE( e_na, e_ijh, e_ikb, e_jkb, nl_h, stage, drho_h, dbs_h )
    CALL free_plans( )
    done = .TRUE.
    !
  END SUBROUTINE drhous_dev
  !
#else
  !
  SUBROUTINE drhous_dev( drhous, dbecsum, wgg, becq, alpq, done )
    USE ions_base,   ONLY : nat
    USE uspp_param,  ONLY : nhm
    USE lsda_mod,    ONLY : nspin
    USE fft_base,    ONLY : dffts
    USE wvfct,       ONLY : nbnd
    USE qpoint,      ONLY : nksq
    USE becmod,      ONLY : bec_type
    COMPLEX(DP), INTENT(INOUT) :: dbecsum(nhm*(nhm+1)/2,nat,nspin,3*nat)
    COMPLEX(DP), INTENT(INOUT) :: drhous(dffts%nnr,nspin,3*nat)
    REAL(DP), INTENT(IN) :: wgg(nbnd,nbnd,nksq)
    TYPE(bec_type), INTENT(IN) :: becq(nksq), alpq(3,nksq)
    LOGICAL, INTENT(OUT) :: done
    done = .FALSE.
  END SUBROUTINE drhous_dev
  !
#endif
  !
END MODULE phq_setup_dev

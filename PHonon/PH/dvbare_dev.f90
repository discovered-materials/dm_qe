!
! Copyright (C) 2026 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
MODULE dvbare_dev
  !-----------------------------------------------------------------------
  !! Bare perturbation \(dV_\text{bare}/du\,\psi\) of ph.x on GPUs.
  !!
  !! The stock dvqpsi_us works band by band (two FFTs, several small
  !! kernels and data regions per band), computes the nonlocal term on the
  !! CPU and rebuilds the derivative of the local potential for every
  !! k-point. Here one call is a handful of device operations:
  !!
  !! * the derivative of the local potential in real space is computed
  !!   once per displacement pattern and kept on the device (a small cache
  !!   of patterns, dvloc_dev_slot);
  !! * the local term of all bands is two batched cuFFTs;
  !! * the nonlocal term is a small host loop for the coefficients
  !!   (dvqpsi_nl_coef) followed by two ZGEMMs with the beta functions.
  !!
  !! dvqpsi_us_dev leaves the result in a device array given by the
  !! caller, so that it can be written straight into the arrays of the
  !! device DFPT engine; dvqpsi_us_host is the variant that copies it to
  !! a host array (used by dvqpsi_us).
  !!
  !! Served: collinear, norm-conserving, ultrasoft and PAW, serial FFT
  !! grids. Noncollinear, DFT+U and multipole perturbations use the stock
  !! path. QE_PH_DEV=0 selects the stock path everywhere.
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
  PUBLIC :: ph_dev_on, dvbare_dev_ok, dvbare_dev_reset, dvbare_dev_free
  !
  LOGICAL :: env_read = .FALSE., env_on = .FALSE.
  !
#if defined(__CUDA)
  !
  PUBLIC :: dvqpsi_us_dev, dvqpsi_us_host, dvqpsi_nl_coef, dvloc_dev_slot, dvl_d, &
            dvbare_blas, dev_fft3
  !
  COMPLEX(DP), PARAMETER :: c_one = (1.0_DP, 0.0_DP), c_zero = (0.0_DP, 0.0_DP)
  INTEGER, PARAMETER :: max_fft_elems = 16777216
  !! complex numbers in one FFT work array (256 MB)
  INTEGER, PARAMETER :: max_tmp_elems = 8388608
  !! complex numbers in the work array of the nonlocal term (128 MB)
  REAL(DP), PARAMETER :: max_cache_bytes = 268435456.0_DP
  !! device memory of the cache of local potential derivatives
  INTEGER, PARAMETER :: max_slots = 48
  !
  ! ... dimensions the work arrays were allocated for
  !
  INTEGER :: w_nnr = 0, w_nr(3) = 0, w_npwx = 0, w_nbnd = 0, w_nkb = 0, w_nat = 0, &
             w_ngms = 0, w_ngm = 0, w_ntyp = 0
  INTEGER :: nfb = 0
  !! bands per FFT batch
  INTEGER :: nbc = 0
  !! bands per block of the nonlocal term
  !
  COMPLEX(DP), ALLOCATABLE, DEVICE :: psig_d(:), psic_d(:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: dvq_d(:,:)
  !! (npwx, nbnd): result of dvqpsi_us_host before the copy to the host
  COMPLEX(DP), ALLOCATABLE, DEVICE :: tmp_d(:,:), ps_d(:,:)
  INTEGER, ALLOCATABLE, DEVICE :: nlk_d(:,:)
  !! FFT index of k+G (:,1) and of k+q+G (:,2)
  REAL(DP), ALLOCATABLE, DEVICE :: kqg_d(:,:)
  !! k+q+G
  INTEGER, ALLOCATABLE, DEVICE :: nls_d(:)
  !! FFT index of the G vectors of the smooth grid
  INTEGER, ALLOCATABLE, DEVICE :: imov_d(:,:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: cmov_d(:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: vlq_d(:,:)
  !! device copy of vlocq
  REAL(DP) :: vlq_q(3) = 0.0_DP
  LOGICAL :: vlq_ready = .FALSE.
  !
  INTEGER, ALLOCATABLE :: nl_h(:,:), imov_h(:,:)
  REAL(DP), ALLOCATABLE :: kqg_h(:,:)
  COMPLEX(DP), ALLOCATABLE :: ps_h(:,:), cmov_h(:,:)
  !
  ! ... cache of dV_loc/du in real space
  !
  INTEGER :: nslot = 0, last_slot = 0
  COMPLEX(DP), ALLOCATABLE, DEVICE :: dvl_d(:,:)
  !! (nnr, nslot)
  COMPLEX(DP), ALLOCATABLE :: key_u(:,:)
  REAL(DP), ALLOCATABLE :: key_q(:,:)
  INTEGER, ALLOCATABLE :: key_tag(:)
  !! -1: empty, 0: without core correction, is > 0: with core correction for spin is
  !
  ! ... cuFFT plans and cuBLAS handle
  !
  INTEGER, PARAMETER :: nplan_max = 6
  INTEGER :: plan(nplan_max) = 0, plan_nb(nplan_max) = 0, plan_age(nplan_max) = 0, &
             plan_dim(3,nplan_max) = 0, plan_clock = 0
  TYPE(cublasHandle) :: hblas
  LOGICAL :: blas_ready = .FALSE.
  !
#endif
  !
CONTAINS
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION ph_dev_on( )
    !-----------------------------------------------------------------------
    !! The device paths of ph.x are on by default in GPU builds;
    !! QE_PH_DEV=0 selects the stock path. The variable is read once.
    !
#if defined(__CUDA)
    CHARACTER(LEN=16) :: val
    INTEGER :: ln, ierr
#endif
    !
    IF ( .NOT. env_read ) THEN
       env_on = .FALSE.
#if defined(__CUDA)
       CALL get_environment_variable( 'QE_PH_DEV', val, ln, ierr )
       env_on = .NOT. ( ierr == 0 .AND. ln > 0 .AND. val(1:1) == '0' )
       ! QE_PH_DYN=0 switches off only the routines of this package
       ! (bare perturbation and dynamical-matrix terms)
       CALL get_environment_variable( 'QE_PH_DYN', val, ln, ierr )
       IF ( ierr == 0 .AND. ln > 0 .AND. val(1:1) == '0' ) env_on = .FALSE.
#endif
       env_read = .TRUE.
    ENDIF
    ph_dev_on = env_on
    !
  END FUNCTION ph_dev_on
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION dvbare_dev_ok( )
    !-----------------------------------------------------------------------
    !! True when the device routines of this module serve the present run.
    !
    USE control_flags,    ONLY : use_gpu, gamma_only
    USE noncollin_module, ONLY : noncolin
    USE ldaU,             ONLY : lda_plus_u
    USE control_ph,       ONLY : lmultipole
    USE fft_base,         ONLY : dffts
    USE mp_bands,         ONLY : nproc_bgrp, nbgrp
    !
    dvbare_dev_ok = .FALSE.
    IF ( .NOT. ph_dev_on( ) ) RETURN
    dvbare_dev_ok = use_gpu .AND. .NOT. ( noncolin .OR. lda_plus_u .OR. lmultipole .OR. &
         gamma_only .OR. dffts%has_task_groups ) .AND. nproc_bgrp == 1 .AND. nbgrp == 1 .AND. &
         dffts%nnr == dffts%nr1*dffts%nr2*dffts%nr3 .AND. dffts%nr1 == dffts%nr1x .AND. &
         dffts%nr2 == dffts%nr2x .AND. dffts%nr3 == dffts%nr3x
    !
  END FUNCTION dvbare_dev_ok
  !
#if defined(__CUDA)
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dvbare_dev_reset( )
    !-----------------------------------------------------------------------
    !! Forget the cached derivatives of the local potential (new q-point).
    !
    IF ( ALLOCATED(key_tag) ) key_tag(:) = -1
    last_slot = 0
    vlq_ready = .FALSE.
    !
  END SUBROUTINE dvbare_dev_reset
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dvbare_dev_free( )
    !-----------------------------------------------------------------------
    !! Release all device memory of this module.
    !
    INTEGER :: i, istat
    !
    IF ( ALLOCATED(psig_d) ) DEALLOCATE( psig_d, psic_d )
    IF ( ALLOCATED(dvq_d) ) DEALLOCATE( dvq_d )
    IF ( ALLOCATED(tmp_d) ) DEALLOCATE( tmp_d )
    IF ( ALLOCATED(ps_d) ) DEALLOCATE( ps_d )
    IF ( ALLOCATED(nlk_d) ) DEALLOCATE( nlk_d, kqg_d )
    IF ( ALLOCATED(nls_d) ) DEALLOCATE( nls_d )
    IF ( ALLOCATED(imov_d) ) DEALLOCATE( imov_d, cmov_d )
    IF ( ALLOCATED(vlq_d) ) DEALLOCATE( vlq_d )
    IF ( ALLOCATED(dvl_d) ) DEALLOCATE( dvl_d )
    IF ( ALLOCATED(key_u) ) DEALLOCATE( key_u, key_q, key_tag )
    IF ( ALLOCATED(nl_h) ) DEALLOCATE( nl_h, kqg_h )
    IF ( ALLOCATED(ps_h) ) DEALLOCATE( ps_h )
    IF ( ALLOCATED(imov_h) ) DEALLOCATE( imov_h, cmov_h )
    DO i = 1, nplan_max
       IF ( plan_nb(i) /= 0 ) istat = cufftDestroy( plan(i) )
    ENDDO
    plan = 0 ; plan_nb = 0 ; plan_age = 0 ; plan_dim = 0
    w_nnr = 0 ; w_nr = 0 ; w_npwx = 0 ; w_nbnd = 0 ; w_nkb = 0 ; w_nat = 0
    w_ngms = 0 ; w_ngm = 0 ; w_ntyp = 0
    nslot = 0 ; last_slot = 0 ; nfb = 0 ; nbc = 0
    vlq_ready = .FALSE.
    !
  END SUBROUTINE dvbare_dev_free
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dvbare_blas( h )
    !-----------------------------------------------------------------------
    !! the cuBLAS handle of the device routines of ph.x
    TYPE(cublasHandle), INTENT(OUT) :: h
    INTEGER :: istat
    IF ( .NOT. blas_ready ) THEN
       istat = cublasCreate( hblas )
       IF ( istat /= 0 ) CALL errore( 'dvbare_dev', 'cublasCreate failed', ABS(istat) )
       blas_ready = .TRUE.
    ENDIF
    h = hblas
  END SUBROUTINE dvbare_blas
  !
  !-----------------------------------------------------------------------
  SUBROUTINE work_setup( )
    !-----------------------------------------------------------------------
    !! (Re)allocate the work arrays when the dimensions of the run change.
    !
    USE fft_base,   ONLY : dffts
    USE wvfct,      ONLY : npwx, nbnd
    USE uspp,       ONLY : nkb
    USE ions_base,  ONLY : nat, ntyp => nsp
    USE gvecs,      ONLY : ngms
    USE gvect,      ONLY : ngm
    !
    INTEGER :: nnr, istat
    TYPE(cublasHandle) :: h
    !
    nnr = dffts%nnr
    IF ( nnr == w_nnr .AND. dffts%nr1 == w_nr(1) .AND. dffts%nr2 == w_nr(2) .AND. &
         dffts%nr3 == w_nr(3) .AND. npwx == w_npwx .AND. nbnd == w_nbnd .AND. nkb == w_nkb &
         .AND. nat == w_nat .AND. ngms == w_ngms .AND. ngm == w_ngm .AND. ntyp == w_ntyp ) RETURN
    !
    CALL dvbare_dev_free( )
    IF ( .NOT. blas_ready ) CALL dvbare_blas( h )
    !
    w_nnr = nnr ; w_nr = (/ dffts%nr1, dffts%nr2, dffts%nr3 /)
    w_npwx = npwx ; w_nbnd = nbnd ; w_nkb = nkb ; w_nat = nat
    w_ngms = ngms ; w_ngm = ngm ; w_ntyp = ntyp
    !
    nfb = MAX( 1, MIN( nbnd, max_fft_elems / nnr ) )
    nbc = MAX( 1, MIN( nbnd, max_tmp_elems / (3*npwx) ) )
    ALLOCATE( psig_d(nnr*nfb), psic_d(nnr*nfb), STAT=istat )
    IF ( istat /= 0 ) CALL errore( 'dvbare_dev', 'device allocation failed', 1 )
    psig_d = c_zero
    ALLOCATE( dvq_d(npwx,nbnd), nlk_d(npwx,2), kqg_d(3,npwx), nls_d(ngms), STAT=istat )
    IF ( istat /= 0 ) CALL errore( 'dvbare_dev', 'device allocation failed', 2 )
    ALLOCATE( nl_h(npwx,2), kqg_h(3,npwx) )
    nls_d(:) = dffts%nl(1:ngms)
    IF ( nkb > 0 ) THEN
       ALLOCATE( tmp_d(npwx,3*nbc), ps_d(nkb,4*nbnd), STAT=istat )
       IF ( istat /= 0 ) CALL errore( 'dvbare_dev', 'device allocation failed', 3 )
       ALLOCATE( ps_h(nkb,4*nbnd) )
    ENDIF
    ALLOCATE( imov_d(2,nat), cmov_d(5,nat), imov_h(2,nat), cmov_h(5,nat) )
    ALLOCATE( vlq_d(ngm,ntyp) )
    !
    nslot = MAX( 1, MIN( 3*nat, max_slots, INT( max_cache_bytes / (16.0_DP*nnr) ) ) )
    ALLOCATE( dvl_d(nnr,nslot), STAT=istat )
    IF ( istat /= 0 ) CALL errore( 'dvbare_dev', 'device allocation failed', 4 )
    ALLOCATE( key_u(3*nat,nslot), key_q(3,nslot), key_tag(nslot) )
    key_tag(:) = -1
    last_slot = 0
    !
  END SUBROUTINE work_setup
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION get_plan( nb )
    !-----------------------------------------------------------------------
    !! index of a plan for nb transforms on the smooth grid (made on demand)
    INTEGER, INTENT(IN) :: nb
    get_plan = get_plan_nd( w_nr(1), w_nr(2), w_nr(3), nb )
  END FUNCTION get_plan
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION get_plan_nd( n1, n2, n3, nb )
    !-----------------------------------------------------------------------
    !! index of a plan for nb transforms on a grid n1 x n2 x n3 (made on
    !! demand; the least recently used plan is replaced when the table is full)
    INTEGER, INTENT(IN) :: n1, n2, n3, nb
    INTEGER :: i, j, istat, fdim(3), nnr
    !
    plan_clock = plan_clock + 1
    DO i = 1, nplan_max
       IF ( plan_nb(i) == nb .AND. plan_dim(1,i) == n1 .AND. plan_dim(2,i) == n2 .AND. &
            plan_dim(3,i) == n3 ) THEN
          plan_age(i) = plan_clock
          get_plan_nd = i
          RETURN
       ENDIF
    ENDDO
    j = 1
    DO i = 1, nplan_max
       IF ( plan_nb(i) == 0 ) THEN
          j = i
          EXIT
       ENDIF
       IF ( plan_age(i) < plan_age(j) ) j = i
    ENDDO
    IF ( plan_nb(j) /= 0 ) istat = cufftDestroy( plan(j) )
    fdim = (/ n3, n2, n1 /)
    nnr = n1 * n2 * n3
    istat = cufftPlanMany( plan(j), 3, fdim, fdim, 1, nnr, fdim, 1, nnr, CUFFT_Z2Z, nb )
    IF ( istat /= 0 ) CALL errore( 'dvbare_dev', 'cufftPlanMany failed', ABS(istat) )
    plan_nb(j) = nb
    plan_dim(:,j) = (/ n1, n2, n3 /)
    plan_age(j) = plan_clock
    get_plan_nd = j
    !
  END FUNCTION get_plan_nd
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dev_fft3( n1, n2, n3, nb, x_d, isign )
    !-----------------------------------------------------------------------
    !! nb FFTs in place on a device array, grid n1 x n2 x n3, no scaling:
    !! isign = +1 from G to r (as invfft), -1 from r to G (fwfft without
    !! the factor 1/(n1 n2 n3)).
    INTEGER, INTENT(IN) :: n1, n2, n3, nb, isign
    COMPLEX(DP), DEVICE :: x_d(*)
    INTEGER :: ip
    ip = get_plan_nd( n1, n2, n3, nb )
    IF ( isign > 0 ) THEN
       CALL fft_inplace( ip, x_d, CUFFT_INVERSE )
    ELSE
       CALL fft_inplace( ip, x_d, CUFFT_FORWARD )
    ENDIF
  END SUBROUTINE dev_fft3
  !
  !-----------------------------------------------------------------------
  SUBROUTINE fft_inplace( ip, x_d, dir )
    !-----------------------------------------------------------------------
    INTEGER, INTENT(IN) :: ip, dir
    COMPLEX(DP), DEVICE :: x_d(*)
    INTEGER :: istat
    istat = cufftExecZ2Z( plan(ip), x_d, x_d, dir )
    IF ( istat /= 0 ) CALL errore( 'dvbare_dev', 'cufftExecZ2Z failed', ABS(istat) )
  END SUBROUTINE fft_inplace
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vlq_sync( )
    !-----------------------------------------------------------------------
    !! vlq_d = vlocq of the current q-point
    USE eqv,    ONLY : vlocq
    USE qpoint, ONLY : xq
    !
    IF ( .NOT. ALLOCATED(vlq_d) ) CALL work_setup( )
    IF ( vlq_ready .AND. ALL( xq(:) == vlq_q(:) ) ) RETURN
    vlq_d = vlocq
    vlq_q(:) = xq(:)
    vlq_ready = .TRUE.
    !
  END SUBROUTINE vlq_sync
  !
  !-----------------------------------------------------------------------
  SUBROUTINE moved_atoms( uact, nmov, imov, cmov )
    !-----------------------------------------------------------------------
    !! List of the atoms displaced by the pattern uact, as in compute_dvloc:
    !! imov = (atom, type), cmov = (u1, u2, u3, q.u, -i tpiba e^{-i q tau}).
    !
    USE ions_base, ONLY : nat, ityp
    USE cell_base, ONLY : tpiba
    USE qpoint,    ONLY : xq, eigqts
    !
    COMPLEX(DP), INTENT(IN) :: uact(3*nat)
    INTEGER, INTENT(OUT) :: nmov, imov(2,nat)
    COMPLEX(DP), INTENT(OUT) :: cmov(5,nat)
    INTEGER :: na, mu
    COMPLEX(DP) :: u1, u2, u3
    !
    nmov = 0
    DO na = 1, nat
       mu = 3 * (na - 1)
       u1 = uact(mu+1) ; u2 = uact(mu+2) ; u3 = uact(mu+3)
       IF ( ABS(u1) + ABS(u2) + ABS(u3) > 1.0d-12 ) THEN
          nmov = nmov + 1
          imov(1,nmov) = na
          imov(2,nmov) = ityp(na)
          cmov(1,nmov) = u1 ; cmov(2,nmov) = u2 ; cmov(3,nmov) = u3
          cmov(4,nmov) = xq(1) * u1 + xq(2) * u2 + xq(3) * u3
          cmov(5,nmov) = tpiba * (0.d0, -1.d0) * eigqts(na)
       ENDIF
    ENDDO
    !
  END SUBROUTINE moved_atoms
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION dvloc_dev_slot( uact, addnlcc )
    !-----------------------------------------------------------------------
    !! Column of dvl_d that holds \(dV_\text{loc}/du\) in real space on the
    !! smooth grid for the pattern uact (compute_dvloc, plus
    !! compute_dvloc_cc when addnlcc is true). It is computed on the first
    !! request and found in the cache afterwards.
    !
    USE ions_base, ONLY : nat
    USE gvect,     ONLY : eigts1, eigts2, eigts3, mill, g
    USE gvecs,     ONLY : ngms
    USE qpoint,    ONLY : xq
    USE uspp,      ONLY : nlcc_any
    USE lsda_mod,  ONLY : lsda, current_spin
    USE fft_base,  ONLY : dffts
    !
    COMPLEX(DP), INTENT(IN) :: uact(3*nat)
    LOGICAL, INTENT(IN) :: addnlcc
    !
    INTEGER :: tag, i, islot, nmov, ig, na, nt, nnr, ngl, ip
    COMPLEX(DP) :: s, gtau, gu
    COMPLEX(DP), ALLOCATABLE :: dvh(:)
    !
    CALL work_setup( )
    tag = 0
    IF ( addnlcc .AND. nlcc_any ) THEN
       tag = 1
       IF ( lsda ) tag = current_spin
    ENDIF
    DO i = 1, nslot
       IF ( key_tag(i) /= tag ) CYCLE
       IF ( ANY( key_q(:,i) /= xq(:) ) ) CYCLE
       IF ( ANY( key_u(:,i) /= uact(:) ) ) CYCLE
       dvloc_dev_slot = i
       RETURN
    ENDDO
    !
    CALL start_clock( 'dvq_dvloc' )
    islot = MOD( last_slot, nslot ) + 1
    last_slot = islot
    key_tag(islot) = -1
    nnr = w_nnr ; ngl = ngms
    !
    IF ( tag == 0 ) THEN
       CALL vlq_sync( )
       CALL moved_atoms( uact, nmov, imov_h, cmov_h )
       !$cuf kernel do(1)
       DO ig = 1, nnr
          dvl_d(ig,islot) = (0.0_DP, 0.0_DP)
       ENDDO
       IF ( nmov > 0 ) THEN
          imov_d = imov_h
          cmov_d = cmov_h
          !$acc parallel loop present(eigts1, eigts2, eigts3, mill, g) private(s, gtau, gu, na, nt)
          DO ig = 1, ngl
             s = (0.0_DP, 0.0_DP)
             !$acc loop seq
             DO i = 1, nmov
                na = imov_d(1,i) ; nt = imov_d(2,i)
                gtau = eigts1(mill(1,ig),na) * eigts2(mill(2,ig),na) * eigts3(mill(3,ig),na)
                gu = cmov_d(4,i) + g(1,ig) * cmov_d(1,i) + g(2,ig) * cmov_d(2,i) + &
                     g(3,ig) * cmov_d(3,i)
                s = s + vlq_d(ig,nt) * gu * cmov_d(5,i) * gtau
             ENDDO
             dvl_d(nls_d(ig),islot) = s
          ENDDO
          ip = get_plan( 1 )
          CALL fft_inplace( ip, dvl_d(1,islot), CUFFT_INVERSE )
       ENDIF
    ELSE
       !
       ! ... with the core correction: the stock routines, then to the device
       !
       ALLOCATE( dvh(nnr) )
       dvh(:) = (0.0_DP, 0.0_DP)
       CALL compute_dvloc( uact, .FALSE., dvh )
       CALL compute_dvloc_cc( uact, dvh )
       dvl_d(:,islot) = dvh(:)
       DEALLOCATE( dvh )
    ENDIF
    !
    key_u(:,islot) = uact(:)
    key_q(:,islot) = xq(:)
    key_tag(islot) = tag
    dvloc_dev_slot = islot
    CALL stop_clock( 'dvq_dvloc' )
    !
  END FUNCTION dvloc_dev_slot
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dvqpsi_nl_coef( ik, uact, becp1, alphap, ps )
    !-----------------------------------------------------------------------
    !! Coefficients of the nonlocal part of \(dV_\text{bare}/du\,\psi\)
    !! (dvqpsi_us_only, collinear case): the term proportional to
    !! \(\beta(k+q+G)\) in ps(:,ibnd) and the three terms proportional to
    !! \((k+q+G)_\alpha\beta(k+q+G)\) in ps(:,nbnd+3*(ibnd-1)+alpha).
    !! The current spin must be set by the caller.
    !
    USE cell_base,  ONLY : tpiba
    USE ions_base,  ONLY : nat, ityp
    USE lsda_mod,   ONLY : current_spin
    USE wvfct,      ONLY : nbnd, et
    USE uspp,       ONLY : okvan, nkb, ofsbeta, deeq, qq_at
    USE uspp_param, ONLY : nh, nhm
    USE phus,       ONLY : int1, int2
    USE qpoint,     ONLY : nksq, ikks
    USE becmod,     ONLY : bec_type
    !
    INTEGER, INTENT(IN) :: ik
    COMPLEX(DP), INTENT(IN) :: uact(3*nat)
    TYPE(bec_type), INTENT(IN) :: becp1(nksq), alphap(3,nksq)
    COMPLEX(DP), INTENT(OUT) :: ps(nkb,4*nbnd)
    !
    REAL(DP), PARAMETER :: eps = 1.d-12
    INTEGER :: ikk, is, na, nb, nt, mu, nu, ih, jh, ikb, jkb, ijkb0, ipol, ibnd, im, nmov, i2
    INTEGER, ALLOCATABLE :: mov(:)
    COMPLEX(DP), ALLOCATABLE :: aus(:,:,:)
    COMPLEX(DP) :: s1, s2(3), bj
    REAL(DP) :: d, e
    LOGICAL :: ok
    !
    ikk = ikks(ik)
    is = current_spin
    ALLOCATE( mov(nat) )
    nmov = 0
    DO na = 1, nat
       mu = 3 * (na - 1)
       IF ( ABS(uact(mu+1)) + ABS(uact(mu+2)) + ABS(uact(mu+3)) > eps ) THEN
          nmov = nmov + 1
          mov(nmov) = na
       ENDIF
    ENDDO
    !
    ! ... ultrasoft: the term with int2 does not depend on the band
    !
    IF ( okvan ) THEN
       ALLOCATE( aus(nhm,nhm,nat) )
       aus(:,:,:) = (0.0_DP, 0.0_DP)
       !$omp parallel do private(na, nt, im, nb, nu, ih, jh, ipol) if(nmov*nat > 64)
       DO na = 1, nat
          nt = ityp(na)
          DO im = 1, nmov
             nb = mov(im)
             nu = 3 * (nb - 1)
             DO ih = 1, nh(nt)
                DO jh = 1, nh(nt)
                   DO ipol = 1, 3
                      aus(ih,jh,na) = aus(ih,jh,na) + int2(ih,jh,ipol,nb,na) * uact(nu+ipol)
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       !$omp end parallel do
    ENDIF
    !
    !$omp parallel do private(ibnd, e, i2, im, na, nt, mu, ijkb0, ih, jh, ikb, jkb, ipol, &
    !$omp                     s1, s2, d, bj) if(nbnd*nkb > 20000)
    DO ibnd = 1, nbnd
       e = et(ibnd,ikk)
       i2 = nbnd + 3*(ibnd-1)
       ps(:,ibnd) = (0.0_DP, 0.0_DP)
       ps(:,i2+1:i2+3) = (0.0_DP, 0.0_DP)
       DO im = 1, nmov
          na = mov(im)
          nt = ityp(na)
          mu = 3 * (na - 1)
          ijkb0 = ofsbeta(na)
          DO ih = 1, nh(nt)
             ikb = ijkb0 + ih
             s1 = (0.0_DP, 0.0_DP)
             s2(:) = (0.0_DP, 0.0_DP)
             DO jh = 1, nh(nt)
                jkb = ijkb0 + jh
                d = deeq(ih,jh,na,is)
                IF ( okvan ) d = d - e * qq_at(ih,jh,na)
                bj = becp1(ik)%k(jkb,ibnd)
                DO ipol = 1, 3
                   s1 = s1 + d * alphap(ipol,ik)%k(jkb,ibnd) * uact(mu+ipol)
                   s2(ipol) = s2(ipol) + d * bj * (0.0_DP,-1.0_DP) * uact(mu+ipol) * tpiba
                   IF ( okvan ) s1 = s1 + ( int1(ih,jh,ipol,na,is) * bj ) * uact(mu+ipol)
                ENDDO
             ENDDO
             ps(ikb,ibnd) = s1
             ps(ikb,i2+1:i2+3) = s2(:)
          ENDDO
       ENDDO
       IF ( okvan .AND. nmov > 0 ) THEN
          DO na = 1, nat
             nt = ityp(na)
             ijkb0 = ofsbeta(na)
             DO ih = 1, nh(nt)
                s1 = (0.0_DP, 0.0_DP)
                DO jh = 1, nh(nt)
                   s1 = s1 + aus(ih,jh,na) * becp1(ik)%k(ijkb0+jh,ibnd)
                ENDDO
                ps(ijkb0+ih,ibnd) = ps(ijkb0+ih,ibnd) + s1
             ENDDO
          ENDDO
       ENDIF
    ENDDO
    !$omp end parallel do
    !
    ! ... as in the stock routine, a term (k+q+G)_alpha beta that is below
    ! ... eps for all bands is dropped
    !
    DO im = 1, nmov
       na = mov(im)
       DO ih = 1, nh(ityp(na))
          ikb = ofsbeta(na) + ih
          DO ipol = 1, 3
             ok = .FALSE.
             DO ibnd = 1, nbnd
                IF ( ABS( ps(ikb,nbnd+3*(ibnd-1)+ipol) ) > eps ) THEN
                   ok = .TRUE.
                   EXIT
                ENDIF
             ENDDO
             IF ( .NOT. ok ) THEN
                DO ibnd = 1, nbnd
                   ps(ikb,nbnd+3*(ibnd-1)+ipol) = (0.0_DP, 0.0_DP)
                ENDDO
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    DEALLOCATE( mov )
    IF ( okvan ) DEALLOCATE( aus )
    !
  END SUBROUTINE dvqpsi_nl_coef
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dvqpsi_us_dev( ik, uact, addnlcc, becp1, alphap, nb, out_d, ldo, &
                            psi_d, ldp, beta_d, ldb )
    !-----------------------------------------------------------------------
    !! \(dV_\text{bare}/du\,\psi_k\) for the first nb bands of k-point ik,
    !! written to the device array out_d(1:npwx,1:nb) (zero above npwq).
    !! Same input as dvqpsi_us: the wavefunctions in evc, the beta
    !! functions of k+q in vkb (device copy), the current spin set.
    !! A caller that already has \(\psi_k\) and/or \(\beta_{k+q}\) on the
    !! device passes them in psi_d (leading dimension ldp, at least nb
    !! columns) and beta_d (leading dimension ldb, nkb columns): evc and
    !! vkb are then not used.
    !
    USE ions_base,     ONLY : nat
    USE fft_base,      ONLY : dffts
    USE wvfct,         ONLY : npwx, nbnd
    USE wavefunctions, ONLY : evc
    USE klist,         ONLY : xk, ngk, igk_k
    USE gvect,         ONLY : g
    USE qpoint,        ONLY : nksq, ikks, ikqs
    USE uspp,          ONLY : nkb, vkb
    USE becmod,        ONLY : bec_type
    !
    INTEGER, INTENT(IN) :: ik, nb, ldo
    COMPLEX(DP), INTENT(IN) :: uact(3*nat)
    LOGICAL, INTENT(IN) :: addnlcc
    TYPE(bec_type), INTENT(IN) :: becp1(nksq), alphap(3,nksq)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: out_d(ldo,nb)
    INTEGER, OPTIONAL, INTENT(IN) :: ldp, ldb
    COMPLEX(DP), DEVICE, OPTIONAL, INTENT(IN) :: psi_d(*), beta_d(*)
    !
    INTEGER :: ikk, ikq, npw, npwq, islot, ig, j, j0, nbl, ip, ir, nnr, istat, npwxl, ipol, ldpl
    REAL(DP) :: scale
    LOGICAL :: own_psi, own_beta
    !
    IF ( nb > nbnd .OR. ldo < npwx ) CALL errore( 'dvqpsi_us_dev', 'wrong dimensions', 1 )
    own_psi = PRESENT(psi_d) .AND. PRESENT(ldp)
    own_beta = PRESENT(beta_d) .AND. PRESENT(ldb)
    ldpl = npwx
    IF ( own_psi ) ldpl = ldp
    CALL work_setup( )
    ikk = ikks(ik) ; ikq = ikqs(ik)
    npw = ngk(ikk) ; npwq = ngk(ikq)
    nnr = w_nnr ; npwxl = npwx
    scale = 1.0_DP / DBLE(nnr)
    !
    islot = dvloc_dev_slot( uact, addnlcc )
    !
    ! ... FFT maps of k+G and k+q+G; k+q+G
    !
    DO ig = 1, npw
       nl_h(ig,1) = dffts%nl(igk_k(ig,ikk))
    ENDDO
    nl_h(npw+1:npwx,1) = 1
    DO ig = 1, npwq
       nl_h(ig,2) = dffts%nl(igk_k(ig,ikq))
       kqg_h(:,ig) = xk(:,ikq) + g(:,igk_k(ig,ikq))
    ENDDO
    nl_h(npwq+1:npwx,2) = 1
    kqg_h(:,npwq+1:npwx) = 0.0_DP
    nlk_d = nl_h
    kqg_d = kqg_h
    !
    ! ... local potential: psi(r) * dV_loc(r), back on the k+q sphere
    !
    IF ( .NOT. own_psi ) THEN
       !$acc update device(evc)
    ENDIF
    DO j0 = 0, nb-1, nfb
       nbl = MIN( nfb, nb-j0 )
       ip = get_plan( nbl )
       IF ( own_psi ) THEN
          !$cuf kernel do(2)
          DO j = 1, nbl
             DO ig = 1, npw
                psig_d(nlk_d(ig,1) + (j-1)*nnr) = psi_d(ig + (j0+j-1)*ldpl)
             ENDDO
          ENDDO
       ELSE
          !$acc parallel loop collapse(2) present(evc)
          DO j = 1, nbl
             DO ig = 1, npw
                psig_d(nlk_d(ig,1) + (j-1)*nnr) = evc(ig,j0+j)
             ENDDO
          ENDDO
       ENDIF
       istat = cufftExecZ2Z( plan(ip), psig_d, psic_d, CUFFT_INVERSE )
       !$cuf kernel do(2)
       DO j = 1, nbl
          DO ig = 1, npw
             psig_d(nlk_d(ig,1) + (j-1)*nnr) = (0.0_DP, 0.0_DP)
          ENDDO
       ENDDO
       !$cuf kernel do(2)
       DO j = 1, nbl
          DO ir = 1, nnr
             psic_d(ir + (j-1)*nnr) = psic_d(ir + (j-1)*nnr) * dvl_d(ir,islot)
          ENDDO
       ENDDO
       istat = cufftExecZ2Z( plan(ip), psic_d, psic_d, CUFFT_FORWARD )
       !$cuf kernel do(2)
       DO j = 1, nbl
          DO ig = 1, npwxl
             IF ( ig <= npwq ) THEN
                out_d(ig,j0+j) = psic_d(nlk_d(ig,2) + (j-1)*nnr) * scale
             ELSE
                out_d(ig,j0+j) = (0.0_DP, 0.0_DP)
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    ! ... nonlocal potential: beta(k+q+G) ps1 + (k+q+G)_alpha beta(k+q+G) ps2_alpha
    !
    IF ( nkb > 0 ) THEN
       CALL start_clock( 'dvq_coef' )
       CALL dvqpsi_nl_coef( ik, uact, becp1, alphap, ps_h )
       ps_d = ps_h
       CALL stop_clock( 'dvq_coef' )
       IF ( own_beta ) THEN
          istat = cublasZgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwq, nb, nkb, c_one, beta_d, ldb, &
                               ps_d, nkb, c_one, out_d, ldo )
       ELSE
          !$acc host_data use_device(vkb)
          istat = cublasZgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwq, nb, nkb, c_one, vkb, npwx, &
                               ps_d, nkb, c_one, out_d, ldo )
          !$acc end host_data
       ENDIF
       IF ( istat /= 0 ) CALL errore( 'dvqpsi_us_dev', 'cublasZgemm failed', ABS(istat) )
       DO j0 = 0, nb-1, nbc
          nbl = MIN( nbc, nb-j0 )
          IF ( own_beta ) THEN
             istat = cublasZgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwq, 3*nbl, nkb, c_one, beta_d, &
                                  ldb, ps_d(1,nbnd+3*j0+1), nkb, c_zero, tmp_d, npwx )
          ELSE
             !$acc host_data use_device(vkb)
             istat = cublasZgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwq, 3*nbl, nkb, c_one, vkb, &
                                  npwx, ps_d(1,nbnd+3*j0+1), nkb, c_zero, tmp_d, npwx )
             !$acc end host_data
          ENDIF
          IF ( istat /= 0 ) CALL errore( 'dvqpsi_us_dev', 'cublasZgemm failed', ABS(istat) )
          !$cuf kernel do(2)
          DO j = 1, nbl
             DO ig = 1, npwq
                out_d(ig,j0+j) = out_d(ig,j0+j) + kqg_d(1,ig) * tmp_d(ig,3*(j-1)+1) + &
                                 kqg_d(2,ig) * tmp_d(ig,3*(j-1)+2) + kqg_d(3,ig) * tmp_d(ig,3*(j-1)+3)
             ENDDO
          ENDDO
       ENDDO
    ENDIF
    !
  END SUBROUTINE dvqpsi_us_dev
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dvqpsi_us_host( ik, uact, addnlcc, becp1, alphap, dvpsi_h )
    !-----------------------------------------------------------------------
    !! dvqpsi_us_dev for all bands, result copied to the host.
    !
    USE ions_base, ONLY : nat
    USE wvfct,     ONLY : npwx, nbnd
    USE qpoint,    ONLY : nksq
    USE becmod,    ONLY : bec_type
    !
    INTEGER, INTENT(IN) :: ik
    COMPLEX(DP), INTENT(IN) :: uact(3*nat)
    LOGICAL, INTENT(IN) :: addnlcc
    TYPE(bec_type), INTENT(IN) :: becp1(nksq), alphap(3,nksq)
    COMPLEX(DP), INTENT(OUT) :: dvpsi_h(npwx,nbnd)
    !
    CALL work_setup( )
    CALL dvqpsi_us_dev( ik, uact, addnlcc, becp1, alphap, nbnd, dvq_d, npwx )
    CALL start_clock( 'dvq_d2h' )
    dvpsi_h = dvq_d
    CALL stop_clock( 'dvq_d2h' )
    !
  END SUBROUTINE dvqpsi_us_host
  !
#else
  !
  SUBROUTINE dvbare_dev_reset( )
  END SUBROUTINE dvbare_dev_reset
  !
  SUBROUTINE dvbare_dev_free( )
  END SUBROUTINE dvbare_dev_free
  !
#endif
  !
END MODULE dvbare_dev

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
  PUBLIC :: dvloc_dev_slot, dvl_d, &
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

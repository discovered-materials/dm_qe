!
! Copyright (C) 2026 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
MODULE usaug_dev
  !-----------------------------------------------------------------------
  !! Ultrasoft / PAW augmentation terms of DFPT on the device.
  !!
  !! All these terms are sums over the G-vectors of the dense grid of the
  !! augmentation functions \(Q_{ij}(q+G)\) times structure factors times
  !! a function of G (a response potential, a response density, the local
  !! potential ...). The stock routines compute \(Q_{ij}\) again for every
  !! (i,j) at every call and issue one small kernel per (i, j, atom).
  !! Here \(Q_{ij}(q+G)\) of all (ij, type) is tabulated once per q on the
  !! device (qg_d) and every term becomes one matrix product per atomic
  !! type:
  !!
  !!   newdq:          int3(ij;na,is,ip) = omega sum_G conj(Q_ij) [conj(S_na) dV(G,is,ip)]
  !!   lr_addusddens:  drho(G,is,ip)    += 2 sum_na S_na sum_ij Q_ij dbecsum(ij,na,is,ip)
  !!   dvanqq:         int1, int2, int4, int5 = sum_G Q_ij w_k, the columns w_k being
  !!                   V_eff(G) or V_loc(q+G) times structure factors times the
  !!                   components of (q+G) and their products
  !!   addusddens_pulay: as lr_addusddens, with the term of the displaced projectors
  !!
  !! with the Fourier transforms of dV / drho batched on the device.
  !!
  !! When the table does not fit in a fraction of the free device memory it
  !! is not kept: blocks of columns are computed when they are needed (one
  !! kernel per block). The columns of the products (atom, spin,
  !! perturbation) and the FFTs are processed in chunks sized on the free
  !! memory as well.
  !!
  !! The table belongs to one q: it is rebuilt when q, the G-vectors, the
  !! cell or the atomic positions differ from those it was built for.
  !!
  !! Served: collinear (nspin = 1, 2), no DFT+U, a dense FFT grid that is not
  !! distributed over MPI ranks. Otherwise, and with QE_PH_DEV=0 in the
  !! environment, the callers keep to their stock code (QE_USAUG=0 does the
  !! same for this module only, leaving the Sternheimer engine on).
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
  PUBLIC :: usaug_on, usaug_check, usaug_invalidate, &
            usaug_interp_on
  !
  LOGICAL :: env_read = .FALSE.
  !! the environment has been read
  LOGICAL :: env_on = .TRUE.
  !! neither QE_PH_DEV nor QE_USAUG is 0
  LOGICAL :: env_check = .FALSE.
  !! QE_USAUG_CHECK=1: the callers run the stock code as well and print the
  !! largest difference between the two results (the stock result is kept)
  INTEGER :: dbg_chunk = 0
  !! QE_USAUG_CHUNK > 0: no table, and at most this number of columns per
  !! block / chunk / FFT batch (to test the paths of large systems)
  !
#if defined(__CUDA)
  !
  COMPLEX(DP), PARAMETER :: c_one = (1.0_DP, 0.0_DP), c_zero = (0.0_DP, 0.0_DP)
  !
  ! ... what the table was built for
  !
  LOGICAL :: tab_ready = .FALSE.
  INTEGER :: ngml = 0, natl = 0, nqxl = 0, nnrp = 0, nr1 = 0, nr2 = 0, nr3 = 0
  REAL(DP) :: xq_tab(3) = 0.0_DP, omega_tab = 0.0_DP
  REAL(DP), ALLOCATABLE :: tau_tab(:,:)
  !
  ! ... entries of the table: e = (type, ij) with ij = (ih, jh >= ih)
  !
  INTEGER :: ne = 0
  !! number of entries
  INTEGER :: mxlm = 0
  !! maximum number of LM in an entry
  INTEGER :: nbcol = 0
  !! columns of qg_d when the table is not kept
  INTEGER :: blk_e1 = 0, blk_nc = 0
  !! entries that are in qg_d when the table is not kept
  LOGICAL :: cached = .FALSE.
  !! qg_d holds all the entries
  INTEGER, ALLOCATABLE :: ent0(:), nij(:)
  !! for each type: entries before the first one of the type, number of entries
  INTEGER, ALLOCATABLE :: tat0(:), tnat(:), atl(:)
  !! atoms sorted by type: atoms before the first of the type, number of atoms, list
  INTEGER, ALLOCATABLE, DEVICE :: e_nlm_d(:), e_ijv_d(:), e_nt_d(:), e_lp_d(:,:), e_l_d(:,:), &
                                  e_ind_d(:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: e_sig_d(:,:)
  INTEGER, ALLOCATABLE, DEVICE :: atl_d(:)
  !
  ! ... data of the G-vectors
  !
  REAL(DP), ALLOCATABLE, DEVICE :: ylm_d(:,:), qmod_d(:), qrad_d(:,:,:,:)
  !! Y_lm(q+G), |q+G|, radial table
  INTEGER, ALLOCATABLE, DEVICE :: mill_d(:,:), nl_d(:)
  REAL(DP), ALLOCATABLE, DEVICE :: g_d(:,:)
  !! G-vectors
  COMPLEX(DP), ALLOCATABLE, DEVICE :: eig1_d(:,:), eig2_d(:,:), eig3_d(:,:), eigq_d(:)
  !! structure factor: e^{-i G tau} factorized, e^{-i q tau}
  COMPLEX(DP), ALLOCATABLE, DEVICE :: qg_d(:,:)
  !! Q_ij(q+G): (ngm, ne), or (ngm, nbcol) when the table is not kept
  !
  ! ... cuFFT plans of the dense grid (one per batch size) and cuBLAS handle
  !
  INTEGER, PARAMETER :: nplan_max = 8
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
  SUBROUTINE read_env( )
    !-----------------------------------------------------------------------
    !! QE_PH_DEV=0 selects the stock code; read once.
    CHARACTER(LEN=16) :: val
    INTEGER :: ln, ierr, ios
    CALL get_environment_variable( 'QE_PH_DEV', val, ln, ierr )
    env_on = .NOT. ( ierr == 0 .AND. ln > 0 .AND. val(1:1) == '0' )
    CALL get_environment_variable( 'QE_USAUG', val, ln, ierr )
    IF ( ierr == 0 .AND. ln > 0 .AND. val(1:1) == '0' ) env_on = .FALSE.
    CALL get_environment_variable( 'QE_USAUG_CHUNK', val, ln, ierr )
    dbg_chunk = 0
    IF ( ierr == 0 .AND. ln > 0 ) THEN
       READ( val, *, IOSTAT=ios ) dbg_chunk
       IF ( ios /= 0 ) dbg_chunk = 0
    ENDIF
    CALL get_environment_variable( 'QE_USAUG_CHECK', val, ln, ierr )
    env_check = ( ierr == 0 .AND. ln > 0 .AND. val(1:1) == '1' )
    env_read = .TRUE.
  END SUBROUTINE read_env
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION usaug_check( what, ndim, dev, ref )
    !-----------------------------------------------------------------------
    !! Self-test (QE_USAUG_CHECK=1). Without arguments: is the self-test
    !! requested? With arguments: print the largest difference between the
    !! result of the device code (dev) and that of the stock code (ref).
    USE io_global, ONLY : stdout
    CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: what
    INTEGER, INTENT(IN), OPTIONAL :: ndim
    COMPLEX(DP), INTENT(IN), OPTIONAL :: dev(*), ref(*)
    REAL(DP) :: dmax, rmax
    INTEGER :: i
    !
    IF ( .NOT. env_read ) CALL read_env( )
    usaug_check = env_check
    IF ( .NOT. env_check .OR. .NOT. PRESENT(what) ) RETURN
    dmax = 0.0_DP ; rmax = 0.0_DP
    DO i = 1, ndim
       dmax = MAX( dmax, ABS( dev(i) - ref(i) ) )
       rmax = MAX( rmax, ABS( ref(i) ) )
    ENDDO
    WRITE( stdout, '(5x,"usaug check ",a,": max |dev-stock| =",es10.3,"  max |stock| =",es10.3)' ) &
         what, dmax, rmax
    !
  END FUNCTION usaug_check
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION usaug_on( )
    !-----------------------------------------------------------------------
    !! The device code of this module can be used in this run.
#if defined(__CUDA)
    USE control_flags,    ONLY : use_gpu, gamma_only
    USE noncollin_module, ONLY : noncolin
    USE ldaU,             ONLY : lda_plus_u
    USE uspp,             ONLY : okvan
    USE fft_base,         ONLY : dfftp
    USE mp_bands,         ONLY : nproc_bgrp
    !
    IF ( .NOT. env_read ) CALL read_env( )
    usaug_on = env_on .AND. use_gpu .AND. okvan .AND. &
         .NOT. ( noncolin .OR. lda_plus_u .OR. gamma_only ) .AND. nproc_bgrp == 1 .AND. &
         dfftp%nnr == dfftp%nr1*dfftp%nr2*dfftp%nr3 .AND. dfftp%nr1 == dfftp%nr1x .AND. &
         dfftp%nr2 == dfftp%nr2x .AND. dfftp%nr3 == dfftp%nr3x
#else
    usaug_on = .FALSE.
#endif
  END FUNCTION usaug_on
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION usaug_interp_on( )
    !-----------------------------------------------------------------------
    !! usaug_interpolate can be used: device code on, smooth grid different
    !! from the dense one and not distributed over MPI ranks.
#if defined(__CUDA)
    USE fft_base,  ONLY : dffts
    USE gvecs,     ONLY : doublegrid
    !
    usaug_interp_on = .FALSE.
    IF ( .NOT. doublegrid ) RETURN
    usaug_interp_on = usaug_on( ) .AND. &
         dffts%nnr == dffts%nr1*dffts%nr2*dffts%nr3 .AND. dffts%nr1 == dffts%nr1x .AND. &
         dffts%nr2 == dffts%nr2x .AND. dffts%nr3 == dffts%nr3x
#else
    usaug_interp_on = .FALSE.
#endif
  END FUNCTION usaug_interp_on
  !
#if defined(__CUDA)
  !
  !-----------------------------------------------------------------------
  SUBROUTINE usaug_invalidate( )
    !-----------------------------------------------------------------------
    !! Release the table and all the device data of the module.
    INTEGER :: i, istat
    !
    tab_ready = .FALSE.
    cached = .FALSE.
    blk_e1 = 0 ; blk_nc = 0
    IF ( ALLOCATED(tau_tab) ) DEALLOCATE( tau_tab )
    IF ( ALLOCATED(ent0) ) DEALLOCATE( ent0, nij, tat0, tnat, atl )
    IF ( ALLOCATED(e_nlm_d) ) DEALLOCATE( e_nlm_d, e_ijv_d, e_nt_d, e_lp_d, e_l_d, e_ind_d, e_sig_d )
    IF ( ALLOCATED(atl_d) ) DEALLOCATE( atl_d )
    IF ( ALLOCATED(ylm_d) ) DEALLOCATE( ylm_d )
    IF ( ALLOCATED(qmod_d) ) DEALLOCATE( qmod_d )
    IF ( ALLOCATED(qrad_d) ) DEALLOCATE( qrad_d )
    IF ( ALLOCATED(mill_d) ) DEALLOCATE( mill_d, nl_d, g_d )
    IF ( ALLOCATED(eig1_d) ) DEALLOCATE( eig1_d, eig2_d, eig3_d, eigq_d )
    IF ( ALLOCATED(qg_d) ) DEALLOCATE( qg_d )
    DO i = 1, nplan_max
       IF ( plan_nb(i) /= 0 ) istat = cufftDestroy( plan(i) )
    ENDDO
    plan = 0 ; plan_nb = 0 ; plan_age = 0 ; plan_dim = 0 ; plan_clock = 0
    !
  END SUBROUTINE usaug_invalidate
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
    !! time: what fits in a quarter of the free device memory.
    REAL(DP), INTENT(IN) :: bytes_per_col
    INTEGER, INTENT(IN) :: nwant
    REAL(DP) :: x
    x = MIN( REAL(nwant,DP), 0.25_DP * free_bytes( ) / bytes_per_col )
    ncol_fit = MAX( 1, INT(x) )
    IF ( dbg_chunk > 0 ) ncol_fit = MIN( ncol_fit, dbg_chunk )
  END FUNCTION ncol_fit
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION get_plan( nb, m1, m2, m3 )
    !-----------------------------------------------------------------------
    !! cuFFT plan of nb transforms on the dense grid, or on the grid
    !! m1 x m2 x m3 when the dimensions are given
    INTEGER, INTENT(IN) :: nb
    INTEGER, INTENT(IN), OPTIONAL :: m1, m2, m3
    INTEGER :: i, k, istat, fdim(3), d(3), nn
    !
    d = (/ nr1, nr2, nr3 /)
    IF ( PRESENT(m1) ) d = (/ m1, m2, m3 /)
    plan_clock = plan_clock + 1
    DO i = 1, nplan_max
       IF ( plan_nb(i) == nb .AND. plan_dim(1,i) == d(1) .AND. plan_dim(2,i) == d(2) .AND. &
            plan_dim(3,i) == d(3) ) THEN
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
    fdim = (/ d(3), d(2), d(1) /)
    nn = d(1) * d(2) * d(3)
    istat = cufftPlanMany( plan(k), 3, fdim, fdim, 1, nn, fdim, 1, nn, CUFFT_Z2Z, nb )
    IF ( istat /= 0 ) CALL errore( 'usaug_dev', 'cufftPlanMany failed', ABS(istat) )
    plan_nb(k) = nb
    plan_dim(:,k) = d(:)
    plan_age(k) = plan_clock
    get_plan = plan(k)
    !
  END FUNCTION get_plan
  !
  !-----------------------------------------------------------------------
  SUBROUTINE fill_qg( n, yl_d, qm_d, e1, e2, dst_d )
    !-----------------------------------------------------------------------
    !! dst_d(:,e-e1+1) = Q of entry e, e = e1 ... e2, for the vectors whose
    !! moduli are in qm_d and spherical harmonics in yl_d (qvan2).
    USE qrad_mod,  ONLY : dq
    !
    INTEGER, INTENT(IN) :: n, e1, e2
    REAL(DP), DEVICE, INTENT(IN) :: yl_d(n,*), qm_d(n)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: dst_d(n,*)
    !
    REAL(DP), PARAMETER :: sixth = 1.0_DP / 6.0_DP
    INTEGER :: ie, ig, lm, lp, l, i0, ijv, nt
    REAL(DP) :: dqi, qm, px, ux, vx, wx, uvx, pwx, work, re, im, om
    !
    dqi = 1.0_DP / dq
    om = omega_tab
    !$cuf kernel do(2)
    DO ie = e1, e2
       DO ig = 1, n
          qm = qm_d(ig) * dqi
          px = qm - INT(qm)
          ux = 1.0_DP - px
          vx = 2.0_DP - px
          wx = 3.0_DP - px
          i0 = INT(qm) + 1
          uvx = ux * vx * sixth
          pwx = px * wx * 0.5_DP
          ijv = e_ijv_d(ie)
          nt = e_nt_d(ie)
          re = 0.0_DP
          im = 0.0_DP
          DO lm = 1, e_nlm_d(ie)
             lp = e_lp_d(lm,ie)
             l = e_l_d(lm,ie)
             work = ( qrad_d(i0,ijv,l,nt) * uvx * wx + &
                      qrad_d(i0+1,ijv,l,nt) * pwx * vx - &
                      qrad_d(i0+2,ijv,l,nt) * pwx * ux + &
                      qrad_d(i0+3,ijv,l,nt) * px * uvx ) / om
             IF ( e_ind_d(lm,ie) == 1 ) THEN
                re = re + e_sig_d(lm,ie) * yl_d(ig,lp) * work
             ELSE
                im = im + e_sig_d(lm,ie) * yl_d(ig,lp) * work
             ENDIF
          ENDDO
          dst_d(ig,ie-e1+1) = CMPLX( re, im, KIND=DP )
       ENDDO
    ENDDO
    !
  END SUBROUTINE fill_qg
  !
  !-----------------------------------------------------------------------
  SUBROUTINE prepare_q( )
    !-----------------------------------------------------------------------
    !! Make the data of the module those of the current q: nothing to do
    !! when they already are.
    !
    USE cell_base,   ONLY : omega, tpiba
    USE gvect,       ONLY : ngm, g, gg, mill, eigts1, eigts2, eigts3
    USE fft_base,    ONLY : dfftp
    USE ions_base,   ONLY : nat, ityp, ntyp => nsp, tau
    USE uspp,        ONLY : lpl, lpx, ap, indv, nhtolm
    USE uspp_param,  ONLY : upf, lmaxq, nh
    USE qrad_mod,    ONLY : tab_qrad
    USE qpoint,      ONLY : xq, eigqts
    USE control_lr,  ONLY : lgamma
    !
    INTEGER :: nt, na, ih, jh, ie, nb, mb, ivl, jvl, lm, lp, l, ind, ig, istat
    REAL(DP) :: sig
    INTEGER, ALLOCATABLE :: e_nlm(:), e_ijv(:), e_nt(:), e_lp(:,:), e_l(:,:), e_ind(:,:)
    REAL(DP), ALLOCATABLE :: e_sig(:,:), qmod(:), qpg(:,:), ylm(:,:)
    !
    IF ( tab_ready ) THEN
       IF ( ngml == ngm .AND. natl == nat .AND. omega_tab == omega .AND. &
            nqxl == SIZE(tab_qrad,1) .AND. nnrp == dfftp%nnr .AND. &
            xq_tab(1) == xq(1) .AND. xq_tab(2) == xq(2) .AND. xq_tab(3) == xq(3) ) THEN
          IF ( ALL( tau_tab(:,:) == tau(:,1:nat) ) ) RETURN
       ENDIF
    ENDIF
    !
    CALL start_clock( 'usaug_qtab' )
    CALL usaug_invalidate( )
    IF ( .NOT. blas_ready ) THEN
       istat = cublasCreate( hblas )
       IF ( istat /= 0 ) CALL errore( 'usaug_dev', 'cublasCreate failed', ABS(istat) )
       blas_ready = .TRUE.
    ENDIF
    !
    ngml = ngm ; natl = nat ; omega_tab = omega ; xq_tab(:) = xq(:)
    nqxl = SIZE(tab_qrad,1)
    nnrp = dfftp%nnr ; nr1 = dfftp%nr1 ; nr2 = dfftp%nr2 ; nr3 = dfftp%nr3
    ALLOCATE( tau_tab(3,nat) )
    tau_tab(:,:) = tau(:,1:nat)
    !
    ! ... entries and atoms by type
    !
    ALLOCATE( ent0(ntyp), nij(ntyp), tat0(ntyp), tnat(ntyp), atl(nat) )
    ne = 0 ; ie = 0
    DO nt = 1, ntyp
       ent0(nt) = ne
       nij(nt) = 0
       IF ( upf(nt)%tvanp ) nij(nt) = ( nh(nt) * (nh(nt)+1) ) / 2
       ne = ne + nij(nt)
       tat0(nt) = ie
       DO na = 1, nat
          IF ( ityp(na) == nt ) THEN
             ie = ie + 1
             atl(ie) = na
          ENDIF
       ENDDO
       tnat(nt) = ie - tat0(nt)
    ENDDO
    mxlm = SIZE(lpl,3)
    ALLOCATE( e_nlm(ne), e_ijv(ne), e_nt(ne), e_lp(mxlm,ne), e_l(mxlm,ne), e_ind(mxlm,ne), &
              e_sig(mxlm,ne) )
    e_lp = 1 ; e_l = 1 ; e_ind = 1 ; e_sig = 0.0_DP
    ie = 0
    DO nt = 1, ntyp
       IF ( .NOT. upf(nt)%tvanp ) CYCLE
       DO ih = 1, nh(nt)
          DO jh = ih, nh(nt)
             ie = ie + 1
             nb = indv(ih,nt)
             mb = indv(jh,nt)
             IF ( nb >= mb ) THEN
                e_ijv(ie) = nb * (nb - 1) / 2 + mb
             ELSE
                e_ijv(ie) = mb * (mb - 1) / 2 + nb
             ENDIF
             e_nt(ie) = nt
             ivl = nhtolm(ih,nt)
             jvl = nhtolm(jh,nt)
             e_nlm(ie) = lpx(ivl,jvl)
             DO lm = 1, lpx(ivl,jvl)
                lp = lpl(ivl,jvl,lm)
                IF ( lp < 1 .OR. lp > 49 ) CALL errore( 'usaug_dev', 'lp wrong', MAX(lp,1) )
                !
                ! ... angular momentum (plus one), sign and real/imaginary part of (-i)^L
                !
                IF ( lp == 1 ) THEN
                   l = 1 ; sig = 1.0_DP ; ind = 1
                ELSEIF ( lp <= 4 ) THEN
                   l = 2 ; sig = -1.0_DP ; ind = 2
                ELSEIF ( lp <= 9 ) THEN
                   l = 3 ; sig = -1.0_DP ; ind = 1
                ELSEIF ( lp <= 16 ) THEN
                   l = 4 ; sig = 1.0_DP ; ind = 2
                ELSEIF ( lp <= 25 ) THEN
                   l = 5 ; sig = 1.0_DP ; ind = 1
                ELSEIF ( lp <= 36 ) THEN
                   l = 6 ; sig = -1.0_DP ; ind = 2
                ELSE
                   l = 7 ; sig = -1.0_DP ; ind = 1
                ENDIF
                e_lp(lm,ie) = lp
                e_l(lm,ie) = l
                e_ind(lm,ie) = ind
                e_sig(lm,ie) = sig * ap(lp,ivl,jvl)
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    ALLOCATE( e_nlm_d(ne), e_ijv_d(ne), e_nt_d(ne), e_lp_d(mxlm,ne), e_l_d(mxlm,ne), &
              e_ind_d(mxlm,ne), e_sig_d(mxlm,ne), atl_d(nat) )
    e_nlm_d = e_nlm ; e_ijv_d = e_ijv ; e_nt_d = e_nt ; e_lp_d = e_lp ; e_l_d = e_l
    e_ind_d = e_ind ; e_sig_d = e_sig ; atl_d = atl
    DEALLOCATE( e_nlm, e_ijv, e_nt, e_lp, e_l, e_ind, e_sig )
    !
    ! ... G-vectors: Miller indices, FFT index, structure factors
    !
    ALLOCATE( mill_d(3,ngm), nl_d(ngm), g_d(3,ngm) )
    mill_d = mill
    nl_d = dfftp%nl
    g_d = g
    ALLOCATE( eig1_d(-nr1:nr1,nat), eig2_d(-nr2:nr2,nat), eig3_d(-nr3:nr3,nat), eigq_d(nat) )
    eig1_d = eigts1 ; eig2_d = eigts2 ; eig3_d = eigts3 ; eigq_d = eigqts
    !
    ! ... |q+G| and Y_lm(q+G); radial table
    !
    ALLOCATE( qmod(ngm), ylm(ngm,lmaxq*lmaxq) )
    IF ( lgamma ) THEN
       CALL ylmr2( lmaxq*lmaxq, ngm, g, gg, ylm )
       DO ig = 1, ngm
          qmod(ig) = SQRT( gg(ig) ) * tpiba
       ENDDO
    ELSE
       ALLOCATE( qpg(3,ngm) )
       CALL setqmod( ngm, xq, g, qmod, qpg )
       CALL ylmr2( lmaxq*lmaxq, ngm, qpg, qmod, ylm )
       DEALLOCATE( qpg )
       DO ig = 1, ngm
          qmod(ig) = SQRT( qmod(ig) ) * tpiba
       ENDDO
    ENDIF
    ALLOCATE( ylm_d(ngm,lmaxq*lmaxq), qmod_d(ngm) )
    ylm_d = ylm ; qmod_d = qmod
    DEALLOCATE( qmod, ylm )
    ALLOCATE( qrad_d(SIZE(tab_qrad,1),SIZE(tab_qrad,2),SIZE(tab_qrad,3),SIZE(tab_qrad,4)) )
    qrad_d = tab_qrad
    !
    ! ... the table, when it fits
    !
    cached = dbg_chunk <= 0 .AND. 16.0_DP * REAL(ngm,DP) * ne <= 0.25_DP * free_bytes( )
    IF ( cached ) THEN
       ALLOCATE( qg_d(ngm,MAX(ne,1)) )
       IF ( ne > 0 ) CALL fill_qg( ngm, ylm_d, qmod_d, 1, ne, qg_d )
       DEALLOCATE( ylm_d, qmod_d )
    ELSE
       nbcol = ncol_fit( 16.0_DP * REAL(ngm,DP), MAX( 1, MAXVAL(nij) ) )
       ALLOCATE( qg_d(ngm,nbcol) )
    ENDIF
    blk_e1 = 0 ; blk_nc = 0
    tab_ready = .TRUE.
    CALL stop_clock( 'usaug_qtab' )
    !
  END SUBROUTINE prepare_q
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION qblock( nt )
    !-----------------------------------------------------------------------
    !! number of entries of type nt that are available at a time
    INTEGER, INTENT(IN) :: nt
    qblock = MAX( 1, nij(nt) )
    IF ( .NOT. cached ) qblock = MIN( qblock, nbcol )
  END FUNCTION qblock
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION qcol( nt, ij0, nc )
    !-----------------------------------------------------------------------
    !! column of qg_d where the entries ij0+1 ... ij0+nc of type nt start
    INTEGER, INTENT(IN) :: nt, ij0, nc
    INTEGER :: e1
    !
    e1 = ent0(nt) + ij0 + 1
    IF ( cached ) THEN
       qcol = e1
    ELSE
       IF ( nc > nbcol ) CALL errore( 'usaug_dev', 'block too large', nc )
       IF ( blk_e1 /= e1 .OR. blk_nc < nc ) THEN
          CALL fill_qg( ngml, ylm_d, qmod_d, e1, e1+nc-1, qg_d )
          blk_e1 = e1 ; blk_nc = nc
       ENDIF
       qcol = 1
    ENDIF
    !
  END FUNCTION qcol
  !
  !-----------------------------------------------------------------------
  SUBROUTINE fft_to_g( f, n, aux_d )
    !-----------------------------------------------------------------------
    !! aux_d(:,j) = Fourier components on the G-vectors of f(:,j), j = 1 ... n
    !! (fwfft 'Rho' of n functions of the dense grid)
    INTEGER, INTENT(IN) :: n
    COMPLEX(DP), INTENT(IN) :: f(nnrp,n)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: aux_d(ngml,n)
    !
    COMPLEX(DP), ALLOCATABLE, DEVICE :: fr_d(:,:)
    INTEGER :: nfb, j0, nb, ip, j, ig, istat, n1
    REAL(DP) :: scale
    !
    scale = 1.0_DP / DBLE(nnrp)
    n1 = ngml
    nfb = ncol_fit( 32.0_DP * REAL(nnrp,DP), n )
    ALLOCATE( fr_d(nnrp,nfb) )
    DO j0 = 0, n-1, nfb
       nb = MIN( nfb, n-j0 )
       fr_d(:,1:nb) = f(:,j0+1:j0+nb)
       ip = get_plan( nb )
       istat = cufftExecZ2Z( ip, fr_d, fr_d, CUFFT_FORWARD )
       IF ( istat /= 0 ) CALL errore( 'usaug_dev', 'cufftExecZ2Z failed', ABS(istat) )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, n1
             aux_d(ig,j0+j) = fr_d(nl_d(ig),j) * scale
          ENDDO
       ENDDO
    ENDDO
    DEALLOCATE( fr_d )
    !
  END SUBROUTINE fft_to_g
  !
  !-----------------------------------------------------------------------
  SUBROUTINE add_from_g( aux_d, n, f )
    !-----------------------------------------------------------------------
    !! f(:,j) = f(:,j) + the function of the dense grid whose Fourier
    !! components are aux_d(:,j), j = 1 ... n  (invfft 'Rho')
    INTEGER, INTENT(IN) :: n
    COMPLEX(DP), DEVICE, INTENT(IN) :: aux_d(ngml,n)
    COMPLEX(DP), INTENT(INOUT) :: f(nnrp,n)
    !
    COMPLEX(DP), ALLOCATABLE, DEVICE :: fr_d(:,:)
    COMPLEX(DP), ALLOCATABLE :: fr_h(:,:)
    INTEGER :: nfb, j0, nb, ip, j, ig, ir, istat, n1, n2
    !
    n1 = ngml ; n2 = nnrp
    nfb = ncol_fit( 32.0_DP * REAL(nnrp,DP), n )
    ALLOCATE( fr_d(nnrp,nfb), fr_h(nnrp,nfb) )
    DO j0 = 0, n-1, nfb
       nb = MIN( nfb, n-j0 )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ir = 1, n2
             fr_d(ir,j) = (0.0_DP, 0.0_DP)
          ENDDO
       ENDDO
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, n1
             fr_d(nl_d(ig),j) = aux_d(ig,j0+j)
          ENDDO
       ENDDO
       ip = get_plan( nb )
       istat = cufftExecZ2Z( ip, fr_d, fr_d, CUFFT_INVERSE )
       IF ( istat /= 0 ) CALL errore( 'usaug_dev', 'cufftExecZ2Z failed', ABS(istat) )
       fr_h(:,1:nb) = fr_d(:,1:nb)
       !$omp parallel do private(ir)
       DO j = 1, nb
          DO ir = 1, n2
             f(ir,j0+j) = f(ir,j0+j) + fr_h(ir,j)
          ENDDO
       ENDDO
       !$omp end parallel do
    ENDDO
    DEALLOCATE( fr_d, fr_h )
    !
  END SUBROUTINE add_from_g
  !
#else
  !
  ! ... stubs for builds without CUDA: usaug_on is false and these are never called
  !
  SUBROUTINE usaug_invalidate( )
  END SUBROUTINE usaug_invalidate
  !
#endif
  !
END MODULE usaug_dev

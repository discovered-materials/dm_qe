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
  PUBLIC :: usaug_on, usaug_check, usaug_invalidate, usaug_newdq, usaug_addusddens, &
            usaug_dvanqq, usaug_addusddens_pulay, usaug_interp_on
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
  !-----------------------------------------------------------------------
  SUBROUTINE usaug_newdq( dvscf, npe )
    !-----------------------------------------------------------------------
    !! Ultrasoft part of int3 (the integral of the response potential and of
    !! the augmentation functions), see newdq:
    !! int3(ih,jh,na,is,ipert) = omega sum_G conj(Q_ij(q+G) S_na(q+G)) dV(G,is,ipert)
    !
    USE cell_base,        ONLY : omega
    USE fft_base,         ONLY : dfftp
    USE ions_base,        ONLY : ntyp => nsp
    USE noncollin_module, ONLY : nspin_mag
    USE uspp_param,       ONLY : upf, nh
    USE lrus,             ONLY : int3
    !
    INTEGER, INTENT(IN) :: npe
    COMPLEX(DP), INTENT(IN) :: dvscf(dfftp%nnr,nspin_mag,npe)
    !
    COMPLEX(DP), ALLOCATABLE, DEVICE :: aux_d(:,:), b_d(:,:), r_d(:,:)
    COMPLEX(DP), ALLOCATABLE :: r_h(:,:)
    INTEGER :: nisp, nt, natc, a0, nac, ncols, ia0, c, ig, na, isp, ij0, nc, col, istat, &
               ia, is, ipert, ih, jh, ij, n1, nqb, nijt
    COMPLEX(DP) :: zomega
    !
    CALL prepare_q( )
    nisp = nspin_mag * npe
    n1 = ngml
    zomega = CMPLX( omega, 0.0_DP, KIND=DP )
    ALLOCATE( aux_d(ngml,nisp) )
    CALL fft_to_g( dvscf, nisp, aux_d )
    !
    int3(:,:,:,:,:) = (0.0_DP, 0.0_DP)
    DO nt = 1, ntyp
       IF ( .NOT. upf(nt)%tvanp .OR. tnat(nt) == 0 ) CYCLE
       nijt = nij(nt)
       nqb = qblock( nt )
       natc = ncol_fit( 16.0_DP * REAL(ngml,DP) * nisp, tnat(nt) )
       ALLOCATE( b_d(ngml,natc*nisp), r_d(nijt,natc*nisp), r_h(nijt,natc*nisp) )
       DO a0 = 0, tnat(nt)-1, natc
          nac = MIN( natc, tnat(nt)-a0 )
          ncols = nac * nisp
          ia0 = tat0(nt) + a0
          !
          ! ... conj(S_na(q+G)) dV(G,is,ipert): column (ia-1)*nisp + isp
          !
          !$cuf kernel do(2)
          DO c = 1, ncols
             DO ig = 1, n1
                na = atl_d(ia0 + (c-1)/nisp + 1)
                isp = MOD(c-1,nisp) + 1
                b_d(ig,c) = CONJG( eig1_d(mill_d(1,ig),na) * eig2_d(mill_d(2,ig),na) * &
                                   eig3_d(mill_d(3,ig),na) * eigq_d(na) ) * aux_d(ig,isp)
             ENDDO
          ENDDO
          DO ij0 = 0, nijt-1, nqb
             nc = MIN( nqb, nijt-ij0 )
             col = qcol( nt, ij0, nc )
             istat = cublasZgemm( hblas, CUBLAS_OP_C, CUBLAS_OP_N, nc, ncols, ngml, zomega, &
                                  qg_d(1,col), ngml, b_d, ngml, c_zero, r_d(ij0+1,1), nijt )
             IF ( istat /= 0 ) CALL errore( 'usaug_newdq', 'cublasZgemm failed', ABS(istat) )
          ENDDO
          r_h(:,1:ncols) = r_d(:,1:ncols)
          !
          DO ia = 1, nac
             na = atl(ia0+ia)
             DO isp = 1, nisp
                c = (ia-1)*nisp + isp
                is = MOD(isp-1,nspin_mag) + 1
                ipert = (isp-1)/nspin_mag + 1
                ij = 0
                DO ih = 1, nh(nt)
                   DO jh = ih, nh(nt)
                      ij = ij + 1
                      int3(ih,jh,na,is,ipert) = r_h(ij,c)
                      int3(jh,ih,na,is,ipert) = r_h(ij,c)
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       DEALLOCATE( b_d, r_d, r_h )
    ENDDO
    DEALLOCATE( aux_d )
    !
  END SUBROUTINE usaug_newdq
  !
  !-----------------------------------------------------------------------
  SUBROUTINE usaug_addusddens( npert, dbecsum, drhop )
    !-----------------------------------------------------------------------
    !! Augmentation part of the response density, see lr_addusddens:
    !! drho(G,is,ipert) += 2 sum_na S_na(q+G) sum_ij Q_ij(q+G) dbecsum(ij,na,is,ipert)
    !
    USE fft_base,         ONLY : dfftp
    USE ions_base,        ONLY : nat, ntyp => nsp
    USE noncollin_module, ONLY : nspin_mag
    USE uspp_param,       ONLY : upf, nhm
    !
    INTEGER, INTENT(IN) :: npert
    COMPLEX(DP), INTENT(IN) :: dbecsum(nhm*(nhm+1)/2,nat,nspin_mag,npert)
    COMPLEX(DP), INTENT(INOUT) :: drhop(dfftp%nnr,nspin_mag,npert)
    !
    COMPLEX(DP), ALLOCATABLE, DEVICE :: aux_d(:,:), t_d(:,:), d_d(:,:)
    COMPLEX(DP), ALLOCATABLE :: d_h(:,:)
    INTEGER :: nisp, nt, natc, a0, nac, ncols, ia0, c, ig, na, isp, ij0, nc, col, istat, &
               ia, is, ipert, n1, nqb, nijt
    COMPLEX(DP) :: sk, beta
    !
    CALL prepare_q( )
    nisp = nspin_mag * npert
    n1 = ngml
    ALLOCATE( aux_d(ngml,nisp) )
    !$cuf kernel do(2)
    DO isp = 1, nisp
       DO ig = 1, n1
          aux_d(ig,isp) = (0.0_DP, 0.0_DP)
       ENDDO
    ENDDO
    !
    DO nt = 1, ntyp
       IF ( .NOT. upf(nt)%tvanp .OR. tnat(nt) == 0 ) CYCLE
       nijt = nij(nt)
       nqb = qblock( nt )
       natc = ncol_fit( 16.0_DP * REAL(ngml,DP) * nisp, tnat(nt) )
       ALLOCATE( t_d(ngml,natc*nisp), d_d(nijt,natc*nisp), d_h(nijt,natc*nisp) )
       DO a0 = 0, tnat(nt)-1, natc
          nac = MIN( natc, tnat(nt)-a0 )
          ncols = nac * nisp
          ia0 = tat0(nt) + a0
          DO ia = 1, nac
             na = atl(ia0+ia)
             DO isp = 1, nisp
                is = MOD(isp-1,nspin_mag) + 1
                ipert = (isp-1)/nspin_mag + 1
                d_h(1:nijt,(ia-1)*nisp+isp) = dbecsum(1:nijt,na,is,ipert)
             ENDDO
          ENDDO
          d_d(:,1:ncols) = d_h(:,1:ncols)
          !
          ! ... t(G,c) = sum_ij Q_ij(q+G) dbecsum(ij,c)
          !
          beta = c_zero
          DO ij0 = 0, nijt-1, nqb
             nc = MIN( nqb, nijt-ij0 )
             col = qcol( nt, ij0, nc )
             istat = cublasZgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, ngml, ncols, nc, c_one, &
                                  qg_d(1,col), ngml, d_d(ij0+1,1), nijt, beta, t_d, ngml )
             IF ( istat /= 0 ) CALL errore( 'usaug_addusddens', 'cublasZgemm failed', ABS(istat) )
             beta = c_one
          ENDDO
          !$cuf kernel do(1)
          DO ig = 1, n1
             DO ia = 1, nac
                na = atl_d(ia0+ia)
                sk = eig1_d(mill_d(1,ig),na) * eig2_d(mill_d(2,ig),na) * &
                     eig3_d(mill_d(3,ig),na) * eigq_d(na)
                DO isp = 1, nisp
                   aux_d(ig,isp) = aux_d(ig,isp) + 2.0_DP * sk * t_d(ig,(ia-1)*nisp+isp)
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       DEALLOCATE( t_d, d_d, d_h )
    ENDDO
    !
    CALL add_from_g( aux_d, nisp, drhop )
    DEALLOCATE( aux_d )
    !
  END SUBROUTINE usaug_addusddens
  !
  !-----------------------------------------------------------------------
  SUBROUTINE usaug_dvanqq( vlocq, int1, int2, int4, int5 )
    !-----------------------------------------------------------------------
    !! The integrals of the augmentation functions and of their derivatives
    !! with V_eff and V_loc, see dvanqq (collinear case, all atoms):
    !!
    !! z_k(ij;nb,is)  = sum_G Q_ij(G)   conj(V_eff(G,is)) S_nb(G) p_k(G)
    !! z_k(ij;na,nb)  = sum_G Q_ij(q+G) conj(V_loc(q+G) S_na(G)) S_nb(G) p_k(q+G)
    !!
    !! with p_1..3(v) = v, p_4..9(v) = v_1 v_1, v_1 v_2, v_1 v_3, v_2 v_2,
    !! v_2 v_3, v_3 v_3. int1 and int4 are made of the first set, int2 and
    !! int5 of the second.
    !
    USE cell_base,        ONLY : omega, tpiba, tpiba2
    USE ions_base,        ONLY : nat, ityp, ntyp => nsp
    USE gvect,            ONLY : ngm, g, gg
    USE scf,              ONLY : v, vltot
    USE noncollin_module, ONLY : nspin_mag
    USE uspp,             ONLY : ijtoh
    USE uspp_param,       ONLY : upf, lmaxq, nh
    USE qpoint,           ONLY : xq, eigqts
    USE control_lr,       ONLY : lgamma
    !
    REAL(DP), INTENT(IN) :: vlocq(:,:)
    !! local potential at q+G (ngm, ntyp)
    COMPLEX(DP), INTENT(INOUT) :: int1(:,:,:,:,:), int2(:,:,:,:,:), int4(:,:,:,:,:), int5(:,:,:,:,:)
    !
    INTEGER, PARAMETER :: kk(3,3) = RESHAPE( (/ 4, 5, 6, 5, 7, 8, 6, 8, 9 /), (/ 3, 3 /) )
    COMPLEX(DP), ALLOCATABLE :: veff(:,:), z_h(:,:)
    REAL(DP), ALLOCATABLE :: qmod(:), ylm(:,:)
    COMPLEX(DP), ALLOCATABLE, DEVICE :: vg_d(:,:), w_d(:,:), z_d(:,:), q0_d(:,:)
    REAL(DP), ALLOCATABLE, DEVICE :: vloc_d(:,:), yl0_d(:,:), qm0_d(:)
    INTEGER, ALLOCATABLE, DEVICE :: ityp_d(:)
    INTEGER :: nt, is, ir, ig, natc, a0, nac, ncols, ia0, c, k, ia, na, nb, nta, ij0, nc, col, &
               istat, ih, jh, ij, ijh, c0, a, b, n1, nqb, nijt, npair, npc, p0, npr, p, nsp_l, &
               nat_l, tat0_l, e1, q0_e1, q0_nc
    REAL(DP) :: xq1, xq2, xq3, v1, v2, v3, pk
    COMPLEX(DP) :: fact, fact1, ee
    !
    CALL prepare_q( )
    n1 = ngml ; nsp_l = nspin_mag ; nat_l = nat
    xq1 = xq(1) ; xq2 = xq(2) ; xq3 = xq(3)
    fact1 = CMPLX( 0.0_DP, - tpiba * omega, KIND=DP )
    !
    ! ... V_eff(G), V_loc(q+G), type of the atoms
    !
    ALLOCATE( veff(nnrp,nspin_mag), vg_d(ngml,nspin_mag) )
    DO is = 1, nspin_mag
       DO ir = 1, nnrp
          veff(ir,is) = CMPLX( vltot(ir) + v%of_r(ir,is), 0.0_DP, KIND=DP )
       ENDDO
    ENDDO
    CALL fft_to_g( veff, nspin_mag, vg_d )
    DEALLOCATE( veff )
    ALLOCATE( vloc_d(ngml,ntyp), ityp_d(nat) )
    vloc_d = vlocq(1:ngm,1:ntyp)
    ityp_d = ityp(1:nat)
    !
    ! ... int1 and int4 need Q_ij(G): the table when q = 0, otherwise it is
    ! ... computed here a block at a time
    !
    q0_e1 = 0 ; q0_nc = 0
    IF ( .NOT. lgamma ) THEN
       ALLOCATE( qmod(ngm), ylm(ngm,lmaxq*lmaxq) )
       CALL ylmr2( lmaxq*lmaxq, ngm, g, gg, ylm )
       DO ig = 1, ngm
          qmod(ig) = SQRT( gg(ig) ) * tpiba
       ENDDO
       ALLOCATE( yl0_d(ngm,lmaxq*lmaxq), qm0_d(ngm) )
       yl0_d = ylm ; qm0_d = qmod
       DEALLOCATE( qmod, ylm )
    ENDIF
    !
    DO nt = 1, ntyp
       IF ( .NOT. upf(nt)%tvanp .OR. tnat(nt) == 0 ) CYCLE
       nijt = nij(nt)
       tat0_l = tat0(nt)
       !
       ! ... first set: columns ((ia-1)*nspin + is-1)*9 + k
       !
       nqb = qblock( nt )
       IF ( .NOT. lgamma ) THEN
          nqb = ncol_fit( 16.0_DP * REAL(ngml,DP), nijt )
          ALLOCATE( q0_d(ngml,nqb) )
          q0_e1 = 0 ; q0_nc = 0
       ENDIF
       natc = ncol_fit( 16.0_DP * REAL(ngml,DP) * 9 * nspin_mag, tnat(nt) )
       ALLOCATE( w_d(ngml,natc*nspin_mag*9), z_d(nijt,natc*nspin_mag*9), z_h(nijt,natc*nspin_mag*9) )
       DO a0 = 0, tnat(nt)-1, natc
          nac = MIN( natc, tnat(nt)-a0 )
          ncols = nac * nspin_mag * 9
          ia0 = tat0(nt) + a0
          !$cuf kernel do(2)
          DO c = 1, ncols
             DO ig = 1, n1
                k = MOD(c-1,9) + 1
                is = MOD((c-1)/9,nsp_l) + 1
                nb = atl_d(ia0 + (c-1)/(9*nsp_l) + 1)
                v1 = g_d(1,ig) ; v2 = g_d(2,ig) ; v3 = g_d(3,ig)
                IF ( k == 1 ) THEN
                   pk = v1
                ELSEIF ( k == 2 ) THEN
                   pk = v2
                ELSEIF ( k == 3 ) THEN
                   pk = v3
                ELSEIF ( k == 4 ) THEN
                   pk = v1 * v1
                ELSEIF ( k == 5 ) THEN
                   pk = v1 * v2
                ELSEIF ( k == 6 ) THEN
                   pk = v1 * v3
                ELSEIF ( k == 7 ) THEN
                   pk = v2 * v2
                ELSEIF ( k == 8 ) THEN
                   pk = v2 * v3
                ELSE
                   pk = v3 * v3
                ENDIF
                w_d(ig,c) = CONJG( vg_d(ig,is) ) * pk * eig1_d(mill_d(1,ig),nb) * &
                            eig2_d(mill_d(2,ig),nb) * eig3_d(mill_d(3,ig),nb)
             ENDDO
          ENDDO
          DO ij0 = 0, nijt-1, nqb
             nc = MIN( nqb, nijt-ij0 )
             IF ( lgamma ) THEN
                col = qcol( nt, ij0, nc )
                istat = cublasZgemm( hblas, CUBLAS_OP_T, CUBLAS_OP_N, nc, ncols, ngml, c_one, &
                                     qg_d(1,col), ngml, w_d, ngml, c_zero, z_d(ij0+1,1), nijt )
             ELSE
                e1 = ent0(nt) + ij0 + 1
                IF ( q0_e1 /= e1 .OR. q0_nc < nc ) THEN
                   CALL fill_qg( ngml, yl0_d, qm0_d, e1, e1+nc-1, q0_d )
                   q0_e1 = e1 ; q0_nc = nc
                ENDIF
                istat = cublasZgemm( hblas, CUBLAS_OP_T, CUBLAS_OP_N, nc, ncols, ngml, c_one, &
                                     q0_d, ngml, w_d, ngml, c_zero, z_d(ij0+1,1), nijt )
             ENDIF
             IF ( istat /= 0 ) CALL errore( 'usaug_dvanqq', 'cublasZgemm failed', ABS(istat) )
          ENDDO
          z_h(:,1:ncols) = z_d(:,1:ncols)
          !
          DO ia = 1, nac
             nb = atl(ia0+ia)
             DO is = 1, nspin_mag
                c0 = ( (ia-1)*nspin_mag + is-1 ) * 9
                ij = 0
                DO ih = 1, nh(nt)
                   DO jh = ih, nh(nt)
                      ij = ij + 1
                      ijh = ijtoh(ih,jh,nt)
                      DO k = 1, 3
                         int1(ih,jh,k,nb,is) = - fact1 * CONJG( z_h(ij,c0+k) )
                         int1(jh,ih,k,nb,is) = int1(ih,jh,k,nb,is)
                      ENDDO
                      DO b = 1, 3
                         DO a = 1, 3
                            int4(ijh,a,b,nb,is) = - tpiba2 * omega * z_h(ij,c0+kk(a,b))
                         ENDDO
                      ENDDO
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       DEALLOCATE( w_d, z_d, z_h )
       IF ( .NOT. lgamma ) DEALLOCATE( q0_d )
       !
       ! ... second set: pair p = (ib-1)*nat + na, nb = atom ib of the type,
       ! ... columns (p-1)*9 + k
       !
       nqb = qblock( nt )
       npair = tnat(nt) * nat
       npc = ncol_fit( 16.0_DP * REAL(ngml,DP) * 9, npair )
       ALLOCATE( w_d(ngml,npc*9), z_d(nijt,npc*9), z_h(nijt,npc*9) )
       DO p0 = 0, npair-1, npc
          npr = MIN( npc, npair-p0 )
          ncols = npr * 9
          !$cuf kernel do(2)
          DO c = 1, ncols
             DO ig = 1, n1
                k = MOD(c-1,9) + 1
                p = p0 + (c-1)/9 + 1
                na = MOD(p-1,nat_l) + 1
                nb = atl_d(tat0_l + (p-1)/nat_l + 1)
                v1 = g_d(1,ig) + xq1 ; v2 = g_d(2,ig) + xq2 ; v3 = g_d(3,ig) + xq3
                IF ( k == 1 ) THEN
                   pk = v1
                ELSEIF ( k == 2 ) THEN
                   pk = v2
                ELSEIF ( k == 3 ) THEN
                   pk = v3
                ELSEIF ( k == 4 ) THEN
                   pk = v1 * v1
                ELSEIF ( k == 5 ) THEN
                   pk = v1 * v2
                ELSEIF ( k == 6 ) THEN
                   pk = v1 * v3
                ELSEIF ( k == 7 ) THEN
                   pk = v2 * v2
                ELSEIF ( k == 8 ) THEN
                   pk = v2 * v3
                ELSE
                   pk = v3 * v3
                ENDIF
                ee = vloc_d(ig,ityp_d(na)) * eig1_d(mill_d(1,ig),na) * eig2_d(mill_d(2,ig),na) * &
                     eig3_d(mill_d(3,ig),na)
                w_d(ig,c) = CONJG( ee ) * pk * eig1_d(mill_d(1,ig),nb) * &
                            eig2_d(mill_d(2,ig),nb) * eig3_d(mill_d(3,ig),nb)
             ENDDO
          ENDDO
          DO ij0 = 0, nijt-1, nqb
             nc = MIN( nqb, nijt-ij0 )
             col = qcol( nt, ij0, nc )
             istat = cublasZgemm( hblas, CUBLAS_OP_T, CUBLAS_OP_N, nc, ncols, ngml, c_one, &
                                  qg_d(1,col), ngml, w_d, ngml, c_zero, z_d(ij0+1,1), nijt )
             IF ( istat /= 0 ) CALL errore( 'usaug_dvanqq', 'cublasZgemm failed', ABS(istat) )
          ENDDO
          z_h(:,1:ncols) = z_d(:,1:ncols)
          !
          DO p = p0+1, p0+npr
             na = MOD(p-1,nat) + 1
             nb = atl(tat0(nt) + (p-1)/nat + 1)
             c0 = (p-p0-1) * 9
             fact = eigqts(na) * CONJG( eigqts(nb) )
             ij = 0
             DO ih = 1, nh(nt)
                DO jh = ih, nh(nt)
                   ij = ij + 1
                   ijh = ijtoh(ih,jh,nt)
                   DO k = 1, 3
                      int2(ih,jh,k,na,nb) = CONJG( z_h(ij,c0+k) ) * fact * fact1
                      int2(jh,ih,k,na,nb) = int2(ih,jh,k,na,nb)
                   ENDDO
                   DO b = 1, 3
                      DO a = 1, 3
                         int5(ijh,a,b,na,nb) = CONJG(fact) * tpiba2 * omega * z_h(ij,c0+kk(a,b))
                      ENDDO
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       DEALLOCATE( w_d, z_d, z_h )
    ENDDO
    !
    DEALLOCATE( vg_d, vloc_d, ityp_d )
    IF ( .NOT. lgamma ) DEALLOCATE( yl0_d, qm0_d )
    !
  END SUBROUTINE usaug_dvanqq
  !
  !-----------------------------------------------------------------------
  SUBROUTINE usaug_addusddens_pulay( npe, mode0, u, dbecsum, alphasum, drhop, lpaw, becsumort )
    !-----------------------------------------------------------------------
    !! Augmentation part of the response density at fixed wavefunctions for
    !! the displacement modes mode0+1 ... mode0+npe, see addusddens_pulay:
    !!
    !! drho(G,is,ip) += sum_na S_na(q+G) sum_ij Q_ij(q+G) [ zsum(ij,na,is,ip)
    !!                  - i tpiba (q+G).u_na,ip becsum(ij,na,is) ]
    !!
    !! zsum = dbecsum + alphasum.u is saved in becsumort in the PAW case.
    !
    USE cell_base,        ONLY : tpiba
    USE fft_base,         ONLY : dfftp
    USE ions_base,        ONLY : nat, ntyp => nsp
    USE noncollin_module, ONLY : nspin_mag
    USE uspp,             ONLY : becsum
    USE uspp_param,       ONLY : upf, nhm
    USE qpoint,           ONLY : xq
    !
    INTEGER, INTENT(IN) :: npe, mode0
    COMPLEX(DP), INTENT(IN) :: u(3*nat,3*nat)
    COMPLEX(DP), INTENT(IN) :: dbecsum(nhm*(nhm+1)/2,nat,nspin_mag,npe)
    REAL(DP), INTENT(IN) :: alphasum(nhm*(nhm+1)/2,3,nat,nspin_mag)
    COMPLEX(DP), INTENT(INOUT) :: drhop(dfftp%nnr,nspin_mag,npe)
    LOGICAL, INTENT(IN) :: lpaw
    COMPLEX(DP), INTENT(INOUT) :: becsumort(:,:,:,:)
    !
    COMPLEX(DP), ALLOCATABLE, DEVICE :: aux_d(:,:), t1_d(:,:), t2_d(:,:), d1_d(:,:), d2_d(:,:), &
                                        uu_d(:,:)
    COMPLEX(DP), ALLOCATABLE :: d1_h(:,:), d2_h(:,:), uu_h(:,:)
    INTEGER :: nisp, nt, natc, a0, nac, ncols, ia0, ig, na, isp, ij0, nc, col, istat, ia, is, &
               ipert, n1, nqb, nijt, mu, mode, ij, nsp_l, npe_l
    REAL(DP) :: xq1, xq2, xq3, v1, v2, v3
    COMPLEX(DP) :: sk, beta, fact, u1, u2, u3, zsum, alpha
    LOGICAL :: moved
    !
    CALL prepare_q( )
    nisp = nspin_mag * npe
    n1 = ngml ; nsp_l = nspin_mag ; npe_l = npe
    xq1 = xq(1) ; xq2 = xq(2) ; xq3 = xq(3)
    fact = CMPLX( 0.0_DP, - tpiba, KIND=DP )
    ALLOCATE( aux_d(ngml,nisp) )
    !$cuf kernel do(2)
    DO isp = 1, nisp
       DO ig = 1, n1
          aux_d(ig,isp) = (0.0_DP, 0.0_DP)
       ENDDO
    ENDDO
    !
    DO nt = 1, ntyp
       IF ( .NOT. upf(nt)%tvanp .OR. tnat(nt) == 0 ) CYCLE
       nijt = nij(nt)
       nqb = qblock( nt )
       natc = ncol_fit( 16.0_DP * REAL(ngml,DP) * (nisp + nspin_mag), tnat(nt) )
       ALLOCATE( t1_d(ngml,natc*nisp), t2_d(ngml,natc*nspin_mag), d1_d(nijt,natc*nisp), &
                 d2_d(nijt,natc*nspin_mag), uu_d(3,natc*npe) )
       ALLOCATE( d1_h(nijt,natc*nisp), d2_h(nijt,natc*nspin_mag), uu_h(3,natc*npe) )
       DO a0 = 0, tnat(nt)-1, natc
          nac = MIN( natc, tnat(nt)-a0 )
          ia0 = tat0(nt) + a0
          DO ia = 1, nac
             na = atl(ia0+ia)
             mu = 3 * (na - 1)
             DO ipert = 1, npe
                mode = mode0 + ipert
                u1 = u(mu+1,mode)
                u2 = u(mu+2,mode)
                u3 = u(mu+3,mode)
                moved = ABS(u1) + ABS(u2) + ABS(u3) >= 1.0E-12_DP
                IF ( .NOT. moved ) THEN
                   u1 = (0.0_DP, 0.0_DP) ; u2 = (0.0_DP, 0.0_DP) ; u3 = (0.0_DP, 0.0_DP)
                ENDIF
                uu_h(1,(ia-1)*npe+ipert) = u1
                uu_h(2,(ia-1)*npe+ipert) = u2
                uu_h(3,(ia-1)*npe+ipert) = u3
                DO is = 1, nspin_mag
                   isp = (ipert-1)*nspin_mag + is
                   DO ij = 1, nijt
                      zsum = dbecsum(ij,na,is,ipert)
                      IF ( moved ) zsum = zsum + ( alphasum(ij,1,na,is) * u1 + &
                                                   alphasum(ij,2,na,is) * u2 + &
                                                   alphasum(ij,3,na,is) * u3 )
                      d1_h(ij,(ia-1)*nisp+isp) = zsum
                      IF ( lpaw ) becsumort(ij,na,is,mode) = zsum
                   ENDDO
                ENDDO
             ENDDO
             DO is = 1, nspin_mag
                DO ij = 1, nijt
                   d2_h(ij,(ia-1)*nspin_mag+is) = CMPLX( becsum(ij,na,is), 0.0_DP, KIND=DP )
                ENDDO
             ENDDO
          ENDDO
          ncols = nac * nisp
          d1_d(:,1:ncols) = d1_h(:,1:ncols)
          d2_d(:,1:nac*nspin_mag) = d2_h(:,1:nac*nspin_mag)
          uu_d(:,1:nac*npe) = uu_h(:,1:nac*npe)
          !
          beta = c_zero
          DO ij0 = 0, nijt-1, nqb
             nc = MIN( nqb, nijt-ij0 )
             col = qcol( nt, ij0, nc )
             istat = cublasZgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, ngml, ncols, nc, c_one, &
                                  qg_d(1,col), ngml, d1_d(ij0+1,1), nijt, beta, t1_d, ngml )
             IF ( istat /= 0 ) CALL errore( 'usaug_addusddens_pulay', 'cublasZgemm failed', ABS(istat) )
             istat = cublasZgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, ngml, nac*nspin_mag, nc, c_one, &
                                  qg_d(1,col), ngml, d2_d(ij0+1,1), nijt, beta, t2_d, ngml )
             IF ( istat /= 0 ) CALL errore( 'usaug_addusddens_pulay', 'cublasZgemm failed', ABS(istat) )
             beta = c_one
          ENDDO
          !$cuf kernel do(1)
          DO ig = 1, n1
             v1 = xq1 + g_d(1,ig) ; v2 = xq2 + g_d(2,ig) ; v3 = xq3 + g_d(3,ig)
             DO ia = 1, nac
                na = atl_d(ia0+ia)
                sk = eig1_d(mill_d(1,ig),na) * eig2_d(mill_d(2,ig),na) * &
                     eig3_d(mill_d(3,ig),na) * eigq_d(na)
                DO isp = 1, nisp
                   is = MOD(isp-1,nsp_l) + 1
                   ipert = (isp-1)/nsp_l + 1
                   alpha = v1 * uu_d(1,(ia-1)*npe_l+ipert) + v2 * uu_d(2,(ia-1)*npe_l+ipert) + &
                           v3 * uu_d(3,(ia-1)*npe_l+ipert)
                   aux_d(ig,isp) = aux_d(ig,isp) + sk * ( t1_d(ig,(ia-1)*nisp+isp) + &
                                   fact * alpha * t2_d(ig,(ia-1)*nsp_l+is) )
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       DEALLOCATE( t1_d, t2_d, d1_d, d2_d, uu_d, d1_h, d2_h, uu_h )
    ENDDO
    !
    CALL add_from_g( aux_d, nisp, drhop )
    DEALLOCATE( aux_d )
    !
  END SUBROUTINE usaug_addusddens_pulay
  !
#else
  !
  ! ... stubs for builds without CUDA: usaug_on is false and these are never called
  !
  SUBROUTINE usaug_invalidate( )
  END SUBROUTINE usaug_invalidate
  !
  SUBROUTINE usaug_newdq( dvscf, npe )
    USE fft_base,         ONLY : dfftp
    USE noncollin_module, ONLY : nspin_mag
    INTEGER, INTENT(IN) :: npe
    COMPLEX(DP), INTENT(IN) :: dvscf(dfftp%nnr,nspin_mag,npe)
    CALL errore( 'usaug_newdq', 'not available without CUDA', 1 )
  END SUBROUTINE usaug_newdq
  !
  SUBROUTINE usaug_addusddens( npert, dbecsum, drhop )
    USE fft_base,         ONLY : dfftp
    USE ions_base,        ONLY : nat
    USE noncollin_module, ONLY : nspin_mag
    USE uspp_param,       ONLY : nhm
    INTEGER, INTENT(IN) :: npert
    COMPLEX(DP), INTENT(IN) :: dbecsum(nhm*(nhm+1)/2,nat,nspin_mag,npert)
    COMPLEX(DP), INTENT(INOUT) :: drhop(dfftp%nnr,nspin_mag,npert)
    CALL errore( 'usaug_addusddens', 'not available without CUDA', 1 )
  END SUBROUTINE usaug_addusddens
  !
  SUBROUTINE usaug_dvanqq( vlocq, int1, int2, int4, int5 )
    REAL(DP), INTENT(IN) :: vlocq(:,:)
    COMPLEX(DP), INTENT(INOUT) :: int1(:,:,:,:,:), int2(:,:,:,:,:), int4(:,:,:,:,:), int5(:,:,:,:,:)
    CALL errore( 'usaug_dvanqq', 'not available without CUDA', 1 )
  END SUBROUTINE usaug_dvanqq
  !
  SUBROUTINE usaug_addusddens_pulay( npe, mode0, u, dbecsum, alphasum, drhop, lpaw, becsumort )
    USE fft_base,         ONLY : dfftp
    USE ions_base,        ONLY : nat
    USE noncollin_module, ONLY : nspin_mag
    USE uspp_param,       ONLY : nhm
    INTEGER, INTENT(IN) :: npe, mode0
    COMPLEX(DP), INTENT(IN) :: u(3*nat,3*nat)
    COMPLEX(DP), INTENT(IN) :: dbecsum(nhm*(nhm+1)/2,nat,nspin_mag,npe)
    REAL(DP), INTENT(IN) :: alphasum(nhm*(nhm+1)/2,3,nat,nspin_mag)
    COMPLEX(DP), INTENT(INOUT) :: drhop(dfftp%nnr,nspin_mag,npe)
    LOGICAL, INTENT(IN) :: lpaw
    COMPLEX(DP), INTENT(INOUT) :: becsumort(:,:,:,:)
    CALL errore( 'usaug_addusddens_pulay', 'not available without CUDA', 1 )
  END SUBROUTINE usaug_addusddens_pulay
  !
#endif
  !
END MODULE usaug_dev

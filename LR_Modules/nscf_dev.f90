!
! Copyright (C) 2026 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
MODULE nscf_dev
  !-----------------------------------------------------------------------
  !! Non-self-consistent band calculation of ph.x (bands at k and k+q) with
  !! the Davidson iterations of all k-points advanced together on the GPU.
  !!
  !! c_bands_nscf_ph diagonalizes one k-point at a time; for the small
  !! cells and dense k-point sets typical of phonon calculations every
  !! kernel it launches is tiny and the GPU idles. Here the starting
  !! wavefunctions are prepared k-point by k-point exactly as in the stock
  !! routine (init_wfc), then the Davidson algorithm of cegterg (same
  !! correction vectors, preconditioner, reduced generalized eigenproblem,
  !! convergence test and restart rule) is run for all k-points in lockstep:
  !! each step is a few batched cuFFT / cuBLAS calls over all the k-points
  !! that are not yet converged; the small reduced eigenproblems are solved
  !! on the host with LAPACK (ZHEGVX, as cdiaghg), one thread per k-point.
  !!
  !! Served: collinear, Davidson (isolve = 0), norm-conserving, ultrasoft
  !! and PAW, FFT grid not distributed. QE_PH_DEV=0 or QE_PH_NSCF_DEV=0
  !! select the stock routine.
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
  !
#if defined(__CUDA)
  COMPLEX(DP), PARAMETER :: c_one = (1.0_DP, 0.0_DP), c_zero = (0.0_DP, 0.0_DP)
  INTEGER, PARAMETER :: maxter = 20
  !! maximum number of Davidson iterations per call (as in cegterg)
  INTEGER, PARAMETER :: maxgap = 64
  !! finished k-points tolerated between two active ones in one batched call
  !
  INTEGER :: nk, nvec, nvecx, npwxl, nkbl, nnr, nr1, nr2, nr3, nspl, nhml, natl, nfb
  LOGICAL :: lus
  !
  COMPLEX(DP), ALLOCATABLE, DEVICE :: psi_d(:,:,:), hpsi_d(:,:,:), spsi_d(:,:,:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: wrk_d(:,:,:), hwrk_d(:,:,:), swrk_d(:,:,:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: vkb_d(:,:,:), bw_d(:,:,:), bw2_d(:,:,:), bw3_d(:,:,:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: vc_d(:,:,:), hcn_d(:,:,:), scn_d(:,:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: g2_d(:,:), hd_d(:,:), sd_d(:,:), ew_d(:,:), v_d(:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: deeq_d(:,:,:,:), qq_d(:,:,:), nrm_d(:,:)
  INTEGER, ALLOCATABLE, DEVICE :: nl_d(:,:), npw_d(:), spin_d(:), nbase_d(:), flag_d(:)
  INTEGER, ALLOCATABLE, DEVICE :: kbna_d(:), kbof_d(:), kbnh_d(:), pk_d(:), pn_d(:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: psig_d(:), psic_d(:)
  !
  INTEGER, PARAMETER :: nplan_max = 8
  INTEGER :: nplan = 0, plan(nplan_max) = 0, plan_nb(nplan_max) = 0
  TYPE(cublasHandle) :: hblas
  LOGICAL :: blas_ready = .FALSE.
#endif
  !
CONTAINS
  !
#if defined(__CUDA)
  !-----------------------------------------------------------------------
  SUBROUTINE init_batched( ok )
    !-----------------------------------------------------------------------
    !! Beta functions, kinetic energies, preconditioner and starting
    !! wavefunctions of all k-points: what g2_kin, init_us_2, usnldiag,
    !! atomic_wfc, init_wfc (startingwfc = 'atomic', the setting of ph.x) and
    !! rotate_wfc_k compute for one k-point, with the same expressions and the
    !! same random numbers, for blocks of k-points. ok = .FALSE.: not served,
    !! nothing was done.
    !
    USE constants,            ONLY : tpi
    USE cell_base,            ONLY : tpiba, tpiba2, omega
    USE ions_base,            ONLY : nat, ityp, tau, nsp
    USE klist,                ONLY : xk, ngk, igk_k
    USE gvect,                ONLY : g, mill, eigts1, eigts2, eigts3
    USE gvecw,                ONLY : qcutz
    USE uspp,                 ONLY : vkb, nkb, nhtol, nhtolm, indv, ofsbeta
    USE uspp_param,           ONLY : upf, nh, nhm, lmaxkb, nbetam, nwfcm
    USE uspp_init,            ONLY : init_us_2
    USE beta_mod,             ONLY : interp_beta
    USE atwfc_mod,            ONLY : interp_atwfc
    USE basis,                ONLY : natomwfc
    USE scf,                  ONLY : v_of_0
    USE lsda_mod,             ONLY : lsda, isk, current_spin
    USE wvfct,                ONLY : current_k
    USE fft_base,             ONLY : dffts
    USE starting_scf,         ONLY : starting_wfc
    USE random_numbers_gpum,  ONLY : randy_vect_gpu
    !
    LOGICAL, INTENT(OUT) :: ok
    !
    INTEGER :: nstart, nr, lmax_wfc, lmm, kbmax, kb, k1, k2, npb, ik, ig, ip, iv, nt, na, nb, ih, jh, &
               lm, nhnt, nat_t, ia, jkb, n, l, m, is, nsl, il, nl, nlmax, istat, n1, nwa, nrr, ioff, &
               cnt, npair, nrnd, kk, r1(1), r2(1)
    INTEGER(8) :: ro, idx
    REAL(DP) :: arg, rr1, rr2, hh, ss, ar, tmpneed, vof0, tpiba_l, tpiba2_l
    COMPLEX(DP) :: sk, pref
    INTEGER(KIND=cuda_count_kind) :: mfree, mtot
    INTEGER, ALLOCATABLE :: nl_h(:,:), iv_h(:), na_t(:), w_na(:), w_nb(:), w_nt(:), w_lm(:), &
                            ikbl_h(:,:,:), nls_h(:), pk_h(:), pn_h(:), info_k(:)
    INTEGER(8), ALLOCATABLE :: roff_h(:)
    REAL(DP), ALLOCATABLE :: gk(:,:), qg(:), ylm(:,:), vq(:,:), vkb1(:,:), chiq(:,:,:), arg_h(:,:), &
                             prl_h(:,:,:), deeq_h(:,:,:,:), qq_h(:,:,:), ewk(:)
    COMPLEX(DP), ALLOCATABLE :: w_ph(:), kph_h(:,:), hr_h(:,:,:), sr_h(:,:,:), vr_h(:,:,:), hk(:,:), &
                                sk2(:,:), vk(:,:)
    INTEGER, ALLOCATABLE, DEVICE :: ikbl_d(:,:,:), nls_d(:)
    INTEGER(8), ALLOCATABLE, DEVICE :: roff_d(:)
    REAL(DP), ALLOCATABLE, DEVICE :: xk_d(:,:), prl_d(:,:,:), rnd_d(:)
    COMPLEX(DP), ALLOCATABLE, DEVICE :: hr_d(:,:,:), sr_d(:,:,:), vr_d(:,:,:)
    !
    ok = .FALSE.
    IF ( TRIM(starting_wfc) /= 'atomic' .OR. qcutz > 0.0_DP ) RETURN
    nstart = MAX( natomwfc, nvec )
    nr = nstart - natomwfc
    IF ( natomwfc < 1 .OR. nwfcm < 1 .OR. nstart > nvecx ) RETURN
    lmax_wfc = 0
    DO nt = 1, nsp
       IF ( upf(nt)%nwfc > 0 ) lmax_wfc = MAX( lmax_wfc, MAXVAL( upf(nt)%lchi(1:upf(nt)%nwfc) ) )
    ENDDO
    lmm = (lmax_wfc+1)**2
    IF ( nkbl > 0 ) lmm = MAX( lmm, (lmaxkb+1)**2 )
    !
    ! ... k-points per block: bounded work space
    !
    kbmax = MAX( 1, MIN( nk, 1048576 / npwxl ) )
    IF ( nr > 0 ) kbmax = MAX( 1, MIN( kbmax, INT( 67108864_8 / ( INT(nr,8) * npwxl ) ) ) )
    tmpneed = 8.0_DP * REAL(npwxl,DP) * kbmax * ( 4 + lmm + nbetam + nhm + nwfcm*nsp + 2*nr ) + &
              16.0_DP * REAL(nstart,DP) * ( 2*nstart + nvec ) * nk
    istat = cudaMemGetInfo( mfree, mtot )
    IF ( tmpneed > 0.5_DP * REAL(mfree,DP) ) RETURN
    !
    n1 = npwxl
    tpiba_l = tpiba ; tpiba2_l = tpiba2 ; vof0 = v_of_0
    !
    ! ... FFT indices, spin and number of plane waves of each k-point
    !
    ALLOCATE( nl_h(npwxl,nk), iv_h(nk) )
    nl_h = 1
    DO ik = 1, nk
       DO ig = 1, ngk(ik)
          nl_h(ig,ik) = dffts%nl(igk_k(ig,ik))
       ENDDO
    ENDDO
    nl_d = nl_h
    iv_h = 1
    IF ( lsda ) iv_h(:) = isk(1:nk)
    spin_d = iv_h
    iv_h(:) = ngk(1:nk)
    npw_d = iv_h
    DEALLOCATE( nl_h, iv_h )
    ALLOCATE( xk_d(3,nk) )
    xk_d = xk(:,1:nk)
    !
    ! ... kinetic energies (g2_kin)
    !
    !$acc parallel loop collapse(2) present(g, igk_k)
    DO ik = 1, nk
       DO ig = 1, n1
          IF ( ig <= npw_d(ik) ) THEN
             iv = igk_k(ig,ik)
             g2_d(ig,ik) = ( ( xk_d(1,ik) + g(1,iv) )*( xk_d(1,ik) + g(1,iv) ) + &
                             ( xk_d(2,ik) + g(2,iv) )*( xk_d(2,ik) + g(2,iv) ) + &
                             ( xk_d(3,ik) + g(3,iv) )*( xk_d(3,ik) + g(3,iv) ) ) * tpiba2_l
          ELSE
             g2_d(ig,ik) = 0.0_DP
          ENDIF
       ENDDO
    ENDDO
    !
    ! ... the atomic wavefunctions in the order of atomic_wfc: atom, radial
    ! ... function, angular momentum and phase (0,1)^l of each
    !
    ALLOCATE( w_na(natomwfc), w_nb(natomwfc), w_nt(natomwfc), w_lm(natomwfc), w_ph(natomwfc) )
    n = 0
    DO na = 1, nat
       nt = ityp(na)
       DO nb = 1, upf(nt)%nwfc
          IF ( upf(nt)%oc(nb) >= 0.0_DP ) THEN
             l = upf(nt)%lchi(nb)
             DO m = 1, 2*l + 1
                n = n + 1
                IF ( n > natomwfc ) CALL errore( 'nscf_dev', 'internal error: too many wfcs', 1 )
                w_na(n) = na ; w_nb(n) = nb ; w_nt(n) = nt ; w_lm(n) = l*l + m
                w_ph(n) = (0.0_DP,1.0_DP)**l
             ENDDO
          ENDIF
       ENDDO
    ENDDO
    IF ( n /= natomwfc ) CALL errore( 'nscf_dev', 'internal error: some wfcs were lost', 1 )
    nwa = natomwfc ; nrr = nr
    !
    ALLOCATE( na_t(nat), arg_h(nat,kbmax), kph_h(nat,kbmax) )
    IF ( nr > 0 ) ALLOCATE( rnd_d(2_8*nr*npwxl*kbmax), roff_h(kbmax), roff_d(kbmax) )
    !
    DO k1 = 1, nk, kbmax
       k2 = MIN( nk, k1+kbmax-1 )
       kb = k2 - k1 + 1
       npb = npwxl * kb
       ALLOCATE( gk(3,npb), qg(npb), ylm(npb,lmm), vq(npb,MAX(nbetam,1)), vkb1(npb,MAX(nhm,1)), &
                 chiq(npb,nwfcm,nsp) )
       !$acc data create(gk, qg, ylm, vq, vkb1, chiq) copyin(w_na, w_nb, w_nt, w_lm, w_ph) &
       !$acc      present(g, igk_k, mill, eigts1, eigts2, eigts3)
       !
       !$acc parallel loop collapse(2)
       DO ik = k1, k2
          DO ig = 1, n1
             ip = (ik-k1)*n1 + ig
             IF ( ig <= npw_d(ik) ) THEN
                iv = igk_k(ig,ik)
                gk(1,ip) = xk_d(1,ik) + g(1,iv)
                gk(2,ip) = xk_d(2,ik) + g(2,iv)
                gk(3,ip) = xk_d(3,ik) + g(3,iv)
                qg(ip) = gk(1,ip)*gk(1,ip) + gk(2,ip)*gk(2,ip) + gk(3,ip)*gk(3,ip)
             ELSE
                gk(1,ip) = 0.0_DP ; gk(2,ip) = 0.0_DP ; gk(3,ip) = 0.0_DP
                qg(ip) = 0.0_DP
             ENDIF
          ENDDO
       ENDDO
       CALL ylmr2( lmm, npb, gk, qg, ylm )
       !$acc parallel loop
       DO ip = 1, npb
          qg(ip) = SQRT( qg(ip) ) * tpiba_l
       ENDDO
       !
       ! ... beta functions (init_us_2)
       !
       IF ( nkbl > 0 ) THEN
          jkb = 0
          DO nt = 1, nsp
             CALL interp_beta( nt, npb, qg, omega, vq )
             nhnt = nh(nt)
             !$acc parallel loop collapse(2) copyin(indv, nhtolm)
             DO ih = 1, nhnt
                DO ip = 1, npb
                   nb = indv(ih,nt)
                   lm = nhtolm(ih,nt)
                   vkb1(ip,ih) = ylm(ip,lm) * vq(ip,nb)
                ENDDO
             ENDDO
             nat_t = 0
             arg_h = 0.0_DP
             DO na = 1, nat
                IF ( ityp(na) == nt ) THEN
                   nat_t = nat_t + 1
                   na_t(nat_t) = na
                   DO ik = k1, k2
                      arg_h(nat_t,ik-k1+1) = ( xk(1,ik)*tau(1,na) + xk(2,ik)*tau(2,na) + &
                                               xk(3,ik)*tau(3,na) ) * tpi
                   ENDDO
                ENDIF
             ENDDO
             IF ( nat_t == 0 ) CYCLE
             !$acc parallel loop collapse(3) copyin(na_t, arg_h, nhtol) private(ip, na, iv, sk, pref, kk)
             DO ia = 1, nat_t
                DO ik = k1, k2
                   DO ig = 1, n1
                      IF ( ig <= npw_d(ik) ) THEN
                         ip = (ik-k1)*n1 + ig
                         kk = ik - k1 + 1
                         na = na_t(ia)
                         iv = igk_k(ig,ik)
                         sk = eigts1(mill(1,iv),na) * eigts2(mill(2,iv),na) * eigts3(mill(3,iv),na) * &
                              CMPLX( COS(arg_h(ia,kk)), -SIN(arg_h(ia,kk)), KIND=DP )
                         !$acc loop seq
                         DO ih = 1, nhnt
                            pref = (0.0_DP,-1.0_DP)**nhtol(ih,nt)
                            vkb_d(ig,jkb+(ia-1)*nhnt+ih,ik) = vkb1(ip,ih) * sk * pref
                         ENDDO
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             jkb = jkb + nat_t*nhnt
          ENDDO
       ENDIF
       !
       ! ... atomic wavefunctions (atomic_wfc)
       !
       CALL interp_atwfc( npb, qg, nwfcm, omega, chiq )
       DO ik = k1, k2
          DO na = 1, nat
             arg = ( xk(1,ik)*tau(1,na) + xk(2,ik)*tau(2,na) + xk(3,ik)*tau(3,na) ) * tpi
             kph_h(na,ik-k1+1) = CMPLX( COS(arg), -SIN(arg), KIND=DP )
          ENDDO
       ENDDO
       !$acc parallel loop collapse(3) copyin(kph_h) private(ip, na, iv, sk)
       DO ik = k1, k2
          DO n = 1, nwa
             DO ig = 1, n1
                IF ( ig <= npw_d(ik) ) THEN
                   ip = (ik-k1)*n1 + ig
                   na = w_na(n)
                   iv = igk_k(ig,ik)
                   sk = kph_h(na,ik-k1+1) * eigts1(mill(1,iv),na) * eigts2(mill(2,iv),na) * &
                        eigts3(mill(3,iv),na)
                   psi_d(ig,n,ik) = w_ph(n) * sk * CMPLX( ylm(ip,w_lm(n)) * chiq(ip,w_nb(n),w_nt(n)), KIND=DP )
                ENDIF
             ENDDO
          ENDDO
       ENDDO
       !
       ! ... random wavefunctions where the atomic ones are too few (init_wfc):
       ! ... the generator is called k-point by k-point as in the stock routine
       !
       IF ( nr > 0 ) THEN
          ro = 0
          DO ik = k1, k2
             roff_h(ik-k1+1) = ro
             nrnd = 2 * nr * ngk(ik)
             CALL randy_vect_gpu( rnd_d(ro+1:ro+nrnd), nrnd )
             ro = ro + nrnd
          ENDDO
          roff_d = roff_h
          !$acc parallel loop collapse(3) private(idx, rr1, rr2, iv, arg)
          DO ik = k1, k2
             DO n = 1, nrr
                DO ig = 1, n1
                   IF ( ig <= npw_d(ik) ) THEN
                      idx = roff_d(ik-k1+1) + 2_8*( INT(ig-1,8) + INT(n-1,8)*npw_d(ik) ) + 1
                      rr1 = rnd_d(idx)
                      rr2 = rnd_d(idx+1)
                      iv = igk_k(ig,ik)
                      rr1 = rr1 / ( ( xk_d(1,ik) + g(1,iv) )**2 + ( xk_d(2,ik) + g(2,iv) )**2 + &
                                    ( xk_d(3,ik) + g(3,iv) )**2 + 1.0_DP )
                      arg = tpi * rr2
                      psi_d(ig,nwa+n,ik) = CMPLX( rr1*COS(arg), rr1*SIN(arg), KIND=DP )
                   ENDIF
                ENDDO
             ENDDO
          ENDDO
       ENDIF
       !
       !$acc end data
       DEALLOCATE( gk, qg, ylm, vq, vkb1, chiq )
    ENDDO
    DEALLOCATE( na_t, arg_h, kph_h, w_na, w_nb, w_nt, w_lm, w_ph )
    IF ( nr > 0 ) DEALLOCATE( rnd_d, roff_h, roff_d )
    !
    ! ... diagonal of H and S for the preconditioner (usnldiag): the nonzero
    ! ... coefficients of each spin, in the order in which they are added
    !
    nsl = 1
    IF ( lsda ) nsl = 2
    nlmax = MAX( 1, natl*nhml*nhml )
    ALLOCATE( prl_h(2,nlmax,nsl), ikbl_h(2,nlmax,nsl), nls_h(nsl) )
    prl_h = 0.0_DP ; ikbl_h = 1 ; nls_h = 0
    IF ( nhml > 0 .AND. nkbl > 0 ) THEN
       ALLOCATE( deeq_h(nhml,nhml,natl,nspl), qq_h(nhml,nhml,natl) )
       deeq_h = deeq_d ; qq_h = qq_d
       DO is = 1, nsl
          nl = 0
          DO nt = 1, nsp
             nhnt = nh(nt)
             DO na = 1, nat
                IF ( ityp(na) == nt ) THEN
                   DO ih = 1, nhnt
                      DO jh = 1, nhnt
                         IF ( deeq_h(ih,jh,na,is) == 0.0_DP .AND. qq_h(ih,jh,na) == 0.0_DP ) CYCLE
                         nl = nl + 1
                         ikbl_h(1,nl,is) = ofsbeta(na) + ih
                         ikbl_h(2,nl,is) = ofsbeta(na) + jh
                         prl_h(1,nl,is) = deeq_h(ih,jh,na,is)
                         prl_h(2,nl,is) = qq_h(ih,jh,na)
                      ENDDO
                   ENDDO
                ENDIF
             ENDDO
          ENDDO
          nls_h(is) = nl
       ENDDO
       DEALLOCATE( deeq_h, qq_h )
    ENDIF
    ALLOCATE( prl_d(2,nlmax,nsl), ikbl_d(2,nlmax,nsl), nls_d(nsl) )
    prl_d = prl_h ; ikbl_d = ikbl_h ; nls_d = nls_h
    !$acc parallel loop collapse(2) private(is, hh, ss, ar)
    DO ik = 1, nk
       DO ig = 1, n1
          IF ( ig <= npw_d(ik) ) THEN
             is = spin_d(ik)
             hh = g2_d(ig,ik) + vof0
             ss = 1.0_DP
             !$acc loop seq
             DO il = 1, nls_d(is)
                ar = DBLE( vkb_d(ig,ikbl_d(1,il,is),ik) * CONJG( vkb_d(ig,ikbl_d(2,il,is),ik) ) )
                hh = hh + prl_d(1,il,is) * ar
                ss = ss + prl_d(2,il,is) * ar
             ENDDO
             hd_d(ig,ik) = hh
             sd_d(ig,ik) = ss
          ELSE
             hd_d(ig,ik) = 1.0_DP
             sd_d(ig,ik) = 1.0_DP
          ENDIF
       ENDDO
    ENDDO
    DEALLOCATE( prl_h, ikbl_h, nls_h, prl_d, ikbl_d, nls_d, xk_d )
    !
    ! ... diagonalization in the space of the starting wavefunctions
    ! ... (rotate_wfc_k): H and S on the nstart vectors, nvec at a time
    !
    CALL start_clock( 'nscfd_irot' )
    ALLOCATE( pk_h(nvec*nk), pn_h(nvec*nk) )
    r1(1) = 1 ; r2(1) = nk
    DO ioff = 0, nstart-1, nvec
       cnt = MIN( nvec, nstart-ioff )
       npair = 0
       DO ik = 1, nk
          DO n = 1, cnt
             npair = npair + 1
             pk_h(npair) = ik ; pn_h(npair) = n
          ENDDO
       ENDDO
       pk_d(1:npair) = pk_h(1:npair) ; pn_d(1:npair) = pn_h(1:npair)
       CALL copy_block( psi_d, wrk_d, ioff, cnt )
       CALL hs_apply( npair, 1, r1, r2 )
       CALL put_block( hwrk_d, hpsi_d, ioff, cnt )
       IF ( lus ) CALL put_block( swrk_d, spsi_d, ioff, cnt )
    ENDDO
    DEALLOCATE( pk_h, pn_h )
    !
    ALLOCATE( hr_d(nstart,nstart,nk), sr_d(nstart,nstart,nk), vr_d(nstart,nvec,nk) )
    ALLOCATE( hr_h(nstart,nstart,nk), sr_h(nstart,nstart,nk), vr_h(nstart,nvec,nk), info_k(nk) )
    CALL gemm_b( CUBLAS_OP_C, nstart, nstart, npwxl, psi_d, npwxl, INT(npwxl,8)*nvecx, &
                 hpsi_d, npwxl, INT(npwxl,8)*nvecx, c_zero, hr_d, nstart, INT(nstart,8)*nstart, nk )
    IF ( lus ) THEN
       CALL gemm_b( CUBLAS_OP_C, nstart, nstart, npwxl, psi_d, npwxl, INT(npwxl,8)*nvecx, &
                    spsi_d, npwxl, INT(npwxl,8)*nvecx, c_zero, sr_d, nstart, INT(nstart,8)*nstart, nk )
    ELSE
       CALL gemm_b( CUBLAS_OP_C, nstart, nstart, npwxl, psi_d, npwxl, INT(npwxl,8)*nvecx, &
                    psi_d, npwxl, INT(npwxl,8)*nvecx, c_zero, sr_d, nstart, INT(nstart,8)*nstart, nk )
    ENDIF
    hr_h = hr_d ; sr_h = sr_d
    info_k = 0
    !$omp parallel do schedule(dynamic,8) private(ik, hk, sk2, vk, ewk)
    DO ik = 1, nk
       ALLOCATE( hk(nstart,nstart), sk2(nstart,nstart), vk(nstart,nvec), ewk(nstart) )
       hk(:,:) = hr_h(:,:,ik)
       sk2(:,:) = sr_h(:,:,ik)
       CALL diag_small( nstart, nvec, hk, sk2, ewk, vk, info_k(ik) )
       vr_h(:,:,ik) = vk(:,:)
       DEALLOCATE( hk, sk2, vk, ewk )
    ENDDO
    !$omp end parallel do
    IF ( ANY( info_k /= 0 ) ) CALL errore( 'c_bands_nscf_dev', &
         'starting wavefunctions: reduced eigenproblem failed (S matrix not positive definite?)', 1 )
    vr_d = vr_h
    CALL gemm_b( CUBLAS_OP_N, npwxl, nvec, nstart, psi_d, npwxl, INT(npwxl,8)*nvecx, &
                 vr_d, nstart, INT(nstart,8)*nvec, c_zero, wrk_d, npwxl, INT(npwxl,8)*nvec, nk )
    CALL put_block( wrk_d, psi_d, 0, nvec )
    CALL zero_cols( nvec, nstart )
    DEALLOCATE( hr_d, sr_d, vr_d, hr_h, sr_h, vr_h, info_k )
    CALL stop_clock( 'nscfd_irot' )
    !
    ! ... module variables as the loop over k-points of the stock routine leaves them
    !
    current_k = nk
    IF ( lsda ) current_spin = isk(nk)
    CALL g2_kin( nk )
    IF ( nkb > 0 ) THEN
       CALL init_us_2( ngk(nk), igk_k(1,nk), xk(1,nk), vkb, .TRUE. )
       !$acc update host(vkb)
    ENDIF
    ok = .TRUE.
    !
  END SUBROUTINE init_batched
  !
  !-----------------------------------------------------------------------
  SUBROUTINE zero_cols( n0, n9 )
    !-----------------------------------------------------------------------
    !! columns n0+1 ... n9 of psi, H psi and S psi set to zero for all k
    INTEGER, INTENT(IN) :: n0, n9
    INTEGER :: ik, n, ig, n1, n3
    n1 = npwxl ; n3 = nk
    IF ( n9 <= n0 ) RETURN
    !$cuf kernel do(3)
    DO ik = 1, n3
       DO n = n0+1, n9
          DO ig = 1, n1
             psi_d(ig,n,ik) = (0.0_DP, 0.0_DP)
             hpsi_d(ig,n,ik) = (0.0_DP, 0.0_DP)
          ENDDO
       ENDDO
    ENDDO
    IF ( lus ) THEN
       !$cuf kernel do(3)
       DO ik = 1, n3
          DO n = n0+1, n9
             DO ig = 1, n1
                spsi_d(ig,n,ik) = (0.0_DP, 0.0_DP)
             ENDDO
          ENDDO
       ENDDO
    ENDIF
  END SUBROUTINE zero_cols
  !
  !-----------------------------------------------------------------------
  SUBROUTINE diag_small( n, m, h, s, e, v, info )
    !-----------------------------------------------------------------------
    !! lowest m eigenpairs of the n x n problem h v = e s v (upper triangles),
    !! as the host branch of cdiaghg
    INTEGER, INTENT(IN) :: n, m
    COMPLEX(DP), INTENT(INOUT) :: h(n,n), s(n,n)
    REAL(DP), INTENT(OUT) :: e(n)
    COMPLEX(DP), INTENT(OUT) :: v(n,m)
    INTEGER, INTENT(OUT) :: info
    INTEGER :: lwork, mm
    INTEGER, ALLOCATABLE :: iwork(:), ifail(:)
    REAL(DP), ALLOCATABLE :: rwork(:)
    COMPLEX(DP), ALLOCATABLE :: work(:), vv(:,:)
    !
    lwork = 33*n
    ALLOCATE( work(lwork), rwork(7*n), iwork(5*n), ifail(n) )
    IF ( m == n ) THEN
       ALLOCATE( vv(n,n) )
       vv = h
       CALL ZHEGV( 1, 'V', 'U', n, vv, n, s, n, e, work, lwork, rwork, info )
       v(:,1:m) = vv(:,1:m)
       DEALLOCATE( vv )
    ELSE
       CALL ZHEGVX( 1, 'V', 'I', 'U', n, h, n, s, n, 0.0_DP, 0.0_DP, 1, m, 0.0_DP, mm, e, v, n, &
                    work, lwork, rwork, iwork, ifail, info )
    ENDIF
    DEALLOCATE( work, rwork, iwork, ifail )
    !
  END SUBROUTINE diag_small
  !
  !-----------------------------------------------------------------------
  SUBROUTINE make_plans( )
    !-----------------------------------------------------------------------
    INTEGER :: nb, istat, fdim(3)
    fdim = (/ nr3, nr2, nr1 /)
    nplan = 0
    nb = nfb
    DO
       nplan = nplan + 1
       plan_nb(nplan) = nb
       istat = cufftPlanMany( plan(nplan), 3, fdim, fdim, 1, nnr, fdim, 1, nnr, CUFFT_Z2Z, nb )
       IF ( istat /= 0 ) CALL errore( 'nscf_dev', 'cufftPlanMany failed', ABS(istat) )
       IF ( nb == 1 .OR. nplan == nplan_max ) EXIT
       nb = MAX( 1, nb / 4 )
    ENDDO
    IF ( plan_nb(nplan) /= 1 ) CALL errore( 'nscf_dev', 'FFT plan table too small', 1 )
  END SUBROUTINE make_plans
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION pick_plan( nb )
    !-----------------------------------------------------------------------
    INTEGER, INTENT(IN) :: nb
    INTEGER :: i
    pick_plan = 1
    DO i = 2, nplan
       IF ( plan_nb(i) >= nb ) pick_plan = i
    ENDDO
  END FUNCTION pick_plan
  !
  !-----------------------------------------------------------------------
  SUBROUTINE gemm_b( trans, m, n, k, a_dev, lda, stra, b_dev, ldb, strb, beta, c_dev, ldc, strc, nbatch )
    !-----------------------------------------------------------------------
    !! nbatch products C_i = op(A_i) B_i + beta C_i, constant strides
    INTEGER, INTENT(IN) :: trans, m, n, k, lda, ldb, ldc, nbatch
    INTEGER(8), INTENT(IN) :: stra, strb, strc
    COMPLEX(DP), INTENT(IN) :: beta
    COMPLEX(DP), DEVICE :: a_dev(*), b_dev(*), c_dev(*)
    INTEGER :: istat
    !
    IF ( nbatch <= 0 .OR. m <= 0 .OR. n <= 0 .OR. k <= 0 ) RETURN
    IF ( nbatch > 1 ) THEN
       istat = cublasZgemmStridedBatched( hblas, trans, CUBLAS_OP_N, m, n, k, c_one, a_dev, lda, &
                                          stra, b_dev, ldb, strb, beta, c_dev, ldc, strc, nbatch )
    ELSE
       istat = cublasZgemm( hblas, trans, CUBLAS_OP_N, m, n, k, c_one, a_dev, lda, b_dev, ldb, &
                            beta, c_dev, ldc )
    ENDIF
    IF ( istat /= 0 ) CALL errore( 'nscf_dev', 'cublas zgemm failed', ABS(istat) )
    !
  END SUBROUTINE gemm_b
  !
  !-----------------------------------------------------------------------
  SUBROUTINE copy_block( src_d, dst_d, ioff, cnt )
    !-----------------------------------------------------------------------
    !! dst(:,1:cnt,k) = src(:,ioff+1:ioff+cnt,k) for all k
    COMPLEX(DP), DEVICE, INTENT(IN) :: src_d(npwxl,nvecx,nk)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: dst_d(npwxl,nvec,nk)
    INTEGER, INTENT(IN) :: ioff, cnt
    INTEGER :: ik, n, ig, n1, n2, n3
    n1 = npwxl ; n2 = cnt ; n3 = nk
    !$cuf kernel do(3)
    DO ik = 1, n3
       DO n = 1, n2
          DO ig = 1, n1
             dst_d(ig,n,ik) = src_d(ig,ioff+n,ik)
          ENDDO
       ENDDO
    ENDDO
  END SUBROUTINE copy_block
  !
  !-----------------------------------------------------------------------
  SUBROUTINE put_block( src_d, dst_d, ioff, cnt )
    !-----------------------------------------------------------------------
    !! dst(:,ioff+1:ioff+cnt,k) = src(:,1:cnt,k) for all k
    COMPLEX(DP), DEVICE, INTENT(IN) :: src_d(npwxl,nvec,nk)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: dst_d(npwxl,nvecx,nk)
    INTEGER, INTENT(IN) :: ioff, cnt
    INTEGER :: ik, n, ig, n1, n2, n3
    n1 = npwxl ; n2 = cnt ; n3 = nk
    !$cuf kernel do(3)
    DO ik = 1, n3
       DO n = 1, n2
          DO ig = 1, n1
             dst_d(ig,ioff+n,ik) = src_d(ig,n,ik)
          ENDDO
       ENDDO
    ENDDO
  END SUBROUTINE put_block
  !
  !-----------------------------------------------------------------------
  SUBROUTINE scale_minus_e( npair )
    !-----------------------------------------------------------------------
    !! wrk(:,np,k) = - e(np,k) wrk(:,np,k) for the listed (np,k)
    INTEGER, INTENT(IN) :: npair
    INTEGER :: p, ig, ik, np, n1
    n1 = npwxl
    !$cuf kernel do(2)
    DO p = 1, npair
       DO ig = 1, n1
          ik = pk_d(p) ; np = pn_d(p)
          wrk_d(ig,np,ik) = - ew_d(np,ik) * wrk_d(ig,np,ik)
       ENDDO
    ENDDO
  END SUBROUTINE scale_minus_e
  !
  !-----------------------------------------------------------------------
  SUBROUTINE precond_normalize( npair )
    !-----------------------------------------------------------------------
    !! g_psi and normalization of the correction vectors (cegterg)
    INTEGER, INTENT(IN) :: npair
    INTEGER :: p, ig, ik, np, n1
    REAL(DP) :: x, denm, s
    n1 = npwxl
    !$acc parallel loop gang private(ik, np, s)
    DO p = 1, npair
       ik = pk_d(p) ; np = pn_d(p)
       s = 0.0_DP
       !$acc loop vector reduction(+:s) private(x, denm)
       DO ig = 1, n1
          IF ( ig <= npw_d(ik) ) THEN
             x = hd_d(ig,ik) - ew_d(np,ik) * sd_d(ig,ik)
             denm = 0.5_DP * ( 1.0_DP + x + SQRT( 1.0_DP + (x-1.0_DP)*(x-1.0_DP) ) )
             wrk_d(ig,np,ik) = wrk_d(ig,np,ik) / denm
             s = s + DBLE( CONJG(wrk_d(ig,np,ik)) * wrk_d(ig,np,ik) )
          ELSE
             wrk_d(ig,np,ik) = (0.0_DP, 0.0_DP)
          ENDIF
       ENDDO
       nrm_d(np,ik) = s
    ENDDO
    !$cuf kernel do(2)
    DO p = 1, npair
       DO ig = 1, n1
          ik = pk_d(p) ; np = pn_d(p)
          wrk_d(ig,np,ik) = wrk_d(ig,np,ik) / SQRT( nrm_d(np,ik) )
       ENDDO
    ENDDO
  END SUBROUTINE precond_normalize
  !
  !-----------------------------------------------------------------------
  SUBROUTINE hs_apply( npair, nrun, run1, run2 )
    !-----------------------------------------------------------------------
    !! hwrk = H wrk and (ultrasoft) swrk = S wrk for the listed vectors
    !! (np,k) of wrk; the nonlocal part is computed for all vectors of the
    !! k-points in the runs run1(i) ... run2(i).
    INTEGER, INTENT(IN) :: npair, nrun, run1(nrun), run2(nrun)
    INTEGER :: j0, nb, ip, j, ig, ir, ik, np, istat, irun, k1, k2, ikb, na, iof, ih, jh, is, &
               n1, n2, nkl
    REAL(DP) :: scale
    COMPLEX(DP) :: s, sq
    !
    scale = 1.0_DP / DBLE(nnr)
    n1 = npwxl ; n2 = nvec ; nkl = nkbl
    DO j0 = 0, npair-1, nfb
       nb = MIN( nfb, npair-j0 )
       ip = pick_plan( nb )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, n1
             ik = pk_d(j0+j) ; np = pn_d(j0+j)
             IF ( ig <= npw_d(ik) ) psig_d(nl_d(ig,ik) + (j-1)*nnr) = wrk_d(ig,np,ik)
          ENDDO
       ENDDO
       istat = cufftExecZ2Z( plan(ip), psig_d, psic_d, CUFFT_INVERSE )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, n1
             ik = pk_d(j0+j)
             IF ( ig <= npw_d(ik) ) psig_d(nl_d(ig,ik) + (j-1)*nnr) = (0.0_DP, 0.0_DP)
          ENDDO
       ENDDO
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ir = 1, nnr
             ik = pk_d(j0+j)
             psic_d(ir + (j-1)*nnr) = psic_d(ir + (j-1)*nnr) * v_d(ir,spin_d(ik))
          ENDDO
       ENDDO
       istat = cufftExecZ2Z( plan(ip), psic_d, psic_d, CUFFT_FORWARD )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, n1
             ik = pk_d(j0+j) ; np = pn_d(j0+j)
             IF ( ig <= npw_d(ik) ) THEN
                hwrk_d(ig,np,ik) = g2_d(ig,ik) * wrk_d(ig,np,ik) + &
                                   psic_d(nl_d(ig,ik) + (j-1)*nnr) * scale
             ELSE
                hwrk_d(ig,np,ik) = (0.0_DP, 0.0_DP)
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    IF ( lus ) THEN
       !$cuf kernel do(2)
       DO j = 1, npair
          DO ig = 1, n1
             ik = pk_d(j) ; np = pn_d(j)
             swrk_d(ig,np,ik) = wrk_d(ig,np,ik)
          ENDDO
       ENDDO
    ENDIF
    IF ( nkbl == 0 ) RETURN
    !
    DO irun = 1, nrun
       k1 = run1(irun) ; k2 = run2(irun)
       CALL gemm_b( CUBLAS_OP_C, nkbl, nvec, npwxl, vkb_d(1,1,k1), npwxl, INT(npwxl,8)*nkbl, &
                    wrk_d(1,1,k1), npwxl, INT(npwxl,8)*nvec, c_zero, bw_d(1,1,k1), nkbl, &
                    INT(nkbl,8)*nvec, k2-k1+1 )
       !$cuf kernel do(3)
       DO ik = k1, k2
          DO np = 1, n2
             DO ikb = 1, nkl
                na = kbna_d(ikb) ; iof = kbof_d(ikb) ; ih = ikb - iof
                is = spin_d(ik)
                s = (0.0_DP, 0.0_DP) ; sq = (0.0_DP, 0.0_DP)
                DO jh = 1, kbnh_d(ikb)
                   s = s + deeq_d(ih,jh,na,is) * bw_d(iof+jh,np,ik)
                   sq = sq + qq_d(ih,jh,na) * bw_d(iof+jh,np,ik)
                ENDDO
                bw2_d(ikb,np,ik) = s
                bw3_d(ikb,np,ik) = sq
             ENDDO
          ENDDO
       ENDDO
       CALL gemm_b( CUBLAS_OP_N, npwxl, nvec, nkbl, vkb_d(1,1,k1), npwxl, INT(npwxl,8)*nkbl, &
                    bw2_d(1,1,k1), nkbl, INT(nkbl,8)*nvec, c_one, hwrk_d(1,1,k1), npwxl, &
                    INT(npwxl,8)*nvec, k2-k1+1 )
       IF ( lus ) CALL gemm_b( CUBLAS_OP_N, npwxl, nvec, nkbl, vkb_d(1,1,k1), npwxl, &
                    INT(npwxl,8)*nkbl, bw3_d(1,1,k1), nkbl, INT(nkbl,8)*nvec, c_one, &
                    swrk_d(1,1,k1), npwxl, INT(npwxl,8)*nvec, k2-k1+1 )
    ENDDO
    !
  END SUBROUTINE hs_apply
  !
  !-----------------------------------------------------------------------
  SUBROUTINE append_block( npair )
    !-----------------------------------------------------------------------
    !! psi(:,nbase(k)+np,k) = wrk(:,np,k), same for H psi and S psi
    INTEGER, INTENT(IN) :: npair
    INTEGER :: p, ig, ik, np, n1
    n1 = npwxl
    IF ( lus ) THEN
       !$cuf kernel do(2)
       DO p = 1, npair
          DO ig = 1, n1
             ik = pk_d(p) ; np = pn_d(p)
             psi_d(ig,nbase_d(ik)+np,ik) = wrk_d(ig,np,ik)
             hpsi_d(ig,nbase_d(ik)+np,ik) = hwrk_d(ig,np,ik)
             spsi_d(ig,nbase_d(ik)+np,ik) = swrk_d(ig,np,ik)
          ENDDO
       ENDDO
    ELSE
       !$cuf kernel do(2)
       DO p = 1, npair
          DO ig = 1, n1
             ik = pk_d(p) ; np = pn_d(p)
             psi_d(ig,nbase_d(ik)+np,ik) = wrk_d(ig,np,ik)
             hpsi_d(ig,nbase_d(ik)+np,ik) = hwrk_d(ig,np,ik)
          ENDDO
       ENDDO
    ENDIF
  END SUBROUTINE append_block
  !
  !-----------------------------------------------------------------------
  SUBROUTINE refresh_block( )
    !-----------------------------------------------------------------------
    !! for the flagged k-points: psi(:,1:nvec) = wrk, H psi = hwrk, S psi = swrk
    INTEGER :: ik, n, ig, n1, n2, n3
    n1 = npwxl ; n2 = nvec ; n3 = nk
    IF ( lus ) THEN
       !$cuf kernel do(3)
       DO ik = 1, n3
          DO n = 1, n2
             DO ig = 1, n1
                IF ( flag_d(ik) == 1 ) THEN
                   psi_d(ig,n,ik) = wrk_d(ig,n,ik)
                   hpsi_d(ig,n,ik) = hwrk_d(ig,n,ik)
                   spsi_d(ig,n,ik) = swrk_d(ig,n,ik)
                ENDIF
             ENDDO
          ENDDO
       ENDDO
    ELSE
       !$cuf kernel do(3)
       DO ik = 1, n3
          DO n = 1, n2
             DO ig = 1, n1
                IF ( flag_d(ik) == 1 ) THEN
                   psi_d(ig,n,ik) = wrk_d(ig,n,ik)
                   hpsi_d(ig,n,ik) = hwrk_d(ig,n,ik)
                ENDIF
             ENDDO
          ENDDO
       ENDDO
    ENDIF
  END SUBROUTINE refresh_block
#endif
  !
END MODULE nscf_dev

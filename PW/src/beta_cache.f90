!
! Copyright (C) 2026 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
MODULE beta_cache
  !-----------------------------------------------------------------------
  !! Beta functions of all the k-points of the current list, computed in
  !! blocks of k-points on the GPU and kept there; init_us_2 serves a
  !! request for one of these k-points with a copy.
  !!
  !! ph.x asks for the beta functions of every k and k+q several times per
  !! q-point (phq_init, compute_becalp, compute_drhous, the set-up of the
  !! solver, the dynamical-matrix routines); init_us_2 costs about 0.13 ms
  !! per call whatever the size, which for thousands of k-points is seconds.
  !! The values are those of init_us_2_acc (same expressions, element by
  !! element). The cache is built on request (beta_cache_build) and is used
  !! only for requests that match one of its k-points exactly (k, number of
  !! plane waves, first and last index of the list of G-vectors), so it can
  !! only return what init_us_2_acc would compute as long as the atomic
  !! positions and the cell do not change: free it (beta_cache_free) when
  !! they do. QE_BETA_CACHE=0 disables it.
  !
  USE kinds, ONLY : DP
#if defined(__CUDA)
  USE cudafor
#endif
  !
  IMPLICIT NONE
  PRIVATE
  SAVE
  !
  PUBLIC :: beta_cache_build, beta_cache_free, beta_cache_get, beta_cache_on, beta_cache_index
  !
  LOGICAL :: beta_cache_on = .FALSE.
  !
#if defined(__CUDA)
  PUBLIC :: bc_d
  COMPLEX(DP), ALLOCATABLE, DEVICE :: bc_d(:,:,:)
  !! (npwx, nkb, nks): beta functions of the k-points (see beta_cache_index)
  INTEGER, ALLOCATABLE :: npw_c(:), ig1_c(:), ign_c(:)
  REAL(DP), ALLOCATABLE :: xk_c(:,:)
  INTEGER :: nkc = 0, npwxc = 0, nkbc = 0, hint = 1
#endif
  !
CONTAINS
  !
  !-----------------------------------------------------------------------
  SUBROUTINE beta_cache_free( )
    !-----------------------------------------------------------------------
#if defined(__CUDA)
    IF ( ALLOCATED(bc_d) ) DEALLOCATE( bc_d )
    IF ( ALLOCATED(npw_c) ) DEALLOCATE( npw_c, ig1_c, ign_c, xk_c )
    nkc = 0
#endif
    beta_cache_on = .FALSE.
  END SUBROUTINE beta_cache_free
  !
  !-----------------------------------------------------------------------
  SUBROUTINE beta_cache_build( )
    !-----------------------------------------------------------------------
    !! Beta functions of the k-points 1 ... nks of klist (init_us_2_acc for
    !! each, in blocks). Nothing is done (and the cache stays off) with fewer
    !! than 8 k-points or when the array would take more than 4 GB or a
    !! tenth of the free device memory.
    !
#if defined(__CUDA)
    USE upf_const,     ONLY : tpi
    USE cell_base,     ONLY : tpiba, omega
    USE ions_base,     ONLY : nat, ityp, tau
    USE klist,         ONLY : nks, xk, ngk, igk_k
    USE gvect,         ONLY : g, mill, eigts1, eigts2, eigts3
    USE wvfct,         ONLY : npwx
    USE uspp,          ONLY : nkb, nhtol, nhtolm, indv
    USE uspp_param,    ONLY : nh, nhm, lmaxkb, nbetam, nsp
    USE beta_mod,      ONLY : interp_beta
    USE control_flags, ONLY : use_gpu
    !
    INTEGER :: ik, ig, ip, iv, k1, k2, kb, kbmax, npb, nt, na, nb, ih, lm, nhnt, nat_t, ia, jkb, &
               lmm, n1, kk, istat, ln, ierr
    REAL(DP) :: need, tpiba_l
    COMPLEX(DP) :: sk, pref
    CHARACTER(LEN=8) :: val
    INTEGER(KIND=cuda_count_kind) :: mfree, mtot
    INTEGER, ALLOCATABLE :: na_t(:), npw_h(:)
    REAL(DP), ALLOCATABLE :: gk(:,:), qg(:), ylm(:,:), vq(:,:), vkb1(:,:), arg_h(:,:)
    INTEGER, ALLOCATABLE, DEVICE :: npw_d(:)
    REAL(DP), ALLOCATABLE, DEVICE :: xk_d(:,:)
#endif
    !
    CALL beta_cache_free( )
#if defined(__CUDA)
    CALL get_environment_variable( 'QE_BETA_CACHE', val, ln, ierr )
    IF ( ierr == 0 .AND. ln > 0 .AND. val(1:1) == '0' ) RETURN
    IF ( .NOT. use_gpu .OR. nkb < 1 .OR. lmaxkb < 0 .OR. nks < 8 ) RETURN
    need = 16.0_DP * REAL(npwx,DP) * nkb * nks
    istat = cudaMemGetInfo( mfree, mtot )
    IF ( need > 4.0E9_DP .OR. need > 0.1_DP * REAL(mfree,DP) ) RETURN
    CALL start_clock( 'beta_cache' )
    !
    n1 = npwx ; tpiba_l = tpiba
    lmm = (lmaxkb+1)**2
    ALLOCATE( bc_d(npwx,nkb,nks), npw_d(nks), xk_d(3,nks), npw_h(nks) )
    bc_d = (0.0_DP, 0.0_DP)
    npw_h(:) = ngk(1:nks)
    npw_d = npw_h
    xk_d = xk(:,1:nks)
    kbmax = MAX( 1, MIN( nks, 1048576 / npwx ) )
    ALLOCATE( na_t(nat), arg_h(nat,kbmax) )
    !
    DO k1 = 1, nks, kbmax
       k2 = MIN( nks, k1+kbmax-1 )
       kb = k2 - k1 + 1
       npb = npwx * kb
       ALLOCATE( gk(3,npb), qg(npb), ylm(npb,lmm), vq(npb,MAX(nbetam,1)), vkb1(npb,MAX(nhm,1)) )
       !$acc data create(gk, qg, ylm, vq, vkb1) present(g, igk_k, mill, eigts1, eigts2, eigts3)
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
                         bc_d(ig,jkb+(ia-1)*nhnt+ih,ik) = vkb1(ip,ih) * sk * pref
                      ENDDO
                   ENDIF
                ENDDO
             ENDDO
          ENDDO
          jkb = jkb + nat_t*nhnt
       ENDDO
       !$acc end data
       DEALLOCATE( gk, qg, ylm, vq, vkb1 )
    ENDDO
    DEALLOCATE( na_t, arg_h, npw_d, xk_d )
    !
    nkc = nks ; npwxc = npwx ; nkbc = nkb ; hint = 1
    ALLOCATE( npw_c(nks), ig1_c(nks), ign_c(nks), xk_c(3,nks) )
    npw_c(:) = npw_h(:)
    xk_c(:,:) = xk(:,1:nks)
    DO ik = 1, nks
       ! (a rank may have no plane waves of a k-point: such an entry never matches)
       ig1_c(ik) = 0 ; ign_c(ik) = 0
       IF ( ngk(ik) >= 1 ) THEN
          ig1_c(ik) = igk_k(1,ik)
          ign_c(ik) = igk_k(ngk(ik),ik)
       ENDIF
    ENDDO
    DEALLOCATE( npw_h )
    beta_cache_on = .TRUE.
    CALL stop_clock( 'beta_cache' )
#endif
    !
  END SUBROUTINE beta_cache_build
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION beta_cache_index( npw_, npwx_, nkb_, igk_, q_ )
    !-----------------------------------------------------------------------
    !! Position in bc_d of the k-point (q_, npw_, igk_), 0 if it is not there.
    INTEGER, INTENT(IN) :: npw_, npwx_, nkb_
    INTEGER, INTENT(IN) :: igk_(npw_)
    REAL(DP), INTENT(IN) :: q_(3)
#if defined(__CUDA)
    INTEGER :: ik, itry
#endif
    beta_cache_index = 0
#if defined(__CUDA)
    IF ( .NOT. beta_cache_on .OR. npwx_ /= npwxc .OR. nkb_ /= nkbc .OR. npw_ < 1 ) RETURN
    DO itry = 1, 4
       ik = hint + itry - 2
       IF ( ik < 1 .OR. ik > nkc ) CYCLE
       IF ( match( ik ) ) THEN
          beta_cache_index = ik
          hint = ik
          RETURN
       ENDIF
    ENDDO
    DO ik = 1, nkc
       IF ( match( ik ) ) THEN
          beta_cache_index = ik
          hint = ik
          RETURN
       ENDIF
    ENDDO
  CONTAINS
    LOGICAL FUNCTION match( jk )
      INTEGER, INTENT(IN) :: jk
      match = npw_c(jk) == npw_
      IF ( .NOT. match ) RETURN
      match = xk_c(1,jk) == q_(1) .AND. xk_c(2,jk) == q_(2) .AND. xk_c(3,jk) == q_(3) .AND. &
              ig1_c(jk) == igk_(1) .AND. ign_c(jk) == igk_(npw_)
    END FUNCTION match
#endif
  END FUNCTION beta_cache_index
  !
  !-----------------------------------------------------------------------
  SUBROUTINE beta_cache_get( npw_, npwx_, nkb_, igk_, q_, vkb_, found )
    !-----------------------------------------------------------------------
    !! If (q_, npw_, igk_) is a k-point of the cache, its beta functions are
    !! copied to the device copy of vkb_ (which the caller has in an OpenACC
    !! data region) and found = .TRUE.
    !
    INTEGER, INTENT(IN) :: npw_, npwx_, nkb_
    INTEGER, INTENT(IN) :: igk_(npw_)
    REAL(DP), INTENT(IN) :: q_(3)
    COMPLEX(DP), INTENT(INOUT) :: vkb_(npwx_,nkb_)
    LOGICAL, INTENT(OUT) :: found
    !
#if defined(__CUDA)
    INTEGER :: ig, ikb, ikh, n1, n2
#endif
    !
    found = .FALSE.
#if defined(__CUDA)
    ikh = beta_cache_index( npw_, npwx_, nkb_, igk_, q_ )
    IF ( ikh == 0 ) RETURN
    n1 = npwxc ; n2 = nkbc
    !$acc parallel loop collapse(2) present(vkb_)
    DO ikb = 1, n2
       DO ig = 1, n1
          vkb_(ig,ikb) = bc_d(ig,ikb,ikh)
       ENDDO
    ENDDO
    found = .TRUE.
#endif
    !
  END SUBROUTINE beta_cache_get
  !
END MODULE beta_cache

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

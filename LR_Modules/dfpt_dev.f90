!
! Copyright (C) 2026 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
MODULE dfpt_dev
  !-----------------------------------------------------------------------
  !! Device-resident Sternheimer engine for DFPT (ph.x) on GPUs.
  !!
  !! The stock sternheimer_kernel loops over k-points and perturbations and
  !! solves one small linear system at a time, band by band, moving
  !! \(\Delta V\psi\), \(\Delta\psi\) and \(\Delta\rho\) between host and
  !! device at every step. Here everything that the solver needs for one
  !! q-point (\(\psi_k\), \(\psi_{k+q}\), \(\beta_{k+q}\), \(|k+q+G|^2\),
  !! FFT maps) is uploaded once and kept on the device, and the linear
  !! systems of all k-points, perturbations and bands are solved together:
  !! every step of the conjugate-gradient iteration is a handful of kernels
  !! and batched cuFFT / cuBLAS calls over all the columns that are not yet
  !! converged.
  !!
  !! Column layout: column c = ((ik-1)*npe + (ipert-1))*nbo + ibnd, with
  !! nbo = max number of occupied bands. Columns of bands above
  !! nbnd_occ(k) are kept at zero and never enter the active list.
  !!
  !! The algorithm (preconditioned CG of cgsolve_all, operator of
  !! ch_psi_all, right-hand side of apply_dpot_bands + adddvscf +
  !! orthogonalize, response density of incdrhoscf + addusdbec) is the
  !! stock one, so iteration counts and results agree with the stock path
  !! to rounding.
  !!
  !! Served: collinear (nspin = 1, 2), norm-conserving, ultrasoft and PAW,
  !! insulators and metals, any q. Not served (stock path is used):
  !! noncollinear, DFT+U, two chemical potentials, finite frequency, an FFT
  !! grid distributed over MPI ranks, band groups, task groups.
  !
  USE kinds, ONLY : DP, SP => sgl
  USE dfpt_dev_grid, ONLY : dfpt_dev_grid_reset
#if defined(__CUDA)
  USE cudafor
  USE cufft
  USE cublas_v2
  USE cusolverDn
#endif
  !
  IMPLICIT NONE
  PRIVATE
  SAVE
  !
  PUBLIC :: dfpt_dev_end, dfpt_dev_flush, dfpt_dev_invalidate, &
            dfpt_dev_in_session
  PUBLIC :: dfpt_dev_set_eta, dfpt_dev_set_work
  PUBLIC :: dfpt_dev_keep_dpsi, dfpt_dev_agree, dfpt_dev_keeping, dfpt_dev_kept
#if defined(__CUDA)
  ! ... for callers that write the bare dV|psi> straight into the engine (see dfpt_dev_prefill_begin)
  PUBLIC :: dvb_d, psik_d, w_d, nbo, npwxl
#endif
  !
  LOGICAL :: q_ready = .FALSE.
  !! data of the current q-point are on the device
  LOGICAL :: in_session = .FALSE.
  !! between dfpt_dev_begin and dfpt_dev_end
  LOGICAL :: dpsi_dirty = .FALSE.
  !! the device dpsi is newer than the one in the buffer iudwf
  LOGICAL :: lin_session = .FALSE.
  !! the session is a plain linear solve (dfpt_dev_lin_*): nothing goes to iudwf
  LOGICAL :: keep_next = .FALSE.
  !! at the end of the session dpsi stays on the device (dfpt_dev_get_bq) and is
  !! not copied to the buffer iudwf
  LOGICAL :: dpsi_kept = .FALSE.
  LOGICAL :: keep_veto = .FALSE.
  !! metals at q = 0: the Fermi-energy shift updates dpsi in the buffer, never kept
  INTEGER :: npe_kept = 0
  LOGICAL :: prefilled = .FALSE.
  !! the session was opened by dfpt_dev_prefill_begin: dvb_d already holds the bare dV|psi>
  INTEGER :: nmodes_hint = 0
  !! number of modes that the run solves for at the current q-point (0: unknown, 3*nat is
  !! assumed); sizes the low-G block of the preconditioner, see dfpt_dev_set_work
  REAL(DP) :: eta_rel = 0.0_DP
  !! if > 0, a linear system also counts as solved when its residual has
  !! dropped by this factor in the current pass (inexact inner solves of
  !! the early self-consistency iterations, see dfpt_dev_set_eta)
  !
#if defined(__CUDA)
  !
  COMPLEX(DP), PARAMETER :: c_one = (1.0_DP, 0.0_DP), c_zero = (0.0_DP, 0.0_DP), &
                            c_mone = (-1.0_DP, 0.0_DP)
  INTEGER, PARAMETER :: maxter = 400
  !! maximum number of CG iterations (as in cgsolve_all)
  !
  ! ... dimensions
  !
  INTEGER :: nks = 0     ! number of k-points (nksq)
  INTEGER :: nbo = 0     ! columns per (k, perturbation): max nbnd_occ
  INTEGER :: nbq = 0     ! bands kept at k+q (projectors of P_c and P_v)
  INTEGER :: nw = 0      ! nkb + nbq: columns of w_d
  INTEGER :: nkbl = 0    ! nkb
  INTEGER :: npwxl = 0   ! npwx
  INTEGER :: nnr = 0, nr1 = 0, nr2 = 0, nr3 = 0
  INTEGER :: nspl = 1    ! nspin (1 or 2)
  INTEGER :: nhml = 0, natl = 0
  INTEGER :: npe = 0     ! perturbations of the session
  INTEGER :: ncolk = 0   ! columns per k: npe*nbo
  INTEGER :: ncol = 0    ! columns: ncolk*nks
  INTEGER :: nlive = 0   ! columns of occupied bands
  INTEGER :: nfb = 0     ! FFTs per batch
  LOGICAL :: lus = .FALSE.     ! ultrasoft / PAW
  LOGICAL :: metal = .FALSE.
  LOGICAL :: have_psir = .FALSE.   ! psi_k(r) of the occupied bands is cached
  LOGICAL :: batched_gemm = .TRUE.
  LOGICAL :: use_z = .FALSE.
  !! spectral preconditioner: the empty bands at k+q kept in w_d / evq_d
  !! (columns zoff+1 ... nbq) are inverted exactly in the preconditioner
  INTEGER :: zoff = 0, nz = 0
  INTEGER :: iudvb = -1, lrdvb = -1
  REAL(DP) :: alpha_pv_l = 0.0_DP
  LOGICAL :: lgamma_l = .FALSE.
  !
  ! ... host copies of per-k data
  !
  INTEGER, ALLOCATABLE :: npw_h(:), npwq_h(:), nocc_h(:), noccq_h(:), spin_h(:)
  REAL(DP), ALLOCATABLE :: wk_h(:)
  !
  ! ... q-level device data
  !
  INTEGER, ALLOCATABLE, DEVICE :: npw_d(:), npwq_d(:), nocc_d(:), noccq_d(:), spin_d(:)
  INTEGER, ALLOCATABLE, DEVICE :: nlk_d(:,:), nlq_d(:,:)
  !! FFT index of k+G and of k+q+G
  REAL(DP), ALLOCATABLE, DEVICE :: g2_d(:,:)
  !! |k+q+G|^2
  REAL(DP), ALLOCATABLE, DEVICE :: kqg_d(:,:,:)
  !! k+q+G (3, npwx, nks), units 2pi/a
  COMPLEX(DP), ALLOCATABLE, DEVICE :: psik_d(:,:,:)
  !! psi_k, occupied bands (npwx, nbo, nks)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: w_d(:,:,:)
  !! (npwx, nw, nks): beta functions at k+q, then S*psi_{k+q}
  COMPLEX(DP), ALLOCATABLE, DEVICE :: evq_d(:,:,:)
  !! psi_{k+q} (npwx, nbq, nks); allocated with ultrasoft PPs only (else S*psi = psi)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: psir_d(:,:,:)
  !! psi_k(r), occupied bands (nnr, nbo, nks)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: becp1_d(:,:,:)
  !! <beta_k|psi_k> (nkb, nbo, nks), ultrasoft only
  REAL(DP), ALLOCATABLE, DEVICE :: et_d(:,:), eprec_d(:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: etq_d(:,:)
  !! eigenvalues at k+q of the bands kept there (nbq, nks)
  REAL(DP), ALLOCATABLE, DEVICE :: wwg_d(:,:,:), wg1_d(:,:)
  !! weights of the projection on the valence manifold (orthogonalize)
  REAL(DP), ALLOCATABLE, DEVICE :: deeq_d(:,:,:,:), qq_d(:,:,:)
  INTEGER, ALLOCATABLE, DEVICE :: kbna_d(:), kbof_d(:), kbnh_d(:)
  !! for each beta function: atom, offset of the atom, number of beta functions of the atom
  REAL(DP), ALLOCATABLE, DEVICE :: v_d(:,:)
  !! local potential vrs
  REAL(DP), ALLOCATABLE, DEVICE :: wgt_d(:)
  !! 2 wk / omega
  !
  ! ... session-level device data
  !
  COMPLEX(DP), ALLOCATABLE, DEVICE :: dvb_d(:,:), dpsi_d(:,:), g_d(:,:), h_d(:,:), t_d(:,:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: cw_d(:,:), cw2_d(:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: e_d(:), ep_d(:), rho_d(:), rhoold_d(:), a_d(:), c_d(:), rhoc_d(:)
  INTEGER, ALLOCATABLE, DEVICE :: act_d(:), live_d(:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: dv_d(:,:,:), drho_d(:,:,:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: int3_d(:,:,:,:,:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: psig_d(:), psic_d(:), psi2_d(:)
  !
  INTEGER, ALLOCATABLE :: live_h(:), act_h(:)
  !
  ! ... mixed precision: the conjugate-gradient iterations run in single
  ! ... precision on the residual computed in double precision
  !
  LOGICAL :: use_sp = .FALSE.
  COMPLEX(SP), ALLOCATABLE, DEVICE :: ws_d(:,:,:), evqs_d(:,:,:)
  COMPLEX(SP), ALLOCATABLE, DEVICE :: gs_d(:,:), hs_d(:,:), ts_d(:,:), cws_d(:,:), cw2s_d(:,:)
  COMPLEX(SP), ALLOCATABLE, DEVICE :: hcs_d(:,:), tcs_d(:,:)
  COMPLEX(SP), ALLOCATABLE, DEVICE :: psigs_d(:), psics_d(:)
  INTEGER :: plans(8) = 0
  !
  ! ... preconditioner with an exact low-G block: see setup_lowg
  !
  LOGICAL :: use_lg = .FALSE.
  INTEGER :: nlg = 0
  !! largest number of plane waves in the block
  INTEGER, ALLOCATABLE, DEVICE :: lgn_d(:), lgi_d(:,:)
  !! per k: number of plane waves of the block, their indices in the list of k+q+G
  COMPLEX(SP), ALLOCATABLE, DEVICE :: lgv_d(:,:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: lge_d(:,:), hdg_d(:,:), sdg_d(:,:)
  !! eigenvectors and eigenvalues of the block; diagonal of H + alpha P_v and of S
  COMPLEX(SP), ALLOCATABLE, DEVICE :: gl_d(:,:), cl_d(:,:)
  !
  ! ... the same in the real mode (setup_lowg_real): block on the pairs of low kinetic energy
  !
  LOGICAL :: use_lgr = .FALSE.
  INTEGER :: nlb = 0
  !! largest size of the block (two numbers per pair)
  INTEGER, ALLOCATABLE, DEVICE :: lgnr_d(:), lgpr_d(:,:)
  !! per k: size of the block, pairs of the block
  REAL(SP), ALLOCATABLE, DEVICE :: lgvr_d(:,:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: lger_d(:,:), hdr_d(:,:), sdr_d(:,:)
  REAL(SP), ALLOCATABLE, DEVICE :: glr_d(:,:), clr_d(:,:)
  INTEGER, ALLOCATABLE :: rnh_h(:), rns_h(:), rhl_h(:,:), rhm_h(:,:)
  REAL(DP), ALLOCATABLE :: rg2_h(:,:)
  !! host copies of the pair lists of the real mode
  TYPE(cusolverDnHandle) :: hsolv
  LOGICAL :: solv_ready = .FALSE.
  !
  ! ... real mode (q = 0, every k-point equivalent to -k): see setup_real
  !
  LOGICAL :: use_real = .FALSE.
  !! the q-level data of the real mode are ready
  LOGICAL :: real_on = .FALSE.
  !! the session has the work arrays of the real mode
  LOGICAL :: real_skip = .FALSE.
  !! set while a solve that the real mode gave up is repeated with complex iterations
  REAL(DP) :: eps_real = 0.0_DP
  !! largest norm of the imaginary part of the rotated bands
  INTEGER :: nhx = 0
  !! largest number of pairs (G, -G-2k) of a k-point
  INTEGER, ALLOCATABLE :: clus_h(:,:)
  !! first band of the group of degenerate bands that each band belongs to (nbo, nks)
  REAL(DP), ALLOCATABLE :: erot_h(:,:)
  !! eigenvalues of the rotated bands (nbo, nks)
  INTEGER, ALLOCATABLE, DEVICE :: nh_d(:), nself_d(:), hl_d(:,:), hm_d(:,:), nlh_d(:,:), nlm_d(:,:)
  !! per k: number of pairs, of self-conjugate G (0 or 1, listed first); for each pair
  !! the indices of G and of its partner in the list of k+G and on the FFT grid
  REAL(DP), ALLOCATABLE, DEVICE :: g2h_d(:,:), er_d(:)
  !! |k+G|^2 of the pairs; eigenvalue of each rotated column
  REAL(SP), ALLOCATABLE, DEVICE :: wr_d(:,:,:)
  !! (2*nhx, nw, nks): beta functions and S*psi (rotated bands) as real vectors
  COMPLEX(SP), ALLOCATABLE, DEVICE :: ur_d(:,:,:)
  !! (nbo, nbo, nks): rotation of the bands that makes them real functions
  REAL(SP), ALLOCATABLE, DEVICE :: gr_d(:,:), hr_d(:,:), tr_d(:,:), dr_d(:,:)
  !! (2*nhx, 2*ncol): gradient, direction, A*direction, correction; column c holds the
  !! real part and column ncol+c the imaginary part (as functions of r) of column c
  REAL(SP), ALLOCATABLE, DEVICE :: hcr_d(:,:), tcr_d(:,:), cwr_d(:,:), cw2r_d(:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: rhor_d(:), rhooldr_d(:), ar_d(:), cr_d(:), rhocr_d(:)
  INTEGER, ALLOCATABLE, DEVICE :: actr_d(:), pa_d(:,:)
  INTEGER, ALLOCATABLE :: actr_h(:), pa_h(:,:)
  !
  ! ... cuFFT plans (batch sizes nfb, nfb/4, nfb/16, ...) and cuBLAS handle
  !
  INTEGER, PARAMETER :: nplan_max = 8
  INTEGER :: nplan = 0
  INTEGER :: plan(nplan_max) = 0, plan_nb(nplan_max) = 0
  TYPE(cublasHandle) :: hblas
  LOGICAL :: blas_ready = .FALSE.
  !
#endif
  !
CONTAINS
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dfpt_dev_set_eta( eta )
    !-----------------------------------------------------------------------
    !! Relative tolerance of the next Sternheimer passes (0: only the
    !! absolute threshold of the caller counts, as in cgsolve_all).
    REAL(DP), INTENT(IN) :: eta
    eta_rel = MAX( 0.0_DP, MIN( eta, 0.5_DP ) )
  END SUBROUTINE dfpt_dev_set_eta
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dfpt_dev_set_work( nmodes )
    !-----------------------------------------------------------------------
    !! Number of perturbations for which the caller is going to solve at the
    !! current q-point (what one-time set-up work may cost is judged by it).
    INTEGER, INTENT(IN) :: nmodes
    nmodes_hint = MAX( 0, nmodes )
  END SUBROUTINE dfpt_dev_set_work
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dfpt_dev_agree( ok )
    !-----------------------------------------------------------------------
    !! The decision to serve a session is taken by each MPI rank (device
    !! memory). The device path and the stock path do not issue the same
    !! collective MPI calls, so all the ranks of the image must take the
    !! same one: a session that is not open on all of them is closed.
    !! To be called by all ranks after an attempt to open a session;
    !! ok = the session is open (everywhere).
    !
    USE mp,        ONLY : mp_sum
    USE mp_images, ONLY : intra_image_comm, nproc_image
    !
    LOGICAL, INTENT(OUT) :: ok
    INTEGER :: n
    !
    n = 0
    IF ( in_session ) n = 1
    CALL mp_sum( n, intra_image_comm )
    IF ( n /= nproc_image .AND. in_session ) THEN
       keep_next = .FALSE.
       dpsi_dirty = .FALSE.
       CALL dfpt_dev_end( )
    ENDIF
    ok = in_session
    !
  END SUBROUTINE dfpt_dev_agree
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION dfpt_dev_keeping( )
    !-----------------------------------------------------------------------
    !! True if the response wavefunctions of the current session will stay
    !! on the device and not be written to the buffer iudwf at its end.
    dfpt_dev_keeping = in_session .AND. keep_next .AND. .NOT. lin_session .AND. &
                       .NOT. keep_veto
  END FUNCTION dfpt_dev_keeping
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION dfpt_dev_kept( nper )
    !-----------------------------------------------------------------------
    !! True if dfpt_dev_get_bq can serve nper perturbations.
    INTEGER, INTENT(IN) :: nper
    dfpt_dev_kept = dpsi_kept .AND. nper == npe_kept
  END FUNCTION dfpt_dev_kept
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION dfpt_dev_in_session( )
    !-----------------------------------------------------------------------
    dfpt_dev_in_session = in_session .AND. .NOT. lin_session
  END FUNCTION dfpt_dev_in_session
  !
#if defined(__CUDA)
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION dev_requested( )
    !-----------------------------------------------------------------------
    !! The engine is on by default; QE_PH_DEV=0 selects the stock path.
    CHARACTER(LEN=16) :: val
    INTEGER :: ln, ierr
    CALL get_environment_variable( 'QE_PH_DEV', val, ln, ierr )
    dev_requested = .NOT. ( ierr == 0 .AND. ln > 0 .AND. val(1:1) == '0' )
  END FUNCTION dev_requested
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION dev_supported( )
    !-----------------------------------------------------------------------
    USE noncollin_module, ONLY : noncolin
    USE ldaU,             ONLY : lda_plus_u
    USE two_chem,         ONLY : twochem
    USE control_flags,    ONLY : gamma_only, use_gpu, scissor
    USE realus,           ONLY : real_space
    USE xc_lib,           ONLY : exx_is_active, xclib_dft_is
    USE mp_bands,         ONLY : nproc_bgrp, nbgrp
    USE fft_base,         ONLY : dffts
    USE bp,               ONLY : lelfield
    !
    dev_supported = use_gpu .AND. .NOT. ( noncolin .OR. lda_plus_u .OR. twochem .OR. &
         gamma_only .OR. real_space .OR. scissor .OR. lelfield .OR. exx_is_active() .OR. &
         xclib_dft_is('meta') .OR. dffts%has_task_groups ) .AND. nproc_bgrp == 1 .AND. &
         nbgrp == 1 .AND. dffts%nnr == dffts%nr1*dffts%nr2*dffts%nr3 .AND. &
         dffts%nr1 == dffts%nr1x .AND. dffts%nr2 == dffts%nr2x .AND. dffts%nr3 == dffts%nr3x
    !
  END FUNCTION dev_supported
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dfpt_dev_invalidate( )
    !-----------------------------------------------------------------------
    !! Forget the data of the current q-point (to be called when the
    !! k-points, the wavefunctions or the potential change).
    !
    INTEGER :: i, istat
    !
    IF ( in_session ) CALL dfpt_dev_end( )
    CALL dfpt_dev_grid_reset( )
    q_ready = .FALSE.
    IF ( ALLOCATED(npw_h) ) DEALLOCATE( npw_h, npwq_h, nocc_h, noccq_h, spin_h, wk_h )
    IF ( ALLOCATED(npw_d) ) DEALLOCATE( npw_d, npwq_d, nocc_d, noccq_d, spin_d )
    IF ( ALLOCATED(nlk_d) ) DEALLOCATE( nlk_d, nlq_d, g2_d )
    IF ( ALLOCATED(kqg_d) ) DEALLOCATE( kqg_d )
    CALL release_kept( )
    IF ( ALLOCATED(psik_d) ) DEALLOCATE( psik_d )
    IF ( ALLOCATED(w_d) ) DEALLOCATE( w_d )
    IF ( ALLOCATED(evq_d) ) DEALLOCATE( evq_d )
    IF ( ALLOCATED(psir_d) ) DEALLOCATE( psir_d )
    IF ( ALLOCATED(becp1_d) ) DEALLOCATE( becp1_d )
    IF ( ALLOCATED(et_d) ) DEALLOCATE( et_d, eprec_d, wwg_d, wg1_d )
    IF ( ALLOCATED(etq_d) ) DEALLOCATE( etq_d )
    use_z = .FALSE. ; zoff = 0 ; nz = 0
    IF ( ALLOCATED(deeq_d) ) DEALLOCATE( deeq_d, qq_d, kbna_d, kbof_d, kbnh_d )
    IF ( ALLOCATED(v_d) ) DEALLOCATE( v_d )
    IF ( ALLOCATED(wgt_d) ) DEALLOCATE( wgt_d )
    IF ( ALLOCATED(psig_d) ) DEALLOCATE( psig_d, psic_d )
    IF ( ALLOCATED(psi2_d) ) DEALLOCATE( psi2_d )
    IF ( ALLOCATED(ws_d) ) DEALLOCATE( ws_d )
    IF ( ALLOCATED(evqs_d) ) DEALLOCATE( evqs_d )
    IF ( ALLOCATED(psigs_d) ) DEALLOCATE( psigs_d, psics_d )
    DO i = 1, nplan
       IF ( plan(i) /= 0 ) istat = cufftDestroy( plan(i) )
       IF ( plans(i) /= 0 ) istat = cufftDestroy( plans(i) )
    ENDDO
    plans = 0
    use_sp = .FALSE.
    use_real = .FALSE.
    use_lg = .FALSE.
    use_lgr = .FALSE.
    IF ( ALLOCATED(lgnr_d) ) DEALLOCATE( lgnr_d, lgpr_d, lgvr_d, lger_d )
    IF ( ALLOCATED(hdr_d) ) DEALLOCATE( hdr_d, sdr_d )
    IF ( ALLOCATED(rnh_h) ) DEALLOCATE( rnh_h, rns_h, rhl_h, rhm_h, rg2_h )
    IF ( ALLOCATED(lgn_d) ) DEALLOCATE( lgn_d, lgi_d, lgv_d, lge_d )
    IF ( ALLOCATED(hdg_d) ) DEALLOCATE( hdg_d, sdg_d )
    IF ( ALLOCATED(clus_h) ) DEALLOCATE( clus_h )
    IF ( ALLOCATED(erot_h) ) DEALLOCATE( erot_h )
    IF ( ALLOCATED(nh_d) ) DEALLOCATE( nh_d, nself_d, hl_d, hm_d, nlh_d, nlm_d, g2h_d )
    IF ( ALLOCATED(wr_d) ) DEALLOCATE( wr_d )
    IF ( ALLOCATED(ur_d) ) DEALLOCATE( ur_d )
    plan = 0 ; plan_nb = 0 ; nplan = 0
    have_psir = .FALSE.
    !
  END SUBROUTINE dfpt_dev_invalidate
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
  SUBROUTINE init_blas( )
    !-----------------------------------------------------------------------
    INTEGER :: istat
    istat = cublasCreate( hblas )
    IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cublasCreate failed', ABS(istat) )
    blas_ready = .TRUE.
  END SUBROUTINE init_blas
  !
  !-----------------------------------------------------------------------
  SUBROUTINE apply_qq( b_d, qb_d, m )
    !-----------------------------------------------------------------------
    !! qb(ikb,i) = sum_jh qq(ih,jh,na) b(ofs+jh,i)
    INTEGER, INTENT(IN) :: m
    COMPLEX(DP), DEVICE, INTENT(IN) :: b_d(MAX(nkbl,1),m)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: qb_d(MAX(nkbl,1),m)
    INTEGER :: i, ikb, jh, na, ih, iof
    COMPLEX(DP) :: s
    !
    !$cuf kernel do(2)
    DO i = 1, m
       DO ikb = 1, nkbl
          na = kbna_d(ikb) ; iof = kbof_d(ikb) ; ih = ikb - iof
          s = (0.0_DP, 0.0_DP)
          DO jh = 1, kbnh_d(ikb)
             s = s + qq_d(ih,jh,na) * b_d(iof+jh,i)
          ENDDO
          qb_d(ikb,i) = s
       ENDDO
    ENDDO
    !
  END SUBROUTINE apply_qq
  !
  !-----------------------------------------------------------------------
  SUBROUTINE set_eprec( )
    !-----------------------------------------------------------------------
    INTEGER :: ik, ibnd, ig, nb, nwl
    REAL(DP) :: s
    !
    nb = nbo ; nwl = nkbl
    IF ( lus ) THEN
       !$acc parallel loop gang collapse(2) private(s)
       DO ik = 1, nks
          DO ibnd = 1, nb
             s = 0.0_DP
             !$acc loop vector reduction(+:s)
             DO ig = 1, npwxl
                s = s + g2_d(ig,ik) * DBLE( CONJG(evq_d(ig,ibnd,ik)) * evq_d(ig,ibnd,ik) )
             ENDDO
             eprec_d(ibnd,ik) = 1.35_DP * s
          ENDDO
       ENDDO
    ELSE
       !$acc parallel loop gang collapse(2) private(s)
       DO ik = 1, nks
          DO ibnd = 1, nb
             s = 0.0_DP
             !$acc loop vector reduction(+:s)
             DO ig = 1, npwxl
                s = s + g2_d(ig,ik) * DBLE( CONJG(w_d(ig,nwl+ibnd,ik)) * w_d(ig,nwl+ibnd,ik) )
             ENDDO
             eprec_d(ibnd,ik) = 1.35_DP * s
          ENDDO
       ENDDO
    ENDIF
    !
  END SUBROUTINE set_eprec
  !
  !-----------------------------------------------------------------------
  SUBROUTINE make_plans( )
    !-----------------------------------------------------------------------
    INTEGER :: nb, istat, fdim(3)
    !
    fdim = (/ nr3, nr2, nr1 /)
    nplan = 0
    nb = nfb
    DO
       nplan = nplan + 1
       plan_nb(nplan) = nb
       istat = cufftPlanMany( plan(nplan), 3, fdim, fdim, 1, nnr, fdim, 1, nnr, CUFFT_Z2Z, nb )
       IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cufftPlanMany failed', ABS(istat) )
       IF ( nb == 1 .OR. nplan == nplan_max ) EXIT
       nb = MAX( 1, nb / 4 )
    ENDDO
    IF ( plan_nb(nplan) /= 1 ) CALL errore( 'dfpt_dev', 'FFT plan table too small', 1 )
    !
  END SUBROUTINE make_plans
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION pick_plan( nb )
    !-----------------------------------------------------------------------
    !! the smallest plan that holds nb transforms
    INTEGER, INTENT(IN) :: nb
    INTEGER :: i
    pick_plan = 1
    DO i = 2, nplan
       IF ( plan_nb(i) >= nb ) pick_plan = i
    ENDDO
  END FUNCTION pick_plan
  !
  !-----------------------------------------------------------------------
  SUBROUTINE fill_psir( )
    !-----------------------------------------------------------------------
    !! psir_d(:,ibnd,ik) = psi_{k,ibnd}(r) for the occupied bands
    INTEGER :: ik, ib0, nb, ip, j, ig, ir, istat, nbl, ikl
    !
    nbl = nbo
    DO ik = 1, nks
       ikl = ik
       DO ib0 = 0, nocc_h(ik)-1, nfb
          nb = MIN( nfb, nocc_h(ik)-ib0 )
          ip = pick_plan( nb )
          !$cuf kernel do(2)
          DO j = 1, nb
             DO ig = 1, npwxl
                IF ( ig <= npw_d(ikl) ) psig_d(nlk_d(ig,ikl) + (j-1)*nnr) = psik_d(ig,ib0+j,ikl)
             ENDDO
          ENDDO
          istat = cufftExecZ2Z( plan(ip), psig_d, psic_d, CUFFT_INVERSE )
          !$cuf kernel do(2)
          DO j = 1, nb
             DO ig = 1, npwxl
                IF ( ig <= npw_d(ikl) ) psig_d(nlk_d(ig,ikl) + (j-1)*nnr) = (0.0_DP, 0.0_DP)
             ENDDO
          ENDDO
          !$cuf kernel do(2)
          DO j = 1, nb
             DO ir = 1, nnr
                psir_d(ir,ib0+j,ikl) = psic_d(ir + (j-1)*nnr)
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    !
  END SUBROUTINE fill_psir
  !
  !-----------------------------------------------------------------------
  SUBROUTINE alloc_session( npert, with_dvb, ok )
    !-----------------------------------------------------------------------
    !! Work arrays for npert perturbations per k-point and the per-column
    !! data (eigenvalue, preconditioner energy, list of live columns).
    !
    USE io_global,   ONLY : stdout
    !
    INTEGER, INTENT(IN) :: npert
    LOGICAL, INTENT(IN) :: with_dvb
    LOGICAL, INTENT(OUT) :: ok
    !
    INTEGER :: ik, ipert, ibnd, c0
    REAL(DP) :: need
    REAL(DP), ALLOCATABLE :: rvec(:), rvec2(:)
    REAL(DP), ALLOCATABLE :: et_h(:,:), ep_h(:,:)
    !
    ok = .FALSE.
    npe = npert
    ncolk = npe * nbo
    ncol = ncolk * nks
    !
    need = 16.0_DP * REAL(npwxl,DP) * REAL(ncol,DP) * 5.0_DP + 32.0_DP * REAL(nw,DP) * ncol + &
           32.0_DP * REAL(nnr,DP) * nspl * npe
    IF ( use_sp ) need = need + 8.0_DP * REAL(npwxl,DP) * REAL(ncol,DP) * 3.5_DP
    IF ( need > 0.8_DP * free_bytes( ) ) THEN
       WRITE( stdout, '(5x,"DFPT device engine: not enough device memory for ",i4, &
            &" perturbations (",f8.1," GB), stock solver is used")' ) npe, need / 1.0E9_DP
       RETURN
    ENDIF
    !
    IF ( with_dvb ) ALLOCATE( dvb_d(npwxl,ncol) )
    IF ( use_sp ) THEN
       ALLOCATE( gs_d(npwxl,ncol), hs_d(npwxl,ncol), ts_d(npwxl,ncol), cws_d(nw,ncol), cw2s_d(nw,ncol) )
       IF ( .NOT. batched_gemm ) ALLOCATE( hcs_d(npwxl,ncolk), tcs_d(npwxl,ncolk) )
       hs_d = (0.0_SP, 0.0_SP)
    ENDIF
    real_on = .FALSE.
    IF ( use_lg ) ALLOCATE( gl_d(nlg,ncol), cl_d(nlg,ncol) )
    ALLOCATE( dpsi_d(npwxl,ncol), g_d(npwxl,ncol), h_d(npwxl,ncol), t_d(npwxl,ncol) )
    ALLOCATE( cw_d(nw,ncol), cw2_d(nw,ncol) )
    ALLOCATE( e_d(ncol), ep_d(ncol), rho_d(ncol), rhoold_d(ncol), a_d(ncol), c_d(ncol), rhoc_d(ncol) )
    ALLOCATE( act_d(ncol), live_d(ncol), live_h(ncol), act_h(ncol) )
    ALLOCATE( dv_d(nnr,nspl,npe), drho_d(nnr,nspl,npe) )
    IF ( lus ) ALLOCATE( int3_d(nhml,nhml,natl,nspl,npe) )
    !
    ! ... per-column eigenvalue and preconditioner energy; list of live columns
    !
    ALLOCATE( rvec(ncol), rvec2(ncol), et_h(nbo,nks), ep_h(nbo,nks) )
    et_h = et_d ; ep_h = eprec_d
    nlive = 0
    DO ik = 1, nks
       DO ipert = 1, npe
          c0 = ( (ik-1)*npe + (ipert-1) ) * nbo
          DO ibnd = 1, nbo
             rvec(c0+ibnd) = et_h(ibnd,ik)
             rvec2(c0+ibnd) = ep_h(ibnd,ik)
             IF ( ibnd <= nocc_h(ik) ) THEN
                nlive = nlive + 1
                live_h(nlive) = c0 + ibnd
             ELSE
                rvec2(c0+ibnd) = 1.0_DP
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    e_d = rvec ; ep_d = rvec2
    live_d = live_h
    !
    ! ... real mode: work arrays, eigenvalues of the rotated columns
    !
    IF ( use_real ) THEN
       need = 4.0_DP * REAL(2*nhx,DP) * ( 4.0_DP * 2.0_DP * ncol + 2.0_DP * 2.0_DP * ncolk ) + &
              8.0_DP * REAL(nw,DP) * 2.0_DP * ncolk
       IF ( need < 0.5_DP * free_bytes( ) ) THEN
          ALLOCATE( gr_d(2*nhx,2*ncol), hr_d(2*nhx,2*ncol), tr_d(2*nhx,2*ncol), dr_d(2*nhx,2*ncol) )
          ALLOCATE( hcr_d(2*nhx,2*ncolk), tcr_d(2*nhx,2*ncolk), cwr_d(nw,2*ncolk), cw2r_d(nw,2*ncolk) )
          ALLOCATE( rhor_d(2*ncol), rhooldr_d(2*ncol), ar_d(2*ncol), cr_d(2*ncol), rhocr_d(2*ncol) )
          ALLOCATE( actr_d(2*ncol), pa_d(2,2*ncol), actr_h(2*ncol), pa_h(2,2*ncol), er_d(ncol) )
          gs_d = (0.0_SP, 0.0_SP) ; ts_d = (0.0_SP, 0.0_SP)
          hr_d = 0.0_SP
          IF ( use_lgr ) ALLOCATE( glr_d(nlb,2*ncolk), clr_d(nlb,2*ncolk) )
          DO ik = 1, nks
             DO ipert = 1, npe
                c0 = ( (ik-1)*npe + (ipert-1) ) * nbo
                DO ibnd = 1, nbo
                   IF ( ibnd <= nocc_h(ik) ) rvec(c0+ibnd) = erot_h(ibnd,ik)
                ENDDO
             ENDDO
          ENDDO
          er_d = rvec
          real_on = .TRUE.
       ENDIF
    ENDIF
    DEALLOCATE( rvec, rvec2, et_h, ep_h )
    !
    ok = .TRUE.
    !
  END SUBROUTINE alloc_session
  !
  !-----------------------------------------------------------------------
  SUBROUTINE release_kept( )
    !-----------------------------------------------------------------------
    IF ( .NOT. dpsi_kept ) RETURN
    dpsi_kept = .FALSE.
    IF ( in_session ) RETURN
    IF ( ALLOCATED(dpsi_d) ) DEALLOCATE( dpsi_d, g_d )
    IF ( ALLOCATED(cw_d) ) DEALLOCATE( cw_d, cw2_d )
  END SUBROUTINE release_kept
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dfpt_dev_keep_dpsi( )
    !-----------------------------------------------------------------------
    !! The caller does not need the response wavefunctions of the current
    !! session in the buffer iudwf: at the end of the session they stay on
    !! the device, where dfpt_dev_get_bq serves the dynamical matrix.
    IF ( in_session .AND. .NOT. lin_session ) keep_next = .TRUE.
  END SUBROUTINE dfpt_dev_keep_dpsi
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dfpt_dev_flush( )
    !-----------------------------------------------------------------------
    !! Copy the device \(\Delta\psi\) to the buffer iudwf, where the rest of
    !! the code expects it.
    !
    USE qpoint,      ONLY : nksq
    USE units_lr,    ONLY : iudwf, lrdwf
    USE buffers,     ONLY : save_buffer
    USE eqv,         ONLY : dpsi
    !
    INTEGER :: ik, ipert, c0, nrec
    COMPLEX(DP), ALLOCATABLE :: stage(:,:)
    !
    IF ( .NOT. in_session .OR. .NOT. dpsi_dirty .OR. lin_session ) RETURN
    CALL start_clock( 'dfptd_flush' )
    ALLOCATE( stage(npwxl,nbo) )
    DO ik = 1, nks
       DO ipert = 1, npe
          nrec = (ipert-1)*nksq + ik
          c0 = ( (ik-1)*npe + (ipert-1) ) * nbo
          stage(:,:) = dpsi_d(:,c0+1:c0+nbo)
          dpsi(:,:) = c_zero
          dpsi(1:npwxl,1:nbo) = stage(:,:)
          CALL save_buffer( dpsi, lrdwf, iudwf, nrec )
       ENDDO
    ENDDO
    DEALLOCATE( stage )
    dpsi_dirty = .FALSE.
    CALL stop_clock( 'dfptd_flush' )
    !
  END SUBROUTINE dfpt_dev_flush
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dfpt_dev_end( )
    !-----------------------------------------------------------------------
    IF ( .NOT. in_session ) RETURN
    dpsi_kept = keep_next .AND. dpsi_dirty .AND. .NOT. lin_session .AND. .NOT. keep_veto
    IF ( dpsi_kept ) npe_kept = npe
    keep_next = .FALSE.
    IF ( .NOT. dpsi_kept ) CALL dfpt_dev_flush( )
    in_session = .FALSE.
    lin_session = .FALSE.
    prefilled = .FALSE.
    IF ( ALLOCATED(dvb_d) ) DEALLOCATE( dvb_d )
    IF ( dpsi_kept ) THEN
       ! dpsi, a work array and the projection array stay for dfpt_dev_get_bq
       IF ( ALLOCATED(h_d) ) DEALLOCATE( h_d, t_d )
    ELSE
       IF ( ALLOCATED(dpsi_d) ) DEALLOCATE( dpsi_d, g_d, h_d, t_d )
       IF ( ALLOCATED(cw_d) ) DEALLOCATE( cw_d, cw2_d )
    ENDIF
    IF ( ALLOCATED(gs_d) ) DEALLOCATE( gs_d, hs_d, ts_d, cws_d, cw2s_d )
    IF ( ALLOCATED(hcs_d) ) DEALLOCATE( hcs_d, tcs_d )
    IF ( ALLOCATED(gl_d) ) DEALLOCATE( gl_d, cl_d )
    IF ( ALLOCATED(glr_d) ) DEALLOCATE( glr_d, clr_d )
    IF ( ALLOCATED(gr_d) ) THEN
       DEALLOCATE( gr_d, hr_d, tr_d, dr_d, hcr_d, tcr_d, cwr_d, cw2r_d )
       DEALLOCATE( rhor_d, rhooldr_d, ar_d, cr_d, rhocr_d, actr_d, pa_d, actr_h, pa_h, er_d )
    ENDIF
    real_on = .FALSE.
    IF ( ALLOCATED(e_d) ) DEALLOCATE( e_d, ep_d, rho_d, rhoold_d, a_d, c_d, rhoc_d )
    IF ( ALLOCATED(act_d) ) DEALLOCATE( act_d, live_d, live_h, act_h )
    IF ( ALLOCATED(dv_d) ) DEALLOCATE( dv_d, drho_d )
    IF ( ALLOCATED(int3_d) ) DEALLOCATE( int3_d )
  END SUBROUTINE dfpt_dev_end
  !
  !-----------------------------------------------------------------------
  SUBROUTINE gemm_k( trans, m, n, k, alpha, a_dev, lda, stra, b_dev, ldb, strb, beta, &
                     c_dev, ldc, strc, nbatch )
    !-----------------------------------------------------------------------
    !! nbatch products C_i = alpha op(A_i) B_i + beta C_i of matrices stored
    !! with constant strides.
    !
    INTEGER, INTENT(IN) :: trans, m, n, k, lda, ldb, ldc, nbatch
    INTEGER(8), INTENT(IN) :: stra, strb, strc
    COMPLEX(DP), INTENT(IN) :: alpha, beta
    COMPLEX(DP), DEVICE :: a_dev(*), b_dev(*), c_dev(*)
    INTEGER :: istat, i
    !
    IF ( nbatch <= 0 .OR. m <= 0 .OR. n <= 0 ) RETURN
    IF ( batched_gemm .AND. nbatch > 1 ) THEN
       istat = cublasZgemmStridedBatched( hblas, trans, CUBLAS_OP_N, m, n, k, alpha, a_dev, lda, &
                                          stra, b_dev, ldb, strb, beta, c_dev, ldc, strc, nbatch )
       IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cublasZgemmStridedBatched failed', ABS(istat) )
    ELSE
       DO i = 0, nbatch-1
          istat = cublasZgemm( hblas, trans, CUBLAS_OP_N, m, n, k, alpha, a_dev(1+i*stra), lda, &
                               b_dev(1+i*strb), ldb, beta, c_dev(1+i*strc), ldc )
          IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cublasZgemm failed', ABS(istat) )
       ENDDO
    ENDIF
    !
  END SUBROUTINE gemm_k
  !
  !-----------------------------------------------------------------------
  SUBROUTINE apply_h( nact, src_d, dst_d )
    !-----------------------------------------------------------------------
    !! dst = ( H - e S + alpha_pv P_v ) src for the nact columns listed in
    !! act_d (ch_psi_all). The nonlocal and projector terms are computed for
    !! all columns of the k-points that have an active column.
    !
    INTEGER, INTENT(IN) :: nact
    COMPLEX(DP), DEVICE, INTENT(IN) :: src_d(npwxl,ncol)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: dst_d(npwxl,ncol)
    !
    INTEGER :: j0, nb, ip, j, ig, ir, c, ik, istat, k1, k2, i, nck, nwl, nkl, nbql, iw, cc, &
               na, iof, ih, jh, is
    REAL(DP) :: scale, apv, ee
    COMPLEX(DP) :: s
    !
    CALL start_clock( 'dfptd_hpsi' )
    scale = 1.0_DP / DBLE(nnr)
    nck = ncolk ; nwl = nw ; nkl = nkbl ; nbql = nbq ; apv = alpha_pv_l
    !
    ! ... kinetic energy and local potential
    !
    DO j0 = 0, nact-1, nfb
       nb = MIN( nfb, nact-j0 )
       ip = pick_plan( nb )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, npwxl
             c = act_d(j0+j)
             ik = (c-1)/nck + 1
             IF ( ig <= npwq_d(ik) ) psig_d(nlq_d(ig,ik) + (j-1)*nnr) = src_d(ig,c)
          ENDDO
       ENDDO
       istat = cufftExecZ2Z( plan(ip), psig_d, psic_d, CUFFT_INVERSE )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, npwxl
             c = act_d(j0+j)
             ik = (c-1)/nck + 1
             IF ( ig <= npwq_d(ik) ) psig_d(nlq_d(ig,ik) + (j-1)*nnr) = (0.0_DP, 0.0_DP)
          ENDDO
       ENDDO
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ir = 1, nnr
             c = act_d(j0+j)
             ik = (c-1)/nck + 1
             psic_d(ir + (j-1)*nnr) = psic_d(ir + (j-1)*nnr) * v_d(ir,spin_d(ik))
          ENDDO
       ENDDO
       istat = cufftExecZ2Z( plan(ip), psic_d, psic_d, CUFFT_FORWARD )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, npwxl
             c = act_d(j0+j)
             ik = (c-1)/nck + 1
             IF ( ig <= npwq_d(ik) ) THEN
                dst_d(ig,c) = ( g2_d(ig,ik) - e_d(c) ) * src_d(ig,c) + &
                              psic_d(nlq_d(ig,ik) + (j-1)*nnr) * scale
             ELSE
                dst_d(ig,c) = (0.0_DP, 0.0_DP)
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    ! ... nonlocal potential, e*S and projector on the valence bands, for
    ! ... runs of consecutive k-points with active columns
    !
    i = 1
    DO WHILE ( i <= nact )
       k1 = (act_h(i)-1)/ncolk + 1
       k2 = k1
       DO WHILE ( i <= nact )
          ik = (act_h(i)-1)/ncolk + 1
          IF ( ik > k2+1 ) EXIT
          k2 = ik
          i = i + 1
       ENDDO
       !
       CALL gemm_k( CUBLAS_OP_C, nw, ncolk, npwxl, c_one, w_d(1,1,k1), npwxl, INT(npwxl,8)*nw, &
                    src_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, c_zero, &
                    cw_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, k2-k1+1 )
       !$cuf kernel do(2)
       DO c = (k1-1)*nck+1, k2*nck
          DO iw = 1, nwl
             ik = (c-1)/nck + 1
             IF ( iw <= nkl ) THEN
                na = kbna_d(iw) ; iof = kbof_d(iw) ; ih = iw - iof
                is = spin_d(ik) ; ee = e_d(c)
                s = (0.0_DP, 0.0_DP)
                DO jh = 1, kbnh_d(iw)
                   s = s + ( deeq_d(ih,jh,na,is) - ee * qq_d(ih,jh,na) ) * cw_d(iof+jh,c)
                ENDDO
                cw2_d(iw,c) = s
             ELSEIF ( iw - nkl <= noccq_d(ik) ) THEN
                cw2_d(iw,c) = apv * cw_d(iw,c)
             ELSE
                cw2_d(iw,c) = (0.0_DP, 0.0_DP)
             ENDIF
          ENDDO
       ENDDO
       CALL gemm_k( CUBLAS_OP_N, npwxl, ncolk, nw, c_one, w_d(1,1,k1), npwxl, INT(npwxl,8)*nw, &
                    cw2_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, c_one, &
                    dst_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, k2-k1+1 )
    ENDDO
    CALL stop_clock( 'dfptd_hpsi' )
    !
  END SUBROUTINE apply_h
  !
  !-----------------------------------------------------------------------
  SUBROUTINE build_rhs( first_iter )
    !-----------------------------------------------------------------------
    !! g_d = -P_c^+ ( dV_bare + dV_scf ) psi   (apply_dpot_bands, adddvscf,
    !! orthogonalize)
    !
    LOGICAL, INTENT(IN) :: first_iter
    !
    INTEGER :: j0, nb, ip, j, ig, ir, c, ik, ibnd, ipert, istat, nck, nbl, npl, nkl, nwl, nbql, &
               iw, na, iof, ih, jh, is, n1, n2
    REAL(DP) :: scale
    COMPLEX(DP) :: s
    !
    CALL start_clock( 'dfptd_rhs' )
    scale = 1.0_DP / DBLE(nnr)
    nck = ncolk ; nbl = nbo ; npl = npe ; nkl = nkbl ; nwl = nw ; nbql = nbq
    n1 = npwxl ; n2 = ncol
    !
    !$cuf kernel do(2)
    DO c = 1, n2
       DO ig = 1, n1
          g_d(ig,c) = dvb_d(ig,c)
       ENDDO
    ENDDO
    !
    IF ( .NOT. first_iter ) THEN
       !
       ! ... dV_scf(r) psi_k(r), back to G-space on the k+q sphere
       !
       DO j0 = 0, nlive-1, nfb
          nb = MIN( nfb, nlive-j0 )
          ip = pick_plan( nb )
          IF ( have_psir ) THEN
             !$cuf kernel do(2)
             DO j = 1, nb
                DO ir = 1, nnr
                   c = live_d(j0+j)
                   ik = (c-1)/nck + 1
                   ibnd = MOD(c-1,nbl) + 1
                   ipert = MOD((c-1)/nbl,npl) + 1
                   psic_d(ir + (j-1)*nnr) = psir_d(ir,ibnd,ik) * dv_d(ir,spin_d(ik),ipert)
                ENDDO
             ENDDO
          ELSE
             CALL psi_to_r( j0, nb, ip )
             !$cuf kernel do(2)
             DO j = 1, nb
                DO ir = 1, nnr
                   c = live_d(j0+j)
                   ik = (c-1)/nck + 1
                   ipert = MOD((c-1)/nbl,npl) + 1
                   psic_d(ir + (j-1)*nnr) = psi2_d(ir + (j-1)*nnr) * dv_d(ir,spin_d(ik),ipert)
                ENDDO
             ENDDO
          ENDIF
          istat = cufftExecZ2Z( plan(ip), psic_d, psic_d, CUFFT_FORWARD )
          !$cuf kernel do(2)
          DO j = 1, nb
             DO ig = 1, npwxl
                c = live_d(j0+j)
                ik = (c-1)/nck + 1
                IF ( ig <= npwq_d(ik) ) g_d(ig,c) = g_d(ig,c) + psic_d(nlq_d(ig,ik) + (j-1)*nnr) * scale
             ENDDO
          ENDDO
       ENDDO
       !
       ! ... ultrasoft term: beta_{k+q} int3 <beta_k|psi_k>
       !
       IF ( lus ) THEN
          !$cuf kernel do(2)
          DO c = 1, n2
             DO iw = 1, nkl
                ik = (c-1)/nck + 1
                ibnd = MOD(c-1,nbl) + 1
                ipert = MOD((c-1)/nbl,npl) + 1
                na = kbna_d(iw) ; iof = kbof_d(iw) ; ih = iw - iof
                is = spin_d(ik)
                s = (0.0_DP, 0.0_DP)
                DO jh = 1, kbnh_d(iw)
                   s = s + int3_d(ih,jh,na,is,ipert) * becp1_d(iof+jh,ibnd,ik)
                ENDDO
                cw2_d(iw,c) = s
             ENDDO
          ENDDO
          CALL gemm_k( CUBLAS_OP_N, npwxl, ncolk, nkbl, c_one, w_d, npwxl, INT(npwxl,8)*nw, &
                       cw2_d, nw, INT(nw,8)*ncolk, c_one, g_d, npwxl, INT(npwxl,8)*ncolk, nks )
       ENDIF
       !
    ENDIF
    !
    ! ... -P_c^+ : g = S psi_{k+q} ( wwg * <psi_{k+q}|g> ) - wg1 * g
    !
    IF ( lus ) THEN
       CALL gemm_k( CUBLAS_OP_C, nbq, ncolk, npwxl, c_one, evq_d, npwxl, INT(npwxl,8)*nbq, &
                    g_d, npwxl, INT(npwxl,8)*ncolk, c_zero, cw_d, nw, INT(nw,8)*ncolk, nks )
    ELSE
       CALL gemm_k( CUBLAS_OP_C, nbq, ncolk, npwxl, c_one, w_d(1,nkbl+1,1), npwxl, INT(npwxl,8)*nw, &
                    g_d, npwxl, INT(npwxl,8)*ncolk, c_zero, cw_d, nw, INT(nw,8)*ncolk, nks )
    ENDIF
    !$cuf kernel do(2)
    DO c = 1, n2
       DO iw = 1, nbql
          ik = (c-1)/nck + 1
          ibnd = MOD(c-1,nbl) + 1
          cw_d(iw,c) = cw_d(iw,c) * wwg_d(iw,ibnd,ik)
       ENDDO
    ENDDO
    IF ( metal ) THEN
       !$cuf kernel do(2)
       DO c = 1, n2
          DO ig = 1, n1
             ik = (c-1)/nck + 1
             ibnd = MOD(c-1,nbl) + 1
             g_d(ig,c) = g_d(ig,c) * wg1_d(ibnd,ik)
          ENDDO
       ENDDO
    ENDIF
    CALL gemm_k( CUBLAS_OP_N, npwxl, ncolk, nbq, c_one, w_d(1,nkbl+1,1), npwxl, INT(npwxl,8)*nw, &
                 cw_d, nw, INT(nw,8)*ncolk, c_mone, g_d, npwxl, INT(npwxl,8)*ncolk, nks )
    CALL stop_clock( 'dfptd_rhs' )
    !
  END SUBROUTINE build_rhs
  !
  !-----------------------------------------------------------------------
  SUBROUTINE psi_to_r( j0, nb, ip )
    !-----------------------------------------------------------------------
    !! psi2_d(:,j) = psi_k(r) of the band of live column j0+j (no cache)
    INTEGER, INTENT(IN) :: j0, nb, ip
    INTEGER :: j, ig, c, ik, ibnd, istat, nck, nbl
    !
    nck = ncolk ; nbl = nbo
    !$cuf kernel do(2)
    DO j = 1, nb
       DO ig = 1, npwxl
          c = live_d(j0+j)
          ik = (c-1)/nck + 1
          ibnd = MOD(c-1,nbl) + 1
          IF ( ig <= npw_d(ik) ) psig_d(nlk_d(ig,ik) + (j-1)*nnr) = psik_d(ig,ibnd,ik)
       ENDDO
    ENDDO
    istat = cufftExecZ2Z( plan(ip), psig_d, psi2_d, CUFFT_INVERSE )
    !$cuf kernel do(2)
    DO j = 1, nb
       DO ig = 1, npwxl
          c = live_d(j0+j)
          ik = (c-1)/nck + 1
          IF ( ig <= npw_d(ik) ) psig_d(nlk_d(ig,ik) + (j-1)*nnr) = (0.0_DP, 0.0_DP)
       ENDDO
    ENDDO
    !
  END SUBROUTINE psi_to_r
  !
  !-----------------------------------------------------------------------
  SUBROUTINE add_drho( )
    !-----------------------------------------------------------------------
    !! drho_d(r,spin,ipert) = sum_kv 2 wk/omega conj(psi_kv(r)) dpsi_kv(r)
    !! (incdrhoscf)
    !
    INTEGER :: j0, nb, ip, j, ig, ir, c, ik, ibnd, ipert, istat, nck, nbl, npl, is, n1, n2, n3
    !
    CALL start_clock( 'dfptd_drho' )
    nck = ncolk ; nbl = nbo ; npl = npe
    n1 = nnr ; n2 = nspl ; n3 = npe
    !$cuf kernel do(3)
    DO ipert = 1, n3
       DO is = 1, n2
          DO ir = 1, n1
             drho_d(ir,is,ipert) = (0.0_DP, 0.0_DP)
          ENDDO
       ENDDO
    ENDDO
    !
    DO j0 = 0, nlive-1, nfb
       nb = MIN( nfb, nlive-j0 )
       ip = pick_plan( nb )
       IF ( .NOT. have_psir ) CALL psi_to_r( j0, nb, ip )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, npwxl
             c = live_d(j0+j)
             ik = (c-1)/nck + 1
             IF ( ig <= npwq_d(ik) ) psig_d(nlq_d(ig,ik) + (j-1)*nnr) = dpsi_d(ig,c)
          ENDDO
       ENDDO
       istat = cufftExecZ2Z( plan(ip), psig_d, psic_d, CUFFT_INVERSE )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, npwxl
             c = live_d(j0+j)
             ik = (c-1)/nck + 1
             IF ( ig <= npwq_d(ik) ) psig_d(nlq_d(ig,ik) + (j-1)*nnr) = (0.0_DP, 0.0_DP)
          ENDDO
       ENDDO
       IF ( have_psir ) THEN
          !$cuf kernel do(1)
          DO ir = 1, nnr
             DO j = 1, nb
                c = live_d(j0+j)
                ik = (c-1)/nck + 1
                ibnd = MOD(c-1,nbl) + 1
                ipert = MOD((c-1)/nbl,npl) + 1
                drho_d(ir,spin_d(ik),ipert) = drho_d(ir,spin_d(ik),ipert) + wgt_d(ik) * &
                     CONJG( psir_d(ir,ibnd,ik) ) * psic_d(ir + (j-1)*nnr)
             ENDDO
          ENDDO
       ELSE
          !$cuf kernel do(1)
          DO ir = 1, nnr
             DO j = 1, nb
                c = live_d(j0+j)
                ik = (c-1)/nck + 1
                ipert = MOD((c-1)/nbl,npl) + 1
                drho_d(ir,spin_d(ik),ipert) = drho_d(ir,spin_d(ik),ipert) + wgt_d(ik) * &
                     CONJG( psi2_d(ir + (j-1)*nnr) ) * psic_d(ir + (j-1)*nnr)
             ENDDO
          ENDDO
       ENDIF
    ENDDO
    CALL stop_clock( 'dfptd_drho' )
    !
  END SUBROUTINE add_drho
  !
  !-----------------------------------------------------------------------
  SUBROUTINE add_dbecsum( dbecsum )
    !-----------------------------------------------------------------------
    !! dbecsum += wk conj(<beta_k|psi_k>) <beta_{k+q}|dpsi_k>  (addusdbec)
    !
    USE ions_base,   ONLY : nat, ityp, ntyp => nsp
    USE uspp,        ONLY : ijtoh, ofsbeta
    USE uspp_param,  ONLY : upf, nh, nhm
    USE noncollin_module, ONLY : nspin_mag
    USE lrus,        ONLY : becp1
    !
    COMPLEX(DP), INTENT(INOUT) :: dbecsum(nhm*(nhm+1)/2, nat, nspin_mag, npe)
    !
    COMPLEX(DP), ALLOCATABLE :: dbq(:,:)
    INTEGER :: ik, ipert, ibnd, c, na, nt, ih, jh, ikb, jkb, ijh, is
    REAL(DP) :: wgt
    !
    CALL start_clock( 'dfptd_dbec' )
    CALL gemm_k( CUBLAS_OP_C, nkbl, ncolk, npwxl, c_one, w_d, npwxl, INT(npwxl,8)*nw, &
                 dpsi_d, npwxl, INT(npwxl,8)*ncolk, c_zero, cw_d, nw, INT(nw,8)*ncolk, nks )
    ALLOCATE( dbq(nkbl,ncol) )
    dbq(:,:) = cw_d(1:nkbl,1:ncol)
    !
    DO ik = 1, nks
       wgt = wk_h(ik)
       is = spin_h(ik)
       DO ipert = 1, npe
          DO ibnd = 1, nocc_h(ik)
             c = ( (ik-1)*npe + (ipert-1) ) * nbo + ibnd
             DO na = 1, nat
                nt = ityp(na)
                IF ( .NOT. upf(nt)%tvanp ) CYCLE
                DO ih = 1, nh(nt)
                   ikb = ofsbeta(na) + ih
                   ijh = ijtoh(ih,ih,nt)
                   dbecsum(ijh,na,is,ipert) = dbecsum(ijh,na,is,ipert) + &
                        wgt * CONJG( becp1(ik)%k(ikb,ibnd) ) * dbq(ikb,c)
                   DO jh = ih+1, nh(nt)
                      ijh = ijtoh(ih,jh,nt)
                      jkb = ofsbeta(na) + jh
                      dbecsum(ijh,na,is,ipert) = dbecsum(ijh,na,is,ipert) + wgt * &
                           ( CONJG( becp1(ik)%k(ikb,ibnd) ) * dbq(jkb,c) + &
                             CONJG( becp1(ik)%k(jkb,ibnd) ) * dbq(ikb,c) )
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    DEALLOCATE( dbq )
    CALL stop_clock( 'dfptd_dbec' )
    !
  END SUBROUTINE add_dbecsum
  !
  !-----------------------------------------------------------------------
  SUBROUTINE setup_sp( )
    !-----------------------------------------------------------------------
    !! single-precision copies of the projectors and FFT plans
    INTEGER :: ik, iw, ig, nb, i, istat, fdim(3), n1, n2, n3, nb2
    !
    n1 = npwxl ; n2 = nw ; n3 = nks ; nb2 = nbq
    ALLOCATE( ws_d(npwxl,nw,nks) )
    !$cuf kernel do(3)
    DO ik = 1, n3
       DO iw = 1, n2
          DO ig = 1, n1
             ws_d(ig,iw,ik) = CMPLX( w_d(ig,iw,ik), KIND=SP )
          ENDDO
       ENDDO
    ENDDO
    IF ( lus .AND. use_z ) THEN
       ALLOCATE( evqs_d(npwxl,nbq,nks) )
       !$cuf kernel do(3)
       DO ik = 1, n3
          DO iw = 1, nb2
             DO ig = 1, n1
                evqs_d(ig,iw,ik) = CMPLX( evq_d(ig,iw,ik), KIND=SP )
             ENDDO
          ENDDO
       ENDDO
    ENDIF
    ALLOCATE( psigs_d(nnr*nfb), psics_d(nnr*nfb) )
    psigs_d = (0.0_SP, 0.0_SP)
    fdim = (/ nr3, nr2, nr1 /)
    DO i = 1, nplan
       istat = cufftPlanMany( plans(i), 3, fdim, fdim, 1, nnr, fdim, 1, nnr, CUFFT_C2C, plan_nb(i) )
       IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cufftPlanMany (C2C) failed', ABS(istat) )
    ENDDO
    !
  END SUBROUTINE setup_sp
  !
  !-----------------------------------------------------------------------
  SUBROUTINE gemm_ks( trans, m, n, k, alpha, a_dev, lda, stra, b_dev, ldb, strb, beta, &
                      c_dev, ldc, strc, nbatch )
    !-----------------------------------------------------------------------
    !! single-precision twin of gemm_k
    INTEGER, INTENT(IN) :: trans, m, n, k, lda, ldb, ldc, nbatch
    INTEGER(8), INTENT(IN) :: stra, strb, strc
    COMPLEX(SP), INTENT(IN) :: alpha, beta
    COMPLEX(SP), DEVICE :: a_dev(*), b_dev(*), c_dev(*)
    INTEGER :: istat, i
    !
    IF ( nbatch <= 0 .OR. m <= 0 .OR. n <= 0 ) RETURN
    IF ( batched_gemm .AND. nbatch > 1 ) THEN
       istat = cublasCgemmStridedBatched( hblas, trans, CUBLAS_OP_N, m, n, k, alpha, a_dev, lda, &
                                          stra, b_dev, ldb, strb, beta, c_dev, ldc, strc, nbatch )
       IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cublasCgemmStridedBatched failed', ABS(istat) )
    ELSE
       DO i = 0, nbatch-1
          istat = cublasCgemm( hblas, trans, CUBLAS_OP_N, m, n, k, alpha, a_dev(1+i*stra), lda, &
                               b_dev(1+i*strb), ldb, beta, c_dev(1+i*strc), ldc )
          IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cublasCgemm failed', ABS(istat) )
       ENDDO
    ENDIF
    !
  END SUBROUTINE gemm_ks
  !
  !-----------------------------------------------------------------------
  SUBROUTINE apply_h_sp( nact )
    !-----------------------------------------------------------------------
    !! ts_d = ( H - e S + alpha_pv P_v ) hs_d in single precision for the
    !! nact columns listed in act_d (see apply_h). With few k-points the
    !! nonlocal and projector terms are computed for the active columns
    !! only.
    !
    INTEGER, INTENT(IN) :: nact
    !
    INTEGER :: j0, nb, ip, j, ig, ir, c, ik, istat, k1, k2, i, nck, nwl, nkl, iw, &
               na, iof, ih, jh, is, j1, j2, nak, jj, n1
    REAL(SP) :: scale
    REAL(DP) :: apv, ee
    COMPLEX(DP) :: s
    COMPLEX(SP), PARAMETER :: s_one = (1.0_SP, 0.0_SP), s_zero = (0.0_SP, 0.0_SP)
    !
    CALL start_clock( 'dfptd_hpsi' )
    scale = 1.0_SP / REAL(nnr,SP)
    nck = ncolk ; nwl = nw ; nkl = nkbl ; apv = alpha_pv_l ; n1 = npwxl
    !
    DO j0 = 0, nact-1, nfb
       nb = MIN( nfb, nact-j0 )
       ip = pick_plan( nb )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, npwxl
             c = act_d(j0+j)
             ik = (c-1)/nck + 1
             IF ( ig <= npwq_d(ik) ) psigs_d(nlq_d(ig,ik) + (j-1)*nnr) = hs_d(ig,c)
          ENDDO
       ENDDO
       istat = cufftExecC2C( plans(ip), psigs_d, psics_d, CUFFT_INVERSE )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, npwxl
             c = act_d(j0+j)
             ik = (c-1)/nck + 1
             IF ( ig <= npwq_d(ik) ) psigs_d(nlq_d(ig,ik) + (j-1)*nnr) = (0.0_SP, 0.0_SP)
          ENDDO
       ENDDO
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ir = 1, nnr
             c = act_d(j0+j)
             ik = (c-1)/nck + 1
             psics_d(ir + (j-1)*nnr) = psics_d(ir + (j-1)*nnr) * REAL( v_d(ir,spin_d(ik)), SP )
          ENDDO
       ENDDO
       istat = cufftExecC2C( plans(ip), psics_d, psics_d, CUFFT_FORWARD )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ig = 1, npwxl
             c = act_d(j0+j)
             ik = (c-1)/nck + 1
             IF ( ig <= npwq_d(ik) ) THEN
                ts_d(ig,c) = REAL( g2_d(ig,ik) - e_d(c), SP ) * hs_d(ig,c) + &
                             psics_d(nlq_d(ig,ik) + (j-1)*nnr) * scale
             ELSE
                ts_d(ig,c) = (0.0_SP, 0.0_SP)
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    IF ( batched_gemm ) THEN
       !
       ! ... many k-points: all columns of runs of consecutive active k-points
       !
       i = 1
       DO WHILE ( i <= nact )
          k1 = (act_h(i)-1)/ncolk + 1
          k2 = k1
          DO WHILE ( i <= nact )
             ik = (act_h(i)-1)/ncolk + 1
             IF ( ik > k2+1 ) EXIT
             k2 = ik
             i = i + 1
          ENDDO
          CALL gemm_ks( CUBLAS_OP_C, nw, ncolk, npwxl, s_one, ws_d(1,1,k1), npwxl, INT(npwxl,8)*nw, &
                        hs_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, s_zero, &
                        cws_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, k2-k1+1 )
          !$cuf kernel do(2)
          DO c = (k1-1)*nck+1, k2*nck
             DO iw = 1, nwl
                ik = (c-1)/nck + 1
                IF ( iw <= nkl ) THEN
                   na = kbna_d(iw) ; iof = kbof_d(iw) ; ih = iw - iof
                   is = spin_d(ik) ; ee = e_d(c)
                   s = (0.0_DP, 0.0_DP)
                   DO jh = 1, kbnh_d(iw)
                      s = s + ( deeq_d(ih,jh,na,is) - ee * qq_d(ih,jh,na) ) * cws_d(iof+jh,c)
                   ENDDO
                   cw2s_d(iw,c) = CMPLX( s, KIND=SP )
                ELSEIF ( iw - nkl <= noccq_d(ik) ) THEN
                   cw2s_d(iw,c) = REAL( apv, SP ) * cws_d(iw,c)
                ELSE
                   cw2s_d(iw,c) = (0.0_SP, 0.0_SP)
                ENDIF
             ENDDO
          ENDDO
          CALL gemm_ks( CUBLAS_OP_N, npwxl, ncolk, nw, s_one, ws_d(1,1,k1), npwxl, INT(npwxl,8)*nw, &
                        cw2s_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, s_one, &
                        ts_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, k2-k1+1 )
       ENDDO
       !
    ELSE
       !
       ! ... few k-points: the active columns of each k-point, gathered
       !
       j1 = 1
       DO WHILE ( j1 <= nact )
          ik = (act_h(j1)-1)/ncolk + 1
          j2 = j1
          DO WHILE ( j2 < nact )
             IF ( (act_h(j2+1)-1)/ncolk + 1 /= ik ) EXIT
             j2 = j2 + 1
          ENDDO
          nak = j2 - j1 + 1
          !$cuf kernel do(2)
          DO jj = 1, nak
             DO ig = 1, n1
                hcs_d(ig,jj) = hs_d(ig,act_d(j1-1+jj))
             ENDDO
          ENDDO
          istat = cublasCgemm( hblas, CUBLAS_OP_C, CUBLAS_OP_N, nw, nak, npwxl, s_one, ws_d(1,1,ik), &
                               npwxl, hcs_d, npwxl, s_zero, cws_d, nw )
          !$cuf kernel do(2)
          DO jj = 1, nak
             DO iw = 1, nwl
                c = act_d(j1-1+jj)
                IF ( iw <= nkl ) THEN
                   na = kbna_d(iw) ; iof = kbof_d(iw) ; ih = iw - iof
                   is = spin_d(ik) ; ee = e_d(c)
                   s = (0.0_DP, 0.0_DP)
                   DO jh = 1, kbnh_d(iw)
                      s = s + ( deeq_d(ih,jh,na,is) - ee * qq_d(ih,jh,na) ) * cws_d(iof+jh,jj)
                   ENDDO
                   cw2s_d(iw,jj) = CMPLX( s, KIND=SP )
                ELSEIF ( iw - nkl <= noccq_d(ik) ) THEN
                   cw2s_d(iw,jj) = REAL( apv, SP ) * cws_d(iw,jj)
                ELSE
                   cw2s_d(iw,jj) = (0.0_SP, 0.0_SP)
                ENDIF
             ENDDO
          ENDDO
          istat = cublasCgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwxl, nak, nw, s_one, ws_d(1,1,ik), &
                               npwxl, cw2s_d, nw, s_zero, tcs_d, npwxl )
          !$cuf kernel do(2)
          DO jj = 1, nak
             DO ig = 1, n1
                ts_d(ig,act_d(j1-1+jj)) = ts_d(ig,act_d(j1-1+jj)) + tcs_d(ig,jj)
             ENDDO
          ENDDO
          j1 = j2 + 1
       ENDDO
       !
    ENDIF
    CALL stop_clock( 'dfptd_hpsi' )
    !
  END SUBROUTINE apply_h_sp
  !
  !-----------------------------------------------------------------------
  SUBROUTINE add_zprec_sp( nact )
    !-----------------------------------------------------------------------
    !! single-precision twin of add_zprec (operates on gs_d, ts_d)
    INTEGER, INTENT(IN) :: nact
    INTEGER :: i, ik, k1, k2, c, jz, ig, j, nck, nzl, zo, n1
    REAL(DP) :: s, de
    COMPLEX(SP), PARAMETER :: s_one = (1.0_SP, 0.0_SP), s_zero = (0.0_SP, 0.0_SP)
    !
    nck = ncolk ; nzl = nz ; zo = zoff ; n1 = npwxl
    i = 1
    DO WHILE ( i <= nact )
       k1 = (act_h(i)-1)/ncolk + 1
       k2 = k1
       DO WHILE ( i <= nact )
          ik = (act_h(i)-1)/ncolk + 1
          IF ( ik > k2+1 ) EXIT
          k2 = ik
          i = i + 1
       ENDDO
       IF ( lus ) THEN
          CALL gemm_ks( CUBLAS_OP_C, nz, ncolk, npwxl, s_one, evqs_d(1,zoff+1,k1), npwxl, &
                        INT(npwxl,8)*nbq, gs_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, s_zero, &
                        cws_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, k2-k1+1 )
       ELSE
          CALL gemm_ks( CUBLAS_OP_C, nz, ncolk, npwxl, s_one, ws_d(1,nkbl+zoff+1,k1), npwxl, &
                        INT(npwxl,8)*nw, gs_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, s_zero, &
                        cws_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, k2-k1+1 )
       ENDIF
       !$cuf kernel do(2)
       DO c = (k1-1)*nck+1, k2*nck
          DO jz = 1, nzl
             ik = (c-1)/nck + 1
             de = etq_d(zo+jz,ik) - e_d(c)
             IF ( zo+jz > noccq_d(ik) .AND. de > 1.0E-3_DP ) THEN
                cws_d(jz,c) = cws_d(jz,c) / REAL( de, SP )
             ELSE
                cws_d(jz,c) = (0.0_SP, 0.0_SP)
             ENDIF
          ENDDO
       ENDDO
       IF ( lus ) THEN
          CALL gemm_ks( CUBLAS_OP_N, npwxl, ncolk, nz, s_one, evqs_d(1,zoff+1,k1), npwxl, &
                        INT(npwxl,8)*nbq, cws_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, s_one, &
                        ts_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, k2-k1+1 )
       ELSE
          CALL gemm_ks( CUBLAS_OP_N, npwxl, ncolk, nz, s_one, ws_d(1,nkbl+zoff+1,k1), npwxl, &
                        INT(npwxl,8)*nw, cws_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, s_one, &
                        ts_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, k2-k1+1 )
       ENDIF
    ENDDO
    !
    !$acc parallel loop gang private(c, s)
    DO j = 1, nact
       c = act_d(j)
       s = 0.0_DP
       !$acc loop vector reduction(+:s)
       DO ig = 1, n1
          s = s + DBLE( CONJG(ts_d(ig,c)) * gs_d(ig,c) )
       ENDDO
       rho_d(c) = s
       rhoc_d(j) = s
    ENDDO
    !
  END SUBROUTINE add_zprec_sp
  !
  !-----------------------------------------------------------------------
  SUBROUTINE setup_real( )
    !-----------------------------------------------------------------------
    !! Real mode of the single-precision iterations, for q = 0 and k-points
    !! that are equivalent to -k (Gamma, zone-boundary points with 2k = G:
    !! the k-points of supercells). There H, S and the projector on the
    !! valence bands are real operators: with K the complex conjugation in
    !! real space, ( K x )(k+G) = conj( x(k+G') ), G' = -G-2k, every vector is
    !! x = x_s + i x_a with K x_s = x_s, K x_a = x_a, and A x = A x_s + i A x_a.
    !! A K-symmetric vector is stored by its values on one G of each pair
    !! (G, G') as a real array of twice that length; then
    !!   - scalar products are real dot products (SGEMM instead of CGEMM on
    !!     arrays of half the length),
    !!   - two vectors go through the FFTs as one complex function.
    !! The imaginary part x_a of the right-hand sides vanishes when the bands
    !! are real functions; the bands of the ground state are made so by a
    !! unitary rotation U inside each group of degenerate bands (to which the
    !! equations are covariant), computed here. What is left of x_a (the
    !! bands are converged to a threshold) is solved for only where it
    !! exceeds a fraction of the convergence threshold.
    !! Only the single-precision iterations change: the residual in double
    !! precision, on which convergence rests, is computed as before on the
    !! original bands. QE_PH_REAL=0 switches the real mode off.
    !
    USE io_global,   ONLY : stdout
    USE klist,       ONLY : xk, igk_k
    USE cell_base,   ONLY : at
    USE gvect,       ONLY : mill
    USE qpoint,      ONLY : ikks, ikqs
    !
    INTEGER :: ik, ikk, ig, j, n, npw, nh, nself, n2k(3), mm(3), istat, lenv, ierr, nbl, npl, ikl, &
               nkl, nocl
    CHARACTER(LEN=16) :: envval
    REAL(DP) :: xc(3), eps, epsr
    LOGICAL :: okk
    INTEGER, ALLOCATABLE :: inv(:), nl_h(:,:), mg_h(:), hl_h(:,:), hm_h(:,:), nlh_h(:,:), nlm_h(:,:), &
                            nh_h(:), nself_h(:), cl_h(:,:)
    REAL(DP), ALLOCATABLE :: g2_h(:,:), g2h_h(:,:), et_h(:,:), er_h(:,:)
    COMPLEX(DP), ALLOCATABLE :: m_h(:,:), u_h(:,:,:)
    COMPLEX(SP), ALLOCATABLE :: us_h(:,:,:)
    INTEGER, ALLOCATABLE, DEVICE :: mg_d(:)
    COMPLEX(DP), ALLOCATABLE, DEVICE :: kp_d(:,:), m_d(:,:), ud_d(:,:)
    !
    use_real = .FALSE.
    CALL get_environment_variable( 'QE_PH_REAL', envval, lenv, ierr )
    IF ( ierr == 0 .AND. lenv > 0 .AND. envval(1:1) == '0' ) RETURN
    IF ( .NOT. ( use_sp .AND. lgamma_l ) .OR. nbo < 1 ) RETURN
    DO ik = 1, nks
       IF ( ikqs(ik) /= ikks(ik) .OR. nocc_h(ik) /= noccq_h(ik) .OR. nocc_h(ik) < 1 ) RETURN
    ENDDO
    nbl = nbo ; npl = npwxl ; nkl = nkbl
    !
    ! ... pairs (G, G') of each k-point
    !
    ALLOCATE( inv(nnr), nl_h(npwxl,nks), mg_h(npwxl), hl_h(npwxl,nks), hm_h(npwxl,nks), &
              nh_h(nks), nself_h(nks) )
    nl_h = nlq_d
    hl_h = 1 ; hm_h = 1
    DO ik = 1, nks
       ikk = ikks(ik)
       npw = npwq_h(ik)
       xc(:) = 2.0_DP * ( xk(1,ikk)*at(1,:) + xk(2,ikk)*at(2,:) + xk(3,ikk)*at(3,:) )
       n2k(:) = NINT( xc(:) )
       IF ( ANY( ABS( xc(:) - n2k(:) ) > 1.0E-8_DP ) ) RETURN
       inv = 0
       DO ig = 1, npw
          inv(nl_h(ig,ik)) = ig
       ENDDO
       DO ig = 1, npw
          mm(:) = mill(:,igk_k(ig,ikk))
          IF ( box_index( mm ) /= nl_h(ig,ik) ) RETURN
          mm(:) = - mm(:) - n2k(:)
          mg_h(ig) = inv( box_index( mm ) )
          IF ( mg_h(ig) == 0 ) RETURN
       ENDDO
       nself = 0
       DO ig = 1, npw
          IF ( mg_h(mg_h(ig)) /= ig ) RETURN
          IF ( mg_h(ig) == ig ) nself = nself + 1
       ENDDO
       IF ( nself > 1 ) RETURN
       nh = 0
       DO ig = 1, npw
          IF ( mg_h(ig) == ig ) THEN
             nh = nh + 1
             hl_h(nh,ik) = ig ; hm_h(nh,ik) = ig
          ENDIF
       ENDDO
       DO ig = 1, npw
          IF ( mg_h(ig) > ig ) THEN
             nh = nh + 1
             hl_h(nh,ik) = ig ; hm_h(nh,ik) = mg_h(ig)
          ENDIF
       ENDDO
       nh_h(ik) = nh ; nself_h(ik) = nself
    ENDDO
    nhx = MAXVAL( nh_h )
    !
    ! ... rotation of the bands: M = <psi| S K |psi>, see real_rotation
    !
    ALLOCATE( u_h(nbo,nbo,nks), m_h(nbo,nbo), cl_h(nbo,nks), er_h(nbo,nks), et_h(nbo,nks) )
    ALLOCATE( mg_d(npwxl), kp_d(npwxl,nbo), m_d(nbo,nbo), ud_d(nbo,nbo) )
    et_h = et_d
    IF ( .NOT. blas_ready ) CALL init_blas( )
    DO ik = 1, nks
       mg_h = 1
       DO j = 1, nh_h(ik)
          mg_h(hl_h(j,ik)) = hm_h(j,ik)
          mg_h(hm_h(j,ik)) = hl_h(j,ik)
       ENDDO
       mg_d = mg_h
       npw = npwq_h(ik) ; ikl = ik
       !$cuf kernel do(2)
       DO n = 1, nbl
          DO ig = 1, npl
             IF ( ig <= npw ) THEN
                kp_d(ig,n) = CONJG( psik_d(mg_d(ig),n,ikl) )
             ELSE
                kp_d(ig,n) = (0.0_DP, 0.0_DP)
             ENDIF
          ENDDO
       ENDDO
       istat = cublasZgemm( hblas, CUBLAS_OP_C, CUBLAS_OP_N, nbo, nbo, npwxl, c_one, &
                            w_d(1,nkbl+1,ik), npwxl, kp_d, npwxl, c_zero, m_d, nbo )
       m_h = m_d
       CALL real_rotation( nbo, nocc_h(ik), et_h(1,ik), m_h, u_h(1,1,ik), er_h(1,ik), cl_h(1,ik), okk )
       IF ( .NOT. okk ) THEN
          WRITE( stdout, '(5x,"DFPT device engine: the bands cannot be made real, complex iterations")' )
          RETURN
       ENDIF
    ENDDO
    !
    ! ... device data: maps, kinetic energies, rotation, real projectors
    !
    ALLOCATE( nlh_h(nhx,nks), nlm_h(nhx,nks), g2_h(npwxl,nks), g2h_h(nhx,nks) )
    g2_h = g2_d
    nlh_h = 1 ; nlm_h = 1 ; g2h_h = 0.0_DP
    DO ik = 1, nks
       DO j = 1, nh_h(ik)
          nlh_h(j,ik) = nl_h(hl_h(j,ik),ik)
          nlm_h(j,ik) = nl_h(hm_h(j,ik),ik)
          g2h_h(j,ik) = g2_h(hl_h(j,ik),ik)
       ENDDO
    ENDDO
    ALLOCATE( nh_d(nks), nself_d(nks), hl_d(nhx,nks), hm_d(nhx,nks), nlh_d(nhx,nks), nlm_d(nhx,nks), &
              g2h_d(nhx,nks) )
    nh_d = nh_h ; nself_d = nself_h
    hl_d = hl_h(1:nhx,:) ; hm_d = hm_h(1:nhx,:)
    nlh_d = nlh_h ; nlm_d = nlm_h ; g2h_d = g2h_h
    ALLOCATE( us_h(nbo,nbo,nks), ur_d(nbo,nbo,nks) )
    us_h = CMPLX( u_h, KIND=SP )
    ur_d = us_h
    ALLOCATE( wr_d(2*nhx,nw,nks) )
    wr_d = 0.0_SP
    epsr = 0.0_DP
    DO ik = 1, nks
       IF ( nkbl > 0 ) CALL compress_dp( ik, nkbl, w_d(1,1,ik), wr_d(1,1,ik), eps )
       ud_d = u_h(:,:,ik)
       istat = cublasZgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwxl, nbo, nbo, c_one, &
                            w_d(1,nkbl+1,ik), npwxl, ud_d, nbo, c_zero, kp_d, npwxl )
       nocl = nocc_h(ik)
       CALL compress_dp( ik, nocl, kp_d, wr_d(1,nkbl+1,ik), eps )
       epsr = MAX( epsr, eps )
    ENDDO
    !
    IF ( epsr > 1.0E-4_DP ) THEN
       ! the bands are not converged well enough for the real operator to be a good
       ! approximation of the operator of the residual
       DEALLOCATE( nh_d, nself_d, hl_d, hm_d, nlh_d, nlm_d, g2h_d, wr_d, ur_d )
       WRITE( stdout, '(5x,"DFPT device engine: imaginary part of the bands ",es8.1, &
            &", complex iterations")' ) epsr
       RETURN
    ENDIF
    IF ( ALLOCATED(clus_h) ) DEALLOCATE( clus_h )
    IF ( ALLOCATED(erot_h) ) DEALLOCATE( erot_h )
    ALLOCATE( clus_h(nbo,nks), erot_h(nbo,nks) )
    clus_h = cl_h ; erot_h = er_h
    eps_real = epsr
    use_real = .TRUE.
    IF ( ALLOCATED(rnh_h) ) DEALLOCATE( rnh_h, rns_h, rhl_h, rhm_h, rg2_h )
    ALLOCATE( rnh_h(nks), rns_h(nks), rhl_h(nhx,nks), rhm_h(nhx,nks), rg2_h(nhx,nks) )
    rnh_h = nh_h ; rns_h = nself_h
    rhl_h = hl_h(1:nhx,:) ; rhm_h = hm_h(1:nhx,:) ; rg2_h = g2h_h
    WRITE( stdout, '(5x,"DFPT device engine: k = -k, real iterations (two bands per FFT, real ", &
         &"projections); imaginary part of the bands ",es8.1)' ) epsr
    !
  CONTAINS
    !
    INTEGER FUNCTION box_index( m )
      !! position on the FFT grid of the G-vector with Miller indices m
      INTEGER, INTENT(IN) :: m(3)
      box_index = 1 + MODULO( m(1), nr1 ) + nr1 * ( MODULO( m(2), nr2 ) + nr2 * MODULO( m(3), nr3 ) )
    END FUNCTION box_index
    !
  END SUBROUTINE setup_real
  !
  !-----------------------------------------------------------------------
  SUBROUTINE real_rotation( nb, nocc, e, mmat, u, er, clus, ok )
    !-----------------------------------------------------------------------
    !! Unitary u (block diagonal over the groups of degenerate bands among
    !! the first nocc) such that the bands psi u are invariant under the
    !! complex conjugation K, given mmat = <psi| S K |psi>: K psi = psi mmat.
    !! In a group mmat is unitary and symmetric, mmat = o d o^T with o real
    !! orthogonal and d a diagonal of phases (its real and imaginary parts
    !! commute), and u = o sqrt(d). The rotated bands of a group are then
    !! mixed by a real orthogonal matrix that diagonalizes their eigenvalue
    !! matrix (the group may be degenerate only within a tolerance); er are
    !! the eigenvalues. ok = .FALSE.: the bands of some group are not closed
    !! under K to the accuracy needed.
    !
    INTEGER, INTENT(IN) :: nb, nocc
    REAL(DP), INTENT(IN) :: e(nb)
    COMPLEX(DP), INTENT(IN) :: mmat(nb,nb)
    COMPLEX(DP), INTENT(OUT) :: u(nb,nb)
    REAL(DP), INTENT(OUT) :: er(nb)
    INTEGER, INTENT(OUT) :: clus(nb)
    LOGICAL, INTENT(OUT) :: ok
    !
    REAL(DP), PARAMETER :: tolc = 1.0E-6_DP, tolm = 1.0E-3_DP
    REAL(DP), PARAMETER :: tval(3) = (/ 0.6180339887_DP, 0.3141592654_DP, 1.7320508076_DP /)
    INTEGER :: n, m, n1, n2, d, i, j, itry, info
    REAL(DP) :: th, dev
    LOGICAL :: found
    REAL(DP), ALLOCATABLE :: aa(:,:), bb(:,:), cc(:,:), ww(:), work(:)
    COMPLEX(DP), ALLOCATABLE :: mb(:,:), dm(:,:), vv(:,:), eh(:,:)
    !
    ok = .FALSE.
    u = (0.0_DP, 0.0_DP)
    DO n = 1, nb
       u(n,n) = (1.0_DP, 0.0_DP)
       er(n) = e(n)
       clus(n) = n
    ENDDO
    n1 = 1
    DO WHILE ( n1 <= nocc )
       n2 = n1
       DO WHILE ( n2 < nocc )
          IF ( e(n2+1) - e(n2) > tolc ) EXIT
          n2 = n2 + 1
       ENDDO
       d = n2 - n1 + 1
       clus(n1:n2) = n1
       DO m = 1, nocc
          IF ( m >= n1 .AND. m <= n2 ) CYCLE
          IF ( MAXVAL( ABS( mmat(m,n1:n2) ) ) > tolm ) RETURN
       ENDDO
       IF ( d == 1 ) THEN
          IF ( ABS( ABS( mmat(n1,n1) ) - 1.0_DP ) > tolm ) RETURN
          th = ATAN2( AIMAG( mmat(n1,n1) ), DBLE( mmat(n1,n1) ) )
          u(n1,n1) = CMPLX( COS(0.5_DP*th), SIN(0.5_DP*th), KIND=DP )
       ELSE
          ALLOCATE( mb(d,d), dm(d,d), vv(d,d), eh(d,d), aa(d,d), bb(d,d), cc(d,d), ww(d), work(34*d) )
          mb(:,:) = mmat(n1:n2,n1:n2)
          aa(:,:) = 0.5_DP * ( DBLE(mb) + TRANSPOSE( DBLE(mb) ) )
          bb(:,:) = 0.5_DP * ( AIMAG(mb) + TRANSPOSE( AIMAG(mb) ) )
          found = .FALSE.
          DO itry = 1, 3
             cc(:,:) = aa(:,:) + tval(itry) * bb(:,:)
             CALL DSYEV( 'V', 'U', d, cc, d, ww, work, SIZE(work), info )
             IF ( info /= 0 ) CYCLE
             dm = MATMUL( TRANSPOSE(cc), MATMUL( mb, CMPLX( cc, 0.0_DP, KIND=DP ) ) )
             dev = 0.0_DP
             DO j = 1, d
                DO i = 1, d
                   IF ( i == j ) THEN
                      dev = MAX( dev, ABS( ABS(dm(i,i)) - 1.0_DP ) )
                   ELSE
                      dev = MAX( dev, ABS( dm(i,j) ) )
                   ENDIF
                ENDDO
             ENDDO
             IF ( dev < tolm ) THEN
                found = .TRUE.
                EXIT
             ENDIF
          ENDDO
          IF ( .NOT. found ) RETURN
          DO j = 1, d
             th = ATAN2( AIMAG( dm(j,j) ), DBLE( dm(j,j) ) )
             vv(:,j) = cc(:,j) * CMPLX( COS(0.5_DP*th), SIN(0.5_DP*th), KIND=DP )
          ENDDO
          DO j = 1, d
             DO i = 1, d
                eh(i,j) = SUM( CONJG( vv(:,i) ) * e(n1:n2) * vv(:,j) )
             ENDDO
          ENDDO
          aa(:,:) = 0.5_DP * ( DBLE(eh) + TRANSPOSE( DBLE(eh) ) )
          CALL DSYEV( 'V', 'U', d, aa, d, ww, work, SIZE(work), info )
          IF ( info /= 0 ) RETURN
          u(n1:n2,n1:n2) = MATMUL( vv, CMPLX( aa, 0.0_DP, KIND=DP ) )
          er(n1:n2) = ww(1:d)
          DEALLOCATE( mb, dm, vv, eh, aa, bb, cc, ww, work )
       ENDIF
       n1 = n2 + 1
    ENDDO
    ok = .TRUE.
    !
  END SUBROUTINE real_rotation
  !
  !-----------------------------------------------------------------------
  SUBROUTINE compress_dp( ik, nc, src_d, dst_d, eps )
    !-----------------------------------------------------------------------
    !! K-symmetric part of nc vectors of k-point ik in the real storage
    !! (the self-conjugate component scaled by 1/sqrt(2), so that scalar
    !! products are twice the real dot products); eps: largest norm of the
    !! part that is dropped.
    INTEGER, INTENT(IN) :: ik, nc
    COMPLEX(DP), DEVICE, INTENT(IN) :: src_d(npwxl,nc)
    REAL(SP), DEVICE, INTENT(INOUT) :: dst_d(2*nhx,nc)
    REAL(DP), INTENT(OUT) :: eps
    INTEGER :: n, j, nhl, ikl
    REAL(DP) :: s, smax
    COMPLEX(DP) :: x1, x2, xs
    REAL(DP), PARAMETER :: rt2i = 0.70710678118654752_DP
    !
    nhl = nhx ; ikl = ik
    smax = 0.0_DP
    !$acc parallel loop gang private(s) reduction(max:smax)
    DO n = 1, nc
       s = 0.0_DP
       !$acc loop vector reduction(+:s) private(x1, x2, xs)
       DO j = 1, nhl
          IF ( j <= nh_d(ikl) ) THEN
             x1 = src_d(hl_d(j,ikl),n)
             IF ( j <= nself_d(ikl) ) THEN
                dst_d(2*j-1,n) = REAL( DBLE(x1) * rt2i, SP )
                dst_d(2*j,n) = 0.0_SP
                s = s + AIMAG(x1)**2
             ELSE
                x2 = src_d(hm_d(j,ikl),n)
                xs = 0.5_DP * ( x1 + CONJG(x2) )
                dst_d(2*j-1,n) = REAL( DBLE(xs), SP )
                dst_d(2*j,n) = REAL( AIMAG(xs), SP )
                s = s + 0.5_DP * DBLE( CONJG(x1 - CONJG(x2)) * (x1 - CONJG(x2)) )
             ENDIF
          ELSE
             dst_d(2*j-1,n) = 0.0_SP
             dst_d(2*j,n) = 0.0_SP
          ENDIF
       ENDDO
       smax = MAX( smax, s )
    ENDDO
    eps = SQRT( smax )
    !
  END SUBROUTINE compress_dp
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION kof( rc )
    !-----------------------------------------------------------------------
    !! k-point of a column of the real mode
    INTEGER, INTENT(IN) :: rc
    INTEGER :: c
    c = rc
    IF ( rc > ncol ) c = rc - ncol
    kof = (c-1)/ncolk + 1
  END FUNCTION kof
  !
  !-----------------------------------------------------------------------
  SUBROUTINE real_list( actf, nar )
    !-----------------------------------------------------------------------
    !! list of the active columns of the real mode, k-point by k-point
    LOGICAL, INTENT(IN) :: actf(2*ncol)
    INTEGER, INTENT(OUT) :: nar
    INTEGER :: ik, ipart, c
    nar = 0
    DO ik = 1, nks
       DO ipart = 0, 1
          DO c = (ik-1)*ncolk+1, ik*ncolk
             IF ( actf(ipart*ncol+c) ) THEN
                nar = nar + 1
                actr_h(nar) = ipart*ncol + c
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    IF ( nar > 0 ) actr_d(1:nar) = actr_h(1:nar)
  END SUBROUTINE real_list
  !
  !-----------------------------------------------------------------------
  SUBROUTINE to_real( nact )
    !-----------------------------------------------------------------------
    !! gradient g_d of the nact columns listed in act_d (complete groups of
    !! degenerate bands): to single precision, to the rotated bands, split
    !! into its K-symmetric parts gr_d(:,c) + i gr_d(:,ncol+c); the
    !! corrections dr_d of these columns are set to zero.
    INTEGER, INTENT(IN) :: nact
    INTEGER :: j, c, ig, jg, ik, n1, n2, nck, nhl, i, blk, c0, lastblk, istat
    COMPLEX(SP) :: x1, x2, xs, xa
    COMPLEX(SP), PARAMETER :: s_one = (1.0_SP, 0.0_SP), s_zero = (0.0_SP, 0.0_SP)
    REAL(SP), PARAMETER :: rt2i = 0.70710678_SP
    !
    n1 = npwxl ; n2 = ncol ; nck = ncolk ; nhl = nhx
    !$acc parallel loop gang private(c)
    DO j = 1, nact
       c = act_d(j)
       !$acc loop vector
       DO ig = 1, n1
          gs_d(ig,c) = CMPLX( g_d(ig,c), KIND=SP )
       ENDDO
    ENDDO
    lastblk = 0
    DO i = 1, nact
       blk = (act_h(i)-1)/nbo + 1
       IF ( blk == lastblk ) CYCLE
       lastblk = blk
       c0 = (blk-1)*nbo
       ik = c0/ncolk + 1
       istat = cublasCgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, npwxl, nbo, nbo, s_one, gs_d(1,c0+1), &
                            npwxl, ur_d(1,1,ik), nbo, s_zero, hs_d(1,c0+1), npwxl )
       IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cublasCgemm failed', ABS(istat) )
    ENDDO
    !$acc parallel loop gang private(c, ik)
    DO j = 1, nact
       c = act_d(j)
       ik = (c-1)/nck + 1
       !$acc loop vector private(x1, x2, xs, xa)
       DO jg = 1, nhl
          IF ( jg <= nh_d(ik) ) THEN
             x1 = hs_d(hl_d(jg,ik),c)
             IF ( jg <= nself_d(ik) ) THEN
                gr_d(2*jg-1,c) = REAL(x1) * rt2i
                gr_d(2*jg,c) = 0.0_SP
                gr_d(2*jg-1,n2+c) = AIMAG(x1) * rt2i
                gr_d(2*jg,n2+c) = 0.0_SP
             ELSE
                x2 = hs_d(hm_d(jg,ik),c)
                xs = 0.5_SP * ( x1 + CONJG(x2) )
                xa = ( x1 - CONJG(x2) ) * CMPLX( 0.0_SP, -0.5_SP )
                gr_d(2*jg-1,c) = REAL(xs)
                gr_d(2*jg,c) = AIMAG(xs)
                gr_d(2*jg-1,n2+c) = REAL(xa)
                gr_d(2*jg,n2+c) = AIMAG(xa)
             ENDIF
          ELSE
             gr_d(2*jg-1,c) = 0.0_SP
             gr_d(2*jg,c) = 0.0_SP
             gr_d(2*jg-1,n2+c) = 0.0_SP
             gr_d(2*jg,n2+c) = 0.0_SP
          ENDIF
          dr_d(2*jg-1,c) = 0.0_SP
          dr_d(2*jg,c) = 0.0_SP
          dr_d(2*jg-1,n2+c) = 0.0_SP
          dr_d(2*jg,n2+c) = 0.0_SP
       ENDDO
    ENDDO
    !
  END SUBROUTINE to_real
  !
  !-----------------------------------------------------------------------
  SUBROUTINE from_real( nact )
    !-----------------------------------------------------------------------
    !! corrections dr_d of the nact columns listed in act_d: back to complex
    !! vectors and to the original bands, added to dpsi_d.
    INTEGER, INTENT(IN) :: nact
    INTEGER :: j, c, ig, jg, ik, n1, n2, nck, nhl, i, blk, c0, lastblk, istat
    COMPLEX(SP) :: ds, da
    COMPLEX(SP), PARAMETER :: s_one = (1.0_SP, 0.0_SP), s_zero = (0.0_SP, 0.0_SP), s_i = (0.0_SP, 1.0_SP)
    REAL(SP), PARAMETER :: rt2 = 1.41421356_SP
    !
    n1 = npwxl ; n2 = ncol ; nck = ncolk ; nhl = nhx
    !$acc parallel loop gang private(c, ik)
    DO j = 1, nact
       c = act_d(j)
       ik = (c-1)/nck + 1
       !$acc loop vector private(ds, da)
       DO jg = 1, nhl
          IF ( jg <= nh_d(ik) ) THEN
             ds = CMPLX( dr_d(2*jg-1,c), dr_d(2*jg,c) )
             da = CMPLX( dr_d(2*jg-1,n2+c), dr_d(2*jg,n2+c) )
             IF ( jg <= nself_d(ik) ) THEN
                ts_d(hl_d(jg,ik),c) = rt2 * CMPLX( REAL(ds), REAL(da) )
             ELSE
                ts_d(hl_d(jg,ik),c) = ds + s_i * da
                ts_d(hm_d(jg,ik),c) = CONJG(ds) + s_i * CONJG(da)
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    lastblk = 0
    DO i = 1, nact
       blk = (act_h(i)-1)/nbo + 1
       IF ( blk == lastblk ) CYCLE
       lastblk = blk
       c0 = (blk-1)*nbo
       ik = c0/ncolk + 1
       istat = cublasCgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_C, npwxl, nbo, nbo, s_one, ts_d(1,c0+1), &
                            npwxl, ur_d(1,1,ik), nbo, s_zero, hs_d(1,c0+1), npwxl )
       IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cublasCgemm failed', ABS(istat) )
    ENDDO
    !$acc parallel loop gang private(c)
    DO j = 1, nact
       c = act_d(j)
       !$acc loop vector
       DO ig = 1, n1
          dpsi_d(ig,c) = dpsi_d(ig,c) + hs_d(ig,c)
       ENDDO
    ENDDO
    !
  END SUBROUTINE from_real
  !
  !-----------------------------------------------------------------------
  SUBROUTINE apply_h_real( nar )
    !-----------------------------------------------------------------------
    !! tr_d = ( H - e S + alpha_pv P_v ) hr_d for the nar columns of the real
    !! mode listed in actr_d (ordered by k-point). Two columns of a k-point
    !! share the FFTs: x1 + i x2 is one complex function of r, and after the
    !! product with the (real) potential the two are separated again with
    !! the values at G and G'. Nonlocal and projector terms: real matrix
    !! products on the active columns of each k-point.
    !
    INTEGER, INTENT(IN) :: nar
    !
    INTEGER :: np, j, p0, nb, ip, jg, rc1, rc2, c1, c2, ik, istat, ir, n2, nck, nhl, nwl, nkl, &
               j1, j2, nak, jj, iw, rc, c, na, iof, ih, jh, is, nrr
    REAL(SP) :: scale
    COMPLEX(SP) :: za, zb, z1, z2, y1, y2
    REAL(DP) :: apv, ee, s
    REAL(SP), PARAMETER :: rt2 = 1.41421356_SP, rt2i = 0.70710678_SP
    COMPLEX(SP), PARAMETER :: s_i = (0.0_SP, 1.0_SP)
    !
    CALL start_clock( 'dfptd_hpsi' )
    n2 = ncol ; nck = ncolk ; nhl = nhx ; nwl = nw ; nkl = nkbl ; apv = alpha_pv_l ; nrr = 2*nhx
    scale = 1.0_SP / REAL(nnr,SP)
    !
    ! ... pairs of active columns of the same k-point
    !
    np = 0
    j = 1
    DO WHILE ( j <= nar )
       np = np + 1
       pa_h(1,np) = actr_h(j)
       pa_h(2,np) = 0
       IF ( j < nar ) THEN
          IF ( kof( actr_h(j+1) ) == kof( actr_h(j) ) ) THEN
             pa_h(2,np) = actr_h(j+1)
             j = j + 1
          ENDIF
       ENDIF
       j = j + 1
    ENDDO
    pa_d(:,1:np) = pa_h(:,1:np)
    !
    ! ... kinetic energy and local potential
    !
    DO p0 = 0, np-1, nfb
       nb = MIN( nfb, np-p0 )
       ip = pick_plan( nb )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO jg = 1, nhl
             rc1 = pa_d(1,p0+j) ; rc2 = pa_d(2,p0+j)
             c1 = rc1
             IF ( rc1 > n2 ) c1 = rc1 - n2
             ik = (c1-1)/nck + 1
             IF ( jg <= nh_d(ik) ) THEN
                za = CMPLX( hr_d(2*jg-1,rc1), hr_d(2*jg,rc1) )
                zb = (0.0_SP, 0.0_SP)
                IF ( rc2 > 0 ) zb = CMPLX( hr_d(2*jg-1,rc2), hr_d(2*jg,rc2) )
                IF ( jg <= nself_d(ik) ) THEN
                   psigs_d(nlh_d(jg,ik) + (j-1)*nnr) = rt2 * CMPLX( REAL(za), REAL(zb) )
                ELSE
                   psigs_d(nlh_d(jg,ik) + (j-1)*nnr) = za + s_i * zb
                   psigs_d(nlm_d(jg,ik) + (j-1)*nnr) = CONJG(za) + s_i * CONJG(zb)
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       istat = cufftExecC2C( plans(ip), psigs_d, psics_d, CUFFT_INVERSE )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO jg = 1, nhl
             rc1 = pa_d(1,p0+j)
             c1 = rc1
             IF ( rc1 > n2 ) c1 = rc1 - n2
             ik = (c1-1)/nck + 1
             IF ( jg <= nh_d(ik) ) THEN
                psigs_d(nlh_d(jg,ik) + (j-1)*nnr) = (0.0_SP, 0.0_SP)
                psigs_d(nlm_d(jg,ik) + (j-1)*nnr) = (0.0_SP, 0.0_SP)
             ENDIF
          ENDDO
       ENDDO
       !$cuf kernel do(2)
       DO j = 1, nb
          DO ir = 1, nnr
             rc1 = pa_d(1,p0+j)
             c1 = rc1
             IF ( rc1 > n2 ) c1 = rc1 - n2
             ik = (c1-1)/nck + 1
             psics_d(ir + (j-1)*nnr) = psics_d(ir + (j-1)*nnr) * REAL( v_d(ir,spin_d(ik)), SP )
          ENDDO
       ENDDO
       istat = cufftExecC2C( plans(ip), psics_d, psics_d, CUFFT_FORWARD )
       !$cuf kernel do(2)
       DO j = 1, nb
          DO jg = 1, nhl
             rc1 = pa_d(1,p0+j) ; rc2 = pa_d(2,p0+j)
             c1 = rc1
             IF ( rc1 > n2 ) c1 = rc1 - n2
             ik = (c1-1)/nck + 1
             IF ( jg <= nh_d(ik) ) THEN
                z1 = psics_d(nlh_d(jg,ik) + (j-1)*nnr) * scale
                IF ( jg <= nself_d(ik) ) THEN
                   y1 = CMPLX( REAL(z1) * rt2i, 0.0_SP )
                   y2 = CMPLX( AIMAG(z1) * rt2i, 0.0_SP )
                ELSE
                   z2 = psics_d(nlm_d(jg,ik) + (j-1)*nnr) * scale
                   y1 = 0.5_SP * ( z1 + CONJG(z2) )
                   y2 = ( z1 - CONJG(z2) ) * CMPLX( 0.0_SP, -0.5_SP )
                ENDIF
                ee = g2h_d(jg,ik) - er_d(c1)
                tr_d(2*jg-1,rc1) = REAL( ee, SP ) * hr_d(2*jg-1,rc1) + REAL(y1)
                tr_d(2*jg,rc1) = REAL( ee, SP ) * hr_d(2*jg,rc1) + AIMAG(y1)
                IF ( rc2 > 0 ) THEN
                   c2 = rc2
                   IF ( rc2 > n2 ) c2 = rc2 - n2
                   ee = g2h_d(jg,ik) - er_d(c2)
                   tr_d(2*jg-1,rc2) = REAL( ee, SP ) * hr_d(2*jg-1,rc2) + REAL(y2)
                   tr_d(2*jg,rc2) = REAL( ee, SP ) * hr_d(2*jg,rc2) + AIMAG(y2)
                ENDIF
             ELSE
                tr_d(2*jg-1,rc1) = 0.0_SP
                tr_d(2*jg,rc1) = 0.0_SP
                IF ( rc2 > 0 ) THEN
                   tr_d(2*jg-1,rc2) = 0.0_SP
                   tr_d(2*jg,rc2) = 0.0_SP
                ENDIF
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    !
    ! ... nonlocal potential, e*S and projector on the valence bands
    !
    j1 = 1
    DO WHILE ( j1 <= nar )
       ik = kof( actr_h(j1) )
       j2 = j1
       DO WHILE ( j2 < nar )
          IF ( kof( actr_h(j2+1) ) /= ik ) EXIT
          j2 = j2 + 1
       ENDDO
       nak = j2 - j1 + 1
       !$cuf kernel do(2)
       DO jj = 1, nak
          DO ir = 1, nrr
             hcr_d(ir,jj) = hr_d(ir,actr_d(j1-1+jj))
          ENDDO
       ENDDO
       istat = cublasSgemm( hblas, CUBLAS_OP_T, CUBLAS_OP_N, nw, nak, nrr, 2.0_SP, wr_d(1,1,ik), nrr, &
                            hcr_d, nrr, 0.0_SP, cwr_d, nw )
       IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cublasSgemm failed', ABS(istat) )
       !$cuf kernel do(2)
       DO jj = 1, nak
          DO iw = 1, nwl
             rc = actr_d(j1-1+jj)
             c = rc
             IF ( rc > n2 ) c = rc - n2
             IF ( iw <= nkl ) THEN
                na = kbna_d(iw) ; iof = kbof_d(iw) ; ih = iw - iof
                is = spin_d(ik) ; ee = er_d(c)
                s = 0.0_DP
                DO jh = 1, kbnh_d(iw)
                   s = s + ( deeq_d(ih,jh,na,is) - ee * qq_d(ih,jh,na) ) * cwr_d(iof+jh,jj)
                ENDDO
                cw2r_d(iw,jj) = REAL( s, SP )
             ELSEIF ( iw - nkl <= noccq_d(ik) ) THEN
                cw2r_d(iw,jj) = REAL( apv, SP ) * cwr_d(iw,jj)
             ELSE
                cw2r_d(iw,jj) = 0.0_SP
             ENDIF
          ENDDO
       ENDDO
       istat = cublasSgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, nrr, nak, nw, 1.0_SP, wr_d(1,1,ik), nrr, &
                            cw2r_d, nw, 0.0_SP, tcr_d, nrr )
       IF ( istat /= 0 ) CALL errore( 'dfpt_dev', 'cublasSgemm failed', ABS(istat) )
       !$cuf kernel do(2)
       DO jj = 1, nak
          DO ir = 1, nrr
             tr_d(ir,actr_d(j1-1+jj)) = tr_d(ir,actr_d(j1-1+jj)) + tcr_d(ir,jj)
          ENDDO
       ENDDO
       j1 = j2 + 1
    ENDDO
    CALL stop_clock( 'dfptd_hpsi' )
    !
  END SUBROUTINE apply_h_real
  !
  !-----------------------------------------------------------------------
  SUBROUTINE setup_lowg( )
    !-----------------------------------------------------------------------
    !! Preconditioner of the single-precision iterations with an exact block
    !! on the plane waves of low kinetic energy. The stock preconditioner,
    !! 1/max(1, |k+q+G|^2/eprec), ignores the potential, which is what
    !! couples the plane waves below a few Ry. Here, for the nlg plane waves
    !! with |k+q+G|^2 <= elow, the block L of B = H + alpha_pv P_v and of S
    !! is built and diagonalized, B_LL v = lambda S_LL v, so that the inverse
    !! of the block of the operator of band n, ( B - e_n S )_LL, is
    !! v ( lambda - e_n )^-1 v^H; the other plane waves get the inverse of
    !! the diagonal of the operator, 1/( h_GG - e_n s_GG ). The block is a
    !! principal submatrix of a positive definite operator, so this is a
    !! valid (positive definite) preconditioner. Convergence is still tested
    !! on the norm that the stock preconditioner defines.
    !! The block is diagonalized on the host (one thread per k-point), which
    !! costs about 1e-10 s * nks * nlg^3; the solves of a q-point that it
    !! accelerates cost about 1e-8 s * modes * bands * nks * (FFT points), so
    !! the block is limited to the nlg lowest plane waves with
    !! nlg^3 <= 2 * modes * bands * nnr (QE_PH_LOWG_CAP replaces the 2; between
    !! 0.8 and 6.4 the sum of set-up and solves hardly changes), and by the
    !! memory of the eigenvectors.
    !! QE_PH_LOWG = elow in Ry (default 24; 0: stock preconditioner).
    !
    USE io_global,   ONLY : stdout
    USE klist,       ONLY : igk_k
    USE gvect,       ONLY : mill
    USE qpoint,      ONLY : ikqs
    !
    INTEGER, PARAMETER :: nlmax = 1024
    REAL(DP), PARAMETER :: elow_def = 24.0_DP, cap_fac = 2.0_DP
    INTEGER :: ik, ikq, ig, i, j, n, nl, is, ikb, jkb, na, iof, ih, jh, v, istat, lenv, ierr, info, &
               mm(3), nwl, nkl, npl, ikl, nll, lwork, lrwork, liwork, nbad, ip, nrl, nkc, k1, k2
    CHARACTER(LEN=16) :: envval
    REAL(DP) :: elow, emax, apv, v0, hh, ss, capf
    INTEGER :: nm
    INTEGER, ALLOCATABLE :: nl_h(:), li_h(:,:), kbna_h(:), kbof_h(:), kbnh_h(:), iwork(:), ind(:)
    REAL(DP), ALLOCATABLE :: g2_h(:,:), le_h(:,:), deeq_h(:,:,:,:), qq_h(:,:,:), et_h(:,:), w(:), rwork(:), &
                             ek_h(:), gs(:)
    LOGICAL, ALLOCATABLE :: bad_h(:)
    INTEGER :: ncap
    COMPLEX(DP), ALLOCATABLE :: vg_h(:,:), wla(:,:,:), x(:,:), b(:,:), s(:,:), work(:)
    COMPLEX(SP), ALLOCATABLE :: lv_h(:,:,:)
    COMPLEX(DP), ALLOCATABLE, DEVICE :: wl_d(:,:,:)
    !
    use_lg = .FALSE.
    elow = elow_def
    CALL get_environment_variable( 'QE_PH_LOWG', envval, lenv, ierr )
    IF ( ierr == 0 .AND. lenv > 0 ) THEN
       READ( envval, *, IOSTAT=ierr ) elow
       IF ( ierr /= 0 ) elow = elow_def
    ENDIF
    IF ( elow <= 0.0_DP .OR. .NOT. use_sp .OR. nbo < 1 ) RETURN
    IF ( .NOT. lowg_cell_ok( ) ) RETURN
    CALL start_clock( 'dfptd_lowg' )
    nwl = nw ; nkl = nkbl ; npl = npwxl ; apv = alpha_pv_l ; nrl = nnr
    capf = cap_fac
    CALL get_environment_variable( 'QE_PH_LOWG_CAP', envval, lenv, ierr )
    IF ( ierr == 0 .AND. lenv > 0 ) THEN
       READ( envval, *, IOSTAT=ierr ) capf
       IF ( ierr /= 0 .OR. capf <= 0.0_DP ) capf = cap_fac
    ENDIF
    nm = nmodes_hint
    IF ( nm <= 0 ) nm = 3 * natl
    ncap = NINT( ( capf * nm * nbo * REAL(nnr,DP) )**(1.0_DP/3.0_DP) )
    ! eigenvectors of all k-points: 8 bytes * nlg^2 * nks on the device (and on the host)
    ncap = MIN( ncap, INT( SQRT( 0.05_DP * free_bytes( ) / ( 8.0_DP * nks ) ) ) )
    ncap = MAX( 16, MIN( nlmax, ncap ) )
    !
    ! ... the plane waves of the block: below elow, the ncap lowest at most
    !
    ALLOCATE( g2_h(npwxl,nks), nl_h(nks), et_h(nbo,nks), ek_h(nks), bad_h(nks), gs(npwxl), ind(npwxl) )
    g2_h = g2_d ; et_h = et_d
    DO ik = 1, nks
       ek_h(ik) = elow
       n = npwq_h(ik)
       IF ( COUNT( g2_h(1:n,ik) <= elow ) > ncap ) THEN
          gs(1:n) = g2_h(1:n,ik)
          ind(1) = 0
          CALL hpsort( n, gs, ind )
          ek_h(ik) = gs(ncap)
       ENDIF
       nl_h(ik) = COUNT( g2_h(1:n,ik) <= ek_h(ik) )
    ENDDO
    DEALLOCATE( gs, ind )
    nlg = MAXVAL( nl_h )
    IF ( nlg < 2 .OR. nlg > 2*nlmax ) THEN
       CALL stop_clock( 'dfptd_lowg' )
       RETURN
    ENDIF
    ALLOCATE( li_h(nlg,nks) )
    li_h = 1
    DO ik = 1, nks
       n = 0
       DO ig = 1, npwq_h(ik)
          IF ( g2_h(ig,ik) <= ek_h(ik) ) THEN
             n = n + 1
             li_h(n,ik) = ig
          ENDIF
       ENDDO
    ENDDO
    !
    ! ... V(G) on the FFT grid: <G_i| V |G_j> = V(G_i - G_j)
    !
    ALLOCATE( vg_h(nnr,nspl) )
    ip = pick_plan( 1 )
    DO is = 1, nspl
       ikl = is
       !$cuf kernel do(1)
       DO i = 1, nrl
          psic_d(i) = CMPLX( v_d(i,ikl), 0.0_DP, KIND=DP )
       ENDDO
       istat = cufftExecZ2Z( plan(ip), psic_d, psic_d, CUFFT_FORWARD )
       vg_h(:,is) = psic_d(1:nnr)
    ENDDO
    vg_h = vg_h / DBLE(nnr)
    !
    ALLOCATE( deeq_h(nhml,nhml,natl,nspl), qq_h(nhml,nhml,natl) )
    ALLOCATE( kbna_h(MAX(nkbl,1)), kbof_h(MAX(nkbl,1)), kbnh_h(MAX(nkbl,1)) )
    deeq_h = deeq_d ; qq_h = qq_d
    kbna_h = kbna_d ; kbof_h = kbof_d ; kbnh_h = kbnh_d
    ALLOCATE( lgn_d(nks), lgi_d(nlg,nks), lv_h(nlg,nlg,nks), le_h(nlg,nks) )
    lgn_d = nl_h ; lgi_d = li_h
    lv_h = (0.0_SP, 0.0_SP)
    le_h = 1.0E30_DP
    bad_h = .FALSE.
    nbad = 0
    nll = nlg
    !
    ! ... rows of W on the plane waves of the block, k-points in chunks of bounded size
    !
    nkc = MAX( 1, MIN( nks, INT( 2.0E9_DP / ( 16.0_DP * REAL(nlg,DP) * nw ) ) ) )
    ALLOCATE( wl_d(nlg,nw,nkc), wla(nlg,nw,nkc) )
    DO k1 = 1, nks, nkc
       k2 = MIN( nks, k1+nkc-1 )
       ikl = k1
       !$cuf kernel do(3)
       DO ik = k1, k2
          DO j = 1, nwl
             DO i = 1, nll
                wl_d(i,j,ik-ikl+1) = w_d(lgi_d(i,ik),j,ik)
             ENDDO
          ENDDO
       ENDDO
       wla = wl_d
       !$omp parallel do schedule(dynamic) default(shared) &
       !$omp    private(ik, ikq, nl, is, b, s, w, x, i, j, mm, jkb, na, iof, jh, ih, v, lwork, lrwork, liwork, &
       !$omp            work, rwork, iwork, info, emax)
       DO ik = k1, k2
          ikq = ikqs(ik)
          nl = nl_h(ik)
          is = spin_h(ik)
          IF ( nl < 1 ) CYCLE
          ALLOCATE( b(nl,nl), s(nl,nl), w(nl), x(nl,MAX(nkbl,1)) )
          b = (0.0_DP, 0.0_DP) ; s = (0.0_DP, 0.0_DP)
          DO j = 1, nl
             DO i = 1, nl
                mm(:) = mill(:,igk_k(li_h(i,ik),ikq)) - mill(:,igk_k(li_h(j,ik),ikq))
                b(i,j) = vg_h( 1 + MODULO(mm(1),nr1) + nr1*( MODULO(mm(2),nr2) + nr2*MODULO(mm(3),nr3) ), is )
             ENDDO
             b(j,j) = b(j,j) + g2_h(li_h(j,ik),ik)
             s(j,j) = (1.0_DP, 0.0_DP)
          ENDDO
          IF ( nkbl > 0 ) THEN
             ! x = W_beta D, then B += x W_beta^H ; the same with q for S
             x = (0.0_DP, 0.0_DP)
             DO jkb = 1, nkbl
                na = kbna_h(jkb) ; iof = kbof_h(jkb) ; jh = jkb - iof
                DO ih = 1, kbnh_h(jkb)
                   x(1:nl,jkb) = x(1:nl,jkb) + wla(1:nl,iof+ih,ik-k1+1) * deeq_h(ih,jh,na,is)
                ENDDO
             ENDDO
             CALL ZGEMM( 'N', 'C', nl, nl, nkbl, (1.0_DP,0.0_DP), x, nl, wla(1,1,ik-k1+1), nlg, &
                         (1.0_DP,0.0_DP), b, nl )
             IF ( lus ) THEN
                x = (0.0_DP, 0.0_DP)
                DO jkb = 1, nkbl
                   na = kbna_h(jkb) ; iof = kbof_h(jkb) ; jh = jkb - iof
                   DO ih = 1, kbnh_h(jkb)
                      x(1:nl,jkb) = x(1:nl,jkb) + wla(1:nl,iof+ih,ik-k1+1) * qq_h(ih,jh,na)
                   ENDDO
                ENDDO
                CALL ZGEMM( 'N', 'C', nl, nl, nkbl, (1.0_DP,0.0_DP), x, nl, wla(1,1,ik-k1+1), nlg, &
                            (1.0_DP,0.0_DP), s, nl )
             ENDIF
          ENDIF
          v = noccq_h(ik)
          IF ( v > 0 ) CALL ZGEMM( 'N', 'C', nl, nl, v, CMPLX(apv,0.0_DP,KIND=DP), wla(1,nkbl+1,ik-k1+1), &
                                   nlg, wla(1,nkbl+1,ik-k1+1), nlg, (1.0_DP,0.0_DP), b, nl )
          lwork = 2*nl + nl*nl + 16 ; lrwork = 1 + 5*nl + 2*nl*nl + 16 ; liwork = 3 + 5*nl + 16
          ALLOCATE( work(lwork), rwork(lrwork), iwork(liwork) )
          IF ( lus ) THEN
             CALL ZHEGVD( 1, 'V', 'U', nl, b, nl, s, nl, w, work, lwork, rwork, lrwork, iwork, liwork, info )
          ELSE
             CALL ZHEEVD( 'V', 'U', nl, b, nl, w, work, lwork, rwork, lrwork, iwork, liwork, info )
          ENDIF
          emax = MAXVAL( et_h(1:nocc_h(ik),ik) )
          IF ( info /= 0 .OR. w(1) - emax < 1.0E-3_DP ) THEN
             bad_h(ik) = .TRUE.
          ELSE
             lv_h(1:nl,1:nl,ik) = CMPLX( b(1:nl,1:nl), KIND=SP )
             le_h(1:nl,ik) = w(1:nl)
          ENDIF
          DEALLOCATE( b, s, w, x, work, rwork, iwork )
       ENDDO
       !$omp end parallel do
    ENDDO
    DEALLOCATE( wl_d, wla )
    DO ik = 1, nks
       IF ( bad_h(ik) ) nl_h(ik) = 0
    ENDDO
    nbad = COUNT( bad_h )
    !
    ALLOCATE( lgv_d(nlg,nlg,nks), lge_d(nlg,nks) )
    lgn_d = nl_h ; lgv_d = lv_h ; lge_d = le_h
    !
    ! ... diagonal of B and of S for the other plane waves
    !
    ALLOCATE( hdg_d(npwxl,nks), sdg_d(npwxl,nks) )
    DO is = 1, nspl
       v0 = DBLE( vg_h(1,is) )
       ikl = is
       !$acc parallel loop gang collapse(2) private(hh, ss, na, iof, ih)
       DO ik = 1, nks
          DO ig = 1, npl
             IF ( spin_d(ik) == ikl ) THEN
                hh = g2_d(ig,ik) + v0
                ss = 1.0_DP
                !$acc loop seq
                DO ikb = 1, nkl
                   na = kbna_d(ikb) ; iof = kbof_d(ikb) ; ih = ikb - iof
                   !$acc loop seq
                   DO jh = 1, kbnh_d(ikb)
                      hh = hh + deeq_d(ih,jh,na,ikl) * DBLE( w_d(ig,ikb,ik) * CONJG( w_d(ig,iof+jh,ik) ) )
                      ss = ss + qq_d(ih,jh,na) * DBLE( w_d(ig,ikb,ik) * CONJG( w_d(ig,iof+jh,ik) ) )
                   ENDDO
                ENDDO
                !$acc loop seq
                DO v = 1, noccq_d(ik)
                   hh = hh + apv * DBLE( w_d(ig,nkl+v,ik) * CONJG( w_d(ig,nkl+v,ik) ) )
                ENDDO
                hdg_d(ig,ik) = hh
                sdg_d(ig,ik) = ss
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    use_lg = .TRUE.
    WRITE( stdout, '(5x,"DFPT device engine: preconditioner with an exact block of ",i5, &
         &" plane waves (below ",f6.2," Ry)")' ) nlg, MINVAL( ek_h )
    IF ( nbad > 0 ) WRITE( stdout, '(5x,"DFPT device engine: ",i6," k-points without the block")' ) nbad
    CALL stop_clock( 'dfptd_lowg' )
    !
  END SUBROUTINE setup_lowg
  !
  !-----------------------------------------------------------------------
  SUBROUTINE lowg_apply_sp( nact )
    !-----------------------------------------------------------------------
    !! low-G block of the preconditioned gradient: ts_L = v (lambda - e)^-1 v^H gs_L
    !! for the active columns (all columns of runs of consecutive active
    !! k-points), then rho = <ts|gs>.
    INTEGER, INTENT(IN) :: nact
    INTEGER :: i, ik, k1, k2, c, il, ig, j, nck, nll, n1
    REAL(DP) :: s
    COMPLEX(SP), PARAMETER :: s_one = (1.0_SP, 0.0_SP), s_zero = (0.0_SP, 0.0_SP)
    !
    nck = ncolk ; nll = nlg ; n1 = npwxl
    i = 1
    DO WHILE ( i <= nact )
       k1 = (act_h(i)-1)/ncolk + 1
       k2 = k1
       DO WHILE ( i <= nact )
          ik = (act_h(i)-1)/ncolk + 1
          IF ( ik > k2+1 ) EXIT
          k2 = ik
          i = i + 1
       ENDDO
       !$cuf kernel do(2)
       DO c = (k1-1)*nck+1, k2*nck
          DO il = 1, nll
             ik = (c-1)/nck + 1
             IF ( il <= lgn_d(ik) ) THEN
                gl_d(il,c) = gs_d(lgi_d(il,ik),c)
             ELSE
                gl_d(il,c) = (0.0_SP, 0.0_SP)
             ENDIF
          ENDDO
       ENDDO
       CALL gemm_ks( CUBLAS_OP_C, nlg, ncolk, nlg, s_one, lgv_d(1,1,k1), nlg, INT(nlg,8)*nlg, &
                     gl_d(1,(k1-1)*ncolk+1), nlg, INT(nlg,8)*ncolk, s_zero, &
                     cl_d(1,(k1-1)*ncolk+1), nlg, INT(nlg,8)*ncolk, k2-k1+1 )
       !$cuf kernel do(2)
       DO c = (k1-1)*nck+1, k2*nck
          DO il = 1, nll
             ik = (c-1)/nck + 1
             cl_d(il,c) = cl_d(il,c) / REAL( MAX( 1.0E-3_DP, lge_d(il,ik) - e_d(c) ), SP )
          ENDDO
       ENDDO
       CALL gemm_ks( CUBLAS_OP_N, nlg, ncolk, nlg, s_one, lgv_d(1,1,k1), nlg, INT(nlg,8)*nlg, &
                     cl_d(1,(k1-1)*ncolk+1), nlg, INT(nlg,8)*ncolk, s_zero, &
                     gl_d(1,(k1-1)*ncolk+1), nlg, INT(nlg,8)*ncolk, k2-k1+1 )
    ENDDO
    !$acc parallel loop gang private(c, ik, s)
    DO j = 1, nact
       c = act_d(j)
       ik = (c-1)/nck + 1
       !$acc loop vector
       DO il = 1, nll
          IF ( il <= lgn_d(ik) ) ts_d(lgi_d(il,ik),c) = gl_d(il,c)
       ENDDO
       s = 0.0_DP
       !$acc loop vector reduction(+:s)
       DO ig = 1, n1
          s = s + DBLE( CONJG(ts_d(ig,c)) * gs_d(ig,c) )
       ENDDO
       rho_d(c) = s
    ENDDO
    !
  END SUBROUTINE lowg_apply_sp
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vloc_g( vg_h )
    !-----------------------------------------------------------------------
    !! Fourier components of the local potential on the FFT grid, each spin:
    !! <G_i| V |G_j> = vg( position of G_i - G_j )
    COMPLEX(DP), INTENT(OUT) :: vg_h(nnr,nspl)
    INTEGER :: is, i, ip, istat, ikl, nrl
    nrl = nnr
    ip = pick_plan( 1 )
    DO is = 1, nspl
       ikl = is
       !$cuf kernel do(1)
       DO i = 1, nrl
          psic_d(i) = CMPLX( v_d(i,ikl), 0.0_DP, KIND=DP )
       ENDDO
       istat = cufftExecZ2Z( plan(ip), psic_d, psic_d, CUFFT_FORWARD )
       vg_h(:,is) = psic_d(1:nnr)
    ENDDO
    vg_h = vg_h / DBLE(nnr)
  END SUBROUTINE vloc_g
  !
  !-----------------------------------------------------------------------
  SUBROUTINE setup_lowg_real( )
    !-----------------------------------------------------------------------
    !! The preconditioner of setup_lowg for the real mode: the block of
    !! B = H + alpha_pv P_v (and of S) on the pairs (G, G') of lowest kinetic
    !! energy, in the real storage of the K-symmetric vectors (two numbers per
    !! pair, the matrix B~ with (B x)~ = B~ x~), diagonalized on the device.
    !! For a vector x = sum_j x_j |G_j> + conj(x_j) |G'_j>:
    !!   (B x)(G_i) = sum_j a_ij x_j + b_ij conj(x_j),
    !!   a_ij = B(G_i,G_j), b_ij = B(G_i,G'_j),
    !! local part from V(G_i - G_j) and V(G_i - G'_j), nonlocal and projector
    !! part 2 W~ D W~^T from the real rows of W. Size: as in setup_lowg, with
    !! the number of modes that the run computes at this q-point.
    !
    USE io_global,   ONLY : stdout
    USE klist,       ONLY : igk_k
    USE gvect,       ONLY : mill
    USE qpoint,      ONLY : ikks
    !
    INTEGER, PARAMETER :: nbmax = 16384
    REAL(DP), PARAMETER :: elow_def = 24.0_DP, cap_fac = 2.0_DP, rt2 = 1.4142135623730951_DP
    INTEGER :: ik, ikk, i, j, n, nlp, nb, is, nm, ncap, npc, lenv, ierr, istat, lwork, info, nbad, &
               pi, pj, mi(3), mj(3), mpj(3), mm(3), nbl, nwl, nkl, ikl, iw, na, iof, ih, jh, nhl
    CHARACTER(LEN=16) :: envval
    REAL(DP) :: elow, capf, emax, apv, v0, hh, ss
    COMPLEX(DP) :: a, b
    LOGICAL :: selfi, selfj
    INTEGER, ALLOCATABLE :: nlp_h(:), lp_h(:,:), ind(:)
    REAL(DP), ALLOCATABLE :: ek_h(:), gs(:), bm(:,:), w_h(:), le_h(:,:)
    COMPLEX(DP), ALLOCATABLE :: vg_h(:,:)
    INTEGER, DEVICE :: devinfo
    REAL(DP), ALLOCATABLE, DEVICE :: bm_d(:,:), sm_d(:,:), wl_d(:,:), y_d(:,:), yq_d(:,:), ww_d(:), work_d(:)
    !
    use_lgr = .FALSE.
    elow = elow_def
    CALL get_environment_variable( 'QE_PH_LOWG', envval, lenv, ierr )
    IF ( ierr == 0 .AND. lenv > 0 ) THEN
       READ( envval, *, IOSTAT=ierr ) elow
       IF ( ierr /= 0 ) elow = elow_def
    ENDIF
    IF ( elow <= 0.0_DP .OR. .NOT. use_real ) RETURN
    IF ( .NOT. lowg_cell_ok( ) ) RETURN
    capf = cap_fac
    CALL get_environment_variable( 'QE_PH_LOWG_CAP', envval, lenv, ierr )
    IF ( ierr == 0 .AND. lenv > 0 ) THEN
       READ( envval, *, IOSTAT=ierr ) capf
       IF ( ierr /= 0 .OR. capf <= 0.0_DP ) capf = cap_fac
    ENDIF
    CALL start_clock( 'dfptd_lowg' )
    nm = nmodes_hint
    IF ( nm <= 0 ) nm = 3 * natl
    ncap = NINT( ( capf * nm * nbo * REAL(nnr,DP) )**(1.0_DP/3.0_DP) )
    ! device: eigenvectors 4 nb^2 nks bytes; work matrices of one k-point 8 nb^2 * 3
    ncap = MIN( ncap, INT( SQRT( 0.05_DP * free_bytes( ) / ( 4.0_DP * nks ) ) ), &
                INT( SQRT( 0.2_DP * free_bytes( ) / 32.0_DP ) ) )
    ncap = MAX( 32, MIN( nbmax, ncap ) )
    npc = ncap / 2
    apv = alpha_pv_l ; nwl = nw ; nkl = nkbl ; nhl = nhx
    !
    ! ... pairs of the block
    !
    ALLOCATE( nlp_h(nks), ek_h(nks), gs(nhx), ind(nhx) )
    DO ik = 1, nks
       n = rnh_h(ik)
       ek_h(ik) = elow
       IF ( COUNT( rg2_h(1:n,ik) <= elow ) > npc ) THEN
          gs(1:n) = rg2_h(1:n,ik)
          ind(1) = 0
          CALL hpsort( n, gs, ind )
          ek_h(ik) = gs(npc)
       ENDIF
       nlp_h(ik) = COUNT( rg2_h(1:n,ik) <= ek_h(ik) )
    ENDDO
    DEALLOCATE( gs, ind )
    nlb = 2 * MAXVAL( nlp_h )
    IF ( nlb < 4 .OR. nlb > 2*nbmax ) THEN
       CALL stop_clock( 'dfptd_lowg' )
       RETURN
    ENDIF
    ALLOCATE( lp_h(nlb/2,nks) )
    lp_h = 1
    DO ik = 1, nks
       n = 0
       DO j = 1, rnh_h(ik)
          IF ( rg2_h(j,ik) <= ek_h(ik) ) THEN
             n = n + 1
             lp_h(n,ik) = j
          ENDIF
       ENDDO
    ENDDO
    ALLOCATE( vg_h(nnr,nspl) )
    CALL vloc_g( vg_h )
    !
    ALLOCATE( lgnr_d(nks), lgpr_d(nlb/2,nks), lgvr_d(nlb,nlb,nks), lger_d(nlb,nks), le_h(nlb,nks) )
    lgpr_d = lp_h
    lgvr_d = 0.0_SP
    le_h = 1.0E30_DP
    nbad = 0
    IF ( .NOT. blas_ready ) CALL init_blas( )
    IF ( .NOT. solv_ready ) THEN
       istat = cusolverDnCreate( hsolv )
       solv_ready = .TRUE.
    ENDIF
    DO ik = 1, nks
       ikk = ikks(ik)
       nlp = nlp_h(ik)
       nb = 2 * nlp
       is = spin_h(ik)
       IF ( nlp < 2 ) THEN
          nlp_h(ik) = 0
          CYCLE
       ENDIF
       !
       ! ... kinetic energy and local potential (host)
       !
       ALLOCATE( bm(nb,nb) )
       !$omp parallel do default(shared) private(j, i, pj, pi, mj, mpj, mi, mm, selfj, selfi, a, b)
       DO j = 1, nlp
          pj = lp_h(j,ik)
          mj(:) = mill(:,igk_k(rhl_h(pj,ik),ikk))
          mpj(:) = mill(:,igk_k(rhm_h(pj,ik),ikk))
          selfj = pj <= rns_h(ik)
          DO i = 1, nlp
             pi = lp_h(i,ik)
             mi(:) = mill(:,igk_k(rhl_h(pi,ik),ikk))
             selfi = pi <= rns_h(ik)
             mm(:) = mi(:) - mj(:)
             a = vg_h( 1 + MODULO(mm(1),nr1) + nr1*( MODULO(mm(2),nr2) + nr2*MODULO(mm(3),nr3) ), is )
             mm(:) = mi(:) - mpj(:)
             b = vg_h( 1 + MODULO(mm(1),nr1) + nr1*( MODULO(mm(2),nr2) + nr2*MODULO(mm(3),nr3) ), is )
             IF ( selfi .AND. selfj ) THEN
                bm(2*i-1,2*j-1) = DBLE(a) ; bm(2*i-1,2*j) = 0.0_DP
                bm(2*i,2*j-1) = 0.0_DP ; bm(2*i,2*j) = 0.0_DP
             ELSEIF ( selfj ) THEN
                bm(2*i-1,2*j-1) = rt2 * DBLE(a) ; bm(2*i-1,2*j) = 0.0_DP
                bm(2*i,2*j-1) = rt2 * AIMAG(a) ; bm(2*i,2*j) = 0.0_DP
             ELSEIF ( selfi ) THEN
                bm(2*i-1,2*j-1) = rt2 * DBLE(a) ; bm(2*i-1,2*j) = - rt2 * AIMAG(a)
                bm(2*i,2*j-1) = 0.0_DP ; bm(2*i,2*j) = 0.0_DP
             ELSE
                bm(2*i-1,2*j-1) = DBLE(a) + DBLE(b) ; bm(2*i-1,2*j) = - AIMAG(a) + AIMAG(b)
                bm(2*i,2*j-1) = AIMAG(a) + AIMAG(b) ; bm(2*i,2*j) = DBLE(a) - DBLE(b)
             ENDIF
          ENDDO
          bm(2*j-1,2*j-1) = bm(2*j-1,2*j-1) + rg2_h(pj,ik)
          IF ( selfj ) THEN
             bm(2*j,2*j) = 1.0E8_DP          ! the imaginary part of a self-conjugate component does not exist
          ELSE
             bm(2*j,2*j) = bm(2*j,2*j) + rg2_h(pj,ik)
          ENDIF
       ENDDO
       !$omp end parallel do
       !
       ! ... nonlocal potential and projector: 2 W~ D W~^T (device)
       !
       ALLOCATE( bm_d(nb,nb), wl_d(nb,nw), y_d(nb,nw), ww_d(nb) )
       bm_d = bm
       DEALLOCATE( bm )
       ikl = ik ; nbl = nb
       !$cuf kernel do(2)
       DO iw = 1, nwl
          DO i = 1, nbl
             wl_d(i,iw) = DBLE( wr_d(2*lgpr_d((i+1)/2,ikl)-MOD(i,2),iw,ikl) )
          ENDDO
       ENDDO
       !$cuf kernel do(2)
       DO iw = 1, nwl
          DO i = 1, nbl
             IF ( iw <= nkl ) THEN
                na = kbna_d(iw) ; iof = kbof_d(iw) ; ih = iw - iof
                hh = 0.0_DP
                DO jh = 1, kbnh_d(iw)
                   hh = hh + deeq_d(jh,ih,na,is) * wl_d(i,iof+jh)
                ENDDO
                y_d(i,iw) = hh
             ELSEIF ( iw - nkl <= noccq_d(ikl) ) THEN
                y_d(i,iw) = apv * wl_d(i,iw)
             ELSE
                y_d(i,iw) = 0.0_DP
             ENDIF
          ENDDO
       ENDDO
       istat = cublasDgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, nw, 2.0_DP, y_d, nb, wl_d, nb, &
                            1.0_DP, bm_d, nb )
       IF ( lus ) THEN
          ALLOCATE( sm_d(nb,nb), yq_d(nb,MAX(nkbl,1)) )
          !$cuf kernel do(2)
          DO j = 1, nbl
             DO i = 1, nbl
                IF ( i == j ) THEN
                   sm_d(i,j) = 1.0_DP
                ELSE
                   sm_d(i,j) = 0.0_DP
                ENDIF
             ENDDO
          ENDDO
          !$cuf kernel do(2)
          DO iw = 1, nkl
             DO i = 1, nbl
                na = kbna_d(iw) ; iof = kbof_d(iw) ; ih = iw - iof
                hh = 0.0_DP
                DO jh = 1, kbnh_d(iw)
                   hh = hh + qq_d(jh,ih,na) * wl_d(i,iof+jh)
                ENDDO
                yq_d(i,iw) = hh
             ENDDO
          ENDDO
          istat = cublasDgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, nkbl, 2.0_DP, yq_d, nb, wl_d, nb, &
                               1.0_DP, sm_d, nb )
          istat = cusolverDnDsygvd_bufferSize( hsolv, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, &
                                               CUBLAS_FILL_MODE_UPPER, nb, bm_d, nb, sm_d, nb, ww_d, lwork )
          ALLOCATE( work_d(lwork) )
          istat = cusolverDnDsygvd( hsolv, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, &
                                    CUBLAS_FILL_MODE_UPPER, nb, bm_d, nb, sm_d, nb, ww_d, work_d, lwork, devinfo )
          DEALLOCATE( sm_d, yq_d )
       ELSE
          istat = cusolverDnDsyevd_bufferSize( hsolv, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, &
                                               nb, bm_d, nb, ww_d, lwork )
          ALLOCATE( work_d(lwork) )
          istat = cusolverDnDsyevd( hsolv, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, nb, bm_d, nb, &
                                    ww_d, work_d, lwork, devinfo )
       ENDIF
       info = devinfo
       ALLOCATE( w_h(nb) )
       w_h = ww_d
       emax = MAXVAL( erot_h(1:nocc_h(ik),ik) )
       IF ( istat /= 0 .OR. info /= 0 .OR. w_h(1) - emax < 1.0E-3_DP ) THEN
          nbad = nbad + 1
          nlp_h(ik) = 0
       ELSE
          !$cuf kernel do(2)
          DO j = 1, nbl
             DO i = 1, nbl
                lgvr_d(i,j,ikl) = REAL( bm_d(i,j), SP )
             ENDDO
          ENDDO
          le_h(1:nb,ik) = w_h(1:nb)
       ENDIF
       DEALLOCATE( bm_d, wl_d, y_d, ww_d, work_d, w_h )
    ENDDO
    nlp_h(:) = 2 * nlp_h(:)
    lgnr_d = nlp_h
    lger_d = le_h
    !
    ! ... diagonal of B and of S for the other pairs
    !
    ALLOCATE( hdr_d(nhx,nks), sdr_d(nhx,nks) )
    DO is = 1, nspl
       v0 = DBLE( vg_h(1,is) )
       ikl = is
       !$acc parallel loop gang collapse(2) private(hh, ss, na, iof, ih)
       DO ik = 1, nks
          DO j = 1, nhl
             IF ( spin_d(ik) == ikl ) THEN
                hh = g2h_d(j,ik) + v0
                ss = 1.0_DP
                IF ( j <= nh_d(ik) ) THEN
                   !$acc loop seq
                   DO iw = 1, nkl
                      na = kbna_d(iw) ; iof = kbof_d(iw) ; ih = iw - iof
                      !$acc loop seq
                      DO jh = 1, kbnh_d(iw)
                         hh = hh + deeq_d(ih,jh,na,ikl) * ( DBLE(wr_d(2*j-1,iw,ik)) * wr_d(2*j-1,iof+jh,ik) + &
                                                           DBLE(wr_d(2*j,iw,ik)) * wr_d(2*j,iof+jh,ik) )
                         ss = ss + qq_d(ih,jh,na) * ( DBLE(wr_d(2*j-1,iw,ik)) * wr_d(2*j-1,iof+jh,ik) + &
                                                      DBLE(wr_d(2*j,iw,ik)) * wr_d(2*j,iof+jh,ik) )
                      ENDDO
                   ENDDO
                   !$acc loop seq
                   DO iw = nkl+1, nkl+noccq_d(ik)
                      hh = hh + apv * ( DBLE(wr_d(2*j-1,iw,ik))**2 + DBLE(wr_d(2*j,iw,ik))**2 )
                   ENDDO
                ENDIF
                hdr_d(j,ik) = hh
                sdr_d(j,ik) = ss
             ENDIF
          ENDDO
       ENDDO
    ENDDO
    use_lgr = .TRUE.
    WRITE( stdout, '(5x,"DFPT device engine: preconditioner with an exact block of ",i5, &
         &" plane waves (below ",f6.2," Ry)")' ) nlb, MINVAL( ek_h )
    IF ( nbad > 0 ) WRITE( stdout, '(5x,"DFPT device engine: ",i6," k-points without the block")' ) nbad
    CALL stop_clock( 'dfptd_lowg' )
    !
  END SUBROUTINE setup_lowg_real
  !
  !-----------------------------------------------------------------------
  SUBROUTINE lowg_apply_real( nar )
    !-----------------------------------------------------------------------
    !! low-G block of the preconditioned gradient of the real mode,
    !! tr_L = v (lambda - e)^-1 v^T gr_L, for the active columns of each
    !! k-point; then rho = <tr|gr>.
    INTEGER, INTENT(IN) :: nar
    INTEGER :: j, j1, j2, ik, nak, jj, i, ir, rc, c, n2, nbl, nrr, istat, nbk
    REAL(DP) :: s
    !
    n2 = ncol ; nbl = nlb ; nrr = 2*nhx
    j1 = 1
    DO WHILE ( j1 <= nar )
       ik = kof( actr_h(j1) )
       j2 = j1
       DO WHILE ( j2 < nar )
          IF ( kof( actr_h(j2+1) ) /= ik ) EXIT
          j2 = j2 + 1
       ENDDO
       nak = j2 - j1 + 1
       !$cuf kernel do(2)
       DO jj = 1, nak
          DO i = 1, nbl
             IF ( i <= lgnr_d(ik) ) THEN
                glr_d(i,jj) = gr_d(2*lgpr_d((i+1)/2,ik)-MOD(i,2),actr_d(j1-1+jj))
             ELSE
                glr_d(i,jj) = 0.0_SP
             ENDIF
          ENDDO
       ENDDO
       istat = cublasSgemm( hblas, CUBLAS_OP_T, CUBLAS_OP_N, nlb, nak, nlb, 1.0_SP, lgvr_d(1,1,ik), nlb, &
                            glr_d, nlb, 0.0_SP, clr_d, nlb )
       !$cuf kernel do(2)
       DO jj = 1, nak
          DO i = 1, nbl
             rc = actr_d(j1-1+jj)
             c = rc
             IF ( rc > n2 ) c = rc - n2
             clr_d(i,jj) = clr_d(i,jj) / REAL( MAX( 1.0E-3_DP, lger_d(i,ik) - er_d(c) ), SP )
          ENDDO
       ENDDO
       istat = cublasSgemm( hblas, CUBLAS_OP_N, CUBLAS_OP_N, nlb, nak, nlb, 1.0_SP, lgvr_d(1,1,ik), nlb, &
                            clr_d, nlb, 0.0_SP, glr_d, nlb )
       !$cuf kernel do(2)
       DO jj = 1, nak
          DO i = 1, nbl
             IF ( i <= lgnr_d(ik) ) tr_d(2*lgpr_d((i+1)/2,ik)-MOD(i,2),actr_d(j1-1+jj)) = glr_d(i,jj)
          ENDDO
       ENDDO
       j1 = j2 + 1
    ENDDO
    !$acc parallel loop gang private(rc, s)
    DO j = 1, nar
       rc = actr_d(j)
       s = 0.0_DP
       !$acc loop vector reduction(+:s)
       DO ir = 1, nrr
          s = s + DBLE(tr_d(ir,rc)) * DBLE(gr_d(ir,rc))
       ENDDO
       rhor_d(rc) = 2.0_DP * s
    ENDDO
    !
  END SUBROUTINE lowg_apply_real
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION lowg_cell_ok( )
    !-----------------------------------------------------------------------
    !! The block preconditioner leaves the residual that the convergence test
    !! allows on the plane waves above the block, where it turns into errors
    !! of the long-wavelength density response more than the residual of the
    !! stock preconditioner does. In small cells this has no effect on the
    !! self-consistency (same number of iterations down to tr2_ph = 1e-19 in
    !! all cases tried); in a 216-atom Si cell (shortest reciprocal-lattice
    !! vector squared 0.04 Ry) it costs 10-30% more iterations, in 54- and
    !! 63-atom cells (0.12, 0.22 Ry) none. The block is used when the
    !! shortest reciprocal-lattice vector (in-plane ones with the 2D cutoff)
    !! squared is above 0.1 Ry. QE_PH_LOWG_CELL=0 lifts the restriction.
    USE cell_base,   ONLY : bg, tpiba2
    USE Coul_cut_2D, ONLY : do_cutoff_2D
    REAL(DP) :: b2min
    INTEGER :: i, nb3, ln, ierr
    CHARACTER(LEN=8) :: val
    lowg_cell_ok = .TRUE.
    CALL get_environment_variable( 'QE_PH_LOWG_CELL', val, ln, ierr )
    IF ( ierr == 0 .AND. ln > 0 .AND. val(1:1) == '0' ) RETURN
    nb3 = 3
    IF ( do_cutoff_2D ) nb3 = 2
    b2min = HUGE( 1.0_DP )
    DO i = 1, nb3
       b2min = MIN( b2min, tpiba2 * ( bg(1,i)**2 + bg(2,i)**2 + bg(3,i)**2 ) )
    ENDDO
    lowg_cell_ok = b2min >= 0.1_DP
  END FUNCTION lowg_cell_ok
  !
  !-----------------------------------------------------------------------
  SUBROUTINE isort_small( n, a )
    !-----------------------------------------------------------------------
    !! ascending sort of an integer list (the pending columns are found in
    !! several passes, each one ascending)
    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(INOUT) :: a(n)
    INTEGER :: i, j, v, gap
    gap = n / 2
    DO WHILE ( gap > 0 )
       DO i = gap+1, n
          v = a(i)
          j = i
          DO WHILE ( j > gap )
             IF ( a(j-gap) <= v ) EXIT
             a(j) = a(j-gap)
             j = j - gap
          ENDDO
          a(j) = v
       ENDDO
       gap = gap / 2
    ENDDO
  END SUBROUTINE isort_small
  !
  !-----------------------------------------------------------------------
  SUBROUTINE add_zprec( nact )
    !-----------------------------------------------------------------------
    !! Spectral part of the preconditioner: t += sum_c psi_c (psi_c^H g) /
    !! (e_c - e_v) over the empty bands c kept at k+q, for the k-points
    !! with active columns; then rho = Re <t|g> of the active columns.
    !! The solution of the linear systems is not affected: only the
    !! preconditioner (and with it the norm of the convergence test, which
    !! becomes slightly stricter) changes.
    !
    INTEGER, INTENT(IN) :: nact
    INTEGER :: i, ik, k1, k2, c, jz, ig, j, nck, nzl, zo, n1
    REAL(DP) :: s, de
    !
    nck = ncolk ; nzl = nz ; zo = zoff ; n1 = npwxl
    i = 1
    DO WHILE ( i <= nact )
       k1 = (act_h(i)-1)/ncolk + 1
       k2 = k1
       DO WHILE ( i <= nact )
          ik = (act_h(i)-1)/ncolk + 1
          IF ( ik > k2+1 ) EXIT
          k2 = ik
          i = i + 1
       ENDDO
       IF ( lus ) THEN
          CALL gemm_k( CUBLAS_OP_C, nz, ncolk, npwxl, c_one, evq_d(1,zoff+1,k1), npwxl, &
                       INT(npwxl,8)*nbq, g_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, c_zero, &
                       cw_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, k2-k1+1 )
       ELSE
          CALL gemm_k( CUBLAS_OP_C, nz, ncolk, npwxl, c_one, w_d(1,nkbl+zoff+1,k1), npwxl, &
                       INT(npwxl,8)*nw, g_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, c_zero, &
                       cw_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, k2-k1+1 )
       ENDIF
       !$cuf kernel do(2)
       DO c = (k1-1)*nck+1, k2*nck
          DO jz = 1, nzl
             ik = (c-1)/nck + 1
             de = etq_d(zo+jz,ik) - e_d(c)
             IF ( zo+jz > noccq_d(ik) .AND. de > 1.0E-3_DP ) THEN
                cw_d(jz,c) = cw_d(jz,c) / de
             ELSE
                cw_d(jz,c) = (0.0_DP, 0.0_DP)
             ENDIF
          ENDDO
       ENDDO
       IF ( lus ) THEN
          CALL gemm_k( CUBLAS_OP_N, npwxl, ncolk, nz, c_one, evq_d(1,zoff+1,k1), npwxl, &
                       INT(npwxl,8)*nbq, cw_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, c_one, &
                       t_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, k2-k1+1 )
       ELSE
          CALL gemm_k( CUBLAS_OP_N, npwxl, ncolk, nz, c_one, w_d(1,nkbl+zoff+1,k1), npwxl, &
                       INT(npwxl,8)*nw, cw_d(1,(k1-1)*ncolk+1), nw, INT(nw,8)*ncolk, c_one, &
                       t_d(1,(k1-1)*ncolk+1), npwxl, INT(npwxl,8)*ncolk, k2-k1+1 )
       ENDIF
    ENDDO
    !
    !$acc parallel loop gang private(c, s)
    DO j = 1, nact
       c = act_d(j)
       s = 0.0_DP
       !$acc loop vector reduction(+:s)
       DO ig = 1, n1
          s = s + DBLE( CONJG(t_d(ig,c)) * g_d(ig,c) )
       ENDDO
       rho_d(c) = s
       rhoc_d(j) = s
    ENDDO
    !
  END SUBROUTINE add_zprec
  !
#else
  !
  ! ... stubs for builds without CUDA: the engine is never active
  !
  SUBROUTINE dfpt_dev_invalidate( )
  END SUBROUTINE dfpt_dev_invalidate
  !
  SUBROUTINE dfpt_dev_flush( )
  END SUBROUTINE dfpt_dev_flush
  !
  SUBROUTINE dfpt_dev_end( )
  END SUBROUTINE dfpt_dev_end
  !
  SUBROUTINE dfpt_dev_keep_dpsi( )
  END SUBROUTINE dfpt_dev_keep_dpsi
  !
#endif
  !
END MODULE dfpt_dev

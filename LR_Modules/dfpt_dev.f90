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

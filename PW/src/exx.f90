! Copyrigh(C) 2005-2018 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
#if defined(__USE_MANY_FFT)
#error USE_MANY_FFT not implemented in the GPU version.
#endif
!-----------------------------------------------------------------------------
MODULE exx
  !-----------------------------------------------------------------------------
  !! Variables and subroutines for calculation of exact-exchange contribution.  
  !! Implements ACE: Lin Lin, J. Chem. Theory Comput. 2016, 12, 2242.  
  !! Contains code for band parallelization over pairs of bands: see T. Barnes,
  !! T. Kurth, P. Carrier, N. Wichmann, D. Prendergast, P.R.C. Kent, J. Deslippe
  !! Computer Physics Communications 2017, doi.org/10.1016/j.cpc.2017.01.008.
  !
  USE kinds,                ONLY : DP
  USE noncollin_module,     ONLY : noncolin, npol
  USE io_global,            ONLY : stdout
  !
  USE control_flags,        ONLY : gamma_only, tqr, use_gpu, many_fft
  USE exx_base,             ONLY : exx_bgrp_type, EXX_BGRP_BANDS, dfftt, exxbuff , exxbuff_d, npwt, x_nbnd_occ, &
                                   ibnd_start, ibnd_end, gt, ggt, gcutmt, gkcut, gstart_t, ngmt_g, &
                                   eps_occ, exxalfa, x_occupation, x_occupation_d, &
                                   locbuff, exxmat, locmat, nbndproj, local_thr
  !
  IMPLICIT NONE
  !
  SAVE
  !
  !
  LOGICAL :: use_ace 
  !! if .TRUE. use Lin Lin's ACE method, if .FALSE. do not use ACE, 
  !! use old algorithm instead
  COMPLEX(DP), ALLOCATABLE :: xi(:,:,:)
  !! ACE projectors
  COMPLEX(DP), ALLOCATABLE :: xi_d(:,:)
#if defined(__CUDA)
  ATTRIBUTES(DEVICE) :: xi_d
#endif
  INTEGER :: xi_d_k = -1
  !! the k-point whose ACE projectors xi_d holds (-1: none): h_psi is applied many times to the
  !! same k-point, and xi is uploaded when the k-point changes, not at every application
  COMPLEX(DP), ALLOCATABLE :: xi_all_d(:,:,:)
  !! the ACE projectors of every k-point on the device, when they fit there comfortably: a
  !! change of k-point is then a copy on the device (ace_to_device)
#if defined(__CUDA)
  ATTRIBUTES(DEVICE) :: xi_all_d
#endif
  LOGICAL :: domat
  !! 
  ! ... energy related variables
  !
  REAL(DP) :: fock0 = 0.0_DP
  !! sum <old|Vx(old)|old>
  REAL(DP) :: fock1 = 0.0_DP
  !! sum <new|vx(old)|new>
  REAL(DP) :: fock2 = 0.0_DP
  !! sum <new|vx(new)|new>
  REAL(DP) :: fock3 = 0.0_DP
  !! sum <old|vx(new)|old>
  REAL(DP) :: dexx  = 0.0_DP
  !! fock1 - 0.5*(fock2+fock0)
  !
  ! ... custom fft grid and related G-vectors
  !
  LOGICAL :: exx_fft_initialized = .FALSE.
  REAL(DP)  :: ecutfock
  !! energy cutoff for custom grid
  !
 CONTAINS
#define _CX(A)  CMPLX(A,0._dp,kind=DP)
#define _CY(A)  CMPLX(0._dp,-A,kind=DP)
  !
  !------------------------------------------------------------------------
  SUBROUTINE exx_fft_create()
    !------------------------------------------------------------------------
    !! Initialise the custom grid that allows to put the wavefunction
    !! onto the new (smaller) grid for \rho=\psi_{k+q}\psi^*_k and vice versa.  
    !! Set up fft descriptors, including parallel stuff: sticks, planes, etc.
    !
    USE gvecw,                ONLY : ecutwfc
    USE gvect,                ONLY : ecutrho, ngm, g, gg, gstart, mill
    USE cell_base,            ONLY : at, bg, tpiba2
    USE recvec_subs,          ONLY : ggens
    USE fft_base,             ONLY : smap
    USE fft_types,            ONLY : fft_type_init
    USE symm_base,            ONLY : fft_fact
    USE mp_exx,               ONLY : negrp, intra_egrp_comm
    USE mp_bands,             ONLY : nproc_bgrp, intra_bgrp_comm, nyfft
    !
    USE klist,                ONLY : nks, xk
    USE mp_pools,             ONLY : inter_pool_comm
    USE mp,                   ONLY : mp_max, mp_sum
    !
    USE control_flags,        ONLY : tqr
    USE realus,               ONLY : qpointlist, tabxx, tabp
    USE command_line_options, ONLY : nmany_, pencil_decomposition_
    !
    USE exx_bp,                 ONLY : set_dfftt_grid_bp
    !
    IMPLICIT NONE
    !
    ! ... local variables
    !
    INTEGER :: ik, ngmt
    INTEGER, EXTERNAL :: n_plane_waves
    LOGICAL :: lpara
    !
    IF ( exx_fft_initialized ) RETURN
    !
    ! ... the cutoff of the Fock operator from the pair densities themselves, unless the input gives one
    CALL exx_auto_cutoff()
    !
    ! Initialise the custom grid that allows us to put the wavefunction
    ! onto the new (smaller) grid for \rho=\psi_{k+q}\psi^*_k and vice versa
    !
    ! gkcut is such that all |k+G|^2 < gkcut (in units of (2pi/a)^2)
    ! Note that with k-points, gkcut > ecutwfc/(2pi/a)^2
    ! gcutmt is such that |q+G|^2 < gcutmt
    !
    IF ( gamma_only ) THEN
       gkcut = ecutwfc / tpiba2
       gcutmt = ecutfock / tpiba2
    ELSE
       gkcut = 0.0_DP
       DO ik = 1, nks
          gkcut = MAX(gkcut, SQRT(SUM(xk(:,ik)**2)))
       ENDDO
       CALL mp_max( gkcut, inter_pool_comm )
       ! Alternatively, variable "qnorm" earlier computed in "exx_grid_init"
       ! could be used as follows:
       ! gkcut = ( SQRT(ecutwfc/tpiba2) + qnorm )**2
       gkcut = ( SQRT(ecutwfc/tpiba2) + gkcut )**2
       ! 
       ! ... the following instruction may be needed if ecutfock \simeq ecutwfc
       ! and guarantees that all k+G are included
       !
       gcutmt = MAX(ecutfock/tpiba2, gkcut)
    ENDIF
    !
    ! ... set up fft descriptors, including parallel stuff: sticks, planes, etc.
    !
    IF (negrp == 1 .or. (exx_bgrp_type .eq. EXX_BGRP_BANDS)) THEN
       !
       ! ... no band parallelization: exx grid is a subgrid of general grid
       !
       lpara = ( nproc_bgrp > 1 )
       IF ( use_gpu .AND. negrp == 1 .AND. exx_bgrp_type /= EXX_BGRP_BANDS .AND. .NOT. lpara .AND. .NOT. tqr ) &
          CALL exx_device_grid()
       CALL fft_type_init( dfftt, smap, "rho", gamma_only, lpara,         &
                           intra_bgrp_comm, at, bg, gcutmt, gcutmt/gkcut, &
                           fft_fact=fft_fact, nyfft=nyfft, nmany=nmany_,  &
                           use_pd=pencil_decomposition_ )
       CALL ggens( dfftt, gamma_only, at, g, gg, mill, gcutmt, ngmt, gt, ggt )
       gstart_t = gstart
       npwt = n_plane_waves(ecutwfc/tpiba2, nks, xk, gt, ngmt)
       ngmt_g = ngmt
       CALL mp_sum( ngmt_g, intra_bgrp_comm )
       !
    ELSE
       !
       ! initialize dfftt grid for massive EXX band parallelism
       Call set_dfftt_grid_bp( )
       !
    ENDIF
    ! define clock labels (this enables the corresponding fft too)
    dfftt%rho_clock_label = 'fftc' ; dfftt%wave_clock_label = 'fftcw' 
    !
    WRITE( stdout, '(/5x,"EXX grid: ",i8," G-vectors", 5x,       &
         &   "FFT dimensions: (",i4,",",i4,",",i4,")")') ngmt_g, &
         &   dfftt%nr1, dfftt%nr2, dfftt%nr3
    !
    exx_fft_initialized = .TRUE.
    !
    IF (tqr) THEN
       IF (ecutfock == ecutrho) THEN
          WRITE( stdout, '(5x,"Real-space augmentation: EXX grid -> DENSE grid")' )
          tabxx => tabp
       ELSE
          WRITE( stdout, '(5x,"Real-space augmentation: initializing EXX grid")' )
          CALL qpointlist( dfftt, tabxx )
       ENDIF
    ENDIF
    !
    RETURN
    !
  END SUBROUTINE exx_fft_create
  !
  !------------------------------------------------------------------------
  SUBROUTINE exx_auto_cutoff()
    !------------------------------------------------------------------------
    !! The default cutoff of the Fock operator, 4*ecutwfc, keeps every component of the pair
    !! densities psi_i psi_j. Most of that sphere holds nothing: the exchange energy is a sum
    !! over G of v(G) |rho_ij(G)|^2, and what lies beyond a cutoff E is exactly what a run with
    !! ecutfock = E leaves out (at fixed orbitals), while the cost of the operator goes with
    !! the number of grid points, E^(3/2).
    !! When ecutfock is not given in the input (norm-conserving, collinear runs), this routine
    !! measures that tail and sets ecutfock to the smallest of 1, 1.25, 1.5, 1.75, 2, 2.5, 3
    !! times ecutwfc beyond which less than a fraction tol of the sum is left. The sum is taken
    !! with the bare 1/G^2 weight (which bounds the screened kernels at large G) over a sample
    !! of pair densities at the first k-point of each pool: the pairs of 8 occupied orbitals,
    !! the highest included, and their pairs with up to 4 empty ones; the two kinds are held to
    !! the tolerance separately, the second being what the eigenvalues of empty bands see.
    !! tol is 1e-8, or QE_EXX_ECUT_TOL from the environment (0 keeps 4*ecutwfc).
    !! Called once, before the EXX grid is made.
    !
    USE gvecw,            ONLY : ecutwfc
    USE gvect,            ONLY : ecutrho, gg, gstart
    USE cell_base,        ONLY : tpiba2
    USE fft_base,         ONLY : dffts
    USE fft_interfaces,   ONLY : invfft, fwfft
    USE wavefunctions,    ONLY : evc
    USE wvfct,            ONLY : nbnd
    USE klist,            ONLY : nks, igk_k, ngk
    USE buffers,          ONLY : get_buffer
    USE io_files,         ONLY : nwordwfc, iunwfc
    USE uspp,             ONLY : okvan
    USE control_flags,    ONLY : iverbosity
    USE mp_bands,         ONLY : intra_bgrp_comm
    USE mp_pools,         ONLY : inter_pool_comm
    USE mp,               ONLY : mp_sum
    !
    IMPLICIT NONE
    !
    INTEGER, PARAMETER :: nl = 7
    REAL(DP), PARAMETER :: ladder(nl) = (/ 1.0_DP, 1.25_DP, 1.5_DP, 1.75_DP, 2.0_DP, 2.5_DP, 3.0_DP /)
    CHARACTER(LEN=32) :: val
    INTEGER :: lenv, stat, ios, ns, ne, nocc, is, js, ig, k, npw, ib, kc, kind, ngs, nnr
    REAL(DP) :: tol, e, wp, sh(nl+1,2), tot(2), tail(nl,2), edef
    INTEGER, ALLOCATABLE :: ibs(:)
    REAL(DP), ALLOCATABLE :: ws(:,:)
    COMPLEX(DP), ALLOCATABLE :: psir(:,:), rho(:)
    LOGICAL :: on_device
#if defined(__CUDA)
    COMPLEX(DP), ALLOCATABLE, DEVICE :: psir_d(:,:), rho_d(:), evc_d(:)
    REAL(DP), ALLOCATABLE, DEVICE :: ws_d(:,:), gg_d(:)
    INTEGER, ALLOCATABLE, DEVICE :: igk_d(:)
    INTEGER, POINTER, DEVICE :: nl_d(:), nlm_d(:)
    INTEGER :: ir
#endif
    !
    edef = MIN( ecutrho, 4.0_DP*ecutwfc )
    ! (ultrasoft and PAW: setup has fixed the cutoff; a cutoff from the input or a restart file is kept)
    IF ( okvan .OR. noncolin .OR. ABS( ecutfock - edef ) > 1.0E-8_DP .OR. x_nbnd_occ < 1 ) RETURN
    tol = 1.0E-8_DP
    CALL GET_ENVIRONMENT_VARIABLE( 'QE_EXX_ECUT_TOL', val, lenv, stat )
    IF ( stat == 0 .AND. lenv > 0 ) THEN
       ios = 0
       READ( val, *, IOSTAT=ios ) tol
       IF ( ios /= 0 .OR. tol < 0.0_DP ) THEN
          WRITE( stdout, '(5X,"EXX: QE_EXX_ECUT_TOL=",A," not understood (a relative tolerance, e.g. 1e-8; 0 keeps 4*ecutwfc): 1e-8")' ) TRIM(val)
          tol = 1.0E-8_DP
       ENDIF
    ENDIF
    IF ( tol == 0.0_DP ) RETURN
    CALL start_clock( 'exx_ecut' )
    !
    ! ... the sample: ns occupied orbitals of the first k-point, the highest included, then ne empty ones
    nocc = x_nbnd_occ
    ns = MIN( nocc, 8 )
    ne = MIN( nbnd - nocc, 4 )
    ALLOCATE( ibs(ns+ne) )
    DO is = 1, ns
       ibs(is) = nocc
       IF ( ns > 1 ) ibs(is) = 1 + NINT( DBLE( (is-1)*(nocc-1) ) / DBLE(ns-1) )
    ENDDO
    DO is = 1, ne
       ibs(ns+is) = nbnd
       IF ( ne > 1 ) ibs(ns+is) = nocc + 1 + NINT( DBLE( (is-1)*(nbnd-nocc-1) ) / DBLE(ne-1) )
    ENDDO
    IF ( nks > 1 ) CALL get_buffer( evc, nwordwfc, iunwfc, 1 )
    npw = ngk(1)
    ngs = dffts%ngm
    nnr = dffts%nnr
    ALLOCATE( ws(ngs,2) )
    !
    ! ... ws(G,kind): sum over the sampled pairs of |rho_ij(G)|^2 / G^2, the pairs standing for all of their
    !     kind: nocc diagonal ones and nocc (nocc-1) / 2 off the diagonal (kind 1), occupied x empty (kind 2)
    on_device = .FALSE.
#if defined(__CUDA)
    on_device = use_gpu .AND. ( dffts%nproc == 1 )
    IF ( on_device ) THEN
       ALLOCATE( psir_d(nnr,ns+ne), rho_d(nnr), evc_d(npw), ws_d(ngs,2), gg_d(ngs), igk_d(npw) )
       nl_d  => dffts%nl_d
       gg_d = gg(1:ngs)
       igk_d = igk_k(1:npw,1)
       ws_d = 0.0_DP
       psir_d = ( 0.0_DP, 0.0_DP )
       DO is = 1, ns + ne
          evc_d = evc(1:npw,ibs(is))
          !$cuf kernel do (1)
          DO ig = 1, npw
             psir_d( nl_d(igk_d(ig)), is ) = evc_d(ig)
          ENDDO
          IF ( gamma_only ) THEN
             nlm_d => dffts%nlm_d
             !$cuf kernel do (1)
             DO ig = 1, npw
                psir_d( nlm_d(igk_d(ig)), is ) = CONJG( evc_d(ig) )
             ENDDO
          ENDIF
          CALL invfft( 'Wave', psir_d(:,is), dffts )
       ENDDO
       DO js = 1, ns + ne
          DO is = 1, MIN( js, ns )
             kind = 1
             wp = DBLE(nocc) / DBLE(ns)
             IF ( is /= js ) wp = DBLE(nocc) * DBLE(nocc-1) / ( DBLE(ns) * DBLE(MAX(ns-1,1)) )
             IF ( js > ns ) THEN
                kind = 2
                wp = 1.0_DP
             ENDIF
             !$cuf kernel do (1)
             DO ir = 1, nnr
                rho_d(ir) = CONJG( psir_d(ir,is) ) * psir_d(ir,js)
             ENDDO
             CALL fwfft( 'Rho', rho_d, dffts )
             !$cuf kernel do (1)
             DO ig = gstart, ngs
                ws_d(ig,kind) = ws_d(ig,kind) + wp * ( DBLE( rho_d(nl_d(ig)) )**2 + AIMAG( rho_d(nl_d(ig)) )**2 ) / gg_d(ig)
             ENDDO
          ENDDO
       ENDDO
       ws = ws_d
       DEALLOCATE( psir_d, rho_d, evc_d, ws_d, gg_d, igk_d )
    ENDIF
#endif
    IF ( .NOT. on_device ) THEN
       ALLOCATE( psir(nnr,ns+ne), rho(nnr) )
       ws = 0.0_DP
       DO is = 1, ns + ne
          ib = ibs(is)
          psir(:,is) = ( 0.0_DP, 0.0_DP )
          DO ig = 1, npw
             psir( dffts%nl(igk_k(ig,1)), is ) = evc(ig,ib)
          ENDDO
          IF ( gamma_only ) THEN
             DO ig = 1, npw
                psir( dffts%nlm(igk_k(ig,1)), is ) = CONJG( evc(ig,ib) )
             ENDDO
          ENDIF
          CALL invfft( 'Wave', psir(:,is), dffts )
       ENDDO
       DO js = 1, ns + ne
          DO is = 1, MIN( js, ns )
             kind = 1
             wp = DBLE(nocc) / DBLE(ns)
             IF ( is /= js ) wp = DBLE(nocc) * DBLE(nocc-1) / ( DBLE(ns) * DBLE(MAX(ns-1,1)) )
             IF ( js > ns ) THEN
                kind = 2
                wp = 1.0_DP
             ENDIF
             rho(:) = CONJG( psir(:,is) ) * psir(:,js)
             CALL fwfft( 'Rho', rho, dffts )
             DO ig = gstart, ngs
                ws(ig,kind) = ws(ig,kind) + wp * ( DBLE( rho(dffts%nl(ig)) )**2 + AIMAG( rho(dffts%nl(ig)) )**2 ) / gg(ig)
             ENDDO
          ENDDO
       ENDDO
       DEALLOCATE( psir, rho )
    ENDIF
    !
    ! ... the shells between the cutoffs of the ladder, summed over the processes of the grid and over the pools
    sh = 0.0_DP
    DO ig = gstart, ngs
       e = gg(ig) * tpiba2
       IF ( e > edef ) CYCLE
       kc = nl + 1
       DO k = nl, 1, -1
          IF ( e <= ladder(k)*ecutwfc ) kc = k
       ENDDO
       sh(kc,:) = sh(kc,:) + ws(ig,:)
    ENDDO
    DEALLOCATE( ibs, ws )
    CALL mp_sum( sh, intra_bgrp_comm )
    CALL mp_sum( sh, inter_pool_comm )
    tot(:) = SUM( sh, 1 )
    tail = 0.0_DP
    DO k = 1, nl
       IF ( tot(1) > 0.0_DP ) tail(k,1) = SUM( sh(k+1:nl+1,1) ) / tot(1)
       IF ( tot(2) > 0.0_DP ) tail(k,2) = SUM( sh(k+1:nl+1,2) ) / tot(2)
    ENDDO
    kc = 0
    DO k = nl, 1, -1
       IF ( ladder(k)*ecutwfc < edef .AND. MAXVAL( tail(k,:) ) <= tol ) kc = k
    ENDDO
    IF ( iverbosity > 0 ) THEN
       WRITE( stdout, '(/5X,"EXX: part of the exchange sum beyond a cutoff of the Fock operator (occupied pairs, occupied x empty)")' )
       DO k = 1, nl
          IF ( ladder(k)*ecutwfc < edef ) WRITE( stdout, '(5X,"EXX:    ecutfock = ",F8.2," Ry:  ",2ES10.2)' ) ladder(k)*ecutwfc, tail(k,:)
       ENDDO
    ENDIF
    IF ( kc > 0 ) THEN
       ecutfock = ladder(kc)*ecutwfc
       WRITE( stdout, '(/5X,"EXX: cutoff of the Fock operator set to ",F8.2," Ry (4*ecutwfc = ",F8.2,"): sampled pair densities hold")' ) ecutfock, edef
       WRITE( stdout, '(5X,"     ",ES8.1," of the exchange sum beyond it (tolerance ",ES8.1,"; ecutfock in the input, or")' ) MAXVAL( tail(kc,:) ), tol
       WRITE( stdout, '(5X,"     QE_EXX_ECUT_TOL=0 in the environment, keeps the full sphere)")' )
    ENDIF
    CALL stop_clock( 'exx_ecut' )
    !
  END SUBROUTINE exx_auto_cutoff
  !
  !------------------------------------------------------------------------
  SUBROUTINE exx_device_grid()
    !------------------------------------------------------------------------
    !! The transforms of the EXX grid of a GPU run are cuFFT's, which is faster on a grid size
    !! 4*7*2^a*3^b*5^c (28, 56, 84, 112, 140, 168, ...) than on the next size with factors 2, 3 and 5
    !! only that good_fft_order would take: 84 against 90, 112 against 120, 168 against 180. Where
    !! such a size holds the transforms, set it before fft_type_init chooses.
    !! The exchange operator does not depend on the choice: the pair densities and their Fock
    !! potentials are band limited, so every grid of n >= nrho + 2*nwfc + 1 points per direction
    !! (nrho, nwfc the largest Miller indices of the spheres of the pair densities and of the
    !! wavefunctions) gives the same V_x|psi> on the wavefunction sphere, aliasing included.
    !! Nothing is set when no direction gains, when a direction would need more points than the
    !! standard grid has, or when the grid would not go with a symmetry operation.
    !
    USE gvect,        ONLY : ngm, gg, mill
    USE symm_base,    ONLY : nsym, s, ft, fft_fact
    USE mp_bands,     ONLY : intra_bgrp_comm
    USE mp,           ONLY : mp_max
    USE fft_support,  ONLY : good_fft_order
    !
    IMPLICIT NONE
    !
    INTEGER :: nrho(3), nwfc(3), nstd(3), nr(3), i, ig, n, m, isym, lenv, stat
    REAL(DP) :: ftn(3)
    LOGICAL :: good
    CHARACTER(LEN=8) :: val
    !
    IF ( dfftt%nr1 /= 0 .OR. dfftt%nr2 /= 0 .OR. dfftt%nr3 /= 0 ) RETURN
    CALL GET_ENVIRONMENT_VARIABLE( 'QE_EXX_GRID7', val, lenv, stat )
    IF ( stat == 0 .AND. lenv > 0 ) THEN
       IF ( val(1:1) == '0' ) RETURN
    ENDIF
    !
    nrho = 0
    nwfc = 0
    DO ig = 1, ngm
       IF ( gg(ig) > gcutmt ) EXIT
       nrho(:) = MAX( nrho(:), ABS( mill(:,ig) ) )
       IF ( gg(ig) <= gkcut ) nwfc(:) = MAX( nwfc(:), ABS( mill(:,ig) ) )
    ENDDO
    CALL mp_max( nrho, intra_bgrp_comm )
    CALL mp_max( nwfc, intra_bgrp_comm )
    !
    good = .FALSE.
    DO i = 1, 3
       nstd(i) = good_fft_order( 2*nrho(i) + 1, fft_fact(i) )
       nr(i) = nstd(i)
       ! the smallest size 28 * 2^a 3^b 5^c that holds the transforms and goes with fft_fact
       DO n = MAX( 2*nrho(i) + 1, nrho(i) + 2*nwfc(i) + 1 ), nstd(i) - 1
          IF ( MOD( n, 28 ) /= 0 .OR. MOD( n, 49 ) == 0 .OR. MOD( n, fft_fact(i) ) /= 0 ) CYCLE
          m = n / 28
          DO WHILE ( MOD( m, 2 ) == 0 )
             m = m / 2
          ENDDO
          DO WHILE ( MOD( m, 3 ) == 0 )
             m = m / 3
          ENDDO
          DO WHILE ( MOD( m, 5 ) == 0 )
             m = m / 5
          ENDDO
          IF ( m == 1 ) THEN
             nr(i) = n
             good = .TRUE.
             EXIT
          ENDIF
       ENDDO
       ! the standard size must hold the transforms too, or the choice is left to fft_type_init
       IF ( nstd(i) < nrho(i) + 2*nwfc(i) + 1 ) RETURN
    ENDDO
    IF ( .NOT. good ) RETURN
    !
    ! ... the rotations must send the grid into itself and the fractional translations must be
    !     grid vectors (the tests of scale_sym_ops, which exx_set_symm will apply to this grid)
    DO isym = 1, nsym
       IF ( MOD( s(2,1,isym)*nr(1), nr(2) ) /= 0 .OR. MOD( s(3,1,isym)*nr(1), nr(3) ) /= 0 .OR. &
            MOD( s(1,2,isym)*nr(2), nr(1) ) /= 0 .OR. MOD( s(3,2,isym)*nr(2), nr(3) ) /= 0 .OR. &
            MOD( s(1,3,isym)*nr(3), nr(1) ) /= 0 .OR. MOD( s(2,3,isym)*nr(3), nr(2) ) /= 0 ) RETURN
       ftn(:) = ft(:,isym) * nr(:)
       IF ( ANY( ABS( ftn(:) - NINT( ftn(:) ) ) / nr(:) > 1.0E-5_DP ) ) RETURN
    ENDDO
    !
    dfftt%nr1 = nr(1)
    dfftt%nr2 = nr(2)
    dfftt%nr3 = nr(3)
    WRITE( stdout, '(5X,"EXX-GPU: EXX grid ",I0," x ",I0," x ",I0," for cuFFT (standard ",I0," x ",I0," x ",I0,")")' ) &
         nr(1), nr(2), nr(3), nstd(1), nstd(2), nstd(3)
    !
  END SUBROUTINE exx_device_grid
  !
  !------------------------------------------------------------------------
  SUBROUTINE exx_gvec_reinit( at_old )
    !----------------------------------------------------------------------
    !! Re-initialize g-vectors after rescaling.
    !
    USE cell_base,  ONLY : bg
    !
    IMPLICIT NONE
    !
    REAL(DP), INTENT(IN) :: at_old(3,3)
    !! the lattice vectors at the previous step
    !
    ! ... local variables
    !
    INTEGER :: ig
    REAL(DP) :: gx, gy, gz
    !
    ! ... rescale g-vectors
    !
    CALL cryst_to_cart( dfftt%ngm, gt, at_old, -1 )
    CALL cryst_to_cart( dfftt%ngm, gt, bg,     +1 )
    !
    DO ig = 1, dfftt%ngm
       gx = gt(1,ig)
       gy = gt(2,ig)
       gz = gt(3,ig)
       ggt(ig) = gx*gx + gy*gy + gz*gz
    ENDDO
    !
  END SUBROUTINE exx_gvec_reinit
  !
  !
  !------------------------------------------------------------------------
  SUBROUTINE deallocate_exx()
    !------------------------------------------------------------------------
    !! Deallocates exx objects.
    !
    USE becmod,    ONLY : deallocate_bec_type, is_allocated_bec_type, bec_type
    USE us_exx,    ONLY : becxx
    USE exx_base,  ONLY : xkq_collect, index_xkq, index_xk, index_sym, rir, &
                          working_pool, exx_grid_initialized, evc0
    !
    IMPLICIT NONE
    !
    INTEGER :: ikq
    !
    exx_grid_initialized = .FALSE.
    !
    IF ( ALLOCATED(index_xkq) )    DEALLOCATE( index_xkq )
    IF ( ALLOCATED(index_xk ) )    DEALLOCATE( index_xk  )
    IF ( ALLOCATED(index_sym) )    DEALLOCATE( index_sym )
    IF ( ALLOCATED(rir) )          DEALLOCATE( rir )
    IF ( ALLOCATED(x_occupation) ) DEALLOCATE( x_occupation )
    IF ( ALLOCATED(x_occupation_d) ) DEALLOCATE( x_occupation_d )
    IF ( ALLOCATED(xkq_collect ) ) DEALLOCATE( xkq_collect  )
    IF ( ALLOCATED(exxbuff) )      DEALLOCATE( exxbuff )
    IF ( ALLOCATED(exxbuff_d) )    DEALLOCATE( exxbuff_d )
    IF ( ALLOCATED(locbuff) )      DEALLOCATE( locbuff )
    IF ( ALLOCATED(locmat) )       DEALLOCATE( locmat )
    IF ( ALLOCATED(exxmat) )       DEALLOCATE( exxmat )
    IF ( ALLOCATED(xi)   )         DEALLOCATE( xi   )
    IF ( ALLOCATED(xi_d) )         DEALLOCATE( xi_d )
    IF ( ALLOCATED(xi_all_d) )     DEALLOCATE( xi_all_d )
    xi_d_k = -1
    IF ( ALLOCATED(evc0) )         DEALLOCATE( evc0 )
    !
    IF ( ALLOCATED(becxx) ) THEN
      DO ikq = 1, SIZE(becxx)
        IF (is_allocated_bec_type(becxx(ikq))) CALL deallocate_bec_type( becxx(ikq) )
      ENDDO
      !
      DEALLOCATE( becxx )
    ENDIF
    !
    IF ( ALLOCATED(working_pool) )  DEALLOCATE( working_pool )
    !
    exx_fft_initialized = .FALSE.
    IF ( ASSOCIATED(gt)  )  DEALLOCATE( gt  )
    IF ( ASSOCIATED(ggt) )  DEALLOCATE( ggt )
    !
  END SUBROUTINE deallocate_exx
  !
  !------------------------------------------------------------------------
  SUBROUTINE exxinit( DoLoc, nbndproj_ )
    !------------------------------------------------------------------------
    !! This subroutine is run before the first H_psi() of each iteration. 
    !! It saves the wavefunctions for the right density matrix, in real space.
    !
    USE wavefunctions,        ONLY : evc, psic
    USE wvfct,                ONLY : nbnd, npwx, wg
    USE klist,                ONLY : nks, nkstot, wk
    USE symm_base,            ONLY : nsym, sr
    USE xc_lib,               ONLY : xclib_get_exx_fraction, start_exx,          &
                                     get_screening_parameter, get_gau_parameter, &
                                     exx_is_active
    USE uspp,                 ONLY : okvan
    USE paw_variables,        ONLY : okpaw
    USE exx_base,             ONLY : nkqs, index_sym, index_xk, xkq_collect, exx_set_symm, exxdiv, &
                                     erfc_scrlen, gau_scrlen, exx_divergence, d_spin
    USE exx_std,              ONLY : exxinit_std
    USE exx_bp,               ONLY : exxinit_bp
    USE us_exx,               ONLY : rotate_becxx
    USE paw_exx,              ONLY : PAW_init_fock_kernel
    !
    IMPLICIT NONE
    !
    LOGICAL, INTENT(IN) :: DoLoc
    !! TRUE:  Real Array locbuff(ir, nbnd, nkqs);  
    !! FALSE: Complex Array exxbuff(ir, nbnd/2, nkqs).
    INTEGER, OPTIONAL, INTENT(IN) :: nbndproj_
    ! if specified (non_scf) it sets nbndproj, else (scf case) nbndproj is automatically set to nbnd 
    !
    ! ... local variables
    !
    INTEGER :: ik, ibnd, ir, isym
    REAL(DP), ALLOCATABLE :: occ(:,:)
    !
    CALL start_clock ('exxinit')
    !
    IF (.NOT.exx_is_active()) THEN
       !
       erfc_scrlen = get_screening_parameter()
       gau_scrlen = get_gau_parameter()
       exxdiv  = exx_divergence()
       exxalfa = xclib_get_exx_fraction()
       !
       CALL start_exx()
    ENDIF
    !
    IF ( use_ace ) &
        WRITE(stdout,'(/,5X,"Using ACE for calculation of exact exchange")') 
    !
    ! set occupations of wavefunctions used in the calculation of exchange term
    IF (.NOT. ALLOCATED(x_occupation)) ALLOCATE( x_occupation(nbnd,nkstot) )
    IF( .NOT. ALLOCATED(x_occupation_d) .and. use_gpu) &
        ALLOCATE( x_occupation_d(nbnd,nkstot) )
    ALLOCATE( occ(nbnd,nks) )
    !
    DO ik = 1, nks
       IF (ABS(wk(ik)) > eps_occ) THEN
          occ(1:nbnd,ik) = wg(1:nbnd,ik) / wk(ik)
       ELSE
          occ(1:nbnd,ik) = 0._DP
       ENDIF
    ENDDO
    !
    CALL poolcollect( nbnd, nks, occ, nkstot, x_occupation )
    IF (use_gpu) x_occupation_d = x_occupation
    !
    DEALLOCATE( occ )
    !
    ! ... find an upper bound to the number of bands with non zero occupation.
    ! Useful to distribute bands among band groups
    !
    x_nbnd_occ = 0
    DO ik = 1, nkstot
       DO ibnd = MAX(1,x_nbnd_occ), nbnd
          IF (ABS(x_occupation(ibnd,ik)) > eps_occ) x_nbnd_occ = ibnd
       ENDDO
    ENDDO
    !
    IF ( use_ace ) &
        WRITE(stdout,'(/,5X,"Using ACE for calculation of exact exchange")') 
    !
    IF(use_ace) THEN 
      IF (present(nbndproj_)) THEN 
       nbndproj = nbndproj_
      ELSE
        IF (nbndproj == 0) nbndproj = nbnd
      END IF
      WRITE(stdout, '(5X,A,2(I5,A))') "ACE projected onto ", nbndproj, " (nbndproj) and applied to ", &
                                                                              nbnd, " (nbnd) bands"
    END IF 
    !
    ! ... prepare the symmetry matrices for the spin part
    !
    IF (noncolin) THEN
       DO isym = 1, nsym
          CALL find_u( sr(:,:,isym), d_spin(:,:,isym) )
       ENDDO
    ENDIF
    !
    IF ( Doloc ) THEN
        WRITE(stdout,'(/,5X,"Using localization algorithm with threshold: ",&
                & D10.2)') local_thr
        ! IF (.NOT.gamma_only) CALL errore('exxinit','SCDM with K-points NYI',1)
        IF (okvan .OR. okpaw) CALL errore( 'exxinit','SCDM with USPP/PAW not &
                                           &implemented', 1 )
    ENDIF 
    !
    CALL exx_fft_create()
    !
    IF (.NOT. gamma_only) CALL exx_set_symm( dfftt%nr1,  dfftt%nr2,  dfftt%nr3, &
                                             dfftt%nr1x, dfftt%nr2x, dfftt%nr3x )
    !
    if( exx_bgrp_type .eq. EXX_BGRP_BANDS ) then
      ! prepare buffers for standard EXX scheme
      Call exxinit_std( DoLoc )
    else 
      ! prepare buffers for massive EXX scheme (pair parallelism, DFT data transposition)
      Call exxinit_bp()
    end if
    !
    ! For US/PAW only: compute <beta_I|psi_j,k+q> for the entire 
    ! de-symmetrized k+q grid by rotating the ones from the irreducible wedge
    !
    IF (okvan) CALL rotate_becxx( nkqs, index_xk, index_sym, xkq_collect )
    !
    ! Initialize 4-wavefunctions one-center Fock integrals
    !    \int \psi_a(r)\phi_a(r)\phi_b(r')\psi_b(r')/|r-r'|
    !
    IF (okpaw) CALL PAW_init_fock_kernel()
    !
    CALL stop_clock( 'exxinit' )
    !
  END SUBROUTINE exxinit
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vexx( lda, n, m, psi, hpsi, becpsi )
    !-----------------------------------------------------------------------
    !! Wrapper routine computing V_x\psi, V_x = exchange potential. 
    !! Calls generic version vexx_k or Gamma-specific one vexx_gamma.
    !
    USE becmod,         ONLY : bec_type
    USE uspp,           ONLY : okvan
    USE paw_variables,  ONLY : okpaw
    USE wvfct,          ONLY : nbnd
    USE exx_std,        ONLY : vexx_std_gamma, vexx_std_k
    USE exx_bp,         ONLY : vexx_bp
    !
    IMPLICIT NONE
    !
    INTEGER :: lda
    !! input: leading dimension of arrays psi and hpsi
    INTEGER :: n
    !! input: true dimension of psi and hpsi
    INTEGER :: m
    !! input: number of states psi
    COMPLEX(DP) :: psi(lda*npol,m)
    !! input: m wavefunctions
    COMPLEX(DP) :: hpsi(lda*npol,m)
    !! output: V_x*psi
    TYPE(bec_type), OPTIONAL :: becpsi
    !! input: <beta|psi>, optional but needed for US and PAW case
    !
    IF ((okvan.OR.okpaw) .AND. .NOT. PRESENT(becpsi)) &
       CALL errore( 'vexx','becpsi needed for US/PAW case', 1 )
    !
    CALL start_clock( 'vexx' )
    !
    if(exx_bgrp_type .eq. EXX_BGRP_BANDS) then
       if (gamma_only ) then
          Call vexx_std_gamma(lda, n, m, psi, hpsi, becpsi )
       else
          Call vexx_std_k(lda, n, m, psi, hpsi, becpsi )
       end if 
    else
       Call vexx_bp(lda, n, m, psi, hpsi, becpsi )
    end if 
    !
    CALL stop_clock( 'vexx' )
    !
  END SUBROUTINE vexx
  !
  !-----------------------------------------------------------------------
  FUNCTION exxenergy()
    !-----------------------------------------------------------------------
    !! NB: This function is meant to give the SAME RESULT as exxenergy2.
    !! It is worth keeping it in the repository because in spite of being
    !! slower it is a simple driver using vexx potential routine so it is
    !! good, from time to time, to replace exxenergy2 with it to check that
    !! everything is ok and energy and potential are consistent as they should.
    !
    USE io_files,               ONLY : iunwfc, iunwfc_exx, nwordwfc
    USE buffers,                ONLY : get_buffer
    USE wvfct,                  ONLY : nbnd, npwx, wg, current_k
    USE gvect,                  ONLY : gstart
    USE wavefunctions,          ONLY : evc
    USE lsda_mod,               ONLY : lsda, current_spin, isk
    USE klist,                  ONLY : ngk, nks, xk, igk_k
    USE mp_pools,               ONLY : inter_pool_comm
    USE mp_exx,                 ONLY : intra_egrp_comm, intra_egrp_comm, &
                                       negrp
    USE mp,                     ONLY : mp_sum
    USE becmod,                 ONLY : bec_type, allocate_bec_type, &
                                       deallocate_bec_type, calbec
    USE uspp,                   ONLY : okvan,nkb,vkb
    USE exx_bp_utils,           ONLY : nwordwfc_exx, igk_exx
    USE uspp_init,              ONLY : init_us_2
    USE mp_bands,               ONLY : intra_bgrp_comm
    IMPLICIT NONE
    !
    TYPE(bec_type) :: becpsi
    REAL(DP) :: exxenergy,  energy
    INTEGER :: npw, ibnd, ik
    COMPLEX(DP) :: vxpsi(npwx*npol,nbnd), psi(npwx*npol,nbnd)
    !
    exxenergy = 0._DP
    !
    IF (okvan) CALL allocate_bec_type( nkb, nbnd, becpsi )
    energy = 0._dp
    !
    DO ik = 1, nks
       npw = ngk(ik)
       ! setup variables for usage by vexx (same logic as for H_psi)
       current_k = ik
       IF ( lsda ) current_spin = isk(ik)
       ! end setup
       IF ( nks > 1 ) THEN
          IF ( exx_bgrp_type .eq. EXX_BGRP_BANDS ) THEN
             CALL get_buffer( psi, nwordwfc, iunwfc, ik )
          ELSE
             CALL get_buffer( psi, nwordwfc_exx, iunwfc_exx, ik )
          END IF
       ELSE
          psi(1:npwx*npol,1:nbnd) = evc(1:npwx*npol,1:nbnd)
       ENDIF
       !
       IF (okvan) THEN
          ! prepare the |beta> function at k+q
          IF ( exx_bgrp_type .eq. EXX_BGRP_BANDS ) THEN
             CALL init_us_2( npw, igk_k(1,ik), xk(:,ik), vkb )
          ELSE
             CALL init_us_2( npw, igk_exx(1,ik), xk(:,ik), vkb )
          END IF
          ! compute <beta_I|psi_j> at this k+q point, for all band and all projectors
          CALL calbec( npw, vkb, psi, becpsi, nbnd )
       ENDIF
       !
       vxpsi(:,:) = (0._dp, 0._dp)
       CALL vexx( npwx, npw, nbnd, psi, vxpsi, becpsi )
       !
       DO ibnd = 1, nbnd
          energy = energy + DBLE(wg(ibnd,ik) * dot_product(psi(1:npw,ibnd),vxpsi(1:npw,ibnd)))
          IF (noncolin) energy = energy + &
                  DBLE(wg(ibnd,ik) * dot_product(psi(npwx+1:npwx+npw,ibnd),vxpsi(npwx+1:npwx+npw,ibnd)))
          !
       ENDDO
       IF (gamma_only .AND. gstart == 2) THEN
           DO ibnd = 1, nbnd
              energy = energy - &
                       DBLE(0.5_dp * wg(ibnd,ik) * CONJG(psi(1,ibnd)) * vxpsi(1,ibnd))
           ENDDO
       ENDIF
    ENDDO
    !
    IF (gamma_only) energy = 2 * energy
    !
    IF ( exx_bgrp_type .eq. EXX_BGRP_BANDS ) THEN
       CALL mp_sum( energy, intra_bgrp_comm )
    ELSE
       CALL mp_sum( energy, intra_egrp_comm )
    END IF
    CALL mp_sum( energy, inter_pool_comm )
    IF (okvan)  CALL deallocate_bec_type( becpsi )
    !
    exxenergy = energy
    !
  END FUNCTION exxenergy
  !
  !
  !-----------------------------------------------------------------------
  FUNCTION exxenergy2()
    !-----------------------------------------------------------------------
    !! Wrapper to \(\texttt{exxenergy2_gamma}\) and \(\texttt{exxenergy2_k}\).
    !
    USE exx_bp,   ONLY : exxenergy_bp_gamma, exxenergy_bp_k 
    USE exx_base, ONLY : exxbuff_sync_host
    !
    IMPLICIT NONE
    !
    REAL(DP) :: exxenergy2
    !
    CALL start_clock( 'exxenergy' )
    CALL exxbuff_sync_host()
    !
    IF ( exx_bgrp_type .eq. EXX_BGRP_BANDS ) THEN
       !
       exxenergy2 = exxenergy()
       !
    ELSE
       !
       IF ( gamma_only ) THEN
          exxenergy2 = exxenergy_bp_gamma()
       ELSE
          exxenergy2 = exxenergy_bp_k()
       ENDIF
       !
    END IF
    !
    CALL stop_clock( 'exxenergy' )
    !
  END FUNCTION  exxenergy2
  !
  !-----------------------------------------------------------------------
  FUNCTION exx_stress()
    !-----------------------------------------------------------------------
    !! This is Eq.(10) of PRB 73, 125120 (2006).
    !
    USE exx_std,  ONLY : exx_stress_std
    USE exx_bp,   ONLY : exx_stress_bp
#if defined(__CUDA)
    USE exx_bp,   ONLY : exx_stress_fast_path, exx_stress_gpu
#endif
    USE exx_base, ONLY : exxbuff_sync_host
    !
    IMPLICIT NONE
    !
    REAL(DP)   :: exx_stress(3,3)
    LOGICAL    :: on_device
    !
    CALL start_clock( 'exx_stress' )
    !
    on_device = .FALSE.
    IF ( exx_bgrp_type .eq. EXX_BGRP_BANDS ) THEN 
       CALL exxbuff_sync_host()
       exx_stress = exx_stress_std() 
    ELSE
#if defined(__CUDA)
       ! the batched device routine when it applies (k-points, norm-conserving, collinear, one band group)
       IF ( use_gpu .AND. exx_stress_fast_path() ) CALL exx_stress_gpu( exx_stress, on_device )
#endif
       IF ( .NOT. on_device ) THEN
          CALL exxbuff_sync_host()
          exx_stress = exx_stress_bp() 
       ENDIF
    END IF
    !
    CALL stop_clock( 'exx_stress' )
    !
  END FUNCTION exx_stress
  !
  !
  !-----------------------------------------------------------------------------
  SUBROUTINE aceinit( DoLoc, exex )
    !----------------------------------------------------------------------------
    !! ACE Initialization
    !
    USE wvfct,              ONLY : nbnd, npwx, current_k
    USE klist,              ONLY : nks, xk, ngk, igk_k
    USE uspp,               ONLY : nkb, vkb, okvan
    USE becmod,             ONLY : allocate_bec_type, deallocate_bec_type, &
                                   bec_type, calbec
    USE lsda_mod,           ONLY : current_spin, lsda, isk
    USE io_files,           ONLY : nwordwfc, iunwfc
    USE buffers,            ONLY : get_buffer
    USE mp_pools,           ONLY : inter_pool_comm
    USE mp,                 ONLY : mp_sum
    USE wavefunctions,      ONLY : evc
    USE uspp_init,          ONLY : init_us_2
#if defined(__CUDA)
    USE exx_bp,             ONLY : vexx_all_path, vexx_bp_k_all_gpu
#endif
    !
    IMPLICIT NONE
    !
    LOGICAL, INTENT(IN) :: DoLoc
    !! if TRUE calculates exact exchange with SCDM orbitals
    REAL(DP), OPTIONAL, INTENT(OUT) :: exex
    !! ACE energy
    !
    ! ... local variables
    !
    REAL(DP) :: ee, eexx
    INTEGER :: ik, npw
    TYPE(bec_type) :: becpsi
    LOGICAL :: all_k
    !
    IF (nbndproj < x_nbnd_occ .OR. nbndproj > nbnd) THEN 
       WRITE( stdout, '(3(A,I4))' ) ' occ = ', x_nbnd_occ, ' proj = ', nbndproj, &
                                    ' tot = ', nbnd
       CALL errore( 'aceinit', 'n_proj must be between occ and tot.', 1 )
    ENDIF
    !
    IF (.NOT. ALLOCATED(xi)) ALLOCATE( xi(npwx*npol,nbndproj,nks) )
#if defined (__CUDA)
    IF (.NOT. ALLOCATED(xi_d)) ALLOCATE( xi_d(npwx*npol,nbndproj) )
#endif
    IF ( okvan ) CALL allocate_bec_type( nkb, nbnd, becpsi )
    !
    eexx = 0.0d0
    xi = (0.0d0,0.0d0)
    !
    ! ... where the batched device operator can, V_x|psi> of every k-point is built in one pass
    !     (each pair potential then serves the two k-points of its pair), into xi
    all_k = .FALSE.
#if defined(__CUDA)
    IF ( .NOT. gamma_only .AND. .NOT. DoLoc .AND. exx_bgrp_type /= EXX_BGRP_BANDS ) THEN
       IF ( vexx_all_path( nbndproj ) ) THEN
          CALL start_clock( 'aceinit' )
          CALL start_clock( 'vexx' )
          CALL vexx_bp_k_all_gpu( npwx, nbndproj, xi, all_k )
          CALL stop_clock( 'vexx' )
          CALL stop_clock( 'aceinit' )
          IF ( .NOT. all_k ) xi = (0.0d0,0.0d0)
       ENDIF
    ENDIF
#endif
    !
    DO ik = 1, nks
       npw = ngk(ik)
       current_k = ik
       IF ( lsda ) current_spin = isk(ik)
       IF ( nks > 1 ) CALL get_buffer( evc, nwordwfc, iunwfc, ik )
       IF ( okvan ) THEN
          CALL init_us_2( npw, igk_k(1,ik), xk(:,ik), vkb )
          CALL calbec( npw, vkb, evc, becpsi, nbnd )
       ENDIF
       IF (gamma_only) THEN
          CALL aceinit_gamma( DoLoc, npw, nbnd, evc, xi(1,1,ik), becpsi, ee )
       ELSE
          CALL aceinit_k( DoLoc, npw, nbnd, evc, xi(1,1,ik), becpsi, ee, all_k )
       ENDIF
       eexx = eexx + ee
    ENDDO
    !
    CALL mp_sum( eexx, inter_pool_comm )
    ! WRITE(stdout,'(/,5X,"ACE energy",f15.8)') eexx
    !
    CALL ace_to_device()
    !
    IF (PRESENT(exex)) exex = eexx
    IF ( okvan ) CALL deallocate_bec_type( becpsi )
    !
    domat = .FALSE.
    !
  END SUBROUTINE aceinit
  !
  !-----------------------------------------------------------------------------
  SUBROUTINE ace_to_device()
    !---------------------------------------------------------------------------
    !! The device copies of the ACE projectors after xi has been (re)written on the host: the
    !! one k-point of a single-k run, or every k-point when that takes little of the device
    !! memory, so that vexxace need not upload xi whenever the k-point changes.
    !
    USE wvfct,      ONLY : npwx
    USE klist,      ONLY : nks
#if defined(__CUDA)
    USE cudafor,    ONLY : cudaMemGetInfo
#endif
    !
    IMPLICIT NONE
    !
#if defined(__CUDA)
    INTEGER :: istat
    INTEGER(KIND=8) :: need, free_, total_
    LOGICAL :: fits
    !
    xi_d_k = -1      ! xi has changed: the device copy of any k-point is stale
    IF ( .NOT. use_gpu .OR. .NOT. ALLOCATED(xi_d) ) RETURN
    IF ( nks == 1 ) THEN
       xi_d(:,:) = xi(:,:,1)
       xi_d_k = 1
       RETURN
    ENDIF
    need = 16_8 * INT(SIZE(xi,1),8) * INT(SIZE(xi,2),8) * INT(nks,8)
    IF ( ALLOCATED(xi_all_d) ) THEN
       IF ( SIZE(xi_all_d,1) /= SIZE(xi,1) .OR. SIZE(xi_all_d,2) /= SIZE(xi,2) .OR. SIZE(xi_all_d,3) /= nks ) &
          DEALLOCATE( xi_all_d )
    ENDIF
    fits = ALLOCATED(xi_all_d)
    IF ( .NOT. fits ) THEN
       istat = cudaMemGetInfo( free_, total_ )
       fits = ( istat == 0 .AND. need <= 2147483648_8 .AND. 8_8*need < free_ )
       IF ( fits ) ALLOCATE( xi_all_d(SIZE(xi,1),SIZE(xi,2),nks) )
    ENDIF
    IF ( fits ) xi_all_d(:,:,:) = xi(:,:,1:nks)
#else
    xi_d_k = -1
#endif
    !
  END SUBROUTINE ace_to_device
  !
  !
  !---------------------------------------------------------------------------------
  SUBROUTINE aceinit_gamma( DoLoc, nnpw, nbnd, phi, xitmp, becpsi, exxe )
    !-------------------------------------------------------------------------------
    !! Compute xi(npw,nbndproj) for the ACE method.
    !
    USE becmod,         ONLY : bec_type
    USE lsda_mod,       ONLY : current_spin
    USE mp,             ONLY : mp_stop
    USE exx_base,       ONLY : evc0
    !
    IMPLICIT NONE
    !
    LOGICAL, INTENT(IN) :: DoLoc
    !! if TRUE calculates exact exchange with SCDM orbitals
    INTEGER :: nnpw
    !! number of pw
    INTEGER :: nbnd
    !! number of bands
    COMPLEX(DP) :: phi(nnpw,nbnd)
    !! wavefunction
    COMPLEX(DP) :: xitmp(nnpw,nbndproj)
    !! xi(npw,nbndproj)
    TYPE(bec_type), INTENT(IN) :: becpsi
    !! <beta|psi>
    REAL(DP) :: exxe
    !! exx energy
    !
    ! ... local variables
    !
    INTEGER :: nrxxs
    REAL(DP), ALLOCATABLE :: mexx(:,:)
    REAL(DP), PARAMETER :: Zero=0._DP
    LOGICAL :: domat0  
    !
    CALL start_clock( 'aceinit' )  
    !
    nrxxs = dfftt%nnr * npol  
    !
    ALLOCATE( mexx(nbndproj,nbndproj) )  
    xitmp = (Zero,Zero)  
    mexx = Zero  
    !  
    IF ( DoLoc ) then    
      CALL vexx_loc( nnpw, nbndproj, xitmp, mexx )
      CALL MatSymm( 'S', 'L', mexx,nbndproj )
    ELSE  
      ! |xi> = Vx[phi]|phi>
      CALL vexx( nnpw, nnpw, nbndproj, phi, xitmp, becpsi )
      ! mexx = <phi|Vx[phi]|phi>
      IF ( .NOT. use_gpu ) &
      CALL matcalc( 'exact', .TRUE., 0, nnpw, nbndproj, nbndproj, phi, xitmp, mexx, exxe )
      ! |xi> = -One * Vx[phi]|phi> * rmexx^T
    ENDIF  
    !
    IF ( use_gpu .AND. .NOT. DoLoc ) THEN
       ! the two steps above and below with the device BLAS
       CALL aceupdate_gpu( nnpw, nnpw, phi, xitmp, exxe, .TRUE. )
    ELSE
       CALL aceupdate( nbndproj, nnpw, xitmp, mexx )
    ENDIF
    !
    DEALLOCATE( mexx )  
    !
    IF ( local_thr > 0.0d0 ) THEN
      domat0 = domat
      domat = .TRUE.  
      CALL vexxace_gamma( nnpw, nbndproj, evc0(1,1,current_spin), exxe )  
      evc0(:,:,current_spin) = phi(:,:)  
      domat = domat0  
    ENDIF
    !
    CALL stop_clock( 'aceinit' )  
    !
  END SUBROUTINE aceinit_gamma
  !
  !
  !----------------------------------------------------------------------------------
  SUBROUTINE vexxace_gamma( nnpw, nbnd, phi, exxe, vphi )
    !-------------------------------------------------------------------------------
    !! Do the ACE potential and (optional) print the ACE matrix representation.
    !
    USE wvfct,        ONLY : current_k, wg
    USE lsda_mod,     ONLY : current_spin
    !
    IMPLICIT NONE
    !
    INTEGER :: nnpw
    !! number of plane waves
    INTEGER :: nbnd
    !! number of bands
    COMPLEX(DP) :: phi(nnpw,nbnd) 
    !! wave function
    REAL(DP) :: exxe
    !! exx energy
    COMPLEX(DP), OPTIONAL :: vphi(nnpw,nbnd)
    !! v times phi
    !
    ! ... local variables
    !
    INTEGER :: i, ik
    REAL(DP), ALLOCATABLE :: rmexx(:,:)
    COMPLEX(DP),ALLOCATABLE :: cmexx(:,:), vv(:,:)  
    REAL(DP), PARAMETER :: Zero=0._DP, One=1._DP
    !
    CALL start_clock( 'vexxace' )
    !
    ALLOCATE( vv(nnpw,nbnd) )
    !
    IF (PRESENT(vphi)) THEN  
      vv = vphi  
    ELSE  
      vv = (Zero,Zero)  
    ENDIF  
    !
    ! do the ACE potential
    ALLOCATE( rmexx(nbndproj,nbnd), cmexx(nbndproj,nbnd) )
    !
    rmexx = Zero  
    cmexx = (Zero,Zero)  
    ! <xi|phi>
    CALL matcalc( '<xi|phi>', .FALSE. , 0, nnpw, nbndproj, nbnd, xi(1,1,current_k), &
                  phi, rmexx, exxe )
    ! |vv> = |vphi> + (-One) * |xi> * <xi|phi>
    cmexx = (One,Zero)*rmexx
    !
    CALL ZGEMM( 'N', 'N', nnpw, nbnd, nbndproj, -(One,Zero), xi(1,1,current_k), &
                nnpw, cmexx, nbndproj, (One,Zero), vv, nnpw )
    !
    DEALLOCATE( cmexx, rmexx )  
    !
    IF (domat) THEN
      ALLOCATE( rmexx(nbnd,nbnd) )
      CALL matcalc( 'ACE', .TRUE., 0, nnpw, nbnd, nbnd, phi, vv, rmexx, exxe )
      DEALLOCATE( rmexx )
#if defined(__DEBUG)
        WRITE(stdout,'(4(A,I3),A,I9,A,f12.6)') 'vexxace: nbnd=', nbnd, ' nbndproj=',nbndproj, &
                                               ' k=',current_k,' spin=',current_spin,' npw=', &
                                               nnpw, ' E=',exxe
      ELSE
        WRITE(stdout,'(4(A,I3),A,I9)')         'vexxace: nbnd=', nbnd, ' nbndproj=',nbndproj, &
                                               ' k=',current_k,' spin=',current_spin,' npw=', &
                                               nnpw
#endif
      ENDIF
      !
      IF (PRESENT(vphi)) vphi = vv
      DEALLOCATE( vv )
      !
      CALL stop_clock( 'vexxace' )
      !
  END SUBROUTINE vexxace_gamma
  !
  !
  !----------------------------------------------------------------------------------
  SUBROUTINE vexxace_gamma_gpu( nnpw, nbnd, phi_d, exxe, vphi_d )
    !-------------------------------------------------------------------------------
    !! Do the ACE potential and (optional) print the ACE matrix representation.
    !
    USE klist,        ONLY : nks
    USE wvfct,        ONLY : current_k, wg
    USE lsda_mod,     ONLY : current_spin
#if defined(__CUDA)
    USE cublas
#endif
    !
    IMPLICIT NONE
    !
    INTEGER :: nnpw
    !! number of plane waves
    INTEGER :: nbnd
    !! number of bands
    COMPLEX(DP) :: phi_d(nnpw,nbnd)
    !! wave function
    REAL(DP) :: exxe
    !! exx energy
    COMPLEX(DP), OPTIONAL :: vphi_d(nnpw,nbnd)
    !! v times phi
#if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: phi_d, vphi_d
#endif
    !
    ! ... local variables
    !
    INTEGER :: i, j
    REAL(DP), ALLOCATABLE :: rmexx_d(:,:)
    COMPLEX(DP),ALLOCATABLE :: cmexx_d(:,:), vv_d(:,:)
#if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: rmexx_d, cmexx_d, vv_d
#endif
    REAL(DP), PARAMETER :: Zero=0._DP, One=1._DP
    !
    CALL start_clock( 'vexxace' )
    !
    IF ( .NOT. PRESENT(vphi_d) ) THEN
      ALLOCATE( vv_d(nnpw,nbnd) )
      vv_d = (Zero,Zero)
    ENDIF
    !
    ! do the ACE potential
    ALLOCATE( rmexx_d(nbndproj,nbnd), cmexx_d(nbndproj,nbnd) )
    !
    IF ( nks > 1 .AND. xi_d_k /= current_k ) THEN
       IF ( ALLOCATED(xi_all_d) ) THEN
          xi_d(:,:) = xi_all_d(:,:,current_k)
       ELSE
          xi_d(:,:) = xi(:,:,current_k)
       ENDIF
       xi_d_k = current_k
    ENDIF
    !
    ! <xi|phi>
    CALL matcalc_gpu( '<xi|phi>', .FALSE. , 0, nnpw, nbndproj, nbnd, xi_d, phi_d, rmexx_d, exxe )
    !
    !$cuf kernel do(2)
    DO j = 1, nbnd
       DO i = 1, nbndproj
          cmexx_d(i,j) = CMPLX(rmexx_d(i,j), KIND=DP)
       ENDDO
    ENDDO
    !
    ! |vv> = |vphi> + (-One) * |xi> * <xi|phi>
    IF ( .NOT. PRESENT(vphi_d) ) THEN
       CALL ZGEMM( 'N', 'N', nnpw, nbnd, nbndproj, -(One,Zero), xi_d, &
                   nnpw, cmexx_d, nbndproj, (One,Zero), vv_d, nnpw )
    ELSE
       CALL ZGEMM( 'N', 'N', nnpw, nbnd, nbndproj, -(One,Zero), xi_d, &
                   nnpw, cmexx_d, nbndproj, (One,Zero), vphi_d, nnpw )
    ENDIF
    !
    DEALLOCATE( cmexx_d )
    !
    IF ( domat ) THEN
       !
       IF ( nbndproj /= nbnd ) THEN
          DEALLOCATE( rmexx_d )
          ALLOCATE( rmexx_d(nbnd,nbnd) )
       ENDIF
       !
       IF ( .NOT. PRESENT(vphi_d) ) THEN
          CALL matcalc_gpu( 'ACE', .TRUE., 0, nnpw, nbnd, nbnd, phi_d, vv_d, rmexx_d, exxe )
       ELSE
          CALL matcalc_gpu( 'ACE', .TRUE., 0, nnpw, nbnd, nbnd, phi_d, vphi_d, rmexx_d, exxe )
       ENDIF
       !
    ENDIF
    !
    DEALLOCATE( rmexx_d )
    IF( .NOT. PRESENT(vphi_d) ) DEALLOCATE( vv_d )
    !
    CALL stop_clock( 'vexxace' )
    !
  END SUBROUTINE vexxace_gamma_gpu
  !
  !-----------------------------------------------------------------------------------------
  SUBROUTINE aceupdate_gpu( ld, nnpw, phi, xitmp, exxe, update )
    !---------------------------------------------------------------------------------------
    !! The matrix <phi|Vx[phi]|phi> with its energy (matcalc, matcalc_k) and the ACE projectors
    !! |xi> = Vx[phi]|phi> L^-H, with -<phi|Vx|phi> = L L^H (aceupdate, aceupdate_k), by the device
    !! BLAS: the two products over plane waves are the cost of an ACE build after V_x itself,
    !! and the host BLAS of a GPU build need not be an optimised one. The factorisation and
    !! inversion of the nbndproj x nbndproj matrix stay on the host.
    !
    USE wvfct,    ONLY : current_k
#if defined(__CUDA)
    USE cublas
#endif
    !
    IMPLICIT NONE
    !
    INTEGER, INTENT(IN) :: ld
    !! leading dimension of phi and xitmp (npwx*npol; the number of PW at Gamma)
    INTEGER, INTENT(IN) :: nnpw
    !! number of PW
    COMPLEX(DP), INTENT(IN) :: phi(ld,nbndproj)
    !! wave functions
    COMPLEX(DP), INTENT(INOUT) :: xitmp(ld,nbndproj)
    !! Vx[phi]|phi> on input, xi on output
    REAL(DP), INTENT(OUT) :: exxe
    !! exx energy
    LOGICAL, INTENT(IN) :: update
    !! if FALSE only the energy is computed (k-points of zero weight)
    !
    ! ... local variables
    !
    COMPLEX(DP), ALLOCATABLE :: phi_d(:,:), x_d(:,:), c_d(:,:), cmexx(:,:)
    REAL(DP), ALLOCATABLE :: r_d(:,:), rmexx(:,:)
#if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: phi_d, x_d, c_d, r_d
#endif
    REAL(DP), PARAMETER :: Zero=0._DP, One=1._DP
    !
#if defined(__CUDA)
    ALLOCATE( phi_d(ld,nbndproj), x_d(ld,nbndproj), c_d(nbndproj,nbndproj), cmexx(nbndproj,nbndproj) )
    phi_d = phi(:,1:nbndproj)
    x_d = xitmp
    IF ( gamma_only ) THEN
       ALLOCATE( r_d(nbndproj,nbndproj), rmexx(nbndproj,nbndproj) )
       CALL matcalc_gpu( 'exact', .TRUE., 0, ld, nbndproj, nbndproj, phi_d, x_d, r_d, exxe )
       rmexx = r_d
       IF ( update ) THEN
          CALL start_clock( 'aceupdate' )
          ! rmexx = -(Cholesky(rmexx))^-1
          rmexx = -rmexx
          CALL MatChol( nbndproj, rmexx )
          CALL MatInv( 'L', nbndproj, rmexx )
          cmexx = (One,Zero)*rmexx
       ENDIF
       DEALLOCATE( r_d, rmexx )
    ELSE
       CALL matcalc_k_gpu( 'exact', .TRUE., 0, current_k, ld, nbndproj, nbndproj, phi_d, x_d, c_d, exxe )
       cmexx = c_d
       IF ( update ) THEN
          CALL start_clock( 'aceupdate' )
          ! mexx = -(Cholesky(mexx))^-1
          cmexx = -cmexx
          CALL invchol_k( nbndproj, cmexx )
       ENDIF
    ENDIF
    IF ( update ) THEN
       ! |xi> = -One * Vx[phi]|phi> * mexx^T
       c_d = cmexx
       CALL ZTRMM( 'R', 'L', 'C', 'N', ld, nbndproj, (One,Zero), c_d, nbndproj, x_d, ld )
       xitmp = x_d
       CALL stop_clock( 'aceupdate' )
    ENDIF
    DEALLOCATE( phi_d, x_d, c_d, cmexx )
#else
    CALL errore( 'aceupdate_gpu', 'called in a build without CUDA', 1 )
#endif
    !
  END SUBROUTINE aceupdate_gpu
  !
  !
  !-------------------------------------------------------------------------------------------
  SUBROUTINE aceupdate( nbndproj, nnpw, xitmp, rmexx )
    !----------------------------------------------------------------------------------------
    !! Build the ACE operator from the potential amd matrix (rmexx is assumed symmetric
    !! and only the Lower Triangular part is considered).
    !
    IMPLICIT NONE
    !
    INTEGER :: nbndproj
    !! number of bands
    INTEGER :: nnpw
    !! number of PW
    COMPLEX(DP) :: xitmp(nnpw,nbndproj)
    !! xi(nnpw,nbndproj)
    REAL(DP) :: rmexx(nbndproj,nbndproj)
    !! |xi> = -One * Vx[phi]|phi> * rmexx^T
    !
    ! ... local variables
    !
    COMPLEX(DP), ALLOCATABLE :: cmexx(:,:)
    REAL(DP), PARAMETER :: Zero=0._DP, One=1._DP
    !
    CALL start_clock( 'aceupdate' )
    !
    ! rmexx = -(Cholesky(rmexx))^-1
    rmexx = -rmexx
    ! CALL invchol( nbndproj, rmexx )
    CALL MatChol( nbndproj, rmexx )
    CALL MatInv( 'L', nbndproj, rmexx )
    !
    ! |xi> = -One * Vx[phi]|phi> * rmexx^T
    ALLOCATE( cmexx(nbndproj,nbndproj) )
    cmexx = (One,Zero)*rmexx
    CALL ZTRMM( 'R', 'L', 'C', 'N', nnpw, nbndproj, (One,Zero), cmexx, nbndproj, xitmp, nnpw )
    !
    DEALLOCATE( cmexx )
    !
    CALL stop_clock( 'aceupdate' )
    !
  END SUBROUTINE
  !
  !
  !---------------------------------------------------------------------------------------------
  SUBROUTINE aceinit_k( DoLoc, nnpw, nbnd, phi, xitmp, becpsi, exxe, have_vx )
    !-----------------------------------------------------------------------------------------
    !! Compute xi(npw,nbndproj) for the ACE method.
    !
    USE becmod,               ONLY : bec_type
    USE wvfct,                ONLY : current_k, npwx
    USE klist,                ONLY : wk
    USE noncollin_module,     ONLY : npol
    USE exx_base,             ONLY : evc0
    !
    IMPLICIT NONE
    !
    LOGICAL, INTENT(IN) :: DoLoc
    !! if TRUE calculates exact exchange with SCDM orbitals
    INTEGER :: nnpw
    !! number of PW
    INTEGER :: nbnd
    !! number of bands
    COMPLEX(DP) :: phi(npwx*npol,nbnd)
    !! wave function
    COMPLEX(DP) :: xitmp(npwx*npol,nbndproj)
    !! xi(nnpw,nbndproj)
    TYPE(bec_type), INTENT(IN) :: becpsi
    !! <beta|psi>
    REAL(DP) :: exxe
    !! exx energy
    LOGICAL, INTENT(IN) :: have_vx
    !! if TRUE xitmp already holds Vx[phi]|phi> on input
    !
    ! ... local variables
    !
    COMPLEX(DP), ALLOCATABLE :: mexx(:,:)
    REAL(DP) :: exxe0
    REAL(DP), PARAMETER :: Zero=0._DP
    INTEGER :: i
    LOGICAL :: domat0
    !
    CALL start_clock( 'aceinit' )
    !
    IF (nbndproj>nbnd) CALL errore( 'aceinit_k', 'nbndproj greater than nbnd.', 1 )
    IF (nbndproj<=0)   CALL errore( 'aceinit_k', 'nbndproj le 0.', 1 )
    !
    ALLOCATE( mexx(nbndproj,nbndproj) )
    IF ( .NOT. have_vx ) xitmp = (Zero,Zero)
    mexx  = (Zero,Zero)
    IF ( DoLoc ) THEN
      CALL vexx_loc_k( nnpw, nbndproj, xitmp, mexx, exxe )
      CALL MatSymm_k( 'S', 'L', mexx, nbndproj )
    ELSE
      ! |xi> = Vx[phi]|phi>
      IF ( .NOT. have_vx ) CALL vexx( npwx, nnpw, nbndproj, phi, xitmp, becpsi )
      ! mexx = <phi|Vx[phi]|phi>
      IF ( .NOT. use_gpu ) &
      CALL matcalc_k( 'exact', .TRUE., 0, current_k, npwx*npol, nbndproj, nbndproj, &
                      phi, xitmp, mexx, exxe )
    ENDIF
#if defined(__DEBUG)
      WRITE( stdout,'(3(A,I3),A,I9,A,f12.6)') 'aceinit_k: nbnd=', nbnd, ' nbndproj=',nbndproj, &
                                              ' k=',current_k,' npw=',nnpw,' Ex(k)=',exxe
#endif
    IF ( use_gpu .AND. .NOT. DoLoc ) THEN
      ! the matrix and the update with the device BLAS (the update skipped at zero weight, as below)
      CALL aceupdate_gpu( npwx*npol, nnpw, phi, xitmp, exxe, wk(current_k)/=0._dp )
    ELSE IF(wk(current_k)/=0._dp)THEN
    ! Skip k-points that have exactly zero weight
      ! |xi> = -One * Vx[phi]|phi> * rmexx^T
      CALL aceupdate_k( nbndproj, nnpw, xitmp, mexx )
    ENDIF
    !
    DEALLOCATE( mexx )
    !
    IF ( DoLoc ) THEN
       domat0 = domat
       domat = .TRUE.
       CALL vexxace_k( nnpw, nbnd, evc0(1,1,current_k), exxe )
       evc0(:,:,current_k) = phi(:,:)
       domat = domat0
    ENDIF 
    !
    CALL stop_clock( 'aceinit' )
    !
  END SUBROUTINE aceinit_k
  !
  !
  !------------------------------------------------------------------------------
  SUBROUTINE aceupdate_k( nbndproj, nnpw, xitmp, mexx )
    !----------------------------------------------------------------------------
    !! Updates xi(npw,nbndproj) for the ACE method.
    !
    USE wvfct,                ONLY : npwx
    USE noncollin_module,     ONLY : noncolin, npol
    !
    IMPLICIT NONE
    !
    INTEGER :: nbndproj
    !! number of bands
    INTEGER :: nnpw
    !! number of PW
    COMPLEX(DP) :: mexx(nbndproj,nbndproj)
    !! mexx = -(Cholesky(mexx))^-1
    COMPLEX(DP) :: xitmp(npwx*npol,nbndproj)
    !! |xi> = -One * Vx[phi]|phi> * mexx^T
    !
    CALL start_clock( 'aceupdate' )
    !
    ! mexx = -(Cholesky(mexx))^-1
    mexx = -mexx
    CALL invchol_k( nbndproj, mexx )
    !
    ! |xi> = -One * Vx[phi]|phi> * mexx^T
    CALL ZTRMM( 'R', 'L', 'C', 'N', npwx*npol, nbndproj, (1.0_dp,0.0_dp), mexx,nbndproj, &
                xitmp, npwx*npol )
    !
    CALL stop_clock( 'aceupdate' )
    !
  END SUBROUTINE aceupdate_k
  !
  !
  !--------------------------------------------------------------------------------------
  SUBROUTINE vexxace_k( nnpw, nbnd, phi, exxe, vphi )
    !-----------------------------------------------------------------------------------
    !! Do the ACE potential and (optional) print the ACE matrix representation.
    !
    USE becmod,               ONLY : calbec
    USE wvfct,                ONLY : current_k, npwx
    USE noncollin_module,     ONLY : npol
    !
    IMPLICIT NONE
    !
    REAL(DP) :: exxe
    !! exx energy
    INTEGER :: nnpw
    !! number of PW
    INTEGER :: nbnd
    !! number of bands
    COMPLEX(DP) :: phi(npwx*npol,nbnd)
    !! wave function
    COMPLEX(DP), OPTIONAL :: vphi(npwx*npol,nbnd)
    !! ACE potential
    !
    ! ... local variables
    !
    COMPLEX(DP), ALLOCATABLE :: cmexx(:,:), vv(:,:)
    REAL(DP), PARAMETER :: Zero=0._DP, One=1._DP
    !
    CALL start_clock( 'vexxace' )
    !
    ALLOCATE( vv(npwx*npol,nbnd) )  
    IF (PRESENT(vphi)) THEN  
      vv = vphi  
    ELSE  
      vv = (Zero, Zero)
    ENDIF  
    !
    ! do the ACE potential! 
    ALLOCATE( cmexx(nbndproj,nbnd) )  
    cmexx = (Zero,Zero)  
    ! <xi|phi>
    CALL matcalc_k( '<xi|phi>', .FALSE., 0, current_k, npwx*npol, nbndproj, nbnd, &
                    xi(1,1,current_k), phi, cmexx, exxe )
    !
    ! |vv> = |vphi> + (-One) * |xi> * <xi|phi>! 
    CALL ZGEMM( 'N', 'N', npwx*npol, nbnd, nbndproj, -(One,Zero), xi(1,1,current_k), &
                npwx*npol, cmexx, nbndproj, (One,Zero), vv, npwx*npol )
    !
    IF (domat) THEN
       !
       IF ( nbndproj /= nbnd) THEN
          DEALLOCATE( cmexx )
          ALLOCATE( cmexx(nbnd,nbnd) )
       ENDIF
       !
       CALL matcalc_k( 'ACE', .TRUE., 0, current_k, npwx*npol, nbnd, nbnd, phi, vv, cmexx, exxe )
       !
#if defined(__DEBUG)
       WRITE(stdout,'(3(A,I3),A,I9,A,f12.6)') 'vexxace_k: nbnd=', nbnd, ' nbndproj=',nbndproj, &
                    ' k=',current_k,' npw=',nnpw, ' Ex(k)=',exxe
    ELSE
       WRITE(stdout,'(3(A,I3),A,I9)') 'vexxace_k: nbnd=', nbnd, ' nbndproj=',nbndproj, &
                       ' k=',current_k,' npw=',nnpw
#endif
    ENDIF
    !  
    IF (PRESENT(vphi)) vphi = vv
    DEALLOCATE( vv, cmexx )
    !
    CALL stop_clock( 'vexxace' )
    !
  END SUBROUTINE vexxace_k
  !
  !
  !--------------------------------------------------------------------------------------
  SUBROUTINE vexxace_k_gpu( nnpw, nbnd, phi_d, exxe, vphi_d )
    !-----------------------------------------------------------------------------------
    !! Do the ACE potential and (optional) print the ACE matrix representation.
    !
    USE becmod,               ONLY : calbec
    USE klist,                ONLY : nks
    USE wvfct,                ONLY : current_k, npwx
    USE noncollin_module,     ONLY : npol
#if defined(__CUDA)
    USE cublas
#endif
    !
    IMPLICIT NONE
    !
    REAL(DP) :: exxe
    !! exx energy
    INTEGER :: nnpw
    !! number of PW
    INTEGER :: nbnd
    !! number of bands
    COMPLEX(DP) :: phi_d(npwx*npol,nbnd)
    !! wave function
    COMPLEX(DP), OPTIONAL :: vphi_d(npwx*npol,nbnd)
    !! ACE potential
#if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: phi_d, vphi_d
#endif
    !
    ! ... local variables
    !
    COMPLEX(DP), ALLOCATABLE :: cmexx_d(:,:), vv_d(:,:)
#if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: cmexx_d, vv_d
#endif
    REAL(DP), PARAMETER :: Zero=0._DP, One=1._DP
    !
    CALL start_clock( 'vexxace' )
    !
    IF ( .NOT. PRESENT(vphi_d) ) THEN
      ALLOCATE( vv_d(npwx*npol,nbnd) )
      vv_d = (Zero,Zero)
    ENDIF
    !
    ! do the ACE potential!
    ALLOCATE( cmexx_d(nbndproj,nbnd) )
    !
    IF ( nks > 1 .AND. xi_d_k /= current_k ) THEN
       IF ( ALLOCATED(xi_all_d) ) THEN
          xi_d(:,:) = xi_all_d(:,:,current_k)
       ELSE
          xi_d(:,:) = xi(:,:,current_k)
       ENDIF
       xi_d_k = current_k
    ENDIF
    !
    ! <xi|phi>
    CALL matcalc_k_gpu( '<xi|phi>', .FALSE., 0, current_k, npwx*npol, nbndproj, nbnd, &
                        xi_d, phi_d, cmexx_d, exxe )
    !
    ! |vv> = |vphi> + (-One) * |xi> * <xi|phi>!
    IF ( .NOT. PRESENT(vphi_d) ) THEN
       CALL ZGEMM( 'N', 'N', npwx*npol, nbnd, nbndproj, -(One,Zero), xi_d, &
                   npwx*npol, cmexx_d, nbndproj, (One,Zero), vv_d, npwx*npol )
    ELSE
       CALL ZGEMM( 'N', 'N', npwx*npol, nbnd, nbndproj, -(One,Zero), xi_d, &
                   npwx*npol, cmexx_d, nbndproj, (One,Zero), vphi_d, npwx*npol )
    ENDIF
    !
    IF ( domat ) THEN
       !
       IF ( nbndproj /= nbnd ) THEN
          DEALLOCATE( cmexx_d )
          ALLOCATE( cmexx_d(nbnd,nbnd) )
       ENDIF
       !
       IF ( .NOT. PRESENT(vphi_d) ) THEN
          CALL matcalc_k_gpu( 'ACE', .TRUE., 0, current_k, npwx*npol, nbnd, nbnd, phi_d, &
                              vv_d, cmexx_d, exxe )
       ELSE
          CALL matcalc_k_gpu( 'ACE', .TRUE., 0, current_k, npwx*npol, nbnd, nbnd, phi_d, &
                              vphi_d, cmexx_d, exxe )
       ENDIF
       !
    ENDIF
    !
    DEALLOCATE( cmexx_d )
    IF( .NOT. PRESENT(vphi_d) ) DEALLOCATE( vv_d )
    !
    CALL stop_clock( 'vexxace' )
    !
  END SUBROUTINE vexxace_k_gpu
  !
  !----------------------------------------------------------------------------
  FUNCTION exxenergyace( )
    !--------------------------------------------------------------------------
    !! Compute exchange energy using ACE
    !
    USE kinds,              ONLY : DP
    USE buffers,            ONLY : get_buffer
    USE klist,              ONLY : nks, ngk
    USE wvfct,              ONLY : nbnd, npwx, current_k
    USE lsda_mod,           ONLY : lsda, isk, current_spin
    USE io_files,           ONLY : iunwfc, nwordwfc
    USE mp_pools,           ONLY : inter_pool_comm
    USE mp_bands,           ONLY : intra_bgrp_comm
    USE mp,                 ONLY : mp_sum
    USE control_flags,      ONLY : gamma_only, use_gpu
    USE wavefunctions,      ONLY : evc
    !
    IMPLICIT NONE
    !
    REAL(DP) :: exxenergyace
    !! computed energy
    !
    ! ... local variables
    !
    REAL(DP) :: ex
    INTEGER :: ik, npw
    !
    domat = .TRUE.
    exxenergyace=0.0_dp
    !
    DO ik = 1, nks
       npw = ngk (ik)
       !
       current_k = ik
       IF ( lsda ) current_spin = isk(ik)
       !
       IF (nks > 1) THEN
          CALL get_buffer( evc, nwordwfc, iunwfc, ik )
          !$acc update device(evc)
       ENDIF
       !
       IF (gamma_only) THEN
          IF (use_gpu) THEN
            !$acc host_data use_device(evc)
            CALL vexxace_gamma_gpu( npw, nbnd, evc, ex )
            !$acc end host_data
          ELSE
            CALL vexxace_gamma( npw, nbnd, evc, ex )
          END IF
       ELSE
          IF (use_gpu) THEN
            !$acc host_data use_device(evc)
            CALL vexxace_k_gpu( npw, nbnd, evc, ex )
            !$acc end host_data
          ELSE
            CALL vexxace_k( npw, nbnd, evc, ex )
          ENDIF
       ENDIF
       exxenergyace = exxenergyace + ex
    ENDDO
    !
    CALL mp_sum( exxenergyace, inter_pool_comm )
    !
    domat = .FALSE.
    !
  END FUNCTION exxenergyace
  !
  !---------------------------------------------------------------------------------
  SUBROUTINE vexx_loc( npw, nbnd, hpsi, mexx )
    !---------------------------------------------------------------------------------
    !! Exact exchange with SCDM orbitals.  
    !! Vx|phi> =  Vx|psi> <psi|Vx|psi>^(-1) <psi|Vx|phi>.  
    !! locmat contains localization integrals.
    !
    USE noncollin_module,  ONLY : npol
    USE cell_base,         ONLY : omega, alat
    USE wvfct,             ONLY : current_k
    USE klist,             ONLY : xk, nks, nkstot
    USE fft_interfaces,    ONLY : fwfft, invfft
    USE mp,                ONLY : mp_stop, mp_barrier, mp_sum
    USE exx_base,          ONLY : exxbuff_sync_host
    USE exx_base,          ONLY : nqs, xkq_collect, index_xkq, index_xk, &
                                  g2_convolution
    !
    IMPLICIT NONE
    !
    INTEGER :: npw
    !! number of PW
    INTEGER :: nbnd
    !! number of bands
    COMPLEX(DP) :: hpsi(npw,nbnd)
    !! hpsi
    REAL(DP) :: mexx(nbnd,nbnd)
    !! mexx contains in output the exchange matrix
    !
    ! ... local variables
    !
    INTEGER :: nrxxs, npairs, ntot, NBands   
    INTEGER :: ig, ir, ik, ikq, iq, ibnd, jbnd, kbnd, NQR  
    INTEGER :: current_ik  
    REAL(DP) :: exxe  
    COMPLEX(DP), ALLOCATABLE :: rhoc(:), vc(:), RESULT(:,:)   
    REAL(DP), ALLOCATABLE :: fac(:)  
    REAL(DP) :: xkp(3), xkq(3)  
    INTEGER, EXTERNAL  :: global_kpoint_index  
    !
    WRITE( stdout, '(5X,A)' ) ' '   
    WRITE( stdout, '(5X,A)' ) 'Exact-exchange with localized orbitals'  
    !
    CALL start_clock( 'vexxloc' )
    CALL exxbuff_sync_host()
    !
    WRITE( stdout,'(7X,A,f24.12)' ) 'local_thr =', local_thr  
    nrxxs = dfftt%nnr  
    NQR = nrxxs*npol  
    !
    ! ... exchange projected onto localized orbitals 
    WRITE( stdout,'(A)' ) 'Allocating exx quantities...'
    ALLOCATE( fac(dfftt%ngm) )
    ALLOCATE( rhoc(nrxxs), vc(NQR) )
    ALLOCATE( RESULT(nrxxs,nbnd) ) 
    WRITE( stdout,'(A)' ) 'Allocations done.'
    !
    current_ik = global_kpoint_index( nkstot, current_k )
    xkp = xk(:,current_k)
    !
    vc = (0.0d0, 0.0d0)
    npairs = 0 
    !
    DO iq = 1, nqs
       ikq  = index_xkq(current_ik,iq)  
       ik   = index_xk(ikq)  
       xkq  = xkq_collect(:,ikq)  
       !  
       CALL g2_convolution( dfftt%ngm, gt, xkp, xkq, fac )  
       !  
       RESULT = (0.0d0, 0.0d0)  
       !  
       DO ibnd = 1, nbnd  
         !
         IF (x_occupation(ibnd,ikq) > 0.0d0) THEN
           !
           DO ir = 1, NQR   
             rhoc(ir) = locbuff(ir,ibnd,ikq) * locbuff(ir,ibnd,ikq) / omega  
           ENDDO
           !
           CALL fwfft( 'Rho', rhoc, dfftt )
           !
           vc = (0.0d0, 0.0d0)  
           DO ig = 1, dfftt%ngm  
               vc(dfftt%nl(ig))  = fac(ig) * rhoc(dfftt%nl(ig))   
               vc(dfftt%nlm(ig)) = fac(ig) * rhoc(dfftt%nlm(ig))  
           ENDDO  
           !
           CALL invfft( 'Rho', vc, dfftt )
           !
           DO ir = 1, NQR   
             RESULT(ir,ibnd) = RESULT(ir,ibnd) + locbuff(ir,ibnd,ikq) * vc(ir)   
           ENDDO  
           !
         ENDIF   
         !
         DO kbnd = 1, ibnd-1  
           IF ( (locmat(ibnd,kbnd,ikq) > local_thr) .AND. &  
                ( (x_occupation(ibnd,ikq) > 0.0d0) .OR.   &
                  (x_occupation(kbnd,ikq) > 0.0d0) ) ) THEN
             !
             !write(stdout,'(3I4,3f12.6,A)') ikq, ibnd, kbnd, x_occupation(ibnd,ikq), &
             !                    x_occupation(kbnd,ikq), locmat(ibnd,kbnd,ikq), ' IN '
             !
             DO ir = 1, NQR   
               rhoc(ir) = locbuff(ir,ibnd,ikq) * locbuff(ir,kbnd,ikq) / omega  
             ENDDO
             !
             npairs = npairs + 1  
             !
             CALL fwfft( 'Rho', rhoc, dfftt )
             !
             vc = (0.0d0, 0.0d0)
             !
             DO ig = 1, dfftt%ngm  
                 vc(dfftt%nl(ig))  = fac(ig) * rhoc(dfftt%nl(ig))   
                 vc(dfftt%nlm(ig)) = fac(ig) * rhoc(dfftt%nlm(ig))  
             ENDDO
             !
             CALL invfft( 'Rho', vc, dfftt )
             !
             DO ir = 1, NQR   
               RESULT(ir,kbnd) = RESULT(ir,kbnd) + x_occupation(ibnd,ikq) * locbuff(ir,ibnd,ikq) * vc(ir)   
             ENDDO
             !
             DO ir = 1, NQR   
               RESULT(ir,ibnd) = RESULT(ir,ibnd) + x_occupation(kbnd,ikq) * locbuff(ir,kbnd,ikq) * vc(ir)   
             ENDDO
             ! ELSE   
             !   write(stdout,'(3I4,3f12.6,A)') ikq, ibnd, kbnd, x_occupation(ibnd,ikq), &
             !               x_occupation(kbnd,ikq), locmat(ibnd,kbnd,ikq), '      OUT '  
           ENDIF
           !
         ENDDO
         !
       ENDDO   
       !
       DO jbnd = 1, nbnd  
         !
         CALL fwfft( 'Wave', RESULT(:,jbnd), dfftt )
         !
         DO ig = 1, npw  
            hpsi(ig,jbnd) = hpsi(ig,jbnd) - exxalfa*RESULT(dfftt%nl(ig),jbnd)   
         ENDDO
         !
       ENDDO
       !
    ENDDO
    !
    DEALLOCATE( fac, vc )
    DEALLOCATE( RESULT )
    !
    ! ... localized functions to G-space and exchange matrix onto localized functions
    ALLOCATE( RESULT(npw,nbnd) )
    RESULT = (0.0d0,0.0d0)
    !
    DO jbnd = 1, nbnd
      rhoc(:) = DBLE(locbuff(:,jbnd,ikq)) + (0.0d0,1.0d0)*0.0d0
      CALL fwfft( 'Wave' , rhoc, dfftt )
      DO ig = 1, npw
        RESULT(ig,jbnd) = rhoc(dfftt%nl(ig))
      ENDDO
    ENDDO
    !
    DEALLOCATE( rhoc )
    !
    CALL matcalc( 'M1-', .TRUE., 0, npw, nbnd, nbnd, RESULT, hpsi, mexx, exxe )
    !
    DEALLOCATE( RESULT )
    !
    NBands = INT(SUM(x_occupation(:,ikq)))
    ntot = NBands * (NBands-1)/2 + NBands * (nbnd-NBands)
    WRITE( stdout,'(7X,2(A,I12),A,f12.2)') '  Pairs(full): ',      ntot, &
                  '   Pairs(included): ', npairs, &
                  '   Pairs(%): ', DBLE(npairs)/DBLE(ntot)*100.0d0
    !
    CALL stop_clock( 'vexxloc' )
    !
  END SUBROUTINE vexx_loc
  !
  !------------------------------------------------------------------------
  SUBROUTINE vexx_loc_k( npw, NBands, hpsi, mexx, exxe )
    !-----------------------------------------------------------------------
    !! Generic, k-point version of \(\texttt{vexx}\).
    !
    USE cell_base,       ONLY : omega
    USE gvect,           ONLY : ngm, g
    USE wvfct,           ONLY : current_k, npwx
    USE klist,           ONLY : xk, nks, nkstot, igk_k
    USE fft_interfaces,  ONLY : fwfft, invfft
    USE exx_base,        ONLY : exxbuff_sync_host
    USE exx_base,        ONLY : index_xkq, nqs, index_xk, xkq_collect, &
                                g2_convolution
    !
    IMPLICIT NONE
    !
    INTEGER :: npw
    !! number of PW
    INTEGER :: NBands 
    !! number of bands
    COMPLEX(DP) :: hpsi(npwx*npol,NBands)
    !! h psi
    COMPLEX(DP) :: mexx(NBands,NBands)
    !! mexx contains in output the exchange matrix
    REAL(DP) :: exxe
    !! exx energy
    !
    ! ... local variables
    !
    COMPLEX(DP), ALLOCATABLE :: RESULT(:), RESULT2(:,:)
    COMPLEX(DP), ALLOCATABLE :: rhoc(:), vc(:)
    REAL(DP), ALLOCATABLE :: fac(:)
    INTEGER :: ibnd, jbnd, ik, ikq, iq
    INTEGER :: ir, ig, NBin, NBtot
    INTEGER :: current_ik, current_jk
    INTEGER :: nrxxs
    REAL(DP) :: xkp(3)
    REAL(DP) :: xkq(3)
    !
    INTEGER, EXTERNAL :: global_kpoint_index
    !
    CALL start_clock( 'vexxloc' )
    CALL exxbuff_sync_host()
    !
    ALLOCATE( fac(dfftt%ngm) )
    !
    nrxxs = dfftt%nnr
    !
    ALLOCATE( RESULT(nrxxs) )
    ALLOCATE( rhoc(nrxxs), vc(nrxxs) )
    !
    current_ik = global_kpoint_index ( nkstot, current_k )
    current_jk = index_xkq(current_ik,1)
    !
    NBin = 0  
    NBtot  = 0
    xkp = xk(:,current_k)
    DO jbnd = 1, NBands 
       RESULT = (0.0_DP, 0.0_DP) 
       DO iq = 1, nqs
          ikq = index_xkq(current_ik,iq)
          ik  = index_xk(ikq)
          xkq = xkq_collect(:,ikq)
          CALL g2_convolution( dfftt%ngm, gt, xkp, xkq, fac )
          DO ibnd = 1, NBands 
             ! IF ( abs(x_occupation(ibnd,ik)) < eps_occ) CYCLE 
             ! 
             NBtot = NBtot + 1 
             IF ((exxmat(ibnd,ikq,jbnd,current_k) > local_thr).AND. &
                ((x_occupation(ibnd,ik) > eps_occ))) THEN 
                  NBin = NBin + 1
               !
               ! write(stdout,'(4I4,f12.6,A)') ibnd, ikq, jbnd, current_k, exxmat(ibnd,ikq,jbnd,current_k), ' IN '
!$omp parallel do default(shared), private(ir)
               DO ir = 1, nrxxs
                  rhoc(ir)=CONJG(exxbuff(ir,ibnd,ikq))*exxbuff(ir,jbnd,current_jk) / omega
               ENDDO
!$omp end parallel do
               CALL fwfft( 'Rho', rhoc, dfftt )
               vc = (0._DP, 0._DP)
!$omp parallel do default(shared), private(ig)
               DO ig = 1, dfftt%ngm  
                  vc(dfftt%nl(ig)) = & 
                        fac(ig) * rhoc(dfftt%nl(ig)) * x_occupation(ibnd,ik) / nqs
               ENDDO
!$omp end parallel do
               CALL invfft( 'Rho', vc, dfftt )
!$omp parallel do default(shared), private(ir)
               DO ir = 1, nrxxs
                  RESULT(ir) = RESULT(ir) + vc(ir)*exxbuff(ir,ibnd,ikq)
               ENDDO
!$omp end parallel do
!            ELSE
!              write(stdout,'(4I4,f12.6,A)') ibnd, ikq, jbnd, current_k, exxmat(ibnd,ikq,jbnd,current_k), ' OUT'
             ENDIF 
         ENDDO
       ENDDO 
       !
       CALL fwfft( 'Wave', RESULT, dfftt )
!$omp parallel do default(shared), private(ig)
       DO ig = 1, npw
          hpsi(ig,jbnd) = hpsi(ig,jbnd) - exxalfa*RESULT(dfftt%nl(igk_k(ig,current_k)))
       ENDDO
!$omp end parallel do
    ENDDO 
    !
    DEALLOCATE( RESULT )
    DEALLOCATE( vc, fac )
    !
    ! ... Localized functions to G-space and exchange matrix onto localized functions
    ALLOCATE( RESULT2(npwx,NBands) )
    RESULT2 = (0.0d0,0.0d0)
    !
    DO jbnd = 1, NBands
      rhoc(:) = exxbuff(:,jbnd,current_jk)
      CALL fwfft( 'Wave' , rhoc, dfftt )
      DO ig = 1, npw
        RESULT2(ig,jbnd) = rhoc(dfftt%nl(igk_k(ig,current_k)))
      ENDDO
    ENDDO
    !
    DEALLOCATE( rhoc )
    CALL matcalc_k( 'M1-', .TRUE., 0, current_k, npwx*npol, NBands, NBands, RESULT2, hpsi, mexx, exxe )
    DEALLOCATE( RESULT2 )
    !
    WRITE(stdout,'(7X,2(A,I12),A,f12.2)') '  Pairs(full): ',  NBtot, &
            '   Pairs(included): ', NBin, &
            '   Pairs(%): ', DBLE(NBin)/DBLE(NBtot)*100.0d0
    !
    CALL stop_clock( 'vexxloc' )
    !
  END SUBROUTINE vexx_loc_k
  !
END MODULE exx

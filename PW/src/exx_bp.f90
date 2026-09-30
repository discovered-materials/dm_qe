! Copyrigh(C) 2005-2018 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------------
MODULE exx_bp
  !
  USE kinds,                ONLY : DP
  USE control_flags,        ONLY : gamma_only, use_gpu, many_fft, tqr
  USE exx_base,             ONLY : dfftt , exxbuff, exxbuff_d, exxbuff_to_device, exxbuff_sync_host, &
                                   exxbuff_host_stale, npwt, x_nbnd_occ, &
                                   gt, ggt, gcutmt, gkcut, gstart_t, ngmt_g, &
                                   eps_occ, exxalfa, x_occupation, x_occupation_d, &
                                   ibnd_start, ibnd_end, ibnd_buff_start, ibnd_buff_end
  USE noncollin_module,     ONLY : noncolin, npol
  USE cell_base,            ONLY : at, bg, tpiba2
  USE gvecw,                ONLY : ecutwfc
  USE symm_base,            ONLY : fft_fact
  USE mp_exx,               ONLY : negrp, inter_egrp_comm, intra_egrp_comm, nproc_egrp
  USE klist,                ONLY : nks, xk
  USE mp,                   ONLY : mp_sum
  USE io_global,            ONLY : stdout
  !
  INTEGER, ALLOCATABLE :: ig_l2gt(:), millt(:,:)
  !
  REAL(DP), ALLOCATABLE :: coulomb_fac(:,:,:)
  !! the Coulomb factor is reused between iterations
  !
  LOGICAL, ALLOCATABLE :: coulomb_done(:,:)
  !! list of which Coulomb factors have been calculated already
  !
  INTEGER :: exxinit_fill_last = -1
  !! which exxinit fill path was reported on stdout last (-1 none, 0 host, 1 device)
  !
  ! persistent state of the batched device exchange operator (vexx_bp_k_gpu_fast): the cuFFT plan
  ! cache, the resident Coulomb factors and work arrays, the ultrasoft/PAW augmentation tables
  ! and the force-mode arrays
  REAL(DP) :: exx_dexx_current = 1.0_DP
  !! the exchange energy change of the last outer step (dexx of module exx, which cannot be
  !! imported here), for the adaptive single-precision mode of the batched operator
  LOGICAL :: p20_force_mode = .FALSE.
  !! the batched operator runs in force mode (exx_force_us)
  LOGICAL :: exx_us_forces_done = .FALSE.
  !! forces.f90 prints the exact-exchange US/PAW term only when it was computed
  LOGICAL :: p15_fast_disabled = .FALSE.   ! P15: the fast path's device work arrays did not fit; original path from then on
#if defined(__CUDA)
  ! P6: persistent state of the batched fast path (vexx_k_gpu_fast)
  INTEGER, PARAMETER :: p6_nplans = 16   ! P11: was 8
  INTEGER :: p6_plans(p6_nplans) = 0, p6_plan_key(5,p6_nplans) = -1, p6_plan_next = 1
  INTEGER :: p6_plan_pinned(3) = 0    ! P11: plans handed to the running vexx_k_gpu_fast call (never evicted)
  LOGICAL :: p6_checked = .FALSE.
  INTEGER :: p6_sp_mode = 0, p6_pblk = -1, p6_nbuild = 0   ! P6e: pair block < 1 means automatic
  REAL(DP) :: p6_sp_thr = 1.0E-6_DP
  ! P10: stdout banners of the path decisions (parsed by generality/compare_matrix.py); -1 = not printed yet
  INTEGER :: p10_fast_last = -1, p10_kernel_last = -1
  LOGICAL, ALLOCATABLE :: cfac_done(:)
  REAL(DP), ALLOCATABLE :: cfac_all_d(:,:,:), p6_cfac_d(:,:), p6_facb_d(:,:), p6_xocc_d(:,:)
  COMPLEX(DP), ALLOCATABLE :: p6_rhoc_d(:,:), p6_psic_d(:,:), p6_res_d(:,:), p6_psi_d(:,:), p6_hpsi_d(:,:)
  COMPLEX(4), ALLOCATABLE :: p6_rhoc_s(:,:)
  INTEGER, ALLOCATABLE :: p6_ikq_d(:)
  INTEGER, ALLOCATABLE :: p6_ikq_h(:)          ! P6c: host work arrays, kept across calls
  REAL(DP), ALLOCATABLE :: gt_d(:,:), p6_xkq_d(:,:), p6_xkq_h(:,:)   ! P6d
  LOGICAL :: gt_d_stale = .TRUE.
  attributes(DEVICE) :: gt_d, p6_xkq_d
  REAL(DP), ALLOCATABLE :: p6_xocc_h(:,:)
  attributes(DEVICE) :: cfac_all_d, p6_cfac_d, p6_facb_d, p6_xocc_d, p6_rhoc_d, p6_psic_d, p6_res_d, &
                        p6_psi_d, p6_hpsi_d, p6_rhoc_s, p6_ikq_d
  ! P18: ultrasoft / PAW augmentation on the batched fast path
  INTEGER :: p18_na = 0, p18_nij = 0, p18_nkb = 0, p18_nat = 0, p18_bec_build = -1, p18_paw_build = -1, p18_sf_build = -1
  INTEGER :: p18_qs_k = -1, p18_qs_slot = 1, p18_qc_mx = 0
  LOGICAL :: p18_qs_cached = .FALSE., p18_banner = .FALSE.
  INTEGER, ALLOCATABLE, DEVICE :: p18_qc_ijv_d(:), p18_qc_nt_d(:), p18_qc_nlm_d(:), p18_qc_lp_d(:,:), p18_qc_l_d(:,:), p18_qc_ind_d(:,:)
  REAL(DP), ALLOCATABLE, DEVICE :: p18_qc_coef_d(:,:)   ! the (-i)^l ap(lp,ivl,jvl) of every pair column (qvan2.f90)
  ! P20: exact-exchange forces of ultrasoft / PAW runs: the fast path runs in force mode and leaves the exchange
  ! D matrix of every band and the structure-factor term of every (atom, ij) pair behind
  REAL(DP) :: p20_ex = 0.0_DP, p20_tpiba = 0.0_DP
  REAL(DP), ALLOCATABLE :: p20_wg_h(:)
  REAL(DP), ALLOCATABLE, DEVICE :: p20_fa_d(:,:), p20_xkq_d(:,:), p20_gt_d(:,:), p20_wg_d(:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: p20_vcg3_d(:,:), p20_y3_d(:,:), p20_dpb_d(:,:)
  LOGICAL, ALLOCATABLE :: p18_qs_done(:), p18_paw_done(:)
  REAL(DP), ALLOCATABLE :: p18_geo_h(:)                ! geometry stamp of the Q(G,A) cache: at, tau, EXX cutoff
  INTEGER, ALLOCATABLE :: p18_acol_h(:), p18_aat_h(:), p18_aih_h(:), p18_ajh_h(:), p18_aof_h(:,:,:), p18_nijt_h(:)
  INTEGER, ALLOCATABLE, DEVICE :: p18_acol_d(:), p18_aat_d(:), p18_aih_d(:), p18_ajh_d(:), p18_aof_d(:,:,:), &
                                   p18_ofs_d(:), p18_kbat_d(:), p18_kbih_d(:), p18_nhat_d(:)
  COMPLEX(DP), ALLOCATABLE, DEVICE :: p18_becxx_d(:,:,:), p18_becpsi_d(:,:), p18_sf_d(:,:), p18_qs_d(:,:,:,:), &
                                       p18_c_d(:,:), p18_aug_d(:,:), p18_vcg_d(:,:), p18_y_d(:,:), p18_deexx_d(:,:), &
                                       p18_vkb_d(:,:), p18_dpaw_d(:,:,:), p18_eigq_d(:,:)
  COMPLEX(DP), ALLOCATABLE :: p18_qgm_acc(:,:), p18_dpaw_h(:,:), p18_vkb_h(:,:), p18_eigq_h(:,:), p18_sf_h(:,:)
#endif
  !
 CONTAINS
#define _CX(A)  CMPLX(A,0._dp,kind=DP)
#define _CY(A)  CMPLX(0._dp,-A,kind=DP)
  !
  !-----------------------------------------------------------------------
  SUBROUTINE g2_convolution_all( ngm, g, xk, xkq, iq, current_k )
    !-----------------------------------------------------------------------
    !! Wrapper for g2_convolution.
    !
    USE kinds,     ONLY : DP
    USE klist,     ONLY : nks
    USE exx_base,  ONLY : g2_convolution, nqs
    !
    IMPLICIT NONE
    !
    INTEGER,  INTENT(IN) :: ngm
    !! Number of G vectors
    REAL(DP), INTENT(IN) :: g(3,ngm)
    !! Cartesian components of G vectors
    REAL(DP), INTENT(IN) :: xk(3)
    !! current k vector
    REAL(DP), INTENT(IN) :: xkq(3)
    !! current q vector
    INTEGER, INTENT(IN) :: current_k
    !! current k-point index
    INTEGER, INTENT(IN) :: iq
    !! q-grid point index
    !
    ! ... Check if coulomb_fac has been allocated
    IF( .NOT. ALLOCATED( coulomb_fac ) ) ALLOCATE( coulomb_fac(ngm,nqs,nks) )
    !
    ! ... Check if coulomb_done has been allocated
    IF( .NOT. ALLOCATED( coulomb_done) ) THEN
       ALLOCATE( coulomb_done(nqs,nks) )
       coulomb_done = .FALSE.
    ENDIF
    !
    ! ... return if this k and k' already computed, otherwise compute it
    IF ( coulomb_done(iq,current_k) ) RETURN
    !
    CALL g2_convolution( ngm, g, xk, xkq, coulomb_fac(:,iq,current_k) )
    !
    coulomb_done(iq,current_k) = .TRUE.
    !
  END SUBROUTINE g2_convolution_all
  !
  !------------------------------------------------------------------------
  SUBROUTINE set_dfftt_grid_bp( )
    !------------------------------------------------------------------------
    USE exx_bp_utils,         ONLY : smap_exx
    USE command_line_options, ONLY : nmany_, pencil_decomposition_
    USE mp_bands,             ONLY : nyfft
    USE symm_base,            ONLY : fft_fact
    USE fft_types,            ONLY : fft_type_init
    USE recvec_subs,          ONLY : ggen
    !
    IMPLICIT NONE
       LOGICAL :: lpara
       INTEGER :: ngmt
    INTEGER, EXTERNAL :: n_plane_waves
    !
    WRITE( 6, "(5X,'Exchange parallelized over bands (',i4,' band groups)')" ) &
           negrp
    lpara = ( nproc_egrp > 1 )
    CALL fft_type_init( dfftt, smap_exx, "rho", gamma_only, lpara,     &
                        intra_egrp_comm, at, bg, gcutmt, gcutmt/gkcut, &
                        fft_fact=fft_fact, nyfft=nyfft, nmany=nmany_,  &
                        use_pd=pencil_decomposition_ )
    ngmt = dfftt%ngm
    ngmt_g = ngmt
    CALL mp_sum( ngmt_g, intra_egrp_comm )
    ALLOCATE( gt(3,dfftt%ngm) )
    ALLOCATE( ggt(dfftt%ngm)  )
    ALLOCATE( millt(3,dfftt%ngm) )
    ALLOCATE( ig_l2gt(dfftt%ngm) )
    !
    CALL ggen( dfftt, gamma_only, at, bg, gcutmt, ngmt_g, ngmt, &
               gt, ggt, millt, ig_l2gt, gstart_t )
    !
    DEALLOCATE( ig_l2gt )
    DEALLOCATE( millt )
    npwt = n_plane_waves( ecutwfc/tpiba2, nks, xk, gt, ngmt )
    !
    RETURN
    !
  END SUBROUTINE set_dfftt_grid_bp
  !
  !------------------------------------------------------------------------
  SUBROUTINE exxinit_bp( )
    !------------------------------------------------------------------------
    !! This subroutine is run before the first H_psi() of each iteration. 
    !! It saves the wavefunctions for the right density matrix, in real space.
    !
    USE wavefunctions,        ONLY : psic, evc
    USE io_files,             ONLY : iunwfc_exx
    USE buffers,              ONLY : get_buffer
    USE wvfct,                ONLY : nbnd, npwx
    USE klist,                ONLY : ngk, nks, nkstot
    USE mp_exx,               ONLY : me_egrp, init_index_over_band,  &
                                     inter_egrp_comm,           &
                                     iexx_start, iexx_end, &
                                     all_start, all_end
    USE mp,                   ONLY : mp_bcast
    USE scatter_mod,          ONLY : gather_grid, scatter_grid
    USE fft_interfaces,       ONLY : invfft
    USE mp_orthopools,        ONLY : intra_orthopool_comm
    USE exx_base,             ONLY : nkqs, index_xk, index_sym,  &
                                     rir, working_pool, d_spin, local_thr
    USE exx_bp_utils,         ONLY : change_data_structure, nwordwfc_exx, &
                                     igk_exx, evc_exx, transform_evc_to_exx, igk_exx_d
    USE mp_pools,             ONLY : npool
    USE symm_base,            ONLY : nsym
    USE io_global,            ONLY : stdout
#if defined(__CUDA)
    USE device_memcpy_m,      ONLY : dev_memset
    USE device_fbuff_m,       ONLY : dev_buf
#endif
    !
    IMPLICIT NONE
    !
    ! ... local variables
    !
    INTEGER :: ik, ibnd, i, j, k, ir, isym, ikq, ig, ierr
    INTEGER :: ibnd_loop_start
    INTEGER :: ipol, jpol
    REAL(DP), ALLOCATABLE :: occ(:,:)
    COMPLEX(DP),ALLOCATABLE :: temppsic(:)
#if defined(__USE_INTEL_HBM_DIRECTIVES)
!DIR$ ATTRIBUTES FASTMEM :: temppsic
#elif defined(__USE_CRAY_HBM_DIRECTIVES)
!DIR$ memory(bandwidth) temppsic
#endif
    COMPLEX(DP),ALLOCATABLE :: temppsic_nc(:,:), psic_nc(:,:)
    COMPLEX(DP),POINTER     :: psic_nc_d(:,:)
#if defined(__CUDA)
    attributes(DEVICE)      :: psic_nc_d
#endif
    ! device-side fill of exxbuff (collinear, one FFT process, one pool, no localization)
    LOGICAL :: gpu_fill
    CHARACTER(LEN=96) :: fill_why
#if defined(__CUDA)
    COMPLEX(DP), ALLOCATABLE :: temppsic_d(:), evc_exx_d(:,:)
    INTEGER, ALLOCATABLE :: rir_d(:,:)
    INTEGER, POINTER :: nlt_d(:), nlmt_d(:)
    INTEGER :: jb1, jb2
    attributes(DEVICE) :: temppsic_d, evc_exx_d, rir_d, nlt_d, nlmt_d
#endif
    INTEGER :: nxxs, nrxxs
#if defined(__MPI)
    COMPLEX(DP),ALLOCATABLE  :: temppsic_all(:), psic_all(:)
    COMPLEX(DP), ALLOCATABLE :: temppsic_all_nc(:,:), psic_all_nc(:,:)
#endif
    INTEGER :: npw, current_ik
    INTEGER, EXTERNAL :: global_kpoint_index
    INTEGER :: ibnd_start_new, ibnd_end_new, max_buff_bands_per_egrp
    INTEGER :: ibnd_exx, evc_offset
    !
    !$acc update device(evc)
    CALL transform_evc_to_exx( 2 )
    !
    ! Note that nxxs is not the same as nrxxs in parallel case
    nxxs = dfftt%nr1x * dfftt%nr2x * dfftt%nr3x
    nrxxs = dfftt%nnr
    !
#if defined(__MPI)
    IF (noncolin) THEN
       ALLOCATE( psic_all_nc(nxxs,npol), temppsic_all_nc(nxxs,npol) )
    ELSEIF ( .NOT. gamma_only ) THEN
       ALLOCATE( psic_all(nxxs), temppsic_all(nxxs) )
    ENDIF
#endif
    IF (noncolin) THEN
       ALLOCATE( temppsic_nc(nrxxs, npol), psic_nc(nrxxs, npol) )
    ELSEIF ( .NOT. gamma_only ) THEN
       ALLOCATE( temppsic(nrxxs) )
    ENDIF
    !
    CALL divide( inter_egrp_comm, x_nbnd_occ, ibnd_start, ibnd_end )
    CALL init_index_over_band( inter_egrp_comm, nbnd, nbnd )
    !
    ! ... this will cause exxbuff to be calculated for every band
    ibnd_start_new = iexx_start
    ibnd_end_new = iexx_end
    !
    IF ( gamma_only ) THEN
        ibnd_buff_start = (ibnd_start_new+1)/2
        ibnd_buff_end   = (ibnd_end_new+1)/2
        max_buff_bands_per_egrp = MAXVAL((all_end(:)+1)/2-(all_start(:)+1)/2)+1
    ELSE
        ibnd_buff_start = ibnd_start_new
        ibnd_buff_end   = ibnd_end_new
        max_buff_bands_per_egrp = MAXVAL(all_end(:)-all_start(:))+1
    ENDIF
    !
    IF (.NOT. ALLOCATED(exxbuff)) THEN
       IF (gamma_only) THEN
          ALLOCATE( exxbuff(nrxxs*npol,ibnd_buff_start:ibnd_buff_start + &
                                        max_buff_bands_per_egrp-1,nkqs) ) ! THIS WORKS as for k
       ELSE
          ALLOCATE( exxbuff(nrxxs*npol,ibnd_buff_start:ibnd_buff_start + &
                                        max_buff_bands_per_egrp-1,nkqs) )
       ENDIF
    ENDIF
    !
    IF (.not. allocated(exxbuff_d) .and. use_gpu) THEN
       IF (gamma_only) THEN
          ALLOCATE( exxbuff_d(nrxxs*npol, ibnd_buff_start:ibnd_buff_start+max_buff_bands_per_egrp-1, nks))
       ELSE
          ALLOCATE( exxbuff_d(nrxxs*npol, ibnd_buff_start:ibnd_buff_start+max_buff_bands_per_egrp-1, nkqs))
       END IF
    ENDIF
    !
    IF (use_gpu) THEN
#if defined (__CUDA)
       ! NB: the array bounds are not passed to the subroutine.
       !
       ! See https://software.intel.com/en-us/forums/intel-fortran-compiler-for-linux-and-mac-os-x/topic/269311
       !
       ! NB: TO BE CORRECTED WITH THE NEW DeviceXlib LIBRARY that dues internal slicing right!
       CALL dev_memset(exxbuff_d, (0.0_DP,0.0_DP), &
                                 (/ 1,nrxxs*npol/), 1, &
                                 (/ ibnd_buff_start, ibnd_buff_end /), ibnd_buff_start, &
                                 (/ 1,SIZE(exxbuff_d,3)/), 1)
#endif
    ELSE
       !$omp parallel do collapse(3) default(shared) firstprivate(npol,nrxxs,nkqs, &
       !$omp                ibnd_buff_start,ibnd_buff_end) private(ir,ibnd,ikq,ipol)
       DO ikq = 1, SIZE(exxbuff,3) 
          DO ibnd = ibnd_buff_start, ibnd_buff_end
             DO ir = 1, nrxxs*npol
                exxbuff(ir,ibnd,ikq) = (0.0_DP,0.0_DP)
             ENDDO
          ENDDO
       ENDDO
       ! the above loops will replaced with the following line soon
       !CALL threaded_memset(exxbuff, 0.0_DP, nrxxs*npol*SIZE(exxbuff,2)*nkqs*2)
    ENDIF
    !
    ! ... fill exxbuff on the device when the layout allows it: collinear, no orbital
    ! ... localization, one FFT process per band group, one pool, undistributed grid.
    ! ... Every other configuration keeps the host fill below.
    !
#if defined(__CUDA)
    p6_nbuild = p6_nbuild + 1     ! the adaptive precision of the fast path counts the builds
#endif
    gpu_fill = .FALSE.
#if defined(__CUDA)
    gpu_fill = use_gpu .AND. (.NOT. noncolin) .AND. (local_thr <= 0.0_DP) .AND. &
               (dfftt%nproc == 1) .AND. (npool == 1) .AND. (nrxxs == nxxs)
    IF (gpu_fill) THEN
       ALLOCATE( temppsic_d(nrxxs), evc_exx_d(SIZE(evc_exx,1),SIZE(evc_exx,2)) )
       nlt_d => dfftt%nl_d
       IF (gamma_only) THEN
          ! no symmetry table at Gamma (exx_set_symm is not called)
          nlmt_d => dfftt%nlm_d
          ALLOCATE( rir_d(1,1) )
       ELSE
          ALLOCATE( rir_d(nxxs,nsym) )
          rir_d = rir
       ENDIF
    ENDIF
#endif
    !
    ! ... This is parallelized over pools. Each pool computes only its k-points
    !
    KPOINTS_LOOP : &
    DO ik = 1, nks
       !
       IF ( nks > 1 ) CALL get_buffer( evc_exx, nwordwfc_exx, iunwfc_exx, ik )
       !
       ! ik         = index of k-point in this pool
       ! current_ik = index of k-point over all pools
       !
       current_ik = global_kpoint_index( nkstot, ik )
#if defined(__CUDA)
       IF (gpu_fill) evc_exx_d = evc_exx
#endif
       !
       IF_GAMMA_ONLY : &
       IF (gamma_only) THEN
          !
          IF (MOD(iexx_start,2) == 0) THEN
             ibnd_loop_start = iexx_start-1
          ELSE
             ibnd_loop_start = iexx_start
          ENDIF
          !
          IF (gpu_fill) THEN
#if defined(__CUDA)
          ! the Gamma-trick fill on the device: the host loop below, band pair by band pair
          evc_offset = 0
          DO ibnd = ibnd_loop_start, iexx_end, 2
             temppsic_d = ( 0._DP, 0._DP )
             IF ( ibnd < iexx_end ) THEN
                IF ( ibnd == ibnd_loop_start .AND. MOD(iexx_start,2) == 0 ) THEN
                   !$cuf kernel do (1)
                   DO ig = 1, npwt
                      temppsic_d(nlt_d(ig))  = ( 0._DP, 1._DP ) * evc_exx_d(ig,1)
                      temppsic_d(nlmt_d(ig)) = ( 0._DP, 1._DP ) * CONJG( evc_exx_d(ig,1) )
                   ENDDO
                   evc_offset = -1
                ELSE
                   jb1 = ibnd - ibnd_loop_start + evc_offset + 1
                   jb2 = jb1 + 1
                   !$cuf kernel do (1)
                   DO ig = 1, npwt
                      temppsic_d(nlt_d(ig))  = evc_exx_d(ig,jb1) + ( 0._DP, 1._DP ) * evc_exx_d(ig,jb2)
                      temppsic_d(nlmt_d(ig)) = CONJG( evc_exx_d(ig,jb1) ) + ( 0._DP, 1._DP ) * CONJG( evc_exx_d(ig,jb2) )
                   ENDDO
                ENDIF
             ELSE
                jb1 = ibnd - ibnd_loop_start + evc_offset + 1
                !$cuf kernel do (1)
                DO ig = 1, npwt
                   temppsic_d(nlt_d(ig))  = evc_exx_d(ig,jb1)
                   temppsic_d(nlmt_d(ig)) = CONJG( evc_exx_d(ig,jb1) )
                ENDDO
             ENDIF
             CALL invfft( 'Wave', temppsic_d, dfftt )
             exxbuff_d(1:nrxxs,(ibnd+1)/2,current_ik) = temppsic_d(1:nrxxs)
          ENDDO
#endif
          ELSE
          evc_offset = 0
          DO ibnd = ibnd_loop_start, iexx_end, 2
             !
             psic(:) = ( 0._DP, 0._DP )
             !
             IF ( ibnd < iexx_end ) THEN
                IF ( ibnd == ibnd_loop_start .AND. MOD(iexx_start,2) == 0 ) THEN
                   DO ig = 1, npwt
                      psic(dfftt%nl(ig))  = ( 0._DP, 1._DP )*evc_exx(ig,1)
                      psic(dfftt%nlm(ig)) = ( 0._DP, 1._DP )*CONJG(evc_exx(ig,1))
                   ENDDO
                   evc_offset = -1
                ELSE
                   DO ig = 1, npwt
                      psic(dfftt%nl(ig))  = evc_exx(ig,ibnd-ibnd_loop_start+evc_offset+1) &
                           + ( 0._DP, 1._DP ) * evc_exx(ig,ibnd-ibnd_loop_start+evc_offset+2)
                      psic(dfftt%nlm(ig)) = CONJG( evc_exx(ig,ibnd-ibnd_loop_start+evc_offset+1) ) &
                           + ( 0._DP, 1._DP ) * CONJG( evc_exx(ig,ibnd-ibnd_loop_start+evc_offset+2) )
                   ENDDO
                ENDIF
             ELSE
                DO ig=1,npwt
                   psic(dfftt%nl (ig)) = evc_exx(ig,ibnd-ibnd_loop_start+evc_offset+1)
                   psic(dfftt%nlm(ig)) = CONJG( evc_exx(ig,ibnd-ibnd_loop_start+evc_offset+1) )
                ENDDO
             ENDIF
             !
             CALL invfft( 'Wave', psic, dfftt )
             !
             exxbuff(1:nrxxs,(ibnd+1)/2,current_ik)=psic(1:nrxxs) 
             !
          ENDDO
          ENDIF   ! gpu_fill
          !
       ELSE IF_GAMMA_ONLY
          !
          npw = ngk (ik)
          IBND_LOOP_K : &
          DO ibnd = iexx_start, iexx_end
             !
             ibnd_exx = ibnd
             IF (noncolin) THEN
!$omp parallel do default(shared) private(ir) firstprivate(nrxxs)
                DO ir = 1, nrxxs
                   temppsic_nc(ir,1) = ( 0._DP, 0._DP )
                   temppsic_nc(ir,2) = ( 0._DP, 0._DP )
                ENDDO
!$omp parallel do default(shared) private(ig) firstprivate(npw,ik,ibnd_exx)
                DO ig = 1, npw
                   temppsic_nc(dfftt%nl(igk_exx(ig,ik)),1) = evc_exx(ig,ibnd-iexx_start+1)
                ENDDO
!$omp end parallel do
                CALL invfft( 'Wave', temppsic_nc(:,1), dfftt )
!$omp parallel do default(shared) private(ig) firstprivate(npw,ik,ibnd_exx,npwx)
                DO ig = 1, npw
                   temppsic_nc(dfftt%nl(igk_exx(ig,ik)),2) = evc_exx(ig+npwx,ibnd-iexx_start+1)
                ENDDO
!$omp end parallel do
                CALL invfft( 'Wave', temppsic_nc(:,2), dfftt )
             ELSE IF (gpu_fill) THEN
#if defined(__CUDA)
                ! psi_k,ibnd(r) on the device
                temppsic_d = ( 0._DP, 0._DP )
                !$cuf kernel do (1)
                DO ig = 1, npw
                   temppsic_d( nlt_d(igk_exx_d(ig,ik)) ) = evc_exx_d(ig,ibnd-iexx_start+1)
                ENDDO
                CALL invfft( 'Wave', temppsic_d, dfftt )
#endif
             ELSE
!$omp parallel do default(shared) private(ir) firstprivate(nrxxs)
                DO ir = 1, nrxxs
                   temppsic(ir) = ( 0._DP, 0._DP )
                ENDDO
!$omp parallel do default(shared) private(ig) firstprivate(npw,ik,ibnd_exx)
                DO ig = 1, npw
                   temppsic(dfftt%nl(igk_exx(ig,ik))) = evc_exx(ig,ibnd-iexx_start+1)
                ENDDO
!$omp end parallel do
                CALL invfft( 'Wave', temppsic, dfftt )
             ENDIF
             !
             DO ikq = 1, nkqs
                !
                IF (index_xk(ikq) /= current_ik) CYCLE
                isym = ABS(index_sym(ikq) )
                !
                IF (noncolin) THEN ! noncolinear
#if defined(__MPI)
                   DO ipol = 1, npol
                      CALL gather_grid( dfftt, temppsic_nc(:,ipol), temppsic_all_nc(:,ipol) )
                   ENDDO
                   !
                   IF ( me_egrp == 0 ) THEN
!$omp parallel do collapse(2)
                      DO ipol = 1, npol
                         DO ir = 1, nxxs
                            psic_all_nc(ir,ipol) = (0.0_DP, 0.0_DP)
                            DO jpol = 1, npol
                               psic_all_nc(ir,ipol) = psic_all_nc(ir,ipol) + &
                                             CONJG(d_spin(jpol,ipol,isym)) * &
                                             temppsic_all_nc(rir(ir,isym),jpol)
                            ENDDO
                         ENDDO
                      ENDDO
!$omp end parallel do
                   ENDIF
                   !
                   DO ipol = 1, npol
                      CALL scatter_grid( dfftt, psic_all_nc(:,ipol), psic_nc(:,ipol) )
                   ENDDO
#else
!$omp parallel do collapse(2)
                   DO ipol = 1, npol
                      DO ir = 1, nxxs
                         psic_nc(ir,ipol) = (0._DP,0._DP)
                         DO jpol = 1, npol
                            psic_nc(ir,ipol) = psic_nc(ir,ipol) + CONJG(d_spin(jpol,ipol,isym))* &
                                               temppsic_nc(rir(ir,isym),jpol)
                         ENDDO
                      ENDDO
                   ENDDO
!$omp end parallel do
#endif
                   !
#if defined (__CUDA)
                   IF (use_gpu) CALL dev_buf%lock_buffer(psic_nc_d, (/nrxxs, npol/), ierr)
                   IF (use_gpu) psic_nc_d = psic_nc
#endif
                   !
                   IF (index_sym(ikq) > 0 ) THEN
                      IF (use_gpu) THEN
                         associate(exxbuff=>exxbuff_d, psic_nc=>psic_nc_d)
                         ! sym. op. without time reversal: normal case
                         !$cuf kernel do 
                         DO ir=1,nrxxs
                            exxbuff(ir,ibnd,ikq)=psic_nc(ir,1)
                            exxbuff(ir+nrxxs,ibnd,ikq)=psic_nc(ir,2)
                         ENDDO
                         end associate
                      ELSE
                         ! sym. op. without time reversal: normal case
!$omp parallel do default(shared) private(ir) firstprivate(ibnd,isym,ikq)
                      DO ir = 1, nrxxs
                         exxbuff(ir,ibnd,ikq) = psic_nc(ir,1)
                         exxbuff(ir+nrxxs,ibnd,ikq) = psic_nc(ir,2)
                      ENDDO
!$omp end parallel do
                      END IF
                   ELSE
                      ! sym. op. with time reversal: spin 1->2*, 2->-1*
                      IF (use_gpu) THEN
                         associate(exxbuff=>exxbuff_d, psic_nc=>psic_nc_d)
                         ! sym. op. with time reversal: spin 1->2*, 2->-1*
                         !$cuf kernel do 
                         DO ir=1,nrxxs
                            exxbuff(ir,ibnd,ikq)=CONJG(psic_nc(ir,2))
                            exxbuff(ir+nrxxs,ibnd,ikq)=-CONJG(psic_nc(ir,1))
                         ENDDO
                         end associate
                      ELSE
!$omp parallel do default(shared) private(ir) firstprivate(ibnd,isym,ikq)
                      DO ir = 1, nrxxs
                         exxbuff(ir,ibnd,ikq) = CONJG(psic_nc(ir,2))
                         exxbuff(ir+nrxxs,ibnd,ikq) = -CONJG(psic_nc(ir,1))
                      ENDDO
!$omp end parallel do
                      ENDIF
                   ENDIF
#if defined(__CUDA)
                IF (use_gpu) CALL dev_buf%release_buffer(psic_nc_d, ierr)
                IF (use_gpu) exxbuff = exxbuff_d
#endif
                ELSE IF (gpu_fill) THEN
#if defined(__CUDA)
                   ! rotate on the device straight into exxbuff_d
                   IF (index_sym(ikq) > 0) THEN
                      !$cuf kernel do (1)
                      DO ir = 1, nrxxs
                         exxbuff_d(ir,ibnd,ikq) = temppsic_d(rir_d(ir,isym))
                      ENDDO
                   ELSE
                      !$cuf kernel do (1)
                      DO ir = 1, nrxxs
                         exxbuff_d(ir,ibnd,ikq) = CONJG( temppsic_d(rir_d(ir,isym)) )
                      ENDDO
                   ENDIF
#endif
                ELSE ! noncolinear
#if defined(__MPI)
                   CALL gather_grid( dfftt, temppsic, temppsic_all )
                   IF ( me_egrp == 0 ) THEN
!$omp parallel do default(shared) private(ir) firstprivate(isym)
                      DO ir = 1, nxxs
                         psic_all(ir) = temppsic_all(rir(ir,isym))
                      ENDDO
!$omp end parallel do
                   ENDIF
                   CALL scatter_grid( dfftt, psic_all, psic )
#else
!$omp parallel do default(shared) private(ir) firstprivate(isym)
                   DO ir = 1, nrxxs
                      psic(ir) = temppsic(rir(ir,isym))
                   ENDDO
!$omp end parallel do
#endif
!$omp parallel do default(shared) private(ir) firstprivate(isym,ibnd,ikq)
                   DO ir = 1, nrxxs
                      IF (index_sym(ikq) < 0 ) THEN
                         psic(ir) = CONJG(psic(ir))
                      ENDIF
                      exxbuff(ir,ibnd,ikq) = psic(ir)
                   ENDDO
!$omp end parallel do
                   !
                ENDIF ! noncolinear
                !
             ENDDO
             !
          ENDDO&
          IBND_LOOP_K
          !
       ENDIF&
       IF_GAMMA_ONLY
    ENDDO&
    KPOINTS_LOOP
#if defined(__CUDA)
    IF (gpu_fill) THEN
       IF (negrp > 1) THEN
          ! band groups rotate the host array in vexx: keep the host copy current
          CALL start_clock( 'exxbuff_d2h' )
          exxbuff = exxbuff_d
          CALL stop_clock( 'exxbuff_d2h' )
          exxbuff_host_stale = .FALSE.
       ELSE
          ! the host readers (exxenergy, exx_stress, the restart write, vexx_loc)
          ! download on demand through exxbuff_sync_host
          exxbuff_host_stale = .TRUE.
       ENDIF
       DEALLOCATE( temppsic_d, rir_d, evc_exx_d )
    ENDIF
#endif
    !
    IF (noncolin) THEN
       DEALLOCATE( temppsic_nc, psic_nc )
#if defined(__MPI)
       DEALLOCATE( temppsic_all_nc, psic_all_nc )
#endif
    ELSE IF ( .NOT. gamma_only ) THEN
       DEALLOCATE( temppsic )
#if defined(__MPI)
       DEALLOCATE( temppsic_all, psic_all )
#endif
    ENDIF
    !
    ! Each wavefunction in exxbuff is computed by a single pool, collect among 
    ! pools in a smart way (i.e. without doing all-to-all sum and bcast)
    ! See also the initialization of working_pool in exx_mp_init
    ! Note that in Gamma-only LSDA can be parallelized over two pools, and there
    ! is no need to communicate anything: each pools deals with its own spin
    !
    IF ( .NOT. gamma_only ) THEN
       DO ikq = 1, nkqs
         CALL mp_bcast( exxbuff(:,:,ikq), working_pool(ikq), intra_orthopool_comm ) 
       ENDDO
    ENDIF
    !
    !
    ! the device copy of the orbital buffer is refreshed here, once per exxinit,
    ! and after every later host-side write (restart read, band-group shift,
    ! localization); the GPU vexx routines use it as it is
    !
    IF (.NOT. gpu_fill) CALL exxbuff_to_device()
    !
    CALL change_data_structure( .FALSE. )
    !
    ! say which fill path this exxinit took, once and again whenever it changes
    !
    IF ( use_gpu .AND. MERGE( 1, 0, gpu_fill ) /= exxinit_fill_last ) THEN
       exxinit_fill_last = MERGE( 1, 0, gpu_fill )
       IF ( gpu_fill ) THEN
          WRITE( stdout, '(5X,"EXX-GPU: exxinit fills the orbital buffer on the device")' )
       ELSE
          fill_why = ''
          IF ( noncolin ) fill_why = TRIM(fill_why) // ' noncolinear'
          IF ( local_thr > 0.0_DP ) fill_why = TRIM(fill_why) // ' localized-orbitals'
          IF ( dfftt%nproc /= 1 ) fill_why = TRIM(fill_why) // ' parallel-FFT'
          IF ( npool /= 1 ) fill_why = TRIM(fill_why) // ' k-point-pools'
          IF ( nrxxs /= nxxs ) fill_why = TRIM(fill_why) // ' distributed-grid'
          WRITE( stdout, '(5X,"EXX-GPU: exxinit fills the orbital buffer on the host;",A)' ) TRIM(fill_why)
       ENDIF
    ENDIF
    !
    RETURN
    !
  END SUBROUTINE exxinit_bp
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vexx_bp( lda, n, m, psi, hpsi, becpsi )
    !-----------------------------------------------------------------------
    !! Wrapper routine computing V_x\psi, V_x = exchange potential. 
    !! Calls generic version vexx_k or Gamma-specific one vexx_gamma.
    !
    USE becmod,         ONLY : bec_type
    USE mp_exx,         ONLY : negrp, inter_egrp_comm, init_index_over_band
    USE wvfct,          ONLY : nbnd
    USE exx_bp_utils,   ONLY : transform_psi_to_exx, transform_hpsi_to_local, &
                               psi_exx, hpsi_exx
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
    LOGICAL :: fast_ok
    !
    IF (negrp > 1) THEN
       CALL init_index_over_band( inter_egrp_comm, nbnd, m )
       !
       ! ... transform psi to the EXX data structure
       CALL transform_psi_to_exx( lda, n, m, psi )
    ENDIF
    !
    ! ... calculate the EXX contribution to hpsi
    !
    IF ( gamma_only ) THEN
       IF (negrp == 1)THEN
          IF (.not. use_gpu) CALL vexx_bp_gamma( lda, n, m, psi, hpsi, becpsi )
          IF (      use_gpu) CALL vexx_bp_gamma_gpu( lda, n, m, psi, hpsi, becpsi )
       ELSE
          IF (.not. use_gpu) CALL vexx_bp_gamma( lda, n, m, psi_exx, hpsi_exx, becpsi )
          IF (      use_gpu) CALL vexx_bp_gamma_gpu( lda, n, m, psi_exx, hpsi_exx, becpsi )
       ENDIF
    ELSE
       IF (negrp == 1)THEN
          IF (.not. use_gpu) CALL vexx_bp_k( lda, n, m, psi, hpsi, becpsi )
#if defined(__CUDA)
          IF ( use_gpu .AND. vexx_fast_path() ) THEN
             CALL vexx_bp_k_gpu_fast( lda, n, m, psi, hpsi, fast_ok, becpsi )
             IF ( .NOT. fast_ok ) CALL vexx_bp_k_gpu( lda, n, m, psi, hpsi, becpsi )
          ELSE IF ( use_gpu ) THEN
             CALL vexx_bp_k_gpu( lda, n, m, psi, hpsi, becpsi )
          ENDIF
#else
          IF (      use_gpu) CALL vexx_bp_k_gpu( lda, n, m, psi, hpsi, becpsi )
#endif
       ELSE
          IF (.not. use_gpu) CALL vexx_bp_k( lda, n, m, psi_exx, hpsi_exx, becpsi )
#if defined(__CUDA)
          IF ( use_gpu .AND. vexx_fast_path() ) THEN
             CALL vexx_bp_k_gpu_fast( lda, n, m, psi_exx, hpsi_exx, fast_ok, becpsi )
             IF ( .NOT. fast_ok ) CALL vexx_bp_k_gpu( lda, n, m, psi_exx, hpsi_exx, becpsi )
          ELSE IF ( use_gpu ) THEN
             CALL vexx_bp_k_gpu( lda, n, m, psi_exx, hpsi_exx, becpsi )
          ENDIF
#else
          IF (      use_gpu) CALL vexx_bp_k_gpu( lda, n, m, psi_exx, hpsi_exx, becpsi )
#endif
       ENDIF
    ENDIF
    !
    IF (negrp > 1) THEN
       !
       ! ... transform hpsi to the local data structure
       !
       CALL transform_hpsi_to_local(lda,n,m,hpsi)
       !
    ENDIF
    !
  END SUBROUTINE vexx_bp
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vexx_bp_gamma( lda, n, m, psi, hpsi, becpsi )
    !-----------------------------------------------------------------------
    !! Gamma-specific version of vexx.
    !
    USE constants,      ONLY : fpi, e2, pi
    USE cell_base,      ONLY : omega
    USE gvect,          ONLY : ngm, g
    USE wvfct,          ONLY : npwx, current_k, nbnd
    USE klist,          ONLY : xk, nks, nkstot, igk_k
    USE fft_interfaces, ONLY : fwfft, invfft
    USE becmod,         ONLY : bec_type
    USE mp_exx,         ONLY : inter_egrp_comm, my_egrp_id, &
                               intra_egrp_comm, me_egrp, &
                               negrp, max_pairs, egrp_pairs, ibands, nibands, &
                               iexx_istart, iexx_iend, &
                               all_start, all_end, iexx_start, jblock, max_ibands
    USE mp,             ONLY : mp_sum, mp_barrier, mp_circular_shift_left
    USE uspp,           ONLY : nkb, okvan
    USE paw_variables,  ONLY : okpaw
    USE us_exx,         ONLY : bexg_merge, becxx, addusxx_g, addusxx_r, &
                               newdxx_g, newdxx_r, add_nlxx_pot, &
                               qvan_init, qvan_clean
    USE paw_exx,        ONLY : PAW_newdxx
    USE exx_base,       ONLY : nqs, index_xkq, index_xk, xkq_collect
    USE exx_bp_utils,   ONLY : result_sum, igk_exx
    !
    IMPLICIT NONE
    !
    INTEGER :: lda
    !! input: leading dimension of arrays psi and hpsi
    INTEGER :: n
    !! input: true dimension of psi and hpsi
    INTEGER :: m
    !! input: number of states psi
    COMPLEX(DP) :: psi(lda*npol,max_ibands)
    !! input: m wavefunctions
    COMPLEX(DP) :: hpsi(lda*npol,max_ibands)
    !! output: V_x*psi
    TYPE(bec_type), OPTIONAL :: becpsi ! or call a calbec(...psi) instead
    !! input: <beta|psi>, optional but needed for US and PAW case
    !
    ! ... local variables
    !
    COMPLEX(DP), ALLOCATABLE :: result(:,:)
    REAL(DP), ALLOCATABLE :: temppsic_dble (:)
    REAL(DP), ALLOCATABLE :: temppsic_aimag(:)
    !
    COMPLEX(DP), ALLOCATABLE :: vc(:), deexx(:,:)
    INTEGER :: ibnd, ik, im , ikq, iq, ipol
    INTEGER :: ir, ig
    INTEGER :: current_ik
    INTEGER :: ibnd_loop_start
    INTEGER :: nrxxs
    REAL(DP) :: x1, x2, xkp(3)
    REAL(DP) :: xkq(3)
    INTEGER, EXTERNAL :: global_kpoint_index
    INTEGER :: ialloc
    COMPLEX(DP), ALLOCATABLE :: big_result(:,:)
    INTEGER :: iproc, nproc_egrp, ii, ipair
    INTEGER :: jbnd, jstart, jend
    ! ... scratch space for fft of psi and rho
    COMPLEX(DP), ALLOCATABLE :: psi_rhoc_work(:)
    INTEGER :: jblock_start, jblock_end
    INTEGER :: iegrp, wegrp
    INTEGER :: exxbuff_index
    INTEGER :: ending_im
    !
    ialloc = nibands(my_egrp_id+1)
    nrxxs = dfftt%nnr
    !
    !ALLOCATE( result(nrxxs), temppsic_DBLE(nrxxs), temppsic_aimag(nrxxs) )
    ALLOCATE( result(nrxxs,ialloc), temppsic_DBLE(nrxxs) )
    ALLOCATE( temppsic_aimag(nrxxs) )
    ALLOCATE( psi_rhoc_work(nrxxs) )
    !
    ALLOCATE( vc(nrxxs) )
    IF (okvan) ALLOCATE( deexx(nkb,ialloc) )
    !
    current_ik = global_kpoint_index( nkstot, current_k )
    xkp = xk(:,current_k)
    !
    ALLOCATE( big_result(n,m) )
    big_result = 0.0_DP
    result = 0.0_DP
    !
    DO ii = 1, nibands(my_egrp_id+1)
       IF (okvan) deexx(:,ii) = 0.0_DP
    ENDDO
    !
    ! Here the loops start
    !
    INTERNAL_LOOP_ON_Q : &
    DO iq = 1, nqs
       !
       ikq  = index_xkq(current_ik,iq)
       ik   = index_xk(ikq)
       xkq  = xkq_collect(:,ikq)
       !
       ! calculate the 1/|r-r'| (actually, k+q+g) factor and place it in fac
       CALL g2_convolution_all( dfftt%ngm, gt, xkp, xkq, iq, current_k )
       IF ( okvan .AND..NOT.tqr ) CALL qvan_init( dfftt%ngm, xkq, xkp )
       !
       DO iegrp = 1, negrp
          !
          ! compute the id of group whose data is currently worked on
          wegrp = MOD(iegrp+my_egrp_id-1, negrp)+1
          !
          jblock_start = all_start(wegrp)
          jblock_end   = all_end(wegrp)
          !
          LOOP_ON_PSI_BANDS : &
          DO ii = 1,  nibands(my_egrp_id+1)
             !
             ibnd = ibands(ii,my_egrp_id+1)
             !
             IF (ibnd==0 .OR. ibnd>m) CYCLE
             !
             IF (MOD(ii,2) == 1) THEN
                !
                psi_rhoc_work = (0._DP,0._DP)
                !
                IF ((ii+1) <= MIN(m,nibands(my_egrp_id+1))) THEN
                   ! deal with double bands
!$omp parallel do  default(shared), private(ig)
                   DO ig = 1, npwt
                      psi_rhoc_work( dfftt%nl(ig) )  =       psi(ig, ii) + (0._DP,1._DP) * psi(ig, ii+1)
                      psi_rhoc_work( dfftt%nlm(ig) ) = CONJG(psi(ig, ii) - (0._DP,1._DP) * psi(ig, ii+1))
                   ENDDO
!$omp end parallel do
                ENDIF
                !
                IF ( ii == MIN(m,nibands(my_egrp_id+1)) ) THEN
                   ! deal with a single last band
!$omp parallel do  default(shared), private(ig)
                   DO ig = 1, npwt
                      psi_rhoc_work( dfftt%nl(ig) )  =       psi(ig,ii)
                      psi_rhoc_work( dfftt%nlm(ig) ) = CONJG(psi(ig,ii))
                   ENDDO
!$omp end parallel do
                ENDIF
                !
                CALL invfft( 'Wave', psi_rhoc_work, dfftt )
!$omp parallel do default(shared), private(ir)
                DO ir = 1, nrxxs
                   temppsic_DBLE(ir)  = DBLE( psi_rhoc_work(ir) )
                   temppsic_aimag(ir) = AIMAG( psi_rhoc_work(ir) )
                ENDDO
!$omp end parallel do
                !
             ENDIF
             !
             !
             !determine which j-bands to calculate
             jstart = 0
             jend = 0
             DO ipair = 1, max_pairs
                IF (egrp_pairs(1,ipair,my_egrp_id+1) == ibnd) THEN
                   IF (jstart == 0)THEN
                      jstart = egrp_pairs(2,ipair,my_egrp_id+1)
                      jend = jstart
                   ELSE
                      jend = egrp_pairs(2,ipair,my_egrp_id+1)
                   ENDIF
                ENDIF
             ENDDO
             !
             jstart = MAX(jstart,jblock_start)
             jend = MIN(jend,jblock_end)
             !
             IF (MOD(jstart,2) ==0 ) THEN
                ibnd_loop_start = jstart-1
             ELSE
                ibnd_loop_start = jstart
             ENDIF
             !
             IBND_LOOP_GAM : &
             DO jbnd = ibnd_loop_start, jend, 2 !for each band of psi
                !
                exxbuff_index = (jbnd+1)/2-(all_start(wegrp)+1)/2+(iexx_start+1)/2
                !
                IF ( jbnd < jstart ) THEN
                   x1 = 0.0_DP
                ELSE
                   x1 = x_occupation(jbnd,ik)
                ENDIF
                IF ( jbnd == jend) THEN
                   x2 = 0.0_DP
                ELSE
                   x2 = x_occupation(jbnd+1,ik)
                ENDIF
                IF ( ABS(x1) < eps_occ .AND. ABS(x2) < eps_occ ) CYCLE
                !
                ! ... calculate rho in real space. Gamma tricks are used.
                ! temppsic is real; tempphic contains one band in the real part,
                ! another one in the imaginary part; the same applies to rhoc
                !
                IF ( MOD(ii,2) == 0 ) THEN
!$omp parallel do default(shared), private(ir)
                   DO ir = 1, nrxxs
                      psi_rhoc_work(ir) = exxbuff(ir,exxbuff_index,ikq) * temppsic_aimag(ir) / omega
                   ENDDO
!$omp end parallel do
                ELSE
!$omp parallel do default(shared), private(ir)
                   DO ir = 1, nrxxs
                      psi_rhoc_work(ir) = exxbuff(ir,exxbuff_index,ikq) * temppsic_DBLE(ir) / omega
                   ENDDO
!$omp end parallel do
                ENDIF
                !
                ! ... bring rho to G-space
                !
                !   >>> add augmentation in REAL SPACE here
                IF (okvan .AND. tqr) THEN
                   IF (jbnd >= jstart) &
                        CALL addusxx_r( psi_rhoc_work, &
                       _CX(becxx(ikq)%r(:,jbnd)), _CX(becpsi%r(:,ibnd)))
                   IF (jbnd < jend) &
                        CALL addusxx_r( psi_rhoc_work, &
                       _CY(becxx(ikq)%r(:,jbnd+1)),_CX(becpsi%r(:,ibnd)))
                ENDIF
                !
                CALL fwfft( 'Rho', psi_rhoc_work, dfftt )
                !   >>>> add augmentation in G SPACE here
                IF (okvan .AND. .NOT. tqr) THEN
                   ! ... contribution from one band added to real (in real space) part of rhoc
                   IF (jbnd >= jstart) &
                        CALL addusxx_g( dfftt, psi_rhoc_work, xkq,  xkp, 'r', &
                        becphi_r=becxx(ikq)%r(:,jbnd), becpsi_r=becpsi%r(:,ibnd) )
                   ! ... contribution from following band added to imaginary (in real space) part of rhoc
                   IF (jbnd < jend) &
                        CALL addusxx_g( dfftt, psi_rhoc_work, xkq,  xkp, 'i', &
                        becphi_r=becxx(ikq)%r(:,jbnd+1), becpsi_r=becpsi%r(:,ibnd) )
                ENDIF
                !   >>>> charge density done
                !
                vc = 0._DP
                !
!$omp parallel do default(shared), private(ig)
                DO ig = 1, dfftt%ngm
                   !
                   vc(dfftt%nl(ig))  = coulomb_fac(ig,iq,current_k) * psi_rhoc_work(dfftt%nl(ig))
                   vc(dfftt%nlm(ig)) = coulomb_fac(ig,iq,current_k) * psi_rhoc_work(dfftt%nlm(ig))
                   !
                ENDDO
!$omp end parallel do
                !
                !   >>>>  compute <psi|H_fock G SPACE here
                IF (okvan .AND. .NOT. tqr) THEN
                   IF (jbnd >= jstart) &
                        CALL newdxx_g( dfftt, vc, xkq, xkp, 'r', deexx(:,ii), &
                           becphi_r=x1*becxx(ikq)%r(:,jbnd) )
                   IF (jbnd<jend) &
                        CALL newdxx_g( dfftt, vc, xkq, xkp, 'i', deexx(:,ii), &
                            becphi_r=x2*becxx(ikq)%r(:,jbnd+1) )
                ENDIF
                !
                !brings back v in real space
                CALL invfft( 'Rho', vc, dfftt )
                !
                !   >>>>  compute <psi|H_fock REAL SPACE here
                IF (okvan .AND. tqr) THEN
                   IF (jbnd >= jstart) &
                        CALL newdxx_r( dfftt,vc, _CX(x1*becxx(ikq)%r(:,jbnd)), deexx(:,ii) )
                   IF (jbnd < jend) &
                        CALL newdxx_r( dfftt,vc, _CY(x2*becxx(ikq)%r(:,jbnd+1)), deexx(:,ii) )
                ENDIF
                !
                IF (okpaw) THEN
                   IF (jbnd >= jstart) &
                        CALL PAW_newdxx( x1/nqs, _CX(becxx(ikq)%r(:,jbnd)),&
                                                _CX(becpsi%r(:,ibnd)), deexx(:,ii) )
                   IF (jbnd < jend) &
                        CALL PAW_newdxx( x2/nqs, _CX(becxx(ikq)%r(:,jbnd+1)),&
                                                _CX(becpsi%r(:,ibnd)), deexx(:,ii) )
                ENDIF
                !
                ! ... accumulates over bands and k points
                !
!$omp parallel do default(shared), private(ir)
                DO ir = 1, nrxxs
                   result(ir,ii) = result(ir,ii) &
                                 + x1* DBLE(vc(ir))* DBLE(exxbuff(ir,exxbuff_index,ikq)) &
                                 + x2*AIMAG(vc(ir))*AIMAG(exxbuff(ir,exxbuff_index,ikq))
                ENDDO
!$omp end parallel do
                !
             ENDDO &
             IBND_LOOP_GAM
             !
          ENDDO &
          LOOP_ON_PSI_BANDS
          !
          ! get the next nbnd/negrp data
          IF (negrp > 1) CALL mp_circular_shift_left( exxbuff(:,:,ikq), me_egrp, inter_egrp_comm )
          !
       ENDDO ! iegrp
       IF (okvan .AND. .NOT.tqr) CALL qvan_clean()
    ENDDO &
    INTERNAL_LOOP_ON_Q
    !
    DO ii = 1, nibands(my_egrp_id+1)
       !
       ibnd = ibands(ii,my_egrp_id+1)
       !
       IF (ibnd == 0 .OR. ibnd>m) CYCLE
       !
       IF (okvan) THEN
          CALL mp_sum( deexx(:,ii), intra_egrp_comm )
       ENDIF
       !
       ! ... brings back result in G-space
       !
       CALL fwfft( 'Wave' , result(:,ii), dfftt )
       !
       ! ... communicate result
       DO ig = 1, n
          big_result(ig,ibnd) = big_result(ig,ibnd) - exxalfa*result(dfftt%nl(igk_exx(ig,current_k)),ii)
       ENDDO
       !
       ! ... add non-local \sum_I |beta_I> \alpha_Ii (the sum on i is outside)
       IF (okvan) CALL add_nlxx_pot( lda, big_result(:,ibnd), xkp, n, &
                                     igk_exx(1,current_k), deexx(:,ii), eps_occ, exxalfa )
    ENDDO
    !
    CALL result_sum( n*npol, m, big_result )
    !
    IF (iexx_istart(my_egrp_id+1) > 0) THEN
       IF (negrp == 1) THEN
          ending_im = m
       ELSE
          ending_im = iexx_iend(my_egrp_id+1) - iexx_istart(my_egrp_id+1) + 1
       END IF
       DO im = 1, ending_im
!$omp parallel do default(shared), private(ig) firstprivate(im,n)
           DO ig = 1, n
              hpsi(ig,im) = hpsi(ig,im) + big_result(ig,im+iexx_istart(my_egrp_id+1)-1)
           ENDDO
!$omp end parallel do
       ENDDO
    ENDIF
    !
    DEALLOCATE( big_result )
    DEALLOCATE( result, temppsic_dble, temppsic_aimag )
    DEALLOCATE( psi_rhoc_work )
    DEALLOCATE( vc )
    IF (okvan) DEALLOCATE( deexx )
    !
  END SUBROUTINE vexx_bp_gamma
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vexx_bp_gamma_gpu(lda, n, m, psi, hpsi, becpsi)
  !-----------------------------------------------------------------------
    !
    ! ... Gamma-specific version of vexx
    !
    USE constants,      ONLY : fpi, e2, pi
    USE cell_base,      ONLY : omega
    USE gvect,          ONLY : ngm, g
    USE wvfct,          ONLY : npwx, current_k, nbnd
    USE klist,          ONLY : xk, nks, nkstot, igk_k
    USE fft_interfaces, ONLY : fwfft, invfft
    USE becmod,         ONLY : bec_type
    USE mp_exx,         ONLY : inter_egrp_comm, my_egrp_id, &
                               intra_egrp_comm, me_egrp, &
                               negrp, max_pairs, egrp_pairs, ibands, nibands, &
                               iexx_istart, iexx_iend, &
                               all_start, all_end, iexx_start, jblock, max_ibands
    USE mp,             ONLY : mp_sum, mp_barrier, mp_circular_shift_left
    USE uspp,           ONLY : nkb, okvan
    USE paw_variables,  ONLY : okpaw
    USE us_exx,         ONLY : bexg_merge, becxx, addusxx_g, addusxx_r, &
                               newdxx_g, newdxx_r, add_nlxx_pot, &
                               qvan_init, qvan_clean
    USE paw_exx,        ONLY : PAW_newdxx
    USE exx_base,       ONLY : nqs, index_xkq, index_xk, xkq_collect
    USE exx_bp_utils,   ONLY : result_sum, igk_exx, igk_exx_d
#if defined(__CUDA)
    USE device_memcpy_m, ONLY : dev_memset
#endif
    !
    !
    IMPLICIT NONE
    !
    INTEGER                  :: lda, n, m
    COMPLEX(DP)              :: psi(lda*npol,max_ibands)
    COMPLEX(DP)              :: hpsi(lda*npol,max_ibands)
    TYPE(bec_type), OPTIONAL :: becpsi ! or call a calbec(...psi) instead
    !
    ! local variables
    COMPLEX(DP), ALLOCATABLE :: psi_d(:,:)
#if defined(__CUDA)
    attributes(DEVICE)       :: psi_d
#endif
    COMPLEX(DP),ALLOCATABLE :: result_d(:,:)
#if defined(__CUDA)
    attributes(DEVICE)       :: result_d
#endif
    REAL(DP),ALLOCATABLE :: temppsic_dble_d (:)
    REAL(DP),ALLOCATABLE :: temppsic_aimag_d(:)
#if defined(__CUDA)
    attributes(DEVICE)   :: temppsic_dble_d, temppsic_aimag_d
#endif
    !
    COMPLEX(DP),ALLOCATABLE :: vc(:), deexx(:,:), vc_d(:)
    REAL(DP),   ALLOCATABLE :: fac_d(:)
#if defined(__CUDA)
    attributes(DEVICE)   :: vc_d, fac_d
#endif
    INTEGER          :: ibnd, ik, im , ikq, iq, ipol
    INTEGER          :: ir, ig
    INTEGER          :: current_ik
    INTEGER          :: ibnd_loop_start
    INTEGER          :: nrxxs
    REAL(DP) :: x1, x2, xkp(3)
    REAL(DP) :: xkq(3)
    INTEGER, EXTERNAL :: global_kpoint_index
    INTEGER :: ialloc
    COMPLEX(DP), ALLOCATABLE :: big_result(:,:)
    COMPLEX(DP), ALLOCATABLE :: big_result_d(:,:)
#if defined(__CUDA)
    attributes(DEVICE) :: big_result_d
    attributes(PINNED) :: big_result
#endif
    INTEGER :: iproc, nproc_egrp, ii, ipair
    INTEGER :: jbnd, jstart, jend
    ! scratch space for fft of psi and rho
    COMPLEX(DP), ALLOCATABLE :: psi_rhoc_work(:)
    COMPLEX(DP), ALLOCATABLE :: psi_rhoc_work_d(:)
#if defined(__CUDA)
    attributes(DEVICE) :: psi_rhoc_work_d
#endif
    INTEGER :: jblock_start, jblock_end
    INTEGER :: iegrp, wegrp
    INTEGER :: exxbuff_index
    INTEGER :: ending_im
    !hack around PGI bug
    INTEGER, POINTER :: dfftt__nl(:)
    INTEGER, POINTER :: dfftt__nlm(:)
#if defined(__CUDA)
    attributes(DEVICE) :: dfftt__nl
    attributes(DEVICE) :: dfftt__nlm
#endif
    !
    ! CUDA Sync
    dfftt__nl=>dfftt%nl_d
    dfftt__nlm=>dfftt%nlm_d
    ALLOCATE(psi_d, source=psi)
    !
    ialloc = nibands(my_egrp_id+1)
    !
    ALLOCATE( fac_d(dfftt%ngm) )
    nrxxs= dfftt%nnr
    !
    ALLOCATE( result_d(nrxxs,ialloc))
    ALLOCATE( temppsic_dble_d(nrxxs) )
    ALLOCATE( temppsic_aimag_d(nrxxs) )
    !
    ALLOCATE( psi_rhoc_work(nrxxs) )
    ALLOCATE( psi_rhoc_work_d(nrxxs) )
    !
    ALLOCATE( vc(nrxxs))
    ALLOCATE( vc_d(nrxxs))
    IF(okvan) ALLOCATE(deexx(nkb,ialloc))
    !
    current_ik = global_kpoint_index ( nkstot, current_k )
    xkp = xk(:,current_k)
    !
    allocate(big_result(n,m))
    allocate(big_result_d(n,m))
    big_result = 0.0_DP
#if defined(__CUDA)
    CALL dev_memset(big_result_d,  (0.0_DP, 0.0_DP))
    CALL dev_memset(result_d,  (0.0_DP, 0.0_DP))
#endif
    !
    DO ii=1, nibands(my_egrp_id+1)
       IF(okvan) deexx(:,ii) = 0.0_DP
    END DO
    !
    ! Here the loops start
    !
    INTERNAL_LOOP_ON_Q : &
    DO iq=1,nqs
       !
       ikq  = index_xkq(current_ik,iq)
       ik   = index_xk(ikq)
       xkq  = xkq_collect(:,ikq)
       !
       ! calculate the 1/|r-r'| (actually, k+q+g) factor and place it in fac
       CALL g2_convolution_all(dfftt%ngm, gt, xkp, xkq, iq, current_k)
       IF ( okvan .and..not.tqr ) CALL qvan_init (dfftt%ngm, xkq, xkp)
       !
       ! copy coulomb_fac to device
       fac_d(:) = coulomb_fac(:,iq,current_k)
       !
       DO iegrp=1, negrp
          !
          ! compute the id of group whose data is currently worked on
          wegrp = MOD(iegrp+my_egrp_id-1, negrp)+1
          !
          jblock_start = all_start(wegrp)
          jblock_end   = all_end(wegrp)
          !
          LOOP_ON_PSI_BANDS : &
          DO ii = 1,  nibands(my_egrp_id+1)
             !
             ibnd = ibands(ii,my_egrp_id+1)
             !
             IF (ibnd.eq.0.or.ibnd.gt.m) CYCLE
             !
             IF ( mod(ii,2)==1 ) THEN
                !
#if defined(__CUDA)
                CALL dev_memset(psi_rhoc_work_d, (0.0_DP,0.0_DP), (/ 1, nrxxs /), 1 )
#endif
                !
                IF ( (ii+1)<=min(m,nibands(my_egrp_id+1)) ) THEN
                   ! deal with double bands
!$cuf kernel do
                   DO ig = 1, npwt
                      psi_rhoc_work_d( dfftt__nl(ig) )  =       psi_d(ig, ii) + (0._DP,1._DP) * psi_d(ig, ii+1)
                      psi_rhoc_work_d( dfftt__nlm(ig) ) = conjg(psi_d(ig, ii) - (0._DP,1._DP) * psi_d(ig, ii+1))
                   ENDDO

                ENDIF
                !
                IF ( ii==min(m,nibands(my_egrp_id+1)) ) THEN
                   ! deal with a single last band
!$cuf kernel do
                   DO ig = 1, npwt
                      psi_rhoc_work_d( dfftt__nl(ig) )  =       psi_d(ig,ii)
                      psi_rhoc_work_d( dfftt__nlm(ig) ) = conjg(psi_d(ig,ii))
                   ENDDO
                   !
                ENDIF
                !
                CALL invfft ('Wave', psi_rhoc_work_d, dfftt)
!$cuf kernel do
                DO ir = 1, nrxxs
                   temppsic_dble_d(ir)  = dble ( psi_rhoc_work_d(ir) )
                   temppsic_aimag_d(ir) = aimag( psi_rhoc_work_d(ir) )
                ENDDO
                !
             ENDIF
             !
             !
             !determine which j-bands to calculate
             jstart = 0
             jend = 0
             DO ipair=1, max_pairs
                IF(egrp_pairs(1,ipair,my_egrp_id+1).eq.ibnd)THEN
                   IF(jstart.eq.0)THEN
                      jstart = egrp_pairs(2,ipair,my_egrp_id+1)
                      jend = jstart
                   ELSE
                      jend = egrp_pairs(2,ipair,my_egrp_id+1)
                   END IF
                END IF
             END DO
             !
             jstart = max(jstart,jblock_start)
             jend = min(jend,jblock_end)
             !
             IF(mod(jstart,2)==0) THEN
                ibnd_loop_start=jstart-1
             ELSE
                ibnd_loop_start=jstart
             ENDIF
             !
             IBND_LOOP_GAM : &
             DO jbnd=ibnd_loop_start,jend, 2 !for each band of psi
                !
                exxbuff_index = (jbnd+1)/2-(all_start(wegrp)+1)/2+(iexx_start+1)/2
                !
                IF( jbnd < jstart ) THEN
                   x1 = 0.0_DP
                ELSE
                   x1 = x_occupation(jbnd,  ik)
                ENDIF
                IF( jbnd == jend) THEN
                   x2 = 0.0_DP
                ELSE
                   x2 = x_occupation(jbnd+1,  ik)
                ENDIF
                IF ( abs(x1) < eps_occ .and. abs(x2) < eps_occ ) CYCLE
                !
                ! calculate rho in real space. Gamma tricks are used.
                ! temppsic is real; tempphic contains one band in the real part,
                ! another one in the imaginary part; the same applies to rhoc
                !
                IF( mod(ii,2) == 0 ) THEN
!$cuf kernel do
                   DO ir = 1, nrxxs
                      psi_rhoc_work_d(ir) = exxbuff_d(ir,exxbuff_index,ikq) * temppsic_aimag_d(ir) / omega
                   ENDDO

                ELSE
!$cuf kernel do
                   DO ir = 1, nrxxs
                      psi_rhoc_work_d(ir) = exxbuff_d(ir,exxbuff_index,ikq) * temppsic_dble_d(ir) / omega
                   ENDDO

                ENDIF
                !
                ! bring rho to G-space
                !
                !   >>>> add augmentation in REAL SPACE here
                IF(okvan .and. tqr) THEN
                   psi_rhoc_work = psi_rhoc_work_d
                   IF(jbnd>=jstart) &
                        CALL addusxx_r(psi_rhoc_work, &
                       _CX(becxx(ikq)%r(:,jbnd)), _CX(becpsi%r(:,ibnd)))
                   IF(jbnd<jend) &
                        CALL addusxx_r(psi_rhoc_work, &
                       _CY(becxx(ikq)%r(:,jbnd+1)),_CX(becpsi%r(:,ibnd)))
                   psi_rhoc_work_d = psi_rhoc_work
                ENDIF
                !
                CALL fwfft ('Rho', psi_rhoc_work_d, dfftt)
                !   >>>> add augmentation in G SPACE here
                IF(okvan .and. .not. tqr) THEN
                   psi_rhoc_work = psi_rhoc_work_d
                   ! contribution from one band added to real (in real space) part of rhoc
                   IF(jbnd>=jstart) &
                        CALL addusxx_g(dfftt, psi_rhoc_work, xkq,  xkp, 'r', &
                        becphi_r=becxx(ikq)%r(:,jbnd), becpsi_r=becpsi%r(:,ibnd) )
                   ! contribution from following band added to imaginary (in real space) part of rhoc
                   IF(jbnd<jend) &
                        CALL addusxx_g(dfftt, psi_rhoc_work, xkq,  xkp, 'i', &
                        becphi_r=becxx(ikq)%r(:,jbnd+1), becpsi_r=becpsi%r(:,ibnd) )
                   psi_rhoc_work_d = psi_rhoc_work 
                ENDIF
                !   >>>> charge density done
                !
                vc_d = 0._DP
                !
!$cuf kernel do
                DO ig = 1, dfftt%ngm
                   !
                   vc_d(dfftt__nl(ig))  = fac_d(ig) * psi_rhoc_work_d(dfftt__nl(ig))
                   vc_d(dfftt__nlm(ig)) = fac_d(ig) * psi_rhoc_work_d(dfftt__nlm(ig))
                   !
                ENDDO
                !
                !   >>>>  compute <psi|H_fock G SPACE here
                IF(okvan .and. .not. tqr) THEN
                   vc = vc_d
                   IF(jbnd>=jstart) &
                        CALL newdxx_g(dfftt, vc, xkq, xkp, 'r', deexx(:,ii), &
                           becphi_r=x1*becxx(ikq)%r(:,jbnd))
                   IF(jbnd<jend) &
                        CALL newdxx_g(dfftt, vc, xkq, xkp, 'i', deexx(:,ii), &
                            becphi_r=x2*becxx(ikq)%r(:,jbnd+1))
                ENDIF
                !
                !brings back v in real space
                CALL invfft ('Rho', vc_d, dfftt)
                !
                !   >>>>  compute <psi|H_fock REAL SPACE here
                IF(okvan .and. tqr) THEN
                   vc = vc_d
                   IF(jbnd>=jstart) &
                        CALL newdxx_r(dfftt,vc, _CX(x1*becxx(ikq)%r(:,jbnd)), deexx(:,ii))
                   IF(jbnd<jend) &
                        CALL newdxx_r(dfftt,vc, _CY(x2*becxx(ikq)%r(:,jbnd+1)), deexx(:,ii))
                ENDIF
                !
                IF(okpaw) THEN
                   IF(jbnd>=jstart) &
                        CALL PAW_newdxx(x1/nqs, _CX(becxx(ikq)%r(:,jbnd)),&
                                                _CX(becpsi%r(:,ibnd)), deexx(:,ii))
                   IF(jbnd<jend) &
                        CALL PAW_newdxx(x2/nqs, _CX(becxx(ikq)%r(:,jbnd+1)),&
                                                _CX(becpsi%r(:,ibnd)), deexx(:,ii))
                ENDIF
                !
                ! accumulates over bands and k points
                !
!$cuf kernel do
                DO ir = 1, nrxxs
                   result_d(ir,ii) = result_d(ir,ii) &
                                 + x1* dble(vc_d(ir))* dble(exxbuff_d(ir,exxbuff_index,ikq)) &
                                 + x2*aimag(vc_d(ir))*aimag(exxbuff_d(ir,exxbuff_index,ikq))
                ENDDO
                !
             ENDDO &
             IBND_LOOP_GAM
             !
          ENDDO &
          LOOP_ON_PSI_BANDS
          !
          ! get the next nbnd/negrp data
          IF (negrp>1) call mp_circular_shift_left( exxbuff_d(:,:,ikq), me_egrp, inter_egrp_comm )
          !
       ENDDO ! iegrp
       IF ( okvan .and..not.tqr ) CALL qvan_clean ()
    ENDDO &
    INTERNAL_LOOP_ON_Q
    !
    DO ii=1, nibands(my_egrp_id+1)
       !
       ibnd = ibands(ii,my_egrp_id+1)
       !
       IF (ibnd.eq.0.or.ibnd.gt.m) CYCLE
       !
       IF(okvan) THEN
          CALL mp_sum(deexx(:,ii),intra_egrp_comm)
       ENDIF
       !
       !
       ! brings back result in G-space
       !
       CALL fwfft( 'Wave' , result_d(:,ii), dfftt )
       !communicate result
       !$cuf kernel do
       DO ig = 1, n
          big_result_d(ig,ibnd) = big_result_d(ig,ibnd) - exxalfa*result_d(dfftt__nl(igk_exx_d(ig,current_k)),ii)
       END DO
       big_result(:,ibnd) = big_result_d(:,ibnd)
       !
       ! add non-local \sum_I |beta_I> \alpha_Ii (the sum on i is outside)
       IF(okvan) CALL add_nlxx_pot (lda, big_result(:,ibnd), xkp, n, &
            igk_exx(1,current_k), deexx(:,ii), eps_occ, exxalfa)
    END DO
    !
    CALL result_sum(n*npol, m, big_result)
    IF (iexx_istart(my_egrp_id+1).gt.0) THEN
       IF (negrp == 1) then
          ending_im = m
       ELSE
          ending_im = iexx_iend(my_egrp_id+1) - iexx_istart(my_egrp_id+1) + 1
       END IF
       DO im=1, ending_im
!$omp parallel do default(shared), private(ig) firstprivate(im,n)
           DO ig = 1, n
              hpsi(ig,im)=hpsi(ig,im) + big_result(ig,im+iexx_istart(my_egrp_id+1)-1)
           ENDDO
!$omp end parallel do
       END DO
    END IF
    !
    DEALLOCATE(big_result)
    DEALLOCATE(big_result_d)
    DEALLOCATE(result_d)
    DEALLOCATE(temppsic_dble_d)
    DEALLOCATE(temppsic_aimag_d)
    DEALLOCATE(psi_rhoc_work_d)
    DEALLOCATE(psi_d)
    DEALLOCATE(vc)
    DEALLOCATE(vc_d)
    DEALLOCATE(fac_d)
    IF(okvan) DEALLOCATE(deexx)
    !
    !-----------------------------------------------------------------------
  END SUBROUTINE vexx_bp_gamma_gpu
  !-----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vexx_bp_k( lda, n, m, psi, hpsi, becpsi )
    !-----------------------------------------------------------------------
    !! Generic, k-point version of vexx.
    !
    USE constants,      ONLY : fpi, e2, pi
    USE cell_base,      ONLY : omega
    USE gvect,          ONLY : ngm, g
    USE wvfct,          ONLY : npwx, current_k, nbnd
    USE klist,          ONLY : xk, nks, nkstot
    USE fft_interfaces, ONLY : fwfft, invfft
    USE becmod,         ONLY : bec_type
    USE mp_exx,         ONLY : inter_egrp_comm, my_egrp_id, negrp, &
                               intra_egrp_comm, me_egrp, &
                               max_pairs, egrp_pairs, ibands, nibands, &
                               max_ibands, iexx_istart, iexx_iend, &
                               all_start, all_end, iexx_start, jblock
    USE mp,             ONLY : mp_sum, mp_barrier, mp_circular_shift_left
    USE uspp,           ONLY : nkb, okvan
    USE paw_variables,  ONLY : okpaw
    USE us_exx,         ONLY : bexg_merge, becxx, addusxx_g, addusxx_r, &
                               newdxx_g, newdxx_r, add_nlxx_pot, &
                               qvan_init, qvan_clean
    USE paw_exx,        ONLY : PAW_newdxx
    USE exx_base,       ONLY : nqs, xkq_collect, index_xkq, index_xk
    USE exx_bp_utils,   ONLY : result_sum, igk_exx
    USE io_global,      ONLY : stdout
    !
    !
    IMPLICIT NONE
    !
    INTEGER :: lda
    !! input: leading dimension of arrays psi and hpsi
    INTEGER :: n
    !! input: true dimension of psi and hpsi
    INTEGER :: m
    !! input: number of states psi
    COMPLEX(DP) :: psi(lda*npol,max_ibands)
    !! input: m wavefunctions
    COMPLEX(DP) :: hpsi(lda*npol,max_ibands)
    !! output: V_x*psi
    TYPE(bec_type), OPTIONAL :: becpsi  ! or call a calbec(...psi) instead
    !! input: <beta|psi>, optional but needed for US and PAW case
    !
    ! ... local variables
    !
    COMPLEX(DP),ALLOCATABLE :: temppsic(:,:), result(:,:)
#if defined(__USE_INTEL_HBM_DIRECTIVES)
!DIR$ ATTRIBUTES FASTMEM :: result
#elif defined(__USE_CRAY_HBM_DIRECTIVES)
!DIR$ memory(bandwidth) result
#endif
    COMPLEX(DP),ALLOCATABLE :: temppsic_nc(:,:,:),result_nc(:,:,:)
    INTEGER :: request_send, request_recv
    !
    COMPLEX(DP),ALLOCATABLE :: deexx(:,:)
    COMPLEX(DP),ALLOCATABLE,TARGET :: rhoc(:,:), vc(:,:)
#if defined(__USE_MANY_FFT)
    COMPLEX(DP),POINTER :: prhoc(:), pvc(:)
#endif
#if defined(__USE_INTEL_HBM_DIRECTIVES)
!DIR$ ATTRIBUTES FASTMEM :: rhoc, vc
#elif defined(__USE_CRAY_HBM_DIRECTIVES)
!DIR$ memory(bandwidth) rhoc, vc
#endif
    REAL(DP),   ALLOCATABLE :: fac(:), facb(:)
    INTEGER :: ibnd, ik, im , ikq, iq, ipol
    INTEGER :: ir, ig, ir_start, ir_end
    INTEGER :: irt, nrt, nblock
    INTEGER :: current_ik
    INTEGER :: ibnd_loop_start
    INTEGER :: nrxxs
    REAL(DP) :: x1, x2, xkp(3), omega_inv, nqs_inv
    REAL(DP) :: xkq(3)
    INTEGER, EXTERNAL :: global_kpoint_index
    DOUBLE PRECISION :: max, tempx
    COMPLEX(DP), ALLOCATABLE :: big_result(:,:)
    INTEGER :: ir_out, ipair, jbnd
    INTEGER :: ii, jstart, jend, jcount, jind
    INTEGER :: ialloc, ending_im
    INTEGER :: ijt, njt, jblock_start, jblock_end
    INTEGER :: iegrp, wegrp
    !
    ialloc = nibands(my_egrp_id+1)
    !
    ALLOCATE( fac(dfftt%ngm) )
    nrxxs= dfftt%nnr
    ALLOCATE( facb(nrxxs) )
    !
    IF (noncolin) THEN
       ALLOCATE( temppsic_nc(nrxxs,npol,ialloc), result_nc(nrxxs,npol,ialloc) )
    ELSE
       ALLOCATE( temppsic(nrxxs,ialloc), result(nrxxs,ialloc) )
    ENDIF
    !
    IF (okvan) ALLOCATE( deexx(nkb,ialloc) )
    !
    current_ik = global_kpoint_index( nkstot, current_k )
    xkp = xk(:,current_k)
    !
    ALLOCATE( big_result(n*npol,m) )
    big_result = 0.0_DP
    !
    ! allocate arrays for rhoc and vc
    ALLOCATE( rhoc(nrxxs,jblock), vc(nrxxs,jblock) )
#if defined(__USE_MANY_FFT)
    prhoc(1:nrxxs*jblock) => rhoc(:,:)
    pvc(1:nrxxs*jblock) => vc(:,:)
#endif
    !
    DO ii = 1, nibands(my_egrp_id+1)
       !
       ibnd = ibands(ii,my_egrp_id+1)
       !
       IF (ibnd==0 .OR. ibnd>m) CYCLE
       IF (okvan) deexx(:,ii) = 0._DP
       !
       IF (noncolin) THEN
          temppsic_nc(:,:,ii) = 0._DP
       ELSE
!$omp parallel do  default(shared), private(ir) firstprivate(nrxxs)
          DO ir = 1, nrxxs
             temppsic(ir,ii) = 0._DP
          ENDDO
       ENDIF
       !
       IF (noncolin) THEN
          !
!$omp parallel do  default(shared), private(ig)
          DO ig = 1, n
             temppsic_nc(dfftt%nl(igk_exx(ig,current_k)),1,ii) = psi(ig,ii)
             temppsic_nc(dfftt%nl(igk_exx(ig,current_k)),2,ii) = psi(npwx+ig,ii)
          ENDDO
!$omp end parallel do
          !
          CALL invfft( 'Wave', temppsic_nc(:,1,ii), dfftt )
          CALL invfft( 'Wave', temppsic_nc(:,2,ii), dfftt )
          !
       ELSE
          !
!$omp parallel do  default(shared), private(ig)
          DO ig = 1, n
             temppsic( dfftt%nl(igk_exx(ig,current_k)), ii ) = psi(ig,ii)
          ENDDO
!$omp end parallel do
          !
          CALL invfft( 'Wave', temppsic(:,ii), dfftt )
          !
       ENDIF
       !
       IF (noncolin) THEN
!$omp parallel do default(shared) firstprivate(nrxxs) private(ir)
          DO ir = 1, nrxxs
             result_nc(ir,1,ii) = 0.0_DP
             result_nc(ir,2,ii) = 0.0_DP
          ENDDO
       ELSE
!$omp parallel do default(shared) firstprivate(nrxxs) private(ir)
          DO ir = 1, nrxxs
             result(ir,ii) = 0.0_DP
          ENDDO
       ENDIF
       !
    ENDDO
    !
    !precompute these guys
    omega_inv = 1.0 / omega
    nqs_inv = 1.0 / nqs
    !
    !------------------------------------------------------------------------!
    ! Beginning of main loop
    !------------------------------------------------------------------------!
    DO iq = 1, nqs
       !
       ikq = index_xkq(current_ik,iq)
       ik  = index_xk(ikq)
       xkq = xkq_collect(:,ikq)
       !
       ! calculate the 1/|r-r'| (actually, k+q+g) factor and place it in fac
       CALL g2_convolution_all( dfftt%ngm, gt, xkp, xkq, iq, current_k )
       !
! JRD - below not threaded
       facb = 0D0
       DO ig = 1, dfftt%ngm
          facb(dfftt%nl(ig)) = coulomb_fac(ig,iq,current_k)
       ENDDO
       !
       IF ( okvan .AND..NOT.tqr ) CALL qvan_init( dfftt%ngm, xkq, xkp )
       !
       DO iegrp = 1, negrp
          !
          ! compute the id of group whose data is currently worked on
          wegrp = MOD(iegrp+my_egrp_id-1, negrp)+1
          njt = (all_end(wegrp)-all_start(wegrp)+jblock)/jblock
          !
          DO ijt = 1, njt
             !
             jblock_start = (ijt - 1) * jblock + all_start(wegrp)
             jblock_end = MIN(jblock_start+jblock-1,all_end(wegrp))
             !
             DO ii = 1, nibands(my_egrp_id+1)
                !
                ibnd = ibands(ii,my_egrp_id+1)
                !
                IF (ibnd==0 .OR. ibnd>m) CYCLE
                !
                !determine which j-bands to calculate
                jstart = 0
                jend = 0
                !
                DO ipair = 1, max_pairs
                   IF (egrp_pairs(1,ipair,my_egrp_id+1) == ibnd) THEN
                      IF (jstart == 0) THEN
                         jstart = egrp_pairs(2,ipair,my_egrp_id+1)
                         jend = jstart
                      ELSE
                         jend = egrp_pairs(2,ipair,my_egrp_id+1)
                      ENDIF
                   ENDIF
                ENDDO
                !
                jstart = MAX( jstart, jblock_start )
                jend = MIN( jend, jblock_end )
                !
                !how many iters
                jcount = jend-jstart+1
                IF (jcount <= 0) CYCLE
                !
                !----------------------------------------------------------------------!
                !INNER LOOP START
                !----------------------------------------------------------------------!
                !
                nblock = 2048
                nrt = (nrxxs+nblock-1)/nblock
                !
!$omp parallel do collapse(2) private(ir_start,ir_end)
                DO irt = 1, nrt
                   DO jbnd = jstart, jend
                      ir_start = (irt - 1) * nblock + 1
                      ir_end = MIN(ir_start+nblock-1, nrxxs)
                      IF (noncolin) THEN
                         DO ir = ir_start, ir_end
                            rhoc(ir,jbnd-jstart+1) = ( CONJG(exxbuff(ir,jbnd-all_start(wegrp)+ &
                                                       iexx_start,ikq))*temppsic_nc(ir,1,ii) + &
                                                       CONJG(exxbuff(nrxxs+ir,jbnd-all_start(wegrp)+ &
                                                       iexx_start,ikq))*temppsic_nc(ir,2,ii) )/omega
                         ENDDO
                      ELSE
!DIR$ vector nontemporal (rhoc)
                         DO ir = ir_start, ir_end
                            rhoc(ir,jbnd-jstart+1) = CONJG(exxbuff(ir,jbnd-all_start(wegrp)+ &
                                                     iexx_start,ikq))*temppsic(ir,ii)*omega_inv
                         ENDDO
                      ENDIF
                   ENDDO
                ENDDO
!$omp end parallel do
                !
                !   ... add augmentation in REAL space HERE
                IF (okvan .AND. tqr) THEN ! augment the "charge" in real space
                   DO jbnd = jstart, jend
                      CALL addusxx_r(rhoc(:,jbnd-jstart+1), becxx(ikq)%k(:,jbnd), becpsi%k(:,ibnd))
                   ENDDO
                ENDIF
                !
                !   ... brings it to G-space
#if defined(__USE_MANY_FFT)
                CALL fwfft( 'Rho', prhoc, dfftt, howmany=jcount )
#else
                DO jbnd=jstart, jend
                   CALL fwfft( 'Rho', rhoc(:,jbnd-jstart+1), dfftt )
                ENDDO
#endif
                !
                !   ... add augmentation in G space HERE
                IF (okvan .AND. .NOT. tqr) THEN
                   DO jbnd = jstart, jend
                      CALL addusxx_g( dfftt, rhoc(:,jbnd-jstart+1), xkq, xkp, &
                      'c', becphi_c=becxx(ikq)%k(:,jbnd),becpsi_c=becpsi%k(:,ibnd) )
                   ENDDO
                ENDIF
                !   ... charge done
                !
!call start_collection()
!$omp parallel do collapse(2) private(ir_start,ir_end)
                DO irt = 1, nrt
                   DO jbnd = jstart, jend
                      ir_start = (irt - 1) * nblock + 1
                      ir_end = MIN(ir_start+nblock-1,nrxxs)
!DIR$ vector nontemporal (vc)
                      DO ir = ir_start, ir_end
                         vc(ir,jbnd-jstart+1) = facb(ir) * rhoc(ir,jbnd-jstart+1)*&
                                                x_occupation(jbnd,ik) * nqs_inv
                      ENDDO
                   ENDDO
                ENDDO
!$omp end parallel do
!call stop_collection()
                !
                ! Add ultrasoft contribution (RECIPROCAL SPACE)
                ! compute alpha_I,j,k+q = \sum_J \int <beta_J|phi_j,k+q> V_i,j,k,q Q_I,J(r) d3r
                IF (okvan .AND. .NOT. tqr) THEN
                   DO jbnd=jstart, jend
                      CALL newdxx_g( dfftt, vc(:,jbnd-jstart+1), xkq, xkp, 'c',&
                                     deexx(:,ii), becphi_c=becxx(ikq)%k(:,jbnd) )
                   ENDDO
                ENDIF
                !
                !brings back v in real space
#if defined(__USE_MANY_FFT)
                !fft many
                CALL invfft( 'Rho', pvc, dfftt, howmany=jcount )
#else
                DO jbnd = jstart, jend
                   CALL invfft( 'Rho', vc(:,jbnd-jstart+1), dfftt )
                ENDDO
#endif
                !
                ! Add ultrasoft contribution (REAL SPACE)
                IF (okvan .AND. tqr) THEN
                   DO jbnd = jstart, jend
                      CALL newdxx_r( dfftt, vc(:,jbnd-jstart+1), becxx(ikq)%k(:,jbnd),deexx(:,ii) )
                   ENDDO
                ENDIF
                !
                ! ... Add PAW one-center contribution
                IF (okpaw) THEN
                   DO jbnd = jstart, jend
                      CALL PAW_newdxx( x_occupation(jbnd,ik)/nqs, becxx(ikq)%k(:,jbnd), &
                                       becpsi%k(:,ibnd), deexx(:,ii) )
                   ENDDO
                ENDIF
                !
                ! ... accumulates over bands and k points
                !
!call start_collection()
!$omp parallel do private(ir_start,ir_end)
                DO irt = 1, nrt
                   DO jbnd = jstart, jend
                      ir_start = (irt - 1) * nblock + 1
                      ir_end = MIN(ir_start+nblock-1, nrxxs)
                      IF (noncolin) THEN
                         DO ir = ir_start, ir_end
                            result_nc(ir,1,ii) = result_nc(ir,1,ii) + vc(ir,jbnd-jstart+1) * &
                                                 exxbuff(ir,jbnd-all_start(wegrp)+iexx_start,ikq)
                            result_nc(ir,2,ii) = result_nc(ir,2,ii) + vc(ir,jbnd-jstart+1) * &
                                                 exxbuff(ir+nrxxs,jbnd-all_start(wegrp)+iexx_start,ikq)
                         ENDDO
                      ELSE
!!dir$ vector nontemporal (result)
                         DO ir = ir_start, ir_end
                            result(ir,ii) = result(ir,ii) + vc(ir,jbnd-jstart+1)* &
                                            exxbuff(ir,jbnd-all_start(wegrp)+iexx_start,ikq)
                         ENDDO
                      ENDIF
                   ENDDO
                ENDDO
!$omp end parallel do
!call stop_collection()
                !
                !----------------------------------------------------------------------!
                !INNER LOOP END
                !----------------------------------------------------------------------!
                !
             ENDDO !I-LOOP
          ENDDO !IJT
          !
          ! get the next nbnd/negrp data
          IF (negrp > 1) CALL mp_circular_shift_left( exxbuff(:,:,ikq), me_egrp, inter_egrp_comm )
          !
       ENDDO !iegrp
       !
       IF ( okvan .AND..NOT.tqr ) CALL qvan_clean()
       !
    ENDDO
    !
    !
    !
    DO ii = 1, nibands(my_egrp_id+1)
       !
       ibnd = ibands(ii,my_egrp_id+1)
       !
       IF (ibnd==0 .OR. ibnd>m) CYCLE
       !
       IF (okvan) THEN
          CALL mp_sum( deexx(:,ii), intra_egrp_comm )
       ENDIF
       !
       IF (noncolin) THEN
          !brings back result in G-space
          CALL fwfft( 'Wave', result_nc(:,1,ii), dfftt )
          CALL fwfft( 'Wave', result_nc(:,2,ii), dfftt )
          !
          DO ig = 1, n
             big_result(ig,ibnd) = big_result(ig,ibnd) - exxalfa* &
                                   result_nc(dfftt%nl(igk_exx(ig,current_k)),1,ii)
             big_result(n+ig,ibnd) = big_result(n+ig,ibnd) - exxalfa* &
                                     result_nc(dfftt%nl(igk_exx(ig,current_k)),2,ii)
          ENDDO
       ELSE
          !
          CALL fwfft( 'Wave', result(:,ii), dfftt )
          !
          DO ig = 1, n
             big_result(ig,ibnd) = big_result(ig,ibnd) - exxalfa* &
                                   result(dfftt%nl(igk_exx(ig,current_k)),ii)
          ENDDO
       ENDIF
       !
       ! add non-local \sum_I |beta_I> \alpha_Ii (the sum on i is outside)
       IF (okvan) CALL add_nlxx_pot( lda, big_result(:,ibnd), xkp, n, igk_exx(:,current_k), &
                                    deexx(:,ii), eps_occ, exxalfa )
       !
    ENDDO
    !
    !deallocate temporary arrays
    DEALLOCATE( rhoc, vc )
    !
    !sum result
    CALL result_sum( n*npol, m, big_result )
    !
    IF (iexx_istart(my_egrp_id+1) > 0) THEN
       !
       IF (negrp == 1) THEN
          ending_im = m
       ELSE
          ending_im = iexx_iend(my_egrp_id+1) - iexx_istart(my_egrp_id+1) + 1
       ENDIF
       !
       IF (noncolin) THEN
          DO im = 1, ending_im
!$omp parallel do default(shared), private(ig) firstprivate(im,n)
             DO ig = 1, n
                hpsi(ig,im) = hpsi(ig,im) + big_result(ig,im+iexx_istart(my_egrp_id+1)-1)
             ENDDO
!$omp end parallel do
!$omp parallel do default(shared), private(ig) firstprivate(im,n)
             DO ig = 1, n
                hpsi(lda+ig,im) = hpsi(lda+ig,im) + big_result(n+ig,im+iexx_istart(my_egrp_id+1)-1)
             ENDDO
!$omp end parallel do
          ENDDO
       ELSE
          DO im = 1, ending_im
!$omp parallel do default(shared), private(ig) firstprivate(im,n)
             DO ig = 1, n
                hpsi(ig,im) = hpsi(ig,im) + big_result(ig,im+iexx_istart(my_egrp_id+1)-1)
             ENDDO
!$omp end parallel do
          ENDDO
       ENDIF
    ENDIF
    !
    IF (noncolin) THEN
       DEALLOCATE( temppsic_nc, result_nc )
    ELSE
       DEALLOCATE( temppsic, result )
    ENDIF
    !
    DEALLOCATE( big_result )
    DEALLOCATE( fac, facb )
    IF (okvan) DEALLOCATE( deexx )
    !
  END SUBROUTINE vexx_bp_k
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vexx_bp_k_gpu(lda, n, m, psi, hpsi, becpsi)
  !-----------------------------------------------------------------------
    !
    ! ... generic, k-point version of vexx
    !
    USE constants,      ONLY : fpi, e2, pi
    USE cell_base,      ONLY : omega
    USE gvect,          ONLY : ngm, g
    USE wvfct,          ONLY : npwx, current_k, nbnd
    USE klist,          ONLY : xk, nks, nkstot
    USE fft_interfaces, ONLY : fwfft, invfft
    USE becmod,         ONLY : bec_type
    USE mp_exx,         ONLY : inter_egrp_comm, my_egrp_id, negrp, &
                               intra_egrp_comm, me_egrp, &
                               max_pairs, egrp_pairs, ibands, nibands, &
                               max_ibands, iexx_istart, iexx_iend, &
                               all_start, all_end, iexx_start, jblock
    USE mp,             ONLY : mp_sum, mp_barrier, mp_circular_shift_left
    USE uspp,           ONLY : nkb, okvan
    USE paw_variables,  ONLY : okpaw
    USE us_exx,         ONLY : bexg_merge, becxx, addusxx_g, addusxx_r, &
                               newdxx_g, newdxx_r, add_nlxx_pot, &
                               qvan_init, qvan_clean
    USE paw_exx,        ONLY : PAW_newdxx
    USE exx_base,       ONLY : nqs, xkq_collect, index_xkq, index_xk
    USE exx_bp_utils,   ONLY : result_sum, igk_exx, igk_exx_d
    !CUDA stuff
    USE mp_exx,         ONLY : iexx_istart_d
    USE io_global,      ONLY : stdout
    !
    !
    IMPLICIT NONE
    !
    INTEGER                  :: lda, n, m
    COMPLEX(DP)              :: psi(lda*npol,max_ibands)
    COMPLEX(DP)              :: hpsi(lda*npol,max_ibands)
#if defined(__CUDA)
    attributes(DEVICE) :: psi_d, hpsi_d
#endif
    TYPE(bec_type), OPTIONAL :: becpsi ! or call a calbec(...psi) instead
    !
    ! local variables
    COMPLEX(DP),ALLOCATABLE :: psi_d(:,:)
    COMPLEX(DP),ALLOCATABLE :: hpsi_d(:,:)
    COMPLEX(DP),ALLOCATABLE :: temppsic_d(:,:)
    COMPLEX(DP),ALLOCATABLE :: temppsic_nc_d(:,:,:)
    COMPLEX(DP),ALLOCATABLE :: result_d(:,:), result_nc_d(:,:,:)
#if defined(__CUDA)
    attributes(DEVICE) :: temppsic_d, temppsic_nc_d, result_d, result_nc_d
#endif
    INTEGER          :: request_send, request_recv
    !
    COMPLEX(DP),ALLOCATABLE :: deexx(:,:)
    COMPLEX(DP),ALLOCATABLE,TARGET :: rhoc(:,:), vc(:,:)
    COMPLEX(DP),ALLOCATABLE,TARGET :: rhoc_d(:,:), vc_d(:,:)
    COMPLEX(DP),POINTER :: prhoc_d(:), pvc_d(:)
#if defined(__CUDA)
    attributes(DEVICE) :: rhoc_d, vc_d, prhoc_d, pvc_d
#endif
    REAL(DP), ALLOCATABLE :: fac(:), facb(:)
    REAL(DP), ALLOCATABLE :: facb_d(:)
#if defined(__CUDA)
    attributes(DEVICE) :: facb_d
#endif
    INTEGER  :: ibnd, ik, im , ikq, iq, ipol
    INTEGER  :: ir, ig, ir_start, ir_end
    INTEGER  :: irt, nrt, nblock
    INTEGER  :: current_ik
    INTEGER  :: ibnd_loop_start
    INTEGER  :: nrxxs
    REAL(DP) :: x1, x2, xkp(3), omega_inv, nqs_inv
    REAL(DP) :: xkq(3)
    INTEGER, EXTERNAL :: global_kpoint_index
    DOUBLE PRECISION :: max, tempx
    COMPLEX(DP), ALLOCATABLE :: big_result(:,:)
    COMPLEX(DP), ALLOCATABLE :: big_result_d(:,:)
#if defined(__CUDA)
    attributes(DEVICE) :: big_result_d
#endif
    INTEGER :: ir_out, ipair, jbnd
    INTEGER :: ii, jstart, jend, jcount, jind, jcurr
    INTEGER :: ialloc, ending_im
    INTEGER :: ijt, njt, jblock_start, jblock_end
    INTEGER :: iegrp, wegrp
    INTEGER :: all_start_tmp
    !hack around PGI bug
    INTEGER, POINTER :: dfftt__nl(:)
#if defined(__CUDA)
    attributes(DEVICE) :: dfftt__nl
#endif
    !
    dfftt__nl=>dfftt%nl_d
    !
    CALL start_clock( 'vexx_k_setup' )
    IF (negrp > 1) CALL exxbuff_sync_host()   ! the band-group shift below rotates the host array

    ialloc = nibands(my_egrp_id+1)
    !
    ALLOCATE( fac(dfftt%ngm) )
    nrxxs= dfftt%nnr
    ALLOCATE( facb(nrxxs) )

    ALLOCATE( psi_d, source=psi )
    ALLOCATE( hpsi_d, source=hpsi )
    ALLOCATE( facb_d(nrxxs) )

    IF (noncolin) THEN
       ALLOCATE( result_nc_d(nrxxs,npol,ialloc) )

       !temppsic_d knows where it is
       ALLOCATE( temppsic_nc_d(nrxxs,npol,ialloc) )
    ELSE
       ALLOCATE( result_d(nrxxs,ialloc) )

       !temppsic_d knows
       ALLOCATE( temppsic_d(nrxxs,ialloc) )
    ENDIF
    !
    IF(okvan) ALLOCATE(deexx(nkb,ialloc))
    !
    current_ik = global_kpoint_index ( nkstot, current_k )
    xkp = xk(:,current_k)
    !
    allocate(big_result(n*npol,m))
    big_result = 0.0_DP
    allocate(big_result_d(n*npol,m))
    big_result_d = 0.0_DP
    !
    !allocate arrays for rhoc and vc
    ALLOCATE(rhoc_d(nrxxs,jblock), vc_d(nrxxs,jblock))
    ALLOCATE(rhoc(nrxxs,jblock), vc(nrxxs,jblock))
    
    !
    
    DO ii=1, nibands(my_egrp_id+1)
       !
       ibnd = ibands(ii,my_egrp_id+1)
       !
       IF (ibnd.eq.0.or.ibnd.gt.m) CYCLE
       !
       IF(okvan) deexx(:,ii) = 0._DP
       !
       IF (noncolin) THEN
          temppsic_nc_d(:,:,ii) = 0._DP
       ELSE
          temppsic_d(:,ii) = 0._DP
       END IF
       !
       IF (noncolin) THEN
          !$cuf kernel do (1)
          DO ig = 1, n
             temppsic_nc_d(dfftt__nl(igk_exx_d(ig,current_k)),1,ii) = psi_d(ig,ii)
             temppsic_nc_d(dfftt__nl(igk_exx_d(ig,current_k)),2,ii) = psi_d(npwx+ig,ii)
          ENDDO
          CALL invfft ('Wave', temppsic_nc_d(:,1,ii), dfftt)
          CALL invfft ('Wave', temppsic_nc_d(:,2,ii), dfftt)
       ELSE
          !$cuf kernel do (1)
          DO ig = 1, n
             temppsic_d( dfftt__nl(igk_exx_d(ig,current_k)), ii ) = psi_d(ig,ii)
          ENDDO
          CALL invfft ('Wave', temppsic_d(:,ii), dfftt)
       END IF
    END DO

    IF (noncolin) THEN
       result_nc_d = 0.0_DP
    ELSE
       result_d = 0.0_DP
    ENDIF

    ! no longer need psi_d
    DEALLOCATE(psi_d)

    !
    !precompute these guys
    omega_inv = 1.0 / omega
    nqs_inv = 1.0 / nqs
    !
    CALL stop_clock( 'vexx_k_setup' )
    CALL start_clock( 'vexx_k_main' )
    !------------------------------------------------------------------------!
    ! Beginning of main loop
    !------------------------------------------------------------------------!
    vexxmain: DO iq=1, nqs
       !
       ikq  = index_xkq(current_ik,iq)
       ik   = index_xk(ikq)
       xkq  = xkq_collect(:,ikq)
       !
       ! calculate the 1/|r-r'| (actually, k+q+g) factor and place it in fac
       CALL g2_convolution_all(dfftt%ngm, gt, xkp, xkq, iq, current_k)
       !
! JRD - below not threaded
       facb = 0D0
       DO ig = 1, dfftt%ngm
          facb(dfftt%nl(ig)) = coulomb_fac(ig,iq,current_k)
       ENDDO
       facb_d = facb
       !
       IF ( okvan .and..not.tqr ) CALL qvan_init (dfftt%ngm, xkq, xkp)
       !
       DO iegrp=1, negrp
          !
          ! compute the id of group whose data is currently worked on
          wegrp = MOD(iegrp+my_egrp_id-1, negrp)+1
          njt = (all_end(wegrp)-all_start(wegrp)+jblock)/jblock
          !
          DO ijt=1, njt
             !
             jblock_start = (ijt - 1) * jblock + all_start(wegrp)
             jblock_end = min(jblock_start+jblock-1,all_end(wegrp))
             !
             DO ii=1, nibands(my_egrp_id+1)
                !
                ibnd = ibands(ii,my_egrp_id+1)
                !
                IF (ibnd.eq.0.or.ibnd.gt.m) CYCLE
                !
                !determine which j-bands to calculate
                jstart = 0
                jend = 0
                DO ipair=1, max_pairs
                   IF(egrp_pairs(1,ipair,my_egrp_id+1).eq.ibnd)THEN
                      IF(jstart.eq.0)THEN
                         jstart = egrp_pairs(2,ipair,my_egrp_id+1)
                         jend = jstart
                      ELSE
                         jend = egrp_pairs(2,ipair,my_egrp_id+1)
                      END IF
                   END IF
                END DO
                !
                jstart = max(jstart,jblock_start)
                jend = min(jend,jblock_end)
                !
                !how many iters
                jcount=jend-jstart+1
                if(jcount<=0) cycle
                !
                !----------------------------------------------------------------------!
                !INNER LOOP START
                !----------------------------------------------------------------------!
                !
                nblock=2048
                nrt = (nrxxs+nblock-1)/nblock
                !
associate(rhoc=>rhoc_d, exxbuff=>exxbuff_d)
                all_start_tmp=all_start(wegrp)
                !$cuf kernel do (2)
                DO jbnd=jstart, jend
                   DO ir = 1, nrxxs

                     IF (noncolin) THEN
                       rhoc(ir,jbnd-jstart+1) = &
                       (conjg(exxbuff(ir,jbnd-all_start_tmp+iexx_start,ikq))*temppsic_nc_d(ir,1,ii) +&
                       conjg(exxbuff(nrxxs+ir,jbnd-all_start_tmp+iexx_start,ikq))*temppsic_nc_d(ir,2,ii)) * omega_inv
                     ELSE

                       rhoc(ir,jbnd-jstart+1) = &
                       conjg(exxbuff(ir,jbnd-all_start_tmp+iexx_start,ikq))*temppsic_d(ir,ii)* omega_inv
                     ENDIF

                   ENDDO
                ENDDO
end associate
                !
                !   >>>> add augmentation in REAL space HERE
                IF(okvan .and. tqr) THEN ! augment the "charge" in real space
                   rhoc = rhoc_d       ! the density lives on the device; addusxx_r reads the host copy
                   DO jbnd=jstart, jend
                      CALL addusxx_r(rhoc(:,jbnd-jstart+1), becxx(ikq)%k(:,jbnd), becpsi%k(:,ibnd))
                   ENDDO
                   rhoc_d = rhoc       ! and the augmented density goes back before the FFT
                ENDIF
                !
                !   >>>> brings it to G-space
                !
                DO jbnd=jstart, jend, many_fft
                  jcurr = min(many_fft, jend-jbnd+1)
                  prhoc_d(1:nrxxs*jcurr) => rhoc_d(:,jbnd-jstart+1:jbnd-jstart+jcurr)
                  CALL fwfft ('Rho', prhoc_d, dfftt, howmany=jcurr)
                ENDDO
                !
                !   >>>> add augmentation in G space HERE
                IF(okvan .and. .not. tqr) THEN
                   rhoc = rhoc_d
                   DO jbnd=jstart, jend
                      CALL addusxx_g(dfftt, rhoc(:,jbnd-jstart+1), xkq, xkp, &
                      'c', becphi_c=becxx(ikq)%k(:,jbnd),becpsi_c=becpsi%k(:,ibnd))
                   ENDDO
                   rhoc_d = rhoc
                ENDIF
                !   >>>> charge done
                !
associate(vc=>vc_d, facb=>facb_d, rhoc=>rhoc_d, x_occupation=>x_occupation_d)
                !$cuf kernel do (2)
                DO jbnd=jstart, jend
                   DO ir = 1, nrxxs
                         vc(ir,jbnd-jstart+1) = facb(ir) * rhoc(ir,jbnd-jstart+1)*&
                                                x_occupation(jbnd,ik) * nqs_inv
                   ENDDO
                ENDDO
end associate
                !
                ! Add ultrasoft contribution (RECIPROCAL SPACE)
                ! compute alpha_I,j,k+q = \sum_J \int <beta_J|phi_j,k+q> V_i,j,k,q Q_I,J(r) d3r
                IF(okvan .and. .not. tqr) THEN
                   vc = vc_d
                   DO jbnd=jstart, jend
                      CALL newdxx_g(dfftt, vc(:,jbnd-jstart+1), xkq, xkp, 'c',&
                                    deexx(:,ii), becphi_c=becxx(ikq)%k(:,jbnd))
                   ENDDO
                   vc_d = vc
                ENDIF
                !
                !brings back v in real space
                DO jbnd=jstart, jend, many_fft
                  jcurr = min(many_fft, jend-jbnd+1)
                  pvc_d(1:nrxxs*jcurr) => vc_d(:,jbnd-jstart+1:jbnd-jstart+jcurr)
                  CALL invfft ('Rho', pvc_d, dfftt, howmany=jcurr)
                ENDDO
                !
                ! Add ultrasoft contribution (REAL SPACE)
                IF(okvan .and. tqr) THEN
                   vc = vc_d
                   DO jbnd=jstart, jend
                      CALL newdxx_r(dfftt, vc(:,jbnd-jstart+1), becxx(ikq)%k(:,jbnd),deexx(:,ii))
                   ENDDO
                   vc_d = vc
                ENDIF
                !
                ! Add PAW one-center contribution
                IF(okpaw) THEN
                   vc = vc_d
                   DO jbnd=jstart, jend
                      CALL PAW_newdxx(x_occupation(jbnd,ik)/nqs, becxx(ikq)%k(:,jbnd), becpsi%k(:,ibnd), deexx(:,ii))
                   ENDDO
                   vc_d = vc
                ENDIF
                !
                !accumulates over bands and k points
                !

associate(exxbuff=>exxbuff_d, vc=>vc_d)
                all_start_tmp=all_start(wegrp)
                DO jbnd=jstart, jend
                   !$cuf kernel do (1)
                   DO ir = 1, nrxxs
                      IF (noncolin) THEN
                         result_nc_d(ir,1,ii) = result_nc_d(ir,1,ii) &
                              + vc(ir,jbnd-jstart+1) * exxbuff(ir,jbnd-all_start_tmp+iexx_start,ikq)
                         result_nc_d(ir,2,ii) = result_nc_d(ir,2,ii) &
                              + vc(ir,jbnd-jstart+1) * exxbuff(ir+nrxxs,jbnd-all_start_tmp+iexx_start,ikq)
                      ELSE
                         result_d(ir,ii) = result_d(ir,ii) &
                              + vc(ir,jbnd-jstart+1)*exxbuff(ir,jbnd-all_start_tmp+iexx_start,ikq)
                      ENDIF
                   ENDDO
                ENDDO
end associate
                !
                !----------------------------------------------------------------------!
                !INNER LOOP END
                !----------------------------------------------------------------------!
                !
             END DO !I-LOOP
          END DO !IJT
          !
          ! get the next nbnd/negrp data
          IF (negrp>1) THEN
             call mp_circular_shift_left( exxbuff(:,:,ikq), me_egrp, inter_egrp_comm )
             exxbuff_d(:,:,ikq) = exxbuff(:,:,ikq)
          ENDIF
          !
       END DO !iegrp
       !
       IF ( okvan .and..not.tqr ) CALL qvan_clean ()
    END DO vexxmain

!move this down to after the vexx_k_fin

    CALL stop_clock( 'vexx_k_main' )
    CALL start_clock( 'vexx_k_fin' )
    !
    !
    !
    DO ii=1, nibands(my_egrp_id+1)
       !
       ibnd = ibands(ii,my_egrp_id+1)
       !
       IF (ibnd.eq.0.or.ibnd.gt.m) CYCLE
       !
       IF(okvan) THEN
          CALL mp_sum(deexx(:,ii),intra_egrp_comm)
       ENDIF
       !
       !big_result_d=big_result !already initialized along with the big_result=1.0D0
       IF (noncolin) THEN
          !brings back result in G-space
          CALL fwfft ('Wave', result_nc_d(:,1,ii), dfftt)
          CALL fwfft ('Wave', result_nc_d(:,2,ii), dfftt)
          !$cuf kernel do (1)
          DO ig = 1, n
             big_result_d(ig,ibnd) = big_result_d(ig,ibnd) - exxalfa*result_nc_d(dfftt__nl(igk_exx_d(ig,current_k)),1,ii)
             big_result_d(n+ig,ibnd) = big_result_d(n+ig,ibnd) - exxalfa*result_nc_d(dfftt__nl(igk_exx_d(ig,current_k)),2,ii)
          ENDDO
       ELSE
          !
          CALL fwfft ('Wave', result_d(:,ii), dfftt)
          !$cuf kernel do (1)
          DO ig = 1, n
             big_result_d(ig,ibnd) = big_result_d(ig,ibnd) - exxalfa*result_d(dfftt__nl(igk_exx_d(ig,current_k)),ii)
          ENDDO
       ENDIF
       big_result(:,ibnd) = big_result_d(:,ibnd)

       IF(okvan) CALL add_nlxx_pot (lda, big_result(:,ibnd), xkp, n, igk_exx(:,current_k),&
            deexx(:,ii), eps_occ, exxalfa)
       !
    END DO

    ! add non-local \sum_I |beta_I> \alpha_Ii (the sum on i is outside)
    !deallocate temporary arrays
    DEALLOCATE(rhoc, vc)
    IF (noncolin) THEN
       DEALLOCATE( result_nc_d )
    ELSE
       DEALLOCATE( result_d )
    ENDIF

    !dealloc stuff
    DEALLOCATE(rhoc_d, vc_d)
    !
    !sum result
    CALL result_sum(n*npol, m, big_result)
    big_result_d = big_result
    IF (iexx_istart(my_egrp_id+1).gt.0) THEN
       IF (negrp == 1) then
          ending_im = m
       ELSE
          ending_im = iexx_iend(my_egrp_id+1) - iexx_istart(my_egrp_id+1) + 1
       END IF

       !iexx_istart_d=iexx_istart
       IF(noncolin) THEN
          !$cuf kernel do (2)
          DO im=1, ending_im
             DO ig = 1, n
                hpsi_d(ig,im) = hpsi_d(ig,im) + big_result_d(ig,im+iexx_istart_d(my_egrp_id+1)-1)
                hpsi_d(lda+ig,im) = hpsi_d(lda+ig,im) + big_result_d(n+ig,im+iexx_istart_d(my_egrp_id+1)-1)
             ENDDO
          END DO
       ELSE
          !$cuf kernel do (2)
          DO im=1, ending_im
             DO ig = 1, n
                hpsi_d(ig,im) = hpsi_d(ig,im) + big_result_d(ig,im+iexx_istart_d(my_egrp_id+1)-1)
             ENDDO
          ENDDO
       END IF
    END IF
    hpsi=hpsi_d

    !these need to be deallocated anyhow
    DEALLOCATE(big_result)

    DEALLOCATE(fac, facb )

    IF (noncolin) THEN
       DEALLOCATE(temppsic_nc_d)
    ELSE
       DEALLOCATE(temppsic_d)
    ENDIF

    IF(okvan) DEALLOCATE( deexx)

    DEALLOCATE(big_result_d)
    DEALLOCATE(facb_d)
    DEALLOCATE(hpsi_d)
    !
    CALL stop_clock( 'vexx_k_fin' )
    !
    !------------------------------------------------------------------------
  END SUBROUTINE vexx_bp_k_gpu
  !
  !-----------------------------------------------------------------------
  FUNCTION exxenergy_bp_gamma()
    !-----------------------------------------------------------------------
    !
    USE constants,               ONLY : fpi, e2, pi
    USE io_files,                ONLY : iunwfc_exx, nwordwfc
    USE buffers,                 ONLY : get_buffer
    USE cell_base,               ONLY : alat, omega, bg, at, tpiba
    USE symm_base,               ONLY : nsym, s
    USE gvect,                   ONLY : ngm, gstart, g
    USE wvfct,                   ONLY : nbnd, npwx, wg
    USE wavefunctions,           ONLY : evc
    USE klist,                   ONLY : xk, ngk, nks, nkstot
    USE lsda_mod,                ONLY : lsda, current_spin, isk
    USE mp_pools,                ONLY : inter_pool_comm
    USE mp_bands,                ONLY : intra_bgrp_comm
    USE mp_exx,                  ONLY : inter_egrp_comm, my_egrp_id, negrp, &
                                        intra_egrp_comm, me_egrp, &
                                        max_pairs, egrp_pairs, ibands, nibands, &
                                        iexx_istart, iexx_iend, &
                                        all_start, all_end, iexx_start, &
                                        init_index_over_band, jblock
    USE mp,                      ONLY : mp_sum, mp_circular_shift_left
    USE fft_interfaces,          ONLY : fwfft, invfft
    USE gvect,                   ONLY : ecutrho
    USE klist,                   ONLY : wk
    USE uspp,                    ONLY : okvan,nkb,vkb
    USE becmod,                  ONLY : bec_type, allocate_bec_type, &
                                        deallocate_bec_type, calbec
    USE paw_variables,           ONLY : okpaw
    USE paw_exx,                 ONLY : PAW_xx_energy
    USE us_exx,                  ONLY : bexg_merge, becxx, addusxx_g, &
                                        addusxx_r, qvan_init, qvan_clean
    USE exx_base,                ONLY : nqs, xkq_collect, index_xkq, index_xk
    USE exx_bp_utils,            ONLY : igk_exx, change_data_structure, &
                                        transform_evc_to_exx, nwordwfc_exx, &
                                        evc_exx
    USE uspp_init,               ONLY : init_us_2
    !
    IMPLICIT NONE
    !
    REAL(DP)   :: exxenergy_bp_gamma
    !
    ! ... local variables
    !
    REAL(DP) :: energy
    COMPLEX(DP), ALLOCATABLE :: temppsic(:)
    COMPLEX(DP), ALLOCATABLE :: rhoc(:)
    REAL(DP),    ALLOCATABLE :: fac(:)
    COMPLEX(DP), ALLOCATABLE :: vkb_exx(:,:)
    INTEGER  :: jbnd, ibnd, ik, ikk, ig, ikq, iq, ir
    INTEGER  :: nrxxs, current_ik, ibnd_loop_start
    REAL(DP) :: x1, x2
    REAL(DP) :: xkq(3), xkp(3), vc
    INTEGER, EXTERNAL :: global_kpoint_index
    !
    TYPE(bec_type) :: becpsi
    COMPLEX(DP), ALLOCATABLE :: psi_t(:), prod_tot(:)
    REAL(DP),ALLOCATABLE :: temppsic_dble (:)
    REAL(DP),ALLOCATABLE :: temppsic_aimag(:)
    INTEGER :: npw
    INTEGER :: istart, iend, ipair, ii, ialloc
    INTEGER :: ijt, njt, jblock_start, jblock_end
    INTEGER :: exxbuff_index
    INTEGER :: calbec_start, calbec_end
    INTEGER :: intra_bgrp_comm_
    INTEGER :: iegrp, wegrp
    INTEGER :: ibnd_start
    !
    CALL init_index_over_band( inter_egrp_comm, nbnd, nbnd )
    !
    CALL transform_evc_to_exx( 0 )
    !
    ialloc = nibands(my_egrp_id+1)
    !
    nrxxs = dfftt%nnr
    ALLOCATE( fac(dfftt%ngm) )
    !
    ALLOCATE( temppsic(nrxxs), temppsic_DBLE(nrxxs), temppsic_aimag(nrxxs) )
    ALLOCATE( rhoc(nrxxs) )
    ALLOCATE( vkb_exx(npwx,nkb) )
    !
    energy = 0.0_DP
    !
    CALL allocate_bec_type( nkb, nbnd, becpsi )
    !
    IKK_LOOP : &
    DO ikk = 1, nks
       current_ik = global_kpoint_index( nkstot, ikk )
       xkp = xk(:,ikk)
       !
       IF ( lsda ) current_spin = isk(ikk)
       npw = ngk (ikk)
       IF ( nks > 1 ) CALL get_buffer( evc_exx, nwordwfc_exx, iunwfc_exx, ikk )
       !
       ! ... prepare the |beta> function at k+q
       CALL init_us_2( npw, igk_exx(:,ikk), xkp, vkb_exx )
       !
       ! ... compute <beta_I|psi_j> at this k+q point, for all band and all projectors
       calbec_start = ibands(1,my_egrp_id+1)
       calbec_end = ibands(nibands(my_egrp_id+1),my_egrp_id+1)
       !
       intra_bgrp_comm_ = intra_bgrp_comm
       intra_bgrp_comm = intra_egrp_comm
       !
       CALL calbec( npw, vkb_exx, evc_exx, becpsi, nibands(my_egrp_id+1) )
       !
       intra_bgrp_comm = intra_bgrp_comm_
       !
       IQ_LOOP : &
       DO iq = 1,nqs
          !
          ikq  = index_xkq(current_ik,iq)
          ik   = index_xk(ikq)
          !
          xkq = xkq_collect(:,ikq)
          !
          CALL g2_convolution_all( dfftt%ngm, gt, xkp, xkq, iq, current_ik )
          !
          fac = coulomb_fac(:,iq,current_ik)
          fac(gstart_t:) = 2 * coulomb_fac(gstart_t:,iq,current_ik)
          !
          IF ( okvan .AND. .NOT.tqr ) CALL qvan_init( dfftt%ngm, xkq, xkp )
          !
          DO iegrp = 1, negrp
             !
             ! ... compute the id of group whose data is currently worked on
             wegrp = MOD(iegrp+my_egrp_id-1, negrp)+1
             !
             jblock_start = all_start(wegrp)
             jblock_end   = all_end(wegrp)
             !
             JBND_LOOP : &
             DO ii = 1, nibands(my_egrp_id+1)
                !
                jbnd = ibands(ii,my_egrp_id+1)
                !
                IF (jbnd==0 .OR. jbnd>nbnd) CYCLE
                !
                IF ( MOD(ii,2)==1 ) THEN
                   !
                   temppsic = (0._DP,0._DP)
                   !
                   IF ( (ii+1) <= nibands(my_egrp_id+1) ) THEN
                      ! deal with double bands
!$omp parallel do  default(shared), private(ig)
                      DO ig = 1, npwt
                         temppsic( dfftt%nl(ig) )  = &
                              evc_exx(ig,ii) + (0._DP,1._DP) * evc_exx(ig,ii+1)
                         temppsic( dfftt%nlm(ig) ) = &
                              CONJG(evc_exx(ig,ii) - (0._DP,1._DP) * evc_exx(ig,ii+1))
                      ENDDO
!$omp end parallel do
                   ENDIF
                   !
                   IF (ii == nibands(my_egrp_id+1)) THEN
                      ! deal with a single last band
!$omp parallel do  default(shared), private(ig)
                      DO ig = 1, npwt
                         temppsic( dfftt%nl(ig) ) = evc_exx(ig,ii)
                         temppsic( dfftt%nlm(ig) ) = CONJG(evc_exx(ig,ii))
                      ENDDO
!$omp end parallel do
                   ENDIF
                   !
                   CALL invfft( 'Wave', temppsic, dfftt )
!$omp parallel do default(shared), private(ir)
                   DO ir = 1, nrxxs
                      temppsic_DBLE(ir) = DBLE( temppsic(ir) )
                      temppsic_aimag(ir) = AIMAG( temppsic(ir) )
                   ENDDO
!$omp end parallel do
                   !
                ENDIF
                !
                !determine which j-bands to calculate
                istart = 0
                iend = 0
                !
                DO ipair = 1, max_pairs
                   IF (egrp_pairs(1,ipair,my_egrp_id+1) == jbnd) THEN
                      IF (istart == 0) THEN
                         istart = egrp_pairs(2,ipair,my_egrp_id+1)
                         iend = istart
                      ELSE
                         iend = egrp_pairs(2,ipair,my_egrp_id+1)
                      ENDIF
                   ENDIF
                ENDDO
                !
                istart = MAX(istart,jblock_start)
                iend = MIN(iend,jblock_end)
                !
                IF (MOD(istart,2) == 0) THEN
                   ibnd_loop_start = istart-1
                ELSE
                   ibnd_loop_start = istart
                ENDIF
                !
                IBND_LOOP_GAM : &
                DO ibnd = ibnd_loop_start, iend, 2       !for each band of psi
                   !
                   exxbuff_index = (ibnd+1)/2-(all_start(wegrp)+1)/2+(iexx_start+1)/2
                   !
                   IF ( ibnd < istart ) THEN
                      x1 = 0.0_DP
                   ELSE
                      x1 = x_occupation(ibnd,ik)
                   ENDIF
                   !
                   IF ( ibnd < iend ) THEN
                      x2 = x_occupation(ibnd+1,ik)
                   ELSE
                      x2 = 0.0_DP
                   ENDIF
                   ! calculate rho in real space. Gamma tricks are used.
                   ! temppsic is real; tempphic contains band 1 in the real part,
                   ! band 2 in the imaginary part; the same applies to rhoc
                   !
                   IF ( MOD(ii,2) == 0 ) THEN
                      rhoc = 0.0_DP
!$omp parallel do default(shared), private(ir)
                      DO ir = 1, nrxxs
                         rhoc(ir) = exxbuff(ir,exxbuff_index,ikq) * temppsic_aimag(ir) / omega
                      ENDDO
!$omp end parallel do
                   ELSE
!$omp parallel do default(shared), private(ir)
                      DO ir = 1, nrxxs
                         rhoc(ir) = exxbuff(ir,exxbuff_index,ikq) * temppsic_DBLE(ir) / omega
                      ENDDO
!$omp end parallel do
                   ENDIF
                   !
                   IF (okvan .AND. tqr) THEN
                      IF (ibnd >= istart) &
                           CALL addusxx_r( rhoc,_CX(becxx(ikq)%r(:,ibnd)), &
                                          _CX(becpsi%r(:,jbnd)) )
                      IF (ibnd<iend) &
                           CALL addusxx_r(rhoc,_CY(becxx(ikq)%r(:,ibnd+1)), &
                                          _CX(becpsi%r(:,jbnd)))
                   ENDIF
                   !
                   ! bring rhoc to G-space
                   CALL fwfft( 'Rho', rhoc, dfftt )
                   !
                   IF (okvan .AND. .NOT.tqr) THEN
                      IF (ibnd >= istart) &
                           CALL addusxx_g( dfftt, rhoc, xkq, xkp, 'r', &
                           becphi_r=becxx(ikq)%r(:,ibnd), becpsi_r=becpsi%r(:,jbnd-calbec_start+1) )
                      IF (ibnd < iend) &
                           CALL addusxx_g( dfftt, rhoc, xkq, xkp, 'i', &
                           becphi_r=becxx(ikq)%r(:,ibnd+1), becpsi_r=becpsi%r(:,jbnd-calbec_start+1) )
                   ENDIF
                   !
                   vc = 0.0_DP
!$omp parallel do  default(shared), private(ig),  reduction(+:vc)
                   DO ig = 1, dfftt%ngm
                      !
                      ! The real part of rhoc contains the contribution from band ibnd
                      ! The imaginary part    contains the contribution from band ibnd+1
                      !
                      vc = vc + fac(ig) * ( x1 * &
                           ABS( rhoc(dfftt%nl(ig)) + CONJG(rhoc(dfftt%nlm(ig))) )**2 &
                                 +x2 * &
                           ABS( rhoc(dfftt%nl(ig)) - CONJG(rhoc(dfftt%nlm(ig))) )**2 )
                   ENDDO
!$omp end parallel do
                   !
                   vc = vc * omega * 0.25_DP / nqs
                   energy = energy - exxalfa * vc * wg(jbnd,ikk)
                   !
                   IF (okpaw) THEN
                      IF (ibnd >= ibnd_start) &
                           energy = energy + exxalfa*wg(jbnd,ikk)*&
                           x1 * PAW_xx_energy(_CX(becxx(ikq)%r(:,ibnd)),_CX(becpsi%r(:,jbnd)) )
                      IF (ibnd < ibnd_end) &
                           energy = energy + exxalfa*wg(jbnd,ikk)*&
                           x2 * PAW_xx_energy(_CX(becxx(ikq)%r(:,ibnd+1)), _CX(becpsi%r(:,jbnd)) )
                   ENDIF
                   !
                ENDDO &
                IBND_LOOP_GAM
             ENDDO &
             JBND_LOOP
             !
             ! get the next nbnd/negrp data
             IF (negrp > 1) CALL mp_circular_shift_left( exxbuff(:,:,ikq), me_egrp, inter_egrp_comm )
             !
          ENDDO ! iegrp
          IF ( okvan .AND. .NOT.tqr ) CALL qvan_clean( )
          !
       ENDDO &
       IQ_LOOP
    ENDDO &
    IKK_LOOP
    !
    DEALLOCATE( temppsic, temppsic_dble, temppsic_aimag )
    !
    DEALLOCATE( rhoc, fac )
    CALL deallocate_bec_type( becpsi )
    !
    CALL mp_sum( energy, inter_egrp_comm )
    CALL mp_sum( energy, intra_egrp_comm )
    CALL mp_sum( energy, inter_pool_comm )
    !
    exxenergy_bp_gamma = energy
    !
    CALL change_data_structure( .FALSE. )
    !
  END FUNCTION  exxenergy_bp_gamma
  !
  !----------------------------------------------------------------------
  SUBROUTINE compute_becpsi( npw_, igk_, q_, evc_exx, becpsi_k )
  !----------------------------------------------------------------------
  !! Calculates becpsi_k = <vkb|evc_exx> - FIXME: untested
  !
  USE kinds,         ONLY : DP
  USE wvfct,         ONLY : npwx, nbnd
  USE uspp,          ONLY : nkb
  USE uspp_param,    ONLY : lmaxkb
  USE becmod,        ONLY : calbec
  USE mp_exx,        ONLY : ibands, nibands, my_egrp_id
  USE uspp_init,     ONLY : init_us_2
  !
  IMPLICIT NONE
  !
  INTEGER, INTENT(IN) :: npw_
  !! number of PWs
  INTEGER, INTENT(IN) :: igk_(npw_)
  !! indices of G in the list of q+G vectors
  REAL(DP), INTENT(IN) :: q_(3)
  !! q vector (2pi/a units)
  COMPLEX(DP), INTENT(IN) :: evc_exx(npwx,nibands(my_egrp_id+1))
  !! wavefunctions from the PW set to exx
  COMPLEX(DP), INTENT(OUT) :: becpsi_k(nkb,nibands(my_egrp_id+1))
  !! <beta|psi> for k points
  !
  ! ... local variables
  !
  COMPLEX(DP), ALLOCATABLE :: vkb_(:,:) !beta functions (npw_ <= npwx)
  INTEGER :: istart, iend
  !
  IF (lmaxkb < 0) RETURN
  !
  istart = ibands(1,my_egrp_id+1)
  iend = ibands(nibands(my_egrp_id+1),my_egrp_id+1)
  !
  write(6,*) 'WARNING: compute_becpsi UNTESTED'
  ALLOCATE( vkb_(npwx,nkb) )
  !
  CALL init_us_2( npw_, igk_, q_, vkb_ )
  !
  CALL calbec( npw_, vkb_, evc_exx, becpsi_k, nibands(my_egrp_id+1) )
  !
  DEALLOCATE( vkb_ )
  !
  RETURN
  !
  END SUBROUTINE compute_becpsi
  !
  !-----------------------------------------------------------------------
  FUNCTION exxenergy_bp_k()
    !-----------------------------------------------------------------------
    !
    USE constants,               ONLY : fpi, e2, pi
    USE io_files,                ONLY : iunwfc_exx, nwordwfc
    USE buffers,                 ONLY : get_buffer
    USE cell_base,               ONLY : alat, omega, bg, at, tpiba
    USE symm_base,               ONLY : nsym, s
    USE gvect,                   ONLY : ngm, gstart, g
    USE wvfct,                   ONLY : nbnd, npwx, wg
    USE wavefunctions,           ONLY : evc
    USE klist,                   ONLY : xk, ngk, nks, nkstot
    USE lsda_mod,                ONLY : lsda, current_spin, isk
    USE mp_pools,                ONLY : inter_pool_comm
    USE mp_exx,                  ONLY : inter_egrp_comm, my_egrp_id, negrp, &
                                        intra_egrp_comm, me_egrp, &
                                        max_pairs, egrp_pairs, ibands, nibands, &
                                        iexx_istart, iexx_iend, &
                                        all_start, all_end, iexx_start, &
                                        init_index_over_band, jblock
    USE mp_bands,                ONLY : intra_bgrp_comm
    USE mp,                      ONLY : mp_sum, mp_circular_shift_left
    USE fft_interfaces,          ONLY : fwfft, invfft
    USE gvect,                   ONLY : ecutrho
    USE klist,                   ONLY : wk
    USE uspp,                    ONLY : okvan,nkb,vkb
    USE becmod,                  ONLY : bec_type, allocate_bec_type, &
                                        deallocate_bec_type, calbec
    USE paw_variables,           ONLY : okpaw
    USE paw_exx,                 ONLY : PAW_xx_energy
    USE us_exx,                  ONLY : bexg_merge, becxx, addusxx_g, &
                                        addusxx_r, qvan_init, qvan_clean
    USE exx_base,                ONLY : nqs, xkq_collect, index_xkq, index_xk
    USE exx_bp_utils,            ONLY : change_data_structure, &
                                        transform_evc_to_exx, nwordwfc_exx, &
                                        igk_exx, evc_exx
    !
    IMPLICIT NONE
    !
    REAL(DP)   :: exxenergy_bp_k
    !
    ! ... local variables
    !
    REAL(DP) :: energy
    COMPLEX(DP), ALLOCATABLE :: temppsic(:,:)
    COMPLEX(DP), ALLOCATABLE :: temppsic_nc(:,:,:)
    COMPLEX(DP), ALLOCATABLE,TARGET :: rhoc(:,:)
#if defined(__USE_MANY_FFT)
    COMPLEX(DP), POINTER :: prhoc(:)
#endif
    REAL(DP),    ALLOCATABLE :: fac(:)
    INTEGER  :: npw, jbnd, ibnd, ibnd_inner_start, ibnd_inner_end, ibnd_inner_count, &
                ik, ikk, ig, ikq, iq, ir
    INTEGER  :: h_ibnd, nrxxs, current_ik, ibnd_loop_start, nblock, nrt, irt, &
                ir_start, ir_end
    REAL(DP) :: x1, x2
    REAL(DP) :: xkq(3), xkp(3), vc, omega_inv
    INTEGER, EXTERNAL :: global_kpoint_index
    !
    TYPE(bec_type) :: becpsi
    COMPLEX(DP), ALLOCATABLE :: psi_t(:), prod_tot(:)
    INTEGER :: intra_bgrp_comm_
    INTEGER :: ii, ialloc, jstart, jend, ipair
    INTEGER :: ijt, njt, jblock_start, jblock_end
    INTEGER :: iegrp, wegrp
    !
    CALL init_index_over_band( inter_egrp_comm, nbnd, nbnd )
    !
    CALL transform_evc_to_exx( 0 )
    !
    ialloc = nibands(my_egrp_id+1)
    !
    nrxxs = dfftt%nnr
    ALLOCATE( fac(dfftt%ngm) )
    !
    IF (noncolin) THEN
       ALLOCATE( temppsic_nc(nrxxs,npol,ialloc) )
    ELSE
       ALLOCATE( temppsic(nrxxs,ialloc) )
    ENDIF
    !
    energy = 0.0_DP
    !
    CALL allocate_bec_type( nkb, nbnd, becpsi )
    !
    !precompute that stuff
    omega_inv = 1.0/omega
    !
    IKK_LOOP : &
    DO ikk = 1, nks
       !
       current_ik = global_kpoint_index ( nkstot, ikk )
       xkp = xk(:,ikk)
       !
       IF ( lsda ) current_spin = isk(ikk)
       npw = ngk(ikk)
       IF ( nks > 1 ) CALL get_buffer( evc_exx, nwordwfc_exx, iunwfc_exx, ikk )
       !
       ! compute <beta_I|psi_j> at this k+q point, for all band and all projectors
       intra_bgrp_comm_ = intra_bgrp_comm
       intra_bgrp_comm = intra_egrp_comm
       !
       IF (okvan .OR. okpaw) THEN
          !! FIXME: can be replaced by a call to init_us_2 + calbec
          CALL compute_becpsi( npw, igk_exx(:,ikk), xkp, evc_exx, &
                               becpsi%k(:,ibands(1,my_egrp_id+1)) )
       ENDIF
       !
       intra_bgrp_comm = intra_bgrp_comm_
       !
       ! ... precompute temppsic
       !
       IF (noncolin) THEN
          temppsic_nc = 0.0_DP
       ELSE
          temppsic = 0.0_DP
       ENDIF
       !
       DO ii = 1, nibands(my_egrp_id+1)
          !
          jbnd = ibands(ii,my_egrp_id+1)
          !
          IF (jbnd == 0 .OR. jbnd > nbnd) CYCLE
          !
          !IF ( abs(wg(jbnd,ikk)) < eps_occ) CYCLE
          !
          IF (noncolin) THEN
             !
!$omp parallel do default(shared), private(ig)
             DO ig = 1, npw
                temppsic_nc(dfftt%nl(igk_exx(ig,ikk)),1,ii) = evc_exx(ig,ii)
                temppsic_nc(dfftt%nl(igk_exx(ig,ikk)),2,ii) = evc_exx(npwx+ig,ii)
             ENDDO
!$omp end parallel do
             !
             CALL invfft( 'Wave', temppsic_nc(:,1,ii), dfftt )
             CALL invfft( 'Wave', temppsic_nc(:,2,ii), dfftt )
             !
          ELSE
!$omp parallel do default(shared), private(ig)
             DO ig = 1, npw
                temppsic(dfftt%nl(igk_exx(ig,ikk)),ii) = evc_exx(ig,ii)
             ENDDO
!$omp end parallel do
             !
             CALL invfft( 'Wave', temppsic(:,ii), dfftt )
             !
          ENDIF
       ENDDO
       !
       IQ_LOOP : &
       DO iq = 1,nqs
          !
          ikq  = index_xkq(current_ik,iq)
          ik   = index_xk(ikq)
          !
          xkq = xkq_collect(:,ikq)
          !
          CALL g2_convolution_all( dfftt%ngm, gt, xkp, xkq, iq, ikk )
          IF ( okvan .AND..NOT.tqr ) CALL qvan_init( dfftt%ngm, xkq, xkp )
          !
          DO iegrp = 1, negrp
             !
             ! ... compute the id of group whose data is currently worked on
             wegrp = MOD(iegrp+my_egrp_id-1, negrp)+1
             njt = (all_end(wegrp)-all_start(wegrp)+jblock)/jblock
             !
             IJT_LOOP : &
             DO ijt = 1, njt
                !
                jblock_start = (ijt - 1) * jblock + all_start(wegrp)
                jblock_end = MIN(jblock_start+jblock-1,all_end(wegrp))
                !
                JBND_LOOP : &
                DO ii = 1, nibands(my_egrp_id+1)
                   !
                   jbnd = ibands(ii,my_egrp_id+1)
                   !
                   IF (jbnd==0 .OR. jbnd>nbnd) CYCLE
                   !
                   !determine which j-bands to calculate
                   jstart = 0
                   jend = 0
                   !
                   DO ipair = 1, max_pairs
                      IF (egrp_pairs(1,ipair,my_egrp_id+1) == jbnd) THEN
                         IF (jstart == 0) THEN
                            jstart = egrp_pairs(2,ipair,my_egrp_id+1)
                            jend = jstart
                         ELSE
                            jend = egrp_pairs(2,ipair,my_egrp_id+1)
                         ENDIF
                      ENDIF
                   ENDDO
                   !
                   !these variables prepare for inner band parallelism
                   jstart = MAX(jstart,jblock_start)
                   jend = MIN(jend,jblock_end)
                   ibnd_inner_start = jstart
                   ibnd_inner_end = jend
                   ibnd_inner_count = jend-jstart+1
                   !
                   !allocate arrays
                   ALLOCATE( rhoc(nrxxs,ibnd_inner_count) )
#if defined(__USE_MANY_FFT)
                   prhoc(1:nrxxs*ibnd_inner_count) => rhoc
#endif 
                   !calculate rho in real space
                   nblock = 2048
                   nrt = (nrxxs+nblock-1) / nblock
!$omp parallel do collapse(2) private(ir_start,ir_end)
                   DO irt = 1, nrt
                      DO ibnd = ibnd_inner_start, ibnd_inner_end
                         ir_start = (irt - 1) * nblock + 1
                         ir_end = MIN(ir_start+nblock-1,nrxxs)
                         IF (noncolin) THEN
                            DO ir = ir_start, ir_end
                               rhoc(ir,ibnd-ibnd_inner_start+1) = &
                                 ( CONJG(exxbuff(ir,ibnd-all_start(wegrp)+iexx_start,ikq)) * &
                                 temppsic_nc(ir,1,ii) + &
                                 CONJG(exxbuff(nrxxs+ir,ibnd-all_start(wegrp)+iexx_start,ikq)) * &
                                 temppsic_nc(ir,2,ii) ) * omega_inv
                            ENDDO
                         ELSE
                            DO ir = ir_start, ir_end
                               rhoc(ir,ibnd-ibnd_inner_start+1) = omega_inv * &
                                 CONJG(exxbuff(ir,ibnd-all_start(wegrp)+iexx_start,ikq)) * &
                                 temppsic(ir,ii)
                            ENDDO
                         ENDIF
                      ENDDO
                   ENDDO
!$omp end parallel do
                   !
                   ! augment the "charge" in real space
                   IF (okvan .AND. tqr) THEN
!$omp parallel do default(shared) private(ibnd) firstprivate(ibnd_inner_start,ibnd_inner_end)
                      DO ibnd = ibnd_inner_start, ibnd_inner_end
                         CALL addusxx_r( rhoc(:,ibnd-ibnd_inner_start+1), &
                                        becxx(ikq)%k(:,ibnd), becpsi%k(:,jbnd))
                      ENDDO
!$omp end parallel do
                   ENDIF
                   !
                   ! bring rhoc to G-space
#if defined(__USE_MANY_FFT)
                   CALL fwfft ('Rho', prhoc, dfftt, howmany=ibnd_inner_count)
#else
                   DO ibnd = ibnd_inner_start, ibnd_inner_end
                      CALL fwfft('Rho', rhoc(:,ibnd-ibnd_inner_start+1), dfftt)
                   ENDDO
#endif
                   ! augment the "charge" in G space
                   IF (okvan .AND. .NOT. tqr) THEN
                      DO ibnd = ibnd_inner_start, ibnd_inner_end
                         CALL addusxx_g(dfftt, rhoc(:,ibnd-ibnd_inner_start+1), &
                              xkq, xkp, 'c', becphi_c=becxx(ikq)%k(:,ibnd),     &
                              becpsi_c=becpsi%k(:,jbnd))
                      ENDDO
                   ENDIF
                   !
!$omp parallel do reduction(+:energy) private(vc)
                   DO ibnd = ibnd_inner_start, ibnd_inner_end
                      vc=0.0_DP
                      DO ig=1,dfftt%ngm
                         vc = vc + coulomb_fac(ig,iq,ikk) * &
                             DBLE(rhoc(dfftt%nl(ig),ibnd-ibnd_inner_start+1) *&
                             CONJG(rhoc(dfftt%nl(ig),ibnd-ibnd_inner_start+1)))
                      ENDDO
                      vc = vc * omega * x_occupation(ibnd,ik) / nqs
                      energy = energy - exxalfa * vc * wg(jbnd,ikk)
                      !
                      IF (okpaw) THEN
                         energy = energy +exxalfa*x_occupation(ibnd,ik)/nqs*wg(jbnd,ikk) &
                              *PAW_xx_energy(becxx(ikq)%k(:,ibnd), becpsi%k(:,jbnd))
                      ENDIF
                   ENDDO
!$omp end parallel do
                   !
                   !deallocate memory
                   DEALLOCATE( rhoc )
                ENDDO &
                JBND_LOOP
                !
             ENDDO&
             IJT_LOOP
             ! get the next nbnd/negrp data
             IF (negrp > 1) call mp_circular_shift_left( exxbuff(:,:,ikq), me_egrp, inter_egrp_comm )
             !
          END DO !iegrp
          !
          IF ( okvan .AND. .NOT.tqr ) CALL qvan_clean()
       ENDDO &
       IQ_LOOP
       !
    ENDDO &
    IKK_LOOP
    !
    IF (noncolin) THEN
       DEALLOCATE( temppsic_nc )
    ELSE
       DEALLOCATE( temppsic )
    ENDIF
    !
    DEALLOCATE( fac )
    !
    CALL deallocate_bec_type( becpsi )
    !
    CALL mp_sum( energy, inter_egrp_comm )
    CALL mp_sum( energy, intra_egrp_comm )
    CALL mp_sum( energy, inter_pool_comm )
    !
    exxenergy_bp_k = energy
    CALL change_data_structure( .FALSE. )
    !
  END FUNCTION  exxenergy_bp_k
  !
  !-----------------------------------------------------------------------
  FUNCTION exx_stress_bp()
    !-----------------------------------------------------------------------
    !! This is Eq.(10) of PRB 73, 125120 (2006).
    !
    USE constants,            ONLY : fpi, e2, pi, tpi
    USE io_files,             ONLY : iunwfc_exx, nwordwfc
    USE buffers,              ONLY : get_buffer
    USE cell_base,            ONLY : alat, omega, bg, at, tpiba
    USE symm_base,            ONLY : nsym, s
    USE wvfct,                ONLY : nbnd, npwx, wg, current_k
    USE wavefunctions,        ONLY : evc
    USE klist,                ONLY : xk, ngk, nks
    USE lsda_mod,             ONLY : lsda, current_spin, isk
    USE gvect,                ONLY : g
    USE mp_pools,             ONLY : npool, inter_pool_comm
    USE mp_exx,               ONLY : inter_egrp_comm, intra_egrp_comm, &
                                     ibands, nibands, my_egrp_id, jblock, &
                                     egrp_pairs, max_pairs, negrp, me_egrp, &
                                     all_start, all_end, iexx_start
    USE mp,                   ONLY : mp_sum, mp_circular_shift_left
    USE fft_base,             ONLY : dffts
    USE fft_interfaces,       ONLY : fwfft, invfft
    USE uspp,                 ONLY : okvan
    !
    USE exx_base,             ONLY : nq1, nq2, nq3, nqs, eps, exxdiv,       &
                                     x_gamma_extrapolation, on_double_grid, &
                                     grid_factor, yukawa, erfc_scrlen,      &
                                     use_coulomb_vcut_ws, use_coulomb_vcut_spheric, &
                                     gau_scrlen, vcut, index_xkq, index_xk, index_sym
    USE exx_bp_utils,         ONLY : change_data_structure, transform_evc_to_exx, &
                                     g_exx, igk_exx, nwordwfc_exx, evc_exx
    USE coulomb_vcut_module,  ONLY : vcut_get,  vcut_spheric_get
    !
    IMPLICIT NONE
    !
    ! ... local variables
    !
    REAL(DP) :: exx_stress_bp(3,3), exx_stress_(3,3)
    !
    COMPLEX(DP),ALLOCATABLE :: tempphic(:), temppsic(:), result(:)
    COMPLEX(DP),ALLOCATABLE :: tempphic_nc(:,:), temppsic_nc(:,:), &
                               result_nc(:,:)
    COMPLEX(DP),ALLOCATABLE :: rhoc(:)
    REAL(DP),   ALLOCATABLE :: fac(:), fac_tens(:,:,:), fac_stress(:)
    INTEGER  :: npw, jbnd, ibnd, ik, ikk, ig, ir, ikq, iq, isym
    INTEGER  :: nqi, iqi, beta, nrxxs, ngm
    INTEGER  :: ibnd_loop_start
    REAL(DP) :: x1, x2
    REAL(DP) :: qq, xk_cryst(3), sxk(3), xkq(3), vc(3,3), x, q(3)
    REAL(DP) :: delta(3,3)
    INTEGER :: jstart, jend, ii, ipair, jblock_start, jblock_end
    INTEGER :: iegrp, wegrp
    INTEGER :: exxbuff_index
    !
    CALL transform_evc_to_exx( 0 )
    !
    IF (npool>1) CALL errore( 'exx_stress2', 'stress not available with pools', 1 )
    IF (noncolin) CALL errore( 'exx_stress2', 'noncolinear stress not implemented', 1 )
    IF (okvan) CALL infomsg( 'exx_stress2', 'USPP stress not tested' )
    !
    nrxxs = dfftt%nnr
    ngm   = dfftt%ngm
    delta = RESHAPE( (/1._dp,0._dp,0._dp, 0._dp,1._dp,0._dp, 0._dp,0._dp,1._dp/), (/3,3/))
    exx_stress_ = 0._dp
    !
    ALLOCATE( tempphic(nrxxs), temppsic(nrxxs), rhoc(nrxxs), fac(ngm) )
    ALLOCATE( fac_tens(3,3,ngm), fac_stress(ngm) )
    !
    nqi = nqs
    !
    ! ... loop over k-points
    DO ikk = 1, nks
       current_k = ikk
       IF (lsda) current_spin = isk(ikk)
       npw = ngk(ikk)
       !
       IF (nks > 1) CALL get_buffer( evc_exx, nwordwfc_exx, iunwfc_exx, ikk )
       !
       DO iqi = 1, nqi
          !
          iq = iqi
          !
          ikq  = index_xkq(current_k,iq)
          ik   = index_xk(ikq)
          isym = ABS(index_sym(ikq))      
          !      

          ! FIXME: use cryst_to_cart and company as above..      
          xk_cryst(:) = at(1,:)*xk(1,ik)+at(2,:)*xk(2,ik)+at(3,:)*xk(3,ik)      
          IF (index_sym(ikq) < 0) xk_cryst = -xk_cryst      
          sxk(:) = s(:,1,isym)*xk_cryst(1) + &      
                   s(:,2,isym)*xk_cryst(2) + &      
                   s(:,3,isym)*xk_cryst(3)      
          xkq(:) = bg(:,1)*sxk(1) + bg(:,2)*sxk(2) + bg(:,3)*sxk(3)      
          !      
          !CALL start_clock ('exxen2_ngmloop')      
          !      
!$omp parallel do default(shared), private(ig, beta, q, qq, on_double_grid, x)
          DO ig = 1, ngm      
             IF (negrp == 1) THEN      
                q(1) = xk(1,current_k) - xkq(1) + g(1,ig)      
                q(2) = xk(2,current_k) - xkq(2) + g(2,ig)      
                q(3) = xk(3,current_k) - xkq(3) + g(3,ig)      
             ELSE      
                q(1) = xk(1,current_k) - xkq(1) + g_exx(1,ig)      
                q(2) = xk(2,current_k) - xkq(2) + g_exx(2,ig)      
                q(3) = xk(3,current_k) - xkq(3) + g_exx(3,ig)      
             ENDIF      
             !      
             q = q * tpiba      
             qq = ( q(1)*q(1) + q(2)*q(2) + q(3)*q(3) )      
             !      
             DO beta = 1, 3      
                fac_tens(1:3,beta,ig) = q(1:3)*q(beta)      
             ENDDO      
             !      
             IF (x_gamma_extrapolation) THEN      
                on_double_grid = .TRUE.      
                x= 0.5d0/tpiba*(q(1)*at(1,1)+q(2)*at(2,1)+q(3)*at(3,1))*nq1      
                on_double_grid = on_double_grid .AND. (ABS(x-NINT(x))<eps)      
                x= 0.5d0/tpiba*(q(1)*at(1,2)+q(2)*at(2,2)+q(3)*at(3,2))*nq2      
                on_double_grid = on_double_grid .AND. (ABS(x-NINT(x))<eps)      
                x= 0.5d0/tpiba*(q(1)*at(1,3)+q(2)*at(2,3)+q(3)*at(3,3))*nq3      
                on_double_grid = on_double_grid .AND. (ABS(x-NINT(x))<eps)      
             ELSE      
                on_double_grid = .FALSE.      
             ENDIF      
             !      
             IF (use_coulomb_vcut_ws) THEN      
                fac(ig) = vcut_get(vcut, q)      
                fac_stress(ig) = 0._dp   ! not implemented      
                IF (gamma_only .AND. qq > 1.d-8) fac(ig) = 2.d0 * fac(ig)      
                !      
             ELSEIF ( use_coulomb_vcut_spheric ) THEN      
                fac(ig) = vcut_spheric_get(vcut, q)      
                fac_stress(ig) = 0._dp   ! not implemented      
                IF (gamma_only .AND. qq > 1.d-8) fac(ig) = 2.d0 * fac(ig)      
                !      
             ELSEIF (gau_scrlen > 0) THEN      
                fac(ig) = e2*((pi/gau_scrlen)**(1.5d0))* &       
                          EXP(-qq/4.d0/gau_scrlen) * grid_factor       
                fac_stress(ig) =  e2*2.d0/4.d0/gau_scrlen  * &       
                                  EXP(-qq/4.d0/gau_scrlen) * &
                                  ((pi/gau_scrlen)**(1.5d0))*grid_factor       
                IF (gamma_only) fac(ig) = 2.d0 * fac(ig)       
                IF (gamma_only) fac_stress(ig) = 2.d0 * fac_stress(ig)       
                IF (on_double_grid) fac(ig) = 0._dp       
                IF (on_double_grid) fac_stress(ig) = 0._dp       
                !       
             ELSEIF (qq > 1.d-8) THEN      
                IF ( erfc_scrlen > 0 ) THEN       
                  fac(ig)=e2*fpi/qq*(1._dp-EXP(-qq/4.d0/erfc_scrlen**2)) * grid_factor       
                  fac_stress(ig) = -e2*fpi * 2.d0/qq**2 * ( &       
                      (1._dp+qq/4.d0/erfc_scrlen**2)*EXP(-qq/4.d0/erfc_scrlen**2) - 1._dp) * &       
                      grid_factor       
                ELSE       
                  fac(ig)=e2*fpi/( qq + yukawa ) * grid_factor       
                  fac_stress(ig) = 2.d0 * e2*fpi/(qq+yukawa)**2 * grid_factor       
                ENDIF       
                !       
                IF (gamma_only) fac(ig) = 2.d0 * fac(ig)       
                IF (gamma_only) fac_stress(ig) = 2.d0 * fac_stress(ig)       
                IF (on_double_grid) fac(ig) = 0._dp       
                IF (on_double_grid) fac_stress(ig) = 0._dp       
                !      
             ELSE 
                ! 
                fac(ig) = -exxdiv ! or rather something else (see f.gygi)       
                fac_stress(ig) = 0._dp  ! or -exxdiv_stress (not yet implemented)       
                IF ( yukawa> 0._dp .AND. .NOT. x_gamma_extrapolation) THEN       
                  fac(ig) = fac(ig) + e2*fpi/( qq + yukawa )       
                  fac_stress(ig) = 2.d0 * e2*fpi/(qq+yukawa)**2       
                ENDIF       
                IF (erfc_scrlen > 0._dp .AND. .NOT. x_gamma_extrapolation) THEN       
                  fac(ig) = e2*fpi / (4.d0*erfc_scrlen**2)       
                  fac_stress(ig) = e2*fpi / (8.d0*erfc_scrlen**4)       
                ENDIF 
                !
             ENDIF
             !
          ENDDO      
!$omp end parallel do
          !CALL stop_clock ('exxen2_ngmloop')      
          DO iegrp = 1, negrp      
             !      
             ! compute the id of group whose data is currently worked on      
             wegrp = MOD(iegrp+my_egrp_id-1, negrp)+1      
             !      
             jblock_start = all_start(wegrp)      
             jblock_end = all_end(wegrp)      
             !      
             ! loop over bands      
             DO ii = 1, nibands(my_egrp_id+1)      
                !      
                jbnd = ibands(ii,my_egrp_id+1)      
                !      
                IF (jbnd==0 .OR. jbnd>nbnd) CYCLE      
                !      
                !determine which j-bands to calculate      
                jstart = 0      
                jend = 0      
                DO ipair=1, max_pairs      
                   IF (egrp_pairs(1,ipair,my_egrp_id+1).eq.jbnd)THEN      
                      IF (jstart == 0)THEN      
                         jstart = egrp_pairs(2,ipair,my_egrp_id+1)      
                         jend = jstart      
                      ELSE      
                         jend = egrp_pairs(2,ipair,my_egrp_id+1)      
                      ENDIF      
                   ENDIF      
                ENDDO      
                !      
                jstart = MAX(jstart,jblock_start)      
                jend = MIN(jend,jblock_end)      
                !      
                temppsic(:) = ( 0._dp, 0._dp )      
!$omp parallel do default(shared), private(ig)
                DO ig = 1, npw      
                   temppsic(dfftt%nl(igk_exx(ig,ikk))) = evc_exx(ig,ii)      
                ENDDO      
!$omp end parallel do
                !      
                IF (gamma_only) THEN      
!$omp parallel do default(shared), private(ig)
                   DO ig = 1, npw      
                      temppsic(dfftt%nlm(igk_exx(ig,ikk))) = CONJG(evc_exx(ig,ii))      
                   ENDDO      
!$omp end parallel do
                ENDIF      
                !      
                CALL invfft( 'Wave', temppsic, dfftt )      
                !      
                IF (gamma_only) THEN      
                   !      
                   IF (MOD(jstart,2) == 0) THEN      
                      ibnd_loop_start = jstart-1      
                   ELSE      
                      ibnd_loop_start = jstart      
                   ENDIF      
                   !      
                   DO ibnd = ibnd_loop_start, jend, 2     !for each band of psi      
                      !      
                      exxbuff_index = (ibnd+1)/2-(all_start(wegrp)+1)/2+(iexx_start+1)/2      
                      !      
                      IF ( ibnd < jstart ) THEN      
                         x1 = 0._dp      
                      ELSE      
                         x1 = x_occupation(ibnd,ik)      
                      ENDIF      
                      !      
                      IF ( ibnd == jend) THEN      
                         x2 = 0._dp      
                      ELSE      
                         x2 = x_occupation(ibnd+1,ik)      
                      ENDIF      
                      !      
                      IF ( ABS(x1) < eps_occ .AND. ABS(x2) < eps_occ ) CYCLE      
                      !      
                      ! calculate rho in real space      
!$omp parallel do default(shared), private(ir)
                      DO ir = 1, nrxxs      
                         tempphic(ir) = exxbuff(ir,exxbuff_index,ikq)      
                         rhoc(ir) = CONJG(tempphic(ir))*temppsic(ir) / omega      
                      ENDDO      
!$omp end parallel do
                      ! bring it to G-space      
                      CALL fwfft( 'Rho', rhoc, dfftt )      
                      !      
                      vc = 0._dp      
!$omp parallel do default(shared), private(ig), reduction(+:vc)
                      DO ig = 1, ngm      
                         !      
                         vc(:,:) = vc(:,:) + x1 * 0.25_dp * &      
                                   ABS( rhoc(dfftt%nl(ig)) + &      
                                   CONJG(rhoc(dfftt%nlm(ig))))**2 * &      
                                   (fac_tens(:,:,ig)*fac_stress(ig)/2.d0 - delta(:,:)*fac(ig))      
                         vc(:,:) = vc(:,:) + x2 * 0.25_dp * &      
                                   ABS( rhoc(dfftt%nl(ig)) - &      
                                   CONJG(rhoc(dfftt%nlm(ig))))**2 * &      
                                   (fac_tens(:,:,ig)*fac_stress(ig)/2.d0 - delta(:,:)*fac(ig))      
                      ENDDO      
!$omp end parallel do
                      vc = vc / nqs / 4.d0
                      exx_stress_ = exx_stress_ + exxalfa * vc * wg(jbnd,ikk)
                   ENDDO
                   !
                ELSE
                   !
                   DO ibnd = jstart, jend    !for each band of psi
                      !
                      IF ( ABS(x_occupation(ibnd,ik)) < 1.d-6) CYCLE
                      !
                      ! calculate rho in real space
!$omp parallel do default(shared), private(ir)
                      DO ir = 1, nrxxs
                         tempphic(ir) = exxbuff(ir,ibnd-all_start(wegrp)+iexx_start,ikq)
                         rhoc(ir) = CONJG(tempphic(ir))*temppsic(ir) / omega
                      ENDDO
!$omp end parallel do
                      !
                      ! bring it to G-space
                      CALL fwfft( 'Rho', rhoc, dfftt )
                      !
                      vc = 0._dp
!$omp parallel do default(shared), private(ig), reduction(+:vc)
                      DO ig = 1, ngm
                         vc(:,:) = vc(:,:) + rhoc(dfftt%nl(ig))  * &
                                   CONJG(rhoc(dfftt%nl(ig))) *     &
                                   (fac_tens(:,:,ig)*fac_stress(ig)/2.d0 - &
                                   delta(:,:)*fac(ig))
                      ENDDO
!$omp end parallel do
                      !
                      vc = vc * x_occupation(ibnd,ik) / nqs / 4.d0
                      exx_stress_ = exx_stress_ + exxalfa * vc * wg(jbnd,ikk)
                      !
                   ENDDO
                   !
                ENDIF ! gamma or k-points
                !
             ENDDO ! jbnd
             !
             ! get the next nbnd/negrp data
             IF (negrp > 1) CALL mp_circular_shift_left( exxbuff(:,:,ikq), me_egrp, &
                                                         inter_egrp_comm )
             !
          ENDDO ! iegrp
          !
       ENDDO ! iqi
       !
    ENDDO ! ikk
    !
    DEALLOCATE( tempphic, temppsic, rhoc, fac, fac_tens, fac_stress )
    !
    CALL mp_sum( exx_stress_, intra_egrp_comm )
    CALL mp_sum( exx_stress_, inter_egrp_comm )
    CALL mp_sum( exx_stress_, inter_pool_comm )
    !
    exx_stress_bp = exx_stress_
    !
    CALL change_data_structure( .FALSE. )
    !
  END FUNCTION exx_stress_bp
  !
#if defined(__CUDA)
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION p14_is_int( s )
    !-----------------------------------------------------------------------
    !! P14: the string is a whole number: optional sign, digits, blanks around
    !
    CHARACTER(LEN=*), INTENT(IN) :: s
    CHARACTER(LEN=LEN(s)) :: t
    INTEGER :: i, n
    !
    t = ADJUSTL( s )
    n = LEN_TRIM( t )
    p14_is_int = .FALSE.
    IF ( n == 0 ) RETURN
    i = 1
    IF ( t(1:1) == '+' .OR. t(1:1) == '-' ) i = 2
    IF ( i > n ) RETURN
    DO WHILE ( i <= n )
       IF ( t(i:i) < '0' .OR. t(i:i) > '9' ) RETURN
       i = i + 1
    ENDDO
    p14_is_int = .TRUE.
    !
  END FUNCTION p14_is_int
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION p14_is_off( s )
    !-----------------------------------------------------------------------
    !! P14: "off", "false" or "no" in any case (the QE_EXX_FAST synonyms of 0)
    !
    CHARACTER(LEN=*), INTENT(IN) :: s
    CHARACTER(LEN=8) :: t
    INTEGER :: i
    !
    t = ADJUSTL( s )
    DO i = 1, LEN( t )
       IF ( t(i:i) >= 'A' .AND. t(i:i) <= 'Z' ) t(i:i) = ACHAR( IACHAR( t(i:i) ) + 32 )
    ENDDO
    p14_is_off = ( TRIM(t) == 'off' .OR. TRIM(t) == 'false' .OR. TRIM(t) == 'no' )
    !
  END FUNCTION p14_is_off
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION vexx_fast_path()
  !-----------------------------------------------------------------------
    !! P4: can the batched norm-conserving collinear fast path be used?
    USE uspp,           ONLY : okvan
    USE paw_variables,  ONLY : okpaw
    USE mp_exx,         ONLY : negrp
    IMPLICIT NONE
    LOGICAL, SAVE :: checked = .FALSE., env_off = .FALSE.
    CHARACTER(LEN=16) :: val
    INTEGER :: lenv, stat, ios
    CHARACTER(LEN=128) :: why
    IF (.NOT. checked) THEN
       CALL GET_ENVIRONMENT_VARIABLE( 'QE_EXX_FAST', val, lenv, stat )
       env_off = ( stat == 0 .AND. lenv > 0 .AND. ( val(1:1) == '0' .OR. p14_is_off( val ) ) )   ! P14
       checked = .TRUE.
       IF (env_off) WRITE( stdout, '(5X,"EXX: batched fast path disabled by QE_EXX_FAST=",A)' ) TRIM(val)
       ! P6: precision mode and pair block. P14: whole numbers only; a rejected value keeps the default and is
       ! reported (a list-directed READ truncates "2.5" to 2, which used to switch single precision on)
       CALL GET_ENVIRONMENT_VARIABLE( 'QE_EXX_SP', val, lenv, stat )
       IF (stat == 0 .AND. lenv > 0) THEN
          IF ( p14_is_int( val ) ) READ( val, *, IOSTAT=ios ) p6_sp_mode
          IF ( .NOT. p14_is_int( val ) .OR. p6_sp_mode < 0 .OR. p6_sp_mode > 2 ) THEN
             WRITE( stdout, '(5X,"EXX: QE_EXX_SP=",A," not understood (0, 1 or 2): double precision")' ) TRIM(val)
             p6_sp_mode = 0
          ENDIF
       ENDIF
       CALL GET_ENVIRONMENT_VARIABLE( 'QE_EXX_SP_THR', val, lenv, stat )
       IF (stat == 0 .AND. lenv > 0) THEN
          ios = 0
          READ( val, *, IOSTAT=ios ) p6_sp_thr
          IF ( ios /= 0 .OR. .NOT. ( p6_sp_thr > 0.0_DP ) ) THEN
             WRITE( stdout, '(5X,"EXX: QE_EXX_SP_THR=",A," not understood (a positive threshold in Ry): 1e-6")' ) TRIM(val)
             p6_sp_thr = 1.0E-6_DP
          ENDIF
       ENDIF
       CALL GET_ENVIRONMENT_VARIABLE( 'QE_EXX_PBLK', val, lenv, stat )
       IF (stat == 0 .AND. lenv > 0) THEN
          IF ( p14_is_int( val ) ) THEN
             READ( val, *, IOSTAT=ios ) p6_pblk
          ELSE
             WRITE( stdout, '(5X,"EXX: QE_EXX_PBLK=",A," not understood (a whole number): automatic pair block")' ) TRIM(val)
             p6_pblk = -1
          ENDIF
       ENDIF
       IF (p6_sp_mode == 1) WRITE( stdout, '(5X,"EXX: single-precision pair FFTs while dexx >",1PE9.1," Ry")' ) p6_sp_thr
       IF (p6_sp_mode >= 2) WRITE( stdout, '(5X,"EXX: single-precision pair FFTs in every ACE build")' )
    ENDIF
    vexx_fast_path = use_gpu .AND. (.NOT. env_off) .AND. (.NOT. p15_fast_disabled) .AND. (.NOT. gamma_only) .AND. (.NOT. noncolin) &
                     .AND. (.NOT. (okvan .AND. tqr)) .AND. (negrp == 1) &      ! P18: USPP/PAW on the fast path, tqr on the original
                     .AND. (dfftt%nproc == 1) .AND. (x_nbnd_occ > 0) &
                     .AND. (dfftt%nr1x == dfftt%nr1) .AND. (dfftt%nr2x == dfftt%nr2) &
                     .AND. (dfftt%nr3x == dfftt%nr3) .AND. (dfftt%nnr == dfftt%nr1*dfftt%nr2*dfftt%nr3)
    ! P10: banner of the decision (once, and again whenever it changes)
    IF ( MERGE( 1, 0, vexx_fast_path ) /= p10_fast_last ) THEN
       p10_fast_last = MERGE( 1, 0, vexx_fast_path )
       IF ( vexx_fast_path ) THEN
          WRITE( stdout, '(5X,"EXX-GPU: vexx takes the batched fast path (P4/P6); QE_EXX_SP mode ",I0)' ) p6_sp_mode
          IF ( okvan ) WRITE( stdout, '(5X,"EXX-GPU: vexx adds the ultrasoft/PAW augmentation on the device (P18); single-precision pair FFTs are off for it")' )
       ELSE
          why = ''
          IF ( .NOT. use_gpu ) why = TRIM(why) // ' no-gpu'
          IF ( env_off ) why = TRIM(why) // ' QE_EXX_FAST=0'
          IF ( p15_fast_disabled ) why = TRIM(why) // ' device-memory'
          IF ( gamma_only ) why = TRIM(why) // ' gamma_only'
          IF ( noncolin ) why = TRIM(why) // ' noncolin'
          IF ( okvan .AND. tqr ) why = TRIM(why) // ' ultrasoft-tqr'
          IF ( negrp /= 1 ) why = TRIM(why) // ' band-groups'
          IF ( dfftt%nproc /= 1 ) why = TRIM(why) // ' parallel-FFT'
          IF ( x_nbnd_occ <= 0 ) why = TRIM(why) // ' no-occupied-bands'
          IF ( dfftt%nr1x /= dfftt%nr1 .OR. dfftt%nr2x /= dfftt%nr2 .OR. dfftt%nr3x /= dfftt%nr3 .OR. &
               dfftt%nnr /= dfftt%nr1*dfftt%nr2*dfftt%nr3 ) why = TRIM(why) // ' padded-FFT-grid'
          WRITE( stdout, '(5X,"EXX-GPU: vexx takes the original vexx_k_gpu path;",A)' ) TRIM(why)
       ENDIF
    ENDIF
  END FUNCTION vexx_fast_path
  !
  !-----------------------------------------------------------------------
  INTEGER FUNCTION vexx_fast_plan( nr1, nr2, nr3, batch, sp )
  !-----------------------------------------------------------------------
    !! P6: cached cuFFT plan for a batch of 3-D transforms on the EXX grid (Z2Z or C2C).
    USE cufft
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nr1, nr2, nr3, batch
    LOGICAL, INTENT(IN) :: sp
    INTEGER :: i, j, istat, key(5), fft_dim(3), ptype
    key = (/ nr1, nr2, nr3, batch, MERGE(1,0,sp) /)
    DO i = 1, p6_nplans
       IF ( ALL( p6_plan_key(:,i) == key ) ) THEN
          vexx_fast_plan = p6_plans(i)
          RETURN
       ENDIF
    ENDDO
    i = p6_plan_next
    DO j = 1, p6_nplans        ! P11: never evict a plan the running call still uses
       IF ( p6_plans(i) == 0 ) EXIT
       IF ( ALL( p6_plans(i) /= p6_plan_pinned ) ) EXIT
       i = MOD( i, p6_nplans ) + 1
    ENDDO
    IF ( p6_plans(i) /= 0 ) THEN
       istat = cufftDestroy( p6_plans(i) )
       p6_plans(i) = 0
    ENDIF
    fft_dim = (/ nr3, nr2, nr1 /)
    ptype = CUFFT_Z2Z
    IF (sp) ptype = CUFFT_C2C
    istat = cufftPlanMany( p6_plans(i), 3, fft_dim, fft_dim, 1, nr1*nr2*nr3, &
                           fft_dim, 1, nr1*nr2*nr3, ptype, batch )
    IF ( istat /= 0 ) CALL errore( 'vexx_fast_plan', 'cufftPlanMany failed', ABS(istat) )
    p6_plan_key(:,i) = key
    p6_plan_next = MOD( i, p6_nplans ) + 1
    vexx_fast_plan = p6_plans(i)
  END FUNCTION vexx_fast_plan
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vexx_fast_main_dp( nrxxs, nbuf, nkqs, ebuf, m, psic, nocc, nqs, ikq, facb, xocc, &
                                npairs, pblk, rhoc, res, plan_full, plan_tail, omega_inv, scale )
  !-----------------------------------------------------------------------
    !! P6: pair densities -> FFT -> Coulomb kernel x occupation / N -> inverse FFT -> accumulate,
    !! double precision.
    USE cufft
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nrxxs, nbuf, nkqs, m, nocc, nqs, npairs, pblk, plan_full, plan_tail
    COMPLEX(DP), DEVICE, INTENT(IN) :: ebuf(nrxxs,nbuf,nkqs), psic(nrxxs,m)
    INTEGER, DEVICE, INTENT(IN) :: ikq(nqs)
    REAL(DP), DEVICE, INTENT(IN) :: facb(nrxxs,nqs), xocc(nocc,nqs)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: rhoc(nrxxs,pblk), res(nrxxs,m)
    REAL(DP), INTENT(IN) :: omega_inv, scale
    INTEGER :: ibnd, p0, np_, ip, ir, ipg, iq, jb, plan, istat
    !
    DO ibnd = 1, m
       DO p0 = 1, npairs, pblk
          np_ = MIN( pblk, npairs - p0 + 1 )
          plan = plan_full
          IF ( np_ /= pblk ) plan = plan_tail
          !$cuf kernel do (2)
          DO ip = 1, np_
             DO ir = 1, nrxxs
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                rhoc(ir,ip) = CONJG( ebuf(ir,jb,ikq(iq)) ) * psic(ir,ibnd) * omega_inv
             ENDDO
          ENDDO
          istat = cufftExecZ2Z( plan, rhoc(1,1), rhoc(1,1), CUFFT_FORWARD )
          IF ( istat /= 0 ) CALL errore( 'vexx_bp_k_gpu_fast', 'cuFFT exec failed (P11 check)', ABS(istat) )
          !$cuf kernel do (2)
          DO ip = 1, np_
             DO ir = 1, nrxxs
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                rhoc(ir,ip) = rhoc(ir,ip) * ( facb(ir,iq) * xocc(jb,iq) * scale )
             ENDDO
          ENDDO
          istat = cufftExecZ2Z( plan, rhoc(1,1), rhoc(1,1), CUFFT_INVERSE )
          IF ( istat /= 0 ) CALL errore( 'vexx_bp_k_gpu_fast', 'cuFFT exec failed (P11 check)', ABS(istat) )
          !$cuf kernel do (1)
          DO ir = 1, nrxxs
             DO ip = 1, np_
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                res(ir,ibnd) = res(ir,ibnd) + rhoc(ir,ip) * ebuf(ir,jb,ikq(iq))
             ENDDO
          ENDDO
       ENDDO
    ENDDO
  END SUBROUTINE vexx_fast_main_dp
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vexx_fast_main_sp( nrxxs, nbuf, nkqs, ebuf, m, psic, nocc, nqs, ikq, facb, xocc, &
                                npairs, pblk, rhoc, res, plan_full, plan_tail, omega_inv, scale )
  !-----------------------------------------------------------------------
    !! P6: as vexx_fast_main_dp, with the pair densities, their transforms and the Coulomb
    !! multiplication in single precision; the accumulation into res is double precision.
    USE cufft
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nrxxs, nbuf, nkqs, m, nocc, nqs, npairs, pblk, plan_full, plan_tail
    COMPLEX(DP), DEVICE, INTENT(IN) :: ebuf(nrxxs,nbuf,nkqs), psic(nrxxs,m)
    INTEGER, DEVICE, INTENT(IN) :: ikq(nqs)
    REAL(DP), DEVICE, INTENT(IN) :: facb(nrxxs,nqs), xocc(nocc,nqs)
    COMPLEX(4), DEVICE, INTENT(INOUT) :: rhoc(nrxxs,pblk)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: res(nrxxs,m)
    REAL(DP), INTENT(IN) :: omega_inv, scale
    INTEGER :: ibnd, p0, np_, ip, ir, ipg, iq, jb, plan, istat
    !
    DO ibnd = 1, m
       DO p0 = 1, npairs, pblk
          np_ = MIN( pblk, npairs - p0 + 1 )
          plan = plan_full
          IF ( np_ /= pblk ) plan = plan_tail
          !$cuf kernel do (2)
          DO ip = 1, np_
             DO ir = 1, nrxxs
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                rhoc(ir,ip) = CMPLX( CONJG( ebuf(ir,jb,ikq(iq)) ) * psic(ir,ibnd) * omega_inv, KIND=4 )
             ENDDO
          ENDDO
          istat = cufftExecC2C( plan, rhoc(1,1), rhoc(1,1), CUFFT_FORWARD )
          IF ( istat /= 0 ) CALL errore( 'vexx_bp_k_gpu_fast', 'cuFFT exec failed (P11 check)', ABS(istat) )
          !$cuf kernel do (2)
          DO ip = 1, np_
             DO ir = 1, nrxxs
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                rhoc(ir,ip) = rhoc(ir,ip) * REAL( facb(ir,iq) * xocc(jb,iq) * scale, KIND=4 )
             ENDDO
          ENDDO
          istat = cufftExecC2C( plan, rhoc(1,1), rhoc(1,1), CUFFT_INVERSE )
          IF ( istat /= 0 ) CALL errore( 'vexx_bp_k_gpu_fast', 'cuFFT exec failed (P11 check)', ABS(istat) )
          !$cuf kernel do (1)
          DO ir = 1, nrxxs
             DO ip = 1, np_
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                res(ir,ibnd) = res(ir,ibnd) + CMPLX( rhoc(ir,ip), KIND=DP ) * ebuf(ir,jb,ikq(iq))
             ENDDO
          ENDDO
       ENDDO
    ENDDO
  END SUBROUTINE vexx_fast_main_sp
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vexx_bp_k_gpu_fast( lda, n, m, psi, hpsi, ok, becpsi )
  !-----------------------------------------------------------------------
    !! P4/P6: batched GPU version of vexx_k for norm-conserving, collinear k-point runs with one
    !! band group and a serial FFT grid. Same math as vexx_k_gpu: for each band i,
    !!   V_x psi_i(r) = -exxalfa * sum_{q,j occ} x_occ(j,k+q)/nqs * phi_{j,k+q}(r) *
    !!                   IFFT[ v(k-k-q+G) FFT[ conj(phi_{j,k+q}) psi_i / omega ] ](r)
    !! but all (q, j) pairs of a band go through one kernel + one batched FFT at a time,
    !! with resident Coulomb factors, persistent work arrays and own cuFFT plans (P6).
    USE cufft
    USE cudafor,        ONLY : cudaMemGetInfo   ! P15
    USE cell_base,      ONLY : omega, tpiba2, at
    USE constants,      ONLY : e2, fpi, pi
    USE wvfct,          ONLY : npwx, current_k
    USE klist,          ONLY : xk, nkstot, nks
    USE exx_base,       ONLY : nqs, xkq_collect, index_xkq, index_xk, &
                               eps_odg => eps, eps_qdiv, exxdiv, &
                               x_gamma_extrapolation, grid_factor, yukawa, erfc_scrlen, erf_scrlen, &
                               gau_scrlen, use_coulomb_vcut_ws, use_coulomb_vcut_spheric, nq1, nq2, nq3
    USE exx_bp_utils,   ONLY : igk_exx_d
    USE mp_exx,         ONLY : max_ibands
    USE becmod,         ONLY : bec_type          ! P18
    USE uspp,           ONLY : okvan, nkb        ! P18
    USE paw_variables,  ONLY : okpaw             ! P18
    USE ions_base,      ONLY : nat               ! P18
    USE uspp_param,     ONLY : nhm               ! P18
    USE cublas                                   ! P18: device ZGEMM for the non-local term
    !
    IMPLICIT NONE
    !
    INTEGER, INTENT(IN) :: lda, n, m
    COMPLEX(DP), INTENT(IN)    :: psi(lda*npol,max_ibands)
    COMPLEX(DP), INTENT(INOUT) :: hpsi(lda*npol,max_ibands)
    LOGICAL, INTENT(OUT) :: ok   ! P15: .FALSE. when the device work arrays would not fit (nothing done)
    TYPE(bec_type), OPTIONAL, INTENT(IN) :: becpsi   ! P18: <beta|psi> of the current k, ultrasoft/PAW runs
    LOGICAL :: us_ok   ! P18
    REAL(DP) :: p20_epaw   ! P20
    INTEGER(KIND=8) :: p15_need, p15_free, p15_total   ! P15
    !
    INTEGER, POINTER :: dfftt__nl(:)
    attributes(DEVICE) :: dfftt__nl
    INTEGER :: ibnd, iq, ikq, ik, ig, ir, nrxxs, current_ik, nocc, npairs, pblk, ntail, ngmt, nks_loc
    INTEGER :: nbuf, nkqs_d, nr1, nr2, nr3, plan_full, plan_tail, plan_wave, istat
    INTEGER(KIND=8) :: cache_bytes
    LOGICAL :: sp, cache_ok
    ! P6d
    LOGICAL :: dev_ok, xgamma, odg1, odg2, odg3
    INTEGER :: p10_key
    REAL(DP) :: q1, q2, q3, qq, x, gf, f, xk1, xk2, xk3, nqh1, nqh2, nqh3, tpiba2_, gfac_
    REAL(DP) :: a11, a21, a31, a12, a22, a32, a13, a23, a33
    REAL(DP) :: e2_, fpi_, pi_, eps_, eps_qdiv_, exxdiv_, yuk_, erfc_, erf_, gau_
    REAL(DP) :: xkp(3), xkq(3), omega_inv, nqs_inv, scale
    INTEGER, EXTERNAL :: global_kpoint_index
    !
    ok = .TRUE.
    CALL start_clock( 'vexx_k_setup' )
    dfftt__nl => dfftt%nl_d
    nrxxs = dfftt%nnr
    nr1 = dfftt%nr1; nr2 = dfftt%nr2; nr3 = dfftt%nr3
    ngmt = dfftt%ngm
    current_ik = global_kpoint_index( nkstot, current_k )
    xkp = xk(:,current_k)
    omega_inv = 1.0_DP / omega
    nqs_inv = 1.0_DP / nqs
    scale = 1.0_DP / DBLE( nr1*nr2*nr3 )      ! fwfft normalisation, folded into the Coulomb kernel
    nocc = x_nbnd_occ
    npairs = nqs * nocc
    ! P6e: a batch of pair densities that fits the L2 cache (~18 MB) unless QE_EXX_PBLK says otherwise
    IF ( p6_pblk > 0 ) THEN
       pblk = MIN( npairs, p6_pblk )
    ELSE
       pblk = NINT( 18.0E6_DP / ( DBLE(nrxxs) * MERGE( 8.0_DP, 16.0_DP, p6_sp_mode >= 2 ) ) )
       pblk = MIN( npairs, MAX( 8, MIN( 128, pblk ) ) )
    ENDIF
    ntail = MOD( npairs, pblk )
    nbuf = SIZE( exxbuff_d, 2 )
    nkqs_d = SIZE( exxbuff_d, 3 )
    sp = ( p6_sp_mode >= 2 ) .OR. ( p6_sp_mode == 1 .AND. ( p6_nbuild <= 2 .OR. exx_dexx_current > p6_sp_thr ) )
    IF ( okvan ) sp = .FALSE.      ! P18: the augmented densities stay in double precision
    !
    ! P15: the runtime aborts on a failed device ALLOCATE even with STAT=, so ask first whether the batched work
    ! arrays (and cuFFT work areas of the same size) fit; if not, this and every later call go to vexx_k_gpu
    p15_need = 0_8
    IF ( .NOT. ALLOCATED(p6_rhoc_d) .AND. .NOT. sp ) p15_need = p15_need + 32_8 * INT(nrxxs,8) * INT(pblk,8)
    IF ( .NOT. ALLOCATED(p6_rhoc_s) .AND. sp )       p15_need = p15_need + 16_8 * INT(nrxxs,8) * INT(pblk,8)
    IF ( .NOT. ALLOCATED(p6_psic_d) )                p15_need = p15_need + 48_8 * INT(nrxxs,8) * INT(m,8)
    IF ( .NOT. ALLOCATED(p6_psi_d) )                 p15_need = p15_need + 32_8 * INT(lda*npol,8) * INT(m,8)
    IF ( .NOT. ALLOCATED(p6_facb_d) )                p15_need = p15_need +  8_8 * INT(nrxxs,8) * INT(nqs,8)
    IF ( p15_need > 0_8 ) THEN
       istat = cudaMemGetInfo( p15_free, p15_total )
       IF ( istat == 0 .AND. p15_need + 268435456_8 > p15_free ) THEN
          CALL p15_release_work_arrays()
          p15_fast_disabled = .TRUE.
          WRITE( stdout, '(5X,"EXX-GPU: the batched fast path would need ",I0," MB of device memory for ",I0," bands on ",I0," grid points, ",I0," MB are free: this run continues on the original vexx_k_gpu path (P15)")' ) &
               p15_need/1000000_8, m, nrxxs, p15_free/1000000_8
          CALL stop_clock( 'vexx_k_setup' )
          ok = .FALSE.
          RETURN
       ENDIF
    ENDIF
    ! ... work arrays, kept across calls
    IF ( ALLOCATED(p6_rhoc_d) ) THEN
       IF ( SIZE(p6_rhoc_d,1) /= nrxxs .OR. SIZE(p6_rhoc_d,2) /= pblk ) DEALLOCATE( p6_rhoc_d )
    ENDIF
    istat = 0   ! P15: STAT= on every work array as well (a belt to the pre-check's braces)
    IF ( .NOT. ALLOCATED(p6_rhoc_d) .AND. .NOT. sp ) ALLOCATE( p6_rhoc_d(nrxxs,pblk), STAT=istat )
    IF ( ALLOCATED(p6_rhoc_s) ) THEN
       IF ( SIZE(p6_rhoc_s,1) /= nrxxs .OR. SIZE(p6_rhoc_s,2) /= pblk ) DEALLOCATE( p6_rhoc_s )
    ENDIF
    IF ( istat == 0 .AND. .NOT. ALLOCATED(p6_rhoc_s) .AND. sp ) ALLOCATE( p6_rhoc_s(nrxxs,pblk), STAT=istat )
    IF ( ALLOCATED(p6_psic_d) ) THEN
       IF ( SIZE(p6_psic_d,1) /= nrxxs .OR. SIZE(p6_psic_d,2) /= m ) DEALLOCATE( p6_psic_d, p6_res_d )
    ENDIF
    IF ( istat == 0 .AND. .NOT. ALLOCATED(p6_psic_d) ) ALLOCATE( p6_psic_d(nrxxs,m), p6_res_d(nrxxs,m), STAT=istat )
    IF ( ALLOCATED(p6_psi_d) ) THEN
       IF ( SIZE(p6_psi_d,1) /= lda*npol .OR. SIZE(p6_psi_d,2) /= m ) DEALLOCATE( p6_psi_d, p6_hpsi_d )
    ENDIF
    IF ( istat == 0 .AND. .NOT. ALLOCATED(p6_psi_d) ) ALLOCATE( p6_psi_d(lda*npol,m), p6_hpsi_d(lda*npol,m), STAT=istat )
    IF ( ALLOCATED(p6_facb_d) ) THEN
       IF ( SIZE(p6_facb_d,1) /= nrxxs .OR. SIZE(p6_facb_d,2) /= nqs ) DEALLOCATE( p6_facb_d, p6_ikq_d )
    ENDIF
    IF ( istat == 0 .AND. .NOT. ALLOCATED(p6_facb_d) ) ALLOCATE( p6_facb_d(nrxxs,nqs), p6_ikq_d(nqs), STAT=istat )
    IF ( ALLOCATED(p6_xocc_d) ) THEN
       IF ( SIZE(p6_xocc_d,1) /= nocc .OR. SIZE(p6_xocc_d,2) /= nqs ) DEALLOCATE( p6_xocc_d )
    ENDIF
    IF ( istat == 0 .AND. .NOT. ALLOCATED(p6_xocc_d) ) ALLOCATE( p6_xocc_d(nocc,nqs), STAT=istat )
    IF ( istat /= 0 ) THEN
       ! P15: no room for the batched work arrays. Release what this path holds, say so once, and let
       ! vexx() redo this call (and take every later one) on the original one-band-at-a-time routine.
       CALL p15_release_work_arrays()
       p15_fast_disabled = .TRUE.
       WRITE( stdout, '(5X,"EXX-GPU: no device memory for the batched fast path work arrays (",I0," bands on ",I0," grid points): this run continues on the original vexx_k_gpu path (P15)")' ) m, nrxxs
       CALL stop_clock( 'vexx_k_setup' )
       ok = .FALSE.
       RETURN
    ENDIF
    !
    ! ... cuFFT plans (cached)
    p6_plan_pinned = 0                                    ! P11: pin this call's plans
    plan_full = vexx_fast_plan( nr1, nr2, nr3, pblk, sp );  p6_plan_pinned(1) = plan_full
    plan_tail = plan_full
    IF ( ntail > 0 ) THEN
       plan_tail = vexx_fast_plan( nr1, nr2, nr3, ntail, sp );  p6_plan_pinned(2) = plan_tail
    ENDIF
    plan_wave = vexx_fast_plan( nr1, nr2, nr3, m, .FALSE. );  p6_plan_pinned(3) = plan_wave
    !
    ! ... per-q data of this k-point: k+q index, occupations; Coulomb factors resident on the device
    CALL start_clock( 'vexx_k_cfac' )
    ! P6d: the host cache is allocated exactly as g2_convolution_all would do it; the device cache
    !      of this k is then filled by a device kernel (or by the host routine for the vcut kernels)
    nks_loc = nks
    IF ( .NOT. ALLOCATED(coulomb_fac) ) ALLOCATE( coulomb_fac(ngmt,nqs,nks_loc) )
    IF ( .NOT. ALLOCATED(coulomb_done) ) THEN
       ALLOCATE( coulomb_done(nqs,nks_loc) )
       coulomb_done = .FALSE.
       IF ( ALLOCATED(cfac_done) ) cfac_done = .FALSE.   ! hinit1 reset the host cache
       gt_d_stale = .TRUE.
    ENDIF
    nks_loc = SIZE( coulomb_fac, 3 )
    cache_bytes = INT(ngmt,8) * INT(nqs,8) * INT(nks_loc,8) * 8_8
    cache_ok = ( cache_bytes <= 4000000000_8 ) .AND. ( SIZE(coulomb_fac,1) == ngmt ) .AND. ( SIZE(coulomb_fac,2) == nqs )
    IF ( cache_ok ) THEN            ! P11: allocate the resident cache here, falling back to the per-k kernel on device OOM
       IF ( ALLOCATED(cfac_all_d) ) THEN
          IF ( SIZE(cfac_all_d,1) /= ngmt .OR. SIZE(cfac_all_d,2) /= nqs .OR. SIZE(cfac_all_d,3) /= nks_loc ) &
             DEALLOCATE( cfac_all_d, cfac_done )
       ENDIF
       IF ( .NOT. ALLOCATED(cfac_all_d) ) THEN
          istat = cudaMemGetInfo( p15_free, p15_total )                      ! P15
          IF ( istat == 0 .AND. cache_bytes + 268435456_8 > p15_free ) THEN
             istat = 1                                                      ! not enough room: per-k kernel
          ELSE
             ALLOCATE( cfac_all_d(ngmt,nqs,nks_loc), STAT=istat )
          ENDIF
          IF ( istat /= 0 ) THEN
             cache_ok = .FALSE.
             IF ( ALLOCATED(cfac_all_d) ) DEALLOCATE( cfac_all_d )
             WRITE( stdout, '(5X,"EXX-GPU: no device memory for the Coulomb kernel cache (",I0," MB): per-k kernel from the host")' ) &
                  cache_bytes/1000000_8
          ELSE
             ALLOCATE( cfac_done(nks_loc) )
             cfac_done = .FALSE.
          ENDIF
       ENDIF
    ENDIF
    dev_ok = cache_ok .AND. ( .NOT. use_coulomb_vcut_ws ) .AND. ( .NOT. use_coulomb_vcut_spheric )
    ! P10: banner of the Coulomb-kernel decision (once, and again whenever it changes)
    p10_key = MERGE( 2, MERGE( 1, 0, cache_ok ), dev_ok )
    IF ( p10_key /= p10_kernel_last ) THEN
       p10_kernel_last = p10_key
       SELECT CASE ( p10_key )
       CASE ( 2 )
          WRITE( stdout, '(5X,"EXX-GPU: Coulomb kernel built on the device, cached for all k (",I0," MB); pair block ",I0," of ",I0," pairs")' ) &
               cache_bytes/1000000_8, pblk, npairs
       CASE ( 1 )
          WRITE( stdout, '(5X,"EXX-GPU: Coulomb kernel built on the host (vcut), cached on the device for all k (",I0," MB); pair block ",I0," of ",I0," pairs")' ) &
               cache_bytes/1000000_8, pblk, npairs
       CASE DEFAULT
          WRITE( stdout, '(5X,"EXX-GPU: Coulomb kernel cache too large or mismatched (",I0," MB): per-k kernel from the host; pair block ",I0," of ",I0," pairs")' ) &
               cache_bytes/1000000_8, pblk, npairs
       END SELECT
    ENDIF
    IF ( cache_ok ) THEN
       IF ( ALLOCATED(cfac_all_d) ) THEN
          IF ( SIZE(cfac_all_d,1) /= ngmt .OR. SIZE(cfac_all_d,2) /= nqs .OR. SIZE(cfac_all_d,3) /= nks_loc ) THEN
             DEALLOCATE( cfac_all_d, cfac_done )
          ENDIF
       ENDIF
       IF ( .NOT. ALLOCATED(cfac_all_d) ) THEN
          ALLOCATE( cfac_all_d(ngmt,nqs,nks_loc), cfac_done(nks_loc) )
          cfac_done = .FALSE.
       ENDIF
       IF ( .NOT. cfac_done(current_k) ) THEN
          IF ( dev_ok ) THEN
             ! ... EXX G vectors on the device (once; refreshed after a cache reset)
             IF ( ALLOCATED(gt_d) ) THEN
                IF ( SIZE(gt_d,2) /= ngmt ) THEN
                   DEALLOCATE( gt_d )
                   gt_d_stale = .TRUE.
                ENDIF
             ENDIF
             IF ( .NOT. ALLOCATED(gt_d) ) THEN
                ALLOCATE( gt_d(3,ngmt) )
                gt_d_stale = .TRUE.
             ENDIF
             IF ( gt_d_stale ) THEN
                gt_d = gt(1:3,1:ngmt)
                gt_d_stale = .FALSE.
             ENDIF
             ! ... the k+q vectors of this k
             IF ( ALLOCATED(p6_xkq_h) ) THEN
                IF ( SIZE(p6_xkq_h,2) /= nqs ) DEALLOCATE( p6_xkq_h, p6_xkq_d )
             ENDIF
             IF ( .NOT. ALLOCATED(p6_xkq_h) ) ALLOCATE( p6_xkq_h(3,nqs), p6_xkq_d(3,nqs) )
             DO iq = 1, nqs
                p6_xkq_h(:,iq) = xkq_collect(:,index_xkq(current_ik,iq))
             ENDDO
             p6_xkq_d = p6_xkq_h
             ! ... scalars of g2_convolution
             xk1 = xkp(1); xk2 = xkp(2); xk3 = xkp(3)
             tpiba2_ = tpiba2; e2_ = e2; fpi_ = fpi; pi_ = pi
             eps_ = eps_odg; eps_qdiv_ = eps_qdiv; exxdiv_ = exxdiv; gfac_ = grid_factor
             yuk_ = yukawa; erfc_ = erfc_scrlen; erf_ = erf_scrlen; gau_ = gau_scrlen
             xgamma = x_gamma_extrapolation
             nqh1 = DBLE(nq1)*0.5_DP; nqh2 = DBLE(nq2)*0.5_DP; nqh3 = DBLE(nq3)*0.5_DP
             a11 = at(1,1); a21 = at(2,1); a31 = at(3,1)
             a12 = at(1,2); a22 = at(2,2); a32 = at(3,2)
             a13 = at(1,3); a23 = at(2,3); a33 = at(3,3)
             !$cuf kernel do (2)
             DO iq = 1, nqs
                DO ig = 1, ngmt
                   q1 = xk1 - p6_xkq_d(1,iq) + gt_d(1,ig)
                   q2 = xk2 - p6_xkq_d(2,iq) + gt_d(2,ig)
                   q3 = xk3 - p6_xkq_d(3,iq) + gt_d(3,ig)
                   qq = ( q1*q1 + q2*q2 + q3*q3 ) * tpiba2_
                   gf = 1.0_DP
                   IF ( xgamma ) THEN
                      x = ( q1*a11 + q2*a21 + q3*a31 ) * nqh1
                      odg1 = ABS( x - NINT(x) ) < eps_
                      x = ( q1*a12 + q2*a22 + q3*a32 ) * nqh2
                      odg2 = ABS( x - NINT(x) ) < eps_
                      x = ( q1*a13 + q2*a23 + q3*a33 ) * nqh3
                      odg3 = ABS( x - NINT(x) ) < eps_
                      IF ( odg1 .AND. odg2 .AND. odg3 ) THEN
                         gf = 0.0_DP
                      ELSE
                         gf = gfac_
                      ENDIF
                   ENDIF
                   IF ( gau_ > 0.0_DP ) THEN
                      f = e2_ * ( (pi_/gau_)**1.5_DP ) * EXP( -qq/4.0_DP/gau_ ) * gf
                   ELSE IF ( qq > eps_qdiv_ ) THEN
                      IF ( erfc_ > 0.0_DP ) THEN
                         f = e2_*fpi_/qq * ( 1.0_DP - EXP( -qq/4.0_DP/erfc_**2 ) ) * gf
                      ELSE IF ( erf_ > 0.0_DP ) THEN
                         f = e2_*fpi_/qq * EXP( -qq/4.0_DP/erf_**2 ) * gf
                      ELSE
                         f = e2_*fpi_/( qq + yuk_ ) * gf
                      ENDIF
                   ELSE
                      f = - exxdiv_
                      IF ( yuk_ > 0.0_DP .AND. .NOT. xgamma ) f = f + e2_*fpi_/( qq + yuk_ )
                      IF ( erfc_ > 0.0_DP .AND. .NOT. xgamma ) f = f + e2_*pi_/( erfc_**2 )
                   ENDIF
                   cfac_all_d(ig,iq,current_k) = f
                ENDDO
             ENDDO
             coulomb_fac(:,:,current_k) = cfac_all_d(:,:,current_k)   ! host mirror for the other readers
             coulomb_done(:,current_k) = .TRUE.
          ELSE
             DO iq = 1, nqs
                ikq = index_xkq(current_ik,iq)
                xkq = xkq_collect(:,ikq)
                CALL g2_convolution_all( ngmt, gt, xkp, xkq, iq, current_k )
             ENDDO
             cfac_all_d(:,:,current_k) = coulomb_fac(:,:,current_k)
          ENDIF
          cfac_done(current_k) = .TRUE.
       ENDIF
       p6_facb_d = 0.0_DP
       !$cuf kernel do (2)
       DO iq = 1, nqs
          DO ig = 1, ngmt
             p6_facb_d( dfftt__nl(ig), iq ) = cfac_all_d(ig,iq,current_k)
          ENDDO
       ENDDO
    ELSE
       DO iq = 1, nqs
          ikq = index_xkq(current_ik,iq)
          xkq = xkq_collect(:,ikq)
          CALL g2_convolution_all( ngmt, gt, xkp, xkq, iq, current_k )   ! cached per (q,k)
       ENDDO
       IF ( ALLOCATED(p6_cfac_d) ) THEN
          IF ( SIZE(p6_cfac_d,1) /= ngmt .OR. SIZE(p6_cfac_d,2) /= nqs ) DEALLOCATE( p6_cfac_d )
       ENDIF
       IF ( .NOT. ALLOCATED(p6_cfac_d) ) ALLOCATE( p6_cfac_d(ngmt,nqs) )
       p6_cfac_d = coulomb_fac(1:ngmt,1:nqs,current_k)
       p6_facb_d = 0.0_DP
       !$cuf kernel do (2)
       DO iq = 1, nqs
          DO ig = 1, ngmt
             p6_facb_d( dfftt__nl(ig), iq ) = p6_cfac_d(ig,iq)
          ENDDO
       ENDDO
    ENDIF
    ! ... k+q indices and occupations of this k (P6c host work arrays)
    IF ( ALLOCATED(p6_ikq_h) ) THEN
       IF ( SIZE(p6_ikq_h) /= nqs .OR. SIZE(p6_xocc_h,1) /= nocc .OR. SIZE(p6_xocc_h,2) /= nqs ) &
          DEALLOCATE( p6_ikq_h, p6_xocc_h )
    ENDIF
    IF ( .NOT. ALLOCATED(p6_ikq_h) ) ALLOCATE( p6_ikq_h(nqs), p6_xocc_h(nocc,nqs) )
    DO iq = 1, nqs
       ikq = index_xkq(current_ik,iq)
       ik  = index_xk(ikq)
       p6_ikq_h(iq) = ikq
       p6_xocc_h(1:nocc,iq) = x_occupation(1:nocc,ik) * nqs_inv
    ENDDO
    p6_ikq_d  = p6_ikq_h
    p6_xocc_d = p6_xocc_h
    CALL stop_clock( 'vexx_k_cfac' )
    !
    ! ... psi -> device -> real space on the EXX grid (one batched inverse FFT)
    CALL start_clock( 'vexx_k_psi' )
    p6_psi_d  = psi(:,1:m)
    p6_hpsi_d = hpsi(:,1:m)
    p6_psic_d = ( 0._DP, 0._DP )
    !$cuf kernel do (2)
    DO ibnd = 1, m
       DO ig = 1, n
          p6_psic_d( dfftt__nl(igk_exx_d(ig,current_k)), ibnd ) = p6_psi_d(ig,ibnd)
       ENDDO
    ENDDO
    istat = cufftExecZ2Z( plan_wave, p6_psic_d(1,1), p6_psic_d(1,1), CUFFT_INVERSE )
    IF ( istat /= 0 ) CALL errore( 'vexx_bp_k_gpu_fast', 'cuFFT exec failed (P11 check)', ABS(istat) )
    p6_res_d = ( 0._DP, 0._DP )
    CALL stop_clock( 'vexx_k_psi' )
    IF ( okvan ) THEN          ! P18
       IF ( .NOT. PRESENT(becpsi) ) CALL errore( 'vexx_bp_k_gpu_fast', 'becpsi needed for the ultrasoft augmentation (P18)', 1 )
       CALL start_clock( 'vexx_k_us' )
       CALL p18_setup( current_k, current_ik, n, m, nocc, ngmt, pblk, xkp, becpsi, us_ok )
       CALL stop_clock( 'vexx_k_us' )
       IF ( .NOT. us_ok ) THEN
          CALL stop_clock( 'vexx_k_setup' )
          ok = .FALSE.
          RETURN
       ENDIF
    ENDIF
    CALL stop_clock( 'vexx_k_setup' )
    !
    CALL start_clock( 'vexx_k_main' )
    IF ( okvan .AND. p20_force_mode ) THEN          ! P20: the force pass (D matrix and structure-factor term, no potential)
       IF ( ALLOCATED(p20_fa_d) ) THEN
          IF ( SIZE(p20_fa_d,2) /= p18_na ) DEALLOCATE( p20_fa_d )
       ENDIF
       IF ( .NOT. ALLOCATED(p20_fa_d) ) ALLOCATE( p20_fa_d(3,p18_na) )
       p20_fa_d = 0.0_DP
       IF ( ALLOCATED(p20_vcg3_d) ) THEN
          IF ( SIZE(p20_vcg3_d,1) /= ngmt .OR. SIZE(p20_vcg3_d,2) /= 3*pblk .OR. SIZE(p20_y3_d,1) /= p18_na ) &
             DEALLOCATE( p20_vcg3_d, p20_y3_d )
       ENDIF
       IF ( .NOT. ALLOCATED(p20_vcg3_d) ) THEN
          p15_need = 48_8*INT(ngmt,8)*INT(pblk,8) + 48_8*INT(p18_na,8)*INT(pblk,8) + 16_8*INT(nkb,8)*INT(m,8)
          istat = cudaMemGetInfo( p15_free, p15_total )
          IF ( istat == 0 .AND. p15_need + 268435456_8 > p15_free ) THEN
             WRITE( stdout, '(5X,"EXX-GPU: the force pass would need ",I0," MB of device memory, ",I0," MB are free (P20)")' ) &
                  p15_need/1000000_8, p15_free/1000000_8
             CALL stop_clock( 'vexx_k_main' )
             ok = .FALSE.
             RETURN
          ENDIF
          ALLOCATE( p20_vcg3_d(ngmt,3*pblk), p20_y3_d(p18_na,3*pblk) )
       ENDIF
       CALL exx_force_main_us( nrxxs, nbuf, nkqs_d, exxbuff_d, m, p6_psic_d, nocc, nqs, p6_ikq_d, p6_facb_d, p6_xocc_d, &
                               npairs, pblk, p6_rhoc_d, plan_full, plan_tail, omega_inv, scale, ngmt, dfftt__nl, &
                               nkb, p18_na, nat, SIZE(p18_becxx_d,2), p18_becxx_d, p18_becpsi_d, p18_qs_d(:,:,:,p18_qs_slot), &
                               p18_eigq_d, p18_aat_d, p18_aih_d, p18_ajh_d, p18_aof_d, p18_ofs_d, p18_kbat_d, p18_kbih_d, &
                               p18_nhat_d, p18_c_d, p18_aug_d, p18_vcg_d, p18_y_d, p18_deexx_d, omega, nhm, &
                               p20_wg_h, xkp(1), xkp(2), xkp(3), p20_xkq_d, p20_gt_d, p20_tpiba, exxalfa, &
                               p20_vcg3_d, p20_y3_d, p20_fa_d, p20_ex )
    ELSE IF ( okvan ) THEN          ! P18
       CALL vexx_fast_main_us( nrxxs, nbuf, nkqs_d, exxbuff_d, m, p6_psic_d, nocc, nqs, p6_ikq_d, p6_facb_d, p6_xocc_d, &
                               npairs, pblk, p6_rhoc_d, p6_res_d, plan_full, plan_tail, omega_inv, scale, ngmt, dfftt__nl, &
                               nkb, p18_na, nat, SIZE(p18_becxx_d,2), p18_becxx_d, p18_becpsi_d, p18_qs_d(:,:,:,p18_qs_slot), &
                               p18_eigq_d, p18_aat_d, p18_aih_d, p18_ajh_d, p18_aof_d, p18_ofs_d, p18_kbat_d, p18_kbih_d, &
                               p18_nhat_d, p18_c_d, p18_aug_d, p18_vcg_d, p18_y_d, p18_deexx_d, omega, nhm )
    ELSE IF ( sp ) THEN
       CALL vexx_fast_main_sp( nrxxs, nbuf, nkqs_d, exxbuff_d, m, p6_psic_d, nocc, nqs, p6_ikq_d, &
                               p6_facb_d, p6_xocc_d, npairs, pblk, p6_rhoc_s, p6_res_d, &
                               plan_full, plan_tail, omega_inv, scale )
    ELSE
       CALL vexx_fast_main_dp( nrxxs, nbuf, nkqs_d, exxbuff_d, m, p6_psic_d, nocc, nqs, p6_ikq_d, &
                               p6_facb_d, p6_xocc_d, npairs, pblk, p6_rhoc_d, p6_res_d, &
                               plan_full, plan_tail, omega_inv, scale )
    ENDIF
    !
    ! ... back to G-space (one batched forward FFT, normalised here) and into hpsi
    istat = cufftExecZ2Z( plan_wave, p6_res_d(1,1), p6_res_d(1,1), CUFFT_FORWARD )
    IF ( istat /= 0 ) CALL errore( 'vexx_bp_k_gpu_fast', 'cuFFT exec failed (P11 check)', ABS(istat) )
    !$cuf kernel do (2)
    DO ibnd = 1, m
       DO ig = 1, n
          p6_hpsi_d(ig,ibnd) = p6_hpsi_d(ig,ibnd) - exxalfa * scale * p6_res_d( dfftt__nl(igk_exx_d(ig,current_k)), ibnd )
       ENDDO
    ENDDO
    IF ( okvan ) THEN          ! P18: the non-local part, sum_I |beta_I> deexx_Ii (add_nlxx_pot as one ZGEMM)
       IF ( okpaw ) CALL ZGEMM( 'N', 'N', nkb, m, nkb, (1.0_DP,0.0_DP), p18_dpaw_d(1,1,current_k), nkb, p18_becpsi_d, nkb, &
                                (1.0_DP,0.0_DP), p18_deexx_d, nkb )
       IF ( p20_force_mode ) THEN        ! P20: the force pass wants the exchange D matrix only, no potential
          IF ( okpaw ) THEN               ! P20: the PAW one-centre part of E_x, -(alpha/2) sum_i w_i b_i^H D_k b_i
             IF ( ALLOCATED(p20_dpb_d) ) THEN
                IF ( SIZE(p20_dpb_d,1) /= nkb .OR. SIZE(p20_dpb_d,2) /= m ) DEALLOCATE( p20_dpb_d )
             ENDIF
             IF ( .NOT. ALLOCATED(p20_dpb_d) ) ALLOCATE( p20_dpb_d(nkb,m) )
             CALL ZGEMM( 'N', 'N', nkb, m, nkb, (1.0_DP,0.0_DP), p18_dpaw_d(1,1,current_k), nkb, p18_becpsi_d, nkb, &
                         (0.0_DP,0.0_DP), p20_dpb_d, nkb )
             p20_epaw = 0.0_DP
             !$cuf kernel do (2)
             DO ibnd = 1, m
                DO ig = 1, nkb
                   p20_epaw = p20_epaw + p20_wg_d(ibnd) * DBLE( CONJG( p18_becpsi_d(ig,ibnd) ) * p20_dpb_d(ig,ibnd) )
                ENDDO
             ENDDO
             p20_ex = p20_ex - 0.5_DP * exxalfa * p20_epaw
          ENDIF
          CALL stop_clock( 'vexx_k_main' )
          RETURN
       ENDIF
       !$cuf kernel do (2)
       DO ibnd = 1, m
          DO ig = 1, nkb
             IF ( ABS( p18_deexx_d(ig,ibnd) ) < eps_occ ) p18_deexx_d(ig,ibnd) = ( 0.0_DP, 0.0_DP )
          ENDDO
       ENDDO
       CALL ZGEMM( 'N', 'N', n, m, nkb, CMPLX(-exxalfa,0.0_DP,KIND=DP), p18_vkb_d, npwx, p18_deexx_d, nkb, (1.0_DP,0.0_DP), p6_hpsi_d, lda*npol )
    ENDIF
    CALL stop_clock( 'vexx_k_main' )
    !
    CALL start_clock( 'vexx_k_fin' )
    hpsi(:,1:m) = p6_hpsi_d
    CALL stop_clock( 'vexx_k_fin' )
    !
  END SUBROUTINE vexx_bp_k_gpu_fast
  !
  !-----------------------------------------------------------------------
  SUBROUTINE p18_qgm_build( ngms, xkq, xkp )
  !-----------------------------------------------------------------------
    !! P18: Q_ij(k-k'+G) of every ultrasoft (ih <= jh) pair for one (k, k+q) into p18_qgm_acc, on the device: the
    !! qvan_init recipe of us_exx.f90 -- q and |q| on the host (3 N_G numbers), Y_lm(q) by upflib's OpenACC ylmr2,
    !! then one fused kernel over every (pair, G) interpolating tab_qrad (the body of qvan2.f90, its Clebsch-Gordan
    !! factors tabulated once in p18_setup). Nothing but q crosses the bus.
    USE cell_base,  ONLY : tpiba, omega
    USE uspp_param, ONLY : lmaxq
    USE gvect,      ONLY : g
    USE qrad_mod,   ONLY : dq, tab_qrad
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: ngms
    REAL(DP), INTENT(IN) :: xkq(3), xkp(3)
    REAL(DP), ALLOCATABLE :: ylmk0(:,:), qmod(:), q(:,:), qq(:)
    INTEGER :: ig
    ALLOCATE( q(3,ngms), qq(ngms), qmod(ngms), ylmk0(ngms,lmaxq*lmaxq) )
    DO ig = 1, ngms
       q(:,ig) = xkp(:) - xkq(:) + g(:,ig)
       qq(ig)  = SUM( q(:,ig)**2 )
       qmod(ig) = SQRT( qq(ig) ) * tpiba
    ENDDO
    !$acc data copyin( q, qq, qmod ) create( ylmk0 ) present( p18_qgm_acc, tab_qrad )
    CALL ylmr2( lmaxq*lmaxq, ngms, q, qq, ylmk0 )
    !$acc host_data use_device( qmod, ylmk0, tab_qrad, p18_qgm_acc )
    CALL p18_qvan_all( ngms, p18_nij, lmaxq*lmaxq, SIZE(tab_qrad,1), SIZE(tab_qrad,2), SIZE(tab_qrad,3), SIZE(tab_qrad,4), &
                       p18_qc_mx, 1.0_DP/dq, omega, qmod, ylmk0, tab_qrad, p18_qc_ijv_d, p18_qc_nt_d, p18_qc_nlm_d, &
                       p18_qc_lp_d, p18_qc_l_d, p18_qc_ind_d, p18_qc_coef_d, p18_qgm_acc )
    !$acc end host_data
    !$acc end data
    DEALLOCATE( q, qq, qmod, ylmk0 )
  END SUBROUTINE p18_qgm_build
  !
  !-----------------------------------------------------------------------
  SUBROUTINE p18_qvan_all( ngms, nij, nlm2, nqx, nijt, nlq, nsp_, mx, dqi, omega, qmod, ylm, tab, ijv, nt, nlm, lp, lq, ind, &
                           coef, qgm )
  !-----------------------------------------------------------------------
    !! P18: qvan2.f90 for every pair column at once. Q_c(G) = sum_lm (-i)^l ap(lp,ivl,jvl) Y_lp(q) Qrad_l(|q|) with the
    !! four-point interpolation of tab_qrad; (-i)^l ap is coef(lm,c), ind(lm,c) = 1/2 the real/imaginary slot, lq the
    !! l+1 index of tab_qrad -- the same arithmetic, in the same order, as the host loop of qvan2, including its
    !! division by the cell volume (tab_qrad carries 4 pi only).
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: ngms, nij, nlm2, nqx, nijt, nlq, nsp_, mx
    REAL(DP), INTENT(IN) :: dqi, omega
    REAL(DP), DEVICE, INTENT(IN) :: qmod(ngms), ylm(ngms,nlm2), tab(nqx,nijt,nlq,nsp_), coef(mx,nij)
    INTEGER, DEVICE, INTENT(IN) :: ijv(nij), nt(nij), nlm(nij), lp(mx,nij), lq(mx,nij), ind(mx,nij)
    COMPLEX(DP), DEVICE, INTENT(OUT) :: qgm(ngms,nij)
    REAL(DP), PARAMETER :: sixth = 1.0_DP / 6.0_DP
    INTEGER :: c, ig, lm, i0, i1, i2, i3, l, kp
    REAL(DP) :: qm, px, ux, vx, wx, uvx, pwx, work, re, im, v
    !$cuf kernel do (2)
    DO c = 1, nij
       DO ig = 1, ngms
          qm = qmod(ig) * dqi
          px = qm - INT(qm)
          ux = 1.0_DP - px
          vx = 2.0_DP - px
          wx = 3.0_DP - px
          i0 = INT(qm) + 1
          i1 = i0 + 1
          i2 = i0 + 2
          i3 = i0 + 3
          uvx = ux * vx * sixth
          pwx = px * wx * 0.5_DP
          re = 0.0_DP
          im = 0.0_DP
          DO lm = 1, nlm(c)
             l = lq(lm,c)
             kp = lp(lm,c)
             work = ( tab(i0,ijv(c),l,nt(c)) * uvx * wx + tab(i1,ijv(c),l,nt(c)) * pwx * vx - &
                      tab(i2,ijv(c),l,nt(c)) * pwx * ux + tab(i3,ijv(c),l,nt(c)) * px * uvx ) / omega
             v = coef(lm,c) * ylm(ig,kp) * work
             IF ( ind(lm,c) == 1 ) THEN
                re = re + v
             ELSE
                im = im + v
             ENDIF
          ENDDO
          qgm(ig,c) = CMPLX( re, im, KIND=DP )
       ENDDO
    ENDDO
  END SUBROUTINE p18_qvan_all
  !
  !-----------------------------------------------------------------------
  SUBROUTINE p18_release()
  !-----------------------------------------------------------------------
    !! P18: free every device (and mirror) array of the ultrasoft path and forget the tables: after an allocation
    !! failure the original routine takes over for the rest of the run (P15 policy), also from p15_release_work_arrays
    IMPLICIT NONE
    IF ( ALLOCATED(p18_acol_h) ) DEALLOCATE( p18_acol_h, p18_aat_h, p18_aih_h, p18_ajh_h, p18_aof_h, p18_nijt_h )
    IF ( ALLOCATED(p18_acol_d) ) DEALLOCATE( p18_acol_d, p18_aat_d, p18_aih_d, p18_ajh_d, p18_aof_d, p18_ofs_d, &
                                             p18_kbat_d, p18_kbih_d, p18_nhat_d )
    IF ( ALLOCATED(p18_qc_ijv_d) ) DEALLOCATE( p18_qc_ijv_d, p18_qc_nt_d, p18_qc_nlm_d, p18_qc_lp_d, p18_qc_l_d, p18_qc_ind_d, p18_qc_coef_d )
    IF ( ALLOCATED(p18_becxx_d) ) DEALLOCATE( p18_becxx_d )
    IF ( ALLOCATED(p18_sf_d) ) DEALLOCATE( p18_sf_d )
    IF ( ALLOCATED(p18_sf_h) ) DEALLOCATE( p18_sf_h )
    IF ( ALLOCATED(p18_eigq_h) ) DEALLOCATE( p18_eigq_h, p18_eigq_d )
    IF ( ALLOCATED(p18_qs_d) ) DEALLOCATE( p18_qs_d )
    IF ( ALLOCATED(p18_qs_done) ) DEALLOCATE( p18_qs_done )
    IF ( ALLOCATED(p18_qgm_acc) ) THEN
       !$acc exit data delete( p18_qgm_acc )
       DEALLOCATE( p18_qgm_acc )
    ENDIF
    IF ( ALLOCATED(p18_aug_d) ) DEALLOCATE( p18_aug_d, p18_vcg_d, p18_c_d, p18_y_d )
    IF ( ALLOCATED(p18_vkb_h) ) DEALLOCATE( p18_vkb_h, p18_vkb_d )
    IF ( ALLOCATED(p18_becpsi_d) ) DEALLOCATE( p18_becpsi_d, p18_deexx_d )
    IF ( ALLOCATED(p18_dpaw_d) ) DEALLOCATE( p18_dpaw_d, p18_paw_done )
    IF ( ALLOCATED(p18_dpaw_h) ) DEALLOCATE( p18_dpaw_h )
    IF ( ALLOCATED(p18_geo_h) ) DEALLOCATE( p18_geo_h )
    p18_na = 0; p18_nij = 0; p18_nkb = 0; p18_nat = 0; p18_qc_mx = 0
    p18_bec_build = -1; p18_sf_build = -1; p18_paw_build = -1; p18_qs_k = -1; p18_qs_cached = .FALSE.
  END SUBROUTINE p18_release
  !
  !-----------------------------------------------------------------------
  SUBROUTINE p18_qs_fill( ngms, nij, na_, qgm, sf, acol, aat, qs )
  !-----------------------------------------------------------------------
    !! P18: Q(G,A) = qgm(G, column of the (ih,jh) pair of A) * e^{-iG.tau_atom(A)}
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: ngms, nij, na_
    COMPLEX(DP), DEVICE, INTENT(IN) :: qgm(ngms,nij), sf(ngms,*)
    INTEGER, DEVICE, INTENT(IN) :: acol(na_), aat(na_)
    COMPLEX(DP), DEVICE, INTENT(OUT) :: qs(ngms,na_)
    INTEGER :: ia, ig
    !$cuf kernel do (2)
    DO ia = 1, na_
       DO ig = 1, ngms
          qs(ig,ia) = qgm(ig,acol(ia)) * sf(ig,aat(ia))
       ENDDO
    ENDDO
  END SUBROUTINE p18_qs_fill
  !
  !-----------------------------------------------------------------------
  SUBROUTINE p18_setup( current_k, current_ik, n, m, nocc, ngmt, pblk, xkp, becpsi, ok )
  !-----------------------------------------------------------------------
    !! P18: everything the augmentation needs for this vexx call on the device -- pair tables, becxx of every k+q,
    !! structure factors, phases, Q(G,A) of the current k (or of every k, cached), the projectors and becpsi of k,
    !! the PAW one-center matrix of k. ok = .FALSE.: no room on the device -- every P18 array is released and the
    !! fast path is switched off for the rest of the run (P15 policy); the original routine serves the call.
    USE cudafor,     ONLY : cudaMemGetInfo
    USE constants,   ONLY : tpi
    USE cell_base,   ONLY : at, alat
    USE ions_base,   ONLY : nat, ntyp => nsp, ityp, tau
    USE uspp,        ONLY : nkb, ofsbeta, ijtoh, indv, nhtolm, lpx, lpl, ap
    USE uspp_param,  ONLY : upf, nh, nhm
    USE lsda_mod,    ONLY : lsda
    USE uspp_init,   ONLY : init_us_2
    USE gvect,       ONLY : eigts1, eigts2, eigts3, mill
    USE paw_variables, ONLY : okpaw
    USE paw_exx,     ONLY : ke
    USE becmod,      ONLY : bec_type
    USE klist,       ONLY : nks, xk   ! P20: xk
    USE wvfct,       ONLY : nbnd, npwx
    USE exx_base,    ONLY : nqs, index_xkq, index_xk, xkq_collect
    USE exx_bp_utils, ONLY : igk_exx
    USE us_exx,      ONLY : becxx
    USE cublas   ! P20
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: current_k, current_ik, n, m, nocc, ngmt, pblk
    REAL(DP), INTENT(IN) :: xkp(3)
    TYPE(bec_type), INTENT(IN) :: becpsi
    LOGICAL, INTENT(OUT) :: ok
    INTEGER :: na, nt, ih, jh, oh, uh, ia, nij, ikq, ik, iq, jb, ig, ikb, istat, nkqs_d, slot, nslot, i, ngeo
    INTEGER :: nkx, kv, ijh, nb, mb, ivl, jvl, lm, lp, l, ind, jk, ikv   ! P20: jk, ikv
    INTEGER, ALLOCATABLE :: kslot(:)   ! P20
    REAL(DP) :: sig
    INTEGER, ALLOCATABLE :: kbat_h(:), kbih_h(:), nhat_h(:), ofs_h(:)
    INTEGER, ALLOCATABLE :: qc_ijv(:), qc_nt(:), qc_nlm(:), qc_lp(:,:), qc_l(:,:), qc_ind(:,:)
    REAL(DP), ALLOCATABLE :: qc_coef(:,:)
    INTEGER(KIND=8) :: need, free_, total_
    REAL(DP) :: arg, w
    REAL(DP), ALLOCATABLE :: geo(:)
    COMPLEX(DP) :: z
    LOGICAL :: rebuild, fits
    CHARACTER(LEN=8) :: envval
    INTEGER :: lenv, estat
    !
    ok = .TRUE.
    nkqs_d = SIZE( exxbuff_d, 3 )
    !
    ! ... the (atom, ih <= jh) pair table of the ultrasoft species (once; again if the system changed)
    IF ( p18_na == 0 .OR. p18_nkb /= nkb .OR. p18_nat /= nat ) THEN
       IF ( ALLOCATED(p18_acol_h) ) DEALLOCATE( p18_acol_h, p18_aat_h, p18_aih_h, p18_ajh_h, p18_aof_h, p18_nijt_h )
       IF ( ALLOCATED(p18_acol_d) ) DEALLOCATE( p18_acol_d, p18_aat_d, p18_aih_d, p18_ajh_d, p18_aof_d, p18_ofs_d, &
                                                p18_kbat_d, p18_kbih_d, p18_nhat_d )
       ALLOCATE( p18_nijt_h(ntyp) )
       nij = 0
       DO nt = 1, ntyp
          p18_nijt_h(nt) = nij
          IF ( upf(nt)%tvanp ) nij = nij + ( nh(nt)*(nh(nt)+1) )/2
       ENDDO
       p18_nij = nij
       p18_na = 0
       DO na = 1, nat
          nt = ityp(na)
          IF ( upf(nt)%tvanp ) p18_na = p18_na + ( nh(nt)*(nh(nt)+1) )/2
       ENDDO
       ALLOCATE( p18_acol_h(p18_na), p18_aat_h(p18_na), p18_aih_h(p18_na), p18_ajh_h(p18_na), p18_aof_h(nhm,nhm,nat) )
       ALLOCATE( kbat_h(nkb), kbih_h(nkb), nhat_h(nat), ofs_h(nat) )
       p18_aof_h = 0; kbat_h = 0; kbih_h = 0
       ia = 0
       DO na = 1, nat
          nt = ityp(na)
          nhat_h(na) = nh(nt)
          ofs_h(na) = ofsbeta(na)
          IF ( .NOT. upf(nt)%tvanp ) CYCLE
          DO ih = 1, nh(nt)
             kbat_h(ofsbeta(na)+ih) = na
             kbih_h(ofsbeta(na)+ih) = ih
             DO jh = ih, nh(nt)
                ia = ia + 1
                p18_acol_h(ia) = p18_nijt_h(nt) + ijtoh(ih,jh,nt)
                p18_aat_h(ia) = na
                p18_aih_h(ia) = ih
                p18_ajh_h(ia) = jh
                p18_aof_h(ih,jh,na) = ia
                p18_aof_h(jh,ih,na) = ia
             ENDDO
          ENDDO
       ENDDO
       ALLOCATE( p18_acol_d(p18_na), p18_aat_d(p18_na), p18_aih_d(p18_na), p18_ajh_d(p18_na), p18_aof_d(nhm,nhm,nat), &
                 p18_ofs_d(nat), p18_kbat_d(nkb), p18_kbih_d(nkb), p18_nhat_d(nat) )
       p18_acol_d = p18_acol_h; p18_aat_d = p18_aat_h; p18_aih_d = p18_aih_h; p18_ajh_d = p18_ajh_h
       p18_aof_d = p18_aof_h; p18_ofs_d = ofs_h; p18_kbat_d = kbat_h; p18_kbih_d = kbih_h; p18_nhat_d = nhat_h
       DEALLOCATE( kbat_h, kbih_h, nhat_h, ofs_h )
       ! the constants of qvan2.f90 for every pair column (nt-major, ih, jh >= ih: the order of ijtoh and of the
       ! Q(G,A) columns): ijv and species of tab_qrad, the lp list with l+1, the (-i)^l ap(lp,ivl,jvl) factors
       IF ( ALLOCATED(p18_qc_ijv_d) ) DEALLOCATE( p18_qc_ijv_d, p18_qc_nt_d, p18_qc_nlm_d, p18_qc_lp_d, p18_qc_l_d, p18_qc_ind_d, p18_qc_coef_d )
       ALLOCATE( qc_ijv(p18_nij), qc_nt(p18_nij), qc_nlm(p18_nij) )
       p18_qc_mx = 1
       ijh = 0
       DO nt = 1, ntyp
          IF ( .NOT. upf(nt)%tvanp ) CYCLE
          DO ih = 1, nh(nt)
             DO jh = ih, nh(nt)
                ijh = ijh + 1
                nb = indv(ih,nt)
                mb = indv(jh,nt)
                qc_ijv(ijh) = MERGE( nb*(nb-1)/2 + mb, mb*(mb-1)/2 + nb, nb >= mb )
                qc_nt(ijh) = nt
                qc_nlm(ijh) = lpx( nhtolm(ih,nt), nhtolm(jh,nt) )
                p18_qc_mx = MAX( p18_qc_mx, qc_nlm(ijh) )
             ENDDO
          ENDDO
       ENDDO
       ALLOCATE( qc_lp(p18_qc_mx,p18_nij), qc_l(p18_qc_mx,p18_nij), qc_ind(p18_qc_mx,p18_nij), qc_coef(p18_qc_mx,p18_nij) )
       qc_lp = 1; qc_l = 1; qc_ind = 1; qc_coef = 0.0_DP
       ijh = 0
       DO nt = 1, ntyp
          IF ( .NOT. upf(nt)%tvanp ) CYCLE
          DO ih = 1, nh(nt)
             DO jh = ih, nh(nt)
                ijh = ijh + 1
                ivl = nhtolm(ih,nt)
                jvl = nhtolm(jh,nt)
                DO lm = 1, lpx(ivl,jvl)
                   lp = lpl(ivl,jvl,lm)
                   IF ( lp < 1 .OR. lp > 49 ) CALL errore( 'p18_setup', 'lp wrong', MAX(lp,1) )
                   IF ( lp == 1 ) THEN            ! l+1 and (-i)^l of the combined index lp, as qvan2.f90 has them
                      l = 1; sig = 1.0_DP; ind = 1
                   ELSEIF ( lp <= 4 ) THEN
                      l = 2; sig = -1.0_DP; ind = 2
                   ELSEIF ( lp <= 9 ) THEN
                      l = 3; sig = -1.0_DP; ind = 1
                   ELSEIF ( lp <= 16 ) THEN
                      l = 4; sig = 1.0_DP; ind = 2
                   ELSEIF ( lp <= 25 ) THEN
                      l = 5; sig = 1.0_DP; ind = 1
                   ELSEIF ( lp <= 36 ) THEN
                      l = 6; sig = -1.0_DP; ind = 2
                   ELSE
                      l = 7; sig = -1.0_DP; ind = 1
                   ENDIF
                   qc_lp(lm,ijh) = lp
                   qc_l(lm,ijh) = l
                   qc_ind(lm,ijh) = ind
                   qc_coef(lm,ijh) = sig * ap(lp,ivl,jvl)
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       ALLOCATE( p18_qc_ijv_d(p18_nij), p18_qc_nt_d(p18_nij), p18_qc_nlm_d(p18_nij), p18_qc_lp_d(p18_qc_mx,p18_nij), &
                 p18_qc_l_d(p18_qc_mx,p18_nij), p18_qc_ind_d(p18_qc_mx,p18_nij), p18_qc_coef_d(p18_qc_mx,p18_nij) )
       p18_qc_ijv_d = qc_ijv; p18_qc_nt_d = qc_nt; p18_qc_nlm_d = qc_nlm
       p18_qc_lp_d = qc_lp; p18_qc_l_d = qc_l; p18_qc_ind_d = qc_ind; p18_qc_coef_d = qc_coef
       DEALLOCATE( qc_ijv, qc_nt, qc_nlm, qc_lp, qc_l, qc_ind, qc_coef )
       p18_nkb = nkb; p18_nat = nat
       p18_bec_build = -1; p18_sf_build = -1; p18_paw_build = -1; p18_qs_k = -1
       IF ( ALLOCATED(p18_qs_done) ) p18_qs_done = .FALSE.
    ENDIF
    !
    ! ... room for what this call has to allocate (P15 style; arrays already resident are not charged again):
    !     Q(G,A) of one k, the block work arrays, projectors, becpsi/deexx, becxx, structure factors, Q_ij(G), PAW D_k
    need = 0_8
    IF ( .NOT. ALLOCATED(p18_qs_d) ) THEN
       need = need + 16_8*INT(ngmt,8)*INT(p18_na,8)*INT(nqs,8)
    ELSE IF ( SIZE(p18_qs_d,1) /= ngmt .OR. SIZE(p18_qs_d,2) /= p18_na .OR. SIZE(p18_qs_d,3) /= nqs ) THEN
       need = need + 16_8*INT(ngmt,8)*INT(p18_na,8)*INT(nqs,8)
    ENDIF
    IF ( .NOT. ALLOCATED(p18_aug_d) ) THEN
       need = need + 32_8*INT(ngmt,8)*INT(pblk,8) + 32_8*INT(p18_na,8)*INT(pblk,8)
    ELSE IF ( SIZE(p18_aug_d,1) /= ngmt .OR. SIZE(p18_aug_d,2) /= pblk .OR. SIZE(p18_c_d,1) /= p18_na ) THEN
       need = need + 32_8*INT(ngmt,8)*INT(pblk,8) + 32_8*INT(p18_na,8)*INT(pblk,8)
    ENDIF
    IF ( .NOT. ALLOCATED(p18_vkb_d) ) THEN
       need = need + 16_8*INT(npwx,8)*INT(nkb,8)
    ELSE IF ( SIZE(p18_vkb_d,1) /= npwx .OR. SIZE(p18_vkb_d,2) /= nkb ) THEN
       need = need + 16_8*INT(npwx,8)*INT(nkb,8)
    ENDIF
    IF ( .NOT. ALLOCATED(p18_becpsi_d) ) THEN
       need = need + 32_8*INT(nkb,8)*INT(m,8)
    ELSE IF ( SIZE(p18_becpsi_d,1) /= nkb .OR. SIZE(p18_becpsi_d,2) /= m ) THEN
       need = need + 32_8*INT(nkb,8)*INT(m,8)
    ENDIF
    IF ( .NOT. ALLOCATED(p18_becxx_d) ) need = need + 16_8*INT(nkb,8)*INT(nbnd,8)*INT(nkqs_d,8)
    IF ( .NOT. ALLOCATED(p18_sf_d) ) need = need + 16_8*INT(ngmt,8)*INT(nat,8)
    IF ( .NOT. ALLOCATED(p18_qgm_acc) ) need = need + 16_8*INT(ngmt,8)*INT(p18_nij,8)
    IF ( okpaw .AND. .NOT. ALLOCATED(p18_dpaw_d) ) need = need + 16_8*INT(nkb,8)*INT(nkb,8)*INT(nks,8)
    istat = cudaMemGetInfo( free_, total_ )
    IF ( istat == 0 .AND. need + 268435456_8 > free_ ) THEN
       WRITE( stdout, '(5X,"EXX-GPU: ultrasoft augmentation would need ",I0," MB of device memory, ",I0," MB are free: this run continues on the original vexx_k_gpu path (P18)")' ) &
            need/1000000_8, free_/1000000_8
       CALL p18_release()
       p15_fast_disabled = .TRUE.
       ok = .FALSE.
       RETURN
    ENDIF
    !
    ! ... becxx of every k+q orbital (rebuilt by exxinit)
    IF ( p18_bec_build /= p6_nbuild .OR. .NOT. ALLOCATED(p18_becxx_d) ) THEN
       CALL start_clock( 'p18_bec' )
       IF ( ALLOCATED(p18_becxx_d) ) THEN
          IF ( SIZE(p18_becxx_d,1) /= nkb .OR. SIZE(p18_becxx_d,2) /= nbnd .OR. SIZE(p18_becxx_d,3) /= nkqs_d ) DEALLOCATE( p18_becxx_d )
       ENDIF
       IF ( .NOT. ALLOCATED(p18_becxx_d) ) ALLOCATE( p18_becxx_d(nkb,nbnd,nkqs_d) )
       DO ikq = 1, nkqs_d
          p18_becxx_d(:,:,ikq) = becxx(ikq)%k(1:nkb,1:nbnd)
       ENDDO
       p18_bec_build = p6_nbuild
       CALL stop_clock( 'p18_bec' )
    ENDIF
    !
    ! ... structure factors e^{-iG.tau} on the EXX G list (the positions may have moved since the last build)
    IF ( p18_sf_build /= p6_nbuild .OR. .NOT. ALLOCATED(p18_sf_d) ) THEN
       IF ( ALLOCATED(p18_sf_d) ) THEN
          IF ( SIZE(p18_sf_d,1) /= ngmt .OR. SIZE(p18_sf_d,2) /= nat ) DEALLOCATE( p18_sf_d )
       ENDIF
       IF ( .NOT. ALLOCATED(p18_sf_d) ) ALLOCATE( p18_sf_d(ngmt,nat) )
       IF ( ALLOCATED(p18_sf_h) ) THEN
          IF ( SIZE(p18_sf_h,1) /= ngmt .OR. SIZE(p18_sf_h,2) /= nat ) DEALLOCATE( p18_sf_h )
       ENDIF
       IF ( .NOT. ALLOCATED(p18_sf_h) ) ALLOCATE( p18_sf_h(ngmt,nat) )
       DO na = 1, nat
          DO ig = 1, ngmt
             p18_sf_h(ig,na) = eigts1(mill(1,ig),na) * eigts2(mill(2,ig),na) * eigts3(mill(3,ig),na)
          ENDDO
       ENDDO
       p18_sf_d = p18_sf_h
       p18_sf_build = p6_nbuild
    ENDIF
    !
    ! ... phases e^{-i(k-k').tau} of the q-points of this k
    IF ( ALLOCATED(p18_eigq_h) ) THEN
       IF ( SIZE(p18_eigq_h,1) /= nat .OR. SIZE(p18_eigq_h,2) /= nqs ) DEALLOCATE( p18_eigq_h, p18_eigq_d )
    ENDIF
    IF ( .NOT. ALLOCATED(p18_eigq_h) ) ALLOCATE( p18_eigq_h(nat,nqs), p18_eigq_d(nat,nqs) )
    DO iq = 1, nqs
       ikq = index_xkq(current_ik,iq)
       DO na = 1, nat
          arg = tpi * SUM( ( xkp(:) - xkq_collect(:,ikq) ) * tau(:,na) )
          p18_eigq_h(na,iq) = CMPLX( COS(arg), -SIN(arg), KIND=DP )
       ENDDO
    ENDDO
    p18_eigq_d = p18_eigq_h
    !
    ! ... Q(G,A) of this k: from the cache of every k while the geometry is unchanged, else rebuilt now
    ngeo = 9 + 3*nat + 1
    ALLOCATE( geo(ngeo) )
    geo(1:9) = RESHAPE( at, (/9/) ) * alat
    geo(10:9+3*nat) = RESHAPE( tau, (/3*nat/) ) * alat
    geo(ngeo) = DBLE( ngmt )
    rebuild = .NOT. ALLOCATED(p18_geo_h)
    IF ( .NOT. rebuild ) rebuild = ( SIZE(p18_geo_h) /= ngeo )
    IF ( .NOT. rebuild ) rebuild = ANY( ABS( p18_geo_h - geo ) > 1.0E-10_DP )
    IF ( rebuild ) THEN
       IF ( ALLOCATED(p18_geo_h) ) DEALLOCATE( p18_geo_h )
       ALLOCATE( p18_geo_h(ngeo) ); p18_geo_h = geo
       IF ( ALLOCATED(p18_qs_done) ) p18_qs_done = .FALSE.
       p18_qs_k = -1
    ENDIF
    DEALLOCATE( geo )
    ! one Q slot per k-vector (the spin-down k of an LSDA run shares the slot of its spin-up twin: same xk, same
    ! k+q list); every k cached when that fits a quarter of the device memory (the buffer, the kernel cache and the
    ! work arrays share it), otherwise one slot rebuilt whenever the k-vector changes. Decided once per shape: the
    ! free memory seen on later calls is smaller by the cache itself, so re-deciding every call could thrash.
    ! P20: slots by k-vector (the spin-down k of an LSDA run shares the slot of its spin-up twin when both are in this
    ! pool; with k-point pools a pool may hold one spin only, so the twin is found by comparing xk, not by position)
    ALLOCATE( kslot(nks) )
    nkx = 0
    DO ik = 1, nks
       ikv = 0
       DO jk = 1, ik - 1
          IF ( ALL( ABS( xk(:,jk) - xk(:,ik) ) < 1.0E-8_DP ) ) THEN
             ikv = jk
             EXIT
          ENDIF
       ENDDO
       IF ( ikv == 0 ) THEN
          nkx = nkx + 1
          kslot(ik) = nkx
       ELSE
          kslot(ik) = kslot(ikv)
       ENDIF
    ENDDO
    kv = kslot(current_k)
    DEALLOCATE( kslot )
    IF ( ALLOCATED(p18_qs_d) ) THEN
       IF ( SIZE(p18_qs_d,1) /= ngmt .OR. SIZE(p18_qs_d,2) /= p18_na .OR. SIZE(p18_qs_d,3) /= nqs .OR. &
            ( p18_qs_cached .AND. SIZE(p18_qs_d,4) /= nkx ) ) THEN
          DEALLOCATE( p18_qs_d )
          IF ( ALLOCATED(p18_qs_done) ) DEALLOCATE( p18_qs_done )
          p18_qs_k = -1
       ENDIF
    ENDIF
    IF ( .NOT. ALLOCATED(p18_qs_d) ) THEN
       fits = .FALSE.
       IF ( istat == 0 ) fits = ( 16_8*INT(ngmt,8)*INT(p18_na,8)*INT(nqs,8)*INT(nkx,8) + 268435456_8 < free_/2 ) .AND. &
                                ( 16_8*INT(ngmt,8)*INT(p18_na,8)*INT(nqs,8)*INT(nkx,8) < total_/4 )
       CALL GET_ENVIRONMENT_VARIABLE( 'QE_EXX_US_QCACHE', envval, lenv, estat )   ! =0 forces the per-call rebuild (testing)
       IF ( estat == 0 .AND. lenv > 0 ) THEN
          IF ( TRIM(envval) == '0' ) fits = .FALSE.
       ENDIF
       nslot = MERGE( nkx, 1, fits )
       ALLOCATE( p18_qs_d(ngmt,p18_na,nqs,nslot), STAT=istat )
       IF ( istat /= 0 ) THEN
          WRITE( stdout, '(5X,"EXX-GPU: no device memory for Q(G,A) (",I0," MB): this run continues on the original vexx_k_gpu path (P18)")' ) &
               16_8*INT(ngmt,8)*INT(p18_na,8)*INT(nqs,8)*INT(nslot,8)/1000000_8
          CALL p18_release()
          p15_fast_disabled = .TRUE.
          ok = .FALSE.
          RETURN
       ENDIF
       ALLOCATE( p18_qs_done(nslot) ); p18_qs_done = .FALSE.
       p18_qs_cached = fits
       p18_qs_k = -1
    ENDIF
    IF ( ALLOCATED(p18_qgm_acc) ) THEN
       IF ( SIZE(p18_qgm_acc,1) /= ngmt .OR. SIZE(p18_qgm_acc,2) /= p18_nij ) THEN
          !$acc exit data delete( p18_qgm_acc )
          DEALLOCATE( p18_qgm_acc )
       ENDIF
    ENDIF
    IF ( .NOT. ALLOCATED(p18_qgm_acc) ) THEN
       ALLOCATE( p18_qgm_acc(ngmt,p18_nij) )
       !$acc enter data create( p18_qgm_acc )
    ENDIF
    slot = MERGE( kv, 1, p18_qs_cached )
    p18_qs_slot = slot
    IF ( p18_qs_cached ) THEN
       rebuild = .NOT. p18_qs_done(slot)
    ELSE
       rebuild = ( p18_qs_k /= kv ) .OR. ( .NOT. p18_qs_done(1) )
    ENDIF
    IF ( rebuild ) THEN
       CALL start_clock( 'p18_q' )
       DO iq = 1, nqs
          ikq = index_xkq(current_ik,iq)
          CALL p18_qgm_build( ngmt, xkq_collect(:,ikq), xkp )
          !$acc host_data use_device( p18_qgm_acc )
          CALL p18_qs_fill( ngmt, p18_nij, p18_na, p18_qgm_acc, p18_sf_d, p18_acol_d, p18_aat_d, p18_qs_d(:,:,iq,slot) )
          !$acc end host_data
       ENDDO
       p18_qs_done(slot) = .TRUE.
       p18_qs_k = kv
       CALL stop_clock( 'p18_q' )
    ENDIF
    !
    ! ... block work arrays
    IF ( ALLOCATED(p18_aug_d) ) THEN
       IF ( SIZE(p18_aug_d,1) /= ngmt .OR. SIZE(p18_aug_d,2) /= pblk .OR. SIZE(p18_c_d,1) /= p18_na ) &
          DEALLOCATE( p18_aug_d, p18_vcg_d, p18_c_d, p18_y_d )
    ENDIF
    IF ( .NOT. ALLOCATED(p18_aug_d) ) ALLOCATE( p18_aug_d(ngmt,pblk), p18_vcg_d(ngmt,pblk), p18_c_d(p18_na,pblk), p18_y_d(p18_na,pblk) )
    !
    ! ... projectors of this k (the ordering of igk_exx, as add_nlxx_pot uses), becpsi, deexx
    IF ( ALLOCATED(p18_vkb_h) ) THEN
       IF ( SIZE(p18_vkb_h,1) /= npwx .OR. SIZE(p18_vkb_h,2) /= nkb ) DEALLOCATE( p18_vkb_h, p18_vkb_d )
    ENDIF
    IF ( .NOT. ALLOCATED(p18_vkb_h) ) ALLOCATE( p18_vkb_h(npwx,nkb), p18_vkb_d(npwx,nkb) )
    CALL start_clock( 'p18_vkb' )
    p18_vkb_h = ( 0.0_DP, 0.0_DP )
    CALL init_us_2( n, igk_exx(1:n,current_k), xkp, p18_vkb_h )
    p18_vkb_d = p18_vkb_h
    CALL stop_clock( 'p18_vkb' )
    IF ( ALLOCATED(p18_becpsi_d) ) THEN
       IF ( SIZE(p18_becpsi_d,1) /= nkb .OR. SIZE(p18_becpsi_d,2) /= m ) DEALLOCATE( p18_becpsi_d, p18_deexx_d )
    ENDIF
    IF ( .NOT. ALLOCATED(p18_becpsi_d) ) ALLOCATE( p18_becpsi_d(nkb,m), p18_deexx_d(nkb,m) )
    ! P20: <beta|psi> from the device copies; the host becpsi%k of h_psi is stale on the GPU build when ace = .false.
    CALL ZGEMM( 'C', 'N', nkb, m, n, (1.0_DP,0.0_DP), p18_vkb_d, npwx, p6_psi_d, SIZE(p6_psi_d,1), (0.0_DP,0.0_DP), &
                p18_becpsi_d, nkb )
    p18_deexx_d = ( 0.0_DP, 0.0_DP )
    !
    ! ... PAW: D_k = sum_{q, j} x_occ(j,k')/nqs * 1/2 K(ih,jh,oh,uh) becxx_jh conj(becxx_uh), once per k per EXX cycle
    IF ( okpaw ) THEN
       IF ( ALLOCATED(p18_dpaw_d) ) THEN
          IF ( SIZE(p18_dpaw_d,1) /= nkb .OR. SIZE(p18_dpaw_d,3) /= nks ) DEALLOCATE( p18_dpaw_d, p18_paw_done )
       ENDIF
       IF ( .NOT. ALLOCATED(p18_dpaw_d) ) THEN
          ALLOCATE( p18_dpaw_d(nkb,nkb,nks), p18_paw_done(nks) )
          p18_paw_done = .FALSE.
       ENDIF
       IF ( p18_paw_build /= p6_nbuild ) THEN
          p18_paw_done = .FALSE.
          p18_paw_build = p6_nbuild
       ENDIF
       IF ( .NOT. p18_paw_done(current_k) ) THEN
          CALL start_clock( 'p18_paw' )
          IF ( ALLOCATED(p18_dpaw_h) ) THEN
             IF ( SIZE(p18_dpaw_h,1) /= nkb ) DEALLOCATE( p18_dpaw_h )
          ENDIF
          IF ( .NOT. ALLOCATED(p18_dpaw_h) ) ALLOCATE( p18_dpaw_h(nkb,nkb) )
          p18_dpaw_h = ( 0.0_DP, 0.0_DP )
          DO iq = 1, nqs
             ikq = index_xkq(current_ik,iq)
             ik  = index_xk(ikq)
             DO jb = 1, nbnd
                w = x_occupation(jb,ik) / DBLE(nqs)
                IF ( w == 0.0_DP ) CYCLE
                DO na = 1, nat
                   nt = ityp(na)
                   IF ( .NOT. upf(nt)%tpawp ) CYCLE
                   DO oh = 1, nh(nt)
                      DO ih = 1, nh(nt)
                         z = ( 0.0_DP, 0.0_DP )
                         DO uh = 1, nh(nt)
                            DO jh = 1, nh(nt)
                               z = z + ke(nt)%k(ih,jh,oh,uh) * becxx(ikq)%k(ofsbeta(na)+jh,jb) * CONJG( becxx(ikq)%k(ofsbeta(na)+uh,jb) )
                            ENDDO
                         ENDDO
                         p18_dpaw_h(ofsbeta(na)+ih,ofsbeta(na)+oh) = p18_dpaw_h(ofsbeta(na)+ih,ofsbeta(na)+oh) + w * 0.5_DP * z
                      ENDDO
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
          p18_dpaw_d(:,:,current_k) = p18_dpaw_h
          p18_paw_done(current_k) = .TRUE.
          CALL stop_clock( 'p18_paw' )
       ENDIF
    ENDIF
    !
    IF ( .NOT. p18_banner ) THEN
       p18_banner = .TRUE.
       IF ( p18_qs_cached ) THEN
          WRITE( stdout, '(5X,"EXX-GPU: ultrasoft augmentation on the device (P18); ",I0," (atom, ij) pairs, Q(G,A) ",I0," MB per k, kept for every k")' ) &
               p18_na, 16_8*INT(ngmt,8)*INT(p18_na,8)*INT(nqs,8)/1000000_8
       ELSE
          WRITE( stdout, '(5X,"EXX-GPU: ultrasoft augmentation on the device (P18); ",I0," (atom, ij) pairs, Q(G,A) ",I0," MB per k, rebuilt per call")' ) &
               p18_na, 16_8*INT(ngmt,8)*INT(p18_na,8)*INT(nqs,8)/1000000_8
       ENDIF
    ENDIF
  END SUBROUTINE p18_setup
  !
  !-----------------------------------------------------------------------
  SUBROUTINE vexx_fast_main_us( nrxxs, nbuf, nkqs, ebuf, m, psic, nocc, nqs, ikq, facb, xocc, npairs, pblk, rhoc, res, &
                                plan_full, plan_tail, omega_inv, scale, ngmt, nl, nkb, na_, nat, nbx, becxx, becpsi, qs, &
                                eigq, aat, aih, ajh, aof, ofs, kbat, kbih, nhat, c, aug, vcg, y, deexx, omega, nhm )
  !-----------------------------------------------------------------------
    !! P18: vexx_fast_main_dp with the ultrasoft augmentation: after the batched forward FFT the pair densities of
    !! each q-segment of the block get rho(G) += N Q(G,A) C(A,j) (one ZGEMM); after the Coulomb multiply the block's
    !! vc(G) is gathered on the G sphere and Y = Q^H vc (one ZGEMM) feeds deexx. Double precision only.
    USE cufft
    USE cublas
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nrxxs, nbuf, nkqs, m, nocc, nqs, npairs, pblk, plan_full, plan_tail, ngmt, nkb, na_, nat, nbx, nhm
    COMPLEX(DP), DEVICE, INTENT(IN) :: ebuf(nrxxs,nbuf,nkqs), psic(nrxxs,m)
    INTEGER, DEVICE, INTENT(IN) :: ikq(nqs), nl(*), aat(na_), aih(na_), ajh(na_), aof(nhm,nhm,nat), ofs(nat), kbat(nkb), kbih(nkb), nhat(nat)
    REAL(DP), DEVICE, INTENT(IN) :: facb(nrxxs,nqs), xocc(nocc,nqs)
    COMPLEX(DP), DEVICE, INTENT(IN) :: becxx(nkb,nbx,nkqs), becpsi(nkb,m), qs(ngmt,na_,nqs), eigq(nat,nqs)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: rhoc(nrxxs,pblk), res(nrxxs,m), c(na_,pblk), aug(ngmt,pblk), vcg(ngmt,pblk), y(na_,pblk), deexx(nkb,m)
    REAL(DP), INTENT(IN) :: omega_inv, scale, omega
    INTEGER :: ibnd, p0, np_, ip, ir, ipg, iq, jb, plan, istat, ig, j, nj, jb0, ia, a, ih, jh, o, kq, ikb, nhb
    REAL(DP) :: nfft
    COMPLEX(DP) :: z, zz
    !
    nfft = 1.0_DP / scale
    DO ibnd = 1, m
       DO p0 = 1, npairs, pblk
          np_ = MIN( pblk, npairs - p0 + 1 )
          plan = plan_full
          IF ( np_ /= pblk ) plan = plan_tail
          !$cuf kernel do (2)
          DO ip = 1, np_
             DO ir = 1, nrxxs
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                rhoc(ir,ip) = CONJG( ebuf(ir,jb,ikq(iq)) ) * psic(ir,ibnd) * omega_inv
             ENDDO
          ENDDO
          istat = cufftExecZ2Z( plan, rhoc(1,1), rhoc(1,1), CUFFT_FORWARD )
          IF ( istat /= 0 ) CALL errore( 'vexx_fast_main_us', 'cuFFT exec failed', ABS(istat) )
          ! ... augmentation charges, one q-segment of the block at a time
          ip = 1
          DO WHILE ( ip <= np_ )
             ipg = p0 + ip - 1
             iq  = (ipg-1)/nocc + 1
             jb0 = MOD(ipg-1,nocc) + 1
             nj  = MIN( nocc - jb0 + 1, np_ - ip + 1 )
             !$cuf kernel do (2)
             DO j = 1, nj
                DO ia = 1, na_
                   a = aat(ia); ih = aih(ia); jh = ajh(ia); o = ofs(a); kq = ikq(iq); jb = jb0 + j - 1
                   z = CONJG( becxx(o+ih,jb,kq) ) * becpsi(o+jh,ibnd)
                   IF ( ih /= jh ) z = z + CONJG( becxx(o+jh,jb,kq) ) * becpsi(o+ih,ibnd)
                   c(ia,j) = eigq(a,iq) * z
                ENDDO
             ENDDO
             CALL ZGEMM( 'N', 'N', ngmt, nj, na_, (1.0_DP,0.0_DP), qs(1,1,iq), ngmt, c, na_, (0.0_DP,0.0_DP), aug, ngmt )
             !$cuf kernel do (2)
             DO j = 1, nj
                DO ig = 1, ngmt
                   rhoc( nl(ig), ip+j-1 ) = rhoc( nl(ig), ip+j-1 ) + nfft * aug(ig,j)
                ENDDO
             ENDDO
             ip = ip + nj
          ENDDO
          !$cuf kernel do (2)
          DO ip = 1, np_
             DO ir = 1, nrxxs
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                rhoc(ir,ip) = rhoc(ir,ip) * ( facb(ir,iq) * xocc(jb,iq) * scale )
             ENDDO
          ENDDO
          ! ... deexx from the Fock potential of each pair (newdxx_g): Y = Q^H vc, then the (ih, jh) contraction
          ip = 1
          DO WHILE ( ip <= np_ )
             ipg = p0 + ip - 1
             iq  = (ipg-1)/nocc + 1
             jb0 = MOD(ipg-1,nocc) + 1
             nj  = MIN( nocc - jb0 + 1, np_ - ip + 1 )
             !$cuf kernel do (2)
             DO j = 1, nj
                DO ig = 1, ngmt
                   vcg(ig,j) = rhoc( nl(ig), ip+j-1 )
                ENDDO
             ENDDO
             CALL ZGEMM( 'C', 'N', na_, nj, ngmt, (1.0_DP,0.0_DP), qs(1,1,iq), ngmt, vcg, ngmt, (0.0_DP,0.0_DP), y, na_ )
             !$cuf kernel do (1)
             DO ikb = 1, nkb
                a = kbat(ikb)
                IF ( a > 0 ) THEN
                   ih = kbih(ikb); o = ofs(a); kq = ikq(iq); nhb = nhat(a)
                   zz = ( 0.0_DP, 0.0_DP )
                   DO j = 1, nj
                      jb = jb0 + j - 1
                      DO jh = 1, nhb
                         zz = zz + becxx(o+jh,jb,kq) * y( aof(ih,jh,a), j )
                      ENDDO
                   ENDDO
                   deexx(ikb,ibnd) = deexx(ikb,ibnd) + omega * CONJG( eigq(a,iq) ) * zz
                ENDIF
             ENDDO
             ip = ip + nj
          ENDDO
          istat = cufftExecZ2Z( plan, rhoc(1,1), rhoc(1,1), CUFFT_INVERSE )
          IF ( istat /= 0 ) CALL errore( 'vexx_fast_main_us', 'cuFFT exec failed', ABS(istat) )
          !$cuf kernel do (1)
          DO ir = 1, nrxxs
             DO ip = 1, np_
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                res(ir,ibnd) = res(ir,ibnd) + rhoc(ir,ip) * ebuf(ir,jb,ikq(iq))
             ENDDO
          ENDDO
       ENDDO
    ENDDO
  END SUBROUTINE vexx_fast_main_us
  !
  !-----------------------------------------------------------------------
  SUBROUTINE exx_force_main_us( nrxxs, nbuf, nkqs, ebuf, m, psic, nocc, nqs, ikq, facb, xocc, npairs, pblk, rhoc, &
                                plan_full, plan_tail, omega_inv, scale, ngmt, nl, nkb, na_, nat, nbx, becxx, becpsi, qs, &
                                eigq, aat, aih, ajh, aof, ofs, kbat, kbih, nhat, c, aug, vcg, y, deexx, omega, nhm, &
                                wg_, xkp1, xkp2, xkp3, xkq, gt, tpiba, alpha, vcg3, y3, fa, ex )
  !-----------------------------------------------------------------------
    !! P20: vexx_fast_main_us in force mode. For every band i (weight w_i) and pair block: the augmented pair
    !! densities and their Coulomb potential v as in P18, then Y = Q^H v feeds the exchange D matrix (deexx) and
    !! Y^alpha = Q^H (i tpiba p_alpha v), p = k - k' + G, gives the structure-factor force of every (atom, ij) pair,
    !! fa(alpha, A) += alpha_x Omega w_i Re[conj(Y^alpha(A)) C(A)]. The exchange energy of the same pair densities
    !! accumulates in ex. No potential is applied (nothing goes back to real space).
    USE cufft
    USE cublas
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nrxxs, nbuf, nkqs, m, nocc, nqs, npairs, pblk, plan_full, plan_tail, ngmt, nkb, na_, nat, nbx, nhm
    COMPLEX(DP), DEVICE, INTENT(IN) :: ebuf(nrxxs,nbuf,nkqs), psic(nrxxs,m)
    INTEGER, DEVICE, INTENT(IN) :: ikq(nqs), nl(*), aat(na_), aih(na_), ajh(na_), aof(nhm,nhm,nat), ofs(nat), kbat(nkb), kbih(nkb), nhat(nat)
    REAL(DP), DEVICE, INTENT(IN) :: facb(nrxxs,nqs), xocc(nocc,nqs), xkq(3,nqs), gt(3,ngmt)
    COMPLEX(DP), DEVICE, INTENT(IN) :: becxx(nkb,nbx,nkqs), becpsi(nkb,m), qs(ngmt,na_,nqs), eigq(nat,nqs)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: rhoc(nrxxs,pblk), c(na_,pblk), aug(ngmt,pblk), vcg(ngmt,pblk), y(na_,pblk), deexx(nkb,m)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: vcg3(ngmt,3*pblk), y3(na_,3*pblk)
    REAL(DP), DEVICE, INTENT(INOUT) :: fa(3,na_)
    REAL(DP), INTENT(IN) :: omega_inv, scale, omega, wg_(m), xkp1, xkp2, xkp3, tpiba, alpha
    REAL(DP), INTENT(INOUT) :: ex
    INTEGER :: ibnd, p0, np_, ip, ir, ipg, iq, jb, plan, istat, ig, j, nj, jb0, ia, a, ih, jh, o, kq, ikb, nhb
    REAL(DP) :: nfft, wi, pref, esum, f, s1, s2, s3, p1, p2, p3
    COMPLEX(DP) :: z, zz
    !
    nfft = 1.0_DP / scale
    DO ibnd = 1, m
       wi = wg_(ibnd)
       IF ( wi == 0.0_DP ) CYCLE
       pref = alpha * omega * wi
       DO p0 = 1, npairs, pblk
          np_ = MIN( pblk, npairs - p0 + 1 )
          plan = plan_full
          IF ( np_ /= pblk ) plan = plan_tail
          !$cuf kernel do (2)
          DO ip = 1, np_
             DO ir = 1, nrxxs
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                rhoc(ir,ip) = CONJG( ebuf(ir,jb,ikq(iq)) ) * psic(ir,ibnd) * omega_inv
             ENDDO
          ENDDO
          istat = cufftExecZ2Z( plan, rhoc(1,1), rhoc(1,1), CUFFT_FORWARD )
          IF ( istat /= 0 ) CALL errore( 'exx_force_main_us', 'cuFFT exec failed', ABS(istat) )
          ! ... augmentation charges, one q-segment of the block at a time; C(A, pair) kept for the whole block
          ip = 1
          DO WHILE ( ip <= np_ )
             ipg = p0 + ip - 1
             iq  = (ipg-1)/nocc + 1
             jb0 = MOD(ipg-1,nocc) + 1
             nj  = MIN( nocc - jb0 + 1, np_ - ip + 1 )
             !$cuf kernel do (2)
             DO j = 1, nj
                DO ia = 1, na_
                   a = aat(ia); ih = aih(ia); jh = ajh(ia); o = ofs(a); kq = ikq(iq); jb = jb0 + j - 1
                   z = CONJG( becxx(o+ih,jb,kq) ) * becpsi(o+jh,ibnd)
                   IF ( ih /= jh ) z = z + CONJG( becxx(o+jh,jb,kq) ) * becpsi(o+ih,ibnd)
                   c(ia,ip+j-1) = eigq(a,iq) * z
                ENDDO
             ENDDO
             CALL ZGEMM( 'N', 'N', ngmt, nj, na_, (1.0_DP,0.0_DP), qs(1,1,iq), ngmt, c(1,ip), na_, (0.0_DP,0.0_DP), aug, ngmt )
             !$cuf kernel do (2)
             DO j = 1, nj
                DO ig = 1, ngmt
                   rhoc( nl(ig), ip+j-1 ) = rhoc( nl(ig), ip+j-1 ) + nfft * aug(ig,j)
                ENDDO
             ENDDO
             ip = ip + nj
          ENDDO
          ! ... the Coulomb potential of every pair density, and the exchange energy of the block
          esum = 0.0_DP
          !$cuf kernel do (2)
          DO ip = 1, np_
             DO ir = 1, nrxxs
                ipg = p0 + ip - 1
                iq  = (ipg-1)/nocc + 1
                jb  = MOD(ipg-1,nocc) + 1
                f = facb(ir,iq) * xocc(jb,iq) * scale
                esum = esum + f * scale * ( DBLE(rhoc(ir,ip))**2 + AIMAG(rhoc(ir,ip))**2 )
                rhoc(ir,ip) = rhoc(ir,ip) * f
             ENDDO
          ENDDO
          ex = ex - 0.5_DP * alpha * wi * omega * esum
          ! ... Y = Q^H v (the exchange D matrix, newdxx_g) and Y^alpha = Q^H (i tpiba p_alpha v) (structure-factor force)
          ip = 1
          DO WHILE ( ip <= np_ )
             ipg = p0 + ip - 1
             iq  = (ipg-1)/nocc + 1
             jb0 = MOD(ipg-1,nocc) + 1
             nj  = MIN( nocc - jb0 + 1, np_ - ip + 1 )
             !$cuf kernel do (2)
             DO j = 1, nj
                DO ig = 1, ngmt
                   z = rhoc( nl(ig), ip+j-1 )
                   vcg(ig,j) = z
                   p1 = ( xkp1 - xkq(1,iq) + gt(1,ig) ) * tpiba
                   p2 = ( xkp2 - xkq(2,iq) + gt(2,ig) ) * tpiba
                   p3 = ( xkp3 - xkq(3,iq) + gt(3,ig) ) * tpiba
                   vcg3(ig,j)      = CMPLX( -p1*AIMAG(z), p1*DBLE(z), KIND=DP )
                   vcg3(ig,nj+j)   = CMPLX( -p2*AIMAG(z), p2*DBLE(z), KIND=DP )
                   vcg3(ig,2*nj+j) = CMPLX( -p3*AIMAG(z), p3*DBLE(z), KIND=DP )
                ENDDO
             ENDDO
             CALL ZGEMM( 'C', 'N', na_, nj, ngmt, (1.0_DP,0.0_DP), qs(1,1,iq), ngmt, vcg, ngmt, (0.0_DP,0.0_DP), y, na_ )
             CALL ZGEMM( 'C', 'N', na_, 3*nj, ngmt, (1.0_DP,0.0_DP), qs(1,1,iq), ngmt, vcg3, ngmt, (0.0_DP,0.0_DP), y3, na_ )
             !$cuf kernel do (1)
             DO ia = 1, na_
                s1 = 0.0_DP; s2 = 0.0_DP; s3 = 0.0_DP
                DO j = 1, nj
                   z = c(ia,ip+j-1)
                   s1 = s1 + DBLE( CONJG( y3(ia,j) ) * z )
                   s2 = s2 + DBLE( CONJG( y3(ia,nj+j) ) * z )
                   s3 = s3 + DBLE( CONJG( y3(ia,2*nj+j) ) * z )
                ENDDO
                fa(1,ia) = fa(1,ia) + pref * s1
                fa(2,ia) = fa(2,ia) + pref * s2
                fa(3,ia) = fa(3,ia) + pref * s3
             ENDDO
             !$cuf kernel do (1)
             DO ikb = 1, nkb
                a = kbat(ikb)
                IF ( a > 0 ) THEN
                   ih = kbih(ikb); o = ofs(a); kq = ikq(iq); nhb = nhat(a)
                   zz = ( 0.0_DP, 0.0_DP )
                   DO j = 1, nj
                      jb = jb0 + j - 1
                      DO jh = 1, nhb
                         zz = zz + becxx(o+jh,jb,kq) * y( aof(ih,jh,a), j )
                      ENDDO
                   ENDDO
                   deexx(ikb,ibnd) = deexx(ikb,ibnd) + omega * CONJG( eigq(a,iq) ) * zz
                ENDIF
             ENDDO
             ip = ip + nj
          ENDDO
       ENDDO
    ENDDO
  END SUBROUTINE exx_force_main_us
  !
  !-----------------------------------------------------------------------
  SUBROUTINE p15_release_work_arrays()
  !-----------------------------------------------------------------------
    !! P15: free every device work array of the batched fast path (after an allocation failure)
    IMPLICIT NONE
    IF ( ALLOCATED(p6_rhoc_d) ) DEALLOCATE( p6_rhoc_d )
    IF ( ALLOCATED(p6_rhoc_s) ) DEALLOCATE( p6_rhoc_s )
    IF ( ALLOCATED(p6_psic_d) ) DEALLOCATE( p6_psic_d )
    IF ( ALLOCATED(p6_res_d) ) DEALLOCATE( p6_res_d )
    IF ( ALLOCATED(p6_psi_d) ) DEALLOCATE( p6_psi_d )
    IF ( ALLOCATED(p6_hpsi_d) ) DEALLOCATE( p6_hpsi_d )
    IF ( ALLOCATED(p6_facb_d) ) DEALLOCATE( p6_facb_d )
    IF ( ALLOCATED(p6_ikq_d) ) DEALLOCATE( p6_ikq_d )
    IF ( ALLOCATED(p6_xocc_d) ) DEALLOCATE( p6_xocc_d )
    IF ( ALLOCATED(cfac_all_d) ) DEALLOCATE( cfac_all_d )
    CALL p18_release()   ! P18
  END SUBROUTINE p15_release_work_arrays
#endif
#if defined(__CUDA)
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION exx_stress_fast_path()
  !-----------------------------------------------------------------------
    !! P17: can the exchange stress be computed on the device? The conditions of the vexx fast path
    !! (norm-conserving, collinear, k-points, one band group, serial unpadded EXX grid), one pool, and
    !! the analytic Coulomb kernels (the vcut kernels are host tables).
    USE uspp,           ONLY : okvan
    USE paw_variables,  ONLY : okpaw
    USE mp_exx,         ONLY : negrp
    USE mp_pools,       ONLY : npool
    USE exx_base,       ONLY : use_coulomb_vcut_ws, use_coulomb_vcut_spheric
    IMPLICIT NONE
    LOGICAL, SAVE :: checked = .FALSE., env_off = .FALSE.
    CHARACTER(LEN=16) :: val
    INTEGER :: lenv, stat
    CHARACTER(LEN=160) :: why
    IF ( .NOT. checked ) THEN
       CALL GET_ENVIRONMENT_VARIABLE( 'QE_EXX_FAST', val, lenv, stat )
       env_off = ( stat == 0 .AND. lenv > 0 .AND. ( val(1:1) == '0' .OR. p14_is_off( val ) ) )
       CALL GET_ENVIRONMENT_VARIABLE( 'QE_EXX_STRESS', val, lenv, stat )
       IF ( stat == 0 .AND. lenv > 0 .AND. ( val(1:1) == '0' .OR. p14_is_off( val ) ) ) THEN
          env_off = .TRUE.
          WRITE( stdout, '(5X,"EXX: device exchange stress disabled by QE_EXX_STRESS=",A)' ) TRIM(val)
       ENDIF
       checked = .TRUE.
    ENDIF
    exx_stress_fast_path = use_gpu .AND. (.NOT. env_off) .AND. (.NOT. gamma_only) .AND. (.NOT. noncolin) &
         .AND. (.NOT. okvan) .AND. (.NOT. okpaw) .AND. (negrp == 1) .AND. (npool == 1) &
         .AND. (.NOT. use_coulomb_vcut_ws) .AND. (.NOT. use_coulomb_vcut_spheric) &
         .AND. (dfftt%nproc == 1) .AND. (x_nbnd_occ > 0) &
         .AND. (dfftt%nr1x == dfftt%nr1) .AND. (dfftt%nr2x == dfftt%nr2) .AND. (dfftt%nr3x == dfftt%nr3) &
         .AND. (dfftt%nnr == dfftt%nr1*dfftt%nr2*dfftt%nr3)
    IF ( .NOT. exx_stress_fast_path ) THEN
       why = ''
       IF ( .NOT. use_gpu ) why = TRIM(why) // ' no-gpu'
       IF ( env_off ) why = TRIM(why) // ' QE_EXX_STRESS=0'
       IF ( gamma_only ) why = TRIM(why) // ' gamma_only'
       IF ( noncolin ) why = TRIM(why) // ' noncolin'
       IF ( okvan ) why = TRIM(why) // ' ultrasoft'
       IF ( okpaw ) why = TRIM(why) // ' PAW'
       IF ( negrp /= 1 ) why = TRIM(why) // ' band-groups'
       IF ( npool /= 1 ) why = TRIM(why) // ' pools'
       IF ( use_coulomb_vcut_ws .OR. use_coulomb_vcut_spheric ) why = TRIM(why) // ' vcut-kernel'
       IF ( dfftt%nproc /= 1 ) why = TRIM(why) // ' parallel-FFT'
       IF ( x_nbnd_occ <= 0 ) why = TRIM(why) // ' no-occupied-bands'
       IF ( dfftt%nr1x /= dfftt%nr1 .OR. dfftt%nr2x /= dfftt%nr2 .OR. dfftt%nr3x /= dfftt%nr3 .OR. &
            dfftt%nnr /= dfftt%nr1*dfftt%nr2*dfftt%nr3 ) why = TRIM(why) // ' padded-FFT-grid'
       WRITE( stdout, '(5X,"EXX-GPU: exchange stress on the host routine (P17);",A)' ) TRIM(why)
    ENDIF
  END FUNCTION exx_stress_fast_path
  !
  !-----------------------------------------------------------------------
  SUBROUTINE exx_stress_gpu( sigma, ok )
  !-----------------------------------------------------------------------
    !! P17: exact-exchange stress, Eq. (10) of PRB 73, 125120 (2006), on the device. The pair loop of
    !! exx_stress_host (k-point, norm-conserving, collinear, one band group): for every k, q and occupied
    !! band j of k, the pair densities conj(phi_{i,k+q}) psi_j / omega of a block of occupied k+q
    !! orbitals are formed from the resident device buffer and transformed by one batched forward cuFFT,
    !! |rho(G)|^2 is gathered on the EXX G sphere and contracted, by one DGEMM per block, with the six
    !! independent components of Q(a,b;G) = q_a q_b fac_stress(G)/2 - delta_ab fac(G), the same kernel
    !! branches as the host loop. Sums are double precision. ok = .FALSE. means nothing was done (no
    !! device memory): the caller then runs the host routine.
    USE cufft
    USE cublas
    USE cudafor,        ONLY : cudaMemGetInfo
    USE constants,      ONLY : fpi, e2, pi
    USE cell_base,      ONLY : omega, tpiba, at
    USE wvfct,          ONLY : nbnd, npwx, wg, current_k
    USE klist,          ONLY : xk, ngk, nks, nkstot
    USE lsda_mod,       ONLY : lsda, current_spin, isk
    USE io_files,       ONLY : iunwfc_exx
    USE buffers,        ONLY : get_buffer
    USE mp,             ONLY : mp_sum
    USE mp_pools,       ONLY : inter_pool_comm
    USE mp_exx,         ONLY : inter_egrp_comm, intra_egrp_comm
    USE exx_base,       ONLY : nqs, nq1, nq2, nq3, eps_odg => eps, exxdiv, x_gamma_extrapolation, &
                               grid_factor, yukawa, erfc_scrlen, gau_scrlen, index_xkq, index_xk, xkq_collect
    USE exx_bp_utils,   ONLY : change_data_structure, transform_evc_to_exx, igk_exx_d, evc_exx, nwordwfc_exx
    !
    IMPLICIT NONE
    !
    REAL(DP), INTENT(OUT) :: sigma(3,3)
    LOGICAL, INTENT(OUT) :: ok
    !
    INTEGER, POINTER :: nl_d(:)
    attributes(DEVICE) :: nl_d
    COMPLEX(DP), ALLOCATABLE, DEVICE :: evc_d(:), psic_d(:), rhoc_d(:,:)
    REAL(DP), ALLOCATABLE, DEVICE :: gtl_d(:,:), qmat_d(:,:), a_d(:,:), s_d(:,:), xw_d(:), sig_d(:)
    INTEGER, ALLOCATABLE, DEVICE :: il_d(:)
    INTEGER, ALLOCATABLE :: il_h(:)
    REAL(DP), ALLOCATABLE :: xw_h(:)
    REAL(DP) :: sig_h(6)
    INTEGER :: nrxxs, ngmt, nr1, nr2, nr3, nbuf, nkqs_d, ikk, iq, ikq, ik, ibnd, jbnd, ig, ir, ip, ic, p0, np_
    INTEGER :: ni, npw, current_ik, pblk, plan_full, plan_one, plan, istat, nj
    INTEGER(KIND=8) :: need, free_, total_
    REAL(DP) :: xk1, xk2, xk3, xq1, xq2, xq3, tpiba_, q1, q2, q3, qq, x, gf, f, fs, hf
    REAL(DP) :: e2_, fpi_, pi_, eps_, exxdiv_, gfac_, yuk_, erfc_, gau_, nqh1, nqh2, nqh3
    REAL(DP) :: a11, a21, a31, a12, a22, a32, a13, a23, a33
    REAL(DP) :: omega_inv, scale2, w
    LOGICAL :: xgamma, odg1, odg2, odg3
    LOGICAL, SAVE :: banner = .FALSE.
    INTEGER, EXTERNAL :: global_kpoint_index
    !
    ok = .FALSE.
    sigma = 0.0_DP
    nrxxs = dfftt%nnr
    ngmt  = dfftt%ngm
    nr1 = dfftt%nr1; nr2 = dfftt%nr2; nr3 = dfftt%nr3
    nbuf   = SIZE( exxbuff_d, 2 )
    nkqs_d = SIZE( exxbuff_d, 3 )
    ! ... pair block: the P6e rule, at most the occupied bands
    IF ( p6_pblk > 0 ) THEN
       pblk = MIN( x_nbnd_occ, p6_pblk )
    ELSE
       pblk = NINT( 18.0E6_DP / ( 16.0_DP * DBLE(nrxxs) ) )
       pblk = MIN( x_nbnd_occ, MAX( 8, MIN( 128, pblk ) ) )
    ENDIF
    ! ... P15-style pre-check of the device work arrays
    need = 16_8*INT(nrxxs,8)*INT(pblk,8) + 8_8*INT(ngmt,8)*INT(pblk,8) + 16_8*INT(nrxxs,8) &
         + 72_8*INT(ngmt,8) + 16_8*INT(npwx,8) + 64_8*INT(pblk,8) + 12_8*INT(nbuf,8) + 64_8
    istat = cudaMemGetInfo( free_, total_ )
    IF ( istat == 0 .AND. need + 268435456_8 > free_ ) THEN
       WRITE( stdout, '(5X,"EXX-GPU: exchange stress on the host routine (P17); device-memory (",I0," MB needed, ",I0," MB free)")' ) &
            need/1000000_8, free_/1000000_8
       RETURN
    ENDIF
    istat = 0
    ALLOCATE( rhoc_d(nrxxs,pblk), a_d(ngmt,pblk), psic_d(nrxxs), evc_d(npwx), gtl_d(3,ngmt), qmat_d(6,ngmt), &
              s_d(6,pblk), il_d(nbuf), xw_d(nbuf), sig_d(6), STAT=istat )
    IF ( istat /= 0 ) THEN
       IF ( ALLOCATED(rhoc_d) ) DEALLOCATE( rhoc_d )
       IF ( ALLOCATED(a_d) ) DEALLOCATE( a_d )
       IF ( ALLOCATED(psic_d) ) DEALLOCATE( psic_d )
       IF ( ALLOCATED(evc_d) ) DEALLOCATE( evc_d )
       IF ( ALLOCATED(gtl_d) ) DEALLOCATE( gtl_d )
       IF ( ALLOCATED(qmat_d) ) DEALLOCATE( qmat_d )
       IF ( ALLOCATED(s_d) ) DEALLOCATE( s_d )
       IF ( ALLOCATED(il_d) ) DEALLOCATE( il_d )
       IF ( ALLOCATED(xw_d) ) DEALLOCATE( xw_d )
       IF ( ALLOCATED(sig_d) ) DEALLOCATE( sig_d )
       WRITE( stdout, '(5X,"EXX-GPU: exchange stress on the host routine (P17); device-memory (allocation failed)")' )
       RETURN
    ENDIF
    ALLOCATE( il_h(nbuf), xw_h(nbuf) )
    !
    CALL transform_evc_to_exx( 0 )
    !
    nl_d => dfftt%nl_d
    gtl_d = gt(1:3,1:ngmt)
    ! ... plans (the P6 cache; pinned for this call)
    p6_plan_pinned = 0
    plan_full = vexx_fast_plan( nr1, nr2, nr3, pblk, .FALSE. );  p6_plan_pinned(1) = plan_full
    plan_one  = vexx_fast_plan( nr1, nr2, nr3, 1, .FALSE. );     p6_plan_pinned(2) = plan_one
    !
    omega_inv = 1.0_DP / omega
    scale2 = ( 1.0_DP / DBLE( nr1*nr2*nr3 ) )**2      ! the two fwfft normalisations of the host formula
    tpiba_ = tpiba
    e2_ = e2; fpi_ = fpi; pi_ = pi; eps_ = eps_odg; exxdiv_ = exxdiv; gfac_ = grid_factor
    yuk_ = yukawa; erfc_ = erfc_scrlen; gau_ = gau_scrlen; xgamma = x_gamma_extrapolation
    nqh1 = DBLE(nq1)*0.5_DP; nqh2 = DBLE(nq2)*0.5_DP; nqh3 = DBLE(nq3)*0.5_DP
    a11 = at(1,1); a21 = at(2,1); a31 = at(3,1)
    a12 = at(1,2); a22 = at(2,2); a32 = at(3,2)
    a13 = at(1,3); a23 = at(2,3); a33 = at(3,3)
    !
    IF ( .NOT. banner ) THEN
       banner = .TRUE.
       WRITE( stdout, '(5X,"EXX-GPU: exchange stress on the device (P17); ",I0," k-points x ",I0," q, pair block ",I0," of at most ",I0," occupied orbitals")' ) &
            nks, nqs, pblk, x_nbnd_occ
    ENDIF
    !
    sig_d = 0.0_DP
    !
    DO ikk = 1, nks
       current_k = ikk
       IF ( lsda ) current_spin = isk(ikk)
       npw = ngk(ikk)
       IF ( nks > 1 ) CALL get_buffer( evc_exx, nwordwfc_exx, iunwfc_exx, ikk )
       current_ik = global_kpoint_index( nkstot, ikk )
       xk1 = xk(1,ikk); xk2 = xk(2,ikk); xk3 = xk(3,ikk)
       !
       DO iq = 1, nqs
          ikq = index_xkq(current_ik,iq)
          ik  = index_xk(ikq)
          xq1 = xkq_collect(1,ikq); xq2 = xkq_collect(2,ikq); xq3 = xkq_collect(3,ikq)
          ! ... the occupied k+q orbitals of this q (the host loop's threshold)
          ni = 0
          DO ibnd = 1, MIN( nbuf, nbnd )
             IF ( ABS( x_occupation(ibnd,ik) ) < 1.d-6 ) CYCLE
             ni = ni + 1
             il_h(ni) = ibnd
             xw_h(ni) = x_occupation(ibnd,ik)
          ENDDO
          IF ( ni == 0 ) CYCLE
          il_d(1:ni) = il_h(1:ni)
          xw_d(1:ni) = xw_h(1:ni)
          !
          ! ... Q(1:6,G): xx, yy, zz, xy, xz, yz of q_a q_b fac_stress/2 - delta_ab fac, the host branches
          !$cuf kernel do (1)
          DO ig = 1, ngmt
             q1 = xk1 - xq1 + gtl_d(1,ig)
             q2 = xk2 - xq2 + gtl_d(2,ig)
             q3 = xk3 - xq3 + gtl_d(3,ig)
             IF ( xgamma ) THEN
                x = ( q1*a11 + q2*a21 + q3*a31 ) * nqh1
                odg1 = ABS( x - NINT(x) ) < eps_
                x = ( q1*a12 + q2*a22 + q3*a32 ) * nqh2
                odg2 = ABS( x - NINT(x) ) < eps_
                x = ( q1*a13 + q2*a23 + q3*a33 ) * nqh3
                odg3 = ABS( x - NINT(x) ) < eps_
             ELSE
                odg1 = .FALSE.; odg2 = .FALSE.; odg3 = .FALSE.
             ENDIF
             q1 = q1 * tpiba_; q2 = q2 * tpiba_; q3 = q3 * tpiba_
             qq = q1*q1 + q2*q2 + q3*q3
             gf = gfac_
             IF ( gau_ > 0.0_DP ) THEN
                f  = e2_ * ( (pi_/gau_)**1.5_DP ) * EXP( -qq/4.0_DP/gau_ ) * gf
                fs = e2_ * 2.0_DP/4.0_DP/gau_ * EXP( -qq/4.0_DP/gau_ ) * ( (pi_/gau_)**1.5_DP ) * gf
                IF ( odg1 .AND. odg2 .AND. odg3 ) THEN
                   f = 0.0_DP; fs = 0.0_DP
                ENDIF
             ELSE IF ( qq > 1.d-8 ) THEN
                IF ( erfc_ > 0.0_DP ) THEN
                   f  = e2_*fpi_/qq * ( 1.0_DP - EXP( -qq/4.0_DP/erfc_**2 ) ) * gf
                   fs = -e2_*fpi_ * 2.0_DP/qq**2 * ( ( 1.0_DP + qq/4.0_DP/erfc_**2 ) * EXP( -qq/4.0_DP/erfc_**2 ) - 1.0_DP ) * gf
                ELSE
                   f  = e2_*fpi_/( qq + yuk_ ) * gf
                   fs = 2.0_DP * e2_*fpi_/( qq + yuk_ )**2 * gf
                ENDIF
                IF ( odg1 .AND. odg2 .AND. odg3 ) THEN
                   f = 0.0_DP; fs = 0.0_DP
                ENDIF
             ELSE
                f  = - exxdiv_
                fs = 0.0_DP
                IF ( yuk_ > 0.0_DP .AND. .NOT. xgamma ) THEN
                   f  = f + e2_*fpi_/( qq + yuk_ )
                   fs = 2.0_DP * e2_*fpi_/( qq + yuk_ )**2
                ENDIF
                IF ( erfc_ > 0.0_DP .AND. .NOT. xgamma ) THEN
                   f  = e2_*fpi_ / ( 4.0_DP*erfc_**2 )
                   fs = e2_*fpi_ / ( 8.0_DP*erfc_**4 )
                ENDIF
             ENDIF
             hf = 0.5_DP * fs
             qmat_d(1,ig) = q1*q1*hf - f
             qmat_d(2,ig) = q2*q2*hf - f
             qmat_d(3,ig) = q3*q3*hf - f
             qmat_d(4,ig) = q1*q2*hf
             qmat_d(5,ig) = q1*q3*hf
             qmat_d(6,ig) = q2*q3*hf
          ENDDO
          !
          ! ... occupied bands j of k (a zero weight adds nothing on the host either)
          DO jbnd = 1, nbnd
             IF ( wg(jbnd,ikk) == 0.0_DP ) CYCLE
             w = exxalfa * wg(jbnd,ikk) / DBLE(nqs) / 4.0_DP * scale2
             evc_d(1:npw) = evc_exx(1:npw,jbnd)
             psic_d = ( 0.0_DP, 0.0_DP )
             !$cuf kernel do (1)
             DO ig = 1, npw
                psic_d( nl_d( igk_exx_d(ig,ikk) ) ) = evc_d(ig)
             ENDDO
             istat = cufftExecZ2Z( plan_one, psic_d, psic_d, CUFFT_INVERSE )
             IF ( istat /= 0 ) CALL errore( 'exx_stress_gpu', 'cuFFT exec failed', ABS(istat) )
             !
             DO p0 = 1, ni, pblk
                np_ = MIN( pblk, ni - p0 + 1 )
                plan = plan_full
                IF ( np_ /= pblk ) THEN
                   plan = vexx_fast_plan( nr1, nr2, nr3, np_, .FALSE. );  p6_plan_pinned(3) = plan
                ENDIF
                !$cuf kernel do (2)
                DO ip = 1, np_
                   DO ir = 1, nrxxs
                      rhoc_d(ir,ip) = CONJG( exxbuff_d(ir,il_d(p0+ip-1),ikq) ) * psic_d(ir) * omega_inv
                   ENDDO
                ENDDO
                istat = cufftExecZ2Z( plan, rhoc_d(1,1), rhoc_d(1,1), CUFFT_FORWARD )
                IF ( istat /= 0 ) CALL errore( 'exx_stress_gpu', 'cuFFT exec failed', ABS(istat) )
                !$cuf kernel do (2)
                DO ip = 1, np_
                   DO ig = 1, ngmt
                      a_d(ig,ip) = DBLE( rhoc_d(nl_d(ig),ip) )**2 + AIMAG( rhoc_d(nl_d(ig),ip) )**2
                   ENDDO
                ENDDO
                CALL DGEMM( 'N', 'N', 6, np_, ngmt, 1.0_DP, qmat_d, 6, a_d, ngmt, 0.0_DP, s_d, 6 )
                !$cuf kernel do (1)
                DO ic = 1, 6
                   DO ip = 1, np_
                      sig_d(ic) = sig_d(ic) + w * s_d(ic,ip) * xw_d(p0+ip-1)
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    !
    sig_h = sig_d
    sigma(1,1) = sig_h(1); sigma(2,2) = sig_h(2); sigma(3,3) = sig_h(3)
    sigma(1,2) = sig_h(4); sigma(2,1) = sig_h(4)
    sigma(1,3) = sig_h(5); sigma(3,1) = sig_h(5)
    sigma(2,3) = sig_h(6); sigma(3,2) = sig_h(6)
    !
    DEALLOCATE( rhoc_d, a_d, psic_d, evc_d, gtl_d, qmat_d, s_d, il_d, xw_d, sig_d, il_h, xw_h )
    p6_plan_pinned = 0
    !
    CALL mp_sum( sigma, intra_egrp_comm )
    CALL mp_sum( sigma, inter_egrp_comm )
    CALL mp_sum( sigma, inter_pool_comm )
    !
    CALL change_data_structure( .FALSE. )
    ok = .TRUE.
    !
  END SUBROUTINE exx_stress_gpu
#endif
  !
  LOGICAL FUNCTION exx_us_forces_available()
  !-----------------------------------------------------------------------
    !! P20: the exact-exchange forces of an ultrasoft / PAW hybrid run come from the device force pass, which needs
    !! the batched path (GPU build, k-points, collinear, G-space augmentation); setup lets relax / md through when true
    USE control_flags,    ONLY : gamma_only, tqr
    USE noncollin_module, ONLY : noncolin
    USE mp_pools,         ONLY : nproc_pool
    USE mp_bands,         ONLY : nbgrp
    IMPLICIT NONE
    LOGICAL, EXTERNAL :: check_gpu_support   ! setup.f90; use_gpu itself is set in setup_para, after setup's guard
    exx_us_forces_available = .FALSE.
#if defined(__CUDA)
    exx_us_forces_available = check_gpu_support() .AND. (.NOT. gamma_only) .AND. (.NOT. noncolin) .AND. (.NOT. tqr) &
                              .AND. ( nproc_pool == 1 ) .AND. ( nbgrp == 1 )
#endif
  END FUNCTION exx_us_forces_available
  !
  LOGICAL FUNCTION exx_force_path_ok()
    !! P20: can the force pass run now? The structural conditions of exx_us_forces_available plus the run-time state
    !! of the batched path (device memory, exxinit done). QE_EXX_FAST switches vexx only: the force pass has no other
    !! implementation, so it runs whatever that switch says.
    USE mp_exx, ONLY : negrp
    IMPLICIT NONE
#if defined(__CUDA)
    exx_force_path_ok = exx_us_forces_available() .AND. (.NOT. p15_fast_disabled) .AND. (x_nbnd_occ > 0) .AND. (negrp == 1) &
                        .AND. (dfftt%nproc == 1) .AND. (dfftt%nr1x == dfftt%nr1) .AND. (dfftt%nr2x == dfftt%nr2) &
                        .AND. (dfftt%nr3x == dfftt%nr3) .AND. (dfftt%nnr == dfftt%nr1*dfftt%nr2*dfftt%nr3)
#else
    exx_force_path_ok = .FALSE.
#endif
  END FUNCTION exx_force_path_ok
  !
  !-----------------------------------------------------------------------
  SUBROUTINE exx_force_us( forcex )
  !-----------------------------------------------------------------------
    !! P20: the forces from the explicit position dependence of the exact-exchange energy of an ultrasoft / PAW
    !! run (augmentation charges and projectors in the pair densities; zero for norm-conserving pseudopotentials).
    !! For every k-point of this pool: <beta|psi> and <d beta/d tau_alpha|psi> of the converged occupied bands, then
    !! the batched fast path in force mode (exx_force_main_us) leaves the structure-factor term of every (atom, ij)
    !! pair and the exchange D matrix of every band; the projector term is 2 alpha_x w_i Re[conj(<d beta|psi_i>) D_i]
    !! (the factor 2 stands for the derivative through the other orbital of each pair, see the patch header).
    USE ions_base,        ONLY : nat, ityp
    USE cell_base,        ONLY : tpiba
    USE klist,            ONLY : nks, nkstot, ngk, igk_k, xk
    USE wvfct,            ONLY : npwx, wg, current_k
    USE wavefunctions,    ONLY : evc
    USE buffers,          ONLY : get_buffer
    USE io_files,         ONLY : iunwfc, nwordwfc
    USE uspp,             ONLY : nkb, vkb, okvan, ofsbeta
    USE uspp_param,       ONLY : nh, upf
    USE uspp_init,        ONLY : init_us_2
    USE becmod,           ONLY : bec_type, calbec, allocate_bec_type, deallocate_bec_type
    USE gvect,            ONLY : g
    USE exx_base,         ONLY : nqs, index_xkq, xkq_collect
    USE mp_exx,           ONLY : max_ibands
    USE mp_pools,         ONLY : inter_pool_comm
    USE mp,               ONLY : mp_sum
    USE symme,            ONLY : symvector
    USE io_global,        ONLY : stdout
    USE noncollin_module, ONLY : npol
    USE control_flags,    ONLY : lbfgs, lmd
    USE xc_lib,           ONLY : exx_is_active
    IMPLICIT NONE
    REAL(DP), INTENT(OUT) :: forcex(3,nat)
#if defined(__CUDA)
    INTEGER, EXTERNAL :: global_kpoint_index
    TYPE(bec_type) :: becpsi, dbec(3)
    COMPLEX(DP), ALLOCATABLE :: vkb1(:,:), hdum(:,:), deexx_h(:,:)
    REAL(DP), ALLOCATABLE :: fa_h(:,:), xkq_h(:,:)
    INTEGER :: ik, ipol, ig, ikb, m, npw, current_ik, iq, ia, na, nt, ih, ibnd, ngmt, k_save
    REAL(DP) :: wi
    LOGICAL :: ok, okall
    INTEGER :: nfail
    !
    forcex = 0.0_DP
    exx_us_forces_done = .FALSE.
    IF ( .NOT. okvan ) RETURN
    IF ( .NOT. exx_is_active() ) THEN
       CALL infomsg( 'exx_force_us', 'exact exchange not active: no exchange force term (P20)' )
       RETURN
    ENDIF
    IF ( .NOT. exx_force_path_ok() ) THEN
       IF ( lbfgs .OR. lmd ) THEN
          IF ( p15_fast_disabled ) CALL errore( 'exx_force_us', &
             'the exact-exchange forces of a US/PAW run need the batched device path, which this run left for lack of device memory; the term has no host implementation (P20)', 1 )
          CALL errore( 'exx_force_us', &
             'the exact-exchange forces of a US/PAW run need the batched device path (k-points, collinear, one band group, serial FFT grid, G-space augmentation) (P20)', 1 )
       ENDIF
       CALL infomsg( 'exx_force_us', 'exact-exchange forces of a US/PAW run not available on this path: the printed forces lack that term (P20)' )
       RETURN
    ENDIF
    CALL start_clock( 'exx_force' )
    okall = .TRUE.
    m = x_nbnd_occ
    ngmt = dfftt%ngm
    k_save = current_k
    CALL allocate_bec_type( nkb, m, becpsi )
    DO ipol = 1, 3
       CALL allocate_bec_type( nkb, m, dbec(ipol) )
    ENDDO
    ALLOCATE( vkb1(npwx,nkb), hdum(npwx*npol,max_ibands), deexx_h(nkb,m), xkq_h(3,nqs) )
    IF ( ALLOCATED(p20_wg_h) ) DEALLOCATE( p20_wg_h )
    ALLOCATE( p20_wg_h(m) )
    IF ( ALLOCATED(p20_wg_d) ) DEALLOCATE( p20_wg_d )
    ALLOCATE( p20_wg_d(m) )
    IF ( ALLOCATED(p20_xkq_d) ) DEALLOCATE( p20_xkq_d )
    ALLOCATE( p20_xkq_d(3,nqs) )
    IF ( ALLOCATED(p20_gt_d) ) DEALLOCATE( p20_gt_d )
    ALLOCATE( p20_gt_d(3,ngmt) )
    p20_gt_d = gt(1:3,1:ngmt)
    p20_tpiba = tpiba
    p20_ex = 0.0_DP
    hdum = ( 0.0_DP, 0.0_DP )
    !
    DO ik = 1, nks
       npw = ngk(ik)
       current_k = ik
       current_ik = global_kpoint_index( nkstot, ik )
       IF ( nks > 1 ) CALL get_buffer( evc, nwordwfc, iunwfc, ik )
       ! <beta|psi> and its derivative: d vkb / d tau_alpha = -i tpiba (k+G)_alpha vkb (vkb carries e^{-i(k+G).tau})
       CALL init_us_2( npw, igk_k(1,ik), xk(:,ik), vkb )
       CALL calbec( npw, vkb, evc, becpsi, m )
       DO ipol = 1, 3
          vkb1 = ( 0.0_DP, 0.0_DP )
          DO ikb = 1, nkb
             DO ig = 1, npw
                vkb1(ig,ikb) = vkb(ig,ikb) * CMPLX( 0.0_DP, -tpiba * ( xk(ipol,ik) + g(ipol,igk_k(ig,ik)) ), KIND=DP )
             ENDDO
          ENDDO
          CALL calbec( npw, vkb1, evc, dbec(ipol), m )
       ENDDO
       DO iq = 1, nqs
          xkq_h(:,iq) = xkq_collect(:,index_xkq(current_ik,iq))
       ENDDO
       p20_xkq_d = xkq_h
       p20_wg_h(1:m) = wg(1:m,ik)
       p20_wg_d = p20_wg_h
       ! the fast path in force mode: pair densities, D matrix, structure-factor term
       p20_force_mode = .TRUE.
       CALL vexx_bp_k_gpu_fast( npwx, npw, m, evc, hdum, ok, becpsi )
       p20_force_mode = .FALSE.
       IF ( .NOT. ok ) THEN
          okall = .FALSE.
          EXIT
       ENDIF
       ! ... structure-factor term: (atom, ij) pair columns into the atoms
       ALLOCATE( fa_h(3,p18_na) )
       fa_h = p20_fa_d
       DO ia = 1, p18_na
          forcex(:,p18_aat_h(ia)) = forcex(:,p18_aat_h(ia)) + fa_h(:,ia)
       ENDDO
       DEALLOCATE( fa_h )
       ! ... projector term: 2 alpha_x w_i Re[ conj(<d beta_m/d tau|psi_i>) D_m,i ]
       deexx_h = p18_deexx_d
       DO ibnd = 1, m
          wi = wg(ibnd,ik)
          IF ( wi == 0.0_DP ) CYCLE
          DO na = 1, nat
             nt = ityp(na)
             IF ( .NOT. upf(nt)%tvanp ) CYCLE
             DO ih = 1, nh(nt)
                ikb = ofsbeta(na) + ih
                DO ipol = 1, 3
                   forcex(ipol,na) = forcex(ipol,na) + 2.0_DP * exxalfa * wi * &
                                     DBLE( CONJG( dbec(ipol)%k(ikb,ibnd) ) * deexx_h(ikb,ibnd) )
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    current_k = k_save
    nfail = MERGE( 0, 1, okall )
    CALL mp_sum( nfail, inter_pool_comm )
    IF ( nfail > 0 ) THEN
       ! a device-memory refusal inside the pass: a relaxation cannot go on without the term, an scf prints its forces without it
       forcex = 0.0_DP
       IF ( lbfgs .OR. lmd ) CALL errore( 'exx_force_us', 'no room on the device for the exact-exchange forces (P20)', 1 )
       CALL infomsg( 'exx_force_us', 'no room on the device for the exact-exchange forces: the printed forces lack that term (P20)' )
    ELSE
       CALL mp_sum( forcex, inter_pool_comm )
       CALL mp_sum( p20_ex, inter_pool_comm )
       ! the k sum ran over the irreducible wedge: symmetrise, as force_us and addusforce do for their terms
       CALL symvector( nat, forcex )
       exx_us_forces_done = .TRUE.
       WRITE( stdout, '(5X,"EXX-GPU: exact-exchange forces of the ultrasoft/PAW run on the device (P20); exchange energy of the force pass ",F17.8," Ry")' ) p20_ex
    ENDIF
    CALL deallocate_bec_type( becpsi )
    DO ipol = 1, 3
       CALL deallocate_bec_type( dbec(ipol) )
    ENDDO
    DEALLOCATE( vkb1, hdum, deexx_h, xkq_h )
    IF ( ALLOCATED(p20_vcg3_d) ) DEALLOCATE( p20_vcg3_d, p20_y3_d )
    IF ( ALLOCATED(p20_dpb_d) ) DEALLOCATE( p20_dpb_d )
    IF ( ALLOCATED(p20_fa_d) ) DEALLOCATE( p20_fa_d )
    IF ( ALLOCATED(p20_xkq_d) ) DEALLOCATE( p20_xkq_d )
    IF ( ALLOCATED(p20_gt_d) ) DEALLOCATE( p20_gt_d )
    IF ( ALLOCATED(p20_wg_d) ) DEALLOCATE( p20_wg_d )
    IF ( ALLOCATED(p20_wg_h) ) DEALLOCATE( p20_wg_h )
    CALL stop_clock( 'exx_force' )
#else
    forcex = 0.0_DP
    IF ( okvan ) CALL infomsg( 'exx_force_us', &
       'exact-exchange forces with US/PAW are computed by the GPU build only (P20): the printed forces lack that term' )
#endif
  END SUBROUTINE exx_force_us
  !
  !
  !
END MODULE exx_bp

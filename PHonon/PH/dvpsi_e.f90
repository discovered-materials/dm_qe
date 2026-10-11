!
! Copyright (C) 2001-2016 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
!----------------------------------------------------------------------
subroutine dvpsi_e (ik, ipol)
  !----------------------------------------------------------------------
  !! On output: \(\text{dvpsi}\) contains \(P_c^+ x | \psi_{ik}\rangle\)
  !! in crystal axis (projected on \(\text{at}(*,\text{ipol})\)).
  !
  !! \(\text{dvpsi}\) is read from file if
  !! \(\text{this_pcxpsi_is_on_file}(\text{ik},\text{ipol})=\text{TRUE}\),
  !! otherwise \(\text{dvpsi}\) is COMPUTED and WRITTEN on file 
  !! (\(\text{vkb}\) and \(\text{evc}\) must be set).
  !
  USE kinds,           ONLY : DP
  USE cell_base,       ONLY : tpiba2, at
  USE io_global,       ONLY : stdout
  USE klist,           ONLY : xk, ngk, igk_k
  USE gvect,           ONLY : g
  USE wvfct,           ONLY : npwx, nbnd, et
  USE wavefunctions, ONLY: evc
  USE buffers,         ONLY : save_buffer, get_buffer
  USE noncollin_module,ONLY : noncolin, npol
  USE becmod,          ONLY : bec_type, becp, calbec, &
                              allocate_bec_type, deallocate_bec_type
  USE mp_bands,        ONLY : use_bgrp_in_hpsi, inter_bgrp_comm
  USE xc_lib,          ONLY : exx_is_active
  USE uspp,            ONLY : okvan, nkb, vkb
  USE uspp_param,      ONLY : nh, nhm
  USE ramanm,          ONLY : eth_rps
  USE units_ph,        ONLY : this_pcxpsi_is_on_file, lrcom, iucom, &
                              lrebar, iuebar

  USE lrus,            ONLY : becp1
  USE qpoint,          ONLY : nksq, ikks
  USE eqv,             ONLY : dpsi, dvpsi
  USE control_lr,      ONLY : nbnd_occ
  USE control_flags,   ONLY : offload_type

  implicit none
  !
  integer, intent(IN) :: ipol, ik
  !
  ! ... Local variables
  !
  integer :: npw
  integer :: ig, na, ibnd, jbnd, ikb, jkb, nt, lter, ih, jh, ijkb0,  &
             nrec, is, js, ijs, ikk
  ! counters

  real(DP), allocatable  :: h_diag (:,:)
  ! the diagonal part of h_scf
  type(bec_type) :: becp2 ! the scalar products
  complex(DP), allocatable :: spsi(:,:)
  real(DP) ::   anorm, thresh
  ! preconditioning cut-off
  ! the desired convergence of linter
  logical :: conv_root
  ! true if convergence has been achieved
  INTEGER :: n_start, n_end

  external ch_psi_all, cg_psi
  !
  call start_clock ('dvpsi_e')
  dpsi=(0.d0, 0.d0)
  dvpsi=(0.d0, 0.d0)
  if (this_pcxpsi_is_on_file(ik,ipol)) then
     nrec = (ipol - 1)*nksq + ik
     call get_buffer(dvpsi, lrebar, iuebar, nrec)
     call stop_clock ('dvpsi_e')
     return
  end if
  !
  ikk=ikks(ik)
  call allocate_bec_type ( nkb, nbnd, becp2)

  ! calculate the commutator [H,x_ipol]  psi > and store it in dpsi

  call commutator_Hx_psi (ikk, nbnd_occ(ikk), at(:, ipol), becp1(ik), becp2, dpsi(:, 1:nbnd_occ(ikk)) )
  !
  !    orthogonalize dpsi to the valence subspace: ps = <evc|dpsi>
  !    Apply -P^+_c
  !    NB it uses dvpsi as workspace
  !
  npw = ngk(ikk)
  CALL orthogonalize(dpsi, evc, ikk, ikk, dvpsi, npw, .false.)
  dpsi=-dpsi
  !
  !   dpsi contains P^+_c [H-eS,x] psi_v for the three crystal polarizations
  !   Now solve the linear systems (H-e_vS)*P_c(x*psi_v)=P_c^+ [H-e_vS,x]*psi_v
  !
  CALL g2_kin(ikk)
  !
  ! compute preconditioning matrix h_diag used by cgsolve_all
  !
  allocate (h_diag( npwx*npol, nbnd))
  CALL h_prec (ik, evc, h_diag)
  !
  dvpsi(:,:) = (0.d0, 0.d0)
  !
  thresh = eth_rps
  call cgsolve_all (ch_psi_all, cg_psi, et (1, ikk), dpsi, dvpsi, &
       h_diag, npwx, npw, thresh, ik, lter, conv_root, anorm, &
       nbnd_occ (ikk), npol)

  if (.not.conv_root) WRITE( stdout, '(5x,"ik",i4," ibnd",i4, &
       & " linter: root not converged ",es10.3)') &
       ik, ibnd, anorm
  !
  FLUSH( stdout )
  deallocate (h_diag)
  !
  ! we have now obtained P_c x |psi>.
  ! In the case of USPP this quantity is needed for the Born
  ! effective charges, so we save it to disc
  !
  ! In the US case we obtain P_c x |psi>, but we need P_c^+ x | psi>,
  ! therefore we apply S again, and then subtract the additional term
  ! furthermore we add the term due to dipole of the augmentation charges.
  !
  if (okvan) then
     !
     ! for effective charges
     !
     nrec = (ipol - 1) * nksq + ik
     call save_buffer(dvpsi, lrcom, iucom, nrec)
     !
     !$acc data copyin(dvpsi)
     !
     allocate (spsi ( npwx*npol, nbnd))
     if (use_bgrp_in_hpsi .AND. .NOT. exx_is_active() .and. nbnd>1 ) then
        call divide(inter_bgrp_comm,nbnd,n_start,n_end)
        if (n_end >= n_start) then
                CALL calbec (offload_type, npw, vkb, dvpsi(:,n_start:n_end), becp , n_end-n_start+1 )
        endif
     else
        CALL calbec (offload_type, npw, vkb, dvpsi, becp )
     end if
     CALL s_psi(npwx,npw,nbnd,dvpsi,spsi)
     !
     !$acc end data 
     !
     call dcopy(2*npwx*npol*nbnd,spsi,1,dvpsi,1)
     deallocate (spsi)
     CALL adddvepsi_us(becp1(ik),becp2,ipol,ikk,dvpsi)
  endif

  IF (nkb > 0) call deallocate_bec_type (becp2)

  nrec = (ipol - 1)*nksq + ik
  call save_buffer(dvpsi, lrebar, iuebar, nrec)
  this_pcxpsi_is_on_file(ik,ipol) = .true.
  call stop_clock ('dvpsi_e')
  return
end subroutine dvpsi_e
!
!----------------------------------------------------------------------
subroutine dvpsi_e_dev (done)
  !----------------------------------------------------------------------
  !! As dvpsi_e, for all k-points and the three polarizations at once:
  !! the right-hand sides \(P_c^+ [H-eS,x]\psi\) are computed k-point by
  !! k-point as in dvpsi_e, the linear systems of all of them are solved
  !! together by the device engine (dfpt_dev), and the results are stored
  !! in the buffers iuebar (and iucom with ultrasoft PPs), where
  !! dvpsi_e finds them. done = .false.: not served, call dvpsi_e.
  !
  USE kinds,           ONLY : DP
  USE cell_base,       ONLY : at
  USE io_global,       ONLY : stdout
  USE klist,           ONLY : xk, ngk, igk_k
  USE lsda_mod,        ONLY : lsda, current_spin, isk
  USE wvfct,           ONLY : npwx, nbnd
  USE wavefunctions,   ONLY : evc
  USE buffers,         ONLY : save_buffer, get_buffer
  USE noncollin_module,ONLY : npol
  USE becmod,          ONLY : bec_type, becp, calbec, &
                              allocate_bec_type, deallocate_bec_type
  USE uspp,            ONLY : okvan, nkb, vkb
  USE uspp_init,       ONLY : init_us_2
  USE ramanm,          ONLY : eth_rps
  USE units_ph,        ONLY : this_pcxpsi_is_on_file, lrcom, iucom, &
                              lrebar, iuebar
  USE units_lr,        ONLY : iuwfc, lrwfc
  USE lrus,            ONLY : becp1
  USE qpoint,          ONLY : nksq, ikks
  USE eqv,             ONLY : dpsi, dvpsi
  USE control_lr,      ONLY : nbnd_occ
  USE control_flags,   ONLY : offload_type
  USE dfpt_dev,        ONLY : dfpt_dev_lin_begin, dfpt_dev_lin_set_rhs, &
                              dfpt_dev_lin_solve, dfpt_dev_lin_get, dfpt_dev_lin_end, &
                              dfpt_dev_agree
  !
  implicit none
  !
  logical, intent(out) :: done
  !
  integer :: ik, ikk, ipol, npw, nrec
  logical :: ok, conv
  type(bec_type) :: becp2
  complex(DP), allocatable :: becp2_all(:,:,:,:), spsi(:,:)
  !
  done = .false.
  ok = .false.
  if ( .not. any(this_pcxpsi_is_on_file) ) call dfpt_dev_lin_begin (3, ok)
  ! on all ranks or on none
  call dfpt_dev_agree (ok)
  if (.not. ok) return
  !
  call start_clock ('dvpsi_e')
  call start_clock ('dvpsi_e_rhs')
  if (okvan) allocate (becp2_all(nkb, nbnd, 3, nksq))
  call allocate_bec_type (nkb, nbnd, becp2)
  !
  ! right-hand sides: -P_c^+ [H-eS, x_ipol] psi
  !
  do ik = 1, nksq
     ikk = ikks(ik)
     npw = ngk(ikk)
     if (lsda) current_spin = isk(ikk)
     if (nksq > 1) then
        call get_buffer (evc, lrwfc, iuwfc, ikk)
        !$acc update device(evc)
     endif
     call init_us_2 (npw, igk_k(1,ikk), xk(1,ikk), vkb, .true.)
     !$acc update host(vkb)
     do ipol = 1, 3
        dpsi = (0.d0, 0.d0)
        dvpsi = (0.d0, 0.d0)
        call commutator_Hx_psi (ikk, nbnd_occ(ikk), at(:, ipol), becp1(ik), becp2, &
                                dpsi(:, 1:nbnd_occ(ikk)) )
        call orthogonalize (dpsi, evc, ikk, ikk, dvpsi, npw, .false.)
        dpsi = -dpsi
        call dfpt_dev_lin_set_rhs (ik, ipol, dpsi)
        if (okvan) becp2_all(:,:,ipol,ik) = becp2%k(:,:)
     enddo
  enddo
  call stop_clock ('dvpsi_e_rhs')
  !
  ! (H - e_v S + alpha_pv P_v) P_c x psi_v = rhs, all k-points and polarizations
  !
  call dfpt_dev_lin_solve (eth_rps, conv)
  if (.not. conv) WRITE( stdout, '(5x,"dvpsi_e: linter: root not converged ")')
  !
  ! P_c x |psi>; with ultrasoft PPs P_c^+ x |psi> and the augmentation dipole
  !
  call start_clock ('dvpsi_e_out')
  if (okvan) allocate (spsi(npwx*npol, nbnd))
  do ik = 1, nksq
     ikk = ikks(ik)
     npw = ngk(ikk)
     if (lsda) current_spin = isk(ikk)
     if (okvan) then
        call init_us_2 (npw, igk_k(1,ikk), xk(1,ikk), vkb, .true.)
        !$acc update host(vkb)
     endif
     do ipol = 1, 3
        nrec = (ipol - 1) * nksq + ik
        call dfpt_dev_lin_get (ik, ipol, dvpsi)
        if (okvan) then
           call save_buffer (dvpsi, lrcom, iucom, nrec)
           !$acc data copyin(dvpsi) copyout(spsi)
           call calbec (offload_type, npw, vkb, dvpsi, becp)
           call s_psi (npwx, npw, nbnd, dvpsi, spsi)
           !$acc end data
           call dcopy (2*npwx*npol*nbnd, spsi, 1, dvpsi, 1)
           becp2%k(:,:) = becp2_all(:,:,ipol,ik)
           call adddvepsi_us (becp1(ik), becp2, ipol, ikk, dvpsi)
        endif
        call save_buffer (dvpsi, lrebar, iuebar, nrec)
        this_pcxpsi_is_on_file(ik,ipol) = .true.
     enddo
  enddo
  if (okvan) deallocate (spsi, becp2_all)
  call deallocate_bec_type (becp2)
  call stop_clock ('dvpsi_e_out')
  !
  call dfpt_dev_lin_end ()
  done = .true.
  call stop_clock ('dvpsi_e')
  return
end subroutine dvpsi_e_dev

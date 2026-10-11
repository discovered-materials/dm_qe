!
! Copyright (C) 2001-2018 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!--------------------------------------------------------------
subroutine zstar_eu_us
  !--------------------------------------------------------------
  !! Calculates the additional part of the Born effective charges
  !! in the case of USPP.
  !
  USE kinds,            ONLY : DP
  USE mp,               ONLY : mp_sum
  USE mp_pools,         ONLY : inter_pool_comm
  USE mp_bands,         ONLY : intra_bgrp_comm, nbgrp
  USE cell_base,        ONLY : omega
  USE ions_base,        ONLY : nat, ntyp => nsp, ityp
  USE buffers,          ONLY : get_buffer
  USE klist,            ONLY : xk, wk, ngk, igk_k
  USE gvecs,            ONLY : doublegrid
  USE fft_base,         ONLY : dfftp, dffts
  USE fft_interfaces,   ONLY : fft_interpolate
  USE lsda_mod,         ONLY : nspin, current_spin, isk, lsda
  USE uspp,             ONLY : okvan, nkb, vkb, nlcc_any
  USE wvfct,            ONLY : nbnd, npwx
  USE paw_variables,    ONLY : okpaw
  USE wavefunctions,    ONLY : evc
  USE uspp_param,       ONLY : upf, nhm, nh
  USE noncollin_module, ONLY : noncolin, npol, nspin_mag
  USE efield_mod,       ONLY : zstareu0
  USE phus,             ONLY : becsumort
  USE modes,            ONLY : u, npert, nirr
  USE units_ph,         ONLY : iucom, lrcom, lrebar, iuebar, &
                               lrdrhous, iudrhous
  USE units_lr,         ONLY : iuwfc, lrwfc, iudwf, lrdwf
  USE mp_pools,         ONLY : nproc_pool, npool
  USE control_lr,       ONLY : nbnd_occ
  USE lrus,             ONLY : int3, int3_paw
  USE eqv,              ONLY : dvpsi, dpsi
  USE qpoint,           ONLY : nksq, ikks
  USE ldaU,             ONLY : lda_plus_u
  USE dv_of_drho_lr
  USE uspp_init,        ONLY : init_us_2
  USE dfpt_dev_grid,    ONLY : dfpt_dev_dv_of_drho
  USE incdrhoscf_mod,   ONLY : incdrhoscf, incdrhoscf_nc
  USE usaug_dev,        ONLY : usaug_on, usaug_check
  !
  implicit none
  !
  integer :: npw, ibnd, jbnd, ipol, jpol, imode0, irr, imode, nrec, mode
  integer :: ik, ikk, ig, ir, is, i, j, mu, ipert
  integer :: ih, jh, ijh
  integer :: iuhxc, lrhxc
  !
  real(DP) :: weight, fact
  !
  complex(DP), allocatable :: dbecsum(:,:,:,:), aux1 (:)
  COMPLEX(DP), ALLOCATABLE :: dbecsum_nc(:, :, :, :, :)
  ! the becsum with dpsi
  ! auxillary work space for fft
  complex(DP) , pointer ::      &
      dvscf(:,:,:)
  complex(DP), allocatable :: pdsp(:,:)
  complex(DP), allocatable :: drhoscfh (:,:)
  complex(DP), allocatable :: dvkb (:,:,:)
  integer :: npe, irr1, imode1, na, nt
  complex(DP), allocatable :: zs_new(:,:), zs_ref(:,:)
  ! the term with the position operator computed for all modes together
  logical :: lnew, lcheck
  logical :: dv_done
  complex(DP), allocatable :: dvtmp(:,:,:)

#ifdef TIMINIG_ZSTAR_US
  call start_clock('zstar_eu_us')
  call start_clock('zstar_us_1')
#endif

  !  auxiliary space for <psi|ds/du|psi>
  allocate (dvscf( dfftp%nnr , nspin_mag, 3))
  allocate (dbecsum( nhm*(nhm+1)/2, nat, nspin_mag, 3))
  if (noncolin) allocate (dbecsum_nc( nhm, nhm, nat, nspin, 3))
  allocate (aux1(  dffts%nnr))
  allocate (pdsp(nbnd,nbnd))

  !
  ! Set the initial values to zero
  !
  pdsp    = (0.d0,0.d0)
  dvscf   = (0.d0,0.d0)
  dbecsum = (0.d0,0.d0)
  if (noncolin) dbecsum_nc=(0.d0,0.d0)
  !
  ! first we calculate the perturbed charge density and the perturbed
  ! Hartree and exchange and correlation potential , which we need later
  ! for the calculation of the Hartree and xc part
  !
  do ik = 1, nksq
     ikk = ikks(ik)
     npw = ngk(ikk)
     if (nksq.gt.1) then 
        call get_buffer (evc, lrwfc, iuwfc, ikk)
        !$acc update device(evc)
     endif
     if (lsda) current_spin = isk (ikk)
     call init_us_2 (npw, igk_k(1,ikk), xk(1,ikk), vkb)
     weight = wk (ikk)
     do jpol = 1, 3
        nrec = (jpol - 1) * nksq + ik
        call get_buffer(dpsi, lrdwf, iudwf, nrec)
        if (noncolin) then
           call incdrhoscf_nc (dvscf(:,:,jpol),weight,ik, &
                              dbecsum_nc(1,1,1,1,jpol), dpsi, 1.0d0)
        else
           call incdrhoscf (dvscf(:,current_spin,jpol),weight,ik, &
                            dbecsum(1,1,current_spin,jpol), dpsi)
        endif
     end do
  end do

     IF (noncolin) THEN
        call mp_sum ( dbecsum_nc, intra_bgrp_comm )
     ELSE
        call mp_sum ( dbecsum, intra_bgrp_comm )
     END IF
#ifdef TIMINIG_ZSTAR_US
  call stop_clock('zstar_us_1')
  call start_clock('zstar_us_2')
#endif

  if (doublegrid) then
     do is = 1, nspin_mag
        do ipol = 1, 3
           call fft_interpolate(dffts, dvscf(:,is,ipol), dfftp, dvscf(:,is,ipol))
        end do
     end do
  end if

  IF (noncolin.and.okvan) CALL set_dbecsum_nc(dbecsum_nc, dbecsum, 3)

  call lr_addusddens (3, dbecsum, dvscf)

  call mp_sum ( dvscf, inter_pool_comm )

#ifdef TIMINIG_ZSTAR_US
  call stop_clock('zstar_us_2')
  call start_clock('zstar_us_3')
#endif

  if (nlcc_any) call addnlcc_zstar_eu_us (dvscf)

  ! the three directions at once on the device, when the case is served
  allocate (dvtmp(dfftp%nnr, nspin_mag, 3))
  call dfpt_dev_dv_of_drho (3, dvscf, dvtmp, dv_done)
  if (dv_done) call zcopy (dfftp%nnr*nspin_mag*3, dvtmp, 1, dvscf, 1)
  deallocate (dvtmp)
  if (.not. dv_done) then
  do ipol = 1, 3
     !
     ! Instead of recalculating the perturbed charge density,
     ! it can also be read from file
     ! NB: Then the variable fildrho must be set
     !
     ! call davcio_drho(dvscf(1,1,ipol),lrdrho,iudrho,ipol,-1)
     !
     call dv_of_drho (dvscf (:, :, ipol))
  enddo
  endif
  call psymdvscf(dvscf, dfftp)

#ifdef TIMINIG_ZSTAR_US
  call stop_clock('zstar_us_3')
  call start_clock('zstar_us_4')
#endif
!
! Calculate the parts with the perturbed Hartree and exchange and correlation
! potential
!
  imode0 = 0
  allocate(drhoscfh(dfftp%nnr,nspin_mag))
  do irr = 1, nirr
     npe = npert(irr)
     do imode = 1, npe
        mode = imode0 + imode
        call get_buffer(drhoscfh, lrdrhous, iudrhous, mode)
        do jpol = 1, 3
           do is=1,nspin_mag
              zstareu0(jpol,mode) =  zstareu0(jpol,mode) -                  &
                 dot_product(dvscf(1:dfftp%nnr,is,jpol),drhoscfh(1:dfftp%nnr,is)) &
               * omega / DBLE(dfftp%nr1*dfftp%nr2*dfftp%nr3)
           end do
        end do
     end do
     imode0 = imode0 + npe
  end do
  deallocate (drhoscfh)
#ifdef TIMINIG_ZSTAR_US
  call stop_clock('zstar_us_4')
  call start_clock('zstar_us_5')
#endif
  !
  !  Calculate the part with the position operator
  !
  ! In the collinear case the scalar products with the wavefunctions that
  ! this term needs are written with <beta|psi>, <beta|(k+G) psi> and the
  ! like, which do not depend on the mode: see zstar_eu_us_pos below
  !
  call start_clock ('zeu_us_pos')
  lnew = usaug_on()
  if (lnew) then
     allocate (zs_new(3,3*nat))
     call zstar_eu_us_pos (zs_new)
     if (usaug_check()) then
        allocate (zs_ref, source = zstareu0)
     else
        zstareu0(:,:) = zstareu0(:,:) + zs_new(:,:)
     endif
  endif
  if (.not. lnew .or. usaug_check()) then
  allocate (dvkb(npwx,nkb,3))
  do ik = 1, nksq
     ikk = ikks(ik)
     npw = ngk(ikk)
     weight = wk (ikk)
     if (nksq.gt.1) call get_buffer (evc, lrwfc, iuwfc, ikk)
     call init_us_2 (npw, igk_k(1,ikk), xk (1, ikk), vkb)
     call dvkb3(ik, dvkb)
     imode0 = 0
     do irr = 1, nirr
        do imode = 1, npert (irr)
           mode = imode+imode0
           do jpol = 1, 3
              dvpsi = (0.d0,0.d0)
              !
              ! read the Commutator+add. terms
              !
              nrec = (jpol - 1) * nksq + ik
              call get_buffer(dvpsi, lrebar, iuebar, nrec)
              !
              pdsp = (0.d0,0.d0)
              call psidspsi (ik, u (1, mode), pdsp )
#if defined(__MPI)
              call mp_sum( pdsp, intra_bgrp_comm )
#endif
              !
              ! DFPT+U: add to dvpsi the scf term 
              ! dV_Hub(is)/dE_jpol|psi(ik)> contributing to 
              ! the derivative of the eigenvalue w.r.t. E_jpol
              !
              if (lda_plus_u) call adddvhubscf (jpol, ik)
              !
              ! add the term of the double summation
              !
              do ibnd = 1, nbnd_occ(ikk)
                 do jbnd = 1, nbnd_occ(ikk)
                    zstareu0(jpol,mode)=zstareu0(jpol, mode) +           &
                         weight *                                        &
                         dot_product(evc(1:npwx*npol,ibnd),              &
                                     dvpsi(1:npwx*npol,jbnd))*pdsp(jbnd,ibnd)
                 enddo
              enddo
              dvpsi = (0.d0,0.d0)
              dpsi  = (0.d0,0.d0)
              !
              ! For the last part, we read the commutator from disc,
              ! but this time we calculate
              ! dS/du P_c [H-eS]|psi> + (dK(r)/du - dS/du)r|psi>
              !
              ! first we read  P_c [H-eS]|psi> and store it in dpsi
              !
              nrec = (jpol - 1) * nksq + ik
              call get_buffer(dpsi, lrcom, iucom, nrec)
              !
              ! Apply the matrix dS/du
              !
              call add_for_charges(ik, u(1,mode))
              !
              ! Add  (dK(r)/du - dS/du) r | psi>
              !
              call add_dkmds(ik, u(1,mode), jpol, dvkb)
              !
              ! And calculate finally the scalar product
              !
              do ibnd = 1, nbnd_occ(ikk)
                 zstareu0(jpol,mode)=zstareu0(jpol, mode) - weight *  &
                      dot_product(evc(1:npwx*npol,ibnd),dvpsi(1:npwx*npol,ibnd))
              enddo
           enddo
        enddo
        imode0 = imode0 + npert (irr)
     enddo
  enddo
  deallocate (dvkb)
  endif
  if (allocated(zs_ref)) then
     zs_ref(:,:) = zstareu0(:,:) - zs_ref(:,:)
     lcheck = usaug_check ('zstar_eu_us pos', size(zs_new), zs_new, zs_ref)
     deallocate (zs_ref)
  endif
  if (allocated(zs_new)) deallocate (zs_new)
  call stop_clock ('zeu_us_pos')

  deallocate (pdsp)
  deallocate (dbecsum)
  if (noncolin) deallocate(dbecsum_nc)
  deallocate (dvscf)
  deallocate (aux1)

  fact=1.0_DP
#if defined(__MPI)
  fact=1.0_DP/nproc_pool/npool*nbgrp
#endif
  IF (okpaw) THEN
     imode0 = 0
     do irr1 = 1, nirr
        do ipert = 1, npert (irr1)
           mode = imode0 + ipert
           do nt=1,ntyp
              if (upf(nt)%tpawp) then
                 ijh=0
                 do ih=1,nh(nt)
                    do jh=ih,nh(nt)
                       ijh=ijh+1
                       do na=1,nat
                          if (ityp(na)==nt) then
                             do jpol = 1, 3
                                do is=1,nspin_mag
                                 zstareu0(jpol,mode)=zstareu0(jpol,mode)  &
                                    -fact*int3_paw(ih,jh,na,is,jpol)* &
                                            becsumort(ijh,na,is,mode)
                                enddo
                             enddo
                          endif
                       enddo
                    enddo
                 enddo
              endif
           enddo
        enddo
        imode0 = imode0 + npert (irr1)
     enddo
  endif

#ifdef TIMINIG_ZSTAR_US
  call stop_clock('zstar_us_5')
  call stop_clock('zstar_eu_us')
#endif

  return
end subroutine zstar_eu_us
!
!-----------------------------------------------------------------------
subroutine zstar_eu_us_pos (zs)
  !-----------------------------------------------------------------------
  !! The part of the ultrasoft term of the effective charges that contains
  !! the position operator, for all the modes (collinear case, no DFT+U):
  !!
  !! zs(jpol,mode) =   w sum_ij <psi_i|dvpsi_j> <psi_j|dS/du|psi_i>
  !!                 - w sum_i <psi_i| dS/du |dpsi_i> + (dK/du - dS/du) r |psi_i> >
  !!
  !! with dvpsi the commutator [H-eS, r]|psi> plus additional terms (buffer
  !! iuebar), dpsi = P_c [H-eS, r]|psi> (buffer iucom), as in the loop over
  !! k-points and modes of zstar_eu_us (psidspsi, add_for_charges, add_dkmds).
  !! dS/du and dK/du are sums of projectors on the beta functions and on
  !! (k+G) times the beta functions, so every scalar product with psi_i is
  !! a sum over the beta functions of products of
  !!   becp1 = <beta|psi>,   alphap = <beta|i(k+G)psi>,
  !!   bedp  = <beta|dpsi>,  alphapp = <beta|i(k+G)dpsi>,
  !!   becp2 = <dbeta/dk|psi>, alphadk = <dbeta/dk|i(k+G)psi>
  !! that do not depend on the mode, times the displacement of the atom:
  !! they are computed once per k-point and direction of the field instead
  !! of once per mode, and nothing is applied to the wavefunctions.
  !
  USE kinds,            ONLY : DP
  USE cell_base,        ONLY : tpiba
  USE ions_base,        ONLY : nat, ityp
  USE buffers,          ONLY : get_buffer
  USE klist,            ONLY : xk, wk, ngk, igk_k
  USE gvect,            ONLY : g
  USE uspp,             ONLY : nkb, vkb, qq_nt, ofsbeta
  USE uspp_param,       ONLY : nh
  USE wvfct,            ONLY : nbnd, npwx
  USE wavefunctions,    ONLY : evc
  USE modes,            ONLY : u
  USE units_ph,         ONLY : iucom, lrcom, lrebar, iuebar
  USE units_lr,         ONLY : iuwfc, lrwfc
  USE control_lr,       ONLY : nbnd_occ
  USE lrus,             ONLY : becp1, dpqq
  USE phus,             ONLY : alphap
  USE eqv,              ONLY : dvpsi, dpsi
  USE qpoint,           ONLY : nksq, ikks
  USE uspp_init,        ONLY : init_us_2
  !
  implicit none
  !
  complex(DP), intent(out) :: zs(3,3*nat)
  !! the contribution to zstareu0 (field direction, mode) of the k-points of this pool
  !
  real(DP), parameter :: eps = 1.d-12
  complex(DP), parameter :: one = (1.d0,0.d0), zero = (0.d0,0.d0), ci = (0.d0,1.d0)
  integer :: ik, ikk, npw, nocc, jpol, ipol, nrec, ibnd, ig, na, nt, ih, jh, ikb, jkb, &
             ijkb0, mu, mode
  real(DP) :: weight, q, dq
  complex(DP) :: sa, sb1, sb2, t, fac
  complex(DP), allocatable :: dvkb(:,:,:), aux1(:,:), mm(:,:), bedp(:,:), alphapp(:,:,:), &
                              becp2(:,:), alphadk(:,:,:), bm(:,:), am(:,:,:), cb(:,:), tc(:,:)
  !
  zs(:,:) = zero
  if (nkb <= 0) return
  allocate (dvkb(npwx,nkb,3), aux1(npwx,nbnd), mm(nbnd,nbnd), bedp(nkb,nbnd), &
            alphapp(nkb,nbnd,3), becp2(nkb,nbnd), alphadk(nkb,nbnd,3), bm(nkb,nbnd), &
            am(nkb,nbnd,3), cb(nkb,nbnd), tc(3,3*nat))
  !
  ! tc(jpol,mu): the term for the displacement of one atom in one cartesian
  ! direction, mu = (atom, direction)
  !
  tc(:,:) = zero
  do ik = 1, nksq
     ikk = ikks(ik)
     npw = ngk(ikk)
     nocc = nbnd_occ(ikk)
     weight = wk(ikk)
     if (nksq > 1) call get_buffer (evc, lrwfc, iuwfc, ikk)
     call init_us_2 (npw, igk_k(1,ikk), xk(1,ikk), vkb)
     call dvkb3 (ik, dvkb)
     do jpol = 1, 3
        nrec = (jpol - 1) * nksq + ik
        !
        ! mm(i,j) = <psi_i|dvpsi_j>
        !
        call get_buffer (dvpsi, lrebar, iuebar, nrec)
        call zgemm ('C', 'N', nocc, nocc, npwx, one, evc, npwx, dvpsi, npwx, zero, mm, nbnd)
        !
        ! projections of dpsi = P_c [H-eS, r]|psi> and of psi
        !
        call get_buffer (dpsi, lrcom, iucom, nrec)
        call zgemm ('C', 'N', nkb, nocc, npw, one, vkb, npwx, dpsi, npwx, zero, bedp, nkb)
        call zgemm ('C', 'N', nkb, nocc, npw, one, dvkb(1,1,jpol), npwx, evc, npwx, zero, becp2, nkb)
        do ipol = 1, 3
           do ibnd = 1, nocc
              do ig = 1, npw
                 aux1(ig,ibnd) = dpsi(ig,ibnd) * tpiba * (0.d0,1.d0) * &
                                 ( xk(ipol,ikk) + g(ipol,igk_k(ig,ikk)) )
              enddo
           enddo
           call zgemm ('C', 'N', nkb, nocc, npw, one, vkb, npwx, aux1, npwx, zero, &
                       alphapp(1,1,ipol), nkb)
           do ibnd = 1, nocc
              do ig = 1, npw
                 aux1(ig,ibnd) = evc(ig,ibnd) * tpiba * (0.d0,1.d0) * &
                                 ( xk(ipol,ikk) + g(ipol,igk_k(ig,ikk)) )
              enddo
           enddo
           call zgemm ('C', 'N', nkb, nocc, npw, one, dvkb(1,1,jpol), npwx, aux1, npwx, zero, &
                       alphadk(1,1,ipol), nkb)
        enddo
        !
        ! bm(ikb,i) = sum_j conj(becp1(ikb,j)) mm(i,j), am likewise with alphap
        !
        cb(:,1:nocc) = conjg(becp1(ik)%k(:,1:nocc))
        call zgemm ('N', 'T', nkb, nocc, nocc, one, cb, nkb, mm, nbnd, zero, bm, nkb)
        do ipol = 1, 3
           cb(:,1:nocc) = conjg(alphap(ipol,ik)%k(:,1:nocc))
           call zgemm ('N', 'T', nkb, nocc, nocc, one, cb, nkb, mm, nbnd, zero, am(1,1,ipol), nkb)
        enddo
        !
        do na = 1, nat
           nt = ityp(na)
           ijkb0 = ofsbeta(na)
           mu = 3 * (na - 1)
           do ipol = 1, 3
              t = zero
              do ih = 1, nh(nt)
                 ikb = ijkb0 + ih
                 do jh = 1, nh(nt)
                    jkb = ijkb0 + jh
                    q = qq_nt(ih,jh,nt)
                    dq = dpqq(ih,jh,jpol,nt)
                    sa = zero ; sb1 = zero ; sb2 = zero
                    do ibnd = 1, nocc
                       sa = sa + bm(ikb,ibnd) * alphap(ipol,ik)%k(jkb,ibnd) + &
                                 am(ikb,ibnd,ipol) * becp1(ik)%k(jkb,ibnd)
                       sb1 = sb1 + conjg(becp1(ik)%k(ikb,ibnd)) * alphapp(jkb,ibnd,ipol) + &
                                   conjg(alphap(ipol,ik)%k(ikb,ibnd)) * bedp(jkb,ibnd)
                       sb2 = sb2 + conjg(becp1(ik)%k(ikb,ibnd)) * &
                                   ( ci * q * alphadk(jkb,ibnd,ipol) + dq * alphap(ipol,ik)%k(jkb,ibnd) ) + &
                                   conjg(alphap(ipol,ik)%k(ikb,ibnd)) * &
                                   ( ci * q * becp2(jkb,ibnd) + dq * becp1(ik)%k(jkb,ibnd) )
                    enddo
                    t = t + weight * q * sa - weight * ( q * sb1 + sb2 )
                 enddo
              enddo
              tc(jpol,mu+ipol) = tc(jpol,mu+ipol) + t
           enddo
        enddo
     enddo
  enddo
  !
  ! from the displacements of the atoms to the modes; the atoms that a mode
  ! does not move are skipped as in psidspsi, add_for_charges and add_dkmds
  !
  do mode = 1, 3 * nat
     do na = 1, nat
        mu = 3 * (na - 1)
        if ( abs(u(mu+1,mode)) + abs(u(mu+2,mode)) + abs(u(mu+3,mode)) > eps ) then
           do ipol = 1, 3
              fac = u(mu+ipol,mode)
              do jpol = 1, 3
                 zs(jpol,mode) = zs(jpol,mode) + tc(jpol,mu+ipol) * fac
              enddo
           enddo
        endif
     enddo
  enddo
  !
  deallocate (dvkb, aux1, mm, bedp, alphapp, becp2, alphadk, bm, am, cb, tc)
  !
  return
end subroutine zstar_eu_us_pos

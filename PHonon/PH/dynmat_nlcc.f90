!
! Copyright (C) 2001-2016 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
subroutine dynmat_nlcc (imode0, drhop, npe)
  !
  !! This routine adds a contribution to the dynamical matrix due
  !! to the NLCC.
  !

  USE kinds, only : DP
  USE mp_bands,  ONLY: intra_bgrp_comm
  USE mp,        ONLY: mp_sum
  USE ions_base, ONLY : nat
  USE cell_base, ONLY : omega
  USE fft_base, ONLY : dfftp
  USE noncollin_module, ONLY : nspin_lsda, nspin_mag
  USE dynmat, ONLY : dyn, dyn_rec
  USE modes,  ONLY : nirr, npert, u
  USE uspp,   ONLY : nlcc_any
  USE dv_of_drho_lr,    ONLY : dv_of_drho_xc
  USE control_ph, ONLY: lmultipole
  USE dyn_dev,   ONLY : dyn_check
#if defined(__CUDA)
  USE ions_base, ONLY : ityp
  USE gvect,     ONLY : ngm
  USE uspp_param,ONLY : upf
  USE nlcc_ph,   ONLY : drc
  USE dyn_dev,   ONLY : dyn_dev_ok, dyn_dform
  USE dfpt_dev_grid, ONLY : dfpt_dev_dv_of_drho
#endif

  implicit none

  integer, intent(in) :: imode0
  !! input: the starting mode
  integer, intent(in) :: npe
  !! input: the number of perturbations
  complex(DP), intent(in) :: drhop (dfftp%nnr, nspin_mag, npe)
  !! input: the change of density due to perturbation

  integer :: nrtot, ipert, jpert, is, irr, mode, mode1
  ! the total number of points
  ! counter on perturbations
  ! counter on spin
  ! counter on representations
  ! counter on modes

  complex(DP) :: dyn1 (3 * nat, 3 * nat)
  ! auxiliary dynamical matrix
  complex(DP), allocatable :: drhoc (:), dvaux (:,:)
  ! the change of the core
  ! the change of the potential

  real(DP) :: fac
  ! auxiliary factor
#if defined(__CUDA)
  complex(DP), allocatable :: xs (:,:), w (:,:,:), drhoca (:,:), dvall (:,:,:)
  complex(DP) :: ps
  integer :: na, mu
  logical :: xc_done
#endif

  if (.not.nlcc_any) return

  call start_clock ('dynmat_nlcc')
#if defined(__CUDA)
  !
  ! Device path: the scalar products of the response xc potential with
  ! the derivative of the core charge for all the patterns from one FFT
  ! of the potential (in reciprocal space), see dyn_dform
  !
  IF ( dyn_dev_ok( ) ) THEN
     allocate (drhoc( dfftp%nnr), dvaux( dfftp%nnr, nspin_mag))
     allocate (xs( dfftp%nnr, npe), w( 3, nat, npe))
     dyn1 (:,:) = (0.d0, 0.d0)
     fac = 1.d0 / DBLE (nspin_lsda)
     !
     ! response xc potential of all the perturbations at once on the device when the
     ! case is served (dfpt_dev_dv_of_drho), else one by one
     !
     allocate (drhoca( dfftp%nnr, npe), dvall( dfftp%nnr, nspin_mag, npe))
     do ipert = 1, npe
        call addcore(u(1, imode0 + ipert), drhoca(1, ipert))
     enddo
     CALL dfpt_dev_dv_of_drho(npe, drhop, dvall, xc_done, drhoc = drhoca, xc_only = .true.)
     do ipert = 1, npe
        mode = imode0 + ipert
        IF (xc_done) THEN
           xs (:, ipert) = dvall (:, 1, ipert)
           do is = 2, nspin_lsda
              xs (:, ipert) = xs (:, ipert) + dvall (:, is, ipert)
           enddo
        ELSE
           dvaux(:, :) = (0.d0, 0.d0)
           CALL dv_of_drho_xc(dvaux, drho = drhop(1, 1, ipert), drhoc = drhoca(:, ipert))
           xs (:, ipert) = dvaux (:, 1)
           do is = 2, nspin_lsda
              xs (:, ipert) = xs (:, ipert) + dvaux (:, is)
           enddo
        ENDIF
     enddo
     deallocate (drhoca, dvall)
     CALL dyn_dform (dfftp%nr1, dfftp%nr2, dfftp%nr3, dfftp%nnr, ngm, dfftp%nl, &
                     npe, xs, ngm, drc, w)
     do ipert = 1, npe
        mode = imode0 + ipert
        do mode1 = 1, 3 * nat
           ps = (0.d0, 0.d0)
           do na = 1, nat
              mu = 3 * (na - 1)
              if ( upf(ityp(na))%nlcc .and. &
                   abs(u(mu+1,mode1)) + abs(u(mu+2,mode1)) + abs(u(mu+3,mode1)) > 1.0d-12) &
                 ps = ps + w(1,na,ipert) * u(mu+1,mode1) + w(2,na,ipert) * u(mu+2,mode1) + &
                           w(3,na,ipert) * u(mu+3,mode1)
           enddo
           dyn1 (mode, mode1) = ps * omega * fac
        enddo
     enddo
     call dyn_check ('dynmat_nlcc', 3*nat, dyn1)
     dyn (:,:) = dyn(:,:) + dyn1(:,:)
     dyn_rec(:,:)=dyn_rec(:,:)+dyn1(:,:)
     deallocate (xs, w, dvaux, drhoc)
     call stop_clock ('dynmat_nlcc')
     return
  ENDIF
#endif

  allocate (drhoc(  dfftp%nnr))
  allocate (dvaux(  dfftp%nnr, nspin_mag))

  dyn1 (:,:) = (0.d0, 0.d0)
!
!  compute the exchange and correlation potential for this mode
!
  nrtot = dfftp%nr1 * dfftp%nr2 * dfftp%nr3
  fac = 1.d0 / DBLE (nspin_lsda)
!
!  Compute the change of xc potential due to the perturbation
!
  do ipert = 1, npe
     mode = imode0 + ipert
     call addcore(u(1, mode), drhoc)
     !
     dvaux(:, :) = (0.d0, 0.d0)
     IF (.NOT. lmultipole) THEN
       CALL dv_of_drho_xc(dvaux, drho = drhop(1, 1, ipert), drhoc = drhoc)
     ELSE
       CALL dv_of_drho_xc(dvaux, drho = drhop(1, 1, ipert))
     ENDIF
     !
     mode1 = 0
     do irr = 1, nirr
        do jpert = 1, npert (irr)
           mode1 = mode1 + 1
           call addcore(u(1, mode1), drhoc)
           do is = 1, nspin_lsda
              dyn1 (mode, mode1) = dyn1 (mode, mode1) + &
                   dot_product (dvaux (:,is), drhoc) * &
                   omega * fac / DBLE (nrtot)
           enddo
        enddo
     enddo
  enddo
  !
  ! collect contributions from all r/G points.
  !
  call mp_sum ( dyn1, intra_bgrp_comm )
  call dyn_check ('dynmat_nlcc', 3*nat, dyn1)

  dyn (:,:) = dyn(:,:) + dyn1(:,:)
  dyn_rec(:,:)=dyn_rec(:,:)+dyn1(:,:)
  deallocate (dvaux)
  deallocate (drhoc)
  call stop_clock ('dynmat_nlcc')
  return
end subroutine dynmat_nlcc

!
! Copyright (C) 2001-2008 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
subroutine drhodvloc (nu_i0, npe, drhos, wdyn)
  !-----------------------------------------------------------------------
  !! This subroutine computes the contribution of the local
  !! potential to the electronic term <psi|dv-e ds|dpsi> of the dynamical
  !! matrix. It can be used both for KB and for US pseudopotentials.
  !
  USE kinds,     ONLY : DP
  USE ions_base, ONLY : nat
  USE fft_base,  ONLY : dffts
  USE cell_base, ONLY : omega
  USE lsda_mod,  ONLY : nspin
  USE noncollin_module, ONLY : nspin_lsda, nspin_mag
  USE modes,     ONLY : u

  USE mp_bands,  ONLY: intra_bgrp_comm
  USE mp,        ONLY: mp_sum
#if defined(__CUDA)
  USE gvect,     ONLY : ngm
  USE gvecs,     ONLY : ngms
  USE eqv,       ONLY : vlocq
  USE dyn_dev,   ONLY : dyn_dev_ok, dyn_dform
#endif

  implicit none

  integer :: npe
  !! input: the number of perturbation of this representations
  integer :: nu_i0
  !! input: the initial position of the mode
  complex(DP) :: drhos (dffts%nnr, nspin_mag, npe)
  !! the change of density due to perturbations
  complex(DP) :: wdyn(3*nat, 3*nat)
  !! auxiliary matrix where drhodv is stored
  !
  ! ... local variables
  !
  integer :: ipert, is, nu_i, nu_j
  ! counter on perturbations
  ! counter on spin polarizations
  ! counter on the i modes
  ! counter on the j modes

  complex(DP) :: dynwrk (3 * nat, 3 * nat)
  complex(DP), allocatable :: dvloc (:)
  ! d Vloc / dtau
#if defined(__CUDA)
  complex(DP), allocatable :: xs (:,:), w (:,:,:)
  complex(DP) :: ps
  integer :: na, mu
  !
  ! Device path: the scalar products of the response density with
  ! dV_loc/du of all the patterns from one FFT of the density
  ! (in reciprocal space), see dyn_dform
  !
  IF ( dyn_dev_ok( ) ) THEN
     call start_clock ('drhodvloc')
     allocate (xs( dffts%nnr, npe), w( 3, nat, npe))
     do ipert = 1, npe
        xs (:, ipert) = drhos (:, 1, ipert)
        do is = 2, nspin_lsda
           xs (:, ipert) = xs (:, ipert) + drhos (:, is, ipert)
        enddo
     enddo
     CALL dyn_dform (dffts%nr1, dffts%nr2, dffts%nr3, dffts%nnr, ngms, dffts%nl, &
                     npe, xs, ngm, vlocq, w)
     do nu_j = 1, 3 * nat
        do ipert = 1, npe
           nu_i = nu_i0 + ipert
           ps = (0.d0, 0.d0)
           do na = 1, nat
              mu = 3 * (na - 1)
              if ( abs(u(mu+1,nu_j)) + abs(u(mu+2,nu_j)) + abs(u(mu+3,nu_j)) > 1.0d-12) &
                 ps = ps + w(1,na,ipert) * u(mu+1,nu_j) + w(2,na,ipert) * u(mu+2,nu_j) + &
                           w(3,na,ipert) * u(mu+3,nu_j)
           enddo
           wdyn (nu_i, nu_j) = wdyn (nu_i, nu_j) + ps * omega
        enddo
     enddo
     deallocate (xs, w)
     call stop_clock ('drhodvloc')
     return
  ENDIF
#endif

  call start_clock ('drhodvloc')
  allocate (dvloc( dffts%nnr))
  dynwrk (:,:) = (0.d0, 0.d0)
  !
  ! We need a sum over all perturbations
  !
  do nu_j = 1, 3 * nat
     call compute_dvloc (u(1, nu_j), .FALSE., dvloc)
     do ipert = 1, npe
        nu_i = nu_i0 + ipert
        do is = 1, nspin_lsda
        ! FIXME : use zgemm instead of dot_product
           dynwrk (nu_i, nu_j) = dynwrk (nu_i, nu_j) + &
                dot_product(drhos (1:dffts%nnr, is, ipert), dvloc) * &
                  omega / (dffts%nr1 * dffts%nr2 * dffts%nr3)
        enddo
     enddo

  enddo
  !
  ! collect contributions from nodes of a pool (sum over G & R space)
  !
  call mp_sum ( dynwrk, intra_bgrp_comm )

  wdyn(:,:) = wdyn(:,:) + dynwrk(:,:)
  deallocate(dvloc)
  call stop_clock ('drhodvloc')
  return
end subroutine drhodvloc

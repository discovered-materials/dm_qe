!
! Copyright (C) 2001-2016 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
!------------------------------------------------
SUBROUTINE addnlcc_zstar_eu_us( drhop )
!----------===================-------------------

  USE kinds, ONLY : DP
  USE mp_pools, ONLY : my_pool_id
  USE cell_base, ONLY : omega
  USE fft_base, ONLY : dfftp
  USE noncollin_module, ONLY : nspin_lsda,  nspin_mag
  USE efield_mod, ONLY : zstareu0
  USE uspp,   ONLY : nlcc_any
  USE modes,  ONLY : npert, nirr, u
  USE dv_of_drho_lr,    ONLY : dv_of_drho_xc
  USE dfpt_dev_grid,    ONLY : dfpt_dev_dv_of_drho

  IMPLICIT NONE

  COMPLEX(DP) :: drhop (dfftp%nnr,nspin_mag,3)


  INTEGER :: nrtot, ipert, is, irr, mode
  INTEGER :: imode0, npe, ipol

  REAL(DP) :: fac

  COMPLEX(DP), ALLOCATABLE :: drhoc(:), dvall(:,:,:)
  LOGICAL :: done

  IF (.NOT.nlcc_any) RETURN

  IF ( my_pool_id /= 0 ) RETURN
  !
  !  the exchange and correlation potential of the response to the field in
  !  the three directions: it does not depend on the mode (it used to be
  !  recomputed for every mode); on the device when the case is served
  !
  ALLOCATE( drhoc(dfftp%nnr), dvall(dfftp%nnr,nspin_mag,3) )
  dvall = (0.0_dp,0.0_dp)
  CALL dfpt_dev_dv_of_drho( 3, drhop, dvall, done, xc_only = .TRUE. )
  IF ( .NOT. done ) THEN
     DO ipol = 1, 3
        dvall(:,:,ipol) = (0.0_dp,0.0_dp)
        CALL dv_of_drho_xc(dvall(:,:,ipol), drho = drhop(1, 1, ipol))
     ENDDO
  ENDIF
  !
  nrtot = dfftp%nr1 * dfftp%nr2 * dfftp%nr3
  fac = 1.d0 / DBLE (nspin_lsda)
  imode0 = 0
  DO irr = 1, nirr
     npe = npert(irr)
     DO ipert = 1, npe
        mode = imode0 + ipert
        CALL addcore(u(1, mode), drhoc)
        DO ipol = 1, 3
           DO is = 1, nspin_lsda
              zstareu0(ipol,mode) = zstareu0(ipol,mode) -                  &
                   omega * fac / REAL(nrtot, DP) *         &
                   DOT_PRODUCT(dvall(1:dfftp%nnr,is,ipol),drhoc(1:dfftp%nnr))
           END DO
        END DO
     END DO
     imode0 = imode0 + npe
  END DO
  DEALLOCATE( drhoc, dvall )

  RETURN
END SUBROUTINE addnlcc_zstar_eu_us

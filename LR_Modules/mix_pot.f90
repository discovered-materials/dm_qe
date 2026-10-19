!
! Copyright (C) 2001-2018 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
subroutine mix_potential (ndim, vout, vin, alphamix, dr2, tr2, &
     iter, n_iter, file_extension, conv)
  !-----------------------------------------------------------------------
  !
  ! Modified Broyden's method for potential/charge density mixing
  !             D.D.Johnson, PRB 38, 12807 (1988)
  ! On input :
  !    ndim      dimension of arrays vout, vin
  !    vout      output potential/rho at current iteration
  !    vin       potential/rho at previous iteration
  !    alphamix  mixing factor (0 < alphamix <= 1)
  !    tr2       threshold for selfconsistency
  !    iter      current iteration number
  !    n_iter    number of iterations used in the mixing
  !    file_extension  if present save previous iterations on
  !                    file 'prefix'.'file_extension'
  !                    otherwise keep everything in memory
  !
  ! On output:
  !    dr2       [(vout-vin)/ndim]^2
  !    vin       mixed potential
  !    vout      vout-vin
  !    conv      true if dr2.le.tr2
  !
  USE kinds,           ONLY : DP
  USE mp_bands,        ONLY : intra_bgrp_comm
  USE mp,              ONLY : mp_sum
  USE io_files,        ONLY : diropn
  !
  implicit none
  !
  !   First the dummy variables
  !
  character (len=256) :: file_extension
  integer :: ndim, iter, n_iter
  real(DP) :: vout (ndim), vin (ndim), alphamix, dr2, tr2
  logical :: conv
  !
  !   Here the local variables
  !
  logical :: saveonfile, exst
  integer :: iunmix, n, i, j, info, iter_used, ipos, inext, ndimtot
  real(DP) :: gamma, norm
  real(DP), allocatable, save :: df (:,:), dv (:,:)
  !! work space containing info from previous iterations:
  !! must be kept in memory and saved between calls if file_extension=' '
  !
  integer, allocatable :: iwork(:)
  real(DP), allocatable :: vinsave(:), beta(:, :), work(:), w(:)
  !
  real(DP) :: w0 = 0.01_dp
  real(DP) :: w1 = 1.0_dp
  !! adjustable parameters as suggested in the original paper. w(:) is set to w1.
  !
  INTEGER, EXTERNAL :: find_free_unit
  REAL(DP), EXTERNAL :: ddot, dnrm2
  !
  call start_clock ('mix_pot')
  !
  if (iter < 1) call errore ('mix_potential', 'iter must be positive', 1)
  if (ndim < 1) call errore ('mix_potential', 'ndim must be positive', 3)
  !
  saveonfile = file_extension /= ' '
  !
  do n = 1, ndim
     vout (n) = vout (n) - vin (n)
  enddo
  !
  dr2 = dnrm2 (ndim, vout, 1) **2
  ndimtot = ndim
  !
  call mp_sum (dr2, intra_bgrp_comm)
  call mp_sum (ndimtot, intra_bgrp_comm)
  !
  dr2 = (sqrt (dr2) / ndimtot) **2
  !
  conv = dr2 < tr2
  !
  if (saveonfile) then
     iunmix = find_free_unit()
     if (conv) then
        ! remove temporary file (open and close it)
        call diropn (iunmix, file_extension, ndim, exst)
        close (unit=iunmix, status='delete')
        call stop_clock ('mix_pot')
        return
     endif
     call diropn (iunmix, file_extension, ndim, exst)
     if (iter > 1 .AND. .NOT. exst) then
        call infomsg ('mix_potential', 'file not found, restarting')
        iter = 1
     endif
     allocate (df( ndim , n_iter))
     allocate (dv( ndim , n_iter))
  else
     if (iter == 1) then
        allocate (df( ndim , n_iter))
        allocate (dv( ndim , n_iter))
     endif
     if (conv) then
        deallocate (dv)
        deallocate (df)
        call stop_clock ('mix_pot')
        return
     endif
     allocate (vinsave( ndim))
  endif
  !
  ! iter_used = iter-1  if iter <= n_iter
  ! iter_used = n_iter  if iter >  n_iter
  !
  iter_used = min (iter - 1, n_iter)
  !
  ! ipos is the position in which results from the present iteraction
  ! are stored. ipos=iter-1 until ipos=n_iter, then back to 1,2,...
  !
  ipos = iter - 1 - ( (iter - 2) / n_iter) * n_iter
  !
  if (iter > 1) then
     if (saveonfile) then
        call davcio (df (1, ipos), ndim, iunmix, 1, - 1)
        call davcio (dv (1, ipos), ndim, iunmix, 2, - 1)
     endif
     do n = 1, ndim
        df (n, ipos) = vout (n) - df (n, ipos)
        dv (n, ipos) = vin (n) - dv (n, ipos)
     enddo
     norm = (dnrm2 (ndim, df (1, ipos), 1) ) **2
     call mp_sum (norm, intra_bgrp_comm)
     norm = sqrt (norm)
     call dscal (ndim, 1.d0 / norm, df (1, ipos), 1)
     call dscal (ndim, 1.d0 / norm, dv (1, ipos), 1)
  endif
  !
  if (saveonfile) then
     do i = 1, iter_used
        if (i /= ipos) then
           call davcio (df (1, i), ndim, iunmix, 2 * i + 1, - 1)
           call davcio (dv (1, i), ndim, iunmix, 2 * i + 2, - 1)
        endif
     enddo
     call davcio (vout, ndim, iunmix, 1, 1)
     call davcio (vin, ndim, iunmix, 2, 1)
     if (iter > 1) then
        call davcio (df (1, ipos), ndim, iunmix, 2 * ipos + 1, 1)
        call davcio (dv (1, ipos), ndim, iunmix, 2 * ipos + 2, 1)
     endif
  else
     call DCOPY (ndim, vin, 1, vinsave, 1)
  endif
  !
  if ( iter_used > 0 ) then
     !
     ALLOCATE(beta(iter_used, iter_used))
     ALLOCATE(w(iter_used))
     ALLOCATE(work(iter_used))
     ALLOCATE(iwork(iter_used))
     w(:) = w1
     !
     beta = 0.0_dp
     do i = 1, iter_used
        do j = i + 1, iter_used
           beta (i, j) = w (i) * w (j) * ddot (ndim, df (1, j), 1, df (1, i), 1)
           call mp_sum ( beta (i, j), intra_bgrp_comm )
        enddo
        beta (i, i) = w0**2 + w (i) **2
     enddo
     !
     call DSYTRF ('U', iter_used, beta, iter_used, iwork, work, iter_used, info)
     call errore ('broyden', 'factorization', info)
     call DSYTRI ('U', iter_used, beta, iter_used, iwork, work, info)
     call errore ('broyden', 'DSYTRI', info)
     deallocate ( iwork )
     !
     do i = 1, iter_used
        do j = i + 1, iter_used
           beta (j, i) = beta (i, j)
        enddo
     enddo
     !
     do i = 1, iter_used
        work (i) = ddot (ndim, df (1, i), 1, vout, 1)
     enddo
     call mp_sum ( work, intra_bgrp_comm )
     !
  end if
  !
  do n = 1, ndim
     vin (n) = vin (n) + alphamix * vout (n)
  enddo
  !
  do i = 1, iter_used
     gamma = 0.d0
     do j = 1, iter_used
        gamma = gamma + beta (j, i) * w (j) * work (j)
     enddo
     !
     do n = 1, ndim
        vin (n) = vin (n) - w (i) * gamma * (alphamix * df (n, i) + dv (n, i) )
     enddo
  enddo
  !
  IF ( iter_used > 0 ) THEN
     deallocate(beta)
     deallocate(w)
     deallocate(work)
  ENDIF
  !
  if (saveonfile) then
     close (iunmix, status='keep')
     deallocate(dv)
     deallocate(df)
  else
     inext = iter - ( (iter - 1) / n_iter) * n_iter
     call DCOPY (ndim, vout, 1, df (1, inext), 1)
     call DCOPY (ndim, vinsave, 1, dv (1, inext), 1)
     deallocate(vinsave)
  endif
  !
  call stop_clock ('mix_pot')
  !
  return
  !
end subroutine mix_potential

!-----------------------------------------------------------------------
subroutine mix_potential_mem (ndim, vout, vin, alphamix, dr2, tr2, &
     iter, n_iter, conv)
  !-----------------------------------------------------------------------
  !
  ! As mix_potential (modified Broyden mixing, same arithmetic), with the
  ! history of the previous iterations kept in memory instead of being read
  ! from and written to a file at every call, and threaded vector
  ! operations. A history that is missing or of the wrong size (first
  ! iteration, restart) restarts the mixing, as mix_potential does when its
  ! file is missing; iter is not changed.
  !
  USE kinds,           ONLY : DP
  USE mp_bands,        ONLY : intra_bgrp_comm
  USE mp,              ONLY : mp_sum
  !
  implicit none
  !
  integer :: ndim, iter, n_iter
  real(DP) :: vout (ndim), vin (ndim), alphamix, dr2, tr2
  logical :: conv
  !
  integer :: n, i, j, info, iter_used, ipos, inext, ndimtot
  real(DP) :: gamma, norm, fac, acc
  real(DP), allocatable, save :: df (:,:), dv (:,:)
  integer, save :: mit = 0
  ! number of mixing steps since the history was started (iter of mix_potential,
  ! except after a restart)
  !
  integer, allocatable :: iwork(:)
  real(DP), allocatable :: vinsave(:), beta(:, :), work(:), w(:)
  !
  real(DP) :: w0 = 0.01_dp
  real(DP) :: w1 = 1.0_dp
  !
  call start_clock ('mix_pot')
  !
  if (iter < 1) call errore ('mix_potential', 'iter must be positive', 1)
  if (ndim < 1) call errore ('mix_potential', 'ndim must be positive', 3)
  !
  acc = 0.0_dp
  !$omp parallel do reduction(+:acc)
  do n = 1, ndim
     vout (n) = vout (n) - vin (n)
     acc = acc + vout (n) * vout (n)
  enddo
  !$omp end parallel do
  dr2 = acc
  ndimtot = ndim
  !
  call mp_sum (dr2, intra_bgrp_comm)
  call mp_sum (ndimtot, intra_bgrp_comm)
  !
  dr2 = (sqrt (dr2) / ndimtot) **2
  !
  conv = dr2 < tr2
  !
  if (allocated (df)) then
     if (iter == 1 .or. size (df, 1) /= ndim .or. size (df, 2) /= n_iter) deallocate (df, dv)
  endif
  if (.not. allocated (df)) then
     ! no history (first iteration, restart of an interrupted run, iterations that
     ! continue after convergence): the mixing starts here, whatever iter is
     mit = 0
     allocate (df( ndim , n_iter))
     allocate (dv( ndim , n_iter))
  endif
  mit = mit + 1
  if (conv) then
     deallocate (dv)
     deallocate (df)
     call stop_clock ('mix_pot')
     return
  endif
  allocate (vinsave( ndim))
  !
  iter_used = min (mit - 1, n_iter)
  ipos = mit - 1 - ( (mit - 2) / n_iter) * n_iter
  !
  if (mit > 1) then
     acc = 0.0_dp
     !$omp parallel do reduction(+:acc)
     do n = 1, ndim
        df (n, ipos) = vout (n) - df (n, ipos)
        dv (n, ipos) = vin (n) - dv (n, ipos)
        acc = acc + df (n, ipos) * df (n, ipos)
     enddo
     !$omp end parallel do
     norm = acc
     call mp_sum (norm, intra_bgrp_comm)
     norm = sqrt (norm)
     fac = 1.d0 / norm
     !$omp parallel do
     do n = 1, ndim
        df (n, ipos) = fac * df (n, ipos)
        dv (n, ipos) = fac * dv (n, ipos)
     enddo
     !$omp end parallel do
  endif
  !
  !$omp parallel do
  do n = 1, ndim
     vinsave (n) = vin (n)
  enddo
  !$omp end parallel do
  !
  if ( iter_used > 0 ) then
     !
     ALLOCATE(beta(iter_used, iter_used))
     ALLOCATE(w(iter_used))
     ALLOCATE(work(iter_used))
     ALLOCATE(iwork(iter_used))
     w(:) = w1
     !
     beta = 0.0_dp
     do i = 1, iter_used
        do j = i + 1, iter_used
           acc = 0.0_dp
           !$omp parallel do reduction(+:acc)
           do n = 1, ndim
              acc = acc + df (n, j) * df (n, i)
           enddo
           !$omp end parallel do
           beta (i, j) = w (i) * w (j) * acc
           call mp_sum ( beta (i, j), intra_bgrp_comm )
        enddo
        beta (i, i) = w0**2 + w (i) **2
     enddo
     !
     call DSYTRF ('U', iter_used, beta, iter_used, iwork, work, iter_used, info)
     call errore ('broyden', 'factorization', info)
     call DSYTRI ('U', iter_used, beta, iter_used, iwork, work, info)
     call errore ('broyden', 'DSYTRI', info)
     deallocate ( iwork )
     !
     do i = 1, iter_used
        do j = i + 1, iter_used
           beta (j, i) = beta (i, j)
        enddo
     enddo
     !
     do i = 1, iter_used
        acc = 0.0_dp
        !$omp parallel do reduction(+:acc)
        do n = 1, ndim
           acc = acc + df (n, i) * vout (n)
        enddo
        !$omp end parallel do
        work (i) = acc
     enddo
     call mp_sum ( work, intra_bgrp_comm )
     !
  end if
  !
  !$omp parallel do
  do n = 1, ndim
     vin (n) = vin (n) + alphamix * vout (n)
  enddo
  !$omp end parallel do
  !
  do i = 1, iter_used
     gamma = 0.d0
     do j = 1, iter_used
        gamma = gamma + beta (j, i) * w (j) * work (j)
     enddo
     !
     fac = w (i) * gamma
     !$omp parallel do
     do n = 1, ndim
        vin (n) = vin (n) - fac * (alphamix * df (n, i) + dv (n, i) )
     enddo
     !$omp end parallel do
  enddo
  !
  IF ( iter_used > 0 ) THEN
     deallocate(beta)
     deallocate(w)
     deallocate(work)
  ENDIF
  !
  inext = mit - ( (mit - 1) / n_iter) * n_iter
  !$omp parallel do
  do n = 1, ndim
     df (n, inext) = vout (n)
     dv (n, inext) = vinsave (n)
  enddo
  !$omp end parallel do
  deallocate(vinsave)
  !
  call stop_clock ('mix_pot')
  !
  return
  !
end subroutine mix_potential_mem

SUBROUTINE setmixout(in1, in2, mix, dvscfout, dbecsum, ndim, flag )
 !
 USE kinds,    ONLY : DP
 USE mp_bands, ONLY : intra_bgrp_comm
 USE mp,       ONLY : mp_sum
 !
 IMPLICIT NONE
 INTEGER :: in1, in2, flag, ndim, startb, lastb
 COMPLEX(DP) :: mix(in1+in2), dvscfout(in1), dbecsum(in2)
 !
 CALL divide (intra_bgrp_comm, in2, startb, lastb)
 ndim=lastb-startb+1
 !
 IF (flag==-1) THEN
    mix(1:in1)=dvscfout(1:in1)
    mix(in1+1:in1+ndim)=dbecsum(startb:lastb)
 ELSE
    dvscfout(1:in1)=mix(1:in1)
    dbecsum=(0.0_DP,0.0_DP)
    dbecsum(startb:lastb)=mix(in1+1:in1+ndim)
    CALL mp_sum(dbecsum, intra_bgrp_comm)
 ENDIF
 !
 RETURN
 !
END SUBROUTINE setmixout

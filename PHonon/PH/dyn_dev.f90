!
! Copyright (C) 2026 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!-----------------------------------------------------------------------
MODULE dyn_dev
  !-----------------------------------------------------------------------
  !! Device (GPU) kernels for the terms of the dynamical matrix of ph.x
  !! that are computed outside the DFPT self-consistency: the contribution
  !! of each irreducible representation (drhodv, dynmat_nlcc) and the
  !! terms that do not depend on the response (dynmat0: d2ionq, dynmat_us,
  !! dynmatcc).
  !!
  !! * dyn_dform: integrals of a field with the derivative of an
  !!   atom-centred function with respect to all atomic displacements,
  !!   done in reciprocal space with one FFT of the field (the stock
  !!   drhodvloc and dynmat_nlcc build the derivative in real space for
  !!   each of the 3*nat patterns: 3*nat FFTs and 3*nat scalar products).
  !! * dyn_d2form: the same for second derivatives (local part of
  !!   dynmat_us, dynmatcc).
  !! * drhodv_nl_dev, dynmat_us_nl_dev: projections on the beta functions
  !!   with one ZGEMM per block of bands, contractions on the host with
  !!   OpenMP.
  !! * d2ionq_g_dev: reciprocal-space part of the Ewald sum.
  !!
  !! Collinear case only; the callers keep the stock code as the fallback
  !! path (QE_PH_DEV=0, noncollinear, DFT+U, parallel FFT grids).
  !
  USE kinds, ONLY : DP
#if defined(__CUDA)
  USE cudafor
  USE cublas_v2
#endif
  !
  IMPLICIT NONE
  PRIVATE
  SAVE
  !
  PUBLIC :: dyn_dev_ok, dyn_check, dyn_dev_reset
  !
  LOGICAL :: check_read = .FALSE., check_on = .FALSE.
  !
#if defined(__CUDA)
  PUBLIC :: dyn_dform, &
            dyn_fwfft
  !
  COMPLEX(DP), PARAMETER :: c_one = (1.0_DP, 0.0_DP), c_zero = (0.0_DP, 0.0_DP)
  INTEGER, PARAMETER :: max_tmp_elems = 8388608
  !! complex numbers in the work array of the projections (128 MB)
  REAL(DP), PARAMETER :: max_drus_bytes = 8589934592.0_DP, max_drus_frac = 0.15_DP
  !! device memory allowed for the charge response at fixed wavefunctions of
  !! all the modes (drhodvus_dev): absolute, and as a fraction of the free memory
  !
  COMPLEX(DP), ALLOCATABLE, DEVICE :: drus_d(:,:)
  !! (nnr*nspin_mag, 3*nat): the records of the buffer iudrhous of the current q
  COMPLEX(DP), ALLOCATABLE :: drus_h(:,:)
  !! the same on the host, when the device memory allowed does not hold them
  REAL(DP), PARAMETER :: max_drus_host = 17179869184.0_DP
  !! host memory allowed for drus_h (16 GB)
  LOGICAL :: drus_ready = .FALSE.
  REAL(DP) :: drus_q(3) = 0.0_DP
  INTEGER :: drus_n = 0, drus_nm = 0
#endif
  !
CONTAINS
  !
  !-----------------------------------------------------------------------
  LOGICAL FUNCTION dyn_dev_ok( )
    !-----------------------------------------------------------------------
    !! True when the device routines of this module serve the present run.
    !
    USE dvbare_dev, ONLY : dvbare_dev_ok
    USE fft_base,   ONLY : dfftp
    !
    dyn_dev_ok = dvbare_dev_ok( )
    IF ( .NOT. dyn_dev_ok ) RETURN
    dyn_dev_ok = dfftp%nnr == dfftp%nr1*dfftp%nr2*dfftp%nr3 .AND. dfftp%nr1 == dfftp%nr1x .AND. &
                 dfftp%nr2 == dfftp%nr2x .AND. dfftp%nr3 == dfftp%nr3x
    !
  END FUNCTION dyn_dev_ok
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dyn_dev_reset( )
    !-----------------------------------------------------------------------
    !! Forget (and release) the device data that belong to a q-point.
    !
#if defined(__CUDA)
    IF ( ALLOCATED(drus_d) ) DEALLOCATE( drus_d )
    IF ( ALLOCATED(drus_h) ) DEALLOCATE( drus_h )
    drus_ready = .FALSE.
#endif
    !
  END SUBROUTINE dyn_dev_reset
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dyn_check( label, n, a )
    !-----------------------------------------------------------------------
    !! Debugging aid: with QE_PH_DYNCHECK set in the environment, print the
    !! norm and a weighted sum of a contribution to the dynamical matrix
    !! with all digits, to compare the device path with the stock path
    !! (QE_PH_DYN=0) term by term.
    !
    USE io_global, ONLY : stdout
    !
    CHARACTER(LEN=*), INTENT(IN) :: label
    INTEGER, INTENT(IN) :: n
    COMPLEX(DP), INTENT(IN) :: a(n,n)
    CHARACTER(LEN=16) :: val
    INTEGER :: ln, ierr, i, j
    REAL(DP) :: s
    COMPLEX(DP) :: c
    !
    IF ( .NOT. check_read ) THEN
       CALL get_environment_variable( 'QE_PH_DYNCHECK', val, ln, ierr )
       check_on = ( ierr == 0 .AND. ln > 0 )
       check_read = .TRUE.
    ENDIF
    IF ( .NOT. check_on ) RETURN
    s = 0.0_DP ; c = (0.0_DP, 0.0_DP)
    DO j = 1, n
       DO i = 1, n
          s = s + DBLE( a(i,j) )**2 + AIMAG( a(i,j) )**2
          c = c + a(i,j) * COS( 0.37_DP * i + 1.13_DP * j )
       ENDDO
    ENDDO
    WRITE( stdout, '(5x,"DYNCHECK ",a12,3es24.15)' ) label, SQRT(s), c
    !
  END SUBROUTINE dyn_check
  !
#if defined(__CUDA)
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dyn_fwfft( n1, n2, n3, nnr, x )
    !-----------------------------------------------------------------------
    !! x(r) -> x(G) in place for a host array on the grid n1 x n2 x n3 (as
    !! fwfft 'Rho'), computed on the device.
    !
    USE dvbare_dev, ONLY : dev_fft3
    !
    INTEGER, INTENT(IN) :: n1, n2, n3, nnr
    COMPLEX(DP), INTENT(INOUT) :: x(nnr)
    COMPLEX(DP), ALLOCATABLE, DEVICE :: x_d(:)
    INTEGER :: ir, istat
    REAL(DP) :: scale
    !
    ALLOCATE( x_d(nnr), STAT=istat )
    IF ( istat /= 0 ) CALL errore( 'dyn_fwfft', 'device allocation failed', 1 )
    x_d = x
    CALL dev_fft3( n1, n2, n3, 1, x_d, -1 )
    scale = 1.0_DP / ( DBLE(n1) * DBLE(n2) * DBLE(n3) )
    !$cuf kernel do(1)
    DO ir = 1, nnr
       x_d(ir) = x_d(ir) * scale
    ENDDO
    x = x_d
    DEALLOCATE( x_d )
    !
  END SUBROUTINE dyn_fwfft
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dyn_dform( n1, n2, n3, nnr, ngl, nl, npe, x, ldf, f, w )
    !-----------------------------------------------------------------------
    !! For npe fields x(r) on the FFT grid n1 x n2 x n3 and the functions
    !! \(F_s(\tau) = \sum_G f(G,\text{type}(s)) e^{i(q+G)(r-\tau_s)}\)
    !! (f = vlocq: local potential, f = drc: core charge), the integrals
    !! \(\omega^{-1}\int x^*(r) \partial F_s/\partial u_{s\alpha}\,dr\):
    !!
    !! w(alpha,s,ipe) = -i tpiba e^{-i q tau_s} sum_G conj(x(G,ipe))
    !!                  f(G,type(s)) (q+G)_alpha e^{-i G tau_s}
    !!
    !! with x(G) the Fourier transform of x(r) as given by fwfft. The
    !! scalar product of x with the derivative for a displacement pattern
    !! u, as computed by compute_dvloc or addcore in real space, is
    !! nr1*nr2*nr3 * sum_{s,alpha} w(alpha,s) u(alpha,s).
    !
    USE ions_base,  ONLY : nat, ityp, ntyp => nsp
    USE cell_base,  ONLY : tpiba
    USE gvect,      ONLY : eigts1, eigts2, eigts3, mill, g
    USE qpoint,     ONLY : xq, eigqts
    USE dvbare_dev, ONLY : dev_fft3
    !
    INTEGER, INTENT(IN) :: n1, n2, n3, nnr, ngl, npe, ldf
    INTEGER, INTENT(IN) :: nl(ngl)
    !! FFT index of the G vectors
    COMPLEX(DP), INTENT(IN) :: x(nnr,npe)
    REAL(DP), INTENT(IN) :: f(ldf,ntyp)
    COMPLEX(DP), INTENT(OUT) :: w(3,nat,npe)
    !
    COMPLEX(DP), ALLOCATABLE, DEVICE :: x_d(:,:)
    REAL(DP), ALLOCATABLE, DEVICE :: f_d(:,:), w_d(:,:,:)
    INTEGER, ALLOCATABLE, DEVICE :: nl_d(:), ityp_d(:)
    REAL(DP), ALLOCATABLE :: w_h(:,:,:)
    INTEGER :: ipe, na, nt, ig, natl, istat
    REAL(DP) :: s1r, s1i, s2r, s2i, s3r, s3i, q1, q2, q3, tr, ti, scale
    COMPLEX(DP) :: t, fact
    !
    CALL start_clock( 'dyn_dform' )
    ALLOCATE( x_d(nnr,npe), f_d(ldf,ntyp), w_d(6,nat,npe), nl_d(ngl), ityp_d(nat), STAT=istat )
    IF ( istat /= 0 ) CALL errore( 'dyn_dform', 'device allocation failed', 1 )
    ALLOCATE( w_h(6,nat,npe) )
    x_d = x
    f_d = f
    nl_d = nl
    ityp_d = ityp
    DO ipe = 1, npe
       CALL dev_fft3( n1, n2, n3, 1, x_d(1,ipe), -1 )
    ENDDO
    q1 = xq(1) ; q2 = xq(2) ; q3 = xq(3)
    natl = nat
    !
    !$acc parallel loop gang collapse(2) present(eigts1, eigts2, eigts3, mill, g) &
    !$acc          private(nt, s1r, s1i, s2r, s2i, s3r, s3i)
    DO ipe = 1, npe
       DO na = 1, natl
          nt = ityp_d(na)
          s1r = 0.0_DP ; s1i = 0.0_DP ; s2r = 0.0_DP ; s2i = 0.0_DP ; s3r = 0.0_DP ; s3i = 0.0_DP
          !$acc loop vector reduction(+:s1r,s1i,s2r,s2i,s3r,s3i) private(t, tr, ti)
          DO ig = 1, ngl
             t = CONJG( x_d(nl_d(ig),ipe) ) * f_d(ig,nt) * ( eigts1(mill(1,ig),na) * &
                 eigts2(mill(2,ig),na) * eigts3(mill(3,ig),na) )
             tr = DBLE(t) ; ti = AIMAG(t)
             s1r = s1r + tr * (q1 + g(1,ig)) ; s1i = s1i + ti * (q1 + g(1,ig))
             s2r = s2r + tr * (q2 + g(2,ig)) ; s2i = s2i + ti * (q2 + g(2,ig))
             s3r = s3r + tr * (q3 + g(3,ig)) ; s3i = s3i + ti * (q3 + g(3,ig))
          ENDDO
          w_d(1,na,ipe) = s1r ; w_d(2,na,ipe) = s1i
          w_d(3,na,ipe) = s2r ; w_d(4,na,ipe) = s2i
          w_d(5,na,ipe) = s3r ; w_d(6,na,ipe) = s3i
       ENDDO
    ENDDO
    w_h = w_d
    !
    scale = 1.0_DP / ( DBLE(n1) * DBLE(n2) * DBLE(n3) )
    DO ipe = 1, npe
       DO na = 1, nat
          fact = tpiba * (0.d0, -1.d0) * eigqts(na) * scale
          w(1,na,ipe) = CMPLX( w_h(1,na,ipe), w_h(2,na,ipe), KIND=DP ) * fact
          w(2,na,ipe) = CMPLX( w_h(3,na,ipe), w_h(4,na,ipe), KIND=DP ) * fact
          w(3,na,ipe) = CMPLX( w_h(5,na,ipe), w_h(6,na,ipe), KIND=DP ) * fact
       ENDDO
    ENDDO
    DEALLOCATE( x_d, f_d, w_d, nl_d, ityp_d, w_h )
    CALL stop_clock( 'dyn_dform' )
    !
  END SUBROUTINE dyn_dform
  !
#endif
  !
END MODULE dyn_dev

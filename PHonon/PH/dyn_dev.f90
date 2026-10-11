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
  PUBLIC :: dyn_dform, dyn_d2form, drhodv_nl_dev, drhodv_nl_k, drhodv_nl_int2, &
            dynmat_us_nl_dev, d2ionq_g_dev, dyn_fwfft, drhodvus_dev
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
  !-----------------------------------------------------------------------
  SUBROUTINE dyn_d2form( ngl, ldc, c, z, s )
    !-----------------------------------------------------------------------
    !! s(p,na) = sum_G g_i g_j c(G,type(na)) z(G) e^{-i G tau_na} for the
    !! pairs p = (i,j) = 11, 21, 22, 31, 32, 33 (g in units of 2pi/a):
    !! second derivatives with respect to the position of atom na of the
    !! integral of an atom-centred function with a field (local part of
    !! dynmat_us, dynmatcc).
    !
    USE constants, ONLY : tpi
    USE ions_base, ONLY : nat, ityp, ntyp => nsp, tau
    USE gvect,     ONLY : g
    !
    INTEGER, INTENT(IN) :: ngl, ldc
    REAL(DP), INTENT(IN) :: c(ldc,ntyp)
    COMPLEX(DP), INTENT(IN) :: z(ngl)
    COMPLEX(DP), INTENT(OUT) :: s(6,nat)
    !
    REAL(DP), ALLOCATABLE, DEVICE :: c_d(:,:), s_d(:,:), tau_d(:,:)
    COMPLEX(DP), ALLOCATABLE, DEVICE :: z_d(:)
    INTEGER, ALLOCATABLE, DEVICE :: ityp_d(:)
    REAL(DP), ALLOCATABLE :: s_h(:,:)
    INTEGER :: na, nt, ig, natl, p, istat
    REAL(DP) :: arg, tr, ti, g1, g2, g3, t1, t2, t3, &
                a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i, a5r, a5i, a6r, a6i
    COMPLEX(DP) :: t
    !
    ALLOCATE( c_d(ldc,ntyp), z_d(ngl), s_d(12,nat), tau_d(3,nat), ityp_d(nat), STAT=istat )
    IF ( istat /= 0 ) CALL errore( 'dyn_d2form', 'device allocation failed', 1 )
    ALLOCATE( s_h(12,nat) )
    c_d = c
    z_d = z
    tau_d = tau
    ityp_d = ityp
    natl = nat
    !
    !$acc parallel loop gang present(g) private(nt, t1, t2, t3, a1r, a1i, a2r, a2i, a3r, a3i, &
    !$acc                                       a4r, a4i, a5r, a5i, a6r, a6i)
    DO na = 1, natl
       nt = ityp_d(na)
       t1 = tau_d(1,na) ; t2 = tau_d(2,na) ; t3 = tau_d(3,na)
       a1r = 0.0_DP ; a1i = 0.0_DP ; a2r = 0.0_DP ; a2i = 0.0_DP ; a3r = 0.0_DP ; a3i = 0.0_DP
       a4r = 0.0_DP ; a4i = 0.0_DP ; a5r = 0.0_DP ; a5i = 0.0_DP ; a6r = 0.0_DP ; a6i = 0.0_DP
       !$acc loop vector private(arg, t, tr, ti, g1, g2, g3) &
       !$acc      reduction(+:a1r,a1i,a2r,a2i,a3r,a3i,a4r,a4i,a5r,a5i,a6r,a6i)
       DO ig = 1, ngl
          g1 = g(1,ig) ; g2 = g(2,ig) ; g3 = g(3,ig)
          arg = tpi * ( g1 * t1 + g2 * t2 + g3 * t3 )
          t = c_d(ig,nt) * z_d(ig) * CMPLX( COS(arg), -SIN(arg), KIND=DP )
          tr = DBLE(t) ; ti = AIMAG(t)
          a1r = a1r + tr * g1 * g1 ; a1i = a1i + ti * g1 * g1
          a2r = a2r + tr * g2 * g1 ; a2i = a2i + ti * g2 * g1
          a3r = a3r + tr * g2 * g2 ; a3i = a3i + ti * g2 * g2
          a4r = a4r + tr * g3 * g1 ; a4i = a4i + ti * g3 * g1
          a5r = a5r + tr * g3 * g2 ; a5i = a5i + ti * g3 * g2
          a6r = a6r + tr * g3 * g3 ; a6i = a6i + ti * g3 * g3
       ENDDO
       s_d(1,na) = a1r ; s_d(2,na) = a1i ; s_d(3,na) = a2r ; s_d(4,na) = a2i
       s_d(5,na) = a3r ; s_d(6,na) = a3i ; s_d(7,na) = a4r ; s_d(8,na) = a4i
       s_d(9,na) = a5r ; s_d(10,na) = a5i ; s_d(11,na) = a6r ; s_d(12,na) = a6i
    ENDDO
    s_h = s_d
    DO na = 1, nat
       DO p = 1, 6
          s(p,na) = CMPLX( s_h(2*p-1,na), s_h(2*p,na), KIND=DP )
       ENDDO
    ENDDO
    DEALLOCATE( c_d, z_d, s_d, tau_d, ityp_d, s_h )
    !
  END SUBROUTINE dyn_d2form
  !
  !-----------------------------------------------------------------------
  SUBROUTINE d2ionq_g_dev( nat, ntyp, ityp, zv, tau, q, ngm, g, fac, facq, nat_l, atomo_l, dy3 )
    !-----------------------------------------------------------------------
    !! Reciprocal-space sums of d2ionq (same formulas), added to dy3.
    !! With all atoms displaced the pairs (nb,na) with nb < na are the
    !! conjugate of (na,nb) and are not computed.
    !
    USE constants, ONLY : tpi
    !
    INTEGER, INTENT(IN) :: nat, ntyp, ngm, nat_l
    INTEGER, INTENT(IN) :: ityp(nat), atomo_l(*)
    REAL(DP), INTENT(IN) :: zv(ntyp), tau(3,nat), q(3), g(3,ngm), fac(ngm), facq(ngm)
    COMPLEX(DP), INTENT(INOUT) :: dy3(3*nat,3*nat)
    !
    REAL(DP), ALLOCATABLE, DEVICE :: g_d(:,:), fac_d(:), facq_d(:), tau_d(:,:), zv_d(:), r_d(:,:,:)
    INTEGER, ALLOCATABLE, DEVICE :: ityp_d(:), atl_d(:)
    REAL(DP), ALLOCATABLE :: r_h(:,:,:)
    INTEGER, ALLOCATABLE :: atl_h(:)
    INTEGER :: na_l, na, nb, ng, na_i, nb_i, ic, jc, p, istat, natl, ngl
    LOGICAL :: sym
    REAL(DP) :: q1, q2, q3, d1, d2, d3, zvab, argq, arg, fn, fr, fi, g1, g2, g3, p1, p2, p3, &
                a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i, a5r, a5i, a6r, a6i, b1, b2, b3, b4, b5, b6
    COMPLEX(DP) :: dy1(3,3), dy2(3,3)
    INTEGER, PARAMETER :: pidx(3,3) = RESHAPE( (/ 1, 2, 4, 2, 3, 5, 4, 5, 6 /), (/ 3, 3 /) )
    !
    sym = ( nat_l == nat )
    ALLOCATE( g_d(3,ngm), fac_d(ngm), facq_d(ngm), tau_d(3,nat), zv_d(ntyp), ityp_d(nat), &
              atl_d(nat_l), r_d(18,nat,nat_l), STAT=istat )
    IF ( istat /= 0 ) CALL errore( 'd2ionq_g_dev', 'device allocation failed', 1 )
    ALLOCATE( r_h(18,nat,nat_l), atl_h(nat_l) )
    DO na_l = 1, nat_l
       IF ( nat_l < nat ) THEN
          atl_h(na_l) = atomo_l(na_l)
       ELSE
          atl_h(na_l) = na_l
       ENDIF
    ENDDO
    g_d = g ; fac_d = fac ; facq_d = facq ; tau_d = tau ; zv_d = zv ; ityp_d = ityp ; atl_d = atl_h
    q1 = q(1) ; q2 = q(2) ; q3 = q(3)
    natl = nat ; ngl = ngm
    !
    !$acc parallel loop gang collapse(2) &
    !$acc   private(na, zvab, d1, d2, d3, argq, a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i, a5r, a5i, &
    !$acc           a6r, a6i, b1, b2, b3, b4, b5, b6)
    DO na_l = 1, nat_l
       DO nb = 1, natl
          na = atl_d(na_l)
          a1r = 0.0_DP ; a1i = 0.0_DP ; a2r = 0.0_DP ; a2i = 0.0_DP ; a3r = 0.0_DP ; a3i = 0.0_DP
          a4r = 0.0_DP ; a4i = 0.0_DP ; a5r = 0.0_DP ; a5i = 0.0_DP ; a6r = 0.0_DP ; a6i = 0.0_DP
          b1 = 0.0_DP ; b2 = 0.0_DP ; b3 = 0.0_DP ; b4 = 0.0_DP ; b5 = 0.0_DP ; b6 = 0.0_DP
          IF ( .NOT. sym .OR. nb >= na ) THEN
             zvab = zv_d(ityp_d(na)) * zv_d(ityp_d(nb))
             d1 = tau_d(1,na) - tau_d(1,nb)
             d2 = tau_d(2,na) - tau_d(2,nb)
             d3 = tau_d(3,na) - tau_d(3,nb)
             argq = tpi * ( q1 * d1 + q2 * d2 + q3 * d3 )
             !$acc loop vector private(arg, fn, fr, fi, g1, g2, g3, p1, p2, p3) &
             !$acc      reduction(+:a1r,a1i,a2r,a2i,a3r,a3i,a4r,a4i,a5r,a5i,a6r,a6i,b1,b2,b3,b4,b5,b6)
             DO ng = 1, ngl
                g1 = g_d(1,ng) ; g2 = g_d(2,ng) ; g3 = g_d(3,ng)
                arg = tpi * ( g1 * d1 + g2 * d2 + g3 * d3 )
                fn = fac_d(ng) * zvab * COS(arg)
                fr = facq_d(ng) * zvab * COS(arg + argq)
                fi = facq_d(ng) * zvab * SIN(arg + argq)
                p1 = q1 + g1 ; p2 = q2 + g2 ; p3 = q3 + g3
                a1r = a1r + fr * p1 * p1 ; a1i = a1i + fi * p1 * p1
                a2r = a2r + fr * p2 * p1 ; a2i = a2i + fi * p2 * p1
                a3r = a3r + fr * p2 * p2 ; a3i = a3i + fi * p2 * p2
                a4r = a4r + fr * p3 * p1 ; a4i = a4i + fi * p3 * p1
                a5r = a5r + fr * p3 * p2 ; a5i = a5i + fi * p3 * p2
                a6r = a6r + fr * p3 * p3 ; a6i = a6i + fi * p3 * p3
                b1 = b1 + fn * g1 * g1
                b2 = b2 + fn * g2 * g1
                b3 = b3 + fn * g2 * g2
                b4 = b4 + fn * g3 * g1
                b5 = b5 + fn * g3 * g2
                b6 = b6 + fn * g3 * g3
             ENDDO
          ENDIF
          r_d(1,nb,na_l) = a1r ; r_d(2,nb,na_l) = a1i ; r_d(3,nb,na_l) = a2r ; r_d(4,nb,na_l) = a2i
          r_d(5,nb,na_l) = a3r ; r_d(6,nb,na_l) = a3i ; r_d(7,nb,na_l) = a4r ; r_d(8,nb,na_l) = a4i
          r_d(9,nb,na_l) = a5r ; r_d(10,nb,na_l) = a5i ; r_d(11,nb,na_l) = a6r ; r_d(12,nb,na_l) = a6i
          r_d(13,nb,na_l) = b1 ; r_d(14,nb,na_l) = b2 ; r_d(15,nb,na_l) = b3
          r_d(16,nb,na_l) = b4 ; r_d(17,nb,na_l) = b5 ; r_d(18,nb,na_l) = b6
       ENDDO
    ENDDO
    r_h = r_d
    !
    IF ( sym ) THEN
       DO na = 1, nat
          DO nb = 1, na-1
             DO p = 1, 6
                r_h(2*p-1,nb,na) = r_h(2*p-1,na,nb)
                r_h(2*p,nb,na) = - r_h(2*p,na,nb)
             ENDDO
             r_h(13:18,nb,na) = r_h(13:18,na,nb)
          ENDDO
       ENDDO
    ENDIF
    !
    ! ... assembled as in the stock routine
    !
    DO na_l = 1, nat_l
       na = atl_h(na_l)
       na_i = 3 * (na - 1)
       dy2(:,:) = (0.d0, 0.d0)
       DO nb = 1, nat
          nb_i = 3 * (nb - 1)
          DO jc = 1, 3
             DO ic = 1, 3
                p = pidx(ic,jc)
                dy1(ic,jc) = CMPLX( r_h(2*p-1,nb,na_l), r_h(2*p,nb,na_l), KIND=DP )
                dy2(ic,jc) = dy2(ic,jc) + r_h(12+p,nb,na_l)
             ENDDO
          ENDDO
          dy3(na_i+1:na_i+3,nb_i+1:nb_i+3) = dy3(na_i+1:na_i+3,nb_i+1:nb_i+3) + dy1(1:3,1:3)
       ENDDO
       dy3(na_i+1:na_i+3,na_i+1:na_i+3) = dy3(na_i+1:na_i+3,na_i+1:na_i+3) - dy2(1:3,1:3)
    ENDDO
    DEALLOCATE( g_d, fac_d, facq_d, tau_d, zv_d, ityp_d, atl_d, r_d, r_h, atl_h )
    !
  END SUBROUTINE d2ionq_g_dev
  !
  !-----------------------------------------------------------------------
  SUBROUTINE host_to_dev( n, x, x_d )
    !-----------------------------------------------------------------------
    INTEGER, INTENT(IN) :: n
    COMPLEX(DP), INTENT(IN) :: x(n)
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: x_d(n)
    x_d = x
  END SUBROUTINE host_to_dev
  !
  !-----------------------------------------------------------------------
  SUBROUTINE drhodvus_dev( imode0, npe, dvscfin, dyn1, done )
    !-----------------------------------------------------------------------
    !! The scalar products of drhodvus: for the npe perturbations of a
    !! representation and all the modes nu_j,
    !! dyn1(imode0+mu,nu_j) += omega/nrtot sum_r conj(dvscfin(r,mu)) drhous(r,nu_j).
    !! The stock routine reads the 3*nat records of drhous from their
    !! buffer for every representation; here they are read once per
    !! q-point and kept on the device (on the host, when the device memory
    !! allowed does not hold them), and the products of a representation are
    !! one ZGEMM. done = .false. when the records fit in neither (the caller
    !! uses the stock loop).
    !
    USE ions_base,        ONLY : nat
    USE cell_base,        ONLY : omega
    USE fft_base,         ONLY : dfftp
    USE noncollin_module, ONLY : nspin_mag
    USE buffers,          ONLY : get_buffer
    USE units_ph,         ONLY : iudrhous, lrdrhous
    USE qpoint,           ONLY : xq
    USE dvbare_dev,       ONLY : dvbare_blas
    !
    INTEGER, INTENT(IN) :: imode0, npe
    COMPLEX(DP), INTENT(IN) :: dvscfin(dfftp%nnr*nspin_mag,npe)
    COMPLEX(DP), INTENT(INOUT) :: dyn1(3*nat,3*nat)
    LOGICAL, INTENT(OUT) :: done
    !
    COMPLEX(DP), ALLOCATABLE, DEVICE :: dvs_d(:,:), res_d(:,:)
    COMPLEX(DP), ALLOCATABLE :: drhous(:,:), res(:,:)
    TYPE(cublasHandle) :: h
    INTEGER(KIND=cuda_count_kind) :: mfree, mtot
    INTEGER :: n, nm, nu_j, mu, istat
    REAL(DP) :: need
    COMPLEX(DP) :: alpha
    !
    done = .FALSE.
    n = dfftp%nnr * nspin_mag
    nm = 3 * nat
    CALL dvbare_blas( h )
    !
    IF ( .NOT. ( drus_ready .AND. ALL( xq(:) == drus_q(:) ) .AND. n == drus_n .AND. &
                 nm == drus_nm ) ) THEN
       CALL dyn_dev_reset( )
       need = 16.0_DP * DBLE(n) * DBLE(nm)
       istat = cudaMemGetInfo( mfree, mtot )
       IF ( need > max_drus_bytes .OR. need > max_drus_frac * DBLE(mfree) ) THEN
          IF ( need > max_drus_host ) RETURN
          ALLOCATE( drus_h(n,nm), STAT=istat )
          IF ( istat /= 0 ) RETURN
          DO nu_j = 1, nm
             CALL get_buffer( drus_h(1,nu_j), lrdrhous, iudrhous, nu_j )
          ENDDO
       ELSE
          ALLOCATE( drus_d(n,nm), STAT=istat )
          IF ( istat /= 0 ) RETURN
          ALLOCATE( drhous(dfftp%nnr,nspin_mag) )
          DO nu_j = 1, nm
             CALL get_buffer( drhous, lrdrhous, iudrhous, nu_j )
             CALL host_to_dev( n, drhous, drus_d(1,nu_j) )
          ENDDO
          DEALLOCATE( drhous )
       ENDIF
       drus_q(:) = xq(:) ; drus_n = n ; drus_nm = nm
       drus_ready = .TRUE.
    ENDIF
    !
    alpha = CMPLX( omega / DBLE( dfftp%nr1 * dfftp%nr2 * dfftp%nr3 ), 0.0_DP, KIND=DP )
    ALLOCATE( res(npe,nm) )
    IF ( ALLOCATED(drus_h) ) THEN
       CALL ZGEMM( 'C', 'N', npe, nm, n, alpha, dvscfin, n, drus_h, n, c_zero, res, npe )
    ELSE
       ALLOCATE( dvs_d(n,npe), res_d(npe,nm), STAT=istat )
       IF ( istat /= 0 ) THEN
          DEALLOCATE( res )
          RETURN
       ENDIF
       dvs_d = dvscfin
       istat = cublasZgemm( h, CUBLAS_OP_C, CUBLAS_OP_N, npe, nm, n, alpha, dvs_d, n, drus_d, n, &
                            c_zero, res_d, npe )
       IF ( istat /= 0 ) CALL errore( 'drhodvus_dev', 'cublasZgemm failed', ABS(istat) )
       res = res_d
       DEALLOCATE( dvs_d, res_d )
    ENDIF
    DO nu_j = 1, nm
       DO mu = 1, npe
          dyn1(imode0+mu,nu_j) = dyn1(imode0+mu,nu_j) + res(mu,nu_j)
       ENDDO
    ENDDO
    DEALLOCATE( res )
    done = .TRUE.
    !
  END SUBROUTINE drhodvus_dev
  !
  !-----------------------------------------------------------------------
  SUBROUTINE drhodv_nl_dev( nu_i0, nper, dynwrk )
    !-----------------------------------------------------------------------
    !! Nonlocal part of drhodv (the loop over k-points of drhodv with
    !! drhodvnl), collinear case: the projections of \(\Delta\psi\) and of
    !! \((k+q+G)_\alpha\Delta\psi\) on the beta functions are one ZGEMM per
    !! block of bands on the device, the sums over bands and projectors
    !! run on the host (drhodv_nl_k, drhodv_nl_int2). The result (rows
    !! nu_i0+1..nu_i0+nper, Cartesian columns) is added to dynwrk.
    !
    USE ions_base,     ONLY : nat
    USE klist,         ONLY : xk, ngk, igk_k
    USE gvect,         ONLY : g
    USE lsda_mod,      ONLY : current_spin, lsda, isk
    USE wvfct,         ONLY : npwx, nbnd
    USE uspp,          ONLY : nkb, vkb, okvan
    USE uspp_param,    ONLY : nhm
    USE uspp_init,     ONLY : init_us_2
    USE buffers,       ONLY : get_buffer
    USE units_lr,      ONLY : lrdwf, iudwf
    USE eqv,           ONLY : dpsi
    USE qpoint,        ONLY : nksq, ikks, ikqs
    USE dvbare_dev,    ONLY : dvbare_blas
    USE dfpt_dev,      ONLY : dfpt_dev_get_bq, dfpt_dev_kept
    !
    INTEGER, INTENT(IN) :: nu_i0, nper
    COMPLEX(DP), INTENT(INOUT) :: dynwrk(3*nat,3*nat)
    !
    COMPLEX(DP), ALLOCATABLE, DEVICE :: dps_d(:,:), aux_d(:,:), bec_d(:,:)
    REAL(DP), ALLOCATABLE, DEVICE :: kqg_d(:,:)
    COMPLEX(DP), ALLOCATABLE :: bq(:,:,:), cus(:,:,:,:), bqa(:,:,:,:)
    REAL(DP), ALLOCATABLE :: kqg_h(:,:)
    LOGICAL :: from_engine
    TYPE(cublasHandle) :: h
    INTEGER :: ik, ikk, ikq, npwq, mu, nrec, j0, nbl, nbc, j, ig, istat
    !
    IF ( nkb == 0 ) RETURN
    !
    ! ... the response wavefunctions are still on the device (engine dfpt_dev):
    ! ... projections of all k-points from there, no buffer and no beta functions
    !
    from_engine = dfpt_dev_kept( nper )
    IF ( from_engine ) THEN
       ALLOCATE( bqa(nkb,4*nbnd,nper,nksq) )
       CALL dfpt_dev_get_bq( nper, nbnd, bqa, from_engine )
       IF ( .NOT. from_engine ) DEALLOCATE( bqa )
    ENDIF
    IF ( from_engine ) THEN
       ALLOCATE( cus(nhm,nhm,nat,nper) )
       cus(:,:,:,:) = (0.0_DP, 0.0_DP)
       DO ik = 1, nksq
          IF ( lsda ) current_spin = isk(ikks(ik))
          CALL drhodv_nl_k( ik, nu_i0, nper, bqa(:,:,:,ik), dynwrk, cus )
       ENDDO
       IF ( okvan ) CALL drhodv_nl_int2( nu_i0, nper, cus, dynwrk )
       DEALLOCATE( bqa, cus )
       RETURN
    ENDIF
    !
    CALL dvbare_blas( h )
    nbc = MAX( 1, MIN( nbnd, max_tmp_elems / (4*npwx) ) )
    ALLOCATE( dps_d(npwx,nbnd), aux_d(npwx,4*nbc), bec_d(nkb,4*nbnd), kqg_d(3,npwx), STAT=istat )
    IF ( istat /= 0 ) CALL errore( 'drhodv_nl_dev', 'device allocation failed', 1 )
    ALLOCATE( bq(nkb,4*nbnd,nper), kqg_h(3,npwx) )
    ALLOCATE( cus(nhm,nhm,nat,nper) )
    cus(:,:,:,:) = (0.0_DP, 0.0_DP)
    !
    DO ik = 1, nksq
       ikk = ikks(ik) ; ikq = ikqs(ik)
       npwq = ngk(ikq)
       IF ( lsda ) current_spin = isk(ikk)
       CALL init_us_2( npwq, igk_k(1,ikq), xk(1,ikq), vkb, .TRUE. )
       DO ig = 1, npwq
          kqg_h(:,ig) = xk(:,ikq) + g(:,igk_k(ig,ikq))
       ENDDO
       kqg_h(:,npwq+1:npwx) = 0.0_DP
       kqg_d = kqg_h
       !
       ! ... <beta|dpsi> and <beta|(k+q+G)_alpha dpsi> of all perturbations
       !
       DO mu = 1, nper
          nrec = (mu - 1) * nksq + ik
          IF ( nksq > 1 .OR. nper > 1 ) CALL get_buffer( dpsi, lrdwf, iudwf, nrec )
          dps_d = dpsi
          DO j0 = 0, nbnd-1, nbc
             nbl = MIN( nbc, nbnd-j0 )
             !$cuf kernel do(2)
             DO j = 1, nbl
                DO ig = 1, npwq
                   aux_d(ig,4*(j-1)+1) = dps_d(ig,j0+j)
                   aux_d(ig,4*(j-1)+2) = dps_d(ig,j0+j) * kqg_d(1,ig)
                   aux_d(ig,4*(j-1)+3) = dps_d(ig,j0+j) * kqg_d(2,ig)
                   aux_d(ig,4*(j-1)+4) = dps_d(ig,j0+j) * kqg_d(3,ig)
                ENDDO
             ENDDO
             !$acc host_data use_device(vkb)
             istat = cublasZgemm( h, CUBLAS_OP_C, CUBLAS_OP_N, nkb, 4*nbl, npwq, c_one, vkb, npwx, &
                                  aux_d, npwx, c_zero, bec_d(1,4*j0+1), nkb )
             !$acc end host_data
             IF ( istat /= 0 ) CALL errore( 'drhodv_nl_dev', 'cublasZgemm failed', ABS(istat) )
          ENDDO
          bq(:,:,mu) = bec_d
       ENDDO
       !
       CALL drhodv_nl_k( ik, nu_i0, nper, bq, dynwrk, cus )
       !
    ENDDO
    !
    IF ( okvan ) CALL drhodv_nl_int2( nu_i0, nper, cus, dynwrk )
    DEALLOCATE( dps_d, aux_d, bec_d, kqg_d, bq, kqg_h, cus )
    !
  END SUBROUTINE drhodv_nl_dev
  !
  !-----------------------------------------------------------------------
  SUBROUTINE drhodv_nl_k( ik, nu_i0, nper, bq, dynwrk, cus )
    !-----------------------------------------------------------------------
    !! Contribution of k-point ik to the nonlocal part of drhodv (the sums
    !! of drhodvnl, collinear case), from the projections of the nper
    !! responses on the beta functions of k+q:
    !!
    !! bq(:,4*(ibnd-1)+1,iper) = <beta|dpsi_ibnd>,
    !! bq(:,4*(ibnd-1)+1+alpha,iper) = <beta|(k+q+G)_alpha dpsi_ibnd>
    !! (units 2pi/a, without the factor i tpiba).
    !!
    !! The terms with D - eS and int1 are added to dynwrk; for ultrasoft
    !! PPs the sums over the bands needed by the term with int2 are
    !! accumulated in cus (see drhodv_nl_int2). The current spin must be
    !! set. Host routine: it can be fed by any code that has the projections.
    !
    USE ions_base,  ONLY : nat, ityp
    USE klist,      ONLY : wk
    USE cell_base,  ONLY : tpiba
    USE lsda_mod,   ONLY : current_spin
    USE wvfct,      ONLY : nbnd, et
    USE uspp,       ONLY : nkb, okvan, ofsbeta, deeq, qq_at
    USE uspp_param, ONLY : nh, nhm
    USE qpoint,     ONLY : ikks
    USE lrus,       ONLY : becp1
    USE phus,       ONLY : alphap, int1
    !
    INTEGER, INTENT(IN) :: ik, nu_i0, nper
    COMPLEX(DP), INTENT(IN) :: bq(nkb,4*nbnd,nper)
    COMPLEX(DP), INTENT(INOUT) :: dynwrk(3*nat,3*nat)
    COMPLEX(DP), INTENT(INOUT) :: cus(nhm,nhm,nat,nper)
    !
    COMPLEX(DP), ALLOCATABLE :: pp1(:), pp2(:,:), acc(:,:)
    INTEGER :: ikk, na, nb, nt, ijkb0, nhn, ibnd, ih, jh, ikb, jkb, ipol, iper, is, ib
    REAL(DP) :: wgt, d, e
    COMPLEX(DP) :: fact, p1, p2(3), bj, s
    !
    ikk = ikks(ik)
    is = current_spin
    fact = CMPLX( 0.d0, tpiba, KIND=DP )
    wgt = 2.d0 * wk(ikk)
    !
    !$omp parallel private(pp1, pp2, acc, na, nt, ijkb0, nhn, ibnd, e, ih, jh, ikb, jkb, ipol, &
    !$omp                  iper, d, bj, p1, p2, ib) if(nkb*nbnd > 5000)
    ALLOCATE( pp1(nhm), pp2(nhm,3), acc(3,nper) )
    !$omp do
    DO na = 1, nat
       nt = ityp(na)
       ijkb0 = ofsbeta(na)
       nhn = nh(nt)
       acc(:,:) = (0.0_DP, 0.0_DP)
       DO ibnd = 1, nbnd
          e = et(ibnd,ikk)
          ib = 4 * (ibnd - 1)
          DO ih = 1, nhn
             p1 = (0.0_DP, 0.0_DP)
             p2(:) = (0.0_DP, 0.0_DP)
             DO jh = 1, nhn
                jkb = ijkb0 + jh
                d = deeq(ih,jh,na,is)
                IF ( okvan ) d = d - e * qq_at(ih,jh,na)
                bj = becp1(ik)%k(jkb,ibnd)
                p1 = p1 + d * bj
                DO ipol = 1, 3
                   p2(ipol) = p2(ipol) + d * alphap(ipol,ik)%k(jkb,ibnd)
                   IF ( okvan ) p2(ipol) = p2(ipol) + int1(ih,jh,ipol,na,is) * bj
                ENDDO
             ENDDO
             pp1(ih) = p1
             pp2(ih,:) = p2(:)
          ENDDO
          DO ipol = 1, 3
             DO ih = 1, nhn
                ikb = ijkb0 + ih
                DO iper = 1, nper
                   acc(ipol,iper) = acc(ipol,iper) + wgt * ( pp2(ih,ipol) * &
                        CONJG( bq(ikb,ib+1,iper) ) + pp1(ih) * CONJG( fact * bq(ikb,ib+1+ipol,iper) ) )
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       DO iper = 1, nper
          DO ipol = 1, 3
             dynwrk(nu_i0+iper,3*(na-1)+ipol) = dynwrk(nu_i0+iper,3*(na-1)+ipol) + acc(ipol,iper)
          ENDDO
       ENDDO
    ENDDO
    !$omp end do
    DEALLOCATE( pp1, pp2, acc )
    !$omp end parallel
    !
    ! ... ultrasoft: 2 wk sum_v conj(<beta_i|dpsi_v>) <beta_j|psi_v> for the term with int2
    !
    IF ( okvan ) THEN
       !$omp parallel do private(iper, nb, nt, ijkb0, ih, jh, ibnd, s) if(nkb*nbnd > 5000)
       DO nb = 1, nat
          nt = ityp(nb)
          ijkb0 = ofsbeta(nb)
          DO iper = 1, nper
             DO jh = 1, nh(nt)
                DO ih = 1, nh(nt)
                   s = (0.0_DP, 0.0_DP)
                   DO ibnd = 1, nbnd
                      s = s + CONJG( bq(ijkb0+ih,4*(ibnd-1)+1,iper) ) * becp1(ik)%k(ijkb0+jh,ibnd)
                   ENDDO
                   cus(ih,jh,nb,iper) = cus(ih,jh,nb,iper) + wgt * s
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       !$omp end parallel do
    ENDIF
    !
  END SUBROUTINE drhodv_nl_k
  !
  !-----------------------------------------------------------------------
  SUBROUTINE drhodv_nl_int2( nu_i0, nper, cus, dynwrk )
    !-----------------------------------------------------------------------
    !! Ultrasoft term of drhodvnl with int2, from the sums over k-points
    !! and bands accumulated by drhodv_nl_k:
    !! dynwrk(nu,(na,alpha)) += sum_{nb,ih,jh} int2(ih,jh,alpha,na,nb) cus(ih,jh,nb,iper)
    !
    USE ions_base,  ONLY : nat, ityp
    USE uspp_param, ONLY : nh, nhm
    USE phus,       ONLY : int2
    !
    INTEGER, INTENT(IN) :: nu_i0, nper
    COMPLEX(DP), INTENT(IN) :: cus(nhm,nhm,nat,nper)
    COMPLEX(DP), INTENT(INOUT) :: dynwrk(3*nat,3*nat)
    INTEGER :: na, nb, nt, ih, jh, ipol, iper
    COMPLEX(DP) :: s
    !
    !$omp parallel do private(na, nb, nt, ih, jh, ipol, iper, s) if(nat > 8)
    DO na = 1, nat
       DO iper = 1, nper
          DO ipol = 1, 3
             s = (0.0_DP, 0.0_DP)
             DO nb = 1, nat
                nt = ityp(nb)
                DO jh = 1, nh(nt)
                   DO ih = 1, nh(nt)
                      s = s + int2(ih,jh,ipol,na,nb) * cus(ih,jh,nb,iper)
                   ENDDO
                ENDDO
             ENDDO
             dynwrk(nu_i0+iper,3*(na-1)+ipol) = dynwrk(nu_i0+iper,3*(na-1)+ipol) + s
          ENDDO
       ENDDO
    ENDDO
    !$omp end parallel do
    !
  END SUBROUTINE drhodv_nl_int2
  !
  !-----------------------------------------------------------------------
  SUBROUTINE dynmat_us_nl_dev( dynwrk )
    !-----------------------------------------------------------------------
    !! Nonlocal part of dynmat_us (the loop over k-points), collinear
    !! case: the second derivatives of <beta|psi> are one ZGEMM per block
    !! of bands on the device, the sums over bands and projectors run on
    !! the host. The result is added to dynwrk (Cartesian basis).
    !
    USE ions_base,     ONLY : nat, ityp
    USE klist,         ONLY : xk, ngk, igk_k
    USE gvect,         ONLY : g
    USE cell_base,     ONLY : tpiba2
    USE lsda_mod,      ONLY : current_spin, lsda, isk
    USE wvfct,         ONLY : npwx, nbnd, et, wg
    USE wavefunctions, ONLY : evc
    USE uspp,          ONLY : nkb, vkb, okvan, ofsbeta, deeq, qq_at
    USE uspp_param,    ONLY : nh
    USE uspp_init,     ONLY : init_us_2
    USE buffers,       ONLY : get_buffer
    USE units_lr,      ONLY : iuwfc, lrwfc
    USE qpoint,        ONLY : nksq, ikks
    USE control_lr,    ONLY : nbnd_occ
    USE lrus,          ONLY : becp1
    USE phus,          ONLY : alphap
    USE dvbare_dev,    ONLY : dvbare_blas
    !
    COMPLEX(DP), INTENT(INOUT) :: dynwrk(3*nat,3*nat)
    !
    COMPLEX(DP), ALLOCATABLE, DEVICE :: aux_d(:,:), gam_d(:,:)
    REAL(DP), ALLOCATABLE, DEVICE :: kg_d(:,:)
    COMPLEX(DP), ALLOCATABLE :: gam(:,:)
    REAL(DP), ALLOCATABLE :: kg_h(:,:)
    TYPE(cublasHandle) :: h
    INTEGER :: ik, ikk, npw, nbo, nbc, j0, nbl, j, ig, istat, na, nt, ijkb0, nhn, ibnd, ih, jh, &
               ikb, jkb, icart, jcart, na_icart, na_jcart, is, p, ib
    REAL(DP) :: wgg, d, e, tp2
    INTEGER, PARAMETER :: pidx(3,3) = RESHAPE( (/ 1, 2, 4, 2, 3, 5, 4, 5, 6 /), (/ 3, 3 /) )
    !
    IF ( nkb == 0 ) RETURN
    CALL dvbare_blas( h )
    nbc = MAX( 1, MIN( nbnd, max_tmp_elems / (6*npwx) ) )
    ALLOCATE( aux_d(npwx,6*nbc), gam_d(nkb,6*nbnd), kg_d(3,npwx), STAT=istat )
    IF ( istat /= 0 ) CALL errore( 'dynmat_us_nl_dev', 'device allocation failed', 1 )
    ALLOCATE( gam(nkb,6*nbnd), kg_h(3,npwx) )
    tp2 = tpiba2
    !
    DO ik = 1, nksq
       ikk = ikks(ik)
       IF ( lsda ) current_spin = isk(ikk)
       is = current_spin
       npw = ngk(ikk)
       nbo = nbnd_occ(ikk)
       IF ( nksq > 1 ) CALL get_buffer( evc, lrwfc, iuwfc, ikk )
       !
       ! ... beta functions on the device (every routine that needs them on the
       ! ... host asks for them again; the copy of nkb*npwx numbers costs
       ! ... seconds in large cells)
       !
       CALL init_us_2( npw, igk_k(1,ikk), xk(1,ikk), vkb, .TRUE. )
       IF ( nbo < 1 ) CYCLE
       DO ig = 1, npw
          kg_h(:,ig) = xk(:,ikk) + g(:,igk_k(ig,ikk))
       ENDDO
       kg_h(:,npw+1:npwx) = 0.0_DP
       kg_d = kg_h
       !
       ! ... gamma terms: <beta| -tpiba2 (k+G)_i (k+G)_j |psi>, pairs 11, 21, 22, 31, 32, 33
       !
       !$acc update device(evc)
       DO j0 = 0, nbo-1, nbc
          nbl = MIN( nbc, nbo-j0 )
          !$acc parallel loop collapse(2) present(evc)
          DO j = 1, nbl
             DO ig = 1, npw
                aux_d(ig,6*(j-1)+1) = - evc(ig,j0+j) * tp2 * kg_d(1,ig) * kg_d(1,ig)
                aux_d(ig,6*(j-1)+2) = - evc(ig,j0+j) * tp2 * kg_d(2,ig) * kg_d(1,ig)
                aux_d(ig,6*(j-1)+3) = - evc(ig,j0+j) * tp2 * kg_d(2,ig) * kg_d(2,ig)
                aux_d(ig,6*(j-1)+4) = - evc(ig,j0+j) * tp2 * kg_d(3,ig) * kg_d(1,ig)
                aux_d(ig,6*(j-1)+5) = - evc(ig,j0+j) * tp2 * kg_d(3,ig) * kg_d(2,ig)
                aux_d(ig,6*(j-1)+6) = - evc(ig,j0+j) * tp2 * kg_d(3,ig) * kg_d(3,ig)
             ENDDO
          ENDDO
          !$acc host_data use_device(vkb)
          istat = cublasZgemm( h, CUBLAS_OP_C, CUBLAS_OP_N, nkb, 6*nbl, npw, c_one, vkb, npwx, &
                               aux_d, npwx, c_zero, gam_d(1,6*j0+1), nkb )
          !$acc end host_data
          IF ( istat /= 0 ) CALL errore( 'dynmat_us_nl_dev', 'cublasZgemm failed', ABS(istat) )
       ENDDO
       gam = gam_d
       !
       ! ... sums over bands and projectors, as in the stock routine
       !
       !$omp parallel do private(na, nt, ijkb0, nhn, ibnd, wgg, e, ib, icart, jcart, na_icart, &
       !$omp                     na_jcart, p, ih, jh, ikb, jkb, d) if(nkb*nbo > 2000)
       DO na = 1, nat
          nt = ityp(na)
          ijkb0 = ofsbeta(na)
          nhn = nh(nt)
          DO ibnd = 1, nbo
             wgg = wg(ibnd,ikk)
             e = et(ibnd,ikk)
             ib = 6 * (ibnd - 1)
             DO icart = 1, 3
                na_icart = 3 * (na - 1) + icart
                DO jcart = 1, 3
                   na_jcart = 3 * (na - 1) + jcart
                   p = ib + pidx(icart,jcart)
                   DO ih = 1, nhn
                      ikb = ijkb0 + ih
                      DO jh = 1, nhn
                         jkb = ijkb0 + jh
                         d = deeq(ih,jh,na,is)
                         IF ( okvan ) d = d - e * qq_at(ih,jh,na)
                         dynwrk(na_icart,na_jcart) = dynwrk(na_icart,na_jcart) + d * wgg * &
                              ( CONJG( gam(ikb,p) ) * becp1(ik)%k(jkb,ibnd) + &
                                CONJG( becp1(ik)%k(ikb,ibnd) ) * gam(jkb,p) + &
                                CONJG( alphap(icart,ik)%k(ikb,ibnd) ) * alphap(jcart,ik)%k(jkb,ibnd) + &
                                CONJG( alphap(jcart,ik)%k(ikb,ibnd) ) * alphap(icart,ik)%k(jkb,ibnd) )
                      ENDDO
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       !$omp end parallel do
    ENDDO
    DEALLOCATE( aux_d, gam_d, kg_d, gam, kg_h )
    !
  END SUBROUTINE dynmat_us_nl_dev
  !
#endif
  !
END MODULE dyn_dev

subroutine el_field_slab(x,a,lambda,eps,elfield)
  ! El. field (along x) of a slab of thickness a along z, infinitely extended along y,
  ! put at x=0, calculated at the point (x,0,0)
  !
  ! We don't use it anymore, the limit a->0 is directly usable (see el_field_wire function)
  implicit none
  double precision, intent(in) :: x ! distance from the slab, in angstrom
  double precision, intent(in) :: a ! thickness along z, in angstrom
  double precision, intent(in) :: lambda ! linear charge density in e/cm
  double precision, intent(in) :: eps ! relative dielectric constant at the point at which
                                      ! we want to calculate the electric field (adimensional)
  double precision :: elfield ! V/ang

  double precision :: PI, FOURPIEPS0
  parameter(PI=3.1415926535897932d0)
  ! FOURPIEPS0 = 4*pi*epsilon_0 in e/(volt * cm)
  parameter(FOURPIEPS0=6944615.72)

  if (x .gt. 0) then
     elfield = (4.d0*lambda*acos((2.d0*x)/Sqrt(a**2 + 4.d0*x**2)))/a
  elseif (x.eq.0) then
     elfield = 0.d0
  else
     elfield = 4.d0*lambda*(-PI + acos((2.d0*x)/Sqrt(a**2 + 4.d0*x**2)))/a
  end if

  ! with the given units, I get in this way the electric field in V/ang
  elfield = elfield / FOURPIEPS0 / eps

end subroutine el_field_slab

subroutine el_field_wire(x,lambda,eps,elfield)
  implicit none
  ! El. field (along x) of a wire at x=z=0, infinitely extended along y,
  ! calculated at the point (x,0,0)
  double precision, intent(in) :: x ! distance from the wire, in angstrom
  double precision, intent(in) :: lambda ! linear charge density in e/cm
  double precision, intent(in) :: eps ! relative dielectric constant at the point at which
                                      ! we want to calculate the electric field (adimensional)
  double precision :: elfield ! V/ang

  double precision :: PI, FOURPIEPS0
  parameter(PI=3.1415926535897932d0)
  ! FOURPIEPS0 = 4*pi*epsilon_0 in e/(volt * cm)
  parameter(FOURPIEPS0=6944615.72)

  if (x .ne. 0) then
     elfield = 2.d0*lambda/x
  else 
     ! To avoid over/underflow errors; in any case, one should not call it for x==0
     elfield = 0.d0
  end if

  ! with the given units, I get in this way the electric field in V/ang
  elfield = elfield / FOURPIEPS0 / eps
  
end subroutine el_field_wire

subroutine el_field_wire_array(x,period,lambda,eps,elfield)
  implicit none
  ! El. field (along x) of a wire array at z=0, infinitely extended along y, and x=n*period,
  ! where n in an integer (positive, negative or zero) and period is the periodicity of the array.
  ! The field is calculated at the point (x,0,0), where x should be between zero and period.
  ! The code internally brings the value within the [0,period] range.
  ! The el. field diverges as 1/x near 0 and period, so at the edges we set the value to zero.
  
  double precision, intent(in) :: x ! distance from the wire, in angstrom
  double precision, intent(in) :: period ! periodicity of the wire array, in angstrom
  double precision, intent(in) :: lambda ! linear charge density in e/cm
  double precision, intent(in) :: eps ! relative dielectric constant at the point at which
                                      ! we want to calculate the electric field (adimensional)
  double precision :: elfield ! V/ang
  double precision :: xi ! x/period in the [0,1[ range

  double precision :: PI, FOURPIEPS0
  parameter(PI=3.1415926535897932d0)
  ! FOURPIEPS0 = 4*pi*epsilon_0 in e/(volt * cm)
  parameter(FOURPIEPS0=6944615.72)

  ! Modulo brings in the [0,1[ interval, and does the correct thing for negative numbers
  xi = MODULO(x/period, 1.d0)

  if ((xi .eq. 0) .or. (xi .eq. 1)) then
     elfield = 0.d0 ! the xi == 1 should not be needed, but I put it just in case
  else 
     ! Cotangent is 1/tangent
     elfield = 2.d0 * PI * lambda / TAN(PI * xi) / period
  end if

  ! with the given units, I get in this way the electric field in V/ang
  elfield = elfield / FOURPIEPS0 / eps
  
end subroutine el_field_wire_array


subroutine v_of_rho_nonperiodic(v,x,lambda,eps,n)
  implicit none
  ! return the electrostatic potential, given the linear density of
  ! some 1d wires, where each wire is is at z=0, and extending along y.
  ! There is a wire for each discretization point of x, with charge lambda.
  ! The potential is set to zero at left edge.
  ! NOTE! This routine assumes that x is sorted, and equally spaced!
  
  ! INPUT
  ! x:      array with the discretization along x in angstrom
  ! lambda: linear charge density in e/cm on the grid defined by x
  ! eps:    relative dielectric constant on the grid defined by x
  ! n:      number of elements of the array

  ! OUTPUT
  ! v:      electrostatic energy in eV
  double precision, intent(out), dimension(n) :: v
  double precision, intent(in), dimension(n) :: x
  double precision, intent(in), dimension(n) :: lambda
  double precision, intent(in), dimension(n) :: eps
  integer, intent(in) :: n

  integer :: i, j
  
  double precision, dimension(n) :: efield ! stores temporarily the el. field
  double precision :: this_efield

  efield = 0.d0
  v = 0.d0

  ! I calculate the electric field in V/angstrom
  do i=1, n
     do j=1,n
        if (i.eq.j) cycle
        ! el.field generated at pos. i, generated by the slab with linear charge density lambda at pos. j
        call el_field_wire(x(i)-x(j),lambda(j),eps(i),this_efield)
        efield(i) = efield(i) + this_efield
     end do
  end do

  do i=1, n
     do j=1, i-1
     ! v is the integral of efield (times e); I set to zero at left edge
     ! here it is e * V/angstrom * angstrom = eV
     v(i) = v(i) + efield(j) * (x(i) - x(j))
     end do
  end do

end subroutine v_of_rho_nonperiodic

subroutine v_of_rho_periodic(v,x,lambda,eps,n)
  implicit none
  ! The same of v_of_rho_nonperiodic, but here the system is thought to be periodic,
  ! and the potential contribution of each point of the grid is not the contribution
  ! of a single wire, but of a periodic repetition of wires
  ! Assume that if x has N points, the (virtual) (N+1)-th point is identical to the
  ! first one
  double precision, intent(out), dimension(n) :: v
  double precision, intent(in), dimension(n) :: x
  double precision, intent(in), dimension(n) :: lambda
  double precision, intent(in), dimension(n) :: eps
  integer, intent(in) :: n

  integer :: i, j
  
  double precision, dimension(n) :: efield ! stores temporarily the el. field
  double precision :: this_efield, period

  efield = 0.d0
  v = 0.d0

  ! Assuming x is sorted, and equally spaced
  ! Assume that if x has N points, the (virtual) (N+1)-th point is identical to the
  ! first one. Remember also that len(x) = n
  period = (x(2) - x(1)) * n

  ! I calculate the electric field in V/angstrom
  do i=1, n
     do j=1,n
        ! even in periodic boundary conditions, the net contribution of 
        ! all periodic images is zero, by symmetry
        if (i.eq.j) cycle 
        ! el.field generated at pos. i, generated by the slab with linear charge density lambda at pos. j
        call el_field_wire_array(x(i)-x(j),period,lambda(j),eps(i),this_efield)
        efield(i) = efield(i) + this_efield
     end do
  end do

  do i=1, n
     do j=1, i-1
     ! v is the integral of efield (times e); I set to zero at left edge
     ! here it is e * V/angstrom * angstrom = eV
     v(i) = v(i) + efield(j) * (x(i) - x(j))
     end do
  end do

end subroutine v_of_rho_periodic

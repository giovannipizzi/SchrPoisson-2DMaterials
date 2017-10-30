!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! This file contains the main logic for calculating the potential due to a single
!! wire and to an array of wires. These functions (that are the most expensive part
!! of the code) are then called from python.
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!
!! If you use this code in your work, please cite the following paper:
!!
!! A. Bussy, G. Pizzi, M. Gibertini, Strain-induced polar discontinuities 
!! in 2D materials from combined first-principles and Schroedinger-Poisson 
!! simulations, Phys. Rev. B 96, 165438 (2017).
!! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!
!! This code is released under a MIT license, see LICENSE.txt file in the main 
!! folder of the code repository, hosted on GitHub at 
!! https://github.com/giovannipizzi/schrpoisson_2dmaterials
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!
!! FORTRAN code version: 1.0.0
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!==================================================================
!==================NONPERIODIC FUNCTIONS===========================

subroutine v_wire_nonperiodic(x,lambda,potential)
  !! This subroutine computes the unscreened potential due to an infinite (along y) 
  !! charged wire at a distance x

  double precision, intent(in) :: x 
  !! x is the distance in ang between the wire and the point of interest
  double precision, intent(in) :: lambda 
  !! lambda is the line charged density of the wire in e/cm

  double precision, intent(out) :: potential 
  !! in V
  double precision :: lamb 
  !! lambda in units of e/ang

  double precision :: PI, EPSILON0
  parameter(PI = 3.1415926535897932d0)
  parameter(EPSILON0 = 0.0055263496) ! in e/(V*ang)

  !first off, need lambda in e/ang
  lamb = lambda*1.E-8

  potential = -lamb/(2.*PI*EPSILON0) * LOG(x)

end subroutine v_wire_nonperiodic


subroutine v_array_wire_nonperiodic(coord,x,rho_in,potential,n)
  !! this subroutine computes the potential at a point from a charge density 
  !! (array of wires with different charges) 

  double precision, intent(in) :: coord 
  !! coordinate of point of interest along x axis (in ang)
  double precision, intent(in), dimension(n) :: x 
  !! the array of grid points in ang
  double precision, intent(in), dimension(n) :: rho_in 
  !! the charge density in e/cm defined at each grid point
  double precision, intent(out) :: potential 
  !! the potential in V at point coord due to array of wires
  integer, intent(in) :: n 
  !! dimension of all arrays

  double precision :: potr 
  !! temporary storing of potential on the right (for integration)
  double precision :: potl 
  !! temporary storing of potential on the left (for integration)

  integer :: i

  ! numerical integration to get potential
  do i=1,n-1 ! up to n-1 because we use trapezoidal method for integration
     call v_wire_nonperiodic(ABS(coord-x(i)),rho_in(i),potl)
     call v_wire_nonperiodic(ABS(coord-x(i+1)),rho_in(i+1),potr)
     potential = potential + 0.5*(potr+potl) !*(x(2)-x(1))
  end do
end subroutine v_array_wire_nonperiodic


subroutine full_v_array_wire_nonperiodic(x,rho_in,potential,n)
  !! this subroutine computes the potential at each grid point from a charge density 
  !! (array of wires with diffrent charges) 

  double precision, intent(in), dimension(n) :: x 
  !! the array of grid points in ang
  double precision, intent(in), dimension(n) :: rho_in 
  !! the charge density in e/cm defined at each grid point
  double precision, intent(out), dimension(n) :: potential 
  !! the potential in V at each grid point due to array of wires
  integer, intent(in) :: n 
  !! dimension of all arrays

  double precision :: coord
  double precision :: pot ! for temporary storage of potential
  integer :: i

  do i=1,n
     pot = 0.0 ! because of the += in the v_array_wire_periodic subroutine
     coord = (i-1)*(x(2)-x(1))
     call v_array_wire_nonperiodic(coord,x,rho_in,pot,n)
     potential(i) = pot
  end do

end subroutine full_v_array_wire_nonperiodic


subroutine nonperiodic_recursive_poisson(x,rho_in,alpha,max_iteration,potential,rho_out,n)
  !! this subroutine solves the screened Poisson equation for 2D materials 
  !! with different polarizabilities

  double precision, intent(in), dimension(n) :: x 
  !! the array containing the position of all grid points in ang
  double precision, intent(in), dimension(n) :: rho_in 
  !! initial charge density in e/cm defined at each grid point
  double precision, intent(in), dimension(n) :: alpha 
  !! the value of the polarizability at each grid point in e/V
  integer, intent(in) :: max_iteration 
  !! the maximum amount of allowed iterations for the recursive Poisson algorithm
  double precision, intent(out), dimension(n) :: potential 
  !! the solution of recursive Poisson in V defined at each grid point
  double precision, intent(out), dimension(n) :: rho_out 
  !! The total charge density including polarization charges in e/cm at each grid point
  integer, intent(in) :: n 
  !! dimension of all grid-related arrays

  integer :: i
  integer :: counter, subcounter
  integer, dimension(1) :: max_ind
  double precision :: beta
  double precision :: Vleft
  double precision, dimension(2) :: indicator ! used to check if there are oscillations while converging
  double precision :: h ! step size in ang
  double precision :: diff,diff_old ! allowed difference for convergence tests
  double precision, dimension(n) :: old_rho ! induced charge density before algorithm step
  double precision, dimension(n) :: new_rho 
  double precision, dimension(n) :: dV ! the space derivative of the potential at each point
  double precision, dimension(n) :: aldV ! product of alpha and dV at each grid point
  double precision, dimension(n) :: V ! temporary storage of the potential
  logical :: conv

  !RESOLUTION

  h = x(2)-x(1)

  ! First off, one computes the unscreened potential due to rho_in
  ! One computes it a half grid point on the right, mainly to avoid things such as LOG(0)
  call full_v_array_wire_nonperiodic(x-0.5*h,rho_in,V,n)

  ! One computes the derivative of V at grid points using central finite difference
  do i=2,n
     dV(i) = 1./h * (V(i)-V(i-1))
  end do
  Vleft = 0.
  call v_array_wire_nonperiodic(-0.5*h,x,rho_in,Vleft,n)
  ! dV(1) = dV(2)
  dV(1)= 1./h * (V(1)-Vleft)

  aldV = alpha*dV

  ! now one computes the induced charge density at each point in e/ang

  do i=2,n-1
     old_rho(i) = 1./(2*h) * (aldV(i+1)-aldV(i-1))
  end do

  old_rho(1) = 1./h * (aldV(2)-aldV(1))
  old_rho(n) = 1./h * (aldV(n)-aldV(n-1))

  ! change it to e/cm
  old_rho = old_rho*h ! e/ang
  old_rho = old_rho*1.E8 ! e/cm

  ! old_rho(1)=old_rho(2)
  ! old_rho(n)=old_rho(n-1)

  ! time for recursive algorithm
  counter = 0
  subcounter = 0
  beta =0.5
  diff = 10000.
  conv = .TRUE.
  max_ind =  MAXLOC(old_rho)

  do while (diff > 1.E-4)
     counter = counter +1
     subcounter = subcounter +1
     diff_old = diff

     !print *,counter, diff

     if (counter > max_iteration) then
        print *, "WARNING: recursive Poisson algorithm does not converge fast enough"
        conv = .FALSE. 
        EXIT
     end if

     ! update the potential including polarization charges
     call full_v_array_wire_nonperiodic(x-0.5*h,rho_in+old_rho,V,n)

     ! doing the same operations as before
     do i=2,n
        dV(i) = 1./h * (V(i)-V(i-1))
     end do
     Vleft=0.
     call v_array_wire_nonperiodic(-0.5*h,x,rho_in+old_rho,Vleft,n)
     dV(1) = 1./h * (V(1)-Vleft)
     ! dV(1) =dV(2)

     aldV = alpha*dV

     do i=2,n-1
        new_rho(i) = 1./(2*h) * (aldV(i+1)-aldV(i-1))
     end do

     new_rho(1) = 1./h * (aldV(2)-aldV(1))
     new_rho(n) = 1./h * (aldV(n)-aldV(n-1))
     
     ! need rho in e/cm
     new_rho = new_rho*h ! e/ang
     new_rho = new_rho*1.E8 ! e/cm

     !new_rho(1)=new_rho(2)
     !new_rho(n)=new_rho(n-1)

     indicator(MOD(counter, 2)+1) = old_rho(max_ind(1))-new_rho(max_ind(1))

     if (indicator(1)*indicator(2) < 0) then !there is an oscillation in the convergence
        beta = beta/1.5
        new_rho = 0.5*(new_rho+old_rho)
        subcounter = 0
     end if

     if (subcounter > 3) then
        beta = beta*2
        subcounter = 0
        if (beta>1.) then
           beta = 1.
        end if
     end if

     diff = ABS(MAXVAL(old_rho-new_rho))/ABS(MAXVAL(rho_in))

     old_rho = beta*new_rho + (1.-beta)*old_rho

  end do

  ! need to return a value of the potential defined on grid points, use linear interpolation
  do i=2,n
     potential(i) = 0.5*(V(i)+V(i-1))
  end do

  call v_array_wire_nonperiodic(-0.5*h,x,rho_in+old_rho,potential(1),n)
  potential(1) = 0.5*(potential(1)+V(1))
  rho_out = old_rho + rho_in ! total charge density

  !convergence check
  call full_v_array_wire_nonperiodic(x-0.5*h,rho_in+old_rho,V,n)

  !doing the same operations as before
  do i=2,n
     dV(i) = 1./h * (V(i)-V(i-1))
  end do
  dV(1) =dV(2)

  aldV = alpha*dV

  do i=2,n-1
     new_rho(i) = 1./(2*h) * (aldV(i+1)-aldV(i-1))
  end do

  !need rho in e/cm
  new_rho = new_rho*h !e/ang
  new_rho = new_rho*1.E8 ! e/cm

  new_rho(1)=new_rho(2)
  new_rho(n)=new_rho(n-1)

  if (conv .eqv. .TRUE.) then
     !print *, "Recursive Poisson algorithm converged in", counter, "steps."
     !print *, "Convergence check:", ABS(MAXVAL(old_rho-new_rho))/ABS(MAXVAL(rho_in))
  else 
     print *, "Recursive Poisson algorithm could not converge after", max_iteration, "steps."
  end if

end subroutine nonperiodic_recursive_poisson

!==================================================================
!====================PERIODIC FUNCTIONS============================

subroutine v_wire_periodic(x,lambda,potential,period)
  !! This subroutine computes the unscreend potential due to a periodic array 
  !! of charged wires at a distance x from one of them

  double precision, intent(in) :: x 
  !! x is the distance in ang between the wire and the point of interest
  double precision, intent(in) :: lambda 
  !! lambda is the line charged density of the wire in e/cm
  double precision, intent(out) :: potential 
  !! in V
  double precision, intent(in) :: period 
  !! periodicity of setup in ang

  double precision :: lamb 
  !! lambda in units of e/ang

  double precision :: PI, EPSILON0
  parameter(PI = 3.1415926535897932d0)
  parameter(EPSILON0 = 0.0055263496) ! in e/(V*ang)

  ! first off, need lambda in e/ang
  lamb = lambda*1.E-8

  potential = -lamb/(2.*PI*EPSILON0) * LOG(SIN(PI/period*x))
end subroutine v_wire_periodic


subroutine v_array_wire_periodic(coord,x,rho_in,potential,n)
  !! this subroutine computes the potential at a point from a periodic 
  !! charge density 

  double precision, intent(in) :: coord 
  !! coordinate of point of interest along x axis (in ang)
  double precision, intent(in), dimension(n) :: x 
  !! the array of grid points in ang
  double precision, intent(in), dimension(n) :: rho_in 
  !! the charge density in e/cm defined at each grid point
  double precision, intent(out) :: potential 
  !! the potential in V at point coord due to array of wires
  integer, intent(in) :: n 
  !! dimension of all arrays

  double precision :: potr 
  !! temporary storing of potential on the right (for integration)
  double precision :: potl 
  !! temporary storing of potential on the left (for integration)
  double precision :: period 
  !! the period along the x axis in ang

  integer :: i

  period = x(n)-x(1)

  ! numerical integration to get potential
  do i=1,n-1 ! up to n-1 cuz one uses trapezoidal method for integration
     call v_wire_periodic(ABS(coord-x(i)),rho_in(i),potl,period)
     call v_wire_periodic(ABS(coord-x(i+1)),rho_in(i+1),potr,period)
     potential = potential + 0.5*(potr+potl) !*(x(2)-x(1))
  end do
end subroutine v_array_wire_periodic


subroutine full_v_array_wire_periodic(x,rho_in,potential,n)
  !! this subroutine computes the potential at each grid point from a 
  !! charge density (array of wires with diffrent charges) 

  double precision, intent(in), dimension(n) :: x 
  !! the array of grid points in ang
  double precision, intent(in), dimension(n) :: rho_in 
  !! the charge density in e/cm defined at each grid point
  double precision, intent(out), dimension(n) :: potential 
  !! the potential in V at each grid point due to array of wires
  integer, intent(in) :: n 
  !! dimension of all arrays

  double precision :: coord
  double precision :: pot ! for temporary storage of potential
  integer :: i

  do i=1,n

     pot = 0.0 ! because of the += in the v_array_wire_periodic subroutine
     coord = (i-1)*(x(2)-x(1))
     call v_array_wire_periodic(coord,x,rho_in,pot,n)
     potential(i) = pot

  end do

end subroutine full_v_array_wire_periodic

subroutine periodic_recursive_poisson(x,rho_in,alpha,max_iteration,potential,rho_out,n)
  !! this subroutine solves the screened Poisson equation for 2D materials 
  !! with diffrent polarizabilities

  double precision, intent(in), dimension(n) :: x 
  !! the array containing the position of all grid points in ang
  double precision, intent(in), dimension(n) :: rho_in 
  !! initial charge density in e/cm defined at each grid point
  double precision, intent(in), dimension(n) :: alpha 
  !! the value of the polarizability at each grid point in e/V
  integer, intent(in) :: max_iteration 
  !! the maximum number of allowed iterations for the recursive Poisson algorithm
  double precision, intent(out), dimension(n) :: potential 
  !! the solution of recursive Poisson in V defined at each grid point
  double precision, intent(out), dimension(n) :: rho_out 
  !! The total charge density including polarization charges in e/cm at each grid point
  integer, intent(in) :: n 
  !! dimension of all grid-related arrays

  integer :: i
  integer :: counter, subcounter
  integer, dimension(1) :: max_ind
  double precision :: beta
  double precision, dimension(2) :: indicator ! used to check if there are oscillations whilec converging
  double precision :: h ! step size in ang
  double precision :: diff,diff_old ! allowed difference for convergence tests
  double precision, dimension(n) :: old_rho ! induced charge density before algorithm step
  double precision, dimension(n) :: new_rho 
  double precision, dimension(n) :: dV ! the space derivative of the potential at each point
  double precision, dimension(n) :: aldV ! product of alpha and dV at each grid point
  double precision, dimension(n) :: V !temporar storage of the potential
  logical :: conv

  h = x(2)-x(1)

  !First off, one computes the unscreened potential due to rho_in
  !One computes it a half grid point on the right, mainly to avoid things such as LOG(0)
  call full_v_array_wire_periodic(x-0.5*h,rho_in,V,n)
  V(n)=V(1) !need that because V(n) is not defined (there is a log(0))

  ! One computes the derivative of V at grid points using central finite difference with periodic BCs
  do i=2,n-1
     dV(i) = 1./h * (V(i)-V(i-1))
  end do
  dV(1) = 1./h * (V(1)-V(n-1)) ! because V(n) is out of bounds
  dV(n) = dV(1)

  aldV = alpha*dV

  ! now one computes the induced charge density at each point in e/ang, using periodic BCs

  do i=2,n-1
     old_rho(i) = 1./(2*h) * (aldV(i+1)-aldV(i-1))
  end do

  old_rho(1)=1./(2*h) * (aldV(2)-aldV(n-1))
  old_rho(n)=old_rho(1)

  ! change it to e/cm
  old_rho = old_rho*h ! e/ang
  old_rho = old_rho*1.E8 ! e/cm

  ! we start now the main iterative algorithm
  counter = 0
  subcounter = 0
  beta =0.5
  diff = 10000.
  conv = .TRUE.
  max_ind = MAXLOC(old_rho)

  do while (diff > 1.E-4)
     counter = counter +1
     subcounter = subcounter +1
     diff_old = diff

     !print *,counter, diff

     if (counter > max_iteration) then
        print *, "WARNING: recursive Poisson algorithm does not converge fast enough"
        conv = .FALSE.
        EXIT
     end if

     ! update the potential including polarization charges
     call full_v_array_wire_periodic(x-0.5*h,rho_in+old_rho,V,n)
     V(n) = V(1)

     !doing the same operations as before
     do i=2,n-1
        dV(i) = 1./h * (V(i)-V(i-1))
     end do
     dV(1) = 1./h * (V(1)-V(n-1)) ! because V(n) is out of bonds
     dV(n) = dV(1)

     aldV = alpha*dV

     do i=2,n-1
        new_rho(i) = 1./(2*h) * (aldV(i+1)-aldV(i-1))
     end do

     new_rho(1)=1./(2*h) * (aldV(2)-aldV(n-1))
     new_rho(n)=new_rho(1)

     ! need rho in e/cm
     new_rho = new_rho*h ! e/ang
     new_rho = new_rho*1.E8 ! e/cm

     indicator(MOD(counter,2)+1) = old_rho(max_ind(1))-new_rho(max_ind(1))

     if (indicator(1)*indicator(2) < 0) then ! there is an oscillation in the convergence
        beta = beta/1.5
        new_rho = 0.5*(new_rho+old_rho)
        subcounter = 0
     end if

     if (subcounter > 3) then
        beta = beta*2
        subcounter = 0
        if (beta>1.) then
           beta = 1.
        end if
     end if

     diff = ABS(MAXVAL(old_rho-new_rho))/ABS(MAXVAL(rho_in))

     old_rho = beta*new_rho + (1.-beta)*old_rho

  end do

  ! need to return a value of the potential defined on grid points, use linear interpolation
  do i=2,n
     potential(i) = 0.5*(V(i)+V(i-1))
  end do

  potential(1) = potential(n) ! periodic BCs
  rho_out = old_rho+rho_in ! total charge density

  ! convergence check
  call full_v_array_wire_periodic(x-0.5*h,rho_in+old_rho,V,n)
  V(n) = V(1)  
  !doing the same operations as before
  do i=2,n
     dV(i) = 1./h * (V(i)-V(i-1))
  end do
  dV(1) = 1./h * (V(1)-V(n-1)) ! because V(n) is out of bonds
  dV(n) = dV(1)

  aldV = alpha*dV

  do i=2,n-1
     new_rho(i) = 1./(2*h) * (aldV(i+1)-aldV(i-1))
  end do

  new_rho(1)=1./(2*h) * (aldV(2)-aldV(n-1))
  new_rho(n)=new_rho(1)

  ! need rho in e/cm
  new_rho = new_rho*h ! e/ang
  new_rho = new_rho*1.E8 ! e/cm

  if (conv .eqv. .TRUE.) then
     !print *, "Recursive Poisson algorithm converged in", counter, "steps."
     !print *, "Convergence check:", ABS(MAXVAL(old_rho-new_rho))/ABS(MAXVAL(rho_in))
  else 
     print *, "Recursive Poisson algorithm could not converge after", max_iteration, "steps."
  end if

end subroutine periodic_recursive_poisson

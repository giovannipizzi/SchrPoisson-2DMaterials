schrpoisson_wire2.so: schrpoisson_wire2.f90 schrpoisson_wire2.pyf
	f2py -c schrpoisson_wire2.pyf $<

clean:
	rm -f schrpoisson_wire2.so *~

#f2py -m schrpoisson_wire2 -h schrpoisson_wire2.pyf schrpoisson_wire2.f90
#this commend has to be used to create the *:pyf file

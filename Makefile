schrpoisson_wire.so: schrpoisson_wire.f90 schrpoisson_wire.pyf
	f2py -c schrpoisson_wire.pyf $<

#I used
# f2py -m schrpoisson_wire -h schrpoisson_wire.pyf schrpoisson_wire.f90
#to create the dsum.pyf file, and then I modified it

clean:
	rm -f schrpoisson_wire.so *~

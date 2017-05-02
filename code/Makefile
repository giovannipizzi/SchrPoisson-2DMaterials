schrpoisson_wire.so: schrpoisson_wire.f90 schrpoisson_wire.pyf
	f2py -c schrpoisson_wire.pyf $<

clean:
	rm -f schrpoisson_wire.so *~

## If you modify the .f90 file significantly (meaning, e.g., that you add
## functions, change the functions signature, ...) then you need to recreate
## the .pyf file. You can do it with the following command:
#
#  f2py -m schrpoisson_wire -h schrpoisson_wire.pyf schrpoisson_wire.f90

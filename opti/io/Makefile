plot3d : libplot3d.so

tecplot : libtecplot.so

# To build "tecplot" you'll need the TecIO library, available from:
#
# http://www.tecplot.com/downloads/tecio-library/
#
# make sure to prepend its name with "lib", e.g. libtecio.a or
# libtecio64.a depending on your system

FC = '/usr/bin/gfortran'

libplot3d.so : plot3d.f90
	f2py -c -m libplot3d --f90exec=$(FC) plot3d.f90

libtecplot.so : tecplot.f90
	f2py -c -m libtecplot --f90exec=$(FC) -L. -ltecio tecplot.f90

clean:
	$(RM) libplot3d.so
	$(RM) libtecplot.so

vim:tw=70:ts=4:ft=make:

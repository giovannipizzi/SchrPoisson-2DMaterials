###########################################
Schrödinger-Poisson solver for 2D materials
###########################################

.. contents::

.. section-numbering::

============
Introduction
============


===========
How to cite
===========
If you use this code in your work, please cite the following paper:

- \A. Bussy, G. Pizzi, M. Gibertini, *Strain-induced polar discontinuities in 2D materials from combined first-principles and Schrödinger-Poisson simulations*, to be submitted.

==============
How to compile
==============
You first need to have/install some dependencies:

- ``python``
- the python ``numpy`` package (this is also needed to have the ``f2py`` compiler)
- the python ``matplotlib`` package (if you want to interactively show the plots)

You can then compile the Fortran part of the code by simply typing ``make``
inside the ``code`` folder.

==========
How to run
==========
Go into the ``code`` folder and run::

  python 2Dschrpoisson.py {material_properties}.json {calc_input}.json

where you need to replace the two command-line options with two valid
input files, the first for the materials properties of the system, and the
second with the code input flags.

==============================
Description of the input files
==============================
The two input files must be in JSON format and contain all the input
needed to specify the system and ask what to compute.

A detailed description can be found in the `PDF documentation`_ in
the ``docs/compiled_docs subfolder``.

.. _PDF documentation: https://github.com/giovannipizzi/schrpoisson_2dmaterials/raw/master/docs/compiled_output/schrpoisson_2dmaterials_docs.pdf


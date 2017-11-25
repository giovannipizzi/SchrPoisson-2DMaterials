###########################################
Schrödinger-Poisson solver for 2D materials
###########################################

.. contents::

.. section-numbering::

============
Introduction
============
This code aims at self-consistently solving the coupled Schrödinger-Poisson equations in 2D
materials. More precisely, in the current implementation it is designed to simulate nanosheets of
a single material with regions of different strains. The Schrödinger equation is in 1D, meaning
that the strain can vary on a single axis *x*. Moreover, the material is assumed to be infinite 
in the *y* direction and periodic in the *x* direction.

===========
How to cite
===========
If you use this code in your work, please cite the following paper:

- \A. Bussy, G. Pizzi, M. Gibertini, *Strain-induced polar discontinuities in 2D materials from combined first-principles and Schrödinger-Poisson simulations*, **Phys. Rev. B 96**, 165438 (2017). [`DOI`_] [`arXiv`_]

============
Try it live!
============
`Run the simulation live in your browser`_ (no login required!)

.. image:: https://mybinder.org/badge.svg 
   :target: https://mybinder.org/v2/gh/giovannipizzi/schrpoisson_2dmaterials/master?urlpath=%2Fapps%2F2D-Schroedinger-Poisson-solver.ipynb


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

  python schrpoisson2D.py {material_properties}.json {calc_input}.json

where you need to replace the two command-line options with two valid
input files, the first for the materials properties of the system, and the
second with the code input flags.

You will find some example input files in the ``input_examples`` subfolder.

=========================================
Description of the input and output files
=========================================
The two input files must be in JSON format and contain all the input
needed to specify the system properties and the input flags.

A detailed description of the input flags can be found in the `PDF documentation`_ in
the ``docs/compiled_output`` subfolder, as well as a longer documentation and the 
description of the output files produced by the code.

============
Code license
============
The code is released open-source under a MIT license (see `LICENSE.txt`_ file).


.. _PDF documentation: https://github.com/giovannipizzi/schrpoisson_2dmaterials/raw/master/docs/compiled_output/schrpoisson_2dmaterials_docs.pdf

.. _DOI: http://doi.org/10.1103/PhysRevB.96.165438

.. _arXiv: http://arxiv.org/abs/1705.01303

.. _LICENSE.txt: https://github.com/giovannipizzi/schrpoisson_2dmaterials/raw/master/LICENSE.txt

.. _Run the simulation live in your browser: https://mybinder.org/v2/gh/giovannipizzi/schrpoisson_2dmaterials/master?urlpath=%2Fapps%2F2D-Schroedinger-Poisson-solver.ipynb


======
thermo
======

.. image:: http://img.shields.io/pypi/v/thermo.svg?style=flat
   :target: https://pypi.python.org/pypi/thermo
   :alt: Version_status
.. image:: http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://thermo.readthedocs.io/en/latest/
   :alt: Documentation
.. image:: http://img.shields.io/travis/CalebBell/thermo/master.svg?style=flat
   :target: https://travis-ci.org/CalebBell/thermo
   :alt: Build_status
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
   :target: https://github.com/CalebBell/thermo/blob/master/LICENSE.txt
   :alt: license
.. image:: https://img.shields.io/coveralls/CalebBell/thermo.svg
   :target: https://coveralls.io/github/CalebBell/thermo
   :alt: Coverage
.. image:: https://img.shields.io/pypi/pyversions/thermo.svg
   :target: https://pypi.python.org/pypi/thermo
   :alt: Supported_versions
.. image:: https://badges.gitter.im/CalebBell/thermo.svg
   :alt: Join the chat at https://gitter.im/CalebBell/thermo
   :target: https://gitter.im/CalebBell/thermo
.. image:: http://img.shields.io/appveyor/ci/calebbell/thermo.svg
   :target: https://ci.appveyor.com/project/calebbell/thermo/branch/master
   :alt: Build_status
.. image:: https://zenodo.org/badge/62404647.svg
   :alt: Zendo
   :target: https://zenodo.org/badge/latestdoi/62404647


.. contents::

What is thermo?
---------------

thermo is open-source software for engineers, scientists, technicians and
anyone trying to understand the universe in more detail. It facilitates 
the retrieval of constants of chemicals, the calculation of temperature
and pressure dependent chemical properties (both thermodynamic and 
transport), the calculation of the same for chemical mixtures (including
phase equilibria), and assorted information of a regulatory or legal 
nature about chemicals.

The thermo library depends on the SciPy library to povide numerical constants,
interpolation, integration, differentiation, and numerical solving functionality.
thermo all operating systems which support Python, is quick to install, and is 
free of charge. thermo is designed to be easy to use while still providing powerful
functionality. If you need to know something about a chemical, give thermo a try.

Installation
------------

Get the latest version of thermo from
https://pypi.python.org/pypi/thermo/

If you have an installation of Python with pip, simple install it with:

    $ pip install thermo
    
Alternatively, if you are using `conda <https://conda.io/en/latest/>`_ as your package management, you can simply
install thermo in your environment from `conda-forge <https://conda-forge.org/>`_ channel with:

    $ conda install -c conda-forge thermo

To get the git version, run:

    $ git clone git://github.com/CalebBell/thermo.git

Documentation
-------------

thermo's documentation is available on the web:

    http://thermo.readthedocs.io/

Getting Started
---------------

The library is designed around base SI units only for development
convenience. All chemicals default to 298.15 K and 101325 Pa on 
creation, unless specified. All constant-properties are loaded on
the creation of a Chemical instance.

.. code-block:: python

    >>> from thermo.chemical import Chemical
    >>> tol = Chemical('toluene')
    >>> tol.Tm, tol.Tb, tol.Tc
    (179.2, 383.75, 591.75)
    >>> tol.rho, tol.Cp, tol.k, tol.mu
    (862.2380125827527, 1706.0746129119084, 0.13034801424538045, 0.0005521951637285534)


For pure species, the phase is easily
identified, allowing for properties to be obtained without needing
to specify the phase. However, the properties are also available in the
hypothetical gas phase (when under the boiling point) and in the hypothetical
liquid phase (when above the boiling point) as these properties are needed
to evaluate mixture properties. Specify the phase of a property to be retrieved 
by appending 'l' or 'g' or 's' to the property.

.. code-block:: python

    >>> tol.rhog, tol.Cpg, tol.kg, tol.mug
    (4.032009635018902, 1126.5533755283168, 0.010736843919054837, 6.973325939594919e-06)

Creating a chemical object involves identifying the appropriate chemical by name
through a database, and retrieving all constant and temperature and pressure dependent
coefficients from Pandas DataFrames - a ~1 ms process. To obtain properties at different
conditions quickly, the method calculate has been implemented. 
    
.. code-block:: python

    >>> tol.calculate(T=310, P=101325)
    >>> tol.rho, tol.Cp, tol.k, tol.mu
    (851.1582219886011, 1743.280497511088, 0.12705495902514785, 0.00048161578053599225)
    >>> tol.calculate(310, 2E6)
    >>> tol.rho, tol.Cp, tol.k, tol.mu
    (852.7643604407997, 1743.280497511088, 0.12773606382684732, 0.0004894942399156052)

Each property is implemented through an independent object-oriented method, based on 
the classes TDependentProperty and TPDependentProperty to allow for shared methods of
plotting, integrating, differentiating, solving, interpolating, sanity checking, and
error handling. For example, to solve for the temperature at which the vapor pressure
of toluene is 2 bar. For each property, as many methods of calculating or estimating
it are included as possible. All methods can be visualized independently:

.. code-block:: python

    >>> Chemical('toluene').VaporPressure.solve_prop(2E5)
    409.5909115602903
    >>> Chemical('toluene').SurfaceTension.plot_T_dependent_property()

Mixtures are supported and many mixing rules have been implemented. However, there is
no error handling. Inputs as mole fractions (`zs`), mass fractions (`ws`), or volume
fractions (`Vfls` or `Vfgs`) are supported. Some shortcuts are supported to predefined
mixtures.

.. code-block:: python

    >>> from thermo.chemical import Mixture
    >>> vodka = Mixture(['water', 'ethanol'], Vfls=[.6, .4], T=300, P=1E5)
    >>> vodka.Prl,vodka.Prg
    (35.13075699606542, 0.9822705235442692)
    >>> air = Mixture('air', T=400, P=1e5)
    >>> air.Cp
    1013.7956176577836

Roadmap
-------

The author's main development item is phase equilibrium, a particularly
tricky area.

Latest source code
------------------

The latest development version of thermo's sources can be obtained at

    https://github.com/CalebBell/thermo


Bug reports
-----------

To report bugs, please use the thermo's Bug Tracker at:

    https://github.com/CalebBell/thermo/issues


License information
-------------------

See ``LICENSE.txt`` for information on the terms & conditions for usage
of this software, and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the thermo license, if it is convenient for you,
please cite thermo if used in your work. Please also consider contributing
any changes you make back, and benefit the community.


Citation
--------

To cite thermo in publications use::

    Caleb Bell and Contributors (2016-2020). thermo: Chemical properties component of Chemical Engineering Design Library (ChEDL)
    https://github.com/CalebBell/thermo.

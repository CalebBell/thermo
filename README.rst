======
Thermo
======

.. image:: http://img.shields.io/pypi/v/thermo.svg?style=flat
   :target: https://pypi.python.org/pypi/thermo
   :alt: Version_status
.. image:: http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://thermo.readthedocs.io/
   :alt: Documentation
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
.. image:: https://zenodo.org/badge/62404647.svg
   :alt: Zendo
   :target: https://zenodo.org/badge/latestdoi/62404647


.. contents::

What is Thermo?
---------------

Thermo is open-source software for engineers, scientists, technicians and
anyone trying to understand the universe in more detail. It facilitates 
the retrieval of constants of chemicals, the calculation of temperature
and pressure dependent chemical properties (both thermodynamic and 
transport), and the calculation of the same for chemical mixtures (including
phase equilibria) using various models.

Thermo runs on all operating systems which support Python, is quick to install, and is
free of charge. Thermo is designed to be easy to use while still providing powerful
functionality. If you need to know something about a chemical or mixture, give Thermo a try.

Installation
------------

Get the latest version of Thermo from
https://pypi.python.org/pypi/thermo/

If you have an installation of Python with pip, simple install it with:

    $ pip install thermo
    
Alternatively, if you are using `conda <https://conda.io/en/latest/>`_ as your package management, you can simply
install Thermo in your environment from `conda-forge <https://conda-forge.org/>`_ channel with:

    $ conda install -c conda-forge thermo

To get the git version, run:

    $ git clone git://github.com/CalebBell/thermo.git

Documentation
-------------

Thermo's documentation is available on the web:

    http://thermo.readthedocs.io/

Getting Started - Rigorous Interface
------------------------------------

Create a pure-component flash object for the compound "decane", using the Peng-Robinson equation of state. Perform a flash calculation at 300 K and 1 bar, and obtain a variety of properties from the resulting object:


.. code-block:: python

    >>> from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas, FlashPureVLS
    >>> # Load the constant properties and correlation properties
    >>> constants, correlations = ChemicalConstantsPackage.from_IDs(['decane'])
    >>> # Configure the liquid and gas phase objects
    >>> eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    >>> liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
    >>> gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
    >>> # Create a flash object with possible phases of 1 gas and 1 liquid
    >>> flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])
    >>> # Flash at 300 K and 1 bar
    >>> res = flasher.flash(T=300, P=1e5)
    >>> # molar enthalpy and entropy [J/mol and J/(mol*K) respectively] and the mass enthalpy and entropy [J/kg and J/(kg*K)]
    >>> res.H(), res.S(), res.H_mass(), res.S_mass()
    (-48458.137745529726, -112.67831317511894, -340578.897757812, -791.9383098029132)
    >>> # molar Cp and Cv [J/(mol*K)] and the mass Cp and Cv [J/(kg*K)]
    >>> res.Cp(), res.Cv(), res.Cp_mass(), res.Cv_mass()
    (295.17313861592686, 269.62465319082014, 2074.568831461133, 1895.0061117553582)
    >>> # Molar volume [m^3/mol], molar density [mol/m^3] and mass density [kg/m^3]
    >>> res.V(), res.rho(), res.rho_mass()
    (0.00020989856076374984, 4764.206082982839, 677.8592453530177)
    >>> # isobatic expansion coefficient [1/K], isothermal compressibility [1/Pa], Joule Thomson coefficient [K/Pa]
    >>> res.isobaric_expansion(), res.kappa(), res.Joule_Thomson()
    (0.0006977350520992281, 1.1999043797490713e-09, -5.622547043844744e-07)
    >>> # Speed of sound in molar [m*kg^0.5/(s*mol^0.5)] and mass [m/s] units
    >>> res.speed_of_sound(), res.speed_of_sound_mass()
    (437.61281158744987, 1160.1537167375043)

The following example shows the retrieval of chemical properties for a two-phase system with methane, ethane, and nitrogen, using a few sample kijs:

.. code-block:: python

    >>> from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, PRMIX, FlashVL
    >>> from thermo.interaction_parameters import IPDB
    >>> constants, properties = ChemicalConstantsPackage.from_IDs(['methane', 'ethane', 'nitrogen'])
    >>> kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')
    >>> kijs
    [[0.0, -0.0059, 0.0289], [-0.0059, 0.0, 0.0533], [0.0289, 0.0533, 0.0]]
    >>> eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    >>> gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    >>> liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    >>> flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
    >>> zs = [0.965, 0.018, 0.017]
    >>> PT = flasher.flash(T=110.0, P=1e5, zs=zs)
    >>> PT.VF, PT.gas.zs, PT.liquid0.zs
    (0.10365, [0.881788, 2.6758e-05, 0.11818], [0.97462, 0.02007, 0.005298])
    >>> flasher.flash(P=1e5, VF=1, zs=zs).T
    133.6
    >>> flasher.flash(T=133, VF=0, zs=zs).P
    518367.4
    >>> flasher.flash(P=PT.P, H=PT.H(), zs=zs).T
    110.0
    >>> flasher.flash(P=PT.P, S=PT.S(), zs=zs).T
    110.0
    >>> flasher.flash(T=PT.T, H=PT.H(), zs=zs).T
    110.0
    >>> flasher.flash(T=PT.T, S=PT.S(), zs=zs).T
    110.0

There is also a N-phase flash algorithm available, FlashVLN. There are no solid models implemented in this interface at this time.


Getting Started - Simple Interface
----------------------------------

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
    (862.238, 1706.07, 0.13034, 0.0005522)


For pure species, the phase is easily
identified, allowing for properties to be obtained without needing
to specify the phase. However, the properties are also available in the
hypothetical gas phase (when under the boiling point) and in the hypothetical
liquid phase (when above the boiling point) as these properties are needed
to evaluate mixture properties. Specify the phase of a property to be retrieved 
by appending 'l' or 'g' or 's' to the property.

.. code-block:: python

    >>> from thermo.chemical import Chemical
    >>> tol = Chemical('toluene')
    >>> tol.rhog, tol.Cpg, tol.kg, tol.mug
    (4.0320096, 1126.553, 0.010736, 6.97332e-06)

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

    >>> Chemical('toluene').VaporPressure.solve_property(2E5)
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

Warning: The phase equilibria of Chemical and Mixture are not presently
as rigorous as the other interface. The property model is not particularly
consistent and uses a variety of ideal and Peng-Robinson methods together.

Latest source code
------------------

The latest development version of Thermo's sources can be obtained at

    https://github.com/CalebBell/thermo


Bug reports
-----------

To report bugs, please use the Thermo's Bug Tracker at:

    https://github.com/CalebBell/thermo/issues


License information
-------------------

See ``LICENSE.txt`` for information on the terms & conditions for usage
of this software, and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the Thermo license, if it is convenient for you,
please cite Thermo if used in your work. Please also consider contributing
any changes you make back, and benefit the community.


Citation
--------

To cite Thermo in publications use::

    Caleb Bell and Contributors (2016-2021). Thermo: Chemical properties component of Chemical Engineering Design Library (ChEDL)
    https://github.com/CalebBell/thermo.

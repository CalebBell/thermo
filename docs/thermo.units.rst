Support for pint Quantities (thermo.units)
=============================================

Basic module which wraps some of thermo functions and classes to be compatible with the
`pint <https://github.com/hgrecco/pint>`_ unit handling library.
All other object - dicts, lists, etc - are not wrapped. 

>>> import thermo
>>> thermo.units.PRMIX # doctest: +ELLIPSIS
fluids.units.PRMIX

>>> kwargs = dict(T=400.0*u.degC, P=30*u.psi, Tcs=[126.1, 190.6]*u.K, Pcs=[33.94E5, 46.04E5]*u.Pa, omegas=[0.04, 0.011]*u.dimensionless, zs=[0.5, 0.5]*u.dimensionless, kijs=[[0.0, 0.0289], [0.0289, 0.0]]*u.dimensionless)
>>> thermo.units.PRMIX(**kwargs)

Note that values which can normally be numpy arrays or python lists, are required to always be numpy arrays in this interface.

This is interface is powerful but not complex enough to handle many of the objects in Thermo. A list of the types of classes which are not supported is as follows:

* TDependentProperty, TPDependentProperty, MixtureProperty
* Phase objects
* Flash object
* ChemicalConstantsPackage
* PropertyCorrelationsPackage

For further information on this interface, please see the documentation of `fluids.units <https://fluids.readthedocs.io/fluids.units.html>`_ which is built in the same way.
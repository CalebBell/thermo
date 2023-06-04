Introduction to ChemicalConstantsPackage and PropertyCorrelationsPackage
========================================================================

.. contents:: :local:

These two objects are designed to contain information needed by flash algorithms.
In the first iteration of thermo, data was automatically looked up in databases and there was no way to replace that data. Thermo now keeps data and algorithms completely separate. This has also been very helpful to make unit tests that do not change their results.

There are five places to configure the flash and phase infrastructure:

* Constant data about chemicals, like melting point or boiling point or UNIFAC groups. This information needs to be put into an immutable :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` object.
* Temperature-dependent data, like Antoine coefficients, Tait pressure-dependent volume parameters, or Laliberte electrolyte viscosity interaction parameters. These are stored in :obj:`TDependentProperty <thermo.utils.TDependentProperty>`, :obj:`TPDependentProperty <thermo.utils.TPDependentProperty>`, and :obj:`MixtureProperty <thermo.utils.MixtureProperty>` objects. More information about configuring those to provide the desired properties can be found in property objects tutorial; this tutorial assumes you have already configured them as desired. These many objects are added to an :obj:`PropertyCorrelationsPackage <thermo.chemical_package.PropertyCorrelationsPackage>` object before being provided to the flash algorithms.
* Phase-specific parameters that are not general and depend on a specific phase configuration for meaning; such as a volume translation coefficient or a binary interaction parameter. This information is provided when configuring each :obj:`Phase <thermo.phases.Phase>`.
* Information about bulk mixing rules or bulk property calculation methods; these don't have true thermodynamic definitions, and are configurable in the :obj:`BulkSettings <thermo.bulk.BulkSettings>` object.
* Settings of the :obj:`Flash <thermo.flash.Flash>` object; ideally no configuration would be required there. In some cases it might be useful to lower the tolerances or change an algorithm.

This tutorial covers the first two places, :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` and  :obj:`PropertyCorrelationsPackage <thermo.chemical_package.PropertyCorrelationsPackage>`.


ChemicalConstantsPackage Object
-------------------------------
Creating ChemicalConstantsPackage Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` can be created by specifying the known constant values of each chemical. All values are technically optional; the requirements of each :obj:`Flash <thermo.flash.Flash>` algorithm are different, but a minimum suggested amount is `names`, `CASs`, `MWs`, `Tcs`, `Pcs`, `omegas`, `Tbs`, and `atomss`. The list of all accepted properties can be found :obj:`here <thermo.chemical_package.ChemicalConstantsPackage.properties>`.

>>> from thermo import ChemicalConstantsPackage, PropertyCorrelationsPackage
>>> constants = ChemicalConstantsPackage(MWs=[18.01528, 106.165, 106.165, 106.165], names=['water', 'o-xylene', 'p-xylene', 'm-xylene'], omegas=[0.344, 0.3118, 0.324, 0.331], Pcs=[22048320.0, 3732000.0, 3511000.0, 3541000.0], Tcs=[647.14, 630.3, 616.2, 617.0])

Using ChemicalConstantsPackage Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once created, all properties, even missing ones, can be accessed as attributes using the same names as required by the constructor:

>>> constants.MWs
[18.01528, 106.165, 106.165, 106.165]
>>> constants.Vml_STPs
[None, None, None, None]

It is the intention for these :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` to be immutable. Python doesn't easily allow this to be enforced, but unexpected behavior will probably result if they are edited. If different properties are desired; create new :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` objects.

The :obj:`__repr__ <thermo.chemical_package.ChemicalConstantsPackage.__repr__>` of the :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` object returns a representation of the object that can be used to reconstruct it:

>>> constants
ChemicalConstantsPackage(MWs=[18.01528, 106.165, 106.165, 106.165], names=['water', 'o-xylene', 'p-xylene', 'm-xylene'], omegas=[0.344, 0.3118, 0.324, 0.331], Pcs=[22048320.0, 3732000.0, 3511000.0, 3541000.0], Tcs=[647.14, 630.3, 616.2, 617.0])
>>> hash(eval(constants.__repr__())) == hash(constants)
True

Creating Smaller ChemicalConstantsPackage Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to create a new, smaller :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` with fewer components by using the :obj:`subset <thermo.chemical_package.ChemicalConstantsPackage.subset>` method, which accepts either indexes or slices and returns a new object:

>>> constants.subset([0, 1])
ChemicalConstantsPackage(MWs=[18.01528, 106.165], names=['water', 'o-xylene'], omegas=[0.344, 0.3118], Pcs=[22048320.0, 3732000.0], Tcs=[647.14, 630.3])
>>> constants.subset(slice(1,3))
ChemicalConstantsPackage(MWs=[106.165, 106.165], names=['o-xylene', 'p-xylene'], omegas=[0.3118, 0.324], Pcs=[3732000.0, 3511000.0], Tcs=[630.3, 616.2])
>>> constants.subset([0])
ChemicalConstantsPackage(MWs=[18.01528], names=['water'], omegas=[0.344], Pcs=[22048320.0], Tcs=[647.14])

It is also possible to reduce the number of properties set with the `subset` methods:

>>> constants.subset([1, 3], properties=('names', 'MWs'))
ChemicalConstantsPackage(MWs=[106.165, 106.165], names=['o-xylene', 'm-xylene'])


Adding or Replacing Constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is possible to create a new :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` with added properties and/or replacing the old properties, from an existing object. This is helpful if better values for select properties are known. The :obj:`with_new_constants <thermo.chemical_package.ChemicalConstantsPackage.with_new_constants>` method does this.

>>> constants.with_new_constants(Tcs=[650.0, 630.0, 620.0, 620.0], Tms=[20.0, 100.0, 50.0, 12.3])
ChemicalConstantsPackage(MWs=[18.01528, 106.165, 106.165, 106.165], names=['water', 'o-xylene', 'p-xylene', 'm-xylene'], omegas=[0.344, 0.3118, 0.324, 0.331], Pcs=[22048320.0, 3732000.0, 3511000.0, 3541000.0], Tcs=[650.0, 630.0, 620.0, 620.0], Tms=[20.0, 100.0, 50.0, 12.3])

Creating ChemicalConstantsPackage Objects from chemicals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A convenience method exists to load these constants from a different data files exists. Some values for all properties are available; not all compounds have all properties.

>>> obj = ChemicalConstantsPackage.constants_from_IDs(['methanol', 'ethanol', 'isopropanol'])
>>> obj.Tbs
[337.632, 351.570441659, 355.36]

When working with a fixed set of components, it may be a good idea to take this generated package, select only those properties being used, convert it to a string, and then embed that new object in a program. This will remove the need to load various data files, and if `chemicals` updates data files, different results won't be obtained from your constants package.

>>> small_obj = obj.subset(properties=('names', 'CASs', 'MWs', 'Tcs', 'Pcs', 'omegas', 'Tbs', 'Tms', 'atomss'))
>>> small_obj
ChemicalConstantsPackage(atomss=[{'C': 1, 'H': 4, 'O': 1}, {'C': 2, 'H': 6, 'O': 1}, {'C': 3, 'H': 8, 'O': 1}], CASs=['67-56-1', '64-17-5', '67-63-0'], MWs=[32.04186, 46.06844, 60.09502], names=['methanol', 'ethanol', 'isopropanol'], omegas=[0.5625, 0.646, 0.665], Pcs=[8215850.0, 6268000.0, 4764000.0], Tbs=[337.632383296, 351.570441659, 355.36], Tcs=[513.38, 514.71, 508.3], Tms=[175.15, 159.05, 183.65])

Once the object is printed, the generated text can be copy/pasted as valid Python into a program:

>>> obj = ChemicalConstantsPackage(atomss=[{'C': 1, 'H': 4, 'O': 1}, {'C': 2, 'H': 6, 'O': 1}, {'C': 3, 'H': 8, 'O': 1}], CASs=['67-56-1', '64-17-5', '67-63-0'], MWs=[32.04186, 46.06844, 60.09502], names=['methanol', 'ethanol', 'isopropanol'], omegas=[0.5589999999999999, 0.635, 0.665], Pcs=[8084000.0, 6137000.0, 4764000.0], Tbs=[337.65, 351.39, 355.36], Tcs=[512.5, 514.0, 508.3], Tms=[175.15, 159.05, 183.65])


.. warning::
    `chemicals <https://github.com/CalebBell/chemicals>`_ is a
    project with a focus on collecting data and
    correlations from various sources. In no way is it a project to
    critically evaluate these and provide recommendations. You are
    strongly encouraged to check values from it and modify them
    if you want different values. If you believe there is a value
    which has a typographical error please report it to the
    `chemicals <https://github.com/CalebBell/chemicals>`_
    project. If data is missing or not as accuracte
    as you would like, and you know of a better method or source,
    new methods and sources can be added to
    `chemicals <https://github.com/CalebBell/chemicals>`_
    fairly easily once the data entry is complete.    

Storing and Loading ChemicalConstantsPackage Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For larger applications with many components, it is not as feasible to convert the :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` to a string and embed it in a program. For that application, the object can be converted back and forth from JSON:

>>> obj = ChemicalConstantsPackage(MWs=[106.165, 106.165], names=['o-xylene', 'm-xylene'])
>>> constants = ChemicalConstantsPackage(MWs=[18.01528, 106.165], names=['water', 'm-xylene'])
>>> string = constants.as_json()
>>> new_constants = ChemicalConstantsPackage.from_json(string)
>>> hash(new_constants) == hash(constants)
True

PropertyCorrelationsPackage
---------------------------
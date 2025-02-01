Introduction to Phase and Flash Calculations
============================================

.. contents:: :local:

The framework for performing phase and flash calculations is designed around the following principles:

* Immutability
* Calculations are completely independent from any databases or lookups - every input must be provided as input
* Default to general-purpose algorithms that make no assumptions about specific systems
* Inclusion of separate flashes algorithms wherever faster algorithms can be used for specific cases
* Allow options to restart a flash from a nearby previously calculated result, with options to skip checking the result for stability
* Use very tight tolerances on all calculations
* Expose all constants used by algorithms

Phase Objects
-------------
A phase is designed to have a single state at any time, and contain all the information needed to compute phase-specific properties.
Phases should always be initialized at a specific molar composition `zs`, `T` and `P`; and new phase objects at different conditions should be created from the existing ones with the :obj:`Phase.to <thermo.phases.Phase.to>` method (a little faster than creating them from scratch). That method also allows the new state to be set from any two of `T`, `P`, or `V`. When working in the `T` and `P` domain only, the :obj:`Phase.to_TP_zs <thermo.phases.Phase.to_TP_zs>` method is a little faster.

Phases are designed to be able to calculate every thermodynamic property. `T` and `P` are always attributes of the phase, but all other properties are functions that need to be called. Some examples of these properties are :obj:`V <thermo.phases.Phase.V>`, :obj:`H <thermo.phases.Phase.H>`, :obj:`S <thermo.phases.Phase.S>`, :obj:`Cp <thermo.phases.Phase.Cp>`, :obj:`dP_dT <thermo.phases.Phase.dP_dT>`, :obj:`d2P_dV2 <thermo.phases.Phase.d2P_dV2>`, :obj:`fugacities <thermo.phases.Phase.fugacities>`, :obj:`lnphis <thermo.phases.Phase.lnphis>`, :obj:`dlnphis_dT <thermo.phases.Phase.dlnphis_dT>`, and :obj:`dlnphis_dP <thermo.phases.Phase.dlnphis_dP>`.

If a system is already known to be single-phase, the phase framework can be used directly without performing flash calculations. This may offer a speed boost in some applications.


Available Phases
^^^^^^^^^^^^^^^^
Although the underlying equations of state often don't distinguish between liquid or vapor phase, it was convenient to create separate phase objects designed to hold gas, liquid, and solid phases separately.

The following phases can represent both a liquid and a vapor state. Their class is not a true indication that their properties are liquid or gas.

* Cubic equations of state - :obj:`CEOSLiquid <thermo.phases.CEOSLiquid>` and :obj:`CEOSGas <thermo.phases.CEOSGas>`
* IAPWS-95 Water and Steam - :obj:`IAPWS95Liquid <thermo.phases.IAPWS95Liquid>` and :obj:`IAPWS95Gas <thermo.phases.IAPWS95Gas>`
* Wrapper objects for CoolProp's Helmholtz EOSs - :obj:`CoolPropLiquid <thermo.phases.CoolPropLiquid>` and :obj:`CoolPropGas <thermo.phases.CoolPropGas>`

The following phase objects can only represent a gas phase:

* Ideal-gas law - :obj:`IdealGas <thermo.phases.IdealGas>`
* High-accuracy properties of dry air - :obj:`DryAirLemmon <thermo.phases.DryAirLemmon>`

The following phase objects can only represent a liquid phase:

* Ideal-liquid and/or activity coefficient models - :obj:`GibbsExcessLiquid <thermo.phases.GibbsExcessLiquid>`

Serialization
^^^^^^^^^^^^^
All phase models offer a :obj:`as_json <thermo.phases.Phase.as_json>` method and a :obj:`from_json <thermo.phases.Phase.from_json>` to serialize the object state for transport over a network, storing to disk, and passing data between processes.

>>> import json
>>> from scipy.constants import R
>>> from thermo import HeatCapacityGas, IdealGas, Phase
>>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])), HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
>>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
>>> json_stuff = json.dumps(phase.as_json())
>>> new_phase = Phase.from_json(json.loads(json_stuff))
>>> assert new_phase == phase

Other json libraries can be used besides the standard json library by design.

Storing and recreating objects with Python's :py:func:`pickle.dumps` library is also tested; this can be faster than using JSON at the cost of being binary data.

Hashing
^^^^^^^
All models have a :obj:`__hash__ <thermo.phases.Phase.exact_hash>` method that can be used to compare different phases to see if they are absolutely identical (including which values have been calculated already).

They also have a :obj:`model_hash <thermo.phases.Phase.model_hash>` method that can be used to compare different phases to see if they have identical model parameters.

They also have a :obj:`state_hash <thermo.phases.Phase.state_hash>` method that can be used to compare different phases to see if they have identical temperature, composition, and model parameters. This is the __hash__ method.



Flashes with Pure Compounds
---------------------------
Pure components are really nice to work with because they have nice boundaries between each state, and the mole fraction is always 1; there is no composition dependence. There is a separate flash interfaces for pure components. These flashes are very mature and should be quite reliable.

Vapor-Liquid Cubic Equation Of State Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following example illustrates some of the types of flashes supported using the component methanol, the stated critical properties, a heat capacity correlation from Poling et. al., and the Peng-Robinson equation of state.

Obtain a heat capacity object, and select a source:

>>> from thermo.heat_capacity import POLING_POLY
>>> CpObj = HeatCapacityGas(CASRN='67-56-1')
>>> CpObj.method = POLING_POLY
>>> HeatCapacityGases = [CpObj]

Create a :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` object which holds constant properties of the object, using a minimum of values:

>>> from thermo import ChemicalConstantsPackage, PropertyCorrelationsPackage, PRMIX, SRKMIX, CEOSLiquid, CEOSGas, FlashPureVLS
>>> constants = ChemicalConstantsPackage(Tcs=[512.5], Pcs=[8084000.0], omegas=[0.559], MWs=[32.04186], CASs=['67-56-1'])

Create a :obj:`PropertyCorrelationsPackage <thermo.chemical_package.PropertyCorrelationsPackage>` object which holds temperature-dependent property objects, also setting `skip_missing` to True so no database lookups are performed:

>>> correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)

Create liquid and gas cubic phase objects using the :obj:`Peng-Robinson equation of state <thermo.eos_mix.PRMIX>`:

>>> eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
>>> liquid = CEOSLiquid(PRMIX, HeatCapacityGases=HeatCapacityGases, eos_kwargs=eos_kwargs)
>>> gas = CEOSGas(PRMIX, HeatCapacityGases=HeatCapacityGases, eos_kwargs=eos_kwargs)

Create the Flash object :obj:`FlashPureVLS <thermo.flash.FlashPureVLS>` for pure components:

>>> flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])

Do a T-P flash:

>>> res = flasher.flash(T=300, P=1e5)
>>> res.phase, res.liquid0
('L', CEOSLiquid(eos_class=PRMIX, eos_kwargs={"Tcs": [512.5], "Pcs": [8084000.0], "omegas": [0.559]}, HeatCapacityGases=[HeatCapacityGas(CASRN="67-56-1", extrapolation="linear", method="POLING_POLY", Tmin=50.0, Tmax=1000.0)], T=300.0, P=100000.0, zs=[1.0]))

Do a temperature and vapor-fraction flash:

>>> res = flasher.flash(T=300, VF=.3)

Do a pressure and vapor-fraction flash:

>>> res = flasher.flash(P=1e5, VF=.5)

Do a pressure and enthalpy flash:

>>> res = flasher.flash(P=1e5, H=100)

Do a pressure and entropy flash:

>>> res = flasher.flash(P=1e5, S=30)

Do a temperature and entropy flash:

>>> res = flasher.flash(T=400.0, S=30)

Do a temperature and enthalpy flash:

>>> res = flasher.flash(T=400.0, H=1000)

Do a volume and internal energy flash:

>>> res = flasher.flash(V=1e-4, U=1000)


As you can see, the interface is convenient and supports most types of flashes. In fact, the algorithms are generic; any of `H`, `S`, `U`, and can be combined with any combination of `T`, `P`, and `V`. Although most of the flashes shown above except TS and TH are usually well behaved, depending on the EOS combination there may be multiple solutions. No real guarantees can be made about which solution will be returned in those cases.

Flashes with two of  `H`, `S`, and `U` are not implemented at present.

It is not necessary to use the same phase model for liquid and gas phases; the below example shows a flash switching the gas phase model to SRK.

>>> SRK_gas = CEOSGas(SRKMIX, HeatCapacityGases=HeatCapacityGases, eos_kwargs=eos_kwargs)
>>> flasher_inconsistent = FlashPureVLS(constants, correlations, gas=SRK_gas, liquids=[liquid], solids=[])
>>> res = flasher_inconsistent.flash(T=400.0, VF=1)

Choosing to use an inconsistent model will slow down many calculations as more checks are required; and some flashes may have issues with discontinuities in some conditions, and simply a lack of solution in other conditions.


Vapor-Liquid Steam Example
^^^^^^^^^^^^^^^^^^^^^^^^^^
The IAPWS-95 standard is implemented and available for easy use:

>>> from thermo import FlashPureVLS, IAPWS95Liquid, IAPWS95Gas, iapws_constants, iapws_correlations
>>> liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
>>> gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
>>> flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])
>>> PT = flasher.flash(T=800.0, P=1e7)
>>> PT.rho_mass()
29.1071839176
>>> print(flasher.flash(T=600, VF=.5))
<EquilibriumState, T=600.0000, P=12344824.3572, zs=[1.0], betas=[0.5, 0.5], phases=[<IAPWS95Gas, T=600 K, P=1.23448e+07 Pa>, <IAPWS95Liquid, T=600 K, P=1.23448e+07 Pa>]>
>>> print(flasher.flash(T=600.0, H=50802))
<EquilibriumState, T=600.0000, P=10000469.1288, zs=[1.0], betas=[1.0], phases=[<IAPWS95Gas, T=600 K, P=1.00005e+07 Pa>]>
>>> print(flasher.flash(P=1e7, S=104.))
<EquilibriumState, T=599.6790, P=10000000.0000, zs=[1.0], betas=[1.0], phases=[<IAPWS95Gas, T=599.679 K, P=1e+07 Pa>]>
>>> print(flasher.flash(V=.00061, U=55850))
<EquilibriumState, T=800.5922, P=10144789.0899, zs=[1.0], betas=[1.0], phases=[<IAPWS95Gas, T=800.592 K, P=1.01448e+07 Pa>]>


Not all flash calculations have been fully optimized, but the basic flashes are quite fast.
Introduction to Phase and Flash Calculations
============================================

I have programmed interfaces for doing property calculations three times now, and have settled on the current interface which is designed around the following principles:

* Immutability
* Calculations are completely independent from any databases or lookups - every input must be provided as input
* Default to general-purpose algorithms that make no assumptions about specific systems
* Inclusion of separate flashes algorithms wherever faster algorithms can be used for specific cases
* Allow options to restart a flash from a nearby previously calculated result, with options to skip checking the result for stability
* Use very tight tolerances on all calculations
* Expose all constants used by algorithms

After a couple of iterations, the interface which is found to work best for 

>>> constants = ChemicalConstantsPackage(Tcs=[768.0], Pcs=[1070000.0], omegas=[0.8805], MWs=[282.54748], CASs=['112-95-8'])
>>> from thermo.phases import Phase

Phase Objects
-------------
A phase is designed to have a single state at any time, and contain all the information needed to compute phase-specific properties.
Phases should always be initialized at a specific molar composition `zs`, `T` and `P`; and new phase objects at different conditions should be created from the existing ones with the :obj:`Phase.to <thermo.phases.Phase.to>` method (a little faster than creating them from scratch). That method also allows the new state to be set from any two of `T`, `P`, or `V`. When working in the `T` and `P` domain only, the :obj:`Phase.to_TP_zs <thermo.phases.Phase.to_TP_zs>` method is a little faster.

Phases are designed to be able to calculate every thermodynamic property. `T` and `P` are always attributes of the phase, but all other properties are functions that need to be called. Some examples of these properties are :obj:`V <thermo.phases.Phase.V>`, :obj:`H <thermo.phases.Phase.H>`, :obj:`S <thermo.phases.Phase.S>`, :obj:`Cp <thermo.phases.Phase.Cp>`, :obj:`dP_dT <thermo.phases.Phase.dP_dT>`, :obj:`d2P_dV2 <thermo.phases.Phase.d2P_dV2>`, :obj:`fugacities <thermo.phases.Phase.fugacities>`, :obj:`lnphis <thermo.phases.Phase.lnphis>`, :obj:`dlnphis_dT <thermo.phases.Phase.dlnphis_dT>`, and :obj:`dlnphis_dP <thermo.phases.Phase.dlnphis_dP>`.



Flashes with Pure Compounds
---------------------------
Pure components are really nice to work with because they have nice boundaries between each state, and the mole fraction is always 1; there is no composition dependence. There is a separate flash interfaces for pure components. These flashes are very mature and should be quite reliable.

The following example illustrates some of the types of flashes supported using the component methanol, the stated critical properties, a heat capacity correlation from Poling et. al., and the Peng-Robinson equation of state.

Obtain a heat capacity object, and select a source:

>>> from thermo.heat_capacity import POLING
>>> CpObj = HeatCapacityGas(CASRN='67-56-1')
>>> CpObj.set_method(POLING)
>>> CpObj.POLING_coefs # Show the coefficients
[4.714, -0.006986, 4.211e-05, -4.443e-08, 1.535e-11]
>>> HeatCapacityGases = [CpObj]

Create a :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` object which holds constant properties of the object, using a minimum of values:

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
('L', <CEOSLiquid, T=300 K, P=100000 Pa>)

Do a temperature and vapor-fraction flash:

>>> flasher.flash(T=300, VF=.3)
<EquilibriumState, T=300.0000, P=17641.8497, zs=[1.0], betas=[0.3, 0.7], phases=[<CEOSGas, T=300 K, P=17641.8 Pa>, <CEOSLiquid, T=300 K, P=17641.8 Pa>]>

Do a pressure and vapor-fraction flash:

>>> flasher.flash(P=1e5, VF=.5)
<EquilibriumState, T=336.9998, P=100000.0000, zs=[1.0], betas=[0.5, 0.5], phases=[<CEOSGas, T=337 K, P=100000 Pa>, <CEOSLiquid, T=337 K, P=100000 Pa>]>

Do a pressure and enthalpy flash:

>>> flasher.flash(P=1e5, H=100)
<EquilibriumState, T=336.9998, P=100000.0000, zs=[1.0], betas=[0.95955195, 0.0404480443], phases=[<CEOSGas, T=337 K, P=100000 Pa>, <CEOSLiquid, T=337 K, P=100000 Pa>]>

Do a pressure and entropy flash:

>>> flasher.flash(P=1e5, S=30)
<EquilibriumState, T=530.7967, P=100000.0000, zs=[1.0], betas=[1.0], phases=[<CEOSGas, T=530.797 K, P=100000 Pa>]>

Do a temperature and entropy flash:

>>> flasher.flash(T=400.0, S=30)
<EquilibriumState, T=400.0000, P=14736.5078, zs=[1.0], betas=[1.0], phases=[<CEOSGas, T=400 K, P=14736.5 Pa>]>

Do a temperature and enthalpy flash:

>>> flasher.flash(T=400.0, H=1000)
<EquilibriumState, T=400.0000, P=801322.3731, zs=[1.0], betas=[0.90923194, 0.09076805], phases=[<CEOSGas, T=400 K, P=801322 Pa>, <CEOSLiquid, T=400 K, P=801322 Pa>]>

Do a volume and internal energy flash:

>>> flasher.flash(V=1e-4, U=1000)
<EquilibriumState, T=655.5447, P=47575958.4564, zs=[1.0], betas=[1.0], phases=[<CEOSLiquid, T=655.545 K, P=4.7576e+07 Pa>]>


As you can see, the interface is convenient and supports most types of flashes. In fact, the algorithms are generic; any of `H`, `S`, `G`, `U`, and `A` can be combined with any combination of `T`, `P`, and `V`. Although most of the flashes shown above except TS and TH are usually well behaved, depending on the EOS combination there may be multiple solutions. No real guarantees can be made about which solution will be returned in those cases.

Flashes with two of  `H`, `S`, `G`, `U`, and `A` are not supported.

It is not necessary to use the same phase model for liquid and gas phases; the below example shows a flash switching the gas phase model to SRK.

>>> SRK_gas = CEOSGas(SRKMIX, HeatCapacityGases=HeatCapacityGases, eos_kwargs=eos_kwargs)
>>> flasher_inconsistent = FlashPureVLS(constants, correlations, gas=SRK_gas, liquids=[liquid], solids=[])
>>> flasher_inconsistent.flash(T=400.0, VF=1)
<EquilibriumState, T=400.0000, P=797342.2263, zs=[1.0], betas=[1, 0.0], phases=[<CEOSGas, T=400 K, P=797342 Pa>, <CEOSLiquid, T=400 K, P=797342 Pa>]>

Choosing to use an inconsistent model will slow down many calculations as more checks are required; and some flashes may have issues with discontinuities in some conditions, and simply a lack of solution in other conditions.





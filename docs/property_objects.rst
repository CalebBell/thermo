Introduction to Property Objects
================================

For every chemical property, there are lots and lots of methods. The methods can be grouped by which phase they apply to, although some methods are valid for both liquids and gases. 

Properties calculations be separated into three categories:

* Properties of chemicals that depend on **temperature**, with weak dependence on pressure, like surface tension and thermal conductivity.
* Properties of chemicals that depend on **temperature** and **pressure** fundamentally, like gas volume.
* Properties of mixtures, that depend on **temperature** and **pressure**  and **composition**.



These properties are implemented in an object oriented way, with the actual functional algorithms themselves having been separated out into the `chemicals <https://github.com/CalebBell/chemicals>`_ library. The goal of these objects is to make it easy to experiment with different methods.

The base classes for the three respective types of properties are:

* :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
* :obj:`TPDependentProperty <thermo.utils.TPDependentProperty>`
* :obj:`MixtureProperty <thermo.utils.MixtureProperty>`

The specific classes for the three respective types of properties are:

* :obj:`HeatCapacityGas <thermo.heat_capacity.HeatCapacityGas>`, :obj:`HeatCapacityLiquid <thermo.heat_capacity.HeatCapacityLiquid>`, :obj:`HeatCapacitySolid <thermo.heat_capacity.HeatCapacitySolid>`, :obj:`VolumeSolid <thermo.volume.VolumeSolid>`, :obj:`VaporPressure <thermo.vapor_pressure.VaporPressure>`, :obj:`SublimationPressure <thermo.vapor_pressure.SublimationPressure>`, :obj:`EnthalpyVaporization <thermo.phase_change.EnthalpyVaporization>`, :obj:`EnthalpySublimation <thermo.phase_change.EnthalpySublimation>`, :obj:`Permittivity <thermo.permittivity.Permittivity>`.

* :obj:`VolumeGas <thermo.volume.VolumeGas>`, :obj:`VolumeLiquid <thermo.volume.VolumeLiquid>`, :obj:`ViscosityGas <thermo.viscosity.ViscosityGas>`, :obj:`ViscosityLiquid <thermo.viscosity.ViscosityLiquid>`, :obj:`ThermalConductivityGas <thermo.thermal_conductivity.ThermalConductivityGas>`, :obj:`ThermalConductivityLiquid <thermo.thermal_conductivity.ThermalConductivityLiquid>`

* :obj:`HeatCapacityGasMixture <thermo.heat_capacity.HeatCapacityGasMixture>`, :obj:`HeatCapacityLiquidMixture <thermo.heat_capacity.HeatCapacityLiquidMixture>`, :obj:`HeatCapacitySolidMixture <thermo.heat_capacity.HeatCapacitySolidMixture>`, :obj:`VolumeGasMixture <thermo.volume.VolumeGasMixture>`, :obj:`VolumeLiquidMixture <thermo.volume.VolumeLiquidMixture>`, :obj:`VolumeSolidMixture <thermo.volume.VolumeSolidMixture>`, :obj:`ViscosityLiquidMixture <thermo.viscosity.ViscosityLiquidMixture>`, :obj:`ViscosityGasMixture <thermo.viscosity.ViscosityGasMixture>`, :obj:`ThermalConductivityLiquidMixture <thermo.thermal_conductivity.ViscosityLiquidMixture>`, :obj:`ThermalConductivityGasMixture <thermo.thermal_conductivity.ViscosityGasMixture>`, :obj:`SurfaceTensionMixture <thermo.interface.SurfaceTensionMixture>`

Temperature Dependent Properties
--------------------------------

All arguments and information the property object requires must be provided in the constructor of the object. If a piece of information is not provided, whichever methods require it will not be available for that object.

>>> from thermo import VaporPressure
>>> ethanol_psat = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')

Various data files will be searched to see if information such as Antoine coefficients is available for the compound during the initialization. This behavior can be avoided by setting the optional `load_data` argument to False.

>>> useless_psat = VaporPressure(CASRN='64-17-5', load_data=False)

As many methods may be available, a single method is always selected automatically during initialization. This method can be inspected with the `method` property; if no methods are available, `method` will be None.

>>> ethanol_psat.method, useless_psat.method
('WAGNER_MCGARRY', None)

All available methods can be found by inspecting the `all_methods` attribute:

>>> ethanol_psat.all_methods
{'ANTOINE_POLING', 'Edalat', 'WAGNER_POLING', 'SANJARI', 'COOLPROP', 'LEE_KESLER_PSAT', 'DIPPR_PERRY_8E', 'VDI_PPDS', 'WAGNER_MCGARRY', 'VDI_TABULAR', 'AMBROSE_WALTON', 'BOILING_CRITICAL'}

Calculation of the property at a specific temperature is as easy as calling the object:

>>> ethanol_psat(300.0)
8753.8160

This is actually a cached wrapper around the specific call, `T_dependent_property`:

>>> ethanol_psat.T_dependent_property(300.0)
8753.8160

In truth, that call itself is a wrapper around the `calculate` method allows any method to be used.

>>> ethanol_psat.calculate(T=300.0, method='WAGNER_MCGARRY')
8753.8160
>>> ethanol_psat.calculate(T=300.0, method='DIPPR_PERRY_8E')
8812.9812


Each correlation is associated with temperature limits. These can be inspected as part of the `T_limits` attribute.

>>> ethanol_psat.T_limits
{'WAGNER_MCGARRY': (293.0, 513.92), 'WAGNER_POLING': (159.05, 513.92), 'ANTOINE_POLING': (276.5, 369.54), 'DIPPR_PERRY_8E': (159.05, 514.0), 'COOLPROP': (159.1, 514.71), 'VDI_TABULAR': (300.0, 513.9), 'VDI_PPDS': (159.05, 513.9), 'BOILING_CRITICAL': (0.01, 514.0), 'LEE_KESLER_PSAT': (0.01, 514.0), 'AMBROSE_WALTON': (0.01, 514.0), 'SANJARI': (0.01, 514.0), 'Edalat': (0.01, 514.0)}

Because there is often a need to obtain a property outside the range of the correlation, there are some extrapolation methods available; depending on the method these may be enabled by default.
The full list of extrapolation methods can be see :obj:`here <thermo.utils.TDependentProperty>`.

For vapor pressure, there are actually two separate extrapolation techniques used, one for the low-pressure and thermodynamically reasonable region and another for extrapolating even past the critical point. This can be useful for obtaining initial estimates of phase equilibrium.

The low-pressure region uses :math:`\log(P_{sat}) = A - B/T`, where the coefficients `A` and `B` are calculated from the low-temperature limit and its temperature derivative. The default high-temperature extrapolation is :math:`P_{sat} = \exp\left(A + B/T + C\log(T)\right)`. The coefficients are also determined from the high-temperature limits and its first two temperature derivatives.

When extrapolation is turned on, it is used automatically if a property is requested out of range:

>>> ethanol_psat(100.0), ethanol_psat(1000)
(1.0475e-11, 3.4945e+22)

The default extrapolation methods may be changed in the future, but can be manually specified also. For example, if the `linear` extrapolation method is set, extrapolation will be linear instead of using those fit equations. Because not all properties are suitable for linear extrapolation, some methods have a default `transform` to make the property behave as linearly as possible. This is also used in tabular interpolation:

>>> ethanol_psat.extrapolation = 'linear'
>>> ethanol_psat(100.0), ethanol_psat(1000)
(1.0475e-11, 385182009.4)

The low-temperature linearly extrapolated value is actually the same as before, because it performs a 1/T transform and a log(P) transform on the output, which results in the fit being the same as the default equation for vapor pressure.

To better understand what methods are available, the `valid_methods` method checks all available correlations against their temperature limits.

>>> ethanol_psat.valid_methods(100)
['AMBROSE_WALTON', 'LEE_KESLER_PSAT', 'Edalat', 'BOILING_CRITICAL', 'SANJARI']

If the temperature is not provided, all available methods are returned; the returned value always favors the methods by the ranking defined in thermo.

>>> ethanol_psat.valid_methods()
['WAGNER_MCGARRY', 'WAGNER_POLING', 'DIPPR_PERRY_8E', 'VDI_PPDS', 'COOLPROP', 'ANTOINE_POLING', 'VDI_TABULAR', 'AMBROSE_WALTON', 'LEE_KESLER_PSAT', 'Edalat', 'BOILING_CRITICAL', 'SANJARI']

It is also possible to compare the correlations graphically.

>>> ethanol_psat.plot_T_dependent_property(Tmin=300)

There is also functionality for reversing the calculation - finding out which temperature produces a specific property value. For vapor pressure, we can find out the normal boiling point as follows:

>>> ethanol_psat.solve_property(101325)
351.43136

Functionality for calculating the derivative of the property is also implemented:

>>> ethanol_psat.T_dependent_property_derivative(300)
498.882

Functionality for integrating over a property is also implemented; in the heat capacity case this represents a change in enthalpy:

>>> CH4_Cp = HeatCapacityGas(CASRN='74-82-8')
>>> CH4_Cp.method = 'Poling et al. (2001)'
>>> CH4_Cp.T_dependent_property_integral(300, 500)
8158.64

Because entropy is also needed, the integral of the property over T is implemented too:

>>> CH4_Cp.T_dependent_property_integral_over_T(300, 500)
20.6088

Where it speed is important, these integrals are implemented analytically; otherwise the integration is performed numerically.

Temperature and Pressure Dependent Properties
---------------------------------------------



Notes
-----
There is also the challenge that there is no clear criteria for distinguishing liquids from gases in supercritical mixtures. If the same method is not used for liquids and gases, there will be a sudden discontinuity which can cause numerical issues in modeling.



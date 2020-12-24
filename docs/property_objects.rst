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




Notes
-----
There is also the challenge that there is no clear criteria for distinguishing liquids from gases in supercritical mixtures. If the same method is not used for liquids and gases, there will be a sudden discontinuity which can cause numerical issues in modeling.



Introduction to Property Objects
================================

.. contents:: :local:

For every chemical property, there are lots and lots of methods. The methods can be grouped by which phase they apply to, although some methods are valid for both liquids and gases. 

Properties calculations be separated into three categories:

* Properties of chemicals that depend on **temperature**. Some properties have weak dependence on pressure, like surface tension, and others have no dependence on pressure like vapor pressure by definition.
* Properties of chemicals that depend on **temperature** and **pressure**. Some properties have weak dependence on pressure like thermal conductivity, while other properties depend on pressure fundamentally, like gas volume.
* Properties of mixtures, that depend on **temperature** and **pressure**  and **composition**. Some properties like gas mixture heat capacity require the pressure as an input but do not use it.



These properties are implemented in an object oriented way, with the actual functional algorithms themselves having been separated out into the `chemicals <https://github.com/CalebBell/chemicals>`_ library. The goal of these objects is to make it easy to experiment with different methods.

The base classes for the three respective types of properties are:

* :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
* :obj:`TPDependentProperty <thermo.utils.TPDependentProperty>`
* :obj:`MixtureProperty <thermo.utils.MixtureProperty>`

The specific classes for the three respective types of properties are:

* :obj:`HeatCapacityGas <thermo.heat_capacity.HeatCapacityGas>`, :obj:`HeatCapacityLiquid <thermo.heat_capacity.HeatCapacityLiquid>`, :obj:`HeatCapacitySolid <thermo.heat_capacity.HeatCapacitySolid>`, :obj:`VolumeSolid <thermo.volume.VolumeSolid>`, :obj:`VaporPressure <thermo.vapor_pressure.VaporPressure>`, :obj:`SublimationPressure <thermo.vapor_pressure.SublimationPressure>`, :obj:`EnthalpyVaporization <thermo.phase_change.EnthalpyVaporization>`, :obj:`EnthalpySublimation <thermo.phase_change.EnthalpySublimation>`, :obj:`Permittivity <thermo.permittivity.Permittivity>`,  :obj:`SurfaceTension <thermo.interface.SurfaceTension>`.

* :obj:`VolumeGas <thermo.volume.VolumeGas>`, :obj:`VolumeLiquid <thermo.volume.VolumeLiquid>`, :obj:`ViscosityGas <thermo.viscosity.ViscosityGas>`, :obj:`ViscosityLiquid <thermo.viscosity.ViscosityLiquid>`, :obj:`ThermalConductivityGas <thermo.thermal_conductivity.ThermalConductivityGas>`, :obj:`ThermalConductivityLiquid <thermo.thermal_conductivity.ThermalConductivityLiquid>`

* :obj:`HeatCapacityGasMixture <thermo.heat_capacity.HeatCapacityGasMixture>`, :obj:`HeatCapacityLiquidMixture <thermo.heat_capacity.HeatCapacityLiquidMixture>`, :obj:`HeatCapacitySolidMixture <thermo.heat_capacity.HeatCapacitySolidMixture>`, :obj:`VolumeGasMixture <thermo.volume.VolumeGasMixture>`, :obj:`VolumeLiquidMixture <thermo.volume.VolumeLiquidMixture>`, :obj:`VolumeSolidMixture <thermo.volume.VolumeSolidMixture>`, :obj:`ViscosityLiquidMixture <thermo.viscosity.ViscosityLiquidMixture>`, :obj:`ViscosityGasMixture <thermo.viscosity.ViscosityGasMixture>`, :obj:`ThermalConductivityLiquidMixture <thermo.thermal_conductivity.ViscosityLiquidMixture>`, :obj:`ThermalConductivityGasMixture <thermo.thermal_conductivity.ViscosityGasMixture>`, :obj:`SurfaceTensionMixture <thermo.interface.SurfaceTensionMixture>`

Temperature Dependent Properties
--------------------------------

The following examples introduce how to use some of the methods of the :obj:`TDependentProperty <thermo.utils.TDependentProperty>` objects. The API documentation for :obj:`TDependentProperty <thermo.utils.TDependentProperty>` as well as each specific property such as
:obj:`VaporPressure <thermo.vapor_pressure.VaporPressure>` should be consulted for full details.

Creating Objects
^^^^^^^^^^^^^^^^

All arguments and information the property object requires must be provided in the constructor of the object. If a piece of information is not provided, whichever methods require it will not be available for that object.

>>> from thermo import VaporPressure
>>> ethanol_psat = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')

Various data files will be searched to see if information such as Antoine coefficients is available for the compound during the initialization. This behavior can be avoided by setting the optional `load_data` argument to False. Loading data requires `pandas`, uses more RAM, and is a once-per-process procedure that takes 20-1000 ms per property. For some applications it may be advantageous to provide your own data instead of using the provided data files.

>>> useless_psat = VaporPressure(CASRN='64-17-5', load_data=False)

Temperature-dependent Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As many methods may be available, a single method is always selected automatically during initialization. This method can be inspected with the :obj:`method <thermo.utils.TDependentProperty.method>` property; if no methods are available, `method` will be None. `method` is also a valid parameter when constructing the object, but if the method specified is not available an exception will be raised.

>>> ethanol_psat.method, useless_psat.method
('WAGNER_MCGARRY', None)

All available methods can be found by inspecting the :obj:`all_methods <thermo.utils.TDependentProperty.all_methods>`  attribute:

>>> ethanol_psat.all_methods
{'ANTOINE_POLING', 'EDALAT', 'WAGNER_POLING', 'SANJARI', 'COOLPROP', 'LEE_KESLER_PSAT', 'DIPPR_PERRY_8E', 'VDI_PPDS', 'WAGNER_MCGARRY', 'VDI_TABULAR', 'AMBROSE_WALTON', 'BOILING_CRITICAL'}

Changing the method is as easy as setting a new value to the attribute:

>>> ethanol_psat.method = 'ANTOINE_POLING'
>>> ethanol_psat.method
'ANTOINE_POLING'
>>> ethanol_psat.method = 'WAGNER_MCGARRY'

Calculating Properties
^^^^^^^^^^^^^^^^^^^^^^

Calculation of the property at a specific temperature is as easy as calling the object which triggers the :obj:`__call__ <thermo.utils.TDependentProperty.__call__>` method:

>>> ethanol_psat(300.0)
8753.8160

This is actually a cached wrapper around the specific call, :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`:

>>> ethanol_psat.T_dependent_property(300.0)
8753.8160

The caching of :obj:`__call__ <thermo.utils.TDependentProperty.__call__>` is quite basic - the previously specified temperature is stored, and if the new `T` is the same as the previous `T` the previously calculated result is returned.

There is a lower-level interface for calculating properties with a specified method by name, :obj:`calculate <thermo.utils.TDependentProperty.calculate>`. :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>` is a wrapper around  :obj:`calculate <thermo.utils.TDependentProperty.calculate>` that includes validation of the result.

>>> ethanol_psat.calculate(T=300.0, method='WAGNER_MCGARRY')
8753.8160
>>> ethanol_psat.calculate(T=300.0, method='DIPPR_PERRY_8E')
8812.9812

Limits and Extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^

Each correlation is associated with temperature limits. These can be inspected as part of the :obj:`T_limits <thermo.utils.TDependentProperty.T_limits>` attribute which is loaded on creation of the property object.

>>> ethanol_psat.T_limits
{'WAGNER_MCGARRY': (293.0, 513.92), 'WAGNER_POLING': (159.05, 513.92), 'ANTOINE_POLING': (276.5, 369.54), 'DIPPR_PERRY_8E': (159.05, 514.0), 'COOLPROP': (159.1, 514.71), 'VDI_TABULAR': (300.0, 513.9), 'VDI_PPDS': (159.05, 513.9), 'BOILING_CRITICAL': (0.01, 514.0), 'LEE_KESLER_PSAT': (0.01, 514.0), 'AMBROSE_WALTON': (0.01, 514.0), 'SANJARI': (0.01, 514.0), 'EDALAT': (0.01, 514.0)}

Because there is often a need to obtain a property outside the range of the correlation, there are some extrapolation methods available; depending on the method these may be enabled by default.
The full list of extrapolation methods can be see :obj:`here <thermo.utils.TDependentProperty>`.

For vapor pressure, there are actually two separate extrapolation techniques used, one for the low-pressure and thermodynamically reasonable region and another for extrapolating even past the critical point. This can be useful for obtaining initial estimates of phase equilibrium.

The low-pressure region uses :math:`\log(P_{sat}) = A - B/T`, where the coefficients `A` and `B` are calculated from the low-temperature limit and its temperature derivative. The default high-temperature extrapolation is :math:`P_{sat} = \exp\left(A + B/T + C\log(T)\right)`. The coefficients are also determined from the high-temperature limits and its first two temperature derivatives.

When extrapolation is turned on, it is used automatically if a property is requested out of range:

>>> ethanol_psat(100.0), ethanol_psat(1000)
(1.0475e-11, 3.4945e+22)

The default extrapolation methods may be changed in the future, but can be manually specified also by changing the value of the :obj:`extrapolation <thermo.utils.TDependentProperty.extrapolation>` attribute. For example, if the `linear` extrapolation method is set, extrapolation will be linear instead of using those fit equations. Because not all properties are suitable for linear extrapolation, some methods have a default `transform` to make the property behave as linearly as possible. This is also used in tabular interpolation:

>>> ethanol_psat.extrapolation = 'linear'
>>> ethanol_psat(100.0), ethanol_psat(1000)
(1.0475e-11, 385182009.4)

The low-temperature linearly extrapolated value is actually the same as before, because it performs a 1/T transform and a log(P) transform on the output, which results in the fit being the same as the default equation for vapor pressure.

To better understand what methods are available, the :obj:`valid_methods <thermo.utils.TDependentProperty.valid_methods>` method checks all available correlations against their temperature limits.

>>> ethanol_psat.valid_methods(100)
['AMBROSE_WALTON', 'LEE_KESLER_PSAT', 'EDALAT', 'BOILING_CRITICAL', 'SANJARI']

If the temperature is not provided, all available methods are returned; the returned value favors the methods by the ranking defined in thermo, with the currently selected method as the first item.

>>> ethanol_psat.valid_methods()
['WAGNER_MCGARRY', 'WAGNER_POLING', 'DIPPR_PERRY_8E', 'VDI_PPDS', 'COOLPROP', 'ANTOINE_POLING', 'VDI_TABULAR', 'AMBROSE_WALTON', 'LEE_KESLER_PSAT', 'EDALAT', 'BOILING_CRITICAL', 'SANJARI']

Plotting
^^^^^^^^

It is also possible to compare the correlations graphically with the method :obj:`plot_T_dependent_property <thermo.utils.TDependentProperty.plot_T_dependent_property>`.

>>> ethanol_psat.plot_T_dependent_property(Tmin=300)

.. plot:: plots/vapor_pressure_ethanol_1.py

By default all methods are shown in the plot, but a smaller selection of methods can be specified. The following example compares 30 points in the temperature range 400 K to 500 K, with three of the best methods.

>>> ethanol_psat.plot_T_dependent_property(Tmin=400, Tmax=500, methods=['COOLPROP', 'WAGNER_MCGARRY', 'DIPPR_PERRY_8E'], pts=30)

.. plot:: plots/vapor_pressure_ethanol_2.py

It is also possible to plot the nth derivative of the methods with the `order` parameter. The following plot shows the first derivative of vapor pressure of three estimation methods, a tabular source being interpolated, and 'DIPPR_PERRY_8E' as a reference method.

>>> ethanol_psat.plot_T_dependent_property(Tmin=400, Tmax=500, methods=['BOILING_CRITICAL', 'SANJARI', 'LEE_KESLER_PSAT', 'VDI_TABULAR', 'DIPPR_PERRY_8E'], pts=50, order=1)

.. plot:: plots/vapor_pressure_ethanol_3.py

Plots show how the extrapolation methods work. By default plots do not show extrapolated values from methods, but this can be forced by setting `only_valid` to False. It is easy to see that extrapolation is designed to show the correct trend, but that individual methods will have very different extrapolations.

>>> ethanol_psat.plot_T_dependent_property(Tmin=1, Tmax=300, methods=['VDI_TABULAR', 'DIPPR_PERRY_8E', 'COOLPROP'], pts=50, only_valid=False)

.. plot:: plots/vapor_pressure_ethanol_4.py


Calculating Temperature From Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is also functionality for reversing the calculation - finding out which temperature produces a specific property value. The method is :obj:`solve_property <thermo.utils.TDependentProperty.solve_property>`. For vapor pressure, we can use this technique to find out the normal boiling point as follows:

>>> ethanol_psat.solve_property(101325)
351.43136

The experimentally reported value is 351.39 K.

Property Derivatives
^^^^^^^^^^^^^^^^^^^^

Functionality for calculating the derivative of the property is also implemented as :obj:`T_dependent_property_derivative <thermo.utils.TDependentProperty.T_dependent_property_derivative>` :

>>> ethanol_psat.T_dependent_property_derivative(300)
498.882

The derivatives are numerical unless a special implementation has been added to the property's  :obj:`calculate_derivative <thermo.utils.TDependentProperty.calculate_derivative>` method.

Higher order derivatives are available as well with the `order` argument. All higher-order derivatives are numerical, and they tend to have reduced numerical precision due to floating point limitations.

>>> ethanol_psat.T_dependent_property_derivative(300.0, order=2)
24.74
>>> ethanol_psat.T_dependent_property_derivative(300.0, order=3)
2.75

Property Integrals
^^^^^^^^^^^^^^^^^^

Functionality for integrating over a property is implemented as :obj:`T_dependent_property_integral <thermo.utils.TDependentProperty.T_dependent_property_integral>`.

.. math::
    \text{integral} = \int_{T_1}^{T_2} \text{property} \; dT

When the property is heat capacity, this calculation represents a change in enthalpy:

.. math::
    \Delta H = \int_{T_1}^{T_2} C_p \; d T

>>> CH4_Cp = HeatCapacityGas(CASRN='74-82-8')
>>> CH4_Cp.method = 'Poling et al. (2001)'
>>> CH4_Cp.T_dependent_property_integral(300, 500)
8158.64

Besides enthalpy, a commonly used integral is that of the property divided by `T`:

.. math::
    \text{integral} = \int_{T_1}^{T_2} \frac{\text{property}}{T} \; dT

When the property is heat capacity, this calculation represents a change in entropy:

.. math::
    \Delta S = \int_{T_1}^{T_2} \frac{C_p}{T} \; d T

This integral, property over T, is implemented as  :obj:`T_dependent_property_integral_over_T <thermo.utils.TDependentProperty.T_dependent_property_integral_over_T>` :

>>> CH4_Cp.T_dependent_property_integral_over_T(300, 500)
20.6088

Where speed has been important so far, these integrals have been implemented analytically in a property object's :obj:`calculate_integral <thermo.utils.TDependentProperty.calculate_integral>`  and :obj:`calculate_integral_over_T <thermo.utils.TDependentProperty.calculate_integral_over_T>` method; otherwise the integration is performed numerically.

Using Tabular Data
^^^^^^^^^^^^^^^^^^

A common scenario is that there are no correlations available for a compound, and that estimation methods are not applicable. However, there may be a few experimental data points available in the literature. In this case, the data can be specified and used directly with the :obj:`add_tabular_data <thermo.utils.TDependentProperty.add_tabular_data>` method. Extrapolation can often show the correct trends for these properties from even a few data points.

In the example below, we take 5 data points on the vapor pressure of water from 300 K to 350 K, and use them to extrapolate and estimate the triple temperature and critical temperature (assuming we know the triple and critical pressures).

>>> from thermo import *
>>> import numpy as np
>>> w = VaporPressure(Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, CASRN='7732-18-5', extrapolation='AntoineAB')
>>> Ts = np.linspace(300, 350, 5).tolist()
>>> Ps = [3533.9, 7125., 13514., 24287., 41619.]
>>> w.add_tabular_data(Ts=Ts, properties=Ps)
>>> w.solve_property(610.707), w.solve_property(22048320)
(272.76, 616.67)

The experimental values are 273.15 K and 647.14 K.

Adding New Methods
^^^^^^^^^^^^^^^^^^

While a great many property methods have been implemented, there is always the case where a new one must be added. To support that, the method :obj:`add_method <thermo.utils.TDependentProperty.add_method>` will add a user-specified method and switch the method selected to the newly added method.

As an example, we can compare the default vapor pressure formulation for n-hexane against a set of Antoine coefficients on the `NIST WebBook <https://webbook.nist.gov/cgi/cbook.cgi?ID=C110543&Mask=4&Type=ANTOINE&Plot=on>`_.

>>> from chemicals import *
>>> from thermo import *
>>> obj = VaporPressure(CASRN= '110-54-3')
>>> obj(200)
20.742
>>> f = lambda T: Antoine(T=T, A=3.45604+5, B=1044.x038, C=-53.893)
>>> obj.add_method(f=f, name='WebBook', Tmin=177.70, Tmax=264.93)
>>> obj.method
'WebBook'
>>> obj.extrapolation = 'AntoineAB'
>>> obj(200.0)
20.432

We can, again, extrapolate quite easily and estimate the triple temperature and critical temperature from these correlations (if we know the triple pressure and critical pressure).

>>> obj.solve_property(1.378), obj.solve_property(3025000.0)
(179.42, 508.033)

Optionally, some derivatives and integrals can be provided for new methods as well. This avoids having to compute derivatives or integrals numerically. SymPy may be helpful to find these analytical derivatives or integrals in many cases, as in the following example:

>>> from sympy import symbols, lambdify, diff
>>> T = symbols('T')
>>> A, B, C = 3.45604+5, 1044.038, -53.893
>>> expr = 10**(A - B/(T + C))
>>> f = lambdify(T, expr)
>>> f_der = lambdify(T, diff(expr, T))
>>> f_der2 = lambdify(T, diff(expr, T, 2))
>>> f_der3 = lambdify(T, diff(expr, T, 3))
>>> obj.add_method(f=f, f_der=f_der, f_der2=f_der2, f_der3=f_der3, name='WebBookSymPy', Tmin=177.70, Tmax=264.93)
>>> obj.method, obj(200), obj.T_dependent_property_derivative(200.0, order=2)
('WebBookSymPy', 20.43298036711, 0.2276289268)

Note that adding methods like this breaks the ability to export as json and the repr of the object is no longer complete.

Adding New Correlation Coefficient Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While adding entirely new methods is useful, it is more common to want to use different coefficients in an existing equation.
A number of different equations are recognized, and accept/require the parameters as per their function name in e.g. :obj:`chemicals.vapor_pressure.Antoine`. More than one set of coefficients can be added for each model. After adding a new correlation the method is set to that method.

>>> obj = VaporPressure()
>>> obj.add_correlation(name='WebBook', model='Antoine', Tmin=177.70, Tmax=264.93,  A=3.45604+5, B=1044.038, C=-53.893)
>>> obj(200)
20.43298036711

It is also possible to specify the parameters in the constructor of the object as well:

>>> obj = VaporPressure(Antoine_parameters={'WebBook': {'A': 8.45604, 'B': 1044.038, 'C': -53.893, 'Tmin': 177.7, 'Tmax': 264.93}})
>>> obj(200)
20.43298036711

More than one set of parameters and more than one model may be specified this way; the model name is the same, with '_parameters' appended to it.

For a full list of supported correlations (and their names), see :obj:`add_correlation <thermo.utils.TDependentProperty.add_correlation>`.

Temperature and Pressure Dependent Properties
---------------------------------------------
The pressure dependent objects work much like the temperature dependent ones; in fact, they subclass :obj:`TDependentProperty <thermo.utils.TDependentProperty>`.
They have many new methods that require pressure as an input however. They work in two parts: a low-pressure correlation component, and a high-pressure correlation component. The high-pressure component usually but not always requires a low-pressure calculation to be performed first as its input.

Creating Objects
^^^^^^^^^^^^^^^^

All arguments and information the property object requires must be provided in the constructor of the object. If a piece of information is not provided, whichever methods require it will not be available for that object. Many pressure-dependent property correlations are actually dependent on other properties being calculated first. A mapping of those dependencies is as follows:


* Liquid molar volume: Depends on :obj:`VaporPressure <thermo.vapor_pressure.VaporPressure>`
* Gas viscosity: Depends on :obj:`VolumeGas <thermo.volume.VolumeGas>`
* Liquid viscosity: Depends on  :obj:`VaporPressure <thermo.vapor_pressure.VaporPressure>`
* Gas thermal conductivity: Depends on :obj:`VolumeGas <thermo.volume.VolumeGas>`, :obj:`HeatCapacityGas <thermo.heat_capacity.HeatCapacityGas>`, :obj:`ViscosityGas <thermo.viscosity.ViscosityGas>`


The required input objects should be created first, and provided as an input to the dependent object:

>>> water_psat = VaporPressure(Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, CASRN='7732-18-5')
>>> water_mu = ViscosityLiquid(CASRN="7732-18-5", MW=18.01528, Tm=273.15, Tc=647.14, Pc=22048320.0, Vc=5.6e-05, omega=0.344, method="DIPPR_PERRY_8E", Psat=water_psat, method_P="LUCAS")

Various data files will be searched to see if information such as DIPPR expression coefficients are available for the compound during the initialization. This behavior can be avoided by setting the optional `load_data` argument to False.

Pressure-dependent Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

The pressure and temperature dependent object selects a low-pressure and a high-pressure method automatically during initialization.
These method can be inspected with the :obj:`method <thermo.utils.TPDependentProperty.method>` and :obj:`method_P <thermo.utils.TPDependentProperty.method_P>` properties.
If no low-pressure methods are available, `method` will be None. If no high-pressure methods are available, `method_P` will be None. `method` and `method_P` are also valid parameters when constructing the object, but if either of the methods specified is not available an exception will be raised.

>>> water_mu.method, water_mu.method_P
('DIPPR_PERRY_8E', 'LUCAS')

All available low-pressure methods can be found by inspecting the :obj:`all_methods <thermo.utils.TPDependentProperty.all_methods>` attribute:

>>> water_mu.all_methods
{'COOLPROP', 'DIPPR_PERRY_8E', 'VISWANATH_NATARAJAN_3', 'VDI_PPDS', 'LETSOU_STIEL'}

All available high-pressure methods can be found by inspecting the :obj:`all_methods_P <thermo.utils.TPDependentProperty.all_methods_P>` attribute:

>>> water_mu.all_methods_P
{'COOLPROP', 'LUCAS'}

Changing the low-pressure method or the high-pressure method is as easy as setting a new value to the attribute:

>>> water_mu.method = 'VDI_PPDS'
>>> water_mu.method
'VDI_PPDS'
>>> water_mu.method_P = 'COOLPROP'
>>> water_mu.method_P
'COOLPROP'

Calculating Properties
^^^^^^^^^^^^^^^^^^^^^^

Calculation of the property at a specific temperature and pressure is as easy as calling the object which triggers the :obj:`__call__ <thermo.utils.TPDependentProperty.__call__>` method:

>>> water_mu.method = 'VDI_PPDS'
>>> water_mu.method_P = 'COOLPROP'
>>> water_mu(T=300.0, P=1e5)
0.000853742

This is actually a cached wrapper around the specific call, :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`:

>>> water_mu.TP_dependent_property(300.0, P=1e5)
0.000853742

The caching of :obj:`__call__ <thermo.utils.TPDependentProperty.__call__>` is quite basic - the previously specified temperature and pressure are stored, and if the new `T` and `P` are the same as the previous `T` and `P` the previously calculated result is returned.

There is a lower-level interface for calculating properties with a specified method by name, :obj:`calculate_P <thermo.utils.TDependentProperty.calculate_P>`. :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>` is a wrapper around  :obj:`calculate_P <thermo.utils.TDependentProperty.calculate_P>` that includes validation of the result.

>>> water_mu.calculate_P(T=300.0, P=1e5, method='COOLPROP')
0.000853742
>>> water_mu.calculate_P(T=300.0, P=1e5, method='LUCAS')
0.000865292

The above examples all show using calculating the property with a pressure specified. The same :obj:`TDependentProperty <thermo.utils.TDependentProperty>` methods are available too, so all the low-pressure calculation calls are also available.

>>> water_mu.calculate(T=300.0, method='VISWANATH_NATARAJAN_3')
0.000856467
>>> water_mu.T_dependent_property(T=400.0) 
0.000217346

Limits and Extrapolation
^^^^^^^^^^^^^^^^^^^^^^^^

The same temperature limits and low-pressure extrapolation methods are available as for :obj:`TDependentProperty <thermo.utils.TDependentProperty>`.

>>> water_mu.valid_methods(T=480)
['DIPPR_PERRY_8E', 'COOLPROP', 'VDI_PPDS', 'LETSOU_STIEL']
>>> water_mu.extrapolation
'linear'

To better understand what methods are available, the :obj:`valid_methods_P <thermo.utils.TDependentProperty.valid_methods_P>` method checks all available high-pressure correlations against their temperature and pressure limits.

>>> water_mu.valid_methods_P(T=300, P=1e9)
['LUCAS', 'COOLPROP']
>>> water_mu.valid_methods_P(T=300, P=1e10)
['LUCAS']
>>> water_mu.valid_methods_P(T=900, P=1e6)
['LUCAS']

If the temperature and pressure are not provided, all available methods are returned; the returned value favors the methods by the ranking defined in thermo, with the currently selected method as the first item.

>>> water_mu.valid_methods_P()
['LUCAS', 'COOLPROP']

Plotting
^^^^^^^^

It is possible to compare the correlations graphically with the method :obj:`plot_TP_dependent_property <thermo.utils.TPDependentProperty.plot_TP_dependent_property>`.

>>> water_mu.plot_TP_dependent_property(Tmin=400, Pmin=1e5, Pmax=1e8, methods_P=['COOLPROP','LUCAS'], pts=15, only_valid=False)

.. plot:: plots/viscosity_water_0.py

This can be a little confusing; but isotherms and isobars can be plotted as well, which are more straight forward. The respective methods are :obj:`plot_isotherm <thermo.utils.TPDependentProperty.plot_isotherm>` and :obj:`plot_isobar <thermo.utils.TPDependentProperty.plot_isobar>`:

>>> water_mu.plot_isotherm(T=350, Pmin=1e5, Pmax=1e7, pts=50)

.. plot:: plots/viscosity_water_1.py

>>> water_mu.plot_isobar(P=1e7, Tmin=300, Tmax=600, pts=50)

.. plot:: plots/viscosity_water_2.py

Calculating Conditions From Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The method is :obj:`solve_property <thermo.utils.TDependentProperty.solve_property>` works only on the low-pressure correlations.

>>> water_mu.solve_property(1e-3)
294.0711641

Property Derivatives
^^^^^^^^^^^^^^^^^^^^

Functionality for calculating the temperature derivative of the property is implemented twice; as :obj:`T_dependent_property_derivative <thermo.utils.TDependentProperty.T_dependent_property_derivative>` using the low-pressure correlations, and as :obj:`TP_dependent_property_derivative_T <thermo.utils.TPDependentProperty.TP_dependent_property_derivative_T>` using the high-pressure correlations that require pressure as an input.

>>> water_mu.T_dependent_property_derivative(300)
-1.893961e-05
>>> water_mu.TP_dependent_property_derivative_T(300, P=1e7)
-1.927268e-05

The derivatives are numerical unless a special implementation has been added to the property's :obj:`calculate_derivative_T <thermo.utils.TPDependentProperty.calculate_derivative_T>`  and/or :obj:`calculate_derivative <thermo.utils.TDependentProperty.calculate_derivative>` method.

Higher order derivatives are available as well with the `order` argument.

>>> water_mu.T_dependent_property_derivative(300.0, order=2)
5.923372e-07
>>> water_mu.TP_dependent_property_derivative_T(300.0, P=1e6, order=2)
-1.40946e-06

Functionality for calculating the pressure derivative of the property is also implemented as :obj:`TP_dependent_property_derivative_P <thermo.utils.TPDependentProperty.TP_dependent_property_derivative_P>`:

>>> water_mu.TP_dependent_property_derivative_P(P=5e7, T=400)
4.27782809e-13

The derivatives are numerical unless a special implementation has been added to the property's  :obj:`calculate_derivative_P <thermo.utils.TPDependentProperty.calculate_derivative_P>` method.

Higher order derivatives are available as well with the `order` argument.

>>> water_mu.TP_dependent_property_derivative_P(P=5e7, T=400, order=2)
-1.1858461e-15

Property Integrals
^^^^^^^^^^^^^^^^^^
The same functionality for integrating over a property as in temperature-dependent objects is available, but only for integrating over temperature using low pressure correlations. No other use cases have been identified requiring integration over high-pressure conditions, or integration over the pressure domain.

>>> water_mu.T_dependent_property_integral(300, 400) # Integrating over viscosity has no physical meaning
0.04243

Using Tabular Data
^^^^^^^^^^^^^^^^^^

If there are experimentally available data for a property at high and low pressure, an interpolation table can be created and used as follows. The CoolProp method is used to generate a small table, and is then added as a new method in the example below.

>>> from thermo import *
>>> import numpy as np
>>> Ts = [300, 400, 500]
>>> Ps = [1e5, 1e6, 1e7]
>>> table = [[water_mu.calculate_P(T, P, "COOLPROP") for T in Ts] for P in Ps]
>>> water_mu.method_P
'LUCAS'
>>> water_mu.add_tabular_data_P(Ts, Ps, table)
>>> water_mu.method_P
'Tabular data series #0'
>>> water_mu(400, 1e7), water_mu.calculate_P(400, 1e7, "COOLPROP")
(0.000221166933349, 0.000221166933349)
>>> water_mu(450, 5e6), water_mu.calculate_P(450, 5e6, "COOLPROP")
(0.00011340, 0.00015423)

The more data points used, the closer a property will match.

Mixture Properties
------------------

Notes
-----
There is also the challenge that there is no clear criteria for distinguishing liquids from gases in supercritical mixtures. If the same method is not used for liquids and gases, there will be a sudden discontinuity which can cause numerical issues in modeling.



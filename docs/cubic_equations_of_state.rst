Introduction to Cubic Equations of State
========================================

.. contents:: :local:

Cubic equations of state provide thermodynamically-consistent and relatively fast models for pure chemicals and mixtures. They are normally used to represent gases and liquids.

The generic three-parameter form is as follows:

    .. math::
        P=\frac{RT}{V-b}-\frac{a\alpha(T)}{V^2 + \delta V + \epsilon}

This forms the basis of the implementation in `thermo`.

Two separate interfaces are provided, :obj:`thermo.eos` for pure component modeling and :obj:`thermo.eos_mix` for multicomponent modeling. Pure components are quite a bit faster than multicomponent mixtures, because the Van der Waals mixing rules conventionally used take N^2 operations to compute :math:`\alpha(T)`:

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

The other slow parts which applies to both types are calculating some basic properties (the list is at :obj:`set_properties_from_solution <thermo.eos.GCEOS.set_properties_from_solution>`) that other properties may depend on, and calculating the molar volume given a pair of (`T`, `P`) inputs (an entire submodule :obj:`thermo.eos_volume` discusses and implements this topic). Both of those calculations are constant-time, so their overhead is the same for pure components and multicomponent mixtures.

Working With Pure Components
----------------------------

We can use the :obj:`GCEOS <thermo.eos.GCEOS>` (short for "General Cubic Equation Of State") interface with any component or implemented equation of state, but for simplicity n-hexane is used with the Peng-Robinson EOS. Its critical temperature is 507.6 K, critical pressure 3.025 MPa, and acentric factor is 0.2975.

The state must be specified along with the critical constants when initializing a :obj:`GCEOS <thermo.eos.GCEOS>` object; we use 400 K and 1e6 Pa here:

>>> from thermo import *
>>> eos = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400., P=1E6)
>>> eos
PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400.0, P=1000000.0)

The :obj:`__repr__ <thermo.eos.GCEOS.__repr__>` string is designed to show all the inputs to the object. 

We can check the volume solutions with the :obj:`raw_volumes <thermo.eos.GCEOS.raw_volumes>` attribute:

>>> eos.raw_volumes
(0.0001560731847856, 0.002141876816741, 0.000919295474982)

At this point there are three real volume, so there is a liquid-like and a vapor-like solution available. The :obj:`phase <thermo.eos.GCEOS.phase>` attribute will have the value of 'l/g' in this state; otherwise it will be 'l' or 'g'.

>>> eos.phase
'l/g'

The basic properties calculated at initialization are directly attributes, and can be accessed as such. Liquid-like properties have "_l" at the end of their name, and "_g" is at the end of gas-like properties.

>>> eos.H_dep_l
-26111.877
>>> eos.S_dep_g
-6.4394518
>>> eos.dP_dT_l
288501.633

All calculations in :obj:`thermo.eos` and :obj:`thermo.eos_mix` are on a molar basis; molecular weight is never provided or needed. All outputs are in base SI units (K, Pa, m^3, mole, etc). This simplified development substantially. For working with mass-based units, use the :obj:`Phase <thermo.phases.Phase>` interface. The :obj:`thermo.eos` and :obj:`thermo.eos_mix` interfaces were developed prior to the :obj:`Phase <thermo.phases.Phase>` interface and does have some features not exposed in the :obj:`Phase <thermo.phases.Phase>` interface however.

Other properties are either implemented as methods that require arguments, or Python properties which act just like attributes but calculate the results on the fly. For example, the liquid-phase fugacity :obj:`fugacity_l <thermo.eos.GCEOS.fugacity_l>` or the gas isobaric (constant-pressure) expansion coefficient are properties.

>>> eos.fugacity_l
421597.00785
>>> eos.beta_g
0.0101232239

There are an awful lot of these properties, because many of them are derivatives subject to similar conditions. A full list is in the documentation for :obj:`GCEOS <thermo.eos.GCEOS>`. There are fewer calls that take temperature, such as :obj:`Hvap <thermo.eos.GCEOS.Hvap>` which calculates the heat of vaporization of the object at a specified temperature:

>>> eos.Hvap(300)
31086.2

Once an object has been created, it can be used to instantiate new :obj:`GCEOS <thermo.eos.GCEOS>` objects at different conditions, without re-specifying the critical constants and other parameters that may be needed.

>>> eos.to(T=300.0, P=1e5)
PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=300.0, P=100000.0)
>>> eos.to(V=1e2, P=1e5)
PR(Tc=507.6, Pc=3025000.0, omega=0.2975, P=100000.0, V=100.0)
>>> eos.to(V=1e2, T=300)
PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=300, V=100.0)

As was seen in the examples above, any two of `T`, `P`, `V` can be used to specify the state of the object. The input variables of the object are stored and can be checked with :obj:`state_specs <thermo.eos.GCEOS.state_specs>` :

>>> eos.state_specs
{'T': 400.0, 'P': 1000000.0}

The individual parts of the generic cubic equation are stored as well. We can use them to check that the pressure equation is satisfied:

>>> from thermo.eos import R
>>> R*eos.T/(eos.V_l-eos.b) - eos.a_alpha/(eos.V_l**2 + eos.V_l*eos.delta + eos.epsilon)
1000000.000000
>>> R*eos.T/(eos.V_g-eos.b) - eos.a_alpha/(eos.V_g**2 + eos.V_g*eos.delta + eos.epsilon)
1000000.000000

Note that as floating points are not perfectly precise, some small error may be shown but great care has been taken to minimize this.

The value of the gas constant used is 8.31446261815324 J/(mol*K). This is near the full precision of floating point numbers, but not quite. It is now an exact value used as a "definition" in the SI system. Note that other implementations of equations of state may not use the full value of the gas constant, but the author strongly recommends anyone considering writing their own EOS implementation use the full gas constant. This will allow more interchangeable results.


Pure Component Equilibrium
--------------------------
Continuing with the same state and example as before, there were two solutions available from the equation of state. However, unless the exact temperature 400 K and pressure 1 MPa happens to be on the saturation line, there is always one more thermodynamically stable state. We need to use the departure Gibbs free energy to determine which state is more stable. For a pure component, the state which minimizes departure Gibbs free energy is the most stable state.

>>> eos = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400., P=1E6)
>>> eos.G_dep_l, eos.G_dep_g
(-2872.498434, -973.5198207)

It is easy to see the liquid phase is more stable. This shortcut of using departure Gibbs free energy is valid only for pure components with all phases using the ideal-gas reference state. The full criterial is whichever state minimizes the actual Gibbs free energy.

The method :obj:`more_stable_phase <thermo.eos.GCEOS.more_stable_phase>` does this check and returns either 'l' or 'g':

>>> eos.more_stable_phase
'l'

For a pure component, there is a vapor-liquid equilibrium line right up to the critical point which defines the vapor pressure of the fluid. This can be calculated using the :obj:`Psat <thermo.eos.GCEOS.Psat>` method:

>>> eos.Psat(400.0)
466205.073739

The result is accurate to more than 10 digits, and is implemented using some fancy mathematical techniques that allow a direct calculation of the vapor pressure. A few more digits can be obtained by setting `polish` to True, which polishes the result with a newton solver to as much accuracy as a floating point number can provide:

>>> 1-eos.Psat(400, polish=True)/eos.Psat(400)
1.6e-14

A few more methods of interest are :obj:`V_l_sat <thermo.eos.GCEOS.V_l_sat>` and :obj:`V_g_sat <thermo.eos.GCEOS.V_g_sat>` which calculate the saturation liquid and molar volumes; :obj:`Tsat <thermo.eos.GCEOS.Tsat>` which calculates the saturation temperature given a specified pressure, and :obj:`phi_sat <thermo.eos.GCEOS.phi_sat>`  which computes the saturation fugacity coefficient given a temperature.

>>> eos.V_l_sat(298.15), eos.V_g_sat(500)
(0.0001303559, 0.0006827569)
>>> eos.Tsat(101325.0)
341.76265
>>> eos.phi_sat(425.0)
0.8349716

Working With Mixtures
---------------------

Using mixture from :obj:`thermo.eos_mix` is first illustrated using an equimolar mixture of nitrogen-methane at 115 K and 1 MPa and the Peng-Robinson equation of state:

>>> eos = PRMIX(T=115.0, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0.0, 0.0289], [0.0289, 0.0]])
>>> eos.V_l, eos.V_g
(3.658707770e-05, 0.00070676607)
>>> eos.fugacities_l, eos.fugacities_g
([838516.99, 78350.27], [438108.61, 359993.48])

All of the properties available in :obj:`GCEOS <thermo.eos.GCEOS>` are also available for :obj:`GCEOSMIX <thermo.eos_mix.GCEOSMIX>` objects.

New  :obj:`GCEOSMIX <thermo.eos_mix.GCEOSMIX>` objects can be created with the :obj:`to <thermo.eos_mix.GCEOSMIX.to>` method, which accepts new mole fractions `zs` as well as new state variables. If a new composition `zs` is not provided, the current composition is also used for the new object.

>>> eos.to(T=300.0, P=1e5)
PRMIX(Tcs=[126.1, 190.6], Pcs=[3394000.0, 4604000.0], omegas=[0.04, 0.011], kijs=[[0.0, 0.0289], [0.0289, 0.0]], zs=[0.5, 0.5], T=300.0, P=100000.0)
>>> eos.to(T=300.0, P=1e5, zs=[.1, .9])
PRMIX(Tcs=[126.1, 190.6], Pcs=[3394000.0, 4604000.0], omegas=[0.04, 0.011], kijs=[[0.0, 0.0289], [0.0289, 0.0]], zs=[0.1, 0.9], T=300.0, P=100000.0)
>>> eos.to(V=1, P=1e5, zs=[.4, .6])
PRMIX(Tcs=[126.1, 190.6], Pcs=[3394000.0, 4604000.0], omegas=[0.04, 0.011], kijs=[[0.0, 0.0289], [0.0289, 0.0]], zs=[0.4, 0.6], P=100000.0, V=1)
>>> eos.to(V=1.0, T=300.0, zs=[.4, .6])
PRMIX(Tcs=[126.1, 190.6], Pcs=[3394000.0, 4604000.0], omegas=[0.04, 0.011], kijs=[[0.0, 0.0289], [0.0289, 0.0]], zs=[0.4, 0.6], T=300.0, V=1.0)


It is possible to create new :obj:`GCEOSMIX <thermo.eos_mix.GCEOSMIX>` objects with the :obj:`subset <thermo.eos_mix.GCEOSMIX.subset>` method which uses only some of the initially specified components:


>>> kijs = [[0.0, 0.00076, 0.00171], [0.00076, 0.0, 0.00061], [0.00171, 0.00061, 0.0]]
>>> PR3 = PRMIX(Tcs=[469.7, 507.4, 540.3], zs=[0.8168, 0.1501, 0.0331], omegas=[0.249, 0.305, 0.349], Pcs=[3.369E6, 3.012E6, 2.736E6], T=322.29, P=101325.0, kijs=kijs)
>>> PR3.subset([1,2])
PRMIX(Tcs=[507.4, 540.3], Pcs=[3012000.0, 2736000.0], omegas=[0.305, 0.349], kijs=[[0.0, 0.00061], [0.00061, 0.0]], zs=[0.8193231441048, 0.1806768558951], T=322.29, P=101325.0)
>>> PR3.subset([1,2], T=500.0, P=1e5, zs=[.2, .8])
PRMIX(Tcs=[507.4, 540.3], Pcs=[3012000.0, 2736000.0], omegas=[0.305, 0.349], kijs=[[0.0, 0.00061], [0.00061, 0.0]], zs=[0.2, 0.8], T=500.0, P=100000.0)
>>> PR3.subset([1,2], zs=[.2, .8])
PRMIX(Tcs=[507.4, 540.3], Pcs=[3012000.0, 2736000.0], omegas=[0.305, 0.349], kijs=[[0.0, 0.00061], [0.00061, 0.0]], zs=[0.2, 0.8], T=322.29, P=101325.0)


It is also possible to create pure :obj:`GCEOS <thermo.eos.GCEOS>` objects:

>>> PR3.pures()
[PR(Tc=469.7, Pc=3369000.0, omega=0.249, T=322.29, P=101325.0), PR(Tc=507.4, Pc=3012000.0, omega=0.305, T=322.29, P=101325.0), PR(Tc=540.3, Pc=2736000.0, omega=0.349, T=322.29, P=101325.0)]

Temperature, pressure, mole number, and mole fraction derivatives of the log fugacity coefficients are available as well with the methods :obj:`dlnphis_dT <thermo.eos_mix.GCEOSMIX.dlnphis_dT>`, :obj:`dlnphis_dP <thermo.eos_mix.GCEOSMIX.dlnphis_dP>`, :obj:`dlnphis_dns <thermo.eos_mix.GCEOSMIX.dlnphis_dns>`, and :obj:`dlnphis_dzs <thermo.eos_mix.GCEOSMIX.dlnphis_dzs>`:

>>> PR3.dlnphis_dT('l')
[0.029486952019, 0.03514175794, 0.040281845273]
>>> PR3.dlnphis_dP('l')
[-9.8253779e-06, -9.8189093031e-06, -9.8122598e-06]
>>> PR3.dlnphis_dns(PR3.Z_l)
[[-0.0010590517, 0.004153228837, 0.007300114797], [0.0041532288, -0.016918292791, -0.0257680231], [0.0073001147, -0.02576802316, -0.0632916462]]
>>> PR3.dlnphis_dzs(PR3.Z_l)
[[0.0099380692, 0.0151503498, 0.0182972357], [-0.038517738, -0.059589260, -0.068438990], [-0.070571069, -0.103639207, -0.141162830]]

Other features
--------------

Hashing
^^^^^^^

It is possible to compare the two objects with each other to see if they have the same kijs, model parameters, and components by using the  :obj:`model_hash <thermo.eos.GCEOS.model_hash>` method:

>>> PR_case = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0.41],[0.41,0]])
>>> SRK_case = SRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0.41],[0.41,0]])

>>> PR_case.model_hash() == SRK_case.model_hash()
False

It is possible to see if both the exact state and the model match between two different objects by using the :obj:`state_hash <thermo.eos.GCEOS.state_hash>` method:

>>> PR_case2 = PRMIX(T=116, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0.41],[0.41,0]])
>>> PR_case.model_hash() == PR_case2.model_hash()
True
>>> PR_case.state_hash() == PR_case2.state_hash()
False

:obj:`state_hash <thermo.eos.GCEOS.state_hash>` is the __hash__ method of the object.

And finally it is possible to see if two objects are exactly identical, including cached calculation results, by using the  :obj:`exact_hash <thermo.eos.GCEOS.exact_hash>` method:

>>> PR_case3 = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0.41],[0.41,0]])
>>> PR_case.state_hash() == PR_case3.state_hash()
True
>>> PR_case.exact_hash() == PR_case3.exact_hash()
True
>>> _ = PR_case.da_alpha_dT_ijs
>>> PR_case.exact_hash() == PR_case3.exact_hash()
False

Serialization
^^^^^^^^^^^^^
All cubic EOS models offer a :obj:`as_json <thermo.eos.GCEOS.as_json>` method and a :obj:`from_json <thermo.eos.GCEOS.from_json>` to serialize the object state for transport over a network, storing to disk, and passing data between processes.

>>> import json
>>> eos = PRSV2MIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6, kappa1s=[0.05104], kappa2s=[0.8634], kappa3s=[0.460])
>>> json_stuff = json.dumps(eos.as_json())
>>> new_eos = GCEOSMIX.from_json(json.loads(json_stuff))
>>> assert new_eos == eos

Other json libraries can be used besides the standard json library by design.

Storing and recreating objects with Python's :py:func:`pickle.dumps` library is also tested; this can be faster than using JSON at the cost of being binary data.

Mixture Equilibrium
-------------------
Unlike pure components, it is not straightforward to determine what the equilibrium state is for mixtures. Different algorithms are used such as sequential substitution and Gibbs minimization. All of those require initial guesses, which usually come from simpler thermodynamic models. While in practice it is possible to determine the equilibrium composition to an N-phase problem, in theory a global optimization algorithm must be used.

More details on this topic can be found in the :obj:`thermo.flash` module.


Using Units with Cubic Equations of State
-----------------------------------------
There is a pint wrapper to use these objects  as well.

>>> from thermo.units import *
>>> kwargs = dict(T=400.0*u.degC, P=30*u.psi, Tcs=[126.1, 190.6]*u.K, Pcs=[33.94E5, 46.04E5]*u.Pa, omegas=[0.04, 0.011]*u.dimensionless, zs=[0.5, 0.5]*u.dimensionless, kijs=[[0.0, 0.0289], [0.0289, 0.0]]*u.dimensionless)
>>> eos_units = PRMIX(**kwargs)
>>> eos_units.H_dep_g, eos_units.T
(<Quantity(-2.53858854, 'joule / mole')>, <Quantity(673.15, 'kelvin')>)


>>> base = IG(T=300.0*u.K, P=1e6*u.Pa)
>>> base.V_g
<Quantity(0.00249433879, 'meter ** 3 / mole')>

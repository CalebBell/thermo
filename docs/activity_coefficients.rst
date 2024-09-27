Introduction to Activity Coefficient Models
===========================================

.. contents:: :local:

Vapor-liquid and liquid-liquid equilibria systems can have all sorts of different behavior. Raoult's law can describe only temperature and pressure dependence, so a correction factor that adds dependence on composition called the "activity coefficient" is often used. This is a separate approach to using an equation of state, but because direct vapor pressure correlations are used with the activity coefficients, a higher-accuracy result can be obtained for phase equilibria.

While these models are often called "activity coefficient models", they are in fact actually a prediction for excess Gibbs energy. The activity coefficients that are used for phase equilibria are derived from the partial mole number derivative of excess Gibbs energy according to the following expression:

.. math::
    \gamma_i = \exp\left(\frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}\right)

There are 5 basic activity coefficient models in thermo:

* :obj:`NRTL <thermo.nrtl.NRTL>`
* :obj:`Wilson <thermo.wilson.Wilson>`
* :obj:`UNIQUAC <thermo.uniquac.UNIQUAC>`
* :obj:`RegularSolution <thermo.regular_solution.RegularSolution>`
* :obj:`UNIFAC <thermo.unifac.UNIFAC>`

Each of these models are object-oriented, and inherit from a base class :obj:`GibbsExcess <thermo.activity.GibbsExcess>` that provides many common methods. A further dummy class that predicts zero excess Gibbs energy and activity coefficients of 1 is available as :obj:`IdealSolution <thermo.activity.IdealSolution>`.

The excess Gibbs energy model is typically fairly simple. A number of derivatives are needed to calculate other properties like activity coefficient so those expressions can seem more complicated than the model really is. In the literature it is common for a model to be shown directly in activity coefficient form without discussion of the Gibbs excess energy model. To illustrate the difference, here is the :obj:`NRTL <thermo.nrtl.NRTL>` model Gibbs energy expression and its activity coefficient model:

.. math::
    g^E = RT\sum_i x_i \frac{\sum_j \tau_{ji} G_{ji} x_j}
    {\sum_j G_{ji}x_j}

.. math::
    \ln(\gamma_i)=\frac{\displaystyle\sum_{j=1}^{n}{x_{j}\tau_{ji}G_{ji}}}
    {\displaystyle\sum_{k=1}^{n}{x_{k}G_{ki}}}+\sum_{j=1}^{n}
    {\frac{x_{j}G_{ij}}{\displaystyle\sum_{k=1}^{n}{x_{k}G_{kj}}}}
    {\left ({\tau_{ij}-\frac{\displaystyle\sum_{m=1}^{n}{x_{m}\tau_{mj}
    G_{mj}}}{\displaystyle\sum_{k=1}^{n}{x_{k}G_{kj}}}}\right )}

The models :obj:`NRTL <thermo.nrtl.NRTL>`, :obj:`Wilson <thermo.wilson.Wilson>`, and :obj:`UNIQUAC <thermo.uniquac.UNIQUAC>` are the most commonly used. Each of them is regression-based - all coefficients must be found in the literature or regressed yourself. Each of these models has extensive temperature dependence parameters in addition to the composition dependence. The temperature dependencies implemented should allow parameters from most other sources to be used here with them.

The model :obj:`RegularSolution <thermo.regular_solution.RegularSolution>` is based on the concept of a :obj:`solubility parameter <chemicals.solubility.solubility_parameter>`; with liquid molar volumes and solubility parameters it is a predictive model. It does not show temperature dependence. Additional regression coefficients can be used with that model also.

The :obj:`UNIFAC <thermo.unifac.UNIFAC>` model is a predictive group-contribution scheme. 
In it, each molecule is fragmented into different sections. These sections have interaction parameters with other sections. Usually the fragmentation is not done by hand. One online tool for doing this is the `DDBST Online Group Assignment Tool <http://www.ddbst.com/unifacga.html>`_.

Object Structure
----------------
The :obj:`GibbsExcess <thermo.activity.GibbsExcess>` object doesn't know anything about phase equilibria, vapor pressure, or flash routines; it is limited in scope to dealing with excess Gibbs energy. Because of that modularity, an initialized :obj:`GibbsExcess <thermo.activity.GibbsExcess>` object is designed to be passed in an argument to a cubic equations of state that use excess Gibbs energy such as :obj:`PSRK <thermo.eos_mix.PSRK>`.

The other place these objects are used are in :obj:`GibbsExcessLiquid <thermo.phases.GibbsExcessLiquid>` objects, which brings the pieces together to construct a thermodynamically (mostly) consistent phase that the :obj:`flash algorithms <thermo.flash.Flash>` can work with.

This modularity allows new Gibbs excess models to be written and used anywhere - so the  :obj:`PSRK <thermo.eos_mix.PSRK>` model will happily allow a UNIFAC object configured like VTPR.

UNIFAC Example
--------------

The UNIFAC model is a group contribution based predictive model that is works using "fragmentations" of each molecule into a number of different "groups" and their "counts", 

The DDBST has published numerous sample problems using UNIFAC; a simple
binary system from example P05.22a in [2]_ with n-hexane and butanone-2
is shown below:

>>> from thermo.unifac import UFIP, UFSG, UNIFAC
>>> GE = UNIFAC.from_subgroups(chemgroups=[{1:2, 2:4}, {1:1, 2:1, 18:1}], T=60+273.15, xs=[0.5, 0.5], version=0, interaction_data=UFIP, subgroups=UFSG)

The solution given by the `DDBST <http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/05.22a%20VLE%20of%20Hexane-Butanone-2%20Via%20UNIFAC%20-%20Step%20by%20Step.xps>`_ has the activity coefficient values [1.428, 1.365], which match those calculated by the UNIFAC object:

>>> GE.gammas()
[1.4276025835, 1.3646545010]

Many other properties are also implemented, a few of which are shown below:

>>> GE.GE(), GE.dGE_dT(), GE.d2GE_dT2()
(923.641197, 0.206721488, -0.00380070204)
>>> GE.HE(), GE.SE(), GE.dHE_dT(), GE.dSE_dT()
(854.77193363, -0.2067214889, 1.266203886, 0.0038007020460)


Note that the :obj:`UFIP <thermo.unifac.UFIP>` and :obj:`UFSG <thermo.unifac.UFSG>` variables contain the actual interaction parameters;
none are hardcoded with the class, so the class could be used for regression. The `version` parameter controls which variant of UNIFAC to
use, as there are quite a few. The different UNIFAC models implemented include original UNIFAC, Dortmund UNIFAC, PSRK, VTPR, Lyngby/Larsen, and UNIFAC KT.
Interaction parameters for all models are included as well, but the `version` argument is not connected to the data files.

For convenience, a number of molecule fragmentations are distributed with the UNIFAC code. All fragmentations were obtained through the DDBST online portal, where molecular structure files can be submitted. This has the advantage that what is submitted is unambiguous; there are no worries about CAS numbers like how graphite and diamond have a different CAS number while being the same element or Air having a CAS number despite being a mixture. Accordingly, The index in these distributed data files are InChI keys, which can be obtained from :obj:`chemicals.identifiers` or in various places online.

>>> import thermo.unifac
>>> thermo.unifac.load_group_assignments_DDBST()
>>> len(thermo.unifac.DDBST_UNIFAC_assignments)
28846
>>> len(thermo.unifac.DDBST_MODIFIED_UNIFAC_assignments)
29271
>>> len(thermo.unifac.DDBST_PSRK_assignments)
30034
>>> from chemicals import search_chemical
>>> search_chemical('toluene').InChI_key
'YXFVVABEGXRONW-UHFFFAOYSA-N'
>>> thermo.unifac.DDBST_MODIFIED_UNIFAC_assignments['YXFVVABEGXRONW-UHFFFAOYSA-N']
{9: 5, 11: 1}

Please note that the identifying integer in these {group: count} elements are not necessarily the same in different UNIFAC versions, making them a royal pain.


Notes on Performance
--------------------
Initializing the object for the first time is a not a high performance operation as certain checks need to be done and data structures set up. Some pieces of the equations of the Gibbs excess model may depend only on temperature or composition, instead of depending on both. Each model implements the method :obj:`to_T_xs <thermo.unifac.UNIFAC.to_T_xs>` which should be used to create a new object at the new temperature and/or composition. The design of the object is to lazy-calculate properties, and to be immutable: calculations at new temperatures and compositions are done in a new object. 

Note also that the :obj:`__repr__ <thermo.activity.GibbsExcess.__repr__>` string for each model is designed to allow lossless reconstruction of the model. This is very useful when building test cases.

>>> GE.to_T_xs(T=400.0, xs=[.1, .9])
UNIFAC(T=400.0, xs=[0.1, 0.9], rs=[4.4998000000000005, 3.2479], qs=[3.856, 2.876], Qs=[0.848, 0.54, 1.488], vs=[[2.0, 1.0], [4.0, 1.0], [0.0, 1.0]], psi_a=[[0.0, 0.0, 476.4], [0.0, 0.0, 476.4], [26.76, 26.76, 0.0]], psi_b=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], psi_c=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], version=0)

When working with small numbers of components (5 or under), PyPy offers the best performance and using the model with Python lists as inputs is the fastest way to perform the calculations even in CPython.

If working with many components or if Numpy arrays are desired as inputs and outputs, numpy arrays can be provided as inputs. This will have a negative impact on performance unless the `numba` interface is used:

>>> import numpy as np
>>> import thermo.numba
>>> N = 3
>>> T = 25.0 + 273.15
>>> xs = np.array([0.7273, 0.0909, 0.1818])
>>> rs = np.array([.92, 2.1055, 3.1878])
>>> qs = np.array([1.4, 1.972, 2.4])
>>> tausA = tausC = tausD = tausE = tausF = np.array([[0.0]*N for i in range(N)])
>>> tausB = np.array([[0, -526.02, -309.64], [318.06, 0, 91.532], [-1325.1, -302.57, 0]])
>>> ABCDEF = (tausA, tausB, tausC, tausD, tausE, tausF)
>>> from thermo import UNIQUAC
>>> GE2 = UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, ABCDEF=ABCDEF)
>>> GE2.gammas()
array([ 1.57039333,  0.29482416, 18.11432905])

The `numba` interface will speed code up and allow calculations with dozens of components. The `numba` interface requires all inputs to be numpy arrays and all of its outputs are also numba arrays.

>>> GE3 = thermo.numba.UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, ABCDEF=ABCDEF)
>>> GE3.gammas()
array([ 1.57039333,  0.29482416, 18.11432905])

As an example of the performance benefits, a 200-component UNIFAC gamma calculation takes 10.6 ms in CPython and 318 µs when accelerated by Numba. In this case PyPy takes at 664 µs.

When the same benchmark is performed with 10 components, the calculation takes 387 µs in CPython, 88.6 µs with numba, and 36.2 µs with PyPy.

It can be quite important to use the :obj:`to_T_xs <thermo.unifac.UNIFAC.to_T_xs>` method re-use parts of the calculation; for UNIFAC, several terms depends only on temperature. If the 200 component calculation is repeated with those already calculated, the timings are 3.26 ms in CPython, 127 µs with numba, and 125 µs with PyPy.

Other features
--------------
The limiting infinite-dilution activity coefficients can be obtained with a call to :obj:`gammas_infinite_dilution <thermo.activity.GibbsExcess.gammas_infinite_dilution>`

>>> GE.gammas_infinite_dilution()
[3.5659995166, 4.32849696]

All activity coefficient models offer a :obj:`as_json <thermo.activity.GibbsExcess.as_json>` method and a :obj:`from_json <thermo.activity.GibbsExcess.from_json>` to serialize the object state for transport over a network, storing to disk, and passing data between processes. 

>>> from thermo import IdealSolution
>>> import json
>>> model = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
>>> json_view = model.as_json()
>>> json_str = json.dumps(json_view)
>>> model_copy = IdealSolution.from_json(json.loads(json_str))
>>> assert model_copy == model

Other json libraries can be used besides the standard json library by design.

Storing and recreating objects with Python's :py:func:`pickle.dumps` library is also tested; this can be faster than using JSON at the cost of being binary data.

All models have a :obj:`__hash__ <thermo.activity.GibbsExcess.exact_hash>` method that can be used to compare different models to see if they are absolutely identical (including which values have been calculated already).

They also have a :obj:`model_hash <thermo.activity.GibbsExcess.model_hash>` method that can be used to compare different models to see if they have identical model parameters.

They also have a :obj:`state_hash <thermo.activity.GibbsExcess.state_hash>` method that can be used to compare different models to see if they have identical temperature, composition, and model parameters. This is the __hash__ method.

Activity Coefficient Identities
-------------------------------

A set of useful equations are as follows. For more information, the reader is
directed to [1]_, [2]_, [3]_, [4]_, and [5]_; no one source contains all this
information.

.. math::
    h^E = -T \frac{\partial g^E}{\partial T} + g^E

.. math::
    \frac{\partial h^E}{\partial T} = -T \frac{\partial^2 g^E}
    {\partial T^2}

.. math::
    \frac{\partial h^E}{\partial x_i} = -T \frac{\partial^2 g^E}
    {\partial T \partial x_i} + \frac{\partial g^E}{\partial x_i}

.. math::
    s^E = \frac{h^E - g^E}{T}

.. math::
    \frac{\partial s^E}{\partial T} = \frac{1}{T}
    \left(\frac{-\partial g^E}{\partial T} + \frac{\partial h^E}{\partial T}
    - \frac{(G + H)}{T}\right)

.. math::
    \frac{\partial S^E}{\partial x_i} = \frac{1}{T}\left( \frac{\partial h^E}
    {\partial x_i} - \frac{\partial g^E}{\partial x_i}\right)

.. math::
    \frac{\partial \gamma_i}{\partial n_i} = \gamma_i
    \left(\frac{\frac{\partial^2 G^E}{\partial x_i \partial x_j}}{RT}\right)

.. math::
    \frac{\partial \gamma_i}{\partial T} =
    \left(\frac{\frac{\partial^2 n G^E}{\partial T \partial n_i}}{RT} -
    \frac{{\frac{\partial n_i G^E}{\partial n_i }}}{RT^2}\right)
     \exp\left(\frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}\right)



References
----------
.. [1] Poling, Bruce E., John M. Prausnitz, and John P. O’Connell. The
   Properties of Gases and Liquids. 5th edition. New York: McGraw-Hill
   Professional, 2000.
.. [2] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
   Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
.. [3] Nevers, Noel de. Physical and Chemical Equilibrium for Chemical 
   Engineers. 2nd edition. Wiley, 2012.
.. [4] Elliott, J., and Carl Lira. Introductory Chemical Engineering 
   Thermodynamics. 2nd edition. Upper Saddle River, NJ: Prentice Hall, 2012.
.. [5] Walas, Dr Stanley M. Phase Equilibria in Chemical Engineering. 
   Butterworth-Heinemann, 1985.



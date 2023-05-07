'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

.. contents:: :local:

The phases subpackage exposes classes that represent the state of single
phase mixture, including the composition, temperature, pressure, enthalpy,
and entropy. Phase objects are immutable and know nothing about bulk properties
or transport properties. The goal is for each phase to be able to compute all
of its thermodynamic properties, including volume-based ones.
Use settings to handle different assumptions.


Base Class
==========

.. autoclass:: Phase
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members:
    :special-members: __hash__, __eq__, __repr__

Ideal Gas Equation of State
===========================

.. autoclass:: IdealGas
   :show-inheritance:
   :members: dlnphis_dP, dlnphis_dT, dphis_dP, dphis_dT, phis, lnphis, fugacities,
             H, S, Cp, dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, dH_dP,
             dS_dT, dS_dP, d2H_dT2, d2H_dP2, d2S_dP2, dH_dT_V, dH_dP_V, dH_dV_T,
             dH_dV_P, dS_dT_V, dS_dP_V, __repr__

Cubic Equations of State
========================

Gas Phases
----------
.. autoclass:: CEOSGas
   :show-inheritance:
   :members: to_TP_zs, V_iter, H, S, Cp, Cv, dP_dT, dP_dV,
             d2P_dT2, d2P_dV2, d2P_dTdV,
             dS_dT_V,
             lnphis, dlnphis_dT, dlnphis_dP, __repr__
   :exclude-members: d2H_dP2, d2H_dT2, d2S_dP2, dH_dP, dH_dP_V, dH_dT_V, dH_dV_P, dH_dV_T, dS_dP, dS_dP_V, dS_dT

Liquid Phases
-------------
.. autoclass:: CEOSLiquid
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__


Virial Equations of State
=========================

Gas Phase Object
----------------
.. autoclass:: VirialGas
   :show-inheritance:
   :members: V, dP_dT, dP_dV, d2P_dTdV, d2P_dV2, d2P_dT2,
             H_dep, dH_dep_dT, S_dep, dS_dep_dT,
             B, dB_dT, d2B_dT2, d3B_dT3,
             C, dC_dT, d2C_dT2, d3C_dT3,
             dB_dzs, d2B_dTdzs, d2B_dzizjs,
             dC_dzs, d2C_dTdzs, d2C_dzizjs,
             d2V_dzizjs, dV_dzs, dG_dep_dzs,
             lnphis, __repr__

Corresponding States Virial Model
---------------------------------
.. autoclass:: VirialCSP
   :show-inheritance:
   :members: to, __repr__, B_pures, dB_dT_pures, d2B_dT2_pures, d3B_dT3_pures,
             B_interactions, dB_dT_interactions, d2B_dT2_interactions, d3B_dT3_interactions,
             C_pures, dC_dT_pures, d2C_dT2_pures, d3C_dT3_pures,
             C_interactions, dC_dT_interactions, d2C_dT2_interactions, d3C_dT3_interactions,


Activity Based Liquids
======================
.. autoclass:: GibbsExcessLiquid
   :show-inheritance:
   :members: __init__, H, S, Cp, gammas, Poyntings, phis_sat, lnHenry_matrix, dlnHenry_matrix_dT, d2lnHenry_matrix_dT2
   :exclude-members: __init__


Fundamental Equations of State
==============================
`HelmholtzEOS` is the base class for all Helmholtz energy fundamental equations
of state.

.. autoclass:: HelmholtzEOS
   :show-inheritance:
   :members: to_TP_zs, V_iter, H, S, Cp, Cv, dP_dT, dP_dV,
             d2P_dT2, d2P_dV2, d2P_dTdV, dH_dP, dS_dP,
             lnphis, __repr__
   :exclude-members: dH_dP_V, dH_dT_V, dH_dV_P, dH_dV_T, dS_dP_V, dS_dT, dS_dT_V, dlnphis_dP, dlnphis_dT

`IAPWS95` is the base class for the IAPWS-95 formulation for water;
`IAPWS95Gas` and `IAPWS95Liquid` are the gas and liquid sub-phases
respectively.

.. autoclass:: IAPWS95
   :show-inheritance:
   :members: T_MAX_FIXED, T_MIN_FIXED, mu, k

.. autoclass:: IAPWS95Gas
   :show-inheritance:
   :members: force_phase

.. autoclass:: IAPWS95Liquid
   :show-inheritance:
   :members: force_phase


`DryAirLemmon` is an implementation of thermophysical properties of air by
Lemmon (2000).

.. autoclass:: DryAirLemmon
   :show-inheritance:
   :members: T_MAX_FIXED, T_MIN_FIXED, mu, k


CoolProp Wrapper
================
.. autoclass:: CoolPropGas
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

.. autoclass:: CoolPropLiquid
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

'''

# # Not ready to be documented or exposed
# Petroleum Specific Phases
# =========================
# .. autoclass:: GraysonStreed
#    :show-inheritance:
#    :members: __init__
#    :exclude-members: __init__

# .. autoclass:: ChaoSeader
#    :show-inheritance:
#    :members: __init__
#    :exclude-members: __init__

# Solids Phases
# =============
# .. autoclass:: GibbsExcessSolid
#    :show-inheritance:
#    :members: __init__
#    :exclude-members: __init__



from thermo.phases import( phase, ideal_gas, ceos, gibbs_excess, helmholtz_eos,
                          air_phase, iapws_phase, coolprop_phase, virial_phase,
                          petroleum, combined)
from thermo.phases.phase import *
from thermo.phases.ideal_gas import *
from thermo.phases.ceos import *
from thermo.phases.gibbs_excess import *
from thermo.phases.helmholtz_eos import *
from thermo.phases.iapws_phase import *
from thermo.phases.air_phase import *
from thermo.phases.coolprop_phase import *
from thermo.phases.virial_phase import *
from thermo.phases.petroleum import *
from thermo.phases.combined import *

__all__ = (phase.__all__ + ideal_gas.__all__ + ceos.__all__
           + gibbs_excess.__all__ + air_phase.__all__ + helmholtz_eos.__all__
           + iapws_phase.__all__ + coolprop_phase.__all__
           + virial_phase.__all__ + petroleum.__all__ + combined.__all__)

gas_phases = (
    IdealGas,
    CEOSGas,
    CoolPropGas,
    IAPWS95Gas,
    VirialGas,
    HumidAirRP1485,
    DryAirLemmon
)
liquid_phases = (
    CEOSLiquid,
    GibbsExcessLiquid,
    CoolPropLiquid,
    IAPWS95Liquid
)
solid_phases = (
    GibbsExcessSolid,
)
many_phases = (
    IAPWS95,
    IAPWS97,
    CoolPropPhase,
)
all_phases = gas_phases + liquid_phases + solid_phases + many_phases
phase_full_path_dict = {c.__full_path__: c for c in all_phases}

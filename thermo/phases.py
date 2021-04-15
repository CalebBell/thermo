# -*- coding: utf-8 -*-
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

Activity Based Liquids
======================
.. autoclass:: GibbsExcessLiquid
   :show-inheritance:
   :members: __init__, H, S, Cp, gammas, Poyntings, phis_sat
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

.. autoclass:: IAPWS95
   :show-inheritance:
   :members: T_MAX_FIXED, T_MIN_FIXED, mu, k

.. autoclass:: IAPWS95Gas
   :show-inheritance:
   :members: force_phase

.. autoclass:: IAPWS95Liquid
   :show-inheritance:
   :members: force_phase


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
from __future__ import division

'''
# Not ready to be documented or exposed
Petroleum Specific Phases
=========================
.. autoclass:: GraysonStreed
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

.. autoclass:: ChaoSeader
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Solids Phases
=============
.. autoclass:: GibbsExcessSolid
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Virial Equations of State
=========================

.. autoclass:: VirialGas
   :show-inheritance:
   :members:

'''
__all__ = ['GibbsExcessLiquid', 'GibbsExcessSolid', 'Phase', 'CEOSLiquid', 'CEOSGas', 'IdealGas', 'IAPWS97', 'HelmholtzEOS',
           'IAPWS95', 'IAPWS95Gas', 'IAPWS95Liquid', 'DryAirLemmon', 'VirialGas',
           'gas_phases', 'liquid_phases', 'solid_phases', 'CombinedPhase', 'CoolPropPhase', 'CoolPropLiquid', 'CoolPropGas', 'INCOMPRESSIBLE_CONST',
           'HumidAirRP1485',
           'derivatives_thermodynamic', 'derivatives_thermodynamic_mass', 'derivatives_jacobian',

           'VirialCorrelationsPitzerCurl', # For testing - try to get rid of
           ]

import sys, os
from math import isinf, isnan, sqrt
from fluids.constants import R, R_inv
import fluids.constants
from fluids.numerics import (horner, horner_and_der, horner_and_der2, horner_log, jacobian, derivative,
                             poly_fit_integral_value, poly_fit_integral_over_T_value,
                             evaluate_linear_fits, evaluate_linear_fits_d,
                             evaluate_linear_fits_d2, quadratic_from_f_ders,
                             newton_system, trunc_log, trunc_exp, newton)
from chemicals.utils import (log, log10, exp, Cp_minus_Cv, phase_identification_parameter,
                          isothermal_compressibility, isobaric_expansion, property_mass_to_molar,
                          Joule_Thomson, speed_of_sound, dxs_to_dns, dns_to_dn_partials,
                          normalize, hash_any_primitive, rho_to_Vm, Vm_to_rho)
from random import randint
from collections import OrderedDict
from chemicals.iapws import *
from chemicals.air import *
from chemicals.viscosity import mu_IAPWS, mu_air_lemmon
from chemicals.thermal_conductivity import k_IAPWS
import chemicals.iapws
from chemicals.iapws import iapws95_d3Ar_ddelta2dtau, iapws95_d3Ar_ddeltadtau2

from thermo.serialize import arrays_to_lists
from thermo.coolprop import has_CoolProp
from thermo.eos import GCEOS, eos_full_path_dict
from thermo.eos_mix import IGMIX, GCEOSMIX, eos_mix_full_path_dict, eos_mix_full_path_reverse_dict
from thermo.eos_mix_methods import PR_lnphis_fastest

from thermo.activity import GibbsExcess, IdealSolution
from thermo.wilson import Wilson
from thermo.unifac import UNIFAC
from thermo.regular_solution import RegularSolution
from thermo.uniquac import UNIQUAC

from thermo.chemical_package import iapws_correlations

from thermo.utils import POLY_FIT
from thermo.heat_capacity import HeatCapacityGas, HeatCapacityLiquid
from thermo.volume import VolumeLiquid, VolumeSolid
from thermo.vapor_pressure import VaporPressure, SublimationPressure
from thermo.phase_change import EnthalpyVaporization, EnthalpySublimation


R2 = R*R
'''
All phase objects are immutable.

Goal is for each phase to be able to compute all of its thermodynamic properties.
This includes volume-based ones. Use settings to handle different; do not worry
about derivatives being calculated correctly.

Phases know nothing about bulk properties.
Phases know nothing about transport properties.

For enthalpy, need to support with ideal gas heat of formation as a separate
enthalpy calculation.
'''

#PY2 = int(sys.version[0]) == 2

SORTED_DICT = sys.version_info >= (3, 6)
INCOMPRESSIBLE_CONST = 1e30

activity_pointer_reference_dicts = {'thermo.activity.IdealSolution': IdealSolution,
                                    'thermo.wilson.Wilson': Wilson,
                                    'thermo.unifac.UNIFAC': UNIFAC,
                                    'thermo.regular_solution.RegularSolution': RegularSolution,
                                    'thermo.uniquac.UNIQUAC': UNIQUAC,
                                    }
activity_reference_pointer_dicts = {v: k for k, v in activity_pointer_reference_dicts.items()}

object_lookups = activity_pointer_reference_dicts.copy()
object_lookups.update(eos_mix_full_path_dict)
object_lookups.update(eos_full_path_dict)



class Phase(object):
    '''
    '''

    '''
    For basic functionality, a subclass should implement:

    H, S, Cp



    dP_dT
    dP_dV
    d2P_dT2
    d2P_dV2
    d2P_dTdV

    Additional functionality is enabled by the methods:

    dH_dP dS_dT dS_dP
    d2H_dT2 d2H_dP2 d2S_dP2

    dH_dT_V dH_dP_V dH_dV_T dH_dV_P
    dS_dT_V dS_dP_V

    d2H_dTdP d2H_dT2_V
    d2P_dTdP d2P_dVdP d2P_dVdT_TP d2P_dT2_PV

    Optional ones for speed:: Cv


    '''

    R = fluids.constants.R
    R_inv = 1.0/R

    is_solid = False

    ideal_gas_basis = False # Parameter fot has the same ideal gas Cp
    T_REF_IG = 298.15
    T_REF_IG_INV = 1.0/T_REF_IG
    '''The numerical inverse of :obj:`T_REF_IG`, stored to save a division.
    '''
    P_REF_IG = 101325.
    P_REF_IG_INV = 1.0/P_REF_IG
    LOG_P_REF_IG = log(P_REF_IG)

    T_MAX_FIXED = 10000.0
    T_MIN_FIXED = 1e-3

    P_MAX_FIXED = 1e9
    P_MIN_FIXED = 1e-2 # 1e-3 was so low issues happened in the root stuff, could not be fixed

    V_MIN_FIXED = 1e-9 # m^3/mol
    V_MAX_FIXED = 1e9 # m^#/mol

    force_phase = None
    '''Attribute which can be set to a global Phase object to force the phases
    identification routines to label it a certain phase. Accepts values of ('g', 'l', 's').'''

    _Psats_data = None
    _Cpgs_data = None
    Psats_poly_fit = False
    Cpgs_poly_fit = False
    composition_independent = False
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    scalar  = True

    pure_references = tuple()
    '''Tuple of attribute names which hold lists of :obj:`thermo.utils.TDependentProperty`
    or :obj:`thermo.utils.TPDependentProperty` instances.'''

    pure_reference_types = tuple()
    '''Tuple of types of :obj:`thermo.utils.TDependentProperty`
    or :obj:`thermo.utils.TPDependentProperty` corresponding to `pure_references`.'''

    obj_references = tuple()
    '''Tuple of object instances which should be stored as json using their own
    as_json method.
    '''
    pointer_references = tuple()
    '''Tuple of attributes which should be stored by converting them to
    a string, and then they will be looked up in their corresponding
    `pointer_reference_dicts` entry.
    '''
    pointer_reference_dicts = tuple()
    '''Tuple of dictionaries for string -> object
    '''
    reference_pointer_dicts = tuple()
    '''Tuple of dictionaries for object -> string
    '''



    def __str__(self):
        s =  '<%s, ' %(self.__class__.__name__)
        try:
            s += 'T=%g K, P=%g Pa' %(self.T, self.P)
        except:
            pass
        s += '>'
        return s

    def as_json(self):
        r'''Method to create a JSON-friendly serialization of the phase
        which can be stored, and reloaded later.

        Returns
        -------
        json_repr : dict
            JSON-friendly representation, [-]

        Notes
        -----

        Examples
        --------
        >>> import json
        >>> phase = IAPWS95Liquid(T=300, P=1e5, zs=[1])
        >>> new_phase = Phase.from_json(json.loads(json.dumps(phase.as_json())))
        >>> assert phase == new_phase
        '''
        d = self.__dict__.copy()
        if not self.scalar:
            d = serialize.arrays_to_lists(d)
        for obj_name in self.obj_references:
            o = d[obj_name]
            if type(o) is list:
                d[obj_name] = [v.as_json() for v in o]
            else:
                d[obj_name] = o.as_json()
        for prop_name in self.pure_references:
            # Known issue: references to other properties
            # Needs special fixing - maybe a function
            l = d[prop_name]
            if l:
                d[prop_name] = [v.as_json() for v in l]
        for ref_name, ref_lookup in zip(self.pointer_references, self.reference_pointer_dicts):
            d[ref_name] = ref_lookup[d[ref_name]]
        d["py/object"] = self.__full_path__
        d['json_version'] = 1
        return d



    @classmethod
    def from_json(cls, json_repr):
        r'''Method to create a phase from a JSON
        serialization of another phase.

        Parameters
        ----------
        json_repr : dict
            JSON-friendly representation, [-]

        Returns
        -------
        phase : :obj:`Phase`
            Newly created phase object from the json serialization, [-]

        Notes
        -----
        It is important that the input string be in the same format as that
        created by :obj:`Phase.as_json`.

        Examples
        --------
        '''
        d = json_repr
        phase_name = d['py/object']
        del d['py/object']
        del d['json_version']
        phase = phase_full_path_dict[phase_name]
        new = phase.__new__(phase)

        for obj_name, obj_cls in zip(new.pure_references, new.pure_reference_types):
            l = d[obj_name]
            if l:
                for i, v in enumerate(l):
                    l[i] = obj_cls.from_json(v)

        for obj_name in new.obj_references:
            o = d[obj_name]
            if type(o) is list:
                d[obj_name] = [object_lookups[v['py/object']].from_json(v) for v in o]
            else:
                obj_cls = object_lookups[o['py/object']]
                d[obj_name] = obj_cls.from_json(o)

        for ref_name, ref_lookup in zip(new.pointer_references, new.pointer_reference_dicts):
            d[ref_name] = ref_lookup[d[ref_name]]

        new.__dict__ = d
        return new

    def __hash__(self):
        r'''Method to calculate and return a hash representing the exact state
        of the object.

        Returns
        -------
        hash : int
            Hash of the object, [-]
        '''
        # Ensure the hash is set so it is always part of the object hash
        self.model_hash(False)
        self.model_hash(True)
        self.state_hash()
        d = self.__dict__

        ans = hash_any_primitive((self.__class__.__name__, d))
        return ans

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def state_hash(self):
        r'''Basic method to calculate a hash of the state of the phase and its
        model parameters.

        Note that the hashes should only be compared on the same system running
        in the same process!

        Returns
        -------
        state_hash : int
            Hash of the object's model parameters and state, [-]
        '''
        return hash_any_primitive((self.model_hash(), self.T, self.P, self.V(), self.zs))

    def model_hash(self, ignore_phase=False):
        r'''Method to compute a hash of a phase.

        Parameters
        ----------
        ignore_phase : bool
            Whether or not to include the specifc class of the model in the
            hash

        Returns
        -------
        hash : int
            Hash representing the settings of the phase; phases with all
            identical model parameters should have the same hash.
        '''
        if ignore_phase:
            try:
                return self._model_hash_ignore_phase
            except AttributeError:
                pass
        else:
            try:
                return self._model_hash
            except AttributeError:
                pass
        # Note: not all attributes are in __dict__, must use getattr
        to_hash = [getattr(self, v) for v in self.model_attributes]
        self._model_hash_ignore_phase = h = hash_any_primitive(to_hash)
        self._model_hash = hash((self.__class__.__name__, h))
        if ignore_phase:
            return self._model_hash_ignore_phase
        else:
            return self._model_hash

    def value(self, name):
        r'''Method to retrieve a property from a string. This more or less
        wraps `getattr`.

        `name` could be a python property like 'Tms' or a callable method
        like 'H'.

        Parameters
        ----------
        name : str
            String representing the property, [-]

        Returns
        -------
        value : various
            Value specified, [various]

        Notes
        -----
        '''
        if name in ('beta_mass',):
            return self.result.value(name, self)

        v = getattr(self, name)
        try:
            v = v()
        except:
            pass
        return v

    ### Methods that should be implemented by subclasses

    def to_TP_zs(self, T, P, zs):
        r'''Method to create a new Phase object with the same constants as the
        existing Phase but at a different `T` and `P`.

        Parameters
        ----------
        zs : list[float]
            Molar composition of the new phase, [-]
        T : float
            Temperature of the new phase, [K]
        P : float
            Pressure of the new phase, [Pa]

        Returns
        -------
        new_phase : Phase
            New phase at the specified conditions, [-]

        Notes
        -----
        This method is marginally faster than :obj:`Phase.to` as it does not
        need to check what the inputs are.

        Examples
        --------

        >>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=[])
        >>> phase.to_TP_zs(T=1e5, P=1e3, zs=[.5, .5])
        IdealGas(HeatCapacityGases=[], T=100000.0, P=1000.0, zs=[0.5, 0.5])
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def to(self, zs, T=None, P=None, V=None):
        r'''Method to create a new Phase object with the same constants as the
        existing Phase but at different conditions. Mole fractions `zs` are
        always required and any two of `T`, `P`, and `V` are required.

        Parameters
        ----------
        zs : list[float]
            Molar composition of the new phase, [-]
        T : float, optional
            Temperature of the new phase, [K]
        P : float, optional
            Pressure of the new phase, [Pa]
        V : float, optional
            Molar volume of the new phase, [m^3/mol]

        Returns
        -------
        new_phase : Phase
            New phase at the specified conditions, [-]

        Notes
        -----

        Examples
        --------

        These sample cases illustrate the three combinations of inputs.
        Note that some thermodynamic models may have multiple solutions for
        some inputs!

        >>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=[])
        >>> phase.to(T=1e5, P=1e3, zs=[.5, .5])
        IdealGas(HeatCapacityGases=[], T=100000.0, P=1000.0, zs=[0.5, 0.5])
        >>> phase.to(V=1e-4, P=1e3, zs=[.1, .9])
        IdealGas(HeatCapacityGases=[], T=0.012027235504, P=1000.0, zs=[0.1, 0.9])
        >>> phase.to(T=1e5, V=1e12, zs=[.2, .8])
        IdealGas(HeatCapacityGases=[], T=100000.0, P=8.31446261e-07, zs=[0.2, 0.8])

        '''
        raise NotImplementedError("Must be implemented by subphases")

    def V(self):
        r'''Method to return the molar volume of the phase.

        Returns
        -------
        V : float
            Molar volume, [m^3/mol]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def dP_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        pressure of the phase.

        Returns
        -------
        dP_dT : float
            First temperature derivative of pressure, [Pa/K]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def dP_dV(self):
        r'''Method to calculate and return the first volume derivative of
        pressure of the phase.

        Returns
        -------
        dP_dV : float
            First volume derivative of pressure, [Pa*mol/m^3]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def d2P_dT2(self):
        r'''Method to calculate and return the second temperature derivative of
        pressure of the phase.

        Returns
        -------
        d2P_dT2 : float
            Second temperature derivative of pressure, [Pa/K^2]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def d2P_dV2(self):
        r'''Method to calculate and return the second volume derivative of
        pressure of the phase.

        Returns
        -------
        d2P_dV2 : float
            Second volume derivative of pressure, [Pa*mol^2/m^6]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def d2P_dTdV(self):
        r'''Method to calculate and return the second derivative of
        pressure with respect to temperature and volume of the phase.

        Returns
        -------
        d2P_dTdV : float
            Second volume derivative of pressure, [mol*Pa^2/(J*K)]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def lnphis(self):
        r'''Method to calculate and return the log of fugacity coefficients of
        each component in the phase.

        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def dlnphis_dT(self):
        r'''Method to calculate and return the temperature derivative of the
        log of fugacity coefficients of each component in the phase.

        Returns
        -------
        dlnphis_dT : list[float]
            First temperature derivative of log fugacity coefficients, [1/K]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def dlnphis_dP(self):
        r'''Method to calculate and return the pressure derivative of the
        log of fugacity coefficients of each component in the phase.

        Returns
        -------
        dlnphis_dP : list[float]
            First pressure derivative of log fugacity coefficients, [1/Pa]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def H(self):
        r'''Method to calculate and return the enthalpy of the phase.
        The reference state for most subclasses is an ideal-gas enthalpy of
        zero at 298.15 K and 101325 Pa.

        Returns
        -------
        H : float
            Molar enthalpy, [J/(mol)]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def S(self):
        r'''Method to calculate and return the entropy of the phase.
        The reference state for most subclasses is an ideal-gas entropy of
        zero at 298.15 K and 101325 Pa.

        Returns
        -------
        S : float
            Molar entropy, [J/(mol*K)]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def Cp(self):
        r'''Method to calculate and return the constant-pressure heat capacity
        of the phase.

        Returns
        -------
        Cp : float
            Molar heat capacity, [J/(mol*K)]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    ### Benchmarking methods
    def _compute_main_properties(self):
        '''Method which computes some basic properties. For benchmarking;
        accepts no arguments and returns nothing. A timer should be used
        outside of this method.
        '''
        self.H()
        self.S()
        self.Cp()
        self.Cv()
        self.dP_dT()
        self.dP_dV()
        self.d2P_dT2()
        self.d2P_dV2()
        self.d2P_dTdV()
        self.PIP()

    ### Consistency Checks
    def S_phi_consistency(self):
        r'''Method to calculate and return a consistency check between ideal
        gas entropy behavior, and the fugacity coefficients and their
        temperature derivatives.

        .. math::
             S = S^{ig} - \sum_{i} z_i R\left(\ln \phi_i +  T \frac{\partial \ln
             \phi_i}{\partial T}\right)

        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - S^{\text{from phi}}/S^\text{implemented}|`, [-]
        '''
        # From coco
        S0 = self.S_ideal_gas()
        lnphis = self.lnphis()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        for i in range(len(zs)):
            S0 -= zs[i]*(R*lnphis[i] + R*T*dlnphis_dT[i])
        return abs(1.0 - S0/self.S())


    def H_phi_consistency(self):
        r'''Method to calculate and return a consistency check between ideal
        gas enthalpy behavior, and the fugacity coefficients and their
        temperature derivatives.

        .. math::
             H^{\text{from phi}} = H^{ig} - RT^2\sum_i z_i \frac{\partial \ln
             \phi_i}{\partial T}

        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - H^{\text{from phi}}/H^\text{implemented}|`, [-]
        '''
        return abs(1.0 - self.H_from_phi()/self.H())

    def G_dep_phi_consistency(self):
        r'''Method to calculate and return a consistency check between
        departure Gibbs free energy, and the fugacity coefficients.

        .. math::
             G^{\text{from phi}}_{dep} = RT\sum_i z_i \phi_i

        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - G^{\text{from phi}}_{dep}/G^\text{implemented}_{dep}|`, [-]
        '''
        # Chapter 2 equation 31 Michaelson
        zs, T = self.zs, self.T
        G_dep_RT = 0.0
        lnphis = self.lnphis()
        G_dep_RT = sum(zs[i]*lnphis[i] for i in range(self.N))
        G_dep = G_dep_RT*R*T
        return abs(1.0 - G_dep/self.G_dep())

    def H_dep_phi_consistency(self):
        r'''Method to calculate and return a consistency check between
        departure enthalpy, and the fugacity coefficients' temperature
        derivatives.

        .. math::
             H^{\text{from phi}}_{dep} = -RT^2\sum_i z_i \frac{\partial \ln
             \phi_i}{\partial T}

        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - H^{\text{from phi}}_{dep}/H^\text{implemented}_{dep}|`, [-]
        '''
        H_dep_RT2 = 0.0
        dlnphis_dTs = self.dlnphis_dT()
        zs, T = self.zs, self.T
        H_dep_RT2 = sum(zs[i]*dlnphis_dTs[i] for i in range(len(zs)))
        H_dep_recalc = -H_dep_RT2*R*T*T
        H_dep = self.H_dep()
        return abs(1.0 - H_dep/H_dep_recalc)

    def S_dep_phi_consistency(self):
        r'''Method to calculate and return a consistency check between ideal
        gas entropy behavior, and the fugacity coefficients and their
        temperature derivatives.

        .. math::
             S_{dep}^{\text{from phi}} = - \sum_{i} z_i R\left(\ln \phi_i
             +  T \frac{\partial \ln \phi_i}{\partial T}\right)

        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - S^{\text{from phi}}_{dep}/S^\text{implemented}_{dep}|`, [-]
        '''
        # From coco
        lnphis = self.lnphis()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        S_dep = 0.0
        for i in range(len(zs)):
            S_dep -= zs[i]*(R*lnphis[i] + R*T*dlnphis_dT[i])
        return abs(1.0 - S_dep/self.S_dep())

    def V_phi_consistency(self):
        r'''Method to calculate and return a consistency check between
        molar volume, and the fugacity coefficients' pressures
        derivatives.

        .. math::
            V^{\text{from phi P der}} = \left(\left(\sum_i z_i \frac{\partial \ln
             \phi_i}{\partial P}\right)P + 1\right)RT/P


        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - V^{\text{from phi P der}}/V^\text{implemented}|`, [-]
        '''
        zs, P = self.zs, self.P
        dlnphis_dP = self.dlnphis_dP()
        lhs = sum(zs[i]*dlnphis_dP[i] for i in range(self.N))
        Z_calc = lhs*P + 1.0
        V_calc = Z_calc*self.R*self.T/P
        V = self.V()
        return abs(1.0 - V_calc/V)

    def H_from_phi(self):
        r'''Method to calculate and return the enthalpy of the fluid as
        calculated from the ideal-gas enthalpy and the the fugacity
        coefficients' temperature derivatives.

        .. math::
             H^{\text{from phi}} = H^{ig} - RT^2\sum_i z_i \frac{\partial \ln
             \phi_i}{\partial T}

        Returns
        -------
        H : float
            Enthalpy as calculated from fugacity coefficient temperature
            derivatives [J/mol]
        '''
        H0 = self.H_ideal_gas()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        for i in range(self.N):
            H0 -= R*T*T*zs[i]*dlnphis_dT[i]
        return H0

    def S_from_phi(self):
        r'''Method to calculate and return the entropy of the fluid as
        calculated from the ideal-gas entropy and the the fugacity
        coefficients' temperature derivatives.

        .. math::
             S = S^{ig} - \sum_{i} z_i R\left(\ln \phi_i +  T \frac{\partial \ln
             \phi_i}{\partial T}\right)

        Returns
        -------
        S : float
            Entropy as calculated from fugacity coefficient temperature
            derivatives [J/(mol*K)]
        '''
        S0 = self.S_ideal_gas()
        lnphis = self.lnphis()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        for i in range(self.N):
            S0 -= zs[i]*(R*lnphis[i] + R*T*dlnphis_dT[i])
        return S0

    def V_from_phi(self):
        r'''Method to calculate and return the molar volume of the fluid as
        calculated from the pressure derivatives of fugacity coefficients.

        .. math::
            V^{\text{from phi P der}} = \left(\left(\sum_i z_i \frac{\partial \ln
             \phi_i}{\partial P}\right)P + 1\right)RT/P


        Returns
        -------
        V : float
            Molar volume, [m^3/mol]
        '''
        zs, P = self.zs, self.P
        dlnphis_dP = self.dlnphis_dP()
        obj = sum(zs[i]*dlnphis_dP[i] for i in range(self.N))
        Z = P*obj + 1.0
        return Z*self.R*self.T/P

    def G_min_criteria(self):
        r'''Method to calculate and return the Gibbs energy criteria required
        for comparing phase stability. This calculation can be faster
        than calculating the full Gibbs energy. For this comparison to work,
        all phases must use the ideal gas basis.

        .. math::
             G^{\text{criteria}} = G^{dep} + RT\sum_i z_i \ln z_i

        Returns
        -------
        G_crit : float
            Gibbs free energy like criteria [J/mol]
        '''
        # Definition implemented that does not use the H, or S ideal gas contribution
        # Allows for faster checking of which phase is at lowest G, but can only
        # be used when all models use an ideal gas basis
        zs = self.zs
        log_zs = self.log_zs()
        G_crit = 0.0
        for i in range(self.N):
            G_crit += zs[i]*log_zs[i]

        G_crit = G_crit*R*self.T + self.G_dep()
        return G_crit

    def lnphis_at_zs(self, zs):
        r'''Method to directly calculate the log fugacity coefficients at a
        different composition than the current phase.
        This is implemented to allow for the possibility of more direct
        calls to obtain fugacities than is possible with the phase interface.
        This base method simply creates a new phase, gets its log fugacity
        coefficients, and returns them.

        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
        '''
        return self.to_TP_zs(self.T, self.P, zs).lnphis()

    def fugacities_at_zs(self, zs):
        r'''Method to directly calculate the figacities at a
        different composition than the current phase.
        This is implemented to allow for the possibility of more direct
        calls to obtain fugacities than is possible with the phase interface.
        This base method simply creates a new phase, gets its log fugacity
        coefficients, exponentiates them, and multiplies them by `P` and
        compositions.

        Returns
        -------
        fugacities : list[float]
            Fugacities, [Pa]
        '''
        P = self.P
        lnphis = self.lnphis_at_zs(zs)
        return [P*zs[i]*trunc_exp(lnphis[i]) for i in range(len(zs))]

    def lnphi(self):
        r'''Method to calculate and return the log of fugacity coefficient of
        the phase; provided the phase is 1 component.

        Returns
        -------
        lnphi : list[float]
            Log fugacity coefficient, [-]
        '''
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.lnphis()[0]

    def phi(self):
        r'''Method to calculate and return the fugacity coefficient of
        the phase; provided the phase is 1 component.

        Returns
        -------
        phi : list[float]
            Fugacity coefficient, [-]
        '''
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.phis()[0]

    def fugacity(self):
        r'''Method to calculate and return the fugacity of
        the phase; provided the phase is 1 component.

        Returns
        -------
        fugacity : list[float]
            Fugacity, [Pa]
        '''
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.fugacities()[0]

    def dfugacity_dT(self):
        r'''Method to calculate and return the temperature derivative of
        fugacity of the phase; provided the phase is 1 component.

        Returns
        -------
        dfugacity_dT : list[float]
            Fugacity first temperature derivative, [Pa/K]
        '''
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.dfugacities_dT()[0]

    def dfugacity_dP(self):
        r'''Method to calculate and return the pressure derivative of
        fugacity of the phase; provided the phase is 1 component.

        Returns
        -------
        dfugacity_dP : list[float]
            Fugacity first pressure derivative, [-]
        '''
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.dfugacities_dP()[0]


    def fugacities(self):
        r'''Method to calculate and return the fugacities of the phase.

        .. math::
            f_i = P z_i \exp(\ln \phi_i)

        Returns
        -------
        fugacities : list[float]
            Fugacities, [Pa]
        '''
        P = self.P
        zs = self.zs
        lnphis = self.lnphis()
        return [P*zs[i]*trunc_exp(lnphis[i]) for i in range(len(zs))]

    def lnfugacities(self):
        r'''Method to calculate and return the log of fugacities of the phase.

        .. math::
            \ln f_i = \ln\left( P z_i \exp(\ln \phi_i)\right)
            = \ln(P) + \ln(z_i) + \ln \phi_i

        Returns
        -------
        lnfugacities : list[float]
            Log fugacities, [log(Pa)]
        '''
        P = self.P
        zs = self.zs
        lnphis = self.lnphis()
        logP = log(P)
        log_zs = self.log_zs()
        return [logP + log_zs[i] + lnphis[i] for i in range(self.N)]

    fugacities_lowest_Gibbs = fugacities

    def dfugacities_dT(self):
        r'''Method to calculate and return the temperature derivative of fugacities
        of the phase.

        .. math::
            \frac{\partial f_i}{\partial T} = P z_i \frac{\partial
            \ln \phi_i}{\partial T}

        Returns
        -------
        dfugacities_dT : list[float]
            Temperature derivative of fugacities of all components
            in the phase, [Pa/K]

        Notes
        -----
        '''
        dphis_dT = self.dphis_dT()
        P, zs = self.P, self.zs
        return [P*zs[i]*dphis_dT[i] for i in range(len(zs))]

    def lnphis_G_min(self):
        r'''Method to calculate and return the log fugacity coefficients of the
        phase. If the phase can have multiple solutions at its `T` and `P`,
        this method should return those with the lowest Gibbs energy. This
        needs to be implemented on phases with that criteria like cubic EOSs.

        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
        '''
        return self.lnphis()

    def phis(self):
        r'''Method to calculate and return the fugacity coefficients of the
        phase.

        .. math::
            \phi_i = \exp (\ln \phi_i)

        Returns
        -------
        phis : list[float]
            Fugacity coefficients, [-]
        '''
        return [trunc_exp(i) for i in self.lnphis()]

    def dphis_dT(self):
        r'''Method to calculate and return the temperature derivative of fugacity
        coefficients of the phase.

        .. math::
            \frac{\partial \phi_i}{\partial T} = \phi_i \frac{\partial
            \ln \phi_i}{\partial T}

        Returns
        -------
        dphis_dT : list[float]
            Temperature derivative of fugacity coefficients of all components
            in the phase, [1/K]

        Notes
        -----
        '''
        try:
            return self._dphis_dT
        except AttributeError:
            pass
        try:
            dlnphis_dT = self._dlnphis_dT
        except AttributeError:
            dlnphis_dT = self.dlnphis_dT()

        try:
            phis = self._phis
        except AttributeError:
            phis = self.phis()

        self._dphis_dT = [dlnphis_dT[i]*phis[i] for i in range(self.N)]
        return self._dphis_dT

    def dphis_dP(self):
        r'''Method to calculate and return the pressure derivative of fugacity
        coefficients of the phase.

        .. math::
            \frac{\partial \phi_i}{\partial P} = \phi_i \frac{\partial
            \ln \phi_i}{\partial P}

        Returns
        -------
        dphis_dP : list[float]
            Pressure derivative of fugacity coefficients of all components
            in the phase, [1/Pa]

        Notes
        -----
        '''
        try:
            return self._dphis_dP
        except AttributeError:
            pass
        try:
            dlnphis_dP = self._dlnphis_dP
        except AttributeError:
            dlnphis_dP = self.dlnphis_dP()

        try:
            phis = self._phis
        except AttributeError:
            phis = self.phis()

        self._dphis_dP = [dlnphis_dP[i]*phis[i] for i in range(self.N)]
        return self._dphis_dP

    def dfugacities_dP(self):
        r'''Method to calculate and return the pressure derivative of the
        fugacities of the components in the phase.

        .. math::
            \frac{\partial f_i}{\partial P} = z_i \left(P \frac{\partial
            \phi_i}{\partial P}  + \phi_i \right)

        Returns
        -------
        dfugacities_dP : list[float]
            Pressure derivative of fugacities of all components
            in the phase, [-]

        Notes
        -----
        For models without pressure dependence of fugacity, the returned result
        may not be exactly zero due to inaccuracy in floating point results;
        results are likely on the order of 1e-14 or lower in that case.
        '''
        try:
            dphis_dP = self._dphis_dP
        except AttributeError:
            dphis_dP = self.dphis_dP()

        try:
            phis = self._phis
        except AttributeError:
            phis = self.phis()

        P, zs = self.P, self.zs
        return [zs[i]*(P*dphis_dP[i] + phis[i]) for i in range(self.N)]

    def dfugacities_dns(self):
        r'''Method to calculate and return the mole number derivative of the
        fugacities of the components in the phase.

        if i != j:

        .. math::
            \frac{\partial f_i}{\partial n_j} = P\phi_i z_i \left(
            \frac{\partial \ln \phi_i}{\partial n_j} - 1
            \right)

        if i == j:

        .. math::
            \frac{\partial f_i}{\partial n_j} = P\phi_i z_i \left(
            \frac{\partial \ln \phi_i}{\partial n_j} - 1
            \right) + P\phi_i

        Returns
        -------
        dfugacities_dns : list[list[float]]
            Mole number derivatives of the fugacities of all components
            in the phase, [Pa/mol]

        Notes
        -----
        '''
        phis = self.phis()
        dlnphis_dns = self.dlnphis_dns()

        P, zs, cmps = self.P, self.zs, range(self.N)
        matrix = []
        for i in cmps:
            phi_P = P*phis[i]
            ziPphi = phi_P*zs[i]
            r = dlnphis_dns[i]
            row = [ziPphi*(r[j] - 1.0) for j in cmps]
            row[i] += phi_P
            matrix.append(row)
        return matrix

    def dlnfugacities_dns(self):
        r'''Method to calculate and return the mole number derivative of the
        log of fugacities of the components in the phase.

        .. math::
            \frac{\partial \ln f_i}{\partial n_j} = \frac{1}{f_i}
            \frac{\partial  f_i}{\partial n_j}

        Returns
        -------
        dlnfugacities_dns : list[list[float]]
            Mole number derivatives of the log of fugacities of all components
            in the phase, [log(Pa)/mol]

        Notes
        -----
        '''
        zs, cmps = self.zs, range(self.N)
        fugacities = self.fugacities()
        dlnfugacities_dns = [list(i) for i in self.dfugacities_dns()]
        fugacities_inv = [1.0/fi for fi in fugacities]
        for i in cmps:
            r = dlnfugacities_dns[i]
            for j in cmps:
                r[j]*= fugacities_inv[i]
        return dlnfugacities_dns

    def dlnfugacities_dzs(self):
        r'''Method to calculate and return the mole fraction derivative of the
        log of fugacities of the components in the phase.

        .. math::
            \frac{\partial \ln f_i}{\partial z_j} = \frac{1}{f_i}
            \frac{\partial  f_i}{\partial z_j}

        Returns
        -------
        dlnfugacities_dzs : list[list[float]]
            Mole fraction derivatives of the log of fugacities of all components
            in the phase, [log(Pa)]

        Notes
        -----
        '''
        zs, cmps = self.zs, range(self.N)
        fugacities = self.fugacities()
        dlnfugacities_dzs = [list(i) for i in self.dfugacities_dzs()]
        fugacities_inv = [1.0/fi for fi in fugacities]
        for i in cmps:
            r = dlnfugacities_dzs[i]
            for j in cmps:
                r[j]*= fugacities_inv[i]
        return dlnfugacities_dzs

    def log_zs(self):
        r'''Method to calculate and return the log of mole fractions specified.
        These are used in calculating entropy and in many other formulas.

        .. math::
            \ln z_i

        Returns
        -------
        log_zs : list[float]
            Log of mole fractions, [-]

        Notes
        -----
        '''
        try:
            return self._log_zs
        except AttributeError:
            pass
        try:
            self._log_zs = [log(zi) for zi in self.zs]
        except ValueError:
            self._log_zs = _log_zs = []
            for zi in self.zs:
                try:
                    _log_zs.append(log(zi))
                except ValueError:
                    _log_zs.append(-690.7755278982137) # log(1e-300)
        return self._log_zs

    def V_iter(self, force=False):
        r'''Method to calculate and return the volume of the phase in a way
        suitable for a TV resolution to converge on the same pressure. This
        often means the return value of this method is an mpmath `mpf`.
        This dummy method simply returns the implemented V method.

        Returns
        -------
        V : float or mpf
            Molar volume, [m^3/mol]

        Notes
        -----
        '''
        return self.V()

    def G(self):
        r'''Method to calculate and return the Gibbs free energy of the phase.

        .. math::
            G = H - TS

        Returns
        -------
        G : float
            Gibbs free energy, [J/mol]

        Notes
        -----
        '''
        try:
            return self._G
        except AttributeError:
            pass
        G = self.H() - self.T*self.S()
        self._G = G
        return G

    G_min = G

    def U(self):
        r'''Method to calculate and return the internal energy of the phase.

        .. math::
            U = H - PV

        Returns
        -------
        U : float
            Internal energy, [J/mol]

        Notes
        -----
        '''
        U = self.H() - self.P*self.V()
        return U

    def A(self):
        r'''Method to calculate and return the Helmholtz energy of the phase.

        .. math::
            A = U - TS

        Returns
        -------
        A : float
            Helmholtz energy, [J/mol]

        Notes
        -----
        '''
        A = self.U() - self.T*self.S()
        return A

    def dH_dns(self):
        r'''Method to calculate and return the mole number derivative of the
        enthalpy of the phase.

        .. math::
            \frac{\partial H}{\partial n_i}

        Returns
        -------
        dH_dns : list[float]
            Mole number derivatives of the enthalpy of the phase,
            [J/mol^2]

        Notes
        -----
        '''
        return dxs_to_dns(self.dH_dzs(), self.zs)

    def dS_dns(self):
        r'''Method to calculate and return the mole number derivative of the
        entropy of the phase.

        .. math::
            \frac{\partial S}{\partial n_i}

        Returns
        -------
        dS_dns : list[float]
            Mole number derivatives of the entropy of the phase,
            [J/(mol^2*K)]

        Notes
        -----
        '''
        return dxs_to_dns(self.dS_dzs(), self.zs)

    def dG_dT(self):
        r'''Method to calculate and return the constant-pressure
        temperature derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial T}\right)_{P}
            = -T\left(\frac{\partial S}{\partial T}\right)_{P}
            - S + \left(\frac{\partial H}{\partial T}\right)_{P}

        Returns
        -------
        dG_dT : float
            Constant-pressure temperature derivative of Gibbs free energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return -self.T*self.dS_dT() - self.S() + self.dH_dT()

    dG_dT_P = dG_dT

    def dG_dT_V(self):
        r'''Method to calculate and return the constant-volume
        temperature derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial T}\right)_{V}
            = -T\left(\frac{\partial S}{\partial T}\right)_{V}
            - S + \left(\frac{\partial H}{\partial T}\right)_{V}

        Returns
        -------
        dG_dT_V : float
            Constant-volume temperature derivative of Gibbs free energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return -self.T*self.dS_dT_V() - self.S() + self.dH_dT_V()

    def dG_dP(self):
        r'''Method to calculate and return the constant-temperature
        pressure derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial P}\right)_{T}
            = -T\left(\frac{\partial S}{\partial P}\right)_{T}
            + \left(\frac{\partial H}{\partial P}\right)_{T}

        Returns
        -------
        dG_dP : float
            Constant-temperature pressure derivative of Gibbs free energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return -self.T*self.dS_dP() + self.dH_dP()

    dG_dP_T = dG_dP

    def dG_dP_V(self):
        r'''Method to calculate and return the constant-volume
        pressure derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial P}\right)_{V}
            = -T\left(\frac{\partial S}{\partial P}\right)_{V}
            - S \left(\frac{\partial T}{\partial P}\right)_{V}
            + \left(\frac{\partial H}{\partial P}\right)_{V}

        Returns
        -------
        dG_dP_V : float
            Constant-volume pressure derivative of Gibbs free energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return -self.T*self.dS_dP_V() - self.dT_dP()*self.S() + self.dH_dP_V()

    def dG_dV_T(self):
        r'''Method to calculate and return the constant-temperature
        volume derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial V}\right)_{T}
            = \left(\frac{\partial G}{\partial P}\right)_{T}
            \left(\frac{\partial P}{\partial V}\right)_{T}

        Returns
        -------
        dG_dV_T : float
            Constant-temperature volume derivative of Gibbs free energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dG_dP_T()*self.dP_dV()

    def dG_dV_P(self):
        r'''Method to calculate and return the constant-pressure
        volume derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial V}\right)_{P}
            = \left(\frac{\partial G}{\partial T}\right)_{P}
            \left(\frac{\partial T}{\partial V}\right)_{P}

        Returns
        -------
        dG_dV_P : float
            Constant-pressure volume derivative of Gibbs free energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dG_dT_P()*self.dT_dV()


    def dU_dT(self):
        r'''Method to calculate and return the constant-pressure
        temperature derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial T}\right)_{P}
            = -P \left(\frac{\partial V}{\partial T}\right)_{P}
             + \left(\frac{\partial H}{\partial T}\right)_{P}

        Returns
        -------
        dU_dT : float
            Constant-pressure temperature derivative of internal energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return -self.P*self.dV_dT() + self.dH_dT()

    dU_dT_P = dU_dT

    def dU_dT_V(self):
        r'''Method to calculate and return the constant-volume
        temperature derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial T}\right)_{V}
            = \left(\frac{\partial H}{\partial T}\right)_{V}
             - V \left(\frac{\partial P}{\partial T}\right)_{V}

        Returns
        -------
        dU_dT_V : float
            Constant-volume temperature derivative of internal energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return self.dH_dT_V() - self.V()*self.dP_dT()

    def dU_dP(self):
        r'''Method to calculate and return the constant-temperature
        pressure derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial P}\right)_{T}
            = -P \left(\frac{\partial V}{\partial P}\right)_{T}
             - V + \left(\frac{\partial H}{\partial P}\right)_{T}

        Returns
        -------
        dU_dP : float
            Constant-temperature pressure derivative of internal energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return -self.P*self.dV_dP() - self.V() + self.dH_dP()

    dU_dP_T = dU_dP

    def dU_dP_V(self):
        r'''Method to calculate and return the constant-volume
        pressure derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial P}\right)_{V}
            = \left(\frac{\partial H}{\partial P}\right)_{V}
             - V

        Returns
        -------
        dU_dP_V : float
            Constant-volume pressure derivative of internal energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return self.dH_dP_V() - self.V()

    def dU_dV_T(self):
        r'''Method to calculate and return the constant-temperature
        volume derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial V}\right)_{T}
            = \left(\frac{\partial U}{\partial P}\right)_{T}
             \left(\frac{\partial P}{\partial V}\right)_{T}

        Returns
        -------
        dU_dV_T : float
            Constant-temperature volume derivative of internal energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dU_dP_T()*self.dP_dV()

    def dU_dV_P(self):
        r'''Method to calculate and return the constant-pressure
        volume derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial V}\right)_{P}
            = \left(\frac{\partial U}{\partial T}\right)_{P}
             \left(\frac{\partial T}{\partial V}\right)_{P}

        Returns
        -------
        dU_dV_P : float
            Constant-pressure volume derivative of internal energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dU_dT_P()*self.dT_dV()

    def dA_dT(self):
        r'''Method to calculate and return the constant-pressure
        temperature derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial T}\right)_{P}
            = -T \left(\frac{\partial S}{\partial T}\right)_{P}
             - S + \left(\frac{\partial U}{\partial T}\right)_{P}

        Returns
        -------
        dA_dT : float
            Constant-pressure temperature derivative of Helmholtz energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return -self.T*self.dS_dT() - self.S() + self.dU_dT()

    dA_dT_P = dA_dT

    def dA_dT_V(self):
        r'''Method to calculate and return the constant-volume
        temperature derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial T}\right)_{V}
            =  \left(\frac{\partial H}{\partial T}\right)_{V}
             - V \left(\frac{\partial P}{\partial T}\right)_{V}
             - T \left(\frac{\partial S}{\partial T}\right)_{V}
             - S

        Returns
        -------
        dA_dT_V : float
            Constant-volume temperature derivative of Helmholtz energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return (self.dH_dT_V() - self.V()*self.dP_dT() - self.T*self.dS_dT_V()
                - self.S())

    def dA_dP(self):
        r'''Method to calculate and return the constant-temperature
        pressure derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial P}\right)_{T}
            = -T \left(\frac{\partial S}{\partial P}\right)_{T}
             + \left(\frac{\partial U}{\partial P}\right)_{T}

        Returns
        -------
        dA_dP : float
            Constant-temperature pressure derivative of Helmholtz energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return -self.T*self.dS_dP() + self.dU_dP()

    dA_dP_T = dA_dP

    def dA_dP_V(self):
        r'''Method to calculate and return the constant-volume
        pressure derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial P}\right)_{V}
            = \left(\frac{\partial H}{\partial P}\right)_{V}
            - V - S\left(\frac{\partial T}{\partial P}\right)_{V}
            -T \left(\frac{\partial S}{\partial P}\right)_{V}

        Returns
        -------
        dA_dP_V : float
            Constant-volume pressure derivative of Helmholtz energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return (self.dH_dP_V() - self.V() - self.dT_dP()*self.S()
                - self.T*self.dS_dP_V())

    def dA_dV_T(self):
        r'''Method to calculate and return the constant-temperature
        volume derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial V}\right)_{T}
            = \left(\frac{\partial A}{\partial P}\right)_{T}
              \left(\frac{\partial P}{\partial V}\right)_{T}

        Returns
        -------
        dA_dV_T : float
            Constant-temperature volume derivative of Helmholtz energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dA_dP_T()*self.dP_dV()

    def dA_dV_P(self):
        r'''Method to calculate and return the constant-pressure
        volume derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial V}\right)_{P}
            = \left(\frac{\partial A}{\partial T}\right)_{P}
              \left(\frac{\partial T}{\partial V}\right)_{P}

        Returns
        -------
        dA_dV_P : float
            Constant-pressure volume derivative of Helmholtz energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dA_dT_P()*self.dT_dV()

    def G_dep(self):
        r'''Method to calculate and return the departure Gibbs free energy of
        the phase.

        .. math::
            G_{dep} = H_{dep} - TS_{dep}

        Returns
        -------
        G_dep : float
            Departure Gibbs free energy, [J/mol]

        Notes
        -----
        '''
        G_dep = self.H_dep() - self.T*self.S_dep()
        return G_dep

    def V_dep(self):
        r'''Method to calculate and return the departure (from ideal gas
        behavior) molar volume of the phase.

        .. math::
            V_{dep} = V - \frac{RT}{P}

        Returns
        -------
        V_dep : float
            Departure molar volume, [m^3/mol]

        Notes
        -----
        '''
        V_dep = self.V() - self.R*self.T/self.P
        return V_dep

    def U_dep(self):
        r'''Method to calculate and return the departure internal energy of
        the phase.

        .. math::
            U_{dep} = H_{dep} - PV_{dep}

        Returns
        -------
        U_dep : float
            Departure internal energy, [J/mol]

        Notes
        -----
        '''
        return self.H_dep() - self.P*self.V_dep()

    def A_dep(self):
        r'''Method to calculate and return the departure Helmholtz energy of
        the phase.

        .. math::
            A_{dep} = U_{dep} - TS_{dep}

        Returns
        -------
        A_dep : float
            Departure Helmholtz energy, [J/mol]

        Notes
        -----
        '''
        return self.U_dep() - self.T*self.S_dep()


    def H_reactive(self):
        r'''Method to calculate and return the enthalpy of the phase on a
        reactive basis, using the `Hfs` values of the phase.

        .. math::
            H_{reactive} = H + \sum_i z_i {H_{f,i}}

        Returns
        -------
        H_reactive : float
            Enthalpy of the phase on a reactive basis, [J/mol]

        Notes
        -----
        '''
        try:
            return self._H_reactive
        except AttributeError:
            pass
        H = self.H()
        for zi, Hf in zip(self.zs, self.Hfs):
            H += zi*Hf
        self._H_reactive = H
        return H

    def S_reactive(self):
        r'''Method to calculate and return the entropy of the phase on a
        reactive basis, using the `Sfs` values of the phase.

        .. math::
            S_{reactive} = S + \sum_i z_i {S_{f,i}}

        Returns
        -------
        S_reactive : float
            Entropy of the phase on a reactive basis, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._S_reactive
        except:
            pass
        S = self.S()
        for zi, Sf in zip(self.zs, self.Sfs):
            S += zi*Sf
        self._S_reactive = S
        return S

    def G_reactive(self):
        r'''Method to calculate and return the Gibbs free energy of the phase
        on a reactive basis.

        .. math::
            G_{reactive} = H_{reactive} - TS_{reactive}

        Returns
        -------
        G_reactive : float
            Gibbs free energy of the phase on a reactive basis, [J/(mol)]

        Notes
        -----
        '''
        G = self.H_reactive() - self.T*self.S_reactive()
        return G

    def U_reactive(self):
        r'''Method to calculate and return the internal energy of the phase
        on a reactive basis.

        .. math::
            U_{reactive} = H_{reactive} - PV

        Returns
        -------
        U_reactive : float
            Internal energy of the phase on a reactive basis, [J/(mol)]

        Notes
        -----
        '''
        U = self.H_reactive() - self.P*self.V()
        return U

    def A_reactive(self):
        r'''Method to calculate and return the Helmholtz free energy of the
        phase on a reactive basis.

        .. math::
            A_{reactive} = U_{reactive} - TS_{reactive}

        Returns
        -------
        A_reactive : float
            Helmholtz free energy of the phase on a reactive basis, [J/(mol)]

        Notes
        -----
        '''
        A = self.U_reactive() - self.T*self.S_reactive()
        return A

    def H_formation_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas enthalpy of formation
        of the phase (as if the phase was an ideal gas).

        .. math::
            H_{reactive}^{ig} = \sum_i z_i {H_{f,i}}

        Returns
        -------
        H_formation_ideal_gas : float
            Enthalpy of formation of the phase on a reactive basis
            as an ideal gas, [J/mol]

        Notes
        -----
        '''
        try:
            return self._H_formation_ideal_gas
        except AttributeError:
            pass
        Hf_ideal_gas = 0.0
        for zi, Hf in zip(self.zs, self.Hfs):
            Hf_ideal_gas += zi*Hf
        self._H_formation_ideal_gas = Hf_ideal_gas
        return Hf_ideal_gas

    def S_formation_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas entropy of formation
        of the phase (as if the phase was an ideal gas).

        .. math::
            S_{reactive}^{ig} = \sum_i z_i {S_{f,i}}

        Returns
        -------
        S_formation_ideal_gas : float
            Entropy of formation of the phase on a reactive basis
            as an ideal gas, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._S_formation_ideal_gas
        except:
            pass
        Sf_ideal_gas = 0.0
        for zi, Sf in zip(self.zs, self.Sfs):
            Sf_ideal_gas += zi*Sf
        self._S_formation_ideal_gas = Sf_ideal_gas
        return Sf_ideal_gas

    def G_formation_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas Gibbs free energy of
        formation of the phase (as if the phase was an ideal gas).

        .. math::
            G_{reactive}^{ig} = H_{reactive}^{ig} - T_{ref}^{ig}
            S_{reactive}^{ig}

        Returns
        -------
        G_formation_ideal_gas : float
            Gibbs free energy of formation of the phase on a reactive basis
            as an ideal gas, [J/(mol)]

        Notes
        -----
        '''
        Gf = self.H_formation_ideal_gas() - self.T_REF_IG*self.S_formation_ideal_gas()
        return Gf

    def U_formation_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas internal energy of
        formation of the phase (as if the phase was an ideal gas).

        .. math::
            U_{reactive}^{ig} = H_{reactive}^{ig} - P_{ref}^{ig}
            V^{ig}

        Returns
        -------
        U_formation_ideal_gas : float
            Internal energy of formation of the phase on a reactive basis
            as an ideal gas, [J/(mol)]

        Notes
        -----
        '''
        Uf = self.H_formation_ideal_gas() - self.P_REF_IG*self.V_ideal_gas()
        return Uf

    def A_formation_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas Helmholtz energy of
        formation of the phase (as if the phase was an ideal gas).

        .. math::
            A_{reactive}^{ig} = U_{reactive}^{ig} - T_{ref}^{ig}
            S_{reactive}^{ig}

        Returns
        -------
        A_formation_ideal_gas : float
            Helmholtz energy of formation of the phase on a reactive basis
            as an ideal gas, [J/(mol)]

        Notes
        -----
        '''
        Af = self.U_formation_ideal_gas() - self.T_REF_IG*self.S_formation_ideal_gas()
        return Af

    def Cv(self):
        r'''Method to calculate and return the constant-volume heat
        capacity `Cv` of the phase.

        .. math::
            C_v = T\left(\frac{\partial P}{\partial T}\right)_V^2/
            \left(\frac{\partial P}{\partial V}\right)_T + Cp

        Returns
        -------
        Cv : float
            Constant volume molar heat capacity, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._Cv
        except AttributeError:
            pass
        # checks out

        Cp_m_Cv = Cp_minus_Cv(self.T, self.dP_dT(), self.dP_dV())
        Cp = self.Cp()
        self._Cv = Cv = Cp - Cp_m_Cv
        return Cv

    def dCv_dT_P(self):
        r'''Method to calculate the temperature derivative of Cv, constant
        volume heat capacity, at constant pressure.

        .. math::
            \left(\frac{\partial C_v}{\partial T}\right)_P =
            - \frac{T \operatorname{dPdT_{V}}^{2}{\left(T \right)} \frac{d}{dT}
            \operatorname{dPdV_{T}}{\left(T \right)}}{\operatorname{dPdV_{T}}^{2}
            {\left(T \right)}} + \frac{2 T \operatorname{dPdT_{V}}{\left(T \right)}
            \frac{d}{d T} \operatorname{dPdT_{V}}{\left(T \right)}}
            {\operatorname{dPdV_{T}}{\left(T \right)}} + \frac{\operatorname{
            dPdT_{V}}^{2}{\left(T \right)}}{\operatorname{dPdV_{T}}{\left(T
            \right)}} + \frac{d}{d T} \operatorname{Cp}{\left(T \right)}

        Returns
        -------
        dCv_dT_P : float
           Temperature derivative of constant volume heat capacity at constant
           pressure, [J/mol/K^2]

        Notes
        -----
        Requires `d2P_dT2_PV`, `d2P_dVdT_TP`, and `d2H_dT2`.
        '''
        T = self.T
        x0 = self.dP_dT_V()
        x1 = x0*x0
        x2 = self.dP_dV_T()
        x3 = 1.0/x2

        x50 = self.d2P_dT2_PV()
        x51 = self.d2P_dVdT_TP()
        x52 = self.d2H_dT2()
        return 2.0*T*x0*x3*x50 - T*x1*x51*x3*x3 + x1*x3 + x52

    def dCv_dP_T(self):
        r'''Method to calculate the pressure derivative of Cv, constant
        volume heat capacity, at constant temperature.

        .. math::
            \left(\frac{\partial C_v}{\partial P}\right)_T =
            - T \operatorname{dPdT_{V}}{\left(P \right)} \frac{d}{d P}
            \operatorname{dVdT_{P}}{\left(P \right)} - T \operatorname{
            dVdT_{P}}{\left(P \right)} \frac{d}{d P} \operatorname{dPdT_{V}}
            {\left(P \right)} + \frac{d}{d P} \operatorname{Cp}{\left(P\right)}

        Returns
        -------
        dCv_dP_T : float
           Pressure derivative of constant volume heat capacity at constant
           temperature, [J/mol/K/Pa]

        Notes
        -----
        Requires `d2V_dTdP`, `d2P_dTdP`, and `d2H_dTdP`.
        '''
        T = self.T
        dP_dT_V = self.dP_dT_V()
        d2V_dTdP = self.d2V_dTdP()
        dV_dT_P = self.dV_dT_P()
        d2P_dTdP = self.d2P_dTdP()
        d2H_dep_dTdP = self.d2H_dTdP()
        return -T*dP_dT_V*d2V_dTdP - T*dV_dT_P*d2P_dTdP + d2H_dep_dTdP

    def chemical_potential(self):
        r'''Method to calculate and return the chemical potentials of each
        component in the phase [-]. For a pure substance, this is the
        molar Gibbs energy on a reactive basis.

        .. math::
            \frac{\partial G}{\partial n_i}_{T, P, N_{j \ne i}}

        Returns
        -------
        chemical_potential : list[float]
            Chemical potentials, [J/mol]
        '''
        try:
            return self._chemical_potentials
        except AttributeError:
            pass
        dS_dzs = self.dS_dzs()
        dH_dzs = self.dH_dzs()
        T, Hfs, Sfs = self.T, self.Hfs, self.Sfs
        dG_reactive_dzs = [Hfs[i] - T*(Sfs[i] + dS_dzs[i]) + dH_dzs[i] for i in range(self.N)]
        dG_reactive_dns = dxs_to_dns(dG_reactive_dzs, self.zs)
        chemical_potentials = dns_to_dn_partials(dG_reactive_dns, self.G_reactive())
        self._chemical_potentials = chemical_potentials
        return chemical_potentials
#        # CORRECT DO NOT CHANGE
#        # TODO analytical implementation
#        def to_diff(ns):
#            tot = sum(ns)
#            zs = normalize(ns)
#            return tot*self.to_TP_zs(self.T, self.P, zs).G_reactive()
#        return jacobian(to_diff, self.zs)

    def activities(self):
        r'''Method to calculate and return the activities of each component
        in the phase [-].

        .. math::
            a_i(T, P, x; f_i^0) = \frac{f_i(T, P, x)}{f_i^0(T, P_i^0)}

        Returns
        -------
        activities : list[float]
            Activities, [-]
        '''
        # For a good discussion, see
        # Thermodynamics: Fundamentals for Applications, J. P. O'Connell, J. M. Haile
        # 5.4 DEVIATIONS FROM IDEAL SOLUTIONS: RATIO MEASURES page 201
        # CORRECT DO NOT CHANGE
        fugacities = self.fugacities()
        fugacities_std = self.fugacities_std() # TODO implement fugacities_std
        return [fugacities[i]/fugacities_std[i] for i in range(self.N)]

    def gammas(self):
        r'''Method to calculate and return the activity coefficients of the
        phase, [-].

        Activity coefficients are defined as the ratio of
        the actual fugacity coefficients times the pressure to the reference
        pure fugacity coefficients times the reference pressure.
        The reference pressure can be set to the actual pressure (the Lewis
        Randall standard state) which makes the pressures cancel.

        .. math::
            \gamma_i(T, P, x; f_i^0(T, P_i^0)) =
            \frac{\phi_i(T, P, x)P}{\phi_i^0(T, P_i^0) P_i^0}

        Returns
        -------
        gammas : list[float]
            Activity coefficients, [-]
        '''
        # For a good discussion, see
        # Thermodynamics: Fundamentals for Applications, J. P. O'Connell, J. M. Haile
        # 5.5 ACTIVITY COEFFICIENTS FROM FUGACITY COEFFICIENTS
        # There is no one single definition for gamma but it is believed this is
        # the most generally used one for EOSs; and activity methods
        # override this
        phis = self.phis()
        gammas = []
        T, P, zs, N = self.T, self.P, self.zs, self.N
        for i in range(N):
            zeros = [0.0]*N
            zeros[i] = 1.0
            phi = self.to_TP_zs(T=T, P=P, zs=zeros).phis()[i]
            gammas.append(phis[i]/phi)
        return gammas

    def Cp_Cv_ratio(self):
        r'''Method to calculate and return the Cp/Cv ratio of the phase.

        .. math::
            \frac{C_p}{C_v}

        Returns
        -------
        Cp_Cv_ratio : float
            Cp/Cv ratio, [-]

        Notes
        -----
        '''
        return self.Cp()/self.Cv()

    isentropic_exponent = Cp_Cv_ratio

    def Z(self):
        r'''Method to calculate and return the compressibility factor of the
        phase.

        .. math::
            Z = \frac{PV}{RT}

        Returns
        -------
        Z : float
            Compressibility factor, [-]

        Notes
        -----
        '''
        return self.P*self.V()/(self.R*self.T)

    def rho(self):
        r'''Method to calculate and return the molar density of the
        phase.

        .. math::
            \rho = frac{1}{V}

        Returns
        -------
        rho : float
            Molar density, [mol/m^3]

        Notes
        -----
        '''
        return 1.0/self.V()

    def dT_dP(self):
        r'''Method to calculate and return the constant-volume pressure
        derivative of temperature of the phase.

        .. math::
            \left(\frac{\partial T}{\partial P}\right)_V = \frac{1}{\left(\frac{
            \partial P}{\partial T}\right)_V}

        Returns
        -------
        dT_dP : float
            Constant-volume pressure derivative of temperature, [K/Pa]

        Notes
        -----
        '''
        return 1.0/self.dP_dT()

    def dV_dT(self):
        r'''Method to calculate and return the constant-pressure temperature
        derivative of volume of the phase.

        .. math::
            \left(\frac{\partial V}{\partial T}\right)_P =
            \frac{-\left(\frac{\partial P}{\partial T}\right)_V}
            {\left(\frac{\partial P}{\partial V}\right)_T}

        Returns
        -------
        dV_dT : float
            Constant-pressure temperature derivative of volume, [m^3/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._dV_dT
        except AttributeError:
            pass
        dV_dT = self._dV_dT = -self.dP_dT()/self.dP_dV()
        return dV_dT

    def dV_dP(self):
        r'''Method to calculate and return the constant-temperature pressure
        derivative of volume of the phase.

        .. math::
            \left(\frac{\partial V}{\partial P}\right)_T =
            {-\left(\frac{\partial V}{\partial T}\right)_P}
            {\left(\frac{\partial T}{\partial P}\right)_V}

        Returns
        -------
        dV_dP : float
            Constant-temperature pressure derivative of volume, [m^3/(mol*Pa)]

        Notes
        -----
        '''
        return -self.dV_dT()*self.dT_dP()

    def dT_dV(self):
        r'''Method to calculate and return the constant-pressure volume
        derivative of temperature of the phase.

        .. math::
            \left(\frac{\partial T}{\partial V}\right)_P =
            \frac{1}
            {\left(\frac{\partial V}{\partial T}\right)_P}

        Returns
        -------
        dT_dV : float
            Constant-pressure volume derivative of temperature, [K*m^3/(m^3)]

        Notes
        -----
        '''
        return 1./self.dV_dT()

    def d2V_dP2(self):
        r'''Method to calculate and return the constant-temperature pressure
        derivative of volume of the phase.

        .. math::
            \left(\frac{\partial^2 V}{\partial P^2}\right)_T =
            -\frac{\left(\frac{\partial^2 P}{\partial V^2}\right)_T}
            {\left(\frac{\partial P}{\partial V}\right)_T^3}

        Returns
        -------
        d2V_dP2 : float
            Constant-temperature pressure derivative of volume, [m^3/(mol*Pa^2)]

        Notes
        -----
        '''
        inverse_dP_dV = 1.0/self.dP_dV()
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV*inverse_dP_dV
        return -self.d2P_dV2()*inverse_dP_dV3

    def d2T_dP2(self):
        r'''Method to calculate and return the constant-volume second pressure
        derivative of temperature of the phase.

        .. math::
            \left(\frac{\partial^2 T}{\partial P^2}\right)_V =
            -\left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \left(\frac{\partial T}{\partial P}\right)_V^3

        Returns
        -------
        d2T_dP2 : float
            Constant-volume second pressure derivative of temperature, [K/Pa^2]

        Notes
        -----
        '''
        dT_dP = self.dT_dP()
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP
        return -self.d2P_dT2()*inverse_dP_dT3

    def d2T_dV2(self):
        r'''Method to calculate and return the constant-pressure second volume
        derivative of temperature of the phase.

        .. math::
            \left(\frac{\partial^2 T}{\partial V^2}\right)_P = -\left[
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) \right]
            \left(\frac{\partial P}{\partial T}\right)^{-2}_V
            + \left[\left(\frac{\partial^2 P}{\partial T\partial V}\right)
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V\right]
            \left(\frac{\partial P}{\partial T}\right)_V^{-3}
            \left(\frac{\partial P}{\partial V}\right)_T

        Returns
        -------
        d2T_dV2 : float
            Constant-pressure second volume derivative of temperature,
            [K*mol^2/m^6]

        Notes
        -----
        '''
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dTdV = self.d2P_dTdV()
        d2P_dT2 = self.d2P_dT2()
        dT_dP = self.dT_dP()
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP

        return (-(self.d2P_dV2()*dP_dT - dP_dV*d2P_dTdV)*inverse_dP_dT2
                   +(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3*dP_dV)

    def d2V_dT2(self):
        r'''Method to calculate and return the constant-pressure second
        temperature derivative of volume of the phase.

        .. math::
            \left(\frac{\partial^2 V}{\partial T^2}\right)_P = -\left[
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) \right]
            \left(\frac{\partial P}{\partial V}\right)^{-2}_T
            + \left[\left(\frac{\partial^2 P}{\partial T\partial V}\right)
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T\right]
            \left(\frac{\partial P}{\partial V}\right)_T^{-3}
            \left(\frac{\partial P}{\partial T}\right)_V

        Returns
        -------
        d2V_dT2 : float
            Constant-pressure second temperature derivative of volume,
            [m^3/(mol*K^2)]

        Notes
        -----
        '''
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dTdV = self.d2P_dTdV()
        d2P_dT2 = self.d2P_dT2()
        d2P_dV2 = self.d2P_dV2()

        inverse_dP_dV = 1.0/dP_dV
        inverse_dP_dV2 = inverse_dP_dV*inverse_dP_dV
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV2

        return  (-(d2P_dT2*dP_dV - dP_dT*d2P_dTdV)*inverse_dP_dV2
                   +(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*inverse_dP_dV3*dP_dT)

    def d2V_dPdT(self):
        r'''Method to calculate and return the derivative of pressure and then
        the derivative of temperature of volume of the phase.

        .. math::
            \left(\frac{\partial^2 V}{\partial T\partial P}\right) =
            - \left[\left(\frac{\partial^2 P}{\partial T \partial V}\right)
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            \right]\left(\frac{\partial P}{\partial V}\right)_T^{-3}

        Returns
        -------
        d2V_dPdT : float
            Derivative of pressure and then the derivative of temperature
            of volume, [m^3/(mol*K*Pa)]

        Notes
        -----
        '''
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dTdV = self.d2P_dTdV()
        d2P_dV2 = self.d2P_dV2()

        inverse_dP_dV = 1.0/dP_dV
        inverse_dP_dV2 = inverse_dP_dV*inverse_dP_dV
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV2

        return -(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*inverse_dP_dV3
    d2V_dTdP = d2V_dPdT

    def d2T_dPdV(self):
        r'''Method to calculate and return the derivative of pressure and then
        the derivative of volume of temperature of the phase.

        .. math::
           \left(\frac{\partial^2 T}{\partial P\partial V}\right) =
            - \left[\left(\frac{\partial^2 P}{\partial T \partial V}\right)
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \right]\left(\frac{\partial P}{\partial T}\right)_V^{-3}

        Returns
        -------
        d2T_dPdV : float
            Derivative of pressure and then the derivative of volume
            of temperature, [K*mol/(Pa*m^3)]

        Notes
        -----
        '''
        dT_dP = self.dT_dP()
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP

        d2P_dTdV = self.d2P_dTdV()
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dT2 = self.d2P_dT2()
        return -(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3

    d2T_dVdP = d2T_dPdV
    def d2P_dVdT(self):
        r'''Method to calculate and return the second derivative of
        pressure with respect to temperature and volume of the phase.
        This is an alias of `d2P_dTdV`.

        .. math::
             \frac{\partial^2 P}{\partial V \partial T}

        Returns
        -------
        d2P_dVdT : float
            Second volume derivative of pressure, [mol*Pa^2/(J*K)]
        '''
        return self.d2P_dTdV()

    def dZ_dzs(self):
        r'''Method to calculate and return the mole fraction derivatives of the
        compressibility factor `Z` of the phase.

        .. math::
            \frac{\partial Z}{\partial z_i}

        Returns
        -------
        dZ_dzs : list[float]
            Mole fraction derivatives of the compressibility factor of the
            phase, [-]

        Notes
        -----
        '''
        factor = self.P/(self.T*self.R)
        return [dV*factor for dV in self.dV_dzs()]

    def dZ_dns(self):
        r'''Method to calculate and return the mole number derivatives of the
        compressibility factor `Z` of the phase.

        .. math::
            \frac{\partial Z}{\partial n_i}

        Returns
        -------
        dZ_dns : list[float]
            Mole number derivatives of the compressibility factor of the
            phase, [1/mol]

        Notes
        -----
        '''
        return dxs_to_dns(self.dZ_dzs(), self.zs)

    def dV_dns(self):
        r'''Method to calculate and return the mole number derivatives of the
        molar volume `V` of the phase.

        .. math::
            \frac{\partial V}{\partial n_i}

        Returns
        -------
        dV_dns : list[float]
            Mole number derivatives of the molar volume of the phase, [m^3]

        Notes
        -----
        '''
        return dxs_to_dns(self.dV_dzs(), self.zs)

    # Derived properties
    def PIP(self):
        r'''Method to calculate and return the phase identification parameter
        of the phase.

        .. math::
            \Pi = V \left[\frac{\frac{\partial^2 P}{\partial V \partial T}}
            {\frac{\partial P }{\partial T}}- \frac{\frac{\partial^2 P}{\partial
            V^2}}{\frac{\partial P}{\partial V}} \right]

        Returns
        -------
        PIP : float
            Phase identification parameter, [-]
        '''
        return phase_identification_parameter(self.V(), self.dP_dT(), self.dP_dV(),
                                              self.d2P_dV2(), self.d2P_dTdV())

    def kappa(self):
        r'''Method to calculate and return the isothermal compressibility
        of the phase.

        .. math::
            \kappa = -\frac{1}{V}\left(\frac{\partial V}{\partial P} \right)_T

        Returns
        -------
        kappa : float
            Isothermal coefficient of compressibility, [1/Pa]
        '''
        return -1.0/self.V()*self.dV_dP()
    isothermal_compressibility = kappa

    def dkappa_dT(self):
        r'''Method to calculate and return the temperature derivative of
        isothermal compressibility of the phase.

        .. math::
            \frac{\partial \kappa}{\partial T} = -\frac{ \left(\frac{\partial^2 V}{\partial P\partial T} \right)}{V}
            + \frac{\left(\frac{\partial V}{\partial P} \right)_T\left(\frac{\partial V}{\partial T} \right)_P}{V^2}

        Returns
        -------
        dkappa_dT : float
            First temperature derivative of isothermal coefficient of
            compressibility, [1/(Pa*K)]
        '''
        V, dV_dP, dV_dT, d2V_dTdP = self.V(), self.dV_dP(), self.dV_dT(), self.d2V_dTdP()
        return -d2V_dTdP/V + dV_dP*dV_dT/(V*V)

    disothermal_compressibility_dT = dkappa_dT

    def isothermal_bulk_modulus(self):
        r'''Method to calculate and return the isothermal bulk modulus
        of the phase.

        .. math::
            K_T = -V\left(\frac{\partial P}{\partial V} \right)_T

        Returns
        -------
        isothermal_bulk_modulus : float
            Isothermal bulk modulus, [Pa]
        '''
        return 1.0/self.kappa()

    def isobaric_expansion(self):
        r'''Method to calculate and return the isobatic expansion coefficient
        of the phase.

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Returns
        -------
        beta : float
            Isobaric coefficient of a thermal expansion, [1/K]
        '''
        return self.dV_dT()/self.V()

    def disobaric_expansion_dT(self):
        r'''Method to calculate and return the temperature derivative of
        isobatic expansion coefficient of the phase.

        .. math::
            \frac{\partial \beta}{\partial T} = \frac{1}{V}\left(
            \left(\frac{\partial^2 V}{\partial T^2} \right)_P
            - \left(\frac{\partial V}{\partial T} \right)_P^2/V
            \right)

        Returns
        -------
        dbeta_dT : float
            Temperature derivative of isobaric coefficient of a thermal
            expansion, [1/K^2]
        '''
        '''
        from sympy import *
        T, P = symbols('T, P')
        V = symbols('V', cls=Function)
        expr = 1/V(T, P)*Derivative(V(T, P), T)
        diff(expr, T)
        Derivative(V(T, P), (T, 2))/V(T, P) - Derivative(V(T, P), T)**2/V(T, P)**2
        # Untested
        '''
        V_inv = 1.0/self.V()
        dV_dT = self.dV_dT()
        return V_inv*(self.d2V_dT2() - dV_dT*dV_dT*V_inv)

    def disobaric_expansion_dP(self):
        r'''Method to calculate and return the pressure derivative of
        isobatic expansion coefficient of the phase.

        .. math::
            \frac{\partial \beta}{\partial P} = \frac{1}{V}\left(
            \left(\frac{\partial^2 V}{\partial T\partial P} \right)
            -\frac{ \left(\frac{\partial V}{\partial T} \right)_P
             \left(\frac{\partial V}{\partial P} \right)_T}{V}
            \right)

        Returns
        -------
        dbeta_dP : float
            Pressure derivative of isobaric coefficient of a thermal
            expansion, [1/(K*Pa)]
        '''
        '''
        from sympy import *
        T, P = symbols('T, P')
        V = symbols('V', cls=Function)
        expr = 1/V(T, P)*Derivative(V(T, P), T)
        diff(expr, P)
        Derivative(V(T, P), P, T)/V(T, P) - Derivative(V(T, P), P)*Derivative(V(T, P), T)/V(T, P)**2

        '''
        V_inv = 1.0/self.V()
        dV_dT = self.dV_dT()
        dV_dP = self.dV_dP()
        return V_inv*(self.d2V_dTdP() - dV_dT*dV_dP*V_inv)

    def Joule_Thomson(self):
        r'''Method to calculate and return the Joule-Thomson coefficient
        of the phase.

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H = \frac{1}{C_p}
            \left[T \left(\frac{\partial V}{\partial T}\right)_P - V\right]
            = \frac{V}{C_p}\left(\beta T-1\right)

        Returns
        -------
        mu_JT : float
            Joule-Thomson coefficient [K/Pa]
        '''
        return Joule_Thomson(self.T, self.V(), self.Cp(), dV_dT=self.dV_dT(), beta=self.isobaric_expansion())

    def speed_of_sound(self):
        r'''Method to calculate and return the molar speed of sound
        of the phase.

        .. math::
            w = \left[-V^2 \left(\frac{\partial P}{\partial V}\right)_T \frac{C_p}
            {C_v}\right]^{1/2}

        A similar expression based on molar density is:

        .. math::
           w = \left[\left(\frac{\partial P}{\partial \rho}\right)_T \frac{C_p}
           {C_v}\right]^{1/2}

        Returns
        -------
        w : float
            Speed of sound for a real gas, [m*kg^0.5/(s*mol^0.5)]
        '''
        # Intentionally molar
        return speed_of_sound(self.V(), self.dP_dV(), self.Cp(), self.Cv())

    ### Compressibility factor derivatives
    def dZ_dT(self):
        r'''Method to calculate and return the temperature derivative of
        compressibility of the phase.

        .. math::
            \frac{\partial Z}{\partial P} = P\frac{\left(\frac{\partial
            V}{\partial T}\right)_P - \frac{-V}{T}}{RT}

        Returns
        -------
        dZ_dT : float
            Temperature derivative of compressibility, [1/K]
        '''
        T_inv = 1.0/self.T
        return self.P*self.R_inv*T_inv*(self.dV_dT() - self.V()*T_inv)

    def dZ_dP(self):
        r'''Method to calculate and return the pressure derivative of
        compressibility of the phase.

        .. math::
            \frac{\partial Z}{\partial P} = \frac{V + P\left(\frac{\partial
            V}{\partial P}\right)_T}{RT}

        Returns
        -------
        dZ_dP : float
            Pressure derivative of compressibility, [1/Pa]
        '''
        return 1.0/(self.T*self.R)*(self.V() + self.P*self.dV_dP())

    def dZ_dV(self):
        r'''Method to calculate and return the volume derivative of
        compressibility of the phase.

        .. math::
            \frac{\partial Z}{\partial V} = \frac{P - \rho \left(\frac{\partial
            P}{\partial \rho}\right)_T}{RT}

        Returns
        -------
        dZ_dV : float
            Volume derivative of compressibility, [mol/(m^3)]
        '''
        return (self.P - self.rho()*self.dP_drho())/(self.R*self.T)
    # Could add more

    ### Derivatives in the molar density basis
    def dP_drho(self):
        r'''Method to calculate and return the molar density derivative of
        pressure of the phase.

        .. math::
            \frac{\partial P}{\partial \rho} = -V^2\left(\frac{\partial
            P}{\partial V}\right)_T

        Returns
        -------
        dP_drho : float
            Molar density derivative of pressure, [Pa*m^3/mol]
        '''
        V = self.V()
        return -V*V*self.dP_dV()

    def drho_dP(self):
        r'''Method to calculate and return the pressure derivative of
        molar density of the phase.

        .. math::
            \frac{\partial \rho}{\partial P} = -\frac{1}{V^2}\left(\frac{\partial
            V}{\partial P}\right)_T

        Returns
        -------
        drho_dP : float
            Pressure derivative of Molar density, [mol/(Pa*m^3)]
        '''
        V = self.V()
        return -self.dV_dP()/(V*V)

    def d2P_drho2(self):
        r'''Method to calculate and return the second molar density derivative
        of pressure of the phase.

        .. math::
            \frac{\partial^2 P}{\partial \rho^2} = -V^2\left(
            -V^2 \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            -2 V \left(\frac{\partial P}{\partial V}\right)_T
            \right)

        Returns
        -------
        d2P_drho2 : float
            Second molar density derivative of pressure, [Pa*m^6/mol^2]
        '''
        V = self.V()
        return V*V*V*(V*self.d2P_dV2() + 2.0*self.dP_dV())

    def d2rho_dP2(self):
        r'''Method to calculate and return the second pressure derivative
        of molar density of the phase.

        .. math::
            \frac{\partial^2 \rho}{\partial P^2} = -\frac{1}{V^2}
            \left(\frac{\partial^2 V}{\partial P^2}\right)_T
            + \frac{2}{V^3} \left(\frac{\partial V}{\partial P}\right)_T^2

        Returns
        -------
        d2rho_dP2 : float
            Second pressure derivative of molar density, [mol^2/(Pa*m^6)]
        '''
        V = self.V()
        return -self.d2V_dP2()/V**2 + 2*self.dV_dP()**2/V**3

    def dT_drho(self):
        r'''Method to calculate and return the molar density derivative of
        temperature of the phase.

        .. math::
            \frac{\partial T}{\partial \rho} = -V^2\left(\frac{\partial
            T}{\partial V}\right)_P

        Returns
        -------
        dT_drho : float
            Molar density derivative of temperature, [K*m^3/mol]
        '''
        V = self.V()
        return -V*V*self.dT_dV()

    def d2T_drho2(self):
        r'''Method to calculate and return the second molar density derivative
        of temperature of the phase.

        .. math::
            \frac{\partial^2 T}{\partial \rho^2} = -V^2\left(
            -V^2 \left(\frac{\partial^2 T}{\partial V^2}\right)_P
            -2 V \left(\frac{\partial T}{\partial V}\right)_P
            \right)

        Returns
        -------
        d2T_drho2 : float
            Second molar density derivative of temperature, [K*m^6/mol^2]
        '''
        V = self.V()
        return V*V*V*(V*self.d2T_dV2() + 2.0*self.dT_dV())

    def drho_dT(self):
        r'''Method to calculate and return the temperature derivative of
        molar density of the phase.

        .. math::
            \frac{\partial \rho}{\partial T} = -\frac{1}{V^2}\left(\frac{\partial
            V}{\partial T}\right)_P

        Returns
        -------
        drho_dT : float
            Temperature derivative of molar density, [mol/(K*m^3)]
        '''
        V = self.V()
        return -self.dV_dT()/(V*V)

    def d2rho_dT2(self):
        r'''Method to calculate and return the second temperature derivative
        of molar density of the phase.

        .. math::
            \frac{\partial^2 \rho}{\partial T^2} = -\frac{1}{V^2}
            \left(\frac{\partial^2 V}{\partial T^2}\right)_P
            + \frac{2}{V^3} \left(\frac{\partial V}{\partial T}\right)_T^2

        Returns
        -------
        d2rho_dT2 : float
            Second temperature derivative of molar density, [mol^2/(K*m^6)]
        '''
        d2V_dT2 = self.d2V_dT2()
        V = self.V()
        dV_dT = self.dV_dT()
        return -d2V_dT2/V**2 + 2*dV_dT**2/V**3

    def d2P_dTdrho(self):
        r'''Method to calculate and return the temperature derivative
        and then molar density derivative of the pressure of the phase.

        .. math::
            \frac{\partial^2 P}{\partial T \partial \rho} = -V^2
            \left(\frac{\partial^2 P}{\partial T \partial V}\right)

        Returns
        -------
        d2P_dTdrho : float
            Temperature derivative and then molar density derivative of the
            pressure, [Pa*m^3/(K*mol)]
        '''
        V = self.V()
        d2P_dTdV = self.d2P_dTdV()
        return -(V*V)*d2P_dTdV

    def d2T_dPdrho(self):
        r'''Method to calculate and return the pressure derivative
        and then molar density derivative of the temperature of the phase.

        .. math::
            \frac{\partial^2 T}{\partial P \partial \rho} = -V^2
            \left(\frac{\partial^2 T}{\partial P \partial V}\right)

        Returns
        -------
        d2T_dPdrho : float
            Pressure derivative and then molar density derivative of the
            temperature, [K*m^3/(Pa*mol)]
        '''
        V = self.V()
        d2T_dPdV = self.d2T_dPdV()
        return -(V*V)*d2T_dPdV

    def d2rho_dPdT(self):
        r'''Method to calculate and return the pressure derivative
        and then temperature derivative of the molar density of the phase.

        .. math::
            \frac{\partial^2 \rho}{\partial P \partial T} = -\frac{1}{V^2}
            \left(\frac{\partial^2 V}{\partial P \partial T}\right)
            + \frac{2}{V^3}  \left(\frac{\partial V}{\partial T}\right)_P
             \left(\frac{\partial V}{\partial P}\right)_T

        Returns
        -------
        d2rho_dPdT : float
            Pressure derivative and then temperature derivative of the
            molar density, [mol/(m^3*K*Pa)]
        '''
        d2V_dPdT = self.d2V_dPdT()
        dV_dT = self.dV_dT()
        dV_dP = self.dV_dP()
        V = self.V()
        return -d2V_dPdT/V**2 + 2*dV_dT*dV_dP/V**3

    def drho_dV_T(self):
        r'''Method to calculate and return the volume derivative of
        molar density of the phase.

        .. math::
            \frac{\partial \rho}{\partial V} = -\frac{1}{V^2}

        Returns
        -------
        drho_dV_T : float
            Molar density derivative of volume, [mol^2/m^6]
        '''
        V = self.V()
        return -1.0/(V*V)

    def drho_dT_V(self):
        r'''Method to calculate and return the temperature derivative of
        molar density of the phase at constant volume.

        .. math::
            \left(\frac{\partial \rho}{\partial T}\right)_V = 0

        Returns
        -------
        drho_dT_V : float
            Temperature derivative of molar density of the phase at constant
            volume, [mol/(m^3*K)]
        '''
        return 0.0

    # Idea gas heat capacity

    def _setup_Cpigs(self, HeatCapacityGases):
        Cpgs_data = None
        Cpgs_poly_fit = all(i.method == POLY_FIT for i in HeatCapacityGases) if HeatCapacityGases is not None else False
        if Cpgs_poly_fit:
            T_REF_IG = self.T_REF_IG
            Cpgs_data = [[i.poly_fit_Tmin for i in HeatCapacityGases],
                              [i.poly_fit_Tmin_slope for i in HeatCapacityGases],
                              [i.poly_fit_Tmin_value for i in HeatCapacityGases],
                              [i.poly_fit_Tmax for i in HeatCapacityGases],
                              [i.poly_fit_Tmax_slope for i in HeatCapacityGases],
                              [i.poly_fit_Tmax_value for i in HeatCapacityGases],
                              [i.poly_fit_log_coeff for i in HeatCapacityGases],
#                              [horner(i.poly_fit_int_coeffs, i.poly_fit_Tmin) for i in HeatCapacityGases],
                              [horner(i.poly_fit_int_coeffs, i.poly_fit_Tmin) - i.poly_fit_Tmin*(0.5*i.poly_fit_Tmin_slope*i.poly_fit_Tmin + i.poly_fit_Tmin_value - i.poly_fit_Tmin_slope*i.poly_fit_Tmin) for i in HeatCapacityGases],
#                              [horner(i.poly_fit_int_coeffs, i.poly_fit_Tmax) for i in HeatCapacityGases],
                              [horner(i.poly_fit_int_coeffs, i.poly_fit_Tmax) - horner(i.poly_fit_int_coeffs, i.poly_fit_Tmin) + i.poly_fit_Tmin*(0.5*i.poly_fit_Tmin_slope*i.poly_fit_Tmin + i.poly_fit_Tmin_value - i.poly_fit_Tmin_slope*i.poly_fit_Tmin) for i in HeatCapacityGases],
#                              [horner_log(i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmin) for i in HeatCapacityGases],
                              [horner_log(i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmin) -(i.poly_fit_Tmin_slope*i.poly_fit_Tmin + (i.poly_fit_Tmin_value - i.poly_fit_Tmin_slope*i.poly_fit_Tmin)*log(i.poly_fit_Tmin)) for i in HeatCapacityGases],
#                              [horner_log(i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmax) for i in HeatCapacityGases],
                              [(horner_log(i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmax)
                                - horner_log(i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmin)
                                + (i.poly_fit_Tmin_slope*i.poly_fit_Tmin + (i.poly_fit_Tmin_value - i.poly_fit_Tmin_slope*i.poly_fit_Tmin)*log(i.poly_fit_Tmin))
                                - (i.poly_fit_Tmax_value -i.poly_fit_Tmax*i.poly_fit_Tmax_slope)*log(i.poly_fit_Tmax)) for i in HeatCapacityGases],
                              [poly_fit_integral_value(T_REF_IG, i.poly_fit_int_coeffs, i.poly_fit_Tmin,
                                                       i.poly_fit_Tmax, i.poly_fit_Tmin_value,
                                                       i.poly_fit_Tmax_value, i.poly_fit_Tmin_slope,
                                                       i.poly_fit_Tmax_slope) for i in HeatCapacityGases],
                              [i.poly_fit_coeffs for i in HeatCapacityGases],
                              [i.poly_fit_int_coeffs for i in HeatCapacityGases],
                              [i.poly_fit_T_int_T_coeffs for i in HeatCapacityGases],
                              [poly_fit_integral_over_T_value(T_REF_IG, i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmin,
                                                       i.poly_fit_Tmax, i.poly_fit_Tmin_value,
                                                       i.poly_fit_Tmax_value, i.poly_fit_Tmin_slope,
                                                       i.poly_fit_Tmax_slope) for i in HeatCapacityGases],

                              ]
        return (Cpgs_poly_fit, Cpgs_data)


    def _Cp_pure_fast(self, Cps_data):
        Cps = []
        T, cmps = self.T, range(self.N)
        Tmins, Tmaxs, coeffs = Cps_data[0], Cps_data[3], Cps_data[12]
        Tmin_slopes = Cps_data[1]
        Tmin_values = Cps_data[2]
        Tmax_slopes = Cps_data[4]
        Tmax_values = Cps_data[5]

        for i in cmps:
            if T < Tmins[i]:
                Cp = (T -  Tmins[i])*Tmin_slopes[i] + Tmin_values[i]
            elif T > Tmaxs[i]:
                Cp = (T - Tmaxs[i])*Tmax_slopes[i] + Tmax_values[i]
            else:
                Cp = 0.0
                for c in coeffs[i]:
                    Cp = Cp*T + c
            Cps.append(Cp)
        return Cps

    def _dCp_dT_pure_fast(self, Cps_data):
        dCps = []
        T, cmps = self.T, range(self.N)
        Tmins, Tmaxs, coeffs = Cps_data[0], Cps_data[3], Cps_data[12]
        Tmin_slopes = Cps_data[1]
        Tmin_values = Cps_data[2]
        Tmax_slopes = Cps_data[4]
        Tmax_values = Cps_data[5]

        for i in cmps:
            if T < Tmins[i]:
                dCp = Tmin_slopes[i]
            elif T > Tmaxs[i]:
                dCp = Tmax_slopes[i]
            else:
                Cp, dCp = 0.0, 0.0
                for c in coeffs[i]:
                    dCp = T*dCp + Cp
                    Cp = T*Cp + c
            dCps.append(dCp)
        return dCps

    def _Cp_integrals_pure_fast(self, Cps_data):
        Cp_integrals_pure = []
        T, cmps = self.T, range(self.N)
        Tmins, Tmaxes, int_coeffs = Cps_data[0], Cps_data[3], Cps_data[13]
        for i in cmps:
            # If indeed everything is working here, need to optimize to decide what to store
            # Try to save lookups to avoid cache misses
            # Instead of storing horner Tmin and Tmax, store -:
            # tot(Tmin) - Cps_data[7][i]
            # and tot1 + tot for the high T
            # Should save quite a bit of lookups! est. .12 go to .09
#                Tmin = Tmins[i]
#                if T < Tmin:
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    H = T*(0.5*Cps_data[1][i]*T + x1)
#                elif (T <= Tmaxes[i]):
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    tot = Tmin*(0.5*Cps_data[1][i]*Tmin + x1)
#
#                    tot1 = 0.0
#                    for c in int_coeffs[i]:
#                        tot1 = tot1*T + c
#                    tot1 -= Cps_data[7][i]
##                    tot1 = horner(int_coeffs[i], T) - horner(int_coeffs[i], Tmin)
#                    H = tot + tot1
#                else:
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    tot = Tmin*(0.5*Cps_data[1][i]*Tmin + x1)
#
#                    tot1 = Cps_data[8][i] - Cps_data[7][i]
#
#                    x1 = Cps_data[5][i] - Cps_data[4][i]*Tmaxes[i]
#                    tot2 = T*(0.5*Cps_data[4][i]*T + x1) - Tmaxes[i]*(0.5*Cps_data[4][i]*Tmaxes[i] + x1)
#                    H = tot + tot1 + tot2



            # ATTEMPT AT FAST HERE (NOW WORKING)
            if T < Tmins[i]:
                x1 = Cps_data[2][i] - Cps_data[1][i]*Tmins[i]
                H = T*(0.5*Cps_data[1][i]*T + x1)
            elif (T <= Tmaxes[i]):
                H = 0.0
                for c in int_coeffs[i]:
                    H = H*T + c
                H -= Cps_data[7][i]
            else:
                Tmax_slope = Cps_data[4][i]
                x1 = Cps_data[5][i] - Tmax_slope*Tmaxes[i]
                H = T*(0.5*Tmax_slope*T + x1) - Tmaxes[i]*(0.5*Tmax_slope*Tmaxes[i] + x1)
                H += Cps_data[8][i]

            Cp_integrals_pure.append(H - Cps_data[11][i])
        return Cp_integrals_pure

    def _Cp_integrals_over_T_pure_fast(self, Cps_data):
        Cp_integrals_over_T_pure = []
        T, cmps = self.T, range(self.N)
        Tmins, Tmaxes, T_int_T_coeffs = Cps_data[0], Cps_data[3], Cps_data[14]
        logT = log(T)
        for i in cmps:
            Tmin = Tmins[i]
            if T < Tmin:
                x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
                S = (Cps_data[1][i]*T + x1*logT)
            elif (Tmin <= T <= Tmaxes[i]):
                S = 0.0
                for c in T_int_T_coeffs[i]:
                    S = S*T + c
                S += Cps_data[6][i]*logT
                # The below should be in a constant - taking the place of Cps_data[9]
                S -= Cps_data[9][i]
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    S += (Cps_data[1][i]*Tmin + x1*log(Tmin))
            else:
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    S = (Cps_data[1][i]*Tmin + x1*log(Tmin))
#                    S += (Cps_data[10][i] - Cps_data[9][i])
                S = Cps_data[10][i]
                # The above should be in the constant Cps_data[10], - x2*log(Tmaxes[i]) also
                x2 = Cps_data[5][i] - Tmaxes[i]*Cps_data[4][i]
                S += -Cps_data[4][i]*(Tmaxes[i] - T) + x2*logT #- x2*log(Tmaxes[i])

            Cp_integrals_over_T_pure.append(S - Cps_data[15][i])
        return Cp_integrals_over_T_pure

    def Cpigs_pure(self):
        r'''Method to calculate and return the ideal-gas heat capacities of
        every component in the phase. This method is powered by the
        `HeatCapacityGases` objects, except when all components have the same
        heat capacity form and a fast implementation has been written for it
        (currently only polynomials).

        Returns
        -------
        Cp_ig : list[float]
            Molar ideal gas heat capacities, [J/(mol*K)]
        '''
        try:
            return self._Cpigs
        except AttributeError:
            pass
        if self.Cpgs_poly_fit:
            self._Cpigs = self._Cp_pure_fast(self._Cpgs_data)
            return self._Cpigs

        T = self.T
        self._Cpigs = [i.T_dependent_property(T) for i in self.HeatCapacityGases]
        return self._Cpigs

    def Cpig_integrals_pure(self):
        r'''Method to calculate and return the integrals of the ideal-gas heat
        capacities of every component in the phase from a temperature of
        :obj:`Phase.T_REF_IG` to the system temperature. This method is powered by the
        `HeatCapacityGases` objects, except when all components have the same
        heat capacity form and a fast implementation has been written for it
        (currently only polynomials).

        .. math::
            \Delta H^{ig} = \int^T_{T_{ref}} C_p^{ig} dT

        Returns
        -------
        dH_ig : list[float]
            Integrals of ideal gas heat capacity from the reference
            temperature to the system temperature, [J/(mol)]
        '''
        try:
            return self._Cpig_integrals_pure
        except AttributeError:
            pass
        if self.Cpgs_poly_fit:
            self._Cpig_integrals_pure = self._Cp_integrals_pure_fast(self._Cpgs_data)
            return self._Cpig_integrals_pure

        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        self._Cpig_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        return self._Cpig_integrals_pure

    def Cpig_integrals_over_T_pure(self):
        r'''Method to calculate and return the integrals of the ideal-gas heat
        capacities divided by temperature of every component in the phase from
        a temperature of :obj:`Phase.T_REF_IG` to the system temperature.
        This method is powered by the
        `HeatCapacityGases` objects, except when all components have the same
        heat capacity form and a fast implementation has been written for it
        (currently only polynomials).

        .. math::
            \Delta S^{ig} = \int^T_{T_{ref}} \frac{C_p^{ig}}{T} dT

        Returns
        -------
        dS_ig : list[float]
            Integrals of ideal gas heat capacity over temperature from the
            reference temperature to the system temperature, [J/(mol)]
        '''
        try:
            return self._Cpig_integrals_over_T_pure
        except AttributeError:
            pass

        if self.Cpgs_poly_fit:
            self._Cpig_integrals_over_T_pure = self._Cp_integrals_over_T_pure_fast(self._Cpgs_data)
            return self._Cpig_integrals_over_T_pure


        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        self._Cpig_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        return self._Cpig_integrals_over_T_pure

    def dCpigs_dT_pure(self):
        r'''Method to calculate and return the first temperature derivative of
        ideal-gas heat capacities of every component in the phase. This method
        is powered by the `HeatCapacityGases` objects, except when all
        components have the same heat capacity form and a fast implementation
        has been written for it (currently only polynomials).

        .. math::
            \frac{\partial C_p^{ig}}{\partial T}

        Returns
        -------
        dCp_ig_dT : list[float]
            First temperature derivatives of molar ideal gas heat capacities,
            [J/(mol*K^2)]
        '''
        try:
            return self._dCpigs_dT
        except AttributeError:
            pass
        if self.Cpgs_poly_fit:
            self._dCpigs_dT = self._dCp_dT_pure_fast(self._Cpgs_data)
            return self._dCpigs_dT

        T = self.T
        self._dCpigs_dT = [i.T_dependent_property_derivative(T) for i in self.HeatCapacityGases]
        return self._dCpigs_dT


    def _Cpls_pure(self):
        try:
            return self._Cpls
        except AttributeError:
            pass
        if self.Cpls_poly_fit:
            self._Cpls = self._Cp_pure_fast(self._Cpls_data)
            return self._Cpls

        T = self.T
        self._Cpls = [i.T_dependent_property(T) for i in self.HeatCapacityLiquids]
        return self._Cpls

    def _Cpl_integrals_pure(self):
        try:
            return self._Cpl_integrals_pure
        except AttributeError:
            pass
#        def to_quad(T, i):
#            l2 = self.to_TP_zs(T, self.P, self.zs)
#            return l2._Cpls_pure()[i] + (l2.Vms_sat()[i] - T*l2.dVms_sat_dT()[i])*l2.dPsats_dT()[i]
#        from scipy.integrate import quad
#        vals = [float(quad(to_quad, self.T_REF_IG, self.T, args=i)[0]) for i in range(self.N)]
##        print(vals, self._Cp_integrals_pure_fast(self._Cpls_data))
#        return vals

        if self.Cpls_poly_fit:
            self._Cpl_integrals_pure = self._Cp_integrals_pure_fast(self._Cpls_data)
            return self._Cpl_integrals_pure

        T, T_REF_IG, HeatCapacityLiquids = self.T, self.T_REF_IG, self.HeatCapacityLiquids
        self._Cpl_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityLiquids]
        return self._Cpl_integrals_pure

    def _Cpl_integrals_over_T_pure(self):
        try:
            return self._Cpl_integrals_over_T_pure
        except AttributeError:
            pass
#        def to_quad(T, i):
#            l2 = self.to_TP_zs(T, self.P, self.zs)
#            return (l2._Cpls_pure()[i] + (l2.Vms_sat()[i] - T*l2.dVms_sat_dT()[i])*l2.dPsats_dT()[i])/T
#        from scipy.integrate import quad
#        vals = [float(quad(to_quad, self.T_REF_IG, self.T, args=i)[0]) for i in range(self.N)]
##        print(vals, self._Cp_integrals_over_T_pure_fast(self._Cpls_data))
#        return vals

        if self.Cpls_poly_fit:
            self._Cpl_integrals_over_T_pure = self._Cp_integrals_over_T_pure_fast(self._Cpls_data)
            return self._Cpl_integrals_over_T_pure


        T, T_REF_IG, HeatCapacityLiquids = self.T, self.T_REF_IG, self.HeatCapacityLiquids
        self._Cpl_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                   for obj in HeatCapacityLiquids]
        return self._Cpl_integrals_over_T_pure

    def V_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas molar volume of the
        phase.

        .. math::
            V^{ig} = \frac{RT}{P}

        Returns
        -------
        V : float
            Ideal gas molar volume, [m^3/mol]
        '''
        return self.R*self.T/self.P

    def H_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas enthalpy of the phase.

        .. math::
            H^{ig} = \sum_i z_i {H_{i}^{ig}}

        Returns
        -------
        H : float
            Ideal gas enthalpy, [J/(mol)]
        '''
        try:
            return self._H_ideal_gas
        except AttributeError:
            pass
        H = 0.0
        for zi, Cp_int in zip(self.zs, self.Cpig_integrals_pure()):
            H += zi*Cp_int
        self._H_ideal_gas = H
        return H

    def S_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas entropy of the phase.

        .. math::
            S^{ig} = \sum_i z_i S_{i}^{ig} - R\ln\left(\frac{P}{P_{ref}}\right)
            - R\sum_i z_i \ln(z_i)

        Returns
        -------
        S : float
            Ideal gas molar entropy, [J/(mol*K)]
        '''
        try:
            return self._S_ideal_gas
        except AttributeError:
            pass
        Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        log_zs = self.log_zs()
        T, P, zs, cmps = self.T, self.P, self.zs, range(self.N)
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)

        for i in cmps:
            S += zs[i]*Cpig_integrals_over_T_pure[i]
        self._S_ideal_gas = S
        return S

    def Cp_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas heat capacity of the
        phase.

        .. math::
            C_p^{ig} = \sum_i z_i {C_{p,i}^{ig}}

        Returns
        -------
        Cp : float
            Ideal gas heat capacity, [J/(mol*K)]
        '''
        try:
            return self._Cp_ideal_gas
        except AttributeError:
            pass
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        for i in range(self.N):
            Cp += zs[i]*Cpigs_pure[i]
        self._Cp_ideal_gas = Cp
        return Cp

    def Cv_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas constant volume heat
        capacity of the phase.

        .. math::
            C_v^{ig} = \sum_i z_i {C_{p,i}^{ig}} - R

        Returns
        -------
        Cv : float
            Ideal gas constant volume heat capacity, [J/(mol*K)]
        '''
        try:
            Cp = self._Cp_ideal_gas
        except AttributeError:
            Cp = self.Cp_ideal_gas()
        return Cp - self.R

    def Cv_dep(self):
        r'''Method to calculate and return the difference between the actual
        `Cv` and the ideal-gas constant volume heat
        capacity :math:`C_v^{ig}` of the phase.

        .. math::
            C_v^{dep} = C_v - C_v^{ig}

        Returns
        -------
        Cv_dep : float
            Departure ideal gas constant volume heat capacity, [J/(mol*K)]
        '''
        return self.Cv() - self.Cv_ideal_gas()

    def Cp_Cv_ratio_ideal_gas(self):
        r'''Method to calculate and return the ratio of the ideal-gas heat
        capacity to its constant-volume heat capacity.

        .. math::
            \frac{C_p^{ig}}{C_v^{ig}}

        Returns
        -------
        Cp_Cv_ratio_ideal_gas : float
            Cp/Cv for the phase as an ideal gas, [-]
        '''
        return self.Cp_ideal_gas()/self.Cv_ideal_gas()

    def G_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas Gibbs free energy of
        the phase.

        .. math::
            G^{ig} = H^{ig} - T S^{ig}

        Returns
        -------
        G_ideal_gas : float
            Ideal gas free energy, [J/(mol)]
        '''
        G_ideal_gas = self.H_ideal_gas() - self.T*self.S_ideal_gas()
        return G_ideal_gas

    def U_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas internal energy of
        the phase.

        .. math::
            U^{ig} = H^{ig} - P V^{ig}

        Returns
        -------
        U_ideal_gas : float
            Ideal gas internal energy, [J/(mol)]
        '''
        U_ideal_gas = self.H_ideal_gas() - self.P*self.V_ideal_gas()
        return U_ideal_gas

    def A_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas Helmholtz energy of
        the phase.

        .. math::
            A^{ig} = U^{ig} - T S^{ig}

        Returns
        -------
        A_ideal_gas : float
            Ideal gas Helmholtz free energy, [J/(mol)]
        '''
        A_ideal_gas = self.U_ideal_gas() - self.T*self.S_ideal_gas()
        return A_ideal_gas

    def _set_mechanical_critical_point(self):
        zs = self.zs
        # Get initial guess
        try:
            try:
                Tcs, Pcs = self.Tcs, self.Pcs
            except:
                try:
                    Tcs, Pcs = self.eos_mix.Tcs, self.eos_mix.Pcs
                except:
                    Tcs, Pcs = self.constants.Tcs, self.constants.Pcs
            Pmc, Tmc = 0.0, 0.0
            for i in range(self.N):
                Pmc += Pcs[i]*zs[i]

            Tc_rts = [sqrt(Tc) for Tc in Tcs]
            for i in range(self.N):
                tot = 0.0
                for j in range(self.N):
                    tot += zs[j]*Tc_rts[j]
                Tmc += tot*Tc_rts[i]*zs[i]
        except:
            Tmc = 300.0
            Pmc = 1e6

        # Try to solve it
        solution = [None]
        def to_solve(TP):
            global new
            T, P = float(TP[0]), float(TP[1])
            new = self.to_TP_zs(T=T, P=P, zs=zs)
            errs = [new.dP_drho(), new.d2P_drho2()]
            solution[0] = new
            return errs

        jac = lambda TP: jacobian(to_solve, TP, scalar=False)
        TP, iters = newton_system(to_solve, [Tmc, Pmc], jac=jac, ytol=1e-10)
#        TP = fsolve(to_solve, [Tmc, Pmc]) # fsolve handles the discontinuities badly
        T, P = float(TP[0]), float(TP[1])
        new = solution[0]
        V = new.V()
        self._mechanical_critical_T = T
        self._mechanical_critical_P = P
        self._mechanical_critical_V = V
        return T, P, V

    def Tmc(self):
        r'''Method to calculate and return the mechanical critical temperature
        of the phase.

        Returns
        -------
        Tmc : float
            Mechanical critical temperature, [K]
        '''
        try:
            return self._mechanical_critical_T
        except:
            self._set_mechanical_critical_point()
            return self._mechanical_critical_T

    def Pmc(self):
        r'''Method to calculate and return the mechanical critical pressure
        of the phase.

        Returns
        -------
        Pmc : float
            Mechanical critical pressure, [Pa]
        '''
        try:
            return self._mechanical_critical_P
        except:
            self._set_mechanical_critical_point()
            return self._mechanical_critical_P

    def Vmc(self):
        r'''Method to calculate and return the mechanical critical volume
        of the phase.

        Returns
        -------
        Vmc : float
            Mechanical critical volume, [m^3/mol]
        '''
        try:
            return self._mechanical_critical_V
        except:
            self._set_mechanical_critical_point()
            return self._mechanical_critical_V

    def Zmc(self):
        r'''Method to calculate and return the mechanical critical
        compressibility of the phase.

        Returns
        -------
        Zmc : float
            Mechanical critical compressibility, [-]
        '''
        try:
            V = self._mechanical_critical_V
        except:
            self._set_mechanical_critical_point()
            V = self._mechanical_critical_V
        return (self.Pmc()*self.Vmc())/(self.R*self.Tmc())

    def dH_dT_P(self):
        r'''Method to calculate and return the temperature derivative of
        enthalpy of the phase at constant pressure.

        Returns
        -------
        dH_dT_P : float
            Temperature derivative of enthalpy, [J/(mol*K)]
        '''
        return self.dH_dT()

    def dH_dP_T(self):
        r'''Method to calculate and return the pressure derivative of
        enthalpy of the phase at constant pressure.

        Returns
        -------
        dH_dP_T : float
            Pressure derivative of enthalpy, [J/(mol*Pa)]
        '''
        return self.dH_dP()

    def dS_dP_T(self):
        r'''Method to calculate and return the pressure derivative of
        entropy of the phase at constant pressure.

        Returns
        -------
        dS_dP_T : float
            Pressure derivative of entropy, [J/(mol*K*Pa)]
        '''
        return self.dS_dP()

    def dS_dV_T(self):
        r'''Method to calculate and return the volume derivative of
        entropy of the phase at constant temperature.

        Returns
        -------
        dS_dV_T : float
            Volume derivative of entropy, [J/(K*m^3)]
        '''
        return self.dS_dP_T()*self.dP_dV()

    def dS_dV_P(self):
        r'''Method to calculate and return the volume derivative of
        entropy of the phase at constant pressure.

        Returns
        -------
        dS_dV_P : float
            Volume derivative of entropy, [J/(K*m^3)]
        '''
        return self.dS_dT_P()*self.dT_dV()

    def dP_dT_P(self):
        r'''Method to calculate and return the temperature derivative of
        temperature of the phase at constant pressure.

        Returns
        -------
        dP_dT_P : float
            Temperature derivative of temperature, [-]
        '''
        return 0.0

    def dP_dV_P(self):
        r'''Method to calculate and return the volume derivative of
        pressure of the phase at constant pressure.

        Returns
        -------
        dP_dV_P : float
            Volume derivative of pressure of the phase at constant pressure,
            [Pa*mol/m^3]
        '''
        return 0.0

    def dT_dP_T(self):
        r'''Method to calculate and return the pressure derivative of
        temperature of the phase at constant temperature.

        Returns
        -------
        dT_dP_T : float
            Pressure derivative of temperature of the phase at constant
            temperature, [K/Pa]
        '''
        return 0.0

    def dT_dV_T(self):
        r'''Method to calculate and return the volume derivative of
        temperature of the phase at constant temperature.

        Returns
        -------
        dT_dV_T : float
            Pressure derivative of temperature of the phase at constant
            temperature, [K*mol/m^3]
        '''
        return 0.0

    def dV_dT_V(self):
        r'''Method to calculate and return the temperature derivative of
        volume of the phase at constant volume.

        Returns
        -------
        dV_dT_V : float
             Temperature derivative of volume of the phase at constant volume,
             [m^3/(mol*K)]
        '''
        return 0.0

    def dV_dP_V(self):
        r'''Method to calculate and return the volume derivative of
        pressure of the phase at constant volume.

        Returns
        -------
        dV_dP_V : float
             Pressure derivative of volume of the phase at constant pressure,
             [m^3/(mol*Pa)]
        '''
        return 0.0

    def dP_dP_T(self):
        r'''Method to calculate and return the pressure derivative of
        pressure of the phase at constant temperature.

        Returns
        -------
        dP_dP_T : float
             Pressure derivative of pressure of the phase at constant
             temperature, [-]
        '''
        return 1.0

    def dP_dP_V(self):
        r'''Method to calculate and return the pressure derivative of
        pressure of the phase at constant volume.

        Returns
        -------
        dP_dP_V : float
             Pressure derivative of pressure of the phase at constant
             volume, [-]
        '''
        return 1.0

    def dT_dT_P(self):
        r'''Method to calculate and return the temperature derivative of
        temperature of the phase at constant pressure.

        Returns
        -------
        dT_dT_P : float
             Temperature derivative of temperature of the phase at constant
             pressure, [-]
        '''
        return 1.0

    def dT_dT_V(self):
        r'''Method to calculate and return the temperature derivative of
        temperature of the phase at constant volume.

        Returns
        -------
        dT_dT_V : float
             Temperature derivative of temperature of the phase at constant
             volume, [-]
        '''
        return 1.0

    def dV_dV_T(self):
        r'''Method to calculate and return the volume derivative of
        volume of the phase at constant temperature.

        Returns
        -------
        dV_dV_T : float
             Volume derivative of volume of the phase at constant
             temperature, [-]
        '''
        return 1.0

    def dV_dV_P(self):
        r'''Method to calculate and return the volume derivative of
        volume of the phase at constant pressure.

        Returns
        -------
        dV_dV_P : float
             Volume derivative of volume of the phase at constant
             pressure, [-]
        '''
        return 1.0

    d2T_dV2_P = d2T_dV2
    d2V_dT2_P = d2V_dT2
    d2V_dP2_T = d2V_dP2
    d2T_dP2_V = d2T_dP2
    dV_dP_T = dV_dP
    dV_dT_P = dV_dT
    dT_dP_V = dT_dP
    dT_dV_P = dT_dV


    # More derivatives - at const H, S, G, U, A
    _derivs_jacobian_x = 'V'
    _derivs_jacobian_y = 'T'

    def _derivs_jacobian(self, a, b, c, x=_derivs_jacobian_x,
                         y=_derivs_jacobian_y):
        r'''Calculates and returns a first-order derivative of one property
        with respect to another property at constant another property.

        This is particularly useful to obtain derivatives with respect to
        another property which is not an intensive variable in a model,
        allowing for example derivatives at constant enthalpy or Gibbs energy
        to be obtained. This formula is obtained from the first derivative
        principles of reciprocity, the chain rule, and the triple product rule
        as shown in [1]_.

        ... math::
            \left(\frac{\partial a}{\partial b}\right)_{c}=
            \frac{\left(\frac{\partial a}{\partial x}\right)_{y}\left(
            \frac{\partial c}{\partial y}\right)_{x}-\left(\frac{\partial a}{
            \partial y}\right)_{x}\left(\frac{\partial c}{\partial x}
            \right)_{y}}{\left(\frac{\partial b}{\partial x}\right)_{y}\left(
            \frac{\partial c}{\partial y}\right)_{x}-\left(\frac{\partial b}
            {\partial y}\right)_{x}\left(\frac{\partial c}{\partial x}
            \right)_{y}}

        References
        ----------
        .. [1] Thorade, Matthis, and Ali Saadat. "Partial Derivatives of
           Thermodynamic State Properties for Dynamic Simulation."
           Environmental Earth Sciences 70, no. 8 (April 10, 2013): 3497-3503.
           https://doi.org/10.1007/s12665-013-2394-z.
        '''
        n0 = getattr(self, 'd%s_d%s_%s'%(a, x, y))()
        n1 = getattr(self, 'd%s_d%s_%s'%(c, y, x))()

        n2 = getattr(self, 'd%s_d%s_%s'%(a, y, x))()
        n3 = getattr(self, 'd%s_d%s_%s'%(c, x, y))()

        d0 = getattr(self, 'd%s_d%s_%s'%(b, x, y))()
        d1 = getattr(self, 'd%s_d%s_%s'%(c, y, x))()

        d2 = getattr(self, 'd%s_d%s_%s'%(b, y, x))()
        d3 = getattr(self, 'd%s_d%s_%s'%(c, x, y))()

        return (n0*n1 - n2*n3)/(d0*d1 - d2*d3)


    ### Transport properties - pass them on!
    # Properties that use `constants` attributes

    def MW(self):
        r'''Method to calculate and return molecular weight of the phase.

        .. math::
            \text{MW} = \sum_i z_i \text{MW}_i

        Returns
        -------
        MW : float
             Molecular weight, [g/mol]
        '''
        try:
            return self._MW
        except AttributeError:
            pass
        zs, MWs = self.zs, self.constants.MWs
        MW = 0.0
        for i in range(self.N):
            MW += zs[i]*MWs[i]
        self._MW = MW
        return MW

    def MW_inv(self):
        r'''Method to calculate and return inverse of molecular weight of the
        phase.

        .. math::
            \frac{1}{\text{MW}} = \frac{1}{\sum_i z_i \text{MW}_i}

        Returns
        -------
        MW_inv : float
             Inverse of molecular weight, [mol/g]
        '''
        try:
            return self._MW_inv
        except AttributeError:
            pass
        self._MW_inv = MW_inv = 1.0/self.MW()
        return MW_inv

    def speed_of_sound_mass(self):
        r'''Method to calculate and return the speed of sound
        of the phase.

        .. math::
            w = \left[-V^2 \frac{1000}{MW}\left(\frac{\partial P}{\partial V}
            \right)_T \frac{C_p}{C_v}\right]^{1/2}

        Returns
        -------
        w : float
            Speed of sound for a real gas, [m/s]
        '''
        # 1000**0.5 = 31.622776601683793
        return 31.622776601683793/sqrt(self.MW())*self.speed_of_sound()

    def rho_mass(self):
        r'''Method to calculate and return mass density of the phase.

        .. math::
            \rho = \frac{MW}{1000\cdot VM}

        Returns
        -------
        rho_mass : float
            Mass density, [kg/m^3]
        '''
        try:
            return self._rho_mass
        except AttributeError:
            pass
        self._rho_mass = rho_mass = self.MW()/(1000.0*self.V())
        return rho_mass

    def drho_mass_dT(self):
        r'''Method to calculate the mass density derivative with respect to
        temperature, at constant pressure.

        .. math::
            \left(\frac{\partial \rho}{\partial T}\right)_{P} =
            \frac{-\text{MW} \frac{\partial V_m}{\partial T}}{1000 V_m^2}

        Returns
        -------
        drho_mass_dT : float
           Temperature derivative of mass density at constant pressure,
           [kg/m^3/K]

        Notes
        -----
        Requires `dV_dT`, `MW`, and `V`.

        This expression is readily obtainable with SymPy:

        >>> from sympy import * # doctest: +SKIP
        >>> T, P, MW = symbols('T, P, MW') # doctest: +SKIP
        >>> Vm = symbols('Vm', cls=Function) # doctest: +SKIP
        >>> rho_mass = (Vm(T))**-1*MW/1000 # doctest: +SKIP
        >>> diff(rho_mass, T) # doctest: +SKIP
        -MW*Derivative(Vm(T), T)/(1000*Vm(T)**2)
        '''
        try:
            return self._drho_mass_dT
        except AttributeError:
            pass
        MW = self.MW()
        V = self.V()
        dV_dT = self.dV_dT()
        self._drho_mass_dT = drho_mass_dT = -MW*dV_dT/(1000.0*V*V)
        return drho_mass_dT

    def drho_mass_dP(self):
        r'''Method to calculate the mass density derivative with respect to
        pressure, at constant temperature.

        .. math::
            \left(\frac{\partial \rho}{\partial P}\right)_{T} =
            \frac{-\text{MW} \frac{\partial V_m}{\partial P}}{1000 V_m^2}

        Returns
        -------
        drho_mass_dP : float
           Pressure derivative of mass density at constant temperature,
           [kg/m^3/Pa]

        Notes
        -----
        Requires `dV_dP`, `MW`, and `V`.

        This expression is readily obtainable with SymTy:

        >>> from sympy import * # doctest: +SKIP
        >>> P, T, MW = symbols('P, T, MW') # doctest: +SKIP
        >>> Vm = symbols('Vm', cls=Function) # doctest: +SKIP
        >>> rho_mass = (Vm(P))**-1*MW/1000 # doctest: +SKIP
        >>> diff(rho_mass, P) # doctest: +SKIP
        -MW*Derivative(Vm(P), P)/(1000*Vm(P)**2)
        '''
        try:
            return self._drho_mass_dP
        except AttributeError:
            pass
        MW = self.MW()
        V = self.V()
        dV_dP = self.dV_dP()
        self._drho_mass_dP = drho_mass_dP = -MW*dV_dP/(1000.0*V*V)
        return drho_mass_dP

    def H_mass(self):
        r'''Method to calculate and return mass enthalpy of the phase.

        .. math::
            H_{mass} = \frac{1000 H_{molar}}{MW}

        Returns
        -------
        H_mass : float
            Mass enthalpy, [J/kg]
        '''
        try:
            return self._H_mass
        except AttributeError:
            pass

        self._H_mass = H_mass = self.H()*1e3*self.MW_inv()
        return H_mass

    def S_mass(self):
        r'''Method to calculate and return mass entropy of the phase.

        .. math::
            S_{mass} = \frac{1000 S_{molar}}{MW}

        Returns
        -------
        S_mass : float
            Mass enthalpy, [J/(kg*K)]
        '''
        try:
            return self._S_mass
        except AttributeError:
            pass

        self._S_mass = S_mass = self.S()*1e3*self.MW_inv()
        return S_mass

    def U_mass(self):
        r'''Method to calculate and return mass internal energy of the phase.

        .. math::
            U_{mass} = \frac{1000 U_{molar}}{MW}

        Returns
        -------
        U_mass : float
            Mass internal energy, [J/(kg)]
        '''
        try:
            return self._U_mass
        except AttributeError:
            pass

        self._U_mass = U_mass = self.U()*1e3*self.MW_inv()
        return U_mass

    def A_mass(self):
        r'''Method to calculate and return mass Helmholtz energy of the phase.

        .. math::
            A_{mass} = \frac{1000 A_{molar}}{MW}

        Returns
        -------
        A_mass : float
            Mass Helmholtz energy, [J/(kg)]
        '''
        try:
            return self._A_mass
        except AttributeError:
            pass

        self._A_mass = A_mass = self.A()*1e3*self.MW_inv()
        return A_mass

    def G_mass(self):
        r'''Method to calculate and return mass Gibbs energy of the phase.

        .. math::
            G_{mass} = \frac{1000 G_{molar}}{MW}

        Returns
        -------
        G_mass : float
            Mass Gibbs energy, [J/(kg)]
        '''
        try:
            return self._G_mass
        except AttributeError:
            pass

        self._G_mass = G_mass = self.G()*1e3*self.MW_inv()
        return G_mass

    def Cp_mass(self):
        r'''Method to calculate and return mass constant pressure heat capacity
        of the phase.

        .. math::
            Cp_{mass} = \frac{1000 Cp_{molar}}{MW}

        Returns
        -------
        Cp_mass : float
            Mass heat capacity, [J/(kg*K)]
        '''
        try:
            return self._Cp_mass
        except AttributeError:
            pass

        self._Cp_mass = Cp_mass = self.Cp()*1e3*self.MW_inv()
        return Cp_mass

    def Cv_mass(self):
        r'''Method to calculate and return mass constant volume heat capacity
        of the phase.

        .. math::
            Cv_{mass} = \frac{1000 Cv_{molar}}{MW}

        Returns
        -------
        Cv_mass : float
            Mass constant volume heat capacity, [J/(kg*K)]
        '''
        try:
            return self._Cv_mass
        except AttributeError:
            pass

        self._Cv_mass = Cv_mass = self.Cv()*1e3*self.MW_inv()
        return Cv_mass

    def P_transitions(self):
        r'''Dummy method. The idea behind this method is to calculate any
        pressures (at constant temperature) which cause the phase properties to
        become discontinuous.

        Returns
        -------
        P_transitions : list[float]
            Transition pressures, [Pa]
        '''
        return []

    def T_max_at_V(self, V):
        r'''Method to calculate the maximum temperature the phase can create at a
        constant volume, if one exists; returns None otherwise.

        Parameters
        ----------
        V : float
            Constant molar volume, [m^3/mol]
        Pmax : float
            Maximum possible isochoric pressure, if already known [Pa]

        Returns
        -------
        T : float
            Maximum possible temperature, [K]

        Notes
        -----
        '''
        return None

    def P_max_at_V(self, V):
        r'''Dummy method. The idea behind this method, which is implemented by some
        subclasses, is to calculate the maximum pressure the phase can create at a
        constant volume, if one exists; returns None otherwise. This method,
        as a dummy method, always returns None.

        Parameters
        ----------
        V : float
            Constant molar volume, [m^3/mol]

        Returns
        -------
        P : float
            Maximum possible isochoric pressure, [Pa]
        '''
        return None

    def dspeed_of_sound_dT_P(self):
        r'''Method to calculate the temperature derivative of speed of sound
        at constant pressure in molar units.

        .. math::
            \left(\frac{\partial c}{\partial T}\right)_P =
            - \frac{\sqrt{- \frac{\operatorname{Cp}{\left(T \right)} V^{2}
            {\left(T \right)} \operatorname{dPdV_{T}}{\left(T \right)}}
            {\operatorname{Cv}{\left(T \right)}}} \left(- \frac{\operatorname{Cp}
            {\left(T \right)} V^{2}{\left(T \right)} \frac{d}{d T}
            \operatorname{dPdV_{T}}{\left(T \right)}}{2 \operatorname{Cv}{\left(T
            \right)}} - \frac{\operatorname{Cp}{\left(T \right)} V{\left(T
            \right)} \operatorname{dPdV_{T}}{\left(T \right)} \frac{d}{d T}
            V{\left(T \right)}}{\operatorname{Cv}{\left(T \right)}}
            + \frac{\operatorname{Cp}{\left(T \right)} V^{2}{\left(T \right)}
            \operatorname{dPdV_{T}}{\left(T \right)} \frac{d}{d T}
            \operatorname{Cv}{\left(T \right)}}{2 \operatorname{Cv}^{2}
            {\left(T \right)}} - \frac{V^{2}{\left(T \right)} \operatorname{
            dPdV_{T}}{\left(T \right)} \frac{d}{d T} \operatorname{Cp}{\left(T
            \right)}}{2 \operatorname{Cv}{\left(T \right)}}\right)
            \operatorname{Cv}{\left(T \right)}}{\operatorname{Cp}{\left(T
            \right)} V^{2}{\left(T \right)} \operatorname{dPdV_{T}}{\left(T
            \right)}}

        Returns
        -------
        dspeed_of_sound_dT_P : float
           Temperature derivative of speed of sound at constant pressure,
           [m*kg^0.5/s/mol^0.5/K]

        Notes
        -----
        Requires the temperature derivative of Cp and Cv both at constant
        pressure, as wel as the volume and temperature derivative of pressure,
        calculated at constant temperature and then pressure respectively.
        These can be tricky to obtain.
        '''
        '''Calculation with SymPy:
        from sympy import *
        T = symbols('T')
        V, dPdV_T, Cp, Cv = symbols('V, dPdV_T, Cp, Cv', cls=Function)
        c = sqrt(-V(T)**2*dPdV_T(T)*Cp(T)/Cv(T))
        '''
        x0 = self.Cp()
        x1 = self.V()
        x2 = self.dP_dV()
        x3 = self.Cv()
        x4 = x0*x2
        x5 = x4/x3
        x6 = 0.5*x1

        x50 = self.d2P_dVdT_TP()
        x51 = self.d2H_dT2()
        x52 = self.dV_dT()
        x53 = self.dCv_dT_P()

        return (-x1*x1*x5)**0.5*(x0*x6*x50 + x2*x6*x51 + x4*x52- x5*x6*x53)/(x0*x1*x2)

    def dspeed_of_sound_dP_T(self):
        r'''Method to calculate the pressure derivative of speed of sound
        at constant temperature in molar units.

        .. math::
            \left(\frac{\partial c}{\partial P}\right)_T =
            - \frac{\sqrt{- \frac{\operatorname{Cp}{\left(P \right)} V^{2}
            {\left(P \right)} \operatorname{dPdV_{T}}{\left(P \right)}}
            {\operatorname{Cv}{\left(P \right)}}} \left(- \frac{
            \operatorname{Cp}{\left(P \right)} V^{2}{\left(P \right)} \frac{d}
            {d P} \operatorname{dPdV_{T}}{\left(P \right)}}{2 \operatorname{Cv}
            {\left(P \right)}} - \frac{\operatorname{Cp}{\left(P \right)}
            V{\left(P \right)} \operatorname{dPdV_{T}}{\left(P \right)}
            \frac{d}{d P} V{\left(P \right)}}{\operatorname{Cv}{\left(P \right)
            }} + \frac{\operatorname{Cp}{\left(P \right)} V^{2}{\left(P \right)
            } \operatorname{dPdV_{T}}{\left(P \right)} \frac{d}{d P}
            \operatorname{Cv}{\left(P \right)}}{2 \operatorname{Cv}^{2}{\left(P
            \right)}} - \frac{V^{2}{\left(P \right)} \operatorname{dPdV_{T}}
            {\left(P \right)} \frac{d}{d P} \operatorname{Cp}{\left(P \right)}}
            {2 \operatorname{Cv}{\left(P \right)}}\right) \operatorname{Cv}
            {\left(P \right)}}{\operatorname{Cp}{\left(P \right)} V^{2}{\left(P
            \right)} \operatorname{dPdV_{T}}{\left(P \right)}}

        Returns
        -------
        dspeed_of_sound_dP_T : float
           Pressure derivative of speed of sound at constant temperature,
           [m*kg^0.5/s/mol^0.5/Pa]

        Notes
        -----
        '''
        '''
        from sympy import *
        P = symbols('P')
        V, dPdV_T, Cp, Cv = symbols('V, dPdV_T, Cp, Cv', cls=Function)
        c = sqrt(-V(P)**2*dPdV_T(P)*Cp(P)/Cv(P))
        print(latex(diff(c, P)))
        '''
        x0 = self.Cp()
        x1 = self.V()
        x2 = self.dP_dV()
        x3 = self.Cv()
        x4 = x0*x2
        x5 = x4/x3
        x6 = 0.5*x1

        x50 = self.d2P_dVdP()
        x51 = self.d2H_dTdP()
        x52 = self.dV_dP()
        x53 = self.dCv_dP_T()

        return (-x1*x1*x5)**0.5*(x0*x6*x50 + x2*x6*x51 + x4*x52- x5*x6*x53)/(x0*x1*x2)


    # Transport properties
    def mu(self):
        if isinstance(self, gas_phases):
            return self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        elif isinstance(self, liquid_phases):
            return self.correlations.ViscosityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        else:
            raise NotImplementedError("Did not work")

    def ws(self):
        r'''Method to calculate and return the mass fractions of the phase, [-]

        Returns
        -------
        ws : list[float]
            Mass fractions, [-]

        Notes
        -----
        '''
        try:
            return self._ws
        except AttributeError:
            pass
        MWs = self.constants.MWs
        zs, cmps = self.zs, range(self.N)
        ws = [zs[i]*MWs[i] for i in cmps]
        Mavg = 1.0/sum(ws)
        for i in cmps:
            ws[i] *= Mavg
        self._ws = ws
        return ws

    def sigma(self):
        r'''Calculate and return the surface tension of the phase.
        For details of the implementation, see
        :obj:`SurfaceTensionMixture <thermo.interface.SurfaceTensionMixture>`.

        This property is strictly the ideal-gas to liquid surface tension,
        not a true inter-phase property.

        Returns
        -------
        sigma : float
            Surface tension, [N/m]
        '''
        try:
            return self._sigma
        except AttributeError:
            pass
        try:
            phase == self.assigned_phase
        except:
            if self.is_liquid:
                phase = 'l'
            else:
                phase = 'g'
        if phase == 'g':
            return None
        elif phase == 'l':
            sigma = self.correlations.SurfaceTensionMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._sigma = sigma
        return sigma

    @property
    def beta(self):
        r'''Method to return the phase fraction of this phase.
        This method is only
        available when the phase is linked to an EquilibriumState.

        Returns
        -------
        beta : float
            Phase fraction on a molar basis, [-]

        Notes
        -----
        '''
        try:
            result = self.result
        except:
            return None
        for i, p in enumerate(result.phases):
            if p is self:
                return result.betas[i]

    @property
    def beta_mass(self):
        r'''Method to return the mass phase fraction of this phase.
        This method is only
        available when the phase is linked to an EquilibriumState.

        Returns
        -------
        beta_mass : float
            Phase fraction on a mass basis, [-]

        Notes
        -----
        '''
        try:
            result = self.result
        except:
            return None
        for i, p in enumerate(result.phases):
            if p is self:
                return result.betas_mass[i]

    @property
    def beta_volume(self):
        r'''Method to return the volumetric phase fraction of this phase.
        This method is only
        available when the phase is linked to an EquilibriumState.

        Returns
        -------
        beta_volume : float
            Phase fraction on a volumetric basis, [-]

        Notes
        -----
        '''
        try:
            result = self.result
        except:
            return None
        for i, p in enumerate(result.phases):
            if p is self:
                return result.betas_volume[i]

    @property
    def VF(self):
        r'''Method to return the vapor fraction of the phase.
        If no vapor/gas is present, 0 is always returned. This method is only
        available when the phase is linked to an EquilibriumState.

        Returns
        -------
        VF : float
            Vapor fraction, [-]

        Notes
        -----
        '''
        return self.result.gas_beta


derivatives_jacobian = []

prop_iter = (('T', 'P', 'V', 'rho'), ('T', 'P', 'V', r'\rho'), ('K', 'Pa', 'm^3/mol', 'mol/m^3'), ('temperature', 'pressure', 'volume', 'density'))
for a, a_str, a_units, a_name in zip(*prop_iter):
    for b, b_str, b_units, b_name in zip(*prop_iter):
        for c, c_name in zip(('H', 'S', 'G', 'U', 'A'), ('enthalpy', 'entropy', 'Gibbs energy', 'internal energy', 'Helmholtz energy')):
            def _der(self, property=a, differentiate_by=b, at_constant=c):
                return self._derivs_jacobian(a=property, b=differentiate_by, c=at_constant)
            t = 'd%s_d%s_%s' %(a, b, c)
            doc = r'''Method to calculate and return the %s derivative of %s of the phase at constant %s.

    .. math::
        \left(\frac{\partial %s}{\partial %s}\right)_{%s}

Returns
-------
%s : float
    The %s derivative of %s of the phase at constant %s, [%s/%s]
''' %(b_name, a_name, c_name, a_str, b_str, c, t, b_name, a_name, c_name, a_units, b_units)
            setattr(Phase, t, _der)
            try:
                _der.__doc__ = doc
            except:
                pass
            derivatives_jacobian.append(t)

derivatives_thermodynamic = ['dA_dP', 'dA_dP_T', 'dA_dP_V', 'dA_dT', 'dA_dT_P', 'dA_dT_V', 'dA_dV_P', 'dA_dV_T',
             'dCv_dP_T', 'dCv_dT_P', 'dG_dP', 'dG_dP_T', 'dG_dP_V', 'dG_dT', 'dG_dT_P', 'dG_dT_V',
             'dG_dV_P', 'dG_dV_T', 'dH_dP', 'dH_dP_T', 'dH_dP_V', 'dH_dT', 'dH_dT_P', 'dH_dT_V',
             'dH_dV_P', 'dH_dV_T', 'dS_dP', 'dS_dP_T', 'dS_dP_V', 'dS_dT', 'dS_dT_P', 'dS_dT_V',
             'dS_dV_P', 'dS_dV_T', 'dU_dP', 'dU_dP_T', 'dU_dP_V', 'dU_dT', 'dU_dT_P', 'dU_dT_V',
             'dU_dV_P', 'dU_dV_T']
derivatives_thermodynamic_mass = []

prop_names = {'A' : 'Helmholtz energy',
              'G': 'Gibbs free energy',
              'U': 'internal energy',
              'H': 'enthalpy',
              'S': 'entropy',
              'T': 'temperature',
              'P': 'pressure',
              'V': 'volume', 'Cv': 'Constant-volume heat capacity'}
prop_units = {'Cv': 'J/(mol*K)', 'A': 'J/mol', 'G': 'J/mol', 'H': 'J/mol', 'S': 'J/(mol*K)', 'U': 'J/mol', 'T': 'K', 'P': 'Pa', 'V': 'm^3/mol'}
for attr in derivatives_thermodynamic:
    def _der(self, prop=attr):
        return getattr(self, prop)()*1e3*self.MW_inv()
    try:
        base, end = attr.split('_', maxsplit=1)
    except:
        splits = attr.split('_')
        base = splits[0]
        end = '_'.join(splits[1:])

    vals = attr.replace('d', '').split('_')
    try:
        prop, diff_by, at_constant = vals
    except:
        prop, diff_by = vals
        at_constant = 'T' if diff_by == 'P' else 'P'
    s = '%s_mass_%s' %(base, end)

    doc = r'''Method to calculate and return the %s derivative of mass %s of the phase at constant %s.

    .. math::
        \left(\frac{\partial %s_{\text{mass}}}{\partial %s}\right)_{%s}

Returns
-------
%s : float
    The %s derivative of mass %s of the phase at constant %s, [%s/%s]
''' %(prop_names[diff_by], prop_names[prop], prop_names[at_constant], prop, diff_by, at_constant, s, prop_names[diff_by], prop_names[prop], prop_names[at_constant], prop_units[prop], prop_units[diff_by])
    try:
        _der.__doc__ = doc#'Automatically generated derivative. %s %s' %(base, end)
    except:
        pass
    setattr(Phase, s, _der)
    derivatives_thermodynamic_mass.append(s)
del prop_names, prop_units


class IdealGas(Phase):
    r'''Class for representing an ideal gas as a phase object. All departure
    properties are zero.

    .. math::
        P = \frac{RT}{V}

    Parameters
    ----------
    HeatCapacityGases : list[HeatCapacityGas]
        Objects proiding pure-component heat capacity correlations, [-]
    Hfs : list[float]
        Molar ideal-gas standard heats of formation at 298.15 K and 1 atm,
        [J/mol]
    Gfs : list[float]
        Molar ideal-gas standard Gibbs energies of formation at 298.15 K and
        1 atm, [J/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    zs : list[float], optional
        Mole fractions of each component, [-]

    Examples
    --------
    T-P initialization for oxygen and nitrogen, using Poling's polynomial heat
    capacities:

    >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
    ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
    >>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
    >>> phase.Cp()
    29.1733530

    '''
    '''DO NOT DELETE - EOS CLASS IS TOO SLOW!
    This will be important for fitting.

    '''
    phase = 'g'
    force_phase = 'g'
    is_gas = True
    is_liquid = False
    composition_independent = True
    ideal_gas_basis = True

    pure_references = ('HeatCapacityGases',)
    pure_reference_types = (HeatCapacityGas,)
    __full_path__ = "%s.%s" %(__module__, __qualname__)

    model_attributes = ('Hfs', 'Gfs', 'Sfs') + pure_references

    def __init__(self, HeatCapacityGases=None, Hfs=None, Gfs=None, T=None, P=None, zs=None):
        self.HeatCapacityGases = HeatCapacityGases
        self.Hfs = Hfs
        self.Gfs = Gfs
        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            self.Sfs = [(Hfi - Gfi)/298.15 for Hfi, Gfi in zip(Hfs, Gfs)]
        else:
            self.Sfs = None

        if zs is not None:
            self.N = N = len(zs)
            self.zeros1d = [0.0]*N
            self.ones1d = [1.0]*N
        elif HeatCapacityGases is not None:
            self.N = N = len(HeatCapacityGases)
            self.zeros1d = [0.0]*N
            self.ones1d = [1.0]*N
        if zs is not None:
            self.zs = zs
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P

    def __repr__(self):
        r'''Method to create a string representation of the phase object, with
        the goal of making it easy to obtain standalone code which reproduces
        the current state of the phase. This is extremely helpful in creating
        new test cases.

        Returns
        -------
        recreation : str
            String which is valid Python and recreates the current state of
            the object if ran, [-]

        Examples
        --------
        >>> from thermo import HeatCapacityGas, IdealGas
        >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
        ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
        >>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
        >>> phase
        IdealGas(HeatCapacityGases=[HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-8.231317991971707e-12, 1.3053706310500586e-08, 5.820123832707268e-07, -0.0021700747433379955, 29.424883205644317])), HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [1.48828880864943e-11, -4.9886775708919434e-08, 5.4709164027448316e-05, -0.014916145936966912, 30.18149930389626]))], T=300, P=100000.0, zs=[0.79, 0.21])

        '''
        Cpgs = ', '.join(str(o) for o in self.HeatCapacityGases)
        base = 'IdealGas(HeatCapacityGases=[%s], '  %(Cpgs,)
        for s in ('Hfs', 'Gfs', 'Sfs', 'T', 'P', 'zs'):
            if hasattr(self, s) and getattr(self, s) is not None:
                base += '%s=%s, ' %(s, getattr(self, s))
        if base[-2:] == ', ':
            base = base[:-2]
        base += ')'
        return base

    def fugacities(self):
        r'''Method to calculate and return the fugacities of each
        component in the phase.

        .. math::
            \text{fugacitiy}_i = z_i P

        Returns
        -------
        fugacities : list[float]
            Fugacities, [Pa]

        Examples
        --------
        >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
        ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
        >>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
        >>> phase.fugacities()
        [79000.0, 21000.0]
        '''
        P = self.P
        return [P*zi for zi in self.zs]

    def lnphis(self):
        r'''Method to calculate and return the log of fugacity coefficients of
        each component in the phase.

        .. math::
            \ln \phi_i = 0.0

        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
        '''
        return self.zeros1d

    lnphis_G_min = lnphis

    def phis(self):
        r'''Method to calculate and return the fugacity coefficients of
        each component in the phase.

        .. math::
             \phi_i = 1

        Returns
        -------
        phis : list[float]
            Fugacity fugacity coefficients, [-]
        '''
        return self.ones1d

    def dphis_dT(self):
        r'''Method to calculate and return the temperature derivative of
        fugacity coefficients of each component in the phase.

        .. math::
             \frac{\partial \phi_i}{\partial T} = 0

        Returns
        -------
        dphis_dT : list[float]
            Temperature derivative of fugacity fugacity coefficients, [1/K]
        '''
        return self.zeros1d

    def dphis_dP(self):
        r'''Method to calculate and return the pressure derivative of
        fugacity coefficients of each component in the phase.

        .. math::
             \frac{\partial \phi_i}{\partial P} = 0

        Returns
        -------
        dphis_dP : list[float]
            Pressure derivative of fugacity fugacity coefficients, [1/Pa]
        '''
        return self.zeros1d

    def dlnphis_dT(self):
        r'''Method to calculate and return the temperature derivative of the
        log of fugacity coefficients of each component in the phase.

        .. math::
             \frac{\partial \ln \phi_i}{\partial T} = 0

        Returns
        -------
        dlnphis_dT : list[float]
            Log fugacity coefficients, [1/K]
        '''
        return self.zeros1d

    def dlnphis_dP(self):
        r'''Method to calculate and return the pressure derivative of the
        log of fugacity coefficients of each component in the phase.

        .. math::
             \frac{\partial \ln \phi_i}{\partial P} = 0

        Returns
        -------
        dlnphis_dP : list[float]
            Log fugacity coefficients, [1/Pa]
        '''
        return self.zeros1d

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        new.N = self.N
        new.zeros1d = self.zeros1d
        new.ones1d = self.ones1d

        new.HeatCapacityGases = self.HeatCapacityGases
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        if T is not None and V is not None:
            P = R*T/V
        elif P is not None and V is not None:
            T = P*V/R
        elif T is not None and P is not None:
            pass
        else:
            raise ValueError("Two of T, P, or V are needed")
        new.P = P
        new.T = T

        new.zs = zs
        new.N = self.N
        new.zeros1d = self.zeros1d
        new.ones1d = self.ones1d

        new.HeatCapacityGases = self.HeatCapacityGases
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs

        return new


    ### Volumetric properties
    def V(self):
        r'''Method to calculate and return the molar volume of the phase.

        .. math::
             V = \frac{RT}{P}

        Returns
        -------
        V : float
            Molar volume, [m^3/mol]
        '''
        return R*self.T/self.P

    def dP_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        pressure of the phase.

        .. math::
             \frac{\partial P}{\partial T} = \frac{P}{T}

        Returns
        -------
        dP_dT : float
            First temperature derivative of pressure, [Pa/K]
        '''
        return self.P/self.T
    dP_dT_V = dP_dT

    def dP_dV(self):
        r'''Method to calculate and return the first volume derivative of
        pressure of the phase.

        .. math::
             \frac{\partial P}{\partial V} = \frac{-P^2}{RT}

        Returns
        -------
        dP_dV : float
            First volume derivative of pressure, [Pa*mol/m^3]
        '''
        return -self.P*self.P/(R*self.T)

    dP_dV_T = dP_dV

    def d2P_dT2(self):
        r'''Method to calculate and return the second temperature derivative of
        pressure of the phase.

        .. math::
             \frac{\partial^2 P}{\partial T^2} = 0

        Returns
        -------
        d2P_dT2 : float
            Second temperature derivative of pressure, [Pa/K^2]
        '''
        return 0.0
    d2P_dT2_V = d2P_dT2

    def d2P_dV2(self):
        r'''Method to calculate and return the second volume derivative of
        pressure of the phase.

        .. math::
             \frac{\partial^2 P}{\partial V^2} = \frac{2P^3}{R^2T^2}

        Returns
        -------
        d2P_dV2 : float
            Second volume derivative of pressure, [Pa*mol^2/m^6]
        '''
        P, T = self.P, self.T
        return 2.0*P*P*P/(R2*T*T)

    d2P_dV2_T = d2P_dV2

    def d2P_dTdV(self):
        r'''Method to calculate and return the second derivative of
        pressure with respect to temperature and volume of the phase.

        .. math::
             \frac{\partial^2 P}{\partial V \partial T} = \frac{-P^2}{RT^2}

        Returns
        -------
        d2P_dTdV : float
            Second volume derivative of pressure, [mol*Pa^2/(J*K)]
        '''
        P, T = self.P, self.T
        return -P*P/(R*T*T)

    def d2T_dV2(self):
        return 0.0

    d2T_dV2_P = d2T_dV2

    def d2V_dT2(self):
        return 0.0

    d2V_dT2_P = d2V_dT2

    def dV_dT(self):
        return R/self.P

    def PIP(self):
        return 1.0 # For speed

    def d2V_dP2(self):
        P, T = self.P, self.T
        return 2.0*R*T/(P*P*P)

    def d2T_dP2(self):
        return 0.0

    def dV_dP(self):
        P, T = self.P, self.T
        return -R*T/(P*P)

    def dT_dP(self):
        return self.T/self.P

    def dT_dV(self):
        return self.P*R_inv

    def dV_dzs(self):
        return self.zeros1d


    d2T_dV2_P = d2T_dV2
    d2V_dT2_P = d2V_dT2
    d2V_dP2_T = d2V_dP2
    d2T_dP2_V = d2T_dP2
    dV_dP_T = dV_dP
    dV_dT_P = dV_dT
    dT_dP_V = dT_dP
    dT_dV_P = dT_dV

    ### Thermodynamic properties

    def H(self):
        r'''Method to calculate and return the enthalpy of the phase.

        .. math::
            H = \sum_i z_i H_{i}^{ig}

        Returns
        -------
        H : float
            Molar enthalpy, [J/(mol)]
        '''
        try:
            return self._H
        except AttributeError:
            pass
        zs = self.zs
        try:
            Cpig_integrals_pure = self._Cpig_integrals_pure
        except AttributeError:
            Cpig_integrals_pure = self.Cpig_integrals_pure()
        H = 0.0
        for i in range(self.N):
            H += zs[i]*Cpig_integrals_pure[i]
        self._H = H
        return H

    def S(self):
        r'''Method to calculate and return the entropy of the phase.

        .. math::
            S = \sum_i z_i S_{i}^{ig} - R\ln\left(\frac{P}{P_{ref}}\right)
            - R\sum_i z_i \ln(z_i)

        Returns
        -------
        S : float
            Molar entropy, [J/(mol*K)]
        '''
        try:
            return self._S
        except AttributeError:
            pass
        Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        log_zs = self.log_zs()
        T, P, zs, cmps = self.T, self.P, self.zs, range(self.N)
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)

        for i in cmps:
            S += zs[i]*Cpig_integrals_over_T_pure[i]
        self._S = S
        return S

    def Cp(self):
        r'''Method to calculate and return the molar heat capacity of the
        phase.

        .. math::
            C_p = \sum_i z_i C_{p,i}^{ig}

        Returns
        -------
        Cp : float
            Molar heat capacity, [J/(mol*K)]
        '''
        try:
            return self._Cp
        except AttributeError:
            pass
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        for i in range(self.N):
            Cp += zs[i]*Cpigs_pure[i]
        self._Cp = Cp
        return Cp

    dH_dT = Cp
    dH_dT_V = Cp # H does not depend on P, so the P is increased without any effect on H

    def dH_dP(self):
        r'''Method to calculate and return the first pressure derivative of
        molar enthalpy of the phase.

        .. math::
            \frac{\partial H}{\partial P} = 0

        Returns
        -------
        dH_dP : float
            First pressure derivative of molar enthalpy, [J/(mol*Pa)]
        '''
        return 0.0

    def d2H_dT2(self):
        r'''Method to calculate and return the first temperature derivative of
        molar heat capacity of the phase.

        .. math::
            \frac{\partial C_p}{\partial T} = \sum_i z_i \frac{\partial
            C_{p,i}^{ig}}{\partial T}

        Returns
        -------
        d2H_dT2 : float
            Second temperature derivative of enthalpy, [J/(mol*K^2)]
        '''
        try:
            return self._d2H_dT2
        except AttributeError:
            pass
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        for i in range(self.N):
            dCp += zs[i]*dCpigs_pure[i]
        self._d2H_dT2 = dCp
        return dCp

    def d2H_dP2(self):
        r'''Method to calculate and return the second pressure derivative of
        molar enthalpy of the phase.

        .. math::
            \frac{\partial^2 H}{\partial P^2} = 0

        Returns
        -------
        d2H_dP2 : float
            Second pressure derivative of molar enthalpy, [J/(mol*Pa^2)]
        '''
        return 0.0

    def d2H_dTdP(self):
        r'''Method to calculate and return the pressure derivative of
        molar heat capacity of the phase.

        .. math::
            \frac{\partial C_p}{\partial P} = 0

        Returns
        -------
        d2H_dTdP : float
            First pressure derivative of heat capacity, [J/(mol*K*Pa)]
        '''
        return 0.0

    def dH_dP_V(self):
        r'''Method to calculate and return the pressure derivative of
        molar enthalpy at constant volume of the phase.

        .. math::
            \left(\frac{\partial H}{\partial P}\right)_{V} = C_p
            \left(\frac{\partial T}{\partial P}\right)_{V}

        Returns
        -------
        dH_dP_V : float
            First pressure derivative of molar enthalpy at constant volume,
            [J/(mol*Pa)]
        '''
        dH_dP_V = self.Cp()*self.dT_dP()
        return dH_dP_V

    def dH_dV_T(self):
        r'''Method to calculate and return the volume derivative of
        molar enthalpy at constant temperature of the phase.

        .. math::
            \left(\frac{\partial H}{\partial V}\right)_{T} = 0

        Returns
        -------
        dH_dV_T : float
            First pressure derivative of molar enthalpy at constant volume,
            [J/(m^3)]
        '''
        return 0.0

    def dH_dV_P(self):
        r'''Method to calculate and return the volume derivative of
        molar enthalpy at constant pressure of the phase.

        .. math::
            \left(\frac{\partial H}{\partial V}\right)_{P} = C_p
            \left(\frac{\partial T}{\partial V}\right)_{P}

        Returns
        -------
        dH_dV_T : float
            First pressure derivative of molar enthalpy at constant volume,
            [J/(m^3)]
        '''
        dH_dV_P = self.dT_dV()*self.Cp()
        return dH_dV_P

    def dH_dzs(self):
        return self.Cpig_integrals_pure()

    def dS_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        molar entropy of the phase.

        .. math::
            \frac{\partial S}{\partial T} = \frac{C_p}{T}

        Returns
        -------
        dS_dT : float
            First temperature derivative of molar entropy, [J/(mol*K^2)]
        '''
        dS_dT = self.Cp()/self.T
        return dS_dT
    dS_dT_P = dS_dT

    def dS_dP(self):
        r'''Method to calculate and return the first pressure derivative of
        molar entropy of the phase.

        .. math::
            \frac{\partial S}{\partial P} = -\frac{R}{P}

        Returns
        -------
        dS_dP : float
            First pressure derivative of molar entropy, [J/(mol*K*Pa)]
        '''
        return -R/self.P

    def d2S_dP2(self):
        r'''Method to calculate and return the second pressure derivative of
        molar entropy of the phase.

        .. math::
            \frac{\partial^2 S}{\partial P^2} = \frac{R}{P^2}

        Returns
        -------
        d2S_dP2 : float
            Second pressure derivative of molar entropy, [J/(mol*K*Pa^2)]
        '''
        P = self.P
        return R/(P*P)

    def dS_dT_V(self):
        r'''Method to calculate and return the first temperature derivative of
        molar entropy at constant volume of the phase.

        .. math::
            \left(\frac{\partial S}{\partial T}\right)_V =
            \frac{C_p}{T} - \frac{R}{P}\frac{\partial P}{\partial T}

        Returns
        -------
        dS_dT_V : float
            First temperature derivative of molar entropy at constant volume,
            [J/(mol*K^2)]
        '''
        dS_dT_V = self.Cp()/self.T - R/self.P*self.dP_dT()
        return dS_dT_V

    def dS_dP_V(self):
        r'''Method to calculate and return the first pressure derivative of
        molar entropy at constant volume of the phase.

        .. math::
            \left(\frac{\partial S}{\partial P}\right)_V =
            \frac{-R}{P} + \frac{C_p}{T}\frac{\partial T}{\partial P}

        Returns
        -------
        dS_dP_V : float
            First pressure derivative of molar entropy at constant volume,
            [J/(mol*K*Pa)]
        '''
        dS_dP_V = -R/self.P + self.Cp()/self.T*self.dT_dP()
        return dS_dP_V

    def d2P_dTdP(self):
        return 0.0

    def d2P_dVdP(self):
        return 0.0

    def d2P_dVdT_TP(self):
        return 0.0

    def d2P_dT2_PV(self):
        return 0.0

    def H_dep(self):
        return 0.0

    G_dep = S_dep = U_dep = A_dep = H_dep

    def dS_dzs(self):
        try:
            return self._dS_dzs
        except AttributeError:
            pass
        N, eos_mix = self.N, self.eos_mix

        log_zs = self.log_zs()
        integrals = self.Cpig_integrals_over_T_pure()

        self._dS_dzs = [integrals[i] - R*(log_zs[i] + 1.0)
                        for i in range(N)]
        return self._dS_dzs

    # Properties using constants, correlations
    def mu(self):
        try:
            return self._mu
        except AttributeError:
            pass
        mu = self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._mu = mu
        return mu

    def k(self):
        try:
            return self._k
        except AttributeError:
            pass
        k = self.correlations.ThermalConductivityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._k = k
        return k

class CEOSGas(Phase):
    r'''Class for representing a cubic equation of state gas phase
    as a phase object. All departure
    properties are actually calculated by the code in :obj:`thermo.eos` and
    :obj:`thermo.eos_mix`.

    .. math::
        P=\frac{RT}{V-b}-\frac{a\alpha(T)}{V^2 + \delta V + \epsilon}

    Parameters
    ----------
    eos_class : :obj:`thermo.eos_mix.GCEOSMIX`
        EOS class, [-]
    eos_kwargs : dict
        Parameters to be passed to the created EOS, [-]
    HeatCapacityGases : list[HeatCapacityGas]
        Objects proiding pure-component heat capacity correlations, [-]
    Hfs : list[float]
        Molar ideal-gas standard heats of formation at 298.15 K and 1 atm,
        [J/mol]
    Gfs : list[float]
        Molar ideal-gas standard Gibbs energies of formation at 298.15 K and
        1 atm, [J/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    zs : list[float], optional
        Mole fractions of each component, [-]

    Examples
    --------
    T-P initialization for oxygen and nitrogen with the PR EOS, using Poling's
    polynomial heat capacities:

    >>> from thermo import HeatCapacityGas, PRMIX, CEOSGas
    >>> eos_kwargs = dict(Tcs=[154.58, 126.2], Pcs=[5042945.25, 3394387.5], omegas=[0.021, 0.04], kijs=[[0.0, -0.0159], [-0.0159, 0.0]])
    >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
    ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
    >>> phase = CEOSGas(eos_class=PRMIX, eos_kwargs=eos_kwargs, T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
    >>> phase.Cp()
    29.2285050

    '''
    is_gas = True
    is_liquid = False
    ideal_gas_basis = True
    __full_path__ = "%s.%s" %(__module__, __qualname__)

    pure_references = ('HeatCapacityGases',)
    pure_reference_types = (HeatCapacityGas,)
    obj_references = ('eos_mix',)


    pointer_references = ('eos_class',)
    pointer_reference_dicts = (eos_mix_full_path_dict,)
    '''Tuple of dictionaries for string -> object
    '''
    reference_pointer_dicts = (eos_mix_full_path_reverse_dict,)

    model_attributes = ('Hfs', 'Gfs', 'Sfs', 'eos_class',
                        'eos_kwargs') + pure_references

    @property
    def phase(self):
        phase = self.eos_mix.phase
        if phase in ('l', 'g'):
            return phase
        return 'g'

    def __repr__(self):
        r'''Method to create a string representation of the phase object, with
        the goal of making it easy to obtain standalone code which reproduces
        the current state of the phase. This is extremely helpful in creating
        new test cases.

        Returns
        -------
        recreation : str
            String which is valid Python and recreates the current state of
            the object if ran, [-]

        Examples
        --------
        >>> from thermo import HeatCapacityGas, PRMIX, CEOSGas
        >>> eos_kwargs = dict(Tcs=[154.58, 126.2], Pcs=[5042945.25, 3394387.5], omegas=[0.021, 0.04], kijs=[[0.0, -0.0159], [-0.0159, 0.0]])
        >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
        ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
        >>> phase = CEOSGas(eos_class=PRMIX, eos_kwargs=eos_kwargs, T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
        >>> phase
        CEOSGas(eos_class=PRMIX, eos_kwargs={"Tcs": [154.58, 126.2], "Pcs": [5042945.25, 3394387.5], "omegas": [0.021, 0.04], "kijs": [[0.0, -0.0159], [-0.0159, 0.0]]}, HeatCapacityGases=[HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-8.231317991971707e-12, 1.3053706310500586e-08, 5.820123832707268e-07, -0.0021700747433379955, 29.424883205644317])), HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [1.48828880864943e-11, -4.9886775708919434e-08, 5.4709164027448316e-05, -0.014916145936966912, 30.18149930389626]))], T=300, P=100000.0, zs=[0.79, 0.21])

        '''
        eos_kwargs = str(self.eos_kwargs).replace("'", '"')
        try:
            Cpgs = ', '.join(str(o) for o in self.HeatCapacityGases)
        except:
            Cpgs = ''
        base = '%s(eos_class=%s, eos_kwargs=%s, HeatCapacityGases=[%s], '  %(self.__class__.__name__, self.eos_class.__name__, eos_kwargs, Cpgs)
        for s in ('Hfs', 'Gfs', 'Sfs', 'T', 'P', 'zs'):
            if hasattr(self, s) and getattr(self, s) is not None:
                base += '%s=%s, ' %(s, getattr(self, s))
        if base[-2:] == ', ':
            base = base[:-2]
        base += ')'
        return base

    def __init__(self, eos_class, eos_kwargs, HeatCapacityGases=None, Hfs=None,
                 Gfs=None, Sfs=None,
                 T=None, P=None, zs=None):
        self.eos_class = eos_class
        self.eos_kwargs = eos_kwargs




        self.HeatCapacityGases = HeatCapacityGases
        if HeatCapacityGases is not None:
            self.N = N = len(HeatCapacityGases)
        elif 'Tcs' in eos_kwargs:
            self.N = N = len(eos_kwargs['Tcs'])

        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs
        self.Cpgs_poly_fit, self._Cpgs_data = self._setup_Cpigs(HeatCapacityGases)
        self.composition_independent = ideal_gas = eos_class is IGMIX
        if ideal_gas:
            self.force_phase = 'g'


        if T is not None and P is not None and zs is not None:
            self.T = T
            self.P = P
            self.zs = zs
            self.eos_mix = eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
        else:
            zs = [1.0/N]*N
            self.eos_mix = eos_mix = self.eos_class(T=298.15, P=101325.0, zs=zs, **self.eos_kwargs)
            self.T = 298.15
            self.P = 101325.0
            self.zs = zs

    def to_TP_zs(self, T, P, zs, other_eos=None):
        r'''Method to create a new Phase object with the same constants as the
        existing Phase but at a different `T` and `P`. This method has a
        special parameter `other_eos`.
        This is added to allow a gas-type phase to be created from
        a liquid-type phase at the same conditions (and vice-versa),
        as :obj:`GCEOSMIX <thermo.eos_mix.GCEOSMIX>` objects were designed to
        have vapor and liquid properties in the same phase. This argument is
        mostly for internal use.

        Parameters
        ----------
        zs : list[float]
            Molar composition of the new phase, [-]
        T : float
            Temperature of the new phase, [K]
        P : float
            Pressure of the new phase, [Pa]
        other_eos : obj:`GCEOSMIX <thermo.eos_mix.GCEOSMIX> object
            Other equation of state object at the same conditions, [-]

        Returns
        -------
        new_phase : Phase
            New phase at the specified conditions, [-]

        Notes
        -----
        This method is marginally faster than :obj:`Phase.to` as it does not
        need to check what the inputs are.

        Examples
        --------

        >>> from thermo.eos_mix import PRMIX
        >>> eos_kwargs = dict(Tcs=[305.32, 369.83], Pcs=[4872000.0, 4248000.0], omegas=[0.098, 0.152])
        >>> gas = CEOSGas(PRMIX, T=300.0, P=1e6, zs=[.2, .8], eos_kwargs=eos_kwargs)
        >>> liquid = CEOSLiquid(PRMIX, T=500.0, P=1e7, zs=[.3, .7], eos_kwargs=eos_kwargs)
        >>> new_liq = liquid.to_TP_zs(T=gas.T, P=gas.P, zs=gas.zs, other_eos=gas.eos_mix)
        >>> new_liq
        CEOSLiquid(eos_class=PRMIX, eos_kwargs={"Tcs": [305.32, 369.83], "Pcs": [4872000.0, 4248000.0], "omegas": [0.098, 0.152]}, HeatCapacityGases=[], T=300.0, P=1000000.0, zs=[0.2, 0.8])
        >>> new_liq.eos_mix is gas.eos_mix
        True
        '''
        # Why so slow
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        if other_eos is not None:
            other_eos.solve_missing_volumes()
            new.eos_mix = other_eos
        else:
            try:
                new.eos_mix = self.eos_mix.to_TP_zs_fast(T=T, P=P, zs=zs, only_g=True,
                                                         full_alphas=True) # optimize alphas?
                                                         # Be very careful doing this in the future - wasted
                                                         # 1 hour on this because the heat capacity calculation was wrong
            except AttributeError:
                new.eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)

        new.eos_class = self.eos_class
        new.eos_kwargs = self.eos_kwargs

        new.HeatCapacityGases = self.HeatCapacityGases
        new._Cpgs_data = self._Cpgs_data
        new.Cpgs_poly_fit = self.Cpgs_poly_fit
        new.composition_independent = self.composition_independent
        if new.composition_independent:
            new.force_phase = 'g'

        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs

        try:
            new.N = self.N
        except:
            pass

        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs

        if T is not None:
            if P is not None:
                try:
                    new.eos_mix = self.eos_mix.to_TP_zs_fast(T=T, P=P, zs=zs, only_g=True,
                                                             full_alphas=True)
                except AttributeError:
                    new.eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
            elif V is not None:
                try:
                    new.eos_mix = self.eos_mix.to(T=T, V=V, zs=zs, fugacities=False)
                except AttributeError:
                    new.eos_mix = self.eos_class(T=T, V=V, zs=zs, **self.eos_kwargs)
                P = new.eos_mix.P
        elif P is not None and V is not None:
            try:
                new.eos_mix = self.eos_mix.to_PV_zs(P=P, V=V, zs=zs, only_g=True, fugacities=False)
            except AttributeError:
                new.eos_mix = self.eos_class(P=P, V=V, zs=zs, only_g=True, **self.eos_kwargs)
            T = new.eos_mix.T
        else:
            raise ValueError("Two of T, P, or V are needed")
        new.P = P
        new.T = T

        new.eos_class = self.eos_class
        new.eos_kwargs = self.eos_kwargs

        new.HeatCapacityGases = self.HeatCapacityGases
        new._Cpgs_data = self._Cpgs_data
        new.Cpgs_poly_fit = self.Cpgs_poly_fit

        new.composition_independent = self.composition_independent
        if new.composition_independent:
            new.force_phase = 'g'

        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs

        try:
            new.N = self.N
        except:
            pass

        return new

    def V_iter(self, force=False):
        # Can be some severe issues in the very low pressure/temperature range
        # For that reason, consider not doing TV iterations.
        # Cal occur also with PV iterations

        T, P = self.T, self.P
#        if 0 and ((P < 1.0 or T < 1.0) or (P/T < 500.0 and T < 50.0)):
        eos_mix = self.eos_mix
        V = self.V()
        P_err = abs((R*T/(V-eos_mix.b) - eos_mix.a_alpha/(V*V + eos_mix.delta*V + eos_mix.epsilon)) - P)
        if (P_err/P) < 1e-9 and not force:
            return V
        try:
            return eos_mix.V_g_mpmath.real
        except:
            return eos_mix.V_l_mpmath.real
#        else:
#            return self.V()

    def lnphis_G_min(self):
        eos_mix = self.eos_mix
        if eos_mix.phase == 'l/g':
            # Check both phases are solved, and complete if not
            eos_mix.solve_missing_volumes()
            if eos_mix.G_dep_l < eos_mix.G_dep_g:
                return eos_mix.fugacity_coefficients(eos_mix.Z_l)
            return eos_mix.fugacity_coefficients(eos_mix.Z_g)
        try:
            return eos_mix.fugacity_coefficients(eos_mix.Z_g)
        except AttributeError:
            return eos_mix.fugacity_coefficients(eos_mix.Z_l)


    def lnphis_args(self):
        eos_mix = self.eos_mix
        return (self.eos_class.model_id, self.T, self.P, eos_mix.kijs, self.is_liquid, self.is_gas,
                eos_mix.bs, eos_mix.a_alphas, eos_mix.a_alpha_roots)

    def lnphis_at_zs(self, zs):
        eos_mix = self.eos_mix
        if eos_mix.__class__.__name__ == 'PRMIX':
            return PR_lnphis_fastest(zs, self.T, self.P, eos_mix.kijs, self.is_liquid, self.is_gas,
                                     eos_mix.bs, eos_mix.a_alphas, eos_mix.a_alpha_roots)
        return self.to_TP_zs(self.T, self.P, zs).lnphis()



    def lnphis(self):
        r'''Method to calculate and return the log of fugacity coefficients of
        each component in the phase. The calculation is performed by
        :obj:`thermo.eos_mix.GCEOSMIX.fugacity_coefficients` or a simpler formula in the case
        of most specific models.

        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
        '''
        try:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_g)
        except AttributeError:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_l)


    def dlnphis_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        the log of fugacity coefficients of
        each component in the phase. The calculation is performed by
        :obj:`thermo.eos_mix.GCEOSMIX.dlnphis_dT` or a simpler formula in the
        case of most specific models.

        Returns
        -------
        dlnphis_dT : list[float]
            First temperature derivative of log fugacity coefficients, [1/K]
        '''
        try:
            return self.eos_mix.dlnphis_dT('g')
        except:
            return self.eos_mix.dlnphis_dT('l')

    def dlnphis_dP(self):
        r'''Method to calculate and return the first pressure derivative of
        the log of fugacity coefficients of
        each component in the phase. The calculation is performed by
        :obj:`thermo.eos_mix.GCEOSMIX.dlnphis_dP` or a simpler formula in the
        case of most specific models.

        Returns
        -------
        dlnphis_dP : list[float]
            First pressure derivative of log fugacity coefficients, [1/Pa]
        '''
        try:
            return self.eos_mix.dlnphis_dP('g')
        except:
            return self.eos_mix.dlnphis_dP('l')

    def dlnphis_dns(self):
        eos_mix = self.eos_mix
        try:
            return eos_mix.dlnphis_dns(eos_mix.Z_g)
        except:
            return eos_mix.dlnphis_dns(eos_mix.Z_l)

    def dlnphis_dzs(self):
        # Confirmed to be mole fraction derivatives - taked with sum not 1 -
        # of the log fugacity coefficients!
        eos_mix = self.eos_mix
        try:
            return eos_mix.dlnphis_dzs(eos_mix.Z_g)
        except:
            return eos_mix.dlnphis_dzs(eos_mix.Z_l)

    def fugacities_lowest_Gibbs(self):
        eos_mix = self.eos_mix
        P = self.P
        zs = self.zs
        try:
            if eos_mix.G_dep_g < eos_mix.G_dep_l:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_g)
            else:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_l)
        except:
            try:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_g)
            except:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_l)
        return [P*zs[i]*trunc_exp(lnphis[i]) for i in range(len(zs))]


    def gammas(self):
        #         liquid.phis()/np.array([i.phi_l for i in liquid.eos_mix.pures()])
        phis = self.phis()
        phis_pure = []
        for i in self.eos_mix.pures():
            try:
                phis_pure.append(i.phi_g)
            except AttributeError:
                phis_pure.append(i.phi_l)
        return [phis[i]/phis_pure[i] for i in range(self.N)]


    def H_dep(self):
        try:
            return self.eos_mix.H_dep_g
        except AttributeError:
            return self.eos_mix.H_dep_l

    def S_dep(self):
        try:
            return self.eos_mix.S_dep_g
        except AttributeError:
            return self.eos_mix.S_dep_l

    def G_dep(self):
        try:
            return self.eos_mix.G_dep_g
        except AttributeError:
            return self.eos_mix.G_dep_l

    def Cp_dep(self):
        try:
            return self.eos_mix.Cp_dep_g
        except AttributeError:
            return self.eos_mix.Cp_dep_l

    def V(self):
        r'''Method to calculate and return the molar volume of the phase.

        Returns
        -------
        V : float
            Molar volume, [m^3/mol]
        '''
        try:
            return self.eos_mix.V_g
        except AttributeError:
            return self.eos_mix.V_l

    def dP_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        pressure of the phase.

        .. math::
            \left(\frac{\partial P}{\partial T}\right)_V = \frac{R}{V - b}
            - \frac{a \frac{d \alpha{\left (T \right )}}{d T}}{V^{2} + V \delta
            + \epsilon}

        Returns
        -------
        dP_dT : float
            First temperature derivative of pressure, [Pa/K]
        '''
        try:
            return self.eos_mix.dP_dT_g
        except AttributeError:
            return self.eos_mix.dP_dT_l

    dP_dT_V = dP_dT

    def dP_dV(self):
        r'''Method to calculate and return the first volume derivative of
        pressure of the phase.

        .. math::
            \left(\frac{\partial P}{\partial V}\right)_T = - \frac{R T}{\left(
            V - b\right)^{2}} - \frac{a \left(- 2 V - \delta\right) \alpha{
            \left (T \right )}}{\left(V^{2} + V \delta + \epsilon\right)^{2}}

        Returns
        -------
        dP_dV : float
            First volume derivative of pressure, [Pa*mol/m^3]
        '''
        try:
            return self.eos_mix.dP_dV_g
        except AttributeError:
            return self.eos_mix.dP_dV_l

    dP_dV_T = dP_dV

    def d2P_dT2(self):
        r'''Method to calculate and return the second temperature derivative of
        pressure of the phase.

        .. math::
            \left(\frac{\partial^2  P}{\partial T^2}\right)_V =  - \frac{a
            \frac{d^{2} \alpha{\left (T \right )}}{d T^{2}}}{V^{2} + V \delta
            + \epsilon}

        Returns
        -------
        d2P_dT2 : float
            Second temperature derivative of pressure, [Pa/K^2]
        '''
        try:
            return self.eos_mix.d2P_dT2_g
        except AttributeError:
            return self.eos_mix.d2P_dT2_l

    d2P_dT2_V = d2P_dT2

    def d2P_dV2(self):
        r'''Method to calculate and return the second volume derivative of
        pressure of the phase.

        .. math::
            \left(\frac{\partial^2  P}{\partial V^2}\right)_T = 2 \left(\frac{
            R T}{\left(V - b\right)^{3}} - \frac{a \left(2 V + \delta\right)^{
            2} \alpha{\left (T \right )}}{\left(V^{2} + V \delta + \epsilon
            \right)^{3}} + \frac{a \alpha{\left (T \right )}}{\left(V^{2} + V
            \delta + \epsilon\right)^{2}}\right)

        Returns
        -------
        d2P_dV2 : float
            Second volume derivative of pressure, [Pa*mol^2/m^6]
        '''
        try:
            return self.eos_mix.d2P_dV2_g
        except AttributeError:
            return self.eos_mix.d2P_dV2_l

    d2P_dV2_T = d2P_dV2

    def d2P_dTdV(self):
        r'''Method to calculate and return the second derivative of
        pressure with respect to temperature and volume of the phase.

        .. math::
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) = - \frac{
            R}{\left(V - b\right)^{2}} + \frac{a \left(2 V + \delta\right)
            \frac{d \alpha{\left (T \right )}}{d T}}{\left(V^{2} + V \delta
            + \epsilon\right)^{2}}

        Returns
        -------
        d2P_dTdV : float
            Second volume derivative of pressure, [mol*Pa^2/(J*K)]
        '''
        try:
            return self.eos_mix.d2P_dTdV_g
        except AttributeError:
            return self.eos_mix.d2P_dTdV_l

    # because of the ideal gas model, for some reason need to use the right ones
    # FOR THIS MODEL ONLY
    def d2T_dV2(self):
        try:
            return self.eos_mix.d2T_dV2_g
        except AttributeError:
            return self.eos_mix.d2T_dV2_l

    d2T_dV2_P = d2T_dV2

    def d2V_dT2(self):
        try:
            return self.eos_mix.d2V_dT2_g
        except AttributeError:
            return self.eos_mix.d2V_dT2_l

    d2V_dT2_P = d2V_dT2

    def dV_dzs(self):
        eos_mix = self.eos_mix
        try:
            dV_dzs = self.eos_mix.dV_dzs(eos_mix.Z_g)
        except AttributeError:
            dV_dzs = self.eos_mix.dV_dzs(eos_mix.Z_l)
        return dV_dzs

    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        H = self.H_dep()
        for zi, Cp_int in zip(self.zs, self.Cpig_integrals_pure()):
            H += zi*Cp_int
        self._H = H
        return H

    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        log_zs = self.log_zs()
        T, P, zs, cmps = self.T, self.P, self.zs, range(self.N)
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)

        for i in cmps:
            S += zs[i]*Cpig_integrals_over_T_pure[i]
        S += self.S_dep()
        self._S = S
        return S



    def Cp(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        for i in range(self.N):
            Cp += zs[i]*Cpigs_pure[i]
        Cp += self.Cp_dep()
        self._Cp = Cp
        return Cp

    dH_dT = dH_dT_P = Cp

    def dH_dP(self):
        try:
            return self._dH_dP
        except AttributeError:
            pass
        try:
            self._dH_dP = dH_dP = self.eos_mix.dH_dep_dP_g
        except AttributeError:
            self._dH_dP = dH_dP = self.eos_mix.dH_dep_dP_l
        return dH_dP

    def d2H_dT2(self):
        try:
            return self._d2H_dT2
        except AttributeError:
            pass
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        for i in range(self.N):
            dCp += zs[i]*dCpigs_pure[i]
        try:
            dCp += self.eos_mix.d2H_dep_dT2_g
        except AttributeError:
            dCp += self.eos_mix.d2H_dep_dT2_l
        self._d2H_dT2 = dCp
        return dCp

    def d2H_dT2_V(self):
        # Turned out not to be needed when I thought it was - ignore this!
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        for i in range(self.N):
            dCp += zs[i]*dCpigs_pure[i]

        try:
            dCp += self.eos_mix.d2H_dep_dT2_g_V
        except AttributeError:
            dCp += self.eos_mix.d2H_dep_dT2_l_V
        return dCp


    def d2H_dP2(self):
        try:
            return self.eos_mix.d2H_dep_dP2_g
        except AttributeError:
            return self.eos_mix.d2H_dep_dP2_l

    def d2H_dTdP(self):
        try:
            return self.eos_mix.d2H_dep_dTdP_g
        except AttributeError:
            return self.eos_mix.d2H_dep_dTdP_l


    def dH_dT_V(self):
        dH_dT_V = self.Cp_ideal_gas()
        try:
            dH_dT_V += self.eos_mix.dH_dep_dT_g_V
        except AttributeError:
            dH_dT_V += self.eos_mix.dH_dep_dT_l_V
        return dH_dT_V

    def dH_dP_V(self):
        dH_dP_V = self.Cp_ideal_gas()*self.dT_dP()
        try:
            dH_dP_V += self.eos_mix.dH_dep_dP_g_V
        except AttributeError:
            dH_dP_V += self.eos_mix.dH_dep_dP_l_V
        return dH_dP_V

    def dH_dV_T(self):
        dH_dV_T = 0.0
        try:
            dH_dV_T += self.eos_mix.dH_dep_dV_g_T
        except AttributeError:
            dH_dV_T += self.eos_mix.dH_dep_dV_l_T
        return dH_dV_T

    def dH_dV_P(self):
        dH_dV_P = self.dT_dV()*self.Cp_ideal_gas()
        try:
            dH_dV_P += self.eos_mix.dH_dep_dV_g_P
        except AttributeError:
            dH_dV_P += self.eos_mix.dH_dep_dV_l_P
        return dH_dV_P

    def dH_dzs(self):
        try:
            return self._dH_dzs
        except AttributeError:
            pass
        eos_mix = self.eos_mix
        try:
            dH_dep_dzs = self.eos_mix.dH_dep_dzs(eos_mix.Z_g)
        except AttributeError:
            dH_dep_dzs = self.eos_mix.dH_dep_dzs(eos_mix.Z_l)
        Cpig_integrals_pure = self.Cpig_integrals_pure()
        self._dH_dzs = [dH_dep_dzs[i] + Cpig_integrals_pure[i] for i in range(self.N)]
        return self._dH_dzs

    def dS_dT(self):
        HeatCapacityGases = self.HeatCapacityGases
        cmps = range(self.N)
        T, zs = self.T, self.zs
        T_REF_IG = self.T_REF_IG
        P_REF_IG_INV = self.P_REF_IG_INV

        dS_dT = self.Cp_ideal_gas()/T
        try:
            dS_dT += self.eos_mix.dS_dep_dT_g
        except AttributeError:
            dS_dT += self.eos_mix.dS_dep_dT_l
        return dS_dT

    def dS_dP(self):
        dS = 0.0
        P = self.P
        dS -= R/P
        try:
            dS += self.eos_mix.dS_dep_dP_g
        except AttributeError:
            dS += self.eos_mix.dS_dep_dP_l
        return dS

    def d2S_dP2(self):
        P = self.P
        d2S = R/(P*P)
        try:
            d2S += self.eos_mix.d2S_dep_dP_g
        except AttributeError:
            d2S += self.eos_mix.d2S_dep_dP_l
        return d2S

    def dS_dT_P(self):
        return self.dS_dT()

    def dS_dT_V(self):
        r'''Method to calculate and return the first temperature derivative of
        molar entropy at constant volume of the phase.

        .. math::
            \left(\frac{\partial S}{\partial T}\right)_V =
            \frac{C_p^{ig}}{T} - \frac{R}{P}\frac{\partial P}{\partial T}
            + \left(\frac{\partial S_{dep}}{\partial T}\right)_V

        Returns
        -------
        dS_dT_V : float
            First temperature derivative of molar entropy at constant volume,
            [J/(mol*K^2)]
        '''
        # Good
        '''
        # Second last bit from
        from sympy import *
        T, R = symbols('T, R')
        P = symbols('P', cls=Function)
        expr =-R*log(P(T)/101325)
        diff(expr, T)
        '''
        dS_dT_V = self.Cp_ideal_gas()/self.T - R/self.P*self.dP_dT()
        try:
            dS_dT_V += self.eos_mix.dS_dep_dT_g_V
        except AttributeError:
            dS_dT_V += self.eos_mix.dS_dep_dT_l_V
        return dS_dT_V

    def dS_dP_V(self):
        dS_dP_V = -R/self.P + self.Cp_ideal_gas()/self.T*self.dT_dP()
        try:
            dS_dP_V += self.eos_mix.dS_dep_dP_g_V
        except AttributeError:
            dS_dP_V += self.eos_mix.dS_dep_dP_l_V
        return dS_dP_V

    # The following - likely should be reimplemented generically
    # http://www.coolprop.org/_static/doxygen/html/class_cool_prop_1_1_abstract_state.html#a0815380e76a7dc9c8cc39493a9f3df46

    def d2P_dTdP(self):

        try:
            return self.eos_mix.d2P_dTdP_g
        except AttributeError:
            return self.eos_mix.d2P_dTdP_l

    def d2P_dVdP(self):
        try:
            return self.eos_mix.d2P_dVdP_g
        except AttributeError:
            return self.eos_mix.d2P_dVdP_l

    def d2P_dVdT_TP(self):
        try:
            return self.eos_mix.d2P_dVdT_TP_g
        except AttributeError:
            return self.eos_mix.d2P_dVdT_TP_l

    def d2P_dT2_PV(self):
        try:
            return self.eos_mix.d2P_dT2_PV_g
        except AttributeError:
            return self.eos_mix.d2P_dT2_PV_l

    def dS_dzs(self):
        try:
            return self._dS_dzs
        except AttributeError:
            pass
        cmps, eos_mix = range(self.N), self.eos_mix

        log_zs = self.log_zs()
        integrals = self.Cpig_integrals_over_T_pure()

        try:
            dS_dep_dzs = self.eos_mix.dS_dep_dzs(eos_mix.Z_g)
        except AttributeError:
            dS_dep_dzs = self.eos_mix.dS_dep_dzs(eos_mix.Z_l)

        self._dS_dzs = [integrals[i] - R*(log_zs[i] + 1.0) + dS_dep_dzs[i]
                        for i in cmps]
        return self._dS_dzs

    def _set_mechanical_critical_point(self):
        zs = self.zs
        new = self.eos_mix.to_mechanical_critical_point()
        self._mechanical_critical_T = new.T
        self._mechanical_critical_P = new.P
        try:
            V = new.V_l
        except:
            V = new.V_g
        self._mechanical_critical_V = V
        return new.T, new.P, V

    def P_transitions(self):
        e = self.eos_mix
        return e.P_discriminant_zeros_analytical(e.T, e.b, e.delta, e.epsilon, e.a_alpha, valid=True)
        # EOS is guaranteed to be at correct temperature
        try:
            return [self.eos_mix.P_discriminant_zero_l()]
        except:
            return [self.eos_mix.P_discriminant_zero_g()]

    def T_transitions(self):
        try:
            return [self.eos_mix.T_discriminant_zero_l()]
        except:
            return [self.eos_mix.T_discriminant_zero_g()]

    def T_max_at_V(self, V):
        T_max = self.eos_mix.T_max_at_V(V)
        if T_max is not None:
            T_max = T_max*(1.0-1e-12)
        return T_max

    def P_max_at_V(self, V):
        P_max = self.eos_mix.P_max_at_V(V)
        if P_max is not None:
            P_max = P_max*(1.0-1e-12)
        return P_max

    def mu(self):
#        try:
#            return self._mu
#        except AttributeError:
#            pass
        try:
            phase == self.assigned_phase
        except:
            phase = self.eos_mix.phase
            if phase == 'l/g': phase = 'g'
        try:
            ws = self._ws
        except:
            ws = self.ws()
        if phase == 'g':
            mu = self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, ws)
        else:
            mu = self.correlations.ViscosityLiquidMixture.mixture_property(self.T, self.P, self.zs, ws)
        self._mu = mu
        return mu

    def k(self):
        try:
            return self._k
        except AttributeError:
            pass
        try:
            phase == self.assigned_phase
        except:
            phase = self.eos_mix.phase
            if phase == 'l/g': phase = 'g'
        if phase == 'g':
            k = self.correlations.ThermalConductivityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        elif phase == 'l':
            k = self.correlations.ThermalConductivityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._k = k
        return k

def build_CEOSLiquid():
    import inspect
    source = inspect.getsource(CEOSGas)
    source = source.replace('CEOSGas', 'CEOSLiquid').replace('only_g', 'only_l')
    source = source.replace("'g'", "'gORIG'")
    source = source.replace("'l'", "'g'")
    source = source.replace("'gORIG'", "'l'")
    source = source.replace("ViscosityGasMixture", "gViscosityGasMixture")
    source = source.replace("ViscosityLiquidMixture", "ViscosityGasMixture")
    source = source.replace("gViscosityGasMixture", "ViscosityLiquidMixture")
    source = source.replace("ThermalConductivityGasMixture", "gThermalConductivityGasMixture")
    source = source.replace("ThermalConductivityLiquidMixture", "ThermalConductivityGasMixture")
    source = source.replace("gThermalConductivityGasMixture", "ThermalConductivityLiquidMixture")
    # TODO add new volume derivatives
    swap_strings = ('Cp_dep', 'd2P_dT2', 'd2P_dTdV', 'd2P_dV2', 'd2T_dV2',
                    'd2V_dT2', 'dH_dep_dP', 'dP_dT', 'dP_dV', 'phi',
                    'dS_dep_dP', 'dS_dep_dT', 'G_dep', 'H_dep', 'S_dep', '.V', '.Z',
                    'd2P_dVdT_TP', 'd2P_dT2_PV', 'd2P_dVdP', 'd2P_dTdP',
                    'd2S_dep_dP', 'dH_dep_dV', 'dH_dep_dT', 'd2H_dep_dTdP',
                    'd2H_dep_dP2', 'd2H_dep_dT2')
    for s in swap_strings:
        source = source.replace(s+'_g', 'gORIG')
        source = source.replace(s+'_l', s+'_g')
        source = source.replace('gORIG', s+'_l')
    return source

from fluids.numerics import is_micropython
if not is_micropython:
    try:
        CEOSLiquid
    except:
        loaded_data = False
        # Cost is ~10 ms - must be pasted in the future!
        try:  # pragma: no cover
            from appdirs import user_data_dir, user_config_dir
            data_dir = user_config_dir('thermo')
        except ImportError:  # pragma: no cover
            data_dir = ''
        if data_dir:
            import marshal
            try:
                1/0
                f = open(os.path.join(data_dir, 'CEOSLiquid.dat'), 'rb')
                compiled_CEOSLiquid = marshal.load(f)
                f.close()
                loaded_data = True
            except:
                pass
            if not loaded_data:
                compiled_CEOSLiquid = compile(build_CEOSLiquid(), '<string>', 'exec')
                f = open(os.path.join(data_dir, 'CEOSLiquid.dat'), 'wb')
                marshal.dump(compiled_CEOSLiquid, f)
                f.close()
        else:
            compiled_CEOSLiquid = compile(build_CEOSLiquid(), '<string>', 'exec')
        exec(compiled_CEOSLiquid)
        # exec(build_CEOSLiquid())
else:
    class CEOSLiquid(object):
        __full_path__ = 'thermo.phases.CEOSLiquid'

CEOSLiquid.is_gas = False
CEOSLiquid.is_liquid = True

class GibbsExcessLiquid(Phase):
    r'''Phase based on combining Raoult's law with a
    :obj:`GibbsExcess <thermo.activity.GibbsExcess>` model, optionally
    including saturation fugacity coefficient corrections (if the vapor phase
    is a cubic equation of state) and Poynting correction factors (if more
    accuracy is desired).

    The equilibrium equation options (controlled by `equilibrium_basis`)
    are as follows:

    * 'Psat': :math:`\phi_i = \frac{\gamma_i P_{i}^{sat}}{P}`
    * 'Poynting&PhiSat': :math:`\phi_i = \frac{\gamma_i P_{i}^{sat} \phi_i^{sat} \text{Poynting}_i}{P}`
    * 'Poynting': :math:`\phi_i = \frac{\gamma_i P_{i}^{sat}\text{Poynting}_i}{P}`
    * 'PhiSat': :math:`\phi_i = \frac{\gamma_i P_{i}^{sat} \phi_i^{sat}}{P}`

    In all cases, the activity coefficient is derived from the
    :obj:`GibbsExcess <thermo.activity.GibbsExcess>` model specified as
    input; use the :obj:`IdealSolution <thermo.activity.IdealSolution>`
    class as an input to set the activity coefficients to one.

    The enthalpy `H` and entropy `S` (and other caloric properties `U`, `G`, `A`)
    equation options are similar to the equilibrium ones. If the same option
    is selected for `equilibrium_basis` and `caloric_basis`, the phase will be
    `thermodynamically consistent`. This is recommended for many reasons.
    The full 'Poynting&PhiSat' equations for `H` and `S` are as follows; see
    :obj:`GibbsExcessLiquid.H` and :obj:`GibbsExcessLiquid.S` for all of the
    other equations:

    .. math::
        H = H_{\text{excess}} + \sum_i z_i \left[-RT^2\left(
        \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
        + \frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
        + \frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}} \right)
        + \int_{T,ref}^T C_{p,ig} dT \right]

    .. math::
        S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
        - \sum_i z_i\left[R\left(
        T \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
        + T\frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
        + T\frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}}
        + \ln(P_{\text{sat},i}) + \ln\left(\frac{\text{Poynting}\cdot\phi_{\text{sat},i}}{P}\right)
        \right) - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

    An additional caloric mode is `Hvap`, which uses enthalpy of vaporization;
    this mode can never be thermodynamically consistent, but is still widely
    used.

    .. math::
        H = H_{\text{excess}} + \sum_i z_i\left[-H_{vap,i}
        + \int_{T,ref}^T C_{p,ig} dT \right]

    .. math::
        S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
        - \sum_i z_i\left[R\left(\ln P_{\text{sat},i} + \ln\left(\frac{1}{P}\right)\right)
        + \frac{H_{vap,i}}{T}
        - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]


    .. warning::
        Note that above the critical point, there is no definition for what vapor
        pressure is. The vapor pressure also tends to reach zero at temperatures
        in the 4-20 K range. These aspects mean extrapolation in the supercritical and
        very low temperature region is critical to ensure the equations will still
        converge. Extrapolation can be performed using either the equation
        :math:`P^{\text{sat}} = \exp\left(A - \frac{B}{T}\right)` or
        :math:`P^{\text{sat}} = \exp\left(A + \frac{B}{T} + C\cdot \ln T\right)` by
        setting `Psat_extrpolation` to either 'AB' or 'ABC' respectively.
        The extremely low temperature region's issue is solved by calculating the
        logarithm of vapor pressures instead of the actual value. While floating
        point values in Python (doubles) can reach a minimum value of around
        1e-308, if only the logarithm of that number is computed no issues arise.
        Both of these features only work when the vapor pressure correlations are
        polynomials.

    .. warning::
        When using 'PhiSat' as an option, note that the factor cannot be
        calculated when a compound is supercritical,
        as there is no longer any vapor-liquid pure-component equilibrium
        (by definition).

    Parameters
    ----------
    VaporPressures : list[:obj:`thermo.vapor_pressure.VaporPressure`]
        Objects holding vapor pressure data and methods, [-]
    VolumeLiquids : list[:obj:`thermo.volume.VolumeLiquid`], optional
        Objects holding liquid volume data and methods; required for Poynting
        factors and volumetric properties, [-]
    HeatCapacityGases : list[:obj:`thermo.heat_capacity.HeatCapacityGas`], optional
        Objects proiding pure-component heat capacity correlations; required
        for caloric properties, [-]
    GibbsExcessModel : :obj:`GibbsExcess <thermo.activity.GibbsExcess>`, optional
        Configured instance for calculating activity coefficients and excess properties;
        set to :obj:`IdealSolution <thermo.activity.IdealSolution>` if not provided, [-]
    eos_pure_instances : list[:obj:`thermo.eos.GCEOS`], optional
        Cubic equation of state object instances for each pure component, [-]
    EnthalpyVaporizations : list[:obj:`thermo.phase_change.EnthalpyVaporization`], optional
        Objects holding enthalpy of vaporization data and methods; used only
        with the 'Hvap' optional, [-]
    HeatCapacityLiquids : list[:obj:`thermo.heat_capacity.HeatCapacityLiquid`], optional
        Objects holding liquid heat capacity data and methods; not used at
        present, [-]
    VolumeSupercriticalLiquids : list[:obj:`thermo.volume.VolumeLiquid`], optional
        Objects holding liquid volume data and methods but that are used for
        supercritical temperatures on a per-component basis only; required for
        Poynting factors and volumetric properties at supercritical conditions;
        `VolumeLiquids` is used if not provided, [-]
    Hfs : list[float], optional
        Molar ideal-gas standard heats of formation at 298.15 K and 1 atm,
        [J/mol]
    Gfs : list[float], optional
        Molar ideal-gas standard Gibbs energies of formation at 298.15 K and
        1 atm, [J/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    zs : list[float], optional
        Mole fractions of each component, [-]
    equilibrium_basis : str, optional
        Which set of equilibrium equations to use when calculating fugacities
        and related properties; valid options are 'Psat', 'Poynting&PhiSat',
        'Poynting', 'PhiSat', [-]
    caloric_basis : str, optional
        Which set of caloric equations to use when calculating fugacities
        and related properties; valid options are 'Psat', 'Poynting&PhiSat',
        'Poynting', 'PhiSat', 'Hvap' [-]
    Psat_extrpolation : str, optional
        One of 'AB' or 'ABC'; configures extrapolation for vapor pressure, [-]


    use_Hvap_caloric : bool, optional
        If True, enthalpy and entropy will be calculated using ideal-gas
        heat capacity and the heat of vaporization of the fluid only. This
        forces enthalpy to be pressure-independent. This supersedes other
        options which would otherwise impact these properties. The molar volume
        of the fluid has no impact on enthalpy or entropy if this option is
        True. This option is not thermodynamically consistent, but is still
        often an assumption that is made.

    '''
    force_phase = 'l'
    phase = 'l'
    is_gas = False
    is_liquid = True
    __full_path__ = "%s.%s" %(__module__, __qualname__)

    P_DEPENDENT_H_LIQ = True
    PHI_SAT_IDEAL_TR = 0.1
    _Psats_data = None
    Psats_poly_fit = False
    Vms_sat_poly_fit = False
    _Vms_sat_data = None
    Hvap_poly_fit = False
    _Hvap_data = None

    use_IG_Cp = True # Deprecated! Remove with S_old and H_old

    ideal_gas_basis = True
    supercritical_volumes = False

    Cpls_poly_fit = False
    _Cpls_data = None

    _Tait_B_data = None
    _Tait_C_data = None

    pure_references = ('HeatCapacityGases', 'VolumeLiquids', 'VaporPressures', 'HeatCapacityLiquids',
                       'EnthalpyVaporizations')
    pure_reference_types = (HeatCapacityGas, VolumeLiquid, VaporPressure, HeatCapacityLiquid,
                            EnthalpyVaporization)

    model_attributes = ('Hfs', 'Gfs', 'Sfs', 'GibbsExcessModel',
                        'eos_pure_instances', 'use_Poynting', 'use_phis_sat',
                        'use_Tait', 'use_eos_volume', 'henry_components',
                        'henry_data', 'Psat_extrpolation') + pure_references

    obj_references = ('GibbsExcessModel', 'eos_pure_instances')

#    pointer_references = ('GibbsExcessModel',)
#    pointer_reference_dicts = (activity_pointer_reference_dicts,)
#    reference_pointer_dicts = (activity_reference_pointer_dicts,)










    def __init__(self, VaporPressures, VolumeLiquids=None,
                 HeatCapacityGases=None,
                 GibbsExcessModel=None,
                 eos_pure_instances=None,
                 EnthalpyVaporizations=None,
                 HeatCapacityLiquids=None,
                 VolumeSupercriticalLiquids=None,

                 use_Hvap_caloric=False,
                 use_Poynting=False,
                 use_phis_sat=False,
                 use_Tait=False,
                 use_eos_volume=False,

                 Hfs=None, Gfs=None, Sfs=None,

                 henry_components=None, henry_data=None,

                 T=None, P=None, zs=None,
                 Psat_extrpolation='AB',
                 equilibrium_basis=None,
                 caloric_basis=None,
                 ):
        '''It is quite possible to introduce a PVT relation ship for liquid
        density and remain thermodynamically consistent. However, must be
        applied on a per-component basis! This class cannot have an
        equation-of-state or VolumeLiquidMixture for a liquid MIXTURE!

        (it might still be nice to generalize the handling; maybe even allow)
        pure EOSs to be used too, and as a form/template for which functions to
        use).

        In conclusion, you have
        1) The standard H/S model
        2) The H/S model with all pressure correction happening at P
        3) The inconsistent model which has no pressure dependence whatsover in H/S
           This model is required due to its popularity, not its consistency (but still volume dependency)

        All mixture volumetric properties have to be averages of the pure
        components properties and derivatives. A Multiphase will be needed to
        allow flashes with different properties from different phases.
        '''


        self.VaporPressures = VaporPressures
        self.Psats_poly_fit = all(i.method == POLY_FIT for i in VaporPressures) if VaporPressures is not None else False
        self.Psat_extrpolation = Psat_extrpolation
        if self.Psats_poly_fit:
            Psats_data = [[i.poly_fit_Tmin for i in VaporPressures],
                               [i.poly_fit_Tmin_slope for i in VaporPressures],
                               [i.poly_fit_Tmin_value for i in VaporPressures],
                               [i.poly_fit_Tmax for i in VaporPressures],
                               [i.poly_fit_Tmax_slope for i in VaporPressures],
                               [i.poly_fit_Tmax_value for i in VaporPressures],
                               [i.poly_fit_coeffs for i in VaporPressures],
                               [i.poly_fit_d_coeffs for i in VaporPressures],
                               [i.poly_fit_d2_coeffs for i in VaporPressures],
                               [i.DIPPR101_ABC for i in VaporPressures]]
            if Psat_extrpolation == 'AB':
                Psats_data.append([i.poly_fit_AB_high_ABC_compat + [0.0] for i in VaporPressures])
            elif Psat_extrpolation == 'ABC':
                Psats_data.append([i.DIPPR101_ABC_high for i in VaporPressures])
            # Other option: raise?
            self._Psats_data = Psats_data

        self.N = len(VaporPressures)

        self.HeatCapacityGases = HeatCapacityGases
        self.Cpgs_poly_fit, self._Cpgs_data = self._setup_Cpigs(HeatCapacityGases)

        self.HeatCapacityLiquids = HeatCapacityLiquids
        if HeatCapacityLiquids is not None:
            self.Cpls_poly_fit, self._Cpls_data = self._setup_Cpigs(HeatCapacityLiquids)
            T_REF_IG = self.T_REF_IG
            T_REF_IG_INV = 1.0/T_REF_IG
            self.Hvaps_T_ref = [obj(T_REF_IG) for obj in EnthalpyVaporizations]
            self.dSvaps_T_ref = [T_REF_IG_INV*dH for dH in self.Hvaps_T_ref]

        self.use_eos_volume = use_eos_volume
        self.VolumeLiquids = VolumeLiquids
        self.Vms_sat_poly_fit = ((not use_eos_volume and all(i.method == POLY_FIT for i in VolumeLiquids)) if VolumeLiquids is not None else False)
        if self.Vms_sat_poly_fit:
            self._Vms_sat_data = [[i.poly_fit_Tmin for i in VolumeLiquids],
                                 [i.poly_fit_Tmin_slope for i in VolumeLiquids],
                                 [i.poly_fit_Tmin_value for i in VolumeLiquids],
                                 [i.poly_fit_Tmax for i in VolumeLiquids],
                                 [i.poly_fit_Tmax_slope for i in VolumeLiquids],
                                 [i.poly_fit_Tmax_value for i in VolumeLiquids],
                                 [i.poly_fit_coeffs for i in VolumeLiquids],
                                 [i.poly_fit_d_coeffs for i in VolumeLiquids],
                                 [i.poly_fit_d2_coeffs for i in VolumeLiquids],
                                 [i.poly_fit_Tmin_quadratic for i in VolumeLiquids],
                                 ]
#            low_fits = self._Vms_sat_data[9]
#            for i in range(self.N):
#                low_fits[i][0] = max(0, low_fits[i][0])

        self.VolumeSupercriticalLiquids = VolumeSupercriticalLiquids
        self.Vms_supercritical_poly_fit = all(i.method == POLY_FIT for i in VolumeSupercriticalLiquids) if VolumeSupercriticalLiquids is not None else False
        if self.Vms_supercritical_poly_fit:
            self.Vms_supercritical_data = [[i.poly_fit_Tmin for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmin_slope for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmin_value for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmax for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmax_slope for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmax_value for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_coeffs for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_d_coeffs for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_d2_coeffs for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmin_quadratic for i in VolumeSupercriticalLiquids],
                                 ]


        self.incompressible = not use_Tait
        self.use_Tait = use_Tait
        if self.use_Tait:
            Tait_B_data, Tait_C_data = [[] for i in range(9)], [[] for i in range(9)]
            for v in VolumeLiquids:
                for (d, store) in zip(v.Tait_data(), [Tait_B_data, Tait_C_data]):
                    for i in range(len(d)):
                        store[i].append(d[i])
            self._Tait_B_data = Tait_B_data
            self._Tait_C_data = Tait_C_data


        self.EnthalpyVaporizations = EnthalpyVaporizations
        self.Hvap_poly_fit = all(i.method == POLY_FIT for i in EnthalpyVaporizations) if EnthalpyVaporizations is not None else False
        if self.Hvap_poly_fit:
            self._Hvap_data = [[i.poly_fit_Tmin for i in EnthalpyVaporizations],
                              [i.poly_fit_Tmax for i in EnthalpyVaporizations],
                              [i.poly_fit_Tc for i in EnthalpyVaporizations],
                              [1.0/i.poly_fit_Tc for i in EnthalpyVaporizations],
                              [i.poly_fit_coeffs for i in EnthalpyVaporizations]]



        if GibbsExcessModel is None:
            GibbsExcessModel = IdealSolution(T=T, xs=zs)

        self.GibbsExcessModel = GibbsExcessModel
        self.eos_pure_instances = eos_pure_instances

        self.equilibrium_basis = equilibrium_basis
        self.caloric_basis = caloric_basis

        if equilibrium_basis is not None:
            if equilibrium_basis == 'Poynting':
                self.use_Poynting = True
                self.use_phis_sat = False
            elif equilibrium_basis == 'Poynting&PhiSat':
                self.use_Poynting = True
                self.use_phis_sat = True
            elif equilibrium_basis == 'PhiSat':
                self.use_phis_sat = True
                self.use_Poynting = False
            elif equilibrium_basis == 'Psat':
                self.use_phis_sat = False
                self.use_Poynting = False
        else:
            self.use_Poynting = use_Poynting
            self.use_phis_sat = use_phis_sat

        if caloric_basis is not None:
            if caloric_basis == 'Poynting':
                self.use_Poynting_caloric = True
                self.use_phis_sat_caloric = False
                self.use_Hvap_caloric = False
            elif caloric_basis == 'Poynting&PhiSat':
                self.use_Poynting_caloric = True
                self.use_phis_sat_caloric = True
                self.use_Hvap_caloric = False
            elif caloric_basis == 'PhiSat':
                self.use_phis_sat_caloric = True
                self.use_Poynting_caloric = False
                self.use_Hvap_caloric = False
            elif caloric_basis == 'Psat':
                self.use_phis_sat_caloric = False
                self.use_Poynting_caloric = False
                self.use_Hvap_caloric = False
            elif caloric_basis == 'Hvap':
                self.use_phis_sat_caloric = False
                self.use_Poynting_caloric = False
                self.use_Hvap_caloric = True
        else:
            self.use_Poynting_caloric = use_Poynting
            self.use_phis_sat_caloric = use_phis_sat
            self.use_Hvap_caloric = use_Hvap_caloric



        if henry_components is None:
            henry_components = [False]*self.N
        self.has_henry_components = any(henry_components)
        self.henry_components = henry_components
        self.henry_data = henry_data

        self.composition_independent = isinstance(GibbsExcessModel, IdealSolution) and not self.has_henry_components

        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs

        if T is not None and P is not None and zs is not None:
            self.T = T
            self.P = P
            self.zs = zs

    def to_TP_zs(self, T, P, zs):
        T_equal = hasattr(self, 'T') and T == self.T
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        new.N = self.N

        self.transfer_data(new, zs, T, T_equal)
        return new


    def to(self, zs, T=None, P=None, V=None):
        try:
            T_equal = T == self.T
        except:
            T_equal = False

        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.N = self.N

        if T is not None:
            if P is not None:
                new.T = T
                new.P = P
            elif V is not None:
                def to_solve(P):
                    return self.to_TP_zs(T, P, zs).V() - V
                P = secant(to_solve, 0.0002, xtol=1e-8, ytol=1e-10)
                new.P = P
        elif P is not None and V is not None:
            def to_solve(T):
                return self.to_TP_zs(T, P, zs).V() - V
            T = secant(to_solve, 300, xtol=1e-9, ytol=1e-5)
            new.T = T
        else:
            raise ValueError("Two of T, P, or V are needed")

        self.transfer_data(new, zs, T, T_equal)
        return new

    def transfer_data(self, new, zs, T, T_equal):
        new.VaporPressures = self.VaporPressures
        new.VolumeLiquids = self.VolumeLiquids
        new.eos_pure_instances = self.eos_pure_instances
        new.HeatCapacityGases = self.HeatCapacityGases
        new.EnthalpyVaporizations = self.EnthalpyVaporizations
        new.HeatCapacityLiquids = self.HeatCapacityLiquids


        new.Psats_poly_fit = self.Psats_poly_fit
        new._Psats_data = self._Psats_data
        new.Psat_extrpolation = self.Psat_extrpolation

        new.Cpgs_poly_fit = self.Cpgs_poly_fit
        new._Cpgs_data = self._Cpgs_data

        new.Cpls_poly_fit = self.Cpls_poly_fit
        new._Cpls_data = self._Cpls_data

        new.Vms_sat_poly_fit = self.Vms_sat_poly_fit
        new._Vms_sat_data = self._Vms_sat_data

        new._Hvap_data = self._Hvap_data
        new.Hvap_poly_fit = self.Hvap_poly_fit

        new.incompressible = self.incompressible

        new.equilibrium_basis = self.equilibrium_basis
        new.caloric_basis = self.caloric_basis

        new.use_phis_sat = self.use_phis_sat
        new.use_Poynting = self.use_Poynting
        new.P_DEPENDENT_H_LIQ = self.P_DEPENDENT_H_LIQ
        new.use_eos_volume = self.use_eos_volume
        new.use_Hvap_caloric = self.use_Hvap_caloric

        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs

        new.henry_data = self.henry_data
        new.henry_components = self.henry_components
        new.has_henry_components = self.has_henry_components

        new.composition_independent = self.composition_independent

        new.use_Tait = self.use_Tait
        new._Tait_B_data = self._Tait_B_data
        new._Tait_C_data = self._Tait_C_data


        if T_equal and (self.composition_independent or self.zs is zs):
            # Allow the composition inconsistency as it is harmless
            new.GibbsExcessModel = self.GibbsExcessModel
        else:
            new.GibbsExcessModel = self.GibbsExcessModel.to_T_xs(T=T, xs=zs)

        try:
            if T_equal:
                if not self.has_henry_components:
                    try:
                        new._Psats = self._Psats
                        new._dPsats_dT = self._dPsats_dT
                        new._d2Psats_dT2 = self._d2Psats_dT2
                    except:
                        pass

                try:
                    new._Vms_sat = self._Vms_sat
                    new._Vms_sat_dT = self._Vms_sat_dT
                    new._d2Vms_sat_dT2 = self._d2Vms_sat_dT2
                except:
                    pass
                try:
                    new._Cpigs = self._Cpigs
                except:
                    pass
                try:
                    new._Cpig_integrals_over_T_pure = self._Cpig_integrals_over_T_pure
                except:
                    pass
                try:
                    new._Cpig_integrals_pure = self._Cpig_integrals_pure
                except:
                    pass
        except:
            pass
        return new


    def Henry_matrix(self):
        '''Generate a matrix of all component-solvent Henry's law values
        Shape N*N; solvent/solvent and gas/gas values are all None, as well
        as solvent/gas values where the parameters are unavailable.
        '''

    def Henry_constants(self):
        '''Mix the parameters in `Henry_matrix` into values to take the place
        in Psats.
        '''

    def Psats_T_ref(self):
        try:
            return self._Psats_T_ref
        except AttributeError:
            pass
        VaporPressures, cmps = self.VaporPressures, range(self.N)
        T_REF_IG = self.T_REF_IG
        self._Psats_T_ref = [VaporPressures[i](T_REF_IG) for i in cmps]
        return self._Psats_T_ref

    def Psats_at(self, T):
        if self.Psats_poly_fit:
            return self._Psats_at_poly_fit(T, self._Psats_data, range(self.N))
        VaporPressures = self.VaporPressures
        return [VaporPressures[i](T) for i in range(self.N)]

    @staticmethod
    def _Psats_at_poly_fit(T, Psats_data, cmps):
        Psats = []
        T_inv = 1.0/T
        logT = log(T)
        Tmins, Tmaxes, coeffs = Psats_data[0], Psats_data[3], Psats_data[6]
        for i in cmps:
            if T < Tmins[i]:
                A, B, C = Psats_data[9][i]
                Psat = (A + B*T_inv + C*logT)
#                    A, B = _Psats_data[9][i]
#                    Psat = (A - B*T_inv)
#                    Psat = (T - Tmins[i])*_Psats_data[1][i] + _Psats_data[2][i]
            elif T > Tmaxes[i]:
                A, B, C = Psats_data[10][i]
                Psat = (A + B*T_inv + C*logT)
#                A, B = _Psats_data[10][i]
#                Psat = (A - B*T_inv)
#                Psat = (T - Tmaxes[i])*_Psats_data[4][i] + _Psats_data[5][i]
            else:
                Psat = 0.0
                for c in coeffs[i]:
                    Psat = Psat*T + c
            try:
                Psats.append(exp(Psat))
            except:
                Psats.append(1.6549840276802644e+300)

        return Psats

    def Psats(self):
        try:
            return self._Psats
        except AttributeError:
            pass
        T, cmps = self.T, range(self.N)
        if self.Psats_poly_fit:
            self._Psats = Psats = self._Psats_at_poly_fit(T, self._Psats_data, cmps)
#            _Psats_data = self._Psats_data
#            Tmins, Tmaxes, coeffs = _Psats_data[0], _Psats_data[3], _Psats_data[6]
#            for i in cmps:
#                if T < Tmins[i]:
#                    A, B, C = _Psats_data[9][i]
#                    Psat = (A + B*T_inv + C*logT)
##                    A, B = _Psats_data[9][i]
##                    Psat = (A - B*T_inv)
##                    Psat = (T - Tmins[i])*_Psats_data[1][i] + _Psats_data[2][i]
#                elif T > Tmaxes[i]:
#                    Psat = (T - Tmaxes[i])*_Psats_data[4][i] + _Psats_data[5][i]
#                else:
#                    Psat = 0.0
#                    for c in coeffs[i]:
#                        Psat = Psat*T + c
#                Psats.append(exp(Psat))
            return Psats


        self._Psats = Psats = []
        for i in self.VaporPressures:
            Psats.append(i.T_dependent_property(T))

        if self.has_henry_components:
            henry_components = self.henry_components
            henry_data = self.henry_data
            zs = self.zs

            for i in range(self.N):
#                Vcs = [1, 1, 1]
                Vcs = [5.6000000000000006e-05, 0.000168, 7.340000000000001e-05]
                if henry_components[i]:
                    # WORKING - Need a bunch of conversions of data in terms of other values
                    # into this basis
                    d = henry_data[i]
                    z_sum = 0.0
                    logH = 0.0
                    for j in cmps:
                        if d[j]:
                            r = d[j]
                            t = T
#                            t = T - 273.15
                            log_Hi = (r[0] + r[1]/t + r[2]*log(t) + r[3]*t + r[4]/t**2)
#                            print(log_Hi)
                            wi = zs[j]*Vcs[j]**(2.0/3.0)/sum([zs[_]*Vcs[_]**(2.0/3.0) for _ in cmps if d[_]])
#                            print(wi)

                            logH += wi*log_Hi
#                            logH += zs[j]*log_Hi
                            z_sum += zs[j]

#                    print(logH, z_sum)
                    z_sum = 1
                    Psats[i] = exp(logH/z_sum)*1e5 # bar to Pa


        return Psats

#    def PIP(self):
#        # Force liquid
#        return 2.0

    @staticmethod
    def _dPsats_dT_at_poly_fit(T, Psats_data, cmps, Psats):
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        dPsats_dT = []
        Tmins, Tmaxes, dcoeffs, coeffs_low, coeffs_high = Psats_data[0], Psats_data[3], Psats_data[7], Psats_data[9], Psats_data[10]
        for i in cmps:
            if T < Tmins[i]:
#                    A, B = _Psats_data[9][i]
#                    dPsat_dT = B*Tinv2*Psats[i]
                dPsat_dT = Psats[i]*(-coeffs_low[i][1]*Tinv2 + coeffs_low[i][2]*T_inv)
#                    dPsat_dT = _Psats_data[1][i]*Psats[i]#*exp((T - Tmins[i])*_Psats_data[1][i]
                                             #   + _Psats_data[2][i])
            elif T > Tmaxes[i]:
                dPsat_dT = Psats[i]*(-coeffs_high[i][1]*Tinv2 + coeffs_high[i][2]*T_inv)

#                dPsat_dT = _Psats_data[4][i]*Psats[i]#*exp((T - Tmaxes[i])
#                                                    #*_Psats_data[4][i]
#                                                    #+ _Psats_data[5][i])
            else:
                dPsat_dT = 0.0
                for c in dcoeffs[i]:
                    dPsat_dT = dPsat_dT*T + c
#                    v, der = horner_and_der(coeffs[i], T)
                dPsat_dT *= Psats[i]
            dPsats_dT.append(dPsat_dT)
        return dPsats_dT

    def dPsats_dT_at(self, T, Psats=None):
        if Psats is None:
            Psats = self.Psats_at(T)
        if self.Psats_poly_fit:
            return self._dPsats_dT_at_poly_fit(T, self._Psats_data, range(self.N), Psats)
        return [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]

    def dPsats_dT(self):
        try:
            return self._dPsats_dT
        except:
            pass
        T, cmps = self.T, range(self.N)
        # Need to reset the method because for the T bounded solver,
        # will normally get a different than prefered method as it starts
        # at the boundaries
        if self.Psats_poly_fit:
            try:
                Psats = self._Psats
            except AttributeError:
                Psats = self.Psats()
            self._dPsats_dT = dPsats_dT = self._dPsats_dT_at_poly_fit(T, self._Psats_data, cmps, Psats)
            return dPsats_dT

        self._dPsats_dT = dPsats_dT = [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]
        return dPsats_dT

    def d2Psats_dT2(self):
        try:
            return self._d2Psats_dT2
        except:
            pass
        try:
            Psats = self._Psats
        except AttributeError:
            Psats = self.Psats()
        try:
            dPsats_dT = self._dPsats_dT
        except AttributeError:
            dPsats_dT = self.dPsats_dT()
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        T_inv2 = T_inv*T_inv
        Tinv3 = T_inv*T_inv*T_inv

        self._d2Psats_dT2 = d2Psats_dT2 = []
        if self.Psats_poly_fit:
            Psats_data = self._Psats_data
            Tmins, Tmaxes, d2coeffs = Psats_data[0], Psats_data[3], Psats_data[8]
            for i in cmps:
                if T < Tmins[i]:
#                    A, B = _Psats_data[9][i]
#                    d2Psat_dT2 = B*Psats[i]*(B*T_inv - 2.0)*Tinv3
                    A, B, C = Psats_data[9][i]
                    x0 = (B*T_inv - C)
                    d2Psat_dT2 = Psats[i]*(2.0*B*T_inv - C + x0*x0)*T_inv2
#                    d2Psat_dT2 = _Psats_data[1][i]*dPsats_dT[i]
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    x0 = (B*T_inv - C)
                    d2Psat_dT2 = Psats[i]*(2.0*B*T_inv - C + x0*x0)*T_inv2
#                    d2Psat_dT2 = _Psats_data[4][i]*dPsats_dT[i]
                else:
                    d2Psat_dT2 = 0.0
                    for c in d2coeffs[i]:
                        d2Psat_dT2 = d2Psat_dT2*T + c
                    d2Psat_dT2 = (dPsats_dT[i]*dPsats_dT[i]/Psats[i] + Psats[i]*d2Psat_dT2)
                d2Psats_dT2.append(d2Psat_dT2)
            return d2Psats_dT2

        self._d2Psats_dT2 = d2Psats_dT2 = [VaporPressure.T_dependent_property_derivative(T=T, n=2)
                     for VaporPressure in self.VaporPressures]
        return d2Psats_dT2

    def lnPsats(self):
        try:
            return self._lnPsats
        except AttributeError:
            pass
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        logT = log(T)
        lnPsats = []
        if self.Psats_poly_fit:
            Psats_data = self._Psats_data
            Tmins, Tmaxes, coeffs = Psats_data[0], Psats_data[3], Psats_data[6]
            for i in cmps:
                if T < Tmins[i]:
                    A, B, C = Psats_data[9][i]
                    Psat = (A + B*T_inv + C*logT)
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    Psat = (A + B*T_inv + C*logT)
#                    Psat = (T - Tmaxes[i])*_Psats_data[4][i] + _Psats_data[5][i]
                else:
                    Psat = 0.0
                    for c in coeffs[i]:
                        Psat = Psat*T + c
                lnPsats.append(Psat)
            self._lnPsats = lnPsats
            return lnPsats
        self._lnPsats = [log(i) for i in self.Psats()]
        return self._lnPsats

    def dlnPsats_dT(self):
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        if self.Psats_poly_fit:
            Psats_data = self._Psats_data
            Tmins, Tmaxes, dcoeffs = Psats_data[0], Psats_data[3], Psats_data[7]
            dlnPsats_dT = []
            for i in cmps:
                if T < Tmins[i]:
                    A, B, C = Psats_data[9][i]
                    dPsat_dT = (-B*Tinv2 + C*T_inv)
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    dPsat_dT = (-B*Tinv2 + C*T_inv)
#                    dPsat_dT = _Psats_data[4][i]
                else:
                    dPsat_dT = 0.0
                    for c in dcoeffs[i]:
                        dPsat_dT = dPsat_dT*T + c
                dlnPsats_dT.append(dPsat_dT)
            return dlnPsats_dT

    def d2lnPsats_dT2(self):
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        T_inv2 = T_inv*T_inv
        Tinv3 = T_inv*T_inv*T_inv
        if self.Psats_poly_fit:
            Psats_data = self._Psats_data
            Tmins, Tmaxes, d2coeffs = Psats_data[0], Psats_data[3], Psats_data[8]
            d2lnPsats_dT2 = []
            for i in cmps:
                if T < Tmins[i]:
                    A, B, C = Psats_data[9][i]
                    d2lnPsat_dT2 = (2.0*B*T_inv - C)*T_inv2
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    d2lnPsat_dT2 = (2.0*B*T_inv - C)*T_inv2
#                    d2lnPsat_dT2 = 0.0
                else:
                    d2lnPsat_dT2 = 0.0
                    for c in d2coeffs[i]:
                        d2lnPsat_dT2 = d2lnPsat_dT2*T + c
                d2lnPsats_dT2.append(d2lnPsat_dT2)
            return d2lnPsats_dT2

    def dPsats_dT_over_Psats(self):
        try:
            return self._dPsats_dT_over_Psats
        except AttributeError:
            pass
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        if self.Psats_poly_fit:
            dPsat_dT_over_Psats = []
            Psats_data = self._Psats_data
            Tmins, Tmaxes, dcoeffs, low_coeffs, high_coeffs = Psats_data[0], Psats_data[3], Psats_data[7], Psats_data[9], Psats_data[10]
            for i in cmps:
                if T < Tmins[i]:
                    dPsat_dT_over_Psat = (-low_coeffs[i][1]*Tinv2 + low_coeffs[i][2]*T_inv)
                elif T > Tmaxes[i]:
                    dPsat_dT_over_Psat = (-high_coeffs[i][1]*Tinv2 + high_coeffs[i][2]*T_inv)
#                    dPsat_dT_over_Psat = _Psats_data[4][i]
                else:
                    dPsat_dT_over_Psat = 0.0
                    for c in dcoeffs[i]:
                        dPsat_dT_over_Psat = dPsat_dT_over_Psat*T + c
                dPsat_dT_over_Psats.append(dPsat_dT_over_Psat)
            self._dPsats_dT_over_Psats = dPsat_dT_over_Psats
            return dPsat_dT_over_Psats

        dPsat_dT_over_Psats = [i/j for i, j in zip(self.dPsats_dT(), self.Psats())]
        self._dPsats_dT_over_Psats = dPsat_dT_over_Psats
        return dPsat_dT_over_Psats

    def d2Psats_dT2_over_Psats(self):
        try:
            return self._d2Psats_dT2_over_Psats
        except AttributeError:
            pass
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        Tinv4 = Tinv2*Tinv2
        c0 = (T + T)*Tinv4
        if self.Psats_poly_fit:
            d2Psat_dT2_over_Psats = []
            Psats_data = self._Psats_data
            Tmins, Tmaxes, dcoeffs, low_coeffs, high_coeffs = Psats_data[0], Psats_data[3], Psats_data[7], Psats_data[9], Psats_data[10]
            for i in cmps:
                if T < Tmins[i]:
                    B, C = low_coeffs[i][1], low_coeffs[i][2]
                    x0 = (B - C*T)
                    d2Psat_dT2_over_Psat = c0*B - C*Tinv2 + x0*x0*Tinv4
#                    d2Psat_dT2_over_Psat = (2*B*T - C*T**2 + (B - C*T)**2)/T**4
                elif T > Tmaxes[i]:
                    B, C = high_coeffs[i][1], high_coeffs[i][2]
                    x0 = (B - C*T)
                    d2Psat_dT2_over_Psat = c0*B - C*Tinv2 + x0*x0*Tinv4
                else:
                    dPsat_dT = 0.0
                    d2Psat_dT2 = 0.0
                    for a in dcoeffs[i]:
                        d2Psat_dT2 = T*d2Psat_dT2 + dPsat_dT
                        dPsat_dT = T*dPsat_dT + a
                    d2Psat_dT2_over_Psat = dPsat_dT*dPsat_dT + d2Psat_dT2

                d2Psat_dT2_over_Psats.append(d2Psat_dT2_over_Psat)
            self._d2Psats_dT2_over_Psats = d2Psat_dT2_over_Psats
            return d2Psat_dT2_over_Psats

        d2Psat_dT2_over_Psats = [i/j for i, j in zip(self.d2Psats_dT2(), self.Psats())]
        self._d2Psats_dT2_over_Psats = d2Psat_dT2_over_Psats
        return d2Psat_dT2_over_Psats

    @staticmethod
    def _Vms_sat_at(T, Vms_sat_data, cmps):
        Tmins, Tmaxes, coeffs, coeffs_Tmin = Vms_sat_data[0], Vms_sat_data[3], Vms_sat_data[6], Vms_sat_data[9]
        Vms_sat = []
        for i in cmps:
            if T < Tmins[i]:
                Vm = 0.0
                for c in coeffs_Tmin[i]:
                    Vm = Vm*T + c
#                    Vm = (T - Tmins[i])*Vms_sat_data[1][i] + Vms_sat_data[2][i]
            elif T > Tmaxes[i]:
                Vm = (T - Tmaxes[i])*Vms_sat_data[4][i] + Vms_sat_data[5][i]
            else:
                Vm = 0.0
                for c in coeffs[i]:
                    Vm = Vm*T + c
            Vms_sat.append(Vm)
        return Vms_sat

    def Vms_sat_at(self, T):
        if self.Vms_sat_poly_fit:
            return self._Vms_sat_at(T, self._Vms_sat_data, range(self.N))
        VolumeLiquids = self.VolumeLiquids
        return [VolumeLiquids[i].T_dependent_property(T) for i in range(self.N)]

    def Vms_sat(self):
        try:
            return self._Vms_sat
        except AttributeError:
            pass
        T = self.T
        if self.Vms_sat_poly_fit:
#            self._Vms_sat = evaluate_linear_fits(self._Vms_sat_data, T)
#            return self._Vms_sat
            self._Vms_sat = Vms_sat = self._Vms_sat_at(T, self._Vms_sat_data, range(self.N))
            return Vms_sat
        elif self.use_eos_volume:
            Vms = []
            eoss = self.eos_pure_instances
            Psats = self.Psats()
            for i, e in enumerate(eoss):
                if T < e.Tc:
                    Vms.append(e.V_l_sat(T))
                else:
                    e = e.to(T=T, P=Psats[i])
                    try:
                        Vms.append(e.V_l)
                    except:
                        Vms.append(e.V_g)
            self._Vms_sat = Vms
            return Vms


        VolumeLiquids = self.VolumeLiquids
#        Psats = self.Psats()
#        self._Vms_sat = [VolumeLiquids[i](T, Psats[i]) for i in range(self.N)]
        self._Vms_sat = [VolumeLiquids[i].T_dependent_property(T) for i in range(self.N)]
        return self._Vms_sat

    @staticmethod
    def _dVms_sat_dT_at(T, Vms_sat_data, cmps):
        Vms_sat_data = Vms_sat_data
        Vms_sat_dT = []
        Tmins, Tmaxes, dcoeffs = Vms_sat_data[0], Vms_sat_data[3], Vms_sat_data[7]
        for i in cmps:
            if T < Tmins[i]:
                dVm = horner_and_der2(Vms_sat_data[9][i], T)[1]
            elif T > Tmaxes[i]:
                dVm = Vms_sat_data[4][i]
            else:
                dVm = 0.0
                for c in dcoeffs[i]:
                    dVm = dVm*T + c
            Vms_sat_dT.append(dVm)
        return Vms_sat_dT

    def dVms_sat_dT_at(self, T):
        if self.Vms_sat_poly_fit:
            return self._dVms_sat_dT_at(T, self._Vms_sat_data, range(self.N))
        return [obj.T_dependent_property_derivative(T=T) for obj in VolumeLiquids]

    def dVms_sat_dT(self):
        try:
            return self._Vms_sat_dT
        except:
            pass
        T = self.T

        if self.Vms_sat_poly_fit:
#            self._Vms_sat_dT = evaluate_linear_fits_d(self._Vms_sat_data, T)
            self._Vms_sat_dT = self._dVms_sat_dT_at(T, self._Vms_sat_data, range(self.N))
            return self._Vms_sat_dT

        VolumeLiquids = self.VolumeLiquids
        self._Vms_sat_dT = Vms_sat_dT = [obj.T_dependent_property_derivative(T=T) for obj in VolumeLiquids]
        return Vms_sat_dT

    def d2Vms_sat_dT2(self):
        try:
            return self._d2Vms_sat_dT2
        except:
            pass

        T = self.T

        if self.Vms_sat_poly_fit:
#            self._d2Vms_sat_dT2 = evaluate_linear_fits_d2(self._Vms_sat_data, T)
#            return self._d2Vms_sat_dT2
            d2Vms_sat_dT2 = self._d2Vms_sat_dT2 = []

            Vms_sat_data = self._Vms_sat_data
            Tmins, Tmaxes, d2coeffs = Vms_sat_data[0], Vms_sat_data[3], Vms_sat_data[8]
            for i in range(self.N):
                d2Vm = 0.0
                if Tmins[i] < T < Tmaxes[i]:
                    for c in d2coeffs[i]:
                        d2Vm = d2Vm*T + c
                elif T < Tmins[i]:
                    d2Vm = horner_and_der2(Vms_sat_data[9][i], T)[2]
                d2Vms_sat_dT2.append(d2Vm)
            return d2Vms_sat_dT2

        VolumeLiquids = self.VolumeLiquids
        self._d2Vms_sat_dT2 = [obj.T_dependent_property_derivative(T=T, order=2) for obj in VolumeLiquids]
        return self._d2Vms_sat_dT2

    def Vms_sat_T_ref(self):
        try:
            return self._Vms_sat_T_ref
        except AttributeError:
            pass
        T_REF_IG = self.T_REF_IG
        if self.Vms_sat_poly_fit:
            self._Vms_sat_T_ref = evaluate_linear_fits(self._Vms_sat_data, T_REF_IG)
        else:
            VolumeLiquids, cmps = self.VolumeLiquids, range(self.N)
            self._Vms_sat_T_ref = [VolumeLiquids[i].T_dependent_property(T_REF_IG) for i in cmps]
        return self._Vms_sat_T_ref

    def dVms_sat_dT_T_ref(self):
        try:
            return self._dVms_sat_dT_T_ref
        except AttributeError:
            pass
        T_REF_IG = self.T_REF_IG
        if self.Vms_sat_poly_fit:
            self._dVms_sat_dT_T_ref = evaluate_linear_fits_d(self._Vms_sat_data, T)
        else:
            VolumeLiquids, cmps = self.VolumeLiquids, range(self.N)
            self._dVms_sat_dT_T_ref = [VolumeLiquids[i].T_dependent_property_derivative(T_REF_IG) for i in cmps]
        return self._dVms_sat_dT_T_ref

    def Vms(self):
        # Fill in tait/eos function to be called instead of Vms_sat
        return self.Vms_sat()

    def dVms_dT(self):
        return self.dVms_sat_dT()

    def d2Vms_dT2(self):
        return self.d2Vms_sat_dT2()

    def dVms_dP(self):
        return [0.0]*self.N

    def d2Vms_dP2(self):
        return [0.0]*self.N

    def d2Vms_dPdT(self):
        return [0.0]*self.N

    def Hvaps(self):
        try:
            return self._Hvaps
        except AttributeError:
            pass
        T, EnthalpyVaporizations, cmps = self.T, self.EnthalpyVaporizations, range(self.N)

        self._Hvaps = Hvaps = []
        if self.Hvap_poly_fit:
            Hvap_data = self._Hvap_data
            Tmins, Tmaxes, Tcs, Tcs_inv, coeffs = Hvap_data[0], Hvap_data[1], Hvap_data[2], Hvap_data[3], Hvap_data[4]
            for i in cmps:
                Hvap = 0.0
                if T < Tcs[i]:
                    x = log(1.0 - T*Tcs_inv[i])
                    for c in coeffs[i]:
                        Hvap = Hvap*x + c
    #                    Vm = horner(coeffs[i], log(1.0 - T*Tcs_inv[i])
                Hvaps.append(Hvap)
            return Hvaps

        self._Hvaps = Hvaps = [EnthalpyVaporizations[i](T) for i in cmps]
        for i in cmps:
            if Hvaps[i] is None:
                Hvaps[i] = 0.0
        return Hvaps

    def dHvaps_dT(self):
        try:
            return self._dHvaps_dT
        except AttributeError:
            pass
        T, EnthalpyVaporizations, cmps = self.T, self.EnthalpyVaporizations, range(self.N)

        self._dHvaps_dT = dHvaps_dT = []
        if self.Hvap_poly_fit:
            Hvap_data = self._Hvap_data
            Tmins, Tmaxes, Tcs, Tcs_inv, coeffs = Hvap_data[0], Hvap_data[1], Hvap_data[2], Hvap_data[3], Hvap_data[4]
            for i in cmps:
                dHvap_dT = 0.0
                if T < Tcs[i]:
                    p = log((Tcs[i] - T)*Tcs_inv[i])
                    x = 1.0
                    a = 1.0
                    for c in coeffs[i][-2::-1]:
                        dHvap_dT += a*c*x
                        x *= p
                        a += 1.0
                    dHvap_dT /= T - Tcs[i]

                dHvaps_dT.append(dHvap_dT)
            return dHvaps_dT

        self._dHvaps_dT = dHvaps_dT = [EnthalpyVaporizations[i].T_dependent_property_derivative(T) for i in cmps]
        for i in cmps:
            if dHvaps_dT[i] is None:
                dHvaps_dT[i] = 0.0
        return dHvaps_dT

    def Hvaps_T_ref(self):
        try:
            return self._Hvaps_T_ref
        except AttributeError:
            pass
        EnthalpyVaporizations, cmps = self.EnthalpyVaporizations, range(self.N)
        T_REF_IG = self.T_REF_IG
        self._Hvaps_T_ref = [EnthalpyVaporizations[i](T_REF_IG) for i in cmps]
        return self._Hvaps_T_ref

    def Poyntings_at(self, T, P, Psats=None, Vms=None):
        if not self.use_Poynting:
            return [1.0]*self.N

        cmps = range(self.N)
        if Psats is None:
            Psats = self.Psats_at(T)
        if Vms is None:
            Vms = self.Vms_sat_at(T)
        RT_inv = 1.0/(R*T)
        return [exp(Vms[i]*(P-Psats[i])*RT_inv) for i in cmps]

    def Poyntings(self):
        r'''Method to calculate and return the Poynting pressure correction
        factors of the phase, [-].

        .. math::
            \text{Poynting}_i = \exp\left(\frac{V_{m,i}(P-P_{sat})}{RT}\right)

        Returns
        -------
        Poyntings : list[float]
            Poynting pressure correction factors, [-]

        Notes
        -----
        The above formula is correct for pressure-independent molar volumes.
        When the volume does depend on pressure, the full expression is:

        .. math::
            \text{Poynting} = \exp\left[\frac{\int_{P_i^{sat}}^P V_i^l dP}{RT}\right]

        When a specified model e.g. the Tait equation is used, an analytical
        integral of this term is normally available.

        '''
        try:
            return self._Poyntings
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._Poyntings = [1.0]*self.N
            return self._Poyntings

        T, P = self.T, self.P
        try:
            Psats = self._Psats
        except AttributeError:
            Psats = self.Psats()
        try:
            Vms_sat = self._Vms_sat
        except AttributeError:
            Vms_sat = self.Vms_sat()
        RT_inv = 1.0/(R*T)
        self._Poyntings = [trunc_exp(Vml*(P-Psat)*RT_inv) for Psat, Vml in zip(Psats, Vms_sat)]
        return self._Poyntings


    def dPoyntings_dT(self):
        try:
            return self._dPoyntings_dT
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._dPoyntings_dT = [0.0]*self.N
            return self._dPoyntings_dT

        T, P = self.T, self.P

        Psats = self.Psats()
        dPsats_dT = self.dPsats_dT()
        Vms = self.Vms_sat()
        dVms_sat_dT = self.dVms_sat_dT()

        x0 = 1.0/R
        x1 = 1.0/T
        RT_inv = x0*x1

        self._dPoyntings_dT = dPoyntings_dT = []
        for i in range(self.N):
            x2 = Vms[i]
            x3 = Psats[i]

            x4 = P - x3
            x5 = x1*x2*x4
            dPoyntings_dTi = -RT_inv*(x2*dPsats_dT[i] - x4*dVms_sat_dT[i] + x5)*trunc_exp(x0*x5)
            dPoyntings_dT.append(dPoyntings_dTi)
        return dPoyntings_dT

    def dPoyntings_dT_at(self, T, P, Psats=None, Vms=None, dPsats_dT=None, dVms_sat_dT=None):
        if not self.use_Poynting:
            return [0.0]*self.N

        cmps = range(self.N)
        if Psats is None:
            Psats = self.Psats_at(T)

        if dPsats_dT is None:
            dPsats_dT = self.dPsats_dT_at(T, Psats)

        if Vms is None:
            Vms = self.Vms_sat_at(T)

        if dVms_sat_dT is None:
            dVms_sat_dT = self.dVms_sat_dT_at(T)
        x0 = 1.0/R
        x1 = 1.0/T
        dPoyntings_dT = []
        for i in range(self.N):
            x2 = Vms[i]
            x4 = P - Psats[i]
            x5 = x1*x2*x4
            dPoyntings_dTi = -x0*x1*(x2*dPsats_dT[i] - x4*dVms_sat_dT[i] + x5)*exp(x0*x5)
            dPoyntings_dT.append(dPoyntings_dTi)
        return dPoyntings_dT

    def d2Poyntings_dT2(self):
        try:
            return self._d2Poyntings_dT2
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._d2Poyntings_dT2 = [0.0]*self.N
            return self._d2Poyntings_dT2

        T, P = self.T, self.P

        Psats = self.Psats()
        dPsats_dT = self.dPsats_dT()
        d2Psats_dT2 = self.d2Psats_dT2()
        Vms = self.Vms_sat()
        dVms_sat_dT = self.dVms_sat_dT()
        d2Vms_sat_dT2 = self.d2Vms_sat_dT2()

        x6 = 1.0/T
        x7 = x6 + x6
        x11 = 1.0/R
        x12 = x11*x6
        c0 = 2.0*x6*x6

        self._d2Poyntings_dT2 = d2Poyntings_dT2 = []
        '''
        from sympy import *
        R, T, P = symbols('R, T, P')
        Vml, Psat = symbols('Vml, Psat', cls=Function)
        RT_inv = 1/(R*T)
        Poy = exp(Vml(T)*(P-Psat(T))*RT_inv)
        cse(diff(Poy, T, 2), optimizations='basic')
        '''
        for i in range(self.N):
            x0 = Vms[i]
            x1 = Psats[i]
            x2 = P - x1
            x3 = x0*x2
            x4 = dPsats_dT[i]
            x5 = x0*x4
            x8 = dVms_sat_dT[i]
            x9 = x2*x8
            x10 = x3*x6
            x50 = (x10 + x5 - x9)
            d2Poyntings_dT2i = (x12*(-x0*d2Psats_dT2[i] + x12*x50*x50
                                    + x2*d2Vms_sat_dT2[i] - 2.0*x4*x8 + x5*x7
                                    - x7*x9 + x3*c0)*exp(x10*x11))
            d2Poyntings_dT2.append(d2Poyntings_dT2i)
        return d2Poyntings_dT2

    def dPoyntings_dP(self):
        '''from sympy import *
        R, T, P, zi = symbols('R, T, P, zi')
        Vml = symbols('Vml', cls=Function)
        cse(diff(exp(Vml(T)*(P - Psati(T))/(R*T)), P), optimizations='basic')
        '''
        try:
            return self._dPoyntings_dP
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._dPoyntings_dP = [0.0]*self.N
            return self._dPoyntings_dP
        T, P = self.T, self.P
        Psats = self.Psats()

        Vms = self.Vms_sat()

        self._dPoyntings_dP = dPoyntings_dPs = []
        for i in range(self.N):
            x0 = Vms[i]/(R*T)
            dPoyntings_dPs.append(x0*exp(x0*(P - Psats[i])))
        return dPoyntings_dPs

    def d2Poyntings_dPdT(self):
        '''
        from sympy import *
        R, T, P = symbols('R, T, P')
        Vml, Psat = symbols('Vml, Psat', cls=Function)
        RT_inv = 1/(R*T)
        Poy = exp(Vml(T)*(P-Psat(T))*RT_inv)
        Poyf = symbols('Poyf')
        cse(diff(Poy, T, P).subs(Poy, Poyf), optimizations='basic')
        '''
        try:
            return self._d2Poyntings_dPdT
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._d2Poyntings_dPdT = [0.0]*self.N
            return self._d2Poyntings_dPdT

        try:
            Psats = self._Psats
        except AttributeError:
            Psats = self.Psats()
        try:
            dPsats_dT = self._dPsats_dT
        except AttributeError:
            dPsats_dT = self.dPsats_dT()
        try:
            Vms = self._Vms_sat
        except AttributeError:
            Vms = self.Vms_sat()
        try:
            dVms_sat_dT = self._dVms_sat_dT
        except AttributeError:
            dVms_sat_dT = self.dVms_sat_dT()
        try:
            Poyntings = self._Poyntings
        except AttributeError:
            Poyntings = self.Poyntings()

        x0 = R_inv
        x1 = 1.0/self.T
        P = self.P
        nRT_inv = -x0*x1
        self._d2Poyntings_dPdT = d2Poyntings_dPdT = []
        for i in range(self.N):
            x2 = Vms[i]
            x3 = x1*x2
            x4 = dVms_sat_dT[i]
            x5 = Psats[i]
            x6 = P - x5
            v = Poyntings[i]*nRT_inv*(x0*x3*(x2*dPsats_dT[i] + x3*x6 - x4*x6) + x3 - x4)
            d2Poyntings_dPdT.append(v)
        return d2Poyntings_dPdT


    d2Poyntings_dTdP = d2Poyntings_dPdT

    def phis_sat_at(self, T):
        if not self.use_phis_sat:
            return [1.0]*self.N
        phis_sat = []
        for i in self.eos_pure_instances:
            try:
                phis_sat.append(i.phi_sat(min(T, i.Tc), polish=True))
            except Exception as e:
                if T < self.PHI_SAT_IDEAL_TR*i.Tc:
                    phis_sat.append(1.0)
                else:
                    raise e
        return phis_sat

    def phis_sat(self):
        r'''Method to calculate and return the saturation fugacity coefficient
        correction factors of the phase, [-].

        These are calculated from the
        provided pure-component equations of state. This term should only be
        used with a consistent vapor-phase cubic equation of state.

        Returns
        -------
        phis_sat : list[float]
            Saturation fugacity coefficient correction factors, [-]

        Notes
        -----

        .. warning::
            This factor cannot be calculated when a compound is supercritical,
            as there is no longer any vapor-liquid pure-component equilibrium
            (by definition).

        '''
        try:
            return self._phis_sat
        except AttributeError:
            pass
        if not self.use_phis_sat:
            self._phis_sat = [1.0]*self.N
            return self._phis_sat

        T = self.T
        self._phis_sat = phis_sat = []
        for i in self.eos_pure_instances:
            try:
                phis_sat.append(i.phi_sat(min(T, i.Tc), polish=True))
            except Exception as e:
                if T < self.PHI_SAT_IDEAL_TR*i.Tc:
                    phis_sat.append(1.0)
                else:
                    raise e
        return phis_sat




    def dphis_sat_dT_at(self, T):
        if not self.use_phis_sat:
            return [0.0]*self.N
        dphis_sat_dT = []
        for i in self.eos_pure_instances:
            try:
                dphis_sat_dT.append(i.dphi_sat_dT(min(T, i.Tc)))
            except Exception as e:
                if T < self.PHI_SAT_IDEAL_TR*i.Tc:
                    dphis_sat_dT.append(0.0)
                else:
                    raise e
        return dphis_sat_dT

    def dphis_sat_dT(self):
        try:
            return self._dphis_sat_dT
        except AttributeError:
            pass

        if not self.use_phis_sat:
            self._dphis_sat_dT = [0.0]*self.N
            return self._dphis_sat_dT

        T = self.T
        self._dphis_sat_dT = dphis_sat_dT = []
        for i in self.eos_pure_instances:
            try:
                dphis_sat_dT.append(i.dphi_sat_dT(min(T, i.Tc)))
            except Exception as e:
                if T < self.PHI_SAT_IDEAL_TR*i.Tc:
                    dphis_sat_dT.append(0.0)
                else:
                    raise e
        return dphis_sat_dT

    def d2phis_sat_dT2(self):
        # Numerically implemented
        try:
            return self._d2phis_sat_dT2
        except AttributeError:
            pass
        if not self.use_phis_sat:
            self._d2phis_sat_dT2 = [0.0]*self.N
            return self._d2phis_sat_dT2

        T = self.T
        self._d2phis_sat_dT2 = d2phis_sat_dT2 = []
        for i in self.eos_pure_instances:
            try:
                d2phis_sat_dT2.append(i.d2phi_sat_dT2(min(T, i.Tc)))
            except Exception as e:
                if T < self.PHI_SAT_IDEAL_TR*i.Tc:
                    d2phis_sat_dT2.append(0.0)
                else:
                    raise e
        return d2phis_sat_dT2


    def phis_at(self, T, P, zs, Psats=None, gammas=None, phis_sat=None, Poyntings=None):
        P_inv = 1.0/P
        if Psats is None:
            Psats = self.Psats_at(T)
        if gammas is None:
            gammas = self.gammas_at(T, zs)
        if phis_sat is None:
            phis_sat = self.phis_sat_at(T)
        if Poyntings is None:
            Poyntings = self.Poyntings_at(T, P, Psats=Psats)
        return [gammas[i]*Psats[i]*Poyntings[i]*phis_sat[i]*P_inv
                for i in range(self.N)]

    def phis(self):
        r'''Method to calculate the fugacity coefficients of the
        GibbsExcessLiquid phase. Depending on the settings of the phase, can
        include the effects of activity coefficients `gammas`, pressure
        correction terms `Poyntings`, and pure component saturation fugacities
        `phis_sat` as well as the pure component vapor pressures.

        .. math::
            \phi_i = \frac{\gamma_i P_{i}^{sat} \phi_i^{sat} \text{Poynting}_i}
            {P}

        Returns
        -------
        phis : list[float]
            Fugacity coefficients of all components in the phase, [-]

        Notes
        -----
        Poyntings, gammas, and pure component saturation phis default to 1.
        '''
        try:
            return self._phis
        except AttributeError:
            pass
        P = self.P
        try:
            gammas = self._gammas
        except AttributeError:
            gammas = self.gammas()

        try:
            Psats = self._Psats
        except AttributeError:
            Psats = self.Psats()

        try:
            phis_sat = self._phis_sat
        except AttributeError:
            phis_sat = self.phis_sat()

        try:
            Poyntings = self._Poyntings
        except AttributeError:
            Poyntings = self.Poyntings()

        P_inv = 1.0/P
        self._phis = [gammas[i]*Psats[i]*Poyntings[i]*phis_sat[i]*P_inv
                for i in range(self.N)]
        return self._phis


    def lnphis(self):
        try:
            return self._lnphis
        except AttributeError:
            pass
        try:
            self._lnphis = [log(i) for i in self.phis()]
        except:
            # Zero Psats - must compute them inline
            P = self.P
            try:
                gammas = self._gammas
            except AttributeError:
                gammas = self.gammas()
            try:
                lnPsats = self._lnPsats
            except AttributeError:
                lnPsats = self.lnPsats()
            try:
                phis_sat = self._phis_sat
            except AttributeError:
                phis_sat = self.phis_sat()
            try:
                Poyntings = self._Poyntings
            except AttributeError:
                Poyntings = self.Poyntings()
            P_inv = 1.0/P
            self._lnphis = [log(gammas[i]*Poyntings[i]*phis_sat[i]*P_inv) + lnPsats[i]
                    for i in range(self.N)]

        return self._lnphis

    lnphis_G_min = lnphis

#    def fugacities(self, T, P, zs):
#        # DO NOT EDIT _ CORRECT
#        gammas = self.gammas(T, zs)
#        Psats = self._Psats(T=T)
#        if self.use_phis_sat:
#            phis = self.phis(T=T, zs=zs)
#        else:
#            phis = [1.0]*self.N
#
#        if self.use_Poynting:
#            Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
#        else:
#            Poyntings = [1.0]*self.N
#        return [zs[i]*gammas[i]*Psats[i]*Poyntings[i]*phis[i]
#                for i in range(self.N)]
#
#    def dphis_dxs(self):
#        if
    def dphis_dT(self):
        try:
            return self._dphis_dT
        except AttributeError:
            pass
        T, P, zs = self.T, self.P, self.zs
        Psats = self.Psats()
        gammas = self.gammas()

        if self.use_Poynting:
            # Evidence suggests poynting derivatives are not worth calculating
            dPoyntings_dT = self.dPoyntings_dT() #[0.0]*self.N
            Poyntings = self.Poyntings()
        else:
            dPoyntings_dT = [0.0]*self.N
            Poyntings = [1.0]*self.N

        dPsats_dT = self.dPsats_dT()

        dgammas_dT = self.GibbsExcessModel.dgammas_dT()

        if self.use_phis_sat:
            dphis_sat_dT = self.dphis_sat_dT()
            phis_sat = self.phis_sat()
        else:
            dphis_sat_dT = [0.0]*self.N
            phis_sat = [1.0]*self.N

#        print(gammas, phis_sat, Psats, Poyntings, dgammas_dT, dPoyntings_dT, dPsats_dT)
        self._dphis_dT = dphis_dTl = []
        for i in range(self.N):
            x0 = gammas[i]
            x1 = phis_sat[i]
            x2 = Psats[i]
            x3 = Poyntings[i]
            x4 = x2*x3
            x5 = x0*x1
            v = (x0*x4*dphis_sat_dT[i] + x1*x4*dgammas_dT[i] + x2*x5*dPoyntings_dT[i] + x3*x5*dPsats_dT[i])/P
            dphis_dTl.append(v)
        return dphis_dTl

    def dphis_dT_at(self, T, P, zs, phis_also=False):
        Psats = self.Psats_at(T)
        dPsats_dT = self.dPsats_dT_at(T, Psats)
        Vms = self.Vms_sat_at(T)
        dVms_sat_dT = self.dVms_sat_dT_at(T)

        gammas = self.gammas_at(T, zs)
        dgammas_dT = self.dgammas_dT_at(T, zs)

        if self.use_Poynting:
            Poyntings = self.Poyntings_at(T, P, Psats, Vms)
            dPoyntings_dT = self.dPoyntings_dT_at(T, P, Psats=Psats, Vms=Vms, dPsats_dT=dPsats_dT, dVms_sat_dT=dVms_sat_dT)
        else:
            Poyntings = [1.0]*self.N
            dPoyntings_dT = [0.0]*self.N

        if self.use_phis_sat:
            dphis_sat_dT = self.dphis_sat_dT_at(T)
            phis_sat = self.phis_sat_at(T)
        else:
            dphis_sat_dT = [0.0]*self.N
            phis_sat = [1.0]*self.N


        dphis_dT = []
        for i in range(self.N):
            x0 = gammas[i]
            x1 = phis_sat[i]
            x2 = Psats[i]
            x3 = Poyntings[i]
            x4 = x2*x3
            x5 = x0*x1
            v = (x0*x4*dphis_sat_dT[i] + x1*x4*dgammas_dT[i] + x2*x5*dPoyntings_dT[i] + x3*x5*dPsats_dT[i])/P
            dphis_dT.append(v)
        if phis_also:
            P_inv = 1.0/P
            phis = [gammas[i]*Psats[i]*Poyntings[i]*phis_sat[i]*P_inv for i in range(self.N)]
            return dphis_dT, phis
        return dphis_dT

    def dlnphis_dT(self):
        try:
            return self._dlnphis_dT
        except AttributeError:
            pass
        dphis_dT = self.dphis_dT()
        phis = self.phis()
        self._dlnphis_dT = [i/j for i, j in zip(dphis_dT, phis)]
        return self._dlnphis_dT

    def dlnphis_dP(self):
        r'''Method to calculate the pressure derivative of log fugacity
        coefficients of the phase. Depending on the settings of the phase, can
        include the effects of activity coefficients `gammas`, pressure
        correction terms `Poyntings`, and pure component saturation fugacities
        `phis_sat` as well as the pure component vapor pressures.

        .. math::
            \frac{\partial \ln \phi_i}{\partial P} =
            \frac{\frac{\partial \text{Poynting}_i}{\partial P}}
            {\text{Poynting}_i} - \frac{1}{P}

        Returns
        -------
        dlnphis_dP : list[float]
            Pressure derivative of log fugacity coefficients of all components
            in the phase, [1/Pa]

        Notes
        -----
        Poyntings, gammas, and pure component saturation phis default to 1. For
        that case, :math:`\frac{\partial \ln \phi_i}{\partial P}=\frac{1}{P}`.
        '''
        try:
            return self._dlnphis_dP
        except AttributeError:
            pass
        try:
            Poyntings = self._Poyntings
        except AttributeError:
            Poyntings = self.Poyntings()

        try:
            dPoyntings_dP = self._dPoyntings_dP
        except AttributeError:
            dPoyntings_dP = self.dPoyntings_dP()

        P_inv = 1.0/self.P

        self._dlnphis_dP = [dPoyntings_dP[i]/Poyntings[i] - P_inv for i in range(self.N)]
        return self._dlnphis_dP

    def gammas_at(self, T, zs):
        if self.composition_independent:
            return [1.0]*self.N
        return self.GibbsExcessModel.to_T_xs(T, xs).gammas()

    def dgammas_dT_at(self, T, zs):
        if self.composition_independent:
            return [0.0]*self.N
        return self.GibbsExcessModel.to_T_xs(T, zs).dgammas_dT()

    def gammas(self):
        r'''Method to calculate and return the activity coefficients of the
        phase, [-]. This is a direct call to
        :obj:`GibbsExcess.gammas <thermo.activity.GibbsExcess.gammas>`.

        Returns
        -------
        gammas : list[float]
            Activity coefficients, [-]
        '''
        try:
            return self.GibbsExcessModel._gammas
        except AttributeError:
            return self.GibbsExcessModel.gammas()

    def dgammas_dT(self):
        r'''Method to calculate and return the temperature derivative of
        activity coefficients of the phase, [-].

        This is a direct call to
        :obj:`GibbsExcess.dgammas_dT <thermo.activity.GibbsExcess.dgammas_dT>`.

        Returns
        -------
        dgammas_dT : list[float]
            First temperature derivative of the activity coefficients, [1/K]
        '''
        return self.GibbsExcessModel.dgammas_dT()

    def H_old(self):
#        try:
#            return self._H
#        except AttributeError:
#            pass
        # Untested
        T = self.T
        RT = R*T
        P = self.P
        zs, cmps = self.zs, range(self.N)
        T_REF_IG = self.T_REF_IG
        P_DEPENDENT_H_LIQ = self.P_DEPENDENT_H_LIQ

        try:
            Cpig_integrals_pure = self._Cpig_integrals_pure
        except AttributeError:
            Cpig_integrals_pure = self.Cpig_integrals_pure()

        H = 0.0

        if P_DEPENDENT_H_LIQ:
            # Page 650  Chemical Thermodynamics for Process Simulation
            # Confirmed with CoolProp via analytical integrals
            # Not actually checked numerically until Hvap is implemented though
            '''
            from scipy.integrate import *
            from CoolProp.CoolProp import PropsSI

            fluid = 'decane'
            T = 400
            Psat = PropsSI('P', 'T', T, 'Q', 0, fluid)
            P2 = Psat*100
            dP = P2 - Psat
            Vm = 1/PropsSI('DMOLAR', 'T', T, 'Q', 0, fluid)
            Vm2 = 1/PropsSI('DMOLAR', 'T', T, 'P', P2, fluid)
            dH = PropsSI('HMOLAR', 'T', T, 'P', P2, fluid) - PropsSI('HMOLAR', 'T', T, 'Q', 0, fluid)

            def to_int(P):
                Vm = 1/PropsSI('DMOLAR', 'T', T, 'P', P, fluid)
                alpha = PropsSI('ISOBARIC_EXPANSION_COEFFICIENT', 'T', T, 'P', P, fluid)
                return Vm -alpha*T*Vm
            quad(to_int, Psat, P2, epsabs=1.49e-14, epsrel=1.49e-14)[0]/dH
            '''

            if self.use_IG_Cp:
                try:
                    Psats = self._Psats
                except AttributeError:
                    Psats = self.Psats()
                try:
                    dPsats_dT = self._dPsats_dT
                except AttributeError:
                    dPsats_dT = self.dPsats_dT()
                try:
                    Vms_sat = self._Vms_sat
                except AttributeError:
                    Vms_sat = self.Vms_sat()
                try:
                    dVms_sat_dT = self._Vms_sat_dT
                except AttributeError:
                    dVms_sat_dT = self.dVms_sat_dT()

                failed_dPsat_dT = False
                try:
                    H = 0.0
                    for i in cmps:
                        dV_vap = R*T/Psats[i] - Vms_sat[i]
    #                    print( R*T/Psats[i] , Vms_sat[i])
                        # ratio of der to value might be easier?
                        dS_vap = dPsats_dT[i]*dV_vap
    #                    print(dPsats_dT[i]*dV_vap)
                        Hvap = T*dS_vap
                        H += zs[i]*(Cpig_integrals_pure[i] - Hvap)
                except ZeroDivisionError:
                    failed_dPsat_dT = True

                if failed_dPsat_dT or isinf(H):
                    # Handle the case where vapor pressure reaches zero - needs special implementations
                    dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
                    H = 0.0
                    for i in cmps:
#                        dV_vap = R*T/Psats[i] - Vms_sat[i]
#                        dS_vap = dPsats_dT[i]*dV_vap
                        Hvap = T*dPsats_dT_over_Psats[i]*RT
                        H += zs[i]*(Cpig_integrals_pure[i] - Hvap)

                if self.use_Tait:
                    dH_dP_integrals_Tait = self.dH_dP_integrals_Tait()
                    for i in cmps:
                        H += zs[i]*dH_dP_integrals_Tait[i]
                elif self.use_Poynting:
                    for i in cmps:
                        # This bit is the differential with respect to pressure
#                        dP = max(0.0, P - Psats[i]) # Breaks thermodynamic consistency
                        dP = P - Psats[i]
                        H += zs[i]*dP*(Vms_sat[i] - T*dVms_sat_dT[i])
            else:
                Psats = self.Psats()
                Vms_sat = self.Vms_sat()
                dVms_sat_dT = self.dVms_sat_dT()
                dPsats_dT = self.dPsats_dT()
                Hvaps_T_ref = self.Hvaps_T_ref()
                Cpl_integrals_pure = self._Cpl_integrals_pure()
                dVms_sat_dT_T_ref = self.dVms_sat_dT_T_ref()
                Vms_sat_T_ref = self.Vms_sat_T_ref()
                Psats_T_ref = self.Psats_T_ref()

                Hvaps = self.Hvaps()

                H = 0.0
                for i in range(self.N):
                    H += zs[i]*(Cpl_integrals_pure[i] - Hvaps_T_ref[i]) #
                    # If we can use the liquid heat capacity and prove its consistency

                    # This bit is the differential with respect to pressure
                    dP = P - Psats_T_ref[i]
                    H += zs[i]*dP*(Vms_sat_T_ref[i] - T_REF_IG*dVms_sat_dT_T_ref[i])
        else:
            Hvaps = self.Hvaps()
            for i in range(self.N):
                H += zs[i]*(Cpig_integrals_pure[i] - Hvaps[i])
        H += self.GibbsExcessModel.HE()
#        self._H = H
        return H
    del H_old

    def H(self):
        r'''Method to calculate the enthalpy of the
        :obj:`GibbsExcessLiquid` phase. Depending on the settings of the phase, this can
        include the effects of activity coefficients
        :obj:`gammas <GibbsExcessLiquid.gammas>`, pressure correction terms
        :obj:`Poyntings <GibbsExcessLiquid.Poyntings>`, and pure component
        saturation fugacities :obj:`phis_sat <GibbsExcessLiquid.phis_sat>`
        as well as the pure component vapor pressures.

        When `caloric_basis` is 'Poynting&PhiSat':

        .. math::
            H = H_{\text{excess}} + \sum_i z_i \left[-RT^2\left(
            \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
            + \frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + \frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}} \right)
            + \int_{T,ref}^T C_{p,ig} dT \right]

        When `caloric_basis` is 'PhiSat':

        .. math::
            H = H_{\text{excess}} + \sum_i z_i \left[-RT^2\left(
            \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
            + \frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            \right)
            + \int_{T,ref}^T C_{p,ig} dT \right]

        When `caloric_basis` is 'Poynting':

        .. math::
            H = H_{\text{excess}} + \sum_i z_i \left[-RT^2\left(
            + \frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + \frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}} \right)
            + \int_{T,ref}^T C_{p,ig} dT \right]

        When `caloric_basis` is 'Psat':

        .. math::
            H = H_{\text{excess}} + \sum_i z_i \left[-RT^2\left(
            + \frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
             \right)
            + \int_{T,ref}^T C_{p,ig} dT \right]

        When `caloric_basis` is 'Hvap':

        .. math::
            H = H_{\text{excess}} + \sum_i z_i\left[-H_{vap,i}
            + \int_{T,ref}^T C_{p,ig} dT \right]

        Returns
        -------
        H : float
            Enthalpy of the phase, [J/(mol)]

        Notes
        -----
        '''
        try:
            return self._H
        except AttributeError:
            pass
        H = 0.0
        T = self.T
        nRT2 = -R*T*T
        zs, cmps = self.zs, range(self.N)
        try:
            Cpig_integrals_pure = self._Cpig_integrals_pure
        except AttributeError:
            Cpig_integrals_pure = self.Cpig_integrals_pure()

        if self.use_Hvap_caloric:
            Hvaps = self.Hvaps()
            for i in range(self.N):
                H += zs[i]*(Cpig_integrals_pure[i] - Hvaps[i])
        else:
            dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
            use_Poynting, use_phis_sat = self.use_Poynting, self.use_phis_sat

            if use_Poynting:
                try:
                    Poyntings = self._Poyntings
                except AttributeError:
                    Poyntings = self.Poyntings()
                try:
                    dPoyntings_dT = self._dPoyntings_dT
                except AttributeError:
                    dPoyntings_dT = self.dPoyntings_dT()
            if use_phis_sat:
                try:
                    dphis_sat_dT = self._dphis_sat_dT
                except AttributeError:
                    dphis_sat_dT = self.dphis_sat_dT()
                try:
                    phis_sat = self._phis_sat
                except AttributeError:
                    phis_sat = self.phis_sat()

            if use_Poynting and use_phis_sat:
                for i in cmps:
                    H += zs[i]*(nRT2*(dphis_sat_dT[i]/phis_sat[i] + dPsats_dT_over_Psats[i] + dPoyntings_dT[i]/Poyntings[i])
                                + Cpig_integrals_pure[i])
            elif use_Poynting:
                for i in cmps:
                    H += zs[i]*(nRT2*(dPsats_dT_over_Psats[i] + dPoyntings_dT[i]/Poyntings[i]) + Cpig_integrals_pure[i])
            elif use_phis_sat:
                for i in cmps:
                    H += zs[i]*(nRT2*(dPsats_dT_over_Psats[i] + dphis_sat_dT[i]/phis_sat[i]) + Cpig_integrals_pure[i])
            else:
                for i in cmps:
                    H += zs[i]*(nRT2*dPsats_dT_over_Psats[i] + Cpig_integrals_pure[i])

        if not self.composition_independent:
            H += self.GibbsExcessModel.HE()
        self._H = H
        return H

    def S_old(self):
#        try:
#            return self._S
#        except AttributeError:
#            pass
        # Untested
        # Page 650  Chemical Thermodynamics for Process Simulation
        '''
        from scipy.integrate import *
        from CoolProp.CoolProp import PropsSI

        fluid = 'decane'
        T = 400
        Psat = PropsSI('P', 'T', T, 'Q', 0, fluid)
        P2 = Psat*100
        dP = P2 - Psat
        Vm = 1/PropsSI('DMOLAR', 'T', T, 'Q', 0, fluid)
        Vm2 = 1/PropsSI('DMOLAR', 'T', T, 'P', P2, fluid)
        dH = PropsSI('HMOLAR', 'T', T, 'P', P2, fluid) - PropsSI('HMOLAR', 'T', T, 'Q', 0, fluid)
        dS = PropsSI('SMOLAR', 'T', T, 'P', P2, fluid) - PropsSI('SMOLAR', 'T', T, 'Q', 0, fluid)
        def to_int2(P):
            Vm = 1/PropsSI('DMOLAR', 'T', T, 'P', P, fluid)
            alpha = PropsSI('ISOBARIC_EXPANSION_COEFFICIENT', 'T', T, 'P', P, fluid)
            return -alpha*Vm
        quad(to_int2, Psat, P2, epsabs=1.49e-14, epsrel=1.49e-14)[0]/dS
        '''
        S = 0.0
        T, P, zs, cmps = self.T, self.P, self.zs, range(self.N)
        log_zs = self.log_zs()
        for i in cmps:
            S -= zs[i]*log_zs[i]
        S *= R
        S_base = S

        T_inv = 1.0/T
        RT = R*T

        P_REF_IG_INV = self.P_REF_IG_INV

        try:
            Cpig_integrals_over_T_pure = self._Cpig_integrals_over_T_pure
        except AttributeError:
            Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        try:
            Psats = self._Psats
        except AttributeError:
            Psats = self.Psats()
        try:
            dPsats_dT = self._dPsats_dT
        except AttributeError:
            dPsats_dT = self.dPsats_dT()
        try:
            Vms_sat = self._Vms_sat
        except AttributeError:
            Vms_sat = self.Vms_sat()
        try:
            dVms_sat_dT = self._Vms_sat_dT
        except AttributeError:
            dVms_sat_dT = self.dVms_sat_dT()

        if self.P_DEPENDENT_H_LIQ:
            if self.use_IG_Cp:
                failed_dPsat_dT = False
                try:
                    for i in range(self.N):
                        dSi = Cpig_integrals_over_T_pure[i]
                        dVsat = R*T/Psats[i] - Vms_sat[i]
                        dSvap = dPsats_dT[i]*dVsat
        #                dSvap = Hvaps[i]/T # Confirmed - this line breaks everything - do not use
                        dSi -= dSvap
        #                dSi = Cpig_integrals_over_T_pure[i] - Hvaps[i]*T_inv # Do the transition at the temperature of the liquid
                        # Take each component to its reference state change - saturation pressure
        #                dSi -= R*log(P*P_REF_IG_INV)
                        dSi -= R*log(Psats[i]*P_REF_IG_INV)
        #                dSi -= R*log(P/101325.0)
                        # Only include the
                        dP = P - Psats[i]
    #                    dP = max(0.0, P - Psats[i])
        #                if dP > 0.0:
                        # I believe should include effect of pressure on all components, regardless of phase
                        dSi -= dP*dVms_sat_dT[i]
                        S += dSi*zs[i]
                except (ZeroDivisionError, ValueError):
                    # Handle the zero division on Psat or the log getting two small
                    failed_dPsat_dT = True

                if failed_dPsat_dT or isinf(S):
                    S = S_base
                    # Handle the case where vapor pressure reaches zero - needs special implementations
                    dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
                    lnPsats = self.lnPsats()
                    LOG_P_REF_IG = self.LOG_P_REF_IG
                    for i in cmps:
                        dSi = Cpig_integrals_over_T_pure[i]
                        dSvap = RT*dPsats_dT_over_Psats[i]
                        dSi -= dSvap
                        dSi -= R*(lnPsats[i] - LOG_P_REF_IG)#   trunc_log(Psats[i]*P_REF_IG_INV)
                        dSi -= P*dVms_sat_dT[i]
                        S += dSi*zs[i]

                if self.use_Tait:
                    pass
                elif self.use_Poynting:
                    pass
#                for i in cmps:

            else:
                # mine
                Hvaps_T_ref = self.Hvaps_T_ref()
                Psats_T_ref = self.Psats_T_ref()
                Cpl_integrals_over_T_pure = self._Cpl_integrals_over_T_pure()
                T_REF_IG_INV = self.T_REF_IG_INV
                dVms_sat_dT_T_ref = self.dVms_sat_dT_T_ref()
                Vms_sat_T_ref = self.Vms_sat_T_ref()

                for i in range(self.N):
                    dSi = Cpl_integrals_over_T_pure[i]
                    dSi -= Hvaps_T_ref[i]*T_REF_IG_INV
                    # Take each component to its reference state change - saturation pressure
                    dSi -= R*log(Psats_T_ref[i]*P_REF_IG_INV)
                    # I believe should include effect of pressure on all components, regardless of phase


                    dP = P - Psats_T_ref[i]
                    dSi -= dP*dVms_sat_dT_T_ref[i]
                    S += dSi*zs[i]
#                else:
#                    # COCO
#                    Hvaps = self.Hvaps()
#                    Psats_T_ref = self.Psats_T_ref()
#                    _Cpl_integrals_over_T_pure = self._Cpl_integrals_over_T_pure()
#                    T_REF_IG_INV = self.T_REF_IG_INV
#
#                    for i in range(self.N):
#                        dSi = -_Cpl_integrals_over_T_pure[i]
#                        dSi -= Hvaps[i]/T
#                        # Take each component to its reference state change - saturation pressure
#                        dSi -= R*log(Psats[i]*P_REF_IG_INV)
#
#                        dP = P - Psats[i]
#                        # I believe should include effect of pressure on all components, regardless of phase
#                        dSi -= dP*dVms_sat_dT[i]
#                        S += dSi*zs[i]
        else:
            Hvaps = self.Hvaps()
            for i in cmps:
                Sg298_to_T = Cpig_integrals_over_T_pure[i]
                Svap = -Hvaps[i]*T_inv # Do the transition at the temperature of the liquid
                S += zs[i]*(Sg298_to_T + Svap - R*log(P*P_REF_IG_INV)) #
#        self._S =
        S = S + self.GibbsExcessModel.SE()
        return S

    def S(self):
        r'''Method to calculate the entropy of the
        :obj:`GibbsExcessLiquid` phase. Depending on the settings of the phase, this can
        include the effects of activity coefficients
        :obj:`gammas <GibbsExcessLiquid.gammas>`, pressure correction terms
        :obj:`Poyntings <GibbsExcessLiquid.Poyntings>`, and pure component
        saturation fugacities :obj:`phis_sat <GibbsExcessLiquid.phis_sat>`
        as well as the pure component vapor pressures.

        When `caloric_basis` is 'Poynting&PhiSat':

        .. math::
            S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
            - \sum_i z_i\left[R\left(
            T \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
            + T\frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + T\frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}}
            + \ln(P_{\text{sat},i}) + \ln\left(\frac{\text{Poynting}\cdot\phi_{\text{sat},i}}{P}\right)
            \right) - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

        When `caloric_basis` is 'PhiSat':

        .. math::
            S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
            - \sum_i z_i\left[R\left(
            T \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
            + T\frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + \ln(P_{\text{sat},i}) + \ln\left(\frac{\phi_{\text{sat},i}}{P}\right)
            \right) - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

        When `caloric_basis` is 'Poynting':

        .. math::
            S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
            - \sum_i z_i\left[R\left(
            T\frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + T\frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}}
            + \ln(P_{\text{sat},i}) + \ln\left(\frac{\text{Poynting}}{P}\right)
            \right) - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

        When `caloric_basis` is 'Psat':

        .. math::
            S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
            - \sum_i z_i\left[R\left(
            T\frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + \ln(P_{\text{sat},i}) + \ln\left(\frac{1}{P}\right)
            \right) - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

        When `caloric_basis` is 'Hvap':

        .. math::
            S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
            - \sum_i z_i\left[R\left(\ln P_{\text{sat},i} + \ln\left(\frac{1}{P}\right)\right)
            + \frac{H_{vap,i}}{T}
            - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

        Returns
        -------
        S : float
            Entropy of the phase, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._S
        except AttributeError:
            pass
        T, P = self.T, self.P
        P_inv = 1.0/P
        zs, cmps = self.zs, range(self.N)

        log_zs = self.log_zs()
        S_comp = 0.0
        for i in cmps:
            S_comp -= zs[i]*log_zs[i]
        S = S_comp - log(P*self.P_REF_IG_INV)
        S *= R
        try:
            Cpig_integrals_over_T_pure = self._Cpig_integrals_over_T_pure
        except AttributeError:
            Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()

        try:
            lnPsats = self._lnPsats
        except AttributeError:
            lnPsats = self.lnPsats()

        use_Poynting, use_phis_sat, use_Hvap_caloric = self.use_Poynting, self.use_phis_sat, self.use_Hvap_caloric

        if use_Hvap_caloric:
            Hvaps = self.Hvaps()
            T_inv = 1.0/T
            logP_inv = log(P_inv)
            # Almost the same as no Poynting
            for i in cmps:
                S -= zs[i]*(R*(lnPsats[i] + logP_inv)
                            - Cpig_integrals_over_T_pure[i] + Hvaps[i]*T_inv)
        else:
            dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
            if use_Poynting:
                try:
                    Poyntings = self._Poyntings
                except AttributeError:
                    Poyntings = self.Poyntings()
                try:
                    dPoyntings_dT = self._dPoyntings_dT
                except AttributeError:
                    dPoyntings_dT = self.dPoyntings_dT()
            if use_phis_sat:
                try:
                    dphis_sat_dT = self._dphis_sat_dT
                except AttributeError:
                    dphis_sat_dT = self.dphis_sat_dT()
                try:
                    phis_sat = self._phis_sat
                except AttributeError:
                    phis_sat = self.phis_sat()

            if use_Poynting and use_phis_sat:
                for i in cmps:
                    S -= zs[i]*(R*(T*(dphis_sat_dT[i]/phis_sat[i] + dPsats_dT_over_Psats[i] + dPoyntings_dT[i]/Poyntings[i])
                                + lnPsats[i] + log(Poyntings[i]*phis_sat[i]*P_inv)) - Cpig_integrals_over_T_pure[i])
            elif use_Poynting:
                for i in cmps:
                    S -= zs[i]*(R*(T*(dPsats_dT_over_Psats[i] + dPoyntings_dT[i]/Poyntings[i])
                                + lnPsats[i] + log(Poyntings[i]*P_inv)) - Cpig_integrals_over_T_pure[i])
            elif use_phis_sat:
                for i in cmps:
                    S -= zs[i]*(R*(T*(dphis_sat_dT[i]/phis_sat[i] + dPsats_dT_over_Psats[i])
                                + lnPsats[i] + log(phis_sat[i]*P_inv)) - Cpig_integrals_over_T_pure[i])
            else:
                logP_inv = log(P_inv)
                for i in cmps:
                    S -= zs[i]*(R*(T*dPsats_dT_over_Psats[i] + lnPsats[i] + logP_inv)
                                - Cpig_integrals_over_T_pure[i])

        if not self.composition_independent:
            S += self.GibbsExcessModel.SE()
        self._S = S
        return S

    def Cp_old(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        # Needs testing
        T, P, P_DEPENDENT_H_LIQ = self.T, self.P, self.P_DEPENDENT_H_LIQ
        Cp, zs = 0.0, self.zs
        Cpigs_pure = self.Cpigs_pure()
        if P_DEPENDENT_H_LIQ:
            try:
                Psats = self._Psats
            except AttributeError:
                Psats = self.Psats()
            try:
                dPsats_dT = self._dPsats_dT
            except AttributeError:
                dPsats_dT = self.dPsats_dT()
            try:
                d2Psats_dT2 = self._d2Psats_dT2
            except AttributeError:
                d2Psats_dT2 = self.d2Psats_dT2()
            try:
                Vms_sat = self._Vms_sat
            except AttributeError:
                Vms_sat = self.Vms_sat()
            try:
                dVms_sat_dT = self._Vms_sat_dT
            except AttributeError:
                dVms_sat_dT = self.dVms_sat_dT()
            try:
                d2Vms_sat_dT2 = self._d2Vms_sat_dT2
            except AttributeError:
                d2Vms_sat_dT2 = self.d2Vms_sat_dT2()

            failed_dPsat_dT = False
            try:
                for i in range(self.N):
                    x0 = Psats[i]
                    Psat_inv = 1.0/x0
                    x1 = Vms_sat[i]
                    x2 = dPsats_dT[i]
                    x3 = R*Psat_inv
                    x4 = T*x3
                    x5 = -x1
                    x6 = dVms_sat_dT[i]
                    x7 = T*x2
    #                print(#-T*(P - x0)*d2Vms_sat_dT2[i],
    #                       - T*(x4 + x5)*d2Psats_dT2[i], T, x4, x5, d2Psats_dT2[i],
                           #x2*(x1 - x4) + x2*(T*x6 + x5) - x7*(-R*x7*Psat_inv*Psat_inv + x3 - x6),
                           #Cpigs_pure[i]
    #                       )
                    Cp += zs[i]*(-T*(P - x0)*d2Vms_sat_dT2[i] - T*(x4 + x5)*d2Psats_dT2[i]
                    + x2*(x1 - x4) + x2*(T*x6 + x5) - x7*(-R*x7*Psat_inv*Psat_inv + x3 - x6) + Cpigs_pure[i])
                    # The second derivative of volume is zero when extrapolating, which causes zero issues, discontinuous derivative
                '''
                from sympy import *
                T, P, R, zi = symbols('T, P, R, zi')
                Psat, Cpig_int, Vmsat = symbols('Psat, Cpig_int, Vmsat', cls=Function)
                dVmsatdT = diff(Vmsat(T), T)
                dPsatdT = diff(Psat(T), T)
                dV_vap = R*T/Psat(T) - Vmsat(T)
                dS_vap = dPsatdT*dV_vap
                Hvap = T*dS_vap
                H = zi*(Cpig_int(T) - Hvap)

                dP = P - Psat(T)
                H += zi*dP*(Vmsat(T) - T*dVmsatdT)

                (cse(diff(H, T), optimizations='basic'))
                '''
            except (ZeroDivisionError, ValueError):
                # Handle the zero division on Psat or the log getting two small
                failed_dPsat_dT = True

            if failed_dPsat_dT or isinf(Cp) or isnan(Cp):
                dlnPsats_dT = self.dlnPsats_dT()
                d2lnPsats_dT2 = self.d2lnPsats_dT2()
                Cp = 0.0
                for i in range(self.N):
                    Cp += zs[i]*(Cpigs_pure[i] - P*T*d2Vms_sat_dT2[i] - R*T*T*d2lnPsats_dT2[i]
                    - 2.0*R*T*dlnPsats_dT[i])
                    '''
                    from sympy import *
                    T, P, R, zi = symbols('T, P, R, zi')
                    lnPsat, Cpig_T_int, Vmsat = symbols('lnPsat, Cpig_T_int, Vmsat', cls=Function)
                    dVmsatdT = diff(Vmsat(T), T)
                    dPsatdT = diff(exp(lnPsat(T)), T)
                    dV_vap = R*T/exp(lnPsat(T)) - Vmsat(T)
                    dS_vap = dPsatdT*dV_vap
                    Hvap = T*dS_vap
                    H = zi*(Cpig_int(T) - Hvap)
                    dP = P
                    H += zi*dP*(Vmsat(T) - T*dVmsatdT)
                    print(simplify(expand(diff(H, T)).subs(exp(lnPsat(T)), 0)/zi))
                    '''
#                Cp += zs[i]*(Cpigs_pure[i] - dHvaps_dT[i])
#                Cp += zs[i]*(-T*(P - Psats[i])*d2Vms_sat_dT2[i] + (T*dVms_sat_dT[i] - Vms_sat[i])*dPsats_dT[i])

        else:
            dHvaps_dT = self.dHvaps_dT()
            for i in range(self.N):
                Cp += zs[i]*(Cpigs_pure[i] - dHvaps_dT[i])

        Cp += self.GibbsExcessModel.CpE()
        return Cp

    def Cp(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        T, P, zs, cmps = self.T, self.P, self.zs, range(self.N)
        Cpigs_pure = self.Cpigs_pure()
        use_Poynting, use_phis_sat = self.use_Poynting, self.use_phis_sat

        if use_Poynting:
            try:
                d2Poyntings_dT2 = self._d2Poyntings_dT2
            except AttributeError:
                d2Poyntings_dT2 = self.d2Poyntings_dT2()
            try:
                dPoyntings_dT = self._dPoyntings_dT
            except AttributeError:
                dPoyntings_dT = self.dPoyntings_dT()
            try:
                Poyntings = self._Poyntings
            except AttributeError:
                Poyntings = self.Poyntings()
        if use_phis_sat:
            try:
                d2phis_sat_dT2 = self._d2phis_sat_dT2
            except AttributeError:
                d2phis_sat_dT2 = self.d2phis_sat_dT2()
            try:
                dphis_sat_dT = self._dphis_sat_dT
            except AttributeError:
                dphis_sat_dT = self.dphis_sat_dT()
            try:
                phis_sat = self._phis_sat
            except AttributeError:
                phis_sat = self.phis_sat()

        dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
        d2Psats_dT2_over_Psats = self.d2Psats_dT2_over_Psats()

        RT = R*T
        RT2 = RT*T
        RT2_2 = RT + RT

        Cp = 0.0
        if use_Poynting and use_phis_sat:
            for i in cmps:
                Poy_inv = 1.0/Poyntings[i]
                phi_inv = 1.0/phis_sat[i]
                dPoy_ratio = dPoyntings_dT[i]*Poy_inv
                dphi_ratio = dphis_sat_dT[i]*phi_inv

                a = (d2phis_sat_dT2[i]*phi_inv - dphi_ratio*dphi_ratio
                     + d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i]
                     + d2Poyntings_dT2[i]*Poy_inv - dPoy_ratio*dPoy_ratio)

                b = dphi_ratio + dPsats_dT_over_Psats[i] + dPoy_ratio
                Cp -= zs[i]*(RT2*a + RT2_2*b - Cpigs_pure[i])
        elif use_Poynting:
            for i in cmps:
                Poy_inv = 1.0/Poyntings[i]
                dPoy_ratio = dPoyntings_dT[i]*Poy_inv
                a = (d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i]
                     + d2Poyntings_dT2[i]*Poy_inv - dPoy_ratio*dPoy_ratio)
                b = dPsats_dT_over_Psats[i] + dPoy_ratio
                Cp -= zs[i]*(RT2*a + RT2_2*b - Cpigs_pure[i])
        elif use_phis_sat:
            for i in cmps:
                phi_inv = 1.0/phis_sat[i]
                dphi_ratio = dphis_sat_dT[i]*phi_inv
                a = (d2phis_sat_dT2[i]*phi_inv - dphi_ratio*dphi_ratio
                     + d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i])
                b = dphi_ratio + dPsats_dT_over_Psats[i]
                Cp -= zs[i]*(RT2*a + RT2_2*b - Cpigs_pure[i])
        else:
            for i in cmps:
                a = (d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i])
                b = dPsats_dT_over_Psats[i]
                Cp -= zs[i]*(RT2*a + RT2_2*b - Cpigs_pure[i])
        if not self.composition_independent:
            Cp += self.GibbsExcessModel.CpE()
        self._Cp = Cp
        return Cp

    dH_dT = Cp

    def dS_dT_old(self):
        # Needs testing
        T, P, P_DEPENDENT_H_LIQ = self.T, self.P, self.P_DEPENDENT_H_LIQ
        RT = R*T
        Cp, zs = 0.0, self.zs
        Cpigs_pure = self.Cpigs_pure()
        dS_dT = 0.0
        T_inv = 1.0/T
        if P_DEPENDENT_H_LIQ:
            d2Vms_sat_dT2 = self.d2Vms_sat_dT2()
            dVms_sat_dT = self.dVms_sat_dT()
            Vms_sat = self.Vms_sat()
            Psats = self.Psats()
            dPsats_dT = self.dPsats_dT()
            d2Psats_dT2 = self.d2Psats_dT2()
            failed_dPsat_dT = False
            for Psat in Psats:
                if Psat < 1e-40:
                    failed_dPsat_dT = True
            if not failed_dPsat_dT:
                try:
                    '''
                    from sympy import *
                    T, P, R, zi, P_REF_IG = symbols('T, P, R, zi, P_REF_IG')

                    Psat, Cpig_T_int, Vmsat = symbols('Psat, Cpig_T_int, Vmsat', cls=Function)
                    dVmsatdT = diff(Vmsat(T), T)
                    dPsatdT = diff(Psat(T), T)

                    S = 0
                    dSi = Cpig_T_int(T)
                    dVsat = R*T/Psat(T) - Vmsat(T)
                    dSvap = dPsatdT*dVsat
                    dSi -= dSvap
                    dSi -= R*log(Psat(T)/P_REF_IG)
                    dP = P - Psat(T)
                    dSi -= dP*dVmsatdT
                    S += dSi*zi
                    # cse(diff(S, T), optimizations='basic')
                    '''
                    for i in range(self.N):
                        x0 = Psats[i]
                        x1 = dPsats_dT[i]
                        x2 = R/x0
                        x3 = Vms_sat[i]
                        x4 = dVms_sat_dT[i]
                        dS_dT -= zs[i]*(x1*x2 - x1*x4 - x1*(RT*x1/x0**2 - x2 + x4) + (P - x0)*d2Vms_sat_dT2[i]
                        + (T*x2 - x3)*d2Psats_dT2[i] - Cpigs_pure[i]*T_inv)
                except (ZeroDivisionError, ValueError):
                    # Handle the zero division on Psat or the log getting two small
                    failed_dPsat_dT = True

            if failed_dPsat_dT:
                lnPsats = self.lnPsats()
                dlnPsats_dT = self.dlnPsats_dT()
                d2lnPsats_dT2 = self.d2lnPsats_dT2()
#                P*Derivative(Vmsat(T), (T, 2))
#                R*T*Derivative(lnPsat(T), (T, 2))
#                 2*R*Derivative(lnPsat(T), T) + Derivative(Cpig_T_int(T), T)
                '''
                from sympy import *
                T, P, R, zi, P_REF_IG = symbols('T, P, R, zi, P_REF_IG')

                lnPsat, Cpig_T_int, Vmsat = symbols('lnPsat, Cpig_T_int, Vmsat', cls=Function)
                # Psat, Cpig_T_int, Vmsat = symbols('Psat, Cpig_T_int, Vmsat', cls=Function)
                dVmsatdT = diff(Vmsat(T), T)
                dPsatdT = diff(exp(lnPsat(T)), T)

                S = 0
                dSi = Cpig_T_int(T)
                dVsat = R*T/exp(lnPsat(T)) - Vmsat(T)
                dSvap = dPsatdT*dVsat
                dSi -= dSvap
                # dSi -= R*log(Psat(T)/P_REF_IG)
                dSi -= R*(lnPsat(T) - log(P_REF_IG))
                dP = P - exp(lnPsat(T))
                dSi -= dP*dVmsatdT
                S += dSi*zi
                # cse(diff(S, T), optimizations='basic')
                print(simplify(expand(diff(S, T)).subs(exp(lnPsat(T)), 0)/zi))


                '''
                dS_dT = 0.0
                for i in range(self.N):
                    dS_dT -= zs[i]*(P*d2Vms_sat_dT2[i] + RT*d2lnPsats_dT2[i]
                    + 2.0*R*dlnPsats_dT[i]- Cpigs_pure[i]*T_inv)

        dS_dT += self.GibbsExcessModel.dSE_dT()
        return dS_dT

    def dS_dT(self):
        try:
            return self._dS_dT
        except AttributeError:
            pass
        T, P, zs, cmps = self.T, self.P, self.zs, range(self.N)
        use_Poynting, use_phis_sat = self.use_Poynting, self.use_phis_sat

        if use_Poynting:
            try:
                d2Poyntings_dT2 = self._d2Poyntings_dT2
            except AttributeError:
                d2Poyntings_dT2 = self.d2Poyntings_dT2()
            try:
                dPoyntings_dT = self._dPoyntings_dT
            except AttributeError:
                dPoyntings_dT = self.dPoyntings_dT()
            try:
                Poyntings = self._Poyntings
            except AttributeError:
                Poyntings = self.Poyntings()
        if use_phis_sat:
            try:
                d2phis_sat_dT2 = self._d2phis_sat_dT2
            except AttributeError:
                d2phis_sat_dT2 = self.d2phis_sat_dT2()
            try:
                dphis_sat_dT = self._dphis_sat_dT
            except AttributeError:
                dphis_sat_dT = self.dphis_sat_dT()
            try:
                phis_sat = self._phis_sat
            except AttributeError:
                phis_sat = self.phis_sat()

        dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
        d2Psats_dT2_over_Psats = self.d2Psats_dT2_over_Psats()
        Cpigs_pure = self.Cpigs_pure()

        T_inv = 1.0/T
        RT = R*T
        R_2 = R + R

        dS_dT = 0.0
        if use_Poynting and use_phis_sat:
            for i in cmps:
                Poy_inv = 1.0/Poyntings[i]
                phi_inv = 1.0/phis_sat[i]
                dPoy_ratio = dPoyntings_dT[i]*Poy_inv
                dphi_ratio = dphis_sat_dT[i]*phi_inv

                a = (d2phis_sat_dT2[i]*phi_inv - dphi_ratio*dphi_ratio
                     + d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i]
                     + d2Poyntings_dT2[i]*Poy_inv - dPoy_ratio*dPoy_ratio)

                b = dphi_ratio + dPsats_dT_over_Psats[i] + dPoy_ratio

                dS_dT -= zs[i]*((RT*a + b*R_2) - Cpigs_pure[i]*T_inv)
        elif use_Poynting:
            for i in cmps:
                Poy_inv = 1.0/Poyntings[i]
                dPoy_ratio = dPoyntings_dT[i]*Poy_inv
                a = (d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i]
                     + d2Poyntings_dT2[i]*Poy_inv - dPoy_ratio*dPoy_ratio)
                b = dPsats_dT_over_Psats[i] + dPoy_ratio
                dS_dT -= zs[i]*((RT*a + b*R_2) - Cpigs_pure[i]*T_inv)
        elif use_phis_sat:
            for i in cmps:
                phi_inv = 1.0/phis_sat[i]
                dphi_ratio = dphis_sat_dT[i]*phi_inv
                a = (d2phis_sat_dT2[i]*phi_inv - dphi_ratio*dphi_ratio
                     + d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i])
                b = dphi_ratio + dPsats_dT_over_Psats[i]
                dS_dT -= zs[i]*((RT*a + b*R_2) - Cpigs_pure[i]*T_inv)
        else:
            for i in cmps:
                a = (d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i])
                b = dPsats_dT_over_Psats[i]
                dS_dT -= zs[i]*((RT*a + b*R_2) - Cpigs_pure[i]*T_inv)
        if not self.composition_independent:
            dS_dT += self.GibbsExcessModel.dSE_dT()
        self._dS_dT = dS_dT
        return dS_dT

    def dH_dP(self):
        try:
            return self._dH_dP
        except AttributeError:
            pass
        T = self.T
        P = self.P
        zs = self.zs
        dH_dP = 0.0
        if self.use_Poynting:
            nRT2 = -R*T*T
            Poyntings = self.Poyntings()
            dPoyntings_dP = self.dPoyntings_dP()
            dPoyntings_dT = self.dPoyntings_dT()
            d2Poyntings_dPdT = self.d2Poyntings_dPdT()
            for i in range(self.N):
                Poy_inv = 1.0/Poyntings[i]
                dH_dP += nRT2*zs[i]*Poy_inv*(d2Poyntings_dPdT[i] - dPoyntings_dP[i]*dPoyntings_dT[i]*Poy_inv)

#        if self.P_DEPENDENT_H_LIQ:
#            if self.use_IG_Cp:
#                Vms_sat = self.Vms_sat()
#                dVms_sat_dT = self.dVms_sat_dT()
#                Psats = self.Psats()
#                for i in range(self.N):
#                    if P > Psats[i]:
#                        dH_dP += zs[i]*(-T*dVms_sat_dT[i] + Vms_sat[i])
        self._dH_dP = dH_dP
        return dH_dP


    def dS_dP(self):
        try:
            return self._dS_dP
        except AttributeError:
            pass
        T = self.T
        P = self.P
        P_inv = 1.0/P
        zs = self.zs
        if self.use_Poynting:
            dS_dP = -R*P_inv
            Poyntings = self.Poyntings()
            dPoyntings_dP = self.dPoyntings_dP()
            dPoyntings_dT = self.dPoyntings_dT()
            d2Poyntings_dPdT = self.d2Poyntings_dPdT()
            for i in range(self.N):
                Poy_inv = 1.0/Poyntings[i]
                dS_dP -= zs[i]*R*Poy_inv*(dPoyntings_dP[i] - Poyntings[i]*P_inv
                        +T*(d2Poyntings_dPdT[i] - dPoyntings_dP[i]*dPoyntings_dT[i]*Poy_inv))
        else:
            dS_dP = 0.0
#        if self.P_DEPENDENT_H_LIQ:
#            if self.use_IG_Cp:
#                dVms_sat_dT = self.dVms_sat_dT()
#                Psats = self.Psats()
#                for i in range(self.N):
#                    if P > Psats[i]:
#                        dS_dP -= zs[i]*(dVms_sat_dT[i])
        self._dS_dP = dS_dP
        return dS_dP

    def H_dep(self):
        return self.H() - self.H_ideal_gas()

    def S_dep(self):
        return self.S() - self.S_ideal_gas()

    def Cp_dep(self):
        return self.Cp() - self.Cp_ideal_gas()

    ### Volumetric properties
    def V(self):
        try:
            return self._V
        except AttributeError:
            pass
        zs = self.zs
        Vms = self.Vms()
        '''To make a fugacity-volume identity consistent, cannot use pressure
        correction unless the Poynting factor is calculated with quadrature/
        integration.
        '''
        V = 0.0
        for i in range(self.N):
            V += zs[i]*Vms[i]
        self._V = V
        return V

    def dV_dT(self):
        try:
            return self._dV_dT
        except AttributeError:
            pass
        zs = self.zs
        dVms_sat_dT = self.dVms_sat_dT()
        dV_dT = 0.0
        for i in range(self.N):
            dV_dT += zs[i]*dVms_sat_dT[i]
        self._dV_dT = dV_dT
        return dV_dT

    def d2V_dT2(self):
        try:
            return self._d2V_dT2
        except AttributeError:
            pass
        zs = self.zs
        d2Vms_sat_dT2 = self.d2Vms_sat_dT2()
        d2V_dT2 = 0.0
        for i in range(self.N):
            d2V_dT2 += zs[i]*d2Vms_sat_dT2[i]
        self._d2V_dT2 = d2V_dT2
        return d2V_dT2

    # Main needed volume derivatives
    def dP_dV(self):
        try:
            return self._dP_dV
        except AttributeError:
            pass
        if self.incompressible:
            self._dP_dV = INCOMPRESSIBLE_CONST #1.0/self.VolumeLiquidMixture.property_derivative_P(self.T, self.P, self.zs, order=1)

        return self._dP_dV

    def d2P_dV2(self):
        try:
            return self._d2P_dV2
        except AttributeError:
            pass
        if self.incompressible:
            self._d2P_dV2 = INCOMPRESSIBLE_CONST#self.d2V_dP2()/-(self.dP_dV())**-3
        return self._d2P_dV2

    def dP_dT(self):
        try:
            return self._dP_dT
        except AttributeError:
            pass
        self._dP_dT = self.dV_dT()/-self.dP_dV()
        return self._dP_dT

    def d2P_dTdV(self):
        try:
            return self._d2P_dTdV
        except AttributeError:
            pass
        if self.incompressible:
            self._d2P_dTdV = 0.0
        else:
            P = self.P
            def dP_dV_for_diff(T):
                return 1.0/self.VolumeLiquidMixture.property_derivative_P(T, P, self.zs, order=1)

            self._d2P_dTdV = derivative(dP_dV_for_diff, self.T)
        return self._d2P_dTdV

    def d2P_dT2(self):
        try:
            return self._d2P_dT2
        except AttributeError:
            pass
        if self.incompressible:
            self._d2P_dT2 = -self.d2V_dT2()/INCOMPRESSIBLE_CONST
        else:
            P, zs = self.P, self.zs
            def dP_dT_for_diff(T):
                dV_dT = self.VolumeLiquidMixture.property_derivative_T(T, P, zs, order=1)
                dP_dV = 1.0/self.VolumeLiquidMixture.property_derivative_P(T, P, zs, order=1)
                dP_dT = dV_dT/-dP_dV
                return dP_dT

            self._d2P_dT2 = derivative(dP_dT_for_diff, self.T)
        return self._d2P_dT2

    # Volume derivatives which needed to be implemented for the main ones
    def d2V_dP2(self):
        try:
            return self._d2V_dP2
        except AttributeError:
            pass
        if self.incompressible:
            self._d2V_dP2 = 0.0
        return self._d2V_dP2

    def Tait_Bs(self):
        try:
            return self._Tait_Bs
        except:
            pass

        self._Tait_Bs = evaluate_linear_fits(self._Tait_B_data, self.T)
        return self._Tait_Bs

    def dTait_B_dTs(self):
        try:
            return self._dTait_B_dTs
        except:
            pass

        self._dTait_B_dTs = evaluate_linear_fits_d(self._Tait_B_data, self.T)
        return self._dTait_B_dTs

    def d2Tait_B_dT2s(self):
        try:
            return self._d2Tait_B_dT2s
        except:
            pass

        self._d2Tait_B_dT2s = evaluate_linear_fits_d2(self._Tait_B_data, self.T)
        return self._d2Tait_B_dT2s

    def Tait_Cs(self):
        try:
            return self._Tait_Cs
        except:
            pass

        self._Tait_Cs = evaluate_linear_fits(self._Tait_C_data, self.T)
        return self._Tait_Cs

    def dTait_C_dTs(self):
        try:
            return self._dTait_C_dTs
        except:
            pass

        self._dTait_C_dTs = evaluate_linear_fits_d(self._Tait_C_data, self.T)
        return self._dTait_C_dTs

    def d2Tait_C_dT2s(self):
        try:
            return self._d2Tait_C_dT2s
        except:
            pass

        self._d2Tait_C_dT2s = evaluate_linear_fits_d2(self._Tait_C_data, self.T)
        return self._d2Tait_C_dT2s

    def Tait_Vs(self):
        Vms_sat = self.Vms_sat()
        Psats = self.Psats()
        Tait_Bs = self.Tait_Bs()
        Tait_Cs = self.Tait_Cs()
        P = self.P
        return [Vms_sat[i]*(1.0  - Tait_Cs[i]*log((Tait_Bs[i] + P)/(Tait_Bs[i] + Psats[i]) ))
                for i in range(self.N)]


    def dH_dP_integrals_Tait(self):
        try:
            return self._dH_dP_integrals_Tait
        except AttributeError:
            pass
        Psats = self.Psats()
        Vms_sat = self.Vms_sat()
        dVms_sat_dT = self.dVms_sat_dT()
        dPsats_dT = self.dPsats_dT()

        Tait_Bs = self.Tait_Bs()
        Tait_Cs = self.Tait_Cs()
        dTait_C_dTs = self.dTait_C_dTs()
        dTait_B_dTs = self.dTait_B_dTs()
        T, P, zs = self.T, self.P, self.zs


        self._dH_dP_integrals_Tait = dH_dP_integrals_Tait = []

#        def to_int(P, i):
#            l = self.to_TP_zs(T, P, zs)
##            def to_diff(T):
##                return self.to_TP_zs(T, P, zs).Tait_Vs()[i]
##            dV_dT = derivative(to_diff, T, dx=1e-5*T, order=11)
#
#            x0 = l.Vms_sat()[i]
#            x1 = l.Tait_Cs()[i]
#            x2 = l.Tait_Bs()[i]
#            x3 = P + x2
#            x4 = l.Psats()[i]
#            x5 = x3/(x2 + x4)
#            x6 = log(x5)
#            x7 = l.dTait_B_dTs()[i]
#            dV_dT = (-x0*(x1*(-x5*(x7 +l.dPsats_dT()[i]) + x7)/x3
#                                   + x6*l.dTait_C_dTs()[i])
#                        - (x1*x6 - 1.0)*l.dVms_sat_dT()[i])
#
##            print(dV_dT, dV_dT2, dV_dT/dV_dT2, T, P)
#
#            V = l.Tait_Vs()[i]
#            return V - T*dV_dT
#        from scipy.integrate import quad
#        _dH_dP_integrals_Tait = [quad(to_int, Psats[i], P, args=i)[0]
#                                      for i in range(self.N)]
##        return self._dH_dP_integrals_Tait
#        print(_dH_dP_integrals_Tait)
#        self._dH_dP_integrals_Tait2 = _dH_dP_integrals_Tait
#        return self._dH_dP_integrals_Tait2

#        dH_dP_integrals_Tait = []
        for i in range(self.N):
            # Very wrong according to numerical integration. Is it an issue with
            # the translation to code, one of the derivatives, what was integrated,
            # or sympy's integration?
            x0 = Tait_Bs[i]
            x1 = P + x0
            x2 = Psats[i]
            x3 = x0 + x2
            x4 = 1.0/x3
            x5 = Tait_Cs[i]
            x6 = Vms_sat[i]
            x7 = x5*x6
            x8 = T*dVms_sat_dT[i]
            x9 = x5*x8
            x10 = T*dTait_C_dTs[i]
            x11 = x0*x6
            x12 = T*x7
            x13 = -x0*x7 + x0*x9 + x10*x11 + x12*dTait_B_dTs[i]
            x14 = x2*x6
            x15 = x4*(x0*x8 + x10*x14 - x11 + x12*dPsats_dT[i] + x13 - x14 - x2*x7 + x2*x8 + x2*x9)
            val = -P*x15 + P*(x10*x6 - x7 + x9)*log(x1*x4) + x13*log(x1) - x13*log(x3) + x15*x2
            dH_dP_integrals_Tait.append(val)
#        print(dH_dP_integrals_Tait, self._dH_dP_integrals_Tait2)
        return dH_dP_integrals_Tait

    def mu(self):
        try:
            return self._mu
        except AttributeError:
            pass
        mu = self._mu = self.correlations.ViscosityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        return mu

    def k(self):
        try:
            return self._k
        except AttributeError:
            pass
        self._k = k = self.correlations.ThermalConductivityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        return k


class GibbsExcessSolid(GibbsExcessLiquid):
    ideal_gas_basis = True
    force_phase = 's'
    phase = 's'
    is_gas = False
    is_liquid = False
    is_solid = True
    __full_path__ = "%s.%s" %(__module__, __qualname__)


    pure_references = ('HeatCapacityGases','SublimationPressures', 'VolumeSolids', 'EnthalpySublimations')
    pure_reference_types = (HeatCapacityGas, SublimationPressure, VolumeSolid, EnthalpySublimation)


    model_attributes = ('Hfs', 'Gfs', 'Sfs','GibbsExcessModel',
                        'eos_pure_instances', 'use_Poynting', 'use_phis_sat',
                        'use_eos_volume', 'henry_components',
                        'henry_data', 'Psat_extrpolation') + pure_references

    def __init__(self, SublimationPressures, VolumeSolids=None,
                 GibbsExcessModel=IdealSolution(),
                 eos_pure_instances=None,
                 VolumeLiquidMixture=None,
                 HeatCapacityGases=None,
                 EnthalpySublimations=None,
                 use_Poynting=False,
                 use_phis_sat=False,
                 Hfs=None, Gfs=None, Sfs=None,
                 henry_components=None, henry_data=None,
                 T=None, P=None, zs=None,
                 ):
        super(GibbsExcessSolid, self).__init__(VaporPressures=SublimationPressures, VolumeLiquids=VolumeSolids,
              HeatCapacityGases=HeatCapacityGases, EnthalpyVaporizations=EnthalpySublimations,
              use_Poynting=use_Poynting,
              Hfs=Hfs, Gfs=Gfs, Sfs=Sfs, T=T, P=P, zs=zs)

# hydrogen, methane
Grayson_Streed_special_CASs = set(['1333-74-0', '74-82-8'])

class GraysonStreed(Phase):
    phase = force_phase = 'l'
    is_gas = False
    is_liquid = True
    # revised one
    __full_path__ = "%s.%s" %(__module__, __qualname__)

    hydrogen_coeffs = (1.50709, 2.74283, -0.0211, 0.00011, 0.0, 0.008585, 0.0, 0.0, 0.0, 0.0)
    methane_coeffs = (1.36822, -1.54831, 0.0, 0.02889, -0.01076, 0.10486, -0.02529, 0.0, 0.0, 0.0)
    simple_coeffs = (2.05135, -2.10889, 0.0, -0.19396, 0.02282, 0.08852, 0.0, -0.00872, -0.00353, 0.00203)
    version = 1

    pure_references = tuple()
    model_attributes = ('Tcs', 'Pcs', 'omegas', '_CASs',
                        'GibbsExcessModel') + pure_references

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs

        new._Tcs = self._Tcs
        new._Pcs = self._Pcs
        new._omegas = self._omegas
        new._CASs = self._CASs
        new.regular = self.regular
        new.GibbsExcessModel = self.GibbsExcessModel.to_T_xs(T, zs)
        new.version = self.version

        try:
            new.N = self.N
        except:
            pass

        return new

    def to(self, zs, T=None, P=None, V=None):
        if T is not None:
            if P is not None:
                return self.to_TP_zs(T=T, P=P, zs=zs)
            elif V is not None:
                raise ValueError("Model does not implement volume")
        elif P is not None and V is not None:
            raise ValueError("Model does not implement volume")
        else:
            raise ValueError("Two of T, P, or V are needed")

    def __init__(self, Tcs, Pcs, omegas, CASs,
                 GibbsExcessModel=IdealSolution(),
                 T=None, P=None, zs=None,
                 ):

        self.T = T
        self.P = P
        self.zs = zs

        self.N = len(zs)
        self._Tcs = Tcs
        self._Pcs = Pcs
        self._omegas = omegas
        self._CASs = CASs
        self.regular = [i not in Grayson_Streed_special_CASs for i in CASs]

        self.GibbsExcessModel = GibbsExcessModel

    def gammas(self):
        try:
            return self.GibbsExcessModel._gammas
        except AttributeError:
            return self.GibbsExcessModel.gammas()

    def phis(self):
        try:
            return self._phis
        except AttributeError:
            pass
        try:
            gammas = self._gammas
        except AttributeError:
            gammas = self.gammas()
        fugacity_coeffs_pure = self.nus()

        self._phis = [gammas[i]*fugacity_coeffs_pure[i]
                for i in range(self.N)]
        return self._phis


    def lnphis(self):
        try:
            return self._lnphis
        except AttributeError:
            pass
        self._lnphis = [log(i) for i in self.phis()]
        return self._lnphis

    lnphis_G_min = lnphis

    def nus(self):
        T, P = self.T, self.P
        Tcs, Pcs, omegas = self._Tcs, self._Pcs, self._omegas
        regular, CASs = self.regular, self._CASs
        nus = []
        limit_Tr = self.version > 0

        for i in range(self.N):
            # TODO validate and take T, P derivatives; n derivatives are from regular solution only
            Tr = T/Tcs[i]
            Pr = P/Pcs[i]

            if regular[i]:
                coeffs = self.simple_coeffs
            elif CASs[i] == '1333-74-0':
                coeffs = self.hydrogen_coeffs
            elif CASs[i] == '74-82-8':
                coeffs = self.methane_coeffs
            else:
                raise ValueError("Fail")
            A0, A1, A2, A3, A4, A5, A6, A7, A8, A9 = coeffs

            log10_v0 = A0 + A1/Tr + A2*Tr + A3*Tr**2 + A4*Tr**3 + (A5 + A6*Tr + A7*Tr**2)*Pr + (A8 + A9*Tr)*Pr**2 - log10(Pr)
            log10_v1 = -4.23893 + 8.65808*Tr - 1.2206/Tr - 3.15224*Tr**3 - 0.025*(Pr - 0.6)
            if Tr > 1.0 and limit_Tr:
                log10_v1 = 1.0

            if regular[i]:
                v = 10.0**(log10_v0 + omegas[i]*log10_v1)
            else:
                # Chao Seader mentions
                v = 10.0**(log10_v0)
            nus.append(v)
        return nus

class ChaoSeader(GraysonStreed):
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    # original one
    hydrogen_coeffs = (1.96718, 1.02972, -0.054009, 0.0005288, 0.0, 0.008585, 0.0, 0.0, 0.0, 0.0)
    methane_coeffs = (2.4384, -2.2455, -0.34084, 0.00212, -0.00223, 0.10486, -0.03691, 0.0, 0.0, 0.0)
    simple_coeffs = (5.75748, -3.01761, -4.985, 2.02299, 0.0, 0.08427, 0.26667, -0.31138, -0.02655, 0.02883)
    version = 0

from chemicals.virial import BVirial_Pitzer_Curl, Z_from_virial_density_form
class VirialCorrelationsPitzerCurl(object):

    def __init__(self, Tcs, Pcs, omegas):
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.N = len(Tcs)

    def C_pures(self, T):
        return [0.0]*self.N

    def dC_dT_pures(self, T):
        return [0.0]*self.N

    def d2C_dT2_pures(self, T):
        return [0.0]*self.N

    def C_interactions(self, T):
        N = self.N
        Ciij = [[0.0]*N for i in range(N)]
        Cijj = [[0.0]*N for i in range(N)]

#        Full return should be (Ciij, Ciji, Cjii), (Cijj, Cjij, Cjji)
#        but due to symmetry there is only those two matrices
        return Ciij, Cijj

    def dC_dT_interactions(self, T):
        N = self.N
        Ciij = [[0.0]*N for i in range(N)]
        Cijj = [[0.0]*N for i in range(N)]
        return Ciij, Cijj

    def d2C_dT2_interactions(self, T):
        N = self.N
        Ciij = [[0.0]*N for i in range(N)]
        Cijj = [[0.0]*N for i in range(N)]
        return Ciij, Cijj

    def B_pures(self, T):
        Tcs, Pcs, omegas = self.Tcs, self.Pcs, self.omegas
        return [BVirial_Pitzer_Curl(T, Tcs[i], Pcs[i], omegas[i]) for i in range(self.N)]

    def dB_dT_pures(self, T):
        Tcs, Pcs, omegas = self.Tcs, self.Pcs, self.omegas
        return [BVirial_Pitzer_Curl(T, Tcs[i], Pcs[i], omegas[i], 1) for i in range(self.N)]

    def B_interactions(self, T):
        N = self.N
        return [[0.0]*N for i in range(N)]

    def dB_dT_interactions(self, T):
        N = self.N
        return [[0.0]*N for i in range(N)]

    def B_matrix(self, T):
        N = self.N
        B_mat = [[0.0]*N for i in range(N)]
        pures = self.B_pures(T)
        B_interactions = self.B_interactions(T)
        for i in range(N):
            B_mat[i][i] = pures[i]
        for i in range(N):
            for j in range(i):
                B_mat[i][j] = B_interactions[i][j]
                B_mat[j][i] = B_interactions[j][i]

        return B_mat

    def dB_dT_matrix(self, T):
        N = self.N
        B_mat = [[0.0]*N for i in range(N)]
        pures = self.dB_dT_pures(T)
        B_interactions = self.dB_dT_interactions(T)
        for i in range(N):
            B_mat[i][i] = pures[i]
        for i in range(N):
            for j in range(i):
                B_mat[i][j] = B_interactions[i][j]
                B_mat[j][i] = B_interactions[j][i]

        return B_mat


    def d2B_dT2_pures(self, T):
        Tcs, Pcs, omegas = self.Tcs, self.Pcs, self.omegas
        return [BVirial_Pitzer_Curl(T, Tcs[i], Pcs[i], omegas[i], 2) for i in range(self.N)]
    def d2B_dT2_interactions(self, T):
        N = self.N
        return [[0.0]*N for i in range(N)]

    def d2B_dT2_matrix(self, T):
        N = self.N
        B_mat = [[0.0]*N for i in range(N)]
        pures = self.d2B_dT2_pures(T)
        B_interactions = self.d2B_dT2_interactions(T)
        for i in range(N):
            B_mat[i][i] = pures[i]
        for i in range(N):
            for j in range(i):
                B_mat[i][j] = B_interactions[i][j]
                B_mat[j][i] = B_interactions[j][i]

        return B_mat



class VirialGas(Phase):
    phase = 'g'
    force_phase = 'g'
    is_gas = True
    is_liquid = False
    ideal_gas_basis = True
    __full_path__ = "%s.%s" %(__module__, __qualname__)

    pure_references = ('HeatCapacityGases',)
    pure_reference_types = (HeatCapacityGas, )

    def __init__(self, model, HeatCapacityGases=None, Hfs=None, Gfs=None, T=None, P=None, zs=None,
                 ):
        self.model = model
        self.HeatCapacityGases = HeatCapacityGases
        self.Hfs = Hfs
        self.Gfs = Gfs
        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            self.Sfs = [(Hfi - Gfi)/298.15 for Hfi, Gfi in zip(Hfs, Gfs)]
        else:
            self.Sfs = None

        if zs is not None:
            self.N = N = len(zs)
        elif HeatCapacityGases is not None:
            self.N = N = len(HeatCapacityGases)
        if zs is not None:
            self.zs = zs
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P
        if T is not None and P is not None and zs is not None:
            Z = Z_from_virial_density_form(T, P, self.B(), self.C())
            self._V = Z*R*T/P

    def V(self):
        return self._V

    def dP_dT(self):
        r'''

        .. math::
            \left(\frac{\partial P}{\partial T}\right)_{V} = \frac{R \left(T
            \left(V \frac{d}{d T} B{\left(T \right)} + \frac{d}{d T} C{\left(T
            \right)}\right) + V^{2} + V B{\left(T \right)} + C{\left(T \right)}
            \right)}{V^{3}}

        '''
        try:
            return self._dP_dT
        except:
            pass
        T, V = self.T, self._V
        self._dP_dT = dP_dT = R*(T*(V*self.dB_dT() + self.dC_dT()) + V*(V + self.B()) + self.C())/(V*V*V)
        return dP_dT

    def dP_dV(self):
        r'''

        .. math::
            \left(\frac{\partial P}{\partial V}\right)_{T} =
            - \frac{R T \left(V^{2} + 2 V B{\left(T \right)} + 3 C{\left(T
            \right)}\right)}{V^{4}}

        '''
        try:
            return self._dP_dV
        except:
            pass
        T, V = self.T, self._V
        self._dP_dV = dP_dV = -R*T*(V*V + 2.0*V*self.B() + 3.0*self.C())/(V*V*V*V)
        return dP_dV

    def d2P_dTdV(self):
        r'''

        .. math::
            \left(\frac{\partial^2 P}{\partial V\partial T}\right)_{T} =
            - \frac{R \left(2 T V \frac{d}{d T} B{\left(T \right)} + 3 T
            \frac{d}{d T} C{\left(T \right)} + V^{2} + 2 V B{\left(T \right)}
            + 3 C{\left(T \right)}\right)}{V^{4}}

        '''
        try:
            return self._d2P_dTdV
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V
        self._d2P_dTdV = d2P_dTdV = -R*(2.0*T*V*self.dB_dT() + 3.0*T*self.dC_dT()
        + V2 + 2.0*V*self.B() + 3.0*self.C())/(V2*V2)

        return d2P_dTdV

    def d2P_dV2(self):
        r'''

        .. math::
            \left(\frac{\partial^2 P}{\partial V^2}\right)_{T} =
            \frac{2 R T \left(V^{2} + 3 V B{\left(T \right)}
            + 6 C{\left(T \right)}\right)}{V^{5}}

        '''
        try:
            return self._d2P_dV2
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V
        self._d2P_dV2 = d2P_dV2 = 2.0*R*T*(V2 + 3.0*V*self.B() + 6.0*self.C())/(V2*V2*V)
        return d2P_dV2

    def d2P_dT2(self):
        r'''

        .. math::
            \left(\frac{\partial^2 P}{\partial T^2}\right)_{V} =
            \frac{R \left(T \left(V \frac{d^{2}}{d T^{2}} B{\left(T \right)}
            + \frac{d^{2}}{d T^{2}} C{\left(T \right)}\right) + 2 V \frac{d}{d T}
            B{\left(T \right)} + 2 \frac{d}{d T} C{\left(T \right)}\right)}{V^{3}}

        '''
        try:
            return self._d2P_dT2
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V
        self._d2P_dT2 = d2P_dT2 = R*(T*(V*self.d2B_dT2() + self.d2C_dT2())
                              + 2.0*V*self.dB_dT() + 2.0*self.dC_dT())/(V*V*V)
        return d2P_dT2

    def H_dep(self):
        r'''

        .. math::
           H_{dep} = \frac{R T^{2} \left(2 V \frac{d}{d T} B{\left(T \right)}
           + \frac{d}{d T} C{\left(T \right)}\right)}{2 V^{2}} - R T \left(-1
            + \frac{V^{2} + V B{\left(T \right)} + C{\left(T \right)}}{V^{2}}
            \right)

        '''
        '''
        from sympy import *
        Z, R, T, V, P = symbols('Z, R, T, V, P')
        B, C = symbols('B, C', cls=Function)
        base =Eq(P*V/(R*T), 1 + B(T)/V + C(T)/V**2)
        P_sln = solve(base, P)[0]
        Z = P_sln*V/(R*T)

        # Two ways to compute H_dep
        Hdep2 = R*T - P_sln*V + integrate(P_sln - T*diff(P_sln, T), (V, oo, V))
        Hdep = -R*T*(Z-1) -integrate(diff(Z, T)/V, (V, oo, V))*R*T**2
        '''
        try:
            return self._H_dep
        except:
            pass

        T, V = self.T, self._V
        V2 = V*V
        RT = R*T
        self._H_dep = H_dep = RT*(T*(2.0*V*self.dB_dT() + self.dC_dT())/(2.0*V2)
               - (-1.0 + (V2 + V*self.B() + self.C())/V2))
        return H_dep

    def dH_dep_dT(self):
        r'''

        .. math::
           \frac{\partial H_{dep}}{\partial T} = \frac{R \left(2 T^{2} V
           \frac{d^{2}}{d T^{2}} B{\left(T \right)} + T^{2} \frac{d^{2}}{d T^{2}}
           C{\left(T \right)} + 2 T V \frac{d}{d T} B{\left(T \right)}
           - 2 V B{\left(T \right)} - 2 C{\left(T \right)}\right)}{2 V^{2}}

        '''
        try:
            return self._dH_dep_dT
        except:
            pass
        T, V = self.T, self._V
        self._dH_dep_dT = dH_dep_dT = (R*(2.0*T*T*V*self.d2B_dT2() + T*T*self.d2C_dT2()
            + 2.0*T*V*self.dB_dT() - 2.0*V*self.B() - 2.0*self.C())/(2.0*V*V))
        return dH_dep_dT

    def S_dep(self):
        r'''

        .. math::
           S_{dep} = \frac{R \left(- T \frac{d}{d T} C{\left(T \right)} + 2 V^{2}
           \ln{\left(\frac{V^{2} + V B{\left(T \right)} + C{\left(T \right)}}
           {V^{2}} \right)} - 2 V \left(T \frac{d}{d T} B{\left(T \right)}
            + B{\left(T \right)}\right) - C{\left(T \right)}\right)}{2 V^{2}}

        '''
        '''
        dP_dT = diff(P_sln, T)
        S_dep = integrate(dP_dT - R/V, (V, oo, V)) + R*log(Z)

        '''
        try:
            return self._S_dep
        except:
            pass

        T, V = self.T, self._V
        V2 = V*V
        RT = R*T
        self._S_dep = S_dep = (R*(-T*self.dC_dT() + 2*V**2*log((V**2 + V*self.B() + self.C())/V**2)
        - 2*V*(T*self.dB_dT() + self.B()) - self.C())/(2*V**2))
        return S_dep

    def dS_dep_dT(self):
        r'''

        .. math::
           \frac{\partial S_{dep}}{\partial T} = \frac{R \left(2 V^{2} \left(V
           \frac{d}{d T} B{\left(T \right)} + \frac{d}{d T} C{\left(T \right)}
           \right) - \left(V^{2} + V B{\left(T \right)} + C{\left(T \right)}
           \right) \left(T \frac{d^{2}}{d T^{2}} C{\left(T \right)} + 2 V
           \left(T \frac{d^{2}}{d T^{2}} B{\left(T \right)} + 2 \frac{d}{d T}
           B{\left(T \right)}\right) + 2 \frac{d}{d T} C{\left(T \right)}
           \right)\right)}{2 V^{2} \left(V^{2} + V B{\left(T \right)}
           + C{\left(T \right)}\right)}

        '''
        try:
            return self._dS_dep_dT
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V

        self._dS_dep_dT = dS_dep_dT = (R*(2.0*V2*(V*self.dB_dT() + self.dC_dT()) - (V2 + V*self.B() + self.C())*(T*self.d2C_dT2()
        + 2.0*V*(T*self.d2B_dT2() + 2.0*self.dB_dT()) + 2.0*self.dC_dT()))/(2.0*V2*(V2 + V*self.B() + self.C())))
        return dS_dep_dT

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        new.N = self.N

        new.HeatCapacityGases = self.HeatCapacityGases
        new.model = self.model
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        Z = Z_from_virial_density_form(T, P, new.B(), new.C())
        new._V = Z*R*T/P
        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.N = self.N
        new.HeatCapacityGases = self.HeatCapacityGases
        new.model = model = self.model
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        if T is not None:
            new.T = T
            if P is not None:
                new.P = P
                Z = Z_from_virial_density_form(T, P, new.B(), new.C())
                new._V = Z*R*T/P
            elif V is not None:
                P = new.P = R*T*(V*V + V*new.B() + new.C())/(V*V*V)
                new._V = V
        elif P is not None and V is not None:
            new.P = P
            # PV specified, solve for T
            def err(T):
                # Solve for P matching; probably there is a better solution here that does not
                # require the cubic solution but this works for now
                # TODO: instead of using self.to_TP_zs to allow calculating B and C,
                # they should be functional
                new_tmp = self.to_TP_zs(T=T, P=P, zs=zs)
                B = new_tmp.B()
                C = new_tmp.C()
                x2 = V*V + V*B + C
                x3 = R/(V*V*V)

                P_err = T*x2*x3 - P
                dP_dT = x3*(T*(V*new_tmp.dB_dT() + new_tmp.dC_dT()) + x2)
                return P_err, dP_dT

            T_ig = P*V/R # guess
            T = newton(err, T_ig, fprime=True, xtol=1e-15)
            new.T = T
        else:
            raise ValueError("Two of T, P, or V are needed")

        return new

    def B(self):
        try:
            return self._B
        except:
            pass
        N = self.N
        T = self.T
        if N == 1:
            return self.model.B_pures(T)[0]
        zs = self.zs
        B_matrix = self.model.B_matrix(T)
        B = 0.0
        for i in range(N):
            B_tmp = 0.0
            row = B_matrix[i]
            for j in range(N):
                B += zs[j]*row[j]
            B += zs[i]*B_tmp

        self._B = B
        return B

    def dB_dT(self):
        try:
            return self._dB_dT
        except:
            pass
        N = self.N
        T = self.T
        if N == 1:
            return self.model.dB_dT_pures(T)[0]
        zs = self.zs
        dB_dT_matrix = self.model.dB_dT_matrix(T)
        dB_dT = 0.0
        for i in range(N):
            dB_dT_tmp = 0.0
            row = dB_dT_matrix[i]
            for j in range(N):
                dB_dT += zs[j]*row[j]
            dB_dT += zs[i]*dB_dT_tmp

        self._dB_dT = dB_dT
        return dB_dT

    def d2B_dT2(self):
        try:
            return self._d2B_dT2
        except:
            pass
        N = self.N
        T = self.T
        if N == 1:
            return self.model.d2B_dT2_pures(T)[0]
        zs = self.zs
        d2B_dT2_matrix = self.model.d2B_dT2_matrix(T)
        d2B_dT2 = 0.0
        for i in range(N):
            d2B_dT2_tmp = 0.0
            row = d2B_dT2_matrix[i]
            for j in range(N):
                d2B_dT2 += zs[j]*row[j]
            d2B_dT2 += zs[i]*d2B_dT2_tmp

        self._d2B_dT2 = d2B_dT2
        return d2B_dT2

    def C(self):
        try:
            return self._C
        except:
            pass
        T = self.T
        zs = self.zs
        C_pures = self.model.C_pures(T)
        Ciij, Cijj = self.model.C_interactions(T)
        C = 0.0
        N = self.N
        for i in range(N):
            for j in range(N):
                # poling 5-4.3b should be able to be used to take out the k loop?
                for k in range(N):
                    if i == j == k:
                        Cval = C_pures[i]
                    elif i == j:
                        Cval = Ciij[i][j]
                    else:
                        Cval = Cijj[i][j]
                    C += zs[i]*zs[j]*zs[k]*Cval
        self._C = C
        return C

    def dC_dT(self):
        try:
            return self._dC_dT
        except:
            pass
        T = self.T
        zs = self.zs
        dC_dT_pures = self.model.dC_dT_pures(T)
        dC_dTiij, dC_dTijj = self.model.dC_dT_interactions(T)
        dC_dT = 0.0
        N = self.N
        for i in range(N):
            for j in range(N):
                # poling 5-4.3b should be able to be used to take out the k loop?
                for k in range(N):
                    if i == j == k:
                        dC_dTval = dC_dT_pures[i]
                    elif i == j:
                        dC_dTval = dC_dTiij[i][j]
                    else:
                        dC_dTval = dC_dTijj[i][j]
                    dC_dT += zs[i]*zs[j]*zs[k]*dC_dTval
        self._dC_dT = dC_dT
        return dC_dT

    def d2C_dT2(self):
        try:
            return self._d2C_dT2
        except:
            pass
        T = self.T
        zs = self.zs
        d2C_dT2_pures = self.model.d2C_dT2_pures(T)
        d2C_dT2iij, d2C_dT2ijj = self.model.d2C_dT2_interactions(T)
        d2C_dT2 = 0.0
        N = self.N
        for i in range(N):
            for j in range(N):
                # poling 5-4.3b should be able to be used to take out the k loop?
                for k in range(N):
                    if i == j == k:
                        d2C_dT2val = d2C_dT2_pures[i]
                    elif i == j:
                        d2C_dT2val = d2C_dT2iij[i][j]
                    else:
                        d2C_dT2val = d2C_dT2ijj[i][j]
                    d2C_dT2 += zs[i]*zs[j]*zs[k]*d2C_dT2val
        self._d2C_dT2 = d2C_dT2
        return d2C_dT2

class HumidAirRP1485(VirialGas):
    is_gas = True
    is_liquid = False
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    def __init__(self, Hfs=None, Gfs=None, T=None, P=None, zs=None,
                 ):
        # Although in put is zs, it is required to be in the order of
        # (air, water) mole fraction
        self.Hfs = Hfs
        self.Gfs = Gfs
        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            self.Sfs = [(Hfi - Gfi)/298.15 for Hfi, Gfi in zip(Hfs, Gfs)]
        else:
            self.Sfs = None

        if zs is not None:
            self.N = N = len(zs)
        elif HeatCapacityGases is not None:
            self.N = N = len(HeatCapacityGases)
        if zs is not None:
            self.psi_w = psi_w = zs[1]
            self.psi_a = psi_a = zs[0]
            self.zs = zs
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P
        if T is not None and P is not None and zs is not None:
            self.air = DryAirLemmon(T=T, P=P)
            self.water = IAPWS95(T=T, P=P)
            Z = Z_from_virial_density_form(T, P, self.B(), self.C())
            self._V = Z*R*T/P
            self._MW = DryAirLemmon._MW*psi_a + IAPWS95._MW*psi_w

    def B(self):
        try:
            return self._B
        except:
            pass
        Baa = self.air.B_virial()
        Baw = TEOS10_BAW_derivatives(self.T)[0]
        Bww = self.water.B_virial()
        psi_a, psi_w = self.psi_a, self.psi_w

        self._B = B = psi_a*psi_a*Baa + 2.0*psi_a*psi_w*Baw + psi_w*psi_w*Bww
        return B

    def C(self):
        try:
            return self._C
        except:
            pass
        T = self.T
        Caaa = self.air.C_virial()
        Cwww = self.water.C_virial()
        Caww = TEOS10_CAWW_derivatives(T)[0]
        Caaw = TEOS10_CAAW_derivatives(T)[0]
        psi_a, psi_w = self.psi_a, self.psi_w
        self._C = C = (psi_a*psi_a*(Caaa + 3.0*psi_w*Caaw)
                       + psi_w*psi_w*(3.0*psi_a*Caww + psi_w*Cwww))
        return C





class HelmholtzEOS(Phase):
    __full_path__ = "%s.%s" %(__module__, __qualname__)

    model_attributes = ('model_name',)

    def __repr__(self):
        r'''Method to create a string representation of the phase object, with
        the goal of making it easy to obtain standalone code which reproduces
        the current state of the phase. This is extremely helpful in creating
        new test cases.

        Returns
        -------
        recreation : str
            String which is valid Python and recreates the current state of
            the object if ran, [-]

        Examples
        --------
        >>> from thermo import IAPWS95Gas
        >>> phase = IAPWS95Gas(T=300, P=1e5, zs=[1])
        >>> phase
        IAPWS95Gas(T=300, P=100000.0, zs=[1.0])
        '''
        base = '%s('  %(self.__class__.__name__)
        for s in ('Hfs', 'Gfs', 'Sfs', 'T', 'P', 'zs'):
            if hasattr(self, s) and getattr(self, s) is not None:
                base += '%s=%s, ' %(s, getattr(self, s))
        if base[-2:] == ', ':
            base = base[:-2]
        base += ')'
        return base

    def V(self):
        return self._V

    def A(self):
        try:
            return self._A
        except:
            pass
        A = self._A = self.A0 + self._Ar_func(self.tau, self.delta)
        return A

    def dA_ddelta(self):
        try:
            return self._dA_ddelta
        except:
            pass
        delta = self.delta
        dA_ddelta = self._dA_ddelta = self._dAr_ddelta_func(self.tau, delta) + 1./delta
        return dA_ddelta

    def d2A_ddelta2(self):
        try:
            return self._d2A_ddelta2
        except:
            pass
        delta = self.delta
        self._d2A_ddelta2 = d2A_ddelta2 = self._d2Ar_ddelta2_func(self.tau, delta) - 1./(delta*delta)
        return d2A_ddelta2


    def d3A_ddelta3(self):
        try:
            return self._d3A_ddelta3
        except:
            pass
        delta = self.delta
        self._d3A_ddelta3 = d3A_ddelta3 = self._d3Ar_ddelta3_func(self.tau, delta) + 2./(delta*delta*delta)
        return d3A_ddelta3

    def dA_dtau(self):
        try:
            return self._dA_dtau
        except:
            pass
        dA_dtau = self._dA_dtau = self._dAr_dtau_func(self.tau, self.delta) + self.dA0_dtau
        return dA_dtau

    def d2A_dtau2(self):
        try:
            return self._d2A_dtau2
        except:
            pass
        self._d2A_dtau2 = d2A_dtau2 = self._d2Ar_dtau2_func(self.tau, self.delta) + self.d2A0_dtau2
        return d2A_dtau2

    def d2A_ddeltadtau(self):
        try:
            return self._d2A_ddeltadtau
        except:
            pass
        self._d2A_ddeltadtau = d2A_ddeltadtau = self._d2Ar_ddeltadtau_func(self.tau, self.delta)
        return d2A_ddeltadtau

    def d3A_ddelta2dtau(self):
        try:
            return self._d3A_ddelta2dtau
        except:
            pass
        self._d3A_ddelta2dtau = d3A_ddelta2dtau = self._d3Ar_ddelta2dtau_func(self.tau, self.delta)
        return d3A_ddelta2dtau

    def d3A_ddeltadtau2(self):
        try:
            return self._d3A_ddeltadtau2
        except:
            pass
        self._d3A_ddeltadtau2 = d3A_ddeltadtau2 = self._d3Ar_ddeltadtau2_func(self.tau, self.delta)
        return d3A_ddeltadtau2

    def lnphis(self):
        delta = self.delta
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            A = self._A
        except:
            A = self.A()
        x0 = delta*(dA_ddelta - 1.0/delta)
        lnphi = A - self.A0 + x0 - log(x0 + 1.0)

        return [lnphi]

    def dlnphis_dV_T(self):
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        delta = self.delta
        rho, rho_red = 1.0/self._V, self.rho_red
        x0 = self.rho_red_inv
        rho_inv = 1.0/rho

        x1 = -rho_red*rho_inv
        x2 = rho*(d2A_ddelta2*x0 + rho_red*rho_inv*rho_inv)
        x3 = dA_ddelta + x1
        dlnphi_dV_T = x0*(2.0*dA_ddelta + x1 + x2 - (x2 + x3)/(rho*x0*x3 + 1.0) - 1.0/delta)
#
        dlnphi_dV_T *= -1.0/(self._V*self._V)

        return [dlnphi_dV_T]

    def dlnphis_dT_V(self):
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()
        try:
            dA_dtau = self._dA_dtau
        except:
            dA_dtau = self.dA_dtau()

        rho = 1.0/self._V
        rho_red_inv = self.rho_red_inv
        rho_red = self.rho_red

        T_red, T = self.T_red, self.T

        dlnphi_dT_V = T_red/(T*T)*(rho_red_inv*rho*d2A_ddeltadtau*(-1.0
                        + 1.0/(rho*(dA_ddelta - rho_red/rho)*rho_red_inv + 1.0))
                        - dA_dtau
                        + self.dA0_dtau)
        return [dlnphi_dT_V]

    def dlnphis_dT_P(self):
        try:
            return self._dlnphis_dT_P
        except:
            pass

        self._dlnphis_dT_P = dlnphis_dT_P = [self.dlnphis_dT_V()[0] - self.dlnphis_dV_T()[0]*self.dP_dT_V()/self.dP_dV_T()]
        return dlnphis_dT_P
    dlnphis_dT = dlnphis_dT_P

    def dlnphis_dP_V(self):
        return [self.dlnphis_dT_V()[0]/self.dP_dT_V()]

    def dlnphis_dV_P(self):
        return [self.dlnphis_dT_P()[0]/self.dV_dT_P()]

    def dlnphis_dP_T(self):
        return [self.dlnphis_dP_V()[0] - self.dlnphis_dV_P()[0]*self.dT_dP_V()/self.dT_dV_P()]

    dlnphis_dP = dlnphis_dP_T

    def S(self):
        try:
            return self._S
        except:
            pass
        try:
            dA_dtau = self._dA_dtau
        except:
            dA_dtau = self.dA_dtau()
        try:
            A = self._A
        except:
            A = self.A()
        self._S = S = (self.tau*dA_dtau - A)*self.R
        return S

    def S_dep(self):
        S_ideal = (self.tau*self.dA0_dtau - self.A0)*self.R
        return self.S() - S_ideal


    def dS_dT_V(self):
        try:
            return self._dS_dT_V
        except:
            pass
        try:
            d2A_dtau2 = self._d2A_dtau2
        except:
            d2A_dtau2 = self.d2A_dtau2()

        T, T_red = self.T, self.T_red
        self._dS_dT_V = dS_dT_V = (-T_red*T_red*d2A_dtau2)*self.R/(T*T*T)
        return dS_dT_V

    def dS_dV_T(self):
        try:
            return self._dS_dV_T
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()

        R, T, V, rho_red, T_red = self.R, self.T, self._V, self.rho_red, self.T_red

        self._dS_dV_T = dS_dV_T = R*(dA_ddelta - T_red*d2A_ddeltadtau/(T))/(V*V*rho_red)
        return dS_dV_T

    def dS_dP_T(self):
        # From chain rule
        return self.dS_dV_T()/self.dP_dV()
    dS_dP = dS_dP_T


    def dS_dT_P(self):
        try:
            return self._dS_dT_P
        except:
            pass

        self._dS_dT_P = dS_dT_P = self.dS_dT_V() - self.dS_dV_T()*self.dP_dT_V()/self.dP_dV_T()
        return dS_dT_P
    dS_dT = dS_dT_P

    def dS_dP_V(self):
        return self.dS_dT_V()/self.dP_dT_V()

    def H(self):
        try:
            return self._H
        except:
            pass
        try:
            dA_dtau = self._dA_dtau
        except:
            dA_dtau = self.dA_dtau()
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        self._H = H = (self.tau*dA_dtau + dA_ddelta*self.delta)*self.T*self.R
        return H

    def H_dep(self):
        try:
            return self._H_dep
        except:
            pass
        # Might be right, might not - hard to check due to reference state
        H = self.H()
        dA_dtau = self.dA0_dtau
        dA_ddelta = 1./self.delta

        H_ideal = (self.tau*dA_dtau + dA_ddelta*self.delta)*self.T*self.R
        self._H_dep = H - H_ideal
        return self._H_dep

    def dH_dT_V(self):
        try:
            return self._dH_dT_V
        except:
            pass
        try:
            dA_dtau = self._dA_dtau
        except:
            dA_dtau = self.dA_dtau()
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()
        try:
            d2A_dtau2 = self._d2A_dtau2
        except:
            d2A_dtau2 = self.d2A_dtau2()


        T, T_red, V, R, rho_red = self.T, self.T_red, self._V, self.R, self.rho_red
        T_inv = 1.0/T

        self._dH_dT_V = dH_dT_V = R*T*(-T_red*dA_dtau*T_inv*T_inv - T_inv*T_inv*T_red*d2A_ddeltadtau/(V*rho_red) - T_red*T_red*T_inv*T_inv*T_inv*d2A_dtau2) + R*(dA_ddelta/(V*rho_red) + T_red*dA_dtau/T)
        return dH_dT_V

    def dH_dV_T(self):
        try:
            return self._dH_dV_T
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()

        T, T_red, V, R, rho_red = self.T, self.T_red, self._V, self.R, self.rho_red

        self._dH_dV_T = dH_dV_T = R*T*(-dA_ddelta/(V*V*rho_red) - d2A_ddelta2/(V*V*V*rho_red*rho_red) - T_red*d2A_ddeltadtau/(T*V*V*rho_red))
        return dH_dV_T

    def dH_dP_V(self):
        return self.dH_dT_V()/self.dP_dT_V()

    def dH_dP_T(self):
        # From chain rule
        return self.dH_dV_T()/self.dP_dV()
    dH_dP = dH_dP_T

    def dH_dV_P(self):
        return self.dH_dT_P()/self.dV_dT_P()



    def Cv(self):
        try:
            return self._Cv
        except:
            pass
        try:
            d2A_dtau2 = self._d2A_dtau2
        except:
            d2A_dtau2 = self.d2A_dtau2()
        tau = self.tau
        self._Cv = Cv = -tau*tau*d2A_dtau2*self.R
        return Cv

    def Cp(self):
        try:
            return self._Cp
        except:
            pass
        tau, delta = self.tau, self.delta
        try:
            d2A_dtau2 = self._d2A_dtau2
        except:
            d2A_dtau2 = self.d2A_dtau2()
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()

        x0 = delta*dA_ddelta - delta*tau*d2A_ddeltadtau
        den = delta*(dA_ddelta + dA_ddelta + delta*d2A_ddelta2)
        self._Cp = Cp = (-tau*tau*d2A_dtau2 + x0*x0/den)*self.R
        return Cp

    dH_dT = dH_dT_P = Cp


    def dP_dT(self):
        try:
            return self._dP_dT
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()
        self._dP_dT = dP_dT = self.R/self._V*self.delta*(dA_ddelta - self.tau*d2A_ddeltadtau)
        return dP_dT

    def dP_dV(self):
        try:
            return self._dP_dV
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        delta = self.delta

        rho = self.rho()
        self._dP_dV = dP_dV = -self.R*rho*rho*self.T*delta*(2.0*dA_ddelta + delta*d2A_ddelta2)
        return dP_dV

    dP_dT_V = dP_dT
    dP_dV_T = dP_dV

    def d2P_dT2(self):
        try:
            return self._d2P_dT2
        except:
            pass
        d3A_ddeltadtau2 = self.d3A_ddeltadtau2()
        T, T_red, delta, V, rho_red, R = self.T, self.T_red, self.delta, self._V, self.rho_red, self.R
        self._d2P_dT2 = d2P_dT2 = R*T_red*T_red*d3A_ddeltadtau2/(T*T*T*V*V*rho_red)
        return d2P_dT2

    def d2P_dV2(self):
        try:
            return self._d2P_dV2
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        try:
            d3A_ddelta3 = self._d3A_ddelta3
        except:
            d3A_ddelta3 = self.d3A_ddelta3()
        '''
        from sympy import *
        from chemicals import Vm_to_rho
        T_red, rho_red, V, R = symbols('T_red, rho_red, V, R')
        T = symbols('T')
        rho = 1/V

        iapws95_dA_ddelta = symbols('ddelta', cls=Function)

        rho_red_inv = (1/rho_red)
        tau = T_red/T
        delta = rho*rho_red_inv
        dA_ddelta = iapws95_dA_ddelta(tau, delta)
        P = (dA_ddelta*delta)*rho*R*T
        print(diff(P,V, 2))
        '''
        R, T, V, rho_red = self.R, self.T, self._V, self.rho_red

        self._d2P_dV2 = d2P_dV2 = R*T*(6.0*dA_ddelta + (2*d2A_ddelta2 + d3A_ddelta3/(V*rho_red))/(V*rho_red) + 4*d2A_ddelta2/(V*rho_red))/(V**4*rho_red)
        return d2P_dV2

    def d2P_dTdV(self):
        try:
            return self._d2P_dTdV
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()
        d3A_ddelta2dtau = self.d3A_ddelta2dtau()
        R, T, T_red, V, rho_red = self.R, self.T, self.T_red, self._V, self.rho_red

        self._d2P_dTdV = d2P_dTdV = R*(-2.0*dA_ddelta - d2A_ddelta2/(V*rho_red) + 2.0*T_red*d2A_ddeltadtau/T
                                       + T_red*d3A_ddelta2dtau/(T*V*rho_red))/(V*V*V*rho_red)
        return d2P_dTdV

    def B_virial(self):
        try:
            f0 = self._dAr_ddelta_func(self.tau, 0.0)
        except:
            f0 = self._dAr_ddelta_func(self.tau, 1e-20)
        return f0/self.rho_red

    def dB_virial_dT(self):
        tau = self.tau
        try:
            f0 = self._d2Ar_ddeltadtau_func(tau, 0.0)
        except:
            f0 = self._d2Ar_ddeltadtau_func(tau, 1e-20)
        return -tau*tau*f0/(self.rho_red*self.T_red)

    def d2B_virial_dT2(self):
        T, T_red, tau = self.T, self.T_red, self.tau

        try:
            f0 = self._d2Ar_ddeltadtau_func(tau, 0.0)
        except:
            f0 = self._d2Ar_ddeltadtau_func(tau, 1e-20)
        try:
            f1 = self._d3Ar_ddeltadtau2_func(tau, 0.0)
        except:
            f1 = self._d3Ar_ddeltadtau2_func(tau, 1e-20)
        return T_red*(2.0*f0 + tau*f1)/(T*T*T*self.rho_red)

    def d3B_virial_dT3(self):
        T, T_red, tau = self.T, self.T_red, self.tau

        try:
            f0 = self._d2Ar_ddeltadtau_func(tau, 0.0)
        except:
            f0 = self._d2Ar_ddeltadtau_func(tau, 1e-20)
        try:
            f1 = self._d3Ar_ddeltadtau2_func(tau, 0.0)
        except:
            f1 = self._d3Ar_ddeltadtau2_func(tau, 1e-20)
        try:
            f2 = self._d4Ar_ddeltadtau3_func(tau, 0.0)
        except:
            f2 = self._d4Ar_ddeltadtau3_func(tau, 1e-20)
        return (-T_red*(6.0*f0 + 6.0*tau*f1 + tau*tau*f2)/(T*T*T*T*self.rho_red))

    def C_virial(self):
        try:
            f0 = self._d2Ar_ddelta2_func(self.tau, 0.0)
        except:
            f0 = self._d2Ar_ddelta2_func(self.tau, 1e-20)
        return f0/(self.rho_red*self.rho_red)

    def dC_virial_dT(self):
        tau = self.tau
        try:
            f0 = self._d3Ar_ddelta2dtau_func(tau, 0.0)
        except:
            f0 = self._d3Ar_ddelta2dtau_func(tau, 1e-20)
        return -tau*tau*f0/(self.rho_red*self.rho_red*self.T_red)

    def d2C_virial_dT2(self):
        T, T_red, tau = self.T, self.T_red, self.tau
        try:
            f0 = self._d3Ar_ddelta2dtau_func(tau, 0.0)
        except:
            f0 = self._d3Ar_ddelta2dtau_func(tau, 1e-20)

        try:
            f1 = self._d4Ar_ddelta2dtau2_func(tau, 0.0)
        except:
            f1 = self._d4Ar_ddelta2dtau2_func(tau, 1e-20)
        return T_red*(2.0*f0 + tau*f1)/(T*T*T*self.rho_red*self.rho_red)

class DryAirLemmon(HelmholtzEOS):
    model_name = 'lemmon2000'
    is_gas = True
    is_liquid = False
    force_phase = 'g'
    __full_path__ = "%s.%s" %(__module__, __qualname__)

    _MW = lemmon2000_air_MW
    _MW = 28.96546 # CoolProp
    rho_red = lemmon2000_air_rho_reducing
    rho_red_inv = 1.0/rho_red
    T_red = lemmon2000_air_T_reducing
    R = lemmon2000_air_R

    zs = [1.0]
    cmps = [0]
    T_MAX_FIXED = 2000.0
    T_MIN_FIXED = 132.6313 # For now gas only.

    _Ar_func = staticmethod(lemmon2000_air_Ar)

    _d3Ar_ddeltadtau2_func = staticmethod(lemmon2000_air_d3Ar_ddeltadtau2)
    _d3Ar_ddelta2dtau_func = staticmethod(lemmon2000_air_d3Ar_ddelta2dtau)
    _d2Ar_ddeltadtau_func = staticmethod(lemmon2000_air_d2Ar_ddeltadtau)

    _dAr_dtau_func = staticmethod(lemmon2000_air_dAr_dtau)
    _d2Ar_dtau2_func = staticmethod(lemmon2000_air_d2Ar_dtau2)
    _d3Ar_dtau3_func = staticmethod(lemmon2000_air_d3Ar_dtau3)
    _d4Ar_dtau4_func = staticmethod(lemmon2000_air_d4Ar_dtau4)

    _dAr_ddelta_func = staticmethod(lemmon2000_air_dAr_ddelta)
    _d2Ar_ddelta2_func = staticmethod(lemmon2000_air_d2Ar_ddelta2)
    _d3Ar_ddelta3_func = staticmethod(lemmon2000_air_d3Ar_ddelta3)
    _d4Ar_ddelta4_func = staticmethod(lemmon2000_air_d4Ar_ddelta4)

    _d4Ar_ddelta2dtau2_func = staticmethod(lemmon2000_air_d4Ar_ddelta2dtau2)
    _d4Ar_ddelta3dtau_func = staticmethod(lemmon2000_air_d4Ar_ddelta3dtau)
    _d4Ar_ddeltadtau3_func = staticmethod(lemmon2000_air_d4Ar_ddeltadtau3)

    def __init__(self, T=None, P=None, zs=None):
        self.T = T
        self.P = P
        self._rho = rho = lemmon2000_rho(T, P)
        self._V = 1.0/rho
        self.tau = tau = self.T_red/T
        self.delta = delta = rho*self.rho_red_inv
        self.A0 = lemmon2000_air_A0(tau, delta)
        self.dA0_dtau = lemmon2000_air_dA0_dtau(tau, delta)
        self.d2A0_dtau2 = lemmon2000_air_d2A0_dtau2(tau, delta)
        self.d3A0_dtau3 = lemmon2000_air_d3A0_dtau3(tau, delta)

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.T = T
        new.P = P
        new._rho = rho = lemmon2000_rho(T, P)
        new._V = 1.0/rho
        new.tau = tau = new.T_red/T
        new.delta = delta = rho*new.rho_red_inv
        new.A0 = lemmon2000_air_A0(tau, delta)
        new.dA0_dtau = lemmon2000_air_dA0_dtau(tau, delta)
        new.d2A0_dtau2 = lemmon2000_air_d2A0_dtau2(tau, delta)
        new.d3A0_dtau3 = lemmon2000_air_d3A0_dtau3(tau, delta)
        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        if T is not None and P is not None:
            new.T = T
            new._rho = lemmon2000_rho(T, P)
            new._V = 1.0/new._rho
            new.P = P
        elif T is not None and V is not None:
            new._rho = 1.0/V
            new._V = V
            P = lemmon2000_P(T, new._rho)
        elif P is not None and V is not None:
            raise NotImplementedError("TODO")
        else:
            raise ValueError("Two of T, P, or V are needed")

        new.P = P
        new.T = T
        new.tau = tau = new.T_red/T
        new.delta = delta = new._rho*new.rho_red_inv

        new.A0 = lemmon2000_air_A0(tau, delta)
        new.dA0_dtau = lemmon2000_air_dA0_dtau(tau, delta)
        new.d2A0_dtau2 = lemmon2000_air_d2A0_dtau2(tau, delta)
        new.d3A0_dtau3 = lemmon2000_air_d3A0_dtau3(tau, delta)
        return new


    def mu(self):
        try:
            return self._mu
        except:
            pass
        self._mu = mu = mu_air_lemmon(self.T, self._rho)
        return mu


class IAPWS95(HelmholtzEOS):
    model_name = 'iapws95'
    _MW = iapws95_MW
    Tc = iapws95_Tc
    Pc = iapws95_Pc
    rhoc_mass = iapws95_rhoc
    rhoc_mass_inv = 1.0/iapws95_rhoc
    __full_path__ = "%s.%s" %(__module__, __qualname__)

    rhoc_inv = rho_to_Vm(rhoc_mass, iapws95_MW)
    rhoc = 1.0/rhoc_inv

    rho_red = rhoc
    rho_red_inv = rhoc_inv

    T_red = Tc

    _MW_kg = _MW*1e-3
    R = _MW_kg*iapws95_R # This is just the gas constant 8.314... but matching iapws to their decimals
    R_inv = 1.0/R

    #R = property_mass_to_molar(iapws95_R, iapws95_MW)
    zs = [1.0]
    cmps = [0]
#    HeatCapacityGases = iapws_correlations.HeatCapacityGases

    T_MAX_FIXED = 5000.0
    T_MIN_FIXED = 235.0

    _d4Ar_ddelta2dtau2_func = staticmethod(iapws95_d4Ar_ddelta2dtau2)
    _d3Ar_ddeltadtau2_func = staticmethod(iapws95_d3Ar_ddeltadtau2)
    _d3Ar_ddelta2dtau_func = staticmethod(iapws95_d3Ar_ddelta2dtau)
    _d2Ar_ddeltadtau_func = staticmethod(iapws95_d2Ar_ddeltadtau)
    _d2Ar_dtau2_func = staticmethod(iapws95_d2Ar_dtau2)
    _dAr_dtau_func = staticmethod(iapws95_dAr_dtau)
    _d3Ar_ddelta3_func = staticmethod(iapws95_d3Ar_ddelta3)
    _d2Ar_ddelta2_func = staticmethod(iapws95_d2Ar_ddelta2)
    _dAr_ddelta_func = staticmethod(iapws95_dAr_ddelta)
    _Ar_func = staticmethod(iapws95_Ar)


    def __init__(self, T=None, P=None, zs=None):
        self.T = T
        self.P = P
        self._rho_mass = rho_mass = iapws95_rho(T, P)
        self._V = rho_to_Vm(rho=rho_mass, MW=self._MW)
        self.tau = tau = self.Tc/T
        self.delta = delta = rho_mass*self.rhoc_mass_inv
        self.A0, self.dA0_dtau, self.d2A0_dtau2, self.d3A0_dtau3 = iapws95_A0_tau_derivatives(tau, delta)

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.T = T
        new.P = P
        new._rho_mass = rho_mass = iapws95_rho(T, P)
        new._V = rho_to_Vm(rho=rho_mass, MW=self._MW)
        new.tau = tau = new.Tc/T
        new.delta = delta = rho_mass*new.rhoc_mass_inv
        new.A0, new.dA0_dtau, new.d2A0_dtau2, new.d3A0_dtau3 = iapws95_A0_tau_derivatives(tau, delta)
        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        if T is not None and P is not None:
            new.T = T
            new._rho_mass = rho_mass = iapws95_rho(T, P)
            new._V = rho_to_Vm(rho=rho_mass, MW=self._MW)
            new.P = P
        elif T is not None and V is not None:
            new.T = T
            new._rho_mass = rho_mass = 1e-3*self._MW/V
            P = iapws95_P(T, rho_mass)
            new._V = V
            new.P = P
        elif P is not None and V is not None:
            new._rho_mass = rho_mass = Vm_to_rho(V, MW=self._MW)
            T = new.T = iapws95_T(P, rho_mass)
            new._V = V
            new.P = P
        else:
            raise ValueError("Two of T, P, or V are needed")

        new.P = P
        new.T = T
        new.tau = tau = new.Tc/T
        new.delta = delta = rho_mass*new.rhoc_mass_inv
        new.A0, new.dA0_dtau, new.d2A0_dtau2, new.d3A0_dtau3 = iapws95_A0_tau_derivatives(tau, delta)

        return new


    def mu(self):
        r'''Calculate and return the viscosity of water according to the IAPWS.
        For details, see :obj:`chemicals.viscosity.mu_IAPWS`.

        Returns
        -------
        mu : float
            Viscosity of water, [Pa*s]
        '''
        try:
            return self._mu
        except:
            pass
        self.__mu_k()
        return self._mu

    def k(self):
        r'''Calculate and return the thermal conductivity of water according to the IAPWS.
        For details, see :obj:`chemicals.thermal_conductivity.k_IAPWS`.

        Returns
        -------
        k : float
            Thermal conductivity of water, [W/m/K]
        '''
        try:
            return self._k
        except:
            pass
        self.__mu_k()
        return self._k

    def __mu_k(self):
        drho_mass_dP = self.drho_mass_dP()

        # TODO: curve fit drho_dP_Tr better than IAPWS did (mpmath)
        drho_dP_Tr = self.to(T=self.Tc*1.5, V=self._V, zs=[1]).drho_mass_dP()
        self._mu = mu_IAPWS(T=self.T, rho=self._rho_mass, drho_dP=drho_mass_dP,
                        drho_dP_Tr=drho_dP_Tr)

        self._k = k_IAPWS(T=self.T, rho=self._rho_mass, Cp=self.Cp_mass(), Cv=self.Cv_mass(),
                       mu=self._mu, drho_dP=drho_mass_dP, drho_dP_Tr=drho_dP_Tr)



class IAPWS95Gas(IAPWS95):
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    is_gas = True
    is_liquid = False
    force_phase = 'g'

class IAPWS95Liquid(IAPWS95):
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    force_phase = 'l'
    is_gas = False
    is_liquid = True

class IAPWS97(Phase):
    model_name = 'iapws97'
    model_attributes = ('model_name',)
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    _MW = 18.015268
    R = 461.526
    Tc = 647.096
    Pc = 22.064E6
    rhoc = 322.
    zs = [1.0]
    cmps = [0]
    def mu(self):
        return mu_IAPWS(T=self.T, rho=self._rho_mass)

    def k(self):
        # TODO add properties; even industrial formulation recommends them
        return k_IAPWS(T=self.T, rho=self._rho_mass)

    ### Region 1,2,5 Gibbs
    def G(self):
        try:
            return self._G
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            G = iapws97_G_region1(tau, pi)
        elif region == 2:
            G = (iapws97_Gr_region2(tau, pi) + iapws97_G0_region2(tau, pi))
        elif region == 5:
            G = (iapws97_Gr_region5(tau, pi) + iapws97_G0_region5(tau, pi))
        elif region == 4:
            G = self.H() - self.T*self.S()
        self._G = G
        return G


    def dG_dpi(self):
        try:
            return self._dG_dpi
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            dG_dpi = iapws97_dG_dpi_region1(tau, pi)
        elif region == 2:
            dG_dpi = 1.0/pi + iapws97_dGr_dpi_region2(tau, pi)
        elif region == 5:
            dG_dpi = 1.0/pi + iapws97_dGr_dpi_region5(tau, pi)
        self._dG_dpi = dG_dpi
        return dG_dpi

    def d2G_d2pi(self):
        try:
            return self._d2G_d2pi
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            d2G_d2pi = iapws97_d2G_dpi2_region1(tau, pi)
        elif region == 2:
            d2G_d2pi = -1.0/(pi*pi) + iapws97_d2Gr_dpi2_region2(tau, pi)
        elif region == 5:
            d2G_d2pi = -1.0/(pi*pi) + iapws97_d2Gr_dpi2_region5(tau, pi)
        self._d2G_d2pi = d2G_d2pi
        return d2G_d2pi

    def dG_dtau(self):
        try:
            return self._dG_dtau
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            dG_dtau = iapws97_dG_dtau_region1(tau, pi)
        elif region == 2:
            dG_dtau = iapws97_dG0_dtau_region2(tau, pi) + iapws97_dGr_dtau_region2(tau, pi)
        elif region == 5:
            dG_dtau = iapws97_dG0_dtau_region5(tau, pi) + iapws97_dGr_dtau_region5(tau, pi)
        self._dG_dtau = dG_dtau
        return dG_dtau

    def d2G_d2tau(self):
        try:
            return self._d2G_d2tau
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            d2G_d2tau = iapws97_d2G_dtau2_region1(tau, pi)
        elif region == 2:
            d2G_d2tau = (iapws97_d2Gr_dtau2_region2(tau, pi)
                         + iapws97_d2G0_dtau2_region2(tau, pi))
        elif region == 5:
            d2G_d2tau = (iapws97_d2Gr_dtau2_region5(tau, pi)
                         + iapws97_d2G0_dtau2_region5(tau, pi))
        self._d2G_d2tau = d2G_d2tau
        return d2G_d2tau

    def d2G_dpidtau(self):
        try:
            return self._d2G_dpidtau
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            d2G_dpidtau = iapws97_d2G_dpidtau_region1(tau, pi)
        elif region == 2:
            d2G_dpidtau = iapws97_d2Gr_dpidtau_region2(tau, pi)
        elif region == 5:
            d2G_dpidtau = iapws97_d2Gr_dpidtau_region5(tau, pi)
        self._d2G_dpidtau = d2G_dpidtau
        return d2G_dpidtau


    ### Region 3 Helmholtz
    def A_region3(self):
        try:
            return self._A_region3
        except:
            pass
        self._A_region3 = A_region3 = iapws97_A_region3_region3(self.tau, self.delta)
        return A_region3

    def dA_ddelta(self):
        try:
            return self._dA_ddelta
        except:
            pass
        self._dA_ddelta = dA_ddelta = iapws97_dA_ddelta_region3(self.tau, self.delta)
        return dA_ddelta

    def d2A_d2delta(self):
        try:
            return self._d2A_d2delta
        except:
            pass
        self._d2A_d2delta = d2A_d2delta = iapws97_d2A_d2delta_region3(self.tau, self.delta)
        return d2A_d2delta

    def dA_dtau(self):
        try:
            return self._dA_dtau
        except:
            pass
        self._dA_dtau = dA_dtau = iapws97_dA_dtau_region3(self.tau, self.delta)
        return dA_dtau

    def d2A_d2tau(self):
        try:
            return self._d2A_d2tau
        except:
            pass
        self._d2A_d2tau = d2A_d2tau = iapws97_d2A_d2tau_region3(self.tau, self.delta)
        return d2A_d2tau

    def d2A_ddeltadtau(self):
        try:
            return self._d2A_ddeltadtau
        except:
            pass
        self._d2A_ddeltadtau = d2A_ddeltadtau = iapws97_d2A_ddeltadtau_region3(self.tau, self.delta)
        return d2A_ddeltadtau

    def __init__(self, T=None, P=None, zs=None):
        self.T = T
        self.P = P
        self._rho_mass = iapws97_rho(T, P)
        self._V = rho_to_Vm(rho=self._rho_mass, MW=self._MW)
        self.region = region = iapws97_identify_region_TP(T, P)
        if region == 1:
            self.pi = P*6.049606775559589e-08 #1/16.53E6
            self.tau = 1386.0/T
            self.Pref = 16.53E6
            self.Tref = 1386.0
        elif region == 2:
            self.pi = P*1e-6
            self.tau = 540.0/T
            self.Pref = 1e6
            self.Tref = 540.0
        elif region == 3:
            self.tau = self.Tc/T
            self.Tref = self.Tc
            self.delta = self._rho_mass*0.003105590062111801 # 1/322.0
            self.rhoref = 322.0
        elif region == 5:
            self.pi = P*1e-6
            self.tau = 1000.0/T
            self.Tref = 1000.0
            self.Pref = 1e6



    def to_TP_zs(self, T, P, zs, other_eos=None):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        self._rho_mass = iapws97_rho(T, P)
        self._V = rho_to_Vm(rho=self._rho_mass, MW=self._MW)
        self.region = region = iapws97_identify_region_TP(T, P)
        if region == 1:
            self.pi = P*6.049606775559589e-08 #1/16.53E6
            self.tau = 1386.0/T
        elif region == 2:
            self.pi = P*1e-6
            self.tau = 540.0/T
        elif region == 3:
            self.tau = self.Tc/T
            self.delta = self._rho_mass*0.003105590062111801 # 1/322.0
        elif region == 5:
            self.pi = P*1e-6
            self.tau = 1000.0/T

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs

        if T is not None:
            new.T = T
            if P is not None:
                new._rho_mass = rho_mass = iapws97_rho(T, P)
                new._V = rho_to_Vm(rho=rho_mass, MW=self._MW)
                new.P = P
            elif V is not None:
                new._rho_mass = rho_mass = Vm_to_rho(V, MW=self._MW)
                P = iapws97_P(T, rho_mass)
                new.V = V
                new.P = P
        elif P is not None and V is not None:
            new._rho_mass = rho_mass = Vm_to_rho(V, MW=self._MW)
            T = new.T = iapws97_T(P, rho_mass)
            new.V = V
            new.P = P
        else:
            raise ValueError("Two of T, P, or V are needed")

        new.region = region = iapws97_identify_region_TP(new.T, new.P)
        if region == 1:
            new.pi = P*6.049606775559589e-08 #1/16.53E6
            new.tau = 1386.0/T
            new.Pref = 16.53E6
            new.Tref = 1386.0
        elif region == 2:
            new.pi = P*1e-6
            new.tau = 540.0/T
            new.Pref = 1e6
            new.Tref = 540.0
        elif region == 3:
            new.tau = new.Tc/T
            new.Tref = new.Tc
            new.delta = new._rho_mass*0.003105590062111801 # 1/322.0
            new.rhoref = 322.0
        elif region == 5:
            new.pi = P*1e-6
            new.tau = 1000.0/T
            new.Tref = 1000.0
            new.Pref = 1e6

        new.P = P
        new.T = T

        return new

    def V(self):
        return self._V

    def U(self):
        try:
            return self._U
        except:
            pass

        if self.region != 3:
            try:
                dG_dtau = self._dG_dtau
            except:
                dG_dtau = self.dG_dtau()
            try:
                dG_dpi = self._dG_dpi
            except:
                dG_dpi = self.dG_dpi()
            U = self.R*self.T(*self.tau*dG_dtau - self.pi*dG_dpi)
        self._U = U
        return U

    def S(self):
        try:
            return self._S
        except:
            pass
        if self.region != 3:
            try:
                G = self._G
            except:
                G = self.G()
            try:
                dG_dtau = self._dG_dtau
            except:
                dG_dtau = self.dG_dtau()
            S = self.R*(self.tau*dG_dtau - G)
        self._S = S
        return S

    def H(self):
        try:
            return self._H
        except:
            pass
        if self.region != 3:
            try:
                dG_dtau = self._dG_dtau
            except:
                dG_dtau = self.dG_dtau()
            H = self.R*self.T*self.tau*dG_dtau
        self._H = H
        return H

    def Cv(self):
        try:
            return self._Cv
        except:
            pass
        if self.region != 3:
            try:
                d2G_d2tau = self._d2G_d2tau
            except:
                d2G_d2tau = self.d2G_d2tau()
            try:
                dG_dpi = self._dG_dpi
            except:
                dG_dpi = self.dG_dpi()
            try:
                d2G_dpidtau = self._d2G_dpidtau
            except:
                d2G_dpidtau = self.d2G_dpidtau()
            try:
                d2G_d2pi = self._d2G_d2pi
            except:
                d2G_d2pi = self.d2G_d2pi()

            tau = self.tau
            x0 = (dG_dpi - tau*d2G_dpidtau)
            Cv = self.R*(-tau*tau*d2G_d2tau + x0*x0/d2G_d2pi)


    def Cp(self):
        try:
            return self._Cp
        except:
            pass

        if self.region == 3:
            tau, delta = self.tau, self.delta # attributes set on init
            try:
                dA_ddelta = self._dA_ddelta
            except:
                dA_ddelta = self.dA_ddelta()
            try:
                d2A_ddeltadtau = self._d2A_ddeltadtau
            except:
                d2A_ddeltadtau = self.d2A_ddeltadtau()
            try:
                d2A_d2delta = self._d2A_d2delta
            except:
                d2A_d2delta = self.d2A_d2delta()
            try:
                d2A_d2tau = self._d2A_d2tau
            except:
                d2A_d2tau = self.d2A_d2tau()

            x0 = (delta*dA_ddelta - delta*tau*d2A_ddeltadtau)
            Cp = self.R*(-tau*tau*d2A_d2tau + x0*x0/(delta*(2.0*dA_ddelta + delta*d2A_d2delta)))

#        self.Cp = (-self.tau**2*self.ddA_ddtau + (self.delta*self.dA_ddelta - self.delta*self.tau*self.ddA_ddelta_dtau)**2\
#                  /(2*self.delta*self.dA_ddelta + self.delta**2*self.ddA_dddelta))*R

        else:
            tau = self.tau
            Cp = -self.R*tau*tau*self.d2G_d2tau()
        Cp *= self._MW*1e-3
        self._Cp = Cp
        return Cp

    dH_dT = dH_dT_P = Cp

    ### Derivatives
    def dV_dP(self):
        '''
        from sympy import *
        R, T, MW, P, Pref, Tref = symbols('R, T, MW, P, Pref, Tref')
        dG_dpif = symbols('dG_dpi', cls=Function)
        pi = P/Pref
        tau = Tref/T
        dG_dpi = dG_dpif(tau, pi)
        V = (R*T*pi*dG_dpi*MW)/(1000*P)
        print(diff(V, P))

        MW*R*T*Subs(Derivative(dG_dpi(Tref/T, _xi_2), _xi_2), _xi_2, P/Pref)/(1000*Pref**2)
        '''
        try:
            return self._dV_dP
        except:
            pass
        if self.region != 3:
            try:
                d2G_d2pi = self._d2G_d2pi
            except:
                d2G_d2pi = self.d2G_d2pi()
            dV_dP = self._MW*self.R*self.T*d2G_d2pi/(1000.0*self.Pref*self.Pref)

        self._dV_dP = dV_dP
        return dV_dP

    def dV_dT(self):
        # similar to dV_dP
        try:
            return self._dV_dT
        except:
            pass
        if self.region != 3:
            try:
                dG_dpi = self._dG_dpi
            except:
                dG_dpi = self.dG_dpi()
            try:
                d2G_dpidtau = self._d2G_dpidtau
            except:
                d2G_dpidtau = self.d2G_dpidtau()


            dV_dT = (self._MW*self.R*dG_dpi/(1000*self.Pref)
            - self._MW*self.R*self.Tref*d2G_dpidtau/(1000*self.Pref*self.T))
        self._dV_dT = dV_dT
        return dV_dT

    def dP_dT(self):
        try:
            return self._dP_dT
        except:
            pass
        if self.region != 3:
            dP_dT = -self.dV_dT()/self.dV_dP()
        self._dP_dT = dP_dT
        return dP_dT

    def dP_dV(self):
        return 1.0/self.dV_dP()




# Emperically measured to be ~140 KB/instance, do not want to cache too many - 35 is 5 MB
max_CoolProp_states = 35
global CoolProp
global CoolProp_constants_set
CoolProp_constants_set = False
def set_coolprop_constants():
    global CPPT_INPUTS, CPrhoT_INPUTS, CPrhoP_INPUTS, CPiP, CPiT, CPiDmolar, CPiHmolar, CPiSmolar
    global CPPQ_INPUTS, CPQT_INPUTS, CoolProp_gas_phases, CoolProp_liquid_phases
    global CPliquid, CPgas, CPunknown, caching_states_CoolProp, caching_state_CoolProp
    global CoolProp
    import CoolProp
    CoolProp_constants_set = True
    CPPT_INPUTS = CoolProp.PT_INPUTS
    CPrhoT_INPUTS = CoolProp.DmolarT_INPUTS
    CPrhoP_INPUTS = CoolProp.DmolarP_INPUTS

    CPiP, CPiT, CPiDmolar = CoolProp.iP, CoolProp.iT, CoolProp.iDmolar
    CPiHmolar, CPiSmolar = CoolProp.iHmolar, CoolProp.iSmolar

    CPPQ_INPUTS, CPQT_INPUTS = CoolProp.PQ_INPUTS, CoolProp.QT_INPUTS

    CoolProp_gas_phases = set([CoolProp.iphase_gas, CoolProp.iphase_supercritical, CoolProp.iphase_unknown,
                              CoolProp.iphase_critical_point, CoolProp.iphase_supercritical_gas])
    CoolProp_liquid_phases = set([CoolProp.iphase_liquid, CoolProp.iphase_supercritical_liquid])

    CPliquid = CoolProp.iphase_liquid
    CPgas = CoolProp.iphase_gas
    CPunknown = CoolProp.iphase_not_imposed

    # Probably todo - hold onto ASs for up to 1 sec, then release them for reuse
    # Do not allow Phase direct access any more, use a decorator
#    CoolProp_AS_cache = {}
#    def get_CoolProp_AS(backend, fluid):
#        key = (backend, fluid)
#        try:
#            in_use, free = CoolProp_AS_cache[key]
#            if free:
#                AS = free.pop()
#            else:
#                AS = CoolProp.AbstractState(backend, fluid)
#            in_use.add(AS)
##            in_use.append(AS)
#            return AS
#        except KeyError:
##            in_use, free = [], []
#            in_use, free = set([]), set([])
#            AS = CoolProp.AbstractState(backend, fluid)
#            in_use.add(AS)
##            in_use.append(AS)
#            CoolProp_AS_cache[key] = (in_use, free)
#            return AS
#
#    def free_CoolProp_AS(AS, backend, fluid):
#        key = (backend, fluid)
#        try:
#            in_use, free = CoolProp_AS_cache[key]
#        except KeyError:
#            raise ValueError("Should not happen")
#        in_use.remove(AS)
##        free.append(AS)
#        free.add(AS)


    # Forget about time - just use them last; make sure the LRU is at the top
    #
    if not SORTED_DICT:
        caching_states_CoolProp = OrderedDict()
    else:
        caching_states_CoolProp = {}

    def caching_state_CoolProp(backend, fluid, spec0, spec1, spec_set, phase, zs):
        # Pretty sure about as optimized as can get!
        # zs should be a tuple, not a list
        key = (backend, fluid, spec0, spec1, spec_set, phase, zs)
        if key in caching_states_CoolProp:
            AS = caching_states_CoolProp[key]
            try:
                caching_states_CoolProp.move_to_end(key)
            except:
                # Move to end the old fashioned way
                del caching_states_CoolProp[key]
                caching_states_CoolProp[key] = AS
        elif len(caching_states_CoolProp) < max_CoolProp_states:
            # Always make a new item until the cache is full
            AS = CoolProp.AbstractState(backend, fluid)
            AS.specify_phase(phase)
            if zs is not None:
                AS.set_mole_fractions(zs)
            AS.update(spec_set, spec0, spec1) # A failed call here takes ~400 us.
            caching_states_CoolProp[key] = AS
            return AS
        else:
            # Reuse an item if not in the cache, making the value go to the end of
            # the ordered dict
            if not SORTED_DICT:
                old_key, AS = caching_states_CoolProp.popitem(False)
            else:
                # Hack - get first item in dict
                old_key = next(iter(caching_states_CoolProp))
                AS = caching_states_CoolProp.pop(old_key)

            if old_key[1] != fluid or old_key[0] != backend:
                # Handle different components - other will be gc
                AS = CoolProp.AbstractState(backend, fluid)
            AS.specify_phase(phase)
            if zs is not None:
                AS.set_mole_fractions(zs)
            AS.update(spec_set, spec0, spec1)
            caching_states_CoolProp[key] = AS
        return AS

CPgas = 5
CPliquid = 0
CPunknown = 8
CPPQ_INPUTS = 2
CPQT_INPUTS = 1
CPiDmolar = 24
CPrhoT_INPUTS = 11
caching_state_CoolProp = None

class CoolPropPhase(Phase):
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    prefer_phase = 8
    ideal_gas_basis = False


    def __str__(self):
        if self.phase == 'g':
            s =  '<%s, ' %('CoolPropGas')
        else:
            s =  '<%s, ' %('CoolPropLiquid')
        try:
            s += 'T=%g K, P=%g Pa' %(self.T, self.P)
        except:
            pass
        s += '>'
        return s

#    def __del__(self):
#        # Not sustainable at all
#        # time-based cache seems next best
#        free_CoolProp_AS(self.AS, self.backend, self.fluid)


    @property
    def phase(self):
        try:
            idx = self.AS.phase()
            if idx in CoolProp_gas_phases:
                return 'g'
            return 'l'
        except:
            if self.prefer_phase == CPliquid:
                return 'l'
            return 'g'

    model_attributes = ('backend', 'fluid', 'Hfs', 'Gfs', 'Sfs')

    def __init__(self, backend, fluid,
                 T=None, P=None, zs=None,  Hfs=None,
                 Gfs=None, Sfs=None,):
        if not CoolProp_constants_set:
            if has_CoolProp():
                set_coolprop_constants()
            else:
                raise ValueError("CoolProp is not installed")

        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs

        self.backend = backend
        self.fluid = fluid

        self.skip_comp = skip_comp = (backend in ('IF97') or fluid in ('water') or '&' not in fluid)
        if zs is None:
            zs = [1.0]
        self.zs = zs
        self.N = N = len(zs)
        if skip_comp or N == 1:
            zs_key = None
        else:
            zs_key = tuple(zs)
        if T is not None and P is not None:
            self.T = T
            self.P = P
            try:
                key = [backend, fluid, P, T, CPPT_INPUTS, self.prefer_phase, zs_key]
                AS = caching_state_CoolProp(*key)
            except:
                key = [backend, fluid, P, T, CPPT_INPUTS, CPunknown, zs_key]
                AS = caching_state_CoolProp(*key)
            self.key = key
            self._cache_easy_properties(AS)
#        if not skip_comp and zs is None:
#            self.zs = [1.0]

#            AS = get_CoolProp_AS(backend, fluid)#CoolProp.AbstractState(backend, fluid)
#            if not skip_comp:
#                AS.set_mole_fractions(zs)
#            AS.specify_phase(self.prefer_phase)
#            try:
#                AS.update(CPPT_INPUTS, P, T)
#            except:
#                AS.specify_phase(CPunknown)
#                AS.update(CPPT_INPUTS, P, T)
#
#            rho = AS.rhomolar()
#            key = (backend, fluid, T, rho)
    @property
    def AS(self):
        return caching_state_CoolProp(*self.key)

    def to_TP_zs(self, T, P, zs):
        return self.to(T=T, P=P, zs=zs)

    def from_AS(self, AS):
        new = self.__class__.__new__(self.__class__)
        new.N = N = self.N
        if N == 1:
            zs_key = None
            new.zs = self.zs
        else:
            new.zs = zs = AS.get_mole_fractions()
            zs_key = tuple(zs)
        new.backend = backend = self.backend
        new.fluid = fluid = self.fluid
        new.skip_comp = self.skip_comp
        new.T, new.P = T, P = AS.T(), AS.p()
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs

        # Always use density as an input - does not require a phase ID spec / setting with AS.phase() seems to not work
        new._cache_easy_properties(AS)
        new.key = (backend, fluid, self._rho, T, CPrhoT_INPUTS, CPunknown, zs_key)
        return new

    def to(self, zs, T=None, P=None, V=None, prefer_phase=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.N = self.N
        new.backend = backend = self.backend
        new.fluid = fluid = self.fluid
        new.skip_comp = skip_comp = self.skip_comp
        if skip_comp or N == 1:
            zs_key = None
        else:
            zs_key = tuple(zs)

        if prefer_phase is None:
            prefer_phase = self.prefer_phase
        try:
            if T is not None:
                if P is not None:
                    new.T, new.P = T, P
                    key = (backend, fluid, P, T, CPPT_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
                elif V is not None:
                    key = (backend, fluid, 1.0/V, T, CPrhoT_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
    #                    AS.update(CPrhoT_INPUTS, 1.0/V, T)
                    new.T, new.P = T, AS.p()
            elif P is not None and V is not None:
                    key = (backend, fluid, 1.0/V, P, CPrhoP_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
    #                AS.update(CPrhoP_INPUTS, 1.0/V, P)
                    new.T, new.P = AS.T(), P
        except ValueError:
            prefer_phase = CPunknown
            if T is not None:
                if P is not None:
                    new.T, new.P = T, P
                    key = (backend, fluid, P, T, CPPT_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
                elif V is not None:
                    key = (backend, fluid, 1.0/V, T, CPrhoT_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
                    new.T, new.P = T, AS.p()
            elif P is not None and V is not None:
                    key = (backend, fluid, 1.0/V, P, CPrhoP_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
                    new.T, new.P = AS.T(), P

        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        new.key = key
        new._cache_easy_properties(AS)
        return new

    def _cache_easy_properties(self, AS):
        self._rho = AS.rhomolar()
        self._V = 1.0/self._rho
        self._H = AS.hmolar()
        self._S = AS.smolar()
        self._Cp = AS.cpmolar()
        self._PIP = AS.PIP()

    def V(self):
        return self._V
#        return 1.0/self.AS.rhomolar()

    def lnphis(self):
        try:
            return self._lnphis
        except AttributeError:
            pass
        self._lnphis = lnphis = []
        AS = self.AS
        for i in range(self.N):
            lnphis.append(log(AS.fugacity_coefficient(i)))
        return lnphis

    lnphis_G_min = lnphis

    def dlnphis_dT(self):
        raise NotImplementedError("Not in CoolProp")

    def dlnphis_dP(self):
        raise NotImplementedError("Not in CoolProp")

    def dlnphis_dns(self):
        raise NotImplementedError("Not in CoolProp")

    def dlnphis_dzs(self):
        raise NotImplementedError("Not in CoolProp")

    def gammas(self):
        raise NotImplementedError("TODO")

    def dP_dT(self):
        return self.AS.first_partial_deriv(CPiP, CPiT, CPiDmolar)
    dP_dT_V = dP_dT

    def dP_dV(self):
        rho = self.AS.rhomolar()
        dP_drho = self.AS.first_partial_deriv(CPiP, CPiDmolar, CPiT)
        return -dP_drho*rho*rho
    dP_dV_T = dP_dV

    def d2P_dT2(self):
        return self.AS.second_partial_deriv(CPiP, CPiT, CPiDmolar, CPiT, CPiDmolar)
    d2P_dT2_V = d2P_dT2

    def d2P_dV2(self):
        d2P_drho2 = self.AS.second_partial_deriv(CPiP, CPiDmolar, CPiT, CPiDmolar, CPiT)
        V = self.V()
        dP_dV = self.dP_dV()
        return (d2P_drho2/-V**2 + 2.0*V*dP_dV)/-V**2
    d2P_dV2_T = d2P_dV2

    def d2P_dTdV(self):
        d2P_dTdrho = self.AS.second_partial_deriv(CPiP, CPiT, CPiDmolar, CPiDmolar, CPiT)
        rho = self.AS.rhomolar()
        return -d2P_dTdrho*rho*rho

    def PIP(self):
        return self._PIP
        # Saves time
#        return self.AS.PIP()

    def H(self):
        return self._H
#        return self.AS.hmolar()

    def S(self):
        return self._S
#        return self.AS.smolar()

    def H_dep(self):
        return self.AS.hmolar_excess()

    def S_dep(self):
        return self.AS.smolar_excess()

    def Cp_dep(self):
        raise NotImplementedError("Not in CoolProp")

    def Cp(self):
        return self._Cp
#        return self.AS.cpmolar()
    dH_dT = Cp

    def dH_dP(self):
        return self.AS.first_partial_deriv(CoolProp.iHmolar, CPiP, CPiT)

    def dH_dT_V(self):
        # Does not need rho multiplication
        return self.AS.first_partial_deriv(CoolProp.iHmolar, CPiT, CPiDmolar)

    def dH_dP_V(self):
        return self.AS.first_partial_deriv(CoolProp.iHmolar, CPiP, CPiDmolar)

    def dH_dV_T(self):
        rho = self.AS.rhomolar()
        return -self.AS.first_partial_deriv(CoolProp.iHmolar, CPiDmolar, CPiT)*rho*rho

    def dH_dV_P(self):
        rho = self.AS.rhomolar()
        return -self.AS.first_partial_deriv(CoolProp.iHmolar, CPiDmolar, CPiP)*rho*rho

    def d2H_dT2(self):
        return self.AS.second_partial_deriv(CoolProp.iHmolar, CPiT, CPiP, CPiT, CPiP)

    def d2H_dP2(self):
        return self.AS.second_partial_deriv(CoolProp.iHmolar, CPiP, CPiT, CPiP, CPiT)

    def d2H_dTdP(self):
        return self.AS.second_partial_deriv(CoolProp.iHmolar, CPiT, CPiP, CPiP, CPiT)

    def dS_dT(self):
        return self.AS.first_partial_deriv(CPiSmolar, CPiT, CPiP)

    def dS_dP(self):
        return self.AS.first_partial_deriv(CPiSmolar, CPiP, CPiT)

    def dS_dT_V(self):
        return self.AS.first_partial_deriv(CPiSmolar, CPiT, CPiDmolar)

    def dS_dP_V(self):
        return self.AS.first_partial_deriv(CPiSmolar, CPiP, CPiDmolar)

    def dS_dV_T(self):
        rho = self.AS.rhomolar()
        return -self.AS.first_partial_deriv(CPiSmolar, CPiDmolar, CPiT)*rho*rho

    def dS_dV_P(self):
        rho = self.AS.rhomolar()
        return -self.AS.first_partial_deriv(CPiSmolar, CPiDmolar, CPiP)*rho*rho

    def d2S_dT2(self):
        return self.AS.second_partial_deriv(CPiSmolar, CPiT, CPiP, CPiT, CPiP)

    def d2S_dP2(self):
        return self.AS.second_partial_deriv(CPiSmolar, CPiP, CPiT, CPiP, CPiT)

    def d2S_dTdP(self):
        return self.AS.second_partial_deriv(CPiSmolar, CPiT, CPiP, CPiP, CPiT)

    def mu(self):
        try:
            return self._mu
        except AttributeError:
            mu = self._mu = self.AS.viscosity()
        return mu

    def k(self):
        try:
            return self._k
        except AttributeError:
            k = self._k = self.AS.conductivity()
        return k

class CoolPropLiquid(CoolPropPhase):
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    prefer_phase = CPliquid
    is_gas = False
    is_liquid = True

class CoolPropGas(CoolPropPhase):
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    prefer_phase = CPgas
    is_gas = True
    is_liquid = False

class CombinedPhase(Phase):
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    def __init__(self, phases, equilibrium=None, thermal=None, volume=None,
                 other_props=None,
                 T=None, P=None, zs=None,
                 ):
        # phases : list[other phases]
        # equilibrium: index
        # thermal: index
        # volume: index
        # other_props: dict[prop] = phase index

        # may be missing S_formation_ideal_gas Hfs arg
        self.equilibrium = equilibrium
        self.thermal = thermal
        self.volume = volume
        self.other_props = other_props

        for i, p in enumerate(phases):
            if p.T != T or p.P != P or p.zs != zs:
                phases[i] = p.to(T=T, P=P, zs=zs)
        self.phases = phases

    def lnphis(self):
        # This style will save the getattr call but takes more time to code
        if 'lnphis' in self.other_props:
            return self.phases[self.other_props['lnphis']].lnphis()
        if self.equilibrium is not None:
            return self.phases[self.equilibrium].lnphis()
        raise ValueError("No method specified")

    def lnphis_G_min(self):
        if 'lnphis' in self.other_props:
            return self.phases[self.other_props['lnphis']].lnphis_G_min()
        if self.equilibrium is not None:
            return self.phases[self.equilibrium].lnphis_G_min()
        raise ValueError("No method specified")

    def makeeqfun(prop_name):
        def fun(self):
            if prop_name in self.other_props:
                return getattr(self.phases[self.other_props[prop_name]], prop_name)()
            if self.equilibrium is not None:
                return getattr(self.phases[self.equilibrium], prop_name)()
            raise ValueError("No method specified")
        return fun

    def makethermalfun(prop_name):
        def fun(self):
            if prop_name in self.other_props:
                return getattr(self.phases[self.other_props[prop_name]], prop_name)()
            if self.thermal is not None:
                return getattr(self.phases[self.thermal], prop_name)()
            raise ValueError("No method specified")
        return fun

    def makevolumefun(prop_name):
        def fun(self):
            if prop_name in self.other_props:
                return getattr(self.phases[self.other_props[prop_name]], prop_name)()
            if self.volume is not None:
                return getattr(self.phases[self.volume], prop_name)()
            raise ValueError("No method specified")
        return fun

    lnphis = makeeqfun("lnphis")
    dlnphis_dT = makeeqfun("dlnphis_dT")
    dlnphis_dP = makeeqfun("dlnphis_dP")
    dlnphis_dns = makeeqfun("dlnphis_dns")

    V = makevolumefun("V")
    dP_dT = makevolumefun("dP_dT")
    dP_dT_V = dP_dT
    dP_dV = makevolumefun("dP_dV")
    dP_dV_T = dP_dV
    d2P_dT2 = makevolumefun("d2P_dT2")
    d2P_dT2_V = d2P_dT2
    d2P_dV2 = makevolumefun("d2P_dV2")
    d2P_dV2_T = d2P_dV2
    d2P_dTdV = makevolumefun("d2P_dTdV")

    H = makethermalfun("H")
    S = makethermalfun("S")
    Cp = makethermalfun("Cp")
    dH_dT = makethermalfun("dH_dT")
    dH_dP = makethermalfun("dH_dP")
    dH_dT_V = makethermalfun("dH_dT_V")
    dH_dP_V = makethermalfun("dH_dP_V")
    dH_dV_T = makethermalfun("dH_dV_T")
    dH_dV_P = makethermalfun("dH_dV_P")
    dH_dzs = makethermalfun("dH_dzs")
    dS_dT = makethermalfun("dS_dT")
    dS_dP = makethermalfun("dS_dP")
    dS_dT_P = makethermalfun("dS_dT_P")
    dS_dT_V = makethermalfun("dS_dT_V")
    dS_dP_V = makethermalfun("dS_dP_V")
    dS_dzs = makethermalfun("dS_dzs")




gas_phases = (IdealGas, CEOSGas, CoolPropGas, IAPWS95Gas, VirialGas, HumidAirRP1485, DryAirLemmon)
liquid_phases = (CEOSLiquid, GibbsExcessLiquid, CoolPropLiquid, IAPWS95Liquid)
solid_phases = (GibbsExcessSolid,)
all_phases = gas_phases + liquid_phases + solid_phases + (IAPWS95, IAPWS97, CoolPropPhase)

phase_full_path_dict =  {c.__full_path__: c for c in all_phases}

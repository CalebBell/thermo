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

'''
__all__ = ['CoolPropPhase', 'CoolPropPhase', 'CoolPropLiquid', 'CoolPropGas']

import sys
from chemicals.utils import log
from collections import OrderedDict
from .phase import Phase
from thermo.coolprop import has_CoolProp


SORTED_DICT = sys.version_info >= (3, 6)
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
        if skip_comp or self.N == 1:
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
    prefer_phase = CPliquid
    is_gas = False
    is_liquid = True

class CoolPropGas(CoolPropPhase):
    prefer_phase = CPgas
    is_gas = True
    is_liquid = False

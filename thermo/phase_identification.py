# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
SOFTWARE.'''

from __future__ import division
__all__ = ['vapor_score_Tpc', 'vapor_score_Vpc', 
           'vapor_score_Tpc_weighted', 'vapor_score_Tpc_Vpc',
           'vapor_score_Wilson', 'vapor_score_Poling', 
           'vapor_score_PIP', 'vapor_score_Bennett_Schmidt', 
           'vapor_score_traces',
           
           'score_phases_S', 'score_phases_VL', 'identity_phase_states',
           'S_ID_METHODS', 'VL_ID_METHODS',
           
           'sort_phases', 'identify_sort_phases',
           
           'WATER_FIRST', 'WATER_LAST', 'WATER_NOT_SPECIAL',
           'WATER_SORT_METHODS', 'KEY_COMPONENTS_SORT', 'PROP_SORT',
           'SOLID_SORT_METHODS', 'LIQUID_SORT_METHODS'
           ]


from thermo.activity import Wilson_K_value, Rachford_Rice_flash_error, flash_inner_loop
from thermo.utils import phase_identification_parameter, Vm_to_rho

def vapor_score_Tpc(T, Tcs, zs):
    # Does not work for pure compounds
    # Basic
    Tpc =  0.0
    for i in range(len(zs)):
        Tpc += zs[i]*Tcs[i]
    return T - Tpc

def vapor_score_Vpc(V, Vcs, zs):
    # Basic
    Vpc =  0.0
    for i in range(len(zs)):
        Vpc += zs[i]*Vcs[i]
    return V - Vpc

def vapor_score_Tpc_weighted(T, Tcs, Vcs, zs, r1=1.0):
    # ECLIPSE method, r1 for tuning
    weight_sum = 0.0
    for i in range(len(zs)):
        weight_sum += zs[i]*Vcs[i]

    Tpc =  0.0
    for i in range(len(zs)):
        Tpc += zs[i]*Tcs[i]*Vcs[i]
    Tpc *= r1/weight_sum
        
    return T - Tpc

def vapor_score_Tpc_Vpc(T, V, Tcs, Vcs, zs):
    # Basic. Different mixing rules could be used to tune the system.
    Tpc =  0.0
    for i in range(len(zs)):
        Tpc += zs[i]*Tcs[i]
    Vpc =  0.0
    for i in range(len(zs)):
        Vpc += zs[i]*Vcs[i]
    return V*T*T - Vpc*Tpc*Tpc


def vapor_score_Wilson(T, P, zs, Tcs, Pcs, omegas):
    N = len(zs)
    if N == 1:
        Psat = Wilson_K_value(T, P, Tcs[0], Pcs[0], omegas[0])*P
        # Lower than vapor pressure - gas; higher than the vapor pressure - liquid
        return P - Psat
    # Does not work for pure compounds
    # Posivie - vapor, negative - liquid
    Ks = [Wilson_K_value(T, P, Tcs[i], Pcs[i], omegas[i]) for i in range(N)]
    # Consider a vapor fraction of more than 0.5 a vapor
    return flash_inner_loop(zs, Ks)[0] - 0.5
    # Go back to the error once unit tested
#    return Rachford_Rice_flash_error(V_over_F=0.5, zs=zs, Ks=Ks)
    

def vapor_score_Poling(kappa):
    # should be tested - may need to reverse sign
    # There is also a second criteria for the vapor phase
    return kappa - 0.05e5

def vapor_score_PIP(V, dP_dT, dP_dV, d2P_dV2, d2P_dVdT):    
    return -(phase_identification_parameter(V, dP_dT, dP_dV, d2P_dV2, d2P_dVdT) - 1.0)

def vapor_score_Bennett_Schmidt(dbeta_dT):
    return -dbeta_dT

def vapor_score_traces(zs, CASs, trace_CASs=['74-82-8', '7727-37-9'], Tcs=None):
    # traces should be the lightest species - high = more vapor like
    if trace_CASs is not None:
        for trace_CAS in trace_CASs:
            try:
                return zs[CASs.index(trace_CAS)]
            except ValueError:
                # trace component not in mixture
                pass
    
    # Return the composition of the compound with the lowest critical temp
    comp = 0.0
    Tc_min = 1e100
    for i in range(len(zs)):
        if Tcs[i] < Tc_min:
            comp = zs[i]
            Tc_min = Tcs[i]
    return comp


VL_ID_TPC = 'Tpc'
VL_ID_VPC = 'Vpc'
VL_ID_TPC_VC_WEIGHTED = 'Tpc Vpc weighted'
VL_ID_TPC_VPC = 'Tpc Vpc'
VL_ID_WILSON = 'Wilson'
VL_ID_POLING = 'Poling'
VL_ID_PIP = 'PIP'
VL_ID_BS = 'Bennett-Schmidt'
VL_ID_TRACES = 'Traces'

VL_ID_METHODS = [VL_ID_TPC, VL_ID_VPC, VL_ID_TPC_VC_WEIGHTED, VL_ID_TPC_VPC,
                 VL_ID_WILSON, VL_ID_POLING, VL_ID_PIP, VL_ID_BS, VL_ID_TRACES]

S_ID_D2P_DVDT = 'd2P_dVdT'
S_ID_METHODS = [S_ID_D2P_DVDT]

def score_phases_S(phases, constants, correlations, method, S_ID_settings=None):
    # The higher the score (above zero), the more solid-like
    if method == S_ID_D2P_DVDT:
        scores = [i.d2P_dVdT() for i in phases]
    return scores

def score_phases_VL(phases, constants, correlations, method, 
                    VL_ID_settings=None):
    # The higher the score (above zero), the more vapor-like
    if phases:
        T = phases[0].T
    if method == VL_ID_TPC:
        Tcs = constants.Tcs
        scores = [vapor_score_Tpc(T, Tcs, i.zs) for i in phases]
    elif method == VL_ID_VPC:
        Vcs = constants.Vcs
        scores = [vapor_score_Vpc(i.V(), Vcs, i.zs) for i in phases]
    elif method == VL_ID_TPC_VC_WEIGHTED:
        Tcs = constants.Tcs
        Vcs = constants.Vcs
        scores = [vapor_score_Tpc_weighted(T, Tcs, Vcs, i.zs) for i in phases]
    elif method == VL_ID_TPC_VPC:
        Tcs = constants.Tcs
        Vcs = constants.Vcs
        scores = [vapor_score_Tpc_Vpc(T, i.V(), Tcs, Vcs, i.zs) for i in phases]
    elif method == VL_ID_WILSON:
        Tcs = constants.Tcs
        Pcs = constants.Pcs
        omegas = constants.omegas
        scores = [vapor_score_Wilson(T, i.P, i.zs, Tcs, Pcs, omegas) for i in phases]
    elif method == VL_ID_POLING:
        scores = [vapor_score_Poling(i.kappa()) for i in phases]
    elif method == VL_ID_PIP:
        scores = [-(i.PIP() - 1.0) for i in phases]
#        scores = [vapor_score_PIP(i.V(), i.dP_dT(), i.dP_dV(),
#                                           i.d2P_dV2(), i.d2P_dVdT()) for i in phases]
    elif method == VL_ID_BS:
        scores = [vapor_score_Bennett_Schmidt(i.dbeta_dT()) for i in phases]
    elif method == VL_ID_TRACES:
        CASs = constants.CASs
        Tcs = constants.Tcs
        scores = [vapor_score_traces(i.zs, CASs, Tcs=Tcs) for i in phases]
    return scores
    
        
def identity_phase_states(phases, constants, correlations, VL_method=VL_ID_PIP, 
                          S_method=S_ID_D2P_DVDT,
                          VL_ID_settings=None, S_ID_settings=None, 
                          skip_solids=False):
    # TODO - unit test
    # TODO - optimize
    # Takes a while

    force_phases = [i.force_phase for i in phases]
    forced = True
    for s in force_phases:
        if s is None:
            forced = False
            break
        
    if not forced:
        VL_scores = score_phases_VL(phases, constants, correlations, 
                                    method=VL_method, VL_ID_settings=VL_ID_settings)
        if not skip_solids:
            S_scores = score_phases_S(phases, constants, correlations, 
                                      method=S_method, S_ID_settings=S_ID_settings)
    
    solids = []
    liquids = []
    possible_gases = []
    possible_gas_scores = []
    
    for i in range(len(phases)):
        if force_phases[i] is not None:
            if force_phases[i] == 'l':
                liquids.append(phases[i])
            if force_phases[i] == 's':
                solids.append(phases[i])
            if force_phases[i] == 'g':
                gases.append(phases[i])
        elif not skip_solids and S_scores[i] >= 0.0:
            solids.append(phases[i])
        elif VL_scores[i] >= 0.0:
            possible_gases.append(phases[i])
            possible_gas_scores.append(VL_scores[i])
        else:
            liquids.append(phases[i])
    
    # Handle multiple matches as gas
    possible_gas_count = len(possible_gases)
    if possible_gas_count > 1:
        gas = possible_gases[possible_gas_scores.index(max(possible_gas_scores))]
        for possible_gas in possible_gases:
            if possible_gas is not gas:
                liquids.append(possible_gas)
        possible_gases[:] = (gas,)
    elif possible_gas_count == 1:
        gas = possible_gases[0]
    else:
        gas = None
    
    return gas, liquids, solids
        


DENSITY_MASS = 'DENSITY_MASS'
DENSITY = 'DENSITY'
ISOTHERMAL_COMPRESSIBILITY = 'ISOTHERMAL_COMPRESSIBILITY'
HEAT_CAPACITY = 'HEAT_CAPACITY'
L_SORT_PROPS = S_SORT_PROPS = [DENSITY_MASS, DENSITY, ISOTHERMAL_COMPRESSIBILITY,
                HEAT_CAPACITY]

WATER_FIRST = 'water first'
WATER_LAST = 'water last'
WATER_NOT_SPECIAL = 'water not special'

WATER_SORT_METHODS = [WATER_FIRST, WATER_LAST, WATER_NOT_SPECIAL]

KEY_COMPONENTS_SORT = 'key components'
PROP_SORT = 'prop'
SOLID_SORT_METHODS = LIQUID_SORT_METHODS = [PROP_SORT, KEY_COMPONENTS_SORT]

def key_cmp_sort(phases, cmps, cmps_neg):
    # TODO
    return phases

def mini_sort_phases(phases, sort_method, prop, cmps, cmps_neg,
                     reverse=True, constants=None):
    if sort_method == PROP_SORT:
        if prop == DENSITY_MASS:
            keys = []
            MWs = constants.MWs
            for p in phases:
                zs = p.zs
                MW = 0.0
                for i in constants.cmps:
                    MW += zs[i]*MWs[i]
                keys.append(Vm_to_rho(p.V(), MW))

            # for i in phases:
            #     i.constants = constants
            # keys = [i.rho_mass() for i in phases]
        elif prop == DENSITY:
            keys = [i.rho() for i in phases]
        elif prop == ISOTHERMAL_COMPRESSIBILITY:
            keys = [i.beta() for i in phases]
        elif prop == HEAT_CAPACITY:
            keys = [i.Cp() for i in phases]
        phases = [p for _, p in sorted(zip(keys, phases))]
        if reverse:
            phases.reverse()
    elif sort_method == KEY_COMPONENTS_SORT:
        phases = key_cmp_sort(phases, cmps, cmps_neg)
    return phases
    
def sort_phases(liquids, solids, constants, settings):
    
    if len(liquids) > 1:
        liquids = mini_sort_phases(liquids, sort_method=settings.liquid_sort_method,
                         prop=settings.liquid_sort_prop,
                         cmps=settings.liquid_sort_cmps, 
                         cmps_neg=settings.liquid_sort_cmps_neg,
                         reverse=settings.phase_sort_higher_first, constants=constants)

        # Handle water special
        if settings.water_sort != WATER_NOT_SPECIAL:
            # water phase - phase with highest fraction water
            water_index = constants.water_index
            if water_index is not None:
                water_zs = [i.zs[water_index] for i in liquids]
                water_max_zs = max(water_zs)
                if water_max_zs > 1e-4:
                    water_phase_index = water_zs.index(water_max_zs)
                    water = liquids.pop(water_phase_index)
                    if settings.water_sort == WATER_LAST:
                        liquids.append(water)
                    elif settings.water_sort == WATER_FIRST:
                        liquids.insert(water)
    if len(solids) > 1:
        solids = mini_sort_phases(solids, sort_method=settings.solid_sort_method,
                         prop=setings.solid_sort_prop,
                         cmps=settings.solid_sort_cmps, 
                         cmps_neg=settings.solid_sort_cmps_neg,
                         reverse=settings.phase_sort_higher_first, constants=constants)
    return liquids, solids


def identify_sort_phases(phases, betas, constants, correlations, settings,
                         skip_solids=False):
    gas, liquids, solids = identity_phase_states(phases, constants, correlations, 
                              VL_method=settings.VL_ID, 
                              S_method=settings.S_ID,
                              VL_ID_settings=settings.VL_ID_settings,
                              S_ID_settings=settings.S_ID_settings,
                              skip_solids=skip_solids)
    
    liquids, solids = sort_phases(liquids, solids, constants, settings)
    if betas is not None:
        new_betas = []
        if gas is not None:
            new_betas.append(betas[phases.index(gas)])
        for liquid in liquids:
            new_betas.append(betas[phases.index(liquid)])
        for solid in solids:
            new_betas.append(betas[phases.index(solid)])
    
    return gas, liquids, solids, betas
    
    

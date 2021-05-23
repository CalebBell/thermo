# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import sys
import importlib.util
import types
from types import ModuleType
import numpy as np
import inspect
import numba
import thermo
import fluids
import fluids.numba
from numba.core.registry import CPUDispatcher
normal_fluids = fluids
normal = thermo

orig_file = __file__
caching = False
'''


'''
__all__ = []
__funcs = {}

numerics = fluids.numba.numerics
replaced = fluids.numba.numerics_dict.copy()


def transform_complete_thermo(replaced, __funcs, __all__, normal, vec=False):
    cache_blacklist = set(['sequential_substitution_2P_functional',])
    __funcs.update(normal_fluids.numba.numbafied_fluids_functions.copy())

    blacklist = set(['identify_sort_phases', 'score_phases_S', 'score_phases_VL',
                     'identity_phase_states', 'sort_phases',
                    'sequential_substitution_2P',
                    'sequential_substitution_NP',
                    'sequential_substitution_Mehra_2P',
                    'sequential_substitution_GDEM3_2P',
                    'nonlin_equilibrium_NP',
                    'nonlin_spec_NP',
                    'nonlin_2P',
                    'nonlin_2P_HSGUAbeta',
                    'nonlin_n_2P',
                    'nonlin_2P_newton',
                    'minimize_gibbs_2P_transformed',
                    'minimize_gibbs_NP_transformed',
                    'TP_solve_VF_guesses',
                    'dew_P_newton',
                    'dew_bubble_newton_zs',
                    'dew_bubble_Michelsen_Mollerup',
                    'existence_3P_Michelsen_Mollerup',
                    'bubble_T_Michelsen_Mollerup',
                    'dew_T_Michelsen_Mollerup',
                    'bubble_P_Michelsen_Mollerup',
                    'dew_P_Michelsen_Mollerup',
                    'TPV_solve_HSGUA_1P',
                    'solve_PTV_HSGUA_1P',
                    'TPV_solve_HSGUA_guesses_1P',
                    'PH_secant_1P',
                    'PH_newton_1P',
                    'TVF_pure_newton',
                    'TVF_pure_secant',
                    'PVF_pure_newton',
                    'PVF_pure_secant',
                    'TSF_pure_newton',
                    'PSF_pure_newton',
                    'solve_T_VF_IG_K_composition_independent',
                    'solve_P_VF_IG_K_composition_independent',
                    'sequential_substitution_2P_sat',
                    'SS_VF_simultaneous',
                    'sequential_substitution_2P_HSGUAbeta',
                    'sequential_substitution_2P_double',
                    'stability_iteration_Michelsen',
                    'TPV_double_solve_1P',
                    'TPV_solve_HSGUA_guesses_VL',
                    'cm_flash_tol',
                    'chemgroups_to_matrix',
                    'load_unifac_ip',
                    'FlashPureVLS',
                    ])

    __funcs.update(normal_fluids.numba.numbafied_fluids_functions.copy())

    new_mods = normal_fluids.numba.transform_module(normal, __funcs, replaced, vec=vec,
                                                    blacklist=blacklist,
                                                    cache_blacklist=cache_blacklist)
    if vec:
        conv_fun = numba.vectorize
    else:
        conv_fun = numba.jit

    import chemicals.numba
    chemicals.numba.iapws # Force the transform to occur
    for name in dir(chemicals.numba):
        obj = getattr(chemicals.numba, name)
        if isinstance(obj, CPUDispatcher) or isinstance(obj, ModuleType):
            __funcs[name] = obj

    for mod in new_mods:
        mod.__dict__.update(__funcs)

    to_change = ['eos.volume_solutions_halley',
                 
                 'eos_mix_methods.a_alpha_quadratic_terms',
                 
                 'eos_mix_methods.a_alpha_and_derivatives_quadratic_terms',
                 'eos_mix_methods.a_alpha_aijs_composition_independent',
                 'eos_mix_methods.a_alpha_and_derivatives_full',

                'eos_mix_methods.PR_lnphis',
             'eos_mix_methods.VDW_lnphis',
             'eos_mix_methods.SRK_lnphis',
             'eos_mix_methods.eos_mix_lnphis_general',
             'eos_mix_methods.VDW_lnphis_fastest',
             'eos_mix_methods.PR_lnphis_fastest',
             'eos_mix_methods.SRK_lnphis_fastest',
             'eos_mix_methods.RK_lnphis_fastest',
             'eos_mix_methods.PR_translated_lnphis_fastest',
             'eos_mix_methods.SRK_translated_lnphis_fastest',
             'eos_mix_methods.G_dep_lnphi_d_helper',
             'eos_mix_methods.PR_translated_ddelta_dzs',
             'eos_mix_methods.PR_translated_ddelta_dns',
             'eos_mix_methods.PR_translated_depsilon_dzs',
             'eos_mix_methods.PR_translated_depsilon_dns',
             'eos_mix_methods.SRK_translated_ddelta_dns',
             'eos_mix_methods.SRK_translated_depsilon_dns',
             'eos_mix_methods.SRK_translated_d2epsilon_dzizjs',
             'eos_mix_methods.SRK_translated_depsilon_dzs',
             'eos_mix_methods.SRK_translated_d2delta_dninjs',
             'eos_mix_methods.SRK_translated_d3delta_dninjnks',
             'eos_mix_methods.SRK_translated_d2epsilon_dninjs',
             'eos_mix_methods.SRK_translated_d3epsilon_dninjnks',
             'eos_mix_methods.eos_mix_db_dns',
             'eos_mix_methods.eos_mix_da_alpha_dns',
             'eos_mix_methods.eos_mix_dV_dzs',
             'eos_mix_methods.eos_mix_a_alpha_volume',
             'eos_mix_methods.PR_ddelta_dzs',
             'eos_mix_methods.PR_ddelta_dns',
             'eos_mix_methods.PR_d2delta_dninjs',
             'eos_mix_methods.PR_d3delta_dninjnks',
             'eos_mix_methods.PR_depsilon_dns',
             'eos_mix_methods.PR_d2epsilon_dzizjs',
             'eos_mix_methods.PR_depsilon_dzs',
             'eos_mix_methods.PR_d2epsilon_dninjs',
             'eos_mix_methods.PR_d3epsilon_dninjnks',
             'eos_mix_methods.PR_translated_d2epsilon_dzizjs',
             'eos_mix_methods.PR_translated_d2epsilon_dninjs',
             'eos_mix_methods.PR_translated_d2delta_dninjs',
             'eos_mix_methods.PR_translated_d3delta_dninjnks',
             'eos_mix_methods.PR_translated_d3epsilon_dninjnks',
             'eos_mix_methods.RK_d3delta_dninjnks',
             

                 'regular_solution.regular_solution_Hi_sums',
                 'regular_solution.regular_solution_dGE_dxs',
                 'regular_solution.regular_solution_d2GE_dxixjs',
                 'regular_solution.regular_solution_d3GE_dxixjxks',
                 'regular_solution.regular_solution_gammas',
                 'regular_solution.RegularSolution',

                 'wilson.Wilson',
                 'wilson.wilson_xj_Lambda_ijs', 'wilson.wilson_d2GE_dTdxs',
                 'wilson.wilson_dGE_dxs', 'wilson.wilson_d2GE_dxixjs',
                 'wilson.wilson_d3GE_dxixjxks', 'wilson.wilson_gammas',

                 'uniquac.UNIQUAC',
                 'uniquac.uniquac_phis',
                 'uniquac.uniquac_dphis_dxs',
                 'uniquac.uniquac_thetaj_taus_jis',
                 'uniquac.uniquac_thetaj_taus_ijs',
                 'uniquac.uniquac_d2phis_dxixjs',
                 'uniquac.uniquac_dGE_dxs',
                 'uniquac.uniquac_d2GE_dTdxs',

                 'nrtl.NRTL',
                 'nrtl.nrtl_gammas',
                 'nrtl.nrtl_taus',
                 'nrtl.nrtl_dtaus_dT',
                 'nrtl.nrtl_d2taus_dT2',
                 'nrtl.nrtl_d3taus_dT3',
                 'nrtl.nrtl_alphas',
                 'nrtl.nrtl_Gs',
                 'nrtl.nrtl_dGs_dT',
                 'nrtl.nrtl_d2Gs_dT2',
                 'nrtl.nrtl_d3Gs_dT3',
                 'nrtl.nrtl_xj_Gs_jis_and_Gs_taus_jis',
                 'nrtl.nrtl_xj_Gs_jis',
                 'nrtl.nrtl_xj_Gs_taus_jis',
                 'nrtl.nrtl_dGE_dxs',
                 'nrtl.nrtl_d2GE_dxixjs',
                 'nrtl.nrtl_d2GE_dTdxs',

                'unifac.unifac_psis',
                'unifac.unifac_dpsis_dT',
                'unifac.unifac_d2psis_dT2',
                'unifac.unifac_d3psis_dT3',
                'unifac.unifac_Vis',
                'unifac.unifac_dVis_dxs',
                'unifac.unifac_d2Vis_dxixjs',
                'unifac.unifac_d3Vis_dxixjxks',
                'unifac.unifac_Xs',
                'unifac.unifac_Thetas',
                'unifac.unifac_dThetas_dxs',
                'unifac.unifac_d2Thetas_dxixjs',
                'unifac.unifac_VSXS',
                'unifac.unifac_Theta_Psi_sums',
                'unifac.unifac_ws',
                'unifac.unifac_Theta_pure_Psi_sums',
                'unifac.unifac_lnGammas_subgroups',
                'unifac.unifac_dlnGammas_subgroups_dxs',
                'unifac.unifac_d2lnGammas_subgroups_dTdxs',
                'unifac.unifac_d2lnGammas_subgroups_dxixjs',
                'unifac.unifac_dlnGammas_subgroups_dT',
                'unifac.unifac_d2lnGammas_subgroups_dT2',
                'unifac.unifac_d3lnGammas_subgroups_dT3',
                'unifac.unifac_Xs_pure',
                'unifac.unifac_Thetas_pure',
                'unifac.unifac_lnGammas_subgroups_pure',
                'unifac.unifac_dlnGammas_subgroups_pure_dT',
                'unifac.unifac_d2lnGammas_subgroups_pure_dT2',
                'unifac.unifac_d3lnGammas_subgroups_pure_dT3',
                'unifac.unifac_lngammas_r',
                'unifac.unifac_dlngammas_r_dxs',
                'unifac.unifac_d2lngammas_r_dxixjs',
                'unifac.unifac_dGE_dxs',
                'unifac.unifac_dGE_dxs_skip_comb',
                'unifac.unifac_d2GE_dTdxs',
                'unifac.unifac_d2GE_dTdxs_skip_comb',
                'unifac.unifac_d2GE_dxixjs',
                'unifac.unifac_d2GE_dxixjs_skip_comb',
                'unifac.unifac_gammas',
                'unifac.unifac_dgammas_dxs',
                'unifac.unifac_dgammas_dxs_skip_comb',
                'unifac.unifac_dgammas_dns',
                'unifac.unifac_lngammas_c',
                'unifac.unifac_dlngammas_c_dxs',
                'unifac.unifac_d2lngammas_c_dxixjs',
                'unifac.unifac_d3lngammas_c_dxixjxks',
                'unifac.UNIFAC',
                'unifac.unifac_gammas_at_T',

                 'activity.gibbs_excess_gammas', 'activity.gibbs_excess_dHE_dxs',
                 'activity.gibbs_excess_dgammas_dns', 'activity.gibbs_excess_dgammas_dT',
                 'activity.interaction_exp', 'activity.dinteraction_exp_dT',
                 'activity.d2interaction_exp_dT2', 'activity.d3interaction_exp_dT3',

                 'eos_alpha_functions.PR_a_alphas_vectorized',
                 'eos_alpha_functions.PR_a_alpha_and_derivatives_vectorized',
                 'eos_alpha_functions.SRK_a_alphas_vectorized',
                 'eos_alpha_functions.SRK_a_alpha_and_derivatives_vectorized',
                 'eos_alpha_functions.RK_a_alphas_vectorized',
                 'eos_alpha_functions.RK_a_alpha_and_derivatives_vectorized',
                 'eos_alpha_functions.PRSV_a_alphas_vectorized',
                 'eos_alpha_functions.PRSV_a_alpha_and_derivatives_vectorized',
                 'eos_alpha_functions.PRSV2_a_alphas_vectorized',
                 'eos_alpha_functions.PRSV2_a_alpha_and_derivatives_vectorized',
                 'eos_alpha_functions.APISRK_a_alphas_vectorized',
                 'eos_alpha_functions.APISRK_a_alpha_and_derivatives_vectorized',

                 'phases.iapws_phase.IAPWS95', 'phases.iapws_phase.IAPWS95Liquid', 'phases.iapws_phase.IAPWS95Gas',
                 'phases.air_phase.DryAirLemmon',

             'phases.phase_utils.lnphis_direct',
             'flash.flash_utils.sequential_substitution_2P_functional',
             
             'fitting.data_fit_statistics',

                 ]
    normal_fluids.numba.transform_lists_to_arrays(normal, to_change, __funcs, cache_blacklist=cache_blacklist)

    __funcs['FlashPureVLS'] = thermo.FlashPureVLS
    __funcs['FlashVL'] = thermo.FlashVL
    __funcs['FlashVLN'] = thermo.FlashVLN

    for mod in new_mods:
        mod.__dict__.update(__funcs)
        try:
            __all__.extend(mod.__all__)
        except AttributeError:
            pass


    __funcs['eos'].GCEOS.volume_solutions = staticmethod(__funcs['volume_solutions_halley'])
    __funcs['eos'].GCEOS.main_derivatives_and_departures = staticmethod(__funcs['main_derivatives_and_departures'])
    __funcs['eos_mix'].GCEOSMIX.volume_solutions = staticmethod(__funcs['volume_solutions_halley'])
    __funcs['eos_mix'].GCEOSMIX.main_derivatives_and_departures = staticmethod(__funcs['main_derivatives_and_departures'])
    
    __funcs['eos_mix'].IGMIX.volume_solutions = staticmethod(__funcs['volume_solutions_ideal'])
transform_complete_thermo(replaced, __funcs, __all__, normal, vec=False)

'''Before jitclasses could be used on Activity models, numba would have to add:
Support type call.
Support class methods.
Support class constants.

This is not likely to happen.

IdealSolution_spec = [('T', float64), ('N', int64), ('xs', float64[:]), ('scalar', boolean)]

IdealSolutionNumba = jitclass(IdealSolution_spec)(thermo.numba.activity.IdealSolution)

# activity.IdealSolution
IdealSolutionNumba(T=300.0, xs=np.array([.2, .5]))
'''


globals().update(__funcs)
globals().update(replaced)

__name__ = 'thermo.numba'
__file__ = orig_file

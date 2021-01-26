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
    cache_blacklist = set([])
    __funcs.update(normal_fluids.numba.numbafied_fluids_functions.copy())
    new_mods = normal_fluids.numba.transform_module(normal, __funcs, replaced, vec=vec,
                                                    cache_blacklist=cache_blacklist)
    if vec:
        conv_fun = numba.vectorize
    else:
        conv_fun = numba.jit

    import chemicals.numba
    for name in dir(chemicals.numba):
        obj = getattr(chemicals.numba, name)
        if isinstance(obj, CPUDispatcher):
            __funcs[name] = obj

    for mod in new_mods:
        mod.__dict__.update(__funcs)

    to_change = ['eos.volume_solutions_halley', 'eos_mix.a_alpha_quadratic_terms',
                 'eos_mix_methods.a_alpha_and_derivatives_quadratic_terms',
                 'eos_mix_methods.PR_lnphis', 'eos_mix_methods.PR_lnphis_fastest',
                 'eos_mix_methods.a_alpha_aijs_composition_independent',
                 'eos_mix_methods.a_alpha_and_derivatives_full',

                 'regular_solution.regular_solution_Hi_sums',
                 'regular_solution.regular_solution_dGE_dxs',
                 'regular_solution.regular_solution_d2GE_dxixjs',
                 'regular_solution.regular_solution_d3GE_dxixjxks',
                 'regular_solution.RegularSolution',

                 'wilson.Wilson',
                 'wilson.wilson_xj_Lambda_ijs', 'wilson.wilson_d2GE_dTdxs',
                 'wilson.wilson_dGE_dxs', 'wilson.wilson_d2GE_dxixjs',
                 'wilson.wilson_d3GE_dxixjxks', 'wilson.wilson_gammas',

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
                 'eos_alpha_functions.APISRK_a_alpha_and_derivatives_vectorized']
    normal_fluids.numba.transform_lists_to_arrays(normal, to_change, __funcs, cache_blacklist=cache_blacklist)



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

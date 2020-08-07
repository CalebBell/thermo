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
        
    to_change = ['eos.volume_solutions_halley', 'eos_mix.a_alpha_quadratic_terms',
                 'eos_mix.a_alpha_and_derivatives_quadratic_terms',
                 'eos_mix.PR_lnphis', 'eos_mix.PR_lnphis_direct',
                 'alpha_functions.PR_a_alphas_vectorized', 
                 'alpha_functions.PR_a_alpha_and_derivatives_vectorized']
    normal_fluids.numba.transform_lists_to_arrays(normal, to_change, __funcs, cache_blacklist=cache_blacklist)

    for mod in new_mods:
        mod.__dict__.update(__funcs)
        
    __funcs['eos'].GCEOS.volume_solutions = staticmethod(__funcs['volume_solutions_halley'])
    __funcs['eos'].GCEOS.main_derivatives_and_departures = staticmethod(__funcs['main_derivatives_and_departures'])


transform_complete_thermo(replaced, __funcs, __all__, normal, vec=False)



globals().update(__funcs)
globals().update(replaced)

__name__ = 'thermo.numba'
__file__ = orig_file

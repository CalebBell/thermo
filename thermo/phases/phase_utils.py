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

__all__ = [
    'activity_pointer_reference_dicts',
    'activity_reference_pointer_dicts',
    'object_lookups',
    'lnphis_direct',
    'fugacities_direct',
]

from fluids.numerics import log, trunc_exp

from thermo.activity import IdealSolution
from thermo.eos import eos_full_path_dict
from thermo.eos_mix import eos_mix_full_path_dict
from thermo.eos_mix_methods import (
    PR_lnphis_fastest,
    PR_translated_lnphis_fastest,
    RK_lnphis_fastest,
    SRK_lnphis_fastest,
    SRK_translated_lnphis_fastest,
    VDW_lnphis_fastest,
)
from thermo.nrtl import NRTL, nrtl_gammas_from_args
from thermo.regular_solution import RegularSolution, Hansen, FloryHuggins, regular_solution_gammas
from thermo.serialize import object_lookups
from thermo.unifac import UNIFAC, unifac_gammas_from_args
from thermo.uniquac import UNIQUAC, uniquac_gammas_from_args
from thermo.wilson import Wilson, wilson_gammas_from_args

activity_pointer_reference_dicts = {
    'thermo.activity.IdealSolution': IdealSolution,
    'thermo.wilson.Wilson': Wilson,
    'thermo.unifac.UNIFAC': UNIFAC,
    'thermo.regular_solution.RegularSolution': RegularSolution,
    'thermo.regular_solution.FloryHuggins': FloryHuggins,
    'thermo.regular_solution.Hansen': Hansen,
    'thermo.uniquac.UNIQUAC': UNIQUAC,
    'thermo.nrtl.NRTL': NRTL,
}
activity_reference_pointer_dicts = {
    v: k for k, v in activity_pointer_reference_dicts.items()
}
object_lookups.update(activity_pointer_reference_dicts)
object_lookups.update(eos_mix_full_path_dict)
object_lookups.update(eos_full_path_dict)

def activity_lnphis(zs, model, T, P, N, lnPsats, Poyntings, phis_sat, *activity_args):
    # It appears that to make numba happy *activity_args will not work
    # and all functions on this level will need to have a fixed number of arguments
    activity_args, lnphis = tuple(activity_args[0:-1]), activity_args[-1]
    if 20000 <= model <= 20099:
        gammas = [1.0]*N
    elif 20100 <= model <= 20199:
        gammas = nrtl_gammas_from_args(zs, *activity_args)
    elif 20200 <= model <= 20299:
        gammas = wilson_gammas_from_args(zs, *activity_args)
    elif 20300 <= model <= 20399:
        gammas = uniquac_gammas_from_args(zs, *activity_args)
    elif 20400 <= model <= 20499:
        gammas = regular_solution_gammas(zs, *activity_args)
    elif 20500 <= model <= 20599:
        gammas = unifac_gammas_from_args(zs, *activity_args)
    else:
        raise ValueError("Model not implemented")
    # lnphis = [0.0]*N
    P_inv = 1.0/P
    for i in range(N):
        lnphis[i] = log(gammas[i]*Poyntings[i]*phis_sat[i]*P_inv) + lnPsats[i]
    return lnphis


def lnphis_direct(zs, model, T, P, N, *args):
    if model == 0:
        lnphis = args[-1]
        for i in range(N):
            lnphis[i] = 0.0
        return lnphis
    elif model in (10200, 10201, 10204, 10205, 10206):
        return PR_lnphis_fastest(zs, T, P, N, *args)
    elif model in (10100, 10104, 10105):
        return SRK_lnphis_fastest(zs, T, P, N, *args)
    elif model == 10002:
        return RK_lnphis_fastest(zs, T, P, N, *args)
    elif model == 10001:
        return VDW_lnphis_fastest(zs, T, P, N, *args)
    elif model in (11202, 11203, 11207):
        return PR_translated_lnphis_fastest(zs, T, P, N, *args)
    elif model in (11101, 11102, 11103):
        return SRK_translated_lnphis_fastest(zs, T, P, N, *args)
    # so far only used by the test test_UNIFAC_lnphis_direct
    elif 20000 <= model <= 29999: # numba: delete
        return activity_lnphis(zs, model, T, P, N, *args) # numba: delete
    raise ValueError("Model not implemented")

def fugacities_direct(zs, model, T, P, N, *args):
    # Obtain fugacities directly.
    lnphis = lnphis_direct(zs, model, T, P, N, *args)
    for i in range(N):
        lnphis[i] = P*zs[i]*trunc_exp(lnphis[i])
    return lnphis


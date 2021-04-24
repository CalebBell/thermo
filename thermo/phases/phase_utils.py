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

__all__ = [
    'activity_pointer_reference_dicts',
    'activity_reference_pointer_dicts',
    'object_lookups',
    'lnphis_direct',
]

from thermo.eos import eos_full_path_dict
from thermo.eos_mix import eos_mix_full_path_dict
from thermo.eos_mix_methods import (PR_lnphis_fastest, PR_translated_lnphis_fastest,
                                    SRK_lnphis_fastest, SRK_translated_lnphis_fastest, 
                                    RK_lnphis_fastest,  VDW_lnphis_fastest)
from thermo.activity import IdealSolution
from thermo.wilson import Wilson
from thermo.unifac import UNIFAC
from thermo.regular_solution import RegularSolution
from thermo.uniquac import UNIQUAC

activity_pointer_reference_dicts = {
    'thermo.activity.IdealSolution': IdealSolution,
    'thermo.wilson.Wilson': Wilson,
    'thermo.unifac.UNIFAC': UNIFAC,
    'thermo.regular_solution.RegularSolution': RegularSolution,
    'thermo.uniquac.UNIQUAC': UNIQUAC,
}
activity_reference_pointer_dicts = {
    v: k for k, v in activity_pointer_reference_dicts.items()
}
object_lookups = {
    **activity_pointer_reference_dicts,
    **eos_mix_full_path_dict,
    **eos_full_path_dict
}


def lnphis_direct(zs, model, T, P, N, *args):
    if model == 10200 or model == 10201 or model == 10204 or model == 10205 or model == 10206:
        return PR_lnphis_fastest(zs, T, P, N, *args)
    elif model == 10202 or model == 10203 or model == 10207:
        return PR_translated_lnphis_fastest(zs, T, P, N, *args)
    elif model == 10100 or model == 10104 or model == 10105:
        return SRK_lnphis_fastest(zs, T, P, N, *args)
    elif model == 10101 or model == 10102 or model == 10103:
        return SRK_translated_lnphis_fastest(zs, T, P, N, *args)
    elif model == 10002:
        return RK_lnphis_fastest(zs, T, P, N, *args)
    elif model == 10001:
        return VDW_lnphis_fastest(zs, T, P, N, *args)
    elif model == 0:
        lnphis = args[-1]
        for i in range(N):
            lnphis[i] = 0.0
        return lnphis
    raise ValueError("Model not implemented")

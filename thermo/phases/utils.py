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
    'SORTED_DICT',
    'activity_pointer_reference_dicts',
    'activity_reference_pointer_dicts',
    'object_lookups',
]

import sys
from thermo.eos import eos_full_path_dict
from thermo.eos_mix import eos_mix_full_path_dict
from thermo.activity import IdealSolution
from thermo.wilson import Wilson
from thermo.unifac import UNIFAC
from thermo.regular_solution import RegularSolution
from thermo.uniquac import UNIQUAC

SORTED_DICT = sys.version_info >= (3, 6)
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
object_lookups = activity_pointer_reference_dicts.copy()
object_lookups.update(eos_mix_full_path_dict)
object_lookups.update(eos_full_path_dict)


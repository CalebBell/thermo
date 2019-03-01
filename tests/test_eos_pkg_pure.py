# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from numpy.testing import assert_allclose
import pytest

from thermo import *
from fluids.numerics import *
from math import *
import json
import os

from thermo.test_utils import *

pure_cases = []

pure_case_compounds = ['water', 'methane', 'ethane', 'ammonia', 'nitrogen', 'oxygen']
pure_case_packages = [SRK_PKG, PR_PKG]
functions = ['TVF', 'PVF', 'TP', 'PH']
functions = ['PH']

for pkg in pure_case_packages:
    for compound in pure_case_compounds:
        case = {'IDs': [compound], 'pkg_ID': pkg, 'zs': [1.0], 'types': functions}
        pure_cases.append(case)

pure_cases_custom = []
pure_cases.extend(pure_cases_custom)

#pure_cases = [{'IDs': ['water'], 'pkg_ID': PR_PKG, 'zs': [1.0]}]
pure_path = os.path.join(os.path.dirname(__file__), '../surfaces/TP tabular data pure')


def export_all_pure_cases(verbose=False):
    for pure_case in pure_cases:
        for func_str in pure_case['types']:
            tabular_data_function = tabular_data_functions[func_str]
            kwargs = dict(pure_case)
            del kwargs['types']
            specs0, specs1, data, metadata = tabular_data_function(**kwargs)
            case_name = '%s %s %s %s' %(pure_case['IDs'], pure_case['zs'], pure_case['pkg_ID'], func_str)
            file_name = os.path.join(pure_path, case_name)
            if 'attrs' in pure_cases:
                attrs = pure_cases['attrs']
            else:
                attrs = default_attrs
            save_tabular_data_as_json(specs0, specs1, metadata, data, attrs, file_name)
            if verbose:
                print('Finished %s' %(case_name))
        
    

if __name__ == '__main__':
    export_all_pure_cases(verbose=True)
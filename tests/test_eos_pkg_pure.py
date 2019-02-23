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


default_attrs = ('phase', 'Hm', 'Sm', 'Gm', 'xs', 'ys', 'V_over_F', 'T', 'P')


def pkg_tabular_data_TP(IDs, pkg_ID, zs, P_pts=50, T_pts=50, T_min=None,
                        T_max=None, P_min=None, P_max=None, attrs=default_attrs):
    pkg = PropertyPackageConstants(IDs, pkg_ID)
    
    if T_min is None:
        T_min = min(i for i in pkg.Tms if i is not None)
    if T_max is None:
        T_max = max(i for i in pkg.Tcs if i is not None)*2
    
    if P_min is None:
        P_min = 100
    if P_max is None:
        P_max = max(i for i in pkg.Pcs if i is not None)*2

    Ts = linspace(T_min, T_max, T_pts)
    Ps = logspace(log10(P_min), log10(P_max), P_pts)
    data = []
    for T in Ts:
        data_row = []
        for P in Ps:
            pkg.pkg.flash(T=T, P=P, zs=zs)
            row = tuple(getattr(pkg.pkg, s) for s in attrs)
            data_row.append(row)
        data.append(data_row)
    
    metadata = {'spec':'T,P', 'pkg': pkg_ID, 'zs': zs, 'IDs': IDs,
                'T_min': T_min, 'T_max': T_max, 'P_min': P_min, 'P_max': P_max,
                'Ts': Ts, 'Ps': Ps}
    
    data_json = {}
    for row, T in zip(data, Ts):
        for i, P in enumerate(Ps):
            data_json[(T, P)] = {k: v for k, v in zip(attrs, row[i])}
    
    json_result = {'metadata': metadata, 'data': data_json}
    return Ts, Ps, data, metadata


def save_tabular_data_as_json(specs0, specs1, metadata, data, attrs, path):
    r'''Saves tabular data from a property package to a json file for testing,
    plotting, and storage.

    Parameters
    ----------
    specs0 : float
        List of values of the first spec; i.e. if the first spec was T,
        a list of temperatures
    specs1 : float
        List of values of the first spec; i.e. if the second spec was P,
        a list of pressures
    metadata : dict
        A dictionary of json-serializeable metadata from the property
        package; the specifications, range of values, compositions, and
        compounds are good values to include
    data : list[list[tuple]]
        A list of lists of tuples containing the data values specified in
        `attrs`
    attrs : tuple[str]
        The attributes stored from the property package
    path : str
        The path for the json-formatted data to be stored

    Notes
    -----

    Examples
    --------

    '''
    data_json = []
    
    for row, spec0 in zip(data, specs0):
        new_rows = []
        
        for i, spec1 in enumerate(specs1):
            new_rows.append({k: v for k, v in zip(attrs, row[i])})
        data_json.append(new_rows)
    
    metadata['attrs'] = attrs
    json_result = {'metadata': metadata, 'tabular_data': data_json}
    
    fp = open(path, 'w')
    json.dump(json_result, fp, sort_keys=True, indent=2)
    fp.close()
    return True

pure_cases = []

pure_case_compounds = ['water', 'methane', 'ethane', 'ammonia', 'nitrogen', 'oxygen']
pure_case_packages = [SRK_PKG, PR_PKG]

for pkg in pure_case_packages:
    for compound in pure_case_compounds:
        case = {'IDs': [compound], 'pkg_ID': pkg, 'zs': [1.0]}
        pure_cases.append(case)

pure_cases_custom = []
pure_cases.extend(pure_cases_custom)

#pure_cases = [{'IDs': ['water'], 'pkg_ID': PR_PKG, 'zs': [1.0]}]
pure_path = os.path.join(os.path.dirname(__file__), '../surfaces/TP tabular data pure')



def export_all_pure_cases(verbose=False):
    for pure_case in pure_cases:
        Ts, Ps, data, metadata = pkg_tabular_data_TP(**pure_case)
        case_name = '%s %s %s' %(pure_case['IDs'], pure_case['zs'], pure_case['pkg_ID'])
        file_name = os.path.join(pure_path, case_name)
        if 'attrs' in pure_cases:
            attrs = pure_cases['attrs']
        else:
            attrs = default_attrs
        save_tabular_data_as_json(Ts, Ps, metadata, data, attrs, file_name)
        if verbose:
            print('Finished %s' %(case_name))
    


if __name__ == '__main__':
    export_all_pure_cases()
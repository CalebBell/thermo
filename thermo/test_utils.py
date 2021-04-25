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
import numpy as np

from thermo import *
from fluids.numerics import *
from math import *
import json
import os
from fluids.numerics import assert_close, assert_close1d, assert_close2d, assert_close3d

def check_np_output_activity(model, modelnp, modelnp2):
    # model is flat, scalar, list-based model
    # modelnp is numba model
    # modelnp2 is created from the numba model with to_T_xs at a different composition

    scalar_attrs = ['d3GE_dT3', 'd2GE_dT2', 'GE', 'dGE_dT']
    for attr in scalar_attrs:
        if hasattr(model, attr):
#            print(attr)
            assert_close(getattr(model, attr)(), getattr(modelnp, attr)(), rtol=2e-13)
            assert_close(getattr(modelnp2, attr)(), getattr(modelnp, attr)(), rtol=2e-13)
            assert type(getattr(model, attr)()) is float
    #        assert type(getattr(modelnp, attr)()) is float
    #        assert type(getattr(modelnp2, attr)()) is float

    vec_attrs = ['dGE_dxs', 'gammas', '_gammas_dGE_dxs',
                 'd2GE_dTdxs', 'dHE_dxs', 'gammas_infinite_dilution', 'dHE_dns',
                'dnHE_dns', 'dSE_dxs', 'dSE_dns', 'dnSE_dns', 'dGE_dns', 'dnGE_dns', 'd2GE_dTdns',
                'd2nGE_dTdns', 'dgammas_dT']

    for attr in vec_attrs:
#        print(attr)
        assert_close1d(getattr(model, attr)(), getattr(modelnp, attr)(), rtol=2e-13)
        assert_close1d(getattr(modelnp2, attr)(), getattr(modelnp, attr)(), rtol=2e-13)
        assert type(getattr(model, attr)()) is list
        assert type(getattr(modelnp, attr)()) is np.ndarray
        assert type(getattr(modelnp2, attr)()) is np.ndarray

    mat_attrs = ['d2GE_dxixjs', 'd2nGE_dninjs', 'dgammas_dns']
    for attr in mat_attrs:
#        print(attr)
        assert_close2d(getattr(model, attr)(), getattr(modelnp, attr)(), rtol=1e-12)
        assert_close2d(getattr(modelnp2, attr)(), getattr(modelnp, attr)(), rtol=1e-12)
        assert type(getattr(model, attr)()) is list
        assert type(getattr(modelnp, attr)()) is np.ndarray
        assert type(getattr(modelnp2, attr)()) is np.ndarray

    attrs_3d = ['d3GE_dxixjxks']
    for attr in attrs_3d:
        if hasattr(model, attr):
#            print(attr)
            # some models do not have this implemented
            assert_close3d(getattr(model, attr)(), getattr(modelnp, attr)(), rtol=1e-13)
            assert_close3d(getattr(modelnp2, attr)(), getattr(modelnp, attr)(), rtol=1e-13)
            assert type(getattr(model, attr)()) is list
            assert type(getattr(modelnp, attr)()) is np.ndarray
            assert type(getattr(modelnp2, attr)()) is np.ndarray


def plot_unsupported(reason, color='r'):
    '''Helper function - draw a plot with an `x` over it displaying a message
    why that plot is not supported.
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.plot([0, 1], [0, 1], lw=5, c=color)
    ax.plot([0, 1], [1, 0], lw=5, c=color)

    ax.text(.5, .5, reason, ha='center', va='center', bbox=dict(fc='white'))
    return fig



def mark_plot_unsupported(plot_fig, reason, color='r'):
    ax = plot_fig.axes[0]
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    xmid = 10**(0.5*(log10(xlims[0]) + log10(xlims[1])))
    ymid = 10**(0.5*(log10(ylims[0]) + log10(ylims[1])))
    ax.text(xmid, ymid, reason, ha='center', va='center', bbox=dict(fc='white'))
    color = 'r'
    ax.plot(xlims, ylims, lw=5, c=color)
    ax.plot(xlims, ylims[::-1], lw=5, c=color)





default_attrs = ('phase', 'Hm', 'Sm', 'Gm', 'xs', 'ys', 'V_over_F', 'T', 'P')

def flash_rounding(x):
    if isinstance(x, float):
        return float('%.10e' %(x))
    return x

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
            try:
                pkg.pkg.flash(T=T, P=P, zs=zs)
                row = tuple(flash_rounding(getattr(pkg.pkg, s)) for s in attrs)
            except Exception as e:
                row = tuple(None for s in attrs)
                print(e, T, P, IDs, pkg_ID, zs)
            data_row.append(row)
        data.append(data_row)

    metadata = {'spec':('T', 'P'), 'pkg': pkg_ID, 'zs': zs, 'IDs': IDs,
                'T_min': T_min, 'T_max': T_max, 'P_min': P_min, 'P_max': P_max,
                'Ts': Ts, 'Ps': Ps}

    return Ts, Ps, data, metadata


def pkg_tabular_data_TVF(IDs, pkg_ID, zs, VF_pts=50, T_pts=50, T_min=None,
                         T_max=None, VF_min=None, VF_max=None, attrs=default_attrs):
    pkg = PropertyPackageConstants(IDs, pkg_ID)

    if T_min is None:
        T_min = min(i for i in pkg.Tms if i is not None)
    if T_max is None:
        if pkg.pkg.N == 1:
            T_max = pkg.Tcs[0]
        else:
            T_max = max(i for i in pkg.Tcs if i is not None)*2

    if VF_min is None:
        VF_min = 0.0
    if VF_max is None:
        VF_max = 1.0

    Ts = linspace(T_min, T_max, T_pts)
    VFs = linspace(VF_min, VF_max, VF_pts)
    data = []
    for T in Ts:
        data_row = []
        for VF in VFs:
#            print(T, VF)
            try:
                pkg.pkg.flash(T=T, VF=VF, zs=zs)
                row = tuple(flash_rounding(getattr(pkg.pkg, s)) for s in attrs)
            except Exception as e:
                row = tuple(None for s in attrs)
                print(e, T, VF, IDs, pkg_ID, zs)
            data_row.append(row)
        data.append(data_row)

    metadata = {'spec':('T', 'VF'), 'pkg': pkg_ID, 'zs': zs, 'IDs': IDs,
                'T_min': T_min, 'T_max': T_max, 'VF_min': VF_min, 'VF_max': VF_max,
                'Ts': Ts, 'VFs': VFs}


    return Ts, VFs, data, metadata

def pkg_tabular_data_PVF(IDs, pkg_ID, zs, VF_pts=50, P_pts=50, P_min=None,
                         P_max=None, VF_min=None, VF_max=None, attrs=default_attrs):
    pkg = PropertyPackageConstants(IDs, pkg_ID)

    if P_min is None:
        P_min = 100.0
    if P_max is None:
        if pkg.pkg.N == 1:
            P_max = pkg.Pcs[0]
        else:
            P_max = max(i for i in pkg.Pcs if i is not None)*2

    if VF_min is None:
        VF_min = 0.0
    if VF_max is None:
        VF_max = 1.0

    Ps = linspace(P_min, P_max, P_pts)
    VFs = linspace(VF_min, VF_max, VF_pts)
    data = []
    for P in Ps:
        data_row = []
        for VF in VFs:
#            print(P, VF)
            try:
                pkg.pkg.flash(P=P, VF=VF, zs=zs)
                row = tuple(flash_rounding(getattr(pkg.pkg, s)) for s in attrs)
            except Exception as e:
                row = tuple(None for s in attrs)
                print(e, P, VF, IDs, pkg_ID, zs)
            data_row.append(row)
        data.append(data_row)

    metadata = {'spec':('P', 'VF'), 'pkg': pkg_ID, 'zs': zs, 'IDs': IDs,
                'P_min': P_min, 'P_max': P_max, 'VF_min': VF_min, 'VF_max': VF_max,
                'Ps': Ps, 'VFs': VFs}


    return Ps, VFs, data, metadata


def pkg_tabular_data_PH(IDs, pkg_ID, zs, P_pts=50, H_pts=50, H_min=None,
                        H_max=None, P_min=None, P_max=None, attrs=default_attrs):
    pkg = PropertyPackageConstants(IDs, pkg_ID).pkg


    if P_min is None:
        P_min = 100
    if P_max is None:
        P_max = max(i for i in pkg.Pcs if i is not None)*2

    if H_min is None or H_max is None:
        T_min = min(i for i in pkg.Tms if i is not None)
        T_max = max(i for i in pkg.Tcs if i is not None)*2

        Hm_range = []
        for P in (P_min, P_max):
            for T in (T_min, T_max):
                pkg.flash(T=T, P=P, zs=zs)
                Hm_range.append(pkg.Hm)

        if H_min is None:
            H_min = min(Hm_range)
        if H_max is None:
            H_max = max(Hm_range)

    Hs = linspace(H_min, H_max, H_pts)
    Ps = logspace(log10(P_min), log10(P_max), P_pts)
    data = []
    for H in Hs:
        data_row = []
        for P in Ps:
#            print(P, H, IDs, zs)
            pkg.flash_caloric(Hm=H, P=P, zs=zs)
            row = tuple(flash_rounding(getattr(pkg, s)) for s in attrs)
            data_row.append(row)
        data.append(data_row)

    metadata = {'spec':('P', 'H'), 'pkg': pkg_ID, 'zs': zs, 'IDs': IDs,
                'H_min': H_min, 'H_max': H_max, 'P_min': P_min, 'P_max': P_max,
                'Hs': Hs, 'Ps': Ps}

    return Hs, Ps, data, metadata


def pkg_tabular_data_PS(IDs, pkg_ID, zs, P_pts=50, S_pts=50, S_min=None,
                        S_max=None, P_min=None, P_max=None, attrs=default_attrs):
    pkg = PropertyPackageConstants(IDs, pkg_ID).pkg


    if P_min is None:
        P_min = 100
    if P_max is None:
        P_max = max(i for i in pkg.Pcs if i is not None)*2

    if S_min is None or S_max is None:
        T_min = min(i for i in pkg.Tms if i is not None)
        T_max = max(i for i in pkg.Tcs if i is not None)*2

        Sm_range = []
        for P in (P_min, P_max):
            for T in (T_min, T_max):
                pkg.flash(T=T, P=P, zs=zs)
                Sm_range.append(pkg.Sm)

        if S_min is None:
            S_min = min(Sm_range)
        if S_max is None:
            S_max = max(Sm_range)

    Ss = linspace(S_min, S_max, S_pts)
    Ps = logspace(log10(P_min), log10(P_max), P_pts)
    data = []
    for S in Ss:
        data_row = []
        for P in Ps:
#            print(P, S, IDs, zs)
            pkg.flash_caloric(Sm=S, P=P, zs=zs)
            row = tuple(flash_rounding(getattr(pkg, s)) for s in attrs)
            data_row.append(row)
        data.append(data_row)

    metadata = {'spec':('P', 'S'), 'pkg': pkg_ID, 'zs': zs, 'IDs': IDs,
                'S_min': S_min, 'S_max': S_max, 'P_min': P_min, 'P_max': P_max,
                'Ss': Ss, 'Ps': Ps}

    return Ss, Ps, data, metadata


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

tabular_data_functions = {'TP': pkg_tabular_data_TP,
                          'TVF': pkg_tabular_data_TVF,
                          'PVF': pkg_tabular_data_PVF,
                          'PH': pkg_tabular_data_PH,
                          'PS': pkg_tabular_data_PS}



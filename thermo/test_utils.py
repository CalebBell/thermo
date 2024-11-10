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
SOFTWARE.
'''

from math import log10

import numpy as np
from fluids.numerics import assert_close, assert_close1d, assert_close2d, assert_close3d
from thermo.activity import GibbsExcess

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

    vec_attrs = ['dGE_dxs', 'gammas', 'gammas_dGE_dxs',
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
        if model.__class__.d2GE_dxixjs is GibbsExcess.d2GE_dxixjs_numerical:
            # no point in checking numerical derivatives of second order, too imprecise
            continue
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



def flash_rounding(x):
    if isinstance(x, float):
        return float(f'{x:.10e}')
    return x

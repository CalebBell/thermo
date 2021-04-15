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

import pytest
import thermo
from thermo import *
from thermo.coolprop import *
from thermo.phases import IAPWS95Gas, IAPWS95Liquid
from thermo.chemical_package import iapws_correlations
from fluids.numerics import *
from math import *
import json
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass

fluid = 'water'
pure_surfaces_dir = os.path.join(thermo.thermo_dir, '..', 'surfaces', 'iapws95')


def test_iapws95_basic_flash():
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])
    PT = flasher.flash(T=300, P=1e6)

    assert_close(PT.rho_mass(), 996.9600226949985, rtol=1e-10)
    assert isinstance(PT.liquid0, IAPWS95Liquid)

    TV = flasher.flash(T=300, V=PT.V())
    assert_close(TV.P, PT.P, rtol=1e-10)
    assert isinstance(TV.liquid0, IAPWS95Liquid)

    PV = flasher.flash(P=PT.P, V=PT.V())
    assert_close(PV.T, PT.T, rtol=1e-13)
    assert isinstance(PV.liquid0, IAPWS95Liquid)

    TVF = flasher.flash(T=400.0, VF=1)
    assert_close(TVF.P, 245769.3455657166, rtol=1e-13)

    PVF = flasher.flash(P=TVF.P, VF=1)
    assert_close(PVF.T, 400.0, rtol=1e-13)

    assert_close(flasher.flash(P=1e5, H=flasher.flash(T=273.15, P=1e5).H()).T, 273.15, rtol=1e-10)

def test_iapws95_basic_flashes_no_hacks():
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])
    flasher.VL_only_IAPWS95 = False

    PT = flasher.flash(T=300, P=1e6)

    assert_close(PT.rho_mass(), 996.9600226949985, rtol=1e-10)

    TV = flasher.flash(T=300, V=PT.V())
    assert_close(TV.P, PT.P, rtol=1e-10)

    PV = flasher.flash(P=PT.P, V=PT.V())
    assert_close(PV.T, PT.T, rtol=1e-13)

    PH = flasher.flash(H=PT.H(), P=1e6)
    assert_close(PH.T, 300)

    PS = flasher.flash(S=PT.S(), P=1e6)
    assert_close(PS.T, 300)

    assert_close(flasher.flash(P=1e5, H=flasher.flash(T=273.15, P=1e5).H()).T, 273.15, rtol=1e-10)

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
def test_TV_plot_iapws95():
    eos = IAPWS95
    T, P = 298.15, 101325.0
    zs = [1.0]
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])


    res = flasher.TPV_inputs(zs=zs, pts=200, spec0='T', spec1='P', check0='T', check1='V', prop0='P',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, "TV")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s' %('TV', eos.__name__, fluid)
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(np.abs(errs))
    # CoolProp has same error characteritic
    assert max_err < 1e-5

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
def test_PV_plot_iapws95():
    eos = IAPWS95
    T, P = 298.15, 101325.0
    zs = [1.0]
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])


    res = flasher.TPV_inputs(zs=zs, pts=100, spec0='T', spec1='P', check0='P', check1='V', prop0='T',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
#                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, "PV")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s' %('PV', eos.__name__, fluid)

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()
    max_err = np.max(np.abs(errs))
    limit = 5e-11
#    assert max_err < limit
@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
def test_PS_plot():
    '''
    '''
    eos = IAPWS95
    path = os.path.join(pure_surfaces_dir, fluid, "PS")
    if not os.path.exists(path):
        os.makedirs(path)
    key = '%s - %s - %s' %('PS', eos.__name__, fluid)


    T, P = 298.15, 101325.0
    zs = [1.0]
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])

    res = flasher.TPV_inputs(zs=zs, pts=200, spec0='T', spec1='P', check0='P', check1='S', prop0='T',
                           trunc_err_low=1e-15,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(errs)
    assert max_err < 1e-8
#
#test_PS_plot()

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
def test_PH_plot():
    '''
    '''
    eos = IAPWS95
    path = os.path.join(pure_surfaces_dir, fluid, "PH")
    if not os.path.exists(path):
        os.makedirs(path)
    key = '%s - %s - %s' %('PH', eos.__name__, fluid)


    T, P = 298.15, 101325.0
    zs = [1.0]
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])

    res = flasher.TPV_inputs(zs=zs, pts=250, spec0='T', spec1='P', check0='P', check1='H', prop0='T',
                           trunc_err_low=1e-15,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(errs)
    assert max_err < 1e-8

#test_PH_plot()

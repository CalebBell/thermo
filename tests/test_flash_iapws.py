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
from thermo.test_utils import mark_plot_unsupported
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
    
    # Case where the density solution was failing
    T = 250.49894958453197
    P = 10595.601792776019
    base = flasher.flash(T=T, P=P)
    new = flasher.flash(P=P, S=base.S())
    assert_close(new.T, base.T)


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("variables", ['VPT', 'VTP',
                                       'PHT', 'PST', 'PUT',
                                       'VUT', 'VST', 'VHT',
                                          'TSV',
                                            'THP', 'TUP',
                                       ])
def test_plot_IAPWS95(variables):
    spec0, spec1, check_prop = variables
    plot_name = variables[0:2]
    eos = IAPWS95
    T, P = 298.15, 101325.0
    zs = [1.0]
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])

    flasher = FlashPureVLS(constants=iapws_constants, correlations=iapws_correlations,
                       gas=gas, liquids=[], solids=[])
    flasher.TPV_HSGUA_xtol = 1e-13
    
    flash_spec = frozenset([spec0, spec1])
    inconsistent = flash_spec in (frozenset(['T', 'H']), frozenset(['T', 'U']),
                                  frozenset(['T', 'S']), # blip issues
                                  frozenset(['V', 'P']), # 4 degree water blip
                                  frozenset(['V', 'S']), frozenset(['V', 'H']),
                                  frozenset(['V', 'U']), # Fun
                                  )

    res = flasher.TPV_inputs(zs=[1.0], pts=200, spec0='T', spec1='P', 
                             check0=spec0, check1=spec1, prop0=check_prop,
                           trunc_err_low=1e-13,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           show=False, verbose=not inconsistent)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, plot_name)
    if not os.path.exists(path):
        os.makedirs(path)
        
    tol = 5e-12

    key = '%s - %s - %s' %(plot_name, eos.__name__, fluid)

    if inconsistent:
        spec_name = spec0 + spec1
        mark_plot_unsupported(plot_fig, reason='EOS is inconsistent for %s inputs' %(spec_name))
        tol = 1e300
    if flash_spec == frozenset(['T', 'V']):
        tol = 1e-5

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(np.abs(errs))
    assert max_err < tol

# test_plot_IAPWS95('VUT')
# test_plot_IAPWS95('PUT')
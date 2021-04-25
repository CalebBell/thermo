# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2021, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from thermo.phases import DryAirLemmon
from thermo.chemical_package import lemmon2000_constants, lemmon2000_correlations
from fluids.numerics import *
from math import *
import json
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass

fluid = 'air'
pure_surfaces_dir = os.path.join(thermo.thermo_dir, '..', 'surfaces', 'lemmon2000')

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("variables", [
                                       'VPT',
                                       'VTP', 
                                       'PHT', 'PST', 'PUT',
                                        # 'VUT', # Unknown error
                                          'TSV', # Had to increase the tolerance
                                         # 'THP', # Needs investigation, interesting error pattern
                                        # 'VST', 'VHT',# Unknown error
                                       ])
def test_plot_lemmon2000(variables):
    spec0, spec1, check_prop = variables
    plot_name = variables[0:2]
    eos = DryAirLemmon
    T, P = 298.15, 101325.0
    gas = DryAirLemmon(T=T, P=P)

    flasher = FlashPureVLS(constants=lemmon2000_constants, correlations=lemmon2000_correlations,
                       gas=gas, liquids=[], solids=[])
    flasher.TPV_HSGUA_xtol = 1e-13

    res = flasher.TPV_inputs(zs=[1.0], pts=200, spec0='T', spec1='P', check0=spec0, check1=spec1, prop0=check_prop,
                           trunc_err_low=1e-16,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, plot_name)
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s' %(plot_name, eos.__name__, fluid)
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(np.abs(errs))
    assert max_err < 1e-11

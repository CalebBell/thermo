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
from thermo.test_utils import mark_plot_unsupported
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
@pytest.mark.parametrize("variables", ['VPT', 'VTP',
                                       'PHT', 'PST', 'PUT',
                                       'VUT', 'VST', 'VHT',
                                          'TSV', # Had to increase the tolerance
                                            'THP', 'TUP', # Not consistent, warning message added
                                       ])
def test_plot_lemmon2000(variables):
    spec0, spec1, check_prop = variables
    plot_name = variables[0:2]
    eos = DryAirLemmon
    T, P = 298.15, 101325.0
    gas = DryAirLemmon(T=T, P=P)

    flasher = FlashPureVLS(constants=lemmon2000_constants, correlations=lemmon2000_correlations,
                       gas=gas, liquids=[], solids=[])
    flasher.TPV_HSGUA_xtol = 1e-14
    
    inconsistent = frozenset([spec0, spec1]) in (frozenset(['T', 'H']), frozenset(['T', 'U']))

    res = flasher.TPV_inputs(zs=[1.0], pts=200, spec0='T', spec1='P', 
                             check0=spec0, check1=spec1, prop0=check_prop,
                           trunc_err_low=1e-13,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           show=False, verbose=not inconsistent)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, plot_name)
    if not os.path.exists(path):
        os.makedirs(path)
        
    tol = 1e-13

    key = '%s - %s - %s' %(plot_name, eos.__name__, fluid)

    if inconsistent:
        spec_name = spec0 + spec1
        mark_plot_unsupported(plot_fig, reason='EOS is inconsistent for %s inputs' %(spec_name))
        tol = 1e300

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(np.abs(errs))
    assert max_err < tol
# test_plot_lemmon2000('VUT')
# test_plot_lemmon2000('THP')

def test_lemmon2000_case_issues():
    gas = DryAirLemmon(T=300.0, P=1e5)
    flasher = FlashPureVLS(constants=lemmon2000_constants, correlations=lemmon2000_correlations,
                           gas=gas, liquids=[], solids=[])
    
    # Cases which were failing because of the iteration variable of P when V specified
    # It is actually less efficient for this type of EOS
    PT = flasher.flash(T=1000.0, P=1e3)
    V = PT.V()
    U = PT.U()
    res = flasher.flash(V=V, U=U)
    assert_close(PT.T, res.T, rtol=1e-10)
    assert_close(PT.P, res.P, rtol=1e-10)
    S = PT.S()
    res = flasher.flash(V=V, S=S)
    assert_close(PT.T, res.T, rtol=1e-10)
    assert_close(PT.P, res.P, rtol=1e-10)
    H = PT.H()
    res = flasher.flash(V=V, H=H)
    assert_close(PT.T, res.T, rtol=1e-10)
    
    # Check we can't do a vapor fraction flash
    with pytest.raises(ValueError):
        flasher.flash(T=400, SF=.5)
    with pytest.raises(ValueError):
        flasher.flash(T=400, VF=.5)
        
    # Check that the minimum temperature of the phases is respected
    with pytest.raises(ValueError):
        flasher.flash(T=132.6312, P=1e3)

    
    
    PT = flasher.flash(T=2000.0000000000002, P=3827.4944785162643)
    V = PT.V()
    U = PT.U()
    res = flasher.flash(V=V, U=U)
    assert_close(PT.T, res.T, rtol=1e-10)
    assert_close(PT.P, res.P, rtol=1e-10)
    
    
    # Inconsistent TH point in fundamental formulation
    PT1 = flasher.flash(T=610.7410404288737, P=6150985.788580353)
    PT2 = flasher.flash(T=610.7410404288737, P=3967475.2794698337)
    assert_close(PT1.H(), PT2.H())
    
    # There are a ton of low-pressure points too
    PT1 = flasher.flash(T=484.38550361282495, P=0.027682980294306617)
    PT2 = flasher.flash(T=484.38550361282495, P=0.02768286630392061)
    assert_close(PT1.H(), PT2.H())

    # Inconsistent TU point in fundamental formulation
    PT1 = flasher.flash(T=1652.4510785539342, P=519770184.42714685,)
    PT2 = flasher.flash(T=1652.4510785539342, P=6985879.746785077)
    assert_close(PT1.U(), PT2.U(), rtol=1e-10)
    '''
    Ps = logspace(log10(6985879.746785077/2), log10(519770184.42714685*2), 2000)
    Us = [flasher.flash(T=1652.4510785539342, P=P).U() for P in Ps ]
    '''
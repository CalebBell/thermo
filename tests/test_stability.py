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
from math import isinf, isnan
from fluids.numerics import *
from thermo.flash import *
from thermo.phases import *
from thermo.eos_mix import *
from thermo.eos import *
from thermo.vapor_pressure import VaporPressure
from thermo.heat_capacity import *
from thermo.phase_change import *
from thermo.property_package import StabilityTester


def test_C2_C5_liq_Wilson():
    T, P = 360, 3e6
    # m = Mixture(['ethane', 'n-pentane'], ws=[.5, .5], T=T, P=P)
    zs = [0.7058336794895449, 0.29416632051045505] # m.zs
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                         HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866]))]

    Tcs, Pcs, omegas = [305.32, 469.7], [4872000.0, 3370000.0], [0.098, 0.251]
    eos_kwargs = {'Pcs': Pcs, 'Tcs': Tcs, 'omegas': omegas}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    stab = StabilityTester(Tcs=Tcs, Pcs=Pcs, omegas=omegas)

    assert_stab_success_2P(liq, gas, stab, 360.0, 3e6, zs, 'Wilson liquid', xs=[0.8178671482462099, 0.18213285175379013],
                       ys=[0.3676551321096354, 0.6323448678903647], VF=0.24884602085493626)


    # 30 Pa off from a bubble point near the critical point
    assert_stab_success_2P(liq, gas, stab, 123.3+273.15, 5475679.470049857, zs, 'Wilson liquid', xs=[0.7058338708478485, 0.29416612915215135],
                           ys=[0.4951812815325094, 0.5048187184674905], VF=9.084070806296505e-07, SS_tol=1e-25, rtol=1e-6)

    # 0.1 Pa off - very stupid/bad/evil, VF cannot be pinned down to ~ a fourth digit
    assert_stab_success_2P(liq, gas, stab, 123.3+273.15, 5475649.470049857+.01, zs, 'Wilson liquid', xs=[0.7058336795533356, 0.2941663204466643],
                           ys=[0.49517801297865455, 0.5048219870213455], VF=3.0282003822009134e-10, SS_tol=1e-25, rtol=1e-3)

    # Check maxiter > 36 allowed for stab convergence; check gas converges
    assert_stab_success_2P(liq, gas, stab, 383, 6e6, zs, 'Wilson gas', xs=[0.6068839791378426, 0.3931160208621572],
                           ys=[0.7735308634810933, 0.22646913651890652], VF=0.5937688162366092, rtol=5e-6)


def test_guesses_bad_ranges():
    stab = StabilityTester(Tcs=[647.086, 514.7], Pcs=[22048320.0, 6137000.0], omegas=[0.344, 0.635], aqueous_check=True, CASs=['7732-18-5', '64-17-5'])
    guesses = list(stab.incipient_guesses(T=6.2, P=5e4, zs=[.4, .6]))
    assert all(not isinf(x) and not isnan(x) for r in guesses for x in r)
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

from thermo.phases import *
from thermo.vapor_pressure import VaporPressure


def test_GibbbsExcessLiquid_VaporPressure():
    # Binary ethanol-water
    VaporPressures = [VaporPressure(best_fit=(159.11, 514.7, [-2.3617526481119e-19, 7.318686894378096e-16, -9.835941684445551e-13, 7.518263303343784e-10, -3.598426432676194e-07, 0.00011171481063640762, -0.022458952185007635, 2.802615041941912, -166.43524219017118])),
                      VaporPressure(best_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))]
    T = 300.0
    P = 1e5
    zs = [.4, .6]
    liquid = GibbbsExcessLiquid(VaporPressures=VaporPressures).to_TP_zs(T, P, zs)

    Psats_expect = [8778.910843769489, 3537.075987237396]
    assert_allclose(liquid.Psats(), Psats_expect, rtol=1e-12)
    
    gammas_expect = [1.0, 1.0]
    assert liquid.gammas() == gammas_expect
    
    Poyntings_expect = [1.0, 1.0]
    assert liquid.Poyntings() == Poyntings_expect
    
    dPoyntings_dT_expect = [0.0, 0.0]
    assert liquid.dPoyntings_dT() == dPoyntings_dT_expect
    
    dPoyntings_dP_expect = [0.0, 0.0]
    assert liquid.dPoyntings_dP() == dPoyntings_dP_expect
    
    phis_sat_expect = [1.0, 1.0]
    assert liquid.phis_sat() == phis_sat_expect
    
    phis_expect = [0.0877891084376949, 0.035370759872373966]
    assert_allclose(liquid.phis(), phis_expect, rtol=1e-12)
    assert_allclose(liquid.fugacity_coefficients(), phis_expect, rtol=1e-12)
    
    lnphis_expect = [-2.432817835720433, -3.3418697924678376]
    assert_allclose(liquid.lnphis(), lnphis_expect, rtol=1e-12)
    
    fugacities_expect = [3511.564337507796, 2122.245592342438]
    assert_allclose(liquid.fugacities(), fugacities_expect, rtol=1e-12)
    
    dlnphis_dT_expect = [0.05691421137269392, 0.058786419948670225]
    assert_allclose(liquid.dlnphis_dT(), dlnphis_dT_expect, rtol=1e-12)
    
    dphis_dT_expect = [0.004996447873843315, 0.0020793203437609493]
    assert_allclose(liquid.dphis_dT(), dphis_dT_expect, rtol=1e-12)
    
    # liquid.dfugacities_dT() # Missing
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

from numpy.testing import assert_allclose
import pytest
import thermo
from chemicals.utils import *
from thermo import *
from fluids.numerics import *
from math import *
import json
import os
import numpy as np


def test_C2_C5_PR():
    T, P = 300, 3e6
    constants = ChemicalConstantsPackage(Tcs=[305.32, 469.7], Pcs=[4872000.0, 3370000.0], 
                                         omegas=[0.098, 0.251], Tms=[90.3, 143.15], 
                                         Tbs=[184.55, 309.21], CASs=['74-84-0', '109-66-0'],
                                         names=['ethane', 'pentane'], MWs=[30.06904, 72.14878])
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                         HeatCapacityGas(best_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866]))]
    correlations = PropertyCorrelationPackage(constants, HeatCapacityGases=HeatCapacityGases)
    zs = ws_to_zs(MWs=constants.MWs, ws=[.5, .5])
    
    
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = EOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    
    flasher = FlashVL(constants, correlations, liq, gas)
    # Check there are two phases near the dew point. don't bother checking the composition most of the time.
    
    # When this test was written, case is still valid for a dP of 0.00000001 Pa
        
    # Issue here was that (sum_criteria < 1e-7) was the check in the stability test result interpretation
    # Fixed it by decreasing the tolerance 10x (1e-8)
    res = flasher.flash(P=5475649.470049857+15, T=123.3+273.15, zs=zs)
    assert_close1d(res.betas, [0.9999995457838572, 4.5421614280893863e-07], rtol=1e-4)
    assert_close1d(res.gas.zs, [0.7058337751720506, 0.29416622482794935], rtol=1e-4)
    assert_close1d(res.liquid0.zs, [0.49517964670906095, 0.504820353290939], rtol=1e-4)
    
    
    # # In this case, the tolerance had to be decreased 10x more - to 1e-9! Triggered at a dP of 0.5
    res = flasher.flash(P=5475649.470049857+0.5, T=123.3+273.15, zs=zs)
    assert_close1d(res.betas, [0.999999984859061, 1.5140938947055815e-08], rtol=1e-4)
    assert_close1d(res.gas.zs, [0.7058336826506021, 0.29416631734939785])
    assert_close1d(res.liquid0.zs, [0.4951780663825745, 0.5048219336174254])
    
    # # This one is too close to the border - the VF from SS is less than 0, 
    # # but if the tolerance is increased, it is positive (and should be)
    res = flasher.flash(P=5475649.470049857+0.001, T=123.3+273.15, zs=zs)
    assert_close1d(res.betas, [0.9999999999697144, 3.028555184414472e-11], rtol=3e-3)
    assert_close1d(res.gas.zs, [0.7058336794959247, 0.29416632050407526])
    assert_close1d(res.liquid0.zs, [0.49517801199759515, 0.5048219880024049])
    
    # This one is presently identified as a LL...  just check the number of phases
    assert flasher.flash(zs=zs, P=6.615e6, T=386).phase_count == 2
    
    
def test_flash_TP_K_composition_idependent_unhappiness():
    constants = ChemicalConstantsPackage(Tcs=[508.1, 536.2, 512.5], Pcs=[4700000.0, 5330000.0, 8084000.0], omegas=[0.309, 0.21600000000000003, 0.5589999999999999],
                                         MWs=[58.07914, 119.37764000000001, 32.04186], CASs=['67-64-1', '67-66-3', '67-56-1'], names=['acetone', 'chloroform', 'methanol'])
    
    HeatCapacityGases = [HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.3320002425347943e-21, 6.4063345232664645e-18, -1.251025808150141e-14, 1.2265314167534311e-11, -5.535306305509636e-09, -4.32538332013644e-08, 0.0010438724775716248, -0.19650919978971002, 63.84239495676709])),
     HeatCapacityGas(best_fit=(200.0, 1000.0, [1.5389278550737367e-21, -8.289631533963465e-18, 1.9149760160518977e-14, -2.470836671137373e-11, 1.9355882067011222e-08, -9.265600540761629e-06, 0.0024825718663005762, -0.21617464276832307, 48.149539665907696])),
     HeatCapacityGas(best_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924]))]
    VolumeLiquids = [VolumeLiquid(best_fit=(178.51, 498.1, [6.564241965071999e-23, -1.6568522275506375e-19, 1.800261692081815e-16, -1.0988731296761538e-13, 4.118691518070104e-11, -9.701938804617744e-09, 1.4022905458596618e-06, -0.00011362923883050033, 0.0040109650220160956])),
                    VolumeLiquid(best_fit=(209.63, 509.5799999999999, [2.034047306563089e-23, -5.45567626310959e-20, 6.331811062990084e-17, -4.149759318710192e-14, 1.6788970104955462e-11, -4.291900093120011e-09, 6.769385838271721e-07, -6.0166473220815445e-05, 0.0023740769479069054])),
                    VolumeLiquid(best_fit=(175.7, 502.5, [3.5725079384600736e-23, -9.031033742820083e-20, 9.819637959370411e-17, -5.993173551565636e-14, 2.2442465416964825e-11, -5.27776114586072e-09, 7.610461006178106e-07, -6.148574498547711e-05, 0.00216398089328537])),]
    
    VaporPressures = [VaporPressure(best_fit=(178.51, 508.09000000000003, [-1.3233111115238975e-19, 4.2217134794609376e-16, -5.861832547132719e-13, 4.6488594950801467e-10, -2.3199079844570237e-07, 7.548290741523459e-05, -0.015966705328994194, 2.093003523977292, -125.39006100979816])),
                      VaporPressure(best_fit=(207.15, 536.4, [-8.714046553871422e-20, 2.910491615051279e-16, -4.2588796020294357e-13, 3.580003116042944e-10, -1.902612144361103e-07, 6.614096470077095e-05, -0.01494801055978542, 2.079082613726621, -130.24643185169472])),
                      VaporPressure(best_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10, -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708]))]
    
    liquid = GibbsExcessLiquid(VaporPressures=VaporPressures, VolumeLiquids=VolumeLiquids, 
                     HeatCapacityGases=HeatCapacityGases, use_Poynting=True,
                     use_phis_sat=False)
    
    correlations = PropertyCorrelationPackage(constants=constants, skip_missing=True, HeatCapacityGases=HeatCapacityGases,
                               VolumeLiquids=VolumeLiquids, VaporPressures=VaporPressures)
    
    T, P = 350.0, 1e6
    zs = [0.2, 0.0, 0.8]
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas':constants.omegas}
    gas = IdealGas(HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, correlations, liquids=[liquid], gas=gas)
    
    # Low - all K under zero
    res =  flashN.flash(T=T, P=P, zs=zs)
    assert_close(res.rho_mass(), 733.1047159397776)
    assert 1 == res.phase_count
    assert res.liquid0 is not None
    
    # High - all K above zero
    res = flashN.flash(T=430, P=1e4, zs=zs)
    assert 1 == res.phase_count
    assert res.gas is not None
    assert_close(res.rho_mass(), 0.10418751067559757)
    
    # One K value is under 1, rest are above - but that component has mole frac of zero
    res = flashN.flash(T=420, P=1e4, zs=zs)
    assert 1 == res.phase_count
    assert res.gas is not None
    
    # phis_at for liquids was broken, breaking this calculation
    res = flashN.flash(T=285.5, P=1e4, zs=zs)
    assert_allclose(res.betas, [0.21860038882559643, 0.7813996111744036])
    assert res.phase_count == 2
    
    # Two cases RR was working on Ks less than 1, and coming up with a made up VF
    # Need to check Ks first
    res = flashN.flash(T=300.0000, P=900000.0000, zs=[0.5, 0.1, 0.4, 0.0],)
    assert 1 == res.phase_count
    assert res.gas is None
    res = flashN.flash(T=300.0000, P=900000.0000, zs=[.5, 0, 0, .5])
    assert 1 == res.phase_count
    assert res.gas is None

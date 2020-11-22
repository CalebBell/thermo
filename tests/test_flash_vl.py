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
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)

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

def test_flash_combustion_products():
    P = 1e5
    T = 794.5305048838037
    zs = [0.5939849621247668, 0.112781954982051, 0.0676691730155464, 0.2255639098776358]
    constants = ChemicalConstantsPackage(atomss=[{'N': 2}, {'C': 1, 'O': 2}, {'O': 2}, {'H': 2, 'O': 1}], CASs=['7727-37-9', '124-38-9', '7782-44-7', '7732-18-5'], MWs=[28.0134, 44.0095, 31.9988, 18.01528], names=['nitrogen', 'carbon dioxide', 'oxygen', 'water'], omegas=[0.04, 0.2252, 0.021, 0.344], Pcs=[3394387.5, 7376460.0, 5042945.25, 22048320.0], Tbs=[77.355, 194.67, 90.18799999999999, 373.124], Tcs=[126.2, 304.2, 154.58, 647.14], Tms=[63.15, 216.65, 54.36, 273.15])
    correlations = PropertyCorrelationPackage(constants=constants, skip_missing=True,
        HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                           HeatCapacityGas(best_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
                           HeatCapacityGas(best_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
                           HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))])
    kijs = [[0.0, -0.0122, -0.0159, 0.0], [-0.0122, 0.0, 0.0, 0.0952], [-0.0159, 0.0, 0.0, 0.0], [0.0, 0.0952, 0.0, 0.0]]
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flasher = FlashVL(constants, correlations, liq, gas)
    res = flasher.flash(T=T, P=P, zs=zs)
    
    assert res.gas
    assert res.phase == 'V'
    
    
def test_bubble_T_PR_VL():
    # Last point at 8e6 Pa not yet found.

    constants = ChemicalConstantsPackage(CASs=['124-38-9', '110-54-3'], MWs=[44.0095, 86.17536], names=['carbon dioxide', 'hexane'], omegas=[0.2252, 0.2975], Pcs=[7376460.0, 3025000.0], Tbs=[194.67, 341.87], Tcs=[304.2, 507.6], Tms=[216.65, 178.075])
    correlations = PropertyCorrelationPackage(constants=constants, skip_missing=True,
    HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
                       HeatCapacityGas(best_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998]))])
    zs = [.5, .5]
    T = 300.0
    P = 1e6
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flasher = FlashVL(constants, correlations, liq, gas)
    res = flasher.flash(P=7.93e6, VF=0, zs=zs)
    assert_close(res.T, 419.0621213529388, rtol=1e-6)
    
def test_PR_four_bubble_dew_cases_VL():
    zs=[.5, .5]
    T=300.0
    P=1E6
    constants = ChemicalConstantsPackage(CASs=['98-01-1', '98-00-0'], MWs=[96.08406000000001, 98.09994], names=['2-furaldehyde', 'furfuryl alcohol'], omegas=[0.4522, 0.7340000000000001], Pcs=[5510000.0, 5350000.0], Tbs=[434.65, 441.15], Tcs=[670.0, 632.0], Tms=[235.9, 250.35])
    correlations = PropertyCorrelationPackage(constants=constants, skip_missing=True,
        HeatCapacityGases=[HeatCapacityGas(best_fit=(298, 1000, [4.245751608816354e-21, -2.470461837781697e-17, 6.221823690784335e-14, -8.847967216702641e-11, 7.749899297737877e-08, -4.250059888737765e-05, 0.013882452355067994, -2.1404621487165327, 185.84988012691903])),
                           HeatCapacityGas(best_fit=(250.35, 632.0, [-9.534610090167143e-20, 3.4583416772306854e-16, -5.304513883184021e-13, 4.410937690059558e-10, -2.0905505018557675e-07, 5.20661895325169e-05, -0.004134468659764938, -0.3746374641720497, 114.90130267531933]))])
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flasher = FlashVL(constants, correlations, liq, gas)
    assert_close(flasher.flash(P=1e6, VF=0, zs=zs).T, 539.1838522423529, rtol=1e-6)
    assert_close(flasher.flash(P=1e6, VF=1, zs=zs).T, 540.2081697501809, rtol=1e-6)
    assert_close(flasher.flash(T=600.0, VF=0, zs=zs).P, 2766476.7473238464, rtol=1e-6)
    assert_close(flasher.flash(T=600.0, VF=1, zs=zs).P, 2702616.6490743402, rtol=1e-6)
    
    
def test_C1_C10_PT_flash_VL():
    IDs = ['methane', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    zs=[.1]*10
    T=300.0
    P=1E5
    constants = ChemicalConstantsPackage(CASs=['74-82-8', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '110-54-3', '142-82-5', '111-65-9', '111-84-2', '124-18-5'], MWs=[16.04246, 30.06904, 44.09562, 58.1222, 72.14878, 86.17536, 100.20194000000001, 114.22852, 128.2551, 142.28168], names=['methane', 'ethane', 'propane', 'butane', 'pentane', 'hexane', 'heptane', 'octane', 'nonane', 'decane'], omegas=[0.008, 0.098, 0.152, 0.193, 0.251, 0.2975, 0.3457, 0.39399999999999996, 0.444, 0.49], Pcs=[4599000.0, 4872000.0, 4248000.0, 3796000.0, 3370000.0, 3025000.0, 2740000.0, 2490000.0, 2290000.0, 2110000.0], Tbs=[111.65, 184.55, 231.04, 272.65, 309.21, 341.87, 371.53, 398.77, 423.95, 447.25], Tcs=[190.56400000000002, 305.32, 369.83, 425.12, 469.7, 507.6, 540.2, 568.7, 594.6, 611.7], Tms=[90.75, 90.3, 85.5, 135.05, 143.15, 178.075, 182.15, 216.3, 219.9, 243.225])
    correlations = PropertyCorrelationPackage(constants=constants, skip_missing=True,
    HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                        HeatCapacityGas(best_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                        HeatCapacityGas(best_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
                        HeatCapacityGas(best_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
                        HeatCapacityGas(best_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
                        HeatCapacityGas(best_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
                        HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.4046935863496273e-21, 5.8024177500786575e-18, -7.977871529098155e-15, 7.331444047402207e-13, 9.954400606484495e-09, -1.2112107913343475e-05, 0.0062964696142858104, -1.0843106737278825, 173.87692850911935])),
                        HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303])),
                        HeatCapacityGas(best_fit=(200.0, 1000.0, [6.513870466670624e-22, -5.318305817618858e-18, 1.8015815307749625e-14, -3.370046452151828e-11, 3.840755097595374e-08, -2.7203677889897072e-05, 0.011224516822410626, -1.842793858054514, 247.3628627781443])),
                        HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735]))])
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flasher = FlashVL(constants, correlations, liq, gas)
    res = flasher.flash(T=T, P=P, zs=zs)
    assert_close(res.VF, 0.3933480634014041, rtol=1e-5)
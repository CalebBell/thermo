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
SOFTWARE.
'''

import os
from math import *

import pytest
from fluids.numerics import *
from fluids.numerics import assert_close, assert_close1d, assert_close2d

import thermo
from thermo import *
import json
try:
    import matplotlib.pyplot as plt
except:
    pass

flashN_surfaces_dir = os.path.join(thermo.thermo_dir, '..', 'surfaces', 'flashN')


def test_water_C1_C8():
    T = 298.15
    P = 101325.0
    omegas = [0.344, 0.008, 0.394]
    Tcs = [647.14, 190.564, 568.7]
    Pcs = [22048320.0, 4599000.0, 2490000.0]
    kijs=[[0,0, 0],[0,0, 0.0496], [0,0.0496,0]]
    zs = [1.0/3.0]*3
    N = len(zs)

    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                         HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                         HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303]))]
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[18.01528, 16.04246, 114.22852],
                                         CASs=['7732-18-5', '74-82-8', '111-65-9'])
    correlations = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    eos_kwargs = dict(Tcs=Tcs, Pcs=Pcs, omegas=omegas, kijs=kijs)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)


    flashN = FlashVLN(constants, correlations, liquids=[liq, liq], gas=gas)

    # This one has a cycle
    res = flashN.flash(zs=zs, T=393.23626311210126, P=500000.0)
    assert_close(res.G_min_criteria(), -4032.6088842679956)

    # this one doesn't converge the liquid liquid
    res = flashN.flash(zs=[.5, .4, .1], T=300, P=14085.443746269448)
    assert res.phase_count == 2
    assert_close(res.G_min_criteria(), -2852.8406721218416)



    res = flashN.flash(T=T, P=P, zs=zs)

    assert_close1d(res.water_phase.zs, [0.9999990988582429, 9.011417571269618e-07, 9.57378962042325e-17])
    assert_close1d(res.gas.zs, [0.026792655758364814, 0.9529209534990141, 0.020286390742620692])
    assert res.phase_count == 3




    # Betas and so on
    assert_close1d(res.betas, [0.3481686901529188, 0.3182988549790946, 0.33353245486798655])
    assert_close1d(res.betas_mass, [0.12740321450325565, 0.11601025808509248, 0.7565865274116519])
    assert_close1d(res.betas_volume, [0.9927185024780876, 0.0007904962655801318, 0.0064910012563324])

    assert res.solids_betas == []
    assert res.VF == res.gas_beta

    assert_close1d(res.betas_states, [0.3481686901529188, 0.6518313098470812, 0])
    assert_close1d(res.betas_mass_states, [0.12740321450325565, 0.8725967854967444, 0.0])
    assert_close1d(res.betas_volume_states, [0.9927185024780874, 0.007281497521912531, 0.0])

    assert_close1d(res.betas_liquids, [0.4883147682086139, 0.5116852317913861])
    assert_close1d(res.betas_mass_liquids, [0.13294829870253438, 0.8670517012974656])
    assert_close1d(res.betas_volume_liquids, [0.10856232020971736, 0.8914376797902827])

    assert_close(res.gas.beta, 0.3481686901529188)
    assert_close(res.liquid0.beta, 0.3182988549790946)
    assert_close(res.liquid1.beta, 0.33353245486798655)

    assert_close(res.gas.beta_mass, 0.12740321450325565)
    assert_close(res.liquid0.beta_mass, 0.11601025808509248)
    assert_close(res.liquid1.beta_mass, 0.7565865274116519)

    assert_close(res.gas.beta_volume, 0.9927185024780876)
    assert_close(res.liquid0.beta_volume, 0.0007904962655801318)
    assert_close(res.liquid1.beta_volume, 0.0064910012563324)


    assert_close(res.liquid_bulk.beta, 0.6518313098470812)
    assert_close(res.bulk.beta, 1.0)



    # Gas phase - high T
    highT = flashN.flash(T=500.0, P=P, zs=zs)
    assert highT.phase_count == 1
    assert highT.gas is not None

    # point where aqueous phase was not happening, only two liquids; I fixed it by trying water stab first
    res = flashN.flash(T=307.838, P=8.191e6, zs=zs)
    zs_water_expect = [0.9999189354094269, 8.106459057259032e-05, 5.225089428308217e-16]
    assert_close1d(res.water_phase.zs, zs_water_expect, rtol=1e-5)
    zs_gas_expect = [0.0014433546288458944, 0.9952286230641441, 0.0033280223070098593]
    assert_close1d(res.gas.zs, zs_gas_expect, rtol=1e-5)
    zs_other_expect = [0.018970964990697076, 0.3070011367278533, 0.6740278982814496]
    assert_close1d(res.lightest_liquid.zs, zs_other_expect, rtol=1e-5)

    # Point where failures are happening with DOUBLE_CHECK_2P
    flashN.DOUBLE_CHECK_2P = True
    res = flashN.flash(T=100.0, P=33932.21771895288, zs=zs)
    assert res.gas is not None
    assert res.phase_count == 3

    # Another point - DOUBLE_CHECK_2P was failing due to flash_inner_loop all having K under 1
    res = flashN.flash(T=100.0, P=49417.13361323757, zs=zs)
    assert res.phase_count == 2
    assert res.gas is not None
    assert_close1d(res.water_phase.zs, [0.9999999999999996, 3.2400602001103857e-16, 1.2679535429897556e-61], atol=1e-10)

    # Another point - the RR solution went so far it had a compositions smaller than 1e-16 causing issues
    res = flashN.flash(T=100.0, P=294705.1702551713, zs=zs)
    assert res.phase_count == 3
    assert res.gas is None
    assert_close(res.water_phase.Z(), 0.006887567438189129)
    assert_close(res.lightest_liquid.Z(), 0.011644749497712284)

    # Test the flash bubble works OK with multiple liquids (T-VF=1 and P-VF=1)
    liq_SRK = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)

    res_PR = FlashVLN(constants, correlations, liquids=[liq], gas=gas).flash(T=298.15, VF=1, zs=zs)
    res_SRK = FlashVLN(constants, correlations, liquids=[liq_SRK], gas=gas).flash(T=298.15, VF=1, zs=zs)
    assert res_PR.G() > res_SRK.G()
    new_flasher = FlashVLN(constants, correlations, liquids=[liq, liq_SRK], gas=gas)
    res_both_liquids = new_flasher.flash(T=298.15, VF=1, zs=zs)
    assert_close(res_both_liquids.G(), res_SRK.G(), rtol=1e-4)
    assert_close(res_PR.P, 6022.265230194498, rtol=5e-4)
    assert_close(res_SRK.P, 5555.019566177178, rtol=5e-4)

    res_PR = FlashVLN(constants, correlations, liquids=[liq], gas=gas).flash(P=6000.0, VF=1, zs=zs)
    res_SRK = FlashVLN(constants, correlations, liquids=[liq_SRK], gas=gas).flash(P=6000.0, VF=1, zs=zs)
    assert res_PR.G() > res_SRK.G()
    res_both_liquids = FlashVLN(constants, correlations, liquids=[liq, liq_SRK], gas=gas).flash(P=6000.0, VF=1, zs=zs)
    assert_close(res_both_liquids.G(), res_SRK.G(), rtol=1e-4)
    assert_close(res_PR.T, 298.0822786634035, rtol=5e-4)
    assert_close(res_SRK.T, 299.5323133142487, rtol=5e-4)

    # Check we can store this
    output = json.loads(json.dumps(flashN.as_json()))
    assert 'constants' not in output
    assert 'correlations' not in output
    assert 'settings' not in output
    new_flasher = FlashVLN.from_json(output, {'constants': constants, 'correlations': correlations, 'settings': flashN.settings})
    assert new_flasher == flashN

def test_C1_to_C5_water_gas():
    zs = normalize([.65, .13, .09, .05, .03, .03, .02, .003, 1e-6])
#
    T = 300.0
    P = 3000e3
    constants = ChemicalConstantsPackage(Tcs=[190.56400000000002, 305.32, 369.83, 425.12, 469.7, 647.14, 126.2, 304.2, 373.2],
                                            Pcs=[4599000.0, 4872000.0, 4248000.0, 3796000.0, 3370000.0, 22048320.0, 3394387.5, 7376460.0, 8936865.0],
                                            omegas=[0.008, 0.098, 0.152, 0.193, 0.251, 0.344, 0.04, 0.2252, 0.1],
                                            MWs=[16.04246, 30.06904, 44.09562, 58.1222, 72.14878, 18.01528, 28.0134, 44.0095, 34.08088],
                                            CASs=['74-82-8', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '7732-18-5', '7727-37-9', '124-38-9', '7783-06-4'],)
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534])), ])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)

    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    res = flashN.flash(T=T, P=P, zs=zs)

    assert res.phase_count == 3
    assert_close(res.betas[0], 0.9254860647854957, rtol=1e-6)
    assert_close(res.liquids_betas[res.water_phase_index], 0.027819781620531732, rtol=1e-5)

    # Gibbs optimizer cannot find anything else for a third phase
    assert 2 == flashN.flash(T=273.15-130, P=1e6, zs=zs).phase_count

    # High temperature point
    highT = flashN.flash(T=400.0, P=206913.80811147945, zs=zs)
    assert highT.gas is not None
    assert highT.phase_count == 1

def test_C1_C8_water_TEG():
    zs = [.25, .25, .25, .25]

    T = 298.15
    P = 101325.0
    constants = ChemicalConstantsPackage(Tcs=[190.56400000000002, 568.7, 780.0, 647.14],
                                            Pcs=[4599000.0, 2490000.0, 3300000.0, 22048320.0],
                                            omegas=[0.008, 0.39399999999999996, 0.5842, 0.344],
                                            MWs=[16.04246, 114.22852, 150.17296, 18.01528],
                                            CASs=['74-82-8', '111-65-9', '112-27-6', '7732-18-5'],)
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303])),
                                                                                     HeatCapacityGas(poly_fit=(266.15, 780.0, [-1.4884362017902977e-20, 5.2260426653411555e-17, -6.389395767183489e-14, 1.5011276073418378e-11, 3.9646353076966065e-08, -4.503513372576425e-05, 0.020923507683244157, -4.012905599723838, 387.9369199281481])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])), ])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)

    flashN = FlashVLN(constants, properties, liquids=[liq, liq, liq], gas=gas)
    VLL = flashN.flash(T=T, P=P, zs=zs)
    assert VLL.phase_count == 3
    assert_close1d(VLL.gas.zs, [0.9588632822157593, 0.014347647576433926, 2.8303984647442933e-06, 0.02678623980934197], )


def test_C5_C6_C7():
    zs = [0.8168, 0.1501, 0.0331]
    # m = Mixture(['n-pentane', 'n-hexane', 'heptane'], zs=zs, T=300, P=1E6)
    kijs = [[0.0, 0.00076, 0.00171], [0.00076, 0.0, 0.00061], [0.00171, 0.00061, 0.0]]
    Tcs = [469.7, 507.6, 540.2]
    Pcs = [3370000.0, 3025000.0, 2740000.0]
    omegas = [0.251, 0.2975, 0.3457]
    MWs = [72.14878, 86.17536, 100.20194]
    CASs = ['109-66-0', '110-54-3', '142-82-5']

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=MWs, CASs=CASs)
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[
            HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
            HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
            HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.4046935863496273e-21, 5.8024177500786575e-18, -7.977871529098155e-15, 7.331444047402207e-13, 9.954400606484495e-09, -1.2112107913343475e-05, 0.0062964696142858104, -1.0843106737278825, 173.87692850911935]))])
    T, P = 180.0, 4.0
    eos_kwargs = dict(Tcs=Tcs, Pcs=Pcs, omegas=omegas, kijs=kijs)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flasher = FlashVLN(constants, properties, liquids=[liq], gas=gas)

    # Extremely low pressure point - genuinely is a VL point, but G is lower by a tiny amount
    res = flasher.flash(T=T, P=P, zs=zs)
    assert_close1d(res.betas, [0.9973290812443733, 0.00267091875562675])

    # point where the flash was failing to found both sides
    res = flasher.flash(T=160, VF=1-1e-8, zs=zs)
    assert_close(res.P, 0.13546634170397667)


    # Do some nasty lengthy checks. Leaving for legacy reasons.
    Ts = linspace(160, 200, 8) + linspace(204, 420, 8)+ linspace(425, 473, 8)
    P_dews_expect = [0.13546631710512694, 0.3899541879697901, 1.0326385606771222, 2.5367431562236216, 5.823754581354464, 12.576541985263733, 25.695934298398683, 49.92852511524303, 77.336111529878, 1224.2477432039364, 8851.275697515553, 38739.02285212159, 121448.37959695983, 302817.44109532895, 642356.4259573148, 1212555.0457456997, 1332379.263527949, 1511043.452511087, 1707364.218790161, 1922627.5124993164, 2158275.2762631513, 2416007.942794255, 2698050.322554657, 3008161.4940377055]
    P_bubbles_expect= [1.6235262252797003, 3.7270329355478946, 8.051176710411168, 16.46641490114325, 32.05492139681795, 59.672217732280416, 106.66299735475893, 183.73589515016087, 263.46184653114784, 2681.961688786174, 14942.285482185738, 55694.65563520535, 157351.54737219962, 365368.2524627791, 735848.7931013029, 1333301.7057641603, 1456215.734621984, 1638045.031281808, 1835927.295044813, 2050617.935270042, 2282799.5140801608, 2532987.1229581917, 2801271.5348919635, 3086319.6690727174]

    P_dews = []
    P_bubbles = []
    for T in Ts:
        res = flasher.flash(T=T, VF=0, zs=zs)
        P_bubbles.append(res.P)
        res = flasher.flash(T=T, VF=1, zs=zs)
        P_dews.append(res.P)

    assert_close1d(P_bubbles, P_bubbles_expect, rtol=5e-5)
    assert_close1d(P_dews, P_dews_expect, rtol=5e-5)

    # For each point, solve it as a T problem.
    for P, T in zip(P_bubbles, Ts):
        res = flasher.flash(P=P, VF=0, zs=zs)
        assert_close(res.T, T, rtol=5e-5)
    for P, T in zip(P_dews, Ts):
        res = flasher.flash(P=P, VF=1, zs=zs)
        assert_close(res.T, T, rtol=5e-5)

    # Test the bubble/dew flashes;
    # Skip most of them as redundant
    idxs = [0, 1, 2, 17, 21, 24]
    # Could comment these out.
    for i, (T, P_bub, P_dew) in enumerate(zip(Ts, P_bubbles_expect, P_dews_expect)):
        if i not in idxs:
            continue
        res = flasher.flash(T=T, VF=0+1e-9, zs=zs)
        assert_close(P_bub, res.P, rtol=5e-5)
        res = flasher.flash(T=T, VF=1-1e-9, zs=zs)
        assert_close(P_dew, res.P, rtol=5e-5)

    for i, (P, T) in enumerate(zip(P_dews_expect, Ts)):
        if i not in idxs:
            continue
        res = flasher.flash(P=P, VF=1-1e-9, zs=zs)
        assert_close(P, res.P)

    for i, (P, T) in enumerate(zip(P_bubbles_expect, Ts)):
        if i not in idxs:
            continue
        res = flasher.flash(P=P, VF=0+1e-9, zs=zs)
        assert_close(P, res.P)


def test_binary_LLL_specified_still_one_phase():
    T = 167.54
    P = 26560
    IDs = ['methane', 'hydrogen sulfide']
    zs = [0.93, 0.07]
    kijs = [[0,.08],[0.08,0]]
    Tcs=[190.6, 373.2]
    Pcs=[46e5, 89.4e5]
    omegas=[0.008, .1]

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[16.04246, 34.08088], CASs=['74-82-8', '7783-06-4'])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534]))], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, correlations, liquids=[liq, liq, liq], gas=gas)
    res = flashN.flash(T=T, P=P, zs=zs)
    assert res.phase_count == 1
    assert res.gas is not None

    # Check we can store this
    output = json.loads(json.dumps(flashN.as_json()))
    new_flasher = FlashVLN.from_json(output)
    assert new_flasher == flashN

def test_binary_phase_switch():
    # Example from Kodama, Daisuke, Ryota Sato, Aya Haneda, and Masahiro Kato. “High-Pressure Phase Equilibrium for Ethylene + Ethanol at 283.65 K.” Journal of Chemical & Engineering Data 50, no. 1 (January 1, 2005): 122–24. https://doi.org/10.1021/je049796y.
    # Shows a phase switch across the VLL line (which can never be flashed at TP)
    zs = [.8, 0.2]
    #m = Mixture(['ethylene', 'ethanol'], zs=zs)
    T = 283.65
    P = 4690033.135557525
    constants = ChemicalConstantsPackage(Tcs=[282.34, 514.0], Pcs=[5041000.0, 6137000.0], omegas=[0.085, 0.635], MWs=[28.05316, 46.06844], CASs=['74-85-1', '64-17-5'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.2701693466919565e-21, 1.660757962278189e-17, -3.525777713754962e-14, 4.01892664375958e-11, -2.608749347072186e-08, 9.23682495982131e-06, -0.0014524032651835623, 0.09701764355901257, 31.034399100170667])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas, kijs=[[0.0, -.0057], [-.0057, 0.0]])


    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)

    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)


    LL = flashN.flash(P=P+100, T=T, zs=zs)
    assert_close1d(LL.lightest_liquid.zs, [0.9145548102807435, 0.08544518971925665])
    assert_close1d(LL.heaviest_liquid.zs, [0.5653776843338143, 0.4346223156661858])
    assert LL.phase_count == 2
    VL = flashN.flash(P=P-100, T=T, zs=zs)
    assert VL.phase_count == 2
    rhos_expect = [106.06889300473189, 562.1609367529746]
    assert_close1d(rhos_expect, [i.rho_mass() for i in VL.phases], rtol=1e-5)
    assert_close1d(VL.liquid0.zs, [0.5646523538677704, 0.43534764613222976])
    assert_close1d(VL.gas.zs, [0.9963432818384895, 0.0036567181615104662])

def test_three_phase_ethylene_ethanol_nitrogen():
    zs = [.8, 0.19, .01]
    # m = Mixture(['ethylene', 'ethanol', 'nitrogen'], zs=zs)
    T = 283.65
    P = 4690033.135557525
    constants = ChemicalConstantsPackage(Tcs=[282.34, 514.0, 126.2], Pcs=[5041000.0, 6137000.0, 3394387.5], omegas=[0.085, 0.635, 0.04], MWs=[28.05316, 46.06844, 28.0134], CASs=['74-85-1', '64-17-5', '7727-37-9'])
    properties =PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.2701693466919565e-21, 1.660757962278189e-17, -3.525777713754962e-14, 4.01892664375958e-11, -2.608749347072186e-08, 9.23682495982131e-06, -0.0014524032651835623, 0.09701764355901257, 31.034399100170667])),
                                                                                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),
                                                                                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas, kijs=[[0.0, -.0057, 0.0], [-.0057, 0.0, 0.0], [0.0, 0.0, 0.0]])


    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)

    # SS trivial solution needs to be checked
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    one_phase = flashN.flash(T=366.66666666666674, P=13335214.32163324, zs=zs)
    assert one_phase.liquid0
    assert one_phase.phase_count == 1

    # over 1000 iterations to converge
    many_iter_VL = flashN.flash(T=266.66666666666663, P=7498942.093324558, zs=zs)
    flashN.PT_SS_TOL = 1e-20
    assert_close1d([588.7927414481507, 473.2881762757268], [i.rho_mass() for i in many_iter_VL.phases], rtol=1e-5)
    assert_close1d(many_iter_VL.betas, [0.1646137076373939, 0.8353862923626061], rtol=2e-3)
    assert_close(many_iter_VL.G(), 3725.6090821958787, atol=1)
    flashN.PT_SS_TOL = 1e-13

    # SS trivial solution to all Ks under 1
    one_phase = flashN.flash(T=283.3333333333333, P=10000000.0, zs=zs)
    assert one_phase.liquid0
    assert one_phase.phase_count == 1

    # point RR was making SS convergence die
    bad_RR = flashN.flash(T=220.2020202020202, P=37649358.067924485, zs=zs)
    assert_close1d(bad_RR.betas, [0.13605690662613873, 0.8639430933738612], rtol=1e-5)

    # Point where the three phase calculations converge to negative betas
    false3P = flashN.flash(T=288.56, P=7.3318e6, zs=zs)
    assert_close1d(false3P.betas, [0.24304503666565203, 0.756954963334348], rtol=1e-5)
    assert false3P.liquid0
    assert false3P.liquid1

    # Three phase region point
    res = flashN.flash(T=200, P=6e5, zs=zs)
    assert_close(res.G(), -2873.6029915490544, rtol=1e-5)
    assert res.phase_count == 3

def test_methane_nitrogen_sharp_T_flash_failure_2_component_dew():
    constants = ChemicalConstantsPackage(atomss=[{'C': 1, 'H': 4}, {'N': 2}], CASs=['74-82-8', '7727-37-9'], Gfgs=[-50443.48000000001, 0.0], Hfgs=[-74534.0, 0.0], MWs=[16.04246, 28.0134], names=['methane', 'nitrogen'], omegas=[0.008, 0.04], Pcs=[4599000.0, 3394387.5], Sfgs=[-80.79999999999997, 0.0], Tbs=[111.65, 77.355], Tcs=[190.564, 126.2])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    VaporPressures=[VaporPressure(load_data=False, exp_poly_fit=(90.8, 190.554, [1.2367137894255505e-16, -1.1665115522755316e-13, 4.4703690477414014e-11, -8.405199647262538e-09, 5.966277509881474e-07, 5.895879890001534e-05, -0.016577129223752325, 1.502408290283573, -42.86926854012409])),
    VaporPressure(load_data=False, exp_poly_fit=(63.2, 126.18199999999999, [5.490876411024536e-15, -3.709517805130509e-12, 1.0593254238679989e-09, -1.6344291780087318e-07, 1.4129990091975526e-05, -0.0005776268289835264, -0.004489180523814208, 1.511854256824242, -36.95425216567675])),
    ],
    VolumeLiquids=[VolumeLiquid(load_data=False, poly_fit=(90.8, 180.564, [7.730541828225242e-20, -7.911042356530585e-17, 3.51935763791471e-14, -8.885734012624568e-12, 1.3922694980104743e-09, -1.3860056394382538e-07, 8.560110533953199e-06, -0.00029978743425740123, 0.004589555868318768])),
    VolumeLiquid(load_data=False, poly_fit=(63.2, 116.192, [9.50261462694019e-19, -6.351064785670885e-16, 1.8491415360234833e-13, -3.061531642102745e-11, 3.151588109585604e-09, -2.0650965261816766e-07, 8.411110954342014e-06, -0.00019458305886755787, 0.0019857193167955463])),
    ],
    HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
    ],
    ViscosityLiquids=[ViscosityLiquid(load_data=False, exp_poly_fit=(90.8, 190.554, [-4.2380635484289735e-14, 4.5493522318564076e-11, -2.116492796413691e-08, 5.572829307777471e-06, -0.0009082059294533955, 0.0937979289971011, -5.994475607852222, 216.68461775245618, -3398.7238594522332])),
    ViscosityLiquid(load_data=False, exp_poly_fit=(63.2, 126.18199999999999, [-2.0668914276868663e-12, 1.4981401912829941e-09, -4.7120662288238675e-07, 8.398998068709547e-05, -0.009278622651904659, 0.6505143753550678, -28.263815807515204, 695.73318887296, -7435.03033205167])),
    ],
    ViscosityGases=[ViscosityGas(load_data=False, poly_fit=(111.67, 190.0, [-1.8353011229090837e-22, 2.3304595414632153e-19, -1.2923749174024514e-16, 4.090196030268957e-14, -8.086198992052456e-12, 1.0237770716128214e-09, -8.126336931959868e-08, 3.756286853489369e-06, -7.574255749674298e-05])),
    ViscosityGas(load_data=False, poly_fit=(63.15, 1970.0, [-1.5388147506777722e-30, 1.4725459575473186e-26, -5.996335123822459e-23, 1.363536066391023e-19, -1.9231682598532616e-16, 1.7971629245352389e-13, -1.208040032788892e-10, 8.598598782963851e-08, -6.144085424881471e-07])),
    ],
    ThermalConductivityLiquids=[ThermalConductivityLiquid(load_data=False, poly_fit=(111.67, 190.0, [3.107229777952286e-14, -3.625875816800732e-11, 1.841942435684153e-08, -5.320308033725668e-06, 0.000955671235716168, -0.10931807345178432, 7.776665057881068, -314.5634801502922, 5539.755822768863])),
    ThermalConductivityLiquid(load_data=False, poly_fit=(63.15, 124.0, [1.480216570198736e-14, -1.0579062797590458e-11, 3.280674520280696e-09, -5.764768809161848e-07, 6.277076059293551e-05, -0.004336498214082066, 0.18560707561089, -4.501802805954049, 47.61060499154285])),
    ],
    ThermalConductivityGases=[ThermalConductivityGas(load_data=False, poly_fit=(111.67, 190.0, [9.856442420122018e-18, -1.166766850419619e-14, 6.0247529665376486e-12, -1.7727942423970953e-09, 3.252242772879454e-07, -3.8104800250279826e-05, 0.002785928728604635, -0.11616844943039516, 2.1217398913337955])),
    ThermalConductivityGas(load_data=False, poly_fit=(63.15, 2000.0, [-1.2990458062959103e-27, 1.2814214036910465e-23, -5.4091093020176117e-20, 1.2856405826340633e-16, -1.9182872623250277e-13, 1.9265592624751723e-10, -1.405720023370681e-07, 0.00012060144315254995, -0.0014802322860337641])),
    ],
    SurfaceTensions=[SurfaceTension(load_data=False, Tc=190.564, exp_poly_fit_ln_tau=(90.67, 188.84, 190.564, [-3.4962367358992735e-05, -0.0008289010193078861, -0.00843287807824276, -0.04779630061606634, -0.16511409618486522, -0.35428823877130233, -0.4686689898988202, 0.8594918266347117, -3.3914597815553464])),
    SurfaceTension(load_data=False, Tc=126.2, exp_poly_fit_ln_tau=(64.8, 120.24, 126.2, [-1.4230749474462855e-08, -1.0305965235744322e-07, -6.754987900429734e-07, -9.296769895431879e-07, -6.091084410916199e-06, 1.0046797865808803e-05, -4.1631671079768105e-05, 1.246078177155456, -3.54114947415937])),
    ],
    )

    gas = IdealGas(HeatCapacityGases=correlations.HeatCapacityGases, Hfs=[-74534.0, 0.0], Gfs=[-50443.48000000001, 0.0], T=298.15, P=101325.0, zs=[0.5, 0.5])


    liquid = GibbsExcessLiquid(GibbsExcessModel=IdealSolution(T=298.15, xs=[0.5, 0.5]),VaporPressures=correlations.VaporPressures, VolumeLiquids=correlations.VolumeLiquids, HeatCapacityGases=correlations.HeatCapacityGases,
                               equilibrium_basis=None, caloric_basis=None, eos_pure_instances=None, Hfs=[-74534.0, 0.0], Gfs=[-50443.48000000001, 0.0], T=298.15, P=101325.0, zs=[0.5, 0.5])


    flasher = FlashVLN(gas=gas, liquids=[liquid, liquid], constants=constants, correlations=correlations)
    res = flasher.flash(P=1.5e5, zs=[.97, .03], VF=1)
    assert_close(res.T, 116.28043156536933)


def test_ethanol_water_cyclohexane_3_liquids():
    zs = [.35, .06, .59]
    # m = Mixture(['ethanol', 'water','cyclohexane'], zs=zs)
    T = 298.15
    P = 1e5

    constants = ChemicalConstantsPackage(Tcs=[514.0, 647.14, 532.7], Pcs=[6137000.0, 22048320.0, 3790000.0], omegas=[0.635, 0.344, 0.213], MWs=[46.06844, 18.01528, 84.15948], CASs=['64-17-5', '7732-18-5', '110-82-7'])

    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                                     HeatCapacityGas(poly_fit=(100.0, 1000.0, [-2.974359561904494e-20, 1.4314483633408613e-16, -2.8871179135718834e-13, 3.1557554273363386e-10, -2.0114283147465467e-07, 7.426722872136983e-05, -0.014718631011050769, 1.6791476987773946, -34.557986234881355]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas)

    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)

    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    res = flashN.flash(T=150, P=1e6, zs=zs)
    assert len(res.liquids) == 3
    assert_close1d(res.heaviest_liquid.zs, [7.29931119445944e-08, 0.9999999270068883, 1.6797752584629787e-26], atol=1e-15)
    assert_close1d(res.lightest_liquid.zs, [0.9938392499953312, 0.0035632064699575947, 0.0025975435347112977])

def test_butanol_water_ethanol_3P():
    zs = [.25, 0.7, .05]
    # m = Mixture(['butanol', 'water', 'ethanol'], zs=zs)

    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0], omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844], CASs=['71-36-3', '7732-18-5', '64-17-5'])
    properties = PropertyCorrelationsPackage(constants=constants,
                                             HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                                            HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                            HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),], )
    T = 298.15
    P = 1e5

    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas)

    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)

    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    # LL solution exists and is found at this point - but does not have as lower of a G
    res = flashN.flash(T=400, P=1e5, zs=zs)
    assert res.gas
    assert res.phase_count == 1

    #  Some bad logic in LL
    failed_logic_LL = flashN.flash(T=186.48648648648657, P=120526.09368708414, zs=zs)
    assert_close1d(failed_logic_LL.betas, [0.6989623717730211, 0.3010376282269789])


    # Test 5 points going  through LL to VL to VLL to VL again to V
    res = flashN.flash(T=354, P=1e5, zs=zs) # LL
    assert_close1d([i.rho_mass() for i in res.phases], [722.1800166595312, 656.8373555931618])
    res = flashN.flash(T=360, P=1e5, zs=zs) # Originally rhough this was VL but it is not
    assert_close(res.G_min_criteria(), -2634.23162820162)
    assert_close1d([i.rho_mass() for i in res.phases], [718.2278806648291, 651.3842378192114])

    res = flashN.flash(T=361, P=1e5, zs=zs) # VLL
    assert_close1d(res.water_phase.zs, [7.619975052224755e-05, 0.9989622883894996, 0.0009615118599771799])
    assert_close1d(res.gas.zs, [0.2384009970908654, 0.57868399351809, 0.18291500939104438])
    assert res.phase_count == 3

    res = flashN.flash(T=364, P=1e5, zs=zs) # VL
    assert_close1d([i.rho_mass() for i in res.phases], [1.203792756430329, 715.8202252076906])
    assert_close(res.G_min_criteria(), -2315.7046889667417)

    res = flashN.flash(T=366, P=1e5, zs=zs) # V
    assert_close1d([i.rho_mass() for i in res.phases], [1.1145608982480968])

def test_VLL_PR_random0_wrong_stab_test():
    # Example was truly random, from a phase plot
    # Needed to change the stability test to try liquid as a third phase first
    constants = ChemicalConstantsPackage(Tcs=[582.0, 400.0, 787.0, 716.4, 870.0], Pcs=[3384255.0, 6000000.0, 4300000.0, 3100000.0, 5501947.0], omegas=[0.4, 0.2791, 0.3855, 0.4892, 0.764], MWs=[114.18545999999999, 52.034800000000004, 143.18516, 122.16439999999999, 138.12404], CASs=['565-80-0', '460-19-5', '611-32-5', '123-07-9', '100-01-6'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(204.15, 582.0, [-4.0299628421346055e-19, 1.4201454382394794e-15, -2.1592100087780104e-12, 1.8386309700739949e-09, -9.479935033660882e-07, 0.00029627483510291316, -0.052529798587495985, 4.852473173045381, -55.11689446455679])),
                                                                                     HeatCapacityGas(poly_fit=(298, 1000, [6.54716256429881e-22, -3.581216791793445e-18, 8.314163057867216e-15, -1.0580587620924602e-11, 7.901083948811444e-09, -3.373497350440364e-06, 0.0006706407999811033, 0.03846325950488974, 32.603549799718024])),
                                                                                     HeatCapacityGas(poly_fit=(193.15, 787.0, [-7.952502901417112e-20, 3.252322049065606e-16, -5.58434715614826e-13, 5.171000969478227e-10, -2.7327189358559686e-07, 7.802319674015784e-05, -0.00906733523259663, 0.0912003293438394, 125.70159427955822])),
                                                                                     HeatCapacityGas(poly_fit=(298.0, 1000.0, [1.3069175742586603e-20, -7.47359807737359e-17, 1.846534761068117e-13, -2.5729525532348086e-10, 2.2085482406550317e-07, -0.0001191457063338021, 0.03890101117609332, -6.504636110790599, 534.0594709902953])),
                                                                                     HeatCapacityGas(poly_fit=(421.15, 870.0, [7.418364464492825e-21, -4.2162303084705005e-17, 1.045156469429536e-13, -1.474360261831268e-10, 1.291049693608951e-07, -7.13282925037787e-05, 0.02365778501014936, -3.7512647021380467, 299.6902129804673]))])
    zs = [.2, .2, .2, .2, .2]

    T, P = 313.0156681953369, 115139.53993264504
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    res = flashN.flash(T=T, P=P, zs=zs)
    assert res.phase_count == 3
    assert_close(res.betas[0], 0.09417025397035537)
    zs_heavy = res.heaviest_liquid.zs
    zs_heavy_expect = [0.01668931264853162, 0.05103372454391458, 0.09312148233246491, 0.013032991760234687, 0.8261224887148544]
    assert_close1d(zs_heavy, zs_heavy_expect, rtol=1e-5)

def test_LLL_PR_random1_missing_LLL_trivial_gas():
    constants = ChemicalConstantsPackage(Tcs=[676.0, 653.0, 269.0, 591.3, 575.6], Pcs=[4000000.0, 14692125.0, 4855494.0, 3880747.0, 3140000.0], omegas=[0.7206, 0.32799999999999996, 0.1599, 0.3691, 0.4345], MWs=[90.121, 32.04516, 32.11726, 104.21378, 116.15828], CASs=['107-88-0', '302-01-2', '7803-62-5', '628-29-5', '123-86-4'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(196.15, 676.0, [-1.6703777121140357e-19, 6.336563235116343e-16, -1.0268904098033873e-12, 9.209775070752545e-10, -4.925751560244259e-07, 0.00015625155455794686, -0.027101271359103958, 2.363660193690524, 5.925941687513358])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.623312561584884e-21, 1.6721115187937393e-17, -3.182771170056967e-14, 3.194692211808055e-11, -1.774280702831314e-08, 5.047063059999335e-06, -0.0005017671060833166, 0.03937920538251642, 32.21829961922973])),
                                                                                     HeatCapacityGas(poly_fit=(88.15, 269.0, [-7.353710707470991e-18, 9.770203085327697e-15, -5.192485575504864e-12, 1.3663121283879677e-09, -1.7426822740884812e-07, 7.875818199129791e-06, 0.00041232162952700355, -0.054997039960075225, 24.92658630549439])),
                                                                                     HeatCapacityGas(poly_fit=(273, 1000, [1.321014390602642e-21, -8.34388600397224e-18, 2.3098226226797758e-14, -3.6634048438548e-11, 3.636963615039002e-08, -2.295930412929758e-05, 0.008674670844432155, -1.4111475354201926, 182.2898224404806])),
                                                                                     HeatCapacityGas(poly_fit=(195.15, 575.6, [-4.4626862569926e-19, 1.5426193954174004e-15, -2.2990858376078466e-12, 1.9176735381876706e-09, -9.677520492563025e-07, 0.0002957533403178217, -0.051211100401858944, 4.615337526886703, -64.55339891849962]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    zs = [.2, .2, .2, .2, .2]
    T = 150.0
    P=381214.0
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    res = flashN.flash(T=T, P=P, zs=zs)
    assert 3 == res.phase_count
    assert res.gas is None

    zs_heavy = [5.663363025793038e-08, 0.9999982518756677, 1.6914852598251962e-06, 5.442002170065923e-12, 2.849791556763424e-16]
    assert_close1d(res.heaviest_liquid.zs, zs_heavy, rtol=1e-4)
    zs_lightest = [0.5958582108826934, 0.005355183270131757, 0.14526839894298765, 0.16882402594657384, 0.08469418095761332]
    assert_close1d(res.lightest_liquid.zs, zs_lightest, rtol=1e-5)


def test_LLL_random2_decrease_tolerance_similar_comp_stab():
    # Increased diff2 to 0.02
    zs = [1.0/6]*6
    T, P = 172.0, 6.9e6
    constants = ChemicalConstantsPackage(Tcs=[796.0, 530.9, 900.0, 597.6, 745.2, 712.0], Pcs=[600000.0, 3530000.0, 3000000.0, 5280000.0, 3538830.0, 1400000.0], omegas=[1.2135, 0.373, 0.4551, 0.4411, 0.5441, 0.6737], MWs=[422.81328, 168.064156, 206.28236, 76.09442, 174.15614, 214.34433999999996], CASs=['111-01-3', '363-72-4', '781-17-9', '109-86-4', '584-84-9', '111-82-0'])
    # heat capacities are made up
    correlations = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(196.15, 676.0, [-1.6703777121140357e-19, 6.336563235116343e-16, -1.0268904098033873e-12, 9.209775070752545e-10, -4.925751560244259e-07, 0.00015625155455794686, -0.027101271359103958, 2.363660193690524, 5.925941687513358])),
                                                                                       HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.623312561584884e-21, 1.6721115187937393e-17, -3.182771170056967e-14, 3.194692211808055e-11, -1.774280702831314e-08, 5.047063059999335e-06, -0.0005017671060833166, 0.03937920538251642, 32.21829961922973])),
                                                                                       HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.623312561584884e-21, 1.6721115187937393e-17, -3.182771170056967e-14, 3.194692211808055e-11, -1.774280702831314e-08, 5.047063059999335e-06, -0.0005017671060833166, 0.03937920538251642, 32.21829961922973])),
                                                                                       HeatCapacityGas(poly_fit=(88.15, 269.0, [-7.353710707470991e-18, 9.770203085327697e-15, -5.192485575504864e-12, 1.3663121283879677e-09, -1.7426822740884812e-07, 7.875818199129791e-06, 0.00041232162952700355, -0.054997039960075225, 24.92658630549439])),
                                                                                       HeatCapacityGas(poly_fit=(273, 1000, [1.321014390602642e-21, -8.34388600397224e-18, 2.3098226226797758e-14, -3.6634048438548e-11, 3.636963615039002e-08, -2.295930412929758e-05, 0.008674670844432155, -1.4111475354201926, 182.2898224404806])),
                                                                                       HeatCapacityGas(poly_fit=(195.15, 575.6, [-4.4626862569926e-19, 1.5426193954174004e-15, -2.2990858376078466e-12, 1.9176735381876706e-09, -9.677520492563025e-07, 0.0002957533403178217, -0.051211100401858944, 4.615337526886703, -64.55339891849962]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, correlations, liquids=[liq, liq], gas=gas)

    res = flashN.flash(T=T, P=P, zs=zs)
    zs_light_expect = [0.7405037446267929, 0.021794147548270967, 0.0001428606542343683, 0.0006934327486605817, 0.00010382544098221231, 0.23676198898105874]
    assert_close1d(res.lightest_liquid.zs, zs_light_expect, rtol=1e-4)
    zs_heavy_expect = [1.6221123733069825e-20, 0.22762091600367182, 0.2569470513971517, 0.2570722261924367, 0.2581212217153263, 0.0002385846914132562]
    assert_close1d(res.heaviest_liquid.zs, zs_heavy_expect, rtol=1e-4, atol=1e-10)
    assert 3 == res.phase_count



def test_VLLL_water_C1_C8_iron():
    T = 300
    P = 1e5
    N = 4
    names = ['water', 'methane', 'octane', 'iron']
    zs = [1.0/N]*N
    constants = ChemicalConstantsPackage(Tcs=[647.14, 190.56400000000002, 568.7, 9340.0], Pcs=[22048320.0, 4599000.0, 2490000.0, 1015000000.0], omegas=[0.344, 0.008, 0.39399999999999996, -0.0106], MWs=[18.01528, 16.04246, 114.22852, 55.845], CASs=['7732-18-5', '74-82-8', '111-65-9', '7439-89-6'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303])),
                                                                                     HeatCapacityGas(poly_fit=(1811.15, 9340.0, [-1.5383706701397728e-29, 7.680207832265723e-25, -1.6470428514880497e-20, 1.9827016753030861e-16, -1.468410412876826e-12, 6.88624351096193e-09, -2.019180028341396e-05, 0.03467141804375052, 61.33135714266571]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq, liq], gas=gas)

    res = flashN.flash(T=T, P=P, zs=zs)
    zs_water = [0.9999990592468204, 9.407531795083966e-07, 1.4320610523053124e-16, 2.6913157388219106e-61]
    assert_close1d(res.water_phase.zs, zs_water, atol=1e-10, rtol=1e-5)
    # print(res.water_phase.zs)
    zs_gas = [0.03040532892726872, 0.9468785790243984, 0.022716092048333275, 7.628086086087928e-78]
    assert_close1d(res.gas.zs, zs_gas, atol=1e-10, rtol=1e-5)
    assert_close1d(res.heaviest_liquid.zs, [0, 0, 0, 1], atol=1e-20)
    zs_light = [0.01803214863396999, 0.005375612317521824, 0.976592239048508, 6.778321658412194e-74]
    assert_close1d(res.lightest_liquid.zs, zs_light, rtol=1e-5, atol=1e-10)

def test_VLLLL_first():
    # Lots of bugs/confusion here, but it works
    T = 300
    P = 1e5
    N = 5
    zs = [1.0/N]*N
    constants = ChemicalConstantsPackage(Tcs=[647.14, 190.56400000000002, 568.7, 632.0, 9340.0], Pcs=[22048320.0, 4599000.0, 2490000.0, 5350000.0, 1015000000.0], omegas=[0.344, 0.008, 0.39399999999999996, 0.7340000000000001, -0.0106], MWs=[18.01528, 16.04246, 114.22852, 98.09994, 55.845], CASs=['7732-18-5', '74-82-8', '111-65-9', '98-00-0', '7439-89-6'], names=['water', 'methane', 'octane', 'furfuryl alcohol', 'iron'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303])),
                                                                                     HeatCapacityGas(poly_fit=(250.35, 632.0, [-9.534610090167143e-20, 3.4583416772306854e-16, -5.304513883184021e-13, 4.410937690059558e-10, -2.0905505018557675e-07, 5.20661895325169e-05, -0.004134468659764938, -0.3746374641720497, 114.90130267531933])),
                                                                                     HeatCapacityGas(poly_fit=(1811.15, 9340.0, [-1.5383706701397728e-29, 7.680207832265723e-25, -1.6470428514880497e-20, 1.9827016753030861e-16, -1.468410412876826e-12, 6.88624351096193e-09, -2.019180028341396e-05, 0.03467141804375052, 61.33135714266571]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq, liq, liq], gas=gas)
    res = flashN.flash(T=T, P=P, zs=zs)
    assert res.phase_count == 5


def test_problem_1_Ivanov_CH4_H2S():
    T = 190
    P = 40.53E5
    IDs = ['methane', 'hydrogen sulfide']
    zs = [.5, .5]
    kijs = [[0.0, .08],[0.08, 0.0]]
    Tcs = [190.6, 373.2]
    Pcs = [46e5, 89.4e5]
    omegas = [0.008, .1]
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[16.04246, 34.08088], CASs=['74-82-8', '7783-06-4'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534]))], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq, liq], gas=gas)

    # Worked fine
    res = flashN.flash(T=T, P=P, zs=[.9885, .0115])
    assert res.gas is not None
    assert res.phase_count == 1
    res = flashN.flash(T=T, P=P, zs=[.9813, .0187])
    assert res.gas is not None
    assert res.phase_count == 2
    betas_expect = [0.9713327921217155, 0.028667207878284473]
    assert_close1d(res.betas, betas_expect, rtol=1e-5)
    res = flashN.flash(T=T, P=P, zs=[0.93, 0.07])
    assert res.liquid0 is not None
    assert res.phase_count == 1

    # 0.5, 0.5 did not work originally - there is a LL and a VL solution
    # The VL solution's stab test guess looks quite reasonable
    # Bugs that appeared when implementing DOUBLE_CHECK_2P
    flashN.DOUBLE_CHECK_2P = True
    res = flashN.flash(T=100.0, P=10000.0, zs=[.5, .5])
    rhos_expect = [0.19372650512324788, 1057.7834954440648]
    assert_close1d(rhos_expect, [i.rho_mass() for i in res.phases], rtol=1e-5)

    # flashN.flash(T=100.0, P=74989.42093324558, zs=[.5, .5]) should be tested - cycle - but is not because slow
    # flashN.flash(T=100.0, P=133352.1432163324, zs=[.5, .5]) # should also be tested - unconverged error

    # Actual point, used to be VL not is correctly LL with DOUBLE_CHECK_2P
    res = flashN.flash(T=T, P=P, zs=[.5, .5])
    assert res.phase_count == 2
    rhos_expect = [877.5095175225118, 275.14553695914424]
    assert_close1d(rhos_expect, [i.rho_mass() for i in res.phases])
    flashN.DOUBLE_CHECK_2P = False

    res = flashN.flash(T=T, P=P, zs=[.112, .888])
    rhos_expect = [877.5095095098309, 275.14542288743456]
    assert_close1d(rhos_expect, [i.rho_mass() for i in res.phases])
    assert_close1d(res.betas, [0.9992370056503082, 0.0007629943496918543], rtol=1e-5, atol=1e-6)

    res = flashN.flash(T=T, P=P, zs=[.11, .89])
    assert res.liquid0 is not None
    assert res.phase_count == 1

def test_problem_2_Ivanov_CH4_propane():
    # Basic two phase solution tests, only VL
    T = 277.6
    P = 100e5
    IDs = ['methane', 'propane']
    zs = [.4, .6]
    Tcs = [190.6, 369.8]
    Pcs = [46e5, 42.5e5]
    omegas = [.008, .152]

    kijs = [[0,0.029],[0.029,0]]

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[16.04246, 44.09562], CASs=['74-82-8', '74-98-6'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])), ], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    # Should converge to two phase solution
    # Does not because trivial point converges instead
    # Fixd by comparing the composition of the spawned phase to the feed
    res = flashN.flash(T=221.2346408571431, P=5168385.0, zs=[.9, .1])
    res.flash_convergence, res, #res.gas.zs, res.liquid0.zs
    assert_close1d(res.betas, [0.8330903352232165, 0.16690966477678348], rtol=1e-5)
    rhos_expect = [70.51771075659963, 423.8157580854203]
    assert_close1d([i.rho_mass() for i in res.phases], rhos_expect)


def test_problem_3_Ivanov_N2_ethane():
    # Basic two phase solution tests, only VL
    T = 270.0
    P = 76e5
    zs = [.1, .9]
    kijs = [[0.0,0.08],[0.08,0.0]]
    Tcs = [126.2, 305.4]
    Pcs = [33.9E5, 48.8E5]
    omegas = [0.04, 0.098]
    IDs = ['nitrogen', 'ethane']
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[28.0134, 30.06904], CASs=['7727-37-9', '74-84-0'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228]))], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)


    res = flashN.flash(T=T, P=P, zs=[.1, .9])
    assert 1 == res.phase_count
    res = flashN.flash(T=T, P=P, zs=[.18, .82])
    assert 2 == res.phase_count
    assert 1 == res.liquid_count
    res = flashN.flash(T=T, P=P, zs=[.3, .7])
    assert 2 == res.phase_count
    assert 1 == res.liquid_count
    res = flashN.flash(T=T, P=P, zs=[.44, .56])
    assert 2 == res.phase_count
    assert 1 == res.liquid_count
    res = flashN.flash(T=T, P=P, zs=[.6, .4])
    assert 1 == res.phase_count

def test_problem_4_Ivanov_CO2_CH4():
    T = 220.0
    P = 60.8e5
    zs = [.43, .57]
    kijs = [[0.0, 0.095], [0.095, 0.0]]
    Tcs = [304.2, 190.6]
    Pcs = [73.8E5, 46.0E5]
    omegas = [0.225, 0.008]
    IDs = ['CO2', 'methane']
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[44.0095, 16.04246], CASs=['124-38-9', '74-82-8'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228]))], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq], gas=gas)

    # Didn't find anything wrong, but here is a point I wanted to double confirm
    # done with evo algorithm
    res = flashN.flash(T=T, P=P, zs=[.43, .57])
    betas_expect = [0.04855003898294571, 0.9514499610170543]
    assert_close1d(betas_expect, res.betas, rtol=1e-5)
    assert res.gas is not None

def test_problem_5_Ivanov_N2_CH4_Ethane():
    # Works great!
    IDs = ['nitrogen', 'methane', 'ethane']
    T = 270.0
    P = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08],
            [0.038, 0, 0.021],
            [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                         MWs=[28.0134, 16.04246, 30.06904],
                                         CASs=['7727-37-9', '74-82-8', '74-84-0'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    res = flashN.flash(T=T, P=P, zs=[0.3, 0.1, 0.6])
    assert res.phase_count == 2
    assert res.gas is not None
    res = flashN.flash(T=T, P=P, zs=[.15, .3, .55])
    assert res.phase_count == 2
    assert res.gas is not None
    res = flashN.flash(T=T, P=P, zs=[.08, .38, .54])
    assert res.phase_count == 1
    assert res.liquid0 is not None
    res = flashN.flash(T=T, P=P, zs=[.05, .05, .9])
    assert res.phase_count == 1
    assert res.liquid0 is not None

def test_problem_6_Ivanov_CH4_CO2_H2S():
    IDs = ['methane', 'CO2', 'H2S']
    T = 300.0
    P = 1e5
    Tcs = [190.6, 304.2, 373.2]
    Pcs = [46.0E5, 73.8E5, 89.4E5]
    omegas = [0.008, 0.225, 0.1]
    kijs = [[0.0, 0.095, 0.0755], [0.095, 0.0, 0.0999], [0.0755, 0.0999, 0.0]]
    zs = [.3, .3, .4]

    Ts = [280.5, 210.5, 210.5, 227.5]
    Ps = [55.1e5, 57.5e5, 57.5e5, 48.6e5]
    zs_list = ([0.4989, 0.0988, 0.4023], [0.4989, 0.0988, 0.4023],
               [0.48, 0.12, 0.4], [0.4989, 0.0988, 0.4023])

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[16.04246, 44.0095, 34.08088], CASs=['74-82-8', '124-38-9', '7783-06-4'])

    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    # There is a good two phase solution very nearly the same Gibbs but not quite
    flashN.SS_NP_STAB_HIGHEST_COMP_DIFF = True
    res = flashN.flash(T=137.17862396147322, P=562341.3251903491, zs=zs_list[1])
    assert res.gas is not None
    assert res.phase_count == 3
    assert_close(res.lightest_liquid.rho_mass(), 1141.0138231355213)
    assert_close(res.heaviest_liquid.rho_mass(), 1197.8947471324532)

    # Another point with a similar issue, same fix fixes it; otherwise the first three phase
    # solution converges to a negative beta; tries to form a new gas I think
    res = flashN.flash(T=120, P=181659.97883753508, zs=zs_list[0])
    assert res.gas is not None
    assert res.phase_count == 3
    assert_close(res.lightest_liquid.rho_mass(), 1169.2257099718627)
    assert_close(res.heaviest_liquid.rho_mass(), 1423.134620262134)


def test_problem_7_Ivanov_CH4_CO2_C6_H2S():
    IDs = ['methane', 'carbon dioxide', 'n-hexane', 'H2S']
    T = 200.0
    P = 42.5E5
    Tcs = [190.6, 304.2, 507.4, 373.2]
    Pcs = [46.0E5, 73.8E5, 29.678E5, 89.4E5]
    omegas = [0.008, 0.225, 0.296, 0.1]
    kijs = [[0.0, 0.12, 0.0, 0.08],
            [.12, 0.0, 0.12, 0.0],
            [0.0, 0.12, 0.0, 0.06],
            [.08, 0.0, .06, 0.0]]
    zs = [0.5000, 0.0574, 0.0263, 0.4163]

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                         MWs=[16.04246, 44.0095, 86.17536, 34.08088],
                                         CASs=['74-82-8', '124-38-9', '110-54-3', '7783-06-4'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)

    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    # Test has only one point
    # Failed to make the third phase
    res = flashN.flash(T=T, P=P, zs=zs)
    assert_close(res.G(), 980.5900455260398, rtol=1e-6)
    assert res.phase_count == 3

def test_problem_8_Ivanov():
    IDs = ['nitrogen', 'methane', 'ethane', 'propane',
           'n-butane', 'n-pentane']
    T = 150.0
    P = 40.52E5
    zs = [0.3040, 0.5479, 0.0708, 0.0367, 0.0208, 0.0198]

    Tcs = [126.2, 190.6, 305.4, 369.8, 425.2, 469.6]
    Pcs = [33.9E5, 46.0E5, 48.8E5, 42.5E5, 37.987E5, 33.731E5]
    omegas = [0.04, 0.008, 0.098, 0.152, 0.193, 0.251]
    MWs = [28.0134, 16.04246, 30.06904, 44.09562, 58.1222, 72.14878]
    CASs = ['7727-37-9', '74-82-8', '74-84-0', '74-98-6', '106-97-8', '109-66-0']
    kijs = [[0, 0.02, 0.06, 0.08, 0.08, 0.08],
    [0.02, 0, 0, 0.029, 0, 0],
    [0.06, 0, 0, 0, 0, 0],
    [0.08, 0.029, 0, 0, 0, 0],
    [0.08, 0, 0, 0, 0, 0],
    [0.08, 0, 0, 0, 0, 0]]
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=MWs, CASs=CASs)
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    # LL point
    res = flashN.flash(T=T, P=P, zs=zs)
    rhos_expect = [507.6192989020208, 458.7217528598331]
    assert_close1d(rhos_expect, [i.rho_mass() for i in res.phases], rtol=1e-5)
    assert res.liquid1 is not None
    assert_close1d(res.betas, [0.43700616001533293, 0.562993839984667], rtol=1e-5)

    # Extremely stupid point - 3000 iterations, G is 1E-7 relative lower with three phase than 2
    # Solution: Acceleration
    res = flashN.flash(T=128.021, P=1.849e6, zs=zs)
    assert res.phase_count == 3

def test_problem_9_Ivanov():
    IDs = ['methane', 'ethane', 'propane', 'n-butane',
           'n-pentane', 'n-hexane', 'C7_16', 'C17+']
    CASs = ['74-82-8', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '110-54-3', None, None]

    T = 353
    P = 385e5
    zs = [0.6883, 0.0914, 0.0460, 0.0333, 0.0139, 0.0152, 0.0896, 0.0223]
    Tcs = [190.6, 305.4, 369.8, 425.2, 469.6, 507.4, 606.28, 825.67]
    Pcs = [46000000, 48800000, 42500000, 37987000, 33731000, 29678000, 25760000, 14580000]
    omegas = [0.008, 0.098, 0.152, 0.193, 0.251, 0.296, 0.4019, 0.7987]

    MWs = [16.04246, 30.06904, 44.09562, 58.1222, 72.14878, 86.17536, 120, 400]
    kijs = [[0]*8 for i in range(8)]

    kijs = [[0, .021, 0, 0, 0, 0, .05, .09],
     [.021, 0, 0, 0, 0, 0, .04, 0.055],
     [0, 0, 0, 0, 0, 0, .01, .01],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0],
     [.05, .04, .01, 0, 0, 0, 0, 0],
     [0.09, 0.055, .01, 0, 0, 0, 0, 0]]
    # made up MWs,; C6 heat capacities used onwards
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=MWs, CASs=CASs)
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    # Could not get a three phase system out of it
    res = flashN.flash(T=T, P=P, zs=zs)
    assert_close1d(res.betas, [0.8263155779926616, 0.17368442200733836], rtol=1e-5)


def test_problem_10_Ivanov():

    IDs = ['methane', 'ethane', 'propane', 'i-butane', 'n-butane',
           'i-pentane', 'n-pentane', 'n-hexane', '2-Methyltetradecane']
    CASs = ['74-82-8', '74-84-0', '74-98-6', '75-28-5', '106-97-8', '78-78-4', '109-66-0', '110-54-3', '1560-95-8']
    T = 314.0
    P = 20.1E5
    zs = [0.61400, 0.10259, 0.04985, 0.00898, 0.02116, 0.00722, 0.01187, 0.01435, 0.16998]
    MWs = [16.04246, 30.06904, 44.09562, 58.1222, 58.1222, 72.14878, 72.14878, 86.17536, 212.41458]
    Tcs = [190.6, 305.4, 369.8, 425.2, 407.7, 469.6, 461, 507.4, 708.2]
    Pcs = [4600000, 4880000, 4250000, 3798700, 3650000, 3373100, 3380000, 2967800, 1500000]
    omegas = [0.008, 0.098, 0.152, 0.193, 0.176, 0.251, 0.227, 0.296, 0.685]

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=MWs, CASs=CASs)
    # 2-Methyltetradecane has same Cp as C6, lack of data
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.964580174158493e-21, -3.0000649905010227e-17, 5.236546883165256e-14, -4.72925510291381e-11, 2.3525503498476575e-08, -6.547240983119115e-06, 0.001017840156793089, 0.1718569854807599, 23.54205941874457])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-3.3940692195566165e-21, 1.7702350857101215e-17, -3.908639931077815e-14, 4.7271630395909585e-11, -3.371762950874771e-08, 1.4169051870903116e-05, -0.003447090979499268, 0.7970786116058249, -8.8294795805881])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998]))])

    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)  # kijs all zero
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    res = flashN.flash(T=T, P=P, zs=zs)
    assert_close1d(res.betas, [0.730991626075863, 0.269008373924137], rtol=1e-5)
    assert_close1d([i.rho_mass() for i in res.phases], [16.370044743837063, 558.9020870173408], rtol=1e-5)


def test_problem_11_Ivanov():
    T = 425.14
    P = 50e5
    zs = [0.6436, 0.0752, 0.0474, 0.0412, 0.0297, 0.0138, 0.0303, 0.0371, 0.0415, 0.0402]
    IDs = ['methane', 'ethane', 'propane', 'n-butane', 'n-pentane', 'n-hexane', 'heptane', 'n-octane', 'n-nonane', 'n-decane']
    Tcs = [190.6, 305.4, 369.8, 425.2, 469.6, 507.4, 540.2, 568.8, 594.6, 617.6]
    Pcs = [4600000, 4880000, 4250000, 3798700, 3373100, 2967800, 2736000, 2482500, 2310000, 2107600]
    omegas = [0.008, 0.098, 0.152, 0.193, 0.251, 0.296, 0.351, 0.394, 0.444, 0.49]

    MWs = [16.04246, 30.06904, 44.09562, 58.1222, 72.14878, 86.17536, 100.20194000000001, 114.22852, 128.2551, 142.28168]
    CASs = ['74-82-8', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '110-54-3', '142-82-5', '111-65-9', '111-84-2', '124-18-5']

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=MWs, CASs=CASs)
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.4046935863496273e-21, 5.8024177500786575e-18, -7.977871529098155e-15, 7.331444047402207e-13, 9.954400606484495e-09, -1.2112107913343475e-05, 0.0062964696142858104, -1.0843106737278825, 173.87692850911935])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [6.513870466670624e-22, -5.318305817618858e-18, 1.8015815307749625e-14, -3.370046452151828e-11, 3.840755097595374e-08, -2.7203677889897072e-05, 0.011224516822410626, -1.842793858054514, 247.3628627781443])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735]))])
    # kijs all zero
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    res = flashN.flash(T=T, P=P, zs=zs)
    assert_close1d(res.betas, [0.832572052180253, 0.16742794781974701], rtol=1e-5)
    rhos_expect = [41.59446117573929, 472.28905699434006]
    assert_close1d([i.rho_mass() for i in res.phases], [41.59446117573929, 472.28905699434006], rtol=1e-5)


def test_problem_12_Ivanov():
    IDs = ['methane', 'ethane', 'propane', 'i-butane', 'n-butane',
           'i-pentane', 'n-pentane', 'n-hexane', 'heptane', 'n-octane', 'nitrogen', 'CO2']
    Tcs = [190.6, 305.4, 369.8, 425.2, 407.7, 469.6, 461, 507.4, 540.2, 568.8, 126.2, 304.2]
    Pcs = [4600000, 4880000, 4250000, 3798700, 3650000, 3373100, 3380000, 2967800, 2736000, 2482500, 3390000, 7380000]
    omegas = [0.008, 0.098, 0.152, 0.193, 0.176, 0.251, 0.227, 0.296, 0.351, 0.394, 0.04, 0.225]
    MWs = [16.04246, 30.06904, 44.09562, 58.1222, 58.1222, 72.14878, 72.14878, 86.17536, 100.20194000000001, 114.22852, 28.0134, 44.0095]
    CASs = ['74-82-8', '74-84-0', '74-98-6', '75-28-5', '106-97-8', '78-78-4', '109-66-0', '110-54-3', '142-82-5', '111-65-9', '7727-37-9', '124-38-9']
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=MWs, CASs=CASs)
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.964580174158493e-21, -3.0000649905010227e-17, 5.236546883165256e-14, -4.72925510291381e-11, 2.3525503498476575e-08, -6.547240983119115e-06, 0.001017840156793089, 0.1718569854807599, 23.54205941874457])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-3.3940692195566165e-21, 1.7702350857101215e-17, -3.908639931077815e-14, 4.7271630395909585e-11, -3.371762950874771e-08, 1.4169051870903116e-05, -0.003447090979499268, 0.7970786116058249, -8.8294795805881])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.4046935863496273e-21, 5.8024177500786575e-18, -7.977871529098155e-15, 7.331444047402207e-13, 9.954400606484495e-09, -1.2112107913343475e-05, 0.0062964696142858104, -1.0843106737278825, 173.87692850911935])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644]))])
    # kijs all zero
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

    T = 240.0
    P = 60e5
    zs = [0.833482, 0.075260, 0.020090, 0.003050, 0.005200, 0.001200, 0.001440, 0.000680, 0.000138, 0.000110, 0.056510, 0.002840]
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    res = flashN.flash(T=T, P=P, zs=zs)
    assert_close1d(res.betas, [0.9734828879098508, 0.026517112090149175], atol=1e-7, rtol=1e-5)
    rhos_expect = [82.83730342326865, 501.1504212828749]
    assert_close1d([i.rho_mass() for i in res.phases], rhos_expect, rtol=1e-5)


def test_phases_at():
    T = 300
    P = 1e5
    zs = [.8, 0.2]
    constants = ChemicalConstantsPackage(Tcs=[282.34, 514.0], Pcs=[5041000.0, 6137000.0], omegas=[0.085, 0.635],
                                         MWs=[28.05316, 46.06844], CASs=['74-85-1', '64-17-5'])
    properties =PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.2701693466919565e-21, 1.660757962278189e-17, -3.525777713754962e-14, 4.01892664375958e-11, -2.608749347072186e-08, 9.23682495982131e-06, -0.0014524032651835623, 0.09701764355901257, 31.034399100170667])),
                                                                                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965]))])

    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas, kijs=[[0.0, -.0057], [-.0057, 0.0]])

    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq2 = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq3 = CEOSLiquid(VDWMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)

    flashN = FlashVLN(constants, properties, liquids=[liq2, liq3, liq, liq3, liq, liq2, liq2], gas=gas)
    assert 3 == len({id(i) for i in flashN.phases_at(T=T, P=P, zs=zs)[1]})

def test_VLL_handles_one_phase_only():
    constants = ChemicalConstantsPackage(atomss=[{'N': 2}, {'O': 2}, {'Ar': 1}, {'H': 2, 'O': 1}], CASs=['7727-37-9', '7782-44-7', '7440-37-1', '7732-18-5'], MWs=[28.0134, 31.9988, 39.948, 18.01528], names=['nitrogen', 'oxygen', 'argon', 'water'], omegas=[0.04, 0.021, -0.004, 0.344], Pcs=[3394387.5, 5042945.25, 4873732.5, 22048320.0], Tcs=[126.2, 154.58, 150.8, 647.14])

    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
                                               HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
    HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
    HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.0939921922581918e-31, 4.144146614006628e-28, -6.289942296644484e-25, 4.873620648503505e-22, -2.0309301195845294e-19, 4.3863747689727484e-17, -4.29308508081826e-15, 20.786156545383236])),
    HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
    ])
    gas = IdealGas(HeatCapacityGases=correlations.HeatCapacityGases, T=298.15, P=101325.0, zs=[.25, .25, .25, .25])
    flasher = FlashVLN(constants, correlations, liquids=[], gas=gas)
    res = flasher.flash(T=300, P=1e5, zs=[.25, .25, .25, .25])
    assert res.phase_count == 1
    assert_close(res.rho_mass(), 1.1824324014080023)

    # TODO: Test PH, etc.

def write_PT_plot(fig, eos, IDs, zs, flashN):
    # Helper function for PT plotting
    path = os.path.join(flashN_surfaces_dir, 'PT', 'Cubic')
    if not os.path.exists(path):
        os.makedirs(path)

    key = '{} - {} - {} - {} liquids'.format(eos.__name__, ', '.join(IDs), ', '.join('%g' %zi for zi in zs), len(flashN.liquids))
    fig.savefig(os.path.join(path, key + '.png'))
    plt.close()


def test_PT_plot_works():
    # Do a small grid test to prove the thing is working.
    IDs = ['butanol', 'water', 'ethanol']
    zs = [.25, 0.7, .05]
    # m = Mixture(['butanol', 'water', 'ethanol'], zs=zs)
    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0], omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844], CASs=['71-36-3', '7732-18-5', '64-17-5'])
    properties = PropertyCorrelationsPackage(constants=constants,
                                             HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                                            HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                            HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=298.15, P=1e5, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=298.15, P=1e5, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    values = flashN.debug_PT(zs=zs, Tmin=300, Tmax=600, Pmin=1e5, Pmax=3e7, pts=3, verbose=False, show=False, values=True)
    assert_close1d(values[0], [300., 424.2640687119287, 600], rtol=1e-12)
    assert_close1d(values[1], [100000.0, 1732050.8075688777, 30000000.00000001], rtol=1e-12)
    assert str(values[2]) == "[['LL', 'LL', 'LL'], ['V', 'LL', 'LL'], ['V', 'V', 'L']]"



@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [PRMIX, SRKMIX, VDWMIX]) # eos_mix_list
def test_PT_plot_butanol_water_ethanol(eos):
    IDs = ['butanol', 'water', 'ethanol']
    zs = [.25, 0.7, .05]
    # m = Mixture(['butanol', 'water', 'ethanol'], zs=zs)
    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0], omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844], CASs=['71-36-3', '7732-18-5', '64-17-5'])
    properties = PropertyCorrelationsPackage(constants=constants,
                                             HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                                            HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                            HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=298.15, P=1e5, zs=zs)
    liq = CEOSLiquid(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=298.15, P=1e5, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    fig = flashN.debug_PT(zs=zs, Tmin=300, Tmax=600, Pmin=1e5, Pmax=3e7, pts=25, verbose=False, show=False, values=False)
    write_PT_plot(fig, eos, IDs, zs, flashN)


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [PRMIX]) # eos_mix_list
def test_PT_plot_ethanol_water_cyclohexane(eos):
    IDs = ['ethanol', 'water', 'cyclohexane']
    zs = [.35, .06, .59]
    # m = Mixture(['ethanol', 'water','cyclohexane'], zs=zs)
    constants = ChemicalConstantsPackage(Tcs=[514.0, 647.14, 532.7], Pcs=[6137000.0, 22048320.0, 3790000.0], omegas=[0.635, 0.344, 0.213], MWs=[46.06844, 18.01528, 84.15948], CASs=['64-17-5', '7732-18-5', '110-82-7'])

    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                                     HeatCapacityGas(poly_fit=(100.0, 1000.0, [-2.974359561904494e-20, 1.4314483633408613e-16, -2.8871179135718834e-13, 3.1557554273363386e-10, -2.0114283147465467e-07, 7.426722872136983e-05, -0.014718631011050769, 1.6791476987773946, -34.557986234881355]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=298.15, P=1e5, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=298.15, P=1e5, zs=zs)

    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    fig = flashN.debug_PT(zs=zs, Tmin=175, Tmax=700, Pmin=1e4, Pmax=2.5e7, pts=50, verbose=False, show=False, values=False)
    write_PT_plot(fig, eos, IDs, zs, flashN)


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [PRMIX]) # eos_mix_list
def test_PT_plot_C1_to_C5_water_gas(eos):
    IDs = ['methane', 'ethane', 'propane', 'butane', 'pentane', 'water', 'nitrogen', 'carbon dioxide', 'hydrogen sulfide']
    zs = normalize([.65, .13, .09, .05, .03, .03, .02, .003, 1e-6])
    T = 300.0
    P = 3000e3
    constants = ChemicalConstantsPackage(Tcs=[190.56400000000002, 305.32, 369.83, 425.12, 469.7, 647.14, 126.2, 304.2, 373.2],
                                            Pcs=[4599000.0, 4872000.0, 4248000.0, 3796000.0, 3370000.0, 22048320.0, 3394387.5, 7376460.0, 8936865.0],
                                            omegas=[0.008, 0.098, 0.152, 0.193, 0.251, 0.344, 0.04, 0.2252, 0.1],
                                            MWs=[16.04246, 30.06904, 44.09562, 58.1222, 72.14878, 18.01528, 28.0134, 44.0095, 34.08088],
                                            CASs=['74-82-8', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '7732-18-5', '7727-37-9', '124-38-9', '7783-06-4'],)
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
                                                                                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534])), ])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    fig = flashN.debug_PT(zs=zs, Tmin=175, Tmax=430, Pmin=1e4, Pmax=3.5e7, pts=150, verbose=False, show=False, values=False)
    write_PT_plot(fig, eos, IDs, zs, flashN)

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [PRMIX]) # eos_mix_list
def test_PT_plot_water_C1_C8(eos):
    IDs = ['water', 'methane', 'octane']
    T = 298.15
    P = 101325.0
    omegas = [0.344, 0.008, 0.394]
    Tcs = [647.14, 190.564, 568.7]
    Pcs = [22048320.0, 4599000.0, 2490000.0]
    kijs=[[0,0, 0],[0,0, 0.0496], [0,0.0496,0]]
    zs = [1.0/3.0]*3
    N = len(zs)
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                         HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                         HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303]))]
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[18.01528, 16.04246, 114.22852],
                                         CASs=['7732-18-5', '74-82-8', '111-65-9'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases)
    eos_kwargs = dict(Tcs=Tcs, Pcs=Pcs, omegas=omegas, kijs=kijs)
    gas = CEOSGas(eos, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(eos, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    fig = flashN.debug_PT(zs=zs, Tmin=175, Tmax=700, Pmin=1e4, Pmax=1e8, pts=35, verbose=False, show=False, values=False)
    write_PT_plot(fig, eos, IDs, zs, flashN)


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [SRKMIX]) # eos_mix_list
def test_PT_plot_ethylene_ethanol_nitrogen(eos):
    IDs = ['ethylene', 'ethanol', 'nitrogen']
    zs = [.8, 0.19, .01]
    T = 283.65
    P = 4690033.135557525
    constants = ChemicalConstantsPackage(Tcs=[282.34, 514.0, 126.2], Pcs=[5041000.0, 6137000.0, 3394387.5], omegas=[0.085, 0.635, 0.04], MWs=[28.05316, 46.06844, 28.0134], CASs=['74-85-1', '64-17-5', '7727-37-9'])
    properties =PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.2701693466919565e-21, 1.660757962278189e-17, -3.525777713754962e-14, 4.01892664375958e-11, -2.608749347072186e-08, 9.23682495982131e-06, -0.0014524032651835623, 0.09701764355901257, 31.034399100170667])),
                                                                                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),
                                                                                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas, kijs=[[0.0, -.0057, 0.0], [-.0057, 0.0, 0.0], [0.0, 0.0, 0.0]])
    gas = CEOSGas(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    fig = flashN.debug_PT(zs=zs, Tmin=175, Tmax=500, Pmin=1e5, Pmax=1e8, pts=25, verbose=False, show=False, values=False)
    write_PT_plot(fig, eos, IDs, zs, flashN)



@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [PRMIX]) # eos_mix_list
def test_PT_plot_LLL_plot_random1(eos):
    IDs = ['1,3-butanediol', 'hydrazine', 'silane', 'butane, 1-(methylthio)-', 'butyl acetate']
    constants = ChemicalConstantsPackage(Tcs=[676.0, 653.0, 269.0, 591.3, 575.6], Pcs=[4000000.0, 14692125.0, 4855494.0, 3880747.0, 3140000.0], omegas=[0.7206, 0.32799999999999996, 0.1599, 0.3691, 0.4345], MWs=[90.121, 32.04516, 32.11726, 104.21378, 116.15828], CASs=['107-88-0', '302-01-2', '7803-62-5', '628-29-5', '123-86-4'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(196.15, 676.0, [-1.6703777121140357e-19, 6.336563235116343e-16, -1.0268904098033873e-12, 9.209775070752545e-10, -4.925751560244259e-07, 0.00015625155455794686, -0.027101271359103958, 2.363660193690524, 5.925941687513358])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.623312561584884e-21, 1.6721115187937393e-17, -3.182771170056967e-14, 3.194692211808055e-11, -1.774280702831314e-08, 5.047063059999335e-06, -0.0005017671060833166, 0.03937920538251642, 32.21829961922973])),
                                                                                     HeatCapacityGas(poly_fit=(88.15, 269.0, [-7.353710707470991e-18, 9.770203085327697e-15, -5.192485575504864e-12, 1.3663121283879677e-09, -1.7426822740884812e-07, 7.875818199129791e-06, 0.00041232162952700355, -0.054997039960075225, 24.92658630549439])),
                                                                                     HeatCapacityGas(poly_fit=(273, 1000, [1.321014390602642e-21, -8.34388600397224e-18, 2.3098226226797758e-14, -3.6634048438548e-11, 3.636963615039002e-08, -2.295930412929758e-05, 0.008674670844432155, -1.4111475354201926, 182.2898224404806])),
                                                                                     HeatCapacityGas(poly_fit=(195.15, 575.6, [-4.4626862569926e-19, 1.5426193954174004e-15, -2.2990858376078466e-12, 1.9176735381876706e-09, -9.677520492563025e-07, 0.0002957533403178217, -0.051211100401858944, 4.615337526886703, -64.55339891849962]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    zs = [.2, .2, .2, .2, .2]
    T = 150.0
    P = 381214.0
    gas = CEOSGas(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    fig = flashN.debug_PT(zs=zs, Tmin=120, Tmax=700, Pmin=1e4, Pmax=1e8, pts=25, verbose=False, show=False, values=False)
    write_PT_plot(fig, eos, IDs, zs, flashN)


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [SRKMIX])
def test_PT_plot_H2S_CH4(eos):
    T = 167.54
    P = 26560
    IDs = ['methane', 'hydrogen sulfide']
    zs = [0.93, 0.07]
    kijs = [[0,.08],[0.08,0]]
    Tcs=[190.6, 373.2]
    Pcs=[46e5, 89.4e5]
    omegas=[0.008, .1]

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[16.04246, 34.08088], CASs=['74-82-8', '7783-06-4'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534]))], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq, liq], gas=gas)

    zs_list = ([.9885, .0115], [.9813, .0187], [0.93, 0.07],
               [.5, .5], [.112, .888], [.11, .89])
    for zs in zs_list:
        fig = flashN.debug_PT(zs=zs, Tmin=100, Tmax=300, Pmin=1e4, Pmax=1e8, pts=25,
                              verbose=False, show=False, values=False)
        write_PT_plot(fig, eos, IDs, zs, flashN)


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [SRKMIX])
def test_PT_plot_CH4_propane(eos):
    T = 277.6
    P = 100e5
    IDs = ['methane', 'propane']
    zs = [.4, .6]
    Tcs = [190.6, 369.8]
    Pcs = [46e5, 42.5e5]
    omegas = [.008, .152]
    kijs = [[0.0, 0.029],  [0.029, 0.0]]
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                         MWs=[16.04246, 44.09562], CASs=['74-82-8', '74-98-6'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])), ], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq, liq], gas=gas)
    zs_list = ([.4, .6], [.68, .32], [.73, .27], [.9, .1])
    for zs in zs_list:
        fig = flashN.debug_PT(zs=zs, Tmin=200, Tmax=350, Pmin=1e4, Pmax=1e8,
#                              Tmin=200, Tmax=450, Pmin=4e5, Pmax=3e6,
                              pts=33,
                              verbose=False, show=False, values=False)
        write_PT_plot(fig, eos, IDs, zs, flashN)



@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [PRMIX])
def test_PT_plot_N2_ethane(eos):
    T = 270.0
    P = 76e5
    zs = [.1, .9]
    kijs = [[0.0,0.08],[0.08,0.0]]
    Tcs = [126.2, 305.4]
    Pcs = [33.9E5, 48.8E5]
    omegas = [0.04, 0.098]
    IDs = ['nitrogen', 'ethane']
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[28.0134, 30.06904], CASs=['7727-37-9', '74-84-0'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228]))], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq], gas=gas)
    zs_list = ([.1, .9], [.18, .82], [.3, .7], [.44, .56], [.6, .4])
    for zs in zs_list:
        fig = flashN.debug_PT(zs=zs, Tmin=120, Tmax=500, Pmin=1e4, Pmax=1e8,
                              pts=33,
                              verbose=False, show=False, values=False)
        write_PT_plot(fig, eos, IDs, zs, flashN)


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [PRMIX])
def test_PT_plot_CO2_CH4(eos):
    T = 220.0
    P = 60.8e5
    zs = [.1, .9]
    zs_list = [zs, [.2, .8], [.3, .7], [.43, .57], [.6, .4] ]
    kijs = [[0.0, 0.095], [0.095, 0.0]]
    Tcs = [304.2, 190.6]
    Pcs = [73.8E5, 46.0E5]
    omegas = [0.225, 0.008]
    IDs = ['CO2', 'methane']

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[44.0095, 16.04246],
                                         CASs=['124-38-9', '74-82-8'])

    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228]))], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)

    gas = CEOSGas(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq], gas=gas)
    for zs in zs_list:
        fig = flashN.debug_PT(zs=zs, Tmin=100, Tmax=300, Pmin=1e4, Pmax=1e8,
                              pts=33,
                              verbose=False, show=False, values=False)
        write_PT_plot(fig, eos, IDs, zs, flashN)



@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("eos", [PRMIX])
def test_PT_plot_N2_CH4_Ethane(eos):
    IDs = ['nitrogen', 'methane', 'ethane']
    T = 270.0
    P = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    all_zs = [[0.3, 0.1, 0.6],
            [.15, .3, .55],
            [.08, .38, .54],
            [.05, .05, .9]]
    zs = all_zs[0]

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                         MWs=[28.0134, 16.04246, 30.06904],
                                         CASs=['7727-37-9', '74-82-8', '74-84-0'])

    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)

    gas = CEOSGas(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(eos, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    for zs in all_zs:
        fig = flashN.debug_PT(zs=zs, Tmin=120, Tmax=400, Pmin=1e4, Pmax=1e8,
                              pts=33,
                              verbose=False, show=False, values=False)
        write_PT_plot(fig, eos, IDs, zs, flashN)


def test_PH_VL_not_identifying_phases():
    constants = ChemicalConstantsPackage(atomss=[{'N': 2}, {'C': 2, 'H': 6}], CASs=['7727-37-9', '74-84-0'], MWs=[28.0134, 30.06904], names=['nitrogen', 'ethane'], omegas=[0.04, 0.098], Pcs=[3394387.5, 4872000.0], Tcs=[126.2, 305.32])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
                                               VaporPressures=[VaporPressure(exp_poly_fit=(63.2, 126.18199999999999, [5.490876411024536e-15, -3.709517805130509e-12, 1.0593254238679989e-09, -1.6344291780087318e-07, 1.4129990091975526e-05, -0.0005776268289835264, -0.004489180523814208, 1.511854256824242, -36.95425216567675])),
        VaporPressure(exp_poly_fit=(90.4, 305.312, [-1.1908381885079786e-17, 2.1355746620587145e-14, -1.66363909858873e-11, 7.380706042464946e-09, -2.052789573477409e-06, 0.00037073086909253047, -0.04336716238170919, 3.1418840094903784, -102.75040650505277])),],
                                               HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
        HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228]))])

    eos_kwargs = {'Pcs':constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(eos_class=PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases,
                  T=298.15, P=101325.0, zs=[0.5, 0.5])
    liquid = CEOSLiquid(eos_class=PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases,
                        T=298.15, P=101325.0, zs=[0.5, 0.5])

    # Case where PH flashes were not having their IDs, ID'd
    flasher = FlashVLN(constants, correlations, [liquid], gas)
    res = flasher.flash(P=.99e6, H=-3805.219843192915, zs=[.5, .5])
    assert res.gas is not None
    assert len(res.liquids) == 1


def test_TVF_PVF_reversed_mole_balance():
    '''This is a test of TVF and PVF consistency. There is definitely error.
    Fixing the error seems to require a tighter PT SS tolerance.
    For now, run with lower tplerance.
    '''
    constants = ChemicalConstantsPackage(atomss=[{'N': 2}, {'C': 2, 'H': 6}], CASs=['7727-37-9', '74-84-0'], MWs=[28.0134, 30.06904], names=['nitrogen', 'ethane'], omegas=[0.04, 0.098], Pcs=[3394387.5, 4872000.0], Tcs=[126.2, 305.32])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
                                               VaporPressures=[VaporPressure(exp_poly_fit=(63.2, 126.18199999999999, [5.490876411024536e-15, -3.709517805130509e-12, 1.0593254238679989e-09, -1.6344291780087318e-07, 1.4129990091975526e-05, -0.0005776268289835264, -0.004489180523814208, 1.511854256824242, -36.95425216567675])),
        VaporPressure(exp_poly_fit=(90.4, 305.312, [-1.1908381885079786e-17, 2.1355746620587145e-14, -1.66363909858873e-11, 7.380706042464946e-09, -2.052789573477409e-06, 0.00037073086909253047, -0.04336716238170919, 3.1418840094903784, -102.75040650505277])),],
                                               HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
        HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228]))])

    eos_kwargs = {'Pcs':constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(eos_class=PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases,
                  T=298.15, P=101325.0, zs=[0.5, 0.5])
    liquid = CEOSLiquid(eos_class=PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases,
                        T=298.15, P=101325.0, zs=[0.5, 0.5])
    flasher = FlashVLN(constants, correlations, [liquid], gas)
    base = flasher.flash(T=200.0, P=1e6, zs=[.5, .5])

    liquid = flasher.flash(zs=base.liquid0.zs, T=base.T, VF=0)

    # The issues comes when using the gas mole fractions
    zs_balanced = [liquid.gas.zs[i]*base.gas_beta + liquid.liquid0.zs[i]*base.betas[1] for i in range(flasher.N)]
    err = sum(abs(i-j) for i, j in zip(base.zs, zs_balanced))
    assert err < 1e-6

    liquid = flasher.flash(zs=base.liquid0.zs, P=base.P, VF=0)

    zs_balanced = [liquid.gas.zs[i]*base.gas_beta + liquid.liquid0.zs[i]*base.betas[1] for i in range(flasher.N)]
    err = sum(abs(i-j) for i, j in zip(base.zs, zs_balanced))
    assert err < 1e-6


def test_methane_water_decane_hot_start_issue():
    constants = ChemicalConstantsPackage(atomss=[{'C': 1, 'H': 4}, {'H': 2, 'O': 1}, {'C': 10, 'H': 22}], MWs=[16.04246, 18.01528, 142.28168], omegas=[0.008, 0.344, 0.49], Pcs=[4599000.0, 22048320.0, 2110000.0], Tcs=[190.564, 647.14, 611.7], Vcs=[9.86e-05, 5.6e-05, 0.000624])
    HeatCapacityGases = [HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="7732-18-5", MW=18.01528, similarity_variable=0.16652530518537598, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
     HeatCapacityGas(CASRN="124-18-5", MW=142.28168, similarity_variable=0.22490597524572384, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735]))]

    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)

    eos_kwargs = {'Pcs': [4599000.0, 22048320.0, 2110000.0],
     'Tcs': [190.564, 647.14, 611.7],
     'omegas': [0.008, 0.344, 0.49],
     'kijs': [[0.0, 0, 0.0411], [0, 0.0, 0], [0.0411, 0, 0.0]]}

    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid1 = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid2 = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    flasher = FlashVLN(constants, properties, liquids=[liquid1, liquid2], gas=gas)

    zs = [1/3, 1/3, 1/3]


    state = flasher.flash(T=400, P=190000, zs=zs)
    reflash = flasher.flash(T=600, P=189900, zs=zs, hot_start = state)
    assert state.phase_count == 2
    assert reflash.phase_count == 1
    assert_close(reflash.V(), 0.026070314738216745, rtol=1e-7)

    # Three phase hot start
    state = flasher.flash(T=300, P=190000, zs=zs)

    # Test convergence to 1 phase from 3
    reflash = flasher.flash(T=600, P=189900, zs=zs, hot_start = state)
    assert reflash.phase_count == 1
    assert_close(reflash.V(), 0.026070314738216745, rtol=1e-7)
    # Test convergence to 3 phase from 3
    reflash = flasher.flash(T=301, P=189900, zs=zs, hot_start = state)
    assert_close1d(reflash.gas.zs, [0.98115515980832, 0.017172713474410774, 0.0016721267172691817], rtol=1e-5)

def test_methane_water_decane_mole_mass_flows():
    constants = ChemicalConstantsPackage(atomss=[{'C': 1, 'H': 4}, {'H': 2, 'O': 1}, {'C': 10, 'H': 22}], MWs=[16.04246, 18.01528, 142.28168], omegas=[0.008, 0.344, 0.49], Pcs=[4599000.0, 22048320.0, 2110000.0], Tcs=[190.564, 647.14, 611.7], Vcs=[9.86e-05, 5.6e-05, 0.000624],
                                         CASs=['74-82-8', '7732-18-5', '124-18-5'],
                                         Vml_STPs=[5.858784737746477e-05, 1.808720510576827e-05, 0.00019580845677729525])
    HeatCapacityGases = [HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="7732-18-5", MW=18.01528, similarity_variable=0.16652530518537598, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
     HeatCapacityGas(CASRN="124-18-5", MW=142.28168, similarity_variable=0.22490597524572384, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735]))]

    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)

    eos_kwargs = {'Pcs': [4599000.0, 22048320.0, 2110000.0],
     'Tcs': [190.564, 647.14, 611.7],
     'omegas': [0.008, 0.344, 0.49],
     'kijs': [[0.0, 0, 0.0411], [0, 0.0, 0], [0.0411, 0, 0.0]]}

    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid1 = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid2 = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    flasher = FlashVLN(constants, properties, liquids=[liquid1, liquid2], gas=gas)

    zs = [1/3, 1/3, 1/3]

    state = EquilibriumStream(T=300, P=190000, zs=zs, flasher=flasher, n=100)

    # Test a few unrelated properties related to mass flow - maybe make this its own test
    assert_close(state.gas.n, 33.60882051261701, rtol=1e-4)
    assert_close(state.liquid0.n, 32.21034435334316, rtol=1e-4)
    assert_close(state.liquid1.n, 34.18083513403984, rtol=1e-4)
    assert_close(state.bulk.n, 100, atol=0, rtol=0)

    assert_close(state.gas.m, 0.5468897314984643, rtol=1e-4)
    assert_close(state.liquid0.m, 0.5802782550073167, rtol=1e-4)
    assert_close(state.liquid1.m, 4.750812680160885, rtol=1e-4)
    assert_close(state.bulk.m, 5.877980666666666, rtol=1e-12)

    assert_close(state.gas.Q, 0.43924036014878864, rtol=1e-4)
    assert_close(state.liquid0.Q, 0.0006851416847518282, rtol=1e-4)
    assert_close(state.liquid1.Q, 0.00702205223320992, rtol=1e-4)
    assert_close(state.bulk.Q, 0.44694755406675035, rtol=1e-4)

    assert_close(state.bulk.beta, 1, rtol=1e-12)
    assert_close(state.bulk.beta_mass, 1, rtol=1e-12)
    assert_close(state.bulk.beta_volume, 1, rtol=1e-12)

    # mole flow rate
    assert_close1d(state.gas.ns, [33.013193071412594, 0.5429462094489919, 0.05268123175541058], rtol=1e-4)
    assert_close1d(state.liquid0.ns, [5.9532925972237186e-05, 32.210284820417186, 7.755417241801349e-20], rtol=1e-4)
    assert_close1d(state.liquid1.ns, [0.32008072801310733, 0.580102307168074, 33.28065209885866], rtol=1e-4)
    assert_close1d(state.bulk.ns, [33.33333333333333, 33.33333333333333, 33.33333333333333], rtol=1e-12)
    assert_close(sum(state.gas.ns), state.gas.n, rtol=1e-13)
    assert_close(sum(state.liquid0.ns), state.liquid0.n, rtol=1e-13)
    assert_close(sum(state.liquid1.ns), state.liquid1.n, rtol=1e-13)
    assert_close(sum(state.bulk.ns), state.bulk.n, rtol=1e-13)

    # mass flow rate
    assert_close1d(state.gas.ms, [0.5296128293506854, 0.009781327988721316, 0.0074955741590576], rtol=1e-4)
    assert_close1d(state.liquid0.ms, [9.550545836471654e-07, 0.580277299952733, 1.1034537943275334e-20], rtol=1e-4)
    assert_close1d(state.liquid1.ms, [0.005134882276214654, 0.010450705492876204, 4.735227092391794], rtol=1e-4)
    assert_close1d(state.bulk.ms, [0.5347486666666665, 0.6005093333333332, 4.742722666666666], rtol=1e-12)
    assert_close(sum(state.gas.ms), state.gas.m, rtol=1e-13)
    assert_close(sum(state.liquid0.ms), state.liquid0.m, rtol=1e-13)
    assert_close(sum(state.liquid1.ms), state.liquid1.m, rtol=1e-13)
    assert_close(sum(state.bulk.ms), state.bulk.m, rtol=1e-13)

    # volume flow rate
    assert_close2d([state.gas.Qgs, state.liquid0.Qgs, state.liquid1.Qgs, state.bulk.Qgs],
                   ([0.7805913391267482, 0.012837870841235948, 0.0012456387709558483],
                     [1.4076459161729235e-06, 0.7616067099972762, 1.8337552254242914e-21],
                     [0.007568254411742214, 0.013716420456613143, 0.7869153623723653],
                     [0.7881610012076177, 0.7881610012076177, 0.7881610012076177]),
                   rtol=1e-9)

    for i in range(3):
        assert_close(state.gas.Qgs[i]+ state.liquid0.Qgs[i]+ state.liquid1.Qgs[i], state.bulk.Qgs[i], rtol=1e-9)

    assert_close(sum(state.gas.Qgs), state.gas.Qg, rtol=1e-14)
    assert_close(sum(state.liquid0.Qgs), state.liquid0.Qg, rtol=1e-14)
    assert_close(sum(state.liquid1.Qgs), state.liquid1.Qg, rtol=1e-14)
    assert_close(sum(state.bulk.Qgs), state.bulk.Qg, rtol=1e-14)

    assert_close2d([state.gas.Qls, state.liquid0.Qls, state.liquid1.Qls, state.bulk.Qls],
                   ([0.0019341719171106985, 9.820379451703334e-06, 1.0315430691153987e-05],
                     [3.487905980795341e-09, 0.0005825940280620999, 1.5185762817811496e-23],
                     [1.8752840841299744e-05, 1.0492429412078341e-05, 0.006516633128019566],
                     [0.0019529282459154922, 0.0006029068368589423, 0.006526948559243174]), rtol=1e-9)

    for i in range(3):
        assert_close(state.gas.Qls[i]+ state.liquid0.Qls[i]+ state.liquid1.Qls[i], state.bulk.Qls[i], rtol=1e-9)

    assert_close(sum(state.gas.Qls), state.gas.Ql, rtol=1e-14)
    assert_close(sum(state.liquid0.Qls), state.liquid0.Ql, rtol=1e-14)
    assert_close(sum(state.liquid1.Qls), state.liquid1.Ql, rtol=1e-14)
    assert_close(sum(state.bulk.Qls), state.bulk.Ql, rtol=1e-14)

    # atom fractions
    for p in (state, state.gas, state.liquid0, state.liquid1, state.bulk):
        assert_close(p.atom_fractions()['C'], p.Carbon_atom_fraction(), rtol=1e-12)
        assert_close(p.atom_fractions()['H'], p.Hydrogen_atom_fraction(), rtol=1e-12)
        assert_close(p.atom_fractions()['O'], p.Oxygen_atom_fraction(), rtol=1e-12)
        assert_close(0, p.Chlorine_atom_fraction(), rtol=1e-12)

        assert_close(p.atom_mass_fractions()['C'], p.Carbon_atom_mass_fraction(), rtol=1e-12)
        assert_close(p.atom_mass_fractions()['H'], p.Hydrogen_atom_mass_fraction(), rtol=1e-12)
        assert_close(p.atom_mass_fractions()['O'], p.Oxygen_atom_mass_fraction(), rtol=1e-12)
        assert_close(0, p.Chlorine_atom_mass_fraction(), rtol=1e-12)
    # partial pressure component specific
    assert_close(state.water_partial_pressure(), 63333.33333333333, rtol=1e-4)
    assert_close(state.gas.water_partial_pressure(), 3069.425770433731, rtol=1e-4)
    assert_close(state.liquid0.water_partial_pressure(), 189999.64883157378, rtol=1e-4)

    assert_close2d([state.gas.concentrations(), state.liquid0.concentrations(), state.liquid1.concentrations(), state.bulk.concentrations()
                    , state.concentrations()],
                   [[75.15974410964803, 1.2361027326019718, 0.11993713814815492],
                    [0.08689140844466525, 47012.589566907445, 1.1319435693962356e-16],
                    [45.582219753268916, 82.61150556877838, 4739.448097731596],
                    [74.57996588198152, 74.57996588198152, 74.57996588198152],
                    [74.57996588198152, 74.57996588198152, 74.57996588198152]],
                   rtol=1e-9)

    assert_close(sum(state.concentrations()), state.rho())

    assert_close2d([state.gas.concentrations_mass(), state.liquid0.concentrations_mass(), state.liquid1.concentrations_mass(), state.bulk.concentrations_mass()
                    , state.concentrations_mass()],
                   [[1.2057471884892643, 0.02226873683658965, 0.01706485751011157],
                   [0.0013939519443172048, 846.9449645729165, 1.61054832718893e-17],
                   [0.7312509371030262, 1.4882694040431017, 674.3366376180555],
                   [1.196446119463053, 1.343578967754344, 10.611362840031012],
                   [1.196446119463053, 1.343578967754344, 10.611362840031012]],
                   rtol=1e-9)

    assert_close(sum(state.concentrations_mass()), state.rho_mass())


    assert_close2d([state.gas.partial_pressures(), state.liquid0.partial_pressures(),
                    state.liquid1.partial_pressures(), state.bulk.partial_pressures()
                    , state.partial_pressures()],
                   [[186632.75258986393, 3069.425770433731, 297.8216397023035],
                   [0.35116842622488303, 189999.64883157378, 4.574708235894151e-16],
                   [1779.223301127769, 3224.597583110814, 184996.17911576145],
                   [63333.33333333333, 63333.33333333333, 63333.33333333333],
                   [63333.33333333333, 63333.33333333333, 63333.33333333333]],
                   rtol=1e-6)

    assert_close(sum(state.partial_pressures()), state.P)
    assert_close(sum(state.bulk.partial_pressures()), state.P)
    assert_close(sum(state.gas.partial_pressures()), state.P)
    assert_close(sum(state.liquid0.partial_pressures()), state.P)
    assert_close(sum(state.liquid1.partial_pressures()), state.P)
    assert_close(sum(state.liquid_bulk.partial_pressures()), state.P)



    atom_content = state.atom_content()
    assert_close(atom_content['C'], (1+10)/3, rtol=1e-13)
    assert_close(atom_content['H'], (4+2+22)/3, rtol=1e-13)
    assert_close(atom_content['O'], (1)/3, rtol=1e-13)

    atom_content = state.gas.atom_content()
    assert_close(atom_content['C'], 0.9979524683520365, rtol=1e-6)
    assert_close(atom_content['H'], 3.995904936704073, rtol=1e-6)
    assert_close(atom_content['O'], 0.016154872475967006, rtol=1e-6)

    atom_content = state.liquid_bulk.atom_content()
    assert_close(atom_content['C'], 5.017634327656933, rtol=1e-6)
    assert_close(atom_content['H'], 12.035268655313866, rtol=1e-6)
    assert_close(atom_content['O'], 0.49389674021105096, rtol=1e-6)

    assert_close(state.Oxygen_atom_mass_flow(), 0.5333133333333332)
    assert_close(state.gas.Oxygen_atom_mass_flow(), 0.008686813583954723)

    assert_close(state.Oxygen_atom_flow(), 100/3, rtol=1e-13)
    assert_close(state.Carbon_atom_flow(), 366.66666666666663, rtol=1e-13)
    assert_close(state.gas.Oxygen_atom_flow(), 0.5429462094489919, rtol=1e-6)


    atom_flows = state.atom_flows()
    assert_close(atom_flows['C'], 100*(1+10)/3, rtol=1e-13)
    assert_close(atom_flows['H'], 100*(4+2+22)/3, rtol=1e-13)
    assert_close(atom_flows['O'], 100*(1)/3, rtol=1e-13)

    assert_close(state.gas.atom_flows()['O'], state.gas.Oxygen_atom_flow(), rtol=1e-13)
    assert_close(state.gas.atom_flows()['H'], state.gas.Hydrogen_atom_flow(), rtol=1e-13)
    assert_close(state.gas.atom_flows()['C'], state.gas.Carbon_atom_flow(), rtol=1e-13)
    assert 0 ==  state.gas.Iron_atom_flow()

    atom_mass_flows = state.atom_mass_flows()
    assert_close(atom_mass_flows['C'], 4.403923333333333, rtol=1e-13)
    assert_close(atom_mass_flows['H'], 0.940744, rtol=1e-13)
    assert_close(atom_mass_flows['O'], 0.5333133333333332, rtol=1e-13)
    assert 0 ==  state.gas.Iron_atom_mass_flow()

    atom_mass_flows = state.gas.atom_mass_flows()
    assert_close(atom_mass_flows['C'], 0.4028389427482879, rtol=1e-6)
    assert_close(atom_mass_flows['H'], 0.13536397516622173, rtol=1e-6)
    assert_close(atom_mass_flows['O'], 0.008686813583954723, rtol=1e-6)

    assert_close(state.Oxygen_atom_count_flow(), 2.007380253333333e+25)
    assert_close(state.Carbon_atom_count_flow(), 2.2081182786666662e+26)
    assert_close(state.gas.Oxygen_atom_count_flow(), 3.269698498410271e+23, rtol=1e-5)
    assert 0 == state.Iron_atom_count_flow()
    assert 0 == state.gas.Iron_atom_count_flow()


    assert_close(state.gas.atom_count_flows()['O'], state.gas.Oxygen_atom_count_flow(), rtol=1e-13)
    assert_close(state.gas.atom_count_flows()['H'], state.gas.Hydrogen_atom_count_flow(), rtol=1e-13)
    assert_close(state.gas.atom_count_flows()['C'], state.gas.Carbon_atom_count_flow(), rtol=1e-13)

    assert_close(state.atom_count_flows()['O'], state.Oxygen_atom_count_flow(), rtol=1e-13)
    assert_close(state.atom_count_flows()['H'], state.Hydrogen_atom_count_flow(), rtol=1e-13)
    assert_close(state.atom_count_flows()['C'], state.Carbon_atom_count_flow(), rtol=1e-13)


    # calc once, also get the cached versions
    for i in range(2):

        calc = [state.gas.H_flow(), state.liquid0.H_flow(), state.liquid1.H_flow(), state.liquid_bulk.H_flow(), state.bulk.H_flow(), state.H_flow()]
        expect = [1025.0344902476986, -1467448.4719594691, -1627495.395822481, -2054769.7382928678, -3093918.8332917024, -3093918.8332917024]
        assert_close1d(calc, expect, rtol=1e-6)

        calc = [state.gas.S_flow(), state.liquid0.S_flow(), state.liquid1.S_flow(), state.liquid_bulk.S_flow(), state.bulk.S_flow(), state.S_flow()]
        expect = [-144.62478667806292, -3949.6037477860937, -3778.6417926825343, -5130.873367998199, -7872.870327146691, -7872.870327146691]
        assert_close1d(calc, expect, rtol=1e-6)

        calc = [state.gas.G_flow(), state.liquid0.G_flow(), state.liquid1.G_flow(), state.liquid_bulk.G_flow(), state.bulk.G_flow(), state.G_flow()]
        expect = [44412.47049366657, -282567.34762364114, -493902.8580177207, -515507.72789340816, -732057.7351476949, -732057.7351476949]
        assert_close1d(calc, expect, rtol=1e-6)

        calc = [state.gas.U_flow(), state.liquid0.U_flow(), state.liquid1.U_flow(), state.liquid_bulk.U_flow(), state.bulk.U_flow(), state.U_flow()]
        expect = [-82430.63393802213, -1467578.648879572, -1628829.5857467907, -2055741.9487128956, -3178838.868564385, -3178838.868564385]
        assert_close1d(calc, expect, rtol=1e-6)

        calc = [state.gas.A_flow(), state.liquid0.A_flow(), state.liquid1.A_flow(), state.liquid_bulk.A_flow(), state.bulk.A_flow(), state.A_flow()]
        expect = [-39043.19793460326, -282697.524543744, -495237.0479420305, -516479.93831343605, -816977.7704203774, -816977.7704203774]
        assert_close1d(calc, expect, rtol=1e-6)


        calc = [state.gas.H_dep_flow(), state.liquid0.H_dep_flow(), state.liquid1.H_dep_flow(), state.liquid_bulk.H_dep_flow(), state.bulk.H_dep_flow(), state.H_dep_flow()]
        expect = [-1221.6033836873603, -1469448.8244505627, -1641935.3674129113, -2071210.0623743918, -3112605.795248172, -3112605.795248172]
        assert_close1d(calc, expect, rtol=1e-5)

        calc = [state.gas.S_dep_flow(), state.liquid0.S_dep_flow(), state.liquid1.S_dep_flow(), state.liquid_bulk.S_dep_flow(), state.bulk.S_dep_flow(), state.S_dep_flow()]
        expect = [-2.8167554595340234, -3787.928480891734, -3687.7291384401638, -4963.1772679128635, -7478.474374791432, -7478.474374791432]
        assert_close1d(calc, expect, rtol=1e-5)

        calc = [state.gas.G_dep_flow(), state.liquid0.G_dep_flow(), state.liquid1.G_dep_flow(), state.liquid_bulk.G_dep_flow(), state.bulk.G_dep_flow(), state.G_dep_flow()]
        expect = [-376.5767458271534, -333070.28018304234, -535616.6258808621, -582256.8820005328, -869063.4828107425, -869063.4828107425]
        assert_close1d(calc, expect, rtol=1e-5)

        calc = [state.gas.U_dep_flow(), state.liquid0.U_dep_flow(), state.liquid1.U_dep_flow(), state.liquid_bulk.U_dep_flow(), state.bulk.U_dep_flow(), state.U_dep_flow()]
        expect = [-845.4872742444248, -1389235.490157551, -1558010.974543451, -1906580.178787535, -2948091.9519762574, -2948091.9519762574]
        assert_close1d(calc, expect, rtol=1e-5)

        calc = [state.gas.A_dep_flow(), state.liquid0.A_dep_flow(), state.liquid1.A_dep_flow(), state.liquid_bulk.A_dep_flow(), state.bulk.A_dep_flow(), state.A_dep_flow()]
        expect = [-0.4606363842178823, -252856.94589003065, -451692.233011402, -417626.9984136762, -704549.6395388279, -704549.6395388279]
        assert_close1d(calc, expect, rtol=1e-5)

def test_ethanol_CO2_water_decane_finding_wrong_two_phase_solution():
    kijs = [[0.0, 0, 0, 0], [0, 0.0, 0.0952, 0.1141], [0, 0.0952, 0.0, 0], [0, 0.1141, 0, 0.0]]
    constants = ChemicalConstantsPackage(MWs=[46.06844, 44.0095, 18.01528, 142.28168], Tbs=[351.39, 194.67, 373.124, 447.25],
                                        Tms=[159.05, 216.65, 273.15, 243.225], Tcs=[514.0, 304.2, 647.14, 611.7],
                                        Pcs=[6137000.0, 7376460.0, 22048320.0, 2110000.0], Vcs=[0.000168, 9.4e-05, 5.6e-05, 0.000624],
                                        Zcs=[0.24125043269792065, 0.2741463389591581, 0.2294727397218464, 0.2588775438016263],
                                        omegas=[0.635, 0.2252, 0.344, 0.49],
                                        atomss=[{'C': 2, 'H': 6, 'O': 1}, {'C': 1, 'O': 2}, {'H': 2, 'O': 1}, {'C': 10, 'H': 22}],
                                        names=['ethanol', 'carbon dioxide', 'water', 'decane'],
                                        CASs=['64-17-5', '124-38-9', '7732-18-5', '124-18-5'])
    HeatCapacityGases = [HeatCapacityGas(CASRN="64-17-5", MW=46.06844, similarity_variable=0.19536150996213458, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),
     HeatCapacityGas(CASRN="124-38-9", MW=44.0095, similarity_variable=0.0681671002851657, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
     HeatCapacityGas(CASRN="7732-18-5", MW=18.01528, similarity_variable=0.16652530518537598, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
     HeatCapacityGas(CASRN="124-18-5", MW=142.28168, similarity_variable=0.22490597524572384, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735]))]


    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    zs = zs = [0.25, 0.25, .25, .25]

    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    flasher = FlashVLN(constants, properties, liquids=[liquid, liquid], gas=gas)


    res = flasher.flash(T=184.84848484848487, P=100000.0, zs=zs)
    assert_close1d(res.betas,  [0.24983289898515929, 0.7501671010148407])
    assert_close(res.G_min_criteria(), -18810.192558625644)


def test_extra_liquid_phase_three_components_no_issue():
    chemicals = ['water', 'methane', 'decane']
    HeatCapacityGases = [HeatCapacityGas(CASRN="7732-18-5", MW=18.01528, similarity_variable=0.16652530518537598, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
     HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="124-18-5", MW=142.28168, similarity_variable=0.22490597524572384, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735]))]

    constants = ChemicalConstantsPackage(atomss=[{'H': 2, 'O': 1}, {'C': 1, 'H': 4}, {'C': 10, 'H': 22}], MWs=[18.01528, 16.04246, 142.28168], omegas=[0.344, 0.008, 0.49], Pcs=[22048320.0, 4599000.0, 2110000.0], Tbs=[373.124, 111.65, 447.25], Tcs=[647.14, 190.564, 611.7], Tms=[273.15, 90.75, 243.225], Vcs=[5.6e-05, 9.86e-05, 0.000624], Zcs=[0.2294727397218464, 0.2861971332411768, 0.2588775438016263])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)

    kijs = [[0.0, 0, 0], [0, 0.0, 0.0411], [0, 0.0411, 0.0]]
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)

    flasher = FlashVLN(constants, properties, liquids=[liquid, liquid, liquid], gas=gas)

    res = flasher.flash(T=298.15, P=101325, zs=[.3, .3, .4])
    assert res.phase_count == 3

def test_RR3_converged_not_returned():
    # constants, properties = ChemicalConstantsPackage.from_IDs(['nitrogen', 'methane', 'ethane', 'propane',
    #                                                            'n-butane', 'n-pentane','isopentane', 'hexane',
    #                                                            'decane', 'hydrogen sulfide', 'CO2', 'water',
    #                                                            'oxygen', 'argon'])

    constants = ChemicalConstantsPackage(atomss=[{'N': 2}, {'C': 1, 'H': 4}, {'C': 2, 'H': 6}, {'C': 3, 'H': 8}, {'C': 4, 'H': 10}, {'C': 5, 'H': 12}, {'C': 5, 'H': 12}, {'C': 6, 'H': 14}, {'C': 10, 'H': 22}, {'H': 2, 'S': 1}, {'C': 1, 'O': 2}, {'H': 2, 'O': 1}, {'O': 2}, {'Ar': 1}], CASs=['7727-37-9', '74-82-8', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '78-78-4', '110-54-3', '124-18-5', '7783-06-4', '124-38-9', '7732-18-5', '7782-44-7', '7440-37-1'], MWs=[28.0134, 16.04246, 30.06904, 44.09562, 58.1222, 72.14878, 72.14878, 86.17536, 142.28168, 34.08088, 44.0095, 18.01528, 31.9988, 39.948], names=['nitrogen', 'methane', 'ethane', 'propane', 'butane', 'pentane', '2-methylbutane', 'hexane', 'decane', 'hydrogen sulfide', 'carbon dioxide', 'water', 'oxygen', 'argon'], omegas=[0.04, 0.008, 0.098, 0.152, 0.193, 0.251, 0.227, 0.2975, 0.49, 0.1, 0.2252, 0.344, 0.021, -0.004], Pcs=[3394387.5, 4599000.0, 4872000.0, 4248000.0, 3796000.0, 3370000.0, 3380000.0, 3025000.0, 2110000.0, 8936865.0, 7376460.0, 22048320.0, 5042945.25, 4873732.5], Tbs=[77.355, 111.65, 184.55, 231.04, 272.65, 309.21, 300.98, 341.87, 447.25, 213.6, 194.67, 373.124, 90.188, 87.302], Tcs=[126.2, 190.564, 305.32, 369.83, 425.12, 469.7, 460.4, 507.6, 611.7, 373.2, 304.2, 647.14, 154.58, 150.8], Tms=[63.15, 90.75, 90.3, 85.5, 135.05, 143.15, 113.15, 178.075, 243.225, 187.65, 216.65, 273.15, 54.36, 83.81], Vcs=[8.95e-05, 9.86e-05, 0.0001455, 0.0002, 0.000255, 0.000311, 0.000306, 0.000368, 0.000624, 9.85e-05, 9.4e-05, 5.6e-05, 7.34e-05, 7.49e-05])

    HeatCapacityGases = [HeatCapacityGas(CASRN="7727-37-9", MW=28.0134, similarity_variable=0.07139440410660612, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
     HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="74-84-0", MW=30.06904, similarity_variable=0.26605438683775734, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
     HeatCapacityGas(CASRN="74-98-6", MW=44.09562, similarity_variable=0.24945788266499033, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
     HeatCapacityGas(CASRN="106-97-8", MW=58.1222, similarity_variable=0.24087181834135665, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
     HeatCapacityGas(CASRN="109-66-0", MW=72.14878, similarity_variable=0.235624219841278, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
     HeatCapacityGas(CASRN="78-78-4", MW=72.14878, similarity_variable=0.235624219841278, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-3.3940692195566165e-21, 1.7702350857101215e-17, -3.908639931077815e-14, 4.7271630395909585e-11, -3.371762950874771e-08, 1.4169051870903116e-05, -0.003447090979499268, 0.7970786116058249, -8.8294795805881])),
     HeatCapacityGas(CASRN="110-54-3", MW=86.17536, similarity_variable=0.2320849022272724, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
     HeatCapacityGas(CASRN="124-18-5", MW=142.28168, similarity_variable=0.22490597524572384, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735])),
     HeatCapacityGas(CASRN="7783-06-4", MW=34.08088, similarity_variable=0.08802589604493781, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534])),
     HeatCapacityGas(CASRN="124-38-9", MW=44.0095, similarity_variable=0.0681671002851657, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
     HeatCapacityGas(CASRN="7732-18-5", MW=18.01528, similarity_variable=0.16652530518537598, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
     HeatCapacityGas(CASRN="7782-44-7", MW=31.9988, similarity_variable=0.06250234383789392, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
     HeatCapacityGas(CASRN="7440-37-1", MW=39.948, similarity_variable=0.025032542304996495, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-1.0939921922581918e-31, 4.144146614006628e-28, -6.289942296644484e-25, 4.873620648503505e-22, -2.0309301195845294e-19, 4.3863747689727484e-17, -4.29308508081826e-15, 20.786156545383236]))]

    VaporPressures = [VaporPressure(CASRN="7727-37-9", Tb=77.355, Tc=126.2, Pc=3394387.5, omega=0.04, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(63.2, 126.18199999999999, [5.490876411024536e-15, -3.709517805130509e-12, 1.0593254238679989e-09, -1.6344291780087318e-07, 1.4129990091975526e-05, -0.0005776268289835264, -0.004489180523814208, 1.511854256824242, -36.95425216567675])),
     VaporPressure(CASRN="74-82-8", Tb=111.65, Tc=190.564, Pc=4599000.0, omega=0.008, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(90.8, 190.554, [1.2367137894255505e-16, -1.1665115522755316e-13, 4.4703690477414014e-11, -8.405199647262538e-09, 5.966277509881474e-07, 5.895879890001534e-05, -0.016577129223752325, 1.502408290283573, -42.86926854012409])),
     VaporPressure(CASRN="74-84-0", Tb=184.55, Tc=305.32, Pc=4872000.0, omega=0.098, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(90.4, 305.312, [-1.1908381885079786e-17, 2.1355746620587145e-14, -1.66363909858873e-11, 7.380706042464946e-09, -2.052789573477409e-06, 0.00037073086909253047, -0.04336716238170919, 3.1418840094903784, -102.75040650505277])),
     VaporPressure(CASRN="74-98-6", Tb=231.04, Tc=369.83, Pc=4248000.0, omega=0.152, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(85.53500000000001, 369.88, [-6.614459112569553e-18, 1.3568029167021588e-14, -1.2023152282336466e-11, 6.026039040950274e-09, -1.877734093773071e-06, 0.00037620249872919755, -0.048277894617307984, 3.790545023359657, -137.90784855852178])),
     VaporPressure(CASRN="106-97-8", Tb=272.65, Tc=425.12, Pc=3796000.0, omega=0.193, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(135, 425.115, [-7.814862594786785e-19, 1.9990087338619667e-15, -2.222658378316964e-12, 1.4088883433786636e-09, -5.606069245923512e-07, 0.00014504634895991808, -0.024330255528558123, 2.5245680747528114, -119.71493146261565])),
     VaporPressure(CASRN="109-66-0", Tb=309.21, Tc=469.7, Pc=3370000.0, omega=0.251, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(144, 469.6900000000001, [-4.615267824440021e-19, 1.2871894720673305e-15, -1.559590276548174e-12, 1.0761461914304972e-09, -4.6539407107989163e-07, 0.0001305817686830925, -0.023686309296601378, 2.64766854437685, -136.66909337025592])),
     VaporPressure(CASRN="78-78-4", Tb=300.98, Tc=460.4, Pc=3380000.0, omega=0.227, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(112.66000000000001, 460.34000000000003, [-1.225276030507864e-18, 3.156674603724423e-15, -3.515687989427183e-12, 2.216101778567771e-09, -8.688771638626119e-07, 0.0002190280669088704, -0.035326967454100376, 3.4738307292778865, -159.25081271644115])),
     VaporPressure(CASRN="110-54-3", Tb=341.87, Tc=507.6, Pc=3025000.0, omega=0.2975, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(177.84, 507.81, [-1.604637831500919e-19, 4.982429526452368e-16, -6.744889606698199e-13, 5.223525465352799e-10, -2.5493665759436215e-07, 8.125588912741348e-05, -0.01686860265374473, 2.174938146922705, -129.28820562410874])),
     VaporPressure(CASRN="124-18-5", Tb=447.25, Tc=611.7, Pc=2110000.0, omega=0.49, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(243.51, 617.69, [-1.9653193622863184e-20, 8.32071200890499e-17, -1.5159284607404818e-13, 1.5658305222329732e-10, -1.0129531274368712e-07, 4.2609908802380584e-05, -0.01163326014833186, 1.962044867057741, -153.15601192906817])),
     VaporPressure(CASRN="7783-06-4", Tb=213.6, Tc=373.2, Pc=8936865.0, omega=0.1, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(187.70999999999998, 373.09000000000003, [1.6044335409399552e-18, -3.2731123710759124e-15, 2.851504830400211e-12, -1.369466760280872e-09, 3.867598663620643e-07, -6.147439138821642e-05, 0.003971709848411946, 0.2532292056665513, -32.18677016824108])),
     VaporPressure(CASRN="124-38-9", Tb=194.67, Tc=304.2, Pc=7376460.0, omega=0.2252, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(216.602, 304.1182, [3.141153183434152e-16, -6.380334249054363e-13, 5.659255411862981e-10, -2.862785039666638e-07, 9.032467874853838e-05, -0.018198707888890574, 2.285855947253462, -163.51141480675344, 5103.172975291208])),
     VaporPressure(CASRN="7732-18-5", Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317])),
     VaporPressure(CASRN="7782-44-7", Tb=90.188, Tc=154.58, Pc=5042945.25, omega=0.021, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(54.370999999999995, 154.57100000000003, [-9.865296960381724e-16, 9.716055729011619e-13, -4.163287834047883e-10, 1.0193358930366495e-07, -1.57202974507404e-05, 0.0015832482627752501, -0.10389607830776562, 4.24779829961549, -74.89465804494587])),
     VaporPressure(CASRN="7440-37-1", Tb=87.302, Tc=150.8, Pc=4873732.5, omega=-0.004, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(83.816, 150.67700000000002, [3.156255133278695e-15, -2.788016448186089e-12, 1.065580375727257e-09, -2.2940542608809444e-07, 3.024735996501385e-05, -0.0024702132398995436, 0.11819673125756014, -2.684020790786307, 20.312746972164785]))]


    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases, VaporPressures=VaporPressures, skip_missing=True)

    kijs = [[0.0, 0.0289, 0.0533, 0.0878, 0.0711, 0.1, 0.0922, 0.1496, 0.1122, 0.1652, -0.0122, 0, -0.0159, -0.0004], [0.0289, 0.0, -0.0059, 0.0119, 0.0185, 0.023, -0.0056, 0.04, 0.0411, 0, 0.0978, 0, 0, 0.0152], [0.0533, -0.0059, 0.0, 0.0011, 0.0089, 0.0078, 0, -0.04, 0.0144, 0.0952, 0.13, 0, 0, 0], [0.0878, 0.0119, 0.0011, 0.0, 0.0033, 0.0267, 0.0111, 0.0007, 0.0, 0.0878, 0.1315, 0, 0, 0], [0.0711, 0.0185, 0.0089, 0.0033, 0.0, 0.0174, 0, -0.0056, 0.0078, 0, 0.1352, 0, 0, 0], [0.1, 0.023, 0.0078, 0.0267, 0.0174, 0.0, 0, 0, 0, 0.063, 0.1252, 0, 0, 0], [0.0922, -0.0056, 0, 0.0111, 0, 0, 0.0, 0, 0, 0, 0.1219, 0, 0, 0], [0.1496, 0.04, -0.04, 0.0007, -0.0056, 0, 0, 0.0, 0, 0, 0.11, 0, 0, 0], [0.1122, 0.0411, 0.0144, 0.0, 0.0078, 0, 0, 0, 0.0, 0.0333, 0.1141, 0, 0, 0], [0.1652, 0, 0.0952, 0.0878, 0, 0.063, 0, 0, 0.0333, 0.0, 0.0967, 0.0394, 0, 0], [-0.0122, 0.0978, 0.13, 0.1315, 0.1352, 0.1252, 0.1219, 0.11, 0.1141, 0.0967, 0.0, 0.0952, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0394, 0.0952, 0.0, 0, 0], [-0.0159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0089], [-0.0004, 0.0152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0089, 0.0]]

    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    flasher = FlashVLN(constants, properties, liquids=[liquid, liquid], gas=gas)

    zs =      [.03, .816, .05, .03,
              .01, .0123, .01204, .00653,
              .0053, .01, .004224, 1e-2,
              .0012, .012306]
    zs = normalize(zs)

    assert 3 == flasher.flash(T=240.0002, P=1.1E5, zs=zs).phase_count



def test_torture_alphabet_compounds():


    constants = ChemicalConstantsPackage(Tcs=[466.0, 584.0, 304.2, 611.7, 514.0, 632.0, 4398.0, 33.2, 819.15, 318.69, 2223.0, 660.0, 9620.0, 2573.0, 1030.0, 469.7, 683.0, 4862.82, 1314.0, 591.75, 13712.6, 11325.0, 14756.0, 289.733, 9381.32, 3170.0],
    names=['acetaldehyde', 'bromine', 'carbon dioxide', 'decane', 'ethanol', 'furfuryl alcohol', 'gold', 'hydrogen', 'iodine', 'sulfur hexafluoride', 'potassium', 'dipentene', 'molybdenum', 'sodium', 'phosphoric acid', 'pentane', 'p-benzoquinone', 'radium', 'sulfur', 'toluene', 'uranium', 'vanadium', 'tungsten', 'xenon', 'yttrium', 'zinc'],
    Pcs=[7500000.0, 10335150.0, 7376460.0, 2110000.0, 6137000.0, 5350000.0, 635440000.0, 1296960.0, 11654000.0, 3759157.5, 16000000.0, 2750000.0, 1034900000.0, 35464000.0, 5070000.0, 3370000.0, 5960000.0, 368430000.0, 11753700.0, 4108000.0, 842160000.0, 1031400000.0, 1149700000.0, 5840373.0, 622200000.0, 290400000.0],
    omegas=[0.303, 0.132, 0.2252, 0.49, 0.635, 0.734, 3.0831153, -0.22, 0.1115, 0.1499, -0.2028, 0.313, 0.924, -0.1055, 0.41583904, 0.251, 0.495, 0.07529596, 0.2463, 0.257, -0.20519912, -0.2408, 0.077, 0.002, 0.019254852, 0.0459],
    MWs=[44.05256, 159.808, 44.0095, 142.28168, 46.06844, 98.09994, 196.96655, 2.01588, 253.80894, 146.0554192, 39.0983, 136.23404, 95.96, 22.98977, 97.99518099999999, 72.14878, 108.09476000000001, 226, 32.065, 92.13842, 238.02891, 50.9415, 183.84, 131.293, 88.90585, 65.38],
    Tbs=[293.95, 331.95, 194.67, 447.25, 351.39, 441.15, 3109.15, 20.271, 457.55, 209.3, 1032.15, 449.15, 4912.15, 1156.09, 680.15, 309.21, 453.15, 2010.15, 717.76, 383.75, 4404.15, 3680.15, 5828.15, 165.051, 3618.15, 1180.15],
    CASs=['75-07-0', '7726-95-6', '124-38-9', '124-18-5', '64-17-5', '98-00-0', '7440-57-5', '1333-74-0', '7553-56-2', '2551-62-4', '7440-09-7', '138-86-3', '7439-98-7', '7440-23-5', '7664-38-2', '109-66-0', '106-51-4', '7440-14-4', '7704-34-9', '108-88-3', '7440-61-1', '7440-62-2', '7440-33-7', '7440-63-3', '7440-65-5', '7440-66-6'],
    atomss=[{'C': 2, 'H': 4, 'O': 1}, {'Br': 2}, {'C': 1, 'O': 2}, {'C': 10, 'H': 22}, {'C': 2, 'H': 6, 'O': 1}, {'C': 5, 'H': 6, 'O': 2}, {'Au': 1}, {'H': 2}, {'I': 2}, {'F': 6, 'S': 1}, {'K': 1}, {'C': 10, 'H': 16}, {'Mo': 1}, {'Na': 1}, {'H': 3, 'O': 4, 'P': 1}, {'C': 5, 'H': 12}, {'C': 6, 'H': 4, 'O': 2}, {'Ra': 1}, {'S': 1}, {'C': 7, 'H': 8}, {'U': 1}, {'V': 1}, {'W': 1}, {'Xe': 1}, {'Y': 1}, {'Zn': 1}],
    Hfgs=[-165370.0, 30880.0, -393474.0, -249500.0, -234570.0, -211800.0, 366100.0, 0.0, 62417.0, -1220500.0, 89000.0, -33460.0, 658100.0, 107500.0, None, -146900.0, -122900.0, 159000.0, 277200.0, 50410.0, 533000.0, 514200.0, 849400.0, 0.0, 421300.0, 130400.0],
    Gfgs=[-132096.46000000002, 3062.6049999999996, -394338.635, 33414.534999999916, -167635.325, -146928.52300000004, 326416.235, 0.0, 19304.510000000002, -1116445.65, 60496.86, None, 612393.605, 76969.44, None, -8296.02800000002, None, 127545.175, 236741.04499999998, 122449.00299999998, 488396.76, 468463.79, 807241.59, 0.0, 381019.935, 94800.89],
    Sfgs=[-111.59999999999994, 93.30000000000001, 2.9000000000000314, -948.8999999999999, -224.5, -217.57999999999987, 133.10000000000005, 0.0, 144.60000000000002, -349.00000000000034, 95.60000000000001, None, 153.30000000000007, 102.4, None, -464.88, None, 105.5, 135.70000000000007, -241.61999999999998, 149.6, 153.4000000000001, 141.40000000000012, 0.0, 135.10000000000002, 119.40000000000002],)


    HeatCapacityGases = [HeatCapacityGas(CASRN="75-07-0", MW=44.05256, similarity_variable=0.158901094510739, extrapolation="linear", method="POLY_FIT", poly_fit=(50, 3000, [1.5061496798989325e-25, -3.1247355323234672e-21, 2.5745974840505e-17, -1.1109912375015796e-13, 2.7228837941184834e-10, -3.730728683589347e-07, 0.00023313403241737574, 0.03513892663086639, 33.33925611902801])),
     HeatCapacityGas(CASRN="7726-95-6", MW=159.808, similarity_variable=0.012515018021625952, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.3644488754387e-21, -2.4870554247502713e-17, 4.793515617987422e-14, -4.9562629470310385e-11, 2.9517104130542507e-08, -1.0034514137059052e-05, 0.0017605032876189344, -0.10275092472419638, 30.977011585478046])),
     HeatCapacityGas(CASRN="124-38-9", MW=44.0095, similarity_variable=0.0681671002851657, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
     HeatCapacityGas(CASRN="124-18-5", MW=142.28168, similarity_variable=0.22490597524572384, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735])),
     HeatCapacityGas(CASRN="64-17-5", MW=46.06844, similarity_variable=0.19536150996213458, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),
     HeatCapacityGas(CASRN="98-00-0", MW=98.09994, similarity_variable=0.13251791999057289, extrapolation="linear", method="POLY_FIT", poly_fit=(250.35, 632.0, [-9.534610090167143e-20, 3.4583416772306854e-16, -5.304513883184021e-13, 4.410937690059558e-10, -2.0905505018557675e-07, 5.20661895325169e-05, -0.004134468659764938, -0.3746374641720497, 114.90130267531933])),
     HeatCapacityGas(CASRN="7440-57-5", MW=196.96655, similarity_variable=0.005077004191828511, extrapolation="linear", method="POLY_FIT", poly_fit=(1337.33, 4398.0, [-9.989010665334643e-27, 2.574584283782789e-22, -2.8883588320917382e-18, 1.8476353185726658e-14, -7.407542758083047e-11, 1.9222170401464297e-07, -0.00032006284193907843, 0.3218076718474423, 123.10398808088817])),
     HeatCapacityGas(CASRN="1333-74-0", MW=2.01588, similarity_variable=0.9921225469770025, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [1.1878323802695824e-20, -5.701277266842367e-17, 1.1513022068830274e-13, -1.270076105261405e-10, 8.309937583537026e-08, -3.2694889968431594e-05, 0.007443050245274358, -0.8722920255910297, 66.82863369121873])),
     HeatCapacityGas(CASRN="7553-56-2", MW=253.80894, similarity_variable=0.007879943078443178, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-2.270571994733194e-22, 1.3130908394750437e-18, -3.2775544409336123e-15, 4.611181264160479e-12, -4.0011498003667424e-09, 2.1960818114128797e-06, -0.0007479235862121262, 0.1473856034930754, 24.03867001277661])),
     HeatCapacityGas(CASRN="2551-62-4", MW=146.0554192, similarity_variable=0.047927013173092864, extrapolation="linear", method="POLY_FIT", poly_fit=(222.35, 318.69, [-2.3143105796725024e-18, 6.038784487518325e-15, -6.897172679070632e-12, 4.480966926934161e-09, -1.793351117768355e-06, 0.00044404044527702135, -0.06411523166178004, 4.934762756119522, -72.62832720623656])),
     HeatCapacityGas(CASRN="7440-09-7", MW=39.0983, similarity_variable=0.025576559594662682, extrapolation="linear", method="POLY_FIT", poly_fit=(336.65, 2223.0, [7.233020831536545e-25, -6.857615277723212e-21, 2.4121242774972968e-17, -2.990176402339586e-14, -3.7193939868337445e-11, 1.7859889859131094e-07, -0.0002653555384870283, 0.202992613575697, -8.09628531176827])),
     HeatCapacityGas(CASRN="138-86-3", MW=136.23404, similarity_variable=0.19084804355798302, extrapolation="linear", method="JOBACK"),
     HeatCapacityGas(CASRN="7439-98-7", MW=95.96, similarity_variable=0.010421008753647354, extrapolation="linear", method="POLY_FIT", poly_fit=(2895.15, 9620.0, [-3.740105074458955e-30, 2.0895566556282873e-25, -5.071159762572619e-21, 6.99854037396419e-17, -6.030938006820437e-13, 3.3460614689041036e-09, -1.1818894010388684e-05, 0.024900527531995398, 119.13313319840668])),
     HeatCapacityGas(CASRN="7440-23-5", MW=22.98977, similarity_variable=0.04349760784905634, extrapolation="linear", method="POLY_FIT", poly_fit=(370.944, 2573.0, [-1.9906237925345646e-25, 2.889390143556504e-21, -1.828941122187749e-17, 6.635326677748039e-14, -1.5253612637664565e-10, 2.317653572611684e-07, -0.00023466307053145337, 0.15304764217161082, -9.827880238833465])),
     HeatCapacityGas(CASRN="7664-38-2", MW=97.99518099999999, similarity_variable=0.08163666741939077, extrapolation="linear", method="POLY_FIT", poly_fit=(304.325, 1030.0, [4.106112340750141e-21, -2.435478820504475e-17, 6.241961208238214e-14, -9.016878686048454e-11, 8.007711693605859e-08, -4.443876494608706e-05, 0.014664847482950962, -2.2985273962944692, 190.49156752515114])),
     HeatCapacityGas(CASRN="109-66-0", MW=72.14878, similarity_variable=0.235624219841278, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
     HeatCapacityGas(CASRN="106-51-4", MW=108.09476000000001, similarity_variable=0.11101370686238629, extrapolation="linear", method="POLY_FIT", poly_fit=(298, 1000, [-4.824297179959332e-22, 2.220661735671593e-18, -3.822302264749128e-15, 2.4349915430844542e-12, 9.131209445533234e-10, -2.2441146254834528e-06, 0.001052870073839329, 0.12673868225853488, 24.781146031681658])),
     HeatCapacityGas(CASRN="7440-14-4", MW=226, similarity_variable=0.004424778761061947, extrapolation="linear", method="POLY_FIT", poly_fit=(969.15, 4862.82, [-1.6793111563326975e-26, 4.408530316688632e-22, -4.982938633613706e-18, 3.17148366028888e-14, -1.2472937453026698e-10, 3.125503098271017e-07, -0.0004942225956906342, 0.46420288523765396, 110.32452569914281])),
     HeatCapacityGas(CASRN="7704-34-9", MW=32.065, similarity_variable=0.031186652112895685, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [3.750541775695683e-21, -1.7624919055969747e-17, 3.454585288085995e-14, -3.647311688717824e-11, 2.228797537491444e-08, -7.817852665451075e-06, 0.0014323903181117574, -0.10257523372185513, 23.186918526164483])),
     HeatCapacityGas(CASRN="108-88-3", MW=92.13842, similarity_variable=0.16279853724428964, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-9.48396765770823e-21, 4.444060985512694e-17, -8.628480671647472e-14, 8.883982004570444e-11, -5.0893293251198045e-08, 1.4947108372371731e-05, -0.0015271248410402886, 0.19186172941013854, 30.797883940134057])),
     HeatCapacityGas(CASRN="7440-61-1", MW=238.02891, similarity_variable=0.004201170353634775, extrapolation="linear", method="POLY_FIT", poly_fit=(1408.15, 13712.6, [-8.039430716597317e-30, 5.444338785685728e-25, -1.5604253266714132e-20, 2.463835588828813e-16, -2.3369716650863285e-12, 1.3616398938511366e-08, -4.774750339078183e-05, 0.09357862347940651, 256.1993500583917])),
     HeatCapacityGas(CASRN="7440-62-2", MW=50.9415, similarity_variable=0.01963036031526359, extrapolation="linear", method="POLY_FIT", poly_fit=(2183.15, 11325.0, [-2.3810033615401465e-30, 1.4383888678686644e-25, -3.730854967745977e-21, 5.428582595978952e-17, -4.855391966409632e-13, 2.7464523846565512e-09, -9.696010667870341e-06, 0.01998994414147579, 63.41707003510133])),
     HeatCapacityGas(CASRN="7440-33-7", MW=183.84, similarity_variable=0.005439512619669277, extrapolation="linear", method="POLY_FIT", poly_fit=(3687.15, 14756.0, [-1.8768510622348932e-31, 1.5442223896404998e-26, -5.489948558396896e-22, 1.10307331024331e-17, -1.3741544278280436e-13, 1.0931906741410427e-09, -5.485240222678327e-06, 0.01624185131016371, 241.70622203374612])),
     HeatCapacityGas(CASRN="7440-63-3", MW=131.293, similarity_variable=0.007616552291439756, extrapolation="linear", method="POLING_CONST"),
     HeatCapacityGas(CASRN="7440-65-5", MW=88.90585, similarity_variable=0.011247853768902721, extrapolation="linear", method="POLY_FIT", poly_fit=(1795.15, 9381.32, [-2.1233457147566044e-29, 1.0620873065902066e-24, -2.280984229829183e-20, 2.748253380550193e-16, -2.0357010627973534e-12, 9.5392111654939e-09, -2.7915444768403683e-05, 0.04776227250193425, 96.09074400693771])),
     HeatCapacityGas(CASRN="7440-66-6", MW=65.38, similarity_variable=0.015295197308045275, extrapolation="linear", method="POLY_FIT", poly_fit=(692.677, 3170.0, [-1.3572055717920397e-25, 2.3891393389504666e-21, -1.8221671672899744e-17, 7.89162817949639e-14, -2.1363646845282273e-10, 3.7451464115536007e-07, -0.0004241315349190985, 0.29542334821814165, -2.7945293101902706]))]

    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)

    # working with compounds with such high Pcs causess issues sometimes
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    flasher = FlashVLN(constants, properties, liquids=[liquid, liquid], gas=gas)
    res = flasher.flash(T=300, P=1e5, zs=[1/constants.N]*constants.N)
    assert res.phase_count >= 2


def test_four_phase_two_methane_phases_co2_water():

    ns = [400, 10000, 500, 200,
          50, 25, 10, 300,
          200, 1000, 1]
    zs = normalize(ns)
    names = ['water', 'methane', 'ethane', 'propane',
             'isobutane', 'butane', 'isopentane', 'decane',
             'hydrogen sulfide', 'CO2', 'gallium']
    kijs = [[0.0, 0, 0, 0, 0, 0, 0, 0, 0.0394, 0.0952, 0], [0, 0.0, -0.0059, 0.0119, 0.0256, 0.0185, -0.0056, 0.0411, 0, 0.0978, 0], [0, -0.0059, 0.0, 0.0011, -0.0067, 0.0089, 0, 0.0144, 0.0952, 0.13, 0], [0, 0.0119, 0.0011, 0.0, -0.0078, 0.0033, 0.0111, 0.0, 0.0878, 0.1315, 0], [0, 0.0256, -0.0067, -0.0078, 0.0, -0.0004, 0, 0, 0.0474, 0.13, 0], [0, 0.0185, 0.0089, 0.0033, -0.0004, 0.0, 0, 0.0078, 0, 0.1352, 0], [0, -0.0056, 0, 0.0111, 0, 0, 0.0, 0, 0, 0.1219, 0], [0, 0.0411, 0.0144, 0.0, 0, 0.0078, 0, 0.0, 0.0333, 0.1141, 0], [0.0394, 0, 0.0952, 0.0878, 0.0474, 0, 0, 0.0333, 0.0, 0.0967, 0], [0.0952, 0.0978, 0.13, 0.1315, 0.13, 0.1352, 0.1219, 0.1141, 0.0967, 0.0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]]
    constants = ChemicalConstantsPackage(MWs=[18.01528, 16.04246, 30.06904, 44.09562, 58.1222, 58.1222, 72.14878, 142.28168, 34.08088, 44.0095, 69.723], Tbs=[373.124, 111.65, 184.55, 231.04, 261.45, 272.65, 300.98, 447.25, 213.6, 194.67, 2502.15], Tms=[273.15, 90.75, 90.3, 85.5, 124.2, 135.05, 113.15, 243.225, 187.65, 216.65, 302.9146], Tcs=[647.14, 190.564, 305.32, 369.83, 407.8, 425.12, 460.4, 611.7, 373.2, 304.2, 7620.0], Pcs=[22048320.0, 4599000.0, 4872000.0, 4248000.0, 3640000.0, 3796000.0, 3380000.0, 2110000.0, 8936865.0, 7376460.0, 512630000.0], Vcs=[5.6e-05, 9.86e-05, 0.0001455, 0.0002, 0.000259, 0.000255, 0.000306, 0.000624, 9.85e-05, 9.4e-05, 7.53e-05], Zcs=[0.2294727397218464, 0.2861971332411768, 0.27924206063561996, 0.2762982798699404, 0.27804797802864245, 0.2738549920828424, 0.27018959898694767, 0.2588775438016263, 0.2836910324879899, 0.2741463389591581, 0.6092700613682566], omegas=[0.344, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.49, 0.1, 0.2252, -0.22387758], atomss=[{'H': 2, 'O': 1}, {'C': 1, 'H': 4}, {'C': 2, 'H': 6}, {'C': 3, 'H': 8}, {'C': 4, 'H': 10}, {'C': 4, 'H': 10}, {'C': 5, 'H': 12}, {'C': 10, 'H': 22}, {'H': 2, 'S': 1}, {'C': 1, 'O': 2}, {'Ga': 1}], names=['water', 'methane', 'ethane', 'propane', 'isobutane', 'butane', '2-methylbutane', 'decane', 'hydrogen sulfide', 'carbon dioxide', 'gallium'], CASs=['7732-18-5', '74-82-8', '74-84-0', '74-98-6', '75-28-5', '106-97-8', '78-78-4', '124-18-5', '7783-06-4', '124-38-9', '7440-55-3'])

    HeatCapacityGases = [HeatCapacityGas(CASRN="7732-18-5", MW=18.01528, similarity_variable=0.16652530518537598, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
     HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="74-84-0", MW=30.06904, similarity_variable=0.26605438683775734, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
     HeatCapacityGas(CASRN="74-98-6", MW=44.09562, similarity_variable=0.24945788266499033, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
     HeatCapacityGas(CASRN="75-28-5", MW=58.1222, similarity_variable=0.24087181834135665, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.964580174158493e-21, -3.0000649905010227e-17, 5.236546883165256e-14, -4.72925510291381e-11, 2.3525503498476575e-08, -6.547240983119115e-06, 0.001017840156793089, 0.1718569854807599, 23.54205941874457])),
     HeatCapacityGas(CASRN="106-97-8", MW=58.1222, similarity_variable=0.24087181834135665, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
     HeatCapacityGas(CASRN="78-78-4", MW=72.14878, similarity_variable=0.235624219841278, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-3.3940692195566165e-21, 1.7702350857101215e-17, -3.908639931077815e-14, 4.7271630395909585e-11, -3.371762950874771e-08, 1.4169051870903116e-05, -0.003447090979499268, 0.7970786116058249, -8.8294795805881])),
     HeatCapacityGas(CASRN="124-18-5", MW=142.28168, similarity_variable=0.22490597524572384, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735])),
     HeatCapacityGas(CASRN="7783-06-4", MW=34.08088, similarity_variable=0.08802589604493781, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534])),
     HeatCapacityGas(CASRN="124-38-9", MW=44.0095, similarity_variable=0.0681671002851657, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
     HeatCapacityGas(CASRN="7440-55-3", MW=69.723, similarity_variable=0.014342469486396168, extrapolation="linear", method="POLY_FIT", poly_fit=(302.9146, 7620.0, [-2.6590455185989126e-27, 9.537532078122126e-23, -1.4334165011831938e-18, 1.1713810802494129e-14, -5.652556124156471e-11, 1.6387109312367543e-07, -0.0002782472438036816, 0.25702700843625986, 2.3335385398603137]))]
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)

    eos = PRMIX
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas,
                  'kijs': kijs}
    gas = CEOSGas(eos, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid1 = CEOSLiquid(eos, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid2 = CEOSLiquid(eos, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid3 = CEOSLiquid(eos, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    flasher = FlashVLN(constants, properties, liquids=[liquid1, liquid2, liquid3], gas=gas)
    PT = flasher.flash(T=120, P=5e5, zs=zs)
    assert 4 == PT.phase_count


def test_44_from_database():
    IDs = ['nitrogen', 'carbon dioxide', 'H2S', 'methane', 'ethane', 'propane', 'isobutane', 'butane', 'isopentane', 'pentane', 'Hexane', 'Heptane', 'Octane', 'Nonane', 'Decane', 'Undecane', 'Dodecane', 'Tridecane', 'Tetradecane', 'Pentadecane', 'Hexadecane', 'Heptadecane', 'Octadecane', 'Nonadecane', 'Eicosane', 'Heneicosane', 'Docosane', 'Tricosane', 'Tetracosane', 'Pentacosane', 'Hexacosane', 'Heptacosane', 'Octacosane', 'Nonacosane', 'Triacontane', 'Benzene', 'Toluene', 'Ethylbenzene', 'Xylene', '1,2,4-Trimethylbenzene', 'Cyclopentane', 'Methylcyclopentane', 'Cyclohexane', 'Methylcyclohexane']
    zs = [9.11975115499676e-05, 9.986813065240533e-05, 0.0010137795304828892, 0.019875879000370657, 0.013528874875432457, 0.021392773691700402, 0.00845450438914824, 0.02500218071904368, 0.016114189201071587, 0.027825798446635016, 0.05583179467176313, 0.0703116540769539, 0.07830577180555454, 0.07236459223729574, 0.0774523322851419, 0.057755091407705975, 0.04030134965162674, 0.03967043780553758, 0.03514481759005302, 0.03175471055284055, 0.025411123554079325, 0.029291866298718154, 0.012084986551713202, 0.01641114551124426, 0.01572454598093482, 0.012145363820829673, 0.01103585282423499, 0.010654818322680342, 0.008777712911254239, 0.008732073853067238, 0.007445155260036595, 0.006402875549212365, 0.0052908087849774296, 0.0048199150683177075, 0.015943943854195963, 0.004452253754752775, 0.01711981267072777, 0.0024032720444511282, 0.032178399403544646, 0.0018219517069058137, 0.003403378548794345, 0.01127516775495176, 0.015133143423489698, 0.029483213283483682]

    constants, correlations = ChemicalConstantsPackage.from_IDs(IDs)

    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases)
    flasher = FlashVLN(constants, correlations, liquids=[liq], gas=gas)
    res = flasher.flash(T=300, P=1e5, zs=zs)
    assert res.phase == 'VL'

    # Check we can store this
    output = json.loads(json.dumps(flasher.as_json()))
    assert 'constants' not in output
    assert 'correlations' not in output
    assert 'settings' not in output
    new_flasher = FlashVLN.from_json(output, {'constants': constants, 'correlations': correlations, 'settings': flasher.settings})
    assert new_flasher == flasher




def test_PH_TODO():
    '''Secant is finding false bounds, meaning a failed PT - but the plot looks good. Needs more detail, and enthalpy plot.
    Nice! it fixed itelf.
    Takes 10 seconds though.

    constants = ChemicalConstantsPackage(atomss=[{'H': 2, 'O': 1}, {'C': 2, 'H': 6, 'O': 1}], CASs=['7732-18-5', '64-17-5'], MWs=[18.01528, 46.06844], names=['water', 'ethanol'], omegas=[0.344, 0.635], Pcs=[22048320.0, 6137000.0], Tcs=[647.14, 514.0])
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965]))]
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)

    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=300.0, P=1e4, zs=[.5, .5])
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=300.0, P=1e4, zs=[.5, .5])
    flashN = FlashVLN(constants, properties, liquids=[liq], gas=gas)
    flashN.flash(H=-1e4, P=2e5, zs=[.5, .5]).H()

    '''

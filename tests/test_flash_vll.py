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
from fluids.numerics import assert_close, assert_close1d, assert_close2d, assert_close3d
from fluids.numerics import *
from math import *
import json
import os
import numpy as np

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
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases)
    eos_kwargs = dict(Tcs=Tcs, Pcs=Pcs, omegas=omegas, kijs=kijs)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)


    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
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

    res_PR = FlashVLN(constants, properties, liquids=[liq], gas=gas).flash(T=298.15, VF=1, zs=zs)
    res_SRK = FlashVLN(constants, properties, liquids=[liq_SRK], gas=gas).flash(T=298.15, VF=1, zs=zs)
    assert res_PR.G() > res_SRK.G()
    res_both_liquids = FlashVLN(constants, properties, liquids=[liq, liq_SRK], gas=gas).flash(T=298.15, VF=1, zs=zs)
    assert_close(res_both_liquids.G(), res_SRK.G(), rtol=1e-7)
    assert_close(res_PR.P, 6022.265230194498, rtol=1e-5)
    assert_close(res_SRK.P, 5555.019566177178, rtol=1e-5)

    res_PR = FlashVLN(constants, properties, liquids=[liq], gas=gas).flash(P=6000.0, VF=1, zs=zs)
    res_SRK = FlashVLN(constants, properties, liquids=[liq_SRK], gas=gas).flash(P=6000.0, VF=1, zs=zs)
    assert res_PR.G() > res_SRK.G()
    res_both_liquids = FlashVLN(constants, properties, liquids=[liq, liq_SRK], gas=gas).flash(P=6000.0, VF=1, zs=zs)
    assert_close(res_both_liquids.G(), res_SRK.G(), rtol=1e-7)
    assert_close(res_PR.T, 298.0822786634035, rtol=1e-5)
    assert_close(res_SRK.T, 299.5323133142487, rtol=1e-5)



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
    flashN = FlashVLN(constants, properties, liquids=[liq], gas=gas)

    # Extremely low pressure point - genuinely is a VL point, but G is lower by a tiny amount
    res = flashN.flash(T=T, P=P, zs=zs)
    assert_close1d(res.betas, [0.9973290812443733, 0.00267091875562675])

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
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534]))], )
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, kijs=kijs)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, properties, liquids=[liq, liq, liq], gas=gas)
    res = flashN.flash(T=T, P=P, zs=zs)
    assert res.phase_count == 1
    assert res.gas is not None


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
    res = flashN.flash(T=360, P=1e5, zs=zs) # VL
    assert_close1d([i.rho_mass() for i in res.phases], [1.390005729844191, 718.4273271368141])

    res = flashN.flash(T=361, P=1e5, zs=zs) # VLL
    assert_close1d(res.water_phase.zs, [7.619975052224755e-05, 0.9989622883894996, 0.0009615118599771799])
    assert_close1d(res.gas.zs, [0.2384009970908654, 0.57868399351809, 0.18291500939104438])
    assert res.phase_count == 3

    res = flashN.flash(T=364, P=1e5, zs=zs) # VL
    assert_close1d([i.rho_mass() for i in res.phases], [1.203792756430329, 715.8202252076906])
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
    CASs = ['74-82-8', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '110-54-3', 'C7_16', 'C17+']

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
    assert 3 == len(set([id(i) for i in flashN.phases_at(T=T, P=P, zs=zs)[1]]))

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

    key = '%s - %s - %s - %s liquids' %(eos.__name__, ', '.join(IDs), ', '.join('%g' %zi for zi in zs), len(flashN.liquids))
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
                                               VaporPressures=[VaporPressure(poly_fit=(63.2, 126.18199999999999, [5.490876411024536e-15, -3.709517805130509e-12, 1.0593254238679989e-09, -1.6344291780087318e-07, 1.4129990091975526e-05, -0.0005776268289835264, -0.004489180523814208, 1.511854256824242, -36.95425216567675])),
        VaporPressure(poly_fit=(90.4, 305.312, [-1.1908381885079786e-17, 2.1355746620587145e-14, -1.66363909858873e-11, 7.380706042464946e-09, -2.052789573477409e-06, 0.00037073086909253047, -0.04336716238170919, 3.1418840094903784, -102.75040650505277])),],
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
                                               VaporPressures=[VaporPressure(poly_fit=(63.2, 126.18199999999999, [5.490876411024536e-15, -3.709517805130509e-12, 1.0593254238679989e-09, -1.6344291780087318e-07, 1.4129990091975526e-05, -0.0005776268289835264, -0.004489180523814208, 1.511854256824242, -36.95425216567675])),
        VaporPressure(poly_fit=(90.4, 305.312, [-1.1908381885079786e-17, 2.1355746620587145e-14, -1.66363909858873e-11, 7.380706042464946e-09, -2.052789573477409e-06, 0.00037073086909253047, -0.04336716238170919, 3.1418840094903784, -102.75040650505277])),],
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


def test_PH_TODO():
    '''Secant is finding false bounds, meaning a failed PT - but the plot looks good. Needs more detail, and enthalpy plot.

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

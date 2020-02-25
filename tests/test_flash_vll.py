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
from thermo import *
from fluids.numerics import *
from math import *
import json
import os
import numpy as np

def test_water_C1_C8():
    T = 298.15
    P = 101325.0
    omegas = [0.344, 0.008, 0.394]
    Tcs = [647.14, 190.564, 568.7]
    Pcs = [22048320.0, 4599000.0, 2490000.0]
    kijs=[[0,0, 0],[0,0, 0.0496], [0,0.0496,0]]
    zs = [1.0/3.0]*3
    N = len(zs)
    
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                         HeatCapacityGas(best_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                         HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303]))]
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[18.01528, 16.04246, 114.22852], 
                                         CASs=['7732-18-5', '74-82-8', '111-65-9'])
    properties = PropertyCorrelationPackage(constants=constants, HeatCapacityGases=HeatCapacityGases)
    eos_kwargs = dict(Tcs=Tcs, Pcs=Pcs, omegas=omegas, kijs=kijs)
    
    gas = EOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    
    
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    res = flashN.flash(T=T, P=P, zs=zs)

    assert_allclose(res.water_phase.zs, [0.9999990988582429, 9.011417571269618e-07, 9.57378962042325e-17])
    assert_allclose(res.gas.zs, [0.026792655758364814, 0.9529209534990141, 0.020286390742620692])
    assert res.phase_count == 3
    
    # Gas phase - high T
    highT = flashN.flash(T=500.0, P=P, zs=zs)
    assert highT.phase_count == 1
    assert highT.gas is not None
    
    
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
    properties = PropertyCorrelationPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                HeatCapacityGas(best_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                                                                HeatCapacityGas(best_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
                                                                HeatCapacityGas(best_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
                                                                HeatCapacityGas(best_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
                                                                HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                HeatCapacityGas(best_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                HeatCapacityGas(best_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
                                                                HeatCapacityGas(best_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534])),])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    
    gas = EOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    res = flashN.flash(T=T, P=P, zs=zs)
    
    assert res.phase_count == 3
    assert_allclose(res.betas[0], 0.9254860647854957, rtol=1e-6)
    assert_allclose(res.betas_liquids[res.water_phase_index], 0.027819781620531732, rtol=1e-5)
    
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
    properties = PropertyCorrelationPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                    HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303])),
                                                                    HeatCapacityGas(best_fit=(266.15, 780.0, [-1.4884362017902977e-20, 5.2260426653411555e-17, -6.389395767183489e-14, 1.5011276073418378e-11, 3.9646353076966065e-08, -4.503513372576425e-05, 0.020923507683244157, -4.012905599723838, 387.9369199281481])),
                                                                    HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    
    gas = EOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flashN = FlashVLN(constants, properties, liquids=[liq, liq, liq], gas=gas)
    VLL = flashN.flash(T=T, P=P, zs=zs)
    assert VLL.phase_count == 3
    assert_allclose(VLL.gas.zs, [0.9588632822157593, 0.014347647576433926, 2.8303984647442933e-06, 0.02678623980934197], )


def test_binary_phase_switch():
    # Example from Kodama, Daisuke, Ryota Sato, Aya Haneda, and Masahiro Kato. “High-Pressure Phase Equilibrium for Ethylene + Ethanol at 283.65 K.” Journal of Chemical & Engineering Data 50, no. 1 (January 1, 2005): 122–24. https://doi.org/10.1021/je049796y.
    # Shows a phase switch across the VLL line (which can never be flashed at TP)
    zs = [.8, 0.2]
    #m = Mixture(['ethylene', 'ethanol'], zs=zs)
    T = 283.65
    P = 4690033.135557525
    constants = ChemicalConstantsPackage(Tcs=[282.34, 514.0], Pcs=[5041000.0, 6137000.0], omegas=[0.085, 0.635], MWs=[28.05316, 46.06844], CASs=['74-85-1', '64-17-5'])
    properties = PropertyCorrelationPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [-3.2701693466919565e-21, 1.660757962278189e-17, -3.525777713754962e-14, 4.01892664375958e-11, -2.608749347072186e-08, 9.23682495982131e-06, -0.0014524032651835623, 0.09701764355901257, 31.034399100170667])),
                                                                                    HeatCapacityGas(best_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas, kijs=[[0.0, -.0057], [-.0057, 0.0]])
    
    
    gas = EOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    
    
    LL = flashN.flash(P=P+100, T=T, zs=zs)
    assert_allclose(LL.lightest_liquid.zs, [0.9145548102807435, 0.08544518971925665])
    assert_allclose(LL.heaviest_liquid.zs, [0.5653776843338143, 0.4346223156661858])
    assert LL.phase_count == 2
    VL = flashN.flash(P=P-100, T=T, zs=zs)
    assert VL.phase_count == 2
    [i.rho_mass() for i in VL.phases]
    assert_allclose(VL.liquid0.zs, [0.5646523538677704, 0.43534764613222976])
    assert_allclose(VL.gas.zs, [0.9963432818384895, 0.0036567181615104662])
    
def test_three_phase_ethylene_ethanol_nitrogen():
    zs = [.8, 0.19, .01]
    # m = Mixture(['ethylene', 'ethanol', 'nitrogen'], zs=zs)
    T = 283.65
    P = 4690033.135557525
    constants = ChemicalConstantsPackage(Tcs=[282.34, 514.0, 126.2], Pcs=[5041000.0, 6137000.0, 3394387.5], omegas=[0.085, 0.635, 0.04], MWs=[28.05316, 46.06844, 28.0134], CASs=['74-85-1', '64-17-5', '7727-37-9'])
    properties =PropertyCorrelationPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [-3.2701693466919565e-21, 1.660757962278189e-17, -3.525777713754962e-14, 4.01892664375958e-11, -2.608749347072186e-08, 9.23682495982131e-06, -0.0014524032651835623, 0.09701764355901257, 31.034399100170667])),
                                                                                    HeatCapacityGas(best_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),
                                                                                    HeatCapacityGas(best_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas, kijs=[[0.0, -.0057, 0.0], [-.0057, 0.0, 0.0], [0.0, 0.0, 0.0]])
    
    
    gas = EOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    
    # SS trivial solution needs to be checked
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    one_phase = flashN.flash(T=366.66666666666674, P=13335214.32163324, zs=zs)
    assert one_phase.liquid0
    assert one_phase.phase_count == 1
    
    # over 1000 iterations to converge
    many_iter_VL = flashN.flash(T=266.66666666666663, P=7498942.093324558, zs=zs)
    assert_allclose(many_iter_VL.betas, [0.1646137076373939, 0.8353862923626061])
    
    # SS trivial solution to all Ks under 1
    one_phase = flashN.flash(T=283.3333333333333, P=10000000.0, zs=zs)
    assert one_phase.liquid0
    assert one_phase.phase_count == 1
    
    # point RR was making SS convergence die
    bad_RR = flashN.flash(T=220.2020202020202, P=37649358.067924485, zs=zs)
    assert_allclose(bad_RR.betas, [0.13605690662613873, 0.8639430933738612])

    # Point where the three phase calculations converge to negative betas
    false3P = flashN.flash(T=288.56, P=7.3318e6, zs=zs)
    assert_allclose(false3P.betas, [0.24304503666565203, 0.756954963334348])
    assert false3P.liquid0
    assert false3P.liquid1
    
    # Three phase region point
    res = flashN.flash(T=200, P=6e5, zs=zs)
    assert_allclose(res.G(), -2873.6029915490544, rtol=1e-5)
    assert res.phase_count == 3
    
def test_ethanol_water_cyclohexane_3_liquids():
    zs = [.35, .06, .59]
    # m = Mixture(['ethanol', 'water','cyclohexane'], zs=zs)
    T = 298.15
    P = 1e5

    constants = ChemicalConstantsPackage(Tcs=[514.0, 647.14, 532.7], Pcs=[6137000.0, 22048320.0, 3790000.0], omegas=[0.635, 0.344, 0.213], MWs=[46.06844, 18.01528, 84.15948], CASs=['64-17-5', '7732-18-5', '110-82-7'])
    
    properties = PropertyCorrelationPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),
                                                                HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                HeatCapacityGas(best_fit=(100.0, 1000.0, [-2.974359561904494e-20, 1.4314483633408613e-16, -2.8871179135718834e-13, 3.1557554273363386e-10, -2.0114283147465467e-07, 7.426722872136983e-05, -0.014718631011050769, 1.6791476987773946, -34.557986234881355]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas)
    
    gas = EOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    res = flashN.flash(T=150, P=1e6, zs=zs)
    assert len(res.liquids) == 3
    assert_allclose(res.heaviest_liquid.zs, [7.29931119445944e-08, 0.9999999270068883, 1.6797752584629787e-26], atol=1e-15)
    assert_allclose(res.lightest_liquid.zs, [0.9938392499953312, 0.0035632064699575947, 0.0025975435347112977])
    
def test_butanol_water_ethanol_3P():
    zs = [.25, 0.7, .05]
    # m = Mixture(['butanol', 'water', 'ethanol'], zs=zs)
    
    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0], omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844], CASs=['71-36-3', '7732-18-5', '64-17-5'])
    properties = PropertyCorrelationPackage(constants=constants,
                                            HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                                            HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                            HeatCapacityGas(best_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),],)
    T = 298.15
    P = 1e5
    
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas)
    
    gas = EOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    
    # LL solution exists and is found at this point - but does not have as lower of a G
    res = flashN.flash(T=400, P=1e5, zs=zs)
    assert res.gas
    assert res.phase_count == 1
    
    #  Some bad logic in LL
    failed_logic_LL = flashN.flash(T=186.48648648648657, P=120526.09368708414, zs=zs)
    assert_allclose(failed_logic_LL.betas, [0.6989623717730211, 0.3010376282269789])


    # Test 5 points going  through LL to VL to VLL to VL again to V
    res = flashN.flash(T=354, P=1e5, zs=zs) # LL
    assert_allclose([i.rho_mass() for i in res.phases], [722.1800166595312, 656.8373555931618])
    res = flashN.flash(T=360, P=1e5, zs=zs) # VL
    assert_allclose([i.rho_mass() for i in res.phases], [1.390005729844191, 718.4273271368141])
    
    res = flashN.flash(T=361, P=1e5, zs=zs) # VLL
    assert_allclose(res.water_phase.zs, [7.619975052224755e-05, 0.9989622883894996, 0.0009615118599771799])
    assert_allclose(res.gas.zs, [0.2384009970908654, 0.57868399351809, 0.18291500939104438])
    assert res.phase_count == 3
    
    res = flashN.flash(T=364, P=1e5, zs=zs) # VL
    assert_allclose([i.rho_mass() for i in res.phases], [1.203792756430329, 715.8202252076906])
    res = flashN.flash(T=366, P=1e5, zs=zs) # V
    assert_allclose([i.rho_mass() for i in res.phases], [1.1145608982480968])



def test_phases_at():
    T = 300
    P = 1e5
    zs = [.8, 0.2]
    constants = ChemicalConstantsPackage(Tcs=[282.34, 514.0], Pcs=[5041000.0, 6137000.0], omegas=[0.085, 0.635], 
                                         MWs=[28.05316, 46.06844], CASs=['74-85-1', '64-17-5'])
    properties =PropertyCorrelationPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [-3.2701693466919565e-21, 1.660757962278189e-17, -3.525777713754962e-14, 4.01892664375958e-11, -2.608749347072186e-08, 9.23682495982131e-06, -0.0014524032651835623, 0.09701764355901257, 31.034399100170667])),
                                                                                    HeatCapacityGas(best_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965]))])
    
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs,
                      omegas=constants.omegas, kijs=[[0.0, -.0057], [-.0057, 0.0]])
    
    gas = EOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq2 = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq3 = EOSLiquid(VDWMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flashN = FlashVLN(constants, properties, liquids=[liq2, liq3, liq, liq3, liq, liq2, liq2], gas=gas)
    assert 3 == len(set([id(i) for i in flashN.phases_at(T=T, P=P, zs=zs)[1]]))
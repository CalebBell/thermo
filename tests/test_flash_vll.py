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
    zs = [.65, .13, .09, .05, .03, .03, .02, .003, 1e-6]
    m = Mixture(['CH4', 'C2', 'C3', 'C4', 'C5', 'water', 'nitrogen', 'CO2', 'H2S'], zs=zs)
    zs = m.zs
    
    T = 300.0
    P = 3000e3
    constants = m.constants
    properties = m.properties()
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
    m = Mixture(['methane', 'octane', 'TEG', 'water'], zs=zs)
    zs = m.zs
    
    T = 298.15
    P = 101325.0
    constants = m.constants
    properties = m.properties()
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    
    gas = EOSGas(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq2 = EOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    
    flashN = FlashVLN(constants, properties, liquids=[liq, liq, liq], gas=gas)
    VLL = flashN.flash(T=T, P=P, zs=zs)
    assert_allclose(VLL.gas.zs, [0.9588632822157593, 0.014347647576433926, 2.8303984647442933e-06, 0.02678623980934197], )
    assert VLL.phase_count == 3

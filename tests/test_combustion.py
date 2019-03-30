# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import numpy as np
import pandas as pd
from thermo.combustion import *


def test_Hcombustion():
    dH = Hcombustion({'H': 4, 'C': 1, 'O': 1}, Hf=-239100)
    assert_allclose(dH, -726024.0) 
    
def test_combustion_products():
    C60_products = combustion_products({'C': 60})
    assert C60_products['CO2'] == 60
    assert C60_products['O2_required'] == 60
    
    methanol_products = combustion_products({'H': 4, 'C': 1, 'O': 1})
    assert methanol_products['H2O'] == 2
    assert methanol_products['CO2'] == 1
    assert methanol_products['SO2'] == 0
    assert methanol_products['O2_required'] == 1.5
    assert methanol_products['N2'] == 0
    assert methanol_products['HF'] == 0
    assert methanol_products['Br2'] == 0
    assert methanol_products['HCl'] == 0
    assert methanol_products['P4O10'] == 0


def test_air_fuel_ratio_solver():
    Vm_air = 0.024936627188566596
    Vm_fuel = 0.024880983160354486
    MW_air = 28.850334
    MW_fuel = 17.86651
    n_fuel = 5.0
    n_air = 25.0
    
    strings = ['mole', 'mass', 'volume']
    ratios = [n_air/n_fuel, MW_air/MW_fuel*5, Vm_air/Vm_fuel*5]
    
    ans_expect = [n_air, n_fuel]
    ans_expect_full = ans_expect + ratios
    
    for ratio, s in zip(ratios, strings):
        for air_spec, fuel_spec in zip([None, n_air], [n_fuel, None]):
            for full in (True, False):
                ans = air_fuel_ratio_solver(ratio=ratio, Vm_air=Vm_air, Vm_fuel=Vm_fuel,
                                            MW_air=MW_air,  MW_fuel=MW_fuel, n_air=air_spec,
                                            n_fuel=fuel_spec, basis=s, full_info=full)
                if full:
                    assert_allclose(ans, ans_expect_full)
                else:
                    assert_allclose(ans, ans_expect)
    
    


def assert_comb_dict_equal(calc, expect):
    for k, v_expect in expect.items():
        v_calc = calc[k]
        assert_allclose(v_expect, v_calc)

def fuel_air_spec_solver_checker(inputs, ans, func=fuel_air_spec_solver):
    # One intensive variable, one extensive variable
    for n_spec, n_name in zip(['n_fuel', 'n_air', 'n_out'], [ans['n_fuel'], ans['n_air'], ans['n_out']]):
        for i_spec, i_name in zip(['O2_excess', 'frac_out_O2', 'frac_out_O2_dry'], [ans['O2_excess'], ans['frac_out_O2'], ans['frac_out_O2_dry']]):
            d2 = inputs.copy()
            d2.update({n_spec: n_name, i_spec: i_name})

            calc = func(**d2)
            assert_comb_dict_equal(calc, ans)

    for n_spec, n_name in zip(['n_fuel', 'n_air'], [ans['n_fuel'], ans['n_air']]):
        for ratio_name, ratio in zip(['mass', 'mole', 'volume'], [ans['mass_ratio'], ans['mole_ratio'], ans['volume_ratio']]):
            kwargs = inputs.copy()
            kwargs.update({n_spec: n_name, 'ratio': ratio, 'ratio_basis': ratio_name})

            calc = func(**kwargs)
            assert_comb_dict_equal(calc, ans)

#     Two extensive variables check
    for n_fuel in (None, ans['n_fuel']):
        for n_air in (None, ans['n_air']):
            for n_out in (None, ans['n_out']):
                if sum(i is not None for i in (n_fuel, n_air, n_out)) == 2:

                    calc = func(n_fuel=n_fuel, n_air=n_air, n_out=n_out, **inputs)
                    assert_comb_dict_equal(calc, ans)


def test_fuel_air_spec_solver():
    ans_N7_messy_fuel = {'O2_excess': 0.5813397129186602,
 'frac_out_O2': 0.0712025316455696,
 'frac_out_O2_dry': 0.08118672947779891,
 'mass_ratio': 24.83317358818291,
 'mole_ratio': 16.0,
 'n_air': 80,
 'n_fuel': 5,
 'n_out': 85.32000000000001,
 'ns_out': [63.325, 6.074999999999999, 0.0, 0.0, 0.0, 10.492500000000001, 5.4275],
 'volume_ratio': 16.032156677022826,
 'zs_out': [0.742205813408345, 0.0712025316455696, 0.0, 0.0, 0.0, 0.12297819971870605, 0.06361345522737927]}
    
    inputs_N7_messy_fuel = {'CASs': ['7727-37-9',  '7782-44-7',  '74-82-8',  '74-84-0',  '74-98-6',  '7732-18-5',  '124-38-9'],
     'MW_air': 28.793413510000008,
     'MW_fuel': 18.551580390000005,
     'Vm_air': 0.024932453821680217,
     'Vm_fuel': 0.024882445274415996,
     'zs_air': [.79, .205, 0, 0, 0, .0045, .0005],
     'atomss': [{'N': 2},  {'O': 2},  {'C': 1, 'H': 4},  {'C': 2, 'H': 6},  {'C': 3, 'H': 8},  {'H': 2, 'O': 1}, {'C': 1, 'O': 2}],
     'zs_fuel': [.025, .025, .85, .07, .029, .0005, .0005]}

    all_inputs = [inputs_N7_messy_fuel]
    all_ans = [ans_N7_messy_fuel]
    
    for inputs, ans in zip(all_inputs, all_ans):
        fuel_air_spec_solver_checker(inputs, ans)

def test_fuel_air_third_spec_solver():

    inputs_N7_messy_fuel = {'CASs': ['7727-37-9',  '7782-44-7',  '74-82-8',  '74-84-0',  '74-98-6',  '7732-18-5',  '124-38-9'],
         'MW_air': 28.793413510000008,
         'MW_fuel': 18.551580390000005,
         'MW_third': 22.594160765550242,
         'Vm_air': 0.024932453821680217,
         'Vm_fuel': 0.024882445274415996,
         'Vm_third': 0.024820446149354414,
         'n_third': 1,
         'zs_air': [.79, .205, 0, 0, 0, .0045, .0005],
         'zs_third': [0.1, 0.005, 0.5, 0.39, 0, 0.005, 0],
         'zs_fuel': [.025, .025, .85, .07, .029, .0005, .0005],
         'atomss': [{'N': 2},  {'O': 2},  {'C': 1, 'H': 4},  {'C': 2, 'H': 6},  {'C': 3, 'H': 8},  {'H': 2, 'O': 1}, {'C': 1, 'O': 2}],
    }
    
    ans_N7_messy_fuel = {'O2_excess': 0.289894654701522,
     'frac_out_O2': 0.04294053054383636,
     'frac_out_O2_dry': 0.0503063746233793,
     'mass_ratio': 19.96906710268543,
     'mole_ratio': 13.333333333333334,
     'n_air': 80,
     'n_fuel': 5,
     'n_out': 86.51500000000001,
     'ns_out': [63.42500000000001,  3.7150000000000034,  0.0,  0.0,  0.0,  12.667499999999999,  6.707499999999999],
     'volume_ratio': 13.365681067247072,
     'zs_out': [0.7331098653412703,  0.04294053054383636,  0.0,  0.0,  0.0, 0.14641969600647284,  0.07752990810842048]}
    
    
    all_inputs = [inputs_N7_messy_fuel]
    all_ans = [ans_N7_messy_fuel]
    
    for inputs, ans in zip(all_inputs, all_ans):
        fuel_air_spec_solver_checker(inputs, ans, func=fuel_air_third_spec_solver)

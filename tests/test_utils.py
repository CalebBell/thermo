# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from chemicals.utils import *
from thermo.utils import *
from thermo.stream import Stream
from fluids.numerics import assert_close, assert_close1d

def test_allclose_variable():
    x = [2.7244322249597719e-08, 3.0105683900110473e-10, 2.7244124924802327e-08, 3.0105259397637556e-10, 2.7243929226310193e-08, 3.0104990272770901e-10, 2.7243666849384451e-08, 3.0104101821236015e-10, 2.7243433745917367e-08, 3.0103707421519949e-10]
    y = [2.7244328304561904e-08, 3.0105753470546008e-10, 2.724412872417824e-08,  3.0105303055834564e-10, 2.7243914341030203e-08, 3.0104819238021998e-10, 2.7243684057561379e-08, 3.0104299541023674e-10, 2.7243436694839306e-08, 3.010374130526363e-10]

    assert allclose_variable(x, y, limits=[.0, .5], rtols=[1E-5, 1E-6])
        
    with pytest.raises(Exception):
        assert allclose_variable(x, y, limits=[.0, .1], rtols=[1E-5, 1E-6])
        
    with pytest.raises(Exception):
        allclose_variable(x, y[1:], limits=[.0, .5], rtols=[1E-5, 1E-6])

    with pytest.raises(Exception):
        ans = allclose_variable(x, y, limits=[.0, .1])


    x = [1,1,1,1,1,1,1,1,1]
    y = [.9,.9,.9,.9,.9,.9,.9,.9, .9]
    assert allclose_variable(x, y, limits=[.0], atols=[.1])


    

def test_phase_select_property():
    assert 150 == phase_select_property(phase='s', s=150, l=10)
    assert None == phase_select_property(phase='s', l=1560.14)
    assert 3312 == phase_select_property(phase='g', l=1560.14, g=3312.)
    assert 1560.14 == phase_select_property(phase='l', l=1560.14, g=3312.)
    assert None == phase_select_property(phase='two-phase', l=1560.14, g=12421.0)
    assert None == phase_select_property(phase=None, l=1560.14, g=12421.0)
    with pytest.raises(Exception):
        phase_select_property(phase='notalphase', l=1560.14, g=12421.0)



def test_identify_phase():
    # Above the melting point, higher pressure than the vapor pressure
    assert 'l' == identify_phase(T=280, P=101325, Tm=273.15, Psat=991)

    # Above the melting point, lower pressure than the vapor pressure
    assert 'g' == identify_phase(T=480, P=101325, Tm=273.15, Psat=1791175)

    # Above the melting point, above the critical pressure (no vapor pressure available)
    assert 'g' == identify_phase(T=650, P=10132500000, Tm=273.15, Psat=None, Tc=647.3)

    # No vapor pressure specified, under the melting point
    assert 's' == identify_phase(T=250, P=100, Tm=273.15)

    # No data, returns None
    assert None == identify_phase(T=500, P=101325)

    # No Tm, under Tb, at normal atmospheric pressure
    assert 'l' == identify_phase(T=200, P=101325, Tb=373.15)

    # Incorrect case by design:
    # at 371 K, Psat is 93753 Pa, meaning the actual phase is gas
    assert 'l' == identify_phase(T=371, P=91000, Tb=373.15)
    
    # Very likely wrong, pressure has dropped substantially
    # Consider behavior
    assert 'l' == identify_phase(T=373.14, P=91327, Tb=373.15)

    # Above Tb, while still atmospheric == gas
    assert 'g' == identify_phase(T=400, P=101325, Tb=373.15)

    # Above Tb, 1 MPa == None - don't try to guess
    assert None == identify_phase(T=400, P=1E6, Tb=373.15)

    # Another wrong point - at 1 GPa, should actually be a solid as well
    assert 'l' == identify_phase(T=371, P=1E9, Tb=373.15)

    # At the critical point, consider it a gas
    assert 'g' == identify_phase(T=647.3, P=22048320.0, Tm=273.15, Psat=22048320.0, Tc=647.3)

    # Just under the critical point
    assert 'l' == identify_phase(T=647.2, P=22048320.0, Tm=273.15, Psat=22032638.96749514, Tc=647.3)



def test_TDependentProperty():
    TEST_METHOD_1 = 'Test method 1'
    TEST_METHOD_2 = 'Test method 2'

    EtOH = TDependentProperty(CASRN='67-56-1')
    # Test pre-run info
    assert EtOH.method == None
    assert EtOH.forced == False
    assert EtOH.name == 'Property name'
    assert EtOH.units == 'Property units'
    assert EtOH.interpolation_T == None
    assert EtOH.interpolation_property == None
    assert EtOH.interpolation_property_inv == None
    assert EtOH.tabular_extrapolation_permitted == True

    # Test __init__
    assert EtOH.CASRN == '67-56-1'
    assert EtOH.ranked_methods == [TEST_METHOD_2, TEST_METHOD_1]

    # Test load_all_methods
    assert EtOH.all_methods == set([TEST_METHOD_2, TEST_METHOD_1])
    assert EtOH.TEST_METHOD_1_Tmin == 200
    assert EtOH.TEST_METHOD_2_Tmin == 300
    assert EtOH.TEST_METHOD_1_Tmax == 350
    assert EtOH.TEST_METHOD_2_Tmax == 400
    assert EtOH.TEST_METHOD_1_coeffs == [1, .002]
    assert EtOH.TEST_METHOD_2_coeffs == [1, .003]

    # Test test_property_validity alone
    assert all([EtOH.test_property_validity(i) for i in [.1, 1.1, 100]])
    assert not any([EtOH.test_property_validity(i) for i in [-.1, 1.1E10, 2**20]])
    assert not EtOH.test_property_validity(1+1j)

    # Test test_method_validity
    assert [False, True] == [EtOH.test_method_validity(i, TEST_METHOD_1) for i in [199, 201]]
    assert [False, True] == [EtOH.test_method_validity(i, TEST_METHOD_2) for i in [299, 301]]
    assert [True, False] == [EtOH.test_method_validity(i, TEST_METHOD_1) for i in [349, 351]]
    assert [True, False] == [EtOH.test_method_validity(i, TEST_METHOD_2) for i in [399, 401]]
    # With pytest.raise(Exception) BAD METHOD here
    # TODO: Interpolate

    # Test select_valid_methods without user_methods
    assert [TEST_METHOD_2, TEST_METHOD_1] == EtOH.select_valid_methods(320) # Both in range, correctly ordered
    assert [TEST_METHOD_1] == EtOH.select_valid_methods(210) # Choice 2 but only one available
    assert [TEST_METHOD_2] == EtOH.select_valid_methods(390) # Choice 1 but only one available

    # Test calculate
    # Mid, all methods, and with __call__, twice
    assert 1.9 == EtOH.T_dependent_property(300)    
    assert 1.9 == EtOH(300)
    assert 1.9 == EtOH(300)
    assert 1.9 == EtOH.calculate(300, TEST_METHOD_2)
    assert 1.6 == EtOH.calculate(300, TEST_METHOD_1)
    # High both methods
    assert 2.125 == EtOH.T_dependent_property(375)
    assert TEST_METHOD_2 == EtOH.method # Test this gets set
    assert 2.125 == EtOH.calculate(375, TEST_METHOD_2)
    assert 1.75 == EtOH.calculate(375, TEST_METHOD_1) # Over limit, but will still calculate
    # Low both methods
    assert 1.5 == EtOH.T_dependent_property(250)
    assert TEST_METHOD_1 == EtOH.method # Test this gets set
    assert 1.5 == EtOH.calculate(250, TEST_METHOD_1)
    assert 1.75 == EtOH.calculate(250, TEST_METHOD_2)
    # Lower and higher than any methods
    assert None == EtOH.T_dependent_property(150)
    assert None == EtOH.T_dependent_property(500)


    # Test some failures
    with pytest.raises(Exception):
        EtOHFail = TDependentProperty(CASRN='67-56-1')
        EtOHFail.set_user_methods([], forced=True)
    with pytest.raises(Exception):
        EtOHFail = TDependentProperty(CASRN='67-56-1')
        EtOHFail.set_user_methods(['NOTAMETHOD'])
    with pytest.raises(Exception):
        EtOHFail = TDependentProperty(CASRN='67-56-1')
        EtOHFail.test_method_validity(300, 'NOTAMETHOD')


    # Test with user methods
    EtOH = TDependentProperty(CASRN='67-56-1')
    EtOH.set_user_methods(TEST_METHOD_1)
    assert EtOH.user_methods == [TEST_METHOD_1]
    assert 1.6 == EtOH.T_dependent_property(300)

    EtOH.set_user_methods(TEST_METHOD_2)
    assert EtOH.user_methods == [TEST_METHOD_2]
    assert 1.5 == EtOH.T_dependent_property(250) # Low, fails to other method though
    EtOH.set_user_methods(TEST_METHOD_2, forced=True)
    assert None == EtOH.T_dependent_property(250) # Test not calculated if user method not specified

    EtOH.set_user_methods([TEST_METHOD_1, TEST_METHOD_2])
    assert EtOH.user_methods == [TEST_METHOD_1, TEST_METHOD_2]
    assert 1.6 == EtOH.T_dependent_property(300)
    assert 1.5 == EtOH.T_dependent_property(250)


    EtOH = TDependentProperty(CASRN='67-56-1')
    Ts = [195, 205, 300, 400, 450]
    props = [1.2, 1.3, 1.7, 1.9, 2.5]

    # Test naming and retrieving
    EtOH.set_tabular_data(Ts=Ts, properties=props)
    assert set(['Tabular data series #0', 'Test method 1', 'Test method 2']) == EtOH.all_methods
    EtOH.set_tabular_data(Ts=Ts, properties=props)
    assert set(['Tabular data series #1','Tabular data series #0', 'Test method 1', 'Test method 2']) == EtOH.all_methods
    EtOH.set_tabular_data(Ts=Ts, properties=props, name='awesome')
    assert set(['awesome', 'Tabular data series #1','Tabular data series #0', 'Test method 1', 'Test method 2']) == EtOH.all_methods


    # Test data retrieval
    assert EtOH.tabular_data['Tabular data series #1'] == (Ts, props)

    # Test old methods settings removed
    assert EtOH.method == None
    assert EtOH.sorted_valid_methods == []

    # Test naming and retrieving with user methods
    EtOH = TDependentProperty(CASRN='67-56-1')
    EtOH.set_user_methods(TEST_METHOD_1)
    EtOH.set_tabular_data(Ts=Ts, properties=props)
    assert EtOH.user_methods == ['Tabular data series #0', 'Test method 1']
    EtOH.set_tabular_data(Ts=Ts, properties=props, name='hi')
    assert EtOH.user_methods == ['hi', 'Tabular data series #0', 'Test method 1']

    with pytest.raises(Exception):
        EtOH.set_tabular_data(Ts=[195, 205, 300, 400, 450], properties=[1.2, 1.3, 1.7, 1.9, -1], name='awesome')

    EtOH.set_tabular_data(Ts=[195, 205, 300, 400, 450], properties=[1.2, 1.3, 1.7, 1.9, -1], name='awesome', check_properties=False)


    # Test interpolation
    EtOH = TDependentProperty(CASRN='67-56-1')
    Ts = [200, 250, 300, 400, 450]
    props = [1.2, 1.3, 1.4, 1.5, 1.6]

    # Test the cubic spline
    EtOH.set_tabular_data(Ts=Ts, properties=props, name='test_set')
    assert_allclose(1.2,  EtOH.T_dependent_property(200))
    assert_allclose(1.1, EtOH.T_dependent_property(150))
    assert_allclose(1.7, EtOH.T_dependent_property(500))
    assert_allclose(1.35441088517, EtOH.T_dependent_property(275), rtol=1E-3)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.T_dependent_property(500)

    # Test linear interpolation if n < 5:
    EtOH = TDependentProperty(CASRN='67-56-1')
    Ts = [200, 250, 400, 450]
    props = [1.2, 1.3, 1.5, 1.6]

    # Test the cubic spline
    EtOH.set_tabular_data(Ts=Ts, properties=props, name='test_set')
    assert_allclose(4/3.,EtOH.T_dependent_property(275))

    # Set the interpolation methods
    EtOH = TDependentProperty(CASRN='67-56-1')
    Ts = [200, 250, 400, 450]
    props = [1.2, 1.3, 1.5, 1.6]
    EtOH.interpolation_T = lambda T: log(T)
    EtOH.interpolation_property = lambda prop: 1./prop
    EtOH.interpolation_property_inv = lambda prop: 1./prop

    # Test the linear interpolation with transform
    EtOH.set_tabular_data(Ts=Ts, properties=props, name='test_set')
    assert_allclose(1.336126372035137, EtOH.T_dependent_property(275))


    # Test uneven temperature spaces
    EtOH = TDependentProperty(CASRN='67-56-1')
    Ts = [195, 400, 300, 400, 450]
    props = [1.2, 1.3, 1.7, 1.9, 2.5]

    # Test naming and retrieving
    with pytest.raises(Exception):
        EtOH.set_tabular_data(Ts=Ts, properties=props)


def test_assert_component_balance():
    f1 = Stream(['water', 'ethanol', 'pentane'], zs=[.5, .4, .1], T=300, P=1E6, n=50)
    f2 = Stream(['water', 'methanol'], zs=[.5, .5], T=300, P=9E5, n=25)
    f3 = Stream(IDs=['109-66-0', '64-17-5', '67-56-1', '7732-18-5'], ns=[5.0, 20.0, 12.5, 37.5], T=300, P=850000)
    
    assert_component_balance([f1, f2], f3)

    f3 = Stream(IDs=['109-66-0', '64-17-5', '67-56-1', '7732-18-5'], ns=[6.0, 20.0, 12.5, 37.5], T=300, P=850000)
    with pytest.raises(Exception):
        assert_component_balance([f1, f2], f3)


#@pytest.xfail
#def test_d2xs_to_dxdn_partials():  
#    # Test point from NRTL 7 component case
#    d2xs = [[-14164.982925400864, -1650.9734661913312, -3993.3175265949762, -62.6722341322099, -123.58209476593841, -100.73815079566052, 124.51146594599442], 
#            [-1650.9734661913312, -3110.806662503744, -2454.5259032930726, -3197.1128126224253, -3248.801743741809, -3201.823885463643, -3107.7104581189647], 
#            [-3993.3175265949762, -2454.5259032930726, -2807.019056119142, -2779.7897798711624, -2647.1996028443173, -2681.4895374221956, -2608.413525790121], 
#            [-62.67223413220967, -3197.1128126224253, -2779.789779871163, -3444.429626351257, -3482.230779295255, -3493.566631342506, -3511.9530683201706], 
#            [-123.58209476593834, -3248.801743741809, -2647.1996028443173, -3482.230779295255, -3445.5549400048712, -3491.151162774257, -3533.2346085085373],
#            [-100.73815079566063, -3201.823885463643, -2681.4895374221956, -3493.566631342506, -3491.151162774257, -3430.5646123588135, -3572.42095177791], 
#            [124.5114659459945, -3107.7104581189656, -2608.413525790121, -3511.9530683201706, -3533.2346085085373, -3572.42095177791, -3762.5337853652773]]
#    xs = [0.14285714285714288, 0.14285714285714288, 0.14285714285714288, 0.14285714285714288, 0.14285714285714288, 0.14285714285714288, 0.14285714285714288]
#    
#    dxdn_partials_expect = [[-11311.87507798158, 1202.1343812279529, -1140.2096791756921, 2790.4356132870744, 2729.5257526533455, 2752.3696966236234, 2977.6193133652787], 
#                            [1202.1343812279529, -257.69881508445997, 398.5819441262115, -344.0049652031412, -395.6938963225248, -348.7160380443588, -254.60261069968055],
#                            [-1140.2096791756921, 398.5819441262115, 46.08879130014202, 73.31806754812169, 205.90824457496683, 171.61830999708855, 244.6943216291629], 
#                            [2790.4356132870744, -344.0049652031412, 73.31806754812123, -591.3217789319729, -629.1229318759711, -640.4587839232217, -658.8452209008865],
#                            [2729.525752653346, -395.6938963225248, 205.90824457496683, -629.1229318759711, -592.4470925855871, -638.043315354973, -680.1267610892533], 
#                            [2752.3696966236234, -348.7160380443588, 171.61830999708855, -640.4587839232217, -638.043315354973, -577.4567649395294, -719.3131043586259], 
#                            [2977.6193133652787, -254.60261069968146, 244.6943216291629, -658.8452209008865, -680.1267610892533, -719.3131043586259, -909.4259379459932]]
#    
#    assert_allclose(d2xs_to_dxdn_partials(d2xs, xs), dxdn_partials_expect, rtol=1e-12)
#    
#    # Test point from PRMIX 4 component case
#    d2xs = [[-6.952875483186023, -14.979399633475984, -17.522090915754813, -10.65365339636402], 
#            [-14.979402266584334, -32.271902511452495, -37.749926454582365, -22.952429693998027],
#            [-17.522094231005436, -37.74992695476248, -44.157823503862645, -26.84851092876711],
#            [-10.653655611967368, -22.952429843075773, -26.848510659085093, -16.32423342050014]]
#    xs = [0.004326109046411728, 0.004737414608367706, 0.04809035601246753, 0.9428461203327531]
#    
#    dxdn_partials_expect = [[4.035568664220445, 8.694305121638651, 10.170128177986037, 6.183565242595064], [-3.990958119177865, -8.59819775633786, -10.057707360841516, -6.1152110550389445], [-6.5336500835989675, -14.076222199647844, -16.465604410121795, -10.011292289808026], [0.3347885354391007, 0.7212749120388615, 0.8437084346557562, 0.5129852184589438]]
#    assert_allclose(d2xs_to_dxdn_partials(d2xs, xs), dxdn_partials_expect, rtol=1e-12)

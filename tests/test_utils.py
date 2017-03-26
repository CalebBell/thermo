# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from thermo.utils import *

def test_to_num():
    assert to_num(['1', '1.1', '1E5', '0xB4', '']) == [1.0, 1.1, 100000.0, '0xB4', None]



def test_none_and_length_check():
    assert True == none_and_length_check([[1,2,3]])
    
    assert True == none_and_length_check(([1, 1], [1, 1], [1, 30], [10,0]), length=2)
    assert True == none_and_length_check(([1, 1], [1, 1], [1, 30], [10,0]))

    assert False == none_and_length_check(([1, 1], [None, 1], [1, 30], [10,0]), length=2)
    assert False == none_and_length_check(([None, 1], [1, 1], [1, 30], [10,0]))
    assert False == none_and_length_check(([1, 1], [None, 1], [1, None], [10,0]), length=2)
    assert False == none_and_length_check(([None, 1], [1, 1], [1, 30], [None,0]))

    assert False == none_and_length_check(([1, 1, 1], [1, 1], [1, 30], [10,0]), length=2)
    assert False == none_and_length_check(([1, 1], [1, 1, 1], [1, 30], [10,0]))
    assert False == none_and_length_check(([1, 1, 1], [1, 1], [1, 30, 1], [10,0]), length=2)
    assert False == none_and_length_check(([1, 1], [1, 1, 1], [1, 30], [10,0, 1]))
    assert False == none_and_length_check(([1, 1, 1], [1, 1, 1], [1, 30, 1], [10,0]), length=3)
    assert False == none_and_length_check(([1, 1], [1, 1, 1], [1, 30, 1], [10,0, 1]))

    assert True == none_and_length_check(([1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]), length=8)
    assert True == none_and_length_check(([1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]))

    assert False == none_and_length_check(([1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]), length=9)
    assert False == none_and_length_check(([1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]), length=7)
    assert False == none_and_length_check(([1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, None]), length=8)
    assert False == none_and_length_check(([1, 1, None, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]), length=8)

    # Test list input instead of tuples
    assert True == none_and_length_check([[1, 1], [1, 1], [1, 30], [10,0]], length=2)
    assert True == none_and_length_check([[1, 1], [1, 1], [1, 30], [10,0]])

    assert True == none_and_length_check([[1, 1], [1, 1], [1, 30], [10,0]], length=2)
    assert True == none_and_length_check([[1, 1], [1, 1], [1, 30], [10,0]])

    # Test with numpy arrays
    assert True == none_and_length_check((np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1, 1, 1, 1]), [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]), length=8)
    assert False == none_and_length_check((np.array([1, 1, 1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1, 1, 1, 1]), [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]), length=8)
    assert False == none_and_length_check((np.array([1, 1, 1, 1, 1, 1, 1, 7]), np.array([1, 1, 1, 1, 1, 1, 1, 1]), [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]), length=7)
    assert True == none_and_length_check(np.array([[1, 1], [1, 1], [1, 30], [10,0]]))

    assert True == none_and_length_check(np.array([[1, 1], [1, 1], [1, 30], [10,0]]))
    assert True == none_and_length_check(np.array([[1, 1], [1, 1], [1, 30], [10,0]]), length=2)
    assert False == none_and_length_check(np.array([[1, 1], [1, 1], [1, 30], [10,0]]), length=3)
    assert False == none_and_length_check(np.array([[1, 1], [1, 1, 10], [1, 30], [10,0]]), length=3)



def test_CAS2int():
    assert CAS2int('7704-34-9') == 7704349

    with pytest.raises(Exception):
        CAS2int(7704349)

def test_int2CAS():
    assert int2CAS(7704349) == '7704-34-9'

    with pytest.raises(Exception):
        CAS2int(7704349.0)


def test_zs_to_ws():
    ws_calc = zs_to_ws([0.5, 0.5], [10, 20])
    ws = [0.3333333333333333, 0.6666666666666666]
    assert_allclose(ws_calc, ws)


def test_ws_to_zs():
    zs_calc = ws_to_zs([0.3333333333333333, 0.6666666666666666], [10, 20])
    zs = [0.5, 0.5]
    assert_allclose(zs_calc, zs)


def test_zs_to_Vfs():
    Vfs_calc = zs_to_Vfs([0.637, 0.363], [8.0234e-05, 9.543e-05])
    Vfs = [0.5960229712956298, 0.4039770287043703]
    assert_allclose(Vfs_calc, Vfs)


def test_Vfs_to_zs():
    zs_calc = Vfs_to_zs([0.596, 0.404], [8.0234e-05, 9.543e-05])
    zs = [0.6369779395901142, 0.3630220604098858]
    assert_allclose(zs_calc, zs)


def test_B_To_Z():
    Z_calc = B_To_Z(-0.0015, 300, 1E5)
    assert_allclose(Z_calc, 0.9398638020957176)


def test_B_from_Z():
    B_calc = B_from_Z(0.94, 300, 1E5)
    assert_allclose(B_calc, -0.0014966027640000014)


def test_Z():
    Z_calc = Z(600, P=1E6, V=0.00463)
    assert_allclose(Z_calc, 0.9281019876560912)


def test_Vm_to_rho():
    rho = Vm_to_rho(0.000132, 86.18)
    assert_allclose(rho, 652.8787878787879)


def test_rho_to_Vm():
    Vm = rho_to_Vm(652.9, 86.18)
    assert_allclose(Vm, 0.00013199571144126206)
    assert_allclose(652.9, Vm_to_rho(rho_to_Vm(652.9, 86.18), 86.18))


def test_isentropic_exponent():
    k = isentropic_exponent(33.6, 25.27)
    assert_allclose(k, 1.329639889196676)

    
def test_Parachor():
    P = Parachor(100.15888, 800.8088185536124, 4.97865317223119, 0.02672166960656005)
    assert_allclose(P, 5.088443542210164e-05)
    

def test_phase_select_property():
    assert 150 == phase_select_property(phase='s', s=150, l=10)
    assert None == phase_select_property(phase='s', l=1560.14)
    assert 3312 == phase_select_property(phase='g', l=1560.14, g=3312.)
    assert 1560.14 == phase_select_property(phase='l', l=1560.14, g=3312.)
    assert None == phase_select_property(phase='two-phase', l=1560.14, g=12421.0)
    assert None == phase_select_property(phase=None, l=1560.14, g=12421.0)
    with pytest.raises(Exception):
        phase_select_property(phase='notalphase', l=1560.14, g=12421.0)



def test_mixing_simple():
    prop = mixing_simple([0.1, 0.9], [0.01, 0.02])
    assert_allclose(prop, 0.019)

    assert None == mixing_simple([0.1], [0.01, 0.02])

def test_mixing_logarithmic():
    prop = mixing_logarithmic([0.1, 0.9], [0.01, 0.02])
    assert_allclose(prop, 0.01866065983073615)

    assert None == mixing_logarithmic([0.1], [0.01, 0.02])

def test_normalize():
    fractions_calc = normalize([3, 2, 1])
    fractions = [0.5, 0.3333333333333333, 0.16666666666666666]
    assert_allclose(fractions, fractions_calc)


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


def test_Z_from_virial_density_form():
    Z_calc = Z_from_virial_density_form(300, 122057.233762653, 1E-4, 1E-5, 1E-6, 1E-7)
    assert_allclose(Z_calc, 1.2843496002100001)
    
    Z_calc = Z_from_virial_density_form(300, 102031.881198762, 1e-4, 1e-5, 1e-6)
    assert_allclose(Z_calc, 1.0736324409999995)

    Z_calc = Z_from_virial_density_form(300, 96775.8831504971, 1e-4, 1e-5)
    assert_allclose(Z_calc, 1.0183261000000003)
    
    Z_calc = Z_from_virial_density_form(300, 95396.3561037084, 1e-4)
    assert_allclose(Z_calc,  1.0038100000000001)
    
    assert_allclose(1, Z_from_virial_density_form(300, 95396.3561037084))
    
    '''B-only solution, derived as follows:
    
    >>> B, C, D, E = symbols('B, C, D, E')
    >>> P, V, Z, R, T = symbols('P, V, Z, R, T', positive=True, real=True, nonzero=True)
    >>> rho = 1/V
    >>> to_slv = Eq(P*V/R/T, 1 + B*rho)
    >>> slns = solve(to_slv, V)
    >>> simplify(slns[1]*P/R/T)
    1/2 + sqrt(4*B*P + R*T)/(2*sqrt(R)*sqrt(T))
    
    To check this, simply disable the if statement and allow the numerical 
    algorithm to run.
    '''


def test_Z_from_virial_pressure_form():
    Z_calc = Z_from_virial_pressure_form(102919.99946855308, 4.032286555169439e-09, 1.6197059494442215e-13, 6.483855042486911e-19)
    assert_allclose(Z_calc, 1.00283753944)
    
    Z_calc = Z_from_virial_pressure_form(102847.17619188508, 4.032286555169439e-09, 1.6197059494442215e-13)
    assert_allclose(Z_calc, 1.00212796)

    Z_calc = Z_from_virial_pressure_form(102671.27455742132, 4.032286555169439e-09)
    assert_allclose(Z_calc, 1.000414)
    
    Z_calc = Z_calc = Z_from_virial_pressure_form(102671.27455742132)
    assert_allclose(Z_calc, 1)


def test_isobaric_expansion():
    beta = isobaric_expansion(0.000130229900873546, 1.58875261849113e-7)
    assert_allclose(beta, 0.0012199599384121608)
    
def test_isothermal_compressibility():
    kappa = isothermal_compressibility(0.000130229900873546, -2.72902118209903e-13)
    assert_allclose(2.09554116511916e-9, kappa)
    

def test_phase_identification_parameter():
    PIP = phase_identification_parameter(0.000130229900874, 582169.397484, -3.66431747236e+12, 4.48067893805e+17, -20518995218.2)
    assert_allclose(PIP, 11.33428990564796)
    
    assert 'l' == phase_identification_parameter_phase(-20518995218.2, 0.000130229900874, 582169.397484, -3.66431747236e+12, 4.48067893805e+17)
    # Artificially give a value to make it be solid
    assert 's' == phase_identification_parameter_phase(+20518995218.2)

def test_Cp_minus_Cv():
    d = Cp_minus_Cv(299, 582232.475794113, -3665180614672.253)
    assert_allclose(d, 27.654681381642394)


def test_speed_of_sound():
    # Matches exactly
    w = speed_of_sound(V=0.00229754, dP_dV=-3.5459e+08, Cp=153.235, Cv=132.435, MW=67.152)
    assert_allclose(w, 179.5868138460819)
    
    # No MW version
    w = speed_of_sound(V=0.00229754, dP_dV=-3.5459e+08, Cp=153.235, Cv=132.435)
    assert_allclose(w, 46.537593457316525)
    
    
def test_mu_JT():
    # Matches exactly
    mu_JT = Joule_Thomson(T=390, V=0.00229754, Cp=153.235, dV_dT=1.226396e-05)
    assert_allclose(mu_JT, 1.621956080529905e-05)
    
    mu_JT = Joule_Thomson(T=390, V=0.00229754, Cp=153.235, beta=0.005337865717245402)
    assert_allclose(mu_JT, 1.621956080529905e-05)

    with pytest.raises(Exception):
        Joule_Thomson(T=390, V=0.00229754, Cp=153.235)
        
        

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


def test_horner():
    assert horner([1,2,3], 3) == 18
    
@pytest.mark.sympy
def test_polylog2():
    x = polylog2(0.5)
    assert_allclose(x, 0.5822405264516294)
    
    from sympy import polylog
    xs = np.linspace(0,0.99999, 50)
    ys_act = [float(polylog(2, x)) for x in xs]
    ys = [polylog2(x) for x in xs]
    assert_allclose(ys, ys_act, rtol=1E-7, atol=1E-10)

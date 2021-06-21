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
from fluids.numerics import assert_close
import pytest
from math import log
from thermo.utils import TDependentProperty

def test_local_constant_method():
    # Test user defined method
    
    # Within valid T range
    obj = TDependentProperty(extrapolation='linear')
    constant = 100.
    T1 = T = 300.
    T2 = 310.
    dT = T2 - T1
    obj.add_method(constant)
    assert_close(obj.T_dependent_property(T), constant)
    for order in (1, 2, 3):
        assert_close(obj.T_dependent_property_derivative(T, order), 0.)
    assert_close(obj.T_dependent_property_integral(T1, T2), constant * dT)
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), constant * log(T2 / T1))
    
    # Extrapolate
    Tmin = 350.
    Tmax = 400.
    obj.add_method(constant, Tmin, Tmax)
    obj.extrapolation = 'constant'
    assert_close(obj.T_dependent_property(T), constant)
    for order in (1, 2, 3):
        assert_close(obj.T_dependent_property_derivative(T, order), 0., atol=1e-6)
    assert_close(obj.T_dependent_property_integral(T1, T2), constant * dT)
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), constant * log(T2 / T1))
    
    # Do not allow extrapolation
    obj.extrapolation = None
    obj.add_method(constant, Tmin, Tmax)
    assert obj.T_dependent_property(T) is None
    # assert obj.T_dependent_property_integral(T, T+10.) is None
    # assert obj.T_dependent_property_integral_over_T(T, T+10) is None
    
def test_local_method():
    # Test user defined method
    
    # Within valid T range
    obj = TDependentProperty(extrapolation='linear')
    T = 300.
    Tmin = 200.
    Tmax = 400.
    T1 = 300.
    T2 = 310.
    dT = T2 - T1
    f = lambda T: T*T / 1e6
    f_der = lambda T: T / 1e6
    f_der2 = lambda T: 1. / 1e6
    f_der3 = lambda T: 0. / 1e6
    f_int = lambda T1, T2: (T2*T2*T2 - T1*T1*T1) / 3. / 1e6
    f_int_over_T = lambda T1, T2: (T2*T2 - T1*T1) / 2. / 1e6
    obj.add_method(f, Tmin, Tmax, f_der, f_der2, f_der3, f_int, f_int_over_T)
    assert_close(obj.T_dependent_property(T), f(T))
    assert_close(obj.T_dependent_property_derivative(T, 1), f_der(T))
    assert_close(obj.T_dependent_property_derivative(T, 2), f_der2(T))
    assert_close(obj.T_dependent_property_derivative(T, 3), f_der3(T))
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, T2))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, T2))
    
    # Extrapolate
    Tmin = 300. + 1e-3
    obj.add_method(f, Tmin, Tmax)
    assert_close(obj.T_dependent_property(T), f(T))
    for order in (1, 2, 3):
        assert obj.T_dependent_property_derivative(T, order) is not None
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, T2))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, T2))
    
    # Do not allow extrapolation
    obj.extrapolation = None
    obj.add_method(f, Tmin, Tmax)
    assert obj.T_dependent_property(T) is None
    
def test_many_local_methods():
    obj = TDependentProperty(extrapolation='linear')
    M1_value = 2.0
    M2_value = 3.0
    obj.add_method(M1_value, name='M1')
    obj.add_method(M2_value, name='M2')
    obj.method = 'M1'
    assert_close(M1_value, obj.T_dependent_property(300.))
    obj.method = 'M2'
    assert_close(M2_value, obj.T_dependent_property(300.))
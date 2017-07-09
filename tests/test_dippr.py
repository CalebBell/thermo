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
from scipy.misc import derivative
from scipy.integrate import quad
import pytest

from thermo.dippr import *

def test_Eqs():
    a = EQ100(300, 276370., -2090.1, 8.125, -0.014116, 0.0000093701)
    assert_allclose(a, 75355.81)

    a = EQ101(300, 73.649, -7258.2, -7.3037, 4.1653E-6, 2)
    assert_allclose(a, 3537.44834545549)

    a = EQ102(300, 1.7096E-8, 1.1146, 0, 0)
    assert_allclose(a, 9.860384711890639e-06)

    a = EQ104(300, 0.02222, -26.38, -16750000, -3.894E19, 3.133E21)
    assert_allclose(a, -1.1204179007265151)

    a = EQ105(300., 0.70824, 0.26411, 507.6, 0.27537)
    assert_allclose(a, 7.593170096339236)

    a = EQ106(300, 647.096, 0.17766, 2.567, -3.3377, 1.9699)
    assert_allclose(a, 0.07231499373541)

    a = EQ107(300., 33363., 26790., 2610.5, 8896., 1169.)
    assert_allclose(a, 33585.90452768923)

    a = EQ114(20, 33.19, 66.653, 6765.9, -123.63, 478.27)
    assert_allclose(a, 19423.948911676463)

    a = EQ116(300., 647.096, 17.863, 58.606, -95.396, 213.89, -141.26)
    assert_allclose(a, 55.17615446406527)

    a = EQ127(20., 3.3258E4, 3.6199E4, 1.2057E3, 1.5373E7, 3.2122E3, -1.5318E7, 3.2122E3)
    assert_allclose(a, 33258.0)

    # Random coefficients
    a = EQ115(300, 0.01, 0.002, 0.0003, 0.00004)
    assert_allclose(a, 37.02960772416336)
    
    
def test_EQ127_more():
    # T derivative
    coeffs = (3.3258E4, 3.6199E4, 1.2057E3, 1.5373E7, 3.2122E3, -1.5318E7, 3.2122E3)
    diff_1T = derivative(EQ127, 50,  dx=1E-3, order=21, args=coeffs)
    diff_1T_analytical = EQ127(50., *coeffs, order=1)
    assert_allclose(diff_1T, diff_1T_analytical, rtol=1E-3)
    assert_allclose(diff_1T, 0.000313581049006, rtol=1E-4)
    
    # Integral
    int_50 = EQ127(50., *coeffs, order=-1) 
    int_20 = EQ127(20., *coeffs, order=-1)
    numerical_1T = quad(EQ127, 20, 50, args=coeffs)[0]
    assert_allclose(int_50 - int_20, numerical_1T)
    assert_allclose(numerical_1T, 997740.00147014)
    
    # Integral over T
    T_int_50 = EQ127(50., *coeffs, order=-1j)
    T_int_20 = EQ127(20., *coeffs, order=-1j)
    
    to_int = lambda T :EQ127(T, *coeffs)/T
    numerical_1_over_T = quad(to_int, 20, 50)[0]
    assert_allclose(T_int_50 - T_int_20, numerical_1_over_T)
    assert_allclose(T_int_50 - T_int_20, 30473.9971912935)

    with pytest.raises(Exception):
        EQ127(20., *coeffs, order=1E100)

    
def test_EQ116_more():
    # T derivative
    coeffs = (647.096, 17.863, 58.606, -95.396, 213.89, -141.26)
    diff_1T = derivative(EQ116, 50,  dx=1E-3, order=21, args=coeffs)
    diff_1T_analytical = EQ116(50., *coeffs, order=1)
    assert_allclose(diff_1T, diff_1T_analytical, rtol=1E-3)
    assert_allclose(diff_1T_analytical, 0.020379262711650914)
    
    # Integral
    int_50 = EQ116(50., *coeffs, order=-1) 
    int_20 = EQ116(20., *coeffs, order=-1)
    numerical_1T = quad(EQ116, 20, 50, args=coeffs)[0]
    assert_allclose(int_50 - int_20, numerical_1T)
    assert_allclose(int_50 - int_20, 1636.962423782701)
    
    # Integral over T
    T_int_50 = EQ116(50., *coeffs, order=-1j)
    T_int_20 = EQ116(20., *coeffs, order=-1j)
    
    to_int = lambda T :EQ116(T, *coeffs)/T
    numerical_1_over_T = quad(to_int, 20, 50)[0]
    assert_allclose(T_int_50 - T_int_20, numerical_1_over_T)
    assert_allclose(T_int_50 - T_int_20, 49.95109104018752)
    
    with pytest.raises(Exception):
        EQ116(20., *coeffs, order=1E100)
    
def test_EQ107_more():
    # T derivative
    coeffs = (33363., 26790., 2610.5, 8896., 1169.)
    diff_1T = derivative(EQ107, 250,  dx=1E-3, order=21, args=coeffs)
    diff_1T_analytical = EQ107(250., *coeffs, order=1)
    assert_allclose(diff_1T, diff_1T_analytical, rtol=1E-3)
    assert_allclose(diff_1T_analytical, 1.985822265543943)
    
    # Integral
    int_50 = EQ107(50., *coeffs, order=-1) 
    int_20 = EQ107(20., *coeffs, order=-1)
    numerical_1T = quad(EQ107, 20, 50, args=coeffs)[0]
    assert_allclose(int_50 - int_20, numerical_1T)
    assert_allclose(numerical_1T, 1000890.0)
    
    # Integral over T
    T_int_50 = EQ107(50., *coeffs, order=-1j)
    T_int_20 = EQ107(20., *coeffs, order=-1j)
    
    to_int = lambda T :EQ107(T, *coeffs)/T
    numerical_1_over_T = quad(to_int, 20, 50)[0]
    assert_allclose(T_int_50 - T_int_20, numerical_1_over_T)
    assert_allclose(T_int_50 - T_int_20, 30570.20768751744)


    with pytest.raises(Exception):
        EQ107(20., *coeffs, order=1E100)

def test_EQ114_more():
    # T derivative
    coeffs = (33.19, 66.653, 6765.9, -123.63, 478.27)
    diff_1T = derivative(EQ114, 20,  dx=1E-3, order=21, args=coeffs)
    diff_1T_analytical = EQ114(20., *coeffs, order=1)
    assert_allclose(diff_1T, diff_1T_analytical, rtol=1E-3)
    assert_allclose(diff_1T, 1135.38618941)
    
    # Integral
    int_50 = EQ114(30., *coeffs, order=-1) 
    int_20 = EQ114(20., *coeffs, order=-1)
    numerical_1T = quad(EQ114, 20, 30, args=coeffs)[0]
    assert_allclose(int_50 - int_20, numerical_1T)
    assert_allclose(int_50 - int_20, 295697.48978888744)
    
#     Integral over T
    T_int_50 = EQ114(30., *coeffs, order=-1j)
    T_int_20 = EQ114(20., *coeffs, order=-1j)
    
    to_int = lambda T :EQ114(T, *coeffs)/T
    numerical_1_over_T = quad(to_int, 20, 30)[0]
    assert_allclose(T_int_50 - T_int_20, numerical_1_over_T)
    assert_allclose(T_int_50 - T_int_20, 11612.331762721366)
    
    with pytest.raises(Exception):
        EQ114(20., *coeffs, order=1E100)


def test_EQ102_more():
    # T derivative
    coeffs = (1.7096E-8, 1.1146, 1.1, 2.1)
    diff_1T = derivative(EQ102, 250,  dx=1E-3, order=21, args=coeffs)
    diff_1T_analytical = EQ102(250., *coeffs, order=1)
    assert_allclose(diff_1T, diff_1T_analytical, rtol=1E-3)
    assert_allclose(diff_1T, 3.5861274167602139e-08)
    
    # Integral
    int_250 = EQ102(250., *coeffs, order=-1) 
    int_220 = EQ102(220., *coeffs, order=-1)
    numerical_1T = quad(EQ102, 220, 250, args=coeffs)[0]
    assert_allclose(int_250 - int_220, numerical_1T)
    assert_allclose(int_250 - int_220, 0.00022428562125110119)
    
#     Integral over T
    T_int_250 = EQ102(250., *coeffs, order=-1j)
    T_int_220 = EQ102(220., *coeffs, order=-1j)
    
    to_int = lambda T :EQ102(T, *coeffs)/T
    numerical_1_over_T = quad(to_int, 220, 250)[0]
    assert_allclose(T_int_250 - T_int_220, numerical_1_over_T)
    assert_allclose(T_int_250 - T_int_220, 9.5425212178091671e-07)
#    
    with pytest.raises(Exception):
        EQ102(20., *coeffs, order=1E100)


def test_EQ100_more():
    # T derivative
    coeffs = (276370., -2090.1, 8.125, -0.014116, 0.0000093701)
    diff_1T = derivative(EQ100, 250,  dx=1E-3, order=21, args=coeffs)
    diff_1T_analytical = EQ100(250., *coeffs, order=1)
    assert_allclose(diff_1T, diff_1T_analytical, rtol=1E-3)
    assert_allclose(diff_1T, -88.7187500531)
    
    # Integral
    int_250 = EQ100(250., *coeffs, order=-1) 
    int_220 = EQ100(220., *coeffs, order=-1)
    numerical_1T = quad(EQ100, 220, 250, args=coeffs)[0]
    assert_allclose(int_250 - int_220, numerical_1T)
    assert_allclose(int_250 - int_220, 2381304.7021859996)
    
#     Integral over T
    T_int_250 = EQ100(250., *coeffs, order=-1j)
    T_int_220 = EQ100(220., *coeffs, order=-1j)
    
    to_int = lambda T :EQ100(T, *coeffs)/T
    numerical_1_over_T = quad(to_int, 220, 250)[0]
    assert_allclose(T_int_250 - T_int_220, numerical_1_over_T)
    assert_allclose(T_int_250 - T_int_220, 10152.09780143667)

    with pytest.raises(Exception):
        EQ100(20., *coeffs, order=1E100)


def test_EQ104_more():
    # T derivative
    coeffs = (0.02222, -26.38, -16750000, -3.894E19, 3.133E21)
    diff_1T = derivative(EQ104, 250,  dx=1E-3, order=21, args=coeffs)
    diff_1T_analytical = EQ104(250., *coeffs, order=1)
    assert_allclose(diff_1T, diff_1T_analytical, rtol=1E-3)
    assert_allclose(diff_1T, 0.0653824814073)
    
    # Integral
    int_250 = EQ104(250., *coeffs, order=-1) 
    int_220 = EQ104(220., *coeffs, order=-1)
    numerical_1T = quad(EQ104, 220, 250, args=coeffs)[0]
    assert_allclose(int_250 - int_220, numerical_1T)
    assert_allclose(int_250 - int_220, -127.91851427119406)
    
#     Integral over T
    T_int_250 = EQ104(250., *coeffs, order=-1j)
    T_int_220 = EQ104(220., *coeffs, order=-1j)
    
    to_int = lambda T :EQ104(T, *coeffs)/T
    numerical_1_over_T = quad(to_int, 220, 250)[0]
    assert_allclose(T_int_250 - T_int_220, numerical_1_over_T)
    assert_allclose(T_int_250 - T_int_220, -0.5494851210308727)

    with pytest.raises(Exception):
        EQ104(20., *coeffs, order=1E100)



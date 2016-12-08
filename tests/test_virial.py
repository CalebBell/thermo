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
from thermo.virial import *
from scipy.constants import R as _R
from scipy.integrate import quad


def test_BVirial_Pitzer_Curl():
    # doctest
    B = BVirial_Pitzer_Curl(510., 425.2, 38E5, 0.193)
    assert_allclose(B, -0.0002084535541385102)

    with pytest.raises(Exception):
        BVirial_Pitzer_Curl(510., 425.2, 38E5, 0.193, order=-3)

@pytest.mark.sympy
def test_BVirial_Pitzer_Curl_calculus():
    from sympy import symbols, Rational, diff, lambdify, integrate
    # Derivatives check
    # Uses SymPy
    T, Tc, Pc, omega, R = symbols('T, Tc, Pc, omega, R')
    Tr = T/Tc
    B0 = Rational(1445,10000) - Rational(33,100)/Tr - Rational(1385,10000)/Tr**2 - Rational(121,10000)/Tr**3
    B1 = Rational(73,1000) + Rational(46,100)/Tr - Rational(1,2)/Tr**2 - Rational(97,1000)/Tr**3 - Rational(73,10000)/Tr**8
        
    # Note: scipy.misc.derivative was attempted, but found to given too
    # incorrect of derivatives for higher orders, so there is no reasons to 
    # implement it. Plus, no uses have yet been found for the higher 
    # derivatives/integrals. For integrals, there is no way to get the 
    # indefinite integral.
    
    # Test 160K points in vector form for order 1, 2, and 3
    # Use lambdify for fast evaluation
    _Ts = np.linspace(5,500,20)
    _Tcs = np.linspace(501,900,20)
    _Pcs = np.linspace(2E5, 1E7,20)
    _omegas = np.linspace(-1, 10,20)
    _Ts, _Tcs, _Pcs, _omegas = np.meshgrid(_Ts, _Tcs, _Pcs, _omegas)
    _Ts, _Tcs, _Pcs, _omegas = _Ts.ravel(), _Tcs.ravel(), _Pcs.ravel(), _omegas.ravel()
    for order in range(1,4):
        B0c = diff(B0, T, order)
        B1c = diff(B1, T, order)
        Br = B0c + omega*B1c
        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        Bcalc2 = BVirial_Pitzer_Curl(_Ts, _Tcs, _Pcs, _omegas, order)
        assert_allclose(Bcalcs, Bcalc2)
        

    # Check integrals using SymPy:
    for order in range(-2, 0):
        B0c = B0
        B1c = B1
        for i in range(abs(order)):
            B0c = integrate(B0c, T)
            B1c = integrate(B1c, T)
        Br = B0c + omega*B1c
        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        Bcalc2 = [BVirial_Pitzer_Curl(T2, Tc2, Pc2, omega2, order) for T2, Tc2, Pc2, omega2 in zip(_Ts, _Tcs, _Pcs, _omegas)]
        assert_allclose(Bcalcs, Bcalc2)


    # Check integrals using numerical methods - quad:
    for order in range(-2, 0):
        for trial in range(10):
            T1, T2 = np.random.random_integers(5, 500, 2)*np.random.rand(2)
            _Tc = np.random.choice(_Tcs)
            _Pc = np.random.choice(_Pcs)
            _omega = np.random.choice(_omegas)
            
            dBint = BVirial_Pitzer_Curl(T2, _Tc, _Pc, _omega, order) - BVirial_Pitzer_Curl(T1, _Tc, _Pc, _omega, order)
            dBquad = quad(BVirial_Pitzer_Curl, T1, T2, args=(_Tc, _Pc, _omega, order+1))[0]
            assert_allclose(dBint, dBquad, rtol=1E-5)


@pytest.mark.sympy
def test_BVirial_Abbott():
    from sympy import symbols, Rational, diff, lambdify, integrate

    B = BVirial_Abbott(510., 425.2, 38E5, 0.193)
    assert_allclose(B, -0.00020570178037383633)
    
    with pytest.raises(Exception):
        BVirial_Abbott(510., 425.2, 38E5, 0.193, order=-3)

    T, Tc, Pc, omega, R = symbols('T, Tc, Pc, omega, R')
    Tr = T/Tc
    B0 = 0.083 - 0.422/Tr**1.6
    B1 = 0.139 - 0.172/Tr**4.2

    # Test 160K points in vector form for order 1, 2, and 3
    # Use lambdify for fast evaluation
    _Ts = np.linspace(5,500,20)
    _Tcs = np.linspace(501,900,20)
    _Pcs = np.linspace(2E5, 1E7,20)
    _omegas = np.linspace(-1, 10,20)
    _Ts, _Tcs, _Pcs, _omegas = np.meshgrid(_Ts, _Tcs, _Pcs, _omegas)
    _Ts, _Tcs, _Pcs, _omegas = _Ts.ravel(), _Tcs.ravel(), _Pcs.ravel(), _omegas.ravel()


    for order in range(1,4):
        B0c = diff(B0, T, order)
        B1c = diff(B1, T, order)
        Br = B0c + omega*B1c
        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        Bcalc2 = BVirial_Abbott(_Ts, _Tcs, _Pcs, _omegas, order)
        assert_allclose(Bcalcs, Bcalc2)


    # Check integrals using SymPy:
    for order in range(-2, 0):
        B0c = B0
        B1c = B1
        for i in range(abs(order)):
            B0c = integrate(B0c, T)
            B1c = integrate(B1c, T)
        Br = B0c + omega*B1c
        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        
        
        Bcalc2 = [BVirial_Abbott(T2, Tc2, Pc2, omega2, order) for T2, Tc2, Pc2, omega2 in zip(_Ts, _Tcs, _Pcs, _omegas)]
        assert_allclose(Bcalcs, Bcalc2)


    # Check integrals using numerical methods - quad:
    for order in range(-2, 0):
        for trial in range(10):
            T1, T2 = np.random.random_integers(5, 500, 2)*np.random.rand(2)
            _Tc = np.random.choice(_Tcs)
            _Pc = np.random.choice(_Pcs)
            _omega = np.random.choice(_omegas)
            
            dBint = BVirial_Abbott(T2, _Tc, _Pc, _omega, order) - BVirial_Abbott(T1, _Tc, _Pc, _omega, order)
            dBquad = quad(BVirial_Abbott, T1, T2, args=(_Tc, _Pc, _omega, order+1))[0]
            assert_allclose(dBint, dBquad, rtol=1E-5)

@pytest.mark.sympy
def test_BVirial_Tsonopoulos():
    from sympy import symbols, Rational, diff, lambdify, integrate

    B = BVirial_Tsonopoulos(510., 425.2, 38E5, 0.193)
    assert_allclose(B, -0.00020935288308483694)

    with pytest.raises(Exception):
        BVirial_Tsonopoulos(510., 425.2, 38E5, 0.193, order=-3)

    T, Tc, Pc, omega, R = symbols('T, Tc, Pc, omega, R')
    Tr = T/Tc
    B0 = Rational(1445, 10000) - Rational(33,100)/Tr - Rational(1385,10000)/Tr**2 - Rational(121,10000)/Tr**3 - Rational(607,1000000)/Tr**8
    B1 = Rational(637,10000) + Rational(331,1000)/Tr**2 - Rational(423,1000)/Tr**3 - Rational(8,1000)/Tr**8

    # Test 160K points in vector form for order 1, 2, and 3
    # Use lambdify for fast evaluation
    _Ts = np.linspace(5,500,20)
    _Tcs = np.linspace(501,900,20)
    _Pcs = np.linspace(2E5, 1E7,20)
    _omegas = np.linspace(-1, 10,20)
    _Ts, _Tcs, _Pcs, _omegas = np.meshgrid(_Ts, _Tcs, _Pcs, _omegas)
    _Ts, _Tcs, _Pcs, _omegas = _Ts.ravel(), _Tcs.ravel(), _Pcs.ravel(), _omegas.ravel()


    for order in range(1,4):
        B0c = diff(B0, T, order)
        B1c = diff(B1, T, order)
        Br = B0c + omega*B1c
        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        Bcalc2 = BVirial_Tsonopoulos(_Ts, _Tcs, _Pcs, _omegas, order)
        assert_allclose(Bcalcs, Bcalc2)

    # Check integrals using SymPy:
    for order in range(-2, 0):
        B0c = B0
        B1c = B1
        for i in range(abs(order)):
            B0c = integrate(B0c, T)
            B1c = integrate(B1c, T)
        Br = B0c + omega*B1c
        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        
        
        Bcalc2 = [BVirial_Tsonopoulos(T2, Tc2, Pc2, omega2, order) for T2, Tc2, Pc2, omega2 in zip(_Ts, _Tcs, _Pcs, _omegas)]
        assert_allclose(Bcalcs, Bcalc2)


    # Check integrals using numerical methods - quad:
    for order in range(-2, 0):
        for trial in range(10):
            T1, T2 = np.random.random_integers(5, 500, 2)*np.random.rand(2)
            _Tc = np.random.choice(_Tcs)
            _Pc = np.random.choice(_Pcs)
            _omega = np.random.choice(_omegas)
            
            dBint = BVirial_Tsonopoulos(T2, _Tc, _Pc, _omega, order) - BVirial_Tsonopoulos(T1, _Tc, _Pc, _omega, order)
            dBquad = quad(BVirial_Tsonopoulos, T1, T2, args=(_Tc, _Pc, _omega, order+1))[0]
            assert_allclose(dBint, dBquad, rtol=1E-5)


@pytest.mark.sympy
def test_BVirial_Tsonopoulos_extended():
    from sympy import symbols, Rational, diff, lambdify, integrate

    B = BVirial_Tsonopoulos_extended(510., 425.2, 38E5, 0.193, species_type='normal', dipole=0)
    assert_allclose(B, -0.00020935288308483694)

    B = BVirial_Tsonopoulos_extended(430., 405.65, 11.28E6, 0.252608, a=0, b=0, species_type='ketone', dipole=1.469)
    assert_allclose(B, -9.679715056695323e-05)

    with pytest.raises(Exception):
        BVirial_Tsonopoulos_extended(510., 425.2, 38E5, 0.193, order=-3)

    # Test all of the different types
    types = ['simple', 'normal', 'methyl alcohol', 'water', 'ketone',
    'aldehyde', 'alkyl nitrile', 'ether', 'carboxylic acid', 'ester', 'carboxylic acid',
    'ester', 'alkyl halide', 'mercaptan', 'sulfide', 'disulfide', 'alkanol']

    Bs_calc = [BVirial_Tsonopoulos_extended(430., 405.65, 11.28E6, 0.252608,
                                            a=0, b=0, species_type=i, dipole=0.1) for i in types]
    Bs = [-9.002529440027288e-05, -9.002529440027288e-05, -8.136805574379563e-05, -9.232250634010228e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.003495446399036e-05, -9.003495446399036e-05, -9.003495446399036e-05, -9.003495446399036e-05, -7.331247111785242e-05]
    assert_allclose(Bs_calc, Bs)


    # Use lambdify for fast evaluation
    _Ts = np.linspace(5,500,20)
    _Tcs = np.linspace(501,900,20)
    _Pcs = np.linspace(2E5, 1E7,20)
    _omegas = np.linspace(-1, 10,20)
    _Ts, _Tcs, _Pcs, _omegas = np.meshgrid(_Ts, _Tcs, _Pcs, _omegas)
    _Ts, _Tcs, _Pcs, _omegas = _Ts.ravel(), _Tcs.ravel(), _Pcs.ravel(), _omegas.ravel()

    T, Tc, Pc, omega, R = symbols('T, Tc, Pc, omega, R')
    Tr = T/Tc
    B0 = Rational(1445, 10000) - Rational(33,100)/Tr - Rational(1385,10000)/Tr**2 - Rational(121,10000)/Tr**3 - Rational(607,1000000)/Tr**8
    B1 = Rational(637,10000) + Rational(331,1000)/Tr**2 - Rational(423,1000)/Tr**3 - Rational(8,1000)/Tr**8
    B2 = 1/Tr**6
    B3 = -1/Tr**8

    a = 0.1
    b = 0.2

    for order in range(1,4):
        B0c = diff(B0, T, order)
        B1c = diff(B1, T, order)
        B2c = diff(B2, T, order)
        B3c = diff(B3, T, order)
        
        Br = B0c + omega*B1c + a*B2c + b*B3c
        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        Bcalc2 = BVirial_Tsonopoulos_extended(_Ts, _Tcs, _Pcs, _omegas, order=order, a=a, b=b)
        assert_allclose(Bcalcs, Bcalc2)


    # Check integrals using SymPy:
    for order in range(-2, 0):
        B0c = B0
        B1c = B1
        B2c = B2
        B3c = B3
        for i in range(abs(order)):
            B0c = integrate(B0c, T)
            B1c = integrate(B1c, T)
            B2c = integrate(B2c, T)
            B3c = integrate(B3c, T)
        Br = B0c + omega*B1c + a*B2c + b*B3c

        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        
        
        Bcalc2 = [BVirial_Tsonopoulos_extended(T2, Tc2, Pc2, omega2, a=a, b=b, order=order) for T2, Tc2, Pc2, omega2 in zip(_Ts, _Tcs, _Pcs, _omegas)]
        assert_allclose(Bcalcs, Bcalc2)


    # Check integrals using numerical methods - quad:
    for order in range(-2, 0):
        for trial in range(10):
            T1, T2 = np.random.random_integers(5, 500, 2)*np.random.rand(2)
            _Tc = np.random.choice(_Tcs)
            _Pc = np.random.choice(_Pcs)
            _omega = np.random.choice(_omegas)
            
            dBint = BVirial_Tsonopoulos_extended(T2, _Tc, _Pc, _omega, a=a, b=b, order=order) - BVirial_Tsonopoulos_extended(T1, _Tc, _Pc, _omega, a=a, b=b, order=order)
            dBquad = quad(BVirial_Tsonopoulos_extended, T1, T2, args=(_Tc, _Pc, _omega, a, b, '', 0, order+1))[0]
            assert_allclose(dBint, dBquad, rtol=3E-5)

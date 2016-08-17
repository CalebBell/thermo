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
#from sympy import *
from sympy import symbols, Rational, diff, lambdify
from scipy.constants import R as _R


def test_BVirial_Pitzer_Curl():

    # doctest
    B = BVirial_Pitzer_Curl(510., 425.2, 38E5, 0.193)
    assert_allclose(B, -0.0002084535541385102)

    # Use SymPy to check that the derivatives are still there
    T, Tc, Pc, omega, R = symbols('T, Tc, Pc, omega, R')
    Tr = T/Tc
    B0 = Rational(1445,10000) - Rational(33,100)/Tr - Rational(1385,10000)/Tr**2 - Rational(121,10000)/Tr**3
    B1 = Rational(73,1000) + Rational(46,100)/Tr - Rational(1,2)/Tr**2 - Rational(97,1000)/Tr**3 - Rational(73,10000)/Tr**8
    
#    _T = 300
#    _Tc = 520
#    _Pc = 1E6
#    _omega = 0.04
    
    # Test 10K points in vector form for order 1, 2, and 3
    # Use lambdify for fast evaluation
    _Ts = np.linspace(5,500,10)
    _Tcs = np.linspace(501,900,10)
    _Pcs = np.linspace(2E5, 1E7,10)
    _omegas = np.linspace(-1, 10,10)
    
    for order in range(1,4):
        B0c = diff(B0, T, order)
        B1c = diff(B1, T, order)
        Br = B0c + omega*B1c
        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        Bcalc2 = BVirial_Pitzer_Curl(_Ts, _Tcs, _Pcs, _omegas, order)
        print(Bcalcs), print(Bcalc2)
        assert_allclose(Bcalcs, Bcalc2)
        
#        for _T in np.linspace(5,500,10):
#            for _Tc in np.linspace(501,900,10):
#                for _Pc in np.linspace(2E5, 1E7,20):
#                    for _omega in np.linspace(-1, 10,10):
#                        #Bcalc = BVirial.subs(T, _T).subs(Tc, _Tc).subs(Pc, _Pc).subs(R, _R).subs(omega, _omega)
#                        Bcalc = f(_T, _Tc, _Pc, _omega)
#                        Bcalc2 = BVirial_Pitzer_Curl(_T, _Tc, _Pc, _omega, order)
#                        assert_allclose(Bcalc, Bcalc2)

#    for order in range(-2, 0):
#        i = order
#        B0c, B1c = B0, B1 # copy problem?
#        for i in range(abs(i)):
#            B0c = integrate(B0c, T)
#            B1c = integrate(B1c, T)
##        print(B0c, B1c)
#
#        Br = B0c + omega*B1c
#        BVirial = (Br*R*Tc/Pc).subs(R, _R)
#        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
#
#        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
#        Bcalc2 = BVirial_Pitzer_Curl(_Ts, _Tcs, _Pcs, _omegas, order)
#        assert_allclose(Bcalcs, Bcalc2)


test_BVirial_Pitzer_Curl()

def test_BVirial_Abbott():
    B = BVirial_Abbott(510., 425.2, 38E5, 0.193)
    assert_allclose(B, -0.00020570178037383633)

    T, Tc, Pc, omega, R = symbols('T, Tc, Pc, omega, R')
    Tr = T/Tc
    B0 = 0.083 - 0.422/Tr**1.6
    B1 = 0.139 - 0.172/Tr**4.2

    _Ts = np.linspace(5,500,10000)
    _Tcs = np.linspace(501,900,10000)
    _Pcs = np.linspace(2E5, 1E7,10000)
    _omegas = np.linspace(-1, 10,10000)


    for order in range(1,4):
        B0c = diff(B0, T, order)
        B1c = diff(B1, T, order)
        Br = B0c + omega*B1c
        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        Bcalc2 = BVirial_Abbott(_Ts, _Tcs, _Pcs, _omegas, order)
        assert_allclose(Bcalcs, Bcalc2)


def test_BVirial_Tsonopoulos():
    B = BVirial_Tsonopoulos(510., 425.2, 38E5, 0.193)
    assert_allclose(B, -0.00020935288308483694)


    T, Tc, Pc, omega, R = symbols('T, Tc, Pc, omega, R')
    Tr = T/Tc
    B0 = Rational(1445, 10000) - Rational(33,100)/Tr - Rational(1385,10000)/Tr**2 - Rational(121,10000)/Tr**3 - Rational(607,1000000)/Tr**8
    B1 = Rational(637,10000) + Rational(331,1000)/Tr**2 - Rational(423,1000)/Tr**3 - Rational(8,1000)/Tr**8

    _Ts = np.linspace(5,500,10000)
    _Tcs = np.linspace(501,900,10000)
    _Pcs = np.linspace(2E5, 1E7,10000)
    _omegas = np.linspace(-1, 10,10000)


    for order in range(1,4):
        B0c = diff(B0, T, order)
        B1c = diff(B1, T, order)
        Br = B0c + omega*B1c
        BVirial = (Br*R*Tc/Pc).subs(R, _R)
        f = lambdify((T, Tc, Pc, omega), BVirial, "numpy")
        
        Bcalcs = f(_Ts, _Tcs, _Pcs, _omegas)
        print(BVirial_Tsonopoulos(_Ts, _Tcs, _Pcs, _omegas, order).shape)
        Bcalc2 = BVirial_Tsonopoulos(_Ts, _Tcs, _Pcs, _omegas, order)
        assert_allclose(Bcalcs, Bcalc2)

test_BVirial_Tsonopoulos()

def test_BVirial_Tsonopoulos_Extended():
    B = BVirial_Tsonopoulos_Extended(510., 425.2, 38E5, 0.193, species_type='normal', dipole=0)
    assert_allclose(B, -0.00020935288308483694)

    B = BVirial_Tsonopoulos_Extended(430., 405.65, 11.28E6, 0.252608, a=0, b=0, species_type='ketone', dipole=1.469)
    assert_allclose(B, -9.679715056695323e-05)

    # Test all of the different types
    types = ['simple', 'normal', 'methyl alcohol', 'water', 'ketone',
    'aldehyde', 'alkyl nitrile', 'ether', 'carboxylic acid', 'ester', 'carboxylic acid',
    'ester', 'alkyl halide', 'mercaptan', 'sulfide', 'disulfide', 'alkanol']

    Bs_calc = [BVirial_Tsonopoulos_Extended(430., 405.65, 11.28E6, 0.252608,
                                            a=0, b=0, species_type=i, dipole=0.1) for i in types]
    Bs = [-9.002529440027288e-05, -9.002529440027288e-05, -8.136805574379563e-05, -9.232250634010228e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.003495446399036e-05, -9.003495446399036e-05, -9.003495446399036e-05, -9.003495446399036e-05, -7.331247111785242e-05]
    assert_allclose(Bs_calc, Bs)


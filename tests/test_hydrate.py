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
from thermo.utils import normalize, TPD
from fluids.core import K2F
from fluids.constants import R, psi
from fluids.numerics import *
from math import log, exp, sqrt
import numpy as np
from thermo.hydrate import *


def test_Caroll_hydrate_formation_P_pure():
    # Ethylene - four points; the 10 and 100 MPa points were validated
    # before the equation form was converted to being T explicit
    P_low_ethylene = Caroll_hydrate_formation_P_pure(273.15, '74-85-1')/1e6
    assert_close(P_low_ethylene, 0.5857960140615912, rtol=1e-10)
    
    P_high_ethylene = Caroll_hydrate_formation_P_pure(273.15+55.0, '74-85-1')/1e6
    assert_close(P_high_ethylene, 445.99727073743344, rtol=1e-10)
    
    P_10MPa_ethylene = Caroll_hydrate_formation_P_pure(292.531593439828, '74-85-1')/1e6
    assert_close(P_10MPa_ethylene, 10, rtol=1e-10)
    P_100MPa_ethylene = Caroll_hydrate_formation_P_pure(304.414007108239, '74-85-1')/1e6
    assert_close(P_100MPa_ethylene, 100, rtol=1e-10)
    
    ethane_0C = Caroll_hydrate_formation_P_pure(273.15, '74-84-0')/1e6
    assert_close(ethane_0C, 0.49057296179018345, rtol=1e-10)
    
    
def test_Motiee_hydrate_formation_T():
    # Ethane
    base = Motiee_hydrate_formation_T(500e3, 1.065)
    assert_allclose(base, 275.7002701118812, rtol=1e-10)
    
    # Check from Ankur of cheresources.com
    T_test_Ankur = 290.2317076736111
    T_test = Motiee_hydrate_formation_T(1000*psi, 0.69)
    assert_allclose(T_test, T_test_Ankur, rtol=1e-14)

    # Check from [3]_ which has 23 points with values
    Ps = ([458, 600, 800] + [250, 480, 980, 2625] 
          + [110, 390, 2050] + [120, 340, 690, 3400]
          + [72, 280, 1700] + [110, 230, 2600] + [60, 195, 440])
    for i in range(len(Ps)):
        Ps[i] *= psi
        
    Ts_compare = ([40.8, 44.8, 49.0] + [36.1, 46.1, 56.0, 68.0]
     + [27.5, 47.4, 68.7] + [33.4, 49.2, 58.7, 76.5]
        + [32.0, 52.0, 72.8] + [42.3, 52.1, 76.6] + [34.8, 50.0, 58.8])
        
    SGs = ([.555]*3 + [.6]*4 + [0.65]*3 + [0.7]*4 + [0.8]*3 + [0.9]*3
           + [1.0]*3)
    
    for i in range(len(Ps)):
        T_lit = Ts_compare[i]
        P = Ps[i]
        SG = SGs[i]
        T = K2F(Motiee_hydrate_formation_T(P, SG))
        assert T_lit == round(T, 1)

def test_Towler_Mokhatab_hydrate_formation_T():

    # point from Ankur
    T_expect = Towler_Mokhatab_hydrate_formation_T(1000*psi, .69)
    assert_allclose(T_expect, 291.0802777198715, rtol=1e-14)
    
    T_experimental = Towler_Mokhatab_hydrate_formation_T(600.0*psi, 0.555)
    assert_allclose(T_experimental, 284.23204260201555)
    
    
def test_Hammerschmidt_hydrate_formation_T():
    # Point from Fattah, SG=0.555 - matches, like the rest of points.
    T_calc = Hammerschmidt_hydrate_formation_T(458*psi)
    assert_allclose(T_calc, 283.7164195466192, rtol=1e-14)
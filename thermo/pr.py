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

from __future__ import division
from scipy.constants import R
from scipy.optimize import fsolve
from numpy import roots
from math import log, exp

R = 8.3145

def alpha(omega,Tr):  # pragma: no cover
    return (1+(0.37464+1.54226*omega-0.26992*omega**2)*(1-Tr**0.5))**2

def a(Tc, Pc):  # pragma: no cover
    a_calc = 0.45724*(R*Tc)**2/Pc
    return a_calc

def b(Tc, Pc):  # pragma: no cover
    b_calc = 0.07780*R*Tc/Pc
    return b_calc

def B(T, P, Tc, Pc):  # pragma: no cover
    b_calc = b(Tc, Pc)
    B_calc = b_calc*P/R/T
    return B_calc

def a_alpha(T, Tc, Pc, omega):  # pragma: no cover
    Tr = T/Tc
    return a(Tc, Pc)*alpha(omega,Tr)

def A(T, P, Tc, Pc, omega):  # pragma: no cover
    A_calc = a_alpha(T, Tc, Pc, omega)*P/(R*T)**2
    return A_calc

# This is the function imported by density
def PR_Vm(T, P, Tc, Pc, omega, phase=''):  # pragma: no cover
    A_calc = A(T, P, Tc, Pc, omega)
    B_calc = B(T, P, Tc, Pc)
    coeffs = [1, -(1-B_calc), (A_calc-3*B_calc**2-2*B_calc), -(A_calc*B_calc-B_calc**2-B_calc**3)]
    numpyroots = roots(coeffs)
    liq_z = min(numpyroots)
    gas_z = max(numpyroots)
    liq = float(liq_z*R*T/P)
    gas = float(gas_z*R*T/P)
    if phase=='liquid' or phase.lower( )== 'l':
        soln = liq
    elif phase=='gas' or phase.lower() == 'g' or T >= Tc or P > Pc:
        soln = gas
    else:
        return liq, gas
    return soln

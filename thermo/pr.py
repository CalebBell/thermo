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

__all__ = ['PR_Vm']
from scipy.constants import R
from scipy.optimize import fsolve
from numpy import roots
from thermo.utils import log, exp

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



### Begin anew here
#from math import sqrt
#from numpy.testing import assert_allclose
#def PR_Vs_analytical(T, P, Tc, Pc, omega):
#    R = 8.3144598
#    a = 0.45724*R*R*Tc*Tc/Pc
#    b = 0.07780*R*Tc/Pc
#    kappa = 0.37464+ 1.54226*omega - 0.26992*omega*omega
#
#    solns = [-(-3*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P*Tc) + (P*b - R*T)**2/P**2)/(3*(sqrt(-4*(-3*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P*Tc) + (P*b - R*T)**2/P**2)**3 + (27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P**2*Tc) + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(2*P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(2*P**2*Tc) + (P*b - R*T)**3/P**3)**(1/3)) - (sqrt(-4*(-3*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P*Tc) + (P*b - R*T)**2/P**2)**3 + (27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P**2*Tc) + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(2*P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(2*P**2*Tc) + (P*b - R*T)**3/P**3)**(1/3)/3 - (P*b - R*T)/(3*P),
#     -(-3*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P*Tc) + (P*b - R*T)**2/P**2)/(3*(-1/2 - sqrt(3)*1j/2)*(sqrt(-4*(-3*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P*Tc) + (P*b - R*T)**2/P**2)**3 + (27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P**2*Tc) + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(2*P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(2*P**2*Tc) + (P*b - R*T)**3/P**3)**(1/3)) - (-1/2 - sqrt(3)*1j/2)*(sqrt(-4*(-3*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P*Tc) + (P*b - R*T)**2/P**2)**3 + (27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P**2*Tc) + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(2*P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(2*P**2*Tc) + (P*b - R*T)**3/P**3)**(1/3)/3 - (P*b - R*T)/(3*P),
#     -(-3*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P*Tc) + (P*b - R*T)**2/P**2)/(3*(-1/2 + sqrt(3)*1j/2)*(sqrt(-4*(-3*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P*Tc) + (P*b - R*T)**2/P**2)**3 + (27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P**2*Tc) + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(2*P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(2*P**2*Tc) + (P*b - R*T)**3/P**3)**(1/3)) - (-1/2 + sqrt(3)*1j/2)*(sqrt(-4*(-3*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P*Tc) + (P*b - R*T)**2/P**2)**3 + (27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(P**2*Tc) + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*Tc*b**3 + R*T*Tc*b**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa**2 + 2*sqrt(T)*sqrt(Tc)*a*b*kappa - T*a*b*kappa**2 - Tc*a*b*kappa**2 - 2*Tc*a*b*kappa - Tc*a*b)/(2*P*Tc) - 9*(P*b - R*T)*(-3*P*Tc*b**2 - 2*R*T*Tc*b - 2*sqrt(T)*sqrt(Tc)*a*kappa**2 - 2*sqrt(T)*sqrt(Tc)*a*kappa + T*a*kappa**2 + Tc*a*kappa**2 + 2*Tc*a*kappa + Tc*a)/(2*P**2*Tc) + (P*b - R*T)**3/P**3)**(1/3)/3 - (P*b - R*T)/(3*P)]
#    return solns
#
#V_test = PR_Vs_analytical(300., P=11E5, Tc=190.6, Pc=46.002E5, omega=0.0080)
#expected = [(1.3375476396313931e-05+3.6182392942487305e-05j), (1.3375476396314256e-05-3.61823929424872e-05j), (0.0022140274484362173-2.126842635914645e-20j)]
#assert_allclose(V_test, expected)
#
#
#

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
from scipy.optimize import fsolve
from math import exp, log
import numpy as np
import os
from thermo.utils import none_and_length_check

from scipy.constants import R
C2J = 4.1868

folder = os.path.join(os.path.dirname(__file__), 'Phase Change')


def K(P, Psat, fugacity=1, gamma=1):
    '''
    >>> K(101325, 3000.)
    0.029607698001480384
    >>> K(101325, 3000., fugacity=0.9, gamma=2.4)
    0.07895386133728102
    '''
    _K = Psat*gamma/P/fugacity
    return _K



### Solutions using a existing algorithms
def Rachford_Rice_flash_error(V_over_F, zs, ks):
    total = 0
    for i in range(len(zs)):
        total += zs[i]*(ks[i]-1)/(1+V_over_F*(ks[i]-1))
    return total


def flash(P, zs, Psats, fugacities=None, gammas=None):
    if not fugacities:
        fugacities = [1 for i in range(len(zs))]
    if not gammas:
        gammas = [1 for i in range(len(zs))]
    if not none_and_length_check((zs, Psats, fugacities, gammas)):
        raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
    ks = [K(P, Psats[i], fugacities[i], gammas[i]) for i in range(len(zs))]

    def valid_range(zs, ks):
        valid = True
        if sum([zs[i]*ks[i] for i in range(len(ks))]) < 1:
            valid = False
        if sum([zs[i]/ks[i] for i in range(len(ks))]) < 1:
            valid = False
        return valid
    if not valid_range(zs, ks):
        raise Exception('Solution does not exist')
#    zs = [0.5, 0.3, 0.2] practice solution
#    ks = [1.685, 0.742, 0.532]
    x0 = np.array([.5])
    V_over_F = fsolve(Rachford_Rice_flash_error, x0=x0, args=(zs, ks))[0]
    if V_over_F < 0:
#        print zs, ks
        raise Exception('V_over_F is negative!')
    xs = [zs[i]/(1+V_over_F*(ks[i]-1)) for i in range(len(zs))]
    ys = [ks[i]*xs[i] for i in range(len(zs))]
    return xs, ys, V_over_F




def dew_at_T(zs, Psats, fugacities=None, gammas=None):
    '''
    >>> dew_at_T([0.5, 0.5], [1400, 7000])
    2333.3333333333335
    >>> dew_at_T([0.5, 0.5], [1400, 7000], gammas=[1.1, .75])
    2381.443298969072
    >>> dew_at_T([0.5, 0.5], [1400, 7000], gammas=[1.1, .75], fugacities=[.995, 0.98])
    2401.621874512658
    '''
    if not fugacities:
        fugacities = [1 for i in range(len(Psats))]
    if not gammas:
        gammas = [1 for i in range(len(Psats))]
    if not none_and_length_check((zs, Psats, fugacities, gammas)):
        raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
    P = 1/sum(zs[i]*fugacities[i]/Psats[i]/gammas[i] for i in range(len(zs)))
    return P


def bubble_at_T(zs, Psats, fugacities=None, gammas=None):
    '''
    >>> bubble_at_T([0.5, 0.5], [1400, 7000])
    4200.0
    >>> bubble_at_T([0.5, 0.5], [1400, 7000], gammas=[1.1, .75])
    3395.0
    >>> bubble_at_T([0.5, 0.5], [1400, 7000], gammas=[1.1, .75], fugacities=[.995, 0.98])
    3452.440775305097
    '''
    if not fugacities:
        fugacities = [1 for i in range(len(Psats))]
    if not gammas:
        gammas = [1 for i in range(len(Psats))]
    if not none_and_length_check((zs, Psats, fugacities, gammas)):
        raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
    P = sum(zs[i]*Psats[i]*gammas[i]/fugacities[i] for i in range(len(zs)))
    return P


def identify_phase(T=None, P=None, Tm=None, Tb=None, Tc=None, Psat=None):
    '''
    >>> identify_phase(T=280, P=101325, Tm=273.15, Psat=991)
    'l'
    >>> identify_phase(T=480, P=101325, Tm=273.15, Psat=1791175)
    'g'
    >>> identify_phase(T=650, P=10132500000, Tm=273.15, Psat=None, Tc=647.3)
    'g'
    >>> identify_phase(T=250, P=100, Tm=273.15)
    's'
    >>> identify_phase(T=500, P=101325)
    '''
    phase = None
    if not T or not P:
        raise Exception('Phase identification requires T and P.')
    if Tm and T <= Tm:
        phase = 's'
    elif Tc and T > Tc:
        phase = 'g'
    elif Psat:
        if P < Psat:
            phase = 'g'
        elif P > Psat:
            phase = 'l'
    elif Tb and Tm:
        if 9E4 < P < 1.1E5: # mild tolerance
            if T < Tb:
                phase = 'l'
            else:
                phase = 'g'
        elif P > 1.1E5 and T < Tb:
            phase = 'l'
        else:
            phase = None
    else:
        phase = None
    return phase


IDEALVLE = 'Ideal'
SUPERCRITICALT = 'Critical temperature criteria'
SUPERCRITICALP = 'Critical pressure criteria'
IDEALVLESUPERCRITICAL = 'Ideal with supercritical components'
NONE = 'None'


def identify_phase_mixture(T=None, P=None, zs=None, Tcs=None, Pcs=None,
                           Psats=None, CASRNs=None,
                           AvailableMethods=False, Method=None):  # pragma: no cover
    '''
    >>> identify_phase_mixture(T=280, P=5000., zs=[0.5, 0.5], Psats=[1400, 7000])
    ('l', [0.5, 0.5], None, 0)
    >>> identify_phase_mixture(T=280, P=3000., zs=[0.5, 0.5], Psats=[1400, 7000])
    ('two-phase', [0.7142857142857143, 0.2857142857142857], [0.33333333333333337, 0.66666666666666663], 0.5625)
    >>> identify_phase_mixture(T=280, P=800., zs=[0.5, 0.5], Psats=[1400, 7000])
    ('g', None, [0.5, 0.5], 1)
    >>> identify_phase_mixture(T=280, P=800., zs=[0.5, 0.5])
    (None, None, None, None)
    '''
    def list_methods():
        methods = []
        if none_and_length_check((Psats, zs)):
            methods.append(IDEALVLE)
        if none_and_length_check([Tcs]) and all([T >= i for i in Tcs]):
            methods.append(SUPERCRITICALT)
        if none_and_length_check([Pcs]) and all([P >= i for i in Pcs]):
            methods.append(SUPERCRITICALP)
        if none_and_length_check((zs, Tcs)) and any([T > Tc for Tc in Tcs]):
            methods.append(IDEALVLESUPERCRITICAL)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    xs, ys, phase, V_over_F = None, None, None, None
    if Method == IDEALVLE:
        Pdew = dew_at_T(zs, Psats)
        Pbubble = bubble_at_T(zs, Psats)
        if P >= Pbubble:
            phase = 'l'
            ys = None
            xs = zs
            V_over_F = 0
        elif P <= Pdew:
            phase = 'g'
            ys = zs
            xs = None
            V_over_F = 1
        elif Pdew < P < Pbubble:
            xs, ys, V_over_F = flash(P, zs, Psats)
            phase = 'two-phase'
    elif Method == SUPERCRITICALT:
        if all([T >= i for i in Tcs]):
            phase = 'g'
        else: # The following is nonsensical
            phase = 'two-phase'
    elif Method == SUPERCRITICALP:
        if all([P >= i for i in Pcs]):
            phase = 'g'
        else: # The following is nonsensical
            phase = 'two-phase'
    elif Method == IDEALVLESUPERCRITICAL:
        Psats = list(Psats)
        for i in range(len(Psats)):
            if not Psats[i] and Tcs[i] and Tcs[i] <= T:
                Psats[i] = 1E8
        Pdew = dew_at_T(zs, Psats)
        Pbubble = 1E99
        if P >= Pbubble:
            phase = 'l'
            ys = None
            xs = zs
            V_over_F = 0
        elif P <= Pdew:
            phase = 'g'
            ys = zs
            xs = None
            V_over_F = 1
        elif Pdew < P < Pbubble:
            xs, ys, V_over_F = flash(P, zs, Psats)
            phase = 'two-phase'

    elif Method == NONE:
        pass
    else:
        raise Exception('Failure in in function')
    return phase, xs, ys, V_over_F


def Pbubble_mixture(T=None, zs=None, Psats=None, CASRNs=None,
                   AvailableMethods=False, Method=None):  # pragma: no cover
    '''
    >>> Pbubble_mixture(zs=[0.5, 0.5], Psats=[1400, 7000])
    4200.0
    '''
    def list_methods():
        methods = []
        if none_and_length_check((Psats, zs)):
            methods.append(IDEALVLE)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == IDEALVLE:
        Pbubble = bubble_at_T(zs, Psats)
    elif Method == NONE:
        Pbubble = None
    else:
        raise Exception('Failure in in function')
    return Pbubble


def Pdew_mixture(T=None, zs=None, Psats=None, CASRNs=None,
                 AvailableMethods=False, Method=None):  # pragma: no cover
    '''
    >>> Pdew_mixture(zs=[0.5, 0.5], Psats=[1400, 7000])
    2333.3333333333335
    '''
    def list_methods():
        methods = []
        if none_and_length_check((Psats, zs)):
            methods.append(IDEALVLE)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == IDEALVLE:
        Pdew = dew_at_T(zs, Psats)
    elif Method == NONE:
        Pdew = None
    else:
        raise Exception('Failure in in function')
    return Pdew




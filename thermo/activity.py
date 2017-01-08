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

__all__ = ['K', 'Rachford_Rice_flash_error', 'Rachford_Rice_solution', 'Li_Johns_Ahmadi_solution', 'flash', 'dew_at_T', 
           'bubble_at_T', 'identify_phase', 'mixture_phase_methods', 
           'identify_phase_mixture', 'Pbubble_mixture', 'Pdew_mixture']

from scipy.optimize import fsolve, newton, brenth
from thermo.utils import exp, log
import numpy as np
import os
from thermo.utils import none_and_length_check

from scipy.constants import R

folder = os.path.join(os.path.dirname(__file__), 'Phase Change')


def K(P, Psat, fugacity=1, gamma=1):
    '''
    >>> K(101325, 3000.)
    0.029607698001480384
    >>> K(101325, 3000., fugacity=0.9, gamma=2.4)
    0.07895386133728102
    '''
    # http://www.jmcampbell.com/tip-of-the-month/2006/09/how-to-determine-k-values/
    return Psat*gamma/P/fugacity





### Solutions using a existing algorithms
def Rachford_Rice_flash_error(V_over_F, zs, Ks):
    r'''Calculates the objective function of the Rachford-Rice flash equation.
    This function should be called by a solver seeking a solution to a flash
    calculation. The unknown variable is `V_over_F`, for which a solution 
    must be between 0 and 1.
    
    .. math::
        \sum_i \frac{z_i(K_i-1)}{1 + \frac{V}{F}(K_i-1)} = 0

    Parameters
    ----------
    V_over_F : float
        Vapor fraction guess [-]
    zs : list[float]
        Overall mole fractions of all species, [-]
    Ks : list[float]
        Equilibrium K-values, [-]

    Returns
    -------
    error : float
        Deviation between the objective function at the correct V_over_F
        and the attempted V_over_F, [-]

    Notes
    -----
    The derivation is as follows:
    
    .. math::
        F z_i = L x_i + V y_i
        
        x_i = \frac{z_i}{1 + \frac{V}{F}(K_i-1)}
        
        \sum_i y_i = \sum_i K_i x_i = 1
        
        \sum_i(y_i - x_i)=0
        
        \sum_i \frac{z_i(K_i-1)}{1 + \frac{V}{F}(K_i-1)} = 0

    Examples
    --------
    >>> Rachford_Rice_flash_error(0.5, zs=[0.5, 0.3, 0.2], 
    ... Ks=[1.685, 0.742, 0.532])
    0.04406445591174976

    References
    ----------
    .. [1] Rachford, H. H. Jr, and J. D. Rice. "Procedure for Use of Electronic
       Digital Computers in Calculating Flash Vaporization Hydrocarbon 
       Equilibrium." Journal of Petroleum Technology 4, no. 10 (October 1, 
       1952): 19-3. doi:10.2118/952327-G.
    '''
    return sum([zi*(Ki-1.)/(1.+V_over_F*(Ki-1.)) for Ki, zi in zip(Ks, zs)])


def Rachford_Rice_solution(zs, Ks):
    r'''Solves the objective function of the Rachford-Rice flash equation.
    Uses the method proposed in [2]_ to obtain an initial guess.
    
    .. math::
        \sum_i \frac{z_i(K_i-1)}{1 + \frac{V}{F}(K_i-1)} = 0

    Parameters
    ----------
    zs : list[float]
        Overall mole fractions of all species, [-]
    Ks : list[float]
        Equilibrium K-values, [-]

    Returns
    -------
    V_over_F : float
        Vapor fraction solution [-]
    xs : list[float]
        Mole fractions of each species in the liquid phase, [-]
    ys : list[float]
        Mole fractions of each species in the vapor phase, [-]

    Notes
    -----
    The initial guess is the average of the following, as described in [2]_.
    
    .. math::
        \left(\frac{V}{F}\right)_{min} = \frac{(K_{max}-K_{min})z_{of\;K_{max}} 
        - (1-K_{min})}{(1-K_{min})(K_{max}-1)}

        \left(\frac{V}{F}\right)_{max} = \frac{1}{1-K_{min}}
    
    Another algorithm for determining the range of the correct solution is
    given in [3]_; [2]_ provides a narrower range however. For both cases,
    each guess should be limited to be between 0 and 1 as they are often
    negative or larger than 1.
    
    .. math::
        \left(\frac{V}{F}\right)_{min} = \frac{1}{1-K_{max}}
        
        \left(\frac{V}{F}\right)_{max} = \frac{1}{1-K_{min}}

    If the `newton` method does not converge, a bisection method (brenth) is 
    used instead. However, it is somewhat slower, especially as newton will
    attempt 50 iterations before giving up.

    Examples
    --------
    >>> Rachford_Rice_solution(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    (0.6907302627738544, [0.33940869696634357, 0.3650560590371706, 0.2955352439964858], [0.5719036543882889, 0.27087159580558057, 0.15722474980613044])

    References
    ----------
    .. [1] Rachford, H. H. Jr, and J. D. Rice. "Procedure for Use of Electronic
       Digital Computers in Calculating Flash Vaporization Hydrocarbon 
       Equilibrium." Journal of Petroleum Technology 4, no. 10 (October 1, 
       1952): 19-3. doi:10.2118/952327-G.
    .. [2] Li, Yinghui, Russell T. Johns, and Kaveh Ahmadi. "A Rapid and Robust
       Alternative to Rachford-Rice in Flash Calculations." Fluid Phase 
       Equilibria 316 (February 25, 2012): 85-97. 
       doi:10.1016/j.fluid.2011.12.005.
    .. [3] Whitson, Curtis H., and Michael L. Michelsen. "The Negative Flash." 
       Fluid Phase Equilibria, Proceedings of the Fifth International 
       Conference, 53 (December 1, 1989): 51-71. 
       doi:10.1016/0378-3812(89)80072-X.
    '''
    # TODO binary and ternary explicit solutions
    Kmin = min(Ks)
    Kmax = max(Ks)
    z_of_Kmax = zs[Ks.index(Kmax)]

    V_over_F_min = ((Kmax-Kmin)*z_of_Kmax - (1.-Kmin))/((1.-Kmin)*(Kmax-1.))
    V_over_F_max = 1./(1.-Kmin)
        
    V_over_F_min2 = max(0., V_over_F_min)
    V_over_F_max2 = min(1., V_over_F_max)

    x0 = (V_over_F_min2 + V_over_F_max2)*0.5
    try:
        # Newton's method is marginally faster than brenth
        V_over_F = newton(Rachford_Rice_flash_error, x0=x0, args=(zs, Ks))
    except:
        V_over_F = brenth(Rachford_Rice_flash_error, V_over_F_max-1E-7, V_over_F_min+1E-7, args=(zs, Ks))
    # Cases not covered by the above solvers: When all components have K > 1, or all have K < 1
    # Should get a solution for all other cases.
    xs = [zi/(1.+V_over_F*(Ki-1.)) for zi, Ki in zip(zs, Ks)]
    ys = [Ki*xi for xi, Ki in zip(xs, Ks)]
    return V_over_F, xs, ys


def Li_Johns_Ahmadi_solution(zs, Ks):
    # Re-order both Ks and Zs by K value, higher coming first
    p = sorted(zip(Ks,zs), reverse=True)
    Ks_sorted, zs_sorted = [K for (K,z) in p], [z for (K,z) in p]

    # Largest K value and corresponding overall mole fraction
    k1 = Ks_sorted[0]
    z1 = zs_sorted[0]
    # Smallest K value
    kn = Ks_sorted[-1]

    V_over_F_min = (1. - kn)/(k1 - kn)*z1
    V_over_F_max = (1. - kn)/(k1 - kn)
    
    V_over_F_min2 = max(0., V_over_F_min)
    V_over_F_max2 = min(1., V_over_F_max)
    
    V_over_F_guess = (V_over_F_min2 + V_over_F_max2)*0.5
    
    length = len(zs)-1
    kn_m_1 = kn-1.
    k1_m_1 = (k1-1.)
    t1 = (k1-kn)/(kn-1.)
    iterable = zip(Ks_sorted[1:length], zs_sorted[1:length])
    
    objective = lambda x1: 1. + t1*x1 + sum([(ki-kn)/(kn_m_1) * zi*k1_m_1*x1 /( (ki-1.)*z1 + (k1-ki)*x1) for ki, zi in iterable])
#    try:
#        x1 = newton(objective, V_over_F_guess)
#    except:
    # newton skips out of its specified range in some cases, finding another solution
    x1 = brenth(objective, V_over_F_max-1E-7, V_over_F_min+1E-7)
    V_over_F = (-x1 + z1)/(x1*(k1 - 1.))
    xs = [zi/(1.+V_over_F*(Ki-1.)) for zi, Ki in zip(zs, Ks)]
    ys = [Ki*xi for xi, Ki in zip(xs, Ks)]
    return V_over_F, xs, ys



def flash(P, zs, Psats, fugacities=None, gammas=None):
    if not fugacities:
        fugacities = [1 for i in range(len(zs))]
    if not gammas:
        gammas = [1 for i in range(len(zs))]
    if not none_and_length_check((zs, Psats, fugacities, gammas)):
        raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
    Ks = [K(P, Psats[i], fugacities[i], gammas[i]) for i in range(len(zs))]

    def valid_range(zs, Ks):
        valid = True
        if sum([zs[i]*Ks[i] for i in range(len(Ks))]) < 1:
            valid = False
        if sum([zs[i]/Ks[i] for i in range(len(Ks))]) < 1:
            valid = False
        return valid
    if not valid_range(zs, Ks):
        raise Exception('Solution does not exist')
    x0 = np.array([.5])
    V_over_F = fsolve(Rachford_Rice_flash_error, x0=x0, args=(zs, Ks))[0]
    if V_over_F < 0:
        raise Exception('V_over_F is negative!')
    xs = [zs[i]/(1+V_over_F*(Ks[i]-1)) for i in range(len(zs))]
    ys = [Ks[i]*xs[i] for i in range(len(zs))]
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


mixture_phase_methods = ['IDEAL_VLE', 'SUPERCRITICAL_T', 'SUPERCRITICAL_P', 'IDEAL_VLE_SUPERCRITICAL']

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
            methods.append('IDEAL_VLE')
        if none_and_length_check([Tcs]) and all([T >= i for i in Tcs]):
            methods.append('SUPERCRITICAL_T')
        if none_and_length_check([Pcs]) and all([P >= i for i in Pcs]):
            methods.append('SUPERCRITICAL_P')
        if none_and_length_check((zs, Tcs)) and any([T > Tc for Tc in Tcs]):
            methods.append('IDEAL_VLE_SUPERCRITICAL')
        methods.append('NONE')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    xs, ys, phase, V_over_F = None, None, None, None
    if Method == 'IDEAL_VLE':
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
    elif Method == 'SUPERCRITICAL_T':
        if all([T >= i for i in Tcs]):
            phase = 'g'
        else: # The following is nonsensical
            phase = 'two-phase'
    elif Method == 'SUPERCRITICAL_P':
        if all([P >= i for i in Pcs]):
            phase = 'g'
        else: # The following is nonsensical
            phase = 'two-phase'
    elif Method == 'IDEAL_VLE_SUPERCRITICAL':
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

    elif Method == 'NONE':
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
            methods.append('IDEAL_VLE')
        methods.append('NONE')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == 'IDEAL_VLE':
        Pbubble = bubble_at_T(zs, Psats)
    elif Method == 'NONE':
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
            methods.append('IDEAL_VLE')
        methods.append('NONE')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == 'IDEAL_VLE':
        Pdew = dew_at_T(zs, Psats)
    elif Method == 'NONE':
        Pdew = None
    else:
        raise Exception('Failure in in function')
    return Pdew




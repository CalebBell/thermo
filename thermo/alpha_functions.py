# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from __future__ import division, print_function

__all__ = ['Poly_a_alpha', 'TwuSRK95_a_alpha', 'TwuPR95_a_alpha',

]


from cmath import atanh as catanh, log as clog
from fluids.numerics import (chebval, brenth, third, sixth, roots_cubic,
                             roots_cubic_a1, numpy as np, py_newton as newton,
                             py_bisect as bisect, inf, polyder, chebder, 
                             trunc_exp, secant, linspace, logspace,
                             horner, horner_and_der, horner_and_der2, derivative,
                             roots_cubic_a2, isclose, NoSolutionError,
                             roots_quartic)
from thermo.utils import R
from thermo.utils import log, log10, exp, sqrt, copysign


def TWU_a_alpha_common(T, Tc, omega, a, full=True, quick=True, method='PR'):
    r'''Function to calculate `a_alpha` and optionally its first and second
    derivatives for the TWUPR or TWUSRK EOS. Returns 'a_alpha', and 
    optionally 'da_alpha_dT' and 'd2a_alpha_dT2'.
    Used by `TWUPR` and `TWUSRK`; has little purpose on its own.
    See either class for the correct reference, and examples of using the EOS.

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tc : float
        Critical temperature, [K]
    omega : float
        Acentric factor, [-]
    a : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
    full : float
        Whether or not to return its first and second derivatives
    quick : bool, optional
        Whether to use a SymPy cse-derived expression (3x faster) or 
        individual formulas
    method : str
        Either 'PR' or 'SRK'
        
    Notes
    -----
    The derivatives are somewhat long and are not described here for 
    brevity; they are obtainable from the following SymPy expression.
    
    >>> from sympy import *
    >>> T, Tc, omega, N1, N0, M1, M0, L1, L0 = symbols('T, Tc, omega, N1, N0, M1, M0, L1, L0')
    >>> Tr = T/Tc
    >>> alpha0 = Tr**(N0*(M0-1))*exp(L0*(1-Tr**(N0*M0)))
    >>> alpha1 = Tr**(N1*(M1-1))*exp(L1*(1-Tr**(N1*M1)))
    >>> alpha = alpha0 + omega*(alpha1-alpha0)
    >>> # diff(alpha, T)
    >>> # diff(alpha, T, T)
    '''
    # e-10 works
    min_a_alpha = 1e-3 # There are a LOT of formulas, and they do not like having zeros
    Tr = T/Tc
    if Tr < 5e-3:
        # not enough: Tr from (x) 0 to 2e-4 to (y) 1e-4 2e-4
        # trying: Tr from (x) 0 to 1e-3 to (y) 5e-4 1e-3
#        Tr = 1e-3 + (Tr - 0.0)*(1e-3 - 5e-4)/1e-3
#        Tr = 5e-4 + (Tr - 0.0)*(5e-4)/1e-3
        Tr = 4e-3 + (Tr - 0.0)*(1e-3)/5e-3
        T = Tc*Tr
    
    if method == 'PR':
        if Tr < 1.0:
            L0, M0, N0 = 0.125283, 0.911807, 1.948150
            L1, M1, N1 = 0.511614, 0.784054, 2.812520
        else:
            L0, M0, N0 = 0.401219, 4.963070, -0.2
            L1, M1, N1 = 0.024955, 1.248089, -8.  
    elif method == 'SRK':
        if Tr < 1.0:
            L0, M0, N0 = 0.141599, 0.919422, 2.496441
            L1, M1, N1 = 0.500315, 0.799457, 3.291790
        else:
            L0, M0, N0 = 0.441411, 6.500018, -0.20
            L1, M1, N1 = 0.032580,  1.289098, -8.0
    else:
        raise Exception('Only `PR` and `SRK` are accepted as method')
    
    if not full:
        alpha0 = Tr**(N0*(M0-1.))*exp(L0*(1.-Tr**(N0*M0)))
        alpha1 = Tr**(N1*(M1-1.))*exp(L1*(1.-Tr**(N1*M1)))
        alpha = alpha0 + omega*(alpha1 - alpha0)
        a_alpha = a*alpha
        if a_alpha < min_a_alpha:
            a_alpha = min_a_alpha
        return a_alpha
    else:
        if quick:
            x0 = Tr
            x1 = M0 - 1
            x2 = N0*x1
            x3 = x0**x2
            x4 = M0*N0
            x5 = x0**x4
            x6 = exp(-L0*(x5 - 1.))
            x7 = x3*x6
            x8 = M1 - 1.
            x9 = N1*x8
            x10 = x0**x9
            x11 = M1*N1
            x12 = x0**x11
            x13 = x2*x7
            x14 = L0*M0*N0*x3*x5*x6
            x15 = x13 - x14
            x16 = exp(-L1*(x12 - 1))
            x17 = -L1*M1*N1*x10*x12*x16 + x10*x16*x9 - x13 + x14
            x18 = N0*N0
            x19 = x18*x3*x6
            x20 = x1**2*x19
            x21 = M0**2
            x22 = L0*x18*x3*x5*x6
            x23 = x21*x22
            x24 = 2*M0*x1*x22
            x25 = L0**2*x0**(2*x4)*x19*x21
            x26 = N1**2
            x27 = x10*x16*x26
            x28 = M1**2
            x29 = L1*x10*x12*x16*x26
            a_alpha = a*(-omega*(-x10*exp(L1*(-x12 + 1)) + x3*exp(L0*(-x5 + 1))) + x7)
            da_alpha_dT = a*(omega*x17 + x15)/T
            d2a_alpha_dT2 = a*(-(omega*(-L1**2*x0**(2.*x11)*x27*x28 + 2.*M1*x29*x8 + x17 + x20 - x23 - x24 + x25 - x27*x8**2 + x28*x29) + x15 - x20 + x23 + x24 - x25)/T**2)
        else:
            alpha0 = Tr**(N0*(M0-1.))*exp(L0*(1.-Tr**(N0*M0)))
            alpha1 = Tr**(N1*(M1-1.))*exp(L1*(1.-Tr**(N1*M1)))
            alpha = alpha0 + omega*(alpha1 - alpha0)
            a_alpha = a*alpha
#            a_alpha = TWU_a_alpha_common(T=T, Tc=Tc, omega=omega, a=a, full=False, quick=quick, method=method)
            da_alpha_dT = a*(-L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T + N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T + omega*(L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T - L1*M1*N1*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(L1*(-(T/Tc)**(M1*N1) + 1))/T - N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T + N1*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)*exp(L1*(-(T/Tc)**(M1*N1) + 1))/T))
            d2a_alpha_dT2 = a*((L0**2*M0**2*N0**2*(T/Tc)**(2*M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - L0*M0**2*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - 2*L0*M0*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) + L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) + N0**2*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)**2*exp(-L0*((T/Tc)**(M0*N0) - 1)) - N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) - omega*(L0**2*M0**2*N0**2*(T/Tc)**(2*M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - L0*M0**2*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - 2*L0*M0*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) + L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - L1**2*M1**2*N1**2*(T/Tc)**(2*M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(-L1*((T/Tc)**(M1*N1) - 1)) + L1*M1**2*N1**2*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(-L1*((T/Tc)**(M1*N1) - 1)) + 2*L1*M1*N1**2*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)*exp(-L1*((T/Tc)**(M1*N1) - 1)) - L1*M1*N1*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(-L1*((T/Tc)**(M1*N1) - 1)) + N0**2*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)**2*exp(-L0*((T/Tc)**(M0*N0) - 1)) - N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) - N1**2*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)**2*exp(-L1*((T/Tc)**(M1*N1) - 1)) + N1*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)*exp(-L1*((T/Tc)**(M1*N1) - 1))))/T**2)
        if a_alpha < min_a_alpha:
            a_alpha = min_a_alpha
            da_alpha_dT = d2a_alpha_dT2 = 0.0
            # Hydrogen at low T
#            a_alpha = da_alpha_dT = d2a_alpha_dT2 = 0.0
        return a_alpha, da_alpha_dT, d2a_alpha_dT2


class a_alpha_base(object):
    def _init_test(self, Tc, a, alpha_coeffs, **kwargs):
        self.Tc = Tc
        self.a = a
        self.alpha_coeffs = alpha_coeffs
        self.__dict__.update(kwargs)

class Poly_a_alpha(object):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        if not full:
            return horner(self.alpha_coeffs, T)
        else:
            return horner_and_der2(self.alpha_coeffs, T)


class Soave_1972_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1972) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Same as `SRK.a_alpha_and_derivatives` but slower and
        requiring `alpha_coeffs` to be set. One coefficient needed.
        
        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{Tc}} + 1\right)
            + 1\right)^{2}

        References
        ----------
        .. [1] Soave, Giorgio. "Equilibrium Constants from a Modified Redlich-
           Kwong Equation of State." Chemical Engineering Science 27, no. 6 
           (June 1972): 1197-1203. doi:10.1016/0009-2509(72)80096-4.
        .. [2] Young, André F., Fernando L. P. Pessoa, and Victor R. R. Ahón. 
           "Comparison of 20 Alpha Functions Applied in the Peng–Robinson  
           Equation of State for Vapor Pressure Estimation." Industrial &  
           Engineering Chemistry Research 55, no. 22 (June 8, 2016): 6506-16. 
           doi:10.1021/acs.iecr.6b00721.
        '''
        c1 = self.alpha_coeffs[0]
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) + 1)**2
        if not full:
            return a_alpha
        else:
            da_alpha_dT = -a*c1*sqrt(T/Tc)*(c1*(-sqrt(T/Tc) + 1) + 1)/T
            d2a_alpha_dT2 = a*c1*(c1/Tc - sqrt(T/Tc)*(c1*(sqrt(T/Tc) - 1) - 1)/T)/(2*T)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

class Heyen_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Heyen (1980) [1]_. Returns `a_alpha`, `da_alpha_dT`,  
        and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Two coefficients needed.
        
        .. math::
            \alpha = e^{c_{1} \left(- \left(\frac{T}{Tc}\right)^{c_{2}}
            + 1\right)}

        References
        ----------
        .. [1] Heyen, G. Liquid and Vapor Properties from a Cubic Equation of 
           State. In "Proceedings of the 2nd International Conference on Phase 
           Equilibria and Fluid Properties in the Chemical Industry". DECHEMA: 
           Frankfurt, 1980; p 9-13.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp(c1*(1 -(T/Tc)**c2))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = -a*c1*c2*(T/Tc)**c2*exp(c1*(-(T/Tc)**c2 + 1))/T
            d2a_alpha_dT2 = a*c1*c2*(T/Tc)**c2*(c1*c2*(T/Tc)**c2 - c2 + 1)*exp(-c1*((T/Tc)**c2 - 1))/T**2
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
        
        
class Harmens_Knapp_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Harmens and Knapp (1980) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Two coefficients needed.
        
        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{Tc}} + 1\right) 
            - c_{2} \left(1 - \frac{Tc}{T}\right) + 1\right)^{2}

        References
        ----------
        .. [1] Harmens, A., and H. Knapp. "Three-Parameter Cubic Equation of 
           State for Normal Substances." Industrial & Engineering Chemistry 
           Fundamentals 19, no. 3 (August 1, 1980): 291-94. 
           doi:10.1021/i160075a010. 
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) - c2*(1 - Tc/T) + 1)**2
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-c1*sqrt(T/Tc)/T - 2*Tc*c2/T**2)*(c1*(-sqrt(T/Tc) + 1) - c2*(1 - Tc/T) + 1)
            d2a_alpha_dT2 = a*((c1*sqrt(T/Tc) + 2*Tc*c2/T)**2 - (c1*sqrt(T/Tc) + 8*Tc*c2/T)*(c1*(sqrt(T/Tc) - 1) + c2*(1 - Tc/T) - 1))/(2*T**2)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Mathias_1983_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Mathias (1983) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Two coefficients needed.
        
        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{Tc}} + 1\right)
            - c_{2} \left(- \frac{T}{Tc} + 0.7\right) \left(- \frac{T}{Tc} 
            + 1\right) + 1\right)^{2}

        References
        ----------
        .. [1] Mathias, Paul M. "A Versatile Phase Equilibrium Equation of 
           State." Industrial & Engineering Chemistry Process Design and 
           Development 22, no. 3 (July 1, 1983): 385-91. 
           doi:10.1021/i200022a008. 
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(1 + c1*(1-sqrt(Tr)) -c2*(1-Tr)*(0.7-Tr))**2
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(c1*(-sqrt(T/Tc) + 1) - c2*(-T/Tc + 0.7)*(-T/Tc + 1) + 1)*(2*c2*(-T/Tc + 0.7)/Tc + 2*c2*(-T/Tc + 1)/Tc - c1*sqrt(T/Tc)/T)
            d2a_alpha_dT2 = a*((8*c2/Tc**2 - c1*sqrt(T/Tc)/T**2)*(c1*(sqrt(T/Tc) - 1) + c2*(T/Tc - 1)*(T/Tc - 0.7) - 1) + (2*c2*(T/Tc - 1)/Tc + 2*c2*(T/Tc - 0.7)/Tc + c1*sqrt(T/Tc)/T)**2)/2
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Mathias_Copeman_untruncated_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Mathias and Copeman (1983) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Three coefficients needed.
        
        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{Tc}} + 1\right)
            + c_{2} \left(- \sqrt{\frac{T}{Tc}} + 1\right)^{2} + c_{3} \left(
            - \sqrt{\frac{T}{Tc}} + 1\right)^{3} + 1\right)^{2}

        References
        ----------
        .. [1] Mathias, Paul M., and Thomas W. Copeman. "Extension of the 
           Peng-Robinson Equation of State to Complex Mixtures: Evaluation of 
           the Various Forms of the Local Composition Concept." Fluid Phase 
           Equilibria 13 (January 1, 1983): 91-108. 
           doi:10.1016/0378-3812(83)80084-3. 
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) + c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-c1*sqrt(T/Tc)/T - 2*c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T - 3*c3*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)**2/T)*(c1*(-sqrt(T/Tc) + 1) + c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)
            d2a_alpha_dT2 = a*(T*(c1 - 2*c2*(sqrt(T/Tc) - 1) + 3*c3*(sqrt(T/Tc) - 1)**2)**2 - (2*T*(c2 - 3*c3*(sqrt(T/Tc) - 1)) + Tc*sqrt(T/Tc)*(c1 - 2*c2*(sqrt(T/Tc) - 1) + 3*c3*(sqrt(T/Tc) - 1)**2))*(c1*(sqrt(T/Tc) - 1) - c2*(sqrt(T/Tc) - 1)**2 + c3*(sqrt(T/Tc) - 1)**3 - 1))/(2*T**2*Tc)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Mathias_Copeman_a_alpha(a_alpha_base):
    
    def a_alpha_and_derivatives_vectorized(self, T, full=False):
        ais, alpha_coeffs, Tcs = self.ais, self.alpha_coeffs, self.Tcs
        if not full:
            a_alphas = []
            for i in self.cmps:
                tau = 1.0 - (T/Tcs[i])**0.5
                if T < Tcs[i]:
                    x0 = horner(alpha_coeffs[i], tau)
                    a_alpha = x0*x0*ais[i]
                else:
                    x = (1.0 + alpha_coeffs[i][-2]*tau)
                    a_alpha = ais[i]*x*x
                a_alphas.append(a_alpha)
            return a_alphas
        else:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [], [], []
            for i in self.cmps:
                a = ais[i]
                Tc = Tcs[i]
                rt = (T/Tc)**0.5
                tau = 1.0 - rt
                if T < Tc:
                    x0, x1, x2 = horner_and_der2(alpha_coeffs[i], tau)
                    a_alpha = x0*x0*a
                    da_alpha_dT = -a*(rt*x0*x1/T)
                    d2a_alpha_dT2 = a*((x0*x2/Tc + x1*x1/Tc + rt*x0*x1/T)/(2.0*T))
                else:
                    c1 = alpha_coeffs[i][-2]
                    x0 = 1.0/T
                    x1 = 1.0/Tc
                    x2 = rt#sqrt(T*x1)
                    x3 = c1*(x2 - 1.0) - 1.0
                    x4 = x0*x2*x3
                    a_alpha = a*x3*x3
                    da_alpha_dT = a*c1*x4
                    d2a_alpha_dT2 = a*0.5*c1*x0*(c1*x1 - x4)
                a_alphas.append(a_alpha)
                da_alpha_dTs.append(da_alpha_dT)
                d2a_alpha_dT2s.append(d2a_alpha_dT2)
            return a_alphas, da_alpha_dTs, d2a_alpha_dT2s

    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        Tc = self.Tc
        a = self.a
        rt = (T/Tc)**0.5
        tau = 1.0 - rt
        alpha_coeffs = self.alpha_coeffs
#        alpha_coeffs [c3, c2, c1, 1] always
        if not full:
            if T < Tc:
                x0 = horner(alpha_coeffs, tau)
                a_alpha = x0*x0*a
                return a_alpha
            else:
                x = (1.0 + alpha_coeffs[-2]*tau)
                return a*x*x
        else:
            if T < Tc:
                # Do not optimize until unit tests are in place
                x0, x1, x2 = horner_and_der2(alpha_coeffs, tau)
                a_alpha = x0*x0*a
                
                da_alpha_dT = -a*(rt*x0*x1/T)
                d2a_alpha_dT2 = a*((x0*x2/Tc + x1*x1/Tc + rt*x0*x1/T)/(2.0*T))
                return a_alpha, da_alpha_dT, d2a_alpha_dT2
            else:
                c1 = alpha_coeffs[-2]
                x0 = 1.0/T
                x1 = 1.0/Tc
                x2 = rt#sqrt(T*x1)
                x3 = c1*(x2 - 1.0) - 1.0
                x4 = x0*x2*x3
                a_alpha = a*x3*x3
                da_alpha_dT = a*c1*x4
                d2a_alpha_dT2 = 0.5*a*c1*x0*(c1*x1 - x4)
                return a_alpha, da_alpha_dT, d2a_alpha_dT2
                '''
                from sympy import *
                T, Tc, c1 = symbols('T, Tc, c1')
                tau = 1 - sqrt(T/Tc)
                alpha = (1 + c1*tau)**2
                cse([alpha, diff(alpha, T), diff(alpha, T, T)], optimizations='basic')
                '''


class Gibbons_Laughton_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Gibbons and Laughton (1984) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Two coefficients needed.
        
        .. math::
            \alpha = c_{1} \left(\frac{T}{Tc} - 1\right) + c_{2} 
            \left(\sqrt{\frac{T}{Tc}} - 1\right) + 1

        References
        ----------
        .. [1] Gibbons, Richard M., and Andrew P. Laughton. "An Equation of 
           State for Polar and Non-Polar Substances and Mixtures" 80, no. 9 
           (January 1, 1984): 1019-38. doi:10.1039/F29848001019.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(T/Tc - 1) + c2*(sqrt(T/Tc) - 1) + 1)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(c1/Tc + c2*sqrt(T/Tc)/(2*T))
            d2a_alpha_dT2 = a*(-c2*sqrt(T/Tc)/(4*T**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Soave_1984_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1984) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Two coefficients needed.
        
        .. math::
            \alpha = c_{1} \left(- \frac{T}{Tc} + 1\right) + c_{2} \left(-1
            + \frac{Tc}{T}\right) + 1

        References
        ----------
        .. [1] Soave, G. "Improvement of the Van Der Waals Equation of State." 
           Chemical Engineering Science 39, no. 2 (January 1, 1984): 357-69. 
           doi:10.1016/0009-2509(84)80034-2.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-T/Tc + 1) + c2*(-1 + Tc/T) + 1)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-c1/Tc - Tc*c2/T**2)
            d2a_alpha_dT2 = a*(2*Tc*c2/T**3)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
    # "Stryjek-Vera" skipped, doesn't match PRSV or PRSV2


class Yu_Lu_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Yu and Lu (1987) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Four coefficients needed.
        
        .. math::
            \alpha = 10^{c_{4} \left(- \frac{T}{Tc} + 1\right) \left(
            \frac{T^{2} c_{3}}{Tc^{2}} + \frac{T c_{2}}{Tc} + c_{1}\right)}

        References
        ----------
        .. [1] Yu, Jin-Min, and Benjamin C. -Y. Lu. "A Three-Parameter Cubic 
           Equation of State for Asymmetric Mixture Density Calculations." 
           Fluid Phase Equilibria 34, no. 1 (January 1, 1987): 1-19. 
           doi:10.1016/0378-3812(87)85047-1. 
        '''
        c1, c2, c3, c4 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*10**(c4*(-T/Tc + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(10**(c4*(-T/Tc + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1))*(c4*(-T/Tc + 1)*(2*T*c3/Tc**2 + c2/Tc) - c4*(T**2*c3/Tc**2 + T*c2/Tc + c1)/Tc)*log(10))
            d2a_alpha_dT2 = a*(10**(-c4*(T/Tc - 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1))*c4*(-4*T*c3/Tc - 2*c2 - 2*c3*(T/Tc - 1) + c4*(T**2*c3/Tc**2 + T*c2/Tc + c1 + (T/Tc - 1)*(2*T*c3/Tc + c2))**2*log(10))*log(10)/Tc**2)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Trebble_Bishnoi_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Trebble and Bishnoi (1987) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. One coefficient needed.
        
        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{Tc} + 1\right)}

        References
        ----------
        .. [1] Trebble, M. A., and P. R. Bishnoi. "Development of a New Four-
           Parameter Cubic Equation of State." Fluid Phase Equilibria 35, no. 1
           (September 1, 1987): 1-18. doi:10.1016/0378-3812(87)80001-8.
        '''
        c1 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp(c1*(-T/Tc + 1))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*-c1*exp(c1*(-T/Tc + 1))/Tc
            d2a_alpha_dT2 = a*c1**2*exp(-c1*(T/Tc - 1))/Tc**2
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
            
            
class Melhem_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Melhem et al. (1989) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Two coefficients needed.
        
        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{Tc} + 1\right) + c_{2} 
            \left(- \sqrt{\frac{T}{Tc}} + 1\right)^{2}}

        References
        ----------
        .. [1] Melhem, Georges A., Riju Saini, and Bernard M. Goodwin. "A 
           Modified Peng-Robinson Equation of State." Fluid Phase Equilibria 
           47, no. 2 (August 1, 1989): 189-237. 
           doi:10.1016/0378-3812(89)80176-1. 
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((-c1/Tc - c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T)*exp(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2))
            d2a_alpha_dT2 = a*(((c1/Tc - c2*sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T)**2 + c2*(1/Tc - sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T)/(2*T))*exp(-c1*(T/Tc - 1) + c2*(sqrt(T/Tc) - 1)**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Androulakis_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Androulakis et al. (1989) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Three coefficients needed.
        
        .. math::
            \alpha = c_{1} \left(- \left(\frac{T}{Tc}\right)^{\frac{2}{3}}
            + 1\right) + c_{2} \left(- \left(\frac{T}{Tc}\right)^{\frac{2}{3}} 
            + 1\right)^{2} + c_{3} \left(- \left(\frac{T}{Tc}\right)^{
            \frac{2}{3}} + 1\right)^{3} + 1

        References
        ----------
        .. [1] Androulakis, I. P., N. S. Kalospiros, and D. P. Tassios. 
           "Thermophysical Properties of Pure Polar and Nonpolar Compounds with
           a Modified VdW-711 Equation of State." Fluid Phase Equilibria 45, 
           no. 2 (April 1, 1989): 135-63. doi:10.1016/0378-3812(89)80254-7. 
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-(T/Tc)**(2/3) + 1) + c2*(-(T/Tc)**(2/3) + 1)**2 + c3*(-(T/Tc)**(2/3) + 1)**3 + 1)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-2*c1*(T/Tc)**(2/3)/(3*T) - 4*c2*(T/Tc)**(2/3)*(-(T/Tc)**(2/3) + 1)/(3*T) - 2*c3*(T/Tc)**(2/3)*(-(T/Tc)**(2/3) + 1)**2/T)
            d2a_alpha_dT2 = a*(2*(T/Tc)**(2/3)*(c1 + 4*c2*(T/Tc)**(2/3) - 2*c2*((T/Tc)**(2/3) - 1) - 12*c3*(T/Tc)**(2/3)*((T/Tc)**(2/3) - 1) + 3*c3*((T/Tc)**(2/3) - 1)**2)/(9*T**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Schwartzentruber_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Schwartzentruber et al. (1990) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Three coefficients needed.
        
        .. math::
            \alpha = \left(c_{4} \left(- \sqrt{\frac{T}{Tc}} + 1\right) 
            - \left(- \sqrt{\frac{T}{Tc}} + 1\right) \left(\frac{T^{2} c_{3}}
            {Tc^{2}} + \frac{T c_{2}}{Tc} + c_{1}\right) + 1\right)^{2}

        References
        ----------
        .. [1] J. Schwartzentruber, H. Renon, and S. Watanasiri, "K-values for 
           Non-Ideal Systems:An Easier Way," Chem. Eng., March 1990, 118-124.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*((c4*(-sqrt(T/Tc) + 1) - (-sqrt(T/Tc) + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1) + 1)**2)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((c4*(-sqrt(T/Tc) + 1) - (-sqrt(T/Tc) + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1) + 1)*(-2*(-sqrt(T/Tc) + 1)*(2*T*c3/Tc**2 + c2/Tc) - c4*sqrt(T/Tc)/T + sqrt(T/Tc)*(T**2*c3/Tc**2 + T*c2/Tc + c1)/T))
            d2a_alpha_dT2 = a*(((-c4*(sqrt(T/Tc) - 1) + (sqrt(T/Tc) - 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1) + 1)*(8*c3*(sqrt(T/Tc) - 1)/Tc**2 + 4*sqrt(T/Tc)*(2*T*c3/Tc + c2)/(T*Tc) + c4*sqrt(T/Tc)/T**2 - sqrt(T/Tc)*(T**2*c3/Tc**2 + T*c2/Tc + c1)/T**2) + (2*(sqrt(T/Tc) - 1)*(2*T*c3/Tc + c2)/Tc - c4*sqrt(T/Tc)/T + sqrt(T/Tc)*(T**2*c3/Tc**2 + T*c2/Tc + c1)/T)**2)/2)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Almeida_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Almeida et al. (1991) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Three coefficients needed.
        
        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{Tc} + 1\right) \left|{
            \frac{T}{Tc} - 1}\right|^{c_{2} - 1} + c_{3} \left(-1 
            + \frac{Tc}{T}\right)}

        References
        ----------
        .. [1] Almeida, G. S., M. Aznar, and A. S. Telles. "Uma Nova Forma de 
           Dependência Com a Temperatura Do Termo Atrativo de Equações de 
           Estado Cúbicas." RBE, Rev. Bras. Eng., Cad. Eng. Quim 8 (1991): 95.
        '''
        # Note: For the second derivative, requires the use a CAS which can 
        # handle the assumption that Tr-1 != 0.
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp(c1*(-T/Tc + 1)*abs(T/Tc - 1)**(c2 - 1) + c3*(-1 + Tc/T))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((c1*(c2 - 1)*(-T/Tc + 1)*abs(T/Tc - 1)**(c2 - 1)*copysign(1, T/Tc - 1)/(Tc*Abs(T/Tc - 1)) - c1*abs(T/Tc - 1)**(c2 - 1)/Tc - Tc*c3/T**2)*exp(c1*(-T/Tc + 1)*abs(T/Tc - 1)**(c2 - 1) + c3*(-1 + Tc/T)))
            d2a_alpha_dT2 = a*exp(c3*(Tc/T - 1) - c1*abs(T/Tc - 1)**(c2 - 1)*(T/Tc - 1))*((c1*abs(T/Tc - 1)**(c2 - 1))/Tc + (Tc*c3)/T**2 + (c1*abs(T/Tc - 1)**(c2 - 2)*copysign(1, T/Tc - 1)*(c2 - 1)*(T/Tc - 1))/Tc)**2 - exp(c3*(Tc/T - 1) - c1*abs(T/Tc - 1)**(c2 - 1)*(T/Tc - 1))*((2*c1*abs(T/Tc - 1)**(c2 - 2)*copysign(1, T/Tc - 1)*(c2 - 1))/Tc**2 - (2*Tc*c3)/T**3 + (c1*abs(T/Tc - 1)**(c2 - 3)*copysign(1, T/Tc - 1)**2*(c2 - 1)*(c2 - 2)*(T/Tc - 1))/Tc**2)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Twu91_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Twu et al. (1991) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Three coefficients needed.
        
        .. math::
            \alpha = \left(\frac{T}{Tc}\right)^{c_{3} \left(c_{2} 
            - 1\right)} e^{c_{1} \left(- \left(\frac{T}{Tc}
            \right)^{c_{2} c_{3}} + 1\right)}

        References
        ----------
        .. [1] Twu, Chorng H., David Bluck, John R. Cunningham, and John E. 
           Coon. "A Cubic Equation of State with a New Alpha Function and a 
           New Mixing Rule." Fluid Phase Equilibria 69 (December 10, 1991): 
           33-50. doi:10.1016/0378-3812(91)90024-2.
        '''
        c0, c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        Tr = T/Tc
        if quick:
            if not full:
                a_alpha = a*(Tr**(c2*(c1 - 1.0))*exp(c0*(1.0 - (Tr)**(c1*c2))))
                return a_alpha
            else:
                T_inv = 1.0/T
                x1 = c1 - 1.0
                x2 = c2*x1
                x3 = c1*c2
                x4 = Tr**x3
                x5 = a*Tr**x2*exp(-c0*(x4 - 1.0))
                x6 = c0*x4
                x7 = c1*x6
                x8 = c2*x5
                x9 = c1*c1*c2
                d2a_alpha_dT2 = (x8*(c0*c0*x4*x4*x9 - c1 + c2*x1*x1 
                                     - 2.0*x2*x7 - x6*x9 + x7 + 1.0)*T_inv*T_inv)            
                return x5, x8*(x1 - x7)*T_inv, d2a_alpha_dT2
        else:
            a_alpha = a*(Tr**(c2*(c1 - 1.0))*exp(c0*(1.0 - (Tr)**(c1*c2))))
            if not full:
                return a_alpha
            else:
                da_alpha_dT = a*(-c0*c1*c2*(T/Tc)**(c1*c2)*(T/Tc)**(c2*(c1 - 1))*exp(c0*(-(T/Tc)**(c1*c2) + 1))/T + c2*(T/Tc)**(c2*(c1 - 1))*(c1 - 1)*exp(c0*(-(T/Tc)**(c1*c2) + 1))/T)
                d2a_alpha_dT2 = a*(c2*(T/Tc)**(c2*(c1 - 1))*(c0**2*c1**2*c2*(T/Tc)**(2*c1*c2) - c0*c1**2*c2*(T/Tc)**(c1*c2) - 2*c0*c1*c2*(T/Tc)**(c1*c2)*(c1 - 1) + c0*c1*(T/Tc)**(c1*c2) - c1 + c2*(c1 - 1)**2 + 1)*exp(-c0*((T/Tc)**(c1*c2) - 1))/T**2)
                return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_and_derivatives_vectorized(self, T, full=False):
        r'''Method to calculate the pure-component `a_alphas` and their first  
        and second derivatives for TWU91 alpha function EOS. This vectorized 
        implementation is added for extra speed.

        .. math::
            \alpha = \left(\frac{T}{Tc}\right)^{c_{3} \left(c_{2} 
            - 1\right)} e^{c_{1} \left(- \left(\frac{T}{Tc}
            \right)^{c_{2} c_{3}} + 1\right)}        
        
        Parameters
        ----------
        T : float
            Temperature, [K]
        full : bool, optional
            If False, calculates and returns only `a_alphas`, [-]
        
        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dTs : list[float]
            Temperature derivative of coefficient calculated by EOS-specific 
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2s : list[float]
            Second temperature derivative of coefficient calculated by  
            EOS-specific method, [J^2/mol^2/Pa/K**2]
        '''
        ais, alpha_coeffs, Tcs = self.ais, self.alpha_coeffs, self.Tcs
        if not full:
            a_alphas = []
            for i in self.cmps:
                coeffs = alpha_coeffs[i] 
                Tr = T/Tcs[i]
                a_alpha = ais[i]*(Tr**(coeffs[2]*(coeffs[1] - 1.0))*exp(coeffs[0]*(1.0 - (Tr)**(coeffs[1]*coeffs[2]))))
                a_alphas.append(a_alpha)
            return a_alphas
        else:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [], [], []
            T_inv = 1.0/T
            for i in self.cmps:
                coeffs = alpha_coeffs[i]
                c0, c1, c2 = coeffs[0], coeffs[1], coeffs[2]
                Tr = T/Tcs[i]
                
                x1 = c1 - 1.0
                x2 = c2*x1
                x3 = c1*c2
                x4 = Tr**x3
                x5 = ais[i]*Tr**x2*exp(-c0*(x4 - 1.0))
                x6 = c0*x4
                x7 = c1*x6
                x8 = c2*x5
                x9 = c1*c1*c2
                
                d2a_alpha_dT2 = (x8*(c0*c0*x4*x4*x9 - c1 + c2*x1*x1 
                                     - 2.0*x2*x7 - x6*x9 + x7 + 1.0)*T_inv*T_inv)            
                a_alphas.append(x5)
                da_alpha_dTs.append(x8*(x1 - x7)*T_inv)
                d2a_alpha_dT2s.append(d2a_alpha_dT2)

            return a_alphas, da_alpha_dTs, d2a_alpha_dT2s


class Soave_93_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1983) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Two coefficient needed.
        
        .. math::
            \alpha = c_{1} \left(- \frac{T}{Tc} + 1\right) + c_{2} 
            \left(- \sqrt{\frac{T}{Tc}} + 1\right)^{2} + 1

        References
        ----------
        .. [1] Soave, G. "Improving the Treatment of Heavy Hydrocarbons by the 
           SRK EOS." Fluid Phase Equilibria 84 (April 1, 1993): 339-42. 
           doi:10.1016/0378-3812(93)85131-5.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2 + 1)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-c1/Tc - c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T)
            d2a_alpha_dT2 = a*(c2*(1/Tc - sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T)/(2*T))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Gasem_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Gasem (2001) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Three coefficients needed.
        
        .. math::
            \alpha = e^{\left(- \left(\frac{T}{Tc}\right)^{c_{3}} + 1\right) 
            \left(\frac{T c_{2}}{Tc} + c_{1}\right)}

        References
        ----------
        .. [1] Gasem, K. A. M, W Gao, Z Pan, and R. L Robinson Jr. "A Modified 
           Temperature Dependence for the Peng-Robinson Equation of State." 
           Fluid Phase Equilibria 181, no. 1–2 (May 25, 2001): 113-25. 
           doi:10.1016/S0378-3812(01)00488-5.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(exp((-(T/Tc)**c3 + 1)*(T*c2/Tc + c1)))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((c2*(-(T/Tc)**c3 + 1)/Tc - c3*(T/Tc)**c3*(T*c2/Tc + c1)/T)*exp((-(T/Tc)**c3 + 1)*(T*c2/Tc + c1)))
            d2a_alpha_dT2 = a*(((c2*((T/Tc)**c3 - 1)/Tc + c3*(T/Tc)**c3*(T*c2/Tc + c1)/T)**2 - c3*(T/Tc)**c3*(2*c2/Tc + c3*(T*c2/Tc + c1)/T - (T*c2/Tc + c1)/T)/T)*exp(-((T/Tc)**c3 - 1)*(T*c2/Tc + c1)))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Coquelet_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Coquelet et al. (2004) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Three coefficients needed.
        
        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{Tc} + 1\right) \left(c_{2} 
            \left(- \sqrt{\frac{T}{Tc}} + 1\right)^{2} + c_{3} 
            \left(- \sqrt{\frac{T}{Tc}} + 1\right)^{3} + 1\right)^{2}}

        References
        ----------
        .. [1] Coquelet, C., A. Chapoy, and D. Richon. "Development of a New 
           Alpha Function for the Peng–Robinson Equation of State: Comparative 
           Study of Alpha Function Models for Pure Gases (Natural Gas 
           Components) and Water-Gas Systems." International Journal of 
           Thermophysics 25, no. 1 (January 1, 2004): 133-58. 
           doi:10.1023/B:IJOT.0000022331.46865.2f.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(exp(c1*(-T/Tc + 1)*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((c1*(-T/Tc + 1)*(-2*c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T - 3*c3*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)**2/T)*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1) - c1*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2/Tc)*exp(c1*(-T/Tc + 1)*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2))
            d2a_alpha_dT2 = a*(c1*(c1*(-(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)/Tc + sqrt(T/Tc)*(-2*c2 + 3*c3*(sqrt(T/Tc) - 1))*(sqrt(T/Tc) - 1)*(T/Tc - 1)/T)**2*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)**2 - ((T/Tc - 1)*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)*(2*c2/Tc - 6*c3*(sqrt(T/Tc) - 1)/Tc - 2*c2*sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T + 3*c3*sqrt(T/Tc)*(sqrt(T/Tc) - 1)**2/T) + 4*sqrt(T/Tc)*(2*c2 - 3*c3*(sqrt(T/Tc) - 1))*(sqrt(T/Tc) - 1)*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)/Tc + (2*c2 - 3*c3*(sqrt(T/Tc) - 1))**2*(sqrt(T/Tc) - 1)**2*(T/Tc - 1)/Tc)/(2*T))*exp(-c1*(T/Tc - 1)*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Haghtalab_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Haghtalab et al. (2010) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Three coefficients needed.
        
        .. math::
            \alpha = e^{\left(- c_{3}^{\log{\left (\frac{T}{Tc} \right )}} 
            + 1\right) \left(- \frac{T c_{2}}{Tc} + c_{1}\right)}

        References
        ----------
        .. [1] Haghtalab, A., M. J. Kamali, S. H. Mazloumi, and P. Mahmoodi. 
           "A New Three-Parameter Cubic Equation of State for Calculation 
           Physical Properties and Vapor-liquid Equilibria." Fluid Phase 
           Equilibria 293, no. 2 (June 25, 2010): 209-18. 
           doi:10.1016/j.fluid.2010.03.029.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp((-c3**log(T/Tc) + 1)*(-T*c2/Tc + c1))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((-c2*(-c3**log(T/Tc) + 1)/Tc - c3**log(T/Tc)*(-T*c2/Tc + c1)*log(c3)/T)*exp((-c3**log(T/Tc) + 1)*(-T*c2/Tc + c1)))
            d2a_alpha_dT2 = a*(((c2*(c3**log(T/Tc) - 1)/Tc + c3**log(T/Tc)*(T*c2/Tc - c1)*log(c3)/T)**2 + c3**log(T/Tc)*(2*c2/Tc + (T*c2/Tc - c1)*log(c3)/T - (T*c2/Tc - c1)/T)*log(c3)/T)*exp((c3**log(T/Tc) - 1)*(T*c2/Tc - c1)))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Saffari_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Saffari and Zahedi (2013) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Three coefficients needed.
        
        .. math::
            \alpha = e^{\frac{T c_{1}}{Tc} + c_{2} \log{\left (\frac{T}{Tc} 
            \right )} + c_{3} \left(- \sqrt{\frac{T}{Tc}} + 1\right)}

        References
        ----------
        .. [1] Saffari, Hamid, and Alireza Zahedi. "A New Alpha-Function for 
           the Peng-Robinson Equation of State: Application to Natural Gas." 
           Chinese Journal of Chemical Engineering 21, no. 10 (October 1, 
           2013): 1155-61. doi:10.1016/S1004-9541(13)60581-9.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(exp(T*c1/Tc + c2*log(T/Tc) + c3*(-sqrt(T/Tc) + 1)))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((c1/Tc + c2/T - c3*sqrt(T/Tc)/(2*T))*exp(T*c1/Tc + c2*log(T/Tc) + c3*(-sqrt(T/Tc) + 1)))
            d2a_alpha_dT2 = a*(((2*c1/Tc + 2*c2/T - c3*sqrt(T/Tc)/T)**2 - (4*c2 - c3*sqrt(T/Tc))/T**2)*exp(T*c1/Tc + c2*log(T/Tc) - c3*(sqrt(T/Tc) - 1))/4)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Chen_Yang_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Hamid and Yang (2017) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Seven coefficients needed.
        
        .. math::
            \alpha = e^{\left(- c_{3}^{\log{\left (\frac{T}{Tc} \right )}} 
            + 1\right) \left(- \frac{T c_{2}}{Tc} + c_{1}\right)}

        References
        ----------
        .. [1] Chen, Zehua, and Daoyong Yang. "Optimization of the Reduced 
           Temperature Associated with Peng–Robinson Equation of State and 
           Soave-Redlich-Kwong Equation of State To Improve Vapor Pressure 
           Prediction for Heavy Hydrocarbon Compounds." Journal of Chemical & 
           Engineering Data, August 31, 2017. doi:10.1021/acs.jced.7b00496.
        '''
        c1, c2, c3, c4, c5, c6, c7 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp(c4*log((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)**2 + (-T/Tc + 1)*(c1 + c2*omega + c3*omega**2))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-(c1 + c2*omega + c3*omega**2)/Tc - c4*sqrt(T/Tc)*(c5 + c6*omega + c7*omega**2)*log((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)/(T*((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)))*exp(c4*log((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)**2 + (-T/Tc + 1)*(c1 + c2*omega + c3*omega**2))
            d2a_alpha_dT2 = a*(((c1 + c2*omega + c3*omega**2)/Tc - c4*sqrt(T/Tc)*(c5 + c6*omega + c7*omega**2)*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)/(T*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)))**2 - c4*(c5 + c6*omega + c7*omega**2)*((c5 + c6*omega + c7*omega**2)*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)/(Tc*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)) - (c5 + c6*omega + c7*omega**2)/(Tc*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)) + sqrt(T/Tc)*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)/T)/(2*T*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)))*exp(c4*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)**2 - (T/Tc - 1)*(c1 + c2*omega + c3*omega**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
        
        

class TwuSRK95_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `Tc`, `omega`, and `a`.
        
        Because of its similarity for the TWUPR EOS, this has been moved to an 
        external `TWU_a_alpha_common` function. See it for further 
        documentation.
        '''
        return TWU_a_alpha_common(T, self.Tc, self.omega, self.a, full=full, quick=quick, method='SRK')

    def a_alpha_and_derivatives_vectorized(self, T, full=False):
        Tcs, omegas, ais = self.Tcs, self.omegas, self.ais
        if not full:
            return [TWU_a_alpha_common(T, Tcs[i], omegas[i], ais[i], full=full, quick=True, method='SRK')
                    for i in self.cmps]
        else:
            r0, r1, r2 = [], [], []
            for i in self.cmps:
                v0, v1, v2 = TWU_a_alpha_common(T, Tcs[i], omegas[i], ais[i], full=full, quick=True, method='SRK')
                r0.append(v0)
                r1.append(v1)
                r2.append(v2)
            return r0, r1, r2
        


class TwuPR95_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `Tc`, `omega`, and `a`.
        
        Because of its similarity for the TWUSRK EOS, this has been moved to an 
        external `TWU_a_alpha_common` function. See it for further 
        documentation.
        '''
        return TWU_a_alpha_common(T, self.Tc, self.omega, self.a, full=full, quick=quick, method='PR')

    def a_alpha_and_derivatives_vectorized(self, T, full=False):
        Tcs, omegas, ais = self.Tcs, self.omegas, self.ais
        if not full:
            return [TWU_a_alpha_common(T, Tcs[i], omegas[i], ais[i], full=full, quick=True, method='PR')
                    for i in self.cmps]
        else:
            r0, r1, r2 = [], [], []
            for i in self.cmps:
                v0, v1, v2 = TWU_a_alpha_common(T, Tcs[i], omegas[i], ais[i], full=full, quick=True, method='PR')
                r0.append(v0)
                r1.append(v1)
                r2.append(v2)
            return r0, r1, r2
        
        
class Soave_79_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1979) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. Three coefficients are needed.
        
        .. math::
            \alpha = 1 + (1 - T_r)(M + \frac{N}{T_r})

        References
        ----------
        .. [1] Soave, G. "Rigorous and Simplified Procedures for Determining 
           the Pure-Component Parameters in the Redlich—Kwong—Soave Equation of
           State." Chemical Engineering Science 35, no. 8 (January 1, 1980): 
           1725-30. https://doi.org/10.1016/0009-2509(80)85007-X.
        '''
        M, N = self.alpha_coeffs#self.M, self.N
        Tc, a = self.Tc, self.a
        if not full:
            Tr = T/Tc
            return a*(1.0 + (1.0 - Tr)*(M + N/Tr))
        else:
            T_inv = 1.0/T
            x0 = Tc_inv = 1.0/Tc
            x1 = T*x0 - 1.0
            x2 = Tc*T_inv
            x3 = M + N*x2
            x4 = N*T_inv*T_inv
            return (a*(1.0 - x1*x3), a*(Tc*x1*x4 - x0*x3), a*(2.0*x4*(1.0 - x1*x2)))

    def a_alpha_and_derivatives_vectorized(self, T, full=False):
        ais, alpha_coeffs, Tcs = self.ais, self.alpha_coeffs, self.Tcs
        if not full:
            a_alphas = []
            for i in self.cmps:
                Tr = T/Tcs[i]
                M, N = alpha_coeffs[i]
                a_alphas.append(a*(1.0 + (1.0 - Tr)*(M + N/Tr)))
            return a_alphas
        else:
            T_inv = 1.0/T
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [], [], []
            for i in self.cmps:
                a = ais[i]
                M, N = alpha_coeffs[i]
                x0 = Tc_inv = 1.0/Tcs[i]
                x1 = T*x0 - 1.0
                x2 = Tc*T_inv
                x3 = M + N*x2
                x4 = N*T_inv*T_inv

                a_alphas.append(a*(1.0 - x1*x3))
                da_alpha_dTs.append(a*(Tc*x1*x4 - x0*x3))
                d2a_alpha_dT2s.append(a*(2.0*x4*(1.0 - x1*x2)))
            return a_alphas, da_alpha_dTs, d2a_alpha_dT2s


a_alpha_bases = [Soave_1972_a_alpha, Heyen_a_alpha, Harmens_Knapp_a_alpha, Mathias_1983_a_alpha,
                 Mathias_Copeman_untruncated_a_alpha, Gibbons_Laughton_a_alpha, Soave_1984_a_alpha, Yu_Lu_a_alpha,
                 Trebble_Bishnoi_a_alpha, Melhem_a_alpha, Androulakis_a_alpha, Schwartzentruber_a_alpha,
                 Almeida_a_alpha, Twu91_a_alpha, Soave_93_a_alpha, Gasem_a_alpha, 
                 Coquelet_a_alpha, Haghtalab_a_alpha, Saffari_a_alpha, Chen_Yang_a_alpha,
                 Mathias_Copeman_a_alpha,
                 TwuSRK95_a_alpha, TwuPR95_a_alpha, Soave_79_a_alpha]


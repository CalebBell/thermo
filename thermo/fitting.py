# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['alpha_Twu91_objf', 'Twu91_check_params', 'postproc_lmfit']

from cmath import atanh as catanh
from fluids.numerics import (chebval, brenth, third, sixth, roots_cubic,
                             roots_cubic_a1, numpy as np, py_newton as newton,
                             py_bisect as bisect, inf, polyder, chebder, 
                             trunc_exp, secant, linspace, logspace,
                             horner, horner_and_der2)
from thermo.utils import R


def Twu91_check_params(params):
    '''Returns True if all constraints are met and the values are OK; false if
    they are not OK.
    '''
    # A consistency test for a-functions of cubic equations of state
    # From On the imperative need to use a consistent α-function for the
    # prediction of pure-compound supercritical properties with a cubic equation of state
    try:
        c0, c1, c2 = params['c0'].value, params['c1'].value, params['c2'].value
    except:
        c0, c1, c2 = params
        
    delta = c2*(c1 - 1.0)
    gamma = c1*c2
    if (-delta) < 0.0:
        return False
    if (c0*gamma) < 0.0:
        return False
    X = -3.0*(gamma + delta - 1.0)
    Y = gamma*gamma + 3.0*gamma*delta - 3.0*gamma + 3.0*delta*delta - 6.0*delta + 2.0
    Z = -delta*(delta*delta - 3.0*delta + 2.0)    
    cond_1 = gamma <= (1.0 - delta)
    cond_2a = (1.0 - 2.0*delta + 2.0*(delta*(delta - 1.0))**0.5 - gamma) >= 0.0
    cond_2b = (4.0*Y**3 + 4.0*Z*X**3 + 27.0*Z*Z - 18.0*X*Y*Z - X*X*Y*Y) >= 0.0
    cond_2 = cond_2a and cond_2b
    return cond_1 or cond_2


def alpha_Twu91_objf(params, Trs, alphas_over_a):
    try:
        c0, c1, c2 = params['c0'].value, params['c1'].value, params['c2'].value
    except:
        c0, c1, c2 = params

    alphas_calc = Trs**(c2*(c1 - 1.0)) * np.exp(c0*(1.0 - Trs**(c1*c2)))
    return alphas_calc - alphas_over_a


def postproc_lmfit(result):
    result.aard = np.abs(result.residual).sum()/result.ndata
    return result


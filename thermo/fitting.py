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

__all__ = ['alpha_Twu91_objf', 'alpha_Twu91_objfc',
           'Twu91_check_params', 'postproc_lmfit',
           'alpha_poly_objf', 'alpha_poly_objfc', 'poly_check_params']

from cmath import atanh as catanh
from fluids.numerics import (chebval, brenth, third, sixth, roots_cubic,
                             roots_cubic_a1, numpy as np, py_newton as newton,
                             py_bisect as bisect, inf, polyder, chebder, 
                             trunc_exp, secant, linspace, logspace,
                             horner, horner_and_der2, horner_and_der3,
                             is_poly_positive, is_poly_negative)
from thermo.utils import R

# Supported methods of lmfit
methods_uncons = ['leastsq', 'least_squares', 'nelder', 'lbfgsb', 'powell', 
                  'cg', 'cobyla', 'bfgs', 'tnc', 'slsqp', 'ampgo']
methods_cons = ['differential_evolution', 'brute', 'dual_annealing', 'shgo', 
                'basinhopping', 'trust-constr', 'emcee']



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


def alpha_Twu91_objfc(params, Trs, alphas_over_a, debug=False):
    try:
        c0, c1, c2 = params['c0'].value, params['c1'].value, params['c2'].value
    except:
        c0, c1, c2 = params

    x0 = c1 - 1.0
    x1 = c2*x0
    x2 = c1*c2
    x3 = Trs**x2
    x4 = Trs**x1*np.exp(-c0*(x3 - 1.0))
    x5 = c0*x3
    x6 = c1*x5
    x7 = c2*x4
    x8 = x1 - 1.0
    x9 = x6*(x2*x5 - x2 + 1.0)
    x10 = 3.0*x1
    x11 = c2*c2
    x12 = 3.0*x2
    x13 = c1*c1*x11
    
    Trs_inv = 1.0/Trs
    Trs_inv2 = Trs_inv*Trs_inv
    Trs_inv3 = Trs_inv*Trs_inv2

    alphas_calc = d0 = x4
    d1 = x7*(x0 - x6)*Trs_inv
    d2 = x7*(x0*x8 - 2.0*x1*x6 + x9)*Trs_inv2
    d3 = (-x7*(-x0*(x0*x0*x11 - x10 + 2.0) + x10*x6*x8 - x10*x9 
               + x6*(Trs**(2.0*x2)*c0*c0*x13 + x12*x5 - x12 - 3.0*x13*x5 + x13 + 2.0)
               )*Trs_inv3)
    
    err = np.abs(alphas_calc - alphas_over_a)
    
    valid = Twu91_check_params(params)
    err = alpha_constrain_err(err, d0, d1, d2, d3, valid=valid, debug=debug)
    return err

def poly_check_params(coeffs, domain=None):
    if domain is None:
        domain = (0, 1)
        
    if not is_poly_positive(coeffs, domain):
        return False
    coeffs_d1 = polyder(coeffs[::-1])[::-1]
    coeffs_d2 = polyder(coeffs_d1[::-1])[::-1]
    coeffs_d3 = polyder(coeffs_d2[::-1])[::-1]
    
    if not is_poly_negative(coeffs_d1, domain):
        return False
    if not is_poly_positive(coeffs_d2, domain):
        return False
    if not is_poly_negative(coeffs_d3, domain):
        return False
    return True
    
def alpha_poly_objf(params, Trs, alphas_over_a, domain=None):
    try:
        coeffs = [i.value for i in params.values()] # already sorted
    except:
        coeffs = params
    v, d1, d2, d3 = horner_and_der3(coeffs, Trs)
    err = v - alphas_over_a
    return err

def alpha_poly_objfc(params, Trs, alphas_over_a, domain=None, debug=False):
    try:
        coeffs = [i.value for i in params.values()] # already sorted
    except:
        coeffs = params
    v, d1, d2, d3 = horner_and_der3(coeffs, Trs)
    err = np.abs(v - alphas_over_a)
    
    valid = poly_check_params(coeffs, domain=domain)
    err = alpha_constrain_err(err, v, d1, d2, d3, valid=valid, debug=debug)
    
    return err



def alpha_constrain_err(err, d0, d1, d2, d3, valid=True, debug=False):
    N = len(d0)
    zeros = np.zeros(N)
    
    d0 = np.min([d0, zeros], axis=0)
    del_0 = np.abs(d0*100)
    
    d1 = np.max([d1, zeros], axis=0)
    del_1 = np.abs(d1*1000)

    d2 = np.min([d2, zeros], axis=0)
    del_2 = np.abs(d2*5000)

    d3 = np.max([d3, zeros], axis=0)
    del_3 = np.abs(d3*500000000)
    
    tot = np.sum(del_0) + np.sum(del_1) + np.sum(del_2) + np.sum(del_3)
    if tot == 0 and not valid:
#        if not Twu91_check_params(params):
        del_0 = 1e5*np.ones(N)
    
    if debug:
        tot = np.sum(err) + tot
        print(tot, err, del_0, del_1, del_2, del_3)
         
    err += del_0
    err += del_1
    err += del_2
    err += del_3
    return err

def postproc_lmfit(result):
    result.aard = np.abs(result.residual).sum()/result.ndata
    return result


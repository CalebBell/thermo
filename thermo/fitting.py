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

__all__ = ['alpha_Twu91_objf', 'alpha_Twu91_objfc', 'fit_function',
           'Twu91_check_params', 'postproc_lmfit',
           'alpha_poly_objf', 'alpha_poly_objfc', 'poly_check_params',
           'fit_cheb_poly', 'poly_fit_statistics', 'fit_cheb_poly_auto',
           'data_fit_statistics']

from fluids.numerics import (chebval, brenth, third, sixth, roots_cubic,
                             roots_cubic_a1, numpy as np, newton,
                             bisect, inf, polyder, chebder, std, min_max_ratios,
                             trunc_exp, secant, linspace, logspace,
                             horner, horner_and_der2, horner_and_der3,
                             is_poly_positive, is_poly_negative,
                             max_abs_error, max_abs_rel_error, max_squared_error, 
                             max_squared_rel_error, mean_abs_error, mean_abs_rel_error, 
                             mean_squared_error, mean_squared_rel_error)
from fluids.constants import R
try:
    from numpy.polynomial.chebyshev import poly2cheb
    from numpy.polynomial.chebyshev import cheb2poly
    from numpy.polynomial.polynomial import Polynomial
except:
    pass

ChebTools = None
def fit_cheb_poly(func, low, high, n,
                  interpolation_property=None, interpolation_property_inv=None,
                  interpolation_x=lambda x: x, interpolation_x_inv=lambda x: x,
                  arg_func=None):
    r'''Fit a function of one variable to a polynomial of degree `n` using the
    Chebyshev approximation technique. Transformations of the base function
    are allowed as lambdas.

    Parameters
    ----------
    func : callable
        Function to fit, [-]
    low : float
        Low limit of fitting range, [-]
    high : float
        High limit of fitting range, [-]
    n : int
        Degree of polynomial fitting, [-]
    interpolation_property : None or callable
        When specified, this callable will transform the output of the function
        before fitting; for example a property like vapor pressure should be
        `interpolation_property=lambda x: log(x)` because it rises
        exponentially. The output of the evaluated polynomial should then have
        the reverse transform applied to it; in this case, `exp`, [-]
    interpolation_property_inv : None or callable
        When specified, this callable reverses `interpolation_property`; it
        must always be provided when `interpolation_property` is set, and it
        must perform the reverse transform, [-]
    interpolation_x : None or callable
        Callable to transform the input variable to fitting. For example,
        enthalpy of vaporization goes from a high value at low temperatures to zero at
        the critical temperature; it is normally hard for a chebyshev series
        to match this, but by setting this to lambda T: log(1. - T/Tc), this
        issue is resolved, [-]
    interpolation_x_inv : None or callable
        Inverse function of `interpolation_x_inv`; must always be provided when
        `interpolation_x` is set, and it must perform the reverse transform,
        [-]
    arg_func : None or callable
        Function which is called with the value of `x` in the original domain,
        and that returns arguments to `func`.

    Returns
    -------
    coeffs : list[float]
        Polynomial coefficients in order for evaluation by `horner`, [-]

    Notes
    -----
    This is powered by Ian Bell's ChebTools.

    '''
    global ChebTools
    if ChebTools is None:
        import ChebTools

    low_orig, high_orig = low, high
    cheb_fun = None
    low, high = interpolation_x(low_orig), interpolation_x(high_orig)
    if arg_func is not None:
        if interpolation_property is not None:
            def func_fun(T):
                arg = interpolation_x_inv(T)
                if arg > high_orig:
                    arg = high_orig
                if arg < low_orig:
                    arg = low_orig
                return interpolation_property(func(arg, *arg_func(arg)))
        else:
            def func_fun(T):
                arg = interpolation_x_inv(T)
                if arg > high_orig:
                    arg = high_orig
                if arg < low_orig:
                    arg = low_orig
                return func(arg, *arg_func(arg))
    else:
        if interpolation_property is not None:
            def func_fun(T):
                arg = interpolation_x_inv(T)
                if arg > high_orig:
                    arg = high_orig
                if arg < low_orig:
                    arg = low_orig
                return interpolation_property(func(arg))
        else:
            def func_fun(T):
                arg = interpolation_x_inv(T)
                if arg > high_orig:
                    arg = high_orig
                if arg < low_orig:
                    arg = low_orig
                return func(arg)
    func_fun = np.vectorize(func_fun)
    if n == 1:
        coeffs = [func_fun(0.5*(low + high)).tolist()]
    else:
        cheb_fun = ChebTools.generate_Chebyshev_expansion(n-1, func_fun, low, high)

        coeffs = cheb_fun.coef()
        coeffs = cheb2poly(coeffs)[::-1].tolist() # Convert to polynomial basis
    # Mix in low high limits to make it a normal polynomial
    if high != low:
        # Handle the case of no transformation, no limits
        my_poly = Polynomial([-0.5*(high + low)*2.0/(high - low), 2.0/(high - low)])
        coeffs = horner(coeffs, my_poly).coef[::-1].tolist()
    return coeffs

def data_fit_statistics(xs, actual_pts, calc_pts):
    pts = len(xs)
    ARDs = [0.0]*pts
    for i in range(pts):
        if actual_pts[i] != 0.0:
            ARDs[i] = abs((calc_pts[i]-actual_pts[i])/actual_pts[i])

    mae = sum(ARDs)/pts
    err_std = std(ARDs)
    min_ratio, max_ratio = min_max_ratios(actual_pts, calc_pts)
    return mae, err_std, min_ratio, max_ratio

    
def poly_fit_statistics(func, coeffs, low, high, pts=200,
                        interpolation_property_inv=None,
                        interpolation_x=lambda x: x,
                        arg_func=None):
    r'''Function to check how accurate a fit function is to a polynomial.

    This function uses the asolute relative error definition.

    Parameters
    ----------
    func : callable
        Function to fit, [-]
    coeffs : list[float]
        Coefficients for calculating the property, [-]
    low : float
        Low limit of fitting range, [-]
    high : float
        High limit of fitting range, [-]
    n : int
        Degree of polynomial fitting, [-]
    interpolation_property_inv : None or callable
        When specified, this callable reverses `interpolation_property`; it
        must always be provided when `interpolation_property` is set, and it
        must perform the reverse transform, [-]
    interpolation_x : None or callable
        Callable to transform the input variable to fitting. For example,
        enthalpy of vaporization goes from a high value at low temperatures to zero at
        the critical temperature; it is normally hard for a chebyshev series
        to match this, but by setting this to lambda T: log(1. - T/Tc), this
        issue is resolved, [-]
    arg_func : None or callable
        Function which is called with the value of `x` in the original domain,
        and that returns arguments to `func`.

    Returns
    -------
    err_avg : float
        Mean error in the evaluated points, [-]
    err_std : float
        Standard deviation of errors in the evaluated points, [-]
    min_ratio : float
        Lowest ratio of calc/actual in any found points, [-]
    max_ratio : float
        Highest ratio of calc/actual in any found points, [-]

    Notes
    -----
    '''

    low_orig, high_orig = low, high
    all_points_orig = linspace(low_orig, high_orig, pts)

    # Get the low, high, and x points in the transformed domain
    low, high = interpolation_x(low_orig), interpolation_x(high_orig)
    all_points = [interpolation_x(v) for v in all_points_orig]

    # Calculate the fit values
    calc_pts = [horner(coeffs, x) for x in all_points]
    if interpolation_property_inv:
        for i in range(pts):
            calc_pts[i] = interpolation_property_inv(calc_pts[i])

    if arg_func is not None:
        actual_pts = [func(v, *arg_func(v)) for v in all_points_orig]
    else:
        actual_pts = [func(v) for v in all_points_orig]

    ARDs = [(abs((i-j)/j) if j != 0 else 0.0) for i, j in zip(calc_pts, actual_pts)]

    err_avg = sum(ARDs)/pts
    err_std = np.std(ARDs)

    actual_pts = np.array(actual_pts)
    calc_pts = np.array(calc_pts)

    max_ratio, min_ratio = max(calc_pts/actual_pts), min(calc_pts/actual_pts)
    return err_avg, err_std, min_ratio, max_ratio

def select_index_from_stats(stats, ns):
    lowest_err_avg, lowest_err_std, lowest_err = 1e100, 1e100, 1e100
    lowest_err_avg_idx, lowest_err_std_idx, lowest_err_idx = None, None, None
    for i, ((err_avg, err_std, min_ratio, max_ratio),n) in enumerate(zip(stats, ns)):
        if err_avg < lowest_err_avg:
            lowest_err_avg = err_avg
            lowest_err_avg_idx = i
        if err_std < lowest_err_std:
            lowest_err_std = err_std
            lowest_err_std_idx = i
        lowest_err_here = max(abs(1.0 - min_ratio), abs(1.0 - max_ratio))
        if lowest_err_here < lowest_err:
            lowest_err = lowest_err_here
            lowest_err_idx = i
    if lowest_err_avg_idx == lowest_err_std_idx == lowest_err_idx:
        return lowest_err_avg_idx
    elif lowest_err_avg_idx == lowest_err_std_idx:
        return lowest_err_avg_idx
    elif lowest_err_idx == lowest_err_std_idx:
        return lowest_err_idx
    else:
        return lowest_err_avg_idx

def fit_many_cheb_poly(func, low, high, ns, eval_pts=30,
                  interpolation_property=None, interpolation_property_inv=None,
                  interpolation_x=lambda x: x, interpolation_x_inv=lambda x: x,
                  arg_func=None):

    def a_fit(n, eval_pts=eval_pts):
        coeffs = fit_cheb_poly(func, low, high, n,
                  interpolation_property=interpolation_property, interpolation_property_inv=interpolation_property_inv,
                  interpolation_x=interpolation_x, interpolation_x_inv=interpolation_x_inv,
                  arg_func=arg_func)
        err_avg, err_std, min_ratio, max_ratio = poly_fit_statistics(func, coeffs, low, high, pts=eval_pts,
                                interpolation_property_inv=interpolation_property_inv,
                                interpolation_x=interpolation_x,
                                arg_func=arg_func)
        return coeffs, err_avg, err_std, min_ratio, max_ratio

    worked_ns, worked_coeffs, worked_stats = [], [], []

    for n in ns:
        try:
            coeffs, err_avg, err_std, min_ratio, max_ratio = a_fit(n)
            worked_ns.append(n)
            worked_coeffs.append(coeffs)
            worked_stats.append((err_avg, err_std, min_ratio, max_ratio))
        except:
            pass
    return worked_ns, worked_coeffs, worked_stats


def fit_cheb_poly_auto(func, low, high, start_n=3, max_n=20, eval_pts=100,
                  interpolation_property=None, interpolation_property_inv=None,
                  interpolation_x=lambda x: x, interpolation_x_inv=lambda x: x,
                  arg_func=None):
    worked_ns, worked_coeffs, worked_stats = fit_many_cheb_poly(func, low, high, ns=range(start_n, max_n+1),
                  interpolation_property=interpolation_property, interpolation_property_inv=interpolation_property_inv,
                  interpolation_x=interpolation_x, interpolation_x_inv=interpolation_x_inv,
                  arg_func=arg_func, eval_pts=eval_pts)
    idx = select_index_from_stats(worked_stats, worked_ns)

    return worked_ns[idx], worked_coeffs[idx], worked_stats[idx]


# Supported methods of lmfit
methods_uncons = ['leastsq', 'least_squares', 'nelder', 'lbfgsb', 'powell',
                  'cg', 'cobyla', 'bfgs', 'tnc', 'slsqp', 'ampgo']
methods_cons = ['differential_evolution', 'brute', 'dual_annealing', 'shgo',
                'basinhopping', 'trust-constr', 'emcee']

methods_uncons_good = ['leastsq', 'least_squares', 'nelder', 'lbfgsb', 'powell',
                       'cg',
#                        'cobyla', 'slsqp',
                       'bfgs', 'tnc',]
methods_cons_good = ['differential_evolution',
#                      'trust-constr'
                    ]


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


def fit_function(fun, x0=None, args=None, check_fun=None, debug=False,
        cons_meth=methods_cons_good, uncons_meth=methods_uncons_good):
    from lmfit import Parameter, Parameters, minimize
    best_result = None
    best_coeffs = None
    best_err = 1e300

    N = len(x0)

    for restart in (False, True):
        if restart:
            if debug:
                print('-'*10 + 'RESTARTING WITH BEST SOLUTION' + '-'*10)
            best_coeffs_restart = best_coeffs
        for method_list, bounded in zip([uncons_meth, cons_meth], [False, True]):
            for method in method_list:
                fit_params = Parameters() # Ordered dict
                for i in range(N):
                    if restart:
                        v = best_coeffs_restart[i]
                    else:
                        try:
                            v = x0[-i]
                        except:
                            v = 1e-20
                    if bounded:
                        if restart:
                            l, h = -abs(best_coeffs_restart[i])*1e3, abs(best_coeffs_restart[i])*1e3
                        else:
                            l, h = -1e5, 1e5
                        fit_params['c' + str(i)] = Parameter(value=v, min=l, max=h, vary=True)
                    else:
                        fit_params['c' + str(i)] = Parameter(value=v, vary=True)

                try:
                    print('starting', [i.value for i in fit_params.values()])
                    result = minimize(fun, fit_params, args=args, method=method)
                except Exception as e:
                    if debug:
                        print(e)
                    continue
                result = postproc_lmfit(result)
                fit = [i.value for i in result.params.values()]

                if result.aard < best_err and (check_fun is None or check_fun(fit)):
                    best_err = result.aard
                    best_coeffs = fit
                    best_result = result

                if debug:
                    if check_fun is None:
                        print(result.aard, method)
                    else:
                        print(result.aard, check_fun(fit), method)
    return best_coeffs, best_result
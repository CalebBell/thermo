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
SOFTWARE.
'''


__all__ = ['alpha_Twu91_objf', 'alpha_Twu91_objfc', 'fit_function',
           'Twu91_check_params', 'postproc_lmfit',
           'alpha_poly_objf', 'alpha_poly_objfc', 'poly_check_params',
           'fit_polynomial', 'poly_fit_statistics', 'fit_cheb_poly_auto',
           'data_fit_statistics', 'fit_customized']

import fluids
from fluids.numerics import (
    chebval,
    curve_fit,
    differential_evolution,
    fit_minimization_targets,
    horner,
    horner_and_der3,
    horner_stable,
    is_poly_negative,
    is_poly_positive,
    leastsq,
    linspace,
    lmder,
    min_max_ratios,
    polyder,
    polynomial_offset_scale,
    stable_poly_to_unstable,
    std,
)
from fluids.numerics import numpy as np

import thermo

try:
    from numpy.polynomial.chebyshev import cheb2poly
    from numpy.polynomial.polynomial import Polynomial
except:
    pass
try:
    from random import Random, uniform
except:
    pass

from math import log, pi, isinf, isnan


def split_data(x, y, folds=5, seed=42):
    pts = len(x)
    if pts != len(y):
        raise ValueError("Wrong size")
    z = list(range(pts))

    Random(seed).shuffle(z)
    # go one by one and assign data to each group

    fold_xs = [[] for _ in range(folds)]
    fold_ys = [[] for _ in range(folds)]
    for i in range(pts):
        l = i%folds
        fold_xs[l].append(x[i])
        fold_ys[l].append(y[i])

    return fold_xs, fold_ys

def assemble_fit_test_groups(x_groups, y_groups):
    folds = len(x_groups)
    train_x_groups = []
    test_x_groups = []
    train_y_groups = []
    test_y_groups = []
    for i in range(folds):
        x_test = x_groups[i]
        y_test = y_groups[i]
        test_x_groups.append(x_test)
        test_y_groups.append(y_test)
        x_train = []
        y_train = []
        for j in range(folds):
            if j != i:
                x_train.extend(x_groups[j])
                y_train.extend(y_groups[j])

        train_x_groups.append(x_train)
        train_y_groups.append(y_train)
    return (train_x_groups, test_x_groups, train_y_groups, test_y_groups)


def AICc(parameters, observations, SSE):
    # need about 200 data points to not use the correction
    # n/k < 40
    # SSE is sum of squared errors
    k = parameters
    n = observations
    if n - k - 1 == 0:
        return 1e200
    elif SSE == 0.0:
        return -1e100
#     return n*log(SSE/n) + 2.0*k + 2.0*k*(k+1)/(n-k-1) + n*log(2*pi) + n
    return n*log(SSE/n) + 2.0*k +(2.0*k*k + 2*k)/(n-k-1) + n*log(2*pi) + n

def BIC(parameters, observations, SSE):
    k = parameters
    n = observations
    if SSE == 0.0:
        return -1e100
    return n*log(SSE/n) + k*log(n) + n*log(2*pi) + n

def round_to_digits(number, n=7):
    # TODO move this to fluids numerics
    return float(f'{number:.{n}g}')


ChebTools = None

FIT_CHEBTOOLS_CHEB = 'ChebTools'
FIT_CHEBTOOLS_POLY = 'ChebTools polynomial'
FIT_CHEBTOOLS_STABLEPOLY = 'ChebTools stable polynomial'
FIT_NUMPY_POLY = 'NumPy polynomial'
FIT_NUMPY_STABLEPOLY = 'NumPy stable polynomial'


FIT_METHOD_MAP = {FIT_CHEBTOOLS_CHEB: 'chebyshev',
                  FIT_CHEBTOOLS_POLY: 'polynomial',
                  FIT_CHEBTOOLS_STABLEPOLY: 'stablepolynomial',
                  FIT_NUMPY_POLY:'polynomial',
                  FIT_NUMPY_STABLEPOLY: 'stablepolynomial',
                  }

def fit_polynomial(func, low, high, n,
                   interpolation_property=None, interpolation_property_inv=None,
                   interpolation_x=lambda x: x, interpolation_x_inv=lambda x: x,
                   arg_func=None, method=FIT_CHEBTOOLS_POLY, data=None):
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
    method : str
        Which fitting method to use, [-]
    data : tuple(list[float], list[float]) or None
        Optionally, data that can be used instead of random points when
        using the polyfit code

    Returns
    -------
    coeffs : list[float]
        Polynomial coefficients in order for evaluation by `horner`, [-]

    Notes
    -----
    This is powered by Ian Bell's ChebTools.

    '''
    global ChebTools

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
        return coeffs
    else:
        if method in (FIT_CHEBTOOLS_CHEB, FIT_CHEBTOOLS_POLY, FIT_CHEBTOOLS_STABLEPOLY):
            if ChebTools is None:
                import ChebTools
            cheb_fun = ChebTools.generate_Chebyshev_expansion(n-1, func_fun, low, high)
            cheb_coeffs = cheb_fun.coef()

            if method in (FIT_CHEBTOOLS_STABLEPOLY, FIT_CHEBTOOLS_POLY):
                coeffs = cheb2poly(cheb_coeffs)[::-1].tolist()
            if method == FIT_CHEBTOOLS_CHEB:
                return cheb_coeffs.tolist()
            elif method == FIT_CHEBTOOLS_STABLEPOLY:
            # Can use Polynomial(coeffs[::-1], domain=(low, high)) to evaluate, or horner_domain(x, coeffs, xmin, xmax)
                return coeffs
            elif method == FIT_CHEBTOOLS_POLY:
                return stable_poly_to_unstable(coeffs, low, high)
        elif method in (FIT_NUMPY_POLY, FIT_NUMPY_STABLEPOLY):
            if data is not None:
                x = [interpolation_x(T) for T in data[0]] if interpolation_x is not None else data[1]
                y = [interpolation_property(v) for v in data[1]] if interpolation_property is not None else data[1]
            else:
                x = linspace(low, high, 200)
                y = [func_fun(xi) for xi in x]
            fit = Polynomial.fit(x, y, n)
            coeffs = fit.coef
            coeffs = coeffs[::-1].tolist()
            if method == FIT_NUMPY_STABLEPOLY:
                return coeffs
            elif method == FIT_NUMPY_POLY:
                # sometimes the low/high bound will change becaues of the transformation, have to account for that
                return stable_poly_to_unstable(coeffs, min(low, high), max(low, high))


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
                        arg_func=None, method='polynomial', data=None):
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
    method : str
        Which type of polynomial the coefficients are;
        one of 'polynomial', 'stablepolynomial', 'chebyshev' [-]

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
    calc : list[float]
        The calculated values, [-]

    Notes
    -----
    '''
    low_orig, high_orig = low, high
    if data is not None:
        all_points_orig = data[0]
        pts = len(data[0])
    else:
        all_points_orig = linspace(low_orig, high_orig, pts)

    # Get the low, high, and x points in the transformed domain
    low, high = interpolation_x(low_orig), interpolation_x(high_orig)
    all_points = [interpolation_x(v) for v in all_points_orig]
    # may have changed basis
    low, high = min(low, high), max(low, high)
    offset, scale = polynomial_offset_scale(low, high)

    if method in FIT_METHOD_MAP:
        method = FIT_METHOD_MAP[method]

    # Calculate the fit values
    if method == 'polynomial':
        calc_pts = [horner(coeffs, x) for x in all_points]
    elif method == 'stablepolynomial':
        calc_pts = [horner_stable(x, coeffs, offset, scale) for x in all_points]
    elif method == 'chebyshev':
        calc_pts = [chebval(x, coeffs, offset, scale) for x in all_points]
    if interpolation_property_inv:
        for i in range(pts):
            calc_pts[i] = interpolation_property_inv(calc_pts[i])

    if arg_func is not None:
        actual_pts = [func(v, *arg_func(v)) for v in all_points_orig]
    else:
        actual_pts = [func(v) for v in all_points_orig]

    ARDs = [(abs((i-j)/j) if j != 0 else 0.0) for i, j in zip(calc_pts, actual_pts)]

    err_avg = sum(ARDs)/pts
    err_std = float(np.std(ARDs))

    actual_pts = np.array(actual_pts)
    calc_pts_np = np.array(calc_pts)

    max_ratio, min_ratio = float(max(calc_pts_np/actual_pts)), float(min(calc_pts_np/actual_pts))
    # max_ratio, min_ratio = max(ARDs), max(ARDs)
    return err_avg, err_std, min_ratio, max_ratio, calc_pts

def select_index_from_stats(stats, ns, selection_criteria=None):
    lowest_err_avg, lowest_err_std, lowest_err = 1e100, 1e100, 1e100
    lowest_err_avg_idx, lowest_err_std_idx, lowest_err_idx = None, None, None
    for i, ((err_avg, err_std, min_ratio, max_ratio, _),n) in enumerate(zip(stats, ns)):
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
        if selection_criteria is not None and selection_criteria(*stats[i]):
            return i
    # return lowest_err_idx

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
                  arg_func=None, method=FIT_CHEBTOOLS_POLY, data=None):

    def a_fit(n, eval_pts=eval_pts):
        coeffs = fit_polynomial(func, low, high, n,
                                interpolation_property=interpolation_property, interpolation_property_inv=interpolation_property_inv,
                                interpolation_x=interpolation_x, interpolation_x_inv=interpolation_x_inv,
                                arg_func=arg_func, method=method, data=data)

        method2 = FIT_METHOD_MAP[method]
        err_avg, err_std, min_ratio, max_ratio, calc_pts = poly_fit_statistics(func, coeffs, low, high, pts=eval_pts,
                                interpolation_property_inv=interpolation_property_inv,
                                interpolation_x=interpolation_x,
                                arg_func=arg_func, method=method2, data=data)
        return coeffs, err_avg, err_std, min_ratio, max_ratio, calc_pts

    worked_ns, worked_coeffs, worked_stats = [], [], []

    for n in ns:
        try:
            coeffs, err_avg, err_std, min_ratio, max_ratio, calc_pts = a_fit(n)
            worked_ns.append(n)
            worked_coeffs.append(coeffs)
            worked_stats.append((err_avg, err_std, min_ratio, max_ratio, calc_pts))
        except:
            pass
    return worked_ns, worked_coeffs, worked_stats


def fit_cheb_poly_auto(func, low, high, start_n=3, max_n=20, eval_pts=100,
                  interpolation_property=None, interpolation_property_inv=None,
                  interpolation_x=lambda x: x, interpolation_x_inv=lambda x: x,
                  arg_func=None, method=FIT_CHEBTOOLS_POLY, selection_criteria=None, data=None):
    if max_n > len(data[0]):
        max_n = len(data[0])

    worked_ns, worked_coeffs, worked_stats = fit_many_cheb_poly(func, low, high, ns=range(start_n, max_n+1),
                  interpolation_property=interpolation_property, interpolation_property_inv=interpolation_property_inv,
                  interpolation_x=interpolation_x, interpolation_x_inv=interpolation_x_inv,
                  arg_func=arg_func, eval_pts=eval_pts, method=method, data=data)
    idx = select_index_from_stats(worked_stats, worked_ns, selection_criteria=selection_criteria)
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
    return is_poly_negative(coeffs_d3, domain)

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




def fit_customized(Ts, data, fitting_func, fit_parameters, use_fit_parameters,
                   fit_method, objective, multiple_tries_max_objective,
                   guesses=None, initial_guesses=None, analytical_jac=None,
                   solver_kwargs=None, use_numba=False, multiple_tries=False,
                   do_statistics=False, multiple_tries_max_err=1e-5,
                   func_wrapped_for_leastsq=None, jac_wrapped_for_leastsq=None,
                   sigma=None):
    if solver_kwargs is None: solver_kwargs = {}
    if use_numba:
        fit_func_dict = fluids.numba.numerics.fit_minimization_targets
    else:
        fit_func_dict = fit_minimization_targets

    err_func = fit_func_dict[objective]
    err_fun_multiple_guesses = fit_func_dict[multiple_tries_max_objective]
    do_minimization = fit_method == 'differential_evolution'

    if do_minimization:
        def minimize_func(params):
            calc = fitting_func(Ts, *params)
            err = err_func(data, calc)
            return err

    p0 = [1.0]*len(fit_parameters)
    if guesses:
        for i, k in enumerate(use_fit_parameters):
            if k in guesses:
                p0[i] = guesses[k]

    if initial_guesses:
        # iterate over all the initial guess parameters we have and find the one
        # with the lowest error (according to the error criteria)
        best_hardcoded_guess = None
        best_hardcoded_err = 1e308
        hardcoded_errors = []
        hardcoded_guesses = initial_guesses
        extra_user_guess = [{k: v for k, v in zip(use_fit_parameters, p0)}]
        all_iter_guesses = hardcoded_guesses + extra_user_guess
        array_init_guesses = []
        array_init_guesses_dedup = set()
        err_func_init = fit_func_dict['MeanRelErr']
        for hardcoded in all_iter_guesses:
            ph = [None]*len(fit_parameters)
            for i, k in enumerate(use_fit_parameters):
                ph[i] = hardcoded[k]
            ph_tuple = tuple(ph)
            if ph_tuple in array_init_guesses_dedup:
                continue
            array_init_guesses_dedup.add(ph_tuple)
            array_init_guesses.append(ph)

            calc = fitting_func(Ts, *ph)
            err = err_func_init(data, calc)
            hardcoded_errors.append(err)
            if err < best_hardcoded_err:
                best_hardcoded_err = err
                best_hardcoded_guess = ph
        p0 = best_hardcoded_guess
        if best_hardcoded_err == 1e308 and fit_method != 'differential_evolution':
            raise ValueError("No attemped fitting parameters yielded remotely reasonable errors. Check input data or provide guesses")
        array_init_guesses = [p0 for _, p0 in sorted(zip(hardcoded_errors, array_init_guesses))]
    else:
        array_init_guesses = [p0]


    if func_wrapped_for_leastsq is None:
        def func_wrapped_for_leastsq(params):
            # jacobian is the same
            return fitting_func(Ts, *params) - data

    if jac_wrapped_for_leastsq is None:
        def jac_wrapped_for_leastsq(params):
            return analytical_jac(Ts, *params)

    pcov = None
    if fit_method == 'differential_evolution':
        if 'bounds' in solver_kwargs:
            working_bounds = solver_kwargs.pop('bounds')
        else:
            factor = 4.0
            if len(array_init_guesses) > 3:
                lowers_guess, uppers_guess = np.array(array_init_guesses).min(axis=0), np.array(array_init_guesses).max(axis=0)
                working_bounds = [(lowers_guess[i]*factor if lowers_guess[i] < 0. else lowers_guess[i]*(1.0/factor),
                                    uppers_guess[i]*(1.0/factor) if uppers_guess[i] < 0. else uppers_guess[i]*(factor),
                                    ) for i in range(len(use_fit_parameters))]
            else:
                working_bounds = [(0, 1000) for k in use_fit_parameters]
        popsize = solver_kwargs.get('popsize', 15)*len(fit_parameters)
        init = array_init_guesses
        for i in range(len(init), popsize):
            to_add = [uniform(ll, lh) for ll, lh in working_bounds]
            init.append(to_add)

        res = differential_evolution(minimize_func,# init=np.array(init),
                                      bounds=working_bounds, **solver_kwargs)
        popt = res['x']
    else:
        lm_direct = (fit_method == 'lm' and sigma is None)
        Dfun = jac_wrapped_for_leastsq if analytical_jac is not None else None
        if 'maxfev' not in solver_kwargs and fit_method == 'lm':
            # DO NOT INCREASE THIS! Make an analytical jacobian instead please.
            # Fought very hard to bring the analytical jacobian maxiters down to 500!
            # 250 seems too small.
            if analytical_jac is not None:
                solver_kwargs['maxfev'] = 500
            else:
                solver_kwargs['maxfev'] = 5000
        if multiple_tries:
            multiple_tries_best_error = 1e300
            best_popt, best_pcov = None, None
            popt = None
            if type(multiple_tries) is int and len(array_init_guesses) > multiple_tries:
                array_init_guesses = array_init_guesses[0:multiple_tries]
            for p0 in array_init_guesses:
                try:
                    if lm_direct:
                        if Dfun is not None:
                            popt, info, status = lmder(func_wrapped_for_leastsq, Dfun, p0, tuple(), True,
                                      0, 1.49012e-8, 1.49012e-8, 0.0, solver_kwargs['maxfev'],
                                      100, None)
                            # print(info, 'info', status, 'status')
                        else:
                            popt, status = leastsq(func_wrapped_for_leastsq, p0, Dfun=Dfun, **solver_kwargs)
                        if Dfun is not None and (not np.all(np.isfinite(info['fjac'])) or not np.all(np.isfinite(info['fvec']))):
                            # Didn't go smoothly at all
                            popt, status = leastsq(func_wrapped_for_leastsq, p0, **solver_kwargs)


                        pcov = None
                    else:
                        popt, pcov = curve_fit(fitting_func, Ts, data, sigma=sigma, p0=p0, jac=analytical_jac,
                                                method=fit_method, absolute_sigma=True, **solver_kwargs)
                except:
                    continue
                calc = fitting_func(Ts, *popt)
                curr_err = err_fun_multiple_guesses(data, calc)
                # print(f'p0 {p0} lead to', curr_err)
                if curr_err < multiple_tries_best_error:
                    # print('accepting')
                    best_popt, best_pcov = popt, pcov
                    multiple_tries_best_error = curr_err
                    if curr_err < multiple_tries_max_err:
                        # print('breaking')
                        break

            if best_popt is None:
                raise ValueError("No guesses converged")
            else:
                popt, pcov = best_popt, best_pcov
        else:
            if lm_direct:
                popt, _ = leastsq(func_wrapped_for_leastsq, p0, Dfun=Dfun, **solver_kwargs)
                pcov = None
            else:
                popt, pcov = curve_fit(fitting_func, Ts, data, sigma=sigma, p0=p0, jac=analytical_jac,
                                        method=fit_method, absolute_sigma=True, **solver_kwargs)
    out_kwargs = {}
    for param_name, param_value in zip(fit_parameters, popt):
        out_kwargs[param_name] = float(param_value)

    if do_statistics:
        if not use_numba:
            stats_func = data_fit_statistics
        else:
            stats_func = thermo.numba.fitting.data_fit_statistics
        calc = fitting_func(Ts, *popt)
        stats = stats_func(Ts, data, calc)
        statistics = {}
        statistics['calc'] = calc
        statistics['MAE'] = float(stats[0])
        statistics['STDEV'] = float(stats[1])
        statistics['min_ratio'] = float(stats[2])
        statistics['max_ratio'] = float(stats[3])
        try:
            pcov = float(pcov)
        except:
            # ndarray or None
            pass
        statistics['pcov'] = pcov
        return out_kwargs, statistics


    return out_kwargs





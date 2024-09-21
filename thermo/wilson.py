'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains a class :obj:`Wilson` for performing activity coefficient
calculations with the Wilson model. An older, functional calculation for
activity coefficients only is also present, :obj:`Wilson_gammas`.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Wilson Class
============

.. autoclass:: Wilson
    :members: to_T_xs, GE, dGE_dT, d2GE_dT2, d3GE_dT3, d2GE_dTdxs, dGE_dxs, d2GE_dxixjs, d3GE_dxixjxks, lambdas, dlambdas_dT, d2lambdas_dT2, d3lambdas_dT3, from_DDBST, from_DDBST_as_matrix
    :undoc-members:
    :show-inheritance:
    :exclude-members: gammas

Wilson Functional Calculations
==============================
.. autofunction:: Wilson_gammas

Wilson Regression Calculations
==============================
.. autofunction:: wilson_gammas_binaries

'''

from math import exp, log

from fluids.constants import R
from fluids.numerics import numpy as np
from fluids.numerics import trunc_exp

from thermo.activity import GibbsExcess, d2interaction_exp_dT2, d3interaction_exp_dT3, dinteraction_exp_dT, interaction_exp

try:
    array, zeros, npsum, nplog, ones = np.array, np.zeros, np.sum, np.log, np.ones
except (ImportError, AttributeError):
    pass

__all__ = ['Wilson', 'Wilson_gammas', 'wilson_gammas_binaries', 'wilson_gammas_binaries_jac']


def wilson_xj_Lambda_ijs(xs, lambdas, N, xj_Lambda_ijs=None):
    if xj_Lambda_ijs is None:
        xj_Lambda_ijs = [0.0]*N
    for i in range(N):
        tot = 0.0
        lambdasi = lambdas[i]
        for j in range(N):
            tot += xs[j]*lambdasi[j]
        xj_Lambda_ijs[i] = tot
    return xj_Lambda_ijs

def wilson_dGE_dT(xs, T, GE, N, xj_Lambda_ijs_inv, xj_dLambda_dTijs):
    tot = GE/T

    sum1 = 0.0
    for i in range(N):
        sum1 += xs[i]*xj_dLambda_dTijs[i]*xj_Lambda_ijs_inv[i]
    tot -= T*R*sum1
    return tot

def wilson_d2GE_dT2(xs, T, N, xj_Lambda_ijs_inv, xj_dLambda_dTijs, xj_d2Lambda_dT2ijs):
    sum0, sum1 = 0.0, 0.0
    for i in range(N):
        t = xs[i]*xj_Lambda_ijs_inv[i]
        t2 = xj_dLambda_dTijs[i]*t
        sum1 += t2

        sum0 += t*(xj_d2Lambda_dT2ijs[i] - xj_dLambda_dTijs[i]*xj_dLambda_dTijs[i]*xj_Lambda_ijs_inv[i])

    d2GE_dT2 = -R*(T*sum0 + 2.0*sum1)
    return d2GE_dT2

def wilson_d3GE_dT3(xs, T, N, xj_Lambda_ijs_inv, xj_dLambda_dTijs, xj_d2Lambda_dT2ijs, xj_d3Lambda_dT3ijs):
    #Term is directly from the one above it
    sum0 = 0.0
    for i in range(N):
        sum0 += xj_Lambda_ijs_inv[i]*xs[i]*(xj_d2Lambda_dT2ijs[i]
                - xj_dLambda_dTijs[i]*xj_dLambda_dTijs[i]*xj_Lambda_ijs_inv[i])

    sum_d3 = 0.0
    for i in range(N):
        sum_d3 += xs[i]*xj_d3Lambda_dT3ijs[i]*xj_Lambda_ijs_inv[i]

    sum_comb = 0.0
    for i in range(N):
        sum_comb += xs[i]*xj_d2Lambda_dT2ijs[i]*xj_dLambda_dTijs[i]*xj_Lambda_ijs_inv[i]*xj_Lambda_ijs_inv[i]
    sum_comb *= 3.0

    sum_last = 0.0
    for i in range(N):
        v = xj_dLambda_dTijs[i]*xj_Lambda_ijs_inv[i]
        sum_last += xs[i]*v*v*v
    sum_last *= 2.0

    d3GE_dT3 = -R*(3.0*sum0 + T*(sum_d3 - sum_comb + sum_last))
    return d3GE_dT3

def wilson_d2GE_dTdxs(xs, T, N, log_xj_Lambda_ijs, lambdas, dlambdas_dT,
                      xj_Lambda_ijs_inv, xj_dLambda_dTijs, d2GE_dTdxs=None):
    if d2GE_dTdxs is None:
        d2GE_dTdxs = [0.0]*N

    for i in range(N):
        tot1 = xj_dLambda_dTijs[i]*xj_Lambda_ijs_inv[i]
        tot2 = 0.0
        for j in range(N):
            t1 = lambdas[j][i]*xj_Lambda_ijs_inv[j]
            tot1 += xs[j]*xj_Lambda_ijs_inv[j]*(dlambdas_dT[j][i] - xj_dLambda_dTijs[j]*t1)
            tot2 += xs[j]*t1

        dG = -R*(T*tot1 + log_xj_Lambda_ijs[i] + tot2)

        d2GE_dTdxs[i] = dG

    return d2GE_dTdxs

def wilson_dGE_dxs(xs, T, N, log_xj_Lambda_ijs, lambdas, xj_Lambda_ijs_inv, dGE_dxs=None):
    if dGE_dxs is None:
        dGE_dxs = [0.0]*N
    mRT = -T*R
    for k in range(N):
        tot = log_xj_Lambda_ijs[k]
        for i in range(N):
            tot += xs[i]*lambdas[i][k]*xj_Lambda_ijs_inv[i]
        dGE_dxs[k] = mRT*tot
    return dGE_dxs

def wilson_d2GE_dxixjs(xs, T, N, lambdas, xj_Lambda_ijs_inv, d2GE_dxixjs=None):
    if d2GE_dxixjs is None:
        d2GE_dxixjs = [[0.0]*N for i in range(N)] # numba: delete
#        d2GE_dxixjs = zeros((N, N)) # numba: uncomment

    RT = R*T
    for k in range(N):
        dG_row = d2GE_dxixjs[k]
        for m in range(N):
            tot = 0.0
            for i in range(N):
                tot += xs[i]*lambdas[i][k]*lambdas[i][m]*(xj_Lambda_ijs_inv[i]*xj_Lambda_ijs_inv[i])
            tot -= lambdas[k][m]*xj_Lambda_ijs_inv[k]
            tot -= lambdas[m][k]*xj_Lambda_ijs_inv[m]
            dG_row[m] = RT*tot

    return d2GE_dxixjs

def wilson_d3GE_dxixjxks(xs, T, N, lambdas, xj_Lambda_ijs_inv, d3GE_dxixjxks=None):
    if d3GE_dxixjxks is None:
        d3GE_dxixjxks = [[[0.0]*N for i in range(N)] for _ in range(N)]# numba: delete
#        d3GE_dxixjxks = zeros((N, N, N)) # numba: uncomment

    nRT = -R*T
    for k in range(N):
        dG_matrix = d3GE_dxixjxks[k]
        for m in range(N):
            dG_row = dG_matrix[m]
            for n in range(N):
                tot = 0.0
                for i in range(N):
                    num = xs[i]*lambdas[i][k]*lambdas[i][m]*lambdas[i][n]
                    den = xj_Lambda_ijs_inv[i]*xj_Lambda_ijs_inv[i]*xj_Lambda_ijs_inv[i]
                    tot += num*den
                tot *= 2.0

                tot -= lambdas[k][m]*lambdas[k][n]*xj_Lambda_ijs_inv[k]*xj_Lambda_ijs_inv[k]
                tot -= lambdas[m][k]*lambdas[m][n]*xj_Lambda_ijs_inv[m]*xj_Lambda_ijs_inv[m]
                tot -= lambdas[n][m]*lambdas[n][k]*xj_Lambda_ijs_inv[n]*xj_Lambda_ijs_inv[n]
                dG_row[n] = nRT*tot

    return d3GE_dxixjxks

def wilson_gammas(xs, N, lambdas, xj_Lambda_ijs_inv, gammas=None, vec0=None):
    if gammas is None:
        gammas = [0.0]*N
    if vec0 is None:
        vec0 = [0.0]*N

    for i in range(N):
        vec0[i] = xs[i]*xj_Lambda_ijs_inv[i]

    for j in range(N):
        const_j = vec0[j]
        for i in range(N):
            gammas[i] += lambdas[j][i]*const_j
    for i in range(N):
        gammas[i] = exp(1.0 - gammas[i])*xj_Lambda_ijs_inv[i]

    return gammas

def wilson_gammas_from_args(xs, N, lambdas, xj_Lambda_ijs=None, vec0=None, gammas=None,):
    if xj_Lambda_ijs is None:
        xj_Lambda_ijs = [0.0]*N
    xj_Lambda_ijs = wilson_xj_Lambda_ijs(xs, lambdas, N, xj_Lambda_ijs)
    for i in range(N):
        # Can make this optimization here only
        xj_Lambda_ijs[i] = 1.0/xj_Lambda_ijs[i]
    return wilson_gammas(xs, N, lambdas, xj_Lambda_ijs, gammas=gammas, vec0=vec0)

MIN_LAMBDA_WILSON = 1e-20

def wilson_gammas_binaries(xs, lambda12, lambda21, calc=None):
    r'''Calculates activity coefficients at fixed `lambda` values for
    a binary system at a series of mole fractions. This is used for
    regression of `lambda` parameters. This function is highly optimized,
    and operates on multiple points at a time.

    .. math::
        \ln \gamma_1 = -\ln(x_1 + \Lambda_{12}x_2) + x_2\left(
        \frac{\Lambda_{12}}{x_1 + \Lambda_{12}x_2}
        - \frac{\Lambda_{21}}{x_2 + \Lambda_{21}x_1}
        \right)

    .. math::
        \ln \gamma_2 = -\ln(x_2 + \Lambda_{21}x_1) - x_1\left(
        \frac{\Lambda_{12}}{x_1 + \Lambda_{12}x_2}
        - \frac{\Lambda_{21}}{x_2 + \Lambda_{21}x_1}
        \right)

    Parameters
    ----------
    xs : list[float]
        Liquid mole fractions of each species in the format
        x0_0, x1_0, (component 1 point1, component 2 point 1),
        x0_1, x1_1, (component 1 point2, component 2 point 2), ...
        [-]
    lambda12 : float
        `lambda` parameter for 12, [-]
    lambda21 : float
        `lambda` parameter for 21, [-]
    gammas : list[float], optional
        Array to store the activity coefficient for each species in the liquid
        mixture, indexed the same as `xs`; can be omitted or provided
        for slightly better performance [-]

    Returns
    -------
    gammas : list[float]
        Activity coefficient for each species in the liquid mixture,
        indexed the same as `xs`, [-]

    Notes
    -----
    The lambda values are hard-coded to replace values under zero which are
    mathematically impossible, with a very small number. This is helpful for
    regression which might try to make those values negative.

    Examples
    --------
    >>> wilson_gammas_binaries([.1, .9, 0.3, 0.7, .85, .15], 0.1759, 0.7991)
    [3.42989, 1.03432, 1.74338, 1.21234, 1.01766, 2.30656]
    '''
    if lambda12 < MIN_LAMBDA_WILSON:
        lambda12 = MIN_LAMBDA_WILSON
    if lambda21 < MIN_LAMBDA_WILSON:
        lambda21 = MIN_LAMBDA_WILSON
    pts = len(xs)//2 # Always even

    if calc is None:
        allocate_size = (pts*2)
        calc = [0.0]*allocate_size

    for i in range(pts):
        i2 = i*2
        x1 = xs[i2]
        x2 = 1.0 - x1

        c0 = 1.0/(x1 + x2*lambda12)
        c1 = 1.0/(x2 + x1*lambda21)
        c3 = lambda12*c0 - lambda21*c1

        calc[i2] = trunc_exp(c3*x2)*c0
        calc[i2 + 1] = trunc_exp(-c3*x1)*c1
    return calc


"""
Actually readable expression of `wilson_gammas_binaries`:

    if lambda12 < MIN_LAMBDA_WILSON:
        lambda12 = MIN_LAMBDA_WILSON
    if lambda21 < MIN_LAMBDA_WILSON:
        lambda21 = MIN_LAMBDA_WILSON
    pts = int(len(xs)/2) # Always even
    allocate_size = (pts*2)

#    lambdas = ones((2,2)) # numba: uncomment
    lambdas = [[1.0, 1.0], [1.0, 1.0]] # numba: delete
    lambdas[0][1] = lambda12
    lambdas[1][0] = lambda21

    if calc is None:
        calc = [0.0]*allocate_size

    xj_Lambda_ijs_vec = [0.0]*2
    vec0 = [0.0]*2
    gammas = [0.0]*2
    xs_pt = [0.0]*2

    for i in range(pts):
        i2 = i*2
        xs_pt[0] = xs[i2]
        xs_pt[1] = 1.0 - xs_pt[0]

        xj_Lambda_ijs = wilson_xj_Lambda_ijs(xs_pt, lambdas, N=2, xj_Lambda_ijs=xj_Lambda_ijs_vec)
        xj_Lambda_ijs[0] = 1.0/xj_Lambda_ijs[0]
        xj_Lambda_ijs[1] = 1.0/xj_Lambda_ijs[1]
        gammas = wilson_gammas(xs_pt, N=2, lambdas=lambdas, xj_Lambda_ijs_inv=xj_Lambda_ijs, gammas=gammas, vec0=vec0)
        calc[i2] = gammas[0]
        calc[i2 + 1] = gammas[1]
    return calc
"""

def wilson_gammas_binaries_jac(xs, lambda12, lambda21, calc=None):
    if lambda12 < MIN_LAMBDA_WILSON:
        lambda12 = MIN_LAMBDA_WILSON
    if lambda21 < MIN_LAMBDA_WILSON:
        lambda21 = MIN_LAMBDA_WILSON
    pts = len(xs)//2 # Always even

    if calc is None:
        allocate_size = (pts*2)
        calc = np.zeros((allocate_size, 2))

    for i in range(pts):
        i2 = i*2
        x1 = xs[i2]
        x2 = 1.0 - x1

        c0 = lambda12*x2
        c1 = c0 + x1
        c2 = 1.0/c1
        c3 = lambda21*x1
        c4 = c3 + x2
        c5 = 1.0/c4
        c6 = c2*lambda12 - c5*lambda21
        c7 = trunc_exp(c6*x2)
        c8 = c2*c5
        c9 = trunc_exp(-c6*x1)

        calc[i2][0] = -c7*lambda12*x2*x2*c2*c2*c2
        calc[i2][1] = c7*c8*x2*(c3*c5 - 1.0)
        calc[i2 + 1][0] = c8*c9*x1*(c0*c2 - 1.0)
        calc[i2 + 1][1] = -c9*lambda21*x1*x1*c5*c5*c5
    return calc


class Wilson(GibbsExcess):
    r'''Class for representing an a liquid with excess gibbs energy represented
    by the Wilson equation. This model is capable of representing most
    nonideal liquids for vapor-liquid equilibria, but is not recommended for
    liquid-liquid equilibria.

    The two basic equations are as follows; all other properties are derived
    from these.

    .. math::
        g^E = -RT\sum_i x_i \ln\left(\sum_j x_j \lambda_{i,j} \right)

    .. math::
        \Lambda_{ij} = \exp\left[a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T
                + d_{ij}T + \frac{e_{ij}}{T^2} + f_{ij}{T^2}\right]

    Parameters
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]
    lambda_coeffs : list[list[list[float]]], optional
        Wilson parameters, indexed by [i][j] and then each value is a 6
        element list with parameters (`a`, `b`, `c`, `d`, `e`, `f`);
        either `lambda_coeffs` or the lambda parameters are required, [various]
    ABCDEF : tuple(list[list[float]], 6), optional
        The lamba parameters can be provided as a tuple, [various]
    lambda_as : list[list[float]], optional
        `a` parameters used in calculating :obj:`Wilson.lambdas`, [-]
    lambda_bs : list[list[float]], optional
        `b` parameters used in calculating :obj:`Wilson.lambdas`, [K]
    lambda_cs : list[list[float]], optional
        `c` parameters used in calculating :obj:`Wilson.lambdas`, [-]
    lambda_ds : list[list[float]], optional
        `d` paraemeters used in calculating :obj:`Wilson.lambdas`, [1/K]
    lambda_es : list[list[float]], optional
        `e` parameters used in calculating :obj:`Wilson.lambdas`, [K^2]
    lambda_fs : list[list[float]], optional
        `f` parameters used in calculating :obj:`Wilson.lambdas`, [1/K^2]

    Attributes
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]
    model_id : int
        Unique identifier for the Wilson activity model, [-]

    Notes
    -----
    In addition to the methods presented here, the methods of its base class
    :obj:`thermo.activity.GibbsExcess` are available as well.

    .. warning::
        If parameters are ommited for all interactions, this model
        reverts to :obj:`thermo.activity.IdealSolution`. In large systems it
        is common to only regress parameters for the most important components;
        set `lambda` parameters for other components to 0 to "ignore" them and
        treat them as ideal components.

    This class works with python lists, numpy arrays, and can be accelerated
    with Numba or PyPy quite effectively.

    Examples
    --------
    **Example 1**

    This object-oriented class provides access to many more thermodynamic
    properties than :obj:`Wilson_gammas`, but it can also be used like that
    function. In the following example, `gammas` are calculated with both
    functions. The `lambdas` cannot be specified in this class; but fixed
    values can be converted with the `log` function so that fixed values will
    be obtained.

    >>> Wilson_gammas([0.252, 0.748], [[1, 0.154], [0.888, 1]])
    [1.881492608717, 1.165577493112]
    >>> GE = Wilson(T=300.0, xs=[0.252, 0.748], lambda_as=[[0, log(0.154)], [log(0.888), 0]])
    >>> GE.gammas()
    [1.881492608717, 1.165577493112]

    We can check that the same lambda values were computed as well, and that
    there is no temperature dependency:

    >>> GE.lambdas()
    [[1.0, 0.154], [0.888, 1.0]]
    >>> GE.dlambdas_dT()
    [[0.0, 0.0], [0.0, 0.0]]

    In this case, there is no temperature dependency in the Wilson model as the
    `lambda` values are fixed, so the excess enthalpy is always zero. Other
    properties are not always zero.

    >>> GE.HE(), GE.CpE()
    (0.0, 0.0)
    >>> GE.GE(), GE.SE(), GE.dGE_dT()
    (683.165839398, -2.277219464, 2.2772194646)

    **Example 2**

    ChemSep is a (partially) free program for modeling distillation. Besides
    being a wonderful program, it also ships with a permissive license several
    sets of binary interaction parameters. The Wilson parameters in it can
    be accessed from Thermo as follows. In the following case, we compute
    activity coefficients of the ethanol-water system at mole fractions of
    [.252, 0.748].

    >>> from thermo.interaction_parameters import IPDB
    >>> CAS1, CAS2 = '64-17-5', '7732-18-5'
    >>> lambda_as = IPDB.get_ip_asymmetric_matrix(name='ChemSep Wilson', CASs=[CAS1, CAS2], ip='aij')
    >>> lambda_bs = IPDB.get_ip_asymmetric_matrix(name='ChemSep Wilson', CASs=[CAS1, CAS2], ip='bij')
    >>> GE = Wilson(T=273.15+70, xs=[.252, .748], lambda_as=lambda_as, lambda_bs=lambda_bs)
    >>> GE.gammas()
    [1.95733110, 1.1600677]

    In ChemSep, the form of the Wilson `lambda` equation is

    .. math::
        \Lambda_{ij} = \frac{V_j}{V_i}\exp\left( \frac{-A_{ij}}{RT}\right)

    The parameters were converted to the form used by Thermo as follows:

    .. math::
        a_{ij} = \log\left(\frac{V_j}{V_i}\right)

    .. math::
        b_{ij} = \frac{-A_{ij}}{R}= \frac{-A_{ij}}{ 1.9872042586408316}


    This system was chosen because there is also a sample problem for the same
    components from the DDBST which can be found here:
    http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/P05.01a%20VLE%20Behavior%20of%20Ethanol%20-%20Water%20Using%20Wilson.xps

    In that example, with different data sets and parameters, they obtain at
    the same conditions activity coefficients of [1.881, 1.165]. Different
    sources of parameters for the same system will generally have similar
    behavior if regressed in the same temperature range. As higher order
    `lambda` parameters are added, models become more likely to behave
    differently. It is recommended in [3]_ to regress the minimum number of
    parameters required.


    **Example 3**

    The DDBST has published some sample problems which are fun to work with.
    Because the DDBST uses a different equation form for the coefficients than
    this model implements, we must initialize the :obj:`Wilson` object with
    a different method.

    >>> T = 331.42
    >>> N = 3
    >>> Vs_ddbst = [74.04, 80.67, 40.73]
    >>> as_ddbst = [[0, 375.2835, 31.1208], [-1722.58, 0, -1140.79], [747.217, 3596.17, 0.0]]
    >>> bs_ddbst = [[0, -3.78434, -0.67704], [6.405502, 0, 2.59359], [-0.256645, -6.2234, 0]]
    >>> cs_ddbst = [[0.0, 7.91073e-3, 8.68371e-4], [-7.47788e-3, 0.0, 3.1e-5], [-1.24796e-3, 3e-5, 0.0]]
    >>> dis = eis = fis = [[0.0]*N for _ in range(N)]
    >>> params = Wilson.from_DDBST_as_matrix(Vs=Vs_ddbst, ais=as_ddbst, bis=bs_ddbst, cis=cs_ddbst, dis=dis, eis=eis, fis=fis, unit_conversion=False)
    >>> xs = [0.229, 0.175, 0.596]
    >>> GE = Wilson(T=T, xs=xs, lambda_as=params[0], lambda_bs=params[1], lambda_cs=params[2], lambda_ds=params[3], lambda_es=params[4], lambda_fs=params[5])
    >>> GE
    Wilson(T=331.42, xs=[0.229, 0.175, 0.596], lambda_as=[[0.0, 3.870101271243586, 0.07939943395502425], [-6.491263271243587, 0.0, -3.276991837288562], [0.8542855660449756, 6.906801837288562, 0.0]], lambda_bs=[[0.0, -375.2835, -31.1208], [1722.58, 0.0, 1140.79], [-747.217, -3596.17, -0.0]], lambda_ds=[[-0.0, -0.00791073, -0.000868371], [0.00747788, -0.0, -3.1e-05], [0.00124796, -3e-05, -0.0]])
    >>> GE.GE(), GE.dGE_dT(), GE.d2GE_dT2()
    (480.26392663, 4.35596276623, -0.02913038452501)
    >>> GE.HE(), GE.SE(), GE.dHE_dT(), GE.dSE_dT()
    (-963.389253354, -4.3559627662, 9.6543920392, 0.029130384525)
    >>> GE.gammas()
    [1.2233934334, 1.100945902470, 1.205289928117]


    The solution given by the DDBST has the same values [1.223, 1.101, 1.205],
    and can be found here:
    http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/05.09%20Compare%20Experimental%20VLE%20to%20Wilson%20Equation%20Results.xps


    **Example 4**

    A simple example is given in [1]_; other textbooks sample problems are
    normally in the same form as this - with only volumes and the `a` term
    specified. The system is 2-propanol/water at 353.15 K, and the mole
    fraction of 2-propanol is 0.25.

    >>> T = 353.15
    >>> N = 2
    >>> Vs = [76.92, 18.07] # cm^3/mol
    >>> ais = [[0.0, 437.98],[1238.0, 0.0]] # cal/mol
    >>> bis = cis = dis = eis = fis = [[0.0]*N for _ in range(N)]
    >>> params = Wilson.from_DDBST_as_matrix(Vs=Vs, ais=ais, bis=bis, cis=cis, dis=dis, eis=eis, fis=fis, unit_conversion=True)
    >>> xs = [0.25, 0.75]
    >>> GE = Wilson(T=T, xs=xs, lambda_as=params[0], lambda_bs=params[1], lambda_cs=params[2], lambda_ds=params[3], lambda_es=params[4], lambda_fs=params[5])
    >>> GE.gammas()
    [2.124064516, 1.1903745834]

    The activity coefficients given in [1]_ are [2.1244, 1.1904]; matching (
    with a slight deviation from their use of 1.987 as a gas constant).

    References
    ----------
    .. [1] Smith, H. C. Van Ness Joseph M. Introduction to Chemical Engineering
       Thermodynamics 4th Edition, Joseph M. Smith, H. C. Van
       Ness, 1987.
    .. [2] Kooijman, Harry A., and Ross Taylor. The ChemSep Book. Books on
       Demand Norderstedt, Germany, 2000.
    .. [3] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
       Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
    '''

    model_id = 200

    _model_attributes = ('lambda_as', 'lambda_bs', 'lambda_cs',
                        'lambda_ds', 'lambda_es', 'lambda_fs')
    _cached_calculated_attributes = ('_d3GE_dxixjxks', '_xj_dLambda_dTijs', '_xj_Lambda_ijs', '_log_xj_Lambda_ijs',
                                     '_dlambdas_dT', '_lambdas', '_d3GE_dT3', '_xj_d2Lambda_dT2ijs', '_xj_Lambda_ijs_inv',
                                     '_d2lambdas_dT2', '_d3lambdas_dT3', '_xj_d3Lambda_dT3ijs')

    __slots__ = GibbsExcess.__slots__ + _model_attributes + _cached_calculated_attributes + ('lambda_coeffs_nonzero',)
    recalculable_attributes = _cached_calculated_attributes + GibbsExcess.recalculable_attributes


    gammas_from_args = staticmethod(wilson_gammas_from_args)
    def gammas_args(self, T=None):
        if T is not None:
            obj = self.to_T_xs(T=T, xs=self.xs)
        else:
            obj = self

        lambdas = obj.lambdas()
        N = obj.N
        if self.vectorized:
            xj_Lambda_ijs, vec0 = zeros(N), zeros(N)
        else:
            xj_Lambda_ijs, vec0 = [0.0]*N, [0.0]*N
        return (N, lambdas, xj_Lambda_ijs, vec0)

    @staticmethod
    def from_DDBST(Vi, Vj, a, b, c, d=0.0, e=0.0, f=0.0, unit_conversion=True):
        r'''Converts parameters for the wilson equation in the DDBST to the
        basis used in this implementation.

        .. math::
            \Lambda_{ij} = \frac{V_j}{V_i}\exp\left(\frac{-\Delta \lambda_{ij}}{RT}
            \right)

        .. math::
            \Delta \lambda_{ij} = a_{ij} + b_{ij}T + c T^2 + d_{ij}T\ln T
            + e_{ij}T^3 + f_{ij}/T

        Parameters
        ----------
        Vi : float
            Molar volume of component i; needs only to be in the same units as
            `Vj`, [cm^3/mol]
        Vj : float
            Molar volume of component j; needs only to be in the same units as
            `Vi`, [cm^3/mol]
        a : float
            `a` parameter in DDBST form, [K]
        b : float
            `b` parameter in DDBST form, [-]
        c : float
            `c` parameter in DDBST form, [1/K]
        d : float, optional
            `d` parameter in DDBST form, [-]
        e : float, optional
            `e` parameter in DDBST form, [1/K^2]
        f : float, optional
            `f` parameter in DDBST form, [K^2]
        unit_conversion : bool
            If True, the input coefficients are in units of cal/K/mol, and a
            `R` gas constant of 1.9872042... is used for the conversion;
            the DDBST uses this generally, [-]

        Returns
        -------
        a : float
            `a` parameter in :obj:`Wilson` form, [-]
        b : float
            `b` parameter in :obj:`Wilson` form, [K]
        c : float
            `c` parameter in :obj:`Wilson` form, [-]
        d : float
            `d` parameter in :obj:`Wilson` form, [1/K]
        e : float
            `e` parameter in :obj:`Wilson` form, [K^2]
        f : float
            `f` parameter in :obj:`Wilson` form, [1/K^2]

        Notes
        -----
        The units show how the different variables are related to each other.

        Examples
        --------
        >>> Wilson.from_DDBST(Vi=74.04, Vj=80.67, a=375.2835, b=-3.78434, c=0.00791073, d=0.0, e=0.0, f=0.0, unit_conversion=False)
        (3.8701012712, -375.2835, -0.0, -0.00791073, -0.0, -0.0)
        '''
        if unit_conversion:
            Rg = 1.9872042586408316 # DDBST document suggests  1.9858775
        else:
            Rg = 1.0 # Not used in some cases - be very careful
        a, b = log(Vj/Vi) - b/Rg, -a/Rg
        c, d = -d/Rg, -c/Rg
        e = -e/Rg
        f = -f/Rg
        return (a, b, c, d, e, f)

    @staticmethod
    def from_DDBST_as_matrix(Vs, ais=None, bis=None, cis=None, dis=None,
                             eis=None, fis=None,
                             unit_conversion=True):
        r'''Converts parameters for the wilson equation in the DDBST to the
        basis used in this implementation. Matrix wrapper around
        :obj:`Wilson.from_DDBST`.

        Parameters
        ----------
        Vs : list[float]
            Molar volume of component; needs only to be in consistent units,
            [cm^3/mol]
        ais : list[list[float]]
            `a` parameters in DDBST form, [K]
        bis : list[list[float]]
            `b` parameters in DDBST form, [-]
        cis : list[list[float]]
            `c` parameters in DDBST form, [1/K]
        dis : list[list[float]], optional
            `d` parameters in DDBST form, [-]
        eis : list[list[float]], optional
            `e` parameters in DDBST form, [1/K^2]
        fis : list[list[float]], optional
            `f` parameters in DDBST form, [K^2]
        unit_conversion : bool
            If True, the input coefficients are in units of cal/K/mol, and a
            `R` gas constant of 1.9872042... is used for the conversion;
            the DDBST uses this generally, [-]

        Returns
        -------
        a : list[list[float]]
            `a` parameters in :obj:`Wilson` form, [-]
        b : list[list[float]]
            `b` parameters in :obj:`Wilson` form, [K]
        c : list[list[float]]
            `c` parameters in :obj:`Wilson` form, [-]
        d : list[list[float]]
            `d` paraemeters in :obj:`Wilson` form, [1/K]
        e : list[list[float]]
            `e` parameters in :obj:`Wilson` form, [K^2]
        f : list[list[float]]
            `f` parameters in :obj:`Wilson` form, [1/K^2]
        '''
        N = len(Vs)
        cmps = range(N)
        if ais is None:
            ais = [[0.0]*N for _ in range(N)]
        if bis is None:
            bis = [[0.0]*N for _ in range(N)]
        if cis is None:
            cis = [[0.0]*N for _ in range(N)]
        if dis is None:
            dis = [[0.0]*N for _ in range(N)]
        if eis is None:
            eis = [[0.0]*N for _ in range(N)]
        if fis is None:
            fis = [[0.0]*N for _ in range(N)]
        a_mat, b_mat, c_mat, d_mat, e_mat, f_mat = [], [], [], [], [], []
        for i in cmps:
            a_row, b_row, c_row, d_row, e_row, f_row = [], [], [], [], [], []
            for j in cmps:
                a, b, c, d, e, f = Wilson.from_DDBST(Vs[i], Vs[j], ais[i][j],
                                              bis[i][j], cis[i][j], dis[i][j],
                                              eis[i][j], fis[i][j],
                                              unit_conversion=unit_conversion)
                a_row.append(a)
                b_row.append(b)
                c_row.append(c)
                d_row.append(d)
                e_row.append(e)
                f_row.append(f)
            a_mat.append(a_row)
            b_mat.append(b_row)
            c_mat.append(c_row)
            d_mat.append(d_row)
            e_mat.append(e_row)
            f_mat.append(f_row)
        return (a_mat, b_mat, c_mat, d_mat, e_mat, f_mat)

    def __init__(self, *, xs, T=GibbsExcess.T_DEFAULT, lambda_coeffs=None, ABCDEF=None, lambda_as=None, lambda_bs=None,
                 lambda_cs=None, lambda_ds=None, lambda_es=None, lambda_fs=None):
        self.T = T
        self.xs = xs
        self.vectorized = vectorized = type(xs) is not list
        self.N = N = len(xs)

        if ABCDEF is None:
            ABCDEF = (lambda_as, lambda_bs, lambda_cs, lambda_ds, lambda_es, lambda_fs)
        if lambda_coeffs is not None:
            pass
        else:
            try:
                all_lengths = tuple(len(coeffs) for coeffs in ABCDEF if coeffs is not None)
                if len(set(all_lengths)) > 1:
                    raise ValueError(f"Coefficient arrays of different size found: {all_lengths}")
                all_lengths_inner = tuple(len(coeffs[0]) for coeffs in ABCDEF if coeffs is not None)
                if len(set(all_lengths_inner)) > 1:
                    raise ValueError(f"Coefficient arrays of different size found: {all_lengths_inner}")
            except:
                raise ValueError("Coefficients not input correctly")

        if vectorized:
            zero_coeffs = zeros((N, N))
        else:
            zero_coeffs = [[0.0]*N for _ in range(N)]

        if lambda_coeffs is not None:
            if vectorized:
                self.lambda_as = array(lambda_coeffs[:,:,0], order='C', copy=True)
                self.lambda_bs = array(lambda_coeffs[:,:,1], order='C', copy=True)
                self.lambda_cs = array(lambda_coeffs[:,:,2], order='C', copy=True)
                self.lambda_ds = array(lambda_coeffs[:,:,3], order='C', copy=True)
                self.lambda_es = array(lambda_coeffs[:,:,4], order='C', copy=True)
                self.lambda_fs = array(lambda_coeffs[:,:,5], order='C', copy=True)
            else:
                self.lambda_as = [[i[0] for i in l] for l in lambda_coeffs]
                self.lambda_bs = [[i[1] for i in l] for l in lambda_coeffs]
                self.lambda_cs = [[i[2] for i in l] for l in lambda_coeffs]
                self.lambda_ds = [[i[3] for i in l] for l in lambda_coeffs]
                self.lambda_es = [[i[4] for i in l] for l in lambda_coeffs]
                self.lambda_fs = [[i[5] for i in l] for l in lambda_coeffs]
        else:
            len_ABCDEF = len(ABCDEF)
            if len_ABCDEF == 0 or ABCDEF[0] is None:
                self.lambda_as = zero_coeffs
            else:
                self.lambda_as = ABCDEF[0]
            if len_ABCDEF < 2 or ABCDEF[1] is None:
                self.lambda_bs = zero_coeffs
            else:
                self.lambda_bs = ABCDEF[1]
            if len_ABCDEF < 3 or ABCDEF[2] is None:
                self.lambda_cs = zero_coeffs
            else:
                self.lambda_cs = ABCDEF[2]
            if len_ABCDEF < 4 or ABCDEF[3] is None:
                self.lambda_ds = zero_coeffs
            else:
                self.lambda_ds = ABCDEF[3]
            if len_ABCDEF < 5 or ABCDEF[4] is None:
                self.lambda_es = zero_coeffs
            else:
                self.lambda_es = ABCDEF[4]
            if len_ABCDEF < 6 or ABCDEF[5] is None:
                self.lambda_fs = zero_coeffs
            else:
                self.lambda_fs = ABCDEF[5]

        # Make an array of values identifying what coefficients are zero.
        # This may be useful for performance optimization in the future but is
        # especially important for reducing the size of the __repr__ string.
        self.lambda_coeffs_nonzero = lambda_coeffs_nonzero = ones(6, bool) if vectorized else [True]*6
        for k, coeffs in enumerate([self.lambda_as, self.lambda_bs, self.lambda_cs,
                           self.lambda_ds, self.lambda_es, self.lambda_fs]):
            nonzero = False
            for i in range(N):
                r = coeffs[i]
                for j in range(N):
                    if r[j] != 0.0:
                        nonzero = True
                        break
                if nonzero:
                    break

            lambda_coeffs_nonzero[k] = nonzero



    def __repr__(self):

        s = f'{self.__class__.__name__}(T={self.T!r}, xs={self.xs!r}'
        for i, attr in enumerate(self._model_attributes):
            if self.lambda_coeffs_nonzero[i]:
                s += f', {attr}={getattr(self, attr)}'
        s += ')'
        return s

    def to_T_xs(self, T, xs):
        r'''Method to construct a new :obj:`Wilson` instance at
        temperature `T`, and mole fractions `xs`
        with the same parameters as the existing object.

        Parameters
        ----------
        T : float
            Temperature, [K]
        xs : list[float]
            Mole fractions of each component, [-]

        Returns
        -------
        obj : Wilson
            New :obj:`Wilson` object at the specified conditions [-]

        Notes
        -----
        If the new temperature is the same temperature as the existing
        temperature, if the `lambda` terms or their derivatives have been
        calculated, they will be set to the new object as well.
        '''
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.vectorized = self.vectorized
        new.N = self.N
        (new.lambda_as, new.lambda_bs, new.lambda_cs,
         new.lambda_ds, new.lambda_es, new.lambda_fs) = (
                 self.lambda_as, self.lambda_bs, self.lambda_cs,
                 self.lambda_ds, self.lambda_es, self.lambda_fs)
        new.lambda_coeffs_nonzero = self.lambda_coeffs_nonzero

        if T == self.T:
            try:
                new._lambdas = self._lambdas
            except AttributeError:
                pass

            try:
                new._dlambdas_dT = self._dlambdas_dT
            except AttributeError:
                pass

            try:
                new._d2lambdas_dT2 = self._d2lambdas_dT2
            except AttributeError:
                pass

            try:
                new._d3lambdas_dT3 = self._d3lambdas_dT3
            except AttributeError:
                pass
        return new

    def lambdas(self):
        r'''Calculate and return the `lambda` terms for the Wilson model for
        at system temperature.

        .. math::
            \Lambda_{ij} = \exp\left[a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T
                    + d_{ij}T + \frac{e_{ij}}{T^2} + f_{ij}{T^2}\right]

        Returns
        -------
        lambdas : list[list[float]]
            Lambda terms, asymmetric matrix [-]

        Notes
        -----
        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._lambdas
        except AttributeError:
            pass

        N = self.N
        if self.vectorized:
            lambdas = zeros((N, N))
        else:
            lambdas = [[0.0]*N for _ in range(N)]

        lambdas = interaction_exp(self.T, N, self.lambda_as, self.lambda_bs,
                                  self.lambda_cs, self.lambda_ds,
                                  self.lambda_es, self.lambda_fs, lambdas)
        self._lambdas = lambdas
        return lambdas

    def dlambdas_dT(self):
        r'''Calculate and return the temperature derivative of the `lambda`
        terms for the Wilson model at the system temperature.

        .. math::
            \frac{\partial \Lambda_{ij}}{\partial T} =
            \left(2 T h_{ij} + d_{ij} + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}}
            - \frac{2 e_{ij}}{T^{3}}\right) e^{T^{2} h_{ij} + T d_{ij} + a_{ij}
            + c_{ij} \ln{\left(T \right)} + \frac{b_{ij}}{T}
            + \frac{e_{ij}}{T^{2}}}

        Returns
        -------
        dlambdas_dT : list[list[float]]
            Temperature deriavtives of Lambda terms, asymmetric matrix [1/K]

        Notes
        -----
        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._dlambdas_dT
        except AttributeError:
            pass

        B = self.lambda_bs
        C = self.lambda_cs
        D = self.lambda_ds
        E = self.lambda_es
        F = self.lambda_fs

        T, N = self.T, self.N
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        if self.vectorized:
            dlambdas_dT = zeros((N, N))
        else:
            dlambdas_dT = [[0.0]*N for _ in range(N)]

        self._dlambdas_dT = dinteraction_exp_dT(T, N, B, C, D, E, F, lambdas, dlambdas_dT)
        return dlambdas_dT

    def d2lambdas_dT2(self):
        r'''Calculate and return the second temperature derivative of the
        `lambda` termsfor the Wilson model at the system temperature.

        .. math::
            \frac{\partial^2 \Lambda_{ij}}{\partial^2 T} =
            \left(2 f_{ij} + \left(2 T f_{ij} + d_{ij} + \frac{c_{ij}}{T}
            - \frac{b_{ij}}{T^{2}} - \frac{2 e_{ij}}{T^{3}}\right)^{2}
                - \frac{c_{ij}}{T^{2}} + \frac{2 b_{ij}}{T^{3}}
                + \frac{6 e_{ij}}{T^{4}}\right) e^{T^{2} f_{ij} + T d_{ij}
                + a_{ij} + c_{ij} \ln{\left(T \right)} + \frac{b_{ij}}{T}
                + \frac{e_{ij}}{T^{2}}}

        Returns
        -------
        d2lambdas_dT2 : list[list[float]]
            Second temperature deriavtives of Lambda terms, asymmetric matrix,
            [1/K^2]

        Notes
        -----
        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._d2lambdas_dT2
        except AttributeError:
            pass
        T, N = self.T, self.N

        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        try:
            dlambdas_dT = self._dlambdas_dT
        except AttributeError:
            dlambdas_dT = self.dlambdas_dT()

        if not self.vectorized:
            d2lambdas_dT2 = [[0.0]*N for _ in range(N)]
        else:
            d2lambdas_dT2 = zeros((N, N))

        self._d2lambdas_dT2 = d2interaction_exp_dT2(T, N, self.lambda_bs,
                                                                     self.lambda_cs,
                                                                     self.lambda_es,
                                                                     self.lambda_fs,
                                                                     lambdas, dlambdas_dT, d2lambdas_dT2)
        return d2lambdas_dT2

    def d3lambdas_dT3(self):
        r'''Calculate and return the third temperature derivative of the
        `lambda` terms for the Wilson model at the system temperature.

        .. math::
            \frac{\partial^3 \Lambda_{ij}}{\partial^3 T} =
            \left(3 \left(2 f_{ij} - \frac{c_{ij}}{T^{2}} + \frac{2 b_{ij}}{T^{3}}
            + \frac{6 e_{ij}}{T^{4}}\right) \left(2 T f_{ij} + d_{ij}
            + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}} - \frac{2 e_{ij}}{T^{3}}\right)
            + \left(2 T f_{ij} + d_{ij} + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}}
            - \frac{2 e_{ij}}{T^{3}}\right)^{3} - \frac{2 \left(- c_{ij}
            + \frac{3 b_{ij}}{T} + \frac{12 e_{ij}}{T^{2}}\right)}{T^{3}}\right)
            e^{T^{2} f_{ij} + T d_{ij} + a_{ij} + c_{ij} \ln{\left(T \right)}
            + \frac{b_{ij}}{T} + \frac{e_{ij}}{T^{2}}}

        Returns
        -------
        d3lambdas_dT3 : list[list[float]]
            Third temperature deriavtives of Lambda terms, asymmetric matrix,
            [1/K^3]

        Notes
        -----
        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._d3lambdas_dT3
        except AttributeError:
            pass

        T, N = self.T, self.N
        lambda_bs = self.lambda_bs
        lambda_cs = self.lambda_cs
        lambda_es = self.lambda_es
        lambda_fs = self.lambda_fs

        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        try:
            dlambdas_dT = self._dlambdas_dT
        except AttributeError:
            dlambdas_dT = self.dlambdas_dT()

        if not self.vectorized:
            d3lambdas_dT3s = [[0.0]*N for _ in range(N)]
        else:
            d3lambdas_dT3s = zeros((N, N))

        self._d3lambdas_dT3 = d3interaction_exp_dT3(T, N, lambda_bs, lambda_cs, lambda_es,
                                                    lambda_fs, lambdas, dlambdas_dT, d3lambdas_dT3s)
        return d3lambdas_dT3s

    def xj_Lambda_ijs(self):
        '''
        '''
        try:
            return self._xj_Lambda_ijs
        except AttributeError:
            pass
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()

        if not self.vectorized:
            xj_Lambda_ijs = [0.0]*self.N
        else:
            xj_Lambda_ijs = zeros(self.N)

        self._xj_Lambda_ijs = wilson_xj_Lambda_ijs(self.xs, lambdas, self.N, xj_Lambda_ijs)
        return xj_Lambda_ijs

    def xj_Lambda_ijs_inv(self):
        '''
        '''
        try:
            return self._xj_Lambda_ijs_inv
        except AttributeError:
            pass

        try:
            xj_Lambda_ijs = self._xj_Lambda_ijs
        except AttributeError:
            xj_Lambda_ijs = self.xj_Lambda_ijs()
        if not self.vectorized:
            self._xj_Lambda_ijs_inv = [1.0/x for x in xj_Lambda_ijs]
        else:
            self._xj_Lambda_ijs_inv = 1.0/xj_Lambda_ijs
        return self._xj_Lambda_ijs_inv

    def log_xj_Lambda_ijs(self):
        '''
        '''
        try:
            return self._log_xj_Lambda_ijs
        except AttributeError:
            pass
        try:
            xj_Lambda_ijs = self._xj_Lambda_ijs
        except AttributeError:
            xj_Lambda_ijs = self.xj_Lambda_ijs()
        if not self.vectorized:
            self._log_xj_Lambda_ijs = [log(i) for i in xj_Lambda_ijs]
        else:
            self._log_xj_Lambda_ijs = nplog(xj_Lambda_ijs)
        return self._log_xj_Lambda_ijs


    def xj_dLambda_dTijs(self):
        '''
        '''
        try:
            return self._xj_dLambda_dTijs
        except AttributeError:
            pass
        try:
            dlambdas_dT = self._dlambdas_dT
        except AttributeError:
            dlambdas_dT = self.dlambdas_dT()

        if not self.vectorized:
            xj_dLambda_dTijs = [0.0]*self.N
        else:
            xj_dLambda_dTijs = zeros(self.N)

        self._xj_dLambda_dTijs = wilson_xj_Lambda_ijs(self.xs, dlambdas_dT, self.N, xj_dLambda_dTijs)
        return xj_dLambda_dTijs


    def xj_d2Lambda_dT2ijs(self):
        '''
        '''
        try:
            return self._xj_d2Lambda_dT2ijs
        except AttributeError:
            pass
        try:
            d2lambdas_dT2 = self._d2lambdas_dT2
        except AttributeError:
            d2lambdas_dT2 = self.d2lambdas_dT2()

        if not self.vectorized:
            xj_d2Lambda_dT2ijs = [0.0]*self.N
        else:
            xj_d2Lambda_dT2ijs = zeros(self.N)

        self._xj_d2Lambda_dT2ijs = wilson_xj_Lambda_ijs(self.xs, d2lambdas_dT2, self.N, xj_d2Lambda_dT2ijs)
        return xj_d2Lambda_dT2ijs

    def xj_d3Lambda_dT3ijs(self):
        '''
        '''
        try:
            return self._xj_d3Lambda_dT3ijs
        except AttributeError:
            pass
        try:
            d3lambdas_dT3 = self._d3lambdas_dT3
        except AttributeError:
            d3lambdas_dT3 = self.d3lambdas_dT3()

        if not self.vectorized:
            xj_d3Lambda_dT3ijs = [0.0]*self.N
        else:
            xj_d3Lambda_dT3ijs = zeros(self.N)

        self._xj_d3Lambda_dT3ijs = wilson_xj_Lambda_ijs(self.xs, d3lambdas_dT3, self.N, xj_d3Lambda_dT3ijs)
        return xj_d3Lambda_dT3ijs


    def GE(self):
        r'''Calculate and return the excess Gibbs energy of a liquid phase
        represented with the Wilson model.

        .. math::
            g^E = -RT\sum_i x_i \ln\left(\sum_j x_j \lambda_{i,j} \right)

        Returns
        -------
        GE : float
            Excess Gibbs energy of an ideal liquid, [J/mol]

        Notes
        -----
        '''
        try:
            return self._GE
        except AttributeError:
            pass

        try:
            log_xj_Lambda_ijs = self._log_xj_Lambda_ijs
        except AttributeError:
            log_xj_Lambda_ijs = self.log_xj_Lambda_ijs()

        if not self.vectorized:
            xs, N = self.xs, self.N
            GE = 0.0
            for i in range(N):
                GE += xs[i]*log_xj_Lambda_ijs[i]
        else:
            GE = float((self.xs*log_xj_Lambda_ijs).sum())
        self._GE = GE = -GE*R*self.T
        return GE

    def dGE_dT(self):
        r'''Calculate and return the temperature derivative of excess Gibbs
        energy of a liquid phase represented by the Wilson model.

        .. math::
            \frac{\partial G^E}{\partial T} = -R\sum_i x_i \ln\left(\sum_j x_i \Lambda_{ij}\right)
            -RT\sum_i \frac{x_i \sum_j x_j \frac{\Lambda _{ij}}{\partial T}}{\sum_j x_j \Lambda_{ij}}

        Returns
        -------
        dGE_dT : float
            First temperature derivative of excess Gibbs energy of a
            liquid phase represented by the Wilson model, [J/(mol*K)]

        Notes
        -----
        '''# Derived with:
        """from sympy import *
        N = 4
        R, T = symbols('R, T')
        x1, x2, x3, x4 = symbols('x1, x2, x3, x4')
        xs = [x1, x2, x3, x4]

        Lambda11, Lambda12, Lambda13, Lambda14, Lambda21, Lambda22, Lambda23, Lambda24, Lambda31, Lambda32, Lambda33, Lambda34, Lambda41, Lambda42, Lambda43, Lambda44 = symbols(
            'Lambda11, Lambda12, Lambda13, Lambda14, Lambda21, Lambda22, Lambda23, Lambda24, Lambda31, Lambda32, Lambda33, Lambda34, Lambda41, Lambda42, Lambda43, Lambda44', cls=Function)
        Lambda_ijs = [[Lambda11(T), Lambda12(T), Lambda13(T), Lambda14(T)],
                   [Lambda21(T), Lambda22(T), Lambda23(T), Lambda24(T)],
                   [Lambda31(T), Lambda32(T), Lambda33(T), Lambda34(T)],
                   [Lambda41(T), Lambda42(T), Lambda43(T), Lambda44(T)]]
        ge = 0
        for i in range(N):
            num = 0
            for j in range(N):
                num += Lambda_ijs[i][j]*xs[j]
            ge -= xs[i]*log(num)
        ge = ge*R*T

        diff(ge, T)
        """
        try:
            return self._dGE_dT
        except AttributeError:
            pass
        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()

        try:
            xj_dLambda_dTijs = self._xj_dLambda_dTijs
        except AttributeError:
            xj_dLambda_dTijs = self.xj_dLambda_dTijs()

        try:
            GE = self._GE
        except AttributeError:
            GE = self.GE()
        self._dGE_dT = dGE_dT = wilson_dGE_dT(self.xs, self.T, GE, self.N, xj_Lambda_ijs_inv, xj_dLambda_dTijs)
        return dGE_dT

    def d2GE_dT2(self):
        r'''Calculate and return the second temperature derivative of excess
        Gibbs energy of a liquid phase using the Wilson activity coefficient model.

        .. math::
            \frac{\partial^2 G^E}{\partial T^2} = -R\left[T\sum_i \left(\frac{x_i \sum_j (x_j \frac{\partial^2 \Lambda_{ij}}{\partial T^2} )}{\sum_j x_j \Lambda_{ij}}
            - \frac{x_i (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T}  )^2}{(\sum_j x_j \Lambda_{ij})^2}
            \right)
            + 2\sum_i \left(\frac{x_i \sum_j  x_j \frac{\partial \Lambda_{ij}}{\partial T}}{\sum_j x_j \Lambda_{ij}}
            \right)
            \right]

        Returns
        -------
        d2GE_dT2 : float
            Second temperature derivative of excess Gibbs energy, [J/(mol*K^2)]

        Notes
        -----
        '''
        try:
            return self._d2GE_dT2
        except AttributeError:
            pass

        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()

        try:
            xj_dLambda_dTijs = self._xj_dLambda_dTijs
        except AttributeError:
            xj_dLambda_dTijs = self.xj_dLambda_dTijs()
        try:
            xj_d2Lambda_dT2ijs = self._xj_d2Lambda_dT2ijs
        except AttributeError:
            xj_d2Lambda_dT2ijs = self.xj_d2Lambda_dT2ijs()

        self._d2GE_dT2 = wilson_d2GE_dT2(self.xs, self.T, self.N, xj_Lambda_ijs_inv, xj_dLambda_dTijs, xj_d2Lambda_dT2ijs)
        return self._d2GE_dT2

    def d3GE_dT3(self):
        r'''Calculate and return the third temperature derivative of excess
        Gibbs energy of a liquid phase using the Wilson activity coefficient
        model.

        .. math::
            \frac{\partial^3 G^E}{\partial T^3} = -R\left[3\left(\frac{x_i \sum_j (x_j \frac{\partial^2 \Lambda_{ij}}{\partial T^2} )}{\sum_j x_j \Lambda_{ij}}
            - \frac{x_i (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T}  )^2}{(\sum_j x_j \Lambda_{ij})^2}
            \right)
            +T\left(
            \sum_i \frac{x_i (\sum_j x_j \frac{\partial^3 \Lambda _{ij}}{\partial T^3})}{\sum_j x_j \Lambda_{ij}}
            - \frac{3x_i (\sum_j x_j \frac{\partial \Lambda_{ij}^2}{\partial T^2})  (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T})}{(\sum_j x_j \Lambda_{ij})^2}
            + 2\frac{x_i(\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T})^3}{(\sum_j x_j \Lambda_{ij})^3}
            \right)\right]

        Returns
        -------
        d3GE_dT3 : float
            Third temperature derivative of excess Gibbs energy, [J/(mol*K^3)]

        Notes
        -----
        '''
        try:
            return self._d3GE_dT3
        except AttributeError:
            pass

        xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()
        xj_dLambda_dTijs = self.xj_dLambda_dTijs()
        xj_d2Lambda_dT2ijs = self.xj_d2Lambda_dT2ijs()
        xj_d3Lambda_dT3ijs = self.xj_d3Lambda_dT3ijs()

        self._d3GE_dT3 = wilson_d3GE_dT3(self.xs, self.T, self.N, xj_Lambda_ijs_inv, xj_dLambda_dTijs,
                                          xj_d2Lambda_dT2ijs, xj_d3Lambda_dT3ijs)
        return self._d3GE_dT3


    def d2GE_dTdxs(self):
        r'''Calculate and return the temperature derivative of mole fraction
        derivatives of excess Gibbs energy of a liquid represented by the
        Wilson model.

        .. math::
            \frac{\partial^2 G^E}{\partial x_k \partial T} = -R\left[T\left(
            \sum_i  \left(\frac{x_i \frac{\partial n_{ik}}{\partial T}}{\sum_j x_j \Lambda_{ij}}
            - \frac{x_i \Lambda_{ik} (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T} )}{(\partial_j x_j \Lambda_{ij})^2}
            \right) + \frac{\sum_i x_i \frac{\partial \Lambda_{ki}}{\partial T}}{\sum_j x_j \Lambda_{kj}}
            \right)
            + \ln\left(\sum_i x_i \Lambda_{ki}\right)
            + \sum_i \frac{x_i \Lambda_{ik}}{\sum_j x_j \Lambda_{ij}}
            \right]

        Returns
        -------
        d2GE_dTdxs : list[float]
            Temperature derivative of mole fraction derivatives of excess Gibbs
            energy, [J/mol/K]

        Notes
        -----
        '''
        try:
            return self._d2GE_dTdxs
        except AttributeError:
            pass

        try:
            log_xj_Lambda_ijs = self._log_xj_Lambda_ijs
        except AttributeError:
            log_xj_Lambda_ijs = self.log_xj_Lambda_ijs()
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        try:
            dlambdas_dT = self._dlambdas_dT
        except AttributeError:
            dlambdas_dT = self.dlambdas_dT()
        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()
        try:
            xj_dLambda_dTijs = self._xj_dLambda_dTijs
        except AttributeError:
            xj_dLambda_dTijs = self.xj_dLambda_dTijs()
        if not self.vectorized:
            d2GE_dTdxs = [0.0]*self.N
        else:
            d2GE_dTdxs = zeros(self.N)

        wilson_d2GE_dTdxs(self.xs, self.T, self.N, log_xj_Lambda_ijs,
                                       lambdas, dlambdas_dT,
                                       xj_Lambda_ijs_inv, xj_dLambda_dTijs, d2GE_dTdxs)
        self._d2GE_dTdxs = d2GE_dTdxs
        return d2GE_dTdxs


    def dGE_dxs(self):
        r'''Calculate and return the mole fraction derivatives of excess Gibbs
        energy for the Wilson model.

        .. math::
            \frac{\partial G^E}{\partial x_k} = -RT\left[
            \sum_i \frac{x_i \Lambda_{ik}}{\sum_j \Lambda_{ij}x_j }
            + \ln\left(\sum_j x_j \Lambda_{kj}\right)
            \right]

        Returns
        -------
        dGE_dxs : list[float]
            Mole fraction derivatives of excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        """
        from sympy import *
        N = 4
        R, T = symbols('R, T')
        x1, x2, x3, x4 = symbols('x1, x2, x3, x4')
        xs = [x1, x2, x3, x4]

        Lambda11, Lambda12, Lambda13, Lambda14, Lambda21, Lambda22, Lambda23, Lambda24, Lambda31, Lambda32, Lambda33, Lambda34, Lambda41, Lambda42, Lambda43, Lambda44 = symbols(
            'Lambda11, Lambda12, Lambda13, Lambda14, Lambda21, Lambda22, Lambda23, Lambda24, Lambda31, Lambda32, Lambda33, Lambda34, Lambda41, Lambda42, Lambda43, Lambda44', cls=Function)
        Lambda_ijs = [[Lambda11(T), Lambda12(T), Lambda13(T), Lambda14(T)],
                   [Lambda21(T), Lambda22(T), Lambda23(T), Lambda24(T)],
                   [Lambda31(T), Lambda32(T), Lambda33(T), Lambda34(T)],
                   [Lambda41(T), Lambda42(T), Lambda43(T), Lambda44(T)]]
        ge = 0
        for i in range(N):
            num = 0
            for j in range(N):
                num += Lambda_ijs[i][j]*xs[j]
            ge -= xs[i]*log(num)
        ge = ge*R*T


        diff(ge, x1)#, diff(ge, x1, x2), diff(ge, x1, x2, x3)
        """
        try:
            return self._dGE_dxs
        except AttributeError:
            pass
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        try:
            log_xj_Lambda_ijs = self._log_xj_Lambda_ijs
        except AttributeError:
            log_xj_Lambda_ijs = self.log_xj_Lambda_ijs()
        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()

        if not self.vectorized:
            dGE_dxs = [0.0]*self.N
        else:
            dGE_dxs = zeros(self.N)

        dGE_dxs = wilson_dGE_dxs(self.xs, self.T, self.N, log_xj_Lambda_ijs, lambdas, xj_Lambda_ijs_inv, dGE_dxs)
        self._dGE_dxs = dGE_dxs
        return dGE_dxs

    def d2GE_dxixjs(self):
        r'''Calculate and return the second mole fraction derivatives of excess
        Gibbs energy for the Wilson model.

        .. math::
            \frac{\partial^2 G^E}{\partial x_k \partial x_m} = RT\left(
            \sum_i \frac{x_i \Lambda_{ik} \Lambda_{im}}{(\sum_j x_j \Lambda_{ij})^2}
            -\frac{\Lambda_{km}}{\sum_j x_j \Lambda_{kj}}
            -\frac{\Lambda_{mk}}{\sum_j x_j \Lambda_{mj}}
            \right)

        Returns
        -------
        d2GE_dxixjs : list[list[float]]
            Second mole fraction derivatives of excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        try:
            return self._d2GE_dxixjs
        except AttributeError:
            pass
        # Correct, tested with hessian
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()
        N = self.N
        if not self.vectorized:
            d2GE_dxixjs = [[0.0]*N for _ in range(N)]
        else:
            d2GE_dxixjs = zeros((N, N))

        d2GE_dxixjs = wilson_d2GE_dxixjs(self.xs, self.T, N, lambdas, xj_Lambda_ijs_inv, d2GE_dxixjs)
        self._d2GE_dxixjs = d2GE_dxixjs
        return d2GE_dxixjs

    def d3GE_dxixjxks(self):
        r'''Calculate and return the third mole fraction derivatives of excess
        Gibbs energy using the Wilson model.

        .. math::
            \frac{\partial^3 G^E}{\partial x_k \partial x_m \partial x_n}
            = -RT\left[
            \sum_i \left(\frac{2x_i \Lambda_{ik}\Lambda_{im}\Lambda_{in}} {(\sum x_j \Lambda_{ij})^3}\right)
            - \frac{\Lambda_{km} \Lambda_{kn}}{(\sum_j x_j \Lambda_{kj})^2}
            - \frac{\Lambda_{mk} \Lambda_{mn}}{(\sum_j x_j \Lambda_{mj})^2}
            - \frac{\Lambda_{nk} \Lambda_{nm}}{(\sum_j x_j \Lambda_{nj})^2}
            \right]

        Returns
        -------
        d3GE_dxixjxks : list[list[list[float]]]
            Third mole fraction derivatives of excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        try:
            return self._d3GE_dxixjxks
        except AttributeError:
            pass
        # Correct, tested with sympy expanding
        lambdas = self.lambdas()
        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()

        N = self.N
        if not self.vectorized:
            d3GE_dxixjxks = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d3GE_dxixjxks = zeros((N, N, N))

        # all the same: analytical[i][j][k] = analytical[i][k][j] = analytical[j][i][k] = analytical[j][k][i] = analytical[k][i][j] = analytical[k][j][i] = float(v)
        d3GE_dxixjxks = wilson_d3GE_dxixjxks(self.xs, self.T, self.N, lambdas, xj_Lambda_ijs_inv, d3GE_dxixjxks)
        self._d3GE_dxixjxks = d3GE_dxixjxks
        return d3GE_dxixjxks


    def gammas(self):
        # With this formula implemented, dgammas_dxs cannot be calculated
        # numerically.
        # Don't bother documenting or exposing; implemented only for a bit more
        # speed and precision.
        try:
            return self._gammas
        except AttributeError:
            pass
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()

        if not self.vectorized:
            gammas = [0.0]*self.N
        else:
            gammas = zeros(self.N)

        wilson_gammas(self.xs, self.N, lambdas, xj_Lambda_ijs_inv, gammas)
        self._gammas = gammas
        return gammas

    @classmethod
    def regress_binary_parameters(cls, gammas, xs, use_numba=False,
                                  do_statistics=True, **kwargs):
        # Load the functions either locally or with numba
        if use_numba:
            from thermo.numba import wilson_gammas_binaries as work_func
            from thermo.numba import wilson_gammas_binaries_jac as jac_func
        else:
            work_func = wilson_gammas_binaries
            jac_func = wilson_gammas_binaries_jac

        # Allocate all working memory
        pts = len(xs)
        pts2 = pts*2
        gammas_iter, jac_iter = zeros(pts2), zeros((pts2, 2))

        # Plain objective functions
        def fitting_func(xs, lambda12, lambda21):
            return work_func(xs, lambda12, lambda21, gammas_iter)

        def analytical_jac(xs, lambda12, lambda21):
            return jac_func(xs, lambda12, lambda21, jac_iter)

        # The extend calls has been tested to be the fastest compared to numpy and list comprehension
        xs_working = []
        for xsi in xs:
            xs_working.extend(xsi)
        gammas_working = []
        for gammasi in gammas:
            gammas_working.extend(gammasi)

        xs_working = array(xs_working)
        gammas_working = array(gammas_working)

        # Objective functions for leastsq maximum speed
        def func_wrapped_for_leastsq(params):
            return work_func(xs_working, params[0], params[1], gammas_iter) - gammas_working

        def jac_wrapped_for_leastsq(params):
            return jac_func(xs_working, params[0], params[1], jac_iter)


        fit_parameters = ['lambda12', 'lambda21']
        return GibbsExcess._regress_binary_parameters(gammas_working, xs_working, fitting_func=fitting_func,
                                                      fit_parameters=fit_parameters,
                                                      use_fit_parameters=fit_parameters,
                                                      initial_guesses=cls._gamma_parameter_guesses,
                                                      analytical_jac=analytical_jac,
                                                      use_numba=use_numba,
                                                      do_statistics=do_statistics,
                                                      func_wrapped_for_leastsq=func_wrapped_for_leastsq,
                                                      jac_wrapped_for_leastsq=jac_wrapped_for_leastsq,
                                                      **kwargs)


    # Larger value on the right always
    _gamma_parameter_guesses = [{'lambda12': 1, 'lambda21': 1},
                               {'lambda12': 2.2, 'lambda21': 3.0},
                               {'lambda12': 0.015, 'lambda21': 37.0},
                               {'lambda12': 0.5, 'lambda21': 40.0},
                               {'lambda12': 1e-7, 'lambda21': .5},
                               {'lambda12': 1e-12, 'lambda21': 1.9},
                               {'lambda12': 1e-12, 'lambda21': 10.0},
                               ]
    for i in range(len(_gamma_parameter_guesses)):
        r = _gamma_parameter_guesses[i]
        _gamma_parameter_guesses.append({'lambda12': r['lambda21'], 'lambda21': r['lambda12']})
    del i, r

def Wilson_gammas(xs, params):
    r'''Calculates the activity coefficients of each species in a mixture
    using the Wilson method, given their mole fractions, and
    dimensionless interaction parameters. Those are normally correlated with
    temperature, and need to be calculated separately.

    .. math::
        \ln \gamma_i = 1 - \ln \left(\sum_j^N \Lambda_{ij} x_j\right)
        -\sum_j^N \frac{\Lambda_{ji}x_j}{\displaystyle\sum_k^N \Lambda_{jk}x_k}

    Parameters
    ----------
    xs : list[float]
        Liquid mole fractions of each species, [-]
    params : list[list[float]]
        Dimensionless interaction parameters of each compound with each other,
        [-]

    Returns
    -------
    gammas : list[float]
        Activity coefficient for each species in the liquid mixture, [-]

    Notes
    -----
    This model needs N^2 parameters.

    The original model correlated the interaction parameters using the standard
    pure-component molar volumes of each species at 25°C, in the following form:

    .. math::
        \Lambda_{ij} = \frac{V_j}{V_i} \exp\left(\frac{-\lambda_{i,j}}{RT}\right)

    If a compound is not liquid at that temperature, the liquid volume is taken
    at the saturated pressure; and if the component is supercritical, its
    liquid molar volume should be extrapolated to 25°C.

    However, that form has less flexibility and offered no advantage over
    using only regressed parameters.

    Most correlations for the interaction parameters include some of the terms
    shown in the following form:

    .. math::
        \ln \Lambda_{ij} =a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T + d_{ij}T
        + \frac{e_{ij}}{T^2} + h_{ij}{T^2}

    The Wilson model is not applicable to liquid-liquid systems.

    For this model to produce ideal acitivty coefficients (gammas = 1),
    all interaction parameters should be 1.

    The specific process simulator implementations are as follows:

    Examples
    --------
    Ethanol-water example, at 343.15 K and 1 MPa, from [2]_ also posted online
    http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/P05.01a%20VLE%20Behavior%20of%20Ethanol%20-%20Water%20Using%20Wilson.xps
    :

    >>> Wilson_gammas([0.252, 0.748], [[1, 0.154], [0.888, 1]])
    [1.881492608717, 1.165577493112]

    References
    ----------
    .. [1] Wilson, Grant M. "Vapor-Liquid Equilibrium. XI. A New Expression for
       the Excess Free Energy of Mixing." Journal of the American Chemical
       Society 86, no. 2 (January 1, 1964): 127-130. doi:10.1021/ja01056a002.
    .. [2] Gmehling, Jurgen, Barbel Kolbe, Michael Kleiber, and Jurgen Rarey.
       Chemical Thermodynamics for Process Simulation. 1st edition. Weinheim:
       Wiley-VCH, 2012.
    '''
    gammas = []
    cmps = range(len(xs))

    sums0 = []
    for j in cmps:
        tot = 0.0
        paramsj = params[j]
        for k in cmps:
            tot += paramsj[k]*xs[k]
        sums0.append(tot)

    for i in cmps:
        tot2 = 0.
        for j in cmps:
            tot2 += params[j][i]*xs[j]/sums0[j]

        gamma = exp(1. - log(sums0[i]) - tot2)
        gammas.append(gamma)
    return gammas

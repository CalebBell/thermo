# -*- coding: utf-8 -*-
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


'''

from __future__ import division
from math import log, exp
from fluids.constants import R
from fluids.numerics import numpy as np
from thermo.activity import GibbsExcess, interaction_exp, dinteraction_exp_dT, d2interaction_exp_dT2, d3interaction_exp_dT3

try:
    array, zeros, npsum, nplog = np.array, np.zeros, np.sum, np.log
except (ImportError, AttributeError):
    pass

__all__ = ['Wilson', 'Wilson_gammas']


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

    for i in range(N):
        tot2 = 1.0
        for j in range(N):
            tot2 -= lambdas[j][i]*vec0[j]
        gammas[i] = exp(tot2)*xj_Lambda_ijs_inv[i]

    return gammas

class Wilson(GibbsExcess):
    r'''Class for representing an a liquid with excess gibbs energy represented
    by the Wilson equation. This model is capable of representing most
    nonideal liquids for vapor-liquid equilibria, but is not recommended for
    liquid-liquid equilibria.

    Parameters
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]
    lambda_coeffs : list[list[list[float]]], optional
        Wilson parameters, indexed by [i][j] and then each value is a 6
        element list with parameters [`a`, `b`, `c`, `d`, `e`, `f`];
        either `lambda_coeffs` or `ABCDEF` are required, [-]
    ABCDEF : tuple[list[list[float]], 6], optional
        Contains the following. One of `lambda_coeffs` or `ABCDEF` are
        required, [-]

        a : list[list[float]]
            `a` parameters used in calculating :obj:`Wilson.lambdas`, [-]
        b : list[list[float]]
            `b` parameters used in calculating :obj:`Wilson.lambdas`, [K]
        c : list[list[float]]
            `c` parameters used in calculating :obj:`Wilson.lambdas`, [-]
        d : list[list[float]]
            `d` paraemeters used in calculating :obj:`Wilson.lambdas`, [1/K]
        e : list[list[float]]
            `e` parameters used in calculating :obj:`Wilson.lambdas`, [K^2]
        f : list[list[float]]
            `f` parameters used in calculating :obj:`Wilson.lambdas`, [1/K^2]

    Attributes
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]

    Notes
    -----
    In addition to the methods presented here, the methods of its base class
    :obj:`thermo.activity.GibbsExcess` are available as well.

    Examples
    --------
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
    >>> GE = Wilson(T=T, xs=xs, ABCDEF=params)
    >>> GE
    Wilson(T=331.42, xs=[0.229, 0.175, 0.596], ABCDEF=([[0.0, 3.870101271243586, 0.07939943395502425], [-6.491263271243587, 0.0, -3.276991837288562], [0.8542855660449756, 6.906801837288562, 0.0]], [[0.0, -375.2835, -31.1208], [1722.58, 0.0, 1140.79], [-747.217, -3596.17, -0.0]], [[-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0]], [[-0.0, -0.00791073, -0.000868371], [0.00747788, -0.0, -3.1e-05], [0.00124796, -3e-05, -0.0]], [[-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0]], [[-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0]]))
    >>> GE.GE(), GE.dGE_dT(), GE.d2GE_dT2()
    (480.2639266306882, 4.355962766232997, -0.029130384525017247)
    >>> GE.HE(), GE.SE(), GE.dHE_dT(), GE.dSE_dT()
    (-963.3892533542517, -4.355962766232997, 9.654392039281216, 0.029130384525017247)
    >>> GE.gammas()
    [1.2233934334, 1.100945902470, 1.205289928117]


    The solution given by the DDBST has the same values [1.223, 1.101, 1.205],
    and can be found here:
    http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/05.09%20Compare%20Experimental%20VLE%20to%20Wilson%20Equation%20Results.xps


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
    >>> GE = Wilson(T=T, xs=xs, ABCDEF=params)
    >>> GE.gammas()
    [2.124064516, 1.1903745834]

    The activity coefficients given in [1]_ are [2.1244, 1.1904]; matching (
    with a slight error from their use of 1.987 as a gas constant).

    References
    ----------x
    .. [1] Smith, H. C. Van Ness Joseph M. Introduction to Chemical Engineering
       Thermodynamics 4th Edition, Joseph M. Smith, H. C. Van
       Ness, 1987.
    '''
    model_id = 200
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
    def from_DDBST_as_matrix(Vs, ais, bis, cis, dis, eis, fis,
                             unit_conversion=True):
        r'''Converts parameters for the wilson equation in the DDBST to the
        basis used in this implementation. Matrix wrapper around
        :obj:`Wilson.from_DDBST`.

        Parameters
        ----------
        Vs : list[float]
            Molar volume of component; needs only to be in consistent units,
            [cm^3/mol]
        a : list[list[float]]
            `a` parameters in DDBST form, [K]
        b : list[list[float]]
            `b` parameters in DDBST form, [-]
        c : list[list[float]]
            `c` parameters in DDBST form, [1/K]
        d : list[list[float]], optional
            `d` parameters in DDBST form, [-]
        e : list[list[float]], optional
            `e` parameters in DDBST form, [1/K^2]
        f : list[list[float]], optional
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
        cmps = range(len(Vs))
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

    def __init__(self, T, xs, lambda_coeffs=None, ABCDEF=None):
        self.T = T
        self.xs = xs
        self.scalar = scalar = type(xs) is list
        if ABCDEF is not None:
            (self.lambda_coeffs_A, self.lambda_coeffs_B, self.lambda_coeffs_C,
            self.lambda_coeffs_D, self.lambda_coeffs_E, self.lambda_coeffs_F) = ABCDEF
            self.N = N = len(self.lambda_coeffs_A)
        else:
            if lambda_coeffs is not None:
                if scalar:
                    self.lambda_coeffs_A = [[i[0] for i in l] for l in lambda_coeffs]
                    self.lambda_coeffs_B = [[i[1] for i in l] for l in lambda_coeffs]
                    self.lambda_coeffs_C = [[i[2] for i in l] for l in lambda_coeffs]
                    self.lambda_coeffs_D = [[i[3] for i in l] for l in lambda_coeffs]
                    self.lambda_coeffs_E = [[i[4] for i in l] for l in lambda_coeffs]
                    self.lambda_coeffs_F = [[i[5] for i in l] for l in lambda_coeffs]
                else:
                    self.lambda_coeffs_A = array(lambda_coeffs[:,:,0], order='C', copy=True)
                    self.lambda_coeffs_B = array(lambda_coeffs[:,:,1], order='C', copy=True)
                    self.lambda_coeffs_C = array(lambda_coeffs[:,:,2], order='C', copy=True)
                    self.lambda_coeffs_D = array(lambda_coeffs[:,:,3], order='C', copy=True)
                    self.lambda_coeffs_E = array(lambda_coeffs[:,:,4], order='C', copy=True)
                    self.lambda_coeffs_F = array(lambda_coeffs[:,:,5], order='C', copy=True)

            else:
                raise ValueError("`lambda_coeffs` or `ABCDEF` is required required")
            self.N = N = len(lambda_coeffs)

    model_attriubtes = ('lambda_coeffs_A', 'lambda_coeffs_B', 'lambda_coeffs_C',
                        'lambda_coeffs_D', 'lambda_coeffs_E', 'lambda_coeffs_F')

    def __repr__(self):
        s = '%s(T=%s, xs=%s, ABCDEF=%s)' %(self.__class__.__name__, repr(self.T), repr(self.xs),
                (self.lambda_coeffs_A,  self.lambda_coeffs_B, self.lambda_coeffs_C,
                 self.lambda_coeffs_D, self.lambda_coeffs_E, self.lambda_coeffs_F))
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
        new.scalar = self.scalar
        new.N = self.N
        (new.lambda_coeffs_A, new.lambda_coeffs_B, new.lambda_coeffs_C,
         new.lambda_coeffs_D, new.lambda_coeffs_E, new.lambda_coeffs_F) = (
                 self.lambda_coeffs_A, self.lambda_coeffs_B, self.lambda_coeffs_C,
                 self.lambda_coeffs_D, self.lambda_coeffs_E, self.lambda_coeffs_F)

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
        if self.scalar:
            lambdas = [[0.0]*N for _ in range(N)]
        else:
            lambdas = zeros((N, N))

        lambdas = interaction_exp(self.T, N, self.lambda_coeffs_A, self.lambda_coeffs_B,
                                  self.lambda_coeffs_C, self.lambda_coeffs_D,
                                  self.lambda_coeffs_E, self.lambda_coeffs_F, lambdas)
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

        B = self.lambda_coeffs_B
        C = self.lambda_coeffs_C
        D = self.lambda_coeffs_D
        E = self.lambda_coeffs_E
        F = self.lambda_coeffs_F

        T, N = self.T, self.N
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        if self.scalar:
            dlambdas_dT = [[0.0]*N for _ in range(N)]
        else:
            dlambdas_dT = zeros((N, N))

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

        if self.scalar:
            d2lambdas_dT2 = [[0.0]*N for _ in range(N)]
        else:
            d2lambdas_dT2 = zeros((N, N))

        self._d2lambdas_dT2 = d2interaction_exp_dT2(T, N, self.lambda_coeffs_B,
                                                                     self.lambda_coeffs_C,
                                                                     self.lambda_coeffs_E,
                                                                     self.lambda_coeffs_F,
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
        lambda_coeffs_B = self.lambda_coeffs_B
        lambda_coeffs_C = self.lambda_coeffs_C
        lambda_coeffs_E = self.lambda_coeffs_E
        lambda_coeffs_F = self.lambda_coeffs_F

        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        try:
            dlambdas_dT = self._dlambdas_dT
        except AttributeError:
            dlambdas_dT = self.dlambdas_dT()

        if self.scalar:
            d3lambdas_dT3s = [[0.0]*N for _ in range(N)]
        else:
            d3lambdas_dT3s = zeros((N, N))

        self._d3lambdas_dT3 = d3interaction_exp_dT3(T, N, lambda_coeffs_B, lambda_coeffs_C, lambda_coeffs_E,
                                                    lambda_coeffs_F, lambdas, dlambdas_dT, d3lambdas_dT3s)
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

        if self.scalar:
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
        if self.scalar:
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
        if self.scalar:
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

        if self.scalar:
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

        if self.scalar:
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

        if self.scalar:
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

        if self.scalar:
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
        '''from sympy import *
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
        '''
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
        if self.scalar:
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
        '''
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
        '''
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

        if self.scalar:
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
        if self.scalar:
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
        if self.scalar:
            d3GE_dxixjxks = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d3GE_dxixjxks = zeros((N, N, N))

        # all the same: analytical[i][j][k] = analytical[i][k][j] = analytical[j][i][k] = analytical[j][k][i] = analytical[k][i][j] = analytical[k][j][i] = float(v)
        d3GE_dxixjxks = wilson_d3GE_dxixjxks(self.xs, self.T, self.N, lambdas, xj_Lambda_ijs_inv, d3GE_dxixjxks)
        self._d3GE_dxixjxks = d3GE_dxixjxks
        return d3GE_dxixjxks


    def gammas(self):
        # Don't bother documenting or exposing; implemented only for a bit more
        # speed and precision.
        try:
            return self._gammas
        except AttributeError:
            pass
#        xs, cmps = self.xs, self.cmps
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()

        if self.scalar:
            gammas = [0.0]*self.N
        else:
            gammas = zeros(self.N)

        wilson_gammas(self.xs, self.N, lambdas, xj_Lambda_ijs_inv, gammas)
        self._gammas = gammas
        return gammas


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
    Ethanol-water example, at 343.15 K and 1 MPa:

    >>> Wilson_gammas([0.252, 0.748], [[1, 0.154], [0.888, 1]])
    [1.8814926087178843, 1.1655774931125487]

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

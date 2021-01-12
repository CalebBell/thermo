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

This module contains a class :obj:`RegularSolution` for performing activity coefficient
calculations with the regular solution model.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Regular Solution Class
======================

.. autoclass:: RegularSolution
    :members: to_T_xs, GE, dGE_dT, d2GE_dT2, d3GE_dT3, d2GE_dTdxs, dGE_dxs, d2GE_dxixjs, d3GE_dxixjxks
    :undoc-members:
    :show-inheritance:
    :exclude-members:
'''

from __future__ import division
from thermo.activity import GibbsExcess
from chemicals.utils import log, exp
from fluids.constants import R
from fluids.numerics import numpy as np

try:
    zeros = np.zeros
except:
    pass

__all__ = ['RegularSolution']


def regular_solution_GE(SPs, coeffs, xsVs, N, xsVs_sum_inv):
    num = 0.0
    for i in range(N):
        coeffsi = coeffs[i]
        for j in range(N):
            SPi_m_SPj = SPs[i] - SPs[j]
            Aij = 0.5*SPi_m_SPj*SPi_m_SPj + coeffsi[j]*SPs[i]*SPs[j]
            num += xsVs[i]*xsVs[j]*Aij
    GE = num*xsVs_sum_inv
    return GE


def regular_solution_dGE_dxs(SPs, Vs, coeffs, xsVs, N, xsVs_sum_inv, GE, dGE_dxs=None):
    if dGE_dxs is None:
        dGE_dxs = [0.0]*N
    for i in range(N):
        # i is what is being differentiated
        tot = 0.0
        for j in range(N):
            SPi_m_SPj = SPs[i] - SPs[j]
            Hij = SPs[i]*SPs[j]*(coeffs[i][j] + coeffs[j][i]) + SPi_m_SPj*SPi_m_SPj
            tot += Vs[i]*xsVs[j]*Hij
        dGE_dxs[i] = (tot - GE*Vs[i])*xsVs_sum_inv
    return dGE_dxs


class RegularSolution(GibbsExcess):
    r'''Class for representing an a liquid with excess gibbs energy represented
    by the Regular Solution model. This model is not temperature dependent and
    has limited predictive ability, but can be used without interaction
    parameters. This model is described in [1]_.

    .. math::
        G^E = \frac{\sum_m \sum_n (x_m x_n V_m V_n A_{mn})}{\sum_m x_m V_m}

    .. math::
        A_{mn} = 0.5(\delta_m - \delta_n)^2 - \delta_m \delta_n k_{mn}

    In the above equation, :math:`\delta` represents the solubility parameters,
    and :math:`k_{mn}` is the interaction coefficient between `m` and `n`.
    The model makes no assumption about the symmetry of this parameter.

    Parameters
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]
    Vs : list[float]
        Molar volumes of each compond at a reference temperature (often 298.15
        K), [K]
    SPs : list[float]
        Solubility parameters of each compound; normally at a reference
        temperature of 298.15 K, [Pa^0.5]
    lambda_coeffs : list[list[float]], optional
        Optional interaction parameters, [-]

    Attributes
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]
    Vs : list[float]
        Molar volumes of each compond at a reference temperature (often 298.15
        K), [K]
    SPs : list[float]
        Solubility parameters of each compound; normally at a reference
        temperature of 298.15 K, [Pa^0.5]
    lambda_coeffs : list[list[float]]
        Interaction parameters, [-]

    Notes
    -----
    In addition to the methods presented here, the methods of its base class
    :obj:`thermo.activity.GibbsExcess` are available as well.

    Examples
    --------
    From [2]_, calculate the activity coefficients at infinite dilution for the
    system benzene-cyclohexane at 253.15 K using the regular solution model
    (example 5.20, with unit conversion in-line):

    >>> from scipy.constants import calorie
    >>> GE = RegularSolution(T=353.15, xs=[.5, .5], Vs=[89E-6, 109E-6], SPs=[9.2*(calorie*1e6)**0.5, 8.2*(calorie*1e6)**0.5])
    >>> GE.gammas_infinite_dilution()
    [1.1352128394, 1.16803058378]

    This matches the solution given of [1.135, 1.168].

    References
    ----------
    .. [1] Poling, Bruce E., John M. Prausnitz, and John P. O’Connell. The
       Properties of Gases and Liquids. 5th edition. New York: McGraw-Hill
       Professional, 2000.
    .. [2] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
       Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
    '''
    def __init__(self, T, xs, Vs, SPs, lambda_coeffs=None):
        # lambda_coeffs is N*N of zeros for no interaction parameters
        self.T = T
        self.xs = xs
        self.Vs = Vs
        self.SPs = SPs
        self.N = N = len(Vs)
        self.scalar = scalar = type(Vs) is list
        self.cmps = cmps = range(N)

        if lambda_coeffs is None:
            if scalar:
                lambda_coeffs = [[0.0]*N for i in range(N)]
            else:
                lambda_coeffs = zeros((N, N))
        self.lambda_coeffs = lambda_coeffs

        if scalar:
            xsVs = []
            xsVs_sum = 0.0
            for i in cmps:
                xV = xs[i]*Vs[i]
                xsVs_sum += xV
                xsVs.append(xV)

        else:
            xsVs =  (xs*Vs)
            xsVs_sum = xsVs.sum()

        self.xsVs = xsVs
        self.xsVs_sum = xsVs_sum
        self.xsVs_sum_inv = 1.0/xsVs_sum

    def __repr__(self):
        s = '%s(T=%s, xs=%s, Vs=%s, SPs=%s, lambda_coeffs=%s)' %(self.__class__.__name__, repr(self.T), repr(self.xs),
                self.Vs, self.SPs, self.lambda_coeffs)
        return s

    def to_T_xs(self, T, xs):
        r'''Method to construct a new :obj:`RegularSolution` instance at
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
        obj : RegularSolution
            New :obj:`RegularSolution` object at the specified conditions [-]

        Notes
        -----
        '''
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.SPs = self.SPs
        new.Vs = Vs = self.Vs
        new.N = self.N
        new.lambda_coeffs = self.lambda_coeffs
        new.cmps = self.cmps
        new.scalar = scalar = self.scalar

        if scalar:
            xsVs = []
            xsVs_sum = 0.0
            for i in self.cmps:
                xV = xs[i]*Vs[i]
                xsVs_sum += xV
                xsVs.append(xV)
        else:
            xsVs = xs*Vs
            xsVs_sum = xsVs.sum()
        new.xsVs = xsVs
        new.xsVs_sum = xsVs_sum
        new.xsVs_sum_inv = 1.0/xsVs_sum
        return new


    def GE(self):
        r'''Calculate and return the excess Gibbs energy of a liquid phase
        using the regular solution model.

        .. math::
            G^E = \frac{\sum_m \sum_n (x_m x_n V_m V_n A_{mn})}{\sum_m x_m V_m}

        .. math::
            A_{mn} = 0.5(\delta_m - \delta_n)^2 - \delta_m \delta_n k_{mn}

        Returns
        -------
        GE : float
            Excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        '''
        from sympy import *
        GEvar, dGEvar_dT, GEvar_dx, dGEvar_dxixj, H = symbols("GEvar, dGEvar_dT, GEvar_dx, dGEvar_dxixj, H", cls=Function)

        N = 3
        cmps = range(N)
        R, T = symbols('R, T')
        xs = x0, x1, x2 = symbols('x0, x1, x2')
        Vs = V0, V1, V2 = symbols('V0, V1, V2')
        SPs = SP0, SP1, SP2 = symbols('SP0, SP1, SP2')
        l00, l01, l02, l10, l11, l12, l20, l21, l22 = symbols('l00, l01, l02, l10, l11, l12, l20, l21, l22')
        l_ijs = [[l00, l01, l02],
                 [l10, l11, l12],
                 [l20, l21, l22]]

        GE = 0
        denom = sum([xs[i]*Vs[i] for i in cmps])
        num = 0
        for i in cmps:
            for j in cmps:
                Aij = (SPs[i] - SPs[j])**2/2 + l_ijs[i][j]*SPs[i]*SPs[j]
                num += xs[i]*xs[j]*Vs[i]*Vs[j]*Aij
        GE = num/denom
        '''
        try:
            return self._GE
        except AttributeError:
            pass
        GE = self._GE = regular_solution_GE(self.SPs, self.lambda_coeffs, self.xsVs, self.N, self.xsVs_sum_inv)
        return GE


    def dGE_dxs(self):
        r'''Calculate and return the mole fraction derivatives of excess Gibbs
        energy of a liquid phase using the regular solution model.

        .. math::
            \frac{\partial G^E}{\partial x_i} = \frac{-V_i G^E + \sum_m V_i V_m
            x_m[\delta_i\delta_m(k_{mi} + k_{im}) + (\delta_i - \delta_m)^2 ]}
            {\sum_m V_m x_m}

        Returns
        -------
        dGE_dxs : list[float]
            Mole fraction derivatives of excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        '''
        dGEdxs = (diff(GE, x0)).subs(GE, GEvar(x0, x1, x2))
        Hi = dGEdxs.args[0].args[1]
        dGEdxs
        '''
        try:
            return self._dGE_dxs
        except AttributeError:
            pass
#        N, xsVs_sum_inv, xsVs = self.N, self.xsVs_sum_inv, self.xsVs
#        SPs, Vs, coeffs = self.SPs, self.Vs, self.lambda_coeffs
        try:
            GE = self._GE
        except:
            GE = self.GE()

        self._dGE_dxs = dGE_dxs = regular_solution_dGE_dxs(self.SPs, self.Vs, self.lambda_coeffs, self.xsVs, self.N, self.xsVs_sum_inv, GE)

#        for i in range(N):
#            # i is what is being differentiated
#            tot = 0.0
#            for j in range(N):
#                SPi_m_SPj = SPs[i] - SPs[j]
#                Hij = SPs[i]*SPs[j]*(coeffs[i][j] + coeffs[j][i]) + SPi_m_SPj*SPi_m_SPj
#                tot += Vs[i]*xsVs[j]*Hij
#            dGE_dxs.append((tot - GE*Vs[i])*xsVs_sum_inv)
        return dGE_dxs

    def d2GE_dxixjs(self):
        r'''Calculate and return the second mole fraction derivatives of excess
        Gibbs energy of a liquid phase using the regular solution model.

        .. math::
            \frac{\partial^2 G^E}{\partial x_i \partial x_j} = \frac{V_j(V_i G^E - H_{ij})}{(\sum_m V_m x_m)^2}
            - \frac{V_i \frac{\partial G^E}{\partial x_j}}{\sum_m V_m x_m}
            + \frac{V_i V_j[\delta_i\delta_j(k_{ji} + k_{ij}) + (\delta_i - \delta_j)^2] }{\sum_m V_m x_m}

        Returns
        -------
        d2GE_dxixjs : list[list[float]]
            Second mole fraction derivatives of excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        '''
        d2GEdxixjs = diff((diff(GE, x0)).subs(GE, GEvar(x0, x1, x2)), x1).subs(Hi, H(x0, x1, x2))
        d2GEdxixjs
        '''
        try:
            return self._d2GE_dxixjs
        except AttributeError:
            pass
        N, xsVs_sum_inv = self.N, self.xsVs_sum_inv
        xs, SPs, Vs, coeffs = self.xs, self.SPs, self.Vs, self.lambda_coeffs
        xsVs = self.xsVs
        GE, dGE_dxs = self.GE(), self.dGE_dxs()

        if self.scalar:
            d2GE_dxixjs = [[0.0]*N for i in range(N)]
        else:
            d2GE_dxixjs = zeros((N, N))
        self._d2GE_dxixjs = d2GE_dxixjs

        # Shared between two things need to make a separate function
        Hi_sums = [0.0]*N
        for i in range(N):
            t = 0.0
            for j in range(N):
                SPi_m_SPj = SPs[i] - SPs[j]
                Hi = SPs[i]*SPs[j]*(coeffs[i][j] + coeffs[j][i]) + SPi_m_SPj*SPi_m_SPj
                t += Vs[i]*xsVs[j]*Hi
            Hi_sums[i] = t

        for i in range(N):
            row = d2GE_dxixjs[i]
            v0 = (Vs[i]*GE - Hi_sums[i])*xsVs_sum_inv*xsVs_sum_inv
            v1 = Vs[i]*xsVs_sum_inv
            for j in range(N):
                SPi_m_SPj = SPs[i] - SPs[j]
                Hi = SPs[i]*SPs[j]*(coeffs[i][j] + coeffs[j][i]) + SPi_m_SPj*SPi_m_SPj
                tot = Vs[j]*v0 + v1*(Vs[j]*Hi - dGE_dxs[j])

                row[j] = tot
        return d2GE_dxixjs

    def d3GE_dxixjxks(self):
        r'''Calculate and return the third mole fraction derivatives of excess
        Gibbs energy.

        .. math::
            \frac{\partial^3 G^E}{\partial x_i \partial x_j \partial x_k} = \frac{-2V_iV_jV_k G^E + 2 V_j V_k H_{ij}} {(\sum_m V_m x_m)^3}
            + \frac{V_i\left(V_j\frac{\partial G^E}{\partial x_k} + V_k\frac{\partial G^E}{\partial x_j}  \right)} {(\sum_m V_m x_m)^2}
            - \frac{V_i \frac{\partial^2 G^E}{\partial x_j \partial x_k}}{\sum_m V_m x_m}
            - \frac{V_iV_jV_k[\delta_i(\delta_j(k_{ij} + k_{ji}) + \delta_k(k_{ik} + k_{ki})) + (\delta_i - \delta_j)^2 + (\delta_i - \delta_k)^2 ]}{(\sum_m V_m x_m)^2}

        Returns
        -------
        d3GE_dxixjxks : list[list[list[float]]]
            Third mole fraction derivatives of excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        try:
            return self._d3GE_dxixjxks
        except:
            pass
        cmps, xsVs_sum_inv = self.cmps, self.xsVs_sum_inv
        xs, SPs, Vs, coeffs = self.xs, self.SPs, self.Vs, self.lambda_coeffs
        GE, dGE_dxs, d2GE_dxixjs = self.GE(), self.dGE_dxs(), self.d2GE_dxixjs()

        # Shared between two things need to make a separate function
        Hi_sums = []
        for i in cmps:
            t = 0.0
            for j in cmps:
                SPi_m_SPj = SPs[i] - SPs[j]
                Hi = SPs[i]*SPs[j]*(coeffs[i][j] + coeffs[j][i]) + SPi_m_SPj*SPi_m_SPj
                t += Vs[i]*Vs[j]*xs[j]*Hi
            Hi_sums.append(t)

        # all the same: analytical[i][j][k] = analytical[i][k][j] = analytical[j][i][k] = analytical[j][k][i] = analytical[k][i][j] = analytical[k][j][i] = float(v)
        self._d3GE_dxixjxks = d3GE_dxixjxks = []
        for i in cmps:
            dG_matrix = []
            for j in cmps:
                dG_row = []
                for k in cmps:
                    tot = 0.0
#                    if j == k:
                    thirds = -2.0*Vs[i]*Vs[j]*Vs[k]*GE + 2.0*Vs[j]*Vs[k]*Hi_sums[i]
                    seconds = Vs[i]*(Vs[j]*dGE_dxs[k] + Vs[k]*dGE_dxs[j])
                    seconds -= Vs[i]*Vs[j]*Vs[k]*(
                                SPs[i]*(SPs[j]*(coeffs[i][j] + coeffs[j][i]) + SPs[k]*(coeffs[i][k] + coeffs[k][i]))
                                 + (SPs[i]-SPs[j])**2 + (SPs[i] - SPs[k])**2
                                 )
                    firsts = -Vs[i]*d2GE_dxixjs[j][k]



                    tot = firsts*xsVs_sum_inv + seconds*xsVs_sum_inv*xsVs_sum_inv + thirds*xsVs_sum_inv*xsVs_sum_inv*xsVs_sum_inv
                    dG_row.append(tot)
                dG_matrix.append(dG_row)
            d3GE_dxixjxks.append(dG_matrix)
        return d3GE_dxixjxks

    def d2GE_dTdxs(self):
        r'''Calculate and return the temperature derivative of mole fraction
        derivatives of excess Gibbs energy.

        .. math::
            \frac{\partial^2 g^E}{\partial x_i \partial T} = 0

        Returns
        -------
        d2GE_dTdxs : list[float]
            Temperature derivative of mole fraction derivatives of excess Gibbs
            energy, [J/(mol*K)]

        Notes
        -----
        '''
        return [0.0]*self.N

    def dGE_dT(self):
        r'''Calculate and return the temperature derivative of excess Gibbs
        energy of a liquid phase.

        .. math::
            \frac{\partial g^E}{\partial T} = 0

        Returns
        -------
        dGE_dT : float
            First temperature derivative of excess Gibbs energy, [J/(mol*K)]

        Notes
        -----
        '''
        return 0.0

    def d2GE_dT2(self):
        r'''Calculate and return the second temperature derivative of excess
        Gibbs energy of a liquid phas.

        .. math::
            \frac{\partial^2 g^E}{\partial T^2} = 0

        Returns
        -------
        d2GE_dT2 : float
            Second temperature derivative of excess Gibbs energy, [J/(mol*K^2)]

        Notes
        -----
        '''
        return 0.0

    def d3GE_dT3(self):
        r'''Calculate and return the third temperature derivative of excess
        Gibbs energy of a liquid phase.

        .. math::
            \frac{\partial^3 g^E}{\partial T^3} = 0

        Returns
        -------
        d3GE_dT3 : float
            Third temperature derivative of excess Gibbs energy, [J/(mol*K^3)]

        Notes
        -----
        '''
        return 0.0

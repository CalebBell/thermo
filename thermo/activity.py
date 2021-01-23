# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains a base class :obj:`GibbsExcess` for handling activity
coefficient based
models. The design is for a sub-class to provide the minimum possible number of
derivatives of Gibbs energy, and for this base class to provide the rest of the
methods.  An ideal-liquid class with no excess Gibbs energy
:obj:`IdealSolution` is also available.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Base Class
==========

.. autoclass:: GibbsExcess
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members:

Idea Liquid Class
=================

.. autoclass:: IdealSolution
    :members: to_T_xs, GE, dGE_dT, d2GE_dT2, d3GE_dT3, d2GE_dTdxs, dGE_dxs, d2GE_dxixjs, d3GE_dxixjxks
    :undoc-members:
    :show-inheritance:
    :exclude-members: gammas

Notes
=====
Excellent references for working with activity coefficient models are [1]_ and
[2]_.

References
----------
.. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
   Butterworth-Heinemann, 1985.
.. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process
   Simulation. Weinheim, Germany: Wiley-VCH, 2012.

'''

from __future__ import division

__all__ = ['GibbsExcess', 'IdealSolution']
from fluids.constants import R, R_inv
from fluids.numerics import numpy as np
from chemicals.utils import exp
from chemicals.utils import normalize, dxs_to_dns, dxs_to_dn_partials, dns_to_dn_partials, d2xs_to_dxdn_partials

try:
    npexp, ones, zeros, array = np.exp, np.ones, np.zeros, np.array
except:
    pass

def gibbs_excess_gammas(xs, dG_dxs, GE, T, gammas=None):
    xdx_totF = GE
    N = len(xs)
    for i in range(N):
        xdx_totF -= xs[i]*dG_dxs[i]
    RT_inv = R_inv/T
    if gammas is None:
        gammas = [0.0]*N
    for i in range(N):
        gammas[i] = exp((dG_dxs[i] + xdx_totF)*RT_inv)
    return gammas

def gibbs_excess_dHE_dxs(dGE_dxs, d2GE_dTdxs, N, T, dHE_dxs=None):
    if dHE_dxs is None:
        dHE_dxs = [0.0]*N
    for i in range(N):
        dHE_dxs[i] = -T*d2GE_dTdxs[i] + dGE_dxs[i]
    return dHE_dxs


def gibbs_excess_dgammas_dns(xs, gammas, d2GE_dxixjs, N, T, dgammas_dns=None, vec0=None):
    if vec0 is None:
        vec0 = [0.0]*N
    if dgammas_dns is None:
        dgammas_dns = [[0.0]*N for _ in range(N)] # numba : delete
#        dgammas_dns = zeros((N, N)) # numba : uncomment

    for j in range(N):
        tot = 0.0
        row = d2GE_dxixjs[j]
        for k in range(N):
            tot += xs[k]*row[k]
        vec0[j] = tot

    RT_inv = R_inv/(T)

    for i in range(N):
        gammai_RT = gammas[i]*RT_inv
        for j in range(N):
            dgammas_dns[i][j] = gammai_RT*(d2GE_dxixjs[i][j] - vec0[j])

    return dgammas_dns

def gibbs_excess_dgammas_dT(xs, GE, dGE_dT, dG_dxs, d2GE_dTdxs, N, T, dgammas_dT=None):
    if dgammas_dT is None:
        dgammas_dT = [0.0]*N

    xdx_totF0 = dGE_dT
    for j in range(N):
        xdx_totF0 -= xs[j]*d2GE_dTdxs[j]
    xdx_totF1 = GE
    for j in range(N):
        xdx_totF1 -= xs[j]*dG_dxs[j]

    T_inv = 1.0/T
    RT_inv = R_inv*T_inv
    for i in range(N):
        dG_dni = xdx_totF1 + dG_dxs[i]
        dgammas_dT[i] = RT_inv*(d2GE_dTdxs[i] - dG_dni*T_inv + xdx_totF0)*exp(dG_dni*RT_inv)
    return dgammas_dT

class GibbsExcess(object):
    r'''Class for representing an activity coefficient model.
    While these are typically presented as tools to compute activity
    coefficients, in truth they are excess Gibbs energy models and activity
    coefficients are just one derived aspect of them.

    This class does not implement any activity coefficient models itself; it
    must be subclassed by another model. All properties are
    derived with the CAS SymPy, not relying on any derivations previously
    published, and checked numerically for consistency.

    Different subclasses have different parameter requirements for
    initialization; :obj:`IdealSolution` is
    available as a simplest model with activity coefficients of 1 to show
    what needs to be implemented in subclasses. It is also intended subclasses
    implement the method `to_T_xs`, which creates a new object at the
    specified temperature and composition but with the same parameters.

    These objects are intended to lazy-calculate properties as much as
    possible, and for the temperature and composition of an object to be
    immutable.

    '''
    def __repr__(self):
        # Other classes with different parameters should expose them here too
        s = '%s(T=%s, xs=%s)' %(self.__class__.__name__, repr(self.T), repr(self.xs))
        return s

    def HE(self):
        r'''Calculate and return the excess entropy of a liquid phase using an
        activity coefficient model.

        .. math::
            h^E = -T \frac{\partial g^E}{\partial T} + g^E

        Returns
        -------
        HE : float
            Excess enthalpy of the liquid phase, [J/mol]

        Notes
        -----
        '''
        '''f = symbols('f', cls=Function)
        T = symbols('T')
        simplify(-T**2*diff(f(T)/T, T))
        '''
        return -self.T*self.dGE_dT() + self.GE()

    def dHE_dT(self):
        r'''Calculate and return the first temperature derivative of excess
        enthalpy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial h^E}{\partial T} = -T \frac{\partial^2 g^E}
            {\partial T^2}

        Returns
        -------
        dHE_dT : float
            First temperature derivative of excess enthalpy of the liquid
            phase, [J/mol/K]

        Notes
        -----
        '''
        return -self.T*self.d2GE_dT2()

    CpE = dHE_dT

    def dHE_dxs(self):
        r'''Calculate and return the mole fraction derivative of excess
        enthalpy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial h^E}{\partial x_i} = -T \frac{\partial^2 g^E}
            {\partial T \partial x_i} + \frac{\partial g^E}{\partial x_i}

        Returns
        -------
        dHE_dxs : list[float]
            First mole fraction derivative of excess enthalpy of the liquid
            phase, [J/mol]

        Notes
        -----
        '''
        try:
            return self._dHE_dxs
        except:
            pass
        # Derived by hand taking into account the expression for excess enthalpy
        d2GE_dTdxs = self.d2GE_dTdxs()
        try:
            dGE_dxs = self._dGE_dxs
        except:
            dGE_dxs = self.dGE_dxs()
        self._dHE_dxs = dHE_dxs = gibbs_excess_dHE_dxs(dGE_dxs, d2GE_dTdxs, self.N, self.T)
        return dHE_dxs

    def dHE_dns(self):
        r'''Calculate and return the mole number derivative of excess
        enthalpy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial h^E}{\partial n_i}

        Returns
        -------
        dHE_dns : list[float]
            First mole number derivative of excess enthalpy of the liquid
            phase, [J/mol^2]

        Notes
        -----
        '''
        return dxs_to_dns(self.dHE_dxs(), self.xs)

    def dnHE_dns(self):
        r'''Calculate and return the partial mole number derivative of excess
        enthalpy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial n h^E}{\partial n_i}

        Returns
        -------
        dnHE_dns : list[float]
            First partial mole number derivative of excess enthalpy of the
            liquid phase, [J/mol]

        Notes
        -----
        '''
        return dxs_to_dn_partials(self.dHE_dxs(), self.xs, self.HE())

    def SE(self):
        r'''Calculates the excess entropy of a liquid phase using an
        activity coefficient model.

        .. math::
            s^E = \frac{h^E - g^E}{T}

        Returns
        -------
        SE : float
            Excess entropy of the liquid phase, [J/mol/K]

        Notes
        -----

        Note also the relationship of the expressions for partial excess
        entropy:

        .. math::
            S_i^E = -R\left(T \frac{\partial \ln \gamma_i}{\partial T}
            + \ln \gamma_i\right)


        '''
        return (self.HE() - self.GE())/self.T

    def dSE_dT(self):
        r'''Calculate and return the first temperature derivative of excess
        entropy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial s^E}{\partial T} = \frac{1}{T}
            \left(\frac{-\partial g^E}{\partial T} + \frac{\partial h^E}{\partial T}
            - \frac{(G + H)}{T}\right)

        Returns
        -------
        dSE_dT : float
            First temperature derivative of excess entropy of the liquid
            phase, [J/mol/K]

        Notes
        -----

        '''
        '''from sympy import *
        T = symbols('T')
        G, H = symbols('G, H', cls=Function)
        S = (H(T) - G(T))/T
        print(diff(S, T))
        # (-Derivative(G(T), T) + Derivative(H(T), T))/T - (-G(T) + H(T))/T**2
        '''
        # excess entropy temperature derivative
        dHE_dT = self.dHE_dT()
        try:
            HE = self._HE
        except:
            HE = self.HE()
        try:
            dGE_dT = self._dGE_dT
        except:
            dGE_dT = self.dGE_dT()
        try:
            GE = self._GE
        except:
            GE = self.GE()
        T_inv = 1.0/self.T
        return T_inv*(-dGE_dT + dHE_dT - (HE - GE)*T_inv)

    def dSE_dxs(self):
        r'''Calculate and return the mole fraction derivative of excess
        entropy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial S^E}{\partial x_i} = \frac{1}{T}\left( \frac{\partial h^E}
            {\partial x_i} - \frac{\partial g^E}{\partial x_i}\right)
            = -\frac{\partial^2 g^E}{\partial x_i \partial T}

        Returns
        -------
        dSE_dxs : list[float]
            First mole fraction derivative of excess entropy of the liquid
            phase, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._dSE_dxs
        except:
            pass
        try:
            d2GE_dTdxs = self._d2GE_dTdxs
        except:
            d2GE_dTdxs = self.d2GE_dTdxs()
        if self.scalar:
            dSE_dxs = [-v for v in d2GE_dTdxs]
        else:
            dSE_dxs = -d2GE_dTdxs
        self._dSE_dxs = dSE_dxs
        return dSE_dxs

    def dSE_dns(self):
        r'''Calculate and return the mole number derivative of excess
        entropy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial S^E}{\partial n_i}

        Returns
        -------
        dSE_dns : list[float]
            First mole number derivative of excess entropy of the liquid
            phase, [J/(mol^2*K)]

        Notes
        -----
        '''
        return dxs_to_dns(self.dSE_dxs(), self.xs)

    def dnSE_dns(self):
        r'''Calculate and return the partial mole number derivative of excess
        entropy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial n S^E}{\partial n_i}

        Returns
        -------
        dnSE_dns : list[float]
            First partial mole number derivative of excess entropy of the liquid
            phase, [J/(mol*K)]

        Notes
        -----
        '''
        return dxs_to_dn_partials(self.dSE_dxs(), self.xs, self.SE())

    def dGE_dns(self):
        r'''Calculate and return the mole number derivative of excess
        Gibbs energy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial G^E}{\partial n_i}

        Returns
        -------
        dGE_dns : list[float]
            First mole number derivative of excess Gibbs entropy of the liquid
            phase, [J/(mol^2*K)]

        Notes
        -----
        '''
        return dxs_to_dns(self.dGE_dxs(), self.xs)

    def dnGE_dns(self):
        r'''Calculate and return the partial mole number derivative of excess
        Gibbs energy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial n G^E}{\partial n_i}

        Returns
        -------
        dnGE_dns : list[float]
            First partial mole number derivative of excess Gibbs entropy of the
            liquid phase, [J/(mol)]

        Notes
        -----
        '''
        return dxs_to_dn_partials(self.dGE_dxs(), self.xs, self.GE())

    def d2GE_dTdns(self):
        r'''Calculate and return the mole number derivative of the first
        temperature derivative of excess Gibbs energy of a liquid phase using
        an activity coefficient model.

        .. math::
            \frac{\partial^2 G^E}{\partial n_i \partial T}

        Returns
        -------
        d2GE_dTdns : list[float]
            First mole number derivative of the temperature derivative of
            excess Gibbs entropy of the liquid phase, [J/(mol^2*K)]

        Notes
        -----
        '''
        return dxs_to_dns(self.d2GE_dTdxs(), self.xs)


    def d2nGE_dTdns(self):
        r'''Calculate and return the partial mole number derivative of the first
        temperature derivative of excess Gibbs energy of a liquid phase using
        an activity coefficient model.

        .. math::
            \frac{\partial^2 n G^E}{\partial n_i \partial T}

        Returns
        -------
        d2nGE_dTdns : list[float]
            First partial mole number derivative of the temperature derivative
            of excess Gibbs entropy of the liquid phase, [J/(mol*K)]

        Notes
        -----
        '''
        # needed in gammas temperature derivatives
        dGE_dT = self.dGE_dT()
        d2GE_dTdns = self.d2GE_dTdns()
        return dns_to_dn_partials(d2GE_dTdns, dGE_dT)


    def d2nGE_dninjs(self):
        r'''Calculate and return the second partial mole number derivative of
        excess Gibbs energy of a liquid phase using
        an activity coefficient model.

        .. math::
            \frac{\partial^2 n G^E}{\partial n_i \partial n_i}

        Returns
        -------
        d2nGE_dninjs : list[list[float]]
            Second partial mole number derivative of excess Gibbs energy of a
            liquid phase, [J/(mol^2)]

        Notes
        -----
        '''
        # This one worked out
        return d2xs_to_dxdn_partials(self.d2GE_dxixjs(), self.xs)

    def gammas_infinite_dilution(self):
        r'''Calculate and return the infinite dilution activity coefficients
        of each component.

        Returns
        -------
        gammas_infinite : list[float]
            Infinite dilution activity coefficients, [-]

        Notes
        -----
        The algorithm is as follows. For each component, set its composition to
        zero. Normalize the remaining compositions to 1. Create a new object
        with that composition, and calculate the activity coefficient of the
        component whose concentration was set to zero.
        '''
        T, N = self.T, self.N
        xs_base = self.xs
        if self.scalar:
            gammas_inf = [0.0]*N
            copy_fun = list
        else:
            gammas_inf = zeros(N)
            copy_fun = array
        for i in range(N):
            xs = copy_fun(xs_base)
            xs[i] = 0.0
            xs = normalize(xs)
            gammas_inf[i] = self.to_T_xs(T, xs=xs).gammas()[i]
        return gammas_inf

    def gammas(self):
        r'''Calculate and return the activity coefficients of a liquid phase
        using an activity coefficient model.

        .. math::
            \gamma_i = \exp\left(\frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}\right)

        Returns
        -------
        gammas : list[float]
            Activity coefficients, [-]

        Notes
        -----
        '''
        try:
            return self._gammas
        except:
            pass
        # Matches the gamma formulation perfectly
        GE = self.GE()
        dG_dxs = self.dGE_dxs()
        if self.scalar:
            dG_dns = dxs_to_dn_partials(dG_dxs, self.xs, GE)
            RT_inv = 1.0/(R*self.T)
            gammas = [exp(i*RT_inv) for i in dG_dns]
        else:
            gammas = gibbs_excess_gammas(self.xs, dG_dxs, GE, self.T)
        self._gammas = gammas
        return gammas

    def _gammas_dGE_dxs(self):
        try:
            del self._gammas
        except:
            pass
        return GibbsExcess.gammas(self)

    def dgammas_dns(self):
        r'''Calculate and return the mole number derivative of activity
        coefficients of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial \gamma_i}{\partial n_i} = \gamma_i
            \left(\frac{\frac{\partial^2 G^E}{\partial x_i \partial x_j}}{RT}\right)

        Returns
        -------
        dgammas_dns : list[list[float]]
            Mole number derivatives of activity coefficients, [-]

        Notes
        -----
        '''
        try:
            return self._dgammas_dns
        except AttributeError:
            pass
        gammas = self.gammas()
        N = self.N
        xs = self.xs
        d2GE_dxixjs = self.d2GE_dxixjs()

        self._dgammas_dns = dgammas_dns = gibbs_excess_dgammas_dns(xs, gammas, d2GE_dxixjs, N, self.T)
        return dgammas_dns

#    def dgammas_dxs(self):
        # TODO - compare with UNIFAC, which has a dx derivative working
#        # NOT WORKING
#        gammas = self.gammas()
#        cmps = self.cmps
#        RT_inv = 1.0/(R*self.T)
#        d2GE_dxixjs = self.d2GE_dxixjs() # Thi smatrix is symmetric
#
#        def thing(d2xs, xs):
#            cmps = range(len(xs))
#
#            double_sums = []
#            for j in cmps:
#                tot = 0.0
#                for k in cmps:
#                    tot += xs[k]*d2xs[j][k]
#                double_sums.append(tot)
#
#            mat = []
#            for i in cmps:
#                row = []
#                for j in cmps:
#                    row.append(d2xs[i][j] - double_sums[i])
#                mat.append(row)
#            return mat
#
#            return [[d2xj - tot for (d2xj, tot) in zip(d2xsi, double_sums)]
#                     for d2xsi in d2xs]
#
#        d2nGE_dxjnis = thing(d2GE_dxixjs, self.xs)
#
#        matrix = []
#        for i in cmps:
#            row = []
#            gammai = gammas[i]
#            for j in cmps:
#                v = gammai*d2nGE_dxjnis[i][j]*RT_inv
#                row.append(v)
#            matrix.append(row)
#        return matrix

#    def dgammas_dxs(self):
#   # Not done
#        return dxs_to_dns(self.dgammas_dx(), self.xs)

#    def dngammas_dxs(self):
#        pass

    def dgammas_dT(self):
        r'''Calculate and return the temperature derivatives of activity
        coefficients of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial \gamma_i}{\partial T} =
            \left(\frac{\frac{\partial^2 n G^E}{\partial T \partial n_i}}{RT} -
            \frac{{\frac{\partial n_i G^E}{\partial n_i }}}{RT^2}\right)
             \exp\left(\frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}\right)

        Returns
        -------
        dgammas_dT : list[float]
            Temperature derivatives of activity coefficients, [1/K]

        Notes
        -----
        '''
        r'''
        from sympy import *
        R, T = symbols('R, T')
        f = symbols('f', cls=Function)
        diff(exp(f(T)/(R*T)), T)
        '''
        try:
            return self._dgammas_dT
        except AttributeError:
            pass
        N, T, xs = self.N, self.T, self.xs
        dGE_dT = self.dGE_dT()
        GE = self.GE()
        dG_dxs = self.dGE_dxs()
        d2GE_dTdxs = self.d2GE_dTdxs()
        dgammas_dT = gibbs_excess_dgammas_dT(xs, GE, dGE_dT, dG_dxs, d2GE_dTdxs, N, T)
        self._dgammas_dT = dgammas_dT
        return dgammas_dT


class IdealSolution(GibbsExcess):
    r'''Class for  representing an ideal liquid, with no excess gibbs energy
    and thus activity coefficients of 1.

    Parameters
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]

    Attributes
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]

    Examples
    --------
    >>> model = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
    >>> model.GE()
    0.0
    >>> model.gammas()
    [1.0, 1.0, 1.0, 1.0]
    >>> model.dgammas_dT()
    [0.0, 0.0, 0.0, 0.0]
    '''
    def __init__(self, T=None, xs=None):
        if T is not None:
            self.T = T
        if xs is not None:
            self.xs = xs
            self.N = len(xs)
            self.scalar = type(xs) is list
        else:
            self.scalar = True

    def to_T_xs(self, T, xs):
        r'''Method to construct a new :obj:`IdealSolution` instance at
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
        obj : IdealSolution
            New :obj:`IdealSolution` object at the specified conditions [-]

        Notes
        -----

        Examples
        --------
        >>> p = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
        >>> p.to_T_xs(T=500.0, xs=[.25, .25, .25, .25])
        IdealSolution(T=500.0, xs=[0.25, 0.25, 0.25, 0.25])
        '''
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.scalar = self.scalar
        new.N = len(xs)
        return new

    def GE(self):
        r'''Calculate and return the excess Gibbs energy of a liquid phase
        using an activity coefficient model.

        .. math::
            g^E = 0

        Returns
        -------
        GE : float
            Excess Gibbs energy of an ideal liquid, [J/mol]

        Notes
        -----
        '''
        return 0.0

    def dGE_dT(self):
        r'''Calculate and return the temperature derivative of excess Gibbs
        energy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial g^E}{\partial T} = 0

        Returns
        -------
        dGE_dT : float
            First temperature derivative of excess Gibbs energy of an
            ideal liquid, [J/(mol*K)]

        Notes
        -----
        '''
        return 0.0

    def d2GE_dT2(self):
        r'''Calculate and return the second temperature derivative of excess
        Gibbs energy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial^2 g^E}{\partial T^2} = 0

        Returns
        -------
        d2GE_dT2 : float
            Second temperature derivative of excess Gibbs energy of an
            ideal liquid, [J/(mol*K^2)]

        Notes
        -----
        '''
        return 0.0

    def d3GE_dT3(self):
        r'''Calculate and return the third temperature derivative of excess
        Gibbs energy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial^3 g^E}{\partial T^3} = 0

        Returns
        -------
        d3GE_dT3 : float
            Third temperature derivative of excess Gibbs energy of an ideal
            liquid, [J/(mol*K^3)]

        Notes
        -----
        '''
        return 0.0

    def d2GE_dTdxs(self):
        r'''Calculate and return the temperature derivative of mole fraction
        derivatives of excess Gibbs energy of an ideal liquid.

        .. math::
            \frac{\partial^2 g^E}{\partial x_i \partial T} = 0

        Returns
        -------
        d2GE_dTdxs : list[float]
            Temperature derivative of mole fraction derivatives of excess Gibbs
            energy of an ideal liquid, [J/(mol*K)]

        Notes
        -----
        '''
        if self.scalar:
            return [0.0]*self.N
        return zeros(self.N)

    def dGE_dxs(self):
        r'''Calculate and return the mole fraction derivatives of excess Gibbs
        energy of an ideal liquid.

        .. math::
            \frac{\partial g^E}{\partial x_i} = 0

        Returns
        -------
        dGE_dxs : list[float]
            Mole fraction derivatives of excess Gibbs energy of an ideal
            liquid, [J/mol]

        Notes
        -----
        '''
        if self.scalar:
            return [0.0]*self.N
        return zeros(self.N)

    def d2GE_dxixjs(self):
        r'''Calculate and return the second mole fraction derivatives of excess
        Gibbs energy of an ideal liquid.

        .. math::
            \frac{\partial^2 g^E}{\partial x_i \partial x_j} = 0

        Returns
        -------
        d2GE_dxixjs : list[list[float]]
            Second mole fraction derivatives of excess Gibbs energy of an ideal
            liquid, [J/mol]

        Notes
        -----
        '''
        N = self.N
        if self.scalar:
            return [[0.0]*N for i in range(self.N)]
        return zeros((N, N))

    def d3GE_dxixjxks(self):
        r'''Calculate and return the third mole fraction derivatives of excess
        Gibbs energy of an ideal liquid.

        .. math::
            \frac{\partial^3 g^E}{\partial x_i \partial x_j \partial x_k} = 0

        Returns
        -------
        d3GE_dxixjxks : list[list[list[float]]]
            Third mole fraction derivatives of excess Gibbs energy of an ideal
            liquid, [J/mol]

        Notes
        -----
        '''
        N = self.N
        if self.scalar:
            return [[[0.0]*N for i in range(N)] for j in range(N)]
        return zeros((N, N, N))

    def gammas(self):
        if self.scalar:
            return [1.0]*self.N
        else:
            return ones(self.N)

    try:
        gammas.__doc__ = GibbsExcess.__doc__
    except:
        pass

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
SOFTWARE.'''

from __future__ import division

__all__ = ['GibbsExcess', 'IdealSolution']
from chemicals.utils import exp
from chemicals.utils import dxs_to_dns, dxs_to_dn_partials, dns_to_dn_partials, d2xs_to_dxdn_partials
from fluids.constants import R, R_inv


class GibbsExcess(object):
    def HE(self):
        # Just plain excess enthalpy here
        '''f = symbols('f', cls=Function)
        T = symbols('T')
        simplify(-T**2*diff(f(T)/T, T))
        '''
        return -self.T*self.dGE_dT() + self.GE()

    def dHE_dT(self):
        # excess enthalpy temperature derivative
        '''from sympy import *
        f = symbols('f', cls=Function)
        T = symbols('T')
        diff(simplify(-T**2*diff(f(T)/T, T)), T)
        '''
        return -self.T*self.d2GE_dT2()

    CpE = dHE_dT

    def dHE_dxs(self):
        try:
            self._dHE_dxs
        except:
            pass
        # Derived by hand taking into account the expression for excess enthalpy
        d2GE_dTdxs = self.d2GE_dTdxs()
        dGE_dxs = self.dGE_dxs()
        T = self.T
        self._dHE_dxs = [-T*d2GE_dTdxs[i] + dGE_dxs[i] for i in self.cmps]
        return self._dHE_dxs

    def dHE_dns(self):
        return dxs_to_dns(self.dHE_dxs(), self.xs)

    def dnHE_dns(self):
        return dxs_to_dn_partials(self.dHE_dxs(), self.xs, self.HE())

    def SE(self):
        r'''Calculates the excess entropy of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_.

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


        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        return (self.HE() - self.GE())/self.T

    def dSE_dT(self):
        '''from sympy import *
        T = symbols('T')
        G, H = symbols('G, H', cls=Function)
        S = (H(T) - G(T))/T
        print(diff(S, T))
        # (-Derivative(G(T), T) + Derivative(H(T), T))/T - (-G(T) + H(T))/T**2
        '''
        # excess entropy temperature derivative
        H = self.HE()
        dHdT = self.dHE_dT()
        dGdT = self.dGE_dT()
        G = self.GE()
        T_inv = 1.0/self.T
        return T_inv*((-dGdT + dHdT) - (-G + H)*T_inv)


    def dSE_dxs(self):
        try:
            return self._dSE_dxs
        except:
            pass
        # Derived by hand.
        dGE_dxs = self.dGE_dxs()
        dHE_dxs = self.dHE_dxs()
        T_inv = 1.0/self.T
        self._dSE_dxs = [T_inv*(dHE_dxs[i] - dGE_dxs[i]) for i in self.cmps]
        return self._dSE_dxs

    def dSE_dns(self):
        return dxs_to_dns(self.dSE_dxs(), self.xs)

    def dnSE_dns(self):
        return dxs_to_dn_partials(self.dSE_dxs(), self.xs, self.SE())

    def dGE_dns(self):
        # Mole number derivatives
        return dxs_to_dns(self.dGE_dxs(), self.xs)

    def dnGE_dns(self):
        return dxs_to_dn_partials(self.dGE_dxs(), self.xs, self.GE())

    def d2GE_dTdns(self):
        return dxs_to_dns(self.d2GE_dTdxs(), self.xs)


    def d2nGE_dTdns(self):
        # needed in gammas temperature derivatives
        dGE_dT = self.dGE_dT()
        d2GE_dTdns = self.d2GE_dTdns()
        return dns_to_dn_partials(d2GE_dTdns, dGE_dT)


    def d2nGE_dninjs(self):
        # This one worked out
        return d2xs_to_dxdn_partials(self.d2GE_dxixjs(), self.xs)


    def gammas(self):
        '''
        .. math::
            \gamma_i = \exp\left(\frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}\right)
        '''
        try:
            return self._gammas
        except:
            pass
        # Matches the gamma formulation perfectly
        dG_dxs = self.dGE_dxs()
        GE = self.GE()
        dG_dns = dxs_to_dn_partials(dG_dxs, self.xs, GE)
        RT_inv = 1.0/(R*self.T)
        self._gammas = [exp(i*RT_inv) for i in dG_dns]
        return self._gammas

    def _gammas_dGE_dxs(self):
        try:
            del self._gammas
        except:
            pass
        return GibbsExcess.gammas(self)

    def dgammas_dns(self):
        try:
            return self._dgammas_dns
        except AttributeError:
            pass
        gammas = self.gammas()
        cmps = self.cmps

        d2GE_dxixjs = self.d2GE_dxixjs()
        d2nGE_dxjnis = d2xs_to_dxdn_partials(d2GE_dxixjs, self.xs)

        RT_inv = 1.0/(R*self.T)

        self._dgammas_dns = matrix = []
        for i in cmps:
            row = []
            gammai = gammas[i]
            for j in cmps:
                v = gammai*d2nGE_dxjnis[i][j]*RT_inv
                row.append(v)
            matrix.append(row)
        return matrix

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
        r'''
        .. math::
            \frac{\partial \gamma_i}{\partial T} =
            \left(\frac{\frac{\partial^2 n G^E}{\partial T \partial n_i}}{RT} -
            \frac{{\frac{\partial n_i G^E}{\partial n_i }}}{RT^2}\right)
             \exp\left(\frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}\right)

        from sympy import *
        R, T = symbols('R, T')
        f = symbols('f', cls=Function)
        diff(exp(f(T)/(R*T)), T)
        '''
        try:
            return self._dgammas_dT
        except AttributeError:
            pass
        d2nGE_dTdns = self.d2nGE_dTdns()

        dG_dxs = self.dGE_dxs()
        GE = self.GE()
        dG_dns = dxs_to_dn_partials(dG_dxs, self.xs, GE)

        T_inv = 1.0/self.T
        RT_inv = R_inv*T_inv
        self._dgammas_dT = dgammas_dT = []
        for i in self.cmps:
            x1 = dG_dns[i]*T_inv
            dgammas_dT.append(RT_inv*(d2nGE_dTdns[i] - x1)*exp(dG_dns[i]*RT_inv))
        return dgammas_dT


class IdealSolution(GibbsExcess):
    def __init__(self, T=None, xs=None):
        if T is not None:
            self.T = T
        if xs is not None:
            self.xs = xs
            self.N = len(xs)
            self.cmps = range(self.N)

    def to_T_xs(self, T, xs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.N = len(xs)
        new.cmps = range(new.N)
        return new

    def GE(self):
        return 0.0

    def dGE_dT(self):
        return 0.0

    def d2GE_dT2(self):
        return 0.0

    def d3GE_dT3(self):
        return 0.0

    def d2GE_dTdxs(self):
        return [0.0 for i in self.cmps]

    def dGE_dxs(self):
        return [0.0 for i in self.cmps]

    def d2GE_dxixjs(self):
        N = self.N
        return [[0.0]*N for i in self.cmps]

    def d3GE_dxixjxks(self):
        N, cmps = self.N, self.cmps
        return [[[0.0]*N for i in cmps] for j in cmps]

    def gammas(self):
        return [1.0 for i in self.cmps]
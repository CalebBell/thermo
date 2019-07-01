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
from thermo.activity import GibbsExcess
from math import log, exp
from fluids.constants import R

__all__ = ['RegularSolution']

class RegularSolution(GibbsExcess):
    def __init__(self, T, xs, Vs, SPs, lambda_coeffs):
        self.T = T
        self.xs = xs
        self.lambda_coeffs = lambda_coeffs
        self.Vs = Vs
        self.SPs = SPs
        self.N = N = len(Vs)
        self.cmps = range(N)
        xsVs = 0.0
        for i in self.cmps:
            xsVs += xs[i]*Vs[i]
        self.xsVs = xsVs
        self.xsVs_inv = 1.0/xsVs

    def to_T_xs(self, T, xs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.SPs = self.SPs
        new.Vs = Vs = self.Vs
        new.N = self.N
        new.lambda_coeffs = self.lambda_coeffs
        new.cmps = self.cmps

        xsVs = 0.0
        for i in self.cmps:
            xsVs += xs[i]*Vs[i]
        new.xsVs = xsVs
        new.xsVs_inv = 1.0/xsVs
        return new
    

    def GE(self):
        try:
            return self._GE
        except AttributeError:
            pass
        cmps = self.cmps
        xs, SPs, Vs, coeffs = self.xs, self.SPs, self.Vs, self.lambda_coeffs
        
        num = 0.0
        for i in cmps:
            coeffsi = coeffs[i]
            for j in cmps:
                SPi_m_SPj = SPs[i] - SPs[j]
                Aij = 0.5*SPi_m_SPj*SPi_m_SPj + coeffsi[j]*SPs[i]*SPs[j]
                num += xs[i]*xs[j]*Vs[i]*Vs[j]*Aij
        GE = num*self.xsVs_inv
        self._GE = GE
        return GE


    def dGE_dxs(self):
        try:
            return self._dGE_dxs
        except AttributeError:
            pass
        cmps, xsVs_inv = self.cmps, self.xsVs_inv
        xs, SPs, Vs, coeffs = self.xs, self.SPs, self.Vs, self.lambda_coeffs
        GE = self.GE()
        
        self._dGE_dxs = dGE_dxs = []
        
        for i in cmps:
            # i is what is being differentiated
            tot = 0.0
            for j in cmps:
                SPi_m_SPj = SPs[i] - SPs[j]
                Hij = SPs[i]*SPs[j]*(coeffs[i][j] + coeffs[j][i]) + SPi_m_SPj*SPi_m_SPj
                tot += Vs[i]*Vs[j]*xs[j]*Hij
            dGE_dxs.append((tot - GE*Vs[i])*xsVs_inv)
        return dGE_dxs

    def d2GE_dxixjs(self):
        try:
            return self._d2GE_dxixjs
        except AttributeError:
            pass
        cmps, xsVs_inv = self.cmps, self.xsVs_inv
        xs, SPs, Vs, coeffs = self.xs, self.SPs, self.Vs, self.lambda_coeffs
        GE, dGE_dxs = self.GE(), self.dGE_dxs()
        
        self._d2GE_dxixjs = d2GE_dxixjs = []
        
        Hi_sums = []
        for i in cmps:
            t = 0.0
            for j in cmps:
                SPi_m_SPj = SPs[i] - SPs[j]
                Hi = SPs[i]*SPs[j]*(coeffs[i][j] + coeffs[j][i]) + SPi_m_SPj*SPi_m_SPj
                t += Vs[i]*Vs[j]*xs[j]*Hi
            Hi_sums.append(t)

        for i in cmps:
            row = []
            for j in cmps:
                SPi_m_SPj = SPs[i] - SPs[j]
                Hi = SPs[i]*SPs[j]*(coeffs[i][j] + coeffs[j][i]) + SPi_m_SPj*SPi_m_SPj
                tot = Vs[j]*(Vs[i]*GE - Hi_sums[i])*xsVs_inv*xsVs_inv - Vs[i]*dGE_dxs[j]*xsVs_inv + Vs[i]*Vs[j]*Hi*xsVs_inv
                
                row.append(tot)
            d2GE_dxixjs.append(row)
        return d2GE_dxixjs
                
    
    def d2GE_dTdxs(self):
        try:
            return self._dGE_dxs
        except AttributeError:
            return self.dGE_dxs()
       
    def dGE_dT(self):
        return 0.0

    def d2GE_dT2(self):
        return 0.0

    def d3GE_dT3(self):
        return 0.0

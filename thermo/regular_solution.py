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
        # lambda_coeffs is N*N of zeros for no interaction parameters
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
        r'''
        
        .. math::
            G^E = \frac{\sum_m \sum_n (x_m x_n V_m V_n A_{mn})}{\sum_m x_m V_m}
        
        .. math::
            A_{mn} = 0.5(\delta_m - \delta_n)^2 - \delta_m \delta_n k_{mn}
            
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
        r'''
        .. math::
            \frac{\partial G^E}{\partial x_i} = \frac{-V_i G^E + \sum_m V_i V_m 
            x_m[\delta_i\delta_m(k_{mi} + k_{im}) + (\delta_i - \delta_m)^2 ]}
            {\sum_m V_m x_m}
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
        r'''
        .. math::
            \frac{\partial^2 G^E}{\partial x_i \partial x_j} = \frac{V_j(V_i G^E - H_{ij})}{(\sum_m V_m x_m)^2}
            - \frac{V_i \frac{\partial G^E}{\partial x_j}}{\sum_m V_m x_m}
            + \frac{V_i V_j[\delta_i\delta_j(k_{ji} + k_{ij}) + (\delta_i - \delta_j)^2] }{\sum_m V_m x_m}
        '''
        
        '''
        d2GEdxixjs = diff((diff(GE, x0)).subs(GE, GEvar(x0, x1, x2)), x1).subs(Hi, H(x0, x1, x2))
        d2GEdxixjs
        '''
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
                
    def d3GE_dxixjxks(self):
        r'''
        
        .. math::
            \frac{\partial^3 G^E}{\partial x_i \partial x_j \partial x_k} = \frac{-2V_iV_jV_k G^E + 2 V_j V_k H_{ij}} {(\sum_m V_m x_m)^3}
            + \frac{V_i\left(V_j\frac{\partial G^E}{\partial x_k} + V_k\frac{\partial G^E}{\partial x_j}  \right)} {(\sum_m V_m x_m)^2}
            - \frac{V_i \frac{\partial^2 G^E}{\partial x_j \partial x_k}}{\sum_m V_m x_m}
            - \frac{V_iV_jV_k[\delta_i(\delta_j(k_{ij} + k_{ji}) + \delta_k(k_{ik} + k_{ki})) + (\delta_i - \delta_j)^2 + (\delta_i - \delta_k)^2 ]}{(\sum_m V_m x_m)^2}
        '''
        try:
            return self._d3GE_dxixjxks
        except:
            pass
        cmps, xsVs_inv = self.cmps, self.xsVs_inv
        xs, SPs, Vs, coeffs = self.xs, self.SPs, self.Vs, self.lambda_coeffs
        GE, dGE_dxs, d2GE_dxixjs = self.GE(), self.dGE_dxs(), self.d2GE_dxixjs()

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
                    
                    
                    
                    tot = firsts*xsVs_inv + seconds*xsVs_inv*xsVs_inv + thirds*xsVs_inv*xsVs_inv*xsVs_inv
                    dG_row.append(tot)
                dG_matrix.append(dG_row)
            d3GE_dxixjxks.append(dG_matrix)
        return d3GE_dxixjxks
    
    def d2GE_dTdxs(self):
        return [0.0]*self.N
       
    def dGE_dT(self):
        return 0.0

    def d2GE_dT2(self):
        return 0.0

    def d3GE_dT3(self):
        return 0.0

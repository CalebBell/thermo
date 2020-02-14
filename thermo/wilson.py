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
SOFTWARE.'''

from __future__ import division
from thermo.activity import GibbsExcess
from math import log, exp
from fluids.constants import R

__all__ = ['Wilson']


class Wilson(GibbsExcess):
    @staticmethod
    def from_DDBST(Vi, Vj, a, b, c, d=0.0, e=0.0, f=0.0, unit_conversion=True):
        '''Converts parameters for the wilson equation in the DDBST to the
        basis used in this implementation.
        '''
        if unit_conversion:
            R = 1.9872042586408316 # DDBST document suggests  1.9858775
        else:
            R = 1.0 # Not used in some cases?
        a, b = log(Vj/Vi) - b/R, -a/R
        c, d = -d/R, -c/R
        e = -e/R
        f = -f/R
        return (a, b, c, d, e, f)
    
    @staticmethod
    def from_DDBST_as_matrix(Vs, ais, bis, cis, dis, eis, fis, 
                             unit_conversion=True):
        
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
        self.lambda_coeffs = lambda_coeffs
        if ABCDEF is not None:
            (self.lambda_coeffs_A, self.lambda_coeffs_B, self.lambda_coeffs_C, 
            self.lambda_coeffs_D, self.lambda_coeffs_E, self.lambda_coeffs_F) = ABCDEF
            self.N = N = len(self.lambda_coeffs_A)
        else:
            if lambda_coeffs is not None:
                self.lambda_coeffs_A = [[i[0] for i in l] for l in lambda_coeffs]
                self.lambda_coeffs_B = [[i[1] for i in l] for l in lambda_coeffs]
                self.lambda_coeffs_C = [[i[2] for i in l] for l in lambda_coeffs]
                self.lambda_coeffs_D = [[i[3] for i in l] for l in lambda_coeffs]
                self.lambda_coeffs_E = [[i[4] for i in l] for l in lambda_coeffs]
                self.lambda_coeffs_F = [[i[5] for i in l] for l in lambda_coeffs]
            else:
                self.lambda_coeffs_A = None
                self.lambda_coeffs_B = None
                self.lambda_coeffs_C = None
                self.lambda_coeffs_D = None
                self.lambda_coeffs_E = None
                self.lambda_coeffs_F = None
            self.N = N = len(lambda_coeffs)
        self.cmps = range(N)
        
    def to_T_xs(self, T, xs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.N = self.N
        new.cmps = self.cmps
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
        r'''Calculate the `lambda` terms for the Wilson model for the system
        temperature.
        
        .. math::
            \Lambda_{ij} = \exp\left[a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T 
                    + d_{ij}T + \frac{e_{ij}}{T^2} + f_{ij}{T^2}\right]
            
            
        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._lambdas
        except AttributeError:
            pass
        # 87% of the time of this routine is the exponential.
        lambda_coeffs_A = self.lambda_coeffs_A
        lambda_coeffs_B = self.lambda_coeffs_B
        lambda_coeffs_C = self.lambda_coeffs_C
        lambda_coeffs_D = self.lambda_coeffs_D
        lambda_coeffs_E = self.lambda_coeffs_E
        lambda_coeffs_F = self.lambda_coeffs_F
        T = self.T
        cmps = self.cmps

        T2 = T*T
        Tinv = 1.0/T
        T2inv = Tinv*Tinv
        logT = log(T)

        self._lambdas = lambdas = []
        for i in cmps:
            lambda_coeffs_Ai = lambda_coeffs_A[i]
            lambda_coeffs_Bi = lambda_coeffs_B[i]
            lambda_coeffs_Ci = lambda_coeffs_C[i]
            lambda_coeffs_Di = lambda_coeffs_D[i]
            lambda_coeffs_Ei = lambda_coeffs_E[i]
            lambda_coeffs_Fi = lambda_coeffs_F[i]
            lambdasi = [exp(lambda_coeffs_Ai[j] + lambda_coeffs_Bi[j]*Tinv 
                        + lambda_coeffs_Ci[j]*logT + lambda_coeffs_Di[j]*T 
                        + lambda_coeffs_Ei[j]*T2inv + lambda_coeffs_Fi[j]*T2)
                        for j in cmps]
            lambdas.append(lambdasi)
        
        return lambdas

    def dlambdas_dT(self):
        r'''Calculate the temperature derivative of the `lambda` terms for the
        Wilson model for a specified temperature.
        
        .. math::
            \frac{\partial \Lambda_{ij}}{\partial T} = 
            \left(2 T h_{ij} + d_{ij} + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}} 
            - \frac{2 e_{ij}}{T^{3}}\right) e^{T^{2} h_{ij} + T d_{ij} + a_{ij} 
            + c_{ij} \log{\left(T \right)} + \frac{b_{ij}}{T} 
            + \frac{e_{ij}}{T^{2}}}
            
            
        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._dlambdas_dT
        except AttributeError:
            pass

        lambda_coeffs_B = self.lambda_coeffs_B
        lambda_coeffs_C = self.lambda_coeffs_C
        lambda_coeffs_D = self.lambda_coeffs_D
        lambda_coeffs_E = self.lambda_coeffs_E
        lambda_coeffs_F = self.lambda_coeffs_F
        
        T, cmps = self.T, self.cmps
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        self._dlambdas_dT = dlambdas_dT = []
        
        T2 = T + T
        Tinv = 1.0/T
        nT2inv = -Tinv*Tinv
        nT3inv2 = 2.0*nT2inv*Tinv
                
        for i in cmps:
            lambdasi = lambdas[i]
            lambda_coeffs_Bi = lambda_coeffs_B[i]
            lambda_coeffs_Ci = lambda_coeffs_C[i]
            lambda_coeffs_Di = lambda_coeffs_D[i]
            lambda_coeffs_Ei = lambda_coeffs_E[i]
            lambda_coeffs_Fi = lambda_coeffs_F[i]
            dlambdas_dTi = [(T2*lambda_coeffs_Fi[j] + lambda_coeffs_Di[j]
                             + lambda_coeffs_Ci[j]*Tinv + lambda_coeffs_Bi[j]*nT2inv
                             + lambda_coeffs_Ei[j]*nT3inv2)*lambdasi[j]
                            for j in cmps]
            dlambdas_dT.append(dlambdas_dTi)
        return dlambdas_dT

    def d2lambdas_dT2(self):
        r'''Calculate the second temperature derivative of the `lambda` terms
         for the Wilson model for a specified temperature.
        
        .. math::
            \frac{\partial^2 \Lambda_{ij}}{\partial^2 T} = 
            \left(2 f_{ij} + \left(2 T f_{ij} + d_{ij} + \frac{c_{ij}}{T}
            - \frac{b_{ij}}{T^{2}} - \frac{2 e_{ij}}{T^{3}}\right)^{2} 
                - \frac{c_{ij}}{T^{2}} + \frac{2 b_{ij}}{T^{3}} 
                + \frac{6 e_{ij}}{T^{4}}\right) e^{T^{2} f_{ij} + T d_{ij} 
                + a_{ij} + c_{ij} \log{\left(T \right)} + \frac{b_{ij}}{T} 
                + \frac{e_{ij}}{T^{2}}}
            
            
        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._d2lambdas_dT2
        except AttributeError:
            pass
        lambda_coeffs_B = self.lambda_coeffs_B
        lambda_coeffs_C = self.lambda_coeffs_C
        lambda_coeffs_E = self.lambda_coeffs_E
        lambda_coeffs_F = self.lambda_coeffs_F
        T, cmps = self.T, self.cmps
        
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        try:
            dlambdas_dT = self._dlambdas_dT
        except AttributeError:
            dlambdas_dT = self.dlambdas_dT()
        
        Tinv = 1.0/T
        nT2inv = -Tinv*Tinv
        T3inv2 = -2.0*nT2inv*Tinv
        T4inv6 = 3.0*T3inv2*Tinv

        self._d2lambdas_dT2 = d2lambdas_dT2s = []
        for i in cmps:
            lambdasi = lambdas[i]
            dlambdas_dTi = dlambdas_dT[i]
            lambda_coeffs_Bi = lambda_coeffs_B[i]
            lambda_coeffs_Ci = lambda_coeffs_C[i]
            lambda_coeffs_Ei = lambda_coeffs_E[i]
            lambda_coeffs_Fi = lambda_coeffs_F[i]
            d2lambdas_dT2i = [(2.0*lambda_coeffs_Fi[j] + nT2inv*lambda_coeffs_Ci[j]
                             + T3inv2*lambda_coeffs_Bi[j] + T4inv6*lambda_coeffs_Ei[j]
                               )*lambdasi[j] + dlambdas_dTi[j]*dlambdas_dTi[j]/lambdasi[j] 
                             for j in cmps]
            d2lambdas_dT2s.append(d2lambdas_dT2i)
        return d2lambdas_dT2s
    
    def d3lambdas_dT3(self):
        r'''Calculate the third temperature derivative of the `lambda` terms
         for the Wilson model for a specified temperature.
        
        .. math::
            \frac{\partial^3 \Lambda_{ij}}{\partial^3 T} = 
            \left(3 \left(2 f_{ij} - \frac{c_{ij}}{T^{2}} + \frac{2 b_{ij}}{T^{3}} 
            + \frac{6 e_{ij}}{T^{4}}\right) \left(2 T f_{ij} + d_{ij}
            + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}} - \frac{2 e_{ij}}{T^{3}}\right) 
            + \left(2 T f_{ij} + d_{ij} + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}}
            - \frac{2 e_{ij}}{T^{3}}\right)^{3} - \frac{2 \left(- c_{ij} 
            + \frac{3 b_{ij}}{T} + \frac{12 e_{ij}}{T^{2}}\right)}{T^{3}}\right)
            e^{T^{2} f_{ij} + T d_{ij} + a_{ij} + c_{ij} \log{\left(T \right)}
            + \frac{b_{ij}}{T} + \frac{e_{ij}}{T^{2}}}
            
        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._d3lambdas_dT3
        except:
            pass
        
        T, cmps = self.T, self.cmps            
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
        
        Tinv = 1.0/T
        Tinv3 = 3.0*Tinv
        nT2inv = -Tinv*Tinv
        nT2inv05 = 0.5*nT2inv
        T3inv = -nT2inv*Tinv
        T3inv2 = T3inv+T3inv
        T4inv6 = 3.0*T3inv2*Tinv
        T4inv3 = 1.5*T3inv2*Tinv
        T2_12 = -12.0*nT2inv

        self._d3lambdas_dT3 = d3lambdas_dT3s = []
        for i in cmps:
            lambdasi = lambdas[i]
            dlambdas_dTi = dlambdas_dT[i]
            lambda_coeffs_Bi = lambda_coeffs_B[i]
            lambda_coeffs_Ci = lambda_coeffs_C[i]
            lambda_coeffs_Ei = lambda_coeffs_E[i]
            lambda_coeffs_Fi = lambda_coeffs_F[i]
            d3lambdas_dT3is = []
            for j in cmps:
                term2 = (lambda_coeffs_Fi[j] + nT2inv05*lambda_coeffs_Ci[j]
                         + T3inv*lambda_coeffs_Bi[j] + T4inv3*lambda_coeffs_Ei[j])
                
                term3 = dlambdas_dTi[j]/lambdasi[j]
                
                term4 = (T3inv2*(lambda_coeffs_Ci[j] - Tinv3*lambda_coeffs_Bi[j]
                         - T2_12*lambda_coeffs_Ei[j]))
                
                d3lambdas_dT3is.append((term3*(6.0*term2 + term3*term3) + term4)*lambdasi[j])
            
            d3lambdas_dT3s.append(d3lambdas_dT3is)
        return d3lambdas_dT3s

    def xj_Lambda_ijs(self):
        try:
            return self._xj_Lambda_ijs
        except AttributeError:
            pass
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        
        xs, cmps = self.xs, self.cmps
        self._xj_Lambda_ijs = xj_Lambda_ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*lambdas[i][j]
            xj_Lambda_ijs.append(tot)
        return xj_Lambda_ijs

    def xj_Lambda_ijs_inv(self):
        try:
            return self._xj_Lambda_ijs_inv
        except AttributeError:
            pass
        
        try:
            xj_Lambda_ijs = self._xj_Lambda_ijs
        except AttributeError:
            xj_Lambda_ijs = self.xj_Lambda_ijs()
        self._xj_Lambda_ijs_inv = [1.0/x for x in xj_Lambda_ijs]
        return self._xj_Lambda_ijs_inv

    def log_xj_Lambda_ijs(self):
        try:
            return self._log_xj_Lambda_ijs
        except AttributeError:
            pass
        try:
            xj_Lambda_ijs = self._xj_Lambda_ijs
        except AttributeError:
            xj_Lambda_ijs = self.xj_Lambda_ijs()
        self._log_xj_Lambda_ijs = [log(i) for i in xj_Lambda_ijs]
        return self._log_xj_Lambda_ijs
        

    def xj_dLambda_dTijs(self):
        try:
            return self._xj_dLambda_dTijs
        except AttributeError:
            pass
        try:
            dlambdas_dT = self._dlambdas_dT
        except AttributeError:
            dlambdas_dT = self.dlambdas_dT()

        xs, cmps = self.xs, self.cmps
        self._xj_dLambda_dTijs = xj_dLambda_dTijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*dlambdas_dT[i][j]
            xj_dLambda_dTijs.append(tot)
        return self._xj_dLambda_dTijs


    def xj_d2Lambda_dT2ijs(self):
        try:
            return self._xj_d2Lambda_dT2ijs
        except AttributeError:
            pass
        try:
            d2lambdas_dT2 = self._d2lambdas_dT2
        except AttributeError:
            d2lambdas_dT2 = self.d2lambdas_dT2()
        
        xs, cmps = self.xs, self.cmps
        self._xj_d2Lambda_dT2ijs = xj_d2Lambda_dT2ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*d2lambdas_dT2[i][j]
            xj_d2Lambda_dT2ijs.append(tot)
        return xj_d2Lambda_dT2ijs

    def xj_d3Lambda_dT3ijs(self):
        try:
            return self._xj_d3Lambda_dT3ijs
        except AttributeError:
            pass
        try:
            d3lambdas_dT3 = self._d3lambdas_dT3
        except AttributeError:
            d3lambdas_dT3 = self.d3lambdas_dT3()
        
        xs, cmps = self.xs, self.cmps
        self._xj_d3Lambda_dT3ijs = xj_d3Lambda_dT3ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*d3lambdas_dT3[i][j]
            xj_d3Lambda_dT3ijs.append(tot)
        return xj_d3Lambda_dT3ijs

    
    def GE(self):
        try:
            return self._GE
        except AttributeError:
            pass
        
        try:
            log_xj_Lambda_ijs = self._log_xj_Lambda_ijs
        except AttributeError:
            log_xj_Lambda_ijs = self.log_xj_Lambda_ijs()
        
        T, xs, cmps = self.T, self.xs, self.cmps
        GE = 0.0
        for i in cmps:
            GE += xs[i]*log_xj_Lambda_ijs[i]
        self._GE = GE = -GE*R*T
        return GE
    
    def dGE_dT(self):
        r'''
        
        .. math::
            \frac{\partial G^E}{\partial T} = -R\sum_i x_i \log\left(\sum_j x_i \Lambda_{ij}\right)
            -RT\sum_i \frac{x_i \sum_j x_j \frac{\Lambda _{ij}}{\partial T}}{\sum_j x_j \Lambda_{ij}}
        '''
        
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
        T, xs, cmps = self.T, self.xs, self.cmps
        
        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()
        
        try:
            xj_dLambda_dTijs = self._xj_dLambda_dTijs
        except AttributeError:
            xj_dLambda_dTijs = self.xj_dLambda_dTijs()
        
        try:
            tot = self._GE/T # First term
        except AttributeError:
            tot = self.GE()/T # First term
            
        # Second term
        sum1 = 0.0
        for i in cmps:
            sum1 += xs[i]*xj_dLambda_dTijs[i]*xj_Lambda_ijs_inv[i]
        tot -= T*R*sum1
        
        self._dGE_dT = tot
        return tot
    



    def d2GE_dT2(self):
        r'''
        .. math::
            \frac{\partial^2 G^E}{\partial T^2} = -R\left[T\sum_i \left(\frac{x_i \sum_j (x_j \frac{\partial^2 \Lambda_{ij}}{\partial T^2} )}{\sum_j x_j \Lambda_{ij}}
            - \frac{x_i (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T}  )^2}{(\sum_j x_j \Lambda_{ij})^2}
            \right)
            + 2\sum_i \left(\frac{x_i \sum_j  x_j \frac{\partial \Lambda_{ij}}{\partial T}}{\sum_j x_j \Lambda_{ij}}
            \right)
            \right]
        '''
        try:
            return self._d2GE_dT2
        except AttributeError:
            pass
        T, xs, cmps = self.T, self.xs, self.cmps        
        
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

        # Last term, also the same term as last term of dGE_dT
        sum0, sum1 = 0.0, 0.0
        for i in cmps:
            t = xs[i]*xj_Lambda_ijs_inv[i]
            t2 = xj_dLambda_dTijs[i]*t
            sum1 += t2
            
            sum0 += t*(xj_d2Lambda_dT2ijs[i] - xj_dLambda_dTijs[i]*xj_dLambda_dTijs[i]*xj_Lambda_ijs_inv[i])
        
        self._d2GE_dT2 = -R*(T*sum0 + 2.0*sum1)
        return self._d2GE_dT2
    
    
    def d3GE_dT3(self):
        r'''
        .. math::
            \frac{\partial^3 G^E}{\partial T^3} = -R\left[3\left(\frac{x_i \sum_j (x_j \frac{\partial^2 \Lambda_{ij}}{\partial T^2} )}{\sum_j x_j \Lambda_{ij}}
            - \frac{x_i (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T}  )^2}{(\sum_j x_j \Lambda_{ij})^2}
            \right)
            +T\left(
            \sum_i \frac{x_i (\sum_j x_j \frac{\partial^3 \Lambda _{ij}}{\partial T^3})}{\sum_j x_j \Lambda_{ij}}
            - \frac{3x_i (\sum_j x_j \frac{\partial \Lambda_{ij}^2}{\partial T^2})  (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T})}{(\sum_j x_j \Lambda_{ij})^2}
            + 2\frac{x_i(\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T})^3}{(\sum_j x_j \Lambda_{ij})^3}
            \right)\right]
            
        '''
        try:
            return self._d3GE_dT3
        except AttributeError:
            pass

        T, xs, cmps = self.T, self.xs, self.cmps
        xj_Lambda_ijs = self.xj_Lambda_ijs()
        xj_dLambda_dTijs = self.xj_dLambda_dTijs()
        xj_d2Lambda_dT2ijs = self.xj_d2Lambda_dT2ijs()
        xj_d3Lambda_dT3ijs = self.xj_d3Lambda_dT3ijs()
        
        #Term is directly from the one above it
        sum0 = 0.0
        for i in cmps:
            sum0 += (xs[i]*xj_d2Lambda_dT2ijs[i]/xj_Lambda_ijs[i]
                    - xs[i]*(xj_dLambda_dTijs[i]*xj_dLambda_dTijs[i])/(xj_Lambda_ijs[i]*xj_Lambda_ijs[i]))

        sum_d3 = 0.0
        for i in cmps:
            sum_d3 += xs[i]*xj_d3Lambda_dT3ijs[i]/xj_Lambda_ijs[i]
            
        sum_comb = 0.0
        for i in cmps:
            sum_comb += 3.0*xs[i]*xj_d2Lambda_dT2ijs[i]*xj_dLambda_dTijs[i]/(xj_Lambda_ijs[i]*xj_Lambda_ijs[i])
        
        sum_last = 0.0
        for i in cmps:
            sum_last += 2.0*xs[i]*(xj_dLambda_dTijs[i])**3/(xj_Lambda_ijs[i]*xj_Lambda_ijs[i]*xj_Lambda_ijs[i])
        
        self._d3GE_dT3 = -R*(3.0*sum0 + T*(sum_d3 - sum_comb + sum_last))
        return self._d3GE_dT3
    

    def d2GE_dTdxs(self):
        r'''
        
        .. math::
            \frac{\partial^2 G^E}{\partial x_k \partial T} = -R\left[T\left(   
            \sum_i  \left(\frac{x_i \frac{\partial n_{ik}}{\partial T}}{\sum_j x_j \Lambda_{ij}} 
            - \frac{x_i \Lambda_{ik} (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T} )}{(\partial_j x_j \Lambda_{ij})^2}
            \right) + \frac{\sum_i x_i \frac{\partial \Lambda_{ki}}{\partial T}}{\sum_j x_j \Lambda_{kj}}
            \right)
            + \log\left(\sum_i x_i \Lambda_{ki}\right)
            + \sum_i \frac{x_i \Lambda_{ik}}{\sum_j x_j \Lambda_{ij}}
            \right]
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
            
        T, xs, cmps = self.T, self.xs, self.cmps
        
        self._d2GE_dTdxs = d2GE_dTdxs = []
        for k in cmps:
            tot1 = xj_dLambda_dTijs[k]*xj_Lambda_ijs_inv[k]
            tot2 = 0.0
            for i in cmps:
                t1 = lambdas[i][k]*xj_Lambda_ijs_inv[i]
                tot1 += xs[i]*xj_Lambda_ijs_inv[i]*(dlambdas_dT[i][k]
                                                   - xj_dLambda_dTijs[i]*t1)
                tot2 += xs[i]*t1

            dG = -R*(T*tot1 + log_xj_Lambda_ijs[k] + tot2)
            
            d2GE_dTdxs.append(dG)
        return d2GE_dTdxs


    def dGE_dxs(self):
        r'''
        
        .. math::
            \frac{\partial G^E}{\partial x_k} = -RT\left[
            \sum_i \frac{x_i \Lambda_{ik}}{\sum_j \Lambda_{ij}x_j }
            + \log\left(\sum_j x_j \Lambda_{kj}\right)
            \right]
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
        
        T, xs, cmps = self.T, self.xs, self.cmps
        mRT = -T*R
        
        # k = what is being differentiated with respect to
        self._dGE_dxs = dGE_dxs = []
        for k in cmps:
            tot = log_xj_Lambda_ijs[k]
            for i in cmps:
                tot += xs[i]*lambdas[i][k]*xj_Lambda_ijs_inv[i]
            
            dGE_dxs.append(mRT*tot)
            
        return dGE_dxs
    
    def d2GE_dxixjs(self):
        r'''
        .. math::
            \frac{\partial^2 G^E}{\partial x_k \partial x_m} = RT\left(
            \sum_i \frac{x_i \Lambda_{ik} \Lambda_{im}}{(\sum_j x_j \Lambda_{ij})^2}
            -\frac{\Lambda_{km}}{\sum_j x_j \Lambda_{kj}}
            -\frac{\Lambda_{mk}}{\sum_j x_j \Lambda_{mj}}
            \right)
        '''
        try:
            return self._d2GE_dxixjs
        except AttributeError:
            pass
        # Correct, tested with hessian
        T, xs, cmps = self.T, self.xs, self.cmps
        RT = R*T
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas()
        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()

        
        self._d2GE_dxixjs = d2GE_dxixjs = []
        for k in cmps:
            dG_row = []
            for m in cmps:
                tot = 0.0
                for i in cmps:
                    tot += xs[i]*lambdas[i][k]*lambdas[i][m]*(xj_Lambda_ijs_inv[i]*xj_Lambda_ijs_inv[i])
                tot -= lambdas[k][m]*xj_Lambda_ijs_inv[k]
                tot -= lambdas[m][k]*xj_Lambda_ijs_inv[m]
                dG_row.append(RT*tot)
            d2GE_dxixjs.append(dG_row)
        return d2GE_dxixjs

    def d3GE_dxixjxks(self):
        r'''
        .. math::
            \frac{\partial^3 G^E}{\partial x_k \partial x_m \partial x_n}
            = -RT\left[
            \sum_i \left(\frac{2x_i \Lambda_{ik}\Lambda_{im}\Lambda_{in}} {(\sum x_j \Lambda_{ij})^3}\right)
            - \frac{\Lambda_{km} \Lambda_{kn}}{(\sum_j x_j \Lambda_{kj})^2}
            - \frac{\Lambda_{mk} \Lambda_{mn}}{(\sum_j x_j \Lambda_{mj})^2}
            - \frac{\Lambda_{nk} \Lambda_{nm}}{(\sum_j x_j \Lambda_{nj})^2}
            \right]
        '''
        try:
            return self._d3GE_dxixjxks
        except:
            pass
        # Correct, tested with sympy expanding
        T, xs, cmps = self.T, self.xs, self.cmps
        nRT = -R*T
        lambdas = self.lambdas()
        xj_Lambda_ijs = self.xj_Lambda_ijs()


        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()
        
        # all the same: analytical[i][j][k] = analytical[i][k][j] = analytical[j][i][k] = analytical[j][k][i] = analytical[k][i][j] = analytical[k][j][i] = float(v)
        self._d3GE_dxixjxks = d3GE_dxixjxks = []
        for k in cmps:
            dG_matrix = []
            for m in cmps:
                dG_row = []
                for n in cmps:
                    tot = 0.0
                    for i in cmps:
                        num = 2.0*xs[i]*lambdas[i][k]*lambdas[i][m]*lambdas[i][n]
                        den = xj_Lambda_ijs_inv[i]*xj_Lambda_ijs_inv[i]*xj_Lambda_ijs_inv[i]
                        tot += num*den
                    
                    tot -= lambdas[k][m]*lambdas[k][n]*xj_Lambda_ijs_inv[k]*xj_Lambda_ijs_inv[k]
                    tot -= lambdas[m][k]*lambdas[m][n]*xj_Lambda_ijs_inv[m]*xj_Lambda_ijs_inv[m]
                    tot -= lambdas[n][m]*lambdas[n][k]*xj_Lambda_ijs_inv[n]*xj_Lambda_ijs_inv[n]
                    dG_row.append(nRT*tot)
                dG_matrix.append(dG_row)
            d3GE_dxixjxks.append(dG_matrix)
        return d3GE_dxixjxks
    
    
    def gammas(self):
        try:
            return self._gammas
        except AttributeError:
            pass
        xs, cmps = self.xs, self.cmps
        try:
            lambdas = self._lambdas
        except AttributeError:
            lambdas = self.lambdas() 
        try:
            xj_Lambda_ijs_inv = self._xj_Lambda_ijs_inv
        except AttributeError:
            xj_Lambda_ijs_inv = self.xj_Lambda_ijs_inv()

        xj_over_xj_Lambda_ijs = [xs[j]*xj_Lambda_ijs_inv[j] for j in cmps]
        
        self._gammas = gammas = []
        for i in cmps:
            tot2 = 1.0
            for j in cmps:
                tot2 -= lambdas[j][i]*xj_over_xj_Lambda_ijs[j]
#                tot2 -= lambdas[j][i]*xj_Lambda_ijs_inv[j]*xs[j]

            gammas.append(exp(tot2)*xj_Lambda_ijs_inv[i])
        return gammas

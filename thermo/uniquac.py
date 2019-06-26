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
from thermo.activity import GibbsExcess, UNIQUAC_gammas
from math import log, exp
from fluids.constants import R

__all__ = ['UNIQUAC']

class UNIQUAC(GibbsExcess):
    z = 10.0
    def __init__(self, T, xs, rs, qs, tau_coeffs=None,
                 ABCDEF=None):
        self.T = T
        self.xs = xs
        self.rs = rs
        self.qs = qs

        if ABCDEF is not None:
            (self.tau_coeffs_A, self.tau_coeffs_B, self.tau_coeffs_C, 
            self.tau_coeffs_D, self.tau_coeffs_E, self.tau_coeffs_F) = ABCDEF
            self.N = N = len(self.tau_coeffs_A)
        else:
            self.tau_coeffs = tau_coeffs
            if tau_coeffs is not None:
                self.tau_coeffs_A = [[i[0] for i in l] for l in tau_coeffs]
                self.tau_coeffs_B = [[i[1] for i in l] for l in tau_coeffs]
                self.tau_coeffs_C = [[i[2] for i in l] for l in tau_coeffs]
                self.tau_coeffs_D = [[i[3] for i in l] for l in tau_coeffs]
                self.tau_coeffs_E = [[i[4] for i in l] for l in tau_coeffs]
                self.tau_coeffs_F = [[i[5] for i in l] for l in tau_coeffs]
            else:
                self.tau_coeffs_A = None
                self.tau_coeffs_B = None
                self.tau_coeffs_C = None
                self.tau_coeffs_D = None
                self.tau_coeffs_E = None
                self.tau_coeffs_F = None
    
            self.N = N = len(self.tau_coeffs_A)
        self.cmps = range(N)
        self.zero_coeffs = [[0.0]*N for _ in range(N)]

    def to_T_xs(self, T, xs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.rs = self.rs
        new.qs = self.qs
        new.N = self.N
        new.cmps = self.cmps
        new.zero_coeffs = self.zero_coeffs
        
        (new.tau_coeffs_A, new.tau_coeffs_B, new.tau_coeffs_C, 
         new.tau_coeffs_D, new.tau_coeffs_E, new.tau_coeffs_F) = (self.tau_coeffs_A, self.tau_coeffs_B, self.tau_coeffs_C, 
                         self.tau_coeffs_D, self.tau_coeffs_E, self.tau_coeffs_F)
        
        if T == self.T:
            try:
                new._taus = self._taus
            except AttributeError:
                pass
            try:
                new._dtaus_dT = self._dtaus_dT
            except AttributeError:
                pass
            try:
                new._d2taus_dT2 = self._d2taus_dT2
            except AttributeError:
                pass
            try:
                new._d3taus_dT3 = self._d3taus_dT3
            except AttributeError:
                pass            
        return new

    def taus(self):
        r'''Calculate the `tau` terms for the UNIQUAC model for the system
        temperature.
        
        .. math::
            \tau_{ij} = \exp\left[a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T 
                    + d_{ij}T + \frac{e_{ij}}{T^2} + f_{ij}{T^2}\right]
            
            
        These `tau ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._taus
        except AttributeError:
            pass
        # 87% of the time of this routine is the exponential.
        tau_coeffs_A = self.tau_coeffs_A
        tau_coeffs_B = self.tau_coeffs_B
        tau_coeffs_C = self.tau_coeffs_C
        tau_coeffs_D = self.tau_coeffs_D
        tau_coeffs_E = self.tau_coeffs_E
        tau_coeffs_F = self.tau_coeffs_F
        T = self.T
        cmps = self.cmps

        T2 = T*T
        Tinv = 1.0/T
        T2inv = Tinv*Tinv
        logT = log(T)

        self._taus = taus = []
        for i in cmps:
            tau_coeffs_Ai = tau_coeffs_A[i]
            tau_coeffs_Bi = tau_coeffs_B[i]
            tau_coeffs_Ci = tau_coeffs_C[i]
            tau_coeffs_Di = tau_coeffs_D[i]
            tau_coeffs_Ei = tau_coeffs_E[i]
            tau_coeffs_Fi = tau_coeffs_F[i]
            tausi = [exp(tau_coeffs_Ai[j] + tau_coeffs_Bi[j]*Tinv 
                        + tau_coeffs_Ci[j]*logT + tau_coeffs_Di[j]*T 
                        + tau_coeffs_Ei[j]*T2inv + tau_coeffs_Fi[j]*T2)
                        for j in cmps]
            taus.append(tausi)
        
        return taus

    def dtaus_dT(self):
        r'''Calculate the temperature derivative of the `tau` terms for the
        UNIQUAC model for a specified temperature.
        
        .. math::
            \frac{\partial \tau_{ij}}{\partial T} = 
            \left(2 T h_{ij} + d_{ij} + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}} 
            - \frac{2 e_{ij}}{T^{3}}\right) e^{T^{2} h_{ij} + T d_{ij} + a_{ij} 
            + c_{ij} \log{\left(T \right)} + \frac{b_{ij}}{T} 
            + \frac{e_{ij}}{T^{2}}}
            
            
        These `tau ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._dtaus_dT
        except AttributeError:
            pass

        tau_coeffs_B = self.tau_coeffs_B
        tau_coeffs_C = self.tau_coeffs_C
        tau_coeffs_D = self.tau_coeffs_D
        tau_coeffs_E = self.tau_coeffs_E
        tau_coeffs_F = self.tau_coeffs_F
        
        T, cmps = self.T, self.cmps
        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()
        self._dtaus_dT = dtaus_dT = []
        
        T2 = T + T
        Tinv = 1.0/T
        nT2inv = -Tinv*Tinv
        nT3inv2 = 2.0*nT2inv*Tinv
        
        for i in cmps:
            tausi = taus[i]
            tau_coeffs_Bi = tau_coeffs_B[i]
            tau_coeffs_Ci = tau_coeffs_C[i]
            tau_coeffs_Di = tau_coeffs_D[i]
            tau_coeffs_Ei = tau_coeffs_E[i]
            tau_coeffs_Fi = tau_coeffs_F[i]
            dtaus_dTi = [(T2*tau_coeffs_Fi[j] + tau_coeffs_Di[j]
                             + tau_coeffs_Ci[j]*Tinv + tau_coeffs_Bi[j]*nT2inv
                             + tau_coeffs_Ei[j]*nT3inv2)*tausi[j]
                            for j in cmps]
            dtaus_dT.append(dtaus_dTi)
        return dtaus_dT

    def d2taus_dT2(self):
        r'''Calculate the second temperature derivative of the `tau` terms
         for the UNIQUAC model for a specified temperature.
        
        .. math::
            \frac{\partial^2 \tau_{ij}}{\partial^2 T} = 
            \left(2 f_{ij} + \left(2 T f_{ij} + d_{ij} + \frac{c_{ij}}{T}
            - \frac{b_{ij}}{T^{2}} - \frac{2 e_{ij}}{T^{3}}\right)^{2} 
                - \frac{c_{ij}}{T^{2}} + \frac{2 b_{ij}}{T^{3}} 
                + \frac{6 e_{ij}}{T^{4}}\right) e^{T^{2} f_{ij} + T d_{ij} 
                + a_{ij} + c_{ij} \log{\left(T \right)} + \frac{b_{ij}}{T} 
                + \frac{e_{ij}}{T^{2}}}
            
            
        These `tau ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._d2taus_dT2
        except AttributeError:
            pass
        tau_coeffs_B = self.tau_coeffs_B
        tau_coeffs_C = self.tau_coeffs_C
        tau_coeffs_E = self.tau_coeffs_E
        tau_coeffs_F = self.tau_coeffs_F
        T, cmps = self.T, self.cmps
        
        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()
        try:
            dtaus_dT = self._dtaus_dT
        except AttributeError:
            dtaus_dT = self.dtaus_dT()
        
        Tinv = 1.0/T
        nT2inv = -Tinv*Tinv
        T3inv2 = -2.0*nT2inv*Tinv
        T4inv6 = 3.0*T3inv2*Tinv

        self._d2taus_dT2 = d2taus_dT2s = []
        for i in cmps:
            tausi = taus[i]
            dtaus_dTi = dtaus_dT[i]
            tau_coeffs_Bi = tau_coeffs_B[i]
            tau_coeffs_Ci = tau_coeffs_C[i]
            tau_coeffs_Ei = tau_coeffs_E[i]
            tau_coeffs_Fi = tau_coeffs_F[i]
            d2taus_dT2i = [(2.0*tau_coeffs_Fi[j] + nT2inv*tau_coeffs_Ci[j]
                             + T3inv2*tau_coeffs_Bi[j] + T4inv6*tau_coeffs_Ei[j]
                               )*tausi[j] + dtaus_dTi[j]*dtaus_dTi[j]/tausi[j] 
                             for j in cmps]
            d2taus_dT2s.append(d2taus_dT2i)
        return d2taus_dT2s
    
    def d3taus_dT3(self):
        r'''Calculate the third temperature derivative of the `tau` terms
         for the UNIQUAC model for a specified temperature.
        
        .. math::
            \frac{\partial^3 \tau_{ij}}{\partial^3 T} = 
            \left(3 \left(2 f_{ij} - \frac{c_{ij}}{T^{2}} + \frac{2 b_{ij}}{T^{3}} 
            + \frac{6 e_{ij}}{T^{4}}\right) \left(2 T f_{ij} + d_{ij}
            + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}} - \frac{2 e_{ij}}{T^{3}}\right) 
            + \left(2 T f_{ij} + d_{ij} + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}}
            - \frac{2 e_{ij}}{T^{3}}\right)^{3} - \frac{2 \left(- c_{ij} 
            + \frac{3 b_{ij}}{T} + \frac{12 e_{ij}}{T^{2}}\right)}{T^{3}}\right)
            e^{T^{2} f_{ij} + T d_{ij} + a_{ij} + c_{ij} \log{\left(T \right)}
            + \frac{b_{ij}}{T} + \frac{e_{ij}}{T^{2}}}
            
        These `tau ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._d3taus_dT3
        except AttributeError:
            pass
        
        T, cmps = self.T, self.cmps            
        tau_coeffs_B = self.tau_coeffs_B
        tau_coeffs_C = self.tau_coeffs_C
        tau_coeffs_E = self.tau_coeffs_E
        tau_coeffs_F = self.tau_coeffs_F
        
        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()
        try:
            dtaus_dT = self._dtaus_dT
        except AttributeError:
            dtaus_dT = self.dtaus_dT()
        
        Tinv = 1.0/T
        Tinv3 = 3.0*Tinv
        nT2inv = -Tinv*Tinv
        nT2inv05 = 0.5*nT2inv
        T3inv = -nT2inv*Tinv
        T3inv2 = T3inv+T3inv
        T4inv6 = 3.0*T3inv2*Tinv
        T4inv3 = 1.5*T3inv2*Tinv
        T2_12 = -12.0*nT2inv

        self._d3taus_dT3 = d3taus_dT3s = []
        for i in cmps:
            tausi = taus[i]
            dtaus_dTi = dtaus_dT[i]
            tau_coeffs_Bi = tau_coeffs_B[i]
            tau_coeffs_Ci = tau_coeffs_C[i]
            tau_coeffs_Ei = tau_coeffs_E[i]
            tau_coeffs_Fi = tau_coeffs_F[i]
            d3taus_dT3is = []
            for j in cmps:
                term2 = (tau_coeffs_Fi[j] + nT2inv05*tau_coeffs_Ci[j]
                         + T3inv*tau_coeffs_Bi[j] + T4inv3*tau_coeffs_Ei[j])
                
                term3 = dtaus_dTi[j]/tausi[j]
                
                term4 = (T3inv2*(tau_coeffs_Ci[j] - Tinv3*tau_coeffs_Bi[j]
                         - T2_12*tau_coeffs_Ei[j]))
                
                d3taus_dT3is.append((term3*(6.0*term2 + term3*term3) + term4)*tausi[j])
            
            d3taus_dT3s.append(d3taus_dT3is)
        return d3taus_dT3s

    def phis(self):
        try:
            return self._phis
        except AttributeError:
            pass
        cmps, xs, rs = self.cmps, self.xs, self.rs
        rsxs = [rs[i]*xs[i] for i in cmps]
        self._rsxs_sum_inv = rsxs_sum_inv = 1.0/sum(rsxs)
        # reuse the array
        for i in cmps:
            rsxs[i] *= rsxs_sum_inv
        self._phis = rsxs
        return rsxs
    
    def dphis_dxs(self):
        r'''
        
        if i != j:
            
        .. math::
            \frac{\partial \phi_i}{x_j} = \frac{-r_i r_j x_i}{(\sum_k r_k x_k)^2}
            
        else:
            
        .. math::
            \frac{\partial \phi_i}{x_j} = \frac{-r_i r_j x_i}{(\sum_k r_k x_k)^2} + \frac{r_i}{\sum_k r_k x_k}
            
        '''
        try:
            return self._dphis_dxs
        except AttributeError:
            pass
        N, cmps, xs, rs = self.N, self.cmps, self.xs, self.rs
        
        rsxs = list(self.phis())
        rsxs_sum_inv = self._rsxs_sum_inv
        rsxs_sum_inv_m = -rsxs_sum_inv
        
        for i in cmps:
            # reuse this array for memory savings
            rsxs[i] *= rsxs_sum_inv_m
        
        self._dphis_dxs = dphis_dxs = [[0.0]*N for _ in cmps]
        for j in cmps:
            for i in cmps:
                dphis_dxs[i][j] = rsxs[i]*rs[j]
            # There is no symmetry to exploit here
            dphis_dxs[j][j] += rs[j]*rsxs_sum_inv
                    
        return dphis_dxs

    def dphis_dxixjs(self):
        r'''
        
        if i != j:
            
        .. math::
            
        else:
            
        .. math::
            
        '''
#        try:
#            return self._dphis_dxixjs
#        except AttributeError:
#            pass
        N, cmps, xs, rs = self.N, self.cmps, self.xs, self.rs

        self.phis() # Ensure the sum is there
        rsxs_sum_inv = self._rsxs_sum_inv
        rsxs_sum_inv2 = rsxs_sum_inv*rsxs_sum_inv
        rsxs_sum_inv3 = rsxs_sum_inv2*rsxs_sum_inv
        
        rsxs_sum_inv_2 = rsxs_sum_inv + rsxs_sum_inv
        rsxs_sum_inv2_2 = rsxs_sum_inv2 + rsxs_sum_inv2
        rsxs_sum_inv3_2 = rsxs_sum_inv3 + rsxs_sum_inv3

        self._dphis_dxixjs = dphis_dxixjs = [[[0.0]*N for _ in cmps] for _ in cmps]
        
        
#        a, b, c = 0, 0, 0
        for k in cmps:
            dphis_dxixjsk = dphis_dxixjs[k]
            for j in cmps:
                dphis_dxixjskj = dphis_dxixjsk[j]
                for i in cmps:
#                    if i == j == k:
#                        v = rs[i]*rs[i]*(rs[i]*xs[i]*rsxs_sum_inv - 1.0)*rsxs_sum_inv2_2
                        
                        
#                    elif i != j and i != k and j == k:
#                        b += 1
#                        v = rs[k]*rs[j]*rs[i]*xs[i]*rsxs_sum_inv3_2
                        
                    if (i == k or i == j) and j != k:
#                        c += 1
                        v = rs[k]*rs[j]*(rs[i]*xs[i]*rsxs_sum_inv_2 - 1.0)*rsxs_sum_inv2
                    else:
#                    elif i != j and i != k:# and j != k:
#                        a += 1
                        v = rs[k]*rs[j]*rs[i]*xs[i]*rsxs_sum_inv3_2
                    dphis_dxixjskj[i] = v
#        print(a, b, c)
            dphis_dxixjs[k][k][k] -= rs[k]*rs[k]*rsxs_sum_inv2_2

        return dphis_dxixjs

    def thetas(self):
        try:
            return self._thetas
        except AttributeError:
            pass
        cmps, xs = self.cmps, self.xs
        rs, qs = self.rs, self.qs
        qsxs = [qs[i]*xs[i] for i in cmps]
        qsxs_sum_inv = 1.0/sum(qsxs)
        
        # reuse the array qsxs to store thetas
        for i in cmps:
            qsxs[i] *= qsxs_sum_inv
        self._thetas = qsxs
        return qsxs

    def thetaj_taus_jis(self):
        # sum1
        try:
            return self._thetaj_taus_jis
        except AttributeError:
            pass
        try:
            thetas = self._thetas
        except AttributeError:
            thetas = self.thetas()
            
        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()
        
        xs, cmps = self.xs, self.cmps
        self._thetaj_taus_jis = thetaj_taus_jis = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += thetas[j]*taus[j][i]
            thetaj_taus_jis.append(tot)
        return thetaj_taus_jis
    
    def thetaj_dtaus_dT_jis(self):
        # sum3 maybe?
        try:
            return self._thetaj_dtaus_dT_jis
        except AttributeError:
            pass
        try:
            thetas = self._thetas
        except AttributeError:
            thetas = self.thetas()
        try:
            dtaus_dT = self._dtaus_dT
        except AttributeError:
            dtaus_dT = self.dtaus_dT()
        
        xs, cmps = self.xs, self.cmps
        self._thetaj_dtaus_dT_jis = thetaj_dtaus_dT_jis = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += thetas[j]*dtaus_dT[j][i]
            thetaj_dtaus_dT_jis.append(tot)
        return thetaj_dtaus_dT_jis

    def thetaj_d2taus_dT2_jis(self):
        # sum3 maybe?
        try:
            return self._thetaj_d2taus_dT2_jis
        except AttributeError:
            pass
        try:
            thetas = self._thetas
        except AttributeError:
            thetas = self.thetas()
        try:
            d2taus_dT2 = self._d2taus_dT2
        except AttributeError:
            d2taus_dT2 = self.d2taus_dT2()
            
        xs, cmps = self.xs, self.cmps
        self._thetaj_d2taus_dT2_jis = thetaj_d2taus_dT2_jis = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += thetas[j]*d2taus_dT2[j][i]
            thetaj_d2taus_dT2_jis.append(tot)
        return thetaj_d2taus_dT2_jis

    def thetaj_d3taus_dT3_jis(self):
        try:
            return self._thetaj_d3taus_dT3_jis
        except AttributeError:
            pass
        try:
            thetas = self._thetas
        except AttributeError:
            thetas = self.thetas()
        try:
            d3taus_dT3 = self._d3taus_dT3
        except AttributeError:
            d3taus_dT3 = self.d3taus_dT3()
            
        xs, cmps = self.xs, self.cmps
        self._thetaj_d3taus_dT3_jis = thetaj_d3taus_dT3_jis = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += thetas[j]*d3taus_dT3[j][i]
            thetaj_d3taus_dT3_jis.append(tot)
        return thetaj_d3taus_dT3_jis
            
    def GE(self):
        '''
        .. math::
            \frac{G^E}{RT} = \sum_i x_i \ln\frac{\phi_i}{x_i} 
            + \frac{z}{2}\sum_i q_i x_i \ln\frac{\theta_i}{\phi_i} 
            - \sum_i q_i x_i \ln\left(\sum_j \theta_j \tau_{ji}   \right)
        '''
        try:
            return self._GE
        except AttributeError:
            pass
        T, cmps, xs = self.T, self.cmps, self.xs
        rs, qs = self.rs, self.qs
        taus = self.taus()
        
        phis = self.phis()
        thetas = self.thetas()
        thetaj_taus_jis = self.thetaj_taus_jis()
        
        gE = 0.0
        z_2 = 0.5*self.z
        for i in cmps:
            gE += xs[i]*log(phis[i]/xs[i])
            gE += z_2*qs[i]*xs[i]*log(thetas[i]/phis[i])
            gE -= qs[i]*xs[i]*log(thetaj_taus_jis[i])
            
        gE *= R*T
        self._GE = gE
        return gE
    
    def dGE_dT(self):
        r'''
        .. math::
            \frac{\partial G^E}{\partial T} = \frac{G^E}{T} - RT\left(\sum_i 
            \frac{q_i x_i(\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T}
            )}{\sum_j \theta_j \tau_{ji}}\right)
        '''
        try:
            return self._dGE_dT
        except AttributeError:
            pass
    
        T, cmps, xs = self.T, self.cmps, self.xs
        qs = self.qs
        thetaj_taus_jis = self.thetaj_taus_jis()
        thetaj_dtaus_dT_jis = self.thetaj_dtaus_dT_jis()
        
        dGE = self.GE()/T
        
        tot = 0.0
        for i in cmps:
            tot -= qs[i]*xs[i]*thetaj_dtaus_dT_jis[i]/thetaj_taus_jis[i]
        dGE += R*T*tot
        self._dGE_dT = dGE
        return dGE
    
    def d2GE_dT2(self):
        r'''
        
        .. math::
            \frac{\partial G^E}{\partial T^2} = -R\left[T\sum_i\left(
            \frac{q_i x_i(\sum_j \theta_j \frac{\partial^2 \tau_{ji}}{\partial T^2})}{\sum_j \theta_j \tau_{ji}} 
            - \frac{q_i x_i(\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T})^2}{(\sum_j \theta_j \tau_{ji})^2}
            \right) + 2\left(\sum_i \frac{q_i x_i(\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T} )}{\sum_j \theta_j \tau_{ji}}\right)
            \right]
        '''
        try:
            return self._d2GE_dT2
        except AttributeError:
            pass
        
        T, cmps, xs = self.T, self.cmps, self.xs
        qs = self.qs
        thetaj_taus_jis = self.thetaj_taus_jis()
        thetaj_dtaus_dT_jis = self.thetaj_dtaus_dT_jis()
        thetaj_d2taus_dT2_jis = self.thetaj_d2taus_dT2_jis()

        GE = self.GE()
        dGE_dT = self.dGE_dT()

        tot = 0.0
        for i in cmps:
            tot += qs[i]*xs[i]*thetaj_d2taus_dT2_jis[i]/thetaj_taus_jis[i]
            tot -= qs[i]*xs[i]*thetaj_dtaus_dT_jis[i]**2/thetaj_taus_jis[i]**2
        d2GE_dT2 = T*tot - 2.0/(R*T)*(dGE_dT - GE/T)
        d2GE_dT2 *= -R
        self._d2GE_dT2 = d2GE_dT2
        return d2GE_dT2
        
    def d3GE_dT3(self):
        r'''
        .. math::
            \frac{\partial^3 G^E}{\partial T^3} = -R\left[T\sum_i\left(
            \frac{q_i x_i(\sum_j \theta_j \frac{\partial^3 \tau_{ji}}{\partial T^3})}{(\sum_j \theta_j \tau_{ji})}
            - \frac{3q_i x_i(\sum_j \theta_j \frac{\partial^2 \tau_{ji}}{\partial T^2}) (\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T})}{(\sum_j \theta_j \tau_{ji})^2}
            + \frac{2q_i x_i(\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T})^3}{(\sum_j \theta_j \tau_{ji})^3}
            \right) + \sum_i \left(\frac{3q_i x_i(\sum_j x_j \frac{\partial^2 \tau_{ji}}{\partial T^2} ) }{\sum_j \theta_j \tau_{ji}}
            - \frac{3q_ix_i (\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T})^2}{(\sum_j \theta_j \tau_{ji})^2}
            \right)\right]
        '''
        try:
            return self._d3GE_dT3
        except AttributeError:
            pass
        
        T, cmps, xs = self.T, self.cmps, self.xs
        qs = self.qs
        thetaj_taus_jis = self.thetaj_taus_jis()
        thetaj_dtaus_dT_jis = self.thetaj_dtaus_dT_jis()
        thetaj_d2taus_dT2_jis = self.thetaj_d2taus_dT2_jis()
        thetaj_d3taus_dT3_jis = self.thetaj_d3taus_dT3_jis()
        
        Ttot, tot = 0.0, 0.0
        
        for i in cmps:
            Ttot += qs[i]*xs[i]*thetaj_d3taus_dT3_jis[i]/thetaj_taus_jis[i]
            Ttot -= 3.0*qs[i]*xs[i]*thetaj_dtaus_dT_jis[i]*thetaj_d2taus_dT2_jis[i]/thetaj_taus_jis[i]**2
            Ttot += 2.0*qs[i]*xs[i]*thetaj_dtaus_dT_jis[i]**3/thetaj_taus_jis[i]**3
            
            tot += 3.0*qs[i]*xs[i]*thetaj_d2taus_dT2_jis[i]/thetaj_taus_jis[i]
            tot -= 3.0*qs[i]*xs[i]*thetaj_dtaus_dT_jis[i]**2/thetaj_taus_jis[i]**2
            
        self._d3GE_dT3 = d3GE_dT3 = -R*(T*Ttot + tot)
        return d3GE_dT3
        
    def dGE_dxs(self):
        T, xs, cmps = self.T, self.xs, self.cmps
        taus = self.taus()
        
        dGE_dxs = []
        
        for k in cmps:
            # k is what is being differentiated
            tot = 0
            for i in cmps:
                pass
            dGE_dxs.append(tot)
        return dGE_dxs

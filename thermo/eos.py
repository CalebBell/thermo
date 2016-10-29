# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from cmath import atanh as catanh
from scipy.constants import R
from scipy.optimize import fsolve
from numpy import roots
from thermo.utils import log, exp, sqrt, Cp_minus_Cv, isothermal_compressibility, phase_identification_parameter, phase_identification_parameter_phase
from thermo.utils import _isobaric_expansion as isobaric_expansion 



class CUBIC_EOS(object):
    
    def all_V(self, T, P, b, a_alpha):
        return [-(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)/(3*(sqrt(-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)) - (sqrt(-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)/3 - (P*b - R*T)/(3*P),
                -(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)/(3*(-1/2 - sqrt(3)*1j/2)*(sqrt(-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)) - (-1/2 - sqrt(3)*1j/2)*(sqrt(-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)/3 - (P*b - R*T)/(3*P),
                -(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)/(3*(-1/2 + sqrt(3)*1j/2)*(sqrt(-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)) - (-1/2 + sqrt(3)*1j/2)*(sqrt(-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)/3 - (P*b - R*T)/(3*P)]
    
    def first_derivatives(self, T, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2):
        dP_dT = R/(V - b) - da_alpha_dT/(V*(V + b) + b*(V - b))
        dP_dV = -R*T/(V - b)**2 - (-2*V - 2*b)*a_alpha/(V*(V + b) + b*(V - b))**2
        dV_dT = -dP_dT/dP_dV
        dV_dP = -dV_dT/dP_dT # or same as dP_dV
        dT_dV = 1./dV_dT
        dT_dP = 1./dP_dT
        return [dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP]

    def second_derivatives(self, T, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2):
        d2P_dT2 = -d2a_alpha_dT2/(V*(V + b) + b*(V - b)) # 0?
        d2P_dV2 = 2*R*T/(V - b)**3 - (-4*V - 4*b)*(-2*V - 2*b)*a_alpha/(V*(V + b) + b*(V - b))**3 + 2*a_alpha/(V*(V + b) + b*(V - b))**2 # 0?
        d2V_dT2 = (R/(V - b)**2 + (-2*V - 2*b)*da_alpha_dT/(V*(V + b) + b*(V - b))**2)*(-R/(V - b) + da_alpha_dT/(V*(V + b) + b*(V - b)))/(-R*T/(V - b)**2 - (-2*V - 2*b)*a_alpha/(V*(V + b) + b*(V - b))**2)**2 + d2a_alpha_dT2/((V*(V + b) + b*(V - b))*(-R*T/(V - b)**2 - (-2*V - 2*b)*a_alpha/(V*(V + b) + b*(V - b))**2))
        d2V_dP2 = 0
        d2T_dV2 = 1./d2V_dT2 # For sure
        d2T_dP2 = 0
        return [d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2]
        
    def second_derivatives_mixed(self, T, V, b, a_alpha, da_alpha_dT):
        d2V_dPdT = 0 # For sure
        d2P_dTdV = -R/(V - b)**2 - (-2*V - 2*b)*da_alpha_dT/(V*(V + b) + b*(V - b))**2
        d2T_dPdV = 0
        return [d2V_dPdT, d2P_dTdV, d2T_dPdV]

    def set_from_PT(self, T, P, Vs, b, a_alpha, da_alpha_dT, d2a_alpha_dT2):
        i_roots = len([True for i in Vs if abs(i.imag) > 1E-9]) # Determine the number of imag roots
        if i_roots == 2 : # Single phase, auto select phase
            V = [i for i in Vs if i.imag == 0][0]
            self.phase = self.set_derivs_from_phase(T, P, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2)
            if self.phase == 'l':
                self.Vl = V
            else:
                self.Vg = V
        elif i_roots == 1: # Two phase, have to deal with both
            Vs = [i for i in Vs if i.imag == 0]
            self.Vl, self.Vg = min(Vs), max(Vs)
            [self.set_derivs_from_phase(T, P, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2) for V in [self.Vl, self.Vg]]
        else:
            raise Exception('Three roots not possible')

    def set_derivs_from_phase(self, T, P, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2):
        [dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP] = self.first_derivatives(T, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2)
        [d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2] = self.second_derivatives(T, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2)
        [d2V_dPdT, d2P_dTdV, d2T_dPdV] = self.second_derivatives_mixed(T, V, b, a_alpha, da_alpha_dT)
        
        beta = isobaric_expansion(V, dV_dT)
        kappa = isothermal_compressibility(V, dV_dP)
        Cp_m_Cv = Cp_minus_Cv(T, d2P_dT2, dP_dV)

        
        PIP = phase_identification_parameter(V, dP_dT, dP_dV, d2P_dV2, d2P_dTdV)
        phase = phase_identification_parameter_phase(PIP)
        
        H_dep = P*V - R*T + sqrt(2)*catanh((V + b)*sqrt(2)/b/2).real * (da_alpha_dT*T-a_alpha)/b/2
        S_dep = R*log(P*V/(R*T)) + (da_alpha_dT*sqrt(2)*catanh((V + b)*sqrt(2)/b/2).real - 2*R*b*(log(V) - log(V - b)))/b/2
        V_dep = (V - R*T/P)        
        U_dep = H_dep - P*V_dep
        G_dep = H_dep - T*S_dep
        A_dep = U_dep - T*S_dep
        fugacity = P*exp(G_dep/(R*T))
        phi = fugacity/P
        
        


        if phase == 'l':
            self.beta_l, self.kappa_l, self.PIP_l, self.Cp_minus_Cv_l = beta, kappa, PIP, Cp_m_Cv
            self.dP_dT_l, self.dP_dV_l, self.dV_dT_l, self.dV_dP_l, self.dT_dV_l, self.dT_dP_l = dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP
            self.d2P_dT2_l, self.d2P_dV2_l, self.d2V_dT2_l, self.d2V_dP2_l, self.d2T_dV2_l, self.d2T_dP2_l = d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2
            self.d2V_dPdT_l, self.d2P_dTdV_l, self.d2T_dPdV_l = d2V_dPdT, d2P_dTdV, d2T_dPdV
            self.H_dep_l, self.S_dep_l = H_dep, S_dep
            
        else:
            self.beta_g, self.kappa_g, self.PIP_g, self.Cp_minus_Cv_g = beta, kappa, PIP, Cp_m_Cv
            self.dP_dT_g, self.dP_dV_g, self.dV_dT_g, self.dV_dP_g, self.dT_dV_g, self.dT_dP_g = dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP
            self.d2P_dT2_g, self.d2P_dV2_g, self.d2V_dT2_g, self.d2V_dP2_g, self.d2T_dV2_g, self.d2T_dP2_g = d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2
            self.d2V_dPdT_g, self.d2P_dTdV_g, self.d2T_dPdV_g = d2V_dPdT, d2P_dTdV, d2T_dPdV
            self.H_dep_g, self.S_dep_g = H_dep, S_dep
        return phase            

        
class PR(CUBIC_EOS):
    # constant part of `a`, 
    # X = (-1 + (6*sqrt(2)+8)**Rational(1,3) - (6*sqrt(2)-8)**Rational(1,3))/3
    # (8*(5*X+1)/(49-37*X)).evalf(40)
    c1 = 0.4572355289213821893834601962251837888504
    
    # Constant part of `b`, (X/(X+3)).evalf(40)
    c2 = 0.0777960739038884559718447100373331839711
    
    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R**2*Tc**2/Pc
        self.b = self.c2*R*Tc/Pc
        self.kappa = 0.37464+ 1.54226*self.omega - 0.26992*self.omega**2
        
        if T and P:
            self.a_alpha = self.a*(1 + self.kappa*(1-(T/self.Tc)**0.5))**2
            self.da_alpha_dT = -self.a*self.kappa*sqrt(T/self.Tc)*(self.kappa*(-sqrt(T/self.Tc) + 1.) + 1.)/T
            self.d2a_alpha_dT2 = self.a*self.kappa*(self.kappa/self.Tc - sqrt(T/self.Tc)*(self.kappa*(sqrt(T/self.Tc) - 1.) - 1.)/T)/(2.*T)
        
            Vs = self.all_V(T, P, self.b, self.a_alpha)
            self.set_from_PT(T, P, Vs, self.b, self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2)

        elif T and V:
            # Copy and pasted
            self.a_alpha = self.a*(1 + self.kappa*(1-(T/self.Tc)**0.5))**2
            self.da_alpha_dT = -self.a*self.kappa*sqrt(T/self.Tc)*(self.kappa*(-sqrt(T/self.Tc) + 1.) + 1.)/T
            self.d2a_alpha_dT2 = self.a*self.kappa*(self.kappa/self.Tc - sqrt(T/self.Tc)*(self.kappa*(sqrt(T/self.Tc) - 1.) - 1.)/T)/(2.*T)

            self.P = P = R*T/(V-self.b) - self.a_alpha/(V*(V+self.b)+self.b*(V-self.b))
            # Only calculate for the specified V
            self.set_from_PT(T, P, [V, 1j, 1j], self.b, self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2)


#a = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
#print(a.d2V_dPdT_l, a.PIP_l, a.Vl, a.Cp_minus_Cv_l)
#
#b = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., V=0.00013022208100139953)
#print(b.d2V_dPdT_l, b.PIP_l, b.Vl, b.P)

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

__all__ = ['CUBIC_EOS', 'PR']
from cmath import atanh as catanh
from scipy.constants import R
from scipy.optimize import fsolve
from numpy import roots
from thermo.utils import log, exp, sqrt, Cp_minus_Cv, isothermal_compressibility, phase_identification_parameter, phase_identification_parameter_phase
from thermo.utils import _isobaric_expansion as isobaric_expansion 




class CUBIC_EOS(object):
    
    def all_V(self, T, P, b, a_alpha, quick=True):
        if quick:
            x0 = 1./P
            x1 = R*T
            x2 = P*b - x1
            x3 = x0*x2
            x4 = x2*x2*x2
            x5 = b*x1
            x6 = a_alpha
            x7 = -x6
            x8 = b*b
            x9 = P*x8
            x10 = b*(x5 + x7 + x9)
            x11 = 1./(P*P)
            x12 = 3.*x6
            x13 = 6.*x5
            x14 = 9.*x9
            x15 = x0*x2*x2
            x16 = (13.5*x0*x10 + 4.5*x11*x2*(2.*x5 + x7 + 3.*x9) + sqrt(x11*(-4.*x0*(-x12 + x13 + x14 + x15)**3 + (27.*x10 + 2.*x11*x4 - x3*(-27.*P*x8 - 18.*R*T*b + 9.*x6))**2))/2. + x4/P**3)**(1./3.)
            x17 = x12 - x13 - x14 - x15
            x18 = 1./x16
            x19 = -2.*x3
            x20 = 1.7320508075688772j
            x21 = x20 + 1.
            x22 = 4.*x0*x17*x18
            x23 = -x20 + 1.
            return [x0*x17*x18/3. - x16/3. - x3/3.,
                    x16*x21/6. + x19/6. - x22/(6.*x21),
                    x16*x23/6. + x19/6. - x22/(6.*x23)]
        else:
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
    
    def derivatives_and_departures(self, T, P, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2):
        x0 = V - b
        x1 = V + b
        x2 = V*x1 + b*x0
        x3 = 1./x2
        x4 = R/x0 - da_alpha_dT*x3
        x5 = R*T
        x6 = 1./(x0*x0)
        x7 = x5*x6
        x8 = 2.*x1
        x9 = 1./(x2*x2)
        x10 = a_alpha*x9
        x11 = x10*x8
        x12 = d2a_alpha_dT2*x3
        x13 = 1./(-x11 + x7)
        x14 = R*x6
        x15 = da_alpha_dT*x8*x9
        x16 = P*V
        x17 = sqrt(2)
        x18 = 1./b
        x19 = x17*x18/2.
        x20 = catanh(x1*x19).real

        [dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2V_dT2, d2P_dTdV, H_dep, S_dep] = [x4,
         x11 - x7,
         -x12,
         -8.*a_alpha*x1*x1/(x2*x2*x2) + 2.*x10 + 2.*x5/(x0*x0*x0),
         -x13*(x12 + x13*x4*(x14 - x15)),
         -x14 + x15,
         x16 + x19*x20*(T*da_alpha_dT - a_alpha) - x5,
         R*log(x16/(R*T)) - x18*(2*R*b*(log(V) - log(x0)) - da_alpha_dT*x17*x20)/2.]
        

        dV_dT = -dP_dT/dP_dV
        dV_dP = -dV_dT/dP_dT # or same as dP_dV
        dT_dV = 1./dV_dT
        dT_dP = 1./dP_dT

        d2V_dP2 = 0
        d2T_dV2 = 1./d2V_dT2 # For sure
        d2T_dP2 = 0

        d2V_dPdT = 0 # For sure
        d2T_dPdV = 0
        
        return ([dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP], 
                [d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2],
                [d2V_dPdT, d2P_dTdV, d2T_dPdV],
                [H_dep, S_dep])

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

    def set_derivs_from_phase(self, T, P, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2, quick=True):
        if quick:
            ([dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP], 
                [d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2],
                [d2V_dPdT, d2P_dTdV, d2T_dPdV],
                [H_dep, S_dep]) = self.derivatives_and_departures(T, P, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2)
        else:
            [dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP] = self.first_derivatives(T, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2)
            [d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2] = self.second_derivatives(T, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2)
            [d2V_dPdT, d2P_dTdV, d2T_dPdV] = self.second_derivatives_mixed(T, V, b, a_alpha, da_alpha_dT)
        
            H_dep = P*V - R*T + sqrt(2)*catanh((V + b)*sqrt(2)/b/2).real * (da_alpha_dT*T-a_alpha)/b/2
            S_dep = R*log(P*V/(R*T)) + (da_alpha_dT*sqrt(2)*catanh((V + b)*sqrt(2)/b/2).real - 2*R*b*(log(V) - log(V - b)))/b/2
        
        beta = isobaric_expansion(V, dV_dT)
        kappa = isothermal_compressibility(V, dV_dP)
        Cp_m_Cv = Cp_minus_Cv(T, d2P_dT2, dP_dV)
        
        PIP = phase_identification_parameter(V, dP_dT, dP_dV, d2P_dV2, d2P_dTdV)
        phase = phase_identification_parameter_phase(PIP)
        
        V_dep = (V - R*T/P)        
        U_dep = H_dep - P*V_dep
        G_dep = H_dep - T*S_dep
        A_dep = U_dep - T*S_dep
        fugacity = P*exp(G_dep/(R*T))
        phi = fugacity/P
        
        if phase == 'l':
            self.beta_l, self.kappa_l = beta, kappa
            self.PIP_l, self.Cp_minus_Cv_l = PIP, Cp_m_Cv
            
            self.dP_dT_l, self.dP_dV_l, self.dV_dT_l = dP_dT, dP_dV, dV_dT
            self.dV_dP_l, self.dT_dV_l, self.dT_dP_l = dV_dP, dT_dV, dT_dP
            
            self.d2P_dT2_l, self.d2P_dV2_l, self.d2V_dT2_l = d2P_dT2, d2P_dV2, d2V_dT2
            self.d2V_dP2_l, self.d2T_dV2_l, self.d2T_dP2_l = d2V_dP2, d2T_dV2, d2T_dP2
            
            self.d2V_dPdT_l, self.d2P_dTdV_l, self.d2T_dPdV_l = d2V_dPdT, d2P_dTdV, d2T_dPdV
            
            self.H_dep_l, self.S_dep_l, self.V_dep_l = H_dep, S_dep, V_dep, 
            self.U_dep_l, self.G_dep_l, self.A_dep_l = U_dep, G_dep, A_dep, 
            self.fugacity_l, self.phi_l = fugacity, phi
        else:
            self.beta_g, self.kappa_g = beta, kappa
            self.PIP_g, self.Cp_minus_Cv_g = PIP, Cp_m_Cv
            
            self.dP_dT_g, self.dP_dV_g, self.dV_dT_g = dP_dT, dP_dV, dV_dT
            self.dV_dP_g, self.dT_dV_g, self.dT_dP_g = dV_dP, dT_dV, dT_dP
            
            self.d2P_dT2_g, self.d2P_dV2_g, self.d2V_dT2_g = d2P_dT2, d2P_dV2, d2V_dT2
            self.d2V_dP2_g, self.d2T_dV2_g, self.d2T_dP2_g = d2V_dP2, d2T_dV2, d2T_dP2
            
            self.d2V_dPdT_g, self.d2P_dTdV_g, self.d2T_dPdV_g = d2V_dPdT, d2P_dTdV, d2T_dPdV
            
            self.H_dep_g, self.S_dep_g, self.V_dep_g = H_dep, S_dep, V_dep, 
            self.U_dep_g, self.G_dep_g, self.A_dep_g = U_dep, G_dep, A_dep, 
            self.fugacity_g, self.phi_g = fugacity, phi
        return phase            

        
class PR(CUBIC_EOS):
    # constant part of `a`, 
    # X = (-1 + (6*sqrt(2)+8)**Rational(1,3) - (6*sqrt(2)-8)**Rational(1,3))/3
    # (8*(5*X+1)/(49-37*X)).evalf(40)
    c1 = 0.4572355289213821893834601962251837888504
    
    # Constant part of `b`, (X/(X+3)).evalf(40)
    c2 = 0.0777960739038884559718447100373331839711
    
    def set_a_alpha(self, T):
        self.a_alpha = self.a*(1 + self.kappa*(1-(T/self.Tc)**0.5))**2
    
    def solve_T(self, P, V, quick=True):
        Tc, a, b, kappa = self.Tc, self.a, self.b, self.kappa
        if quick:
            x0 = V*V
            x1 = R*Tc
            x2 = x0*x1
            x3 = kappa*kappa
            x4 = a*x3
            x5 = b*x4
            x6 = 2.*V*b
            x7 = x1*x6
            x8 = b*b
            x9 = x1*x8
            x10 = V*x4
            x11 = (-x10 + x2 + x5 + x7 - x9)**2
            x12 = x0*x0
            x13 = R*R
            x14 = Tc*Tc
            x15 = x13*x14
            x16 = x8*x8
            x17 = a*a
            x18 = x3*x3
            x19 = x17*x18
            x20 = x0*V
            x21 = 2.*R*Tc*a*x3
            x22 = x8*b
            x23 = 4.*V*x22
            x24 = 4.*b*x20
            x25 = a*x1
            x26 = x25*x8
            x27 = x26*x3
            x28 = x0*x25
            x29 = x28*x3
            x30 = 2.*x8
            x31 = 6.*V*x27 - 2.*b*x29 + x0*x13*x14*x30 + x0*x19 + x12*x15 + x15*x16 - x15*x23 + x15*x24 - x19*x6 + x19*x8 - x20*x21 - x21*x22
            x32 = V - b
            x33 = 2.*(R*Tc*a*kappa)
            x34 = P*x2
            x35 = P*x5
            x36 = x25*x3
            x37 = P*x10
            x38 = P*R*Tc
            x39 = V*x17
            x40 = 2.*kappa*x3
            x41 = b*x17
            x42 = P*a*x3
            return -Tc*(2.*a*kappa*x11*sqrt(x32**3*(x0 + x6 - x8)*(P*x7 - P*x9 + x25 + x33 + x34 + x35 + x36 - x37))*(kappa + 1.) - x31*x32*((4.*V)*(R*Tc*a*b*kappa) + x0*x33 - x0*x35 + x12*x38 + x16*x38 + x18*x39 - x18*x41 - x20*x42 - x22*x42 - x23*x38 + x24*x38 + x25*x6 - x26 - x27 + x28 + x29 + x3*x39 - x3*x41 + x30*x34 - x33*x8 + x36*x6 + 3*x37*x8 + x39*x40 - x40*x41))/(x11*x31)
        else:
            return Tc*(-2*a*kappa*sqrt((V - b)**3*(V**2 + 2*V*b - b**2)*(P*R*Tc*V**2 + 2*P*R*Tc*V*b - P*R*Tc*b**2 - P*V*a*kappa**2 + P*a*b*kappa**2 + R*Tc*a*kappa**2 + 2*R*Tc*a*kappa + R*Tc*a))*(kappa + 1)*(R*Tc*V**2 + 2*R*Tc*V*b - R*Tc*b**2 - V*a*kappa**2 + a*b*kappa**2)**2 + (V - b)*(R**2*Tc**2*V**4 + 4*R**2*Tc**2*V**3*b + 2*R**2*Tc**2*V**2*b**2 - 4*R**2*Tc**2*V*b**3 + R**2*Tc**2*b**4 - 2*R*Tc*V**3*a*kappa**2 - 2*R*Tc*V**2*a*b*kappa**2 + 6*R*Tc*V*a*b**2*kappa**2 - 2*R*Tc*a*b**3*kappa**2 + V**2*a**2*kappa**4 - 2*V*a**2*b*kappa**4 + a**2*b**2*kappa**4)*(P*R*Tc*V**4 + 4*P*R*Tc*V**3*b + 2*P*R*Tc*V**2*b**2 - 4*P*R*Tc*V*b**3 + P*R*Tc*b**4 - P*V**3*a*kappa**2 - P*V**2*a*b*kappa**2 + 3*P*V*a*b**2*kappa**2 - P*a*b**3*kappa**2 + R*Tc*V**2*a*kappa**2 + 2*R*Tc*V**2*a*kappa + R*Tc*V**2*a + 2*R*Tc*V*a*b*kappa**2 + 4*R*Tc*V*a*b*kappa + 2*R*Tc*V*a*b - R*Tc*a*b**2*kappa**2 - 2*R*Tc*a*b**2*kappa - R*Tc*a*b**2 + V*a**2*kappa**4 + 2*V*a**2*kappa**3 + V*a**2*kappa**2 - a**2*b*kappa**4 - 2*a**2*b*kappa**3 - a**2*b*kappa**2))/((R*Tc*V**2 + 2*R*Tc*V*b - R*Tc*b**2 - V*a*kappa**2 + a*b*kappa**2)**2*(R**2*Tc**2*V**4 + 4*R**2*Tc**2*V**3*b + 2*R**2*Tc**2*V**2*b**2 - 4*R**2*Tc**2*V*b**3 + R**2*Tc**2*b**4 - 2*R*Tc*V**3*a*kappa**2 - 2*R*Tc*V**2*a*b*kappa**2 + 6*R*Tc*V*a*b**2*kappa**2 - 2*R*Tc*a*b**3*kappa**2 + V**2*a**2*kappa**4 - 2*V*a**2*b*kappa**4 + a**2*b**2*kappa**4))

    
    
    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.kappa = 0.37464+ 1.54226*omega - 0.26992*omega*omega
        
        if not ([T and P] or [T and V] or [P and V]):
            raise Exception('Either T and P, or T and V, or P and V are required')
        
        if V:
            if P:
                a, b, kappa = self.a, self.b, self.kappa
                self.T = T = self.solve_T(P, V)
                self.set_a_alpha(T)
            else:
                self.set_a_alpha(T)
                self.P = P = R*T/(V-self.b) - self.a_alpha/(V*(V+self.b)+self.b*(V-self.b))
            Vs = [V, 1j, 1j]
        else:
            self.set_a_alpha(T)
            Vs = self.all_V(T, P, self.b, self.a_alpha)
        
        self.da_alpha_dT = -self.a*self.kappa*sqrt(T/self.Tc)*(self.kappa*(-sqrt(T/self.Tc) + 1.) + 1.)/T
        self.d2a_alpha_dT2 = self.a*self.kappa*(self.kappa/self.Tc - sqrt(T/self.Tc)*(self.kappa*(sqrt(T/self.Tc) - 1.) - 1.)/T)/(2.*T)

        self.set_from_PT(T, P, Vs, self.b, self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2)


a = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
print(a.d2V_dPdT_l, a.PIP_l, a.Vl)

b = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., V=0.00013022208100139953)
print(b.d2V_dPdT_l, b.PIP_l, b.Vl, b.P)

c = PR(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00013022208100139953, P=1E6)
print(c.d2V_dPdT_l, c.PIP_l, c.Vl, c.T)

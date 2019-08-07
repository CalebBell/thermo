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
__all__ = ['GibbbsExcessLiquid', 'Phase', 'EOSLiquid', 'EOSGas', 'IdealGas']

from fluids.constants import R, R_inv
from fluids.numerics import horner, horner_and_der
from thermo.utils import (log, exp, Cp_minus_Cv, phase_identification_parameter,
                          isothermal_compressibility, isobaric_expansion,
                          Joule_Thomson, dxs_to_dns)
from thermo.activity import IdealSolution


'''
All phase objects are immutable.

Goal is for each phase to be able to compute all of its thermodynamic properties.
This includes volume-based ones. Use settings to handle different; do not worry
about derivatives being calculated correctly.

Phases know nothing about bulk properties.
Phases know nothing about transport properties.

For enthalpy, need to support with ideal gas heat of formation as a separate
enthalpy calculation.
'''


class Phase(object):
    T_REF_IG = 298.15
    P_REF_IG = 101325.
    P_REF_IG_INV = 1.0/P_REF_IG
    
    T_MAX_FIXED = 10000.0
    T_MIN_FIXED = 1e-3
    
    P_MAX_FIXED = 1e9
    P_MIN_FIXED = 1e-3
    
    def fugacities(self):
        P = self.P
        zs = self.zs
        lnphis = self.lnphis()
        return [P*zs[i]*exp(lnphis[i]) for i in range(len(zs))]

    def phis(self):
        return [exp(i) for i in self.lnphis()]
    
    def log_zs(self):
        try:
            return self._log_zs
        except AttributeError:
            pass
        self._log_zs = [log(i) for i in self.zs]
        return self._log_zs

    def G(self):
        G = self.H() - self.T*self.S()
        return G
    
    def U(self):
        U = self.H() - self.P*self.V()
        return U
    
    def A(self):
        A = self.U() - self.T*self.S()
        return A

    def dH_dns(self):
        return dxs_to_dns(self.dH_dzs(), self.zs)
    
    def dS_dns(self):
        return dxs_to_dns(self.dS_dzs(), self.zs)
    
    def dG_dT(self):
        return -self.T*self.dS_dT() - self.S() + self.dH_dT()
    
    def dG_dP(self):
        return -self.T*self.dS_dP() + self.dH_dP()
    
    def dU_dT(self):
        # Correct
        return -self.P*self.dV_dT() + self.dH_dT()
    
    def dU_dP(self):
        # Correct
        return -self.P*self.dV_dP() - self.V() + self.dH_dP()
    
    def dA_dT(self):
        return -self.T*self.dS_dT() - self.S() + self.dU_dT()
    
    def dA_dP(self):
        return -self.T*self.dS_dP() + self.dU_dP()
        
    def G_dep(self):
        G_dep = self.H_dep() - self.T*self.S_dep()
        return G_dep
    
    def V_dep(self):
        # from ideal gas behavior
        V_dep = self.V() - R*self.T/self.P
        return V_dep
    
    def U_dep(self):
        return self.H_dep() - self.P*self.V_dep()
    
    def A_dep(self):
        return self.U_dep() - self.T*self.S_dep()


    def H_reactive(self):
        H = self.H
        for zi, Hf in zip(self.zs, self.Hfs):
            H += zi*Hf
        return H

    def S_reactive(self):
        S = self.S
        for zi, Sf in zip(self.zs, self.Sfs):
            S += zi*Sf
        return S
    
    def G_reactive(self):
        G = self.H_reactive ()- self.T*self.S_reactive()
        return G
    
    def U_reactive(self):
        U = self.H_reactive() - self.P*self.V()
        return U
    
    def A_reactive(self):
        A = self.U_reactive() - self.T*self.S_reactive()
        return A
    
    def Cv(self):
        # checks out
        Cp_m_Cv = Cp_minus_Cv(self.T, self.dP_dT(), self.dP_dV())
        Cp = self.Cp()
        return Cp - Cp_m_Cv
    
    def Cp_Cv_ratio(self):
        return self.Cp()/self.Cv()
        
    def rho(self):
        return 1.0/self.V()
    
    def dT_dP(self):
        return 1.0/self.dP_dT()
    
    def dV_dT(self):
        return -self.dP_dT()/self.dP_dV()
    
    def dV_dP(self):
        return -self.dV_dT()*self.dT_dP()
    
    def dT_dV(self):
        return 1./self.dV_dT()
    
    def d2V_dP2(self):
        inverse_dP_dV = 1.0/self.dP_dV()
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV*inverse_dP_dV
        return -self.d2P_dV2()*inverse_dP_dV3

    def d2T_dP2(self):
        dT_dP = self.dT_dP()
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP
        return -self.d2P_dT2()*inverse_dP_dT3
    
    def d2T_dV2(self):
        # Wrong
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dTdV = self.d2P_dTdV()
        d2P_dT2 = self.d2P_dT2()
        dT_dP = self.dT_dP()
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP
        
        return (-(self.d2P_dV2()*dP_dT - dP_dV*d2P_dTdV)*inverse_dP_dT2
                   +(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3*dP_dV)
        
    def d2V_dT2(self):
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dTdV = self.d2P_dTdV()
        d2P_dT2 = self.d2P_dT2()
        d2P_dV2 = self.d2P_dV2()

        inverse_dP_dV = 1.0/dP_dV
        inverse_dP_dV2 = inverse_dP_dV*inverse_dP_dV
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV2
        
        return  (-(d2P_dT2*dP_dV - dP_dT*d2P_dTdV)*inverse_dP_dV2
                   +(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*inverse_dP_dV3*dP_dT)

    def d2V_dPdT(self):
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dTdV = self.d2P_dTdV()
        d2P_dV2 = self.d2P_dV2()
        
        inverse_dP_dV = 1.0/dP_dV
        inverse_dP_dV2 = inverse_dP_dV*inverse_dP_dV
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV2
        
        return -(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*inverse_dP_dV3

    def d2T_dPdV(self):
        dT_dP = self.dT_dP()
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP
        
        d2P_dTdV = self.d2P_dTdV()
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dT2 = self.d2P_dT2()
        return -(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3
    
    def PIP(self):
        return phase_identification_parameter(self.V(), self.dP_dT(), self.dP_dV(), 
                                              self.d2P_dV2(), self.d2P_dTdV())
        
    def kappa(self):
        return isothermal_compressibility(self.V(), self.dV_dP())

    def beta(self):
        return isobaric_expansion(self.V(), self.dV_dT())
    
    def Joule_Thomson(self):
        return Joule_Thomson(self.T, self.V(), self.Cp(), dV_dT=self.dV_dT(), beta=self.beta())
        
    def dZ_dT(self):
        T_inv = 1.0/self.T
        return self.P*R_inv*T_inv*(self.dV_dT() - self.V()*T_inv)

    def dZ_dP(self):
        return 1.0/(self.T*R)*(self.V() + self.P*self.dV_dP())

    ### Derivatives in the molar density basis

    def d2V_dTdP(self):
        return self.d2V_dPdT()

    def d2P_dVdT(self):
        return self.d2P_dTdV()

    def d2T_dVdP(self):
        return self.d2T_dPdV()

    def dP_drho(self):
        V = self.V()
        return -V*V*self.dP_dV()

    def drho_dP(self):
        V = self.V()
        return -self.dV_dP()/(V*V)

    def d2P_drho2(self):
        V = self.V()
        return -V**2*(-V**2*self.d2P_dV2() - 2*V*self.dP_dV())

    def d2rho_dP2(self):
        V = self.V()
        return -self.d2V_dP2()/V**2 + 2*self.dV_dP()**2/V**3

    def dT_drho(self):
        V = self.V()
        return -V*V*self.dT_dV()

    def d2T_drho2(self):
        V = self.V()
        return -V**2*(-V**2*self.d2T_dV2() - 2*V*self.dT_dV())

    def drho_dT(self):
        V = self.V()
        return -self.dV_dT()/(V*V)

    def d2rho_dT2(self):
        d2V_dT2 = self.d2V_dT2()
        V = self.V()
        dV_dT = self.dV_dT()
        return -d2V_dT2/V**2 + 2*dV_dT**2/V**3

    def d2P_dTdrho(self):
        V = self.V()
        d2P_dTdV = self.d2P_dTdV()
        return -(V*V)*d2P_dTdV

    def d2T_dPdrho(self):
        V = self.V()
        d2T_dPdV = self.d2T_dPdV()
        return -(V*V)*d2T_dPdV

    def d2rho_dPdT(self):
        d2V_dPdT = self.d2V_dPdT()
        dV_dT = self.dV_dT()
        dV_dP = self.dV_dP()
        V = self.V()
        return -d2V_dPdT/V**2 + 2*dV_dT*dV_dP/V**3


class IdealGas(Phase):
    '''DO NOT DELETE - EOS CLASS IS TOO SLOW!
    This will be important for fitting.
    
    '''
    def __init__(self, HeatCapacityGases=None, Hfs=None, Gfs=None):
        self.HeatCapacityGases = HeatCapacityGases
        self.Hfs = Hfs
        self.Gfs = Gfs
        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            self.Sfs = [(Hfi - Gfi)/298.15 for Hfi, Gfi in zip(Hfs, Gfs)]
        else:
            self.Sfs = None
            
        if HeatCapacityGases is not None:
            self.N = len(HeatCapacityGases)
        
    def fugacities(self):
        P = self.P
        return [P*zi for zi in self.zs]
    
    def lnphis(self):
        return [0.0]*self.N
    
    def dlnphis_dT(self):
        return [0.0]*self.N

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        new.N = len(zs)
        
        new.HeatCapacityGases = self.HeatCapacityGases
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        return new


class EOSLiquid(Phase):
    # DO NOT MAKE EDITS TO THIS CLASS!!!
    def __init__(self, eos_class, eos_kwargs, HeatCapacityGases=None, Hfs=None,
                 Gfs=None, Sfs=None,
                 T=None, P=None, zs=None):
        self.eos_class = eos_class
        self.eos_kwargs = eos_kwargs

        self.HeatCapacityGases = HeatCapacityGases
        if HeatCapacityGases is not None:
            self.N = N = len(HeatCapacityGases)
            self.cmps = range(self.N)
        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs
        
        if T is not None and P is not None and zs is not None:
            self.T = T
            self.P = P
            self.zs = zs
            self.eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
            
        
    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        try:
            new.eos_mix = self.eos_mix.to_TP_zs_fast(T=T, P=P, zs=zs, only_l=True,
                                                     full_alphas=True) # optimize alphas?
                                                     # Be very careful doing this in the future - wasted
                                                     # 1 hour on this because the heat capacity calculation was wrong
        except AttributeError:
            new.eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
        
        new.eos_class = self.eos_class
        new.eos_kwargs = self.eos_kwargs
        
        new.HeatCapacityGases = self.HeatCapacityGases
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        
        try:
            new.N = self.N
            new.cmps = self.cmps
        except:
            pass

        return new
        
    def lnphis(self):
        try:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_l, self.zs)
        except AttributeError:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_g, self.zs)
        
        
    def dlnphis_dT(self):
        try:
            return self.eos_mix.dlnphis_dT('l')
        except:
            return self.eos_mix.dlnphis_dT('g')

    def dlnphis_dP(self):
        try:
            return self.eos_mix.dlnphis_dP('l')
        except:
            return self.eos_mix.dlnphis_dP('g')
    
    def H_dep(self):
        try:
            return self.eos_mix.H_dep_l
        except AttributeError:
            return self.eos_mix.H_dep_g

    def S_dep(self):
        try:
            return self.eos_mix.S_dep_l
        except AttributeError:
            return self.eos_mix.S_dep_g

    def Cp_dep(self):
        try:
            return self.eos_mix.Cp_dep_l
        except AttributeError:
            return self.eos_mix.Cp_dep_g        
        
    def V(self):
        try:
            return self.eos_mix.V_l
        except AttributeError:
            return self.eos_mix.V_g

    def Z(self):
        try:
            return self.eos_mix.Z_l
        except AttributeError:
            return self.eos_mix.Z_g
    
    
    def dP_dT(self):
        try:
            return self.eos_mix.dP_dT_l
        except AttributeError:
            return self.eos_mix.dP_dT_g

    def dP_dV(self):
        try:
            return self.eos_mix.dP_dV_l
        except AttributeError:
            return self.eos_mix.dP_dV_g
    
    def d2P_dT2(self):
        try:
            return self.eos_mix.d2P_dT2_l
        except AttributeError:
            return self.eos_mix.d2P_dT2_g

    def d2P_dV2(self):
        try:
            return self.eos_mix.d2P_dV2_l
        except AttributeError:
            return self.eos_mix.d2P_dV2_g

    def d2P_dTdV(self):
        try:
            return self.eos_mix.d2P_dTdV_l
        except AttributeError:
            return self.eos_mix.d2P_dTdV_g
        
    # because of the ideal gas model, for some reason need to use the right ones
    # FOR THIS MODEL ONLY
    def d2T_dV2(self):
        try:
            return self.eos_mix.d2T_dV2_l
        except AttributeError:
            return self.eos_mix.d2T_dV2_g

    def d2V_dT2(self):
        try:
            return self.eos_mix.d2V_dT2_l
        except AttributeError:
            return self.eos_mix.d2V_dT2_g

        
    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        H = self.H_dep        
        for zi, Cp_int in zip(self.zs, self.Cp_integrals_pure):
            H += zi*Cp_int
        self._H = H
        return H

    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        Cp_integrals_over_T_pure = self.Cp_integrals_over_T_pure
        log_zs = self.log_zs()
        T, P, zs, cmps = self.T, self.P, self.zs, self.cmps
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)
        
        for i in cmps:
            S += zs[i]*Cp_integrals_over_T_pure[i]
        S += self.S_dep
        self._S = S
        return S
    
    def Cps_pure(self):
        try:
            return self._Cps
        except AttributeError:
            pass
        T = self.T
        self._Cps = [i.T_dependent_property(T) for i in self.HeatCapacityGases]
        return self._Cps
    
    def Cp_integrals_pure(self):
        try:
            return self._Cp_integrals_pure
        except AttributeError:
            pass
        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        self._Cp_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        return self._Cp_integrals_pure

    def Cp_integrals_over_T_pure(self):
        try:
            return self._Cp_integrals_over_T_pure
        except AttributeError:
            pass
        
        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        self._Cp_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        return self._Cp_integrals_over_T_pure
        

    def Cp(self):
        Cps_pure = self.Cps_pure
        Cp, zs = 0.0, self.zs
        for i in self.cmps:
            Cp += zs[i]*Cps_pure[i]
        return Cp + self.Cp_dep

    def dH_dT(self):
        return self.Cp

    def dH_dP(self):
        try:
            return self.eos_mix.dH_dep_dP_l
        except AttributeError:
            return self.eos_mix.dH_dep_dP_g

    def dH_dzs(self):
        try:
            return self._dH_dzs
        except AttributeError:
            pass
        eos_mix = self.eos_mix
        try:
            dH_dep_dzs = self.eos_mix.dH_dep_dzs(eos_mix.Z_l, eos_mix.zs)
        except AttributeError:
            dH_dep_dzs = self.eos_mix.dH_dep_dzs(eos_mix.Z_g, eos_mix.zs)
        Cp_integrals_pure = self.Cp_integrals_pure
        self._dH_dzs = [dH_dep_dzs[i] + Cp_integrals_pure[i] for i in self.cmps]
        return self._dH_dzs

    def dS_dT(self):
        HeatCapacityGases = self.HeatCapacityGases
        cmps = self.cmps
        T, zs = self.T, self.zs
        T_REF_IG = self.T_REF_IG
        P_REF_IG_INV = self.P_REF_IG_INV

        S = 0.0
        dS_pure_sum = 0.0
        for zi, obj in zip(zs, HeatCapacityGases):
            dS_pure_sum += zi*obj.T_dependent_property(T)
        S += dS_pure_sum/T
        try:
            S += self.eos_mix.dS_dep_dT_l
        except AttributeError:
            S += self.eos_mix.dS_dep_dT_g
        return S

    def dS_dP(self):
        dS = 0.0
        P = self.P
        dS -= R/P
        try:
            dS += self.eos_mix.dS_dep_dP_l
        except AttributeError:
            dS += self.eos_mix.dS_dep_dP_g
        return dS
            
    def dS_dzs(self):
        try:
            return self._dS_dzs
        except AttributeError:
            pass
        cmps, eos_mix = self.cmps, self.eos_mix
    
        log_zs = self.log_zs()
        integrals = self.Cp_integrals_over_T_pure

        try:
            dS_dep_dzs = self.eos_mix.dS_dep_dzs(eos_mix.Z_l, eos_mix.zs)
        except AttributeError:
            dS_dep_dzs = self.eos_mix.dS_dep_dzs(eos_mix.Z_g, eos_mix.zs)
        
        self._dS_dzs = [integrals[i] - R*(log_zs[i] + 1.0) + dS_dep_dzs[i] 
                        for i in cmps]
        return self._dS_dzs
 
class EOSGas(Phase):
    def __init__(self, eos_class, eos_kwargs, HeatCapacityGases=None, Hfs=None,
                 Gfs=None, Sfs=None,
                 T=None, P=None, zs=None):
        self.eos_class = eos_class
        self.eos_kwargs = eos_kwargs

        self.HeatCapacityGases = HeatCapacityGases
        if HeatCapacityGases is not None:
            self.N = N = len(HeatCapacityGases)
            self.cmps = range(self.N)
        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs
        
        if T is not None and P is not None and zs is not None:
            self.T = T
            self.P = P
            self.zs = zs
            self.eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
            
        
    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        try:
            new.eos_mix = self.eos_mix.to_TP_zs_fast(T=T, P=P, zs=zs, only_g=True,
                                                     full_alphas=True) # optimize alphas?
                                                     # Be very careful doing this in the future - wasted
                                                     # 1 hour on this because the heat capacity calculation was wrong
        except AttributeError:
            new.eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
        
        new.eos_class = self.eos_class
        new.eos_kwargs = self.eos_kwargs
        
        new.HeatCapacityGases = self.HeatCapacityGases
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        
        try:
            new.N = self.N
            new.cmps = self.cmps
        except:
            pass

        return new
        
    def lnphis(self):
        try:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_g, self.zs)
        except AttributeError:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_l, self.zs)
        
        
    def dlnphis_dT(self):
        try:
            return self.eos_mix.dlnphis_dT('g')
        except:
            return self.eos_mix.dlnphis_dT('l')

    def dlnphis_dP(self):
        try:
            return self.eos_mix.dlnphis_dP('g')
        except:
            return self.eos_mix.dlnphis_dP('l')
    
    def H_dep(self):
        try:
            return self.eos_mix.H_dep_g
        except AttributeError:
            return self.eos_mix.H_dep_l

    def S_dep(self):
        try:
            return self.eos_mix.S_dep_g
        except AttributeError:
            return self.eos_mix.S_dep_l

    def Cp_dep(self):
        try:
            return self.eos_mix.Cp_dep_g
        except AttributeError:
            return self.eos_mix.Cp_dep_l        
        
    def V(self):
        try:
            return self.eos_mix.V_g
        except AttributeError:
            return self.eos_mix.V_l

    def Z(self):
        try:
            return self.eos_mix.Z_g
        except AttributeError:
            return self.eos_mix.Z_l
    
    
    def dP_dT(self):
        try:
            return self.eos_mix.dP_dT_g
        except AttributeError:
            return self.eos_mix.dP_dT_l

    def dP_dV(self):
        try:
            return self.eos_mix.dP_dV_g
        except AttributeError:
            return self.eos_mix.dP_dV_l
    
    def d2P_dT2(self):
        try:
            return self.eos_mix.d2P_dT2_g
        except AttributeError:
            return self.eos_mix.d2P_dT2_l

    def d2P_dV2(self):
        try:
            return self.eos_mix.d2P_dV2_g
        except AttributeError:
            return self.eos_mix.d2P_dV2_l

    def d2P_dTdV(self):
        try:
            return self.eos_mix.d2P_dTdV_g
        except AttributeError:
            return self.eos_mix.d2P_dTdV_l
        
    # because of the ideal gas model, for some reason need to use the right ones
    # FOR THIS MODEL ONLY
    def d2T_dV2(self):
        try:
            return self.eos_mix.d2T_dV2_g
        except AttributeError:
            return self.eos_mix.d2T_dV2_l

    def d2V_dT2(self):
        try:
            return self.eos_mix.d2V_dT2_g
        except AttributeError:
            return self.eos_mix.d2V_dT2_l

        
    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        H = self.H_dep        
        for zi, Cp_int in zip(self.zs, self.Cp_integrals_pure):
            H += zi*Cp_int
        self._H = H
        return H

    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        Cp_integrals_over_T_pure = self.Cp_integrals_over_T_pure
        log_zs = self.log_zs()
        T, P, zs, cmps = self.T, self.P, self.zs, self.cmps
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)
        
        for i in cmps:
            S += zs[i]*Cp_integrals_over_T_pure[i]
        S += self.S_dep
        self._S = S
        return S
    
    def Cps_pure(self):
        try:
            return self._Cps
        except AttributeError:
            pass
        T = self.T
        self._Cps = [i.T_dependent_property(T) for i in self.HeatCapacityGases]
        return self._Cps
    
    def Cp_integrals_pure(self):
        try:
            return self._Cp_integrals_pure
        except AttributeError:
            pass
        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        self._Cp_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        return self._Cp_integrals_pure

    def Cp_integrals_over_T_pure(self):
        try:
            return self._Cp_integrals_over_T_pure
        except AttributeError:
            pass
        
        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        self._Cp_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        return self._Cp_integrals_over_T_pure
        

    def Cp(self):
        Cps_pure = self.Cps_pure
        Cp, zs = 0.0, self.zs
        for i in self.cmps:
            Cp += zs[i]*Cps_pure[i]
        return Cp + self.Cp_dep

    def dH_dT(self):
        return self.Cp

    def dH_dP(self):
        try:
            return self.eos_mix.dH_dep_dP_g
        except AttributeError:
            return self.eos_mix.dH_dep_dP_l

    def dH_dzs(self):
        try:
            return self._dH_dzs
        except AttributeError:
            pass
        eos_mix = self.eos_mix
        try:
            dH_dep_dzs = self.eos_mix.dH_dep_dzs(eos_mix.Z_g, eos_mix.zs)
        except AttributeError:
            dH_dep_dzs = self.eos_mix.dH_dep_dzs(eos_mix.Z_l, eos_mix.zs)
        Cp_integrals_pure = self.Cp_integrals_pure
        self._dH_dzs = [dH_dep_dzs[i] + Cp_integrals_pure[i] for i in self.cmps]
        return self._dH_dzs

    def dS_dT(self):
        HeatCapacityGases = self.HeatCapacityGases
        cmps = self.cmps
        T, zs = self.T, self.zs
        T_REF_IG = self.T_REF_IG
        P_REF_IG_INV = self.P_REF_IG_INV

        S = 0.0
        dS_pure_sum = 0.0
        for zi, obj in zip(zs, HeatCapacityGases):
            dS_pure_sum += zi*obj.T_dependent_property(T)
        S += dS_pure_sum/T
        try:
            S += self.eos_mix.dS_dep_dT_g
        except AttributeError:
            S += self.eos_mix.dS_dep_dT_l
        return S

    @property
    def dS_dP(self):
        dS = 0.0
        P = self.P
        dS -= R/P
        try:
            dS += self.eos_mix.dS_dep_dP_g
        except AttributeError:
            dS += self.eos_mix.dS_dep_dP_l
        return dS
            
    def dS_dzs(self):
        try:
            return self._dS_dzs
        except AttributeError:
            pass
        cmps, eos_mix = self.cmps, self.eos_mix
    
        log_zs = self.log_zs()
        integrals = self.Cp_integrals_over_T_pure

        try:
            dS_dep_dzs = self.eos_mix.dS_dep_dzs(eos_mix.Z_g, eos_mix.zs)
        except AttributeError:
            dS_dep_dzs = self.eos_mix.dS_dep_dzs(eos_mix.Z_l, eos_mix.zs)
        
        self._dS_dzs = [integrals[i] - R*(log_zs[i] + 1.0) + dS_dep_dzs[i] 
                        for i in cmps]
        return self._dS_dzs
 
            

class GibbbsExcessLiquid(Phase):
    def __init__(self, VaporPressures, VolumeLiquids=None, 
                 GibbsExcessModel=IdealSolution(), 
                 eos_pure_instances=None,
                 VolumeLiquidMixture=None,
                 HeatCapacityGases=None, 
                 EnthalpyVaporizations=None,
                 use_Poynting=False,
                 use_phis_sat=False,
                 Hfs=None, Gfs=None, Sfs=None,
                 T=None, P=None, zs=None,
                 ):
        self.VaporPressures = VaporPressures
        self.Psats_locked = all(i.locked for i in VaporPressures)
        self.Psats_data = ([i.best_fit_Tmin for i in VaporPressures],
                         [i.best_fit_Tmin_slope for i in VaporPressures],
                         [i.best_fit_Tmin_value for i in VaporPressures],
                         [i.best_fit_Tmax for i in VaporPressures],
                         [i.best_fit_Tmax_slope for i in VaporPressures],
                         [i.best_fit_Tmax_value for i in VaporPressures],
                         [i.best_fit_coeffs for i in VaporPressures])

        self.VolumeLiquids = VolumeLiquids
        self.GibbsExcessModel = GibbsExcessModel
        self.eos_pure_instances = eos_pure_instances
        self.VolumeLiquidMixture = VolumeLiquidMixture
        
        self.HeatCapacityGases = HeatCapacityGases
        self.EnthalpyVaporizations = EnthalpyVaporizations
        
        self.N = len(VaporPressures)
        self.cmps = range(self.N)
        
        self.use_Poynting = use_Poynting
        self.use_phis_sat = use_phis_sat
        
        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs

        if T is not None and P is not None and zs is not None:
            self.T = T
            self.P = P
            self.zs = zs
        
    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        new.N = self.N
        new.cmps = self.cmps
        
        new.VaporPressures = self.VaporPressures
        new.VolumeLiquids = self.VolumeLiquids
        new.VolumeLiquidMixture = self.VolumeLiquidMixture
        new.eos_pure_instances = self.eos_pure_instances
        new.HeatCapacityGases = self.HeatCapacityGases
        new.EnthalpyVaporizations = self.EnthalpyVaporizations
        
        new.GibbsExcessModel = self.GibbsExcessModel.to_T_xs(T=T, xs=zs)
        
        new.Psats_locked = self.Psats_locked
        new.Psats_data = self.Psats_data
        
        new.use_phis_sat = self.use_phis_sat
        new.use_Poynting = self.use_Poynting

        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        
        try:
            if T == self.T:
                try:
                    new._Psats = self._Psats
                except:
                    pass
        except:
            pass
        return new
        
        
        
    def Psats(self):
        try:
            return self._Psats
        except AttributeError:
            pass
        T, cmps = self.T, self.cmps
        # Need to reset the method because for the T bounded solver,
        # will normally get a different than prefered method as it starts
        # at the boundaries
        self._Psats = Psats = []
        if self.Psats_locked:
            Psats_data = self.Psats_data
            Tmins, Tmaxes, coeffs = Psats_data[0], Psats_data[3], Psats_data[6]
            for i in cmps:
                if T < Tmins[i]:
                    Psat = (T - Tmins[i])*Psats_data[1][i] + Psats_data[2][i]
                elif T > Tmaxes[i]:
                    Psat = (T - Tmaxes[i])*Psats_data[4][i] + Psats_data[5][i]
                else:
                    Psat = 0.0
                    for c in coeffs[i]:
                        Psat = Psat*T + c
#                    Psat = horner(coeffs[i], T)
                Psats.append(exp(Psat))
            return Psats


        
        for i in self.VaporPressures:
            if i.locked:
                Psats.append(i(T))
            else:
                if T < i.Tmax:
                    i.method = None
                    Psat = i(T)
                    if Psat is None:
                        Psat = i.extrapolate_tabular(T)
                    Psats.append(Psat)
                else:
                    Psats.append(i.extrapolate_tabular(T))
        return Psats
    
    
    def dPsats_dT(self):
        try:
            return self._dPsats_dT
        except:
            pass
        Psats = self.Psats()
        T, cmps = self.T, self.cmps
        # Need to reset the method because for the T bounded solver,
        # will normally get a different than prefered method as it starts
        # at the boundaries
        self._dPsats_dT = dPsats_dT = []
        if self.Psats_locked:
            Psats_data = self.Psats_data
            Tmins, Tmaxes, coeffs = Psats_data[0], Psats_data[3], Psats_data[6]
            for i in cmps:
                if T < Tmins[i]:
                    dPsat_dT = Psats_data[1][i]*Psats[i]#*exp((T - Tmins[i])*Psats_data[1][i]
                                                 #   + Psats_data[2][i])
                elif T > Tmaxes[i]:
                    dPsat_dT = Psats_data[4][i]*Psats[i]#*exp((T - Tmaxes[i])
                                                        #*Psats_data[4][i]
                                                        #+ Psats_data[5][i])
                else:
                    v, der = horner_and_der(coeffs[i], T)
                    dPsat_dT = der*Psats[i]
                dPsats_dT.append(dPsat_dT)
            return dPsats_dT

        self._dPsats_dT = dPsats_dT = [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]
        return dPsats_dT

    def Poyntings(self):
        try:
            return self._Poyntings
        except AttributeError:
            pass
        if not self.use_Poynting:
            return [1.0]*self.N
        
        T, P = self.T, self.P
        Psats = self.Psats()
        Vmls = [VolumeLiquid.T_dependent_property(T=T) for VolumeLiquid in self.VolumeLiquids]        
#        Vmls = [VolumeLiquid(T=T, P=P) for VolumeLiquid in self.VolumeLiquids]
        self._Poyntings = [exp(Vml*(P-Psat)/(R*T)) for Psat, Vml in zip(Psats, Vmls)]
        return self._Poyntings
    
    def dPoyntings_dT(self):
        try:
            return self._dPoyntings_dT
        except AttributeError:
            pass
        if not self.use_Poynting:
            return [0.0]*self.N
        
        Psats = self.Psats()
        T, P = self.T, self.P
            
        dPsats_dT = [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]

        Vmls = [VolumeLiquid.T_dependent_property(T=T) for VolumeLiquid in self.VolumeLiquids]                    
        dVml_dTs = [VolumeLiquid.T_dependent_property_derivative(T=T) 
                    for VolumeLiquid in self.VolumeLiquids]
#        Vmls = [VolumeLiquid(T=T, P=P) for VolumeLiquid in self.VolumeLiquids]
#        dVml_dTs = [VolumeLiquid.TP_dependent_property_derivative_T(T=T, P=P) 
#                    for VolumeLiquid in self.VolumeLiquids]
        
        x0 = 1.0/R
        x1 = 1.0/T
        
        self._dPoyntings_dT = dPoyntings_dT = []
        for i in self.cmps:
            x2 = Vmls[i]
            x3 = Psats[i]
            
            x4 = P - x3
            x5 = x1*x2*x4
            dPoyntings_dTi = -x0*x1*(x2*dPsats_dT[i] - x4*dVml_dTs[i] + x5)*exp(x0*x5)
            dPoyntings_dT.append(dPoyntings_dTi)
        return dPoyntings_dT
    
    
    def dPoyntings_dP(self):
        '''from sympy import *
        R, T, P, zi = symbols('R, T, P, zi')
        Vml = symbols('Vml', cls=Function)
        cse(diff(exp(Vml(T)*(P - Psati(T))/(R*T)), P), optimizations='basic')
        '''
        try:
            return self._dPoyntings_dP
        except AttributeError:
            pass
        if not self.use_Poynting:
            return [0.0]*self.N
        T, P = self.T, self.P
        Psats = self.Psats()
        
        Vmls = [VolumeLiquid(T=T, P=P) for VolumeLiquid in self.VolumeLiquids]
        
        self._dPoyntings_dP = dPoyntings_dPs = []
        for i in self.cmps:
            x0 = Vmls[i]/(R*T)
            dPoyntings_dPs.append(x0*exp(x0*(P - Psats[i])))
        return dPoyntings_dPs
        
    def phis_sat(self):
        try:
            return self._phis_sat
        except AttributeError:
            pass
        if not self.use_phis_sat:
            return [1.0]*self.N
        
        T = self.T
        self._phis_sat = phis_sat = []
        for obj in self.eos_pure_instances:
            Psat = obj.Psat(T)
            obj = obj.to_TP(T=T, P=Psat)
            # Along the saturation line, may not exist for one phase or the other even though incredibly precise
            try:
                phi = obj.phi_l
            except:
                phi = obj.phi_g
            phis_sat.append(phi)
        return phis_sat
                

    def fugacity_coefficients(self):
        try:
            return self._fugacity_coefficients
        except AttributeError:
            pass
        # DO NOT EDIT _ CORRECT
        T, P = self.T, self.P
        
        gammas = self.gammas()
        Psats = self.Psats()
        
        if self.use_phis_sat:
            phis = self.phis_sat()
        else:
            phis = [1.0]*self.N
            
        if self.use_Poynting:
            Poyntings = self.Poyntings()
        else:
            Poyntings = [1.0]*self.N
            
        P_inv = 1.0/P
        self._fugacity_coefficients = [gammas[i]*Psats[i]*Poyntings[i]*phis[i]*P_inv
                for i in self.cmps]
        return self._fugacity_coefficients
        
        
    def lnphis(self):
        try:
            return self._lnphis
        except AttributeError:
            pass
        self._lnphis = [log(i) for i in self.fugacity_coefficients()]        
        return self._lnphis
        
#    def fugacities(self, T, P, zs):
#        # DO NOT EDIT _ CORRECT
#        gammas = self.gammas(T, zs)
#        Psats = self._Psats(T=T)
#        if self.use_phis_sat:
#            phis = self.phis(T=T, zs=zs)
#        else:
#            phis = [1.0]*self.N
#            
#        if self.use_Poynting:
#            Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
#        else:
#            Poyntings = [1.0]*self.N
#        return [zs[i]*gammas[i]*Psats[i]*Poyntings[i]*phis[i]
#                for i in self.cmps]
#

    def dphis_dT(self):
        try:
            return self._dphis_dT
        except AttributeError:
            pass
        T, P, zs = self.T, self.P, self.zs
        Psats = self.Psats()
        gammas = self.gammas()
        
        
        if self.use_Poynting:
            # Evidence suggests poynting derivatives are not worth calculating
            dPoyntings_dT = [0.0]*self.N#self.dPoyntings_dT(T, P, Psats=Psats)
            Poyntings = self.Poyntings()
        else:
            dPoyntings_dT = [0.0]*self.N
            Poyntings = [1.0]*self.N

        dPsats_dT = self.dPsats_dT()
        
        dgammas_dT = self.GibbsExcessModel.dgammas_dT()
        
        if self.use_phis_sat:
            dphis_sat_dT = 0.0
            phis_sat = self.phis_sat()
        else:
            dphis_sat_dT = 0.0
            phis_sat = [1.0]*self.N
        
#        print(gammas, phis_sat, Psats, Poyntings, dgammas_dT, dPoyntings_dT, dPsats_dT)
        self._dphis_dT = dphis_dTl = []
        for i in self.cmps:
            x0 = gammas[i]
            x1 = phis_sat[i]
            x2 = Psats[i]
            x3 = Poyntings[i]
            x4 = x2*x3
            x5 = x0*x1
            v = (x0*x4*dphis_sat_dT + x1*x4*dgammas_dT[i] + x2*x5*dPoyntings_dT[i] + x3*x5*dPsats_dT[i])/P
            dphis_dTl.append(v)
        return dphis_dTl
        
    def dlnphis_dT(self):
        try:
            return self._dlnphis_dT
        except AttributeError:
            pass
        dphis_dT = self.dphis_dT()
        phis = self.fugacity_coefficients()
        self._dlnphis_dT = [i/j for i, j in zip(dphis_dT, phis)]
        return self._dlnphis_dT
    
    # TODO - implement dlnphis_dx, convert do dlnphis_dn

    def gammas(self):
        return self.GibbsExcessModel.gammas()
    
    def H(self):
        # Untested
        H = 0
        T = self.T
        P = self.P
        Psats = self.Psats()
        for i in self.cmps:
            # No further contribution needed
            Hg298_to_T = self.HeatCapacityGases[i].T_dependent_property_integral(self.T_REF_IG, T)
            Hvap = self.EnthalpyVaporizations[i](T) # Do the transition at the temperature of the liquid
            if Hvap is None:
                Hvap = 0 # Handle the case of a package predicting a transition past the Tc
            H_i = Hg298_to_T - Hvap 
            if self.P_DEPENDENT_H_LIQ:
                Vl = self.VolumeLiquids[i](T, P)
                if Vl is None:
                    # Handle an inability to get a liquid volume by taking
                    # one at the boiling point (and system P)
                    Vl = self.VolumeLiquids[i](self.Tbs[i], P)
                H_i += (P - Psats[i])*Vl
            H += self.zs[i]*(H_i) 
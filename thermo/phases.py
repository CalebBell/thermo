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
__all__ = ['GibbbsExcessLiquid', 'Phase', 'EOSLiquid', 'EOSGas']

from fluids.constants import R, R_inv
from thermo.utils import (log, exp, Cp_minus_Cv, phase_identification_parameter,
                          isothermal_compressibility, isobaric_expansion,
                          Joule_Thomson)


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

    @property
    def G(self):
        G = self.H - self.T*self.S
        return G
    
    @property
    def U(self):
        U = self.H - self.P*self.V
        return U
    
    @property
    def A(self):
        A = self.U - self.T*self.S
        return A
    
    @property
    def dG_dT(self):
        return -self.T*self.dS_dT - self.S + self.dH_dT
    
    @property
    def dG_dP(self):
        return -self.T*self.dS_dP + self.dH_dP
    
    @property
    def dU_dT(self):
        # Correct
        return -self.P*self.dV_dT + self.dH_dT
    
    @property
    def dU_dP(self):
        # Correct
        return -self.P*self.dV_dP - self.V + self.dH_dP
    
    @property
    def dA_dT(self):
        return -self.T*self.dS_dT - self.S + self.dU_dT
    
    @property
    def dA_dP(self):
        return -self.T*self.dS_dP + self.dU_dP
        
    @property
    def G_dep(self):
        G_dep = self.H_dep - self.T*self.S_dep
        return G_dep
    
    @property
    def V_dep(self):
        # from ideal gas behavior
        V_dep = self.V - R*self.T/self.P
        return V_dep
    
    @property
    def U_dep(self):
        return self.H_dep - self.P*self.V_dep
    
    @property
    def A_dep(self):
        return self.U_dep - self.T*self.S_dep


    @property
    def H_reactive(self):
        H = self.H
        for zi, Hf in zip(self.zs, self.Hfs):
            H += zi*Hf
        return H

    @property
    def S_reactive(self):
        S = self.S
        for zi, Sf in zip(self.zs, self.Sfs):
            S += zi*Sf
        return S
    
    @property
    def G_reactive(self):
        G = self.H_reactive - self.T*self.S_reactive
        return G
    
    @property
    def U_reactive(self):
        U = self.H_reactive - self.P*self.V
        return U
    
    @property
    def A_reactive(self):
        A = self.U_reactive - self.T*self.S_reactive
        return A
    
    @property
    def Cv(self):
        # checks out
        Cp_m_Cv = Cp_minus_Cv(self.T, self.dP_dT, self.dP_dV)
        Cp = self.Cp
        return Cp - Cp_m_Cv
    
    @property
    def Cp_Cv_ratio(self):
        return self.Cp/self.Cv
        
    @property
    def rho(self):
        return 1.0/self.V
    
    @property
    def dT_dP(self):
        return 1.0/self.dP_dT
    
    @property
    def dV_dT(self):
        return -self.dP_dT/self.dP_dV
    
    @property
    def dV_dP(self):
        return -self.dV_dT*self.dT_dP 
    
    @property
    def dT_dV(self):
        return 1./self.dV_dT
    
    @property
    def d2V_dP2(self):
        inverse_dP_dV = 1.0/self.dP_dV
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV*inverse_dP_dV
        return -self.d2P_dV2*inverse_dP_dV3

    @property
    def d2T_dP2(self):
        dT_dP = self.dT_dP
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP
        return -self.d2P_dT2*inverse_dP_dT3
    
    @property
    def d2T_dV2(self):
        # Wrong
        dP_dT = self.dP_dT
        dP_dV = self.dP_dV
        d2P_dTdV = self.d2P_dTdV
        d2P_dT2 = self.d2P_dT2
        dT_dP = self.dT_dP
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP
        
        return (-(self.d2P_dV2*dP_dT - dP_dV*d2P_dTdV)*inverse_dP_dT2
                   +(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3*dP_dV)
        
    @property
    def d2V_dT2(self):
        dP_dT = self.dP_dT
        dP_dV = self.dP_dV
        d2P_dTdV = self.d2P_dTdV
        d2P_dT2 = self.d2P_dT2
        d2P_dV2 = self.d2P_dV2

        inverse_dP_dV = 1.0/dP_dV
        inverse_dP_dV2 = inverse_dP_dV*inverse_dP_dV
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV2
        
        return  (-(d2P_dT2*dP_dV - dP_dT*d2P_dTdV)*inverse_dP_dV2
                   +(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*inverse_dP_dV3*dP_dT)

    @property
    def d2V_dPdT(self):
        dP_dT = self.dP_dT
        dP_dV = self.dP_dV
        d2P_dTdV = self.d2P_dTdV
        d2P_dV2 = self.d2P_dV2
        
        inverse_dP_dV = 1.0/dP_dV
        inverse_dP_dV2 = inverse_dP_dV*inverse_dP_dV
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV2
        
        return -(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*inverse_dP_dV3

    @property
    def d2T_dPdV(self):
        dT_dP = self.dT_dP
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP
        
        d2P_dTdV = self.d2P_dTdV
        dP_dT = self.dP_dT
        dP_dV = self.dP_dV
        d2P_dT2 = self.d2P_dT2
        return -(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3
    
    
    @property
    def PIP(self):
        return phase_identification_parameter(self.V, self.dP_dT, self.dP_dV, 
                                              self.d2P_dV2, self.d2P_dTdV)
        
    @property
    def kappa(self):
        return isothermal_compressibility(self.V, self.dV_dP)

    @property
    def beta(self):
        return isobaric_expansion(self.V, self.dV_dT)
    
    @property
    def Joule_Thomson(self):
        return Joule_Thomson(self.T, self.V, self.Cp, dV_dT=self.dV_dT, beta=self.beta)
        
    @property
    def dZ_dT(self):
        T_inv = 1.0/self.T
        return self.P*R_inv*T_inv*(self.dV_dT - self.V*T_inv)

    @property
    def dZ_dP(self):
        return 1.0/(self.T*R)*(self.V + self.P*self.dV_dP)

    ### Derivatives in the molar density basis

    @property
    def d2V_dTdP(self):
        return self.d2V_dPdT

    @property
    def d2P_dVdT(self):
        return self.d2P_dTdV

    @property
    def d2T_dVdP(self):
        return self.d2T_dPdV

    @property
    def dP_drho(self):
        return -self.V*self.V*self.dP_dV 

    @property
    def drho_dP(self):
        return -self.dV_dP/(self.V*self.V)

    @property
    def d2P_drho2(self):
        return -self.V**2*(-self.V**2*self.d2P_dV2 - 2*self.V*self.dP_dV)

    @property
    def d2rho_dP2(self):
        return -self.d2V_dP2/self.V**2 + 2*self.dV_dP**2/self.V**3

    @property
    def dT_drho(self):
        return -self.V*self.V*self.dT_dV

    @property
    def d2T_drho2(self):
        return -self.V**2*(-self.V**2*self.d2T_dV2 - 2*self.V*self.dT_dV)

    @property
    def drho_dT(self):
        return -self.dV_dT/(self.V*self.V)

    @property
    def d2rho_dT2(self):
        return -self.d2V_dT2/self.V**2 + 2*self.dV_dT**2/self.V**3

    @property
    def d2P_dTdrho(self):
        return -(self.V*self.V)*self.d2P_dTdV

    @property
    def d2T_dPdrho(self):
        return -(self.V*self.V)*self.d2T_dPdV

    @property
    def d2rho_dPdT(self):
        return -self.d2V_dPdT/self.V**2 + 2*self.dV_dT*self.dV_dP/self.V**3



class EOSLiquid(Phase):
    pass

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
    
    @property
    def H_dep(self):
        try:
            return self.eos_mix.H_dep_g
        except AttributeError:
            return self.eos_mix.H_dep_l

    @property
    def S_dep(self):
        try:
            return self.eos_mix.S_dep_g
        except AttributeError:
            return self.eos_mix.S_dep_l

    @property
    def Cp_dep(self):
        try:
            return self.eos_mix.Cp_dep_g
        except AttributeError:
            return self.eos_mix.Cp_dep_l        
        
    @property
    def V(self):
        try:
            return self.eos_mix.V_g
        except AttributeError:
            return self.eos_mix.V_l

    @property
    def Z(self):
        try:
            return self.eos_mix.Z_g
        except AttributeError:
            return self.eos_mix.Z_l
    
    
    @property
    def dP_dT(self):
        try:
            return self.eos_mix.dP_dT_g
        except AttributeError:
            return self.eos_mix.dP_dT_l

    @property
    def dP_dV(self):
        try:
            return self.eos_mix.dP_dV_g
        except AttributeError:
            return self.eos_mix.dP_dV_l
    
    @property
    def d2P_dT2(self):
        try:
            return self.eos_mix.d2P_dT2_g
        except AttributeError:
            return self.eos_mix.d2P_dT2_l

    @property
    def d2P_dV2(self):
        try:
            return self.eos_mix.d2P_dV2_g
        except AttributeError:
            return self.eos_mix.d2P_dV2_l

    @property
    def d2P_dTdV(self):
        try:
            return self.eos_mix.d2P_dTdV_g
        except AttributeError:
            return self.eos_mix.d2P_dTdV_l
        
    # because of the ideal gas model, for some reason need to use the right ones
    # FOR THIS MODEL ONLY
    @property
    def d2T_dV2(self):
        try:
            return self.eos_mix.d2T_dV2_g
        except AttributeError:
            return self.eos_mix.d2T_dV2_l

    @property
    def d2V_dT2(self):
        try:
            return self.eos_mix.d2V_dT2_g
        except AttributeError:
            return self.eos_mix.d2V_dT2_l

        
    @property
    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        T_REF_IG = self.T_REF_IG
        HeatCapacityGases = self.HeatCapacityGases
        zs = self.zs
        T = self.T

        H = 0.0        
        for zi, obj in zip(zs, HeatCapacityGases):
            H += zi*obj.T_dependent_property_integral(T_REF_IG, T)
        H += self.H_dep
        self._H = H
        return H

    @property
    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        HeatCapacityGases = self.HeatCapacityGases
        cmps = self.cmps
        T, P, zs = self.T, self.P, self.zs
        
        T_REF_IG = self.T_REF_IG
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zi*log(zi) for zi in zs if zi > 0.0]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)
        for i in cmps:
            dS = HeatCapacityGases[i].T_dependent_property_integral_over_T(T_REF_IG, T)
            S += zs[i]*dS
        S += self.S_dep
        self._S = S
        return S

    @property
    def Cp(self):
        Cp = 0.0
        for i in self.cmps:
            Cp += self.zs[i]*self.HeatCapacityGases[i].T_dependent_property(self.T)
        return Cp + self.Cp_dep

    @property
    def dH_dT(self):
        return self.Cp

    @property
    def dH_dP(self):
        try:
            return self.eos_mix.dH_dep_dP_g
        except AttributeError:
            return self.eos_mix.dH_dep_dP_l

    @property
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
            
            

class GibbbsExcessLiquid(Phase):
    
    use_Poynting = False
    use_phis_sat = False
    def __init__(self, VaporPressures, VolumeLiquids, GibbsExcessModel, 
                 eos_pure_instances, VolumeLiquidMixture=None,
                 HeatCapacityGases=None, EnthalpyVaporizations=None):
        self.VaporPressures = VaporPressures
        self.VolumeLiquids = VolumeLiquids
        self.GibbsExcessModel = GibbsExcessModel
        self.eos_pure_instances = eos_pure_instances
        self.VolumeLiquidMixture = VolumeLiquidMixture
        
        self.HeatCapacityGases = HeatCapacityGases
        self.EnthalpyVaporizations = EnthalpyVaporizations
        
        self.N = len(VaporPressures)
        self.cmps = range(self.N)
        
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
        
        try:
            if T == self.T:
                new._Psats = self._Psats
        except:
            pass
        return new
        
        
        
    def Psats(self):
        try:
            return self._Psats
        except AttributeError:
            pass
        T = self.T
        # Need to reset the method because for the T bounded solver,
        # will normally get a different than prefered method as it starts
        # at the boundaries
        self._Psats = Psats = []
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
        
    def Poyntings(self):
        try:
            return self._Poyntings
        except AttributeError:
            pass
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
        
    def fugacities(self, T, P, zs):
        # DO NOT EDIT _ CORRECT
        gammas = self.gammas(T, zs)
        Psats = self._Psats(T=T)
        if self.use_phis_sat:
            phis = self.phis(T=T, zs=zs)
        else:
            phis = [1.0]*self.N
            
        if self.use_Poynting:
            Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
        else:
            Poyntings = [1.0]*self.N
        return [zs[i]*gammas[i]*Psats[i]*Poyntings[i]*phis[i]
                for i in self.cmps]


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

        dPsats_dT = [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]
        
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

    def gammas(self):
        return self.GibbsExcessModel.gammas()
    
    @property
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
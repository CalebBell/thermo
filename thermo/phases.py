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
__all__ = ['GibbsExcessLiquid', 'GibbsExcessSolid', 'Phase', 'EOSLiquid', 'EOSGas', 'IdealGas',
           'gas_phases', 'liquid_phases', 'solid_phases']

from fluids.constants import R, R_inv
from fluids.numerics import (horner, horner_and_der, horner_log, jacobian, derivative,
                             best_fit_integral_value, best_fit_integral_over_T_value)
from thermo.utils import (log, exp, Cp_minus_Cv, phase_identification_parameter,
                          isothermal_compressibility, isobaric_expansion,
                          Joule_Thomson, speed_of_sound, dxs_to_dns,
                          normalize)
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

    Psats_data = None
    Cpgs_data = None
    Psats_locked = False 
    Cpgs_locked = False
    
    def fugacities(self):
        P = self.P
        zs = self.zs
        lnphis = self.lnphis()
        return [P*zs[i]*exp(lnphis[i]) for i in range(len(zs))]
    
    def dfugacities_dT(self):
        r'''
        '''
        dphis_dT = self.dphis_dT()
        P, zs = self.P, self.zs
        return [P*zs[i]*dphis_dT[i] for i in range(len(zs))]

    def phis(self):
        return [exp(i) for i in self.lnphis()]

    def dphis_dT(self):
        r'''Method to calculate the temperature derivative of fugacity 
        coefficients of the phase.
        
        .. math::
            \frac{\partial \phi_i}{\partial T} = \phi_i \frac{\partial 
            \log \phi_i}{\partial T} 

        Returns
        -------
        dphis_dT : list[float]
            Temperature derivative of fugacity coefficients of all components
            in the phase, [1/K]
            
        Notes
        -----
        '''        
        try:
            return self._dphis_dT
        except AttributeError:
            pass
        try:
            dlnphis_dT = self._dlnphis_dT
        except AttributeError:
            dlnphis_dT = self.dlnphis_dT()
            
        try:
            phis = self._phis
        except AttributeError:
            phis = self.phis()

        self._dphis_dT = [dlnphis_dT[i]*phis[i] for i in self.cmps]
        return self._dphis_dT
    
    def dphis_dP(self):
        r'''Method to calculate the pressure derivative of fugacity 
        coefficients of the phase.
        
        .. math::
            \frac{\partial \phi_i}{\partial P} = \phi_i \frac{\partial 
            \log \phi_i}{\partial P} 

        Returns
        -------
        dphis_dP : list[float]
            Pressure derivative of fugacity coefficients of all components
            in the phase, [1/Pa]
            
        Notes
        -----
        '''        
        try:
            return self._dphis_dP
        except AttributeError:
            pass
        try:
            dlnphis_dP = self._dlnphis_dP
        except AttributeError:
            dlnphis_dP = self.dlnphis_dP()
            
        try:
            phis = self._phis
        except AttributeError:
            phis = self.phis()

        self._dphis_dP = [dlnphis_dP[i]*phis[i] for i in self.cmps]
        return self._dphis_dP

    def dfugacities_dP(self):
        r'''Method to calculate the pressure derivative of the fugacities
        of the components in the phase phase.
        
        .. math::
            \frac{\partial f_i}{\partial P} = z_i \left(P \frac{\partial 
            \phi_i}{\partial P}  + \phi_i \right)

        Returns
        -------
        dfugacities_dP : list[float]
            Pressure derivative of fugacities of all components
            in the phase, [-]
            
        Notes
        -----
        For models without pressure dependence of fugacity, the returned result
        may not be exactly zero due to inaccuracy in floating point results;
        results are likely on the order of 1e-14 or lower in that case.
        '''        
        try:
            dphis_dP = self._dphis_dP
        except AttributeError:
            dphis_dP = self.dphis_dP()
            
        try:
            phis = self._phis
        except AttributeError:
            phis = self.phis()

        P, zs = self.P, self.zs
        return [zs[i]*(P*dphis_dP[i] + phis[i]) for i in self.cmps]


    def log_zs(self):
        try:
            return self._log_zs
        except AttributeError:
            pass
        try:
            self._log_zs = [log(zi) for zi in self.zs]
        except ValueError:
            self._log_zs = _log_zs = []
            for zi in self.zs:
                try:
                    _log_zs.append(log(zi))
                except ValueError:
                    _log_zs.append(-690.7755278982137) # log(1e-300)
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
        H = self.H()
        for zi, Hf in zip(self.zs, self.Hfs):
            H += zi*Hf
        return H

    def S_reactive(self):
        S = self.S()
        for zi, Sf in zip(self.zs, self.Sfs):
            S += zi*Sf
        return S
    
    def G_reactive(self):
        G = self.H_reactive() - self.T*self.S_reactive()
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
    

    def chemical_potential(self):
        # CORRECT DO NOT CHANGE
        # TODO analytical implementation
        def to_diff(ns):
            tot = sum(ns)
            zs = normalize(ns)
            return tot*self.to_TP_zs(self.T, self.P, zs).G_reactive()
        return jacobian(to_diff, self.zs)
    
    def activities(self):
        # CORRECT DO NOT CHANGE
        fugacities = self.fugacities()
        fugacities_std = self.fugacities_std() # TODO implement
        return [fugacities[i]/fugacities_std[i] for i in self.cmps]
    
    def gammas(self):
        # For a good discussion, see 
        # Thermodynamics: Fundamentals for Applications, J. P. O'Connell, J. M. Haile
        # There is no one single definition for gamma but it is believed this is
        # the most generally used one for EOSs; and activity methods
        # override this
        phis = self.phis()
        phis_pure = []
        T, P, zs, cmps, N = self.T, self.P, self.zs, self.cmps, self.N
        for i in cmps:
            zeros = [0.0]*N
            zeros[i] = 1.0
            phi = self.to_TP_zs(T=T, P=P, zs=zeros).phis()[i]
            phis_pure.append(phi)
        return [phis[i]/phis_pure[i] for i in cmps]
        
        

    def Cp_Cv_ratio(self):
        return self.Cp()/self.Cv()
    
    def Z(self):
        return self.P*self.V()/(R*self.T)
        
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

    # A few aliases
    def d2V_dTdP(self):
        return self.d2V_dPdT()

    def d2P_dVdT(self):
        return self.d2P_dTdV()

    def d2T_dVdP(self):
        return self.d2T_dPdV()

    # Derived properties    
    def PIP(self):
        return phase_identification_parameter(self.V(), self.dP_dT(), self.dP_dV(), 
                                              self.d2P_dV2(), self.d2P_dTdV())
        
    def kappa(self):
        return isothermal_compressibility(self.V(), self.dV_dP())

    def beta(self):
        return isobaric_expansion(self.V(), self.dV_dT())
    
    def Joule_Thomson(self):
        return Joule_Thomson(self.T, self.V(), self.Cp(), dV_dT=self.dV_dT(), beta=self.beta())
    
    def speed_of_sound(self):
        return speed_of_sound(self.V(), self.dP_dV(), self.Cp(), self.Cv())
    
    ### Compressibility factor derivatives
    def dZ_dT(self):
        T_inv = 1.0/self.T
        return self.P*R_inv*T_inv*(self.dV_dT() - self.V()*T_inv)

    def dZ_dP(self):
        return 1.0/(self.T*R)*(self.V() + self.P*self.dV_dP())
    # Could add more

    ### Derivatives in the molar density basis
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
    
    # Idea gas heat capacity
    
    def setup_Cpigs(self, HeatCapacityGases):
        Cpgs_data = None
        Cpgs_locked = all(i.locked for i in HeatCapacityGases) if HeatCapacityGases is not None else False
        if Cpgs_locked:
            T_REF_IG = self.T_REF_IG
            Cpgs_data = ([i.best_fit_Tmin for i in HeatCapacityGases],
                              [i.best_fit_Tmin_slope for i in HeatCapacityGases],
                              [i.best_fit_Tmin_value for i in HeatCapacityGases],
                              [i.best_fit_Tmax for i in HeatCapacityGases],
                              [i.best_fit_Tmax_slope for i in HeatCapacityGases],
                              [i.best_fit_Tmax_value for i in HeatCapacityGases],
                              [i.best_fit_log_coeff for i in HeatCapacityGases],
#                              [horner(i.best_fit_int_coeffs, i.best_fit_Tmin) for i in HeatCapacityGases],
                              [horner(i.best_fit_int_coeffs, i.best_fit_Tmin) - i.best_fit_Tmin*(0.5*i.best_fit_Tmin_slope*i.best_fit_Tmin + i.best_fit_Tmin_value - i.best_fit_Tmin_slope*i.best_fit_Tmin) for i in HeatCapacityGases],
#                              [horner(i.best_fit_int_coeffs, i.best_fit_Tmax) for i in HeatCapacityGases],
                              [horner(i.best_fit_int_coeffs, i.best_fit_Tmax) - horner(i.best_fit_int_coeffs, i.best_fit_Tmin) + i.best_fit_Tmin*(0.5*i.best_fit_Tmin_slope*i.best_fit_Tmin + i.best_fit_Tmin_value - i.best_fit_Tmin_slope*i.best_fit_Tmin) for i in HeatCapacityGases],
#                              [horner_log(i.best_fit_T_int_T_coeffs, i.best_fit_log_coeff, i.best_fit_Tmin) for i in HeatCapacityGases],
                              [horner_log(i.best_fit_T_int_T_coeffs, i.best_fit_log_coeff, i.best_fit_Tmin) -(i.best_fit_Tmin_slope*i.best_fit_Tmin + (i.best_fit_Tmin_value - i.best_fit_Tmin_slope*i.best_fit_Tmin)*log(i.best_fit_Tmin)) for i in HeatCapacityGases],
#                              [horner_log(i.best_fit_T_int_T_coeffs, i.best_fit_log_coeff, i.best_fit_Tmax) for i in HeatCapacityGases],
                              [(horner_log(i.best_fit_T_int_T_coeffs, i.best_fit_log_coeff, i.best_fit_Tmax)
                                - horner_log(i.best_fit_T_int_T_coeffs, i.best_fit_log_coeff, i.best_fit_Tmin) 
                                + (i.best_fit_Tmin_slope*i.best_fit_Tmin + (i.best_fit_Tmin_value - i.best_fit_Tmin_slope*i.best_fit_Tmin)*log(i.best_fit_Tmin)) 
                                - (i.best_fit_Tmax_value -i.best_fit_Tmax*i.best_fit_Tmax_slope)*log(i.best_fit_Tmax)) for i in HeatCapacityGases],
                              [best_fit_integral_value(T_REF_IG, i.best_fit_int_coeffs, i.best_fit_Tmin, 
                                                       i.best_fit_Tmax, i.best_fit_Tmin_value,
                                                       i.best_fit_Tmax_value, i.best_fit_Tmin_slope,
                                                       i.best_fit_Tmax_slope) for i in HeatCapacityGases],
                              [i.best_fit_coeffs for i in HeatCapacityGases],
                              [i.best_fit_int_coeffs for i in HeatCapacityGases],
                              [i.best_fit_T_int_T_coeffs for i in HeatCapacityGases],
                              [best_fit_integral_over_T_value(T_REF_IG, i.best_fit_T_int_T_coeffs, i.best_fit_log_coeff, i.best_fit_Tmin, 
                                                       i.best_fit_Tmax, i.best_fit_Tmin_value,
                                                       i.best_fit_Tmax_value, i.best_fit_Tmin_slope,
                                                       i.best_fit_Tmax_slope) for i in HeatCapacityGases],
                              
                              )
        return (Cpgs_locked, Cpgs_data)

    
    def Cpigs_pure(self):
        try:
            return self._Cps
        except AttributeError:
            pass
        if self.Cpgs_locked:
            self._Cps = Cps = []
            T, Cpgs_data, cmps = self.T, self.Cpgs_data, self.cmps
            Tmins, Tmaxs, coeffs = Cpgs_data[0], Cpgs_data[3], Cpgs_data[12]
            Tmin_slopes = Cpgs_data[1]
            Tmin_values = Cpgs_data[2]
            Tmax_slopes = Cpgs_data[4]
            Tmax_values = Cpgs_data[5]
            
            for i in cmps:
                if T < Tmins[i]:
                    Cp = (T -  Tmins[i])*Tmin_slopes[i] + Tmin_values[i]
                elif T > Tmaxs[i]:
                    Cp = (T - Tmaxs[i])*Tmax_slopes[i] + Tmax_values[i]
                else:
                    Cp = 0.0
                    for c in coeffs[i]:
                        Cp = Cp*T + c
                Cps.append(Cp)
            return Cps
                
        
        T = self.T
        self._Cps = [i.T_dependent_property(T) for i in self.HeatCapacityGases]
        return self._Cps
    
    def Cpig_integrals_pure(self):
        try:
            return self._Cpig_integrals_pure
        except AttributeError:
            pass
        if self.Cpgs_locked:
            self._Cpig_integrals_pure = Cpig_integrals_pure = []
            T, Cpgs_data, cmps = self.T, self.Cpgs_data, self.cmps
            Tmins, Tmaxes, int_coeffs = Cpgs_data[0], Cpgs_data[3], Cpgs_data[13]
            for i in cmps:
                # If indeed everything is working here, need to optimize to decide what to store
                # Try to save lookups to avoid cache misses
                # Instead of storing horner Tmin and Tmax, store -:
                # tot(Tmin) - Cpgs_data[7][i]
                # and tot1 + tot for the high T
                # Should save quite a bit of lookups! est. .12 go to .09
#                Tmin = Tmins[i]
#                if T < Tmin:
#                    x1 = Cpgs_data[2][i] - Cpgs_data[1][i]*Tmin
#                    H = T*(0.5*Cpgs_data[1][i]*T + x1)
#                elif (T <= Tmaxes[i]):
#                    x1 = Cpgs_data[2][i] - Cpgs_data[1][i]*Tmin
#                    tot = Tmin*(0.5*Cpgs_data[1][i]*Tmin + x1)
#                    
#                    tot1 = 0.0
#                    for c in int_coeffs[i]:
#                        tot1 = tot1*T + c
#                    tot1 -= Cpgs_data[7][i]
##                    tot1 = horner(int_coeffs[i], T) - horner(int_coeffs[i], Tmin)
#                    H = tot + tot1
#                else:
#                    x1 = Cpgs_data[2][i] - Cpgs_data[1][i]*Tmin
#                    tot = Tmin*(0.5*Cpgs_data[1][i]*Tmin + x1)
#                    
#                    tot1 = Cpgs_data[8][i] - Cpgs_data[7][i]
#                    
#                    x1 = Cpgs_data[5][i] - Cpgs_data[4][i]*Tmaxes[i]
#                    tot2 = T*(0.5*Cpgs_data[4][i]*T + x1) - Tmaxes[i]*(0.5*Cpgs_data[4][i]*Tmaxes[i] + x1)
#                    H = tot + tot1 + tot2
                    
                    
                    
                # ATTEMPT AT FAST HERE (NOW WORKING)
                if T < Tmins[i]:
                    x1 = Cpgs_data[2][i] - Cpgs_data[1][i]*Tmins[i]
                    H = T*(0.5*Cpgs_data[1][i]*T + x1)
                elif (T <= Tmaxes[i]):
                    H = 0.0
                    for c in int_coeffs[i]:
                        H = H*T + c
                    H -= Cpgs_data[7][i]
                else:
                    Tmax_slope = Cpgs_data[4][i]
                    x1 = Cpgs_data[5][i] - Tmax_slope*Tmaxes[i]
                    H = T*(0.5*Tmax_slope*T + x1) - Tmaxes[i]*(0.5*Tmax_slope*Tmaxes[i] + x1)
                    H += Cpgs_data[8][i]

                Cpig_integrals_pure.append(H - Cpgs_data[11][i])
            return Cpig_integrals_pure



        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        self._Cpig_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        return self._Cpig_integrals_pure

    def Cpig_integrals_over_T_pure(self):
        try:
            return self._Cpig_integrals_over_T_pure
        except AttributeError:
            pass
        
        if self.Cpgs_locked:
            # Room for more optimizing in here
            self._Cpig_integrals_over_T_pure = Cpig_integrals_over_T_pure = []
            T, Cpgs_data, cmps = self.T, self.Cpgs_data, self.cmps
            Tmins, Tmaxes, T_int_T_coeffs = Cpgs_data[0], Cpgs_data[3], Cpgs_data[14]
            logT = log(T)
            for i in cmps:
                Tmin = Tmins[i]
                if T < Tmin:
                    x1 = Cpgs_data[2][i] - Cpgs_data[1][i]*Tmin
                    S = (Cpgs_data[1][i]*T + x1*logT)
                elif (Tmin <= T <= Tmaxes[i]):
                    S = 0.0
                    for c in T_int_T_coeffs[i]:
                        S = S*T + c
                    S += Cpgs_data[6][i]*logT
                    # The below should be in a constant - taking the place of Cpgs_data[9]
                    S -= Cpgs_data[9][i]
#                    x1 = Cpgs_data[2][i] - Cpgs_data[1][i]*Tmin
#                    S += (Cpgs_data[1][i]*Tmin + x1*log(Tmin))
                else:        
#                    x1 = Cpgs_data[2][i] - Cpgs_data[1][i]*Tmin
#                    S = (Cpgs_data[1][i]*Tmin + x1*log(Tmin))
#                    S += (Cpgs_data[10][i] - Cpgs_data[9][i])
                    S = Cpgs_data[10][i] 
                    # The above should be in the constant Cpgs_data[10], - x2*log(Tmaxes[i]) also
                    x2 = Cpgs_data[5][i] - Tmaxes[i]*Cpgs_data[4][i]
                    S += -Cpgs_data[4][i]*(Tmaxes[i] - T) + x2*logT #- x2*log(Tmaxes[i])
                    
                Cpig_integrals_over_T_pure.append(S - Cpgs_data[15][i])
            return Cpig_integrals_over_T_pure

                
        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        self._Cpig_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        return self._Cpig_integrals_over_T_pure

    ### Transport properties - pass them on!
    # Properties that use `constants` attributes
    
    def MW(self):
        try:
            return self._MW
        except AttributeError:
            pass
        zs, MWs = self.zs, self.constants.MWs
        MW = 0.0
        for i in self.cmps:
            MW += zs[i]*MWs[i]
        self._MW = MW
        return MW
    
    def MW_inv(self):
        try:
            return self._MW_inv
        except AttributeError:
            pass
        self._MW_inv = MW_inv = 1.0/self.MW()
        return MW_inv
    
    def mu(self):
        return self.result.mu(self)

    def k(self):
        return self.result.k(self)

    def ws(self):
        return self.result.ws(self)
        
    def speed_of_sound_mass(self):
        # 1000**0.5 = 31.622776601683793
        return 31.622776601683793*self.MW()**-0.5*self.speed_of_sound()
    
    def rho_mass(self):
        try:
            return self._rho_mass
        except AttributeError:
            pass
        self._rho_mass = rho_mass = self.MW()/(1000.0*self.V())
        return rho_mass
    
    def H_mass(self):
        try:
            return self._H_mass
        except AttributeError:
            pass
        
        self._H_mass = H_mass = self.H()*1e3*self.MW_inv()
        return H_mass

    def S_mass(self):
        try:
            return self._S_mass
        except AttributeError:
            pass
        
        self._S_mass = S_mass = self.S()*1e3*self.MW_inv()
        return S_mass

    def U_mass(self):
        try:
            return self._U_mass
        except AttributeError:
            pass
        
        self._U_mass = U_mass = self.U()*1e3*self.MW_inv()
        return U_mass

    def A_mass(self):
        try:
            return self._A_mass
        except AttributeError:
            pass
        
        self._A_mass = A_mass = self.A()*1e3*self.MW_inv()
        return A_mass

    def G_mass(self):
        try:
            return self._G_mass
        except AttributeError:
            pass
        
        self._G_mass = G_mass = self.G()*1e3*self.MW_inv()
        return G_mass

    def Cp_mass(self):
        try:
            return self._Cp_mass
        except AttributeError:
            pass
        
        self._Cp_mass = Cp_mass = self.Cp()*1e3*self.MW_inv()
        return Cp_mass

    def Cv_mass(self):
        try:
            return self._Cv_mass
        except AttributeError:
            pass
        
        self._Cv_mass = Cv_mass = self.Cv()*1e3*self.MW_inv()
        return Cv_mass

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

    def dlnphis_dP(self):
        return [0.0]*self.N

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        new.N = len(zs)
        new.cmps = range(new.N)
        
        new.HeatCapacityGases = self.HeatCapacityGases
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        return new
 
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
        elif 'Tcs' in eos_kwargs:
            self.N = N = len(eos_kwargs['Tcs'])
            self.cmps = range(self.N)
        
        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs
        self.Cpgs_locked, self.Cpgs_data = self.setup_Cpigs(HeatCapacityGases)
        
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
        new.Cpgs_data = self.Cpgs_data
        new.Cpgs_locked = self.Cpgs_locked
        
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
        
        
    def gammas(self):
        #         liquid.phis()/np.array([i.phi_l for i in liquid.eos_mix.pures()])
        phis = self.phis()
        phis_pure = []
        for i in self.eos_mix.pures():
            try:
                phis_pure.append(i.phi_g)
            except AttributeError:
                phis_pure.append(i.phi_l)
        return [phis[i]/phis_pure[i] for i in self.cmps]

    
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
        H = self.H_dep() 
        for zi, Cp_int in zip(self.zs, self.Cpig_integrals_pure()):
            H += zi*Cp_int
        self._H = H
        return H

    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        log_zs = self.log_zs()
        T, P, zs, cmps = self.T, self.P, self.zs, self.cmps
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)
        
        for i in cmps:
            S += zs[i]*Cpig_integrals_over_T_pure[i]
        S += self.S_dep()
        self._S = S
        return S
    
        

    def Cp(self):
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        for i in self.cmps:
            Cp += zs[i]*Cpigs_pure[i]
        return Cp + self.Cp_dep()

    def dH_dT(self):
        return self.Cp()

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
        Cpig_integrals_pure = self.Cpig_integrals_pure()
        self._dH_dzs = [dH_dep_dzs[i] + Cpig_integrals_pure[i] for i in self.cmps]
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
        integrals = self.Cpig_integrals_over_T_pure()

        try:
            dS_dep_dzs = self.eos_mix.dS_dep_dzs(eos_mix.Z_g, eos_mix.zs)
        except AttributeError:
            dS_dep_dzs = self.eos_mix.dS_dep_dzs(eos_mix.Z_l, eos_mix.zs)
        
        self._dS_dzs = [integrals[i] - R*(log_zs[i] + 1.0) + dS_dep_dzs[i] 
                        for i in cmps]
        return self._dS_dzs
 
            
def build_EOSLiquid():
    import inspect
    source = inspect.getsource(EOSGas)
    source = source.replace('EOSGas', 'EOSLiquid').replace('only_g', 'only_l')
    source = source.replace("'g'", "'gORIG'")
    source = source.replace("'l'", "'g'")
    source = source.replace("'gORIG'", "'l'")
    
    swap_strings = ('Cp_dep', 'd2P_dT2', 'd2P_dTdV', 'd2P_dV2', 'd2T_dV2', 'd2V_dT2', 'dH_dep_dP', 'dP_dT', 'dP_dV', 'phi', 'dS_dep_dP', 'dS_dep_dT', 'H_dep', 'S_dep', '.V', '.Z')
    for s in swap_strings:
        source = source.replace(s+'_g', 'gORIG')
        source = source.replace(s+'_l', s+'_g')
        source = source.replace('gORIG', s+'_l')
    return source

try:
    EOSLiquid
except:
    # Cost is ~10 ms - must be pasted in the future!
    exec(build_EOSLiquid())

class GibbsExcessLiquid(Phase):
    P_DEPENDENT_H_LIQ = True
    Psats_data = None
    Psats_locked = False
    Vms_sat_locked = False
    Vms_sat_data = None
    Hvap_locked = False
    Hvap_data = None
    use_IG_Cp = True
    def __init__(self, VaporPressures, VolumeLiquids=None, 
                 GibbsExcessModel=IdealSolution(), 
                 eos_pure_instances=None,
                 VolumeLiquidMixture=None,
                 HeatCapacityGases=None, 
                 EnthalpyVaporizations=None,
                 HeatCapacityLiquids=None, 
                 use_Poynting=False,
                 use_phis_sat=False,
                 use_IG_Cp=True,
                 Hfs=None, Gfs=None, Sfs=None,
                 henry_components=None, henry_data=None,
                 T=None, P=None, zs=None,
                 ):
        self.VaporPressures = VaporPressures
        self.Psats_locked = all(i.locked for i in VaporPressures) if VaporPressures is not None else False
        if self.Psats_locked:
            self.Psats_data = ([i.best_fit_Tmin for i in VaporPressures],
                               [i.best_fit_Tmin_slope for i in VaporPressures],
                               [i.best_fit_Tmin_value for i in VaporPressures],
                               [i.best_fit_Tmax for i in VaporPressures],
                               [i.best_fit_Tmax_slope for i in VaporPressures],
                               [i.best_fit_Tmax_value for i in VaporPressures],
                               [i.best_fit_coeffs for i in VaporPressures],
                               [i.best_fit_d_coeffs for i in VaporPressures],
                               [i.best_fit_d2_coeffs for i in VaporPressures])
            
        self.N = len(VaporPressures)
        self.cmps = range(self.N)
            
        self.HeatCapacityGases = HeatCapacityGases
        self.Cpgs_locked, self.Cpgs_data = self.setup_Cpigs(HeatCapacityGases)
        
        self.HeatCapacityLiquids = HeatCapacityLiquids
        if HeatCapacityLiquids is not None:
            self.Cpls_locked, self.Cpls_data = self.setup_Cpigs(HeatCapacityLiquids)
            T_REF_IG = self.T_REF_IG
            T_REF_IG_INV = 1.0/T_REF_IG
            self.Hvaps_T_ref = [obj(T_REF_IG) for obj in EnthalpyVaporizations]
            self.dSvaps_T_ref = [T_REF_IG_INV*dH for dH in self.Hvaps_T_ref]
            
            
        self.VolumeLiquids = VolumeLiquids
        self.Vms_sat_locked = all(i.locked for i in VolumeLiquids) if VolumeLiquids is not None else False
        if self.Vms_sat_locked:
            self.Vms_sat_data = ([i.best_fit_Tmin for i in VolumeLiquids],
                                 [i.best_fit_Tmin_slope for i in VolumeLiquids],
                                 [i.best_fit_Tmin_value for i in VolumeLiquids],
                                 [i.best_fit_Tmax for i in VolumeLiquids],
                                 [i.best_fit_Tmax_slope for i in VolumeLiquids],
                                 [i.best_fit_Tmax_value for i in VolumeLiquids],
                                 [i.best_fit_coeffs for i in VolumeLiquids],
                                 [i.best_fit_d_coeffs for i in VolumeLiquids],
                                 [i.best_fit_d2_coeffs for i in VolumeLiquids])
        
        self.EnthalpyVaporizations = EnthalpyVaporizations
        self.Hvap_locked = all(i.locked for i in EnthalpyVaporizations) if EnthalpyVaporizations is not None else False
        if self.Hvap_locked:
            self.Hvap_data = ([i.best_fit_Tmin for i in EnthalpyVaporizations],
                              [i.best_fit_Tmax for i in EnthalpyVaporizations],
                              [i.best_fit_Tc for i in EnthalpyVaporizations],
                              [1.0/i.best_fit_Tc for i in EnthalpyVaporizations],
                              [i.best_fit_coeffs for i in EnthalpyVaporizations])
        
        
        
        
        
        self.GibbsExcessModel = GibbsExcessModel
        self.eos_pure_instances = eos_pure_instances
        self.VolumeLiquidMixture = VolumeLiquidMixture
        
        
        self.use_Poynting = use_Poynting
        self.use_phis_sat = use_phis_sat
        
        if henry_components is None:
            henry_components = [False]*self.N
        self.has_henry_components = any(henry_components)
        self.henry_components = henry_components
        self.henry_data = henry_data
        
        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs

        if T is not None and P is not None and zs is not None:
            self.T = T
            self.P = P
            self.zs = zs
        
    def to_TP_zs(self, T, P, zs):
        T_equal = hasattr(self, 'T') and T == self.T
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
        
                
        new.Psats_locked = self.Psats_locked
        new.Psats_data = self.Psats_data
        
        new.Cpgs_locked = self.Cpgs_locked
        new.Cpgs_data = self.Cpgs_data
        
        new.Vms_sat_locked = self.Vms_sat_locked
        new.Vms_sat_data = self.Vms_sat_data
        
        new.Hvap_data = self.Hvap_data
        new.Hvap_locked = self.Hvap_locked

        
        new.use_phis_sat = self.use_phis_sat
        new.use_Poynting = self.use_Poynting
        new.P_DEPENDENT_H_LIQ = self.P_DEPENDENT_H_LIQ

        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        
        new.henry_data = self.henry_data
        new.henry_components = self.henry_components
        new.has_henry_components = self.has_henry_components
        
        
        if T_equal and self.zs is zs:
            new.GibbsExcessModel = self.GibbsExcessModel
        else:
            new.GibbsExcessModel = self.GibbsExcessModel.to_T_xs(T=T, xs=zs)
        
        
        try:
            if T_equal:
                try:
                    1/0 # zs for henry
                    new._Psats = self._Psats
                except:
                    pass
                


        except:
            pass
        return new
        
    def Henry_matrix(self):
        '''Generate a matrix of all component-solvent Henry's law values
        Shape N*N; solvent/solvent and gas/gas values are all None, as well
        as solvent/gas values where the parameters are unavailable.
        '''
        
    def Henry_constants(self):
        '''Mix the parameters in `Henry_matrix` into values to take the place
        in Psats.
        '''
        
    def Psats(self):
        try:
            return self._Psats
        except AttributeError:
            pass
        T, cmps = self.T, self.cmps
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
        # Need to reset the method because for the T bounded solver,
        # will normally get a different than prefered method as it starts
        # at the boundaries
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
                    
                    
        if self.has_henry_components:
            henry_components = self.henry_components
            henry_data = self.henry_data
            cmps = self.cmps
            zs = self.zs
            
            for i in cmps:
#                Vcs = [1, 1, 1]
                Vcs = [5.6000000000000006e-05, 0.000168, 7.340000000000001e-05]
                if henry_components[i]:
                    # WORKING - Need a bunch of conversions of data in terms of other values
                    # into this basis
                    d = henry_data[i]
                    z_sum = 0.0
                    logH = 0.0
                    for j in cmps:
                        if d[j]:
                            r = d[j]
                            t = T
#                            t = T - 273.15
                            log_Hi = (r[0] + r[1]/t + r[2]*log(t) + r[3]*t + r[4]/t**2)
#                            print(log_Hi)
                            wi = zs[j]*Vcs[j]**(2.0/3.0)/sum([zs[_]*Vcs[_]**(2.0/3.0) for _ in cmps if d[_]])
#                            print(wi)
                            
                            logH += wi*log_Hi
#                            logH += zs[j]*log_Hi
                            z_sum += zs[j]
                    
#                    print(logH, z_sum)
                    z_sum = 1
                    Psats[i] = exp(logH/z_sum)*1e5 # bar to Pa
                    
                
        return Psats

    
    
    def dPsats_dT(self):
        try:
            return self._dPsats_dT
        except:
            pass
        T, cmps = self.T, self.cmps
        # Need to reset the method because for the T bounded solver,
        # will normally get a different than prefered method as it starts
        # at the boundaries
        self._dPsats_dT = dPsats_dT = []
        if self.Psats_locked:
            Psats = self.Psats()
            Psats_data = self.Psats_data
            Tmins, Tmaxes, dcoeffs = Psats_data[0], Psats_data[3], Psats_data[7]
            for i in cmps:
                if T < Tmins[i]:
                    dPsat_dT = Psats_data[1][i]*Psats[i]#*exp((T - Tmins[i])*Psats_data[1][i]
                                                 #   + Psats_data[2][i])
                elif T > Tmaxes[i]:
                    dPsat_dT = Psats_data[4][i]*Psats[i]#*exp((T - Tmaxes[i])
                                                        #*Psats_data[4][i]
                                                        #+ Psats_data[5][i])
                else:
                    dPsat_dT = 0.0
                    for c in dcoeffs[i]:
                        dPsat_dT = dPsat_dT*T + c
#                    v, der = horner_and_der(coeffs[i], T)
                    dPsat_dT *= Psats[i]
                dPsats_dT.append(dPsat_dT)
            return dPsats_dT

        self._dPsats_dT = dPsats_dT = [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]
        return dPsats_dT

    def d2Psats_dT2(self):
        try:
            return self._d2Psats_dT2
        except:
            pass
        Psats = self.Psats()
        dPsats_dT = self.dPsats_dT()
        T, cmps = self.T, self.cmps

        self._d2Psats_dT2 = d2Psats_dT2 = []
        if self.Psats_locked:
            Psats_data = self.Psats_data
            Tmins, Tmaxes, d2coeffs = Psats_data[0], Psats_data[3], Psats_data[8]
            for i in cmps:
                d2Psat_dT2 = 0.0
                if Tmins[i] < T < Tmaxes[i]:
                    for c in d2coeffs[i]:
                        d2Psat_dT2 = d2Psat_dT2*T + c
                    d2Psat_dT2 = (dPsats_dT[i]*dPsats_dT[i]/Psats[i] + Psats[i]*d2Psat_dT2)
                d2Psats_dT2.append(d2Psat_dT2)
            return d2Psats_dT2

        self._d2Psats_dT2 = d2Psats_dT2 = [VaporPressure.T_dependent_property_derivative(T=T, n=2)
                     for VaporPressure in self.VaporPressures]
        return d2Psats_dT2


    def Vms_sat(self):
        try:
            return self._Vms_sat
        except AttributeError:
            pass
        T, cmps = self.T, self.cmps
        self._Vms_sat = Vms_sat = []
        if self.Vms_sat_locked:
            Vms_data = self.Vms_sat_data
            Tmins, Tmaxes, coeffs = Vms_data[0], Vms_data[3], Vms_data[6]
            for i in cmps:
                if T < Tmins[i]:
                    Vm = (T - Tmins[i])*Vms_data[1][i] + Vms_data[2][i]
                elif T > Tmaxes[i]:
                    Vm = (T - Tmaxes[i])*Vms_data[4][i] + Vms_data[5][i]
                else:
                    Vm = 0.0
                    for c in coeffs[i]:
                        Vm = Vm*T + c
#                    Vm = horner(coeffs[i], T)
                Vms_sat.append(Vm)
            return Vms_sat
        
#        Psats = self.Psats() # T, Psats[i]
        VolumeLiquids = self.VolumeLiquids
        self._Vms_sat = [VolumeLiquids[i].T_dependent_property(T) for i in cmps]
        return self._Vms_sat

    def dVms_sat_dT(self):
        try:
            return self._Vms_sat_dT
        except:
            pass
        T, cmps = self.T, self.cmps
        self._Vms_sat_dT = Vms_sat_dT = []
        if self.Vms_sat_locked:
            Vms_data = self.Vms_sat_data
            Tmins, Tmaxes, dcoeffs = Vms_data[0], Vms_data[3], Vms_data[7]
            for i in cmps:
                if T < Tmins[i]:
                    dVm = Vms_data[1][i]
                elif T > Tmaxes[i]:
                    dVm = Vms_data[4][i]
                else:
                    dVm = 0.0
                    for c in dcoeffs[i]:
                        dVm = dVm*T + c
                Vms_sat_dT.append(dVm)
            return Vms_sat_dT

        VolumeLiquids = self.VolumeLiquids
        self._Vms_sat_dT = Vms_sat_dT = [obj.T_dependent_property_derivative(T=T) for obj in VolumeLiquids]
        return Vms_sat_dT

    def d2Vms_sat_dT2(self):
        try:
            return self._d2Vms_sat_dT2
        except:
            pass

        T, cmps = self.T, self.cmps
        self._d2Vms_sat_dT2 = d2Vms_sat_dT2 = []
        
        if self.Vms_sat_locked:
            Vms_data = self.Vms_sat_data
            Tmins, Tmaxes, d2coeffs = Vms_data[0], Vms_data[3], Vms_data[8]
            for i in cmps:
                d2Vm = 0.0
                if Tmins[i] < T < Tmaxes[i]:
                    for c in d2coeffs[i]:
                        d2Vm = d2Vm*T + c
                d2Vms_sat_dT2.append(d2Vm)
            return d2Vms_sat_dT2

        VolumeLiquids = self.VolumeLiquids
        self._d2Vms_sat_dT2 = [obj.T_dependent_property_derivative(T=T, order=2) for obj in VolumeLiquids]
        return self._d2Vms_sat_dT2

    def Hvaps(self):
        try:
            return self._Hvaps
        except AttributeError:
            pass
        T, EnthalpyVaporizations, cmps = self.T, self.EnthalpyVaporizations, self.cmps

        self._Hvaps = Hvaps = []
        if self.Hvap_locked and 0:
            Hvap_data = self.Hvap_data
            Tmins, Tmaxes, Tcs, Tcs_inv, coeffs = Hvap_data[0], Hvap_data[1], Hvap_data[2], Hvap_data[3], Hvap_data[4]
            for i in cmps:
                Hvap = 0.0
                if T < Tcs[i]:
                    x = log(1.0 - T*Tcs_inv[i])
                    for c in coeffs[i]:
                        Hvap = Hvap*x + c
    #                    Vm = horner(coeffs[i], log(1.0 - T*Tcs_inv[i])
                Hvaps.append(Hvap)
            return Hvaps
        
        self._Hvaps = Hvaps = [EnthalpyVaporizations[i](T) for i in cmps] 
        for i in cmps:
            if Hvaps[i] is None:
                Hvaps[i] = 0.0
        return Hvaps

    def dHvaps_dT(self):
        try:
            return self._dHvaps_dT
        except AttributeError:
            pass
        T, EnthalpyVaporizations, cmps = self.T, self.EnthalpyVaporizations, self.cmps

        self._dHvaps_dT = dHvaps_dT = []
        if self.Hvap_locked:
            Hvap_data = self.Hvap_data
            Tmins, Tmaxes, Tcs, Tcs_inv, coeffs = Hvap_data[0], Hvap_data[1], Hvap_data[2], Hvap_data[3], Hvap_data[4]
            for i in cmps:
                dHvap_dT = 0.0
                if T < Tcs[i]:
                    p = log((Tcs[i] - T)*Tcs_inv[i])
                    x = 1.0
                    a = 1.0
                    for c in coeffs[i][-2::-1]:
                        dHvap_dT += a*c*x
                        x *= p
                        a += 1.0
                    dHvap_dT /= T - Tcs[i]

                dHvaps_dT.append(dHvap_dT)
            return dHvaps_dT
        
        self._dHvaps_dT = dHvaps_dT = [EnthalpyVaporizations[i].T_dependent_property_derivative(T) for i in cmps] 
        for i in cmps:
            if dHvaps_dT[i] is None:
                dHvaps_dT[i] = 0.0
        return dHvaps_dT

    def Poyntings(self):
        try:
            return self._Poyntings
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._Poyntings = [1.0]*self.N
            return self._Poyntings
        
        T, P = self.T, self.P
        Psats = self.Psats()
        Vmls = self.Vms_sat()
#        Vmls = [VolumeLiquid.T_dependent_property(T=T) for VolumeLiquid in self.VolumeLiquids]        
#        Vmls = [VolumeLiquid(T=T, P=P) for VolumeLiquid in self.VolumeLiquids]
        self._Poyntings = [exp(Vml*(P-Psat)/(R*T)) for Psat, Vml in zip(Psats, Vmls)]
        return self._Poyntings
    
    def dPoyntings_dT(self):
        try:
            return self._dPoyntings_dT
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._dPoyntings_dT = [0.0]*self.N
            return self._dPoyntings_dT
        
        Psats = self.Psats()
        T, P = self.T, self.P
            
        dPsats_dT = self.dPsats_dT()
#        dPsats_dT = [VaporPressure.T_dependent_property_derivative(T=T)
#                     for VaporPressure in self.VaporPressures]

        Vmls = self.Vms_sat()
#        Vmls = [VolumeLiquid.T_dependent_property(T=T) for VolumeLiquid in self.VolumeLiquids]                    
#        dVml_dTs = [VolumeLiquid.T_dependent_property_derivative(T=T) 
#                    for VolumeLiquid in self.VolumeLiquids]
        dVml_dTs = self.dVms_sat_dT()
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
            self._dPoyntings_dP = [0.0]*self.N
            return self._dPoyntings_dP
        T, P = self.T, self.P
        Psats = self.Psats()
        
        Vmls = self.Vms_sat()
#        Vmls = [VolumeLiquid(T=T, P=P) for VolumeLiquid in self.VolumeLiquids]
        
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
            self._phis_sat = [1.0]*self.N
            return self._phis_sat
        
        T = self.T
        self._phis_sat = [i.phi_sat(T) for i in self.eos_pure_instances]
        return self._phis_sat
                
    def dphis_sat_dT(self):
        # Not actually implemented
        try:
            return self._dphis_sat_dT
        except AttributeError:
            pass

        if not self._dphis_sat_dT:
            self._dphis_sat_dT = [0.0]*self.N
            return self._dphis_sat_dT

        T = self.T
        # Not implemented
        self._dphis_sat_dT = [i.dphi_sat_dT(T) for i in self.eos_pure_instances]
        return self._dphis_sat_dT


    def phis(self):
        r'''Method to calculate the fugacity coefficients of the
        GibbsExcessLiquid phase. Depending on the settings of the phase, can
        include the effects of activity coefficients `gammas`, pressure
        correction terms `Poyntings`, and pure component saturation fugacities
        `phis_sat` as well as the pure component vapor pressures.
        
        .. math::
            \phi_i = \frac{\gamma_i P_{i}^{sat} \phi_i^{sat} \text{Poynting}_i}
            {P}

        Returns
        -------
        phis : list[float]
            Fugacity coefficients of all components in the phase, [-]
            
        Notes
        -----
        Poyntings, gammas, and pure component saturation phis default to 1.
        '''
        try:
            return self._phis
        except AttributeError:
            pass
        P = self.P
        try:
            gammas = self._gammas
        except AttributeError:
            gammas = self.gammas()
            
        try:
            Psats = self._Psats
        except AttributeError:
            Psats = self.Psats()
        
        try:
             phis_sat = self._phis_sat
        except AttributeError:
            phis_sat = self.phis_sat()

        try:
            Poyntings = self._Poyntings
        except AttributeError:
            Poyntings = self.Poyntings()
            
        P_inv = 1.0/P
        self._phis = [gammas[i]*Psats[i]*Poyntings[i]*phis_sat[i]*P_inv
                for i in self.cmps]
        return self._phis
        
        
    def lnphis(self):
        try:
            return self._lnphis
        except AttributeError:
            pass
        self._lnphis = [log(i) for i in self.phis()]        
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
            dPoyntings_dT = self.dPoyntings_dT() #[0.0]*self.N#self.dPoyntings_dT(T, P, Psats=Psats)
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
        phis = self.phis()
        self._dlnphis_dT = [i/j for i, j in zip(dphis_dT, phis)]
        return self._dlnphis_dT

    def dlnphis_dP(self):
        r'''Method to calculate the pressure derivative of log fugacity 
        coefficients of the phase. Depending on the settings of the phase, can
        include the effects of activity coefficients `gammas`, pressure
        correction terms `Poyntings`, and pure component saturation fugacities
        `phis_sat` as well as the pure component vapor pressures.
        
        .. math::
            \frac{\partial \log \phi_i}{\partial P} = 
            \frac{\frac{\partial \text{Poynting}_i}{\partial P}}
            {\text{Poynting}_i} - \frac{1}{P}

        Returns
        -------
        dlnphis_dP : list[float]
            Pressure derivative of log fugacity coefficients of all components
            in the phase, [1/Pa]
            
        Notes
        -----
        Poyntings, gammas, and pure component saturation phis default to 1. For
        that case, :math:`\frac{\partial \log \phi_i}{\partial P}=\frac{1}{P}`.
        '''
        try:
            return self._dlnphis_dP
        except AttributeError:
            pass
        try:
            Poyntings = self._Poyntings
        except AttributeError:
            Poyntings = self.Poyntings()
            
        try:
            dPoyntings_dP = self._dPoyntings_dP
        except AttributeError:
            dPoyntings_dP = self.dPoyntings_dP()
            
        P_inv = 1.0/self.P
        
        self._dlnphis_dP = [dPoyntings_dP[i]/Poyntings[i] - P_inv for i in self.cmps]
        return self._dlnphis_dP
                    
    

    def gammas(self):
        try:
            return self.GibbsExcessModel._gammas
        except AttributeError:
            return self.GibbsExcessModel.gammas()
    
    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        # Untested
        H = 0
        T = self.T
        P = self.P
        zs = self.zs
        T_REF_IG = self.T_REF_IG
        P_DEPENDENT_H_LIQ = self.P_DEPENDENT_H_LIQ

        try:
            Cpig_integrals_pure = self._Cpig_integrals_pure
        except AttributeError:
            Cpig_integrals_pure = self.Cpig_integrals_pure()
                    
        H = 0.0
        
        if P_DEPENDENT_H_LIQ:
            # Page 650  Chemical Thermodynamics for Process Simulation
            # Confirmed with CoolProp via analytical integrals
            # Not actually checked numerically until Hvap is implemented though
            '''
            from scipy.integrate import *
            from CoolProp.CoolProp import PropsSI
            
            fluid = 'decane'
            T = 400
            Psat = PropsSI('P', 'T', T, 'Q', 0, fluid)
            P2 = Psat*100
            dP = P2 - Psat
            Vm = 1/PropsSI('DMOLAR', 'T', T, 'Q', 0, fluid)
            Vm2 = 1/PropsSI('DMOLAR', 'T', T, 'P', P2, fluid)
            dH = PropsSI('HMOLAR', 'T', T, 'P', P2, fluid) - PropsSI('HMOLAR', 'T', T, 'Q', 0, fluid)
            
            def to_int(P):
                Vm = 1/PropsSI('DMOLAR', 'T', T, 'P', P, fluid)
                alpha = PropsSI('ISOBARIC_EXPANSION_COEFFICIENT', 'T', T, 'P', P, fluid)
                return Vm -alpha*T*Vm 
            quad(to_int, Psat, P2, epsabs=1.49e-14, epsrel=1.49e-14)[0]/dH            
            '''
            Psats = self.Psats()
            Vms_sat = self.Vms_sat()
            dVms_sat_dT = self.dVms_sat_dT()
#            for i in self.cmps:
#                H += zs[i]*(Cpig_integrals_pure[i] - Hvaps[i])
#                dP = P - Psats[i]
##                if dP > 0.0:
#                    # Could ignore the pressure contribution for sub-saturation pure fluids
#                H += zs[i]*dP*(Vms_sat[i] - T*dVms_sat_dT[i])
            # Trying the DTU formulation:
            dPsats_dT = self.dPsats_dT()
            H = 0.0
            for i in self.cmps:
                dV_vap = R*T/Psats[i] - Vms_sat[i]
                dS_vap = dPsats_dT[i]*dV_vap
                Hvap = T*dS_vap
#                Hvap = Hvaps[i]
#                H += zs[i]*(Cpig_integrals_pure[i]) # If we can use the liquid heat capacity and prove its consistency, enthalpy is easy
                H += zs[i]*(Cpig_integrals_pure[i] - Hvap)
                dP = P - Psats[i]
                H += zs[i]*dP*(Vms_sat[i] - T*dVms_sat_dT[i])
        else:
            Hvaps = self.Hvaps()
            for i in self.cmps:
                H += zs[i]*(Cpig_integrals_pure[i] - Hvaps[i]) 
        H += self.GibbsExcessModel.HE()
        self._H = H
        return H
            
    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        # Untested
        # Page 650  Chemical Thermodynamics for Process Simulation
        '''
        from scipy.integrate import *
        from CoolProp.CoolProp import PropsSI
        
        fluid = 'decane'
        T = 400
        Psat = PropsSI('P', 'T', T, 'Q', 0, fluid)
        P2 = Psat*100
        dP = P2 - Psat
        Vm = 1/PropsSI('DMOLAR', 'T', T, 'Q', 0, fluid)
        Vm2 = 1/PropsSI('DMOLAR', 'T', T, 'P', P2, fluid)
        dH = PropsSI('HMOLAR', 'T', T, 'P', P2, fluid) - PropsSI('HMOLAR', 'T', T, 'Q', 0, fluid)
        dS = PropsSI('SMOLAR', 'T', T, 'P', P2, fluid) - PropsSI('SMOLAR', 'T', T, 'Q', 0, fluid)
        def to_int2(P):
            Vm = 1/PropsSI('DMOLAR', 'T', T, 'P', P, fluid)
            alpha = PropsSI('ISOBARIC_EXPANSION_COEFFICIENT', 'T', T, 'P', P, fluid)
            return -alpha*Vm 
        quad(to_int2, Psat, P2, epsabs=1.49e-14, epsrel=1.49e-14)[0]/dS
        '''
        S = 0.0
        T, P, zs, cmps = self.T, self.P, self.zs, self.cmps
        log_zs = self.log_zs()
        for i in cmps:
#            S -= log_zs[i]
            S -= zs[i]*log_zs[i]
        S *= R
        
        T_inv = 1.0/T
        
        P_REF_IG_INV = self.P_REF_IG_INV
        
        Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        Psats = self.Psats()
        Vms_sat = self.Vms_sat()
        dPsats_dT = self.dPsats_dT()
        if self.P_DEPENDENT_H_LIQ:
            dVms_sat_dT = self.dVms_sat_dT()
            # Holy - actually consistent! Do NOT CHANGE ANYTHING
            for i in self.cmps:
                dSi = Cpig_integrals_over_T_pure[i] 
                dVsat =  R*T/Psats[i] - Vms_sat[i]
                dSvap = dPsats_dT[i]*dVsat
#                dSvap = Hvaps[i]/T # Confirmed - this line breaks everything - do not use
                
                dSi -= dSvap

#                dSi = Cpig_integrals_over_T_pure[i] - Hvaps[i]*T_inv # Do the transition at the temperature of the liquid
                # Take each component to its reference state change - saturation pressure
#                dSi -= R*log(P*P_REF_IG_INV)
                dSi -= R*log(Psats[i]*P_REF_IG_INV)
#                dSi -= R*log(P/101325.0)
                # Only include the
                dP = P - Psats[i]
#                if dP > 0.0:
                # I believe should include effect of pressure on all components, regardless of phase
                dSi -= dP*dVms_sat_dT[i]
                S += dSi*zs[i]
                
                
#            # Presented in CoCo - easily checked as correct in EOS model
#            S2 = 0.0
#            S2 -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
#            S2 -= R*log(P*P_REF_IG_INV) # This one specifically uses P, not Psat
#            
#            for i in cmps:
#                S2 += zs[i]*Cpig_integrals_over_T_pure[i]
#            
#            lnphis = self.lnphis()
#            dlnphis_dT = self.dlnphis_dT()
#            for i in cmps:
#                S2 -= zs[i]*(R*lnphis[i] + R*T*dlnphis_dT[i])
#            print(S, S2, S/S2)                
        else:
            Hvaps = self.Hvaps()
            for i in cmps:
                Sg298_to_T = Cpig_integrals_over_T_pure[i]
                Svap = -Hvaps[i]*T_inv # Do the transition at the temperature of the liquid
                S += zs[i]*(Sg298_to_T + Svap - R*log(P*P_REF_IG_INV)) # 
        self._S = S + self.GibbsExcessModel.SE()
        return S

    def Cp(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        # Needs testing
        T, P, P_DEPENDENT_H_LIQ = self.T, self.P, self.P_DEPENDENT_H_LIQ
        Cp, zs = 0.0, self.zs
        dHvaps_dT = self.dHvaps_dT()
        Cpigs_pure = self.Cpigs_pure()
        if P_DEPENDENT_H_LIQ:
            d2Vms_sat_dT2 = self.d2Vms_sat_dT2()
            dVms_sat_dT = self.dVms_sat_dT()
            Vms_sat = self.Vms_sat()
            Psats = self.Psats()
            dPsats_dT = self.dPsats_dT()
            for i in self.cmps:
                Cp += zs[i]*(Cpigs_pure[i] - dHvaps_dT[i])
                Cp += zs[i]*(-T*(P - Psats[i])*d2Vms_sat_dT2[i] + (T*dVms_sat_dT[i] - Vms_sat[i])*dPsats_dT[i])

        else:
            for i in self.cmps:
                Cp += zs[i]*(Cpigs_pure[i] - dHvaps_dT[i])
            
        Cp += self.GibbsExcessModel.CpE()
        self._Cp = Cp
        return Cp

    
    ### Volumetric properties
    def V(self):
        try:
            return self._V
        except AttributeError:
            pass
        zs = self.zs
        Vms = self.Vms_sat()
        '''To make a fugacity-volume identity consistent, cannot use pressure
        correction unless the Poynting factor is calculated with quadrature/
        integration.
        '''
        self._V = sum([zs[i]*Vms[i] for i in self.cmps])
#        self._V = self.VolumeLiquidMixture(self.T, self.P, self.zs)
        return self._V

    # Main needed volume derivatives
    def dP_dV(self):
        try:
            return self._dP_dV
        except AttributeError:
            pass
        self._dP_dV = 1.0/self.VolumeLiquidMixture.property_derivative_P(self.T, self.P, self.zs, order=1)
        return self._dP_dV
    
    def d2P_dV2(self):
        try:
            return self._d2P_dV2
        except AttributeError:
            pass
        self._d2P_dV2 = self.d2V_dP2()/-(self.dP_dV())**-3
        return self._d2P_dV2
    
    def dP_dT(self):
        try:
            return self._dP_dT
        except AttributeError:
            pass
        self._dP_dT = self.dV_dT()/-self.dP_dV()
        return self._dP_dT
    
    def d2P_dTdV(self):
        try:
            return self._d2P_dTdV
        except AttributeError:
            pass
        P = self.P
        def dP_dV_for_diff(T):
            return 1.0/self.VolumeLiquidMixture.property_derivative_P(T, P, self.zs, order=1)

        self._d2P_dTdV = derivative(dP_dV_for_diff, self.T)
        return self._d2P_dTdV

    def d2P_dT2(self):
        try:
            return self._d2P_dT2
        except AttributeError:
            pass
        P, zs = self.P, self.zs
        def dP_dT_for_diff(T):
            dV_dT = self.VolumeLiquidMixture.property_derivative_T(T, P, zs, order=1)
            dP_dV = 1.0/self.VolumeLiquidMixture.property_derivative_P(T, P, zs, order=1)
            dP_dT = dV_dT/-dP_dV
            return dP_dT
        
        self._d2P_dT2 = derivative(dP_dT_for_diff, self.T)
        return self._d2P_dT2

    # Volume derivatives which needed to be implemented for the main ones
    def d2V_dP2(self):
        try:
            return self._d2V_dP2
        except AttributeError:
            pass
        self._d2V_dP2 = self.VolumeLiquidMixture.property_derivative_P(self.T, self.P, self.zs, order=2)
        return self._d2V_dP2

    def dV_dT(self):
        try:
            return self._dV_dT
        except AttributeError:
            pass
        self._dV_dT = self.VolumeLiquidMixture.property_derivative_T(self.T, self.P, self.zs, order=1)
        return self._dV_dT
    
    
class GibbsExcessSolid(GibbsExcessLiquid):
    def __init__(self, SublimationPressures, VolumeSolids=None, 
                 GibbsExcessModel=IdealSolution(), 
                 eos_pure_instances=None,
                 VolumeLiquidMixture=None,
                 HeatCapacityGases=None, 
                 EnthalpySublimations=None,
                 use_Poynting=False,
                 use_phis_sat=False,
                 Hfs=None, Gfs=None, Sfs=None,
                 henry_components=None, henry_data=None,
                 T=None, P=None, zs=None,
                 ):
        super(self, VaporPressures=SublimationPressures, VolumeLiquids=VolumeSolids,
              HeatCapacityGases=HeatCapacityGases, EnthalpyVaporizations=EnthalpySublimations,
              use_Poynting=use_Poynting,
              Hfs=Hfs, Gfs=Gfs, Sfs=Sfs, T=T, P=P, zs=zs)


gas_phases = (IdealGas, EOSGas)
liquid_phases = (EOSLiquid, GibbsExcessLiquid)
solid_phases = (GibbsExcessSolid,)
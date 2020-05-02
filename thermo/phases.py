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
           'gas_phases', 'liquid_phases', 'solid_phases', 'CombinedPhase', 'CoolPropPhase', 'CoolPropLiquid', 'CoolPropGas', 'INCOMPRESSIBLE_CONST']

import sys
from math import isinf, isnan
from fluids.constants import R, R_inv
from fluids.numerics import (horner, horner_and_der, horner_and_der2, horner_log, jacobian, derivative,
                             best_fit_integral_value, best_fit_integral_over_T_value,
                             evaluate_linear_fits, evaluate_linear_fits_d,
                             evaluate_linear_fits_d2, quadratic_from_f_ders,
                             newton_system, trunc_log, trunc_exp)
from thermo.utils import (log, log10, exp, Cp_minus_Cv, phase_identification_parameter,
                          isothermal_compressibility, isobaric_expansion,
                          Joule_Thomson, speed_of_sound, dxs_to_dns, dns_to_dn_partials,
                          normalize, hash_any_primitive)
from thermo.activity import IdealSolution
from thermo.coolprop import has_CoolProp, CP as CoolProp
from thermo.eos_mix import IGMIX
from random import randint
from scipy.optimize import fsolve
from collections import OrderedDict
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

#PY2 = int(sys.version[0]) == 2

SORTED_DICT = sys.version_info >= (3, 6)
INCOMPRESSIBLE_CONST = 1e30


class Phase(object):
    T_REF_IG = 298.15
    T_REF_IG_INV = 1.0/T_REF_IG
    P_REF_IG = 101325.
    P_REF_IG_INV = 1.0/P_REF_IG
    LOG_P_REF_IG = log(P_REF_IG)
    
    T_MAX_FIXED = 10000.0
    T_MIN_FIXED = 1e-3
    
    P_MAX_FIXED = 1e9
    P_MIN_FIXED = 1e-2 # 1e-3 was so low issues happened in the root stuff, could not be fixed
    
    V_MIN_FIXED = 1e-9 # m^3/mol
    V_MAX_FIXED = 1e9 # m^#/mol
    
    force_phase = None

    Psats_data = None
    Cpgs_data = None
    Psats_locked = False 
    Cpgs_locked = False
    composition_independent = False
    
    def __repr__(self):
        s =  '<%s, ' %(self.__class__.__name__)
        try:
            s += 'T=%g K, P=%g Pa' %(self.T, self.P)
        except:
            pass
        s += '>'
        return s
    
    def model_hash(self, ignore_phase=False):
        return randint(0, 10000000)
    
    def value(self, name):
        if name in ('beta_mass',):
            return self.result.value(name, self)
        
        v = getattr(self, name)
        try:
            v = v()
        except:
            pass
        return v
    
    def S_phi_consistency(self):
        # From coco
        S0 = self.S_ideal_gas()
        lnphis = self.lnphis()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        for i in range(len(zs)):
            S0 -= zs[i]*(R*lnphis[i] + R*T*dlnphis_dT[i])
        return abs(1.0 - S0/self.S())
    

    def H_phi_consistency(self):
        return abs(1.0 - self.H_from_phi()/self.H())
    
    def G_phi_consistency(self):
        # Chapter 2 equation 31 Michaelson
        zs, T = self.zs, self.T
        G_dep_RT = 0.0
        lnphis = self.lnphis()
        G_dep_RT = sum(zs[i]*lnphis[i] for i in self.cmps)
        G_dep = G_dep_RT*R*T
        return abs(1.0 - G_dep/self.G_dep())
    
    def H_dep_phi_consistency(self):
        H_dep_RT2 = 0.0
        dlnphis_dTs = self.dlnphis_dT()
        zs, T = self.zs, self.T
        H_dep_RT2 = sum(zs[i]*dlnphis_dTs[i] for i in range(len(zs)))
        H_dep_recalc = -H_dep_RT2*R*T*T
        H_dep = self.H_dep()
        return abs(1.0 - H_dep/H_dep_recalc)

    def H_from_phi(self):
        H0 = self.H_ideal_gas()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        for i in self.cmps:
            H0 -= R*T*T*zs[i]*dlnphis_dT[i]
        return H0

    def S_from_phi(self):
        S0 = self.S_ideal_gas()
        lnphis = self.lnphis()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        for i in self.cmps:
            S0 -= zs[i]*(R*lnphis[i] + R*T*dlnphis_dT[i])
        return S0

    def V_phi_consistency(self):
        zs, P = self.zs, self.P
        dlnphis_dP = self.dlnphis_dP()
        obj = sum(zs[i]*dlnphis_dP[i] for i in self.cmps)
        base = (self.Z() - 1.0)/P
        return abs(1.0 - obj/base)
    
    def V_from_phi(self):
        zs, P = self.zs, self.P
        dlnphis_dP = self.dlnphis_dP()
        obj = sum(zs[i]*dlnphis_dP[i] for i in self.cmps)
        Z = P*obj + 1.0
        return Z*R*self.T/P

    def lnphi(self):
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.lnphis()[0]
    
    def phi(self):
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.phis()[0]
    
    def fugacity(self):
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.fugacities()[0]
    
    def dfugacity_dT(self):
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.dfugacities_dT()[0]
    
    def dfugacity_dP(self):
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.dfugacities_dP()[0]
    

    def fugacities(self):
        P = self.P
        zs = self.zs
        lnphis = self.lnphis()
        return [P*zs[i]*trunc_exp(lnphis[i]) for i in range(len(zs))]
    
    def lnfugacities(self):
        return [log(i) for i in self.fugacities()]
    
    fugacities_lowest_Gibbs = fugacities
    
    def dfugacities_dT(self):
        r'''
        '''
        dphis_dT = self.dphis_dT()
        P, zs = self.P, self.zs
        return [P*zs[i]*dphis_dT[i] for i in range(len(zs))]
    
    def lnphis_G_min(self):
        return self.lnphis()

    def phis(self):
        return [trunc_exp(i) for i in self.lnphis()]

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

    def dfugacities_dns(self):
        phis = self.phis()
        dlnphis_dns = self.dlnphis_dns()
        
        P, zs, cmps = self.P, self.zs, self.cmps
        matrix = []
        for i in cmps:
            phi_P = P*phis[i]
            ziPphi = phi_P*zs[i]
            r = dlnphis_dns[i]
            row = [ziPphi*(r[j] - 1.0) for j in cmps]
            row[i] += phi_P
            matrix.append(row)
        return matrix

    def dlnfugacities_dns(self):
        zs, cmps = self.zs, self.cmps
        fugacities = self.fugacities()
        dlnfugacities_dns = [list(i) for i in self.dfugacities_dns()]
        fugacities_inv = [1.0/fi for fi in fugacities]
        for i in cmps:
            r = dlnfugacities_dns[i]
            for j in cmps:
                r[j]*= fugacities_inv[i]
        return dlnfugacities_dns

    def dlnfugacities_dzs(self):
        zs, cmps = self.zs, self.cmps
        fugacities = self.fugacities()
        dlnfugacities_dzs = [list(i) for i in self.dfugacities_dzs()]
        fugacities_inv = [1.0/fi for fi in fugacities]
        for i in cmps:
            r = dlnfugacities_dzs[i]
            for j in cmps:
                r[j]*= fugacities_inv[i]
        return dlnfugacities_dzs



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
    
    def V_iter(self, force=False):
        return self.V()

    def G(self):
        try:
            return self._G
        except AttributeError:
            pass
        G = self.H() - self.T*self.S()
        self._G = G
        return G
    
    G_min = G
    
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
    
    dG_dT_P = dG_dT

    def dG_dT_V(self):
        return -self.T*self.dS_dT_V() - self.S() + self.dH_dT_V()
    
    def dG_dP(self):
        return -self.T*self.dS_dP() + self.dH_dP()

    dG_dP_T = dG_dP

    def dG_dP_V(self):
        return -self.T*self.dS_dP_V() - self.dT_dP()*self.S() + self.dH_dP_V()
    
    def dG_dV_T(self):
        return self.dG_dP_T()*self.dP_dV()

    def dG_dV_P(self):
        return self.dG_dT_P()*self.dT_dV()


    def dU_dT(self):
        # Correct
        return -self.P*self.dV_dT() + self.dH_dT()
    
    dU_dT_P = dU_dT
    
    def dU_dT_V(self):
        return self.dH_dT_V() - self.V()*self.dP_dT()
    
    def dU_dP(self):
        return -self.P*self.dV_dP() - self.V() + self.dH_dP()
    
    dU_dP_T = dU_dP
    
    def dU_dP_V(self):
        return self.dH_dP_V() - self.V()
    
    def dU_dV_T(self):
        return self.dU_dP_T()*self.dP_dV()

    def dU_dV_P(self):
        return self.dU_dT_P()*self.dT_dV()
    
    def dA_dT(self):
        return -self.T*self.dS_dT() - self.S() + self.dU_dT()
    
    dA_dT_P = dA_dT
    
    def dA_dT_V(self):
        return (self.dH_dT_V() - self.V()*self.dP_dT() - self.T*self.dS_dT_V()
                - self.S())

    def dA_dP(self):
        return -self.T*self.dS_dP() + self.dU_dP()
    
    dA_dP_T = dA_dP
    
    def dA_dP_V(self):
        return (self.dH_dP_V() - self.V() - self.dT_dP()*self.S() 
                - self.T*self.dS_dP_V())

    def dA_dV_T(self):
        return self.dA_dP_T()*self.dP_dV()

    def dA_dV_P(self):
        return self.dA_dT_P()*self.dT_dV()
    


    
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
        try:
            return self._H_reactive
        except AttributeError:
            pass
        H = self.H()
        for zi, Hf in zip(self.zs, self.Hfs):
            H += zi*Hf
        self._H_reactive = H
        return H

    def S_reactive(self):
        try:
            return self._S_reactive
        except:
            pass
        S = self.S()
        for zi, Sf in zip(self.zs, self.Sfs):
            S += zi*Sf
        self._S_reactive = S
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

    def H_formation_ideal_gas(self):
        try:
            return self._H_formation_ideal_gas
        except AttributeError:
            pass
        Hf_ideal_gas = 0.0
        for zi, Hf in zip(self.zs, self.Hfs):
            Hf_ideal_gas += zi*Hf
        self._H_formation_ideal_gas = Hf_ideal_gas
        return Hf_ideal_gas

    def S_formation_ideal_gas(self):
        try:
            return self._S_formation_ideal_gas
        except:
            pass
        Sf_ideal_gas = 0.0
        for zi, Sf in zip(self.zs, self.Sfs):
            Sf_ideal_gas += zi*Sf
        self._S_formation_ideal_gas = Sf_ideal_gas
        return Sf_ideal_gas
    
    def G_formation_ideal_gas(self):
        Gf = self.H_formation_ideal_gas() - self.T_REF_IG*self.S_formation_ideal_gas()
        return Gf
    
    def U_formation_ideal_gas(self):
        Uf = self.H_formation_ideal_gas() - self.P_REF_IG*self.V_ideal_gas()
        return Uf
    
    def A_formation_ideal_gas(self):
        Af = self.U_formation_ideal_gas() - self.T_REF_IG*self.S_formation_ideal_gas()
        return Af
    
    def Cv(self):
        try:
            return self._Cv
        except AttributeError:
            pass
        # checks out
        Cp_m_Cv = Cp_minus_Cv(self.T, self.dP_dT(), self.dP_dV())
        Cp = self.Cp()
        self._Cv = Cv = Cp - Cp_m_Cv
        return Cv
    
    def dCv_dT_P(self):
        r'''Method to calculate the temperature derivative of Cv, constant
        volume heat capacity, at constant pressure.
        
        .. math::
            \left(\frac{\partial C_v}{\partial T}\right)_P = 
            - \frac{T \operatorname{dPdT_{V}}^{2}{\left(T \right)} \frac{d}{dT}
            \operatorname{dPdV_{T}}{\left(T \right)}}{\operatorname{dPdV_{T}}^{2}
            {\left(T \right)}} + \frac{2 T \operatorname{dPdT_{V}}{\left(T \right)}
            \frac{d}{d T} \operatorname{dPdT_{V}}{\left(T \right)}}
            {\operatorname{dPdV_{T}}{\left(T \right)}} + \frac{\operatorname{
            dPdT_{V}}^{2}{\left(T \right)}}{\operatorname{dPdV_{T}}{\left(T 
            \right)}} + \frac{d}{d T} \operatorname{Cp}{\left(T \right)}

        Returns
        -------
        dCv_dT_P : float
           Temperature derivative of constant volume heat capacity at constant 
           pressure, [J/mol/K^2]
            
        Notes
        -----
        Requires `d2P_dT2_PV`, `d2P_dVdT_TP`, and `d2H_dT2`.
        '''
        T = self.T
        x0 = self.dP_dT_V()
        x1 = x0*x0
        x2 = self.dP_dV_T()
        x3 = 1.0/x2
        
        x50 = self.d2P_dT2_PV()
        x51 = self.d2P_dVdT_TP()
        x52 = self.d2H_dT2()
        return 2.0*T*x0*x3*x50 - T*x1*x51*x3*x3 + x1*x3 + x52

    def dCv_dP_T(self):
        r'''Method to calculate the pressure derivative of Cv, constant
        volume heat capacity, at constant temperature.
        
        .. math::
            \left(\frac{\partial C_v}{\partial P}\right)_T = 
            - T \operatorname{dPdT_{V}}{\left(P \right)} \frac{d}{d P}
            \operatorname{dVdT_{P}}{\left(P \right)} - T \operatorname{
            dVdT_{P}}{\left(P \right)} \frac{d}{d P} \operatorname{dPdT_{V}}
            {\left(P \right)} + \frac{d}{d P} \operatorname{Cp}{\left(P\right)}
            
        Returns
        -------
        dCv_dP_T : float
           Pressure derivative of constant volume heat capacity at constant 
           temperature, [J/mol/K/Pa]
            
        Notes
        -----
        Requires `d2V_dTdP`, `d2P_dTdP`, and `d2H_dTdP`.
        '''
        T = self.T
        dP_dT_V = self.dP_dT_V()
        d2V_dTdP = self.d2V_dTdP()
        dV_dT_P = self.dV_dT_P()
        d2P_dTdP = self.d2P_dTdP()
        d2H_dep_dTdP = self.d2H_dTdP()
        return -T*dP_dT_V*d2V_dTdP - T*dV_dT_P*d2P_dTdP + d2H_dep_dTdP

    def chemical_potential(self):
        try:
            return self._chemical_potentials
        except AttributeError:
            pass
        dS_dzs = self.dS_dzs()
        dH_dzs = self.dH_dzs()
        T, Hfs, Sfs = self.T, self.Hfs, self.Sfs
        dG_reactive_dzs = [Hfs[i] - T*(Sfs[i] + dS_dzs[i]) + dH_dzs[i] for i in self.cmps]
        dG_reactive_dns = dxs_to_dns(dG_reactive_dzs, self.zs)
        chemical_potentials = dns_to_dn_partials(dG_reactive_dns, self.G_reactive())   
        self._chemical_potentials = chemical_potentials
        return chemical_potentials
#        # CORRECT DO NOT CHANGE
#        # TODO analytical implementation
#        def to_diff(ns):
#            tot = sum(ns)
#            zs = normalize(ns)
#            return tot*self.to_TP_zs(self.T, self.P, zs).G_reactive()
#        return jacobian(to_diff, self.zs)
    
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
        try:
            return self._dV_dT
        except AttributeError:
            pass
        dV_dT = self._dV_dT = -self.dP_dT()/self.dP_dV()
        return dV_dT
    
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

    def dZ_dzs(self):
        factor = self.P/(self.T*R)
        return [dV*factor for dV in self.dV_dzs()]

    def dZ_dns(self):
        return dxs_to_dns(self.dZ_dzs(), self.zs)

    def dV_dns(self):
        return dxs_to_dns(self.dV_dzs(), self.zs)

    # Derived properties    
    def PIP(self):
        return phase_identification_parameter(self.V(), self.dP_dT(), self.dP_dV(), 
                                              self.d2P_dV2(), self.d2P_dTdV())
        
    def kappa(self):
        return isothermal_compressibility(self.V(), self.dV_dP())
    
    def isothermal_bulk_modulus(self):
        return 1.0/self.kappa()

    def isobaric_expansion(self):
        return isobaric_expansion(self.V(), self.dV_dT())
    
    def isentropic_exponent(self):
#        return 1.3
        return self.Cp()/self.Cv()
    
    def dbeta_dT(self):
        '''
        from sympy import *
        T, P = symbols('T, P')
        V = symbols('V', cls=Function)
        expr = 1/V(T, P)*Derivative(V(T, P), T)
        diff(expr, T)
        Derivative(V(T, P), (T, 2))/V(T, P) - Derivative(V(T, P), T)**2/V(T, P)**2
        # Untested
        '''
        V_inv = 1.0/self.V()
        dV_dT = self.dV_dT()
        return V_inv*(self.d2V_dT2() - dV_dT*dV_dT*V_inv)
    
    def dbeta_dP(self):
        '''
        from sympy import *
        T, P = symbols('T, P')
        V = symbols('V', cls=Function)
        expr = 1/V(T, P)*Derivative(V(T, P), T)
        diff(expr, P)
        Derivative(V(T, P), P, T)/V(T, P) - Derivative(V(T, P), P)*Derivative(V(T, P), T)/V(T, P)**2
        
        '''
        V_inv = 1.0/self.V()
        dV_dT = self.dV_dT()
        dV_dP = self.dV_dP()
        return V_inv*(self.d2V_dTdP() - dV_dT*dV_dP*V_inv)


    def Joule_Thomson(self):
        return Joule_Thomson(self.T, self.V(), self.Cp(), dV_dT=self.dV_dT(), beta=self.isobaric_expansion())
    
    def speed_of_sound(self):
        # Intentionally molar
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
    
    def drho_dV_T(self):
        V = self.V()
        return -1.0/(V*V)
    
    def drho_dT_V(self):
        return 0.0
    
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

    
    def _Cp_pure_fast(self, Cps_data):
        Cps = []
        T, cmps = self.T, self.cmps
        Tmins, Tmaxs, coeffs = Cps_data[0], Cps_data[3], Cps_data[12]
        Tmin_slopes = Cps_data[1]
        Tmin_values = Cps_data[2]
        Tmax_slopes = Cps_data[4]
        Tmax_values = Cps_data[5]
        
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

    def _dCp_dT_pure_fast(self, Cps_data):
        dCps = []
        T, cmps = self.T, self.cmps
        Tmins, Tmaxs, coeffs = Cps_data[0], Cps_data[3], Cps_data[12]
        Tmin_slopes = Cps_data[1]
        Tmin_values = Cps_data[2]
        Tmax_slopes = Cps_data[4]
        Tmax_values = Cps_data[5]
        
        for i in cmps:
            if T < Tmins[i]:
                dCp = Tmin_slopes[i]
            elif T > Tmaxs[i]:
                dCp = Tmax_slopes[i]
            else:
                Cp, dCp = 0.0, 0.0
                for c in coeffs[i]:
                    dCp = T*dCp + Cp
                    Cp = T*Cp + c
            dCps.append(dCp)
        return dCps
        
    def _Cp_integrals_pure_fast(self, Cps_data):
        Cp_integrals_pure = []
        T, cmps = self.T, self.cmps
        Tmins, Tmaxes, int_coeffs = Cps_data[0], Cps_data[3], Cps_data[13]
        for i in cmps:
            # If indeed everything is working here, need to optimize to decide what to store
            # Try to save lookups to avoid cache misses
            # Instead of storing horner Tmin and Tmax, store -:
            # tot(Tmin) - Cps_data[7][i]
            # and tot1 + tot for the high T
            # Should save quite a bit of lookups! est. .12 go to .09
#                Tmin = Tmins[i]
#                if T < Tmin:
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    H = T*(0.5*Cps_data[1][i]*T + x1)
#                elif (T <= Tmaxes[i]):
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    tot = Tmin*(0.5*Cps_data[1][i]*Tmin + x1)
#                    
#                    tot1 = 0.0
#                    for c in int_coeffs[i]:
#                        tot1 = tot1*T + c
#                    tot1 -= Cps_data[7][i]
##                    tot1 = horner(int_coeffs[i], T) - horner(int_coeffs[i], Tmin)
#                    H = tot + tot1
#                else:
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    tot = Tmin*(0.5*Cps_data[1][i]*Tmin + x1)
#                    
#                    tot1 = Cps_data[8][i] - Cps_data[7][i]
#                    
#                    x1 = Cps_data[5][i] - Cps_data[4][i]*Tmaxes[i]
#                    tot2 = T*(0.5*Cps_data[4][i]*T + x1) - Tmaxes[i]*(0.5*Cps_data[4][i]*Tmaxes[i] + x1)
#                    H = tot + tot1 + tot2
                
                
                
            # ATTEMPT AT FAST HERE (NOW WORKING)
            if T < Tmins[i]:
                x1 = Cps_data[2][i] - Cps_data[1][i]*Tmins[i]
                H = T*(0.5*Cps_data[1][i]*T + x1)
            elif (T <= Tmaxes[i]):
                H = 0.0
                for c in int_coeffs[i]:
                    H = H*T + c
                H -= Cps_data[7][i]
            else:
                Tmax_slope = Cps_data[4][i]
                x1 = Cps_data[5][i] - Tmax_slope*Tmaxes[i]
                H = T*(0.5*Tmax_slope*T + x1) - Tmaxes[i]*(0.5*Tmax_slope*Tmaxes[i] + x1)
                H += Cps_data[8][i]

            Cp_integrals_pure.append(H - Cps_data[11][i])
        return Cp_integrals_pure

    def _Cp_integrals_over_T_pure_fast(self, Cps_data):
        Cp_integrals_over_T_pure = []
        T, cmps = self.T, self.cmps
        Tmins, Tmaxes, T_int_T_coeffs = Cps_data[0], Cps_data[3], Cps_data[14]
        logT = log(T)
        for i in cmps:
            Tmin = Tmins[i]
            if T < Tmin:
                x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
                S = (Cps_data[1][i]*T + x1*logT)
            elif (Tmin <= T <= Tmaxes[i]):
                S = 0.0
                for c in T_int_T_coeffs[i]:
                    S = S*T + c
                S += Cps_data[6][i]*logT
                # The below should be in a constant - taking the place of Cps_data[9]
                S -= Cps_data[9][i]
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    S += (Cps_data[1][i]*Tmin + x1*log(Tmin))
            else:        
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    S = (Cps_data[1][i]*Tmin + x1*log(Tmin))
#                    S += (Cps_data[10][i] - Cps_data[9][i])
                S = Cps_data[10][i] 
                # The above should be in the constant Cps_data[10], - x2*log(Tmaxes[i]) also
                x2 = Cps_data[5][i] - Tmaxes[i]*Cps_data[4][i]
                S += -Cps_data[4][i]*(Tmaxes[i] - T) + x2*logT #- x2*log(Tmaxes[i])
                
            Cp_integrals_over_T_pure.append(S - Cps_data[15][i])
        return Cp_integrals_over_T_pure

    def Cpigs_pure(self):
        try:
            return self._Cpigs
        except AttributeError:
            pass
        if self.Cpgs_locked:
            self._Cpigs = self._Cp_pure_fast(self.Cpgs_data)
            return self._Cpigs
                
        T = self.T
        self._Cpigs = [i.T_dependent_property(T) for i in self.HeatCapacityGases]
        return self._Cpigs

    def Cpig_integrals_pure(self):
        try:
            return self._Cpig_integrals_pure
        except AttributeError:
            pass
        if self.Cpgs_locked:
            self._Cpig_integrals_pure = self._Cp_integrals_pure_fast(self.Cpgs_data)
            return self._Cpig_integrals_pure

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
            self._Cpig_integrals_over_T_pure = self._Cp_integrals_over_T_pure_fast(self.Cpgs_data)
            return self._Cpig_integrals_over_T_pure

                
        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        self._Cpig_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        return self._Cpig_integrals_over_T_pure

    def dCpigs_dT_pure(self):
        try:
            return self._dCpigs_dT
        except AttributeError:
            pass
        if self.Cpgs_locked:
            self._dCpigs_dT = self._dCp_dT_pure_fast(self.Cpgs_data)
            return self._dCpigs_dT
                
        T = self.T
        self._dCpigs_dT = [i.T_dependent_property_derivative(T) for i in self.HeatCapacityGases]
        return self._dCpigs_dT


    def Cpls_pure(self):
        try:
            return self._Cpls
        except AttributeError:
            pass
        if self.Cpls_locked:
            self._Cpls = self._Cp_pure_fast(self.Cpls_data)
            return self._Cpls
                
        T = self.T
        self._Cpls = [i.T_dependent_property(T) for i in self.HeatCapacityLiquids]
        return self._Cpls

    def Cpl_integrals_pure(self):
        try:
            return self._Cpl_integrals_pure
        except AttributeError:
            pass
#        def to_quad(T, i):
#            l2 = self.to_TP_zs(T, self.P, self.zs)
#            return l2.Cpls_pure()[i] + (l2.Vms_sat()[i] - T*l2.dVms_sat_dT()[i])*l2.dPsats_dT()[i]
#        from scipy.integrate import quad
#        vals = [float(quad(to_quad, self.T_REF_IG, self.T, args=i)[0]) for i in self.cmps]
##        print(vals, self._Cp_integrals_pure_fast(self.Cpls_data))
#        return vals
        
        if self.Cpls_locked:
            self._Cpl_integrals_pure = self._Cp_integrals_pure_fast(self.Cpls_data)
            return self._Cpl_integrals_pure

        T, T_REF_IG, HeatCapacityLiquids = self.T, self.T_REF_IG, self.HeatCapacityLiquids
        self._Cpl_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityLiquids]
        return self._Cpl_integrals_pure

    def Cpl_integrals_over_T_pure(self):
        try:
            return self._Cpl_integrals_over_T_pure
        except AttributeError:
            pass
#        def to_quad(T, i):
#            l2 = self.to_TP_zs(T, self.P, self.zs)
#            return (l2.Cpls_pure()[i] + (l2.Vms_sat()[i] - T*l2.dVms_sat_dT()[i])*l2.dPsats_dT()[i])/T
#        from scipy.integrate import quad
#        vals = [float(quad(to_quad, self.T_REF_IG, self.T, args=i)[0]) for i in self.cmps]
##        print(vals, self._Cp_integrals_over_T_pure_fast(self.Cpls_data))
#        return vals

        if self.Cpls_locked:
            self._Cpl_integrals_over_T_pure = self._Cp_integrals_over_T_pure_fast(self.Cpls_data)
            return self._Cpl_integrals_over_T_pure

                
        T, T_REF_IG, HeatCapacityLiquids = self.T, self.T_REF_IG, self.HeatCapacityLiquids
        self._Cpl_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                   for obj in HeatCapacityLiquids]
        return self._Cpl_integrals_over_T_pure

    def V_ideal_gas(self):
        return R*self.T/self.P
    
    def H_ideal_gas(self):
        try:
            return self._H_ideal_gas
        except AttributeError:
            pass
        H = 0.0
        for zi, Cp_int in zip(self.zs, self.Cpig_integrals_pure()):
            H += zi*Cp_int
        self._H_ideal_gas = H
        return H

    def S_ideal_gas(self):
        try:
            return self._S_ideal_gas
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
        self._S_ideal_gas = S
        
        # dS_ideal_gas_dP = -R/P
        return S
    
    def Cp_ideal_gas(self):
        try:
            return self._Cp_ideal_gas
        except AttributeError:
            pass
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        for i in self.cmps:
            Cp += zs[i]*Cpigs_pure[i]
        self._Cp_ideal_gas = Cp
        return Cp
    
    def Cv_ideal_gas(self):
        try:
            Cp = self._Cp_ideal_gas
        except AttributeError:
            Cp = self.Cp_ideal_gas()
        return Cp - R

    def Cv_dep(self):
        return self.Cv() - self.Cv_ideal_gas()
    
    def Cp_Cv_ratio_ideal_gas(self):
        return self.Cp_ideal_gas()/self.Cv_ideal_gas()

    def G_ideal_gas(self):
        G_ideal_gas = self.H_ideal_gas() - self.T*self.S_ideal_gas()
        return G_ideal_gas

    def U_ideal_gas(self):
        U_ideal_gas = self.H_ideal_gas() - self.P*self.V_ideal_gas()
        return U_ideal_gas

    def A_ideal_gas(self):
        A_ideal_gas = self.U_ideal_gas() - self.T*self.S_ideal_gas()
        return A_ideal_gas
    
    def mechanical_critical_point(self):
        zs = self.zs
        # Get initial guess
        try:
            try:
                Tcs, Pcs = self.Tcs, self.Pcs
            except:
                Tcs, Pcs = self.eos_mix.Tcs, self.eos_mix.Pcs
            Pmc = sum([Pcs[i]*zs[i] for i in self.cmps])
            Tmc = sum([(Tcs[i]*Tcs[j])**0.5*zs[j]*zs[i] for i in self.cmps
                      for j in self.cmps])
        except Exception as e:
            Tmc = 300.0
            Pmc = 1e6
        
        # Try to solve it
        global new
        def to_solve(TP):
            global new
            T, P = float(TP[0]), float(TP[1])
            new = self.to_TP_zs(T=T, P=P, zs=zs)
            errs = [new.dP_drho(), new.d2P_drho2()]
            return errs
        
        jac = lambda TP: jacobian(to_solve, TP, scalar=False)
        TP, iters = newton_system(to_solve, [Tmc, Pmc], jac=jac, ytol=1e-10) 
#        TP = fsolve(to_solve, [Tmc, Pmc]) # fsolve handles the discontinuities badly
        T, P = float(TP[0]), float(TP[1])
        V = new.V()
        self._mechanical_critical_T = T
        self._mechanical_critical_P = P
        self._mechanical_critical_V = V
        return T, P, V
    
    def Tmc(self):
        try:
            return self._mechanical_critical_T
        except:
            self.mechanical_critical_point()
            return self._mechanical_critical_T

    def Pmc(self):
        try:
            return self._mechanical_critical_P
        except:
            self.mechanical_critical_point()
            return self._mechanical_critical_P

    def Vmc(self):
        try:
            return self._mechanical_critical_V
        except:
            self.mechanical_critical_point()
            return self._mechanical_critical_V

    def Zmc(self):
        try:
            V = self._mechanical_critical_V
        except:
            self.mechanical_critical_point()
            V = self._mechanical_critical_V
        return (self.Pmc()*self.Vmc())/(R*self.Tmc())
            
    def dH_dT_P(self):
        return self.dH_dT()

    def dH_dP_T(self):
        return self.dH_dP()

    def dS_dP_T(self):
        return self.dS_dP()

    def dS_dV_T(self):
        return self.dS_dP_T()*self.dP_dV()
            
    def dS_dV_P(self):
        return self.dS_dT_P()*self.dT_dV()
    
    def dP_dT_P(self):
        return 0.0

    def dP_dV_P(self):
        return 0.0

    def dT_dP_T(self):
        return 0.0

    def dT_dV_T(self):
        return 0.0

    def dV_dT_V(self):
        return 0.0

    def dV_dP_V(self):
        return 0.0
    
    def dP_dP_T(self):
        return 1.0

    def dP_dP_V(self):
        return 1.0

    def dT_dT_P(self):
        return 1.0

    def dT_dT_V(self):
        return 1.0

    def dV_dV_T(self):
        return 1.0

    def dV_dV_P(self):
        return 1.0
    
    d2T_dV2_P = d2T_dV2
    d2V_dT2_P = d2V_dT2
    d2V_dP2_T = d2V_dP2
    d2T_dP2_V = d2T_dP2
    dV_dP_T = dV_dP
    dV_dT_P = dV_dT
    dT_dP_V = dT_dP
    dT_dV_P = dT_dV
    
    
    # More derivatives - at const H, S, G, U, A
    _derivs_jacobian_x = 'V'
    _derivs_jacobian_y = 'T'
    
    def _derivs_jacobian(self, a, b, c, x=_derivs_jacobian_x,
                         y=_derivs_jacobian_y):
        r'''Calculates and returns a first-order derivative of one property
        with respect to another property at constant another property.
        
        This is particularly useful to obtain derivatives with respect to 
        another property which is not an intensive variable in a model,
        allowing for example derivatives at constant enthalpy or Gibbs energy
        to be obtained. This formula is obtained from the first derivative
        principles of reciprocity, the chain rule, and the triple product rule
        as shown in [1]_.
        
        ... math::
            \left(\frac{\partial a}{\partial b}\right)_{c}=
            \frac{\left(\frac{\partial a}{\partial x}\right)_{y}\left(
            \frac{\partial c}{\partial y}\right)_{x}-\left(\frac{\partial a}{
            \partial y}\right)_{x}\left(\frac{\partial c}{\partial x}
            \right)_{y}}{\left(\frac{\partial b}{\partial x}\right)_{y}\left(
            \frac{\partial c}{\partial y}\right)_{x}-\left(\frac{\partial b}
            {\partial y}\right)_{x}\left(\frac{\partial c}{\partial x}
            \right)_{y}}
    
        References
        ----------
        .. [1] Thorade, Matthis, and Ali Saadat. "Partial Derivatives of
           Thermodynamic State Properties for Dynamic Simulation." 
           Environmental Earth Sciences 70, no. 8 (April 10, 2013): 3497-3503. 
           https://doi.org/10.1007/s12665-013-2394-z.
        '''
        n0 = getattr(self, 'd%s_d%s_%s'%(a, x, y))()
        n1 = getattr(self, 'd%s_d%s_%s'%(c, y, x))()

        n2 = getattr(self, 'd%s_d%s_%s'%(a, y, x))()
        n3 = getattr(self, 'd%s_d%s_%s'%(c, x, y))()
        
        d0 = getattr(self, 'd%s_d%s_%s'%(b, x, y))()
        d1 = getattr(self, 'd%s_d%s_%s'%(c, y, x))()
        
        d2 = getattr(self, 'd%s_d%s_%s'%(b, y, x))()
        d3 = getattr(self, 'd%s_d%s_%s'%(c, x, y))()
        
        return (n0*n1 - n2*n3)/(d0*d1 - d2*d3)
    

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
    
#    def mu(self):
#        return self.result.mu(self)

#    def k(self):
#        return self.result.k(self)
#    
#    def ws(self):
#        return self.result.ws(self)
        
    
#    def atom_fractions(self):
#        return self.result.atom_fractions(self)
#    
#    def atom_mass_fractions(self):
#        return self.result.atom_mass_fractions(self)

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
    
    def drho_mass_dT(self):
        r'''Method to calculate the mass density derivative with respect to
        temperature, at constant pressure.
        
        .. math::
            \left(\frac{\partial \rho}{\partial T}\right)_{P} = 
            \frac{-\text{MW} \frac{\partial V_m}{\partial T}}{1000 V_m^2}
            
        Returns
        -------
        drho_mass_dT : float
           Temperature derivative of mass density at constant pressure,
           [kg/m^3/K]
            
        Notes
        -----
        Requires `dV_dT`, `MW`, and `V`.
        
        This expression is readily obtainable with SymPy:
        
        >>> from sympy import * # doctest: +SKIP
        >>> T, P, MW = symbols('T, P, MW') # doctest: +SKIP
        >>> Vm = symbols('Vm', cls=Function) # doctest: +SKIP
        >>> rho_mass = (Vm(T))**-1*MW/1000 # doctest: +SKIP
        >>> diff(rho_mass, T) # doctest: +SKIP
        -MW*Derivative(Vm(T), T)/(1000*Vm(T)**2)
        '''
        try:
            return self._drho_mass_dT
        except AttributeError:
            pass
        MW = self.MW()
        V = self.V()
        dV_dT = self.dV_dT()
        self._drho_mass_dT = drho_mass_dT = -MW*dV_dT/(1000.0*V*V)
        return drho_mass_dT

    def drho_mass_dP(self):
        r'''Method to calculate the mass density derivative with respect to
        pressure, at constant temperature.
        
        .. math::
            \left(\frac{\partial \rho}{\partial P}\right)_{T} = 
            \frac{-\text{MW} \frac{\partial V_m}{\partial P}}{1000 V_m^2}
            
        Returns
        -------
        drho_mass_dP : float
           Pressure derivative of mass density at constant temperature,
           [kg/m^3/Pa]
            
        Notes
        -----
        Requires `dV_dP`, `MW`, and `V`.
        
        This expression is readily obtainable with SymTy:
        
        >>> from sympy import * # doctest: +SKIP
        >>> P, T, MW = symbols('P, T, MW') # doctest: +SKIP
        >>> Vm = symbols('Vm', cls=Function) # doctest: +SKIP
        >>> rho_mass = (Vm(P))**-1*MW/1000 # doctest: +SKIP
        >>> diff(rho_mass, P) # doctest: +SKIP
        -MW*Derivative(Vm(P), P)/(1000*Vm(P)**2)
        '''
        try:
            return self._drho_mass_dP
        except AttributeError:
            pass
        MW = self.MW()
        V = self.V()
        dV_dP = self.dV_dP()
        self._drho_mass_dP = drho_mass_dP = -MW*dV_dP/(1000.0*V*V)
        return drho_mass_dP
    
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
    
    def P_transitions(self):
        return []
    
    def T_max_at_V(self, V):
        return None
    
    def P_max_at_V(self, V):
        return None
    
    def dspeed_of_sound_dT_P(self):
        r'''Method to calculate the temperature derivative of speed of sound
        at constant pressure in molar units.
        
        .. math::
            \left(\frac{\partial c{\partial T}\right)_P = 
            - \frac{\sqrt{- \frac{\operatorname{Cp}{\left(T \right)} V^{2}
            {\left(T \right)} \operatorname{dPdV_{T}}{\left(T \right)}}
            {\operatorname{Cv}{\left(T \right)}}} \left(- \frac{\operatorname{Cp}
            {\left(T \right)} V^{2}{\left(T \right)} \frac{d}{d T}
            \operatorname{dPdV_{T}}{\left(T \right)}}{2 \operatorname{Cv}{\left(T
            \right)}} - \frac{\operatorname{Cp}{\left(T \right)} V{\left(T 
            \right)} \operatorname{dPdV_{T}}{\left(T \right)} \frac{d}{d T}
            V{\left(T \right)}}{\operatorname{Cv}{\left(T \right)}} 
            + \frac{\operatorname{Cp}{\left(T \right)} V^{2}{\left(T \right)}
            \operatorname{dPdV_{T}}{\left(T \right)} \frac{d}{d T}
            \operatorname{Cv}{\left(T \right)}}{2 \operatorname{Cv}^{2}
            {\left(T \right)}} - \frac{V^{2}{\left(T \right)} \operatorname{
            dPdV_{T}}{\left(T \right)} \frac{d}{d T} \operatorname{Cp}{\left(T
            \right)}}{2 \operatorname{Cv}{\left(T \right)}}\right) 
            \operatorname{Cv}{\left(T \right)}}{\operatorname{Cp}{\left(T 
            \right)} V^{2}{\left(T \right)} \operatorname{dPdV_{T}}{\left(T
            \right)}}

        Returns
        -------
        dspeed_of_sound_dT_P : float
           Temperature derivative of speed of sound at constant pressure, 
           [m*kg^0.5/s/mol^0.5/K]
            
        Notes
        -----
        Requires the temperature derivative of Cp and Cv both at constant
        pressure, as wel as the volume and temperature derivative of pressure,
        calculated at constant temperature and then pressure respectively. 
        These can be tricky to obtain.
        '''
        '''Calculation with SymPy:
        from sympy import *
        T = symbols('T')
        V, dPdV_T, Cp, Cv = symbols('V, dPdV_T, Cp, Cv', cls=Function)
        c = sqrt(-V(T)**2*dPdV_T(T)*Cp(T)/Cv(T))
        '''
        x0 = self.Cp()
        x1 = self.V()
        x2 = self.dP_dV()
        x3 = self.Cv()
        x4 = x0*x2
        x5 = x4/x3
        x6 = 0.5*x1
        
        x50 = self.d2P_dVdT_TP()
        x51 = self.d2H_dT2()
        x52 = self.dV_dT()
        x53 = self.dCv_dT_P()
        
        return (-x1*x1*x5)**0.5*(x0*x6*x50 + x2*x6*x51 + x4*x52- x5*x6*x53)/(x0*x1*x2)

    def dspeed_of_sound_dP_T(self):
        x0 = self.Cp()
        x1 = self.V()
        x2 = self.dP_dV()
        x3 = self.Cv()
        x4 = x0*x2
        x5 = x4/x3
        x6 = 0.5*x1
        
        x50 = self.d2P_dVdP()
        x51 = self.d2H_dTdP()
        x52 = self.dV_dP()
        x53 = self.dCv_dP_T()
        
        return (-x1*x1*x5)**0.5*(x0*x6*x50 + x2*x6*x51 + x4*x52- x5*x6*x53)/(x0*x1*x2)
    
    
    # Transport properties
    def mu(self):
        if isinstance(self, gas_phases):
            return self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        elif isinstance(self, liquid_phases):
            return self.correlations.ViscosityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        else:
            raise NotImplementedError("Did not work")

    def ws(self):
        try:
            return self._ws
        except AttributeError:
            pass
        MWs = self.constants.MWs
        zs, cmps = self.zs, self.cmps
        ws = [zs[i]*MWs[i] for i in cmps]
        Mavg = 1.0/sum(ws)
        for i in cmps:
            ws[i] *= Mavg
        self._ws = ws
        return ws

    @property
    def beta(self):
        try:
            result = self.result
        except:
            return None
        return result.betas[result.phases.index(self)]

    @property
    def beta_mass(self):
        try:
            result = self.result
        except:
            return None
        return result.betas_mass[result.phases.index(self)]

    @property
    def beta_volume(self):
        try:
            result = self.result
        except:
            return None
        return result.betas_volume[result.phases.index(self)]
    
    @property
    def VF(self):
        return self.result.gas_beta



for a in ('T', 'P', 'V', 'rho'):
    for b in ('T', 'P', 'V', 'rho'):
        for c in ('H', 'S', 'G', 'U', 'A'):
           def _der(self, a=a, b=b, c=c):
               return self._derivs_jacobian(a=a, b=b, c=c)
           setattr(Phase, 'd%s_d%s_%s' %(a, b, c), _der)


class IdealGas(Phase):
    '''DO NOT DELETE - EOS CLASS IS TOO SLOW!
    This will be important for fitting.
    
    '''
    phase = 'g'
    force_phase = 'g'
    composition_independent = True
    def __init__(self, HeatCapacityGases=None, Hfs=None, Gfs=None, T=None, P=None, zs=None):
        self.HeatCapacityGases = HeatCapacityGases
        self.Hfs = Hfs
        self.Gfs = Gfs
        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            self.Sfs = [(Hfi - Gfi)/298.15 for Hfi, Gfi in zip(Hfs, Gfs)]
        else:
            self.Sfs = None
            
        if zs is not None:
            self.N = N = len(zs)
            self.cmps = range(N)
            self.zeros1d = [0.0]*N
            self.ones1d = [1.0]*N
        elif HeatCapacityGases is not None:
            self.N = N = len(HeatCapacityGases)
            self.cmps = range(N)
            self.zeros1d = [0.0]*N
            self.ones1d = [1.0]*N
        if zs is not None:
            self.zs = zs
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P
        
    def fugacities(self):
        P = self.P
        return [P*zi for zi in self.zs]
    
    def lnphis(self):
        return self.zeros1d
    
    lnphis_G_min = lnphis
    
    def phis(self):
        return self.ones1d
    
    def dphis_dT(self):
        return self.zeros1d

    def dphis_dP(self):
        return self.zeros1d
    
    def dlnphis_dT(self):
        return self.zeros1d

    def dlnphis_dP(self):
        return self.zeros1d

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        new.N = self.N
        new.cmps = self.cmps
        new.zeros1d = self.zeros1d
        new.ones1d = self.ones1d
        
        new.HeatCapacityGases = self.HeatCapacityGases
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        return new
    
    def to_zs_TPV(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        if T is not None and V is not None:
            P = R*T/V
        elif P is not None and V is not None:
            T = P*V/R
        elif T is not None and P is not None:
            pass
        else:
            raise ValueError("Two of T, P, or V are needed")
        new.P = P
        new.T = T
        
        new.zs = zs
        new.N = self.N
        new.cmps = self.cmps
        new.zeros1d = self.zeros1d
        new.ones1d = self.ones1d
        
        new.HeatCapacityGases = self.HeatCapacityGases
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs

        return new
        
    to = to_zs_TPV
    
    ### Volumetric properties
    def V(self):
        return R*self.T/self.P

    def dP_dT(self):
        return self.P/self.T
    dP_dT_V = dP_dT

    def dP_dV(self):
        return -self.P*self.P/(R*self.T)

    dP_dV_T = dP_dV

    def d2P_dT2(self):
        return 0.0
    d2P_dT2_V = d2P_dT2

    def d2P_dV2(self):
        P, T = self.P, self.T
        return 2.0*P*P*P/(R*R*T*T)

    d2P_dV2_T = d2P_dV2

    def d2P_dTdV(self):
        P, T = self.P, self.T
        return -P*P/(R*T*T)

    def d2T_dV2(self):
        return 0.0
        
    d2T_dV2_P = d2T_dV2

    def d2V_dT2(self):
        return 0.0
        
    d2V_dT2_P = d2V_dT2
    
    def dV_dT(self):
        return R/self.P
    
    def PIP(self):
        return 1.0 # For speed

    def d2V_dP2(self):
        P, T = self.P, self.T
        return 2.0*R*T/(P*P*P)
    
    def d2T_dP2(self):
        return 0.0
    
    def dV_dP(self):
        P, T = self.P, self.T
        return -R*T/(P*P)
    
    def dT_dP(self):
        return self.T/self.P
    
    def dT_dV(self):
        return self.P*R_inv
    
    def dV_dzs(self):
        return self.zeros1d

    
    d2T_dV2_P = d2T_dV2
    d2V_dT2_P = d2V_dT2
    d2V_dP2_T = d2V_dP2
    d2T_dP2_V = d2T_dP2
    dV_dP_T = dV_dP
    dV_dT_P = dV_dT
    dT_dP_V = dT_dP
    dT_dV_P = dT_dV
    
    ### Thermodynamic properties

    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        zs = self.zs
        try:
            Cpig_integrals_pure = self._Cpig_integrals_pure
        except AttributeError:
            Cpig_integrals_pure = self.Cpig_integrals_pure()
        H = 0.0
        for i in self.cmps:
            H += zs[i]*Cpig_integrals_pure[i]
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
        self._S = S
        return S
    
    def Cp(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        for i in self.cmps:
            Cp += zs[i]*Cpigs_pure[i]
        self._Cp = Cp
        return Cp 

    dH_dT = Cp
    dH_dT_V = Cp # H does not depend on P, so the P is increased without any effect on H

    def dH_dP(self):
        return 0.0
        
    def d2H_dT2(self):
        try:
            return self._d2H_dT2
        except AttributeError:
            pass
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        for i in self.cmps:
            dCp += zs[i]*dCpigs_pure[i]
        self._d2H_dT2 = dCp
        return dCp

    def d2H_dP2(self):
        return 0.0
        
    def d2H_dTdP(self):
        return 0.0

    def dH_dP_V(self):
        dH_dP_V = self.Cp()*self.dT_dP()
        return dH_dP_V

    def dH_dV_T(self):
        return 0.0
        
    def dH_dV_P(self):
        dH_dV_P = self.dT_dV()*self.Cp()
        return dH_dV_P

    def dH_dzs(self):
        return self.Cpig_integrals_pure()

    def dS_dT(self):
        dS_dT = self.Cp()/self.T
        return dS_dT
    dS_dT_P = dS_dT

    def dS_dP(self):
        return -R/self.P

    def d2S_dP2(self):
        P = self.P
        return R/(P*P)
    
    def dS_dT_V(self):
        dS_dT_V = self.Cp()/self.T - R/self.P*self.dP_dT()
        return dS_dT_V

    def dS_dP_V(self):
        dS_dP_V = -R/self.P + self.Cp()/self.T*self.dT_dP()
        return dS_dP_V
    
    def d2P_dTdP(self):
        return 0.0
        
    def d2P_dVdP(self):
        return 0.0
    
    def d2P_dVdT_TP(self):
        return 0.0
    
    def d2P_dT2_PV(self):
        return 0.0
    
    def dS_dzs(self):
        try:
            return self._dS_dzs
        except AttributeError:
            pass
        cmps, eos_mix = self.cmps, self.eos_mix
    
        log_zs = self.log_zs()
        integrals = self.Cpig_integrals_over_T_pure()

        self._dS_dzs = [integrals[i] - R*(log_zs[i] + 1.0)
                        for i in cmps]
        return self._dS_dzs

    # Properties using constants, correlations
    def mu(self):
        try:
            return self._mu
        except AttributeError:
            pass
        mu = self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._mu = mu
        return mu

    def k(self):
        try:
            return self._k
        except AttributeError:
            pass
        k = self.correlations.ThermalConductivityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._k = k
        return k

class EOSGas(Phase):
    
    def model_hash(self, ignore_phase=False):
        if ignore_phase:
            try:
                return self._model_hash_ignore_phase
            except AttributeError:
                pass
        else:
            try:
                return self._model_hash
            except AttributeError:
                pass
        to_hash = [self.eos_class, self.eos_kwargs,
                                   self.Hfs, self.Gfs, self.Sfs, self.HeatCapacityGases]
        if not ignore_phase:
            to_hash.append(self.__class__)
        h =  hash_any_primitive(to_hash)
        if ignore_phase:
            self._model_hash_ignore_phase = h
        else:
            self._model_hash = h
        return h
    
    @property
    def phase(self):
        phase = self.eos_mix.phase
        if phase in ('l', 'g'):
            return phase
        return 'g'
    
    def as_args(self):
        eos_kwargs = self.eos_kwargs.copy()
        base = 'EOSGas(eos_class=%s, eos_kwargs=%s, HeatCapacityGases=correlations.HeatCapacityGases,'  %(self.eos_class.__name__, self.eos_kwargs)
        for s in ('Hfs', 'Gfs', 'Sfs', 'T', 'P', 'zs'):
            if hasattr(self, s) and getattr(self, s) is not None:
                base += '%s=%s, ' %(s, getattr(self, s))
        base += ')'
        return base
        
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
        self.composition_independent = eos_class is IGMIX
        
        if T is not None and P is not None and zs is not None:
            self.T = T
            self.P = P
            self.zs = zs
            self.eos_mix = eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
            self.eos_pures_STP = [eos_mix.to_TPV_pure(T=298.15, P=101325.0, V=None, i=i) for i in self.cmps]
        else:
            eos_mix = self.eos_class(T=298.15, P=101325.0, zs=[1.0/N]*N, **self.eos_kwargs)
            self.eos_pures_STP = [eos_mix.to_TPV_pure(T=298.15, P=101325.0, V=None, i=i) for i in self.cmps]
            
        
    def to_TP_zs(self, T, P, zs, other_eos=None):
        # Why so slow
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        if other_eos is not None:
            other_eos.solve_missing_volumes()
            new.eos_mix = other_eos
        else:
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
        new.composition_independent = self.composition_independent
        
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        
        try:
            new.N = self.N
            new.cmps = self.cmps
            new.eos_pures_STP = self.eos_pures_STP
        except:
            pass

        return new

    def to_zs_TPV(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        
        if T is not None:
            if P is not None:
                try:
                    new.eos_mix = self.eos_mix.to_TP_zs_fast(T=T, P=P, zs=zs, only_g=True,
                                                             full_alphas=True)
                except AttributeError:
                    new.eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
            elif V is not None:
                try:
                    new.eos_mix = self.eos_mix.to_TV_zs(T=T, V=V, zs=zs, fugacities=False)
                except AttributeError:
                    new.eos_mix = self.eos_class(T=T, V=V, zs=zs, **self.eos_kwargs)
                P = new.eos_mix.P
        elif P is not None and V is not None:
            try:
                new.eos_mix = self.eos_mix.to_PV_zs(P=P, V=V, zs=zs, only_g=True, fugacities=False)
            except AttributeError:
                new.eos_mix = self.eos_class(P=P, V=V, zs=zs, only_g=True, **self.eos_kwargs)
            T = new.eos_mix.T
        else:
            raise ValueError("Two of T, P, or V are needed")
        new.P = P
        new.T = T
        
        new.eos_class = self.eos_class
        new.eos_kwargs = self.eos_kwargs
        
        new.HeatCapacityGases = self.HeatCapacityGases
        new.Cpgs_data = self.Cpgs_data
        new.Cpgs_locked = self.Cpgs_locked
        
        new.composition_independent = self.composition_independent
        
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        
        try:
            new.N = self.N
            new.cmps = self.cmps
            new.eos_pures_STP = self.eos_pures_STP
        except:
            pass

        return new
        
    to = to_zs_TPV
        
    def V_iter(self, force=False):
        # Can be some severe issues in the very low pressure/temperature range
        # For that reason, consider not doing TV iterations.
        # Cal occur also with PV iterations
        
        T, P = self.T, self.P
#        if 0 and ((P < 1.0 or T < 1.0) or (P/T < 500.0 and T < 50.0)):
        eos_mix = self.eos_mix
        V = self.V()
        P_err = abs((R*T/(V-eos_mix.b) - eos_mix.a_alpha/(V*V + eos_mix.delta*V + eos_mix.epsilon)) - P)
        if (P_err/P) < 1e-9 and not force:
            return V
        try:
            return eos_mix.V_g_mpmath.real
        except:
            return eos_mix.V_l_mpmath.real
#        else:
#            return self.V()
            
    def lnphis_G_min(self):
        eos_mix = self.eos_mix
        if eos_mix.phase == 'l/g':
            # Check both phases are solved, and complete if not
            eos_mix.solve_missing_volumes()
            if eos_mix.G_dep_l < eos_mix.G_dep_g:
                return eos_mix.fugacity_coefficients(eos_mix.Z_l, self.zs)
            return eos_mix.fugacity_coefficients(eos_mix.Z_g, self.zs)
        try:
            return eos_mix.fugacity_coefficients(eos_mix.Z_g, self.zs)
        except AttributeError:
            return eos_mix.fugacity_coefficients(eos_mix.Z_l, self.zs)
        
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
        
    def dlnphis_dns(self):
        eos_mix = self.eos_mix
        try:
            return eos_mix.dlnphis_dns(eos_mix.Z_g, eos_mix.zs)
        except:
            return eos_mix.dlnphis_dns(eos_mix.Z_l, eos_mix.zs)
        
    def dlnphis_dzs(self):
        # Confirmed to be mole fraction derivatives - taked with sum not 1 -
        # of the log fugacity coefficients!
        eos_mix = self.eos_mix
        try:
            return eos_mix.d_lnphi_dzs(eos_mix.Z_g, eos_mix.zs)
        except:
            return eos_mix.d_lnphi_dzs(eos_mix.Z_l, eos_mix.zs)

    def fugacities_lowest_Gibbs(self):
        eos_mix = self.eos_mix
        P = self.P
        zs = self.zs        
        try:
            if eos_mix.G_dep_g < eos_mix.G_dep_l:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_g, zs)
            else:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_l, zs)
        except:
            try:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_g, zs)
            except:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_l, zs)
        return [P*zs[i]*trunc_exp(lnphis[i]) for i in range(len(zs))]
        

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

    dP_dT_V = dP_dT

    def dP_dV(self):
        try:
            return self.eos_mix.dP_dV_g
        except AttributeError:
            return self.eos_mix.dP_dV_l

    dP_dV_T = dP_dV
    
    def d2P_dT2(self):
        try:
            return self.eos_mix.d2P_dT2_g
        except AttributeError:
            return self.eos_mix.d2P_dT2_l
        
    d2P_dT2_V = d2P_dT2

    def d2P_dV2(self):
        try:
            return self.eos_mix.d2P_dV2_g
        except AttributeError:
            return self.eos_mix.d2P_dV2_l
        
    d2P_dV2_T = d2P_dV2

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
        
    d2T_dV2_P = d2T_dV2

    def d2V_dT2(self):
        try:
            return self.eos_mix.d2V_dT2_g
        except AttributeError:
            return self.eos_mix.d2V_dT2_l
        
    d2V_dT2_P = d2V_dT2
    
    def dV_dzs(self):
        eos_mix = self.eos_mix
        try:
            dV_dzs = self.eos_mix.dV_dzs(eos_mix.Z_g, eos_mix.zs)
        except AttributeError:
            dV_dzs = self.eos_mix.dV_dzs(eos_mix.Z_l, eos_mix.zs)
        return dV_dzs
        
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
        try:
            return self._Cp
        except AttributeError:
            pass
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        for i in self.cmps:
            Cp += zs[i]*Cpigs_pure[i]
        Cp += self.Cp_dep()
        self._Cp = Cp
        return Cp 

    def dH_dT(self):
        return self.Cp()

    def dH_dP(self):
        try:
            return self._dH_dP
        except AttributeError:
            pass
        try:
            self._dH_dP = dH_dP = self.eos_mix.dH_dep_dP_g
        except AttributeError:
            self._dH_dP = dH_dP = self.eos_mix.dH_dep_dP_l
        return dH_dP
        
    def d2H_dT2(self):
        try:
            return self._d2H_dT2
        except AttributeError:
            pass
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        for i in self.cmps:
            dCp += zs[i]*dCpigs_pure[i]
        try:
            dCp += self.eos_mix.d2H_dep_dT2_g
        except AttributeError:
            dCp += self.eos_mix.d2H_dep_dT2_l
        self._d2H_dT2 = dCp
        return dCp
    
    def d2H_dT2_V(self):
        # Turned out not to be needed when I thought it was - ignore this!
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        for i in self.cmps:
            dCp += zs[i]*dCpigs_pure[i]
        
        try:
            dCp += self.eos_mix.d2H_dep_dT2_g_V
        except AttributeError:
            dCp += self.eos_mix.d2H_dep_dT2_l_V
        return dCp
    
    
    def d2H_dP2(self):
        try:
            return self.eos_mix.d2H_dep_dP2_g
        except AttributeError:
            return self.eos_mix.d2H_dep_dP2_l
        
    def d2H_dTdP(self):
        try:
            return self.eos_mix.d2H_dep_dTdP_g
        except AttributeError:
            return self.eos_mix.d2H_dep_dTdP_l

        
    def dH_dT_V(self):
        dH_dT_V = self.Cp_ideal_gas()
        try:
            dH_dT_V += self.eos_mix.dH_dep_dT_g_V
        except AttributeError:
            dH_dT_V += self.eos_mix.dH_dep_dT_l_V
        return dH_dT_V

    def dH_dP_V(self):
        dH_dP_V = self.Cp_ideal_gas()*self.dT_dP()
        try:
            dH_dP_V += self.eos_mix.dH_dep_dP_g_V 
        except AttributeError:
            dH_dP_V += self.eos_mix.dH_dep_dP_l_V
        return dH_dP_V

    def dH_dV_T(self):
        dH_dV_T = 0.0
        try:
            dH_dV_T += self.eos_mix.dH_dep_dV_g_T
        except AttributeError:
            dH_dV_T += self.eos_mix.dH_dep_dV_l_T
        return dH_dV_T
        
    def dH_dV_P(self):
        dH_dV_P = self.dT_dV()*self.Cp_ideal_gas()
        try:
            dH_dV_P += self.eos_mix.dH_dep_dV_g_P
        except AttributeError:
            dH_dV_P += self.eos_mix.dH_dep_dV_l_P
        return dH_dV_P

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

        dS_dT = self.Cp_ideal_gas()/T
        try:
            dS_dT += self.eos_mix.dS_dep_dT_g
        except AttributeError:
            dS_dT += self.eos_mix.dS_dep_dT_l
        return dS_dT

    def dS_dP(self):
        dS = 0.0
        P = self.P
        dS -= R/P
        try:
            dS += self.eos_mix.dS_dep_dP_g
        except AttributeError:
            dS += self.eos_mix.dS_dep_dP_l
        return dS

    def d2S_dP2(self):
        P = self.P
        d2S = R/(P*P)
        try:
            d2S += self.eos_mix.d2S_dep_dP_g
        except AttributeError:
            d2S += self.eos_mix.d2S_dep_dP_l
        return d2S
    
    def dS_dT_P(self):
        return self.dS_dT()
    
    def dS_dT_V(self):
        # Good
        '''
        # Second last bit from 
        from sympy import *
        T, R = symbols('T, R')
        P = symbols('P', cls=Function)
        expr =-R*log(P(T)/101325)
        diff(expr, T)
        '''
        dS_dT_V = self.Cp_ideal_gas()/self.T - R/self.P*self.dP_dT()
        try:
            dS_dT_V += self.eos_mix.dS_dep_dT_g_V
        except AttributeError:
            dS_dT_V += self.eos_mix.dS_dep_dT_l_V
        return dS_dT_V

    def dS_dP_V(self):
        dS_dP_V = -R/self.P + self.Cp_ideal_gas()/self.T*self.dT_dP()
        try:
            dS_dP_V += self.eos_mix.dS_dep_dP_g_V
        except AttributeError:
            dS_dP_V += self.eos_mix.dS_dep_dP_l_V
        return dS_dP_V
    
    # The following - likely should be reimplemented generically
    # http://www.coolprop.org/_static/doxygen/html/class_cool_prop_1_1_abstract_state.html#a0815380e76a7dc9c8cc39493a9f3df46

    def d2P_dTdP(self):
        
        try:
            return self.eos_mix.d2P_dTdP_g
        except AttributeError:
            return self.eos_mix.d2P_dTdP_l
        
    def d2P_dVdP(self):
        try:
            return self.eos_mix.d2P_dVdP_g
        except AttributeError:
            return self.eos_mix.d2P_dVdP_l
    
    def d2P_dVdT_TP(self):
        try:
            return self.eos_mix.d2P_dVdT_TP_g
        except AttributeError:
            return self.eos_mix.d2P_dVdT_TP_l
    
    def d2P_dT2_PV(self):
        try:
            return self.eos_mix.d2P_dT2_PV_g
        except AttributeError:
            return self.eos_mix.d2P_dT2_PV_l
    
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
 
    def mechanical_critical_point(self):
        zs = self.zs
        new = self.eos_mix.to_mechanical_critical_point()
        self._mechanical_critical_T = new.T
        self._mechanical_critical_P = new.P
        try:
            V = new.V_l
        except:
            V = new.V_g
        self._mechanical_critical_V = V
        return new.T, new.P, V
    
    def P_transitions(self):
        e = self.eos_mix
        return e.P_discriminant_zeros_analytical(e.T, e.b, e.delta, e.epsilon, e.a_alpha, valid=True)
        # EOS is guaranteed to be at correct temperature
        try:
            return [self.eos_mix.P_discriminant_zero_l()]
        except:
            return [self.eos_mix.P_discriminant_zero_g()]

    def T_transitions(self):
        try:
            return [self.eos_mix.T_discriminant_zero_l()]
        except:
            return [self.eos_mix.T_discriminant_zero_g()]

    def T_max_at_V(self, V):
        T_max = self.eos_mix.T_max_at_V(V)
        if T_max is not None:
            T_max = T_max*(1.0-1e-12)
        return T_max
    
    def P_max_at_V(self, V):
        P_max = self.eos_mix.P_max_at_V(V)
        if P_max is not None:
            P_max = P_max*(1.0-1e-12)
        return P_max
    
    def mu(self):
#        try:
#            return self._mu
#        except AttributeError:
#            pass
        phase = self.eos_mix.phase
        if phase == 'g':
            mu = self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        elif phase == 'l':
            mu = self.correlations.ViscosityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        else:
            mu = self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._mu = mu
        return mu

    def k(self):
        try:
            return self._k
        except AttributeError:
            pass
        phase = self.eos_mix.phase
        if phase == 'g':
            k = self.correlations.ThermalConductivityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        elif phase == 'l':
            k = self.correlations.ThermalConductivityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        else:
            k = self.correlations.ThermalConductivityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._k = k
        return k

def build_EOSLiquid():
    import inspect
    source = inspect.getsource(EOSGas)
    source = source.replace('EOSGas', 'EOSLiquid').replace('only_g', 'only_l')
    source = source.replace("'g'", "'gORIG'")
    source = source.replace("'l'", "'g'")
    source = source.replace("'gORIG'", "'l'")
    source = source.replace("ViscosityGasMixture", "gViscosityGasMixture")
    source = source.replace("ViscosityLiquidMixture", "ViscosityGasMixture")
    source = source.replace("gViscosityGasMixture", "ViscosityLiquidMixture")
    source = source.replace("ThermalConductivityGasMixture", "gThermalConductivityGasMixture")
    source = source.replace("ThermalConductivityLiquidMixture", "ThermalConductivityGasMixture")
    source = source.replace("gThermalConductivityGasMixture", "ThermalConductivityLiquidMixture")
    # TODO add new volume derivatives
    swap_strings = ('Cp_dep', 'd2P_dT2', 'd2P_dTdV', 'd2P_dV2', 'd2T_dV2',
                    'd2V_dT2', 'dH_dep_dP', 'dP_dT', 'dP_dV', 'phi',
                    'dS_dep_dP', 'dS_dep_dT', 'G_dep', 'H_dep', 'S_dep', '.V', '.Z',
                    'd2P_dVdT_TP', 'd2P_dT2_PV', 'd2P_dVdP', 'd2P_dTdP',
                    'd2S_dep_dP', 'dH_dep_dV', 'dH_dep_dT', 'd2H_dep_dTdP',
                    'd2H_dep_dP2', 'd2H_dep_dT2')
    for s in swap_strings:
        source = source.replace(s+'_g', 'gORIG')
        source = source.replace(s+'_l', s+'_g')
        source = source.replace('gORIG', s+'_l')
#    print(source)
    return source

try:
    EOSLiquid
except:
    # Cost is ~10 ms - must be pasted in the future!
    exec(build_EOSLiquid())

class GibbsExcessLiquid(Phase):
    force_phase = 'l'
    phase = 'l'
    P_DEPENDENT_H_LIQ = True
    Psats_data = None
    Psats_locked = False
    Vms_sat_locked = False
    Vms_sat_data = None
    Hvap_locked = False
    Hvap_data = None
    use_IG_Cp = True
    
    Cpls_locked = False
    Cpls_data = None
    
    Tait_B_data = None
    Tait_C_data = None
    def __init__(self, VaporPressures, VolumeLiquids=None, 
                 GibbsExcessModel=None, 
                 eos_pure_instances=None,
                 HeatCapacityGases=None, 
                 EnthalpyVaporizations=None,
                 HeatCapacityLiquids=None, 
                 use_Poynting=False,
                 use_phis_sat=False,
                 use_Tait=False,
                 use_IG_Cp=True,
                 Hfs=None, Gfs=None, Sfs=None,
                 henry_components=None, henry_data=None,
                 T=None, P=None, zs=None,
                 Psat_extrpolation='AB',
                 ):
        '''It is quite possible to introduce a PVT relation ship for liquid 
        density and remain thermodynamically consistent. However, must be 
        applied on a per-component basis! This class cannot have an 
        equation-of-state for a liquid MIXTURE!
        
        (it might still be nice to generalize the handling; maybe even allow)
        pure EOSs to be used too, and as a form/template for which functions to
        use).
        
        In conclusion, you have
        1) The standard H/S model
        2) The H/S model with all pressure correction happening at P
        3) The inconsistent model which has no pressure dependence whatsover in H/S
           This model is required due to its popularity, not its consistency (but still volume dependency)
           
        All mixture volumetric properties have to be averages of the pure 
        components properties and derivatives. A Multiphase will be needed to
        allow flashes with different properties from different phases.
        '''
        
        
        self.VaporPressures = VaporPressures
        self.Psats_locked = all(i.locked for i in VaporPressures) if VaporPressures is not None else False
        if self.Psats_locked:
            Psats_data = [[i.best_fit_Tmin for i in VaporPressures],
                               [i.best_fit_Tmin_slope for i in VaporPressures],
                               [i.best_fit_Tmin_value for i in VaporPressures],
                               [i.best_fit_Tmax for i in VaporPressures],
                               [i.best_fit_Tmax_slope for i in VaporPressures],
                               [i.best_fit_Tmax_value for i in VaporPressures],
                               [i.best_fit_coeffs for i in VaporPressures],
                               [i.best_fit_d_coeffs for i in VaporPressures],
                               [i.best_fit_d2_coeffs for i in VaporPressures],
                               [i.DIPPR101_ABC for i in VaporPressures]]
            if Psat_extrpolation == 'AB':
                Psats_data.append([i.best_fit_AB_high_ABC_compat + (0.0,) for i in VaporPressures])
            elif Psat_extrpolation == 'ABC':
                Psats_data.append([i.DIPPR101_ABC_high for i in VaporPressures])
            # Other option: raise?
            self.Psats_data = Psats_data
            
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
                                 [i.best_fit_d2_coeffs for i in VolumeLiquids],
                                 [i.best_fit_Tmin_quadratic for i in VolumeLiquids],
                                 )
#            low_fits = self.Vms_sat_data[9]
#            for i in self.cmps:
#                low_fits[i][0] = max(0, low_fits[i][0])
        
            
        self.incompressible = not use_Tait
        self.use_Tait = use_Tait
        if self.use_Tait:
            Tait_B_data, Tait_C_data = [[] for i in range(9)], [[] for i in range(9)]
            for v in VolumeLiquids:
                for (d, store) in zip(v.Tait_data(), [Tait_B_data, Tait_C_data]):
                    for i in range(len(d)):
                        store[i].append(d[i])
            self.Tait_B_data = Tait_B_data
            self.Tait_C_data = Tait_C_data
            
        
        self.EnthalpyVaporizations = EnthalpyVaporizations
        self.Hvap_locked = all(i.locked for i in EnthalpyVaporizations) if EnthalpyVaporizations is not None else False
        if self.Hvap_locked:
            self.Hvap_data = ([i.best_fit_Tmin for i in EnthalpyVaporizations],
                              [i.best_fit_Tmax for i in EnthalpyVaporizations],
                              [i.best_fit_Tc for i in EnthalpyVaporizations],
                              [1.0/i.best_fit_Tc for i in EnthalpyVaporizations],
                              [i.best_fit_coeffs for i in EnthalpyVaporizations])
        
        
        
        if GibbsExcessModel is None:
            GibbsExcessModel = IdealSolution(T=T, xs=zs)
        
        self.GibbsExcessModel = GibbsExcessModel
        self.eos_pure_instances = eos_pure_instances
#        self.VolumeLiquidMixture = VolumeLiquidMixture
        
        self.use_IG_Cp = use_IG_Cp
        self.use_Poynting = use_Poynting
        self.use_phis_sat = use_phis_sat
        
        if henry_components is None:
            henry_components = [False]*self.N
        self.has_henry_components = any(henry_components)
        self.henry_components = henry_components
        self.henry_data = henry_data

        self.composition_independent = isinstance(GibbsExcessModel, IdealSolution) and not self.has_henry_components
        
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
        
        self.transfer_data(new, zs, T, T_equal)
        return new


    def to_zs_TPV(self, zs, T=None, P=None, V=None):
        try:
            T_equal = T == self.T
        except:
            T_equal = False
        
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.N = self.N
        new.cmps = self.cmps
        
        if T is not None:
            if P is not None:
                new.T = T
                new.P = P
            elif V is not None:
                def to_solve(P):
                    return self.to_TP_zs(T, P, zs).V() - V
                P = secant(to_solve, 0.0002, xtol=1e-8, ytol=1e-10)
                new.P = P
        elif P is not None and V is not None:
            def to_solve(T):
                return self.to_TP_zs(T, P, zs).V() - V
            T = secant(to_solve, 300, xtol=1e-9, ytol=1e-5)
            new.T = T
        else:
            raise ValueError("Two of T, P, or V are needed")
        
        self.transfer_data(new, zs, T, T_equal)
        return new
    
    to = to_zs_TPV
    
    def transfer_data(self, new, zs, T, T_equal):
        new.VaporPressures = self.VaporPressures
        new.VolumeLiquids = self.VolumeLiquids
        new.eos_pure_instances = self.eos_pure_instances
        new.HeatCapacityGases = self.HeatCapacityGases
        new.EnthalpyVaporizations = self.EnthalpyVaporizations
        new.HeatCapacityLiquids = self.HeatCapacityLiquids
        
                
        new.Psats_locked = self.Psats_locked
        new.Psats_data = self.Psats_data
        
        new.Cpgs_locked = self.Cpgs_locked
        new.Cpgs_data = self.Cpgs_data
        
        new.Cpls_locked = self.Cpls_locked
        new.Cpls_data = self.Cpls_data
                
        new.Vms_sat_locked = self.Vms_sat_locked
        new.Vms_sat_data = self.Vms_sat_data
        
        new.Hvap_data = self.Hvap_data
        new.Hvap_locked = self.Hvap_locked
        
        new.incompressible = self.incompressible
        
        new.use_phis_sat = self.use_phis_sat
        new.use_Poynting = self.use_Poynting
        new.P_DEPENDENT_H_LIQ = self.P_DEPENDENT_H_LIQ
        new.use_IG_Cp = self.use_IG_Cp

        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        
        new.henry_data = self.henry_data
        new.henry_components = self.henry_components
        new.has_henry_components = self.has_henry_components
        
        new.composition_independent = self.composition_independent
        
        new.use_Tait = self.use_Tait
        new.Tait_B_data = self.Tait_B_data
        new.Tait_C_data = self.Tait_C_data
        
        
        if T_equal and (self.composition_independent or self.zs is zs):
            # Allow the composition inconsistency as it is harmless
            new.GibbsExcessModel = self.GibbsExcessModel
        else:
            new.GibbsExcessModel = self.GibbsExcessModel.to_T_xs(T=T, xs=zs)
        
        try:
            if T_equal:
                if not self.has_henry_components:
                    try:
                        new._Psats = self._Psats
                        new._dPsats_dT = self._dPsats_dT
                        new._d2Psats_dT2 = self._d2Psats_dT2
                    except:
                        pass
                    
                try:
                    new._Vms_sat = self._Vms_sat
                    new._Vms_sat_dT = self._Vms_sat_dT
                    new._d2Vms_sat_dT2 = self._d2Vms_sat_dT2
                except:
                    pass
                try:
                    new._Cpigs = self._Cpigs
                except:
                    pass
                try:
                    new._Cpig_integrals_over_T_pure = self._Cpig_integrals_over_T_pure
                except:
                    pass
                try:
                    new._Cpig_integrals_pure = self._Cpig_integrals_pure
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

    def Psats_T_ref(self):
        try:
            return self._Psats_T_ref
        except AttributeError:
            pass
        VaporPressures, cmps = self.VaporPressures, self.cmps
        T_REF_IG = self.T_REF_IG
        self._Psats_T_ref = [VaporPressures[i](T_REF_IG) for i in cmps] 
        return self._Psats_T_ref

    def Psats_at(self, T):
        if self.Psats_locked:
            return self._Psats_at_locked(T, self.Psats_data, self.cmps)
        VaporPressures = self.VaporPressures
        return [VaporPressures[i](T) for i in self.cmps] 
    
    @staticmethod
    def _Psats_at_locked(T, Psats_data, cmps):
        Psats = []
        T_inv = 1.0/T
        logT = log(T)
        Tmins, Tmaxes, coeffs = Psats_data[0], Psats_data[3], Psats_data[6]
        for i in cmps:
            if T < Tmins[i]:
                A, B, C = Psats_data[9][i]
                Psat = (A + B*T_inv + C*logT)
#                    A, B = Psats_data[9][i]
#                    Psat = (A - B*T_inv)
#                    Psat = (T - Tmins[i])*Psats_data[1][i] + Psats_data[2][i]
            elif T > Tmaxes[i]:
                A, B, C = Psats_data[10][i]
                Psat = (A + B*T_inv + C*logT)
#                A, B = Psats_data[10][i]
#                Psat = (A - B*T_inv)
#                Psat = (T - Tmaxes[i])*Psats_data[4][i] + Psats_data[5][i]
            else:
                Psat = 0.0
                for c in coeffs[i]:
                    Psat = Psat*T + c
            try:
                Psats.append(exp(Psat))
            except:
                Psats.append(1.6549840276802644e+300)
                
        return Psats

    def Psats(self):
        try:
            return self._Psats
        except AttributeError:
            pass
        T, cmps = self.T, self.cmps
        if self.Psats_locked:
            self._Psats = Psats = self._Psats_at_locked(T, self.Psats_data, cmps)
#            Psats_data = self.Psats_data
#            Tmins, Tmaxes, coeffs = Psats_data[0], Psats_data[3], Psats_data[6]
#            for i in cmps:
#                if T < Tmins[i]:
#                    A, B, C = Psats_data[9][i]
#                    Psat = (A + B*T_inv + C*logT)
##                    A, B = Psats_data[9][i]
##                    Psat = (A - B*T_inv)
##                    Psat = (T - Tmins[i])*Psats_data[1][i] + Psats_data[2][i]
#                elif T > Tmaxes[i]:
#                    Psat = (T - Tmaxes[i])*Psats_data[4][i] + Psats_data[5][i]
#                else:
#                    Psat = 0.0
#                    for c in coeffs[i]:
#                        Psat = Psat*T + c
#                Psats.append(exp(Psat))
            return Psats


        self._Psats = Psats = []
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

#    def PIP(self):
#        # Force liquid
#        return 2.0

    @staticmethod
    def _dPsats_dT_at_locked(T, Psats_data, cmps, Psats):
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        dPsats_dT = []
        Tmins, Tmaxes, dcoeffs, coeffs_low, coeffs_high = Psats_data[0], Psats_data[3], Psats_data[7], Psats_data[9], Psats_data[10]
        for i in cmps:
            if T < Tmins[i]:
#                    A, B = Psats_data[9][i]
#                    dPsat_dT = B*Tinv2*Psats[i]                    
                dPsat_dT = Psats[i]*(-coeffs_low[i][1]*Tinv2 + coeffs_low[i][2]*T_inv)
#                    dPsat_dT = Psats_data[1][i]*Psats[i]#*exp((T - Tmins[i])*Psats_data[1][i]
                                             #   + Psats_data[2][i])
            elif T > Tmaxes[i]:
                dPsat_dT = Psats[i]*(-coeffs_high[i][1]*Tinv2 + coeffs_high[i][2]*T_inv)
                
#                dPsat_dT = Psats_data[4][i]*Psats[i]#*exp((T - Tmaxes[i])
#                                                    #*Psats_data[4][i]
#                                                    #+ Psats_data[5][i])
            else:
                dPsat_dT = 0.0
                for c in dcoeffs[i]:
                    dPsat_dT = dPsat_dT*T + c
#                    v, der = horner_and_der(coeffs[i], T)
                dPsat_dT *= Psats[i]
            dPsats_dT.append(dPsat_dT)
        return dPsats_dT
    
    def dPsats_dT_at(self, T, Psats=None):
        if Psats is None:
            Psats = self.Psats_at(T)
        if self.Psats_locked:
            return self._dPsats_dT_at_locked(T, self.Psats_data, self.cmps, Psats)
        return [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]

    def dPsats_dT(self):
        try:
            return self._dPsats_dT
        except:
            pass
        T, cmps = self.T, self.cmps
        # Need to reset the method because for the T bounded solver,
        # will normally get a different than prefered method as it starts
        # at the boundaries
        if self.Psats_locked:
            try:
                Psats = self._Psats
            except AttributeError:
                Psats = self.Psats()
            self._dPsats_dT = dPsats_dT = self._dPsats_dT_at_locked(T, self.Psats_data, cmps, Psats)
            return dPsats_dT

        self._dPsats_dT = dPsats_dT = [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]
        return dPsats_dT
    
    def d2Psats_dT2(self):
        try:
            return self._d2Psats_dT2
        except:
            pass
        try:
            Psats = self._Psats
        except AttributeError:
            Psats = self.Psats()
        try:
            dPsats_dT = self._dPsats_dT
        except AttributeError:
            dPsats_dT = self.dPsats_dT()
        T, cmps = self.T, self.cmps
        T_inv = 1.0/T
        T_inv2 = T_inv*T_inv
        Tinv3 = T_inv*T_inv*T_inv

        self._d2Psats_dT2 = d2Psats_dT2 = []
        if self.Psats_locked:
            Psats_data = self.Psats_data
            Tmins, Tmaxes, d2coeffs = Psats_data[0], Psats_data[3], Psats_data[8]
            for i in cmps:
                if T < Tmins[i]:
#                    A, B = Psats_data[9][i]
#                    d2Psat_dT2 = B*Psats[i]*(B*T_inv - 2.0)*Tinv3
                    A, B, C = Psats_data[9][i]
                    x0 = (B*T_inv - C)
                    d2Psat_dT2 = Psats[i]*(2.0*B*T_inv - C + x0*x0)*T_inv2                    
#                    d2Psat_dT2 = Psats_data[1][i]*dPsats_dT[i]
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    x0 = (B*T_inv - C)
                    d2Psat_dT2 = Psats[i]*(2.0*B*T_inv - C + x0*x0)*T_inv2                    
#                    d2Psat_dT2 = Psats_data[4][i]*dPsats_dT[i]
                else:
                    d2Psat_dT2 = 0.0
                    for c in d2coeffs[i]:
                        d2Psat_dT2 = d2Psat_dT2*T + c
                    d2Psat_dT2 = (dPsats_dT[i]*dPsats_dT[i]/Psats[i] + Psats[i]*d2Psat_dT2)
                d2Psats_dT2.append(d2Psat_dT2)
            return d2Psats_dT2

        self._d2Psats_dT2 = d2Psats_dT2 = [VaporPressure.T_dependent_property_derivative(T=T, n=2)
                     for VaporPressure in self.VaporPressures]
        return d2Psats_dT2

    def lnPsats(self):
        try:
            return self._lnPsats
        except AttributeError:
            pass
        T, cmps = self.T, self.cmps
        T_inv = 1.0/T
        logT = log(T)
        lnPsats = []
        if self.Psats_locked:
            Psats_data = self.Psats_data
            Tmins, Tmaxes, coeffs = Psats_data[0], Psats_data[3], Psats_data[6]
            for i in cmps:
                if T < Tmins[i]:
                    A, B, C = Psats_data[9][i]
                    Psat = (A + B*T_inv + C*logT)
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    Psat = (A + B*T_inv + C*logT)
#                    Psat = (T - Tmaxes[i])*Psats_data[4][i] + Psats_data[5][i]
                else:
                    Psat = 0.0
                    for c in coeffs[i]:
                        Psat = Psat*T + c
                lnPsats.append(Psat)
            self._lnPsats = lnPsats
            return lnPsats
        self._lnPsats = [log(i) for i in self.Psats()]
        return self._lnPsats
    
    def dlnPsats_dT(self):
        T, cmps = self.T, self.cmps
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        if self.Psats_locked:
            Psats_data = self.Psats_data
            Tmins, Tmaxes, dcoeffs = Psats_data[0], Psats_data[3], Psats_data[7]
            dlnPsats_dT = []
            for i in cmps:
                if T < Tmins[i]:
                    A, B, C = Psats_data[9][i]
                    dPsat_dT = (-B*Tinv2 + C*T_inv)
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    dPsat_dT = (-B*Tinv2 + C*T_inv)
#                    dPsat_dT = Psats_data[4][i]
                else:
                    dPsat_dT = 0.0
                    for c in dcoeffs[i]:
                        dPsat_dT = dPsat_dT*T + c
                dlnPsats_dT.append(dPsat_dT)
            return dlnPsats_dT

    def d2lnPsats_dT2(self):
        T, cmps = self.T, self.cmps
        T_inv = 1.0/T
        T_inv2 = T_inv*T_inv
        Tinv3 = T_inv*T_inv*T_inv
        if self.Psats_locked:
            Psats_data = self.Psats_data
            Tmins, Tmaxes, d2coeffs = Psats_data[0], Psats_data[3], Psats_data[8]
            d2lnPsats_dT2 = []
            for i in cmps:
                if T < Tmins[i]:
                    A, B, C = Psats_data[9][i]
                    d2lnPsat_dT2 = (2.0*B*T_inv - C)*T_inv2                    
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    d2lnPsat_dT2 = (2.0*B*T_inv - C)*T_inv2                    
#                    d2lnPsat_dT2 = 0.0
                else:
                    d2lnPsat_dT2 = 0.0
                    for c in d2coeffs[i]:
                        d2lnPsat_dT2 = d2lnPsat_dT2*T + c
                d2lnPsats_dT2.append(d2lnPsat_dT2)
            return d2lnPsats_dT2

    def dPsats_dT_over_Psats(self):
        try:
            return self._dPsats_dT_over_Psats
        except AttributeError:
            pass
        T, cmps = self.T, self.cmps
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        if self.Psats_locked:
            dPsat_dT_over_Psats = []
            Psats_data = self.Psats_data
            Tmins, Tmaxes, dcoeffs, low_coeffs, high_coeffs = Psats_data[0], Psats_data[3], Psats_data[7], Psats_data[9], Psats_data[10]
            for i in cmps:
                if T < Tmins[i]:
                    dPsat_dT_over_Psat = (-low_coeffs[i][1]*Tinv2 + low_coeffs[i][2]*T_inv)
                elif T > Tmaxes[i]:
                    dPsat_dT_over_Psat = (-high_coeffs[i][1]*Tinv2 + high_coeffs[i][2]*T_inv)
#                    dPsat_dT_over_Psat = Psats_data[4][i]
                else:
                    dPsat_dT_over_Psat = 0.0
                    for c in dcoeffs[i]:
                        dPsat_dT_over_Psat = dPsat_dT_over_Psat*T + c
                dPsat_dT_over_Psats.append(dPsat_dT_over_Psat)
            self._dPsats_dT_over_Psats = dPsat_dT_over_Psats
            return dPsat_dT_over_Psats

        dPsat_dT_over_Psats = [i/j for i, j in zip(self.dPsats_dT(), self.Psats())]
        self._dPsats_dT_over_Psats = dPsat_dT_over_Psats
        return dPsat_dT_over_Psats

    def d2Psats_dT2_over_Psats(self):
        try:
            return self._d2Psats_dT2_over_Psats
        except AttributeError:
            pass
        T, cmps = self.T, self.cmps
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        Tinv4 = Tinv2*Tinv2
        c0 = (T + T)*Tinv4
        if self.Psats_locked:
            d2Psat_dT2_over_Psats = []
            Psats_data = self.Psats_data
            Tmins, Tmaxes, dcoeffs, low_coeffs, high_coeffs = Psats_data[0], Psats_data[3], Psats_data[7], Psats_data[9], Psats_data[10]
            for i in cmps:
                if T < Tmins[i]:
                    B, C = low_coeffs[i][1], low_coeffs[i][2]
                    x0 = (B - C*T)
                    d2Psat_dT2_over_Psat = c0*B - C*Tinv2 + x0*x0*Tinv4
#                    d2Psat_dT2_over_Psat = (2*B*T - C*T**2 + (B - C*T)**2)/T**4
                elif T > Tmaxes[i]:
                    B, C = high_coeffs[i][1], high_coeffs[i][2]
                    x0 = (B - C*T)
                    d2Psat_dT2_over_Psat = c0*B - C*Tinv2 + x0*x0*Tinv4
                else:
                    dPsat_dT = 0.0
                    d2Psat_dT2 = 0.0
                    for a in dcoeffs[i]:
                        d2Psat_dT2 = T*d2Psat_dT2 + dPsat_dT
                        dPsat_dT = T*dPsat_dT + a
                    d2Psat_dT2_over_Psat = dPsat_dT*dPsat_dT + d2Psat_dT2

                d2Psat_dT2_over_Psats.append(d2Psat_dT2_over_Psat)
            self._d2Psats_dT2_over_Psats = d2Psat_dT2_over_Psats
            return d2Psat_dT2_over_Psats

        d2Psat_dT2_over_Psats = [i/j for i, j in zip(self.d2Psats_dT2(), self.Psats())]
        self._d2Psats_dT2_over_Psats = d2Psat_dT2_over_Psats
        return d2Psat_dT2_over_Psats

    @staticmethod
    def _Vms_sat_at(T, Vms_sat_data, cmps):
        Tmins, Tmaxes, coeffs, coeffs_Tmin = Vms_sat_data[0], Vms_sat_data[3], Vms_sat_data[6], Vms_sat_data[9]
        Vms_sat = []
        for i in cmps:
            if T < Tmins[i]:
                Vm = 0.0
                for c in coeffs_Tmin[i]:
                    Vm = Vm*T + c
#                    Vm = (T - Tmins[i])*Vms_sat_data[1][i] + Vms_sat_data[2][i]
            elif T > Tmaxes[i]:
                Vm = (T - Tmaxes[i])*Vms_sat_data[4][i] + Vms_sat_data[5][i]
            else:
                Vm = 0.0
                for c in coeffs[i]:
                    Vm = Vm*T + c
            Vms_sat.append(Vm)
        return Vms_sat

    def Vms_sat_at(self, T):
        if self.Vms_sat_locked:
            return self._Vms_sat_at(T, self.Vms_sat_data, self.cmps)
        VolumeLiquids = self.VolumeLiquids
        return [VolumeLiquids[i].T_dependent_property(T) for i in self.cmps]

    def Vms_sat(self):
        try:
            return self._Vms_sat
        except AttributeError:
            pass
        T = self.T
        if self.Vms_sat_locked:
#            self._Vms_sat = evaluate_linear_fits(self.Vms_sat_data, T)
#            return self._Vms_sat
            self._Vms_sat = Vms_sat = self._Vms_sat_at(T, self.Vms_sat_data, self.cmps)
            return Vms_sat
        
        VolumeLiquids = self.VolumeLiquids
#        Psats = self.Psats()
#        self._Vms_sat = [VolumeLiquids[i](T, Psats[i]) for i in self.cmps]
        self._Vms_sat = [VolumeLiquids[i].T_dependent_property(T) for i in self.cmps]
        return self._Vms_sat

    @staticmethod
    def _dVms_sat_dT_at(T, Vms_sat_data, cmps):
        Vms_sat_data = Vms_sat_data
        Vms_sat_dT = []
        Tmins, Tmaxes, dcoeffs = Vms_sat_data[0], Vms_sat_data[3], Vms_sat_data[7]
        for i in cmps:
            if T < Tmins[i]:
                dVm = horner_and_der2(Vms_sat_data[9][i], T)[1]
            elif T > Tmaxes[i]:
                dVm = Vms_sat_data[4][i]
            else:
                dVm = 0.0
                for c in dcoeffs[i]:
                    dVm = dVm*T + c
            Vms_sat_dT.append(dVm)
        return Vms_sat_dT

    def dVms_sat_dT_at(self, T):
        if self.Vms_sat_locked:
            return self._dVms_sat_dT_at(T, self.Vms_sat_data, self.cmps)
        return [obj.T_dependent_property_derivative(T=T) for obj in VolumeLiquids]

    def dVms_sat_dT(self):
        try:
            return self._Vms_sat_dT
        except:
            pass
        T = self.T

        if self.Vms_sat_locked:
#            self._Vms_sat_dT = evaluate_linear_fits_d(self.Vms_sat_data, T)
            self._Vms_sat_dT = self._dVms_sat_dT_at(T, self.Vms_sat_data, self.cmps)
            return self._Vms_sat_dT

        VolumeLiquids = self.VolumeLiquids
        self._Vms_sat_dT = Vms_sat_dT = [obj.T_dependent_property_derivative(T=T) for obj in VolumeLiquids]
        return Vms_sat_dT

    def d2Vms_sat_dT2(self):
        try:
            return self._d2Vms_sat_dT2
        except:
            pass

        T = self.T
        
        if self.Vms_sat_locked:
#            self._d2Vms_sat_dT2 = evaluate_linear_fits_d2(self.Vms_sat_data, T)
#            return self._d2Vms_sat_dT2
            d2Vms_sat_dT2 = self._d2Vms_sat_dT2 = []
        
            Vms_sat_data = self.Vms_sat_data
            Tmins, Tmaxes, d2coeffs = Vms_sat_data[0], Vms_sat_data[3], Vms_sat_data[8]
            for i in self.cmps:
                d2Vm = 0.0
                if Tmins[i] < T < Tmaxes[i]:
                    for c in d2coeffs[i]:
                        d2Vm = d2Vm*T + c
                elif T < Tmins[i]:
                    d2Vm = horner_and_der2(Vms_sat_data[9][i], T)[2]
                d2Vms_sat_dT2.append(d2Vm)
            return d2Vms_sat_dT2

        VolumeLiquids = self.VolumeLiquids
        self._d2Vms_sat_dT2 = [obj.T_dependent_property_derivative(T=T, order=2) for obj in VolumeLiquids]
        return self._d2Vms_sat_dT2

    def Vms_sat_T_ref(self):
        try:
            return self._Vms_sat_T_ref
        except AttributeError:
            pass
        T_REF_IG = self.T_REF_IG
        if self.Vms_sat_locked:
            self._Vms_sat_T_ref = evaluate_linear_fits(self.Vms_sat_data, T_REF_IG)
        else:
            VolumeLiquids, cmps = self.VolumeLiquids, self.cmps
            self._Vms_sat_T_ref = [VolumeLiquids[i].T_dependent_property(T_REF_IG) for i in cmps] 
        return self._Vms_sat_T_ref

    def dVms_sat_dT_T_ref(self):
        try:
            return self._dVms_sat_dT_T_ref
        except AttributeError:
            pass
        T_REF_IG = self.T_REF_IG
        if self.Vms_sat_locked:
            self._dVms_sat_dT_T_ref = evaluate_linear_fits_d(self.Vms_sat_data, T)
        else:
            VolumeLiquids, cmps = self.VolumeLiquids, self.cmps
            self._dVms_sat_dT_T_ref = [VolumeLiquids[i].T_dependent_property_derivative(T_REF_IG) for i in cmps] 
        return self._dVms_sat_dT_T_ref

    def Vms(self):
        # Fill in tait/eos function to be called instead of Vms_sat
        return self.Vms_sat()

    def dVms_dT(self):
        return self.dVms_sat_dT()
    
    def d2Vms_dT2(self):
        return self.d2Vms_sat_dT2()

    def dVms_dP(self):
        return [0.0]*self.N

    def d2Vms_dP2(self):
        return [0.0]*self.N

    def d2Vms_dPdT(self):
        return [0.0]*self.N

    def Hvaps(self):
        try:
            return self._Hvaps
        except AttributeError:
            pass
        T, EnthalpyVaporizations, cmps = self.T, self.EnthalpyVaporizations, self.cmps

        self._Hvaps = Hvaps = []
        if self.Hvap_locked:
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

    def Hvaps_T_ref(self):
        try:
            return self._Hvaps_T_ref
        except AttributeError:
            pass
        EnthalpyVaporizations, cmps = self.EnthalpyVaporizations, self.cmps
        T_REF_IG = self.T_REF_IG
        self._Hvaps_T_ref = [EnthalpyVaporizations[i](T_REF_IG) for i in cmps] 
        return self._Hvaps_T_ref
    
    def Poyntings_at(self, T, P, Psats=None, Vms=None):
        if not self.use_Poynting:
            return [1.0]*self.N
        
        cmps = self.cmps
        if Psats is None:
            Psats = self.Psats_at(T)
        if Vms is None:
            Vms = self.Vms_sat_at(T)
        RT_inv = 1.0/(R*T)
        return [exp(Vms[i]*(P-Psats[i])*RT_inv) for i in cmps]

    def Poyntings(self):
        try:
            return self._Poyntings
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._Poyntings = [1.0]*self.N
            return self._Poyntings
        
        T, P = self.T, self.P
        try:
            Psats = self._Psats
        except AttributeError:
            Psats = self.Psats()
        try:
            Vms_sat = self._Vms_sat
        except AttributeError:
            Vms_sat = self.Vms_sat()
        RT_inv = 1.0/(R*T)
        self._Poyntings = [exp(Vml*(P-Psat)*RT_inv) for Psat, Vml in zip(Psats, Vms_sat)]
        return self._Poyntings

    
    def dPoyntings_dT(self):
        try:
            return self._dPoyntings_dT
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._dPoyntings_dT = [0.0]*self.N
            return self._dPoyntings_dT
        
        T, P = self.T, self.P
            
        Psats = self.Psats()
        dPsats_dT = self.dPsats_dT()
        Vms = self.Vms_sat()
        dVms_sat_dT = self.dVms_sat_dT()
        
        x0 = 1.0/R
        x1 = 1.0/T
        RT_inv = x0*x1
        
        self._dPoyntings_dT = dPoyntings_dT = []
        for i in self.cmps:
            x2 = Vms[i]
            x3 = Psats[i]
            
            x4 = P - x3
            x5 = x1*x2*x4
            dPoyntings_dTi = -RT_inv*(x2*dPsats_dT[i] - x4*dVms_sat_dT[i] + x5)*exp(x0*x5)
            dPoyntings_dT.append(dPoyntings_dTi)
        return dPoyntings_dT

    def dPoyntings_dT_at(self, T, P, Psats=None, Vms=None, dPsats_dT=None, dVms_sat_dT=None):
        if not self.use_Poynting:
            return [0.0]*self.N
        
        cmps = self.cmps
        if Psats is None:
            Psats = self.Psats_at(T)
            
        if dPsats_dT is None:
            dPsats_dT = self.dPsats_dT_at(T, Psats)
            
        if Vms is None:
            Vms = self.Vms_sat_at(T)
            
        if dVms_sat_dT is None:
            dVms_sat_dT = self.dVms_sat_dT_at(T)
        x0 = 1.0/R
        x1 = 1.0/T
        dPoyntings_dT = []
        for i in self.cmps:
            x2 = Vms[i]            
            x4 = P - Psats[i]
            x5 = x1*x2*x4
            dPoyntings_dTi = -x0*x1*(x2*dPsats_dT[i] - x4*dVms_sat_dT[i] + x5)*exp(x0*x5)
            dPoyntings_dT.append(dPoyntings_dTi)
        return dPoyntings_dT

    def d2Poyntings_dT2(self):
        try:
            return self._d2Poyntings_dT2
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._d2Poyntings_dT2 = [0.0]*self.N
            return self._d2Poyntings_dT2

        T, P = self.T, self.P
            
        Psats = self.Psats()
        dPsats_dT = self.dPsats_dT()
        d2Psats_dT2 = self.d2Psats_dT2()
        Vms = self.Vms_sat()
        dVms_sat_dT = self.dVms_sat_dT()
        d2Vms_sat_dT2 = self.d2Vms_sat_dT2()

        x6 = 1.0/T
        x7 = x6 + x6
        x11 = 1.0/R
        x12 = x11*x6
        c0 = 2.0*x6*x6

        self._d2Poyntings_dT2 = d2Poyntings_dT2 = []
        '''
        from sympy import *
        R, T, P = symbols('R, T, P')
        Vml, Psat = symbols('Vml, Psat', cls=Function)
        RT_inv = 1/(R*T)
        Poy = exp(Vml(T)*(P-Psat(T))*RT_inv)
        cse(diff(Poy, T, 2), optimizations='basic')
        '''
        for i in self.cmps:
            x0 = Vms[i]
            x1 = Psats[i]
            x2 = P - x1
            x3 = x0*x2
            x4 = dPsats_dT[i]
            x5 = x0*x4
            x8 = dVms_sat_dT[i]
            x9 = x2*x8
            x10 = x3*x6
            x50 = (x10 + x5 - x9)
            d2Poyntings_dT2i = (x12*(-x0*d2Psats_dT2[i] + x12*x50*x50
                                    + x2*d2Vms_sat_dT2[i] - 2.0*x4*x8 + x5*x7 
                                    - x7*x9 + x3*c0)*exp(x10*x11))
            d2Poyntings_dT2.append(d2Poyntings_dT2i)
        return d2Poyntings_dT2
    
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
        
        Vms = self.Vms_sat()
        
        self._dPoyntings_dP = dPoyntings_dPs = []
        for i in self.cmps:
            x0 = Vms[i]/(R*T)
            dPoyntings_dPs.append(x0*exp(x0*(P - Psats[i])))
        return dPoyntings_dPs

    def d2Poyntings_dPdT(self):
        '''
        from sympy import *
        R, T, P = symbols('R, T, P')
        Vml, Psat = symbols('Vml, Psat', cls=Function)
        RT_inv = 1/(R*T)
        Poy = exp(Vml(T)*(P-Psat(T))*RT_inv)
        Poyf = symbols('Poyf')
        cse(diff(Poy, T, P).subs(Poy, Poyf), optimizations='basic')
        '''
        try:
            return self._d2Poyntings_dPdT
        except AttributeError:
            pass
        if not self.use_Poynting:
            self._d2Poyntings_dPdT = [0.0]*self.N
            return self._d2Poyntings_dPdT
        
        try:
            Psats = self._Psats
        except AttributeError:
            Psats = self.Psats()
        try:
            dPsats_dT = self._dPsats_dT
        except AttributeError:
            dPsats_dT = self.dPsats_dT()
        try:
            Vms = self._Vms_sat
        except AttributeError:
            Vms = self.Vms_sat()
        try:
            dVms_sat_dT = self._dVms_sat_dT
        except AttributeError:
            dVms_sat_dT = self.dVms_sat_dT()
        try:
            Poyntings = self._Poyntings
        except AttributeError:
            Poyntings = self.Poyntings()
        
        x0 = R_inv
        x1 = 1.0/self.T
        P = self.P
        nRT_inv = -x0*x1
        self._d2Poyntings_dPdT = d2Poyntings_dPdT = []
        for i in self.cmps:
            x2 = Vms[i]
            x3 = x1*x2
            x4 = dVms_sat_dT[i]
            x5 = Psats[i]
            x6 = P - x5
            v = Poyntings[i]*nRT_inv*(x0*x3*(x2*dPsats_dT[i] + x3*x6 - x4*x6) + x3 - x4)
            d2Poyntings_dPdT.append(v)
        return d2Poyntings_dPdT

        
    d2Poyntings_dTdP = d2Poyntings_dPdT

    def phis_sat_at(self, T):
        if not self.use_phis_sat:
            return [1.0]*self.N
        return [i.phi_sat(T, polish=True) for i in self.eos_pure_instances]
        
    def phis_sat(self):
        try:
            return self._phis_sat
        except AttributeError:
            pass
        # Goal: Have the polynomial here. Fitting specific to the compound is required.
    
        if not self.use_phis_sat:
            self._phis_sat = [1.0]*self.N
            return self._phis_sat
        
        T = self.T
        self._phis_sat = [i.phi_sat(T, polish=True) for i in self.eos_pure_instances]
        return self._phis_sat




    def dphis_sat_dT_at(self, T):
        if not self.use_phis_sat:
            return [0.0]*self.N
        return [i.dphi_sat_dT(T) for i in self.eos_pure_instances]
                
    def dphis_sat_dT(self):
        try:
            return self._dphis_sat_dT
        except AttributeError:
            pass

        if not self.use_phis_sat:
            self._dphis_sat_dT = [0.0]*self.N
            return self._dphis_sat_dT

        T = self.T
        self._dphis_sat_dT = [i.dphi_sat_dT(T) for i in self.eos_pure_instances]
        return self._dphis_sat_dT
    
    def d2phis_sat_dT2(self):
        # Numerically implemented
        try:
            return self._d2phis_sat_dT2
        except AttributeError:
            pass
        if not self.use_phis_sat:
            self._d2phis_sat_dT2 = [0.0]*self.N
            return self._d2phis_sat_dT2

        T = self.T
        self._d2phis_sat_dT2 = [i.d2phi_sat_dT2(T) for i in self.eos_pure_instances]
        return self._d2phis_sat_dT2


    def phis_at(self, T, P, zs, Psats=None, gammas=None, phis_sat=None, Poyntings=None):
        P_inv = 1.0/P
        if Psats is None:
            Psats = self.Psats_at(T)
        if gammas is None:
            gammas = self.gammas_at(T, zs)
        if phis_sat is None:
            phis_sat = self.phis_sat_at(T)
        if Poyntings is None:
            Poyntings = self.Poyntings_at(T, P, Psats=Psats)
        return [gammas[i]*Psats[i]*Poyntings[i]*phis_sat[i]*P_inv
                for i in self.cmps]

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
    
    lnphis_G_min = lnphis
        
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
#    def dphis_dxs(self):
#        if
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
            dPoyntings_dT = self.dPoyntings_dT() #[0.0]*self.N
            Poyntings = self.Poyntings()
        else:
            dPoyntings_dT = [0.0]*self.N
            Poyntings = [1.0]*self.N

        dPsats_dT = self.dPsats_dT()
        
        dgammas_dT = self.GibbsExcessModel.dgammas_dT()
        
        if self.use_phis_sat:
            dphis_sat_dT = self.dphis_sat_dT()
            phis_sat = self.phis_sat()
        else:
            dphis_sat_dT = [0.0]*self.N
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
            v = (x0*x4*dphis_sat_dT[i] + x1*x4*dgammas_dT[i] + x2*x5*dPoyntings_dT[i] + x3*x5*dPsats_dT[i])/P
            dphis_dTl.append(v)
        return dphis_dTl

    def dphis_dT_at(self, T, P, zs, phis_also=False):
        Psats = self.Psats_at(T)
        dPsats_dT = self.dPsats_dT_at(T, Psats)
        Vms = self.Vms_sat_at(T)
        dVms_sat_dT = self.dVms_sat_dT_at(T)

        gammas = self.gammas_at(T, zs)
        dgammas_dT = self.dgammas_dT_at(T, zs)
        
        if self.use_Poynting:
            Poyntings = self.Poyntings_at(T, P, Psats, Vms)
            dPoyntings_dT = self.dPoyntings_dT_at(T, P, Psats=Psats, Vms=Vms, dPsats_dT=dPsats_dT, dVms_sat_dT=dVms_sat_dT)
        else:
            Poyntings = [1.0]*self.N
            dPoyntings_dT = [0.0]*self.N
        
        if self.use_phis_sat:
            dphis_sat_dT = self.dphis_sat_dT_at(T)
            phis_sat = self.phis_sat_at(T)
        else:
            dphis_sat_dT = [0.0]*self.N
            phis_sat = [1.0]*self.N
        
        
        dphis_dT = []
        for i in self.cmps:
            x0 = gammas[i]
            x1 = phis_sat[i]
            x2 = Psats[i]
            x3 = Poyntings[i]
            x4 = x2*x3
            x5 = x0*x1
            v = (x0*x4*dphis_sat_dT[i] + x1*x4*dgammas_dT[i] + x2*x5*dPoyntings_dT[i] + x3*x5*dPsats_dT[i])/P
            dphis_dT.append(v)
        if phis_also:
            P_inv = 1.0/P
            phis = [gammas[i]*Psats[i]*Poyntings[i]*phis_sat[i]*P_inv for i in self.cmps]
            return dphis_dT, phis
        return dphis_dT
        
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
                    
    def gammas_at(self, T, zs):
        if self.composition_independent:
            return [1.0]*self.N
        return self.GibbsExcessModel.to_T_xs(T, xs).gammas()

    def dgammas_dT_at(self, T, zs):
        if self.composition_independent:
            return [0.0]*self.N
        return self.GibbsExcessModel.to_T_xs(T, zs).dgammas_dT()

    def gammas(self):
        try:
            return self.GibbsExcessModel._gammas
        except AttributeError:
            return self.GibbsExcessModel.gammas()

    def dgammas_dT(self):
        return self.GibbsExcessModel.dgammas_dT()
        
    def H_old(self):
#        try:
#            return self._H
#        except AttributeError:
#            pass
        # Untested
        T = self.T
        RT = R*T
        P = self.P
        zs, cmps = self.zs, self.cmps
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
            
            if self.use_IG_Cp:
                try:
                    Psats = self._Psats
                except AttributeError:
                    Psats = self.Psats()
                try:
                    dPsats_dT = self._dPsats_dT
                except AttributeError:
                    dPsats_dT = self.dPsats_dT()
                try:
                    Vms_sat = self._Vms_sat
                except AttributeError:
                    Vms_sat = self.Vms_sat()
                try:
                    dVms_sat_dT = self._Vms_sat_dT
                except AttributeError:
                    dVms_sat_dT = self.dVms_sat_dT()
                
                failed_dPsat_dT = False
                try:
                    H = 0.0
                    for i in cmps:
                        dV_vap = R*T/Psats[i] - Vms_sat[i]
    #                    print( R*T/Psats[i] , Vms_sat[i])
                        # ratio of der to value might be easier?
                        dS_vap = dPsats_dT[i]*dV_vap
    #                    print(dPsats_dT[i]*dV_vap)
                        Hvap = T*dS_vap
                        H += zs[i]*(Cpig_integrals_pure[i] - Hvap)
                except ZeroDivisionError:
                    failed_dPsat_dT = True
                
                if failed_dPsat_dT or isinf(H):
                    # Handle the case where vapor pressure reaches zero - needs special implementations
                    dPsats_dT_over_Psats = self.dPsats_dT_over_Psats() 
                    H = 0.0
                    for i in cmps:
#                        dV_vap = R*T/Psats[i] - Vms_sat[i]
#                        dS_vap = dPsats_dT[i]*dV_vap
                        Hvap = T*dPsats_dT_over_Psats[i]*RT
                        H += zs[i]*(Cpig_integrals_pure[i] - Hvap)

                if self.use_Tait:
                    dH_dP_integrals_Tait = self.dH_dP_integrals_Tait()
                    for i in cmps:
                        H += zs[i]*dH_dP_integrals_Tait[i]
                elif self.use_Poynting:
                    for i in cmps:
                        # This bit is the differential with respect to pressure
#                        dP = max(0.0, P - Psats[i]) # Breaks thermodynamic consistency
                        dP = P - Psats[i]
                        H += zs[i]*dP*(Vms_sat[i] - T*dVms_sat_dT[i])
            else:
                Psats = self.Psats()
                Vms_sat = self.Vms_sat()
                dVms_sat_dT = self.dVms_sat_dT()
                dPsats_dT = self.dPsats_dT()
                Hvaps_T_ref = self.Hvaps_T_ref()
                Cpl_integrals_pure = self.Cpl_integrals_pure()
                dVms_sat_dT_T_ref = self.dVms_sat_dT_T_ref()
                Vms_sat_T_ref = self.Vms_sat_T_ref()
                Psats_T_ref = self.Psats_T_ref()
                
                Hvaps = self.Hvaps()
                
                H = 0.0
                for i in self.cmps:
                    H += zs[i]*(Cpl_integrals_pure[i] - Hvaps_T_ref[i]) # 
                    # If we can use the liquid heat capacity and prove its consistency
                    
                    # This bit is the differential with respect to pressure
                    dP = P - Psats_T_ref[i]
                    H += zs[i]*dP*(Vms_sat_T_ref[i] - T_REF_IG*dVms_sat_dT_T_ref[i])
        else:
            Hvaps = self.Hvaps()
            for i in self.cmps:
                H += zs[i]*(Cpig_integrals_pure[i] - Hvaps[i]) 
        H += self.GibbsExcessModel.HE()
#        self._H = H
        return H

    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        T = self.T
        nRT2 = -R*T*T
        zs, cmps = self.zs, self.cmps
        try:
            Cpig_integrals_pure = self._Cpig_integrals_pure
        except AttributeError:
            Cpig_integrals_pure = self.Cpig_integrals_pure()
                    
#        try:
#            Psats = self._Psats
#        except AttributeError:
#            Psats = self.Psats()
#        try:
#            dPsats_dT = self._dPsats_dT
#        except AttributeError:
#            dPsats_dT = self.dPsats_dT()
        dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
        use_Poynting, use_phis_sat = self.use_Poynting, self.use_phis_sat
        
        if use_Poynting:
            try:
                Poyntings = self._Poyntings
            except AttributeError:
                Poyntings = self.Poyntings()
            try:
                dPoyntings_dT = self._dPoyntings_dT
            except AttributeError:
                dPoyntings_dT = self.dPoyntings_dT()        
        if use_phis_sat:
            try:
                dphis_sat_dT = self._dphis_sat_dT
            except AttributeError:
                dphis_sat_dT = self.dphis_sat_dT()
            try:
                phis_sat = self._phis_sat
            except AttributeError:
                phis_sat = self.phis_sat()

        H = 0.0
        if use_Poynting and use_phis_sat:
            for i in cmps:
                H += zs[i]*(nRT2*(dphis_sat_dT[i]/phis_sat[i] + dPsats_dT_over_Psats[i] + dPoyntings_dT[i]/Poyntings[i]) 
                            + Cpig_integrals_pure[i])
        elif use_Poynting:
            for i in cmps:
                H += zs[i]*(nRT2*(dPsats_dT_over_Psats[i] + dPoyntings_dT[i]/Poyntings[i]) + Cpig_integrals_pure[i])
        elif use_phis_sat:
            for i in cmps:
                H += zs[i]*(nRT2*(dPsats_dT_over_Psats[i] + dphis_sat_dT[i]/phis_sat[i]) + Cpig_integrals_pure[i])
        else:
            for i in cmps:
                H += zs[i]*(nRT2*dPsats_dT_over_Psats[i] + Cpig_integrals_pure[i])
        
        if not self.composition_independent:
            H += self.GibbsExcessModel.HE()
        self._H = H
        return H
            
    def S_old(self):
#        try:
#            return self._S
#        except AttributeError:
#            pass
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
            S -= zs[i]*log_zs[i]
        S *= R
        S_base = S
        
        T_inv = 1.0/T
        RT = R*T
        
        P_REF_IG_INV = self.P_REF_IG_INV
        
        try:
            Cpig_integrals_over_T_pure = self._Cpig_integrals_over_T_pure
        except AttributeError:
            Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        try:
            Psats = self._Psats
        except AttributeError:
            Psats = self.Psats()
        try:
            dPsats_dT = self._dPsats_dT
        except AttributeError:
            dPsats_dT = self.dPsats_dT()
        try:
            Vms_sat = self._Vms_sat
        except AttributeError:
            Vms_sat = self.Vms_sat()
        try:
            dVms_sat_dT = self._Vms_sat_dT
        except AttributeError:
            dVms_sat_dT = self.dVms_sat_dT()
        
        if self.P_DEPENDENT_H_LIQ:
            if self.use_IG_Cp:
                failed_dPsat_dT = False
                try:
                    for i in self.cmps:
                        dSi = Cpig_integrals_over_T_pure[i] 
                        dVsat = R*T/Psats[i] - Vms_sat[i]
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
    #                    dP = max(0.0, P - Psats[i])
        #                if dP > 0.0:
                        # I believe should include effect of pressure on all components, regardless of phase
                        dSi -= dP*dVms_sat_dT[i]
                        S += dSi*zs[i]
                except (ZeroDivisionError, ValueError):
                    # Handle the zero division on Psat or the log getting two small
                    failed_dPsat_dT = True
                
                if failed_dPsat_dT or isinf(S):
                    S = S_base
                    # Handle the case where vapor pressure reaches zero - needs special implementations
                    dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
                    lnPsats = self.lnPsats()
                    LOG_P_REF_IG = self.LOG_P_REF_IG
                    for i in cmps:
                        dSi = Cpig_integrals_over_T_pure[i] 
                        dSvap = RT*dPsats_dT_over_Psats[i]
                        dSi -= dSvap
                        dSi -= R*(lnPsats[i] - LOG_P_REF_IG)#   trunc_log(Psats[i]*P_REF_IG_INV)
                        dSi -= P*dVms_sat_dT[i]
                        S += dSi*zs[i]
                
                if self.use_Tait:
                    pass
                elif self.use_Poynting:
                    pass
#                for i in cmps:
                    
            else:
                # mine
                Hvaps_T_ref = self.Hvaps_T_ref()
                Psats_T_ref = self.Psats_T_ref()
                Cpl_integrals_over_T_pure = self.Cpl_integrals_over_T_pure()
                T_REF_IG_INV = self.T_REF_IG_INV
                dVms_sat_dT_T_ref = self.dVms_sat_dT_T_ref()
                Vms_sat_T_ref = self.Vms_sat_T_ref()
                
                for i in self.cmps:
                    dSi = Cpl_integrals_over_T_pure[i] 
                    dSi -= Hvaps_T_ref[i]*T_REF_IG_INV
                    # Take each component to its reference state change - saturation pressure
                    dSi -= R*log(Psats_T_ref[i]*P_REF_IG_INV)
                    # I believe should include effect of pressure on all components, regardless of phase


                    dP = P - Psats_T_ref[i]
                    dSi -= dP*dVms_sat_dT_T_ref[i]
                    S += dSi*zs[i]
#                else:
#                    # COCO
#                    Hvaps = self.Hvaps()
#                    Psats_T_ref = self.Psats_T_ref()
#                    Cpl_integrals_over_T_pure = self.Cpl_integrals_over_T_pure()
#                    T_REF_IG_INV = self.T_REF_IG_INV
#                    
#                    for i in self.cmps:
#                        dSi = -Cpl_integrals_over_T_pure[i] 
#                        dSi -= Hvaps[i]/T
#                        # Take each component to its reference state change - saturation pressure
#                        dSi -= R*log(Psats[i]*P_REF_IG_INV)
#                        
#                        dP = P - Psats[i]
#                        # I believe should include effect of pressure on all components, regardless of phase
#                        dSi -= dP*dVms_sat_dT[i]
#                        S += dSi*zs[i]
        else:
            Hvaps = self.Hvaps()
            for i in cmps:
                Sg298_to_T = Cpig_integrals_over_T_pure[i]
                Svap = -Hvaps[i]*T_inv # Do the transition at the temperature of the liquid
                S += zs[i]*(Sg298_to_T + Svap - R*log(P*P_REF_IG_INV)) # 
#        self._S = 
        S = S + self.GibbsExcessModel.SE()
        return S

    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        T, P = self.T, self.P
        P_inv = 1.0/P
        zs, cmps = self.zs, self.cmps
        
        log_zs = self.log_zs()
        S = 0.0
        for i in cmps:
            S -= zs[i]*log_zs[i]
        S -= log(P*self.P_REF_IG_INV)
        S *= R
        try:
            Cpig_integrals_over_T_pure = self._Cpig_integrals_over_T_pure
        except AttributeError:
            Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()

        try:
            lnPsats = self._lnPsats
        except AttributeError:
            lnPsats = self.lnPsats()

        dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
        use_Poynting, use_phis_sat = self.use_Poynting, self.use_phis_sat
        
        if use_Poynting:
            try:
                Poyntings = self._Poyntings
            except AttributeError:
                Poyntings = self.Poyntings()
            try:
                dPoyntings_dT = self._dPoyntings_dT
            except AttributeError:
                dPoyntings_dT = self.dPoyntings_dT()        
        if use_phis_sat:
            try:
                dphis_sat_dT = self._dphis_sat_dT
            except AttributeError:
                dphis_sat_dT = self.dphis_sat_dT()
            try:
                phis_sat = self._phis_sat
            except AttributeError:
                phis_sat = self.phis_sat()

        if use_Poynting and use_phis_sat:
            for i in cmps:
                S -= zs[i]*(R*(T*(dphis_sat_dT[i]/phis_sat[i] + dPsats_dT_over_Psats[i] + dPoyntings_dT[i]/Poyntings[i])
                            + lnPsats[i] + log(Poyntings[i]*phis_sat[i]*P_inv)) - Cpig_integrals_over_T_pure[i])
        elif use_Poynting:
            for i in cmps:
                S -= zs[i]*(R*(T*(dPsats_dT_over_Psats[i] + dPoyntings_dT[i]/Poyntings[i])
                            + lnPsats[i] + log(Poyntings[i]*P_inv)) - Cpig_integrals_over_T_pure[i])
        elif use_phis_sat:
            for i in cmps:
                S -= zs[i]*(R*(T*(dphis_sat_dT[i]/phis_sat[i] + dPsats_dT_over_Psats[i])
                            + lnPsats[i] + log(phis_sat[i]*P_inv)) - Cpig_integrals_over_T_pure[i])
        else:
            logP_inv = log(P_inv)
            for i in cmps:
                S -= zs[i]*(R*(T*dPsats_dT_over_Psats[i] + lnPsats[i] + logP_inv)
                            - Cpig_integrals_over_T_pure[i])

        if not self.composition_independent:
            S += self.GibbsExcessModel.SE()
        self._S = S
        return S

    def Cp_old(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        # Needs testing
        T, P, P_DEPENDENT_H_LIQ = self.T, self.P, self.P_DEPENDENT_H_LIQ
        Cp, zs = 0.0, self.zs
        Cpigs_pure = self.Cpigs_pure()
        if P_DEPENDENT_H_LIQ:
            try:
                Psats = self._Psats
            except AttributeError:
                Psats = self.Psats()
            try:
                dPsats_dT = self._dPsats_dT
            except AttributeError:
                dPsats_dT = self.dPsats_dT()
            try:
                d2Psats_dT2 = self._d2Psats_dT2
            except AttributeError:
                d2Psats_dT2 = self.d2Psats_dT2()
            try:
                Vms_sat = self._Vms_sat
            except AttributeError:
                Vms_sat = self.Vms_sat()
            try:
                dVms_sat_dT = self._Vms_sat_dT
            except AttributeError:
                dVms_sat_dT = self.dVms_sat_dT()
            try:
                d2Vms_sat_dT2 = self._d2Vms_sat_dT2
            except AttributeError:
                d2Vms_sat_dT2 = self.d2Vms_sat_dT2()

            failed_dPsat_dT = False
            try:
                for i in self.cmps:
                    x0 = Psats[i]
                    Psat_inv = 1.0/x0
                    x1 = Vms_sat[i]
                    x2 = dPsats_dT[i]
                    x3 = R*Psat_inv
                    x4 = T*x3
                    x5 = -x1
                    x6 = dVms_sat_dT[i]
                    x7 = T*x2
    #                print(#-T*(P - x0)*d2Vms_sat_dT2[i],
    #                       - T*(x4 + x5)*d2Psats_dT2[i], T, x4, x5, d2Psats_dT2[i],
                           #x2*(x1 - x4) + x2*(T*x6 + x5) - x7*(-R*x7*Psat_inv*Psat_inv + x3 - x6),
                           #Cpigs_pure[i]
    #                       )
                    Cp += zs[i]*(-T*(P - x0)*d2Vms_sat_dT2[i] - T*(x4 + x5)*d2Psats_dT2[i] 
                    + x2*(x1 - x4) + x2*(T*x6 + x5) - x7*(-R*x7*Psat_inv*Psat_inv + x3 - x6) + Cpigs_pure[i])
                    # The second derivative of volume is zero when extrapolating, which causes zero issues, discontinuous derivative
                '''
                from sympy import *
                T, P, R, zi = symbols('T, P, R, zi')
                Psat, Cpig_int, Vmsat = symbols('Psat, Cpig_int, Vmsat', cls=Function)
                dVmsatdT = diff(Vmsat(T), T)
                dPsatdT = diff(Psat(T), T)
                dV_vap = R*T/Psat(T) - Vmsat(T)
                dS_vap = dPsatdT*dV_vap
                Hvap = T*dS_vap
                H = zi*(Cpig_int(T) - Hvap)
                
                dP = P - Psat(T)
                H += zi*dP*(Vmsat(T) - T*dVmsatdT)
                
                (cse(diff(H, T), optimizations='basic'))
                '''
            except (ZeroDivisionError, ValueError):
                # Handle the zero division on Psat or the log getting two small
                failed_dPsat_dT = True
            
            if failed_dPsat_dT or isinf(Cp) or isnan(Cp):
                dlnPsats_dT = self.dlnPsats_dT()
                d2lnPsats_dT2 = self.d2lnPsats_dT2()
                Cp = 0.0
                for i in self.cmps:
                    Cp += zs[i]*(Cpigs_pure[i] - P*T*d2Vms_sat_dT2[i] - R*T*T*d2lnPsats_dT2[i]
                    - 2.0*R*T*dlnPsats_dT[i])
                    '''
                    from sympy import *
                    T, P, R, zi = symbols('T, P, R, zi')
                    lnPsat, Cpig_T_int, Vmsat = symbols('lnPsat, Cpig_T_int, Vmsat', cls=Function)
                    dVmsatdT = diff(Vmsat(T), T)
                    dPsatdT = diff(exp(lnPsat(T)), T)
                    dV_vap = R*T/exp(lnPsat(T)) - Vmsat(T)
                    dS_vap = dPsatdT*dV_vap
                    Hvap = T*dS_vap
                    H = zi*(Cpig_int(T) - Hvap)
                    dP = P
                    H += zi*dP*(Vmsat(T) - T*dVmsatdT)
                    print(simplify(expand(diff(H, T)).subs(exp(lnPsat(T)), 0)/zi))
                    '''                
#                Cp += zs[i]*(Cpigs_pure[i] - dHvaps_dT[i])
#                Cp += zs[i]*(-T*(P - Psats[i])*d2Vms_sat_dT2[i] + (T*dVms_sat_dT[i] - Vms_sat[i])*dPsats_dT[i])

        else:
            dHvaps_dT = self.dHvaps_dT()
            for i in self.cmps:
                Cp += zs[i]*(Cpigs_pure[i] - dHvaps_dT[i])
            
        Cp += self.GibbsExcessModel.CpE()
        return Cp
    
    def Cp(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        T, P, zs, cmps = self.T, self.P, self.zs, self.cmps
        Cpigs_pure = self.Cpigs_pure()
        use_Poynting, use_phis_sat = self.use_Poynting, self.use_phis_sat
        
        if use_Poynting:
            try:
                d2Poyntings_dT2 = self._d2Poyntings_dT2
            except AttributeError:
                d2Poyntings_dT2 = self.d2Poyntings_dT2()        
            try:
                dPoyntings_dT = self._dPoyntings_dT
            except AttributeError:
                dPoyntings_dT = self.dPoyntings_dT()        
            try:
                Poyntings = self._Poyntings
            except AttributeError:
                Poyntings = self.Poyntings()
        if use_phis_sat:
            try:
                d2phis_sat_dT2 = self._d2phis_sat_dT2
            except AttributeError:
                d2phis_sat_dT2 = self.d2phis_sat_dT2()
            try:
                dphis_sat_dT = self._dphis_sat_dT
            except AttributeError:
                dphis_sat_dT = self.dphis_sat_dT()
            try:
                phis_sat = self._phis_sat
            except AttributeError:
                phis_sat = self.phis_sat()
            
        dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
        d2Psats_dT2_over_Psats = self.d2Psats_dT2_over_Psats()
        
        RT = R*T
        RT2 = RT*T
        RT2_2 = RT + RT
        
        Cp = 0.0
        if use_Poynting and use_phis_sat:
            for i in cmps:
                Poy_inv = 1.0/Poyntings[i]
                phi_inv = 1.0/phis_sat[i]
                dPoy_ratio = dPoyntings_dT[i]*Poy_inv
                dphi_ratio = dphis_sat_dT[i]*phi_inv
                
                a = (d2phis_sat_dT2[i]*phi_inv - dphi_ratio*dphi_ratio
                     + d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i]
                     + d2Poyntings_dT2[i]*Poy_inv - dPoy_ratio*dPoy_ratio)
                
                b = dphi_ratio + dPsats_dT_over_Psats[i] + dPoy_ratio
                Cp -= zs[i]*(RT2*a + RT2_2*b - Cpigs_pure[i])
        elif use_Poynting:
            for i in cmps:
                Poy_inv = 1.0/Poyntings[i]
                dPoy_ratio = dPoyntings_dT[i]*Poy_inv                
                a = (d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i]
                     + d2Poyntings_dT2[i]*Poy_inv - dPoy_ratio*dPoy_ratio)
                b = dPsats_dT_over_Psats[i] + dPoy_ratio
                Cp -= zs[i]*(RT2*a + RT2_2*b - Cpigs_pure[i])
        elif use_phis_sat:
            for i in cmps:
                phi_inv = 1.0/phis_sat[i]
                dphi_ratio = dphis_sat_dT[i]*phi_inv
                a = (d2phis_sat_dT2[i]*phi_inv - dphi_ratio*dphi_ratio
                     + d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i])
                b = dphi_ratio + dPsats_dT_over_Psats[i]
                Cp -= zs[i]*(RT2*a + RT2_2*b - Cpigs_pure[i])
        else:
            for i in cmps:
                a = (d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i])
                b = dPsats_dT_over_Psats[i]
                Cp -= zs[i]*(RT2*a + RT2_2*b - Cpigs_pure[i])
        if not self.composition_independent:
            Cp += self.GibbsExcessModel.CpE()
        self._Cp = Cp
        return Cp
    
    dH_dT = Cp

    def dS_dT_old(self):
        # Needs testing
        T, P, P_DEPENDENT_H_LIQ = self.T, self.P, self.P_DEPENDENT_H_LIQ
        RT = R*T
        Cp, zs = 0.0, self.zs
        Cpigs_pure = self.Cpigs_pure()
        dS_dT = 0.0
        T_inv = 1.0/T
        if P_DEPENDENT_H_LIQ:
            d2Vms_sat_dT2 = self.d2Vms_sat_dT2()
            dVms_sat_dT = self.dVms_sat_dT()
            Vms_sat = self.Vms_sat()
            Psats = self.Psats()
            dPsats_dT = self.dPsats_dT()
            d2Psats_dT2 = self.d2Psats_dT2()
            failed_dPsat_dT = False
            for Psat in Psats:
                if Psat < 1e-40:
                    failed_dPsat_dT = True
            if not failed_dPsat_dT:
                try:
                    '''
                    from sympy import *
                    T, P, R, zi, P_REF_IG = symbols('T, P, R, zi, P_REF_IG')
                    
                    Psat, Cpig_T_int, Vmsat = symbols('Psat, Cpig_T_int, Vmsat', cls=Function)
                    dVmsatdT = diff(Vmsat(T), T)
                    dPsatdT = diff(Psat(T), T)
                    
                    S = 0
                    dSi = Cpig_T_int(T)
                    dVsat = R*T/Psat(T) - Vmsat(T)
                    dSvap = dPsatdT*dVsat
                    dSi -= dSvap
                    dSi -= R*log(Psat(T)/P_REF_IG)
                    dP = P - Psat(T)
                    dSi -= dP*dVmsatdT
                    S += dSi*zi
                    # cse(diff(S, T), optimizations='basic')
                    '''
                    for i in self.cmps:
                        x0 = Psats[i]
                        x1 = dPsats_dT[i]
                        x2 = R/x0
                        x3 = Vms_sat[i]
                        x4 = dVms_sat_dT[i]
                        dS_dT -= zs[i]*(x1*x2 - x1*x4 - x1*(RT*x1/x0**2 - x2 + x4) + (P - x0)*d2Vms_sat_dT2[i]
                        + (T*x2 - x3)*d2Psats_dT2[i] - Cpigs_pure[i]*T_inv)
                except (ZeroDivisionError, ValueError):
                    # Handle the zero division on Psat or the log getting two small
                    failed_dPsat_dT = True
                
            if failed_dPsat_dT:
                lnPsats = self.lnPsats()
                dlnPsats_dT = self.dlnPsats_dT()
                d2lnPsats_dT2 = self.d2lnPsats_dT2()
#                P*Derivative(Vmsat(T), (T, 2))
#                R*T*Derivative(lnPsat(T), (T, 2))
#                 2*R*Derivative(lnPsat(T), T) + Derivative(Cpig_T_int(T), T)
                '''
                from sympy import *
                T, P, R, zi, P_REF_IG = symbols('T, P, R, zi, P_REF_IG')
                
                lnPsat, Cpig_T_int, Vmsat = symbols('lnPsat, Cpig_T_int, Vmsat', cls=Function)
                # Psat, Cpig_T_int, Vmsat = symbols('Psat, Cpig_T_int, Vmsat', cls=Function)
                dVmsatdT = diff(Vmsat(T), T)
                dPsatdT = diff(exp(lnPsat(T)), T)
                
                S = 0
                dSi = Cpig_T_int(T)
                dVsat = R*T/exp(lnPsat(T)) - Vmsat(T)
                dSvap = dPsatdT*dVsat
                dSi -= dSvap
                # dSi -= R*log(Psat(T)/P_REF_IG)
                dSi -= R*(lnPsat(T) - log(P_REF_IG))
                dP = P - exp(lnPsat(T))
                dSi -= dP*dVmsatdT
                S += dSi*zi
                # cse(diff(S, T), optimizations='basic')
                print(simplify(expand(diff(S, T)).subs(exp(lnPsat(T)), 0)/zi))


                '''
                dS_dT = 0.0
                for i in self.cmps:
                    dS_dT -= zs[i]*(P*d2Vms_sat_dT2[i] + RT*d2lnPsats_dT2[i]
                    + 2.0*R*dlnPsats_dT[i]- Cpigs_pure[i]*T_inv)
                
        dS_dT += self.GibbsExcessModel.dSE_dT()
        return dS_dT

    def dS_dT(self):
        try:
            return self._dS_dT
        except AttributeError:
            pass
        T, P, zs, cmps = self.T, self.P, self.zs, self.cmps
        use_Poynting, use_phis_sat = self.use_Poynting, self.use_phis_sat
        
        if use_Poynting:
            try:
                d2Poyntings_dT2 = self._d2Poyntings_dT2
            except AttributeError:
                d2Poyntings_dT2 = self.d2Poyntings_dT2()        
            try:
                dPoyntings_dT = self._dPoyntings_dT
            except AttributeError:
                dPoyntings_dT = self.dPoyntings_dT()        
            try:
                Poyntings = self._Poyntings
            except AttributeError:
                Poyntings = self.Poyntings()
        if use_phis_sat:
            try:
                d2phis_sat_dT2 = self._d2phis_sat_dT2
            except AttributeError:
                d2phis_sat_dT2 = self.d2phis_sat_dT2()
            try:
                dphis_sat_dT = self._dphis_sat_dT
            except AttributeError:
                dphis_sat_dT = self.dphis_sat_dT()
            try:
                phis_sat = self._phis_sat
            except AttributeError:
                phis_sat = self.phis_sat()
            
        dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
        d2Psats_dT2_over_Psats = self.d2Psats_dT2_over_Psats()
        Cpigs_pure = self.Cpigs_pure()

        T_inv = 1.0/T
        RT = R*T
        R_2 = R + R
        
        dS_dT = 0.0
        if use_Poynting and use_phis_sat:
            for i in cmps:
                Poy_inv = 1.0/Poyntings[i]
                phi_inv = 1.0/phis_sat[i]
                dPoy_ratio = dPoyntings_dT[i]*Poy_inv
                dphi_ratio = dphis_sat_dT[i]*phi_inv
                
                a = (d2phis_sat_dT2[i]*phi_inv - dphi_ratio*dphi_ratio
                     + d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i]
                     + d2Poyntings_dT2[i]*Poy_inv - dPoy_ratio*dPoy_ratio)
                
                b = dphi_ratio + dPsats_dT_over_Psats[i] + dPoy_ratio
                
                dS_dT -= zs[i]*((RT*a + b*R_2) - Cpigs_pure[i]*T_inv)
        elif use_Poynting:
            for i in cmps:
                Poy_inv = 1.0/Poyntings[i]
                dPoy_ratio = dPoyntings_dT[i]*Poy_inv                
                a = (d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i]
                     + d2Poyntings_dT2[i]*Poy_inv - dPoy_ratio*dPoy_ratio)
                b = dPsats_dT_over_Psats[i] + dPoy_ratio
                dS_dT -= zs[i]*((RT*a + b*R_2) - Cpigs_pure[i]*T_inv)
        elif use_phis_sat:
            for i in cmps:
                phi_inv = 1.0/phis_sat[i]
                dphi_ratio = dphis_sat_dT[i]*phi_inv
                a = (d2phis_sat_dT2[i]*phi_inv - dphi_ratio*dphi_ratio
                     + d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i])
                b = dphi_ratio + dPsats_dT_over_Psats[i]
                dS_dT -= zs[i]*((RT*a + b*R_2) - Cpigs_pure[i]*T_inv)
        else:
            for i in cmps:
                a = (d2Psats_dT2_over_Psats[i] - dPsats_dT_over_Psats[i]*dPsats_dT_over_Psats[i])
                b = dPsats_dT_over_Psats[i]
                dS_dT -= zs[i]*((RT*a + b*R_2) - Cpigs_pure[i]*T_inv)
        if not self.composition_independent:
            dS_dT += self.GibbsExcessModel.dSE_dT()
        self._dS_dT = dS_dT
        return dS_dT

    def dH_dP(self):
        try:
            return self._dH_dP
        except AttributeError:
            pass
        T = self.T
        P = self.P
        zs = self.zs
        dH_dP = 0.0
        if self.use_Poynting:
            nRT2 = -R*T*T
            Poyntings = self.Poyntings()
            dPoyntings_dP = self.dPoyntings_dP()
            dPoyntings_dT = self.dPoyntings_dT()
            d2Poyntings_dPdT = self.d2Poyntings_dPdT()
            for i in self.cmps:
                Poy_inv = 1.0/Poyntings[i]
                dH_dP += nRT2*zs[i]*Poy_inv*(d2Poyntings_dPdT[i] - dPoyntings_dP[i]*dPoyntings_dT[i]*Poy_inv)
        
#        if self.P_DEPENDENT_H_LIQ:
#            if self.use_IG_Cp:
#                Vms_sat = self.Vms_sat()
#                dVms_sat_dT = self.dVms_sat_dT()
#                Psats = self.Psats()
#                for i in self.cmps:
#                    if P > Psats[i]:
#                        dH_dP += zs[i]*(-T*dVms_sat_dT[i] + Vms_sat[i])
        self._dH_dP = dH_dP
        return dH_dP


    def dS_dP(self):
        try:
            return self._dS_dP
        except AttributeError:
            pass
        T = self.T
        P = self.P
        P_inv = 1.0/P
        zs = self.zs
        if self.use_Poynting:
            dS_dP = -R*P_inv
            Poyntings = self.Poyntings()
            dPoyntings_dP = self.dPoyntings_dP()
            dPoyntings_dT = self.dPoyntings_dT()
            d2Poyntings_dPdT = self.d2Poyntings_dPdT()
            for i in self.cmps:
                Poy_inv = 1.0/Poyntings[i]
                dS_dP -= zs[i]*R*Poy_inv*(dPoyntings_dP[i] - Poyntings[i]*P_inv
                        +T*(d2Poyntings_dPdT[i] - dPoyntings_dP[i]*dPoyntings_dT[i]*Poy_inv))
        else:
            dS_dP = 0.0
#        if self.P_DEPENDENT_H_LIQ:
#            if self.use_IG_Cp:
#                dVms_sat_dT = self.dVms_sat_dT()
#                Psats = self.Psats()
#                for i in self.cmps:
#                    if P > Psats[i]:
#                        dS_dP -= zs[i]*(dVms_sat_dT[i])
        self._dS_dP = dS_dP
        return dS_dP

    def H_dep(self):
        return self.H() - self.H_ideal_gas()

    def S_dep(self):
        return self.S() - self.S_ideal_gas()

    def Cp_dep(self):
        return self.Cp() - self.Cp_ideal_gas()

    ### Volumetric properties
    def V(self):
        try:
            return self._V
        except AttributeError:
            pass
        zs = self.zs
        Vms = self.Vms()
        '''To make a fugacity-volume identity consistent, cannot use pressure
        correction unless the Poynting factor is calculated with quadrature/
        integration.
        '''
        V = 0.0
        for i in self.cmps:
            V += zs[i]*Vms[i]
        self._V = V
        return V

    def dV_dT(self):
        try:
            return self._dV_dT
        except AttributeError:
            pass
        zs = self.zs
        dVms_sat_dT = self.dVms_sat_dT()
        dV_dT = 0.0
        for i in self.cmps:
            dV_dT += zs[i]*dVms_sat_dT[i]
        self._dV_dT = dV_dT
        return dV_dT

    def d2V_dT2(self):
        try:
            return self._d2V_dT2
        except AttributeError:
            pass
        zs = self.zs
        d2Vms_sat_dT2 = self.d2Vms_sat_dT2()
        d2V_dT2 = 0.0
        for i in self.cmps:
            d2V_dT2 += zs[i]*d2Vms_sat_dT2[i]
        self._d2V_dT2 = d2V_dT2
        return d2V_dT2

    # Main needed volume derivatives
    def dP_dV(self):
        try:
            return self._dP_dV
        except AttributeError:
            pass
        if self.incompressible:
            self._dP_dV = INCOMPRESSIBLE_CONST #1.0/self.VolumeLiquidMixture.property_derivative_P(self.T, self.P, self.zs, order=1)
            
        return self._dP_dV
    
    def d2P_dV2(self):
        try:
            return self._d2P_dV2
        except AttributeError:
            pass
        if self.incompressible:
            self._d2P_dV2 = INCOMPRESSIBLE_CONST#self.d2V_dP2()/-(self.dP_dV())**-3
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
        if self.incompressible:
            self._d2P_dTdV = 0.0
        else:
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
        if self.incompressible:
            self._d2P_dT2 = -self.d2V_dT2()/INCOMPRESSIBLE_CONST
        else:
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
        if self.incompressible:
            self._d2V_dP2 = 0.0
        return self._d2V_dP2

    def Tait_Bs(self):
        try:
            return self._Tait_Bs
        except:
            pass
        
        self._Tait_Bs = evaluate_linear_fits(self.Tait_B_data, self.T)
        return self._Tait_Bs
        
    def dTait_B_dTs(self):
        try:
            return self._dTait_B_dTs
        except:
            pass
        
        self._dTait_B_dTs = evaluate_linear_fits_d(self.Tait_B_data, self.T)
        return self._dTait_B_dTs
        
    def d2Tait_B_dT2s(self):
        try:
            return self._d2Tait_B_dT2s
        except:
            pass
        
        self._d2Tait_B_dT2s = evaluate_linear_fits_d2(self.Tait_B_data, self.T)
        return self._d2Tait_B_dT2s

    def Tait_Cs(self):
        try:
            return self._Tait_Cs
        except:
            pass
        
        self._Tait_Cs = evaluate_linear_fits(self.Tait_C_data, self.T)
        return self._Tait_Cs
        
    def dTait_C_dTs(self):
        try:
            return self._dTait_C_dTs
        except:
            pass
        
        self._dTait_C_dTs = evaluate_linear_fits_d(self.Tait_C_data, self.T)
        return self._dTait_C_dTs
        
    def d2Tait_C_dT2s(self):
        try:
            return self._d2Tait_C_dT2s
        except:
            pass
        
        self._d2Tait_C_dT2s = evaluate_linear_fits_d2(self.Tait_C_data, self.T)
        return self._d2Tait_C_dT2s
    
    def Tait_Vs(self):
        Vms_sat = self.Vms_sat()
        Psats = self.Psats()
        Tait_Bs = self.Tait_Bs()
        Tait_Cs = self.Tait_Cs()
        P = self.P
        return [Vms_sat[i]*(1.0  - Tait_Cs[i]*log((Tait_Bs[i] + P)/(Tait_Bs[i] + Psats[i]) ))
                for i in self.cmps]

        
    def dH_dP_integrals_Tait(self):
        try:
            return self._dH_dP_integrals_Tait
        except AttributeError:
            pass
        Psats = self.Psats()
        Vms_sat = self.Vms_sat()
        dVms_sat_dT = self.dVms_sat_dT()
        dPsats_dT = self.dPsats_dT()
        
        Tait_Bs = self.Tait_Bs()
        Tait_Cs = self.Tait_Cs()
        dTait_C_dTs = self.dTait_C_dTs()
        dTait_B_dTs = self.dTait_B_dTs()
        T, P, zs = self.T, self.P, self.zs
        
        
        self._dH_dP_integrals_Tait = dH_dP_integrals_Tait = []
        
#        def to_int(P, i):
#            l = self.to_TP_zs(T, P, zs)
##            def to_diff(T):
##                return self.to_TP_zs(T, P, zs).Tait_Vs()[i]
##            dV_dT = derivative(to_diff, T, dx=1e-5*T, order=11)
#            
#            x0 = l.Vms_sat()[i]
#            x1 = l.Tait_Cs()[i]
#            x2 = l.Tait_Bs()[i]
#            x3 = P + x2
#            x4 = l.Psats()[i]
#            x5 = x3/(x2 + x4)
#            x6 = log(x5)
#            x7 = l.dTait_B_dTs()[i]
#            dV_dT = (-x0*(x1*(-x5*(x7 +l.dPsats_dT()[i]) + x7)/x3 
#                                   + x6*l.dTait_C_dTs()[i])
#                        - (x1*x6 - 1.0)*l.dVms_sat_dT()[i])
#                        
##            print(dV_dT, dV_dT2, dV_dT/dV_dT2, T, P)   
#            
#            V = l.Tait_Vs()[i]
#            return V - T*dV_dT
#        from scipy.integrate import quad
#        _dH_dP_integrals_Tait = [quad(to_int, Psats[i], P, args=i)[0]
#                                      for i in self.cmps]
##        return self._dH_dP_integrals_Tait
#        print(_dH_dP_integrals_Tait)
#        self._dH_dP_integrals_Tait2 = _dH_dP_integrals_Tait
#        return self._dH_dP_integrals_Tait2
        
#        dH_dP_integrals_Tait = []
        for i in self.cmps:
            # Very wrong according to numerical integration. Is it an issue with
            # the translation to code, one of the derivatives, what was integrated,
            # or sympy's integration?
            x0 = Tait_Bs[i]
            x1 = P + x0
            x2 = Psats[i]
            x3 = x0 + x2
            x4 = 1.0/x3
            x5 = Tait_Cs[i]
            x6 = Vms_sat[i]
            x7 = x5*x6
            x8 = T*dVms_sat_dT[i]
            x9 = x5*x8
            x10 = T*dTait_C_dTs[i]
            x11 = x0*x6
            x12 = T*x7
            x13 = -x0*x7 + x0*x9 + x10*x11 + x12*dTait_B_dTs[i]
            x14 = x2*x6
            x15 = x4*(x0*x8 + x10*x14 - x11 + x12*dPsats_dT[i] + x13 - x14 - x2*x7 + x2*x8 + x2*x9)
            val = -P*x15 + P*(x10*x6 - x7 + x9)*log(x1*x4) + x13*log(x1) - x13*log(x3) + x15*x2
            dH_dP_integrals_Tait.append(val)
#        print(dH_dP_integrals_Tait, self._dH_dP_integrals_Tait2)
        return dH_dP_integrals_Tait
        
    def mu(self):
        try:
            return self._mu
        except AttributeError:
            pass
        mu = self._mu = self.correlations.ViscosityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        return mu

    def k(self):
        try:
            return self._k
        except AttributeError:
            pass
        self._k = k = self.correlations.ThermalConductivityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        return k

    
class GibbsExcessSolid(GibbsExcessLiquid):
    force_phase = 's'
    phase = 's'
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
        super(GibbsExcessSolid, self).__init__(VaporPressures=SublimationPressures, VolumeLiquids=VolumeSolids,
              HeatCapacityGases=HeatCapacityGases, EnthalpyVaporizations=EnthalpySublimations,
              use_Poynting=use_Poynting,
              Hfs=Hfs, Gfs=Gfs, Sfs=Sfs, T=T, P=P, zs=zs)

# hydrogen, methane
Grayson_Streed_special_CASs = set(['1333-74-0', '74-82-8'])

class GraysonStreed(Phase):
    phase = force_phase = 'l'
    # revised one
    
    hydrogen_coeffs = (1.50709, 2.74283, -0.0211, 0.00011, 0.0, 0.008585, 0.0, 0.0, 0.0, 0.0)
    methane_coeffs = (1.36822, -1.54831, 0.0, 0.02889, -0.01076, 0.10486, -0.02529, 0.0, 0.0, 0.0)
    simple_coeffs = (2.05135, -2.10889, 0.0, -0.19396, 0.02282, 0.08852, 0.0, -0.00872, -0.00353, 0.00203)
    version = 1

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        
        new._Tcs = self._Tcs
        new._Pcs = self._Pcs
        new._omegas = self._omegas
        new._CASs = self._CASs
        new.regular = self.regular
        new.GibbsExcessModel = self.GibbsExcessModel.to_T_xs(T, zs)
        new.version = self.version
        
        try:
            new.cmps = self.cmps
            new.N = self.N
        except:
            pass

        return new

    def to_zs_TPV(self, zs, T=None, P=None, V=None):
        if T is not None:
            if P is not None:
                return self.to_TP_zs(T=T, P=P, zs=zs)
            elif V is not None:
                raise ValueError("Model does not implement volume")
        elif P is not None and V is not None:
            raise ValueError("Model does not implement volume")
        else:
            raise ValueError("Two of T, P, or V are needed")

    def __init__(self, Tcs, Pcs, omegas, CASs,
                 GibbsExcessModel=IdealSolution(), 
                 T=None, P=None, zs=None,
                 ):
        
        self.T = T
        self.P = P
        self.zs = zs
        
        self.N = len(zs)
        self.cmps = range(self.N)
        self._Tcs = Tcs
        self._Pcs = Pcs
        self._omegas = omegas
        self._CASs = CASs
        self.regular = [i not in Grayson_Streed_special_CASs for i in CASs]
        
        self.GibbsExcessModel = GibbsExcessModel

    def gammas(self):
        try:
            return self.GibbsExcessModel._gammas
        except AttributeError:
            return self.GibbsExcessModel.gammas()
        
    def phis(self):
        try:
            return self._phis
        except AttributeError:
            pass
        try:
            gammas = self._gammas
        except AttributeError:
            gammas = self.gammas()
        fugacity_coeffs_pure = self.nus()
        
        self._phis = [gammas[i]*fugacity_coeffs_pure[i]
                for i in self.cmps]
        return self._phis
        
        
    def lnphis(self):
        try:
            return self._lnphis
        except AttributeError:
            pass
        self._lnphis = [log(i) for i in self.phis()]        
        return self._lnphis
    
    lnphis_G_min = lnphis

    def nus(self):
        T, P = self.T, self.P
        Tcs, Pcs, omegas = self._Tcs, self._Pcs, self._omegas
        regular, CASs = self.regular, self._CASs
        nus = []
        limit_Tr = self.version > 0
        
        for i in self.cmps:
            # TODO validate and take T, P derivatives; n derivatives are from regular solution only
            Tr = T/Tcs[i]
            Pr = P/Pcs[i]
            
            if regular[i]:
                coeffs = self.simple_coeffs
            elif CASs[i] == '1333-74-0':
                coeffs = self.hydrogen_coeffs
            elif CASs[i] == '74-82-8':
                coeffs = self.methane_coeffs
            else:
                raise ValueError("Fail")
            A0, A1, A2, A3, A4, A5, A6, A7, A8, A9 = coeffs
    
            log10_v0 = A0 + A1/Tr + A2*Tr + A3*Tr**2 + A4*Tr**3 + (A5 + A6*Tr + A7*Tr**2)*Pr + (A8 + A9*Tr)*Pr**2 - log10(Pr)
            log10_v1 = -4.23893 + 8.65808*Tr - 1.2206/Tr - 3.15224*Tr**3 - 0.025*(Pr - 0.6)
            if Tr > 1.0 and limit_Tr:
                log10_v1 = 1.0
            
            if regular[i]:
                v = 10.0**(log10_v0 + omegas[i]*log10_v1)
            else:
                # Chao Seader mentions
                v = 10.0**(log10_v0)
            nus.append(v)
        return nus

class ChaoSeader(GraysonStreed):
    # original one
    hydrogen_coeffs = (1.96718, 1.02972, -0.054009, 0.0005288, 0.0, 0.008585, 0.0, 0.0, 0.0, 0.0)
    methane_coeffs = (2.4384, -2.2455, -0.34084, 0.00212, -0.00223, 0.10486, -0.03691, 0.0, 0.0, 0.0)
    simple_coeffs = (5.75748, -3.01761, -4.985, 2.02299, 0.0, 0.08427, 0.26667, -0.31138, -0.02655, 0.02883)
    version = 0


if has_CoolProp:
    CPPT_INPUTS = CoolProp.PT_INPUTS
    CPrhoT_INPUTS = CoolProp.DmolarT_INPUTS
    CPrhoP_INPUTS = CoolProp.DmolarP_INPUTS

    CPiP, CPiT, CPiDmolar = CoolProp.iP, CoolProp.iT, CoolProp.iDmolar
    CPiHmolar, CPiSmolar = CoolProp.iHmolar, CoolProp.iSmolar
    
    CPPQ_INPUTS, CPQT_INPUTS = CoolProp.PQ_INPUTS, CoolProp.QT_INPUTS

    CoolProp_gas_phases = set([CoolProp.iphase_gas, CoolProp.iphase_supercritical, CoolProp.iphase_unknown,
                              CoolProp.iphase_critical_point, CoolProp.iphase_supercritical_gas])
    CoolProp_liquid_phases = set([CoolProp.iphase_liquid, CoolProp.iphase_supercritical_liquid])
    
    CPliquid = CoolProp.iphase_liquid
    CPgas = CoolProp.iphase_gas
    CPunknown = CoolProp.iphase_not_imposed
    
    # Probably todo - hold onto ASs for up to 1 sec, then release them for reuse
    # Do not allow Phase direct access any more, use a decorator
#    CoolProp_AS_cache = {}
#    def get_CoolProp_AS(backend, fluid):
#        key = (backend, fluid)
#        try:
#            in_use, free = CoolProp_AS_cache[key]
#            if free:
#                AS = free.pop()
#            else:
#                AS = CoolProp.AbstractState(backend, fluid)
#            in_use.add(AS)
##            in_use.append(AS)
#            return AS
#        except KeyError:
##            in_use, free = [], []
#            in_use, free = set([]), set([])
#            AS = CoolProp.AbstractState(backend, fluid)
#            in_use.add(AS)
##            in_use.append(AS)
#            CoolProp_AS_cache[key] = (in_use, free)
#            return AS
#        
#    def free_CoolProp_AS(AS, backend, fluid):
#        key = (backend, fluid)
#        try:
#            in_use, free = CoolProp_AS_cache[key]
#        except KeyError:
#            raise ValueError("Should not happen")
#        in_use.remove(AS)
##        free.append(AS)
#        free.add(AS)


    # Emperically measured to be ~140 KB/instance, do not want to cache too many - 35 is 5 MB
    max_CoolProp_states = 35
    # Forget about time - just use them last; make sure the LRU is at the top
    # 
    if not SORTED_DICT:
        caching_states_CoolProp = OrderedDict()
    else:
        caching_states_CoolProp = {}
        
    def caching_state_CoolProp(backend, fluid, spec0, spec1, spec_set, phase, zs):
        # Pretty sure about as optimized as can get!
        # zs should be a tuple, not a list
        key = (backend, fluid, spec0, spec1, spec_set, phase, zs)
        if key in caching_states_CoolProp:
            AS = caching_states_CoolProp[key]
            try:
                caching_states_CoolProp.move_to_end(key)
            except:
                # Move to end the old fashioned way
                del caching_states_CoolProp[key]
                caching_states_CoolProp[key] = AS
        elif len(caching_states_CoolProp) < max_CoolProp_states:
            # Always make a new item until the cache is full
            AS = CoolProp.AbstractState(backend, fluid)
            AS.specify_phase(phase)
            if zs is not None:
                AS.set_mole_fractions(zs)
            AS.update(spec_set, spec0, spec1)
            caching_states_CoolProp[key] = AS
            return AS
        else:
            # Reuse an item if not in the cache, making the value go to the end of
            # the ordered dict
            if not SORTED_DICT:
                old_key, AS = caching_states_CoolProp.popitem(False)
            else:
                # Hack - get first item in dict
                old_key = next(iter(caching_states_CoolProp))
                AS = caching_states_CoolProp.pop(old_key)
            
            if old_key[1] != fluid or old_key[0] != backend:
                # Handle different components - other will be gc
                AS = CoolProp.AbstractState(backend, fluid)
            AS.specify_phase(phase)
            if zs is not None:
                AS.set_mole_fractions(zs)
            AS.update(spec_set, spec0, spec1)
            caching_states_CoolProp[key] = AS
        return AS
        

class CoolPropPhase(Phase):
    prefer_phase = CPunknown
    
        
    def __repr__(self):
        if self.phase == 'g':
            s =  '<%s, ' %('CoolPropGas')
        else:
            s =  '<%s, ' %('CoolPropLiquid')
        try:
            s += 'T=%g K, P=%g Pa' %(self.T, self.P)
        except:
            pass
        s += '>'
        return s
    
#    def __del__(self):
#        # Not sustainable at all
#        # time-based cache seems next best
#        free_CoolProp_AS(self.AS, self.backend, self.fluid)
        
        
    @property
    def phase(self):
        try:
            idx = self.AS.phase()
            if idx in CoolProp_gas_phases:
                return 'g'
            return 'l'
        except:
            if self.prefer_phase == CPliquid:
                return 'l'
            return 'g'

    def model_hash(self, ignore_phase=False):
        return hash_any_primitive([self.backend, self.fluid, self.Hfs, self.Gfs, self.Sfs, self.__class__])
        
    def __init__(self, backend, fluid,
                 T=None, P=None, zs=None,  Hfs=None,
                 Gfs=None, Sfs=None,):

        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs
        
        self.backend = backend
        self.fluid = fluid
        
        self.skip_comp = skip_comp = (backend in ('IF97') or fluid in ('water') or '&' not in fluid)
        if zs is None:
            zs = [1.0]
        self.zs = zs
        self.N = N = len(zs)
        self.cmps = range(N)
        if skip_comp or N == 1:
            zs_key = None
        else:
            zs_key = tuple(zs)
        if T is not None and P is not None:
            self.T = T
            self.P = P
            try:
                key = (backend, fluid, P, T, CPPT_INPUTS, self.prefer_phase, zs_key)
                AS = caching_state_CoolProp(*key)
            except:
                key = (backend, fluid, P, T, CPPT_INPUTS, CPunknown, zs_key)
                AS = caching_state_CoolProp(*key)
            self.key = key
            self._cache_easy_properties(AS)
#        if not skip_comp and zs is None:
#            self.zs = [1.0]
                
#            AS = get_CoolProp_AS(backend, fluid)#CoolProp.AbstractState(backend, fluid)
#            if not skip_comp:
#                AS.set_mole_fractions(zs)
#            AS.specify_phase(self.prefer_phase)
#            try:
#                AS.update(CPPT_INPUTS, P, T)
#            except:
#                AS.specify_phase(CPunknown)
#                AS.update(CPPT_INPUTS, P, T)
#
#            rho = AS.rhomolar()
#            key = (backend, fluid, T, rho)
    @property
    def AS(self):
        return caching_state_CoolProp(*self.key)
                
    def to_TP_zs(self, T, P, zs):
        return self.to_zs_TPV(T=T, P=P, zs=zs)
    
    def from_AS(self, AS):
        new = self.__class__.__new__(self.__class__)
        new.N = N = self.N
        if N == 1:
            zs_key = None
            new.zs = self.zs
        else:
            new.zs = zs = AS.get_mole_fractions()
            zs_key = tuple(zs)
        new.cmps = self.cmps
        new.backend = backend = self.backend
        new.fluid = fluid = self.fluid
        new.skip_comp = self.skip_comp
        new.T, new.P = T, P = AS.T(), AS.p()
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs

        # Always use density as an input - does not require a phase ID spec / setting with AS.phase() seems to not work
        new._cache_easy_properties(AS)
        new.key = (backend, fluid, self._rho, T, CPrhoT_INPUTS, CPunknown, zs_key)
        return new

    def to_zs_TPV(self, zs, T=None, P=None, V=None, prefer_phase=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.N = self.N
        new.cmps = self.cmps
        new.backend = backend = self.backend
        new.fluid = fluid = self.fluid
        new.skip_comp = skip_comp = self.skip_comp
        if skip_comp or N == 1:
            zs_key = None
        else:
            zs_key = tuple(zs)
        
        if prefer_phase is None:
            prefer_phase = self.prefer_phase
        try:
            if T is not None:
                if P is not None:
                    new.T, new.P = T, P
                    key = (backend, fluid, P, T, CPPT_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
                elif V is not None:
                    key = (backend, fluid, 1.0/V, T, CPrhoT_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
    #                    AS.update(CPrhoT_INPUTS, 1.0/V, T)
                    new.T, new.P = T, AS.p()
            elif P is not None and V is not None:
                    key = (backend, fluid, 1.0/V, P, CPrhoP_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
    #                AS.update(CPrhoP_INPUTS, 1.0/V, P)
                    new.T, new.P = AS.T(), P
        except ValueError:
            prefer_phase = CPunknown
            if T is not None:
                if P is not None:
                    new.T, new.P = T, P
                    key = (backend, fluid, P, T, CPPT_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
                elif V is not None:
                    key = (backend, fluid, 1.0/V, T, CPrhoT_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
                    new.T, new.P = T, AS.p()
            elif P is not None and V is not None:
                    key = (backend, fluid, 1.0/V, P, CPrhoP_INPUTS, prefer_phase, zs_key)
                    AS = caching_state_CoolProp(*key)
                    new.T, new.P = AS.T(), P
        
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        new.key = key
        new._cache_easy_properties(AS)
        return new
    
    def _cache_easy_properties(self, AS):
        self._rho = AS.rhomolar()
        self._V = 1.0/self._rho
        self._H = AS.hmolar()
        self._S = AS.smolar()
        self._Cp = AS.cpmolar()
        self._PIP = AS.PIP()
        
    to = to_zs_TPV

    def V(self):
        return self._V
#        return 1.0/self.AS.rhomolar()

    def lnphis(self):
        try:
            return self._lnphis
        except AttributeError:
            pass
        self._lnphis = lnphis = []
        AS = self.AS
        for i in self.cmps:
            lnphis.append(log(AS.fugacity_coefficient(i)))
        return lnphis
    
    lnphis_G_min = lnphis

    def dlnphis_dT(self):
        raise NotImplementedError("Not in CoolProp")

    def dlnphis_dP(self):
        raise NotImplementedError("Not in CoolProp")
        
    def dlnphis_dns(self):
        raise NotImplementedError("Not in CoolProp")
        
    def dlnphis_dzs(self):
        raise NotImplementedError("Not in CoolProp")
        
    def gammas(self):
        raise NotImplementedError("TODO")
                
    def dP_dT(self):
        return self.AS.first_partial_deriv(CPiP, CPiT, CPiDmolar)
    dP_dT_V = dP_dT
    
    def dP_dV(self):
        rho = self.AS.rhomolar()
        dP_drho = self.AS.first_partial_deriv(CPiP, CPiDmolar, CPiT)
        return -dP_drho*rho*rho
    dP_dV_T = dP_dV
    
    def d2P_dT2(self):
        return self.AS.second_partial_deriv(CPiP, CPiT, CPiDmolar, CPiT, CPiDmolar)
    d2P_dT2_V = d2P_dT2
    
    def d2P_dV2(self):
        d2P_drho2 = self.AS.second_partial_deriv(CPiP, CPiDmolar, CPiT, CPiDmolar, CPiT)
        V = self.V()
        dP_dV = self.dP_dV()
        return (d2P_drho2/-V**2 + 2.0*V*dP_dV)/-V**2
    d2P_dV2_T = d2P_dV2
    
    def d2P_dTdV(self):
        d2P_dTdrho = self.AS.second_partial_deriv(CPiP, CPiT, CPiDmolar, CPiDmolar, CPiT)
        rho = self.AS.rhomolar()
        return -d2P_dTdrho*rho*rho
    
    def PIP(self):
        return self._PIP
        # Saves time
#        return self.AS.PIP()

    def H(self):
        return self._H
#        return self.AS.hmolar()

    def S(self):
        return self._S
#        return self.AS.smolar()

    def H_dep(self):
        return self.AS.hmolar_excess()

    def S_dep(self):
        return self.AS.smolar_excess()

    def Cp_dep(self):
        raise NotImplementedError("Not in CoolProp")
    
    def Cp(self):
        return self._Cp
#        return self.AS.cpmolar()
    dH_dT = Cp 
    
    def dH_dP(self):
        return self.AS.first_partial_deriv(CoolProp.iHmolar, CPiP, CPiT)
    
    def dH_dT_V(self):
        # Does not need rho multiplication
        return self.AS.first_partial_deriv(CoolProp.iHmolar, CPiT, CPiDmolar)

    def dH_dP_V(self):
        return self.AS.first_partial_deriv(CoolProp.iHmolar, CPiP, CPiDmolar)

    def dH_dV_T(self):
        rho = self.AS.rhomolar()
        return -self.AS.first_partial_deriv(CoolProp.iHmolar, CPiDmolar, CPiT)*rho*rho
        
    def dH_dV_P(self):
        rho = self.AS.rhomolar()
        return -self.AS.first_partial_deriv(CoolProp.iHmolar, CPiDmolar, CPiP)*rho*rho

    def d2H_dT2(self):
        return self.AS.second_partial_deriv(CoolProp.iHmolar, CPiT, CPiP, CPiT, CPiP)

    def d2H_dP2(self):
        return self.AS.second_partial_deriv(CoolProp.iHmolar, CPiP, CPiT, CPiP, CPiT)
    
    def d2H_dTdP(self):
        return self.AS.second_partial_deriv(CoolProp.iHmolar, CPiT, CPiP, CPiP, CPiT)
    
    def dS_dT(self):
        return self.AS.first_partial_deriv(CPiSmolar, CPiT, CPiP)
    
    def dS_dP(self):
        return self.AS.first_partial_deriv(CPiSmolar, CPiP, CPiT)
    
    def dS_dT_V(self):
        return self.AS.first_partial_deriv(CPiSmolar, CPiT, CPiDmolar)

    def dS_dP_V(self):
        return self.AS.first_partial_deriv(CPiSmolar, CPiP, CPiDmolar)

    def dS_dV_T(self):
        rho = self.AS.rhomolar()
        return -self.AS.first_partial_deriv(CPiSmolar, CPiDmolar, CPiT)*rho*rho
        
    def dS_dV_P(self):
        rho = self.AS.rhomolar()
        return -self.AS.first_partial_deriv(CPiSmolar, CPiDmolar, CPiP)*rho*rho

    def d2S_dT2(self):
        return self.AS.second_partial_deriv(CPiSmolar, CPiT, CPiP, CPiT, CPiP)

    def d2S_dP2(self):
        return self.AS.second_partial_deriv(CPiSmolar, CPiP, CPiT, CPiP, CPiT)
    
    def d2S_dTdP(self):
        return self.AS.second_partial_deriv(CPiSmolar, CPiT, CPiP, CPiP, CPiT)
    
    def mu(self):
        try:
            return self._mu
        except AttributeError:
            mu = self._mu = self.AS.viscosity()
        return mu

    def k(self):
        try:
            return self._k
        except AttributeError:
            k = self._k = self.AS.conductivity()
        return k

class CoolPropLiquid(CoolPropPhase):
    prefer_phase = CPliquid

class CoolPropGas(CoolPropPhase):
    prefer_phase = CPgas

class CombinedPhase(Phase):
    def __init__(self, phases, equilibrium=None, thermal=None, volume=None,
                 other_props=None,
                 T=None, P=None, zs=None,
                 ):
        # phases : list[other phases]
        # equilibrium: index
        # thermal: index
        # volume: index
        # other_props: dict[prop] = phase index
        
        # may be missing S_formation_ideal_gas Hfs arg
        self.equilibrium = equilibrium
        self.thermal = thermal
        self.volume = volume
        self.other_props = other_props
        
        for i, p in enumerate(phases):
            if p.T != T or p.P != P or p.zs != zs:
                phases[i] = p.to_zs_TPV(T=T, P=P, zs=zs)
        self.phases = phases
        
    def lnphis(self):
        # This style will save the getattr call but takes more time to code
        if 'lnphis' in self.other_props:
            return self.phases[self.other_props['lnphis']].lnphis()
        if self.equilibrium is not None:
            return self.phases[self.equilibrium].lnphis()
        raise ValueError("No method specified")
        
    def lnphis_G_min(self):
        if 'lnphis' in self.other_props:
            return self.phases[self.other_props['lnphis']].lnphis_G_min()
        if self.equilibrium is not None:
            return self.phases[self.equilibrium].lnphis_G_min()
        raise ValueError("No method specified")
        
    def makeeqfun(prop_name):
        def fun(self):
            if prop_name in self.other_props:
                return getattr(self.phases[self.other_props[prop_name]], prop_name)()
            if self.equilibrium is not None:
                return getattr(self.phases[self.equilibrium], prop_name)()
            raise ValueError("No method specified")
        return fun

    def makethermalfun(prop_name):
        def fun(self):
            if prop_name in self.other_props:
                return getattr(self.phases[self.other_props[prop_name]], prop_name)()
            if self.thermal is not None:
                return getattr(self.phases[self.thermal], prop_name)()
            raise ValueError("No method specified")
        return fun

    def makevolumefun(prop_name):
        def fun(self):
            if prop_name in self.other_props:
                return getattr(self.phases[self.other_props[prop_name]], prop_name)()
            if self.volume is not None:
                return getattr(self.phases[self.volume], prop_name)()
            raise ValueError("No method specified")
        return fun
    
    lnphis = makeeqfun("lnphis")
    dlnphis_dT = makeeqfun("dlnphis_dT")
    dlnphis_dP = makeeqfun("dlnphis_dP")
    dlnphis_dns = makeeqfun("dlnphis_dns")
    
    V = makevolumefun("V")
    dP_dT = makevolumefun("dP_dT")
    dP_dT_V = dP_dT
    dP_dV = makevolumefun("dP_dV")
    dP_dV_T = dP_dV
    d2P_dT2 = makevolumefun("d2P_dT2")
    d2P_dT2_V = d2P_dT2
    d2P_dV2 = makevolumefun("d2P_dV2")
    d2P_dV2_T = d2P_dV2
    d2P_dTdV = makevolumefun("d2P_dTdV")
    
    H = makethermalfun("H")
    S = makethermalfun("S")
    Cp = makethermalfun("Cp")
    dH_dT = makethermalfun("dH_dT")
    dH_dP = makethermalfun("dH_dP")
    dH_dT_V = makethermalfun("dH_dT_V")
    dH_dP_V = makethermalfun("dH_dP_V")
    dH_dV_T = makethermalfun("dH_dV_T")
    dH_dV_P = makethermalfun("dH_dV_P")
    dH_dzs = makethermalfun("dH_dzs")
    dS_dT = makethermalfun("dS_dT")
    dS_dP = makethermalfun("dS_dP")
    dS_dT_P = makethermalfun("dS_dT_P")
    dS_dT_V = makethermalfun("dS_dT_V")
    dS_dP_V = makethermalfun("dS_dP_V")
    dS_dzs = makethermalfun("dS_dzs")
    
    
    
    
gas_phases = (IdealGas, EOSGas, CoolPropGas)
liquid_phases = (EOSLiquid, GibbsExcessLiquid, CoolPropLiquid)
solid_phases = (GibbsExcessSolid,)
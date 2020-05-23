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
__all__ = ['Bulk', 'BulkSettings', 'default_settings']

from fluids.constants import R, R_inv, atm
from fluids.two_phase_voidage import (McAdams, Beattie_Whalley, Cicchitti, 
                                      Lin_Kwok, Fourar_Bories, Duckler, gas_liquid_viscosity)
from chemicals.utils import (log, exp, phase_identification_parameter,
                          isothermal_compressibility, isobaric_expansion,
                          Joule_Thomson, speed_of_sound)
from thermo.phases import Phase

'''Class designed to have multiple phases.

Calculates dew, bubble points as properties (going to call back to property package)
I guess it's going to need MW as well.

Does not have any flow property.
'''

MOLE_WEIGHTED = 'mole weighted'
MASS_WEIGHTED = 'mass weighted'
VOLUME_WEIGHTED = 'volume weighted'
EQUILIBRIUM_DERIVATIVE = 'Equilibrium derivative'
EQUILIBRIUM_DERIVATIVE_SAME_PHASES = 'Equilibrium at same phases'
MINIMUM_PHASE_PROP = 'minimum phase'
MAXIMUM_PHASE_PROP = 'maximum phase'
LIQUID_MOLE_WEIGHTED_PROP = 'liquid phase average, ignore other phases'

DP_DT_METHODS = [MOLE_WEIGHTED, MASS_WEIGHTED, EQUILIBRIUM_DERIVATIVE,
                 MINIMUM_PHASE_PROP, MAXIMUM_PHASE_PROP]

DP_DV_METHODS = D2P_DV2_METHODS = D2P_DT2_METHODS = D2P_DTDV_METHODS = DP_DT_METHODS


FROM_DERIVATIVE_SETTINGS = 'from derivative settings'

SPEED_OF_SOUND_METHODS = [MOLE_WEIGHTED, EQUILIBRIUM_DERIVATIVE]

BETA_METHODS = [MOLE_WEIGHTED, EQUILIBRIUM_DERIVATIVE, FROM_DERIVATIVE_SETTINGS,
                MASS_WEIGHTED, MINIMUM_PHASE_PROP, MAXIMUM_PHASE_PROP]
KAPPA_METHODS = JT_METHODS = BETA_METHODS


LOG_PROP_MOLE_WEIGHTED = 'log prop mole weighted'
LOG_PROP_MASS_WEIGHTED = 'log prop mass weighted'
LOG_PROP_VOLUME_WEIGHTED = 'log prop volume weighted'

POWER_PROP_MOLE_WEIGHTED = 'power prop mole weighted'
POWER_PROP_MASS_WEIGHTED = 'power prop mass weighted'
POWER_PROP_VOLUME_WEIGHTED = 'power prop volume weighted'

AS_ONE_LIQUID = 'as one liquid' # Calculate a transport property as if there was one liquid phase
AS_ONE_GAS = 'as one gas' # Calculate a transport property as if there was one gas phase and liquids or solids

MU_LL_METHODS = [MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED,
                 AS_ONE_LIQUID, 
                 LOG_PROP_MOLE_WEIGHTED, LOG_PROP_MASS_WEIGHTED, LOG_PROP_VOLUME_WEIGHTED,
                 POWER_PROP_MOLE_WEIGHTED, POWER_PROP_MASS_WEIGHTED, POWER_PROP_VOLUME_WEIGHTED
                 ]

BEATTIE_WHALLEY_MU_VL = 'Beattie Whalley'
MCADAMS_MU_VL = 'McAdams'
CICCHITTI_MU_VL = 'Cicchitti'
LUN_KWOK_MU_VL = 'Lin Kwok'
FOURAR_BORIES_MU_VL = 'Fourar Bories'
DUCKLER_MU_VL = 'Duckler'

MU_VL_CORRELATIONS = [BEATTIE_WHALLEY_MU_VL, MCADAMS_MU_VL, CICCHITTI_MU_VL,
                      LUN_KWOK_MU_VL, FOURAR_BORIES_MU_VL, DUCKLER_MU_VL]
MU_VL_CORRELATIONS_SET = set(MU_VL_CORRELATIONS)
MU_VL_METHODS = MU_LL_METHODS + [AS_ONE_GAS] + MU_VL_CORRELATIONS

K_LL_METHODS = MU_LL_METHODS

ViSCOSITY_METHODS = [MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED]
THERMAL_CONDUCTIVIVTY_METHODS = [MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED]


__all__.extend(['MOLE_WEIGHTED', 'MASS_WEIGHTED', 'VOLUME_WEIGHTED', 
                'LOG_PROP_MOLE_WEIGHTED', 'LOG_PROP_MASS_WEIGHTED', 'LOG_PROP_VOLUME_WEIGHTED',
                'POWER_PROP_MOLE_WEIGHTED', 'POWER_PROP_MASS_WEIGHTED', 'POWER_PROP_VOLUME_WEIGHTED',
                'AS_ONE_GAS', 'AS_ONE_LIQUID',
                'BEATTIE_WHALLEY_MU_VL', 'MCADAMS_MU_VL','CICCHITTI_MU_VL', 
                'LUN_KWOK_MU_VL', 'FOURAR_BORIES_MU_VL', 'DUCKLER_MU_VL',
                ])

mole_methods = set([MOLE_WEIGHTED, LOG_PROP_MOLE_WEIGHTED, POWER_PROP_MOLE_WEIGHTED])
mass_methods = set([MASS_WEIGHTED, LOG_PROP_MASS_WEIGHTED, POWER_PROP_MASS_WEIGHTED])
volume_methods = set([VOLUME_WEIGHTED, LOG_PROP_VOLUME_WEIGHTED, POWER_PROP_VOLUME_WEIGHTED])

linear_methods = set([MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED])
log_prop_methods = set([LOG_PROP_MOLE_WEIGHTED, LOG_PROP_MASS_WEIGHTED, LOG_PROP_VOLUME_WEIGHTED])
prop_power_methods = set([POWER_PROP_MOLE_WEIGHTED, POWER_PROP_MASS_WEIGHTED, POWER_PROP_VOLUME_WEIGHTED])



from thermo.phase_identification import VL_ID_PIP, S_ID_D2P_DVDT
from thermo.phase_identification import DENSITY_MASS, PROP_SORT, WATER_NOT_SPECIAL

class BulkSettings(object):
    def __init__(self, dP_dT=MOLE_WEIGHTED, dP_dV=MOLE_WEIGHTED, 
                 d2P_dV2=MOLE_WEIGHTED, d2P_dT2=MOLE_WEIGHTED, 
                 d2P_dTdV=MOLE_WEIGHTED, mu=MASS_WEIGHTED, k=MASS_WEIGHTED,
                 
                 mu_LL=LOG_PROP_MASS_WEIGHTED, mu_LL_power_exponent=0.4,
                 mu_VL=MCADAMS_MU_VL, mu_VL_power_exponent=0.4,
                 c=MOLE_WEIGHTED,
                 isobaric_expansion=MOLE_WEIGHTED, kappa=MOLE_WEIGHTED, JT=MOLE_WEIGHTED,
                 T_normal=288.15, P_normal=atm,
                 T_standard=288.7055555555555, P_standard=atm,
                 T_liquid_volume_ref=298.15,
                 T_gas_ref=288.15, P_gas_ref=atm,
                 
                 VL_ID=VL_ID_PIP, VL_ID_settings=None,
                 S_ID=S_ID_D2P_DVDT, S_ID_settings=None,

                 solid_sort_method=PROP_SORT,
                 liquid_sort_method=PROP_SORT,
                 liquid_sort_cmps=[], solid_sort_cmps=[],
                 liquid_sort_cmps_neg=[], solid_sort_cmps_neg=[],
                 liquid_sort_prop=DENSITY_MASS, 
                 solid_sort_prop=DENSITY_MASS,
                 phase_sort_higher_first=True,
                 water_sort=WATER_NOT_SPECIAL,
                 
                 ):
        self.dP_dT = dP_dT
        self.dP_dV = dP_dV
        self.d2P_dV2 = d2P_dV2
        self.d2P_dT2 = d2P_dT2
        self.d2P_dTdV = d2P_dTdV
        self.mu = mu
        self.mu_LL = mu_LL
        self.mu_LL_power_exponent = mu_LL_power_exponent
        self.mu_VL = mu_VL
        self.mu_VL_power_exponent = mu_VL_power_exponent
        
        self.k = k
        self.c = c
        self.T_normal = T_normal
        self.P_normal = P_normal
        self.T_standard = T_standard
        self.P_standard = P_standard
        self.T_liquid_volume_ref = T_liquid_volume_ref
        
        self.T_gas_ref = T_gas_ref
        self.P_gas_ref = P_gas_ref
        
        self.isobaric_expansion = isobaric_expansion
        self.kappa = kappa
        self.JT = JT
        self.VL_ID = VL_ID
        self.VL_ID_settings = VL_ID_settings
        
        self.S_ID = S_ID
        self.S_ID_settings = S_ID_settings
        
        
        # These are all lists of lists; can be any number; each has booleans,
        # length number of components
        self.liquid_sort_cmps = liquid_sort_cmps
        self.liquid_sort_cmps_neg = liquid_sort_cmps_neg
        self.solid_sort_cmps = solid_sort_cmps
        self.solid_sort_cmps_neg = solid_sort_cmps_neg
        
        self.liquid_sort_prop = liquid_sort_prop
        self.solid_sort_prop = solid_sort_prop
        
        self.phase_sort_higher_first = phase_sort_higher_first
        self.water_sort = water_sort
        
        self.solid_sort_method = solid_sort_method
        self.liquid_sort_method = liquid_sort_method
        
        self.phase_sort_higher_first = phase_sort_higher_first

default_settings = BulkSettings()

class Bulk(Phase):
    def __init__(self, T, P, zs, phases, phase_fractions, state=None):
        self.T = T
        self.P = P
        self.zs = zs
        self.phases = phases
        self.phase_fractions = phase_fractions
        self.N = N = len(zs)
        self.cmps = range(N)
        self.state = state
    


    @property
    def beta(self):
        return sum(self.phase_fractions)

    def mu(self):
        try:
            return self._mu
        except AttributeError:
            pass
        phase_fractions = self.phase_fractions
        phase_count = len(phase_fractions)
        result = self.result
        
        if phase_count == 1:
            self._mu = mu = self.phases[0].mu()
            return mu
        elif self.state == 'l' or self.result.gas is None:
            # Multiple liquids - either a bulk liquid, or a result with no gases
            method = self.settings.mu_LL
            if method == AS_ONE_LIQUID:
                mu = self.correlations.ViscosityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
            else:
                mus = [i.mu() for i in self.phases]
                if method in mole_methods:
                    betas = self.phase_fractions
                elif method in mass_methods:
                    betas = self.betas_mass
                elif method in volume_methods:
                    betas = self.betas_volume
                
                mu = 0.0
                if method in linear_methods:
                    for i in range(len(self.phase_fractions)):
                        mu += betas[i]*mus[i]
                elif method in prop_power_methods:
                    exponent = self.settings.mu_LL_power_exponent
                    for i in range(len(self.phase_fractions)):
                        mu += betas[i]*mus[i]**exponent
                    mu = mu**(1.0/exponent)
                elif method in log_prop_methods:
                    for i in range(len(self.phase_fractions)):
                        mu += betas[i]*log(mus[i])
                    mu = exp(mu)
            self._mu = mu
            return mu
        
        
        method = self.settings.mu_VL
        if method == AS_ONE_LIQUID:
            self._mu = mu = self.correlations.ViscosityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
            return mu
        elif method == AS_ONE_GAS:
            self._mu = mu = self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
            return mu
        
        mug = result.gas.mu()
        if phase_count == 2:
            mul = result.liquids[0].mu()
        else:
            mul = result.liquid_bulk.mu()
            
        if method in MU_VL_CORRELATIONS_SET:
            x = result.betas_mass[0]
            rhog = result.gas.rho_mass()
            if phase_count == 2:
                rhol = result.liquids[0].rho_mass()
            else:
                rhol = result.liquid_bulk.rho_mass()
            
            mu = gas_liquid_viscosity(x, mul, mug, rhol, rhog, method=method)
        else:
            mus = [mug, mul]
            if method in mole_methods:
                VF = self.result.beta_gas
                betas = [VF, 1.0 - VF]
            elif method in mass_methods:
                betas = self.betas_mass_states[:2]
            elif method in volume_methods:
                betas = self.betas_volume_states[:2]
                
            if method in linear_methods:
                mu = betas[0]*mus[0] + betas[1]*mus[1]
            elif method in prop_power_methods:
                exponent = self.settings.mu_LL_power_exponent
                mu = (betas[0]*mus[0]**exponent + betas[1]*mus[1]**exponent)**(1.0/exponent)
            elif method in log_prop_methods:
                mu = exp(betas[0]*log(mus[0]) + betas[1]*log(mus[1]))
        self._mu = mu
        return mu
            
            
                
    @property
    def betas_mass(self):
        betas = self.phase_fractions
        phase_iter = range(len(betas))
        MWs_phases = [i.MW() for i in self.phases]
        tot = 0.0
        for i in phase_iter:
            tot += MWs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        return [betas[i]*MWs_phases[i]*tot_inv for i in phase_iter]
    
    @property
    def betas_volume(self):
        betas = self.phase_fractions
        phase_iter = range(len(betas))
        Vs_phases = [i.V() for i in self.phases]
        
        tot = 0.0
        for i in phase_iter:
            tot += Vs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        return [betas[i]*Vs_phases[i]*tot_inv for i in phase_iter]


    def MW(self):
        MWs = self.constants.MWs
        zs = self.zs
        MW = 0.0
        for i in range(len(MWs)):
            MW += zs[i]*MWs[i]
        return MW
        
    def V(self):
        # Is there a point to anything else?
        try:
            return self._V
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        V = 0.0
        for i in range(len(betas)):
            V += betas[i]*phases[i].V()
        self._V = V
        return V
    
    def V_iter(self, force=False):
        betas, phases = self.phase_fractions, self.phases
        V = 0.0
        for i in range(len(betas)):
            V += betas[i]*phases[i].V_iter(force)
        return V
    
    
    def Cp(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        Cp = 0.0
        for i in range(len(betas)):
            Cp += betas[i]*phases[i].Cp()
        self._Cp = Cp
        return Cp

    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        H = 0.0
        for i in range(len(betas)):
            H += betas[i]*phases[i].H()
        self._H = H
        return H

    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        S = 0.0
        for i in range(len(betas)):
            S += betas[i]*phases[i].S()
        self._S = S
        return S
    
    def dH_dT(self):
        try:
            return self._dH_dT
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dH_dT = 0.0
        for i in range(len(betas)):
            dH_dT += betas[i]*phases[i].dH_dT()
        self._dH_dT = dH_dT
        return dH_dT

    def dH_dP(self):
        try:
            return self._dH_dP
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dH_dP = 0.0
        for i in range(len(betas)):
            dH_dP += betas[i]*phases[i].dH_dP()
        self._dH_dP = dH_dP
        return dH_dP

    def dS_dP(self):
        try:
            return self._dS_dP
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dS_dP = 0.0
        for i in range(len(betas)):
            dS_dP += betas[i]*phases[i].dS_dP()
        self._dS_dP = dS_dP
        return dS_dP

    def dS_dT(self):
        try:
            return self._dS_dT
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dS_dT = 0.0
        for i in range(len(betas)):
            dS_dT += betas[i]*phases[i].dS_dT()
        self._dS_dT = dS_dT
        return dS_dT

    def dG_dT(self):
        try:
            return self._dG_dT
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dG_dT = 0.0
        for i in range(len(betas)):
            dG_dT += betas[i]*phases[i].dG_dT()
        self._dG_dT = dG_dT
        return dG_dT

    def dG_dP(self):
        try:
            return self._dG_dP
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dG_dP = 0.0
        for i in range(len(betas)):
            dG_dP += betas[i]*phases[i].dG_dP()
        self._dG_dP = dG_dP
        return dG_dP

    def dU_dT(self):
        try:
            return self._dU_dT
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dU_dT = 0.0
        for i in range(len(betas)):
            dU_dT += betas[i]*phases[i].dU_dT()
        self._dU_dT = dU_dT
        return dU_dT

    def dU_dP(self):
        try:
            return self._dU_dP
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dU_dP = 0.0
        for i in range(len(betas)):
            dU_dP += betas[i]*phases[i].dU_dP()
        self._dU_dP = dU_dP
        return dU_dP

    def dA_dT(self):
        try:
            return self._dA_dT
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dA_dT = 0.0
        for i in range(len(betas)):
            dA_dT += betas[i]*phases[i].dA_dT()
        self._dA_dT = dA_dT
        return dA_dT

    def dA_dP(self):
        try:
            return self._dA_dP
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dA_dP = 0.0
        for i in range(len(betas)):
            dA_dP += betas[i]*phases[i].dA_dP()
        self._dA_dP = dA_dP
        return dA_dP
    
    def H_reactive(self):
        try:
            return self._H_reactive
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        H_reactive = 0.0
        for i in range(len(betas)):
            H_reactive += betas[i]*phases[i].H_reactive()
        self._H_reactive = H_reactive
        return H_reactive

    def S_reactive(self):
        try:
            return self._S_reactive
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        S_reactive = 0.0
        for i in range(len(betas)):
            S_reactive += betas[i]*phases[i].S_reactive()
        self._S_reactive = S_reactive
        return S_reactive

    def dP_dT_frozen(self):
        try:
            return self._dP_dT_frozen
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dP_dT_frozen = 0.0
        for i in range(len(betas)):
            dP_dT_frozen += betas[i]*phases[i].dP_dT()
        self._dP_dT_frozen = dP_dT_frozen
        return dP_dT_frozen

    def dP_dV_frozen(self):
        try:
            return self._dP_dV_frozen
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dP_dV_frozen = 0.0
        for i in range(len(betas)):
            dP_dV_frozen += betas[i]*phases[i].dP_dV()
        self._dP_dV_frozen = dP_dV_frozen
        return dP_dV_frozen

    def d2P_dT2_frozen(self):
        try:
            return self._d2P_dT2_frozen
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        d2P_dT2_frozen = 0.0
        for i in range(len(betas)):
            d2P_dT2_frozen += betas[i]*phases[i].d2P_dT2()
        self._d2P_dT2_frozen = d2P_dT2_frozen
        return d2P_dT2_frozen

    def d2P_dV2_frozen(self):
        try:
            return self._d2P_dV2_frozen
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        d2P_dV2_frozen = 0.0
        for i in range(len(betas)):
            d2P_dV2_frozen += betas[i]*phases[i].d2P_dV2()
        self._d2P_dV2_frozen = d2P_dV2_frozen
        return d2P_dV2_frozen

    def d2P_dTdV_frozen(self):
        try:
            return self._d2P_dTdV_frozen
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        d2P_dTdV_frozen = 0.0
        for i in range(len(betas)):
            d2P_dTdV_frozen += betas[i]*phases[i].d2P_dTdV()
        self._d2P_dTdV_frozen = d2P_dTdV_frozen
        return d2P_dTdV_frozen

    def dP_dT_equilibrium(self):
        # At constant volume
        try:
            return self._dP_dT_equilibrium
        except AttributeError:
            pass
        
        dP_dT_1 = self.dP_dT_frozen()
        dT = self.T*1e-6
        T2 = self.T + dT
        
        dT_results = self.flasher.flash(T=T2, V=self.V(), zs=self.zs)
#        dP_dT_2 = dT_results.bulk.dP_dT_frozen()
        dP_dT_equilibrium = (dT_results.P - self.P)/dT

        self._dP_dT_equilibrium = dP_dT_equilibrium
        return dP_dT_equilibrium

    def dP_dT(self):
        dP_dT_method = self.settings.dP_dT
        if dP_dT_method == MOLE_WEIGHTED:
            return self.dP_dT_frozen()
        elif dP_dT_method == EQUILIBRIUM_DERIVATIVE:
            return self.dP_dT_equilibrium()
        elif dP_dT_method in (MASS_WEIGHTED, MINIMUM_PHASE_PROP, 
                              MAXIMUM_PHASE_PROP):
            phases = self.phases
            dP_dTs = [p.dP_dT() for p in phases]
            
            if dP_dT_method == MASS_WEIGHTED:
                ws = self.result.betas_mass
                dP_dT = 0.0
                for i in range(len(ws)):
                    dP_dT += ws[i]*dP_dTs[i]
                return dP_dT
            elif dP_dT_method == MINIMUM_PHASE_PROP:
                return min(dP_dTs)
            elif dP_dT_method == MAXIMUM_PHASE_PROP:
                return max(dP_dTs)
        else:
            raise ValueError("Unspecified error")

    def dP_dV(self):
        dP_dV_method = self.settings.dP_dV
        if dP_dV_method == MOLE_WEIGHTED:
            return self.dP_dV_frozen()
        elif dP_dV_method == EQUILIBRIUM_DERIVATIVE:
            return self.dP_dV_equilibrium()
        elif dP_dV_method in (MASS_WEIGHTED, MINIMUM_PHASE_PROP, 
                              MAXIMUM_PHASE_PROP):
            phases = self.phases
            dP_dVs = [p.dP_dV() for p in phases]
            
            if dP_dV_method == MASS_WEIGHTED:
                ws = self.result.betas_mass
                dP_dV = 0.0
                for i in range(len(ws)):
                    dP_dV += ws[i]*dP_dVs[i]
                return dP_dV
            elif dP_dV_method == MINIMUM_PHASE_PROP:
                return min(dP_dVs)
            elif dP_dV_method == MAXIMUM_PHASE_PROP:
                return max(dP_dVs)
        else:
            raise ValueError("Unspecified error")

    def d2P_dT2(self):
        d2P_dT2_method = self.settings.d2P_dT2
        if d2P_dT2_method == MOLE_WEIGHTED:
            return self.d2P_dT2_frozen()
        elif d2P_dT2_method == EQUILIBRIUM_DERIVATIVE:
            return self.d2P_dT2_equilibrium()
        elif d2P_dT2_method in (MASS_WEIGHTED, MINIMUM_PHASE_PROP, 
                              MAXIMUM_PHASE_PROP):
            phases = self.phases
            d2P_dT2s = [p.d2P_dT2() for p in phases]
            
            if d2P_dT2_method == MASS_WEIGHTED:
                ws = self.result.betas_mass
                d2P_dT2 = 0.0
                for i in range(len(ws)):
                    d2P_dT2 += ws[i]*d2P_dT2s[i]
                return d2P_dT2
            elif d2P_dT2_method == MINIMUM_PHASE_PROP:
                return min(d2P_dT2s)
            elif d2P_dT2_method == MAXIMUM_PHASE_PROP:
                return max(d2P_dT2s)
        else:
            raise ValueError("Unspecified error")

    def d2P_dV2(self):
        d2P_dV2_method = self.settings.d2P_dV2
        if d2P_dV2_method == MOLE_WEIGHTED:
            return self.d2P_dV2_frozen()
        elif d2P_dV2_method == EQUILIBRIUM_DERIVATIVE:
            return self.d2P_dV2_equilibrium()
        elif d2P_dV2_method in (MASS_WEIGHTED, MINIMUM_PHASE_PROP, 
                              MAXIMUM_PHASE_PROP):
            phases = self.phases
            d2P_dV2s = [p.d2P_dV2() for p in phases]
            
            if d2P_dV2_method == MASS_WEIGHTED:
                ws = self.result.betas_mass
                d2P_dV2 = 0.0
                for i in range(len(ws)):
                    d2P_dV2 += ws[i]*d2P_dV2s[i]
                return d2P_dV2
            elif d2P_dV2_method == MINIMUM_PHASE_PROP:
                return min(d2P_dV2s)
            elif d2P_dV2_method == MAXIMUM_PHASE_PROP:
                return max(d2P_dV2s)
        else:
            raise ValueError("Unspecified error")

    def d2P_dTdV(self):
        d2P_dTdV_method = self.settings.d2P_dTdV
        if d2P_dTdV_method == MOLE_WEIGHTED:
            return self.d2P_dTdV_frozen()
        elif d2P_dTdV_method == EQUILIBRIUM_DERIVATIVE:
            return self.d2P_dTdV_equilibrium()
        elif d2P_dTdV_method in (MASS_WEIGHTED, MINIMUM_PHASE_PROP, 
                              MAXIMUM_PHASE_PROP):
            phases = self.phases
            d2P_dTdVs = [p.d2P_dTdV() for p in phases]
            
            if d2P_dTdV_method == MASS_WEIGHTED:
                ws = self.result.betas_mass
                d2P_dTdV = 0.0
                for i in range(len(ws)):
                    d2P_dTdV += ws[i]*d2P_dTdVs[i]
                return d2P_dTdV
            elif d2P_dTdV_method == MINIMUM_PHASE_PROP:
                return min(d2P_dTdVs)
            elif d2P_dTdV_method == MAXIMUM_PHASE_PROP:
                return max(d2P_dTdVs)
        else:
            raise ValueError("Unspecified error")
            
            
    def isobaric_expansion(self):
        beta_method = self.settings.isobaric_expansion
        if beta_method == EQUILIBRIUM_DERIVATIVE:
            return self.beta_equilibrium()
        elif beta_method == FROM_DERIVATIVE_SETTINGS:
            return isobaric_expansion(self.V(), self.dV_dT())
        elif beta_method in (MOLE_WEIGHTED, MASS_WEIGHTED, MINIMUM_PHASE_PROP, 
                             MAXIMUM_PHASE_PROP):
            phases = self.phases
            betas = [p.isobaric_expansion() for p in phases]
            
            if beta_method == MOLE_WEIGHTED:
                phase_fracs = self.phase_fractions
                beta = 0.0
                for i in range(len(phase_fracs)):
                    beta += phase_fracs[i]*betas[i]
                return beta
            elif beta_method == MASS_WEIGHTED:
                ws = self.result.betas_mass
                beta = 0.0
                for i in range(len(ws)):
                    beta += ws[i]*betas[i]
                return beta
            elif beta_method == MINIMUM_PHASE_PROP:
                return min(betas)
            elif beta_method == MAXIMUM_PHASE_PROP:
                return max(betas)
        else:
            raise ValueError("Unspecified error")

    def kappa(self):
        kappa_method = self.settings.kappa
        if kappa_method == EQUILIBRIUM_DERIVATIVE:
            return self.kappa_equilibrium()
        elif kappa_method == FROM_DERIVATIVE_SETTINGS:
            return isobaric_expansion(self.V(), self.dV_dT())
        elif kappa_method in (MOLE_WEIGHTED, MASS_WEIGHTED, MINIMUM_PHASE_PROP, 
                             MAXIMUM_PHASE_PROP):
            phases = self.phases
            kappas = [p.kappa() for p in phases]
            
            if kappa_method == MOLE_WEIGHTED:
                phase_fracs = self.phase_fractions
                kappa = 0.0
                for i in range(len(phase_fracs)):
                    kappa += phase_fracs[i]*kappas[i]
                return kappa
            elif kappa_method == MASS_WEIGHTED:
                ws = self.result.kappas_mass
                kappa = 0.0
                for i in range(len(ws)):
                    kappa += ws[i]*kappas[i]
                return kappa
            elif kappa_method == MINIMUM_PHASE_PROP:
                return min(kappas)
            elif kappa_method == MAXIMUM_PHASE_PROP:
                return max(kappas)
        else:
            raise ValueError("Unspecified error")

    def Joule_Thomson(self):
        Joule_Thomson_method = self.settings.JT
        if Joule_Thomson_method == EQUILIBRIUM_DERIVATIVE:
            return self.Joule_Thomson_equilibrium()
        elif Joule_Thomson_method == FROM_DERIVATIVE_SETTINGS:
            return isobaric_expansion(self.V(), self.dV_dT())
        elif Joule_Thomson_method in (MOLE_WEIGHTED, MASS_WEIGHTED, MINIMUM_PHASE_PROP, 
                             MAXIMUM_PHASE_PROP):
            phases = self.phases
            Joule_Thomsons = [p.Joule_Thomson() for p in phases]
            
            if Joule_Thomson_method == MOLE_WEIGHTED:
                phase_fracs = self.phase_fractions
                Joule_Thomson = 0.0
                for i in range(len(phase_fracs)):
                    Joule_Thomson += phase_fracs[i]*Joule_Thomsons[i]
                return Joule_Thomson
            elif Joule_Thomson_method == MASS_WEIGHTED:
                ws = self.result.Joule_Thomsons_mass
                Joule_Thomson = 0.0
                for i in range(len(ws)):
                    Joule_Thomson += ws[i]*Joule_Thomsons[i]
                return Joule_Thomson
            elif Joule_Thomson_method == MINIMUM_PHASE_PROP:
                return min(Joule_Thomsons)
            elif Joule_Thomson_method == MAXIMUM_PHASE_PROP:
                return max(Joule_Thomsons)
        else:
            raise ValueError("Unspecified error")

    def speed_of_sound(self):
        speed_of_sound_method = self.settings.c
        if speed_of_sound_method == EQUILIBRIUM_DERIVATIVE:
            return self.speed_of_sound_equilibrium()
        elif speed_of_sound_method == FROM_DERIVATIVE_SETTINGS:
            return isobaric_expansion(self.V(), self.dV_dT())
        elif speed_of_sound_method in (MOLE_WEIGHTED, MASS_WEIGHTED, MINIMUM_PHASE_PROP, 
                             MAXIMUM_PHASE_PROP):
            phases = self.phases
            speed_of_sounds = [p.speed_of_sound() for p in phases]
            
            if speed_of_sound_method == MOLE_WEIGHTED:
                phase_fracs = self.phase_fractions
                speed_of_sound = 0.0
                for i in range(len(phase_fracs)):
                    speed_of_sound += phase_fracs[i]*speed_of_sounds[i]
                return speed_of_sound
            elif speed_of_sound_method == MASS_WEIGHTED:
                ws = self.result.speed_of_sounds_mass
                speed_of_sound = 0.0
                for i in range(len(ws)):
                    speed_of_sound += ws[i]*speed_of_sounds[i]
                return speed_of_sound
            elif speed_of_sound_method == MINIMUM_PHASE_PROP:
                return min(speed_of_sounds)
            elif speed_of_sound_method == MAXIMUM_PHASE_PROP:
                return max(speed_of_sounds)
        else:
            raise ValueError("Unspecified error")

    def Tmc(self):
        try:
            return self._Tmc
        except AttributeError:
            pass
        betas, phases = self.phase_fractions, self.phases
        Tmc = 0.0
        for i in range(len(betas)):
            Tmc += betas[i]*phases[i].Tmc()
        self._Tmc = Tmc
        return Tmc

    def Pmc(self):
        try:
            return self._Pmc
        except AttributeError:
            pass
        betas, phases = self.phase_fractions, self.phases
        Pmc = 0.0
        for i in range(len(betas)):
            Pmc += betas[i]*phases[i].Pmc()
        self._Pmc = Pmc
        return Pmc

    def Vmc(self):
        try:
            return self._Vmc
        except AttributeError:
            pass
        betas, phases = self.phase_fractions, self.phases
        Vmc = 0.0
        for i in range(len(betas)):
            Vmc += betas[i]*phases[i].Vmc()
        self._Vmc = Vmc
        return Vmc

    def Zmc(self):
        try:
            return self._Zmc
        except AttributeError:
            pass
        betas, phases = self.phase_fractions, self.phases
        Zmc = 0.0
        for i in range(len(betas)):
            Zmc += betas[i]*phases[i].Zmc()
        self._Zmc = Zmc
        return Zmc

### Functions depending on correlations - here for speed
    def H_ideal_gas(self):
        HeatCapacityGases = self.correlations.HeatCapacityGases
        T, T_REF_IG = self.T, self.T_REF_IG
        Cpig_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        H = 0.0
        for zi, Cp_int in zip(self.zs, Cpig_integrals_pure):
            H += zi*Cp_int
        return H

    def Cp_ideal_gas(self):
        HeatCapacityGases = self.correlations.HeatCapacityGases
        T = self.T
        Cpigs_pure = [i.T_dependent_property(T) for i in HeatCapacityGases]

        Cp, zs = 0.0, self.zs
        for i in self.cmps:
            Cp += zs[i]*Cpigs_pure[i]
        return Cp

    def S_ideal_gas(self):
        HeatCapacityGases = self.correlations.HeatCapacityGases
        T, T_REF_IG = self.T, self.T_REF_IG

        Cpig_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                      for obj in HeatCapacityGases]

        log_zs = self.log_zs()
        T, P, zs, cmps = self.T, self.P, self.zs, self.cmps
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)
        
        for i in cmps:
            S += zs[i]*Cpig_integrals_over_T_pure[i]

        return S

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
SOFTWARE.

This module contains a phase wrapper for obtaining properties of a pseudo-phase
made of multiple other phases. This is useful in the context of multiple liquid
phases; or multiple solid phases; or looking at all the phases together.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Bulk Class
==========
.. autoclass:: Bulk
    :members: beta, mu, betas_mass, betas_volume, MW, V, V_iter, Cp, H, S,
              dH_dP, dS_dP, dS_dT, dG_dT, dG_dP, dU_dT, dU_dP, dA_dT, dA_dP,
              H_reactive, S_reactive, dP_dT_frozen, dP_dV_frozen,
              d2P_dT2_frozen, d2P_dV2_frozen, d2P_dTdV_frozen,
              dP_dT_equilibrium, dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV,
              isobaric_expansion, kappa, Joule_Thomson, speed_of_sound, Tmc,
              Pmc, Vmc, Zmc, H_ideal_gas, Cp_ideal_gas, S_ideal_gas
    :undoc-members:
    :show-inheritance:
    :exclude-members:

Bulk Settings Class
===================

.. autoclass:: BulkSettings
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: MU_LL_METHODS

'''

from __future__ import division
__all__ = ['Bulk', 'BulkSettings', 'default_settings']

from fluids.constants import R, R_inv, atm
from fluids.two_phase_voidage import (McAdams, Beattie_Whalley, Cicchitti,
                                      Lin_Kwok, Fourar_Bories, Duckler, gas_liquid_viscosity)
from chemicals.utils import (log, exp, phase_identification_parameter,
                          isothermal_compressibility, isobaric_expansion,
                          Joule_Thomson, speed_of_sound)
from thermo.phases import Phase
from thermo.phase_identification import VL_ID_PIP, S_ID_D2P_DVDT
from thermo.phase_identification import DENSITY_MASS, PROP_SORT, WATER_NOT_SPECIAL

'''Class designed to have multiple phases.

Calculates dew, bubble points as properties (going to call back to property package)
I guess it's going to need MW as well.

Does not have any flow property.
'''

EQUILIBRIUM_DERIVATIVE = 'Equilibrium derivative'
EQUILIBRIUM_DERIVATIVE_SAME_PHASES = 'Equilibrium at same phases'
FROM_DERIVATIVE_SETTINGS = 'from derivative settings'

MOLE_WEIGHTED = 'MOLE_WEIGHTED'
MASS_WEIGHTED = 'MASS_WEIGHTED'
VOLUME_WEIGHTED = 'VOLUME_WEIGHTED'

MINIMUM_PHASE_PROP = 'MINIMUM_PHASE_PROP'
MAXIMUM_PHASE_PROP = 'MAXIMUM_PHASE_PROP'

DP_DT_METHODS = [MOLE_WEIGHTED, MASS_WEIGHTED, EQUILIBRIUM_DERIVATIVE,
                 MINIMUM_PHASE_PROP, MAXIMUM_PHASE_PROP]

DP_DV_METHODS = D2P_DV2_METHODS = D2P_DT2_METHODS = D2P_DTDV_METHODS = DP_DT_METHODS

SPEED_OF_SOUND_METHODS = [MOLE_WEIGHTED, EQUILIBRIUM_DERIVATIVE]

BETA_METHODS = [MOLE_WEIGHTED, EQUILIBRIUM_DERIVATIVE, FROM_DERIVATIVE_SETTINGS,
                MASS_WEIGHTED, MINIMUM_PHASE_PROP, MAXIMUM_PHASE_PROP]
KAPPA_METHODS = JT_METHODS = BETA_METHODS


LOG_PROP_MOLE_WEIGHTED = 'LOG_PROP_MOLE_WEIGHTED'
LOG_PROP_MASS_WEIGHTED = 'LOG_PROP_MASS_WEIGHTED'
LOG_PROP_VOLUME_WEIGHTED = 'LOG_PROP_VOLUME_WEIGHTED'

POWER_PROP_MOLE_WEIGHTED = 'POWER_PROP_MOLE_WEIGHTED'
POWER_PROP_MASS_WEIGHTED = 'POWER_PROP_MASS_WEIGHTED'
POWER_PROP_VOLUME_WEIGHTED = 'POWER_PROP_VOLUME_WEIGHTED'

AS_ONE_LIQUID = 'AS_ONE_LIQUID' # Calculate a transport property as if there was one liquid phase
AS_ONE_GAS = 'AS_ONE_GAS' # Calculate a transport property as if there was one gas phase and liquids or solids

MU_LL_METHODS = [MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED,
                 AS_ONE_LIQUID,
                 LOG_PROP_MOLE_WEIGHTED, LOG_PROP_MASS_WEIGHTED, LOG_PROP_VOLUME_WEIGHTED,
                 POWER_PROP_MOLE_WEIGHTED, POWER_PROP_MASS_WEIGHTED, POWER_PROP_VOLUME_WEIGHTED
                 ]
'''List of all valid and implemented mixing rules for the `MU_LL` setting'''

MU_LL_METHODS_set = frozenset(MU_LL_METHODS)

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




class BulkSettings(object):
    r'''Class containing configuration methods for determining how properties of
    a `Bulk` phase made of different phases are handled.

    Parameters
    ----------
    mu_LL : str
        Mixing rule for multiple liquid phase, liquid viscosity calculations,
        [-]
    mu_LL_power_exponent : float
        Liquid-liquid power-law mixing parameter, used only when a power
        law mixing rule is selected, [-]

    Notes
    -----

    The linear mixing rules "MOLE_WEIGHTED", "MASS_WEIGHTED", and
    "VOLUME_WEIGHTED" have the following formula, with :math:`\beta`
    representing molar, mass, or volume phase fraction:

    .. math::
        \text{bulk property} = \left(\sum_i^{phases} \beta_i \text{property}
        \right)

    The power mixing rules "POWER_PROP_MOLE_WEIGHTED",
    "POWER_PROP_MASS_WEIGHTED", and "POWER_PROP_VOLUME_WEIGHTED" have the
    following formula, with :math:`\beta` representing molar, mass, or volume
    phase fraction:

    .. math::
        \text{bulk property} = \left(\sum_i^{phases} \beta_i \text{property
        }^{\text{exponent}} \right)^{1/\text{exponent}}

    The logarithmic mixing rules "LOG_PROP_MOLE_WEIGHTED",
    "LOG_PROP_MASS_WEIGHTED", and "LOG_PROP_VOLUME_WEIGHTED" have the
    following formula, with :math:`\beta` representing molar, mass, or volume
    phase fraction:

    .. math::
        \text{bulk property} = \exp\left(\sum_i^{phases} \beta_i \ln(\text{property
        })\right)

    The mixing rule "MINIMUM_PHASE_PROP" selects the lowest phase value of the
    property, always. The mixing rule "MAXIMUM_PHASE_PROP" selects the highest
    phase value of the property, always.

    The mixing rule "AS_ONE_LIQUID" calculates a property using the bulk
    composition but applied to the liquid model only.
    The mixing rule "AS_ONE_GAS" calculates a property using the bulk
    composition but applied to the gas model only.

    '''
    __full_path__ = "%s.%s" %(__module__, __qualname__)

    def as_json(self):
        return self.__dict__.copy()

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
        if mu_LL not in MU_LL_METHODS_set:
            raise ValueError("Unrecognized option for mu_LL")
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
        '''float : Temperature of the liquid volume reference, [K]'''

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
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    def __init__(self, T, P, zs, phases, phase_fractions, state=None):
        self.T = T
        self.P = P
        self.zs = zs
        self.phases = phases
        self.phase_fractions = phase_fractions
        self.N = N = len(zs)
        self.state = state



    @property
    def beta(self):
        r'''Phase fraction of the bulk phase. Should always be one when
        representing all phases of a flash; but can be less than one if
        representing multiple solids or liquids as a single phase in a larger
        mixture.

        Returns
        -------
        beta : float
            Phase fraction of bulk, [-]
        '''
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

            mu = gas_liquid_viscosity(x, mul, mug, rhol, rhog, Method=method)
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
        r'''Method to calculate and return the molecular weight of the bulk
        phase. This is a phase-fraction weighted calculation.

        .. math::
            \text{MW} = \sum_i^p \text{MW}_i \beta_i

        Returns
        -------
        MW : float
            Molecular weight, [g/mol]
        '''
        try:
            return self._MW
        except:
            pass
        MWs = self.constants.MWs
        zs = self.zs
        MW = 0.0
        for i in range(len(MWs)):
            MW += zs[i]*MWs[i]
        self._MW = MW
        return MW

    def V(self):
        r'''Method to calculate and return the molar volume of the bulk phase.
        This is a phase-fraction weighted calculation.

        .. math::
            V = \sum_i^p V_i \beta_i

        Returns
        -------
        V : float
            Molar volume, [m^3/mol]
        '''
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
        r'''Method to calculate and return the molar volume of the bulk phase,
        with precision suitable for a `TV` calculation to calculate a matching
        pressure. This is a phase-fraction weighted calculation.

        .. math::
            V = \sum_i^p V_i \beta_i

        Returns
        -------
        V : float or mpf
            Molar volume, [m^3/mol]
        '''
        betas, phases = self.phase_fractions, self.phases
        V = 0.0
        for i in range(len(betas)):
            V += betas[i]*phases[i].V_iter(force)
        return V


    def Cp(self):
        r'''Method to calculate and return the constant-temperature and
        constant phase-fraction heat capacity of the bulk phase.
        This is a phase-fraction weighted calculation.

        .. math::
            C_p = \sum_i^p C_{p,i} \beta_i

        Returns
        -------
        Cp : float
            Molar heat capacity, [J/(mol*K)]
        '''
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

    dH_dT = Cp

    def H(self):
        r'''Method to calculate and return the constant-temperature and
        constant phase-fraction enthalpy of the bulk phase.
        This is a phase-fraction weighted calculation.

        .. math::
            H = \sum_i^p H_{i} \beta_i

        Returns
        -------
        H : float
            Molar enthalpy, [J/(mol)]
        '''
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
        r'''Method to calculate and return the constant-temperature and
        constant phase-fraction entropy of the bulk phase.
        This is a phase-fraction weighted calculation.

        .. math::
            S = \sum_i^p S_{i} \beta_i

        Returns
        -------
        S : float
            Molar entropy, [J/(mol*K)]
        '''
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
        for i in range(self.N):
            Cp += zs[i]*Cpigs_pure[i]
        return Cp

    def S_ideal_gas(self):
        HeatCapacityGases = self.correlations.HeatCapacityGases
        T, T_REF_IG = self.T, self.T_REF_IG

        Cpig_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                      for obj in HeatCapacityGases]

        log_zs = self.log_zs()
        T, P, zs, cmps = self.T, self.P, self.zs, range(self.N)
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)

        for i in cmps:
            S += zs[i]*Cpig_integrals_over_T_pure[i]

        return S

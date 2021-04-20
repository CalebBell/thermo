# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020, 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
    :members: beta, betas_mass, betas_volume, k, mu, sigma, MW, V, V_iter, Cp, H, S,
              dG_dT, dG_dP, dU_dT, dU_dP, dA_dT, dA_dP,
              H_reactive, S_reactive, dP_dT_frozen, dP_dV_frozen,
              d2P_dT2_frozen, d2P_dV2_frozen, d2P_dTdV_frozen,
              dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV,
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

.. autodata:: DP_DT_METHODS
.. autodata:: DP_DV_METHODS
.. autodata:: D2P_DV2_METHODS
.. autodata:: D2P_DT2_METHODS
.. autodata:: D2P_DTDV_METHODS


.. autodata:: MU_LL_METHODS
.. autodata:: MU_VL_METHODS
.. autodata:: K_LL_METHODS
.. autodata:: K_VL_METHODS
.. autodata:: SIGMA_LL_METHODS

.. autodata:: BETA_METHODS
.. autodata:: SPEED_OF_SOUND_METHODS
.. autodata:: KAPPA_METHODS
.. autodata:: JT_METHODS

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
from thermo.chemical_package import ChemicalConstantsPackage
'''Class designed to have multiple phases.

Calculates dew, bubble points as properties (going to call back to property package)
I guess it's going to need MW as well.

Does not have any flow property.
'''

EQUILIBRIUM_DERIVATIVE = 'EQUILIBRIUM_DERIVATIVE'
FROM_DERIVATIVE_SETTINGS = 'FROM_DERIVATIVE_SETTINGS'

MOLE_WEIGHTED = 'MOLE_WEIGHTED'
MASS_WEIGHTED = 'MASS_WEIGHTED'
VOLUME_WEIGHTED = 'VOLUME_WEIGHTED'

LOG_PROP_MOLE_WEIGHTED = 'LOG_PROP_MOLE_WEIGHTED'
LOG_PROP_MASS_WEIGHTED = 'LOG_PROP_MASS_WEIGHTED'
LOG_PROP_VOLUME_WEIGHTED = 'LOG_PROP_VOLUME_WEIGHTED'

POWER_PROP_MOLE_WEIGHTED = 'POWER_PROP_MOLE_WEIGHTED'
POWER_PROP_MASS_WEIGHTED = 'POWER_PROP_MASS_WEIGHTED'
POWER_PROP_VOLUME_WEIGHTED = 'POWER_PROP_VOLUME_WEIGHTED'

MINIMUM_PHASE_PROP = 'MINIMUM_PHASE_PROP'
MAXIMUM_PHASE_PROP = 'MAXIMUM_PHASE_PROP'

DP_DT_METHODS = [MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED,
                 LOG_PROP_MOLE_WEIGHTED, LOG_PROP_MASS_WEIGHTED, LOG_PROP_VOLUME_WEIGHTED,
                 EQUILIBRIUM_DERIVATIVE,
                 MINIMUM_PHASE_PROP, MAXIMUM_PHASE_PROP]
'''List of all valid and implemented calculation methods for the `DP_DT` bulk setting'''
DP_DV_METHODS = DP_DT_METHODS
'''List of all valid and implemented calculation methods for the `DP_DV` bulk setting'''
D2P_DV2_METHODS =  [MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED,
                 LOG_PROP_MOLE_WEIGHTED, LOG_PROP_MASS_WEIGHTED, LOG_PROP_VOLUME_WEIGHTED,
                 MINIMUM_PHASE_PROP, MAXIMUM_PHASE_PROP]
'''List of all valid and implemented calculation methods for the `D2P_DV2` bulk setting'''
D2P_DT2_METHODS = D2P_DV2_METHODS
'''List of all valid and implemented calculation methods for the `D2P_DT2` bulk setting'''
D2P_DTDV_METHODS = D2P_DV2_METHODS
'''List of all valid and implemented calculation methods for the `D2P_DTDV` bulk setting'''


SPEED_OF_SOUND_METHODS = [MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED,
                LOG_PROP_MOLE_WEIGHTED, LOG_PROP_MASS_WEIGHTED,
                LOG_PROP_VOLUME_WEIGHTED, MINIMUM_PHASE_PROP,
                MAXIMUM_PHASE_PROP, FROM_DERIVATIVE_SETTINGS]
'''List of all valid and implemented calculation methods for the `speed_of_sound` bulk setting'''

BETA_METHODS = [MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED,
                LOG_PROP_MOLE_WEIGHTED, LOG_PROP_MASS_WEIGHTED,
                LOG_PROP_VOLUME_WEIGHTED, MINIMUM_PHASE_PROP,
                MAXIMUM_PHASE_PROP, EQUILIBRIUM_DERIVATIVE,
                FROM_DERIVATIVE_SETTINGS]
'''List of all valid and implemented calculation methods for the `isothermal_compressibility` bulk setting'''


KAPPA_METHODS = BETA_METHODS
'''List of all valid and implemented calculation methods for the `kappa` bulk setting'''
JT_METHODS = BETA_METHODS
'''List of all valid and implemented calculation methods for the `JT` bulk setting'''




AS_ONE_LIQUID = 'AS_ONE_LIQUID' # Calculate a transport property as if there was one liquid phase
AS_ONE_GAS = 'AS_ONE_GAS' # Calculate a transport property as if there was one gas phase and liquids or solids

MU_LL_METHODS = [MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED,
                 AS_ONE_LIQUID,
                 LOG_PROP_MOLE_WEIGHTED, LOG_PROP_MASS_WEIGHTED, LOG_PROP_VOLUME_WEIGHTED,
                 POWER_PROP_MOLE_WEIGHTED, POWER_PROP_MASS_WEIGHTED, POWER_PROP_VOLUME_WEIGHTED,
                 MINIMUM_PHASE_PROP, MAXIMUM_PHASE_PROP
                 ]
'''List of all valid and implemented mixing rules for the `MU_LL` setting'''

MU_LL_METHODS_set = frozenset(MU_LL_METHODS)

K_LL_METHODS = MU_LL_METHODS
'''List of all valid and implemented mixing rules for the `K_LL` setting'''
K_LL_METHODS_set = frozenset(K_LL_METHODS)

SIGMA_LL_METHODS = MU_LL_METHODS
'''List of all valid and implemented mixing rules for the `SIGMA_LL` setting'''
SIGMA_LL_METHODS_set = frozenset(SIGMA_LL_METHODS)


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
'''List of all valid and implemented mixing rules for the `MU_VL` setting'''
MU_VL_METHODS_SET = set(MU_VL_METHODS)

K_VL_METHODS = K_LL_METHODS + [AS_ONE_GAS]
'''List of all valid and implemented mixing rules for the `K_VL` setting'''
K_VL_METHODS_SET = set(K_VL_METHODS)

__all__.extend(['MOLE_WEIGHTED', 'MASS_WEIGHTED', 'VOLUME_WEIGHTED', 'EQUILIBRIUM_DERIVATIVE',
                'LOG_PROP_MOLE_WEIGHTED', 'LOG_PROP_MASS_WEIGHTED', 'LOG_PROP_VOLUME_WEIGHTED',
                'POWER_PROP_MOLE_WEIGHTED', 'POWER_PROP_MASS_WEIGHTED', 'POWER_PROP_VOLUME_WEIGHTED',
                'AS_ONE_GAS', 'AS_ONE_LIQUID',
                'BEATTIE_WHALLEY_MU_VL', 'MCADAMS_MU_VL','CICCHITTI_MU_VL',
                'LUN_KWOK_MU_VL', 'FOURAR_BORIES_MU_VL', 'DUCKLER_MU_VL',
                'MINIMUM_PHASE_PROP', 'MAXIMUM_PHASE_PROP', 'FROM_DERIVATIVE_SETTINGS',
                ])

mole_methods = set([MOLE_WEIGHTED, LOG_PROP_MOLE_WEIGHTED, POWER_PROP_MOLE_WEIGHTED])
mass_methods = set([MASS_WEIGHTED, LOG_PROP_MASS_WEIGHTED, POWER_PROP_MASS_WEIGHTED])
volume_methods = set([VOLUME_WEIGHTED, LOG_PROP_VOLUME_WEIGHTED, POWER_PROP_VOLUME_WEIGHTED])

linear_methods = set([MOLE_WEIGHTED, MASS_WEIGHTED, VOLUME_WEIGHTED])
log_prop_methods = set([LOG_PROP_MOLE_WEIGHTED, LOG_PROP_MASS_WEIGHTED, LOG_PROP_VOLUME_WEIGHTED])
prop_power_methods = set([POWER_PROP_MOLE_WEIGHTED, POWER_PROP_MASS_WEIGHTED, POWER_PROP_VOLUME_WEIGHTED])

class BulkSettings(object):
    r'''Class containing configuration methods for determining how properties of
    a `Bulk` phase made of different phases are handled. All parameters are also
    attributes.


    Parameters
    ----------
    dP_dT : str, optional
        The method used to calculate the constant-volume temperature derivative
        of pressure of the bulk. One of :obj:`DP_DT_METHODS`, [-]
    dP_dV : str, optional
        The method used to calculate the constant-temperature volume derivative
        of pressure of the bulk. One of :obj:`DP_DV_METHODS`, [-]
    d2P_dV2 : str, optional
        The method used to calculate the second constant-temperature volume derivative
        of pressure of the bulk. One of :obj:`D2P_DV2_METHODS`, [-]
    d2P_dT2 : str, optional
        The method used to calculate the second constant-volume temperature derivative
        of pressure of the bulk. One of :obj:`D2P_DT2_METHODS`, [-]
    d2P_dTdV : str, optional
        The method used to calculate the temperature and volume derivative
        of pressure of the bulk. One of :obj:`D2P_DTDV_METHODS`, [-]
    T_liquid_volume_ref : float, optional
        Liquid molar volume reference temperature; if this is 298.15 K exactly,
        the molar volumes in
        :obj:`Vml_STPs <ChemicalConstantsPackage.Vml_STPs>` will be used, and
        if it is 288.7055555555555 K exactly,
        :obj:`Vml_60Fs <ChemicalConstantsPackage.Vml_60Fs>` will be used, and
        otherwise the molar liquid volumes will be obtained from the
        temperature-dependent correlations specified, [K]
    T_gas_ref : float, optional
        Reference temperature to use for the calculation of ideal-gas
        molar volume and flow rate, [K]
    P_gas_ref : float, optional
        Reference pressure to use for the calculation of ideal-gas
        molar volume and flow rate, [Pa]
    T_normal : float, optional
        "Normal" gas reference temperature for the calculation of ideal-gas
        molar volume in the "normal" reference state; default 273.15 K (0 C)
        according to [1]_, [K]
    P_normal : float, optional
        "Normal" gas reference pressure for the calculation of ideal-gas
        molar volume in the "normal" reference state; default 101325 Pa
        (1 atm) according to [1]_, [Pa]
    T_standard : float, optional
        "Standard" gas reference temperature for the calculation of ideal-gas
        molar volume in the "standard" reference state; default 288.15 K (15° C)
        according to [2]_; 288.7055555555555 is also often used (60° F), [K]
    P_standard : float, optional
        "Standard" gas reference pressure for the calculation of ideal-gas
        molar volume in the "standard" reference state; default 101325 Pa
        (1 atm) according to [2]_, [Pa]
    mu_LL : str, optional
        Mixing rule for multiple liquid phase liquid viscosity calculations;
        see :obj:`MU_LL_METHODS` for available options,
        [-]
    mu_LL_power_exponent : float, optional
        Liquid-liquid viscosity power-law mixing parameter, used only when a
        power law mixing rule is selected, [-]
    mu_VL : str, optional
        Mixing rule for vapor-liquid viscosity calculations;
        see :obj:`MU_VL_METHODS` for available options,
        [-]
    mu_VL_power_exponent : float, optional
        Vapor-liquid viscosity power-law mixing parameter, used only when a
        power law mixing rule is selected, [-]
    k_LL : str, optional
        Mixing rule for multiple liquid phase liquid thermal conductivity calculations;
        see :obj:`K_LL_METHODS` for available options,
        [-]
    k_LL_power_exponent : float, optional
        Liquid-liquid thermal conductivity power-law mixing parameter,
        used only when a power law mixing rule is selected, [-]
    k_VL : str, optional
        Mixing rule for vapor-liquid thermal conductivity calculations;
        see :obj:`K_VL_METHODS` for available options,
        [-]
    k_VL_power_exponent : float, optional
        Vapor-liquid thermal conductivity power-law mixing parameter,
        used only when a power law mixing rule is selected, [-]
    sigma_LL : str, optional
        Mixing rule for multiple liquid phase, air-liquid surface tension
        calculations; see :obj:`SIGMA_LL_METHODS` for available options,
        [-]
    sigma_LL_power_exponent : float, optional
        Air-liquid Liquid-liquid surface tension power-law mixing parameter,
        used only when a power law mixing rule is selected, [-]
    equilibrium_perturbation : float, optional
        The relative perturbation to use when calculating equilibrium
        derivatives numerically; for example if this is 1e-3 and `T` is the
        perturbation variable and the statis is 500 K, the perturbation
        calculation temperature will be 500.5 K, [various]
    isobaric_expansion : str, optional
        Mixing rule for multiphase isobaric expansion calculations;
        see :obj:`BETA_METHODS` for available options, [-]
    speed_of_sound : str, optional
        Mixing rule for multiphase speed of sound calculations;
        see :obj:`SPEED_OF_SOUND_METHODS` for available options, [-]
    kappa : str, optional
        Mixing rule for multiphase `kappa` calculations;
        see :obj:`KAPPA_METHODS` for available options, [-]
    Joule_Thomson : str, optional
        Mixing rule for multiphase `Joule-Thomson` calculations;
        see :obj:`JT_METHODS` for available options, [-]


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

    The mixing rule "FROM_DERIVATIVE_SETTINGS" is used to indicate that the
    property depends on other configurable properties; and when this is the
    specified option, those configurations will be used in the calculation
    of this property.

    The mixing rule "EQUILIBRIUM_DERIVATIVE" performs derivative calculations
    on flashes themselves. This is quite slow in comparison to other methods.

    References
    ----------
    .. [1] 14:00-17:00. "ISO 10780:1994." ISO. Accessed March 29, 2021.
       https://www.iso.org/cms/render/live/en/sites/isoorg/contents/data/standard/01/88/18855.html.
    .. [2] 14:00-17:00. "ISO 13443:1996." ISO. Accessed March 29, 2021.
       https://www.iso.org/cms/render/live/en/sites/isoorg/contents/data/standard/02/04/20461.html.
    '''
    
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    
    def as_json(self):
        return self.__dict__.copy()

    def __init__(self,
                 dP_dT=MOLE_WEIGHTED, dP_dV=MOLE_WEIGHTED,
                 d2P_dV2=MOLE_WEIGHTED, d2P_dT2=MOLE_WEIGHTED,
                 d2P_dTdV=MOLE_WEIGHTED,

                 mu_LL=LOG_PROP_MASS_WEIGHTED, mu_LL_power_exponent=0.4,
                 mu_VL=MCADAMS_MU_VL, mu_VL_power_exponent=0.4,

                 k_LL=MASS_WEIGHTED, k_LL_power_exponent=0.4,
                 k_VL=MASS_WEIGHTED, k_VL_power_exponent=0.4,

                 sigma_LL=MASS_WEIGHTED, sigma_LL_power_exponent=0.4,

                 T_liquid_volume_ref=298.15,
                 T_normal=273.15, P_normal=atm,
                 T_standard=288.15, P_standard=atm,
                 T_gas_ref=288.15, P_gas_ref=atm,

                 speed_of_sound=MOLE_WEIGHTED, kappa=MOLE_WEIGHTED,
                 isobaric_expansion=MOLE_WEIGHTED, Joule_Thomson=MOLE_WEIGHTED,

                # Undocumented
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

                 equilibrium_perturbation=1e-7,

                 ):
        self.dP_dT = dP_dT
        self.dP_dV = dP_dV
        self.d2P_dV2 = d2P_dV2
        self.d2P_dT2 = d2P_dT2
        self.d2P_dTdV = d2P_dTdV
        if mu_LL not in MU_LL_METHODS_set:
            raise ValueError("Unrecognized option for mu_LL")
        self.mu_LL = mu_LL
        self.mu_LL_power_exponent = mu_LL_power_exponent

        if mu_VL not in MU_VL_METHODS_SET:
            raise ValueError("Unrecognized option for mu_VL")
        self.mu_VL = mu_VL
        self.mu_VL_power_exponent = mu_VL_power_exponent

        if k_LL not in K_LL_METHODS_set:
            raise ValueError("Unrecognized option for k_LL")
        self.k_LL = k_LL
        self.k_LL_power_exponent = k_LL_power_exponent

        if k_VL not in K_VL_METHODS_SET:
            raise ValueError("Unrecognized option for k_VL")
        self.k_VL = k_VL
        self.k_VL_power_exponent = k_VL_power_exponent

        if sigma_LL not in SIGMA_LL_METHODS_set:
            raise ValueError("Unrecognized option for sigma_LL")
        self.sigma_LL = sigma_LL
        self.sigma_LL_power_exponent = sigma_LL_power_exponent

        self.T_normal = T_normal
        self.P_normal = P_normal
        self.T_standard = T_standard
        self.P_standard = P_standard
        self.T_liquid_volume_ref = T_liquid_volume_ref
        self.T_gas_ref = T_gas_ref
        self.P_gas_ref = P_gas_ref

        self.equilibrium_perturbation = equilibrium_perturbation

        self.isobaric_expansion = isobaric_expansion
        self.speed_of_sound = speed_of_sound
        self.kappa = kappa
        self.Joule_Thomson = Joule_Thomson

        # Phase identification settings
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
    r'''Class to encapsulate multiple :obj:`Phase <thermo.phases.Phase>` objects and provide a
    unified interface for obtaining properties from a group of phases.

    This class exists for three purposes:

    * Providing a common interface for obtaining properties like `Cp` - whether
      there is one phase or 100, calling `Cp` on the bulk will retrieve that
      value.
    * Retrieving "bulk" properties that do make sense to be calculated for a
      combination of phases together.
    * Allowing configurable estimations of non-bulk properties like isothermal
      compressibility or speed of sound for the group of phases together.

    Parameters
    ----------
    T : float
        Temperature of the bulk, [K]
    P : float
        Pressure of the bulk, [Pa]
    zs : list[float]
        Mole fractions of the bulk, [-]
    phases : list[:obj:`Phase <thermo.phases.Phase>`]
        Phase objects, [-]
    phase_fractions : list[float]
        Molar fractions of each phase, [-]
    phase_bulk : str, optional
        None to represent a bulk of all present phases; 'l' to represent a bulk
        of only liquid phases; `s` to represent a bulk of only solid phases, [-]

    Notes
    -----
    Please think carefully when retrieving a property of the bulk. If there are
    two liquid phases in a bulk, and a single viscosity value is retrieved,
    can that be used directly for a single phase pressure drop calculation?
    Not with any theoretical consistency, that's for sure.



    '''
    def __init__(self, T, P, zs, phases, phase_fractions, phase_bulk=None):
        self.T = T
        self.P = P
        self.zs = zs
        self.phases = phases
        self.phase_fractions = phase_fractions
        self.N = N = len(zs)
        self.phase_bulk = phase_bulk



    @property
    def beta(self):
        r'''Phase fraction of the bulk phase. Should always be 1 when
        representing all phases of a flash; but can be less than one if
        representing multiple solids or liquids as a single phase in a larger
        mixture.

        Returns
        -------
        beta : float
            Phase fraction of bulk, [-]
        '''
        return sum(self.phase_fractions)

    @property
    def betas_mass(self):
        r'''Method to calculate and return the mass fraction of all of the
        phases in the bulk.

        Returns
        -------
        betas_mass : list[float]
            Mass phase fractions of all the phases in the bulk object, ordered
            vapor, liquid, then solid, [-]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the volume fraction of all of the
        phases in the bulk.

        Returns
        -------
        betas_volume : list[float]
            Volume phase fractions of all the phases in the bulk, ordered
            vapor, liquid, then solid , [-]

        Notes
        -----
        '''
        betas = self.phase_fractions
        phase_iter = range(len(betas))
        Vs_phases = [i.V() for i in self.phases]

        tot = 0.0
        for i in phase_iter:
            tot += Vs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        return [betas[i]*Vs_phases[i]*tot_inv for i in phase_iter]

    def _property_mixing_rule(self, method, exponent, mix_obj, attr):
        if method == AS_ONE_LIQUID:
            prop = mix_obj.mixture_property(self.T, self.P, self.zs, self.ws())
        else:
            props = [getattr(i, attr)() for i in self.phases]
            if method in mole_methods:
                betas = self.phase_fractions
            elif method in mass_methods:
                betas = self.betas_mass
            elif method in volume_methods:
                betas = self.betas_volume

            prop = 0.0
            if method in linear_methods:
                for i in range(len(self.phase_fractions)):
                    prop += betas[i]*props[i]
            elif method in prop_power_methods:
                for i in range(len(self.phase_fractions)):
                    prop += betas[i]*props[i]**exponent
                prop = prop**(1.0/exponent)
            elif method in log_prop_methods:
                for i in range(len(self.phase_fractions)):
                    prop += betas[i]*log(props[i])
                prop = exp(prop)
            elif method == MINIMUM_PHASE_PROP:
                prop = min(props)
            elif method == MAXIMUM_PHASE_PROP:
                prop = max(props)
            else:
                raise ValueError("Unknown method")
        return prop

    def _mu_k_VL(self, method, props, exponent):
        if method in mole_methods:
            VF = self.result.VF
            betas = [VF, 1.0 - VF]
        elif method in mass_methods:
            betas = self.result.betas_mass_states[:2]
        elif method in volume_methods:
            betas = self.result.betas_volume_states[:2]

        if method in linear_methods:
            prop = betas[0]*props[0] + betas[1]*props[1]
        elif method in prop_power_methods:
            prop = (betas[0]*props[0]**exponent + betas[1]*props[1]**exponent)**(1.0/exponent)
        elif method in log_prop_methods:
            prop = exp(betas[0]*log(props[0]) + betas[1]*log(props[1]))
        elif method == MINIMUM_PHASE_PROP:
            prop = min(props)
        elif method == MAXIMUM_PHASE_PROP:
            prop = max(props)
        else:
            raise ValueError("Unrecognized method")

        return prop


    def mu(self):
        r'''Calculate and return the viscosity of the bulk according to the
        selected viscosity settings in :obj:`BulkSettings`, the settings in
        :obj:`ViscosityGasMixture <thermo.viscosity.ViscosityGasMixture>` and
        :obj:`ViscosityLiquidMixture <thermo.viscosity.ViscosityLiquidMixture>`,
        and the configured pure-component settings in
        :obj:`ViscosityGas <thermo.viscosity.ViscosityGas>` and
        :obj:`ViscosityLiquid <thermo.viscosity.ViscosityLiquid>`.

        Returns
        -------
        mu : float
            Viscosity of bulk phase calculated with mixing rules, [Pa*s]
        '''
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
        elif self.phase_bulk == 'l' or self.result.gas is None:
            # Multiple liquids - either a bulk liquid, or a result with no gases
            mu = self._property_mixing_rule(self.settings.mu_LL, self.settings.mu_LL_power_exponent,
                                            self.correlations.ViscosityLiquidMixture, 'mu')
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
            mu = self._mu_k_VL(method, props=[mug, mul],
                               exponent=self.settings.mu_VL_power_exponent)
        self._mu = mu
        return mu

    def k(self):
        r'''Calculate and return the thermal conductivity of the bulk according to the
        selected thermal conductivity settings in :obj:`BulkSettings`, the settings in
        :obj:`ThermalConductivityGasMixture <thermo.thermal_conductivity.ThermalConductivityGasMixture>` and
        :obj:`ThermalConductivityLiquidMixture <thermo.thermal_conductivity.ThermalConductivityLiquidMixture>`,
        and the configured pure-component settings in
        :obj:`ThermalConductivityGas <thermo.thermal_conductivity.ThermalConductivityGas>` and
        :obj:`ThermalConductivityLiquid <thermo.thermal_conductivity.ThermalConductivityLiquid>`.

        Returns
        -------
        k : float
            Thermal Conductivity of bulk phase calculated with mixing rules, [Pa*s]
        '''
        try:
            return self._k
        except AttributeError:
            pass
        phase_fractions = self.phase_fractions
        phase_count = len(phase_fractions)
        result = self.result
        if phase_count == 1:
            self._k = k = self.phases[0].k()
            return k
        elif self.phase_bulk == 'l' or self.result.gas is None:
            # Multiple liquids - either a bulk liquid, or a result with no gases
            k = self._property_mixing_rule(self.settings.k_LL, self.settings.k_LL_power_exponent,
                                           self.correlations.ThermalConductivityLiquidMixture, 'k')
            self._k = k
            return k

        method = self.settings.k_VL
        if method == AS_ONE_LIQUID:
            self._k = k = self.correlations.ThermalConductivityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
            return k
        elif method == AS_ONE_GAS:
            self._k = k = self.correlations.ThermalConductivityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
            return k

        kg = result.gas.k()
        if phase_count == 2:
            kl = result.liquids[0].k()
        else:
            kl = result.liquid_bulk.k()

        k = self._mu_k_VL(method, props=[kg, kl],
                            exponent=self.settings.k_VL_power_exponent)
        self._k = k
        return k

    def sigma(self):
        r'''Calculate and return the surface tension of the bulk according to the
        selected surface tension settings in :obj:`BulkSettings`, the settings in
        :obj:`SurfaceTensionMixture <thermo.interface.SurfaceTensionMixture>`
        and the configured pure-component settings in
        :obj:`SurfaceTension <thermo.interface.SurfaceTension>`.

        Returns
        -------
        sigma : float
            Surface tension of bulk phase calculated with mixing rules, [N/m]

        Notes
        -----
        A value is only returned if all phases in the bulk are liquids; this
        property is for a liquid-ideal gas calculation, not the interfacial
        tension between two liquid phases.
        '''
        try:
            return self._sigma
        except AttributeError:
            pass
        phase_fractions = self.phase_fractions
        phase_count = len(phase_fractions)
        result = self.result
        state = self.phase_bulk
        if phase_count == 1 and self.result.gas is None:
            self._sigma = sigma = self.phases[0].sigma()
            return sigma
        elif self.phase_bulk == 'l' or self.result.gas is None:
            # Multiple liquids - either a bulk liquid, or a result with no gases
            sigma = self._property_mixing_rule(self.settings.sigma_LL, self.settings.sigma_LL_power_exponent,
                                               self.correlations.SurfaceTensionMixture, 'sigma')
            self._sigma = sigma
            return sigma
        else:
            return None




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

    try:
        dH_dP.__doc__ = Phase.dH_dP_T.__doc__
    except:
        pass

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

    try:
        dS_dP.__doc__ = Phase.dS_dP_T.__doc__
    except:
        pass

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

    try:
        dS_dT.__doc__ = Phase.dS_dT.__doc__
    except:
        pass

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
        r'''Method to calculate and return the constant-temperature and
        constant phase-fraction reactive enthalpy of the bulk phase.
        This is a phase-fraction weighted calculation.

        .. math::
            H_{\text{reactive}} = \sum_i^p H_{\text{reactive}, i} \beta_i

        Returns
        -------
        H_reactive : float
            Reactive molar enthalpy, [J/(mol)]
        '''
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
        r'''Method to calculate and return the constant-temperature and
        constant phase-fraction reactive entropy of the bulk phase.
        This is a phase-fraction weighted calculation.

        .. math::
            S_{\text{reactive}} = \sum_i^p S_{\text{reactive}, i} \beta_i

        Returns
        -------
        S_reactive : float
            Reactive molar entropy, [J/(mol*K)]
        '''
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
        r'''Method to calculate and return the constant-volume derivative of
        pressure with respect to temperature of the bulk phase, at constant
        phase fractions and phase compositions.
        This is a molar phase-fraction weighted calculation.

        .. math::
            \left(\frac{\partial P}{\partial T}\right)_{V, \beta, {zs}} =
            \sum_{i}^{\text{phases}} \beta_i \left(\frac{\partial P}
            {\partial T}\right)_{i, V_i, \beta_i, {zs}_i}

        Returns
        -------
        dP_dT_frozen : float
            Frozen constant-volume derivative of pressure with respect to
            temperature of the bulk phase, [Pa/K]
        '''
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
        r'''Method to calculate and return the constant-temperature derivative of
        pressure with respect to volume of the bulk phase, at constant
        phase fractions and phase compositions.
        This is a molar phase-fraction weighted calculation.

        .. math::
            \left(\frac{\partial P}{\partial V}\right)_{T, \beta, {zs}} =
            \sum_{i}^{\text{phases}} \beta_i \left(\frac{\partial P}
            {\partial V}\right)_{i, T, \beta_i, {zs}_i}

        Returns
        -------
        dP_dV_frozen : float
            Frozen constant-temperature derivative of pressure with respect to
            volume of the bulk phase, [Pa*mol/m^3]
        '''
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
        r'''Method to calculate and return the second constant-volume derivative
        of pressure with respect to temperature of the bulk phase, at constant
        phase fractions and phase compositions.
        This is a molar phase-fraction weighted calculation.

        .. math::
            \left(\frac{\partial^2 P}{\partial T^2}\right)_{V, \beta, {zs}} =
            \sum_{i}^{\text{phases}} \beta_i \left(\frac{\partial^2 P}
            {\partial T^2}\right)_{i, V_i, \beta_i, {zs}_i}

        Returns
        -------
        d2P_dT2_frozen : float
            Frozen constant-volume second derivative of pressure with respect to
            temperature of the bulk phase, [Pa/K^2]
        '''
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
        r'''Method to calculate and return the constant-temperature second
        derivative of pressure with respect to volume of the bulk phase, at
        constant phase fractions and phase compositions.
        This is a molar phase-fraction weighted calculation.

        .. math::
            \left(\frac{\partial^2 P}{\partial V^2}\right)_{T, \beta, {zs}} =
            \sum_{i}^{\text{phases}} \beta_i \left(\frac{\partial^2 P}
            {\partial V^2}\right)_{i, T, \beta_i, {zs}_i}

        Returns
        -------
        d2P_dV2_frozen : float
            Frozen constant-temperature second derivative of pressure with
            respect to volume of the bulk phase, [Pa*mol^2/m^6]
        '''
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
        r'''Method to calculate and return the second
        derivative of pressure with respect to volume and temperature of the
        bulk phase, at constant phase fractions and phase compositions.
        This is a molar phase-fraction weighted calculation.

        .. math::
            \left(\frac{\partial^2 P}{\partial V \partial T}\right)_{\beta, {zs}} =
            \sum_{i}^{\text{phases}} \beta_i \left(\frac{\partial^2 P}
            {\partial V \partial T}\right)_{i, \beta_i, {zs}_i}

        Returns
        -------
        d2P_dTdV_frozen : float
            Frozen second derivative of pressure with
            respect to volume and temperature of the bulk phase, [Pa*mol^2/m^6]
        '''
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

    def _equilibrium_derivative(self, of='P', wrt='T', const='V'):
        '''Calculate the equilibrium derivative of a property by performing
        a numerical derivative on flash calculations.
        '''
        const_value = self.value(const)
        wrt_value = self.value(wrt)
        of_value = self.value(of)

        pert = self.settings.equilibrium_perturbation
        wrt_value2 = wrt_value*(1.0 + pert)
        delta = wrt_value2 - wrt_value
        kwargs = {wrt: wrt_value2, const: const_value}
        results = self.flasher.flash(zs=self.zs, **kwargs)

        of_value2 = results.value(of)
        value = (of_value2 - of_value)/delta
        return value



    def dP_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        pressure of the bulk according to the selected calculation methodology.

        Returns
        -------
        dP_dT : float
            First temperature derivative of pressure, [Pa/K]
        '''
        dP_dT_method = self.settings.dP_dT
        if dP_dT_method == MOLE_WEIGHTED:
            return self.dP_dT_frozen()
        elif dP_dT_method == EQUILIBRIUM_DERIVATIVE:
            return self._equilibrium_derivative(of='P', wrt='T', const='V')
        return self._property_mixing_rule(dP_dT_method, None, None, 'dP_dT')

    def dP_dV(self):
        r'''Method to calculate and return the first volume derivative of
        pressure of the bulk according to the selected calculation methodology.

        Returns
        -------
        dP_dV : float
            First volume derivative of pressure, [Pa*mol/m^3]
        '''
        dP_dV_method = self.settings.dP_dV
        if dP_dV_method == MOLE_WEIGHTED:
            return self.dP_dV_frozen()
        elif dP_dV_method == EQUILIBRIUM_DERIVATIVE:
            return self._equilibrium_derivative(of='P', wrt='V', const='T')
        return self._property_mixing_rule(dP_dV_method, None, None, 'dP_dV')

    def d2P_dT2(self):
        r'''Method to calculate and return the second temperature derivative of
        pressure of the bulk according to the selected calculation methodology.

        Returns
        -------
        d2P_dT2 : float
            Second temperature derivative of pressure, [Pa/K^2]
        '''
        d2P_dT2_method = self.settings.d2P_dT2
        if d2P_dT2_method == MOLE_WEIGHTED:
            return self.d2P_dT2_frozen()
        return self._property_mixing_rule(d2P_dT2_method, None, None, 'd2P_dT2')

    def d2P_dV2(self):
        r'''Method to calculate and return the second volume derivative of
        pressure of the bulk according to the selected calculation methodology.

        Returns
        -------
        d2P_dV2 : float
            Second volume derivative of pressure, [Pa*mol^2/m^6]
        '''
        d2P_dV2_method = self.settings.d2P_dV2
        if d2P_dV2_method == MOLE_WEIGHTED:
            return self.d2P_dV2_frozen()
        return self._property_mixing_rule(d2P_dV2_method, None, None, 'd2P_dV2')

    def d2P_dTdV(self):
        r'''Method to calculate and return the second derivative of
        pressure with respect to temperature and volume of the bulk according
        to the selected calculation methodology.

        Returns
        -------
        d2P_dTdV : float
            Second volume derivative of pressure, [mol*Pa^2/(J*K)]
        '''
        d2P_dTdV_method = self.settings.d2P_dTdV
        if d2P_dTdV_method == MOLE_WEIGHTED:
            return self.d2P_dTdV_frozen()
        return self._property_mixing_rule(d2P_dTdV_method, None, None, 'd2P_dTdV')


    def isobaric_expansion(self):
        r'''Method to calculate and return the isobatic expansion coefficient
        of the bulk according to the selected calculation methodology.

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Returns
        -------
        beta : float
            Isobaric coefficient of a thermal expansion, [1/K]
        '''
        beta_method = self.settings.isobaric_expansion
        if beta_method == EQUILIBRIUM_DERIVATIVE:
            if self.phase_bulk is not None:
                # Cannot perform an equilibrium derivative for a sub-bulk
                # equilibrium conditions are not satisfied
                return None
            return self._equilibrium_derivative(of='V', wrt='T', const='P')/self.V()
        elif beta_method == FROM_DERIVATIVE_SETTINGS:
            return isobaric_expansion(self.V(), self.dV_dT())
        return self._property_mixing_rule(beta_method, None, None, 'isobaric_expansion')

    def kappa(self):
        r'''Method to calculate and return the isothermal compressibility
        of the bulk according to the selected calculation methodology.

        .. math::
            \kappa = -\frac{1}{V}\left(\frac{\partial V}{\partial P} \right)_T

        Returns
        -------
        kappa : float
            Isothermal coefficient of compressibility, [1/Pa]
        '''
        kappa_method = self.settings.kappa
        if kappa_method == EQUILIBRIUM_DERIVATIVE:
            if self.phase_bulk is not None:
                # Cannot perform an equilibrium derivative for a sub-bulk
                # equilibrium conditions are not satisfied
                return None
            return -self._equilibrium_derivative(of='V', wrt='P', const='T')/self.V()
        elif kappa_method == FROM_DERIVATIVE_SETTINGS:
            return isothermal_compressibility(self.V(), self.dV_dP())
        return self._property_mixing_rule(kappa_method, None, None, 'kappa')

    def Joule_Thomson(self):
        r'''Method to calculate and return the Joule-Thomson coefficient
        of the bulk according to the selected calculation methodology.

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H

        Returns
        -------
        mu_JT : float
            Joule-Thomson coefficient [K/Pa]
        '''
        Joule_Thomson_method = self.settings.Joule_Thomson
        if Joule_Thomson_method == EQUILIBRIUM_DERIVATIVE:
            if self.phase_bulk is not None:
                # Cannot perform an equilibrium derivative for a sub-bulk
                # equilibrium conditions are not satisfied
                return None
            return self._equilibrium_derivative(of='T', wrt='P', const='H')
        elif Joule_Thomson_method == FROM_DERIVATIVE_SETTINGS:
            return Joule_Thomson(self.T, self.V(), self.Cp(), self.dV_dT())
        return self._property_mixing_rule(Joule_Thomson_method, None, None, 'Joule_Thomson')

    def speed_of_sound(self):
        r'''Method to calculate and return the molar speed of sound
        of the bulk according to the selected calculation methodology.

        .. math::
            w = \left[-V^2 \left(\frac{\partial P}{\partial V}\right)_T \frac{C_p}
            {C_v}\right]^{1/2}

        A similar expression based on molar density is:

        .. math::
           w = \left[\left(\frac{\partial P}{\partial \rho}\right)_T \frac{C_p}
           {C_v}\right]^{1/2}

        Returns
        -------
        w : float
            Speed of sound for a real gas, [m*kg^0.5/(s*mol^0.5)]
        '''
        speed_of_sound_method = self.settings.speed_of_sound
        if speed_of_sound_method == FROM_DERIVATIVE_SETTINGS:
            return speed_of_sound(self.V(), self.dP_dV(), self.Cp(), self.Cv())
        return self._property_mixing_rule(speed_of_sound_method, None, None, 'speed_of_sound')

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

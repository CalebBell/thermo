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

'''
__all__ = ['GibbsExcessLiquid', 'GibbsExcessSolid']

from math import isinf, isnan
from fluids.constants import R, R_inv
from fluids.numerics import (horner_and_der2, derivative,
                             evaluate_linear_fits, evaluate_linear_fits_d,
                             evaluate_linear_fits_d2,
                             trunc_exp, secant)
from chemicals.utils import log, exp
from thermo.activity import IdealSolution
from thermo.utils import POLY_FIT
from thermo.heat_capacity import HeatCapacityGas, HeatCapacityLiquid
from thermo.volume import VolumeLiquid, VolumeSolid
from thermo.vapor_pressure import VaporPressure, SublimationPressure
from thermo.phase_change import EnthalpyVaporization, EnthalpySublimation

from thermo.phases.phase import Phase

class GibbsExcessLiquid(Phase):
    r'''Phase based on combining Raoult's law with a
    :obj:`GibbsExcess <thermo.activity.GibbsExcess>` model, optionally
    including saturation fugacity coefficient corrections (if the vapor phase
    is a cubic equation of state) and Poynting correction factors (if more
    accuracy is desired).

    The equilibrium equation options (controlled by `equilibrium_basis`)
    are as follows:

    * 'Psat': :math:`\phi_i = \frac{\gamma_i P_{i}^{sat}}{P}`
    * 'Poynting&PhiSat': :math:`\phi_i = \frac{\gamma_i P_{i}^{sat} \phi_i^{sat} \text{Poynting}_i}{P}`
    * 'Poynting': :math:`\phi_i = \frac{\gamma_i P_{i}^{sat}\text{Poynting}_i}{P}`
    * 'PhiSat': :math:`\phi_i = \frac{\gamma_i P_{i}^{sat} \phi_i^{sat}}{P}`

    In all cases, the activity coefficient is derived from the
    :obj:`GibbsExcess <thermo.activity.GibbsExcess>` model specified as
    input; use the :obj:`IdealSolution <thermo.activity.IdealSolution>`
    class as an input to set the activity coefficients to one.

    The enthalpy `H` and entropy `S` (and other caloric properties `U`, `G`, `A`)
    equation options are similar to the equilibrium ones. If the same option
    is selected for `equilibrium_basis` and `caloric_basis`, the phase will be
    `thermodynamically consistent`. This is recommended for many reasons.
    The full 'Poynting&PhiSat' equations for `H` and `S` are as follows; see
    :obj:`GibbsExcessLiquid.H` and :obj:`GibbsExcessLiquid.S` for all of the
    other equations:

    .. math::
        H = H_{\text{excess}} + \sum_i z_i \left[-RT^2\left(
        \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
        + \frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
        + \frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}} \right)
        + \int_{T,ref}^T C_{p,ig} dT \right]

    .. math::
        S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
        - \sum_i z_i\left[R\left(
        T \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
        + T\frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
        + T\frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}}
        + \ln(P_{\text{sat},i}) + \ln\left(\frac{\text{Poynting}\cdot\phi_{\text{sat},i}}{P}\right)
        \right) - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

    An additional caloric mode is `Hvap`, which uses enthalpy of vaporization;
    this mode can never be thermodynamically consistent, but is still widely
    used.

    .. math::
        H = H_{\text{excess}} + \sum_i z_i\left[-H_{vap,i}
        + \int_{T,ref}^T C_{p,ig} dT \right]

    .. math::
        S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
        - \sum_i z_i\left[R\left(\ln P_{\text{sat},i} + \ln\left(\frac{1}{P}\right)\right)
        + \frac{H_{vap,i}}{T}
        - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]


    .. warning::
        Note that above the critical point, there is no definition for what vapor
        pressure is. The vapor pressure also tends to reach zero at temperatures
        in the 4-20 K range. These aspects mean extrapolation in the supercritical and
        very low temperature region is critical to ensure the equations will still
        converge. Extrapolation can be performed using either the equation
        :math:`P^{\text{sat}} = \exp\left(A - \frac{B}{T}\right)` or
        :math:`P^{\text{sat}} = \exp\left(A + \frac{B}{T} + C\cdot \ln T\right)` by
        setting `Psat_extrpolation` to either 'AB' or 'ABC' respectively.
        The extremely low temperature region's issue is solved by calculating the
        logarithm of vapor pressures instead of the actual value. While floating
        point values in Python (doubles) can reach a minimum value of around
        1e-308, if only the logarithm of that number is computed no issues arise.
        Both of these features only work when the vapor pressure correlations are
        polynomials.

    .. warning::
        When using 'PhiSat' as an option, note that the factor cannot be
        calculated when a compound is supercritical,
        as there is no longer any vapor-liquid pure-component equilibrium
        (by definition).

    Parameters
    ----------
    VaporPressures : list[:obj:`thermo.vapor_pressure.VaporPressure`]
        Objects holding vapor pressure data and methods, [-]
    VolumeLiquids : list[:obj:`thermo.volume.VolumeLiquid`], optional
        Objects holding liquid volume data and methods; required for Poynting
        factors and volumetric properties, [-]
    HeatCapacityGases : list[:obj:`thermo.heat_capacity.HeatCapacityGas`], optional
        Objects proiding pure-component heat capacity correlations; required
        for caloric properties, [-]
    GibbsExcessModel : :obj:`GibbsExcess <thermo.activity.GibbsExcess>`, optional
        Configured instance for calculating activity coefficients and excess properties;
        set to :obj:`IdealSolution <thermo.activity.IdealSolution>` if not provided, [-]
    eos_pure_instances : list[:obj:`thermo.eos.GCEOS`], optional
        Cubic equation of state object instances for each pure component, [-]
    EnthalpyVaporizations : list[:obj:`thermo.phase_change.EnthalpyVaporization`], optional
        Objects holding enthalpy of vaporization data and methods; used only
        with the 'Hvap' optional, [-]
    HeatCapacityLiquids : list[:obj:`thermo.heat_capacity.HeatCapacityLiquid`], optional
        Objects holding liquid heat capacity data and methods; not used at
        present, [-]
    VolumeSupercriticalLiquids : list[:obj:`thermo.volume.VolumeLiquid`], optional
        Objects holding liquid volume data and methods but that are used for
        supercritical temperatures on a per-component basis only; required for
        Poynting factors and volumetric properties at supercritical conditions;
        `VolumeLiquids` is used if not provided, [-]
    Hfs : list[float], optional
        Molar ideal-gas standard heats of formation at 298.15 K and 1 atm,
        [J/mol]
    Gfs : list[float], optional
        Molar ideal-gas standard Gibbs energies of formation at 298.15 K and
        1 atm, [J/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    zs : list[float], optional
        Mole fractions of each component, [-]
    equilibrium_basis : str, optional
        Which set of equilibrium equations to use when calculating fugacities
        and related properties; valid options are 'Psat', 'Poynting&PhiSat',
        'Poynting', 'PhiSat', [-]
    caloric_basis : str, optional
        Which set of caloric equations to use when calculating fugacities
        and related properties; valid options are 'Psat', 'Poynting&PhiSat',
        'Poynting', 'PhiSat', 'Hvap' [-]
    Psat_extrpolation : str, optional
        One of 'AB' or 'ABC'; configures extrapolation for vapor pressure, [-]


    use_Hvap_caloric : bool, optional
        If True, enthalpy and entropy will be calculated using ideal-gas
        heat capacity and the heat of vaporization of the fluid only. This
        forces enthalpy to be pressure-independent. This supersedes other
        options which would otherwise impact these properties. The molar volume
        of the fluid has no impact on enthalpy or entropy if this option is
        True. This option is not thermodynamically consistent, but is still
        often an assumption that is made.

    '''
    force_phase = 'l'
    phase = 'l'
    is_gas = False
    is_liquid = True
    P_DEPENDENT_H_LIQ = True
    PHI_SAT_IDEAL_TR = 0.1
    _Psats_data = None
    Psats_poly_fit = False
    Vms_sat_poly_fit = False
    _Vms_sat_data = None
    Hvap_poly_fit = False
    _Hvap_data = None

    use_IG_Cp = True # Deprecated! Remove with S_old and H_old

    ideal_gas_basis = True
    supercritical_volumes = False

    Cpls_poly_fit = False
    _Cpls_data = None

    _Tait_B_data = None
    _Tait_C_data = None

    pure_references = ('HeatCapacityGases', 'VolumeLiquids', 'VaporPressures', 'HeatCapacityLiquids',
                       'EnthalpyVaporizations')
    pure_reference_types = (HeatCapacityGas, VolumeLiquid, VaporPressure, HeatCapacityLiquid,
                            EnthalpyVaporization)

    model_attributes = ('Hfs', 'Gfs', 'Sfs', 'GibbsExcessModel',
                        'eos_pure_instances', 'use_Poynting', 'use_phis_sat',
                        'use_Tait', 'use_eos_volume', 'henry_components',
                        'henry_data', 'Psat_extrpolation') + pure_references

    obj_references = ('GibbsExcessModel', 'eos_pure_instances')
    
    def __init__(self, VaporPressures, VolumeLiquids=None,
                 HeatCapacityGases=None,
                 GibbsExcessModel=None,
                 eos_pure_instances=None,
                 EnthalpyVaporizations=None,
                 HeatCapacityLiquids=None,
                 VolumeSupercriticalLiquids=None,

                 use_Hvap_caloric=False,
                 use_Poynting=False,
                 use_phis_sat=False,
                 use_Tait=False,
                 use_eos_volume=False,

                 Hfs=None, Gfs=None, Sfs=None,

                 henry_components=None, henry_data=None,

                 T=None, P=None, zs=None,
                 Psat_extrpolation='AB',
                 equilibrium_basis=None,
                 caloric_basis=None,
                 ):
        '''It is quite possible to introduce a PVT relation ship for liquid
        density and remain thermodynamically consistent. However, must be
        applied on a per-component basis! This class cannot have an
        equation-of-state or VolumeLiquidMixture for a liquid MIXTURE!

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
        self.Psats_poly_fit = all(i.method == POLY_FIT for i in VaporPressures) if VaporPressures is not None else False
        self.Psat_extrpolation = Psat_extrpolation
        if self.Psats_poly_fit:
            Psats_data = [[i.poly_fit_Tmin for i in VaporPressures],
                               [i.poly_fit_Tmin_slope for i in VaporPressures],
                               [i.poly_fit_Tmin_value for i in VaporPressures],
                               [i.poly_fit_Tmax for i in VaporPressures],
                               [i.poly_fit_Tmax_slope for i in VaporPressures],
                               [i.poly_fit_Tmax_value for i in VaporPressures],
                               [i.poly_fit_coeffs for i in VaporPressures],
                               [i.poly_fit_d_coeffs for i in VaporPressures],
                               [i.poly_fit_d2_coeffs for i in VaporPressures],
                               [i.DIPPR101_ABC for i in VaporPressures]]
            if Psat_extrpolation == 'AB':
                Psats_data.append([i.poly_fit_AB_high_ABC_compat + [0.0] for i in VaporPressures])
            elif Psat_extrpolation == 'ABC':
                Psats_data.append([i.DIPPR101_ABC_high for i in VaporPressures])
            # Other option: raise?
            self._Psats_data = Psats_data

        self.N = len(VaporPressures)

        self.HeatCapacityGases = HeatCapacityGases
        self.Cpgs_poly_fit, self._Cpgs_data = self._setup_Cpigs(HeatCapacityGases)

        self.HeatCapacityLiquids = HeatCapacityLiquids
        if HeatCapacityLiquids is not None:
            self.Cpls_poly_fit, self._Cpls_data = self._setup_Cpigs(HeatCapacityLiquids)
            T_REF_IG = self.T_REF_IG
            T_REF_IG_INV = 1.0/T_REF_IG
            self.Hvaps_T_ref = [obj(T_REF_IG) for obj in EnthalpyVaporizations]
            self.dSvaps_T_ref = [T_REF_IG_INV*dH for dH in self.Hvaps_T_ref]

        self.use_eos_volume = use_eos_volume
        self.VolumeLiquids = VolumeLiquids
        self.Vms_sat_poly_fit = ((not use_eos_volume and all(i.method == POLY_FIT for i in VolumeLiquids)) if VolumeLiquids is not None else False)
        if self.Vms_sat_poly_fit:
            self._Vms_sat_data = [[i.poly_fit_Tmin for i in VolumeLiquids],
                                 [i.poly_fit_Tmin_slope for i in VolumeLiquids],
                                 [i.poly_fit_Tmin_value for i in VolumeLiquids],
                                 [i.poly_fit_Tmax for i in VolumeLiquids],
                                 [i.poly_fit_Tmax_slope for i in VolumeLiquids],
                                 [i.poly_fit_Tmax_value for i in VolumeLiquids],
                                 [i.poly_fit_coeffs for i in VolumeLiquids],
                                 [i.poly_fit_d_coeffs for i in VolumeLiquids],
                                 [i.poly_fit_d2_coeffs for i in VolumeLiquids],
                                 [i.poly_fit_Tmin_quadratic for i in VolumeLiquids],
                                 ]
#            low_fits = self._Vms_sat_data[9]
#            for i in range(self.N):
#                low_fits[i][0] = max(0, low_fits[i][0])

        self.VolumeSupercriticalLiquids = VolumeSupercriticalLiquids
        self.Vms_supercritical_poly_fit = all(i.method == POLY_FIT for i in VolumeSupercriticalLiquids) if VolumeSupercriticalLiquids is not None else False
        if self.Vms_supercritical_poly_fit:
            self.Vms_supercritical_data = [[i.poly_fit_Tmin for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmin_slope for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmin_value for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmax for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmax_slope for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmax_value for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_coeffs for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_d_coeffs for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_d2_coeffs for i in VolumeSupercriticalLiquids],
                                 [i.poly_fit_Tmin_quadratic for i in VolumeSupercriticalLiquids],
                                 ]


        self.incompressible = not use_Tait
        self.use_Tait = use_Tait
        if self.use_Tait:
            Tait_B_data, Tait_C_data = [[] for i in range(9)], [[] for i in range(9)]
            for v in VolumeLiquids:
                for (d, store) in zip(v.Tait_data(), [Tait_B_data, Tait_C_data]):
                    for i in range(len(d)):
                        store[i].append(d[i])
            self._Tait_B_data = Tait_B_data
            self._Tait_C_data = Tait_C_data


        self.EnthalpyVaporizations = EnthalpyVaporizations
        self.Hvap_poly_fit = all(i.method == POLY_FIT for i in EnthalpyVaporizations) if EnthalpyVaporizations is not None else False
        if self.Hvap_poly_fit:
            self._Hvap_data = [[i.poly_fit_Tmin for i in EnthalpyVaporizations],
                              [i.poly_fit_Tmax for i in EnthalpyVaporizations],
                              [i.poly_fit_Tc for i in EnthalpyVaporizations],
                              [1.0/i.poly_fit_Tc for i in EnthalpyVaporizations],
                              [i.poly_fit_coeffs for i in EnthalpyVaporizations]]



        if GibbsExcessModel is None:
            GibbsExcessModel = IdealSolution(T=T, xs=zs)

        self.GibbsExcessModel = GibbsExcessModel
        self.eos_pure_instances = eos_pure_instances

        self.equilibrium_basis = equilibrium_basis
        self.caloric_basis = caloric_basis

        if equilibrium_basis is not None:
            if equilibrium_basis == 'Poynting':
                self.use_Poynting = True
                self.use_phis_sat = False
            elif equilibrium_basis == 'Poynting&PhiSat':
                self.use_Poynting = True
                self.use_phis_sat = True
            elif equilibrium_basis == 'PhiSat':
                self.use_phis_sat = True
                self.use_Poynting = False
            elif equilibrium_basis == 'Psat':
                self.use_phis_sat = False
                self.use_Poynting = False
        else:
            self.use_Poynting = use_Poynting
            self.use_phis_sat = use_phis_sat

        if caloric_basis is not None:
            if caloric_basis == 'Poynting':
                self.use_Poynting_caloric = True
                self.use_phis_sat_caloric = False
                self.use_Hvap_caloric = False
            elif caloric_basis == 'Poynting&PhiSat':
                self.use_Poynting_caloric = True
                self.use_phis_sat_caloric = True
                self.use_Hvap_caloric = False
            elif caloric_basis == 'PhiSat':
                self.use_phis_sat_caloric = True
                self.use_Poynting_caloric = False
                self.use_Hvap_caloric = False
            elif caloric_basis == 'Psat':
                self.use_phis_sat_caloric = False
                self.use_Poynting_caloric = False
                self.use_Hvap_caloric = False
            elif caloric_basis == 'Hvap':
                self.use_phis_sat_caloric = False
                self.use_Poynting_caloric = False
                self.use_Hvap_caloric = True
        else:
            self.use_Poynting_caloric = use_Poynting
            self.use_phis_sat_caloric = use_phis_sat
            self.use_Hvap_caloric = use_Hvap_caloric



        if henry_components is None:
            henry_components = [False]*self.N
        self.has_henry_components = any(henry_components)
        self.henry_components = henry_components
        self.henry_data = henry_data

        self.composition_independent = isinstance(GibbsExcessModel, IdealSolution) and not self.has_henry_components

        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs
        
        self.model_id = 20000 + GibbsExcessModel.model_id

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

        self.transfer_data(new, zs, T, T_equal)
        return new


    def to(self, zs, T=None, P=None, V=None):
        try:
            T_equal = T == self.T
        except:
            T_equal = False

        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.N = self.N

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

    def transfer_data(self, new, zs, T, T_equal):
        new.VaporPressures = self.VaporPressures
        new.VolumeLiquids = self.VolumeLiquids
        new.eos_pure_instances = self.eos_pure_instances
        new.HeatCapacityGases = self.HeatCapacityGases
        new.EnthalpyVaporizations = self.EnthalpyVaporizations
        new.HeatCapacityLiquids = self.HeatCapacityLiquids


        new.Psats_poly_fit = self.Psats_poly_fit
        new._Psats_data = self._Psats_data
        new.Psat_extrpolation = self.Psat_extrpolation

        new.Cpgs_poly_fit = self.Cpgs_poly_fit
        new._Cpgs_data = self._Cpgs_data

        new.Cpls_poly_fit = self.Cpls_poly_fit
        new._Cpls_data = self._Cpls_data

        new.Vms_sat_poly_fit = self.Vms_sat_poly_fit
        new._Vms_sat_data = self._Vms_sat_data

        new._Hvap_data = self._Hvap_data
        new.Hvap_poly_fit = self.Hvap_poly_fit

        new.incompressible = self.incompressible

        new.equilibrium_basis = self.equilibrium_basis
        new.caloric_basis = self.caloric_basis

        new.use_phis_sat = self.use_phis_sat
        new.use_Poynting = self.use_Poynting
        new.P_DEPENDENT_H_LIQ = self.P_DEPENDENT_H_LIQ
        new.use_eos_volume = self.use_eos_volume
        new.use_Hvap_caloric = self.use_Hvap_caloric

        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs

        new.henry_data = self.henry_data
        new.henry_components = self.henry_components
        new.has_henry_components = self.has_henry_components

        new.composition_independent = self.composition_independent
        new.model_id = self.model_id

        new.use_Tait = self.use_Tait
        new._Tait_B_data = self._Tait_B_data
        new._Tait_C_data = self._Tait_C_data


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
    
    def lnphis_args(self):
        lnPsats = self.lnPsats()
        Poyntings = self.Poyntings()
        phis_sat = self.phis_sat()
        activity_args = self.GibbsExcessModel.lnphis_args()
        return (self.model_id, self.T, self.P, self.N, lnPsats, Poyntings, phis_sat) + activity_args


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
        VaporPressures, cmps = self.VaporPressures, range(self.N)
        T_REF_IG = self.T_REF_IG
        self._Psats_T_ref = [VaporPressures[i](T_REF_IG) for i in cmps]
        return self._Psats_T_ref

    def Psats_at(self, T):
        if self.Psats_poly_fit:
            return self._Psats_at_poly_fit(T, self._Psats_data, range(self.N))
        VaporPressures = self.VaporPressures
        return [VaporPressures[i](T) for i in range(self.N)]

    @staticmethod
    def _Psats_at_poly_fit(T, Psats_data, cmps):
        Psats = []
        T_inv = 1.0/T
        logT = log(T)
        Tmins, Tmaxes, coeffs = Psats_data[0], Psats_data[3], Psats_data[6]
        for i in cmps:
            if T < Tmins[i]:
                A, B, C = Psats_data[9][i]
                Psat = (A + B*T_inv + C*logT)
#                    A, B = _Psats_data[9][i]
#                    Psat = (A - B*T_inv)
#                    Psat = (T - Tmins[i])*_Psats_data[1][i] + _Psats_data[2][i]
            elif T > Tmaxes[i]:
                A, B, C = Psats_data[10][i]
                Psat = (A + B*T_inv + C*logT)
#                A, B = _Psats_data[10][i]
#                Psat = (A - B*T_inv)
#                Psat = (T - Tmaxes[i])*_Psats_data[4][i] + _Psats_data[5][i]
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
        T, cmps = self.T, range(self.N)
        if self.Psats_poly_fit:
            self._Psats = Psats = self._Psats_at_poly_fit(T, self._Psats_data, cmps)
#            _Psats_data = self._Psats_data
#            Tmins, Tmaxes, coeffs = _Psats_data[0], _Psats_data[3], _Psats_data[6]
#            for i in cmps:
#                if T < Tmins[i]:
#                    A, B, C = _Psats_data[9][i]
#                    Psat = (A + B*T_inv + C*logT)
##                    A, B = _Psats_data[9][i]
##                    Psat = (A - B*T_inv)
##                    Psat = (T - Tmins[i])*_Psats_data[1][i] + _Psats_data[2][i]
#                elif T > Tmaxes[i]:
#                    Psat = (T - Tmaxes[i])*_Psats_data[4][i] + _Psats_data[5][i]
#                else:
#                    Psat = 0.0
#                    for c in coeffs[i]:
#                        Psat = Psat*T + c
#                Psats.append(exp(Psat))
            return Psats


        self._Psats = Psats = []
        for i in self.VaporPressures:
            Psats.append(i.T_dependent_property(T))

        if self.has_henry_components:
            henry_components = self.henry_components
            henry_data = self.henry_data
            zs = self.zs

            for i in range(self.N):
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
    def _dPsats_dT_at_poly_fit(T, Psats_data, cmps, Psats):
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        dPsats_dT = []
        Tmins, Tmaxes, dcoeffs, coeffs_low, coeffs_high = Psats_data[0], Psats_data[3], Psats_data[7], Psats_data[9], Psats_data[10]
        for i in cmps:
            if T < Tmins[i]:
#                    A, B = _Psats_data[9][i]
#                    dPsat_dT = B*Tinv2*Psats[i]
                dPsat_dT = Psats[i]*(-coeffs_low[i][1]*Tinv2 + coeffs_low[i][2]*T_inv)
#                    dPsat_dT = _Psats_data[1][i]*Psats[i]#*exp((T - Tmins[i])*_Psats_data[1][i]
                                             #   + _Psats_data[2][i])
            elif T > Tmaxes[i]:
                dPsat_dT = Psats[i]*(-coeffs_high[i][1]*Tinv2 + coeffs_high[i][2]*T_inv)

#                dPsat_dT = _Psats_data[4][i]*Psats[i]#*exp((T - Tmaxes[i])
#                                                    #*_Psats_data[4][i]
#                                                    #+ _Psats_data[5][i])
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
        if self.Psats_poly_fit:
            return self._dPsats_dT_at_poly_fit(T, self._Psats_data, range(self.N), Psats)
        return [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]

    def dPsats_dT(self):
        try:
            return self._dPsats_dT
        except:
            pass
        T, cmps = self.T, range(self.N)
        # Need to reset the method because for the T bounded solver,
        # will normally get a different than prefered method as it starts
        # at the boundaries
        if self.Psats_poly_fit:
            try:
                Psats = self._Psats
            except AttributeError:
                Psats = self.Psats()
            self._dPsats_dT = dPsats_dT = self._dPsats_dT_at_poly_fit(T, self._Psats_data, cmps, Psats)
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
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        T_inv2 = T_inv*T_inv
        # Tinv3 = T_inv*T_inv*T_inv

        self._d2Psats_dT2 = d2Psats_dT2 = []
        if self.Psats_poly_fit:
            Psats_data = self._Psats_data
            Tmins, Tmaxes, d2coeffs = Psats_data[0], Psats_data[3], Psats_data[8]
            for i in cmps:
                if T < Tmins[i]:
#                    A, B = _Psats_data[9][i]
#                    d2Psat_dT2 = B*Psats[i]*(B*T_inv - 2.0)*Tinv3
                    A, B, C = Psats_data[9][i]
                    x0 = (B*T_inv - C)
                    d2Psat_dT2 = Psats[i]*(2.0*B*T_inv - C + x0*x0)*T_inv2
#                    d2Psat_dT2 = _Psats_data[1][i]*dPsats_dT[i]
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    x0 = (B*T_inv - C)
                    d2Psat_dT2 = Psats[i]*(2.0*B*T_inv - C + x0*x0)*T_inv2
#                    d2Psat_dT2 = _Psats_data[4][i]*dPsats_dT[i]
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
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        logT = log(T)
        lnPsats = []
        if self.Psats_poly_fit:
            Psats_data = self._Psats_data
            Tmins, Tmaxes, coeffs = Psats_data[0], Psats_data[3], Psats_data[6]
            for i in cmps:
                if T < Tmins[i]:
                    A, B, C = Psats_data[9][i]
                    Psat = (A + B*T_inv + C*logT)
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    Psat = (A + B*T_inv + C*logT)
#                    Psat = (T - Tmaxes[i])*_Psats_data[4][i] + _Psats_data[5][i]
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
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        if self.Psats_poly_fit:
            Psats_data = self._Psats_data
            Tmins, Tmaxes, dcoeffs = Psats_data[0], Psats_data[3], Psats_data[7]
            dlnPsats_dT = []
            for i in cmps:
                if T < Tmins[i]:
                    A, B, C = Psats_data[9][i]
                    dPsat_dT = (-B*Tinv2 + C*T_inv)
                elif T > Tmaxes[i]:
                    A, B, C = Psats_data[10][i]
                    dPsat_dT = (-B*Tinv2 + C*T_inv)
#                    dPsat_dT = _Psats_data[4][i]
                else:
                    dPsat_dT = 0.0
                    for c in dcoeffs[i]:
                        dPsat_dT = dPsat_dT*T + c
                dlnPsats_dT.append(dPsat_dT)
            return dlnPsats_dT

    def d2lnPsats_dT2(self):
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        T_inv2 = T_inv*T_inv
        # Tinv3 = T_inv*T_inv*T_inv
        if self.Psats_poly_fit:
            Psats_data = self._Psats_data
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
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        if self.Psats_poly_fit:
            dPsat_dT_over_Psats = []
            Psats_data = self._Psats_data
            Tmins, Tmaxes, dcoeffs, low_coeffs, high_coeffs = Psats_data[0], Psats_data[3], Psats_data[7], Psats_data[9], Psats_data[10]
            for i in cmps:
                if T < Tmins[i]:
                    dPsat_dT_over_Psat = (-low_coeffs[i][1]*Tinv2 + low_coeffs[i][2]*T_inv)
                elif T > Tmaxes[i]:
                    dPsat_dT_over_Psat = (-high_coeffs[i][1]*Tinv2 + high_coeffs[i][2]*T_inv)
#                    dPsat_dT_over_Psat = _Psats_data[4][i]
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
        T, cmps = self.T, range(self.N)
        T_inv = 1.0/T
        Tinv2 = T_inv*T_inv
        Tinv4 = Tinv2*Tinv2
        c0 = (T + T)*Tinv4
        if self.Psats_poly_fit:
            d2Psat_dT2_over_Psats = []
            Psats_data = self._Psats_data
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
        if self.Vms_sat_poly_fit:
            return self._Vms_sat_at(T, self._Vms_sat_data, range(self.N))
        VolumeLiquids = self.VolumeLiquids
        return [VolumeLiquids[i].T_dependent_property(T) for i in range(self.N)]

    def Vms_sat(self):
        try:
            return self._Vms_sat
        except AttributeError:
            pass
        T = self.T
        if self.Vms_sat_poly_fit:
#            self._Vms_sat = evaluate_linear_fits(self._Vms_sat_data, T)
#            return self._Vms_sat
            self._Vms_sat = Vms_sat = self._Vms_sat_at(T, self._Vms_sat_data, range(self.N))
            return Vms_sat
        elif self.use_eos_volume:
            Vms = []
            eoss = self.eos_pure_instances
            Psats = self.Psats()
            for i, e in enumerate(eoss):
                if T < e.Tc:
                    Vms.append(e.V_l_sat(T))
                else:
                    e = e.to(T=T, P=Psats[i])
                    try:
                        Vms.append(e.V_l)
                    except:
                        Vms.append(e.V_g)
            self._Vms_sat = Vms
            return Vms


        VolumeLiquids = self.VolumeLiquids
#        Psats = self.Psats()
#        self._Vms_sat = [VolumeLiquids[i](T, Psats[i]) for i in range(self.N)]
        self._Vms_sat = [VolumeLiquids[i].T_dependent_property(T) for i in range(self.N)]
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
        if self.Vms_sat_poly_fit:
            return self._dVms_sat_dT_at(T, self._Vms_sat_data, range(self.N))
        return [obj.T_dependent_property_derivative(T=T) for obj in self.VolumeLiquids]

    def dVms_sat_dT(self):
        try:
            return self._Vms_sat_dT
        except:
            pass
        T = self.T

        if self.Vms_sat_poly_fit:
#            self._Vms_sat_dT = evaluate_linear_fits_d(self._Vms_sat_data, T)
            self._Vms_sat_dT = self._dVms_sat_dT_at(T, self._Vms_sat_data, range(self.N))
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

        if self.Vms_sat_poly_fit:
#            self._d2Vms_sat_dT2 = evaluate_linear_fits_d2(self._Vms_sat_data, T)
#            return self._d2Vms_sat_dT2
            d2Vms_sat_dT2 = self._d2Vms_sat_dT2 = []

            Vms_sat_data = self._Vms_sat_data
            Tmins, Tmaxes, d2coeffs = Vms_sat_data[0], Vms_sat_data[3], Vms_sat_data[8]
            for i in range(self.N):
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
        if self.Vms_sat_poly_fit:
            self._Vms_sat_T_ref = evaluate_linear_fits(self._Vms_sat_data, T_REF_IG)
        else:
            VolumeLiquids, cmps = self.VolumeLiquids, range(self.N)
            self._Vms_sat_T_ref = [VolumeLiquids[i].T_dependent_property(T_REF_IG) for i in cmps]
        return self._Vms_sat_T_ref

    def dVms_sat_dT_T_ref(self):
        try:
            return self._dVms_sat_dT_T_ref
        except AttributeError:
            pass
        T_REF_IG = self.T_REF_IG
        if self.Vms_sat_poly_fit:
            self._dVms_sat_dT_T_ref = evaluate_linear_fits_d(self._Vms_sat_data, self.T)
        else:
            VolumeLiquids, cmps = self.VolumeLiquids, range(self.N)
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
        T, EnthalpyVaporizations, cmps = self.T, self.EnthalpyVaporizations, range(self.N)

        self._Hvaps = Hvaps = []
        if self.Hvap_poly_fit:
            Hvap_data = self._Hvap_data
            Tcs, Tcs_inv, coeffs = Hvap_data[2], Hvap_data[3], Hvap_data[4]
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
        T, EnthalpyVaporizations, cmps = self.T, self.EnthalpyVaporizations, range(self.N)

        self._dHvaps_dT = dHvaps_dT = []
        if self.Hvap_poly_fit:
            Hvap_data = self._Hvap_data
            Tcs, Tcs_inv, coeffs = Hvap_data[2], Hvap_data[3], Hvap_data[4]
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
        EnthalpyVaporizations, cmps = self.EnthalpyVaporizations, range(self.N)
        T_REF_IG = self.T_REF_IG
        self._Hvaps_T_ref = [EnthalpyVaporizations[i](T_REF_IG) for i in cmps]
        return self._Hvaps_T_ref

    def Poyntings_at(self, T, P, Psats=None, Vms=None):
        if not self.use_Poynting:
            return [1.0]*self.N

        cmps = range(self.N)
        if Psats is None:
            Psats = self.Psats_at(T)
        if Vms is None:
            Vms = self.Vms_sat_at(T)
        RT_inv = 1.0/(R*T)
        return [exp(Vms[i]*(P-Psats[i])*RT_inv) for i in cmps]

    def Poyntings(self):
        r'''Method to calculate and return the Poynting pressure correction
        factors of the phase, [-].

        .. math::
            \text{Poynting}_i = \exp\left(\frac{V_{m,i}(P-P_{sat})}{RT}\right)

        Returns
        -------
        Poyntings : list[float]
            Poynting pressure correction factors, [-]

        Notes
        -----
        The above formula is correct for pressure-independent molar volumes.
        When the volume does depend on pressure, the full expression is:

        .. math::
            \text{Poynting} = \exp\left[\frac{\int_{P_i^{sat}}^P V_i^l dP}{RT}\right]

        When a specified model e.g. the Tait equation is used, an analytical
        integral of this term is normally available.

        '''
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
        self._Poyntings = [trunc_exp(Vml*(P-Psat)*RT_inv) for Psat, Vml in zip(Psats, Vms_sat)]
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
        for i in range(self.N):
            x2 = Vms[i]
            x3 = Psats[i]

            x4 = P - x3
            x5 = x1*x2*x4
            dPoyntings_dTi = -RT_inv*(x2*dPsats_dT[i] - x4*dVms_sat_dT[i] + x5)*trunc_exp(x0*x5)
            dPoyntings_dT.append(dPoyntings_dTi)
        return dPoyntings_dT

    def dPoyntings_dT_at(self, T, P, Psats=None, Vms=None, dPsats_dT=None, dVms_sat_dT=None):
        if not self.use_Poynting:
            return [0.0]*self.N

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
        for i in range(self.N):
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
        for i in range(self.N):
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
        for i in range(self.N):
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
        for i in range(self.N):
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
        phis_sat = []
        for i in self.eos_pure_instances:
            try:
                phis_sat.append(i.phi_sat(min(T, i.Tc), polish=True))
            except Exception as e:
                if T < self.PHI_SAT_IDEAL_TR*i.Tc:
                    phis_sat.append(1.0)
                else:
                    raise e
        return phis_sat

    def phis_sat(self):
        r'''Method to calculate and return the saturation fugacity coefficient
        correction factors of the phase, [-].

        These are calculated from the
        provided pure-component equations of state. This term should only be
        used with a consistent vapor-phase cubic equation of state.

        Returns
        -------
        phis_sat : list[float]
            Saturation fugacity coefficient correction factors, [-]

        Notes
        -----

        .. warning::
            This factor cannot be calculated when a compound is supercritical,
            as there is no longer any vapor-liquid pure-component equilibrium
            (by definition).

        '''
        try:
            return self._phis_sat
        except AttributeError:
            pass
        if not self.use_phis_sat:
            self._phis_sat = [1.0]*self.N
            return self._phis_sat

        T = self.T
        self._phis_sat = phis_sat = []
        for i in self.eos_pure_instances:
            try:
                phis_sat.append(i.phi_sat(min(T, i.Tc), polish=True))
            except Exception as e:
                if T < self.PHI_SAT_IDEAL_TR*i.Tc:
                    phis_sat.append(1.0)
                else:
                    raise e
        return phis_sat




    def dphis_sat_dT_at(self, T):
        if not self.use_phis_sat:
            return [0.0]*self.N
        dphis_sat_dT = []
        for i in self.eos_pure_instances:
            try:
                dphis_sat_dT.append(i.dphi_sat_dT(min(T, i.Tc)))
            except Exception as e:
                if T < self.PHI_SAT_IDEAL_TR*i.Tc:
                    dphis_sat_dT.append(0.0)
                else:
                    raise e
        return dphis_sat_dT

    def dphis_sat_dT(self):
        try:
            return self._dphis_sat_dT
        except AttributeError:
            pass

        if not self.use_phis_sat:
            self._dphis_sat_dT = [0.0]*self.N
            return self._dphis_sat_dT

        T = self.T
        self._dphis_sat_dT = dphis_sat_dT = []
        for i in self.eos_pure_instances:
            try:
                dphis_sat_dT.append(i.dphi_sat_dT(min(T, i.Tc)))
            except Exception as e:
                if T < self.PHI_SAT_IDEAL_TR*i.Tc:
                    dphis_sat_dT.append(0.0)
                else:
                    raise e
        return dphis_sat_dT

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
        self._d2phis_sat_dT2 = d2phis_sat_dT2 = []
        for i in self.eos_pure_instances:
            try:
                d2phis_sat_dT2.append(i.d2phi_sat_dT2(min(T, i.Tc)))
            except Exception as e:
                if T < self.PHI_SAT_IDEAL_TR*i.Tc:
                    d2phis_sat_dT2.append(0.0)
                else:
                    raise e
        return d2phis_sat_dT2


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
                for i in range(self.N)]

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
                for i in range(self.N)]
        return self._phis


    def lnphis(self):
        try:
            return self._lnphis
        except AttributeError:
            pass
        try:
            self._lnphis = [log(i) for i in self.phis()]
        except:
            # Zero Psats - must compute them inline
            P = self.P
            try:
                gammas = self._gammas
            except AttributeError:
                gammas = self.gammas()
            try:
                lnPsats = self._lnPsats
            except AttributeError:
                lnPsats = self.lnPsats()
            try:
                phis_sat = self._phis_sat
            except AttributeError:
                phis_sat = self.phis_sat()
            try:
                Poyntings = self._Poyntings
            except AttributeError:
                Poyntings = self.Poyntings()
            P_inv = 1.0/P
            self._lnphis = [log(gammas[i]*Poyntings[i]*phis_sat[i]*P_inv) + lnPsats[i]
                    for i in range(self.N)]

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
#                for i in range(self.N)]
#
#    def dphis_dxs(self):
#        if
    def dphis_dT(self):
        try:
            return self._dphis_dT
        except AttributeError:
            pass
        P = self.P
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
        for i in range(self.N):
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
        for i in range(self.N):
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
            phis = [gammas[i]*Psats[i]*Poyntings[i]*phis_sat[i]*P_inv for i in range(self.N)]
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
            \frac{\partial \ln \phi_i}{\partial P} =
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
        that case, :math:`\frac{\partial \ln \phi_i}{\partial P}=\frac{1}{P}`.
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

        self._dlnphis_dP = [dPoyntings_dP[i]/Poyntings[i] - P_inv for i in range(self.N)]
        return self._dlnphis_dP

    def gammas_at(self, T, zs):
        if self.composition_independent:
            return [1.0]*self.N
        return self.GibbsExcessModel.to_T_xs(T, zs).gammas()

    def dgammas_dT_at(self, T, zs):
        if self.composition_independent:
            return [0.0]*self.N
        return self.GibbsExcessModel.to_T_xs(T, zs).dgammas_dT()

    def gammas(self):
        r'''Method to calculate and return the activity coefficients of the
        phase, [-]. This is a direct call to
        :obj:`GibbsExcess.gammas <thermo.activity.GibbsExcess.gammas>`.

        Returns
        -------
        gammas : list[float]
            Activity coefficients, [-]
        '''
        try:
            return self.GibbsExcessModel._gammas
        except AttributeError:
            return self.GibbsExcessModel.gammas()

    def dgammas_dT(self):
        r'''Method to calculate and return the temperature derivative of
        activity coefficients of the phase, [-].

        This is a direct call to
        :obj:`GibbsExcess.dgammas_dT <thermo.activity.GibbsExcess.dgammas_dT>`.

        Returns
        -------
        dgammas_dT : list[float]
            First temperature derivative of the activity coefficients, [1/K]
        '''
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
        zs, cmps = self.zs, range(self.N)
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
                Cpl_integrals_pure = self._Cpl_integrals_pure()
                dVms_sat_dT_T_ref = self.dVms_sat_dT_T_ref()
                Vms_sat_T_ref = self.Vms_sat_T_ref()
                Psats_T_ref = self.Psats_T_ref()

                Hvaps = self.Hvaps()

                H = 0.0
                for i in range(self.N):
                    H += zs[i]*(Cpl_integrals_pure[i] - Hvaps_T_ref[i]) #
                    # If we can use the liquid heat capacity and prove its consistency

                    # This bit is the differential with respect to pressure
                    dP = P - Psats_T_ref[i]
                    H += zs[i]*dP*(Vms_sat_T_ref[i] - T_REF_IG*dVms_sat_dT_T_ref[i])
        else:
            Hvaps = self.Hvaps()
            for i in range(self.N):
                H += zs[i]*(Cpig_integrals_pure[i] - Hvaps[i])
        H += self.GibbsExcessModel.HE()
#        self._H = H
        return H
    del H_old

    def H(self):
        r'''Method to calculate the enthalpy of the
        :obj:`GibbsExcessLiquid` phase. Depending on the settings of the phase, this can
        include the effects of activity coefficients
        :obj:`gammas <GibbsExcessLiquid.gammas>`, pressure correction terms
        :obj:`Poyntings <GibbsExcessLiquid.Poyntings>`, and pure component
        saturation fugacities :obj:`phis_sat <GibbsExcessLiquid.phis_sat>`
        as well as the pure component vapor pressures.

        When `caloric_basis` is 'Poynting&PhiSat':

        .. math::
            H = H_{\text{excess}} + \sum_i z_i \left[-RT^2\left(
            \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
            + \frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + \frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}} \right)
            + \int_{T,ref}^T C_{p,ig} dT \right]

        When `caloric_basis` is 'PhiSat':

        .. math::
            H = H_{\text{excess}} + \sum_i z_i \left[-RT^2\left(
            \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
            + \frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            \right)
            + \int_{T,ref}^T C_{p,ig} dT \right]

        When `caloric_basis` is 'Poynting':

        .. math::
            H = H_{\text{excess}} + \sum_i z_i \left[-RT^2\left(
            + \frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + \frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}} \right)
            + \int_{T,ref}^T C_{p,ig} dT \right]

        When `caloric_basis` is 'Psat':

        .. math::
            H = H_{\text{excess}} + \sum_i z_i \left[-RT^2\left(
            + \frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
             \right)
            + \int_{T,ref}^T C_{p,ig} dT \right]

        When `caloric_basis` is 'Hvap':

        .. math::
            H = H_{\text{excess}} + \sum_i z_i\left[-H_{vap,i}
            + \int_{T,ref}^T C_{p,ig} dT \right]

        Returns
        -------
        H : float
            Enthalpy of the phase, [J/(mol)]

        Notes
        -----
        '''
        try:
            return self._H
        except AttributeError:
            pass
        H = 0.0
        T = self.T
        nRT2 = -R*T*T
        zs, cmps = self.zs, range(self.N)
        try:
            Cpig_integrals_pure = self._Cpig_integrals_pure
        except AttributeError:
            Cpig_integrals_pure = self.Cpig_integrals_pure()

        if self.use_Hvap_caloric:
            Hvaps = self.Hvaps()
            for i in range(self.N):
                H += zs[i]*(Cpig_integrals_pure[i] - Hvaps[i])
        else:
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
        T, P, zs, cmps = self.T, self.P, self.zs, range(self.N)
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
                    for i in range(self.N):
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
                Cpl_integrals_over_T_pure = self._Cpl_integrals_over_T_pure()
                T_REF_IG_INV = self.T_REF_IG_INV
                dVms_sat_dT_T_ref = self.dVms_sat_dT_T_ref()
                # Vms_sat_T_ref = self.Vms_sat_T_ref()

                for i in range(self.N):
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
#                    _Cpl_integrals_over_T_pure = self._Cpl_integrals_over_T_pure()
#                    T_REF_IG_INV = self.T_REF_IG_INV
#
#                    for i in range(self.N):
#                        dSi = -_Cpl_integrals_over_T_pure[i]
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
        r'''Method to calculate the entropy of the
        :obj:`GibbsExcessLiquid` phase. Depending on the settings of the phase, this can
        include the effects of activity coefficients
        :obj:`gammas <GibbsExcessLiquid.gammas>`, pressure correction terms
        :obj:`Poyntings <GibbsExcessLiquid.Poyntings>`, and pure component
        saturation fugacities :obj:`phis_sat <GibbsExcessLiquid.phis_sat>`
        as well as the pure component vapor pressures.

        When `caloric_basis` is 'Poynting&PhiSat':

        .. math::
            S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
            - \sum_i z_i\left[R\left(
            T \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
            + T\frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + T\frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}}
            + \ln(P_{\text{sat},i}) + \ln\left(\frac{\text{Poynting}\cdot\phi_{\text{sat},i}}{P}\right)
            \right) - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

        When `caloric_basis` is 'PhiSat':

        .. math::
            S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
            - \sum_i z_i\left[R\left(
            T \frac{\frac{\partial \phi_{\text{sat},i}}{\partial T}}{\phi_{\text{sat},i}}
            + T\frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + \ln(P_{\text{sat},i}) + \ln\left(\frac{\phi_{\text{sat},i}}{P}\right)
            \right) - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

        When `caloric_basis` is 'Poynting':

        .. math::
            S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
            - \sum_i z_i\left[R\left(
            T\frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + T\frac{\frac{\text{Poynting}}{\partial T}}{\text{Poynting}}
            + \ln(P_{\text{sat},i}) + \ln\left(\frac{\text{Poynting}}{P}\right)
            \right) - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

        When `caloric_basis` is 'Psat':

        .. math::
            S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
            - \sum_i z_i\left[R\left(
            T\frac{\frac{\partial P_{\text{sat},i}}{\partial T}}{P_{\text{sat},i}}
            + \ln(P_{\text{sat},i}) + \ln\left(\frac{1}{P}\right)
            \right) - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

        When `caloric_basis` is 'Hvap':

        .. math::
            S = S_{\text{excess}} - R\sum_i z_i\ln z_i - R\ln\left(\frac{P}{P_{ref}}\right)
            - \sum_i z_i\left[R\left(\ln P_{\text{sat},i} + \ln\left(\frac{1}{P}\right)\right)
            + \frac{H_{vap,i}}{T}
            - \int_{T,ref}^T \frac{C_{p,ig,i}}{T} dT \right]

        Returns
        -------
        S : float
            Entropy of the phase, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._S
        except AttributeError:
            pass
        T, P = self.T, self.P
        P_inv = 1.0/P
        zs, cmps = self.zs, range(self.N)

        log_zs = self.log_zs()
        S_comp = 0.0
        for i in cmps:
            S_comp -= zs[i]*log_zs[i]
        S = S_comp - log(P*self.P_REF_IG_INV)
        S *= R
        try:
            Cpig_integrals_over_T_pure = self._Cpig_integrals_over_T_pure
        except AttributeError:
            Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()

        try:
            lnPsats = self._lnPsats
        except AttributeError:
            lnPsats = self.lnPsats()

        use_Poynting, use_phis_sat, use_Hvap_caloric = self.use_Poynting, self.use_phis_sat, self.use_Hvap_caloric

        if use_Hvap_caloric:
            Hvaps = self.Hvaps()
            T_inv = 1.0/T
            logP_inv = log(P_inv)
            # Almost the same as no Poynting
            for i in cmps:
                S -= zs[i]*(R*(lnPsats[i] + logP_inv)
                            - Cpig_integrals_over_T_pure[i] + Hvaps[i]*T_inv)
        else:
            dPsats_dT_over_Psats = self.dPsats_dT_over_Psats()
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
                for i in range(self.N):
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
                for i in range(self.N):
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
            for i in range(self.N):
                Cp += zs[i]*(Cpigs_pure[i] - dHvaps_dT[i])

        Cp += self.GibbsExcessModel.CpE()
        return Cp

    def Cp(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        T, zs, cmps = self.T, self.zs, range(self.N)
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
        zs = self.zs
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
                    for i in range(self.N):
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
                # lnPsats = self.lnPsats()
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
                for i in range(self.N):
                    dS_dT -= zs[i]*(P*d2Vms_sat_dT2[i] + RT*d2lnPsats_dT2[i]
                    + 2.0*R*dlnPsats_dT[i]- Cpigs_pure[i]*T_inv)

        dS_dT += self.GibbsExcessModel.dSE_dT()
        return dS_dT

    def dS_dT(self):
        try:
            return self._dS_dT
        except AttributeError:
            pass
        T, zs, cmps = self.T, self.zs, range(self.N)
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
        zs = self.zs
        dH_dP = 0.0
        if self.use_Poynting:
            nRT2 = -R*T*T
            Poyntings = self.Poyntings()
            dPoyntings_dP = self.dPoyntings_dP()
            dPoyntings_dT = self.dPoyntings_dT()
            d2Poyntings_dPdT = self.d2Poyntings_dPdT()
            for i in range(self.N):
                Poy_inv = 1.0/Poyntings[i]
                dH_dP += nRT2*zs[i]*Poy_inv*(d2Poyntings_dPdT[i] - dPoyntings_dP[i]*dPoyntings_dT[i]*Poy_inv)

#        if self.P_DEPENDENT_H_LIQ:
#            if self.use_IG_Cp:
#                Vms_sat = self.Vms_sat()
#                dVms_sat_dT = self.dVms_sat_dT()
#                Psats = self.Psats()
#                for i in range(self.N):
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
            for i in range(self.N):
                Poy_inv = 1.0/Poyntings[i]
                dS_dP -= zs[i]*R*Poy_inv*(dPoyntings_dP[i] - Poyntings[i]*P_inv
                        +T*(d2Poyntings_dPdT[i] - dPoyntings_dP[i]*dPoyntings_dT[i]*Poy_inv))
        else:
            dS_dP = 0.0
#        if self.P_DEPENDENT_H_LIQ:
#            if self.use_IG_Cp:
#                dVms_sat_dT = self.dVms_sat_dT()
#                Psats = self.Psats()
#                for i in range(self.N):
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
        for i in range(self.N):
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
        for i in range(self.N):
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
        for i in range(self.N):
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
            self._dP_dV = self.INCOMPRESSIBLE_CONST #1.0/self.VolumeLiquidMixture.property_derivative_P(self.T, self.P, self.zs, order=1)

        return self._dP_dV

    def d2P_dV2(self):
        try:
            return self._d2P_dV2
        except AttributeError:
            pass
        if self.incompressible:
            self._d2P_dV2 = self.INCOMPRESSIBLE_CONST #self.d2V_dP2()/-(self.dP_dV())**-3
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
            self._d2P_dT2 = -self.d2V_dT2()/self.INCOMPRESSIBLE_CONST
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

        self._Tait_Bs = evaluate_linear_fits(self._Tait_B_data, self.T)
        return self._Tait_Bs

    def dTait_B_dTs(self):
        try:
            return self._dTait_B_dTs
        except:
            pass

        self._dTait_B_dTs = evaluate_linear_fits_d(self._Tait_B_data, self.T)
        return self._dTait_B_dTs

    def d2Tait_B_dT2s(self):
        try:
            return self._d2Tait_B_dT2s
        except:
            pass

        self._d2Tait_B_dT2s = evaluate_linear_fits_d2(self._Tait_B_data, self.T)
        return self._d2Tait_B_dT2s

    def Tait_Cs(self):
        try:
            return self._Tait_Cs
        except:
            pass

        self._Tait_Cs = evaluate_linear_fits(self._Tait_C_data, self.T)
        return self._Tait_Cs

    def dTait_C_dTs(self):
        try:
            return self._dTait_C_dTs
        except:
            pass

        self._dTait_C_dTs = evaluate_linear_fits_d(self._Tait_C_data, self.T)
        return self._dTait_C_dTs

    def d2Tait_C_dT2s(self):
        try:
            return self._d2Tait_C_dT2s
        except:
            pass

        self._d2Tait_C_dT2s = evaluate_linear_fits_d2(self._Tait_C_data, self.T)
        return self._d2Tait_C_dT2s

    def Tait_Vs(self):
        Vms_sat = self.Vms_sat()
        Psats = self.Psats()
        Tait_Bs = self.Tait_Bs()
        Tait_Cs = self.Tait_Cs()
        P = self.P
        return [Vms_sat[i]*(1.0  - Tait_Cs[i]*log((Tait_Bs[i] + P)/(Tait_Bs[i] + Psats[i]) ))
                for i in range(self.N)]


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
        T, P = self.T, self.P


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
#                                      for i in range(self.N)]
##        return self._dH_dP_integrals_Tait
#        print(_dH_dP_integrals_Tait)
#        self._dH_dP_integrals_Tait2 = _dH_dP_integrals_Tait
#        return self._dH_dP_integrals_Tait2

#        dH_dP_integrals_Tait = []
        for i in range(self.N):
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
    ideal_gas_basis = True
    force_phase = 's'
    phase = 's'
    is_gas = False
    is_liquid = False
    is_solid = True
    pure_references = ('HeatCapacityGases','SublimationPressures', 'VolumeSolids', 'EnthalpySublimations')
    pure_reference_types = (HeatCapacityGas, SublimationPressure, VolumeSolid, EnthalpySublimation)


    model_attributes = ('Hfs', 'Gfs', 'Sfs','GibbsExcessModel',
                        'eos_pure_instances', 'use_Poynting', 'use_phis_sat',
                        'use_eos_volume', 'henry_components',
                        'henry_data', 'Psat_extrpolation') + pure_references

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

'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains implementations of :obj:`TPDependentProperty <thermo.utils.TPDependentProperty>`
representing liquid and vapor viscosity. A variety of estimation
and data methods are available as included in the `chemicals` library.
Additionally liquid and vapor mixture viscosity predictor objects
are implemented subclassing  :obj:`MixtureProperty <thermo.utils.MixtureProperty>`.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.


.. contents:: :local:

Pure Liquid Viscosity
=====================
.. autoclass:: ViscosityLiquid
    :members: calculate, calculate_P, test_method_validity, test_method_validity_P,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods, ranked_methods_P
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: viscosity_liquid_methods
.. autodata:: viscosity_liquid_methods_P

Pure Gas Viscosity
==================
.. autoclass:: ViscosityGas
    :members: calculate, calculate_P, test_method_validity, test_method_validity_P,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods, ranked_methods_P
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: viscosity_gas_methods
.. autodata:: viscosity_gas_methods_P

Mixture Liquid Viscosity
========================
.. autoclass:: ViscosityLiquidMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: viscosity_liquid_mixture_methods

Mixture Gas Viscosity
=====================
.. autoclass:: ViscosityGasMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: viscosity_gas_mixture_methods


'''


__all__ = ['viscosity_liquid_methods', 'viscosity_liquid_methods_P',
           'ViscosityLiquid', 'ViscosityGas', 'viscosity_gas_methods',
           'viscosity_gas_methods_P', 'ViscosityLiquidMixture',
           'ViscosityGasMixture', 'viscosity_liquid_mixture_methods',
           'viscosity_gas_mixture_methods',
           'MIXING_LOG_MOLAR', 'MIXING_LOG_MASS',
           'BROKAW', 'HERNING_ZIPPERER', 'WILKE',
           'DUTT_PRASAD', 'VISWANATH_NATARAJAN_3', 'VISWANATH_NATARAJAN_2',
           'VISWANATH_NATARAJAN_2E', 'LETSOU_STIEL', 'PRZEDZIECKI_SRIDHAR',
           'LUCAS', 'GHARAGHEIZI', 'YOON_THODOS', 'STIEL_THODOS', 'LUCAS_GAS']

from chemicals import miscdata, viscosity
from chemicals.dippr import EQ101, EQ102
from chemicals.identifiers import CAS_to_int
from chemicals.miscdata import JOBACK, lookup_VDI_tabular_data
from chemicals.utils import none_and_length_check
from chemicals.viscosity import (
    PPDS9,
    Brokaw,
    Herning_Zipperer,
    Letsou_Stiel,
    Lucas,
    Lucas_gas,
    Przedziecki_Sridhar,
    Stiel_Thodos,
    Viswanath_Natarajan_2,
    Viswanath_Natarajan_2_exponential,
    Viswanath_Natarajan_3,
    Wilke_prefactored,
    Wilke_prefactors,
    Yoon_Thodos,
    dPPDS9_dT,
    viscosity_gas_Gharagheizi,
)
from fluids.numerics import brenth, exp, horner, isinf, isnan, log, sqrt

from thermo import electrochem
from thermo.coolprop import CoolProp_failing_PT_flashes, CoolProp_T_dependent_property, PhaseSI, PropsSI, coolprop_dict, coolprop_fluids, has_CoolProp
from thermo.electrochem import Laliberte_viscosity
from thermo.utils import (
    COOLPROP,
    DIPPR_PERRY_8E,
    LINEAR,
    MIXING_LOG_MASS,
    MIXING_LOG_MOLAR,
    NEGLECT_P,
    REFPROP_FIT,
    VDI_PPDS,
    VDI_TABULAR,
    MixtureProperty,
    TPDependentProperty,
)
from thermo.vapor_pressure import VaporPressure
from thermo.volume import VolumeGas, VolumeLiquid

DUTT_PRASAD = 'DUTT_PRASAD'
VISWANATH_NATARAJAN_3 = 'VISWANATH_NATARAJAN_3'
VISWANATH_NATARAJAN_2 = 'VISWANATH_NATARAJAN_2'
VISWANATH_NATARAJAN_2E = 'VISWANATH_NATARAJAN_2E'
LETSOU_STIEL = 'LETSOU_STIEL'
PRZEDZIECKI_SRIDHAR = 'PRZEDZIECKI_SRIDHAR'
LUCAS = 'LUCAS'

viscosity_liquid_methods = [REFPROP_FIT, COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, DUTT_PRASAD, VISWANATH_NATARAJAN_3,
                         VISWANATH_NATARAJAN_2, VISWANATH_NATARAJAN_2E,
                         VDI_TABULAR, LETSOU_STIEL, JOBACK, PRZEDZIECKI_SRIDHAR]
"""Holds all low-pressure methods available for the ViscosityLiquid class, for
use in iterating over them."""
viscosity_liquid_methods_P = [COOLPROP, LUCAS, NEGLECT_P]
"""Holds all high-pressure methods available for the ViscosityLiquid class, for
use in iterating over them."""


class ViscosityLiquid(TPDependentProperty):
    r'''Class for dealing with liquid viscosity as a function of
    temperature and pressure.

    For low-pressure (at 1 atm while under the vapor pressure; along the
    saturation line otherwise) liquids, there are six coefficient-based methods
    from three data sources, one source of tabular information, two
    corresponding-states estimators, one group contribution method, and the
    external library CoolProp.

    For high-pressure liquids (also, <1 atm liquids), there is one
    corresponding-states estimator, and the external library CoolProp.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical
    MW : float, optional
        Molecular weight, [g/mol]
    Tm : float, optional
        Melting point, [K]
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    Vc : float, optional
        Critical volume, [m^3/mol]
    omega : float, optional
        Acentric factor, [-]
    Psat : float or callable, optional
        Vapor pressure at a given temperature or callable for the same, [Pa]
    Vml : float or callable, optional
        Liquid molar volume at a given temperature and pressure or callable
        for the same, [m^3/mol]
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    To iterate over all methods, use the lists stored in
    :obj:`viscosity_liquid_methods` and :obj:`viscosity_liquid_methods_P` for
    low and high pressure methods respectively.

    Low pressure methods:

    **DUTT_PRASAD**:
        A simple function as expressed in [1]_, with data available for
        100 fluids. Temperature limits are available for all fluids. See
        :obj:`chemicals.viscosity.Viswanath_Natarajan_3` for details.
    **VISWANATH_NATARAJAN_3**:
        A simple function as expressed in [1]_, with data available for
        432 fluids. Temperature limits are available for all fluids. See
        :obj:`chemicals.viscosity.Viswanath_Natarajan_3` for details.
    **VISWANATH_NATARAJAN_2**:
        A simple function as expressed in [1]_, with data available for
        135 fluids. Temperature limits are available for all fluids. See
        :obj:`chemicals.viscosity.Viswanath_Natarajan_2` for details.
    **VISWANATH_NATARAJAN_2E**:
        A simple function as expressed in [1]_, with data available for
        14 fluids. Temperature limits are available for all fluids. See
        :obj:`chemicals.viscosity.Viswanath_Natarajan_2_exponential` for details.
    **DIPPR_PERRY_8E**:
        A collection of 337 coefficient sets from the DIPPR database published
        openly in [4]_. Provides temperature limits for all its fluids.
        :obj:`EQ101 <chemicals.dippr.EQ101>` is used for its fluids.
    **LETSOU_STIEL**:
        CSP method, described in :obj:`chemicals.viscosity.Letsou_Stiel`.
    **PRZEDZIECKI_SRIDHAR**:
        CSP method, described in :obj:`chemicals.viscosity.Przedziecki_Sridhar`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [2]_. Very slow.
    **VDI_TABULAR**:
        Tabular data in [3]_ along the saturation curve; interpolation is as
        set by the user or the default.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS, published
        openly in [3]_. Provides no temperature limits, but has been designed
        for extrapolation. Extrapolated to low temperatures it provides a
        smooth exponential increase. However, for some chemicals such as
        glycerol, extrapolated to higher temperatures viscosity is predicted
        to increase above a certain point.
    **JOBACK**:
        An estimation method for organic substances in [5]_; this also requires
        molecular weight as an input.
    **REFPROP_FIT**:
        A series of higher-order polynomial fits to the calculated results from
        the equations implemented in REFPROP.

    High pressure methods:

    **LUCAS**:
        CSP method, described in :obj:`chemicals.viscosity.Lucas`. Calculates a
        low-pressure liquid viscosity as its input.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [2]_. Very slow, but unparalled in accuracy for pressure
        dependence.

    A minimum viscosity value of 1e-5 Pa*s is set according to [4]_.
    This is also just above the lowest experimental values of viscosity of
    helium, 9.4e-6 Pa*s. This excludes the behavior of superfluids,
    and also systems where the mean free path between moleules approaches
    the geometry of the system and then the viscosity is geometry-dependent.

    See Also
    --------
    chemicals.viscosity.Viswanath_Natarajan_3
    chemicals.viscosity.Viswanath_Natarajan_2
    chemicals.viscosity.Viswanath_Natarajan_2_exponential
    chemicals.viscosity.Letsou_Stiel
    chemicals.viscosity.Przedziecki_Sridhar
    chemicals.viscosity.Lucas
    thermo.joback.Joback

    References
    ----------
    .. [1] Viswanath, Dabir S., and G. Natarajan. Databook On The Viscosity Of
       Liquids. New York: Taylor & Francis, 1989
    .. [2] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    .. [4] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [5] Joback, K.G., and R.C. Reid. "Estimation of Pure-Component
       Properties from Group-Contributions." Chemical Engineering
       Communications 57, no. 1-6 (July 1, 1987): 233-43.
       doi:10.1080/00986448708960487.
    .. [6] Trachenko, K., and V. V. Brazhkin. "Minimal Quantum Viscosity from
       Fundamental Physical Constants." Science Advances, April 2020.
       https://doi.org/10.1126/sciadv.aba3747.
    '''

    name = 'liquid viscosity'
    units = 'Pa*s'

    @staticmethod
    def interpolation_P(P):
        '''log(P) interpolation transformation by default.
        '''
        return log(P)

    @staticmethod
    def interpolation_T(T):
        '''Function to make the data-based interpolation as linear as possible.
        This transforms the input `T` into the `1/T` domain.
        '''
        return 1./T

    @staticmethod
    def interpolation_property(P):
        '''log(P) interpolation transformation by default.
        '''
        return log(P)

    @staticmethod
    def interpolation_property_inv(P):
        '''exp(P) interpolation transformation by default; reverses
        :obj:`interpolation_property_inv`.
        '''
        return exp(P)

    tabular_extrapolation_permitted = True
    """Allow tabular extrapolation by default."""
    property_min = 0.0
    """Mimimum valid value of liquid viscosity."""
    property_max = 2E8
    """Maximum valid value of liquid viscosity. Generous limit, as
    the value is that of bitumen in a Pitch drop experiment."""

    ranked_methods = [REFPROP_FIT, COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, DUTT_PRASAD, VISWANATH_NATARAJAN_3,
                      VISWANATH_NATARAJAN_2, VISWANATH_NATARAJAN_2E,
                      VDI_TABULAR, LETSOU_STIEL, JOBACK, PRZEDZIECKI_SRIDHAR]
    """Default rankings of the low-pressure methods."""
    ranked_methods_P = [COOLPROP, LUCAS, NEGLECT_P]
    """Default rankings of the high-pressure methods."""

    obj_references = pure_references = ('Psat', 'Vml')
    obj_references_types = pure_reference_types = (VaporPressure, VolumeLiquid)

    custom_args = ('MW', 'Tm', 'Tc', 'Pc', 'Vc', 'omega', 'Psat', 'Vml')

    DEFAULT_EXTRAPOLATION_MIN = 1e-5
    def __init__(self, CASRN='', MW=None, Tm=None, Tc=None, Pc=None, Vc=None,
                 omega=None, Psat=None, Vml=None, extrapolation='Arrhenius',
                 extrapolation_min=DEFAULT_EXTRAPOLATION_MIN,
                 **kwargs):
        self.CASRN = CASRN
        self.MW = MW
        self.Tm = Tm
        self.Tc = Tc
        self.Pc = Pc
        self.Vc = Vc
        self.omega = omega
        self.Psat = Psat
        self.Vml = Vml
        if 'extrapolation_min' not in kwargs:
            kwargs['extrapolation_min'] = extrapolation_min
        super().__init__(extrapolation, **kwargs)

    def load_all_methods(self, load_data=True):
        r'''Method which picks out coefficients for the specified chemical
        from the various dictionaries and DataFrames storing it. All data is
        stored as attributes. This method also sets :obj:`Tmin`, :obj:`Tmax`,
        :obj:`all_methods` and obj:`all_methods_P` as a set of methods for
        which the data exists for.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods, methods_P = [], [NEGLECT_P]
        self.T_limits = T_limits = {}
        self.all_methods = set()
        CASRN = self.CASRN
        if load_data and CASRN:
            CASRN_int = None if not CASRN else CAS_to_int(CASRN)
            jb_df = miscdata.joback_predictions
            if self.MW is not None and CASRN_int in jb_df.index:
                mul0 = float(jb_df.at[CASRN_int, 'mul0'])
                if not isnan(mul0):
                    methods.append(JOBACK)
                    self.joback_coeffs = [mul0, float(jb_df.at[CASRN_int, 'mul1'])]

                    Tmin_jb, Tmax_jb = float(jb_df.at[CASRN_int, 'Tm']), float(jb_df.at[CASRN_int, 'Tc'])*2.5
                    T_limits[JOBACK] = (Tmin_jb, Tmax_jb)

            if has_CoolProp() and CASRN in coolprop_dict:
                CP_f = coolprop_fluids[CASRN]
                if CP_f.has_mu:
                    self.CP_f = CP_f
                    methods.append(COOLPROP)
                    methods_P.append(COOLPROP)
                    T_limits[COOLPROP] = (max(CP_f.Tmin, CP_f.Tt), self.CP_f.Tc)
            if CASRN in miscdata.VDI_saturation_dict:
                Ts, props = lookup_VDI_tabular_data(CASRN, 'Mu (l)')
                self.add_tabular_data(Ts, props, VDI_TABULAR, check_properties=False)
                del self._method
            if CASRN in viscosity.mu_data_Dutt_Prasad.index:
                methods.append(DUTT_PRASAD)
                A, B, C, self.DUTT_PRASAD_Tmin, self.DUTT_PRASAD_Tmax = viscosity.mu_values_Dutt_Prasad[viscosity.mu_data_Dutt_Prasad.index.get_loc(CASRN)].tolist()
                self.DUTT_PRASAD_coeffs = [A - 3.0, B, C]
                T_limits[DUTT_PRASAD] = (self.DUTT_PRASAD_Tmin, self.DUTT_PRASAD_Tmax)
            if CASRN in viscosity.mu_data_VN3.index:
                methods.append(VISWANATH_NATARAJAN_3)
                A, B, C, self.VISWANATH_NATARAJAN_3_Tmin, self.VISWANATH_NATARAJAN_3_Tmax = viscosity.mu_values_VN3[viscosity.mu_data_VN3.index.get_loc(CASRN)].tolist()
                self.VISWANATH_NATARAJAN_3_coeffs = [A - 3.0, B, C]
                T_limits[VISWANATH_NATARAJAN_3] = (self.VISWANATH_NATARAJAN_3_Tmin, self.VISWANATH_NATARAJAN_3_Tmax)
            if CASRN in viscosity.mu_data_VN2.index:
                methods.append(VISWANATH_NATARAJAN_2)
                A, B, self.VISWANATH_NATARAJAN_2_Tmin, self.VISWANATH_NATARAJAN_2_Tmax = viscosity.mu_values_VN2[viscosity.mu_data_VN2.index.get_loc(CASRN)].tolist()
                self.VISWANATH_NATARAJAN_2_coeffs = [A - 4.605170185988092, B] # log(100) = 4.605170185988092
                T_limits[VISWANATH_NATARAJAN_2] = (self.VISWANATH_NATARAJAN_2_Tmin, self.VISWANATH_NATARAJAN_2_Tmax)
            if CASRN in viscosity.mu_data_VN2E.index:
                methods.append(VISWANATH_NATARAJAN_2E)
                C, D, self.VISWANATH_NATARAJAN_2E_Tmin, self.VISWANATH_NATARAJAN_2E_Tmax = viscosity.mu_values_VN2E[viscosity.mu_data_VN2E.index.get_loc(CASRN)].tolist()
                self.VISWANATH_NATARAJAN_2E_coeffs = [C, D]
                T_limits[VISWANATH_NATARAJAN_2E] = (self.VISWANATH_NATARAJAN_2E_Tmin, self.VISWANATH_NATARAJAN_2E_Tmax)
            if CASRN in viscosity.mu_data_Perrys_8E_2_313.index:
                methods.append(DIPPR_PERRY_8E)
                C1, C2, C3, C4, C5, self.Perrys2_313_Tmin, self.Perrys2_313_Tmax = viscosity.mu_values_Perrys_8E_2_313[viscosity.mu_data_Perrys_8E_2_313.index.get_loc(CASRN)].tolist()
                self.Perrys2_313_coeffs = [C1, C2, C3, C4, C5]
                T_limits[DIPPR_PERRY_8E] = (self.Perrys2_313_Tmin, self.Perrys2_313_Tmax)
            if CASRN in viscosity.mu_data_VDI_PPDS_7.index:
                methods.append(VDI_PPDS)
                # No temperature limits - ideally could use critical point
                self.VDI_PPDS_coeffs = VDI_PPDS_coeffs = viscosity.mu_values_PPDS_7[viscosity.mu_data_VDI_PPDS_7.index.get_loc(CASRN)].tolist()
                low = low_orig = min(self.VDI_PPDS_coeffs[2], self.VDI_PPDS_coeffs[3])# + 5.0
                high = high_orig = max(self.VDI_PPDS_coeffs[2], self.VDI_PPDS_coeffs[3])# - 5.0
                if low > 0.0:
                    dmu_low_under, mu_low_under = dPPDS9_dT(low*0.9995, *VDI_PPDS_coeffs)
                    dmu_low_above, mu_low_above = dPPDS9_dT(low*1.0005, *VDI_PPDS_coeffs)
                if high > 0.0:
                    dmu_high_under, mu_high_under = dPPDS9_dT(high*0.9995, *VDI_PPDS_coeffs)
                    dmu_high_above, mu_high_above = dPPDS9_dT(high*1.0005, *VDI_PPDS_coeffs)
                if self.Tm is not None:
                    dmu_Tm, mu_Tm = dPPDS9_dT(self.Tm, *VDI_PPDS_coeffs)
                if self.Tc is not None:
                    dmu_Tc_under, mu_Tc_under = dPPDS9_dT(self.Tc, *VDI_PPDS_coeffs)


                if high > 0.0 and low < 0.0 or isinf(dmu_low_under) or isinf(dmu_low_above):
                    # high + a few K as lower limit
                    low = 0.1*high
                    high = high-1.0
                else:
                    low, high = low + 5.0, high + 5.0
                if self.Tm is not None:
                    low = self.Tm
                if self.Tc is not None:
                    if dmu_Tc_under < 0.0:
                        high = self.Tc
                if self.Tm is not None and self.Tc is not None and low_orig < 0 and self.Tm < high_orig < self.Tc and dmu_Tc_under < 0.0:
                    low = high_orig + 1.0
                if high == high_orig:
                    high -= 1.0

                dmu_low, mu_low = dPPDS9_dT(low, *VDI_PPDS_coeffs)
                dmu_high, mu_high = dPPDS9_dT(high, *VDI_PPDS_coeffs)
                if dmu_low*dmu_high < 0.0:
                    def to_solve(T):
                        return dPPDS9_dT(T, *VDI_PPDS_coeffs)[0]
                    T_switch = brenth(to_solve, low, high)
                    if dmu_high > 0.0:
                        high = T_switch
                    else:
                        low = T_switch

                T_limits[VDI_PPDS] = (low, high)

        if all((self.MW, self.Tc, self.Pc, self.omega)):
            methods.append(LETSOU_STIEL)
            T_limits[LETSOU_STIEL] = (0.25*self.Tc, self.Tc)
        if all((self.MW, self.Tm, self.Tc, self.Pc, self.Vc, self.omega, self.Vml)):
            methods.append(PRZEDZIECKI_SRIDHAR)
            T_limits[PRZEDZIECKI_SRIDHAR] = (self.Tm, self.Tc)
        if all([self.Tc, self.Pc, self.omega]):
            methods_P.append(LUCAS)
        self.all_methods.update(methods)
        self.all_methods_P = set(methods_P)

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {COOLPROP : [CAS for CAS in coolprop_dict if (coolprop_fluids[CAS].has_mu and CAS not in CoolProp_failing_PT_flashes)],
                VDI_TABULAR: list(miscdata.VDI_saturation_dict.keys()),
                DUTT_PRASAD: viscosity.mu_data_Dutt_Prasad.index,
                VISWANATH_NATARAJAN_3: viscosity.mu_data_VN3.index,
                VISWANATH_NATARAJAN_2: viscosity.mu_data_VN2.index,
                VISWANATH_NATARAJAN_2E: viscosity.mu_data_VN2E.index,
                DIPPR_PERRY_8E: viscosity.mu_data_Perrys_8E_2_313.index,
                VDI_PPDS: viscosity.mu_data_VDI_PPDS_7.index,
                }


    def calculate(self, T, method):
        r'''Method to calculate low-pressure liquid viscosity at tempearture
        `T` with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate viscosity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        mu : float
            Viscosity of the liquid at T and a low pressure, [Pa*s]
        '''
        if method == DUTT_PRASAD:
            A, B, C = self.DUTT_PRASAD_coeffs
            mu = Viswanath_Natarajan_3(T, A, B, C, )
        elif method == VISWANATH_NATARAJAN_3:
            A, B, C = self.VISWANATH_NATARAJAN_3_coeffs
            mu = Viswanath_Natarajan_3(T, A, B, C)
        elif method == VISWANATH_NATARAJAN_2:
            A, B = self.VISWANATH_NATARAJAN_2_coeffs
            mu = Viswanath_Natarajan_2(T, self.VISWANATH_NATARAJAN_2_coeffs[0], self.VISWANATH_NATARAJAN_2_coeffs[1])
        elif method == VISWANATH_NATARAJAN_2E:
            C, D = self.VISWANATH_NATARAJAN_2E_coeffs
            mu = Viswanath_Natarajan_2_exponential(T, C, D)
        elif method == DIPPR_PERRY_8E:
            mu = EQ101(T, *self.Perrys2_313_coeffs)
        elif method == COOLPROP:
            mu = CoolProp_T_dependent_property(T, self.CASRN, 'V', 'l')
        elif method == JOBACK:
            A, B = self.joback_coeffs
            mu = self.MW*exp(A/T + B)
        elif method == LETSOU_STIEL:
            mu = Letsou_Stiel(T, self.MW, self.Tc, self.Pc, self.omega)
        elif method == PRZEDZIECKI_SRIDHAR:
            if type(self.Vml) is float:
                Vml = self.Vml
            elif type(self.Vml) is VolumeLiquid:
                Vml = self.Vml.T_dependent_property(T)
            else:
                Vml = self.Vml(T)
            mu = Przedziecki_Sridhar(T, self.Tm, self.Tc, self.Pc, self.Vc, Vml, self.omega, self.MW)
        elif method == VDI_PPDS:
            return PPDS9(T, *self.VDI_PPDS_coeffs)
        else:
            return self._base_calculate(T, method)
        return mu

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For CSP methods, the models
        are considered valid from 0 K to the critical point. For tabular data,
        extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the
        extrapolation is considered valid for all temperatures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        Parameters
        ----------
        T : float
            Temperature at which to test the method, [K]
        method : str
            Name of the method to test

        Returns
        -------
        validity : bool
            Whether or not a method is valid
        '''
        return super().test_method_validity(T, method)

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent liquid viscosity at
        temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate viscosity, [K]
        P : float
            Pressure at which to calculate viscosity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        mu : float
            Viscosity of the liquid at T and P, [Pa*s]
        '''
        if method == LUCAS:
            mu = self.T_dependent_property(T)
            Psat = self.Psat(T) if hasattr(self.Psat, '__call__') else self.Psat
            mu = Lucas(T, P, self.Tc, self.Pc, self.omega, Psat, mu)
        elif method == COOLPROP:
            mu = PropsSI('V', 'T', T, 'P', P, self.CASRN)
        else:
            return self._base_calculate_P(T, P, method)
        return mu

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a high-pressure method. For
        **COOLPROP**, the fluid must be both a liquid and under the maximum
        pressure of the fluid's EOS. **LUCAS** doesn't work on some occasions,
        due to something related to Tr and negative powers - but is otherwise
        considered correct for all circumstances.

        For tabular data, extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the
        extrapolation is considered valid for all temperatures and pressures.

        Parameters
        ----------
        T : float
            Temperature at which to test the method, [K]
        P : float
            Pressure at which to test the method, [Pa]
        method : str
            Name of the method to test

        Returns
        -------
        validity : bool
            Whether or not a method is valid
        '''
        if method == LUCAS:
            validity = True
        elif method == COOLPROP:
            validity = PhaseSI('T', T, 'P', P, self.CASRN) in ['liquid', 'supercritical_liquid']
        else:
            validity = super().test_method_validity_P(T, P, method)
        return validity

GHARAGHEIZI = 'GHARAGHEIZI'
YOON_THODOS = 'YOON_THODOS'
STIEL_THODOS = 'STIEL_THODOS'
LUCAS_GAS = 'LUCAS_GAS'

viscosity_gas_methods = [REFPROP_FIT, COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, VDI_TABULAR, GHARAGHEIZI, YOON_THODOS,
                         STIEL_THODOS, LUCAS_GAS]
"""Holds all low-pressure methods available for the ViscosityGas
class, for use in iterating over them."""
viscosity_gas_methods_P = [NEGLECT_P, COOLPROP]
"""Holds all high-pressure methods available for the ViscosityGas
class, for use in iterating over them."""


class ViscosityGas(TPDependentProperty):
    r'''Class for dealing with gas viscosity as a function of
    temperature and pressure.

    For gases at atmospheric pressure, there are 4 corresponding-states
    estimators, two sources of coefficient-based models, one source of tabular
    information, and the external library CoolProp.

    For gases under the fluid's boiling point (at sub-atmospheric pressures),
    and high-pressure gases above the boiling point, there are zero
    corresponding-states estimators, and the external library CoolProp.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical
    MW : float, optional
        Molecular weight, [g/mol]
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    Zc : float, optional
        Critical compressibility, [-]
    dipole : float, optional
        Dipole moment of the fluid, [debye]
    Vmg : float, optional
        Molar volume of the fluid at a pressure and temperature, [m^3/mol]
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    A string holding each method's name is assigned to the following variables
    in this module, intended as the most convenient way to refer to a method.
    To iterate over all methods, use the lists stored in
    :obj:`viscosity_gas_methods` and
    :obj:`viscosity_gas_methods_P` for low and high pressure
    methods respectively.

    Low pressure methods:

    **GHARAGHEIZI**:
        CSP method, described in :obj:`chemicals.viscosity.Gharagheizi_gas_viscosity`.
    **YOON_THODOS**:
        CSP method, described in :obj:`chemicals.viscosity.Yoon_Thodos`.
    **STIEL_THODOS**:
        CSP method, described in :obj:`chemicals.viscosity.Stiel_Thodos`.
    **LUCAS_GAS**:
        CSP method, described in :obj:`chemicals.viscosity.Lucas_gas`.
    **DIPPR_PERRY_8E**:
        A collection of 345 coefficient sets from the DIPPR database published
        openly in [3]_. Provides temperature limits for all its fluids.
        :obj:`chemicals.dippr.EQ102` is used for its fluids.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS, published
        openly in [2]_. Provides no temperature limits, but provides reasonable
        values at fairly high and very low temperatures.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow.
    **VDI_TABULAR**:
        Tabular data in [2]_ along the saturation curve; interpolation is as
        set by the user or the default.
    **REFPROP_FIT**:
        A series of higher-order polynomial fits to the calculated results from
        the equations implemented in REFPROP.

    High pressure methods:

    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow, but unparalled in accuracy for pressure
        dependence.


    A minimum viscosity value of 1e-5 Pa*s is set according to [4]_.
    This is also just above the lowest experimental values of viscosity of
    helium, 9.4e-6 Pa*s.

    See Also
    --------
    chemicals.viscosity.Gharagheizi_gas_viscosity
    chemicals.viscosity.Yoon_Thodos
    chemicals.viscosity.Stiel_Thodos
    chemicals.viscosity.Lucas_gas

    References
    ----------
    .. [1] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    .. [3] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [4] Trachenko, K., and V. V. Brazhkin. "Minimal Quantum Viscosity from
       Fundamental Physical Constants." Science Advances, April 2020.
       https://doi.org/10.1126/sciadv.aba3747.
    '''

    name = 'Gas viscosity'
    units = 'Pa*s'
    interpolation_T = None
    """No interpolation transformation by default."""
    interpolation_P = None
    """No interpolation transformation by default."""
    interpolation_property = None
    """No interpolation transformation by default."""
    interpolation_property_inv = None
    """No interpolation transformation by default."""
    tabular_extrapolation_permitted = True
    """Allow tabular extrapolation by default."""
    property_min = 0.0
    """Mimimum valid value of gas viscosity; limiting condition at low pressure
    is 0."""
    property_max = 1E-3
    """Maximum valid value of gas viscosity. Might be too high, or too low."""

    ranked_methods = [REFPROP_FIT, COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, VDI_TABULAR, GHARAGHEIZI, YOON_THODOS,
                      STIEL_THODOS, LUCAS_GAS]
    """Default rankings of the low-pressure methods."""
    ranked_methods_P = [NEGLECT_P, COOLPROP]
    """Default rankings of the high-pressure methods."""

    obj_references = pure_references = ('Vmg',)
    obj_references_types = pure_reference_types = (VolumeGas,)

    custom_args = ('MW', 'Tc', 'Pc', 'Zc', 'dipole', 'Vmg')
    DEFAULT_EXTRAPOLATION_MIN = 1e-5
    def __init__(self, CASRN='', MW=None, Tc=None, Pc=None, Zc=None,
                 dipole=None, Vmg=None, extrapolation='linear', extrapolation_min=DEFAULT_EXTRAPOLATION_MIN,
                 **kwargs):
        self.CASRN = CASRN
        self.MW = MW
        self.Tc = Tc
        self.Pc = Pc
        self.Zc = Zc
        self.dipole = dipole
        self.Vmg = Vmg
        if 'extrapolation_min' not in kwargs:
            kwargs['extrapolation_min'] = extrapolation_min
        super().__init__(extrapolation, **kwargs)

    def load_all_methods(self, load_data=True):
        r'''Method which picks out coefficients for the specified chemical
        from the various dictionaries and DataFrames storing it. All data is
        stored as attributes. This method also sets :obj:`Tmin`, :obj:`Tmax`,
        :obj:`all_methods` and obj:`all_methods_P` as a set of methods for
        which the data exists for.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods, methods_P = [], [NEGLECT_P]
        self.T_limits = T_limits = {}
        self.all_methods = set()
        CASRN = self.CASRN
        if load_data and CASRN:
            if CASRN in miscdata.VDI_saturation_dict:
                Ts, props = lookup_VDI_tabular_data(CASRN, 'Mu (g)')
                self.add_tabular_data(Ts, props, VDI_TABULAR, check_properties=False)
                del self._method
            if has_CoolProp() and CASRN in coolprop_dict:
                CP_f = coolprop_fluids[CASRN]
                if CP_f.has_mu:
                    self.CP_f = CP_f
                    methods.append(COOLPROP)
                    methods_P.append(COOLPROP)
#                    T_limits[COOLPROP] = (self.CP_f.Tmin, self.CP_f.Tmax)
                    T_limits[COOLPROP] = (self.CP_f.Tmin, self.CP_f.Tmax*.9999)
            if CASRN in viscosity.mu_data_Perrys_8E_2_312.index:
                methods.append(DIPPR_PERRY_8E)
                C1, C2, C3, C4, self.Perrys2_312_Tmin, self.Perrys2_312_Tmax = viscosity.mu_values_Perrys_8E_2_312[viscosity.mu_data_Perrys_8E_2_312.index.get_loc(CASRN)].tolist()
                self.Perrys2_312_coeffs = [C1, C2, C3, C4]
                T_limits[DIPPR_PERRY_8E] = (self.Perrys2_312_Tmin, self.Perrys2_312_Tmax)
            if CASRN in viscosity.mu_data_VDI_PPDS_8.index:
                methods.append(VDI_PPDS)
                self.VDI_PPDS_coeffs = viscosity.mu_values_PPDS_8[viscosity.mu_data_VDI_PPDS_8.index.get_loc(CASRN)].tolist()
                self.VDI_PPDS_coeffs.reverse() # in format for horner's scheme
                T_limits[VDI_PPDS] = (1e-3, 10000)
        if all([self.Tc, self.Pc, self.MW]):
            methods.append(GHARAGHEIZI)
            methods.append(YOON_THODOS)
            methods.append(STIEL_THODOS)
            T_limits[YOON_THODOS] = T_limits[STIEL_THODOS] = (1e-3, 5E3)
            T_limits[GHARAGHEIZI] = (20.0, 2e3)
            # GHARAGHEIZI turns nonsesical at ~15 K, YOON_THODOS fine to 0 K,
            # same as STIEL_THODOS
        if all([self.Tc, self.Pc, self.Zc, self.MW]):
            methods.append(LUCAS_GAS)
            T_limits[LUCAS_GAS] = (1e-3, 1E3)
        self.all_methods.update(methods)
        self.all_methods_P = set(methods_P)

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {COOLPROP : [CAS for CAS in coolprop_dict if (coolprop_fluids[CAS].has_mu and CAS not in CoolProp_failing_PT_flashes)],
                VDI_TABULAR: list(miscdata.VDI_saturation_dict.keys()),
                DIPPR_PERRY_8E: viscosity.mu_data_Perrys_8E_2_312.index,
                VDI_PPDS: viscosity.mu_data_VDI_PPDS_8.index,
                }


    def calculate(self, T, method):
        r'''Method to calculate low-pressure gas viscosity at
        tempearture `T` with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature of the gas, [K]
        method : str
            Name of the method to use

        Returns
        -------
        mu : float
            Viscosity of the gas at T and a low pressure, [Pa*s]
        '''
        if method == GHARAGHEIZI:
            mu = viscosity_gas_Gharagheizi(T, self.Tc, self.Pc, self.MW)
        elif method == COOLPROP:
            mu = CoolProp_T_dependent_property(T, self.CASRN, 'V', 'g')
        elif method == DIPPR_PERRY_8E:
            mu = EQ102(T, *self.Perrys2_312_coeffs)
        elif method == VDI_PPDS:
            mu = horner(self.VDI_PPDS_coeffs, T)
        elif method == YOON_THODOS:
            mu = Yoon_Thodos(T, self.Tc, self.Pc, self.MW)
        elif method == STIEL_THODOS:
            mu = Stiel_Thodos(T, self.Tc, self.Pc, self.MW)
        elif method == LUCAS_GAS:
            mu = Lucas_gas(T, self.Tc, self.Pc, self.Zc, self.MW, self.dipole, CASRN=self.CASRN)
        else:
            return self._base_calculate(T, method)

        return mu

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a temperature-dependent
        low-pressure method. For CSP most methods, the all methods are
        considered valid from 0 K up to 5000 K. For method **GHARAGHEIZI**,
        the method is considered valud from 20 K to 2000 K.

        For tabular data, extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the extrapolation
        is considered valid for all temperatures.


        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        Parameters
        ----------
        T : float
            Temperature at which to test the method, [K]
        method : str
            Name of the method to test

        Returns
        -------
        validity : bool
            Whether or not a method is valid
        '''
        return super().test_method_validity(T, method)

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent gas viscosity
        at temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate gas viscosity, [K]
        P : float
            Pressure at which to calculate gas viscosity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        mu : float
            Viscosity of the gas at T and P, [Pa*]
        '''
        if method == COOLPROP:
            mu = PropsSI('V', 'T', T, 'P', P, self.CASRN)
        else:
            return self._base_calculate_P(T, P, method)
        return mu

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a high-pressure method. For
        **COOLPROP**, the fluid must be both a gas and under the maximum
        pressure of the fluid's EOS. No other methods are implemented.

        For tabular data, extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the
        extrapolation is considered valid for all temperatures and pressures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        Parameters
        ----------
        T : float
            Temperature at which to test the method, [K]
        P : float
            Pressure at which to test the method, [Pa]
        method : str
            Name of the method to test

        Returns
        -------
        validity : bool
            Whether or not a method is valid
        '''
        validity = True
        if method == COOLPROP:
            validity = PhaseSI('T', T, 'P', P, self.CASRN) in ['gas', 'supercritical_gas', 'supercritical', 'supercritical_liquid']
        else:
            validity = super().test_method_validity_P(T, P, method)
        return validity


LALIBERTE_MU = 'Laliberte'

viscosity_liquid_mixture_methods = [LALIBERTE_MU, MIXING_LOG_MOLAR, MIXING_LOG_MASS, LINEAR]
"""Holds all mixing rules available for the :obj:`ViscosityLiquidMixture`
class, for use in iterating over them."""


class ViscosityLiquidMixture(MixtureProperty):
    '''Class for dealing with the viscosity of a liquid mixture as a
    function of temperature, pressure, and composition.
    Consists of one electrolyte-specific method, and logarithmic rules based
    on either mole fractions of mass fractions.

    Prefered method is :obj:`mixing_logarithmic <chemicals.utils.mixing_logarithmic>` with mole
    fractions, or **Laliberte** if the mixture is aqueous and has electrolytes.

    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture, [-]
    ViscosityLiquids : list[ViscosityLiquid], optional
        ViscosityLiquid objects created for all species in the mixture, [-]
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    correct_pressure_pure : bool, optional
        Whether to try to use the better pressure-corrected pure component
        models or to use only the T-only dependent pure species models, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`viscosity_liquid_mixture_methods`.

    **LALIBERTE_MU**:
        Electrolyte model equation with coefficients; see
        :obj:`thermo.electrochem.Laliberte_viscosity` for more details.
    **MIXING_LOG_MOLAR**:
        Logarithmic mole fraction mixing rule described in
        :obj:`chemicals.utils.mixing_logarithmic`.
    **MIXING_LOG_MASS**:
        Logarithmic mole fraction mixing rule described in
        :obj:`chemicals.utils.mixing_logarithmic`.
    **LINEAR**:
        Linear mole fraction mixing rule described in
        :obj:`mixing_simple <chemicals.utils.mixing_simple>`.

    See Also
    --------
    :obj:`thermo.electrochem.Laliberte_viscosity`

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''

    name = 'liquid viscosity'
    units = 'Pa*s'
    property_min = 0
    """Mimimum valid value of liquid viscosity."""
    property_max = 2E8
    """Maximum valid value of liquid viscosity. Generous limit, as
    the value is that of bitumen in a Pitch drop experiment."""

    ranked_methods = [LALIBERTE_MU, MIXING_LOG_MOLAR, MIXING_LOG_MASS, LINEAR]

    pure_references = ('ViscosityLiquids',)
    pure_reference_types = (ViscosityLiquid, )
    obj_references = ('ViscosityLiquids',)

    pure_constants = ('MWs', )
    custom_args = pure_constants

    def __init__(self, CASs=[], ViscosityLiquids=[], MWs=[], **kwargs):
        self.CASs = CASs
        self.ViscosityLiquids = ViscosityLiquids
        self.MWs = MWs
        super().__init__(**kwargs)

    def load_all_methods(self):
        r'''Method to initialize the object by precomputing any values which
        may be used repeatedly and by retrieving mixture-specific variables.
        All data are stored as attributes. This method also sets :obj:`Tmin`,
        :obj:`Tmax`, and :obj:`all_methods` as a set of methods which should
        work to calculate the property.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods = [MIXING_LOG_MOLAR, MIXING_LOG_MASS, LINEAR]
        if len(self.CASs) > 1 and '7732-18-5' in self.CASs:
            wCASs = [i for i in self.CASs if i != '7732-18-5']
            Laliberte_data = electrochem.Laliberte_data
            laliberte_incomplete = False

            v1s, v2s, v3s, v4s, v5s, v6s = [], [], [], [], [], []
            for CAS in wCASs:
                if CAS in Laliberte_data.index:
                    dat = Laliberte_data.loc[CAS].values
                    if isnan(dat[12]):
                        laliberte_incomplete = True
                        break
                    v1s.append(float(dat[12]))
                    v2s.append(float(dat[13]))
                    v3s.append(float(dat[14]))
                    v4s.append(float(dat[15]))
                    v5s.append(float(dat[16]))
                    v6s.append(float(dat[17]))
                else:
                    laliberte_incomplete = True
                    break
            if not laliberte_incomplete:
                methods.append(LALIBERTE_MU)
                self.wCASs = wCASs
                self.index_w = self.CASs.index('7732-18-5')
                self.Laliberte_v1s = v1s
                self.Laliberte_v2s = v2s
                self.Laliberte_v3s = v3s
                self.Laliberte_v4s = v4s
                self.Laliberte_v5s = v5s
                self.Laliberte_v6s = v6s



        self.all_methods = all_methods = set(methods)
        Tmins = [i.Tmin for i in self.ViscosityLiquids if i.Tmin]
        Tmaxs = [i.Tmax for i in self.ViscosityLiquids if i.Tmax]
        if Tmins:
            self.Tmin = max(Tmins)
        if Tmaxs:
            self.Tmax = max(Tmaxs)

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate viscosity of a liquid mixture at
        temperature `T`, pressure `P`, mole fractions `zs` and weight fractions
        `ws` with a given method.

        This method has no exception handling; see :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Name of the method to use

        Returns
        -------
        mu : float
            Viscosity of the liquid mixture, [Pa*s]
        '''
        if method == LALIBERTE_MU:
            ws = list(ws)
            ws.pop(self.index_w)
            return Laliberte_viscosity(T, ws, self.wCASs)
        return super().calculate(T, P, zs, ws, method)

    def test_method_validity(self, T, P, zs, ws, method):
        r'''Method to test the validity of a specified method for the given
        conditions. If **Laliberte** is applicable, all other methods are
        returned as inapplicable. Otherwise, there are no checks or strict
        ranges of validity.

        Parameters
        ----------
        T : float
            Temperature at which to check method validity, [K]
        P : float
            Pressure at which to check method validity, [Pa]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Method name to use

        Returns
        -------
        validity : bool
            Whether or not a specifid method is valid
        '''
        if LALIBERTE_MU in self.all_methods:
            # If everything is an electrolyte, accept only it as a method
            if method in self.all_methods:
                return method == LALIBERTE_MU
        if method in self.all_methods:
            return True
        return super().test_method_validity(T, P, zs, ws, method)

BROKAW = 'BROKAW'
HERNING_ZIPPERER = 'HERNING_ZIPPERER'
WILKE = 'WILKE'
viscosity_gas_mixture_methods = [BROKAW, HERNING_ZIPPERER, WILKE, LINEAR]
"""Holds all mixing rules available for the :obj:`ViscosityGasMixture`
class, for use in iterating over them."""


class ViscosityGasMixture(MixtureProperty):
    '''Class for dealing with the viscosity of a gas mixture as a
    function of temperature, pressure, and composition.
    Consists of three gas viscosity specific mixing rules and a mole-weighted
    simple mixing rule.

    Prefered method is :obj:`Brokaw <chemicals.viscosity.Brokaw>`.

    Parameters
    ----------
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    molecular_diameters : list[float], optional
        Lennard-Jones molecular diameters, [angstrom]
    Stockmayers : list[float], optional
        Lennard-Jones depth of potential-energy minimum over k
        or epsilon_k, [K]
    CASs : list[str], optional
        The CAS numbers of all species in the mixture, [-]
    ViscosityGases : list[ViscosityGas], optional
        ViscosityGas objects created for all species in the mixture, [-]
    correct_pressure_pure : bool, optional
        Whether to try to use the better pressure-corrected pure component
        models or to use only the T-only dependent pure species models, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`viscosity_liquid_mixture_methods`.

    **BROKAW**:
        Mixing rule described in :obj:`Brokaw <chemicals.viscosity.Brokaw>`.
    **HERNING_ZIPPERER**:
        Mixing rule described in :obj:`Herning_Zipperer <chemicals.viscosity.Herning_Zipperer>`.
    **WILKE**:
        Mixing rule described in :obj:`Wilke <chemicals.viscosity.Wilke>`.
    **LINEAR**:
        Mixing rule described in :obj:`mixing_simple <chemicals.utils.mixing_simple>`.

    See Also
    --------
    chemicals.viscosity.Brokaw
    chemicals.viscosity.Herning_Zipperer
    chemicals.viscosity.Wilke

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''

    name = 'gas viscosity'
    units = 'Pa*s'
    property_min = 0
    """Mimimum valid value of gas viscosity; limiting condition at low pressure
    is 0."""
    property_max = 1E-3
    """Maximum valid value of gas viscosity. Might be too high, or too low."""

    ranked_methods = [BROKAW, HERNING_ZIPPERER, LINEAR, WILKE]

    pure_references = ('ViscosityGases',)
    obj_references = ('ViscosityGases',)
    pure_reference_types = (ViscosityGas, )

    pure_constants = ('MWs', 'molecular_diameters', 'Stockmayers')
    custom_args = pure_constants

    non_json_attributes = TPDependentProperty.non_json_attributes + ['MW_roots', 'Wilke_t0s', 'Wilke_t1s', 'Wilke_t2s']

    def __init__(self, MWs=[], molecular_diameters=[], Stockmayers=[], CASs=[],
                 ViscosityGases=[], **kwargs):
        self.MWs = MWs
        self.molecular_diameters = molecular_diameters
        self.Stockmayers = Stockmayers
        self.CASs = CASs
        self.ViscosityGases = ViscosityGases
        self._set_Wilke_data()
        super().__init__(**kwargs)

    def _set_Wilke_data(self):
        MWs = self.MWs
        try:
            self.MW_roots = [sqrt(i) for i in MWs]
            self.Wilke_t0s, self.Wilke_t1s, self.Wilke_t2s = Wilke_prefactors(MWs)
        except:
            pass

    def _custom_from_json(self, *args):
        self._set_Wilke_data()
        super()._custom_from_json(*args)


    def load_all_methods(self):
        r'''Method to initialize the object by precomputing any values which
        may be used repeatedly and by retrieving mixture-specific variables.
        All data are stored as attributes. This method also sets :obj:`Tmin`,
        :obj:`Tmax`, and :obj:`all_methods` as a set of methods which should
        work to calculate the property.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods = [LINEAR]
        if none_and_length_check((self.MWs, self.molecular_diameters, self.Stockmayers)):
            methods.append(BROKAW)
        if none_and_length_check([self.MWs]):
            methods.extend([WILKE, HERNING_ZIPPERER])

        self.all_methods = all_methods = set(methods)
        Tmins = [i.Tmin for i in self.ViscosityGases if i.Tmin]
        Tmaxs = [i.Tmax for i in self.ViscosityGases if i.Tmax]
        if Tmins:
            self.Tmin = max(Tmins)
        if Tmaxs:
            self.Tmax = max(Tmaxs)

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate viscosity of a gas mixture at
        temperature `T`, pressure `P`, mole fractions `zs` and weight fractions
        `ws` with a given method.

        This method has no exception handling; see :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Name of the method to use

        Returns
        -------
        mu : float
            Viscosity of gas mixture, [Pa*s]
        '''
        if method == HERNING_ZIPPERER:
            mus = self.calculate_pures_corrected(T, P, fallback=True)
            return Herning_Zipperer(zs, mus, None, self.MW_roots)
        elif method == WILKE:
            mus = self.calculate_pures_corrected(T, P, fallback=True)
            return Wilke_prefactored(zs, mus, self.Wilke_t0s, self.Wilke_t1s, self.Wilke_t2s)
        elif method == BROKAW:
            mus = self.calculate_pures_corrected(T, P, fallback=True)
            return Brokaw(T, zs, mus, self.MWs, self.molecular_diameters, self.Stockmayers)
        return super().calculate(T, P, zs, ws, method)

    def test_method_validity(self, T, P, zs, ws, method):
        if method in self.all_methods:
            return True
        return super().test_method_validity(T, P, zs, ws, method)


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

This module contains implementations of :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
representing liquid, vapor, and solid volume. A variety of estimation
and data methods are available as included in the `chemicals` library.
Additionally liquid, vapor, and solid mixture volume predictor objects
are implemented subclassing :obj:`MixtureProperty <thermo.utils.MixtureProperty>`.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.


.. contents:: :local:

Pure Liquid Volume
==================
.. autoclass:: VolumeLiquid
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods,
              calculate_P, test_method_validity_P, ranked_methods_P
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: volume_liquid_methods
.. autodata:: volume_liquid_methods_P

Pure Gas Volume
===============
.. autoclass:: VolumeGas
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods,
              calculate_P, test_method_validity_P, ranked_methods_P
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: volume_gas_methods

Pure Solid Volume
=================
.. autoclass:: VolumeSolid
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: volume_solid_methods

Mixture Liquid Volume
=====================
.. autoclass:: VolumeLiquidMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: volume_liquid_mixture_methods

Mixture Gas Volume
==================
.. autoclass:: VolumeGasMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: volume_gas_mixture_methods

Mixture Solid Volume
====================
.. autoclass:: VolumeSolidMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: volume_solid_mixture_methods
'''


__all__ = [
'volume_liquid_methods', 'volume_liquid_methods_P', 'VolumeLiquid', 'VolumeSupercriticalLiquid',
 'volume_gas_methods', 'VolumeGas',
'volume_gas_mixture_methods', 'volume_solid_mixture_methods',
           'volume_solid_methods', 'VolumeSolid',
           'VolumeLiquidMixture', 'VolumeGasMixture', 'VolumeSolidMixture',
           'Tait_parameters_COSTALD']

from chemicals import miscdata, volume
from chemicals.dippr import EQ105, EQ116
from chemicals.miscdata import COMMON_CHEMISTRY, lookup_VDI_tabular_data
from chemicals.utils import mixing_simple, none_and_length_check, rho_to_Vm
from chemicals.virial import BVirial_Abbott, BVirial_Pitzer_Curl, BVirial_Tsonopoulos, BVirial_Tsonopoulos_extended
from chemicals.volume import (
    COSTALD,
    SNM0,
    Amgat,
    Bhirud_normal,
    Campbell_Thodos,
    COSTALD_compressed,
    COSTALD_mixture,
    CRC_inorganic,
    Goodman,
    Rackett,
    Rackett_mixture,
    Townsend_Hales,
    Yamada_Gunn,
    Yen_Woods_saturation,
    ideal_gas,
)
from fluids.numerics import exp, horner, horner_and_der2, isnan, linspace, np, polyder, quadratic_from_f_ders

from thermo import electrochem
from thermo.coolprop import CoolProp_T_dependent_property, PhaseSI, PropsSI, coolprop_dict, coolprop_fluids, has_CoolProp
from thermo.electrochem import Laliberte_density
from thermo.utils import (
    COOLPROP,
    DIPPR_PERRY_8E,
    EOS,
    HEOS_FIT,
    LINEAR,
    NEGLECT_P,
    VDI_PPDS,
    VDI_TABULAR,
    MixtureProperty,
    TDependentProperty,
    TPDependentProperty,
)
from thermo.vapor_pressure import VaporPressure


def Tait_parameters_COSTALD(Tc, Pc, omega, Tr_min=.27, Tr_max=.95):
    # Limits of any of their data for Tr
    a = -9.070217
    b = 62.45326
    d = -135.1102
    f = 4.79594
    g = 0.250047
    h = 1.14188
    j = 0.0861488
    k = 0.0344483
    e = exp(f + omega*(g + h*omega))
    C = j + k*omega

    Tc_inv = 1.0/Tc
    def B_fun(T):
        tau = 1.0 - T*Tc_inv
        tau13 = tau**(1.0/3.0)
        return Pc*(-1.0 + a*tau13 + b*tau13*tau13 + d*tau + e*tau*tau13)
    from fluids.optional.pychebfun import cheb_to_poly, chebfun

    fun = chebfun(B_fun, domain=[Tr_min*Tc, Tr_max*Tc], N=3)
    B_params = cheb_to_poly(fun)
    return B_params, [C]

MMSNM0 = 'MMSNM0'
MMSNM0FIT = 'MMSNM0FIT'
HTCOSTALD = 'HTCOSTALD'
HTCOSTALDFIT = 'HTCOSTALDFIT'
COSTALD_COMPRESSED = 'COSTALD_COMPRESSED'
RACKETT = 'RACKETT'
RACKETTFIT = 'RACKETTFIT'
YEN_WOODS_SAT = 'YEN_WOODS_SAT'
YAMADA_GUNN = 'YAMADA_GUNN'
BHIRUD_NORMAL = 'BHIRUD_NORMAL'
TOWNSEND_HALES = 'TOWNSEND_HALES'
CAMPBELL_THODOS = 'CAMPBELL_THODOS'


CRC_INORG_L = 'CRC_INORG_L'
CRC_INORG_L_CONST = 'CRC_INORG_L_CONST'

volume_liquid_methods = [HEOS_FIT, DIPPR_PERRY_8E, VDI_PPDS, COOLPROP, MMSNM0FIT, VDI_TABULAR,
                         HTCOSTALDFIT, RACKETTFIT, CRC_INORG_L,
                         CRC_INORG_L_CONST, COMMON_CHEMISTRY, MMSNM0, HTCOSTALD,
                         YEN_WOODS_SAT, RACKETT, YAMADA_GUNN,
                         BHIRUD_NORMAL, TOWNSEND_HALES, CAMPBELL_THODOS, EOS]
"""Holds all low-pressure methods available for the :obj:`VolumeLiquid` class, for use
in iterating over them."""

volume_liquid_methods_P = [COOLPROP, COSTALD_COMPRESSED, EOS, NEGLECT_P]
"""Holds all high-pressure methods available for the :obj:`VolumeLiquid` class, for
use in iterating over them."""


class VolumeLiquid(TPDependentProperty):
    r'''Class for dealing with liquid molar volume as a function of
    temperature and pressure.

    For low-pressure (at 1 atm while under the vapor pressure; along the
    saturation line otherwise) liquids, there are six coefficient-based methods
    from five data sources, one source of tabular information, one source of
    constant values, eight corresponding-states estimators, the external
    library CoolProp and the equation of state.

    For high-pressure liquids (also, <1 atm liquids), there is one
    corresponding-states estimator, and the external library CoolProp.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical
    MW : float, optional
        Molecular weight, [g/mol]
    Tb : float, optional
        Boiling point, [K]
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    Vc : float, optional
        Critical volume, [m^3/mol]
    Zc : float, optional
        Critical compressibility
    omega : float, optional
        Acentric factor, [-]
    dipole : float, optional
        Dipole, [debye]
    Psat : float or callable, optional
        Vapor pressure at a given temperature, or callable for the same [Pa]
    eos : object, optional
        Equation of State object after :obj:`thermo.eos.GCEOS`
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
    :obj:`volume_liquid_methods` and :obj:`volume_liquid_methods_P` for low
    and high pressure methods respectively.

    Low pressure methods:

    **DIPPR_PERRY_8E**:
        A simple polynomial as expressed in [1]_, with data available for
        344 fluids. Temperature limits are available for all fluids. Believed
        very accurate.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS (:obj:`EQ116 <chemicals.dippr.EQ116>` in
        terms of mass density), published
        openly in [3]_. Valid up to the critical temperature, and extrapolates
        to very low temperatures well.
    **MMSNM0FIT**:
        Uses a fit coefficient for better accuracy in the :obj:`SNM0 <chemicals.volume.SNM0>` method,
        Coefficients available for 73 fluids from [2]_. Valid to the critical
        point.
    **HTCOSTALDFIT**:
        A method with two fit coefficients to the :obj:`COSTALD <chemicals.volume.COSTALD>` method.
        Coefficients available for 192 fluids, from [3]_. Valid to the critical
        point.
    **RACKETTFIT**:
        The :obj:`Rackett <chemicals.volume.Rackett>` method, with a fit coefficient Z_RA. Data is
        available for 186 fluids, from [3]_. Valid to the critical point.
    **CRC_INORG_L**:
        Single-temperature coefficient linear model in terms of mass density
        for the density of inorganic liquids; converted to molar units
        internally. Data is available for 177 fluids normally valid over a
        narrow range above the melting point, from [4]_; described in
        :obj:`CRC_inorganic <chemicals.volume.CRC_inorganic>`.
    **MMSNM0**:
        CSP method, described in :obj:`SNM0 <chemicals.volume.SNM0>`.
    **HTCOSTALD**:
        CSP method, described in :obj:`COSTALD <chemicals.volume.COSTALD>`.
    **YEN_WOODS_SAT**:
        CSP method, described in :obj:`Yen_Woods_saturation <chemicals.volume.Yen_Woods_saturation>`.
    **RACKETT**:
        CSP method, described in :obj:`Rackett <chemicals.volume.Rackett>`.
    **YAMADA_GUNN**:
        CSP method, described in :obj:`Yamada_Gunn <chemicals.volume.Yamada_Gunn>`.
    **BHIRUD_NORMAL**:
        CSP method, described in :obj:`Bhirud_normal <chemicals.volume.Bhirud_normal>`.
    **TOWNSEND_HALES**:
        CSP method, described in :obj:`Townsend_Hales <chemicals.volume.Townsend_Hales>`.
    **CAMPBELL_THODOS**:
        CSP method, described in :obj:`Campbell_Thodos <chemicals.volume.Campbell_Thodos>`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [5]_. Very slow.
    **HEOS_FIT**:
        A series of higher-order polynomial fits to the calculated results from
        fundamental helmholtz equations of state as calculated with REFPROP
    **CRC_INORG_L_CONST**:
        Constant inorganic liquid densities, in [4]_.
    **VDI_TABULAR**:
        Tabular data in [6]_ along the saturation curve; interpolation is as
        set by the user or the default.
    **EOS**:
        Equation of state provided by user.

    High pressure methods:

    **COSTALD_COMPRESSED**:
        CSP method, described in :obj:`COSTALD_compressed <chemicals.volume.COSTALD_compressed>`. Calculates a
        low-pressure molar volume first, using :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [5]_. Very slow, but unparalled in accuracy for pressure
        dependence.
    **EOS**:
        Equation of state provided by user.

    See Also
    --------
    chemicals.volume.Yen_Woods_saturation
    chemicals.volume.Rackett
    chemicals.volume.Yamada_Gunn
    chemicals.volume.Townsend_Hales
    chemicals.volume.Bhirud_normal
    chemicals.volume.COSTALD
    chemicals.volume.Campbell_Thodos
    chemicals.volume.SNM0
    chemicals.volume.CRC_inorganic
    chemicals.volume.COSTALD_compressed

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       8E. McGraw-Hill Professional, 2007.
    .. [2] Mchaweh, A., A. Alsaygh, Kh. Nasrifar, and M. Moshfeghian.
       "A Simplified Method for Calculating Saturated Liquid Densities."
       Fluid Phase Equilibria 224, no. 2 (October 1, 2004): 157-67.
       doi:10.1016/j.fluid.2004.06.054
    .. [3] Hankinson, Risdon W., and George H. Thomson. "A New Correlation for
       Saturated Densities of Liquids and Their Mixtures." AIChE Journal
       25, no. 4 (1979): 653-663. doi:10.1002/aic.690250412
    .. [4] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    .. [5] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    .. [6] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''

    name = 'Liquid molar volume'
    units = 'm^3/mol'
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
    property_min = 0
    """Mimimum valid value of liquid molar volume. It should normally occur at the
    triple point, and be well above this."""
    property_max = 2e-3
    """Maximum valid value of liquid molar volume. Generous limit."""

    ranked_methods = [HEOS_FIT, DIPPR_PERRY_8E, VDI_PPDS, COOLPROP, MMSNM0FIT, VDI_TABULAR,
                      HTCOSTALDFIT, RACKETTFIT, CRC_INORG_L,
                      CRC_INORG_L_CONST, COMMON_CHEMISTRY, MMSNM0, HTCOSTALD,
                      YEN_WOODS_SAT, RACKETT, YAMADA_GUNN,
                      BHIRUD_NORMAL, TOWNSEND_HALES, CAMPBELL_THODOS, EOS]
    """Default rankings of the low-pressure methods."""

    ranked_methods_P = [COOLPROP, COSTALD_COMPRESSED, EOS, NEGLECT_P]
    """Default rankings of the high-pressure methods."""

    obj_references = ('eos', 'Psat')
    pure_references = ('Psat')
    obj_references_types = pure_reference_types = (VaporPressure,)


    custom_args = ('MW', 'Tb', 'Tc', 'Pc', 'Vc', 'Zc', 'omega', 'dipole',
                   'Psat', 'eos')
    def __init__(self, MW=None, Tb=None, Tc=None, Pc=None, Vc=None, Zc=None,
                 omega=None, dipole=None, Psat=None, CASRN='', eos=None,
                 has_hydroxyl=False, extrapolation='constant', **kwargs):
        self.CASRN = CASRN
        self.MW = MW
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.Vc = Vc
        self.Zc = Zc
        self.omega = omega
        self.dipole = dipole
        self.Psat = Psat
        self.eos = eos
        self.has_hydroxyl = has_hydroxyl
        super().__init__(extrapolation, **kwargs)

    def _custom_set_poly_fit(self):
        try:
            Tmin, Tmax = self.poly_fit_Tmin, self.poly_fit_Tmax
            poly_fit_coeffs = self.poly_fit_coeffs
            v_Tmin = horner(poly_fit_coeffs, Tmin)
            for T_trans in linspace(Tmin, Tmax, 25):
                # Create a new polynomial approximating the fit at T_trans;
                p = list(quadratic_from_f_ders(Tmin, *horner_and_der2(poly_fit_coeffs, T_trans)))
                # Evaluate the first and second derivative at Tmin
                v_Tmin_refit, d1_Tmin, d2_Tmin = horner_and_der2(p, Tmin)
                # If the first derivative is negative (volume liquid should always be posisitive except for water)
                # try a point higher up the curve
                if d1_Tmin < 0.0:
                    continue
                # If the second derivative would ever make the first derivative negative until
                # it reaches zero K, limit the second derivative; this introduces only 1 discontinuity
                if Tmin - d1_Tmin/d2_Tmin > 0.0:
                    # When this happens, note the middle `p` coefficient becomes zero - this is expected
                    d2_Tmin = d1_Tmin/Tmin
                self._Tmin_T_trans = T_trans
                p = list(quadratic_from_f_ders(Tmin, v_Tmin, d1_Tmin, d2_Tmin))
                self.poly_fit_Tmin_quadratic = p
                break

        except:
            pass


    def load_all_methods(self, load_data):
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
        self.T_limits = T_limits = {}
        self.all_methods = set()
        methods = []
        methods_P = [NEGLECT_P]
        CASRN = self.CASRN
        if load_data and CASRN:
            if has_CoolProp() and CASRN in coolprop_dict:
                methods.append(COOLPROP)
                methods_P.append(COOLPROP)
                self.CP_f = coolprop_fluids[CASRN]
                T_limits[COOLPROP] = (max(self.CP_f.Tt,self.CP_f.Tmin), self.CP_f.Tc)
            if CASRN in volume.rho_data_CRC_inorg_l.index:
                methods.append(CRC_INORG_L)
                self.CRC_INORG_L_MW, self.CRC_INORG_L_rho, self.CRC_INORG_L_k, self.CRC_INORG_L_Tm, self.CRC_INORG_L_Tmax = volume.rho_values_CRC_inorg_l[volume.rho_data_CRC_inorg_l.index.get_loc(CASRN)].tolist()
                T_limits[CRC_INORG_L] = (self.CRC_INORG_L_Tm, self.CRC_INORG_L_Tmax)
            if CASRN in volume.rho_data_Perry_8E_105_l.index:
                methods.append(DIPPR_PERRY_8E)
                C1, C2, C3, C4, self.DIPPR_Tmin, self.DIPPR_Tmax = volume.rho_values_Perry_8E_105_l[volume.rho_data_Perry_8E_105_l.index.get_loc(CASRN)].tolist()
                self.DIPPR_coeffs = [C1, C2, C3, C4]
                T_limits[DIPPR_PERRY_8E] = (self.DIPPR_Tmin, self.DIPPR_Tmax)
            if CASRN in volume.rho_data_VDI_PPDS_2.index:
                methods.append(VDI_PPDS)
                MW, Tc, rhoc, A, B, C, D = volume.rho_values_VDI_PPDS_2[volume.rho_data_VDI_PPDS_2.index.get_loc(CASRN)].tolist()
                self.VDI_PPDS_coeffs = [A, B, C, D]
                self.VDI_PPDS_MW = MW
                self.VDI_PPDS_Tc = Tc
                self.VDI_PPDS_rhoc = rhoc
                T_limits[VDI_PPDS] = (0.3*self.VDI_PPDS_Tc, self.VDI_PPDS_Tc)
            if CASRN in miscdata.VDI_saturation_dict:
                Ts, props = lookup_VDI_tabular_data(CASRN, 'Volume (l)')
                self.add_tabular_data(Ts, props, VDI_TABULAR, check_properties=False)
                del self._method
            if self.Tc and CASRN in volume.rho_data_COSTALD.index:
                methods.append(HTCOSTALDFIT)
                self.COSTALD_Vchar = float(volume.rho_data_COSTALD.at[CASRN, 'Vchar'])
                self.COSTALD_omega_SRK = float(volume.rho_data_COSTALD.at[CASRN, 'omega_SRK'])
                T_limits[HTCOSTALDFIT] = (0.0, self.Tc)
            if self.Tc and self.Pc and CASRN in volume.rho_data_COSTALD.index and not isnan(volume.rho_data_COSTALD.at[CASRN, 'Z_RA']):
                methods.append(RACKETTFIT)
                self.RACKETT_Z_RA = float(volume.rho_data_COSTALD.at[CASRN, 'Z_RA'])
                T_limits[RACKETTFIT] = (0.0, self.Tc)
            if CASRN in volume.rho_data_CRC_inorg_l_const.index:
                methods.append(CRC_INORG_L_CONST)
                self.CRC_INORG_L_CONST_Vm = float(volume.rho_data_CRC_inorg_l_const.at[CASRN, 'Vm'])
                T_limits[CRC_INORG_L_CONST] = (298.15, 298.15)
                # Roughly data at STP; not guaranteed however; not used for Trange
        if all((self.Tc, self.Vc, self.Zc)):
            methods.append(YEN_WOODS_SAT)
            T_limits[YEN_WOODS_SAT] = (0.0, self.Tc)
        if all((self.Tc, self.Pc, self.Zc)):
            methods.append(RACKETT)
            T_limits[RACKETT] = (0.0, self.Tc)
        if all((self.Tc, self.Pc, self.omega)):
            methods.append(YAMADA_GUNN)
            methods.append(BHIRUD_NORMAL)
            # Has bad interpolation behavior lower than roughly this
            T_limits[YAMADA_GUNN] = T_limits[BHIRUD_NORMAL] = (0.35*self.Tc, self.Tc)
        if all((self.Tc, self.Vc, self.omega)):
            methods.append(TOWNSEND_HALES)
            methods.append(HTCOSTALD)
            methods.append(MMSNM0)
            if load_data and CASRN and CASRN in volume.rho_data_SNM0.index:
                methods.append(MMSNM0FIT)
                self.SNM0_delta_SRK = float(volume.rho_data_SNM0.at[CASRN, 'delta_SRK'])
                T_limits[MMSNM0FIT] = (0.0, self.Tc)
            T_limits[TOWNSEND_HALES] = T_limits[HTCOSTALD] = T_limits[MMSNM0] = (0.0, self.Tc)
        if all((self.Tc, self.Vc, self.omega, self.Tb, self.MW)):
            methods.append(CAMPBELL_THODOS)
            T_limits[CAMPBELL_THODOS] = (0.0, self.Tc)
        if self.eos:
            try:
                T_limits[EOS] = (0.2*self.eos[0].Tc, self.eos[0].Tc)
                methods.append(EOS)
            except:
                pass
        if all((self.Tc, self.Pc, self.omega)):
            methods_P.append(COSTALD_COMPRESSED)
            if self.eos:
                methods_P.append(EOS)

        self.all_methods.update(methods)
        self.all_methods_P = set(methods_P)
        for m in self.ranked_methods_P:
            if m in self.all_methods_P:
                self.method_P = m
                break
        if not hasattr(self, '_method_P'):
            self._method_P = None

    def calculate(self, T, method):
        r'''Method to calculate low-pressure liquid molar volume at tempearture
        `T` with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate molar volume, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Vm : float
            Molar volume of the liquid at T and a low pressure, [m^3/mol]
        '''
        if method == RACKETT:
            Vm = Rackett(T, self.Tc, self.Pc, self.Zc)
        elif method == YAMADA_GUNN:
            Vm = Yamada_Gunn(T, self.Tc, self.Pc, self.omega)
        elif method == BHIRUD_NORMAL:
            Vm = Bhirud_normal(T, self.Tc, self.Pc, self.omega)
        elif method == TOWNSEND_HALES:
            Vm = Townsend_Hales(T, self.Tc, self.Vc, self.omega)
        elif method == HTCOSTALD:
            Vm = COSTALD(T, self.Tc, self.Vc, self.omega)
        elif method == YEN_WOODS_SAT:
            Vm = Yen_Woods_saturation(T, self.Tc, self.Vc, self.Zc)
        elif method == MMSNM0:
            Vm = SNM0(T, self.Tc, self.Vc, self.omega)
        elif method == MMSNM0FIT:
            Vm = SNM0(T, self.Tc, self.Vc, self.omega, self.SNM0_delta_SRK)
        elif method == CAMPBELL_THODOS:
            Vm = Campbell_Thodos(T, self.Tb, self.Tc, self.Pc, self.MW, self.dipole, self.has_hydroxyl)
        elif method == HTCOSTALDFIT:
            Vm = COSTALD(T, self.Tc, self.COSTALD_Vchar, self.COSTALD_omega_SRK)
        elif method == RACKETTFIT:
            Vm = Rackett(T, self.Tc, self.Pc, self.RACKETT_Z_RA)
        elif method == DIPPR_PERRY_8E:
            A, B, C, D = self.DIPPR_coeffs
            Vm = 1./EQ105(T, A, B, C, D)
        elif method == CRC_INORG_L:
            rho = CRC_inorganic(T, self.CRC_INORG_L_rho, self.CRC_INORG_L_k, self.CRC_INORG_L_Tm)
            Vm = rho_to_Vm(rho, self.CRC_INORG_L_MW)
        elif method == VDI_PPDS:
            A, B, C, D = self.VDI_PPDS_coeffs
            rho = EQ116(T, self.VDI_PPDS_Tc, self.VDI_PPDS_rhoc, A, B, C, D)
            Vm = rho_to_Vm(rho, self.VDI_PPDS_MW)
        elif method == CRC_INORG_L_CONST:
            Vm = self.CRC_INORG_L_CONST_Vm
        elif method == COOLPROP:
            Vm = 1./CoolProp_T_dependent_property(T, self.CASRN, 'DMOLAR', 'l')
        elif method == EOS:
            Vm = self.eos[0].V_l_sat(T)
        else:
            return self._base_calculate(T, method)
        return Vm

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent liquid molar volume at
        temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate molar volume, [K]
        P : float
            Pressure at which to calculate molar volume, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Vm : float
            Molar volume of the liquid at T and P, [m^3/mol]
        '''
        if method == COSTALD_COMPRESSED:
            Vm = self.T_dependent_property(T)
            Psat = self.Psat(T) if callable(self.Psat) else self.Psat
            if P > Psat:
                Vm = COSTALD_compressed(T, P, Psat, self.Tc, self.Pc, self.omega, Vm)
        elif method == COOLPROP:
#            assert PhaseSI('T', T, 'P', P, self.CASRN) == 'liquid'
            Vm = 1./PropsSI('DMOLAR', 'T', T, 'P', P, self.CASRN)
        elif method == EOS:
            self.eos[0] = self.eos[0].to_TP(T=T, P=P)
            Vm = self.eos[0].V_l
        else:
            return self._base_calculate_P(T, P, method)
        return Vm

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For CSP methods, the models
        are considered valid from 0 K to the critical point. For tabular data,
        extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the extrapolation
        is considered valid for all temperatures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        **BHIRUD_NORMAL** behaves poorly at low temperatures and is not used
        under 0.35Tc. The constant value available for inorganic chemicals,
        from method **CRC_INORG_L_CONST**, is considered valid for all
        temperatures.

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
        validity = True
        if method in (RACKETT, YAMADA_GUNN, TOWNSEND_HALES,
                        HTCOSTALD, YEN_WOODS_SAT, MMSNM0, MMSNM0FIT,
                        CAMPBELL_THODOS, HTCOSTALDFIT, RACKETTFIT):
            if T >= self.Tc:
                validity = False
        elif method == EOS:
            if T >= self.eos[0].Tc:
                validity = False
        else:
            return super().test_method_validity(T, method)
        return validity

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a high-pressure method. For
        **COOLPROP**, the fluid must be both a liquid and under the maximum
        pressure of the fluid's EOS. **COSTALD_COMPRESSED** is considered
        valid for all values of temperature and pressure. However, it very
        often will not actually work, due to the form of the polynomial in
        terms of Tr, the result of which is raised to a negative power.
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
        if method == COSTALD_COMPRESSED:
            pass
        elif method == COOLPROP:
            validity = PhaseSI('T', T, 'P', P, self.CASRN) in ('liquid', 'supercritical_liquid')
        elif method == EOS:
            self.eos[0] = self.eos[0].to_TP(T=T, P=P)
            validity = hasattr(self.eos[0], 'V_l')
        else:
            return super().test_method_validity_P(T, P, method)
        return validity


    def Tait_data(self):
        Tr_min = .27
        Tr_max = .95
        Tc = self.Tc

        Tmin, Tmax = Tc*Tr_min, Tc*Tr_max
        B_coeffs_COSTALD, C_coeffs_COSTALD = Tait_parameters_COSTALD(Tc, self.Pc, self.omega,
                                                     Tr_min=Tr_min,
                                                     Tr_max=Tr_max)
        if not hasattr(self, 'B_coeffs'):
            B_coeffs = B_coeffs_COSTALD
        else:
            B_coeffs = self.B_coeffs

        if not hasattr(self, 'C_coeffs'):
            C_coeffs = C_coeffs_COSTALD
        else:
            C_coeffs = self.C_coeffs

        B_coeffs_d = polyder(B_coeffs[::-1])[::-1]
        B_coeffs_d2 = polyder(B_coeffs_d[::-1])[::-1]

        B_data = [Tmin, horner(B_coeffs_d, Tmin), horner(B_coeffs, Tmin),
                  Tmax, horner(B_coeffs_d, Tmax), horner(B_coeffs, Tmax),
                  B_coeffs, B_coeffs_d, B_coeffs_d2]

        C_coeffs_d = polyder(C_coeffs[::-1])[::-1]
        C_coeffs_d2 = polyder(C_coeffs_d[::-1])[::-1]

        C_data = [Tmin, horner(C_coeffs_d, Tmin), horner(C_coeffs, Tmin),
                  Tmax, horner(C_coeffs_d, Tmax), horner(C_coeffs, Tmax),
                  C_coeffs, C_coeffs_d, C_coeffs_d2]

        return B_data, C_data

volume_supercritical_liquid_methods = []
"""Holds all low-pressure methods available for the :obj:`VolumeSupercriticalLiquid` class, for use
in iterating over them."""

volume_supercritical_liquid_methods_P = [COOLPROP, EOS]
"""Holds all high-pressure methods available for the :obj:`VolumeSupercriticalLiquid` class, for
use in iterating over them."""


class VolumeSupercriticalLiquid(VolumeLiquid):
    r'''Class for dealing with a supercritical liquid-like fluid's  molar
    volume as a function of temperature and pressure.

    Only EOSs and CoolProp are supported here.

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
    omega : float, optional
        Acentric factor, [-]
    Psat : float or callable, optional
        Vapor pressure at a given temperature, or callable for the same [Pa]
    eos : object, optional
        Equation of State object after :obj:`thermo.eos.GCEOS`

    Notes
    -----
    A string holding each method's name is assigned to the following variables
    in this module, intended as the most convenient way to refer to a method.
    To iterate over all methods, use the lists stored in
    :obj:`volume_supercritical_liquid_methods` and :obj:`volume_supercritical_liquid_methods_P` for low
    and high pressure methods respectively.

    There are no low pressure methods implemented by design; the supercritical
    liquid state has no fixed definition so volume is always a function of
    pressure as well as temperature.

    High pressure methods:

    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow, but unparalled in accuracy for pressure
        dependence.
    **EOS**:
        Equation of state provided by user.

    References
    ----------
    .. [1] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    '''

    def __init__(self, MW=None, Tc=None, Pc=None,
                 omega=None,  Psat=None, CASRN='', eos=None,
                 extrapolation=None):
        self.CASRN = CASRN
        self.MW = MW
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.Psat = Psat
        self.eos = eos

        self.tabular_data = {}
        """tabular_data, dict: Stored (Ts, properties) for any
        tabular data; indexed by provided or autogenerated name."""
        self.tabular_data_interpolators = {}
        """tabular_data_interpolators, dict: Stored (extrapolator,
        spline) tuples which are interp1d instances for each set of tabular
        data; indexed by tuple of (name, interpolation_T,
        interpolation_property, interpolation_property_inv) to ensure that
        if an interpolation transform is altered, the old interpolator which
        had been created is no longer used."""

        self.tabular_data_P = {}
        """tabular_data_P, dict: Stored (Ts, Ps, properties) for any
        tabular data; indexed by provided or autogenerated name."""
        self.tabular_data_interpolators_P = {}
        """tabular_data_interpolators_P, dict: Stored (extrapolator,
        spline) tuples which are interp2d instances for each set of tabular
        data; indexed by tuple of (name, interpolation_T, interpolation_P,
        interpolation_property, interpolation_property_inv) to ensure that
        if an interpolation transform is altered, the old interpolator which
        had been created is no longer used."""

        self.load_all_methods()
        self.extrapolation = extrapolation


    def load_all_methods(self):
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
        methods = []
        methods_P = []
        self.T_limits = T_limits = {}
        if has_CoolProp() and self.CASRN in coolprop_dict:
            methods_P.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
            T_limits[COOLPROP] = (self.CP_f.Tc, self.CP_f.Tmax)
        if all((self.Tc, self.Pc, self.omega)):
            if self.eos:
                methods_P.append(EOS)
                T_limits[EOS] = (self.Tc, self.Tc*100)
        self.all_methods = set(methods)
        self.all_methods_P = set(methods_P)

    def calculate(self, T, method):
        r'''Method to calculate low-pressure liquid molar volume at tempearture
        `T` with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate molar volume, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Vm : float
            Molar volume of the liquid at T and a supercritical pressure, [m^3/mol]
        '''
        return self._base_calculate(T, method)

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent liquid molar volume at
        temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate molar volume, [K]
        P : float
            Pressure at which to calculate molar volume, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Vm : float
            Molar volume of the supercritical liquid at T and P, [m^3/mol]
        '''
        if method == COOLPROP:
            Vm = 1./PropsSI('DMOLAR', 'T', T, 'P', P, self.CASRN)
        elif method == EOS:
            self.eos[0] = self.eos[0].to_TP(T=T, P=P)
            try:
                Vm = self.eos[0].V_l
            except AttributeError:
                Vm =  self.eos[0].V_g
        elif method in self.tabular_data:
            Vm = self.interpolate_P(T, P, method)
        return Vm

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method.For tabular data,
        extrapolation outside of the range is used if
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

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a high-pressure method. For
        **COOLPROP**, the fluid must be both a liquid and under the maximum
        pressure of the fluid's EOS.
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
            validity = PhaseSI('T', T, 'P', P, self.CASRN) in ('liquid', 'supercritical', 'supercritical_gas', 'supercritical_liquid')
        elif method == EOS:
            self.eos[0] = self.eos[0].to_TP(T=T, P=P)
            validity = hasattr(self.eos[0], 'V_l')
        else:
            return super().test_method_validity_P(T, P, method)
        return validity

LALIBERTE = 'LALIBERTE'
COSTALD_MIXTURE = 'COSTALD_MIXTURE'
COSTALD_MIXTURE_FIT = 'COSTALD_MIXTURE_FIT'
RACKETT = 'RACKETT'
RACKETT_PARAMETERS = 'RACKETT_PARAMETERS'
volume_liquid_mixture_methods = [LALIBERTE, LINEAR, COSTALD_MIXTURE_FIT, RACKETT_PARAMETERS, COSTALD, RACKETT]
"""Holds all low-pressure methods available for the :obj:`VolumeLiquidMixture` class, for use
in iterating over them."""

volume_liquid_mixture_P_methods = [COSTALD]
"""Holds all high-pressure methods available for the :obj:`VolumeLiquidMixture` class, for use
in iterating over them."""

class VolumeLiquidMixture(MixtureProperty):
    '''Class for dealing with the molar volume of a liquid mixture as a
    function of temperature, pressure, and composition.
    Consists of one electrolyte-specific method, four corresponding states
    methods which do not use pure-component volumes, and one mole-weighted
    averaging method.

    Prefered method is **LINEAR**, or **LALIBERTE** if the mixture is aqueous
    and has electrolytes.

    Parameters
    ----------
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    Tcs : list[float], optional
        Critical temperatures of all species in the mixture, [K]
    Pcs : list[float], optional
        Critical pressures of all species in the mixture, [Pa]
    Vcs : list[float], optional
        Critical molar volumes of all species in the mixture, [m^3/mol]
    Zcs : list[float], optional
        Critical compressibility factors of all species in the mixture, [Pa]
    omegas : list[float], optional
        Accentric factors of all species in the mixture, [-]
    CASs : list[str], optional
        The CAS numbers of all species in the mixture, [-]
    VolumeLiquids : list[VolumeLiquid], optional
        VolumeLiquid objects created for all species in the mixture, [-]
    correct_pressure_pure : bool, optional
        Whether to try to use the better pressure-corrected pure component
        models or to use only the T-only dependent pure species models, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`volume_liquid_mixture_methods`.

    **LALIBERTE**:
        Aqueous electrolyte model equation with coefficients; see
        :obj:`thermo.electrochem.Laliberte_density` for more details.
    **COSTALD_MIXTURE**:
        CSP method described in :obj:`COSTALD_mixture <chemicals.volume.COSTALD_mixture>`.
    **COSTALD_MIXTURE_FIT**:
        CSP method described in :obj:`COSTALD_mixture <chemicals.volume.COSTALD_mixture>`, with two mixture
        composition independent fit coefficients, `Vc` and `omega`.
    **RACKETT**:
        CSP method described in :obj:`Rackett_mixture <chemicals.volume.Rackett_mixture>`.
    **RACKETT_PARAMETERS**:
        CSP method described in :obj:`Rackett_mixture <chemicals.volume.Rackett_mixture>`, but with a mixture
        independent fit coefficient for compressibility factor for each species.
    **LINEAR**:
        Linear mole fraction mixing rule described in
        :obj:`mixing_simple <chemicals.utils.mixing_simple>`; also known as Amgat's law.

    See Also
    --------


    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''

    name = 'Liquid volume'
    units = 'm^3/mol'
    property_min = 0
    """Mimimum valid value of liquid molar volume. It should normally occur at the
    triple point, and be well above this."""
    property_max = 2e-3
    """Maximum valid value of liquid molar volume. Generous limit."""

    ranked_methods = [LALIBERTE, LINEAR, COSTALD_MIXTURE_FIT,
                      RACKETT_PARAMETERS, COSTALD_MIXTURE, RACKETT]

    pure_references = ('VolumeLiquids',)
    pure_reference_types = (VolumeLiquid, )
    obj_references = ('VolumeLiquids',)

    pure_constants = ('MWs', 'Tcs', 'Pcs', 'Vcs', 'Zcs', 'omegas')
    custom_args = pure_constants

    def __init__(self, MWs=[], Tcs=[], Pcs=[], Vcs=[], Zcs=[], omegas=[],
                 CASs=[], VolumeLiquids=[], **kwargs):
        self.MWs = MWs
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.Vcs = Vcs
        self.Zcs = Zcs
        self.omegas = omegas
        self.CASs = CASs
        self.VolumeLiquids = VolumeLiquids
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
        methods = [LINEAR]

        if none_and_length_check([self.Tcs, self.Vcs, self.omegas]):
            methods.append(COSTALD_MIXTURE)
            if none_and_length_check([self.Tcs, self.CASs]) and all(i in volume.rho_data_COSTALD.index for i in self.CASs):
                self.COSTALD_Vchars = [float(volume.rho_data_COSTALD.at[CAS, 'Vchar']) for CAS in self.CASs]
                self.COSTALD_omegas = [float(volume.rho_data_COSTALD.at[CAS, 'omega_SRK']) for CAS in self.CASs]
                methods.append(COSTALD_MIXTURE_FIT)

        if none_and_length_check([self.MWs, self.Tcs, self.Pcs, self.Zcs]):
            methods.append(RACKETT)
            if none_and_length_check([self.Tcs, self.CASs]) and all(CAS in volume.rho_data_COSTALD.index for CAS in self.CASs):
                Z_RAs = [float(volume.rho_data_COSTALD.at[CAS, 'Z_RA']) for CAS in self.CASs]
                if not any(np.isnan(Z_RAs)):
                    self.Z_RAs = Z_RAs
                    methods.append(RACKETT_PARAMETERS)

        if len(self.CASs) > 1 and '7732-18-5' in self.CASs:
            Laliberte_data = electrochem.Laliberte_data
            v1s, v2s, v3s, v4s, v5s, v6s = [], [], [], [], [], []
            laliberte_incomplete = False
            for CAS in self.CASs:
                if CAS == '7732-18-5':
                    continue
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
                self.Laliberte_v1s = v1s
                self.Laliberte_v2s = v2s
                self.Laliberte_v3s = v3s
                self.Laliberte_v4s = v4s
                self.Laliberte_v5s = v5s
                self.Laliberte_v6s = v6s
                wCASs = [i for i in self.CASs if i != '7732-18-5']
                methods.append(LALIBERTE)
                self.wCASs = wCASs
                self.index_w = self.CASs.index('7732-18-5')

        self.all_methods = all_methods = set(methods)

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate molar volume of a liquid mixture at
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
        Vm : float
            Molar volume of the liquid mixture at the given conditions,
            [m^3/mol]
        '''
        if method == LINEAR:
            if self._correct_pressure_pure:
                Vms = []
                for obj in self.VolumeLiquids:
                    Vm = obj.TP_dependent_property(T, P)
                    if Vm is None:
                        Vm = obj.T_dependent_property(T)
                    Vms.append(Vm)
            else:
                Vms = [i.T_dependent_property(T) for i in self.VolumeLiquids]
            return Amgat(zs, Vms)
        elif method == LALIBERTE:
            ws = list(ws)
            ws.pop(self.index_w)
            rho = Laliberte_density(T, ws, self.wCASs)
            MW = mixing_simple(zs, self.MWs)
            return rho_to_Vm(rho, MW)
        # TODO: pressure dependence for the following methods:
        elif method == COSTALD_MIXTURE:
            return COSTALD_mixture(zs, T, self.Tcs, self.Vcs, self.omegas)
        elif method == COSTALD_MIXTURE_FIT:
            return COSTALD_mixture(zs, T, self.Tcs, self.COSTALD_Vchars, self.COSTALD_omegas)
        elif method == RACKETT:
            return Rackett_mixture(T, zs, self.MWs, self.Tcs, self.Pcs, self.Zcs)
        elif method == RACKETT_PARAMETERS:
            return Rackett_mixture(T, zs, self.MWs, self.Tcs, self.Pcs, self.Z_RAs)
        return super().calculate(T, P, zs, ws, method)

    def test_method_validity(self, T, P, zs, ws, method):
        if LALIBERTE in self.all_methods:
            # If everything is an electrolyte, accept only it as a method
            if method in self.all_methods:
                return method == LALIBERTE
        if method in self.all_methods:
            return True
        return super().test_method_validity(T, P, zs, ws, method)


#PR = 'PR'
CRC_VIRIAL = 'CRC_VIRIAL'
TSONOPOULOS_EXTENDED = 'TSONOPOULOS_EXTENDED'
TSONOPOULOS = 'TSONOPOULOS'
ABBOTT = 'ABBOTT'
PITZER_CURL = 'PITZER_CURL'
IDEAL = 'IDEAL'
volume_gas_methods = [COOLPROP, EOS, CRC_VIRIAL, TSONOPOULOS_EXTENDED, TSONOPOULOS,
                      ABBOTT, PITZER_CURL, IDEAL]
"""Holds all methods available for the :obj:`VolumeGas` class, for use in
iterating over them."""


class VolumeGas(TPDependentProperty):
    r'''Class for dealing with gas molar volume as a function of
    temperature and pressure.

    All considered methods are both temperature and pressure dependent. Included
    are four CSP methods for calculating second virial coefficients, one
    source of polynomials for calculating second virial coefficients, one
    equation of state (Peng-Robinson), and the ideal gas law.

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
    omega : float, optional
        Acentric factor, [-]
    dipole : float, optional
        Dipole, [debye]
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
    To iterate over all methods, use the list stored in
    :obj:`volume_gas_methods`.

    **PR**:
        Peng-Robinson Equation of State. See the appropriate module for more
        information.
    **CRC_VIRIAL**:
        Short polynomials, for 105 fluids from [1]_.  The full expression is:

        .. math::
            B = \sum_1^4 a_i\left[T_0/298.15-1\right]^{i-1}

    **TSONOPOULOS_EXTENDED**:
        CSP method for second virial coefficients, described in
        :obj:`chemicals.virial.BVirial_Tsonopoulos_extended`
    **TSONOPOULOS**:
        CSP method for second virial coefficients, described in
        :obj:`chemicals.virial.BVirial_Tsonopoulos`
    **ABBOTT**:
        CSP method for second virial coefficients, described in
        :obj:`chemicals.virial.BVirial_Abbott`. This method is the simplest CSP
        method implemented.
    **PITZER_CURL**:
        CSP method for second virial coefficients, described in
        :obj:`chemicals.virial.BVirial_Pitzer_Curl`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [2]_. Very slow, but unparalled in accuracy for pressure
        dependence.

    See Also
    --------
    :obj:`chemicals.virial.BVirial_Pitzer_Curl`
    :obj:`chemicals.virial.BVirial_Abbott`
    :obj:`chemicals.virial.BVirial_Tsonopoulos`
    :obj:`chemicals.virial.BVirial_Tsonopoulos_extended`

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    .. [2] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    '''

    name = 'Gas molar volume'
    units = 'm^3/mol'
    interpolation_T = None
    """No interpolation transformation by default."""

    @staticmethod
    def interpolation_P(P):
        '''Function to make the data-based interpolation as linear as possible.
        This transforms the input `P` into the `1/P` domain.
        '''
        return 1./P
    interpolation_property = None
    """No interpolation transformation by default."""
    interpolation_property_inv = None
    """No interpolation transformation by default."""
    tabular_extrapolation_permitted = True
    """Allow tabular extrapolation by default."""
    property_min = 0
    """Mimimum valid value of gas molar volume. It should normally be well
    above this."""
    property_max = 1E10
    """Maximum valid value of gas molar volume. Set roughly at an ideal gas
    at 1 Pa and 2 billion K."""

    Pmax = 1E9  # 1 GPa
    """Maximum pressure at which no method can calculate gas molar volume
    above."""
    Pmin = 0
    """Minimum pressure at which no method can calculate gas molar volume
    under."""
    ranked_methods = []
    """Default rankings of the low-pressure methods."""
    ranked_methods_P = [IDEAL, COOLPROP, EOS, TSONOPOULOS_EXTENDED, TSONOPOULOS, ABBOTT,
                        PITZER_CURL, CRC_VIRIAL]
    """Default rankings of the pressure-dependent methods."""


    custom_args = ('MW', 'Tc', 'Pc', 'omega', 'dipole', 'eos')
    def __init__(self, CASRN='', MW=None, Tc=None, Pc=None, omega=None,
                 dipole=None, eos=None, extrapolation=None,
                 **kwargs):
        # Only use TPDependentPropoerty functions here
        self.CASRN = CASRN
        self.MW = MW
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.dipole = dipole
        self.eos = eos
        super().__init__(extrapolation, **kwargs)

    def load_all_methods(self, load_data):
        r'''Method which picks out coefficients for the specified chemical
        from the various dictionaries and DataFrames storing it. All data is
        stored as attributes. This method also sets obj:`all_methods_P` as a
        set of methods for which the data exists for.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods_P = [IDEAL]
        self.T_limits = T_limits = {}
        CASRN = self.CASRN
        # no point in getting Tmin, Tmax
        if all((self.Tc, self.Pc, self.omega)):
            methods_P.extend([TSONOPOULOS_EXTENDED, TSONOPOULOS, ABBOTT,
                            PITZER_CURL])
            if self.eos:
                methods_P.append(EOS)
            T_limits[TSONOPOULOS_EXTENDED] = (1e-4, 1e5)
            T_limits[TSONOPOULOS] = (1e-4, 1e5)
            T_limits[ABBOTT] = (1e-4, 1e5)
            T_limits[PITZER_CURL] = (1e-4, 1e5)
        if load_data and CASRN:
            if CASRN in volume.rho_data_CRC_virial.index:
                methods_P.append(CRC_VIRIAL)
                self.CRC_VIRIAL_coeffs = volume.rho_values_CRC_virial[volume.rho_data_CRC_virial.index.get_loc(CASRN)].tolist()
                T_limits[CRC_VIRIAL] = (1e-4, 1e5)
            if has_CoolProp() and CASRN in coolprop_dict:
                methods_P.append(COOLPROP)
                self.CP_f = coolprop_fluids[CASRN]
                T_limits[CRC_VIRIAL] = (self.CP_f.Tmin, self.CP_f.Tmax)
        self.all_methods = set()
        self.all_methods_P = set(methods_P)
        for m in self.ranked_methods_P:
            if m in self.all_methods_P:
                self.method_P = m
                break

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent gas molar volume at
        temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate molar volume, [K]
        P : float
            Pressure at which to calculate molar volume, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Vm : float
            Molar volume of the gas at T and P, [m^3/mol]
        '''
        if method == EOS:
            if T < 0.0 or P < 0.0:
                return None
            self.eos[0] = self.eos[0].to_TP(T=T, P=P)
            Vm = self.eos[0].V_g
        elif method == TSONOPOULOS_EXTENDED:
            B = BVirial_Tsonopoulos_extended(T, self.Tc, self.Pc, self.omega, dipole=self.dipole)
            Vm = ideal_gas(T, P) + B
        elif method == TSONOPOULOS:
            B = BVirial_Tsonopoulos(T, self.Tc, self.Pc, self.omega)
            Vm = ideal_gas(T, P) + B
        elif method == ABBOTT:
            B = BVirial_Abbott(T, self.Tc, self.Pc, self.omega)
            Vm = ideal_gas(T, P) + B
        elif method == PITZER_CURL:
            B = BVirial_Pitzer_Curl(T, self.Tc, self.Pc, self.omega)
            Vm = ideal_gas(T, P) + B
        elif method == CRC_VIRIAL:
            a1, a2, a3, a4, a5 = self.CRC_VIRIAL_coeffs
            t = 298.15/T - 1.
            B = (a1 + t*(a2 + t*(a3 + t*(a4 + a5*t))))*1E-6
            Vm = ideal_gas(T, P) + B
        elif method == IDEAL:
            Vm = ideal_gas(T, P)
        elif method == COOLPROP:
            assert PhaseSI('T', T, 'P', P, self.CASRN) in ['gas', 'supercritical_gas', 'supercritical', 'supercritical_liquid']
            Vm = 1./PropsSI('DMOLAR', 'T', T, 'P', P, self.CASRN)
        else:
            return self._base_calculate_P(T, P, method)
        return Vm

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a pressure and temperature
        dependent gas molar volume method. For the four CSP methods that
        calculate second virial coefficient, the method is considered valid for
        all temperatures and pressures, with validity checking based on the
        result only. For **CRC_VIRIAL**, there is no limit but there should
        be one; at some conditions, a negative volume will result!
        For **COOLPROP**, the fluid must be both a gas at the given conditions
        and under the maximum pressure of the fluid's EOS.

        For the equation of state **PR**, the determined phase must be a gas.
        For **IDEAL**, there are no limits.

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
        if T < 0 or P < 0:
            validity = False
        elif method in [TSONOPOULOS_EXTENDED, TSONOPOULOS, ABBOTT,
                        PITZER_CURL, CRC_VIRIAL, IDEAL, EOS]:
            pass
            # Would be nice to have a limit on CRC_VIRIAL
        elif method == EOS:
            eos = self.eos[0]
            # Some EOSs do not implement Psat, and so we must assume Vmg is
            # unavailable
            try:
                if T < eos.Tc and P > eos.Psat(T):
                    validity = False
            except:
                validity = False
        elif method == COOLPROP:
            validity = PhaseSI('T', T, 'P', P, self.CASRN) in ['gas', 'supercritical_gas', 'supercritical', 'supercritical_liquid']
        else:
            return super().test_method_validity_P(T, P, method)
        return validity

LINEAR_MISSING_IDEAL = 'LINEAR_MISSING_IDEAL'

volume_gas_mixture_methods = [EOS, LINEAR, IDEAL]
"""Holds all methods available for the :obj:`VolumeGasMixture` class, for use
in iterating over them."""




class VolumeGasMixture(MixtureProperty):
    '''Class for dealing with the molar volume of a gas mixture as a
    function of temperature, pressure, and composition.
    Consists of an equation of state, the ideal gas law, and one mole-weighted
    averaging method.

    Prefered method is **EOS**, or **IDEAL** if critical properties of
    components are unavailable.

    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture, [-]
    VolumeGases : list[VolumeGas], optional
        VolumeGas objects created for all species in the mixture, [-]
    eos : container[EOS Object], optional
        Equation of state mixture object, [-]
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`volume_gas_mixture_methods`.

    **EOS**:
        Equation of state mixture object; see :obj:`thermo.eos_mix` for more
        details.
    **LINEAR**:
        Linear mole fraction mixing rule described in
        :obj:`mixing_simple <chemicals.utils.mixing_simple>`; more correct than the ideal gas
        law.
    **IDEAL**:
        The ideal gas law.

    See Also
    --------
    chemicals.volume.ideal_gas
    :obj:`thermo.eos_mix`

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''

    name = 'Gas volume'
    units = 'm^3/mol'
    property_min = 0.
    """Mimimum valid value of gas molar volume. It should normally be well
    above this."""
    property_max = 1E10
    """Maximum valid value of gas molar volume. Set roughly at an ideal gas
    at 1 Pa and 2 billion K."""

    ranked_methods = [EOS, LINEAR, IDEAL, LINEAR_MISSING_IDEAL]

    pure_references = ('VolumeGases',)
    pure_reference_types = (VolumeGas, )
    obj_references = ('VolumeGases', 'eos')

    pure_constants = ('MWs', )
    custom_args = ('MWs', 'eos')

    def __init__(self, eos=None, CASs=[], VolumeGases=[], MWs=[], **kwargs):
        self.CASs = CASs
        self.VolumeGases = VolumeGases
        self.eos = eos
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
        methods = [LINEAR, IDEAL, LINEAR_MISSING_IDEAL]
        if self.eos:
            methods.append(EOS)
        self.all_methods = all_methods =  set(methods)

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate molar volume of a gas mixture at
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
        Vm : float
            Molar volume of the gas mixture at the given conditions, [m^3/mol]
        '''
        if method == LINEAR:
            Vms = [i(T, P) for i in self.VolumeGases]
            return mixing_simple(zs, Vms)
        elif method == LINEAR_MISSING_IDEAL:
            Vms = [i(T, P) for i in self.VolumeGases]
            V_ideal = ideal_gas(T, P)
            for i, Vm in enumerate(Vms):
                if Vm is None:
                    Vms[i] = V_ideal
            return mixing_simple(zs, Vms)
        elif method == IDEAL:
            return ideal_gas(T, P)
        elif method == EOS:
            self.eos[0] = self.eos[0].to_TP_zs(T=T, P=P, zs=zs)
            return self.eos[0].V_g
        return super().calculate(T, P, zs, ws, method)

    def test_method_validity(self, T, P, zs, ws, method):
        if method in self.all_methods:
            return True
        return super().test_method_validity(T, P, zs, ws, method)



GOODMAN = 'GOODMAN'
CRC_INORG_S = 'CRC_INORG_S'
volume_solid_methods = [GOODMAN, CRC_INORG_S]
"""Holds all methods available for the :obj:`VolumeSolid` class, for use in
iterating over them."""


class VolumeSolid(TDependentProperty):
    r'''Class for dealing with solid molar volume as a function of temperature.
    Consists of one constant value source, and one simple estimator based on
    liquid molar volume.

    Parameters
    ----------
    CASRN : str, optional
        CAS number
    MW : float, optional
        Molecular weight, [g/mol]
    Tt : float, optional
        Triple temperature
    Vml_Tt : float, optional
        Liquid molar volume at the triple point
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
    To iterate over all methods, use the list stored in
    :obj:`volume_solid_methods`.

    **CRC_INORG_S**:
        Constant values in [1]_, for 1872 chemicals.
    **GOODMAN**:
        Simple method using the liquid molar volume. Good up to 0.3*Tt.
        See :obj:`Goodman <chemicals.volume.Goodman>` for details.

    See Also
    --------
    chemicals.volume.Goodman

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    '''

    name = 'Solid molar volume'
    units = 'm^3/mol'
    interpolation_T = None
    """No interpolation transformation by default."""
    interpolation_property = None
    """No interpolation transformation by default."""
    interpolation_property_inv = None
    """No interpolation transformation by default."""
    tabular_extrapolation_permitted = True
    """Allow tabular extrapolation by default."""
    property_min = 0.
    """Molar volume cannot be under 0."""
    property_max = 2e-3
    """Maximum value of Heat capacity; arbitrarily set to 0.002, as the largest
    in the data is 0.00136."""

    ranked_methods = [CRC_INORG_S, GOODMAN]
    """Default rankings of the available methods."""

    custom_args = ('MW', 'Tt', 'Vml_Tt')

    def __init__(self, CASRN='', MW=None, Tt=None, Vml_Tt=None,
                 extrapolation='linear', **kwargs):
        self.CASRN = CASRN
        self.MW = MW
        self.Tt = Tt
        self.Vml_Tt = Vml_Tt

        super().__init__(extrapolation, **kwargs)



    def load_all_methods(self, load_data=True):
        r'''Method which picks out coefficients for the specified chemical
        from the various dictionaries and DataFrames storing it. All data is
        stored as attributes. This method also sets :obj:`Tmin`, :obj:`Tmax`,
        and :obj:`all_methods` as a set of methods for which the data exists for.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods = []
        self.T_limits = T_limits = {}
        CASRN = self.CASRN
        if load_data and CASRN:
            if CASRN in volume.rho_data_CRC_inorg_s_const.index:
                methods.append(CRC_INORG_S)
                self.CRC_INORG_S_Vm = float(volume.rho_data_CRC_inorg_s_const.at[CASRN, 'Vm'])
                T_limits[CRC_INORG_S] = (1e-4, 1e4)
        if all((self.Tt, self.Vml_Tt, self.MW)):
            methods.append(GOODMAN)
            T_limits[GOODMAN] = (1e-4, self.Tt)
        self.all_methods = set(methods)

    def calculate(self, T, method):
        r'''Method to calculate the molar volume of a solid at tempearture `T`
        with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate molar volume, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Vms : float
            Molar volume of the solid at T, [m^3/mol]
        '''
        if method == CRC_INORG_S:
            Vms = self.CRC_INORG_S_Vm
        elif method == GOODMAN:
            Vms = Goodman(T, self.Tt, self.Vml_Tt)
        else:
            return self._base_calculate(T, method)
        return Vms

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For tabular data,
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
        validity = True
        if T < 0:
            validity = False
        elif method == CRC_INORG_S:
            pass
            # Assume the solid density value is good at any possible T
        elif method == GOODMAN:
            if T < self.Tt*0.3:
                validity = False
        else:
            return super().test_method_validity(T, method)
        return validity

try:
    VolumeSolid._custom_set_poly_fit = VolumeLiquid._custom_set_poly_fit
except:
    pass


volume_solid_mixture_methods = [LINEAR]
"""Holds all methods available for the :obj:`VolumeSolidMixture` class, for use
in iterating over them."""

class VolumeSolidMixture(MixtureProperty):
    '''Class for dealing with the molar volume of a solid mixture as a
    function of temperature, pressure, and composition.
    Consists of only mole-weighted averaging.

    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture, [-]
    VolumeSolids : list[VolumeSolid], optional
        VolumeSolid objects created for all species in the mixture, [-]
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`volume_solid_mixture_methods`.

    **LINEAR**:
        Linear mole fraction mixing rule described in
        :obj:`mixing_simple <chemicals.utils.mixing_simple>`.
    '''

    name = 'Solid molar volume'
    units = 'm^3/mol'
    property_min = 0
    """Molar volume cannot be under 0."""
    property_max = 2e-3
    """Maximum value of Heat capacity; arbitrarily set to 0.002, as the largest
    in the data is 0.00136."""

    ranked_methods = [LINEAR]

    pure_references = ('VolumeSolids',)
    pure_reference_types = (VolumeSolid, )
    obj_references = ('VolumeSolids',)

    pure_constants = ('MWs', )
    custom_args = pure_constants

    def __init__(self, CASs=[], VolumeSolids=[], MWs=[], **kwargs):
        self.CASs = CASs
        self.VolumeSolids = VolumeSolids
        self.MWs = MWs

        self.Tmin = 0
        self.Tmax = 1E4
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
        methods = [LINEAR]
        self.all_methods = all_methods = set(methods)

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate molar volume of a solid mixture at
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
        Vm : float
            Molar volume of the solid mixture at the given conditions,
            [m^3/mol]
        '''
        return super().calculate(T, P, zs, ws, method)

    def test_method_validity(self, T, P, zs, ws, method):
        if method in self.all_methods:
            return True
        return super().test_method_validity(T, P, zs, ws, method)

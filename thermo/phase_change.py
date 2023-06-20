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

This module contains implementations of :obj:`thermo.utils.TDependentProperty`
representing enthalpy of vaporization and enthalpy of sublimation. A variety of
estimation and data methods are available as included in the `chemicals` library.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Enthalpy of Vaporization
========================
.. autoclass:: EnthalpyVaporization
    :members: calculate, test_method_validity,
              interpolation_T, interpolation_property,
              interpolation_property_inv, name, property_max, property_min,
              units, ranked_methods, Watson_exponent
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: enthalpy_vaporization_methods

Enthalpy of Sublimation
=======================
.. autoclass:: EnthalpySublimation
    :members: calculate, test_method_validity,
              interpolation_T, interpolation_property,
              interpolation_property_inv, name, property_max, property_min,
              units, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: enthalpy_sublimation_methods
'''


__all__ = ['enthalpy_vaporization_methods', 'EnthalpyVaporization',
           'enthalpy_sublimation_methods', 'EnthalpySublimation']



from chemicals import miscdata, phase_change
from chemicals.dippr import EQ106
from chemicals.identifiers import CAS_to_int
from chemicals.miscdata import lookup_VDI_tabular_data
from chemicals.phase_change import MK, PPDS12, SMK, Alibakhshi, Chen, Clapeyron, Liu, Pitzer, Riedel, Velasco, Vetere, Watson
from fluids.numerics import isnan

from thermo.coolprop import CoolProp_failing_PT_flashes, PropsSI, coolprop_dict, coolprop_fluids, has_CoolProp
from thermo.heat_capacity import HeatCapacityGas, HeatCapacitySolid
from thermo.utils import COOLPROP, DIPPR_PERRY_8E, HEOS_FIT, VDI_PPDS, VDI_TABULAR, TDependentProperty

CRC_HVAP_TB = 'CRC_HVAP_TB'
CRC_HVAP_298 = 'CRC_HVAP_298'
GHARAGHEIZI_HVAP_298 = 'GHARAGHEIZI_HVAP_298'
MORGAN_KOBAYASHI = 'MORGAN_KOBAYASHI'
SIVARAMAN_MAGEE_KOBAYASHI = 'SIVARAMAN_MAGEE_KOBAYASHI'
VELASCO = 'VELASCO'
PITZER = 'PITZER'
CLAPEYRON = 'CLAPEYRON'
ALIBAKHSHI = 'ALIBAKHSHI'

RIEDEL = 'RIEDEL'
CHEN = 'CHEN'
LIU = 'LIU'
VETERE = 'VETERE'

enthalpy_vaporization_methods = [HEOS_FIT, DIPPR_PERRY_8E, VDI_PPDS, COOLPROP, VDI_TABULAR,
                                 MORGAN_KOBAYASHI,
                      SIVARAMAN_MAGEE_KOBAYASHI, VELASCO, PITZER, ALIBAKHSHI,
                      CRC_HVAP_TB, CRC_HVAP_298, GHARAGHEIZI_HVAP_298,
                      CLAPEYRON, RIEDEL, CHEN, VETERE, LIU]
"""Holds all methods available for the EnthalpyVaporization class, for use in
iterating over them."""


class EnthalpyVaporization(TDependentProperty):
    '''Class for dealing with heat of vaporization as a function of temperature.
    Consists of three constant value data sources, one source of tabular
    information, three coefficient-based methods, nine corresponding-states
    estimators, and the external library CoolProp.

    Parameters
    ----------
    Tb : float, optional
        Boiling point, [K]
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    omega : float, optional
        Acentric factor, [-]
    similarity_variable : float, optional
        similarity variable, n_atoms/MW, [mol/g]
    Psat : float or callable, optional
        Vapor pressure at T or callable for the same, [Pa]
    Zl : float or callable, optional
        Compressibility of liquid at T or callable for the same, [-]
    Zg : float or callable, optional
        Compressibility of gas at T or callable for the same, [-]
    CASRN : str, optional
        The CAS number of the chemical
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
    To iterate over all methods, use the list stored in
    :obj:`enthalpy_vaporization_methods`.

    **CLAPEYRON**:
        The Clapeyron fundamental model desecribed in :obj:`Clapeyron <chemicals.phase_change.Clapeyron>`.
        This is the model which uses `Zl`, `Zg`, and `Psat`, all of which
        must be set at each temperature change to allow recalculation of
        the heat of vaporization.
    **MORGAN_KOBAYASHI**:
        The MK CSP model equation documented in :obj:`MK <chemicals.phase_change.MK>`.
    **SIVARAMAN_MAGEE_KOBAYASHI**:
        The SMK CSP model equation documented in :obj:`SMK <chemicals.phase_change.SMK>`.
    **VELASCO**:
        The Velasco CSP model equation documented in :obj:`Velasco <chemicals.phase_change.Velasco>`.
    **PITZER**:
        The Pitzer CSP model equation documented in :obj:`Pitzer <chemicals.phase_change.Pitzer>`.
    **RIEDEL**:
        The Riedel CSP model equation, valid at the boiling point only,
        documented in :obj:`Riedel <chemicals.phase_change.Riedel>`. This is adjusted with the :obj:`Watson <chemicals.phase_change.Watson>`
        equation unless `Tc` is not available.
    **CHEN**:
        The Chen CSP model equation, valid at the boiling point only,
        documented in :obj:`Chen <chemicals.phase_change.Chen>`. This is adjusted with the :obj:`Watson <chemicals.phase_change.Watson>`
        equation unless `Tc` is not available.
    **VETERE**:
        The Vetere CSP model equation, valid at the boiling point only,
        documented in :obj:`Vetere <chemicals.phase_change.Vetere>`. This is adjusted with the :obj:`Watson <chemicals.phase_change.Watson>`
        equation unless `Tc` is not available.
    **LIU**:
        The Liu CSP model equation, valid at the boiling point only,
        documented in :obj:`Liu <chemicals.phase_change.Liu>`. This is adjusted with the :obj:`Watson <chemicals.phase_change.Watson>`
        equation unless `Tc` is not available.
    **CRC_HVAP_TB**:
        The constant value available in [4]_ at the normal boiling point. This
        is adusted  with the :obj:`Watson <chemicals.phase_change.Watson>` equation unless `Tc` is not
        available. Data is available for 707 chemicals.
    **CRC_HVAP_298**:
        The constant value available in [4]_ at 298.15 K. This
        is adusted  with the :obj:`Watson <chemicals.phase_change.Watson>` equation unless `Tc` is not
        available. Data is available for 633 chemicals.
    **GHARAGHEIZI_HVAP_298**:
        The constant value available in [5]_ at 298.15 K. This
        is adusted  with the :obj:`Watson <chemicals.phase_change.Watson>` equation unless `Tc` is not
        available. Data is available for 2730 chemicals.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [3]_. Very slow but accurate.
    **VDI_TABULAR**:
        Tabular data in [4]_ along the saturation curve; interpolation is as
        set by the user or the default.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS, published
        openly in [3]_. Extrapolates poorly at low temperatures.
    **DIPPR_PERRY_8E**:
        A collection of 344 coefficient sets from the DIPPR database published
        openly in [6]_. Provides temperature limits for all its fluids.
        :obj:`chemicals.dippr.EQ106` is used for its fluids.
    **ALIBAKHSHI**:
        One-constant limited temperature range regression method presented
        in [7]_, with constants for ~2000 chemicals from the DIPPR database.
        Valid up to 100 K below the critical point, and 50 K under the boiling
        point.
    **HEOS_FIT**:
        A series of higher-order polynomial fits to the calculated results from
        fundamental helmholtz equations of state as calculated with REFPROP

    See Also
    --------
    chemicals.phase_change.MK
    chemicals.phase_change.SMK
    chemicals.phase_change.Velasco
    chemicals.phase_change.Clapeyron
    chemicals.phase_change.Riedel
    chemicals.phase_change.Chen
    chemicals.phase_change.Vetere
    chemicals.phase_change.Liu
    chemicals.phase_change.Watson

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [2] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       “Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp.” Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    .. [4] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    .. [5] Gharagheizi, Farhad, Poorandokht Ilani-Kashkouli, William E. Acree Jr.,
       Amir H. Mohammadi, and Deresh Ramjugernath. "A Group Contribution Model for
       Determining the Vaporization Enthalpy of Organic Compounds at the Standard
       Reference Temperature of 298 K." Fluid Phase Equilibria 360
       (December 25, 2013): 279-92. doi:10.1016/j.fluid.2013.09.021.
    .. [6] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    .. [7] Alibakhshi, Amin. "Enthalpy of Vaporization, Its Temperature
       Dependence and Correlation with Surface Tension: A Theoretical
       Approach." Fluid Phase Equilibria 432 (January 25, 2017): 62-69.
       doi:10.1016/j.fluid.2016.10.013.
    '''

    name = 'Enthalpy of vaporization'
    units = 'J/mol'
    interpolation_T = None
    """No interpolation transformation by default."""
    interpolation_property = None
    """No interpolation transformation by default."""
    interpolation_property_inv = None
    """No interpolation transformation by default."""
    tabular_extrapolation_permitted = True
    """Allow tabular extrapolation by default; values below 0 will be obtained
    at high temperatures."""
    property_min = 0
    """Mimimum valid value of heat of vaporization. This occurs at the critical
    point exactly."""
    property_max = 1E6
    """Maximum valid of heat of vaporization. Set to twice the value in the
    available data."""
    critical_zero = True
    """Whether or not the property is declining and reaching zero at the
    critical point."""

    ranked_methods = [HEOS_FIT, COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, MORGAN_KOBAYASHI,
                      SIVARAMAN_MAGEE_KOBAYASHI, VELASCO, PITZER, VDI_TABULAR,
                      ALIBAKHSHI,
                      CRC_HVAP_TB, CRC_HVAP_298, GHARAGHEIZI_HVAP_298,
                      CLAPEYRON, RIEDEL, CHEN, VETERE, LIU]
    """Default rankings of the available methods."""
    boiling_methods = [RIEDEL, CHEN, VETERE, LIU]
    CSP_methods = [MORGAN_KOBAYASHI, SIVARAMAN_MAGEE_KOBAYASHI,
                   VELASCO, PITZER]
    Watson_exponent = 0.38
    """Exponent used in the Watson equation"""

    custom_args = ('Tb', 'Tc', 'Pc', 'omega', 'similarity_variable', 'Psat',
                   'Zl', 'Zg',)

    def __init__(self, CASRN='', Tb=None, Tc=None, Pc=None, omega=None,
                 similarity_variable=None, Psat=None, Zl=None, Zg=None,
                 extrapolation='Watson', **kwargs):
        self.CASRN = CASRN
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.similarity_variable = similarity_variable
        self.Psat = Psat
        self.Zl = Zl
        self.Zg = Zg
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
        self.all_methods = set()
        CASRN = self.CASRN
        if load_data and CASRN:
            if has_CoolProp() and CASRN in coolprop_dict:
                methods.append(COOLPROP)
                self.CP_f = coolprop_fluids[CASRN]
                T_limits[COOLPROP] = (self.CP_f.Tt, self.CP_f.Tc*.9999)
            if CASRN in miscdata.VDI_saturation_dict:
                Ts, props = lookup_VDI_tabular_data(CASRN, 'Hvap')
                self.add_tabular_data(Ts, props, VDI_TABULAR, check_properties=False)
                del self._method
            if CASRN in phase_change.phase_change_data_Alibakhshi_Cs.index and self.Tc is not None:
                methods.append(ALIBAKHSHI)
                self.Alibakhshi_C = float(phase_change.phase_change_data_Alibakhshi_Cs.at[CASRN, 'C'])
                T_limits[ALIBAKHSHI] = (self.Tc*.3, max(self.Tc-100., 0))
            if CASRN in phase_change.Hvap_data_CRC.index and not isnan(phase_change.Hvap_data_CRC.at[CASRN, 'HvapTb']):
                methods.append(CRC_HVAP_TB)
                self.CRC_HVAP_TB_Tb = float(phase_change.Hvap_data_CRC.at[CASRN, 'Tb'])
                self.CRC_HVAP_TB_Hvap = float(phase_change.Hvap_data_CRC.at[CASRN, 'HvapTb'])
                if self.Tc is not None:
                    T_limits[CRC_HVAP_TB] = (self.Tc*.001, self.Tc)
                else:
                    T_limits[CRC_HVAP_TB] = (self.CRC_HVAP_TB_Tb, self.CRC_HVAP_TB_Tb)
            if CASRN in phase_change.Hvap_data_CRC.index and not isnan(phase_change.Hvap_data_CRC.at[CASRN, 'Hvap298']):
                methods.append(CRC_HVAP_298)
                self.CRC_HVAP_298 = float(phase_change.Hvap_data_CRC.at[CASRN, 'Hvap298'])
                if self.Tc is not None:
                    T_limits[CRC_HVAP_298] = (self.Tc*.001, self.Tc)
                else:
                    T_limits[CRC_HVAP_298] =  (298.15, 298.15)
            if CASRN in phase_change.Hvap_data_Gharagheizi.index:
                methods.append(GHARAGHEIZI_HVAP_298)
                self.GHARAGHEIZI_HVAP_298_Hvap = float(phase_change.Hvap_data_Gharagheizi.at[CASRN, 'Hvap298'])
                if self.Tc is not None:
                    T_limits[GHARAGHEIZI_HVAP_298] = (self.Tc*.001, self.Tc)
                else:
                    T_limits[GHARAGHEIZI_HVAP_298] =  (298.15, 298.15)
            if CASRN in phase_change.phase_change_data_Perrys2_150.index:
                methods.append(DIPPR_PERRY_8E)
                Tc, C1, C2, C3, C4, self.Perrys2_150_Tmin, self.Perrys2_150_Tmax = phase_change.phase_change_values_Perrys2_150[phase_change.phase_change_data_Perrys2_150.index.get_loc(CASRN)].tolist()
                self.Perrys2_150_coeffs = [Tc, C1, C2, C3, C4]
                T_limits[DIPPR_PERRY_8E] = (self.Perrys2_150_Tmin, self.Perrys2_150_Tmax)
            if CASRN in phase_change.phase_change_data_VDI_PPDS_4.index:
                Tc, A, B, C, D, E = phase_change.phase_change_values_VDI_PPDS_4[phase_change.phase_change_data_VDI_PPDS_4.index.get_loc(CASRN)].tolist()
                self.VDI_PPDS_coeffs = [A, B, C, D, E]
                self.VDI_PPDS_Tc = Tc
                methods.append(VDI_PPDS)
                T_limits[VDI_PPDS] = (0.1*self.VDI_PPDS_Tc, self.VDI_PPDS_Tc)
        if all((self.Tc, self.omega)):
            methods.extend(self.CSP_methods)
            for m in self.CSP_methods:
                T_limits[m] = (1e-4, self.Tc)
        if all((self.Tc, self.Pc)):
            methods.append(CLAPEYRON)
            T_limits[CLAPEYRON] = (1e-4, self.Tc)
        if all((self.Tb, self.Tc, self.Pc)):
            methods.extend(self.boiling_methods)
            for m in self.boiling_methods:
                T_limits[m] = (1e-4, self.Tc)
        self.all_methods.update(methods)

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {COOLPROP : [CAS for CAS in coolprop_dict if (CAS not in CoolProp_failing_PT_flashes)],
                VDI_TABULAR: list(miscdata.VDI_saturation_dict.keys()),
                DIPPR_PERRY_8E: phase_change.phase_change_data_Perrys2_150.index,
                VDI_PPDS: phase_change.phase_change_data_VDI_PPDS_4.index,
                }


    def calculate(self, T, method):
        r'''Method to calculate heat of vaporization of a liquid at
        temperature `T` with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate heat of vaporization, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Hvap : float
            Heat of vaporization of the liquid at T, [J/mol]
        '''
        if method == COOLPROP:
            Hvap = PropsSI('HMOLAR', 'T', T, 'Q', 1, self.CASRN) - PropsSI('HMOLAR', 'T', T, 'Q', 0, self.CASRN)
        elif method == DIPPR_PERRY_8E:
            Hvap = EQ106(T, *self.Perrys2_150_coeffs)
        # CSP methods
        elif method == VDI_PPDS:
            Hvap = PPDS12(T, self.VDI_PPDS_Tc, *self.VDI_PPDS_coeffs)
        elif method == ALIBAKHSHI:
            Hvap = Alibakhshi(T=T, Tc=self.Tc, C=self.Alibakhshi_C)
        elif method == MORGAN_KOBAYASHI:
            Hvap = MK(T, self.Tc, self.omega)
        elif method == SIVARAMAN_MAGEE_KOBAYASHI:
            Hvap = SMK(T, self.Tc, self.omega)
        elif method == VELASCO:
            Hvap = Velasco(T, self.Tc, self.omega)
        elif method == PITZER:
            Hvap = Pitzer(T, self.Tc, self.omega)
        elif method == CLAPEYRON:
            Psat = self.Psat(T) if callable(self.Psat) else self.Psat
            Zg = self.Zg(T, Psat) if callable(self.Zg) else self.Zg
            Zl = self.Zl(T, Psat) if callable(self.Zl) else self.Zl
            if Zg:
                if Zl:
                    dZ = Zg-Zl
                else:
                    dZ = Zg
            Hvap = Clapeyron(T, self.Tc, self.Pc, dZ=dZ, Psat=Psat)
        # CSP methods at Tb only
        elif method == RIEDEL:
            Hvap = Riedel(self.Tb, self.Tc, self.Pc)
        elif method == CHEN:
            Hvap = Chen(self.Tb, self.Tc, self.Pc)
        elif method == VETERE:
            Hvap = Vetere(self.Tb, self.Tc, self.Pc)
        elif method == LIU:
            Hvap = Liu(self.Tb, self.Tc, self.Pc)
        # Individual data point methods
        elif method == CRC_HVAP_TB:
            Hvap = self.CRC_HVAP_TB_Hvap
        elif method == CRC_HVAP_298:
            Hvap = self.CRC_HVAP_298
        elif method == GHARAGHEIZI_HVAP_298:
            Hvap = self.GHARAGHEIZI_HVAP_298_Hvap
        else:
            return self._base_calculate(T, method)
        # Adjust with the watson equation if estimated at Tb or Tc only
        if method in self.boiling_methods or (self.Tc and method in (CRC_HVAP_TB, CRC_HVAP_298, GHARAGHEIZI_HVAP_298)):
            if method in self.boiling_methods:
                Tref = self.Tb
            elif method == CRC_HVAP_TB:
                Tref = self.CRC_HVAP_TB_Tb
            elif method in [CRC_HVAP_298, GHARAGHEIZI_HVAP_298]:
                Tref = 298.15
            Hvap = Watson(T, Hvap, Tref, self.Tc, self.Watson_exponent)
        return Hvap

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. For CSP methods, the
        models are considered valid from 0 K to the critical point. For
        tabular data, extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the
        extrapolation is considered valid for all temperatures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        The constant methods **CRC_HVAP_TB**, **CRC_HVAP_298**, and
        **GHARAGHEIZI_HVAP** are adjusted for temperature dependence according
        to the :obj:`Watson <chemicals.phase_change.Watson>` equation, with a temperature exponent as set in
        :obj:`Watson_exponent`, usually regarded as 0.38. However, if Tc is
        not set, then the adjustment cannot be made. In that case the methods
        are considered valid for within 5 K of their boiling point or 298.15 K
        as appropriate.

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
        if method == COOLPROP:
            if T <= self.CP_f.Tmin or T > self.CP_f.Tc:
                validity = False
        elif method == DIPPR_PERRY_8E:
            if T < self.Perrys2_150_Tmin or T > self.Perrys2_150_Tmax:
                return False
        elif method == CRC_HVAP_TB:
            if not self.Tc:
                if T < self.CRC_HVAP_TB_Tb - 5 or T > self.CRC_HVAP_TB_Tb + 5:
                    validity = False
            else:
                validity = T <= self.Tc
        elif method in (CRC_HVAP_298, GHARAGHEIZI_HVAP_298):
            if not self.Tc:
                if T < 298.15 - 5 or T > 298.15 + 5:
                    validity = False
        elif method == VDI_PPDS:
            validity = T <= self.VDI_PPDS_Tc
        elif method in self.boiling_methods:
            if T > self.Tc:
                validity = False
        elif method in self.CSP_methods:
            if T > self.Tc:
                validity = False
        elif method == ALIBAKHSHI:
            if T > self.Tc - 100:
                validity = False
#            elif (self.Tb and T < self.Tb - 50):
#                validity = False

        elif method == CLAPEYRON:
            if not (self.Psat and T < self.Tc):
                validity = False
        else:
            return super().test_method_validity(T, method)
        return validity

### Heat of Sublimation


GHARAGHEIZI_HSUB_298 = 'GHARAGHEIZI_HSUB_298'
GHARAGHEIZI_HSUB = 'GHARAGHEIZI_HSUB'
CRC_HFUS_HVAP_TM = 'CRC_HFUS_HVAP_TM' # Gets Tm
WEBBOOK_HSUB = 'WEBBOOK_HSUB'

enthalpy_sublimation_methods = [WEBBOOK_HSUB, GHARAGHEIZI_HSUB, CRC_HFUS_HVAP_TM,
                                GHARAGHEIZI_HSUB_298]
"""Holds all methods available for the EnthalpySublimation class, for use in
iterating over them."""


class EnthalpySublimation(TDependentProperty):
    '''Class for dealing with heat of sublimation as a function of temperature.
    Consists of one temperature-dependent method based on the heat of
    sublimation at 298.15 K.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical
    Tm : float, optional
        Normal melting temperature, [K]
    Tt : float, optional
        Triple point temperature, [K]
    Cpg : float or callable, optional
        Gaseous heat capacity at a given temperature or callable for the same,
        [J/mol/K]
    Cps : float or callable, optional
        Solid heat capacity at a given temperature or callable for the same,
        [J/mol/K]
    Hvap : float of callable, optional
        Enthalpy of Vaporization at a given temperature or callable for the
        same, [J/mol]
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
    To iterate over all methods, use the list stored in
    :obj:`enthalpy_sublimation_methods`.

    **WEBBOOK_HSUB**:
        Enthalpy of sublimation at a constant temperature of 298.15 K as given
        in [3]_.
    **GHARAGHEIZI_HSUB_298**:
        Enthalpy of sublimation at a constant temperature of 298 K as given in
        [1]_.
    **GHARAGHEIZI_HSUB**:
        Enthalpy of sublimation at a constant temperature of 298 K as given in
        [1]_ are adjusted using the solid and gas heat capacity functions to
        correct for any temperature.
    **CRC_HFUS_HVAP_TM**:
        Enthalpies of fusion in [1]_ are corrected to be enthalpies of
        sublimation by adding the enthalpy of vaporization at the fusion
        temperature, and then adjusted using the solid and gas heat capacity
        functions to correct for any temperature.

    See Also
    --------


    References
    ----------
    .. [1] Gharagheizi, Farhad, Poorandokht Ilani-Kashkouli, William E. Acree
       Jr., Amir H. Mohammadi, and Deresh Ramjugernath. "A Group Contribution
       Model for Determining the Sublimation Enthalpy of Organic Compounds at
       the Standard Reference Temperature of 298 K." Fluid Phase Equilibria 354
       (September 25, 2013): 265-doi:10.1016/j.fluid.2013.06.046.
    .. [2] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics, 95E. Boca Raton, FL: CRC press, 2014.
    .. [3] Shen, V.K., Siderius, D.W., Krekelberg, W.P., and Hatch, H.W., Eds.,
       NIST WebBook, NIST, http://doi.org/10.18434/T4M88Q
    '''

    name = 'Enthalpy of sublimation'
    units = 'J/mol'
    interpolation_T = None
    """No interpolation transformation by default."""
    interpolation_property = None
    """No interpolation transformation by default."""
    interpolation_property_inv = None
    """No interpolation transformation by default."""
    tabular_extrapolation_permitted = True
    """Allow tabular extrapolation by default; values below 0 will be obtained
    at high temperatures."""
    property_min = 0
    """Mimimum valid value of heat of vaporization. A theoretical concept only."""
    property_max = 1E6
    """Maximum valid of heat of sublimation. A theoretical concept only."""

    ranked_methods = [WEBBOOK_HSUB, GHARAGHEIZI_HSUB, CRC_HFUS_HVAP_TM,
                      GHARAGHEIZI_HSUB_298]

    obj_references = pure_references = ('Cpg', 'Cps', 'Hvap')
    obj_references_types = pure_reference_types = (HeatCapacityGas, HeatCapacitySolid, EnthalpyVaporization)


    custom_args = ('Tm', 'Tt', 'Cpg', 'Cps', 'Hvap')
    def __init__(self, CASRN='', Tm=None, Tt=None, Cpg=None, Cps=None,
                 Hvap=None, extrapolation='linear', **kwargs):
        self.CASRN = CASRN
        self.Tm = Tm
        self.Tt = Tt
        self.Cpg = Cpg
        self.Cps = Cps
        self.Hvap = Hvap
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
        CASRN_int = None if not CASRN else CAS_to_int(CASRN)
        if load_data and CASRN:
            if CASRN_int in miscdata.webbook_data.index and not isnan(float(miscdata.webbook_data.at[CASRN_int, 'Hsub'])):
                methods.append(WEBBOOK_HSUB)
                self.webbook_Hsub = float(miscdata.webbook_data.at[CASRN_int, 'Hsub'])
                if self.Tm is not None:
                    T_limits[WEBBOOK_HSUB] = (self.Tm, self.Tm)
                else:
                    T_limits[WEBBOOK_HSUB] = (298.15, 298.15)

            if CASRN in phase_change.Hsub_data_Gharagheizi.index:
                methods.append(GHARAGHEIZI_HSUB_298)
                self.GHARAGHEIZI_Hsub = float(phase_change.Hsub_data_Gharagheizi.at[CASRN, 'Hsub'])
                if self.Cpg is not None and self.Cps is not None:
                    methods.append(GHARAGHEIZI_HSUB)
                T_limits[GHARAGHEIZI_HSUB_298] = (298.15, 298.15)
            if CASRN in phase_change.Hfus_data_CRC.index:
                methods.append(CRC_HFUS_HVAP_TM)
                self.CRC_Hfus = float(phase_change.Hfus_data_CRC.at[CASRN, 'Hfus'])
                if self.Tm is not None:
                    T_limits[CRC_HFUS_HVAP_TM] = (self.Tm, self.Tm)
                else:
                    T_limits[CRC_HFUS_HVAP_TM] = (298.15, 298.15)
        self.all_methods = set(methods)

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {}

    def calculate(self, T, method):
        r'''Method to calculate heat of sublimation of a solid at
        temperature `T` with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate heat of sublimation, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Hsub : float
            Heat of sublimation of the solid at T, [J/mol]
        '''
        if method == GHARAGHEIZI_HSUB_298:
            Hsub = self.GHARAGHEIZI_Hsub
        elif method == WEBBOOK_HSUB:
            Hsub = self.webbook_Hsub
        elif method == GHARAGHEIZI_HSUB:
            T_base = 298.15
            Hsub = self.GHARAGHEIZI_Hsub
        elif method == CRC_HFUS_HVAP_TM:
            T_base = self.Tm
            Hsub = self.CRC_Hfus
            try:
                Hsub += self.Hvap(T_base)
            except:
                Hsub += self.Hvap
        else:
            return self._base_calculate(T, method)
        if method in (GHARAGHEIZI_HSUB, CRC_HFUS_HVAP_TM):
            try:
#                Cpg, Cps = self.Cpg(T_base), self.Cps(T_base)
#                Hsub += (T - T_base)*(Cpg - Cps)
                Hsub += self.Cpg.T_dependent_property_integral(T_base, T) - self.Cps.T_dependent_property_integral(T_base, T)
            except:
                Hsub += (T - T_base)*(self.Cpg - self.Cps)

        return Hsub

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. For
        tabular data, extrapolation outside of the range is used if
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
        if method in (GHARAGHEIZI_HSUB_298, GHARAGHEIZI_HSUB, CRC_HFUS_HVAP_TM, WEBBOOK_HSUB):
            validity = True
        else:
            return super().test_method_validity(T, method)
        return validity


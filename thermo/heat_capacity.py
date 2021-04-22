# -*- coding: utf-8 -*-
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
representing liquid, vapor, and solid heat capacity. A variety of estimation
and data methods are available as included in the `chemicals` library.
Additionally liquid, vapor, and solid mixture heat capacity predictor objects
are implemented subclassing  :obj:`MixtureProperty <thermo.utils.MixtureProperty>`.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.


.. contents:: :local:

Pure Liquid Heat Capacity
=========================
.. autoclass:: HeatCapacityLiquid
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: heat_capacity_liquid_methods

Pure Gas Heat Capacity
======================
.. autoclass:: HeatCapacityGas
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: heat_capacity_gas_methods

Pure Solid Heat Capacity
========================
.. autoclass:: HeatCapacitySolid
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: heat_capacity_solid_methods

Mixture Liquid Heat Capacity
============================
.. autoclass:: HeatCapacityLiquidMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: heat_capacity_liquid_mixture_methods


Mixture Gas Heat Capacity
=========================
.. autoclass:: HeatCapacityGasMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: heat_capacity_gas_mixture_methods

Mixture Solid Heat Capacity
===========================
.. autoclass:: HeatCapacitySolidMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: heat_capacity_solid_mixture_methods

'''

from __future__ import division

__all__ = ['heat_capacity_gas_methods',
           'HeatCapacityGas',
           'heat_capacity_liquid_methods',
           'HeatCapacityLiquid',
           'heat_capacity_solid_methods',
           'HeatCapacitySolid', 'HeatCapacitySolidMixture',
           'HeatCapacityGasMixture', 'HeatCapacityLiquidMixture']
import os
from fluids.numerics import (polyint_over_x, horner_log, horner, polyint,
                             fit_integral_linear_extrapolation,
                             fit_integral_over_T_linear_extrapolation, quad,
                             numpy as np)
from fluids.numerics import brenth, secant, polylog2
from fluids.constants import R, calorie
from chemicals.heat_capacity import *
from chemicals.utils import log, exp, isnan
from chemicals.utils import (to_num, property_molar_to_mass, none_and_length_check,
                          mixing_simple, property_mass_to_molar)
from chemicals import heat_capacity
from chemicals import miscdata
from chemicals.miscdata import lookup_VDI_tabular_data
from thermo import electrochem
from thermo.electrochem import Laliberte_heat_capacity
from thermo.utils import TDependentProperty, MixtureProperty
from thermo.coolprop import *
from cmath import log as clog, exp as cexp
from thermo.utils import VDI_TABULAR, COOLPROP, POLY_FIT, LINEAR

TRCIG = 'TRCIG'
POLING_POLY = 'POLING_POLY'
POLING_CONST = 'POLING_CONST'
CRCSTD = 'CRCSTD'
LASTOVKA_SHAW = 'LASTOVKA_SHAW'
heat_capacity_gas_methods = [COOLPROP, TRCIG, POLING_POLY, LASTOVKA_SHAW, CRCSTD,
                             POLING_CONST, VDI_TABULAR]
'''Holds all methods available for the :obj:`HeatCapacityGas` class, for use in
iterating over them.'''


class HeatCapacityGas(TDependentProperty):
    r'''Class for dealing with gas heat capacity as a function of temperature.
    Consists of two coefficient-based methods, two constant methods,
    one tabular source, one simple estimator, and the external library
    CoolProp.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical
    MW : float, optional
        Molecular weight, [g/mol]
    similarity_variable : float, optional
        similarity variable, n_atoms/MW, [mol/g]
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    poly_fit : tuple(float, float, list[float]), optional
        Tuple of (Tmin, Tmax, coeffs) representing a prefered fit to the
        heat capacity of the compound; the coefficients are evaluated with
        horner's method, and the input variable and output are transformed by
        the default transformations of this object; used instead of any other
        default method if provided. [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    A string holding each method's name is assigned to the following variables
    in this module, intended as the most convenient way to refer to a method.
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_gas_methods`.

    **TRCIG**:
        A rigorous expression derived in [1]_ for modeling gas heat capacity.
        Coefficients for 1961 chemicals are available.
    **POLING_POLY**:
        Simple polynomials in [2]_ not suitable for extrapolation. Data is
        available for 308 chemicals.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [3]_. Very slow.
    **LASTOVKA_SHAW**:
        A basic estimation method using the `similarity variable` concept;
        requires only molecular structure, so is very convenient. See
        :obj:`Lastovka_Shaw <chemicals.heat_capacity.Lastovka_Shaw>` for details.
    **CRCSTD**:
        Constant values tabulated in [4]_ at 298.15 K; data is available for
        533 gases.
    **POLING_CONST**:
        Constant values in [2]_ at 298.15 K; available for 348 gases.
    **VDI_TABULAR**:
        Tabular data up to the critical point available in [5]_. Note that this
        data is along the saturation curve.


    See Also
    --------
    chemicals.heat_capacity.TRCCp
    chemicals.heat_capacity.Lastovka_Shaw
    chemicals.heat_capacity.Rowlinson_Poling
    chemicals.heat_capacity.Rowlinson_Bondi

    Examples
    --------

    >>> CpGas = HeatCapacityGas(CASRN='142-82-5', MW=100.2, similarity_variable=0.2295)
    >>> CpGas(700)
    317.244

    References
    ----------
    .. [1] Kabo, G. J., and G. N. Roganov. Thermodynamics of Organic Compounds
       in the Gas State, Volume II: V. 2. College Station, Tex: CRC Press, 1994.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [3] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    .. [4] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    .. [5] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    name = 'gas heat capacity'
    units = 'J/mol/K'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default; gases are fairly linear in
    heat capacity at high temperatures even if not low temperatures.'''

    property_min = 0
    '''Heat capacities have a minimum value of 0 at 0 K.'''
    property_max = 1E4
    '''Maximum valid of Heat capacity; arbitrarily set. For fluids very near
    the critical point, this value can be obscenely high.'''

    ranked_methods = [TRCIG, POLING_POLY, COOLPROP, LASTOVKA_SHAW, CRCSTD, POLING_CONST, VDI_TABULAR]
    '''Default rankings of the available methods.'''


    _fit_force_n = {}
    '''Dictionary containing method: fit_n, for use in methods which should
    only ever be fit to a specific `n` value'''
    _fit_force_n[CRCSTD] = 1
    _fit_force_n[POLING_CONST] = 1

    custom_args = ('MW', 'similarity_variable')

    def __init__(self, CASRN='', MW=None, similarity_variable=None,
                 extrapolation='linear', **kwargs):
        self.CASRN = CASRN
        self.MW = MW
        self.similarity_variable = similarity_variable
        super(HeatCapacityGas, self).__init__(extrapolation, **kwargs)


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
        Tmins, Tmaxs = [], []
        self.T_limits = T_limits = {}
        if load_data:
            if self.CASRN in heat_capacity.TRC_gas_data.index:
                methods.append(TRCIG)
                self.TRCIG_Tmin, self.TRCIG_Tmax, a0, a1, a2, a3, a4, a5, a6, a7, _, _, _ = heat_capacity.TRC_gas_values[heat_capacity.TRC_gas_data.index.get_loc(self.CASRN)].tolist()
                self.TRCIG_coefs = [a0, a1, a2, a3, a4, a5, a6, a7]
                Tmins.append(self.TRCIG_Tmin); Tmaxs.append(self.TRCIG_Tmax)
                T_limits[TRCIG] = (self.TRCIG_Tmin, self.TRCIG_Tmax)
            if self.CASRN in heat_capacity.Cp_data_Poling.index and not isnan(heat_capacity.Cp_data_Poling.at[self.CASRN, 'a0']):
                POLING_Tmin, POLING_Tmax, a0, a1, a2, a3, a4, Cpg, Cpl = heat_capacity.Cp_values_Poling[heat_capacity.Cp_data_Poling.index.get_loc(self.CASRN)].tolist()
                methods.append(POLING_POLY)
                if isnan(POLING_Tmin):
                    POLING_Tmin = 50.0
                if isnan(POLING_Tmax):
                    POLING_Tmax = 1000.0
                self.POLING_Tmin = POLING_Tmin
                Tmins.append(POLING_Tmin)
                self.POLING_Tmax = POLING_Tmax
                Tmaxs.append(POLING_Tmax)
                self.POLING_coefs = [a0, a1, a2, a3, a4]
                T_limits[POLING_POLY] = (POLING_Tmin, POLING_Tmax)
            if self.CASRN in heat_capacity.Cp_data_Poling.index and not isnan(heat_capacity.Cp_data_Poling.at[self.CASRN, 'Cpg']):
                methods.append(POLING_CONST)
                self.POLING_T = 298.15
                self.POLING_constant = float(heat_capacity.Cp_data_Poling.at[self.CASRN, 'Cpg'])
                T_limits[POLING_CONST] = (self.POLING_T-50.0, self.POLING_T+50.0)
            if self.CASRN in heat_capacity.CRC_standard_data.index and not isnan(heat_capacity.CRC_standard_data.at[self.CASRN, 'Cpg']):
                methods.append(CRCSTD)
                self.CRCSTD_T = 298.15
                self.CRCSTD_constant = float(heat_capacity.CRC_standard_data.at[self.CASRN, 'Cpg'])
                T_limits[CRCSTD] = (self.CRCSTD_T-50.0, self.CRCSTD_T+50.0)
            if self.CASRN in miscdata.VDI_saturation_dict:
                # NOTE: VDI data is for the saturation curve, i.e. at increasing
                # pressure; it is normally substantially higher than the ideal gas
                # value
                methods.append(VDI_TABULAR)
                Ts, props = lookup_VDI_tabular_data(self.CASRN, 'Cp (g)')
                self.VDI_Tmin = Ts[0]
                self.VDI_Tmax = Ts[-1]
                self.tabular_data[VDI_TABULAR] = (Ts, props)
                Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
                T_limits[VDI_TABULAR] = (self.VDI_Tmin, self.VDI_Tmax)
            if has_CoolProp() and self.CASRN in coolprop_dict:
                methods.append(COOLPROP)
                self.CP_f = coolprop_fluids[self.CASRN]
                Tmin = max(self.CP_f.Tt, self.CP_f.Tmin)
                Tmax = min(self.CP_f.Tc, self.CP_f.Tmax)
                Tmins.append(Tmin); Tmaxs.append(Tmax)
                T_limits[COOLPROP] = (Tmin, Tmax)
        if self.MW and self.similarity_variable:
            methods.append(LASTOVKA_SHAW)
            T_limits[LASTOVKA_SHAW] = (1e-3, 1e5)
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {TRCIG: heat_capacity.TRC_gas_data.index,
                POLING_POLY: [i for i in heat_capacity.Cp_data_Poling.index if not isnan(heat_capacity.Cp_data_Poling.at[i, 'a0'])],
                POLING_CONST: [i for i in heat_capacity.Cp_data_Poling.index if not isnan(heat_capacity.Cp_data_Poling.at[i, 'Cpg'])],
                CRCSTD: [i for i in heat_capacity.CRC_standard_data.index if not isnan(heat_capacity.CRC_standard_data.at[i, 'Cpg'])],
                COOLPROP: coolprop_dict,
                VDI_TABULAR: list(miscdata.VDI_saturation_dict.keys()),
                }

    def calculate(self, T, method):
        r'''Method to calculate surface tension of a liquid at temperature `T`
        with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate heat capacity, [K]
        method : str
            Method name to use

        Returns
        -------
        Cp : float
            Calculated heat capacity, [J/mol/K]
        '''
        if method == POLY_FIT:
            if T < self.poly_fit_Tmin:
                Cp = (T - self.poly_fit_Tmin)*self.poly_fit_Tmin_slope + self.poly_fit_Tmin_value
            elif T > self.poly_fit_Tmax:
                Cp = (T - self.poly_fit_Tmax)*self.poly_fit_Tmax_slope + self.poly_fit_Tmax_value
            else:
                Cp = horner(self.poly_fit_coeffs, T)
        elif method == TRCIG:
            Cp = TRCCp(T, *self.TRCIG_coefs)
        elif method == COOLPROP:
            try:
                # Some cases due to melting point need a high pressure
                Cp = PropsSI('Cp0molar', 'T', T,'P', 10132500.0, self.CASRN)
            except:
                # And some cases don't converge at high P
                Cp = PropsSI('Cp0molar', 'T', T,'P', 101325.0, self.CASRN)
        elif method == POLING_POLY:
            Cp = R*(self.POLING_coefs[0] + self.POLING_coefs[1]*T
            + self.POLING_coefs[2]*T**2 + self.POLING_coefs[3]*T**3
            + self.POLING_coefs[4]*T**4)
        elif method == POLING_CONST:
            Cp = self.POLING_constant
        elif method == CRCSTD:
            Cp = self.CRCSTD_constant
        elif method == LASTOVKA_SHAW:
            Cp = Lastovka_Shaw(T, self.similarity_variable)
            Cp = property_mass_to_molar(Cp, self.MW)
        else:
            return self._base_calculate(T, method)
        return Cp

    def test_method_validity(self, T, method):
        r'''Method to test the validity of a specified method for a given
        temperature.

        'TRC' and 'Poling' both have minimum and maimum temperatures. The
        constant temperatures in POLING_CONST and CRCSTD are considered valid
        for 50 degrees around their specified temperatures.
        :obj:`Lastovka_Shaw <chemicals.heat_capacity.Lastovka_Shaw>` is considered valid for the whole range of
        temperatures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        Parameters
        ----------
        T : float
            Temperature at which to determine the validity of the method, [K]
        method : str
            Name of the method to test

        Returns
        -------
        validity : bool
            Whether or not a specifid method is valid
        '''
        validity = True
        if method == TRCIG:
            if T < self.TRCIG_Tmin or T > self.TRCIG_Tmax:
                return False
        elif method == POLING_POLY:
            if T < self.POLING_Tmin or T > self.POLING_Tmax:
                return False
        elif method == POLING_CONST:
            if T > self.POLING_T + 50.0 or T < self.POLING_T - 50.0:
                return False
        elif method == CRCSTD:
            if T > self.CRCSTD_T + 50.0 or T < self.CRCSTD_T - 50.0:
                return False
        elif method == LASTOVKA_SHAW:
            pass # Valid everywhere
        elif method == COOLPROP:
            if T <= self.CP_f.Tmin or T >= self.CP_f.Tmax:
                return False
        elif method == POLY_FIT:
            validity = True
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity

    def calculate_integral(self, T1, T2, method):
        r'''Method to calculate the integral of a property with respect to
        temperature, using a specified method. Implements the analytical
        integrals of all available methods except for tabular data.

        Parameters
        ----------
        T1 : float
            Lower limit of integration, [K]
        T2 : float
            Upper limit of integration, [K]
        method : str
            Method for which to find the integral

        Returns
        -------
        integral : float
            Calculated integral of the property over the given range,
            [`units*K`]
        '''
        if method == POLY_FIT:
            return fit_integral_linear_extrapolation(T1, T2,
                self.poly_fit_int_coeffs, self.poly_fit_Tmin,
                self.poly_fit_Tmax, self.poly_fit_Tmin_value,
                self.poly_fit_Tmax_value, self.poly_fit_Tmin_slope,
                self.poly_fit_Tmax_slope)
        elif method == TRCIG:
            H2 = TRCCp_integral(T2, *self.TRCIG_coefs)
            H1 = TRCCp_integral(T1, *self.TRCIG_coefs)
            return H2 - H1
        elif method == POLING_POLY:
            A, B, C, D, E = self.POLING_coefs
            H2 = (((((0.2*E)*T2 + 0.25*D)*T2 + C/3.)*T2 + 0.5*B)*T2 + A)*T2
            H1 = (((((0.2*E)*T1 + 0.25*D)*T1 + C/3.)*T1 + 0.5*B)*T1 + A)*T1
            return R*(H2 - H1)
        elif method == POLING_CONST:
            return (T2 - T1)*self.POLING_constant
        elif method == CRCSTD:
            return (T2 - T1)*self.CRCSTD_constant
        elif method == LASTOVKA_SHAW:
            dH = (Lastovka_Shaw_integral(T2, self.similarity_variable)
                    - Lastovka_Shaw_integral(T1, self.similarity_variable))
            return property_mass_to_molar(dH, self.MW)
        else:
            return super(HeatCapacityGas, self).calculate_integral(T1, T2, method)


    def calculate_integral_over_T(self, T1, T2, method):
        r'''Method to calculate the integral of a property over temperature
        with respect to temperature, using a specified method. Implements the
        analytical integrals of all available methods except for tabular data.

        Parameters
        ----------
        T1 : float
            Lower limit of integration, [K]
        T2 : float
            Upper limit of integration, [K]
        method : str
            Method for which to find the integral

        Returns
        -------
        integral : float
            Calculated integral of the property over the given range,
            [`units`]
        '''
        if method == POLY_FIT:
            return fit_integral_over_T_linear_extrapolation(T1, T2,
                self.poly_fit_T_int_T_coeffs, self.poly_fit_log_coeff,
                self.poly_fit_Tmin, self.poly_fit_Tmax,
                self.poly_fit_Tmin_value,
                self.poly_fit_Tmax_value, self.poly_fit_Tmin_slope,
                self.poly_fit_Tmax_slope)
        elif method == TRCIG:
            S2 = TRCCp_integral_over_T(T2, *self.TRCIG_coefs)
            S1 = TRCCp_integral_over_T(T1, *self.TRCIG_coefs)
            return S2 - S1
        elif method == CRCSTD:
            return self.CRCSTD_constant*log(T2/T1)
        elif method == POLING_CONST:
            return self.POLING_constant*log(T2/T1)
        elif method == POLING_POLY:
            A, B, C, D, E = self.POLING_coefs
            S2 = ((((0.25*E)*T2 + D/3.)*T2 + 0.5*C)*T2 + B)*T2
            S1 = ((((0.25*E)*T1 + D/3.)*T1 + 0.5*C)*T1 + B)*T1
            return R*(S2-S1 + A*log(T2/T1))
        elif method == LASTOVKA_SHAW:
            dS = (Lastovka_Shaw_integral_over_T(T2, self.similarity_variable)
                 - Lastovka_Shaw_integral_over_T(T1, self.similarity_variable))
            return property_mass_to_molar(dS, self.MW)
        return super(HeatCapacityGas, self).calculate_integral_over_T(T1, T2, method)




ZABRANSKY_SPLINE = 'ZABRANSKY_SPLINE'
ZABRANSKY_QUASIPOLYNOMIAL = 'ZABRANSKY_QUASIPOLYNOMIAL'
ZABRANSKY_SPLINE_C = 'ZABRANSKY_SPLINE_C'
ZABRANSKY_QUASIPOLYNOMIAL_C = 'ZABRANSKY_QUASIPOLYNOMIAL_C'
ZABRANSKY_SPLINE_SAT = 'ZABRANSKY_SPLINE_SAT'
ZABRANSKY_QUASIPOLYNOMIAL_SAT = 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'
ROWLINSON_POLING = 'ROWLINSON_POLING'
ROWLINSON_BONDI = 'ROWLINSON_BONDI'
DADGOSTAR_SHAW = 'DADGOSTAR_SHAW'

heat_capacity_liquid_methods = [ZABRANSKY_SPLINE, ZABRANSKY_QUASIPOLYNOMIAL,
                      ZABRANSKY_SPLINE_C, ZABRANSKY_QUASIPOLYNOMIAL_C,
                      ZABRANSKY_SPLINE_SAT, ZABRANSKY_QUASIPOLYNOMIAL_SAT,
                      VDI_TABULAR, ROWLINSON_POLING, ROWLINSON_BONDI, COOLPROP,
                      DADGOSTAR_SHAW, POLING_CONST, CRCSTD]
'''Holds all methods available for the :obj:`HeatCapacityLiquid class`, for use in
iterating over them.'''


class HeatCapacityLiquid(TDependentProperty):
    r'''Class for dealing with liquid heat capacity as a function of temperature.
    Consists of six coefficient-based methods, two constant methods,
    one tabular source, two CSP methods based on gas heat capacity, one simple
    estimator, and the external library CoolProp.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical
    MW : float, optional
        Molecular weight, [g/mol]
    similarity_variable : float, optional
        similarity variable, n_atoms/MW, [mol/g]
    Tc : float, optional
        Critical temperature, [K]
    omega : float, optional
        Acentric factor, [-]
    Cpgm : float or callable, optional
        Idea-gas molar heat capacity at T or callable for the same, [J/mol/K]
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    poly_fit : tuple(float, float, list[float]), optional
        Tuple of (Tmin, Tmax, coeffs) representing a prefered fit to the
        heat capacity of the compound; the coefficients are evaluated with
        horner's method, and the input variable and output are transformed by
        the default transformations of this object; used instead of any other
        default method if provided. [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    A string holding each method's name is assigned to the following variables
    in this module, intended as the most convenient way to refer to a method.
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_liquid_methods`.

    **ZABRANSKY_SPLINE, ZABRANSKY_QUASIPOLYNOMIAL, ZABRANSKY_SPLINE_C,
    and ZABRANSKY_QUASIPOLYNOMIAL_C**:

        Rigorous expressions developed in [1]_ following critical evaluation
        of the available data. The spline methods use the form described in
        :obj:`Zabransky_cubic <chemicals.heat_capacity.Zabransky_cubic>` over short ranges with varying coefficients
        to obtain a wider range. The quasi-polynomial methods use the form
        described in :obj:`Zabransky_quasi_polynomial <chemicals.heat_capacity.Zabransky_quasi_polynomial>`, more suitable for
        extrapolation, and over then entire range. Respectively, there is data
        available for 588, 146, 51, and 26 chemicals. 'C' denotes constant-
        pressure data available from more precise experiments. The others
        are heat capacity values averaged over a temperature changed.

    **ZABRANSKY_SPLINE_SAT and ZABRANSKY_QUASIPOLYNOMIAL_SAT**:

        Rigorous expressions developed in [1]_ following critical evaluation
        of the available data. The spline method use the form described in
        :obj:`Zabransky_cubic <chemicals.heat_capacity.Zabransky_cubic>` over short ranges with varying coefficients
        to obtain a wider range. The quasi-polynomial method use the form
        described in :obj:`Zabransky_quasi_polynomial <chemicals.heat_capacity.Zabransky_quasi_polynomial>`, more suitable for
        extrapolation, and over their entire range. Respectively, there is data
        available for 203, and 16 chemicals. Note that these methods are for
        the saturation curve!

    **VDI_TABULAR**:

        Tabular data up to the critical point available in [5]_. Note that this
        data is along the saturation curve.

    **ROWLINSON_POLING**:

        CSP method described in :obj:`Rowlinson_Poling <chemicals.heat_capacity.Rowlinson_Poling>`. Requires a ideal gas
        heat capacity value at the same temperature as it is to be calculated.

    **ROWLINSON_BONDI**:

        CSP method described in :obj:`Rowlinson_Bondi <chemicals.heat_capacity.Rowlinson_Bondi>`. Requires a ideal gas
        heat capacity value at the same temperature as it is to be calculated.

    **COOLPROP**:

        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [3]_. Very slow.

    **DADGOSTAR_SHAW**:

        A basic estimation method using the `similarity variable` concept;
        requires only molecular structure, so is very convenient. See
        :obj:`Dadgostar_Shaw <chemicals.heat_capacity.Dadgostar_Shaw>` for details.

    **POLING_CONST**:

        Constant values in [2]_ at 298.15 K; available for 245 liquids.

    **CRCSTD**:

        Constant values tabulated in [4]_ at 298.15 K; data is available for 433
        liquids.

    See Also
    --------
    chemicals.heat_capacity.Zabransky_quasi_polynomial
    chemicals.heat_capacity.Zabransky_cubic
    chemicals.heat_capacity.Rowlinson_Poling
    chemicals.heat_capacity.Rowlinson_Bondi
    chemicals.heat_capacity.Dadgostar_Shaw

    Examples
    --------
    >>> CpLiquid = HeatCapacityLiquid(CASRN='142-82-5', MW=100.2, similarity_variable=0.2295, Tc=540.2, omega=0.3457, Cpgm=165.2)

    References
    ----------
    .. [1] Zabransky, M., V. Ruzicka Jr, V. Majer, and Eugene S. Domalski.
       Heat Capacity of Liquids: Critical Review and Recommended Values.
       2 Volume Set. Washington, D.C.: Amer Inst of Physics, 1996.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [3] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    .. [4] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    .. [5] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    name = 'Liquid heat capacity'
    units = 'J/mol/K'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = False
    '''Disallow tabular extrapolation by default; higher-temeprature behavior
    is not well predicted by most extrapolation.'''

    property_min = 1
    '''Allow very low heat capacities; arbitrarily set; liquid heat capacity
    should always be somewhat substantial.'''
    property_max = 1E4 # Originally 1E4
    '''Maximum valid of Heat capacity; arbitrarily set. For fluids very near
    the critical point, this value can be obscenely high.'''


    ranked_methods = [ZABRANSKY_SPLINE, ZABRANSKY_QUASIPOLYNOMIAL,
                      ZABRANSKY_SPLINE_C, ZABRANSKY_QUASIPOLYNOMIAL_C,
                      ZABRANSKY_SPLINE_SAT, ZABRANSKY_QUASIPOLYNOMIAL_SAT,
                      VDI_TABULAR, COOLPROP, DADGOSTAR_SHAW, ROWLINSON_POLING,
                      ROWLINSON_BONDI,
                      POLING_CONST, CRCSTD]
    '''Default rankings of the available methods.'''

    _fit_force_n = {}
    '''Dictionary containing method: fit_n, for use in methods which should
    only ever be fit to a specific `n` value'''
    _fit_force_n[CRCSTD] = 1
    _fit_force_n[POLING_CONST] = 1

    _json_obj_by_CAS = ('Zabransky_spline', 'Zabransky_spline_iso', 'Zabransky_spline_sat',
                        'Zabransky_quasipolynomial', 'Zabransky_quasipolynomial_iso',
                        'Zabransky_quasipolynomial_sat', 'CP_f')

    obj_references = pure_references = ('Cpgm',)
    obj_references_types = pure_reference_types = (HeatCapacityGas,)


    custom_args = ('MW', 'similarity_variable', 'Tc', 'omega', 'Cpgm')
    def __init__(self, CASRN='', MW=None, similarity_variable=None, Tc=None,
                 omega=None, Cpgm=None, extrapolation='linear',  **kwargs):
        self.CASRN = CASRN
        self.MW = MW
        self.Tc = Tc
        self.omega = omega
        self.Cpgm = Cpgm
        self.similarity_variable = similarity_variable
        super(HeatCapacityLiquid, self).__init__(extrapolation, **kwargs)

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {ZABRANSKY_SPLINE: list(heat_capacity.zabransky_dict_const_s),
                ZABRANSKY_QUASIPOLYNOMIAL: list(heat_capacity.zabransky_dict_const_p),
                ZABRANSKY_SPLINE_C: list(heat_capacity.zabransky_dict_iso_s),
                ZABRANSKY_QUASIPOLYNOMIAL_C: list(heat_capacity.zabransky_dict_iso_p),
                ZABRANSKY_SPLINE_SAT: list(heat_capacity.zabransky_dict_sat_s),
                ZABRANSKY_QUASIPOLYNOMIAL_SAT: list(heat_capacity.zabransky_dict_sat_p),
                POLING_CONST: [i for i in heat_capacity.Cp_data_Poling.index if not isnan(heat_capacity.Cp_data_Poling.at[i, 'Cpl'])],
                CRCSTD: [i for i in heat_capacity.CRC_standard_data.index if not isnan(heat_capacity.CRC_standard_data.at[i, 'Cpl'])],
                COOLPROP: coolprop_dict,
                VDI_TABULAR: list(miscdata.VDI_saturation_dict.keys()),
                }

    @classmethod
    def _load_json_CAS_references(cls, d):
        CASRN = d['CASRN']
        if CASRN in heat_capacity.zabransky_dict_const_s:
            d['Zabransky_spline'] = heat_capacity.zabransky_dict_const_s[CASRN]
        if CASRN in heat_capacity.zabransky_dict_const_p:
            d['Zabransky_quasipolynomial'] = heat_capacity.zabransky_dict_const_p[CASRN]
        if CASRN in heat_capacity.zabransky_dict_iso_s:
            d['Zabransky_spline_iso'] = heat_capacity.zabransky_dict_iso_s[CASRN]
        if CASRN in heat_capacity.zabransky_dict_iso_p:
            d['Zabransky_quasipolynomial_iso'] = heat_capacity.zabransky_dict_iso_p[CASRN]
        if CASRN in heat_capacity.zabransky_dict_sat_s:
            d['Zabransky_spline_sat'] = heat_capacity.zabransky_dict_sat_s[CASRN]
        if CASRN in heat_capacity.zabransky_dict_sat_p:
            d['Zabransky_quasipolynomial_sat'] = heat_capacity.zabransky_dict_sat_p[CASRN]
        if 'CP_f' in d:
            d['CP_f'] = coolprop_fluids[CASRN]

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
        Tmins, Tmaxs = [], []
        self.T_limits = T_limits = {}
        if load_data:
            if self.CASRN in heat_capacity.zabransky_dict_const_s:
                methods.append(ZABRANSKY_SPLINE)
                self.Zabransky_spline = heat_capacity.zabransky_dict_const_s[self.CASRN]
                T_limits[ZABRANSKY_SPLINE] = (self.Zabransky_spline.Tmin, self.Zabransky_spline.Tmax)
            if self.CASRN in heat_capacity.zabransky_dict_const_p:
                methods.append(ZABRANSKY_QUASIPOLYNOMIAL)
                self.Zabransky_quasipolynomial = heat_capacity.zabransky_dict_const_p[self.CASRN]
                T_limits[ZABRANSKY_QUASIPOLYNOMIAL] = (self.Zabransky_quasipolynomial.Tmin, self.Zabransky_quasipolynomial.Tmax)
            if self.CASRN in heat_capacity.zabransky_dict_iso_s:
                methods.append(ZABRANSKY_SPLINE_C)
                self.Zabransky_spline_iso = heat_capacity.zabransky_dict_iso_s[self.CASRN]
                T_limits[ZABRANSKY_SPLINE_C] = (self.Zabransky_spline_iso.Tmin, self.Zabransky_spline_iso.Tmax)
            if self.CASRN in heat_capacity.zabransky_dict_iso_p:
                methods.append(ZABRANSKY_QUASIPOLYNOMIAL_C)
                self.Zabransky_quasipolynomial_iso = heat_capacity.zabransky_dict_iso_p[self.CASRN]
                T_limits[ZABRANSKY_QUASIPOLYNOMIAL_C] = (self.Zabransky_quasipolynomial_iso.Tmin, self.Zabransky_quasipolynomial_iso.Tmax)


            if self.CASRN in heat_capacity.Cp_data_Poling.index and not isnan(heat_capacity.Cp_data_Poling.at[self.CASRN, 'Cpl']):
                methods.append(POLING_CONST)
                self.POLING_T = 298.15
                self.POLING_constant = float(heat_capacity.Cp_data_Poling.at[self.CASRN, 'Cpl'])
                T_limits[POLING_CONST] = (298.15-50.0, 298.15+50.0)
            if self.CASRN in heat_capacity.CRC_standard_data.index and not isnan(heat_capacity.CRC_standard_data.at[self.CASRN, 'Cpl']):
                methods.append(CRCSTD)
                self.CRCSTD_T = 298.15
                self.CRCSTD_constant = float(heat_capacity.CRC_standard_data.at[self.CASRN, 'Cpl'])
                T_limits[CRCSTD] = (298.15-50.0, 298.15+50.0)
            # Saturation functions
            if self.CASRN in heat_capacity.zabransky_dict_sat_s:
                methods.append(ZABRANSKY_SPLINE_SAT)
                self.Zabransky_spline_sat = heat_capacity.zabransky_dict_sat_s[self.CASRN]
                T_limits[ZABRANSKY_SPLINE_SAT] = (self.Zabransky_spline_sat.Tmin, self.Zabransky_spline_sat.Tmax)
            if self.CASRN in heat_capacity.zabransky_dict_sat_p:
                methods.append(ZABRANSKY_QUASIPOLYNOMIAL_SAT)
                self.Zabransky_quasipolynomial_sat = heat_capacity.zabransky_dict_sat_p[self.CASRN]
                T_limits[ZABRANSKY_QUASIPOLYNOMIAL_SAT] = (self.Zabransky_quasipolynomial_sat.Tmin, self.Zabransky_quasipolynomial_sat.Tmax)
            if self.CASRN in miscdata.VDI_saturation_dict:
                # NOTE: VDI data is for the saturation curve, i.e. at increasing
                # pressure; it is normally substantially higher than the ideal gas
                # value
                methods.append(VDI_TABULAR)
                Ts, props = lookup_VDI_tabular_data(self.CASRN, 'Cp (l)')
                self.VDI_Tmin = Ts[0]
                self.VDI_Tmax = Ts[-1]
                self.tabular_data[VDI_TABULAR] = (Ts, props)
                Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
                T_limits[VDI_TABULAR] = (self.VDI_Tmin, self.VDI_Tmax)
            if has_CoolProp() and self.CASRN in coolprop_dict:
                methods.append(COOLPROP)
                self.CP_f = coolprop_fluids[self.CASRN]
                Tmin = max(self.CP_f.Tt, self.CP_f.Tmin)
                Tmax = min(self.CP_f.Tc*.9999, self.CP_f.Tmax)
                Tmins.append(Tmin); Tmaxs.append(Tmax)
                T_limits[COOLPROP] = (Tmin, Tmax)
        if self.Tc and self.omega:
            methods.extend([ROWLINSON_POLING, ROWLINSON_BONDI])
            limits_Tc = (0.3*self.Tc, self.Tc-0.1)
            T_limits[ROWLINSON_POLING] = limits_Tc
            T_limits[ROWLINSON_BONDI] = limits_Tc
        if self.MW and self.similarity_variable:
            methods.append(DADGOSTAR_SHAW)
            T_limits[DADGOSTAR_SHAW] = (1e-3, self.Tc if self.Tc is not None else 10000)
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            # TODO: More Tmin, Tmax ranges
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)


    def calculate(self, T, method):
        r'''Method to calculate heat capacity of a liquid at temperature `T`
        with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate heat capacity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Cp : float
            Heat capacity of the liquid at T, [J/mol/K]
        '''
        if method == POLY_FIT:
            if T < self.poly_fit_Tmin:
                return (T - self.poly_fit_Tmin)*self.poly_fit_Tmin_slope + self.poly_fit_Tmin_value
            elif T > self.poly_fit_Tmax:
                return (T - self.poly_fit_Tmax)*self.poly_fit_Tmax_slope + self.poly_fit_Tmax_value
            else:
                return horner(self.poly_fit_coeffs, T)
        elif method == ZABRANSKY_SPLINE:
            return self.Zabransky_spline.force_calculate(T)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL:
            return self.Zabransky_quasipolynomial.calculate(T)
        elif method == ZABRANSKY_SPLINE_C:
            return self.Zabransky_spline_iso.force_calculate(T)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_C:
            return self.Zabransky_quasipolynomial_iso.calculate(T)
        elif method == ZABRANSKY_SPLINE_SAT:
            return self.Zabransky_spline_sat.force_calculate(T)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_SAT:
            return self.Zabransky_quasipolynomial_sat.calculate(T)
        elif method == COOLPROP:
            return CoolProp_T_dependent_property(T, self.CASRN , 'CPMOLAR', 'l')
        elif method == POLING_CONST:
            return self.POLING_constant
        elif method == CRCSTD:
            return self.CRCSTD_constant
        elif method == ROWLINSON_POLING:
            Cpgm = self.Cpgm(T) if hasattr(self.Cpgm, '__call__') else self.Cpgm
            return Rowlinson_Poling(T, self.Tc, self.omega, Cpgm)
        elif method == ROWLINSON_BONDI:
            Cpgm = self.Cpgm(T) if hasattr(self.Cpgm, '__call__') else self.Cpgm
            return Rowlinson_Bondi(T, self.Tc, self.omega, Cpgm)
        elif method == DADGOSTAR_SHAW:
            Cp = Dadgostar_Shaw(T, self.similarity_variable)
            return property_mass_to_molar(Cp, self.MW)
        else:
            return self._base_calculate(T, method)

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For the CSP method
        :obj:`Rowlinson_Poling <chemicals.heat_capacity.Rowlinson_Poling>`, the model is considered valid for all
        temperatures. The simple method :obj:`Dadgostar_Shaw <chemicals.heat_capacity.Dadgostar_Shaw>` is considered
        valid for all temperatures. For tabular data,
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
        if method == ZABRANSKY_SPLINE:
            if T < self.Zabransky_spline.Tmin or T > self.Zabransky_spline.Tmax:
                return False
        elif method == ZABRANSKY_SPLINE_C:
            if T < self.Zabransky_spline_iso.Tmin or T > self.Zabransky_spline_iso.Tmax:
                return False
        elif method == ZABRANSKY_SPLINE_SAT:
            if T < self.Zabransky_spline_sat.Tmin or T > self.Zabransky_spline_sat.Tmax:
                return False
        elif method == ZABRANSKY_QUASIPOLYNOMIAL:
            if T > self.Zabransky_quasipolynomial.Tc:
                return False
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_C:
            if T > self.Zabransky_quasipolynomial_iso.Tc:
                return False
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_SAT:
            if T > self.Zabransky_quasipolynomial_sat.Tc:
                return False
        elif method == COOLPROP:
            if T <= self.CP_f.Tmin or T >= self.CP_f.Tmax:
                return False
        elif method == POLING_CONST:
            if T > self.POLING_T + 50 or T < self.POLING_T - 50:
                return False
        elif method == CRCSTD:
            if T > self.CRCSTD_T + 50 or T < self.CRCSTD_T - 50:
                return False
        elif method == DADGOSTAR_SHAW:
            pass # Valid everywhere
        elif method in [ROWLINSON_POLING, ROWLINSON_BONDI]:
            if self.Tc and T > self.Tc:
                return False
        elif method == POLY_FIT:
            validity = True
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity

    def calculate_integral(self, T1, T2, method):
        r'''Method to calculate the integral of a property with respect to
        temperature, using a specified method.  Implements the
        analytical integrals of all available methods except for tabular data,
        the case of multiple coefficient sets needed to encompass the temperature
        range of any of the ZABRANSKY methods, and the CSP methods using the
        vapor phase properties.

        Parameters
        ----------
        T1 : float
            Lower limit of integration, [K]
        T2 : float
            Upper limit of integration, [K]
        method : str
            Method for which to find the integral

        Returns
        -------
        integral : float
            Calculated integral of the property over the given range,
            [`units*K`]
        '''
        if method == POLY_FIT:
            return fit_integral_linear_extrapolation(T1, T2,
                self.poly_fit_int_coeffs, self.poly_fit_Tmin,
                self.poly_fit_Tmax, self.poly_fit_Tmin_value,
                self.poly_fit_Tmax_value, self.poly_fit_Tmin_slope,
                self.poly_fit_Tmax_slope)
        elif method == ZABRANSKY_SPLINE:
            return self.Zabransky_spline.calculate_integral(T1, T2)
        elif method == ZABRANSKY_SPLINE_C:
            return self.Zabransky_spline_iso.force_calculate_integral(T1, T2)
        elif method == ZABRANSKY_SPLINE_SAT:
            return self.Zabransky_spline_sat.calculate_integral(T1, T2)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL:
            return self.Zabransky_quasipolynomial.calculate_integral(T1, T2)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_C:
            return self.Zabransky_quasipolynomial_iso.calculate_integral(T1, T2)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_SAT:
            return self.Zabransky_quasipolynomial_sat.calculate_integral(T1, T2)
        elif method == POLING_CONST:
            return (T2 - T1)*self.POLING_constant
        elif method == CRCSTD:
            return (T2 - T1)*self.CRCSTD_constant
        elif method == DADGOSTAR_SHAW:
            dH = (Dadgostar_Shaw_integral(T2, self.similarity_variable)
                    - Dadgostar_Shaw_integral(T1, self.similarity_variable))
            return property_mass_to_molar(dH, self.MW)
        elif method in self.tabular_data or method == COOLPROP or method in [ROWLINSON_POLING, ROWLINSON_BONDI]:
            return float(quad(self.calculate, T1, T2, args=(method,))[0])
        return super(HeatCapacityLiquid, self).calculate_integral(T1, T2, method)

    def calculate_integral_over_T(self, T1, T2, method):
        r'''Method to calculate the integral of a property over temperature
        with respect to temperature, using a specified method.   Implements the
        analytical integrals of all available methods except for tabular data,
        the case of multiple coefficient sets needed to encompass the temperature
        range of any of the ZABRANSKY methods, and the CSP methods using the
        vapor phase properties.

        Parameters
        ----------
        T1 : float
            Lower limit of integration, [K]
        T2 : float
            Upper limit of integration, [K]
        method : str
            Method for which to find the integral

        Returns
        -------
        integral : float
            Calculated integral of the property over the given range,
            [`units`]
        '''
        if method == POLY_FIT:
            return fit_integral_over_T_linear_extrapolation(T1, T2,
                self.poly_fit_T_int_T_coeffs, self.poly_fit_log_coeff,
                self.poly_fit_Tmin, self.poly_fit_Tmax,
                self.poly_fit_Tmin_value,
                self.poly_fit_Tmax_value, self.poly_fit_Tmin_slope,
                self.poly_fit_Tmax_slope)
        elif method == ZABRANSKY_SPLINE:
            return self.Zabransky_spline.calculate_integral_over_T(T1, T2)
        elif method == ZABRANSKY_SPLINE_C:
            return self.Zabransky_spline_iso.calculate_integral_over_T(T1, T2)
        elif method == ZABRANSKY_SPLINE_SAT:
            return self.Zabransky_spline_sat.calculate_integral_over_T(T1, T2)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL:
            return self.Zabransky_quasipolynomial.calculate_integral_over_T(T1, T2)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_C:
            return self.Zabransky_quasipolynomial_iso.calculate_integral_over_T(T1, T2)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_SAT:
            return self.Zabransky_quasipolynomial_sat.calculate_integral_over_T(T1, T2)
        elif method == POLING_CONST:
            return self.POLING_constant*log(T2/T1)
        elif method == CRCSTD:
            return self.CRCSTD_constant*log(T2/T1)
        elif method == DADGOSTAR_SHAW:
            dS = (Dadgostar_Shaw_integral_over_T(T2, self.similarity_variable)
                    - Dadgostar_Shaw_integral_over_T(T1, self.similarity_variable))
            return property_mass_to_molar(dS, self.MW)
        elif method in self.tabular_data or method == COOLPROP or method in [ROWLINSON_POLING, ROWLINSON_BONDI]:
            return float(quad(lambda T: self.calculate(T, method)/T, T1, T2)[0])
        return super(HeatCapacityLiquid, self).calculate_integral_over_T(T1, T2, method)

LASTOVKA_S = 'LASTOVKA_S'
PERRY151 = '''PERRY151'''
heat_capacity_solid_methods = [PERRY151, CRCSTD, LASTOVKA_S]
'''Holds all methods available for the :obj:`HeatCapacitySolid` class, for use in
iterating over them.'''


class HeatCapacitySolid(TDependentProperty):
    r'''Class for dealing with solid heat capacity as a function of temperature.
    Consists of one temperature-dependent simple expression, one constant
    value source, and one simple estimator.

    Parameters
    ----------
    similarity_variable : float, optional
        similarity variable, n_atoms/MW, [mol/g]
    MW : float, optional
        Molecular weight, [g/mol]
    CASRN : str, optional
        The CAS number of the chemical
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    poly_fit : tuple(float, float, list[float]), optional
        Tuple of (Tmin, Tmax, coeffs) representing a prefered fit to the
        heat capacity of the compound; the coefficients are evaluated with
        horner's method, and the input variable and output are transformed by
        the default transformations of this object; used instead of any other
        default method if provided. [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    A string holding each method's name is assigned to the following variables
    in this module, intended as the most convenient way to refer to a method.
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_solid_methods`.

    **PERRY151**:
        Simple polynomials with vaious exponents selected for each expression.
        Coefficients are in units of calories/mol/K. The full expression is:

        .. math::
            Cp = a + bT + c/T^2 + dT^2

        Data is available for 284 solids, from [2]_.

    **CRCSTD**:
        Values tabulated in [1]_ at 298.15 K; data is available for 529
        solids.
    **LASTOVKA_S**:
        A basic estimation method using the `similarity variable` concept;
        requires only molecular structure, so is very convenient. See
        :obj:`Lastovka_solid <chemicals.heat_capacity.Lastovka_solid>` for details.

    See Also
    --------
    chemicals.heat_capacity.Lastovka_solid

    Examples
    --------
    >>> CpSolid = HeatCapacitySolid(CASRN='142-82-5', MW=100.2, similarity_variable=0.2295)
    >>> CpSolid(200)
    131.205824

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    .. [2] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    name = 'solid heat capacity'
    units = 'J/mol/K'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default; a theoretical solid phase exists
    for all chemicals at sufficiently high pressures, although few chemicals
    could stably exist in those conditions.'''
    property_min = 0
    '''Heat capacities have a minimum value of 0 at 0 K.'''
    property_max = 1E4
    '''Maximum value of Heat capacity; arbitrarily set.'''

    ranked_methods = [PERRY151, CRCSTD, LASTOVKA_S]
    '''Default rankings of the available methods.'''

    _fit_force_n = {}
    '''Dictionary containing method: fit_n, for use in methods which should
    only ever be fit to a specific `n` value'''
    _fit_force_n[CRCSTD] = 1


    custom_args = ('MW', 'similarity_variable')

    def __init__(self, CASRN='', similarity_variable=None, MW=None,
                 extrapolation='linear', **kwargs):
        self.similarity_variable = similarity_variable
        self.MW = MW
        self.CASRN = CASRN

        super(HeatCapacitySolid, self).__init__(extrapolation, **kwargs)

    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {PERRY151: [i for i in heat_capacity.Cp_dict_PerryI.keys() if 'c' in heat_capacity.Cp_dict_PerryI[i]],
                CRCSTD: [i for i in heat_capacity.CRC_standard_data.index if not isnan(heat_capacity.CRC_standard_data.at[i, 'Cps'])],
                }
    def load_all_methods(self, load_data):
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
        Tmins, Tmaxs = [], []
        self.T_limits = T_limits = {}
        if load_data:
            if self.CASRN and self.CASRN in heat_capacity.Cp_dict_PerryI and 'c' in heat_capacity.Cp_dict_PerryI[self.CASRN]:
                self.PERRY151_Tmin = heat_capacity.Cp_dict_PerryI[self.CASRN]['c']['Tmin'] if heat_capacity.Cp_dict_PerryI[self.CASRN]['c']['Tmin'] else 0
                self.PERRY151_Tmax = heat_capacity.Cp_dict_PerryI[self.CASRN]['c']['Tmax'] if heat_capacity.Cp_dict_PerryI[self.CASRN]['c']['Tmax'] else 2000
                self.PERRY151_const = heat_capacity.Cp_dict_PerryI[self.CASRN]['c']['Const']
                self.PERRY151_lin = heat_capacity.Cp_dict_PerryI[self.CASRN]['c']['Lin']
                self.PERRY151_quad = heat_capacity.Cp_dict_PerryI[self.CASRN]['c']['Quad']
                self.PERRY151_quadinv = heat_capacity.Cp_dict_PerryI[self.CASRN]['c']['Quadinv']
                methods.append(PERRY151)
                Tmins.append(self.PERRY151_Tmin); Tmaxs.append(self.PERRY151_Tmax)
                T_limits[PERRY151] = (self.PERRY151_Tmin, self.PERRY151_Tmax)
            if self.CASRN in heat_capacity.CRC_standard_data.index and not isnan(heat_capacity.CRC_standard_data.at[self.CASRN, 'Cps']):
                self.CRCSTD_Cp = float(heat_capacity.CRC_standard_data.at[self.CASRN, 'Cps'])
                methods.append(CRCSTD)
                T_limits[CRCSTD] = (298.15, 298.15)
        if self.MW and self.similarity_variable:
            methods.append(LASTOVKA_S)
            Tmins.append(1.0); Tmaxs.append(10000)
            T_limits[LASTOVKA_S] = (1.0, 1e4)
            # Works above roughly 1 K up to 10K.
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)


    def calculate(self, T, method):
        r'''Method to calculate heat capacity of a solid at temperature `T`
        with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate heat capacity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Cp : float
            Heat capacity of the solid at T, [J/mol/K]
        '''
        if method == POLY_FIT:
            if T < self.poly_fit_Tmin:
                Cp = (T - self.poly_fit_Tmin)*self.poly_fit_Tmin_slope + self.poly_fit_Tmin_value
            elif T > self.poly_fit_Tmax:
                Cp = (T - self.poly_fit_Tmax)*self.poly_fit_Tmax_slope + self.poly_fit_Tmax_value
            else:
                Cp = horner(self.poly_fit_coeffs, T)
        elif method == PERRY151:
            Cp = (self.PERRY151_const + self.PERRY151_lin*T
            + self.PERRY151_quadinv/T**2 + self.PERRY151_quad*T**2)*calorie
        elif method == CRCSTD:
            Cp = self.CRCSTD_Cp
        elif method == LASTOVKA_S:
            Cp = Lastovka_solid(T, self.similarity_variable)
            Cp = property_mass_to_molar(Cp, self.MW)
        else:
            return self._base_calculate(T, method)
        return Cp


    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For tabular data,
        extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the
        extrapolation is considered valid for all temperatures.
        For the :obj:`Lastovka_solid <chemicals.heat_capacity.Lastovka_solid>` method, it is considered valid under
        10000K.

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
        if method == PERRY151:
            if T < self.PERRY151_Tmin or T > self.PERRY151_Tmax:
                validity = False
        elif method == CRCSTD:
            if T < 298.15-50 or T > 298.15+50:
                validity = False
        elif method == LASTOVKA_S:
            if T > 10000 or T < 0:
                validity = False
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        elif method == POLY_FIT:
            validity = True
        else:
            raise Exception('Method not valid')
        return validity

    def calculate_integral(self, T1, T2, method):
        r'''Method to calculate the integral of a property with respect to
        temperature, using a specified method. Implements the analytical
        integrals of all available methods except for tabular data.

        Parameters
        ----------
        T1 : float
            Lower limit of integration, [K]
        T2 : float
            Upper limit of integration, [K]
        method : str
            Method for which to find the integral

        Returns
        -------
        integral : float
            Calculated integral of the property over the given range,
            [`units*K`]
        '''
        if method == POLY_FIT:
            return fit_integral_linear_extrapolation(T1, T2,
                self.poly_fit_int_coeffs, self.poly_fit_Tmin,
                self.poly_fit_Tmax, self.poly_fit_Tmin_value,
                self.poly_fit_Tmax_value, self.poly_fit_Tmin_slope,
                self.poly_fit_Tmax_slope)
        elif method == PERRY151:
            H2 = (self.PERRY151_const*T2 + 0.5*self.PERRY151_lin*T2**2
                  - self.PERRY151_quadinv/T2 + self.PERRY151_quad*T2**3/3.)
            H1 = (self.PERRY151_const*T1 + 0.5*self.PERRY151_lin*T1**2
                  - self.PERRY151_quadinv/T1 + self.PERRY151_quad*T1**3/3.)
            return (H2-H1)*calorie
        elif method == CRCSTD:
            return (T2-T1)*self.CRCSTD_Cp
        elif method == LASTOVKA_S:
            dH = (Lastovka_solid_integral(T2, self.similarity_variable)
                    - Lastovka_solid_integral(T1, self.similarity_variable))
            return property_mass_to_molar(dH, self.MW)
        else:
            return super(HeatCapacitySolid, self).calculate_integral(T1, T2, method)

    def calculate_integral_over_T(self, T1, T2, method):
        r'''Method to calculate the integral of a property over temperature
        with respect to temperature, using a specified method. Implements the
        analytical integrals of all available methods except for tabular data.

        Parameters
        ----------
        T1 : float
            Lower limit of integration, [K]
        T2 : float
            Upper limit of integration, [K]
        method : str
            Method for which to find the integral

        Returns
        -------
        integral : float
            Calculated integral of the property over the given range,
            [`units`]
        '''
        if method == POLY_FIT:
            return fit_integral_over_T_linear_extrapolation(T1, T2,
                self.poly_fit_T_int_T_coeffs, self.poly_fit_log_coeff,
                self.poly_fit_Tmin, self.poly_fit_Tmax,
                self.poly_fit_Tmin_value,
                self.poly_fit_Tmax_value, self.poly_fit_Tmin_slope,
                self.poly_fit_Tmax_slope)
        elif method == PERRY151:
            S2 = (self.PERRY151_const*log(T2) + self.PERRY151_lin*T2
                  - self.PERRY151_quadinv/(2.*T2**2) + 0.5*self.PERRY151_quad*T2**2)
            S1 = (self.PERRY151_const*log(T1) + self.PERRY151_lin*T1
                  - self.PERRY151_quadinv/(2.*T1**2) + 0.5*self.PERRY151_quad*T1**2)
            return (S2 - S1)*calorie
        elif method == CRCSTD:
            S2 = self.CRCSTD_Cp*log(T2)
            S1 = self.CRCSTD_Cp*log(T1)
            return (S2 - S1)
        elif method == LASTOVKA_S:
            dS = (Lastovka_solid_integral_over_T(T2, self.similarity_variable)
                    - Lastovka_solid_integral_over_T(T1, self.similarity_variable))
            return property_mass_to_molar(dS, self.MW)
        else:
            return super(HeatCapacitySolid, self).calculate_integral_over_T(T1, T2, method)



### Mixture heat capacities
LALIBERTE = 'LALIBERTE'
heat_capacity_gas_mixture_methods = [LINEAR]
'''Holds all methods available for the :obj:`HeatCapacityGasMixture` class, for use in
iterating over them.'''

heat_capacity_liquid_mixture_methods = [LALIBERTE, LINEAR]
'''Holds all methods available for the :obj:`HeatCapacityLiquidMixture` class, for use in
iterating over them.'''

heat_capacity_solid_mixture_methods = [LINEAR]
'''Holds all methods available for the :obj:`HeatCapacitySolidMixture` class, for use in
iterating over them.'''


class HeatCapacityLiquidMixture(MixtureProperty):
    '''Class for dealing with liquid heat capacity of a mixture as a function
    of temperature, pressure, and composition.
    Consists only of mole weighted averaging, and the Laliberte method for
    aqueous electrolyte solutions.

    Parameters
    ----------
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    CASs : str, optional
        The CAS numbers of all species in the mixture
    HeatCapacityLiquids : list[HeatCapacityLiquid], optional
        HeatCapacityLiquid objects created for all species in the mixture [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_liquid_mixture_methods`.

    **LALIBERTE**:
        Electrolyte model equation with coefficients; see
        :obj:`thermo.electrochem.Laliberte_heat_capacity` for more details.
    **LINEAR**:
        Mixing rule described in :obj:`mixing_simple <chemicals.utils.mixing_simple>`.
    '''
    name = 'Liquid heat capacity'
    units = 'J/mol'
    property_min = 1
    '''Allow very low heat capacities; arbitrarily set; liquid heat capacity
    should always be somewhat substantial.'''
    property_max = 1E4 # Originally 1E4
    '''Maximum valid of Heat capacity; arbitrarily set. For fluids very near
    the critical point, this value can be obscenely high.'''

    ranked_methods = [LALIBERTE, LINEAR]
    pure_references = ('HeatCapacityLiquids',)
    pure_reference_types = (HeatCapacityLiquid,)

    custom_args = ('MWs', )

    def __init__(self, MWs=[], CASs=[], HeatCapacityLiquids=[]):
        self.MWs = MWs
        self.CASs = CASs
        self.HeatCapacityLiquids = HeatCapacityLiquids

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        heat capacity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        heat capacity above.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`.'''
        self.all_methods = set()
        '''Set of all methods available for a given set of information;
        filled by :obj:`load_all_methods`.'''
        self.load_all_methods()

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
        if len(self.CASs) > 1 and '7732-18-5' in self.CASs:
            Laliberte_data = electrochem.Laliberte_data
            a1s, a2s, a3s, a4s, a5s, a6s = [], [], [], [], [], []
            laliberte_incomplete = False
            for CAS in self.CASs:
                if CAS == '7732-18-5':
                    continue
                if CAS in Laliberte_data.index:
                    dat = Laliberte_data.loc[CAS].values
                    if isnan(dat[22]):
                        laliberte_incomplete = True
                        break
                    a1s.append(float(dat[22]))
                    a2s.append(float(dat[23]))
                    a3s.append(float(dat[24]))
                    a4s.append(float(dat[25]))
                    a5s.append(float(dat[26]))
                    a6s.append(float(dat[27]))
                else:
                    laliberte_incomplete = True
                    break

            if not laliberte_incomplete:
                self.Laliberte_a1s = a1s
                self.Laliberte_a2s = a2s
                self.Laliberte_a3s = a3s
                self.Laliberte_a4s = a4s
                self.Laliberte_a5s = a5s
                self.Laliberte_a6s = a6s

            wCASs = [i for i in self.CASs if i != '7732-18-5']
            methods.append(LALIBERTE)
            self.wCASs = wCASs
            self.index_w = self.CASs.index('7732-18-5')
        self.all_methods = all_methods = set(methods)
        for m in self.ranked_methods:
            if m in all_methods:
                self.method = m
                break

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate heat capacity of a liquid mixture at
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
        Cplm : float
            Molar heat capacity of the liquid mixture at the given conditions,
            [J/mol]
        '''
        if method == LINEAR:
            Cplms = [i(T) for i in self.HeatCapacityLiquids]
            return mixing_simple(zs, Cplms)
        elif method == LALIBERTE:
            ws = list(ws) ; ws.pop(self.index_w)
            Cpl = Laliberte_heat_capacity(T, ws, self.wCASs)
            MW = mixing_simple(zs, self.MWs)
            return property_mass_to_molar(Cpl, MW)
        else:
            raise Exception('Method not valid')

    def test_method_validity(self, T, P, zs, ws, method):
        r'''Method to test the validity of a specified method for the given
        conditions. No methods have implemented checks or strict ranges of
        validity.

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
        if method in self.all_methods:
            return True
        else:
            raise Exception('Method not valid')



class HeatCapacitySolidMixture(MixtureProperty):
    '''Class for dealing with solid heat capacity of a mixture as a function of
    temperature, pressure, and composition.
    Consists only of mole weighted averaging.

    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
    HeatCapacitySolids : list[HeatCapacitySolid], optional
        HeatCapacitySolid objects created for all species in the mixture [-]
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_solid_mixture_methods`.

    **LINEAR**:
        Mixing rule described in :obj:`mixing_simple <chemicals.utils.mixing_simple>`.
    '''
    name = 'Solid heat capacity'
    units = 'J/mol'
    property_min = 0
    '''Heat capacities have a minimum value of 0 at 0 K.'''
    property_max = 1E4
    '''Maximum value of Heat capacity; arbitrarily set.'''

    ranked_methods = [LINEAR]
    pure_references = ('HeatCapacitySolids',)
    pure_reference_types = (HeatCapacitySolid,)

    custom_args = ('MWs', )

    def __init__(self, CASs=[], HeatCapacitySolids=[], MWs=[]):
        self.CASs = CASs
        self.HeatCapacitySolids = HeatCapacitySolids
        self.MWs = MWs

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        heat capacity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        heat capacity above.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`.'''
        self.all_methods = set()
        '''Set of all methods available for a given set of information;
        filled by :obj:`load_all_methods`.'''
        self.load_all_methods()

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
        for m in self.ranked_methods:
            if m in all_methods:
                self.method = m
                break

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate heat capacity of a solid mixture at
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
        Cpsm : float
            Molar heat capacity of the solid mixture at the given conditions, [J/mol]
        '''
        if method == LINEAR:
            Cpsms = [i(T) for i in self.HeatCapacitySolids]
            return mixing_simple(zs, Cpsms)
        else:
            raise Exception('Method not valid')

    def test_method_validity(self, T, P, zs, ws, method):
        r'''Method to test the validity of a specified method for the given
        conditions. No methods have implemented checks or strict ranges of
        validity.

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
        if method in self.all_methods:
            return True
        else:
            raise Exception('Method not valid')


class HeatCapacityGasMixture(MixtureProperty):
    '''Class for dealing with the gas heat capacity of a mixture as a function
    of temperature, pressure, and composition. Consists only of mole weighted
    averaging.

    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
    HeatCapacityGases : list[HeatCapacityGas], optional
        HeatCapacityGas objects created for all species in the mixture [-]
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_gas_mixture_methods`.

    **LINEAR**:
        Mixing rule described in :obj:`mixing_simple <chemicals.utils.mixing_simple>`.
    '''
    name = 'Gas heat capacity'
    units = 'J/mol'
    property_min = 0
    '''Heat capacities have a minimum value of 0 at 0 K.'''
    property_max = 1E4
    '''Maximum valid of Heat capacity; arbitrarily set. For fluids very near
    the critical point, this value can be obscenely high.'''

    ranked_methods = [LINEAR]
    pure_references = ('HeatCapacityGases',)
    pure_reference_types = (HeatCapacityGas,)

    custom_args = ('MWs', )

    def __init__(self, CASs=[], HeatCapacityGases=[], MWs=[]):
        self.CASs = CASs
        self.HeatCapacityGases = HeatCapacityGases
        self.MWs = MWs

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        heat capacity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        heat capacity above.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`.'''
        self.all_methods = set()
        '''Set of all methods available for a given set of information;
        filled by :obj:`load_all_methods`.'''
        self.load_all_methods()

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
        for m in self.ranked_methods:
            if m in all_methods:
                self.method = m
                break

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate heat capacity of a gas mixture at
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
        Cpgm : float
            Molar heat capacity of the gas mixture at the given conditions,
            [J/mol]
        '''
        if method == LINEAR:
            Cpgms = [i(T) for i in self.HeatCapacityGases]
            return mixing_simple(zs, Cpgms)
        else:
            raise Exception('Method not valid')

    def test_method_validity(self, T, P, zs, ws, method):
        r'''Method to test the validity of a specified method for the given
        conditions. No methods have implemented checks or strict ranges of
        validity.

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
        if method in self.all_methods:
            return True
        else:
            raise Exception('Method not valid')


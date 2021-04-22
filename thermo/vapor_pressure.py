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


This module contains implementations of :obj:`thermo.utils.TDependentProperty`
representing vapor pressure and sublimation pressure. A variety of estimation
and data methods are available as included in the `chemicals` library.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Vapor Pressure
==============
.. autoclass:: VaporPressure
    :members: calculate, test_method_validity,
              interpolation_T, interpolation_property,
              interpolation_property_inv, name, property_max, property_min,
              units, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: vapor_pressure_methods

Sublimation Pressure
====================
.. autoclass:: SublimationPressure
    :members: calculate, test_method_validity,
              interpolation_T, interpolation_property,
              interpolation_property_inv, name, property_max, property_min,
              units, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: sublimation_pressure_methods
'''

from __future__ import division

__all__ = ['vapor_pressure_methods', 'VaporPressure', 'SublimationPressure',
           'sublimation_pressure_methods']

import os
from fluids.constants import R
from fluids.numerics import polyint_over_x, horner_log, horner, polyint, horner_and_der2, horner_and_der, derivative, newton, linspace, numpy as np

from math import e
from chemicals.utils import log, exp, isnan
from chemicals.dippr import EQ101
from chemicals import miscdata
from chemicals.miscdata import lookup_VDI_tabular_data
from chemicals.vapor_pressure import *
from chemicals.vapor_pressure import dAntoine_dT, d2Antoine_dT2, dWagner_original_dT, d2Wagner_original_dT2, dWagner_dT, d2Wagner_dT2, dTRC_Antoine_extended_dT, d2TRC_Antoine_extended_dT2

from chemicals import vapor_pressure
from thermo.utils import TDependentProperty
from thermo.utils import VDI_TABULAR, DIPPR_PERRY_8E, VDI_PPDS, COOLPROP, EOS, POLY_FIT
from thermo.coolprop import has_CoolProp, PropsSI, coolprop_dict, coolprop_fluids
from thermo.base import source_path


WAGNER_MCGARRY = 'WAGNER_MCGARRY'
WAGNER_POLING = 'WAGNER_POLING'
ANTOINE_POLING = 'ANTOINE_POLING'
ANTOINE_EXTENDED_POLING = 'ANTOINE_EXTENDED_POLING'

BOILING_CRITICAL = 'BOILING_CRITICAL'
LEE_KESLER_PSAT = 'LEE_KESLER_PSAT'
AMBROSE_WALTON = 'AMBROSE_WALTON'
SANJARI = 'SANJARI'
EDALAT = 'EDALAT'
BEST_FIT_AB = 'POLY_FIT AB extrapolation'
BEST_FIT_ABC = 'POLY_FIT ABC extrapolation'

vapor_pressure_methods = [WAGNER_MCGARRY, WAGNER_POLING, ANTOINE_EXTENDED_POLING,
                          DIPPR_PERRY_8E, VDI_PPDS, COOLPROP, ANTOINE_POLING, VDI_TABULAR, AMBROSE_WALTON,
                          LEE_KESLER_PSAT, EDALAT, EOS, BOILING_CRITICAL, SANJARI]
'''Holds all methods available for the VaporPressure class, for use in
iterating over them.'''


class VaporPressure(TDependentProperty):
    '''Class for dealing with vapor pressure as a function of temperature.
    Consists of four coefficient-based methods and four data sources, one
    source of tabular information, four corresponding-states estimators,
    any provided equation of state, and the external library CoolProp.

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
    CASRN : str, optional
        The CAS number of the chemical
    eos : object, optional
        Equation of State object after :obj:`thermo.eos.GCEOS`
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files;
        this can be used to reduce the memory consumption of an object as well,
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    poly_fit : tuple(float, float, list[float]), optional
        Tuple of (Tmin, Tmax, coeffs) representing a prefered fit to the
        vapor pressure of a species; the coefficients are evaluated with
        horner's method, and the input variable and output are transformed by
        the default transformations of this object; used instead of any other
        default method if provided . [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`vapor_pressure_methods`.

    **WAGNER_MCGARRY**:
        The Wagner 3,6 original model equation documented in
        :obj:`chemicals.vapor_pressure.Wagner_original`, with data for 245 chemicals, from [1]_,
    **WAGNER_POLING**:
        The Wagner 2.5, 5 model equation documented in :obj:`chemicals.vapor_pressure.Wagner` in [2]_,
        with data for  104 chemicals.
    **ANTOINE_EXTENDED_POLING**:
        The TRC extended Antoine model equation documented in
        :obj:`chemicals.vapor_pressure.TRC_Antoine_extended` with data for 97 chemicals in [2]_.
    **ANTOINE_POLING**:
        Standard Antoine equation, as documented in the function
        :obj:`chemicals.vapor_pressure.Antoine` and with data for 325 fluids from [2]_.
        Coefficients were altered to be in units of Pa and Celcius.
    **DIPPR_PERRY_8E**:
        A collection of 341 coefficient sets from the DIPPR database published
        openly in [5]_. Provides temperature limits for all its fluids.
        :obj:`chemicals.dippr.EQ101` is used for its fluids.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS, published
        openly in [4]_.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [3]_. Very slow.
    **BOILING_CRITICAL**:
        Fundamental relationship in thermodynamics making several
        approximations; see :obj:`chemicals.vapor_pressure.boiling_critical_relation` for details.
        Least accurate method in most circumstances.
    **LEE_KESLER_PSAT**:
        CSP method documented in :obj:`chemicals.vapor_pressure.Lee_Kesler`. Widely used.
    **AMBROSE_WALTON**:
        CSP method documented in :obj:`chemicals.vapor_pressure.Ambrose_Walton`.
    **SANJARI**:
        CSP method documented in :obj:`chemicals.vapor_pressure.Sanjari`.
    **EDALAT**:
        CSP method documented in :obj:`chemicals.vapor_pressure.Edalat`.
    **VDI_TABULAR**:
        Tabular data in [4]_ along the saturation curve; interpolation is as
        set by the user or the default.
    **EOS**:
        Equation of state provided by user; must implement
        :obj:`thermo.eos.GCEOS.Psat`

    See Also
    --------
    chemicals.vapor_pressure.Wagner_original
    chemicals.vapor_pressure.Wagner
    chemicals.vapor_pressure.TRC_Antoine_extended
    chemicals.vapor_pressure.Antoine
    chemicals.vapor_pressure.boiling_critical_relation
    chemicals.vapor_pressure.Lee_Kesler
    chemicals.vapor_pressure.Ambrose_Walton
    chemicals.vapor_pressure.Sanjari
    chemicals.vapor_pressure.Edalat

    References
    ----------
    .. [1] McGarry, Jack. "Correlation and Prediction of the Vapor Pressures of
       Pure Liquids over Large Pressure Ranges." Industrial & Engineering
       Chemistry Process Design and Development 22, no. 2 (April 1, 1983):
       313-22. doi:10.1021/i200021a023.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [3] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    .. [4] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    .. [5] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    name = 'Vapor pressure'
    units = 'Pa'

    @staticmethod
    def interpolation_T(T):
        '''Function to make the data-based interpolation as linear as possible.
        This transforms the input `T` into the `1/T` domain.'''
        return 1./T

    @staticmethod
    def interpolation_property(P):
        '''log(P) interpolation transformation by default.
        '''
        return log(P)

    @staticmethod
    def interpolation_property_inv(P):
        '''exp(P) interpolation transformation by default; reverses
        :obj:`interpolation_property_inv`.'''
        return exp(P)

    tabular_extrapolation_permitted = False
    '''Disallow tabular extrapolation by default.'''
    property_min = 0
    '''Mimimum valid value of vapor pressure.'''
    property_max = 1E10
    '''Maximum valid value of vapor pressure. Set slightly above the critical
    point estimated for Iridium; Mercury's 160 MPa critical point is the
    highest known.'''

    ranked_methods = [WAGNER_MCGARRY, WAGNER_POLING, ANTOINE_EXTENDED_POLING,
                      DIPPR_PERRY_8E, VDI_PPDS, COOLPROP, ANTOINE_POLING, VDI_TABULAR, AMBROSE_WALTON,
                      LEE_KESLER_PSAT, EDALAT, BOILING_CRITICAL, EOS, SANJARI]
    '''Default rankings of the available methods.'''

    custom_args = ('Tb', 'Tc', 'Pc', 'omega', 'eos')

    def __init__(self, Tb=None, Tc=None, Pc=None, omega=None, CASRN='',
                 eos=None, extrapolation='AntoineAB|DIPPR101_ABC', **kwargs):
        self.CASRN = CASRN
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.eos = eos

        super(VaporPressure, self).__init__(extrapolation, **kwargs)

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {WAGNER_MCGARRY: vapor_pressure.Psat_data_WagnerMcGarry.index,
                WAGNER_POLING: vapor_pressure.Psat_data_WagnerPoling.index,
                ANTOINE_EXTENDED_POLING: vapor_pressure.Psat_data_AntoineExtended.index,
                ANTOINE_POLING: vapor_pressure.Psat_data_AntoinePoling.index,
                DIPPR_PERRY_8E: vapor_pressure.Psat_data_Perrys2_8.index,
                COOLPROP: coolprop_dict,
                VDI_TABULAR: list(miscdata.VDI_saturation_dict.keys()),
                VDI_PPDS: vapor_pressure.Psat_data_VDI_PPDS_3.index,
                }

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
        self.T_limits = T_limits = {}
        methods = []
        Tmins, Tmaxs = [], []
        if load_data:
            if self.CASRN in vapor_pressure.Psat_data_WagnerMcGarry.index:
                methods.append(WAGNER_MCGARRY)
                A, B, C, D, self.WAGNER_MCGARRY_Pc, self.WAGNER_MCGARRY_Tc, self.WAGNER_MCGARRY_Tmin = vapor_pressure.Psat_values_WagnerMcGarry[vapor_pressure.Psat_data_WagnerMcGarry.index.get_loc(self.CASRN)].tolist()
                self.WAGNER_MCGARRY_coefs = [A, B, C, D]
                Tmins.append(self.WAGNER_MCGARRY_Tmin); Tmaxs.append(self.WAGNER_MCGARRY_Tc)
                T_limits[WAGNER_MCGARRY] = (self.WAGNER_MCGARRY_Tmin, self.WAGNER_MCGARRY_Tc)

            if self.CASRN in vapor_pressure.Psat_data_WagnerPoling.index:
                methods.append(WAGNER_POLING)
                A, B, C, D, self.WAGNER_POLING_Tc, self.WAGNER_POLING_Pc, Tmin, self.WAGNER_POLING_Tmax = vapor_pressure.Psat_values_WagnerPoling[vapor_pressure.Psat_data_WagnerPoling.index.get_loc(self.CASRN)].tolist()
                # Some Tmin values are missing; Arbitrary choice of 0.1 lower limit
                Tmin = Tmin if not isnan(Tmin) else self.WAGNER_POLING_Tmax*0.1
                self.WAGNER_POLING_Tmin = Tmin
                self.WAGNER_POLING_coefs = [A, B, C, D]
                Tmins.append(Tmin); Tmaxs.append(self.WAGNER_POLING_Tmax)
                T_limits[WAGNER_POLING] = (self.WAGNER_POLING_Tmin, self.WAGNER_POLING_Tmax)

            if self.CASRN in vapor_pressure.Psat_data_AntoineExtended.index:
                methods.append(ANTOINE_EXTENDED_POLING)
                A, B, C, Tc, to, n, E, F, self.ANTOINE_EXTENDED_POLING_Tmin, self.ANTOINE_EXTENDED_POLING_Tmax = vapor_pressure.Psat_values_AntoineExtended[vapor_pressure.Psat_data_AntoineExtended.index.get_loc(self.CASRN)].tolist()
                self.ANTOINE_EXTENDED_POLING_coefs = [Tc, to, A, B, C, n, E, F]
                Tmins.append(self.ANTOINE_EXTENDED_POLING_Tmin); Tmaxs.append(self.ANTOINE_EXTENDED_POLING_Tmax)
                T_limits[ANTOINE_EXTENDED_POLING] = (self.ANTOINE_EXTENDED_POLING_Tmin, self.ANTOINE_EXTENDED_POLING_Tmax)

            if self.CASRN in vapor_pressure.Psat_data_AntoinePoling.index:
                methods.append(ANTOINE_POLING)
                A, B, C, self.ANTOINE_POLING_Tmin, self.ANTOINE_POLING_Tmax = vapor_pressure.Psat_values_AntoinePoling[vapor_pressure.Psat_data_AntoinePoling.index.get_loc(self.CASRN)].tolist()
                self.ANTOINE_POLING_coefs = [A, B, C]
                Tmins.append(self.ANTOINE_POLING_Tmin); Tmaxs.append(self.ANTOINE_POLING_Tmax)
                T_limits[ANTOINE_POLING] = (self.ANTOINE_POLING_Tmin, self.ANTOINE_POLING_Tmax)

            if self.CASRN in vapor_pressure.Psat_data_Perrys2_8.index:
                methods.append(DIPPR_PERRY_8E)
                C1, C2, C3, C4, C5, self.Perrys2_8_Tmin, self.Perrys2_8_Tmax = vapor_pressure.Psat_values_Perrys2_8[vapor_pressure.Psat_data_Perrys2_8.index.get_loc(self.CASRN)].tolist()
                self.Perrys2_8_coeffs = [C1, C2, C3, C4, C5]
                Tmins.append(self.Perrys2_8_Tmin); Tmaxs.append(self.Perrys2_8_Tmax)
                T_limits[DIPPR_PERRY_8E] = (self.Perrys2_8_Tmin, self.Perrys2_8_Tmax)
            if has_CoolProp() and self.CASRN in coolprop_dict:
                methods.append(COOLPROP)
                self.CP_f = coolprop_fluids[self.CASRN]
                Tmins.append(self.CP_f.Tmin); Tmaxs.append(self.CP_f.Tc)
                T_limits[COOLPROP] = (self.CP_f.Tmin, self.CP_f.Tc)

            if self.CASRN in miscdata.VDI_saturation_dict:
                methods.append(VDI_TABULAR)
                Ts, props = lookup_VDI_tabular_data(self.CASRN, 'P')
                self.VDI_Tmin = Ts[0]
                self.VDI_Tmax = Ts[-1]
                self.tabular_data[VDI_TABULAR] = (Ts, props)
                Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
                T_limits[VDI_TABULAR] = (self.VDI_Tmin, self.VDI_Tmax)

            if self.CASRN in vapor_pressure.Psat_data_VDI_PPDS_3.index:
                Tm, Tc, Pc, A, B, C, D = vapor_pressure.Psat_values_VDI_PPDS_3[vapor_pressure.Psat_data_VDI_PPDS_3.index.get_loc(self.CASRN)].tolist()
                self.VDI_PPDS_coeffs = [A, B, C, D]
                self.VDI_PPDS_Tc = Tc
                self.VDI_PPDS_Tm = Tm
                self.VDI_PPDS_Pc = Pc
                methods.append(VDI_PPDS)
                Tmins.append(self.VDI_PPDS_Tm); Tmaxs.append(self.VDI_PPDS_Tc)
                T_limits[VDI_PPDS] = (self.VDI_PPDS_Tm, self.VDI_PPDS_Tc)
        if all((self.Tb, self.Tc, self.Pc)):
            methods.append(BOILING_CRITICAL)
            Tmins.append(0.01); Tmaxs.append(self.Tc)
            T_limits[BOILING_CRITICAL] = (0.01, self.Tc)
        if all((self.Tc, self.Pc, self.omega)):
            methods.append(LEE_KESLER_PSAT)
            methods.append(AMBROSE_WALTON)
            methods.append(SANJARI)
            methods.append(EDALAT)
            if self.eos:
                methods.append(EOS)
                T_limits[EOS] = (0.1*self.Tc, self.Tc)
            Tmins.append(0.01); Tmaxs.append(self.Tc)
            T_limits[LEE_KESLER_PSAT] = T_limits[AMBROSE_WALTON] = T_limits[SANJARI] = T_limits[EDALAT] = (0.01, self.Tc)
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin = min(Tmins)
            self.Tmax = max(Tmaxs)

    def calculate(self, T, method):
        r'''Method to calculate vapor pressure of a fluid at temperature `T`
        with a given method.

        This method has no exception handling; see :obj:`thermo.utils.TDependentProperty.T_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature at calculate vapor pressure, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Psat : float
            Vapor pressure at T, [pa]
        '''
        if method == POLY_FIT:
            if T < self.poly_fit_Tmin:
                Psat = (T - self.poly_fit_Tmin)*self.poly_fit_Tmin_slope + self.poly_fit_Tmin_value
            elif T > self.poly_fit_Tmax:
                Psat = (T - self.poly_fit_Tmax)*self.poly_fit_Tmax_slope + self.poly_fit_Tmax_value
            else:
                Psat = horner(self.poly_fit_coeffs, T)
            Psat = exp(Psat)
        elif method == BEST_FIT_AB:
            if T < self.poly_fit_Tmax:
                return self.calculate(T, POLY_FIT)
            A, B = self.poly_fit_AB_high_ABC_compat
            return exp(A + B/T)
        elif method == BEST_FIT_ABC:
            if T < self.poly_fit_Tmax:
                return self.calculate(T, POLY_FIT)
            A, B, C = self.DIPPR101_ABC_high
            return exp(A + B/T + C*log(T))
        elif method == WAGNER_MCGARRY:
            Psat = Wagner_original(T, self.WAGNER_MCGARRY_Tc, self.WAGNER_MCGARRY_Pc, *self.WAGNER_MCGARRY_coefs)
        elif method == WAGNER_POLING:
            Psat = Wagner(T, self.WAGNER_POLING_Tc, self.WAGNER_POLING_Pc, *self.WAGNER_POLING_coefs)
        elif method == ANTOINE_EXTENDED_POLING:
            Psat = TRC_Antoine_extended(T, *self.ANTOINE_EXTENDED_POLING_coefs)
        elif method == ANTOINE_POLING:
            A, B, C = self.ANTOINE_POLING_coefs
            Psat = Antoine(T, A, B, C, base=10.0)
        elif method == DIPPR_PERRY_8E:
            Psat = EQ101(T, *self.Perrys2_8_coeffs)
        elif method == VDI_PPDS:
            Psat = Wagner(T, self.VDI_PPDS_Tc, self.VDI_PPDS_Pc, *self.VDI_PPDS_coeffs)
        elif method == COOLPROP:
            Psat = PropsSI('P','T', T,'Q',0, self.CASRN)
        elif method == BOILING_CRITICAL:
            Psat = boiling_critical_relation(T, self.Tb, self.Tc, self.Pc)
        elif method == LEE_KESLER_PSAT:
            Psat = Lee_Kesler(T, self.Tc, self.Pc, self.omega)
        elif method == AMBROSE_WALTON:
            Psat = Ambrose_Walton(T, self.Tc, self.Pc, self.omega)
        elif method == SANJARI:
            Psat = Sanjari(T, self.Tc, self.Pc, self.omega)
        elif method == EDALAT:
            Psat = Edalat(T, self.Tc, self.Pc, self.omega)
        elif method == EOS:
            Psat = self.eos[0].Psat(T)
        elif method == POLY_FIT:
            Psat = exp(horner(self.poly_fit_coeffs, T))
        else:
            return self._base_calculate(T, method)
        return Psat

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For CSP methods, the models
        are considered valid from 0 K to the critical point. For tabular data,
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
        T_limits = self.T_limits
        if method in T_limits:
            Tmin, Tmax = T_limits[method]
            return Tmin <= T <= Tmax
        elif method == POLY_FIT:
            return True
        elif method in self.tabular_data:
            if self.tabular_extrapolation_permitted:
                # good to go without checking
                return True
            else:
                Ts, properties = self.tabular_data[method]
                return Ts[0] < T < Ts[-1]
        else:
            raise ValueError("invalid method '%s'" %method)

    def calculate_derivative(self, T, method, order=1):
        r'''Method to calculate a derivative of a vapor pressure with respect to
        temperature, of a given order  using a specified method. If the method
        is POLY_FIT, an anlytical derivative is used; otherwise SciPy's
        derivative function, with a delta of 1E-6 K and a number of points
        equal to 2*order + 1.

        If the calculation does not succeed, returns the actual error
        encountered.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        method : str
            Method for which to find the derivative
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        derivative : float
            Calculated derivative property, [`units/K^order`]
        '''
        Tmin, Tmax = self.T_limits[method]
        if order == 1 and method == POLY_FIT:
            v, der = horner_and_der(self.poly_fit_coeffs, T)
            return der*exp(v)
        elif method == WAGNER_MCGARRY:
            if Tmin <= T <= Tmax:
                if order == 1:
                    return dWagner_original_dT(T, self.WAGNER_MCGARRY_Tc, self.WAGNER_MCGARRY_Pc, *self.WAGNER_MCGARRY_coefs)
                if order == 2 and T < Tmax:
                    # infinity at Tmax
                    return d2Wagner_original_dT2(T, self.WAGNER_MCGARRY_Tc, self.WAGNER_MCGARRY_Pc, *self.WAGNER_MCGARRY_coefs)
        elif method == WAGNER_POLING:
            if Tmin <= T <= Tmax:
                if order == 1:
                    return dWagner_dT(T, self.WAGNER_POLING_Tc, self.WAGNER_POLING_Pc, *self.WAGNER_POLING_coefs)
                if order == 2 and T < Tmax:
                    # infinity at Tmax
                    return d2Wagner_dT2(T, self.WAGNER_POLING_Tc, self.WAGNER_POLING_Pc, *self.WAGNER_POLING_coefs)
        elif method == VDI_PPDS:
            if Tmin <= T <= Tmax:
                if order == 1:
                    return dWagner_dT(T, self.VDI_PPDS_Tc, self.VDI_PPDS_Pc, *self.VDI_PPDS_coeffs)
                if order == 2 and T < Tmax:
                    # infinity at Tmax
                    return d2Wagner_dT2(T, self.VDI_PPDS_Tc, self.VDI_PPDS_Pc, *self.VDI_PPDS_coeffs)
        elif method == ANTOINE_EXTENDED_POLING:
            if Tmin <= T <= Tmax:
                if order == 1:
                    return dTRC_Antoine_extended_dT(T, *self.ANTOINE_EXTENDED_POLING_coefs)
                if order == 2:
                    return d2TRC_Antoine_extended_dT2(T, *self.ANTOINE_EXTENDED_POLING_coefs)
        elif method == ANTOINE_POLING:
            A, B, C = self.ANTOINE_POLING_coefs
            if Tmin <= T <= Tmax:
                if order == 1:
                    return dAntoine_dT(T, A, B, C, base=10.0)
                if order == 2:
                    return d2Antoine_dT2(T, A, B, C, base=10.0)
        elif method == DIPPR_PERRY_8E:
            if Tmin <= T <= Tmax:
                if order == 1:
                    return EQ101(T, *self.Perrys2_8_coeffs, order=1)
                if order == 2:
                    return EQ101(T, *self.Perrys2_8_coeffs, order=2)
        elif order == 2 and method == POLY_FIT:
            v, der, der2 = horner_and_der2(self.poly_fit_coeffs, T)
            return (der*der + der2)*exp(v)
        return super(VaporPressure, self).calculate_derivative(T, method, order)

    def _custom_set_poly_fit(self):
        try:
            Tmin, Tmax = self.poly_fit_Tmin, self.poly_fit_Tmax
            poly_fit_coeffs = self.poly_fit_coeffs
            v_Tmin = horner(poly_fit_coeffs, Tmin)
            for T_trans in linspace(Tmin, Tmax, 25):
                v, d1, d2 = horner_and_der2(poly_fit_coeffs, T_trans)
                Psat = exp(v)
                dPsat_dT = Psat*d1
                d2Psat_dT2 = Psat*(d1*d1 + d2)

                A, B, C = Antoine_ABC = Antoine_coeffs_from_point(T_trans, Psat, dPsat_dT, d2Psat_dT2, base=e)
                self.poly_fit_AB = list(Antoine_AB_coeffs_from_point(T_trans, Psat, dPsat_dT, base=e))
                self.DIPPR101_ABC = list(DIPPR101_ABC_coeffs_from_point(T_trans, Psat, dPsat_dT, d2Psat_dT2))

                B_OK = B > 0.0 # B is negated in this implementation, so the requirement is reversed
                C_OK = -T_trans < C < 0.0
                if B_OK and C_OK:
                    self.poly_fit_Antoine = Antoine_ABC
                    break
                else:
                    continue

            # Calculate the extrapolation values
            v_Tmax = horner(poly_fit_coeffs, Tmax)
            v, d1, d2 = horner_and_der2(poly_fit_coeffs, Tmax)
            Psat = exp(v)
            dPsat_dT = Psat*d1
            d2Psat_dT2 = Psat*(d1*d1 + d2)
#                A, B, C = Antoine_ABC = Antoine_coeffs_from_point(T_trans, Psat, dPsat_dT, d2Psat_dT2, base=e)
            self.poly_fit_AB_high = list(Antoine_AB_coeffs_from_point(Tmax, Psat, dPsat_dT, base=e))
            self.poly_fit_AB_high_ABC_compat = [self.poly_fit_AB_high[0], -self.poly_fit_AB_high[1]]
            self.DIPPR101_ABC_high = list(DIPPR101_ABC_coeffs_from_point(Tmax, Psat, dPsat_dT, d2Psat_dT2))


        except:
            pass

    def solve_prop_poly_fit(self, goal):
        poly_fit_Tmin, poly_fit_Tmax = self.poly_fit_Tmin, self.poly_fit_Tmax
        poly_fit_Tmin_slope, poly_fit_Tmax_slope = self.poly_fit_Tmin_slope, self.poly_fit_Tmax_slope
        poly_fit_Tmin_value, poly_fit_Tmax_value = self.poly_fit_Tmin_value, self.poly_fit_Tmax_value
        coeffs = self.poly_fit_coeffs

        T_low = log(goal*exp(poly_fit_Tmin*poly_fit_Tmin_slope - poly_fit_Tmin_value))/poly_fit_Tmin_slope
        if T_low <= poly_fit_Tmin:
            return T_low
        T_high = log(goal*exp(poly_fit_Tmax*poly_fit_Tmax_slope - poly_fit_Tmax_value))/poly_fit_Tmax_slope
        if T_high >= poly_fit_Tmax:
            return T_high
        else:
            lnPGoal = log(goal)
            def to_solve(T):
                # dPsat and Psat are both in log basis
                dPsat = Psat = 0.0
                for c in coeffs:
                    dPsat = T*dPsat + Psat
                    Psat = T*Psat + c

                return Psat - lnPGoal, dPsat
            # Guess with the two extrapolations from the linear fits
            # By definition both guesses are in the range of they would have been returned
            if T_low > poly_fit_Tmax:
                T_low = poly_fit_Tmax
            if T_high < poly_fit_Tmin:
                T_high = poly_fit_Tmin
            T = newton(to_solve, 0.5*(T_low + T_high), fprime=True, low=poly_fit_Tmin, high=poly_fit_Tmax)
            return T


PSUB_CLAPEYRON = 'PSUB_CLAPEYRON'

sublimation_pressure_methods = [PSUB_CLAPEYRON]
'''Holds all methods available for the SublimationPressure class, for use in
iterating over them.'''


class SublimationPressure(TDependentProperty):
    '''Class for dealing with sublimation pressure as a function of temperature.
    Consists of one estimation method.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical
    Tt : float, optional
        Triple temperature, [K]
    Pt : float, optional
        Triple pressure, [Pa]
    Hsub_t : float, optional
        Sublimation enthalpy at the triple point, [J/mol]
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files;
        this can be used to reduce the memory consumption of an object as well,
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    poly_fit : tuple(float, float, list[float]), optional
        Tuple of (Tmin, Tmax, coeffs) representing a prefered fit to the
        vapor pressure of a species; the coefficients are evaluated with
        horner's method, and the input variable and output are transformed by
        the default transformations of this object; used instead of any other
        default method if provided . [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`sublimation_pressure_methods`.

    **PSUB_CLAPEYRON**:
        Clapeyron thermodynamic identity, :obj:`Psub_Clapeyron <chemicals.vapor_pressure.Psub_Clapeyron>`

    See Also
    --------
    chemicals.vapor_pressure.Psub_Clapeyron

    References
    ----------
    .. [1] Goodman, B. T., W. V. Wilding, J. L. Oscarson, and R. L. Rowley.
       "Use of the DIPPR Database for the Development of QSPR Correlations:
       Solid Vapor Pressure and Heat of Sublimation of Organic Compounds."
       International Journal of Thermophysics 25, no. 2 (March 1, 2004):
       337-50. https://doi.org/10.1023/B:IJOT.0000028471.77933.80.
    '''
    name = 'Sublimation pressure'
    units = 'Pa'

    interpolation_T = staticmethod(VaporPressure.interpolation_T)
    interpolation_property = staticmethod(VaporPressure.interpolation_property)
    interpolation_property_inv = staticmethod(VaporPressure.interpolation_property_inv)

    tabular_extrapolation_permitted = False
    '''Disallow tabular extrapolation by default.'''
    property_min = 1e-300
    '''Mimimum valid value of sublimation pressure.'''
    property_max = 1e5
    '''Maximum valid value of sublimation pressure. Set to 1 bar tentatively.'''

    ranked_methods = [PSUB_CLAPEYRON]
    '''Default rankings of the available methods.'''

    custom_args = ('Tt', 'Pt', 'Hsub_t')

    def __init__(self, CASRN=None, Tt=None, Pt=None, Hsub_t=None,
                 extrapolation=None, **kwargs):
        self.CASRN = CASRN
        self.Tt = Tt
        self.Pt = Pt
        self.Hsub_t = Hsub_t

        super(SublimationPressure, self).__init__(extrapolation, **kwargs)

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
        if all((self.Tt, self.Pt, self.Hsub_t)):
            methods.append(PSUB_CLAPEYRON)
            Tmins.append(1.0); Tmaxs.append(self.Tt*1.5)
            T_limits[PSUB_CLAPEYRON] = (1.0, self.Tt*1.5)
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin = min(Tmins)
            self.Tmax = max(Tmaxs)

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {}

    def calculate(self, T, method):
        r'''Method to calculate sublimation pressure of a fluid at temperature
        `T` with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at calculate sublimation pressure, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Psub : float
            Sublimation pressure at T, [pa]
        '''
        if method == POLY_FIT:
            if T < self.poly_fit_Tmin:
                Psub = (T - self.poly_fit_Tmin)*self.poly_fit_Tmin_slope + self.poly_fit_Tmin_value
            elif T > self.poly_fit_Tmax:
                Psub = (T - self.poly_fit_Tmax)*self.poly_fit_Tmax_slope + self.poly_fit_Tmax_value
            else:
                Psub = horner(self.poly_fit_coeffs, T)
            Psub = exp(Psub)
        elif method == PSUB_CLAPEYRON:
            Psub = max(Psub_Clapeyron(T, Tt=self.Tt, Pt=self.Pt, Hsub_t=self.Hsub_t), 1e-200)
        else:
            return self._base_calculate(T, method)
        return Psub

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
        if method in [PSUB_CLAPEYRON]:
            return True
            # No lower limit
        elif method == POLY_FIT:
            validity = True
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    return False
        else:
            raise Exception('Method not valid')
        return True
try:
    SublimationPressure._custom_set_poly_fit = VaporPressure._custom_set_poly_fit
except:
    pass


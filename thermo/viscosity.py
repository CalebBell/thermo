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
SOFTWARE.'''

from __future__ import division

__all__ = ['viscosity_liquid_methods', 'viscosity_liquid_methods_P',
           'ViscosityLiquid', 'ViscosityGas', 'viscosity_gas_methods',
           'viscosity_gas_methods_P', 'ViscosityLiquidMixture',
           'ViscosityGasMixture',
           'MIXING_LOG_MOLAR', 'MIXING_LOG_MASS',
           'BROKAW', 'HERNING_ZIPPERER', 'WILKE']

import os
import numpy as np
from fluids.numerics import newton, interp, horner

from chemicals.utils import log, exp, log10
from chemicals.utils import none_and_length_check, mixing_simple, mixing_logarithmic
from thermo.utils import TPDependentProperty, MixtureProperty
from chemicals import miscdata
from chemicals.miscdata import lookup_VDI_tabular_data
from thermo import electrochem
from thermo.electrochem import Laliberte_viscosity
from thermo.coolprop import has_CoolProp, PropsSI, PhaseSI, coolprop_fluids, coolprop_dict, CoolProp_T_dependent_property
from chemicals.dippr import EQ101, EQ102
from chemicals import viscosity
from chemicals.viscosity import *
from chemicals.viscosity import viscosity_gas_Gharagheizi






NONE = 'NONE'
VDI_TABULAR = 'VDI_TABULAR'
VDI_PPDS = 'VDI_PPDS'
COOLPROP = 'COOLPROP'
SUPERCRITICAL = 'SUPERCRITICAL'
DUTT_PRASAD = 'DUTT_PRASAD'
VISWANATH_NATARAJAN_3 = 'VISWANATH_NATARAJAN_3'
VISWANATH_NATARAJAN_2 = 'VISWANATH_NATARAJAN_2'
VISWANATH_NATARAJAN_2E = 'VISWANATH_NATARAJAN_2E'
LETSOU_STIEL = 'LETSOU_STIEL'
PRZEDZIECKI_SRIDHAR = 'PRZEDZIECKI_SRIDHAR'
LUCAS = 'LUCAS'
NEGLIGIBLE = 'NEGLIGIBLE'
DIPPR_PERRY_8E = 'DIPPR_PERRY_8E'
BESTFIT = 'Best fit'

viscosity_liquid_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, DUTT_PRASAD, VISWANATH_NATARAJAN_3,
                         VISWANATH_NATARAJAN_2, VISWANATH_NATARAJAN_2E,
                         VDI_TABULAR, LETSOU_STIEL, PRZEDZIECKI_SRIDHAR]
'''Holds all low-pressure methods available for the ViscosityLiquid class, for
use in iterating over them.'''
viscosity_liquid_methods_P = [COOLPROP, LUCAS]
'''Holds all high-pressure methods available for the ViscosityLiquid class, for
use in iterating over them.'''


class ViscosityLiquid(TPDependentProperty):
    r'''Class for dealing with liquid viscosity as a function of
    temperature and pressure.

    For low-pressure (at 1 atm while under the vapor pressure; along the
    saturation line otherwise) liquids, there are six coefficient-based methods
    from three data sources, one source of tabular information, two
    corresponding-states estimators, and the external library CoolProp.

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
        If False, do not load property coefficients from data sources in files;
        [-]

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
        :obj:`thermo.dippr.EQ101` is used for its fluids.
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

    High pressure methods:

    **LUCAS**:
        CSP method, described in :obj:`chemicals.viscosity.Lucas`. Calculates a
        low-pressure liquid viscosity first, using `T_dependent_property`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [5]_. Very slow, but unparalled in accuracy for pressure
        dependence.

    See Also
    --------
    Viswanath_Natarajan_3
    Viswanath_Natarajan_2
    Viswanath_Natarajan_2_exponential
    Letsou_Stiel
    Przedziecki_Sridhar
    Lucas

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
    '''
    name = 'liquid viscosity'
    units = 'Pa*s'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_P = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default.'''
    property_min = 0.0
    '''Mimimum valid value of liquid viscosity.'''
    property_max = 2E8
    '''Maximum valid value of liquid viscosity. Generous limit, as
    the value is that of bitumen in a Pitch drop experiment.'''

    ranked_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, DUTT_PRASAD, VISWANATH_NATARAJAN_3,
                      VISWANATH_NATARAJAN_2, VISWANATH_NATARAJAN_2E,
                      VDI_TABULAR, LETSOU_STIEL, PRZEDZIECKI_SRIDHAR]
    '''Default rankings of the low-pressure methods.'''
    ranked_methods_P = [COOLPROP, LUCAS]
    '''Default rankings of the high-pressure methods.'''

    def __init__(self, CASRN='', MW=None, Tm=None, Tc=None, Pc=None, Vc=None,
                 omega=None, Psat=None, Vml=None, best_fit=None,
                 load_data=True):
        self.CASRN = CASRN
        self.MW = MW
        self.Tm = Tm
        self.Tc = Tc
        self.Pc = Pc
        self.Vc = Vc
        self.omega = omega
        self.Psat = Psat
        self.Vml = Vml

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        liquid viscosity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        liquid viscosity above.'''

        self.tabular_data = {}
        '''tabular_data, dict: Stored (Ts, properties) for any
        tabular data; indexed by provided or autogenerated name.'''
        self.tabular_data_interpolators = {}
        '''tabular_data_interpolators, dict: Stored (extrapolator,
        spline) tuples which are interp1d instances for each set of tabular
        data; indexed by tuple of (name, interpolation_T,
        interpolation_property, interpolation_property_inv) to ensure that
        if an interpolation transform is altered, the old interpolator which
        had been created is no longer used.'''

        self.tabular_data_P = {}
        '''tabular_data_P, dict: Stored (Ts, Ps, properties) for any
        tabular data; indexed by provided or autogenerated name.'''
        self.tabular_data_interpolators_P = {}
        '''tabular_data_interpolators_P, dict: Stored (extrapolator,
        spline) tuples which are interp2d instances for each set of tabular
        data; indexed by tuple of (name, interpolation_T, interpolation_P,
        interpolation_property, interpolation_property_inv) to ensure that
        if an interpolation transform is altered, the old interpolator which
        had been created is no longer used.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by `T_dependent_property`.'''
        self.sorted_valid_methods_P = []
        '''sorted_valid_methods_P, list: Stored methods which were found valid
        at a specific temperature; set by `TP_dependent_property`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `T_dependent_property`.'''
        self.user_methods_P = []
        '''user_methods_P, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `TP_dependent_property`.'''

        self.all_methods = set()
        '''Set of all low-pressure methods available for a given CASRN and
        properties; filled by :obj:`load_all_methods`.'''
        self.all_methods_P = set()
        '''Set of all high-pressure methods available for a given CASRN and
        properties; filled by :obj:`load_all_methods`.'''

        self.load_all_methods(load_data)
        if best_fit is not None:
            self.set_best_fit(best_fit)
        else:
            methods = self.select_valid_methods(T=None, check_validity=False)
            if methods:
                self.set_method(methods[0])


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
        methods, methods_P = [], []
        Tmins, Tmaxs = [], []
        self.T_limits = T_limits = {}
        if load_data:
            if has_CoolProp() and self.CASRN in coolprop_dict:
                CP_f = coolprop_fluids[self.CASRN]
                if CP_f.has_mu:
                    self.CP_f = CP_f
                    methods.append(COOLPROP); methods_P.append(COOLPROP)
                    Tmins.append(self.CP_f.Tmin); Tmaxs.append(self.CP_f.Tc)
                    T_limits[COOLPROP] = (self.CP_f.Tmin, self.CP_f.Tc)
            if self.CASRN in miscdata.VDI_saturation_dict:
                methods.append(VDI_TABULAR)
                Ts, props = lookup_VDI_tabular_data(self.CASRN, 'Mu (l)')
                self.VDI_Tmin = Ts[0]
                self.VDI_Tmax = Ts[-1]
                self.tabular_data[VDI_TABULAR] = (Ts, props)
                Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
                T_limits[VDI_TABULAR] = (self.VDI_Tmin, self.VDI_Tmax)
            if self.CASRN in viscosity.mu_data_Dutt_Prasad.index:
                methods.append(DUTT_PRASAD)
                A, B, C, self.DUTT_PRASAD_Tmin, self.DUTT_PRASAD_Tmax = viscosity.mu_values_Dutt_Prasad[viscosity.mu_data_Dutt_Prasad.index.get_loc(self.CASRN)].tolist()
                self.DUTT_PRASAD_coeffs = [A - 3.0, B, C]
                Tmins.append(self.DUTT_PRASAD_Tmin); Tmaxs.append(self.DUTT_PRASAD_Tmax)
                T_limits[DUTT_PRASAD] = (self.DUTT_PRASAD_Tmin, self.DUTT_PRASAD_Tmax)
            if self.CASRN in viscosity.mu_data_VN3.index:
                methods.append(VISWANATH_NATARAJAN_3)
                A, B, C, self.VISWANATH_NATARAJAN_3_Tmin, self.VISWANATH_NATARAJAN_3_Tmax = viscosity.mu_values_VN3[viscosity.mu_data_VN3.index.get_loc(self.CASRN)].tolist()
                self.VISWANATH_NATARAJAN_3_coeffs = [A - 3.0, B, C]
                Tmins.append(self.VISWANATH_NATARAJAN_3_Tmin); Tmaxs.append(self.VISWANATH_NATARAJAN_3_Tmax)
                T_limits[VISWANATH_NATARAJAN_3] = (self.VISWANATH_NATARAJAN_3_Tmin, self.VISWANATH_NATARAJAN_3_Tmax)
            if self.CASRN in viscosity.mu_data_VN2.index:
                methods.append(VISWANATH_NATARAJAN_2)
                A, B, self.VISWANATH_NATARAJAN_2_Tmin, self.VISWANATH_NATARAJAN_2_Tmax = viscosity.mu_values_VN2[viscosity.mu_data_VN2.index.get_loc(self.CASRN)].tolist()
                self.VISWANATH_NATARAJAN_2_coeffs = [A - 4.605170185988092, B] # log(100) = 4.605170185988092
                Tmins.append(self.VISWANATH_NATARAJAN_2_Tmin); Tmaxs.append(self.VISWANATH_NATARAJAN_2_Tmax)
                T_limits[VISWANATH_NATARAJAN_2] = (self.VISWANATH_NATARAJAN_2_Tmin, self.VISWANATH_NATARAJAN_2_Tmax)
            if self.CASRN in viscosity.mu_data_VN2E.index:
                methods.append(VISWANATH_NATARAJAN_2E)
                C, D, self.VISWANATH_NATARAJAN_2E_Tmin, self.VISWANATH_NATARAJAN_2E_Tmax = viscosity.mu_values_VN2E[viscosity.mu_data_VN2E.index.get_loc(self.CASRN)].tolist()
                self.VISWANATH_NATARAJAN_2E_coeffs = [C, D]
                Tmins.append(self.VISWANATH_NATARAJAN_2E_Tmin); Tmaxs.append(self.VISWANATH_NATARAJAN_2E_Tmax)
                T_limits[VISWANATH_NATARAJAN_2E] = (self.VISWANATH_NATARAJAN_2E_Tmin, self.VISWANATH_NATARAJAN_2E_Tmax)
            if self.CASRN in viscosity.mu_data_Perrys_8E_2_313.index:
                methods.append(DIPPR_PERRY_8E)
                C1, C2, C3, C4, C5, self.Perrys2_313_Tmin, self.Perrys2_313_Tmax = viscosity.mu_values_Perrys_8E_2_313[viscosity.mu_data_Perrys_8E_2_313.index.get_loc(self.CASRN)].tolist()
                self.Perrys2_313_coeffs = [C1, C2, C3, C4, C5]
                Tmins.append(self.Perrys2_313_Tmin); Tmaxs.append(self.Perrys2_313_Tmax)
                T_limits[DIPPR_PERRY_8E] = (self.Perrys2_313_Tmin, self.Perrys2_313_Tmax)
            if self.CASRN in viscosity.mu_data_VDI_PPDS_7.index:
                methods.append(VDI_PPDS)
                # No temperature limits - ideally could use critical point
                self.VDI_PPDS_coeffs = viscosity.mu_values_PPDS_7[viscosity.mu_data_VDI_PPDS_7.index.get_loc(self.CASRN)].tolist()
        if all((self.MW, self.Tc, self.Pc, self.omega)):
            methods.append(LETSOU_STIEL)
            Tmins.append(self.Tc/4); Tmaxs.append(self.Tc) # TODO: test model at low T
        if all((self.MW, self.Tm, self.Tc, self.Pc, self.Vc, self.omega, self.Vml)):
            methods.append(PRZEDZIECKI_SRIDHAR)
            Tmins.append(self.Tm); Tmaxs.append(self.Tc) # TODO: test model at Tm
        if all([self.Tc, self.Pc, self.omega]):
            methods_P.append(LUCAS)
        self.all_methods = set(methods)
        self.all_methods_P = set(methods_P)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)

    def calculate(self, T, method):
        r'''Method to calculate low-pressure liquid viscosity at tempearture
        `T` with a given method.

        This method has no exception handling; see `T_dependent_property`
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
        elif method == LETSOU_STIEL:
            mu = Letsou_Stiel(T, self.MW, self.Tc, self.Pc, self.omega)
        elif method == PRZEDZIECKI_SRIDHAR:
            Vml = self.Vml(T) if hasattr(self.Vml, '__call__') else self.Vml
            mu = Przedziecki_Sridhar(T, self.Tm, self.Tc, self.Pc, self.Vc, Vml, self.omega, self.MW)
        elif method == VDI_PPDS:
            return PPDS9(T, *self.VDI_PPDS_coeffs)
        elif method == BESTFIT:
            if T < self.best_fit_Tmin:
                mu = (T - self.best_fit_Tmin)*self.best_fit_Tmin_slope + self.best_fit_Tmin_value
            elif T > self.best_fit_Tmax:
                mu = (T - self.best_fit_Tmax)*self.best_fit_Tmax_slope + self.best_fit_Tmax_value
            else:
                mu = 0.0
                for c in self.best_fit_coeffs:
                    mu = mu*T + c
            mu = exp(mu)
        elif method in self.tabular_data:
            mu = self.interpolate(T, method)
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
        if method == DUTT_PRASAD:
            if T < self.DUTT_PRASAD_Tmin or T > self.DUTT_PRASAD_Tmax:
                return False
        elif method == VISWANATH_NATARAJAN_3:
            if T < self.VISWANATH_NATARAJAN_3_Tmin or T > self.VISWANATH_NATARAJAN_3_Tmax:
                return False
        elif method == VISWANATH_NATARAJAN_2:
            if T < self.VISWANATH_NATARAJAN_2_Tmin or T > self.VISWANATH_NATARAJAN_2_Tmax:
                return False
        elif method == VISWANATH_NATARAJAN_2E:
            if T < self.VISWANATH_NATARAJAN_2E_Tmin or T > self.VISWANATH_NATARAJAN_2E_Tmax:
                return False
        elif method == DIPPR_PERRY_8E:
            if T < self.Perrys2_313_Tmin or T > self.Perrys2_313_Tmax:
                return False
        elif method == COOLPROP:
            if T < self.CP_f.Tmin or T < self.CP_f.Tt or T > self.CP_f.Tc:
                return False
        elif method in [LETSOU_STIEL, PRZEDZIECKI_SRIDHAR]:
            if T > self.Tc:
                return False
            # No lower limit
        elif method == VDI_PPDS:
            # If the derivative is positive, return invalid.
            # This is very important as no maximum temperatures are specified.
            if self.Tc and T > self.Tc:
                return False
            A, B, C, D, E = self.VDI_PPDS_coeffs
            term = (C - T)/(T - D)
            # Derived with sympy
            if term > 0:
                der = E*((-C + T)/(D - T))**(1/3.)*(A + 4*B*(-C + T)/(D - T))*(C - D)*exp(((-C + T)/(D - T))**(1/3.)*(A + B*(-C + T)/(D - T)))/(3*(C - T)*(D - T))
            else:
                der = E*((C - T)/(D - T))**(1/3.)*(-A*(C - D)*(D - T)**6 + B*(C - D)*(C - T)*(D - T)**5 + 3*B*(C - T)**2*(D - T)**5 - 3*B*(C - T)*(D - T)**6)*exp(-((C - T)/(D - T))**(1/3.)*(A*(D - T) - B*(C - T))/(D - T))/(3*(C - T)*(D - T)**7)
            return der < 0
        elif method == BESTFIT:
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

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent liquid viscosity at
        temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see `TP_dependent_property`
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
        elif method in self.tabular_data:
            mu = self.interpolate_P(T, P, method)
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
        validity = True
        if method == LUCAS:
            pass
        elif method == COOLPROP:
            validity = PhaseSI('T', T, 'P', P, self.CASRN) in ['liquid', 'supercritical_liquid']
        elif method in self.tabular_data:
            if not self.tabular_extrapolation_permitted:
                Ts, Ps, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1] or P < Ps[0] or P > Ps[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity



GHARAGHEIZI = 'GHARAGHEIZI'
YOON_THODOS = 'YOON_THODOS'
STIEL_THODOS = 'STIEL_THODOS'
LUCAS_GAS = 'LUCAS_GAS'

viscosity_gas_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, VDI_TABULAR, GHARAGHEIZI, YOON_THODOS,
                         STIEL_THODOS, LUCAS_GAS]
'''Holds all low-pressure methods available for the ViscosityGas
class, for use in iterating over them.'''
viscosity_gas_methods_P = [COOLPROP]
'''Holds all high-pressure methods available for the ViscosityGas
class, for use in iterating over them.'''


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
        If False, do not load property coefficients from data sources in files;
        [-]

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

    High pressure methods:

    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow, but unparalled in accuracy for pressure
        dependence.

    See Also
    --------
    Gharagheizi_gas_viscosity
    Yoon_Thodos
    Stiel_Thodos
    Lucas_gas

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
    '''
    name = 'Gas viscosity'
    units = 'Pa*s'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_P = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default.'''
    property_min = 0
    '''Mimimum valid value of gas viscosity; limiting condition at low pressure
    is 0.'''
    property_max = 1E-3
    '''Maximum valid value of gas viscosity. Might be too high, or too low.'''

    ranked_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, VDI_TABULAR, GHARAGHEIZI, YOON_THODOS,
                      STIEL_THODOS, LUCAS_GAS]
    '''Default rankings of the low-pressure methods.'''
    ranked_methods_P = [COOLPROP]
    '''Default rankings of the high-pressure methods.'''

    def __init__(self, CASRN='', MW=None, Tc=None, Pc=None, Zc=None,
                 dipole=None, Vmg=None, best_fit=None, load_data=True):
        self.CASRN = CASRN
        self.MW = MW
        self.Tc = Tc
        self.Pc = Pc
        self.Zc = Zc
        self.dipole = dipole
        self.Vmg = Vmg

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        gas viscosity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        gas viscosity above.'''


        self.tabular_data = {}
        '''tabular_data, dict: Stored (Ts, properties) for any
        tabular data; indexed by provided or autogenerated name.'''
        self.tabular_data_interpolators = {}
        '''tabular_data_interpolators, dict: Stored (extrapolator,
        spline) tuples which are interp1d instances for each set of tabular
        data; indexed by tuple of (name, interpolation_T,
        interpolation_property, interpolation_property_inv) to ensure that
        if an interpolation transform is altered, the old interpolator which
        had been created is no longer used.'''

        self.tabular_data_P = {}
        '''tabular_data_P, dict: Stored (Ts, Ps, properties) for any
        tabular data; indexed by provided or autogenerated name.'''
        self.tabular_data_interpolators_P = {}
        '''tabular_data_interpolators_P, dict: Stored (extrapolator,
        spline) tuples which are interp2d instances for each set of tabular
        data; indexed by tuple of (name, interpolation_T, interpolation_P,
        interpolation_property, interpolation_property_inv) to ensure that
        if an interpolation transform is altered, the old interpolator which
        had been created is no longer used.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by `T_dependent_property`.'''
        self.sorted_valid_methods_P = []
        '''sorted_valid_methods_P, list: Stored methods which were found valid
        at a specific temperature; set by `TP_dependent_property`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `T_dependent_property`.'''
        self.user_methods_P = []
        '''user_methods_P, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `TP_dependent_property`.'''

        self.all_methods = set()
        '''Set of all low-pressure methods available for a given CASRN and
        properties; filled by :obj:`load_all_methods`.'''
        self.all_methods_P = set()
        '''Set of all high-pressure methods available for a given CASRN and
        properties; filled by :obj:`load_all_methods`.'''

        self.load_all_methods(load_data)
        if best_fit is not None:
            self.set_best_fit(best_fit)
        else:
            methods = self.select_valid_methods(T=None, check_validity=False)
            if methods:
                self.set_method(methods[0])

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
        methods, methods_P = [], []
        Tmins, Tmaxs = [], []
        if load_data:
            if self.CASRN in miscdata.VDI_saturation_dict:
                methods.append(VDI_TABULAR)
                Ts, props = lookup_VDI_tabular_data(self.CASRN, 'Mu (g)')
                self.VDI_Tmin = Ts[0]
                self.VDI_Tmax = Ts[-1]
                self.tabular_data[VDI_TABULAR] = (Ts, props)
                Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
            if has_CoolProp() and self.CASRN in coolprop_dict:
                CP_f = coolprop_fluids[self.CASRN]
                if CP_f.has_mu:
                    self.CP_f = CP_f
                    methods.append(COOLPROP); methods_P.append(COOLPROP)
#                    T_limits[COOLPROP] = (self.CP_f.Tmin, self.CP_f.Tmax)
                    Tmins.append(self.CP_f.Tmin); Tmaxs.append(self.CP_f.Tmax)
            if self.CASRN in viscosity.mu_data_Perrys_8E_2_312.index:
                methods.append(DIPPR_PERRY_8E)
                C1, C2, C3, C4, self.Perrys2_312_Tmin, self.Perrys2_312_Tmax = viscosity.mu_values_Perrys_8E_2_312[viscosity.mu_data_Perrys_8E_2_312.index.get_loc(self.CASRN)].tolist()
                self.Perrys2_312_coeffs = [C1, C2, C3, C4]
                Tmins.append(self.Perrys2_312_Tmin); Tmaxs.append(self.Perrys2_312_Tmax)
            if self.CASRN in viscosity.mu_data_VDI_PPDS_8.index:
                methods.append(VDI_PPDS)
                self.VDI_PPDS_coeffs = viscosity.mu_values_PPDS_8[viscosity.mu_data_VDI_PPDS_8.index.get_loc(self.CASRN)].tolist()
                self.VDI_PPDS_coeffs.reverse() # in format for horner's scheme
        if all([self.Tc, self.Pc, self.MW]):
            methods.append(GHARAGHEIZI)
            methods.append(YOON_THODOS)
            methods.append(STIEL_THODOS)
            Tmins.append(0); Tmaxs.append(5E3)  # Intelligently set limit
            # GHARAGHEIZI turns nonsesical at ~15 K, YOON_THODOS fine to 0 K,
            # same as STIEL_THODOS
        if all([self.Tc, self.Pc, self.Zc, self.MW]):
            methods.append(LUCAS_GAS)
            Tmins.append(0); Tmaxs.append(1E3)
        self.all_methods = set(methods)
        self.all_methods_P = set(methods_P)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)

    def calculate(self, T, method):
        r'''Method to calculate low-pressure gas viscosity at
        tempearture `T` with a given method.

        This method has no exception handling; see `T_dependent_property`
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
            mu =  horner(self.VDI_PPDS_coeffs, T)
        elif method == YOON_THODOS:
            mu = Yoon_Thodos(T, self.Tc, self.Pc, self.MW)
        elif method == STIEL_THODOS:
            mu = Stiel_Thodos(T, self.Tc, self.Pc, self.MW)
        elif method == LUCAS_GAS:
            mu = Lucas_gas(T, self.Tc, self.Pc, self.Zc, self.MW, self.dipole, CASRN=self.CASRN)
        elif method in self.tabular_data:
            mu = self.interpolate(T, method)
        elif method == BESTFIT:
            if T < self.best_fit_Tmin:
                mu = (T - self.best_fit_Tmin)*self.best_fit_Tmin_slope + self.best_fit_Tmin_value
            elif T > self.best_fit_Tmax:
                mu = (T - self.best_fit_Tmax)*self.best_fit_Tmax_slope + self.best_fit_Tmax_value
            else:
                mu = 0.0
                for c in self.best_fit_coeffs:
                    mu = mu*T + c
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
        validity = True
        if method in [YOON_THODOS, STIEL_THODOS, LUCAS_GAS]:
            if T < 0 or T > 5000:
                # Arbitrary limit
                return False
        elif method == DIPPR_PERRY_8E:
            if T < self.Perrys2_312_Tmin or T > self.Perrys2_312_Tmax:
                return False
        elif method == GHARAGHEIZI:
            if T < 20 or T > 2E3:
                validity = False
                # Doesn't do so well as the other methods
        elif method == COOLPROP:
            if T < self.CP_f.Tmin or T > self.CP_f.Tmax:
                return False
        elif method == VDI_PPDS:
            pass # Polynomial always works
        elif method == BESTFIT:
            validity = True
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    return False
        else:
            raise Exception('Method not valid')
        return validity

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent gas viscosity
        at temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see `TP_dependent_property`
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
        elif method in self.tabular_data:
            mu = self.interpolate_P(T, P, method)
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
        elif method in self.tabular_data:
            if not self.tabular_extrapolation_permitted:
                Ts, Ps, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1] or P < Ps[0] or P > Ps[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity


LALIBERTE_MU = 'Laliberte'
MIXING_LOG_MOLAR = 'Logarithmic mixing, molar'
MIXING_LOG_MASS = 'Logarithmic mixing, mass'
SIMPLE = 'Simple'

viscosity_liquid_mixture_methods = [LALIBERTE_MU, MIXING_LOG_MOLAR, MIXING_LOG_MASS, SIMPLE]


class ViscosityLiquidMixture(MixtureProperty):
    '''Class for dealing with the viscosity of a liquid mixture as a
    function of temperature, pressure, and composition.
    Consists of one electrolyte-specific method, and logarithmic rules based
    on either mole fractions of mass fractions.

    Prefered method is :obj:`mixing_logarithmic` with mole
    fractions, or **Laliberte** if the mixture is aqueous and has electrolytes.

    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
    ViscosityLiquids : list[ViscosityLiquid], optional
        ViscosityLiquid objects created for all species in the mixture,
        normally created by :obj:`thermo.chemical.Chemical`.
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
        :obj:`thermo.utils.mixing_logarithmic`.
    **MIXING_LOG_MASS**:
        Logarithmic mole fraction mixing rule described in
        :obj:`thermo.utils.mixing_logarithmic`.
    **SIMPLE**:
        Linear mole fraction mixing rule described in
        :obj:`thermo.utils.mixing_simple`.

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
    '''Mimimum valid value of liquid viscosity.'''
    property_max = 2E8
    '''Maximum valid value of liquid viscosity. Generous limit, as
    the value is that of bitumen in a Pitch drop experiment.'''

    ranked_methods = [LALIBERTE_MU, MIXING_LOG_MOLAR, MIXING_LOG_MASS, SIMPLE]

    def __init__(self, CASs=[], ViscosityLiquids=[], MWs=[],
                 correct_pressure_pure=True):
        self.CASs = CASs
        self.ViscosityLiquids = self.pure_objs = ViscosityLiquids
        self.MWs = MWs

        self._correct_pressure_pure = correct_pressure_pure

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        liquid viscosity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        liquid viscosity above.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by `mixture_property`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `mixture_property`.'''
        self.all_methods = set()
        '''Set of all methods available for a given set of information;
        filled by :obj:`load_all_methods`.'''
        self.load_all_methods()
        self.set_best_fit_coeffs()

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
        methods = [MIXING_LOG_MOLAR, MIXING_LOG_MASS, SIMPLE]
        if len(self.CASs) > 1 and '7732-18-5' in self.CASs:
            wCASs = [i for i in self.CASs if i != '7732-18-5']
            if all([i in electrochem._Laliberte_Viscosity_ParametersDict for i in wCASs]):
                methods.append(LALIBERTE_MU)
                self.wCASs = wCASs
                self.index_w = self.CASs.index('7732-18-5')
        self.all_methods = set(methods)
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

        This method has no exception handling; see `mixture_property`
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
            ws = list(ws) ; ws.pop(self.index_w)
            return Laliberte_viscosity(T, ws, self.wCASs)

        if self._correct_pressure_pure:
            mus = []
            for obj in self.ViscosityLiquids:
                mu = obj.TP_dependent_property(T, P)
                if mu is None:
                    mu = obj.T_dependent_property(T)
                mus.append(mu)
        else:
            if self.locked:
                best_fit_data = self.best_fit_data
                Tmins, Tmaxs, coeffs = best_fit_data[0], best_fit_data[3], best_fit_data[6]
                mus = []
                for i in range(len(zs)):
                    if T < Tmins[i]:
                        mu = (T - Tmins[i])*best_fit_data[1][i] + best_fit_data[2][i]
                    elif T > Tmaxs[i]:
                        mu = (T - Tmaxs[i])*best_fit_data[4][i] + best_fit_data[5][i]
                    else:
                        mu = 0.0
                        for c in coeffs[i]:
                            mu = mu*T + c
                    mus.append(exp(mu))
            else:
                mus = [i.T_dependent_property(T) for i in self.ViscosityLiquids]
        if method == MIXING_LOG_MOLAR:
            ln_mu = 0.0
            for i in range(len(zs)):
                ln_mu += zs[i]*log(mus[i])
            return exp(ln_mu)
        elif method == MIXING_LOG_MASS:
            ln_mu = 0.0
            for i in range(len(ws)):
                ln_mu += ws[i]*log(mus[i])
            return exp(ln_mu)
        elif method == SIMPLE:
            mu = 0.0
            for i in range(len(zs)):
                mu += zs[i]*mus[i]
            return mu
        else:
            raise Exception('Method not valid')

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
        else:
            raise Exception('Method not valid')




BROKAW = 'Brokaw'
HERNING_ZIPPERER = 'Herning-Zipperer'
WILKE = 'Wilke'
viscosity_gas_mixture_methods = [BROKAW, HERNING_ZIPPERER, WILKE, SIMPLE]


class ViscosityGasMixture(MixtureProperty):
    '''Class for dealing with the viscosity of a gas mixture as a
    function of temperature, pressure, and composition.
    Consists of three gas viscosity specific mixing rules and a mole-weighted
    simple mixing rule.

    Prefered method is :obj:`Brokaw`.

    Parameters
    ----------
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    molecular_diameters : list[float], optional
        Lennard-Jones molecular diameters, [Angstrom]
    Stockmayers : list[float], optional
        Lennard-Jones depth of potential-energy minimum over k
        or epsilon_k, [K]
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
    ViscosityGases : list[ViscosityGas], optional
        ViscosityGas objects created for all species in the mixture,
        normally created by :obj:`thermo.chemical.Chemical`.
    correct_pressure_pure : bool, optional
        Whether to try to use the better pressure-corrected pure component
        models or to use only the T-only dependent pure species models, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`viscosity_liquid_mixture_methods`.

    **Brokaw**:
        Mixing rule described in :obj:`Brokaw`.
    **Herning-Zipperer**:
        Mixing rule described in :obj:`Herning_Zipperer`.
    **Wilke**:
        Mixing rule described in :obj:`Wilke`.
    **SIMPLE**:
        Mixing rule described in :obj:`thermo.utils.mixing_simple`.

    See Also
    --------
    Brokaw
    Herning_Zipperer
    Wilke

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    name = 'gas viscosity'
    units = 'Pa*s'
    property_min = 0
    '''Mimimum valid value of gas viscosity; limiting condition at low pressure
    is 0.'''
    property_max = 1E-3
    '''Maximum valid value of gas viscosity. Might be too high, or too low.'''

    ranked_methods = [BROKAW, HERNING_ZIPPERER, SIMPLE, WILKE]

    def __init__(self, MWs=[], molecular_diameters=[], Stockmayers=[], CASs=[],
                 ViscosityGases=[], correct_pressure_pure=True):
        self.MWs = MWs
        self.molecular_diameters = molecular_diameters
        self.Stockmayers = Stockmayers
        self.CASs = CASs
        self.ViscosityGases = self.pure_objs = ViscosityGases
        try:
            self.MW_roots = [i**0.5 for i in MWs]
            MWs_inv = [1.0/MWi for MWi in MWs]
            self.Wilke_t0s, self.Wilke_t1s, self.Wilke_t2s = Wilke_prefactors(MWs)
        except:
            pass
        self._correct_pressure_pure = correct_pressure_pure

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        gas viscosity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        gas viscosity above.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by `mixture_property`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `mixture_property`.'''
        self.all_methods = set()
        '''Set of all methods available for a given set of information;
        filled by :obj:`load_all_methods`.'''
        self.load_all_methods()

        self.set_best_fit_coeffs()

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
        methods = [SIMPLE]
        if none_and_length_check((self.MWs, self.molecular_diameters, self.Stockmayers)):
            methods.append(BROKAW)
        if none_and_length_check([self.MWs]):
            methods.extend([WILKE, HERNING_ZIPPERER])
        self.all_methods = set(methods)
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

        This method has no exception handling; see `mixture_property`
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
        if self._correct_pressure_pure:
            mus = []
            for obj in self.ViscosityGases:
                mu = obj.TP_dependent_property(T, P)
                if mu is None:
                    mu = obj.T_dependent_property(T)
                mus.append(mu)
        else:
            if self.locked:
                best_fit_data = self.best_fit_data
                Tmins, Tmaxs, coeffs = best_fit_data[0], best_fit_data[3], best_fit_data[6]
                mus = []
                for i in range(len(zs)):
                    if T < Tmins[i]:
                        mu = (T - Tmins[i])*best_fit_data[1][i] + best_fit_data[2][i]
                    elif T > Tmaxs[i]:
                        mu = (T - Tmaxs[i])*best_fit_data[4][i] + best_fit_data[5][i]
                    else:
                        mu = 0.0
                        for c in coeffs[i]:
                            mu = mu*T + c
                    mus.append(mu)
            else:
                mus = [i.T_dependent_property(T) for i in self.ViscosityGases]

        if method == SIMPLE:
            return mixing_simple(zs, mus)
        elif method == HERNING_ZIPPERER:
            return Herning_Zipperer(zs, mus, None, self.MW_roots)
        elif method == WILKE:
            return Wilke_prefactored(zs, mus, self.Wilke_t0s, self.Wilke_t1s, self.Wilke_t2s)
        elif method == BROKAW:
            return Brokaw(T, zs, mus, self.MWs, self.molecular_diameters, self.Stockmayers)
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


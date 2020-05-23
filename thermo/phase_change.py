# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['enthalpy_vaporization_methods', 'EnthalpyVaporization', 
           'enthalpy_sublimation_methods', 'EnthalpySublimation']

import os
import numpy as np
import pandas as pd

from fluids.numerics import horner, horner_and_der
from fluids.constants import R, pi, N_A

from chemicals.utils import log, isnan
from chemicals.utils import property_molar_to_mass, mixing_simple, none_and_length_check
from chemicals.dippr import EQ106
from chemicals import miscdata
from chemicals.miscdata import lookup_VDI_tabular_data
from chemicals import phase_change
from chemicals.phase_change import *
from thermo.vapor_pressure import VaporPressure
from thermo.utils import TDependentProperty
from thermo.coolprop import has_CoolProp, PropsSI, coolprop_dict, coolprop_fluids


COOLPROP = 'COOLPROP'
VDI_TABULAR = 'VDI_TABULAR'
CRC_HVAP_TB = 'CRC_HVAP_TB'
CRC_HVAP_298 = 'CRC_HVAP_298'
GHARAGHEIZI_HVAP_298 = 'GHARAGHEIZI_HVAP_298'
MORGAN_KOBAYASHI = 'MORGAN_KOBAYASHI'
SIVARAMAN_MAGEE_KOBAYASHI = 'SIVARAMAN_MAGEE_KOBAYASHI'
VELASCO = 'VELASCO'
PITZER = 'PITZER'
CLAPEYRON = 'CLAPEYRON'
DIPPR_PERRY_8E = 'DIPPR_PERRY_8E'
VDI_PPDS = 'VDI_PPDS'
ALIBAKHSHI = 'ALIBAKHSHI'

RIEDEL = 'RIEDEL'
CHEN = 'CHEN'
LIU = 'LIU'
VETERE = 'VETERE'
BESTFIT = 'Best fit'

enthalpy_vaporization_methods = [DIPPR_PERRY_8E, VDI_PPDS, COOLPROP, VDI_TABULAR, 
                                 MORGAN_KOBAYASHI,
                      SIVARAMAN_MAGEE_KOBAYASHI, VELASCO, PITZER, ALIBAKHSHI,
                      CRC_HVAP_TB, CRC_HVAP_298, GHARAGHEIZI_HVAP_298,
                      CLAPEYRON, RIEDEL, CHEN, VETERE, LIU]
'''Holds all methods available for the EnthalpyVaporization class, for use in
iterating over them.'''


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

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`enthalpy_vaporization_methods`.

    **CLAPEYRON**:
        The Clapeyron fundamental model desecribed in :obj:`Clapeyron`.
        This is the model which uses `Zl`, `Zg`, and `Psat`, all of which
        must be set at each temperature change to allow recalculation of
        the heat of vaporization.
    **MORGAN_KOBAYASHI**:
        The MK CSP model equation documented in :obj:`MK`.
    **SIVARAMAN_MAGEE_KOBAYASHI**:
        The SMK CSP model equation documented in :obj:`SMK`.
    **VELASCO**:
        The Velasco CSP model equation documented in :obj:`Velasco`.
    **PITZER**:
        The Pitzer CSP model equation documented in :obj:`Pitzer`.
    **RIEDEL**:
        The Riedel CSP model equation, valid at the boiling point only,
        documented in :obj:`Riedel`. This is adjusted with the :obj:`Watson`
        equation unless `Tc` is not available.
    **CHEN**:
        The Chen CSP model equation, valid at the boiling point only,
        documented in :obj:`Chen`. This is adjusted with the :obj:`Watson`
        equation unless `Tc` is not available.
    **VETERE**:
        The Vetere CSP model equation, valid at the boiling point only,
        documented in :obj:`Vetere`. This is adjusted with the :obj:`Watson`
        equation unless `Tc` is not available.
    **LIU**:
        The Liu CSP model equation, valid at the boiling point only,
        documented in :obj:`Liu`. This is adjusted with the :obj:`Watson`
        equation unless `Tc` is not available.
    **CRC_HVAP_TB**:
        The constant value available in [4]_ at the normal boiling point. This
        is adusted  with the :obj:`Watson` equation unless `Tc` is not
        available. Data is available for 707 chemicals.
    **CRC_HVAP_298**:
        The constant value available in [4]_ at 298.15 K. This
        is adusted  with the :obj:`Watson` equation unless `Tc` is not
        available. Data is available for 633 chemicals.
    **GHARAGHEIZI_HVAP_298**:
        The constant value available in [5]_ at 298.15 K. This
        is adusted  with the :obj:`Watson` equation unless `Tc` is not
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
        :obj:`thermo.dippr.EQ106` is used for its fluids.
    **ALIBAKHSHI**:
        One-constant limited temperature range regression method presented
        in [7]_, with constants for ~2000 chemicals from the DIPPR database.
        Valid up to 100 K below the critical point, and 50 K under the boiling
        point.

    See Also
    --------
    MK
    SMK
    Velasco
    Clapeyron
    Riedel
    Chen
    Vetere
    Liu
    Watson

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
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default; values below 0 will be obtained
    at high temperatures.'''
    property_min = 0
    '''Mimimum valid value of heat of vaporization. This occurs at the critical
    point exactly.'''
    property_max = 1E6
    '''Maximum valid of heat of vaporization. Set to twice the value in the
    available data.'''

    ranked_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, MORGAN_KOBAYASHI,
                      SIVARAMAN_MAGEE_KOBAYASHI, VELASCO, PITZER, VDI_TABULAR, 
                      ALIBAKHSHI, 
                      CRC_HVAP_TB, CRC_HVAP_298, GHARAGHEIZI_HVAP_298,
                      CLAPEYRON, RIEDEL, CHEN, VETERE, LIU]
    '''Default rankings of the available methods.'''
    boiling_methods = [RIEDEL, CHEN, VETERE, LIU]
    CSP_methods = [MORGAN_KOBAYASHI, SIVARAMAN_MAGEE_KOBAYASHI,
                   VELASCO, PITZER]
    Watson_exponent = 0.38
    '''Exponent used in the Watson equation'''

    def __init__(self, CASRN='', Tb=None, Tc=None, Pc=None, omega=None,
                 similarity_variable=None, Psat=None, Zl=None, Zg=None,
                 best_fit=None):
        self.CASRN = CASRN
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.similarity_variable = similarity_variable
        self.Psat = Psat
        self.Zl = Zl
        self.Zg = Zg

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        heat of vaporization under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        heat of vaporization above.'''

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

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by `T_dependent_property`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `T_dependent_property`.'''

        self.all_methods = set()
        '''Set of all methods available for a given CASRN and properties;
        filled by :obj:`load_all_methods`.'''
        if best_fit is not None:
            self.best_fit_Tc = best_fit[2]
            self.set_best_fit((best_fit[0], best_fit[1], best_fit[3]))

        self.load_all_methods()
    def as_best_fit(self):
        return '%s(best_fit=(%s, %s, %s, %s))' %(self.__class__.__name__,
                  repr(self.best_fit_Tmin), repr(self.best_fit_Tmax),
                  repr(self.best_fit_Tc), repr(self.best_fit_coeffs))

    def load_all_methods(self):
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
        if has_CoolProp and self.CASRN in coolprop_dict:
            methods.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
            Tmins.append(self.CP_f.Tt); Tmaxs.append(self.CP_f.Tc)
        if self.CASRN in miscdata.VDI_saturation_dict:
            methods.append(VDI_TABULAR)
            Ts, props = lookup_VDI_tabular_data(self.CASRN, 'Hvap')
            self.VDI_Tmin = Ts[0]
            self.VDI_Tmax = Ts[-1]
            self.tabular_data[VDI_TABULAR] = (Ts, props)
            Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
            
        if self.CASRN in phase_change.phase_change_data_Alibakhshi_Cs.index and self.Tc is not None:
            methods.append(ALIBAKHSHI)
            self.Alibakhshi_C = float(phase_change.phase_change_data_Alibakhshi_Cs.at[self.CASRN, 'C'])
            Tmaxs.append( max(self.Tc-100., 0) )
            
        if self.CASRN in phase_change.Hvap_data_CRC.index and not isnan(phase_change.Hvap_data_CRC.at[self.CASRN, 'HvapTb']):
            methods.append(CRC_HVAP_TB)
            self.CRC_HVAP_TB_Tb = float(phase_change.Hvap_data_CRC.at[self.CASRN, 'Tb'])
            self.CRC_HVAP_TB_Hvap = float(phase_change.Hvap_data_CRC.at[self.CASRN, 'HvapTb'])

        if self.CASRN in phase_change.Hvap_data_CRC.index and not isnan(phase_change.Hvap_data_CRC.at[self.CASRN, 'Hvap298']):
            methods.append(CRC_HVAP_298)
            self.CRC_HVAP_298 = float(phase_change.Hvap_data_CRC.at[self.CASRN, 'Hvap298'])
        if self.CASRN in phase_change.Hvap_data_Gharagheizi.index:
            methods.append(GHARAGHEIZI_HVAP_298)
            self.GHARAGHEIZI_HVAP_298_Hvap = float(phase_change.Hvap_data_Gharagheizi.at[self.CASRN, 'Hvap298'])

        if all((self.Tc, self.omega)):
            methods.extend(self.CSP_methods)
            Tmaxs.append(self.Tc); Tmins.append(0)
        if all((self.Tc, self.Pc)):
            methods.append(CLAPEYRON)
            Tmaxs.append(self.Tc); Tmins.append(0)
        if all((self.Tb, self.Tc, self.Pc)):
            methods.extend(self.boiling_methods)
            Tmaxs.append(self.Tc); Tmins.append(0)
        if self.CASRN in phase_change.phase_change_data_Perrys2_150.index:
            methods.append(DIPPR_PERRY_8E)
            Tc, C1, C2, C3, C4, self.Perrys2_150_Tmin, self.Perrys2_150_Tmax = phase_change.phase_change_data_Perrys2_150_values[phase_change.phase_change_data_Perrys2_150.index.get_loc(self.CASRN)].tolist()
            self.Perrys2_150_coeffs = [Tc, C1, C2, C3, C4]
            Tmins.append(self.Perrys2_150_Tmin); Tmaxs.append(self.Perrys2_150_Tmax)
        if self.CASRN in phase_change.phase_change_data_VDI_PPDS_4.index:
            Tc, A, B, C, D, E = phase_change.phase_change_data_VDI_PPDS_4_values[phase_change.phase_change_data_VDI_PPDS_4.index.get_loc(self.CASRN)].tolist()
            self.VDI_PPDS_coeffs = [A, B, C, D, E]
            self.VDI_PPDS_Tc = Tc
            methods.append(VDI_PPDS)
            Tmaxs.append(self.VDI_PPDS_Tc); 
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)

    def calculate(self, T, method):
        r'''Method to calculate heat of vaporization of a liquid at
        temperature `T` with a given method.

        This method has no exception handling; see `T_dependent_property`
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
        if method == BESTFIT:
            if T > self.best_fit_Tc:
                Hvap = 0
            else:
                Hvap = horner(self.best_fit_coeffs, log(1.0 - T/self.best_fit_Tc))

        elif method == COOLPROP:
            Hvap = PropsSI('HMOLAR', 'T', T, 'Q', 1, self.CASRN) - PropsSI('HMOLAR', 'T', T, 'Q', 0, self.CASRN)
        elif method == DIPPR_PERRY_8E:
            Hvap = EQ106(T, *self.Perrys2_150_coeffs)
        # CSP methods
        elif method == VDI_PPDS:
            A, B, C, D, E = self.VDI_PPDS_coeffs
            tau = 1. - T/self.VDI_PPDS_Tc
            Hvap = R*self.VDI_PPDS_Tc*(A*tau**(1/3.) + B*tau**(2/3.) + C*tau
                                       + D*tau**2 + E*tau**6)
        elif method == ALIBAKHSHI:
            Hvap = (4.5*pi*N_A)**(1/3.)*4.2E-7*(self.Tc-6.) - R/2.*T*log(T) + self.Alibakhshi_C*T
        elif method == MORGAN_KOBAYASHI:
            Hvap = MK(T, self.Tc, self.omega)
        elif method == SIVARAMAN_MAGEE_KOBAYASHI:
            Hvap = SMK(T, self.Tc, self.omega)
        elif method == VELASCO:
            Hvap = Velasco(T, self.Tc, self.omega)
        elif method == PITZER:
            Hvap = Pitzer(T, self.Tc, self.omega)
        elif method == CLAPEYRON:
            Zg = self.Zg(T) if hasattr(self.Zg, '__call__') else self.Zg
            Zl = self.Zl(T) if hasattr(self.Zl, '__call__') else self.Zl
            Psat = self.Psat(T) if hasattr(self.Psat, '__call__') else self.Psat
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
        elif method in self.tabular_data:
            Hvap = self.interpolate(T, method)
        # Adjust with the watson equation if estimated at Tb or Tc only
        if method in self.boiling_methods or (self.Tc and method in [CRC_HVAP_TB, CRC_HVAP_298, GHARAGHEIZI_HVAP_298]):
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
        to the :obj:`Watson` equation, with a temperature exponent as set in
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
        elif method in [CRC_HVAP_298, GHARAGHEIZI_HVAP_298]:
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
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        elif method == BESTFIT:
            validity = True
        elif method == CLAPEYRON:
            if not (self.Psat and T < self.Tc):
                validity = False
        else:
            raise Exception('Method not valid')
        return validity

### Heat of Sublimation


GHARAGHEIZI_HSUB_298 = 'GHARAGHEIZI_HSUB_298'
GHARAGHEIZI_HSUB = 'GHARAGHEIZI_HSUB'
CRC_HFUS_HVAP_TM = 'CRC_HFUS_HVAP_TM' # Gets Tm

enthalpy_sublimation_methods = [GHARAGHEIZI_HSUB, CRC_HFUS_HVAP_TM, 
                                GHARAGHEIZI_HSUB_298]
'''Holds all methods available for the EnthalpySublimation class, for use in
iterating over them.'''


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

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`enthalpy_sublimation_methods`.

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
    '''
    name = 'Enthalpy of sublimation'
    units = 'J/mol'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default; values below 0 will be obtained
    at high temperatures.'''
    property_min = 0
    '''Mimimum valid value of heat of vaporization. A theoretical concept only.'''
    property_max = 1E6
    '''Maximum valid of heat of sublimation. A theoretical concept only.'''

    ranked_methods = [GHARAGHEIZI_HSUB, CRC_HFUS_HVAP_TM, 
                      GHARAGHEIZI_HSUB_298]

    def __init__(self, CASRN='', Tm=None, Tt=None, Cpg=None, Cps=None,
                 Hvap=None, best_fit=None):
        self.CASRN = CASRN
        self.Tm = Tm
        self.Tt = Tt
        self.Cpg = Cpg
        self.Cps = Cps
        self.Hvap = Hvap

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        heat of sublimation under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        heat of sublimation above.'''

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

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by `T_dependent_property`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `T_dependent_property`.'''

        self.all_methods = set()
        '''Set of all methods available for a given CASRN and properties;
        filled by :obj:`load_all_methods`.'''
        if best_fit is not None:
            self.set_best_fit(best_fit)

        self.load_all_methods()
        
    def load_all_methods(self):
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
        if self.CASRN in phase_change.Hsub_data_Gharagheizi.index:
            methods.append(GHARAGHEIZI_HSUB_298)
            self.GHARAGHEIZI_Hsub = float(phase_change.Hsub_data_Gharagheizi.at[self.CASRN, 'Hsub'])
            if self.Cpg is not None and self.Cps is not None:
                methods.append(GHARAGHEIZI_HSUB)
        if self.CASRN in phase_change.Hfus_data_CRC.index:
            methods.append(CRC_HFUS_HVAP_TM)
            self.CRC_Hfus = phase_change.Hfus_data_CRC.at[self.CASRN, 'Hfus']
        
        try:
            Tmins.append(0.01*self.Tm)
            Tmaxs.append(self.Tm)
            Tmaxs.append(self.Tt)
        except:
            pass
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(i for i in Tmins if i is not None), max(i for i in Tmaxs if i is not None)
            
    def calculate(self, T, method):
        r'''Method to calculate heat of sublimation of a solid at
        temperature `T` with a given method.

        This method has no exception handling; see `T_dependent_property`
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
        if method == BESTFIT:
            if T < self.best_fit_Tmin:
                Hsub = (T - self.best_fit_Tmin)*self.best_fit_Tmin_slope + self.best_fit_Tmin_value
            elif T > self.best_fit_Tmax:
                Hsub = (T - self.best_fit_Tmax)*self.best_fit_Tmax_slope + self.best_fit_Tmax_value
            else:
                Hsub = horner(self.best_fit_coeffs, T)

        elif method == GHARAGHEIZI_HSUB_298:
            Hsub = self.GHARAGHEIZI_Hsub
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
        if method == BESTFIT:
            validity = True
        elif method in (GHARAGHEIZI_HSUB_298, GHARAGHEIZI_HSUB, CRC_HFUS_HVAP_TM):
            validity = True
        elif method == BESTFIT:
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


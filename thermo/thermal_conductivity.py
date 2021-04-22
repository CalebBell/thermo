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

This module contains implementations of :obj:`TPDependentProperty <thermo.utils.TPDependentProperty>`
representing liquid and vapor thermal conductivity. A variety of estimation
and data methods are available as included in the `chemicals` library.
Additionally liquid and vapor mixture thermal conductivity predictor objects
are implemented subclassing  :obj:`MixtureProperty <thermo.utils.MixtureProperty>`.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.


.. contents:: :local:

Pure Liquid Thermal Conductivity
================================
.. autoclass:: ThermalConductivityLiquid
    :members: calculate, calculate_P, test_method_validity, test_method_validity_P,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods, ranked_methods_P
    :undoc-members:
    :show-inheritance:
    :exclude-members:

The following variables are available to specify which method to use.

.. data:: COOLPROP
.. data:: DIPPR_PERRY_8E
.. data:: VDI_PPDS
.. data:: VDI_TABULAR
.. data:: GHARAGHEIZI_L
.. data:: SHEFFY_JOHNSON
.. data:: SATO_RIEDEL
.. data:: LAKSHMI_PRASAD
.. data:: BAHADORI_L
.. data:: NICOLA
.. data:: NICOLA_ORIGINAL

The following variables contain lists of available methods.

.. autodata:: thermal_conductivity_liquid_methods
.. autodata:: thermal_conductivity_liquid_methods_P

Pure Gas Thermal Conductivity
=============================
.. autoclass:: ThermalConductivityGas
    :members: calculate, calculate_P, test_method_validity, test_method_validity_P,
              name, property_max, property_min,
              units, ranked_methods, ranked_methods_P
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: thermal_conductivity_gas_methods
.. autodata:: thermal_conductivity_gas_methods_P


Mixture Liquid Thermal Conductivity
===================================
.. autoclass:: ThermalConductivityLiquidMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: thermal_conductivity_liquid_mixture_methods

Mixture Gas Thermal Conductivity
================================
.. autoclass:: ThermalConductivityGasMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: thermal_conductivity_gas_mixture_methods

'''

from __future__ import division

__all__ = [
 'ThermalConductivityGasMixture', 'ThermalConductivityLiquidMixture',
 'MAGOMEDOV', 'DIPPR_9H', 'FILIPPOV', 'LINDSAY_BROMLEY',
 'thermal_conductivity_liquid_methods', 'ThermalConductivityLiquid',

 'thermal_conductivity_gas_methods',
 'thermal_conductivity_gas_methods_P', 'ThermalConductivityGas',

'GHARAGHEIZI_L', 'NICOLA', 'NICOLA_ORIGINAL', 'SATO_RIEDEL', 'SHEFFY_JOHNSON',
'BAHADORI_L', 'LAKSHMI_PRASAD', 'MISSENARD', 'DIPPR_9G',

           ]

import os

from fluids.numerics import horner
from fluids.constants import R, R_inv, N_A, k
from chemicals.utils import log, exp, sqrt
from chemicals.utils import mixing_simple, none_and_length_check
from chemicals.dippr import EQ100, EQ102
from chemicals.thermal_conductivity import *
from chemicals import thermal_conductivity

from thermo.utils import TPDependentProperty, MixtureProperty
from chemicals import miscdata
from chemicals.miscdata import lookup_VDI_tabular_data
from thermo.coolprop import has_CoolProp, coolprop_dict, coolprop_fluids, CoolProp_T_dependent_property, PropsSI, PhaseSI, CoolProp_failing_PT_flashes
from thermo import electrochem
from thermo.electrochem import thermal_conductivity_Magomedov
from thermo.viscosity import ViscosityGas
from thermo.heat_capacity import HeatCapacityGas
from thermo.volume import VolumeGas


from thermo.utils import NEGLIGIBLE, DIPPR_PERRY_8E, POLY_FIT, VDI_TABULAR, VDI_PPDS, COOLPROP, LINEAR


GHARAGHEIZI_L = 'GHARAGHEIZI_L'
NICOLA = 'NICOLA'
NICOLA_ORIGINAL = 'NICOLA_ORIGINAL'
SATO_RIEDEL = 'SATO_RIEDEL'
SHEFFY_JOHNSON = 'SHEFFY_JOHNSON'
BAHADORI_L = 'BAHADORI_L'
LAKSHMI_PRASAD = 'LAKSHMI_PRASAD'
MISSENARD = 'MISSENARD'
DIPPR_9G = 'DIPPR_9G'

thermal_conductivity_liquid_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS,
                                       VDI_TABULAR, GHARAGHEIZI_L,
                                       SHEFFY_JOHNSON, SATO_RIEDEL,
                                       LAKSHMI_PRASAD, BAHADORI_L,
                                       NICOLA, NICOLA_ORIGINAL]
'''Holds all low-pressure methods available for the :obj:`ThermalConductivityLiquid`
class, for use in iterating over them.'''

thermal_conductivity_liquid_methods_P = [COOLPROP, DIPPR_9G, MISSENARD]
'''Holds all high-pressure methods available for the :obj:`ThermalConductivityLiquid`
class, for use in iterating over them.'''

class ThermalConductivityLiquid(TPDependentProperty):
    r'''Class for dealing with liquid thermal conductivity as a function of
    temperature and pressure.

    For low-pressure (at 1 atm while under the vapor pressure; along the
    saturation line otherwise) liquids, there is one source of tabular
    information, one polynomial-based method, 7 corresponding-states estimators,
    and the external library CoolProp.

    For high-pressure liquids (also, <1 atm liquids), there are two
    corresponding-states estimator, and the external library CoolProp.

    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
    MW : float, optional
        Molecular weight, [g/mol]
    Tm : float, optional
        Melting point, [K]
    Tb : float, optional
        Boiling point, [K]
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    omega : float, optional
        Acentric factor, [-]
    Hfus : float, optional
        Heat of fusion, [J/mol]
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    poly_fit : tuple(float, float, list[float]), optional
        Tuple of (Tmin, Tmax, coeffs) representing a prefered fit to the
        viscosity of the compound; the coefficients are evaluated with
        horner's method, and the input variable and output are transformed by
        the default transformations of this object; used instead of any other
        default low-pressure method if provided. [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    To iterate over all methods, use the lists stored in
    :obj:`thermal_conductivity_liquid_methods` and
    :obj:`thermal_conductivity_liquid_methods_P` for low and high pressure
    methods respectively.

    Low pressure methods:

    **GHARAGHEIZI_L**:
        CSP method, described in :obj:`Gharagheizi_liquid <chemicals.thermal_conductivity.Gharagheizi_liquid>`.
    **SATO_RIEDEL**:
        CSP method, described in :obj:`Sato_Riedel <chemicals.thermal_conductivity.Sato_Riedel>`.
    **NICOLA**:
        CSP method, described in :obj:`Nicola <chemicals.thermal_conductivity.Nicola>`.
    **NICOLA_ORIGINAL**:
        CSP method, described in :obj:`Nicola_original <chemicals.thermal_conductivity.Nicola_original>`.
    **SHEFFY_JOHNSON**:
        CSP method, described in :obj:`Sheffy_Johnson <chemicals.thermal_conductivity.Sheffy_Johnson>`.
    **BAHADORI_L**:
        CSP method, described in :obj:`Bahadori_liquid <chemicals.thermal_conductivity.Bahadori_liquid>`.
    **LAKSHMI_PRASAD**:
        CSP method, described in :obj:`Lakshmi_Prasad <chemicals.thermal_conductivity.Lakshmi_Prasad>`.
    **DIPPR_PERRY_8E**:
        A collection of 340 coefficient sets from the DIPPR database published
        openly in [3]_. Provides temperature limits for all its fluids.
        :obj:`EQ100 <chemicals.dippr.EQ100>` is used for its fluids.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS, published
        openly in [2]_. Covers a large temperature range, but does not
        extrapolate well at very high or very low temperatures. 271 compounds.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow.
    **VDI_TABULAR**:
        Tabular data in [2]_ along the saturation curve; interpolation is as
        set by the user or the default.

    High pressure methods:

    **DIPPR_9G**:
        CSP method, described in :obj:`DIPPR9G <chemicals.thermal_conductivity.DIPPR9G>`. Calculates a
        low-pressure thermal conductivity first from the low-pressure method.
    **MISSENARD**:
        CSP method, described in :obj:`Missenard <chemicals.thermal_conductivity.Missenard>`. Calculates a
        low-pressure thermal conductivity first from the low-pressure method.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow, but unparalled in accuracy for pressure
        dependence.

    See Also
    --------
    chemicals.thermal_conductivity.Sheffy_Johnson
    chemicals.thermal_conductivity.Sato_Riedel
    chemicals.thermal_conductivity.Lakshmi_Prasad
    chemicals.thermal_conductivity.Gharagheizi_liquid
    chemicals.thermal_conductivity.Nicola_original
    chemicals.thermal_conductivity.Nicola
    chemicals.thermal_conductivity.Bahadori_liquid
    chemicals.thermal_conductivity.DIPPR9G
    chemicals.thermal_conductivity.Missenard

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
    name = 'liquid thermal conductivity'
    units = 'W/m/K'
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
    '''Mimimum valid value of liquid thermal conductivity.'''
    property_max = 10.0
    '''Maximum valid value of liquid thermal conductivity. Generous limit.'''

    ranked_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, VDI_TABULAR,
                      GHARAGHEIZI_L, SHEFFY_JOHNSON, SATO_RIEDEL,
                      LAKSHMI_PRASAD, BAHADORI_L, NICOLA, NICOLA_ORIGINAL]
    '''Default rankings of the low-pressure methods.'''
    ranked_methods_P = [COOLPROP, DIPPR_9G, MISSENARD]
    '''Default rankings of the high-pressure methods.'''


    custom_args = ('MW', 'Tm', 'Tb', 'Tc', 'Pc', 'omega', 'Hfus')

    def __init__(self, CASRN='', MW=None, Tm=None, Tb=None, Tc=None, Pc=None,
                 omega=None, Hfus=None, extrapolation='linear', **kwargs):
        self.CASRN = CASRN
        self.MW = MW
        self.Tm = Tm
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.Hfus = Hfus

        super(ThermalConductivityLiquid, self).__init__(extrapolation, **kwargs)

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
            if self.CASRN in miscdata.VDI_saturation_dict:
                methods.append(VDI_TABULAR)
                Ts, props = lookup_VDI_tabular_data(self.CASRN, 'K (l)')
                self.VDI_Tmin = Ts[0]
                self.VDI_Tmax = Ts[-1]
                self.tabular_data[VDI_TABULAR] = (Ts, props)
                Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
                T_limits[VDI_TABULAR] = (self.VDI_Tmin, self.VDI_Tmax)
            if has_CoolProp() and self.CASRN in coolprop_dict:
                CP_f = coolprop_fluids[self.CASRN]
                if CP_f.has_k:
                    self.CP_f = CP_f
                    methods.append(COOLPROP); methods_P.append(COOLPROP)
                    Tmins.append(self.CP_f.Tmin); Tmaxs.append(self.CP_f.Tc)
                    T_limits[COOLPROP] = (self.CP_f.Tmin*1.001, self.CP_f.Tc*0.9999)
            if self.CASRN in thermal_conductivity.k_data_Perrys_8E_2_315.index:
                methods.append(DIPPR_PERRY_8E)
                C1, C2, C3, C4, C5, self.Perrys2_315_Tmin, self.Perrys2_315_Tmax = thermal_conductivity.k_values_Perrys_8E_2_315[thermal_conductivity.k_data_Perrys_8E_2_315.index.get_loc(self.CASRN)].tolist()
                self.Perrys2_315_coeffs = [C1, C2, C3, C4, C5]
                Tmins.append(self.Perrys2_315_Tmin); Tmaxs.append(self.Perrys2_315_Tmax)
                T_limits[DIPPR_PERRY_8E] = (self.Perrys2_315_Tmin, self.Perrys2_315_Tmax)
            if self.CASRN in thermal_conductivity.k_data_VDI_PPDS_9.index:
                A, B, C, D, E = thermal_conductivity.k_values_VDI_PPDS_9[thermal_conductivity.k_data_VDI_PPDS_9.index.get_loc(self.CASRN)].tolist()
                self.VDI_PPDS_coeffs = [A, B, C, D, E]
                self.VDI_PPDS_coeffs.reverse()
                methods.append(VDI_PPDS)
                T_limits[VDI_PPDS] = (1e-3, 1e4)
        if self.MW:
            methods.extend([BAHADORI_L, LAKSHMI_PRASAD])
            T_limits[BAHADORI_L] = (1e-3, 1e4)
            T_limits[LAKSHMI_PRASAD] = (1e-3, 50.0*(131.0*sqrt(self.MW) + 2771.0)/(50.0*self.MW**0.5 + 197.0))
            # Tmin and Tmax are not extended by these simple models, who often
            # give values of 0; BAHADORI_L even has 3 roots.
            # LAKSHMI_PRASAD works down to 0 K, and has an upper limit of
            # 50.0*(131.0*sqrt(M) + 2771.0)/(50.0*M**0.5 + 197.0)
            # where it becomes 0.
        if all([self.MW, self.Tm]):
            methods.append(SHEFFY_JOHNSON)
            Tmins.append(0); Tmaxs.append(self.Tm + 793.65)
            T_limits[SHEFFY_JOHNSON] = (1e-3, self.Tm + 793.65)
            # Works down to 0, has a nice limit at T = Tm+793.65 from Sympy
        if all([self.Tb, self.Pc, self.omega]):
            methods.append(GHARAGHEIZI_L)
            T_limits[GHARAGHEIZI_L] = (self.Tb, self.Tc)
            Tmins.append(self.Tb); Tmaxs.append(self.Tc)
            # Chosen as the model is weird
        if all([self.Tc, self.Pc, self.omega]):
            methods.append(NICOLA)
            T_limits[NICOLA] = (0.01*self.Tc, self.Tc)
        if all([self.Tb, self.Tc]):
            methods.append(SATO_RIEDEL)
            T_limits[SATO_RIEDEL] = (0.01*self.Tb, self.Tc-0.1)
        if all([self.Hfus, self.Tc, self.omega]):
            methods.append(NICOLA_ORIGINAL)
            T_limits[NICOLA_ORIGINAL] = (0.01*self.Tc, self.Tc-0.1)
        if all([self.Tc, self.Pc]):
            methods_P.extend([DIPPR_9G, MISSENARD])
        self.all_methods = set(methods)
        self.all_methods_P = set(methods_P)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)
        for m in self.ranked_methods_P:
            if m in self.all_methods_P:
                self.method_P = m
                break

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {COOLPROP : [CAS for CAS in coolprop_dict if (coolprop_fluids[CAS].has_k and CAS not in CoolProp_failing_PT_flashes)],
                VDI_TABULAR: list(miscdata.VDI_saturation_dict.keys()),
                DIPPR_PERRY_8E: thermal_conductivity.k_data_Perrys_8E_2_315.index,
                VDI_PPDS: thermal_conductivity.k_data_VDI_PPDS_9.index,
                }

    def calculate(self, T, method):
        r'''Method to calculate low-pressure liquid thermal conductivity at
        tempearture `T` with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature of the liquid, [K]
        method : str
            Name of the method to use

        Returns
        -------
        kl : float
            Thermal conductivity of the liquid at T and a low pressure, [W/m/K]
        '''
        if method == SHEFFY_JOHNSON:
            kl = Sheffy_Johnson(T, self.MW, self.Tm)
        elif method == SATO_RIEDEL:
            kl = Sato_Riedel(T, self.MW, self.Tb, self.Tc)
        elif method == GHARAGHEIZI_L:
            kl = Gharagheizi_liquid(T, self.MW, self.Tb, self.Pc, self.omega)
        elif method == NICOLA:
            kl = Nicola(T, self.MW, self.Tc, self.Pc, self.omega)
        elif method == NICOLA_ORIGINAL:
            kl = Nicola_original(T, self.MW, self.Tc, self.omega, self.Hfus)
        elif method == LAKSHMI_PRASAD:
            kl = Lakshmi_Prasad(T, self.MW)
        elif method == BAHADORI_L:
            kl = Bahadori_liquid(T, self.MW)
        elif method == DIPPR_PERRY_8E:
            kl = EQ100(T, *self.Perrys2_315_coeffs)
        elif method == VDI_PPDS:
            kl = horner(self.VDI_PPDS_coeffs, T)
        elif method == COOLPROP:
            kl = CoolProp_T_dependent_property(T, self.CASRN, 'L', 'l')
        elif method == POLY_FIT:
            if T < self.poly_fit_Tmin:
                kl = (T - self.poly_fit_Tmin)*self.poly_fit_Tmin_slope + self.poly_fit_Tmin_value
            elif T > self.poly_fit_Tmax:
                kl = (T - self.poly_fit_Tmax)*self.poly_fit_Tmax_slope + self.poly_fit_Tmax_value
            else:
                kl = horner(self.poly_fit_coeffs, T)
        else:
            return self._base_calculate(T, method)
        return kl

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent liquid thermal conductivity
        at temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate liquid thermal conductivity, [K]
        P : float
            Pressure at which to calculate liquid thermal conductivity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        kl : float
            Thermal conductivity of the liquid at T and P, [W/m/K]
        '''
        if method == DIPPR_9G:
            kl = self.T_dependent_property(T)
            kl = DIPPR9G(T, P, self.Tc, self.Pc, kl)
        elif method == MISSENARD:
            kl = self.T_dependent_property(T)
            kl = Missenard(T, P, self.Tc, self.Pc, kl)
        elif method == COOLPROP:
            kl = PropsSI('L', 'T', T, 'P', P, self.CASRN)
        else:
            return self._base_calculate_P(T, P, method)
        return kl

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a temperature-dependent
        low-pressure method. For CSP methods, the models **BAHADORI_L**,
        **LAKSHMI_PRASAD**, and **SHEFFY_JOHNSON** are considered valid for all
        temperatures. For methods **GHARAGHEIZI_L**, **NICOLA**,
        and **NICOLA_ORIGINAL**, the methods are considered valid up to 1.5Tc
        and down to 0 K. Method **SATO_RIEDEL** does not work above the
        critical point, so it is valid from 0 K to the critical point.

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
        if method == SATO_RIEDEL:
            if T > self.Tc:
                return False
                # Doesn't run, no lower limit though
        elif method in [GHARAGHEIZI_L, NICOLA, NICOLA_ORIGINAL]:
            if T > self.Tc*1.5:
                return False
            # No lower limit, give a wide margin of acceptability here
        elif method == DIPPR_PERRY_8E:
            if T < self.Perrys2_315_Tmin or T > self.Perrys2_315_Tmax:
                return False
        elif method in [BAHADORI_L, LAKSHMI_PRASAD, SHEFFY_JOHNSON]:
            pass
            # no limits at all
        elif method == VDI_PPDS:
            if self.Tc and T > self.Tc:
                return False
        elif method == COOLPROP:
            if T < self.CP_f.Tt or T > self.CP_f.Tc:
                return False
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    return False
        else:
            raise Exception('Method not valid')
        return True

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a high-pressure method. For
        **COOLPROP**, the fluid must be both a liquid and under the maximum
        pressure of the fluid's EOS. **MISSENARD** has defined limits;
        between 0.5Tc and 0.8Tc, and below 200Pc. The CSP method **DIPPR_9G**
        is considered valid for all temperatures and pressures.

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
        if method == MISSENARD:
            if T/self.Tc < 0.5 or T/self.Tc > 0.8 or P/self.Pc > 200:
                validity = False
        elif method == DIPPR_9G:
            if T < 0 or P < 0:
                validity = False
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

MAGOMEDOV = 'MAGOMEDOV'
DIPPR_9H = 'DIPPR_9H'
FILIPPOV = 'FILIPPOV'

thermal_conductivity_liquid_mixture_methods = [MAGOMEDOV, DIPPR_9H, FILIPPOV, LINEAR]
'''Holds all mixing rules available for the :obj:`ThermalConductivityLiquidMixture`
class, for use in iterating over them.'''


class ThermalConductivityLiquidMixture(MixtureProperty):
    '''Class for dealing with thermal conductivity of a liquid mixture as a
    function of temperature, pressure, and composition.
    Consists of two mixing rule specific to liquid thremal conductivity, one
    coefficient-based method for aqueous electrolytes, and mole weighted
    averaging. Most but not all methods are shown in [1]_.

    Prefered method is :obj:`DIPPR_9H <chemicals.thermal_conductivity.DIPPR9H>` which requires mass
    fractions, and pure component liquid thermal conductivities. This is
    substantially better than the ideal mixing rule based on mole fractions,
    **LINEAR**. :obj:`Filippov <chemicals.thermal_conductivity.Filippov>`
    is of similar accuracy but applicable to binary systems only.

    Parameters
    ----------
    CASs : str, optional
        The CAS numbers of all species in the mixture, [-]
    ThermalConductivityLiquids : list[ThermalConductivityLiquid], optional
        ThermalConductivityLiquid objects created for all species in the
        mixture, [-]
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    correct_pressure_pure : bool, optional
        Whether to try to use the better pressure-corrected pure component
        models or to use only the T-only dependent pure species models, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`thermal_conductivity_liquid_mixture_methods`.

    **DIPPR_9H**:
        Mixing rule described in :obj:`DIPPR9H <chemicals.thermal_conductivity.DIPPR9H>`.
    **FILIPPOV**:
        Mixing rule described in :obj:`Filippov <chemicals.thermal_conductivity.Filippov>`; for two binary systems only.
    **MAGOMEDOV**:
        Coefficient-based method for aqueous electrolytes only, described in
        :obj:`thermo.electrochem.thermal_conductivity_Magomedov`.
    **LINEAR**:
        Mixing rule described in :obj:`mixing_simple <chemicals.utils.mixing_simple>`.

    See Also
    --------
    chemicals.thermal_conductivity.DIPPR9H
    chemicals.thermal_conductivity.Filippov
    chemicals.thermal_conductivity.thermal_conductivity_Magomedov

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    name = 'liquid thermal conductivity'
    units = 'W/m/K'
    property_min = 0
    '''Mimimum valid value of liquid thermal conductivity.'''
    property_max = 10
    '''Maximum valid value of liquid thermal conductivity. Generous limit.'''

    ranked_methods = [MAGOMEDOV, DIPPR_9H, LINEAR, FILIPPOV]

    pure_references = ('ThermalConductivityLiquids',)
    pure_reference_types = (ThermalConductivityLiquid,)

    custom_args = ('MWs', )

    def __init__(self, CASs=[], ThermalConductivityLiquids=[], MWs=[],
                 **kwargs):
        self.CASs = CASs
        self.ThermalConductivityLiquids = ThermalConductivityLiquids
        self.MWs = MWs
        super(ThermalConductivityLiquidMixture, self).__init__(**kwargs)

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
        methods = [DIPPR_9H, LINEAR]
        if len(self.CASs) == 2:
            methods.append(FILIPPOV)
        if '7732-18-5' in self.CASs and len(self.CASs)>1:
            wCASs = [i for i in self.CASs if i != '7732-18-5']
            if all([i in electrochem.Magomedovk_thermal_cond.index for i in wCASs]):
                methods.append(MAGOMEDOV)
                self.wCASs = wCASs
                self.index_w = self.CASs.index('7732-18-5')

        self.all_methods = all_methods = set(methods)
        Tmins = [i.Tmin for i in self.ThermalConductivityLiquids if i.Tmin]
        Tmaxs = [i.Tmax for i in self.ThermalConductivityLiquids if i.Tmax]
        if Tmins:
            self.Tmin = max(Tmins)
        if Tmaxs:
            self.Tmax = max(Tmaxs)
        for m in self.ranked_methods:
            if m in all_methods:
                self.method = m
                break

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate thermal conductivity of a liquid mixture at
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
        k : float
            Thermal conductivity of the liquid mixture, [W/m/K]
        '''
        if method == MAGOMEDOV:
            k_w = self.ThermalConductivityLiquids[self.index_w](T, P)
            ws = list(ws) ; ws.pop(self.index_w)
            return thermal_conductivity_Magomedov(T, P, ws, self.wCASs, k_w)

        if self._correct_pressure_pure:
            ks = []
            for obj in self.ThermalConductivityLiquids:
                k = obj.TP_dependent_property(T, P)
                if k is None:
                    k = obj.T_dependent_property(T)
                ks.append(k)
        else:
            ks = [i.T_dependent_property(T) for i in self.ThermalConductivityLiquids]

        if method == LINEAR:
            return mixing_simple(zs, ks)
        elif method == DIPPR_9H:
            return DIPPR9H(ws, ks)
        elif method == FILIPPOV:
            return Filippov(ws, ks)
        else:
            raise Exception('Method not valid')

    def test_method_validity(self, T, P, zs, ws, method):
        r'''Method to test the validity of a specified method for the given
        conditions. If **MAGOMEDOV** is applicable (electrolyte system), no
        other methods are considered viable. Otherwise, there are no easy
        checks that can be performed here.

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
        if MAGOMEDOV in self.all_methods:
            if method in self.all_methods:
                return method == MAGOMEDOV
        if method in [LINEAR, DIPPR_9H, FILIPPOV]:
            return True
        else:
            raise Exception('Method not valid')


GHARAGHEIZI_G = 'GHARAGHEIZI_G'
CHUNG = 'CHUNG'
ELI_HANLEY = 'ELI_HANLEY'
ELI_HANLEY_DENSE = 'ELI_HANLEY_DENSE'
CHUNG_DENSE = 'CHUNG_DENSE'
EUCKEN_MOD = 'EUCKEN_MOD'
EUCKEN = 'EUCKEN'
BAHADORI_G = 'BAHADORI_G'
STIEL_THODOS_DENSE = 'STIEL_THODOS_DENSE'
DIPPR_9B = 'DIPPR_9B'



thermal_conductivity_gas_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, VDI_TABULAR, GHARAGHEIZI_G,
                                    DIPPR_9B, CHUNG, ELI_HANLEY, EUCKEN_MOD,
                                    EUCKEN, BAHADORI_G]
'''Holds all low-pressure methods available for the :obj:`ThermalConductivityGas`
class, for use in iterating over them.'''
thermal_conductivity_gas_methods_P = [COOLPROP, ELI_HANLEY_DENSE, CHUNG_DENSE,
                                      STIEL_THODOS_DENSE]
'''Holds all high-pressure methods available for the :obj:`ThermalConductivityGas`
class, for use in iterating over them.'''

class ThermalConductivityGas(TPDependentProperty):
    r'''Class for dealing with gas thermal conductivity as a function of
    temperature and pressure.

    For gases at atmospheric pressure, there are 7 corresponding-states
    estimators, one source of tabular information, and the external library
    CoolProp.

    For gases under the fluid's boiling point (at sub-atmospheric pressures),
    and high-pressure gases above the boiling point, there are three
    corresponding-states estimators, and the external library CoolProp.

    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
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
        Critical compressibility, [-]
    omega : float, optional
        Acentric factor, [-]
    dipole : float, optional
        Dipole moment of the fluid, [debye]
    Vmg : float or callable, optional
        Molar volume of the fluid at a pressure and temperature or callable for
        the same, [m^3/mol]
    Cpgm : float or callable, optional
        Molar constant-pressure heat capacity of the fluid at a pressure and
        temperature or callable for the same, [J/mol/K]
    mug : float or callable, optional
        Gas viscosity of the fluid at a pressure and temperature or callable
        for the same, [Pa*s]
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    poly_fit : tuple(float, float, list[float]), optional
        Tuple of (Tmin, Tmax, coeffs) representing a prefered fit to the
        viscosity of the compound; the coefficients are evaluated with
        horner's method, and the input variable and output are transformed by
        the default transformations of this object; used instead of any other
        default low-pressure method if provided. [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    To iterate over all methods, use the lists stored in
    :obj:`thermal_conductivity_gas_methods` and
    :obj:`thermal_conductivity_gas_methods_P` for low and high pressure
    methods respectively.

    Low pressure methods:

    **GHARAGHEIZI_G**:
        CSP method, described in :obj:`Gharagheizi_gas <chemicals.thermal_conductivity.Gharagheizi_gas>`.
    **DIPPR_9B**:
        CSP method, described in :obj:`DIPPR9B <chemicals.thermal_conductivity.DIPPR9B>`.
    **CHUNG**:
        CSP method, described in :obj:`Chung <chemicals.thermal_conductivity.Chung>`.
    **ELI_HANLEY**:
        CSP method, described in :obj:`Eli_Hanley <chemicals.thermal_conductivity.Eli_Hanley>`.
    **EUCKEN_MOD**:
        CSP method, described in :obj:`Eucken_modified <chemicals.thermal_conductivity.Eucken_modified>`.
    **EUCKEN**:
        CSP method, described in :obj:`Eucken <chemicals.thermal_conductivity.Eucken>`.
    **BAHADORI_G**:
        CSP method, described in :obj:`Bahadori_gas <chemicals.thermal_conductivity.Bahadori_gas>`.
    **DIPPR_PERRY_8E**:
        A collection of 345 coefficient sets from the DIPPR database published
        openly in [3]_. Provides temperature limits for all its fluids.
        :obj:`chemicals.dippr.EQ102` is used for its fluids.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS, published
        openly in [2]_. Covers a large temperature range, but does not
        extrapolate well at very high or very low temperatures. 275 compounds.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow.
    **VDI_TABULAR**:
        Tabular data in [2]_ along the saturation curve; interpolation is as
        set by the user or the default.

    High pressure methods:

    **STIEL_THODOS_DENSE**:
        CSP method, described in :obj:`Stiel_Thodos_dense <chemicals.thermal_conductivity.Stiel_Thodos_dense>`. Calculates a
        low-pressure thermal conductivity first.
    **ELI_HANLEY_DENSE**:
        CSP method, described in :obj:`Eli_Hanley_dense <chemicals.thermal_conductivity.Eli_Hanley_dense>`. Calculates a
        low-pressure thermal conductivity first.
    **CHUNG_DENSE**:
        CSP method, described in :obj:`Chung_dense <chemicals.thermal_conductivity.Chung_dense>`. Calculates a
        low-pressure thermal conductivity first.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow, but unparalled in accuracy for pressure
        dependence.

    See Also
    --------
    chemicals.thermal_conductivity.Bahadori_gas
    chemicals.thermal_conductivity.Gharagheizi_gas
    chemicals.thermal_conductivity.Eli_Hanley
    chemicals.thermal_conductivity.Chung
    chemicals.thermal_conductivity.DIPPR9B
    chemicals.thermal_conductivity.Eucken_modified
    chemicals.thermal_conductivity.Eucken
    chemicals.thermal_conductivity.Stiel_Thodos_dense
    chemicals.thermal_conductivity.Eli_Hanley_dense
    chemicals.thermal_conductivity.Chung_dense

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
    name = 'gas thermal conductivity'
    units = 'W/m/K'
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
    '''Mimimum valid value of gas thermal conductivity.'''
    property_max = 10
    '''Maximum valid value of gas thermal conductivity. Generous limit.'''

    ranked_methods = [COOLPROP, VDI_PPDS, DIPPR_PERRY_8E, VDI_TABULAR, GHARAGHEIZI_G, DIPPR_9B,
                      CHUNG, ELI_HANLEY, EUCKEN_MOD, EUCKEN,
                      BAHADORI_G]
    '''Default rankings of the low-pressure methods.'''
    ranked_methods_P = [COOLPROP, ELI_HANLEY_DENSE, CHUNG_DENSE,
                        STIEL_THODOS_DENSE]
    '''Default rankings of the high-pressure methods.'''

    obj_references = pure_references = ('mug', 'Vmg', 'Cpgm')
    obj_references_types = pure_reference_types = (ViscosityGas, VolumeGas, HeatCapacityGas)

    custom_args = ('MW', 'Tb', 'Tc', 'Pc', 'Vc', 'Zc', 'omega', 'dipole',
                   'Vmg', 'Cpgm', 'mug')

    def __init__(self, CASRN='', MW=None, Tb=None, Tc=None, Pc=None, Vc=None,
                 Zc=None, omega=None, dipole=None, Vmg=None, Cpgm=None, mug=None,
                 extrapolation='linear', **kwargs):
        self.CASRN = CASRN
        self.MW = MW
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.Vc = Vc
        self.Zc = Zc
        self.omega = omega
        self.dipole = dipole
        self.Vmg = Vmg
        self.Cpgm = Cpgm
        self.mug = mug

        super(ThermalConductivityGas, self).__init__(extrapolation, **kwargs)


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
            if self.CASRN in miscdata.VDI_saturation_dict:
                methods.append(VDI_TABULAR)
                Ts, props = lookup_VDI_tabular_data(self.CASRN, 'K (g)')
                self.VDI_Tmin = Ts[0]
                self.VDI_Tmax = Ts[-1]
                self.tabular_data[VDI_TABULAR] = (Ts, props)
                Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
                T_limits[VDI_TABULAR] = (self.VDI_Tmin, self.VDI_Tmax)
            if has_CoolProp() and self.CASRN in coolprop_dict:
                CP_f = coolprop_fluids[self.CASRN]
                if CP_f.has_k:
                    self.CP_f = CP_f
                    methods.append(COOLPROP); methods_P.append(COOLPROP)
                    Tmins.append(self.CP_f.Tmin); Tmaxs.append(self.CP_f.Tc)
                    T_limits[COOLPROP] = (self.CP_f.Tmin, self.CP_f.Tc*0.9999)
            if self.CASRN in thermal_conductivity.k_data_Perrys_8E_2_314.index:
                methods.append(DIPPR_PERRY_8E)
                C1, C2, C3, C4, self.Perrys2_314_Tmin, self.Perrys2_314_Tmax = thermal_conductivity.k_values_Perrys_8E_2_314[thermal_conductivity.k_data_Perrys_8E_2_314.index.get_loc(self.CASRN)].tolist()
                self.Perrys2_314_coeffs = [C1, C2, C3, C4]
                Tmins.append(self.Perrys2_314_Tmin); Tmaxs.append(self.Perrys2_314_Tmax)
                T_limits[DIPPR_PERRY_8E] = (self.Perrys2_314_Tmin, self.Perrys2_314_Tmax)
            if self.CASRN in thermal_conductivity.k_data_VDI_PPDS_10.index:
                A, B, C, D, E = thermal_conductivity.k_values_VDI_PPDS_10[thermal_conductivity.k_data_VDI_PPDS_10.index.get_loc(self.CASRN)].tolist()
                self.VDI_PPDS_coeffs = [A, B, C, D, E]
                self.VDI_PPDS_coeffs.reverse()
                methods.append(VDI_PPDS)
                T_limits[VDI_PPDS] = (1e-3, 10000.0)
        if all((self.MW, self.Tb, self.Pc, self.omega)):
            methods.append(GHARAGHEIZI_G)
            # Turns negative at low T; do not set Tmin
            Tmaxs.append(3000)
            T_limits[GHARAGHEIZI_G] = (1e-3, 3000.0)
        if all((self.Cpgm, self.mug, self.MW, self.Tc)):
            methods.append(DIPPR_9B)
            Tmins.append(0.01); Tmaxs.append(1E4)  # No limit here
            T_limits[DIPPR_9B] = (1e-2, 1e4)
        if all((self.Cpgm, self.mug, self.MW, self.Tc, self.omega)):
            methods.append(CHUNG)
            Tmins.append(0.01); Tmaxs.append(1E4)  # No limit
            T_limits[CHUNG] = (1e-2, 1e4)
        if all((self.Cpgm, self.MW, self.Tc, self.Vc, self.Zc, self.omega)):
            methods.append(ELI_HANLEY)
            Tmaxs.append(1E4)  # Numeric error at low T
            T_limits[ELI_HANLEY] = (self.Tc*0.4, 1e4)
        if all((self.Cpgm, self.mug, self.MW)):
            methods.append(EUCKEN_MOD)
            methods.append(EUCKEN)
            Tmins.append(0.01); Tmaxs.append(1E4)  # No limits
            T_limits[EUCKEN] = T_limits[EUCKEN_MOD] = (1e-2, 1e4)
        if self.MW:
            methods.append(BAHADORI_G)
            T_limits[BAHADORI_G] = (1e-2, 1e4)
            # Terrible method, so don't set methods
        if all([self.MW, self.Tc, self.Vc, self.Zc, self.omega]):
            methods_P.append(ELI_HANLEY_DENSE)
        if all([self.MW, self.Tc, self.Vc, self.omega, self.dipole]):
            methods_P.append(CHUNG_DENSE)
        if all([self.MW, self.Tc, self.Pc, self.Vc, self.Zc]):
            methods_P.append(STIEL_THODOS_DENSE)
        self.all_methods = set(methods)
        self.all_methods_P = set(methods_P)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)
        for m in self.ranked_methods_P:
            if m in self.all_methods_P:
                self.method_P = m
                break

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {COOLPROP : [CAS for CAS in coolprop_dict if (coolprop_fluids[CAS].has_k and CAS not in CoolProp_failing_PT_flashes)],
                VDI_TABULAR: list(miscdata.VDI_saturation_dict.keys()),
                DIPPR_PERRY_8E: thermal_conductivity.k_data_Perrys_8E_2_314.index,
                VDI_PPDS: thermal_conductivity.k_data_VDI_PPDS_10.index,
                }

    def calculate(self, T, method):
        r'''Method to calculate low-pressure gas thermal conductivity at
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
        kg : float
            Thermal conductivity of the gas at T and a low pressure, [W/m/K]
        '''
        if method in (DIPPR_9B, CHUNG, ELI_HANLEY, EUCKEN_MOD, EUCKEN):
            Cvgm = self.Cpgm(T)-R if hasattr(self.Cpgm, '__call__') else self.Cpgm - R
            if method != ELI_HANLEY:
                mug = self.mug(T) if hasattr(self.mug, '__call__') else self.mug
        if method == GHARAGHEIZI_G:
            kg = Gharagheizi_gas(T, self.MW, self.Tb, self.Pc, self.omega)
        elif method == DIPPR_9B:
            kg = DIPPR9B(T, self.MW, Cvgm, mug, self.Tc)
        elif method == CHUNG:
            kg = Chung(T, self.MW, self.Tc, self.omega, Cvgm, mug)
        elif method == ELI_HANLEY:
            kg = Eli_Hanley(T, self.MW, self.Tc, self.Vc, self.Zc, self.omega, Cvgm)
        elif method == EUCKEN_MOD:
            kg = Eucken_modified(self.MW, Cvgm, mug)
        elif method == EUCKEN:
            kg = Eucken(self.MW, Cvgm, mug)
        elif method == DIPPR_PERRY_8E:
            kg = EQ102(T, *self.Perrys2_314_coeffs)
        elif method == VDI_PPDS:
            kg = horner(self.VDI_PPDS_coeffs, T)
        elif method == BAHADORI_G:
            kg = Bahadori_gas(T, self.MW)
        elif method == COOLPROP:
            kg = CoolProp_T_dependent_property(T, self.CASRN, 'L', 'g')
        elif method == POLY_FIT:
            if T < self.poly_fit_Tmin:
                kg = (T - self.poly_fit_Tmin)*self.poly_fit_Tmin_slope + self.poly_fit_Tmin_value
            elif T > self.poly_fit_Tmax:
                kg = (T - self.poly_fit_Tmax)*self.poly_fit_Tmax_slope + self.poly_fit_Tmax_value
            else:
                kg = horner(self.poly_fit_coeffs, T)
        else:
            return self._base_calculate(T, method)
        return kg

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent gas thermal conductivity
        at temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate gas thermal conductivity, [K]
        P : float
            Pressure at which to calculate gas thermal conductivity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        kg : float
            Thermal conductivity of the gas at T and P, [W/m/K]
        '''
        if method == ELI_HANLEY_DENSE:
            Vmg = self.Vmg(T, P) if hasattr(self.Vmg, '__call__') else self.Vmg
            Cpgm = self.Cpgm(T) if hasattr(self.Cpgm, '__call__') else self.Cpgm
            kg = Eli_Hanley_dense(T, self.MW, self.Tc, self.Vc, self.Zc, self.omega, Cpgm-R, Vmg)
        elif method == CHUNG_DENSE:
            Vmg = self.Vmg(T, P) if hasattr(self.Vmg, '__call__') else self.Vmg
            Cpgm = self.Cpgm(T) if hasattr(self.Cpgm, '__call__') else self.Cpgm
            mug = self.mug(T, P) if hasattr(self.mug, '__call__') else self.mug
            kg = Chung_dense(T, self.MW, self.Tc, self.Vc, self.omega, Cpgm-R, Vmg, mug, self.dipole)
        elif method == STIEL_THODOS_DENSE:
            kg = self.T_dependent_property(T)
            Vmg = self.Vmg(T, P) if hasattr(self.Vmg, '__call__') else self.Vmg
            kg = Stiel_Thodos_dense(T, self.MW, self.Tc, self.Pc, self.Vc, self.Zc, Vmg, kg)
        elif method == COOLPROP:
            kg = PropsSI('L', 'T', T, 'P', P, self.CASRN)
        else:
            return self._base_calculate_P(T, P, method)
        return kg

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a temperature-dependent
        low-pressure method. For CSP methods, the all methods are considered
        valid from 0 K and up.

        For tabular data, extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the extrapolation
        is considered valid for all temperatures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.
        **GHARAGHEIZI_G** and **BAHADORI_G** are known to sometimes produce
        negative results.

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
        if method in [GHARAGHEIZI_G, DIPPR_9B, CHUNG, ELI_HANLEY, EUCKEN_MOD,
                      EUCKEN, BAHADORI_G, VDI_PPDS]:
            pass
        elif method == DIPPR_PERRY_8E:
            if T < self.Perrys2_314_Tmin or T > self.Perrys2_314_Tmax:
                return False
        elif method == COOLPROP:
            if T < self.CP_f.Tmin or T > self.CP_f.Tmax:
                return False
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    return False
        else:
            raise Exception('Method not valid')
        return True

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a high-pressure method. For
        **COOLPROP**, the fluid must be both a gas and under the maximum
        pressure of the fluid's EOS. The CSP method **ELI_HANLEY_DENSE**,
        **CHUNG_DENSE**, and **STIEL_THODOS_DENSE** are considered valid for
        all temperatures and pressures.

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
        if method in [ELI_HANLEY_DENSE, CHUNG_DENSE, STIEL_THODOS_DENSE]:
            if T < 0 or P < 0:
                validity = False
            # no better checks known
        elif method == COOLPROP:
            if T < self.CP_f.Tmin or T > self.CP_f.Tmax or P > self.CP_f.Pmax:
                return False
            else:
                return PhaseSI('T', T, 'P', P, self.CASRN) in ['gas', 'supercritical_gas', 'supercritical', 'supercritical_liquid']
        elif method in self.tabular_data:
            if not self.tabular_extrapolation_permitted:
                Ts, Ps, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1] or P < Ps[0] or P > Ps[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity


LINDSAY_BROMLEY = 'LINDSAY_BROMLEY'
thermal_conductivity_gas_mixture_methods = [LINDSAY_BROMLEY, LINEAR]
'''Holds all mixing rules available for the :obj:`ThermalConductivityGasMixture`
class, for use in iterating over them.'''

class ThermalConductivityGasMixture(MixtureProperty):
    '''Class for dealing with thermal conductivity of a gas mixture as a
    function of temperature, pressure, and composition.
    Consists of one mixing rule specific to gas thremal conductivity, and mole
    weighted averaging.

    Prefered method is :obj:`Lindsay_Bromley <chemicals.thermal_conductivity.Lindsay_Bromley>` which requires mole
    fractions, pure component viscosities and thermal conductivities, and the
    boiling point and molecular weight of each pure component. This is
    substantially better than the ideal mixing rule based on mole fractions,
    **LINEAR** which is also available. More information on this topic can
    be found in [1]_.

    Parameters
    ----------
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    Tbs : list[float], optional
        Boiling points of all species in the mixture, [K]
    CASs : str, optional
        The CAS numbers of all species in the mixture
    ThermalConductivityGases : list[ThermalConductivityGas], optional
        ThermalConductivityGas objects created for all species in the mixture,
        [-]
    ViscosityGases : list[ViscosityGas], optional
        ViscosityGas objects created for all species in the mixture, [-]
    correct_pressure_pure : bool, optional
        Whether to try to use the better pressure-corrected pure component
        models or to use only the T-only dependent pure species models, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`thermal_conductivity_gas_methods`.

    **LINDSAY_BROMLEY**:
        Mixing rule described in :obj:`Lindsay_Bromley <chemicals.thermal_conductivity.Lindsay_Bromley>`.
    **LINEAR**:
        Mixing rule described in :obj:`mixing_simple <chemicals.utils.mixing_simple>`.

    See Also
    --------
    chemicals.thermal_conductivity.Lindsay_Bromley

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    name = 'gas thermal conductivity'
    units = 'W/m/K'
    property_min = 0.
    '''Mimimum valid value of gas thermal conductivity.'''
    property_max = 10.
    '''Maximum valid value of gas thermal conductivity. Generous limit.'''

    ranked_methods = [LINDSAY_BROMLEY, LINEAR]

    pure_references = ('ViscosityGases', 'ThermalConductivityGases')
    pure_reference_types = (ViscosityGas, ThermalConductivityGas)

    custom_args = ('MWs', 'Tbs', )

    def __init__(self, MWs=[], Tbs=[], CASs=[], ThermalConductivityGases=[],
                 ViscosityGases=[],  **kwargs):
        self.MWs = MWs
        self.Tbs = Tbs
        self.CASs = CASs
        self.ThermalConductivityGases = ThermalConductivityGases
        self.ViscosityGases = ViscosityGases

        super(ThermalConductivityGasMixture, self).__init__(**kwargs)


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
        methods = []
        methods.append(LINEAR)
        if none_and_length_check((self.Tbs, self.MWs)):
            methods.append(LINDSAY_BROMLEY)
        self.all_methods = all_methods = set(methods)
        Tmins = [i.Tmin for i in self.ThermalConductivityGases if i.Tmin]
        Tmaxs = [i.Tmax for i in self.ThermalConductivityGases if i.Tmax]
        if Tmins:
            self.Tmin = max(Tmins)
        if Tmaxs:
            self.Tmax = max(Tmaxs)
        for m in self.ranked_methods:
            if m in all_methods:
                self.method = m
                break

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate thermal conductivity of a gas mixture at
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
        kg : float
            Thermal conductivity of gas mixture, [W/m/K]
        '''
        if self._correct_pressure_pure:
            ks = []
            for obj in self.ThermalConductivityGases:
                k = obj.TP_dependent_property(T, P)
                if k is None:
                    k = obj.T_dependent_property(T)
                ks.append(k)
        else:
            ks = [i.T_dependent_property(T) for i in self.ThermalConductivityGases]

        if method == LINEAR:
            return mixing_simple(zs, ks)
        elif method == LINDSAY_BROMLEY:
            if self._correct_pressure_pure:
                mus = []
                for obj in self.ViscosityGases:
                    mu = obj.TP_dependent_property(T, P)
                    if mu is None:
                        mu = obj.T_dependent_property(T)
                    mus.append(mu)
            else:
                mus = [i.T_dependent_property(T) for i in self.ViscosityGases]
            return Lindsay_Bromley(T=T, ys=zs, ks=ks, mus=mus, Tbs=self.Tbs, MWs=self.MWs)
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
        if method in [LINEAR, LINDSAY_BROMLEY]:
            return True
        else:
            raise Exception('Method not valid')


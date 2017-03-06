# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['Dutt_Prasad', 'VN3_data', 'VN2_data', 'VN2E_data', 'Perrys2_313',
'Perrys2_312','VDI_PPDS_7', 'VDI_PPDS_8',
'ViswanathNatarajan2', 'ViswanathNatarajan2Exponential', 'ViswanathNatarajan3',
'Letsou_Stiel', 'Przedziecki_Sridhar', 'viscosity_liquid_methods', 
'viscosity_liquid_methods_P', 'ViscosityLiquid', 'ViscosityGas', 'Lucas', 
'Yoon_Thodos', 'Stiel_Thodos', 'lucas_gas', 
'Gharagheizi_gas_viscosity', 'viscosity_gas_methods', 'viscosity_gas_methods_P', 
'Herning_Zipperer', 'Wilke', 'Brokaw', 
'viscosity_index', 'ViscosityLiquidMixture', 'ViscosityGasMixture']

import os
import numpy as np
import pandas as pd

from thermo.utils import log, exp
from thermo.utils import horner, none_and_length_check, mixing_simple, mixing_logarithmic, TPDependentProperty, MixtureProperty
from thermo.miscdata import _VDISaturationDict, VDI_tabular_data
from thermo.electrochem import _Laliberte_Viscosity_ParametersDict, Laliberte_viscosity
from thermo.coolprop import has_CoolProp, PropsSI, PhaseSI, coolprop_fluids, coolprop_dict, CoolProp_T_dependent_property
from thermo.dippr import EQ101, EQ102

folder = os.path.join(os.path.dirname(__file__), 'Viscosity')

Dutt_Prasad = pd.read_csv(os.path.join(folder, 'Dutt Prasad 3 term.tsv'),
                          sep='\t', index_col=0)
_Dutt_Prasad_values = Dutt_Prasad.values

VN3_data = pd.read_csv(os.path.join(folder, 'Viswanath Natarajan Dynamic 3 term.tsv'),
                       sep='\t', index_col=0)
_VN3_data_values = VN3_data.values

VN2_data = pd.read_csv(os.path.join(folder, 'Viswanath Natarajan Dynamic 2 term.tsv'),
                       sep='\t', index_col=0)
_VN2_data_values = VN2_data.values

VN2E_data = pd.read_csv(os.path.join(folder, 'Viswanath Natarajan Dynamic 2 term Exponential.tsv'),
                        sep='\t', index_col=0)
_VN2E_data_values = VN2E_data.values

Perrys2_313 = pd.read_csv(os.path.join(folder, 'Table 2-313 Viscosity of Inorganic and Organic Liquids.tsv'),
                          sep='\t', index_col=0)
_Perrys2_313_values = Perrys2_313.values

Perrys2_312 = pd.read_csv(os.path.join(folder, 'Table 2-312 Vapor Viscosity of Inorganic and Organic Substances.tsv'),
                          sep='\t', index_col=0)
_Perrys2_312_values = Perrys2_312.values

VDI_PPDS_7 = pd.read_csv(os.path.join(folder, 'VDI PPDS Dynamic viscosity of saturated liquids polynomials.tsv'),
                          sep='\t', index_col=0)
_VDI_PPDS_7_values = VDI_PPDS_7.values

VDI_PPDS_8 = pd.read_csv(os.path.join(folder, 'VDI PPDS Dynamic viscosity of gases polynomials.tsv'),
                          sep='\t', index_col=0)
_VDI_PPDS_8_values = VDI_PPDS_8.values


def ViswanathNatarajan2(T, A, B):
    '''
    This function is known to produce values 10 times too low.
    The author's data must have an error.
    I have adjusted it to fix this.

    # DDBST has 0.0004580 as a value at this temperature
    >>> ViswanathNatarajan2(348.15, -5.9719, 1007.0)
    0.00045983686956829517
    '''
    mu = exp(A + B/T)
    mu = mu/1000.
    mu = mu*10
    return mu

#print(ViswanathNatarajan2(298.15, -5.1466, 625.44))


def ViswanathNatarajan2Exponential(T, C, D):
    '''
    This function is genuinely bad at what it does.

    >>> ViswanathNatarajan2Exponential(298.15, 4900800,  -3.8075)
    0.0018571903840928496
    '''
    mu = C*T**D
    return mu


def ViswanathNatarajan3(T, A, B, C):
    r'''Calculate the viscosity of a liquid using the 3-term Antoine form
    representation developed in [1]_. Requires input coefficients. The `A`
    coefficient is assumed to yield coefficients in centipoise, as all 
    coefficients found so far have been.

    .. math::
        \log_{10} \mu = A + B/(T + C)

    Parameters
    ----------
    T : float
        Temperature of fluid [K]

    Returns
    -------
    mu : float
        Liquid viscosity, [Pa*s]

    Notes
    -----
    No other source for these coefficients has been found.

    Examples
    --------
    >>> ViswanathNatarajan3(298.15, -2.7173, -1071.18, -129.51)
    0.0006129806445142112

    References
    ----------
    .. [1] Viswanath, Dabir S., and G. Natarajan. Databook On The Viscosity Of
       Liquids. New York: Taylor & Francis, 1989
    '''
    mu = 10**(A + B/(C - T))
    return mu/1000.


def Letsou_Stiel(T, MW, Tc, Pc, omega):
    r'''Calculates the viscosity of a liquid using an emperical model
    developed in [1]_. However. the fitting parameters for tabulated values
    in the original article are found in ChemSep.

    .. math::
        \xi = \frac{2173.424 T_c^{1/6}}{\sqrt{MW} P_c^{2/3}}

        \xi^{(0)} = (1.5174 - 2.135T_r + 0.75T_r^2)\cdot 10^{-5}

        \xi^{(1)} = (4.2552 - 7.674 T_r + 3.4 T_r^2)\cdot 10^{-5}

        \mu = (\xi^{(0)} + \omega \xi^{(1)})/\xi

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    MW : float
        Molwcular weight of fluid [g/mol]
    Tc : float
        Critical temperature of the fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    omega : float
        Acentric factor of compound

    Returns
    -------
    mu_l : float
        Viscosity of liquid, [Pa*S]

    Notes
    -----
    The form of this equation is a polynomial fit to tabulated data.
    The fitting was performed by the DIPPR. This is DIPPR Procedure 8G: Method
    for the viscosity of pure, nonhydrocarbon liquids at high temperatures
    internal units are SI standard. [1]_'s units were different.
    DIPPR test value for ethanol is used.

    Average error 34%. Range of applicability is 0.76 < Tr < 0.98.

    Examples
    --------
    >>> Letsou_Stiel(400., 46.07, 516.25, 6.383E6, 0.6371)
    0.0002036150875308151

    References
    ----------
    .. [1] Letsou, Athena, and Leonard I. Stiel. "Viscosity of Saturated
       Nonpolar Liquids at Elevated Pressures." AIChE Journal 19, no. 2 (1973):
       409-11. doi:10.1002/aic.690190241.
    '''
    Tr = T/Tc
    xi0 = (1.5174-2.135*Tr + 0.75*Tr**2)*1E-5
    xi1 = (4.2552-7.674*Tr + 3.4*Tr**2)*1E-5
    xi = 2173.424*Tc**(1/6.)/(MW**0.5*Pc**(2/3.))
    return (xi0 + omega*xi1)/xi


def Przedziecki_Sridhar(T, Tm, Tc, Pc, Vc, Vm, omega, MW):
    r'''Calculates the viscosity of a liquid using an emperical formula
    developed in [1]_.

    .. math::
        \mu=\frac{V_o}{E(V-V_o)}

        E=-1.12+\frac{V_c}{12.94+0.10MW-0.23P_c+0.0424T_{m}-11.58(T_{m}/T_c)}

        V_o = 0.0085\omega T_c-2.02+\frac{V_{m}}{0.342(T_m/T_c)+0.894}

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    Tm : float
        Melting point of fluid [K]
    Tc : float
        Critical temperature of the fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    Vc : float
        Critical volume of the fluid [m^3/mol]
    Vm : float
        Molar volume of the fluid at temperature [K]
    omega : float
        Acentric factor of compound
    MW : float
        Molwcular weight of fluid [g/mol]

    Returns
    -------
    mu_l : float
        Viscosity of liquid, [Pa*S]

    Notes
    -----
    A test by Reid (1983) is used, but only mostly correct.
    This function is not recommended. Its use has been removed from the Liquid Viscosity function.
    Internal units are bar and mL/mol.
    TODO: Test again with data from 5th ed table.

    Examples
    --------
    >>> Przedziecki_Sridhar(383., 178., 591.8, 41E5, 316E-6, 95E-6, .263, 92.14)
    0.0002198147995603383

    References
    ----------
    .. [1] Przedziecki, J. W., and T. Sridhar. "Prediction of Liquid
       Viscosities." AIChE Journal 31, no. 2 (February 1, 1985): 333-35.
       doi:10.1002/aic.690310225.
    '''
    Pc = Pc/1E5  # Pa to atm
    Vm, Vc = Vm*1E6, Vc*1E6  # m^3/mol to mL/mol
    Tr = T/Tc
    Gamma = 0.29607 - 0.09045*Tr - 0.04842*Tr**2
    VrT = 0.33593-0.33953*Tr + 1.51941*Tr**2 - 2.02512*Tr**3 + 1.11422*Tr**4
    V = VrT*(1-omega*Gamma)*Vc

    Vo = 0.0085*omega*Tc - 2.02 + Vm/(0.342*(Tm/Tc) + 0.894)  # checked
    E = -1.12 + Vc/(12.94 + 0.1*MW - 0.23*Pc + 0.0424*Tm - 11.58*(Tm/Tc))
    return Vo/(E*(V-Vo))/1000.


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

    Notes
    -----
    To iterate over all methods, use the lists stored in
    :obj:`viscosity_liquid_methods` and :obj:`viscosity_liquid_methods_P` for 
    low and high pressure methods respectively.

    Low pressure methods:

    **DUTT_PRASAD**:
        A simple function as expressed in [1]_, with data available for
        100 fluids. Temperature limits are available for all fluids. See
        :obj:`ViswanathNatarajan3` for details.
    **VISWANATH_NATARAJAN_3**:
        A simple function as expressed in [1]_, with data available for
        432 fluids. Temperature limits are available for all fluids. See
        :obj:`ViswanathNatarajan3` for details.
    **VN2_data**:
        A simple function as expressed in [1]_, with data available for
        135 fluids. Temperature limits are available for all fluids. See
        :obj:`ViswanathNatarajan2` for details.
    **VISWANATH_NATARAJAN_2E**:
        A simple function as expressed in [1]_, with data available for
        14 fluids. Temperature limits are available for all fluids. See
        :obj:`ViswanathNatarajan2Exponential` for details.
    **DIPPR_PERRY_8E**:
        A collection of 337 coefficient sets from the DIPPR database published
        openly in [4]_. Provides temperature limits for all its fluids. 
        :obj:`thermo.dippr.EQ101` is used for its fluids.
    **LETSOU_STIEL**:
        CSP method, described in :obj:`Letsou_Stiel`.
    **PRZEDZIECKI_SRIDHAR**:
        CSP method, described in :obj:`Przedziecki_Sridhar`.
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
        CSP method, described in :obj:`Lucas`. Calculates a
        low-pressure liquid viscosity first, using `T_dependent_property`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [5]_. Very slow, but unparalled in accuracy for pressure
        dependence.

    See Also
    --------
    ViswanathNatarajan3
    ViswanathNatarajan2
    ViswanathNatarajan2Exponential
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
    property_min = 0
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
                 omega=None, Psat=None, Vml=None):
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

        self.load_all_methods()

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
        methods, methods_P = [], []
        Tmins, Tmaxs = [], []
        if has_CoolProp and self.CASRN in coolprop_dict:
            methods.append(COOLPROP); methods_P.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
            Tmins.append(self.CP_f.Tmin); Tmaxs.append(self.CP_f.Tc)
        if self.CASRN in _VDISaturationDict:
            methods.append(VDI_TABULAR)
            Ts, props = VDI_tabular_data(self.CASRN, 'Mu (l)')
            self.VDI_Tmin = Ts[0]
            self.VDI_Tmax = Ts[-1]
            self.tabular_data[VDI_TABULAR] = (Ts, props)
            Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
        if self.CASRN in Dutt_Prasad.index:
            methods.append(DUTT_PRASAD)
            _, A, B, C, self.DUTT_PRASAD_Tmin, self.DUTT_PRASAD_Tmax = _Dutt_Prasad_values[Dutt_Prasad.index.get_loc(self.CASRN)].tolist()
            self.DUTT_PRASAD_coeffs = [A, B, C]
            Tmins.append(self.DUTT_PRASAD_Tmin); Tmaxs.append(self.DUTT_PRASAD_Tmax)
        if self.CASRN in VN3_data.index:
            methods.append(VISWANATH_NATARAJAN_3)
            _, _, A, B, C, self.VISWANATH_NATARAJAN_3_Tmin, self.VISWANATH_NATARAJAN_3_Tmax = _VN3_data_values[VN3_data.index.get_loc(self.CASRN)].tolist()
            self.VISWANATH_NATARAJAN_3_coeffs = [A, B, C]
            Tmins.append(self.VISWANATH_NATARAJAN_3_Tmin); Tmaxs.append(self.VISWANATH_NATARAJAN_3_Tmax)
        if self.CASRN in VN2_data.index:
            methods.append(VISWANATH_NATARAJAN_2)
            _, _, A, B, self.VISWANATH_NATARAJAN_2_Tmin, self.VISWANATH_NATARAJAN_2_Tmax = _VN2_data_values[VN2_data.index.get_loc(self.CASRN)].tolist()
            self.VISWANATH_NATARAJAN_2_coeffs = [A, B]
            Tmins.append(self.VISWANATH_NATARAJAN_2_Tmin); Tmaxs.append(self.VISWANATH_NATARAJAN_2_Tmax)
        if self.CASRN in VN2E_data.index:
            methods.append(VISWANATH_NATARAJAN_2E)
            _, _, C, D, self.VISWANATH_NATARAJAN_2E_Tmin, self.VISWANATH_NATARAJAN_2E_Tmax = _VN2E_data_values[VN2E_data.index.get_loc(self.CASRN)].tolist()
            self.VISWANATH_NATARAJAN_2E_coeffs = [C, D]
            Tmins.append(self.VISWANATH_NATARAJAN_2E_Tmin); Tmaxs.append(self.VISWANATH_NATARAJAN_2E_Tmax)
        if self.CASRN in Perrys2_313.index:
            methods.append(DIPPR_PERRY_8E)
            _, C1, C2, C3, C4, C5, self.Perrys2_313_Tmin, self.Perrys2_313_Tmax = _Perrys2_313_values[Perrys2_313.index.get_loc(self.CASRN)].tolist()
            self.Perrys2_313_coeffs = [C1, C2, C3, C4, C5]
            Tmins.append(self.Perrys2_313_Tmin); Tmaxs.append(self.Perrys2_313_Tmax)
        if self.CASRN in VDI_PPDS_7.index:
            methods.append(VDI_PPDS)
            self.VDI_PPDS_coeffs = _VDI_PPDS_7_values[VDI_PPDS_7.index.get_loc(self.CASRN)].tolist()[2:]
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
            Viscosity of the liquid at T and a low pressure, [Pa*S]
        '''
        if method == DUTT_PRASAD:
            A, B, C = self.DUTT_PRASAD_coeffs
            mu = ViswanathNatarajan3(T, A, B, C, )
        elif method == VISWANATH_NATARAJAN_3:
            A, B, C = self.VISWANATH_NATARAJAN_3_coeffs
            mu = ViswanathNatarajan3(T, A, B, C)
        elif method == VISWANATH_NATARAJAN_2:
            A, B = self.VISWANATH_NATARAJAN_2_coeffs
            mu = ViswanathNatarajan2(T, self.VISWANATH_NATARAJAN_2_coeffs[0], self.VISWANATH_NATARAJAN_2_coeffs[1])
        elif method == VISWANATH_NATARAJAN_2E:
            C, D = self.VISWANATH_NATARAJAN_2E_coeffs
            mu = ViswanathNatarajan2Exponential(T, C, D)
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
            A, B, C, D, E = self.VDI_PPDS_coeffs
            term = (C - T)/(T-D)
            if term < 0:
                term1 = -((T - C)/(T-D))**(1/3.)
            else:
                term1 = term**(1/3.)
            term2 = term*term1
            mu = E*exp(A*term1 + B*term2)
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
            Viscosity of the liquid at T and P, [Pa*S]
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


### Viscosity of Dense Liquids


def Lucas(T, P, Tc, Pc, omega, P_sat, mu_l):
    r'''Adjustes for pressure the viscosity of a liquid using an emperical
    formula developed in [1]_, but as discussed in [2]_ as the original source
    is in German.

    .. math::
        \frac{\mu}{\mu_{sat}}=\frac{1+D(\Delta P_r/2.118)^A}{1+C\omega \Delta P_r}

        \Delta P_r = \frac{P-P^{sat}}{P_c}

        A=0.9991-\frac{4.674\times 10^{-4}}{1.0523T_r^{-0.03877}-1.0513}

        D = \frac{0.3257}{(1.0039-T_r^{2.573})^{0.2906}}-0.2086

        C = -0.07921+2.1616T_r-13.4040T_r^2+44.1706T_r^3-84.8291T_r^4+
        96.1209T_r^5-59.8127T_r^6+15.6719T_r^7

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    P : float
        Pressure of fluid [Pa]
    Tc: float
        Critical point of fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    omega : float
        Acentric factor of compound
    P_sat : float
        Saturation pressure of the fluid [Pa]
    mu_l : float
        Viscosity of liquid at 1 atm or saturation, [Pa*S]

    Returns
    -------
    mu_l_dense : float
        Viscosity of liquid, [Pa*s]

    Notes
    -----
    This equation is entirely dimensionless; all dimensions cancel.
    The example is from Reid (1987); all results agree.
    Above several thousand bar, this equation does not represent true behavior.
    If Psat is larger than P, the fluid may not be liquid; dPr is set to 0.

    Examples
    --------
    >>> Lucas(300., 500E5, 572.2, 34.7E5, 0.236, 0, 0.00068) # methylcyclohexane
    0.0010683738499316518

    References
    ----------
    .. [1] Lucas, Klaus. "Ein Einfaches Verfahren Zur Berechnung Der
       Viskositat von Gasen Und Gasgemischen." Chemie Ingenieur Technik 46, no. 4
       (February 1, 1974): 157-157. doi:10.1002/cite.330460413.
    .. [2] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    Tr = T/Tc
    C = -0.07921+2.1616*Tr - 13.4040*Tr**2 + 44.1706*Tr**3 - 84.8291*Tr**4 \
        + 96.1209*Tr**5-59.8127*Tr**6+15.6719*Tr**7
    D = 0.3257/((1.0039-Tr**2.573)**0.2906) - 0.2086
    A = 0.9991 - 4.674E-4/(1.0523*Tr**-0.03877 - 1.0513)
    dPr = (P-P_sat)/Pc
    if dPr < 0:
        dPr = 0
    return (1. + D*(dPr/2.118)**A)/(1. + C*omega*dPr)*mu_l

### Viscosity of liquid mixtures


LALIBERTE_MU = 'Laliberte'
MIXING_LOG_MOLAR = 'Logarithmic mixing, molar'
MIXING_LOG_MASS = 'Logarithmic mixing, mass'

viscosity_liquid_mixture_methods = [LALIBERTE_MU, MIXING_LOG_MOLAR, MIXING_LOG_MASS]


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
                            
    ranked_methods = [LALIBERTE_MU, MIXING_LOG_MOLAR, MIXING_LOG_MASS]

    def __init__(self, CASs=[], ViscosityLiquids=[]):
        self.CASs = CASs
        self.ViscosityLiquids = ViscosityLiquids

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
        methods = [MIXING_LOG_MOLAR, MIXING_LOG_MASS]
        if len(self.CASs) > 1 and '7732-18-5' in self.CASs:
            wCASs = [i for i in self.CASs if i != '7732-18-5'] 
            if all([i in _Laliberte_Viscosity_ParametersDict for i in wCASs]):
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
        if method == MIXING_LOG_MOLAR:
            mus = [i(T, P) for i in self.ViscosityLiquids]
            return mixing_logarithmic(zs, mus)
        elif method == MIXING_LOG_MASS:
            mus = [i(T, P) for i in self.ViscosityLiquids]
            return mixing_logarithmic(ws, mus)
        elif method == LALIBERTE_MU:
            ws = list(ws) ; ws.pop(self.index_w)
            return Laliberte_viscosity(T, ws, self.wCASs)
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


### Viscosity of Gases - low pressure

def Yoon_Thodos(T, Tc, Pc, MW):
    r'''Calculates the viscosity of a gas using an emperical formula
    developed in [1]_.

    .. math::
        \eta \xi \times 10^8 = 46.10 T_r^{0.618} - 20.40 \exp(-0.449T_r) + 1
        9.40\exp(-4.058T_r)+1

        \xi = 2173.424 T_c^{1/6} MW^{-1/2} P_c^{-2/3}

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    Tc : float
        Critical temperature of the fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    MW : float
        Molwcular weight of fluid [g/mol]

    Returns
    -------
    mu_g : float
        Viscosity of gas, [Pa*S]

    Notes
    -----
    This equation has been tested. The equation uses SI units only internally.
    The constant 2173.424 is an adjustment factor for units.
    Average deviation within 3% for most compounds.
    Greatest accuracy with dipole moments close to 0.
    Hydrogen and helium have different coefficients, not implemented.
    This is DIPPR Procedure 8B: Method for the Viscosity of Pure,
    non hydrocarbon, nonpolar gases at low pressures

    Examples
    --------
    >>> Yoon_Thodos(300., 556.35, 4.5596E6, 153.8)
    1.0194885727776819e-05

    References
    ----------
    .. [1]  Yoon, Poong, and George Thodos. "Viscosity of Nonpolar Gaseous
       Mixtures at Normal Pressures." AIChE Journal 16, no. 2 (1970): 300-304.
       doi:10.1002/aic.690160225.
    '''
    Tr = T/Tc
    xi = 2173.4241*Tc**(1/6.)/(MW**0.5*Pc**(2/3.))
    a = 46.1
    b = 0.618
    c = 20.4
    d = -0.449
    e = 19.4
    f = -4.058
    return (1. + a*Tr**b - c * exp(d*Tr) + e*exp(f*Tr))/(1E8*xi)


def Stiel_Thodos(T, Tc, Pc, MW):
    r'''Calculates the viscosity of a gas using an emperical formula
    developed in [1]_.

    .. math::
        TODO

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    Tc : float
        Critical temperature of the fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    MW : float
        Molwcular weight of fluid [g/mol]

    Returns
    -------
    mu_g : float
        Viscosity of gas, [Pa*S]

    Notes
    -----
    Untested.
    Claimed applicability from 0.2 to 5 atm.
    Developed with data from 52 nonpolar, and 53 polar gases.
    internal units are poise and atm.
    Seems to give reasonable results.

    Examples
    --------
    >>> Stiel_Thodos(300., 556.35, 4.5596E6, 153.8) #CCl4
    1.0408926223608723e-05

    References
    ----------
    .. [1] Stiel, Leonard I., and George Thodos. "The Viscosity of Nonpolar
       Gases at Normal Pressures." AIChE Journal 7, no. 4 (1961): 611-15.
       doi:10.1002/aic.690070416.
    '''
    Pc = Pc/101325.
    Tr = T/Tc
    xi = Tc**(1/6.)/(MW**0.5*Pc**(2/3.))
    if Tr > 1.5:
        mu_g = 17.78E-5*(4.58*Tr-1.67)**.625/xi
    else:
        mu_g = 34E-5*Tr**0.94/xi
    return mu_g/1000.

_lucas_Q_dict = {'7440-59-7': 1.38, '1333-74-0': 0.76, '7782-39-0': 0.52}


def lucas_gas(T, Tc, Pc, Zc, MW, dipole=0, CASRN=None):
    r'''Estimate the viscosity of a gas using an emperical
    formula developed in several sources, but as discussed in [1]_ as the
    original sources are in German or merely personal communications with the
    authors of [1]_.

    .. math::
        \eta  = \left[0.807T_r^{0.618}-0.357\exp(-0.449T_r) + 0.340\exp(-4.058
        T_r) + 0.018\right]F_p^\circ F_Q^\circ /\xi

        F_p^\circ=1, 0 \le \mu_{r} < 0.022

        F_p^\circ = 1+30.55(0.292-Z_c)^{1.72}, 0.022 \le \mu_{r} < 0.075

        F_p^\circ = 1+30.55(0.292-Z_c)^{1.72}|0.96+0.1(T_r-0.7)| 0.075 < \mu_{r}

        F_Q^\circ = 1.22Q^{0.15}\left\{ 1+0.00385[(T_r-12)^2]^{1/M}\text{sign}
        (T_r-12)\right\}

        \mu_r = 52.46 \frac{\mu^2 P_c}{T_c^2}

        \xi=0.176\left(\frac{T_c}{MW^3 P_c^4}\right)^{1/6}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc: float
        Critical point of fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    Zc : float
        Critical compressibility of the fluid [Pa]
    dipole : float
        Dipole moment of fluid [debye]
    CASRN : str, optional
        CAS of the fluid

    Returns
    -------
    mu_g : float
        Viscosity of gas, [Pa*s]

    Notes
    -----
    The example is from [1]_; all results agree.
    Viscosity is calculated in micropoise, and converted to SI internally (1E-7).
    Q for He = 1.38; Q for H2 = 0.76; Q for D2 = 0.52.

    Examples
    --------
    >>> lucas_gas(T=550., Tc=512.6, Pc=80.9E5, Zc=0.224, MW=32.042, dipole=1.7)
    1.7822676912698928e-05

    References
    ----------
    .. [1] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    Tr = T/Tc
    xi = 0.176*(Tc/MW**3/(Pc/1E5)**4)**(1/6.)  # bar arrording to example in Poling
    if dipole is None:
        dipole = 0
    dipoler = 52.46*dipole**2*(Pc/1E5)/Tc**2  # bar arrording to example in Poling
    if dipoler < 0.022:
        Fp = 1
    elif 0.022 <= dipoler < 0.075:
        Fp = 1 + 30.55*(0.292 - Zc)**1.72
    else:
        Fp = 1 + 30.55*(0.292 - Zc)**1.72*abs(0.96 + 0.1*(Tr-0.7))
    if CASRN and CASRN in _lucas_Q_dict:
        Q = _lucas_Q_dict[CASRN]
        if Tr - 12 > 0:
            value = 1
        else:
            value = -1
        FQ = 1.22*Q**0.15*(1 + 0.00385*((Tr-12)**2)**(1./MW)*value)
    else:
        FQ = 1
    eta = (0.807*Tr**0.618 - 0.357*exp(-0.449*Tr) + 0.340*exp(-4.058*Tr) + 0.018)*Fp*FQ/xi
    return eta/1E7


def Gharagheizi_gas_viscosity(T, Tc, Pc, MW):
    r'''Calculates the viscosity of a gas using an emperical formula
    developed in [1]_.

    .. math::
        \mu = 10^{-7} | 10^{-5} P_cT_r + \left(0.091-\frac{0.477}{M}\right)T +
        M \left(10^{-5}P_c-\frac{8M^2}{T^2}\right)
        \left(\frac{10.7639}{T_c}-\frac{4.1929}{T}\right)|

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    Tc : float
        Critical temperature of the fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    MW : float
        Molwcular weight of fluid [g/mol]

    Returns
    -------
    mu_g : float
        Viscosity of gas, [Pa*S]

    Notes
    -----
    Example is first point in supporting information of article, for methane.
    This is the prefered function for gas viscosity.
    7% average relative deviation. Deviation should never be above 30%.
    Developed with the DIPPR database. It is believed theoretically predicted values
    are included in the correlation.

    Examples
    --------
    >>> Gharagheizi_gas_viscosity(120., 190.564, 45.99E5, 16.04246)
    5.215761625399613e-06

    References
    ----------
    .. [1] Gharagheizi, Farhad, Ali Eslamimanesh, Mehdi Sattari, Amir H.
       Mohammadi, and Dominique Richon. "Corresponding States Method for
       Determination of the Viscosity of Gases at Atmospheric Pressure."
       Industrial & Engineering Chemistry Research 51, no. 7
       (February 22, 2012): 3179-85. doi:10.1021/ie202591f.
    '''
    Tr = T/Tc
    mu_g = 1E-5*Pc*Tr + (0.091 - 0.477/MW)*T + MW*(1E-5*Pc - 8*MW**2/T**2)*(10.7639/Tc - 4.1929/T)
    return 1E-7 * abs(mu_g)


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
        CSP method, described in :obj:`Gharagheizi_gas_viscosity`.
    **YOON_THODOS**:
        CSP method, described in :obj:`Yoon_Thodos`.
    **STIEL_THODOS**:
        CSP method, described in :obj:`Stiel_Thodos`.
    **LUCAS_GAS**:
        CSP method, described in :obj:`lucas_gas`.
    **DIPPR_PERRY_8E**:
        A collection of 345 coefficient sets from the DIPPR database published
        openly in [3]_. Provides temperature limits for all its fluids. 
        :obj:`thermo.dippr.EQ102` is used for its fluids.
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
    lucas_gas

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
                 dipole=None, Vmg=None):
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

        self.load_all_methods()

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
        methods, methods_P = [], []
        Tmins, Tmaxs = [], []
        if self.CASRN in _VDISaturationDict:
            methods.append(VDI_TABULAR)
            Ts, props = VDI_tabular_data(self.CASRN, 'Mu (g)')
            self.VDI_Tmin = Ts[0]
            self.VDI_Tmax = Ts[-1]
            self.tabular_data[VDI_TABULAR] = (Ts, props)
            Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
        if has_CoolProp and self.CASRN in coolprop_dict:
            methods.append(COOLPROP); methods_P.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
            Tmins.append(self.CP_f.Tmin); Tmaxs.append(self.CP_f.Tmax)
        if self.CASRN in Perrys2_312.index:
            methods.append(DIPPR_PERRY_8E)
            _, C1, C2, C3, C4, self.Perrys2_312_Tmin, self.Perrys2_312_Tmax = _Perrys2_312_values[Perrys2_312.index.get_loc(self.CASRN)].tolist()
            self.Perrys2_312_coeffs = [C1, C2, C3, C4]
            Tmins.append(self.Perrys2_312_Tmin); Tmaxs.append(self.Perrys2_312_Tmax)
        if self.CASRN in VDI_PPDS_8.index:
            methods.append(VDI_PPDS)
            self.VDI_PPDS_coeffs = _VDI_PPDS_8_values[VDI_PPDS_8.index.get_loc(self.CASRN)].tolist()[1:]
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
            Viscosity of the gas at T and a low pressure, [Pa*S]
        '''
        if method == GHARAGHEIZI:
            mu = Gharagheizi_gas_viscosity(T, self.Tc, self.Pc, self.MW)
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
            mu = lucas_gas(T, self.Tc, self.Pc, self.Zc, self.MW, self.dipole, CASRN=self.CASRN)
        elif method in self.tabular_data:
            mu = self.interpolate(T, method)
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


### Viscosity of gas mixtures

def Herning_Zipperer(zs, mus, MWs):
    r'''Calculates viscosity of a gas mixture according to
    mixing rules in [1]_.

    .. math::
        TODO

    Parameters
    ----------
    zs : float
        Mole fractions of components
    mus : float
        Gas viscosities of all components, [Pa*S]
    MWs : float
        Molecular weights of all components, [g/mol]

    Returns
    -------
    mug : float
        Viscosity of gas mixture, Pa*S]

    Notes
    -----
    This equation is entirely dimensionless; all dimensions cancel.
    The original source has not been reviewed.

    Examples
    --------

    References
    ----------
    .. [1] Herning, F. and Zipperer, L,: "Calculation of the Viscosity of
       Technical Gas Mixtures from the Viscosity of Individual Gases, german",
       Gas u. Wasserfach (1936) 79, No. 49, 69.
    '''
    if not none_and_length_check([zs, mus, MWs]):  # check same-length inputs
        raise Exception('Function inputs are incorrect format')
    MW_roots = [MWi**0.5 for MWi in MWs]
    denominator = sum([zi*MW_root_i for zi, MW_root_i in zip(zs, MW_roots)])
    k = sum([zi*mui*MW_root_i for zi, mui, MW_root_i in zip(zs, mus, MW_roots)])
    return k/denominator


def Wilke(ys, mus, MWs):
    r'''Calculates viscosity of a gas mixture according to
    mixing rules in [1]_.

    .. math::
        \eta_{mix} = \sum_{i=1}^n \frac{y_i \eta_i}{\sum_{j=1}^n y_j \phi_{ij}}

        \phi_{ij} = \frac{(1 + \sqrt{\eta_i/\eta_j}(MW_j/MW_i)^{0.25})^2}
        {\sqrt{8(1+MW_i/MW_j)}}

    Parameters
    ----------
    ys : float
        Mole fractions of gas components
    mus : float
        Gas viscosities of all components, [Pa*S]
    MWs : float
        Molecular weights of all components, [g/mol]

    Returns
    -------
    mug : float
        Viscosity of gas mixture, Pa*S]

    Notes
    -----
    This equation is entirely dimensionless; all dimensions cancel.
    The original source has not been reviewed or found.

    Examples
    --------
    >>> Wilke([0.05, 0.95], [1.34E-5, 9.5029E-6], [64.06, 46.07])
    9.701614885866193e-06

    References
    ----------
    .. [1] TODO
    '''
    if not none_and_length_check([ys, mus, MWs]):  # check same-length inputs
        raise Exception('Function inputs are incorrect format')
    cmps = range(len(ys))
    phis = [[(1 + (mus[i]/mus[j])**0.5*(MWs[j]/MWs[i])**0.25)**2/(8*(1 + MWs[i]/MWs[j]))**0.5
                    for j in cmps] for i in cmps]

    return sum([ys[i]*mus[i]/sum([ys[j]*phis[i][j] for j in cmps]) for i in cmps])


def Brokaw(T, ys, mus, MWs, molecular_diameters, Stockmayers):
    r'''Calculates viscosity of a gas mixture according to
    mixing rules in [1]_.

    .. math::
        \eta_{mix} = \sum_{i=1}^n \frac{y_i \eta_i}{\sum_{j=1}^n y_j \phi_{ij}}

        \phi_{ij} = \left( \frac{\eta_i}{\eta_j} \right)^{0.5} S_{ij} A_{ij}

        A_{ij} = m_{ij} M_{ij}^{-0.5} \left[1 +
        \frac{M_{ij} - M_{ij}^{0.45}}
        {2(1+M_{ij}) + \frac{(1 + M_{ij}^{0.45}) m_{ij}^{-0.5}}{1 + m_{ij}}} \right]

        m_{ij} = \left[ \frac{4}{(1+M_{ij}^{-1})(1+M_{ij})}\right]^{0.25}

        M_{ij} = \frac{M_i}{M_j}

        S_{ij} = \frac{1 + (T_i^* T_j^*)^{0.5} + (\delta_i \delta_j/4)}
        {[1+T_i^* + (\delta_i^2/4)]^{0.5}[1+T_j^*+(\delta_j^2/4)]^{0.5}}

        T^* = kT/\epsilon

    Parameters
    ----------
    T : float
        Temperature of fluid, [K]
    ys : float
        Mole fractions of gas components
    mus : float
        Gas viscosities of all components, [Pa*S]
    MWs : float
        Molecular weights of all components, [g/mol]
    molecular_diameters : float
        L-J molecular diameter  of all components, [angstroms]
    Stockmayers : float
        L-J Stockmayer energy parameters of all components, []

    Returns
    -------
    mug : float
        Viscosity of gas mixture, [Pa*S]

    Notes
    -----
    This equation is entirely dimensionless; all dimensions cancel.
    The original source has not been reviewed.

    This is DIPPR Procedure 8D: Method for the Viscosity of Nonhydrocarbon
    Vapor Mixtures at Low Pressure (Polar and Nonpolar)

    Examples
    --------
    >>> Brokaw(308.2, [0.05, 0.95], [1.34E-5, 9.5029E-6], [64.06, 46.07], [0.42, 0.19], [347, 432])
    9.699085099801568e-06

    References
    ----------
    .. [1] Brokaw, R. S. "Predicting Transport Properties of Dilute Gases."
       Industrial & Engineering Chemistry Process Design and Development
       8, no. 2 (April 1, 1969): 240-53. doi:10.1021/i260030a015.
    .. [2] Brokaw, R. S. Viscosity of Gas Mixtures, NASA-TN-D-4496, 1968.
    .. [3] Danner, Ronald P, and Design Institute for Physical Property Data.
       Manual for Predicting Chemical Process Design Data. New York, N.Y, 1982.
    '''
    cmps = range(len(ys))
    MDs = molecular_diameters
    if not none_and_length_check([ys, mus, MWs, molecular_diameters, Stockmayers]): # check same-length inputs
        raise Exception('Function inputs are incorrect format')
    Tsts = [T/Stockmayer_i for Stockmayer_i in Stockmayers]
    Sij = [[0 for i in cmps] for j in cmps]
    Mij = [[0 for i in cmps] for j in cmps]
    mij = [[0 for i in cmps] for j in cmps]
    Aij = [[0 for i in cmps] for j in cmps]
    phiij =[[0 for i in cmps] for j in cmps]

    for i in cmps:
        for j in cmps:
            Sij[i][j] = (1+(Tsts[i]*Tsts[j])**0.5 + (MDs[i]*MDs[j])/4.)/(1 + Tsts[i] + (MDs[i]**2/4.))**0.5/(1 + Tsts[j] + (MDs[j]**2/4.))**0.5
            if MDs[i] <= 0.1 and MDs[j] <= 0.1:
                Sij[i][j] = 1
            Mij[i][j] = MWs[i]/MWs[j]
            mij[i][j] = (4./(1+Mij[i][j]**-1)/(1+Mij[i][j]))**0.25

            Aij[i][j] = mij[i][j]*Mij[i][j]**-0.5*(1 + (Mij[i][j]-Mij[i][j]**0.45)/(2*(1+Mij[i][j]) + (1+Mij[i][j]**0.45)*mij[i][j]**-0.5/(1+mij[i][j])))

            phiij[i][j] = (mus[i]/mus[j])**0.5*Sij[i][j]*Aij[i][j]

    return sum([ys[i]*mus[i]/sum([ys[j]*phiij[i][j] for j in cmps]) for i in cmps])


BROKAW = 'Brokaw'
HERNING_ZIPPERER = 'Herning-Zipperer'
WILKE = 'Wilke'
SIMPLE = 'Simple'
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
    ViscosityGass : list[ViscosityGas], optional
        ViscosityGas objects created for all species in the mixture,  
        normally created by :obj:`thermo.chemical.Chemical`.
    
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

    def __init__(self, MWs=[], molecular_diameters=[], Stockmayers=[], CASs=[], ViscosityGases=[]):
        self.MWs = MWs
        self.molecular_diameters = molecular_diameters
        self.Stockmayers = Stockmayers
        self.CASs = CASs
        self.ViscosityGases = ViscosityGases

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
        if method == SIMPLE:
            mus = [i(T, P) for i in self.ViscosityGases]
            return mixing_simple(zs, mus)
        elif method == HERNING_ZIPPERER:
            mus = [i(T, P) for i in self.ViscosityGases]
            return Herning_Zipperer(zs, mus, self.MWs)
        elif method == WILKE:
            mus = [i(T, P) for i in self.ViscosityGases]
            return Wilke(zs, mus, self.MWs)
        elif method == BROKAW:
            mus = [i(T, P) for i in self.ViscosityGases]
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


### Misc functions


def _round_whole_even(i):
    r'''Round a number to the nearest whole number. If the number is exactly
    between two numbers, round to the even whole number. Used by
    `viscosity_index`.

    Parameters
    ----------
    i : float
        Number, [-]

    Returns
    -------
    i : int
        Rounded number, [-]

    Notes
    -----
    Should never run with inputs from a practical function, as numbers on
    computers aren't really normally exactly between two numbers.

    Examples
    --------
    _round_whole_even(116.5)
    116
    '''
    if i % .5 == 0:
        if (i + 0.5) % 2 == 0:
            i = i + 0.5
        else:
            i = i - 0.5
    else:
        i = round(i, 0)
    return int(i)


VI_nus = np.array([2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1,
                   3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3,
                   4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 5.1, 5.2, 5.3, 5.4, 5.5,
                   5.6, 5.7, 5.8, 5.9, 6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
                   6.8, 6.9, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9,
                   8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9, 9.1,
                   9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10, 10.1, 10.2,
                   10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11, 11.1, 11.2,
                   11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12, 12.1, 12.2,
                   12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13, 13.1, 13.2,
                   13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14, 14.1, 14.2,
                   14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2,
                   15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16, 16.1, 16.2,
                   16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17, 17.1, 17.2,
                   17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 18, 18.1, 18.2,
                   18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 19, 19.1, 19.2,
                   19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20, 20.2, 20.4,
                   20.6, 20.8, 21, 21.2, 21.4, 21.6, 21.8, 22, 22.2, 22.4,
                   22.6, 22.8, 23, 23.2, 23.4, 23.6, 23.8, 24, 24.2, 24.4,
                   24.6, 24.8, 25, 25.2, 25.4, 25.6, 25.8, 26, 26.2, 26.4,
                   26.6, 26.8, 27, 27.2, 27.4, 27.6, 27.8, 28, 28.2, 28.4,
                   28.6, 28.8, 29, 29.2, 29.4, 29.6, 29.8, 30, 30.5, 31,
                   31.5, 32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36, 36.5,
                   37, 37.5, 38, 38.5, 39, 39.5, 40, 40.5, 41, 41.5, 42,
                   42.5, 43, 43.5, 44, 44.5, 45, 45.5, 46, 46.5, 47, 47.5,
                   48, 48.5, 49, 49.5, 50, 50.5, 51, 51.5, 52, 52.5, 53,
                   53.5, 54, 54.5, 55, 55.5, 56, 56.5, 57, 57.5, 58, 58.5,
                   59, 59.5, 60, 60.5, 61, 61.5, 62, 62.5, 63, 63.5, 64,
                   64.5, 65, 65.5, 66, 66.5, 67, 67.5, 68, 68.5, 69, 69.5, 70])
VI_Ls = np.array([7.994, 8.64, 9.309, 10, 10.71, 11.45, 12.21, 13, 13.8,
                  14.63, 15.49, 16.36, 17.26, 18.18, 19.12, 20.09, 21.08,
                  22.09, 23.13, 24.19, 25.32, 26.5, 27.75, 29.07, 30.48,
                  31.96, 33.52, 35.13, 36.79, 38.5, 40.23, 41.99, 43.76,
                  45.53, 47.31, 49.09, 50.87, 52.64, 54.42, 56.2, 57.97,
                  59.74, 61.52, 63.32, 65.18, 67.12, 69.16, 71.29, 73.48,
                  75.72, 78, 80.25, 82.39, 84.53, 86.66, 88.85, 91.04, 93.2,
                  95.43, 97.72, 100, 102.3, 104.6, 106.9, 109.2, 111.5, 113.9,
                  116.2, 118.5, 120.9, 123.3, 125.7, 128, 130.4, 132.8, 135.3,
                  137.7, 140.1, 142.7, 145.2, 147.7, 150.3, 152.9, 155.4, 158,
                  160.6, 163.2, 165.8, 168.5, 171.2, 173.9, 176.6, 179.4,
                  182.1, 184.9, 187.6, 190.4, 193.3, 196.2, 199, 201.9, 204.8,
                  207.8, 210.7, 213.6, 216.6, 219.6, 222.6, 225.7, 228.8,
                  231.9, 235, 238.1, 241.2, 244.3, 247.4, 250.6, 253.8, 257,
                  260.1, 263.3, 266.6, 269.8, 273, 276.3, 279.6, 283, 286.4,
                  289.7, 293, 296.5, 300, 303.4, 306.9, 310.3, 313.9, 317.5,
                  321.1, 324.6, 328.3, 331.9, 335.5, 339.2, 342.9, 346.6,
                  350.3, 354.1, 358, 361.7, 365.6, 369.4, 373.3, 377.1, 381,
                  384.9, 388.9, 392.7, 396.7, 400.7, 404.6, 408.6, 412.6,
                  416.7, 420.7, 424.9, 429, 433.2, 437.3, 441.5, 445.7,
                  449.9, 454.2, 458.4, 462.7, 467, 471.3, 475.7, 479.7,
                  483.9, 488.6, 493.2, 501.5, 510.8, 519.9, 528.8, 538.4,
                  547.5, 556.7, 566.4, 575.6, 585.2, 595, 604.3, 614.2,
                  624.1, 633.6, 643.4, 653.8, 663.3, 673.7, 683.9, 694.5,
                  704.2, 714.9, 725.7, 736.5, 747.2, 758.2, 769.3, 779.7,
                  790.4, 801.6, 812.8, 824.1, 835.5, 847, 857.5, 869, 880.6,
                  892.3, 904.1, 915.8, 927.6, 938.6, 951.2, 963.4, 975.4,
                  987.1, 998.9, 1011, 1023, 1055, 1086, 1119, 1151, 1184,
                  1217, 1251, 1286, 1321, 1356, 1391, 1427, 1464, 1501, 1538,
                  1575, 1613, 1651, 1691, 1730, 1770, 1810, 1851, 1892, 1935,
                  1978, 2021, 2064, 2108, 2152, 2197, 2243, 2288, 2333, 2380,
                  2426, 2473, 2521, 2570, 2618, 2667, 2717, 2767, 2817, 2867,
                  2918, 2969, 3020, 3073, 3126, 3180, 3233, 3286, 3340, 3396,
                  3452, 3507, 3563, 3619, 3676, 3734, 3792, 3850, 3908, 3966,
                  4026, 4087, 4147, 4207, 4268, 4329, 4392, 4455, 4517, 4580,
                  4645, 4709, 4773, 4839, 4905])
VI_Hs = np.array([6.394, 6.894, 7.41, 7.944, 8.496, 9.063, 9.647, 10.25,
                  10.87, 11.5, 12.15, 12.82, 13.51, 14.21, 14.93, 15.66,
                  16.42, 17.19, 17.97, 18.77, 19.56, 20.37, 21.21, 22.05,
                  22.92, 23.81, 24.71, 25.63, 26.57, 27.53, 28.49, 29.46,
                  30.43, 31.4, 32.37, 33.34, 34.32, 35.29, 36.26, 37.23,
                  38.19, 39.17, 40.15, 41.13, 42.14, 43.18, 44.24, 45.33,
                  46.44, 47.51, 48.57, 49.61, 50.69, 51.78, 52.88, 53.98,
                  55.09, 56.2, 57.31, 58.45, 59.6, 60.74, 61.89, 63.05,
                  64.18, 65.32, 66.48, 67.64, 68.79, 69.94, 71.1, 72.27,
                  73.42, 74.57, 75.73, 76.91, 78.08, 79.27, 80.46, 81.67,
                  82.87, 84.08, 85.3, 86.51, 87.72, 88.95, 90.19, 91.4,
                  92.65, 93.92, 95.19, 96.45, 97.71, 98.97, 100.2, 101.5,
                  102.8, 104.1, 105.4, 106.7, 108, 109.4, 110.7, 112, 113.3,
                  114.7, 116, 117.4, 118.7, 120.1, 121.5, 122.9, 124.2,
                  125.6, 127, 128.4, 129.8, 131.2, 132.6, 134, 135.4, 136.8,
                  138.2, 139.6, 141, 142.4, 143.9, 145.3, 146.8, 148.2,
                  149.7, 151.2, 152.6, 154.1, 155.6, 157, 158.6, 160.1,
                  161.6, 163.1, 164.6, 166.1, 167.7, 169.2, 170.7, 172.3,
                  173.8, 175.4, 177, 178.6, 180.2, 181.7, 183.3, 184.9,
                  186.5, 188.1, 189.7, 191.3, 192.9, 194.6, 196.2, 197.8,
                  199.4, 201, 202.6, 204.3, 205.9, 207.6, 209.3, 211, 212.7,
                  214.4, 216.1, 217.7, 219.4, 221.1, 222.8, 224.5, 226.2,
                  227.7, 229.5, 233, 236.4, 240.1, 243.5, 247.1, 250.7,
                  254.2, 257.8, 261.5, 264.9, 268.6, 272.3, 275.8, 279.6,
                  283.3, 286.8, 290.5, 294.4, 297.9, 301.8, 305.6, 309.4,
                  313, 317, 320.9, 324.9, 328.8, 332.7, 336.7, 340.5, 344.4,
                  348.4, 352.3, 356.4, 360.5, 364.6, 368.3, 372.3, 376.4,
                  380.6, 384.6, 388.8, 393, 396.6, 401.1, 405.3, 409.5,
                  413.5, 417.6, 421.7, 432.4, 443.2, 454, 464.9, 475.9, 487,
                  498.1, 509.6, 521.1, 532.5, 544, 555.6, 567.1, 579.3,
                  591.3, 603.1, 615, 627.1, 639.2, 651.8, 664.2, 676.6,
                  689.1, 701.9, 714.9, 728.2, 741.3, 754.4, 767.6, 780.9,
                  794.5, 808.2, 821.9, 835.5, 849.2, 863, 876.9, 890.9,
                  905.3, 919.6, 933.6, 948.2, 962.9, 977.5, 992.1, 1007,
                  1021, 1036, 1051, 1066, 1082, 1097, 1112, 1127, 1143,
                  1159, 1175, 1190, 1206, 1222, 1238, 1254, 1270, 1286,
                  1303, 1319, 1336, 1352, 1369, 1386, 1402, 1419, 1436,
                  1454, 1471, 1488, 1506, 1523, 1541, 1558])


def viscosity_index(nu_40, nu_100, rounding=False):
    r'''Calculates the viscosity index of a liquid. Requires dynamic viscosity
    of a liquid at 40°C and 100°C. Value may either be returned with or
    without rounding. Rounding is performed per the standard.

    if nu_100 < 70:

    .. math::
        L, H = interp(nu_100)

    else:

    .. math::
        L = 0.8353\nu_{100}^2 + 14.67\nu_{100} - 216

        H = 0.1684\nu_{100}^2 + 11.85\nu_{100} - 97

    if nu_40 > H:

    .. math::
        VI = \frac{L-nu_{40}}{L-H}\cdot 100

    else:

    .. math::
        N = \frac{\log(H) - \log(\nu_{40})}{\log (\nu_{100})}

         VI = \frac{10^N-1}{0.00715} + 100

    Parameters
    ----------
    nu_40 : float
        Dynamic viscosity of fluid at 40°C, [m^2/s]
    nu_100 : float
        Dynamic viscosity of fluid at 100°C, [m^2/s]
    rounding : bool, optional
        Whether to round the value or not.

    Returns
    -------
    VI: float
        Viscosity index [-]

    Notes
    -----
    VI is undefined for nu_100 under 2 mm^2/s. None is returned if this is the
    case. Internal units are mm^2/s. Higher values of viscosity index suggest
    a lesser decrease in kinematic viscosity as temperature increases.
    
    Note that viscosity is a pressure-dependent property, and that the 
    viscosity index is defined for a fluid at whatever pressure it is at.
    The viscosity index is thus also a function of pressure.

    Examples
    --------
    >>> viscosity_index(73.3E-6, 8.86E-6, rounding=True)
    92

    References
    ----------
    .. [1] ASTM D2270-10(2016) Standard Practice for Calculating Viscosity
       Index from Kinematic Viscosity at 40 °C and 100 °C, ASTM International,
       West Conshohocken, PA, 2016, http://dx.doi.org/10.1520/D2270-10R16
    '''
    nu_40, nu_100 = nu_40*1E6, nu_100*1E6  # m^2/s to mm^2/s
    if nu_100 < 2:
        return None  # Not defined for under this
    elif nu_100 < 70:
        L = np.interp(nu_100, VI_nus, VI_Ls)
        H = np.interp(nu_100, VI_nus, VI_Hs)
    else:
        L = 0.8353*nu_100**2 + 14.67*nu_100 - 216
        H = 0.1684*nu_100**2 + 11.85*nu_100 - 97
    if nu_40 > H:
        VI = (L-nu_40)/(L-H)*100
    else:
        N = (log(H) - log(nu_40))/log(nu_100)
        VI = (10**N-1)/0.00715 + 100
    if rounding:
        VI = _round_whole_even(VI)
    return VI

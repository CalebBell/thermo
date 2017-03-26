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

__all__ = ['WagnerMcGarry', 'AntoinePoling', 'WagnerPoling', 'AntoineExtended',
           'Antoine', 'Wagner_original', 'Wagner', 'TRC_Antoine_extended', 
           'vapor_pressure_methods', 'VaporPressure', 'Perrys2_8', 'VDI_PPDS_3',
           'boiling_critical_relation', 'Lee_Kesler', 'Ambrose_Walton', 
           'Edalat', 'Sanjari']

import os
import numpy as np
import pandas as pd
from thermo.utils import log, exp
from thermo.miscdata import _VDISaturationDict, VDI_tabular_data
from thermo.utils import TDependentProperty
from thermo.coolprop import has_CoolProp, PropsSI, coolprop_dict, coolprop_fluids
from thermo.dippr import EQ101

folder = os.path.join(os.path.dirname(__file__), 'Vapor Pressure')

WagnerMcGarry = pd.read_csv(os.path.join(folder, 'Wagner Original McGarry.tsv'),
                            sep='\t', index_col=0)
_WagnerMcGarry_values = WagnerMcGarry.values

AntoinePoling = pd.read_csv(os.path.join(folder, 'Antoine Collection Poling.tsv'),
                            sep='\t', index_col=0)
_AntoinePoling_values = AntoinePoling.values

WagnerPoling = pd.read_csv(os.path.join(folder, 'Wagner Collection Poling.tsv'),
                           sep='\t', index_col=0)
_WagnerPoling_values = WagnerPoling.values

AntoineExtended = pd.read_csv(os.path.join(folder, 'Antoine Extended Collection Poling.tsv'),
                              sep='\t', index_col=0)
_AntoineExtended_values = AntoineExtended.values

Perrys2_8 = pd.read_csv(os.path.join(folder, 'Table 2-8 Vapor Pressure of Inorganic and Organic Liquids.tsv'),
                          sep='\t', index_col=0)
_Perrys2_8_values = Perrys2_8.values

VDI_PPDS_3 = pd.read_csv(os.path.join(folder, 'VDI PPDS Boiling temperatures at different pressures.tsv'),
                          sep='\t', index_col=0)
_VDI_PPDS_3_values = VDI_PPDS_3.values


def Antoine(T, A, B, C, base=10.0):
    r'''Calculates vapor pressure of a chemical using the Antoine equation.
    Parameters `A`, `B`, and `C` are chemical-dependent. Parameters can be 
    found in numerous sources; however units of the coefficients used vary.
    Originally proposed by Antoine (1888) [2]_.

    .. math::
        \log_{\text{base}} P^{\text{sat}} = A - \frac{B}{T+C}

    Parameters
    ----------
    T : float
        Temperature of fluid, [K]
    A, B, C : floats
        Regressed coefficients for Antoine equation for a chemical

    Returns
    -------
    Psat : float
        Vapor pressure calculated with coefficients [Pa]
    
    Other Parameters
    ----------------
    Base : float
        Optional base of logarithm; 10 by default

    Notes
    -----
    Assumes coefficients are for calculating vapor pressure in Pascal. 
    Coefficients should be consistent with input temperatures in Kelvin;
    however, if both the given temperature and units are specific to degrees
    Celcius, the result will still be correct.
    
    **Converting units in input coefficients:**
    
        * **ln to log10**: Divide A and B by ln(10)=2.302585 to change  
          parameters for a ln equation to a log10 equation.
        * **log10 to ln**: Multiply A and B by ln(10)=2.302585 to change 
          parameters for a log equation to a ln equation.
        * **mmHg to Pa**: Add log10(101325/760)= 2.1249 to A.
        * **kPa to Pa**: Add log_{base}(1000)= 6.908 to A for log(base)
        * **°C to K**: Subtract 273.15 from C only!

    Examples
    --------
    Methane, coefficients from [1]_, at 100 K:
    
    >>> Antoine(100.0, 8.7687, 395.744, -6.469)
    34478.367349639906
    
    Tetrafluoromethane, coefficients from [1]_, at 180 K
    
    >>> Antoine(180, A=8.95894, B=510.595, C=-15.95)
    702271.0518579542
    
    Oxygen at 94.91 K, with coefficients from [3]_ in units of °C, mmHg, log10,
    showing the conversion of coefficients A (mmHg to Pa) and C (°C to K)
    
    >>> Antoine(94.91, 6.83706+2.1249, 339.2095, 268.70-273.15)
    162978.88655572367

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [2] Antoine, C. 1888. Tensions des Vapeurs: Nouvelle Relation Entre les 
       Tensions et les Tempé. Compt.Rend. 107:681-684.
    .. [3] Yaws, Carl L. The Yaws Handbook of Vapor Pressure: Antoine 
       Coefficients. 1 edition. Houston, Tex: Gulf Publishing Company, 2007.
    '''
    return base**(A-B/(T+C))


def TRC_Antoine_extended(T, Tc, to, A, B, C, n, E, F):
    r'''Calculates vapor pressure of a chemical using the TRC Extended Antoine
    equation. Parameters are chemical dependent, and said to be from the 
    Thermodynamics Research Center (TRC) at Texas A&M. Coefficients for various
    chemicals can be found in [1]_.

    .. math::
        \log_{10} P^{sat} = A - \frac{B}{T + C} + 0.43429x^n + Ex^8 + Fx^{12}
        
        x = \max \left(\frac{T-t_o-273.15}{T_c}, 0 \right)

    Parameters
    ----------
    T : float
        Temperature of fluid, [K]
    A, B, C, n, E, F : floats
        Regressed coefficients for the Antoine Extended (TRC) equation,
        specific for each chemical, [-]

    Returns
    -------
    Psat : float
        Vapor pressure calculated with coefficients [Pa]
    
    Notes
    -----
    Assumes coefficients are for calculating vapor pressure in Pascal. 
    Coefficients should be consistent with input temperatures in Kelvin;

    Examples
    --------
    Tetrafluoromethane, coefficients from [1]_, at 180 K:
    
    >>> TRC_Antoine_extended(180.0, 227.51, -120., 8.95894, 510.595, -15.95, 
    ... 2.41377, -93.74, 7425.9) 
    706317.0898414153

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    x = max((T - to - 273.15)/Tc, 0.)
    return 10.**(A - B/(T+C) + 0.43429*x**n + E*x**8 + F*x**12)


def Wagner_original(T, Tc, Pc, a, b, c, d):
    r'''Calculates vapor pressure using the Wagner equation (3, 6 form).

    Requires critical temperature and pressure as well as four coefficients
    specific to each chemical.

    .. math::
        \ln P^{sat}= \ln P_c + \frac{a\tau + b \tau^{1.5} + c\tau^3 + d\tau^6}
        {T_r}
        
        \tau = 1 - \frac{T}{T_c}

    Parameters
    ----------
    T : float
        Temperature of fluid, [K]
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    a, b, c, d : floats
        Parameters for wagner equation. Specific to each chemical. [-]

    Returns
    -------
    Psat : float
        Vapor pressure at T [Pa]

    Notes
    -----
    Warning: Pc is often treated as adjustable constant.

    Examples
    --------
    Methane, coefficients from [2]_, at 100 K.

    >>> Wagner_original(100.0, 190.53, 4596420., a=-6.00435, b=1.1885, 
    ... c=-0.834082, d=-1.22833)
    34520.44601450496

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [2] McGarry, Jack. "Correlation and Prediction of the Vapor Pressures of
       Pure Liquids over Large Pressure Ranges." Industrial & Engineering
       Chemistry Process Design and Development 22, no. 2 (April 1, 1983):
       313-22. doi:10.1021/i200021a023.
    '''
    Tr = T/Tc
    tau = 1.0 - Tr
    return Pc*exp((a*tau + b*tau**1.5 + c*tau**3 + d*tau**6)/Tr)


def Wagner(T, Tc, Pc, a, b, c, d):
    r'''Calculates vapor pressure using the Wagner equation (2.5, 5 form).

    Requires critical temperature and pressure as well as four coefficients
    specific to each chemical.

    .. math::
        \ln P^{sat}= \ln P_c + \frac{a\tau + b \tau^{1.5} + c\tau^{2.5}
        + d\tau^5} {T_r}

        \tau = 1 - \frac{T}{T_c}

    Parameters
    ----------
    T : float
        Temperature of fluid, [K]
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    a, b, c, d : floats
        Parameters for wagner equation. Specific to each chemical. [-]

    Returns
    -------
    Psat : float
        Vapor pressure at T [Pa]

    Notes
    -----
    Warning: Pc is often treated as adjustable constant.

    Examples
    --------
    Methane, coefficients from [2]_, at 100 K.

    >>> Wagner(100., 190.551, 4599200, -6.02242, 1.26652, -0.5707, -1.366)
    34415.00476263708

    References
    ----------
    .. [1] Wagner, W. "New Vapour Pressure Measurements for Argon and Nitrogen and
       a New Method for Establishing Rational Vapour Pressure Equations."
       Cryogenics 13, no. 8 (August 1973): 470-82. doi:10.1016/0011-2275(73)90003-9
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    Tr = T/Tc
    tau = 1.0 - T/Tc
    return Pc*exp((a*tau + b*tau**1.5 + c*tau**2.5 + d*tau**5)/Tr)


WAGNER_MCGARRY = 'WAGNER_MCGARRY'
WAGNER_POLING = 'WAGNER_POLING'
ANTOINE_POLING = 'ANTOINE_POLING'
ANTOINE_EXTENDED_POLING = 'ANTOINE_EXTENDED_POLING'
VDI_TABULAR = 'VDI_TABULAR'
COOLPROP = 'COOLPROP'
DIPPR_PERRY_8E = 'DIPPR_PERRY_8E'
VDI_PPDS = 'VDI_PPDS'

BOILING_CRITICAL = 'BOILING_CRITICAL'
LEE_KESLER_PSAT = 'LEE_KESLER_PSAT'
AMBROSE_WALTON = 'AMBROSE_WALTON'
SANJARI = 'SANJARI'
EDALAT = 'Edalat'
EOS = 'EOS'

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

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`vapor_pressure_methods`.

    **WAGNER_MCGARRY**:
        The Wagner 3,6 original model equation documented in
        :obj:`Wagner_original`, with data for 245 chemicals, from [1]_,
    **WAGNER_POLING**:
        The Wagner 2.5, 5 model equation documented in :obj:`Wagner` in [2]_,
        with data for  104 chemicals.
    **ANTOINE_EXTENDED_POLING**:
        The TRC extended Antoine model equation documented in
        :obj:`TRC_Antoine_extended` with data for 97 chemicals in [2]_.
    **ANTOINE_POLING**:
        Standard Antoine equation, as documented in the function
        :obj:`Antoine` and with data for 325 fluids from [2]_.
        Coefficients were altered to be in units of Pa and Celcius.
    **DIPPR_PERRY_8E**:
        A collection of 341 coefficient sets from the DIPPR database published
        openly in [5]_. Provides temperature limits for all its fluids. 
        :obj:`thermo.dippr.EQ101` is used for its fluids.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS, published 
        openly in [4]_. 
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [3]_. Very slow.
    **BOILING_CRITICAL**:
        Fundamental relationship in thermodynamics making several
        approximations; see :obj:`boiling_critical_relation` for details.
        Least accurate method in most circumstances.
    **LEE_KESLER_PSAT**:
        CSP method documented in :obj:`Lee_Kesler`. Widely used.
    **AMBROSE_WALTON**:
        CSP method documented in :obj:`Ambrose_Walton`.
    **SANJARI**:
        CSP method documented in :obj:`Sanjari`.
    **EDALAT**:
        CSP method documented in :obj:`Edalat`.
    **VDI_TABULAR**:
        Tabular data in [4]_ along the saturation curve; interpolation is as
        set by the user or the default.
    **EOS**:
        Equation of state provided by user; must implement 
        :obj:`thermo.eos.GCEOS.Psat`

    See Also
    --------
    Wagner_original
    Wagner
    TRC_Antoine_extended
    Antoine
    boiling_critical_relation
    Lee_Kesler
    Ambrose_Walton
    Sanjari
    Edalat

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
    interpolation_T = lambda self, T: 1./T
    '''1/T interpolation transformation by default.'''
    interpolation_property = lambda self, P: log(P)
    '''log(P) interpolation transformation by default.'''
    interpolation_property_inv = lambda self, P: exp(P)
    '''exp(P) interpolation transformation by default; reverses
    :obj:`interpolation_property_inv`.'''
    tabular_extrapolation_permitted = False
    '''Disallow tabular extrapolation by default; CSP methods prefered
    normally.'''
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

    def __init__(self, Tb=None, Tc=None, Pc=None, omega=None, CASRN='', 
                 eos=None):
        self.CASRN = CASRN
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.eos = eos

        self.Tmin = None
        '''Minimum temperature at which no method can calculate vapor pressure
        under.'''

        self.Tmax = None
        '''Maximum temperature at which no method can calculate vapor pressure
        above; by definition the critical point.'''

        self.method = None
        '''The method was which was last used successfully to calculate a property;
        set only after the first property calculation.'''

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
        if self.CASRN in WagnerMcGarry.index:
            methods.append(WAGNER_MCGARRY)
            _, A, B, C, D, self.WAGNER_MCGARRY_Pc, self.WAGNER_MCGARRY_Tc, self.WAGNER_MCGARRY_Tmin = _WagnerMcGarry_values[WagnerMcGarry.index.get_loc(self.CASRN)].tolist()
            self.WAGNER_MCGARRY_coefs = [A, B, C, D]
            Tmins.append(self.WAGNER_MCGARRY_Tmin); Tmaxs.append(self.WAGNER_MCGARRY_Tc)
        if self.CASRN in WagnerPoling.index:
            methods.append(WAGNER_POLING)
            _, A, B, C, D, self.WAGNER_POLING_Tc, self.WAGNER_POLING_Pc, Tmin, self.WAGNER_POLING_Tmax = _WagnerPoling_values[WagnerPoling.index.get_loc(self.CASRN)].tolist()
            # Some Tmin values are missing; Arbitrary choice of 0.1 lower limit
            self.WAGNER_POLING_Tmin = Tmin if not np.isnan(Tmin) else self.WAGNER_POLING_Tmax*0.1
            self.WAGNER_POLING_coefs = [A, B, C, D]
            Tmins.append(Tmin); Tmaxs.append(self.WAGNER_POLING_Tmax)
        if self.CASRN in AntoineExtended.index:
            methods.append(ANTOINE_EXTENDED_POLING)
            _, A, B, C, Tc, to, n, E, F, self.ANTOINE_EXTENDED_POLING_Tmin, self.ANTOINE_EXTENDED_POLING_Tmax = _AntoineExtended_values[AntoineExtended.index.get_loc(self.CASRN)].tolist()
            self.ANTOINE_EXTENDED_POLING_coefs = [Tc, to, A, B, C, n, E, F]
            Tmins.append(self.ANTOINE_EXTENDED_POLING_Tmin); Tmaxs.append(self.ANTOINE_EXTENDED_POLING_Tmax)
        if self.CASRN in AntoinePoling.index:
            methods.append(ANTOINE_POLING)
            _, A, B, C, self.ANTOINE_POLING_Tmin, self.ANTOINE_POLING_Tmax = _AntoinePoling_values[AntoinePoling.index.get_loc(self.CASRN)].tolist()
            self.ANTOINE_POLING_coefs = [A, B, C]
            Tmins.append(self.ANTOINE_POLING_Tmin); Tmaxs.append(self.ANTOINE_POLING_Tmax)
        if self.CASRN in Perrys2_8.index:
            methods.append(DIPPR_PERRY_8E)
            _, C1, C2, C3, C4, C5, self.Perrys2_8_Tmin, self.Perrys2_8_Tmax = _Perrys2_8_values[Perrys2_8.index.get_loc(self.CASRN)].tolist()
            self.Perrys2_8_coeffs = [C1, C2, C3, C4, C5]
            Tmins.append(self.Perrys2_8_Tmin); Tmaxs.append(self.Perrys2_8_Tmax)
        if has_CoolProp and self.CASRN in coolprop_dict:
            methods.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
            Tmins.append(self.CP_f.Tmin); Tmaxs.append(self.CP_f.Tc)
        if self.CASRN in _VDISaturationDict:
            methods.append(VDI_TABULAR)
            Ts, props = VDI_tabular_data(self.CASRN, 'P')
            self.VDI_Tmin = Ts[0]
            self.VDI_Tmax = Ts[-1]
            self.tabular_data[VDI_TABULAR] = (Ts, props)
            Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
        if self.CASRN in VDI_PPDS_3.index:
            _,  Tm, Tc, Pc, A, B, C, D = _VDI_PPDS_3_values[VDI_PPDS_3.index.get_loc(self.CASRN)].tolist()
            self.VDI_PPDS_coeffs = [A, B, C, D]
            self.VDI_PPDS_Tc = Tc
            self.VDI_PPDS_Tm = Tm
            self.VDI_PPDS_Pc = Pc
            methods.append(VDI_PPDS)
            Tmins.append(self.VDI_PPDS_Tm); Tmaxs.append(self.VDI_PPDS_Tc)
        if all((self.Tb, self.Tc, self.Pc)):
            methods.append(BOILING_CRITICAL)
            Tmins.append(0.01); Tmaxs.append(self.Tc)
        if all((self.Tc, self.Pc, self.omega)):
            methods.append(LEE_KESLER_PSAT)
            methods.append(AMBROSE_WALTON)
            methods.append(SANJARI)
            methods.append(EDALAT)
            if self.eos:
                methods.append(EOS)
            Tmins.append(0.01); Tmaxs.append(self.Tc)
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin = min(Tmins)
            self.Tmax = max(Tmaxs)

    def calculate(self, T, method):
        r'''Method to calculate vapor pressure of a fluid at temperature `T`
        with a given method.

        This method has no exception handling; see `T_dependent_property`
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
        if method == WAGNER_MCGARRY:
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
        elif method in self.tabular_data:
            Psat = self.interpolate(T, method)
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
        if method == WAGNER_MCGARRY:
            if T < self.WAGNER_MCGARRY_Tmin or T > self.WAGNER_MCGARRY_Tc:
                return False
        elif method == WAGNER_POLING:
            if T < self.WAGNER_POLING_Tmin or T > self.WAGNER_POLING_Tmax:
                return False
        elif method == ANTOINE_EXTENDED_POLING:
            if T < self.ANTOINE_EXTENDED_POLING_Tmin or T > self.ANTOINE_EXTENDED_POLING_Tmax:
                return False
        elif method == ANTOINE_POLING:
            if T < self.ANTOINE_POLING_Tmin or T > self.ANTOINE_POLING_Tmax:
                return False
        elif method == DIPPR_PERRY_8E:
            if T < self.Perrys2_8_Tmin or T > self.Perrys2_8_Tmax:
                return False
        elif method == VDI_PPDS:
            if T > self.VDI_PPDS_Tc or T < self.VDI_PPDS_Tm:
                return False
        elif method == COOLPROP:
            if T < self.CP_f.Tmin or T < self.CP_f.Tt or T > self.CP_f.Tmax or T > self.CP_f.Tc:
                return False
        elif method in [BOILING_CRITICAL, LEE_KESLER_PSAT, AMBROSE_WALTON, SANJARI, EDALAT, EOS]:
            if T > self.Tc or T < 0:
                return False
            # No lower limit
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    return False
        else:
            raise Exception('Method not valid')
        return True


### CSP Methods

def boiling_critical_relation(T, Tb, Tc, Pc):
    r'''Calculates vapor pressure of a fluid at arbitrary temperatures using a
    CSP relationship as in [1]_; requires a chemical's critical temperature
    and pressure as well as boiling point.

    The vapor pressure is given by:

    .. math::
        \ln P^{sat}_r = h\left( 1 - \frac{1}{T_r}\right)

        h = T_{br} \frac{\ln(P_c/101325)}{1-T_{br}}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tb : float
        Boiling temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]

    Returns
    -------
    Psat : float
        Vapor pressure at T [Pa]

    Notes
    -----
    Units are Pa. Formulation makes intuitive sense; a logarithmic form of
    interpolation.

    Examples
    --------
    Example as in [1]_ for ethylbenzene

    >>> boiling_critical_relation(347.2, 409.3, 617.1, 36E5)
    15209.467273093938

    References
    ----------
    .. [1] Reid, Robert C..; Prausnitz, John M.;; Poling, Bruce E.
       The Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    Tbr = Tb/Tc
    Tr = T/Tc
    h = Tbr*log(Pc/101325.)/(1 - Tbr)
    return exp(h*(1-1/Tr))*Pc


def Lee_Kesler(T, Tc, Pc, omega):
    r'''Calculates vapor pressure of a fluid at arbitrary temperatures using a
    CSP relationship by [1]_; requires a chemical's critical temperature and
    acentric factor.

    The vapor pressure is given by:

    .. math::
        \ln P^{sat}_r = f^{(0)} + \omega f^{(1)}

        f^{(0)} = 5.92714-\frac{6.09648}{T_r}-1.28862\ln T_r + 0.169347T_r^6

        f^{(1)} = 15.2518-\frac{15.6875}{T_r} - 13.4721 \ln T_r + 0.43577T_r^6

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    omega : float
        Acentric factor [-]

    Returns
    -------
    Psat : float
        Vapor pressure at T [Pa]

    Notes
    -----
    This equation appears in [1]_ in expanded form.
    The reduced pressure form of the equation ensures predicted vapor pressure 
    cannot surpass the critical pressure.

    Examples
    --------
    Example from [2]_; ethylbenzene at 347.2 K.

    >>> Lee_Kesler(347.2, 617.1, 36E5, 0.299)
    13078.694162949312

    References
    ----------
    .. [1] Lee, Byung Ik, and Michael G. Kesler. "A Generalized Thermodynamic
       Correlation Based on Three-Parameter Corresponding States." AIChE Journal
       21, no. 3 (1975): 510-527. doi:10.1002/aic.690210313.
    .. [2] Reid, Robert C..; Prausnitz, John M.;; Poling, Bruce E.
       The Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    Tr = T/Tc
    f0 = 5.92714 - 6.09648/Tr - 1.28862*log(Tr) + 0.169347*Tr**6
    f1 = 15.2518 - 15.6875/Tr - 13.4721*log(Tr) + 0.43577*Tr**6
    return exp(f0 + omega*f1)*Pc


def Ambrose_Walton(T, Tc, Pc, omega):
    r'''Calculates vapor pressure of a fluid at arbitrary temperatures using a
    CSP relationship by [1]_; requires a chemical's critical temperature and
    acentric factor.

    The vapor pressure is given by:

    .. math::
        \ln P_r=f^{(0)}+\omega f^{(1)}+\omega^2f^{(2)}

        f^{(0)}=\frac{-5.97616\tau + 1.29874\tau^{1.5}- 0.60394\tau^{2.5}
        -1.06841\tau^5}{T_r}

        f^{(1)}=\frac{-5.03365\tau + 1.11505\tau^{1.5}- 5.41217\tau^{2.5}
        -7.46628\tau^5}{T_r}

        f^{(2)}=\frac{-0.64771\tau + 2.41539\tau^{1.5}- 4.26979\tau^{2.5}
        +3.25259\tau^5}{T_r}

        \tau = 1-T_{r}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    omega : float
        Acentric factor [-]

    Returns
    -------
    Psat : float
        Vapor pressure at T [Pa]

    Notes
    -----
    Somewhat more accurate than the :obj:`Lee_Kesler` formulation.

    Examples
    --------
    Example from [2]_; ethylbenzene at 347.25 K.

    >>> Ambrose_Walton(347.25, 617.15, 36.09E5, 0.304)
    13278.878504306222

    References
    ----------
    .. [1] Ambrose, D., and J. Walton. "Vapour Pressures up to Their Critical
       Temperatures of Normal Alkanes and 1-Alkanols." Pure and Applied
       Chemistry 61, no. 8 (1989): 1395-1403. doi:10.1351/pac198961081395.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    Tr = T/Tc
    tau = 1 - T/Tc
    f0 = (-5.97616*tau + 1.29874*tau**1.5 - 0.60394*tau**2.5 - 1.06841*tau**5)/Tr
    f1 = (-5.03365*tau + 1.11505*tau**1.5 - 5.41217*tau**2.5 - 7.46628*tau**5)/Tr
    f2 = (-0.64771*tau + 2.41539*tau**1.5 - 4.26979*tau**2.5 + 3.25259*tau**5)/Tr
    return Pc*exp(f0 + omega*f1 + omega**2*f2)


def Sanjari(T, Tc, Pc, omega):
    r'''Calculates vapor pressure of a fluid at arbitrary temperatures using a
    CSP relationship by [1]_. Requires a chemical's critical temperature,
    pressure, and acentric factor. Although developed for refrigerants,
    this model should have some general predictive ability.

    The vapor pressure of a chemical at `T` is given by:

    .. math::
        P^{sat} = P_c\exp(f^{(0)} + \omega f^{(1)} + \omega^2 f^{(2)})

        f^{(0)} = a_1 + \frac{a_2}{T_r} + a_3\ln T_r + a_4 T_r^{1.9}

        f^{(1)} = a_5 + \frac{a_6}{T_r} + a_7\ln T_r + a_8 T_r^{1.9}

        f^{(2)} = a_9 + \frac{a_{10}}{T_r} + a_{11}\ln T_r + a_{12} T_r^{1.9}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    omega : float
        Acentric factor [-]

    Returns
    -------
    Psat : float
        Vapor pressure, [Pa]

    Notes
    -----
    a[1-12] are as follows:
    6.83377, -5.76051, 0.90654, -1.16906,
    5.32034, -28.1460, -58.0352, 23.57466,
    18.19967, 16.33839, 65.6995, -35.9739.

    For a claimed fluid not included in the regression, R128, the claimed AARD
    was 0.428%. A re-calculation using 200 data points from 125.45 K to
    343.90225 K evenly spaced by 1.09775 K as generated by NIST Webbook April
    2016 produced an AARD of 0.644%. It is likely that the author's regression
    used more precision in its coefficients than was shown here. Nevertheless,
    the function is reproduced as shown in [1]_.

    For Tc=808 K, Pc=1100000 Pa, omega=1.1571, this function actually declines
    after 770 K.

    Examples
    --------
    >>> Sanjari(347.2, 617.1, 36E5, 0.299)
    13651.916109552498

    References
    ----------
    .. [1] Sanjari, Ehsan, Mehrdad Honarmand, Hamidreza Badihi, and Ali
       Ghaheri. "An Accurate Generalized Model for Predict Vapor Pressure of
       Refrigerants." International Journal of Refrigeration 36, no. 4
       (June 2013): 1327-32. doi:10.1016/j.ijrefrig.2013.01.007.
    '''
    Tr = T/Tc
    f0 = 6.83377 + -5.76051/Tr + 0.90654*log(Tr) + -1.16906*Tr**1.9
    f1 = 5.32034 + -28.1460/Tr + -58.0352*log(Tr) + 23.57466*Tr**1.9
    f2 = 18.19967 + 16.33839/Tr + 65.6995*log(Tr) + -35.9739*Tr**1.9
    return Pc*exp(f0 + omega*f1 + omega**2*f2)


def Edalat(T, Tc, Pc, omega):
    r'''Calculates vapor pressure of a fluid at arbitrary temperatures using a
    CSP relationship by [1]_. Requires a chemical's critical temperature,
    pressure, and acentric factor. Claimed to have a higher accuracy than the
    Lee-Kesler CSP relationship.

    The vapor pressure of a chemical at `T` is given by:

    .. math::
        \ln(P^{sat}/P_c) = \frac{a\tau + b\tau^{1.5} + c\tau^3 + d\tau^6}
        {1-\tau}
        
        a = -6.1559 - 4.0855\omega
        
        b = 1.5737 - 1.0540\omega - 4.4365\times 10^{-3} d
        
        c = -0.8747 - 7.8874\omega
        
        d = \frac{1}{-0.4893 - 0.9912\omega + 3.1551\omega^2}
        
        \tau = 1 - \frac{T}{T_c}
        
    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    omega : float
        Acentric factor [-]

    Returns
    -------
    Psat : float
        Vapor pressure, [Pa]

    Notes
    -----
    [1]_ found an average error of 6.06% on 94 compounds and 1106 data points.
    
    Examples
    --------
    >>> Edalat(347.2, 617.1, 36E5, 0.299)
    13461.273080743307

    References
    ----------
    .. [1] Edalat, M., R. B. Bozar-Jomehri, and G. A. Mansoori. "Generalized 
       Equation Predicts Vapor Pressure of Hydrocarbons." Oil and Gas Journal; 
       91:5 (February 1, 1993).
    '''
    tau = 1. - T/Tc
    a = -6.1559 - 4.0855*omega
    c = -0.8747 - 7.8874*omega
    d = 1./(-0.4893 - 0.9912*omega + 3.1551*omega**2)
    b = 1.5737 - 1.0540*omega - 4.4365E-3*d
    lnPr = (a*tau + b*tau**1.5 + c*tau**3 + d*tau**6)/(1.-tau)
    return exp(lnPr)*Pc

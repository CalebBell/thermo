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
import os
from math import log, exp
import numpy as np
import pandas as pd

from scipy.constants import R, calorie
from scipy.integrate import quad

from thermo.utils import (to_num, property_molar_to_mass, none_and_length_check,
                          mixing_simple, property_mass_to_molar)
from thermo.miscdata import _VDISaturationDict, VDI_tabular_data
from thermo.electrochem import (Laliberte_heat_capacity,
                                _Laliberte_Heat_Capacity_ParametersDict)
from thermo.utils import TDependentProperty
from thermo.coolprop import *


folder = os.path.join(os.path.dirname(__file__), 'Heat Capacity')


Poling_data = pd.read_csv(os.path.join(folder,
                       'PolingDatabank.csv'), sep='\t',
                       index_col=0)
_Poling_data_values = Poling_data.values


TRC_gas_data = pd.read_csv(os.path.join(folder,
                       'TRC Thermodynamics of Organic Compounds in the Gas State.csv'), sep='\t',
                       index_col=0)
_TRC_gas_data_values = TRC_gas_data.values



_PerryI = {}
with open(os.path.join(folder, 'Perrys Table 2-151.csv')) as f:
    '''Read in a dict of heat capacities of irnorganic and elemental solids.
    These are in section 2, table 151 in:
    Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
    Eighth Edition. McGraw-Hill Professional, 2007.

    Formula:
    Cp(Cal/mol/K) = Const + Lin*T + Quadinv/T^2 + Quadinv*T^2

    Phases: c, gls, l, g.
    '''
    next(f)
    for line in f:
        values = to_num(line.strip('\n').split('\t'))
        (CASRN, _formula, _phase, _subphase, Const, Lin, Quadinv, Quad, Tmin,
         Tmax, err) = values
        if Lin is None:
            Lin = 0
        if Quadinv is None:
            Quadinv = 0
        if Quad is None:
            Quad = 0
        if CASRN in _PerryI and CASRN:
            a = _PerryI[CASRN]
            a.update({_phase: {"Formula": _formula, "Phase": _phase,
                               "Subphase": _subphase, "Const": Const,
                               "Lin": Lin, "Quadinv": Quadinv, "Quad": Quad,
                               "Tmin": Tmin, "Tmax": Tmax, "Error": err}})
            _PerryI[CASRN] = a
        else:
            _PerryI[CASRN] = {_phase: {"Formula": _formula, "Phase": _phase,
                                       "Subphase": _subphase, "Const": Const,
                                       "Lin": Lin, "Quadinv": Quadinv,
                                       "Quad": Quad, "Tmin": Tmin,
                                       "Tmax": Tmax, "Error": err}}


#    '''Read in a dict of 2481 thermodynamic property sets of different phases from:
#        Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
#        Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
#        Warning: 11 duplicated chemicals are present and currently clobbered.
CRC_standard_data = pd.read_csv(os.path.join(folder,
                       'CRC Standard Thermodynamic Properties of Chemical Substances.csv'), sep='\t',
                       index_col=0)



### Heat capacities of gases

def Lastovka_Shaw(T, similarity_variable, cyclic_aliphatic=False):
    r'''Calculate ideal-gas constant-pressure heat capacitiy with the similarity
    variable concept and method as shown in [1]_.

    .. math::
        C_p^0 = \left(A_2 + \frac{A_1 - A_2}{1 + \exp(\alpha-A_3/A_4)}\right)
        + (B_{11} + B_{12}\alpha)\left(-\frac{(C_{11} + C_{12}\alpha)}{T}\right)^2
        \frac{\exp(-(C_{11} + C_{12}\alpha)/T)}{[1-\exp(-(C_{11}+C_{12}\alpha)/T)]^2}\\
        + (B_{21} + B_{22}\alpha)\left(-\frac{(C_{21} + C_{22}\alpha)}{T}\right)^2
        \frac{\exp(-(C_{21} + C_{22}\alpha)/T)}{[1-\exp(-(C_{21}+C_{22}\alpha)/T)]^2}

    Parameters
    ----------
    T : float
        Temperature of gas [K]
    similarity_variable : float
        similarity variable as defined in [1]_, [mol/g]

    Returns
    -------
    Cpg : float
        Gas constant-pressure heat capacitiy, [J/kg/K]

    Notes
    -----
    Original model is in terms of J/g/K. Note that the model is for predicting
    mass heat capacity, not molar heat capacity like most other methods!

    A1 = 0.58, A2 = 1.25, A3 = 0.17338003, A4 = 0.014, B11 = 0.73917383,
    B12 = 8.88308889, C11 = 1188.28051, C12 = 1813.04613, B21 = 0.0483019,
    B22 = 4.35656721, C21 = 2897.01927, C22 = 5987.80407.

    Examples
    --------
    >>> Lastovka_Shaw(1000.0, 0.1333)
    2467.113309084757

    References
    ----------
    .. [1] Lastovka, Vaclav, and John M. Shaw. "Predictive Correlations for
       Ideal Gas Heat Capacities of Pure Hydrocarbons and Petroleum Fractions."
       Fluid Phase Equilibria 356 (October 25, 2013): 338-370.
       doi:10.1016/j.fluid.2013.07.023.
    '''
    a = similarity_variable
    if cyclic_aliphatic:
        A1 = -0.1793547
        A2 = 3.86944439
        first = A1 + A2*a
    else:
        A1 = 0.58
        A2 = 1.25
        A3 = 0.17338003 # 803 instead of 8003 in another paper
        A4 = 0.014
        first = A2 + (A1-A2)/(1+exp((a-A3)/A4)) # One reference says exp((a-A3)/A4)
        # Personal communication confirms the change

    B11 = 0.73917383
    B12 = 8.88308889
    C11 = 1188.28051
    C12 = 1813.04613
    B21 = 0.0483019
    B22 = 4.35656721
    C21 = 2897.01927
    C22 = 5987.80407
    Cp = first + (B11 + B12*a)*(-(C11+C12*a)/T)**2*exp(-(C11 + C12*a)/T)/(1-exp(-(C11+C12*a)/T))**2
    Cp += (B21 + B22*a)*(-(C21+C22*a)/T)**2*exp(-(C21 + C22*a)/T)/(1-exp(-(C21+C22*a)/T))**2
    Cp = Cp*1000 # J/g/K to J/kg/K
    return Cp


def TRCCp(T, a0, a1, a2, a3, a4, a5, a6, a7):
    r'''Calculates ideal gas heat capacity using the model developed in [1]_.

    The ideal gas heat capacity is given by:

    .. math::
        C_p = R\left(a_0 + (a_1/T^2) \exp(-a_2/T) + a_3 y^2
        + (a_4 - a_5/(T-a_7)^2 )y^j \right)

        y = \frac{T-a_7}{T+a_6} \text{ for } T > a_7 \text{ otherwise } 0

    Parameters
    ----------
    T : float
        Temperature [K]
    a1-a7 : float
        Coefficients

    Returns
    -------
    Cp : float
        Ideal gas heat capacity , [J/mol/K]

    Notes
    -----
    j is set to 8. Analytical integrals are available for this expression.

    Examples
    --------
    >>> TRCCp(300, 4.0, 7.65E5, 720., 3.565, -0.052, -1.55E6, 52., 201.)
    42.06525682312236

    References
    ----------
    .. [1] Kabo, G. J., and G. N. Roganov. Thermodynamics of Organic Compounds
       in the Gas State, Volume II: V. 2. College Station, Tex: CRC Press, 1994.
    '''
    j = 8.
    if T <= a7:
        y = 0.
    else:
        y = (T - a7)/(T + a6)
    Cp = R*(a0 + (a1/T**2)*exp(-a2/T) + a3*y**2 + (a4 - a5/(T-a7)**2 )*y**j )
    return Cp


TRCIG = 'TRC Thermodynamics of Organic Compounds in the Gas State (1994)'
POLING = 'Poling et al. (2001)'
POLING_CONST = 'Poling et al. (2001) constant'
CRCSTD = 'CRC Standard Thermodynamic Properties of Chemical Substances'
VDI_TABULAR = 'VDI Heat Atlas'
LASTOVKA_SHAW = 'Lastovka and Shaw (2013)'
COOLPROP = 'CoolProp'
heat_capacity_gas_methods = [TRCIG, POLING, COOLPROP, LASTOVKA_SHAW, CRCSTD,
                             POLING_CONST, VDI_TABULAR]
'''Holds all methods available for the HeatCapacityGas class, for use in
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

    Notes
    -----
    A string holding each method's name is assigned to the following variables
    in this module, intended as the most convenient way to refer to a method.
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_gas_methods`.

    **TRCIG**:
        A rigorous expression derived in [1]_ for modeling gas heat capacity.
        Coefficients for 1961 chemicals are available.
    **POLING**:
        Simple polynomials in [2]_ not suitable for extrapolation. Data is
        available for 308 chemicals.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [3]_. Very slow.
    **LASTOVKA_SHAW**:
        A basic estimation method using the `similarity variable` concept;
        requires only molecular structure, so is very convenient. See
        :obj:`Lastovka_Shaw` for details.
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
    TRCCp
    Lastovka_Shaw
    Rowlinson_Poling
    Rowlinson_Bondi

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
    '''Maximum valid of Heat capacity; arbitrarily set.'''


    ranked_methods = [TRCIG, POLING, COOLPROP, LASTOVKA_SHAW, CRCSTD, POLING_CONST, VDI_TABULAR]
    '''Default rankings of the available methods.'''

    def __init__(self, CASRN='', MW=None, similarity_variable=None):
        self.CASRN = CASRN
        self.MW = MW
        self.similarity_variable = similarity_variable

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        surface tension under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        surface tension above.'''

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
        if self.CASRN in TRC_gas_data.index:
            methods.append(TRCIG)
            _, self.TRCIG_Tmin, self.TRCIG_Tmax, a0, a1, a2, a3, a4, a5, a6, a7, _, _, _ = _TRC_gas_data_values[TRC_gas_data.index.get_loc(self.CASRN)].tolist()
            self.TRCIG_coefs = [a0, a1, a2, a3, a4, a5, a6, a7]
            Tmins.append(self.TRCIG_Tmin); Tmaxs.append(self.TRCIG_Tmax)
        if self.CASRN in Poling_data.index and not np.isnan(Poling_data.at[self.CASRN, 'a0']):
            _, self.POLING_Tmin, self.POLING_Tmax, a0, a1, a2, a3, a4, Cpg, Cpl = _Poling_data_values[Poling_data.index.get_loc(self.CASRN)].tolist()
            methods.append(POLING)
            self.POLING_coefs = [a0, a1, a2, a3, a4]
            Tmins.append(self.POLING_Tmin); Tmaxs.append(self.POLING_Tmax)
        if self.CASRN in Poling_data.index and not np.isnan(Poling_data.at[self.CASRN, 'Cpg']):
            methods.append(POLING_CONST)
            self.POLING_T = 298.15
            self.POLING_constant = float(Poling_data.at[self.CASRN, 'Cpg'])
        if self.CASRN in CRC_standard_data.index and not np.isnan(CRC_standard_data.at[self.CASRN, 'Cpg']):
            methods.append(CRCSTD)
            self.CRCSTD_T = 298.15
            self.CRCSTD_constant = float(CRC_standard_data.at[self.CASRN, 'Cpg'])
        if self.CASRN in _VDISaturationDict:
            # NOTE: VDI data is for the saturation curve, i.e. at increasing
            # pressure; it is normally substantially higher than the ideal gas
            # value
            methods.append(VDI_TABULAR)
            Ts, props = VDI_tabular_data(self.CASRN, 'Cp (g)')
            self.VDI_Tmin = Ts[0]
            self.VDI_Tmax = Ts[-1]
            self.tabular_data[VDI_TABULAR] = (Ts, props)
            Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
        if has_CoolProp and self.CASRN in coolprop_dict:
            methods.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
            Tmins.append(self.CP_f.Tt); Tmaxs.append(self.CP_f.Tc)
        if self.MW and self.similarity_variable:
            methods.append(LASTOVKA_SHAW)
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)


    def calculate(self, T, method):
        r'''Method to calculate surface tension of a liquid at temperature `T`
        with a given method.

        This method has no exception handling; see `T_dependent_property`
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
        if method == TRCIG:
            a0, a1, a2, a3, a4, a5, a6, a7 = self.TRCIG_coefs
            Cp = TRCCp(T, a0, a1, a2, a3, a4, a5, a6, a7)
        elif method == COOLPROP:
            Cp = PropsSI('Cp0molar', 'T', T,'P', 101325.0, self.CASRN)
        elif method == POLING:
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
        elif method in self.tabular_data:
            Cp = self.interpolate(T, method)
        return Cp


    def test_method_validity(self, T, method):
        r'''Method to test the validity of a specified method for a given
        temperature.

        'TRC' and 'Poling' both have minimum and maimum temperatures. The
        constant temperatures in POLING_CONST and CRCSTD are considered valid
        for 50 degrees around their specified temperatures.
        :obj:`Lastovka_Shaw` is considered valid for the whole range of
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
                validity = False
        elif method == POLING:
            if T < self.POLING_Tmin or T > self.POLING_Tmax:
                validity = False
        elif method == POLING_CONST:
            if T > self.POLING_T + 50 or T < self.POLING_T - 50:
                validity = False
        elif method == CRCSTD:
            if T > self.CRCSTD_T + 50 or T < self.CRCSTD_T - 50:
                validity = False
        elif method == LASTOVKA_SHAW:
            pass # Valid everywhere
        elif method == COOLPROP:
            if T <= self.CP_f.Tmin or T >= self.CP_f.Tmax:
                validity = False
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity


### Heat capacities of liquids

def Rowlinson_Poling(T, Tc, omega, Cpgm):
    r'''Calculate liquid constant-pressure heat capacitiy with the [1]_ CSP method.

    This equation is not terrible accurate.

    The heat capacity of a liquid is given by:

    .. math::
        \frac{Cp^{L} - Cp^{g}}{R} = 1.586 + \frac{0.49}{1-T_r} +
        \omega\left[ 4.2775 + \frac{6.3(1-T_r)^{1/3}}{T_r} + \frac{0.4355}{1-T_r}\right]

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    omega : float
        Acentric factor for fluid, [-]
    Cpgm : float
        Constant-pressure gas heat capacity, [J/mol/K]

    Returns
    -------
    Cplm : float
        Liquid constant-pressure heat capacitiy, [J/mol/K]

    Notes
    -----
    Poling compared 212 substances, and found error at 298K larger than 10%
    for 18 of them, mostly associating. Of the other 194 compounds, AARD is 2.5%.

    Examples
    --------
    >>> Rowlinson_Poling(350.0, 435.5, 0.203, 91.21)
    143.80194441498296

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    Tr = T/Tc
    Cplm = Cpgm+ R*(1.586 + 0.49/(1.-Tr) + omega*(4.2775
    + 6.3*(1-Tr)**(1/3.)/Tr + 0.4355/(1.-Tr)))
    return Cplm


def Rowlinson_Bondi(T, Tc, omega, Cpgm):
    r'''Calculate liquid constant-pressure heat capacitiy with the CSP method
    shown in [1]_.

    The heat capacity of a liquid is given by:

    .. math::
        \frac{Cp^L - Cp^{ig}}{R} = 1.45 + 0.45(1-T_r)^{-1} + 0.25\omega
        [17.11 + 25.2(1-T_r)^{1/3}T_r^{-1} + 1.742(1-T_r)^{-1}]

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    omega : float
        Acentric factor for fluid, [-]
    Cpgm : float
        Constant-pressure gas heat capacity, [J/mol/K]

    Returns
    -------
    Cplm : float
        Liquid constant-pressure heat capacitiy, [J/mol/K]

    Notes
    -----
    Less accurate than `Rowlinson_Poling`.

    Examples
    --------
    >>> Rowlinson_Bondi(T=373.28, Tc=535.55, omega=0.323, Cpgm=119.342)
    175.39760730048116

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    .. [3] J.S. Rowlinson, Liquids and Liquid Mixtures, 2nd Ed.,
       Butterworth, London (1969).
    '''
    Tr = T/Tc
    Cplm = Cpgm + R*(1.45 + 0.45/(1.-Tr) + 0.25*omega*(17.11
    + 25.2*(1-Tr)**(1/3.)/Tr + 1.742/(1.-Tr)))
    return Cplm


def Dadgostar_Shaw(T, similarity_variable):
    r'''Calculate liquid constant-pressure heat capacitiy with the similarity
    variable concept and method as shown in [1]_.

    .. math::
        C_{p} = 24.5(a_{11}\alpha + a_{12}\alpha^2)+ (a_{21}\alpha
        + a_{22}\alpha^2)T +(a_{31}\alpha + a_{32}\alpha^2)T^2

    Parameters
    ----------
    T : float
        Temperature of liquid [K]
    similarity_variable : float
        similarity variable as defined in [1]_, [mol/g]

    Returns
    -------
    Cpl : float
        Liquid constant-pressure heat capacitiy, [J/kg/K]

    Notes
    -----
    Many restrictions on its use.

    Original model is in terms of J/g/K. Note that the model is for predicting
    mass heat capacity, not molar heat capacity like most other methods!

    a11 = -0.3416; a12 = 2.2671; a21 = 0.1064; a22 = -0.3874l;
    a31 = -9.8231E-05; a32 = 4.182E-04

    Examples
    --------
    >>> Dadgostar_Shaw(355.6, 0.139)
    1802.5291501191516

    References
    ----------
    .. [1] Dadgostar, Nafiseh, and John M. Shaw. "A Predictive Correlation for
       the Constant-Pressure Specific Heat Capacity of Pure and Ill-Defined
       Liquid Hydrocarbons." Fluid Phase Equilibria 313 (January 15, 2012):
       211-226. doi:10.1016/j.fluid.2011.09.015.
    '''
    a11 = -0.3416
    a12 = 2.2671
    a21 = 0.1064
    a22 = -0.3874
    a31 = -9.8231E-05
    a32 = 4.182E-04

    # Didn't seem to improve the comparison; sum of errors on some
    # points included went from 65.5  to 286.
    # Author probably used more precision in their calculation.
#    theta = 151.8675
#    constant = 3*R*(theta/T)**2*exp(theta/T)/(exp(theta/T)-1)**2
    constant = 24.5

    Cp = (constant*(a11*similarity_variable + a12*similarity_variable**2)
    + (a21*similarity_variable + a22*similarity_variable**2)*T
    + (a31*similarity_variable + a32*similarity_variable**2)*T**2)
    Cp = Cp*1000 # J/g/K to J/kg/K
    return Cp


_ZabranskySats = {}
_ZabranskyConsts = {}
_ZabranskyIsos = {}

_ZabranskySatp = {}
_ZabranskyConstp = {}
_ZabranskyIsop = {}


def _append2dict(maindict, newdict):
    '''
    Inputs: Dict entry or []; and the dict type
    '''
    data = None
    for datum in maindict:  # list of dicts
        if newdict["Tmin"] < datum["Tmin"]:
            data = maindict  # list of dicts
            data.insert(data.index(datum), newdict)
            break
    if not data:  # Run if Tmin never under Tmins in dict (put at end)
        data = maindict
        data.extend([newdict])
    return data


with open(os.path.join(folder, 'Zabransky.csv')) as f:
    next(f)
    for line in f:
        values = to_num(line.strip('\n').split('\t'))
        # s for spline, p for quasipolynomial
        (CASRN, _name, Type, Uncertainty, Tmin, Tmax, a1s, a2s, a3s, a4s, a1p, a2p, a3p, a4p, a5p, a6p, Tc) = values

        _ZabranskyDict = {"name" : _name , "Type" : Type,
        "Uncertainty" : Uncertainty, "Tmin": Tmin, "Tmax" : Tmax, "a1s" : a1s,
        "a2s" : a2s, "a3s": a3s, "a4s" : a4s, "a1p" : a1p, "a2p" : a2p,
        "a3p" : a3p, "a4p" : a4p,  "a5p" : a5p, "a6p" : a6p, "Tc" : Tc }
        # dict[CASRN] = [subdict1, subdict2] where each is sorted
        if Type == 'sat':
            if a1s: # sat, spline ONLY
                _ZabranskySats[CASRN] = _append2dict((_ZabranskySats[CASRN] if CASRN in _ZabranskySats else []), _ZabranskyDict)
            elif a1p: # sat, polynomial ONLY
                _ZabranskySatp[CASRN] = _append2dict((_ZabranskySatp[CASRN] if CASRN in _ZabranskySatp else []), _ZabranskyDict)
        elif Type == 'p':
            if a1s: # sat, spline ONLY
                _ZabranskyConsts[CASRN] = _append2dict((_ZabranskyConsts[CASRN] if CASRN in _ZabranskyConsts else []), _ZabranskyDict)
            elif a1p: # sat, polynomial ONLY
                _ZabranskyConstp[CASRN] = _append2dict((_ZabranskyConstp[CASRN] if CASRN in _ZabranskyConstp else []), _ZabranskyDict)
        elif Type == 'C':
            if a1s: # sat, spline ONLY
                _ZabranskyIsos[CASRN] = _append2dict((_ZabranskyIsos[CASRN] if CASRN in _ZabranskyIsos else []), _ZabranskyDict)
            elif a1p: # sat, polynomial ONLY
                _ZabranskyIsop[CASRN] = _append2dict((_ZabranskyIsop[CASRN] if CASRN in _ZabranskyIsop else []), _ZabranskyDict)



def Zabransky_quasi_polynomial(T, Tc, a1, a2, a3, a4, a5, a6):
    r'''Calculates liquid heat capacity using the model developed in [1]_.

    .. math::
        \frac{C}{R}=A_1\ln(1-T_r) + \frac{A_2}{1-T_r}
        + \sum_{j=0}^m A_{j+3} T_r^j

    Parameters
    ----------
    T : float
        Temperature [K]
    Tc : float
        Critical temperature of fluid, [K]
    a1-a6 : float
        Coefficients

    Returns
    -------
    Cp : float
        Liquid heat capacity, [J/mol/K]

    Notes
    -----
    Used only for isobaric heat capacities, not saturation heat capacities.
    Designed for reasonable extrapolation behavior caused by using the reduced
    critical temperature. Used by the authors of [1]_ when critical temperature
    was available for the fluid.
    Analytical integrals are available for this expression.

    Examples
    --------
    >>> Zabransky_quasi_polynomial(330, 591.79, -3.12743, 0.0857315, 13.7282, 1.28971, 6.42297, 4.10989)
    165.4728226923247

    References
    ----------
    .. [1] Zabransky, M., V. Ruzicka Jr, V. Majer, and Eugene S. Domalski.
       Heat Capacity of Liquids: Critical Review and Recommended Values.
       2 Volume Set. Washington, D.C.: Amer Inst of Physics, 1996.
    '''
    Tr = T/Tc
    return R*(a1*log(1-Tr) + a2/(1-Tr) + a3 + a4*Tr + a5*Tr**2 + a6*Tr**3)


def Zabransky_cubic(T, a1, a2, a3, a4):
    r'''Calculates liquid heat capacity using the model developed in [1]_.

    .. math::
        \frac{C}{R}=\sum_{j=0}^3 A_{j+1} \left(\frac{T}{100}\right)^j

    Parameters
    ----------
    T : float
        Temperature [K]
    a1-a4 : float
        Coefficients

    Returns
    -------
    Cp : float
        Liquid heat capacity, [J/mol/K]

    Notes
    -----
    Most often form used in [1]_.
    Analytical integrals are available for this expression.

    Examples
    --------
    >>> Zabransky_cubic(298.15, 20.9634, -10.1344, 2.8253, -0.256738)
    75.31462591538555

    References
    ----------
    .. [1] Zabransky, M., V. Ruzicka Jr, V. Majer, and Eugene S. Domalski.
       Heat Capacity of Liquids: Critical Review and Recommended Values.
       2 Volume Set. Washington, D.C.: Amer Inst of Physics, 1996.
    '''
    T = T/100.
    return R*(a1 + a2*T**1 + a3*T**2 + a4*T**3)


def _ZabranskyDictChoser(T, diclist, strict=False):
    ans = None
    if len(diclist) == 1: # one entry
            ans = diclist[0]
    else:
        for data in diclist:
            if T < data["Tmin"]: # multiple entries, under Tmin
                ans = data
                break
            elif T >= data["Tmin"] and T <= data["Tmax"]: # Tmin < T < Tmax
                ans = data
                break
        if not ans:
            ans = diclist[-1] # last entry; T > Tmax, last case

#    if strict:
#        if T < ans["Tmin"] or T > ans["Tmax"]: # either side error
#            ans = None
    return ans





POST_CRITICAL = 'Post-critical'
ZABRANSKY_SPLINE = 'Zabransky spline'
ZABRANSKY_QUASIPOLYNOMIAL = 'Zabransky quasipolynomial'
ZABRANSKY_SPLINE_C = 'Zabransky spline, C'
ZABRANSKY_QUASIPOLYNOMIAL_C = 'Zabransky quasipolynomial, C'
ZABRANSKY_SPLINE_SAT = 'Zabransky spline, saturation'
ZABRANSKY_QUASIPOLYNOMIAL_SAT = 'Zabransky quasipolynomial, saturation'
ROWLINSON_POLING = 'Rowlinson and Poling (2001)'
ROWLINSON_BONDI = 'Rowlinson and Bondi (1969)'
DADGOSTAR_SHAW = 'Dadgostar and Shaw (2011)'


ZABRANSKY_TO_DICT = {ZABRANSKY_SPLINE: _ZabranskyConsts,
                     ZABRANSKY_QUASIPOLYNOMIAL: _ZabranskyConstp,
                     ZABRANSKY_SPLINE_C: _ZabranskyIsos,
                     ZABRANSKY_QUASIPOLYNOMIAL_C: _ZabranskyIsop,
                     ZABRANSKY_SPLINE_SAT: _ZabranskySats,
                     ZABRANSKY_QUASIPOLYNOMIAL_SAT: _ZabranskySatp}
heat_capacity_liquid_methods = [ZABRANSKY_SPLINE, ZABRANSKY_QUASIPOLYNOMIAL,
                      ZABRANSKY_SPLINE_C, ZABRANSKY_QUASIPOLYNOMIAL_C,
                      ZABRANSKY_SPLINE_SAT, ZABRANSKY_QUASIPOLYNOMIAL_SAT,
                      VDI_TABULAR, ROWLINSON_POLING, ROWLINSON_BONDI, COOLPROP,
                      DADGOSTAR_SHAW, POLING_CONST, CRCSTD]
'''Holds all methods available for the HeatCapacityLiquid class, for use in
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

    Notes
    -----
    A string holding each method's name is assigned to the following variables
    in this module, intended as the most convenient way to refer to a method.
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_gas_methods`.

    **ZABRANSKY_SPLINE, ZABRANSKY_QUASIPOLYNOMIAL, ZABRANSKY_SPLINE_C,
    and ZABRANSKY_QUASIPOLYNOMIAL_C**:
        Rigorous expressions developed in [1]_ following critical evaluation
        of the available data. The spline methods use the form described in
        :obj:`Zabransky_cubic` over short ranges with varying coefficients
        to obtain a wider range. The quasi-polynomial methods use the form
        described in :obj:`Zabransky_quasi_polynomial`, more suitable for
        extrapolation, and over then entire range. Respectively, there is data
        available for 588, 146, 51, and 26 chemicals.
    **ZABRANSKY_SPLINE_SAT and ZABRANSKY_QUASIPOLYNOMIAL_SAT**:
        Rigorous expressions developed in [1]_ following critical evaluation
        of the available data. The spline method use the form described in
        :obj:`Zabransky_cubic` over short ranges with varying coefficients
        to obtain a wider range. The quasi-polynomial method use the form
        described in :obj:`Zabransky_quasi_polynomial`, more suitable for
        extrapolation, and over then entire range. Respectively, there is data
        available for 203, and 16 chemicals. Note that these methods are for
        the saturation curve!
    **VDI_TABULAR**:
        Tabular data up to the critical point available in [5]_. Note that this
        data is along the saturation curve.
    **ROWLINSON_POLING**:
        CSP method described in :obj:`Rowlinson_Poling`. Requires a ideal gas
        heat capacity value at the same temperature as it is to be calculated.
    **ROWLINSON_BONDI**:
        CSP method described in :obj:`Rowlinson_Bondi`. Requires a ideal gas
        heat capacity value at the same temperature as it is to be calculated.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [3]_. Very slow.
    **DADGOSTAR_SHAW**:
        A basic estimation method using the `similarity variable` concept;
        requires only molecular structure, so is very convenient. See
        :obj:`Dadgostar_Shaw` for details.
    **POLING_CONST**:
        Constant values in [2]_ at 298.15 K; available for 245 liquids.
    **CRCSTD**:
        Consta values tabulated in [4]_ at 298.15 K; data is available for 433
        liquids.

    See Also
    --------
    Zabransky_quasi_polynomial
    Zabransky_cubic
    Rowlinson_Poling
    Rowlinson_Bondi
    Dadgostar_Shaw

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
    property_max = 1E4
    '''Maximum valid of Heat capacity; arbitrarily set.'''


    ranked_methods = [ZABRANSKY_SPLINE, ZABRANSKY_QUASIPOLYNOMIAL,
                      ZABRANSKY_SPLINE_C, ZABRANSKY_QUASIPOLYNOMIAL_C,
                      ZABRANSKY_SPLINE_SAT, ZABRANSKY_QUASIPOLYNOMIAL_SAT,
                      VDI_TABULAR, ROWLINSON_POLING, ROWLINSON_BONDI,
                      COOLPROP, DADGOSTAR_SHAW, POLING_CONST, CRCSTD]
    '''Default rankings of the available methods.'''


    def __init__(self, CASRN='', MW=None, similarity_variable=None, Tc=None,
                 omega=None, Cpgm=None):
        self.CASRN = CASRN
        self.MW = MW
        self.Tc = Tc
        self.omega = omega
        self.Cpgm = Cpgm
        self.similarity_variable = similarity_variable

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        heat capacity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        heat capacity above.'''

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
        if self.CASRN in _ZabranskyConsts:
            methods.append(ZABRANSKY_SPLINE)
            self.ZABRANSKY_SPLINE_data = _ZabranskyConsts[self.CASRN]
        if self.CASRN in _ZabranskyConstp:
            methods.append(ZABRANSKY_QUASIPOLYNOMIAL)
            self.ZABRANSKY_QUASIPOLYNOMIAL_data = _ZabranskyConstp[self.CASRN]
        if self.CASRN in _ZabranskyIsos:
            methods.append(ZABRANSKY_SPLINE_C)
            self.ZABRANSKY_SPLINE_C_data = _ZabranskyIsos[self.CASRN]
        if self.CASRN in _ZabranskyIsop:
            methods.append(ZABRANSKY_QUASIPOLYNOMIAL_C)
            self.ZABRANSKY_QUASIPOLYNOMIAL_C_data = _ZabranskyIsop[self.CASRN]
        if self.CASRN in Poling_data.index and not np.isnan(Poling_data.at[self.CASRN, 'Cpl']):
            methods.append(POLING_CONST)
            self.POLING_T = 298.15
            self.POLING_constant = float(Poling_data.at[self.CASRN, 'Cpl'])
        if self.CASRN in CRC_standard_data.index and not np.isnan(CRC_standard_data.at[self.CASRN, 'Cpl']):
            methods.append(CRCSTD)
            self.CRCSTD_T = 298.15
            self.CRCSTD_constant = float(CRC_standard_data.at[self.CASRN, 'Cpl'])
        # Saturation functions
        if self.CASRN in _ZabranskySats:
            methods.append(ZABRANSKY_SPLINE_SAT)
            self.ZABRANSKY_SPLINE_SAT_data = _ZabranskySats[self.CASRN]
        if self.CASRN in _ZabranskySatp:
            methods.append(ZABRANSKY_QUASIPOLYNOMIAL_SAT)
            self.ZABRANSKY_QUASIPOLYNOMIAL_SAT_data = _ZabranskySatp[self.CASRN]
        if self.CASRN in _VDISaturationDict:
            # NOTE: VDI data is for the saturation curve, i.e. at increasing
            # pressure; it is normally substantially higher than the ideal gas
            # value
            methods.append(VDI_TABULAR)
            Ts, props = VDI_tabular_data(self.CASRN, 'Cp (l)')
            self.VDI_Tmin = Ts[0]
            self.VDI_Tmax = Ts[-1]
            self.tabular_data[VDI_TABULAR] = (Ts, props)
            Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
        if self.Tc and self.omega:
            methods.extend([ROWLINSON_POLING, ROWLINSON_BONDI])
        if has_CoolProp and self.CASRN in coolprop_dict:
            methods.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
            Tmins.append(self.CP_f.Tt); Tmaxs.append(self.CP_f.Tc)
        if self.MW and self.similarity_variable:
            methods.append(DADGOSTAR_SHAW)
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            # TODO: More Tmin, Tmax ranges
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)


    def calculate(self, T, method):
        r'''Method to calculate heat capacity of a liquid at temperature `T`
        with a given method.

        This method has no exception handling; see `T_dependent_property`
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
        if method == ZABRANSKY_SPLINE:
            data = _ZabranskyDictChoser(T, self.ZABRANSKY_SPLINE_data)
            Cp = Zabransky_cubic(T, data["a1s"], data["a2s"], data["a3s"], data["a4s"])
        elif method == ZABRANSKY_QUASIPOLYNOMIAL:
            data = _ZabranskyDictChoser(T, self.ZABRANSKY_QUASIPOLYNOMIAL_data)
            Cp = Zabransky_quasi_polynomial(T, data["Tc"], data["a1p"], data["a2p"], data["a3p"], data["a4p"], data["a5p"], data["a6p"])
        elif method == ZABRANSKY_SPLINE_C:
            data = _ZabranskyDictChoser(T,self.ZABRANSKY_SPLINE_C_data)
            Cp = Zabransky_cubic(T, data["a1s"], data["a2s"], data["a3s"], data["a4s"])
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_C:
            data = _ZabranskyDictChoser(T, self.ZABRANSKY_QUASIPOLYNOMIAL_C_data)
            Cp = Zabransky_quasi_polynomial(T, data["Tc"], data["a1p"], data["a2p"], data["a3p"], data["a4p"], data["a5p"], data["a6p"])
        elif method == ZABRANSKY_SPLINE_SAT:
            data = _ZabranskyDictChoser(T, self.ZABRANSKY_SPLINE_SAT_data)
            Cp = Zabransky_cubic(T, data["a1s"], data["a2s"], data["a3s"], data["a4s"])
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_SAT:
            data = _ZabranskyDictChoser(T, self.ZABRANSKY_QUASIPOLYNOMIAL_SAT_data)
            Cp = Zabransky_quasi_polynomial(T, data["Tc"], data["a1p"], data["a2p"], data["a3p"], data["a4p"], data["a5p"], data["a6p"])
        elif method == COOLPROP:
            Cp = CoolProp_T_dependent_property(T, self.CASRN , 'CPMOLAR', 'l')
        elif method == POLING_CONST:
            Cp = self.POLING_constant
        elif method == CRCSTD:
            Cp = self.CRCSTD_constant
        elif method == ROWLINSON_POLING:
            Cpgm = self.Cpgm(T) if hasattr(self.Cpgm, '__call__') else self.Cpgm
            Cp = Rowlinson_Poling(T, self.Tc, self.omega, Cpgm)
        elif method == ROWLINSON_BONDI:
            Cpgm = self.Cpgm(T) if hasattr(self.Cpgm, '__call__') else self.Cpgm
            Cp = Rowlinson_Bondi(T, self.Tc, self.omega, Cpgm)
        elif method == DADGOSTAR_SHAW:
            Cp = Dadgostar_Shaw(T, self.similarity_variable)
            Cp = property_mass_to_molar(Cp, self.MW)
        elif method in self.tabular_data:
            Cp = self.interpolate(T, method)
        return Cp


    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For the CSP method
        :obj:`Rowlinson_Poling`, the model is considered valid for all
        temperatures. The simple method :obj:`Dadgostar_Shaw` is considered
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
        if method in [ZABRANSKY_SPLINE, ZABRANSKY_QUASIPOLYNOMIAL,
                      ZABRANSKY_SPLINE_C, ZABRANSKY_QUASIPOLYNOMIAL_C,
                      ZABRANSKY_SPLINE_SAT, ZABRANSKY_QUASIPOLYNOMIAL_SAT]:
            data = _ZabranskyDictChoser(T, ZABRANSKY_TO_DICT[method][self.CASRN])
            Tmin, Tmax = data['Tmin'], data['Tmax']
            if T < Tmin or T > Tmax:
                validity = False
        elif method == COOLPROP:
            if T <= self.CP_f.Tt or T >= self.CP_f.Tc:
                validity = False
        elif method == POLING_CONST:
            if T > self.POLING_T + 50 or T < self.POLING_T - 50:
                validity = False
        elif method == CRCSTD:
            if T > self.CRCSTD_T + 50 or T < self.CRCSTD_T - 50:
                validity = False
        elif method == DADGOSTAR_SHAW:
            pass # Valid everywhere
        elif method in [ROWLINSON_POLING, ROWLINSON_BONDI]:
            pass # No limit here
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity



### Solid

def Lastovka_solid(T, similarity_variable):
    r'''Calculate solid constant-pressure heat capacitiy with the similarity
    variable concept and method as shown in [1]_.

    .. math::
        C_p = 3(A_1\alpha + A_2\alpha^2)R\left(\frac{\theta}{T}\right)^2
        \frac{\exp(\theta/T)}{[\exp(\theta/T)-1]^2}
        + (C_1\alpha + C_2\alpha^2)T + (D_1\alpha + D_2\alpha^2)T^2

    Parameters
    ----------
    T : float
        Temperature of solid [K]
    similarity_variable : float
        similarity variable as defined in [1]_, [mol/g]

    Returns
    -------
    Cps : float
        Solid constant-pressure heat capacitiy, [J/kg/K]

    Notes
    -----
    Many restrictions on its use. Trained on data with MW from 12.24 g/mol
    to 402.4 g/mol, C mass fractions from 61.3% to 95.2%,
    H mass fractions from 3.73% to 15.2%, N mass fractions from 0 to 15.4%,
    O mass fractions from 0 to 18.8%, and S mass fractions from 0 to 29.6%.
    Recommended for organic compounds with low mass fractions of hetero-atoms
    and especially when molar mass exceeds 200 g/mol. This model does not show
    and effects of phase transition but should not be used passed the triple
    point.

    Original model is in terms of J/g/K. Note that the model s for predicting
    mass heat capacity, not molar heat capacity like most other methods!

    A1 = 0.013183; A2 = 0.249381; theta = 151.8675; C1 = 0.026526;
    C2 = -0.024942; D1 = 0.000025; D2 = -0.000123.

    Examples
    --------
    >>> Lastovka_solid(300, 0.2139)
    1682.063629222013

    References
    ----------
    .. [1] Laštovka, Václav, Michal Fulem, Mildred Becerra, and John M. Shaw.
       "A Similarity Variable for Estimating the Heat Capacity of Solid Organic
       Compounds: Part II. Application: Heat Capacity Calculation for
       Ill-Defined Organic Solids." Fluid Phase Equilibria 268, no. 1-2
       (June 25, 2008): 134-41. doi:10.1016/j.fluid.2008.03.018.
    '''
    A1 = 0.013183
    A2 = 0.249381
    theta = 151.8675
    C1 = 0.026526
    C2 = -0.024942
    D1 = 0.000025
    D2 = -0.000123

    Cp = (3*(A1*similarity_variable + A2*similarity_variable**2)*R*(theta/T
    )**2*exp(theta/T)/(exp(theta/T)-1)**2
    + (C1*similarity_variable + C2*similarity_variable**2)*T
    + (D1*similarity_variable + D2*similarity_variable**2)*T**2)
    Cp = Cp*1000 # J/g/K to J/kg/K
    return Cp


LASTOVKA_S = 'Lastovka, Fulem, Becerra and Shaw (2008)'
PERRY151 = '''Perry's Table 2-151'''
heat_capacity_solid_methods = [PERRY151, CRCSTD, LASTOVKA_S]
'''Holds all methods available for the HeatCapacitySolid class, for use in
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
        :obj:`Lastovka_solid` for details.

    See Also
    --------
    Lastovka_solid

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

    def __init__(self, CASRN='', similarity_variable=None, MW=None):
        self.similarity_variable = similarity_variable
        self.MW = MW
        self.CASRN = CASRN

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        heat capacity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        heat capacity above.'''

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
        if self.CASRN and self.CASRN in _PerryI and 'c' in _PerryI[self.CASRN]:
            self.PERRY151_Tmin = _PerryI[self.CASRN]['c']['Tmin']
            self.PERRY151_Tmax = _PerryI[self.CASRN]['c']['Tmax']
            self.PERRY151_const = _PerryI[self.CASRN]['c']['Const']
            self.PERRY151_lin = _PerryI[self.CASRN]['c']['Lin']
            self.PERRY151_quad = _PerryI[self.CASRN]['c']['Quad']
            self.PERRY151_quadinv = _PerryI[self.CASRN]['c']['Quadinv']
            methods.append(PERRY151)
            Tmins.append(self.PERRY151_Tmin); Tmaxs.append(self.PERRY151_Tmax)
        if self.CASRN in CRC_standard_data.index and not np.isnan(CRC_standard_data.at[self.CASRN, 'Cpc']):
            self.CRCSTD_Cp = float(CRC_standard_data.at[self.CASRN, 'Cpc'])
            methods.append(CRCSTD)
        if self.MW and self.similarity_variable:
            methods.append(LASTOVKA_S)
            Tmins.append(1.0); Tmaxs.append(10000)
            # Works above roughly 1 K up to 10K.
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)


    def calculate(self, T, method):
        r'''Method to calculate heat capacity of a solid at temperature `T`
        with a given method.

        This method has no exception handling; see `T_dependent_property`
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
        if method == PERRY151:
            Cp = (self.PERRY151_const + self.PERRY151_lin*T
            + self.PERRY151_quadinv/T**2 + self.PERRY151_quad*T**2)*calorie
        elif method == CRCSTD:
            Cp = self.CRCSTD_Cp
        elif method == LASTOVKA_S:
            Cp = Lastovka_solid(T, self.similarity_variable)
            Cp = property_mass_to_molar(Cp, self.MW)
        elif method in self.tabular_data:
            Cp = self.interpolate(T, method)
        return Cp


    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For tabular data,
        extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the
        extrapolation is considered valid for all temperatures.
        For the :obj:`Lastovka_solid` method, it is considered valid under
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
        else:
            raise Exception('Method not valid')
        return validity



### Mixture heat capacities

def Cp_liq_mixture(zs=None, ws=None, Cps=None, T=None, CASRNs=None, AvailableMethods=False, Method=None):  # pragma: no cover
    '''This function handles the retrival of a mixture's liquid heat capacity.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> Cp_liq_mixture(ws=[0.6, 0.3, 0.1], Cps=[4180.59, 2532.45, 2268.80])
    3494.969
    '''
    def list_methods():
        methods = []
        if CASRNs and len(CASRNs) > 1 and '7732-18-5' in CASRNs and T and ws:
            wCASRNs = list(CASRNs)
            wCASRNs.remove('7732-18-5')
            if all([i in _Laliberte_Heat_Capacity_ParametersDict for i in wCASRNs]):
                methods.append('Laliberte')
        if none_and_length_check([Cps]):
            methods.append('Simple')
        methods.append('None')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if not none_and_length_check([Cps, ws]): # check same-length inputs
#        raise Exception('Function inputs are incorrect format')
        return None
    if Method == 'Simple':
        _cp = mixing_simple(ws, Cps)
    elif Method == 'Laliberte':
        ws = list(ws)
        ws.remove(ws[CASRNs.index('7732-18-5')])
        wCASRNs = list(CASRNs)
        wCASRNs.remove('7732-18-5')
        _cp = Laliberte_heat_capacity(T, ws, wCASRNs)
    elif Method == 'None':
        return None
    else:
        raise Exception('Failure in in function')
    return _cp


def Cp_gas_mixture(zs=None, ws=None, Cps=None, CASRNs=None, AvailableMethods=False, Method=None):  # pragma: no cover
    '''This function handles the retrival of a mixture's gas heat capacity.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> Cp_gas_mixture(ws=[0.6, 0.3, 0.1], Cps=[1864.17, 1375.76, 1654.71])
    1696.701
    '''
    def list_methods():
        methods = []
        if none_and_length_check([Cps]):
            methods.append('Simple')
        methods.append('None')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if not none_and_length_check([Cps, ws]): # check same-length inputs
        return None
#        raise Exception('Function inputs are incorrect format')
    if Method == 'Simple':
        _cp = mixing_simple(ws, Cps)
    elif Method == 'None':
        return None
    else:
        raise Exception('Failure in in function')
    return _cp


def Cv_gas_mixture(zs=None,  ws=None, Cps=None, CASRNs=None, AvailableMethods=False, Method=None):  # pragma: no cover
    '''This function handles the retrival of a mixture's gas constant
    volume heat capacity.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

   >>> Cv_gas_mixture(ws=[0.6, 0.3, 0.1], Cps=[1402.64, 1116.27, 1558.23])
    1332.2880000000002
    '''
    def list_methods():
        methods = []
        if none_and_length_check([Cps]):
            methods.append('Simple')
        methods.append('None')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if not none_and_length_check((Cps, ws)): # check same-length inputs
        return None
#        raise Exception('Function inputs are incorrect format')
    if Method == 'Simple':
        _cp = mixing_simple(ws, Cps)
    elif Method == 'None':
        return None
    else:
        raise Exception('Failure in in function')
    return _cp



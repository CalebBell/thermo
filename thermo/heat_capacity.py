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

__all__ = ['Poling_data', 'TRC_gas_data', '_PerryI', 'CRC_standard_data', 
           'Lastovka_Shaw', 'Lastovka_Shaw_integral', 
           'Lastovka_Shaw_integral_over_T', 'TRCCp', 
           'TRCCp_integral', 'TRCCp_integral_over_T', 
           'heat_capacity_gas_methods', 'HeatCapacityGas', 
           'Rowlinson_Poling', 'Rowlinson_Bondi', 'Dadgostar_Shaw', 
           'Zabransky_quasi_polynomial', 'Zabransky_quasi_polynomial_integral',
           'Zabransky_quasi_polynomial_integral_over_T', 'Zabransky_cubic', 
           'Zabransky_cubic_integral', 'Zabransky_cubic_integral_over_T',
           'Zabransky_quasipolynomial', 'Zabransky_spline',
           'ZABRANSKY_TO_DICT', 'heat_capacity_liquid_methods', 
           'HeatCapacityLiquid', 'Lastovka_solid', 'Lastovka_solid_integral', 
           'Lastovka_solid_integral_over_T', 'heat_capacity_solid_methods', 
           'HeatCapacitySolid', 'HeatCapacitySolidMixture', 
           'HeatCapacityGasMixture', 'HeatCapacityLiquidMixture']
import os
from io import open
from thermo.utils import log, exp, polylog2
import numpy as np
import pandas as pd

from scipy.integrate import quad
from thermo.utils import R, calorie
from thermo.utils import (to_num, property_molar_to_mass, none_and_length_check,
                          mixing_simple, property_mass_to_molar)
from thermo.miscdata import _VDISaturationDict, VDI_tabular_data
from thermo.electrochem import (Laliberte_heat_capacity,
                                _Laliberte_Heat_Capacity_ParametersDict)
from thermo.utils import TDependentProperty, MixtureProperty
from thermo.coolprop import *

                         
                         
                         
folder = os.path.join(os.path.dirname(__file__), 'Heat Capacity')


Poling_data = pd.read_csv(os.path.join(folder,
                       'PolingDatabank.tsv'), sep='\t',
                       index_col=0)
_Poling_data_values = Poling_data.values


TRC_gas_data = pd.read_csv(os.path.join(folder,
                       'TRC Thermodynamics of Organic Compounds in the Gas State.tsv'), sep='\t',
                       index_col=0)
_TRC_gas_data_values = TRC_gas_data.values



_PerryI = {}
with open(os.path.join(folder, 'Perrys Table 2-151.tsv'), encoding='utf-8') as f:
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
                       'CRC Standard Thermodynamic Properties of Chemical Substances.tsv'), sep='\t',
                       index_col=0)



### Heat capacities of gases

def Lastovka_Shaw(T, similarity_variable, cyclic_aliphatic=False):
    r'''Calculate ideal-gas constant-pressure heat capacitiy with the similarity
    variable concept and method as shown in [1]_.

    .. math::
        C_p^0 = \left(A_2 + \frac{A_1 - A_2}{1 + \exp(\frac{\alpha-A_3}{A_4})}\right)
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
        first = A2 + (A1-A2)/(1. + exp((a - A3)/A4))
        # Personal communication confirms the change

    B11 = 0.73917383
    B12 = 8.88308889
    C11 = 1188.28051
    C12 = 1813.04613
    B21 = 0.0483019
    B22 = 4.35656721
    C21 = 2897.01927
    C22 = 5987.80407
    Cp = first + (B11 + B12*a)*((C11+C12*a)/T)**2*exp(-(C11 + C12*a)/T)/(1.-exp(-(C11+C12*a)/T))**2
    Cp += (B21 + B22*a)*((C21+C22*a)/T)**2*exp(-(C21 + C22*a)/T)/(1.-exp(-(C21+C22*a)/T))**2
    return Cp*1000. # J/g/K to J/kg/K


def Lastovka_Shaw_integral(T, similarity_variable, cyclic_aliphatic=False):
    r'''Calculate the integral of ideal-gas constant-pressure heat capacitiy 
    with the similarity variable concept and method as shown in [1]_.

    Parameters
    ----------
    T : float
        Temperature of gas [K]
    similarity_variable : float
        similarity variable as defined in [1]_, [mol/g]

    Returns
    -------
    H : float
        Difference in enthalpy from 0 K, [J/kg]

    Notes
    -----
    Original model is in terms of J/g/K. Note that the model is for predicting
    mass heat capacity, not molar heat capacity like most other methods!
    Integral was computed with SymPy.

    See Also
    --------
    Lastovka_Shaw
    Lastovka_Shaw_integral_over_T

    Examples
    --------
    >>> Lastovka_Shaw_integral(300.0, 0.1333)
    5283095.816018478

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
        first = A2 + (A1-A2)/(1.+exp((a-A3)/A4)) # One reference says exp((a-A3)/A4)
        # Personal communication confirms the change

    B11 = 0.73917383
    B12 = 8.88308889
    C11 = 1188.28051
    C12 = 1813.04613
    B21 = 0.0483019
    B22 = 4.35656721
    C21 = 2897.01927
    C22 = 5987.80407
    return 1000.*(T*first - (B11 + B12*a)*(-C11 - C12*a)**2/(-C11 - C12*a + (C11 
    + C12*a)*exp((-C11 - C12*a)/T)) - (B21 + B22*a)*(-C21 - C22*a)**2/(-C21 
    - C22*a + (C21 + C22*a)*exp((-C21 - C22*a)/T)))


def Lastovka_Shaw_integral_over_T(T, similarity_variable, cyclic_aliphatic=False):
    r'''Calculate the integral over temperature of ideal-gas constant-pressure 
    heat capacitiy with the similarity variable concept and method as shown in
    [1]_.

    Parameters
    ----------
    T : float
        Temperature of gas [K]
    similarity_variable : float
        similarity variable as defined in [1]_, [mol/g]

    Returns
    -------
    S : float
        Difference in entropy from 0 K, [J/kg/K]

    Notes
    -----
    Original model is in terms of J/g/K. Note that the model is for predicting
    mass heat capacity, not molar heat capacity like most other methods!
    Integral was computed with SymPy.

    See Also
    --------
    Lastovka_Shaw
    Lastovka_Shaw_integral

    Examples
    --------
    >>> Lastovka_Shaw_integral_over_T(300.0, 0.1333)
    3609.791928945323

    References
    ----------
    .. [1] Lastovka, Vaclav, and John M. Shaw. "Predictive Correlations for
       Ideal Gas Heat Capacities of Pure Hydrocarbons and Petroleum Fractions."
       Fluid Phase Equilibria 356 (October 25, 2013): 338-370.
       doi:10.1016/j.fluid.2013.07.023.
    '''
    from cmath import log, exp
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
        first = A2 + (A1-A2)/(1. + exp((a - A3)/A4))

    a2 = a*a
    B11 = 0.73917383
    B12 = 8.88308889
    C11 = 1188.28051
    C12 = 1813.04613
    B21 = 0.0483019
    B22 = 4.35656721
    C21 = 2897.01927
    C22 = 5987.80407
    S = (first*log(T) + (-B11 - B12*a)*log(exp((-C11 - C12*a)/T) - 1.) 
        + (-B11*C11 - B11*C12*a - B12*C11*a - B12*C12*a2)/(T*exp((-C11
        - C12*a)/T) - T) - (B11*C11 + B11*C12*a + B12*C11*a + B12*C12*a2)/T)
    S += ((-B21 - B22*a)*log(exp((-C21 - C22*a)/T) - 1.) + (-B21*C21 - B21*C22*a
        - B22*C21*a - B22*C22*a2)/(T*exp((-C21 - C22*a)/T) - T) - (B21*C21
        + B21*C22*a + B22*C21*a + B22*C22*a**2)/T)
    # There is a non-real component, but it is only a function of similariy 
    # variable and so will always cancel out.
    return S.real*1000.


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
    if T <= a7:
        y = 0.
    else:
        y = (T - a7)/(T + a6)
    Cp = R*(a0 + (a1/T**2)*exp(-a2/T) + a3*y**2 + (a4 - a5/(T-a7)**2 )*y**8.)
    return Cp


def TRCCp_integral(T, a0, a1, a2, a3, a4, a5, a6, a7, I=0):
    r'''Integrates ideal gas heat capacity using the model developed in [1]_.
    Best used as a delta only.

    The difference in enthalpy with respect to 0 K is given by:

    .. math::
        \frac{H(T) - H^{ref}}{RT} = a_0 + a_1x(a_2)/(a_2T) + I/T + h(T)/T
        
        h(T) = (a_5 + a_7)\left[(2a_3 + 8a_4)\ln(1-y)+ \left\{a_3\left(1 + 
        \frac{1}{1-y}\right) + a_4\left(7 + \frac{1}{1-y}\right)\right\}y
        + a_4\left\{3y^2 + (5/3)y^3 + y^4 + (3/5)y^5 + (1/3)y^6\right\} 
        + (1/7)\left\{a_4 - \frac{a_5}{(a_6+a_7)^2}\right\}y^7\right]
        
        h(T) = 0 \text{ for } T \le a_7

        y = \frac{T-a_7}{T+a_6} \text{ for } T > a_7 \text{ otherwise } 0

    Parameters
    ----------
    T : float
        Temperature [K]
    a1-a7 : float
        Coefficients
    I : float, optional
        Integral offset

    Returns
    -------
    H-H(0) : float
        Difference in enthalpy from 0 K , [J/mol]

    Notes
    -----
    Analytical integral as provided in [1]_ and verified with numerical
    integration. 

    Examples
    --------
    >>> TRCCp_integral(298.15, 4.0, 7.65E5, 720., 3.565, -0.052, -1.55E6, 52., 
    ... 201., 1.2)
    10802.532600592816
    
    References
    ----------
    .. [1] Kabo, G. J., and G. N. Roganov. Thermodynamics of Organic Compounds
       in the Gas State, Volume II: V. 2. College Station, Tex: CRC Press, 1994.
    '''
    if T <= a7:
        y = 0.
    else:
        y = (T - a7)/(T + a6)
    y2 = y*y
    y4 = y2*y2
    if T <= a7:
        h = 0.0
    else:
        first = a6 + a7
        second = (2.*a3 + 8.*a4)*log(1. - y)
        third = (a3*(1. + 1./(1. - y)) + a4*(7. + 1./(1. - y)))*y
        fourth = a4*(3.*y2 + 5./3.*y*y2 + y4 + 0.6*y4*y + 1/3.*y4*y2)
        fifth = 1/7.*(a4 - a5/((a6 + a7)**2))*y4*y2*y
        h = first*(second + third + fourth + fifth)
    return (a0 + a1*exp(-a2/T)/(a2*T) + I/T + h/T)*R*T


def TRCCp_integral_over_T(T, a0, a1, a2, a3, a4, a5, a6, a7, J=0):
    r'''Integrates ideal gas heat capacity over T using the model developed in 
    [1]_. Best used as a delta only.

    The difference in ideal-gas entropy with respect to 0 K is given by:

    .. math::
        \frac{S^\circ}{R} = J + a_0\ln T + \frac{a_1}{a_2^2}\left(1
        + \frac{a_2}{T}\right)x(a_2) + s(T)

        s(T) = \left[\left\{a_3 + \left(\frac{a_4 a_7^2 - a_5}{a_6^2}\right)
        \left(\frac{a_7}{a_6}\right)^4\right\}\left(\frac{a_7}{a_6}\right)^2
        \ln z + (a_3 + a_4)\ln\left(\frac{T+a_6}{a_6+a_7}\right)
        +\sum_{i=1}^7 \left\{\left(\frac{a_4 a_7^2 - a_5}{a_6^2}\right)\left(
        \frac{-a_7}{a_6}\right)^{6-i} - a_4\right\}\frac{y^i}{i}
        - \left\{\frac{a_3}{a_6}(a_6 + a_7) + \frac{a_5 y^6}{7a_7(a_6+a_7)}
        \right\}y\right]

        s(T) = 0 \text{ for } T \le a_7
        
        z = \frac{T}{T+a_6} \cdot \frac{a_7 + a_6}{a_7}

        y = \frac{T-a_7}{T+a_6} \text{ for } T > a_7 \text{ otherwise } 0

    Parameters
    ----------
    T : float
        Temperature [K]
    a1-a7 : float
        Coefficients
    J : float, optional
        Integral offset

    Returns
    -------
    S-S(0) : float
        Difference in entropy from 0 K , [J/mol/K]

    Notes
    -----
    Analytical integral as provided in [1]_ and verified with numerical
    integration. 

    Examples
    --------
    >>> TRCCp_integral_over_T(300, 4.0, 124000, 245, 50.539, -49.469, 
    ... 220440000, 560, 78)
    213.80148972435018
    
    References
    ----------
    .. [1] Kabo, G. J., and G. N. Roganov. Thermodynamics of Organic Compounds
       in the Gas State, Volume II: V. 2. College Station, Tex: CRC Press, 1994.
    '''
    # Possible optimizations: pre-cache as much as possible.
    # If this were replaced by a cache, much of this would not need to be computed.
    if T <= a7:
        y = 0.
    else:
        y = (T - a7)/(T + a6)

    z = T/(T + a6)*(a7 + a6)/a7
    if T <= a7:
        s = 0.
    else:
        a72 = a7*a7
        a62 = a6*a6
        a7_a6 = a7/a6 # a7/a6
        a7_a6_2 = a7_a6*a7_a6
        a7_a6_4 = a7_a6_2*a7_a6_2
        x1 = (a4*a72 - a5)/a62 # part of third, sum
        first = (a3 + ((a4*a72 - a5)/a62)*a7_a6_4)*a7_a6_2*log(z)
        second = (a3 + a4)*log((T + a6)/(a6 + a7))
        fourth = -(a3/a6*(a6 + a7) + a5*y**6/(7.*a7*(a6 + a7)))*y
        third = sum([(x1*(-a7_a6)**(6-i) - a4)*y**i/i for i in range(1, 8)])
        s = first + second + third + fourth
    return R*(J + a0*log(T) + a1/(a2*a2)*(1. + a2/T)*exp(-a2/T) + s)
    
    
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
    '''Maximum valid of Heat capacity; arbitrarily set. For fluids very near
    the critical point, this value can be obscenely high.'''

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
            Cp = TRCCp(T, *self.TRCIG_coefs)
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
        if method == TRCIG:
            H2 = TRCCp_integral(T2, *self.TRCIG_coefs)
            H1 = TRCCp_integral(T1, *self.TRCIG_coefs)
            return H2 - H1
        elif method == POLING:
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
        elif method in self.tabular_data or method == COOLPROP:
            return float(quad(self.calculate, T1, T2, args=(method))[0])
        else:
            raise Exception('Method not valid')


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
        if method == TRCIG:
            S2 = TRCCp_integral_over_T(T2, *self.TRCIG_coefs)
            S1 = TRCCp_integral_over_T(T1, *self.TRCIG_coefs)
            return S2 - S1
        elif method == CRCSTD:
            return self.CRCSTD_constant*log(T2/T1)
        elif method == POLING_CONST:
            return self.POLING_constant*log(T2/T1)
        elif method == POLING:
            A, B, C, D, E = self.POLING_coefs
            S2 = ((((0.25*E)*T2 + D/3.)*T2 + 0.5*C)*T2 + B)*T2 
            S1 = ((((0.25*E)*T1 + D/3.)*T1 + 0.5*C)*T1 + B)*T1
            return R*(S2-S1 + A*log(T2/T1))
        elif method == LASTOVKA_SHAW:
            dS = (Lastovka_Shaw_integral_over_T(T2, self.similarity_variable)
                 - Lastovka_Shaw_integral_over_T(T1, self.similarity_variable))
            return property_mass_to_molar(dS, self.MW)
        elif method in self.tabular_data or method == COOLPROP:
            return float(quad(lambda T: self.calculate(T, method)/T, T1, T2)[0])
        else:
            raise Exception('Method not valid')


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
    a = similarity_variable
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

    Cp = (constant*(a11*a + a12*a**2) + (a21*a + a22*a**2)*T
          + (a31*a + a32*a**2)*T**2)
    Cp = Cp*1000 # J/g/K to J/kg/K
    return Cp


def Dadgostar_Shaw_integral(T, similarity_variable):
    r'''Calculate the integral of liquid constant-pressure heat capacitiy 
    with the similarity variable concept and method as shown in [1]_.

    Parameters
    ----------
    T : float
        Temperature of gas [K]
    similarity_variable : float
        similarity variable as defined in [1]_, [mol/g]

    Returns
    -------
    H : float
        Difference in enthalpy from 0 K, [J/kg]

    Notes
    -----
    Original model is in terms of J/g/K. Note that the model is for predicting
    mass heat capacity, not molar heat capacity like most other methods!
    Integral was computed with SymPy.

    See Also
    --------
    Dadgostar_Shaw
    Dadgostar_Shaw_integral_over_T

    Examples
    --------
    >>> Dadgostar_Shaw_integral(300.0, 0.1333)
    238908.15142664989

    References
    ----------
    .. [1] Dadgostar, Nafiseh, and John M. Shaw. "A Predictive Correlation for
       the Constant-Pressure Specific Heat Capacity of Pure and Ill-Defined
       Liquid Hydrocarbons." Fluid Phase Equilibria 313 (January 15, 2012):
       211-226. doi:10.1016/j.fluid.2011.09.015.
    '''
    a = similarity_variable
    a2 = a*a
    T2 = T*T
    a11 = -0.3416
    a12 = 2.2671
    a21 = 0.1064
    a22 = -0.3874
    a31 = -9.8231E-05
    a32 = 4.182E-04
    constant = 24.5
    H = T2*T/3.*(a2*a32 + a*a31) + T2*0.5*(a2*a22 + a*a21) + T*constant*(a2*a12 + a*a11)
    return H*1000. # J/g/K to J/kg/K


def Dadgostar_Shaw_integral_over_T(T, similarity_variable):
    r'''Calculate the integral of liquid constant-pressure heat capacitiy 
    with the similarity variable concept and method as shown in [1]_.

    Parameters
    ----------
    T : float
        Temperature of gas [K]
    similarity_variable : float
        similarity variable as defined in [1]_, [mol/g]

    Returns
    -------
    S : float
        Difference in entropy from 0 K, [J/kg/K]

    Notes
    -----
    Original model is in terms of J/g/K. Note that the model is for predicting
    mass heat capacity, not molar heat capacity like most other methods!
    Integral was computed with SymPy.

    See Also
    --------
    Dadgostar_Shaw
    Dadgostar_Shaw_integral

    Examples
    --------
    >>> Dadgostar_Shaw_integral_over_T(300.0, 0.1333)
    1201.1409113147927

    References
    ----------
    .. [1] Dadgostar, Nafiseh, and John M. Shaw. "A Predictive Correlation for
       the Constant-Pressure Specific Heat Capacity of Pure and Ill-Defined
       Liquid Hydrocarbons." Fluid Phase Equilibria 313 (January 15, 2012):
       211-226. doi:10.1016/j.fluid.2011.09.015.
    '''
    a = similarity_variable
    a2 = a*a
    a11 = -0.3416
    a12 = 2.2671
    a21 = 0.1064
    a22 = -0.3874
    a31 = -9.8231E-05
    a32 = 4.182E-04
    constant = 24.5
    S = T*T*0.5*(a2*a32 + a*a31) + T*(a2*a22 + a*a21) + a*constant*(a*a12 + a11)*log(T)
    return S*1000. # J/g/K to J/kg/K


class Zabransky_quasipolynomial(object):
    r'''Quasi-polynomial object for calculating the heat capacity of a chemical.
    Implements the enthalpy and entropy integrals as well.

    .. math::
        \frac{C}{R}=A_1\ln(1-T_r) + \frac{A_2}{1-T_r}
        + \sum_{j=0}^m A_{j+3} T_r^j

    Parameters
    ----------
    CAS : str
        CAS number.
    name : str
        Name of the chemical as given in [1]_.
    uncertainty : str
        Uncertainty class of the heat capacity as given in [1]_.
    Tmin : float
        Minimum temperature any experimental data was available at.
    Tmax : float
        Maximum temperature any experimental data was available at.
    Tc : float
        Critical temperature of the chemical, as used in the formula.
    coeffs : list[float]
        Six coefficients for the equation.

    References
    ----------
    .. [1] Zabransky, M., V. Ruzicka Jr, V. Majer, and Eugene S. Domalski.
       Heat Capacity of Liquids: Critical Review and Recommended Values.
       2 Volume Set. Washington, D.C.: Amer Inst of Physics, 1996.
    '''
    __slots__ = ['CAS', 'name', 'uncertainty', 'Tmin', 'Tmax', 'Tc', 'coeffs']
    def __init__(self, CAS, name, uncertainty, Tmin, Tmax, Tc, coeffs):
        self.CAS = CAS
        '''CAS number.'''
        self.name = name
        '''Name of the chemical.'''
        self.uncertainty = uncertainty
        '''Uncertainty class of the heat capacity.'''
        self.Tmin = Tmin
        '''Minimum temperature any experimental data was available at.'''
        self.Tmax = Tmax
        '''Maximum temperature any experimental data was available at.'''
        self.Tc = Tc 
        '''Critical temperature of the chemical, as used in the formula.'''
        self.coeffs = coeffs
        '''Six coefficients for the equation.'''

    def calculate(self, T):
        r'''Method to actually calculate heat capacity as a function of 
        temperature.
            
        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        Cp : float
            Liquid heat capacity as T, [J/mol/K]
        '''        
        return Zabransky_quasi_polynomial(T, self.Tc, *self.coeffs)
                                          
    def calculate_integral(self, T1, T2):
        r'''Method to compute the enthalpy integral of heat capacity from 
         `T1` to `T2`.
            
        Parameters
        ----------
        T1 : float
            Initial temperature, [K]
        T2 : float
            Final temperature, [K]
            
        Returns
        -------
        dH : float
            Enthalpy difference between `T1` and `T2`, [J/mol]
        '''        
        return (Zabransky_quasi_polynomial_integral(T2, self.Tc, *self.coeffs)
               - Zabransky_quasi_polynomial_integral(T1, self.Tc, *self.coeffs))
    
    def calculate_integral_over_T(self, T1, T2):
        r'''Method to compute the entropy integral of heat capacity from 
         `T1` to `T2`.
            
        Parameters
        ----------
        T1 : float
            Initial temperature, [K]
        T2 : float
            Final temperature, [K]
            
        Returns
        -------
        dS : float
            Entropy difference between `T1` and `T2`, [J/mol/K]
        '''        
        return (Zabransky_quasi_polynomial_integral_over_T(T2, self.Tc, *self.coeffs)
               - Zabransky_quasi_polynomial_integral_over_T(T1, self.Tc, *self.coeffs))

        
class Zabransky_spline(object):
    r'''Implementation of the cubic spline method presented in [1]_ for 
    calculating the heat capacity of a chemical.
    Implements the enthalpy and entropy integrals as well.

    .. math::
        \frac{C}{R}=\sum_{j=0}^3 A_{j+1} \left(\frac{T}{100}\right)^j

    Parameters
    ----------
    CAS : str
        CAS number.
    name : str
        Name of the chemical as in [1]_.
    uncertainty : str
        Uncertainty class of the heat capacity as in [1]_.

        References
    ----------
    .. [1] Zabransky, M., V. Ruzicka Jr, V. Majer, and Eugene S. Domalski.
       Heat Capacity of Liquids: Critical Review and Recommended Values.
       2 Volume Set. Washington, D.C.: Amer Inst of Physics, 1996.
    '''
    __slots__ = ['Ts', 'coeff_sets', 'n', 'CAS', 'name', 'uncertainty']
    def __init__(self, CAS, name, uncertainty):
        self.CAS = CAS
        '''CAS number.'''
        self.name = name
        '''Name of the chemical.'''
        self.uncertainty = uncertainty
        '''Uncertainty class of the heat capacity.'''
        self.Ts = []
        '''Temperatures at which the coefficient sets transition.'''
        self.coeff_sets = []
        '''Actual coefficients used to describe the chemical.'''
        self.n = 0
        '''Number of coefficient sets used to describe the chemical.'''
        
    def add_coeffs(self, Tmin, Tmax, coeffs):
        '''Called internally during the parsing of the Zabransky database, to
        add coefficients as they are read one per line'''
        self.n += 1
        if not self.Ts:
            self.Ts = [Tmin, Tmax]
            self.coeff_sets = [coeffs]
        else:
            for ind, T in enumerate(self.Ts):
                if Tmin < T:
                    # Under an existing coefficient set - assume Tmax will come from another set
                    self.Ts.insert(ind, Tmin) 
                    self.coeff_sets.insert(ind, coeffs)
                    return
            # Must be appended to end instead
            self.Ts.append(Tmax)
            self.coeff_sets.append(coeffs)
       
    def _coeff_ind_from_T(self, T):
        '''Determines the index at which the coefficients for the current
        temperature are stored in `coeff_sets`.
        '''
        # DO NOT CHANGE
        if self.n == 1:
            return 0
        for i in range(self.n):
            if T <= self.Ts[i+1]:
                return i
        return self.n - 1

    def calculate(self, T):
        r'''Method to actually calculate heat capacity as a function of 
        temperature.
            
        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        Cp : float
            Liquid heat capacity as T, [J/mol/K]
        '''        
        return Zabransky_cubic(T, *self.coeff_sets[self._coeff_ind_from_T(T)])

    def calculate_integral(self, T1, T2):
        r'''Method to compute the enthalpy integral of heat capacity from 
        `T1` to `T2`. Analytically integrates across the piecewise spline
        as necessary.
            
        Parameters
        ----------
        T1 : float
            Initial temperature, [K]
        T2 : float
            Final temperature, [K]
            
        Returns
        -------
        dS : float
            Enthalpy difference between `T1` and `T2`, [J/mol/K]
        '''        
        # Simplify the problem so we can assume T2 >= T1
        if T2 < T1:
            flipped = True
            T1, T2 = T2, T1
        else:
            flipped = False
        
        # Fastest case - only one coefficient set, occurs surprisingly often
        if self.n == 1:
            dH = (Zabransky_cubic_integral(T2, *self.coeff_sets[0])
                  - Zabransky_cubic_integral(T1, *self.coeff_sets[0]))
        else:
            ind_T1, ind_T2 = self._coeff_ind_from_T(T1), self._coeff_ind_from_T(T2)
            # Second fastest case - both are in the same coefficient set
            if ind_T1 == ind_T2:
                dH = (Zabransky_cubic_integral(T2, *self.coeff_sets[ind_T2])
                        - Zabransky_cubic_integral(T1, *self.coeff_sets[ind_T1]))
            # Fo through the loop if we need to - inevitably slow 
            else:
                dH = (Zabransky_cubic_integral(self.Ts[ind_T1], *self.coeff_sets[ind_T1])
                      - Zabransky_cubic_integral(T1, *self.coeff_sets[ind_T1]))
                for i in range(ind_T1, ind_T2):
                    diff =(Zabransky_cubic_integral(self.Ts[i+1], *self.coeff_sets[i])
                          - Zabransky_cubic_integral(self.Ts[i], *self.coeff_sets[i]))
                    dH += diff
                end = (Zabransky_cubic_integral(T2, *self.coeff_sets[ind_T2])
                      - Zabransky_cubic_integral(self.Ts[ind_T2], *self.coeff_sets[ind_T2]))
                dH += end
        return -dH if flipped else dH

    def calculate_integral_over_T(self, T1, T2):
        r'''Method to compute the entropy integral of heat capacity from 
        `T1` to `T2`. Analytically integrates across the piecewise spline
        as necessary.
            
        Parameters
        ----------
        T1 : float
            Initial temperature, [K]
        T2 : float
            Final temperature, [K]
            
        Returns
        -------
        dS : float
            Entropy difference between `T1` and `T2`, [J/mol/K]
        '''        
        # Simplify the problem so we can assume T2 >= T1
        if T2 < T1:
            flipped = True
            T1, T2 = T2, T1
        else:
            flipped = False
        
        # Fastest case - only one coefficient set, occurs surprisingly often
        if self.n == 1:
            dS = (Zabransky_cubic_integral_over_T(T2, *self.coeff_sets[0])
                  - Zabransky_cubic_integral_over_T(T1, *self.coeff_sets[0]))
        else:
            ind_T1, ind_T2 = self._coeff_ind_from_T(T1), self._coeff_ind_from_T(T2)
            # Second fastest case - both are in the same coefficient set
            if ind_T1 == ind_T2:
                dS = (Zabransky_cubic_integral_over_T(T2, *self.coeff_sets[ind_T2])
                        - Zabransky_cubic_integral_over_T(T1, *self.coeff_sets[ind_T1]))
            # Fo through the loop if we need to - inevitably slow 
            else:
                dS = (Zabransky_cubic_integral_over_T(self.Ts[ind_T1], *self.coeff_sets[ind_T1])
                      - Zabransky_cubic_integral_over_T(T1, *self.coeff_sets[ind_T1]))
                for i in range(ind_T1, ind_T2):
                    diff =(Zabransky_cubic_integral_over_T(self.Ts[i+1], *self.coeff_sets[i])
                          - Zabransky_cubic_integral_over_T(self.Ts[i], *self.coeff_sets[i]))
                    dS += diff
                end = (Zabransky_cubic_integral_over_T(T2, *self.coeff_sets[ind_T2])
                      - Zabransky_cubic_integral_over_T(self.Ts[ind_T2], *self.coeff_sets[ind_T2]))
                dS += end
        return -dS if flipped else dS

zabransky_dict_sat_s = {}
zabransky_dict_sat_p = {}
zabransky_dict_const_s = {}
zabransky_dict_const_p = {}
zabransky_dict_iso_s = {}
zabransky_dict_iso_p = {}

# C means average heat capacity values, from less rigorous experiments
# sat means heat capacity along the saturation line
# p means constant-pressure values, 
type_to_zabransky_dict = {('C', True): zabransky_dict_const_s, 
                       ('C', False):   zabransky_dict_const_p,
                       ('sat', True):  zabransky_dict_sat_s,
                       ('sat', False): zabransky_dict_sat_p,
                       ('p', True):    zabransky_dict_iso_s,
                       ('p', False):   zabransky_dict_iso_p}

                     
with open(os.path.join(folder, 'Zabransky.tsv'), encoding='utf-8') as f:
    next(f)
    for line in f:
        values = to_num(line.strip('\n').split('\t'))
        (CAS, name, Type, uncertainty, Tmin, Tmax, a1s, a2s, a3s, a4s, a1p, a2p, a3p, a4p, a5p, a6p, Tc) = values
        spline = bool(a1s) # False if Quasypolynomial, True if spline
        d = type_to_zabransky_dict[(Type, spline)]
        if spline:
            if CAS not in d:
                d[CAS] = Zabransky_spline(CAS, name, uncertainty)
            d[CAS].add_coeffs(Tmin, Tmax, [a1s, a2s, a3s, a4s])
        else:
            # No duplicates for quasipolynomials
            d[CAS] = Zabransky_quasipolynomial(CAS, name, uncertainty, Tmin, Tmax, 
                                           Tc, [a1p, a2p, a3p, a4p, a5p, a6p])

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


def Zabransky_quasi_polynomial_integral(T, Tc, a1, a2, a3, a4, a5, a6):
    r'''Calculates the integral of liquid heat capacity using the  
    quasi-polynomial model developed in [1]_.

    Parameters
    ----------
    T : float
        Temperature [K]
    a1-a6 : float
        Coefficients

    Returns
    -------
    H : float
        Difference in enthalpy from 0 K, [J/mol]

    Notes
    -----
    The analytical integral was derived with SymPy; it is a simple polynomial
    plus some logarithms.

    Examples
    --------
    >>> H2 = Zabransky_quasi_polynomial_integral(300, 591.79, -3.12743, 
    ... 0.0857315, 13.7282, 1.28971, 6.42297, 4.10989)
    >>> H1 = Zabransky_quasi_polynomial_integral(200, 591.79, -3.12743, 
    ... 0.0857315, 13.7282, 1.28971, 6.42297, 4.10989)
    >>> H2 - H1
    14662.026406892925

    References
    ----------
    .. [1] Zabransky, M., V. Ruzicka Jr, V. Majer, and Eugene S. Domalski.
       Heat Capacity of Liquids: Critical Review and Recommended Values.
       2 Volume Set. Washington, D.C.: Amer Inst of Physics, 1996.
    '''
    Tc2 = Tc*Tc
    Tc3 = Tc2*Tc
    term = T - Tc
    return R*(T*(T*(T*(T*a6/(4.*Tc3) + a5/(3.*Tc2)) + a4/(2.*Tc)) - a1 + a3) 
              + T*a1*log(1. - T/Tc) - 0.5*Tc*(a1 + a2)*log(term*term))


def Zabransky_quasi_polynomial_integral_over_T(T, Tc, a1, a2, a3, a4, a5, a6):
    r'''Calculates the integral of liquid heat capacity over T using the 
    quasi-polynomial model  developed in [1]_.

    Parameters
    ----------
    T : float
        Temperature [K]
    a1-a6 : float
        Coefficients

    Returns
    -------
    S : float
        Difference in entropy from 0 K, [J/mol/K]

    Notes
    -----
    The analytical integral was derived with Sympy. It requires the 
    Polylog(2,x) function, which is unimplemented in SciPy. A very accurate 
    numerical approximation was implemented as :obj:`thermo.utils.polylog2`.
    Relatively slow due to the use of that special function.

    Examples
    --------
    >>> S2 = Zabransky_quasi_polynomial_integral_over_T(300, 591.79, -3.12743, 
    ... 0.0857315, 13.7282, 1.28971, 6.42297, 4.10989)
    >>> S1 = Zabransky_quasi_polynomial_integral_over_T(200, 591.79, -3.12743, 
    ... 0.0857315, 13.7282, 1.28971, 6.42297, 4.10989)
    >>> S2 - S1
    59.16997291893654
    
    References
    ----------
    .. [1] Zabransky, M., V. Ruzicka Jr, V. Majer, and Eugene S. Domalski.
       Heat Capacity of Liquids: Critical Review and Recommended Values.
       2 Volume Set. Washington, D.C.: Amer Inst of Physics, 1996.
    '''
    term = T - Tc
    logT = log(T)
    Tc2 = Tc*Tc
    Tc3 = Tc2*Tc
    return R*(a3*logT -a1*polylog2(T/Tc) - a2*(-logT + 0.5*log(term*term))
              + T*(T*(T*a6/(3.*Tc3) + a5/(2.*Tc2)) + a4/Tc))


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
    75.31462591538556

    References
    ----------
    .. [1] Zabransky, M., V. Ruzicka Jr, V. Majer, and Eugene S. Domalski.
       Heat Capacity of Liquids: Critical Review and Recommended Values.
       2 Volume Set. Washington, D.C.: Amer Inst of Physics, 1996.
    '''
    T = T/100.
    return R*(((a4*T + a3)*T + a2)*T + a1)


def Zabransky_cubic_integral(T, a1, a2, a3, a4):
    r'''Calculates the integral of liquid heat capacity using the model 
    developed in [1]_.

    Parameters
    ----------
    T : float
        Temperature [K]
    a1-a4 : float
        Coefficients

    Returns
    -------
    H : float
        Difference in enthalpy from 0 K, [J/mol]

    Notes
    -----
    The analytical integral was derived with Sympy; it is a simple polynomial.

    Examples
    --------
    >>> Zabransky_cubic_integral(298.15, 20.9634, -10.1344, 2.8253, -0.256738)
    31051.679845520586

    References
    ----------
    .. [1] Zabransky, M., V. Ruzicka Jr, V. Majer, and Eugene S. Domalski.
       Heat Capacity of Liquids: Critical Review and Recommended Values.
       2 Volume Set. Washington, D.C.: Amer Inst of Physics, 1996.
    '''
    T = T/100.
    return 100*R*T*(T*(T*(T*a4*0.25 + a3/3.) + a2*0.5) + a1)


def Zabransky_cubic_integral_over_T(T, a1, a2, a3, a4):
    r'''Calculates the integral of liquid heat capacity over T using the model 
    developed in [1]_.

    Parameters
    ----------
    T : float
        Temperature [K]
    a1-a4 : float
        Coefficients

    Returns
    -------
    S : float
        Difference in entropy from 0 K, [J/mol/K]

    Notes
    -----
    The analytical integral was derived with Sympy; it is a simple polynomial,
    plus a logarithm

    Examples
    --------
    >>> Zabransky_cubic_integral_over_T(298.15, 20.9634, -10.1344, 2.8253, 
    ... -0.256738)
    24.73245695987246

    References
    ----------
    .. [1] Zabransky, M., V. Ruzicka Jr, V. Majer, and Eugene S. Domalski.
       Heat Capacity of Liquids: Critical Review and Recommended Values.
       2 Volume Set. Washington, D.C.: Amer Inst of Physics, 1996.
    '''
    T = T/100.
    return R*(T*(T*(T*a4/3 + a3/2) + a2) + a1*log(T))





ZABRANSKY_SPLINE = 'Zabransky spline, averaged heat capacity'
ZABRANSKY_QUASIPOLYNOMIAL = 'Zabransky quasipolynomial, averaged heat capacity'
ZABRANSKY_SPLINE_C = 'Zabransky spline, constant-pressure'
ZABRANSKY_QUASIPOLYNOMIAL_C = 'Zabransky quasipolynomial, constant-pressure'
ZABRANSKY_SPLINE_SAT = 'Zabransky spline, saturation'
ZABRANSKY_QUASIPOLYNOMIAL_SAT = 'Zabransky quasipolynomial, saturation'
ROWLINSON_POLING = 'Rowlinson and Poling (2001)'
ROWLINSON_BONDI = 'Rowlinson and Bondi (1969)'
DADGOSTAR_SHAW = 'Dadgostar and Shaw (2011)'


ZABRANSKY_TO_DICT = {ZABRANSKY_SPLINE: zabransky_dict_const_s,
                     ZABRANSKY_QUASIPOLYNOMIAL: zabransky_dict_const_p,
                     ZABRANSKY_SPLINE_C: zabransky_dict_iso_s,
                     ZABRANSKY_QUASIPOLYNOMIAL_C: zabransky_dict_iso_p,
                     ZABRANSKY_SPLINE_SAT: zabransky_dict_sat_s,
                     ZABRANSKY_QUASIPOLYNOMIAL_SAT: zabransky_dict_sat_p}
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
        available for 588, 146, 51, and 26 chemicals. 'C' denotes constant-
        pressure data available from more precise experiments. The others
        are heat capacity values averaged over a temperature changed.
    **ZABRANSKY_SPLINE_SAT and ZABRANSKY_QUASIPOLYNOMIAL_SAT**:
        Rigorous expressions developed in [1]_ following critical evaluation
        of the available data. The spline method use the form described in
        :obj:`Zabransky_cubic` over short ranges with varying coefficients
        to obtain a wider range. The quasi-polynomial method use the form
        described in :obj:`Zabransky_quasi_polynomial`, more suitable for
        extrapolation, and over their entire range. Respectively, there is data
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
    property_max = 1E4 # Originally 1E4
    '''Maximum valid of Heat capacity; arbitrarily set. For fluids very near
    the critical point, this value can be obscenely high.'''


    ranked_methods = [ZABRANSKY_SPLINE, ZABRANSKY_QUASIPOLYNOMIAL,
                      ZABRANSKY_SPLINE_C, ZABRANSKY_QUASIPOLYNOMIAL_C,
                      ZABRANSKY_SPLINE_SAT, ZABRANSKY_QUASIPOLYNOMIAL_SAT,
                      VDI_TABULAR, DADGOSTAR_SHAW, ROWLINSON_POLING, 
                      ROWLINSON_BONDI,
                      COOLPROP, POLING_CONST, CRCSTD]
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
        if self.CASRN in zabransky_dict_const_s:
            methods.append(ZABRANSKY_SPLINE)
            self.Zabransky_spline = zabransky_dict_const_s[self.CASRN]
        if self.CASRN in zabransky_dict_const_p:
            methods.append(ZABRANSKY_QUASIPOLYNOMIAL)
            self.Zabransky_quasipolynomial = zabransky_dict_const_p[self.CASRN]
        if self.CASRN in zabransky_dict_iso_s:
            methods.append(ZABRANSKY_SPLINE_C)
            self.Zabransky_spline_iso = zabransky_dict_iso_s[self.CASRN]
        if self.CASRN in zabransky_dict_iso_p:
            methods.append(ZABRANSKY_QUASIPOLYNOMIAL_C)
            self.Zabransky_quasipolynomial_iso = zabransky_dict_iso_p[self.CASRN]
        if self.CASRN in Poling_data.index and not np.isnan(Poling_data.at[self.CASRN, 'Cpl']):
            methods.append(POLING_CONST)
            self.POLING_T = 298.15
            self.POLING_constant = float(Poling_data.at[self.CASRN, 'Cpl'])
        if self.CASRN in CRC_standard_data.index and not np.isnan(CRC_standard_data.at[self.CASRN, 'Cpl']):
            methods.append(CRCSTD)
            self.CRCSTD_T = 298.15
            self.CRCSTD_constant = float(CRC_standard_data.at[self.CASRN, 'Cpl'])
        # Saturation functions
        if self.CASRN in zabransky_dict_sat_s:
            methods.append(ZABRANSKY_SPLINE_SAT)
            self.Zabransky_spline_sat = zabransky_dict_sat_s[self.CASRN]
        if self.CASRN in zabransky_dict_sat_p:
            methods.append(ZABRANSKY_QUASIPOLYNOMIAL_SAT)
            self.Zabransky_quasipolynomial_sat = zabransky_dict_sat_p[self.CASRN]
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
            return self.Zabransky_spline.calculate(T)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL:
            return self.Zabransky_quasipolynomial.calculate(T)
        elif method == ZABRANSKY_SPLINE_C:
            return self.Zabransky_spline_iso.calculate(T)
        elif method == ZABRANSKY_QUASIPOLYNOMIAL_C:
            return self.Zabransky_quasipolynomial_iso.calculate(T)
        elif method == ZABRANSKY_SPLINE_SAT:
            return self.Zabransky_spline_sat.calculate(T)
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
        elif method in self.tabular_data:
            return self.interpolate(T, method)
        else:
            raise Exception('Method not valid')

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
        if method == ZABRANSKY_SPLINE:
            if T < self.Zabransky_spline.Ts[0] or T > self.Zabransky_spline.Ts[-1]:
                return False
        elif method == ZABRANSKY_SPLINE_C:
            if T < self.Zabransky_spline_iso.Ts[0] or T > self.Zabransky_spline_iso.Ts[-1]:
                return False
        elif method == ZABRANSKY_SPLINE_SAT:
            if T < self.Zabransky_spline_sat.Ts[0] or T > self.Zabransky_spline_sat.Ts[-1]:
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
            if T <= self.CP_f.Tt or T >= self.CP_f.Tc:
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
        if method == ZABRANSKY_SPLINE:
            return self.Zabransky_spline.calculate_integral(T1, T2)
        elif method == ZABRANSKY_SPLINE_C:
            return self.Zabransky_spline_iso.calculate_integral(T1, T2)
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
            return float(quad(self.calculate, T1, T2, args=(method))[0])
        else:
            raise Exception('Method not valid')

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
        if method == ZABRANSKY_SPLINE:
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
        else:
            raise Exception('Method not valid')


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


def Lastovka_solid_integral(T, similarity_variable):
    r'''Integrates solid constant-pressure heat capacitiy with the similarity
    variable concept and method as shown in [1]_.
    
    Uses a explicit form as derived with Sympy.

    Parameters
    ----------
    T : float
        Temperature of solid [K]
    similarity_variable : float
        similarity variable as defined in [1]_, [mol/g]

    Returns
    -------
    H : float
        Difference in enthalpy from 0 K, [J/kg]

    Notes
    -----
    Original model is in terms of J/g/K. Note that the model is for predicting
    mass heat capacity, not molar heat capacity like most other methods!

    See Also
    --------
    Lastovka_solid

    Examples
    --------
    >>> Lastovka_solid_integral(300, 0.2139)
    283246.1242170376

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
    similarity_variable2 = similarity_variable*similarity_variable
    
    return (T*T*T*(1000.*D1*similarity_variable/3. 
        + 1000.*D2*similarity_variable2/3.) + T*T*(500.*C1*similarity_variable 
        + 500.*C2*similarity_variable2)
        + (3000.*A1*R*similarity_variable*theta
        + 3000.*A2*R*similarity_variable2*theta)/(exp(theta/T) - 1.))


def Lastovka_solid_integral_over_T(T, similarity_variable):
    r'''Integrates over T solid constant-pressure heat capacitiy with the 
    similarity variable concept and method as shown in [1]_.
    
    Uses a explicit form as derived with Sympy.

    Parameters
    ----------
    T : float
        Temperature of solid [K]
    similarity_variable : float
        similarity variable as defined in [1]_, [mol/g]

    Returns
    -------
    S : float
        Difference in entropy from 0 K, [J/kg/K]

    Notes
    -----
    Original model is in terms of J/g/K. Note that the model is for predicting
    mass heat capacity, not molar heat capacity like most other methods!

    See Also
    --------
    Lastovka_solid

    Examples
    --------
    >>> Lastovka_solid_integral_over_T(300, 0.2139)
    1947.553552666818

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
    
    sim2 = similarity_variable*similarity_variable
    exp_theta_T = exp(theta/T)
    
    return (-3000.*R*similarity_variable*(A1 + A2*similarity_variable)*log(exp_theta_T - 1.) 
    + T**2*(500.*D1*similarity_variable + 500.*D2*sim2)
    + T*(1000.*C1*similarity_variable + 1000.*C2*sim2)
    + (3000.*A1*R*similarity_variable*theta 
    + 3000.*A2*R*sim2*theta)/(T*exp_theta_T - T) 
    + (3000.*A1*R*similarity_variable*theta 
    + 3000.*A2*R*sim2*theta)/T)


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
            self.PERRY151_Tmin = _PerryI[self.CASRN]['c']['Tmin'] if _PerryI[self.CASRN]['c']['Tmin'] else 0
            self.PERRY151_Tmax = _PerryI[self.CASRN]['c']['Tmax'] if _PerryI[self.CASRN]['c']['Tmax'] else 2000
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
        if method == PERRY151:
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
        elif method in self.tabular_data:
            return float(quad(self.calculate, T1, T2, args=(method))[0])
        else:
            raise Exception('Method not valid')

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
        if method == PERRY151:
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
        elif method in self.tabular_data:
            return float(quad(lambda T: self.calculate(T, method)/T, T1, T2)[0])
        else:
            raise Exception('Method not valid')



### Mixture heat capacities
SIMPLE = 'SIMPLE'
LALIBERTE = 'Laliberte'
heat_capacity_gas_mixture_methods = [SIMPLE]
heat_capacity_liquid_mixture_methods = [LALIBERTE, SIMPLE]
heat_capacity_solid_mixture_methods = [SIMPLE]


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
        HeatCapacityLiquid objects created for all species in the mixture,  
        normally created by :obj:`thermo.chemical.Chemical`.

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_liquid_mixture_methods`.

    **LALIBERTE**:
        Electrolyte model equation with coefficients; see
        :obj:`thermo.electrochem.Laliberte_heat_capacity` for more details.
    **SIMPLE**:
        Mixing rule described in :obj:`thermo.utils.mixing_simple`.
    '''
    name = 'Liquid heat capacity'
    units = 'J/mol'
    property_min = 1
    '''Allow very low heat capacities; arbitrarily set; liquid heat capacity
    should always be somewhat substantial.'''
    property_max = 1E4 # Originally 1E4
    '''Maximum valid of Heat capacity; arbitrarily set. For fluids very near
    the critical point, this value can be obscenely high.'''
                            
    ranked_methods = [LALIBERTE, SIMPLE]

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
        if len(self.CASs) > 1 and '7732-18-5' in self.CASs:
            wCASs = [i for i in self.CASs if i != '7732-18-5'] 
            if all([i in _Laliberte_Heat_Capacity_ParametersDict for i in wCASs]):
                methods.append(LALIBERTE)
                self.wCASs = wCASs
                self.index_w = self.CASs.index('7732-18-5')
        self.all_methods = set(methods)
            
    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate heat capacity of a liquid mixture at 
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
        Cplm : float
            Molar heat capacity of the liquid mixture at the given conditions,
            [J/mol]
        '''
        if method == SIMPLE:
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
        HeatCapacitySolid objects created for all species in the mixture,  
        normally created by :obj:`thermo.chemical.Chemical`.

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_solid_mixture_methods`.

    **SIMPLE**:
        Mixing rule described in :obj:`thermo.utils.mixing_simple`.
    '''
    name = 'Solid heat capacity'
    units = 'J/mol'
    property_min = 0
    '''Heat capacities have a minimum value of 0 at 0 K.'''
    property_max = 1E4
    '''Maximum value of Heat capacity; arbitrarily set.'''
                            
    ranked_methods = [SIMPLE]

    def __init__(self, CASs=[], HeatCapacitySolids=[]):
        self.CASs = CASs
        self.HeatCapacitySolids = HeatCapacitySolids

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        heat capacity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        heat capacity above.'''

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
        self.all_methods = set(methods)
            
    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate heat capacity of a solid mixture at 
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
        Cpsm : float
            Molar heat capacity of the solid mixture at the given conditions, [J/mol]
        '''
        if method == SIMPLE:
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
        HeatCapacityGas objects created for all species in the mixture,  
        normally created by :obj:`thermo.chemical.Chemical`.

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`heat_capacity_gas_mixture_methods`.

    **SIMPLE**:
        Mixing rule described in :obj:`thermo.utils.mixing_simple`.
    '''
    name = 'Gas heat capacity'
    units = 'J/mol'
    property_min = 0
    '''Heat capacities have a minimum value of 0 at 0 K.'''
    property_max = 1E4
    '''Maximum valid of Heat capacity; arbitrarily set. For fluids very near
    the critical point, this value can be obscenely high.'''
                            
    ranked_methods = [SIMPLE]

    def __init__(self, CASs=[], HeatCapacityGases=[]):
        self.CASs = CASs
        self.HeatCapacityGases = HeatCapacityGases

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        heat capacity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        heat capacity above.'''

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
        self.all_methods = set(methods)
            
    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate heat capacity of a gas mixture at 
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
        Cpgm : float
            Molar heat capacity of the gas mixture at the given conditions,
            [J/mol]
        '''
        if method == SIMPLE:
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


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

__all__ = ['API_TDB_data', 'ATcT_l', 'ATcT_g', 'Hf_methods', 'Hf', 
           'Hf_l_methods', 'Hf_l', 'Hf_g_methods', 'Hf_g', 'Gibbs_formation',
           'entropy_formation', 'Hf_basis_converter',
           'S0_g_methods', 'S0_g', 'Yaws_Hf_S0']
           
import os
import numpy as np
import pandas as pd
from thermo.utils import isnan
from thermo.elements import periodic_table, CAS_by_number_standard
from thermo.heat_capacity import TRC_gas_data, CRC_standard_data

folder = os.path.join(os.path.dirname(__file__), 'Reactions')


API_TDB_data = pd.read_csv(os.path.join(folder, 'API TDB Albahri Hf.tsv'),
                           sep='\t', index_col=0)

ATcT_l = pd.read_csv(os.path.join(folder, 'ATcT 1.112 (l).tsv'),
                     sep='\t', index_col=0)

ATcT_g = pd.read_csv(os.path.join(folder, 'ATcT 1.112 (g).tsv'),
                     sep='\t', index_col=0)

Yaws_Hf_S0 = pd.read_csv(os.path.join(folder, 'Yaws Hf(g) S0(g).tsv'),
                     sep='\t', index_col=0)



# TODO: more data from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3692305/
# has dippr standard heats of formation, about 55% of the database

CRC = 'CRC'
YAWS = 'YAWS'

API_TDB = 'API_TDB'
NONE = 'NONE'
DEFINITION = 'Definition'
Hf_methods = [DEFINITION, API_TDB]


def Hf(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's standard-phase
    heat of formation. The lookup is based on CASRNs. Selects the only
    data source available ('API TDB') if the chemical is in it.
    Returns None if the data is not available.

    Function has data for 571 chemicals and the elements.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Hf : float
        Standard-state heat of formation, [J/mol]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Hf with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        Hf_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Hf for the desired chemical, and will return methods instead of Hf

    Notes
    -----
    Only two source of information is available to this function. They are:
        
        * 'Definition', which applies for the elements defined to be zero
        * 'API_TDB', a compilation of heats of formation of unspecified phase.
          Not the original data, but as reproduced in [1]_. Some chemicals with
          duplicated CAS numbers were removed.

    The API data for pentane is incorrect - it is for the gas phase, not the 
    standard liquid state. Likely further data are incorrect.
    
    Examples
    --------
    >>> Hf(CASRN='7732-18-5')
    -241820.0

    References
    ----------
    .. [1] Albahri, Tareq A., and Abdulla F. Aljasmi. "SGC Method for
       Predicting the Standard Enthalpy of Formation of Pure Compounds from
       Their Molecular Structures." Thermochimica Acta 568
       (September 20, 2013): 46-60. doi:10.1016/j.tca.2013.06.020.
    '''
    def list_methods():
        methods = []
        if CASRN in API_TDB_data.index:
            methods.append(API_TDB)
        if CASRN in CAS_by_number_standard:
            methods.append(DEFINITION)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == API_TDB:
        return float(API_TDB_data.at[CASRN, 'Hf'])
    elif Method == DEFINITION:
        return 0.0
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')


ATCT_L = 'ATCT_L'
ATCT_G = 'ATCT_G'

Hf_l_methods = [ATCT_L]


def Hf_l(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's liquid standard
    phase heat of formation. The lookup is based on CASRNs. Selects the only
    data source available, Active Thermochemical Tables (l), if the chemical is
    in it. Returns None if the data is not available.

    Function has data for 34 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Hfl : float
        Liquid standard-state heat of formation, [J/mol]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Hf(l) with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        Hf_l_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Hf(l) for the desired chemical, and will return methods instead of Hf(l)

    Notes
    -----
    Only one source of information is available to this function. It is:

        * 'ATCT_L', the Active Thermochemical Tables version 1.112.

    Examples
    --------
    >>> Hf_l('67-56-1')
    -238400.0

    References
    ----------
    .. [1] Ruscic, Branko, Reinhardt E. Pinzon, Gregor von Laszewski, Deepti
       Kodeboyina, Alexander Burcat, David Leahy, David Montoy, and Albert F.
       Wagner. "Active Thermochemical Tables: Thermochemistry for the 21st
       Century." Journal of Physics: Conference Series 16, no. 1
       (January 1, 2005): 561. doi:10.1088/1742-6596/16/1/078.
    '''
    def list_methods():
        methods = []
        if CASRN in ATcT_l.index:
            methods.append(ATCT_L)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == ATCT_L:
        _Hfl = float(ATcT_l.at[CASRN, 'Hf_298K'])
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Hfl


TRC = 'TRC'
Hf_g_methods = [ATCT_G, TRC, CRC, YAWS]


def Hf_g(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's gas heat of
    formation. Lookup is based on CASRNs. Will automatically select a data
    source to use if no Method is provided; returns None if the data is not
    available.

    Prefered sources are 'CRC' and 'Yaws'.
    Function has data for approximately 8700 chemicals.
    
    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Hfg : float
        Ideal gas phase heat of formation, [J/mol]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Hf(g) with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        Hf_g_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Hf(g) for the desired chemical, and will return methods instead of Hf(g)

    Notes
    -----
    Sources are:

        * 'ATCT_G', the Active Thermochemical Tables version 1.112 (600 values)
        * 'TRC', from a 1994 compilation (1750 values)
        * 'CRC', from the CRC handbook (1360 values)
        * 'YAWS', a large compillation of values, mostly estimated (5000 values)

    'TRC' data may have come from computational procedures, for example petane
    is off by 30%.

    Examples
    --------
    >>> Hf_g('67-56-1')
    -200700.0
    >>> Hf_g('67-56-1', Method='YAWS')
    -200900.0
    >>> Hf_g('67-56-1', Method='CRC')
    -201000.0
    >>> Hf_g('67-56-1', Method='TRC')
    -190100.0

    References
    ----------
    .. [1] Ruscic, Branko, Reinhardt E. Pinzon, Gregor von Laszewski, Deepti
       Kodeboyina, Alexander Burcat, David Leahy, David Montoy, and Albert F.
       Wagner. "Active Thermochemical Tables: Thermochemistry for the 21st
       Century." Journal of Physics: Conference Series 16, no. 1
       (January 1, 2005): 561. doi:10.1088/1742-6596/16/1/078.
    .. [2] Frenkelʹ, M. L, Texas Engineering Experiment Station, and
       Thermodynamics Research Center. Thermodynamics of Organic Compounds in
       the Gas State. College Station, Tex.: Thermodynamics Research Center,
       1994.
    .. [3] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    .. [4] Yaws, Carl L. Thermophysical Properties of Chemicals and
       Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
       Publishing, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in ATcT_g.index:
            methods.append(ATCT_G)
        if CASRN in Yaws_Hf_S0.index and not isnan(Yaws_Hf_S0.at[CASRN, 'Hf(g)']):
            methods.append(YAWS)
        if CASRN in CRC_standard_data.index and not isnan(CRC_standard_data.at[CASRN, 'Hfg']):
            methods.append(CRC)
        if CASRN in TRC_gas_data.index and not isnan(TRC_gas_data.at[CASRN, 'Hf']):
            methods.append(TRC)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == ATCT_G:
        _Hfg = float(ATcT_g.at[CASRN, 'Hf_298K'])
    elif Method == TRC:
        _Hfg = float(TRC_gas_data.at[CASRN, 'Hf'])
    elif Method == YAWS:
        _Hfg = float(Yaws_Hf_S0.at[CASRN, 'Hf(g)'])
    elif Method == CRC:
        _Hfg = float(CRC_standard_data.at[CASRN, 'Hfg'])
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Hfg



S0_g_methods = [CRC, YAWS]


def S0_g(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's absolute
    entropy at a reference temperature of 298.15 K and pressure of 1 bar,
    in the ideal gas state.

    Lookup is based on CASRNs. Will automatically select a data
    source to use if no Method is provided; returns None if the data is not
    available.

    Prefered sources are 'CRC', or 'Yaws'.
    Function has data for approximately 5400 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    S0_g : float
        Ideal gas standard absolute entropy of compound, [J/mol/K]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain S0(g) with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        S0_g_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        S0(g) for the desired chemical, and will return methods instead of S0(g)

    Notes
    -----
    Sources are:

        * 'CRC', from the CRC handbook (520 values)
        * 'YAWS', a large compillation of values, mostly estimated (4890 values)

    Examples
    --------
    >>> S0_g('67-56-1')
    239.9
    >>> S0_g('67-56-1', Method='YAWS')
    239.88

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    .. [2] Yaws, Carl L. Thermophysical Properties of Chemicals and
       Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
       Publishing, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in CRC_standard_data.index and not isnan(CRC_standard_data.at[CASRN, 'Sfg']):
            methods.append(CRC)
        if CASRN in Yaws_Hf_S0.index and not isnan(Yaws_Hf_S0.at[CASRN, 'S0(g)']):
            methods.append(YAWS)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == CRC:
        return float(CRC_standard_data.at[CASRN, 'Sfg'])
    elif Method == YAWS:
        return float(Yaws_Hf_S0.at[CASRN, 'S0(g)'])
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')


def Hf_basis_converter(Hvapm, Hf_liq=None, Hf_gas=None):
    r'''This function converts a liquid or gas enthalpy of formation to the
    other. This is useful, as thermodynamic packages often work with ideal-
    gas as the reference state and require ideal-gas enthalpies of formation.

    Parameters
    ----------
    Hvapm : float
        Molar enthalpy of vaporization of compound at 298.15 K or (unlikely)
        the reference temperature, [J/mol]
    Hf_liq : float, optional
        Enthalpy of formation of the compound in its liquid state, [J/mol]
    Hf_gas : float, optional
        Enthalpy of formation of the compound in its ideal-gas state, [J/mol]
        
    Returns
    -------
    Hf_calc : float, optional
        Enthalpy of formation of the compound in the other state to the one
        provided, [J/mol]

    Notes
    -----
    
    Examples
    --------
    Calculate the ideal-gas enthalpy of formation for water, from its standard-
    state (liquid) value:
        
    >>> Hf_basis_converter(44018, Hf_liq=-285830)
    -241812
    
    Calculate the standard-state (liquid) enthalpy of formation for water, from
    its ideal-gas value:

    >>> Hf_basis_converter(44018, Hf_gas=-241812)
    -285830
    '''
    if Hf_liq is None and Hf_gas is None:
        raise ValueError("Provide either a liquid or a gas enthalpy of formation")
    if Hvapm is None or Hvapm < 0.0:
        raise ValueError("Enthalpy of formation unknown or zero")
    if Hf_liq is None:
        return Hf_gas - Hvapm
    else:
        return Hf_liq + Hvapm


def Gibbs_formation(dHf, S0_abs, dHfs_std, S0_abs_elements, coeffs_elements,
                    T_ref=298.15):
    r'''This function calculates the Gibbs free energy of formation of a
    compound, from its constituent elements.
    
    The calculated value will be for a "standard-state" value if `dHf` and 
    `S0_abs` are provided in the standard state; or it will be in an
    "ideal gas" basis if they are both for an ideal gas. For compounds which
    are gases at STP, the two values are the same.

    Parameters
    ----------
    dHf : float
        Molar enthalpy of formation of the created compound, [J/mol]
    S0_abs : float
        Absolute molar entropy of the created compound at the reference
        temperature, [J/mol/K]
    dHfs_std : list[float]
        List of standard molar enthalpies of formation of all elements used in 
        the formation of the created compound, [J/mol]
    S0_abs_elements : list[float] 
        List of standard absolute molar entropies at the reference temperature
        of all elements used in the formation of the created compound, 
        [J/mol/K]
    coeffs_elements : list[float]
        List of coefficients for each compound (i.e. 1 for C, 2 for H2 if the 
        target is methane), in the same order as `dHfs_std` and 
        `S0_abs_elements`, [-]
    T_ref : float, optional
        The standard state temperature, default 298.15 K; few values are
        tabulated at other temperatures, [-]

    Returns
    -------
    dGf : float
        Gibbs free energy of formation for the created compound, [J/mol]

    Notes
    -----
    Be careful for elements like Bromine - is the tabulated value for Br2 or
    Br?

    Examples
    --------
    Calculate the standard-state Gibbs free energy of formation for water, 
    using water's standard state heat of formation and absolute entropy
    at 298.15 K:
        
    >>> Gibbs_formation(-285830, 69.91,  [0, 0], [130.571, 205.147], [1, .5])
    -237161.633825
    
    Calculate the ideal-gas state Gibbs free energy of formation for water, 
    using water's ideal-gas state heat of formation and absolute entropy
    at 298.15 K as a gas:
        
    >>> Gibbs_formation(-241818, 188.825,  [0, 0], [130.571, 205.147], [1, .5])
    -228604.141075
    
    Calculate the Gibbs free energy of formation for CBrF3 (it is a gas at STP,
    so its standard-state and ideal-gas state values are the same) at 298.15 K:

    >>> Gibbs_formation(-648980, 297.713, [0, 0, 0], [5.74, 152.206, 202.789], [1, .5, 1.5])
    -622649.329975
    
    Note in the above calculation that the Bromine's `S0` and `Hf` are for Br2;
    and that the value for Bromine as a liquid, which is its standard state,
    is used.
    
    References
    ----------
    .. [1] "Standard Gibbs Free Energy of Formation Calculations Chemistry 
       Tutorial." Accessed March, 2019. https://www.ausetute.com.au/gibbsform.html.
    '''
    dH = dHf - sum([Hi*ci for Hi, ci in zip(dHfs_std, coeffs_elements)])
    dS = S0_abs - sum([Si*ci for Si, ci in zip(S0_abs_elements, coeffs_elements)])
    return dH - T_ref*dS


def entropy_formation(Hf, Gf, T_ref=298.15):
    r'''This function calculates the entropy of formation of a
    compound, from its constituent elements.
    
    The calculated value will be for a "standard-state" value if `Hf` and 
    `Gf` are provided in the standard state; or it will be in an
    "ideal gas" basis if they are both for an ideal gas. For compounds which
    are gases at STP, the two values are the same.

    Parameters
    ----------
    Hf : float
        Molar enthalpy of formation of the compound, [J/mol]
    Gf : float
        Molar Gibbs free energy of formation of the compound, [J/mol]
    T_ref : float, optional
        The standard state temperature, default 298.15 K; few values are
        tabulated at other temperatures, [-]

    Returns
    -------
    Sf : float
        Entropy of formation of the compound, [J/mol/K]

    Notes
    -----
    
    Examples
    --------
    Entropy of formation of methane:
    
    >>> entropy_formation(Hf=-74520, Gf=-50490)
    -80.59701492537314
    
    Entropy of formation of water in ideal gas state:
        
    >>> entropy_formation(Hf=-241818, Gf=-228572)
    -44.427301693778304
    '''
    return (Hf - Gf)/T_ref

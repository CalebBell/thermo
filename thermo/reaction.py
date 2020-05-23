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

__all__ = ['API_TDB_data', 'ATcT_l', 'ATcT_g', 'Hf_methods', 'Hf', 
           'Hf_l_methods', 'Hf_l', 'Hf_g_methods', 'Hf_g', 'Gibbs_formation',
           'entropy_formation', 'Hf_basis_converter',
           'S0_g_methods', 'S0_g', 'Yaws_Hf_S0',
           'balance_stoichiometry', 'stoichiometric_matrix']
           
import os
from fractions import Fraction
import numpy as np
import scipy.linalg
import pandas as pd
from chemicals.utils import isnan, ceil, log10
from chemicals.elements import periodic_table, CAS_by_number_standard
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


def Hf(CASRN, get_methods=False, method=None):
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
    methods : list, only returned if get_methods == True
        List of methods which can be used to obtain Hf with the given inputs

    Other Parameters
    ----------------
    method : string, optional
        A string for the method name to use, as defined by constants in
        Hf_methods
    get_methods : bool, optional
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
    if get_methods:
        return list_methods()
    if not method:
        method = list_methods()[0]

    if method == API_TDB:
        return float(API_TDB_data.at[CASRN, 'Hf'])
    elif method == DEFINITION:
        return 0.0
    elif method == NONE:
        return None
    else:
        raise Exception('Failure in in function')


ATCT_L = 'ATCT_L'
ATCT_G = 'ATCT_G'

Hf_l_methods = [ATCT_L]


def Hf_l(CASRN, get_methods=False, method=None):
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
    methods : list, only returned if get_methods == True
        List of methods which can be used to obtain Hf(l) with the given inputs

    Other Parameters
    ----------------
    method : string, optional
        A string for the method name to use, as defined by constants in
        Hf_l_methods
    get_methods : bool, optional
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
    if get_methods:
        return list_methods()
    if not method:
        method = list_methods()[0]

    if method == ATCT_L:
        _Hfl = float(ATcT_l.at[CASRN, 'Hf_298K'])
    elif method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Hfl


TRC = 'TRC'
Hf_g_methods = [ATCT_G, TRC, CRC, YAWS]


def Hf_g(CASRN, get_methods=False, method=None):
    r'''This function handles the retrieval of a chemical's gas heat of
    formation. Lookup is based on CASRNs. Will automatically select a data
    source to use if no method is provided; returns None if the data is not
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
    methods : list, only returned if get_methods == True
        List of methods which can be used to obtain Hf(g) with the given inputs

    Other Parameters
    ----------------
    method : string, optional
        A string for the method name to use, as defined by constants in
        Hf_g_methods
    get_methods : bool, optional
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
    >>> Hf_g('67-56-1', method='YAWS')
    -200900.0
    >>> Hf_g('67-56-1', method='CRC')
    -201000.0
    >>> Hf_g('67-56-1', method='TRC')
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
    if get_methods:
        return list_methods()
    if not method:
        method = list_methods()[0]

    if method == ATCT_G:
        _Hfg = float(ATcT_g.at[CASRN, 'Hf_298K'])
    elif method == TRC:
        _Hfg = float(TRC_gas_data.at[CASRN, 'Hf'])
    elif method == YAWS:
        _Hfg = float(Yaws_Hf_S0.at[CASRN, 'Hf(g)'])
    elif method == CRC:
        _Hfg = float(CRC_standard_data.at[CASRN, 'Hfg'])
    elif method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Hfg



S0_g_methods = [CRC, YAWS]


def S0_g(CASRN, get_methods=False, method=None):
    r'''This function handles the retrieval of a chemical's absolute
    entropy at a reference temperature of 298.15 K and pressure of 1 bar,
    in the ideal gas state.

    Lookup is based on CASRNs. Will automatically select a data
    source to use if no method is provided; returns None if the data is not
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
    methods : list, only returned if get_methods == True
        List of methods which can be used to obtain S0(g) with the given inputs

    Other Parameters
    ----------------
    method : string, optional
        A string for the method name to use, as defined by constants in
        S0_g_methods
    get_methods : bool, optional
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
    >>> S0_g('67-56-1', method='YAWS')
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
    if get_methods:
        return list_methods()
    if not method:
        method = list_methods()[0]

    if method == CRC:
        return float(CRC_standard_data.at[CASRN, 'Sfg'])
    elif method == YAWS:
        return float(Yaws_Hf_S0.at[CASRN, 'S0(g)'])
    elif method == NONE:
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


def stoichiometric_matrix(atomss, reactants):
    r'''This function calculates a stoichiometric matrix of reactants and 
    stoichiometric matrix, as required by a solver to compute the reation
    coefficients.

    Parameters
    ----------
    atomss : list[dict[(str, float)]]
        A list of dictionaties of (element, element_count) pairs for each 
        chemical, [-]
    reactants : list[bool]
        List of booleans indicating whether each chemical is a reactant (True)
        or a product (False), [-]

    Returns
    -------
    matrix : list[list[float]]
        Chemical reaction matrix for further processing, [-]

    Notes
    -----
    The rows of the matrix contain the element counts of each compound,
    and the columns represent each chemical. 
    
    Examples
    --------
    MgO2 -> Mg + 1/2 O2
    (k=1)
    
    >>> stoichiometric_matrix([{'Hg': 1, 'O': 1}, {u'Hg': 1}, {'O': 2}], [True, False, False])
    [[1, -1, 0.0], [1, 0.0, -2]]
    
    
    Cl2 + propylene -> allyl chloride + HCl
    
    >>> stoichiometric_matrix([{'Cl': 2}, {'C': 3, 'H': 6}, {'C': 3, 'Cl': 1, 'H': 5}, {'Cl': 1, 'H': 1}], [True, True, False, False, False])
    [[0.0, 6, -5, -1], [0.0, 3, -3, 0.0], [2, 0.0, -1, -1]]

    
    Al + 4HNO3 -> Al(NO3)3 + NO + 2H2O 
    (k=1)

    >>> stoichiometric_matrix([{'Al': 1}, {'H': 1, 'N': 1, 'O': 3}, {'Al': 1, 'N': 3, 'O': 9}, {'N': 1, 'O': 1}, {'H': 2, 'O': 1}], [True, True, False, False, False])
    [[0.0, 1, 0.0, 0.0, -2],
     [1, 0.0, -1, 0.0, 0.0],
     [0.0, 3, -9, -1, -1],
     [0.0, 1, -3, -1, 0.0]]

    
    4Fe + 3O2 -> 2(Fe2O3)
    (k=2)
    
    >>> stoichiometric_matrix([{'Fe': 1}, {'O': 2}, {'Fe':2, 'O': 3}], [True, True, False])
    [[1, 0.0, -2], [0.0, 2, -3]]

    
    4NH3 + 5O2 -> 4NO + 6(H2O)
    (k=4)
    
    >>> stoichiometric_matrix([{'N': 1, 'H': 3}, {'O': 2}, {'N': 1, 'O': 1}, {'H': 2, 'O': 1}], [True, True, False, False])
    [[3, 0.0, 0.0, -2], [0.0, 2, -1, -1], [1, 0.0, -1, 0.0]]

    
    No unique solution:
    C2H5NO2 + C3H7NO3 + 2C6H14N4O2 + 3C5H9NO2 + 2C9H11NO2 -> 8H2O + C50H73N15O11
    
    >>> stoichiometric_matrix([{'C': 2, 'H': 5, 'N': 1, 'O': 2}, {'C': 3, 'H': 7, 'N': 1, 'O': 3}, {'C': 6, 'H': 14, 'N': 4, 'O': 2}, {'C': 5, 'H': 9, 'N': 1, 'O': 2}, {'C': 9, 'H': 11, 'N': 1, 'O': 2}, {'H': 2, 'O': 1}, {'C': 50, 'H': 73, 'N': 15, 'O': 11}], [True, True, True, True, True, False, False])
    [[5, 7, 14, 9, 11, -2, -73],
     [2, 3, 6, 5, 9, 0.0, -50],
     [2, 3, 2, 2, 2, -1, -11],
     [1, 1, 4, 1, 1, 0.0, -15]]

    References
    ----------
    .. [1] Sen, S. K., Hans Agarwal, and Sagar Sen. "Chemical Equation 
       Balancing: An Integer Programming Approach." Mathematical and Computer 
       Modelling 44, no. 7 (October 1, 2006): 678-91.
       https://doi.org/10.1016/j.mcm.2006.02.004.
    .. [2] URAVNOTE, NOVOODKRITI PARADOKSI V. TEORIJI, and ENJA KEMIJSKIH 
       REAKCIJ. "New Discovered Paradoxes in Theory of Balancing Chemical 
       Reactions." Materiali in Tehnologije 45, no. 6 (2011): 503-22.
    '''
    n_compounds = len(atomss)
    elements = set()
    for atoms in atomss:
        elements.update(atoms.keys())
    elements = list(elements)
    n_elements = len(elements)

    matrix = [[0]*n_compounds for _ in range(n_elements)]
    for i, atoms in enumerate(atomss):
        for k, v in atoms.items():
            if not reactants[i]:
                v = -v
            matrix[elements.index(k)][i] = v
    return matrix


def balance_stoichiometry(matrix, rounding=9, allow_fractional=False):
    done = scipy.linalg.null_space(matrix)
    if len(done[0]) > 1:
        raise ValueError("No solution")
    d = done[:, 0].tolist()

    min_value_inv = 1.0/min(d)
    d = [i*min_value_inv for i in d]

    if not allow_fractional:
        max_denominator = 10**rounding
        fs = [Fraction(x).limit_denominator(max_denominator=max_denominator) for x in d]
        all_denominators = set([i.denominator for i in fs])
        if 1 in all_denominators: 
            all_denominators.remove(1)
        
        for den in sorted(list(all_denominators), reverse=True):
            fs = [num*den for num in fs]
            if all(i.denominator == 1 for i in fs):
                break
        
        # May have gone too far
        return [float(i) for i in fs]
#        done = False
#        for i in range(100):
#            for c in d:
#                ratio = c.as_integer_ratio()[1]
#                if ratio != 1:
#                    d = [di*ratio for di in d]
#                    break
#                done = True
#            if done:
#                break
#
#        d_as_int = [int(i) for i in d]
#        for i, j in zip(d, d_as_int):
#            if i != j:
#                raise ValueError("Could not find integer coefficients (%s, %s)" %(i, j))
#        return d_as_int
    else:
        d = [round(i, rounding + int(ceil(log10(abs(i))))) for i in d]
        return d
        
    

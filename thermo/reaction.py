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
           'Hf_l_methods', 'Hf_l', 'Hf_g_methods', 'Hf_g']
           
import os
import numpy as np
import pandas as pd
from thermo.heat_capacity import TRC_gas_data

folder = os.path.join(os.path.dirname(__file__), 'Reactions')


API_TDB_data = pd.read_csv(os.path.join(folder, 'API TDB Albahri Hf.tsv'),
                           sep='\t', index_col=0)

ATcT_l = pd.read_csv(os.path.join(folder, 'ATcT 1.112 (l).tsv'),
                     sep='\t', index_col=0)

ATcT_g = pd.read_csv(os.path.join(folder, 'ATcT 1.112 (g).tsv'),
                     sep='\t', index_col=0)


API_TDB = 'API_TDB'
NONE = 'NONE'
Hf_methods = [API_TDB]


def Hf(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's standard-phase
    heat of formation. The lookup is based on CASRNs. Selects the only
    data source available ('API TDB') if the chemical is in it.
    Returns None if the data is not available.

    Function has data for 571 chemicals.

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
    Only one source of information is available to this function. it is:

        * 'API_TDB', a compilation of heats of formation of unspecified phase.
          Not the original data, but as reproduced in [1]_. Some chemicals with
          duplicated CAS numbers were removed.

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
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == API_TDB:
        _Hf = float(API_TDB_data.at[CASRN, 'Hf'])
    elif Method == NONE:
        _Hf = None
    else:
        raise Exception('Failure in in function')
    return _Hf


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
Hf_g_methods = [ATCT_G, TRC]


def Hf_g(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's gas heat of
    formation. Lookup is based on CASRNs. Will automatically select a data
    source to use if no Method is provided; returns None if the data is not
    available.

    Prefered sources are 'Active Thermochemical Tables (g)' for high accuracy,
    and 'TRC' for less accuracy but more chemicals.
    Function has data for approximately 2000 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    _Hfg : float
        Gas phase heat of formation, [J/mol]
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

        * 'ATCT_G', the Active Thermochemical Tables version 1.112.
        * 'TRC', from a 1994 compilation.

    Examples
    --------
    >>> Hf_g('67-56-1')
    -200700.0

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
    '''
    def list_methods():
        methods = []
        if CASRN in ATcT_g.index:
            methods.append(ATCT_G)
        if CASRN in TRC_gas_data.index and not np.isnan(TRC_gas_data.at[CASRN, 'Hf']):
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
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Hfg

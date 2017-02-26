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

__all__ = ['GWP', 'ODP', 'logP', 'GWP_methods', 'GWP_data', 'ODP_methods', 
           'ODP_data', 'logP_methods', 'CRClogPDict', 'SyrresDict2']

import os
import pandas as pd


folder = os.path.join(os.path.dirname(__file__), 'Environment')


### Global Warming Potentials

GWP_data = pd.read_csv(os.path.join(folder,
                       'Official Global Warming Potentials.tsv'), sep='\t',
                       index_col=0)

ODP_data = pd.read_csv(os.path.join(folder,
                       'Ozone Depletion Potentials.tsv'), sep='\t',
                       index_col=0)

CRClogPDict = pd.read_csv(os.path.join(folder,
                       'CRC logP table.tsv'), sep='\t',
                       index_col=0)

SyrresDict2 = pd.read_csv(os.path.join(folder,
                       'Syrres logP data.csv.gz'), sep='\t',
                       index_col=0, compression='gzip')

IPCC100 = 'IPCC (2007) 100yr'
IPCC100SAR = 'IPCC (2007) 100yr-SAR'
IPCC20 = 'IPCC (2007) 20yr'
IPCC500 = 'IPCC (2007) 500yr'
NONE = 'None'
GWP_methods = [IPCC100, IPCC100SAR, IPCC20, IPCC500]


def GWP(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's Global Warming
    Potential, relative to CO2. Lookup is based on CASRNs. Will automatically
    select a data source to use if no Method is provided; returns None if the
    data is not available.

    Returns the GWP for the 100yr outlook by default.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    GWP : float
        Global warming potential, [(impact/mass chemical)/(impact/mass CO2)]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain GWP with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are IPCC (2007) 100yr',
        'IPCC (2007) 100yr-SAR', 'IPCC (2007) 20yr', and 'IPCC (2007) 500yr'. 
        All valid values are also held in the list GWP_methods.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the GWP for the desired chemical, and will return methods
        instead of the GWP

    Notes
    -----
    All data is from [1]_, the official source. Several chemicals are available
    in [1]_ are not included here as they do not have a CAS.
    Methods are 'IPCC (2007) 100yr', 'IPCC (2007) 100yr-SAR',
    'IPCC (2007) 20yr', and 'IPCC (2007) 500yr'.

    Examples
    --------
    Methane, 100-yr outlook

    >>> GWP(CASRN='74-82-8')
    25.0

    References
    ----------
    .. [1] IPCC. "2.10.2 Direct Global Warming Potentials - AR4 WGI Chapter 2:
       Changes in Atmospheric Constituents and in Radiative Forcing." 2007.
       https://www.ipcc.ch/publications_and_data/ar4/wg1/en/ch2s2-10-2.html.
    '''
    def list_methods():
        methods = []
        if CASRN in GWP_data.index:
            methods.append(IPCC100)
            if not pd.isnull(GWP_data.at[CASRN, 'SAR 100yr']):
                methods.append(IPCC100SAR)
            methods.append(IPCC20)
            methods.append(IPCC500)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IPCC100:
        return float(GWP_data.at[CASRN, '100yr GWP'])
    elif Method == IPCC100SAR:
        return float(GWP_data.at[CASRN, 'SAR 100yr'])
    elif Method == IPCC20:
        return float(GWP_data.at[CASRN, '20yr GWP'])
    elif Method == IPCC500:
        return float(GWP_data.at[CASRN, '500yr GWP'])
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')


### Ozone Depletion Potentials


ODP2MAX = 'ODP2 Max'
ODP2MIN = 'ODP2 Min'
ODP2STR = 'ODP2 string'
ODP2LOG = 'ODP2 logarithmic average'
ODP1MAX = 'ODP1 Max'
ODP1MIN = 'ODP1 Min'
ODP1STR = 'ODP1 string'
ODP1LOG = 'ODP1 logarithmic average'
NONE = 'None'
ODP_methods = [ODP2MAX, ODP1MAX, ODP2LOG, ODP1LOG, ODP2MIN, ODP1MIN, ODP2STR, ODP1STR]


def ODP(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's Ozone Depletion
    Potential, relative to CFC-11 (trichlorofluoromethane). Lookup is based on
    CASRNs. Will automatically select a data source to use if no Method is
    provided; returns None if the data is not available.

    Returns the ODP of a chemical according to [2]_ when a method is not
    specified. If a range is provided in [2]_, the highest value is returned.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    ODP : float or str
        Ozone Depletion potential, [(impact/mass chemical)/(impact/mass CFC-11)];
        if method selected has `string` in it, this will be returned as a
        string regardless of if a range is given or a number
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain ODP with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'ODP2 Max', 'ODP2 Min', 
        'ODP2 string', 'ODP2 logarithmic average', and methods for older values
        are 'ODP1 Max', 'ODP1 Min', 'ODP1 string', and 'ODP1 logarithmic average'.
        All valid values are also held in the list ODP_methods.
    Method : string, optional
        A string for the method name to use, as defined by constants in
        ODP_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the ODP for the desired chemical, and will return methods
        instead of the ODP

    Notes
    -----
    Values are tabulated only for a small number of halogenated hydrocarbons,
    responsible for the largest impact. The original values of ODP as defined
    in the Montreal Protocol are also available, as methods with the `ODP1`
    prefix.

    All values are somewhat emperical, as actual reaction rates of chemicals
    with ozone depend on temperature which depends on latitude, longitude,
    time of day, weather, and the concentrations of other pollutants.

    All data is from [1]_. Several mixtures listed in [1]_ are not included
    here as they are not pure species.
    Methods for values in [2]_ are 'ODP2 Max', 'ODP2 Min', 'ODP2 string',
    'ODP2 logarithmic average',  and methods for older values are 'ODP1 Max',
    'ODP1 Min', 'ODP1 string', and 'ODP1 logarithmic average'.

    Examples
    --------
    Dichlorotetrafluoroethane, according to [2]_.

    >>> ODP(CASRN='76-14-2')
    0.58

    References
    ----------
    .. [1] US EPA, OAR. "Ozone-Depleting Substances." Accessed April 26, 2016.
       https://www.epa.gov/ozone-layer-protection/ozone-depleting-substances.
    .. [2] WMO (World Meteorological Organization), 2011: Scientific Assessment
       of Ozone Depletion: 2010. Global Ozone Research and Monitoring
       Project-Report No. 52, Geneva, Switzerland, 516 p.
       https://www.wmo.int/pages/prog/arep/gaw/ozone_2010/documents/Ozone-Assessment-2010-complete.pdf
    '''
    def list_methods():
        methods = []
        if CASRN in ODP_data.index:
            if not pd.isnull(ODP_data.at[CASRN, 'ODP2 Max']):
                methods.append(ODP2MAX)
            if not pd.isnull(ODP_data.at[CASRN, 'ODP1 Max']):
                methods.append(ODP1MAX)
            if not pd.isnull(ODP_data.at[CASRN, 'ODP2 Design']):
                methods.append(ODP2LOG)
            if not pd.isnull(ODP_data.at[CASRN, 'ODP1 Design']):
                methods.append(ODP1LOG)
            if not pd.isnull(ODP_data.at[CASRN, 'ODP2 Min']):
                methods.append(ODP2MIN)
            if not pd.isnull(ODP_data.at[CASRN, 'ODP1 Min']):
                methods.append(ODP1MIN)
            if not pd.isnull(ODP_data.at[CASRN, 'ODP2']):
                methods.append(ODP2STR)
            if not pd.isnull(ODP_data.at[CASRN, 'ODP1']):
                methods.append(ODP1STR)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == ODP2MAX:
        return float(ODP_data.at[CASRN, 'ODP2 Max'])
    elif Method == ODP1MAX:
        return float(ODP_data.at[CASRN, 'ODP1 Max'])
    elif Method == ODP2MIN:
        return float(ODP_data.at[CASRN, 'ODP2 Min'])
    elif Method == ODP1MIN:
        return float(ODP_data.at[CASRN, 'ODP1 Min'])
    elif Method == ODP2LOG:
        return float(ODP_data.at[CASRN, 'ODP2 Design'])
    elif Method == ODP1LOG:
        return float(ODP_data.at[CASRN, 'ODP1 Design'])
    elif Method == ODP2STR:
        return str(ODP_data.at[CASRN, 'ODP2'])
    elif Method == ODP1STR:
        return str(ODP_data.at[CASRN, 'ODP1'])
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')


### log P


SYRRES = 'SYRRES'
CRC = 'CRC'
NONE = 'NONE'
logP_methods = [SYRRES, CRC]


def logP(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's octanol-water
    partition coefficient. Lookup is based on CASRNs. Will automatically
    select a data source to use if no Method is provided; returns None if the
    data is not available.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    logP : float
        Octanol-water partition coefficient, [-]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain logP with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'SYRRES', or 'CRC', 
        All valid values are also held in the list logP_methods.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the logP for the desired chemical, and will return methods
        instead of the logP

    Notes
    -----
    .. math::
        \log P_{ oct/wat} = \log\left(\frac{\left[{solute}
        \right]_{ octanol}^{un-ionized}}{\left[{solute}
        \right]_{ water}^{ un-ionized}}\right)

    Examples
    --------
    >>> logP('67-56-1')
    -0.74

    References
    ----------
    .. [1] Syrres. 2006. KOWWIN Data, SrcKowData2.zip.
       http://esc.syrres.com/interkow/Download/SrcKowData2.zip
    .. [2] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics, 95E. Boca Raton, FL: CRC press, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in CRClogPDict.index:
            methods.append(CRC)
        if CASRN in SyrresDict2.index:
            methods.append(SYRRES)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == CRC:
        return float(CRClogPDict.at[CASRN, 'logP'])
    elif Method == SYRRES:
        return float(SyrresDict2.at[CASRN, 'logP'])
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')

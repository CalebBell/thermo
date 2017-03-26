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

__all__ = ['ppmv_to_mgm3', 'mgm3_to_ppmv', 'NFPA_2008', 'IEC_2010', 
'_OntarioExposureLimits', 'NTP_data', 'NTP_codes', 'IARC_data', 'IARC_codes', 
'TWA_methods', 'TWA', 'STEL', 'Ceiling', 'Skin', 'Carcinogen_methods', 
'Carcinogen', 'Tflash_methods', 'Tflash', 'Tautoignition_methods', 
'Tautoignition', 'LFL_methods', 'LFL', 'UFL_methods', 'UFL', 'fire_mixing', 
'inerts', 'LFL_mixture', 'UFL_mixture', 'Suzuki_LFL', 'Suzuki_UFL', 
'Crowl_Louvar_LFL', 'Crowl_Louvar_UFL']

import os
from io import open
import numpy as np
import pandas as pd
from thermo.utils import R
from thermo.utils import to_num, none_and_length_check, normalize
from thermo.identifiers import CASfromAny, MW

folder = os.path.join(os.path.dirname(__file__), 'Safety')


def ppmv_to_mgm3(ppmv, MW, T=298.15, P=101325.):
    r'''Converts a concentration in ppmv to units of mg/m^3. Used in
    industrial toxicology.

    .. math::
        \frac{mg}{m^3} = \frac{ppmv\cdot P}{RT}\cdot \frac{MW}{1000}

    Parameters
    ----------
    ppmv : float
        Concentratoin of a component in a gas mixure [parts per million,
        volumetric]
    MW : float
        Molecular weight of the trace gas [g/mol]
    T : float, optional
        Temperature of the gas at which the ppmv is reported
    P : float, optional
        Pressure of the gas at which the ppmv is reported

    Returns
    -------
    mgm3 : float
        Concentration of a substance in an ideal gas mixture [mg/m^3]

    Notes
    -----
    The term P/(RT)/1000 converts to 0.040874 at STP. Its inverse is reported
    as 24.45 in [1]_.

    Examples
    --------
    >>> ppmv_to_mgm3(1, 40)
    1.6349623351068687

    References
    ----------
    .. [1] ACGIH. Industrial Ventilation: A Manual of Recommended Practice,
       23rd Edition. American Conference of Governmental and Industrial
       Hygenists, 2004.
    '''
    parts = ppmv*1E-6
    n = parts*P/(R*T)
    mgm3 = MW*n*1000  # mol toxin /m^3 * g/mol toxis * 1000 mg/g
    return mgm3


def mgm3_to_ppmv(mgm3, MW, T=298.15, P=101325.):
    r'''Converts a concentration in  mg/m^3 to units of ppmv. Used in
    industrial toxicology.

    .. math::
        ppmv = \frac{1000RT}{MW\cdot P} \cdot \frac{mg}{m^3}

    Parameters
    ----------
    mgm3 : float
        Concentration of a substance in an ideal gas mixture [mg/m^3]
    MW : float
        Molecular weight of the trace gas [g/mol]
    T : float, optional
        Temperature of the gas at which the ppmv is reported
    P : float, optional
        Pressure of the gas at which the ppmv is reported

    Returns
    -------
    ppmv : float
        Concentration of a component in a gas mixure [parts per million,
        volumetric]

    Notes
    -----
    The term P/(RT)/1000 converts to 0.040874 at STP. Its inverse is reported
    as 24.45 in [1]_.

    Examples
    --------
    >>> mgm3_to_ppmv(1.635, 40)
    1.0000230371625833

    References
    ----------
    .. [1] ACGIH. Industrial Ventilation: A Manual of Recommended Practice,
       23rd Edition. American Conference of Governmental and Industrial
       Hygenists, 2004.
    '''
    n = mgm3/MW/1000.
    parts = n*R*T/P
    ppm = parts/1E-6
    return ppm


def str_to_ppm_mgm3(line, CAS):  # pragma: no cover
    if not line:
        return None, None
    if 'ppm' in line:
        _ppm = float(line.split('ppm')[0])
        try:
            _mgm3 = ppmv_to_mgm3(_ppm, MW(CASfromAny(CAS)))
        except:
            _mgm3 = None
    elif 'mg/m3' in line:
        _mgm3 = float(line.split('mg/m3')[0])
        try:
            _ppm = mgm3_to_ppmv(_mgm3, MW(CASfromAny(CAS)))
        except:
            _ppm = None
    if not _ppm and not _mgm3:
        raise Exception('failure in function')
    return (_ppm, _mgm3)


NFPA_2008 = pd.read_csv(os.path.join(folder, 'NFPA 497 2008.tsv'),
                        sep='\t', index_col=0)


IEC_2010 = pd.read_csv(os.path.join(folder, 'IS IEC 60079-20-1 2010.tsv'),
                       sep='\t', index_col=0)

_OntarioExposureLimits = {}


with open(os.path.join(folder, 'Ontario Exposure Limits.tsv'), encoding='utf-8') as f:
    '''Read in a dict of TWAs, STELs, and Ceiling Limits. The data source
    is the Ontario Labor Website. They have obtained their data in part from
    their own reviews, and also from ACGIH.
    Warning: The lowest value is taken, when multiple units or different forms
             of a compound are listed.
    Note that each province has a different set of values, but these serve
    as general values.
    '''
    next(f)
    for line in f:
        values = to_num(line.strip('\n').split('\t'))
        if values[0]:
            for CASRN in values[0].split(';'):
                _ppm_TWA, _mgm3_TWA = str_to_ppm_mgm3(values[2], CASRN.strip())
                _ppm_STEL, _mgm3_STEL = str_to_ppm_mgm3(values[3], CASRN.strip())
                _ppm_C, _mgm3_C = str_to_ppm_mgm3(values[4], CASRN.strip())
                if values[5] == 'Skin':
                    _skin = True
                else:
                    _skin = False
                _OntarioExposureLimits[CASRN] = {"Name": values[1],  "TWA (ppm)": _ppm_TWA,
                "TWA (mg/m^3)": _mgm3_TWA, "STEL (ppm)": _ppm_STEL,
                "STEL (mg/m^3)": _mgm3_STEL, "Ceiling (ppm)": _ppm_C,
                "Ceiling (mg/m^3)": _mgm3_C, "Skin":_skin}

#TODO: Add CRC exposure limits. Note that functions should be used.
#_CRCExposureLimits = {}
#with open(os.path.join(folder,'CRC Exposure Limits.csv')) as f:
#    '''Read in a dict of TWAs and STELs. The data source
#    is the CRC Handbook.. They have obtained their data from
#    NIOSH, and OSHA Chemical Information Manual, and ACGIH.
#    '''
#    f.next()
#    for line in f:
#        values = to_num(line.strip('\n').split('\t'))
#        _ppm_TWA, _mgm3_TWA = str_to_ppm_mgm3(values[2], CASRN.strip())
#        _ppm_STEL, _mgm3_STEL = str_to_ppm_mgm3(values[3], CASRN.strip())
#        _CRCExposureLimits[CASRN] = {"Name": values[1],  "TWA (ppm)": _ppm_TWA,
#        "TWA (mg/m^3)": _mgm3_TWA, "STEL (ppm)": _ppm_STEL,
#        "STEL (mg/m^3)": _mgm3_STEL}



#del _ppm_TWA, _mgm3_TWA, _ppm_STEL, _mgm3_STEL, _ppm_C, _mgm3_C, status

#print _OntarioExposureLimits['109-73-9']
##{'STEL (ppm)': None, 'Name': 'n-Butylamine [109-73-9]', 'Ceiling (mg/m^3)': 14.956408997955013, 'Ceiling (ppm)': 5.0, 'TWA (mg/m^3)': None, 'STEL (mg/m^3)': None, 'TWA (ppm)': None}
#print _OntarioExposureLimits['34590-94-8']
#{'STEL (ppm)': 150.0, 'Name': '(2-Methoxymethylethoxy) propanol (DPGME) [34590-94-8]', 'Ceiling (mg/m^3)': None, 'Ceiling (ppm)': None, 'TWA (mg/m^3)': None, 'STEL (mg/m^3)': None, 'TWA (ppm)': 100.0}


NTP_data = pd.read_csv(os.path.join(folder, 'National Toxicology Program Carcinogens.tsv'),
                       sep='\t', index_col=0)

NTP_codes = {1: 'Known', 2: 'Reasonably Anticipated'}

IARC_data = pd.read_csv(os.path.join(folder, 'IARC Carcinogen Database.tsv'),
                        sep='\t', index_col=0)

IARC_codes = {1: 'Carcinogenic to humans (1)',
              11: 'Probably carcinogenic to humans (2A)',  # 2A
              12: 'Possibly carcinogenic to humans (2B)',  # 2B
              3: 'Not classifiable as to its carcinogenicity to humans (3)',
              4: 'Probably not carcinogenic to humans (4)'}

### OSHA exposure limit functions

ONTARIO = 'Ontario Limits'
NONE = 'None'

TWA_methods = [ONTARIO]


def TWA(CASRN, AvailableMethods=False, Method=None):  # pragma: no cover
    '''This function handles the retrieval of Time-Weighted Average limits on worker
    exposure to dangerous chemicals.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> TWA('98-00-0')
    (10.0, 'ppm')
    >>> TWA('1303-00-0')
    (5.0742430905659505e-05, 'ppm')
    >>> TWA('7782-42-5', AvailableMethods=True)
    ['Ontario Limits', 'None']
    '''
    def list_methods():
        methods = []
        if CASRN in _OntarioExposureLimits and (_OntarioExposureLimits[CASRN]["TWA (ppm)"] or _OntarioExposureLimits[CASRN]["TWA (mg/m^3)"]):
            methods.append(ONTARIO)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == ONTARIO:
        if _OntarioExposureLimits[CASRN]["TWA (ppm)"]:
            _TWA = (_OntarioExposureLimits[CASRN]["TWA (ppm)"], 'ppm')
        elif _OntarioExposureLimits[CASRN]["TWA (mg/m^3)"]:
            _TWA = (_OntarioExposureLimits[CASRN]["TWA (mg/m^3)"], 'mg/m^3')
    elif Method == NONE:
        _TWA = None
    else:
        raise Exception('Failure in in function')
    return _TWA


def STEL(CASRN, AvailableMethods=False, Method=None):  # pragma: no cover
    '''This function handles the retrieval of Short-term Exposure Limit on
    worker exposure to dangerous chemicals.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> STEL('67-64-1')
    (750.0, 'ppm')
    >>> STEL('7664-38-2')
    (0.7489774978301237, 'ppm')
    >>> STEL('55720-99-5')
    (2.0, 'mg/m^3')
    >>> STEL('86290-81-5', AvailableMethods=True)
    ['Ontario Limits', 'None']
    '''
    def list_methods():
        methods = []
        if CASRN in _OntarioExposureLimits and (_OntarioExposureLimits[CASRN]["STEL (ppm)"] or _OntarioExposureLimits[CASRN]["STEL (mg/m^3)"]):
            methods.append(ONTARIO)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == ONTARIO:
        if _OntarioExposureLimits[CASRN]["STEL (ppm)"]:
            _STEL = (_OntarioExposureLimits[CASRN]["STEL (ppm)"], 'ppm')
        elif _OntarioExposureLimits[CASRN]["STEL (mg/m^3)"]:
            _STEL = (_OntarioExposureLimits[CASRN]["STEL (mg/m^3)"], 'mg/m^3')
    elif Method == NONE:
        _STEL = None
    else:
        raise Exception('Failure in in function')
    return _STEL


def Ceiling(CASRN, AvailableMethods=False, Method=None):  # pragma: no cover
    '''This function handles the retrieval of Ceiling limits on worker
    exposure to dangerous chemicals.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> Ceiling('75-07-0')
    (25.0, 'ppm')
    >>> Ceiling('1395-21-7')
    (6e-05, 'mg/m^3')
    >>> Ceiling('7572-29-4', AvailableMethods=True)
    ['Ontario Limits', 'None']
    '''
    def list_methods():
        methods = []
        if CASRN in _OntarioExposureLimits and (_OntarioExposureLimits[CASRN]["Ceiling (ppm)"] or _OntarioExposureLimits[CASRN]["Ceiling (mg/m^3)"]):
            methods.append(ONTARIO)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == ONTARIO:
        if _OntarioExposureLimits[CASRN]["Ceiling (ppm)"]:
            _Ceiling = (_OntarioExposureLimits[CASRN]["Ceiling (ppm)"], 'ppm')
        elif _OntarioExposureLimits[CASRN]["Ceiling (mg/m^3)"]:
            _Ceiling = (_OntarioExposureLimits[CASRN]["Ceiling (mg/m^3)"], 'mg/m^3')
    elif Method == NONE:
        _Ceiling = None
    else:
        raise Exception('Failure in in function')
    return _Ceiling


def Skin(CASRN, AvailableMethods=False, Method=None):  # pragma: no cover
    '''This function handles the retrieval of whether or not a chemical can
    be absorbed through the skin, relevant to chemical safety calculations.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> Skin('108-94-1')
    True
    >>> Skin('1395-21-7')
    False
    >>> Skin('7572-29-4', AvailableMethods=True)
    ['Ontario Limits', 'None']
    '''
    def list_methods():
        methods = []
        if CASRN in _OntarioExposureLimits:
            methods.append(ONTARIO)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == ONTARIO:
        _Skin = (_OntarioExposureLimits[CASRN]["Skin"])
    elif Method == NONE:
        _Skin = None
    else:
        raise Exception('Failure in in function')
    return _Skin

### Carcinogen functions

IARC = 'International Agency for Research on Cancer'
NTP = 'National Toxicology Program 13th Report on Carcinogens'
UNLISTED = 'Unlisted'
COMBINED = 'Combined'

Carcinogen_methods = [IARC, NTP]


def Carcinogen(CASRN, AvailableMethods=False, Method=None):
    r'''Looks up if a chemical is listed as a carcinogen or not according to
    either a specifc method or with all methods.

    Returns either the status as a string for a specified method, or the
    status of the chemical in all available data sources, in the format
    {source: status}.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    status : str or dict
        Carcinogen status information [-]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain carcinogen status with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        Carcinogen_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        if a chemical is listed as carcinogenic, and will return methods
        instead of the status

    Notes
    -----
    Supported methods are:

        * **IARC**: International Agency for Research on Cancer, [1]_. As
          extracted with a last update of  February 22, 2016. Has listing
          information of 843 chemicals with CAS numbers. Chemicals without
          CAS numbers not included here. If two listings for the same CAS
          were available, that closest to the CAS number was used. If two
          listings were available published at different times, the latest
          value was used. All else equal, the most pessimistic value was used.
        * **NTP**: National Toxicology Program, [2]_. Has data on 226
          chemicals.

    Examples
    --------
    >>> Carcinogen('61-82-5')
    {'National Toxicology Program 13th Report on Carcinogens': 'Reasonably Anticipated', 'International Agency for Research on Cancer': 'Not classifiable as to its carcinogenicity to humans (3)'}

    References
    ----------
    .. [1] International Agency for Research on Cancer. Agents Classified by
       the IARC Monographs, Volumes 1-115. Lyon, France: IARC; 2016 Available
       from: http://monographs.iarc.fr/ENG/Classification/
    .. [2] NTP (National Toxicology Program). 2014. Report on Carcinogens,
       Thirteenth Edition. Research Triangle Park, NC: U.S. Department of
       Health and Human Services, Public Health Service.
       http://ntp.niehs.nih.gov/pubhealth/roc/roc13/
    '''
    methods = [COMBINED, IARC, NTP]
    if AvailableMethods:
        return methods
    if not Method:
        Method = methods[0]
    if Method == IARC:
        if CASRN in IARC_data.index:
            status = IARC_codes[IARC_data.at[CASRN, 'group']]
        else:
            status = UNLISTED
    elif Method == NTP:
        if CASRN in NTP_data.index:
            status = NTP_codes[NTP_data.at[CASRN, 'Listing']]
        else:
            status = UNLISTED
    elif Method == COMBINED:
        status = {}
        for method in methods[1:]:
            status[method] = Carcinogen(CASRN, Method=method)
    else:
        raise Exception('Failure in in function')
    return status

#print(Carcinogen2('61-82-5')) # {'National Toxicology Program 13th Report on Carcinogens': 'Reasonably Anticipated', 'International Agency for Research on Cancer': 'Not classifiable as to its carcinogenicity to humans (3)'}
#print Carcinogen2('71-43-2') # {'National Toxicology Program 13th Report on Carcinogens': 'Known', 'International Agency for Research on Cancer': 'Carcinogenic to humans (1)'}
#print Carcinogen2('7732-18-5') # {'National Toxicology Program 13th Report on Carcinogens': 'Unlisted', 'International Agency for Research on Cancer': 'Unlisted'}


### Fire-related functions


IEC = 'IEC 60079-20-1 (2010)'
NFPA = 'NFPA 497 (2008)'

Tflash_methods = [IEC, NFPA]


def Tflash(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval or calculation of a chemical's
    flash point. Lookup is based on CASRNs. No predictive methods are currently
    implemented. Will automatically select a data source to use if no Method
    is provided; returns None if the data is not available.

    Prefered source is 'IEC 60079-20-1 (2010)' [1]_, with the secondary source
    'NFPA 497 (2008)' [2]_ having very similar data.

    Examples
    --------
    >>> Tflash(CASRN='64-17-5')
    285.15

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Tflash : float
        Flash point of the chemical, [K]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Tflash with the given
        inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        Tflash_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Tflash for the desired chemical, and will return methods instead of
        Tflash

    Notes
    -----

    References
    ----------
    .. [1] IEC. “IEC 60079-20-1:2010 Explosive atmospheres - Part 20-1:
       Material characteristics for gas and vapour classification - Test
       methods and data.” https://webstore.iec.ch/publication/635. See also
       https://law.resource.org/pub/in/bis/S05/is.iec.60079.20.1.2010.pdf
    .. [2] National Fire Protection Association. NFPA 497: Recommended
       Practice for the Classification of Flammable Liquids, Gases, or Vapors
       and of Hazardous. NFPA, 2008.
    '''
    def list_methods():
        methods = []
        if CASRN in IEC_2010.index and not np.isnan(IEC_2010.at[CASRN, 'Tflash']):
            methods.append(IEC)
        if CASRN in NFPA_2008.index and not np.isnan(NFPA_2008.at[CASRN, 'Tflash']):
            methods.append(NFPA)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IEC:
        return float(IEC_2010.at[CASRN, 'Tflash'])
    elif Method == NFPA:
        return float(NFPA_2008.at[CASRN, "Tflash"])
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')


Tautoignition_methods = [IEC, NFPA]


def Tautoignition(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval or calculation of a chemical's
    autoifnition temperature. Lookup is based on CASRNs. No predictive methods
    are currently implemented. Will automatically select a data source to use
    if no Method is provided; returns None if the data is not available.

    Prefered source is 'IEC 60079-20-1 (2010)' [1]_, with the secondary source
    'NFPA 497 (2008)' [2]_ having very similar data.

    Examples
    --------
    >>> Tautoignition(CASRN='71-43-2')
    771.15

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Tautoignition : float
        Autoignition point of the chemical, [K]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Tautoignition with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        Tautoignition_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Tautoignition for the desired chemical, and will return methods
        instead of Tautoignition

    Notes
    -----

    References
    ----------
    .. [1] IEC. “IEC 60079-20-1:2010 Explosive atmospheres - Part 20-1:
       Material characteristics for gas and vapour classification - Test
       methods and data.” https://webstore.iec.ch/publication/635. See also
       https://law.resource.org/pub/in/bis/S05/is.iec.60079.20.1.2010.pdf
    .. [2] National Fire Protection Association. NFPA 497: Recommended
       Practice for the Classification of Flammable Liquids, Gases, or Vapors
       and of Hazardous. NFPA, 2008.
    '''
    def list_methods():
        methods = []
        if CASRN in IEC_2010.index and not np.isnan(IEC_2010.at[CASRN, 'Tautoignition']):
            methods.append(IEC)
        if CASRN in NFPA_2008.index and not np.isnan(NFPA_2008.at[CASRN, 'Tautoignition']):
            methods.append(NFPA)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IEC:
        return float(IEC_2010.at[CASRN, 'Tautoignition'])
    elif Method == NFPA:
        return float(NFPA_2008.at[CASRN, 'Tautoignition'])
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')


SUZUKI = 'Suzuki (1994)'
CROWLLOUVAR = 'Crowl and Louvar (2001)'

LFL_methods = [IEC, NFPA, SUZUKI, CROWLLOUVAR]


def LFL(Hc=None, atoms={}, CASRN='', AvailableMethods=False, Method=None):
    r'''This function handles the retrieval or calculation of a chemical's
    Lower Flammability Limit. Lookup is based on CASRNs. Two predictive methods
    are currently implemented. Will automatically select a data source to use
    if no Method is provided; returns None if the data is not available.

    Prefered source is 'IEC 60079-20-1 (2010)' [1]_, with the secondary source
    'NFPA 497 (2008)' [2]_ having very similar data. If the heat of combustion
    is provided, the estimation method `Suzuki_LFL` can be used. If the atoms
    of the molecule are available, the method `Crowl_Louvar_LFL` can be used.

    Examples
    --------
    >>> LFL(CASRN='71-43-2')
    0.012

    Parameters
    ----------
    Hc : float, optional
        Heat of combustion of gas [J/mol]
    atoms : dict, optional
        Dictionary of atoms and atom counts
    CASRN : string, optional
        CASRN [-]

    Returns
    -------
    LFL : float
        Lower flammability limit of the gas in an atmosphere at STP, [mole fraction]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain LFL with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        LFL_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the Lower Flammability Limit for the desired chemical, and will return
        methods instead of Lower Flammability Limit.

    Notes
    -----

    References
    ----------
    .. [1] IEC. “IEC 60079-20-1:2010 Explosive atmospheres - Part 20-1:
       Material characteristics for gas and vapour classification - Test
       methods and data.” https://webstore.iec.ch/publication/635. See also
       https://law.resource.org/pub/in/bis/S05/is.iec.60079.20.1.2010.pdf
    .. [2] National Fire Protection Association. NFPA 497: Recommended
       Practice for the Classification of Flammable Liquids, Gases, or Vapors
       and of Hazardous. NFPA, 2008.
    '''
    def list_methods():
        methods = []
        if CASRN in IEC_2010.index and not np.isnan(IEC_2010.at[CASRN, 'LFL']):
            methods.append(IEC)
        if CASRN in NFPA_2008.index and not np.isnan(NFPA_2008.at[CASRN, 'LFL']):
            methods.append(NFPA)
        if Hc:
            methods.append(SUZUKI)
        if atoms:
            methods.append(CROWLLOUVAR)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IEC:
        return float(IEC_2010.at[CASRN, 'LFL'])
    elif Method == NFPA:
        return float(NFPA_2008.at[CASRN, 'LFL'])
    elif Method == SUZUKI:
        return Suzuki_LFL(Hc=Hc)
    elif Method == CROWLLOUVAR:
        return Crowl_Louvar_LFL(atoms=atoms)
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')


UFL_methods = [IEC, NFPA, SUZUKI, CROWLLOUVAR]


def UFL(Hc=None, atoms={}, CASRN='', AvailableMethods=False, Method=None):
    r'''This function handles the retrieval or calculation of a chemical's
    Upper Flammability Limit. Lookup is based on CASRNs. Two predictive methods
    are currently implemented. Will automatically select a data source to use
    if no Method is provided; returns None if the data is not available.

    Prefered source is 'IEC 60079-20-1 (2010)' [1]_, with the secondary source
    'NFPA 497 (2008)' [2]_ having very similar data. If the heat of combustion
    is provided, the estimation method `Suzuki_UFL` can be used. If the atoms
    of the molecule are available, the method `Crowl_Louvar_UFL` can be used.

    Examples
    --------
    >>> UFL(CASRN='71-43-2')
    0.086

    Parameters
    ----------
    Hc : float, optional
        Heat of combustion of gas [J/mol]
    atoms : dict, optional
        Dictionary of atoms and atom counts
    CASRN : string, optional
        CASRN [-]

    Returns
    -------
    UFL : float
        Upper flammability limit of the gas in an atmosphere at STP, [mole fraction]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain UFL with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        UFL_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the Upper Flammability Limit for the desired chemical, and will return
        methods instead of Upper Flammability Limit.

    Notes
    -----

    References
    ----------
    .. [1] IEC. “IEC 60079-20-1:2010 Explosive atmospheres - Part 20-1:
       Material characteristics for gas and vapour classification - Test
       methods and data.” https://webstore.iec.ch/publication/635. See also
       https://law.resource.org/pub/in/bis/S05/is.iec.60079.20.1.2010.pdf
    .. [2] National Fire Protection Association. NFPA 497: Recommended
       Practice for the Classification of Flammable Liquids, Gases, or Vapors
       and of Hazardous. NFPA, 2008.
    '''
    def list_methods():
        methods = []
        if CASRN in IEC_2010.index and not np.isnan(IEC_2010.at[CASRN, 'UFL']):
            methods.append(IEC)
        if CASRN in NFPA_2008.index and not np.isnan(NFPA_2008.at[CASRN, 'UFL']):
            methods.append(NFPA)
        if Hc:
            methods.append(SUZUKI)
        if atoms:
            methods.append(CROWLLOUVAR)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IEC:
        return float(IEC_2010.at[CASRN, 'UFL'])
    elif Method == NFPA:
        return float(NFPA_2008.at[CASRN, 'UFL'])
    elif Method == SUZUKI:
        return Suzuki_UFL(Hc=Hc)
    elif Method == CROWLLOUVAR:
        return Crowl_Louvar_UFL(atoms=atoms)
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')


def fire_mixing(ys=None, FLs=None):  # pragma: no cover
    '''
    Crowl, Daniel A., and Joseph F. Louvar. Chemical Process Safety:
    Fundamentals with Applications. 2E. Upper Saddle River, N.J:
    Prentice Hall, 2001.


    >>> fire_mixing(ys=normalize([0.0024, 0.0061, 0.0015]), FLs=[.012, .053, .031])
    0.02751172136637643
    >>> fire_mixing(ys=normalize([0.0024, 0.0061, 0.0015]), FLs=[.075, .15, .32])
    0.12927551844869378
    '''
    return 1./sum([yi/FLi for yi, FLi in zip(ys, FLs)])

#print (fire_mixing(ys=[0.0024, 0.0061, 0.0015], FLs=[.075, .15, .32]), 0)
inerts = {"7440-37-1": "Argon", "124-38-9": "Carbon Dioxide", "7440-59-7":
          "Helium", "7440-01-9": "Neon", "7727-37-9": "Nitrogen",
          "7440-63-3": "Xenon", "10102-43-9": "Nitric Oxide", "10102-44-0":
          "Nitrogen Dioxide", "7782-44-7": "Oxygen", "132259-10-0": "Air",
          "7439-90-9": "krypton", "10043-92-2": "radon", "7732-18-5":
          "water", "7782-50-5": "chlorine", "7782-41-4": "fluorine"}


def LFL_mixture(ys=None, LFLs=None, CASRNs=None, AvailableMethods=False,
                Method=None):  # pragma: no cover
    '''Inert gases are ignored.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> LFL_mixture(ys=normalize([0.0024, 0.0061, 0.0015]), LFLs=[.012, .053, .031])
    0.02751172136637643
    >>> LFL_mixture(LFLs=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.025, 0.06, 0.073, 0.020039, 0.011316], ys=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], CASRNs=['7440-37-1', '124-38-9', '7440-59-7', '7440-01-9', '7727-37-9', '7440-63-3', '10102-43-9', '7782-44-7', '132259-10-0', '7439-90-9', '10043-92-2', '7732-18-5', '7782-50-5', '7782-41-4', '67-64-1', '67-56-1', '75-52-5', '590-19-2', '277-10-1'])
    0.023964903630937385
    '''
    def list_methods():
        methods = []
        if CASRNs:
            CASRNs2 = list(CASRNs)
            LFLs2 = list(LFLs)
            for i in inerts:
                if i in CASRNs2:
                    ind = CASRNs.index(i)
                    CASRNs2.remove(i)
                    LFLs2.remove(LFLs[ind])
            if none_and_length_check([LFLs2]):
                methods.append('Summed Inverse, inerts removed')
        else:
            if none_and_length_check([LFLs]):
                methods.append('Summed Inverse')
        methods.append('None')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
#    if not none_and_length_check([LFLs, ys]):
#        raise Exception('Function inputs are incorrect format')
    if Method == 'Summed Inverse':
        return fire_mixing(ys, LFLs)
    elif Method == 'Summed Inverse, inerts removed':
        CASRNs2 = list(CASRNs)
        LFLs2 = list(LFLs)
        ys2 = list(ys)
        for i in inerts:
            if i in CASRNs2:
                ind = CASRNs2.index(i)
                CASRNs2.remove(i)
                LFLs2.pop(ind)
                ys2.pop(ind)
        return fire_mixing(normalize(ys2), LFLs2)
    elif Method == 'None':
        return None
    else:
        raise Exception('Failure in in function')


def UFL_mixture(ys=None, UFLs=None, CASRNs=None, AvailableMethods=False,
                Method=None):  # pragma: no cover
    '''Inert gases are ignored.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> UFL_mixture(ys=normalize([0.0024, 0.0061, 0.0015]), UFLs=[.075, .15, .32])
    0.12927551844869378
    >>> LFL_mixture(LFLs=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.143, 0.36, 0.63, 0.1097, 0.072], ys=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], CASRNs=['7440-37-1', '124-38-9', '7440-59-7', '7440-01-9', '7727-37-9', '7440-63-3', '10102-43-9', '7782-44-7', '132259-10-0', '7439-90-9', '10043-92-2', '7732-18-5', '7782-50-5', '7782-41-4', '67-64-1', '67-56-1', '75-52-5', '590-19-2', '277-10-1'])
    0.14550641757359664
    '''
    def list_methods():
        methods = []
        if CASRNs:
            CASRNs2 = list(CASRNs)
            UFLs2 = list(UFLs)
            for i in inerts:
                if i in CASRNs2:
                    ind = CASRNs.index(i)
                    CASRNs2.remove(i)
                    UFLs2.remove(UFLs[ind])
            if none_and_length_check([UFLs2]):
                methods.append('Summed Inverse, inerts removed')
        if none_and_length_check([UFLs, ys]):
            methods.append('Summed Inverse')
        methods.append('None')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
#    if not none_and_length_check([UFLs, ys]):  # check same-length inputs
#        raise Exception('Function inputs are incorrect format')
    if Method == 'Summed Inverse':
        return fire_mixing(ys, UFLs)
    elif Method == 'Summed Inverse, inerts removed':
        CASRNs2 = list(CASRNs)
        UFLs2 = list(UFLs)
        ys2 = list(ys)
        for i in inerts:
            if i in CASRNs2:
                ind = CASRNs2.index(i)
                CASRNs2.remove(i)
                UFLs2.pop(ind)
                ys2.pop(ind)
        return fire_mixing(normalize(ys2), UFLs2)
    elif Method == 'None':
        return None
    else:
        raise Exception('Failure in in function')


def Suzuki_LFL(Hc=None):
    r'''Calculates lower flammability limit, using the Suzuki [1]_ correlation.
    Uses heat of combustion only.

    The lower flammability limit of a gas is air is:

    .. math::
        \text{LFL} = \frac{-3.42}{\Delta H_c^{\circ}} + 0.569
        \Delta H_c^{\circ} + 0.0538\Delta H_c^{\circ 2} + 1.80

    Parameters
    ----------
        Hc : float
            Heat of combustion of gas [J/mol]

    Returns
    -------
        LFL : float
            Lower flammability limit, mole fraction

    Notes
    -----
    Fit performed with 112 compounds, r^2 was 0.977.
    LFL in percent volume in air. Hc is at standard conditions, in MJ/mol.
    11 compounds left out as they were outliers.
    Equation does not apply for molecules with halogen atoms, only hydrocarbons
    with oxygen or nitrogen or sulfur.
    No sample calculation provided with the article. However, the equation is
    straightforward.

    Limits of equations's validity are -6135596 J where it predicts a
    LFL of 0, and -48322129 J where it predicts a LFL of 1.

    Examples
    --------
    Pentane, 1.5 % LFL in literature

    >>> Suzuki_LFL(-3536600)
    0.014276107095811815

    References
    ----------
    .. [1] Suzuki, Takahiro. "Note: Empirical Relationship between Lower
       Flammability Limits and Standard Enthalpies of Combustion of Organic
       Compounds." Fire and Materials 18, no. 5 (September 1, 1994): 333-36.
       doi:10.1002/fam.810180509.
    '''
    Hc = Hc/1E6
    LFL = -3.42/Hc + 0.569*Hc + 0.0538*Hc*Hc + 1.80
    return LFL/100.


def Suzuki_UFL(Hc=None):
    r'''Calculates upper flammability limit, using the Suzuki [1]_ correlation.
    Uses heat of combustion only.

    The upper flammability limit of a gas is air is:

    .. math::
        \text{UFL} = 6.3\Delta H_c^\circ + 0.567\Delta H_c^{\circ 2} + 23.5

    Parameters
    ----------
        Hc : float
            Heat of combustion of gas [J/mol]

    Returns
    -------
        UFL : float
            Upper flammability limit, mole fraction

    Notes
    -----
    UFL in percent volume in air according to original equation.
    Hc is at standard conditions in the equation, in units of MJ/mol.
    AAPD = 1.2% for 95 compounds used in fit.
    Somewhat better results than the High and Danner method.
    4.9% < UFL < 23.0%
    -890.3 kJ/mol < dHc < -6380 kJ/mol
    r^2 = 0.989
    Sample calculations provided for all chemicals, both this method and
    High and Danner. Examples are from the article.

    Predicts a UFL of 1 at 7320190 J and a UFL of 0 at -5554160 J.

    Examples
    --------
    Pentane, literature 7.8% UFL

    >>> Suzuki_UFL(-3536600)
    0.0831119493052

    References
    ----------
    .. [1] Suzuki, Takahiro, and Kozo Koide. "Short Communication: Correlation
       between Upper Flammability Limits and Thermochemical Properties of
       Organic Compounds." Fire and Materials 18, no. 6 (November 1, 1994):
       393-97. doi:10.1002/fam.810180608.
    '''
    Hc = Hc/1E6
    UFL = 6.3*Hc + 0.567*Hc*Hc + 23.5
    return UFL/100.


def Crowl_Louvar_LFL(atoms):
    r'''Calculates lower flammability limit, using the Crowl-Louvar [1]_
    correlation. Uses molecular formula only.

    The lower flammability limit of a gas is air is:

    .. math::
        C_mH_xO_y + zO_2 \to mCO_2 + \frac{x}{2}H_2O

        \text{LFL} = \frac{0.55}{4.76m + 1.19x - 2.38y + 1}

    Parameters
    ----------
        atoms : dict
            Dictionary of atoms and atom counts

    Returns
    -------
        LFL : float
            Lower flammability limit, mole fraction

    Notes
    -----
    Coefficient of 0.55 taken from [2]_

    Examples
    --------
    Hexane, example from [1]_, lit. 1.2 %

    >>> Crowl_Louvar_LFL({'H': 14, 'C': 6})
    0.011899610558199915

    References
    ----------
    .. [1] Crowl, Daniel A., and Joseph F. Louvar. Chemical Process Safety:
       Fundamentals with Applications. 2E. Upper Saddle River, N.J:
       Prentice Hall, 2001.
    .. [2] Jones, G. W. "Inflammation Limits and Their Practical Application
       in Hazardous Industrial Operations." Chemical Reviews 22, no. 1
       (February 1, 1938): 1-26. doi:10.1021/cr60071a001
    '''
    nC, nH, nO = 0, 0, 0
    if 'C' in atoms and atoms['C']:
        nC = atoms['C']
    else:
        return None
    if 'H' in atoms:
        nH = atoms['H']
    if 'O' in atoms:
        nO = atoms['O']
    return 0.55/(4.76*nC + 1.19*nH - 2.38*nO + 1.)


def Crowl_Louvar_UFL(atoms):
    r'''Calculates upper flammability limit, using the Crowl-Louvar [1]_
    correlation. Uses molecular formula only.

    The upper flammability limit of a gas is air is:

    .. math::
        C_mH_xO_y + zO_2 \to mCO_2 + \frac{x}{2}H_2O

        \text{UFL} = \frac{3.5}{4.76m + 1.19x - 2.38y + 1}

    Parameters
    ----------
        atoms : dict
            Dictionary of atoms and atom counts

    Returns
    -------
        UFL : float
            Upper flammability limit, mole fraction

    Notes
    -----
    Coefficient of 3.5 taken from [2]_

    Examples
    --------
    Hexane, example from [1]_, lit. 7.5 %

    >>> Crowl_Louvar_UFL({'H': 14, 'C': 6})
    0.07572479446127219

    References
    ----------
    .. [1] Crowl, Daniel A., and Joseph F. Louvar. Chemical Process Safety:
       Fundamentals with Applications. 2E. Upper Saddle River, N.J:
       Prentice Hall, 2001.
    .. [2] Jones, G. W. "Inflammation Limits and Their Practical Application
       in Hazardous Industrial Operations." Chemical Reviews 22, no. 1
       (February 1, 1938): 1-26. doi:10.1021/cr60071a001
    '''
    nC, nH, nO = 0, 0, 0
    if 'C' in atoms and atoms['C']:
        nC = atoms['C']
    else:
        return None
    if 'H' in atoms:
        nH = atoms['H']
    if 'O' in atoms:
        nO = atoms['O']
    return 3.5/(4.76*nC + 1.19*nH - 2.38*nO + 1.)


#CAS = ['71-43-2', '8006-61-9'] # methanol, gasoline
#for i in CAS:
#    print Tflash(i)
#    print Tautoignition(i)
#    print UFL(i)
#    print LFL(i)
#
#print Tautoignition('8006-61-9', AvailableMethods=True)

#
#def flammable(ys=None, LFLs=None, UFLs=None, CASRNs=None):
#    return 'TODO - determine how to handle nitrogen, oxygen concentrations'

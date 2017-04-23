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

__all__ = ['DSL_data', 'CAN_DSL_flags', 'TSCA_flags', 'TSCA_data', 
           'EINECS_data', 'SPIN_data', 'NLP_data', 'legal_status_methods', 
           'legal_status', 'HPV_data', '_ECHATonnageDict', '_EPACDRDict', 
           'economic_status', 'economic_status_methods', 'load_economic_data',
           'load_law_data']

import os
import zipfile
from thermo.utils import to_num, CAS2int
import pandas as pd
from pprint import pprint


folder = os.path.join(os.path.dirname(__file__), 'Law')

DSL = 'DSL'
TSCA = 'TSCA'
EINECS = 'EINECS'
NLP = 'NLP'
SPIN = 'SPIN'
NONE = 'NONE'
COMBINED = 'COMBINED'
UNLISTED = 'UNLISTED'
LISTED = 'LISTED'




CAN_DSL_flags = {0: LISTED,
                 1: 'Non-Domestic Substances List (NDSL)',
                 2: 'Significant New Activity (SNAc)',
                 3: 'Ministerial Condition pertaining to this substance',
                 4: 'Domestic Substances List, removed (DSL_REM)',
                 5: 'Minister of the Environment has imposed a Ministerial \
                 Prohibition pertaining to this substance'}


TSCA_flags = {
    'UV': 'Class 2 substance within the UVCB group; unknown molecular formula/structural diagram',
    'E': 'subject of a Section 5(e) Consent Order under TSCA',
    'F': 'subject of a Section 5(f) Rule under TSCA',
    'N': 'polymeric substance containing no free-radical initiator in its Inventory name but is considered to cover the designated polymer made with any free-radical initiator regardless of the amount used',
    'P': 'commenced Premanufacture Notice (PMN) substance',
    'R': 'subject of a Section 6 risk management rule under TSCA',
    'S': 'identified in a final Significant New Uses Rule',
    'SP': 'identified in a proposed Significant New Uses Rule',
    'T': 'subject of a final Section 4 test rule under TSCA',
    'TP': 'subject of a proposed Section 4 test rule under TSCA',
    'XU': 'exempt from reporting under Chemical Date Reporting Rule (formerly the Inventory Update Reporting Rule), i.e., Partial Updating of the TSCA Inventory Data Base Production and Site Reports (40 CFR 711)',
    'Y1': 'exempt polymer that has a number-average molecular weight of 1,000 or greater',
    'Y2': 'exempt polymer that is a polyester and is made only from reactants included in a specified list of low-concern reactants that comprises one of the eligibility criteria for the exemption rule'
}

DSL_data, TSCA_data, EINECS_data, SPIN_data, NLP_data = [None]*5

def load_law_data():
    global DSL_data
    if DSL_data is not None:
        return None
    global TSCA_data, EINECS_data, SPIN_data, NLP_data

# Data is stored as integers to reduce memory usage
    DSL_data = pd.read_csv(os.path.join(folder, 'Canada Feb 11 2015 - DSL.csv.gz'),
                           sep='\t', index_col=0, compression='gzip')
    
    TSCA_data = pd.read_csv(os.path.join(folder, 'TSCA Inventory 2016-01.csv.gz'),
                           sep='\t', index_col=0, compression='gzip')
    
    
    EINECS_data = pd.read_csv(os.path.join(folder, 'EINECS 2015-03.csv.gz'),
                              index_col=0, compression='gzip')
    
    SPIN_data = pd.read_csv(os.path.join(folder, 'SPIN Inventory 2015-03.csv.gz'),
                           compression='gzip', index_col=0)
    
    NLP_data = pd.read_csv(os.path.join(folder, 'EC Inventory No Longer Polymers (NLP).csv'),
                           sep='\t', index_col=0)
    # 161162-67-6 is not a valid CAS number and was removed.

legal_status_methods = [COMBINED, DSL, TSCA, EINECS, SPIN, NLP]


def legal_status(CASRN, Method=None, AvailableMethods=False, CASi=None):
    r'''Looks up the legal status of a chemical according to either a specifc
    method or with all methods.

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
        Legal status information [-]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain legal status with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        legal_status_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the legal status for the desired chemical, and will return methods
        instead of the status
    CASi : int, optional
        CASRN as an integer, used internally [-]

    Notes
    -----

    Supported methods are:

        * **DSL**: Canada Domestic Substance List, [1]_. As extracted on Feb 11, 2015
          from a html list. This list is updated continuously, so this version
          will always be somewhat old. Strictly speaking, there are multiple
          lists but they are all bundled together here. A chemical may be
          'Listed', or be on the 'Non-Domestic Substances List (NDSL)',
          or be on the list of substances with 'Significant New Activity (SNAc)',
          or be on the DSL but with a 'Ministerial Condition pertaining to this
          substance', or have been removed from the DSL, or have had a
          Ministerial prohibition for the substance.
        * **TSCA**: USA EPA Toxic Substances Control Act Chemical Inventory, [2]_.
          This list is as extracted on 2016-01. It is believed this list is
          updated on a periodic basis (> 6 month). A chemical may simply be
          'Listed', or may have certain flags attached to it. All these flags
          are described in the dict TSCA_flags.
        * **EINECS**: European INventory of Existing Commercial chemical
          Substances, [3]_. As extracted from a spreadsheet dynamically
          generated at [1]_. This list was obtained March 2015; a more recent
          revision already exists.
        * **NLP**: No Longer Polymers, a list of chemicals with special
          regulatory exemptions in EINECS. Also described at [3]_.
        * **SPIN**: Substances Prepared in Nordic Countries. Also a boolean
          data type. Retrieved 2015-03 from [4]_.

    Other methods which could be added are:

        * Australia: AICS Australian Inventory of Chemical Substances
        * China: Inventory of Existing Chemical Substances Produced or Imported
          in China (IECSC)
        * Europe: REACH List of Registered Substances
        * India: List of Hazardous Chemicals
        * Japan: ENCS: Inventory of existing and new chemical substances
        * Korea: Existing Chemicals Inventory (KECI)
        * Mexico: INSQ National Inventory of Chemical Substances in Mexico
        * New Zealand:  Inventory of Chemicals (NZIoC)
        * Philippines: PICCS Philippines Inventory of Chemicals and Chemical
          Substances

    Examples
    --------
    >>> pprint(legal_status('64-17-5'))
    {'DSL': 'LISTED',
     'EINECS': 'LISTED',
     'NLP': 'UNLISTED',
     'SPIN': 'LISTED',
     'TSCA': 'LISTED'}

    References
    ----------
    .. [1] Government of Canada.. "Substances Lists" Feb 11, 2015.
       https://www.ec.gc.ca/subsnouvelles-newsubs/default.asp?n=47F768FE-1.
    .. [2] US EPA. "TSCA Chemical Substance Inventory." Accessed April 2016.
       https://www.epa.gov/tsca-inventory.
    .. [3] ECHA. "EC Inventory". Accessed March 2015.
       http://echa.europa.eu/information-on-chemicals/ec-inventory.
    .. [4] SPIN. "SPIN Substances in Products In Nordic Countries." Accessed
       March 2015. http://195.215.202.233/DotNetNuke/default.aspx.
    '''
    load_law_data()
    if not CASi:
        CASi = CAS2int(CASRN)
    methods = [COMBINED, DSL, TSCA, EINECS, NLP, SPIN]
    if AvailableMethods:
        return methods
    if not Method:
        Method = methods[0]
    if Method == DSL:
        if CASi in DSL_data.index:
            status = CAN_DSL_flags[DSL_data.at[CASi, 'Registry']]
        else:
            status = UNLISTED
    elif Method == TSCA:
        if CASi in TSCA_data.index:
            data = TSCA_data.loc[CASi].to_dict()
            if any(data.values()):
                status = sorted([TSCA_flags[i] for i in data.keys() if data[i]])
            else:
                status = LISTED
        else:
            status = UNLISTED
    elif Method == EINECS:
        if CASi in EINECS_data.index:
            status = LISTED
        else:
            status = UNLISTED
    elif Method == NLP:
        if CASi in NLP_data.index:
            status = LISTED
        else:
            status = UNLISTED
    elif Method == SPIN:
        if CASi in SPIN_data.index:
            status = LISTED
        else:
            status = UNLISTED
    elif Method == COMBINED:
        status = {}
        for method in methods[1:]:
            status[method] = legal_status(CASRN, Method=method, CASi=CASi)
    else:
        raise Exception('Failure in in function')
    return status

#print  legal_status(CASRN='64-17-5')
#for i in [DSL, TSCA, EINECS, SPIN, NLP]:
#    print  legal_status(CASRN='64-17-5', Method=i)
#print 'hi'

#print legal_status(CASRN='13775-50-3', Method=DSL)
#print legal_status(CASRN='52-89-1')
# _ECHATonnageDict, _EPACDRDict
# 2.135340690612793, 3.499225616455078
#print legal_status(CASRN='1648727-81-4')



HPV_data, _EPACDRDict, _ECHATonnageDict = [None]*3
                                          
def load_economic_data():
    global HPV_data
    if HPV_data is not None:
        return None
    global _EPACDRDict, _ECHATonnageDict
    
    '''OECD are chemicals produced by and OECD members in > 1000 tonnes/year.'''
    HPV_data = pd.read_csv(os.path.join(folder, 'HPV 2015 March 3.csv'),
                           sep='\t', index_col=0)
    # 13061-29-2 not valid and removed
    
    _ECHATonnageDict = {}
    with zipfile.ZipFile(os.path.join(folder, 'ECHA Tonnage Bands.csv.zip')) as z:
        with z.open(z.namelist()[0]) as f:
            for line in f.readlines():
                # for some reason, the file must be decoded to UTF8 first
                CAS, band = line.decode("utf-8").strip('\n').split('\t')
                if CAS in _ECHATonnageDict:
                    if band in _ECHATonnageDict[CAS]:
                        pass
                    else:
                        _ECHATonnageDict[CAS].append(band)
                else:
                    _ECHATonnageDict[CAS] = [band]
    
    
    _EPACDRDict = {}
    with open(os.path.join(folder, 'EPA 2012 Chemical Data Reporting.csv')) as f:
        '''EPA summed reported chemical usages. In metric tonnes/year after conversion.
        Many producers keep their date confidential.
        This was originally in terms of lb/year, but rounded to the nearest kg.
    
        '''
        next(f)
        for line in f:
            values = line.rstrip().split('\t')
            CAS, manufactured, imported, exported = to_num(values)
            _EPACDRDict[CAS] = {"Manufactured": manufactured/1000., "Imported": imported/1000.,
                                "Exported": exported/1000.}


EPACDR = 'EPA Chemical Data Reporting (2012)'
ECHA = 'European Chemicals Agency Total Tonnage Bands'
OECD = 'OECD high production volume chemicals'

economic_status_methods = [EPACDR, ECHA, OECD]


def economic_status(CASRN, Method=None, AvailableMethods=False):  # pragma: no cover
    '''Look up the economic status of a chemical.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> pprint(economic_status(CASRN='98-00-0'))
    ["US public: {'Manufactured': 0.0, 'Imported': 10272.711, 'Exported': 184.127}",
     u'10,000 - 100,000 tonnes per annum',
     'OECD HPV Chemicals']

    >>> economic_status(CASRN='13775-50-3')  # SODIUM SESQUISULPHATE
    []
    >>> economic_status(CASRN='98-00-0', Method='OECD high production volume chemicals')
    'OECD HPV Chemicals'
    >>> economic_status(CASRN='98-01-1', Method='European Chemicals Agency Total Tonnage Bands')
    [u'10,000 - 100,000 tonnes per annum']
    '''
    load_economic_data()
    CASi = CAS2int(CASRN)

    def list_methods():
        methods = []
        methods.append('Combined')
        if CASRN in _EPACDRDict:
            methods.append(EPACDR)
        if CASRN in _ECHATonnageDict:
            methods.append(ECHA)
        if CASi in HPV_data.index:
            methods.append(OECD)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == EPACDR:
        status = 'US public: ' + str(_EPACDRDict[CASRN])
    elif Method == ECHA:
        status = _ECHATonnageDict[CASRN]
    elif Method == OECD:
        status = 'OECD HPV Chemicals'
    elif Method == 'Combined':
        status = []
        if CASRN in _EPACDRDict:
            status += ['US public: ' + str(_EPACDRDict[CASRN])]
        if CASRN in _ECHATonnageDict:
            status += _ECHATonnageDict[CASRN]
        if CASi in HPV_data.index:
            status += ['OECD HPV Chemicals']
    elif Method == NONE:
        status = None
    else:
        raise Exception('Failure in in function')
    return status


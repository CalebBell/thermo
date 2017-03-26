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

__all__ = ['Tc', 'Pc', 'Vc', 'Zc', 'third_property', 'critical_surface', 
           'Ihmels', 'Meissner', 'Grigoras', 'Li', 
           'Chueh_Prausnitz_Tc', 'Grieves_Thodos', 'modified_Wilson_Tc', 
           'Tc_mixture', 'Pc_mixture', 'Chueh_Prausnitz_Vc', 
           'modified_Wilson_Vc', 'Vc_mixture']
__all__.extend(['Tc_methods', 'Pc_methods', 'Vc_methods', 'Zc_methods', 
                'critical_surface_methods', '_crit_IUPAC', '_crit_Matthews', 
                '_crit_CRC', '_crit_PSRKR4', '_crit_PassutDanner', '_crit_Yaws'])

import os
import numpy as np
import pandas as pd
from thermo.utils import R
from thermo.utils import log
from thermo.utils import mixing_simple, none_and_length_check


folder = os.path.join(os.path.dirname(__file__), 'Critical Properties')


### Read the various data files

# IUPAC Organic data series
# TODO: 12E of this data http://pubsdc3.acs.org/doi/10.1021/acs.jced.5b00571

_crit_IUPAC = pd.read_csv(os.path.join(folder, 'IUPACOrganicCriticalProps.tsv'),
                          sep='\t', index_col=0)

_crit_Matthews = pd.read_csv(os.path.join(folder,
'Mathews1972InorganicCriticalProps.tsv'), sep='\t', index_col=0)

# CRC Handbook from TRC Organic data section (only in 2015)
# No Inorganic table was taken, although it is already present;
# data almost all from IUPAC
_crit_CRC = pd.read_csv(os.path.join(folder,
'CRCCriticalOrganics.tsv'), sep='\t', index_col=0)
_crit_CRC['Zc'] = pd.Series(_crit_CRC['Pc']*_crit_CRC['Vc']/_crit_CRC['Tc']/R,
 index=_crit_CRC.index)


_crit_PSRKR4 = pd.read_csv(os.path.join(folder,
'Appendix to PSRK Revision 4.tsv'), sep='\t', index_col=0)
_crit_PSRKR4['Zc'] = pd.Series(_crit_PSRKR4['Pc']*_crit_PSRKR4['Vc']/_crit_PSRKR4['Tc']/R,
                             index=_crit_PSRKR4.index)


_crit_PassutDanner = pd.read_csv(os.path.join(folder, 'PassutDanner1973.tsv'),
                                 sep='\t', index_col=0)


_crit_Yaws = pd.read_csv(os.path.join(folder, 'Yaws Collection.tsv'),
                         sep='\t', index_col=0)
_crit_Yaws['Zc'] = pd.Series(_crit_Yaws['Pc']*_crit_Yaws['Vc']/_crit_Yaws['Tc']/R,
                             index=_crit_Yaws.index)

### Strings defining each method

IUPAC = 'IUPAC'
MATTHEWS = 'MATTHEWS'
CRC = 'CRC'
PSRK = 'PSRK'
PD = 'PD'
YAWS = 'YAWS'
SURF = 'SURF'
NONE = 'NONE'
Tc_methods = [IUPAC, MATTHEWS, CRC, PSRK, PD, YAWS, SURF]


def Tc(CASRN, AvailableMethods=False, Method=None, IgnoreMethods=[SURF]):
    r'''This function handles the retrieval of a chemical's critical
    temperature. Lookup is based on CASRNs. Will automatically select a data
    source to use if no Method is provided; returns None if the data is not
    available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'MATTHEWS' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Tc : float
        Critical temperature, [K]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Tc with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'MATTHEWS', 
        'CRC', 'PSRK', 'PD', 'YAWS', and 'SURF'. All valid values are also held  
        in the list `Tc_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Tc for the desired chemical, and will return methods instead of Tc
    IgnoreMethods : list, optional
        A list of methods to ignore in obtaining the full list of methods,
        useful for for performance reasons and ignoring inaccurate methods

    Notes
    -----
    A total of seven sources are available for this function. They are:

        * 'IUPAC Organic Critical Properties', a series of critically evaluated
          experimental datum for organic compounds in [1]_, [2]_, [3]_, [4]_,
          [5]_, [6]_, [7]_, [8]_, [9]_, [10]_, [11]_, and [12]_.
        * 'Matthews Inorganic Critical Properties', a series of critically
          evaluated data for inorganic compounds in [13]_.
        * 'CRC Organic Critical Properties', a compillation of critically
          evaluated data by the TRC as published in [14]_.
        * 'PSRK Revision 4 Appendix', a compillation of experimental and
          estimated data published in [15]_.
        * 'Passut Danner 1973 Critical Properties', an older compillation of
          data published in [16]_
        * 'Yaws Critical Properties', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [17]_.
        * Critical Surface', an estimation method using a
          simple quadratic method for estimating Tc from Pc and Vc. This is
          ignored and not returned as a method by default, as no compounds
          have values of Pc and Vc but not Tc currently.

    Examples
    --------
    >>> Tc(CASRN='64-17-5')
    514.0

    References
    ----------
    .. [1] Ambrose, Douglas, and Colin L. Young. "Vapor-Liquid Critical
       Properties of Elements and Compounds. 1. An Introductory Survey."
       Journal of Chemical & Engineering Data 41, no. 1 (January 1, 1996):
       154-154. doi:10.1021/je950378q.
    .. [2] Ambrose, Douglas, and Constantine Tsonopoulos. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 2. Normal Alkanes."
       Journal of Chemical & Engineering Data 40, no. 3 (May 1, 1995): 531-46.
       doi:10.1021/je00019a001.
    .. [3] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 3. Aromatic
       Hydrocarbons." Journal of Chemical & Engineering Data 40, no. 3
       (May 1, 1995): 547-58. doi:10.1021/je00019a002.
    .. [4] Gude, Michael, and Amyn S. Teja. "Vapor-Liquid Critical Properties
       of Elements and Compounds. 4. Aliphatic Alkanols." Journal of Chemical
       & Engineering Data 40, no. 5 (September 1, 1995): 1025-36.
       doi:10.1021/je00021a001.
    .. [5] Daubert, Thomas E. "Vapor-Liquid Critical Properties of Elements
       and Compounds. 5. Branched Alkanes and Cycloalkanes." Journal of
       Chemical & Engineering Data 41, no. 3 (January 1, 1996): 365-72.
       doi:10.1021/je9501548.
    .. [6] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 6. Unsaturated Aliphatic
       Hydrocarbons." Journal of Chemical & Engineering Data 41, no. 4
       (January 1, 1996): 645-56. doi:10.1021/je9501999.
    .. [7] Kudchadker, Arvind P., Douglas Ambrose, and Constantine Tsonopoulos.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 7. Oxygen
       Compounds Other Than Alkanols and Cycloalkanols." Journal of Chemical &
       Engineering Data 46, no. 3 (May 1, 2001): 457-79. doi:10.1021/je0001680.
    .. [8] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 8. Organic Sulfur,
       Silicon, and Tin Compounds (C + H + S, Si, and Sn)." Journal of Chemical
       & Engineering Data 46, no. 3 (May 1, 2001): 480-85.
       doi:10.1021/je000210r.
    .. [9] Marsh, Kenneth N., Colin L. Young, David W. Morton, Douglas Ambrose,
       and Constantine Tsonopoulos. "Vapor-Liquid Critical Properties of
       Elements and Compounds. 9. Organic Compounds Containing Nitrogen."
       Journal of Chemical & Engineering Data 51, no. 2 (March 1, 2006):
       305-14. doi:10.1021/je050221q.
    .. [10] Marsh, Kenneth N., Alan Abramson, Douglas Ambrose, David W. Morton,
       Eugene Nikitin, Constantine Tsonopoulos, and Colin L. Young.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 10. Organic
       Compounds Containing Halogens." Journal of Chemical & Engineering Data
       52, no. 5 (September 1, 2007): 1509-38. doi:10.1021/je700336g.
    .. [11] Ambrose, Douglas, Constantine Tsonopoulos, and Eugene D. Nikitin.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 11. Organic
       Compounds Containing B + O; Halogens + N, + O, + O + S, + S, + Si;
       N + O; and O + S, + Si." Journal of Chemical & Engineering Data 54,
       no. 3 (March 12, 2009): 669-89. doi:10.1021/je800580z.
    .. [12] Ambrose, Douglas, Constantine Tsonopoulos, Eugene D. Nikitin, David
       W. Morton, and Kenneth N. Marsh. "Vapor-Liquid Critical Properties of
       Elements and Compounds. 12. Review of Recent Data for Hydrocarbons and
       Non-Hydrocarbons." Journal of Chemical & Engineering Data, October 5,
       2015, 151005081500002. doi:10.1021/acs.jced.5b00571.
    .. [13] Mathews, Joseph F. "Critical Constants of Inorganic Substances."
       Chemical Reviews 72, no. 1 (February 1, 1972): 71-100.
       doi:10.1021/cr60275a004.
    .. [14] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics, 95E. Boca Raton, FL: CRC press, 2014.
    .. [15] Horstmann, Sven, Anna Jabłoniec, Jörg Krafczyk, Kai Fischer, and
       Jürgen Gmehling. "PSRK Group Contribution Equation of State:
       Comprehensive Revision and Extension IV, Including Critical Constants
       and Α-Function Parameters for 1000 Components." Fluid Phase Equilibria
       227, no. 2 (January 25, 2005): 157-64. doi:10.1016/j.fluid.2004.11.002.
    .. [16] Passut, Charles A., and Ronald P. Danner. "Acentric Factor. A
       Valuable Correlating Parameter for the Properties of Hydrocarbons."
       Industrial & Engineering Chemistry Process Design and Development 12,
       no. 3 (July 1, 1973): 365–68. doi:10.1021/i260047a026.
    .. [17] Yaws, Carl L. Thermophysical Properties of Chemicals and
       Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
       Publishing, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in _crit_IUPAC.index and not np.isnan(_crit_IUPAC.at[CASRN, 'Tc']):
            methods.append(IUPAC)
        if CASRN in _crit_Matthews.index and not np.isnan(_crit_Matthews.at[CASRN, 'Tc']):
            methods.append(MATTHEWS)
        if CASRN in _crit_CRC.index and not np.isnan(_crit_CRC.at[CASRN, 'Tc']):
            methods.append(CRC)
        if CASRN in _crit_PSRKR4.index and not np.isnan(_crit_PSRKR4.at[CASRN, 'Tc']):
            methods.append(PSRK)
        if CASRN in _crit_PassutDanner.index and not np.isnan(_crit_PassutDanner.at[CASRN, 'Tc']):
            methods.append(PD)
        if CASRN in _crit_Yaws.index and not np.isnan(_crit_Yaws.at[CASRN, 'Tc']):
            methods.append(YAWS)
        if CASRN:
            methods.append(SURF)
        if IgnoreMethods:
            for Method in IgnoreMethods:
                if Method in methods:
                    methods.remove(Method)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IUPAC:
        _Tc = float(_crit_IUPAC.at[CASRN, 'Tc'])
    elif Method == MATTHEWS:
        _Tc = float(_crit_Matthews.at[CASRN, 'Tc'])
    elif Method == PSRK:
        _Tc = float(_crit_PSRKR4.at[CASRN, 'Tc'])
    elif Method == PD:
        _Tc = float(_crit_PassutDanner.at[CASRN, 'Tc'])
    elif Method == CRC:
        _Tc = float(_crit_CRC.at[CASRN, 'Tc'])
    elif Method == YAWS:
        _Tc = float(_crit_Yaws.at[CASRN, 'Tc'])
    elif Method == SURF:
        _Tc = third_property(CASRN=CASRN, T=True)
    elif Method == NONE:
        _Tc = None
    else:
        raise Exception('Failure in in function')
    return _Tc


Pc_methods = [IUPAC, MATTHEWS, CRC, PSRK, PD, YAWS, SURF]


def Pc(CASRN, AvailableMethods=False, Method=None, IgnoreMethods=[SURF]):
    r'''This function handles the retrieval of a chemical's critical
    pressure. Lookup is based on CASRNs. Will automatically select a data
    source to use if no Method is provided; returns None if the data is not
    available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'MATTHEWS' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Examples
    --------
    >>> Pc(CASRN='64-17-5')
    6137000.0

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Pc : float
        Critical pressure, [Pa]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Pc with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'MATTHEWS', 
        'CRC', 'PSRK', 'PD', 'YAWS', and 'SURF'. All valid values are also held  
        in the list `Pc_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Pc for the desired chemical, and will return methods instead of Pc
    IgnoreMethods : list, optional
        A list of methods to ignore in obtaining the full list of methods,
        useful for for performance reasons and ignoring inaccurate methods

    Notes
    -----
    A total of seven sources are available for this function. They are:

        * 'IUPAC', a series of critically evaluated
          experimental datum for organic compounds in [1]_, [2]_, [3]_, [4]_,
          [5]_, [6]_, [7]_, [8]_, [9]_, [10]_, [11]_, and [12]_.
        * 'MATTHEWS', a series of critically
          evaluated data for inorganic compounds in [13]_.
        * 'CRC', a compillation of critically
          evaluated data by the TRC as published in [14]_.
        * 'PSRK', a compillation of experimental and
          estimated data published in [15]_.
        * 'PD', an older compillation of
          data published in [16]_
        * 'YAWS', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [17]_.
        * SURF', an estimation method using a
          simple quadratic method for estimating Pc from Tc and Vc. This is
          ignored and not returned as a method by default.

    References
    ----------
    .. [1] Ambrose, Douglas, and Colin L. Young. "Vapor-Liquid Critical
       Properties of Elements and Compounds. 1. An Introductory Survey."
       Journal of Chemical & Engineering Data 41, no. 1 (January 1, 1996):
       154-154. doi:10.1021/je950378q.
    .. [2] Ambrose, Douglas, and Constantine Tsonopoulos. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 2. Normal Alkanes."
       Journal of Chemical & Engineering Data 40, no. 3 (May 1, 1995): 531-46.
       doi:10.1021/je00019a001.
    .. [3] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 3. Aromatic
       Hydrocarbons." Journal of Chemical & Engineering Data 40, no. 3
       (May 1, 1995): 547-58. doi:10.1021/je00019a002.
    .. [4] Gude, Michael, and Amyn S. Teja. "Vapor-Liquid Critical Properties
       of Elements and Compounds. 4. Aliphatic Alkanols." Journal of Chemical
       & Engineering Data 40, no. 5 (September 1, 1995): 1025-36.
       doi:10.1021/je00021a001.
    .. [5] Daubert, Thomas E. "Vapor-Liquid Critical Properties of Elements
       and Compounds. 5. Branched Alkanes and Cycloalkanes." Journal of
       Chemical & Engineering Data 41, no. 3 (January 1, 1996): 365-72.
       doi:10.1021/je9501548.
    .. [6] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 6. Unsaturated Aliphatic
       Hydrocarbons." Journal of Chemical & Engineering Data 41, no. 4
       (January 1, 1996): 645-56. doi:10.1021/je9501999.
    .. [7] Kudchadker, Arvind P., Douglas Ambrose, and Constantine Tsonopoulos.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 7. Oxygen
       Compounds Other Than Alkanols and Cycloalkanols." Journal of Chemical &
       Engineering Data 46, no. 3 (May 1, 2001): 457-79. doi:10.1021/je0001680.
    .. [8] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 8. Organic Sulfur,
       Silicon, and Tin Compounds (C + H + S, Si, and Sn)." Journal of Chemical
       & Engineering Data 46, no. 3 (May 1, 2001): 480-85.
       doi:10.1021/je000210r.
    .. [9] Marsh, Kenneth N., Colin L. Young, David W. Morton, Douglas Ambrose,
       and Constantine Tsonopoulos. "Vapor-Liquid Critical Properties of
       Elements and Compounds. 9. Organic Compounds Containing Nitrogen."
       Journal of Chemical & Engineering Data 51, no. 2 (March 1, 2006):
       305-14. doi:10.1021/je050221q.
    .. [10] Marsh, Kenneth N., Alan Abramson, Douglas Ambrose, David W. Morton,
       Eugene Nikitin, Constantine Tsonopoulos, and Colin L. Young.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 10. Organic
       Compounds Containing Halogens." Journal of Chemical & Engineering Data
       52, no. 5 (September 1, 2007): 1509-38. doi:10.1021/je700336g.
    .. [11] Ambrose, Douglas, Constantine Tsonopoulos, and Eugene D. Nikitin.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 11. Organic
       Compounds Containing B + O; Halogens + N, + O, + O + S, + S, + Si;
       N + O; and O + S, + Si." Journal of Chemical & Engineering Data 54,
       no. 3 (March 12, 2009): 669-89. doi:10.1021/je800580z.
    .. [12] Ambrose, Douglas, Constantine Tsonopoulos, Eugene D. Nikitin, David
       W. Morton, and Kenneth N. Marsh. "Vapor-Liquid Critical Properties of
       Elements and Compounds. 12. Review of Recent Data for Hydrocarbons and
       Non-Hydrocarbons." Journal of Chemical & Engineering Data, October 5,
       2015, 151005081500002. doi:10.1021/acs.jced.5b00571.
    .. [13] Mathews, Joseph F. "Critical Constants of Inorganic Substances."
       Chemical Reviews 72, no. 1 (February 1, 1972): 71-100.
       doi:10.1021/cr60275a004.
    .. [14] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics, 95E. Boca Raton, FL: CRC press, 2014.
    .. [15] Horstmann, Sven, Anna Jabłoniec, Jörg Krafczyk, Kai Fischer, and
       Jürgen Gmehling. "PSRK Group Contribution Equation of State:
       Comprehensive Revision and Extension IV, Including Critical Constants
       and Α-Function Parameters for 1000 Components." Fluid Phase Equilibria
       227, no. 2 (January 25, 2005): 157-64. doi:10.1016/j.fluid.2004.11.002.
    .. [16] Passut, Charles A., and Ronald P. Danner. "Acentric Factor. A
       Valuable Correlating Parameter for the Properties of Hydrocarbons."
       Industrial & Engineering Chemistry Process Design and Development 12,
       no. 3 (July 1, 1973): 365–68. doi:10.1021/i260047a026.
    .. [17] Yaws, Carl L. Thermophysical Properties of Chemicals and
       Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
       Publishing, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in _crit_IUPAC.index and not np.isnan(_crit_IUPAC.at[CASRN, 'Pc']):
            methods.append(IUPAC)
        if CASRN in _crit_Matthews.index and not np.isnan(_crit_Matthews.at[CASRN, 'Pc']):
            methods.append(MATTHEWS)
        if CASRN in _crit_CRC.index and not np.isnan(_crit_CRC.at[CASRN, 'Pc']):
            methods.append(CRC)
        if CASRN in _crit_PSRKR4.index and not np.isnan(_crit_PSRKR4.at[CASRN, 'Pc']):
            methods.append(PSRK)
        if CASRN in _crit_PassutDanner.index and not np.isnan(_crit_PassutDanner.at[CASRN, 'Pc']):
            methods.append(PD)
        if CASRN in _crit_Yaws.index and not np.isnan(_crit_Yaws.at[CASRN, 'Pc']):
            methods.append(YAWS)
        if CASRN:
            methods.append(SURF)
        if IgnoreMethods:
            for Method in IgnoreMethods:
                if Method in methods:
                    methods.remove(Method)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IUPAC:
        _Pc = float(_crit_IUPAC.at[CASRN, 'Pc'])
    elif Method == MATTHEWS:
        _Pc = float(_crit_Matthews.at[CASRN, 'Pc'])
    elif Method == CRC:
        _Pc = float(_crit_CRC.at[CASRN, 'Pc'])
    elif Method == PSRK:
        _Pc = float(_crit_PSRKR4.at[CASRN, 'Pc'])
    elif Method == PD:
        _Pc = float(_crit_PassutDanner.at[CASRN, 'Pc'])
    elif Method == YAWS:
        _Pc = float(_crit_Yaws.at[CASRN, 'Pc'])
    elif Method == SURF:
        _Pc = third_property(CASRN=CASRN, P=True)
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Pc


Vc_methods = [IUPAC, MATTHEWS, CRC, PSRK, YAWS, SURF]


def Vc(CASRN, AvailableMethods=False, Method=None, IgnoreMethods=[SURF]):
    r'''This function handles the retrieval of a chemical's critical
    volume. Lookup is based on CASRNs. Will automatically select a data
    source to use if no Method is provided; returns None if the data is not
    available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'MATTHEWS' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Examples
    --------
    >>> Vc(CASRN='64-17-5')
    0.000168

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Vc : float
        Critical volume, [m^3/mol]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Vc with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'MATTHEWS', 
        'CRC', 'PSRK', 'YAWS', and 'SURF'. All valid values are also held  
        in the list `Vc_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Vc for the desired chemical, and will return methods instead of Vc
    IgnoreMethods : list, optional
        A list of methods to ignore in obtaining the full list of methods,
        useful for for performance reasons and ignoring inaccurate methods

    Notes
    -----
    A total of six sources are available for this function. They are:

        * 'IUPAC', a series of critically evaluated
          experimental datum for organic compounds in [1]_, [2]_, [3]_, [4]_,
          [5]_, [6]_, [7]_, [8]_, [9]_, [10]_, [11]_, and [12]_.
        * 'MATTHEWS', a series of critically
          evaluated data for inorganic compounds in [13]_.
        * 'CRC', a compillation of critically
          evaluated data by the TRC as published in [14]_.
        * 'PSRK', a compillation of experimental and
          estimated data published in [15]_.
        * 'YAWS', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [16]_.
        * 'SURF', an estimation method using a
          simple quadratic method for estimating Pc from Tc and Vc. This is
          ignored and not returned as a method by default

    References
    ----------
    .. [1] Ambrose, Douglas, and Colin L. Young. "Vapor-Liquid Critical
       Properties of Elements and Compounds. 1. An Introductory Survey."
       Journal of Chemical & Engineering Data 41, no. 1 (January 1, 1996):
       154-154. doi:10.1021/je950378q.
    .. [2] Ambrose, Douglas, and Constantine Tsonopoulos. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 2. Normal Alkanes."
       Journal of Chemical & Engineering Data 40, no. 3 (May 1, 1995): 531-46.
       doi:10.1021/je00019a001.
    .. [3] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 3. Aromatic
       Hydrocarbons." Journal of Chemical & Engineering Data 40, no. 3
       (May 1, 1995): 547-58. doi:10.1021/je00019a002.
    .. [4] Gude, Michael, and Amyn S. Teja. "Vapor-Liquid Critical Properties
       of Elements and Compounds. 4. Aliphatic Alkanols." Journal of Chemical
       & Engineering Data 40, no. 5 (September 1, 1995): 1025-36.
       doi:10.1021/je00021a001.
    .. [5] Daubert, Thomas E. "Vapor-Liquid Critical Properties of Elements
       and Compounds. 5. Branched Alkanes and Cycloalkanes." Journal of
       Chemical & Engineering Data 41, no. 3 (January 1, 1996): 365-72.
       doi:10.1021/je9501548.
    .. [6] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 6. Unsaturated Aliphatic
       Hydrocarbons." Journal of Chemical & Engineering Data 41, no. 4
       (January 1, 1996): 645-56. doi:10.1021/je9501999.
    .. [7] Kudchadker, Arvind P., Douglas Ambrose, and Constantine Tsonopoulos.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 7. Oxygen
       Compounds Other Than Alkanols and Cycloalkanols." Journal of Chemical &
       Engineering Data 46, no. 3 (May 1, 2001): 457-79. doi:10.1021/je0001680.
    .. [8] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 8. Organic Sulfur,
       Silicon, and Tin Compounds (C + H + S, Si, and Sn)." Journal of Chemical
       & Engineering Data 46, no. 3 (May 1, 2001): 480-85.
       doi:10.1021/je000210r.
    .. [9] Marsh, Kenneth N., Colin L. Young, David W. Morton, Douglas Ambrose,
       and Constantine Tsonopoulos. "Vapor-Liquid Critical Properties of
       Elements and Compounds. 9. Organic Compounds Containing Nitrogen."
       Journal of Chemical & Engineering Data 51, no. 2 (March 1, 2006):
       305-14. doi:10.1021/je050221q.
    .. [10] Marsh, Kenneth N., Alan Abramson, Douglas Ambrose, David W. Morton,
       Eugene Nikitin, Constantine Tsonopoulos, and Colin L. Young.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 10. Organic
       Compounds Containing Halogens." Journal of Chemical & Engineering Data
       52, no. 5 (September 1, 2007): 1509-38. doi:10.1021/je700336g.
    .. [11] Ambrose, Douglas, Constantine Tsonopoulos, and Eugene D. Nikitin.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 11. Organic
       Compounds Containing B + O; Halogens + N, + O, + O + S, + S, + Si;
       N + O; and O + S, + Si." Journal of Chemical & Engineering Data 54,
       no. 3 (March 12, 2009): 669-89. doi:10.1021/je800580z.
    .. [12] Ambrose, Douglas, Constantine Tsonopoulos, Eugene D. Nikitin, David
       W. Morton, and Kenneth N. Marsh. "Vapor-Liquid Critical Properties of
       Elements and Compounds. 12. Review of Recent Data for Hydrocarbons and
       Non-Hydrocarbons." Journal of Chemical & Engineering Data, October 5,
       2015, 151005081500002. doi:10.1021/acs.jced.5b00571.
    .. [13] Mathews, Joseph F. "Critical Constants of Inorganic Substances."
       Chemical Reviews 72, no. 1 (February 1, 1972): 71-100.
       doi:10.1021/cr60275a004.
    .. [14] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics, 95E. Boca Raton, FL: CRC press, 2014.
    .. [15] Horstmann, Sven, Anna Jabłoniec, Jörg Krafczyk, Kai Fischer, and
       Jürgen Gmehling. "PSRK Group Contribution Equation of State:
       Comprehensive Revision and Extension IV, Including Critical Constants
       and Α-Function Parameters for 1000 Components." Fluid Phase Equilibria
       227, no. 2 (January 25, 2005): 157-64. doi:10.1016/j.fluid.2004.11.002.
    .. [16] Yaws, Carl L. Thermophysical Properties of Chemicals and
       Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
       Publishing, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in _crit_IUPAC.index and not np.isnan(_crit_IUPAC.at[CASRN, 'Vc']):
            methods.append(IUPAC)
        if CASRN in _crit_Matthews.index and not np.isnan(_crit_Matthews.at[CASRN, 'Vc']):
            methods.append(MATTHEWS)
        if CASRN in _crit_CRC.index and not np.isnan(_crit_CRC.at[CASRN, 'Vc']):
            methods.append(CRC)
        if CASRN in _crit_PSRKR4.index and not np.isnan(_crit_PSRKR4.at[CASRN, 'Vc']):
            methods.append(PSRK)
        if CASRN in _crit_Yaws.index and not np.isnan(_crit_Yaws.at[CASRN, 'Vc']):
            methods.append(YAWS)
        if CASRN:
            methods.append(SURF)
        if IgnoreMethods:
            for Method in IgnoreMethods:
                if Method in methods:
                    methods.remove(Method)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IUPAC:
        _Vc = float(_crit_IUPAC.at[CASRN, 'Vc'])
    elif Method == PSRK:
        _Vc = float(_crit_PSRKR4.at[CASRN, 'Vc'])
    elif Method == MATTHEWS:
        _Vc = float(_crit_Matthews.at[CASRN, 'Vc'])
    elif Method == CRC:
        _Vc = float(_crit_CRC.at[CASRN, 'Vc'])
    elif Method == YAWS:
        _Vc = float(_crit_Yaws.at[CASRN, 'Vc'])
    elif Method == SURF:
        _Vc = third_property(CASRN=CASRN, V=True)
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Vc


COMBINED = 'COMBINED'
Zc_methods = [IUPAC, MATTHEWS, CRC, PSRK, YAWS, COMBINED]


def Zc(CASRN, AvailableMethods=False, Method=None, IgnoreMethods=[COMBINED]):
    r'''This function handles the retrieval of a chemical's critical
    compressibility. Lookup is based on CASRNs. Will automatically select a
    data source to use if no Method is provided; returns None if the data is
    not available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'MATTHEWS' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Examples
    --------
    >>> Zc(CASRN='64-17-5')
    0.24100000000000002

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Zc : float
        Critical compressibility, [-]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Vc with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'MATTHEWS', 
        'CRC', 'PSRK', 'YAWS', and 'COMBINED'. All valid values are also held  
        in `Zc_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Zc for the desired chemical, and will return methods instead of Zc
    IgnoreMethods : list, optional
        A list of methods to ignore in obtaining the full list of methods,
        useful for for performance reasons and ignoring inaccurate methods

    Notes
    -----
    A total of five sources are available for this function. They are:

        * 'IUPAC', a series of critically evaluated
          experimental datum for organic compounds in [1]_, [2]_, [3]_, [4]_,
          [5]_, [6]_, [7]_, [8]_, [9]_, [10]_, [11]_, and [12]_.
        * 'MATTHEWS', a series of critically
          evaluated data for inorganic compounds in [13]_.
        * 'CRC', a compillation of critically
          evaluated data by the TRC as published in [14]_.
        * 'PSRK', a compillation of experimental and
          estimated data published in [15]_.
        * 'YAWS', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [16]_.

    References
    ----------
    .. [1] Ambrose, Douglas, and Colin L. Young. "Vapor-Liquid Critical
       Properties of Elements and Compounds. 1. An Introductory Survey."
       Journal of Chemical & Engineering Data 41, no. 1 (January 1, 1996):
       154-154. doi:10.1021/je950378q.
    .. [2] Ambrose, Douglas, and Constantine Tsonopoulos. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 2. Normal Alkanes."
       Journal of Chemical & Engineering Data 40, no. 3 (May 1, 1995): 531-46.
       doi:10.1021/je00019a001.
    .. [3] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 3. Aromatic
       Hydrocarbons." Journal of Chemical & Engineering Data 40, no. 3
       (May 1, 1995): 547-58. doi:10.1021/je00019a002.
    .. [4] Gude, Michael, and Amyn S. Teja. "Vapor-Liquid Critical Properties
       of Elements and Compounds. 4. Aliphatic Alkanols." Journal of Chemical
       & Engineering Data 40, no. 5 (September 1, 1995): 1025-36.
       doi:10.1021/je00021a001.
    .. [5] Daubert, Thomas E. "Vapor-Liquid Critical Properties of Elements
       and Compounds. 5. Branched Alkanes and Cycloalkanes." Journal of
       Chemical & Engineering Data 41, no. 3 (January 1, 1996): 365-72.
       doi:10.1021/je9501548.
    .. [6] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 6. Unsaturated Aliphatic
       Hydrocarbons." Journal of Chemical & Engineering Data 41, no. 4
       (January 1, 1996): 645-56. doi:10.1021/je9501999.
    .. [7] Kudchadker, Arvind P., Douglas Ambrose, and Constantine Tsonopoulos.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 7. Oxygen
       Compounds Other Than Alkanols and Cycloalkanols." Journal of Chemical &
       Engineering Data 46, no. 3 (May 1, 2001): 457-79. doi:10.1021/je0001680.
    .. [8] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
       Critical Properties of Elements and Compounds. 8. Organic Sulfur,
       Silicon, and Tin Compounds (C + H + S, Si, and Sn)." Journal of Chemical
       & Engineering Data 46, no. 3 (May 1, 2001): 480-85.
       doi:10.1021/je000210r.
    .. [9] Marsh, Kenneth N., Colin L. Young, David W. Morton, Douglas Ambrose,
       and Constantine Tsonopoulos. "Vapor-Liquid Critical Properties of
       Elements and Compounds. 9. Organic Compounds Containing Nitrogen."
       Journal of Chemical & Engineering Data 51, no. 2 (March 1, 2006):
       305-14. doi:10.1021/je050221q.
    .. [10] Marsh, Kenneth N., Alan Abramson, Douglas Ambrose, David W. Morton,
       Eugene Nikitin, Constantine Tsonopoulos, and Colin L. Young.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 10. Organic
       Compounds Containing Halogens." Journal of Chemical & Engineering Data
       52, no. 5 (September 1, 2007): 1509-38. doi:10.1021/je700336g.
    .. [11] Ambrose, Douglas, Constantine Tsonopoulos, and Eugene D. Nikitin.
       "Vapor-Liquid Critical Properties of Elements and Compounds. 11. Organic
       Compounds Containing B + O; Halogens + N, + O, + O + S, + S, + Si;
       N + O; and O + S, + Si." Journal of Chemical & Engineering Data 54,
       no. 3 (March 12, 2009): 669-89. doi:10.1021/je800580z.
    .. [12] Ambrose, Douglas, Constantine Tsonopoulos, Eugene D. Nikitin, David
       W. Morton, and Kenneth N. Marsh. "Vapor-Liquid Critical Properties of
       Elements and Compounds. 12. Review of Recent Data for Hydrocarbons and
       Non-Hydrocarbons." Journal of Chemical & Engineering Data, October 5,
       2015, 151005081500002. doi:10.1021/acs.jced.5b00571.
    .. [13] Mathews, Joseph F. "Critical Constants of Inorganic Substances."
       Chemical Reviews 72, no. 1 (February 1, 1972): 71-100.
       doi:10.1021/cr60275a004.
    .. [14] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics, 95E. Boca Raton, FL: CRC press, 2014.
    .. [15] Horstmann, Sven, Anna Jabłoniec, Jörg Krafczyk, Kai Fischer, and
       Jürgen Gmehling. "PSRK Group Contribution Equation of State:
       Comprehensive Revision and Extension IV, Including Critical Constants
       and Α-Function Parameters for 1000 Components." Fluid Phase Equilibria
       227, no. 2 (January 25, 2005): 157-64. doi:10.1016/j.fluid.2004.11.002.
    .. [16] Yaws, Carl L. Thermophysical Properties of Chemicals and
       Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
       Publishing, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in _crit_IUPAC.index and not np.isnan(_crit_IUPAC.at[CASRN, 'Zc']):
            methods.append(IUPAC)
        if CASRN in _crit_Matthews.index and not np.isnan(_crit_Matthews.at[CASRN, 'Zc']):
            methods.append(MATTHEWS)
        if CASRN in _crit_CRC.index and not np.isnan(_crit_CRC.at[CASRN, 'Zc']):
            methods.append(CRC)
        if CASRN in _crit_PSRKR4.index and not np.isnan(_crit_PSRKR4.at[CASRN, 'Zc']):
            methods.append(PSRK)
        if CASRN in _crit_Yaws.index and not np.isnan(_crit_Yaws.at[CASRN, 'Zc']):
            methods.append(YAWS)
        if Tc(CASRN) and Vc(CASRN) and Pc(CASRN):
            methods.append(COMBINED)
        if IgnoreMethods:
            for Method in IgnoreMethods:
                if Method in methods:
                    methods.remove(Method)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == IUPAC:
        _Zc = float(_crit_IUPAC.at[CASRN, 'Zc'])
    elif Method == PSRK:
        _Zc = float(_crit_PSRKR4.at[CASRN, 'Zc'])
    elif Method == MATTHEWS:
        _Zc = float(_crit_Matthews.at[CASRN, 'Zc'])
    elif Method == CRC:
        _Zc = float(_crit_CRC.at[CASRN, 'Zc'])
    elif Method == YAWS:
        _Zc = float(_crit_Yaws.at[CASRN, 'Zc'])
    elif Method == COMBINED:
        _Zc = Vc(CASRN)*Pc(CASRN)/Tc(CASRN)/R
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Zc


### Critical Property Relationships


def Ihmels(Tc=None, Pc=None, Vc=None):
    r'''Most recent, and most recommended method of estimating critical
    properties from each other. Two of the three properties are required.
    This model uses the "critical surface", a general plot of Tc vs Pc vs Vc.
    The model used 421 organic compounds to derive equation.
    The general equation is in [1]_:

    .. math::
        P_c = -0.025 + 2.215 \frac{T_c}{V_c}

    Parameters
    ----------
    Tc : float
        Critical temperature of fluid (optional) [K]
    Pc : float
        Critical pressure of fluid (optional) [Pa]
    Vc : float
        Critical volume of fluid (optional) [m^3/mol]

    Returns
    -------
    Tc, Pc or Vc : float
        Critical property of fluid [K], [Pa], or [m^3/mol]

    Notes
    -----
    The prediction of Tc from Pc and Vc is not tested, as this is not necessary
    anywhere, but it is implemented.
    Internal units are MPa, cm^3/mol, and K. A slight error occurs when
    Pa, cm^3/mol and K are used instead, on the order of <0.2%.
    Their equation was also compared with 56 inorganic and elements.
    Devations of 20% for <200K or >1000K points.

    Examples
    --------a
    Succinic acid [110-15-6]

    >>> Ihmels(Tc=851.0, Vc=0.000308)
    6095016.233766234

    References
    ----------
    .. [1] Ihmels, E. Christian. "The Critical Surface." Journal of Chemical
           & Engineering Data 55, no. 9 (September 9, 2010): 3474-80.
           doi:10.1021/je100167w.
    '''
    if Tc and Vc:
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Pc = -0.025+2.215*Tc/Vc
        Pc = Pc*1E6  # MPa to Pa
        return Pc
    elif Tc and Pc:
        Pc = Pc/1E6  # Pa to MPa
        Vc = 443*Tc/(200*Pc+5)
        Vc = Vc/1E6  # cm^3/mol to m^3/mol
        return Vc
    elif Pc and Vc:
        Pc = Pc/1E6  # Pa to MPa
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Tc = 5.0/443*(40*Pc*Vc + Vc)
        return Tc
    else:
        raise Exception('Two of Tc, Pc, and Vc must be provided')


def Meissner(Tc=None, Pc=None, Vc=None):
    r'''Old (1942) relationship for estimating critical
    properties from each other. Two of the three properties are required.
    This model uses the "critical surface", a general plot of Tc vs Pc vs Vc.
    The model used 42 organic and inorganic compounds to derive the equation.
    The general equation is in [1]_:

    .. math::
        P_c = \frac{2.08 T_c}{V_c-8}

    Parameters
    ----------
    Tc : float, optional
        Critical temperature of fluid [K]
    Pc : float, optional
        Critical pressure of fluid [Pa]
    Vc : float, optional
        Critical volume of fluid [m^3/mol]

    Returns
    -------
    Tc, Pc or Vc : float
        Critical property of fluid [K], [Pa], or [m^3/mol]

    Notes
    -----
    The prediction of Tc from Pc and Vc is not tested, as this is not necessary
    anywhere, but it is implemented.
    Internal units are atm, cm^3/mol, and K. A slight error occurs when
    Pa, cm^3/mol and K are used instead, on the order of <0.2%.
    This equation is less accurate than that of Ihmels, but surprisingly close.
    The author also proposed means of estimated properties independently.

    Examples
    --------
    Succinic acid [110-15-6]

    >>> Meissner(Tc=851.0, Vc=0.000308)
    5978445.199999999

    References
    ----------
    .. [1] Meissner, H. P., and E. M. Redding. "Prediction of Critical
           Constants." Industrial & Engineering Chemistry 34, no. 5
           (May 1, 1942): 521-26. doi:10.1021/ie50389a003.
    '''
    if Tc and Vc:
        Vc = Vc*1E6
        Pc = 20.8*Tc/(Vc-8)
        Pc = 101325*Pc  # atm to Pa
        return Pc
    elif Tc and Pc:
        Pc = Pc/101325.  # Pa to atm
        Vc = 104/5.0*Tc/Pc+8
        Vc = Vc/1E6  # cm^3/mol to m^3/mol
        return Vc
    elif Pc and Vc:
        Pc = Pc/101325.  # Pa to atm
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Tc = 5./104.0*Pc*(Vc-8)
        return Tc
    else:
        raise Exception('Two of Tc, Pc, and Vc must be provided')


def Grigoras(Tc=None, Pc=None, Vc=None):
    r'''Relatively recent (1990) relationship for estimating critical
    properties from each other. Two of the three properties are required.
    This model uses the "critical surface", a general plot of Tc vs Pc vs Vc.
    The model used 137 organic and inorganic compounds to derive the equation.
    The general equation is in [1]_:

    .. math::
        P_c = 2.9 + 20.2 \frac{T_c}{V_c}

    Parameters
    ----------
    Tc : float
        Critical temperature of fluid (optional) [K]
    Pc : float
        Critical pressure of fluid (optional) [Pa]
    Vc : float
        Critical volume of fluid (optional) [m^3/mol]

    Returns
    -------
    Tc, Pc or Vc : float
        Critical property of fluid [K], [Pa], or [m^3/mol]

    Notes
    -----
    The prediction of Tc from Pc and Vc is not tested, as this is not necessary
    anywhere, but it is implemented.
    Internal units are bar, cm^3/mol, and K. A slight error occurs when
    Pa, cm^3/mol and K are used instead, on the order of <0.2%.
    This equation is less accurate than that of Ihmels, but surprisingly close.
    The author also investigated an early QSPR model.

    Examples
    --------
    Succinic acid [110-15-6]

    >>> Grigoras(Tc=851.0, Vc=0.000308)
    5871233.766233766

    References
    ----------
    .. [1] Grigoras, Stelian. "A Structural Approach to Calculate Physical
           Properties of Pure Organic Substances: The Critical Temperature,
           Critical Volume and Related Properties." Journal of Computational
           Chemistry 11, no. 4 (May 1, 1990): 493-510.
           doi:10.1002/jcc.540110408
    '''
    if Tc and Vc:
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Pc = 2.9 + 20.2*Tc/Vc
        Pc = Pc*1E5  # bar to Pa
        return Pc
    elif Tc and Pc:
        Pc = Pc/1E5  # Pa to bar
        Vc = 202.0*Tc/(10*Pc-29.0)
        Vc = Vc/1E6  # cm^3/mol to m^3/mol
        return Vc
    elif Pc and Vc:
        Pc = Pc/1E5  # Pa to bar
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Tc = 1.0/202*(10*Pc-29.0)*Vc
        return Tc
    else:
        raise Exception('Two of Tc, Pc, and Vc must be provided')


IHMELS = 'IHMELS'
MEISSNER = 'MEISSNER'
GRIGORAS = 'GRIGORAS'
critical_surface_methods = [IHMELS, MEISSNER, GRIGORAS]


def critical_surface(Tc=None, Pc=None, Vc=None, AvailableMethods=False,
                     Method=None):
    r'''Function for calculating a critical property of a substance from its
    other two critical properties. Calls functions Ihmels, Meissner, and
    Grigoras, each of which use a general 'Critical surface' type of equation.
    Limited accuracy is expected due to very limited theoretical backing.

    Parameters
    ----------
    Tc : float
        Critical temperature of fluid (optional) [K]
    Pc : float
        Critical pressure of fluid (optional) [Pa]
    Vc : float
        Critical volume of fluid (optional) [m^3/mol]
    AvailableMethods : bool
        Request available methods for given parameters
    Method : string
        Request calculation uses the requested method

    Returns
    -------
    Tc, Pc or Vc : float
        Critical property of fluid [K], [Pa], or [m^3/mol]

    Notes
    -----

    Examples
    --------
    Decamethyltetrasiloxane [141-62-8]

    >>> critical_surface(Tc=599.4, Pc=1.19E6, Method='IHMELS')
    0.0010927333333333334
    '''
    def list_methods():
        methods = []
        if (Tc and Pc) or (Tc and Vc) or (Pc and Vc):
            methods.append(IHMELS)
            methods.append(MEISSNER)
            methods.append(GRIGORAS)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == IHMELS:
        Third = Ihmels(Tc=Tc, Pc=Pc, Vc=Vc)
    elif Method == MEISSNER:
        Third = Meissner(Tc=Tc, Pc=Pc, Vc=Vc)
    elif Method == GRIGORAS:
        Third = Grigoras(Tc=Tc, Pc=Pc, Vc=Vc)
    elif Method == NONE:
        Third = None
    else:
        raise Exception('Failure in in function')
    return Third


def third_property(CASRN=None, T=False, P=False, V=False):
    r'''Function for calculating a critical property of a substance from its
    other two critical properties, but retrieving the actual other critical
    values for convenient calculation.
    Calls functions Ihmels, Meissner, and
    Grigoras, each of which use a general 'Critical surface' type of equation.
    Limited accuracy is expected due to very limited theoretical backing.

    Parameters
    ----------
    CASRN : string
        The CAS number of the desired chemical
    T : bool
        Estimate critical temperature
    P : bool
        Estimate critical pressure
    V : bool
        Estimate critical volume

    Returns
    -------
    Tc, Pc or Vc : float
        Critical property of fluid [K], [Pa], or [m^3/mol]

    Notes
    -----
    Avoids recursion only by eliminating the None and critical surface options
    for calculating each critical property. So long as it never calls itself.
    Note that when used by Tc, Pc or Vc, this function results in said function
    calling the other functions (to determine methods) and (with method specified)

    Examples
    --------
    >>> # Decamethyltetrasiloxane [141-62-8]
    >>> third_property('141-62-8', V=True)
    0.0010920041152263375

    >>> # Succinic acid 110-15-6
    >>> third_property('110-15-6', P=True)
    6095016.233766234
    '''
    Third = None
    if V:
        Tc_methods = Tc(CASRN, AvailableMethods=True)[0:-2]
        Pc_methods = Pc(CASRN, AvailableMethods=True)[0:-2]
        if Tc_methods and Pc_methods:
            _Tc = Tc(CASRN=CASRN, Method=Tc_methods[0])
            _Pc = Pc(CASRN=CASRN, Method=Pc_methods[0])
            Third = critical_surface(Tc=_Tc, Pc=_Pc, Vc=None)
    elif P:
        Tc_methods = Tc(CASRN, AvailableMethods=True)[0:-2]
        Vc_methods = Vc(CASRN, AvailableMethods=True)[0:-2]
        if Tc_methods and Vc_methods:
            _Tc = Tc(CASRN=CASRN, Method=Tc_methods[0])
            _Vc = Vc(CASRN=CASRN, Method=Vc_methods[0])
            Third = critical_surface(Tc=_Tc, Vc=_Vc, Pc=None)
    elif T:
        Pc_methods = Pc(CASRN, AvailableMethods=True)[0:-2]
        Vc_methods = Vc(CASRN, AvailableMethods=True)[0:-2]
        if Pc_methods and Vc_methods:
            _Pc = Pc(CASRN=CASRN, Method=Pc_methods[0])
            _Vc = Vc(CASRN=CASRN, Method=Vc_methods[0])
            Third = critical_surface(Pc=_Pc, Vc=_Vc, Tc=None)
    else:
        raise Exception('Error in function')
    if not Third:
        return None
    return Third


### Critical Properties - Mixtures


### Crtical Temperature of Mixtures


def Li(zs, Tcs, Vcs):
    r'''Calculates critical temperature of a mixture according to
    mixing rules in [1]_. Better than simple mixing rules.

    .. math::
        T_{cm} = \sum_{i=1}^n \Phi_i T_{ci}\\
        \Phi = \frac{x_i V_{ci}}{\sum_{j=1}^n x_j V_{cj}}

    Parameters
    ----------
    zs : array-like
        Mole fractions of all components
    Tcs : array-like
        Critical temperatures of all components, [K]
    Vcs : array-like
        Critical volumes of all components, [m^3/mol]

    Returns
    -------
    Tcm : float
        Critical temperatures of the mixture, [K]

    Notes
    -----
    Reviewed in many papers on critical mixture temperature.

    Second example is from Najafi (2015), for ethylene, Benzene, ethylbenzene.
    This is similar to but not identical to the result from the article. The
    experimental point is 486.9 K.

    2rd example is from Najafi (2015), for:
    butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.
    Its result is identical to that calculated in the article.

    Examples
    --------
    Nitrogen-Argon 50/50 mixture

    >>> Li([0.5, 0.5], [126.2, 150.8], [8.95e-05, 7.49e-05])
    137.40766423357667

    butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.

    >>> Li([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6],
    ... [0.000255, 0.000313, 0.000371])
    449.68261498555444

    References
    ----------
    .. [1] Li, C. C. "Critical Temperature Estimation for Simple Mixtures."
       The Canadian Journal of Chemical Engineering 49, no. 5
       (October 1, 1971): 709-10. doi:10.1002/cjce.5450490529.
    '''
    if not none_and_length_check([zs, Tcs, Vcs]):
        raise Exception('Function inputs are incorrect format')

    denominator = sum(zs[i]*Vcs[i] for i in range(len(zs)))
    Tcm = 0
    for i in range(len(zs)):
        Tcm += zs[i]*Vcs[i]*Tcs[i]/denominator
    return Tcm


def Chueh_Prausnitz_Tc(zs, Tcs, Vcs, taus):
    r'''Calculates critical temperature of a mixture according to
    mixing rules in [1]_.

    .. math::
        T_{cm} = \sum_i^n \theta_i Tc_i + \sum_i^n\sum_j^n(\theta_i \theta_j
        \tau_{ij})T_{ref}

        \theta = \frac{x_i V_{ci}^{2/3}}{\sum_{j=1}^n x_j V_{cj}^{2/3}}

    For a binary mxiture, this simplifies to:

    .. math::
        T_{cm} = \theta_1T_{c1} + \theta_2T_{c2}  + 2\theta_1\theta_2\tau_{12}

    Parameters
    ----------
    zs : array-like
        Mole fractions of all components
    Tcs : array-like
        Critical temperatures of all components, [K]
    Vcs : array-like
        Critical volumes of all components, [m^3/mol]
    taus : array-like of shape `zs` by `zs`
        Interaction parameters

    Returns
    -------
    Tcm : float
        Critical temperatures of the mixture, [K]

    Notes
    -----
    All parameters, even if zero, must be given to this function.

    Examples
    --------
    butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.

    >>> Chueh_Prausnitz_Tc([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6],
    ... [0.000255, 0.000313, 0.000371], [[0, 1.92681, 6.80358],
    ... [1.92681, 0, 1.89312], [ 6.80358, 1.89312, 0]])
    450.1225764723492

    References
    ----------
    .. [1] Chueh, P. L., and J. M. Prausnitz. "Vapor-Liquid Equilibria at High
       Pressures: Calculation of Critical Temperatures, Volumes, and Pressures
       of Nonpolar Mixtures." AIChE Journal 13, no. 6 (November 1, 1967):
       1107-13. doi:10.1002/aic.690130613.
    .. [2] Najafi, Hamidreza, Babak Maghbooli, and Mohammad Amin Sobati.
       "Prediction of True Critical Temperature of Multi-Component Mixtures:
       Extending Fast Estimation Methods." Fluid Phase Equilibria 392
       (April 25, 2015): 104-26. doi:10.1016/j.fluid.2015.02.001.
    '''
    if not none_and_length_check([zs, Tcs, Vcs]):
        raise Exception('Function inputs are incorrect format')

    denominator = sum(zs[i]*Vcs[i]**(2/3.) for i in range(len(zs)))
    Tcm = 0
    for i in range(len(zs)):
        Tcm += zs[i]*Vcs[i]**(2/3.)*Tcs[i]/denominator
        for j in range(len(zs)):
            Tcm += (zs[i]*Vcs[i]**(2/3.)/denominator)*(zs[j]*Vcs[j]**(2/3.)/denominator)*taus[i][j]
    return Tcm


#print Chueh_Prausnitz_Tc([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6], [0.000255, 0.000313, 0.000371], [[0, 1.92681, 6.80358], [1.92681, 0, 1.89312], [ 6.80358, 1.89312, 0]])
#butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.
# 450.12258 is expected
# butane/pentane 1.92681
#butane/hexane 6.80358
# pentane/hexane 1.89312


##print Chueh_Prausnitz_Tc([0.5, 0.5], [508.1, 425.12], [0.000213, 0.000255], [[0, -14.2619], [-14.2619, 0]])
####
##print Li([0.5, 0.5], [508.1, 425.12], [0.000213, 0.000255])
#
#print Chueh_Prausnitz_Tc([0.5, 0.447, .053], [282.34, 562.05, 617.15], [0.0001311, 0.000256, 0.000374], [[0, 37.9570, 0], [37.9570, 0, 4.2459], [0, 4.2459, 0]])
## ethylene, Benzene, ethylbenzene
##ethylene	74-85-1	1-ALKENES	0.5	benzene	71-43-2	N-ALKYLBENZENES	0.447	ethylbenzene	100-41-4	N-ALKYLBENZENES	0.053
##['74-85-1', '71-43-2', '100-41-4']
##[[0, 37.9570, 0], [37.9570, 0, 4.2459], [0, 4.2459, 0]]
#
##benzene	ethylene	14	37.9570
##benzene	ethylbenzene	9	4.2459


def Grieves_Thodos(zs, Tcs, Aijs):
    r'''Calculates critical temperature of a mixture according to
    mixing rules in [1]_.

    .. math::
        T_{cm} = \sum_{i} \frac{T_{ci}}{1 + (1/x_i)\sum_j A_{ij} x_j}

    For a binary mxiture, this simplifies to:

    .. math::
        T_{cm} = \frac{T_{c1}}{1 + (x_2/x_1)A_{12}} +  \frac{T_{c2}}
        {1 + (x_1/x_2)A_{21}}

    Parameters
    ----------
    zs : array-like
        Mole fractions of all components
    Tcs : array-like
        Critical temperatures of all components, [K]
    Aijs : array-like of shape `zs` by `zs`
        Interaction parameters

    Returns
    -------
    Tcm : float
        Critical temperatures of the mixture, [K]

    Notes
    -----
    All parameters, even if zero, must be given to this function.
    Giving 0s gives really bad results however.

    Examples
    --------
    butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.

    >>> Grieves_Thodos([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6], [[0, 1.2503, 1.516], [0.799807, 0, 1.23843], [0.659633, 0.807474, 0]])
    450.1839618758971

    References
    ----------
    .. [1] Grieves, Robert B., and George Thodos. "The Critical Temperatures of
       Multicomponent Hydrocarbon Systems." AIChE Journal 8, no. 4
       (September 1, 1962): 550-53. doi:10.1002/aic.690080426.
    .. [2] Najafi, Hamidreza, Babak Maghbooli, and Mohammad Amin Sobati.
       "Prediction of True Critical Temperature of Multi-Component Mixtures:
       Extending Fast Estimation Methods." Fluid Phase Equilibria 392
       (April 25, 2015): 104-26. doi:10.1016/j.fluid.2015.02.001.
    '''
    if not none_and_length_check([zs, Tcs]):
        raise Exception('Function inputs are incorrect format')
    Tcm = 0
    for i in range(len(zs)):
            Tcm += Tcs[i]/(1. + 1./zs[i]*sum(Aijs[i][j]*zs[j] for j in range(len(zs))))
    return Tcm

#print Grieves_Thodos([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6], [[0, 1.2503, 1.516], [0.799807, 0, 1.23843], [0.659633, 0.807474, 0]])
#butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.
# 450.18396 is expected
# butane/pentane 1.2503000	0.7998070
#butane/hexane 1.5160000	0.6596330
# pentane/hexane 1.238430	0.807474


#print Grieves_Thodos([0.5, 0.447, .053], [282.34, 562.05, 617.15], [[0, 0.8166850, 0], [0.7727120, 0, 1.5038], [0, 0.6650, 0]])
## ethylene, Benzene, ethylbenzene
## Vcs=[0.0001311, 0.000256, 0.000374]
#
#1.5038	0.6650 # benzene to ethylbenzene
#
#0.7727120	0.8166850 # benzene to ethylene
# Author claims result of 473.74.


def modified_Wilson_Tc(zs, Tcs, Aijs):
    r'''Calculates critical temperature of a mixture according to
    mixing rules in [1]_. Equation

    .. math::
        T_{cm} = \sum_i x_i T_{ci} + C\sum_i x_i \ln \left(x_i + \sum_j x_j A_{ij}\right)T_{ref}

    For a binary mxiture, this simplifies to:

    .. math::
        T_{cm} = x_1 T_{c1} + x_2 T_{c2} + C[x_1 \ln(x_1 + x_2A_{12}) + x_2\ln(x_2 + x_1 A_{21})]

    Parameters
    ----------
    zs : float
        Mole fractions of all components
    Tcs : float
        Critical temperatures of all components, [K]
    Aijs : matrix
        Interaction parameters

    Returns
    -------
    Tcm : float
        Critical temperatures of the mixture, [K]

    Notes
    -----
    The equation and original article has been reviewed.
    [1]_ has 75 binary systems, and additional multicomponent mixture parameters.
    All parameters, even if zero, must be given to this function.

    2rd example is from [2]_, for:
    butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.
    Its result is identical to that calculated in the article.

    Examples
    --------
    >>> modified_Wilson_Tc([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6],
    ... [[0, 1.174450, 1.274390], [0.835914, 0, 1.21038],
    ... [0.746878, 0.80677, 0]])
    450.0305966823031

    References
    ----------
    .. [1] Teja, Amyn S., Kul B. Garg, and Richard L. Smith. "A Method for the
       Calculation of Gas-Liquid Critical Temperatures and Pressures of
       Multicomponent Mixtures." Industrial & Engineering Chemistry Process
       Design and Development 22, no. 4 (1983): 672-76.
    .. [2] Najafi, Hamidreza, Babak Maghbooli, and Mohammad Amin Sobati.
       "Prediction of True Critical Temperature of Multi-Component Mixtures:
       Extending Fast Estimation Methods." Fluid Phase Equilibria 392
       (April 25, 2015): 104-26. doi:10.1016/j.fluid.2015.02.001.
    '''
    if not none_and_length_check([zs, Tcs]):
        raise Exception('Function inputs are incorrect format')
    C = -2500
    Tcm = sum(zs[i]*Tcs[i] for i in range(len(zs)))
    for i in range(len(zs)):
            Tcm += C*zs[i]*log(zs[i] + sum(zs[j]*Aijs[i][j] for j in range(len(zs))))
    return Tcm

#print modified_Wilson_Tc([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6], [[0, 1.174450, 1.274390], [0.835914, 0, 1.21038], [0.746878, 0.80677, 0]])
#butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.
# 450.0306 is expected
# butane/pentane  1.174450	0.835914
#butane/hexane 1.274390	0.746878
# pentane/hexane 1.21038	0.80677


#print modified_Wilson_Tc([0.5, 0.5], [508.1, 425.12],  [[0, 0.8359], [0, 1.1963]]) # Acetone/butane 50-50
#print modified_Wilson_Tc([0.5, 0.447, .053], [282.34, 562.05, 617.15], [[0,1.0853, 0 ], [0.8425, 0, 1.2514], [0, 0.7688, 0]])
#Tc exp: 486.90
# Author claims MW gives 471.49

## ethylene, Benzene, ethylbenzene
# Vcs=[0.0001311, 0.000256, 0.000374]
#benzene-ethylene  0.8425	1.0853
#benzene	ethylbenzene 1.2514	0.7688
# ethylene-ethylbenzene   (26.166530797247095, 3.2152024634944754) CALCULATED


#print Grieves_Thodos([0.5, 0.5], [508.1, 425.12], [[0, 0.7137], [1.6496, 0]])
#print Grieves_Thodos([0.5, 0.5], [508.1, 425.12], [[0, 0.1305], [0.09106, 0]])


def Tc_mixture(Tcs=None, zs=None, CASRNs=None, AvailableMethods=False, Method=None):  # pragma: no cover
    '''This function handles the retrival of a mixture's critical temperature.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> Tc_mixture([400, 550], [0.3, 0.7])
    505.0
    '''
    def list_methods():
        methods = []
        if none_and_length_check([Tcs]):
            methods.append('Simple')
        methods.append('None')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == 'Simple':
        return mixing_simple(zs, Tcs)
    elif Method == 'None':
        return None
    else:
        raise Exception('Failure in in function')

### Crtical Pressure of Mixtures


def Pc_mixture(Pcs=None, zs=None, CASRNs=None, AvailableMethods=False, Method=None):  # pragma: no cover
    '''This function handles the retrival of a mixture's critical temperature.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> Pc_mixture([2.2E7, 1.1E7], [0.3, 0.7])
    14300000.0
    '''
    def list_methods():
        methods = []
        if none_and_length_check([Pcs]):
            methods.append('Simple')
        methods.append('None')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == 'Simple':
        return mixing_simple(zs, Pcs)
    elif Method == 'None':
        return None
    else:
        raise Exception('Failure in in function')


### Crtical Volume of Mixtures
def Chueh_Prausnitz_Vc(zs, Vcs, nus):
    r'''Calculates critical volume of a mixture according to
    mixing rules in [1]_ with an interaction parameter.

    .. math::
        V_{cm} = \sum_i^n \theta_i V_{ci} + \sum_i^n\sum_j^n(\theta_i \theta_j \nu_{ij})V_{ref}
        \theta = \frac{x_i V_{ci}^{2/3}}{\sum_{j=1}^n x_j V_{cj}^{2/3}}

    Parameters
    ----------
    zs : float
        Mole fractions of all components
    Vcs : float
        Critical volumes of all components, [m^3/mol]
    nus : matrix
        Interaction parameters, [cm^3/mol]

    Returns
    -------
    Vcm : float
        Critical volume of the mixture, [m^3/mol]

    Notes
    -----
    All parameters, even if zero, must be given to this function.
    nu parameters are in cm^3/mol, but are converted to m^3/mol inside the function


    Examples
    --------
    1-butanol/benzene 0.4271/0.5729 mixture, Vcm = 268.096 mL/mol.

    >>> Chueh_Prausnitz_Vc([0.4271, 0.5729], [0.000273, 0.000256], [[0, 5.61847], [5.61847, 0]])
    0.00026620503424517445

    References
    ----------
    .. [1] Chueh, P. L., and J. M. Prausnitz. "Vapor-Liquid Equilibria at High
       Pressures: Calculation of Critical Temperatures, Volumes, and Pressures
       of Nonpolar Mixtures." AIChE Journal 13, no. 6 (November 1, 1967):
       1107-13. doi:10.1002/aic.690130613.
    .. [2] Najafi, Hamidreza, Babak Maghbooli, and Mohammad Amin Sobati.
       "Prediction of True Critical Volume of Multi-Component Mixtures:
       Extending Fast Estimation Methods." Fluid Phase Equilibria 386
       (January 25, 2015): 13-29. doi:10.1016/j.fluid.2014.11.008.
    '''
    if not none_and_length_check([zs, Vcs]): # check same-length inputs
        raise Exception('Function inputs are incorrect format')

    denominator = sum(zs[i]*Vcs[i]**(2/3.) for i in range(len(zs)))
    Vcm = 0
    for i in range(len(zs)):
        Vcm += zs[i]*Vcs[i]**(2/3.)*Vcs[i]/denominator
        for j in range(len(zs)):
            Vcm += (zs[i]*Vcs[i]**(2/3.)/denominator)*(zs[j]*Vcs[j]**(2/3.)/denominator)*nus[i][j]/1E6
    return Vcm

#print Chueh_Prausnitz_Vc([0.4271, 0.5729], [0.000273, 0.000256], [[0, 5.61847], [5.61847, 0]])
## 1-butanol/benzene 0.4271/0.5729 mixture, Vcm = 268.096 mL/mol
## Expected result: 266.205034245174

def modified_Wilson_Vc(zs, Vcs, Aijs):
    r'''Calculates critical volume of a mixture according to
    mixing rules in [1]_ with parameters. Equation

    .. math::
        V_{cm} = \sum_i x_i V_{ci} + C\sum_i x_i \ln \left(x_i + \sum_j x_j A_{ij}\right)V_{ref}

    For a binary mxiture, this simplifies to:

    .. math::
        V_{cm} = x_1 V_{c1} + x_2 V_{c2} + C[x_1 \ln(x_1 + x_2A_{12}) + x_2\ln(x_2 + x_1 A_{21})]

    Parameters
    ----------
    zs : float
        Mole fractions of all components
    Vcs : float
        Critical volumes of all components, [m^3/mol]
    Aijs : matrix
        Interaction parameters, [cm^3/mol]

    Returns
    -------
    Vcm : float
        Critical volume of the mixture, [m^3/mol]

    Notes
    -----
    The equation and original article has been reviewed.
    All parameters, even if zero, must be given to this function.
    C = -2500

    All parameters, even if zero, must be given to this function.
    nu parameters are in cm^3/mol, but are converted to m^3/mol inside the function


    Examples
    --------
    1-butanol/benzene 0.4271/0.5729 mixture, Vcm = 268.096 mL/mol.

    >>> modified_Wilson_Vc([0.4271, 0.5729], [0.000273, 0.000256],
    ... [[0, 0.6671250], [1.3939900, 0]])
    0.0002664335032706881

    References
    ----------
    .. [1] Teja, Amyn S., Kul B. Garg, and Richard L. Smith. "A Method for the
       Calculation of Gas-Liquid Critical Temperatures and Pressures of
       Multicomponent Mixtures." Industrial & Engineering Chemistry Process
       Design and Development 22, no. 4 (1983): 672-76.
    .. [2] Najafi, Hamidreza, Babak Maghbooli, and Mohammad Amin Sobati.
       "Prediction of True Critical Temperature of Multi-Component Mixtures:
       Extending Fast Estimation Methods." Fluid Phase Equilibria 392
       (April 25, 2015): 104-26. doi:10.1016/j.fluid.2015.02.001.
    '''
    if not none_and_length_check([zs, Vcs]): # check same-length inputs
        raise Exception('Function inputs are incorrect format')
    C = -2500
    Vcm = sum(zs[i]*Vcs[i] for i in range(len(zs)))
    for i in range(len(zs)):
            Vcm += C*zs[i]*log(zs[i] + sum(zs[j]*Aijs[i][j] for j in range(len(zs))))/1E6
    return Vcm



def Vc_mixture(Vcs=None, zs=None, CASRNs=None, AvailableMethods=False, Method=None):  # pragma: no cover
    '''This function handles the retrival of a mixture's critical temperature.

    This API is considered experimental, and is expected to be removed in a
    future release in favor of a more complete object-oriented interface.

    >>> Vc_mixture([5.6E-5, 2E-4], [0.3, 0.7])
    0.0001568
    '''
    def list_methods():
        methods = []
        if none_and_length_check([Vcs]):
            methods.append('Simple')
        methods.append('None')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == 'Simple':
        return mixing_simple(zs, Vcs)
    elif Method == 'None':
        return None
    else:
        raise Exception('Failure in in function')



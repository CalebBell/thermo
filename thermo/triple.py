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

__all__ = ['Staveley_data', 'Tt_methods', 'Tt', 'Pt_methods', 'Pt']

import os
import numpy as np
import pandas as pd

from thermo.phase_change import Tm
from thermo.vapor_pressure import VaporPressure

folder = os.path.join(os.path.dirname(__file__), 'Triple Properties')


Staveley_data = pd.read_csv(os.path.join(folder, 'Staveley 1981.tsv'),
                       sep='\t', index_col=0)

STAVELEY = 'STAVELEY'
MELTING = 'MELTING'
NONE = 'NONE'

Tt_methods = [STAVELEY, MELTING]


def Tt(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's triple temperature.
    Lookup is based on CASRNs. Will automatically select a data source to use
    if no Method is provided; returns None if the data is not available.

    Returns data from [1]_, or a chemical's melting point if available.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Tt : float
        Triple point temperature, [K]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Tt with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        Tt_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the Tt for the desired chemical, and will return methods
        instead of the Tt

    Notes
    -----
    Median difference between melting points and triple points is 0.02 K.
    Accordingly, this should be more than good enough for engineering
    applications.

    Temperatures are on the ITS-68 scale.

    Examples
    --------
    Ammonia

    >>> Tt('7664-41-7')
    195.47999999999999

    References
    ----------
    .. [1] Staveley, L. A. K., L. Q. Lobo, and J. C. G. Calado. "Triple-Points
       of Low Melting Substances and Their Use in Cryogenic Work." Cryogenics
       21, no. 3 (March 1981): 131-144. doi:10.1016/0011-2275(81)90264-2.
    '''
    def list_methods():
        methods = []
        if CASRN in Staveley_data.index:
            methods.append(STAVELEY)
        if Tm(CASRN):
            methods.append(MELTING)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == STAVELEY:
        Tt = Staveley_data.at[CASRN, "Tt68"]
    elif Method == MELTING:
        Tt = Tm(CASRN)
    elif Method == NONE:
        Tt = None
    else:
        raise Exception('Failure in in function')
    return Tt

DEFINITION = 'DEFINITION'
Pt_methods = [STAVELEY, DEFINITION]


def Pt(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's triple pressure.
    Lookup is based on CASRNs. Will automatically select a data source to use
    if no Method is provided; returns None if the data is not available.

    Returns data from [1]_, or attempts to calculate the vapor pressure at the
    triple temperature, if data is available.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Pt : float
        Triple point pressure, [Pa]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Pt with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        Pt_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the Pt for the desired chemical, and will return methods
        instead of the Pt

    Notes
    -----

    Examples
    --------
    Ammonia

    >>> Pt('7664-41-7')
    6079.5

    References
    ----------
    .. [1] Staveley, L. A. K., L. Q. Lobo, and J. C. G. Calado. "Triple-Points
       of Low Melting Substances and Their Use in Cryogenic Work." Cryogenics
       21, no. 3 (March 1981): 131-144. doi:10.1016/0011-2275(81)90264-2.
    '''
    def list_methods():
        methods = []
        if CASRN in Staveley_data.index and not np.isnan(Staveley_data.at[CASRN, 'Pt']):
            methods.append(STAVELEY)
        if Tt(CASRN) and VaporPressure(CASRN=CASRN).T_dependent_property(T=Tt(CASRN)):
            methods.append(DEFINITION)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == STAVELEY:
        Pt = Staveley_data.at[CASRN, 'Pt']
    elif Method == DEFINITION:
        Pt = VaporPressure(CASRN=CASRN).T_dependent_property(T=Tt(CASRN))
    elif Method == NONE:
        Pt = None
    else:
        raise Exception('Failure in in function')
    return Pt


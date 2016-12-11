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

__all__ = ['dipole_moment', '_dipole_Poling', '_dipole_CCDB', '_dipole_Muller', 'dipole_methods']
import os
import numpy as np
import pandas as pd


folder = os.path.join(os.path.dirname(__file__), 'Misc')

_dipole_Poling = pd.read_csv(os.path.join(folder, 'Poling Dipole.csv'),
                          sep='\t', index_col=0)

_dipole_CCDB = pd.read_csv(os.path.join(folder, 'cccbdb.nist.gov Dipoles.csv'),
                          sep='\t', index_col=0)

_dipole_Muller = pd.read_csv(os.path.join(folder, 'Muller Supporting Info Dipoles.csv'),
                          sep='\t', index_col=0)


CCCBDB = 'CCCBDB'
MULLER = 'MULLER'
POLING = 'POLING'
NONE = 'NONE'

dipole_methods = [CCCBDB, MULLER, POLING]


def dipole_moment(CASRN, AvailableMethods=False, Method=None):
    r'''This function handles the retrieval of a chemical's dipole moment.
    Lookup is based on CASRNs. Will automatically select a data source to use
    if no Method is provided; returns None if the data is not available.

    Prefered source is 'CCCBDB'. Considerable variation in reported data has
    found.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    dipole : float
        Dipole moment, [debye]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain dipole moment with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'CCCBDB', 'MULLER', or
        'POLING'. All valid values are also held in the list `dipole_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the dipole moment for the desired chemical, and will return methods
        instead of the dipole moment

    Notes
    -----
    A total of three sources are available for this function. They are:

        * 'CCCBDB', a series of critically evaluated data for compounds in
          [1]_, intended for use in predictive modeling.
        * 'MULLER', a collection of data in a
          group-contribution scheme in [2]_.
        * 'POLING', in the appendix in [3].

    Examples
    --------
    >>> dipole_moment(CASRN='64-17-5')
    1.44

    References
    ----------
    .. [1] NIST Computational Chemistry Comparison and Benchmark Database
       NIST Standard Reference Database Number 101 Release 17b, September 2015,
       Editor: Russell D. Johnson III http://cccbdb.nist.gov/
    .. [2] Muller, Karsten, Liudmila Mokrushina, and Wolfgang Arlt. "Second-
       Order Group Contribution Method for the Determination of the Dipole
       Moment." Journal of Chemical & Engineering Data 57, no. 4 (April 12,
       2012): 1231-36. doi:10.1021/je2013395.
    .. [3] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    def list_methods():
        methods = []
        if CASRN in _dipole_CCDB.index and not np.isnan(_dipole_CCDB.at[CASRN, 'Dipole']):
            methods.append(CCCBDB)
        if CASRN in _dipole_Muller.index and not np.isnan(_dipole_Muller.at[CASRN, 'Dipole']):
            methods.append(MULLER)
        if CASRN in _dipole_Poling.index and not np.isnan(_dipole_Poling.at[CASRN, 'Dipole']):
            methods.append(POLING)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == CCCBDB:
        _dipole = float(_dipole_CCDB.at[CASRN, 'Dipole'])
    elif Method == MULLER:
        _dipole = float(_dipole_Muller.at[CASRN, 'Dipole'])
    elif Method == POLING:
        _dipole = float(_dipole_Poling.at[CASRN, 'Dipole'])
    elif Method == NONE:
        _dipole = None
    else:
        raise Exception('Failure in in function')
    return _dipole

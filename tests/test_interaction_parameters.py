# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import exp, log
import pytest
import numpy as np
import pandas as pd
from fluids.constants import calorie, R
from thermo.interaction_parameters import IPDB
from fluids.numerics import assert_close, assert_close1d, assert_close2d


def run_validate_db():
    from thermo.interaction_parameters import ip_files

    for name in ip_files.keys():
        IPDB.validate_table(name)

def test_basic_chemsep_PR():
    kij = IPDB.get_ip_specific('ChemSep PR', ['124-38-9', '67-56-1'], 'kij')
    assert_close(kij, 0.0583)

    kij_auto = IPDB.get_ip_automatic(['124-38-9', '67-56-1'], 'PR kij', 'kij')
    assert_close(kij, kij_auto)

    kij_missing = IPDB.get_ip_specific('ChemSep PR', ['1249-38-9', '67-56-1'], 'kij')
    assert kij_missing == 0
    assert False == IPDB.has_ip_specific('ChemSep PR', ['1249-38-9', '67-56-1'], 'kij')
    assert True == IPDB.has_ip_specific('ChemSep PR', ['124-38-9', '67-56-1'], 'kij')

    assert IPDB.get_tables_with_type('PR kij') == ['ChemSep PR']

    # interaction parameter matrix
    kij_C1C4 = IPDB.get_ip_symmetric_matrix('ChemSep PR', ['74-82-8', '74-84-0', '74-98-6', '106-97-8'], 'kij')
    kij_C1C4_known = [[0.0, -0.0059, 0.0119, 0.0185],
                     [-0.0059, 0.0, 0.0011, 0.0089],
                     [0.0119, 0.0011, 0.0, 0.0033],
                     [0.0185, 0.0089, 0.0033, 0.0]]
    assert_close2d(kij_C1C4, kij_C1C4_known)
    # Test for asymetric works the same since the model is asymmetric
    kij_C1C4 = IPDB.get_ip_symmetric_matrix('ChemSep PR', ['74-82-8', '74-84-0', '74-98-6', '106-97-8'], 'kij')
    assert_close2d(kij_C1C4, kij_C1C4_known)


def test_basic_chemsep_NRTL():
    # ethanol water, converted to metric, simple T dependence
    bijs = IPDB.get_ip_asymmetric_matrix('ChemSep NRTL', ['64-17-5', '7732-18-5'], 'bij')

    alphas_known = [[0.0, 0.2937, 0.3009], [0.2937, 0.0, 0.2999], [0.3009, 0.2999, 0.0]]
    # Test is works both symmetric and asymmetric
    alphas = IPDB.get_ip_asymmetric_matrix('ChemSep NRTL', ['64-17-5', '7732-18-5', '67-56-1'], 'alphaij')
    assert_close2d(alphas, alphas_known)
    alphas = IPDB.get_ip_symmetric_matrix('ChemSep NRTL', ['64-17-5', '7732-18-5', '67-56-1'], 'alphaij')
    assert_close2d(alphas, alphas_known)
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

from numpy.testing import assert_allclose
import pytest
from thermo.law import *
from thermo.utils import int2CAS
from thermo.identifiers import checkCAS

load_law_data()
load_economic_data()
from thermo.law import DSL_data, TSCA_data, EINECS_data, SPIN_data, NLP_data
from thermo.law import HPV_data, _EPACDRDict, _ECHATonnageDict


@pytest.mark.slow
def test_DSL_data():
    assert DSL_data.index.is_unique
    assert DSL_data['Registry'].sum() == 48363
    assert DSL_data.shape == (73036, 1)

    assert all([checkCAS(int2CAS(i)) for i in DSL_data.index])


@pytest.mark.slow
def test_TSCA_data():
    tots_calc = [TSCA_data[i].sum() for i in ['UV', 'E', 'F', 'N', 'P', 'S', 'R', 'T', 'XU', 'SP', 'TP', 'Y1', 'Y2']]
    tots = [16829, 271, 3, 713, 8371, 1173, 13, 151, 19035, 74, 50, 352, 9]
    assert tots_calc == tots

    assert TSCA_data.index.is_unique
    assert TSCA_data.shape == (67635, 13)

    assert all([checkCAS(int2CAS(i)) for i in TSCA_data.index])


@pytest.mark.slow
def test_EINECS_data():
    assert EINECS_data.index.is_unique
    assert EINECS_data.shape == (100203, 0)
    assert sum(list(EINECS_data.index))  == 4497611272838

    assert all([checkCAS(int2CAS(i)) for i in EINECS_data.index])


@pytest.mark.slow
def test_SPIN_data():
    assert SPIN_data.index.is_unique
    assert SPIN_data.shape == (26023, 0)
    assert sum(list(SPIN_data.index)) == 1666688770043

    assert all([checkCAS(int2CAS(i)) for i in SPIN_data.index])


def test_NLP_data():
    assert NLP_data.index.is_unique
    assert NLP_data.shape == (698, 0)
    assert sum(list(NLP_data.index)) == 83268755392

    assert all([checkCAS(int2CAS(i)) for i in NLP_data.index])


def test_HPV_data():
    assert HPV_data.index.is_unique
    assert HPV_data.shape == (5067, 0)
    assert sum(list(HPV_data.index)) == 176952023632

    assert all([checkCAS(int2CAS(i)) for i in HPV_data.index])


def test_legal_status():
    DSL = 'DSL'
    TSCA = 'TSCA'
    EINECS = 'EINECS'
    NLP = 'NLP'
    SPIN = 'SPIN'
    COMBINED = 'COMBINED'
    UNLISTED = 'UNLISTED'
    LISTED = 'LISTED'

    
    hit = legal_status(CASRN='1648727-81-4')
    hit_desc = {TSCA: sorted([TSCA_flags['N'], TSCA_flags['P'], TSCA_flags['XU']]),
                SPIN: UNLISTED, DSL: UNLISTED, EINECS: UNLISTED, NLP: UNLISTED}
    assert hit == hit_desc

    hit = legal_status(CASRN='1071-83-6')
    hit_desc = {TSCA: UNLISTED, SPIN: LISTED, DSL: LISTED,
                EINECS: LISTED, NLP: UNLISTED}
    assert hit == hit_desc

    # Ethanol
    hit = legal_status(CASRN='64-17-5')
    hit_desc = {TSCA: LISTED, SPIN: LISTED, DSL: LISTED,
                EINECS: LISTED, NLP: UNLISTED}
    assert hit == hit_desc

    # Random chemical on NLP
    hit = legal_status('98478-71-8')
    hit_desc = {TSCA: UNLISTED, SPIN: UNLISTED, DSL: UNLISTED,
                EINECS: UNLISTED, NLP: LISTED}
    assert hit == hit_desc

    for Method, hit in zip([DSL, TSCA, EINECS, SPIN, NLP], [LISTED, LISTED, LISTED, LISTED, UNLISTED]):
        assert hit == legal_status(CASRN='64-17-5', Method=Method)

    assert legal_status(CASRN='1648727-81-4', AvailableMethods=True) == [COMBINED, DSL, TSCA, EINECS, NLP, SPIN]

    with pytest.raises(Exception):
        legal_status()

    with pytest.raises(Exception):
        legal_status(CASRN=1648727814)

    with pytest.raises(Exception):
        legal_status(CASRN='1648727-81-4', Method='BADMETHOD')

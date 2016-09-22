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
import pandas as pd

from thermo.identifiers import checkCAS
from thermo.reaction import *
from thermo.heat_capacity import TRC_gas_data


def test_API_TDB_data():
    assert API_TDB_data.index.is_unique
    assert API_TDB_data['Hf'].abs().sum() == 101711260
    assert API_TDB_data.shape == (571, 2)
    assert all([checkCAS(i) for i in API_TDB_data.index])


def test_ATcT_l():
    assert ATcT_l.index.is_unique
    assert ATcT_l.shape == (34,5)
    assert all([checkCAS(i) for i in ATcT_l.index])
    tots_calc = [ATcT_l[i].abs().sum() for i in ['Hf_0K', 'Hf_298K', 'uncertainty']]
    tots = [2179500.0, 6819443, 19290]
    assert_allclose(tots_calc, tots)


def test_ATcT_g():
    assert ATcT_g.index.is_unique
    assert ATcT_g.shape == (595, 5)
    assert all([checkCAS(i) for i in ATcT_g.index])
    tots_calc = [ATcT_g[i].abs().sum() for i in ['Hf_0K', 'Hf_298K', 'uncertainty']]
    tots = [300788330, 300592764, 829204]
    assert_allclose(tots_calc, tots)


def test_Hf():
    Hfs = [Hf('7732-18-5'), Hf('7732-18-5', Method='API_TDB')]
    assert_allclose(Hfs, [-241820.0]*2)

    assert Hf('7732-18-5', AvailableMethods=True) == ['API_TDB', 'NONE']

    assert None == Hf('98-00-0')

    tot = sum([abs(Hf(i)) for i in API_TDB_data.index])
    assert_allclose(tot, 101711260.0)

    with pytest.raises(Exception):
        Hf('98-00-0', Method='BADMETHOD')


def test_Hf_l():
    Hfs = [Hf_l('67-56-1'), Hf_l('67-56-1', Method='ATCT_L')]
    assert_allclose(Hfs, [-238400.0]*2)

    assert Hf_l('67-56-1', AvailableMethods=True) == ['ATCT_L', 'NONE']
    assert None == Hf_l('98-00-0')

    tot = sum([abs(Hf_l(i)) for i in ATcT_l.index])
    assert_allclose(tot, 6819443.0)

    with pytest.raises(Exception):
        Hf_l('98-00-0', Method='BADMETHOD')


def test_Hf_g():
    Hfs = [Hf_g('67-56-1', Method=i) for i in Hf_g_methods]
    assert_allclose(Hfs, [-200700.0, -190100.0])

    assert Hf_g('67-56-1', AvailableMethods=True) == ['ATCT_G', 'TRC', 'NONE']
    assert None == Hf_g('98-00-0')

    with pytest.raises(Exception):
        Hf_g('98-00-0', Method='BADMETHOD')

    tot1 = sum([abs(Hf_g(i, Method='TRC')) for i in TRC_gas_data.index[pd.notnull(TRC_gas_data['Hf'])]])
    assert_allclose(tot1, 495689880.0)

    tot2 = sum([abs(Hf_g(i, Method='ATCT_G')) for i in ATcT_g.index])
    assert_allclose(tot2, 300592764.0)

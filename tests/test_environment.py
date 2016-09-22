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
import numpy as np
import pandas as pd
from thermo.environment import *


def test_GWP_data():
    dat_calc = [GWP_data[i].sum() for i in [u'Lifetime, years', u'Radiative efficienty, W/m^2/ppb', u'SAR 100yr', u'20yr GWP', u'100yr GWP', u'500yr GWP']]
    dat = [85518.965000000011, 17.063414000000002, 128282.0, 288251, 274671.70000000001, 269051.29999999999]
    assert_allclose(dat_calc, dat)


def test_GWP():
    GWP1_calc = GWP(CASRN='74-82-8')
    GWP2_calc = GWP(CASRN='74-82-8', Method='IPCC (2007) 100yr-SAR')
    assert [GWP1_calc, GWP2_calc] == [25.0, 21.0]

    GWP_available = GWP(CASRN='56-23-5', AvailableMethods=True)
    assert GWP_available == ['IPCC (2007) 100yr', 'IPCC (2007) 100yr-SAR', 'IPCC (2007) 20yr', 'IPCC (2007) 500yr', 'NONE']
    tot = pd.DataFrame( [GWP(i, Method=j) for i in GWP_data.index for j in GWP(i, AvailableMethods=True)]).sum()
    assert_allclose(tot, 960256)

    with pytest.raises(Exception):
        GWP(CASRN='74-82-8', Method='BADMETHOD')


def test_logP_data():
    tot = np.abs(CRClogPDict['logP']).sum()
    assert_allclose(tot, 1216.99)
    assert CRClogPDict.index.is_unique

    tot = np.abs(SyrresDict2['logP']).sum()
    assert_allclose(tot, 25658.06)
    assert SyrresDict2.index.is_unique


def test_logP():
    vals = logP('67-56-1'), logP('124-18-5'), logP('7732-18-5')
    assert_allclose(vals, [-0.74, 6.25, -1.38])

    tot_CRC = np.sum(np.abs(np.array([logP(i, Method='CRC') for i in CRClogPDict.index])))
    assert_allclose(tot_CRC, 1216.99)

    tot_SYRRES = np.sum(np.abs(np.array([logP(i, Method='SYRRES') for i in SyrresDict2.index])))
    assert_allclose(tot_SYRRES, 25658.060000000001)

    with pytest.raises(Exception):
        logP(CASRN='74-82-8', Method='BADMETHOD')

    logP_available = logP('110-54-3', AvailableMethods=True)
    assert logP_available == ['CRC', 'SYRRES', 'NONE']

    assert logP('1124321250-54-3') == None


def test_ODP_data():

    dat_calc = [ODP_data[i].sum() for i in ['ODP2 Max', 'ODP2 Min', 'ODP1 Max', 'ODP1 Min', 'ODP2 Design', 'ODP1 Design', 'Lifetime']]
    dat = [77.641999999999996, 58.521999999999998, 64.140000000000001, 42.734000000000002, 63.10509761272651, 47.809027930358717, 2268.1700000000001]
    assert_allclose(dat_calc, dat)

    assert ODP_data.index.is_unique


def test_ODP():
    V1 = ODP(CASRN='460-86-6')
    V2 = ODP(CASRN='76-14-2', Method='ODP2 Max')
    V3 = ODP(CASRN='76-14-2', Method='ODP1 Max')
    assert_allclose([V1, V2, V3], [7.5, 0.58, 1.0])

    assert ODP(CASRN='148875-98-3', Method='ODP2 string') == '0.2-2.1'

    methods = ['ODP2 Max', 'ODP1 Max', 'ODP2 logarithmic average', 'ODP1 logarithmic average', 'ODP2 Min', 'ODP1 Min', 'ODP2 string', 'ODP1 string', 'NONE']

    assert methods == ODP(CASRN='148875-98-3', AvailableMethods=True)

    with pytest.raises(Exception):
        ODP(CASRN='148875-98-3', Method='BADMETHOD')

    assert ODP(CASRN='14882353275-98-3') == None

    dat_calc = [pd.to_numeric(pd.Series([ODP(i, Method=j) for i in ODP_data.index]), errors='coerce').sum() for j in ODP_methods]

    dat = [77.641999999999996, 64.140000000000001, 63.10509761272651, 47.809027930358717, 58.521999999999998, 42.734000000000002, 54.342000000000006, 38.280000000000001]
    assert_allclose(dat_calc, dat)
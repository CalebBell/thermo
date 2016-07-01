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

from thermo.miscdata import *
from thermo.miscdata import _VDISaturationDict
from thermo.identifiers import checkCAS


### CRC Inorganic compounds

def test_CRC_inorganic():
    tots_calculated =  [CRC_inorganic_data[i].sum() for i in ['Tm', 'Tb', 'rho']]
    tots = [1543322.6125999999, 639213.23099999991, 8767127.4880000018]
    assert_allclose(tots, tots_calculated)

    assert CRC_inorganic_data.index.is_unique
    assert CRC_inorganic_data.shape == (2438, 4)
    assert all([checkCAS(i) for i in CRC_inorganic_data.index])


def test_CRC_organic():
    tots_calc = [CRC_organic_data[i].sum() for i in ['Tm', 'Tb', 'rho', 'RI']]
    tots = [2571284.4804000002, 2280667.0800000001, 6020405.0616999995, 6575.4144047999998]
    assert_allclose(tots_calc, tots)

    assert CRC_organic_data.index.is_unique
    assert CRC_organic_data.shape == (10867, 5)
    assert all([checkCAS(i) for i in CRC_organic_data.index])


def test_VDI_tabular_data():
    dat = VDI_tabular_data('67-56-1', 'Mu (g)')
    sTs, sprops = sum(dat[0]), sum(dat[1])
    assert_allclose([sTs, sprops], [2887.63, 0.0001001])

    with pytest.raises(Exception):
        VDI_tabular_data('67-56-1000', 'Mu (g)')
    with pytest.raises(Exception):
        VDI_tabular_data('67-56-1', 'Mug')


    prop_keys = ['Mu (g)', 'P', 'Pr (l)', 'Cp (l)', 'K (l)', 'Volume (l)', 'Beta', 'Density (l)', 'Pr (g)', 'Hvap', 'Density (g)', 'Mu (l)', 'Volume (g)', 'Cp (g)', 'sigma', 'K (g)', 'T']
    sums = {}
    for i in prop_keys:
        sums[i] = 0

    for prop in prop_keys:
        for CASRN in sorted(_VDISaturationDict.keys()):
            sums[prop]+= sum(VDI_tabular_data(CASRN, prop)[1])

    sums_calc = {'Volume (g)': 4480967.380663272, 'Mu (g)': 0.01092262, 'K (g)': 13.3338, 'P': 830362576.5100002, 'Pr (l)': 2561.917999999999, 'Cp (l)': 109652.14080415797, 'K (l)': 62.80901, 'T': 210960.66, 'Volume (l)': 0.06622819718826786, 'Beta': 3.66542, 'Pr (g)': 2561.917999999999, 'Hvap': 11146818.439615589, 'Mu (l)': 0.14801666, 'Cp (g)': 215899.05447308993, 'sigma': 9.08229, 'Density (l)': 510315.80999999994, 'Density (g)': 33258.34707901}
    assert sums_calc == sums
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
from thermo.temperature import *


def test_data():
    Ts_sums_calc = [i.sum() for i in [Ts_68, Ts_48, Ts_76, Ts_27]]
    Ts_sums = [186818.69999999998, 175181.39999999997, 368, 133893.09999999998]
    assert_allclose(Ts_sums_calc, Ts_sums)
    diffs_sums_calc = [abs(i).sum() for i in [diffs_68, diffs_48, diffs_76, diffs_27]]
    diffs_sums = [46.304000000000016, 151.31800000000001, 0.038800000000000001, 411.17999999999995]
    assert_allclose(diffs_sums_calc, diffs_sums)


def test_conversion():
    T2 = T_converter(500, 'ITS-68', 'ITS-48')
    assert_allclose(T2, 499.9470092992346)

    high_scales = ['ITS-90', 'ITS-68', 'ITS-27', 'ITS-48']

    for scale1 in high_scales:
        for scale2 in high_scales:
            T = T_converter(1000, scale1, scale2)
            assert_allclose(T_converter(T, scale2, scale1), 1000)

    mid_scales = ['ITS-90', 'ITS-68', 'ITS-48']

    for Ti in range(100, 1000, 200):
        for scale1 in mid_scales:
            for scale2 in mid_scales:
                T = T_converter(Ti, scale1, scale2)
                assert_allclose(T_converter(T, scale2, scale1), Ti, rtol=1e-6)

    low_scales = ['ITS-90', 'ITS-68', 'ITS-76']

    for Ti in range(15, 27, 2):
        for scale1 in low_scales:
            for scale2 in low_scales:
                T = T_converter(Ti, scale1, scale2)
                assert_allclose(T_converter(T, scale2, scale1), Ti)

    with pytest.raises(Exception):
        T_converter(10, 'ITS-27', 'ITS-48')

    with pytest.raises(Exception):
        T_converter(10, 'FAIL', 'ITS-48')

    with pytest.raises(Exception):
        T_converter(10, 'ITS-76', 'FAIL')


def test_diff_68():
    dTs_calc = [ITS90_68_difference(i) for i in [13.7, 70, 80.5, 298.15, 1000, 1500]]


    dTs = [0, 0.006818871618271216, 0, -0.006253950277664615,
           0.01231818956580355, -0.31455]
    assert_allclose(dTs, dTs_calc)
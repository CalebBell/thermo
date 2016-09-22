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
from thermo.triple import *


def test_data():
    Tt_sum = Staveley_data['Tt68'].sum()
    assert_allclose(Tt_sum, 31251.845000000001)

    Pt_sum = Staveley_data['Pt'].sum()
    assert_allclose(Pt_sum, 1886624.8374376972)

    Pt_uncertainty_sum = Staveley_data['Pt_uncertainty'].sum()
    assert_allclose(Pt_uncertainty_sum, 138.65526315789461)

    assert Staveley_data.index.is_unique
    assert Staveley_data.shape == (189, 5)


def test_Tt():
    Tt1_calc = Tt('7664-41-7')
    Tt1 = 195.48
    Tt2_calc = Tt('74-82-8', Method='MELTING')
    Tt2 = 90.75
    Tt3_calc = Tt('74-82-8')
    Tt3 = 90.69
    assert_allclose([Tt1_calc, Tt2_calc, Tt3_calc], [Tt1, Tt2, Tt3])

    m = Tt('7439-90-9', AvailableMethods=True)
    assert m == ['STAVELEY', 'MELTING', 'NONE']
    assert None == Tt('72433223439-90-9')
    with pytest.raises(Exception):
        Tt('74-82-8', Method='BADMETHOD')

    Tt_sum = sum([Tt(i) for i in Staveley_data.index])
    assert_allclose(Tt_sum, 31251.845000000001)
    Tt_sum2 = pd.Series([Tt(i, Method='MELTING') for i in Staveley_data.index]).sum()
    assert_allclose(Tt_sum2, 28778.196000000004)


def test_Pt():
    Pt1_calc = Pt('7664-41-7')
    Pt1 = 6079.5
    Pt2_calc = Pt('7664-41-7', Method='DEFINITION')
    Pt2 = 6042.920357447978
    # Add a test back for water if you allow extrapolation, letting water have
    # its triple pressure back

    assert_allclose([Pt1_calc, Pt2_calc], [Pt1, Pt2])

    m = Pt('7664-41-7', AvailableMethods=True)
    assert m == ['STAVELEY', 'DEFINITION', 'NONE']
    assert None == Pt('72433223439-90-9')
    with pytest.raises(Exception):
        Pt('74-82-8', Method='BADMETHOD')

    Pt_sum = sum([Pt(i) for i in Staveley_data.index if pd.notnull(Staveley_data.at[i, 'Pt'])])
    assert_allclose(Pt_sum, 1886624.8374376972)

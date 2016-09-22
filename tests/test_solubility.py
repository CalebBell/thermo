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
from thermo.solubility import *


def test_solubility():
    # From [1]_, matching examples 1 and 2.
    x1 = solubility_eutectic(293.15, 369.4, 18640., 0, 0, 1)
    x2 = solubility_eutectic(T=260., Tm=278.68, Hm=9952., Cpl=0, Cps=0, gamma=3.0176)
    x3 = solubility_eutectic(T=260., Tm=278.68, Hm=9952., Cpl=195, Cps=60, gamma=3.0176)
    assert_allclose([x1, x2, x3], [0.20626904089059944, 0.24340068761677464, 0.2533343503913555])

    dTm1 = Tm_depression_eutectic(353.35, 19110, 0.02)
    dTm2 = Tm_depression_eutectic(353.35, 19110, M=0.4, MW=40.)
    assert_allclose([dTm1, dTm2], [1.0864594900639515, 0.8691675920511612])
    with pytest.raises(Exception):
        Tm_depression_eutectic(353.35, 19110)


def test_solubility_parameter():
    delta = solubility_parameter(T=298.2, Hvapm=26403.3, Vml=0.000116055)
    assert delta == solubility_parameter(298.2, 26403.3, 0.000116055)
    assert_allclose(delta, 14357.681538173534)

    assert None == solubility_parameter(298.2, 26403.3, Method='DEFINITION')
    assert None == solubility_parameter(T=298.2, Vml=0.000116055, Method='DEFINITION')
    assert None == solubility_parameter(T=298.2, Vml=0.000116055)
    assert ['DEFINITION', 'NONE'] == solubility_parameter(T=298.2, Hvapm=26403.3, Vml=0.000116055, AvailableMethods=True)

    with pytest.raises(Exception):
        solubility_parameter(CASRN='132451235-2151234-1234123', Method='BADMETHOD')

    assert None == solubility_parameter(T=3500.2, Hvapm=26403.3, Vml=0.000116055)

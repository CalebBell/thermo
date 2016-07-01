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

from thermo.dippr import *

def test_Eqs():
    a = EQ100(300, 276370., -2090.1, 8.125, -0.014116, 0.0000093701)
    assert_allclose(a, 75355.81)

    a = EQ101(300, 73.649, -7258.2, -7.3037, 4.1653E-6, 2)
    assert_allclose(a, 3537.44834545549)

    a = EQ102(300, 1.7096E-8, 1.1146, 0, 0)
    assert_allclose(a, 9.860384711890639e-06)

    a = EQ104(300, 0.02222, -26.38, -16750000, -3.894E19, 3.133E21)
    assert_allclose(a, -1.1204179007265151)

    a = EQ105(300., 0.70824, 0.26411, 507.6, 0.27537)
    assert_allclose(a, 7.593170096339236)

    a = EQ106(300, 647.096, 0.17766, 2.567, -3.3377, 1.9699)
    assert_allclose(a, 0.07231499373541)

    a = EQ107(300., 33363., 26790., 2610.5, 8896., 1169.)
    assert_allclose(a, 33585.90452768923)

    a = EQ114(20, 33.19, 66.653, 6765.9, -123.63, 478.27)
    assert_allclose(a, 19423.948911676463)

    a = EQ116(300., 647.096, 17.863, 58.606, -95.396, 213.89, -141.26)
    assert_allclose(a, 55.17615446406527)

    a = EQ127(20., 3.3258E4, 3.6199E4, 1.2057E3, 1.5373E7, 3.2122E3, -1.5318E7, 3.2122E3)
    assert_allclose(a, 33258.0)

    # Random coefficients
    a = EQ115(300, 0.01, 0.002, 0.0003, 0.00004)
    assert_allclose(a, 37.02960772416336)
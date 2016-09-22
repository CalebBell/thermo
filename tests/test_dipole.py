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
from thermo.dipole import *


def test_dipole_moment_methods():

    tot = _dipole_Poling['Dipole'].sum()
    assert_allclose(tot, 248.59999999999999)

    tot = _dipole_CCDB['Dipole'].sum()
    assert_allclose(tot, 632.97000000000003)

    tot = _dipole_Muller['Dipole'].sum()
    assert_allclose(tot, 420.05190108045235)

    assert _dipole_CCDB.index.is_unique
    assert _dipole_Muller.index.is_unique
    assert _dipole_Poling.index.is_unique


def test_dipole():
    d = dipole_moment(CASRN='64-17-5')
    assert_allclose(d, 1.44)

    d = dipole_moment(CASRN='75-52-5', Method='POLING')
    assert_allclose(d, 3.1)

    d = dipole_moment(CASRN='56-81-5', Method='MULLER')
    assert_allclose(d, 4.21)

    methods = dipole_moment(CASRN='78-78-4', AvailableMethods=True)
    methods_fixed = ['CCCBDB', 'MULLER', 'POLING', 'NONE']
    assert methods == methods_fixed

    assert None == dipole_moment(CASRN='78-78-4', Method='NONE')

    with pytest.raises(Exception):
        dipole_moment(CASRN='78-78-4', Method='FAIL')
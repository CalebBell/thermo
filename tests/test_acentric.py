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
from thermo.acentric import *
from thermo.acentric import _crit_PSRKR4, _crit_PassutDanner, _crit_Yaws


@pytest.mark.slow
def test_acentric_main():
    sources = [_crit_PSRKR4, _crit_PassutDanner, _crit_Yaws]

    CASs = set()
    [CASs.update(list(k.index.values)) for k in sources]

    # Use the default method for each chemical in this file
    omegas = [omega(i, IgnoreMethods=None) for i in CASs] # This is quite slow
    omegas_default_sum = pd.Series(omegas).sum()
    assert_allclose(omegas_default_sum, 3520.538831100246)

    omega_calc = omega('629-92-5', Method='DEFINITION')
    assert_allclose(omega_calc, 0.84483960799853519)

    omega_calc = omega('629-92-5', Method='PSRK')
    assert_allclose(omega_calc, 0.8486)

    omega_calc = omega('629-92-5', Method='PD')
    assert_allclose(omega_calc, 0.8271)

    methods = omega('74-98-6', AvailableMethods=True, IgnoreMethods=None)
    assert methods == ['PSRK', 'PD', 'YAWS', 'LK', 'DEFINITION', 'NONE']

    # Error handling
    assert None == omega(CASRN='BADCAS')
    with pytest.raises(Exception):
        omega(CASRN='98-01-1', Method='BADMETHOD')



def test_acentric_correlation():
    omega = LK_omega(425.6, 631.1, 32.1E5)
    assert_allclose(omega, 0.32544249926397856)


def test_acentric_mixture():
    o = omega_mixture([0.025, 0.12], [0.3, 0.7])
    assert_allclose(o, 0.0915)

    methods = omega_mixture([0.025, 0.12], [0.3, 0.7], AvailableMethods=True)
    assert methods == ['SIMPLE', 'NONE']

    assert ['NONE'] == omega_mixture([0.025, 0.12], [], AvailableMethods=True)
    assert None == omega_mixture([0.025, 0.12], [])

    with pytest.raises(Exception):
        omega_mixture([0.025, 0.12], [0.3, 0.7], Method='Fail')


def test_StielPolar():
    factor = StielPolar(647.3, 22048321.0, 0.344, CASRN='7732-18-5')
    assert_allclose(factor, 0.02458114034873482)

    assert ['DEFINITION', 'NONE'] == StielPolar(647.3, 22048321.0, 0.344, CASRN='7732-18-5', AvailableMethods=True)

    # Test missing Pc
    assert None == StielPolar(647.3, 22048321)
    # Test Tc, Pc, omega available - with no vapor pressure data
    assert None == StielPolar(647.3, 22048321.0, 0.344, CASRN='58-08-2')

    with pytest.raises(Exception):
        StielPolar(647.3, 22048321.0, 0.344, CASRN='58-08-2', Method='FAIL')
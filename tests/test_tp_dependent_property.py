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
SOFTWARE.
'''
import pytest

from thermo.utils import TPDependentProperty


def test_tp_dependent_property_exceptions():
    BAD_METHOD = 'BAD_METHOD'
    CRAZY_METHOD = 'CRAZY_METHOD'
    class MockHeatCapacity(TPDependentProperty):
        name = 'Heat capacity'
        units = 'J/mol'
        ranked_methods_P = [BAD_METHOD, CRAZY_METHOD]

        def __init__(self, extrapolation, CASRN, **kwargs):
            self.CASRN = CASRN
            super().__init__(extrapolation, **kwargs)

        def load_all_methods(self, load_data):
            if load_data:
                self.T_limits = {BAD_METHOD: (300., 500.),
                                 CRAZY_METHOD: (300, 500)}
                self.all_methods = set()
                self.all_methods_P = {BAD_METHOD, CRAZY_METHOD}

        def calculate_P(self, T, P, method):
            if method == BAD_METHOD:
                raise Exception('BAD CALCULATION')
            elif method == CRAZY_METHOD:
                return -1e6
            else:
                return self._base_calculate(T, method)

    MVP = MockHeatCapacity(extrapolation='linear', CASRN='7732-18-5')
    MVP.RAISE_PROPERTY_CALCULATION_ERROR = True
    with pytest.raises(RuntimeError):
        try:
            MVP.TP_dependent_property(340., 101325.)
        except RuntimeError as error:
            assert str(error) == ("Failed to evaluate heat capacity method "
                                  "'BAD_METHOD' at T=340.0 K and P=101325.0 Pa "
                                  "for component with CASRN '7732-18-5'")
            raise error
    with pytest.raises(RuntimeError):
        try:
            MVP.TP_dependent_property(520., 101325.)
        except RuntimeError as error:
            assert str(error) == ("Heat capacity method 'BAD_METHOD' is not "
                                  "valid at T=520.0 K and P=101325.0 Pa for "
                                  "component with CASRN '7732-18-5'")
            raise error
    MVP.method_P = CRAZY_METHOD
    with pytest.raises(RuntimeError):
        try:
            MVP.TP_dependent_property(350., 101325.)
        except RuntimeError as error:
            assert str(error) == ("Heat capacity method 'CRAZY_METHOD' "
                                  "computed an invalid value of -1000000.0 J/mol "
                                  "for component with CASRN '7732-18-5'")
            raise error
    MVP.method_P = None
    with pytest.raises(RuntimeError):
        try:
            MVP.TP_dependent_property(340., 101325.)
        except RuntimeError as error:
            assert str(error) == ("No pressure-dependent heat capacity method "
                                  "selected for component with CASRN '7732-18-5'")
            raise error

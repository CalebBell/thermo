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
from math import log

import pytest
from fluids.numerics import assert_close

from thermo.utils import TDependentProperty


def test_local_constant_method():
    # Test user defined method

    # Within valid T range
    obj = TDependentProperty(extrapolation='linear')
    constant = 100.
    T1 = T = 300.
    T2 = 310.
    dT = T2 - T1
    obj.add_method(constant)
    assert_close(obj.T_dependent_property(T), constant)
    for order in (1, 2, 3):
        assert_close(obj.T_dependent_property_derivative(T, order), 0.)
    assert_close(obj.T_dependent_property_integral(T1, T2), constant * dT)
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), constant * log(T2 / T1))

    # Extrapolate
    Tmin = 350.
    Tmax = 400.
    obj.add_method(constant, Tmin, Tmax)
    obj.extrapolation = 'constant'
    assert_close(obj.T_dependent_property(T), constant)
    for order in (1, 2, 3):
        assert_close(obj.T_dependent_property_derivative(T, order), 0., atol=1e-6)
    assert_close(obj.T_dependent_property_integral(T1, T2), constant * dT)
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), constant * log(T2 / T1))

    # Test extrapolation on both sides
    assert_close(obj.T_dependent_property_integral(Tmin - 50, Tmax + 50), constant * (100 + Tmax - Tmin))
    assert_close(obj.T_dependent_property_integral_over_T(Tmin - 50, Tmax + 50), constant * log((Tmax + 50) / (Tmin - 50)))

    # Do not allow extrapolation
    obj.extrapolation = None
    obj.add_method(constant, Tmin, Tmax)
    assert obj.T_dependent_property(T) is None

def test_extrapolation():
    obj = TDependentProperty(extrapolation='constant')
    Tmin = 300
    Tmax = 400
    f = lambda T: T
    f_int = lambda T1, T2: (T2*T2 - T1*T1) / 2.
    f_int_over_T = lambda T1, T2: T2 - T1
    obj.add_method(f=f, Tmin=Tmin, Tmax=Tmax)

    with pytest.raises(ValueError):
        obj.extrapolate(350, obj.method, in_range='error')

    T_low = Tmin - 100
    T_inrange = (Tmin + Tmax) * 0.5
    T_high = Tmax + 100
    assert_close(Tmin, obj.extrapolate(T_low, obj.method, in_range='error'))
    assert_close(Tmax, obj.extrapolate(T_high, obj.method, in_range='error'))
    assert_close(Tmin, obj.extrapolate(T_low, obj.method, in_range='low'))
    assert_close(Tmin, obj.extrapolate(T_inrange, obj.method, in_range='low'))
    assert_close(Tmax, obj.extrapolate(T_high, obj.method, in_range='low'))
    assert_close(Tmin, obj.extrapolate(T_low, obj.method, in_range='high'))
    assert_close(Tmax, obj.extrapolate(T_inrange, obj.method, in_range='high'))
    assert_close(Tmax, obj.extrapolate(T_high, obj.method, in_range='high'))

def test_integrals():
    obj = TDependentProperty(extrapolation='constant')
    Tmin = 300
    Tmax = 400
    f = lambda T: T + 10.
    f_int = lambda T1, T2: (T2*T2 - T1*T1) * 0.5 + 10 * (T2 - T1)
    f_int_over_T = lambda T1, T2: T2 - T1 + 10 * log(T2/T1)
    obj.add_method(f=f, f_int=f_int, f_int_over_T=f_int_over_T, Tmin=Tmin, Tmax=Tmax)

    # Within valid T range
    T1 = 300.
    T2 = 310.
    dT = T2 - T1
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, T2))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, T2))

    # Extrapolate left
    T1 = Tmin - 50
    T2 = Tmin - 25
    dT = T2 - T1
    constant_low = f(Tmin)
    assert_close(obj.T_dependent_property_integral(T1, T2), constant_low * dT)
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), constant_low * log(T2/T1))

    # Extrapolate right
    T1 = Tmax + 25
    T2 = Tmax + 50
    dT = T2 - T1
    constant_high = f(Tmax)
    assert_close(obj.T_dependent_property_integral(T1, T2), constant_high * dT)
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), constant_high * log(T2/T1))

    # Extrapolate left to center (piece-wise)
    T1 = Tmin - 50
    T2 = Tmin + 25
    constant_low = f(Tmin)
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(Tmin, T2) + constant_low * (Tmin - T1))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(Tmin, T2) + constant_low * log(Tmin/T1))

    # Extrapolate right to center (piece-wise)
    T1 = Tmax - 25
    T2 = Tmax + 50
    constant_high = f(Tmax)
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, Tmax) + constant_high * (T2 - Tmax))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, Tmax) + constant_high * log(T2/Tmax))

    # Extrapolate across both sides (piece-wise)
    T1 = Tmin - 50
    T2 = Tmax + 50
    constant_low = f(Tmin)
    constant_high = f(Tmax)
    assert_close(obj.T_dependent_property_integral(T1, T2), constant_low * (Tmin - T1) + f_int(Tmin, Tmax) + constant_high * (T2 - Tmax))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), constant_low * log(Tmin/T1) + f_int_over_T(Tmin, Tmax) + constant_high * log(T2/Tmax))

    ### Test linear extrapolation
    obj.extrapolation = 'linear'

    # Extrapolate left
    T1 = Tmin - 50
    T2 = Tmin - 25
    dT = T2 - T1
    constant_low = f(Tmin)
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, T2))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, T2))

    # Extrapolate right
    T1 = Tmax + 25
    T2 = Tmax + 50
    dT = T2 - T1
    constant_high = f(Tmax)
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, T2))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, T2))

    # Extrapolate left to center (piece-wise)
    T1 = Tmin - 50
    T2 = Tmin + 25
    constant_low = f(Tmin)
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, T2))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, T2))

    # Extrapolate right to center (piece-wise)
    T1 = Tmax - 25
    T2 = Tmax + 50
    constant_high = f(Tmax)
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, T2))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, T2))

    # Extrapolate across both sides (piece-wise)
    T1 = Tmin - 50
    T2 = Tmax + 50
    constant_low = f(Tmin)
    constant_high = f(Tmax)
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, T2))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, T2))

def test_derivative():
    obj = TDependentProperty(extrapolation='linear')
    Tmin = 300
    Tmax = 400
    f = lambda T: T*T*T
    f_der = lambda T: 3*T*T
    f_der2 = lambda T: 6*T
    f_der3 = lambda T: 6
    obj.add_method(f=f, f_der=f_der, f_der2=f_der2, f_der3=f_der3, Tmin=Tmin, Tmax=Tmax)


    T_in_range = (Tmin + Tmax) * 0.5
    for order, fun in zip([1,2,3], [f_der, f_der2, f_der3]):
        # Within valid T range
        assert_close(obj.T_dependent_property_derivative(T_in_range, order=order), fun(T_in_range))

    # Extrapolate left
    T = Tmin - 50
    obj.RAISE_PROPERTY_CALCULATION_ERROR = True
    assert_close(obj.T_dependent_property_derivative(T, order=1), f_der(Tmin))

    # Extrapolate right
    T = Tmax + 50
    assert_close(obj.T_dependent_property_derivative(T, order=1), f_der(Tmax))


def test_local_method():
    # Test user defined method

    # Within valid T range
    obj = TDependentProperty(extrapolation='linear')
    T = 300.
    Tmin = 200.
    Tmax = 400.
    T1 = 300.
    T2 = 310.
    dT = T2 - T1
    f = lambda T: T*T / 1e6
    f_der = lambda T: T / 1e6
    f_der2 = lambda T: 1. / 1e6
    f_der3 = lambda T: 0. / 1e6
    f_int = lambda T1, T2: (T2*T2*T2 - T1*T1*T1) / 3. / 1e6
    f_int_over_T = lambda T1, T2: (T2*T2 - T1*T1) / 2. / 1e6
    obj.add_method(f, Tmin, Tmax, f_der, f_der2, f_der3, f_int, f_int_over_T)
    assert_close(obj.T_dependent_property(T), f(T))
    assert_close(obj.T_dependent_property_derivative(T, 1), f_der(T))
    assert_close(obj.T_dependent_property_derivative(T, 2), f_der2(T))
    assert_close(obj.T_dependent_property_derivative(T, 3), f_der3(T))
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, T2))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, T2))

    # Extrapolate
    Tmin = 300. + 1e-3
    obj.add_method(f, Tmin, Tmax)
    # obj.RAISE_PROPERTY_CALCULATION_ERROR = True
    # obj.CASRN = 'CASRN'
    assert_close(obj.T_dependent_property(T), f(T))
    for order in (1, 2, 3):
        assert obj.T_dependent_property_derivative(T, order) is not None
    assert_close(obj.T_dependent_property_integral(T1, T2), f_int(T1, T2))
    assert_close(obj.T_dependent_property_integral_over_T(T1, T2), f_int_over_T(T1, T2))

    # Do not allow extrapolation
    obj.extrapolation = None
    obj.add_method(f, Tmin, Tmax)
    assert obj.T_dependent_property(T) is None
    assert obj.T_dependent_property_integral(T1, T2) is None
    assert obj.T_dependent_property_integral_over_T(T1, T2) is None

def test_many_local_methods():
    obj = TDependentProperty(extrapolation='linear')
    M1_value = 2.0
    M2_value = 3.0
    obj.add_method(M1_value, name='M1')
    obj.add_method(M2_value, name='M2')
    obj.method = 'M1'
    assert_close(M1_value, obj.T_dependent_property(300.))
    obj.method = 'M2'
    assert_close(M2_value, obj.T_dependent_property(300.))

def test_t_dependent_property_exceptions():
    BAD_METHOD = 'BAD_METHOD'
    CRAZY_METHOD = 'CRAZY_METHOD'
    class MockVaporPressure(TDependentProperty):
        name = 'Vapor pressure'
        units = 'Pa'
        ranked_methods = [BAD_METHOD, CRAZY_METHOD]

        def __init__(self, extrapolation, CASRN, **kwargs):
            self.CASRN = CASRN
            super().__init__(extrapolation, **kwargs)

        def load_all_methods(self, load_data):
            if load_data:
                self.T_limits = {BAD_METHOD: (300., 500.),
                                 CRAZY_METHOD: (300, 500)}
                self.all_methods = {BAD_METHOD, CRAZY_METHOD}

        def calculate(self, T, method):
            if method == BAD_METHOD:
                raise Exception('BAD CALCULATION')
            elif method == CRAZY_METHOD:
                return -1e6
            else:
                return self._base_calculate(T, method)

    MVP = MockVaporPressure(extrapolation='linear', CASRN='7732-18-5')
    MVP.RAISE_PROPERTY_CALCULATION_ERROR = True
    with pytest.raises(RuntimeError):
        try:
            MVP.T_dependent_property(340.)
        except RuntimeError as error:
            assert str(error) == ("Failed to evaluate vapor pressure method "
                                  "'BAD_METHOD' at T=340.0 K for component with "
                                  "CASRN '7732-18-5'")
            raise error
    with pytest.raises(RuntimeError):
        try:
            MVP.T_dependent_property(520.)
        except RuntimeError as error:
            assert str(error) == ("Failed to extrapolate vapor pressure method "
                                  "'BAD_METHOD' at T=520.0 K for component with "
                                  "CASRN '7732-18-5'")
            raise error
    MVP.extrapolation = None
    with pytest.raises(RuntimeError):
        try:
            MVP.T_dependent_property(520.)
        except RuntimeError as error:
            assert str(error) == ("Vapor pressure method 'BAD_METHOD' is not "
                                  "valid at T=520.0 K for component with CASRN "
                                  "'7732-18-5'")
            raise error
    MVP.method = CRAZY_METHOD
    with pytest.raises(RuntimeError):
        try:
            MVP.T_dependent_property(350.)
        except RuntimeError as error:
            assert str(error) == ("Vapor pressure method 'CRAZY_METHOD' "
                                  "computed an invalid value of -1000000.0 Pa "
                                  "for component with CASRN '7732-18-5'")
            raise error
    MVP.method = None
    with pytest.raises(RuntimeError):
        try:
            MVP.T_dependent_property(340.)
        except RuntimeError as error:
            assert str(error) == ("No vapor pressure method selected for "
                                  "component with CASRN '7732-18-5'")
            raise error


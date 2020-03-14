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
from pytest import approx
from thermo.steam_properties import get_steam_table_entry, PTVEntry, PhaseRegion, PhaseInfo

def inc(x):
    return x + 1

# temperature, pressure, enthalpy, entropy, phase_region, is_saturated, expected_is_success, expected_entry, expected_error_message
testdata = [
    # temperature and pressure
    (750, 78.309563916917e6, None, None, None, None, 
    True, 
    PTVEntry(
        _temperature=750, 
        _pressure=78.309563916917e6, 
        _phase_info=PhaseInfo(PhaseRegion.SupercriticalFluid, 0, 0, 0),
        _internal_energy=2102.069317626429e3,
        _enthalpy=2258.688445460262e3,
        _entropy=4.469719056217e3,
        _isochoric_heat_capacity=2.71701677121e3,
        _isobaric_heat_capacity=6.341653594791e3,
        _speed_of_sound=760.696040876798,
        _density=500
    ), 
    None),
    (473.15, 40e6, None, None, None, None, 
    True, 
    PTVEntry(
        _temperature=473.15, 
        _pressure=40e6, 
        _phase_info=PhaseInfo(PhaseRegion.Liquid, 0, 1, 0),
        _internal_energy=825.228016170348e3,
        _enthalpy=870.124259682489e3,
        _entropy=2.275752861241e3,
        _isochoric_heat_capacity=3.292858637199e3,
        _isobaric_heat_capacity=4.315767590903e3,
        _speed_of_sound=1457.418351596083,
        _density=1.0 / 0.001122406088
    ), 
    None),
    (2000, 30e6, None, None, None, None, 
    True, 
    PTVEntry(
        _temperature=2000, 
        _pressure=30e6, 
        _phase_info=PhaseInfo(PhaseRegion.SupercriticalFluid, 0, 0, 0),
        _internal_energy=5637.070382521894e3,
        _enthalpy=6571.226038618478e3,
        _entropy=8.536405231138e3,
        _isochoric_heat_capacity=2.395894362358e3,
        _isobaric_heat_capacity=2.885698818781e3,
        _speed_of_sound=1067.369478777425,
        _density=1.0 / 0.03113852187
    ), 
    None),
    (823.15, 14e6, None, None, None, None, 
    True, 
    PTVEntry(
        _temperature=823.15, 
        _pressure=14e6, 
        _phase_info=PhaseInfo(PhaseRegion.Gas, 0, 0, 0),
        _internal_energy=3114.302136294585e3,
        _enthalpy=3460.987255128561e3,
        _entropy=6.564768889364e3,
        _isochoric_heat_capacity=1.892708832325e3,
        _isobaric_heat_capacity=2.666558503968e3,
        _speed_of_sound=666.050616844223,
        _density=1.0 / 0.024763222774
    ), 
    None),
    # saturated and pressure
    (None, 0.2e6, None, None, PhaseRegion.Liquid, True, 
    True, 
    PTVEntry(
        _temperature=393.361545936488, 
        _pressure=0.2e6, 
        _phase_info=PhaseInfo(PhaseRegion.Liquid, 0, 1, 0),
        _internal_energy=504471.741847973,
        _enthalpy=504683.84552926,
        _entropy=1530.0982011075,
        _isochoric_heat_capacity=3666.99397284121,
        _isobaric_heat_capacity=4246.73524917536,
        _speed_of_sound=1520.69128792808,
        _density=1.0 / 0.00106051840643552
    ), 
    None),
    (None, 0.2e6, None, None, PhaseRegion.Vapor, True, 
    True, 
    PTVEntry(
        _temperature=393.361545936488, 
        _pressure=0.2e6, 
        _phase_info=PhaseInfo(PhaseRegion.Vapor, 1, 0, 0),
        _internal_energy=2529094.32835793,
        _enthalpy=2706241.34137425,
        _entropy=7126.8563914686,
        _isochoric_heat_capacity=1615.96336473298,
        _isobaric_heat_capacity=2175.22318865273,
        _speed_of_sound=481.883535821489,
        _density=1.0 / 0.885735065081644
    ), 
    None),
    # saturated and temperature
        (393.361545936488, None, None, None, PhaseRegion.Liquid, True, 
    True, 
    PTVEntry(
        _temperature=393.361545936488, 
        _pressure=0.2e6, 
        _phase_info=PhaseInfo(PhaseRegion.Liquid, 0, 1, 0),
        _internal_energy=504471.741847973,
        _enthalpy=504683.84552926,
        _entropy=1530.0982011075,
        _isochoric_heat_capacity=3666.99397284121,
        _isobaric_heat_capacity=4246.73524917536,
        _speed_of_sound=1520.69128792808,
        _density=1.0 / 0.00106051840643552
    ), 
    None),
    (393.361545936488, None, None, None, PhaseRegion.Vapor, True, 
    True, 
    PTVEntry(
        _temperature=393.361545936488, 
        _pressure=0.2e6, 
        _phase_info=PhaseInfo(PhaseRegion.Vapor, 1, 0, 0),
        _internal_energy=2529094.32835793,
        _enthalpy=2706241.34137425,
        _entropy=7126.8563914686,
        _isochoric_heat_capacity=1615.96336473298,
        _isobaric_heat_capacity=2175.22318865273,
        _speed_of_sound=481.883535821489,
        _density=1.0 / 0.885735065081644
    ), 
    None),
    # entropy and pressure
    (None, 78.309563916917e6, None, 4.469719056217e3, None, None, 
    True, 
    PTVEntry(
        _temperature=750, 
        _pressure=78.309563916917e6, 
        _phase_info=PhaseInfo(PhaseRegion.SupercriticalFluid, 0, 0, 0),
        _internal_energy=2102.069317626429e3,
        _enthalpy=2258.688445460262e3,
        _entropy=4.469719056217e3,
        _isochoric_heat_capacity=2.71701677121e3,
        _isobaric_heat_capacity=6.341653594791e3,
        _speed_of_sound=760.696040876798,
        _density=500
    ), 
    None),
    (None, 40e6, None, 2.275752861241e3, None, None, 
    True, 
    PTVEntry(
        _temperature=473.15, 
        _pressure=40e6, 
        _phase_info=PhaseInfo(PhaseRegion.Liquid, 0, 1, 0),
        _internal_energy=825.228016170348e3,
        _enthalpy=870.124259682489e3,
        _entropy=2.275752861241e3,
        _isochoric_heat_capacity=3.292858637199e3,
        _isobaric_heat_capacity=4.315767590903e3,
        _speed_of_sound=1457.418351596083,
        _density=1.0 / 0.001122406088
    ), 
    None),
    (None, 30e6, None, 8.536405231138e3, None, None, 
    True, 
    PTVEntry(
        _temperature=2000, 
        _pressure=30e6, 
        _phase_info=PhaseInfo(PhaseRegion.SupercriticalFluid, 0, 0, 0),
        _internal_energy=5637.070382521894e3,
        _enthalpy=6571.226038618478e3,
        _entropy=8.536405231138e3,
        _isochoric_heat_capacity=2.395894362358e3,
        _isobaric_heat_capacity=2.885698818781e3,
        _speed_of_sound=1067.369478777425,
        _density=1.0 / 0.03113852187
    ), 
    None),
    (None, 14e6, None, 6.564768889364e3, None, None, 
    True, 
    PTVEntry(
        _temperature=823.15, 
        _pressure=14e6, 
        _phase_info=PhaseInfo(PhaseRegion.Gas, 0, 0, 0),
        _internal_energy=3114.302136294585e3,
        _enthalpy=3460.987255128561e3,
        _entropy=6.564768889364e3,
        _isochoric_heat_capacity=1.892708832325e3,
        _isobaric_heat_capacity=2.666558503968e3,
        _speed_of_sound=666.050616844223,
        _density=1.0 / 0.024763222774
    ), 
    None),
    (None, 10e3, None, 6.6858e3, None, None, 
    True, 
    PTVEntry(
        _temperature=318.957548207023, 
        _pressure=10e3, 
        _phase_info=PhaseInfo(PhaseRegion.LiquidVapor, 0.8049124470781327, 0.1950875529218673, 0),
        _internal_energy=1999135.82661328,
        _enthalpy=2117222.94886314,
        _entropy=6.6858e3,
        _isochoric_heat_capacity=1966.28009225455,
        _isobaric_heat_capacity=2377.86300751001,
        _speed_of_sound=655.005141924186,
        _density=193.16103883
    ), 
    None),
    #enthalpy and pressure
    (None, 78.309563916917e6, 2258.688445460262e3, None, None, None, 
    True, 
    PTVEntry(
        _temperature=750, 
        _pressure=78.309563916917e6, 
        _phase_info=PhaseInfo(PhaseRegion.SupercriticalFluid, 0, 0, 0),
        _internal_energy=2102.069317626429e3,
        _enthalpy=2258.688445460262e3,
        _entropy=4.469719056217e3,
        _isochoric_heat_capacity=2.71701677121e3,
        _isobaric_heat_capacity=6.341653594791e3,
        _speed_of_sound=760.696040876798,
        _density=500
    ), 
    None),
    (None, 40e6, 870.124259682489e3, None, None, None, 
    True, 
    PTVEntry(
        _temperature=473.15, 
        _pressure=40e6, 
        _phase_info=PhaseInfo(PhaseRegion.Liquid, 0, 1, 0),
        _internal_energy=825.228016170348e3,
        _enthalpy=870.124259682489e3,
        _entropy=2.275752861241e3,
        _isochoric_heat_capacity=3.292858637199e3,
        _isobaric_heat_capacity=4.315767590903e3,
        _speed_of_sound=1457.418351596083,
        _density=1.0 / 0.001122406088
    ), 
    None),
    (None, 30e6, 6571.226038618478e3, None, None, None, 
    True, 
    PTVEntry(
        _temperature=2000, 
        _pressure=30e6, 
        _phase_info=PhaseInfo(PhaseRegion.SupercriticalFluid, 0, 0, 0),
        _internal_energy=5637.070382521894e3,
        _enthalpy=6571.226038618478e3,
        _entropy=8.536405231138e3,
        _isochoric_heat_capacity=2.395894362358e3,
        _isobaric_heat_capacity=2.885698818781e3,
        _speed_of_sound=1067.369478777425,
        _density=1.0 / 0.03113852187
    ), 
    None),
    (None, 14e6, 3460.987255128561e3, None, None, None, 
    True, 
    PTVEntry(
        _temperature=823.15, 
        _pressure=14e6, 
        _phase_info=PhaseInfo(PhaseRegion.Gas, 0, 0, 0),
        _internal_energy=3114.302136294585e3,
        _enthalpy=3460.987255128561e3,
        _entropy=6.564768889364e3,
        _isochoric_heat_capacity=1.892708832325e3,
        _isobaric_heat_capacity=2.666558503968e3,
        _speed_of_sound=666.050616844223,
        _density=1.0 / 0.024763222774
    ), 
    None),
    (None, 10e3, 2117222.94886314, None, None, None, 
    True, 
    PTVEntry(
        _temperature=318.957548207023, 
        _pressure=10e3, 
        _phase_info=PhaseInfo(PhaseRegion.LiquidVapor, 0.804912447078132, 0.195087552921868, 0),
        _internal_energy=1999135.82661328,
        _enthalpy=2117222.94886314,
        _entropy=6.6858e3,
        _isochoric_heat_capacity=1966.28009225455,
        _isobaric_heat_capacity=2377.86300751001,
        _speed_of_sound=655.005141924186,
        _density=193.16103883
    ), 
    None),
]

@pytest.mark.parametrize("temperature, pressure, enthalpy, entropy, phase_region, is_saturated, expected_is_success, expected_entry, expected_error_message", testdata)
def test_answer(temperature,
    pressure,
    enthalpy,
    entropy,
    phase_region,
    is_saturated,
    expected_is_success,
    expected_entry,
    expected_error_message):


    (actual_is_success, actual_entry, actual_error_message) = get_steam_table_entry(
        pressure=pressure, 
        temperature=temperature, 
        enthalpy=enthalpy, 
        entropy=entropy, 
        phase_region=phase_region, 
        is_saturated=is_saturated)

    assert actual_is_success == expected_is_success
    if expected_is_success:
        assert actual_entry is not None
        assert expected_entry is not None
        assert expected_entry.get_pressure() == approx(actual_entry.get_pressure())
        assert expected_entry.get_temperature() == approx(actual_entry.get_temperature())
        assert expected_entry.get_enthalpy() == approx(actual_entry.get_enthalpy())
        assert expected_entry.get_entropy() == approx(actual_entry.get_entropy())
        assert expected_entry.get_isochoric_heat_capacity() == approx(actual_entry.get_isochoric_heat_capacity())
        assert expected_entry.get_isobaric_heat_capacity() == approx(actual_entry.get_isobaric_heat_capacity())
        assert expected_entry.get_isobaric_heat_capacity() == approx(actual_entry.get_isobaric_heat_capacity())
        assert expected_entry.get_speed_of_sound() == approx(actual_entry.get_speed_of_sound())
        assert expected_entry.get_density() == approx(actual_entry.get_density())

        expect_phase_info = expected_entry.get_phase_info()
        actual_phase_info = actual_entry.get_phase_info()
        assert expect_phase_info is not None
        assert actual_phase_info is not None
        assert expect_phase_info.get_phase_region() == actual_phase_info.get_phase_region()
        assert expect_phase_info.get_vapor() == actual_phase_info.get_vapor()
        assert expect_phase_info.get_liquid() == actual_phase_info.get_liquid()
        assert expect_phase_info.get_solid() == actual_phase_info.get_solid()
    else :
        assert actual_entry is None
    assert actual_error_message == expected_error_message

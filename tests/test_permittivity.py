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

from math import isnan

import pytest
from chemicals.permittivity import permittivity_data_CRC
from fluids.numerics import assert_close, assert_close1d, linspace

from thermo.permittivity import *


@pytest.mark.meta_T_dept
def test_Permittivity_class():
    # Test some cases
    water = PermittivityLiquid(CASRN='7732-18-5')
    assert (False, False) == (water.test_property_validity(2000), water.test_property_validity(-10))

    with pytest.raises(Exception):
        water.test_method_validity(300, 'BADMETHOD' )

    epsilon = water.T_dependent_property(298.15)
    assert_close(epsilon, 78.40540700004574)
    assert PermittivityLiquid(CASRN='7732-18-5').all_methods == {'CRC', 'CRC_CONSTANT', 'IAPWS_PERMITTIVITY'}

    assert PermittivityLiquid(CASRN='132451235-2151234-1234123').all_methods == set()
    assert PermittivityLiquid(CASRN='132451235-2151234-1234123').T_dependent_property(300) is None

    assert False is PermittivityLiquid(CASRN='7732-18-5').test_method_validity(228.15, 'CRC_CONSTANT')
    assert False is PermittivityLiquid(CASRN='7732-18-5').test_method_validity(228.15, 'CRC')

    assert_close(PermittivityLiquid(CASRN='7732-18-5').calculate(T=300, method='CRC_CONSTANT'), 80.1)


    assert PermittivityLiquid(CASRN='100-06-1').all_methods == {'CRC_CONSTANT'}
    assert_close(PermittivityLiquid(CASRN='100-06-1').calculate(T=300, method='CRC_CONSTANT'), 17.3)

    # Tabular data
    w = PermittivityLiquid(CASRN='7732-18-5')
    Ts = linspace(273, 372, 10)
    permittivities = [87.75556413000001, 83.530500320000016, 79.48208925000003, 75.610330919999996, 71.915225330000013, 68.396772480000024, 65.05497237000003, 61.889825000000044, 58.901330369999997, 56.08948848]
    w.add_tabular_data(Ts=Ts, properties=permittivities)
    assert_close(w.T_dependent_property(305.), 75.95500925000006)
    w.extrapolation = 'linear'
    assert_close(w.T_dependent_property(200.), 116.38076077348933)

    w.extrapolation_coeffs.clear()
    assert w.T_dependent_property(800.0) is not None

    w.extrapolation = None
    assert w.T_dependent_property(200.) is None

    assert PermittivityLiquid.from_json(w.as_json()) == w

    # Case where a nan was stored
    obj = PermittivityLiquid(CASRN='57-10-3')
    assert not isnan(obj.T_limits['CRC_CONSTANT'][0])
    assert not isnan(obj.T_limits['CRC_CONSTANT'][1])

@pytest.mark.meta_T_dept
def test_Permittivity_class_from_method_kwargs():
    obj = PermittivityLiquid(CASRN='67-56-1')
    assert_close(PermittivityLiquid(**obj.as_method_kwargs())(300), obj(300), rtol=1e-10)

    obj = PermittivityLiquid(CASRN='7732-18-5')
    assert_close(PermittivityLiquid(**obj.as_method_kwargs())(300), obj(300), rtol=1e-10)

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.meta_T_dept
def test_Permittivity_class_fuzz():
    tot_constant = sum([PermittivityLiquid(CASRN=i).calculate(T=298.15, method='CRC_CONSTANT') for i in permittivity_data_CRC.index])
    assert_close(tot_constant, 13526.653700000023)



    sums_min, sums_avg, sums_max = 0, 0, 0
    for i in permittivity_data_CRC.index:
        a = PermittivityLiquid(CASRN=i)
        if 'CRC' in a.all_methods:
            sums_min += a.calculate(a.T_limits['CRC'][0], 'CRC')
            sums_avg += a.calculate((a.T_limits['CRC'][1]+a.T_limits['CRC'][0])/2., 'CRC')
            sums_max += a.calculate(a.T_limits['CRC'][1], 'CRC')
    assert_close1d([sums_min, sums_avg, sums_max], [10582.970609439253, 8312.897581451223, 6908.073524704013])

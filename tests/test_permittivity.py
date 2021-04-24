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

from fluids.numerics import assert_close, assert_close1d, linspace
import pytest
import pandas as pd
from math import isnan
import numpy as np
from thermo.permittivity import *
from chemicals.permittivity import permittivity_data_CRC


@pytest.mark.meta_T_dept
def test_Permittivity_class():
    # Test some cases
    water = PermittivityLiquid(CASRN='7732-18-5')
    assert (False, False) == (water.test_property_validity(2000), water.test_property_validity(-10))

    with pytest.raises(Exception):
        water.test_method_validity(300, 'BADMETHOD' )

    epsilon = water.T_dependent_property(298.15)
    assert_close(epsilon, 78.35530812232503)
    assert PermittivityLiquid(CASRN='7732-18-5').all_methods == set(['CRC', 'CRC_CONSTANT'])

    assert PermittivityLiquid(CASRN='132451235-2151234-1234123').all_methods == set()
    assert PermittivityLiquid(CASRN='132451235-2151234-1234123').T_dependent_property(300) is None

    assert False == PermittivityLiquid(CASRN='7732-18-5').test_method_validity(228.15, 'CRC_CONSTANT')
    assert False == PermittivityLiquid(CASRN='7732-18-5').test_method_validity(228.15, 'CRC')



    # Tabular data
    w = PermittivityLiquid(CASRN='7732-18-5')
    Ts = linspace(273, 372, 10)
    permittivities = [87.75556413000001, 83.530500320000016, 79.48208925000003, 75.610330919999996, 71.915225330000013, 68.396772480000024, 65.05497237000003, 61.889825000000044, 58.901330369999997, 56.08948848]
    w.add_tabular_data(Ts=Ts, properties=permittivities)
    assert_close(w.T_dependent_property(305.), 75.95500925000006)
    w.extrapolation = 'interp1d'
    assert_close(w.T_dependent_property(200.), 115.79462395999997)

    del w.interp1d_extrapolators
    assert w.T_dependent_property(800.0) is not None

    w.extrapolation = None
    assert w.T_dependent_property(200.) is None

    assert PermittivityLiquid.from_json(w.as_json()) == w

    # Case where a nan was stored
    obj = PermittivityLiquid(CASRN='57-10-3')
    assert not isnan(obj.CRC_Tmin)
    assert not isnan(obj.CRC_Tmax)

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
            sums_min += a.calculate(a.CRC_Tmin, 'CRC')
            sums_avg += a.calculate((a.CRC_Tmax+a.CRC_Tmin)/2., 'CRC')
            sums_max += a.calculate(a.CRC_Tmax, 'CRC')
    assert_close1d([sums_min, sums_avg, sums_max], [10582.970609439253, 8312.897581451223, 6908.073524704013])

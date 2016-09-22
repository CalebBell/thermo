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
import numpy as np
from thermo.permittivity import *

def test_permittivity_data():
    assert CRC_Permittivity_data.index.is_unique
    assert CRC_Permittivity_data.shape == (1303, 9)


@pytest.mark.slow
@pytest.mark.meta_T_dept
def test_Permittivity_class():
    # Test some cases
    epsilon = Permittivity(CASRN='7732-18-5').T_dependent_property(298.15)
    assert_allclose(epsilon, 78.35530812232503)
    assert Permittivity(CASRN='7732-18-5').all_methods == set(['CRC', 'CRC_CONSTANT'])

    assert Permittivity(CASRN='132451235-2151234-1234123').all_methods == set()
    assert None == Permittivity(CASRN='132451235-2151234-1234123').T_dependent_property(300)

    tot_constant = sum([Permittivity(CASRN=i).calculate(T=298.15, method='CRC_CONSTANT') for i in CRC_Permittivity_data.index])
    assert_allclose(tot_constant, 13526.653700000023)

    sums_min, sums_avg, sums_max = 0, 0, 0
    for i in CRC_Permittivity_data.index:
        a = Permittivity(CASRN=i)
        if 'CRC' in a.all_methods:
            sums_min += a.calculate(a.CRC_Tmin, 'CRC')
            sums_avg += a.calculate((a.CRC_Tmax+a.CRC_Tmin)/2., 'CRC')
            sums_max += a.calculate(a.CRC_Tmax, 'CRC')
    assert_allclose([sums_min, sums_avg, sums_max], [10582.970609439253, 8312.897581451223, 6908.073524704013])


    assert (False, False) == (a.test_property_validity(2000), a.test_property_validity(-10))
    assert False == Permittivity(CASRN='7732-18-5').test_method_validity(228.15, 'CRC_CONSTANT')
    assert False == Permittivity(CASRN='7732-18-5').test_method_validity(228.15, 'CRC')

    with pytest.raises(Exception):
        a.test_method_validity(300, 'BADMETHOD' )


    # Tabular data
    w = Permittivity(CASRN='7732-18-5')
    Ts = np.linspace(273, 372, 10)
    permittivities = [87.75556413000001, 83.530500320000016, 79.48208925000003, 75.610330919999996, 71.915225330000013, 68.396772480000024, 65.05497237000003, 61.889825000000044, 58.901330369999997, 56.08948848]
    w.set_tabular_data(Ts=Ts, properties=permittivities)
    assert_allclose(w.T_dependent_property(305.), 75.95500925000006)
    w.tabular_extrapolation_permitted = True
    assert_allclose(w.T_dependent_property(200.), 115.79462395999997)
    w.tabular_extrapolation_permitted = False
    assert None == w.T_dependent_property(200.)


def test_permittivity_IAPWS():
    Ts = [238., 256., 273., 273., 323., 323., 373., 373., 510., 523., 614., 647., 673., 673., 773., 773., 873.]
    rhos = [975.06, 995.25, 999.83, 1180., 988.10,  1258., 958.46, 1110., 15.832, 900., 94.29, 358., 100., 900., 100., 900., 450.]
    permittivity_calc = [permittivity_IAPWS(T, rho) for T, rho in zip(Ts, rhos)]
    permittivity_exp = [106.31159697963018, 95.19633650530638, 87.96431108641572, 107.06291112337524, 69.96455480833566, 97.7606839686273, 55.56584297721836, 67.73206302035597, 1.1224589212024803, 32.23229227177932, 1.7702660877662086, 6.194373838662447, 1.7541419715602131, 23.59653593827129, 1.6554135047590008, 20.160449425540186, 6.283091796558804]
    assert_allclose(permittivity_calc, permittivity_exp)

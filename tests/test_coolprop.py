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
from thermo.coolprop import *
from thermo.identifiers import checkCAS

def test_fluid_props():
    tots = [sum([getattr(f, prop) for f in coolprop_fluids.values()]) for prop in ['Tmin', 'Tmax', 'Pmax', 'Tc', 'Pc', 'Tt', 'omega']]
    tots_exp =  [18589.301, 71575.0, 30817000000.0, 45189.59849999997, 440791794.7987591, 18589.301, 30.90243968446593]
              

    assert_allclose(tots_exp, tots)

    assert len(coolprop_fluids) == len(coolprop_dict)
    assert len(coolprop_dict) == 105
    assert all([checkCAS(i) for i in coolprop_dict])



def test_CoolProp_T_dependent_property():
    # Below the boiling point
    rhow = CoolProp_T_dependent_property(298.15, '7732-18-5', 'D', 'l')
    assert_allclose(rhow, 997.0476367603451)

    rhow = CoolProp_T_dependent_property(298.15, '7732-18-5', 'D', 'g')
    assert_allclose(rhow, 0.0230748041827597)

    # Above the boiling point
    rhow = CoolProp_T_dependent_property(450, '7732-18-5', 'D', 'l')
    assert_allclose(rhow, 890.3412497616716)

    rhow = CoolProp_T_dependent_property(450, '7732-18-5', 'D', 'g')
    assert_allclose(rhow, 0.49104706182775576)

    with pytest.raises(Exception):
        CoolProp_T_dependent_property(298.15, 'BADCAS', 'D', 'l')

    with pytest.raises(Exception):
        CoolProp_T_dependent_property(298.15, '7732-18-5', 'BADKEY', 'l')

    with pytest.raises(Exception):
        CoolProp_T_dependent_property(298.15, '7732-18-5', 'D', 'badphase')

    # Above the critical point
    with pytest.raises(Exception):
        CoolProp_T_dependent_property(700, '7732-18-5', 'D', 'l')

    rhow = CoolProp_T_dependent_property(700, '7732-18-5', 'D', 'g')
    assert_allclose(rhow, 0.3139926976198761)
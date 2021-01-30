# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import pytest
import thermo
from thermo import *
import chemicals
from chemicals import *
from fluids.numerics import *
from math import *
import json
import os
import numpy as np

@pytest.mark.fuzz
@pytest.mark.slow
def test_ChemicalConstantsPackage_from_json_as_json_large():
    create_compounds = []
    for k in dippr_compounds():
        try:
            if search_chemical(k) is not None:
                create_compounds.append(k)
        except:
            pass

    obj = ChemicalConstantsPackage.constants_from_IDs(create_compounds)
    obj2 = ChemicalConstantsPackage.from_json(obj.as_json())

    assert hash(obj) == hash(obj2)
    assert new_constants == constants
    assert id(new_constants) != id(constants)

def test_ChemicalConstantsPackage_json_version_exported():
    constants = ChemicalConstantsPackage(MWs=[18.01528, 106.165], names=['water', 'm-xylene'])
    string = constants.as_json()
    c2 = ChemicalConstantsPackage.from_json(string)
    assert 'json_version' in string
    assert not hasattr(c2, 'json_version')

@pytest.mark.CoolProp
def test_lemmon2000_package():
    for T in (150.0, 200.0, 300.0, 1000.0, 2000.0):
        assert_close(PropsSI('Cp0molar', 'T', T, 'P', 101325.0, 'Air'),
                     lemmon2000_correlations.HeatCapacityGases[0](T), rtol=2e-7)

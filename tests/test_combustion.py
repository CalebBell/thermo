# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import numpy as np
import pandas as pd
from thermo.combustion import *


def test_Hcombustion():
    dH = Hcombustion({'H': 4, 'C': 1, 'O': 1}, Hf=-239100)
    assert_allclose(dH, -726024.0) 
    
def test_combustion_products():
    C60_products = combustion_products({'C': 60})
    assert C60_products['CO2'] == 60
    assert C60_products['O2_required'] == 60
    
    methanol_products = combustion_products({'H': 4, 'C': 1, 'O': 1})
    assert methanol_products['H2O'] == 2
    assert methanol_products['CO2'] == 1
    assert methanol_products['SO2'] == 0
    assert methanol_products['O2_required'] == 1.5
    assert methanol_products['N2'] == 0
    assert methanol_products['HF'] == 0
    assert methanol_products['Br2'] == 0
    assert methanol_products['HCl'] == 0
    assert methanol_products['P4O10'] == 0

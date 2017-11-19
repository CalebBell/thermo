# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from __future__ import division
import types
import numpy as np
from numpy.testing import assert_allclose
import pytest
import fluids
import thermo
from thermo.units import *



def assert_pint_allclose(value, magnitude, units):
    assert_allclose(value.to_base_units().magnitude, magnitude)
    assert dict(value.dimensionality) == units


def test_custom_wraps():
    
    C = Stream(['ethane'], T=200*u.K, zs=[1], n=1*u.mol/u.s)
    D = Stream(['water', 'ethanol'], ns=[1, 2,]*u.mol/u.s, T=300*u.K, P=1E5*u.Pa)
    E = C + D
    
    assert_pint_allclose(E.zs, [ 0.5,   0.25,  0.25], {})
     
    assert_pint_allclose(E.T, 200, {'[temperature]': 1.0})



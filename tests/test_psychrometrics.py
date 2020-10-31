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
from fluids.numerics import *
from thermo import *


def test_x_w_to_humidity_ratio():
    # Checked
    RH = x_w_to_humidity_ratio(0.031417, MW_water=18.015268, MW_dry_air=28.96546)
    assert_close(0.020173821183401795, RH, rtol=1e-14)


def test_x_w_to_humidity_ratio_CoolProp():
    from CoolProp.CoolProp import HAPropsSI
    T, P = 298.15, 101325
    implemented = x_w_to_humidity_ratio(HAPropsSI('psi_w', 'T', T, 'P', P, 'RH', 1))
    val_CoolProp = HAPropsSI('W', 'T', T, 'P', P, 'RH', 1)
    assert_close(implemented, val_CoolProp, rtol=1e-4)
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

from numpy.testing import assert_allclose
import pytest
import thermo
from thermo import *
from thermo.coolprop import *
from thermo.phases import IAPWS95Gas, IAPWS95Liquid
from fluids.numerics import *
from math import *
import json
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass


def test_iapws95_basic_flash():
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])
    PT = flasher.flash(T=300, P=1e6)
    
    assert_close(PT.rho_mass(), 996.9600226949985, rtol=1e-10)
    assert isinstance(PT.liquid0, IAPWS95Liquid)
    
    TV = flasher.flash(T=300, V=PT.V())
    assert_close(TV.P, PT.P, rtol=1e-10)
    assert isinstance(TV.liquid0, IAPWS95Liquid)
    
    PV = flasher.flash(P=PT.P, V=PT.V())
    assert_close(PV.T, PT.T, rtol=1e-13)
    assert isinstance(PV.liquid0, IAPWS95Liquid)
    
    TVF = flasher.flash(T=400.0, VF=1)
    assert_close(TVF.P, 245769.3455657166, rtol=1e-13)
    
    PVF = flasher.flash(P=TVF.P, VF=1)
    assert_close(PVF.T, 400.0, rtol=1e-13)

def test_iapws95_basic_flashes_no_hacks():
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])
    flasher.VL_only_IAPWS95 = False
    
    PT = flasher.flash(T=300, P=1e6)
    
    assert_close(PT.rho_mass(), 996.9600226949985, rtol=1e-10)
    
    TV = flasher.flash(T=300, V=PT.V())
    assert_close(TV.P, PT.P, rtol=1e-10)
    
    PV = flasher.flash(P=PT.P, V=PT.V())
    assert_close(PV.T, PT.T, rtol=1e-13)
    
    PH = flasher.flash(H=PT.H(), P=1e6)
    assert_close(PH.T, 300)
    
    PS = flasher.flash(S=PT.S(), P=1e6)
    assert_close(PS.T, 300)
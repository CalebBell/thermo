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

import pytest
import numpy as np
import pandas as pd
from thermo.interface import *
from thermo.interface import VDI_TABULAR
from chemicals.identifiers import checkCAS
from fluids.numerics import assert_close, assert_close1d

@pytest.mark.meta_T_dept
def test_SurfaceTension():
    # Ethanol, test as many methods as possible at once
    EtOH = SurfaceTension(Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, StielPolar=-0.01266, CASRN='64-17-5')
    EtOH.T_dependent_property(305.)
    methods = EtOH.sorted_valid_methods
    methods.remove(VDI_TABULAR)
    sigma_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(305.))[1] for i in methods]
    sigma_exp = [0.021222422444285592, 0.02171156653650729, 0.02171156653650729, 0.021462066798796135, 0.02140008, 0.038055725907414066, 0.03739257387107131, 0.02645171690486362, 0.03905907338532845, 0.03670733205970745]

    assert_close1d(sorted(sigma_calcs), sorted(sigma_exp), rtol=1e-6)
    assert_close(EtOH.calculate(305., VDI_TABULAR), 0.021533867879206747, rtol=1E-4)

    # Test that methods return None
    sigma_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(5000))[1] for i in EtOH.sorted_valid_methods]
    assert [None]*11 == sigma_calcs

    EtOH.set_user_methods('VDI_TABULAR', forced=True)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.T_dependent_property(700.)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    # Test Aleem
    
    CH4 = SurfaceTension(Tb=111.65, Cpl=2465., Hvap_Tb=510870., MW=16.04246, Vml=3.497e-05)
    assert_close(CH4.T_dependent_property(90), 0.016704545538936296)
    
    assert not CH4.test_method_validity(600, 'Aleem')
    assert CH4.test_method_validity(100, 'Aleem')



def test_SurfaceTensionMixture():
    from thermo.mixture import Mixture
    from thermo.interface import SurfaceTensionMixture, DIGUILIOTEJA, SIMPLE, WINTERFELDSCRIVENDAVIS
    m = Mixture(['pentane', 'dichloromethane'], zs=[.1606, .8394], T=298.15)
    SurfaceTensions = [i.SurfaceTension for i in m.Chemicals]
    VolumeLiquids = [i.VolumeLiquid for i in m.Chemicals]
    
    a = SurfaceTensionMixture(MWs=m.MWs, Tbs=m.Tbs, Tcs=m.Tcs, CASs=m.CASs, SurfaceTensions=SurfaceTensions, VolumeLiquids=VolumeLiquids)

    sigma = a.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_close(sigma, 0.023887948426185343)
    
    sigma = a.calculate(m.T, m.P, m.zs, m.ws, SIMPLE)
    assert_close(sigma, 0.025331490604571537)
    
    sigmas = [a.calculate(m.T, m.P, m.zs, m.ws, i) for i in [DIGUILIOTEJA, SIMPLE, WINTERFELDSCRIVENDAVIS]]
    assert_close1d(sigmas, [0.025257338967448677, 0.025331490604571537, 0.023887948426185343])
    
    with pytest.raises(Exception):
        a.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
    with pytest.raises(Exception):
        a.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


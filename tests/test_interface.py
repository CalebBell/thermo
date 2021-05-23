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
import json
import pandas as pd
from thermo.interface import *
from chemicals.utils import property_mass_to_molar, zs_to_ws
from chemicals.dippr import EQ106
from thermo.interface import VDI_TABULAR
from chemicals.identifiers import check_CAS
from fluids.numerics import assert_close, assert_close1d, linspace
from thermo.volume import VolumeLiquid, VDI_PPDS
from thermo.utils import POLY_FIT
from thermo.interface import SurfaceTensionMixture, DIGUILIOTEJA, LINEAR, WINTERFELDSCRIVENDAVIS

@pytest.mark.meta_T_dept
def test_SurfaceTension():
    # Ethanol, test as many methods as possible at once
    EtOH = SurfaceTension(Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, StielPolar=-0.01266, CASRN='64-17-5')
    methods = list(EtOH.all_methods)
    methods_nontabular = list(methods)
    methods_nontabular.remove(VDI_TABULAR)
    sigma_calcs = []
    for i in methods_nontabular:
        EtOH.method = i
        sigma_calcs.append(EtOH.T_dependent_property(305.))

    sigma_exp = [0.021222422444285592, 0.02171156653650729, 0.02171156653650729, 0.021462066798796135, 0.02140008, 0.038055725907414066, 0.03739257387107131, 0.02645171690486362, 0.03905907338532845, 0.03670733205970745]

    assert_close1d(sorted(sigma_calcs), sorted(sigma_exp), rtol=1e-6)
    assert_close(EtOH.calculate(305., VDI_TABULAR), 0.021533867879206747, rtol=1E-4)

    # Test that methods return None
    for i in methods:
        EtOH.method = i
        assert EtOH.T_dependent_property(5000) is None


    assert SurfaceTension.from_json(EtOH.as_json()) == EtOH


    EtOH.method = 'VDI_TABULAR'
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.T_dependent_property(700.)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    # Test Aleem

    CH4 = SurfaceTension(Tb=111.65, Cpl=property_mass_to_molar(2465.,16.04246), Hvap_Tb=510870., MW=16.04246, Vml=3.497e-05)
    assert_close(CH4.T_dependent_property(90), 0.016704545538936296)

    assert not CH4.test_method_validity(600, 'Aleem')
    assert CH4.test_method_validity(100, 'Aleem')

@pytest.mark.meta_T_dept
def test_SurfaceTensionJasperMissingLimits():
    obj = SurfaceTension(CASRN='110-01-0')
    assert_close(obj.calculate(obj.JASPER_Tmax, 'JASPER'), 0, atol=1e-10)

    obj = SurfaceTension(CASRN='14901-07-6')
    assert_close(obj.calculate(obj.JASPER_Tmax, 'JASPER'), 0, atol=1e-10)

@pytest.mark.meta_T_dept
def test_SurfaceTensionVDITabularMissingZeroLimits():
    obj = SurfaceTension(CASRN='7782-41-4')
    assert_close(obj.calculate(144.41, 'VDI_TABULAR'), 0, atol=1e-10)

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_SurfaceTension_fitting0_yaws():
    A, Tc, n = 117.684, 7326.47, 1.2222
    Ts = linspace(378.4, 7326.47, 10)
    props_calc = [EQ106(T, Tc, A, n, 0.0, 0.0, 0.0, 0) for T in Ts]
    res, stats = SurfaceTension.fit_data_to_model(Ts=Ts, data=props_calc, model='YawsSigma',
                          do_statistics=True, use_numba=False, model_kwargs={'Tc': Tc},
                          fit_method='lm')        
    assert stats['MAE'] < 1e-5



def test_SurfaceTensionMixture():
    # ['pentane', 'dichloromethane']
    T = 298.15
    P = 101325.0
    zs = [.1606, .8394]

    MWs = [72.14878, 84.93258]
    ws = zs_to_ws(zs, MWs=MWs)

    VolumeLiquids = [VolumeLiquid(CASRN="109-66-0", MW=72.14878, Tb=309.21, Tc=469.7, Pc=3370000.0, Vc=0.000311, Zc=0.26837097540904814, omega=0.251, dipole=0.0, extrapolation="constant", method=POLY_FIT, poly_fit=(144, 459.7000000000001, [1.0839519373491257e-22, -2.420837244222272e-19, 2.318236501104612e-16, -1.241609625841306e-13, 4.0636406847721776e-11, -8.315431504053525e-09, 1.038485128954003e-06, -7.224842789857136e-05, 0.0022328080060137396])),
     VolumeLiquid(CASRN="75-09-2", MW=84.93258, Tb=312.95, Tc=508.0, Pc=6350000.0, Vc=0.000177, Zc=0.26610258553203137, omega=0.2027, dipole=1.6, extrapolation="constant", method=POLY_FIT, poly_fit=(178.01, 484.5, [1.5991056738532454e-23, -3.92303910541969e-20, 4.1522438881104836e-17, -2.473595776587317e-14, 9.064684097377694e-12, -2.0911320815626796e-09, 2.9653069375266426e-07, -2.3580713574913447e-05, 0.0008567355308938564]))]

    SurfaceTensions = [SurfaceTension(CASRN="109-66-0", MW=72.14878, Tb=309.21, Tc=469.7, Pc=3370000.0, Vc=0.000311, Zc=0.26837097540904814, omega=0.251, StielPolar=0.005164116344598568, Hvap_Tb=357736.5860890071, extrapolation=None, method=VDI_PPDS),
                       SurfaceTension(CASRN="75-09-2", MW=84.93258, Tb=312.95, Tc=508.0, Pc=6350000.0, Vc=0.000177, Zc=0.26610258553203137, omega=0.2027, StielPolar=-0.027514125341022044, Hvap_Tb=333985.7240672881, extrapolation=None, method=VDI_PPDS)]

    obj = SurfaceTensionMixture(MWs=MWs, Tbs=[309.21, 312.95], Tcs=[469.7, 508.0], correct_pressure_pure=False, CASs=['109-66-0', '75-09-2'], SurfaceTensions=SurfaceTensions, VolumeLiquids=VolumeLiquids)

    sigma_wsd = obj.calculate(T, P, zs, method=WINTERFELDSCRIVENDAVIS, ws=ws)
    assert_close(sigma_wsd, 0.02388914831076298)

    # Check the default calculation method is still WINTERFELDSCRIVENDAVIS
    sigma = obj.mixture_property(T, P, zs)
    assert_close(sigma, sigma_wsd)

    # Check the other implemented methods
    sigma = obj.calculate(T, P, zs, ws, LINEAR)
    assert_close(sigma, 0.025332871945242523)

    sigma = obj.calculate(T, P, zs, ws, DIGUILIOTEJA)
    assert_close(sigma, 0.025262398831653664)

    with pytest.raises(Exception):
        obj.test_method_validity(T, P, zs, ws, 'BADMETHOD')
    with pytest.raises(Exception):
        obj.calculate(T, P, zs, ws, 'BADMETHOD')

    hash0 = hash(obj)
    obj2 = SurfaceTensionMixture.from_json(json.loads(json.dumps(obj.as_json())))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

    obj2 = eval(str(obj))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0


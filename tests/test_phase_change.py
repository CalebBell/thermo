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

from fluids.numerics import assert_close, assert_close1d, linspace
from thermo.phase_change import *
from chemicals.miscdata import CRC_inorganic_data, CRC_organic_data
from chemicals.identifiers import check_CAS
from thermo.coolprop import has_CoolProp
import chemicals
from thermo.phase_change import COOLPROP, VDI_PPDS, CLAPEYRON, LIU, ALIBAKHSHI, MORGAN_KOBAYASHI, VELASCO, PITZER, RIEDEL, SIVARAMAN_MAGEE_KOBAYASHI, CHEN, CRC_HVAP_TB, DIPPR_PERRY_8E, VETERE, CRC_HVAP_298, VDI_TABULAR, GHARAGHEIZI_HVAP_298

@pytest.mark.meta_T_dept
def test_EnthalpyVaporization():
    EtOH = EnthalpyVaporization(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, similarity_variable=0.1954, Psat=7872.2, Zg=0.9633, Zl=0.0024, CASRN='64-17-5')

    EtOH.method = VDI_PPDS
    assert_close(EtOH.T_dependent_property(305), 42099.23631527565)
    EtOH.method = CLAPEYRON
    assert_close(EtOH.T_dependent_property(305), 39904.512005771176)
    EtOH.method = LIU
    assert_close(EtOH.T_dependent_property(305), 40315.087291316195)
    EtOH.method = ALIBAKHSHI
    assert_close(EtOH.T_dependent_property(305), 39244.0137575973)
    EtOH.method = MORGAN_KOBAYASHI
    assert_close(EtOH.T_dependent_property(305), 42182.87752489718)
    EtOH.method = VELASCO
    assert_close(EtOH.T_dependent_property(305), 43056.23753606326)
    EtOH.method = PITZER
    assert_close(EtOH.T_dependent_property(305), 41716.88048400951)
    EtOH.method = RIEDEL
    assert_close(EtOH.T_dependent_property(305), 44258.89496024996)
    EtOH.method = SIVARAMAN_MAGEE_KOBAYASHI
    assert_close(EtOH.T_dependent_property(305), 42279.09568184713)
    EtOH.method = CHEN
    assert_close(EtOH.T_dependent_property(305), 42951.50714053451)
    EtOH.method = CRC_HVAP_TB
    assert_close(EtOH.T_dependent_property(305), 42423.58947282491)
    EtOH.method = DIPPR_PERRY_8E
    assert_close(EtOH.T_dependent_property(305), 42115.102057622214)
    EtOH.method = VETERE
    assert_close(EtOH.T_dependent_property(305), 41382.22039928848)
    EtOH.method = CRC_HVAP_298
    assert_close(EtOH.T_dependent_property(305), 41804.5417918726)
    EtOH.method = VDI_TABULAR
    assert_close(EtOH.T_dependent_property(305), 42119.6665416816)
    EtOH.method = GHARAGHEIZI_HVAP_298
    assert_close(EtOH.T_dependent_property(305), 41686.00339359697)


    EtOH.extrapolation = None
    for i in EtOH.all_methods:
        EtOH.method = i
        assert EtOH.T_dependent_property(5000) is None


    assert EnthalpyVaporization.from_json(EtOH.as_json()) == EtOH

    EtOH = EnthalpyVaporization(CASRN='64-17-5', Tc=514.0)
    Hvap_calc = []
    for i in ['GHARAGHEIZI_HVAP_298', 'CRC_HVAP_298', 'VDI_TABULAR']:
        EtOH.method = i
        Hvap_calc.append(EtOH.T_dependent_property(310.0))
    Hvap_exp = [41304.19234346344, 41421.6450231131, 41857.962450207546]
    assert_close1d(Hvap_calc, Hvap_exp)

    # Test Clapeyron, without Zl
    EtOH = EnthalpyVaporization(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, similarity_variable=0.1954, Psat=7872.2, Zg=0.9633, CASRN='64-17-5')
    assert_close(EtOH.calculate(298.15, 'CLAPEYRON'), 37864.70507798813)

    EtOH = EnthalpyVaporization(Tb=351.39, Pc=6137000.0, omega=0.635, similarity_variable=0.1954, Psat=7872.2, Zg=0.9633, CASRN='64-17-5')
    assert EtOH.test_method_validity(351.39, 'CRC_HVAP_TB')
    assert not EtOH.test_method_validity(351.39+10, 'CRC_HVAP_TB')
    assert not EtOH.test_method_validity(351.39, 'CRC_HVAP_298')


    Ts = [200, 250, 300, 400, 450]
    props = [46461.62768429649, 44543.08561867195, 42320.381894706225, 34627.726535926406, 27634.46144486471]
    EtOH.add_tabular_data(Ts=Ts, properties=props, name='CPdata')
    EtOH.forced = True
    assert_close(43499.47575887933, EtOH.T_dependent_property(275), rtol=1E-4)

    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.T_dependent_property(5000)

    with pytest.raises(Exception):
        EtOH.test_method_validity('BADMETHOD', 300)

    assert EnthalpyVaporization.from_json(EtOH.as_json()) == EtOH

@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_EnthalpyVaporization_CoolProp():
    EtOH = EnthalpyVaporization(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, similarity_variable=0.1954, Psat=7872.2, Zg=0.9633, Zl=0.0024, CASRN='64-17-5')

    EtOH.method = COOLPROP
    assert_close(EtOH.T_dependent_property(305), 42062.9371631488)

    # Reduced property inputs
    EtOH = EnthalpyVaporization(CASRN='64-17-5', Tc=514.0)
    EtOH.method = COOLPROP
    assert_close(EtOH.T_dependent_property(310.0), 41796.56243049473)

    # Watson extrapolation
    obj = EnthalpyVaporization(CASRN='7732-18-5', Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344,
                         similarity_variable=0.16652530518537598, Psat=3167, Zl=1.0, Zg=0.96,
                         extrapolation='Watson')
    obj.method == COOLPROP
    assert 0 == obj(obj.Tc)
    assert_close(obj(1e-5), 54787.16649491286, rtol=1e-4)

    assert_close(obj.solve_property(5e4), 146.3404577534453)
    assert_close(obj.solve_property(1), 647.1399999389462)
    assert_close(obj.solve_property(1e-20), 647.13999999983)
    assert EnthalpyVaporization.from_json(obj.as_json()) == obj

@pytest.mark.meta_T_dept
def test_EnthalpyVaporization_Watson_extrapolation():
    obj = EnthalpyVaporization(CASRN='7732-18-5', Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344,
                         similarity_variable=0.16652530518537598, Psat=3167, Zl=1.0, Zg=0.96,
                         extrapolation='Watson')

    # Data from CoolProp
    Ts = [300, 400, 500, 600]
    Hvaps = [43908.418874478055, 39322.84456586401, 32914.75657594899, 21122.090961998296]
    obj.add_tabular_data(Ts=Ts, properties=Hvaps, name='test0')
    obj.method = 'test0'
    assert 0 == obj(obj.Tc)
    assert_close(obj.solve_property(1), 647.1399999998296)

    obj.solve_property(5e4)
    obj.solve_property(1e-20)
    assert EnthalpyVaporization.from_json(obj.as_json()) == obj
    assert eval(str(obj)) == obj


@pytest.mark.meta_T_dept
def test_EnthalpySublimation_no_numpy():
    assert type(EnthalpySublimation(CASRN='1327-53-3').CRC_Hfus) is float

@pytest.mark.meta_T_dept
@pytest.mark.fitting
def test_EnthalpyVaporization_fitting0():
    ammonia_Ts_Hvaps = [195.41, 206.344, 217.277, 228.211, 239.145, 239.82, 250.078, 261.012, 271.946, 282.879, 293.813, 304.747, 315.681, 326.614, 337.548, 348.482, 359.415, 370.349, 381.283, 392.216, 403.15]
    ammonia_Hvaps = [25286.1, 24832.3, 24359.1, 23866.9, 23354.6, 23322.3, 22820.1, 22259.6, 21667.7, 21037.4, 20359.5, 19622.7, 18812.8, 17912.4, 16899.8, 15747, 14416.3, 12852.3, 10958.8, 8510.62, 4311.94]
    obj = EnthalpyVaporization(CASRN='7664-41-7', load_data=False)
    
    with pytest.raises(ValueError):
        # Tc needs to be specified
        obj.fit_data_to_model(Ts=ammonia_Ts_Hvaps, data=ammonia_Hvaps, model='DIPPR106',
                              do_statistics=True, use_numba=False, fit_method='lm')
        
    fit, res = obj.fit_data_to_model(Ts=ammonia_Ts_Hvaps, data=ammonia_Hvaps, model='DIPPR106',
                      do_statistics=True, use_numba=False, model_kwargs={'Tc': 405.400},
                      fit_method='lm'
                     )
    assert res['MAE'] < 1e-5
    
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_EnthalpyVaporization_fitting1_dippr_106():
    check_CAS_fits = ['74-86-2', '108-05-4', '7440-37-1', '7440-59-7',
                      '1333-74-0', '7664-39-3']
    for CAS in check_CAS_fits:
        obj = EnthalpyVaporization(CASRN=CAS)
        Ts = linspace(obj.Perrys2_150_Tmin, obj.Perrys2_150_Tmax, 8)
        props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR106',
                              do_statistics=True, use_numba=False, fit_method='lm', 
                                           model_kwargs={'Tc': obj.Perrys2_150_coeffs[0]})
        assert stats['MAE'] < 1e-7

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_EnthalpyVaporization_fitting2_dippr_106():

    for i, CAS in enumerate(chemicals.phase_change.phase_change_data_VDI_PPDS_4.index):
        obj = EnthalpyVaporization(CASRN=CAS)
        Ts = linspace(obj.T_limits[VDI_PPDS][0], obj.T_limits[VDI_PPDS][1], 8)
        props_calc = [obj.calculate(T, VDI_PPDS) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='PPDS12',
                              do_statistics=True, use_numba=False, fit_method='lm', 
                                           model_kwargs={'Tc': obj.VDI_PPDS_Tc})
        assert stats['MAE'] < 1e-7

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_EnthalpyVaporization_fitting3_dippr_106_full():
    for CAS in chemicals.phase_change.phase_change_data_Perrys2_150.index:
        obj = EnthalpyVaporization(CASRN=CAS)
        Ts = linspace(obj.Perrys2_150_Tmin, obj.Perrys2_150_Tmax, 8)
        props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR106',
                              do_statistics=True, use_numba=False, fit_method='lm', 
                                           model_kwargs={'Tc': obj.Perrys2_150_coeffs[0]})
        assert stats['MAE'] < 1e-7

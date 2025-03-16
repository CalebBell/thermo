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
SOFTWARE.
'''

import chemicals
import pytest
from fluids.numerics import assert_close, assert_close1d, linspace

from thermo.coolprop import has_CoolProp
from thermo.phase_change import *
from thermo.phase_change import (
    ALIBAKHSHI,
    CHEN,
    CLAPEYRON,
    COOLPROP,
    CRC_HVAP_298,
    CRC_HVAP_TB,
    DIPPR_PERRY_8E,
    GHARAGHEIZI_HVAP_298,
    LIU,
    MORGAN_KOBAYASHI,
    PITZER,
    RIEDEL,
    SIVARAMAN_MAGEE_KOBAYASHI,
    VDI_PPDS,
    VDI_TABULAR,
    VELASCO,
    VETERE,
)


@pytest.mark.meta_T_dept
def test_EnthalpyVaporization():
    EtOH = EnthalpyVaporization(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, similarity_variable=0.1954,
                                Psat=7872.2, Zg=0.9633, Zl=0.0024, CASRN='64-17-5')

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
    EtOH = EnthalpyVaporization(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, similarity_variable=0.1954,
                                Psat=7872.2, Zg=0.9633, CASRN='64-17-5')
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
    assert None is EtOH.T_dependent_property(5000)

    with pytest.raises(Exception):
        EtOH.test_method_validity('BADMETHOD', 300)

    assert EnthalpyVaporization.from_json(EtOH.as_json()) == EtOH


    # Case was raising a complex
    obj = EnthalpyVaporization(CASRN="7782-41-4", Tb=85.04, Tc=144.3, Pc=5215197.75, omega=0.0588, similarity_variable=0.05263600258783854, extrapolation="Watson")
    assert 0 == obj.calculate(144.41400000000002, 'VDI_PPDS')



@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_EnthalpyVaporization_CoolProp():
    EtOH = EnthalpyVaporization(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, similarity_variable=0.1954,
                                Psat=7872.2, Zg=0.9633, Zl=0.0024, CASRN='64-17-5')

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
    obj.method = COOLPROP
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
def test_EnthalpyVaporization_cheb_fit_ln_tau():
    coeffs = [18231.740838720892, -18598.514785409734, 5237.841944302821, -1010.5549489362293, 147.88312821848922, -17.412144225239444, 1.7141064359038864, -0.14493639179363527, 0.01073811633477817, -0.0007078634084791702, 4.202655964036239e-05, -2.274648068123497e-06, 1.1239490049774759e-07]
    obj2 = EnthalpyVaporization(Tc=591.75, cheb_fit_ln_tau=((178.18, 591.0, 591.75, coeffs)))
    T = 500
    assert_close(obj2(T), 24498.131947622023, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(T), -100.77476795241955, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(T, order=2), -0.6838185834436981, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(T, order=3), -0.012093191904152178, rtol=1e-13)


@pytest.mark.meta_T_dept
def test_EnthalpyVaporization_stablepoly_fit_ln_tau():
    coeffs = [-0.00854738149791956, 0.05600738152861595, -0.30758192972280085, 1.6848304651211947, -8.432931053161155, 37.83695791102946, -150.87603890354512, 526.4891248463246, -1574.7593541151946, 3925.149223414621, -7826.869059381197, 11705.265444382389, -11670.331914006258, 5817.751307862842]
    obj2 = EnthalpyVaporization(Tc=591.75, stablepoly_fit_ln_tau=((178.18, 591.74, 591.75, coeffs)))

    T = 500
    assert_close(obj2(T), 24498.131947494512, rtol=1e-13)

    assert_close(obj2.T_dependent_property_derivative(T), -100.77476796035525, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(T, order=2), -0.6838185833621794, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(T, order=3), -0.012093191888904656, rtol=1e-13)



@pytest.mark.meta_T_dept
def test_EnthalpyVaporization_polynomial_ln_tau_parameters():
    coeffs = [9.661381155485653, 224.16316385569456, 2195.419519751738, 11801.26111760343, 37883.05110910901, 74020.46380982929, 87244.40329893673, 69254.45831263301, 61780.155823216155]
    Tc = 591.75

    obj_polynomial = EnthalpyVaporization(Tc=Tc, polynomial_ln_tau_parameters={'test': {'coeffs': coeffs,
                                                        'Tmin': 178.01, 'Tmax': 586.749, 'Tc': Tc}})

    assert EnthalpyVaporization.from_json(obj_polynomial.as_json()) == obj_polynomial

    vals = obj_polynomial(500),
    for v in vals:
        assert_close(v, 24168.867169087476, rtol=1e-13)

@pytest.mark.meta_T_dept
def test_EnthalpyVaporization_poly_fit_ln_tau():
    coeffs = [9.661381155485653, 224.16316385569456, 2195.419519751738, 11801.26111760343, 37883.05110910901, 74020.46380982929, 87244.40329893673, 69254.45831263301, 61780.155823216155]
    Tc = 591.75
    T = 300.0
    obj2 = EnthalpyVaporization(Tc=Tc, poly_fit_ln_tau=(178.01, 586.749, Tc, coeffs))
    assert_close(obj2(T), 37900.38881665646, rtol=1e-13)

    assert_close(obj2.T_dependent_property_derivative(T), -54.63227984184944, rtol=1e-14)
    assert_close(obj2.T_dependent_property_derivative(T, order=2), 0.037847046150971016, rtol=1e-14)
    assert_close(obj2.T_dependent_property_derivative(T, order=3), -0.001920502581912092, rtol=1e-13)

    assert EnthalpyVaporization.from_json(obj2.as_json()) == obj2
    assert eval(str(obj2)) == obj2

@pytest.mark.meta_T_dept
def test_EnthalpySublimation_no_numpy():
    assert type(EnthalpySublimation(CASRN='1327-53-3').CRC_Hfus) is float

def test_EnthalpySublimation_webbook():
    # Magnesium, bis(η5-2,4-cyclopentadien-1-yl)- Hsub present in the One dimensional data sections
    obj = EnthalpySublimation(CASRN='1284-72-6')
    obj.method = 'WEBBOOK_HSUB'
    assert_close(obj(300), 68200.0, rtol=1e-10)

def test_EnthalpySublimation_GHARAGHEIZI():
    obj = EnthalpySublimation(CASRN='51-20-7')
    obj.method = 'GHARAGHEIZI_HSUB_298'
    assert_close(obj(300), 151400, rtol=1e-10)

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
        Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 8)
        props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR106',
                              do_statistics=True, use_numba=False, fit_method='lm',
                                           model_kwargs={'Tc': obj.DIPPR106_parameters['DIPPR_PERRY_8E']['Tc']})
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
                                           model_kwargs={'Tc': obj.PPDS12_parameters['VDI_PPDS']['Tc']})
        assert stats['MAE'] < 1e-7

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_EnthalpyVaporization_fitting3_dippr_106_full():
    for CAS in chemicals.phase_change.phase_change_data_Perrys2_150.index:
        obj = EnthalpyVaporization(CASRN=CAS)
        Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 8)
        props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR106',
                              do_statistics=True, use_numba=False, fit_method='lm',
                                           model_kwargs={'Tc': obj.DIPPR106_parameters['DIPPR_PERRY_8E']['Tc']})
        assert stats['MAE'] < 1e-7

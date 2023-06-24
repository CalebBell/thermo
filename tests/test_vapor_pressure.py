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

import json
from math import *
from math import isnan

import chemicals
import numpy as np
import pytest
from chemicals.vapor_pressure import *
from fluids.numerics import assert_close, assert_close1d, derivative, linspace

from thermo.coolprop import has_CoolProp
from thermo.utils import TDependentProperty
from thermo.vapor_pressure import *
from thermo.vapor_pressure import (
    AMBROSE_WALTON,
    ANTOINE_EXTENDED_POLING,
    ANTOINE_POLING,
    ANTOINE_WEBBOOK,
    BOILING_CRITICAL,
    COOLPROP,
    DIPPR_PERRY_8E,
    EDALAT,
    LEE_KESLER_PSAT,
    SANJARI,
    VDI_PPDS,
    VDI_TABULAR,
    WAGNER_MCGARRY,
    WAGNER_POLING,
)


@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_VaporPressure_CoolProp():
    EtOH = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
    assert_close(EtOH.calculate(305.0, COOLPROP), 11592.205263402893, rtol=1e-7)

@pytest.mark.meta_T_dept
def test_VaporPressure_ethanol():
    # Ethanol, test as many methods asa possible at once
    EtOH = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')

    # Check that assigning a bad method does not change the object and raises a ValueERror
    h0 = hash(EtOH)
    with pytest.raises(ValueError):
        EtOH.method = 'NOTAMETHOD'
    assert hash(EtOH) == h0

    # Check valid_methods call
    assert set(EtOH.valid_methods()) == EtOH.all_methods
    # For Ethanol, all methods are valid around 300 K
    assert EtOH.valid_methods(365) == EtOH.valid_methods()

    # Check test_property_validity
    assert not EtOH.test_property_validity(1j)

    # Check can't calculate with a bad method
    with pytest.raises(ValueError):
        EtOH.calculate(300, 'NOTAMETHOD')

    methods = list(EtOH.all_methods)
    methods.remove(VDI_TABULAR)
    if COOLPROP in methods:
        methods.remove(COOLPROP)

    EtOH.extrapolation = 'nolimit'

    Psats_expected = {WAGNER_MCGARRY: 11579.634014300127,
                     WAGNER_POLING: 11590.408779316374,
                     ANTOINE_POLING: 11593.661615921257,
                     DIPPR_PERRY_8E: 11659.154222044575,
                     VDI_PPDS: 11698.02742876088,
                     BOILING_CRITICAL: 14088.453409816764,
                     LEE_KESLER_PSAT: 11350.156640503357,
                     AMBROSE_WALTON: 11612.378633936816,
                     SANJARI: 9210.262000640221,
                     EDALAT: 12081.738947110121,
                     ANTOINE_WEBBOOK: 10827.523813517917,
                     }

    T = 305.0
    Psat_calcs = {}
    for i in methods:
        EtOH.method = i
        Psat_calcs[i] = EtOH.T_dependent_property(T)
        Tmin, Tmax = EtOH.T_limits[i]
        if i not in (ANTOINE_WEBBOOK,):
            assert Tmin < T < Tmax

    for k, v in Psats_expected.items():
        assert_close(v, Psat_calcs[k], rtol=1e-11)
    assert len(Psats_expected) == len(Psats_expected)

    assert_close(EtOH.calculate(305, VDI_TABULAR), 11690.81660829924, rtol=1E-4)

    s = EtOH.as_json()
    assert 'json_version' in s
    obj2 = VaporPressure.from_json(s)
    assert EtOH == obj2

    # Test that methods return None
    EtOH = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
    EtOH.extrapolation = None
    for i in list(EtOH.all_methods):
        EtOH.method = i
        assert EtOH.T_dependent_property(5000) is None



@pytest.mark.meta_T_dept
def test_VaporPressure_extended_poling():
    # Use another chemical to get in ANTOINE_EXTENDED_POLING
    a = VaporPressure(CASRN='589-81-1')

    Psat_calcs = []
    for i in list(a.all_methods):
        a.method = i
        Psat_calcs.append(a.T_dependent_property(410))


    Psat_exp = [162944.82134710113, 162870.44794192078, 170508.47471278594, 162865.5380455795]
    assert_close1d(sorted(Psat_calcs), sorted(Psat_exp))

    s = a.as_json()
    obj2 = VaporPressure.from_json(s)
    assert a == obj2

@pytest.mark.meta_T_dept
def test_VaporPressure_water():

    # Test interpolation, extrapolation
    w = VaporPressure(Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, CASRN='7732-18-5')
    Ts = linspace(300, 350, 10)
    Ps = [3533.918074415897, 4865.419832056078, 6612.2351036034115, 8876.854141719203, 11780.097759775277, 15462.98385942125, 20088.570250257424, 25843.747665059742, 32940.95821687677, 41619.81654904555]
    w.add_tabular_data(Ts=Ts, properties=Ps)
    assert_close(w.T_dependent_property(305.), 4715.122890601165)
    w.extrapolation = 'interp1d'
    assert_close(w.T_dependent_property(200.), 0.5364148240126076)

    Ts_bad = [300, 325, 350]
    Ps_bad = [1, -1, 1j]
    with pytest.raises(ValueError):
        w.add_tabular_data(Ts=Ts_bad, properties=Ps_bad)
    Ts_rev = list(reversed(Ts))
    with pytest.raises(ValueError):
        w.add_tabular_data(Ts=Ts_rev, properties=Ps)

@pytest.mark.meta_T_dept
def test_VaporPressure_cycloheptane():

    # Get a check for Antoine Extended
    cycloheptane = VaporPressure(Tb=391.95, Tc=604.2, Pc=3820000.0, omega=0.2384, CASRN='291-64-5')
    cycloheptane.method = ('ANTOINE_EXTENDED_POLING')
    cycloheptane.extrapolation = None
    assert_close(cycloheptane.T_dependent_property(410), 161647.35219882353)
    assert None is cycloheptane.T_dependent_property(400)

    with pytest.raises(Exception):
        cycloheptane.test_method_validity(300, 'BADMETHOD')

    obj = VaporPressure(CASRN="71-43-2", Tb=353.23, Tc=562.05, Pc=4895000.0, omega=0.212, extrapolation="AntoineAB|DIPPR101_ABC", method="WAGNER_MCGARRY")
    assert_close(obj.T_dependent_property_derivative(600.0), 89196.777813, rtol=1e-4)

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting0():
    obj = VaporPressure(CASRN='13838-16-9')
    Tmin, Tmax = obj.WAGNER_POLING_Tmin, obj.WAGNER_POLING_Tmax
    Ts = linspace(Tmin, Tmax, 10)
    Ps = [obj(T) for T in Ts]
    Tc, Pc = obj.WAGNER_POLING_Tc, obj.WAGNER_POLING_Pc
    fitted = obj.fit_data_to_model(Ts=Ts, data=Ps, model='Wagner', use_numba=False,
                          model_kwargs={'Tc': obj.WAGNER_POLING_Tc, 'Pc': obj.WAGNER_POLING_Pc})
    res = fitted
    assert 'Tc' in res
    assert 'Pc' in res
    assert_close(res['a'], obj.WAGNER_POLING_coefs[0])
    assert_close(res['b'], obj.WAGNER_POLING_coefs[1])
    assert_close(res['c'], obj.WAGNER_POLING_coefs[2])
    assert_close(res['d'], obj.WAGNER_POLING_coefs[3])

    # Heavy compound fit
    Ts = linspace(179.15, 237.15, 5)
    props_calc = [Antoine(T, A=138., B=520200.0, C=3670.0) for T in Ts]
    res, stats = TDependentProperty.fit_data_to_model(Ts=Ts, data=props_calc, model='Antoine',
                          do_statistics=True, use_numba=False, model_kwargs={'base':10.0},
                          fit_method='lm')
    assert stats['MAE'] < 1e-5

    # Fit with very low range and no C
    Ts = linspace(374, 377.0, 5)
    props_calc = [Antoine(T, A=12.852103, B=2942.98, C=0.0) for T in Ts]
    res, stats = TDependentProperty.fit_data_to_model(Ts=Ts, data=props_calc, model='Antoine',
                          do_statistics=True, use_numba=False, model_kwargs={'base':10.0},
                          fit_method='lm')
    assert stats['MAE'] < 1e-5


@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting1():
    # Ammonia data fitting from chemsep
    ammonia_Ts_Psats = [191.24, 202.546, 213.852, 225.157, 236.463, 239.82, 247.769, 259.075, 270.381, 281.686, 292.992, 304.298, 315.604, 326.909, 338.215, 349.521, 360.827, 372.133, 383.438, 394.744, 406.05]
    ammonia_Psats = [4376.24, 10607.70, 23135.70, 46190.70, 85593.30, 101505.00, 148872.00, 245284.00, 385761.00, 582794.00, 850310.00, 1203550.00, 1658980.00, 2234280.00, 2948340.00, 3821410.00, 4875270.00, 6133440.00, 7621610.00, 9367940.00, 11403600.00]
    res, stats = VaporPressure.fit_data_to_model(Ts=ammonia_Ts_Psats, data=ammonia_Psats, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-4

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting2_dippr():
    pts = 10
    fit_issue_CASs = ['75-07-0', '107-02-8', '108-38-3', '7732-18-5', '85-44-9',
                      '67-64-1', '78-87-5', '624-72-6', '118-96-7', '124-18-5',
                      # '526-73-8'
                      ]
    for CAS in fit_issue_CASs:
        obj = VaporPressure(CASRN=CAS)
        Ts = linspace(obj.Perrys2_8_Tmin, obj.Perrys2_8_Tmax, pts)
        props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                              do_statistics=True, use_numba=False, multiple_tries=True, fit_method='lm')
        assert stats['MAE'] < 1e-5

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting3_WagnerMcGarry():
    for i, CAS in enumerate(chemicals.vapor_pressure.Psat_data_WagnerMcGarry.index):
        obj = VaporPressure(CASRN=CAS)
        Ts = linspace(obj.WAGNER_MCGARRY_Tmin, obj.WAGNER_MCGARRY_Tc, 10)
        props_calc = [obj.calculate(T, WAGNER_MCGARRY) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='Wagner_original',
                              do_statistics=True, use_numba=False,
                              fit_method='lm', model_kwargs={'Tc': obj.WAGNER_MCGARRY_Tc, 'Pc': obj.WAGNER_MCGARRY_Pc})
        assert stats['MAE'] < 1e-7

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting4_WagnerPoling():
    for i, CAS in enumerate(chemicals.vapor_pressure.Psat_data_WagnerPoling.index):
        obj = VaporPressure(CASRN=CAS)
        Ts = linspace(obj.WAGNER_POLING_Tmin, obj.WAGNER_POLING_Tc, 10)
        props_calc = [obj.calculate(T, WAGNER_POLING) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='Wagner',
                              do_statistics=True, use_numba=False,
                              fit_method='lm', model_kwargs={'Tc': obj.WAGNER_POLING_Tc, 'Pc': obj.WAGNER_POLING_Pc})
        assert stats['MAE'] < 1e-7

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting5_AntoinePoling():
    for i, CAS in enumerate(chemicals.vapor_pressure.Psat_data_AntoinePoling.index):
        obj = VaporPressure(CASRN=CAS)
        Ts = linspace(obj.ANTOINE_POLING_Tmin, obj.ANTOINE_POLING_Tmax, 10)
        props_calc = [obj.calculate(T, ANTOINE_POLING) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='Antoine',
                              do_statistics=True, use_numba=False,
                              multiple_tries=True,
                              model_kwargs={'base': 10.0},  fit_method='lm')
        assert stats['MAE'] < 1e-7

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting6_VDI_PPDS():
    for i, CAS in enumerate(chemicals.vapor_pressure.Psat_data_VDI_PPDS_3.index):
        obj = VaporPressure(CASRN=CAS)
        Ts = linspace(obj.VDI_PPDS_Tm, obj.VDI_PPDS_Tc, 10)
        props_calc = [obj.calculate(T, VDI_PPDS) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='Wagner',
                              do_statistics=True, use_numba=False,
                              fit_method='lm', model_kwargs={'Tc': obj.VDI_PPDS_Tc, 'Pc': obj.VDI_PPDS_Pc})
        assert stats['MAE'] < 1e-7

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting7_reduced_fit_params_with_jac():
    obj = VaporPressure(CASRN='13838-16-9')
    Tmin, Tmax = obj.WAGNER_POLING_Tmin, obj.WAGNER_POLING_Tmax
    Ts = linspace(Tmin, Tmax, 10)
    Ps = [obj(T) for T in Ts]
    fit = obj.fit_data_to_model(Ts=Ts, data=Ps, model='Wagner', use_numba=False,
                                model_kwargs={'Tc': obj.WAGNER_POLING_Tc, 'Pc': obj.WAGNER_POLING_Pc, 'd': -4.60})
    assert fit['d'] == -4.6

    fit = obj.fit_data_to_model(Ts=Ts, data=Ps, model='Wagner', use_numba=False,
                                model_kwargs={'Tc': obj.WAGNER_POLING_Tc, 'Pc': obj.WAGNER_POLING_Pc, 'b': 2.4})
    assert fit['b'] == 2.4

    fit = obj.fit_data_to_model(Ts=Ts, data=Ps, model='Wagner', use_numba=False,
                                model_kwargs={'Tc': obj.WAGNER_POLING_Tc, 'Pc': obj.WAGNER_POLING_Pc, 'd': -4.60, 'a': -8.329, 'b': 2.4})
    assert fit['a'] == -8.329
    assert fit['b'] == 2.4
    assert fit['d'] == -4.6

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting8_TRC_AntoineExtended():
    hard_CASs = frozenset(['110-82-7'])
    for i, CAS in enumerate(chemicals.vapor_pressure.Psat_data_AntoineExtended.index):
        obj = VaporPressure(CASRN=CAS)
        Ts = linspace(obj.ANTOINE_EXTENDED_POLING_Tmin, obj.ANTOINE_EXTENDED_POLING_Tmax, 10)
        props_calc = [obj.calculate(T, ANTOINE_EXTENDED_POLING) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='TRC_Antoine_extended',
                              do_statistics=True, use_numba=False, multiple_tries=CAS in hard_CASs,# multiple_tries_max_err=1e-4,
                              fit_method='lm', model_kwargs={'Tc': obj.ANTOINE_EXTENDED_POLING_coefs[0],
                                                             'to': obj.ANTOINE_EXTENDED_POLING_coefs[1]})
        assert stats['MAE'] < 1e-4

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting9_Yaws_Psat():
    A, B, C, D, E, Tmin, Tmax = 53.93890302013294, -788.24, -22.734, 0.051225, 6.1896e-11, 68.15, 132.92
    Ts = linspace(Tmin, Tmax, 10)
    props_calc = [Yaws_Psat(T, A, B, C, D, E) for T in Ts]
    res, stats = VaporPressure.fit_data_to_model(Ts=Ts, data=props_calc, model='Yaws_Psat',
                          do_statistics=True, use_numba=False,
                          fit_method='lm')
    assert stats['MAE'] < 1e-5


@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_mandatory_arguments_for_all():
    kwargs = {'Ts': [293.12, 303.13, 313.16, 323.08, 333.13, 343.1, 353.14, 363.17, 373.14, 383.21, 393.11, 403.1, 413.14, 423.14, 433.06, 443.01, 454.15], 'data': [102100.0, 145300.0, 202100.0, 274700.0, 368700.0, 487100.0, 636400.0, 822200.0, 1048500.0, 1327100.0, 1659300.0, 2060200.0, 2541300.0, 3111500.0, 3778100.0, 4566000.0, 5611600.0], 'model': 'Wagner', 'model_kwargs': {'Tc': 461.0, 'Pc': 6484800.0}, 'params_points_max': 2, 'model_selection': 'min(BIC, AICc)', 'do_statistics': True, 'use_numba': False, 'multiple_tries': False, 'multiple_tries_max_err': 1e-05, 'fit_method': 'lm'}
    VaporPressure.fit_data_to_model(**kwargs)

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting_K_fold():
    kwargs = {'Ts': [223.149993896484, 223.149993896484, 273.149993896484], 
            'data': [0.1637, 0.1622, 0.1453], 'model': 'DIPPR100', 'model_kwargs': {}, 
            'params_points_max': 1, 'model_selection': 'KFold(3)', 'do_statistics': True, 
            'use_numba': False, 'multiple_tries': False, 'multiple_tries_max_err': 1e-05, 'fit_method': 'lm'}

    res, stats = TDependentProperty.fit_data_to_model(**kwargs)
    assert res['C'] == 0
    assert_close(res['A'], 0.24172194784598627, rtol=1e-3)
    assert_close(res['B'], -0.00035300000000077007, rtol=1e-3)

    # Case wheer only A was being returned
    kwargs = {'Ts': [278.149993896484, 283.149993896484, 298.149993896484, 323.149993896484, 343.149993896484], 
          'data': [0.113, 0.11187, 0.1085, 0.10287, 0.098375], 'model': 'DIPPR100', 'model_kwargs': {}, 
          'params_points_max': 2, 'model_selection': 'KFold(5)', 'do_statistics': True, 'use_numba': False, 
          'multiple_tries': False, 'multiple_tries_max_err': 1e-05, 'fit_method': 'lm'}
    res, stats = TDependentProperty.fit_data_to_model(**kwargs)
    assert res['C'] == 0
    assert_close(res['A'], 0.17557973443592295, rtol=1e-3)
    assert_close(res['B'],-0.0002249933993404257, rtol=1e-3)

    # Another case where only A was returned
    kwargs = {'Ts': [288.15, 293.15, 303.15, 293.15, 298.15], 
          'data': [0.03286, 0.03232, 0.03127, 0.032100000000000004, 0.0314], 
          'model': 'DIPPR100', 'model_kwargs': {}, 'params_points_max': 2, 
          'model_selection': 'min(BIC, AICc, KFold(5))', 'do_statistics': True, 'use_numba': False, 
          'multiple_tries': False, 'multiple_tries_max_err': 1e-05, 'fit_method': 'lm'}
    res, stats = TDependentProperty.fit_data_to_model(**kwargs)
    assert res['C'] == 0
    assert_close(res['A'], 0.06491057692331224, rtol=1e-3)
    assert_close(res['B'], -0.0001115384615387056, rtol=1e-3)

    # Case with one point, check it fits to one
    kwargs = {'Ts': [623.150024414063], 'data': [0.35564], 'model': 'DIPPR100',
          'model_kwargs': {}, 'params_points_max': 2, 'model_selection': 'min(BIC, AICc, KFold(5))',
          'do_statistics': True, 'use_numba': False, 'multiple_tries': False,
          'multiple_tries_max_err': 1e-05, 'fit_method': 'lm'}

    res, stats = TDependentProperty.fit_data_to_model(**kwargs)
    assert_close(res['A'], 0.35564, rtol=1e-5)
    assert res['B'] == 0
    assert len(res) == 7

    # Another case a single point was returned instead of a linear fit
    kwargs = {'Ts': [281.15, 287.15, 293.15, 292.95, 319.04999999999995], 
            'data': [0.017215, 0.016550000000000002, 0.015880000000000002, 0.014199999999999999, 0.01137], 
            'model': 'DIPPR100', 'model_kwargs': {}, 'params_points_max': 2, 
            'model_selection': 'min(BIC, AICc, KFold(5))', 'do_statistics': True, 
            'use_numba': False, 'multiple_tries': False, 'multiple_tries_max_err': 1e-05, 'fit_method': 'lm'}
    res, stats = TDependentProperty.fit_data_to_model(**kwargs)
    res

    res, stats = TDependentProperty.fit_data_to_model(**kwargs)
    assert res['C'] == 0
    assert_close(res['A'], 0.06072849925273897, rtol=1e-3)
    assert_close(res['B'], -0.00015502901100344948, rtol=1e-3)

    # aic and bic went to three parameters but only 2 are justified
    # Implemented rounding to make this one work
    kwargs = {'Ts': [196.34999999999997, 208.45, 215.95, 223.14999999999998, 227.64999999999998],
            'data': [0.02639, 0.02384, 0.0228, 0.021570000000000002, 0.02077], 
            'model': 'DIPPR100', 'model_kwargs': {}, 'params_points_max': 2, 
            'model_selection': 'min(BIC, AICc, KFold(5))', 'do_statistics': True, 
            'use_numba': False, 'multiple_tries': False, 'multiple_tries_max_err': 1e-05, 'fit_method': 'lm'}

    res, stats = TDependentProperty.fit_data_to_model(**kwargs)
    assert res['C'] == 0
    assert_close(res['A'], 0.06084909088165748, rtol=1e-3)
    assert_close(res['B'], -0.00017626378088590118, rtol=1e-3)

    # aic numerical problem was causing one parameter
    kwargs = {'Ts': [158.84999999999997, 168.64999999999998, 178.84999999999997],
            'data': [0.029500000000000002, 0.027600000000000003, 0.0257],
            'model': 'DIPPR100', 'model_kwargs': {}, 'params_points_max': 1, 
            'model_selection': 'min(BIC, AICc, KFold(5))', 'do_statistics': True,
            'use_numba': False, 'multiple_tries': False, 'multiple_tries_max_err': 1e-05, 'fit_method': 'lm'}

    res, stats = TDependentProperty.fit_data_to_model(**kwargs)
    assert res['C'] == 0
    assert_close(res['A'], 0.059664558032730346, rtol=1e-3)
    assert_close(res['B'], -0.0001899746698887944, rtol=1e-3)

    # aic not working was causing the wrong choice
    kwargs = {'Ts': [300.0, 313.149993896484, 293.149993896484, 303.149993896484],
    'data': [0.155, 0.14644, 0.15774, 0.15983], 'model': 'DIPPR100', 'model_kwargs': {},
    'params_points_max': 2, 'model_selection': 'min(BIC, AICc, KFold(5))', 'do_statistics': True,
    'use_numba': False, 'multiple_tries': False, 'multiple_tries_max_err': 1e-05, 'fit_method': 'lm'}


    res, stats = TDependentProperty.fit_data_to_model(**kwargs)
    assert res['C'] == 0
    assert_close(res['A'], 0.3205950796110783, rtol=1e-3)
    assert_close(res['B'], -0.000548489254194966, rtol=1e-3)








@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting10_example():
    Ts = [203.65, 209.55, 212.45, 234.05, 237.04, 243.25, 249.35, 253.34, 257.25, 262.12, 264.5, 267.05, 268.95, 269.74, 272.95, 273.46, 275.97, 276.61, 277.23, 282.03, 283.06, 288.94, 291.49, 293.15, 293.15, 293.85, 294.25, 294.45, 294.6, 294.63, 294.85, 297.05, 297.45, 298.15, 298.15, 298.15, 298.15, 298.15, 299.86, 300.75, 301.35, 303.15, 303.15, 304.35, 304.85, 305.45, 306.25, 308.15, 308.15, 308.15, 308.22, 308.35, 308.45, 308.85, 309.05, 311.65, 311.85, 311.85, 311.95, 312.25, 314.68, 314.85, 317.75, 317.85, 318.05, 318.15, 318.66, 320.35, 320.35, 320.45, 320.65, 322.55, 322.65, 322.85, 322.95, 322.95, 323.35, 323.55, 324.65, 324.75, 324.85, 324.85, 325.15, 327.05, 327.15, 327.2, 327.25, 327.35, 328.22, 328.75, 328.85, 333.73, 338.95]
    Psats = [58.93, 94.4, 118.52, 797.1, 996.5, 1581.2, 2365, 3480, 3893, 5182, 6041, 6853, 7442, 7935, 9290, 9639, 10983, 11283, 13014, 14775, 15559, 20364, 22883, 24478, 24598, 25131, 25665, 25931, 25998, 26079, 26264, 29064, 29598, 30397, 30544, 30611, 30784, 30851, 32636, 33931, 34864, 37637, 37824, 39330, 40130, 41063, 42396, 45996, 46090, 46356, 45462, 46263, 46396, 47129, 47396, 52996, 52929, 53262, 53062, 53796, 58169, 59328, 66395, 66461, 67461, 67661, 67424, 72927, 73127, 73061, 73927, 79127, 79527, 80393, 79927, 80127, 81993, 80175, 85393, 85660, 85993, 86260, 86660, 92726, 92992, 92992, 93126, 93326, 94366, 98325, 98592, 113737, 136626]
    res, stats = TDependentProperty.fit_data_to_model(Ts=Ts, data=Psats, model='Antoine', do_statistics=True, multiple_tries=True, model_kwargs={'base': 10.0})
    assert stats['MAE'] < 0.014

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VaporPressure_fitting11_bad_method():
    Ts = [203.65, 209.55, 212.45, 234.05, 237.04, 243.25, 249.35, 253.34, 257.25, 262.12, 264.5, 267.05, 268.95, 269.74, 272.95, 273.46, 275.97, 276.61, 277.23, 282.03, 283.06, 288.94, 291.49, 293.15, 293.15, 293.85, 294.25, 294.45, 294.6, 294.63, 294.85, 297.05, 297.45, 298.15, 298.15, 298.15, 298.15, 298.15, 299.86, 300.75, 301.35, 303.15, 303.15, 304.35, 304.85, 305.45, 306.25, 308.15, 308.15, 308.15, 308.22, 308.35, 308.45, 308.85, 309.05, 311.65, 311.85, 311.85, 311.95, 312.25, 314.68, 314.85, 317.75, 317.85, 318.05, 318.15, 318.66, 320.35, 320.35, 320.45, 320.65, 322.55, 322.65, 322.85, 322.95, 322.95, 323.35, 323.55, 324.65, 324.75, 324.85, 324.85, 325.15, 327.05, 327.15, 327.2, 327.25, 327.35, 328.22, 328.75, 328.85, 333.73, 338.95]
    Psats = [58.93, 94.4, 118.52, 797.1, 996.5, 1581.2, 2365, 3480, 3893, 5182, 6041, 6853, 7442, 7935, 9290, 9639, 10983, 11283, 13014, 14775, 15559, 20364, 22883, 24478, 24598, 25131, 25665, 25931, 25998, 26079, 26264, 29064, 29598, 30397, 30544, 30611, 30784, 30851, 32636, 33931, 34864, 37637, 37824, 39330, 40130, 41063, 42396, 45996, 46090, 46356, 45462, 46263, 46396, 47129, 47396, 52996, 52929, 53262, 53062, 53796, 58169, 59328, 66395, 66461, 67461, 67661, 67424, 72927, 73127, 73061, 73927, 79127, 79527, 80393, 79927, 80127, 81993, 80175, 85393, 85660, 85993, 86260, 86660, 92726, 92992, 92992, 93126, 93326, 94366, 98325, 98592, 113737, 136626]
    with pytest.raises(ValueError):
        TDependentProperty.fit_data_to_model(Ts=Ts, data=Psats, model='NOTAMETHOD')

@pytest.mark.meta_T_dept
def test_VaporPressure_analytical_derivatives():
    Psat = VaporPressure(CASRN="108-38-3", Tb=412.25, Tc=617.0, Pc=3541000.0, omega=0.331,
                         extrapolation="AntoineAB|DIPPR101_ABC", method=WAGNER_MCGARRY)
    assert_close(Psat.calculate_derivative(T=400.0, method=WAGNER_MCGARRY), 2075.9195652247963, rtol=1e-13)
    assert_close(Psat.calculate_derivative(T=400.0, method=WAGNER_MCGARRY, order=2), 47.61112509616565, rtol=1e-13)

    assert_close(Psat.calculate_derivative(T=400.0, method=WAGNER_POLING), 2073.565462948561, rtol=1e-13)
    assert_close(Psat.calculate_derivative(T=400.0, method=WAGNER_POLING, order=2), 47.60007499952595, rtol=1e-13)

    assert_close(Psat.calculate_derivative(T=400.0, method=DIPPR_PERRY_8E), 2075.1783812355125, rtol=1e-13)
    assert_close(Psat.calculate_derivative(T=400.0, method=DIPPR_PERRY_8E, order=2), 47.566696599306596, rtol=1e-13)

    assert_close(Psat.calculate_derivative(T=400.0, method=VDI_PPDS), 2073.5972901257196, rtol=1e-13)
    assert_close(Psat.calculate_derivative(T=400.0, method=VDI_PPDS, order=2), 47.489535848986364, rtol=1e-13)

    cycloheptane = VaporPressure(Tb=391.95, Tc=604.2, Pc=3820000.0, omega=0.2384, CASRN='291-64-5')
    cycloheptane.method = ANTOINE_EXTENDED_POLING
    assert_close(cycloheptane.calculate_derivative(T=500.0, method=ANTOINE_EXTENDED_POLING, order=2), 176.89903538855853, rtol=1e-13)
    assert_close(cycloheptane.calculate_derivative(T=500.0, method=ANTOINE_EXTENDED_POLING, order=1), 15046.47337900798, rtol=1e-13)

    cycloheptane.method = ANTOINE_POLING
    assert_close(cycloheptane.calculate_derivative(T=400.0, method=ANTOINE_POLING, order=1), 3265.237029987264, rtol=1e-13)
    assert_close(cycloheptane.calculate_derivative(T=400.0, method=ANTOINE_POLING, order=2), 65.83298769903531, rtol=1e-13)



def test_VaporPressure_no_isnan():
    assert not isnan(VaporPressure(CASRN='4390-04-9').Tmin)

def test_VaporPressure_linear_extrapolation_non_negative():
    ethanol_psat = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')

    # Make sure the constants are set to guard against future changes to defaults
    ethanol_psat.method = WAGNER_MCGARRY
    ethanol_psat.interpolation_T = (lambda T: 1/T)
    ethanol_psat.interpolation_property = (lambda P: log(P))
    ethanol_psat.interpolation_property_inv = (lambda P: exp(P))
    ethanol_psat.extrapolation = 'linear'

    assert_close(ethanol_psat(700), 59005875.32878946, rtol=1e-4)
    assert_close(ethanol_psat(100), 1.0475828451230242e-11, rtol=1e-4)

    assert ethanol_psat.T_limits['WAGNER_MCGARRY'][0] == ethanol_psat.WAGNER_MCGARRY_Tmin
    assert ethanol_psat.T_limits['WAGNER_MCGARRY'][1] == ethanol_psat.WAGNER_MCGARRY_Tc

    assert_close(ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tc),
                 ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tc-1e-6))
    assert_close(ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tc),
                 ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tc+1e-6))

    assert_close(ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tmin),
                 ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tmin-1e-6))
    assert_close(ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tmin),
                 ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tmin+1e-6))



    Tmin = ethanol_psat.T_limits[ethanol_psat.method][0]
    Ts = linspace(0.7*Tmin, Tmin*(1-1e-10), 10)
    Ps = [ethanol_psat(T) for T in Ts]

    # Confirms it's linear
    # plt.plot(1/np.array(Ts), np.log(Ps))
    # plt.show()
#    def rsquared(x, y):
#        import scipy.stats
#        _, _, r_value, _, _ = scipy.stats.linregress(x, y)
#        return r_value*r_value

#    assert_close(rsquared(1/np.array(Ts), np.log(Ps)), 1, atol=1e-5)
    assert abs(np.polyfit(1/np.array(Ts), np.log(Ps), 1, full=True)[1][0]) < 1e-13


    # TODO make work with different interpolation methods
#    assert ethanol_psat == VaporPressure.from_json(ethanol_psat.as_json())

@pytest.mark.meta_T_dept
def test_VaporPressure_extrapolation_solve_prop():
    cycloheptane = VaporPressure(Tb=391.95, Tc=604.2, Pc=3820000.0, omega=0.2384, CASRN='291-64-5')
    cycloheptane.method = 'ANTOINE_EXTENDED_POLING'
    cycloheptane.extrapolation = 'AntoineAB|DIPPR101_ABC'
    cycloheptane.T_dependent_property(T=4000)

    assert_close(cycloheptane.solve_property(1), 187.25621087267422)

    assert_close(cycloheptane.solve_property(1e-20), 60.677576120119156)

    assert_close(cycloheptane.solve_property(1e5), 391.3576035137979)
    assert_close(cycloheptane.solve_property(1e6), 503.31772463155266)
    assert_close(cycloheptane.solve_property(1e7), 711.8047771523733)
    assert_close(cycloheptane.solve_property(3e7), 979.2026813626704)

def test_VaporPressure_bestfit_derivatives():
    obj = VaporPressure(exp_poly_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10,
                                                -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708]))
    assert_close(obj.T_dependent_property(300), 18601.061401014867, rtol=1e-11)
    assert_close(obj.T_dependent_property_derivative(300), 954.1652489206775, rtol=1e-11)
    assert_close(obj.T_dependent_property_derivative(300, order=2), 41.8787546283273, rtol=1e-11)
    assert_close(derivative(obj.T_dependent_property, 300, dx=300*1e-7), obj.T_dependent_property_derivative(300))
    assert_close(derivative(obj.T_dependent_property_derivative, 300, dx=300*1e-7), obj.T_dependent_property_derivative(300, order=2))


@pytest.mark.meta_T_dept
def test_VaporPressure_extrapolation_AB():
    obj = VaporPressure(Tb=309.21, Tc=469.7, Pc=3370000.0, omega=0.251, CASRN='109-66-0', load_data=True, extrapolation='AntoineAB')
    obj.method = WAGNER_MCGARRY
    obj.calculate_derivative(300, WAGNER_MCGARRY)

    for extrapolation in ('AntoineAB', 'DIPPR101_ABC', 'AntoineAB|AntoineAB', 'DIPPR101_ABC|DIPPR101_ABC',
                          'DIPPR101_ABC|AntoineAB', 'AntoineAB|DIPPR101_ABC'):
        obj.extrapolation = extrapolation

        assert_close(obj.T_dependent_property(obj.WAGNER_MCGARRY_Tc),
                     obj.T_dependent_property(obj.WAGNER_MCGARRY_Tc-1e-6))
        assert_close(obj.T_dependent_property(obj.WAGNER_MCGARRY_Tc),
                     obj.T_dependent_property(obj.WAGNER_MCGARRY_Tc+1e-6))

        assert_close(obj.T_dependent_property(obj.WAGNER_MCGARRY_Tmin),
                     obj.T_dependent_property(obj.WAGNER_MCGARRY_Tmin-1e-6))
        assert_close(obj.T_dependent_property(obj.WAGNER_MCGARRY_Tmin),
                     obj.T_dependent_property(obj.WAGNER_MCGARRY_Tmin+1e-6))


# @pytest.mark.meta_T_dept
# def test_VaporPressure_fast_Psat_poly_fit():
#     corr = VaporPressure(exp_poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))

#     # Low transition
#     P_trans = corr.exp_poly_fit_Tmin_value
#     assert_close(corr.solve_property(P_trans), corr.solve_property_exp_poly_fit(P_trans), rtol=1e-10)
#     assert_close(corr.solve_property(P_trans + 1e-7), corr.solve_property_exp_poly_fit(P_trans + 1e-7), rtol=1e-10)

#     # High transition
#     P_trans = corr.exp_poly_fit_Tmax_value
#     assert_close(corr.solve_property(P_trans), corr.solve_property_exp_poly_fit(P_trans), rtol=1e-10)
#     assert_close(corr.solve_property(P_trans + 1e-7), corr.solve_property_exp_poly_fit(P_trans + 1e-7), rtol=1e-10)


#     # Low temperature values - up to 612 Pa
#     assert_close(corr.solve_property(1e-5), corr.solve_property_exp_poly_fit(1e-5), rtol=1e-10)
#     assert_close(corr.solve_property(1), corr.solve_property_exp_poly_fit(1), rtol=1e-10)
#     assert_close(corr.solve_property(100), corr.solve_property_exp_poly_fit(100), rtol=1e-10)


#     # Solver region
#     assert_close(corr.solve_property(1e5), corr.solve_property_exp_poly_fit(1e5), rtol=1e-10)
#     assert_close(corr.solve_property(1e7), corr.solve_property_exp_poly_fit(1e7), rtol=1e-10)

#     # High T
#     assert_close(corr.solve_property(1e8), corr.solve_property_exp_poly_fit(1e8), rtol=1e-10)

#     # Extrapolation
#     from thermo.vapor_pressure import POLY_FIT, BEST_FIT_AB, BEST_FIT_ABC
#     obj = VaporPressure(poly_fit=(178.01, 591.74, [-8.638045111752356e-20, 2.995512203611858e-16, -4.5148088801006036e-13, 3.8761537879200513e-10, -2.0856828984716705e-07, 7.279010846673517e-05, -0.01641020023565049, 2.2758331029405516, -146.04484159879843]))
#     assert_close(obj.calculate(1000, BEST_FIT_AB), 78666155.90418352, rtol=1e-10)
#     assert_close(obj.calculate(1000, BEST_FIT_ABC), 156467764.5930495, rtol=1e-10)

#     assert_close(obj.calculate(400, POLY_FIT), 157199.6909849476, rtol=1e-10)
#     assert_close(obj.calculate(400, BEST_FIT_AB), 157199.6909849476, rtol=1e-10)
#     assert_close(obj.calculate(400, BEST_FIT_ABC), 157199.6909849476, rtol=1e-10)

@pytest.mark.meta_T_dept
def test_VaporPressure_generic_polynomial_exp_parameters():
    coeffs = [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10,
                                                -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708]

    obj_bestfit = VaporPressure(exp_poly_fit=(175.7, 512.49, coeffs))
    obj_polynomial = VaporPressure(exp_polynomial_parameters={'test': {'coeffs': coeffs,
                                                        'Tmin': 175.7, 'Tmax': 512.49}})
    assert_close(obj_bestfit.T_dependent_property(300), 18601.061401014867, rtol=1e-11)

    assert_close(obj_polynomial(300), obj_bestfit.T_dependent_property(300), rtol=1e-13)

    assert VaporPressure.from_json(obj_bestfit.as_json()) == obj_bestfit
    assert eval(str(obj_bestfit)) == obj_bestfit

    assert VaporPressure.from_json(obj_polynomial.as_json()) == obj_polynomial
    assert eval(str(obj_polynomial)) == obj_polynomial

    T = 300.0

    assert_close(obj_polynomial.T_dependent_property_derivative(T), 954.1652489206775, rtol=1e-14)
    assert_close(obj_polynomial.T_dependent_property_derivative(T, order=2), 41.8787546283273, rtol=1e-14)
    assert_close(obj_polynomial.T_dependent_property_derivative(T, order=3), 1.496803960985584, rtol=1e-13)

@pytest.mark.meta_T_dept
def test_VaporPressure_generic_polynomial_exp_parameters_complicated():
    coeffs = [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10,
    -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708]
    T = 300.0

    obj2 = VaporPressure(exp_poly_fit=(175.7, 512.49, coeffs))
    assert_close(obj2(T), 18601.061401014867, rtol=1e-13)

    # All derivatives/integrals are numerical with the generic form
    assert_close(obj2.T_dependent_property_derivative(T), 954.1652489206775, rtol=1e-14)
    assert_close(obj2.T_dependent_property_derivative(T, order=2), 41.8787546283273, rtol=1e-14)
    assert_close(obj2.T_dependent_property_derivative(T, order=3), 1.496803960985584, rtol=1e-13)


@pytest.mark.meta_T_dept
def test_VaporPressure_exp_stablepoly_fit():
    obj2 = VaporPressure(Tc=591.72, exp_stablepoly_fit=((309.0, 591.72, [0.008603558174828078, 0.007358688688856427, -0.016890323025782954, -0.005289197721114957, -0.0028824712174469625, 0.05130960832946553, -0.12709896610233662, 0.37774977659528036, -0.9595325030688526, 2.7931528759840174, 13.10149649770156])))
    assert_close(obj2(400), 157191.01706242564, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(400, order=1), 4056.436943642117, rtol=1e-13)

    assert_close(obj2.T_dependent_property_derivative(400, order=2), 81.32645570045084, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(400, order=3), 1.103603650822488, rtol=1e-13)

@pytest.mark.meta_T_dept
def test_VaporPressure_exp_cheb_fit():
    obj2 = VaporPressure(Tc=591.72, exp_cheb_fit=((309.0, 591.72, [12.570668791524573, 3.1092695610681673, -0.5485217707981505, 0.11115875762247596, -0.01809803938553478, 0.003674911307077089, -0.00037626163070525465, 0.0001962813915017403, 6.120764548889213e-05, 3.602752453735203e-05])))
    assert_close(obj2(300), 4186.189338463003, rtol=1e-13)

    assert_close(obj2.T_dependent_property_derivative(400, order=1), 4056.277312107932, rtol=1e-13)

    assert_close(obj2.T_dependent_property_derivative(400, order=2), 81.34302144188977, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(400, order=3),1.105438780935656, rtol=1e-13)

@pytest.mark.meta_T_dept
def test_VaporPressure_extrapolation_no_validation():
    N2 = VaporPressure(CASRN='7727-37-9', extrapolation='DIPPR101_ABC')
    N2.method = WAGNER_MCGARRY
    assert N2(298.15) is not None
    assert N2(1000.15) is not None


@pytest.mark.meta_T_dept
def test_VaporPressure_fast_Psat_poly_fit_extrapolation():
    obj = VaporPressure(exp_poly_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10,
                                                -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708]))
    obj.extrapolation = 'AntoineAB|DIPPR101_ABC'
    assert_close(obj.solve_property(1e-13), 88.65839225764933)
    assert_close(obj.solve_property(300), 237.7793675652309)
    assert_close(1e8, obj.extrapolate(obj.solve_property(1e8), 'EXP_POLY_FIT'))
    assert_close(obj.extrapolate(800, 'EXP_POLY_FIT'), 404793143.0358333)

@pytest.mark.meta_T_dept
def test_VaporPressure_Antoine_inputs():
    obj = VaporPressure()
    obj.add_correlation(name='WebBook', model='Antoine', Tmin=177.70, Tmax=264.93,  A=3.45604+5, B=1044.038, C=-53.893)
    assert_close(obj(200), 20.432980367117192, rtol=1e-12)

    # json
    hash0 = hash(obj)

    obj2 = VaporPressure.from_json(json.loads(json.dumps(obj.as_json())))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

    obj2 = eval(str(obj))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0


    obj2 = VaporPressure(Antoine_parameters={'WebBook': {'A': 8.45604, 'B': 1044.038, 'C': -53.893, 'Tmin': 177.7, 'Tmax': 264.93}})
    assert_close(obj2(200), 20.432980367117192, rtol=1e-12)
    assert obj == obj2

    with pytest.raises(ValueError):
        obj.add_correlation(name='WebBook2', model='Antoine', Tmin=177.70, Tmax=264.93,  A=3.45604+5, B=1044.038)
    with pytest.raises(ValueError):
        obj.add_correlation(name='WebBook', model='Antoine', Tmin=177.70, Tmax=264.93,  A=3.45604+5, B=1044.038, C=-53.893)
    with pytest.raises(ValueError):
        obj.add_correlation(name='WebBook4', model='NOTAMODEL', Tmin=177.70, Tmax=264.93,  A=3.45604+5, B=1044.038, C=-53.893)


    # Test with the new 'coefficients' input method
    obj = VaporPressure(Antoine_parameters={'WebBook': {'coefficients': [8.45604, 1044.038, -53.893],
                                                    'Tmin': 177.7, 'Tmax': 264.93}})
    assert_close(obj(220), 148.15143004993493, rtol=1e-13)

@pytest.mark.meta_T_dept
def test_VaporPressure_DIPPR101_inputs():
    obj = VaporPressure()
    # From Perry's 8th edition
    obj.add_correlation(name='Eq101test', model='DIPPR101', Tmin=175.47, Tmax=512.5,  A=82.718, B=-6904.5, C=-8.8622, D=7.4664E-6, E=2.0)
    assert_close(obj(298.15), 16825.750567754883, rtol=1e-13)
    assert_close(obj.T_dependent_property_derivative(298.15, order=1), 881.6678722199089, rtol=1e-12)
    assert_close(obj.T_dependent_property_derivative(298.15, order=2), 39.36139219676838, rtol=1e-12)
    assert_close(obj.T_dependent_property_derivative(298.15, order=3), 1.4228777458080808, rtol=1e-12)

    # json
    hash0 = hash(obj)
    obj2 = VaporPressure.from_json(json.loads(json.dumps(obj.as_json())))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

    obj2 = eval(str(obj))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0


@pytest.mark.meta_T_dept
def test_VaporPressure_extrapolate_derivatives():
    obj = VaporPressure(extrapolation='DIPPR101_ABC|AntoineAB', exp_poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))
    assert_close(obj.T_dependent_property_derivative(230), 1.4571835115958078, rtol=1e-11)
    assert_close(obj.T_dependent_property_derivative(230, order=2), 0.14109259717547848, rtol=1e-11)
    assert_close(obj.T_dependent_property_derivative(230, order=3), 0.012236116569167774, rtol=1e-11)

    assert_close(obj.T_dependent_property_derivative(700, order=1), 403088.9468522063, rtol=1e-8)
    assert_close(obj.T_dependent_property_derivative(700, order=2), 2957.4520772886904, rtol=1e-8)

@pytest.mark.meta_T_dept
def test_VaporPressure_weird_signatures():
    from thermo.utils import PROPERTY_TRANSFORM_D2_X, PROPERTY_TRANSFORM_D2LN, PROPERTY_TRANSFORM_D_X, PROPERTY_TRANSFORM_DLN, PROPERTY_TRANSFORM_LN

    obj = VaporPressure(extrapolation='DIPPR101_ABC|AntoineAB', exp_poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))

    # Within range
    assert_close(obj.T_dependent_property_transform(300, PROPERTY_TRANSFORM_LN),
                 log(obj.T_dependent_property(300)))

    assert_close(obj.T_dependent_property_transform(300, PROPERTY_TRANSFORM_D_X),
                 obj.T_dependent_property_derivative(300)/obj.T_dependent_property(300))

    assert_close(obj.T_dependent_property_transform(300, PROPERTY_TRANSFORM_D2_X),
                 obj.T_dependent_property_derivative(300, 2)/obj.T_dependent_property(300))

    dln = derivative(lambda T: log(obj(T)), 300, dx=300*1e-6)
    assert_close(dln, obj.T_dependent_property_transform(300, PROPERTY_TRANSFORM_DLN))

    dln = derivative(lambda T: log(obj(T)), 300, n=2, dx=300*1e-5)
    assert_close(dln, obj.T_dependent_property_transform(300, PROPERTY_TRANSFORM_D2LN), rtol=1e-5)

    # Extrapolations
    for extrapolation in ('interp1d', 'DIPPR101_ABC', 'AntoineAB'):
        obj.extrapolation = extrapolation
        for T in (100, 1000):
            assert_close(obj.T_dependent_property_transform(T, PROPERTY_TRANSFORM_LN),
                         log(obj.T_dependent_property(T)))

            assert_close(obj.T_dependent_property_transform(T, PROPERTY_TRANSFORM_D_X),
                         obj.T_dependent_property_derivative(T)/obj.T_dependent_property(T))

            assert_close(obj.T_dependent_property_transform(T, PROPERTY_TRANSFORM_D2_X),
                         obj.T_dependent_property_derivative(T, 2)/obj.T_dependent_property(T))

            dln = derivative(lambda T: log(obj(T)), T, dx=T*1e-6)
            assert_close(dln, obj.T_dependent_property_transform(T, PROPERTY_TRANSFORM_DLN))

            dln = derivative(lambda T: log(obj(T)), T, n=2, dx=T*1e-5)
            assert_close(dln, obj.T_dependent_property_transform(T, PROPERTY_TRANSFORM_D2LN), rtol=4e-4)

@pytest.mark.meta_T_dept
def test_VaporPressure_WebBook():
    obj = VaporPressure(CASRN='7440-57-5')
    obj.method = 'ANTOINE_WEBBOOK'
    assert_close(obj(3000), 36784.98996094166, rtol=1e-13)

@pytest.mark.meta_T_dept
def test_VaporPressure_IAPWS95():

    water = VaporPressure(CASRN="7732-18-5", Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344,
                          extrapolation="AntoineAB|DIPPR101_ABC", method="IAPWS")
    assert water(3000) is not None
    assert water(2000) > water(600)

    assert water(20) > water(10)
    assert water(10) > water(4)
    assert_close(water(400), 245769.3455657166, rtol=1e-10)


@pytest.mark.meta_T_dept
def test_accurate_vapor_pressure_mercury():
    obj = VaporPressure(CASRN='7439-97-6')
    assert_close(obj(333.15), 3.508170, rtol=1e-7)

    new = VaporPressure.from_json(json.loads(json.dumps(obj.as_json())))
    assert new == obj


@pytest.mark.meta_T_dept
def test_accurate_vapor_pressure_beryllium():
    obj = VaporPressure(CASRN='7440-41-7')
    assert_close(obj(2742.30), 1e5, atol=.01)


@pytest.mark.meta_T_dept
def test_vapor_pressure_element_metals():
    K = VaporPressure(CASRN='7439-93-2')
    assert_close(K.calculate(1000, 'ALCOCK_ELEMENTS'), 104.2837434179671)

@pytest.mark.meta_T_dept
def test_accurate_vapor_pressure_H2O2():
    obj = VaporPressure(CASRN="7722-84-1")
    assert_close(obj(160+273.15), 124484.67628951524, rtol=0.005)

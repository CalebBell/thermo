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

import numpy as np
import pytest
from chemicals.utils import ws_to_zs
from fluids.numerics import assert_close, assert_close1d
from scipy.integrate import quad

from thermo.coolprop import has_CoolProp
from thermo.heat_capacity import *
from thermo.heat_capacity import (
    COOLPROP,
    CRCSTD,
    DADGOSTAR_SHAW,
    HEOS_FIT,
    JOBACK,
    LASTOVKA_SHAW,
    POLING_CONST,
    POLING_POLY,
    ROWLINSON_BONDI,
    ROWLINSON_POLING,
    TRCIG,
    VDI_TABULAR,
    WEBBOOK_SHOMATE,
    ZABRANSKY_QUASIPOLYNOMIAL_C,
    ZABRANSKY_SPLINE_C,
    ZABRANSKY_SPLINE_SAT,
)


@pytest.mark.meta_T_dept
def test_HeatCapacityGas():
    EtOH = HeatCapacityGas(CASRN='64-17-5', similarity_variable=0.1953615, MW=46.06844)

    assert eval(str(EtOH)) == EtOH
    json_repr = EtOH.as_json()
    assert 'kwargs' not in json_repr

    new = HeatCapacityGas.from_json(json_repr)
    assert new == EtOH

    methods = list(EtOH.all_methods)
    methods.remove(VDI_TABULAR)
    if COOLPROP in methods:
        methods.remove(COOLPROP)


    Cps_expected = {TRCIG: 66.35087249951643,
                     POLING_POLY: 66.40066070415111,
                     POLING_CONST: 65.21,
                     CRCSTD: 65.6,
                     LASTOVKA_SHAW: 71.07236200126606,
                     JOBACK: 65.74656180000001,
                     HEOS_FIT: 66.25906466813824}

    T = 305.0
    Cps_calc = {}
    for i in methods:
        EtOH.method = i
        Cps_calc[i] = EtOH.T_dependent_property(305)
        Tmin, Tmax = EtOH.T_limits[i]
        assert Tmin < T < Tmax

    for k, v in Cps_calc.items():
        assert_close(v, Cps_calc[k], rtol=1e-11)
    assert len(Cps_expected) == len(Cps_calc)

    # VDI interpolation, treat separately due to change in behavior of scipy in 0.19
    assert_close(EtOH.calculate(305, VDI_TABULAR), 74.6763493522965, rtol=1E-4)



    EtOH.extrapolation = None
    for i in [TRCIG, POLING_POLY, CRCSTD, POLING_CONST, VDI_TABULAR]:
        EtOH.method = i
        assert EtOH.T_dependent_property(5000) is None

    with pytest.raises(Exception):
        EtOH.test_method_validity('BADMETHOD', 300)

    assert False is EtOH.test_property_validity(-1)
    assert False is EtOH.test_property_validity(10100.0)


    Ts = [200, 250, 300, 400, 450]
    props = [1.2, 1.3, 1.4, 1.5, 1.6]
    EtOH.add_tabular_data(Ts=Ts, properties=props, name='test_set')
    assert_close(1.35441088517, EtOH.T_dependent_property(275), rtol=2E-4)

    assert None is EtOH.T_dependent_property(5000)

    new = HeatCapacityGas.from_json(EtOH.as_json())
    assert new == EtOH

    # Case where the limits were nans
    obj = HeatCapacityGas(CASRN='7440-37-1', MW=39.948, similarity_variable=0.025032542304996495)
    assert not isnan(obj.Tmax)
    assert not isnan(obj.Tmin)
    assert not isnan(obj.T_limits[POLING_POLY][0])
    assert not isnan(obj.T_limits[POLING_POLY][1])

    new = HeatCapacityGas.from_json(obj.as_json())
    assert new == obj

@pytest.mark.meta_T_dept
def test_HeatCapacityGas_webbook():
    obj = HeatCapacityGas(CASRN='7732-18-5')
    assert_close(obj.calculate(700, WEBBOOK_SHOMATE), 37.5018469222449)

    obj.method = WEBBOOK_SHOMATE
    assert_close(obj.calculate(700, WEBBOOK_SHOMATE), 37.5018469222449)
    assert_close(obj.calculate(3000, WEBBOOK_SHOMATE), 55.74187422222222)
    assert_close(obj.calculate(6000, WEBBOOK_SHOMATE), 60.588267555555554)

    assert_close(obj.calculate_integral(1, 700, WEBBOOK_SHOMATE), 105354.51298924106)
    assert_close(obj.calculate_integral(1, 3000, WEBBOOK_SHOMATE), 217713.25484195515)
    assert_close(obj.calculate_integral(1, 6000, WEBBOOK_SHOMATE), 393461.64992528845)

    assert_close(obj.calculate_integral_over_T(1, 700, WEBBOOK_SHOMATE), 41272.70183405447)
    assert_close(obj.calculate_integral_over_T(1, 3000, WEBBOOK_SHOMATE), 41340.46968129815)
    assert_close(obj.calculate_integral_over_T(1, 6000, WEBBOOK_SHOMATE), 41380.892814134764)

    assert eval(str(obj)) == obj
    new = HeatCapacityGas.from_json(json.loads(json.dumps(obj.as_json())))
    assert new == obj

    # Hydrogen peroxide
    obj = HeatCapacityGas(CASRN='7722-84-1')
    obj.method = WEBBOOK_SHOMATE
    assert_close(obj.calculate(700, WEBBOOK_SHOMATE), 57.91556132204081)
    assert_close(obj(700), 57.91556132204081)
    assert_close(obj.calculate(3000, WEBBOOK_SHOMATE), 128.73412366666665)
    assert_close(obj.calculate_integral(1, 700, WEBBOOK_SHOMATE), -387562.193149271)
    assert_close(obj.calculate_integral_over_T(1, 700, WEBBOOK_SHOMATE), -210822.6509202891)

    assert_close(obj.T_dependent_property_integral(330, 700), 19434.50551451386)
    assert_close(obj.T_dependent_property_integral_over_T(330, 700), 38.902796756264706)

    assert eval(str(obj)) == obj
    new = HeatCapacityGas.from_json(json.loads(json.dumps(obj.as_json())))
    assert new == obj


@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_HeatCapacityGas_CoolProp():
    from CoolProp.CoolProp import PropsSI
    EtOH = HeatCapacityGas(CASRN='64-17-5', similarity_variable=0.1953615, MW=46.06844, extrapolation=None, method=COOLPROP)
    assert EtOH.T_dependent_property(5000.0) is None
    assert EtOH.T_dependent_property(1.0) is None
    assert_close(EtOH.T_dependent_property(305.0), 66.25918325111196, rtol=1e-7)

    dH5 = EtOH.calculate_integral(200, 300,'COOLPROP')
    assert_close(dH5, 5838.118293585357, rtol=5e-5)

    dS =  EtOH.calculate_integral_over_T(200, 300, 'COOLPROP')
    assert_close(dS, 23.487556909586853, rtol=1e-5)


    # flash not converge due to melting P
    obj = HeatCapacityGas(CASRN='106-97-8')
    assert eval(str(obj)) == obj
    new = HeatCapacityGas.from_json(obj.as_json())
    assert new == obj
    assert_close(obj.calculate(134.895, COOLPROP), 64.30715649610785)

    # flash not converge at high P
    obj = HeatCapacityGas(CASRN='306-83-2')
    assert eval(str(obj)) == obj
    new = HeatCapacityGas.from_json(obj.as_json())
    assert new == obj
    assert_close(obj.calculate(obj.T_limits[COOLPROP][0], COOLPROP), 72.45489837498226, rtol=1e-7)

    # issue # 120
    H2 = HeatCapacityGas(CASRN='1333-74-0')
    for T in [50, 100, 200, 500, 800, 1000, 2000]:

        assert_close(H2.calculate(T, 'COOLPROP'),
                     PropsSI('Cp0molar', 'T', T,'P', 101325.0, 'hydrogen'), rtol=1e-13)


@pytest.mark.meta_T_dept
def test_HeatCapacityGas_Joback():
    obj = HeatCapacityGas(CASRN='124-18-5')
    obj.method = JOBACK
    assert_close(obj(300), 236.04260000000002)
    assert_close(obj.calculate(300, JOBACK), 236.04260000000002)
    assert_close(obj.T_dependent_property_integral(300, 400), 26820.025000000005)
    assert_close(obj.T_dependent_property_integral_over_T(300, 400), 76.72232911998739)

    # Chemical for which  Joback did not work
    obj = HeatCapacityGas(CASRN='55-18-5')
    assert JOBACK not in obj.all_methods

    # Chemical with no Tc
    obj = HeatCapacityGas(CASRN='23213-96-9')
    assert not isnan(obj.T_limits[JOBACK][1])



@pytest.mark.meta_T_dept
def test_HeatCapacityGas_cheb_fit():
    Tmin, Tmax = 50, 1500.0
    toluene_TRC_cheb_fit = [194.9993931442641, 135.143566535142, -31.391834328585, -0.03951841213554952, 5.633110876073714, -3.686554783541794, 1.3108038668007862, -0.09053861376310801, -0.2614279887767278, 0.24832452742026911, -0.15919652548841812, 0.09374295717647019, -0.06233192560577938, 0.050814520356653126, -0.046331125185531064, 0.0424579816955023, -0.03739513702085129, 0.031402017733109244, -0.025212485578021915, 0.01939423141593144, -0.014231480849538403, 0.009801281575488097, -0.006075456686871594, 0.0029909809015365996, -0.0004841890018462136, -0.0014991199985455728, 0.0030051480117581075, -0.004076901418829215, 0.004758297389532928, -0.005096275567543218, 0.00514099984344718, -0.004944736724873944, 0.004560044671604424, -0.004037777783658769, 0.0034252408915679267, -0.002764690626354871, 0.0020922734527478726, -0.0014374230267101273, 0.0008226963858916081, -0.00026400260413972365, -0.0002288377348015347, 0.0006512726893767029, -0.0010030137199867895, 0.0012869214641443305, -0.001507857723972772, 0.001671575150882565, -0.0017837100581746812, 0.001848935469520696, -0.0009351605848800237]
    fit_obj = HeatCapacityGas(cheb_fit=(Tmin, Tmax, toluene_TRC_cheb_fit))

    assert_close(fit_obj(300), 104.46956642594124, rtol=1e-13)
    obj = np.polynomial.chebyshev.Chebyshev(toluene_TRC_cheb_fit, domain=(Tmin, Tmax))
    assert_close(obj.deriv(1)(300), fit_obj.T_dependent_property_derivative(300, order=1), rtol=1e-15)
    assert_close(obj.deriv(2)(300), fit_obj.T_dependent_property_derivative(300, order=2), rtol=1e-15)
    assert_close(obj.deriv(3)(300), fit_obj.T_dependent_property_derivative(300, order=3), rtol=1e-15)
    assert_close(obj.deriv(4)(300), fit_obj.T_dependent_property_derivative(300, order=4), rtol=1e-15)
    assert_close(obj.integ()(500) - obj.integ()(300), fit_obj.T_dependent_property_integral(300, 500), rtol=1e-15)


    assert_close(fit_obj.T_dependent_property_derivative(300, order=1), 0.36241217517888635, rtol=1e-13)
    assert_close(fit_obj.T_dependent_property_derivative(300, order=2), -6.445511348110282e-06, rtol=1e-13)
    assert_close(fit_obj.T_dependent_property_derivative(300, order=3), -8.804754988590911e-06, rtol=1e-13)
    assert_close(fit_obj.T_dependent_property_derivative(300, order=4), 1.2298003967617247e-07, rtol=1e-13)
    assert_close(fit_obj.T_dependent_property_integral(300, 500), 27791.638479021327, rtol=1e-13)

@pytest.mark.meta_T_dept
def test_HeatCapacityGas_linear_extrapolation():
    CpObj = HeatCapacityGas(CASRN='67-56-1', extrapolation='linear')
    CpObj.method = TRCIG
    assert_close(CpObj.T_dependent_property(CpObj.T_limits[TRCIG][1]),
                 CpObj.T_dependent_property(CpObj.T_limits[TRCIG][1]-1e-6))
    assert_close(CpObj.T_dependent_property(CpObj.T_limits[TRCIG][1]),
                 CpObj.T_dependent_property(CpObj.T_limits[TRCIG][1]+1e-6))

    assert_close(CpObj.T_dependent_property(CpObj.T_limits[TRCIG][0]),
                 CpObj.T_dependent_property(CpObj.T_limits[TRCIG][0]-1e-6))
    assert_close(CpObj.T_dependent_property(CpObj.T_limits[TRCIG][0]),
                 CpObj.T_dependent_property(CpObj.T_limits[TRCIG][0]+1e-6))



@pytest.mark.meta_T_dept
def test_HeatCapacityGas_integrals():
    # Enthalpy integrals
    EtOH = HeatCapacityGas(CASRN='64-17-5', similarity_variable=0.1953615, MW=46.06844)
    dH1 = EtOH.calculate_integral(200, 300, 'TRCIG')
    assert_close(dH1, 5828.905647337944)

    dH2 = EtOH.calculate_integral(200, 300, 'POLING_POLY')
    assert_close(dH2, 5851.1980281476)

    dH3 = EtOH.calculate_integral(200, 300, 'POLING_CONST')
    assert_close(dH3, 6520.999999999999)

    dH4 = EtOH.calculate_integral(200, 300, 'CRCSTD')
    assert_close(dH4, 6559.999999999999)

    dH4 = EtOH.calculate_integral(200, 300,'LASTOVKA_SHAW')
    assert_close(dH4, 6183.016942750752, rtol=1e-5)


    dH = EtOH.calculate_integral(200, 300, 'VDI_TABULAR')
    assert_close(dH, 6610.821140000002)

    # Entropy integrals
    dS = EtOH.calculate_integral_over_T(200, 300, 'POLING_POLY')
    assert_close(dS, 23.5341074921551)

    dS = EtOH.calculate_integral_over_T(200, 300, 'POLING_CONST')
    assert_close(dS, 26.4403796997334)

    dS = EtOH.calculate_integral_over_T(200, 300, 'TRCIG')
    assert_close(dS, 23.4427894111345)

    dS = EtOH.calculate_integral_over_T(200, 300, 'CRCSTD')
    assert_close(dS, 26.59851109189558)

    dS = EtOH.calculate_integral_over_T(200, 300, 'LASTOVKA_SHAW')
    assert_close(dS, 24.86700348570956, rtol=1e-5)

    dS = EtOH.calculate_integral_over_T(200, 300, VDI_TABULAR)
    assert_close(dS, 26.590569427910076)


@pytest.mark.meta_T_dept
def test_HeatCapacitySolid():
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)

    assert eval(str(NaCl)) == NaCl
    new = HeatCapacitySolid.from_json(NaCl.as_json())
    assert new == NaCl


    NaCl.method = 'PERRY151'
    assert_close(NaCl.T_dependent_property(298.15), 50.38469032, rtol=1e-6)
    NaCl.method = 'JANAF'
    assert_close(NaCl.T_dependent_property(298.15), 50.509, rtol=1e-6)
    NaCl.method = 'LASTOVKA_S'
    assert_close(NaCl.T_dependent_property(298.15), 20.0650724340588, rtol=1e-6)
    NaCl.method = 'CRCSTD'
    assert_close(NaCl.T_dependent_property(298.15), 50.5, rtol=1e-6)
    NaCl.method = 'WEBBOOK_SHOMATE'
    assert_close(NaCl.T_dependent_property(298.15), 50.50124702353165, rtol=1e-6)
    
    NaCl.extrapolation = None
    for i in NaCl.all_methods:
        NaCl.method = i
        assert NaCl.T_dependent_property(20000) is None

    with pytest.raises(Exception):
        NaCl.test_method_validity('BADMETHOD', 300)

    assert False is NaCl.test_property_validity(-1)
    assert False is NaCl.test_property_validity(101000.0)

    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Ts = [200, 300, 400, 500, 600]
    Cps = [12.965044960703908, 20.206353934945987, 28.261467986645872, 37.14292010552292, 46.85389719453655]
    NaCl.add_tabular_data(Ts=Ts, properties=Cps, name='stuff')
    assert_close(NaCl.T_dependent_property(275), 18.320355898506502, rtol=1E-5)

    NaCl.extrapolation = None
    assert NaCl.T_dependent_property(601) is None

#    assert eval(str(NaCl)) == NaCl # Need tabular data in init
    new = HeatCapacitySolid.from_json(NaCl.as_json())
    assert new == NaCl

@pytest.mark.meta_T_dept
def test_HeatCapacitySolid_BaBr2():
    BaBr2 = HeatCapacitySolid(CASRN='10553-31-8')
    BaBr2.extrapolation = 'nolimit'
    assert_close(BaBr2(200), 74.890978176)
    assert_close(BaBr2(400),79.216280568)

    assert_close(BaBr2.calculate_integral(330, 450, WEBBOOK_SHOMATE), 9480.005596336567)
    assert_close(BaBr2.calculate_integral_over_T(330, 450, WEBBOOK_SHOMATE), 24.481487055089474)

    assert_close(BaBr2.T_dependent_property_integral(330, 450), 9480.005596336567)
    assert_close(BaBr2.T_dependent_property_integral_over_T(330, 450), 24.481487055089474)

    assert_close(BaBr2.T_dependent_property_integral(1, 10000), 1790167.9273896758)
    assert_close(BaBr2.T_dependent_property_integral_over_T(1, 10000), 832.6150719313177)

    assert eval(str(BaBr2)) == BaBr2
    new = HeatCapacitySolid.from_json(json.loads(json.dumps(BaBr2.as_json())))
    assert new == BaBr2


@pytest.mark.meta_T_dept
def test_HeatCapacitySolid_integrals():
    from thermo.heat_capacity import CRCSTD, LASTOVKA_S, PERRY151
    # Enthalpy integrals
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    dH1 = NaCl.calculate_integral(100, 150, LASTOVKA_S)
    assert_close(dH1, 401.58058175282446)

    dH2 = NaCl.calculate_integral(100, 150, CRCSTD)
    assert_close(dH2, 2525.0) # 50*50.5

    dH3 = NaCl.calculate_integral(100, 150,  PERRY151)
    assert_close(dH3, 2367.097999999999)

    # Tabular integration - not great
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Ts = [200, 300, 400, 500, 600]
    Cps = [12.965044960703908, 20.206353934945987, 28.261467986645872, 37.14292010552292, 46.85389719453655]
    NaCl.add_tabular_data(Ts=Ts, properties=Cps, name='stuff')
    dH4 = NaCl.calculate_integral(200, 300, 'stuff')
    assert_close(dH4, 1651.8556007162392, rtol=1E-5)

    # Entropy integrals
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    dS1 = NaCl.calculate_integral_over_T(100, 150, LASTOVKA_S)
    assert_close(dS1, 3.213071341895563)

    dS2 = NaCl.calculate_integral_over_T(100, 150,  PERRY151)
    assert_close(dS2, 19.183508272982)

    dS3 = NaCl.calculate_integral_over_T(100, 150, CRCSTD)
    assert_close(dS3, 20.4759879594623)

    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Ts = [200, 300, 400, 500, 600]
    Cps = [12.965044960703908, 20.206353934945987, 28.261467986645872, 37.14292010552292, 46.85389719453655]
    NaCl.add_tabular_data(Ts=Ts, properties=Cps, name='stuff')
    dS4 = NaCl.calculate_integral_over_T(100, 150, 'stuff')
    assert_close(dS4, 3.00533159156869)


@pytest.mark.meta_T_dept
def test_HeatCapacityLiquid():
    tol = HeatCapacityLiquid(CASRN='108-88-3', MW=92.13842, Tc=591.75, omega=0.257, Cpgm=115.30398669098454, similarity_variable=0.16279853724428964)
#    Cpl_calc = []
#    for i in tol.all_methods:
#        tol.method = i
#        Cpl_calc.append(tol.T_dependent_property(330))

    tol.method = ROWLINSON_BONDI
    assert_close(tol.T_dependent_property(330.0), 165.45542037912406, rtol=1e-10)
    tol.method = ZABRANSKY_QUASIPOLYNOMIAL_C
    assert_close(tol.T_dependent_property(330.0), 165.472878778683, rtol=1e-10)
    tol.method = CRCSTD
    assert_close(tol.T_dependent_property(330.0), 157.3, rtol=1e-10)
    tol.method = ROWLINSON_POLING
    assert_close(tol.T_dependent_property(330.0), 167.33806248209527, rtol=1e-10)
    tol.method = POLING_CONST
    assert_close(tol.T_dependent_property(330.0), 157.29, rtol=1e-10)
    tol.method = ZABRANSKY_SPLINE_SAT
    assert_close(tol.T_dependent_property(330.0), 166.69813077891135, rtol=1e-10)
    tol.method = DADGOSTAR_SHAW
    assert_close(tol.T_dependent_property(330.0), 175.34392562391267, rtol=1e-10)
    tol.method = ZABRANSKY_SPLINE_C
    assert_close(tol.T_dependent_property(330.0), 166.7156677848114, rtol=1e-10)
    tol.method = VDI_TABULAR
    assert_close(tol.T_dependent_property(330.0), 166.52477714085708, rtol=1e-10)

    assert eval(str(tol)) == tol
    new = HeatCapacityLiquid.from_json(tol.as_json())
    assert new == tol
    tol.extrapolation = None
    for i in tol.all_methods:
        tol.method = i
        assert tol.T_dependent_property(2000) is None


    with pytest.raises(Exception):
        tol.test_method_validity('BADMETHOD', 300)

    assert False is tol.test_property_validity(-1)
    assert False is tol.test_property_validity(101000.0)
    assert True is tol.test_property_validity(100)



def test_heat_capacity_liquid_methods():
    # Test case 1: Propylbenzene
    expected_values = {
        'ZABRANSKY_QUASIPOLYNOMIAL': 214.6499551694668,
        'ZABRANSKY_SPLINE': 214.69679325320664,
        'POLING_CONST': 214.71,
        'CRCSTD': 214.7
    }
    
    propylbenzene = HeatCapacityLiquid(MW=120.19158, CASRN='103-65-1', Tc=638.35)
    for method, expected in expected_values.items():
        propylbenzene.method = method
        assert_close(propylbenzene.T_dependent_property(298.15), expected, rtol=1e-13)
        
    # Test JSON serialization
    new = HeatCapacityLiquid.from_json(propylbenzene.as_json())
    assert new == propylbenzene

    # Test case 2: CTP (Chloro-thiophenol?)
    expected_values = {
        'ZABRANSKY_SPLINE_SAT': 134.1186737739494,
        'ZABRANSKY_QUASIPOLYNOMIAL_SAT': 134.1496585096233
    }
    
    ctp = HeatCapacityLiquid(MW=118.58462, CASRN='96-43-5')
    for method, expected in expected_values.items():
        ctp.method = method
        assert_close(ctp.T_dependent_property(250), expected, rtol=1e-13)
        
    # Test JSON serialization
    new = HeatCapacityLiquid.from_json(ctp.as_json())
    assert new == ctp
    
@pytest.mark.meta_T_dept
def test_HeatCapacityLiquid_webbook():
    obj = HeatCapacityLiquid(CASRN='7732-18-5')
    obj.method = WEBBOOK_SHOMATE
    assert_close(obj.calculate(300, WEBBOOK_SHOMATE), 75.35107055555557)
    assert_close(obj(340), 75.41380745425609)
    assert_close(obj.calculate(250, WEBBOOK_SHOMATE), 77.78926287500002)
    assert_close(obj.calculate(510, WEBBOOK_SHOMATE), 84.94628487578056)


    assert_close(obj.calculate_integral(330, 450, WEBBOOK_SHOMATE), 9202.213117294952)
    assert_close(obj.calculate_integral_over_T(330, 450, WEBBOOK_SHOMATE), 23.75523929492681)

    assert_close(obj.T_dependent_property_integral(330, 450), 9202.213117294952)
    assert_close(obj.T_dependent_property_integral_over_T(330, 450), 23.75523929492681)

    assert eval(str(obj)) == obj
    new = HeatCapacityLiquid.from_json(json.loads(json.dumps(obj.as_json())))
    assert new == obj

@pytest.mark.meta_T_dept
def test_HeatCapacityLiquid_webbook_multi_range():
    obj = HeatCapacityLiquid(CASRN='17702-41-9')
    obj.method = WEBBOOK_SHOMATE
    
    # Test points in first range (371.93 - 500.0)
    assert_close(obj.calculate(380, WEBBOOK_SHOMATE), 304.733284779612, rtol=1e-12)
    assert_close(obj.calculate(450, WEBBOOK_SHOMATE), 353.08496493827084, rtol=1e-12)
    assert_close(obj(400), 323.00484249999954)
    
    # Test points in second range (500.0 - 1500.0)
    assert_close(obj.calculate(600, WEBBOOK_SHOMATE), 390.46000317333335, rtol=1e-12)
    assert_close(obj.calculate(1000, WEBBOOK_SHOMATE), 502.16795399999995, rtol=1e-12)
    assert_close(obj.calculate(1400, WEBBOOK_SHOMATE), 543.929336315102, rtol=1e-12)
    
    # Test integral within first range
    assert_close(obj.calculate_integral(380, 450, WEBBOOK_SHOMATE), 23297.823098363715, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(380, 450, WEBBOOK_SHOMATE), 56.155399861337735, rtol=1e-12)
    
    # Test integral within second range
    assert_close(obj.calculate_integral(600, 1000, WEBBOOK_SHOMATE), 181160.8711573333, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(600, 1000, WEBBOOK_SHOMATE), 228.92179217773193, rtol=1e-12)
    
    # Test integral across both ranges (increasing temperature)
    assert_close(obj.calculate_integral(400, 600, WEBBOOK_SHOMATE), 71668.1105926666, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(400, 600, WEBBOOK_SHOMATE), 144.60158290807362, rtol=1e-12)
    
    # Test integral across both ranges (decreasing temperature)
    assert_close(obj.calculate_integral(600, 400, WEBBOOK_SHOMATE), -71668.1105926666, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(600, 400, WEBBOOK_SHOMATE), -144.60158290807362, rtol=1e-12)
    
    # Test the T-dependent property methods
    assert_close(obj.T_dependent_property_integral(400, 600), 71668.1105926666, rtol=1e-12)
    assert_close(obj.T_dependent_property_integral_over_T(400, 600), 144.60158290807362, rtol=1e-12)
    assert_close(obj.T_dependent_property_integral(600, 400), -71668.1105926666, rtol=1e-12)
    assert_close(obj.T_dependent_property_integral_over_T(600, 400), -144.60158290807362, rtol=1e-12)
    
    # Test edge cases at range boundaries
    assert_close(obj.calculate(500, WEBBOOK_SHOMATE), 347.2722559999989, rtol=1e-12)  # Exactly at range boundary
    assert_close(obj.calculate_integral(495, 505, WEBBOOK_SHOMATE), 3485.0857254289776, rtol=1e-12)  # Small interval across boundary

@pytest.mark.meta_T_dept
def test_HeatCapacityGas_webbook_multi_range():
    obj = HeatCapacityGas(CASRN='1333-74-0')
    obj.method = WEBBOOK_SHOMATE

    # Test points in range 1 (298.0 - 1000.0)
    assert_close(obj.calculate(400, WEBBOOK_SHOMATE), 29.181610324, rtol=1e-12)
    assert_close(obj.calculate(800, WEBBOOK_SHOMATE), 29.624988277, rtol=1e-12)
    assert_close(obj(400), 29.181610324)

    # Test points in range 2 (1000.0 - 2500.0)
    assert_close(obj.calculate(1200, WEBBOOK_SHOMATE), 30.990938990666667, rtol=1e-12)
    assert_close(obj.calculate(2000, WEBBOOK_SHOMATE), 34.2790545, rtol=1e-12)

    # Test points in range 3 (2500.0 - 6000.0)
    assert_close(obj.calculate(3000, WEBBOOK_SHOMATE), 37.08898277777777, rtol=1e-12)
    assert_close(obj.calculate(5000, WEBBOOK_SHOMATE), 40.82801051999999, rtol=1e-12)

    # Test integral within range 1
    assert_close(obj.calculate_integral(398.0, 900.0, WEBBOOK_SHOMATE), 14775.319358526369, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(398.0, 900.0, WEBBOOK_SHOMATE), 23.98231431098887, rtol=1e-12)

    # Test integral within range 2
    assert_close(obj.calculate_integral(1100.0, 2400.0, WEBBOOK_SHOMATE), 43209.09458952272, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(1100.0, 2400.0, WEBBOOK_SHOMATE), 25.673382170455824, rtol=1e-12)

    # Test integral within range 3
    assert_close(obj.calculate_integral(2600.0, 5900.0, WEBBOOK_SHOMATE), 130051.97018395108, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(2600.0, 5900.0, WEBBOOK_SHOMATE), 31.96511968233716, rtol=1e-12)

    # Test integrals across ranges
    assert_close(obj.calculate_integral(400, 1200, WEBBOOK_SHOMATE), 23837.7216288, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(400, 1200, WEBBOOK_SHOMATE), 32.574667531356454, rtol=1e-12)
    assert_close(obj.calculate_integral(1200, 400, WEBBOOK_SHOMATE), -23837.7216288, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(1200, 400, WEBBOOK_SHOMATE), -32.574667531356454, rtol=1e-12)
    assert_close(obj.calculate_integral(1200, 3000, WEBBOOK_SHOMATE), 61944.86637305, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(1200, 3000, WEBBOOK_SHOMATE), 31.10165669473423, rtol=1e-12)
    assert_close(obj.calculate_integral(3000, 1200, WEBBOOK_SHOMATE), -61944.86637305, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(3000, 1200, WEBBOOK_SHOMATE), -31.10165669473423, rtol=1e-12)

    # Test at range boundaries
    assert_close(obj.calculate(1000.0, WEBBOOK_SHOMATE), 30.204145, rtol=1e-12)
    assert_close(obj.calculate(2500.0, WEBBOOK_SHOMATE), 35.84051015, rtol=1e-12)
    assert_close(obj.calculate_integral(995.0, 1005.0, WEBBOOK_SHOMATE), 302.05397832957533, rtol=1e-12)
    assert_close(obj.calculate_integral(2495.0, 2505.0, WEBBOOK_SHOMATE), 358.37646756316826, rtol=1e-12)

@pytest.mark.meta_T_dept
def test_HeatCapacitySolid_webbook_multi_range():
    obj = HeatCapacitySolid(CASRN='12034-59-2')
    obj.method = WEBBOOK_SHOMATE

    # Test points in range 1 (298.0 - 1090.0)
    assert_close(obj.calculate(400, WEBBOOK_SHOMATE), 63.51568005, rtol=1e-12)
    assert_close(obj.calculate(800, WEBBOOK_SHOMATE), 79.47318521249998, rtol=1e-12)
    assert_close(obj(400), 63.51568005)

    # Test points in range 2 (1090.0 - 1200.0)
    assert_close(obj.calculate(1120, WEBBOOK_SHOMATE), 92.88414379954652, rtol=1e-12)
    assert_close(obj.calculate(1180, WEBBOOK_SHOMATE), 92.88549941984948, rtol=1e-12)

    # Test points in range 3 (1200.0 - 2175.0)
    assert_close(obj.calculate(1500, WEBBOOK_SHOMATE), 83.05239470833335, rtol=1e-12)
    assert_close(obj.calculate(2000, WEBBOOK_SHOMATE), 83.05239650000001, rtol=1e-12)

    # Test integral within range 1
    assert_close(obj.calculate_integral(398.0, 990.0, WEBBOOK_SHOMATE), 44605.360970269205, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(398.0, 990.0, WEBBOOK_SHOMATE), 67.06507385030875, rtol=1e-12)

    # Test integral within range 2
    assert_close(obj.calculate_integral(1190.0, 1100.0, WEBBOOK_SHOMATE), -8359.631002585797, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(1190.0, 1100.0, WEBBOOK_SHOMATE), -7.30474943842529, rtol=1e-12)

    # Test integral within range 3
    assert_close(obj.calculate_integral(1300.0, 2075.0, WEBBOOK_SHOMATE), 64365.60636545849, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(1300.0, 2075.0, WEBBOOK_SHOMATE), 38.83504164532849, rtol=1e-12)

    # Test integrals across ranges
    assert_close(obj.calculate_integral(400, 1120, WEBBOOK_SHOMATE), 56178.2961682884, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(400, 1120, WEBBOOK_SHOMATE), 77.84288396170393, rtol=1e-12)
    assert_close(obj.calculate_integral(1120, 400, WEBBOOK_SHOMATE), -56178.2961682884, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(1120, 400, WEBBOOK_SHOMATE), -77.84288396170393, rtol=1e-12)
    assert_close(obj.calculate_integral(1120, 1500, WEBBOOK_SHOMATE), 32346.516118692438, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(1120, 1500, WEBBOOK_SHOMATE), 24.94100662029956, rtol=1e-12)
    assert_close(obj.calculate_integral(1500, 1120, WEBBOOK_SHOMATE), -32346.516118692438, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(1500, 1120, WEBBOOK_SHOMATE), -24.94100662029956, rtol=1e-12)
    assert_close(obj.calculate_integral(400, 1500, WEBBOOK_SHOMATE), 88524.81228698083, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(400, 1500, WEBBOOK_SHOMATE), 102.7838905820035, rtol=1e-12)
    assert_close(obj.calculate_integral(1500, 400, WEBBOOK_SHOMATE), -88524.81228698083, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(1500, 400, WEBBOOK_SHOMATE), -102.7838905820035, rtol=1e-12)

    # Test at range boundaries
    assert_close(obj.calculate(1090.0, WEBBOOK_SHOMATE), 91.12680253387941, rtol=1e-12)
    assert_close(obj.calculate(1200.0, WEBBOOK_SHOMATE), 92.88504085888887, rtol=1e-12)
    assert_close(obj.calculate_integral(1085.0, 1095.0, WEBBOOK_SHOMATE), 919.5654260597075, rtol=1e-12)
    assert_close(obj.calculate_integral(1195.0, 1205.0, WEBBOOK_SHOMATE), 879.6877349119604, rtol=1e-12)

@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
def test_HeatCapacityLiquid_CoolProp():
    tol = HeatCapacityLiquid(CASRN='108-88-3', MW=92.13842, Tc=591.75,
          omega=0.257, Cpgm=115.30398669098454, similarity_variable=0.16279853724428964)

    tol.method = COOLPROP
    assert_close(tol.T_dependent_property(330.0), 166.52164399712314, rtol=1e-10)

    dH = tol.calculate_integral(200, 300, COOLPROP)
    assert_close(dH, 14501.714588188637)

    dS = tol.calculate_integral_over_T(200, 300, COOLPROP)
    assert_close(dS, 58.50970500781979)

@pytest.mark.meta_T_dept
def test_HeatCapacityLiquid_Custom_parameters():
    obj = HeatCapacityLiquid()
    # Coefficients from Perry's 8th edition, all multiplied by 1e-3 to switch from J/kmol/K to J/mol/K
    obj.add_correlation('Eq100test', 'DIPPR100', A=105.8, B=-.36223, C=0.9379e-3, Tmin=175.47, Tmax=400.0)
    assert_close(obj(400), 110.972, rtol=1e-13)
    assert_close(obj.calculate_integral_over_T(200, 300, 'Eq100test'), 30.122708437843926, rtol=1e-12)
    assert_close(obj.calculate_integral(200, 300, 'Eq100test'), 7464.283333333329, rtol=1e-12)


@pytest.mark.meta_T_dept
def test_HeatCapacityLiquid_integrals():
    from thermo.heat_capacity import (
        CRCSTD,
        DADGOSTAR_SHAW,
        ROWLINSON_BONDI,
        ROWLINSON_POLING,
        ZABRANSKY_QUASIPOLYNOMIAL,
        ZABRANSKY_QUASIPOLYNOMIAL_C,
        ZABRANSKY_QUASIPOLYNOMIAL_SAT,
        ZABRANSKY_SPLINE,
        ZABRANSKY_SPLINE_C,
        ZABRANSKY_SPLINE_SAT,
    )
    tol = HeatCapacityLiquid(CASRN='108-88-3', MW=92.13842, Tc=591.75,
          omega=0.257, Cpgm=115.30398669098454, similarity_variable=0.16279853724428964)

    propylbenzene = HeatCapacityLiquid(MW=120.19158, CASRN='103-65-1', Tc=638.35)

    ctp = HeatCapacityLiquid(MW=118.58462, CASRN='96-43-5')

    dH = tol.calculate_integral(200, 300, CRCSTD)
    assert_close(dH, 15730)

    dH = tol.calculate_integral(200, 300, DADGOSTAR_SHAW)
    assert_close(dH, 14395.231307169146)

    dH = tol.calculate_integral(200, 300, ROWLINSON_POLING)
    assert_close(dH, 17332.447330329327)

    dH = tol.calculate_integral(200, 300, ROWLINSON_BONDI)
    assert_close(dH, 17161.367460370562)

    dH = tol.calculate_integral(200, 300, ZABRANSKY_SPLINE_C)
    assert_close(dH, 14588.050659771678)

    # Test over different coefficient sets
    dH = tol.calculate_integral(200, 500, ZABRANSKY_SPLINE_SAT)
    assert_close(dH, 52806.422778119224)

    dH = tol.calculate_integral(200, 300, ZABRANSKY_SPLINE_SAT)
    assert_close(dH, 14588.10920744596)

    dH = tol.calculate_integral(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_C)
    assert_close(dH, 14662.031376528757)

    dH = propylbenzene.calculate_integral(200, 300, ZABRANSKY_QUASIPOLYNOMIAL)
    assert_close(dH, 19863.944414041936)

    dH = propylbenzene.calculate_integral(200, 300, ZABRANSKY_SPLINE)
    assert_close(dH, 19865.186385942456)

    dH = ctp.calculate_integral(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_SAT)
    assert_close(dH, 13437.28621249451)

    # Entropy integrals
    dS = tol.calculate_integral_over_T(200, 300, CRCSTD)
    assert_close(dS, 63.779661505414275)

    dS = tol.calculate_integral_over_T(200, 300, DADGOSTAR_SHAW)
    assert_close(dS, 57.78686119989654)

    dS = tol.calculate_integral_over_T(200, 300, ROWLINSON_POLING)
    assert_close(dS, 70.42885653432398)

    dS = tol.calculate_integral_over_T(200, 300, ROWLINSON_BONDI)
    assert_close(dS, 69.73750128980184)

    dS = tol.calculate_integral_over_T(200, 300, ZABRANSKY_SPLINE_C)
    assert_close(dS, 58.866392640147374)

    dS = tol.calculate_integral_over_T(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_C)
    assert_close(dS, 59.16999297436473)

    dS = tol.calculate_integral_over_T(200, 300, ZABRANSKY_SPLINE_SAT)
    assert_close(dS, 58.86648035527116)

    dS = tol.calculate_integral_over_T(200, 500, ZABRANSKY_SPLINE_SAT)
    assert_close(dS, 154.94766581118256)

    dS = propylbenzene.calculate_integral_over_T(200, 300, ZABRANSKY_QUASIPOLYNOMIAL)
    assert_close(dS, 80.13493128839104)

    dS = propylbenzene.calculate_integral_over_T(200, 300, ZABRANSKY_SPLINE)
    assert_close(dS, 80.13636874689294)

    dS = ctp.calculate_integral_over_T(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_SAT)
    assert_close(dS, 54.34708465297109)

@pytest.mark.meta_T_dept
def test_ZABRANSKY_QUASIPOLYNOMIAL_68_12_2():
    obj = HeatCapacityLiquid(CASRN='68-12-2')
    obj.method = 'ZABRANSKY_QUASIPOLYNOMIAL'

    # Test value at T=222.6
    assert_close(obj.calculate(222.6, 'ZABRANSKY_QUASIPOLYNOMIAL'), 136.48274906893334, rtol=1e-12)

    # Test value at T=317.9
    assert_close(obj.calculate(317.9, 'ZABRANSKY_QUASIPOLYNOMIAL'), 151.50687495151507, rtol=1e-12)

    # Test value at T=413.2
    assert_close(obj.calculate(413.2, 'ZABRANSKY_QUASIPOLYNOMIAL'), 171.30335948385098, rtol=1e-12)

    # Test integral across range
    assert_close(obj.calculate_integral(222.6, 413.2, 'ZABRANSKY_QUASIPOLYNOMIAL'), 29025.184901451255, rtol=1e-12)

    # Test integral over T across range
    assert_close(obj.calculate_integral_over_T(222.6, 413.2, 'ZABRANSKY_QUASIPOLYNOMIAL'), 93.11886710198587, rtol=1e-12)

    # Test small interval at T=227.6
    assert_close(obj.calculate_integral(227.6, 247.6, 'ZABRANSKY_QUASIPOLYNOMIAL'), 2772.9552142454486, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(227.6, 247.6, 'ZABRANSKY_QUASIPOLYNOMIAL'), 11.67585743145662, rtol=1e-12)

    # Test small interval at T=317.9
    assert_close(obj.calculate_integral(317.9, 337.9, 'ZABRANSKY_QUASIPOLYNOMIAL'), 3066.1991774556227, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(317.9, 337.9, 'ZABRANSKY_QUASIPOLYNOMIAL'), 9.352790455066042, rtol=1e-12)


@pytest.mark.meta_T_dept
def test_ZABRANSKY_QUASIPOLYNOMIAL_C_75_71_8():
    obj = HeatCapacityLiquid(CASRN='75-71-8')
    obj.method = 'ZABRANSKY_QUASIPOLYNOMIAL_C'

    # Test value at T=179.0
    assert_close(obj.calculate(179.0, 'ZABRANSKY_QUASIPOLYNOMIAL_C'), 99.90463518818633, rtol=1e-12)

    # Test value at T=231.25
    assert_close(obj.calculate(231.25, 'ZABRANSKY_QUASIPOLYNOMIAL_C'), 106.94445437288353, rtol=1e-12)

    # Test value at T=283.5
    assert_close(obj.calculate(283.5, 'ZABRANSKY_QUASIPOLYNOMIAL_C'), 115.55696601613087, rtol=1e-12)

    # Test integral across range
    assert_close(obj.calculate_integral(179.0, 283.5, 'ZABRANSKY_QUASIPOLYNOMIAL_C'), 11200.574885375045, rtol=1e-12)

    # Test integral over T across range
    assert_close(obj.calculate_integral_over_T(179.0, 283.5, 'ZABRANSKY_QUASIPOLYNOMIAL_C'), 49.02014006874771, rtol=1e-12)

    # Test small interval at T=184.0
    assert_close(obj.calculate_integral(184.0, 204.0, 'ZABRANSKY_QUASIPOLYNOMIAL_C'), 2038.4162640895847, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(184.0, 204.0, 'ZABRANSKY_QUASIPOLYNOMIAL_C'), 10.514249886023379, rtol=1e-12)

    # Test small interval at T=231.25
    assert_close(obj.calculate_integral(231.25, 251.25, 'ZABRANSKY_QUASIPOLYNOMIAL_C'), 2167.5097413800795, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(231.25, 251.25, 'ZABRANSKY_QUASIPOLYNOMIAL_C'), 8.987982696196866, rtol=1e-12)


@pytest.mark.meta_T_dept
def test_ZABRANSKY_QUASIPOLYNOMIAL_SAT_16587_33_0():
    obj = HeatCapacityLiquid(CASRN='16587-33-0')
    obj.method = 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'

    # Test value at T=292.3
    assert_close(obj.calculate(292.3, 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'), 265.1994265364768, rtol=1e-12)

    # Test value at T=441.15
    assert_close(obj.calculate(441.15, 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'), 344.7400719038499, rtol=1e-12)

    # Test value at T=590.0
    assert_close(obj.calculate(590.0, 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'), 424.74456270573734, rtol=1e-12)

    # Test integral across range
    assert_close(obj.calculate_integral(292.3, 590.0, 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'), 102681.66003067768, rtol=1e-12)

    # Test integral over T across range
    assert_close(obj.calculate_integral_over_T(292.3, 590.0, 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'), 235.66671132682495, rtol=1e-12)

    # Test small interval at T=297.3
    assert_close(obj.calculate_integral(297.3, 317.3, 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'), 5457.492384415586, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(297.3, 317.3, 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'), 17.76213267424123, rtol=1e-12)

    # Test small interval at T=441.15
    assert_close(obj.calculate_integral(441.15, 461.15, 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'), 7005.321824996499, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(441.15, 461.15, 'ZABRANSKY_QUASIPOLYNOMIAL_SAT'), 15.528432597873007, rtol=1e-12)

@pytest.mark.meta_T_dept
def test_ZABRANSKY_SPLINE_74_11_3():
    obj = HeatCapacityLiquid(CASRN='74-11-3')
    obj.method = 'ZABRANSKY_SPLINE'

    # Test calculations in range 1 (514.9 - 545.0 K)
    assert_close(obj.calculate(529.95, 'ZABRANSKY_SPLINE'), 331.4138170004083, rtol=1e-12)

    # Test integral within range
    assert_close(obj.calculate_integral(519.9, 540.0, 'ZABRANSKY_SPLINE'), 6659.2056956999, rtol=1e-12)

    # Test integral over T within range
    assert_close(obj.calculate_integral_over_T(519.9, 540.0, 'ZABRANSKY_SPLINE'), 12.564785797683726, rtol=1e-12)

    # Test calculations in range 2 (545.0 - 579.3 K)
    assert_close(obj.calculate(562.15, 'ZABRANSKY_SPLINE'), 349.50635869347667, rtol=1e-12)

    # Test integral within range
    assert_close(obj.calculate_integral(550.0, 574.3, 'ZABRANSKY_SPLINE'), 8515.009298082441, rtol=1e-12)

    # Test integral over T within range
    assert_close(obj.calculate_integral_over_T(550.0, 574.3, 'ZABRANSKY_SPLINE'), 15.14802485601831, rtol=1e-12)

    # Test calculations across ranges (forward)
    assert_close(obj.calculate_integral(524.9, 569.3, 'ZABRANSKY_SPLINE'), 15221.916206590831, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(524.9, 569.3, 'ZABRANSKY_SPLINE'), 27.82474840330542, rtol=1e-12)

    # Test calculations across ranges (reverse)
    assert_close(obj.calculate_integral(569.3, 524.9, 'ZABRANSKY_SPLINE'), -15221.916206590831, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(569.3, 524.9, 'ZABRANSKY_SPLINE'), -27.82474840330542, rtol=1e-12)

    # Test calculations exactly at range boundary
    assert_close(obj.calculate(545.0, 'ZABRANSKY_SPLINE'), 344.42108099394665, rtol=1e-12)


@pytest.mark.meta_T_dept
def test_ZABRANSKY_SPLINE_SAT_57_55_6():
    obj = HeatCapacityLiquid(CASRN='57-55-6')
    obj.method = 'ZABRANSKY_SPLINE_SAT'

    # Test calculations in range 1 (194.3 - 400.0 K)
    assert_close(obj.calculate(297.15, 'ZABRANSKY_SPLINE_SAT'), 189.62973212653358, rtol=1e-12)

    # Test integral within range
    assert_close(obj.calculate_integral(199.3, 395.0, 'ZABRANSKY_SPLINE_SAT'), 37276.51343084486, rtol=1e-12)

    # Test integral over T within range
    assert_close(obj.calculate_integral_over_T(199.3, 395.0, 'ZABRANSKY_SPLINE_SAT'), 127.0711810931164, rtol=1e-12)

    # Test calculations in range 2 (400.0 - 600.0 K)
    assert_close(obj.calculate(500.0, 'ZABRANSKY_SPLINE_SAT'), 258.0237577369764, rtol=1e-12)

    # Test integral within range
    assert_close(obj.calculate_integral(405.0, 595.0, 'ZABRANSKY_SPLINE_SAT'), 48990.265739913164, rtol=1e-12)

    # Test integral over T within range
    assert_close(obj.calculate_integral_over_T(405.0, 595.0, 'ZABRANSKY_SPLINE_SAT'), 98.68310292056651, rtol=1e-12)

    # Test calculations across ranges (forward)
    assert_close(obj.calculate_integral(204.3, 590.0, 'ZABRANSKY_SPLINE_SAT'), 86447.08477908312, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(204.3, 590.0, 'ZABRANSKY_SPLINE_SAT'), 225.459968679079, rtol=1e-12)

    # Test calculations across ranges (reverse)
    assert_close(obj.calculate_integral(590.0, 204.3, 'ZABRANSKY_SPLINE_SAT'), -86447.08477908312, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(590.0, 204.3, 'ZABRANSKY_SPLINE_SAT'), -225.459968679079, rtol=1e-12)

    # Test calculations exactly at range boundary
    assert_close(obj.calculate(400.0, 'ZABRANSKY_SPLINE_SAT'), 233.83903965303566, rtol=1e-12)


@pytest.mark.meta_T_dept
def test_ZABRANSKY_SPLINE_C_57_55_6():
    obj = HeatCapacityLiquid(CASRN='57-55-6')
    obj.method = 'ZABRANSKY_SPLINE_C'

    # Test calculations in range 1 (194.3 - 400.0 K)
    assert_close(obj.calculate(297.15, 'ZABRANSKY_SPLINE_C'), 189.61905985803662, rtol=1e-12)

    # Test integral within range
    assert_close(obj.calculate_integral(199.3, 395.0, 'ZABRANSKY_SPLINE_C'), 37276.75002099501, rtol=1e-12)

    # Test integral over T within range
    assert_close(obj.calculate_integral_over_T(199.3, 395.0, 'ZABRANSKY_SPLINE_C'), 127.06726880740395, rtol=1e-12)

    # Test calculations in range 2 (400.0 - 600.0 K)
    assert_close(obj.calculate(500.0, 'ZABRANSKY_SPLINE_C'), 258.38709975338986, rtol=1e-12)

    # Test integral within range
    assert_close(obj.calculate_integral(405.0, 595.0, 'ZABRANSKY_SPLINE_C'), 49205.038074676486, rtol=1e-12)

    # Test integral over T within range
    assert_close(obj.calculate_integral_over_T(405.0, 595.0, 'ZABRANSKY_SPLINE_C'), 99.06377620128535, rtol=1e-12)

    # Test calculations across ranges (forward)
    assert_close(obj.calculate_integral(204.3, 590.0, 'ZABRANSKY_SPLINE_C'), 86636.85504670914, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(204.3, 590.0, 'ZABRANSKY_SPLINE_C'), 225.79423175871472, rtol=1e-12)

    # Test calculations across ranges (reverse)
    assert_close(obj.calculate_integral(590.0, 204.3, 'ZABRANSKY_SPLINE_C'), -86636.85504670914, rtol=1e-12)
    assert_close(obj.calculate_integral_over_T(590.0, 204.3, 'ZABRANSKY_SPLINE_C'), -225.79423175871472, rtol=1e-12)

    # Test calculations exactly at range boundary
    assert_close(obj.calculate(400.0, 'ZABRANSKY_SPLINE_C'), 233.8644819086472, rtol=1e-12)

@pytest.mark.meta_T_dept
def test_HeatCapacitySolidMixture():
    MWs = [107.8682, 195.078]
    ws = [0.95, 0.05]
    zs = ws_to_zs(ws, MWs)
    T = 298.15
    P = 101325.0
#    m = Mixture(['silver', 'platinum'], ws)
    HeatCapacitySolids = [HeatCapacitySolid(CASRN="7440-22-4", similarity_variable=0.009270572791610502, MW=107.8682, extrapolation="linear", method="POLY_FIT", poly_fit=(273.0, 1234.0, [-1.3937807081430088e-31, 7.536244381034841e-28, -1.690680423482461e-24, 2.0318654521005364e-21, -1.406580331667692e-18, 5.582457975071205e-16, 0.0062759999998829325, 23.430400000009975])),
                          HeatCapacitySolid(CASRN="7440-06-4", similarity_variable=0.005126154666338593, MW=195.078, extrapolation="linear", method="POLY_FIT", poly_fit=(273.0, 1873.0, [3.388131789017202e-37, -6.974130474513008e-33, 4.1785562369164175e-29, -1.19324731353925e-25, 1.8802914323231756e-22, -1.7047279440501588e-19, 8.714473096125867e-17, 0.004853439999977098, 24.76928000000236]))]

    obj = HeatCapacitySolidMixture(CASs=['7440-22-4', '7440-06-4'], HeatCapacitySolids=HeatCapacitySolids, MWs=MWs)

    Cp = obj(T, P, zs, ws)
    assert_close(Cp, 25.327457963474732)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(T, P, zs, ws, 'BADMETHOD')

    with pytest.raises(Exception):
        obj.test_method_validity(T, P, zs, ws, 'BADMETHOD')


@pytest.mark.meta_T_dept
def test_HeatCapacityGasMixture():
    # oxygen and nitrogen

#    m = Mixture(['oxygen', 'nitrogen'], ws=[.4, .6], T=350, P=1E6)
    HeatCapacityGases = [HeatCapacityGas(CASRN="7782-44-7", similarity_variable=0.06250234383789392, MW=31.9988, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
                         HeatCapacityGas(CASRN="7727-37-9", similarity_variable=0.07139440410660612, MW=28.0134, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264]))]
    obj = HeatCapacityGasMixture(CASs=['7782-44-7', '7727-37-9'], HeatCapacityGases=HeatCapacityGases, MWs=[31.9988, 28.0134])

    Cp = obj(350.0, P=1e5, ws=[.4, .6])
    assert_close(Cp, 29.359579781030867, rtol=1e-5)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
def test_HeatCapacityLiquidMixture_aqueous():
    from thermo.heat_capacity import LINEAR, HeatCapacityLiquidMixture

    HeatCapacityLiquids = [HeatCapacityLiquid(method='COOLPROP', CASRN="7732-18-5", MW=18.01528, similarity_variable=0.16652530518537598, Tc=647.14, omega=0.344, extrapolation="linear"),
                           HeatCapacityLiquid(CASRN="7647-14-5", MW=58.44277, similarity_variable=0.034221512772238546, Tc=3400.0, omega=0.1894, extrapolation="linear", method="POLY_FIT", poly_fit=(1077.15, 3400.0, [-1.7104845836996866e-36, 3.0186209409101436e-32, -2.2964525212600158e-28, 9.82884046707168e-25, -2.585879179539946e-21, 4.276734025134063e-18, -0.00016783912672163547, 0.1862851719065294, -12.936804011905963]))]

    obj = HeatCapacityLiquidMixture(MWs=[18.01528, 58.44277], CASs=['7732-18-5', '7647-14-5'], HeatCapacityLiquids=HeatCapacityLiquids)

    Cp = obj(T=301.5, P=101325.0, ws=[.9, .1])
    assert_close(Cp, 72.29666542719279, rtol=1e-5)
    ws = [.9, .1]
    zs = ws_to_zs(ws, [18.01528, 58.44277])

    Cp = obj.calculate(T=301.5, P=101325.0, zs=zs, ws=[.9, .1], method=LINEAR)
    assert_close(Cp, 77.08377446120679, rtol=1e-7)

@pytest.mark.meta_T_dept
def test_HeatCapacityLiquidMixture():
    ws = [.9, .1]
    MWs = [92.13842, 142.28168]
    zs = ws_to_zs(ws, MWs)
#    m = Mixture(['toluene', 'decane'], ws=ws, T=300)

    HeatCapacityLiquids = [HeatCapacityLiquid(CASRN="108-88-3", MW=92.13842, similarity_variable=0.16279853724428964, Tc=591.75, omega=0.257, extrapolation="linear", method="POLY_FIT", poly_fit=(162.0, 570.0, [7.171090290089724e-18, -1.8175720506858e-14, 1.9741936612209287e-11, -1.1980168324612502e-08, 4.438245228007343e-06, -0.0010295403891115538, 0.1475922271028815, -12.06203901868023, 565.3058820511594])),
                           HeatCapacityLiquid(CASRN="124-18-5", MW=142.28168, similarity_variable=0.22490597524572384, Tc=611.7, omega=0.49, extrapolation="linear", method="POLY_FIT", poly_fit=(247.0, 462.4, [-1.0868671859366803e-16, 3.1665821629357415e-13, -4.003344986303024e-10, 2.8664283740244695e-07, -0.00012702231712595518, 0.03563345983140416, -6.170059527783799, 601.9757437895033, -25015.243163919513]))]

    obj = HeatCapacityLiquidMixture(CASs=['108-88-3', '124-18-5'], HeatCapacityLiquids=HeatCapacityLiquids, MWs=MWs)
    assert_close(obj(300.0, P=101325.0, zs=zs, ws=ws), 168.29157865567112, rtol=1E-4)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(300.0, P=101325.0, zs=zs, ws=ws, method='BADMETHOD')

    with pytest.raises(Exception):
        obj.test_method_validity(300.0, P=101325.0, zs=zs, ws=ws, method='BADMETHOD')


@pytest.mark.meta_T_dept
def test_HeatCapacityGas_polynomial_input_forms():
    obj_basic = HeatCapacityGas(quadratic_parameters={'WebBook': {'A': 1e-5, 'B': 2e-5, 'C': 4e-5,
                                                        'Tmin': 177.7, 'Tmax': 264.93}})
    val0 = obj_basic.T_dependent_property(200)
    obj_polynomial = HeatCapacityGas(polynomial_parameters={'test': {'coeffs': [1e-5, 2e-5, 4e-5][::-1],
                                                        'Tmin': 177.7, 'Tmax': 264.93}})
    val1 = obj_polynomial.T_dependent_property(200)
    obj_bestfit = HeatCapacityGas(poly_fit=(177.7, 264.93, [1e-5, 2e-5, 4e-5][::-1]))
    val2 = obj_bestfit.T_dependent_property(200)

    for v in (val0, val1, val2):
        assert_close(v, 1.60401, rtol=1e-13)

    for o in (obj_basic, obj_polynomial, obj_bestfit):
        assert HeatCapacityGas.from_json(o.as_json()) == o
        assert eval(str(o)) == o

@pytest.mark.meta_T_dept
def test_HeatCapacitySolid_titanium_custom():
    # First test case with customized CAS and literature data
    # alpha
    obj = HeatCapacitySolid(CASRN="2099571000-00-0")
    assert_close(obj(300), 25.809979690958304, rtol=.005)

    # beta
    obj = HeatCapacitySolid(CASRN="2099555000-00-0")
    assert_close(obj(1800), 35.667417599655764, rtol=.005)

    # Test we can add a method that combines them
    obj = HeatCapacitySolid(CASRN="7440-32-6")
    a, b = 'Fit 2023 alpha titanium', 'Fit 2023 beta titanium'
    obj.add_piecewise_method('auto', method_names=[a,b], T_ranges=[*obj.T_limits[a], obj.T_limits[b][1]])
    assert_close(obj(1500), 31.89694904655883)
    assert_close(obj(200), 22.149215492789644)

    # Did we break storing and representation?
    assert HeatCapacitySolid.from_json(obj.as_json()) == obj
    assert eval(str(obj)) == obj

@pytest.mark.meta_T_dept
def test_HeatCapacityGas_stable_polynomial_parameters():
    coeffs = [-1.1807560231661693, 1.0707500453237926, 6.219226796524199, -5.825940187155626, -13.479685202800221, 12.536206919506746, 16.713022858280983, -14.805461693468148, -13.840786121365808, 11.753575516231718, 7.020730111250113, -5.815540568906596, -2.001592044472603, 0.9210441915058972, 1.6279658993728698, -1.0508623065949019, -0.2536958793947375, 1.1354682714079252, -1.3567430363825075, 0.3415188644466688, 1.604997313795647, -2.26022568959622, -1.62374341299051, 10.875220288166474, 42.85532802412628]
    Tmin, Tmax = 251.165, 2000.0
    kwargs = {'stable_polynomial_parameters': {'test': {'coeffs': coeffs,'Tmin': Tmin, 'Tmax': Tmax  }}}
    obj = HeatCapacityGas(**kwargs)
    assert_close(obj(300), 33.59562103706877, rtol=1e-13)
    assert_close(obj.calculate_integral(300, 400, 'test'), 3389.544332887159, rtol=1e-13)

    # No real way to force check that quad isn't used but this at least checks the results
    assert 'int_coeffs' in obj.correlations['test'][3] 
    expect_coeffs = [-41.298949195476155, 39.0117740732049, 236.44351075433477, -231.5592751637343, -561.2796493247411, 548.0939357018894, 769.1662981674165, -719.2308222415658, -711.9191528399639, 642.3457574352843, 409.2699514702697, -363.22931752942026, -134.6328547344325, 67.11476327717565, 129.41107925589787, -91.88923909769476, -24.648457402294206, 124.10916590172992, -169.47997914514303, 49.77167860871583, 280.687547727181, -494.09522423312563, -473.27655194287644, 4754.741468163904, 37473.448792536445, 0.0]
    assert_close1d(obj.correlations['test'][3]['int_coeffs'], expect_coeffs)

@pytest.mark.meta_T_dept
def test_HeatCapacityGas_stable_polynomial_parameters_with_entropy():
    coeffs_num = [-1.1807560231661693, 1.0707500453237926, 6.219226796524199, -5.825940187155626, -13.479685202800221, 12.536206919506746, 16.713022858280983, -14.805461693468148, -13.840786121365808, 11.753575516231718, 7.020730111250113, -5.815540568906596, -2.001592044472603, 0.9210441915058972, 1.6279658993728698, -1.0508623065949019, -0.2536958793947375, 1.1354682714079252, -1.3567430363825075, 0.3415188644466688, 1.604997313795647, -2.26022568959622, -1.62374341299051, 10.875220288166474, 42.85532802412628]
    Tmin, Tmax = 251.165, 2000.0
    kwargs = {'stable_polynomial_parameters': {'test': {'coeffs': coeffs_num,'Tmin': Tmin, 'Tmax': Tmax  }}}
    obj = HeatCapacityGas(extrapolation="linear",**kwargs)
    dSs = []
    dHs = []
    pairs = [(252, 400), (252, 1000), (252, 2000), (Tmin, Tmax), (1, 10000), (1, 300), (1000, 10000)]
    dSs_expect = [15.587741399855446, 49.53701171038426, 81.56763739024905, 81.67858973242306, 377.4045671786281, 187.97907173107453, 145.72969875093]

    for (T1, T2), dS_expect in zip(pairs, dSs_expect):
        dS = obj.T_dependent_property_integral_over_T(T1, T2)
        # print(T1, T2, dS/dS_expect, dS_expect, dS)
        dSs.append(dS)

    for T1, T2 in pairs:
        dH = obj.T_dependent_property_integral(T1, T2)
        dHs.append(dH)
    dHs_expect = [4997.762485980464, 27546.16190448647, 74335.86952819106, 74363.78317065322, 701770.1907885703, 9924.44653308318, 665907.8005040939]
    assert_close1d(dHs, dHs_expect, rtol=1e-13)

    # Essentially what writting this test discovered is that there kind of is a numerical problem with so many parameters;
    # whether the parameters are converted to stable poly form or not their calculation is imprecise.
    # With more math work calculating them without rounding errors is probably possible. However, it will be slow
    # It is already slow!!
    # See test_HeatCapacityGas_stable_polynomial_parameters_with_entropy_fixed for how the situation is to be handled.
    assert_close1d(dSs, dSs_expect, rtol=2e-6)



    assert_close(obj.correlations['test'][3]['int_T_log_coeff'], 33.94307866530403, rtol=1e-13)
    assert_close1d(obj.correlations['test'][3]['int_T_coeffs'], 
        [-0.04919816763192373, 0.11263751330617944, 0.131110452463151, -0.45423261676208426, -0.06004401011932714, 0.7411591719219359, -0.0785488258334226, -0.7638508616073523, 0.17966145346872509, 0.5368869720259681, -0.23898457409814, -0.11605650605633855, -0.004958115052431822, 0.09069587616249919, 0.03437954094260931, -0.1659396644681692, 0.2085834185127169, -0.14463942032307386, -0.00890484161209315, 0.08206677765701897, 0.2691963085962925, -1.2154266047764395, 1.5349439248270755, 6.923550077860796, 3.225469171838455],
        rtol=4e-3)


@pytest.mark.meta_T_dept
def test_HeatCapacityGas_stable_polynomial_parameters_with_entropy_fixed():
    coeffs_num = [-1.1807560231661693, 1.0707500453237926, 6.219226796524199, -5.825940187155626, -13.479685202800221, 12.536206919506746, 16.713022858280983, -14.805461693468148, -13.840786121365808, 11.753575516231718, 7.020730111250113, -5.815540568906596, -2.001592044472603, 0.9210441915058972, 1.6279658993728698, -1.0508623065949019, -0.2536958793947375, 1.1354682714079252, -1.3567430363825075, 0.3415188644466688, 1.604997313795647, -2.26022568959622, -1.62374341299051, 10.875220288166474, 42.85532802412628]
    Tmin, Tmax = 251.165, 2000.0

    int_T_coeffs = [-0.04919816763192372, 0.11263751330617996, 0.13111045246317343, -0.45423261676182597, -0.060044010117414455, 0.7411591719418316, -0.07854882531076747, -0.7638508595972384, 0.17966146925345727, 0.5368870307001614, -0.238984435878899, -0.1160558677189392, -0.004958654675811305, 0.09069452297160363, 0.034376716486728694, -0.16593023302511933, 0.20857847967437174, -0.1446358723821105, -0.008913096668590397, 0.08207169354184629, 0.26919218469883643, -1.215427392470841, 1.5349428351610082, 6.923550076265145, 3.2254691726042486]
    int_T_log_coeff = 33.943078665304306
    kwargs = {'stable_polynomial_parameters': {'test': {'coeffs': coeffs_num,'Tmin': Tmin, 'Tmax': Tmax,
                                                    'int_T_coeffs': int_T_coeffs, 'int_T_log_coeff': int_T_log_coeff}}}
    obj = HeatCapacityGas(extrapolation="linear",**kwargs)
    assert_close(obj.calculate_integral_over_T(300, 400, 'test'), 9.746526386096955, rtol=1e-13)
    dSs = []
    dHs = []
    pairs = [(252, 400), (252, 1000), (252, 2000), (Tmin, Tmax), (1, 10000), (1, 300), (1000, 10000)]
    dSs_expect = [15.587741399857379, 49.53701171701343, 81.56762248012046, 81.67857482229451, 377.4045522684995, 187.97907173107427, 145.72968383417225]
    dHs_expect = [4997.762485980464, 27546.16190448647, 74335.86952819106, 74363.78317065322, 701770.1907885703, 9924.44653308318, 665907.8005040939]

    for (T1, T2), dS_expect in zip(pairs, dSs_expect):
        dS = obj.T_dependent_property_integral_over_T(T1, T2)
        print(T1, T2, dS/dS_expect, dS_expect, dS)
        dSs.append(dS)

    for T1, T2 in pairs:
        dH = obj.T_dependent_property_integral(T1, T2)
        dHs.append(dH)

    assert_close1d(dHs, dHs_expect, rtol=1e-13)
    assert_close1d(dSs, dSs_expect, rtol=1e-14)


@pytest.mark.meta_T_dept
def test_HeatCapacityGas_UNARY_calphad():
    # silver
    obj = HeatCapacitySolid(CASRN='7440-22-4')
    obj.method = 'UNARY'
    # obj.T_dependent
    assert_close(obj(400), 25.81330053581182, rtol=1e-7)
    assert_close(obj(2500),obj(2600), rtol=1e-4)
    assert_close(obj(2000),obj(2600), rtol=1e-3)
    assert_close(obj(2000),obj(3000), rtol=2e-3)
    Tmin, Tmax = 300, 800

    assert 'int_T_coeffs' in obj.correlations['UNARY'][3]
    assert 'int_T_log_coeff' in  obj.correlations['UNARY'][3]

    assert_close(obj.calculate_integral_over_T(Tmin, Tmax, 'UNARY'), 25.966064453125)
    assert_close(obj.calculate_integral(Tmin, Tmax, 'UNARY'), 13348.922120498239)

    obj = HeatCapacityLiquid(CASRN='7440-22-4')
    obj.method = 'UNARY'
    assert_close(obj.calculate_integral_over_T(Tmin, Tmax, 'UNARY'), 25.92645263671875)
    assert_close(obj.calculate_integral(Tmin, Tmax, 'UNARY'), 13343.854184041487)
    assert_close(obj(500), 26.323860726469874)


    assert 'int_T_coeffs' in obj.correlations['UNARY'][3]
    assert 'int_T_log_coeff' in  obj.correlations['UNARY'][3]

@pytest.mark.meta_T_dept
def test_HeatCapacity_stable_polynomial_water():
    obj = HeatCapacityGas(CASRN='7732-18-5')
    obj.method = 'HEOS_FIT'
    assert 'int_T_coeffs' in obj.correlations['HEOS_FIT'][3]
    assert 'int_T_log_coeff' in  obj.correlations['HEOS_FIT'][3]
    Tmin, Tmax = 300, 800

    assert_close(obj.calculate_integral_over_T(Tmin, Tmax, 'HEOS_FIT'), 34.78301770087086)
    assert_close(obj.calculate_integral(Tmin, Tmax, 'HEOS_FIT'), 17940.09005349636)


    obj = HeatCapacityLiquid(CASRN='7732-18-5')
    obj.method = 'HEOS_FIT'
    assert 'int_T_coeffs' in obj.correlations['HEOS_FIT'][3]
    assert 'int_T_log_coeff' in  obj.correlations['HEOS_FIT'][3]

    Tmin, Tmax = 300, 550
    assert_close(obj.calculate_integral_over_T(Tmin, Tmax, 'HEOS_FIT'), 47.9072265625)
    assert_close(obj.calculate_integral(Tmin, Tmax, 'HEOS_FIT'), 19944.787214230466)

@pytest.mark.meta_T_dept
def test_JANAF_fit_carbon_exists():
    obj = HeatCapacitySolid(CASRN="7440-44-0")
    assert obj.method == 'JANAF_FIT'


@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.meta_T_dept
def test_locked_integral():
    obj = HeatCapacityGas(load_data=False, CASRN="7732-18-5", similarity_variable=0.16652530518537598, MW=18.01528,
                          extrapolation="linear", method="POLY_FIT",
                          poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))
    T1s = [1929.096, 1140.749, 3458.609, 2136.875, 1291.477, 603.78, 2566.611, 1815.559, 2487.887, 189.325]
    T2s = [994.916, 2341.203, 2094.687, 3626.799, 2680.001, 3435.09, 1105.486, 71.664, 1066.234, 3191.909]
    for T1, T2 in zip(T1s, T2s):
        quad_ans = quad(obj, T1, T2)[0]
        analytical_ans = obj.T_dependent_property_integral(T1, T2)
        assert_close(quad_ans, analytical_ans, rtol=1e-6)


@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.meta_T_dept
def test_locked_integral_over_T():
    obj = HeatCapacityGas(load_data=False, CASRN="7732-18-5", similarity_variable=0.16652530518537598, MW=18.01528, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))

    def to_int(T):
        return obj(T)/T
    T1s = [1929.096, 1140.749, 3458.609, 2136.875, 1291.477, 603.78, 2566.611, 1815.559, 2487.887, 189.325]
    T2s = [994.916, 2341.203, 2094.687, 3626.799, 2680.001, 3435.09, 1105.486, 71.664, 1066.234, 3191.909]
    for T1, T2 in zip(T1s, T2s):
        quad_ans = quad(to_int, T1, T2)[0]
        analytical_ans = obj.T_dependent_property_integral_over_T(T1, T2)
        assert_close(quad_ans, analytical_ans, rtol=1e-5)

@pytest.mark.meta_T_dept
def test_heat_capacity_interp1d_removed_extrapolation_method_compatibility():
    EtOH = HeatCapacityGas(CASRN='64-17-5', similarity_variable=0.1953615, MW=46.06844, extrapolation='interp1d', method='POLING_POLY')
    assert_close(EtOH(EtOH.Tmin - 10), 37.421995400002054)
    assert_close(EtOH(EtOH.Tmax + 10), 142.81153700770574)
    EtOH = HeatCapacityGas(CASRN='64-17-5', similarity_variable=0.1953615, MW=46.06844, extrapolation='interp1d|interp1d', method='POLING_POLY')
    assert_close(EtOH(EtOH.Tmin - 10), 37.421995400002054)
    assert_close(EtOH(EtOH.Tmax + 10), 142.81153700770574)    


@pytest.mark.meta_T_dept
def test_as_method_kwargs_tabular():
    obj = HeatCapacitySolid()
    Ts = [200, 300, 400, 500, 600]
    Cps = [12.965044960703908, 20.206353934945987, 28.261467986645872, 37.14292010552292, 46.85389719453655]
    obj.add_tabular_data(Ts=Ts, properties=Cps, name='stuff')
    obj.add_tabular_data(Ts=Ts, properties=[v*2 for v in Cps], name='stuff2')
    obj2 = HeatCapacitySolid(**obj.as_method_kwargs())

    assert_close(obj(321), obj2(321))
    obj.method = 'stuff'

    assert_close(2*obj(321), obj2(321))

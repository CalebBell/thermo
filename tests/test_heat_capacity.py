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

import numpy as np
import pytest
from chemicals.utils import ws_to_zs
from thermo.heat_capacity import *
from thermo.heat_capacity import TRCIG, POLING_POLY, CRCSTD, COOLPROP, POLING_CONST, VDI_TABULAR, LASTOVKA_SHAW, ROWLINSON_BONDI, ZABRANSKY_QUASIPOLYNOMIAL_C, CRCSTD, ROWLINSON_POLING, POLING_CONST, ZABRANSKY_SPLINE_SAT, DADGOSTAR_SHAW, COOLPROP, ZABRANSKY_SPLINE_C, VDI_TABULAR
from random import uniform
from math import *
from fluids.numerics import linspace, logspace, NotBoundedError, assert_close, assert_close1d
from thermo.chemical import lock_properties, Chemical
from scipy.integrate import quad
from thermo.coolprop import has_CoolProp


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
                     LASTOVKA_SHAW: 71.07236200126606}

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

    assert False == EtOH.test_property_validity(-1)
    assert False == EtOH.test_property_validity(1.01E4)


    Ts = [200, 250, 300, 400, 450]
    props = [1.2, 1.3, 1.4, 1.5, 1.6]
    EtOH.add_tabular_data(Ts=Ts, properties=props, name='test_set')
    assert_close(1.35441088517, EtOH.T_dependent_property(275), rtol=2E-4)

    assert None == EtOH.T_dependent_property(5000)

    new = HeatCapacityGas.from_json(EtOH.as_json())
    assert new == EtOH

    # Case where the limits were nans
    obj = HeatCapacityGas(CASRN='7440-37-1', MW=39.948, similarity_variable=0.025032542304996495)
    assert not isnan(obj.Tmax)
    assert not isnan(obj.Tmin)
    assert not isnan(obj.POLING_Tmin)
    assert not isnan(obj.POLING_Tmax)

    new = HeatCapacityGas.from_json(obj.as_json())
    assert new == obj

@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_HeatCapacityGas_CoolProp():
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


@pytest.mark.meta_T_dept
def test_HeatCapacityGas_linear_extrapolation():
    CpObj = HeatCapacityGas(CASRN='67-56-1', extrapolation='linear')
    CpObj.method = TRCIG
    assert_close(CpObj.T_dependent_property(CpObj.TRCIG_Tmax),
                 CpObj.T_dependent_property(CpObj.TRCIG_Tmax-1e-6))
    assert_close(CpObj.T_dependent_property(CpObj.TRCIG_Tmax),
                 CpObj.T_dependent_property(CpObj.TRCIG_Tmax+1e-6))

    assert_close(CpObj.T_dependent_property(CpObj.TRCIG_Tmin),
                 CpObj.T_dependent_property(CpObj.TRCIG_Tmin-1e-6))
    assert_close(CpObj.T_dependent_property(CpObj.TRCIG_Tmin),
                 CpObj.T_dependent_property(CpObj.TRCIG_Tmin+1e-6))



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

    Cps_calc = []
    for i in NaCl.all_methods:
        NaCl.method = i
        Cps_calc.append(NaCl.T_dependent_property(298.15))

    Cps_exp = [50.38469032, 50.5, 20.065072074682337]
    assert_close1d(sorted(Cps_calc), sorted(Cps_exp))

    NaCl.extrapolation = None
    for i in NaCl.all_methods:
        NaCl.method = i
        assert NaCl.T_dependent_property(20000) is None

    with pytest.raises(Exception):
        NaCl.test_method_validity('BADMETHOD', 300)

    assert False == NaCl.test_property_validity(-1)
    assert False == NaCl.test_property_validity(1.01E5)

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
def test_HeatCapacitySolid_integrals():
    from thermo.heat_capacity import LASTOVKA_S, PERRY151, CRCSTD
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

    assert False == tol.test_property_validity(-1)
    assert False == tol.test_property_validity(1.01E5)
    assert True == tol.test_property_validity(100)



    propylbenzene = HeatCapacityLiquid(MW=120.19158, CASRN='103-65-1', Tc=638.35)

    Cpl_calc = []
    for i in propylbenzene.all_methods:
        propylbenzene.method = i
        Cpl_calc.append(propylbenzene.T_dependent_property(298.15))

    Cpls = [214.6499551694668, 214.69679325320664, 214.7, 214.71]
    assert_close1d(sorted(Cpl_calc), sorted(Cpls))

    new = HeatCapacityLiquid.from_json(propylbenzene.as_json())
    assert new == propylbenzene

    ctp = HeatCapacityLiquid(MW=118.58462, CASRN='96-43-5')

    Cpl_calc = []
    for i in ctp.all_methods:
        ctp.method = i
        Cpl_calc.append(ctp.T_dependent_property(250))

    Cpls = [134.1186737739494, 134.1496585096233]
    assert_close1d(sorted(Cpl_calc), sorted(Cpls))
    new = HeatCapacityLiquid.from_json(ctp.as_json())
    assert new == ctp

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
    from thermo.heat_capacity import (CRCSTD, COOLPROP, DADGOSTAR_SHAW,
                                      ROWLINSON_POLING, ROWLINSON_BONDI,
                                      ZABRANSKY_SPLINE,
                                      ZABRANSKY_QUASIPOLYNOMIAL,
                                      ZABRANSKY_SPLINE_SAT,
                                      ZABRANSKY_QUASIPOLYNOMIAL_SAT,
                                      ZABRANSKY_QUASIPOLYNOMIAL_C,
                                      ZABRANSKY_SPLINE_C)
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
def test_HeatCapacitySolidMixture():
    MWs = [107.8682, 195.078]
    ws = [0.95, 0.05]
    zs = ws_to_zs(ws=ws, MWs=MWs)
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
    from thermo.heat_capacity import HeatCapacityLiquidMixture, LINEAR

    HeatCapacityLiquids = [HeatCapacityLiquid(method='COOLPROP', CASRN="7732-18-5", MW=18.01528, similarity_variable=0.16652530518537598, Tc=647.14, omega=0.344, extrapolation="linear"),
                           HeatCapacityLiquid(CASRN="7647-14-5", MW=58.44277, similarity_variable=0.034221512772238546, Tc=3400.0, omega=0.1894, extrapolation="linear", method="POLY_FIT", poly_fit=(1077.15, 3400.0, [-1.7104845836996866e-36, 3.0186209409101436e-32, -2.2964525212600158e-28, 9.82884046707168e-25, -2.585879179539946e-21, 4.276734025134063e-18, -0.00016783912672163547, 0.1862851719065294, -12.936804011905963]))]

    obj = HeatCapacityLiquidMixture(MWs=[18.01528, 58.44277], CASs=['7732-18-5', '7647-14-5'], HeatCapacityLiquids=HeatCapacityLiquids)

    Cp = obj(T=301.5, P=101325.0, ws=[.9, .1])
    assert_close(Cp, 72.29666542719279, rtol=1e-5)
    ws = [.9, .1]
    zs = ws_to_zs(ws=ws, MWs=[18.01528, 58.44277])

    Cp = obj.calculate(T=301.5, P=101325.0, zs=zs, ws=[.9, .1], method=LINEAR)
    assert_close(Cp, 77.08377446120679, rtol=1e-7)

@pytest.mark.meta_T_dept
def test_HeatCapacityLiquidMixture():
    ws = [.9, .1]
    MWs = [92.13842, 142.28168]
    zs = ws_to_zs(ws=ws, MWs=MWs)
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


@pytest.mark.slow
@pytest.mark.fuzz
def test_locked_integral():
    obj = HeatCapacityGas(load_data=False, CASRN="7732-18-5", similarity_variable=0.16652530518537598, MW=18.01528, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))

    def to_int(T):
        return obj.calculate(T, 'POLY_FIT')
    for i in range(10):
        T1 = uniform(0, 4000)
        T2 = uniform(0, 4000)
        quad_ans = quad(to_int, T1, T2)[0]
        analytical_ans = obj.calculate_integral(T1, T2, "POLY_FIT")
        assert_close(quad_ans, analytical_ans, rtol=1e-6)


@pytest.mark.slow
@pytest.mark.fuzz
def test_locked_integral_over_T():
    obj = HeatCapacityGas(load_data=False, CASRN="7732-18-5", similarity_variable=0.16652530518537598, MW=18.01528, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))

    def to_int(T):
        return obj.calculate(T, 'POLY_FIT')/T
    for i in range(10):
        T1 = uniform(0, 4000)
        T2 = uniform(0, 4000)
        quad_ans = quad(to_int, T1, T2)[0]
        analytical_ans = obj.calculate_integral_over_T(T1, T2, "POLY_FIT")
        assert_close(quad_ans, analytical_ans, rtol=1e-5)

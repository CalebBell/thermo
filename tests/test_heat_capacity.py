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

from numpy.testing import assert_allclose
import numpy as np
import pytest
from thermo.heat_capacity import *
from thermo.heat_capacity import TRCIG, POLING, CRCSTD, COOLPROP, POLING_CONST, VDI_TABULAR


def test_heat_capacity_CSP():
    # Example is for cis-2-butene at 350K from Poling. It is not consistent with
    # the example presented. The error is in the main expressesion.
    Cp1 = Rowlinson_Poling(350.0, 435.5, 0.203, 91.21)
    Cp2 = Rowlinson_Poling(373.28, 535.55, 0.323, 119.342)
    assert_allclose([Cp1, Cp2], [143.80194441498296, 177.62600344957252])

    # Example in VDI Heat Atlas, Example 11 in VDI D1 p 139.
    # Also formerly in ChemSep. Also in COCO.
    Cp = Rowlinson_Bondi(373.28, 535.55, 0.323, 119.342)
    assert_allclose(Cp, 175.39760730048116)


def test_solid_models():
    # Point at 300K from fig 5 for polyethylene, as read from the pixels of the
    # graph. 1684 was the best the pixels could give.
    Cp = Lastovka_solid(300, 0.2139)
    assert_allclose(Cp, 1682.063629222013)
    
    dH = Lastovka_solid_integral(300, 0.2139)
    assert_allclose(dH, 283246.1242170376)
    
    dS = Lastovka_solid_integral_over_T(300, 0.2139)
    assert_allclose(dS, 1947.553552666818)

    Cp = Dadgostar_Shaw(355.6, 0.139)
    assert_allclose(Cp, 1802.5291501191516)

    # Data in article:
    alphas = [0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.154, 0.154, 0.154, 0.154, 0.154, 0.135, 0.135, 0.135, 0.135, 0.135, 0.144, 0.144, 0.144, 0.144]
    Ts = [196.42, 208.55, 220.61, 229.57, 238.45, 250.18, 261.78, 270.39, 276.2, 281.74, 287.37, 295.68, 308.45, 317.91, 330.38, 336.54, 342.66, 223.2, 227.5, 244.5, 275, 278.2, 283.3, 289.4, 295, 318.15, 333.15, 348.15, 363.15, 373.15, 246.73, 249.91, 257.54, 276.24, 298.54, 495, 497, 498.15, 500, 502, 401.61, 404.6, 407.59, 410.57]
    expecteds = [1775, 1821, 1867, 1901, 1934, 1978, 2022, 2054, 2076, 2097, 2118, 2149, 2196, 2231, 2277, 2299, 2322, 1849, 1866, 1932, 2050, 2062, 2082, 2105, 2126, 2213, 2269, 2325, 2380, 2416, 1485, 1500, 1535, 1618, 1711, 2114, 2117, 2119, 2122, 2126, 1995, 2003, 2012, 2020]

    # I guess 5 J isn't bad! Could try recalculating alphas as well.
    Calculated = [Dadgostar_Shaw(T, alpha) for T, alpha in zip(Ts, alphas)]
    assert_allclose(expecteds, Calculated, atol=5)


def test_Zabransky_quasi_polynomial():
    Cp = Zabransky_quasi_polynomial(330, 591.79, -3.12743, 0.0857315, 13.7282, 1.28971, 6.42297, 4.10989)
    assert_allclose(Cp, 165.4728226923247)

    H2 = Zabransky_quasi_polynomial_integral(300, 591.79, -3.12743, 0.0857315, 13.7282, 1.28971, 6.42297, 4.10989)
    H1 = Zabransky_quasi_polynomial_integral(200, 591.79, -3.12743, 0.0857315, 13.7282, 1.28971, 6.42297, 4.10989)
    assert_allclose(H2 - H1, 14662.026406892926)
    
    S2 = Zabransky_quasi_polynomial_integral_over_T(300, 591.79, -3.12743, 0.0857315, 13.7282, 1.28971, 6.42297, 4.10989)
    S1 = Zabransky_quasi_polynomial_integral_over_T(200, 591.79, -3.12743, 0.0857315, 13.7282, 1.28971, 6.42297, 4.10989)
    assert_allclose(S2-S1, 59.169972919442074) # result from quadrature, not the integral


def test_Zabransky_cubic():
    Cp = Zabransky_cubic(298.15, 20.9634, -10.1344, 2.8253, -0.256738)
    assert_allclose(Cp, 75.31462591538555)
    
    H0 = Zabransky_cubic_integral(298.15, 20.9634, -10.1344, 2.8253, -0.256738)
    assert_allclose(H0, 31051.679845520584)
    
    S0 = Zabransky_cubic_integral_over_T(298.15, 20.9634, -10.1344, 2.8253,  -0.256738)
    assert_allclose(S0, 24.73245695987246)
    


def test_Lastovka_Shaw():
    # C64H52S2 (M = 885.2 alpha = 0.1333 mol/g
    # From figure 22, part b
    # Some examples didn't match so well; were the coefficients rounded?
    assert_allclose(Lastovka_Shaw(1000.0, 0.1333), 2467.113309084757)

    # Same, but try the correlation for cyclic aliphatic compounds
    assert_allclose(Lastovka_Shaw(1000.0, 0.1333, cyclic_aliphatic=True), 2187.36187944884)


def test_Lastovka_Shaw_integral():
    # C64H52S2 (M = 885.2 alpha = 0.1333 mol/g
    # From figure 22, part b
    # Some examples didn't match so well; were the coefficients rounded?
    assert_allclose(Lastovka_Shaw_integral(1000.0, 0.1333), 6615282.290516732)

    # Same, but try the correlation for cyclic aliphatic compounds
    assert_allclose(Lastovka_Shaw_integral(1000.0, 0.1333, cyclic_aliphatic=True), 6335530.860880815)

def test_Lastovka_Shaw_integral_over_T():
    dS = Lastovka_Shaw_integral_over_T(300.0, 0.1333)
    assert_allclose(dS, 3609.791928945323)

    dS = Lastovka_Shaw_integral_over_T(1000.0, 0.1333, cyclic_aliphatic=True)
    assert_allclose(dS, 3790.4489380423597)


def test_CRC_standard_data():
    tots_calc = [CRC_standard_data[i].abs().sum() for i in [u'Hfc', u'Gfc', u'Sc', u'Cpc', u'Hfl', u'Gfl', u'Sfl', 'Cpl', u'Hfg', u'Gfg', u'Sfg', u'Cpg']]
    tots = [628580900.0, 306298700.0, 68541.800000000003, 56554.400000000001, 265782700.0, 23685900.0, 61274.0, 88464.399999999994, 392946600.0, 121270700.0, 141558.29999999999, 33903.300000000003]
    assert_allclose(tots_calc, tots)

    assert CRC_standard_data.index.is_unique
    assert CRC_standard_data.shape == (2470, 13)

def test_Poling_data():
    tots_calc = [Poling_data[i].abs().sum() for i in ['Tmin', 'Tmax', 'a0', 'a1', 'a2', 'a3', 'a4', 'Cpg', 'Cpl']]
    tots = [40966.0, 301000.0, 1394.7919999999999, 10.312580799999999, 0.024578948000000003, 3.1149672999999997e-05, 1.2539125599999999e-08, 43530.690000000002, 50002.459999999999]
    assert_allclose(tots_calc, tots)


    assert Poling_data.index.is_unique
    assert Poling_data.shape == (368, 10)


def test_TRC_gas_data():
    tots_calc = [TRC_gas_data[i].abs().sum() for i in ['Tmin', 'Tmax', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'I', 'J', 'Hf']]
    tots = [365114, 3142000, 7794.1999999999998, 24465781000, 1056180, 133537.068, 67639.309000000008, 156121050000, 387884, 212320, 967467.89999999991, 30371.91, 495689880.0]
    assert_allclose(tots_calc, tots)

    assert TRC_gas_data.index.is_unique
    assert TRC_gas_data.shape == (1961, 14)

def test_TRC_gas():
    Cps = [TRCCp(T, 4.0, 7.65E5, 720., 3.565, -0.052, -1.55E6, 52., 201.) for T in [150, 300]]
    assert_allclose(Cps, [35.584319834110346, 42.06525682312236])


def test_TRCCp_integral():
    dH = TRCCp_integral(298.15, 4.0, 7.65E5, 720., 3.565, -0.052, -1.55E6, 52., 201., 1.2)
    assert_allclose(dH, 10802.532600592816)
    dH = TRCCp_integral(150, 4.0, 7.65E5, 720., 3.565, -0.052, -1.55E6, 52., 201., 1.2)
    assert_allclose(dH, 5071.355751575949)


def test_TRCCp_integral_over_T():
    coeffs = [4.0, 124000, 245, 50.538999999999994, -49.468999999999994, 220440000, 560, 78]
    dS = TRCCp_integral_over_T(300, *coeffs) - TRCCp_integral_over_T(200, *coeffs)
    assert_allclose(dS, 23.44278146529652)
    
    
@pytest.mark.meta_T_dept
def test_HeatCapacityGas():
    EtOH = HeatCapacityGas(CASRN='64-17-5', similarity_variable=0.1953615, MW=46.06844)
    methods = list(EtOH.all_methods)
    methods.remove(VDI_TABULAR)
    Cps_calc = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(305))[1] for i in methods]
    assert_allclose(sorted(Cps_calc), sorted([66.35085001015844, 66.40063819791762, 66.25918325111196, 71.07236200126606, 65.6, 65.21]))

    # VDI interpolation, treat separately due to change in behavior of scipy in 0.19
    assert_allclose(EtOH.calculate(305, VDI_TABULAR), 74.6763493522965, rtol=1E-4)



    EtOH.tabular_extrapolation_permitted = False
    assert [None]*6 == [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(5000))[1] for i in [TRCIG, POLING, CRCSTD, COOLPROP, POLING_CONST, VDI_TABULAR]]

    with pytest.raises(Exception):
        EtOH.test_method_validity('BADMETHOD', 300)

    assert False == EtOH.test_property_validity(-1)
    assert False == EtOH.test_property_validity(1.01E4)


    Ts = [200, 250, 300, 400, 450]
    props = [1.2, 1.3, 1.4, 1.5, 1.6]
    EtOH.set_tabular_data(Ts=Ts, properties=props, name='test_set')
    EtOH.forced = True
    assert_allclose(1.35441088517, EtOH.T_dependent_property(275), rtol=2E-4)

    assert None == EtOH.T_dependent_property(5000)
#test_HeatCapacityGas()

@pytest.mark.meta_T_dept
def test_HeatCapacityGas_integrals():
    # Enthalpy integrals
    EtOH = HeatCapacityGas(CASRN='64-17-5', similarity_variable=0.1953615, MW=46.06844)
    dH1 = EtOH.calculate_integral(200, 300, 'TRC Thermodynamics of Organic Compounds in the Gas State (1994)')
    assert_allclose(dH1, 5828.903671654116)

    dH2 = EtOH.calculate_integral(200, 300, 'Poling et al. (2001)')
    assert_allclose(dH2, 5851.196044907861)
    
    dH3 = EtOH.calculate_integral(200, 300, 'Poling et al. (2001) constant')
    assert_allclose(dH3, 6520.999999999999)
    
    dH4 = EtOH.calculate_integral(200, 300, 'CRC Standard Thermodynamic Properties of Chemical Substances')
    assert_allclose(dH4, 6559.999999999999)
    
    dH4 = EtOH.calculate_integral(200, 300,'Lastovka and Shaw (2013)')
    assert_allclose(dH4, 6183.016942750677)

    dH5 = EtOH.calculate_integral(200, 300,'CoolProp')
    assert_allclose(dH5, 5838.118293585357)
    
    dH = EtOH.calculate_integral(200, 300, 'VDI Heat Atlas')
    assert_allclose(dH, 6610.821140000002)
    
    # Entropy integrals
    dS = EtOH.calculate_integral_over_T(200, 300, 'Poling et al. (2001)')
    assert_allclose(dS, 23.53409951536522)
        
    dS = EtOH.calculate_integral_over_T(200, 300, 'Poling et al. (2001) constant')
    assert_allclose(dS, 26.440379699733395)

    dS = EtOH.calculate_integral_over_T(200, 300, 'TRC Thermodynamics of Organic Compounds in the Gas State (1994)')
    assert_allclose(dS, 23.442781465296523)
    
    dS = EtOH.calculate_integral_over_T(200, 300, 'CRC Standard Thermodynamic Properties of Chemical Substances')
    assert_allclose(dS, 26.59851109189558)
    
    dS =  EtOH.calculate_integral_over_T(200, 300, 'CoolProp')
    assert_allclose(dS, 23.487556909586853)
    
    dS = EtOH.calculate_integral_over_T(200, 300, 'Lastovka and Shaw (2013)')
    assert_allclose(dS, 24.86700348570948)
    
    dS = EtOH.calculate_integral_over_T(200, 300, 'VDI Heat Atlas')
    assert_allclose(dS, 26.590569427910076)


@pytest.mark.meta_T_dept
def test_HeatCapacitySolid():
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Cps_calc =  [(NaCl.set_user_methods(i, forced=True), NaCl.T_dependent_property(298.15))[1] for i in NaCl.all_methods]
    Cps_exp = [50.38469032, 50.5, 20.065072074682337]
    assert_allclose(sorted(Cps_calc), sorted(Cps_exp))

    assert [None]*3 == [(NaCl.set_user_methods(i, forced=True), NaCl.T_dependent_property(20000))[1] for i in NaCl.all_methods]

    with pytest.raises(Exception):
        NaCl.test_method_validity('BADMETHOD', 300)

    assert False == NaCl.test_property_validity(-1)
    assert False == NaCl.test_property_validity(1.01E5)

    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Ts = [200, 300, 400, 500, 600]
    Cps = [12.965044960703908, 20.206353934945987, 28.261467986645872, 37.14292010552292, 46.85389719453655]
    NaCl.set_tabular_data(Ts=Ts, properties=Cps, name='stuff')
    NaCl.forced = True
    assert_allclose(NaCl.T_dependent_property(275), 18.320355898506502, rtol=1E-5)

    NaCl.tabular_extrapolation_permitted = False
    assert None == NaCl.T_dependent_property(601)


@pytest.mark.meta_T_dept
def test_HeatCapacitySolid_integrals():
    from thermo.heat_capacity import LASTOVKA_S, PERRY151, CRCSTD
    # Enthalpy integrals
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    dH1 = NaCl.calculate_integral(100, 150, LASTOVKA_S)
    assert_allclose(dH1, 401.58058175282446)
    
    dH2 = NaCl.calculate_integral(100, 150, CRCSTD)
    assert_allclose(dH2, 2525.0) # 50*50.5
    
    dH3 = NaCl.calculate_integral(100, 150,  PERRY151)
    assert_allclose(dH3, 2367.097999999999)

    # Tabular integration - not great
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Ts = [200, 300, 400, 500, 600]
    Cps = [12.965044960703908, 20.206353934945987, 28.261467986645872, 37.14292010552292, 46.85389719453655]
    NaCl.set_tabular_data(Ts=Ts, properties=Cps, name='stuff')
    dH4 = NaCl.calculate_integral(200, 300, 'stuff')
    assert_allclose(dH4, 1651.8556007162392, rtol=1E-5)
    
    # Entropy integrals
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    dS1 = NaCl.calculate_integral_over_T(100, 150, LASTOVKA_S)
    assert_allclose(dS1, 3.213071341895563)
    
    dS2 = NaCl.calculate_integral_over_T(100, 150,  PERRY151)
    assert_allclose(dS2, 19.183508272982)
    
    dS3 = NaCl.calculate_integral_over_T(100, 150, CRCSTD)
    assert_allclose(dS3, 20.4759879594623)
    
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Ts = [200, 300, 400, 500, 600]
    Cps = [12.965044960703908, 20.206353934945987, 28.261467986645872, 37.14292010552292, 46.85389719453655]
    NaCl.set_tabular_data(Ts=Ts, properties=Cps, name='stuff')
    dS4 = NaCl.calculate_integral_over_T(100, 150, 'stuff')
    assert_allclose(dS4, 3.00533159156869)


@pytest.mark.meta_T_dept
def test_HeatCapacityLiquid():
    tol = HeatCapacityLiquid(CASRN='108-88-3', MW=92.13842, Tc=591.75, omega=0.257, Cpgm=115.30398669098454, similarity_variable=0.16279853724428964)
    Cpl_calc = [(tol.set_user_methods(i, forced=True), tol.T_dependent_property(330))[1] for i in tol.all_methods]
    Cpls = [165.4728226923247, 166.5239869108539, 166.52164399712314, 175.3439256239127, 166.71561127721478, 157.3, 165.4554033804999, 166.69807427725885, 157.29, 167.3380448453572]
    assert_allclose(sorted(Cpl_calc), sorted(Cpls), rtol=5e-6)

    assert [None]*10 == [(tol.set_user_methods(i, forced=True), tol.T_dependent_property(2000))[1] for i in tol.all_methods]

    with pytest.raises(Exception):
        tol.test_method_validity('BADMETHOD', 300)

    assert False == tol.test_property_validity(-1)
    assert False == tol.test_property_validity(1.01E5)
    assert True == tol.test_property_validity(100)



    propylbenzene = HeatCapacityLiquid(MW=120.19158, CASRN='103-65-1', Tc=638.35)
    Cpl_calc = [(propylbenzene.set_user_methods(i, forced=True), propylbenzene.T_dependent_property(298.15))[1] for i in propylbenzene.all_methods]
    Cpls = [214.696720482603, 214.71, 214.7, 214.6498824147387]
    assert_allclose(sorted(Cpl_calc), sorted(Cpls))

    ctp = HeatCapacityLiquid(MW=118.58462, CASRN='96-43-5')
    Cpl_calc = [(ctp.set_user_methods(i, forced=True), ctp.T_dependent_property(250))[1] for i in ctp.all_methods]
    Cpls = [134.1186283149712, 134.14961304014292]
    assert_allclose(sorted(Cpl_calc), sorted(Cpls))


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
    assert_allclose(dH, 15730)
        
    dH = tol.calculate_integral(200, 300, COOLPROP)
    assert_allclose(dH, 14501.714588188637)
    
    dH = tol.calculate_integral(200, 300, DADGOSTAR_SHAW)
    assert_allclose(dH, 14395.231307169146)
    
    dH = tol.calculate_integral(200, 300, ROWLINSON_POLING)
    assert_allclose(dH, 17332.445363748568)
    
    dH = tol.calculate_integral(200, 300, ROWLINSON_BONDI)
    assert_allclose(dH, 17161.365551776624)
    
    dH = tol.calculate_integral(200, 300, ZABRANSKY_SPLINE_C)
    assert_allclose(dH, 14588.045715211323)
    
    # Test over different coefficient sets
    dH = tol.calculate_integral(200, 500, ZABRANSKY_SPLINE_SAT)
    assert_allclose(dH, 52806.40487959729)
    
    dH = tol.calculate_integral(200, 300, ZABRANSKY_SPLINE_SAT)
    assert_allclose(dH, 14588.104262865752)
    
    dH = tol.calculate_integral(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_C)
    assert_allclose(dH, 14662.026406892926)
    
    dH = propylbenzene.calculate_integral(200, 300, ZABRANSKY_QUASIPOLYNOMIAL)
    assert_allclose(dH, 19863.93768123914)
    
    dH = propylbenzene.calculate_integral(200, 300, ZABRANSKY_SPLINE)
    assert_allclose(dH, 19865.17965271845)
    
    dH = ctp.calculate_integral(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_SAT)
    assert_allclose(dH, 13437.281657981055)
        
    # Entropy integrals
    dS = tol.calculate_integral_over_T(200, 300, CRCSTD)
    assert_allclose(dS, 63.779661505414275)
    
    dS = tol.calculate_integral_over_T(200, 300, COOLPROP)
    assert_allclose(dS, 58.50970500781979)
    
    dS = tol.calculate_integral_over_T(200, 300, DADGOSTAR_SHAW)
    assert_allclose(dS, 57.78686119989654)
    
    dS = tol.calculate_integral_over_T(200, 300, ROWLINSON_POLING)
    assert_allclose(dS, 70.42884850906293)
    
    dS = tol.calculate_integral_over_T(200, 300, ROWLINSON_BONDI)
    assert_allclose(dS, 69.73749349887285)
    
    dS = tol.calculate_integral_over_T(200, 300, ZABRANSKY_SPLINE_C)
    assert_allclose(dS, 58.866372687623326)
    
    dS = tol.calculate_integral_over_T(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_C)
    assert_allclose(dS, 59.169972919442074)
    
    dS = tol.calculate_integral_over_T(200, 300, ZABRANSKY_SPLINE_SAT)
    assert_allclose(dS, 58.866460402717365)
         
    dS = tol.calculate_integral_over_T(200, 500, ZABRANSKY_SPLINE_SAT)
    assert_allclose(dS, 154.94761329230238)
    
    dS = propylbenzene.calculate_integral_over_T(200, 300, ZABRANSKY_QUASIPOLYNOMIAL)
    assert_allclose(dS, 80.13490411695314)
    
    dS = propylbenzene.calculate_integral_over_T(200, 300, ZABRANSKY_SPLINE)
    assert_allclose(dS, 80.13634158499711)

    dS = ctp.calculate_integral_over_T(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_SAT)
    assert_allclose(dS, 54.34706623224253)

    
    
def test_ZABRANSKY_SPLINE():
    from thermo.heat_capacity import zabransky_dict_iso_s
    d = zabransky_dict_iso_s['7732-18-5']
    assert_allclose(d.calculate(645), 4521.766269344189)
    assert_allclose(d.calculate(400), 76.53792824292003)
    assert_allclose(d.calculate(250), 77.10824821982627)
    assert_allclose(d.Ts, [273.6, 380.0, 590.0, 635.0, 644.6])
    assert_allclose(d.coeff_sets, [[20.9634, -10.1344, 2.8253, -0.256738],
                                   [-22.0666, 23.8366, -6.11445, 0.52745],
                                   [-40151.8, 20428.8, -3464.58, 195.921],
                                   [-23803400.0, 11247200.0, -1771450.0, 93003.7]])
    
    
    # Test enthalpy integrals
    
    assert_allclose(d.calculate_integral(200, 270), 5505.732579532694)
    assert_allclose(d.calculate_integral(200, 280), 6264.125429184165)
    assert_allclose(d.calculate_integral(200, 380), 13817.509769449993)
    assert_allclose(d.calculate_integral(200, 380+1E-9), 13817.509769449993)
    assert_allclose(d.calculate_integral(300, 590-1E-9), 24309.128901680062)
    assert_allclose(d.calculate_integral(300, 590+1E-9), 24309.128901680062)
    assert_allclose(d.calculate_integral(200, 635-1E-9), 39698.84625418573)
    assert_allclose(d.calculate_integral(200, 635+1E-9), 39698.84625418573)
    assert_allclose(d.calculate_integral(200, 644.6), 76304.7781125316)
    assert_allclose(d.calculate_integral(200, 645), 78093.6635609874)
    
    # Same test cases, flipped around
    assert_allclose(d.calculate_integral(270, 200), -5505.732579532694)
    assert_allclose(d.calculate_integral(280, 200), -6264.125429184165)
    assert_allclose(d.calculate_integral(380, 200), -13817.509769449993)
    assert_allclose(d.calculate_integral(380+1E-9, 200), -13817.509769449993)
    assert_allclose(d.calculate_integral(590-1E-9, 300), -24309.128901680062)
    assert_allclose(d.calculate_integral(590+1E-9, 300), -24309.128901680062)
    assert_allclose(d.calculate_integral(635-1E-9, 200), -39698.84625418573)
    assert_allclose(d.calculate_integral(635+1E-9, 200), -39698.84625418573)
    assert_allclose(d.calculate_integral(644.6, 200), -76304.7781125316)
    assert_allclose(d.calculate_integral(645, 200), -78093.6635609874)

    # Test entropy integrals
    assert_allclose(d.calculate_integral_over_T(200, 270), 23.65403751138625)
    assert_allclose(d.calculate_integral_over_T(200, 280), 26.412172179347728)
    assert_allclose(d.calculate_integral_over_T(200, 380), 49.473623647253014)
    assert_allclose(d.calculate_integral_over_T(200, 380+1E-9), 49.473623647253014)
    assert_allclose(d.calculate_integral_over_T(300, 590-1E-9), 55.55836880929313)
    assert_allclose(d.calculate_integral_over_T(300, 590+1E-9), 55.55836880929313)
    assert_allclose(d.calculate_integral_over_T(200, 635-1E-9), 99.54361626259453)
    assert_allclose(d.calculate_integral_over_T(200, 635+1E-9), 99.54361626259453)
    assert_allclose(d.calculate_integral_over_T(200, 644.6), 156.74427000699035)
    assert_allclose(d.calculate_integral_over_T(200, 645), 159.51859302013582)
    
    # Same test cases, flipped around
    assert_allclose(d.calculate_integral_over_T(270, 200), -23.65403751138625)
    assert_allclose(d.calculate_integral_over_T(280, 200), -26.412172179347728)
    assert_allclose(d.calculate_integral_over_T(380, 200), -49.473623647253014)
    assert_allclose(d.calculate_integral_over_T(380+1E-9, 200), -49.473623647253014)
    assert_allclose(d.calculate_integral_over_T(590-1E-9, 300), -55.55836880929313)
    assert_allclose(d.calculate_integral_over_T(590+1E-9, 300), -55.55836880929313)
    assert_allclose(d.calculate_integral_over_T(635-1E-9, 200), -99.54361626259453)
    assert_allclose(d.calculate_integral_over_T(635+1E-9, 200), -99.54361626259453)
    assert_allclose(d.calculate_integral_over_T(644.6, 200), -156.74427000699035)
    assert_allclose(d.calculate_integral_over_T(645, 200), -159.51859302013582)
    
    
    # Test a chemical with only one set of coefficients
    d = zabransky_dict_iso_s['2016-57-1']
    assert_allclose(d.calculate(310), 375.54305039281155)
    assert_allclose(d.calculate_integral(290, 340), 18857.287976064617)
    assert_allclose(d.calculate_integral_over_T(290, 340), 59.965097029680805)

    
def test_Zabransky_dicts():
    from thermo.heat_capacity import zabransky_dict_sat_s, zabransky_dict_sat_p, zabransky_dict_const_s, zabransky_dict_const_p, zabransky_dict_iso_s, zabransky_dict_iso_p
    quasi_dicts = [zabransky_dict_sat_p, zabransky_dict_const_p, zabransky_dict_iso_p]
    spline_dicts = [zabransky_dict_sat_s, zabransky_dict_const_s, zabransky_dict_iso_s]
    
    sums = [[4811.400000000001, 7889.5, 11323.099999999999], 
            [6724.9, 11083.200000000004, 17140.75],
            [37003.999999999985, 72467.1, 91646.64999999997]]
    coeff_sums = [[1509.3503910000002, 170.63668272100003, 553.71843, 3602.9634764, 2731.1423000000004, 2505.7230399999994],
                  [394736.92543421406, 52342.25656440046, 54451.52735, 366067.89141800004, 61161.632348850006, 207335.59372000452],
                  [85568.9366422, 9692.534972905993, 13110.905983999992, 97564.75442449997, 30855.65738500001, 73289.607074896]]
    attrs = ['Tmin', 'Tmax', 'Tc']
    for i in range(len(quasi_dicts)):
        tot_calc = [sum(np.abs([getattr(k, j) for k in quasi_dicts[i].values()])) for j in attrs]
        assert_allclose(tot_calc, sums[i])
        coeff_calc = sum(np.abs([getattr(k, 'coeffs') for k in quasi_dicts[i].values()]))
        assert_allclose(coeff_calc, coeff_sums[i])
    
    sums = [[209671.79999999999, 16980025.795606382],
            [35846.100000000006, 60581.6685163],
            [461964.80000000005, 146377687.7182098]] 
    attrs = ['Ts', 'coeff_sets']
    for i in range(len(spline_dicts)):
        tot = [np.sum([np.sum(np.abs(getattr(k, j))) for k in spline_dicts[i].values()]) for j in attrs]
        assert_allclose(tot, sums[i])


def test_HeatCapacitySolidMixture():
    from thermo import Mixture
    from thermo.heat_capacity import HeatCapacitySolidMixture
    
    m = Mixture(['silver', 'platinum'], ws=[0.95, 0.05])
    obj = HeatCapacitySolidMixture(CASs=m.CASs, HeatCapacitySolids=m.HeatCapacitySolids)
    
    Cp = obj(m.T, m.P, m.zs, m.ws)
    assert_allclose(Cp, 25.32745719036059)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


def test_HeatCapacityGasMixture():
    from thermo import Mixture
    from thermo.heat_capacity import HeatCapacityGasMixture
    
    m = Mixture(['oxygen', 'nitrogen'], ws=[.4, .6], T=350, P=1E6)
    obj = HeatCapacityGasMixture(CASs=m.CASs, HeatCapacityGases=m.HeatCapacityGases)
    
    Cp = obj(m.T, m.P, m.zs, m.ws)
    assert_allclose(Cp, 29.361044582498046)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


def test_HeatCapacityLiquidMixture():
    from thermo import Mixture
    from thermo.heat_capacity import HeatCapacityLiquidMixture, SIMPLE
    
    m = Mixture(['water', 'sodium chloride'], ws=[.9, .1], T=301.5)
    obj = HeatCapacityLiquidMixture(MWs=m.MWs, CASs=m.CASs, HeatCapacityLiquids=m.HeatCapacityLiquids)
    
    Cp = obj(m.T, m.P, m.zs, m.ws)
    assert_allclose(Cp, 72.29643435124115)
    
    Cp = obj.calculate(m.T, m.P, m.zs, m.ws, SIMPLE)
    assert_allclose(Cp, 78.90470515935154)
    
    m = Mixture(['toluene', 'decane'], ws=[.9, .1], T=300)
    obj = HeatCapacityLiquidMixture(CASs=m.CASs, HeatCapacityLiquids=m.HeatCapacityLiquids)
    assert_allclose(obj(m.T, m.P, m.zs, m.ws), 168.29157865567112)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')



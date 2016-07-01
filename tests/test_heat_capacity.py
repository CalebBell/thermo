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
import pytest
from thermo.heat_capacity import *


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

    Cp = Zabransky_cubic(298.15, 20.9634, -10.1344, 2.8253, -0.256738)
    assert_allclose(Cp, 75.31462591538555)


def test_Lastovka_Shaw():
    # C64H52S2 (M = 885.2 alpha = 0.1333 mol/g
    # From figure 22, part b
    # Some examples didn't match so well; were the coefficients rounded?
    assert_allclose(Lastovka_Shaw(1000.0, 0.1333), 2467.113309084757)

    # Same, but try the correlation for cyclic aliphatic compounds
    assert_allclose(Lastovka_Shaw(1000.0, 0.1333, cyclic_aliphatic=True), 2187.36187944884)


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


@pytest.mark.meta_T_dept
def test_HeatCapacityGas():
    EtOH = HeatCapacityGas(CASRN='64-17-5', similarity_variable=0.1953615, MW=46.06844)
    Cps_calc = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(305))[1] for i in EtOH.all_methods]
    assert_allclose(sorted(Cps_calc), sorted([66.35085001015844, 74.6763493522965, 66.40063819791762, 66.25918325111196, 71.07236200126606, 65.6, 65.21]))

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
    assert_allclose(1.35441088517, EtOH.T_dependent_property(275))

    assert None == EtOH.T_dependent_property(5000)


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
    assert_allclose(NaCl.T_dependent_property(275), 18.320355898506502)

    NaCl.tabular_extrapolation_permitted = False
    assert None == NaCl.T_dependent_property(601)

@pytest.mark.meta_T_dept
def test_HeatCapacityLiquid():
    tol = HeatCapacityLiquid(CASRN='108-88-3', MW=92.13842, Tc=591.75, omega=0.257, Cpgm=115.30398669098454, similarity_variable=0.16279853724428964)
    Cpl_calc = [(tol.set_user_methods(i, forced=True), tol.T_dependent_property(330))[1] for i in tol.all_methods]
    Cpls = [165.4728226923247, 166.5239869108539, 166.52164399712314, 175.3439256239127, 166.71561127721478, 157.3, 165.4554033804999, 166.69807427725885, 157.29, 167.3380448453572]
    assert_allclose(sorted(Cpl_calc), sorted(Cpls))

    assert [None]*10 == [(tol.set_user_methods(i, forced=True), tol.T_dependent_property(2000))[1] for i in tol.all_methods]

    with pytest.raises(Exception):
        tol.test_method_validity('BADMETHOD', 300)

    assert False == tol.test_property_validity(-1)
    assert False == tol.test_property_validity(1.01E5)
    assert True == tol.test_property_validity(100)



    pb = HeatCapacityLiquid(MW=120.19158, CASRN='103-65-1', Tc=638.35)
    Cpl_calc = [(pb.set_user_methods(i, forced=True), pb.T_dependent_property(298.15))[1] for i in pb.all_methods]
    Cpls = [214.696720482603, 214.71, 214.7, 214.6498824147387]
    assert_allclose(sorted(Cpl_calc), sorted(Cpls))

    ctp = HeatCapacityLiquid(MW=118.58462, CASRN='96-43-5')
    Cpl_calc = [(ctp.set_user_methods(i, forced=True), ctp.T_dependent_property(250))[1] for i in ctp.all_methods]
    Cpls = [134.1186283149712, 134.14961304014292]
    assert_allclose(sorted(Cpl_calc), sorted(Cpls))

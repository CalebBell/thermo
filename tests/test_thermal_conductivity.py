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
from thermo.thermal_conductivity import *


def test_CSP_liq():
    kl = Sheffy_Johnson(300, 47, 280)
    assert_allclose(kl, 0.17740150413112196)

    kl = Sato_Riedel(300, 47, 390, 520)
    assert_allclose(kl, 0.2103769246133769)

    kl = Lakshmi_Prasad(273.15, 100)
    assert_allclose(kl, 0.013664450000000009)

    kl = Gharagheizi_liquid(300, 40, 350, 1E6, 0.27)
    assert_allclose(kl, 0.2171113029534838)

    kl = Nicola_original(300, 142.3, 611.7, 0.49, 201853)
    assert_allclose(kl, 0.2305018632230984)

    kl = Nicola(300, 142.3, 611.7, 2110000.0, 0.49)
    assert_allclose(kl, 0.10863821554584034)
    # Not at all sure about this one

    kl = Bahadori_liquid(273.15, 170)
    assert_allclose(kl, 0.14274278108272603)


def test_CSP_liq_dense():
    # From [2]_, for butyl acetate.
    kl_dense = DIPPR9G(515.05, 3.92E7, 579.15, 3.212E6, 7.085E-2)
    assert_allclose(kl_dense, 0.0864419738671184)

    kld1 = Missenard(304., 6330E5, 591.8, 41E5, 0.129)
    # # butyl acetate
    kld2 = Missenard(515.05, 3.92E7, 579.15, 3.212E6, 7.085E-2)
    assert_allclose([kld1, kld2], [0.21983757770696569, 0.086362465280714396])


def test_CSP_gas():
    # 2-methylbutane at low pressure, 373.15 K. Mathes calculation in [1]_.
    kg =  Eucken(72.151, 135.9, 8.77E-6)
    assert_allclose(kg, 0.018792644287722975)

    # 2-methylbutane at low pressure, 373.15 K. Mathes calculation in [1]_.
    kg = Eucken_modified(72.151, 135.9, 8.77E-6)
    assert_allclose(kg, 0.023593536999201956)

    # CO, brute force tests on three  options for chemtype
    kg1 = DIPPR9B(200., 28.01, 20.826, 1.277E-5, 132.92, chemtype='linear')
    assert kg1 == DIPPR9B(200., 28.01, 20.826, 1.277E-5, 132.92) # No argument
    kg2 = DIPPR9B(200., 28.01, 20.826, 1.277E-5, 132.92, chemtype='monoatomic')
    kg3 = DIPPR9B(200., 28.01, 20.826, 1.277E-5, 132.92, chemtype='nonlinear')
    assert_allclose([kg1, kg2, kg3], [0.01813208676438415, 0.023736881470903245, 0.018625352738307743])

    with pytest.raises(Exception):
        DIPPR9B(200., 28.01, 20.826, 1.277E-5, 132.92, chemtype='FAIL')


    kg = Chung(T=373.15, MW=72.151, Tc=460.4, omega=0.227, Cvm=135.9, mu=8.77E-6)
    assert_allclose(kg, 0.023015653729496946)

    kg = eli_hanley(T=373.15, MW=72.151, Tc=460.4, Vc=3.06E-4, Zc=0.267, omega=0.227, Cvm=135.9)
    assert_allclose(kg, 0.022479517891353377)

    kg = eli_hanley(T=1000, MW=72.151, Tc=460.4, Vc=3.06E-4, Zc=0.267, omega=0.227, Cvm=135.9)
    assert_allclose(kg, 0.06369581356766069)

    kg = Bahadori_gas(40+273.15, 20)
    assert_allclose(kg, 0.031968165337873326)

    kg = Gharagheizi_gas(580., 16.04246, 111.66, 4599000.0, 0.0115478000)
    assert_allclose(kg, 0.09594861261873211)


def test_CSP_gas_dense():
    kgs = [stiel_thodos_dense(T=378.15, MW=44.013, Tc=309.6, Pc=72.4E5, Vc=97.4E-6, Zc=0.274, Vm=i, kg=2.34E-2) for i in [144E-6, 24E-6, 240E-6]]
    kgs_exp = [0.041245574404863684, 0.9158718777539487, 0.03258313269922979]
    assert_allclose(kgs, kgs_exp)


    kgs = [eli_hanley_dense(T=T, MW=42.081, Tc=364.9, Vc=1.81E-4, Zc=0.274, omega=0.144, Cvm=82.70, Vm=1.721E-4) for T in [473., 900]]
    kgs_exp = [0.06038475936515042, 0.08987438807653142]
    assert_allclose(kgs, kgs_exp)

    kg = eli_hanley_dense(700, MW=42.081, Tc=364.9, Vc=1.81E-4, Zc=0.274, omega=0.144, Cvm=82.70, Vm=3.721E-4)
    assert_allclose(kg, 0.06953791121177173)

    kg = chung_dense(T=473., MW=42.081, Tc=364.9, Vc=184.6E-6, omega=0.142, Cvm=82.67, Vm=172.1E-6, mu=134E-7, dipole=0.4)
    assert_allclose(kg, 0.06160570379787278)


@pytest.mark.meta_T_dept
def test_ThermalConductivityLiquid():
    EtOH = ThermalConductivityLiquid(CASRN='64-17-5', MW=46.06844, Tm=159.05, Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, Hfus=4931.0)

    EtOH.T_dependent_property(305.)
    kl_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(305.))[1] for i in EtOH.sorted_valid_methods]
    kl_exp = [0.162183005823234, 0.17417420086033197, 0.20068212675966418, 0.18526367184633258, 0.18846433785041306, 0.16837295487233528, 0.16883011582627103, 0.09330268101157643, 0.028604363267557775]
    assert_allclose(kl_calcs, kl_exp)

    # Test that methods return None
    kl_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(5000))[1] for i in EtOH.sorted_valid_methods]
    assert [None]*9 == kl_calcs

    EtOH.set_user_methods(VDI_TABULAR, forced=True)
    assert_allclose(EtOH.T_dependent_property(600.), 0.040117737789202995)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.T_dependent_property(600.)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')


    # Ethanol compressed
    assert [False, True] == [EtOH.test_method_validity_P(300, P, COOLPROP) for P in (1E3, 1E5)]
    assert [True, True] == [EtOH.test_method_validity_P(300, P, DIPPR_9G) for P in (1E3, 1E5)]
    assert [True, True, False] == [EtOH.test_method_validity_P(300, P, MISSENARD) for P in (1E3, 1E5, 1E10)]

    EtOH = ThermalConductivityLiquid(CASRN='64-17-5', MW=46.06844, Tm=159.05, Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, Hfus=4931.0)
    assert_allclose(EtOH.calculate_P(298.15, 1E6, COOLPROP), 0.1639626989794703)
    assert_allclose(EtOH.calculate_P(298.15, 1E6, DIPPR_9G), 0.1606146938795702)
    assert_allclose(EtOH.calculate_P(298.15, 1E6, MISSENARD), 0.1641582259181843)


    # Ethanol data, calculated from CoolProp
    Ts = [275, 300, 350]
    Ps = [1E5, 5E5, 1E6]
    TP_data = [[0.16848555706973622, 0.16313525757474362, 0.15458068887966378], [0.16868861153075654, 0.163343255114212, 0.1548036152853355], [0.16894182645698885, 0.1636025336196736, 0.15508116339039268]]
    EtOH.set_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_allclose(TP_data, recalc_pts)

    EtOH.forced_P = True
    assert_allclose(EtOH.TP_dependent_property(274, 9E4), 0.16848555706973622)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.TP_dependent_property(300, 9E4)

    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')

    assert False == EtOH.test_method_validity_P(-10, 1E5, DIPPR_9G)


@pytest.mark.meta_T_dept
def test_ThermalConductivityGas():
    EtOH = ThermalConductivityGas(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.2412, omega=0.635, dipole=1.44, Vmg=0.02357, Cvgm=56.98, mug=7.903e-6, CASRN='64-17-5')
    EtOH.T_dependent_property(298.15)
    kg_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(298.15))[1] for i in EtOH.sorted_valid_methods]
    kg_exp = [0.015227631457903644, 0.01494275, 0.016338750949017277, 0.014353317470206847, 0.011676848981094841, 0.01137910777526855, 0.015427444948536088, 0.012984129385510995, 0.017556325226536728]
    assert_allclose(kg_calcs, kg_exp)

    # Test that those mthods which can, do, return NoneEtOH.forced_P
    kg_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(5E20))[1] for i in [COOLPROP, VDI_TABULAR, GHARAGHEIZI_G, ELI_HANLEY, BAHADORI_G]]
    assert [None]*5 == kg_calcs

    # Test tabular limits/extrapolation
    EtOH.set_user_methods(VDI_TABULAR, forced=True)
    assert_allclose(EtOH.T_dependent_property(600.), 0.05755089974293061)

    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.T_dependent_property(600.)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')


    # Ethanol compressed

    assert [True, False] == [EtOH.test_method_validity_P(300, P, COOLPROP) for P in (1E3, 1E5)]
    assert [True, False] == [EtOH.test_method_validity_P(300, P, ELI_HANLEY_DENSE) for P in (1E5, -1E5)]
    assert [True, False] == [EtOH.test_method_validity_P(300, P, CHUNG_DENSE) for P in (1E5, -1E5)]
    assert [True, False] == [EtOH.test_method_validity_P(300, P, STIEL_THODOS_DENSE) for P in (1E5, -1E5)]


    EtOH = ThermalConductivityGas(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.2412, omega=0.635, dipole=1.44, Vmg=0.02357, Cvgm=56.98, mug=7.903e-6, CASRN='64-17-5')
    assert_allclose(EtOH.calculate_P(298.15, 1E2, COOLPROP), 0.015207849649231962)
    assert_allclose(EtOH.calculate_P(298.15, 1E6, ELI_HANLEY_DENSE), 0.011210125242396791)
    assert_allclose(EtOH.calculate_P(298.15, 1E6, CHUNG_DENSE), 0.011770372068085207)
    assert_allclose(EtOH.calculate_P(298.15, 1E6, STIEL_THODOS_DENSE), 0.015447836685420897)


      # Ethanol data, calculated from CoolProp
    Ts = [400, 500, 600]
    Ps = [1E4, 1E5, 2E5]
    TP_data = [[0.025825794817543015, 0.037905383602635095, 0.05080124980338535], [0.02601702567554805, 0.03806794452306919, 0.050946301396380594], [0.026243171168075605, 0.03825284803978187, 0.05110925652065333]]
    EtOH.set_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_allclose(TP_data, recalc_pts)

    EtOH.forced_P = True
    EtOH.tabular_extrapolation_permitted = True
    assert_allclose(EtOH.TP_dependent_property(399, 9E3), 0.025825794817543015)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.TP_dependent_property(399, 9E3)

    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')

    assert False == EtOH.test_method_validity_P(100, 1E5, COOLPROP)


def test_DIPPR9I():
    k = DIPPR9I([0.258, 0.742], [0.1692, 0.1528])
    assert_allclose(k, 0.15657104706719646)

    with pytest.raises(Exception):
        DIPPR9I([0.258, 0.742], [0.1692])

def test_Filippov():
    kl = Filippov([0.258, 0.742], [0.1692, 0.1528])
    assert_allclose(kl, 0.15929167628799998)

    with pytest.raises(Exception):
        Filippov([0.258], [0.1692, 0.1528])


def test_Lindsay_Bromley():
    kg = Lindsay_Bromley(323.15, [0.23, 0.77], [1.939E-2, 1.231E-2], [1.002E-5, 1.015E-5], [248.31, 248.93], [46.07, 50.49])
    assert_allclose(kg, 0.01390264417969313)

    with pytest.raises(Exception):
        Lindsay_Bromley(323.15, [0.23], [1.939E-2, 1.231E-2], [1.002E-5, 1.015E-5], [248.31, 248.93], [46.07, 50.49])
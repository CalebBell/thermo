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
import numpy as np
import pandas as pd
from thermo.viscosity import *
from thermo.identifiers import checkCAS
from thermo.viscosity import COOLPROP, LUCAS
from thermo.chemical import Mixture
from thermo.viscosity import LALIBERTE_MU, MIXING_LOG_MOLAR, MIXING_LOG_MASS, BROKAW, HERNING_ZIPPERER, WILKE, SIMPLE

### Check data integrity

def test_Dutt_Prasad_data():
    assert all([checkCAS(i) for i in Dutt_Prasad.index])
    tots_calc = [Dutt_Prasad[i].abs().sum() for i in ['A', 'B', 'C', 'Tmin', 'Tmax']]
    tots = [195.89260000000002, 65395.299999999996, 9849.1899999999987, 25952, 35016]
    assert_allclose(tots_calc, tots)

    assert Dutt_Prasad.index.is_unique
    assert Dutt_Prasad.shape == (100, 6)


def test_VN2E_data():
    assert all([checkCAS(i) for i in VN2E_data.index])
    tots_calc = [VN2E_data[i].abs().sum() for i in ['C', 'D', 'Tmin', 'Tmax']]
    tots = [567743298666.74878, 48.8643, 3690, 4860]
    assert_allclose(tots_calc, tots)

    assert VN2E_data.index.is_unique
    assert VN2E_data.shape == (14, 6)


def test_VN3_data():
    assert all([checkCAS(i) for i in VN3_data.index])
    tots_calc = [VN3_data[i].abs().sum() for i in ['A', 'B', 'C', 'Tmin', 'Tmax']]
    tots = [645.18849999999998, 169572.65159999998, 50050.151870000002, 126495, 175660]
    assert_allclose(tots_calc, tots)

    assert VN3_data.index.is_unique
    assert VN3_data.shape == (432, 7)

def test_VN2_data():
    assert all([checkCAS(i) for i in VN3_data.index])
    tots_calc = [VN2_data[i].abs().sum() for i in ['A', 'B', 'Tmin', 'Tmax']]
    tots = [674.10069999999996, 83331.98599999999, 39580, 47897]
    assert_allclose(tots_calc, tots)

    assert VN2_data.index.is_unique
    assert VN2_data.shape == (135, 6)


def test_Perrys2_313_data():
    # All values calculated at Tmin and Tmax check out to at least 5E-3 precision
    # The rounding has some effect, but it is not worrying.
    assert all([checkCAS(i) for i in Perrys2_313.index])
    tots_calc = [Perrys2_313[i].abs().sum() for i in [u'C1', u'C2', u'C3', u'C4', u'C5', u'Tmin', u'Tmax']]
    tots = [9166.6971369999992, 615425.94497999991, 1125.5317557875198, 9.054869390623603e+34, 402.21244000000002, 72467.140000000014, 136954.85999999999]
    assert_allclose(tots_calc, tots)
    
    assert Perrys2_313.index.is_unique
    assert Perrys2_313.shape == (337, 8)

    
def test_Perrys2_312_data():
    # Argon, Difluoromethane, 1-Hexyne, Methylsilane, Nitrogen trifluoride, 
    # Vinyl chloride all do not match on Tmax at 1E-3 - their models predict 
    # ~1E-5 Pa*S, but listed values are ~1E-10 to 1E-12. Unsure of the cause.
    # All coumpounds match at 1E-3 for Tmin.
    
    assert all([checkCAS(i) for i in Perrys2_312.index])
    tots_calc = [Perrys2_312[i].abs().sum() for i in [u'C1', u'C2', u'C3', u'C4', u'Tmin', u'Tmax']]
    tots = [0.00019683902626010103, 250.10520100000002, 65862.829200000007, 191286, 74802.639999999999, 355064.37]
    assert_allclose(tots_calc, tots)
    
    assert Perrys2_312.index.is_unique
    assert Perrys2_312.shape == (345, 7)


def test_VDI_PPDS_7_data():
    assert all([checkCAS(i) for i in VDI_PPDS_7.index])
    tots_calc = [VDI_PPDS_7[i].abs().sum() for i in [u'A', u'B', u'C', u'D', u'E']]
    tots = [507.14607000000001, 1680.7624099999998, 165461.14259999999, 46770.887000000002, 0.057384780000000003]
    assert_allclose(tots_calc, tots)
    
    assert VDI_PPDS_7.index.is_unique
    assert VDI_PPDS_7.shape == (271, 7)

def test_VDI_PPDS_8_data():
    # Coefficients for water are incorrect - obtained an average deviation of 150%!
    assert all([checkCAS(i) for i in VDI_PPDS_8.index])
    tots_calc = [VDI_PPDS_8[i].abs().sum() for i in [u'A', u'B', u'C', u'D', u'E']]
    tots = [0.00032879559999999999, 9.5561339999999995e-06, 2.8377710000000001e-09, 2.8713399999999998e-12, 2.8409200000000004e-15]
    assert_allclose(tots_calc, tots)
    
    assert VDI_PPDS_8.index.is_unique
    assert VDI_PPDS_8.shape == (274, 6)

def test_ViswanathNatarajan():
    mu = ViswanathNatarajan2(348.15, -5.9719, 1007.0)
    assert_allclose(mu, 0.00045983686956829517)

    mu = ViswanathNatarajan2Exponential(298.15, 4900800,  -3.8075)
    assert_allclose(mu, 0.0018571903840928496)

    mu = ViswanathNatarajan3(298.15, -2.7173, -1071.18, -129.51)
    assert_allclose(mu, 0.0006129806445142112)


def test_Letsou_Stiel():
    # Checked 2017-03-05
    mu = Letsou_Stiel(400., 46.07, 516.25, 6.383E6, 0.6371)
    assert_allclose(mu, 0.0002036150875308151)

    # Butane, custom case; vs 0.000166 exp
    mu = Letsou_Stiel(298.15, 58.1222, 425.12, 3796000.0, 0.193)
    assert_allclose(mu, 0.00015385075092073157)


def test_Przedziecki_Sridhar():
    # Reid (1983) toluene, 383K
    mu = Przedziecki_Sridhar(383., 178., 591.8, 41E5, 316E-6, 95E-6, .263, 92.14)
    assert_allclose(mu, 0.0002198147995603383)


def test_Lucas():
    # methylcyclohexane
    mu = Lucas(300., 500E5, 572.2, 34.7E5, 0.236, 0, 0.00068)
    assert_allclose(mu, 0.0010683738499316518)

    # Psat > P:
    mu = Lucas(300., 500E5, 572.2, 34.7E5, 0.236, 501E5, 0.00068)
    assert_allclose(mu, 0.00068)


def test_Stiel_Thodos():
    # CCl4
    mu = Stiel_Thodos(300., 556.35, 4.5596E6, 153.8)
    assert_allclose(mu, 1.0408926223608723e-05)

    # Tr > 1.5
    mu = Stiel_Thodos(900., 556.35, 4.5596E6, 153.8)
    assert_allclose(mu, 2.899111242556782e-05)


def test_Yoon_Thodos():
    # CCl4
    mu = Yoon_Thodos(300., 556.35, 4.5596E6, 153.8)
    assert_allclose(mu, 1.0194885727776819e-05)


def test_Gharagheizi_gas_viscosity():
    mu = Gharagheizi_gas_viscosity(120., 190.564, 45.99E5, 16.04246)
    assert_allclose(mu, 5.215761625399613e-06)


def test_Lucas_gas():
    mu = lucas_gas(T=550., Tc=512.6, Pc=80.9E5, Zc=0.224, MW=32.042, dipole=1.7)
    assert_allclose(mu, 1.7822676912698928e-05)

    mu = lucas_gas(T=550., Tc=512.6, Pc=80.9E5, Zc=0.224, MW=32.042, dipole=None)
    assert_allclose(mu, 1.371116974367763e-05)

    mu = lucas_gas(T=550., Tc=512.6, Pc=80.9E5, Zc=0.224, MW=32.042, dipole=8)
    assert_allclose(mu, 1.7811559961984407e-05)

    # Helium, testing Q
    mu = lucas_gas(T=6, Tc=5.1889, Pc=226968.0, Zc=0.3014, MW=4.002602, CASRN='7440-59-7')
    assert_allclose(mu, 1.3042945737346396e-06)

    mu = lucas_gas(T=150, Tc=5.1889, Pc=226968.0, Zc=0.3014, MW=4.002602, CASRN='7440-59-7')
    assert_allclose(mu, 1.2558477184738118e-05)


def test_Wilke():
    mu = Wilke([0.05, 0.95], [1.34E-5, 9.5029E-6], [64.06, 46.07])
    assert_allclose(mu, 9.701614885866193e-06)

    with pytest.raises(Exception):
        Wilke([0.05], [1.34E-5, 9.5029E-6], [64.06, 46.07])


def test_Herning_Zipperer():
    mu = Herning_Zipperer([0.05, 0.95], [1.34E-5, 9.5029E-6], [64.06, 46.07])
    assert_allclose(mu, 9.730630997268096e-06)

    with pytest.raises(Exception):
        Herning_Zipperer([0.05], [1.34E-5, 9.5029E-6], [64.06, 46.07])


def test_Brockaw():
    mu = Brokaw(308.2, [0.05, 0.95], [1.34E-5, 9.5029E-6], [64.06, 46.07], [0.42, 0.19], [347, 432])
    assert_allclose(mu, 9.699085099801568e-06)

    with pytest.raises(Exception):
        Brokaw(308.2, [0.95], [1.34E-5, 9.5029E-6], [64.06, 46.07], [0.42, 0.19], [347, 432])

    # Test < 0.1 MD
    mu = Brokaw(308.2, [0.05, 0.95], [1.34E-5, 9.5029E-6], [64.06, 46.07], [0.42, 0.05], [347, 432])
    assert_allclose(mu, 9.70504431025103e-06)


def test_round_whole_even():
    from thermo.viscosity import _round_whole_even
    assert _round_whole_even(116.4) == 116
    assert _round_whole_even(116.6) == 117
    assert _round_whole_even(116.5) == 116
    assert _round_whole_even(115.5) == 116


def test_viscosity_index():
    # 4/5 examples from the official document, one custom example
    assert_allclose(92.4296472453428, viscosity_index(73.3E-6, 8.86E-6))
    assert_allclose(156.42348293257797, viscosity_index(22.83E-6, 5.05E-6))
    assert_allclose(111.30701701381422, viscosity_index(53.47E-6, 7.80E-6))
    assert_allclose(92.03329369797858, viscosity_index(73.5E-6, 8.86E-6))
    assert_allclose(192.9975428057893, viscosity_index(1000E-6, 100E-6)) # Custom
    assert 193 == viscosity_index(1000E-6, 100E-6, rounding=True) # custom, rounded
    assert 92 == viscosity_index(73.3E-6, 8.86E-6, rounding=True)
    assert None == viscosity_index(3E-6, 1.5E-6)


@pytest.mark.meta_T_dept
def test_ViscosityLiquid():
    EtOH = ViscosityLiquid(MW=46.06844, Tm=159.05, Tc=514.0, Pc=6137000.0, Vc=0.000168, omega=0.635, Psat=7872.16, Vml=5.8676e-5, CASRN='64-17-5')

    mul_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(298.15))[1] for i in EtOH.all_methods]
    mul_exp = [0.0010623746999654108, 0.0004191198228004424, 0.0010823506202025659, 0.0010720812586059742, 0.0010713697500000004, 0.0031157679801337825, 0.0010774308462863267, 0.0010823506202025659]
    assert_allclose(sorted(mul_calcs), sorted(mul_exp))

    # Test that methods return None
    EtOH.tabular_extrapolation_permitted = False
    Vml_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(600))[1] for i in EtOH.all_methods]
    assert [None]*8 == Vml_calcs

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    # Acetic acid to test ViswanathNatarajan2Exponential
    acetic_acid = ViscosityLiquid(CASRN='64-19-7', Tc=590.7)
    mul_calcs = [(acetic_acid.set_user_methods(i), acetic_acid.T_dependent_property(350.0))[1] for i in acetic_acid.all_methods]
    mul_exp = [0.0005744169247310638, 0.0005089289428076254, 0.0005799665143154318, 0.0005727888422607339, 0.000587027903931889]
    assert_allclose(sorted(mul_calcs), sorted(mul_exp))
    assert [None]*5 == [(acetic_acid.set_user_methods(i), acetic_acid.T_dependent_property(650.0))[1] for i in acetic_acid.all_methods]

    # Test ViswanathNatarajan2 with boron trichloride
    mu = ViscosityLiquid(CASRN='10294-34-5').T_dependent_property(250)
    assert_allclose(mu, 0.0003389255178814321)
    assert None == ViscosityLiquid(CASRN='10294-34-5').T_dependent_property(350)


    # Ethanol compressed
    EtOH = ViscosityLiquid(MW=46.06844, Tm=159.05, Tc=514.0, Pc=6137000.0, Vc=0.000168, omega=0.635, Psat=7872.16, Vml=5.8676e-5, CASRN='64-17-5')

    assert [False, True] == [EtOH.test_method_validity_P(300, P, COOLPROP) for P in (1E3, 1E5)]
    assert [True, True] == [EtOH.test_method_validity_P(300, P, LUCAS) for P in (1E3, 1E5)]

    assert_allclose(EtOH.calculate_P(298.15, 1E6, LUCAS), 0.0010880229239312313)
    assert_allclose(EtOH.calculate_P(298.15, 1E6, COOLPROP), 0.0010885493279015608)

    EtOH = ViscosityLiquid(MW=46.06844, Tm=159.05, Tc=514.0, Pc=6137000.0, Vc=0.000168, omega=0.635, Psat=7872.16, Vml=5.8676e-5, CASRN='64-17-5')
    # Ethanol data, calculated from CoolProp
    Ts = [275, 300, 350]
    Ps = [1E5, 5E5, 1E6]
    TP_data = [[0.0017455993713216815, 0.0010445175985089377, 0.00045053170256051774], [0.0017495149679815605, 0.0010472128172002075, 0.000452108003076486], [0.0017543973013034444, 0.0010505716944451827, 0.00045406921275411145]]
    EtOH.set_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_allclose(TP_data, recalc_pts)

    EtOH.tabular_extrapolation_permitted = False
    EtOH.forced_P = True
    assert None == EtOH.TP_dependent_property(300, 9E4)
    EtOH.tabular_extrapolation_permitted = True
    assert_allclose(EtOH.TP_dependent_property(300, 9E4), 0.0010445175985089377)

    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')

@pytest.mark.meta_T_dept
def test_ViscosityGas():
    EtOH = ViscosityGas(MW=46.06844, Tc=514.0, Pc=6137000.0, Zc=0.2412, dipole=1.44, Vmg=0.02357, CASRN='64-17-5')

    mug_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(298.15))[1] for i in EtOH.all_methods]
    mug_exp = [8.934627758386856e-06, 8.933684639927153e-06, 7.414017252400231e-06, 8.772549629893446e-06, 8.5445e-06, 7.902892297231681e-06, 8.805532218477024e-06, 7.536618820670175e-06]
    assert_allclose(sorted(mug_calcs), sorted(mug_exp))

    # Test that methods return None
    EtOH.tabular_extrapolation_permitted = False
    mug_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(6000))[1] for i in EtOH.all_methods]
    assert [None]*8 == mug_calcs

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')


    # Ethanol compressed
    EtOH = ViscosityGas(MW=46.06844, Tc=514.0, Pc=6137000.0, Zc=0.2412, dipole=1.44, Vmg=0.02357, CASRN='64-17-5')

    assert [True, False] == [EtOH.test_method_validity_P(300, P, COOLPROP) for P in (1E3, 1E5)]

    assert_allclose(EtOH.calculate_P(298.15, 1E3, COOLPROP), 8.77706377246337e-06)


    # Ethanol data, calculated from CoolProp
    Ts = [400, 500, 550]
    Ps = [1E3, 1E4, 1E5]
    TP_data = [[1.18634700291489e-05, 1.4762189560203758e-05, 1.6162732753470533e-05], [1.1862505513959454e-05, 1.4762728590964208e-05, 1.6163602669178767e-05], [1.1853229260926176e-05, 1.4768417536555742e-05, 1.617257402798515e-05]]
    EtOH.set_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_allclose(TP_data, recalc_pts)

    EtOH.tabular_extrapolation_permitted = False
    EtOH.forced_P = True
    assert None == EtOH.TP_dependent_property(300, 9E4)
    EtOH.tabular_extrapolation_permitted = True
    assert_allclose(EtOH.TP_dependent_property(300, 9E4), 1.1854259955707653e-05)

    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')



def test_ViscosityLiquidMixture():
    # DIPPR  1983 manual example
    m = Mixture(['carbon tetrachloride', 'isopropanol'], zs=[0.5, 0.5], T=313.2)
    
    ViscosityLiquids = [i.ViscosityLiquid for i in m.Chemicals]

    obj = ViscosityLiquidMixture(ViscosityLiquids=ViscosityLiquids, CASs=m.CASs)
    mu = obj.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(mu, 0.0009956952502281852)
    
    mu = obj.calculate(m.T, m.P, m.zs, m.ws, MIXING_LOG_MOLAR)
    assert_allclose(mu, 0.0009956952502281852)
    mu = obj.calculate(m.T, m.P, m.zs, m.ws, MIXING_LOG_MASS)
    assert_allclose(mu, 0.0008741268796817256)
    
    # Test Laliberte
    m = Mixture(['water', 'sulfuric acid'], zs=[0.5, 0.5], T=298.15)
    ViscosityLiquids = [i.ViscosityLiquid for i in m.Chemicals]
    obj = ViscosityLiquidMixture(ViscosityLiquids=ViscosityLiquids, CASs=m.CASs)
    mu = obj.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(mu, 0.024955325569420893)
    assert obj.sorted_valid_methods == [LALIBERTE_MU]
    
    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


def test_ViscosityGasMixture():
    # DIPPR  1983 manual example
    m = Mixture(['dimethyl ether', 'sulfur dioxide'], zs=[.95, .05], T=308.2)
    ViscosityGases = [i.ViscosityGas for i in m.Chemicals]
    obj = ViscosityGasMixture(MWs=m.MWs, molecular_diameters=m.molecular_diameters, Stockmayers=m.Stockmayers, CASs=m.CASs, ViscosityGases=ViscosityGases)
    
    mu =  obj.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(mu, 9.637173494726528e-06)
    
    viscosity_gas_mixture_methods = [BROKAW, HERNING_ZIPPERER, WILKE, SIMPLE]
    mus = [obj.calculate(m.T, m.P, m.zs, m.ws, method) for method in viscosity_gas_mixture_methods]
    assert_allclose(mus, [9.637173494726528e-06, 9.672122280295219e-06, 9.642294904686337e-06, 9.638962759382555e-06])

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

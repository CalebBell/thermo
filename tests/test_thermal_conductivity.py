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
from thermo.chemical import Mixture
from thermo.identifiers import checkCAS
from thermo.thermal_conductivity import MAGOMEDOV, DIPPR_9H, FILIPPOV, SIMPLE, ThermalConductivityLiquidMixture
from thermo.thermal_conductivity import (GHARAGHEIZI_G, CHUNG, ELI_HANLEY, VDI_PPDS,
                                        ELI_HANLEY_DENSE, CHUNG_DENSE, 
                                        EUCKEN_MOD, EUCKEN, BAHADORI_G, 
                                        STIEL_THODOS_DENSE, DIPPR_9B, COOLPROP,
                                        DIPPR_PERRY_8E, VDI_TABULAR, GHARAGHEIZI_L,
                                       SATO_RIEDEL, NICOLA, NICOLA_ORIGINAL,
                                       SHEFFY_JOHNSON, BAHADORI_L,
                                       LAKSHMI_PRASAD, DIPPR_9G, MISSENARD)

def test_Perrys2_314_data():
    # In perry's, only 102 is used. No chemicals are missing.
    # Tmaxs all match to 5E-4. Tmins match to 1E-3.
    assert all([checkCAS(i) for i in Perrys2_314.index])
    tots_calc = [Perrys2_314[i].abs().sum() for i in [u'C1', u'C2', u'C3', u'C4', u'Tmin', u'Tmax']]
    tots = [48935634.823768869, 297.41545078799999, 421906466448.71423, 232863514627157.62, 125020.26000000001, 347743.42000000004]
    assert_allclose(tots_calc, tots)
    
    assert Perrys2_314.index.is_unique
    assert Perrys2_314.shape == (345, 7)


def test_Perrys2_315_data():
    # From perry's - Deuterium  ,  Nitrogen trifluoride  ,  Nitrous oxide   Silicon tetrafluoride  ,  Terephthalic acid  all have no data
    # All perry's use #100.
    # Tmins all match at 5E-4.
    # Tmaxs all match at 2E-3.
    assert all([checkCAS(i) for i in Perrys2_315.index])
    tots_calc = [Perrys2_315[i].abs().sum() for i in [u'C1', u'C2', u'C3', u'C4', u'C5', u'Tmin', u'Tmax']]
    tots = [82.001667499999996, 0.19894598900000002, 0.0065330144999999999, 0.00046928630199999995, 1.0268010799999999e-07, 70996.369999999995, 138833.41]
    assert_allclose(tots_calc, tots)
    
    assert Perrys2_315.index.is_unique
    assert Perrys2_315.shape == (340, 8)


def test_VDI_PPDS_10_data():
    '''Average deviation of 2.4% from tabulated values. Many chemicals have
    much higher deviations. 10% or more deviations:
    ['75-34-3', '107-06-2', '106-93-4', '420-46-2', '71-55-6', '79-34-5', 
    '67-72-1', '76-12-0', '76-13-1', '76-14-2', '540-54-5', '75-01-4', 
    '75-35-4', '79-01-6', '127-18-4', '462-06-6', '108-90-7', '108-86-1', 
    '108-41-8', '100-44-7', '108-93-0', '100-61-8', '121-69-7', '91-66-7']
    
    These have been checked - it appears the tabulated data is just incorrect.
    '''

    assert all([checkCAS(i) for i in VDI_PPDS_10.index])
    tots_calc = [VDI_PPDS_10[i].abs().sum() for i in [u'A', u'B', u'C', u'D', u'E']]
    tots = [2.2974640014599998, 0.015556001460000001, 1.9897655000000001e-05, 6.7747269999999993e-09, 2.3260109999999999e-12]
    assert_allclose(tots_calc, tots)
    
    assert VDI_PPDS_10.index.is_unique
    assert VDI_PPDS_10.shape == (275, 6)


def test_VDI_PPDS_9_data():
    '''Average deviation of 0.71% from tabulated values. The following have 
    larger deviations
        
    ['124-18-5', '629-59-4', '629-78-7', '526-73-8', '95-63-6']
    
    These have been checked - it appears the tabulated data is just incorrect.
    '''

    assert all([checkCAS(i) for i in VDI_PPDS_9.index])
    tots_calc = [VDI_PPDS_9[i].abs().sum() for i in [u'A', u'B', u'C', u'D', u'E']]
    tots = [63.458699999999993, 0.14461469999999998, 0.00042270770000000005, 1.7062660000000002e-06, 3.2715370000000003e-09]
    assert_allclose(tots_calc, tots)
    
    assert VDI_PPDS_9.index.is_unique
    assert VDI_PPDS_9.shape == (271, 6)


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
    methods = list(EtOH.sorted_valid_methods)
    methods.remove(VDI_TABULAR)
    kl_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(305.))[1] for i in methods]
    kl_exp = [0.162183005823234, 0.16627999999999998, 0.166302, 0.20068212675966418, 0.18526367184633258, 0.18846433785041306, 0.16837295487233528, 0.16883011582627103, 0.09330268101157643, 0.028604363267557775]
    assert_allclose(sorted(kl_calcs), sorted(kl_exp))
    
    assert_allclose(EtOH.calculate(305., VDI_TABULAR), 0.17417420086033197, rtol=1E-4)
    

    # Test that methods return None
    kl_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(5000))[1] for i in EtOH.sorted_valid_methods]
    assert [None]*11 == kl_calcs

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
    kg_exp = [0.015227631457903644, 0.015025094773729045, 0.01520257225203181, 0.01494275, 0.016338750949017277, 0.014353317470206847, 0.011676848981094841, 0.01137910777526855, 0.015427444948536088, 0.012984129385510995, 0.017556325226536728]
    assert_allclose(kg_calcs, kg_exp)

    # Test that those mthods which can, do, return NoneEtOH.forced_P
    kg_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(5E20))[1] for i in [COOLPROP, DIPPR_PERRY_8E, VDI_TABULAR, GHARAGHEIZI_G, ELI_HANLEY, BAHADORI_G, VDI_PPDS]]
    assert [None]*7 == kg_calcs

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


def test_DIPPR9H():
    k = DIPPR9H([0.258, 0.742], [0.1692, 0.1528])
    assert_allclose(k, 0.15657104706719646)

    with pytest.raises(Exception):
        DIPPR9H([0.258, 0.742], [0.1692])

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



def test_ThermalConductivityGasMixture():
    from thermo.chemical import Mixture
    from thermo.thermal_conductivity import ThermalConductivityGasMixture, LINDSAY_BROMLEY, SIMPLE
    
    m2 = Mixture(['nitrogen', 'argon', 'oxygen'], ws=[0.7557, 0.0127, 0.2316])
    ThermalConductivityGases = [i.ThermalConductivityGas for i in m2.Chemicals]
    ViscosityGases = [i.ViscosityGas for i in m2.Chemicals]

    kg_mix = ThermalConductivityGasMixture(MWs=m2.MWs, Tbs=m2.Tbs, CASs=m2.CASs, 
                                      ThermalConductivityGases=ThermalConductivityGases, 
                                      ViscosityGases=ViscosityGases)
    
    k = kg_mix.mixture_property(m2.T, m2.P, m2.zs, m2.ws)
    assert_allclose(k, 0.025864474514829254) # test LINDSAY_BROMLEY and mixture property
    # Do it twice to test the stored method
    k = kg_mix.mixture_property(m2.T, m2.P, m2.zs, m2.ws)
    assert_allclose(k, 0.025864474514829254) # test LINDSAY_BROMLEY and mixture property

    k =  kg_mix.calculate(m2.T, m2.P, m2.zs, m2.ws, SIMPLE) # Test calculate, and simple
    assert_allclose(k, 0.02586655464213776)
    
    dT1 = kg_mix.calculate_derivative_T(m2.T, m2.P, m2.zs, m2.ws, LINDSAY_BROMLEY)
    dT2 = kg_mix.property_derivative_T(m2.T, m2.P, m2.zs, m2.ws)
    assert_allclose([dT1, dT2], [7.3391064059347144e-05]*2)
    
    dP1 = kg_mix.calculate_derivative_P(m2.P, m2.T, m2.zs, m2.ws, LINDSAY_BROMLEY)
    dP2 = kg_mix.property_derivative_P(m2.T, m2.P, m2.zs, m2.ws)
    
    assert_allclose([dP1, dP2], [3.5325319058809868e-10]*2, rtol=1E-4)
    
    # Test other methods
    
    assert kg_mix.user_methods == []
    assert kg_mix.all_methods == {LINDSAY_BROMLEY, SIMPLE}
    assert kg_mix.ranked_methods == [LINDSAY_BROMLEY, SIMPLE]
    
    # set a method
    kg_mix.set_user_method([SIMPLE])
    assert None == kg_mix.method
    k = kg_mix.mixture_property(m2.T, m2.P, m2.zs, m2.ws)
    assert_allclose(k, 0.02586655464213776)
    
    # Unhappy paths
    with pytest.raises(Exception):
        kg_mix.calculate(m2.T, m2.P, m2.zs, m2.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        kg_mix.test_method_validity(m2.T, m2.P, m2.zs, m2.ws, 'BADMETHOD')



def test_ThermalConductivityLiquidMixture():
    from thermo.thermal_conductivity import MAGOMEDOV, DIPPR_9H, FILIPPOV, SIMPLE, ThermalConductivityLiquidMixture

    m = Mixture(['ethanol', 'pentanol'], ws=[0.258, 0.742], T=298.15)
    ThermalConductivityLiquids = [i.ThermalConductivityLiquid for i in m.Chemicals]
    
    kl_mix = ThermalConductivityLiquidMixture(CASs=m.CASs, ThermalConductivityLiquids=ThermalConductivityLiquids)
    k = kl_mix.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(k, 0.15300152782218343)
                           
    k = kl_mix.calculate(m.T, m.P, m.zs, m.ws, FILIPPOV)
    assert_allclose(k, 0.15522139770330717)
    
    k = kl_mix.calculate(m.T, m.P, m.zs, m.ws, SIMPLE)
    assert_allclose(k, 0.1552717795028546)
    
    # Test electrolytes
    m = Mixture(['water', 'sulfuric acid'], ws=[.5, .5], T=298.15)
    ThermalConductivityLiquids = [i.ThermalConductivityLiquid for i in m.Chemicals]
    kl_mix = ThermalConductivityLiquidMixture(CASs=m.CASs, ThermalConductivityLiquids=ThermalConductivityLiquids)
    k = kl_mix.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(k, 0.4677453168207703)
    assert kl_mix.sorted_valid_methods == [MAGOMEDOV]
                     

    # Unhappy paths
    with pytest.raises(Exception):
        kl_mix.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        kl_mix.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

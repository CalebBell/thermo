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
from fluids.numerics import assert_close, assert_close1d
from thermo.thermal_conductivity import *
from thermo.mixture import Mixture
from thermo.thermal_conductivity import MAGOMEDOV, DIPPR_9H, FILIPPOV, SIMPLE, ThermalConductivityLiquidMixture
from thermo.thermal_conductivity import (GHARAGHEIZI_G, CHUNG, ELI_HANLEY, VDI_PPDS,
                                        ELI_HANLEY_DENSE, CHUNG_DENSE,
                                        EUCKEN_MOD, EUCKEN, BAHADORI_G,
                                        STIEL_THODOS_DENSE, DIPPR_9B, COOLPROP,
                                        DIPPR_PERRY_8E, VDI_TABULAR, GHARAGHEIZI_L,
                                       SATO_RIEDEL, NICOLA, NICOLA_ORIGINAL,
                                       SHEFFY_JOHNSON, BAHADORI_L,
                                       LAKSHMI_PRASAD, DIPPR_9G, MISSENARD)



@pytest.mark.meta_T_dept
def test_ThermalConductivityLiquid():
    EtOH = ThermalConductivityLiquid(CASRN='64-17-5', MW=46.06844, Tm=159.05, Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, Hfus=4931.0)

    EtOH.T_dependent_property(305.)
    all_methods = EtOH.available_methods
    methods = list(all_methods)
    methods.remove(VDI_TABULAR)
    kl_calcs = [(EtOH.set_method(i), EtOH.T_dependent_property(305.))[1] for i in methods]
    kl_exp = [0.162183005823234, 0.16627999999999998, 0.166302, 0.20068212675966418, 0.18526367184633258, 0.18846433785041306, 0.16837295487233528, 0.16883011582627103, 0.09330268101157643, 0.028604363267557775]
    assert_allclose(sorted(kl_calcs), sorted(kl_exp))

    assert_allclose(EtOH.calculate(305., VDI_TABULAR), 0.17417420086033197, rtol=1E-4)


    # Test that methods return None
    kl_calcs = [(EtOH.set_method(i), EtOH.T_dependent_property(5000))[1] for i in all_methods]
    assert [None]*11 == kl_calcs

    EtOH.set_method(VDI_TABULAR)
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
    all_methods = EtOH.available_methods
    kg_calcs = [(EtOH.set_method(i), EtOH.T_dependent_property(298.15))[1] for i in all_methods]
    kg_exp = [0.015227631457903644, 0.015025094773729045, 0.01520257225203181, 0.01494275, 0.016338750949017277, 0.014353317470206847, 0.011676848981094841, 0.01137910777526855, 0.015427444948536088, 0.012984129385510995, 0.017556325226536728]
    assert_allclose(sorted(kg_calcs), sorted(kg_exp))

    # Test that those mthods which can, do, return NoneEtOH.forced_P
    kg_calcs = [(EtOH.set_method(i), EtOH.T_dependent_property(5E20))[1] for i in [COOLPROP, DIPPR_PERRY_8E, VDI_TABULAR, GHARAGHEIZI_G, ELI_HANLEY, BAHADORI_G, VDI_PPDS]]
    assert [None]*7 == kg_calcs

    # Test tabular limits/extrapolation
    EtOH.set_method(VDI_TABULAR)
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
    assert_allclose(EtOH.calculate_P(298.15, 1E6, CHUNG_DENSE), 0.011770368783141446)
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


def test_ThermalConductivityGasMixture():
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

    kl_mix = ThermalConductivityLiquidMixture(CASs=m.CASs, ThermalConductivityLiquids=ThermalConductivityLiquids, MWs=m.MWs)
    k = kl_mix.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(k, 0.15300152782218343)

    k = kl_mix.calculate(m.T, m.P, m.zs, m.ws, FILIPPOV)
    assert_allclose(k, 0.15522139770330717)

    k = kl_mix.calculate(m.T, m.P, m.zs, m.ws, SIMPLE)
    assert_allclose(k, 0.1552717795028546)

    # Test electrolytes
    m = Mixture(['water', 'sulfuric acid'], ws=[.5, .5], T=298.15)
    ThermalConductivityLiquids = [i.ThermalConductivityLiquid for i in m.Chemicals]
    kl_mix = ThermalConductivityLiquidMixture(CASs=m.CASs, ThermalConductivityLiquids=ThermalConductivityLiquids, MWs=m.MWs)
    k = kl_mix.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(k, 0.4677453168207703)
    assert kl_mix.sorted_valid_methods == [MAGOMEDOV]


    # Unhappy paths
    with pytest.raises(Exception):
        kl_mix.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    with pytest.raises(Exception):
        kl_mix.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

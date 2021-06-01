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

import json
import pytest
from fluids.numerics import assert_close, assert_close1d, assert_close2d, linspace
from fluids.constants import R
from chemicals.utils import ws_to_zs
import chemicals
from thermo.thermal_conductivity import *
from thermo.viscosity import ViscosityGas
from thermo.mixture import Mixture
from thermo.coolprop import has_CoolProp
from thermo.thermal_conductivity import MAGOMEDOV, DIPPR_9H, FILIPPOV, LINEAR, ThermalConductivityLiquidMixture
from thermo.thermal_conductivity import (GHARAGHEIZI_G, CHUNG, ELI_HANLEY, VDI_PPDS,
                                        ELI_HANLEY_DENSE, CHUNG_DENSE,
                                        EUCKEN_MOD, EUCKEN, BAHADORI_G,
                                        STIEL_THODOS_DENSE, DIPPR_9B, COOLPROP,
                                        DIPPR_PERRY_8E, VDI_TABULAR, GHARAGHEIZI_L,
                                       SATO_RIEDEL, NICOLA, NICOLA_ORIGINAL,
                                       SHEFFY_JOHNSON, BAHADORI_L,
                                       LAKSHMI_PRASAD, DIPPR_9G, MISSENARD)
from thermo.thermal_conductivity import ThermalConductivityGasMixture, LINDSAY_BROMLEY, LINEAR



@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_ThermalConductivityLiquid_CoolProp():
    EtOH = ThermalConductivityLiquid(CASRN='64-17-5', MW=46.06844, Tm=159.05, Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, Hfus=4931.0)

    EtOH.method = COOLPROP
    assert_close(EtOH.T_dependent_property(305.), 0.162183005823234)

    assert_close(EtOH.calculate_P(298.15, 1E6, COOLPROP), 0.1639626989794703)
    assert [False, True] == [EtOH.test_method_validity_P(300, P, COOLPROP) for P in (1E3, 1E5)]

@pytest.mark.meta_T_dept
def test_ThermalConductivityLiquid():
    EtOH = ThermalConductivityLiquid(CASRN='64-17-5', MW=46.06844, Tm=159.05, Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, Hfus=4931.0)

    EtOH.method = NICOLA
    assert_close(EtOH.T_dependent_property(305.), 0.18846433785041308)
    EtOH.method = LAKSHMI_PRASAD
    assert_close(EtOH.T_dependent_property(305.), 0.028604363267557775)
    EtOH.method = SHEFFY_JOHNSON
    assert_close(EtOH.T_dependent_property(305.), 0.16883011582627103)
    EtOH.method = SATO_RIEDEL
    assert_close(EtOH.T_dependent_property(305.), 0.18526367184633263)
    EtOH.method = VDI_PPDS
    assert_close(EtOH.T_dependent_property(305.), 0.166302)
    EtOH.method = DIPPR_PERRY_8E
    assert_close(EtOH.T_dependent_property(305.), 0.16627999999999998)
    EtOH.method = VDI_TABULAR
    assert_close(EtOH.T_dependent_property(305.), 0.17418277049234407)
    EtOH.method = GHARAGHEIZI_L
    assert_close(EtOH.T_dependent_property(305.), 0.2006821267600352)
    EtOH.method = BAHADORI_L
    assert_close(EtOH.T_dependent_property(305.), 0.09330268101157693)
    EtOH.method = NICOLA_ORIGINAL
    assert_close(EtOH.T_dependent_property(305.), 0.16837295487233528)

    assert_close(EtOH.calculate(305., VDI_TABULAR), 0.17417420086033197, rtol=1E-4)


    # Test that methods return None
    EtOH.extrapolation = None
    for i in EtOH.all_methods:
        EtOH.method = i
        assert EtOH.T_dependent_property(5000) is None

    EtOH.method = VDI_TABULAR
    EtOH.extrapolation = 'interp1d'
    assert_close(EtOH.T_dependent_property(600.), 0.040117737789202995)
    EtOH.extrapolation = None
    assert None == EtOH.T_dependent_property(600.)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    assert ThermalConductivityLiquid.from_json(EtOH.as_json()) == EtOH

    # Ethanol compressed
    assert [True, True] == [EtOH.test_method_validity_P(300, P, DIPPR_9G) for P in (1E3, 1E5)]
    assert [True, True, False] == [EtOH.test_method_validity_P(300, P, MISSENARD) for P in (1E3, 1E5, 1E10)]

    EtOH.method = DIPPR_PERRY_8E
    assert_close(EtOH.calculate_P(298.15, 1E6, DIPPR_9G), 0.16512516068013278)
    assert_close(EtOH.calculate_P(298.15, 1E6, MISSENARD), 0.1687682040600248)


    # Ethanol data, calculated from CoolProp
    Ts = [275, 300, 350]
    Ps = [1E5, 5E5, 1E6]
    TP_data = [[0.16848555706973622, 0.16313525757474362, 0.15458068887966378], [0.16868861153075654, 0.163343255114212, 0.1548036152853355], [0.16894182645698885, 0.1636025336196736, 0.15508116339039268]]
    EtOH.add_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_close1d(TP_data, recalc_pts)

    assert_close(EtOH.TP_dependent_property(274, 9E4), 0.16848555706973622)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.TP_dependent_property(300, 9E4)

    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')

    assert False == EtOH.test_method_validity_P(-10, 1E5, DIPPR_9G)

    assert ThermalConductivityLiquid.from_json(EtOH.as_json()) == EtOH

    # Hash checks
    hash0 = hash(EtOH)
    EtOH2 = ThermalConductivityLiquid.from_json(json.loads(json.dumps(EtOH.as_json())))

    assert EtOH == EtOH2
    assert hash(EtOH) == hash0
    assert hash(EtOH2) == hash0

    EtOH2 = eval(str(EtOH))
    assert EtOH == EtOH2
    assert hash(EtOH) == hash0
    assert hash(EtOH2) == hash0

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_ThermalConductivityLiquid_fitting0():
    ammonia_Ts_kg = [200, 236.842, 239.82, 273.684, 310.526, 347.368, 384.211, 421.053, 457.895, 494.737, 531.579, 568.421, 605.263, 642.105, 678.947, 715.789, 752.632, 789.474, 826.316, 863.158, 900, ]
    ammonia_kgs = [0.0146228, 0.0182568, 0.0185644, 0.0221923, 0.0263847, 0.0308034, 0.0354254, 0.040233, 0.0452116, 0.0503493, 0.0556358, 0.0610625, 0.0666218, 0.072307, 0.0781123, 0.0840323, 0.0900624, 0.0961983, 0.102436, 0.108772, 0.115203]
    
    # note: equation does not yet support numba
    fit, res = ThermalConductivityLiquid.fit_data_to_model(Ts=ammonia_Ts_kg, data=ammonia_kgs, model='DIPPR102',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert res['MAE'] < 1e-5

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_ThermalConductivityLiquid_fitting1_dippr100():
    for i, CAS in enumerate(chemicals.thermal_conductivity.k_data_Perrys_8E_2_315.index):
        obj = ThermalConductivityLiquid(CASRN=CAS)
        Ts = linspace(obj.Perrys2_315_Tmin, obj.Perrys2_315_Tmax, 10)
        props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR100',
                                           do_statistics=True, use_numba=False, fit_method='lm')
        assert stats['MAE'] < 1e-8

@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_ThermalConductivityGas_CoolProp():
    EtOH = ThermalConductivityGas(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.2412, omega=0.635, dipole=1.44, Vmg=0.02357, Cpgm=56.98+R, mug=7.903e-6, CASRN='64-17-5')
    EtOH.method = COOLPROP
    assert_close(EtOH.T_dependent_property(305), 0.015870725750339945)

    assert [True, False] == [EtOH.test_method_validity_P(300, P, COOLPROP) for P in (1E3, 1E5)]
    assert_close(EtOH.calculate_P(298.15, 1E2, COOLPROP), 0.015207849649231962)

    assert False == EtOH.test_method_validity_P(100, 1E5, COOLPROP)

@pytest.mark.meta_T_dept
def test_ThermalConductivityGas():
    EtOH = ThermalConductivityGas(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.2412, omega=0.635, dipole=1.44, Vmg=0.02357, Cpgm=56.98+R, mug=7.903e-6, CASRN='64-17-5')
    all_methods = list(EtOH.all_methods)

    EtOH.method = EUCKEN_MOD
    assert_close(EtOH.T_dependent_property(305), 0.015427445804245578)
    EtOH.method = EUCKEN
    assert_close(EtOH.T_dependent_property(305), 0.012984130473277289)
    EtOH.method = VDI_PPDS
    assert_close(EtOH.T_dependent_property(305), 0.015661846372995)
    EtOH.method = BAHADORI_G
    assert_close(EtOH.T_dependent_property(305), 0.018297587287579457)
    EtOH.method = GHARAGHEIZI_G
    assert_close(EtOH.T_dependent_property(305), 0.016862968023145547)
    EtOH.method = DIPPR_9B
    assert_close(EtOH.T_dependent_property(305), 0.014372770946906635)
    EtOH.method = ELI_HANLEY
    assert_close(EtOH.T_dependent_property(305), 0.011684946002735508)
    EtOH.method = VDI_TABULAR
    assert_close(EtOH.T_dependent_property(305), 0.015509857659914554)
    EtOH.method = CHUNG
    assert_close(EtOH.T_dependent_property(305), 0.011710616856383785)
    EtOH.method = DIPPR_PERRY_8E
    assert_close(EtOH.T_dependent_property(305), 0.015836254853225484)


    EtOH.extrapolation = None

    for i in (DIPPR_PERRY_8E, VDI_TABULAR, GHARAGHEIZI_G, ELI_HANLEY, BAHADORI_G, VDI_PPDS):
        EtOH.method = i
        assert EtOH.T_dependent_property(5E20) is None

    # Test tabular limits/extrapolation
    EtOH.method = VDI_TABULAR
    EtOH.extrapolation = 'interp1d'
    assert_close(EtOH.T_dependent_property(600.), 0.05755089974293061)

    EtOH.extrapolation = None
    assert None == EtOH.T_dependent_property(600.)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')


    # Ethanol compressed

    assert [True, False] == [EtOH.test_method_validity_P(300, P, ELI_HANLEY_DENSE) for P in (1E5, -1E5)]
    assert [True, False] == [EtOH.test_method_validity_P(300, P, CHUNG_DENSE) for P in (1E5, -1E5)]
    assert [True, False] == [EtOH.test_method_validity_P(300, P, STIEL_THODOS_DENSE) for P in (1E5, -1E5)]

    assert ThermalConductivityGas.from_json(EtOH.as_json()) == EtOH

    EtOH = ThermalConductivityGas(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.2412, omega=0.635, dipole=1.44, Vmg=0.02357, Cpgm=56.98+R, mug=7.903e-6, CASRN='64-17-5')
    assert_close(EtOH.calculate_P(298.15, 1E6, ELI_HANLEY_DENSE), 0.011210125242396791)
    assert_close(EtOH.calculate_P(298.15, 1E6, CHUNG_DENSE), 0.011770368783141446)
    EtOH.method = DIPPR_PERRY_8E
    assert_close(EtOH.calculate_P(298.15, 1E6, STIEL_THODOS_DENSE), 0.015422777479549062)


      # Ethanol data, calculated from CoolProp
    Ts = [400, 500, 600]
    Ps = [1E4, 1E5, 2E5]
    TP_data = [[0.025825794817543015, 0.037905383602635095, 0.05080124980338535], [0.02601702567554805, 0.03806794452306919, 0.050946301396380594], [0.026243171168075605, 0.03825284803978187, 0.05110925652065333]]
    EtOH.add_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_close2d(TP_data, recalc_pts)

    EtOH.tabular_extrapolation_permitted = True
    assert_close(EtOH.TP_dependent_property(399, 9E3), 0.025825794817543015)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.TP_dependent_property(399, 9E3)

    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')

    assert ThermalConductivityGas.from_json(EtOH.as_json()) == EtOH

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_ThermalConductivityGas_fitting0():
    try_CASs_dippr = ['106-97-8', '142-29-0', '565-59-3', '74-83-9', '115-21-9', 
                      '78-78-4', '111-27-3', '74-90-8', '64-17-5', '50-00-0', '74-98-6',
                      '7664-39-3', '107-21-1', '592-76-7', '115-07-1', '64-19-7']
    for CAS in try_CASs_dippr:
        obj = ThermalConductivityGas(CASRN=CAS)
        Ts = linspace(obj.Perrys2_314_Tmin, obj.Perrys2_314_Tmax, 10)
        props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR102', multiple_tries=True,
                              do_statistics=True, use_numba=False, fit_method='lm')
        assert stats['MAE'] < 1e-5
    
def test_ThermalConductivityGasMixture():
    T, P = 298.15, 101325.0
    MWs = [28.0134, 39.948, 31.9988]
    Tbs = [77.355, 87.302, 90.188]
    CASs = ['7727-37-9', '7440-37-1', '7782-44-7']
    ws=[0.7557, 0.0127, 0.2316]
    zs = ws_to_zs(ws, MWs=MWs)
    # ['nitrogen', 'argon', 'oxygen']

    ViscosityGases = [
        ViscosityGas(CASRN="7727-37-9", MW=28.0134, Tc=126.2, Pc=3394387.5, Zc=0.2895282296391198, dipole=0.0, extrapolation="linear", method=DIPPR_PERRY_8E, method_P=None),
        ViscosityGas(CASRN="7440-37-1", MW=39.948, Tc=150.8, Pc=4873732.5, Zc=0.29114409080360165, dipole=0.0, extrapolation="linear", method=DIPPR_PERRY_8E, method_P=None),
        ViscosityGas(CASRN="7782-44-7", MW=31.9988, Tc=154.58, Pc=5042945.25, Zc=0.2880002236716698, dipole=0.0, extrapolation="linear", method=DIPPR_PERRY_8E, method_P=None),
    ]
    ThermalConductivityGases = [
        ThermalConductivityGas(CASRN="7727-37-9", MW=28.0134, Tb=77.355, Tc=126.2, Pc=3394387.5, Vc=8.95e-05, Zc=0.2895282296391198, omega=0.04, dipole=0.0, extrapolation="linear", method=VDI_PPDS, method_P=ELI_HANLEY_DENSE),
        ThermalConductivityGas(CASRN="7440-37-1", MW=39.948, Tb=87.302, Tc=150.8, Pc=4873732.5, Vc=7.49e-05, Zc=0.29114409080360165, omega=-0.004, dipole=0.0, extrapolation="linear", method=VDI_PPDS, method_P=ELI_HANLEY_DENSE),
        ThermalConductivityGas(CASRN="7782-44-7", MW=31.9988, Tb=90.188, Tc=154.58, Pc=5042945.25, Vc=7.34e-05, Zc=0.2880002236716698, omega=0.021, dipole=0.0, extrapolation="linear", method=VDI_PPDS, method_P=ELI_HANLEY_DENSE),
        ]

    kg_mix = ThermalConductivityGasMixture(MWs=MWs, Tbs=Tbs, CASs=CASs, correct_pressure_pure=False,
                                      ThermalConductivityGases=ThermalConductivityGases,
                                      ViscosityGases=ViscosityGases)

    kg_mix.method = LINEAR
    kg_expect = 0.025593922564292677
    kg = kg_mix.mixture_property(T, P, zs, ws)
    assert_close(kg, kg_expect, rtol=1e-13)
    kg = kg_mix.mixture_property(T, P, zs, ws)
    assert_close(kg, kg_expect, rtol=1e-13)

    kg_mix.method = LINDSAY_BROMLEY
    kg_expect = 0.025588076672276125
    kg = kg_mix.mixture_property(T, P, zs, ws)
    assert_close(kg, kg_expect, rtol=1e-13)

    dT1 = kg_mix.calculate_derivative_T(T, P, zs, ws, LINDSAY_BROMLEY)
    dT2 = kg_mix.property_derivative_T(T, P, zs, ws)
    assert_close1d([dT1, dT2], [7.456379610970565e-05]*2)

    dP1 = kg_mix.calculate_derivative_P(T, P, zs, ws, LINDSAY_BROMLEY)
    dP2 = kg_mix.property_derivative_P(T, P, zs, ws)
    assert_close1d([dP1, dP2], [0]*2, rtol=1E-4)

    # Unhappy paths
    with pytest.raises(Exception):
        kg_mix.calculate(m2.T, m2.P, m2.zs, m2.ws, 'BADMETHOD')

    with pytest.raises(Exception):
        kg_mix.test_method_validity(m2.T, m2.P, m2.zs, m2.ws, 'BADMETHOD')

    # json
    hash0 = hash(kg_mix)
    kg_mix2 = ThermalConductivityGasMixture.from_json(json.loads(json.dumps(kg_mix.as_json())))
    assert kg_mix == kg_mix2
    assert hash(kg_mix) == hash0
    assert hash(kg_mix2) == hash0

    kg_mix2 = eval(str(kg_mix))
    assert kg_mix == kg_mix2
    assert hash(kg_mix) == hash0
    assert hash(kg_mix2) == hash0


def test_ThermalConductivityLiquidMixture():
    T, P = 298.15, 101325.0
    ws = [0.258, 0.742]
    MWs = [46.06844, 88.14818]
    CASs = ['64-17-5', '71-41-0']
    # ['ethanol', 'pentanol']
    zs = ws_to_zs(ws=ws, MWs=MWs)
    ThermalConductivityLiquids = [
        ThermalConductivityLiquid(CASRN="64-17-5", MW=46.06844, Tm=159.05, Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, Hfus=4931.0, extrapolation="linear", method="DIPPR_PERRY_8E", method_P="DIPPR_9G"),
        ThermalConductivityLiquid(CASRN="71-41-0", MW=88.14818, Tm=194.7, Tb=410.75, Tc=588.1, Pc=3897000.0, omega=0.58, Hfus=10500.0, extrapolation="linear", method="DIPPR_PERRY_8E", method_P="DIPPR_9G")
    ]
    kl_mix = ThermalConductivityLiquidMixture(CASs=CASs, ThermalConductivityLiquids=ThermalConductivityLiquids, MWs=MWs)
    kl_mix.method = DIPPR_9H
    k = kl_mix.mixture_property(T, P, zs, ws)
    assert_close(k, 0.15326768195303517, rtol=1e-13)

    k = kl_mix.calculate(T, P, zs, ws, FILIPPOV)
    assert_close(k, 0.15572417368293207)

    k = kl_mix.calculate(T, P, zs, ws, LINEAR)
    assert_close(k, 0.1557792273701113)

    # Unhappy paths
    with pytest.raises(Exception):
        kl_mix.calculate(T, P, zs, ws, 'BADMETHOD')

    with pytest.raises(Exception):
        kl_mix.test_method_validity(T, P, zs, ws, 'BADMETHOD')

    # json export
    hash0 = hash(kl_mix)
    kl_mix2 = ThermalConductivityLiquidMixture.from_json(json.loads(json.dumps(kl_mix.as_json())))
    assert kl_mix == kl_mix2
    assert hash(kl_mix) == hash0
    assert hash(kl_mix2) == hash0

    kl_mix2 = eval(str(kl_mix))
    assert kl_mix == kl_mix2
    assert hash(kl_mix) == hash0
    assert hash(kl_mix2) == hash0

def test_ThermalConductivityLiquidMixture_electrolytes():
    # Test electrolytes
    # m = Mixture(['water', 'sulfuric acid'], ws=[.5, .5], T=298.15)
    T, P = 298.15, 101325.0
    ws = [0.5, 0.5]
    MWs = [18.01528, 98.07848]
    zs = ws_to_zs(ws, MWs)
    CASs = ['7732-18-5', '7664-93-9']
    ThermalConductivityLiquids = [
        ThermalConductivityLiquid(CASRN="7732-18-5", MW=18.01528, Tm=273.15, Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, Hfus=6010.0, extrapolation="linear", method=DIPPR_PERRY_8E, method_P=DIPPR_9G),
        ThermalConductivityLiquid(CASRN="7664-93-9", MW=98.07848, Tm=277.305, Tb=610.15, Tc=924.0, Pc=6400000.0, omega=0.494, Hfus=10710.0, extrapolation="linear", method=GHARAGHEIZI_L, method_P=DIPPR_9G)
    ]
    kl_mix = ThermalConductivityLiquidMixture(CASs=CASs, ThermalConductivityLiquids=ThermalConductivityLiquids, MWs=MWs)
    kl_mix.method == MAGOMEDOV
    k = kl_mix.mixture_property(T, P, zs, ws)
    assert_close(k, 0.45824995874859015, rtol=1e-13)

    # json export
    hash0 = hash(kl_mix)
    kl_mix2 = ThermalConductivityLiquidMixture.from_json(json.loads(json.dumps(kl_mix.as_json())))
    assert kl_mix == kl_mix2
    assert hash(kl_mix) == hash0
    assert hash(kl_mix2) == hash0

    kl_mix2 = eval(str(kl_mix))
    assert kl_mix == kl_mix2
    assert hash(kl_mix) == hash0
    assert hash(kl_mix2) == hash0

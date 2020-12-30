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

from fluids.numerics import assert_close, assert_close1d
from numpy.testing import assert_allclose
import numpy as np
import pytest
import pandas as pd
from thermo.volume import *
from thermo.volume import VDI_TABULAR
from thermo.eos import *
from chemicals.utils import Vm_to_rho


@pytest.mark.meta_T_dept
def test_VolumeGas():
    eos = [PR(T=300, P=1E5, Tc=430.8, Pc=7884098.25, omega=0.251)]
    SO2 = VolumeGas(CASRN='7446-09-5', MW=64.0638,  Tc=430.8, Pc=7884098.25, omega=0.251, dipole=1.63, eos=eos)

    Vm_calcs = []
    for i in SO2.all_methods_P:
        SO2.method_P = i
        Vm_calcs.append(SO2.TP_dependent_property(305, 1E5))


    Vm_exp = [0.025024302563892417, 0.02499978619699621, 0.02499586901117375, 0.02499627309459868, 0.02499978619699621, 0.024971467450477493, 0.02535910239]
    assert_allclose(sorted(Vm_calcs), sorted(Vm_exp), rtol=1e-5)

    # Test that methods return None
    for i in SO2.all_methods_P:
        SO2.method_P = i
        assert SO2.TP_dependent_property(-100, 1E5) is None
        assert SO2.TP_dependent_property(100, -1E5) is None

    with pytest.raises(Exception):
        SO2.test_method_validity_P(300, 1E5, 'BADMETHOD')



    # Ethanol data, calculated from CoolProp
    EtOH = VolumeGas(MW=46.06844, Tc=514.0, Pc=6137000.0, omega=0.635, dipole=1.44, CASRN='64-17-5')
    Ts = [400, 500, 600]
    Ps = [5E3, 1E4, 5E4]

    TP_data = [[0.6646136629870959, 0.8312205608372635, 0.9976236498685168], [0.33203434186506636, 0.4154969056659669, 0.49875532241189763], [0.06596765735649, 0.08291758227639608, 0.09966060658661156]]
    EtOH.set_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_allclose(TP_data, recalc_pts)

    EtOH.forced_P = True
    assert_allclose(EtOH.TP_dependent_property(300, 9E4), 0.06596765735649)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.TP_dependent_property(300, 9E4)

    # Test CRC Virial data
    H2 = VolumeGas(CASRN='1333-74-0')
    H2.method_P = 'CRC_VIRIAL'
    assert_allclose(H2.TP_dependent_property(300, 1E5), 0.024958843346854165)



@pytest.mark.meta_T_dept
def test_VolumeLiquid():
    # Ethanol, test all methods at once
    EtOH = VolumeLiquid(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, dipole=1.44, CASRN='64-17-5')
    methods = list(EtOH.all_methods)
    methods.remove(VDI_TABULAR)

    Vm_calcs = []
    for i in methods:
        EtOH.method = i
        Vm_calcs.append(EtOH.T_dependent_property(305.))

    Vm_exp = [5.905316741206586e-05, 5.784760660832295e-05, 5.7594571728502063e-05, 5.594757794216803e-05, 5.912157674597306e-05, 5.9082910221835385e-05, 5.526836182702171e-05, 5.821947224585489e-05, 5.1921776627430897e-05, 5.9680793094807483e-05, 5.4848470492414296e-05, 5.507075716132008e-05, 5.3338182234795054e-05]
    assert_allclose(sorted(Vm_calcs), sorted(Vm_exp), rtol=1e-5)
    assert_allclose(EtOH.calculate(305, VDI_TABULAR), 5.909768693432104e-05,rtol=1E-4)

    # Test that methods return None
    EtOH = VolumeLiquid(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, dipole=1.44, CASRN='64-17-5')
    EtOH.T_dependent_property(305.) # Initialize the sorted_valid_methods
    EtOH.tabular_extrapolation_permitted = False

    Vml_calcs = []
    for i in list(EtOH.all_methods):
        EtOH.method = i
        Vml_calcs.append(EtOH.T_dependent_property(600))

    assert [None]*14 == Vml_calcs

    EtOH.method = 'VDI_TABULAR'
    EtOH.tabular_extrapolation_permitted = True
    assert_allclose(EtOH.T_dependent_property(700.), 0.0005648005718236466)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    SnCl4 = VolumeLiquid(CASRN='7646-78-8')
    Vm_calcs = []
    for i in SnCl4.all_methods:
        SnCl4.method = i
        Vm_calcs.append(SnCl4.T_dependent_property(305.))

    # Get MMC parameter
    SO2 = VolumeLiquid(MW=64.0638, Tb=263.1, Tc=430.8, Pc=7884098.25, Vc=0.000122, Zc=0.26853, omega=0.251, dipole=1.63, CASRN='7446-09-5')
    Vm_calcs = []
    for i in SO2.all_methods:
        SO2.method = i
        Vm_calcs.append(SO2.T_dependent_property(200.))

    Vm_exp = [3.9697664371887463e-05, 3.748481829074182e-05, 4.0256041843356724e-05, 3.982522377343308e-05, 4.062166881078707e-05, 4.0608189210203123e-05, 3.949103647364349e-05, 3.994849780626379e-05, 4.109189955368007e-05, 3.965944731935354e-05, 4.0948267317531393e-05, 4.0606869929178414e-05, 4.060446067691708e-05, 3.993451478384902e-05]
    assert_allclose(sorted(Vm_calcs), sorted(Vm_exp), rtol=3e-5)

    # Get CRC Inorganic invalid
    U = VolumeLiquid(CASRN='7440-61-1')
    assert_allclose(U.T_dependent_property(1420), 1.3758901734104049e-05)
    assert False == U.test_method_validity(1430, U.method)

    # Test lower limit for BHIRUD_NORMAL
    fake = VolumeLiquid(Tc=1000)
    assert False == fake.test_method_validity(349.5, 'BHIRUD_NORMAL')

    # Ethanol compressed
    EtOH = VolumeLiquid(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, dipole=1.44, CASRN='64-17-5', Psat=7882.16)

    assert [False, True] == [EtOH.test_method_validity_P(300, P, 'COOLPROP') for P in (1E3, 1E5)]
    assert [True, True] == [EtOH.test_method_validity_P(300, P, 'COSTALD_COMPRESSED') for P in (1E3, 1E5)]

    assert_allclose(EtOH.calculate_P(298.15, 1E6, 'COSTALD_COMPRESSED'), 5.859498172626399e-05)
    assert_allclose(EtOH.calculate_P(298.15, 1E6, 'COOLPROP'), 5.861494869071554e-05)


    # Ethanol data, calculated from CoolProp
    Ts = [275, 300, 350]
    Ps = [1E5, 5E5, 1E6]
    TP_data = [[5.723868454722602e-05, 5.879532690584185e-05, 6.242643879647073e-05], [5.721587962307716e-05, 5.8767676310784456e-05, 6.238352991918547e-05], [5.718753361659462e-05, 5.8733341196316644e-05, 6.23304095312013e-05]]
    EtOH.set_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_allclose(TP_data, recalc_pts)

    EtOH.forced_P = True
    assert_allclose(EtOH.TP_dependent_property(274, 9E4), 5.723868454722602e-05)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.TP_dependent_property(300, 9E4)


    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')

@pytest.mark.meta_T_dept
def test_VolumeLiquidPolynomialTmin():
    # toluene
    v = VolumeLiquid(poly_fit=(178.01, 581.75, [2.2801490297347937e-23, -6.411956871696508e-20, 7.723152902379232e-17, -5.197203733189603e-14, 2.1348482785660093e-11, -5.476649499770259e-09, 8.564670053875876e-07, -7.455178589434267e-05, 0.0028545812080104068]))
    assert v._Tmin_T_trans == v.poly_fit_Tmin
    assert_allclose(v.poly_fit_Tmin_quadratic, [2.159638355215081e-10, 0.0, 8.76710398245817e-05])

    # methanol
    v = VolumeLiquid(poly_fit=(175.7, 502.5, [3.5725079384600736e-23, -9.031033742820083e-20, 9.819637959370411e-17, -5.993173551565636e-14, 2.2442465416964825e-11, -5.27776114586072e-09, 7.610461006178106e-07, -6.148574498547711e-05, 0.00216398089328537]))
    assert v._Tmin_T_trans == v.poly_fit_Tmin
    assert_allclose(v.poly_fit_Tmin_quadratic, [9.044411973585966e-11, 0.0, 3.2636013401752355e-05])

    # furufryl alcohol
    v = VolumeLiquid(poly_fit=(189.6, 568.8000000000001, [6.647049391841594e-25, -1.8278965754470383e-21, 2.170226311490262e-18, -1.4488205117650477e-15, 5.940737543663388e-13, -1.529528764741801e-10, 2.4189938181664235e-08, -2.0920414326130236e-06, 0.00014008105452139704]))
    assert v._Tmin_T_trans == v.poly_fit_Tmin
    assert_allclose(v.poly_fit_Tmin_quadratic, [1.0450567029924047e-10, 2.3832444455827036e-08, 6.0795970965526267e-05])

    # Water - may change, second should always be zero
    v = VolumeLiquid(poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652]))
    assert_allclose(v.poly_fit_Tmin_quadratic,  [8.382589086490995e-12, 0.0, 1.739254665187681e-05])


@pytest.mark.meta_T_dept
def test_VolumeSolid():
    Vm = VolumeSolid(CASRN='10022-31-8').T_dependent_property(300)
    assert_allclose(Vm, 8.06592592592592e-05)

    assert None == VolumeSolid(CASRN='10022-31-8').T_dependent_property(-100)

    with pytest.raises(Exception):
        VolumeSolid(CASRN='10022-31-8').test_method_validity(200, 'BADMETHOD')

    BaN2O6 = VolumeSolid(CASRN='10022-31-8')
    BaN2O6.set_tabular_data([200,300], [8.06e-5, 8.05e-5], 'fake')
    assert_allclose(8.055e-05, BaN2O6.T_dependent_property(250))
    BaN2O6.tabular_extrapolation_permitted = False
    BaN2O6.test_method_validity(150, 'fake')

@pytest.mark.meta_T_dept
def test_VolumeSolid_works_with_no_data():
    # Test pentane has no property but that it succeeds
    VolumeSolid(CASRN='109-66-0').T_dependent_property(300)



@pytest.mark.meta_T_dept
def test_VolumeLiquidMixture():
    from thermo.mixture import Mixture
    from thermo.volume import LALIBERTE, COSTALD_MIXTURE_FIT, RACKETT_PARAMETERS, COSTALD_MIXTURE,  SIMPLE, RACKETT
    m = Mixture(['benzene', 'toluene'], zs=[.5, .5], T=298.15, P=101325.)

    VolumeLiquids = [i.VolumeLiquid for i in m.Chemicals]

    obj = VolumeLiquidMixture(MWs=m.MWs, Tcs=m.Tcs, Pcs=m.Pcs, Vcs=m.Vcs, Zcs=m.Zcs, omegas=m.omegas,
                              CASs=m.CASs, VolumeLiquids=VolumeLiquids)

    Vm = obj.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(Vm, 9.814092676573469e-05)

    Vms = [obj.calculate(m.T, m.P, m.zs, m.ws, method) for method in obj.all_methods]
    Vms_expect = [9.814092676573469e-05, 9.737758899339708e-05, 9.8109833265793461e-05, 9.8154006097783393e-05, 9.858773618507426e-05]
    assert_allclose(sorted(Vms), sorted(Vms_expect), rtol=1e-5)

    # Test Laliberte
    m = Mixture(['water', 'sulfuric acid'], zs=[0.01, 0.99], T=298.15)
    VolumeLiquids = [i.VolumeLiquid for i in m.Chemicals]
    obj = VolumeLiquidMixture(MWs=m.MWs, Tcs=m.Tcs, Pcs=m.Pcs, Vcs=m.Vcs, Zcs=m.Zcs, omegas=m.omegas,
                              CASs=m.CASs, VolumeLiquids=VolumeLiquids)

    Vm = obj.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(Vm, 4.824170609370422e-05)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    # Excess volume
    # Vs -7.5E-7 in ddbst http://www.ddbst.com/en/EED/VE/VE0%20Ethanol%3BWater.php
    drink = Mixture(['water', 'ethanol'], zs=[1- 0.15600,  0.15600], T=298.15, P=101325)
    V_Ex = drink.VolumeLiquidMixture.excess_property(drink.T, drink.P, drink.zs, drink.ws)
    assert_allclose(V_Ex, -7.242450496000289e-07, rtol=.05)

@pytest.mark.meta_T_dept
def test_VolumeGasMixture():
    from thermo.mixture import Mixture
    from thermo.volume import VolumeGasMixture, EOS, SIMPLE, IDEAL
    m = Mixture(['oxygen', 'nitrogen'], zs=[.5, .5], T=298.15, P=1E6)
    obj = VolumeGasMixture(CASs=m.CASs, VolumeGases=m.VolumeGases, eos=m.eos_in_a_box, MWs=m.MWs)

    # TODO FIX For now, this is broken; the molar volume of a gas is not being calculated
    # as there is no eos object being set to the mixture.
#    assert_allclose(obj.mixture_property(m.T, m.P, m.zs, m.ws), 0.0024628053244477232)
    assert_allclose(obj.calculate(m.T, m.P, m.zs, m.ws, SIMPLE), 0.002468989614515616)
    assert_allclose(obj.calculate(m.T, m.P, m.zs, m.ws, IDEAL), 0.002478957029602388)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


@pytest.mark.meta_T_dept
def test_VolumeSupercriticalLiquidMixture():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=700., P=1e8)
    obj2 = VolumeSupercriticalLiquid(eos=[eos], Tc=eos.Tc, Pc=eos.Pc, omega=eos.omega)
    V_implemented = obj2.calculate_P(T=700.0, P=1e8, method='EOS')
    assert_close(V_implemented, eos.V_l, rtol=1e-13)
    V_implemented = obj2.calculate_P(T=700.0, P=1e3, method='EOS')
#    assert V_implemented is None

    assert obj2.test_method_validity_P(T=700.0, P=1e8, method='EOS')
    assert not obj2.test_method_validity_P(T=700.0, P=1e4, method='EOS')

@pytest.mark.meta_T_dept
@pytest.mark.CoolProp
def test_VolumeSupercriticalLiquidMixtureCoolProp():
    from CoolProp.CoolProp import PropsSI
    obj = VolumeSupercriticalLiquid(CASRN='7732-18-5')
    V_implemented = obj.calculate_P(T=700.0, P=1e8, method='COOLPROP')
    V_CoolProp = 1/PropsSI('DMOLAR', 'T', 700, 'P', 1e8, 'water')
    assert_close(V_implemented, V_CoolProp, rtol=1e-13)
    assert obj.test_method_validity_P(T=700.0, P=1e8, method='COOLPROP')
    assert obj.test_method_validity_P(T=700.0, P=1e4, method='COOLPROP')

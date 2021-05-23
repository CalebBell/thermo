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

from fluids.numerics import assert_close, assert_close1d, assert_close2d, linspace
import numpy as np
import pytest
import json
import pandas as pd
from thermo.volume import *
from thermo.volume import VDI_TABULAR
from thermo.eos import *
import chemicals
from chemicals.utils import Vm_to_rho, zs_to_ws
from thermo.vapor_pressure import VaporPressure
from thermo.utils import POLY_FIT
from chemicals.volume import *
from thermo.volume import LALIBERTE, COSTALD_MIXTURE_FIT, RACKETT_PARAMETERS, COSTALD_MIXTURE, LINEAR, RACKETT
from thermo.volume import HTCOSTALD, COOLPROP, DIPPR_PERRY_8E, VDI_TABULAR, RACKETTFIT, YEN_WOODS_SAT, BHIRUD_NORMAL, VDI_PPDS, TOWNSEND_HALES, HTCOSTALDFIT, CAMPBELL_THODOS, MMSNM0, RACKETT, YAMADA_GUNN
from thermo.volume import PITZER_CURL, EOS, TSONOPOULOS_EXTENDED, ABBOTT, COOLPROP, IDEAL, TSONOPOULOS, CRC_VIRIAL
from thermo.volume import EOS, LINEAR, IDEAL
from thermo.eos_mix import PRMIX
from thermo.coolprop import has_CoolProp

@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_VolumeGas_CoolProp():
    SO2 = VolumeGas(CASRN='7446-09-5', MW=64.0638,  Tc=430.8, Pc=7884098.25, omega=0.251, dipole=1.63)
    SO2.method_P = COOLPROP
    assert_close(SO2.TP_dependent_property(305, 1E5), 0.024971467450477493, rtol=1e-13)

@pytest.mark.meta_T_dept
def test_VolumeGas():
    eos = [PR(T=300, P=1E5, Tc=430.8, Pc=7884098.25, omega=0.251)]
    SO2 = VolumeGas(CASRN='7446-09-5', MW=64.0638,  Tc=430.8, Pc=7884098.25, omega=0.251, dipole=1.63, eos=eos)

    SO2.method_P = ABBOTT
    assert_close(SO2.TP_dependent_property(305, 1E5), 0.024995877483424613, rtol=1e-13)

    SO2.method_P = PITZER_CURL
    assert_close(SO2.TP_dependent_property(305, 1E5), 0.024996281566986505, rtol=1e-13)

    SO2.method_P = EOS
    assert_close(SO2.TP_dependent_property(305, 1E5), 0.02502431104578071, rtol=1e-13)

    SO2.method_P = TSONOPOULOS_EXTENDED
    assert_close(SO2.TP_dependent_property(305, 1E5), 0.02499979467057479, rtol=1e-13)

    SO2.method_P = IDEAL
    assert_close(SO2.TP_dependent_property(305, 1E5), 0.02535911098536738, rtol=1e-13)


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
    EtOH.add_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_close2d(TP_data, recalc_pts)

    assert_close(EtOH.TP_dependent_property(300, 9E4), 0.06596765735649)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.TP_dependent_property(300, 9E4)

    # Test CRC Virial data
    H2 = VolumeGas(CASRN='1333-74-0')
    H2.method_P = CRC_VIRIAL
    assert_close(H2.TP_dependent_property(300, 1E5), 0.024958843346854165)

@pytest.mark.meta_T_dept
def test_VolumeGas_Tabular_json_eval():
    obj = VolumeGas(MW=46.06844, Tc=514.0, Pc=6137000.0, omega=0.635, dipole=1.44, CASRN='64-17-5')
    Ts = [400, 500, 600]
    Ps = [5E3, 1E4, 5E4]

    TP_data = [[0.6646136629870959, 0.8312205608372635, 0.9976236498685168], [0.33203434186506636, 0.4154969056659669, 0.49875532241189763], [0.06596765735649, 0.08291758227639608, 0.09966060658661156]]
    obj.add_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    hash0 = hash(obj)
    obj2 = eval(str(obj))
    assert obj2 == obj
    assert hash(obj) == hash0
    assert hash0 == hash(obj2)

    obj2 = VolumeGas.from_json(json.loads(json.dumps(obj.as_json())))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_VolumeLiquid_CoolProp():
    # Ethanol, test all methods at once
    EtOH = VolumeLiquid(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, dipole=1.44, CASRN='64-17-5', Psat=7882.16)
    EtOH.method = COOLPROP
    assert_close(EtOH.T_dependent_property(305.), 5.912157674597306e-05, rtol=1e-7)

    # Ethanol compressed
    assert [False, True] == [EtOH.test_method_validity_P(300, P, 'COOLPROP') for P in (1E3, 1E5)]
    assert [True, True] == [EtOH.test_method_validity_P(300, P, 'COSTALD_COMPRESSED') for P in (1E3, 1E5)]

    EtOH.method = DIPPR_PERRY_8E
    assert_close(EtOH.calculate_P(298.15, 1E6, 'COSTALD_COMPRESSED'), 5.859498172626399e-05)
    assert_close(EtOH.calculate_P(298.15, 1E6, 'COOLPROP'), 5.861494869071554e-05)



@pytest.mark.meta_T_dept
def test_VolumeLiquid():
    # Ethanol, test all methods at once
    EtOH = VolumeLiquid(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, dipole=1.44, CASRN='64-17-5')
    methods = list(EtOH.all_methods)
    methods.remove(VDI_TABULAR)

    EtOH.method = HTCOSTALD
    assert_close(EtOH.T_dependent_property(305.), 5.526836182702171e-05, rtol=1e-7)
    EtOH.method = DIPPR_PERRY_8E
    assert_close(EtOH.T_dependent_property(305.), 5.9082910221835385e-05, rtol=1e-7)
    EtOH.method = VDI_TABULAR
    assert_close(EtOH.T_dependent_property(305.), 5.9096619917816446e-05, rtol=1e-7)
    EtOH.method = RACKETTFIT
    assert_close(EtOH.T_dependent_property(305.), 5.9680813323376087e-05, rtol=1e-7)
    EtOH.method = YEN_WOODS_SAT
    assert_close(EtOH.T_dependent_property(305.), 5.7847606608322876e-05, rtol=1e-7)
    EtOH.method = BHIRUD_NORMAL
    assert_close(EtOH.T_dependent_property(305.), 5.5070775827335304e-05, rtol=1e-7)
    EtOH.method = VDI_PPDS
    assert_close(EtOH.T_dependent_property(305.), 5.9053167412065864e-05, rtol=1e-7)
    EtOH.method = TOWNSEND_HALES
    assert_close(EtOH.T_dependent_property(305.), 5.4848470492414296e-05, rtol=1e-7)
    EtOH.method = HTCOSTALDFIT
    assert_close(EtOH.T_dependent_property(305.), 5.759457172850207e-05, rtol=1e-7)
    EtOH.method = CAMPBELL_THODOS
    assert_close(EtOH.T_dependent_property(305.), 5.192179422611158e-05, rtol=1e-7)
    EtOH.method = MMSNM0
    assert_close(EtOH.T_dependent_property(305.), 5.821947224585489e-05, rtol=1e-7)
    EtOH.method = RACKETT
    assert_close(EtOH.T_dependent_property(305.), 5.594759690537813e-05, rtol=1e-7)
    EtOH.method = YAMADA_GUNN
    assert_close(EtOH.T_dependent_property(305.), 5.3338200313560806e-05, rtol=1e-7)

    # Test that methods return None
    EtOH = VolumeLiquid(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, dipole=1.44, CASRN='64-17-5')
    EtOH.T_dependent_property(305.) # Initialize the sorted_valid_methods
    EtOH.tabular_extrapolation_permitted = False
    EtOH.extrapolation = None

    for i in list(EtOH.all_methods):
        EtOH.method = i
        assert EtOH.T_dependent_property(600) is None

    EtOH.method = VDI_TABULAR
    EtOH.extrapolation = 'interp1d'
    assert_close(EtOH.T_dependent_property(700.), 0.0005648005718236466)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    SnCl4 = VolumeLiquid(CASRN='7646-78-8')
    Vm_calcs = []
    for i in SnCl4.all_methods:
        SnCl4.method = i
        Vm_calcs.append(SnCl4.T_dependent_property(305.))

    # Get MMC parameter
    SO2 = VolumeLiquid(MW=64.0638, Tb=263.1, Tc=430.8, Pc=7884098.25, Vc=0.000122, Zc=0.26853, omega=0.251, dipole=1.63, CASRN='7446-09-5', extrapolation=None)
    SO2.method = HTCOSTALD
    assert_close(SO2.T_dependent_property(200.), 4.062166881078707e-05, rtol=1e-7)
    SO2.method = DIPPR_PERRY_8E
    assert_close(SO2.T_dependent_property(200.), 3.965944731935354e-05, rtol=1e-7)
    SO2.method = RACKETTFIT
    assert_close(SO2.T_dependent_property(200.), 3.993452831949474e-05, rtol=1e-7)
    SO2.method = YEN_WOODS_SAT
    assert_close(SO2.T_dependent_property(200.), 4.109189955368029e-05, rtol=1e-7)
    SO2.method = BHIRUD_NORMAL
    assert_close(SO2.T_dependent_property(200.), 4.0604474439638224e-05, rtol=1e-7)
    SO2.method = VDI_PPDS
    assert_close(SO2.T_dependent_property(200.), 3.9697664371887463e-05, rtol=1e-7)
    SO2.method = TOWNSEND_HALES
    assert_close(SO2.T_dependent_property(200.), 4.0256041843356724e-05, rtol=1e-7)
    SO2.method = HTCOSTALDFIT
    assert_close(SO2.T_dependent_property(200.), 3.994849780626379e-05, rtol=1e-7)
    SO2.method = CAMPBELL_THODOS
    assert_close(SO2.T_dependent_property(200.), 3.7484830996072615e-05, rtol=1e-7)
    SO2.method = MMSNM0
    assert_close(SO2.T_dependent_property(200.), 4.0948267317531393e-05, rtol=1e-7)
    SO2.method = RACKETT
    assert_close(SO2.T_dependent_property(200.), 4.060688369271625e-05, rtol=1e-7)
    SO2.method = YAMADA_GUNN
    assert_close(SO2.T_dependent_property(200.), 4.0608202974188125e-05, rtol=1e-7)


    # Get CRC Inorganic invalid
    U = VolumeLiquid(CASRN='7440-61-1')
    assert_close(U.T_dependent_property(1420), 1.3758901734104049e-05)
    assert False == U.test_method_validity(1430, U.method)

    # Test lower limit for BHIRUD_NORMAL
    fake = VolumeLiquid(Tc=1000)
    assert False == fake.test_method_validity(349.5, 'BHIRUD_NORMAL')



    # Ethanol data, calculated from CoolProp
    Ts = [275, 300, 350]
    Ps = [1E5, 5E5, 1E6]
    TP_data = [[5.723868454722602e-05, 5.879532690584185e-05, 6.242643879647073e-05], [5.721587962307716e-05, 5.8767676310784456e-05, 6.238352991918547e-05], [5.718753361659462e-05, 5.8733341196316644e-05, 6.23304095312013e-05]]
    EtOH.add_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_close2d(TP_data, recalc_pts)

    EtOH.tabular_extrapolation_permitted = True
    assert_close(EtOH.TP_dependent_property(274, 9E4), 5.723868454722602e-05)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.TP_dependent_property(300, 9E4)


    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')


@pytest.mark.meta_T_dept
def test_VolumeLiquid_eval_not_duplicate_VDI_tabular():
    assert 'tabular_data' not in str(VolumeLiquid(CASRN="109-66-0"))

@pytest.mark.meta_T_dept
def test_VolumeLiquidPolynomialTmin():
    # toluene
    v = VolumeLiquid(poly_fit=(178.01, 581.75, [2.2801490297347937e-23, -6.411956871696508e-20, 7.723152902379232e-17, -5.197203733189603e-14, 2.1348482785660093e-11, -5.476649499770259e-09, 8.564670053875876e-07, -7.455178589434267e-05, 0.0028545812080104068]))
    assert v._Tmin_T_trans == v.poly_fit_Tmin
    assert_close1d(v.poly_fit_Tmin_quadratic, [2.159638355215081e-10, 0.0, 8.76710398245817e-05])

    # methanol
    v = VolumeLiquid(poly_fit=(175.7, 502.5, [3.5725079384600736e-23, -9.031033742820083e-20, 9.819637959370411e-17, -5.993173551565636e-14, 2.2442465416964825e-11, -5.27776114586072e-09, 7.610461006178106e-07, -6.148574498547711e-05, 0.00216398089328537]))
    assert v._Tmin_T_trans == v.poly_fit_Tmin
    assert_close1d(v.poly_fit_Tmin_quadratic, [9.044411973585966e-11, 0.0, 3.2636013401752355e-05])

    # furufryl alcohol
    v = VolumeLiquid(poly_fit=(189.6, 568.8000000000001, [6.647049391841594e-25, -1.8278965754470383e-21, 2.170226311490262e-18, -1.4488205117650477e-15, 5.940737543663388e-13, -1.529528764741801e-10, 2.4189938181664235e-08, -2.0920414326130236e-06, 0.00014008105452139704]))
    assert v._Tmin_T_trans == v.poly_fit_Tmin
    assert_close1d(v.poly_fit_Tmin_quadratic, [1.0450567029924047e-10, 2.3832444455827036e-08, 6.0795970965526267e-05])

    # Water - may change, second should always be zero
    v = VolumeLiquid(poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652]))
    assert_close1d(v.poly_fit_Tmin_quadratic,  [8.382589086490995e-12, 0.0, 1.739254665187681e-05])

@pytest.mark.meta_T_dept
def test_VolumeLiquidConstantExtrapolation():
    obj = VolumeLiquid(CASRN="108-38-3", MW=106.165, Tb=412.25, Tc=617.0, Pc=3541000.0,
                       Vc=0.000375, Zc=0.25884384676233363, omega=0.331, dipole=0.299792543559857,
                       Psat=None, extrapolation="constant", method="DIPPR_PERRY_8E")

    assert_close(obj.T_dependent_property(obj.DIPPR_Tmax), 0.0003785956866273838, rtol=1e-9)
    assert_close(obj.T_dependent_property(obj.DIPPR_Tmin), 0.00011563344813684695, rtol=1e-9)


    assert obj.T_dependent_property(618) == obj.T_dependent_property(obj.DIPPR_Tmax)
    assert obj.T_dependent_property(1223532.0) == obj.T_dependent_property(obj.DIPPR_Tmax)

    assert obj.T_dependent_property(1.0) == obj.T_dependent_property(obj.DIPPR_Tmin)
    assert obj.T_dependent_property(.0) == obj.T_dependent_property(obj.DIPPR_Tmin)
    assert obj.T_dependent_property(100) == obj.T_dependent_property(obj.DIPPR_Tmin)



@pytest.mark.meta_T_dept
def test_VolumeLiquidDumpEOS():
    eos = [PR(T=300, P=1E5, Tc=430.8, Pc=7884098.25, omega=0.251)]
    obj = VolumeLiquid(CASRN='7446-09-5', MW=64.0638,  Tc=430.8, Pc=7884098.25, omega=0.251, dipole=1.63, eos=eos)
    hash0 = hash(obj)
    obj2 = VolumeLiquid.from_json(json.loads(json.dumps(obj.as_json())))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

    obj = VolumeLiquid(CASRN='7446-09-5', MW=64.0638,  Tc=430.8, Pc=7884098.25, omega=0.251, dipole=1.63)
    hash0 = hash(obj)
    obj2 = VolumeLiquid.from_json(json.loads(json.dumps(obj.as_json())))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

@pytest.mark.meta_T_dept
def test_VolumeSolid():
    Vm = VolumeSolid(CASRN='10022-31-8').T_dependent_property(300)
    assert_close(Vm, 8.06592592592592e-05)

    assert None == VolumeSolid(CASRN='10022-31-8').T_dependent_property(-100)

    with pytest.raises(Exception):
        VolumeSolid(CASRN='10022-31-8').test_method_validity(200, 'BADMETHOD')

    BaN2O6 = VolumeSolid(CASRN='10022-31-8')
    BaN2O6.add_tabular_data([200, 300], [8.06e-5, 8.05e-5], 'fake')
    assert_close(8.055e-05, BaN2O6.T_dependent_property(250))
    BaN2O6.tabular_extrapolation_permitted = False
    BaN2O6.test_method_validity(150, 'fake')

@pytest.mark.meta_T_dept
def test_VolumeSolid_works_with_no_data():
    # Test pentane has no property but that it succeeds
    VolumeSolid(CASRN='109-66-0').T_dependent_property(300)

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VolumeSolid_fitting2():
    obj = VolumeSolid(CASRN='7782-44-7', load_data=False)
    Ts_gamma = [42.801, 44.0, 46.0, 48.0,
               50.0, 52.0, 54.0, 54.361]
    
    Vms_gamma = [23.05e-6, 23.06e-6, 23.18e-6, 23.30e-6,
                23.43e-6, 23.55e-6, 23.67e-6, 23.69e-6]
    obj.fit_add_model(Ts=Ts_gamma, data=Vms_gamma, model='DIPPR100', name='gamma')
    assert obj.method == 'gamma'

@pytest.mark.meta_T_dept
@pytest.mark.fitting
def test_VolumeSolid_fitting1():
    # Initial Result: went poorly + Don't have that exact model
    # 1) A special polynomial fitting may be useful
    # 2) If the model does not "converge", it may be worth trying other solvers
    ammonia_Ts = [194.150, 194.650, 195.150, 195.650, 196.150, 196.650, 197.150, 197.650, 198.150, 198.650, 199.150, 199.650, 200.150, 200.650, 201.150, 201.650, 202.150, 202.650, 203.150, 203.650, 204.150]
    ammonia_Vms = [1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3, 1/47.9730E3]
    obj = VolumeSolid(CASRN='7664-41-7', load_data=False)
    
    res, stats = VolumeSolid.fit_data_to_model(Ts=ammonia_Ts, data=ammonia_Vms, model='DIPPR100',
                          do_statistics=True, use_numba=False)
    assert stats['MAE'] < 1e-8




@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VolumeSolid_fitting0():
    Ts_alpha = [4.2, 18.5, 20, 22, 23.880]
    Vms_alpha = [20.75e-6, 20.75e-6, 20.75e-6, 20.78e-6, 20.82e-6]
    
    Ts_beta = [23.880, 24, 26, 28,
               30, 32, 34, 36,
               38, 40, 42, 43.801]
    Vms_beta = [20.95e-6, 20.95e-6, 21.02e-6, 21.08e-6,
                21.16e-6, 21.24e-6, 21.33e-6, 21.42e-6, 
                21.52e-6, 21.63e-6, 21.75e-6, 21.87e-6]
                
    Ts_gamma = [42.801, 44.0, 46.0, 48.0,
               50.0, 52.0, 54.0, 54.361]
    
    Vms_gamma = [23.05e-6, 23.06e-6, 23.18e-6, 23.30e-6,
                23.43e-6, 23.55e-6, 23.67e-6, 23.69e-6]
    
    obj = VolumeSolid(CASRN='7782-44-7', load_data=False)
    # obj.add_tabular_data(Ts=Ts_alpha, properties=Vms_alpha, name='O2_alpha')
    # obj.add_tabular_data(Ts=Ts_beta, properties=Vms_beta, name='O2_beta')
    # obj.add_tabular_data(Ts=Ts_gamma, properties=Vms_gamma, name='O2_gamma')
    
    fit_zeros_specified = obj.fit_data_to_model(Ts=Ts_gamma, data=Vms_gamma, model='DIPPR100', do_statistics=False, use_numba=False,
                     model_kwargs={'C': 0.0, 'D': 0.0, 'E': 0.0, 'F': 0.0, 'G': 0.0, 'B': 0.0})
    for l in ('B', 'C', 'D', 'E', 'F'):
        assert fit_zeros_specified[l] == 0.0
        
    fit_constant = obj.fit_data_to_model(Ts=Ts_gamma, data=Vms_gamma, model='constant', do_statistics=False, use_numba=False)
    assert_close(fit_zeros_specified['A'], fit_constant['A'], rtol=1e-13)

@pytest.mark.meta_T_dept
@pytest.mark.fitting
def test_VolumeLiquid_fitting0():

    ammonia_Ts_V_l = [195, 206.061, 217.121, 228.182, 239.242, 239.82, 250.303, 261.363, 272.424, 283.484, 294.545, 305.605, 316.666, 327.726, 338.787, 349.847, 360.908, 371.968, 383.029, 394.089, 405.15]
    ammonia_V_ls = [1/43200.2, 1/42436.8, 1/41658, 1/40862.4, 1/40048.1, 1/40005, 1/39213, 1/38354.7, 1/37470.3, 1/36556, 1/35607.3, 1/34618.5, 1/33582.2, 1/32488.1, 1/31322.5, 1/30064.9, 1/28683.4, 1/27123.2, 1/25274.3, 1/22840.2, 1/16984.9]
    ammonia_rhom_ls = [1/v for v in ammonia_V_ls]
    obj = VolumeLiquid(CASRN='7664-41-7', load_data=False)

    coeffs, stats = obj.fit_data_to_model(Ts=ammonia_Ts_V_l, data=ammonia_rhom_ls, model='DIPPR105',
                          do_statistics=True, use_numba=False,
                          guesses= {'A': 4.0518E3, 'B': 0.27129, 'C': 405.4,'D': 0.31349})
    assert stats['MAE'] < 1e-5
    
@pytest.mark.meta_T_dept
@pytest.mark.fitting
def test_VolumeLiquid_fitting1_dippr():
    fit_check_CASs = ['124-38-9', '74-98-6', '1333-74-0', '630-08-0', 
                      '100-21-0', '624-92-0', '624-72-6', '74-86-2',
                      '115-07-1', '64-18-6']
    for CAS in fit_check_CASs:
        obj = VolumeLiquid(CASRN=CAS)
        Ts = linspace(obj.DIPPR_Tmin, obj.DIPPR_Tmax, 8)
        props_calc = [1.0/obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR105',
                              do_statistics=True, use_numba=False, fit_method='lm')
        assert stats['MAE'] < 1e-5

@pytest.mark.slow
@pytest.mark.fuzz
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_VolumeLiquid_fitting2_dippr_116_ppds():
    for i, CAS in enumerate(chemicals.volume.rho_data_VDI_PPDS_2.index):
        obj = VolumeLiquid(CASRN=CAS)
        Ts = linspace(obj.T_limits[VDI_PPDS][0], obj.T_limits[VDI_PPDS][1], 8)
        props_calc = [Vm_to_rho(obj.calculate(T, VDI_PPDS), obj.VDI_PPDS_MW) for T in Ts]
        
        res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR116',
                              do_statistics=True, use_numba=False, fit_method='lm', 
                              model_kwargs={'Tc': obj.VDI_PPDS_Tc, 'A': obj.VDI_PPDS_rhoc})
        assert stats['MAE'] < 1e-7

@pytest.mark.meta_T_dept
@pytest.mark.fitting
def test_VolumeLiquid_fitting3():
    # From yaws
    Tc, rhoc, b, n, MW = 627.65, 433.128, 0.233, 0.2587, 66.0
    Ts = linspace(293.15, 298.15, 10)
    props_calc = [Rackett_fit(T, Tc, rhoc, b, n, MW) for T in Ts]
    res, stats = VolumeLiquid.fit_data_to_model(Ts=Ts, data=props_calc, model='Rackett_fit',
                          do_statistics=True, use_numba=False, model_kwargs={'MW':MW, 'Tc': Tc,},
                          fit_method='lm')
    assert stats['MAE'] < 1e-5
    
    # From yaws
    Tc, rhoc, b, n, MW = 1030.0, 1795.0521319999998, 0.96491407, 0.15872, 97.995
    Ts = linspace(315.51, 393.15, 10)
    props_calc = [Rackett_fit(T, Tc, rhoc, b, n, MW) for T in Ts]
    res, stats = VolumeLiquid.fit_data_to_model(Ts=Ts, data=props_calc, model='Rackett_fit',
                          do_statistics=True, use_numba=False, model_kwargs={'MW':MW, 'Tc': Tc,},
                          fit_method='lm')
    assert stats['MAE'] < 1e-5

    # From yaws
    Tc, rhoc, b, n, MW = 545.03, 739.99, 0.3, 0.28571, 105.921
    Ts = linspace(331.15, 332.9, 10)
    props_calc = [Rackett_fit(T, Tc, rhoc, b, n, MW) for T in Ts]
    res, stats = VolumeLiquid.fit_data_to_model(Ts=Ts, data=props_calc, model='Rackett_fit',
                          do_statistics=True, use_numba=False, model_kwargs={'MW':MW, 'Tc': Tc,},
                          fit_method='lm')
    assert stats['MAE'] < 1e-5

    # From yaws
    Tc, rhoc, b, n, MW = 1800.0, 2794.568007, 0.647077183, 0.8, 98.999
    Ts = linspace(703.0, 1620., 10)
    props_calc = [Rackett_fit(T, Tc, rhoc, b, n, MW) for T in Ts]
    res, stats = VolumeLiquid.fit_data_to_model(Ts=Ts, data=props_calc, model='Rackett_fit',
                          do_statistics=True, use_numba=False, model_kwargs={'MW':MW, 'Tc': Tc,},
                          fit_method='lm')
    assert stats['MAE'] < 1e-5


@pytest.mark.meta_T_dept
def test_VolumeLiquidMixture():
#    from thermo.mixture import Mixture
#    m = Mixture(['benzene', 'toluene'], zs=[.5, .5], T=298.15, P=101325.)
    T, P, zs = 298.15, 101325.0, [0.5, 0.5]
    MWs = [78.11184, 92.13842]
    ws = zs_to_ws(zs, MWs=MWs)

    VaporPressures = [
        VaporPressure(CASRN="71-43-2", Tb=353.23, Tc=562.05, Pc=4895000.0, omega=0.212, extrapolation="AntoineAB|DIPPR101_ABC", method=POLY_FIT, poly_fit=(278.68399999999997, 562.01, [4.547344107145341e-20, -1.3312501882259186e-16, 1.6282983902136683e-13, -1.0498233680158312e-10, 3.535838362096064e-08, -3.6181923213017173e-06, -0.001593607608896686, 0.6373679536454406, -64.4285974110459])),
        VaporPressure(CASRN="108-88-3", Tb=383.75, Tc=591.75, Pc=4108000.0, omega=0.257, extrapolation="AntoineAB|DIPPR101_ABC", method=POLY_FIT, poly_fit=(178.01, 591.74, [-8.638045111752356e-20, 2.995512203611858e-16, -4.5148088801006036e-13, 3.8761537879200513e-10, -2.0856828984716705e-07, 7.279010846673517e-05, -0.01641020023565049, 2.2758331029405516, -146.04484159879843]))
    ]

    VolumeLiquids = [
        VolumeLiquid(CASRN="71-43-2", MW=78.11184, Tb=353.23, Tc=562.05, Pc=4895000.0, Vc=0.000256, Zc=0.2681535335844513, omega=0.212, dipole=0.0, Psat=VaporPressures[0], extrapolation="constant", method=POLY_FIT, poly_fit=(278.68399999999997, 552.02, [2.5040222732960933e-22, -7.922607445206804e-19, 1.088548130214618e-15, -8.481605391952225e-13, 4.098451788397536e-10, -1.257577461969114e-07, 2.3927976459304723e-05, -0.0025810882828932375, 0.12092854717588034])),
        VolumeLiquid(CASRN="108-88-3", MW=92.13842, Tb=383.75, Tc=591.75, Pc=4108000.0, Vc=0.000316, Zc=0.2638426898300023, omega=0.257, dipole=0.33, Psat=VaporPressures[1], extrapolation="constant", method=POLY_FIT, poly_fit=(178.01, 581.75, [2.2801490297347937e-23, -6.411956871696508e-20, 7.723152902379232e-17, -5.197203733189603e-14, 2.1348482785660093e-11, -5.476649499770259e-09, 8.564670053875876e-07, -7.455178589434267e-05, 0.0028545812080104068]))
    ]

    obj = VolumeLiquidMixture(MWs=MWs, Tcs=[562.05, 591.75], Pcs=[4895000.0, 4108000.0], Vcs=[0.000256, 0.000316], Zcs=[0.2681535335844513, 0.2638426898300023], omegas=[0.212, 0.257],
                              CASs=['71-43-2', '108-88-3'], VolumeLiquids=VolumeLiquids, correct_pressure_pure=False)

    hash0 = hash(obj)
    obj2 = VolumeLiquidMixture.from_json(json.loads(json.dumps(obj.as_json())))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

    obj2 = eval(str(obj))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0


    Vml = 9.858773618507427e-05
    assert_close(obj.calculate(T, P, zs, ws, COSTALD_MIXTURE), Vml, rtol=1e-11)
    obj.method = COSTALD_MIXTURE
    assert_close(obj(T, P, zs, ws),  Vml, rtol=1e-11)

    Vml = 9.815400609778346e-05
    assert_close(obj.calculate(T, P, zs, ws, COSTALD_MIXTURE_FIT), Vml, rtol=1e-11)
    obj.method = COSTALD_MIXTURE_FIT
    assert_close(obj(T, P, zs, ws),  Vml, rtol=1e-11)

    Vml = 9.814047221052766e-05
    assert_close(obj.calculate(T, P, zs, ws, LINEAR), Vml, rtol=1e-11)
    obj.method = LINEAR
    assert_close(obj(T, P, zs, ws),  Vml, rtol=1e-11)

    Vml = 9.7377562180953e-05
    assert_close(obj.calculate(T, P, zs, ws, RACKETT), Vml, rtol=1e-11)
    obj.method = RACKETT
    assert_close(obj(T, P, zs, ws), Vml, rtol=1e-11)

    Vml = 9.810986651973312e-05
    assert_close(obj.calculate(T, P, zs, ws, RACKETT_PARAMETERS), Vml, rtol=1e-11)
    obj.method = RACKETT_PARAMETERS
    assert_close(obj(T, P, zs, ws), Vml, rtol=1e-11)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

@pytest.mark.meta_T_dept
def test_VolumeLiquidMixture_Laliberte():
    # Test Laliberte
#    m = Mixture(['water', 'sulfuric acid'], zs=[0.01, 0.99], T=298.15)
    T = 298.15
    P = 101325.0
    MWs = [18.01528, 98.07848]
    VolumeLiquids = [
        VolumeLiquid(CASRN="7732-18-5", MW=18.01528, Tb=373.124, Tc=647.14, Pc=22048320.0, Vc=5.6000000000000006e-05, Zc=0.22947273972184645, omega=0.344, dipole=1.85, extrapolation="constant", method="POLY_FIT", poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652])),
        VolumeLiquid(CASRN="7664-93-9", MW=98.07848, Tb=610.15, Tc=924.0, Pc=6400000.0, Vc=0.00017769999999999998, Zc=0.14803392201622453, omega=0.494, dipole=2.72, extrapolation="constant", method="POLY_FIT", poly_fit=(277.2, 831.6, [2.2367522080473987e-26, -8.993118194782151e-23, 1.5610918942741286e-19, -1.5237052699290813e-16, 9.134548476347517e-14, -3.438497628955312e-11, 7.949714844467134e-09, -1.0057972242365927e-06, 0.00010045015725585257]))
    ]
    zs = [0.01, 0.99]
    ws = zs_to_ws(zs, MWs=MWs)

    obj = VolumeLiquidMixture(MWs=MWs, CASs=['7732-18-5', '7664-93-9'], VolumeLiquids=VolumeLiquids)
    Vm = obj.mixture_property(T, P, zs, ws)
    assert_close(Vm, 4.824170609370422e-05, rtol=1e-12)


@pytest.mark.meta_T_dept
def test_VolumeLiquidMixture_ExcessVolume():
    # Excess volume
    # Vs -7.5E-7 in ddbst http://www.ddbst.com/en/EED/VE/VE0%20Ethanol%3BWater.php
    T = 298.15
    P = 101325.0
    MWs = [18.01528, 46.06844]
    zs = [1- 0.15600,  0.15600]
    ws = zs_to_ws(zs, MWs)

    VaporPressures = [VaporPressure(CASRN="7732-18-5", Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, extrapolation="AntoineAB|DIPPR101_ABC", method="WAGNER_MCGARRY"),
    VaporPressure(CASRN="64-17-5", Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, extrapolation="AntoineAB|DIPPR101_ABC", method="WAGNER_MCGARRY")]

    drink = VolumeLiquidMixture(MWs=[18.01528, 46.06844], Tcs=[647.14, 514.0], Pcs=[22048320.0, 6137000.0], Vcs=[5.6000000000000006e-05, 0.000168], Zcs=[0.22947273972184645, 0.24125043269792065], omegas=[0.344, 0.635], CASs=['7732-18-5', '64-17-5'], correct_pressure_pure=True, method="LALIBERTE",
        VolumeLiquids=[VolumeLiquid(CASRN="7732-18-5", MW=18.01528, Tb=373.124, Tc=647.14, Pc=22048320.0, Vc=5.6000000000000006e-05, Zc=0.22947273972184645, omega=0.344, dipole=1.85, Psat=VaporPressures[0], extrapolation="constant", method="VDI_PPDS", method_P="COSTALD_COMPRESSED"),
                       VolumeLiquid(CASRN="64-17-5", MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125043269792065, omega=0.635, dipole=1.44, Psat=VaporPressures[1], extrapolation="constant", method="DIPPR_PERRY_8E", method_P="COSTALD_COMPRESSED")]
    )
    V_Ex = drink.excess_property(T, P, zs, ws)
    assert_close(V_Ex, -7.242450496000289e-07, rtol=1e-12)


@pytest.mark.meta_T_dept
def test_VolumeGasMixture():
    zs = [.5, .5]
    T=298.15
    P=1E6
    MWs = [31.9988, 28.0134]
    ws = zs_to_ws(zs, MWs)
    # ['oxygen', 'nitrogen']
    VolumeGases = [VolumeGas(CASRN="7782-44-7", MW=31.9988, Tc=154.58, Pc=5042945.25, omega=0.021, dipole=0.0, eos=[PR(Tc=154.58, Pc=5042945.25, omega=0.021, T=298.15, P=1000000.0)], extrapolation=None, method=None, method_P="EOS"),
                   VolumeGas(CASRN="7727-37-9", MW=28.0134, Tc=126.2, Pc=3394387.5, omega=0.04, dipole=0.0, eos=[PR(Tc=126.2, Pc=3394387.5, omega=0.04, T=298.15, P=1000000.0)], extrapolation=None, method=None, method_P="EOS")]
    eos = PRMIX(T=T, P=P, zs=zs, Tcs=[154.58, 126.2], Pcs=[5042945.25, 3394387.5], omegas=[0.021, 0.04])

    obj = VolumeGasMixture(CASs=["7782-44-7" "7727-37-9"], VolumeGases=VolumeGases, eos=[eos], MWs=MWs)

    assert_close(obj.calculate(T, P, zs, ws, EOS), 0.00246280615920584, rtol=1e-7)
    assert_close(obj.calculate(T, P, zs, ws, IDEAL), 0.002478957029602388, rtol=1e-7)
    assert_close(obj.calculate(T, P, zs, ws, LINEAR), 0.002462690948679184, rtol=1e-7)

    hash0 = hash(obj)
    obj2 = VolumeGasMixture.from_json(json.loads(json.dumps(obj.as_json())))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

    obj2 = eval(str(obj))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

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
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_VolumeSupercriticalLiquidMixtureCoolProp():
    from CoolProp.CoolProp import PropsSI
    obj = VolumeSupercriticalLiquid(CASRN='7732-18-5')
    V_implemented = obj.calculate_P(T=700.0, P=1e8, method='COOLPROP')
    V_CoolProp = 1/PropsSI('DMOLAR', 'T', 700, 'P', 1e8, 'water')
    assert_close(V_implemented, V_CoolProp, rtol=1e-13)
    assert obj.test_method_validity_P(T=700.0, P=1e8, method='COOLPROP')
    assert obj.test_method_validity_P(T=700.0, P=1e4, method='COOLPROP')

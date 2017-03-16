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
import pandas as pd
from thermo.volume import *
from thermo.volume import VDI_TABULAR
from thermo.eos import *
from thermo.utils import Vm_to_rho
from thermo.identifiers import checkCAS

def test_volume_CSP():
    V1_calc = Yen_Woods_saturation(300, 647.14, 55.45E-6, 0.245)
    V1 = 1.7695330765295693e-05
    V2_calc = Yen_Woods_saturation(300, 647.14, 55.45E-6, 0.27)
    V2 = 1.8750391558570308e-05
    assert_allclose([V1_calc, V2_calc], [V1, V2])

    V1_calc = Rackett(300, 647.14, 22048320.0, 0.23)
    V1 = 1.640447373010018e-05
    V2_calc = Rackett(272.03889, 369.83, 4248000.0, 0.2763)
    V2 = 8.299222192473635e-05
    assert_allclose([V1_calc, V2_calc], [V1, V2])

    V1_calc = Yamada_Gunn(300, 647.14, 22048320.0, 0.245)
    assert_allclose(V1_calc, 2.1882836429895796e-05)

    V1_calc = Townsend_Hales(300, 647.14, 55.95E-6, 0.3449)
    assert_allclose(V1_calc, 1.8007361992619923e-05)

    V1_calc = Bhirud_normal(280.0, 469.7, 33.7E5, 0.252)
    V1 = 0.00011249654029488583
    V2_calc = Bhirud_normal(469.7*.99, 469.7, 33.7E5, 0.252)
    V2 = 0.00021992866778218994
    assert_allclose([V1_calc, V2_calc], [V1, V2])

    # Test above Tc, where interpolation table fails
    with pytest.raises(Exception):
        Bhirud_normal(500, 469.7, 33.7E5, 0.252)

    V1_calc = COSTALD(298, 647.13, 55.95E-6, 0.3449)
    V1 = 1.8133760480018036e-05
    # Propane, from API Handbook example; I have used their exact values,
    # but they rounded each step, getting 530.1
    V2_calc = COSTALD(272.03889, 369.83333, 0.20008161E-3, 0.1532)
    V2 = 8.315466172295678e-05
    assert_allclose([V1_calc, V2_calc], [V1, V2])

    rho_ex = Vm_to_rho(COSTALD(272.03889, 369.83333, 0.20008161E-3, 0.1532), 44.097)
    assert_allclose(rho_ex, 530.3009967969841)

    # Argon, with constants from [1]_ Table II and compared with listed
    # calculated result for critical volume in Table III. Tabulated s, lambda,
    # alpha, and beta are also a match.
    V1_calc = Campbell_Thodos(150.65, 87.28, 150.65, 48.02*101325, 39.948, 0.0)
    V1 = 7.538925368472471e-05

    # Water, with constants from [1]_ Table II and compared with listed
    # calculated result for critical volume in Table V. Tabulated s, lambda,
    # alpha, and beta are also a match. Deviation of 0.1% is due to author's
    # rearrangement of the formula.
    V2_calc = Campbell_Thodos(T=647.3, Tb=373.15, Tc=647.3, Pc=218.3*101325, M=18.015, dipole=1.85, hydroxyl=True)
    V2 = 5.47870007721154e-05

    # Ammonia, with constants from [1]_ Table II and compared with listed
    # calculated result for critical volume in Table IV. Tabulated s, lambda,
    # alpha, and beta are also a match. Deviation of 0.1% is due to author's
    # rearrangement of the formula.
    V3_calc = Campbell_Thodos(T=405.45, Tb=239.82, Tc=405.45, Pc=111.7*101325, M=17.03, dipole=1.47)
    V3 = 7.347363635885525e-05
    assert_allclose([V1_calc, V2_calc, V3_calc], [V1, V2, V3])

    # No examples for this model have been found, but it is simple and well
    # understood.
    V1_calc = SNM0(121, 150.8, 7.49e-05, -0.004)
    V1 = 3.4402256402733416e-05
    V2_calc = SNM0(121, 150.8, 7.49e-05, -0.004, -0.03259620)
    V2 = 3.493288100008123e-05
    assert_allclose([V1_calc, V2_calc], [V1, V2])

def test_volume_CSP_dense():
    V = COSTALD_compressed(303., 9.8E7, 85857.9, 466.7, 3640000.0, 0.281, 0.000105047)
    assert_allclose(V, 9.287482879788506e-05)


def test_CRC_inorganic():
#    # Lithium Sulfate:
    rho1_calc = CRC_inorganic(1133.15, 2003.0, 0.407, 1133.15)
    rho1 = 2003.0
    rho2_calc = CRC_inorganic(1405, 2003.0, 0.407, 1133.15)
    rho2 = 1892.35705
    assert_allclose([rho1_calc, rho2_calc], [rho1, rho2])

    # Tin tetrachloride
    rho = CRC_inorganic(300, 2370.0, 2.687, 239.08)
    assert_allclose(rho, 2206.30796)

def test_COSTALD_parameters():
    assert all([checkCAS(i) for i in COSTALD_data.index])
    assert COSTALD_data.index.is_unique
    tots_calc = [COSTALD_data[i].sum() for i in ['omega_SRK', 'Vchar', 'Z_RA']]
    tots = [72.483900000000006, 0.086051663333333334, 49.013500000000001]
    assert_allclose(tots_calc, tots)

def test_SN0_data():
    assert SNM0_data.index.is_unique
    assert all([checkCAS(i) for i in SNM0_data.index])
    tot = SNM0_data['delta_SRK'].abs().sum()
    assert_allclose(tot, 2.0715134)


def test_Perry_l_data():
    assert Perry_l_data.index.is_unique
    assert all([checkCAS(i) for i in Perry_l_data.index])

    tots_calc = [Perry_l_data[i].sum() for i in ['C1', 'C2', 'C3', 'C4', 'Tmin', 'Tmax']]
    tots = [376364.41000000003, 89.676429999999996, 189873.32999999999, 96.68741, 71151.899999999994, 189873.32999999999]
    assert_allclose(tots_calc, tots)


def test_VDI_PPDS_2_data():
    '''Plenty of interesting errors here.
    The chemicals 463-58-1, 75-44-5, 75-15-0, 7446-11-9, 2551-62-4
    do not match the tabulated data. They are all in the same section, so a
    mixup was probably made there. The errors versus the tabulated data are
    very large. 
    
    Note this table needed to have Tc and MW added to it as well, from the 
    same source.
    '''
    assert all([checkCAS(i) for i in VDI_PPDS_2.index])
    tots_calc = [VDI_PPDS_2[i].abs().sum() for i in [u'A', u'B', u'C', u'D', u'Tc', u'rhoc', u'MW']]
    tots = [208878.27130000002, 117504.59450000001, 202008.99950000001, 85280.333600000013, 150142.28, 97269, 27786.919999999998]
    assert_allclose(tots_calc, tots)
    
    assert VDI_PPDS_2.index.is_unique
    assert VDI_PPDS_2.shape == (272, 8)

def test_CRC_inorg_l_data2():
    tots_calc = [CRC_inorg_l_data[i].abs().sum() for i in ['rho', 'k', 'Tm', 'Tmax']]
    tots = [882131, 181.916, 193785.09499999997, 233338.04999999996]
    assert_allclose(tots_calc, tots)

    assert CRC_inorg_l_data.index.is_unique
    assert all([checkCAS(i) for i in CRC_inorg_l_data.index])


def test_CRC_const_inorg_l():
    assert CRC_inorg_l_const_data.index.is_unique
    assert all([checkCAS(i) for i in CRC_inorg_l_const_data.index])

    tot_calc = CRC_inorg_l_const_data['Vm'].sum()
    tot = 0.01106122489849834
    assert_allclose(tot_calc, tot)

def test_CRC_const_inorg_s():
    tot = CRC_inorg_s_const_data['Vm'].sum()
    assert_allclose(tot, 0.13528770767318143)
    assert CRC_inorg_s_const_data.index.is_unique
    assert all([checkCAS(i) for i in CRC_inorg_s_const_data.index])


def test_CRC_virial_poly():
    assert CRC_virial_data.index.is_unique
    assert all([checkCAS(i) for i in CRC_virial_data.index])
    tots_calc = [CRC_virial_data[i].abs().sum() for i in ['a1', 'a2', 'a3', 'a4', 'a5']]
    tots = [146559.69999999998, 506997.70000000001, 619708.59999999998, 120772.89999999999, 4483]
    assert_allclose(tots_calc, tots)


def test_solids_CSP():
    from thermo.volume import Goodman
    V = Goodman(281.46, 353.43, 7.6326)
    assert_allclose(V, 8.797191839062899)

    # TODO: lots of testing


@pytest.mark.meta_T_dept
def test_VolumeLiquid():
    # Ethanol, test all methods at once
    EtOH = VolumeLiquid(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, dipole=1.44, CASRN='64-17-5')
    methods = list(EtOH.all_methods)
    methods.remove(VDI_TABULAR)
    Vm_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(305.))[1] for i in methods]
    
    Vm_exp = [5.905316741206586e-05, 5.784760660832295e-05, 5.7594571728502063e-05, 5.594757794216803e-05, 5.912157674597306e-05, 5.9082910221835385e-05, 5.526836182702171e-05, 5.821947224585489e-05, 5.1921776627430897e-05, 5.9680793094807483e-05, 5.4848470492414296e-05, 5.507075716132008e-05, 5.3338182234795054e-05]
    assert_allclose(sorted(Vm_calcs), sorted(Vm_exp))
    assert_allclose(EtOH.calculate(305, VDI_TABULAR), 5.909768693432104e-05,rtol=1E-4)

    # Test that methods return None
    EtOH = VolumeLiquid(MW=46.06844, Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, dipole=1.44, CASRN='64-17-5')
    EtOH.T_dependent_property(305.) # Initialize the sorted_valid_methods
    EtOH.tabular_extrapolation_permitted = False
    Vml_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(600))[1] for i in EtOH.sorted_valid_methods]
    assert [None]*14 == Vml_calcs

    EtOH.set_user_methods('VDI_TABULAR', forced=True)
    EtOH.tabular_extrapolation_permitted = True
    assert_allclose(EtOH.T_dependent_property(700.), 0.0005648005718236466)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    SnCl4 = VolumeLiquid(CASRN='7646-78-8')
    Vm_calcs = [(SnCl4.set_user_methods(i), SnCl4.T_dependent_property(305.))[1] for i in SnCl4.all_methods]


    # Get MMC parameter
    SO2 = VolumeLiquid(MW=64.0638, Tb=263.1, Tc=430.8, Pc=7884098.25, Vc=0.000122, Zc=0.26853, omega=0.251, dipole=1.63, CASRN='7446-09-5')
    Vm_calcs = [(SO2.set_user_methods(i), SO2.T_dependent_property(200.))[1] for i in SO2.all_methods]
    Vm_exp = [3.9697664371887463e-05, 3.748481829074182e-05, 4.0256041843356724e-05, 3.982522377343308e-05, 4.062166881078707e-05, 4.0608189210203123e-05, 3.949103647364349e-05, 3.994849780626379e-05, 4.109189955368007e-05, 3.965944731935354e-05, 4.0948267317531393e-05, 4.0606869929178414e-05, 4.060446067691708e-05, 3.993451478384902e-05]
    assert_allclose(sorted(Vm_calcs), sorted(Vm_exp))

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


# More gases:
def test_ideal_gas():
    assert_allclose(ideal_gas(298.15, 101325.), 0.02446539540458919)


@pytest.mark.meta_T_dept
def test_VolumeGas():
    eos = [PR(T=300, P=1E5, Tc=430.8, Pc=7884098.25, omega=0.251)]
    SO2 = VolumeGas(CASRN='7446-09-5', MW=64.0638,  Tc=430.8, Pc=7884098.25, omega=0.251, dipole=1.63, eos=eos)
    Vm_calcs = [(SO2.set_user_methods_P(i, forced_P=True), SO2.TP_dependent_property(305, 1E5))[1] for i in SO2.all_methods_P]
    Vm_exp = [0.025024302563892417, 0.02499978619699621, 0.02499586901117375, 0.02499627309459868, 0.02499978619699621, 0.024971467450477493, 0.02535910239]
    assert_allclose(sorted(Vm_calcs), sorted(Vm_exp))

    # Test that methods return None
    assert [None]*7 == [(SO2.set_user_methods_P(i, forced_P=True), SO2.TP_dependent_property(-100, 1E5))[1] for i in SO2.all_methods_P]
    assert [None]*7 == [(SO2.set_user_methods_P(i, forced_P=True), SO2.TP_dependent_property(100, -1E5))[1] for i in SO2.all_methods_P]


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
    H2.set_user_methods_P('CRC_VIRIAL', forced_P=True)
    assert_allclose(H2.TP_dependent_property(300, 1E5), 0.024958834892394446)


def test_Amgat():
    Vl = Amgat([0.5, 0.5], [4.057e-05, 5.861e-05])
    assert_allclose(Vl, 4.9590000000000005e-05)

    with pytest.raises(Exception):
        Amgat([0.5], [4.057e-05, 5.861e-05])


def test_Rackett_mixture():
    Vl = Rackett_mixture(T=298., xs=[0.4576, 0.5424], MWs=[32.04, 18.01], Tcs=[512.58, 647.29], Pcs=[8.096E6, 2.209E7], Zrs=[0.2332, 0.2374])
    assert_allclose(Vl, 2.625288603174508e-05)

    with pytest.raises(Exception):
        Rackett_mixture(T=298., xs=[0.4576], MWs=[32.04, 18.01], Tcs=[512.58, 647.29], Pcs=[8.096E6, 2.209E7], Zrs=[0.2332, 0.2374])

def test_COSTALD_mixture():
    Vl = COSTALD_mixture([0.4576, 0.5424], 298.,  [512.58, 647.29],[0.000117, 5.6e-05], [0.559,0.344] )
    assert_allclose(Vl, 2.706588773271354e-05)

    with pytest.raises(Exception):
        COSTALD_mixture([0.4576, 0.5424], 298.,  [512.58],[0.000117, 5.6e-05], [0.559,0.344] )


def test_VolumeLiquidMixture():
    from thermo.chemical import Mixture
    from thermo.volume import LALIBERTE, COSTALD_MIXTURE_FIT, RACKETT_PARAMETERS, COSTALD_MIXTURE,  SIMPLE, RACKETT
    m = Mixture(['benzene', 'toluene'], zs=[.5, .5], T=298.15, P=101325.)
    
    VolumeLiquids = [i.VolumeLiquid for i in m.Chemicals]
    
    obj = VolumeLiquidMixture(MWs=m.MWs, Tcs=m.Tcs, Pcs=m.Pcs, Vcs=m.Vcs, Zcs=m.Zcs, omegas=m.omegas, 
                              CASs=m.CASs, VolumeLiquids=VolumeLiquids)
    
    Vm = obj.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(Vm, 9.814092676573469e-05)
    
    Vms = [obj.calculate(m.T, m.P, m.zs, m.ws, method) for method in obj.all_methods]
    Vms_expect = [9.814092676573469e-05, 9.737758899339708e-05, 9.8109833265793461e-05, 9.8154006097783393e-05, 9.858773618507426e-05]
    assert_allclose(sorted(Vms), sorted(Vms_expect))
    
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


def test_VolumeGasMixture():
    from thermo.chemical import Mixture
    from thermo.volume import VolumeGasMixture, EOS, SIMPLE, IDEAL
    m = Mixture(['oxygen', 'nitrogen'], zs=[.5, .5], T=298.15, P=1E6)
    obj = VolumeGasMixture(CASs=m.CASs, VolumeGases=m.VolumeGases, eos=m.eos_in_a_box)
    
    assert_allclose(obj.mixture_property(m.T, m.P, m.zs, m.ws), 0.0024628053244477232)
    assert_allclose(obj.calculate(m.T, m.P, m.zs, m.ws, SIMPLE), 0.002468989614515616)
    assert_allclose(obj.calculate(m.T, m.P, m.zs, m.ws, IDEAL), 0.0024789561893699998)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

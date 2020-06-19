# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import exp, log
from numpy.testing import assert_allclose
import pytest
import numpy as np
import pandas as pd
from fluids.constants import calorie, R
from thermo.flash_basic import *
from chemicals.rachford_rice import *
from fluids.numerics import assert_close, normalize
from random import uniform
import random


def test_K_value():
    K = K_value(101325, 3000.)
    assert_allclose(K, 0.029607698001480384)

    K = K_value(P=101325, Psat=3000, gamma=0.9)
    assert_allclose(K, 0.026646928201332347)

    K = K_value(P=101325, Psat=3000, gamma=0.9, Poynting=1.1)
    assert_allclose(K, 0.029311621021465586)

    K = K_value(phi_l=1.6356, phi_g=0.88427)
    assert_allclose(K, 1.8496613025433408)

    K = K_value(P=1E6, Psat=1938800, phi_l=1.4356, phi_g=0.88427, gamma=0.92)
    assert_allclose(K, 2.8958055544121137)

    K = K_value(P=1E6, Psat=1938800, phi_l=1.4356, phi_g=0.88427, gamma=0.92, Poynting=0.999)
    assert_allclose(K, 2.8929097488577016)

    with pytest.raises(Exception):
        K_value(101325)

    with pytest.raises(Exception):
        K_value(101325, gamma=0.9)

    with pytest.raises(Exception):
        K_value(P=1E6, Psat=1938800, phi_l=0.88427, gamma=0.92)

def test_Wilson_K_value():
    K = Wilson_K_value(270.0, 7600000.0, 305.4, 4880000.0, 0.098)
    assert_allclose(K, 0.2963932297479371)

def test_PR_water_K_value():
    K = PR_water_K_value(300, 1e5, 568.7, 2490000.0)
    assert_allclose(K, 76131.19143239626)




def test_identify_phase():
    # Above the melting point, higher pressure than the vapor pressure
    assert 'l' == identify_phase(T=280, P=101325, Tm=273.15, Psat=991)

    # Above the melting point, lower pressure than the vapor pressure
    assert 'g' == identify_phase(T=480, P=101325, Tm=273.15, Psat=1791175)

    # Above the melting point, above the critical pressure (no vapor pressure available)
    assert 'g' == identify_phase(T=650, P=10132500000, Tm=273.15, Psat=None, Tc=647.3)

    # No vapor pressure specified, under the melting point
    assert 's' == identify_phase(T=250, P=100, Tm=273.15)

    # No data, returns None
    assert None == identify_phase(T=500, P=101325)

    # No Tm, under Tb, at normal atmospheric pressure
    assert 'l' == identify_phase(T=200, P=101325, Tb=373.15)

    # Incorrect case by design:
    # at 371 K, Psat is 93753 Pa, meaning the actual phase is gas
    assert 'l' == identify_phase(T=371, P=91000, Tb=373.15)

    # Above Tb, while still atmospheric == gas
    assert 'g' == identify_phase(T=400, P=101325, Tb=373.15)

    # Above Tb, 1 MPa == None - don't try to guess
    assert None == identify_phase(T=400, P=1E6, Tb=373.15)

    # Another wrong point - at 1 GPa, should actually be a solid as well
    assert 'l' == identify_phase(T=371, P=1E9, Tb=373.15)

    # At the critical point, consider it a gas
    assert 'g' == identify_phase(T=647.3, P=22048320.0, Tm=273.15, Psat=22048320.0, Tc=647.3)

    # Just under the critical point
    assert 'l' == identify_phase(T=647.2, P=22048320.0, Tm=273.15, Psat=22032638.96749514, Tc=647.3)






def test_flash_wilson_no_solution_high_P():
    Tcs = [647.14, 190.56400000000002, 611.7, 755.0, 514.0]
    Pcs = [22048320.0, 4599000.0, 2110000.0, 1160000.0, 6137000.0]
    omegas = [0.344, 0.008, 0.49, 0.8486, 0.635]
    zs = [.5, .2, .1, .000001, .199999]
    with pytest.raises(ValueError):
        flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, VF=1e-9, P=1e13)


def test_flash_wilson_7_pts():
    # One point for each solver
    Tcs = [305.322, 540.13]
    Pcs = [4872200.0, 2736000.0]
    omegas = [0.099, 0.349]
    zs = [0.4, 0.6]

    pts_expect = [(0.42219453293637355, [0.020938815080034565, 0.9790611849199654], [0.9187741856225791, 0.08122581437742094], 300, 100000.0),
                  (0, [0.4000000000000001, 0.6], [0.9971719383958112, 0.002828061604188798], 300, 1760137.622367434),
    (1.0000000000000002, [0.0012588941742382993, 0.9987411058257617], [0.3999999999999999, 0.6000000000000001], 300, 13809.75314624744),
    (0.09999999999999991, [0.333751127818513, 0.666248872181487], [0.9962398496333837, 0.0037601503666162338], 300, 1469993.84209323),
    (0.9999999999999998, [0.0010577951059655955, 0.9989422048940344], [0.40000000000000013, 0.5999999999999999], 292.7502201800744, 10000.0),
    (0, [0.3999999999999997, 0.6000000000000003], [0.9999933387478749, 6.6612521250351704e-06], 161.25716180072513, 10000.0),
    (0.1, [0.33333455513993754, 0.6666654448600624], [0.9999890037407793, 1.09962592206854e-05], 163.9318379558994, 10000.0),
     (1.0, [0.5373624144993883, 0.4626375855006118], [0.39999999999999997, 0.6], 10039.470370815543, 2000000000.0), 
      (-0.0, [0.4, 0.6], [0.2861558177249859, 0.7138441822750141], 8220.290630716623, 2000000000.0)]
    
    kwargs = [dict(T=300, P=1e5), dict(T=300, VF=0), dict(T=300, VF=1), dict(T=300, VF=0.1),
              dict(P=1e4, VF=1), dict(P=1e4, VF=0), dict(P=1e4, VF=0.1),
             dict(P=2e9, VF=1), dict(P=2e9, VF=0)
             ]

    for (VF, xs, ys, T, P), kw in zip(pts_expect, kwargs):
        ans = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, **kw)
        assert_allclose([ans[0], ans[1], ans[2]], [T, P, VF], atol=1e-9)
        assert_allclose(ans[3], xs)
        assert_allclose(ans[4], ys)

    with pytest.raises(Exception):
        flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=300)
        
    for VF in (0, 1):
        with pytest.raises(Exception):
            flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, P=1e10, VF=VF)

    for VF in (0, 1):
        # not recommended but should work
        flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=1e12, VF=VF)  


def test_flash_wilson_7_pts_44_components():
    zs = [9.11975115499676e-05, 9.986813065240533e-05, 0.0010137795304828892, 0.019875879000370657, 0.013528874875432457, 0.021392773691700402, 0.00845450438914824, 0.02500218071904368, 0.016114189201071587, 0.027825798446635016, 0.05583179467176313, 0.0703116540769539, 0.07830577180555454, 0.07236459223729574, 0.0774523322851419, 0.057755091407705975, 0.04030134965162674, 0.03967043780553758, 0.03514481759005302, 0.03175471055284055, 0.025411123554079325, 0.029291866298718154, 0.012084986551713202, 0.01641114551124426, 0.01572454598093482, 0.012145363820829673, 0.01103585282423499, 0.010654818322680342, 0.008777712911254239, 0.008732073853067238, 0.007445155260036595, 0.006402875549212365, 0.0052908087849774296, 0.0048199150683177075, 0.015943943854195963, 0.004452253754752775, 0.01711981267072777, 0.0024032720444511282, 0.032178399403544646, 0.0018219517069058137, 0.003403378548794345, 0.01127516775495176, 0.015133143423489698, 0.029483213283483682]
    Tcs = [126.2, 304.2, 373.2, 190.56400000000002, 305.32, 369.83, 407.8, 425.12, 460.4, 469.7, 507.6, 540.2, 568.7, 594.6, 611.7, 639.0, 658.0, 675.0, 693.0, 708.0, 723.0, 736.0, 747.0, 755.0, 768.0, 778.0, 786.0, 790.0, 800.0, 812.0, 816.0, 826.0, 824.0, 838.0, 843.0, 562.05, 591.75, 617.15, 630.3, 649.1, 511.7, 553.8, 532.7, 572.1]
    Pcs = [3394387.5, 7376460.0, 8936865.0, 4599000.0, 4872000.0, 4248000.0, 3640000.0, 3796000.0, 3380000.0, 3370000.0, 3025000.0, 2740000.0, 2490000.0, 2290000.0, 2110000.0, 1980000.0, 1820000.0, 1680000.0, 1570000.0, 1480000.0, 1400000.0, 1340000.0, 1290000.0, 1160000.0, 1070000.0, 1030000.0, 980000.0, 920000.0, 870000.0, 950000.0, 800000.0, 883000.0, 800000.0, 826000.0, 600000.0, 4895000.0, 4108000.0, 3609000.0, 3732000.0, 3232000.0, 4510000.0, 4080000.0, 3790000.0, 3480000.0]
    omegas = [0.04, 0.2252, 0.1, 0.008, 0.098, 0.152, 0.17600000000000002, 0.193, 0.22699999999999998, 0.251, 0.2975, 0.3457, 0.39399999999999996, 0.444, 0.49, 0.535, 0.562, 0.623, 0.679, 0.6897, 0.742, 0.7564, 0.8087, 0.8486, 0.8805, 0.9049, 0.9423, 1.0247, 1.0411, 1.105, 1.117, 1.214, 1.195, 1.265, 1.26, 0.212, 0.257, 0.301, 0.3118, 0.3771, 0.1921, 0.239, 0.213, 0.2477]
    
    a = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=300, P=1e5)
    
    b = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=300, VF=0)
    assert_allclose(b[2], 0, atol=1e-5)
    c = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=300, VF=1)
    assert_allclose(c[2], 1, atol=1e-5)
    d = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=300, VF=.1)
    assert_allclose(d[2], .1, atol=1e-5)
    
    e = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, P=1e4, VF=1)
    # Tough problem - didn't converge with newton needed bisecting
    assert_allclose(e[2], 1, atol=1e-4)
    
    f = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, P=1e4, VF=0)
    assert_allclose(f[2], 0, atol=1e-5)
    g = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, P=1e4, VF=.1)
    assert_allclose(g[2], .1, atol=1e-5)
    
@pytest.mark.slow
def test_flash_wilson_PVFs():
    zs = [.5, .2, .1, .000001, .199999]
    Tcs = [647.14, 190.56400000000002, 611.7, 755.0, 514.0]
    Pcs = [22048320.0, 4599000.0, 2110000.0, 1160000.0, 6137000.0]
    omegas = [0.344, 0.008, 0.49, 0.8486, 0.635]
    
    # 8*7 = 56 cases
    VFs = [1-1e-9, .9999965432, .5, .00012344231, 0+1e-9, 1.0, 0.0]
    Ps = [1e-6, 1e-3, 1.0, 1e2, 1e4, 1e5, 1e7, 1e9]
    
    for VF in VFs:
        for P in Ps:
            T, _, _, xs, ys = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, VF=VF, P=P)
            _, _, _, xs_TP, ys_TP = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=T, P=P)
            assert_allclose(xs, xs_TP, rtol=1e-7)
            assert_allclose(ys, ys_TP, rtol=1e-7)

def test_flash_wilson_PVFs_issues():
    zs = [.5, .2, .1, .000001, .199999]
    Tcs = [647.14, 190.56400000000002, 611.7, 755.0, 514.0]
    Pcs = [22048320.0, 4599000.0, 2110000.0, 1160000.0, 6137000.0]
    omegas = [0.344, 0.008, 0.49, 0.8486, 0.635]
    # Problem point - RR had a hard time converging
    zs = [0.4050793625620341, 0.07311645032153137, 0.0739927977508874, 0.0028093939126068498, 0.44500199545294034]
    P = 100.0
    VF = 0.0
    T, _, _, xs, ys = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, VF=VF, P=P)
    _, _, VF_calc, xs_TP, ys_TP = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=T, P=P)
    assert_allclose(xs, xs_TP)
    assert_allclose(ys, ys_TP)
    assert_allclose(VF_calc, VF, atol=1e-15)
    
    # point where initial guess was lower than T_low
    VF, P, zs = (1.0, 1000000000.0, [0.019533438038283966, 0.8640664164566196, 0.05411762997156959, 0.04013739663879973, 0.022145118894727096])
    T, _, _, xs, ys = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, VF=VF, P=P)
    _, _, VF_calc, xs_TP, ys_TP = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=T, P=P)
    assert_allclose(xs, xs_TP)
    assert_allclose(ys, ys_TP)
    assert_allclose(VF_calc, VF, atol=1e-15)

        
def test_flash_wilson_singularity():
    '''This was an issue at some point. At the time, it was thought
    singularity detection and lagrange multipliers to avoid them would work. 
    However, the enhanced solver alone did the trick.
    '''
    # methane, hydrogen
    zs, Tcs, Pcs, omegas = [0.01, 0.99], [190.564, 33.2], [4599000.0, 1296960.0], [0.008, -0.22]
    T, P = 33.0, 1e6
    _, _, VF, xs, ys = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, P=P, T=T)
    assert_allclose(VF, 0.952185734369369)
    _, _, _, xs_P, ys_P = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, P=P, VF=VF)
    assert_allclose(xs_P, xs)
    assert_allclose(ys_P, ys)
    _, _, _, xs_T, ys_T = flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=T, VF=VF)
    assert_allclose(xs_T, xs)
    assert_allclose(ys_T, ys)


def test_flash_Tb_Tc_Pc():
    zs = [0.4, 0.6]
    Tcs = [305.322, 540.13]
    Pcs = [4872200.0, 2736000.0]
    Tbs = [184.55, 371.53]
    
    pts_expect = [(300, 100000.0, 0.3807040748145384, [0.031157843036568357, 0.9688421569634317], [0.9999999998827085, 1.1729141887515062e-10]),
    (300, 1283785.913825573, -0.0, [0.4, 0.6], [0.9999999999943419, 5.658116122925235e-12]),
    (300, 2.0177249381618095e-05, 0.9999999999999999, [2.5147182768635663e-12, 0.9999999999974851], [0.40000000000000013, 0.5999999999999999]),
    (300, 1069821.5948593523, 0.09999999999999983, [0.3333333333341717, 0.6666666666658283], [0.9999999999924559, 7.544154830524291e-12]),
    (454.8655563262036, 10000.0, 0.9999999999999968, [3.58554151581514e-07, 0.9999996414458484], [0.40000000000000124, 0.5999999999999986]),
    (249.4334109464263, 10000.0, 0, [0.4000000000000017, 0.5999999999999984], [0.9999999999999953, 4.827701406949022e-15]),
    (251.02234192078578, 10000.0, 0.10000000000005567, [0.33333333333329307, 0.6666666666667069], [0.9999999999999916, 8.393212330904787e-15]),
    (553.4361030097119, 10000000.0, 0.9999999999999971, [2.1518338960168313e-05, 0.9999784816610399], [0.4000000000000012, 0.5999999999999989]),
    (328.1256155656682, 10000000.0, 2.9605947334417166e-16, [.4, .6], [0.9999999998877286, 1.1227149915365143e-10])]
    
        
    kwargs = [dict(T=300, P=1e5), dict(T=300, VF=0), dict(T=300, VF=1), dict(T=300, VF=0.1),
              dict(P=1e4, VF=1), dict(P=1e4, VF=0), dict(P=1e4, VF=0.1),
             dict(P=1e7, VF=1), dict(P=1e7, VF=0)
             ]
    
    for (T, P, VF, xs, ys), kw in zip(pts_expect, kwargs):
        ans = flash_Tb_Tc_Pc(zs=zs, Tcs=Tcs, Pcs=Pcs, Tbs=Tbs, **kw)
        assert_allclose([ans[0], ans[1], ans[2]], [T, P, VF], atol=1e-9)
        assert_allclose(ans[3], xs)
        assert_allclose(ans[4], ys)

    with pytest.raises(Exception):
        flash_Tb_Tc_Pc(zs=zs, Tcs=Tcs, Pcs=Pcs, Tbs=Tbs, T=300)
    
    # Does not seem to fail
    for VF in (0, 1):
        with pytest.raises(Exception):
            # 0 fails ValueError, 1 fails VF is not converged
            _, _, VF_calc, _, _ = flash_Tb_Tc_Pc(zs=zs, Tcs=Tcs, Pcs=Pcs, Tbs=Tbs, P=1e25, VF=VF)

    for VF in (0, 1):
        # not recommended but should work
        flash_Tb_Tc_Pc(zs=zs, Tcs=Tcs, Pcs=Pcs, Tbs=Tbs, T=1e12, VF=VF)  

def test_flash_Tb_Tc_Pc_cases():

    T_calc, P_calc, VF_calc, xs, ys = flash_Tb_Tc_Pc(zs=[0.7058334393128614, 0.2941665606871387],
                                                     Tcs=[305.32, 469.7], Pcs=[4872000.0, 3370000.0],
                                                     Tbs=[184.55, 309.21], P=6.5e6, VF=0)
    assert_allclose(T_calc, 313.8105619996756)
    assert_allclose(VF_calc, 0, atol=1e-12)
    assert_allclose(xs, [0.7058334393128627, 0.29416656068713737])
    assert_allclose(ys, [0.9999999137511981, 8.624880199749853e-08])



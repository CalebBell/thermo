# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from thermo import normalize
from fluids.constants import calorie, R
from thermo.activity import *
from thermo.mixture import Mixture
from thermo.activity import Rachford_Rice_solution_numpy
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

def test_bubble_at_P_with_ideal_mixing():
    '''Check to see if the bubble pressure calculated from the temperature
    matches the temperature calculated by the test function'''

    test_mix = Mixture(['ethylene oxide',
                        'tetrahydrofuran',
                        'beta-propiolactone'],
                       ws=[6021, 111569.76, 30711.21, ],
                       T=273.15 + 80,
                       P=101325 + 1.5e5)

    bubble_temp = bubble_at_P(test_mix.Pbubble,
                              test_mix.zs,
                              test_mix.VaporPressures)

    assert_allclose(test_mix.T, bubble_temp)

def test_RR_numpy():
    Tcs = [369.83, 407.8, 425.12, 433.8, 460.4, 469.7, 507.6, 126.2, 190.56400000000002, 304.2, 305.32]
    Pcs = [4248000.0, 3640000.0, 3796000.0, 3196000.0, 3380000.0, 3370000.0, 3025000.0, 3394387.5, 4599000.0, 7376460.0, 4872000.0]
    omegas = [0.152, 0.17600000000000002, 0.193, 0.19699999999999998, 0.22699999999999998, 0.251, 0.2975, 0.04, 0.008, 0.2252, 0.098]
    zs = [1.7400001740000172e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004817800481780048, 0.9874633987463398, 0.006334800633480063, 0.0013666001366600135]
    P = 1e3
    T_dew = 98.49898995287606
    Ks = [Wilson_K_value(T_dew, P, Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i]) for i in range(len(zs))]
    
    from thermo.activity import Rachford_Rice_solution_numpy
    VF, xs, ys = Rachford_Rice_solution_numpy(zs, Ks)
    assert_allclose(VF, 1)

def test_Rachford_Rice_flash_error():
    err = Rachford_Rice_flash_error(0.5, zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    assert_allclose(err, 0.04406445591174976)

def test_Rachford_Rice_solution():
    xs_expect = [0.33940869696634357, 0.3650560590371706, 0.2955352439964858]
    ys_expect = [0.5719036543882889, 0.27087159580558057, 0.15722474980613044]
    V_over_F_expect = 0.6907302627738544
    zs = [0.5, 0.3, 0.2]
    Ks = [1.685, 0.742, 0.532]
    for args in [(False, False), (True, False), (True, True), (False, True)]:
        V_over_F, xs, ys = Rachford_Rice_solution(zs=zs, Ks=Ks, fprime=args[0], fprime2=args[1])
        assert_allclose(V_over_F, V_over_F_expect)
        assert_allclose(xs, xs_expect)
        assert_allclose(ys, ys_expect)
        
    V_over_F, xs, ys = Rachford_Rice_solution_numpy(zs=zs, Ks=Ks)
    assert_allclose(V_over_F, V_over_F_expect)
    assert_allclose(xs, xs_expect)
    assert_allclose(ys, ys_expect)
    
    # TODO support
    zs = [0.1]*10
    Ks = [8392.392499558426, 12360.984782058651, 13065.127660554343, 13336.292668013915, 14828.275288641305, 15830.9627719128, 17261.101575196506, 18943.481861916727, 21232.279762917482, 23663.61696650799]
#    flash_inner_loop(zs, Ks)



def test_flash_inner_loop():
    V_over_F, xs, ys = flash_inner_loop(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532], Method='Analytical')
    xs_expect = [0.33940869696634357, 0.3650560590371706, 0.2955352439964858]
    ys_expect = [0.5719036543882889, 0.27087159580558057, 0.15722474980613044]
    assert_allclose(V_over_F, 0.6907302627738544)
    assert_allclose(xs, xs_expect)
    assert_allclose(ys, ys_expect)

    zs = [0.1, 0.2, 0.3, 0.4]
    Ks = [4.2, 1.75, 0.74, 0.34]
    xs_expect = [0.07194096138571988, 0.18324869220986345, 0.3098180825880347, 0.4349922638163819]
    ys_expect = [0.30215203782002353, 0.320685211367261, 0.2292653811151457, 0.14789736969756986]
    V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks, Method='Analytical')
    assert_allclose(V_over_F, 0.12188396426827647)
    assert_allclose(xs, xs_expect)
    assert_allclose(ys, ys_expect)

#     Self created random case, twice, to force the recognition
    V_over_F, xs, ys = flash_inner_loop(zs=[0.6, 0.4], Ks=[1.685, 0.4], Method='Analytical')
    assert_allclose(V_over_F, 0.416058394160584)
    V_over_F, xs, ys = flash_inner_loop(zs=[0.6, 0.4], Ks=[1.685, 0.4])
    assert_allclose(V_over_F, 0.416058394160584)

    with pytest.raises(Exception):
        flash_inner_loop(zs=[0.6, 0.4], Ks=[1.685, 0.4], Method='FAIL')
    with pytest.raises(Exception):
        flash_inner_loop(zs=[0.1, 0.2, 0.3, 0.3, .01], Ks=[4.2, 1.75, 0.74, 0.34, .01], Method='Analytical')

    methods = flash_inner_loop(zs=[0.1, 0.2, 0.3, 0.4], Ks=[4.2, 1.75, 0.74, 0.34], AvailableMethods=True)
    assert methods == ['Analytical', 'Leibovici and Nichita 2', 'Rachford-Rice (Secant)',
                            'Rachford-Rice (Newton-Raphson)', 
                            'Rachford-Rice (Halley)', 'Rachford-Rice (NumPy)',
                            'Li-Johns-Ahmadi',
                             'Rachford-Rice (polynomial)']


    # Zero composition - technically this is incorrect? But quite useful in saturation calcs
    zs, Ks = [0.79, 0.0, 0.21], [51257.70115063271, 0.01720948221285233, 3939.4410123117154]
    V_over_F, xs, ys = flash_inner_loop(zs, Ks)
    assert_allclose(1.0175108346095985, V_over_F)
    assert_allclose([1.5147085263221123e-05, 0.0, 5.2389897372063305e-05], xs)
    assert_allclose([0.7764047697253411, 0.0, 0.20638691033830794], ys)



def test_flash_solution_algorithms():
    # Derive the analytical solution with:
#    from sympy import *
#    z1, z2, K1, K2, VF = symbols('z1, z2, K1, K2, VF')
#    expr = z1*(K1 - 1)/(1 + VF*(K1-1)) + z2*(K2 - 1)/(1 + VF*(K2-1))
#    solve(expr, VF)
    
#    from sympy import *
#    z1, z2, z3, K1, K2, K3, VF = symbols('z1, z2, z3, K1, K2, K3, VF')
#    expr = z1*(K1 - 1)/(1 + VF*(K1-1)) + z2*(K2 - 1)/(1 + VF*(K2-1)) + z3*(K3 - 1)/(1 + VF*(K3-1))
#    ans = solve(expr, VF)

    
    flash_inner_loop_secant = lambda zs, Ks: flash_inner_loop(zs=zs, Ks=Ks, Method='Rachford-Rice (Secant)')
    flash_inner_loop_NR = lambda zs, Ks: flash_inner_loop(zs=zs, Ks=Ks, Method='Rachford-Rice (Newton-Raphson)')
    flash_inner_loop_halley = lambda zs, Ks: flash_inner_loop(zs=zs, Ks=Ks, Method='Rachford-Rice (Halley)')
    flash_inner_loop_numpy = lambda zs, Ks: flash_inner_loop(zs=zs, Ks=Ks, Method='Rachford-Rice (NumPy)')
    flash_inner_loop_LJA = lambda zs, Ks: flash_inner_loop(zs=zs, Ks=Ks, Method='Li-Johns-Ahmadi')
    flash_inner_loop_poly = lambda zs, Ks: flash_inner_loop(zs=zs, Ks=Ks, Method='Rachford-Rice (polynomial)')
    flash_inner_loop_LN2 = lambda zs, Ks: flash_inner_loop(zs=zs, Ks=Ks, Method='Leibovici and Nichita 2')
    

    algorithms = [Rachford_Rice_solution, Li_Johns_Ahmadi_solution,
                  flash_inner_loop, flash_inner_loop_secant, 
                  flash_inner_loop_NR, flash_inner_loop_halley, 
                  flash_inner_loop_numpy, flash_inner_loop_LJA,
                  flash_inner_loop_poly, flash_inner_loop_LN2]
    for algo in algorithms:
        
        
        # dummpy 2 test
        zs, Ks = [.4, .6], [2, .5]
        V_over_F_expect = 0.2
        xs_expect = [1/3., 2/3.]
        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect)
        assert_allclose(xs, xs_expect)

        # Dummpy 3 test
        zs = [0.5, 0.3, 0.2]
        Ks = [1.685, 0.742, 0.532]
        V_over_F_expect = 0.6907302627738541
        xs_expect = [0.3394086969663436, 0.3650560590371706, 0.29553524399648573]
        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect)
        assert_allclose(xs, xs_expect)

        # Said to be in:  J.D. Seader, E.J. Henley, D.K. Roper, Separation Process Principles, third ed., John Wiley & Sons, New York, 2010.
        zs = [0.1, 0.2, 0.3, 0.4]
        Ks = [4.2, 1.75, 0.74, 0.34]
        V_over_F_expect = 0.12188885
        xs_expect = [0.07194015, 0.18324807, 0.30981849, 0.43499379]

        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-4)
        assert_allclose(xs, xs_expect, rtol=1E-4)

        # Said to be in:  B.A. Finlayson, Introduction to Chemical Engineering Computing, second ed., John Wiley & Sons, New York, 2012.
        zs = [0.1, 0.3, 0.4, 0.2]
        Ks = [6.8, 2.2, 0.8, 0.052]
        V_over_F_expect = 0.42583973
        xs_expect = [0.02881952, 0.19854300, 0.43723872, 0.33539943]

        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-5)
        assert_allclose(xs, xs_expect, rtol=1E-5)

        # Said to be in: J. Vidal, Thermodynamics: Applications in Chemical Engineering and the Petroleum Industry, Technip, Paris, 2003.
        zs = [0.2, 0.3, 0.4, 0.05, 0.05]
        Ks = [2.5250, 0.7708, 1.0660, 0.2401, 0.3140]
        V_over_F_expect = 0.52360688
        xs_expect = [0.11120375, 0.34091324, 0.38663852, 0.08304114, 0.07802677]
        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-2)
        assert_allclose(xs, xs_expect, rtol=1E-2)


        # Said to be in: R. Monroy-Loperena, F.D. Vargas-Villamil, On the determination of the polynomial defining of vapor-liquid split of multicomponent mixtures, Chem.Eng. Sci. 56 (2001) 5865–5868.
        zs = [0.05, 0.10, 0.15, 0.30, 0.30, 0.10]
        Ks = [6.0934, 2.3714, 1.3924, 1.1418, 0.6457, 0.5563]
        V_over_F_expect = 0.72073810
        xs_expect = [0.01070433, 0.05029118, 0.11693011, 0.27218275, 0.40287788, 0.14701374]

        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-6)
        assert_allclose(xs, xs_expect, rtol=1E-6)

        # Said to be in: R. Monroy-Loperena, F.D. Vargas-Villamil, On the determination of the polynomial defining of vapor-liquid split of multicomponent mixtures, Chem.Eng. Sci. 56 (2001) 5865–5868.
        zs = [0.3727, 0.0772, 0.0275, 0.0071, 0.0017, 0.0028, 0.0011, 0.0015, 0.0333, 0.0320, 0.0608, 0.0571, 0.0538, 0.0509, 0.0483, 0.0460, 0.0439, 0.0420, 0.0403]
        Ks = [7.11, 4.30, 3.96, 1.51, 1.20, 1.27, 1.16, 1.09, 0.86, 0.80, 0.73, 0.65, 0.58, 0.51, 0.45, 0.39, 0.35, 0.30, 0.26]
        V_over_F_expect = 0.84605135
        xs_expect = [0.06041132, 0.02035881, 0.00784747, 0.00495988, 0.00145397, 0.00227932, 0.00096884, 0.00139386, 0.03777425, 0.03851756, 0.07880076, 0.08112154, 0.08345504, 0.08694391, 0.09033579, 0.09505925, 0.09754111, 0.10300074, 0.10777648]

        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-6)
        assert_allclose(xs, xs_expect, rtol=1E-5)

        # Random example from MultiComponentFlash.xlsx, https://6507a56d-a-62cb3a1a-s-sites.googlegroups.com/site/simulationsmodelsandworksheets/MultiComponentFlash.xlsx?attachauth=ANoY7coiZq4OX8HjlI75HGTWiegJ9Tqz6cyqmmmH9ib-dhcNL89TIUTmQw3HxrnolKHgYuL66drYGasDgTkf4_RrWlciyRKwJCbSi5YgTG1GfZR_UhlBuaoKQvrW_L8HdboB3PYejRbzVQaCshwzYcOeGCZycdXQdF9scxoiZLpy7wbUA0xx8j9e4nW1D9PjyApC-MjsjqjqL10HFcw1KVr5sD0LZTkZCqFYA1HReqLzOGZE01_b9sfk351BB33mwSgWQlo3DLVe&attredirects=0&d=1
        Ks = [0.90000, 2.70000, 0.38000, 0.09800, 0.03800, 0.02400, 0.07500, 0.00019, 0.00070]
        zs = [0.0112, 0.8957, 0.0526, 0.0197, 0.0068, 0.0047, 0.0038, 0.0031, 0.0024]
        V_over_F_expect = 0.964872854762834
        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-7)

        # Random example from Rachford-Rice-Exercise.xls http://www.ipt.ntnu.no/~curtis/courses/PhD-PVT/PVT-HOT-Vienna-May-2016x/e-course/Day2_Part2/Exercises/Rachford-Rice-Exercise.xls
        zs = [0.001601, 0.009103, 0.364815, 0.096731, 0.069522, 0.014405, 0.039312, 0.014405, 0.014104, 0.043219, 0.111308, 0.086659, 0.065183, 0.032209, 0.037425]
        Ks = [1.081310969639700E+002, 6.600350291317650E+000, 3.946099352050670E+001, 4.469649874919970E+000, 9.321795620021620E-001, 3.213910680361160E-001, 2.189276413305250E-001, 7.932561445994600E-002, 5.868520215582420E-002, 2.182440138190620E-002, 1.769601670781200E-003, 2.855879877894100E-005, 2.718731754877420E-007, 2.154768511018220E-009, 2.907309385811110E-013]
        V_over_F, _, _ = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, 0.48908229446749, rtol=1E-5)

        # Random example from VLE_Pete310.xls http://www.pe.tamu.edu/barrufet/public_html/PETE310/goodies/VLE_Pete310.xls
        # Had to resolve because the goal which was specified by its author was far off from 0
        Ks = [3.578587993622110000, 10.348850231319200000, 2.033984472604390000, 0.225176162885930000, 0.096215673714140800, 0.070685757228660000, 0.001509595637954720, ]
        zs = [0.0387596899224806, 0.1937984496124030, 0.0775193798449612, 0.1162790697674420, 0.1085271317829460, 0.1550387596899220, 0.3100775193798450]
        V_over_F, _, _ = algo(zs=zs, Ks=Ks)

        assert_allclose(V_over_F, 0.191698639911785)
        
        # Example 2 in Gaganis, Vassilis, Dimitris Marinakis, and Nikos Varotsis. “A General Framework of Model Functions for Rapid and Robust Solution of Rachford–Rice Type of Equations.” Fluid Phase Equilibria 322–323 (May 25, 2012): 9–18. https://doi.org/10.1016/j.fluid.2012.03.001.
        # Claims 4 iterations
        # Some methods fail
        Ks = [2.9086E1, 8.7438E0, 1.9317E0, 7.9137E-1, 3.2918E-1, 1.5721E-1, 1.7684E-2, 1.4677E-5]
        zs = [2.8688E-5, 7.0701E-2, 1.3198E-1, 1.3039E-1, 2.7631E-2, 3.5986E-2, 4.5207E-1, 1.5122E-1]   
        if algo not in (Li_Johns_Ahmadi_solution, flash_inner_loop_LJA):
            # The LJA algo finds a perfect zero in its own method... but for this problem, converges differently
            V_over_F, _, _ = algo(zs=zs, Ks=Ks)
            assert_allclose(V_over_F, -1.8928931615799782e-05, rtol=5e-4)

        # Very very tough problem for all methods due to floating point issues! 
        # Came from a Wilson flash for VF = 1, T = 300K; pressure is around 0.006 Pa
        zs = [9.11975115499676e-05, 9.986813065240533e-05, 0.0010137795304828892, 0.019875879000370657, 0.013528874875432457, 0.021392773691700402, 0.00845450438914824, 0.02500218071904368, 0.016114189201071587, 0.027825798446635016, 0.05583179467176313, 0.0703116540769539, 0.07830577180555454, 0.07236459223729574, 0.0774523322851419, 0.057755091407705975, 0.04030134965162674, 0.03967043780553758, 0.03514481759005302, 0.03175471055284055, 0.025411123554079325, 0.029291866298718154, 0.012084986551713202, 0.01641114551124426, 0.01572454598093482, 0.012145363820829673, 0.01103585282423499, 0.010654818322680342, 0.008777712911254239, 0.008732073853067238, 0.007445155260036595, 0.006402875549212365, 0.0052908087849774296, 0.0048199150683177075, 0.015943943854195963, 0.004452253754752775, 0.01711981267072777, 0.0024032720444511282, 0.032178399403544646, 0.0018219517069058137, 0.003403378548794345, 0.01127516775495176, 0.015133143423489698, 0.029483213283483682]
        Ks = [11266712779.420027, 878492232.6773384, 276137963.8440058, 4326171618.091249, 573047115.7000155, 131436201.37184711, 49144960.82699592, 34263188.970916145, 13026505.192595435, 9843992.470472587, 3181564.1430952367, 1098600.248012159, 398336.89725376305, 147470.4607802813, 67566.76902751227, 23297.414225523942, 10686.776470987174, 4072.207866361747, 1521.3427782410724, 845.019208473066, 341.75877360772205, 194.91347949534864, 87.37639967685602, 43.81742706130358, 20.123099010095398, 11.2277426008874, 5.873713068861075, 2.3291630622640436, 1.3236952322694902, 0.5190977953895624, 0.33652998006213003, 0.1020194939160233, 0.11957263833876645, 0.036296673021294995, 0.022599820560813874, 2170843.2559104185, 756159.852797745, 292204.024675241, 208695.52033667514, 77287.61740255292, 6429786.948979916, 1914236.7164609663, 3164023.516416859, 1042916.2797133088]
        if algo is not flash_inner_loop_poly:
            V_over_F, _, _ = algo(zs=zs, Ks=Ks)
            assert_allclose(V_over_F, 1, atol=0.001)


@pytest.mark.slow
def test_fuzz_flash_inner_loop():
#    np.random.seed(0)
    for i in range(5000):
        n = np.random.randint(2,100)
        Ks = 2*np.random.random(n)
        zs = np.random.random(n)
        zs = zs/sum(zs)
        if any(Ks > 1) and any(Ks < 1):
            zs, Ks = list(zs), list(Ks)
            flash_inner_loop(zs=zs, Ks=Ks)


def validate_RR_convergence(ns, Ks, betas, n=1000):
    # TODO better support for n phases
    from random import uniform
    from thermo.activity import Rachford_Rice_valid_solution_naive, Rachford_Rice_solution2
    from numpy.testing import assert_allclose
    
    for _ in range(n):
        beta_guess = []
        for _ in range(len(betas)):
            beta_guess.append(uniform(0, 1.0 - sum(beta_guess)))
        is_valid = Rachford_Rice_valid_solution_naive(ns, beta_guess, Ks)
        if not is_valid:
            raise ValueError("Not valid guess")

        ans = Rachford_Rice_solution2(ns, Ks[0], Ks[1], beta_guess[0], beta_guess[1])
        # 1e-5 - not testing convergence tightness
        assert_allclose([ans[0], ans[1]], betas, rtol=1e-5)



def test_Rachford_Rice_solution2():
    
    n_composition_fuzz = 10
    from thermo.activity import Rachford_Rice_flash2_f_jac, Rachford_Rice_flashN_f_jac
    # Example 1 in Okuno 2010
    zs = [0.204322076984, 0.070970999150, 0.267194323384, 0.296291964579, 0.067046080882, 0.062489248292, 0.031685306730]
    Ks_y = [1.23466988745, 0.89727701141, 2.29525708098, 1.58954899888, 0.23349348597, 0.02038108640, 1.40715641002]
    Ks_z = [1.52713341421, 0.02456487977, 1.46348240453, 1.16090546194, 0.24166289908, 0.14815282572, 14.3128010831]
    betas = [0.01, .6]

    # Obtained with numdiffftools's Jacobian function
    fs_expect = [0.22327453005006953, -0.10530391302991113]
    jac_expect = [[-0.7622803760231517, -0.539733935411029], [-0.539733935411029, -0.86848106463838]]
    
    for func in (Rachford_Rice_flash2_f_jac, Rachford_Rice_flashN_f_jac):
        f, jac = Rachford_Rice_flash2_f_jac(betas, zs, Ks_y, Ks_z)
        assert_allclose(f, fs_expect)
        assert_allclose(jac, jac_expect)
    
    ans = Rachford_Rice_solution2(zs, Ks_y, Ks_z, beta_y=.1, beta_z=.6)
    xs_expect = [0.1712804659711611, 0.08150738616425436, 0.1393433949193188, 0.20945175387703213, 0.15668977784027893, 0.22650123851718007, 0.015225982711774586]
    ys_expect = [0.21147483364299702, 0.07313470386530294, 0.31982891387635903, 0.33293382568889657, 0.036586042443791586, 0.004616341311925655, 0.02142533917172731]
    zs_expect = [0.26156812278601893, 0.00200221914149187, 0.20392660665189805, 0.2431536850887592, 0.03786610596908295, 0.03355679851539993, 0.21792646184834918]
    assert_allclose([ans[0], ans[1]], [0.6868328915094766, 0.06019424397668606])
    assert_allclose(ans[2], xs_expect)
    assert_allclose(ans[3], ys_expect)
    assert_allclose(ans[4], zs_expect)
    # Guesses that go out of bounds:
#    x0 = [.3, .55], [.3, .8]
    validate_RR_convergence(zs, [Ks_y, Ks_z], [0.6868328915094766, 0.06019424397668606], n=n_composition_fuzz)
    
    

    # Example 2 in Okuno 2010
    zs = [0.132266176697, 0.205357472415, 0.170087543100, 0.186151796211, 0.111333894738, 0.034955417168, 0.159847699672]
    Ks_y = [26.3059904941, 1.91580344867, 1.42153325608, 3.21966622946, 0.22093634359, 0.01039336513, 19.4239894458]
    Ks_z = [66.7435876079, 1.26478653025, 0.94711004430, 3.94954222664, 0.35954341233, 0.09327536295, 12.0162990083]
    ans = Rachford_Rice_solution2(zs, Ks_y, Ks_z, beta_y=0.7)
    assert_allclose([ans[0], ans[1]], [0.46945316414811566, 0.47024451567068165])
    validate_RR_convergence(zs, [Ks_y, Ks_z], [0.46945316414811566, 0.47024451567068165], n=n_composition_fuzz)

    # Example 3 in Okuno 2010
    zs = [0.896646630194, 0.046757914522, 0.000021572890, 0.000026632729, 0.016499094171, 0.025646758089, 0.014401397406]
    Ks_y = [1.64571122126, 1.91627717926, 0.71408616431, 0.28582415424, 0.04917567928, 0.00326226927, 0.00000570946]
    Ks_z = [1.61947897153, 2.65352105653, 0.68719907526, 0.18483049029, 0.01228448216, 0.00023212526, 0.00000003964]
    ans = Rachford_Rice_solution2(zs, Ks_y, Ks_z, beta_y=0.9)
    assert_allclose([ans[0], ans[1]], [0.8701633566336909, 2.1803031624194252e-06,])
    validate_RR_convergence(zs, [Ks_y, Ks_z], [0.8701633566336909, 2.1803031624194252e-06], n=n_composition_fuzz)

    # Example 4 in Okuno 2010 (only of their examples with a test)
    # Negative flash, values of beta confirmed in Fig. 12
    zs = [0.08860, 0.81514, 0.09626]
    Ks_y = [0.112359551, 13.72549020, 3.389830508]
    Ks_z = [1.011235955, 0.980392157, 0.847457627]
    ans = Rachford_Rice_solution2(zs, Ks_y, Ks_z, 0.5, 0.3)
    assert_allclose([ans[0], ans[1]], [1.2, 14.66])
    validate_RR_convergence(zs, [Ks_y, Ks_z], [1.2, 14.66], n=n_composition_fuzz)


    # example 1 Li and Firoozabadi 2012: Initialization of phase fractions in Rachford-Rice equations for robust and efficient three-phase split calculation
    zs = [0.47, 0.126754033873246, 0.123759275876241, 0.190491864809508, 5.352678894647322e-2, 3.546803696453197e-2]
    Ks_y = [0.886975564280731, 183.729456216368, 28.8439229979536, 0.762796901964099, 6.805250689498878e-2, 0.345376016039736]
    Ks_z = [1.85133355509695, 0.567851997436811, 0.291644844783998, 0.182989507250403, 8.745408265736165e-2, 0.623957189693138]
    ans = Rachford_Rice_solution2(zs, Ks_y, Ks_z, beta_y=.7, beta_z=.2)
    assert_allclose([ans[0], ans[1]], [0.7151778078967964, 0.06609909166404299])
    validate_RR_convergence(zs, [Ks_y, Ks_z], [0.7151778078967964, 0.06609909166404299], n=n_composition_fuzz)

    # example 2 Li and Firoozabadi 2012: Initialization of phase fractions in Rachford-Rice equations for robust and efficient three-phase split calculation
    zs = [0.66731, 0.09575, 0.03540, 0.04452, 0.08589, 0.04470, 0.02643]
    Ks_y = [1.40089114681102, 2.41359153035331, 0.684675481993755, 0.192706323169157, 1.344316808771735e-2, 2.913379631601974e-4, 9.614643893437818e-8]
    Ks_z = [1.42336619958799, 1.56360101270076, 0.805778846552492, 0.437918929556065, 0.136423337258229, 2.241151325196582e-2, 3.114699395928320e-4]
    ans = Rachford_Rice_solution2(zs, Ks_y, Ks_z, beta_y=1e-3)
    assert_allclose([ans[0], ans[1]], [0.3886026201178722, 1.1532086735243752e-05])
    validate_RR_convergence(zs, [Ks_y, Ks_z], [0.3886026201178722, 1.1532086735243752e-05], n=n_composition_fuzz)

    # example 3 Li and Firoozabadi 2012: Initialization of phase fractions in Rachford-Rice equations for robust and efficient three-phase split calculation
    zs = [0.466 , 0.127710667872289 , 0.124693307875307 , 0.191929538808070 , 5.393076494606923e-2 , 3.573572096426427e-2]
    Ks_y = [0.367489928755904 , 91.9551101941298 , 17.6437660816506 , 0.523968443113866 , 5.444380423358842e-2, 0.192716832533260 ]
    Ks_z = [1.45983188593810, 0.627700554178016, 0.405472131110146, 0.291902855037650, 0.172272959622522, 0.704057279260822]
    ans = Rachford_Rice_solution2(zs, Ks_y, Ks_z, beta_y=.3, beta_z=.01)
    assert_allclose([ans[0], ans[1]], [0.3753717656603343, 0.04710389352175518])
    validate_RR_convergence(zs, [Ks_y, Ks_z], [0.3753717656603343, 0.04710389352175518], n=n_composition_fuzz)

    # example (Table 2) in Gao 2018 Hybrid Newton-Successive  Substitution Method for Multiphase Rachford-Rice Equations
    zs = [0.0315583329803, 0.4071270076623, 0.4751941671726, 0.0545811711566, 0.0115700446895, 0.0189113955704, 0.0000455484805, 0.0006404014374, 0.0003675973687, 0.0000037504895, 0.0000002428846, 0.0000001594408, 0.0000000228589, 0.0000000202543, 0.0000001375537]
    Ks_y = [1.8528741428663, 0.2314055308386, 0.5041709444335, 0.0635482083897, 0.4078908506272, 0.5066231481075, 27.1901689643580, 0.0765095693423, 0.1284992832837, 1.4795557872248, 12.7769884293417, 13.7666844672269, 52.4995561949013, 33.9539240672109, 5.1979194333821]
    Ks_z = [1.8115659762243, 0.6954909860157, 0.0001084501767, 0.0012603511720, 0.0013474694481, 0.0000038929319, 0.0035219133166, 0.0000171923836, 0.0000021965300, 0.0001633840436, 0.0016090228536, 0.0007523046170, 0.0000798682401, 0.0000023516037, 0.0000127574489]
    
    xs_expect = [0.4366767940810, 0.3003165208873, 0.2227137374279, 0.0255077448600, 0.0054220598442, 0.0088630653086, 0.0000271151486, 0.0002991176091, 0.0001717657315, 0.0000017714844, 0.0000001261729, 0.0000000835080, 0.0000000181866, 0.0000000129031, 0.0000000669487]
    ys_expect = [0.8091071405424, 0.0694949039355, 0.1122857953373, 0.0016209714859, 0.0022116086020, 0.0044902340485, 0.0007372654706, 0.0000228853595, 0.0000220717734, 0.0000026210100, 0.0000016121100, 0.0000011496277, 0.0000009547879, 0.0000004381100, 0.0000003479939]
    zs_expect = [0.7910688227638, 0.2088674332287, 0.0000241533442, 0.0000321487161, 0.0000073060600, 0.0000000345033, 0.0000000954972, 0.0000000051425, 0.0000000003773, 0.0000000002894, 0.0000000002030, 0.0000000000628, 0.0000000000015, 0.0000000000000, 0.0000000000009]
    # Two other solutions -(0.8, 0.26) or (0.2, 0.9) will converge, both have negative compositions
    
    # -0.01686263294, -1.1254155641 claimed to be ans 
    ans = Rachford_Rice_solution2(zs, Ks_y, Ks_z, beta_y=0, beta_z=-1.12)
    
    assert_allclose([ans[0], ans[1]], [-0.01686263291292747, -1.1254155641065355])
    assert_allclose(ans[2], xs_expect, atol=1e-12)
    assert_allclose(ans[3], ys_expect, atol=1e-11)
    assert_allclose(ans[4], zs_expect, atol=1e-11)
    validate_RR_convergence(zs, [Ks_y, Ks_z], [-0.01686263291292747, -1.1254155641065355], n=n_composition_fuzz)


    # Ks from tables 8 and 9 from Pan, Huanquan, Michael Connolly, and Hamdi
    # Tchelepi. “Multiphase Equilibrium Calculation Framework for Compositional 
    # Simulation of CO2 Injection in Low-Temperature Reservoirs.” Industrial &
    # Engineering Chemistry Research, January 4, 2019. https://doi.org/10.1021/acs.iecr.8b05229.
    # Compositions in supporting information
    JEMA_Ks_g = [1.546336826, 2.379576919, 0.8818093562, 0.3597420492, 0.0440420288, 0.0008924834994, 1.302131664E-06]
    JEMA_Ks_l2 = [1.544580305, 2.336017325, 0.8869147335, 0.3712005301, 0.0476811452, 0.001059266958, 1.76617359E-06]
    
    # CO2 first component in all lists
    JEMA_zs = [0.0693, 0.1742, 0.1944, 0.3138, 0.1549, 0.0742]
    JEMA_zs = [0.893] + [i*(1 - 0.893)/sum(JEMA_zs) for i in JEMA_zs]
    # CO2 frac 0.893 according to fig. 14 label
    beta_y, beta_z, xs, ys, zs = Rachford_Rice_solution2(JEMA_zs, JEMA_Ks_g, JEMA_Ks_l2)
    assert_allclose(beta_y, -0.37293697094541645)
    assert_allclose(beta_z, 1.2080206455621543)
    
    MSO_Ks_g = [1.420340741, 0.2964408254, 0.1805854981, 0.09303281846, 0.03891826286, 0.01263057652, 0.001106068886]
    MSO_Ks_l2 = [1.479781968, 0.06117039468, 0.01625155948, 0.002749741486, 0.0002664184498, 1.309992275E-05, 1.928151262E-08]
    MSO_zs =  [0.2354, 0.3295, 0.1713, 0.1099, 0.0574, 0.0965]
    
    MSO_zs = [0.9483] + [i*(1 - 0.9483)/sum(MSO_zs) for i in MSO_zs]
    
    beta_y, beta_z, xs, ys, zs = Rachford_Rice_solution2(MSO_zs, MSO_Ks_g, MSO_Ks_l2)
    assert_allclose([beta_y, beta_z], [0.0005228950085238463, 0.857047095608143])




def test_Rachford_Rice_solutionN():
    # 5 phase example!
    # Example 2 in Gao, Ran, Xiaolong Yin, and Zhiping Li. "Hybrid Newton-Successive 
    # Substitution Method for Multiphase Rachford-Rice Equations." Entropy 20,
    #  no. 6 (June 2018): 452. https://doi.org/10.3390/e20060452.
    zs = [0.3817399509140, 0.0764336433731, 0.1391487737570, 0.0643992218952, 0.1486026004951, 0.0417212486653, 0.1227693500767, 0.0213087870239, 0.0016270350309, 0.0021307432306, 0.0000917810305, 0.0000229831930, 0.0000034782551, 0.0000001126367, 0.0000002344634, 0.0000000038064, 0.0000000173126, 0.0000000281366, 0.0000000042589, 0.0000000024453]
    Ks0 = [2.3788914318714, 0.8354537404402, 0.1155938461254, 0.0062262830625, 0.0022156584248, 0.0115951444765, 0.0064167472255, 0.0038946321018, 0.0134366496720, 0.0008734024997, 0.0108844870333, 0.0305288385881, 0.0184206758492, 1.9556944123756, 0.2874467036782, 1.5356775373006, 0.7574272230786, 0.0074377713757, 0.0004574024029, 0.0847561330613]
    Ks1 = [ 0.1826346252218, 2.0684286685920, 2.8473183476162, 2.1383860381928, 0.7946416111326, 2.1603434367941, 0.1593792034596, 0.0335917624138, 0.7223258415919, 2.6132706480239, 24.4065005309508, 25.8494898790919, 10.4748859551860, 57.6425128090423, 1.0419187660436, 53.5513911183565, 7.6910401287961, 6.7681727478028, 28.1394115659509, 1.6486494033625]
    Ks2 = [2.1341148433378, 1.9043018392943, 0.0144945209799, 0.0442168936781, 0.0787337170042, 0.0560494950996, 0.0770042412753, 0.0025050231128, 0.1031743167040, 0.0022130957042, 0.1928690729187, 0.0588393075672, 0.3556852336181, 1.7486777932718, 1.8885719459373, 97.7361071036055, 6.0072238022229, 4.0574761982724, 35.1553173521778, 31.9676317062480]
    Ks3 = [0.7101073236142, 6.0440859895389, 0.4369041160293, 0.9918488866995, 0.7768884555186, 0.2134611795537, 0.0239948965688, 0.0218059421417, 0.1708086119388, 0.0932727495955, 1.0014414881636, 4.0996590670858, 0.1045382819199, 29.0578470200348, 13.7002699311125, 6.6483533942909, 18.7742085574180, 5.2779281096742, 9.0540032759730, 2.5158440811075]
    betas = [.2, .2, .2, .2]
    
    comps_expect = [[0.3851281921976, 0.0786796419224, 0.1384439969943, 0.0640229919586, 0.1468925975170, 0.0413515667922, 0.1207019216039, 0.0209321985655, 0.0016041800354, 0.0021136824783, 0.0000995608488, 0.0000254194834, 0.0000035608768, 0.0000001699095, 0.0000002477114, 0.0000000128093, 0.0000000197267, 0.0000000295911, 0.0000000060755, 0.0000000029023],
                    [0.9161781565910, 0.0657332011406, 0.0160032740856, 0.0003986252704, 0.0003254638212, 0.0004794773913, 0.0007745137206, 0.0000815232125, 0.0000215548051, 0.0000018460956, 0.0000010836688, 0.0000007760273, 0.0000000655938, 0.0000003322911, 0.0000000712038, 0.0000000196710, 0.0000000149415, 0.0000000002201, 0.0000000000028, 0.0000000002460],
                    [0.0703377430444, 0.1627432269869, 0.3941941327593, 0.1369058721276, 0.1167269703543, 0.0893335859206, 0.0192373761213, 0.0007031494410, 0.0011587406941, 0.0055236243798, 0.0024299319085, 0.0006570806789, 0.0000372997780, 0.0000097940117, 0.0000002580951, 0.0000006859566, 0.0000001517188, 0.0000002002775, 0.0000001709606, 0.0000000047849],
                    [0.8219077915568, 0.1498297868279, 0.0020066794190, 0.0028308978284, 0.0115654002029, 0.0023177344403, 0.0092945598936, 0.0000524356412, 0.0001655101790, 0.0000046777816, 0.0000192022086, 0.0000014956648, 0.0000012665513, 0.0000002971170, 0.0000004678207, 0.0000012519326, 0.0000001185027, 0.0000001200651, 0.0000002135856, 0.0000000927811],
                    [0.2734823498098, 0.4755465214053, 0.0604867521264, 0.0635011332973, 0.1141191632121, 0.0088269542238, 0.0028962301245, 0.0004564463108, 0.0002740077651, 0.0001971489765, 0.0000997043646, 0.0001042112156, 0.0000003722479, 0.0000049372049, 0.0000033937126, 0.0000000851609, 0.0000003703530, 0.0000001561795, 0.0000000550075, 0.0000000073018]]
    
    beta_solution = [-0.00538660799, -0.00373696250, -0.00496311432, -0.00415370309]
    Ks = [Ks0, Ks1, Ks2, Ks3]
    
    betas, comps = Rachford_Rice_solutionN(zs, Ks, betas)
    for beta_i, beta_known in zip(betas, beta_solution):
        assert_allclose(beta_i, beta_known, atol=1e-8)
    
    for comp_calc, comp_expect in zip(comps, comps_expect):
        assert_allclose(comp_calc, comp_expect, atol=1e-9)


    
    
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


def test_NRTL():
    # P05.01b VLE Behavior of Ethanol - Water Using NRTL
    gammas = NRTL_gammas([0.252, 0.748], [[0, -0.178], [1.963, 0]], [[0, 0.2974],[.2974, 0]])
    assert_allclose(gammas, [1.9363183763514304, 1.1537609663170014])

    # Test the general form against the simpler binary form
    def NRTL2(xs, taus, alpha):
        x1, x2 = xs
        tau12, tau21 = taus
        G12 = exp(-alpha*tau12)
        G21 = exp(-alpha*tau21)
        gamma1 = exp(x2**2*(tau21*(G21/(x1+x2*G21))**2 + G12*tau12/(x2+x1*G12)**2))
        gamma2 = exp(x1**2*(tau12*(G12/(x2+x1*G12))**2 + G21*tau21/(x1+x2*G21)**2))
        return gamma1, gamma2

    gammas = NRTL2(xs=[0.252, 0.748], taus=[-0.178, 1.963], alpha=0.2974)
    assert_allclose(gammas, [1.9363183763514304, 1.1537609663170014])


    # Example by
    # https://github.com/iurisegtovich/PyTherm-applied-thermodynamics/blob/master/contents/main-lectures/GE1-NRTL-graphically.ipynb
    tau = [[0.0, 2.291653777670652, 0.5166949715946564], [4.308652420938829, 0.0, 1.6753963198550983], [0.5527434579849811, 0.15106032392136134, 0.0]]
    alpha = [[0.0, 0.4, 0.3], [0.4, 0.0, 0.3], [0.3, 0.3, 0.0]]
    xs = [.1, .3, .6]
    gammas = NRTL_gammas(xs, tau, alpha)
    assert_allclose(gammas, [2.7175098659360413, 2.1373006474468697, 1.085133765593844])

    # Test the values which give activity coefficients of 1:
    gammas = NRTL_gammas([0.252, 0.748], [[0, 0], [0, 0]], [[0, 0.5],[.9, 0]])
    assert_allclose(gammas, [1, 1])
    # alpha does not matter
    
    a = b = np.zeros((6, 6)).tolist()
    gammas = NRTL_gammas([0., 1, 0, 0, 0, 0], a, b)
    assert_allclose(gammas, [1,1,1,1,1,1]) 

    # Test vs chemsep parameters, same water ethanol T and P
    T = 343.15
    b12 = -57.9601*calorie
    b21 = 1241.7396*calorie
    tau12 = b12/(R*T)
    tau21 = b21/(R*T)

    gammas = NRTL_gammas(xs=[0.252, 0.748], taus=[[0, tau12], [tau21, 0]],
    alphas=[[0, 0.2937],[.2937, 0]])
    assert_allclose(gammas, [1.9853834856640085, 1.146380779201308])
    
    
    
    # Random bad example
    alphas = [[0.0, 0.35, 0.35, 0.35, 0.35],
             [0.35, 0.0, 0.35, 0.35, 0.35],
             [0.35, 0.35, 0.0, 0.35, 0.35],
             [0.35, 0.35, 0.35, 0.0, 0.35],
             [0.35, 0.35, 0.35, 0.35, 0.0]]
    taus = [[0.0, 0.651, 2.965, 1.738, 0.761], [1.832, 0.0, 2.783, 1.35, 0.629],
            [0.528, 1.288, 0.0, 0.419, 2.395], [1.115, 1.838, 2.16, 0.0, 0.692], 
            [1.821, 2.466, 1.587, 1.101, 0.0]]
    
    xs = [0.18736982702111407, 0.2154173017033719, 0.2717319464745698, 0.11018333572613222, 0.215297589074812]
    gammas = NRTL_gammas(xs, taus, alphas)
    gammas_expect = [2.503204848288857, 2.910723989902569, 2.2547951278295497, 2.9933258413917154, 2.694165187439594]
    assert_allclose(gammas, gammas_expect)
    
    # ten component
#    m = Mixture(['water', 'ethanol', 'methanol', '1-pentanol', '2-pentanol', '3-pentanol',
#             '1-decanol', '2-decanol', '3-decanol', '4-decanol'], 
#             P=1e5, zs=[.1]*10, T=273.15+70)

    xs = [.1]*10
    T = 343.15
    alphas = [[0.0, 0.2937, 0.2999, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
     [0.2937, 0.0, 0.3009, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
     [0.2999, 0.3009, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
     [0.3, 0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
     [0.3, 0.3, 0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3],
     [0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 0.3],
     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 0.3, 0.3, 0.3],
     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 0.3, 0.3],
     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 0.3],
     [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0]]
    
    taus = [[0.0, 1.8209751485908323, 1.162621164496251, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.08499680747061582, 0.0, -0.10339969905688821, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.2772318019157448, 0.0986791288154995, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    
    gammas = NRTL_gammas(xs=xs, taus=taus, alphas=alphas)
    gammas_expect = [1.1600804309840225, 1.0892286716705042, 1.0384940848807305, 0.9836770920034531, 0.9836770920034531, 0.9836770920034531, 0.9836770920034531, 0.9836770920034531, 0.9836770920034531, 0.9836770920034531]
    assert_allclose(gammas, gammas_expect)
    
    N = 44
    taus = [[random.random() for i in range(N)] for j in range(N)]
    alphas = [[random.random() for i in range(N)] for j in range(N)]
    xs = normalize([random.random() for i in range(N)])
    gammas = NRTL_gammas(xs=xs, taus=taus, alphas=alphas)
    
    
    # Takes 40 ms - not a great idea
    N = 200
    taus = [[random.random() for i in range(N)] for j in range(N)]
    alphas = [[random.random() for i in range(N)] for j in range(N)]
    xs = normalize([random.random() for i in range(N)])
    gammas = NRTL_gammas(xs=xs, taus=taus, alphas=alphas)


def test_Wilson():
    # P05.01a VLE Behavior of Ethanol - Water Using Wilson
    # http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/P05.01a%20VLE%20Behavior%20of%20Ethanol%20-%20Water%20Using%20Wilson.xps
    gammas = Wilson_gammas([0.252, 0.748], [[1, 0.154], [0.888, 1]])
    assert_allclose(gammas, [1.8814926087178843, 1.1655774931125487])

    # Test the general form against the simpler binary form
    def Wilson2(molefracs, lambdas):
        x1 = molefracs[0]
        x2 = molefracs[1]
        l12 = lambdas[0]
        l21 = lambdas[1]
        gamma1 = exp(-log(x1+x2*l12) + x2*(l12/(x1+x2*l12) - l21/(x2+x1*l21)))
        gamma2 = exp(-log(x2+x1*l21) - x1*(l12/(x1+x2*l12) - l21/(x2+x1*l21)))
        return [gamma1, gamma2]
    gammas = Wilson2([0.252, 0.748], [0.154, 0.888])

    assert_allclose(gammas, [1.8814926087178843, 1.1655774931125487])

    # Test 3 parameter version:
    # 05.09 Compare Experimental VLE to Wilson Equation Results
    # http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/05.09%20Compare%20Experimental%20VLE%20to%20Wilson%20Equation%20Results.xps
    # Extra decimals obtained via the actual MathCad worksheet
    xs = [0.229, 0.175, 0.596]
    params = [[1, 1.1229699812593, 0.73911816162836],
              [3.26947621620298, 1, 1.16749678447695],
              [0.37280197780932, 0.01917909648619, 1]]
    gammas = Wilson_gammas(xs, params)
    assert_allclose(gammas, [1.22339343348885, 1.10094590247015, 1.2052899281172])

    # Test the values which produce gamma = 1
    gammas = Wilson_gammas([0.252, 0.748], [[1, 1], [1, 1]])
    assert_allclose(gammas, [1, 1])


def test_UNIQUAC():
    # P05.01c VLE Behavior of Ethanol - Water Using UNIQUAC
    # http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/P05.01c%20VLE%20Behavior%20of%20Ethanol%20-%20Water%20Using%20UNIQUAC.xps

    gammas = UNIQUAC(xs=[0.252, 0.748], rs=[2.1055, 0.9200], qs=[1.972, 1.400], taus=[[1.0, 1.0919744384510301], [0.37452902779205477, 1.0]])
    assert_allclose(gammas, [2.35875137797083, 1.2442093415968987])

    # Example 8.3  in [2]_ for solubility of benzene (2) in ethanol (1) at 260 K.
    # Worked great here
    gammas = UNIQUAC(xs=[.7566, .2434], rs=[2.1055, 3.1878], qs=[1.972, 2.4], taus=[[1.0, 1.17984681869376], [0.22826016391070073, 1.0]])
    assert_allclose(gammas, [1.0826343452263132, 3.0176007269546083])

    # Example 7.3 in [2], for electrolytes
    gammas = UNIQUAC(xs=[0.05, 0.025, 0.925], rs=[1., 1., 0.92], qs=[1., 1., 1.4], taus=[[1.0, 0.4052558731309731, 2.7333668483468143], [21.816716876191823, 1.0, 0.06871094878791346], [0.4790878929721784, 3.3901086879605944, 1.0]])
    assert_allclose(gammas, [0.3838177662072466, 0.49469915162858774, 1.0204435746722416])


    def UNIQUAC_original_form(xs, rs, qs, taus):
        # This works too - just slower.
        cmps = range(len(xs))

        rsxs = sum([rs[i]*xs[i] for i in cmps])
        qsxs = sum([qs[i]*xs[i] for i in cmps])

        Phis = [rs[i]*xs[i]/rsxs for i in cmps]
        thetas = [qs[i]*xs[i]/qsxs for i in cmps]

        ls = [5*(ri - qi) - (ri - 1.) for ri, qi in zip(rs, qs)]

        gammas = []
        for i in cmps:
            lngamma = (log(Phis[i]/xs[i]) + 5*qs[i]*log(thetas[i]/Phis[i]) + ls[i]
            - Phis[i]/xs[i]*sum([xs[j]*ls[j] for j in cmps])
            - qs[i]*log(sum([thetas[j]*taus[j][i] for j in cmps]))
            + qs[i]
            - qs[i]*sum([thetas[j]*taus[i][j]/sum([thetas[k]*taus[k][j] for k in cmps]) for j in cmps]))
            gammas.append(exp(lngamma))
        return gammas

    gammas = UNIQUAC_original_form(xs=[.7566, .2434], rs=[2.1055, 3.1878], qs=[1.972, 2.4], taus=[[1.0, 1.17984681869376], [0.22826016391070073, 1.0]])
    assert_allclose(gammas, [1.0826343452263132, 3.0176007269546083])

    gammas = UNIQUAC_original_form(xs=[0.252, 0.748], rs=[2.1055, 0.9200], qs=[1.972, 1.400], taus=[[1.0, 1.0919744384510301], [0.37452902779205477, 1.0]])
    assert_allclose(gammas, [2.35875137797083, 1.2442093415968987])

    gammas = UNIQUAC_original_form(xs=[0.05, 0.025, 0.925], rs=[1., 1., 0.92], qs=[1., 1., 1.4], taus=[[1.0, 0.4052558731309731, 2.7333668483468143], [21.816716876191823, 1.0, 0.06871094878791346], [0.4790878929721784, 3.3901086879605944, 1.0]])
    assert_allclose(gammas, [0.3838177662072466, 0.49469915162858774, 1.0204435746722416])



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
    

@pytest.mark.xfail
def test_flash_wilson_failure_singularity():
    '''TODO: Singularity detection and lagrange multipliers to avoid them
    '''
    m = Mixture(['methane', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'], zs=[.1]*10, T=300, P=1E6)
    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs, 
                     Tcs=m.Tcs, Pcs=m.Pcs, omegas=m.omegas, kijs=None, eos_kwargs=None)
    
    m = Mixture(['methane', 'hydrogen'], zs=[.01, .99], T=300, P=1E6)
    _, _, VF, _, _ = flash_wilson(zs=m.zs, Tcs=m.Tcs, Pcs=m.Pcs, omegas=m.omegas, P=1e6, VF=.95)
    assert_allclose(VF, 0.95)
    # .9521857343693697 is where the point below lands
    flash_wilson(zs=m.zs, Tcs=m.Tcs, Pcs=m.Pcs, omegas=m.omegas, P=1e6, T=33)
    
    
    


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


def test_Rachford_Rice_polynomial():
    zs, Ks = [.4, .6], [2, .5]
    poly = Rachford_Rice_polynomial(zs, Ks)
    coeffs_2 = [1.0, -0.20000000000000007]
    assert_allclose(coeffs_2, poly)

    zs = [0.5, 0.3, 0.2]
    Ks = [1.685, 0.742, 0.532]
    coeffs_3 = [1, -3.692652996676083, 2.073518878815094]
    poly = Rachford_Rice_polynomial(zs, Ks)
    assert_allclose(coeffs_3, poly)
    
    zs = [0.2, 0.3, 0.4, 0.1]
    Ks = [2.5250, 0.7708, 1.0660, 0.2401]
    coeffs_4 =  [1, 5.377031669207758, -24.416684496523914, 10.647389883139642]
    poly = Rachford_Rice_polynomial(zs, Ks)
    assert_allclose(coeffs_4, poly)
    
    zs = [0.2, 0.3, 0.4, 0.05, 0.05]
    Ks = [2.5250, 0.7708, 1.0660, 0.2401, 0.3140]
    poly = Rachford_Rice_polynomial(zs, Ks)
    coeffs_5 = [1.0, 3.926393887728915, -32.1738043292604, 45.82179827480925, -15.828236126660224]
    assert_allclose(coeffs_5, poly)
    
    zs = [0.05, 0.10, 0.15, 0.30, 0.30, 0.10]
    Ks = [6.0934, 2.3714, 1.3924, 1.1418, 0.6457, 0.5563]
    coeffs_6 = [1.0, 3.9413425113979077, -9.44556472337601, -18.952349132451488, 9.04210538319183, 5.606427780744831]
    poly = Rachford_Rice_polynomial(zs, Ks)
    assert_allclose(coeffs_6, poly)
    
    Ks = [0.9, 2.7, 0.38, 0.098, 0.038, 0.024, 0.075]
    zs = [0.0112, 0.8957, 0.0526, 0.0197, 0.0068, 0.0047, 0.0093]
    poly = Rachford_Rice_polynomial(zs, Ks)
    coeffs_7 = [1.0, -15.564752719919635, 68.96609128282495, -141.05508474225547, 150.04980583027202, -80.97492465198536, 17.57885132690501]
    assert_allclose(coeffs_7, poly)
    
    Ks = [0.90000, 2.70000, 0.38000, 0.09800, 0.03800, 0.02400, 0.07500, 0.00019]
    zs = [0.0112, 0.8957, 0.0526, 0.0197, 0.0068, 0.0047, 0.0038, 0.0055]
    poly = Rachford_Rice_polynomial(zs, Ks)
    coeffs_8 = [1.0, -16.565387656773854, 84.54011830455603, -210.05547256828095, 291.1575729888513, -231.05951648043205, 98.55989361947283, -17.577207793453983]
    assert_allclose(coeffs_8, poly)
    
    # 19 takes ~1 sec
    zs = [0.3727, 0.0772, 0.0275, 0.0071, 0.0017, 0.0028, 0.0011, 0.0015, 0.0333, 0.0320, 0.0608, 0.0571, 0.0538, 0.0509, 0.0483, 0.0460, 0.0439, 0.0420, 0.0403]
    Ks = [7.11, 4.30, 3.96, 1.51, 1.20, 1.27, 1.16, 1.09, 0.86, 0.80, 0.73, 0.65, 0.58, 0.51, 0.45, 0.39, 0.35, 0.30, 0.26]
    coeffs_19 = [1.0, -0.8578819552817947, -157.7870481947649, 547.7859890170784, 6926.565858999385, 
                 -39052.793041087636, -71123.61208697906, 890809.1105085013, -1246174.7361619857, 
                 -5633651.629883111, 21025868.75287835, -15469951.107862322, -41001954.18122998,
                 97340936.26910116, -72754773.28565726, 4301672.656674517, 17784298.9111024,
                 -3479139.4994188584, -1635369.1552006816]
    
    poly = Rachford_Rice_polynomial(zs, Ks)
    assert_allclose(coeffs_19, poly)
    
    # doubling 19 runs out of ram. 
    
    
def test_Rachford_Rice_polynomial_solution_VFs():
    zs = [0.2, 0.3, 0.4, 0.05, 0.05]
    Ks = [2.5250, 0.7708, 1.0660, 0.2401, 0.3140]
    VF =  Rachford_Rice_solution_polynomial(zs, Ks)[0]
    assert_allclose(VF, 0.5247206476383832)

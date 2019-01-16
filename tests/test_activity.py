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
from fluids.constants import calorie, R
from thermo.activity import *
from thermo.mixture import Mixture
from thermo.activity import Rachford_Rice_solution_numpy

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
#    zs, Ks =([0.10000000000000002, 0.10000000000000002, 0.10000000000000002, 0.10000000000000002, 0.10000000000000002, 0.10000000000000002, 0.10000000000000002, 0.10000000000000002, 0.10000000000000002, 0.10000000000000002], [8392.392499558426, 12360.984782058651, 13065.127660554343, 13336.292668013915, 14828.275288641305, 15830.9627719128, 17261.101575196506, 18943.481861916727, 21232.279762917482, 23663.61696650799])
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
    assert methods == ['Analytical', 'Rachford-Rice (Secant)',
                            'Rachford-Rice (Newton-Raphson)', 
                            'Rachford-Rice (Halley)', 'Rachford-Rice (NumPy)',
                            'Li-Johns-Ahmadi',
                             'Rachford-Rice (polynomial)']


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

    algorithms = [Rachford_Rice_solution, Li_Johns_Ahmadi_solution,
                  flash_inner_loop, flash_inner_loop_secant, 
                  flash_inner_loop_NR, flash_inner_loop_halley, 
                  flash_inner_loop_numpy, flash_inner_loop_LJA,
                  flash_inner_loop_poly]
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


@pytest.mark.slow
def test_fuzz():
#    np.random.seed(0)
    for i in range(5000):
        n = np.random.randint(2,100)
        Ks = 2*np.random.random(n)
        zs = np.random.random(n)
        zs = zs/sum(zs)
        if any(Ks > 1) and any(Ks < 1):
            zs, Ks = list(zs), list(Ks)
            flash_inner_loop(zs=zs, Ks=Ks)

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
    gammas = NRTL([0.252, 0.748], [[0, -0.178], [1.963, 0]], [[0, 0.2974],[.2974, 0]])
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
    gammas = NRTL(xs, tau, alpha)
    assert_allclose(gammas, [2.7175098659360413, 2.1373006474468697, 1.085133765593844])

    # Test the values which give activity coefficients of 1:
    gammas = NRTL([0.252, 0.748], [[0, 0], [0, 0]], [[0, 0.5],[.9, 0]])
    assert_allclose(gammas, [1, 1])
    # alpha does not matter
    
    a = b = np.zeros((6, 6)).tolist()
    gammas = NRTL([0., 1, 0, 0, 0, 0], a, b)
    assert_allclose(gammas, [1,1,1,1,1,1]) 

    # Test vs chemsep parameters, same water ethanol T and P
    T = 343.15
    b12 = -57.9601*calorie
    b21 = 1241.7396*calorie
    tau12 = b12/(R*T)
    tau21 = b21/(R*T)

    gammas = NRTL(xs=[0.252, 0.748], taus=[[0, tau12], [tau21, 0]],
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
    gammas = NRTL(xs, taus, alphas)
    gammas_expect = [2.503204848288857, 2.910723989902569, 2.2547951278295497, 2.9933258413917154, 2.694165187439594]
    assert_allclose(gammas, gammas_expect)
    


def test_Wilson():
    # P05.01a VLE Behavior of Ethanol - Water Using Wilson
    # http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/P05.01a%20VLE%20Behavior%20of%20Ethanol%20-%20Water%20Using%20Wilson.xps
    gammas = Wilson([0.252, 0.748], [[1, 0.154], [0.888, 1]])
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
    gammas = Wilson(xs, params)
    assert_allclose(gammas, [1.22339343348885, 1.10094590247015, 1.2052899281172])

    # Test the values which produce gamma = 1
    gammas = Wilson([0.252, 0.748], [[1, 1], [1, 1]])
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
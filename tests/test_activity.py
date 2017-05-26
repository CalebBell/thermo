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
from math import exp, log
from thermo.activity import *
from thermo.chemical import Mixture

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


def test_Rachford_Rice_flash_error():
    err = Rachford_Rice_flash_error(0.5, zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    assert_allclose(err, 0.04406445591174976)

def test_Rachford_Rice_solution():
    V_over_F, xs, ys = Rachford_Rice_solution(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    xs_expect = [0.33940869696634357, 0.3650560590371706, 0.2955352439964858]
    ys_expect = [0.5719036543882889, 0.27087159580558057, 0.15722474980613044]
    assert_allclose(V_over_F, 0.6907302627738544)
    assert_allclose(xs, xs_expect)
    assert_allclose(ys, ys_expect)


def test_flash_inner_loop():
    V_over_F, xs, ys = flash_inner_loop(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532], Method='Analytical')
    xs_expect = [0.33940869696634357, 0.3650560590371706, 0.2955352439964858]
    ys_expect = [0.5719036543882889, 0.27087159580558057, 0.15722474980613044]
    assert_allclose(V_over_F, 0.6907302627738544)
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
        flash_inner_loop(zs=[0.1, 0.2, 0.3, 0.4], Ks=[4.2, 1.75, 0.74, 0.34], Method='Analytical')

    methods = flash_inner_loop(zs=[0.1, 0.2, 0.3, 0.4], Ks=[4.2, 1.75, 0.74, 0.34], AvailableMethods=True)
    assert methods == ['Rachford-Rice', 'Li-Johns-Ahmadi']


def test_flash_solution_algorithms():
    flash_inner_loop_RR = lambda zs, Ks: flash_inner_loop(zs=zs, Ks=Ks, Method='Rachford-Rice')
    flash_inner_loop_LJA = lambda zs, Ks: flash_inner_loop(zs=zs, Ks=Ks, Method='Li-Johns-Ahmadi')

    algorithms = [Rachford_Rice_solution, Li_Johns_Ahmadi_solution,
                  flash_inner_loop, flash_inner_loop_RR, flash_inner_loop_LJA]
    for algo in algorithms:
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
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-11)

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
    gammas = (NRTL(xs, tau, alpha))
    assert_allclose(gammas, [2.7175098659360413, 2.1373006474468697, 1.085133765593844])


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

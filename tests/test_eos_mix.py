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
from thermo.utils import normalize
from thermo.eos import *
from thermo.eos_mix import *
from scipy.misc import derivative
from scipy.optimize import minimize, newton
from math import log, exp, sqrt
from thermo import Mixture
from thermo.property_package import eos_Z_test_phase_stability, eos_Z_trial_phase_stability


def test_PRMIX_quick():
    # Two-phase nitrogen-methane
    eos = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(3.625735065042031e-05-1.8973538018496328e-19j), (0.00019383466511209503+1.6263032587282567e-19j), (0.0007006656856469095-4.743384504624082e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [0.2187647518144111, -0.0006346633654774628, 3.6800240532105057e-06]
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [3.625735065042031e-05, 0.0007006656856469095]:
        eos = PRMIX(V=V, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        assert_allclose(eos.T, 115)
        T_slow = eos.solve_T(P=1E6, V=V, quick=False)
        assert_allclose(T_slow, 115)


    # Fugacities
    eos = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_l, [1.587721676422927, 0.14693710450607692])
    assert_allclose(eos.phis_g, [0.8730618494018239, 0.7162292765506479])
    assert_allclose(eos.fugacities_l, [793860.8382114634, 73468.55225303846])
    assert_allclose(eos.fugacities_g, [436530.9247009119, 358114.63827532396])
    
    # Numerically test fugacities at one point
    def numerical_fugacity_coefficient(n1, n2=0.5, switch=False, l=True):
        if switch:
            n1, n2 = n2, n1
        tot = n1+n2
        zs = [i/tot for i in [n1,n2]]
        a = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=zs, kijs=[[0,0],[0,0]])
        phi = a.phi_l if l else a.phi_g
        return tot*log(phi)

    phis = [[derivative(numerical_fugacity_coefficient, 0.5, dx=1E-6, order=25, args=(0.5, i, j)) for i in [False, True]] for j in [False, True]]
    assert_allclose(phis, [eos.lnphis_g, eos.lnphis_l])

    # Gas phase only test point
    a = PRMIX(T=300, P=1E7, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(a.phis_g, [0.9855336740251448, 0.8338953860988254]) # Both models.
    assert_allclose(a.phis_g, [0.9855, 0.8339], rtol=1E-4) # Calculated with thermosolver V1
    
    # Test against PrFug.xlsx
    # chethermo (Elliott, Richard and Lira, Carl T. - 2012 - Introductory Chemical Engineering Thermodynamics)
    kijs = [[0, 0.00076, 0.00171], [0.00076, 0, 0.00061], [0.00171, 0.00061, 0]]
    e = PRMIX(Tcs=[469.7, 507.4, 540.3], zs=[0.8168, 0.1501, 0.0331], 
              omegas=[0.249, 0.305, 0.349], Pcs=[3.369E6, 3.012E6, 2.736E6],
              T=322.29, P=101325, kijs=kijs)
    assert_allclose(e.V_g, 0.0254513065119208)
    assert_allclose(e.V_l, 0.00012128148428794801)
    
    assert_allclose(e.fugacity_g, 97639.120236046)
    assert_allclose(e.fugacity_l, 117178.31044886599, rtol=5E-5)
    
    assert_allclose(e.fugacities_g, [79987.657739064, 14498.518199677, 3155.0680076450003])
    assert_allclose(e.fugacities_l, [120163.95699262699, 7637.916974562, 620.954835936], rtol=5E-5)
    
    assert_allclose(e.phis_g, [0.966475030274237, 0.953292801077091, 0.940728104174207])
    assert_allclose(e.phis_l, [1.45191729893103, 0.502201064053298, 0.185146457753801], rtol=1E-4)
    
    
    # CH4-H2S mixture - no gas kij
    # checked values - accurate to with a gas constant for a standard PR EOS
    # These are very very good values confirming fugacity and fugacity coefficients are correct!
    ks = [[0,.0],[0.0,0]]
    eos = PRMIX(T=190.0, P=40.53e5, Tcs=[190.63, 373.55], Pcs=[46.17E5, 90.07E5], omegas=[0.01, 0.1], zs=[0.5, 0.5], kijs=ks)
    assert_allclose(eos.phis_l, [1.227364, 0.0114921], rtol=4e-4)
    assert_allclose(eos.fugacities_l, [2487250, 23288.73], rtol=3e-4)

    # CH4-H2S mixture - with kij - two phase, vapor frac 0.44424170
    # TODO use this as a test case
    # checked values - accurate to with a gas constant for a standard PR EOS
    ks = [[0,.083],[0.083,0]]
    xs = [0.1164203, 0.8835797]
    ys = [0.9798684, 0.0201315]

    eos = PRMIX(T=190.0, P=40.53e5, Tcs=[190.63, 373.55], Pcs=[46.17E5, 90.07E5], omegas=[0.01, 0.1], zs=xs, kijs=ks)
    assert_allclose([5.767042, 0.00774973], eos.phis_l, rtol=4e-4)
    assert_allclose([2721190, 27752.94], eos.fugacities_l, rtol=4e-4)
    eos = PRMIX(T=190.0, P=40.53e5, Tcs=[190.63, 373.55], Pcs=[46.17E5, 90.07E5], omegas=[0.01, 0.1], zs=ys, kijs=ks)
    assert_allclose([0.685195, 0.3401376], eos.phis_g, rtol=4e-4)
    assert_allclose([2721190, 27752.94], eos.fugacities_g, rtol=4e-4)
    
    # Check the kij can get copied
    kijs = [[0,.083],[0.083,0]]
    eos = PRMIX(T=190.0, P=40.53e5, Tcs=[190.6, 373.2], Pcs=[46e5, 89.4e5], omegas=[0.011, .097], zs=[.5, .5], kijs=kijs)
    eos2 = eos.to_TP_zs(T=200, P=5e6, zs=eos.zs)
    assert_allclose(eos2.kijs, kijs)
    assert_allclose(eos.T, 190)
    assert_allclose(eos.P, 40.53e5)
    assert_allclose(eos2.T, 200)
    assert_allclose(eos2.P, 5e6)
    assert eos.V_l != eos2.V_l
    
    
    # Test high temperature fugacities
    # Phase Identification Parameter would make both these roots the same phase
    eos = PRMIX(T=700, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    
    fugacities_l_expect = [55126630.003539115, 27887160.323921766]
    assert_allclose(eos.fugacities_l, fugacities_l_expect)
    
    fugacities_g_expect = [501802.41653963586, 500896.73250179]
    assert_allclose(eos.fugacities_g, fugacities_g_expect)


def test_derivatives_density():
    # Check some extra derivatives
    T = 420.0
    zs = [.5, .5]
    P = 2.7e6
    Tcs = [305.32, 540.2]
    Pcs = [4872000.0, 2740000.0]
    omegas = [0.098, 0.3457]
    kijs=[[0,0.0067],[0.0067,0]]
    
    eos = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    assert_allclose(eos.dP_drho_g, 77.71287762789959)
    assert_allclose(eos.dP_drho_l, 1712.067636831951)


def test_density_extrapolation():
    # Check some extra derivatives
    T = 420.0
    zs = [.5, .5]
    P = 2.7e6
    Tcs = [305.32, 540.2]
    Pcs = [4872000.0, 2740000.0]
    omegas = [0.098, 0.3457]
    kijs=[[0,0.0067],[0.0067,0]]
    eos = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    # Check the discriminant is zero
    P_transition = newton(eos.discriminant_at_T_zs, 2.7E6, tol=1e-12)
    assert_allclose(P_transition, 2703430.005691234)
    
    P = P_transition + .01
    eos = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    assert eos.raw_volumes[0].imag == 0
    # Check there is a small imaginary component in the others
    assert all(abs(eos.raw_volumes[i].imag) > 1e-9 for i in (1, 2))
    
    P = P_transition - .01
    eos = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    # Check the imaginary components are infinitesimally all small (real roots)
    assert all(abs(eos.raw_volumes[i].imag) < 1e-15 for i in (0, 1, 2))
    
    # See the second pressure transition place
    P_transition = newton(eos.discriminant_at_T_zs, 1e10, tol=1e-12)
    assert_allclose(P_transition, 110574232.59024328)
    
    # Below the transition point = one real root in this case.    
    P = P_transition - .01
    eos = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    # Third root is the important one in this case
    assert abs(eos.raw_volumes[2].imag) < 1e-15
    # Check there is a small but larger imaginary component in the others
    assert all(abs(eos.raw_volumes[i].imag) > 1e-9 for i in (0, 1))
    
    
    P = P_transition + 1
    eos = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    # Third root is the important one in this case
    assert abs(eos.raw_volumes[2].imag) < 1e-15
    # Check there is a very tiny imaginary component in the others
    assert all(abs(eos.raw_volumes[i].imag) < 1e-15 for i in (0, 1))
    
    
    eos = PRMIX(T=T, P=2.8E6, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    assert_allclose(eos.V_g_extrapolated(), 0.0005133247390466003)



def test_mechanical_critical_point():
    '''Test from:
    Watson, Harry A. J., and Paul I. Barton. "Reliable Flash Calculations: 
    Part 3. A Nonsmooth Approach to Density Extrapolation and Pseudoproperty
    Evaluation." Industrial & Engineering Chemistry Research, November 11, 2017.
    '''
    m = Mixture(['ethane', 'heptane'], zs=[.5, .5], T=300., P=1e5)
    eos = PRMIX(T=m.T, P=m.P, Tcs=m.Tcs, Pcs=m.Pcs, omegas=m.omegas, zs=m.zs, kijs=[[0,0.0067],[0.0067,0]])
    eos =  eos.to_mechanical_critical_point()
    assert_allclose(eos.T, 439.18795430359467, rtol=1e-5)
    assert_allclose(eos.P, 3380687.3020791663, rtol=1e-5)
    assert_allclose(1/eos.V_g, 3010, rtol=1e-3) # A more correct answer
    
    # exact answer believed to be:
    3011.7228497511787 # mol/m^3
    # 439.18798489 with Tc = 439.18798489 or so.
    
    
    
    
def test_TPD_stuff():
    # Two-phase nitrogen-methane
    eos = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    # Get a negative TPD proving there is a phase split
    TPD = eos.TPD(eos.Z_g, eos.Z_l, eos.zs, eos.zs)
    assert_allclose(TPD, -471.36283584737305)
    
    



all_zs_SRKMIX_CH4_H2S = [[0.9885, 0.0115], [0.9813, 0.0187], [0.93, 0.07], 
      [.5, .5], [0.112, 0.888], [.11, .89]]
all_expected_SRKMIX_CH4_H2S = [[.9885],
            [0.9813, 0.10653187, 0.52105697, 0.92314194],
           [.93, 0.11616261, 0.48903294, 0.98217018],
            [0.5, 0.11339494, 0.92685263, 0.98162794],
            [0.112, 0.50689794, 0.9243395, 0.98114659],
            [.11, 0.51373521, 0.92278719, 0.9809485]
           ]

def test_Stateva_Tsvetkov_TPDF_SRKMIX_CH4_H2S():
    '''Data and examples from 
    Ivanov, Boyan B., Anatolii A. Galushko, and Roumiana P. Stateva. "Phase 
    Stability Analysis with Equations of State-A Fresh Look from a Different 
    Perspective." Industrial & Engineering Chemistry Research 52, no. 32 
    (August 14, 2013): 11208-23. https://doi.org/10.1021/ie401072x.
    
    Some of the points are a little off - explained by differences in the
    a, b values of the SRK c1, and c2 values, as well as the gas constant; this
    is a very sensitive calculation. However, all the trivial points match
    exactly, and no *extra* roots could be found at all.
    
    This is all believe to be correct.
    Note: future scipy.minimize behavior might make some guesses converge elsewhere.
    
    This example is the closest - other examples do not match so well, though
    there is no reason for that! Perhaps this is the "easiest" case.
    
    '''
    all_guesses = [[[0.98]], # No other answers close to zero found
           [[0.98], [.11], [.5, .6], [0.9, 0.91]],
           [[.92], [.12], [0.5], [0.98]],
           [[.47, .49, .499], [.11], [.92], [0.98]],
           [[0.11], [.505, .52], [0.9], [0.98]],
           [[.12], [0.5], [0.9], [0.98]]
          ]

    for i in range(len(all_zs_SRKMIX_CH4_H2S)):
        zs = all_zs_SRKMIX_CH4_H2S[i]
        kijs = [[0,.08],[0.08,0]]
        eos = SRKMIX(T=190.0, P=40.53e5, Tcs=[190.6, 373.2], Pcs=[46e5, 89.4e5], omegas=[0.008, .1], zs=zs, kijs=kijs)
        Z_eos, prefer, alt = eos_Z_test_phase_stability(eos)
        
        def func(z1):
            zs_trial = [z1, 1-z1]
            eos2 = eos.to_TP_zs(T=eos.T, P=eos.P, zs=zs_trial)
            Z_trial = eos_Z_trial_phase_stability(eos2, prefer, alt)    
            TPD = eos.Stateva_Tsvetkov_TPDF(Z_eos, Z_trial, eos.zs, zs_trial)
            return TPD
        guesses = all_guesses[i]
        expected = all_expected_SRKMIX_CH4_H2S[i]
        for j in range(len(expected)):
            for k in range(len(guesses[j])):
                ans = minimize(func, guesses[j][k], bounds=[(1e-9, 1-1e-6)])
                assert_allclose(float(ans['x']), expected[j], rtol=1e-6)        

def test_d_TPD_Michelson_modified_SRKMIX_CH4_H2S():
    all_guesses = [[[0.98]],
               [[0.98], [[6530, 18900]], [[59000, 53600]], [0.91]],
               [[.92], [[9, 18]], [[20, 4]], [0.98]],
               [[.499], [[6., 18.]], [.92], [0.98]],
               [[0.11], [[142, 140]], [[6, 19]], [0.98]],
               [[.12], [[141, 141]], [[39, 9]], [0.98]]
              ]
    for i in range(len(all_zs_SRKMIX_CH4_H2S)):
        zs = all_zs_SRKMIX_CH4_H2S[i]
        kijs = [[0,.08],[0.08,0]]
        eos = SRKMIX(T=190.0, P=40.53e5, Tcs=[190.6, 373.2], Pcs=[46e5, 89.4e5], omegas=[0.008, .1], zs=zs, kijs=kijs)
        Z_eos, prefer, alt = eos_Z_test_phase_stability(eos)
    
        def func(alphas):
            Ys = [(alph/2.)**2 for alph in alphas]
            ys = normalize(Ys)
            eos2 = eos.to_TP_zs(T=eos.T, P=eos.P, zs=ys)
            Z_trial = eos_Z_trial_phase_stability(eos2, prefer, alt)    
            TPD = eos.d_TPD_Michelson_modified(Z_eos, Z_trial, eos.zs, alphas)
            return TPD
    
        guesses = all_guesses[i]
        expected = all_expected_SRKMIX_CH4_H2S[i]
        for j in range(len(expected)):
            for k in range(len(guesses[j])):
                if type(guesses[j][k])== list:
                    guess = guesses[j][k]
                else:
                    x0 = guesses[j][k]
                    x1 = 1 - x0
                    guess = [xi**0.5*2 for xi in [x0, x1]] # convert to appropriate basis
                
                # Initial guesses were obtained by trying repetitively and are specific to NM
                ans = minimize(func, guess, tol=1e-12, method='Nelder-Mead') 
                Ys = [(alph/2.)**2 for alph in ans['x']]
                ys = normalize(Ys)
                assert ans['fun'] < 1e-12
                assert_allclose(ys[0], expected[j], rtol=1e-7)        


def test_Stateva_Tsvetkov_TPDF_PRMIX_Nitrogen_Methane_Ethane():
    '''Data and examples from 
    Ivanov, Boyan B., Anatolii A. Galushko, and Roumiana P. Stateva. "Phase 
    Stability Analysis with Equations of State-A Fresh Look from a Different 
    Perspective." Industrial & Engineering Chemistry Research 52, no. 32 
    (August 14, 2013): 11208-23. https://doi.org/10.1021/ie401072x.
    
    Some of the points are a little off - explained by differences in the
    a, b values of the SRK c1, and c2 values, as well as the gas constant; this
    is a very sensitive calculation. However, all the trivial points match
    exactly. One extra root was found for the third case.
    
    This is all believe to be correct.
    Note: future scipy.minimize behavior might make some guesses converge elsewhere.
    '''
    # Problem 5: Nitrogen + Methane + Ethane at T = 270 K and P = 76 bar.
    # PR
    # Sources 8,9,11,14,17,20,43,45,46
    T = 270.0
    P = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    all_zs = [[0.3, 0.1, 0.6],
            [.15, .3, .55],
            [.08, .38, .54],
            [.05, .05, .9]]
    
    all_expected = [[[0.35201577,  0.10728462], [0.3, 0.1], [0.08199763,  0.04915481]],
                    [ [0.06451677,  0.19372499], [0.17128871, 0.31864581], [0.15, 0.3] ],
                    [[0.08,  0.38], [0.09340211,  0.40823117]],
                    [[0.05, 0.05]]
                   ]
    
    all_guesses = [[[[.35, 0.107]], [[.29,  0.107]], [[.08,  0.05]]],
              [[[.08,  0.05]], [ [.2,  0.3]], [[.155,  0.31]]],
               [[[.09,  0.39]], [[.2,  0.4]]],
               [[[.01,  0.03]]]
              ]
    
    for i in range(len(all_zs)):
        zs = all_zs[i]
        eos = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, kijs=kijs, zs=zs)
        Z_eos, prefer, alt = eos_Z_test_phase_stability(eos)
    
        def func(zs):
            zs_trial = [float(zs[0]), float(zs[1]), float(1 - sum(zs))]
            eos2 = eos.to_TP_zs(T=eos.T, P=eos.P, zs=zs_trial)
            Z_trial = eos_Z_trial_phase_stability(eos2, prefer, alt)
            TPD = eos.Stateva_Tsvetkov_TPDF(Z_eos, Z_trial, eos.zs, zs_trial)
            return TPD
    
        guesses = all_guesses[i]
        expected = all_expected[i]
        for j in range(len(expected)):
            for k in range(len(guesses[j])):
                ans = minimize(func, guesses[j][k], bounds=[(1e-9, .5-1e-6), (1e-9, .5-1e-6)], tol=1e-11)
                assert_allclose(ans['x'], expected[j], rtol=5e-6)        

    
def test_PRMIX_VS_PR():
    # Test solution for molar volumes
    eos = PRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.00013022208100139953-0j), (0.001123630932618011+0.0012926962852843173j), (0.001123630932618011-0.0012926962852843173j)]
    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.801259426590328, -0.006647926028616357, 1.6930127618563258e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives(299, quick=True)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)
    
    # PR back calculation for T
    eos = PRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00013022208100139953, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00013022208100139953, quick=False)
    assert_allclose(T_slow, 299)
    
    
    diffs_1 = [582232.4757941157, -3665180614672.2373, 1.588550570914177e-07, -2.7283785033590384e-13, 6295046.681608136, 1.717527004374129e-06]
    diffs_2 = [-506.2012523140166, 4.482165856521269e+17, 1.1688513685432287e-09, 9.103361314057314e-21, -291578941282.6521, 2.564684443970742e-15]
    diffs_mixed = [-3.772507759880179e-15, -20523303691.115646, 0.06994170496262654]
    departures = [-31134.740290463407, -72.47559475426019, 25.165377505266793]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
        
    # Integration tests
    eos = PRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00013, T=299)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]


def test_PR78MIX():
    # Copied and pasted example from PR78.
    eos = PR78MIX(Tcs=[632], Pcs=[5350000], omegas=[0.734], zs=[1], T=299., P=1E6)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [8.351960066075052e-05, -63764.64948050847, -130.737108912626]
    assert_allclose(three_props, expect_props)

    # Fugacities
    eos = PR78MIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.6, 0.7], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    # Numerically test fugacities at one point, with artificially high omegas
    def numerical_fugacity_coefficient(n1, n2=0.5, switch=False, l=True):
        if switch:
            n1, n2 = n2, n1
        tot = n1+n2
        zs = [i/tot for i in [n1,n2]]
        a = PR78MIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.6, 0.7], zs=zs, kijs=[[0,0],[0,0]])
        phi = a.phi_l if l else a.phi_g
        return tot*log(phi)

    phis = [[derivative(numerical_fugacity_coefficient, 0.5, dx=1E-6, order=25, args=(0.5, i, j)) for i in [False, True]] for j in [False, True]]
    assert_allclose(phis, [eos.lnphis_g, eos.lnphis_l])


def test_SRKMIX_quick():
    # Two-phase nitrogen-methane
    eos = SRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(4.104755570185178e-05-1.6263032587282567e-19j), (0.0002040997573162298+1.8973538018496328e-19j), (0.0007110155639819184-4.743384504624082e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [0.21053493863769052, -0.0007568158918021953, 4.650777611039976e-06]
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [4.104755570185178e-05, 0.0007110155639819184]:
        eos = SRKMIX(V=V, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        assert_allclose(eos.T, 115)
        T_slow = eos.solve_T(P=1E6, V=V, quick=False)
        assert_allclose(T_slow, 115)


    # Fugacities
    eos = SRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_l, [1.6356832861093693, 0.14476563850405255])
    assert_allclose(eos.phis_g, [0.8842742560249208, 0.7236415842381881])
    
    # Numerically test fugacities at one point
    def numerical_fugacity_coefficient(n1, n2=0.5, switch=False, l=True):
        if switch:
            n1, n2 = n2, n1
        tot = n1+n2
        zs = [i/tot for i in [n1,n2]]
        a = SRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=zs, kijs=[[0,0],[0,0]])
        phi = a.phi_l if l else a.phi_g
        return tot*log(phi)

    phis = [[derivative(numerical_fugacity_coefficient, 0.5, dx=1E-6, order=25, args=(0.5, i, j)) for i in [False, True]] for j in [False, True]]
    assert_allclose(phis, [eos.lnphis_g, eos.lnphis_l])

    # Gas phase only test point
    a = SRKMIX(T=300, P=1E7, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(a.phis_g, [1.0200087028463556, 0.8717783536379076]) 


def test_SRKMIX_vs_SRK():
    # Copy and paste from SRK, changed to list inputs only
    # Test solution for molar volumes
    eos = SRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.00014682102759032+0j), (0.00116960122630484+0.0013040890734249049j), (0.00116960122630484-0.0013040890734249049j)]
    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.7271789178606376, -0.007332989159328508, 1.947612023379061e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # PR back calculation for T
    eos = SRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00014682102759032, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00014682102759032, quick=False)
    assert_allclose(T_slow, 299)
    
    # Derivatives
    diffs_1 = [507071.37815795804, -2693849768980.0884, 1.8823298314439377e-07, -3.7121594957338967e-13, 5312565.222604471, 1.9721089437796854e-06]
    diffs_2 = [-495.525429968177, 2.685153659083702e+17, 1.3462639881888625e-09, 1.3735644012106488e-20, -201856646370.53476, 3.800656805086382e-15]
    diffs_mixed = [-4.991347301209541e-15, -14322106590.423191, 0.06594013142212454]
    departures = [-31754.65309653571, -74.3732468359525, 28.936520816725874]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
    
        
    # Integration tests
    eos = SRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]



def test_VDWMIX_vs_VDW():
    eos = VDWMIX(Tcs=[507.6], Pcs=[3025000], zs=[1], T=299., P=1E6)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.00022332978038490077, -13385.722837649315, -32.65922018109096]
    assert_allclose(three_props, expect_props)
    
    # Test of a_alphas
    a_alphas = [2.4841036545673676, 0, 0]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # Back calculation for P
    eos = VDWMIX(Tcs=[507.6], Pcs=[3025000], zs=[1], T=299, V=0.00022332978038490077)
    assert_allclose(eos.P, 1E6)
    
    # Back calculation for T
    eos = VDWMIX(Tcs=[507.6], Pcs=[3025000], zs=[1], P=1E6, V=0.00022332978038490077)
    assert_allclose(eos.T, 299)


def test_VDWIX_quick():
    # Two-phase nitrogen-methane
    eos = VDWMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(5.8813678514166464e-05-2.439454888092385e-19j), (0.0001610823684175308+2.439454888092385e-19j), (0.0007770869741895237-6.776263578034403e-21j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [0.18035220037679928, 0.0, 0.0]
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [5.8813678514166464e-05, 0.0007770869741895237]:
        eos = VDWMIX(V=V, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        assert_allclose(eos.T, 115)
        T_slow = eos.solve_T(P=1E6, V=V, quick=False)
        assert_allclose(T_slow, 115)


    # Fugacities
    eos = VDWMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_l, [1.7090665338410205, 0.414253699455241])
    assert_allclose(eos.phis_g, [0.896941472676147, 0.7956530879998579])
    
    # Numerically test fugacities at one point
    def numerical_fugacity_coefficient(n1, n2=0.5, switch=False, l=True):
        if switch:
            n1, n2 = n2, n1
        tot = n1+n2
        zs = [i/tot for i in [n1,n2]]
        a = VDWMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], zs=zs, kijs=[[0,0],[0,0]])
        phi = a.phi_l if l else a.phi_g
        return tot*log(phi)

    phis = [[derivative(numerical_fugacity_coefficient, 0.5, dx=1E-6, order=25, args=(0.5, i, j)) for i in [False, True]] for j in [False, True]]
    assert_allclose(phis, [eos.lnphis_g, eos.lnphis_l])

    # Gas phase only test point
    a = VDWMIX(T=300, P=1E7, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(a.phis_g, [0.9564004482475513, 0.8290411371501448]) 


def test_PRSVMIX_vs_PRSV():
    eos = PRSVMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6, kappa1s=[0.05104])
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.0001301268694484059, -31698.916002476708, -74.1674902435042]
    assert_allclose(three_props, expect_props)
    
    # Test of a_alphas
    a_alphas = [3.8129831135199463, -0.006976898745266429, 2.0026547235203598e-05]
    
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    
    a_alphas_fast = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # PR back calculation for T
    eos = PRSVMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.0001301268694484059, P=1E6, kappa1s=[0.05104])
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001301268694484059, quick=False)
    assert_allclose(T_slow, 299)
    
    
    # Test the bool to control its behavior
    eos = PRSVMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=406.08, P=1E6, kappa1s=[0.05104])
    assert_allclose(eos.kappas, 0.7977689278061457)
    eos.kappa1_Tr_limit = True
    eos.__init__(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=406.08, P=1E6, kappa1s=[0.05104])
    assert_allclose(eos.kappas, 0.8074380841890093)
    
    # Test the limit is not enforced while under Tr =0.7
    eos = PRSVMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=304.56, P=1E6, kappa1s=[0.05104])
    assert_allclose(eos.kappas, 0.8164956255888178)
    eos.kappa1_Tr_limit = True
    eos.__init__(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=304.56, P=1E6, kappa1s=[0.05104])
    assert_allclose(eos.kappas, 0.8164956255888178)


def test_PRSVMIX_quick():
    # Two-phase nitrogen-methane
    eos = PRSVMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(3.623552388375633e-05-1.3552527156068805e-19j), (0.0001942800283219127+2.168404344971009e-19j), (0.0007002421492037557-4.743384504624082e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [0.21897578471489654, -0.0006396067113212318, 3.7150128655294028e-06]
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [3.623552388375633e-05, 0.0007002421492037557]:
        eos = PRSVMIX(V=V, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        assert_allclose(eos.T, 115)
        T_slow = eos.solve_T(P=1E6, V=V, quick=False)
        assert_allclose(T_slow, 115)


    # Fugacities
    eos = PRSVMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_l, [1.5881151663681092, 0.14570244654356812])
    assert_allclose(eos.phis_g, [0.8731073123670093, 0.7157562213377993])
    
    # Numerically test fugacities at one point
    def numerical_fugacity_coefficient(n1, n2=0.5, switch=False, l=True):
        if switch:
            n1, n2 = n2, n1
        tot = n1+n2
        zs = [i/tot for i in [n1,n2]]
        a = PRSVMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=zs, kijs=[[0,0],[0,0]])
        phi = a.phi_l if l else a.phi_g
        return tot*log(phi)

    phis = [[derivative(numerical_fugacity_coefficient, 0.5, dx=1E-6, order=25, args=(0.5, i, j)) for i in [False, True]] for j in [False, True]]
    assert_allclose(phis, [eos.lnphis_g, eos.lnphis_l])

    # Gas phase only test point
    eos = PRSVMIX(T=300, P=1E7, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_g, [0.9859363207073754, 0.8344831291870667]) 


def test_PRSV2MIX_vs_PRSV():
    eos = PRSV2MIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6, kappa1s=[0.05104], kappa2s=[0.8634], kappa3s=[0.460])
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.00013018821346475254, -31496.173493225753, -73.6152580115141]
    assert_allclose(three_props, expect_props)
    
    # Test of PRSV2 a_alphas
    a_alphas = [3.8054176315098256, -0.00687315871653124, 2.3078008060652167e-05]
    
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_fast = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # PSRV2 back calculation for T
    eos = PRSV2MIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00013018821346475254, P=1E6, kappa1s=[0.05104], kappa2s=[0.8634], kappa3s=[0.460])
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00013018821346475254, quick=False)
    assert_allclose(T_slow, 299)

    # Check this is the same as PRSV
    eos = PRSVMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6, kappa1s=[0.05104])
    three_props_PRSV = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    eos = PRSV2MIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6, kappa1s=[0.05104])
    three_props_PRSV2 = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    assert_allclose(three_props_PRSV, three_props_PRSV2)
    

def test_PRSV2MIX_quick():
    # Two-phase nitrogen-methane
    eos = PRSV2MIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(3.623552388375633e-05-1.3552527156068805e-19j), (0.0001942800283219127+2.168404344971009e-19j), (0.0007002421492037557-4.743384504624082e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [0.21897578471489654, -0.0006396067113212318, 3.7150128655294028e-06]
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [3.623552388375633e-05, 0.0007002421492037557]:
        eos = PRSV2MIX(V=V, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        assert_allclose(eos.T, 115)
        T_slow = eos.solve_T(P=1E6, V=V, quick=False)
        assert_allclose(T_slow, 115)


    # Fugacities
    eos = PRSV2MIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_l, [1.5881151663681092, 0.14570244654356812])
    assert_allclose(eos.phis_g, [0.8731073123670093, 0.7157562213377993])
    
    # Numerically test fugacities at one point
    def numerical_fugacity_coefficient(n1, n2=0.5, switch=False, l=True):
        if switch:
            n1, n2 = n2, n1
        tot = n1+n2
        zs = [i/tot for i in [n1,n2]]
        a = PRSVMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=zs, kijs=[[0,0],[0,0]])
        phi = a.phi_l if l else a.phi_g
        return tot*log(phi)

    phis = [[derivative(numerical_fugacity_coefficient, 0.5, dx=1E-6, order=25, args=(0.5, i, j)) for i in [False, True]] for j in [False, True]]
    assert_allclose(phis, [eos.lnphis_g, eos.lnphis_l])

    # Gas phase only test point
    eos = PRSV2MIX(T=300, P=1E7, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_g, [0.9859363207073754, 0.8344831291870667]) 


def test_TWUPRMIX_vs_TWUPR():
    # Copy and pasted
    # Test solution for molar volumes
    eos = TWUPRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.0001301754975832377+0j), (0.0011236542243270918+0.0012949257976571766j), (0.0011236542243270918-0.0012949257976571766j)]

    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.806982284033079, -0.006971709974815854, 2.3667018824561144e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T
    eos = TWUPRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.0001301754975832377, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001301754975832377, quick=False)
    assert_allclose(T_slow, 299)

    
    eos = TWUPRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    # Derivatives
    diffs_1 = [592877.7698667891, -3683686154532.3066, 1.6094687359218388e-07, -2.7146720921640605e-13, 6213230.351611896, 1.6866883037707508e-06]
    diffs_2 = [-708.101408196832, 4.512488462413035e+17, 1.168546207434993e-09, 9.027515426758444e-21, -280283966933.572, 3.397816790678971e-15]
    diffs_mixed = [-3.82370615408822e-15, -20741143317.758797, 0.07152333089484428]
    departures = [-31652.726391608117, -74.1128253091799, 35.189125483239366]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
        
    # Integration tests
    eos = TWUPRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # Superctitical test
    eos = TWUPRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=900., P=1E6)
    eos = TWUPRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.0073716980824289815, P=1E6)
    assert_allclose(eos.T, 900)


def test_TWUPRMIX_quick():
    # Two-phase nitrogen-methane
    eos = TWUPRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(3.62456981315702e-05-1.8973538018496328e-19j), (0.0001940721088661993+1.6263032587282567e-19j), (0.0007004398944116554-4.0657581468206416e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [0.21887729989547594, -0.0006338024691450689, 3.3584606049895343e-06]
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [3.62456981315702e-05, 0.0007004398944116554]:
        eos = TWUPRMIX(V=V, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        assert_allclose(eos.T, 115)
        T_slow = eos.solve_T(P=1E6, V=V, quick=False)
        assert_allclose(T_slow, 115)


    # Fugacities
    eos = TWUPRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_l, [1.5843100443266374, 0.14661177659453567])
    assert_allclose(eos.phis_g, [0.8729379355284885, 0.716098499114619])
    
    # Numerically test fugacities at one point
    def numerical_fugacity_coefficient(n1, n2=0.5, switch=False, l=True):
        if switch:
            n1, n2 = n2, n1
        tot = n1+n2
        zs = [i/tot for i in [n1,n2]]
        a = TWUPRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=zs, kijs=[[0,0],[0,0]])
        phi = a.phi_l if l else a.phi_g
        return tot*log(phi)

    phis = [[derivative(numerical_fugacity_coefficient, 0.5, dx=1E-6, order=25, args=(0.5, i, j)) for i in [False, True]] for j in [False, True]]
    assert_allclose(phis, [eos.lnphis_g, eos.lnphis_l])

    # Gas phase only test point
    eos = TWUPRMIX(T=300, P=1E7, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_g, [0.97697895693183, 0.8351530876083071]) 


def test_TWUSRKMIX_vs_TWUSRK():
    # Test solution for molar volumes
    eos = TWUSRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.00014689217317770398+0j), (0.001169565653511148+0.0013011778220658073j), (0.001169565653511148-0.0013011778220658073j)]

    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.71966709357206, -0.007269721309490377, 2.305588658885629e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T
    eos = TWUSRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00014689217317770398, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00014689217317770398, quick=False)
    assert_allclose(T_slow, 299)

    
    eos = TWUSRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    # Derivatives
    diffs_1 = [504446.40946384566, -2676841551251.3125, 1.8844836341846153e-07, -3.735745956022468e-13, 5306493.417400694, 1.982371132471449e-06]
    diffs_2 = [-586.164516927993, 2.6624358487625542e+17, 1.308861795972448e-09, 1.3880693263695398e-20, -195576504984.95178, 4.566404923205853e-15]
    diffs_mixed = [-5.015403880635795e-15, -14235388178.812284, 0.06816570409464781]
    departures = [-31612.591872087483, -74.02294100343829, 34.24266185576879]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
    
    # Integration tests
    eos = TWUSRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # Superctitical test
    eos = TWUSRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=900., P=1E6)
    eos = TWUSRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.007422210444471012, P=1E6)
    assert_allclose(eos.T, 900)


def test_TWUSRKMIX_quick():
    # Two-phase nitrogen-methane
    eos = TWUSRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(4.108791361639091e-05-1.0842021724855044e-19j), (0.00020336787935593015+1.8973538018496328e-19j), (0.0007117070840276789-2.0328790734103208e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [0.21019046890564463, -0.0007321997444448587, 2.6003157171957985e-06]
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [4.108791361639091e-05, 0.0007117070840276789]:
        eos = TWUSRKMIX(V=V, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        assert_allclose(eos.T, 115)
        T_slow = eos.solve_T(P=1E6, V=V, quick=False)
        assert_allclose(T_slow, 115)


    # Fugacities
    eos = TWUSRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_l, [1.6193856616533904, 0.14818727763145562])
    assert_allclose(eos.phis_g, [0.8835668629797101, 0.7249406348215529])
    
    # Numerically test fugacities at one point
    def numerical_fugacity_coefficient(n1, n2=0.5, switch=False, l=True):
        if switch:
            n1, n2 = n2, n1
        tot = n1+n2
        zs = [i/tot for i in [n1,n2]]
        a = TWUSRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=zs, kijs=[[0,0],[0,0]])
        phi = a.phi_l if l else a.phi_g
        return tot*log(phi)

    phis = [[derivative(numerical_fugacity_coefficient, 0.5, dx=1E-6, order=25, args=(0.5, i, j)) for i in [False, True]] for j in [False, True]]
    assert_allclose(phis, [eos.lnphis_g, eos.lnphis_l])

    # Gas phase only test point
    eos = TWUSRKMIX(T=300, P=1E7, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_g, [1.0172098207387374, 0.8799658237051876]) 


def test_APISRKMIX_vs_APISRK():
    # Test solution for molar volumes
    eos = APISRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.00014681823858766455+0j), (0.0011696026208061676+0.001304203394096485j), (0.0011696026208061676-0.001304203394096485j)]
    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.727474247064678, -0.0073349099227097685, 1.9482539852821945e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # SRK back calculation for T
    eos = APISRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00014681823858766455, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00014681823858766455, quick=False)
    assert_allclose(T_slow, 299)
    # with a S1 set
    eos = APISRKMIX(Tcs=[514.0], Pcs=[6137000], zs=[1], S1s=[1.678665], S2s=[-0.216396], P=1E6, V=7.045692682173252e-05)
    assert_allclose(eos.T, 299)
    eos = APISRKMIX(Tcs=[514.0], Pcs=[6137000], zs=[1], omegas=[0.635], S2s=[-0.216396], P=1E6, V=7.184691383223729e-05)
    assert_allclose(eos.T, 299)
    
    T_slow = eos.solve_T(P=1E6, V=7.184691383223729e-05, quick=False)
    assert_allclose(T_slow, 299)

    
    eos = APISRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    # Derivatives
    diffs_1 = [507160.19725861016, -2694519535687.8096, 1.8821915764257067e-07, -3.7112367780430196e-13, 5312955.453232907, 1.9717635678142185e-06]
    diffs_2 = [-495.7033432051597, 2.686049371238787e+17, 1.3462136329121424e-09, 1.3729982416974442e-20, -201893579486.30624, 3.80002419401769e-15]
    diffs_mixed = [-4.990227751881803e-15, -14325368140.50364, 0.06593414440492529]
    departures = [-31759.397282361704, -74.38420560550391, 28.946472091343608]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
            
    # Integration tests
    eos = APISRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]


def test_APISRKMIX_quick():
    # Two-phase nitrogen-methane
    eos = APISRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(4.101590920556748e-05-1.0842021724855044e-19j), (0.00020467837830150093+1.3552527156068805e-19j), (0.0007104685894929316-1.3552527156068805e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [0.21080673112868986, -0.0007639197799377851, 4.705533602981202e-06]
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [4.101590920556748e-05, 0.0007104685894929316]:
        eos = APISRKMIX(V=V, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        assert_allclose(eos.T, 115)
        T_slow = eos.solve_T(P=1E6, V=V, quick=False)
        assert_allclose(T_slow, 115)


    # Fugacities
    eos = APISRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_l, [1.6357646066980698, 0.14324096476246728])
    assert_allclose(eos.phis_g, [0.8843165822638349, 0.7230395975514106])
    
    # Numerically test fugacities at one point
    def numerical_fugacity_coefficient(n1, n2=0.5, switch=False, l=True):
        if switch:
            n1, n2 = n2, n1
        tot = n1+n2
        zs = [i/tot for i in [n1,n2]]
        a = APISRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=zs, kijs=[[0,0],[0,0]])
        phi = a.phi_l if l else a.phi_g
        return tot*log(phi)

    phis = [[derivative(numerical_fugacity_coefficient, 0.5, dx=1E-6, order=25, args=(0.5, i, j)) for i in [False, True]] for j in [False, True]]
    assert_allclose(phis, [eos.lnphis_g, eos.lnphis_l])

    # Gas phase only test point
    a = APISRKMIX(T=300, P=1E7, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(a.phis_g, [1.020708538988692, 0.8725461195162044]) 

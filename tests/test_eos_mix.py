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
import numpy as np


def test_PRMIX_quick():
    # Two-phase nitrogen-methane
    eos = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(3.625736293970586e-05-1.4456028966473392e-19j), (0.00019383473081158748+1.4456028966473392e-19j), (0.0007006659231347704-4.5175090520229353e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = (0.21876490011332972, -0.0006346637957108072, 3.6800265478701025e-06)
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [3.625736293970586e-05, 0.0007006659231347704]:
        eos = PRMIX(V=V, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        assert_allclose(eos.T, 115)
        T_slow = eos.solve_T(P=1E6, V=V, quick=False)
        assert_allclose(T_slow, 115)


    # Fugacities
    eos = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    assert_allclose(eos.phis_l, [1.5877216764229214, 0.14693710450607678])
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
    assert_allclose(e.V_g, 0.025451314884217785)
    assert_allclose(e.V_l, 0.00012128151502941696)
    
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


def test_many_components():
    liquid_IDs = ['nitrogen', 'carbon dioxide', 'H2S', 'methane', 'ethane', 'propane', 'isobutane', 'butane', 'isopentane', 'pentane', 'Hexane', 'Heptane', 'Octane', 'Nonane', 'Decane', 'Undecane', 'Dodecane', 'Tridecane', 'Tetradecane', 'Pentadecane', 'Hexadecane', 'Heptadecane', 'Octadecane', 'Nonadecane', 'Eicosane', 'Heneicosane', 'Docosane', 'Tricosane', 'Tetracosane', 'Pentacosane', 'Hexacosane', 'Heptacosane', 'Octacosane', 'Nonacosane', 'Triacontane', 'Benzene', 'Toluene', 'Ethylbenzene', 'Xylene', '1,2,4-Trimethylbenzene', 'Cyclopentane', 'Methylcyclopentane', 'Cyclohexane', 'Methylcyclohexane']
    zs = [9.11975115499676e-05, 9.986813065240533e-05, 0.0010137795304828892, 0.019875879000370657, 0.013528874875432457, 0.021392773691700402, 0.00845450438914824, 0.02500218071904368, 0.016114189201071587, 0.027825798446635016, 0.05583179467176313, 0.0703116540769539, 0.07830577180555454, 0.07236459223729574, 0.0774523322851419, 0.057755091407705975, 0.04030134965162674, 0.03967043780553758, 0.03514481759005302, 0.03175471055284055, 0.025411123554079325, 0.029291866298718154, 0.012084986551713202, 0.01641114551124426, 0.01572454598093482, 0.012145363820829673, 0.01103585282423499, 0.010654818322680342, 0.008777712911254239, 0.008732073853067238, 0.007445155260036595, 0.006402875549212365, 0.0052908087849774296, 0.0048199150683177075, 0.015943943854195963, 0.004452253754752775, 0.01711981267072777, 0.0024032720444511282, 0.032178399403544646, 0.0018219517069058137, 0.003403378548794345, 0.01127516775495176, 0.015133143423489698, 0.029483213283483682]
    Tcs = [126.2, 304.2, 373.2, 190.56400000000002, 305.32, 369.83, 407.8, 425.12, 460.4, 469.7, 507.6, 540.2, 568.7, 594.6, 611.7, 639.0, 658.0, 675.0, 693.0, 708.0, 723.0, 736.0, 747.0, 755.0, 768.0, 778.0, 786.0, 790.0, 800.0, 812.0, 816.0, 826.0, 824.0, 838.0, 843.0, 562.05, 591.75, 617.15, 630.3, 649.1, 511.7, 553.8, 532.7, 572.1]
    Pcs = [3394387.5, 7376460.0, 8936865.0, 4599000.0, 4872000.0, 4248000.0, 3640000.0, 3796000.0, 3380000.0, 3370000.0, 3025000.0, 2740000.0, 2490000.0, 2290000.0, 2110000.0, 1980000.0, 1820000.0, 1680000.0, 1570000.0, 1480000.0, 1400000.0, 1340000.0, 1290000.0, 1160000.0, 1070000.0, 1030000.0, 980000.0, 920000.0, 870000.0, 950000.0, 800000.0, 883000.0, 800000.0, 826000.0, 600000.0, 4895000.0, 4108000.0, 3609000.0, 3732000.0, 3232000.0, 4510000.0, 4080000.0, 3790000.0, 3480000.0]
    omegas = [0.04, 0.2252, 0.1, 0.008, 0.098, 0.152, 0.17600000000000002, 0.193, 0.22699999999999998, 0.251, 0.2975, 0.3457, 0.39399999999999996, 0.444, 0.49, 0.535, 0.562, 0.623, 0.679, 0.6897, 0.742, 0.7564, 0.8087, 0.8486, 0.8805, 0.9049, 0.9423, 1.0247, 1.0411, 1.105, 1.117, 1.214, 1.195, 1.265, 1.26, 0.212, 0.257, 0.301, 0.3118, 0.3771, 0.1921, 0.239, 0.213, 0.2477]
    eos = PRMIX(T=300, P=1e5, zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas)
    
    
    assert_allclose(eos.V_g, 0.019092551264336028)
    assert_allclose(eos.V_l, 0.0002453974598582871)

    assert_allclose(eos.a_alpha, 11.996512274167202)
    assert_allclose(eos.da_alpha_dT, -0.0228875173310534)
    assert_allclose(eos.d2a_alpha_dT2, 5.997880989552689e-05)
    
    V_over_F, xs, ys = eos.sequential_substitution_VL()
    assert_allclose(V_over_F, 0.03547152723457448, rtol=5e-5)
    assert_allclose(xs, [5.729733527056475e-06, 4.0516737456029636e-05, 0.0006069285358060455, 0.0030221509527402807, 0.006670434145198826, 0.016845301723389206, 0.007760188627667261, 0.023716884273864994, 0.016120427201854567, 0.028071761635467454, 0.05738553540904704, 0.07269474153625576, 0.08111242740513086, 0.07500425089850274, 0.08029153901604559, 0.05987718593915933, 0.04178296619077928, 0.04112921011785296, 0.036437266797871466, 0.03292250673929231, 0.02634563984465752, 0.03036910265441899, 0.01252942406087858, 0.017014682097712515, 0.01630283227693387, 0.012592022051117679, 0.011441707659942228, 0.011046660229078003, 0.009100522335947752, 0.009053204855420115, 0.007718958506744987, 0.006638347887060363, 0.005485383723073785, 0.004997172405387115, 0.016530298798161255, 0.004574527233511734, 0.017701045589161052, 0.0024893880550883388, 0.033338891413257424, 0.0018885749821301962, 0.003452886350289371, 0.011614131308001385, 0.01553751739014118, 0.030473502992154296],
                    rtol=5e-5, atol=1e-5)
    assert_allclose(ys, [0.0024152052956508234, 0.0017137289677579672, 0.01207671927772537, 0.47815613478650093, 0.2000210244617312, 0.14504592692452017, 0.02733407832487352, 0.05995147988851196, 0.01594456782197141, 0.02113765961031682, 0.013583058932377816, 0.005511633136594507, 0.0019882447609517653, 0.0005879833453478944, 0.0002496870786300975, 5.189233339016621e-05, 1.3788073507311385e-05, 4.047880869664237e-06, 1.0231799554219332e-06, 4.3437603783102945e-07, 1.0686553748606369e-07, 5.9095141645558586e-08, 8.391056490347942e-09, 4.875695250293468e-09, 1.7792547398641979e-09, 6.437996463823593e-10, 2.5830362538596066e-10, 7.806691559916385e-11, 3.36860845201539e-11, 6.662408195909387e-12, 5.247905701692434e-12, 5.760475376250616e-13, 8.102134731211449e-13, 1.1667142269975863e-13, 1.390262805287062e-12, 0.0011274391521227964, 0.0013151450162989817, 6.163776207935758e-05, 0.0006227356628028977, 1.035657941516073e-05, 0.0020571809675571477, 0.002058197874178186, 0.004137558093848116, 0.0025556267157302547],
                    rtol=5e-5, atol=1e-5)
    
    V_over_F, xs, ys = PRMIX(T=669.1, P=3.25e6, zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas).sequential_substitution_VL()
    assert_allclose(V_over_F, 0.341342933080815, rtol=1e-4)

    V_over_F, xs, ys = PRMIX(T=669.1, P=3.19e6, zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas).sequential_substitution_VL()
    assert_allclose(V_over_F, 0.40427364770048313, rtol=1e-4)

    V_over_F, xs, ys = PRMIX(T=660, P=3.2e6, zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas).sequential_substitution_VL()
    assert_allclose(V_over_F, 0.27748085589254096, rtol=1e-4)

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
    assert_allclose(eos.dP_drho_g, 77.71290396836963)
    assert_allclose(eos.dP_drho_l, 1712.0682171304466)


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
    assert_allclose(P_transition, 2703430.0056912485)
    
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
    assert_allclose(eos.V_g_extrapolated(), 0.0005133249130364282)



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
    
def test_sequential_substitution_VL():
    omegas = [0.2252, 0.2975]
    Tcs = [304.2, 507.4]
    Pcs = [7.38E6, 3.014E6]
    kijs=[[0,0],[0,0]]
    
    eos = PRMIX(T=313.0, P=1E6, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=[.5, .5], kijs=kijs)
    V_over_F, xs, ys = eos.sequential_substitution_VL()
    assert_allclose(V_over_F, 0.4128783868475084)
    assert_allclose(xs, [0.17818473607425783, 0.8218152639257423])
    assert_allclose(ys, [0.9576279672468324, 0.04237203275316752])    
    
    
def test_TPD_stuff():
    # Two-phase nitrogen-methane
    eos = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    # Get a negative TPD proving there is a phase split
    TPD = eos.TPD(eos.Z_g, eos.Z_l, eos.zs, eos.zs)
    assert_allclose(TPD, -471.36299561394253)
    
    



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
    Vs_expected = [(0.00013022212513965833+0j), (0.001123631313468268+0.0012926967234386066j), (0.001123631313468268-0.0012926967234386066j)]
    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = (3.8012620034344384, -0.006647930535193548, 1.693013909536469e-05)
    a_alphas_fast = eos.a_alpha_and_derivatives(299, quick=True)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)
    
    # PR back calculation for T
    eos = PRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00013022212513965833, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00013022212513965833, quick=False)
    assert_allclose(T_slow, 299)
    
    
    diffs_1 = [582232.4757941177, -3665179372374.2607, 1.5885511093470827e-07, -2.728379428132085e-13, 6295044.547927793, 1.717527004374123e-06]
    diffs_2 = [-506.20125231401545, 4.482162818098147e+17, 1.1688517647207335e-09, 9.103364399605293e-21, -291578743623.699, 2.56468444397071e-15]
    diffs_mixed = [-3.772509038556631e-15, -20523296734.825638, 0.0699416812561707]
    departures = [-31134.75084346042, -72.47561931957617, 25.165386034971867]
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
    expect_props = [8.35196289693885e-05, -63764.67109328409, -130.7371532254518]
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
    Vs_expected = [(4.104756961475803e-05-1.0842021724855044e-19j), (0.00020409982649503516+1.4456028966473392e-19j), (0.0007110158049778292-1.807003620809174e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = (0.21053508135768303, -0.0007568164048417844, 4.650780763765838e-06)
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [4.104756961475803e-05, 0.0007110158049778292]:
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
    Vs_expected = [(0.0001468210773547259+0j), (0.0011696016227365465+0.001304089515440735j), (0.0011696016227365465-0.001304089515440735j)]
    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = (3.72718144448615, -0.007332994130304653, 1.9476133436500582e-05)
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # PR back calculation for T
    eos = SRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.0001468210773547259, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001468210773547259, quick=False)
    assert_allclose(T_slow, 299)
    
    # Derivatives
    diffs_1, diffs_2, diffs_mixed, departures = ([507071.3781579619, -2693848855910.751, 1.8823304694521492e-07, -3.712160753955569e-13, 5312563.421932224, 1.97210894377967e-06],
                                                 [-495.5254299681785, 2.6851518388403037e+17, 1.3462644444996599e-09, 1.3735648667748027e-20, -201856509533.58496, 3.800656805086307e-15],
                                                 [-4.9913489930067516e-15, -14322101736.003756, 0.06594010907198579],
                                                 [-31754.663859649736, -74.37327204447028, 28.936530624645137])
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
    expect_props = [0.00022332985608164609, -13385.727374687076, -32.65923125080434]
    assert_allclose(three_props, expect_props)
    
    # Test of a_alphas
    a_alphas = [2.4841053385218554, 0, 0]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # Back calculation for P
    eos = VDWMIX(Tcs=[507.6], Pcs=[3025000], zs=[1], T=299, V=0.00022332985608164609)
    assert_allclose(eos.P, 1E6)
    
    # Back calculation for T
    eos = VDWMIX(Tcs=[507.6], Pcs=[3025000], zs=[1], P=1E6, V=0.00022332985608164609)
    assert_allclose(eos.T, 299)


def test_VDWIX_quick():
    # Two-phase nitrogen-methane
    eos = VDWMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(5.881369844882989e-05-1.8070036208091741e-19j), (0.00016108242301576215+1.4456028966473392e-19j), (0.0007770872375800777-3.162256336416055e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [0.18035232263614895, 0.0, 0.0]
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [5.881369844882989e-05, 0.0007770872375800777]:
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
    expect_props = [0.0001301269135543934, -31698.926746698795, -74.16751538228138]
    assert_allclose(three_props, expect_props)
    
    # Test of a_alphas
    a_alphas = [3.812985698311453, -0.006976903474851659, 2.0026560811043733e-05]
    
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    
    a_alphas_fast = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # PR back calculation for T
    eos = PRSVMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.0001301269135543934, P=1E6, kappa1s=[0.05104])
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001301269135543934, quick=False)
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
    Vs_expected = [(3.623553616564366e-05-1.4456028966473392e-19j), (0.00019428009417235927+1.4456028966473392e-19j), (0.0007002423865480607-2.710505431213761e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = (0.21897593315687267, -0.0006396071449056316, 3.715015383907643e-06)
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [3.623553616564366e-05, 0.0007002423865480607]:
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
    expect_props = [0.00013018825759153257, -31496.184168729033, -73.6152829631142]
    assert_allclose(three_props, expect_props)
    
    # Test of PRSV2 a_alphas
    a_alphas = (3.80542021117275, -0.006873163375791913, 2.3078023705053794e-05)
    
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_fast = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # PSRV2 back calculation for T
    eos = PRSV2MIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00013018825759153257, P=1E6, kappa1s=[0.05104], kappa2s=[0.8634], kappa3s=[0.460])
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00013018825759153257, quick=False)
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
    Vs_expected = [(3.623553616564366e-05-1.4456028966473392e-19j), (0.00019428009417235927+1.4456028966473392e-19j), (0.0007002423865480607-2.710505431213761e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = (0.21897593315687267, -0.0006396071449056315, 3.715015383907642e-06)
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [3.623553616564366e-05, 0.0007002423865480607]:
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
    Vs_expected = [(0.00013017554170570767+0j), (0.0011236546051852433+0.0012949262365671505j), (0.0011236546051852433-0.0012949262365671505j)]

    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = (3.8069848647566698, -0.006971714700883658, 2.366703486824857e-05)
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T
    eos = TWUPRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00013017554170570767, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00013017554170570767, quick=False)
    assert_allclose(T_slow, 299)

    
    eos = TWUPRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    # Derivatives
    diffs_1, diffs_2, diffs_mixed, departures = ([592877.7698667824, -3683684905961.741, 1.6094692814449423e-07, -2.7146730122915294e-13, 6213228.245662597, 1.6866883037707698e-06], 
                                                 [-708.1014081968287, 4.512485403434166e+17, 1.1685466035091765e-09, 9.027518486599707e-21, -280283776931.3797, 3.3978167906790706e-15], 
                                                 [-3.823707450118526e-15, -20741136287.632187, 0.0715233066523022], 
                                                 [-31652.73712017438, -74.1128504294285, 35.18913741045412])
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
    eos = TWUPRMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.007371700581036866, P=1E6)
    assert_allclose(eos.T, 900)


def test_TWUPRMIX_quick():
    # Two-phase nitrogen-methane
    eos = TWUPRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(3.624571041690618e-05-1.264902534566422e-19j), (0.00019407217464617222+1.4456028966473392e-19j), (0.0007004401318229852-3.614007241618348e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = (0.21887744827068994, -0.0006338028987948183, 3.358462881663777e-06)
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [3.624571041690618e-05, 0.0007004401318229852]:
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
    Vs_expected = [0.00014689222296622483, (0.0011695660499307968+0.0013011782630948806j), (0.0011695660499307968-0.0013011782630948806j)]

    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = (3.7196696151053654, -0.00726972623757774, 2.3055902218261955e-05)
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T
    eos = TWUSRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00014689222296622483, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00014689222296622483, quick=False)
    assert_allclose(T_slow, 299)

    
    eos = TWUSRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    # Derivatives
    diffs_1, diffs_2, diffs_mixed, departures = ([504446.40946384973, -2676840643946.846, 1.884484272922847e-07, -3.735747222238669e-13, 5306491.618786469, 1.982371132471433e-06], 
                                                 [-586.1645169279951, 2.662434043919377e+17, 1.3088622396059171e-09, 1.388069796850075e-20, -195576372405.25793, 4.566404923205759e-15], 
                                                 [-5.0154055805868715e-15, -14235383353.785719, 0.06816568099016031], 
                                                 [-31612.602587050424, -74.02296609322131, 34.24267346218357])
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
    eos = TWUSRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.007422212960199866, P=1E6)
    assert_allclose(eos.T, 900)


def test_TWUSRKMIX_quick():
    # Two-phase nitrogen-methane
    eos = TWUSRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])

    Vs_fast = eos.volume_solutions(115, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(4.108792754297647e-05-7.228014483236696e-20j), (0.00020336794828666816+7.228014483236696e-20j), (0.0007117073252579778-9.03501810404587e-21j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = (0.2101906113921238, -0.0007322002407973534, 2.600317479929538e-06)
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [4.108792754297647e-05, 0.0007117073252579778]:
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
    Vs_expected = [(0.00014681828835112518+0j), (0.0011696030172383468+0.0013042038361510636j), (0.0011696030172383468-0.0013042038361510636j)]
    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = (3.727476773890392, -0.007334914894987986, 1.9482553059883725e-05)
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # SRK back calculation for T
    eos = APISRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], V=0.00014681828835112518, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00014681828835112518, quick=False)
    assert_allclose(T_slow, 299)
    # with a S1 set
    eos = APISRKMIX(Tcs=[514.0], Pcs=[6137000], zs=[1], S1s=[1.678665], S2s=[-0.216396], P=1E6, V=7.045695070282895e-05)
    assert_allclose(eos.T, 299)
    eos = APISRKMIX(Tcs=[514.0], Pcs=[6137000], zs=[1], omegas=[0.635], S2s=[-0.216396], P=1E6, V=7.184693818446427e-05)
    assert_allclose(eos.T, 299)
    
    T_slow = eos.solve_T(P=1E6, V=7.184693818446427e-05, quick=False)
    assert_allclose(T_slow, 299)

    
    eos = APISRKMIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6)
    # Derivatives
    diffs_1 = [507160.1972586132, -2694518622391.442, 1.882192214387065e-07, -3.7112380359519615e-13, 5312953.652428371, 1.9717635678142066e-06]
    diffs_2 = [-495.70334320516093, 2.6860475503881738e+17, 1.3462140892058854e-09, 1.3729987070697146e-20, -201893442624.31924, 3.80002419401763e-15]
    diffs_mixed = [-4.990229443299593e-15, -14325363284.978655, 0.06593412205681572]
    departures = [-31759.40804708375, -74.3842308177361, 28.946481902635792]
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
    Vs_expected = [(4.1015923107747434e-05-7.228014483236696e-20j), (0.00020467844767642728+7.228014483236696e-20j), (0.0007104688303034478-2.710505431213761e-20j)]
    assert_allclose(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = (0.2108068740329283, -0.0007639202977930443, 4.705536792825722e-06)
    a_alphas_fast = eos.a_alpha_and_derivatives(115)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(115, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T, both solutions
    for V in [4.1015923107747434e-05, 0.0007104688303034478]:
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


def test_fugacities_PR_vs_coolprop():
    import CoolProp.CoolProp as CP
        
    zs = [0.4, 0.6]
    Tcs = [305.322, 540.13]
    Pcs = [4872200.0, 2736000.0]
    omegas = [0.099, 0.349]
    kij = .0067
    kijs = [[0,kij],[kij,0]]
    c1, c2 = PRMIX.c1, PRMIX.c2
    # match coolprop
    PRMIX.c1, PRMIX.c2 = 0.45724, 0.07780

    T, P = 300, 1e5
    eos = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)

    AS = CP.AbstractState("PR", "Ethane&Heptane")
    AS.set_mole_fractions(zs)
    AS.set_binary_interaction_double(0,1,"kij", kij)
    AS.update(CP.PT_INPUTS, P, T)

    fugacities_CP = [AS.fugacity(0), AS.fugacity(1)]
    assert_allclose(fugacities_CP, eos.fugacities_g, rtol=1e-13)

    T, P = 300, 1e6
    eos = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    AS.update(CP.PT_INPUTS, P, T)
    fugacities_CP = [AS.fugacity(0), AS.fugacity(1)]
    assert_allclose(fugacities_CP, eos.fugacities_l, rtol=1e-13)
    
    # Set the coefficients back
    PRMIX.c1, PRMIX.c2 = c1, c2


def test_fugacities_SRK_vs_coolprop():
    import CoolProp.CoolProp as CP
    zs = [0.4, 0.6]
    Tcs = [305.322, 540.13]
    Pcs = [4872200.0, 2736000.0]
    omegas = [0.099, 0.349]
    kij = .0067
    kijs = [[0,kij],[kij,0]]
    c1, c2 = SRKMIX.c1, SRKMIX.c2
    # match coolprop
    SRKMIX.c1, SRKMIX.c2 = 0.42747, 0.08664
    
    T, P = 300, 1e5
    eos = SRKMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    
    AS = CP.AbstractState("SRK", "Ethane&Heptane")
    AS.set_mole_fractions(zs)
    AS.set_binary_interaction_double(0,1,"kij", kij)
    AS.update(CP.PT_INPUTS, P, T)
    
    fugacities_CP = [AS.fugacity(0), AS.fugacity(1)]
    assert_allclose(fugacities_CP, eos.fugacities_g, rtol=1e-13)
    
    T, P = 300, 1e6
    eos = SRKMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    AS.update(CP.PT_INPUTS, P, T)
    fugacities_CP = [AS.fugacity(0), AS.fugacity(1)]
    assert_allclose(fugacities_CP, eos.fugacities_l, rtol=1e-13)
    
    # Set the coefficients back
    SRKMIX.c1, SRKMIX.c2 = c1, c2


def test_Z_derivative_T():
    from fluids.constants import R
    T = 115
    dT = 1e-5
    P = 1e6
    eos1 = PRMIX(T=T, P=P, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    eos2 = PRMIX(T=T+dT, P=P, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    
    dZ_dT_numerical = (eos2.Z_g - eos1.Z_g)/dT
    dZ_dT_analytical = P/R*(-eos1.V_g*T**-2 + eos1.dV_dT_g/T)
    
    assert_allclose(dZ_dT_numerical, dZ_dT_analytical, rtol=1e-6)
    assert_allclose(eos1.dZ_dT_g, 0.008538861194633872, rtol=1e-11)
    
    dZ_dT_numerical = (eos2.Z_l - eos1.Z_l)/dT
    dZ_dT_analytical = P/R*(-eos1.V_l*T**-2 + eos1.dV_dT_l/T)
    
    assert_allclose(dZ_dT_numerical, dZ_dT_analytical, rtol=1e-5)
    assert_allclose(eos1.dZ_dT_l, -5.234447550711918e-05, rtol=1e-11)
    
def test_Z_derivative_P():
    from fluids.constants import R

    T = 115
    dP = 1e-2
    P = 1e6
    eos1 = PRMIX(T=T, P=P, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    eos2 = PRMIX(T=T, P=P + dP, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    
    dZ_dP_numerical = (eos2.Z_g - eos1.Z_g)/dP
    dZ_dP_analytical = 1/(T*R)*(eos1.V_g + P*eos1.dV_dP_g)
    
    assert_allclose(dZ_dP_analytical, dZ_dP_numerical)
    assert_allclose(eos1.dZ_dP_g, dZ_dP_numerical)
    
    dZ_dP_numerical = (eos2.Z_l - eos1.Z_l)/dP
    dZ_dP_analytical = 1/(T*R)*(eos1.V_l + P*eos1.dV_dP_l)
    
    assert_allclose(dZ_dP_analytical, dZ_dP_numerical)
    assert_allclose(eos1.dZ_dP_l, dZ_dP_numerical)
    
def test_PR_d_lbphis_dT():
    dT = 1e-6
    T = 270.0
    P = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    eos = eos1 = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    eos2 = PRMIX(T=T + dT, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    numerical_diffs = (np.array(eos2.lnphis_g) - eos1.lnphis_g) / dT
    
    
    expected_diffs = [-0.0125202946344780, -0.00154326287196778, 0.0185468995722353]
    analytical_diffs = eos.d_lnphis_dT(Z=eos.Z_g, dZ_dT=eos.dZ_dT_g, zs=zs)
    assert_allclose(analytical_diffs, expected_diffs, rtol=1e-11)
    assert_allclose(expected_diffs, numerical_diffs, rtol=1e-5)

@pytest.mark.sympy
def test_PR_d_lbphis_dT_sympy():
    from sympy import Derivative, symbols, sqrt, diff, log, N
    from fluids.constants import R as R_num
    T_num = 270.0
    P_num = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    eos = PRMIX(T=T_num, P=P_num, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    diffs_implemented = eos.d_lnphis_dT(Z=eos.Z_g, dZ_dT=eos.dZ_dT_g, zs=zs)

    # Symbolic part
    T, P, R, a_alpha_f, b1, b2, b3, b, = symbols('T, P, R, a_alpha_f, b1, b2, b3, b')
    Z_f, sum_f = symbols('Z_f, sum_f')
    
    root_two = sqrt(2)
    two_root_two = 2*sqrt(2)
    a_alpha, sum_fun = a_alpha_f(T), sum_f(T)
    
    A = a_alpha*P/(R*R*T*T)
    B = b*P/(R*T)
    A, B
    
    Z = Z_f(T) # Change to f(P) when doing others
    
    needed = []
    for bi in [b1, b2, b3]:
        t1 = bi/b*(Z - 1) - log(Z - B)
        t2 = 2/a_alpha*sum_fun
        t3 = t1 - A/(two_root_two*B)*(t2 - bi/b)*log((Z + (root_two + 1)*B)/(Z - (root_two - 1)*B))
        needed.append(diff(t3, T))


    
    sympy_diffs = []
    for i in range(3):
        subs = {Derivative(Z_f(T), T): eos.dZ_dT_g, 
                R: R_num, 'b': eos.b,
                Z_f(T): eos.Z_g, 
                a_alpha: eos.a_alpha, 
                Derivative(a_alpha_f(T), T) : eos.da_alpha_dT,
                Derivative(sum_f(T), T): sum([zs[j]*eos.da_alpha_dT_ijs[i][j] for j in eos.cmps]),
                }
        subs1 = {sum_f(T): sum([zs[j]*eos.a_alpha_ijs[i][j] for j in eos.cmps])}
    
        subs2 = {P: eos.P, 
                 T: eos.T,
                 {0: b1, 1:b2, 2:b3}[i]: eos.bs[i]}
    
        working = needed[i].subs(subs)
        working = working.subs(subs1)
        working = working.subs(subs2)
        
        sympy_diffs.append(float(N(working)))
    
    assert_allclose(diffs_implemented, sympy_diffs, rtol=1e-10)
    
def test_SRK_d_lbphis_dT():
    dT = 1e-6
    T = 270.0
    P = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    eos = eos1 = SRKMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    eos2 = SRKMIX(T=T + dT, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    numerical_diffs = (np.array(eos2.lnphis_g) - eos1.lnphis_g) / dT
    
    
    expected_diffs = [-0.013492637405729917, -0.002035560753468637, 0.019072382634936852]
    analytical_diffs = eos.d_lnphis_dT(Z=eos.Z_g, dZ_dT=eos.dZ_dT_g, zs=zs)
    assert_allclose(analytical_diffs, expected_diffs, rtol=1e-11)
    assert_allclose(expected_diffs, numerical_diffs, rtol=1e-5)


@pytest.mark.sympy
def test_SRK_d_lbphis_dT_sympy():
    from sympy import Derivative, symbols, sqrt, diff, log, N
    from fluids.constants import R as R_num
    T_num = 270.0
    P_num = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    eos = SRKMIX(T=T_num, P=P_num, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    diffs_implemented = eos.d_lnphis_dT(Z=eos.Z_g, dZ_dT=eos.dZ_dT_g, zs=zs)

    # Symbolic part
    T, P, R, a_alpha_f, b1, b2, b3, b, = symbols('T, P, R, a_alpha_f, b1, b2, b3, b')
    Z_f, sum_f = symbols('Z_f, sum_f')
    a_alpha, sum_fun = a_alpha_f(T), sum_f(T)
    
    A = a_alpha*P/(R*R*T*T)
    B = b*P/(R*T)
    A, B
    
    Z = Z_f(T) # Change to f(P) when doing others
    
    needed = []
    for bi in [b1, b2, b3]:
        Bi = bi*P/R/T
        t1 = Bi/B*(Z-1) - log(Z - B)
        t3 = log(1. + B/Z)
        t2 = A/B*(Bi/B - 2./a_alpha*sum_fun)
        t4 = t1 + t2*t3
        needed.append(diff(t4, T))


    
    sympy_diffs = []
    for i in range(3):
        subs = {Derivative(Z_f(T), T): eos.dZ_dT_g, 
                R: R_num, 'b': eos.b,
                Z_f(T): eos.Z_g, 
                a_alpha: eos.a_alpha, 
                Derivative(a_alpha_f(T), T) : eos.da_alpha_dT,
                Derivative(sum_f(T), T): sum([zs[j]*eos.da_alpha_dT_ijs[i][j] for j in eos.cmps]),
                }
        subs1 = {sum_f(T): sum([zs[j]*eos.a_alpha_ijs[i][j] for j in eos.cmps])}
    
        subs2 = {P: eos.P, 
                 T: eos.T,
                 {0: b1, 1:b2, 2:b3}[i]: eos.bs[i]}
    
        working = needed[i].subs(subs)
        working = working.subs(subs1)
        working = working.subs(subs2)
        
        sympy_diffs.append(float(N(working)))
    
    assert_allclose(diffs_implemented, sympy_diffs, rtol=1e-10)

def test_VDW_d_lnphis_dT():
    dT = 1e-5
    T = 280.0 # Need increase T a little
    P = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    eos = eos1 = VDWMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    eos2 = VDWMIX(T=T + dT, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    numerical_diffs = (np.array(eos2.lnphis_g) - eos1.lnphis_g) / dT
    
    expected_diffs =[-0.00457948776557392, 0.000404824835196203, 0.0105772883904069]
    analytical_diffs = eos.d_lnphis_dT(Z=eos.Z_g, dZ_dT=eos.dZ_dT_g, zs=zs)
    
    assert_allclose(analytical_diffs, expected_diffs, rtol=1e-11)
    assert_allclose(expected_diffs, numerical_diffs, rtol=1e-5)

@pytest.mark.sympy
def test_VDW_d_lnphis_dT_sympy():
    from sympy import Derivative, symbols, sqrt, diff, log, N
    from fluids.constants import R as R_num
    T_num = 280.0
    P_num = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    eos = VDWMIX(T=T_num, P=P_num, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    diffs_implemented = eos.d_lnphis_dT(Z=eos.Z_g, dZ_dT=eos.dZ_dT_g, zs=zs)

    # Symbolic part
    T, P, R, a_alpha_f, b1, b2, b3, b, = symbols('T, P, R, a_alpha_f, b1, b2, b3, b')
    a1, a2, a3 = symbols('a1, a2, a3')
    Z_f, sum_f = symbols('Z_f, sum_f')
    a_alpha, sum_fun = a_alpha_f(T), sum_f(T)
    
    A = a_alpha*P/(R*R*T*T)
    B = b*P/(R*T)
    A, B
    
    Z = Z_f(T) # Change to f(P) when doing others
    
    needed = []
    for ai, bi in zip([a1, a2, a3], [b1, b2, b3]):
        V = Z*R*T/P
        t1 = log(Z*(1 - b/V))
        t2 = 2/(R*T*V)
        t3 = 1/(V - b)
        logphi = (bi*t3 - t1 - t2*sqrt(a_alpha*ai))
        
        needed.append(diff(logphi, T))


    
    sympy_diffs = []
    for i in range(3):
        subs = {Derivative(Z_f(T), T): eos.dZ_dT_g, 
                R: R_num, 'b': eos.b,
                Z_f(T): eos.Z_g, 
                a_alpha: eos.a_alpha, 
                Derivative(a_alpha_f(T), T) : eos.da_alpha_dT,
                Derivative(sum_f(T), T): sum([zs[j]*eos.da_alpha_dT_ijs[i][j] for j in eos.cmps]),
                }
        subs1 = {sum_f(T): sum([zs[j]*eos.a_alpha_ijs[i][j] for j in eos.cmps])}
    
        subs2 = {P: eos.P, 
                 T: eos.T,
                 {0: b1, 1:b2, 2:b3}[i]: eos.bs[i],
                 {0: a1, 1:a2, 2:a3}[i]: eos.ais[i]}
    
        working = needed[i].subs(subs)
        working = working.subs(subs1)
        working = working.subs(subs2)
        
        sympy_diffs.append(float(N(working)))
    
    assert_allclose(diffs_implemented, sympy_diffs, rtol=1e-10)


def test_PR_d_lnphis_dP():
    dP = 1e-1
    T = 270.0
    P = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    eos = eos1 = PRMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    eos2 = PRMIX(T=T, P=P + dP, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    numerical_diffs = (np.array(eos2.lnphis_g) - eos1.lnphis_g) / dP
    
    expected_diffs = [8.49254218440054e-8, -3.44512799711331e-9, -1.52343107476988e-7]
    analytical_diffs = eos.d_lnphis_dP(Z=eos.Z_g, dZ_dP=eos.dZ_dP_g, zs=zs)

    assert_allclose(analytical_diffs, expected_diffs, rtol=1e-11)
    assert_allclose(expected_diffs, numerical_diffs, rtol=1e-5)


@pytest.mark.sympy
def test_PR_d_lnphis_dP_sympy():
    from sympy import Derivative, symbols, sqrt, diff, log, N
    from fluids.constants import R as R_num
    T_num = 270.0
    P_num = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    
    eos = PRMIX(T=T_num, P=P_num, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    diffs_implemented = eos.d_lnphis_dP(Z=eos.Z_g, dZ_dP=eos.dZ_dP_g, zs=zs)

    T, P, R, a_alpha_f, b1, b2, b3, b, = symbols('T, P, R, a_alpha_f, b1, b2, b3, b')
    Z_f, sum_f = symbols('Z_f, sum_f')
    
    root_two = sqrt(2)
    two_root_two = 2*sqrt(2)
    a_alpha, sum_fun = symbols('a_alpha, sum_fun')
    
    A = a_alpha*P/(R*R*T*T)
    B = b*P/(R*T)
    A, B
    
    Z = Z_f(P) # Change to f(P) when doing others
    
    needed = []
    for bi in [b1, b2, b3]:
        t1 = bi/b*(Z - 1) - log(Z - B)
        t2 = 2/a_alpha*sum_fun
        t3 = t1 - A/(two_root_two*B)*(t2 - bi/b)*log((Z + (root_two + 1)*B)/(Z - (root_two - 1)*B))
        needed.append(diff(t3, P))

    sympy_diffs = []
    for i in range(3):
        subs = {Derivative(Z_f(P), P): eos.dZ_dP_g, 
                R: R_num, 'b': eos.b,
                Z_f(P): eos.Z_g, 
                a_alpha: eos.a_alpha, 
                sum_fun: sum([zs[j]*eos.a_alpha_ijs[i][j] for j in eos.cmps])
                }
    
        subs2 = {P: eos.P, 
                 T: eos.T,
                 {0: b1, 1:b2, 2:b3}[i]: eos.bs[i]}
    
        working = needed[i].subs(subs)
        working = working.subs(subs2)
    
        sympy_diffs.append(float((N(working))))
    assert_allclose(sympy_diffs, diffs_implemented, rtol=1e-11)
    
    
def test_SRK_d_lnphis_dP():
    dP = 1e-1
    T = 270.0
    P = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    eos = eos1 = SRKMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    eos2 = SRKMIX(T=T, P=P + dP, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    numerical_diffs = (np.array(eos2.lnphis_g) - eos1.lnphis_g) / dP
    
    expected_diffs = [9.40786516520732e-8, 3.03133250647420e-9, -1.51771425140191e-7]
    analytical_diffs = eos.d_lnphis_dP(Z=eos.Z_g, dZ_dP=eos.dZ_dP_g, zs=zs)
    assert_allclose(analytical_diffs, expected_diffs, rtol=1e-11)
    assert_allclose(expected_diffs, numerical_diffs, rtol=1e-5)



@pytest.mark.sympy
def test_SRK_d_lnphis_dP_sympy():
    from sympy import Derivative, symbols, sqrt, diff, log, N
    from fluids.constants import R as R_num
    T_num = 270.0
    P_num = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    
    eos = SRKMIX(T=T_num, P=P_num, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    diffs_implemented = eos.d_lnphis_dP(Z=eos.Z_g, dZ_dP=eos.dZ_dP_g, zs=zs)

    T, P, R, a_alpha_f, b1, b2, b3, b, = symbols('T, P, R, a_alpha_f, b1, b2, b3, b')
    Z_f, sum_f = symbols('Z_f, sum_f')
    a_alpha, sum_fun = a_alpha_f(T), sum_f(T)
    
    A = a_alpha*P/(R*R*T*T)
    B = b*P/(R*T)
    A, B
    
    Z = Z_f(P) # Change to f(P) when doing others
    
    needed = []
    for bi in [b1, b2, b3]:
        Bi = bi*P/R/T
        t1 = Bi/B*(Z-1) - log(Z - B)
        t3 = log(1. + B/Z)
        t2 = A/B*(Bi/B - 2./a_alpha*sum_fun)
        t4 = t1 + t2*t3
        needed.append(diff(t4, P))

    sympy_diffs = []
    for i in range(3):
        subs = {Derivative(Z_f(P), P): eos.dZ_dP_g, 
                R: R_num, 'b': eos.b,
                Z_f(P): eos.Z_g, 
                a_alpha: eos.a_alpha, 
                sum_fun: sum([zs[j]*eos.a_alpha_ijs[i][j] for j in eos.cmps])
                }
    
        subs2 = {P: eos.P, 
                 T: eos.T,
                 {0: b1, 1:b2, 2:b3}[i]: eos.bs[i]}
    
        working = needed[i].subs(subs)
        working = working.subs(subs2)
    
        sympy_diffs.append(float((N(working))))
    assert_allclose(sympy_diffs, diffs_implemented, rtol=1e-11)


def test_VDW_d_lnphis_dP():
    dP = 1e-1
    T = 280.0 # Need increase T a little
    P = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    eos = eos1 = VDWMIX(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    eos2 = VDWMIX(T=T, P=P + dP, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    numerical_diffs = (np.array(eos2.lnphis_g) - eos1.lnphis_g) / dP
    
    expected_diffs = [4.48757918014496e-8, -1.17254464726201e-8, -1.20732168353728e-7]
    analytical_diffs = eos.d_lnphis_dP(Z=eos.Z_g, dZ_dP=eos.dZ_dP_g, zs=zs)
    
    assert_allclose(analytical_diffs, expected_diffs, rtol=1e-11)
    assert_allclose(expected_diffs, numerical_diffs, rtol=1e-5)
    
@pytest.mark.sympy
def test_VDW_d_lnphis_dP_sympy():
    from sympy import Derivative, symbols, sqrt, diff, log, N
    from fluids.constants import R as R_num

    T_num = 280.0
    P_num = 76E5
    Tcs = [126.2, 190.6, 305.4]
    Pcs = [33.9E5, 46.0E5, 48.8E5]
    omegas = [0.04, 0.008, 0.098]
    kijs = [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]]
    zs = [0.3, 0.1, 0.6]
    
    
    eos = VDWMIX(T=T_num, P=P_num, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=kijs)
    diffs_implemented = eos.d_lnphis_dP(Z=eos.Z_g, dZ_dP=eos.dZ_dP_g, zs=zs)

    T, P, R, a_alpha_f, b1, b2, b3, b, = symbols('T, P, R, a_alpha_f, b1, b2, b3, b')
    a1, a2, a3 = symbols('a1, a2, a3')
    Z_f, sum_f = symbols('Z_f, sum_f')
    a_alpha, sum_fun = a_alpha_f(T), sum_f(T)
    
    Z = Z_f(P)
    
    needed = []
    for ai, bi in zip([a1, a2, a3], [b1, b2, b3]):
        V = Z*R*T/P
        t1 = log(Z*(1 - b/V))
        t2 = 2/(R*T*V)
        t3 = 1/(V - b)
        logphi = (bi*t3 - t1 - t2*sqrt(a_alpha*ai))
        
        needed.append(diff(logphi, P))
    
    sympy_diffs = []
    for i in range(3):
        subs = {Derivative(Z_f(P), P): eos.dZ_dP_g, 
                R: R_num, 'b': eos.b,
                Z_f(P): eos.Z_g, 
                a_alpha: eos.a_alpha, 
                sum_f(T): sum([zs[j]*eos.a_alpha_ijs[i][j] for j in eos.cmps])
                }
    
        subs2 = {P: eos.P, 
                 T: eos.T,
                 {0: b1, 1:b2, 2:b3}[i]: eos.bs[i],
                {0: a1, 1:a2, 2:a3}[i]: eos.ais[i]}
    
        working = needed[i].subs(subs)
        working = working.subs(subs2)
        
        sympy_diffs.append(float(N(working)))

    assert_allclose(sympy_diffs, diffs_implemented, rtol=1e-11)

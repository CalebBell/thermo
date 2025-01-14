'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020, 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
SOFTWARE.
'''

import pickle
from math import exp
from random import random

import numpy as np
import pytest
from chemicals.rachford_rice import *
from chemicals.utils import object_data
from fluids.constants import R, calorie
from fluids.numerics import assert_close, assert_close1d, assert_close2d, derivative, hessian, jacobian, linspace, normalize

from thermo import *
from thermo.nrtl import NRTL
from thermo.test_utils import check_np_output_activity
import json

def test_NRTL_gammas():
    # P05.01b VLE Behavior of Ethanol - Water Using NRTL
    gammas = NRTL_gammas([0.252, 0.748], [[0, -0.178], [1.963, 0]], [[0, 0.2974],[.2974, 0]])
    assert_close1d(gammas, [1.9363183763514304, 1.1537609663170014])

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
    assert_close1d(gammas, [1.9363183763514304, 1.1537609663170014])


    # Example by
    # https://github.com/iurisegtovich/PyTherm-applied-thermodynamics/blob/master/contents/main-lectures/GE1-NRTL-graphically.ipynb
    tau = [[0.0, 2.291653777670652, 0.5166949715946564], [4.308652420938829, 0.0, 1.6753963198550983], [0.5527434579849811, 0.15106032392136134, 0.0]]
    alpha = [[0.0, 0.4, 0.3], [0.4, 0.0, 0.3], [0.3, 0.3, 0.0]]
    xs = [.1, .3, .6]
    gammas = NRTL_gammas(xs, tau, alpha)
    assert_close1d(gammas, [2.7175098659360413, 2.1373006474468697, 1.085133765593844])

    # Test the values which give activity coefficients of 1:
    gammas = NRTL_gammas([0.252, 0.748], [[0, 0], [0, 0]], [[0, 0.5],[.9, 0]])
    assert_close1d(gammas, [1, 1])
    # alpha does not matter

    a = b = np.zeros((6, 6)).tolist()
    gammas = NRTL_gammas([0., 1, 0, 0, 0, 0], a, b)
    assert_close1d(gammas, [1,1,1,1,1,1])

    # Test vs chemsep parameters, same water ethanol T and P
    T = 343.15
    b12 = -57.9601*calorie
    b21 = 1241.7396*calorie
    tau12 = b12/(R*T)
    tau21 = b21/(R*T)

    gammas = NRTL_gammas(xs=[0.252, 0.748], taus=[[0, tau12], [tau21, 0]],
    alphas=[[0, 0.2937],[.2937, 0]])
    assert_close1d(gammas, [1.9853834856640085, 1.146380779201308])



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
    assert_close1d(gammas, gammas_expect)

def test_NRTL_gammas_10():
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
    assert_close1d(gammas, gammas_expect)

def NRTL_gammas_44():
    N = 44
    taus = [[random() for i in range(N)] for j in range(N)]
    alphas = [[random() for i in range(N)] for j in range(N)]
    xs = normalize([random() for i in range(N)])
    gammas = NRTL_gammas(xs=xs, taus=taus, alphas=alphas)


def NRTL_gammas_200():
    # ten component
    # Takes 40 ms - not a great idea
    N = 200
    taus = [[random() for i in range(N)] for j in range(N)]
    alphas = [[random() for i in range(N)] for j in range(N)]
    xs = normalize([random() for i in range(N)])
    gammas = NRTL_gammas(xs=xs, taus=taus, alphas=alphas)

def make_alphas(N):
    cmps = range(N)
    data = []
    for i in cmps:
        row = []
        for j in cmps:
            if i == j:
                row.append([0.0, 0.0])
            else:
                row.append([round(random()*0.3, 3), round(random()*1e-5, 8)])
        data.append(row)
    return data

def make_taus(N):
    cmps = range(N)
    data = []
    base = [3e-5, 600.0, 1e-4, 7e-5, 5e-3, 9e-7]

    for i in cmps:
        row = []
        for j in cmps:
            if i == j:
                row.append([0.0]*6)
            else:
                row.append([float('%.3g'%(random()*n)) for n in base])
        data.append(row)
    return data


def test_madeup_NRTL():
    N = 4
    alphas = make_alphas(N)
    taus = make_taus(N)
    xs = normalize([random() for i in range(N)])
    T = 350.0
    GE = NRTL(T=T, xs=xs, tau_coeffs=taus, alpha_coeffs=alphas)
    # Run the various checks for storing/loading
    assert eval(str(GE)).GE() == GE.GE()

    GE_pickle = pickle.loads(pickle.dumps(GE))
    assert GE_pickle == GE
    assert GE_pickle.model_hash() == GE.model_hash()


    assert NRTL.from_json(json.loads(json.dumps(GE.as_json()))) == GE

    # Check after we calculate things
    for attr in GE._point_properties:
        try:
            getattr(GE, attr)()
        except:
            pass
    # Check we get the attributes copied
    GE_copy = NRTL.from_json(json.loads(json.dumps(GE.as_json(option=0))))
    assert_close(GE_copy.dGE_dT(), GE.dGE_dT())
    assert GE_copy == GE

    # Check we can also copy without calculated values
    GE_copy = NRTL.from_json(json.loads(json.dumps(GE.as_json(option=1))))
    assert GE_copy == GE


def test_water_ethanol_methanol_madeup():
    alphas = [[[0.0, 2e-05], [0.2937, 7e-05], [0.2999, 0.0001]],
     [[0.2937, 1e-05], [0.0, 4e-05], [0.3009, 8e-05]],
     [[0.2999, 1e-05], [0.3009, 3e-05], [0.0, 5e-05]]]

    taus = [[[6e-05, 0.0, 7e-05, 7e-05, 0.00788, 3.6e-07],
      [3e-05, 624.868, 9e-05, 7e-05, 0.00472, 8.5e-07],
      [3e-05, 398.953, 4e-05, 1e-05, 0.00279, 5.6e-07]],
     [[1e-05, -29.167, 8e-05, 9e-05, 0.00256, 1e-07],
      [2e-05, 0.0, 7e-05, 6e-05, 0.00587, 4.2e-07],
      [0.0, -35.482, 8e-05, 4e-05, 0.00889, 8.2e-07]],
     [[9e-05, -95.132, 6e-05, 1e-05, 0.00905, 5.2e-07],
      [9e-05, 33.862, 2e-05, 6e-05, 0.00517, 1.4e-07],
      [0.0001, 0.0, 6e-05, 2e-05, 0.00095, 7.4e-07]]]

    N = 3
    T = 273.15+70
    dT = T*1e-8
    xs = [.2, .3, .5]
    GE = NRTL(T=T, xs=xs, tau_coeffs=taus, alpha_coeffs=alphas)
    assert eval(str(GE)).GE() == GE.GE()

    assert object_data(NRTL.from_json(GE.as_json())) == object_data(GE)

    GEnp = NRTL(T=T, xs=np.array(xs), tau_coeffs=np.array(taus), alpha_coeffs=np.array(alphas))
    assert_close(GEnp.GE(), GE.GE(), rtol=1e-12)

    # gammas
    assert_close1d(GE.gammas(), [1.7795902383749216, 1.1495597830749005, 1.0736702352016942])
    assert_close1d(GEnp.gammas(), GE.gammas(), rtol=1e-12)
    assert_close1d(GEnp.gammas_numerical(), GE.gammas(), rtol=1e-5)
    assert_close1d(GE.gammas_numerical(), GE.gammas(), rtol=1e-5)

    ### Tau and derivatives
    taus_expected = [[0.06687993075720595, 1.9456413587531054, 1.2322559725492486],
     [-0.04186204696272491, 0.07047352903742096, 0.007348860249786843],
     [-0.21212866478360642, 0.13596095401379812, 0.0944497207779701]]
    assert_close2d(taus_expected, GE.taus(), rtol=1e-14)

    # Tau temperature derivative
    dtaus_dT_numerical = (np.array(GE.to_T_xs(T+dT, xs).taus()) - GE.taus())/dT
    dtaus_dT_analytical = GE.dtaus_dT()
    dtaus_dT_expected = [[0.000317271602387579, -0.004653030923421638, -0.0029936361350323625],
     [0.000406561723402744, 0.00034844970187634483, 0.0009043271077256468],
     [0.0011749522864265571, -0.00013143064874333648, 0.0005280368036263511]]
    assert_close2d(dtaus_dT_analytical, dtaus_dT_expected, rtol=1e-12)
    assert_close2d(dtaus_dT_numerical, dtaus_dT_analytical, rtol=4e-7)

    # tau second derivative
    d2taus_dT2_analytical = GE.d2taus_dT2()
    d2taus_dT2_expected = [[7.194089397742681e-07, 3.2628265646047626e-05, 2.0866597625703075e-05],
     [-1.2443543228779117e-06, 8.394080699905074e-07, -1.169244972344292e-07],
     [-3.669244570181493e-06, 1.955896362917401e-06, 1.4794908652710361e-06]]
    assert_close2d(d2taus_dT2_analytical, d2taus_dT2_expected, rtol=1e-12)
    d2taus_dT2_numerical = (np.array(GE.to_T_xs(T+dT, xs).dtaus_dT()) - GE.dtaus_dT())/dT
    assert_close2d(d2taus_dT2_analytical, d2taus_dT2_numerical, rtol=4e-7)

    # tau third derivative
    d3taus_dT3_analytical = GE.d3taus_dT3()
    d3taus_dT3_expected = [[3.425034691577827e-12, -2.703935984244539e-07, -1.7263626338812435e-07],
     [1.2625331389834536e-08, 3.4351735085462344e-12, 1.535797829031479e-08],
     [4.116922701015044e-08, -1.4652079774338131e-08, 2.9650219270685846e-12]]

    # Not sure why precision of numerical test is so low, but confirmed with sympy.
    assert_close2d(d3taus_dT3_analytical, d3taus_dT3_expected, rtol=1e-12)
    d3taus_dT3_numerical = (np.array(GE.to_T_xs(T+dT, xs).d2taus_dT2()) - GE.d2taus_dT2())/dT
    assert_close2d(d3taus_dT3_numerical, d3taus_dT3_analytical, rtol=2e-5)

    # alphas
    alphas_expect = [[0.006863, 0.3177205, 0.334215],
     [0.2971315, 0.013726, 0.328352],
     [0.3033315, 0.3111945, 0.0171575]]
    assert_close2d(alphas_expect,  GE.alphas(), rtol=1e-12)

    # dalphas_dT
    dalphas_dT_expect = [[2e-05, 7e-05, 0.0001], [1e-05, 4e-05, 8e-05], [1e-05, 3e-05, 5e-05]]
    dalphas_dT_analytical = GE.dalphas_dT()
    assert_close2d(dalphas_dT_expect, dalphas_dT_analytical, rtol=1e-12)
    dalphas_dT_numerical = (np.array(GE.to_T_xs(T+1e-4, xs).alphas()) - GE.alphas())/1e-4
    assert_close2d(dalphas_dT_expect, dalphas_dT_numerical)


    # Gs
    Gs_expect = [[0.9995411083582052, 0.5389296989069797, 0.6624312965198783],
     [1.0125162130984628, 0.999033148043276, 0.9975898960147673],
     [1.0664605905163351, 0.9585722883795924, 0.9983807912510595]]
    assert_close2d(Gs_expect, GE.Gs())

    # dGs_dT
    dGs_dT_expect = [[-3.513420602780159e-06, 0.0007233344205287423, 0.000581146010596799],
     [-0.00012189042196807427, -7.594411991210204e-06, -0.00029680845584651057],
     [-0.000377824327942326, 3.529622902294454e-05, -1.3759961112813966e-05]]
    dGs_dT_analytical = GE.dGs_dT()
    assert_close2d(dGs_dT_expect, dGs_dT_analytical, rtol=1e-12)
    dGs_dT_numerical = (np.array(GE.to_T_xs(T+ 2.5e-5, xs).Gs()) - GE.Gs())/2.5e-5
    assert_close2d(dGs_dT_numerical, dGs_dT_expect, rtol=3e-7) # Closest I could get

    # d2Gs_dT2
    d2Gs_dT2_expect = [[-1.7607728438831442e-08, -4.264997204215131e-06, -3.7132987505208185e-06],
     [3.808051820202208e-07, -3.9301868673119065e-08, -1.773565965539868e-08],
     [1.2957622532360591e-06, -5.745898135094948e-07, -7.78717985147786e-08]]
    d2Gs_dT2_analytical = GE.d2Gs_dT2()
    assert_close2d(d2Gs_dT2_expect, d2Gs_dT2_analytical, rtol=1e-12)
    d2Gs_dT2_numerical = (np.array(GE.to_T_xs(T+4e-6, xs).dGs_dT()) - GE.dGs_dT())/4e-6
    assert_close2d(d2Gs_dT2_numerical, d2Gs_dT2_analytical, rtol=2e-6)

    # d3Gs_dT3
    d3Gs_dT3_analytical = GE.d3Gs_dT3()
    d3Gs_dT3_expect = [[-4.298246167557067e-11, 2.282743128709054e-08, 2.3406412447900822e-08],
     [-3.894534156411875e-09, -9.978151596051988e-11, -4.934296656436311e-09],
     [-1.448282405010784e-08, 4.1384450135276364e-09, -2.1839009944794027e-10]]
    d3Gs_dT3_numerical = (np.array(GE.to_T_xs(T+dT, xs).d2Gs_dT2()) - GE.d2Gs_dT2())/dT
    assert_close2d(d3Gs_dT3_analytical, d3Gs_dT3_numerical, rtol=1e-7)
    assert_close2d(d3Gs_dT3_expect, d3Gs_dT3_analytical, rtol=1e-12)

    # dGE_dT
    dGE_dT_expect = 2.258093406765717
    dGE_dT_analytical = GE.dGE_dT()
    assert_close(dGE_dT_expect, dGE_dT_analytical, rtol=1e-12)

    def to_diff(T):
        return GE.to_T_xs(T, xs).GE()
    dGE_dT_numerical = derivative(to_diff, T, dx=1, order=17)
    assert_close(dGE_dT_analytical, dGE_dT_numerical, rtol=1e-12)

    # d2GE_dT2
    def to_diff(T):
        return GE.to_T_xs(T, xs).dGE_dT()

    d2GE_dT2_numerical = derivative(to_diff, T, dx=5, order=17)
    d2GE_dT2_expected = 0.007412479461681191
    d2GE_dT2_analytical = GE.d2GE_dT2()
    assert_close(d2GE_dT2_expected, d2GE_dT2_analytical, rtol=1e-12)
    assert_close(d2GE_dT2_numerical, d2GE_dT2_analytical, rtol=1e-12)

    # dGE_dxs
    def to_jac(xs):
        return GE.to_T_xs(T, xs).GE()

    dGE_dxs_analytical = GE.dGE_dxs()
    dGE_dxs_expected = [1644.4832445701338, 397.6635234141318, 202.8071951151063]
    assert_close1d(dGE_dxs_analytical, dGE_dxs_expected, rtol=1e-12)
    dGE_dxs_numerical = jacobian(to_jac, xs, perturbation=1e-8)
    assert_close1d(dGE_dxs_numerical, dGE_dxs_expected, rtol=1e-7)



    # d2GE_dxixjs - more decimals in numdifftools
    def to_hess(xs):
        return GE.to_T_xs(T, xs).GE()

    d2GE_dxixjs_expect = [[-1690.7876619180952, 1208.6126803238276, -48.852543427058286],
     [1208.6126803238276, -468.99032836115686, -202.0508751128366],
     [-48.852543427058606, -202.0508751128365, 140.77154243852513]]
    d2GE_dxixjs_analytical = GE.d2GE_dxixjs()
    assert_close2d(d2GE_dxixjs_expect, d2GE_dxixjs_analytical, rtol=1e-12)
    d2GE_dxixjs_numerical = hessian(to_hess, xs, perturbation=5e-5)
    assert_close2d(d2GE_dxixjs_numerical, d2GE_dxixjs_analytical, rtol=3e-4)


    # d2GE_dTdxs - matched really well
    d2GE_dTdxs_expect = [3.053720836458414, 1.8759446742883084, 2.1691316743750826]
    d2GE_dTdxs_analytical = GE.d2GE_dTdxs()
    assert_close1d(d2GE_dTdxs_expect, d2GE_dTdxs_analytical, rtol=1e-12)

    def to_jac(xs):
        return GE.to_T_xs(T, xs).dGE_dT()

    d2GE_dTdxs_numerical = jacobian(to_jac, xs, perturbation=3e-8)
    assert_close1d(d2GE_dTdxs_analytical, d2GE_dTdxs_numerical, rtol=1e-7)

    # Direct call for gammas
    gammas_args = GE.gammas_args()
    gammas = GE.gammas_from_args(GE.xs, *gammas_args)
    assert_close1d(gammas, GE.gammas(), rtol=1e-13)

    # gammas at another T
    T_another = 401.234
    gammas_args_at_T = GE.gammas_args(T=T_another)
    gammas_at_T = GE.gammas_from_args(GE.xs, *gammas_args_at_T)
    assert_close1d(gammas_at_T, GE.to_T_xs(T=T_another, xs=GE.xs).gammas(), rtol=1e-13)

def test_NRTL_numpy_output():
    alphas = [[[0.0, 2e-05], [0.2937, 7e-05], [0.2999, 0.0001]],
     [[0.2937, 1e-05], [0.0, 4e-05], [0.3009, 8e-05]],
     [[0.2999, 1e-05], [0.3009, 3e-05], [0.0, 5e-05]]]

    taus = [[[6e-05, 0.0, 7e-05, 7e-05, 0.00788, 3.6e-07],
      [3e-05, 624.868, 9e-05, 7e-05, 0.00472, 8.5e-07],
      [3e-05, 398.953, 4e-05, 1e-05, 0.00279, 5.6e-07]],
     [[1e-05, -29.167, 8e-05, 9e-05, 0.00256, 1e-07],
      [2e-05, 0.0, 7e-05, 6e-05, 0.00587, 4.2e-07],
      [0.0, -35.482, 8e-05, 4e-05, 0.00889, 8.2e-07]],
     [[9e-05, -95.132, 6e-05, 1e-05, 0.00905, 5.2e-07],
      [9e-05, 33.862, 2e-05, 6e-05, 0.00517, 1.4e-07],
      [0.0001, 0.0, 6e-05, 2e-05, 0.00095, 7.4e-07]]]

    N = 3
    T = 273.15+70
    dT = T*1e-8
    xs = [.2, .3, .5]
    model = NRTL(T=T, xs=xs, tau_coeffs=taus, alpha_coeffs=alphas)
    modelnp = NRTL(T=T, xs=np.array(xs), tau_coeffs=np.array(taus), alpha_coeffs=np.array(alphas))
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)
    json_string = modelnp.as_json()
    new = NRTL.from_json(json_string)
    assert new == modelnp

    assert model.model_hash() == modelnp.model_hash()
    assert new.model_hash() == modelnp.model_hash()
    assert new.model_hash() == modelnp2.model_hash()


    # Pickle checks
    modelnp_pickle = pickle.loads(pickle.dumps(modelnp))
    assert modelnp_pickle == modelnp
    model_pickle = pickle.loads(pickle.dumps(model))
    assert model_pickle == model

def test_NRTL_numpy_output_correct_array_internal_ownership():
    '''Without the array calls and the order bit, performance was probably bad
    and pypy gave a different object hash.
    '''
    alphas = [[[0.0, 2e-05], [0.2937, 7e-05], [0.2999, 0.0001]],
     [[0.2937, 1e-05], [0.0, 4e-05], [0.3009, 8e-05]],
     [[0.2999, 1e-05], [0.3009, 3e-05], [0.0, 5e-05]]]

    taus = [[[6e-05, 0.0, 7e-05, 7e-05, 0.00788, 3.6e-07],
      [3e-05, 624.868, 9e-05, 7e-05, 0.00472, 8.5e-07],
      [3e-05, 398.953, 4e-05, 1e-05, 0.00279, 5.6e-07]],
     [[1e-05, -29.167, 8e-05, 9e-05, 0.00256, 1e-07],
      [2e-05, 0.0, 7e-05, 6e-05, 0.00587, 4.2e-07],
      [0.0, -35.482, 8e-05, 4e-05, 0.00889, 8.2e-07]],
     [[9e-05, -95.132, 6e-05, 1e-05, 0.00905, 5.2e-07],
      [9e-05, 33.862, 2e-05, 6e-05, 0.00517, 1.4e-07],
      [0.0001, 0.0, 6e-05, 2e-05, 0.00095, 7.4e-07]]]

    N = 3
    T = 273.15+70
    dT = T*1e-8
    xs = [.2, .3, .5]
    modelnp = NRTL(T=T, xs=np.array(xs), tau_coeffs=np.array(taus), alpha_coeffs=np.array(alphas))

    for name in ('tau_as', 'tau_bs', 'tau_es', 'tau_fs', 'tau_gs', 'tau_hs', 'alpha_cs', 'alpha_ds'):
        obj = getattr(modelnp, name)
        assert obj.flags.c_contiguous
        assert obj.flags.owndata

def test_NRTL_missing_inputs():
    alphas = [[[0.0, 2e-05], [0.2937, 7e-05], [0.2999, 0.0001]],
     [[0.2937, 1e-05], [0.0, 4e-05], [0.3009, 8e-05]],
     [[0.2999, 1e-05], [0.3009, 3e-05], [0.0, 5e-05]]]

    taus = [[[6e-05, 0.0, 7e-05, 7e-05, 0.00788, 3.6e-07],
      [3e-05, 624.868, 9e-05, 7e-05, 0.00472, 8.5e-07],
      [3e-05, 398.953, 4e-05, 1e-05, 0.00279, 5.6e-07]],
     [[1e-05, -29.167, 8e-05, 9e-05, 0.00256, 1e-07],
      [2e-05, 0.0, 7e-05, 6e-05, 0.00587, 4.2e-07],
      [0.0, -35.482, 8e-05, 4e-05, 0.00889, 8.2e-07]],
     [[9e-05, -95.132, 6e-05, 1e-05, 0.00905, 5.2e-07],
      [9e-05, 33.862, 2e-05, 6e-05, 0.00517, 1.4e-07],
      [0.0001, 0.0, 6e-05, 2e-05, 0.00095, 7.4e-07]]]

    N = 3
    T = 273.15+70
    dT = T*1e-8
    xs = [.2, .3, .5]
    GE = NRTL(T=T, xs=xs, tau_coeffs=taus)
    assert_close1d(GE.gammas(), [1, 1, 1], rtol=1e-13)

    GE = NRTL(T=T, xs=xs, tau_coeffs=taus)
    assert_close1d(GE.gammas(), [1, 1, 1], rtol=1e-13)

    GE = NRTL(T=T, xs=xs, alpha_coeffs=alphas)
    assert_close1d(GE.gammas(), [1, 1, 1], rtol=1e-13)

def test_NRTL_chemsep():
    from thermo.interaction_parameters import IPDB
    tausB = IPDB.get_ip_asymmetric_matrix('ChemSep NRTL', ['64-17-5', '7732-18-5'], 'bij')
    alphaC = IPDB.get_ip_asymmetric_matrix('ChemSep NRTL', ['64-17-5', '7732-18-5'], 'alphaij')
    N = 2
    T = 343.15
    xs = [0.252, 0.748]
    tausA = tausE = tausF = tausG = tausH = alphaD = [[0.0]*N for i in range(N)]
    ABEFGHCD = (tausA, tausB, tausE, tausF, tausG, tausH, alphaC, alphaD)
    GE = NRTL(T=T, xs=xs, ABEFGHCD=ABEFGHCD)
    gammas = GE.gammas()
    assert_close1d(gammas, [1.985383485664009, 1.146380779201308], rtol=1e-7)

    # Table of values right from ChemSep
    gammas_ethanol = [5.66232, 4.283, 3.3749, 2.75365, 2.31475, 1.99622, 1.75984, 1.58121, 1.44425, 1.33807, 1.25512, 1.19003, 1.13894, 1.09896, 1.06796, 1.0443, 1.02671, 1.0142, 1.00598, 1.00142, 1]
    gammas_water = [1, 1.00705, 1.02657, 1.05673, 1.09626, 1.14429, 1.20021, 1.26357, 1.33405, 1.41139, 1.49541, 1.58593, 1.68281, 1.78591, 1.89509, 2.0102, 2.1311, 2.25761, 2.38956, 2.52675, 2.66895]
    pts = 21
    xs_ethanol = linspace(0, 1, 21)
    for i in range(pts):
        GE2 = GE.to_T_xs(T=T, xs=[xs_ethanol[i], 1-xs_ethanol[i]])
        gammas = GE2.gammas()
        assert_close(gammas[0], gammas_ethanol[i], rtol=1e-5)
        assert_close(gammas[1], gammas_water[i], rtol=1e-5)

def test_NRTL_partial_inputs():
    from fluids.constants import R, calorie
    N = 2
    T = 70.0 + 273.15
    xs = [0.252, 0.748]
    tau_bs = [[0, -121.2691/R*calorie], [1337.8574/R*calorie, 0]]
    alpha_cs =  [[0, 0.2974],[.2974, 0]]
    GE = NRTL(T=T, xs=xs, tau_bs=tau_bs, alpha_cs=alpha_cs)
    gammas_expect = [1.936051651447544, 1.1536630452052914]
    assert_close1d(GE.gammas(), gammas_expect, rtol=1e-13)

    # String tests
    s = str(GE)

    with pytest.raises(ValueError):
        NRTL(T=T, xs=xs, tau_bs=tau_bs, alpha_cs=alpha_cs, alpha_ds=[[0],[.2974, 0]])

    with pytest.raises(ValueError):
        NRTL(T=T, xs=xs, tau_bs=tau_bs, alpha_cs=alpha_cs, tau_es=[[0],[.2974, 0]])

    with pytest.raises(ValueError):
        NRTL(T=T, xs=xs, tau_bs=tau_bs, alpha_cs=alpha_cs, tau_es=[[0, 3]])

    with pytest.raises(ValueError):
        NRTL(T=T, xs=xs, tau_bs=tau_bs, alpha_cs=alpha_cs, tau_es=[None, 3])

    with pytest.raises(ValueError):
        NRTL(T=T, xs=xs, tau_bs=tau_bs, alpha_cs=alpha_cs, ABEFGHCD=(tau_bs,))

def test_NRTL_ideal_ones():
    T = 70.0 + 273.15
    xs = [0.252, 0.748]
    GE = NRTL(T=T, xs=xs)
    assert_close1d(GE.gammas(), [1.0, 1.0], rtol=1e-13)

def test_NRTL_asymmetric_alpha():
    GE = NRTL(T=343.15, xs=[0.252, 0.748], tau_bs=[[0, -61.02497992981518], [673.2359767158717, 0]],
          alpha_cs=[[0, 0.2974], [0.35, 0]])
    assert_close1d(GE.gammas(), [1.8254928709184535, 1.1578302838386516], rtol=1e-12)

def test_NRTL_gammas_binaries():
    gammas_calc = NRTL_gammas_binaries([0.252, 0.748, .7, .3, 1e-10, 1-1e-10], -0.1778376218266507, 1.9619291176333142, .2974, .35)
    gammas_expect = [1.825492870918453, 1.1578302838386514, 1.0471700487668791, 1.782792759735783, 5.896934778222999, 1.0]
    assert_close1d(gammas_calc, gammas_expect, rtol=1e-13)

    # Test providing the output array
    gammas_calc_dup = NRTL_gammas_binaries([0.1, 0.9, .2, .8, .5, .5], -0.1778376218266507, 1.9619291176333142, .2974, .35)
    gammas_calc_same_array = NRTL_gammas_binaries([0.1, 0.9, .2, .8, .5, .5], -0.1778376218266507, 1.9619291176333142, .2974, .35, gammas_calc)
    assert_close1d(gammas_calc_dup, gammas_calc_same_array, rtol=1e-13)
    assert gammas_calc is gammas_calc_same_array

def test_NRTL_gammas_binaries_vs_object():
    GE = NRTL(T=343.15, xs=[.2, .8], tau_bs=[[0, -61.02497992981518], [673.2359767158717, 0]],
          alpha_cs=[[0, 0.2974], [0.35, 0]])
    gammas_calc = NRTL_gammas_binaries(GE.xs, GE.taus()[0][1], GE.taus()[1][0], GE.alphas()[0][1], GE.alphas()[1][0])
    assert_close1d(gammas_calc, GE.gammas(), rtol=1e-13)


def test_NRTL_gammas_binaries_jac():
    expect = [[1.1789916307339643, -0.26111310616433153, -3.3538119188199227, -6.741171640572135], [0.028442736322554146, 0.12389510256475463, -0.4669633093241108, -0.24619435634886436], [0.6880803492773255, -0.1604580514041915, -1.2421051941611734, -3.0460013493889275], [0.07924102114006676, 0.14519919300123013, -0.9799779542489997, -0.9023280256735463]]
    jac_calc = NRTL_gammas_binaries_jac([.3, .7, .4, .6], 2, 3, .2, .4)
    assert_close2d(expect, jac_calc, rtol=1e-12)

    
    # Test with provided calc array
    calc_provided = np.zeros((4, 4))
    jac_calc_same_array = NRTL_gammas_binaries_jac([.3, .7, .4, .6], 2, 3, .2, .4, calc_provided)
    assert_close2d(expect, jac_calc_same_array, rtol=1e-12)
    assert calc_provided is jac_calc_same_array
    
    # Test with different input, reusing the calc array
    new_expect = NRTL_gammas_binaries_jac([.2, .8, .6, .4], 1, 4, .3, .5)
    jac_calc_reused = NRTL_gammas_binaries_jac([.2, .8, .6, .4], 1, 4, .3, .5, calc_provided)
    assert_close2d(new_expect, jac_calc_reused, rtol=1e-12)
    assert calc_provided is jac_calc_reused
    # The optimized code was generated as follows with sympy
    """
    from sympy import *
    tau01, tau10, alpha01, alpha10, x0, x1 = symbols('tau01, tau10, alpha01, alpha10, x0, x1')

    G01 = exp(-alpha01*tau01)
    G10 = exp(-alpha10*tau10)

    gamma0 = exp(x1**2*(tau10*(G10/(x0+x1*G10))**2 + G01*tau01/(x1+x0*G01)**2))
    gamma1 = exp(x0**2*(tau01*(G01/(x1+x0*G01))**2 + G10*tau10/(x0+x1*G10)**2))

    outs, mains  = cse([gamma0, gamma1], optimizations='basic')

    gammas = [simplify(gamma0), simplify(gamma1)]
    to_diff_on = [tau01, tau10, alpha01, alpha10]

    to_cse = [diff(g, v) for v in to_diff_on for g in gammas]
    outs, mains  = cse(to_cse, optimizations='basic')
    for k, v in outs:
        print("%s = %s" %(k, v))
    print(mains)
    """


def test_NRTL_regression_basics():
    # ethanol-water
    GE = UNIFAC.from_subgroups(T=298.15, xs=[.9, .1], chemgroups=[{1: 1, 2: 1, 14: 1}, {16: 1}])
    pts = 10
    xs_points = [[xi, 1-xi] for xi in linspace(0.01, 0.99, pts)]
    many_gammas_expect = [GE.to_T_xs(T=GE.T, xs=xs_points[i]).gammas() for i in range(pts)]

    res, stats = NRTL.regress_binary_parameters(gammas=many_gammas_expect, xs=xs_points, use_numba=False,
                                                symmetric_alphas=True, multiple_tries=False)

    assert_close(res['tau12'], 0.38703486403040827)
    assert_close(res['tau21'], 1.7353246253228976)
    assert_close(res['alpha12'], 0.64618267549806)
    assert stats['MAE'] < 0.01


    # Regression without symmetry
    res, stats = NRTL.regress_binary_parameters(gammas=many_gammas_expect, xs=xs_points, use_numba=False,
                                                symmetric_alphas=False, multiple_tries=False)

    assert_close(res['tau12'], 0.229234995043471)
    assert_close(res['tau21'], 1.974654718742523)
    assert_close(res['alpha12'], 6.207804509017788)
    assert_close(res['alpha21'], 0.4898957291346906)
    assert stats['MAE'] < 0.001

    # regression with multiple alpha values forced
    forced_alphas = [0.1, 0.2, 0.3, 0.7, 0.9] 
    expected_taus = [
        (-2.0482635713094823, 4.507094631730519),
        (-0.7915760197977199, 2.9274685163699594),
        (-0.3043327536515584, 2.3396166105505105),
        (0.45660391657378036, 1.7122942946463962),
        (0.6760802269114482, 1.7130983381263352)
    ]

    for i, forced_alpha in enumerate(forced_alphas):
        res, stats = NRTL.regress_binary_parameters(gammas=many_gammas_expect, xs=xs_points, use_numba=False,
                                                    symmetric_alphas=True, multiple_tries=False, force_alpha=forced_alpha)

        assert 'alpha21' not in res
        assert len(res) == 2  # Should only have tau12 and tau21
        assert_close(res['tau12'], expected_taus[i][0], rtol=1e-3)
        assert_close(res['tau21'], expected_taus[i][1], rtol=1e-3)
        assert stats['MAE'] < 0.05

def test_NRTL_one_component():
    GE = NRTL(T=350.0, xs=[1.0], ABEFGHCD=([[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]))
    for s in GE._point_properties:
        if hasattr(GE, s):
            res = getattr(GE, s)()

def test_NRTL_can_return_zero_may_need_lngamma_call():
    obj = NRTL(T=8, xs=[0.999999, 2.5e-07, 2.5e-07, 2.5e-07, 2.5e-07], tau_as=[[0, -5.1549, 0, 0, 0], [5.8547, 0, -0.40926, 0, 0], [0, -0.39036, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], tau_bs=[[0, 2270.62, 284.966, 0, 0], [229.497, 0, 1479.46, 0, 0], [-216.256, 447.003, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], alpha_cs=[[0, 0.2, 0.3, 0, 0], [0.2, 0, 0.46, 0, 0], [0.3, 0.46, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    assert obj.gammas()[2] == 0



def test_NRTL_missing_interaction_parameters():
    # Test Case 1: Everything Present
    alphas = [[0, 0.3, 0.3], [0.3, 0, 0.3], [0.3, 0.3, 0]]
    taus = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    GE = NRTL(T=300, xs=[0.2, 0.5, 0.3], tau_bs=taus, alpha_cs=alphas)
    assert GE.missing_interaction_parameters() == []

    # Test Case 2: One Missing Parameter (Symmetric)
    alphas = [[0, 0.3, 0.3], [0.3, 0, 0.3], [0.3, 0.3, 0]]
    taus = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    GE = NRTL(T=300, xs=[0.2, 0.5, 0.3], tau_bs=taus, alpha_cs=alphas)
    assert GE.missing_interaction_parameters() == [(1, 2), (2, 1)]

    # Test Case 3: One Missing Parameter (Asymmetric)
    alphas = [[0, 0.3, 0.3], [0.3, 0, 0.3], [0.3, 0.3, 0]]
    taus = [[0, 1, 1], [1, 0, 0], [1, 1, 0]]
    GE = NRTL(T=300, xs=[0.2, 0.5, 0.3], tau_bs=taus, alpha_cs=alphas)
    assert GE.missing_interaction_parameters() == [(1, 2)]

    # Test Case 4: 30% Missing Parameters (5 Compounds, Unique Alpha and Tau)
    alphas = [
        [0.0, 0.1, 0.2, 0.3, 0.4],
        [0.1, 0.0, 0.5, 0.6, 0.7],
        [0.2, 0.5, 0.0, 0.8, 0.9],
        [0.3, 0.6, 0.8, 0.0, 1.0],
        [0.4, 0.7, 0.9, 1.0, 0.0]
    ]
    taus = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [5.0, 0.0, 6.0, 7.0, 0.0],  # Missing (1,4)
        [8.0, 9.0, 0.0, 0.0, 10.0],  # Missing (2,3)
        [11.0, 12.0, 13.0, 0.0, 14.0],
        [0.0, 15.0, 16.0, 0.0, 0.0]  # Missing (0,4) and (3,4)
    ]
    GE = NRTL(T=300, xs=[0.2] * 5, tau_bs=taus, alpha_cs=alphas)
    expected_missing = [(1, 4), (2, 3), (4, 0), (4, 3)]
    assert sorted(GE.missing_interaction_parameters()) == sorted(expected_missing)
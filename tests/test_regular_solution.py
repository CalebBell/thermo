# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import log
from fluids.constants import R

from thermo.activity import GibbsExcess
from thermo import *
import numpy as np
from fluids.numerics import jacobian, hessian, assert_close, assert_close1d, assert_close2d, assert_close3d
from random import random
from chemicals import normalize
from thermo.test_utils import check_np_output_activity
import pickle

def test_no_interactions():
    GE = RegularSolution(T=325.0, xs=[.25, .75], Vs=[7.421e-05, 8.068e-05], SPs=[19570.2, 18864.7])
    assert_close2d(GE.lambda_coeffs, [[0, 0], [0, 0]], atol=0, rtol=0)

    GE = RegularSolution(T=325.0, xs=np.array([.25, .75]), Vs=np.array([7.421e-05, 8.068e-05]), SPs=np.array([19570.2, 18864.7]))
    assert_close2d(GE.lambda_coeffs, [[0, 0], [0, 0]], atol=0, rtol=0)
    assert type(GE.lambda_coeffs) is np.ndarray

def test_4_components():
#    m = Mixture(['acetone', 'chloroform', 'methanol', 'water'], zs=xs, T=300)
    xs = [.4, .3, .2, .1]
    SPs = [19570.2, 18864.7, 29261.4, 47863.5]
    Vs = [7.421e-05, 8.068e-05, 4.083e-05, 1.808e-05]
    N = 4
    T = 300.0
    # Made up asymmetric parameters
    lambda_coeffs = [[0.0, 0.01811, 0.01736, 0.02111],
     [0.00662, 0.0, 0.00774, 0.01966],
     [0.01601, 0.01022, 0.0, 0.00698],
     [0.0152, 0.00544, 0.02579, 0.0]]

    GE = RegularSolution(T, xs, Vs, SPs, lambda_coeffs)
    assert eval(str(GE)).GE() == GE.GE()

    GE2 = RegularSolution.from_json(GE.as_json())
    assert GE2.__dict__ == GE.__dict__

    dT = 1e-7*T
    gammas_expect = [1.1928784349228994, 1.3043087978251762, 3.2795596493820955, 197.92137114651274]
    assert_close1d(GE.gammas(), gammas_expect, rtol=1e-12)
    assert_close1d(GibbsExcess.gammas(GE), gammas_expect)

    # Gammas
    assert_close(GE.GE(), 2286.257263714889, rtol=1e-12)
    gammas = GE.gammas()
    GE_from_gammas = R*T*sum(xi*log(gamma) for xi, gamma in zip(xs, gammas))
    assert_close(GE_from_gammas, GE.GE(), rtol=1e-12)

    # Gamma direct call
    assert_close1d(GE.gammas(), 
                   regular_solution_gammas(T=T, xs=xs, Vs=Vs, SPs=SPs, lambda_coeffs=lambda_coeffs, N=N), rtol=1e-12)


    # dGE dT
    dGE_dT_numerical = ((np.array(GE.to_T_xs(T+dT, xs).GE()) - np.array(GE.GE()))/dT)
    dGE_dT_analytical = GE.dGE_dT()
    assert_close(dGE_dT_analytical, 0, rtol=1e-12, atol=1e-9)
    assert_close(dGE_dT_numerical, dGE_dT_analytical)

    # d2GE dT2
    d2GE_dT2_numerical = ((np.array(GE.to_T_xs(T+dT, xs).dGE_dT()) - np.array(GE.dGE_dT()))/dT)
    d2GE_dT2_analytical = GE.d2GE_dT2()
    assert_close(d2GE_dT2_analytical, 0, rtol=1e-12, atol=1e-9)
    assert_close(d2GE_dT2_analytical, d2GE_dT2_numerical, rtol=1e-8)

    # d3GE dT3
    d3GE_dT3_numerical = ((np.array(GE.to_T_xs(T+dT, xs).d2GE_dT2()) - np.array(GE.d2GE_dT2()))/dT)
    d3GE_dT3_analytical = GE.d3GE_dT3()
    assert_close(d3GE_dT3_analytical, 0, rtol=1e-12, atol=1e-9)
    assert_close(d3GE_dT3_numerical, d3GE_dT3_analytical, rtol=1e-7)

    # d2GE_dTdxs
    def dGE_dT_diff(xs):
        return GE.to_T_xs(T, xs).dGE_dT()

    d2GE_dTdxs_numerical = jacobian(dGE_dT_diff, xs, perturbation=1e-7)
    d2GE_dTdxs_analytical = GE.d2GE_dTdxs()
    d2GE_dTdxs_expect = [0]*4
    assert_close1d(d2GE_dTdxs_analytical, d2GE_dTdxs_expect, rtol=1e-12)
    assert_close1d(d2GE_dTdxs_numerical, d2GE_dTdxs_analytical, rtol=1e-7)

    # dGE_dxs
    def dGE_dx_diff(xs):
        return GE.to_T_xs(T, xs).GE()

    dGE_dxs_numerical = jacobian(dGE_dx_diff, xs, perturbation=1e-7)
    dGE_dxs_analytical = GE.dGE_dxs()
    dGE_dxs_expect = [439.92463410596037, 662.6790758115604, 2962.5490239819123, 13189.738825326536]
    assert_close1d(dGE_dxs_analytical, dGE_dxs_expect, rtol=1e-12)
    assert_close1d(dGE_dxs_analytical, dGE_dxs_numerical, rtol=1e-7)

    # d2GE_dxixjs
    d2GE_dxixjs_numerical = hessian(dGE_dx_diff, xs, perturbation=1e-5)
    d2GE_dxixjs_analytical = GE.d2GE_dxixjs()
    d2GE_dxixjs_expect = [[-1022.4173091041094, -423.20895951381453, 1638.9017092099375, 2081.4926965380164],
                          [-423.20895951381453, -1674.3900233778054, 1920.6043029143648, 2874.797302359955],
                          [1638.901709209937, 1920.6043029143648, -3788.1956922483323, -4741.028361086175],
                          [2081.4926965380164, 2874.797302359955, -4741.028361086175, -7468.305971059591]]
    d2GE_dxixjs_sympy = [[-1022.4173091041112, -423.208959513817, 1638.9017092099352, 2081.492696538016],
                         [-423.208959513817, -1674.3900233778083, 1920.6043029143652, 2874.7973023599534],
                         [1638.9017092099352, 1920.6043029143652, -3788.1956922483323, -4741.028361086176],
                         [2081.492696538016, 2874.7973023599534, -4741.028361086176, -7468.305971059591]]
    assert_close2d(d2GE_dxixjs_analytical, d2GE_dxixjs_sympy, rtol=1e-12)
    assert_close2d(d2GE_dxixjs_analytical, d2GE_dxixjs_expect, rtol=1e-12)
    assert_close2d(d2GE_dxixjs_analytical, d2GE_dxixjs_numerical, rtol=2.5e-4)


    d3GE_dxixjxks_analytical = GE.d3GE_dxixjxks()
    d3GE_dxixjxks_sympy = [[[3564.2598967437325, 2275.2388316927168, -3155.248707372427, -4548.085576267108],
                            [2275.2388316927168, 3015.024292098843, -4031.740524903445, -5850.4575581223535],
                            [-3155.248707372427, -4031.740524903445, 2306.3682432066844, 3714.462825687298],
                            [-4548.085576267108, -5850.4575581223535, 3714.462825687298, 7499.862362680743]],
                           [[2275.2388316927168, 3015.024292098843, -4031.740524903445, -5850.4575581223535],
                            [3015.024292098843, 6346.017369615182, -3782.270609497761, -6789.70782446731],
                            [-4031.740524903445, -3782.270609497761, 2329.947090204009, 3607.836718555389],
                            [-5850.4575581223535, -6789.70782446731, 3607.836718555389, 7807.307245181044]],
                           [[-3155.248707372427, -4031.740524903445, 2306.3682432066844, 3714.462825687298],
                            [-4031.740524903445, -3782.270609497761, 2329.947090204009, 3607.836718555389],
                            [2306.3682432066844, 2329.947090204009, 7265.918548487337, 7134.805582069884],
                            [3714.462825687298, 3607.836718555389, 7134.805582069884, 7459.310988306651]],
                           [[-4548.085576267108, -5850.4575581223535, 3714.462825687298, 7499.862362680743],
                            [-5850.4575581223535, -6789.70782446731, 3607.836718555389, 7807.307245181044],
                            [3714.462825687298, 3607.836718555389, 7134.805582069884, 7459.310988306651],
                            [7499.862362680743, 7807.307245181044, 7459.310988306651, 6343.066547716518]]]
    assert_close3d(d3GE_dxixjxks_analytical, d3GE_dxixjxks_sympy, rtol=1e-12)

    # Test with some stored results
    GE2 = RegularSolution.from_json(GE.as_json())
    assert GE2.__dict__ == GE.__dict__

def test_create_many_components_regular_solution():
    # Just create it. This can be used for easy benchmarking.
    N = 10
    xs = normalize([random() for _ in range(N)])
    xs2 = normalize([random() for _ in range(N)])
    SPs = [50000.0*random() for _ in range(N)]
    Vs = [1e-5*random() for _ in range(N)]

    T = 300.0
    lambda_coeffs = [[random()*1e-4 for _ in range(N)] for _ in range(N)]

    GE = RegularSolution(T, xs, Vs, SPs, lambda_coeffs)


def test_numpy_inputs():
    xs = [.4, .3, .2, .1]
    SPs = [19570.2, 18864.7, 29261.4, 47863.5]
    Vs = [7.421e-05, 8.068e-05, 4.083e-05, 1.808e-05]
    N = 4
    T = 300.0
    # Made up asymmetric parameters
    lambda_coeffs = [[0.0, 0.01811, 0.01736, 0.02111],
     [0.00662, 0.0, 0.00774, 0.01966],
     [0.01601, 0.01022, 0.0, 0.00698],
     [0.0152, 0.00544, 0.02579, 0.0]]

    model = RegularSolution(T, xs, Vs, SPs, lambda_coeffs)
    modelnp =  RegularSolution(T, np.array(xs), np.array(Vs), np.array(SPs), np.array(lambda_coeffs))
    modelnp2 = modelnp.to_T_xs(T=model.T*(1.0-1e-16), xs=np.array(xs))
    check_np_output_activity(model, modelnp, modelnp2)

    json_string = modelnp.as_json()
    new = RegularSolution.from_json(json_string)
    assert new == modelnp

    assert model.model_hash() == modelnp.model_hash()
    assert new.model_hash() == modelnp.model_hash()
    assert new.model_hash() == modelnp2.model_hash()


    # Pickle checks
    modelnp_pickle = pickle.loads(pickle.dumps(modelnp))
    assert modelnp_pickle == modelnp
    model_pickle = pickle.loads(pickle.dumps(model))
    assert model_pickle == model

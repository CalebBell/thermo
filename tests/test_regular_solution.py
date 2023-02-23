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
from chemicals.utils import object_data
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
    assert object_data(GE2) == object_data(GE)

    
    # Test with no interaction parameters
    GE3 = RegularSolution(T, xs, Vs, SPs)
    GE4 = eval(str(GE3))
    assert GE3 == GE4
    assert GE4.GE() == GE3.GE()
    GE5 = RegularSolution.from_json(GE4.as_json())
    assert object_data(GE4) == object_data(GE3)
    assert object_data(GE5) == object_data(GE3)
    assert GE5 == GE3
    assert hash(GE5) == hash(GE3)
    GE6 = eval(str(GE4))
    assert GE5.model_hash() == GE6.model_hash()
    assert GE5.state_hash() == GE6.state_hash()
    
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
                   regular_solution_gammas(xs=xs, T=T, Vs=Vs, SPs=SPs, lambda_coeffs=lambda_coeffs, N=N), rtol=1e-12)


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
    assert object_data(GE2) == object_data(GE)

    # Direct call for gammas
    gammas_args = GE.gammas_args()
    gammas = GE.gammas_from_args(GE.xs, *gammas_args)
    assert_close1d(gammas, GE.gammas(), rtol=1e-13)

    # gammas at another T
    T_another = 401.234
    gammas_args_at_T = GE.gammas_args(T=T_another)
    gammas_at_T = GE.gammas_from_args(GE.xs, *gammas_args_at_T)
    assert_close1d(gammas_at_T, GE.to_T_xs(T=T_another, xs=GE.xs).gammas(), rtol=1e-13)

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
def test_regular_solution_gammas_binaries():
    kwargs = dict(xs=[.1, .9, 0.3, 0.7, .85, .15], Vs=[7.421e-05, 8.068e-05], SPs=[19570.2, 18864.7], Ts=[300.0, 400.0, 500.0], lambda12=0.1759, lambda21=0.7991)
    gammas_expect = [6818.906971998236, 1.105437709428331, 62.66284813913256, 2.0118436126911754, 1.1814344452004402, 137.6232341969005]
    gammas_1 = regular_solution_gammas_binaries(**kwargs)
    assert_close1d(gammas_1, gammas_expect, rtol=1e-13)
    
    # Check the array can be set to bad values and still be right
    for i in range(len(gammas_1)):
        gammas_1[i] = -1e100
    gammas_out = regular_solution_gammas_binaries(gammas=gammas_1, **kwargs)
    assert_close1d(gammas_out, gammas_expect, rtol=1e-13)
    
    assert gammas_out is gammas_1
    
def test_regular_solution_gammas_binaries_jac():
    kwargs = dict(xs=[.1, .9, 0.3, 0.7, .85, .15], Vs=[7.421e-05, 8.068e-05], SPs=[19570.2, 18864.7], Ts=[300.0, 400.0, 500.0], lambda12=0.1759, lambda21=0.7991)

    res = regular_solution_gammas_binaries_jac(**kwargs)
    res_expect = [[61651.76714001304, 61651.76714001304],
     [0.1134949564269304, 0.1134949564269304],
     [265.5654802072593, 265.5654802072593],
     [1.4404516377618741, 1.4404516377618741],
     [0.20175156664075444, 0.20175156664075444],
     [694.1461546179326, 694.1461546179326]]
    assert_close2d(res, res_expect, rtol=1e-13)
    
    for i in range(len(res)):
        for j in range(len(res[i])):
            res[i][j] = -1e100
    jac = res
    res = regular_solution_gammas_binaries_jac(jac=res, **kwargs)
    assert res is jac
    assert_close2d(res, res_expect, rtol=1e-13)
    '''
    # from math import *
    # from fluids.constants import R
    from sympy import *
    x0, x1, V0, V1, SP0, SP1, l01, l10, T, R = symbols('x0, x1, V0, V1, SP0, SP1, l01, l10, T, R')
    
    x0V0 = x0*V0
    x1V1 = x1*V1
    den_inv = 1/(x0V0 + x1V1)
    phi0, phi1 = x0V0*den_inv, x1V1*den_inv
    
    c0 = (SP0-SP1)
    term = (c0*c0 + l01*SP0*SP1 + l10*SP0*SP1)/(R*T)
    gamma0 = exp(V0*phi1*phi1*term)
    gamma1 = exp(V1*phi0*phi0*term)
    res, v0ut = cse([diff(gamma0, l01),diff(gamma0, l10), diff(gamma1, l01),diff(gamma1, l10)], optimizations='basic')
    
    for k, v in res:
        print("%s = %s" %(k, v))
    print(v0ut)
    '''

def test_regular_solution_gammas_fit_not_great():
    kwargs = {'gammas': [[3.829745434386257, 1.0000000000000115], [3.4810418573295845, 1.0020344642061478], [3.175726369649537, 1.0081872885481726], [2.907490956454007, 1.018588416656049], [2.671083674402019, 1.0334476519274989], [2.462114139900209, 1.0530644544412704], [2.2768984638071914, 1.0778419332378688], [2.1123350932679226, 1.1083061694346559], [1.9658050199531931, 1.145132507958062], [1.8350913252806818, 1.1891811756807438], [1.7183141857813118, 1.2415456387911439], [1.6138783502934815, 1.3036186894783692], [1.520430796000932, 1.3771836563945803], [1.4368268271500177, 1.46454187236492], [1.3621033430122802, 1.5686934707154079], [1.295458409763599, 1.6935982269416847], [1.2362366645881748, 1.844559220108991], [1.183920507790576, 2.0287995532670373], [1.1381275682311767, 2.2563507731094825], [1.0986156680300956, 2.5414598130378963], [1.065297654195464, 2.9048880634217995], [1.0382703599803569, 3.3778051922368593], [1.0178652958072034, 4.008661386646446], [1.004734863028303, 4.875911541714439], [1.0000000000000289, 6.112940534948909]],
              'xs': [[1e-07, 0.9999999], [0.04166675833333334, 0.9583332416666667], [0.08333341666666667, 0.9166665833333333], [0.12500007500000002, 0.874999925], [0.16666673333333334, 0.8333332666666666], [0.20833339166666667, 0.7916666083333334], [0.25000005000000003, 0.74999995], [0.29166670833333336, 0.7083332916666667], [0.3333333666666667, 0.6666666333333333], [0.375000025, 0.6249999749999999], [0.41666668333333334, 0.5833333166666667], [0.45833334166666667, 0.5416666583333334], [0.5, 0.5], [0.5416666583333334, 0.4583333416666666], [0.5833333166666668, 0.41666668333333323], [0.6249999750000002, 0.37500002499999985], [0.6666666333333335, 0.33333336666666646], [0.7083332916666669, 0.2916667083333331], [0.7499999500000003, 0.2500000499999997], [0.7916666083333337, 0.2083333916666663], [0.8333332666666671, 0.16666673333333293], [0.8749999250000005, 0.12500007499999954], [0.9166665833333338, 0.08333341666666616], [0.9583332416666672, 0.041666758333332776], [0.9999999, 9.999999994736442e-08]],
              'Vs': [0.00015769266229156943, 0.00010744696872937524], 'SPs': [17501.26952205472, 20382.758536369853],
              'Ts': [298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15], 
              'use_numba': False, 'multiple_tries': False}
    
    res, stats = RegularSolution.regress_binary_parameters(**kwargs)
    assert stats['MAE'] < .12
    


def test_regular_solution_one_component():
    GE = RegularSolution(T=325.0, xs=[1], Vs=[7.421e-05], SPs=[19570.2])
    
    for s in GE._point_properties:
        if hasattr(GE, s):
            res = getattr(GE, s)()


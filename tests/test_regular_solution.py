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
SOFTWARE.
'''

import pickle
from math import log
from random import random

import numpy as np
from chemicals import normalize
from chemicals.utils import object_data
from fluids.constants import R
from fluids.numerics import assert_close, assert_close1d, assert_close2d, assert_close3d, hessian, jacobian

from thermo import *
from thermo.activity import GibbsExcess
from thermo.test_utils import check_np_output_activity
import json

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

    GE = RegularSolution(T=T, xs=xs, Vs=Vs, SPs=SPs, lambda_coeffs=lambda_coeffs)
    assert eval(str(GE)).GE() == GE.GE()

    assert_close2d(GE.Aijs, [[0.0, 6934822.7146334015, 56900907.89686081, 420029112.876147], [2692876.1268428005, 0.0, 58318223.74716921, 438216815.695727], [56127828.43898281, 59687202.42796761, 0.0, 182794922.276922], [414493226.11404, 425377143.01236796, 209139324.56243098, 0.0]])

    GE2 = RegularSolution.from_json(GE.as_json())
    assert object_data(GE2) == object_data(GE)


    # Test with no interaction parameters
    GE3 = RegularSolution(T=T, xs=xs, Vs=Vs, SPs=SPs)
    GE4 = eval(str(GE3))
    assert GE3 == GE4
    assert GE4.GE() == GE3.GE()
    GE5 = RegularSolution.from_json(GE4.as_json())
    GE5.model_hash() # need to compute this or it won't match
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
    assert_close1d(GE.gammas_dGE_dxs(), gammas_expect)

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

    # Test a few more storing
    GE_copy = RegularSolution.from_json(json.loads(json.dumps(GE.as_json(option=1))))
    assert GE_copy == GE

    # Direct call for gammas
    gammas_args = GE.gammas_args()
    gammas = GE.gammas_from_args(GE.xs, *gammas_args)
    assert_close1d(gammas, GE.gammas(), rtol=1e-13)

    # gammas at another T
    T_another = 401.234
    gammas_args_at_T = GE.gammas_args(T=T_another)
    gammas_at_T = GE.gammas_from_args(GE.xs, *gammas_args_at_T)
    assert_close1d(gammas_at_T, GE.to_T_xs(T=T_another, xs=GE.xs).gammas(), rtol=1e-13)

def test_regular_solution_additional_test_cases():
    # test case with parameters
    xs = [0.35, 0.25, 0.15, 0.25]
    SPs = [20450.1, 18020.5, 25540.3, 42010.7]
    Vs = [6.921e-05, 7.521e-05, 3.912e-05, 1.95e-05]
    lambda_coeffs = [[0.0, 0.01234, 0.01987, 0.02211], [0.00999, 0.0, 0.01457, 0.02033], [0.01543, 0.01276, 0.0, 0.00787], [0.01723, 0.00654, 0.02345, 0.0]]
    T = 313.0
    GE = RegularSolution(T=T, xs=xs, Vs=Vs, SPs=SPs, lambda_coeffs=lambda_coeffs)

    assert_close(GE.GE(), 2651.9714504398053, rtol=1e-12)
    assert_close(GE.dGE_dT(), 0.0, rtol=1e-12)
    assert_close(GE.d2GE_dT2(), 0.0, rtol=1e-12)
    assert_close(GE.d3GE_dT3(), 0.0, rtol=1e-12)
    assert_close1d(GE.dGE_dxs(), [392.4991832288646, 1399.966872640466, 879.1111482134028, 8130.953383670306], rtol=1e-12)
    assert_close2d(GE.d2GE_dxixjs(), [[-1010.4286288110145, -982.9146190883498, 815.9438189050574, 1907.948408080732], [-982.9146190883499, -3916.4391560672307, 1534.320970704279, 4371.9270403683495], [815.9438189050577, 1534.3209707042793, -1279.2065360378033, -1909.1183955486792], [1907.9484080807324, 4371.9270403683495, -1909.11839554868, -5897.583774352173]], rtol=1e-12)
    assert_close3d(GE.d3GE_dxixjxks(), [[[3901.789064331325, 3943.7106650130345, -1365.3773634203949, -4545.274421780581], [3943.7106650130345, 7790.860365539298, -2401.1208894261813, -7939.7242865484295], [-1365.3773634203949, -2401.1208894261813, 459.26909498120983, 773.3124656057823], [-4545.274421780583, -7939.724286548428, 773.3124656057832, 6207.327365354859]], [[3943.7106650130354, 7790.860365539301, -2401.1208894261817, -7939.724286548426], [7790.860365539304, 16434.491375763908, -1442.8659753387205, -10810.21967804676], [-2401.1208894261817, -1442.8659753387196, -443.3065367125964, -1066.8647402541787], [-7939.724286548428, -10810.21967804676, -1066.8647402541797, 5078.244361893684]], [[-1365.377363420393, -2401.1208894261777, 459.2690949812102, 773.3124656057827], [-2401.1208894261777, -1442.8659753387183, -443.3065367125969, -1066.8647402541792], [459.2690949812102, -443.3065367125969, 2792.08612898504, 3241.904270499095], [773.312465605782, -1066.864740254182, 3241.904270499096, 5675.558308301354]], [[-4545.274421780585, -7939.724286548415, 773.3124656057789, 6207.327365354859], [-7939.7242865484295, -10810.219678046768, -1066.8647402541828, 5078.244361893683], [773.3124656057789, -1066.8647402541828, 3241.904270499097, 5675.558308301355], [6207.327365354858, 5078.244361893683, 5675.558308301355, 6416.497439037405]]], rtol=1e-12)

    assert_close1d(GE.gammas(), [1.1627878617775766, 1.7124869764213169, 1.4018662309653585, 22.745640971617053], rtol=1e-12)
    assert_close1d(GE.gammas_dGE_dxs(), [1.1627878617775766, 1.7124869764213169, 1.4018662309653585, 22.745640971617053], rtol=1e-12)
    assert_close1d(GE.gammas_infinite_dilution(), [1.6479084374328064, 3.568103846536692, 1.5305660547296676, 43.74896036741677], rtol=1e-10)

    assert_close1d(GE.d2GE_dTdxs(), [0.0, 0.0, 0.0, 0.0], rtol=1e-12)

    # Test case without parameters
    xs = [0.35, 0.25, 0.15, 0.25]
    SPs = [20450.1, 18020.5, 25540.3, 42010.7]
    Vs = [6.921e-05, 7.521e-05, 3.912e-05, 1.95e-05]
    T = 313.0
    GE = RegularSolution(T=T, xs=xs, Vs=Vs, SPs=SPs, lambda_coeffs=None)

    assert_close(GE.GE(), 2380.928482132091, rtol=1e-12)
    assert_close(GE.dGE_dT(), 0.0, rtol=1e-12)
    assert_close(GE.d2GE_dT2(), 0.0, rtol=1e-12)
    assert_close(GE.d3GE_dT3(), 0.0, rtol=1e-12)
    assert_close1d(GE.dGE_dxs(), [190.87803299614657, 1258.3105377035863, 460.1062893945279, 7722.110370993459], rtol=1e-12)
    assert_close2d(GE.d2GE_dxixjs(), [[-491.3860649691572, -1315.2010514136468, 573.5733232874426, 1658.9975483979983], [-1315.2010514136468, -3520.1523383617605, 1535.1762934054984, 4440.328034297561], [573.5733232874423, 1535.1762934054987, -669.5068921167934, -1936.474810737843], [1658.997548397998, 4440.328034297565, -1936.4748107378423, -5601.039715612062]], rtol=1e-12)
    assert_close3d(GE.d3GE_dxixjxks(), [[[1897.496488116688, 4073.1141639793836, -1119.06482448724, -4092.6260927737617], [4073.1141639793823, 8210.34956000034, -1821.4461079756238, -7559.037519131501], [-1119.0648244872386, -1821.4461079756225, 27.156822507277127, 1077.549475603624], [-4092.626092773761, -7559.037519131506, 1077.5494756036235, 6006.19417006061]], [[4073.1141639793796, 8210.34956000034, -1821.4461079756238, -7559.037519131502], [8210.349560000344, 14771.55930191308, -1733.5799173751266, -11145.291382041421], [-1821.4461079756215, -1733.5799173751252, -1297.3754364028002, -1078.6754432393122], [-7559.037519131501, -11145.291382041425, -1078.675443239315, 4613.83703757887]], [[-1119.064824487238, -1821.446107975622, 27.15682250727673, 1077.549475603626], [-1821.446107975622, -1733.5799173751232, -1297.375436402801, -1078.675443239314], [27.156822507276956, -1297.3754364028014, 1461.3128170288999, 3060.5957631424485], [1077.5494756036264, -1078.6754432393184, 3060.595763142448, 5479.64796246015]], [[-4092.626092773764, -7559.037519131511, 1077.5494756036242, 6006.194170060609], [-7559.037519131511, -11145.291382041421, -1078.6754432393172, 4613.83703757887], [1077.5494756036242, -1078.6754432393209, 3060.595763142448, 5479.647962460149], [6006.1941700606085, 4613.837037578869, 5479.647962460149, 6093.861209308443]]], rtol=1e-12)

    assert_close1d(GE.gammas(), [1.0761029847623977, 1.6217637421404603, 1.193391121115375, 19.43883451204506], rtol=1e-12)
    assert_close1d(GE.gammas_dGE_dxs(), [1.0761029847623977, 1.6217637421404603, 1.193391121115375, 19.43883451204506], rtol=1e-12)
    assert_close1d(GE.gammas_infinite_dilution(), [1.2749633329647372, 3.1371755931764684, 1.2495316747929026, 36.178962626768936], rtol=1e-10)

    assert_close1d(GE.d2GE_dTdxs(), [0.0, 0.0, 0.0, 0.0], rtol=1e-12)

    # Test case with symmetric parameter matrix, no particular reason to have this except to check it's handled
    xs = [0.35, 0.25, 0.15, 0.25]
    SPs = [20450.1, 18020.5, 25540.3, 42010.7]
    Vs = [6.921e-05, 7.521e-05, 3.912e-05, 1.95e-05]
    lambda_coeffs = [[0.0, 0.01345, 0.01876, 0.02011], [0.01345, 0.0, 0.01567, 0.02223], [0.01876, 0.01567, 0.0, 0.01989], [0.02011, 0.02223, 0.01989, 0.0]]
    T = 1234.0
    GE = RegularSolution(T=T, xs=xs, Vs=Vs, SPs=SPs, lambda_coeffs=lambda_coeffs)
    assert_close(GE.GE(), 2702.280835275276, rtol=1e-12)
    assert_close(GE.dGE_dT(), 0.0, rtol=1e-12)
    assert_close(GE.d2GE_dT2(), 0.0, rtol=1e-12)
    assert_close(GE.d3GE_dT3(), 0.0, rtol=1e-12)
    assert_close1d(GE.dGE_dxs(), [382.0039859938338, 1492.6133977499169, 920.3866506466712, 8229.472372571818], rtol=1e-12)
    assert_close2d(GE.d2GE_dxixjs(), [[-983.4103617561509, -924.44764670667, 828.837152869782, 1803.9198614434076], [-924.4476467066698, -4175.619916486126, 1510.1710897077596, 4563.743968050809], [828.8371528697819, 1510.1710897077596, -1339.2670785507553, -1866.9828565950029], [1803.9198614434063, 4563.7439680508105, -1866.9828565950029, -5969.042060114582]], rtol=1e-12)
    assert_close3d(GE.d3GE_dxixjxks(), [[[3797.4575203449876, 3755.4043512956823, -1418.226586849944, -4287.267480644079], [3755.4043512956787, 7960.908133638751, -2430.6085559866387, -8062.31850503404], [-1418.2265868499435, -2430.6085559866374, 517.8161331987953, 790.0874861781573], [-4287.267480644077, -8062.318505034039, 790.0874861781573, 6374.761040455234]], [[3755.4043512956805, 7960.9081336387535, -2430.6085559866374, -8062.31850503404], [7960.908133638756, 17522.087480830312, -1186.737417115885, -11252.836751710527], [-2430.6085559866365, -1186.7374171158835, -324.15535124203143, -1256.6017525886414], [-8062.318505034038, -11252.83675171053, -1256.6017525886468, 5039.067838108133]], [[-1418.2265868499435, -2430.6085559866347, 517.8161331987949, 790.087486178156], [-2430.6085559866347, -1186.737417115883, -324.1553512420321, -1256.6017525886414], [517.8161331987951, -324.1553512420321, 2923.1784920440523, 3202.37398374031], [790.0874861781572, -1256.6017525886464, 3202.37398374031, 5696.986308075055]], [[-4287.267480644074, -8062.318505034033, 790.0874861781576, 6374.761040455233], [-8062.3185050340435, -11252.836751710536, -1256.6017525886455, 5039.067838108134], [790.0874861781576, -1256.6017525886455, 3202.373983740312, 5696.986308075055], [6374.761040455233, 5039.067838108133, 5696.986308075055, 6494.243160867844]]], rtol=1e-12)

    assert_close1d(GE.gammas(), [1.0379339878709646, 1.1565925484420827, 1.0938525249883237, 2.2301952989686216], rtol=1e-12)
    assert_close1d(GE.gammas_dGE_dxs(), [1.0379339878709646, 1.1565925484420827, 1.0938525249883237, 2.2301952989686216], rtol=1e-12)
    assert_close1d(GE.gammas_infinite_dilution(), [1.1312355258340416, 1.41057767504653, 1.1196661045356717, 2.6379662312061507], rtol=1e-10)

    assert_close1d(GE.d2GE_dTdxs(), [0.0, 0.0, 0.0, 0.0], rtol=1e-12)



def test_create_many_components_regular_solution():
    # Just create it. This can be used for easy benchmarking.
    N = 10
    xs = normalize([random() for _ in range(N)])
    xs2 = normalize([random() for _ in range(N)])
    SPs = [50000.0*random() for _ in range(N)]
    Vs = [1e-5*random() for _ in range(N)]

    T = 300.0
    lambda_coeffs = [[random()*1e-4 for _ in range(N)] for _ in range(N)]

    GE = RegularSolution(T=T, xs=xs, Vs=Vs, SPs=SPs, lambda_coeffs=lambda_coeffs)

    assert_close1d(GE.SPs, SPs)
    assert_close1d(GE.xs, xs)
    assert_close1d(GE.Vs, Vs)
    assert_close2d(GE.lambda_coeffs, lambda_coeffs)


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

    model = RegularSolution(T=T, xs=xs, Vs=Vs, SPs=SPs, lambda_coeffs=lambda_coeffs)
    modelnp =  RegularSolution(T=T, xs=np.array(xs), Vs=np.array(Vs), SPs=np.array(SPs), lambda_coeffs=np.array(lambda_coeffs))
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
    from thermo.regular_solution import MIN_LAMBDA_REGULAR_SOLUTION
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


    # Test case 2: Check with pre-allocated array
    gammas_preallocated = [0.0] * len(gammas_expect)
    gammas_out = regular_solution_gammas_binaries(gammas=gammas_preallocated, **kwargs)
    assert_close1d(gammas_out, gammas_expect, rtol=1e-13)
    assert gammas_out is gammas_preallocated

    # Test case 3: Verify behavior with lambda values outside bounds
    kwargs_small_lambda = kwargs.copy()
    kwargs_small_lambda['lambda12'] = MIN_LAMBDA_REGULAR_SOLUTION - 1
    kwargs_small_lambda['lambda21'] = MIN_LAMBDA_REGULAR_SOLUTION - 1
    gammas_small_lambda = regular_solution_gammas_binaries(**kwargs_small_lambda)
    kwargs_small_lambda['lambda12'] = MIN_LAMBDA_REGULAR_SOLUTION
    kwargs_small_lambda['lambda21'] = MIN_LAMBDA_REGULAR_SOLUTION
    gammas_min_lambda = regular_solution_gammas_binaries(**kwargs_small_lambda)
    assert_close1d(gammas_small_lambda, gammas_min_lambda, rtol=1e-13)




def test_regular_solution_gammas_binaries_jac():
    from thermo.regular_solution import MIN_LAMBDA_REGULAR_SOLUTION
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


    # Test case 2: Check with pre-allocated array
    jac_preallocated = np.zeros((len(res_expect), 2))
    res_preallocated = regular_solution_gammas_binaries_jac(jac=jac_preallocated, **kwargs)
    assert_close2d(res_preallocated, res_expect, rtol=1e-13)
    assert res_preallocated is jac_preallocated

    # Test case 3: Verify behavior with lambda values outside bounds
    kwargs_small_lambda = kwargs.copy()
    kwargs_small_lambda['lambda12'] = MIN_LAMBDA_REGULAR_SOLUTION - 1
    kwargs_small_lambda['lambda21'] = MIN_LAMBDA_REGULAR_SOLUTION - 1
    jac_small_lambda = regular_solution_gammas_binaries_jac(**kwargs_small_lambda)
    kwargs_small_lambda['lambda12'] = MIN_LAMBDA_REGULAR_SOLUTION
    kwargs_small_lambda['lambda21'] = MIN_LAMBDA_REGULAR_SOLUTION
    jac_min_lambda = regular_solution_gammas_binaries_jac(**kwargs_small_lambda)
    assert_close2d(jac_small_lambda, jac_min_lambda, rtol=1e-13)



    """
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
    """

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





def test_FloryHuggins():
    GE = FloryHuggins(T=298.15, xs=[0.5, 0.5], Vs=[0.05868e-3, 0.01807e-3], SPs=[26140.0, 47860.0])

    assert_close1d(GE.gammas(), [1.67294569300073, 5.96634717252448])
    assert_close(GE.dGE_dT(), -1.3654856671723135)
    assert_close(GE.GE(), 2851.6940458559634)

    # Test at new temperature and composition
    GE2 = GE.to_T_xs(T=350.0, xs=[0.3, 0.3, 0.4])
    
    # Check basic property copying
    assert GE2.Vs == GE.Vs
    assert GE2.Aijs == GE.Aijs
    assert GE2.His == GE.His



    GE = FloryHuggins(T=298.15, xs=[0.1, 0.2, .3, .4], Vs=[0.05868e-3, 0.01807e-3, .01e-3, .015e-3], SPs=[26140.0, 47860.0, 25000, 29000])
    assert_close3d(GE.d3GE_dxixjxks(), 
            [[[4682.268003264035, -60243.275483154124, 23822.555766391255, 11165.425283221184], [-60243.275483154124, 15377.860774237011, -4713.424419068196, -6544.967488966422], [23822.555766391255, -4713.424419068196, 8028.151243405697, 7620.37590806656], [11165.425283221184, -6544.967488966422, 7620.37590806656, 6034.659174784874]], [[-60243.275483154124, 15377.860774237011, -4713.424419068196, -6544.967488966422], [15377.860774237011, 31481.39619239953, 3857.240312914296, 8439.073451792432], [-4713.424419068196, 3857.240312914296, 156.82387703512063, 252.36498431714358], [-6544.967488966422, 8439.073451792432, 252.36498431714358, 785.1252189005604]], [[23822.555766391255, -4713.424419068196, 8028.151243405697, 7620.37590806656], [-4713.424419068196, 3857.240312914296, 156.82387703512063, 252.36498431714358], [8028.151243405697, 156.82387703512063, 2052.0131749207917, 2284.417753313513], [7620.37590806656, 252.36498431714358, 2284.417753313513, 2447.0061228868267]], [[11165.425283221184, -6544.967488966422, 7620.37590806656, 6034.659174784874], [-6544.967488966422, 8439.073451792432, 252.36498431714358, 785.1252189005604], [7620.37590806656, 252.36498431714358, 2284.417753313513, 2447.0061228868267], [6034.659174784874, 785.1252189005604, 2447.0061228868267, 2517.2521834821773]]])

    # Check the repr feature
    hash(GE)
    assert eval(str(GE)).GE() == GE.GE()
    GE2 = FloryHuggins.from_json(GE.as_json())
    GE2.model_hash() # need to compute this or it won't match
    assert object_data(GE2) == object_data(GE)
    assert hash(GE) == hash(GE2)
    assert GE2.state_hash() == GE.state_hash()

    # numpy tests
    modelnp = FloryHuggins(T=298.15, xs=np.array([0.1, 0.2, .3, .4]), Vs=np.array([0.05868e-3, 0.01807e-3, .01e-3, .015e-3]), SPs=np.array([26140.0, 47860.0, 25000, 29000]))
    modelnp2 = modelnp.to_T_xs(T=modelnp.T*(1.0-2e-16), xs=modelnp.xs)
    check_np_output_activity(GE, modelnp, modelnp2)

    # # Pickle checks
    modelnp_pickle = pickle.loads(pickle.dumps(modelnp))
    assert modelnp_pickle == modelnp
    model_pickle = pickle.loads(pickle.dumps(GE))
    assert model_pickle == GE

def test_FloryHuggins_very_painful_discovery_why_derivatives_not_working():
    GE = FloryHuggins(T=298.15, xs=[0.1, 0.2, .7], Vs=[0.05868e-3, 0.01807e-3, .01e-3], SPs=[26140.0, 47860.0, 25000])
    assert_close1d(GE.dGE_dxs(), [-4603.741796921449, 3007.310038738686, -2449.1463565656745])
    assert_close2d(GE.d2GE_dxixjs(), [[6122.120688122769, 7733.91667069719, -6625.646332220143],
    [7733.91667069719, -14509.37310982012, -500.67724958297595],
    [-6625.646332220143, -500.67724958297595, -2451.7956378053977]])



def test_FloryHuggins_bulk():
    # asymmetric
    xs = [0.35, 0.25, 0.15, 0.25]
    SPs = [20450.1, 18020.5, 25540.3, 42010.7]
    Vs = [6.921e-05, 7.521e-05, 3.912e-05, 1.95e-05]
    lambda_coeffs = [[0.0, 0.01234, 0.01987, 0.02211], [0.00999, 0.0, 0.01457, 0.02033], [0.01543, 0.01276, 0.0, 0.00787], [0.01723, 0.00654, 0.02345, 0.0]]
    T = 298.0
    GE = FloryHuggins(T=T, xs=xs, Vs=Vs, SPs=SPs, lambda_coeffs=lambda_coeffs)

    assert_close(GE.GE(), 2332.2812854606336, rtol=1e-12)
    assert_close(GE.dGE_dT(), -1.072785788520712, rtol=1e-12)
    assert_close(GE.d2GE_dT2(), 0.0, rtol=1e-12)
    assert_close(GE.d3GE_dT3(), 0.0, rtol=1e-12)
    assert_close1d(GE.dGE_dxs(), [-2171.2482559002356, -1234.2704748637198, -1711.6318813614912, 4719.282862944814], rtol=1e-12)
    assert_close2d(GE.d2GE_dxixjs(), [[-3283.8060231555314, -3176.893347854803, -1855.6178833134484, -1023.2469305780412], [-3176.8933478548033, -6000.166813746368, -1212.5668795650683, 1264.5181876434758], [-1855.6178833134481, -1212.5668795650681, -3573.007605781338, -3956.6029611659615], [-1023.246930578041, 1264.5181876434758, -3956.6029611659624, -7368.85014897331]], rtol=1e-12)
    assert_close3d(GE.d3GE_dxixjxks(), [[[5649.144030788884, 5486.666009210266, 1407.0417086727073, -1104.4685848962067], [5486.666009210264, 9080.843732052752, 410.488922855227, -4300.896107241049], [1407.0417086727086, 410.4889228552297, 3035.146605029971, 3221.0362552387896], [-1104.4685848962076, -4300.896107241046, 3221.0362552387915, 8007.5180952646415]], [[5486.666009210264, 9080.843732052754, 410.488922855227, -4300.896107241046], [9080.843732052754, 17416.044864225805, 1393.893229603289, -6964.894771876156], [410.48892285522925, 1393.8932296032908, 2242.1790713758874, 1536.382353834133], [-4300.896107241046, -6964.894771876156, 1536.3823538341317, 7006.247159139253]], [[1407.0417086727102, 410.48892285523243, 3035.1466050299705, 3221.03625523879], [410.48892285523243, 1393.893229603294, 2242.179071375887, 1536.3823538341328], [3035.1466050299705, 2242.1790713758865, 4818.2790273645915, 4909.678688288753], [3221.03625523879, 1536.3823538341294, 4909.678688288754, 6834.771520522168]], [[-1104.4685848962085, -4300.896107241033, 3221.036255238785, 8007.5180952646415], [-4300.896107241058, -6964.894771876161, 1536.382353834129, 7006.247159139252], [3221.036255238785, 1536.382353834129, 4909.678688288752, 6834.771520522169], [8007.518095264641, 7006.247159139252, 6834.771520522169, 7157.765191070198]]], rtol=1e-12)

    assert_close1d(GE.gammas(), [1.131662110093631, 1.6517746894900311, 1.362318364755486, 18.2596830409796], rtol=1e-12)
    assert_close1d(GE.gammas_dGE_dxs(), [1.131662110093631, 1.6517746894900311, 1.362318364755486, 18.2596830409796], rtol=1e-12)
    assert_close1d(GE.gammas_infinite_dilution(), [1.525715030529842, 3.3237300224669286, 1.4738221603973796, 31.900460063468852], rtol=1e-10)

    assert_close1d(GE.d2GE_dTdxs(), [-8.603179325936576, -8.839722642631493, -8.693768555620448, -11.448558794380846], rtol=1e-12)


    # symmetric
    xs = [0.35, 0.25, 0.15, 0.25]
    SPs = [20450.1, 18020.5, 25540.3, 42010.7]
    Vs = [6.921e-05, 7.521e-05, 3.912e-05, 1.95e-05]
    lambda_coeffs = [[0.0, 0.01345, 0.01876, 0.02011], [0.01345, 0.0, 0.01567, 0.02223], [0.01876, 0.01567, 0.0, 0.01989], [0.02011, 0.02223, 0.01989, 0.0]]
    T = 1234.0
    GE = FloryHuggins(T=T, xs=xs, Vs=Vs, SPs=SPs, lambda_coeffs=lambda_coeffs)

    assert_close(GE.GE(), 1378.4631722407178, rtol=1e-12)
    assert_close(GE.dGE_dT(), -1.072785788520712, rtol=1e-12)
    assert_close(GE.d2GE_dT2(), 0.0, rtol=1e-12)
    assert_close(GE.d3GE_dT3(), 0.0, rtol=1e-12)
    assert_close1d(GE.dGE_dxs(), [-10234.319302211903, -9415.604343257346, -9807.723746988962, -5898.049179694146], rtol=1e-12)
    assert_close2d(GE.d2GE_dxixjs(), [[-10397.328833639156, -10009.581040323463, -10233.938486518262, -10333.982983875138], [-10009.581040323463, -12804.210284191015, -9864.525578857258, -8303.84839524615], [-10233.938486518262, -9864.525578857258, -10837.758756616266, -10345.492769251805], [-10333.98298387514, -8303.848395246148, -10345.492769251805, -12061.467248981971]], rtol=1e-12)
    assert_close3d(GE.d3GE_dxixjxks(), [[[11033.148891514904, 10144.689232971488, 10062.193329132913, 9960.901655984511], [10144.689232971476, 13302.651335912591, 9212.097847890076, 7005.849190487127], [10062.19332913292, 9212.09784789008, 11184.369312394007, 10925.963849960479], [9960.901655984515, 7005.849190487126, 10925.963849960479, 13829.24211665883]], [[10144.689232971476, 13302.651335912593, 9212.097847890078, 7005.849190487131], [13302.651335912593, 21586.641188085294, 10560.11110267755, 4670.421416794596], [9212.097847890082, 10560.111102677552, 10796.278341312294, 9523.287220917995], [7005.849190487133, 4670.421416794592, 9523.287220917988, 13022.810964957236]], [[10062.19332913292, 9212.097847890083, 11184.369312394005, 10925.963849960479], [9212.097847890083, 10560.111102677554, 10796.278341312294, 9523.287220917991], [11184.369312394005, 10796.278341312294, 11313.52089674327, 10108.527109755203], [10925.963849960479, 9523.287220917984, 10108.527109755203, 10497.218200291445]], [[9960.901655984511, 7005.849190487135, 10925.963849960477, 13829.242116658828], [7005.849190487128, 4670.421416794592, 9523.287220917986, 13022.810964957234], [10925.963849960477, 9523.287220917986, 10108.527109755203, 10497.218200291447], [13829.24211665883, 13022.810964957232, 10497.218200291445, 9563.788147473437]]], rtol=1e-12)

    assert_close1d(GE.gammas(), [1.0025106953661578, 1.0857857993900115, 1.0450720945613647, 1.529810744892781], rtol=1e-12)
    assert_close1d(GE.gammas_dGE_dxs(), [1.0025106953661578, 1.0857857993900115, 1.0450720945613647, 1.529810744892781], rtol=1e-12)
    assert_close1d(GE.gammas_infinite_dilution(), [1.0213484462730071, 1.2324746758420784, 1.0553023406568633, 1.590374464631847], rtol=1e-10)

    assert_close1d(GE.d2GE_dTdxs(), [-8.603179325936576, -8.839722642631493, -8.693768555620448, -11.448558794380846], rtol=1e-12)

    # No coeffs
    xs = [0.35, 0.25, 0.15, 0.25]
    SPs = [20450.1, 18020.5, 25540.3, 42010.7]
    Vs = [6.921e-05, 7.521e-05, 3.912e-05, 1.95e-05]
    lambda_coeffs = [[0.0, 0.01234, 0.01987, 0.02211], [0.00999, 0.0, 0.01457, 0.02033], [0.01543, 0.01276, 0.0, 0.00787], [0.01723, 0.00654, 0.02345, 0.0]]
    T = 298.0
    GE = FloryHuggins(T=T, xs=xs, Vs=Vs, SPs=SPs)

    assert_close(GE.GE(), 2061.2383171529195, rtol=1e-12)
    assert_close(GE.dGE_dT(), -1.072785788520712, rtol=1e-12)
    assert_close(GE.d2GE_dT2(), 0.0, rtol=1e-12)
    assert_close(GE.d3GE_dT3(), 0.0, rtol=1e-12)
    assert_close1d(GE.dGE_dxs(), [-2372.8694061329534, -1375.9268098005991, -2130.636740180366, 4310.439850267967], rtol=1e-12)
    assert_close2d(GE.d2GE_dxixjs(), [[-2764.7634593136754, -3509.1797801801013, -2097.988378931064, -1272.1977902607748], [-3509.1797801801013, -5603.879996040898, -1211.7115568638492, 1332.9191815726876], [-2097.988378931064, -1211.7115568638487, -2963.3079618603283, -3983.9593763551256], [-1272.1977902607737, 1332.9191815726913, -3983.9593763551247, -7072.306090233198]], rtol=1e-12)
    assert_close3d(GE.d3GE_dxixjxks(), [[[3644.851454574251, 5616.069508176618, 1653.3542476058637, -651.8202558893859], [5616.069508176615, 9500.332926513793, 990.1637043057858, -3920.20933982412], [1653.354247605866, 990.1637043057881, 2603.0343325560384, 3525.2732652366312], [-651.8202558893877, -3920.2093398241245, 3525.2732652366312, 7806.384899970393]], [[5616.0695081766125, 9500.332926513793, 990.1637043057854, -3920.209339824122], [9500.332926513796, 15753.112790374977, 1103.1792875668866, -7299.966475870815], [990.163704305789, 1103.1792875668866, 1388.1101716856836, 1524.5716508489995], [-3920.209339824122, -7299.966475870819, 1524.5716508489966, 6541.839834824439]], [[1653.354247605867, 990.1637043057888, 2603.034332556038, 3525.2732652366335], [990.1637043057888, 1103.179287566888, 1388.1101716856833, 1524.571650848998], [2603.034332556038, 1388.1101716856824, 3487.5057154084516, 4728.370180932106], [3525.2732652366326, 1524.5716508489932, 4728.3701809321055, 6638.861174680964]], [[-651.8202558893877, -3920.209339824129, 3525.273265236632, 7806.384899970392], [-3920.209339824129, -7299.966475870815, 1524.5716508489909, 6541.839834824439], [3525.273265236632, 1524.5716508489909, 4728.370180932106, 6638.861174680963], [7806.3848999703905, 6541.839834824438, 6638.861174680963, 6835.128961341236]]], rtol=1e-12)

    assert_close1d(GE.gammas(), [1.0432214260177457, 1.5599877895558278, 1.1503637457481943, 15.482139782546332], rtol=1e-12)
    assert_close1d(GE.gammas_dGE_dxs(), [1.0432214260177457, 1.5599877895558278, 1.1503637457481943, 15.482139782546332], rtol=1e-12)
    assert_close1d(GE.gammas_infinite_dilution(), [1.165276206329276, 2.903443579297798, 1.1909827437757015, 26.129557166601312], rtol=1e-10)

    assert_close1d(GE.d2GE_dTdxs(), [-8.603179325936576, -8.839722642631493, -8.693768555620448, -11.448558794380846], rtol=1e-12)





def test_hansen():
    GE = Hansen(T=298.15, xs=[0.1, 0.2, 0.7], Vs=[0.00001806861809178579, 0.00005867599253092197, 0.00019583836334716688], delta_d=[15500.0, 15800.0, 15700.0], delta_p=[16000.0, 8800.0, 0.0], delta_h=[42300.0, 19400.0, 0.0])
    def basic_checks(GE):
        assert_close(GE.GE(), 1535.289806166052, rtol=1e-12)
        assert_close(GE.dGE_dT(), -1.8033466594907037, rtol=1e-12)
        assert_close(GE.d2GE_dT2(), 0.0, rtol=1e-12)
        assert_close(GE.d3GE_dT3(), 0.0, rtol=1e-12)
        assert_close1d(GE.dGE_dxs(), [2808.8577796977524, 2050.0323153458125, -2335.08494925039], rtol=1e-12)
        assert_close2d(GE.d2GE_dxixjs(), [[-2565.4618655079526, -4032.467086102314, -2022.7391797587563], [-4032.467086102314, -5725.419190797466, -1329.4664040466637], [-2022.7391797587563, -1329.4664040466637, -2872.5569011674006]], rtol=5e-3)
        assert_close1d(GE.gammas(), [8.440828466583636, 6.215049686485893, 1.059754569789273], rtol=1e-12)
        assert_close1d(GE.gammas_dGE_dxs(), [8.440828466583636, 6.215049686485893, 1.059754569789273], rtol=1e-12)
        assert_close1d(GE.gammas_infinite_dilution(), [8.43922245454429, 8.3046371474743, 21372.922653838257], rtol=1e-8)
        assert_close1d(GE.d2GE_dTdxs(), [-18.62936149437263, -11.077596921521437, -8.627648205574845], rtol=1e-10)
    basic_checks(GE)

    # Check the repr feature
    hash(GE)
    assert eval(str(GE)).GE() == GE.GE()
    GE2 = Hansen.from_json(GE.as_json())
    GE2.model_hash() # need to compute this or it won't match
    assert object_data(GE2) == object_data(GE)
    assert hash(GE) == hash(GE2)
    assert GE2.state_hash() == GE.state_hash()

    # numpy tests
    modelnp = Hansen(T=298.15, xs=np.array([0.1, 0.2, 0.7]), Vs=np.array([0.00001806861809178579, 0.00005867599253092197, 0.00019583836334716688]), 
                    delta_d=np.array([15500.0, 15800.0, 15700.0]), delta_p=np.array([16000.0, 8800.0, 0.0]), delta_h=np.array([42300.0, 19400.0, 0.0]))
    basic_checks(modelnp)
    modelnp2 = modelnp.to_T_xs(T=modelnp.T*(1.0-2e-16), xs=modelnp.xs)
    basic_checks(modelnp2)
    check_np_output_activity(GE, modelnp, modelnp2)


    # # Pickle checks
    modelnp_pickle = pickle.loads(pickle.dumps(modelnp))
    assert modelnp_pickle == modelnp
    model_pickle = pickle.loads(pickle.dumps(GE))
    assert model_pickle == GE

    # Test at new temperature and composition
    GE2 = GE.to_T_xs(T=350.0, xs=[0.3, 0.3, 0.4])
    
    # Check basic property copying
    assert GE2.Vs == GE.Vs
    assert GE2.delta_d == GE.delta_d
    assert GE2.delta_p == GE.delta_p
    assert GE2.delta_h == GE.delta_h
    assert GE2.alpha == GE.alpha
    assert GE2.N == GE.N
    assert GE2.Aijs == GE.Aijs
    assert GE2.His == GE.His


def test_hansen_4_components():
    GE = Hansen(T=298.15, xs=[0.4, 0.3, 0.2, 0.1], Vs=[0.00007401429647338668, 0.00008050481841759284, 0.00004074923154522018, 0.00001806861809178579], delta_d=[15500.0, 17800.0, 14700.0, 15500.0], delta_p=[10400.0, 3100.0, 12300.0, 16000.0], delta_h=[7000.0, 5700.0, 22300.0, 42300.0])
    assert_close(GE.GE(), 1091.1793973093831, rtol=1e-12)
    assert_close(GE.dGE_dT(), -0.7092139721671984, rtol=1e-12)
    assert_close(GE.d2GE_dT2(), 0.0, rtol=1e-12)
    assert_close(GE.d3GE_dT3(), 0.0, rtol=1e-12)
    assert_close1d(GE.dGE_dxs(), [-2248.4224908768606, -1588.3667895462709, -729.6159134360269, 1340.2458360882529], rtol=1e-12)
    assert_close2d(GE.d2GE_dxixjs(), [[-3017.799708308594, -2041.3476891966825, -2227.6134445986563, -2139.1015060021423], [-2041.3476891966825, -4742.943746948989, -681.0688178170785, -1033.2106627560217], [-2227.6134445986563, -681.0688178170785, -4668.93137118455, -4498.04732180892], [-2139.1015060021423, -1033.2106627560217, -4498.04732180892, -4137.437640129403]], rtol=5e-3)
    assert_close1d(GE.gammas(), [1.097457987713046, 1.4322692226443197, 2.025215786863174, 4.6676189435647295], rtol=1e-12)
    assert_close1d(GE.gammas_dGE_dxs(), [1.097457987713046, 1.4322692226443197, 2.025215786863174, 4.6676189435647295], rtol=1e-12)
    assert_close1d(GE.gammas_infinite_dilution(), [1.3876605163393116, 2.554734591753295, 2.5135666705238293, 4.994913561254703], rtol=1e-8)
    assert_close1d(GE.d2GE_dTdxs(), [-8.412664267695877, -8.560753751475001, -9.033924810384095, -12.83599795722768], rtol=1e-10)




def test_RegularSolution_FloryHuggins_missing_interaction_parameters():
    """Test Regular Solution model's missing parameter detection"""
    for model in (FloryHuggins, RegularSolution):
        # Test Case 1: All parameters present
        N = 2
        lambda_coeffs = [[0.0, 0.5], [-0.3, 0.0]]
        
        GE = model(T=300, xs=[0.4, 0.6], 
                            Vs=[89E-6, 109E-6],
                            SPs=[9000, 8000],
                            lambda_coeffs=lambda_coeffs)
        assert GE.missing_interaction_parameters() == []
        
        # Test Case 2: One direction missing (asymmetric case)
        lambda_coeffs = [[0.0, 0.5], [0.0, 0.0]]
        
        GE = model(T=300, xs=[0.4, 0.6],
                            Vs=[89E-6, 109E-6],
                            SPs=[9000, 8000],
                            lambda_coeffs=lambda_coeffs)
        assert GE.missing_interaction_parameters() == [(1, 0)]
        
        # Test Case 3: Multiple components with missing parameters
        N = 3
        lambda_coeffs = [
            [0.0, 0.5, 0.0],
            [0.3, 0.0, 0.0],
            [0.2, 0.0, 0.0]
        ]
        
        GE = model(T=300, xs=[0.3, 0.3, 0.4],
                            Vs=[89E-6, 109E-6, 120E-6],
                            SPs=[9000, 8000, 8500],
                            lambda_coeffs=lambda_coeffs)
        expected_missing = [(0, 2), (1, 2), (2, 1)] 
        assert sorted(GE.missing_interaction_parameters()) == sorted(expected_missing)
        
        # Test Case 4: All parameters missing (default case, no lambda_coeffs provided)
        GE = model(T=300, xs=[0.3, 0.3, 0.4],
                            Vs=[89E-6, 109E-6, 120E-6],
                            SPs=[9000, 8000, 8500])
        
        expected_missing = [(i, j) for i in range(N) for j in range(N) if i != j]
        assert sorted(GE.missing_interaction_parameters()) == sorted(expected_missing)




def test_Hansen_missing_interaction_parameters():
    """Test Hansen model's missing parameter detection"""
    
    # Test with typical values including zeros
    N = 2
    xs = [0.4, 0.6]
    Vs = [89E-6, 109E-6]
    delta_d = [16.6, 14.5]  # Dispersive parameters
    delta_p = [12.3, 0.0]   # One polar parameter is zero
    delta_h = [5.5, 0.0]    # One hydrogen bonding parameter is zero
    
    GE = Hansen(T=300, xs=xs, Vs=Vs, 
                delta_d=delta_d, delta_p=delta_p, delta_h=delta_h)
    assert GE.missing_interaction_parameters() == []
    
    # Test with all zero parameters (still valid)
    delta_d = [0.0, 0.0]
    delta_p = [0.0, 0.0]
    delta_h = [0.0, 0.0]
    
    GE = Hansen(T=300, xs=xs, Vs=Vs,
                delta_d=delta_d, delta_p=delta_p, delta_h=delta_h)
    assert GE.missing_interaction_parameters() == []
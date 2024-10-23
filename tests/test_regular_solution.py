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
    assert_close1d(GE.gammas_infinite_dilution(), [1.6479084374328064, 3.568103846536692, 1.5305660547296676, 43.74896036741677], rtol=1e-12)

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
    assert_close1d(GE.gammas_infinite_dilution(), [1.2749633329647372, 3.1371755931764684, 1.2495316747929026, 36.178962626768936], rtol=1e-12)

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
    assert_close1d(GE.gammas_infinite_dilution(), [1.1312355258340416, 1.41057767504653, 1.1196661045356717, 2.6379662312061507], rtol=1e-12)

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


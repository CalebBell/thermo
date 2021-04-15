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

from math import exp, log
import pytest
import numpy as np
from fluids.constants import calorie, R
from chemicals.rachford_rice import *
from thermo.mixture import Mixture
from thermo.uniquac import UNIQUAC
from random import random
from thermo import *
import numpy as np
from fluids.numerics import jacobian, hessian, derivative, normalize, assert_close, assert_close1d, assert_close2d, assert_close3d
from thermo.test_utils import check_np_output_activity
import pickle

def test_UNIQUAC_functional():
    # P05.01c VLE Behavior of Ethanol - Water Using UNIQUAC
    # http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/P05.01c%20VLE%20Behavior%20of%20Ethanol%20-%20Water%20Using%20UNIQUAC.xps

    gammas = UNIQUAC_gammas(xs=[0.252, 0.748], rs=[2.1055, 0.9200], qs=[1.972, 1.400], taus=[[1.0, 1.0919744384510301], [0.37452902779205477, 1.0]])
    assert_close1d(gammas, [2.35875137797083, 1.2442093415968987])

    # Example 8.3  in [2]_ for solubility of benzene (2) in ethanol (1) at 260 K.
    # Worked great here
    gammas = UNIQUAC_gammas(xs=[.7566, .2434], rs=[2.1055, 3.1878], qs=[1.972, 2.4], taus=[[1.0, 1.17984681869376], [0.22826016391070073, 1.0]])
    assert_close1d(gammas, [1.0826343452263132, 3.0176007269546083])

    # Example 7.3 in [2], for electrolytes
    gammas = UNIQUAC_gammas(xs=[0.05, 0.025, 0.925], rs=[1., 1., 0.92], qs=[1., 1., 1.4], taus=[[1.0, 0.4052558731309731, 2.7333668483468143], [21.816716876191823, 1.0, 0.06871094878791346], [0.4790878929721784, 3.3901086879605944, 1.0]])
    assert_close1d(gammas, [0.3838177662072466, 0.49469915162858774, 1.0204435746722416])


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
    assert_close1d(gammas, [1.0826343452263132, 3.0176007269546083])

    gammas = UNIQUAC_original_form(xs=[0.252, 0.748], rs=[2.1055, 0.9200], qs=[1.972, 1.400], taus=[[1.0, 1.0919744384510301], [0.37452902779205477, 1.0]])
    assert_close1d(gammas, [2.35875137797083, 1.2442093415968987])

    gammas = UNIQUAC_original_form(xs=[0.05, 0.025, 0.925], rs=[1., 1., 0.92], qs=[1., 1., 1.4], taus=[[1.0, 0.4052558731309731, 2.7333668483468143], [21.816716876191823, 1.0, 0.06871094878791346], [0.4790878929721784, 3.3901086879605944, 1.0]])
    assert_close1d(gammas, [0.3838177662072466, 0.49469915162858774, 1.0204435746722416])


def make_rsqs(N):
    cmps = range(N)
    rs = [float('%.3g'%(random()*2.5)) for _ in cmps]
    qs = [float('%.3g'%(random()*1.3)) for _ in cmps]
    return rs, qs

def make_taus(N):
    cmps = range(N)
    data = []
    base = [1e-4, 200.0, -5e-4, -7e-5, 300, 9e-8]

    for i in cmps:
        row = []
        for j in cmps:
            if i == j:
                row.append([0.0]*6)
            else:
                row.append([float('%.3g'%(random()*n)) for n in base])
        data.append(row)
    return data

def test_madeup_20():
    N = 20
    rs, qs = make_rsqs(N)
    taus = make_taus(N)
    xs = normalize([random() for i in range(N)])
    T = 350.0
    GE = UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, tau_coeffs=taus)

def test_UNIQUAC_madeup_ternary():
    N = 3
    T = 331.42
    xs = [0.229, 0.175, 0.596]
    rs = [2.5735, 2.87, 1.4311]
    qs = [2.336, 2.41, 1.432]

    # madeup numbers to match Wilson example roughly
    tausA = [[0.0, -1.05e-4, -2.5e-4], [3.9e-4, 0.0, 1.6e-4], [-1.123e-4, 6.5e-4, 0]]
    tausB = [[0.0, 235.0, -169.0], [-160, 0.0, -715.0], [11.2, 144.0, 0.0]]
    tausC = [[0.0, -4.23e-4, 2.9e-4], [6.1e-4, 0.0, 8.2e-5], [-7.8e-4, 1.11e-4, 0]]
    tausD = [[0.0, -3.94e-5, 2.22e-5], [8.5e-5, 0.0, 4.4e-5], [-7.9e-5, 3.22e-5, 0]]
    tausE = [[0.0, -4.2e2, 8.32e2], [2.7e2, 0.0, 6.8e2], [3.7e2, 7.43e2, 0]]
    tausF = [[0.0, 9.64e-8, 8.94e-8], [1.53e-7, 0.0, 1.11e-7], [7.9e-8, 2.276e-8, 0]]
    ABCDEF = (tausA, tausB, tausC, tausD, tausE, tausF)
    GE = UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, ABCDEF=ABCDEF)
    assert eval(str(GE)).GE() == GE.GE()

    GE2 = UNIQUAC.from_json(GE.as_json())
    assert GE2.__dict__ == GE.__dict__

    # GE
    GE_expect = 415.5805110962149
    GE_analytical = GE.GE()
    assert_close(GE_expect, GE_analytical, rtol=1e-13)
    gammas = UNIQUAC_gammas(taus=GE.taus(), rs=rs, qs=qs, xs=xs)
    GE_identity = R*T*sum(xi*log(gamma) for xi, gamma in zip(xs, gammas))
    assert_close(GE_identity, GE_analytical, rtol=1e-12)

    # dGE_dT
    dGE_dT_expect = 0.9907140284750982
    dGE_dT_analytical = GE.dGE_dT()
    dGE_dT_numerical = derivative(lambda T: GE.to_T_xs(T, xs).GE(), T, order=7, dx=T*1e-3)
    assert_close(dGE_dT_analytical, dGE_dT_numerical, rtol=1e-12)
    assert_close(dGE_dT_expect, dGE_dT_analytical, rtol=1e-13)

    # d2GE_dT2
    d2GE_dT2_expect = -0.007148011229475758
    d2GE_dT2_analytical = GE.d2GE_dT2()
    d2GE_dT2_numerical = derivative(lambda T: GE.to_T_xs(T, xs).dGE_dT(), T, order=7, dx=T*1e-3)
    assert_close(d2GE_dT2_expect, d2GE_dT2_analytical, rtol=1e-12)
    assert_close(d2GE_dT2_analytical, d2GE_dT2_numerical, rtol=1e-12)

    # d3GE_dT3
    d3GE_dT3_expect = 2.4882477326368877e-05
    d3GE_dT3_analytical = GE.d3GE_dT3()
    assert_close(d3GE_dT3_expect, d3GE_dT3_analytical, rtol=1e-13)
    d3GE_dT3_numerical = derivative(lambda T: GE.to_T_xs(T, xs).d2GE_dT2(), T, order=11, dx=T*1e-2)
    assert_close(d3GE_dT3_analytical, d3GE_dT3_numerical, rtol=1e-12)

    # dphis_dxs
    dphis_dxs_analytical = GE.dphis_dxs()
    dphis_dxs_expect = [[0.9223577846000854, -0.4473196931643269, -0.2230519905531248],
     [-0.3418381934661886, 1.094722540086528, -0.19009311780433752],
     [-0.5805195911338968, -0.6474028469222008, 0.41314510835746243]]
    assert_close2d(dphis_dxs_expect, dphis_dxs_analytical, rtol=1e-12)
    dphis_dxs_numerical = jacobian(lambda xs: GE.to_T_xs(T, xs).phis(), xs, scalar=False, perturbation=2e-8)
    assert_close2d(dphis_dxs_numerical, dphis_dxs_analytical, rtol=3e-8)

    # d2phis_dxixjs - checked to the last decimal with sympy
    d2phis_dxixjs_expect = [[[-2.441416183656415, 0.9048216556030662, 1.536594528053349],
      [-0.7693373390462084, -0.9442924629794809, 1.7136298020256895],
      [-0.3836232285397313, 0.5031631130108988, -0.11953988447116741]],
     [[-0.7693373390462084, -0.9442924629794809, 1.7136298020256895],
      [1.3204383950972896, -3.231500191022578, 1.9110617959252876],
      [0.658424873597119, -0.5251124708645561, -0.13331240273256284]],
     [[-0.3836232285397313, 0.5031631130108987, -0.11953988447116741],
      [0.6584248735971189, -0.5251124708645561, -0.13331240273256284],
      [0.32831771310273056, 0.27980444182238084, -0.6081221549251116]]]

    d2phis_dxixjs_analytical = GE.d2phis_dxixjs()
    assert_close3d(d2phis_dxixjs_analytical, d2phis_dxixjs_expect, rtol=1e-12)
    d2phis_dxixjs_numerical = hessian(lambda xs: GE.to_T_xs(T, xs).phis(), xs, scalar=False, perturbation=1e-5)
    assert_close3d(d2phis_dxixjs_numerical, d2phis_dxixjs_analytical, rtol=8e-5)


    d2thetas_dxixjs_expect = [[[-2.346422740416712, 0.7760247163009644, 1.5703980241157476],
      [-0.7026345706138027, -0.9175106511836936, 1.6201452217974965],
      [-0.4174990477672056, 0.47571378156805694, -0.05821473380085118]],
     [[-0.7026345706138027, -0.9175106511836936, 1.6201452217974965],
      [1.0476523499983839, -2.7191206652946023, 1.6714683152962189],
      [0.6225054627376287, -0.5624465978146614, -0.06005886492296719]],
     [[-0.4174990477672056, 0.47571378156805694, -0.05821473380085118],
      [0.6225054627376287, -0.5624465978146614, -0.06005886492296719],
      [0.3698870633362176, 0.2916190647283637, -0.6615061280645813]]]
    d2thetas_dxixjs_analytical = GE.d2thetas_dxixjs()
    assert_close3d(d2thetas_dxixjs_analytical, d2thetas_dxixjs_expect, rtol=1e-12)
    d2thetas_dxixjs_numerical = hessian(lambda xs: GE.to_T_xs(T, xs).thetas(), xs, scalar=False, perturbation=2e-5)
    assert_close3d(d2thetas_dxixjs_numerical, d2thetas_dxixjs_analytical, rtol=1e-4)

    def to_jac(xs):
        return GE.to_T_xs(T, xs).GE()

    # Obtained 12 decimals of precision with numdifftools
    dGE_dxs_analytical = GE.dGE_dxs()
    dGE_dxs_expect = [-2651.3181821109024, -2085.574403592012, -2295.0860830203587]
    assert_close1d(dGE_dxs_analytical, dGE_dxs_expect, rtol=1e-12)
    dGE_dxs_numerical = jacobian(to_jac, xs, perturbation=1e-8)
    assert_close1d(dGE_dxs_numerical, dGE_dxs_analytical, rtol=1e-6)

    # d2GE_dTdxs
    def to_jac(xs):
        return GE.to_T_xs(T, xs).dGE_dT()
    d2GE_dTdxs_expect = [-9.940433543371945, -3.545963210296949, -7.427593534302016]
    d2GE_dTdxs = GE.d2GE_dTdxs()
    d2GE_dTdxs_numerical = jacobian(to_jac, xs, perturbation=1e-8)
    assert_close1d(d2GE_dTdxs_numerical, d2GE_dTdxs, rtol=1e-6)
    assert_close1d(d2GE_dTdxs, d2GE_dTdxs_expect, rtol=1e-12)

    # d2GE_dxixjs

    def to_hess(xs):
        return GE.to_T_xs(T, xs).GE()

    d2GE_dxixjs_numerical = hessian(to_hess, xs, perturbation=1e-4)
    d2GE_dxixjs_sympy = [[-2890.4327598108343, -6687.099054095988, -1549.3754436994557],
     [-6687.099054095988, -2811.283290487096, -1228.622385377738],
     [-1549.3754436994557, -1228.622385377738, -3667.3880987585053]]
    d2GE_dxixjs_analytical = GE.d2GE_dxixjs()
    assert_close2d(d2GE_dxixjs_numerical, d2GE_dxixjs_analytical, rtol=1e-4)
    assert_close2d(d2GE_dxixjs_analytical, d2GE_dxixjs_sympy, rtol=1e-12)

    # Check json storage again, with some results
    GE2 = UNIQUAC.from_json(GE.as_json())
    assert GE2.__dict__ == GE.__dict__

def test_UNIQUAC_require_parameters():
    xs = [0.7273, 0.0909, 0.1818]
    rs = [.92, 2.1055, 3.1878]
    qs = [1.4, 1.972, 2.4]
    with pytest.raises(ValueError):
        GE = UNIQUAC(T=300.0, xs=xs, rs=rs, qs=qs)

def test_UNIQUAC_numpy_inputs():


    N = 3
    T = 331.42
    xs = [0.229, 0.175, 0.596]
    rs = [2.5735, 2.87, 1.4311]
    qs = [2.336, 2.41, 1.432]
    # madeup numbers to match Wilson example roughly
    tausA = [[0.0, -1.05e-4, -2.5e-4], [3.9e-4, 0.0, 1.6e-4], [-1.123e-4, 6.5e-4, 0]]
    tausB = [[0.0, 235.0, -169.0], [-160, 0.0, -715.0], [11.2, 144.0, 0.0]]
    tausC = [[0.0, -4.23e-4, 2.9e-4], [6.1e-4, 0.0, 8.2e-5], [-7.8e-4, 1.11e-4, 0]]
    tausD = [[0.0, -3.94e-5, 2.22e-5], [8.5e-5, 0.0, 4.4e-5], [-7.9e-5, 3.22e-5, 0]]
    tausE = [[0.0, -4.2e2, 8.32e2], [2.7e2, 0.0, 6.8e2], [3.7e2, 7.43e2, 0]]
    tausF = [[0.0, 9.64e-8, 8.94e-8], [1.53e-7, 0.0, 1.11e-7], [7.9e-8, 2.276e-8, 0]]
    ABCDEF = (tausA, tausB, tausC, tausD, tausE, tausF)
    ABCDEFnp = tuple(np.array(v) for v in ABCDEF)

    model = UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, ABCDEF=ABCDEF)
    modelnp = UNIQUAC(T=T, xs=np.array(xs), rs=np.array(rs), qs=np.array(qs), ABCDEF=ABCDEFnp)
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)

    json_string = modelnp.as_json()
    new = UNIQUAC.from_json(json_string)
    assert new == modelnp

    # Pickle checks
    modelnp_pickle = pickle.loads(pickle.dumps(modelnp))
    assert modelnp_pickle == modelnp
    model_pickle = pickle.loads(pickle.dumps(model))
    assert model_pickle == model

def test_UNIQUAC_np_hash_different_input_forms():
    N = 3
    T = 331.42
    xs = [0.229, 0.175, 0.596]
    rs = [2.5735, 2.87, 1.4311]
    qs = [2.336, 2.41, 1.432]
    # madeup numbers to match Wilson example roughly
    tausA = [[0.0, -1.05e-4, -2.5e-4], [3.9e-4, 0.0, 1.6e-4], [-1.123e-4, 6.5e-4, 0]]
    tausB = [[0.0, 235.0, -169.0], [-160, 0.0, -715.0], [11.2, 144.0, 0.0]]
    tausC = [[0.0, -4.23e-4, 2.9e-4], [6.1e-4, 0.0, 8.2e-5], [-7.8e-4, 1.11e-4, 0]]
    tausD = [[0.0, -3.94e-5, 2.22e-5], [8.5e-5, 0.0, 4.4e-5], [-7.9e-5, 3.22e-5, 0]]
    tausE = [[0.0, -4.2e2, 8.32e2], [2.7e2, 0.0, 6.8e2], [3.7e2, 7.43e2, 0]]
    tausF = [[0.0, 9.64e-8, 8.94e-8], [1.53e-7, 0.0, 1.11e-7], [7.9e-8, 2.276e-8, 0]]
    ABCDEF = (tausA, tausB, tausC, tausD, tausE, tausF)
    A, B, C, D, E, F = ABCDEF

    tau_coeffs = [[[A[i][j], B[i][j], C[i][j], D[i][j], E[i][j], F[i][j]] for j in range(N)] for i in range(N)]

    model_ABCDEF = UNIQUAC(T=T, xs=xs,  rs=rs, qs=qs, ABCDEF=(A, B, C, D, E, F))
    model_tau_coeffs = UNIQUAC(T=T, xs=xs,  rs=rs, qs=qs, tau_coeffs=tau_coeffs)
    model_tau_coeffs2 = UNIQUAC.from_json(model_tau_coeffs.as_json())

    assert model_ABCDEF.GE() == model_tau_coeffs.GE()
    assert hash(model_ABCDEF) == hash(model_tau_coeffs)
    assert model_ABCDEF.GE() == model_tau_coeffs2.GE()
    assert hash(model_ABCDEF) == hash(model_tau_coeffs2)


    modelnp = UNIQUAC(T=T, xs=np.array(xs),rs=np.array(rs), qs=np.array(qs), ABCDEF=(np.array(A), np.array(B), np.array(C), np.array(D), np.array(E), np.array(F)))
    model_tau_coeffsnp = UNIQUAC(T=T,rs=np.array(rs), qs=np.array(qs), xs=np.array(xs), tau_coeffs=np.array(tau_coeffs))
    assert modelnp.GE() == model_tau_coeffsnp.GE()
    assert hash(modelnp) == hash(model_tau_coeffsnp)
    model_tau_coeffsnp2 = UNIQUAC.from_json(model_tau_coeffsnp.as_json())
    assert hash(modelnp) == hash(model_tau_coeffsnp2)

def test_Uniquac_numpy_output_correct_array_internal_ownership():
    N = 3
    T = 331.42
    xs = [0.229, 0.175, 0.596]
    rs = [2.5735, 2.87, 1.4311]
    qs = [2.336, 2.41, 1.432]
    # madeup numbers to match Wilson example roughly
    tausA = [[0.0, -1.05e-4, -2.5e-4], [3.9e-4, 0.0, 1.6e-4], [-1.123e-4, 6.5e-4, 0]]
    tausB = [[0.0, 235.0, -169.0], [-160, 0.0, -715.0], [11.2, 144.0, 0.0]]
    tausC = [[0.0, -4.23e-4, 2.9e-4], [6.1e-4, 0.0, 8.2e-5], [-7.8e-4, 1.11e-4, 0]]
    tausD = [[0.0, -3.94e-5, 2.22e-5], [8.5e-5, 0.0, 4.4e-5], [-7.9e-5, 3.22e-5, 0]]
    tausE = [[0.0, -4.2e2, 8.32e2], [2.7e2, 0.0, 6.8e2], [3.7e2, 7.43e2, 0]]
    tausF = [[0.0, 9.64e-8, 8.94e-8], [1.53e-7, 0.0, 1.11e-7], [7.9e-8, 2.276e-8, 0]]
    ABCDEF = (tausA, tausB, tausC, tausD, tausE, tausF)
    A, B, C, D, E, F = ABCDEF

    tau_coeffs = [[[A[i][j], B[i][j], C[i][j], D[i][j], E[i][j], F[i][j]] for j in range(N)] for i in range(N)]
    modelnp = UNIQUAC(T=T,rs=np.array(rs), qs=np.array(qs), xs=np.array(xs), tau_coeffs=np.array(tau_coeffs))
    for name in ('tau_coeffs_A', 'tau_coeffs_B', 'tau_coeffs_C',
                 'tau_coeffs_D', 'tau_coeffs_E', 'tau_coeffs_F'):
        obj = getattr(modelnp, name)
        assert obj.flags.c_contiguous
        assert obj.flags.owndata
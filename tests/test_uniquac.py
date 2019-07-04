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
from numpy.testing import assert_allclose
import pytest
import numpy as np
from thermo import normalize
from fluids.constants import calorie, R
from thermo.activity import *
from thermo.mixture import Mixture
from thermo.uniquac import UNIQUAC
from random import random
from thermo import *
import numpy as np
from fluids.numerics import jacobian, hessian, derivative

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
    
    # GE
    GE_expect = 415.5805110962149
    GE_analytical = GE.GE()
    assert_allclose(GE_expect, GE_analytical, rtol=1e-13)
    gammas = UNIQUAC_gammas(taus=GE.taus(), rs=rs, qs=qs, xs=xs)
    GE_identity = R*T*sum(xi*log(gamma) for xi, gamma in zip(xs, gammas))
    assert_allclose(GE_identity, GE_analytical, rtol=1e-12)
    
    # dGE_dT
    dGE_dT_expect = 0.9907140284750982
    dGE_dT_analytical = GE.dGE_dT()
    dGE_dT_numerical = derivative(lambda T: GE.to_T_xs(T, xs).GE(), T, order=7, dx=T*1e-3)
    assert_allclose(dGE_dT_analytical, dGE_dT_numerical, rtol=1e-12)
    assert_allclose(dGE_dT_expect, dGE_dT_analytical, rtol=1e-13)
    
    # d2GE_dT2
    d2GE_dT2_expect = -0.007148011229475758
    d2GE_dT2_analytical = GE.d2GE_dT2()
    d2GE_dT2_numerical = derivative(lambda T: GE.to_T_xs(T, xs).dGE_dT(), T, order=7, dx=T*1e-3)
    assert_allclose(d2GE_dT2_expect, d2GE_dT2_analytical, rtol=1e-12)
    assert_allclose(d2GE_dT2_analytical, d2GE_dT2_numerical, rtol=1e-12)
    
    # d3GE_dT3
    d3GE_dT3_expect = 2.4882477326368877e-05
    d3GE_dT3_analytical = GE.d3GE_dT3()
    assert_allclose(d3GE_dT3_expect, d3GE_dT3_analytical, rtol=1e-13)
    d3GE_dT3_numerical = derivative(lambda T: GE.to_T_xs(T, xs).d2GE_dT2(), T, order=11, dx=T*1e-2)
    assert_allclose(d3GE_dT3_analytical, d3GE_dT3_numerical, rtol=1e-12)
    
    # dphis_dxs
    dphis_dxs_analytical = GE.dphis_dxs()
    dphis_dxs_expect = [[0.9223577846000854, -0.4473196931643269, -0.2230519905531248],
     [-0.3418381934661886, 1.094722540086528, -0.19009311780433752],
     [-0.5805195911338968, -0.6474028469222008, 0.41314510835746243]]
    assert_allclose(dphis_dxs_expect, dphis_dxs_analytical, rtol=1e-12)
    dphis_dxs_numerical = jacobian(lambda xs: GE.to_T_xs(T, xs).phis(), xs, scalar=False, perturbation=2e-8)
    assert_allclose(dphis_dxs_numerical, dphis_dxs_analytical, rtol=3e-8)
    
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
    assert_allclose(d2phis_dxixjs_analytical, d2phis_dxixjs_expect, rtol=1e-12)
    d2phis_dxixjs_numerical = hessian(lambda xs: GE.to_T_xs(T, xs).phis(), xs, scalar=False, perturbation=1e-5)
    assert_allclose(d2phis_dxixjs_numerical, d2phis_dxixjs_analytical, rtol=8e-5)
    

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
    assert_allclose(d2thetas_dxixjs_analytical, d2thetas_dxixjs_expect, rtol=1e-12)
    d2thetas_dxixjs_numerical = hessian(lambda xs: GE.to_T_xs(T, xs).thetas(), xs, scalar=False, perturbation=2e-5)
    assert_allclose(d2thetas_dxixjs_numerical, d2thetas_dxixjs_analytical, rtol=1e-4)
    
    def to_jac(xs):
        return GE.to_T_xs(T, xs).GE()
    
    # Obtained 12 decimals of precision with numdifftools
    dGE_dxs_analytical = GE.dGE_dxs()
    dGE_dxs_expect = [-2651.3181821109024, -2085.574403592012, -2295.0860830203587]
    assert_allclose(dGE_dxs_analytical, dGE_dxs_expect, rtol=1e-12)
    dGE_dxs_numerical = jacobian(to_jac, xs, perturbation=1e-8)
    assert_allclose(dGE_dxs_numerical, dGE_dxs_analytical, rtol=1e-6)

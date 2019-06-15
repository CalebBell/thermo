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
from thermo.nrtl import NRTL
import random
from thermo import *
import numpy as np
from fluids.numerics import jacobian, hessian, derivative


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
    GE = NRTL(T, xs, taus, alphas)
    
    ### Tau and derivatives
    taus_expected = [[0.06687993075720595, 1.9456413587531054, 1.2322559725492486],
     [-0.04186204696272491, 0.07047352903742096, 0.007348860249786843],
     [-0.21212866478360642, 0.13596095401379812, 0.0944497207779701]]
    assert_allclose(taus_expected, GE.taus(), rtol=1e-12)
    
    # Tau temperature derivative
    dtaus_dT_numerical = (np.array(GE.to_T_xs(T+dT, xs).taus()) - GE.taus())/dT
    dtaus_dT_analytical = GE.dtaus_dT()
    dtaus_dT_expected = [[0.000317271602387579, -0.004653030923421638, -0.0029936361350323625],
     [0.000406561723402744, 0.00034844970187634483, 0.0009043271077256468],
     [0.0011749522864265571, -0.00013143064874333648, 0.0005280368036263511]]
    assert_allclose(dtaus_dT_analytical, dtaus_dT_expected, rtol=1e-12)
    assert_allclose(dtaus_dT_numerical, dtaus_dT_analytical, rtol=4e-8)
    
    # tau second derivative
    d2taus_dT2_analytical = GE.d2taus_dT2()
    d2taus_dT2_expected = [[7.194089397742681e-07, 3.2628265646047626e-05, 2.0866597625703075e-05],
     [-1.2443543228779117e-06, 8.394080699905074e-07, -1.169244972344292e-07],
     [-3.669244570181493e-06, 1.955896362917401e-06, 1.4794908652710361e-06]]
    assert_allclose(d2taus_dT2_analytical, d2taus_dT2_expected, rtol=1e-12)
    d2taus_dT2_numerical = (np.array(GE.to_T_xs(T+dT, xs).dtaus_dT()) - GE.dtaus_dT())/dT
    assert_allclose(d2taus_dT2_analytical, d2taus_dT2_numerical, rtol=4e-7)

    # tau third derivative
    d3taus_dT3_analytical = GE.d3taus_dT3()
    d3taus_dT3_expected = [[3.425034691577827e-12, -2.703935984244539e-07, -1.7263626338812435e-07],
     [1.2625331389834536e-08, 3.4351735085462344e-12, 1.535797829031479e-08],
     [4.116922701015044e-08, -1.4652079774338131e-08, 2.9650219270685846e-12]]
    
    # Not sure why precision of numerical test is so low, but confirmed with sympy.
    assert_allclose(d3taus_dT3_analytical, d3taus_dT3_expected, rtol=1e-12)
    d3taus_dT3_numerical = (np.array(GE.to_T_xs(T+dT, xs).d2taus_dT2()) - GE.d2taus_dT2())/dT
    assert_allclose(d3taus_dT3_numerical, d3taus_dT3_analytical, rtol=2e-5)
    
    # alphas
    alphas_expect = [[0.006863, 0.3177205, 0.334215],
     [0.2971315, 0.013726, 0.328352],
     [0.3033315, 0.3111945, 0.0171575]]
    assert_allclose(alphas_expect,  GE.alphas(), rtol=1e-12)
    
    # dalphas_dT
    dalphas_dT_expect = [[2e-05, 7e-05, 0.0001], [1e-05, 4e-05, 8e-05], [1e-05, 3e-05, 5e-05]]
    dalphas_dT_analytical = GE.dalphas_dT()
    assert_allclose(dalphas_dT_expect, dalphas_dT_analytical, rtol=1e-12)
    dalphas_dT_numerical = (np.array(GE.to_T_xs(T+1e-4, xs).alphas()) - GE.alphas())/1e-4
    assert_allclose(dalphas_dT_expect, dalphas_dT_numerical)
    
    # d2alphas_d2T
    d2alphas_d2T_numerical = (np.array(GE.to_T_xs(T+dT, xs).dalphas_dT()) - GE.dalphas_dT())/dT
    d2alphas_d2T_analytical = GE.d2alphas_dT2()
    assert_allclose(d2alphas_d2T_analytical, [[0]*N for _ in range(N)])
    assert_allclose(d2alphas_d2T_numerical, d2alphas_d2T_analytical, rtol=1e-12)
    
    # d3alphas_d3T
    d3alphas_d3T_numerical = (np.array(GE.to_T_xs(T+dT, xs).d2alphas_dT2()) - GE.d2alphas_dT2())/dT
    d3alphas_d3T_analytical = GE.d3alphas_dT3()
    assert_allclose(d3alphas_d3T_analytical, [[0]*N for _ in range(N)])
    assert_allclose(d3alphas_d3T_numerical, d3alphas_d3T_analytical, rtol=1e-12)

    # Gs
    Gs_expect = [[0.9995411083582052, 0.5389296989069797, 0.6624312965198783],
     [1.0125162130984628, 0.999033148043276, 0.9975898960147673],
     [1.0664605905163351, 0.9585722883795924, 0.9983807912510595]]
    assert_allclose(Gs_expect, GE.Gs())
    
    # dGs_dT
    dGs_dT_expect = [[-3.513420602780159e-06, 0.0007233344205287423, 0.000581146010596799],
     [-0.00012189042196807427, -7.594411991210204e-06, -0.00029680845584651057],
     [-0.000377824327942326, 3.529622902294454e-05, -1.3759961112813966e-05]]
    dGs_dT_analytical = GE.dGs_dT()
    assert_allclose(dGs_dT_expect, dGs_dT_analytical, rtol=1e-12)
    dGs_dT_numerical = (np.array(GE.to_T_xs(T+ 2.5e-5, xs).Gs()) - GE.Gs())/2.5e-5
    assert_allclose(dGs_dT_numerical, dGs_dT_expect, rtol=3e-7) # Closest I could get
    
    # d2Gs_dT2
    d2Gs_dT2_expect = [[-1.7607728438831442e-08, -4.264997204215131e-06, -3.7132987505208185e-06],
     [3.808051820202208e-07, -3.9301868673119065e-08, -1.773565965539868e-08],
     [1.2957622532360591e-06, -5.745898135094948e-07, -7.78717985147786e-08]]
    d2Gs_dT2_analytical = GE.d2Gs_dT2()
    assert_allclose(d2Gs_dT2_expect, d2Gs_dT2_analytical, rtol=1e-12)
    d2Gs_dT2_numerical = (np.array(GE.to_T_xs(T+4e-6, xs).dGs_dT()) - GE.dGs_dT())/4e-6
    assert_allclose(d2Gs_dT2_numerical, d2Gs_dT2_analytical, rtol=2e-6)  

    # d3Gs_dT3
    d3Gs_dT3_analytical = GE.d3Gs_dT3()
    d3Gs_dT3_expect = [[-4.298246167557067e-11, 2.282743128709054e-08, 2.3406412447900822e-08],
     [-3.894534156411875e-09, -9.978151596051988e-11, -4.934296656436311e-09],
     [-1.448282405010784e-08, 4.1384450135276364e-09, -2.1839009944794027e-10]]
    d3Gs_dT3_numerical = (np.array(GE.to_T_xs(T+dT, xs).d2Gs_dT2()) - GE.d2Gs_dT2())/dT
    assert_allclose(d3Gs_dT3_analytical, d3Gs_dT3_numerical, rtol=1e-7)
    assert_allclose(d3Gs_dT3_expect, d3Gs_dT3_analytical, rtol=1e-12)
    
    # dGE_dT
    dGE_dT_expect = 2.258093406765717
    dGE_dT_analytical = GE.dGE_dT()
    assert_allclose(dGE_dT_expect, dGE_dT_analytical, rtol=1e-12)
    
    def to_diff(T):
        return GE.to_T_xs(T, xs).GE()
    dGE_dT_numerical = derivative(to_diff, T, dx=1, order=17)
    assert_allclose(dGE_dT_analytical, dGE_dT_numerical, rtol=1e-12)
    
    # d2GE_dT2
    def to_diff(T):
        return GE.to_T_xs(T, xs).dGE_dT()
    
    d2GE_dT2_numerical = derivative(to_diff, T, dx=5, order=17)
    d2GE_dT2_expected = 0.007412479461681191
    d2GE_dT2_analytical = GE.d2GE_dT2()
    assert_allclose(d2GE_dT2_expected, d2GE_dT2_analytical, rtol=1e-12)
    assert_allclose(d2GE_dT2_numerical, d2GE_dT2_analytical, rtol=1e-12)
    
    # dGE_dxs
    def to_jac(xs):
        return GE.to_T_xs(T, xs).GE()
    
    dGE_dxs_analytical = GE.dGE_dxs()
    dGE_dxs_expected = [1644.4832445701338, 397.6635234141318, 202.8071951151063]
    assert_allclose(dGE_dxs_analytical, dGE_dxs_expected, rtol=1e-12)
    dGE_dxs_numerical = jacobian(to_jac, xs, perturbation=1e-8)
    assert_allclose(dGE_dxs_numerical, dGE_dxs_expected, rtol=1e-7)
    
    
    
    # d2GE_dxixjs - more decimals in numdifftools
    def to_hess(xs):
        return GE.to_T_xs(T, xs).GE()
    
    d2GE_dxixjs_expect = [[-1690.7876619180952, 1208.6126803238276, -48.852543427058286],
     [1208.6126803238276, -468.99032836115686, -202.0508751128366],
     [-48.852543427058606, -202.0508751128365, 140.77154243852513]]
    d2GE_dxixjs_analytical = GE.d2GE_dxixjs()
    assert_allclose(d2GE_dxixjs_expect, d2GE_dxixjs_analytical, rtol=1e-12)
    d2GE_dxixjs_numerical = hessian(to_hess, xs, perturbation=5e-5)
    assert_allclose(d2GE_dxixjs_numerical, d2GE_dxixjs_analytical, rtol=3e-4)
    
    
    # d2GE_dTdxs - matched really well
    d2GE_dTdxs_expect = [3.053720836458414, 1.8759446742883084, 2.1691316743750826]
    d2GE_dTdxs_analytical = GE.d2GE_dTdxs()
    assert_allclose(d2GE_dTdxs_expect, d2GE_dTdxs_analytical, rtol=1e-12)
    
    def to_jac(xs):
        return GE.to_T_xs(T, xs).dGE_dT()
    
    d2GE_dTdxs_numerical = jacobian(to_jac, xs, perturbation=3e-8)
    assert_allclose(d2GE_dTdxs_analytical, d2GE_dTdxs_numerical, rtol=1e-8)
# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
"""

from __future__ import division
from thermo import *
import thermo
from math import *
from random import random
from fluids.constants import *
from fluids.numerics import assert_close, assert_close1d, assert_close2d, assert_close3d
from numpy.testing import assert_allclose
from chemicals import normalize
from thermo.test_utils import check_np_output_activity

import pytest
try:
    import numba
    import thermo.numba
    import numba.core
except:
    numba = None
import numpy as np


def swap_funcs_and_test(names, substitutions, test):
    '''
    names : list[str]
        object names to switch out
    substitutions : list[obj]
        Objects to put in
    test : function
        Unit test to run in the file
    '''
    originals = {}
    glob = test.__globals__
    for name, sub in zip(names, substitutions):
        originals[name] = glob[name]
        glob[name] = sub
    try:
        test()
    except Exception as e:
        glob.update(originals)
        raise e
    glob.update(originals)

def mark_as_numba(func):
    func = pytest.mark.numba(func)
    func = pytest.mark.skipif(numba is None, reason="Numba is missing")(func)
    return func

@mark_as_numba
def test_PRMIX_outputs_inputs_np():
    kwargs = dict(Tcs=[190.56400000000002, 305.32, 369.83, 126.2],
                  Pcs=[4599000.0, 4872000.0, 4248000.0, 3394387.5],
                  omegas=[0.008, 0.098, 0.152, 0.04],
                  zs=[.1, .2, .3, .4],
                  kijs=[[0.0, -0.0059, 0.0119, 0.0289], [-0.0059, 0.0, 0.0011, 0.0533], [0.0119, 0.0011, 0.0, 0.0878], [0.0289, 0.0533, 0.0878, 0.0]])
    kwargs_np = {k:np.array(v) for k, v in kwargs.items()}

    from thermo.numba import PRMIX as PRMIXNP

    eos = PRMIX(T=200, P=1e5, **kwargs)
    eos_np = PRMIXNP(T=200, P=1e5, **kwargs_np)

    base_vec_attrs = ['a_alphas', 'da_alpha_dTs', 'd2a_alpha_dT2s', 'a_alpha_roots', 'a_alpha_j_rows', 'da_alpha_dT_j_rows', 'lnphis_l', 'phis_l', 'fugacities_l', 'lnphis_g', 'phis_g', 'fugacities_g']
    extra_vec_attrs = ['db_dzs', 'db_dns', 'dnb_dns', 'd2b_dzizjs', 'd2b_dninjs', 'd3b_dzizjzks', 'd3b_dninjnks', 'd3epsilon_dzizjzks', 'da_alpha_dzs', 'da_alpha_dns', 'dna_alpha_dns', 'd2a_alpha_dzizjs']
    alpha_vec_attrs = ['_a_alpha_j_rows', '_da_alpha_dT_j_rows', 'a_alpha_ijs', 'da_alpha_dT_ijs', 'd2a_alpha_dT2_ijs']
    # TODO: _d2a_alpha_dT2_j_rows, and _a_alpha_j_rows', '_da_alpha_dT_j_rows with .to methods

    for attr in base_vec_attrs + extra_vec_attrs + alpha_vec_attrs:
        assert_close1d(getattr(eos, attr), getattr(eos_np, attr), rtol=1e-14)
        assert type(getattr(eos, attr)) is list
        assert type(getattr(eos_np, attr)) is np.ndarray


@mark_as_numba
def test_IdealSolution_np_out():
    from thermo import IdealSolution
    from thermo.numba import IdealSolution as IdealSolutionnp
    model = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
    modelnp = IdealSolutionnp(T=300.0, xs=np.array([.1, .2, .3, .4]))
    modelnp2 = modelnp.to_T_xs(T=310.0, xs=np.array([.2, .2, .2, .4]))

    check_np_output_activity(model, modelnp, modelnp2)


@mark_as_numba
def test_Wilson_numpy_output():
    T = 331.42
    N = 3

    from thermo.numba import Wilson as Wilsonnp
    A = [[0.0, 3.870101271243586, 0.07939943395502425],
                 [-6.491263271243587, 0.0, -3.276991837288562],
                 [0.8542855660449756, 6.906801837288562, 0.0]]
    B = [[0.0, -375.2835, -31.1208],
                 [1722.58, 0.0, 1140.79],
                 [-747.217, -3596.17, -0.0]]
    D = [[-0.0, -0.00791073, -0.000868371],
                 [0.00747788, -0.0, -3.1e-05],
                 [0.00124796, -3e-05, -0.0]]

    C = E = F = [[0.0]*N for _ in range(N)]

    xs = [0.229, 0.175, 0.596]

    model = thermo.wilson.Wilson(T=T, xs=xs, ABCDEF=(A, B, C, D, E, F))
    modelnp = Wilsonnp(T=T, xs=np.array(xs), ABCDEF=(np.array(A), np.array(B), np.array(C), np.array(D), np.array(E), np.array(F)))
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)

@mark_as_numba
def test_NRTL_numpy_output():
    NRTLnp = thermo.numba.nrtl.NRTL
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
    model = NRTL(T, xs, taus, alphas)
    modelnp = NRTLnp(T=T, xs=np.array(xs), tau_coeffs=np.array(taus), alpha_coeffs=np.array(alphas))
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)

@mark_as_numba
def test_UNIQUAC_numpy_output():
    UNIQUACnp = thermo.numba.uniquac.UNIQUAC

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
    modelnp = UNIQUACnp(T=T, xs=np.array(xs), rs=np.array(rs), qs=np.array(qs), ABCDEF=ABCDEFnp)
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)

@mark_as_numba
def test_UNIFAC_numpy_output():
    from thermo.unifac import DOUFIP2006, DOUFSG

    UNIFACnp = thermo.numba.unifac.UNIFAC
    N = 4
    T = 373.15
    xs = [0.2, 0.3, 0.1, 0.4]
    chemgroups = [{9:6}, {78:6}, {1:1, 18:1}, {1:1, 2:1, 14:1}]
    model = thermo.unifac.UNIFAC.from_subgroups(T=T, xs=xs, chemgroups=chemgroups, version=1,
                               interaction_data=DOUFIP2006, subgroups=DOUFSG)


    modelnp = UNIFACnp.from_subgroups(T=T, xs=np.array(xs), chemgroups=chemgroups, version=1,
                           interaction_data=DOUFIP2006, subgroups=DOUFSG)
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)

    json_string = modelnp.as_json()
    new = UNIFACnp.from_json(json_string)
    assert new == modelnp




@mark_as_numba
def test_a_alpha_aijs_composition_independent_in_all():
    assert 'a_alpha_aijs_composition_independent' in thermo.numba.__all__


@mark_as_numba
def test_a_alpha_aijs_composition_independent():
    # TODO: a_alpha_aijs_composition_independent is being overwritten in thermo.numba somehow!

    kijs = np.array([[0,.083],[0.083,0]])
    a_alphas = np.array([0.2491099357671155, 0.6486495863528039])
    a0, a1, a2 = thermo.numba.eos_mix_methods.a_alpha_aijs_composition_independent(a_alphas, kijs)
    assert type(a0) is np.ndarray
    assert type(a1) is np.ndarray
    assert type(a2) is np.ndarray

    b0, b1, b2 = thermo.eos_mix_methods.a_alpha_aijs_composition_independent(a_alphas, kijs)
    assert_close1d(a1, b1, rtol=1e-13)
    assert_close2d(a0, b0, rtol=1e-13)
    assert_close2d(a2, b2, rtol=1e-13)

    assert thermo.numba.eos_mix_methods.a_alpha_aijs_composition_independent is not thermo.eos_mix_methods.a_alpha_aijs_composition_independent



@mark_as_numba
def test_a_alpha_and_derivatives_full():
    kijs = np.array([[0,.083],[0.083,0]])
    zs = np.array([0.1164203, 0.8835797])
    a_alphas = np.array([0.2491099357671155, 0.6486495863528039])
    da_alpha_dTs = np.array([-0.0005102028006086241, -0.0011131153520304886])
    d2a_alpha_dT2s = np.array([1.8651128859234162e-06, 3.884331923127011e-06])
    a_alpha, da_alpha_dT, d2a_alpha_dT2, a_alpha_ijs, da_alpha_dT_ijs, d2a_alpha_dT2_ijs = thermo.numba.a_alpha_and_derivatives_full(a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs, d2a_alpha_dT2s=d2a_alpha_dT2s, T=299.0, zs=zs, kijs=kijs)

    a_alpha0, da_alpha_dT0, d2a_alpha_dT20, a_alpha_ijs0, da_alpha_dT_ijs0, d2a_alpha_dT2_ijs0 = thermo.eos_mix_methods.a_alpha_and_derivatives_full(a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs, d2a_alpha_dT2s=d2a_alpha_dT2s, T=299.0, zs=zs, kijs=kijs)


    assert_close(a_alpha, a_alpha0, rtol=1e-13)
    assert_close(da_alpha_dT, da_alpha_dT0, rtol=1e-13)
    assert_close(d2a_alpha_dT2, d2a_alpha_dT20, rtol=1e-13)

    assert_close1d(a_alpha_ijs, a_alpha_ijs0, rtol=1e-13)
    assert_close1d(da_alpha_dT_ijs, da_alpha_dT_ijs0, rtol=1e-13)
    assert_close1d(d2a_alpha_dT2_ijs0, d2a_alpha_dT2_ijs, rtol=1e-13)


@mark_as_numba
def test_IAPWS95_numba():
    assert isinstance(thermo.numba.flash.iapws95_Psat, numba.core.registry.CPUDispatcher)
    assert isinstance(thermo.numba.phases.IAPWS95._d3Ar_ddeltadtau2_func, numba.core.registry.CPUDispatcher)

    from thermo.numba import IAPWS95, IAPWS95Liquid, IAPWS95Gas, FlashPureVLS

    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])

    assert_close(flasher.flash(T=1000,P=1e4).H(), 71901.67235666412, rtol=1e-8) # TP
    assert_close(flasher.flash(P=1e5, V=.1).T, 1202.8504507662728, rtol=1e-8) # PV
    assert_close(flasher.flash(T=1000, V=.1).P, 83126.1092778793, rtol=1e-8) # TV

    assert_close(flasher.flash(P=1e5,VF=1).T, 372.7559288971221, rtol=1e-8) # PVF
    assert_close(flasher.flash(T=300, VF=.5).P, 3536.806752274638, rtol=1e-8) # TVF
# assert_close(flasher.flash(P=1e4, H=71901.67235666412).T, 1000, rtol=1e-8) # PH - not working yet



@mark_as_numba
def test_RegularSolution_numba():
    N = 20
    xs = normalize([random() for _ in range(N)])
    xs2 = normalize([random() for _ in range(N)])
    SPs = [50000.0*random() for _ in range(N)]
    Vs = [1e-5*random() for _ in range(N)]


    T = 300.0
    lambda_coeffs = [[random()*1e-4 for _ in range(N)] for _ in range(N)]

    GE = RegularSolution(T, xs, Vs, SPs, lambda_coeffs)
    xsnp = np.array(xs)
    xs2np = np.array(xs2)
    Vsnp = np.array(Vs)
    SPsnp = np.array(SPs)
    lambda_coeffsnp = np.array(lambda_coeffs)

    GE = RegularSolution(T, xsnp, Vsnp, SPsnp, lambda_coeffsnp)
    GE.gammas()
    GE.to_T_xs(T=T+1.0, xs=xs2np).gammas()

    assert_close1d(GE.d2GE_dTdxs(), [0.0]*N, atol=0)

@mark_as_numba
def test_volume_numba_solvers():

    args = (0.0001, 0.01, 2.590839755349289e-05, 2.590839755349289e-05, 0.0, 348530.6151663297)
    slns = thermo.numba.eos_volume.volume_solutions_halley(*args)
    assert slns[1] == 0
    assert slns[2] == 0
    assert_close(slns[0], 2.5908397553496088e-05, rtol=1e-15)

    args = (0.0001, 154141458.17537114, 2.590839755349289e-05, 2.590839755349289e-05, 0.0, 348530.6151663297)
    slns = thermo.numba.eos_volume.volume_solutions_halley(*args)
    assert slns[1] == 0
    assert slns[2] == 0
    assert_close(slns[0], 2.5908397553496098e-05, rtol=1e-15)
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

import pytest
try:
    import numba
    import thermo.numba
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


def check_numba_np_output_activity(model, modelnp, modelnp2):
    # model is flat, scalar, list-based model
    # modelnp is numba model
    # modelnp2 is created from the numba model with to_T_xs at a different composition
    vec_attrs = ['dGE_dxs', 'gammas', '_gammas_dGE_dxs',
                 'd2GE_dTdxs', 'dHE_dxs', 'gammas_infinite_dilution', 'dHE_dns',
                'dnHE_dns', 'dSE_dxs', 'dSE_dns', 'dnSE_dns', 'dGE_dns', 'dnGE_dns', 'd2GE_dTdns',
                'd2nGE_dTdns', 'dgammas_dT']

    for attr in vec_attrs:
        assert_close1d(getattr(model, attr)(), getattr(modelnp, attr)(), rtol=1e-13)
        assert_close1d(getattr(modelnp2, attr)(), getattr(modelnp, attr)(), rtol=1e-13)
        assert type(getattr(model, attr)()) is list
        assert type(getattr(modelnp, attr)()) is np.ndarray
        assert type(getattr(modelnp2, attr)()) is np.ndarray

    mat_attrs = ['d2GE_dxixjs', 'd2nGE_dninjs', 'dgammas_dns']
    for attr in mat_attrs:
        assert_close2d(getattr(model, attr)(), getattr(modelnp, attr)(), rtol=1e-13)
        assert_close2d(getattr(modelnp2, attr)(), getattr(modelnp, attr)(), rtol=1e-13)
        assert type(getattr(model, attr)()) is list
        assert type(getattr(modelnp, attr)()) is np.ndarray
        assert type(getattr(modelnp2, attr)()) is np.ndarray

    attrs_3d = ['d3GE_dxixjxks']
    for attr in attrs_3d:
        if hasattr(model, attr):
            # some models do not have this implemented
            assert_close3d(getattr(model, attr)(), getattr(modelnp, attr)(), rtol=1e-13)
            assert_close3d(getattr(modelnp2, attr)(), getattr(modelnp, attr)(), rtol=1e-13)
            assert type(getattr(model, attr)()) is list
            assert type(getattr(modelnp, attr)()) is np.ndarray
            assert type(getattr(modelnp2, attr)()) is np.ndarray

def test_IdealSolution_np_out():
    from thermo import IdealSolution
    from thermo.numba import IdealSolution as IdealSolutionnp
    model = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
    modelnp = IdealSolutionnp(T=300.0, xs=np.array([.1, .2, .3, .4]))
    modelnp2 = modelnp.to_T_xs(T=310.0, xs=np.array([.2, .2, .2, .4]))

    check_numba_np_output_activity(model, modelnp, modelnp2)


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

    check_numba_np_output_activity(model, modelnp, modelnp2)

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

    check_numba_np_output_activity(model, modelnp, modelnp2)

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
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
from math import *
from random import random
from fluids.constants import *
from fluids.numerics import assert_close, assert_close1d, assert_close2d
from numpy.testing import assert_allclose
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


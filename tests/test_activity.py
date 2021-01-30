# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from thermo.activity import GibbsExcess, IdealSolution
from random import random
import numpy as np
from fluids.numerics import jacobian, hessian, derivative, normalize, assert_close, assert_close1d, assert_close2d


def test_IdealSolution():
    GE = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])

    assert 0.0 == GE.GE()
    assert 0.0 == GE.dGE_dT()
    assert 0.0 == GE.d2GE_dT2()
    assert 0.0 == GE.d3GE_dT3()

    assert GE.d2GE_dTdxs() == [0.0]*4
    assert GE.dGE_dxs() == [0.0]*4
    assert GE.d2GE_dxixjs() == [[0.0]*4 for _ in range(4)]
    assert GE.d3GE_dxixjxks() == [[[0.0]*4 for _ in range(4)] for _ in range(4)]


    assert_close(GE.gammas(), [1]*4, atol=0)
    assert_close(GE._gammas_dGE_dxs(), [1]*4, atol=0)
    assert_close(GE.gammas_infinite_dilution(), [1]*4, atol=0)



    assert eval(str(GE)).GE() == GE.GE()

    string = GE.as_json()
    assert 'json_version' in string

    assert IdealSolution.from_json(string).__dict__ == GE.__dict__

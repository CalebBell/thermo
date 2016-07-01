# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from numpy.testing import assert_allclose
import pytest
import numpy as np
from thermo.virial import *


def test_general():
    B = BVirial_Pitzer_Curl(510., 425.2, 38E5, 0.193)
    assert_allclose(B, -0.0002084535541385102)

    B = BVirial_Abbott(510., 425.2, 38E5, 0.193)
    assert_allclose(B, -0.00020570178037383633)

    B = BVirial_Tsonopoulos(510., 425.2, 38E5, 0.193)
    assert_allclose(B, -0.00020935288308483694)


def test_BVirial_Tsonopoulos_Extended():
    B = BVirial_Tsonopoulos_Extended(510., 425.2, 38E5, 0.193, speciestype='normal', dipole=0)
    assert_allclose(B, -0.00020935288308483694)

    B = BVirial_Tsonopoulos_Extended(430., 405.65, 11.28E6, 0.252608, a=0, b=0, speciestype='ketone', dipole=1.469)
    assert_allclose(B, -9.679715056695323e-05)

    # Test all of the different types
    types = ['simple', 'normal', 'methyl alcohol', 'water', 'ketone',
    'aldehyde', 'alkyl nitrile', 'ether', 'carboxylic acid', 'ester', 'carboxylic acid',
    'ester', 'alkyl halide', 'mercaptan', 'sulfide', 'disulfide', 'alkanol']

    Bs_calc = [BVirial_Tsonopoulos_Extended(430., 405.65, 11.28E6, 0.252608,
                                            a=0, b=0, speciestype=i, dipole=0.1) for i in types]
    Bs = [-9.002529440027288e-05, -9.002529440027288e-05, -8.136805574379563e-05, -9.232250634010228e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.00558069055045e-05, -9.003495446399036e-05, -9.003495446399036e-05, -9.003495446399036e-05, -9.003495446399036e-05, -7.331247111785242e-05]
    assert_allclose(Bs_calc, Bs)


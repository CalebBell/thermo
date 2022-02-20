# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2022, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from fluids.numerics import assert_close, assert_close1d
import pytest
import numpy as np
from math import *

from thermo.activity import GibbsExcess
from thermo.unifac import *
from fluids.numerics import *
from fluids.constants import R
from thermo.unifac import UFIP, LLEUFIP, LUFIP, DOUFIP2006, DOUFIP2016, NISTUFIP, NISTKTUFIP, PSRKIP, VTPRIP, DOUFSG
from thermo import Chemical
from thermo.joback import smarts_fragment_priority

group_ids = list(range(1, 22)) + [27]
groups = [UFSG[i] for i in group_ids]

def test_UNIFAC_original():
    rdkitmol = Chemical('17059-44-8').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {'ACCH3': 2, 'ACCH2': 1, 'ACH': 3, 'CH3': 1, 'CH2': 1}
    assert success
    
    rdkitmol = Chemical('methanol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {'CH3OH': 1}
    assert success
    
    rdkitmol = Chemical('water').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {'H2O': 1}
    assert success
    
    rdkitmol = Chemical('4-hydroxybenzaldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {'ACOH': 1, 'ACH': 4, 'AC': 1, 'CHO': 1}
    assert success
    
    rdkitmol = Chemical('acetaldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {'CHO': 1, 'CH3': 1}
    assert success

    rdkitmol = Chemical('camphor').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {'CH2CO': 1, 'CH3': 3, 'CH2': 2, 'CH': 1, 'C': 2}
    assert success

    rdkitmol = Chemical('butyraldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {'CHO': 1, 'CH3': 1, 'CH2': 2}
    assert success


def test_UNIFAC_failures():
    rdkitmol = Chemical('5-Methylfurfuryl alcohol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {'ACCH3': 1, 'ACCH2': 1, 'ACH': 2, 'OH': 1}
    assert not success


test_UNIFAC_original()
test_UNIFAC_failures()
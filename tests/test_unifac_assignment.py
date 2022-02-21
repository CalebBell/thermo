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

group_ids = list(range(1, 61))
groups = [UFSG[i] for i in group_ids]

def test_UNIFAC_original():
    rdkitmol = Chemical('17059-44-8').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {11: 2, 12: 1, 9: 3, 1: 1, 2: 1}
    assert success
    
    rdkitmol = Chemical('methanol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {15: 1}
    assert success
    
    rdkitmol = Chemical('water').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {16: 1}
    assert success
    
    rdkitmol = Chemical('4-hydroxybenzaldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {17: 1, 9: 4, 10: 1, 20: 1}
    assert success
    
    rdkitmol = Chemical('acetaldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {20: 1, 1: 1}
    assert success

    rdkitmol = Chemical('camphor').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {19: 1, 1: 3, 2: 2, 3: 1, 4: 2}
    assert success

    rdkitmol = Chemical('butyraldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {20: 1, 1: 1, 2: 2}
    assert success

    rdkitmol = Chemical('glutaraldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {20: 2, 2: 3}
    assert success
    
    rdkitmol = Chemical('triacetin').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {21: 3, 2: 2, 3: 1}
    assert success
    
    rdkitmol = Chemical('1,4-dioxin').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {27: 2, 6: 1}
    assert success

    rdkitmol = Chemical('phthalan').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {12: 1, 9: 4, 10: 1, 27: 1}
    assert success
    
    rdkitmol = Chemical('butyl propionate').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {22: 1, 1: 2, 2: 3}
    assert success

    rdkitmol = Chemical('ethyl formate').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {23: 1, 1: 1, 2: 1}
    assert success

    rdkitmol = Chemical('dimethyl ether').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {24: 1, 1: 1}
    assert success

    rdkitmol = Chemical('diethyl ether').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {25: 1, 1: 2, 2: 1}
    assert success

    rdkitmol = Chemical('diisopropyl ether').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {26: 1, 1: 4, 3: 1}
    assert success
    
    rdkitmol = Chemical('tetrahydrofuran').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {27: 1, 2: 3}
    assert success
    
    rdkitmol = Chemical('malondialdehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {2: 1, 20: 2}
    assert success
    
    rdkitmol = Chemical('glycolaldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {20: 1, 14: 1, 2: 1}
    assert success

    rdkitmol = Chemical('methylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {28: 1}
    assert success

    rdkitmol = Chemical('ethylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {29: 1, 1: 1}
    assert success
    
    rdkitmol = Chemical('isopropyl amine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {30: 1, 1: 2}
    assert success
    
    rdkitmol = Chemical('dimethylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {31: 1, 1: 1}
    assert success
    
    
    rdkitmol = Chemical('diethylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {32: 1, 1: 2, 2: 1}
    assert success
    
    rdkitmol = Chemical('diisopropyl amine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {33: 1, 1: 4, 3: 1}
    assert success

    rdkitmol = Chemical('trimethylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 34: 1}
    assert success

    rdkitmol = Chemical('triethylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 3, 2: 2, 35: 1}
    assert success

    rdkitmol = Chemical('aniline').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {9: 5, 36: 1}
    assert success

    rdkitmol = Chemical('pyridine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {37: 1}
    assert success


    rdkitmol = Chemical('2-methylpyridine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {38: 1, 1: 1}
    assert success

    rdkitmol = Chemical('2,3-Dimethylpyridine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 39: 1}
    assert success
    
    rdkitmol = Chemical('acetonitrile').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {40:1}
    assert success
    
    rdkitmol = Chemical('propionitrile').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 41: 1}
    assert success
    
    
    rdkitmol = Chemical('acetic acid').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 42: 1}
    assert success
    
    rdkitmol = Chemical('formic acid').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {43: 1}
    assert success
    
    rdkitmol = Chemical('Butyl chloride').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 2: 2, 44: 1}
    assert success
    
    rdkitmol = Chemical('2-chloropropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 45: 1}
    assert success
    
    rdkitmol = Chemical('2-Chloro-2-methylpropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 3, 46: 1}
    assert success

    rdkitmol = Chemical('Dichloromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {47: 1}
    assert success
    rdkitmol = Chemical('Ethylidene chloride').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 48: 1}
    assert success

    rdkitmol = Chemical('2,2-Dichloropropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 49: 1}
    assert success
    
    rdkitmol = Chemical('chloroform').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {50: 1}
    assert success
    rdkitmol = Chemical('1,1,1-Trichloroethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 51: 1}
    assert success

    rdkitmol = Chemical('tetrachloromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {52: 1}
    assert success

    rdkitmol = Chemical('chlorobenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {9: 5, 53: 1}
    assert success
    
    
    rdkitmol = Chemical('nitromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {54: 1}
    assert success

    rdkitmol = Chemical('1-nitropropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {55: 1, 1: 1, 2: 1}
    assert success

    rdkitmol = Chemical('2-nitropropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {56: 1, 1: 2}
    assert success
    
    rdkitmol = Chemical('nitrobenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {9: 5, 57: 1}
    assert success
    
    rdkitmol = Chemical('carbon disulphide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {58: 1}
    assert success

    rdkitmol = Chemical('methanethiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {59: 1}
    assert success

    rdkitmol = Chemical('ethanethiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 60: 1}
    assert success


def test_UNIFAC_failures():
    rdkitmol = Chemical('5-Methylfurfuryl alcohol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=groups, rdkitmol=rdkitmol)
    assert assignment == {11: 1, 12: 1, 9: 2, 14: 1}
    assert not success


'''
The following compounds were investigated and found to have a different fragmentation.
hydroxyacetone
'''

'''The following compounds need isotopic help
c = Chemical('n,n-dideuterio-1-phenyl-methanamine')
'''

'''The following compounds need a better tria-and-error algorithm
c = Chemical('5-[1,3-bis(oxidanyl)propan-2-ylamino]-1-(hydroxymethyl)cyclohexane-1,2,3,4-tetrol')
2-[4-[bis(2-hydroxyethyl)amino]-2,6-bis[4-[bis(2-hydroxyethyl)amino]phenyl]phenyl]ethanenitrile
'''

'''
The following compounds were processed by the DDBST web application but don't
actually have UNIFAC groups.
4,5-didehydropyridine

'''
test_UNIFAC_original()
test_UNIFAC_failures()
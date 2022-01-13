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

import os
import pytest
from fluids.numerics import assert_close, assert_close1d
from thermo.functional_groups import *
from thermo import Chemical

try:
    import rdkit
    from rdkit import Chem
except:
    rdkit = None
    
    
@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_is_mercaptan():
    mercaptan_chemicals = ['methanethiol', 'Ethanethiol', '1-Propanethiol', '2-Propanethiol',
                           'Allyl mercaptan', 'Butanethiol', 'tert-Butyl mercaptan', 'pentyl mercaptan',
                          'Thiophenol', 'Dimercaptosuccinic acid', 'Thioacetic acid',
                          'Glutathione', 'Cysteine', '2-Mercaptoethanol',
                          'Dithiothreitol', 'Furan-2-ylmethanethiol', '3-Mercaptopropane-1,2-diol',
                           '3-Mercapto-1-propanesulfonic acid', '1-Hexadecanethiol', 'Pentachlorobenzenethiol']
    for c in mercaptan_chemicals:
        assert is_mercaptan(Chemical(c).rdkitmol)
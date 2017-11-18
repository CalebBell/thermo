# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from thermo.joback import *

@pytest.mark.rdkit
def test_Joback_acetone():
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolDescriptors
    for i in [Chem.MolFromSmiles('CC(=O)C'), 'CC(=O)C']:
        ex = Joback(i) # Acetone example
        assert_allclose(ex.Tb(ex.counts), 322.11) 
        assert_allclose(ex.Tm(ex.counts), 173.5)
        assert_allclose(ex.Tc(ex.counts), 500.5590049525365) 
        assert_allclose(ex.Tc(ex.counts, 322.11), 500.5590049525365) 
        assert_allclose(ex.Pc(ex.counts, ex.atom_count), 4802499.604994407)
        assert_allclose(ex.Vc(ex.counts), 0.0002095)
        assert_allclose(ex.Hf(ex.counts), -217830)
        assert_allclose(ex.Gf(ex.counts), -154540)
        assert_allclose(ex.Hfus(ex.counts), 5125)
        assert_allclose(ex.Hvap(ex.counts), 29018)
        assert_allclose(ex.Cpig_coeffs(ex.counts),[7.52, 0.26084, -0.0001207, 1.546e-08] )
        assert_allclose(ex.Cpig(300), 75.32642000000001)
        assert_allclose(ex.mul_coeffs(ex.counts), [839.11, -14.99])
        assert_allclose(ex.mul(300), 0.0002940378347162687)
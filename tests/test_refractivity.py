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
import pandas as pd
import numpy as np
from thermo.refractivity import *

def test_refractivity_CRC():
    assert_allclose(CRC_RI_organic['RI'].sum(), 6602.78821)
    assert_allclose(CRC_RI_organic['RIT'].sum(), 1304152.35)

def test_refractivity_general():
    mat = np.array([refractive_index(i) for i in  CRC_RI_organic.index.values])
    Ts = mat[:,1]
    # Test refractive index sum
    assert_allclose(mat[:,0].sum(), 6602.78821)
    # Test temperature sum
    assert_allclose(Ts[~np.isnan(Ts)].sum(), 1304152.35)

    vals = refractive_index(CASRN='64-17-5')
    assert_allclose(vals, (1.3611, 293.15))

    # One value only
    val = refractive_index(CASRN='64-17-5', full_info=False)
    assert_allclose(val, 1.3611)

    vals = refractive_index(CASRN='64-17-5', AvailableMethods=True)
    assert vals ==  ['CRC', 'NONE']

    assert (None, None) == refractive_index(CASRN='64-17-5', Method='NONE')
    assert CRC_RI_organic.index.is_unique
    assert CRC_RI_organic.shape == (4490, 2)


    with pytest.raises(Exception):
        refractive_index(CASRN='64-17-5', Method='FAIL')



def test_polarizability_from_RI():
    # Ethanol, with known datum RI and Vm
    alpha = polarizability_from_RI(1.3611, 5.8676E-5)
    assert_allclose(alpha, 5.147658123614415e-30)
    # Experimental value is 5.112 Angstrom^3 from cccbdb, http://cccbdb.nist.gov/polcalccomp2.asp?method=55&basis=9
    # Very good comparison.


def test_molar_refractivity_from_RI():
    # Ethanol, with known datum RI and Vm
    Rm = molar_refractivity_from_RI(1.3611, 5.8676E-5)
    assert_allclose(Rm, 1.2985217089649597e-05)
    # Confirmed with a value of 12.5355 cm^3/mol in http://rasayanjournal.co.in/vol-4/issue-4/38.pdf


def test_RI_from_molar_refractivity():
    RI = RI_from_molar_refractivity(1.2985e-5, 5.8676E-5)
    assert_allclose(RI, 1.3610932757685672)
    # Same value backwards

    assert_allclose(RI_from_molar_refractivity(molar_refractivity_from_RI(1.3611, 5.8676E-5), 5.8676E-5), 1.3611)




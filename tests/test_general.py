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

def test_Hcombustion():
    from thermo.combustion import Hcombustion
    H_Nicotinic_acid = -2730506.5
    H_calc = Hcombustion({'H': 5, 'C': 6, 'O': 2, 'N': 1}, Hf=-344900)
    assert_allclose(H_calc, H_Nicotinic_acid)

    H_methanol = -726024.0
    H_calc = Hcombustion({'H': 4, 'C': 1, 'O': 1}, Hf=-239100)
    assert_allclose(H_calc, H_methanol)

    # Custom example of compound, to show all lines used
    H_custom = -7090590.5
    H_calc = Hcombustion({'C': 10, 'H': 5, 'N': 3, 'O': 5, 'S': 2, 'Br': 8,
    'I':3, 'Cl': 3, 'F':2, 'P': 3}, Hf=-300000)
    assert_allclose(H_calc, H_custom)

    # Custom example, with different Hf for each compound
    H_custom = -7090575.75
    H_calc = Hcombustion({'C': 10, 'H': 5, 'N': 3, 'O': 5, 'S': 2, 'Br': 8,
    'I':3, 'Cl': 3, 'F':2, 'P': 3}, Hf=-300000, HfH2O=-285824, HfCO2=-393473,
    HfSO2=-296801, HfBr2=30881, HfI2=62416, HfHCl=-92172, HfHF=-272710,
    HfP4O10=-3009941, HfO2=0, HfN2=0)
    assert_allclose(H_calc, H_custom)

    # Function returns none if Hf not provided
    assert None == Hcombustion({'H': 4, 'C': 1, 'O': 1})
    # Function returns None if no C is present
    assert None == Hcombustion({'H': 4, 'C': 0, 'O': 1}, Hf=-239100)
    # Function returns None if no C is present in dict
    assert None == Hcombustion({'H': 4, 'O': 1}, Hf=-239100)



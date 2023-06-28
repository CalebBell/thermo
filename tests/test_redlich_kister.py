'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2023, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
'''
import pytest
from fluids.numerics import assert_close, assert_close1d, assert_close2d, linspace

from thermo.redlich_kister import redlich_kister_reverse, redlich_kister_excess_inner, redlich_kister_build_structure
import numpy as np
from chemicals import mixing_simple

def test_redlich_kister_reverse():
    test1 = [1,2,3,4,5,6,7]
    
    out1 = redlich_kister_reverse(test1)
    # check we made a clone of the list
    assert out1 is not test1

    # check the values
    assert out1 == [1, -2, 3, -4, 5, -6, 7]

    # do a numpy array

    test2 = np.array([1,2,3,4,5,6,7])
    out2 = redlich_kister_reverse(test2)
    assert out2 is not test2

    assert_close1d(out2, [1, -2, 3, -4, 5, -6, 7], atol=0, rtol=0)
    assert out2.dtype is not np.dtype(np.float64)
    
    # check the dtype
    test3 = np.array([1,2,3,4,5,6,7], dtype=np.float64)
    out3 = redlich_kister_reverse(test3)

    assert out3.dtype is np.dtype(np.float64)


def test_first_ternary():
    '''surfate tension

    Rafati, A. A., A. Bagheri, and M. Najafi. "Surface Tension of Non-Ideal 
    Binary and Ternary Liquid Mixtures at Various Temperatures and P=81.5kPa."
    The Journal of Chemical Thermodynamics 43, no. 3 (March 1, 2011): 
    248-54. https://doi.org/10.1016/j.jct.2010.09.003.
    '''

    ethylene_glycol_water_Ais = [-32.16, 21.70, -9.78, 18.05, -24.61]
    water_acetonitrile_Ais = [-78.71, 76.03, 5.12, 95.1, -308.86]
    ethylene_glycol_acetonitrile_Ais = [-24.46, 15.91, 13.04, -15.01, 0]
    Ais_many = [ethylene_glycol_water_Ais, ethylene_glycol_acetonitrile_Ais, water_acetonitrile_Ais]
    
    # at 298.15 K from thermo
    sigmas = [0.04822486171003225, 0.07197220523023058, 0.02818895959662564]

    rk_A_struct = redlich_kister_build_structure(3, (5,), Ais_many, [(0,1), (0,2), (1,2)])    

    # Test some points from Table 7
    excess = redlich_kister_excess_inner(3, 5, rk_A_struct, [0.3506, 0.2996, 1-0.3506-0.2996]) # -15.15 data table
    assert_close(excess, -14.911398202855809, rtol=1e-12)

    # Do a check on the actual value, excess units are mN/m so we need to convert
    value = mixing_simple([0.3506, 0.2996, 1-0.3506-0.2996], sigmas) + excess*1e-3
    assert_close(value, .03322, atol=0.001)

    excess = redlich_kister_excess_inner(3, 5, rk_A_struct, [0.7988, 0.0508, 1-0.7988-0.0508]) # -4.25 data table
    assert_close(excess, -3.066466393135123, rtol=1e-12)

    excess = redlich_kister_excess_inner(3, 5, rk_A_struct, [0.2528, 0.3067, 1-0.2528-0.3067])  # -14.75 data table
    assert_close(excess, -17.620727304876826, rtol=1e-12)
    
    # Should be hard zeros
    assert 0 == redlich_kister_excess_inner(3, 5, rk_A_struct, [0, 0, 1])
    assert 0 == redlich_kister_excess_inner(3, 5, rk_A_struct, [0, 1, 0])
    assert 0 == redlich_kister_excess_inner(3, 5, rk_A_struct, [1, 0, 0])

    # Do tests alternating component order, the main one plus 5
    base_point = redlich_kister_excess_inner(3, 5, rk_A_struct, [0.2, 0.5, 0.3])

    struct = redlich_kister_build_structure(3, (5,), Ais_many, [(2,1), (2,0), (1,0)])
    calc = redlich_kister_excess_inner(3, 5, struct, [.3, .5, .2])
    assert_close(base_point, calc, rtol=1e-14)

    struct = redlich_kister_build_structure(3, (5,), Ais_many, [(1, 0), (1,2), (0,2)])
    calc = redlich_kister_excess_inner(3, 5, struct, [.5, .2, .3])
    assert_close(base_point, calc, rtol=1e-14)

    struct = redlich_kister_build_structure(3, (5,), Ais_many, [(0,2), (0,1), (2,1)])
    calc = redlich_kister_excess_inner(3, 5, struct, [.2, .3, .5])
    assert_close(base_point, calc, rtol=1e-14)

    struct = redlich_kister_build_structure(3, (5,), Ais_many, [(2,0), (2,1), (0,1)])
    calc = redlich_kister_excess_inner(3, 5, struct, [.5, .3, .2])
    assert_close(base_point, calc, rtol=1e-14)

    struct = redlich_kister_build_structure(3, (5,), Ais_many, [(1,2), (1,0), (2,0)])
    calc = redlich_kister_excess_inner(3, 5, struct, [.3, .2, .5])
    assert_close(base_point, calc, rtol=1e-14)

     
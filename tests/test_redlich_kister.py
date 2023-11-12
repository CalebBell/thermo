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
from fluids.numerics import assert_close, assert_close1d, assert_close2d, assert_close3d, linspace

from thermo.redlich_kister import (redlich_kister_reverse, redlich_kister_excess_inner, redlich_kister_build_structure, 
redlich_kister_T_dependence, redlich_kister_excess_inner_binary, redlich_kister_excess_binary,
redlich_kister_fitting_to_use)
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

     
def test_redlich_kister_T_dependence_3rd():
    # Randomly generated set of coefficients with 3rd order dependency
    RK_sigma01 = [[0.03134233882904084, -9.38065562363369, 7.238199975960139e-08],
                [0.018041729754263294, -7.234869214731977, 4.623768804321993e-08],
                [-0.019183290707416683, 3.96218526740622, 9.12491453245703e-08],
                [-0.044049639164719614, 12.94883342635889, 3.8515380870305296e-08]]

    RK_sigma02 = [[0.0015895484601951401, -2.678603715700514, 3.937687872620623e-09],
    [-0.037292189095405055, 14.856212772514711, 5.770419306603914e-09],
    [0.058000864253512675, -18.48460220192677, 7.313607254756557e-08],
    [0.03544180374542461, -7.774895194901972, 1.7677329322253075e-08]]

    RK_sigma12 = [[2.0844513776435454, -82.48614430714416, -0.3184488365614371],
    [-1.5070485321350244, 92.83000996326459, 0.31480353198770716],
    [7.921171922937935, -327.88614669085206, -1.0232486013499382],
    [-1.8243600152323596, 94.47768501141788, 0.32939037023488993]]

    T = 298.15
    xs = [.2, .5, .3]

    T_deps_many = [RK_sigma01, RK_sigma02, RK_sigma12]
    rk_struct = redlich_kister_build_structure(3, (4,3), T_deps_many, [(0,1), (0,2), (1,2)])    

    Ais_matrix_for_calc = redlich_kister_T_dependence(rk_struct, T, N=3, N_T=3, N_terms=4)
    Ais_matrix_expect = [[[0.0, 0.0, 0.0, 0.0], [-0.00012012189726080038, -0.006223877051012449, -0.005893536302456449, -0.0006188195976637497],
     [-0.007394509988903227, 0.012535825578242179, -0.0039963786204348884, 0.0093647782020982]],
      [[-0.00012012189726080038, 0.006223877051012449, -0.005893536302456449, 0.0006188195976637497], 
      [0.0, 0.0, 0.0, 0.0], [-0.006601551265295713, 0.5979284168160395, 0.9913785279706477, 0.36925318459691003]], 
      [[-0.007394509988903227, -0.012535825578242179, -0.0039963786204348884, -0.0093647782020982], 
      [-0.006601551265295713, -0.5979284168160395, 0.9913785279706477, -0.36925318459691003], [0.0, 0.0, 0.0, 0.0]]]

    assert_close3d(Ais_matrix_for_calc, Ais_matrix_expect, rtol=1e-13)

    out = redlich_kister_excess_inner(3, 4, Ais_matrix_for_calc, xs)
    assert_close(out, 0.02294048264535485, rtol=1e-13)

def test_redlich_kister_T_dependence_6th():
    RK_sigma01 = [[0.03738404577312337, -12.386159720931362, 8.388613943575708e-08, 8.346242122027595e-08, 7.368750561670829e-12, 5.056541649552947e-16], [0.024080500866817275, -7.821906515628251, 1.925638246015354e-08, 5.52281021650652e-09, 6.188019657764872e-12, 1.1738008530639154e-16], [-0.012364336687028429, 5.3506302051198835, 3.5449448354733586e-08, 6.420967664918157e-08, 5.724246764912454e-12, 7.342354845322604e-16], [-0.06117306692404841, 15.223109174084271, 9.13349139811571e-08, 2.9665381138196562e-08, 5.15333343104307e-12, 6.076353903461948e-16]]
    RK_sigma02 = [[0.001561540466201283, -2.084298267562414, 4.53178736359719e-09, 1.8480009383703937e-08, 5.975063675693564e-12, 2.603960792244431e-16], [-0.04330584108003633, 14.899674474472159, 5.3436134146622455e-08, 6.892026509389344e-08, 5.449083291492482e-12, 2.626172201120956e-16], [0.06713314886825975, -15.386135458625589, 5.7891233744820056e-08, 3.639179056907004e-08, 6.782489285515486e-12, 7.860659633737535e-16], [0.04237489923197258, -6.603271987642499, 8.49023676331543e-08, 3.640634165527224e-08, 8.586020630577965e-12, 6.388300692486393e-16]]
    RK_sigma12 = [[2.1784870396106553, -93.73825856210327, -0.3724267216770331, 6.390469959040341e-08, 9.416780274906474e-12, 5.044320024265512e-16], [-1.1953383982167187, 101.54769212350683, 0.33267920500199, 3.552670195116932e-08, 9.96357206222083e-12, 4.453930888235232e-16], [4.818935659572348, -185.3402389981059, -0.6187050356792033, 7.148044461621326e-08, 2.0030117841998403e-12, 4.2058643585383114e-16], [-2.0908520248150295, 125.25022449078952, 0.34504039266477454, 5.670271099991968e-08, 9.172137430657962e-12, 3.9704543339447306e-16]]

    T = 298.15
    xs = [.2, .5, .3]

    T_deps_many = [RK_sigma01, RK_sigma02, RK_sigma12]
    rk_struct = redlich_kister_build_structure(3, (4,6), T_deps_many, [(0,1), (0,2), (1,2)])    

    Ais_matrix_for_calc = redlich_kister_T_dependence(rk_struct, T, N=3, N_T=6, N_terms=4)
    Ais_matrix_expect = [[[0.0, 0.0, 0.0, 0.0], [-0.004133975178229664, -0.0021525457857327016, 0.005601111113863956, -0.010105143390326695], [-0.005423694519737096, 0.006688765036658436, 0.01553897779694625, 0.020238754809757446]], [[-0.004133975178229664, 0.0021525457857327016, 0.005601111113863956, 0.010105143390326695], [0.0, 0.0, 0.0, 0.0], [-0.25783083315118505, 1.040736768152664, 0.6721909847051324, 0.29515720050858113]], [[-0.005423694519737096, -0.006688765036658436, 0.01553897779694625, -0.020238754809757446], [-0.25783083315118505, -1.040736768152664, 0.6721909847051324, -0.29515720050858113], [0.0, 0.0, 0.0, 0.0]]]

    assert_close3d(Ais_matrix_for_calc, Ais_matrix_expect, rtol=1e-13)

    out = redlich_kister_excess_inner(3, 4, Ais_matrix_for_calc, xs)
    assert_close(out, -0.0036937598363436623, rtol=1e-13)

def test_redlich_kister_excess_inner_binary():
    # from Surface tension of non-ideal binary and ternary liquid mixtures at various temperatures and p=81.5kPa
    As_binary_test = [-79.56, 102.76, -55.68, -30.06, -164.43, 213.01]
    excess = redlich_kister_excess_inner_binary(As_binary_test, [.3, .7])
    assert_close(excess, -28.148313983999994, rtol=1e-13)


def test_redlich_kister_excess_binary():
    As_binary_test = [-79.56, 102.76, -55.68, -30.06, -164.43, 213.01]
    # run the test with 1 T dep term and it should still work
    excess = redlich_kister_excess_binary(As_binary_test, T=298.15, x0=.3, N_T=1, N_terms=6)
    assert_close(excess, -28.148313983999994, rtol=1e-13)


def test_redlich_kister_fitting_to_use():
    coeffs = [-0.001524469890834797, 1.0054473481826371, 0.011306977309368058, -3.260062182048661, 0.015341200351987746, -4.484565752157498, -0.030368567453463967, 8.99609823333603]
    out = redlich_kister_fitting_to_use(coeffs, N_terms=4, N_T=2)
    assert_close2d(out, [[-0.001524469890834797, 1.0054473481826371],
    [0.011306977309368058, -3.260062182048661],
    [0.015341200351987746, -4.484565752157498],
    [-0.030368567453463967, 8.99609823333603]], rtol=1e-16)
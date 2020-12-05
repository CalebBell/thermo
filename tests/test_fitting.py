# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from fluids.numerics import assert_close, assert_close1d, linspace, horner
from numpy.testing import assert_allclose
import pytest
from thermo import fitting
from thermo.fitting import *
import os
import pandas as pd
from math import log, exp


def test_poly_fit_statistics():
    from thermo.eos import PR
    eos = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400., P=1E6)
    coeffs_linear_short = [4.237517714500429e-17, -1.6220845282077796e-13, 2.767061931081117e-10, -2.7334899251582114e-07, 0.00017109676992782628, -0.06958709998929116, 18.296622011252442, -3000.9526306002426, 279584.4945619958, -11321565.153797101]
    assert_close1d(poly_fit_statistics(eos.Psat, coeffs_linear_short, 350, 370, pts=20),
                   ((5.275985987963585e-13, 2.3609241443957343e-13, 0.9999999999997593, 1.0000000000010527)))


def test_fit_cheb_poly():
    from thermo.eos import PR
    from chemicals import SMK
    eos = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400., P=1E6)

    coeffs_linear_short = fit_cheb_poly(eos.Psat, 350, 370, 10)
    for T in linspace(350, 370, 30):
        assert_close(eos.Psat(T), horner(coeffs_linear_short, T), rtol=1e-9)



    # Test transformation of the output only
    coeffs_log_wide = fit_cheb_poly(eos.Psat, 200, 400, 15, interpolation_property=lambda x: log(x),
                                            interpolation_property_inv=lambda x: exp(x))

    for T in linspace(200, 400, 30):
        assert_close(eos.Psat(T), exp(horner(coeffs_log_wide, T)), rtol=1e-9)

    # Test ability to have other arguments depend on it
    coeffs_linear_short_under_P = fit_cheb_poly(lambda T, P: eos.to(T=T, P=P).V_l, 350, 370, 7,
                                                arg_func=lambda T: (eos.Psat(T)*1.1,))

    for T in linspace(350, 370, 30):
        P = eos.Psat(T)*1.1
        assert_close(eos.to(T=T, P=P).V_l, horner(coeffs_linear_short_under_P, T), rtol=1e-9)

    # Test ability to have other arguments depend on it
    coeffs_log_short_above_P = fit_cheb_poly(lambda T, P: eos.to(T=T, P=P).V_g, 350, 370, 7,
                                             arg_func=lambda T: (eos.Psat(T)*.7,),
                                             interpolation_property=lambda x: log(x),
                                             interpolation_property_inv=lambda x: exp(x))
    for T in linspace(350, 370, 30):
        P = eos.Psat(T)*0.7
        assert_close(eos.to(T=T, P=P).V_g, exp(horner(coeffs_log_short_above_P, T)), rtol=1e-9)


    # test interpolation_x
    Tc = 750.0
    coeffs_linear_short_SMK_x_trans = fit_cheb_poly(lambda T: SMK(T, Tc=Tc, omega=0.04), 200, 748, 20,
                                            interpolation_x=lambda T: log(1. - T/Tc),
                                            interpolation_x_inv=lambda x: -(exp(x)-1.0)*Tc)
    for T in linspace(200, 748, 30):
        x =  log(1. - T/Tc)
        assert_close(SMK(T, Tc=Tc, omega=0.04), horner(coeffs_linear_short_SMK_x_trans, x), rtol=1e-7)

def test_Twu91_check_params():
    assert Twu91_check_params((0.694911381318495, 0.919907783415812, 1.70412689631515)) # Ian Bell, methanol
    assert not Twu91_check_params((0.81000842, 0.94790489, 1.49618907)) # Fit without constraints for methanol

    # CH4
    # Twu91_check_params((0.1471, 0.9074, 1.8253))  # Should be consistent - probably a decimal problem
    assert not  Twu91_check_params((0.0777, 0.9288, 3.0432)) # Should be inconsistent

    # N2
    assert Twu91_check_params((0.1240, 0.8897, 2.0138))# consistent
    assert not Twu91_check_params((0.0760, 0.9144, 2.9857)) # inconsistent


def test_Twu91_check_params_Bell():
    folder = os.path.join(os.path.dirname(fitting.__file__), 'Phase Change')

    Bell_2018_data = pd.read_csv(os.path.join(folder, 'Bell 2018 je7b00967_si_001.tsv'),
                                        sep='\t', index_col=6)
    v = Bell_2018_data_values = Bell_2018_data.values

    for (c0, c1, c2) in zip(v[:, 2], v[:, 3], v[:, 4]):
        assert Twu91_check_params((c0, c1, c2))

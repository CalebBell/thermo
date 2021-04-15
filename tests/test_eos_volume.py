# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import pytest
from thermo.eos import *
from thermo.eos_volume import *
from fluids.constants import R
from math import log, exp, sqrt, log10
from fluids.numerics import linspace, derivative, logspace, assert_close, assert_close1d, assert_close2d, assert_close3d

def validate_volume(args, solver, rtol=1e-15):
    b = args[2]
    Vs_good = volume_solutions_mpmath_float(*args)
    Vs_test = solver(*args)
#     print(Vs_good, Vs_test, 'unfiltered')


    Vs_filtered = [i.real for i in Vs_good if (i.real ==0 or abs(i.imag/i.real) < 1E-20) and i.real > b]
    Vs_test_filtered = [i.real for i in Vs_test if (i.real ==0 or abs(i.imag/i.real) < 1E-20) and i.real > b]

#     print(Vs_filtered, Vs_test_filtered, 'filtered')

    if len(Vs_filtered) in (2, 3):
        two_roots_mpmath = True
        Vs_mpmath = min(Vs_filtered), max(Vs_filtered)
    elif len(Vs_filtered) == 0:
        raise ValueError("No volumes found in mpmath")
    else:
        two_roots_mpmath = False
        V_mpmath = Vs_filtered[0]

    if len(Vs_test_filtered) in (2, 3):
        two_roots = True
        Vs = min(Vs_test_filtered), max(Vs_test_filtered)
    elif len(Vs_test_filtered) == 1:
        V = Vs_test_filtered[0]
        two_roots = False
    else:
        raise ValueError("No volumes found")

    if two_roots != two_roots_mpmath:
        raise ValueError("mpmath and volume do not agree")
    if two_roots:
        assert_close(Vs_mpmath[0], Vs[0], rtol=rtol)
        assert_close(Vs_mpmath[1], Vs[1], rtol=rtol)
    else:
        assert_close(V_mpmath, V, rtol=rtol)

hard_parameters = [(0.01, 1e-05, 2.5405184201558786e-05, 5.081036840311757e-05, -6.454233843151321e-10, 0.3872747173781095),
    (0.0001, 154141458.17537114, 2.590839755349289e-05, 2.590839755349289e-05, 0.0, 348530.6151663297),
# (1e-50, 154141458.17537114, 2.590839755349289e-05, 2.590839755349289e-05, 0.0, 348530.6151663297),
# (1e-50, 1e-10, 2.590839755349289e-05, 2.590839755349289e-05, 0.0, 348530.6151663297),
   (0.0001, 0.01, 2.590839755349289e-05, 2.590839755349289e-05, 0.0, 348530.6151663297),
  (3274.5491628777654, 464158883.3612621, 2.581307744572419e-05, 4.4568245538197885e-05, -6.600882066641172e-10, 1.1067350681279088e-16),
(3944.2060594377003, 2.1544346900318865, 2.581307744572419e-05, 4.4568245538197885e-05, -6.600882066641172e-10, 1.9534962802746123e-22),
(0.0001, 1e-10, 2.590839755349289e-05, 2.590839755349289e-05, 0.0, 348530.6151663297),
(25.950242113997547, 4.534878508129528e-05, 1.6557828000856244e-05, 3.311565600171249e-05, -2.741616681059391e-10, 0.027002116417835224),
(25.950242113997547, 1.26185688306607e-56, 1.6557828000856244e-05, 3.311565600171249e-05, -2.741616681059391e-10, 0.027002116417835224),

(613.5907273413233, 9.326033468834164e-10, 0.00018752012786874033, 0.00037504025573748066, -3.5163798355908724e-08, 5.586919514409722),
(613.5907273413233, 3.59381366380539e-09, 0.00018752012786874033, 0.00037504025573748066, -3.5163798355908724e-08, 5.586919514409722),
(1072.267222010334, 3.59381366380539e-09, 0.00018752012786874033, 0.00037504025573748066, -3.5163798355908724e-08, 2.4032479793276456),
]


# TODO important make mpmath cache answers here

@pytest.mark.parametrize("solver", [volume_solutions_halley, GCEOS.volume_solutions])
def test_hard_default_solver_volumes(solver):
    for args in hard_parameters:
        validate_volume(args, solver)
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
import numpy as np
import pandas as pd
from thermo.unifac import *


def test_UNIFAC():
    # Gmehling
    # 05.22 VLE of Hexane-Butanone-2 Via UNIFAC (p. 289)
    # Mathcad (2001) - Solution (zip) - step by step
    # http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/05.22a%20VLE%20of%20Hexane-Butanone-2%20Via%20UNIFAC%20-%20Step%20by%20Step.xps
    gammas_0522A = UNIFAC(chemgroups=[{1:2, 2:4}, {1:1, 2:1, 18:1}], T=60+273.15, xs=[0.5, 0.5])
    assert_allclose(gammas_0522A, [1.4276025835624173, 1.3646545010104225])
    assert_allclose(gammas_0522A, [1.428, 1.365], atol=0.001)
    
    
    # Example 4.14 Activity Coefficients of Ethanol + Benzene with the UNIFAC method
    # Walas, Phase Equilibria in Chemical Engineering
    gammas_414 = UNIFAC(chemgroups=[{1:1, 2:1, 14:1}, {9:6}], T=345., xs=[0.2, 0.8])
    assert_allclose(gammas_414, [2.90999524962436, 1.1038643452317465])
    # Matches faily closely. Confirmed it uses the same coefficients.
    assert_allclose(gammas_414, [2.9119, 1.10832], atol=0.005)
    
    # Examples from ACTCOEFF.XLS, chethermo, from Introductory Chemical Engineering 
    # Thermodynamics, 2nd Ed.: an undergraduate chemical engineering text,
    #  J.Richard Elliott and Carl T. Lira, http://chethermo.net/
    # All match exactly; not even a different gas constant used
    # isopropyl alcohol-water
    gammas_ACTCOEFF_1 = UNIFAC(chemgroups=[{1:2, 3:1, 14:1}, {16:1}], T=80.37+273.15, xs=[0.5, 0.5])
    gammas_ACTCOEFF_1_expect = [1.2667572876079400, 1.700192255741180]
    assert_allclose(gammas_ACTCOEFF_1, gammas_ACTCOEFF_1_expect)
    gammas_ACTCOEFF_2 = UNIFAC(chemgroups=[{1:2, 3:1, 14:1}, {16:1}], T=80.37+273.15, xs=[0.1, 0.9])
    gammas_ACTCOEFF_2_expect = [5.0971362612830500, 1.058637792621310]
    assert_allclose(gammas_ACTCOEFF_2, gammas_ACTCOEFF_2_expect)
    # Add in Propionic acid
    gammas_ACTCOEFF_3 = UNIFAC(chemgroups=[{1:2, 3:1, 14:1}, {16:1}, {1:1, 2:1, 42:1}], T=80.37+273.15, xs=[0.1, 0.1, 0.8])
    gammas_ACTCOEFF_3_expect = [0.9968890535625640, 2.170957708441830, 1.0011209111895400]
    assert_allclose(gammas_ACTCOEFF_3, gammas_ACTCOEFF_3_expect)
    # Add in ethanol
    gammas_ACTCOEFF_4 = UNIFAC(chemgroups=[{1:2, 3:1, 14:1}, {16:1}, {1:1, 2:1, 42:1}, {1:1, 2:1, 14:1}], T=80.37+273.15, xs=[0.01, 0.01, 0.01, .97])
    gammas_ACTCOEFF_4_expect = [1.0172562157805600, 2.721887947655120, 0.9872477280109450, 1.000190624095510]
    assert_allclose(gammas_ACTCOEFF_4, gammas_ACTCOEFF_4_expect)
    # Add in pentane
    gammas_ACTCOEFF_5 = UNIFAC(chemgroups=[{1:2, 3:1, 14:1}, {16:1}, {1:1, 2:1, 42:1}, {1:1, 2:1, 14:1}, {1:2, 2:3}], T=80.37+273.15, xs=[.1, .05, .1, .25, .5])
    gammas_ACTCOEFF_5_expect = [1.2773557137580500, 8.017146862811100, 1.1282116576861800, 1.485860948162550, 1.757426505841570]
    assert_allclose(gammas_ACTCOEFF_5, gammas_ACTCOEFF_5_expect)

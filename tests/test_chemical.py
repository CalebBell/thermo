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
from thermo.chemical import *

def test_Mixture():
    Mixture(['water', 'ethanol'], ws=[.5, .5], T=320, P=1E5)
    Mixture(['water', 'phosphoric acid'], ws=[.5, .5], T=320, P=1E5)
    Mixture('air', T=320, P=1E5)


def test_H_Chemical():
    w = Chemical('water', T=298.15, P=101325.0)
    w.set_ref(T_ref=298.15, P_ref=101325, phase_ref='l', H_ref=0, S_ref=0)
    assert 0 == w.H
    assert 0 == w.S
    w.calculate(297.15, w.P)
    assert_allclose(w.Hm, 1000*(1.814832712-1.890164074), rtol=1E-3)
    w.calculate(274.15, w.P)
    assert_allclose(w.Hm, 1000*(0.07708322535-1.890164074), rtol=1E-4)
    w.calculate(273.15001, w.P) 
    assert w.phase == 'l'
    H_pre_transition = w.Hm
    w.calculate(273.15, w.P)
    assert w.phase == 's'
    dH_transition = w.Hm - H_pre_transition
    assert_allclose(dH_transition, -6010.0, rtol=1E-5)
    # There is not solid heat capacity for water in the database


    w = Chemical('water', T=273.0, P=101325.0)
    w.set_ref(T_ref=273.0, P_ref=101325, phase_ref='s', H_ref=0, S_ref=0)
    w.set_thermo()
    assert 0 == w.H
    assert 0 == w.S
    assert w.phase == 's'
    w.calculate(273.15)
    H_pre_transition = w.Hm
    w.calculate(273.15001, w.P)
    dH_transition = w.Hm - H_pre_transition
    assert_allclose(dH_transition, 6010.0, rtol=1E-5)
    assert w.phase == 'l'
    w.calculate(274.15)
    H_initial_liquid = w.Hm
    w.calculate(298.15, w.P)
    initial_liq_to_STP = w.Hm - H_initial_liquid
    assert_allclose(initial_liq_to_STP, -1000*(0.07708322535-1.890164074), rtol=1E-3)


    w = Chemical('water', T=373, P=101325.0)
    H_pre_vap = w.Hm
    w.calculate(w.Tb+1E-1)
    dH = w.Hm - H_pre_vap
    assert_allclose(dH, 40650, 1E-3) # google search answer


    Hm_as_vapor = w.Hm
    w.calculate(w.T+20)
    dH_20K_gas = w.Hm - Hm_as_vapor
    assert_allclose(dH_20K_gas, 1000*(48.9411675-48.2041134), rtol=1E-1) # Web tables, but hardly matches because of the excess


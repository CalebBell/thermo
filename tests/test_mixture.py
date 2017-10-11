# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from thermo.mixture import Mixture
import thermo
from scipy.integrate import quad
from math import *
from scipy.constants import R

def test_Mixture():
    Mixture(['water', 'ethanol'], ws=[.5, .5], T=320, P=1E5)
    Mixture(['water', 'phosphoric acid'], ws=[.5, .5], T=320, P=1E5)
    Mixture('air', T=320, P=1E5)
    
    Mixture(['ethanol', 'water'], ws=[0.5, 0.5], T=500)
    
    Mixture('water')
    
    s = Mixture(['water', 'ethanol'], P=5200, zs=[0.5, 0.5])
    assert_allclose(s.V_over_F, 0.3061646720256255, rtol=1E-3)
    
    s = Mixture(['water', 'ethanol'], P=5200, zs=[0.5, 0.5])
    assert_allclose(s.quality, 0.34745483870024646, rtol=1E-3)
    
    with pytest.raises(Exception):
        Mixture(['2,2-Dichloro-1,1,1-trifluoroethane'], T=276.15, P=37000, zs=[0.5, 0.5])
    
    m = Mixture(['Na+', 'Cl-', 'water'], ws=[.01, .02, .97]).charge_balance
    assert_allclose(m, -0.0023550338411239182)

def test_Mixture_input_forms():
    # Run a test initializing a mixture from mole fractions, mass fractions,
    # liquid fractions, gas fractions (liq/gas are with volumes of pure components at T and P)
    kwargs = {'ws': [0.5, 0.5], 'zs': [0.7188789914193495, 0.2811210085806504],
              'Vfls': [0.44054617180108374, 0.5594538281989162],
              'Vfgs': [0.7229421485513368, 0.2770578514486633]}
    for key, val in kwargs.items():
        m = Mixture(['water', 'ethanol'], **{key:val})
        assert_allclose(m.zs, kwargs['zs'], rtol=1E-6)
        assert_allclose(m.zs, m.xs)
        assert_allclose(m.Vfls(), kwargs['Vfls'], rtol=1E-5)
        assert_allclose(m.Vfgs(), kwargs['Vfgs'])

    with pytest.raises(Exception):
        Mixture(['water', 'ethanol'])
        
    Mixture(['water'], ws=[1], T=300, P=1E5)
            
def test_Mixture_input_vfs_TP():
    # test against the default arguments of T and P
    m0 = Mixture(['hexane', 'decane'], Vfls=[.5, .5])
    m1 = Mixture(['hexane', 'decane'], Vfls=[.5, .5], Vf_TP=(298.15, None))
    m2 = Mixture(['hexane', 'decane'], Vfls=[.5, .5], Vf_TP=(298.15, None))
    m3 = Mixture(['hexane', 'decane'], Vfls=[.5, .5], Vf_TP=(None, 101325))
    assert_allclose(m0.zs, m1.zs)
    assert_allclose(m0.zs, m2.zs)
    assert_allclose(m0.zs, m3.zs)

    # change T, P slightly - check that's it's still close to the result
    # and do one rough test that the result is still working
    m0 = Mixture(['hexane', 'decane'], Vfls=[.5, .5])
    m1 = Mixture(['hexane', 'decane'], Vfls=[.5, .5], Vf_TP=(300, None))
    m2 = Mixture(['hexane', 'decane'], Vfls=[.5, .5], Vf_TP=(300, 1E5))
    m3 = Mixture(['hexane', 'decane'], Vfls=[.5, .5], Vf_TP=(None, 1E5))
    assert_allclose(m0.zs, m1.zs, rtol=1E-3)
    assert_allclose(m2.zs, [0.5979237361861229, 0.402076263813877], rtol=1E-4)
    assert_allclose(m0.zs, m2.zs, rtol=1E-3)
    assert_allclose(m0.zs, m3.zs, rtol=1E-3)


def test_Mixture_calculated_Vfs():
    # Liquid standard fractions
    S = Mixture(['hexane', 'decane'], zs=[0.25, 0.75])
    Vfls = S.Vfls(298.16, 101326)
    assert_allclose(Vfls, [0.18301434895886864, 0.8169856510411313])
    assert_allclose(S.Vfls(), [0.18301384717011993, 0.8169861528298801])
    assert_allclose(S.Vfls(P=1E6), [0.18292777184777048, 0.8170722281522296])
    assert_allclose(S.Vfls(T=299.15), [0.1830642468206885, 0.8169357531793114])
    
    # gas fractions
    S = Mixture(['hexane', 'decane'], zs=[0.25, 0.75], T=699)
    assert_allclose(S.Vfgs(700, 101326), [0.251236709756207, 0.748763290243793])
    assert_allclose(S.Vfgs(), [0.25124363058052673, 0.7487563694194732])
    assert_allclose(S.Vfgs(P=101326), [0.2512436429605387, 0.7487563570394613])
    assert_allclose(S.Vfgs(T=699), [0.25124363058052673, 0.7487563694194732])

def test_Mixture_predefined():
    for name in ['Air', 'air', u'Air', ['air']]:
        air = Mixture(name)
        assert air.CASs == ['7727-37-9', '7440-37-1', '7782-44-7']
        assert_allclose(air.zs, [0.7811979754734807, 0.009206322604387548, 0.20959570192213187], rtol=1E-4)
        assert_allclose(air.ws, [0.7557, 0.0127, 0.2316], rtol=1E-3)
    
    R401A = Mixture('R401A')
    assert R401A.CASs == ['75-45-6', '75-37-6', '2837-89-0']
    assert_allclose(R401A.zs, [0.578852219944875, 0.18587468325478565, 0.2352730968003393], rtol=1E-4)
    assert_allclose(R401A.ws, [0.53, 0.13, 0.34], rtol=1E-3)
    
    natural_gas = Mixture('Natural gas')
    assert natural_gas.CASs == ['74-82-8', '7727-37-9', '124-38-9', '74-84-0', '74-98-6', '75-28-5', '106-97-8', '78-78-4', '109-66-0', '110-54-3']
    assert_allclose(natural_gas.zs, [0.9652228316853225, 0.002594967217109564, 0.005955831022086067, 0.018185509193506685, 0.004595963476244077, 0.0009769695915451998, 0.001006970610302194, 0.0004729847624453981, 0.0003239924667435125, 0.0006639799746946288], rtol=1E-3)
    assert_allclose(natural_gas.ws, [0.921761382642074, 0.004327306490959737, 0.015603023404535107, 0.03255104882657324, 0.012064018096027144, 0.0033802050703076055, 0.0034840052260078393, 0.002031403047104571, 0.001391502087253131, 0.003406105109157664], rtol=1E-4)


def test_constant_properties():
    R_specific = Mixture(['N2', 'O2'], zs=[0.79, .21]).R_specific
    assert_allclose(R_specific, 288.1928437986195, rtol=1E-5)
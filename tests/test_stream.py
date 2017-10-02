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
from thermo.chemical import Chemical
from thermo.mixture import Mixture
from thermo.stream import Stream
import thermo
from scipy.integrate import quad
from math import *
from scipy.constants import R


def test_Stream():   
    Stream(['H2', 'NH3', 'CO', 'Ar', 'CH4', 'N2'],
           zs=[.7371, 0, .024, .027, .013, .2475], 
    T=500, P=20.5E5, m=300)


def test_Stream_inputs():
    compositions = {'zs': [0.5953064630759212, 0.4046935369240788], 'ws': [0.365177574313603, 0.634822425686397],
                   'Vfgs': [0.6, 0.4], 'Vfls': [0.3114290329842817, 0.6885709670157184]}
    inputs = {'m': 100, 'n': 3405.042096313374, 'Q': 0.11409951553902598}
    flow_inputs = {'ns': [2027.0435669809347, 1377.998529332439], 'ms': [36.517757431360295, 63.482242568639705],
                  'Qls': [0.036643922302061455, 0.08101987400787004], 'Qgs': [48.673177307086064, 32.448784871390714]}
    
    for key1, val1 in compositions.items():
        for key2, val2 in inputs.items():
            m = Stream(['water', 'ethanol'], T=300, P=1E5, **{key1:val1, key2:val2})
            assert_allclose(m.n, inputs['n'])
            assert_allclose(m.m, inputs['m'])
            assert_allclose(m.Q, inputs['Q'])
            assert_allclose(m.ns, flow_inputs['ns'])
            assert_allclose(m.ms, flow_inputs['ms'])
            assert_allclose(m.Qls, flow_inputs['Qls'])
            assert_allclose(m.Qgs, flow_inputs['Qgs'])
            
    for key, val in flow_inputs.items():
        m = Stream(['water', 'ethanol'], T=300, P=1E5, **{key:val})
        assert_allclose(m.n, inputs['n'])
        assert_allclose(m.m, inputs['m'])
        assert_allclose(m.Q, inputs['Q'])
        assert_allclose(m.ns, flow_inputs['ns'])
        assert_allclose(m.ms, flow_inputs['ms'])
        assert_allclose(m.Qls, flow_inputs['Qls'])
        assert_allclose(m.Qgs, flow_inputs['Qgs'])

    with pytest.raises(Exception):
        # two compositions specified
        Stream(['water', 'ethanol'], ns=[6, 4], ws=[.4, .6], T=300, P=1E5)
    with pytest.raises(Exception):
        # two flow rates specified
        Stream(['water', 'ethanol'], ns=[6, 4], n=10, T=300, P=1E5)
    with pytest.raises(Exception):
        # no composition
        Stream(['water', 'ethanol'], n=1, T=300, P=1E5)
    with pytest.raises(Exception):
        # no flow rate
        Stream(['water', 'ethanol'], zs=[.5, .5], T=300, P=1E5)

def test_add_streams():
    # simple example, same components
    ans = {'zs': [0.6, 0.4], 'ws': [0.7932081497794828, 0.20679185022051716], 'm': 0.34847176, 'n': 10}
    prod = Stream(['water', 'ethanol'], ns=[1, 2], T=300, P=1E5) + Stream(['water', 'ethanol'], ns=[3, 4], T=300, P=1E5)
    assert_allclose(prod.zs, ans['zs'])
    assert_allclose(prod.ws, ans['ws'])
    assert_allclose(prod.m, ans['m'])
    assert_allclose(prod.n, ans['n'])

    # add a not a stream
    with pytest.raises(Exception):
        Stream(['decane', 'octane'],  T=300, P=1E5, ns=[4, 5]) +1
    
    # Add two streams, check they're the same if added in a different order
    ans = {'zs': [1/6., 1/3., 1/3., 1/6.], 
           'ws': [0.12364762781718204, 0.3687607770917325, 0.3080280163630483, 0.1995635787280373],
           'm': 0.92382298, 'n': 6}
    
    S1 = Stream(['decane', 'octane'],  T=300, P=1E5, ns=[2, 1])
    S2 = Stream(['Dodecane', 'Tridecane'],  T=300, P=1E5, ns=[2, 1]) 
    prod = S1 + S2
    assert_allclose(prod.ws, ans['ws'])
    assert_allclose(prod.zs, ans['zs'])
    assert_allclose(prod.m, ans['m'])
    assert_allclose(prod.n, ans['n'])
    prod = S2 + S1
    assert_allclose(prod.ws, ans['ws'])
    assert_allclose(prod.zs, ans['zs'])
    assert_allclose(prod.m, ans['m'])
    assert_allclose(prod.n, ans['n'])


def test_sub_streams():
    with pytest.raises(Exception):
        # remove a component not present
        Stream(['water', 'ethanol'], ns=[1, 2], T=300, P=1E5) - Stream(['decane'], ns=[.5], T=300, P=1E5)

    with pytest.raises(Exception):
        # Remove too much of a component 
        Stream(['water', 'ethanol'], ns=[1, 2], T=300, P=1E5) - Stream(['ethanol'], ns=[3], T=300, P=1E5)

    # Take a component completely away
    no_ethanol = Stream(['water', 'ethanol'], ns=[1, 2], T=300, P=1E5) - Stream(['ethanol'], ns=[2], T=300, P=1E5)
    assert len(no_ethanol.zs) == 1
    assert_allclose(no_ethanol.zs, 1)
    assert_allclose(no_ethanol.n, 1)
    assert_allclose(no_ethanol.m, 0.01801528)
    
    # basic case
    m = Stream(['water', 'ethanol'], ns=[1, 2], T=300, P=1E5) - Stream(['ethanol'], ns=[1], T=300, P=1E5)
    assert_allclose(m.ns, [1, 1])
    
    # test case
    m = Stream(['water', 'ethanol', 'decane', 'pentane'], ns=[1, 2, 3, 1E-9], T=300, P=1E5) - Stream(['ethanol'], ns=[2], T=300, P=1E5)
    assert_allclose(m.ns, [1, 3.0, 1e-09])
    assert m.CASs == ['7732-18-5', '124-18-5', '109-66-0']
    
    # Remove a bit more of the chemical that the tolerange allows for  wrt total stream flow:
    with pytest.raises(Exception):
        Stream(['water', 'ethanol', 'decane', 'pentane'], ns=[1, 2, 3, 1E-9], T=300, P=1E5) - Stream(['ethanol', 'pentane'], ns=[2, 1E-9+1E-11], T=300, P=1E5)
    with pytest.raises(Exception):
        Stream(['water', 'ethanol'], ns=[1, 1], T=300, P=1E5) - Stream(['ethanol'], ns=[1+1E-12], T=300, P=1E5)
    m = Stream(['water', 'ethanol'], ns=[1, 1], T=300, P=1E5) - Stream(['ethanol'], ns=[1+9E-13], T=300, P=1E5)
    assert m.CASs == ['7732-18-5']
    
    # Relative to its own stream, removal threshold
    with pytest.raises(Exception):
        # test abs(ns_self[i] - nj)/ns_self[i] > 1E-9
        Stream(['water', 'ethanol'], ns=[1, 1E-12], T=300, P=1E5) - Stream(['ethanol'], ns=[1E-12+1E-20], T=300, P=1E5)
    # test with a little less it gets removed safely, one part in nine extra of the component
    m = Stream(['water', 'ethanol'], ns=[1, 1E-12], T=300, P=1E5) - Stream(['ethanol'], ns=[1E-12+1E-21], T=300, P=1E5)
    assert m.CASs == ['7732-18-5']
    
    # test relative to the product flow rate, ensure we don't remove any extra that results in the extra being much more than the product
    with pytest.raises(Exception):
        m = Stream(['water', 'ethanol'], ns=[1E-7, 1], T=300, P=1E5) - Stream(['ethanol'], ns=[1+1E-15], T=300, P=1E5)
    m = Stream(['water', 'ethanol'], ns=[1E-7, 1], T=300, P=1E5) - Stream(['ethanol'], ns=[1+1E-16], T=300, P=1E5)
    assert_allclose(m.n, 1E-7, rtol=1E-12)
    assert m.CASs == ['7732-18-5']
    
    

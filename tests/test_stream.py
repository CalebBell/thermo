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
from collections import OrderedDict
import pytest
from chemicals.exceptions import OverspeficiedError
from thermo.chemical import Chemical
from thermo.mixture import Mixture
from thermo.stream import Stream, StreamArgs, mole_balance
import thermo
from scipy.integrate import quad
from math import *
from fluids.constants import R
from fluids.numerics import assert_close, assert_close1d


@pytest.mark.deprecated
def test_Stream():
    Stream(['H2', 'NH3', 'CO', 'Ar', 'CH4', 'N2'],
           zs=[.7371, 0, .024, .027, .013, .2475],
    T=500, P=20.5E5, m=300)

    # Test the pressure will be set if none is specified
    obj = Stream(['methanol'], T=305, zs=[1], n=1)
    assert obj.P == obj.P_default
    obj = Stream(['methanol'], P=1E5, zs=[1], n=1)
    assert obj.T == obj.T_default

@pytest.mark.deprecated
def test_Stream_inputs():
    compositions = {'zs': [0.5953064630759212, 0.4046935369240788], 'ws': [0.365177574313603, 0.634822425686397],
                   'Vfgs': [0.6, 0.4], 'Vfls': [0.3114290329842817, 0.6885709670157184]}
    inputs = {'m': 100, 'n': 3405.042096313374, 'Q': 0.11409951553902598}
    flow_inputs = {'ns': [2027.0435669809347, 1377.998529332439], 'ms': [36.517757431360295, 63.482242568639705],
                  'Qls': [0.036643922302061455, 0.08101987400787004], 'Qgs': [50.561333889092964, 34.371951780014115]}

    for key1, val1 in compositions.items():
        for key2, val2 in inputs.items():
            m = Stream(['water', 'ethanol'], T=300, P=1E5, **{key1:val1, key2:val2})
            assert_allclose(m.n, inputs['n'])
            assert_allclose(m.m, inputs['m'])
            assert_allclose(m.Q, inputs['Q'], rtol=1e-5)
            assert_allclose(m.ns, flow_inputs['ns'])
            assert_allclose(m.ms, flow_inputs['ms'])
            assert_allclose(m.Qls, flow_inputs['Qls'], rtol=1e-5)
            assert_allclose(m.Qgs, flow_inputs['Qgs'], rtol=1e-5)

    for key, val in flow_inputs.items():
        m = Stream(['water', 'ethanol'], T=300, P=1E5, **{key:val})
        other_tol = 1e-7 if key not in ('Qls', 'Qgs') else 1e-5
        assert_allclose(m.n, inputs['n'], rtol=other_tol)
        assert_allclose(m.m, inputs['m'], rtol=other_tol)
        assert_allclose(m.Q, inputs['Q'], rtol=1e-5)
        assert_allclose(m.ns, flow_inputs['ns'], rtol=other_tol)
        assert_allclose(m.ms, flow_inputs['ms'], rtol=other_tol)
        assert_allclose(m.Qls, flow_inputs['Qls'], rtol=1e-5)
        assert_allclose(m.Qgs, flow_inputs['Qgs'], rtol=1e-5)

    # Test ordereddict input
    IDs = ['water', 'ethanol']

    for key1, val1 in compositions.items():
        d = OrderedDict()
        for i, j in zip(IDs, val1):
            d.update({i: j})

        for key2, val2 in inputs.items():
            m = Stream(T=300, P=1E5, **{key1:d, key2:val2})
            # Check the composition
            assert_allclose(m.zs, compositions['zs'], rtol=1E-6)
            assert_allclose(m.zs, m.xs)
            assert_allclose(m.Vfls(), compositions['Vfls'], rtol=1E-5)
            assert_allclose(m.Vfgs(), compositions['Vfgs'], rtol=1E-5)

            assert_allclose(m.n, inputs['n'])
            assert_allclose(m.m, inputs['m'])
            assert_allclose(m.Q, inputs['Q'], rtol=1e-5)
            assert_allclose(m.ns, flow_inputs['ns'])
            assert_allclose(m.ms, flow_inputs['ms'])
            assert_allclose(m.Qls, flow_inputs['Qls'], rtol=1e-5)
            assert_allclose(m.Qgs, flow_inputs['Qgs'], rtol=1e-5)

    # Test ordereddict input with flow rates being given as dicts
    for key, val in flow_inputs.items():
        other_tol = 1e-7 if key not in ('Qls', 'Qgs') else 1e-5
        d = OrderedDict()
        for i, j in zip(IDs, val):
            d.update({i: j})

        m = Stream(T=300, P=1E5, **{key:d})
        assert_allclose(m.n, inputs['n'], rtol=other_tol)
        assert_allclose(m.m, inputs['m'], rtol=other_tol)
        assert_allclose(m.Q, inputs['Q'], rtol=1e-5)
        assert_allclose(m.ns, flow_inputs['ns'], rtol=other_tol)
        assert_allclose(m.ms, flow_inputs['ms'], rtol=other_tol)
        assert_allclose(m.Qls, flow_inputs['Qls'], rtol=1e-5)
        assert_allclose(m.Qgs, flow_inputs['Qgs'], rtol=1e-5)


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


@pytest.mark.deprecated
def test_stream_TP_Q():
    n_known = 47.43612113350473
    n = Stream(T=433.0, P=680E3, Q=3800.0/3600, IDs=['CO2'], zs=[1], Q_TP=(273.15, 101325, 'g')).n
    assert_allclose(n, n_known, rtol=1e-3)
    n = Stream(T=433.0, P=680E3, Q=3800.0/3600, IDs=['CO2'], zs=[1], Q_TP=(273.15, 101325, '')).n
    assert_allclose(n, n_known, rtol=1e-3)
    n = Stream(T=433.0, P=680E3, Q=3800.0/3600, IDs=['CO2'], zs=[1], Q_TP=(273.15, 101325, None)).n
    assert_allclose(n, n_known, rtol=1e-3)

@pytest.mark.deprecated
def test_add_streams():
    # simple example, same components
    ans = {'zs': [0.4, 0.6], 'ws': [0.20679185022051716, 0.7932081497794828], 'm': 0.34847176, 'n': 10}
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
    assert_allclose(prod.ws, ans['ws'], rtol=2E-5)
    assert_allclose(prod.zs, ans['zs'], rtol=2E-5)
    assert_allclose(prod.m, ans['m'], rtol=1E-4)
    assert_allclose(prod.n, ans['n'], rtol=2E-5)
    prod = S2 + S1
    assert_allclose(prod.ws, ans['ws'], rtol=2E-5)
    assert_allclose(prod.zs, ans['zs'], rtol=2E-5)
    assert_allclose(prod.m, ans['m'], rtol=1E-4)
    assert_allclose(prod.n, ans['n'], rtol=2E-5)


@pytest.mark.deprecated
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





@pytest.mark.deprecated
def test_StreamArgs():
    s = StreamArgs(T=540)
    s.P = 1E6
    s.n = 2
    s.ws = [.5, .5]
    s.IDs = ['water', 'ethane']
    assert s.stream is not None

    s = StreamArgs(T=540, P=1E6, n=2, ws=[.5, .5], IDs=['water', 'ethanol'])
    assert s.stream is not None

def test_StreamArgs_flow_overspecified():

    with pytest.raises(OverspeficiedError):
        StreamArgs(n=6, m=4)
    with pytest.raises(OverspeficiedError):
        StreamArgs(Q=4, m=4)
    with pytest.raises(OverspeficiedError):
        StreamArgs(Q=4, n=2)


def test_StreamArgs_balance_ns():
    f1 = StreamArgs(n=6, ns=[1,None, 4])
    assert_close1d(f1.ns, [1, 1.0, 4])

    f1 = StreamArgs(n=6, ns=[None])
    assert_close1d(f1.ns, [6])

    f1 = StreamArgs(n=6, ns=[1, None])
    assert_close1d(f1.ns, [1, 5])

    f1 = StreamArgs(n=-6, ns=[-1, None])
    assert_close1d(f1.ns, [-1, -5])

    f1 = StreamArgs(n=-6, ns=[-1, 2, None])
    assert_close1d(f1.ns, [-1, 2, -7.0])

    # Basic check that n_calc works
    f1 = StreamArgs(ns=[3, 2, 5])
    assert_close(f1.n_calc, 10)

    # Check that all can be specified without an error
    f1 = StreamArgs(n=10, ns=[3, 2, 5])

    # Check that the value can indeed differ, but that a stricter check works

    f1 = StreamArgs(n=10*(1+5e-16), ns=[3, 2, 5])
    with pytest.raises(ValueError):
        f1.reconcile_flows(n_tol=3e-16)

def test_StreamArgs_balance_ms():
    f1 = StreamArgs(m=6, ms=[1,None, 4])
    assert_close1d(f1.ms, [1, 1.0, 4])

    f1 = StreamArgs(m=6, ms=[None])
    assert_close1d(f1.ms, [6])

    f1 = StreamArgs(m=6, ms=[1, None])
    assert_close1d(f1.ms, [1, 5])

    f1 = StreamArgs(m=-6, ms=[-1, None])
    assert_close1d(f1.ms, [-1, -5])

    f1 = StreamArgs(m=-6, ms=[-1, 2, None])
    assert_close1d(f1.ms, [-1, 2, -7.0])

    # Basic check that m_calc works
    f1 = StreamArgs(ms=[3, 2, 5])
    assert_close(f1.m_calc, 10)

    # Check that all cam be specified without an error
    f1 = StreamArgs(m=10, ms=[3, 2, 5])

    # Check that the value can indeed differ, but that a stricter check works

    f1 = StreamArgs(m=10*(1+5e-16), ms=[3, 2, 5])
    with pytest.raises(ValueError):
        f1.reconcile_flows(m_tol=3e-16)

def test_mole_balance_forward():
    # First test case - solve for an outlet ns

    f0 = StreamArgs(ns=[1,2,3,4])
    f1 = StreamArgs(n=5, zs=[.5, 0, 0, .5])
    p0 = StreamArgs()
    progress = mole_balance([f0, f1], [p0], compounds=4)
    assert progress
    p0_ns_expect = [3.5, 2.0, 3.0, 6.5]
    assert_allclose(p0.ns, p0_ns_expect)
    progress = mole_balance([f0, f1], [p0], compounds=4)
    assert not progress

    # Second test case - solve for an outlet ns with two outlets
    f0 = StreamArgs(ns=[1,2,3,4])
    f1 = StreamArgs(n=5, zs=[.5, 0, 0, .5])
    p0 = StreamArgs()
    p1 = StreamArgs(n=1, zs=[.25, .25, .25, .25])
    progress = mole_balance([f0, f1], [p0, p1], compounds=4)

    p0_ns_expect = [3.25, 1.75, 2.75, 6.25]
    assert_allclose(p0.ns, p0_ns_expect)

    progress = mole_balance([f0, f1], [p0, p1], compounds=4)
    assert not progress

    # Third test case - solve for an outlet ns with three outlets, p1 unknown
    f0 = StreamArgs(ns=[1,2,3,4])
    f1 = StreamArgs(n=5, zs=[.5, 0, 0, .5])
    p0 = StreamArgs(n=1, zs=[.25, .25, .25, .25])
    p1 = StreamArgs()
    p2 = StreamArgs(n=1, zs=[.5, .5, .5, .5])
    progress = mole_balance([f0, f1], [p0, p1, p2], compounds=4)

    assert progress
    p1_ns_expect = [2.75, 1.25, 2.25, 5.75]
    assert_allclose(p1.ns, p1_ns_expect)

    progress = mole_balance([f0, f1], [p0, p1, p2], compounds=4)
    assert not progress

    # Fourth test case - solve for an outlet ns with three outlets, p1 unknown, negative flows
    f0 = StreamArgs(ns=[1,2,3,4])
    f1 = StreamArgs(n=5, zs=[.5, 0, 0, .5])
    p0 = StreamArgs(n=1, zs=[.25, .25, .25, .25])
    p1 = StreamArgs()
    p2 = StreamArgs(n=1, zs=[-50, .5, 50, -50])
    progress = mole_balance([f0, f1], [p0, p1, p2], compounds=4)

    assert progress
    p1_ns_expect = [53.25, 1.25, -47.25, 56.25]
    assert_allclose(p1.ns, p1_ns_expect)

    progress = mole_balance([f0, f1], [p0, p1, p2], compounds=4)
    assert not progress

    # Fifth test - do component balances at end
    f0 = StreamArgs(ns=[1, 2, 3, None])
    f1 = StreamArgs(ns=[3, 5, 9, 3])
    p0 = StreamArgs(ns=[None, None, None, 5])
    progress = mole_balance([f0, f1], [p0], compounds=4)

    assert progress
    f0_ns_expect = [1, 2, 3, 2]
    f1_ns_expect = [3, 5, 9, 3]
    p0_ns_expect = [4, 7, 12, 5]
    assert_allclose(p0.ns, p0_ns_expect)
    assert_allclose(f0.ns, f0_ns_expect)
    assert_allclose(f1.ns, f1_ns_expect)

    progress = mole_balance([f0, f1], [p0], compounds=4)
    assert not progress

    # Six test - can only solve some specs not all specs
    f0 = StreamArgs(ns=[1, None, 3, None])
    f1 = StreamArgs(ns=[None, 5, 9, 3])
    p0 = StreamArgs(ns=[4, None, None, 5])
    progress = mole_balance([f0, f1], [p0], compounds=4)

    assert progress
    f0_ns_expect = [1, None, 3, 2]
    f1_ns_expect = [3, 5, 9, 3]
    p0_ns_expect = [4, None, 12, 5]
    assert_close1d(p0.ns, p0_ns_expect)
    assert_close1d(f0.ns, f0_ns_expect)
    assert_close1d(f1.ns, f1_ns_expect)

    progress = mole_balance([f0, f1], [p0], compounds=4)
    assert not progress

    # 7th random test
    f0 = StreamArgs(ns=[1, 2, 3, 4])
    f1 = StreamArgs(ns=[None, None, None, None])
    p0 = StreamArgs(ns=[0, 6, 7, 9])
    progress = mole_balance([p0], [f0, f1], compounds=4)

    assert progress
    assert_close1d(p0.ns, [0, 6, 7, 9])
    assert_close1d(f0.ns,[1, 2, 3, 4])
    assert_close1d(f1.ns, [-1.0, 4.0, 4.0, 5.0])

    progress = mole_balance([p0], [f0, f1], compounds=4)
    assert not progress

    # 8th test, mole balance at end
    f0 = StreamArgs(ns=[1, 2, 3, 4])
    f1 = StreamArgs()
    f2 = StreamArgs(n=5)
    p0 = StreamArgs(ns=[0, 6, 7, 9])
    progress = mole_balance([f0, f1, f2], [p0], compounds=4)
    assert progress
    ns_expect = [10, 7.0, 5, 22]
    ns_now = [f0.n_calc, f1.n_calc, f2.n_calc, p0.n_calc]
    assert_close1d(ns_expect, ns_now)

    f0.ns, f1.ns, f2.ns, p0.ns, ()

    progress = mole_balance([f0, f1, f2], [p0], compounds=4)
    assert not progress

    # 9th test, mole balance at end goes nowhere
    f0 = StreamArgs(ns=[1, 2, 3, 4])
    f1 = StreamArgs()
    f2 = StreamArgs()
    p0 = StreamArgs(ns=[0, 6, 7, 9])
    progress = mole_balance([f0, f1, f2], [p0], compounds=4)
    assert not progress
    ns_expect = [10, None, None, 22]
    ns_now = [f0.n_calc, f1.n_calc, f2.n_calc, p0.n_calc]
    assert_close1d(ns_expect, ns_now)



def test_mole_balance_backward():
    # THESE ARE ALL THE SAME TEST CASES - JUST SWITCH WHICH in/out LIST IS GIVEN AS INLET/OUTLET

    # First test case - solve for an outlet ns
    f0 = StreamArgs(ns=[1,2,3,4])
    f1 = StreamArgs(n=5, zs=[.5, 0, 0, .5])
    p0 = StreamArgs()
    progress = mole_balance([p0], [f0, f1], compounds=4)
    assert progress
    p0_ns_expect = [3.5, 2.0, 3.0, 6.5]
    assert_allclose(p0.ns, p0_ns_expect)
    progress = mole_balance([p0], [f0, f1], compounds=4)
    assert not progress

    # Second test case - solve for an outlet ns with two outlets
    f0 = StreamArgs(ns=[1,2,3,4])
    f1 = StreamArgs(n=5, zs=[.5, 0, 0, .5])
    p0 = StreamArgs()
    p1 = StreamArgs(n=1, zs=[.25, .25, .25, .25])
    progress = mole_balance([p0, p1], [f0, f1], compounds=4)

    p0_ns_expect = [3.25, 1.75, 2.75, 6.25]
    assert_allclose(p0.ns, p0_ns_expect)

    progress = mole_balance([p0, p1], [f0, f1], compounds=4)
    assert not progress

    # Third test case - solve for an outlet ns with three outlets, p1 unknown
    f0 = StreamArgs(ns=[1,2,3,4])
    f1 = StreamArgs(n=5, zs=[.5, 0, 0, .5])
    p0 = StreamArgs(n=1, zs=[.25, .25, .25, .25])
    p1 = StreamArgs()
    p2 = StreamArgs(n=1, zs=[.5, .5, .5, .5])
    progress = mole_balance([p0, p1, p2], [f0, f1], compounds=4)

    assert progress
    p1_ns_expect = [2.75, 1.25, 2.25, 5.75]
    assert_allclose(p1.ns, p1_ns_expect)

    progress = mole_balance([p0, p1, p2], [f0, f1], compounds=4)
    assert not progress

    # Fourth test case - solve for an outlet ns with three outlets, p1 unknown, negative flows
    f0 = StreamArgs(ns=[1,2,3,4])
    f1 = StreamArgs(n=5, zs=[.5, 0, 0, .5])
    p0 = StreamArgs(n=1, zs=[.25, .25, .25, .25])
    p1 = StreamArgs()
    p2 = StreamArgs(n=1, zs=[-50, .5, 50, -50])
    progress = mole_balance([p0, p1, p2], [f0, f1], compounds=4)

    assert progress
    p1_ns_expect = [53.25, 1.25, -47.25, 56.25]
    assert_allclose(p1.ns, p1_ns_expect)

    progress = mole_balance( [p0, p1, p2], [f0, f1], compounds=4)
    assert not progress

    # Fifth test - do component balances at end
    f0 = StreamArgs(ns=[1, 2, 3, None])
    f1 = StreamArgs(ns=[3, 5, 9, 3])
    p0 = StreamArgs(ns=[None, None, None, 5])
    progress = mole_balance([p0], [f0, f1], compounds=4)

    assert progress
    f0_ns_expect = [1, 2, 3, 2]
    f1_ns_expect = [3, 5, 9, 3]
    p0_ns_expect = [4, 7, 12, 5]
    assert_allclose(p0.ns, p0_ns_expect)
    assert_allclose(f0.ns, f0_ns_expect)
    assert_allclose(f1.ns, f1_ns_expect)

    progress = mole_balance([p0], [f0, f1], compounds=4)
    assert not progress

    # Six test - can only solve some specs not all specs
    f0 = StreamArgs(ns=[1, None, 3, None])
    f1 = StreamArgs(ns=[None, 5, 9, 3])
    p0 = StreamArgs(ns=[4, None, None, 5])
    progress = mole_balance([p0], [f0, f1], compounds=4)

    assert progress
    f0_ns_expect = [1, None, 3, 2]
    f1_ns_expect = [3, 5, 9, 3]
    p0_ns_expect = [4, None, 12, 5]
    assert_close1d(p0.ns, p0_ns_expect)
    assert_close1d(f0.ns, f0_ns_expect)
    assert_close1d(f1.ns, f1_ns_expect)

    progress = mole_balance([p0], [f0, f1], compounds=4)
    assert not progress

    # 7th random test
    f0 = StreamArgs(ns=[1, 2, 3, 4])
    f1 = StreamArgs(ns=[None, None, None, None])
    p0 = StreamArgs(ns=[0, 6, 7, 9])
    progress = mole_balance([f0, f1], [p0], compounds=4)

    assert progress
    assert_close1d(p0.ns, [0, 6, 7, 9])
    assert_close1d(f0.ns,[1, 2, 3, 4])
    assert_close1d(f1.ns, [-1.0, 4.0, 4.0, 5.0])

    progress = mole_balance([f0, f1], [p0], compounds=4)
    assert not progress

    # 8th test, mole balance at end
    f0 = StreamArgs(ns=[1, 2, 3, 4])
    f1 = StreamArgs()
    f2 = StreamArgs(n=5)
    p0 = StreamArgs(ns=[0, 6, 7, 9])
    progress = mole_balance([p0], [f0, f1, f2], compounds=4)
    assert progress
    ns_expect = [10, 7.0, 5, 22]
    ns_now = [f0.n_calc, f1.n_calc, f2.n_calc, p0.n_calc]
    assert_close1d(ns_expect, ns_now)

    progress = mole_balance([p0], [f0, f1, f2], compounds=4)
    assert not progress

    # 9th test, mole balance at end goes nowhere
    f0 = StreamArgs(ns=[1, 2, 3, 4])
    f1 = StreamArgs()
    f2 = StreamArgs()
    p0 = StreamArgs(ns=[0, 6, 7, 9])
    progress = mole_balance([p0], [f0, f1, f2], compounds=4)
    assert not progress
    ns_expect = [10, None, None, 22]
    ns_now = [f0.n_calc, f1.n_calc, f2.n_calc, p0.n_calc]
    assert_close1d(ns_expect, ns_now)

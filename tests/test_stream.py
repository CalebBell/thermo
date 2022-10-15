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
from thermo.stream import Stream, StreamArgs, mole_balance, EquilibriumStream
import thermo
from scipy.integrate import quad
from math import *
from fluids.constants import R
from fluids.numerics import assert_close, assert_close1d

from thermo import ChemicalConstantsPackage, HeatCapacityGas, PropertyCorrelationsPackage, CEOSLiquid,IdealGas, CEOSGas, PR78MIX, PRMIX, FlashPureVLS, VolumeLiquid
from thermo import FlashVLN
from chemicals import property_molar_to_mass


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
    
    
def test_EquilibriumStream_unusual_inputs():
    constants = ChemicalConstantsPackage(Tcs=[647.14], Pcs=[22048320.0], omegas=[0.344], MWs=[18.01528],  CASs=['7732-18-5'],)
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)
    
    P = 1e5
    T = 200
    liquid = CEOSLiquid(PR78MIX, T=T, P=P, zs=[1], **kwargs)
    gas = CEOSGas(PR78MIX, T=T, P=P, zs=[1], **kwargs)
    flasher = FlashPureVLS(constants, correlations, gas, [liquid], []) #
    
    
    stream = EquilibriumStream(T=T, P=P, zs=[1], m=1, flasher=flasher)
    
    check_base = EquilibriumStream(P=P, V=stream.V(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check_base.T)
    
    check = EquilibriumStream(P=P, rho=stream.rho(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    
    
    check = EquilibriumStream(P=P, rho_mass=stream.rho_mass(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    
    check = EquilibriumStream(P=P, H_mass=stream.H_mass(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    
    check = EquilibriumStream(P=P, S_mass=stream.S_mass(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    
    check = EquilibriumStream(P=P, U_mass=stream.U_mass(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    
    # Hit up the vapor fractions
    stream = EquilibriumStream(VF=.5, P=P, zs=[1], m=1, flasher=flasher)
    
    check = EquilibriumStream(VF=stream.VF, A_mass=stream.A_mass(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    
    check = EquilibriumStream(VF=stream.VF, G_mass=stream.G_mass(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    
def test_EquilibriumStream_different_input_sources():
    constants = ChemicalConstantsPackage(atomss=[{'H': 2, 'O': 1}, {'C': 1, 'H': 4}, {'C': 10, 'H': 22}], CASs=['7732-18-5', '74-82-8', '124-18-5'], Gfgs=[-228554.325, -50443.48000000001, 33414.534999999916], Hfgs=[-241822.0, -74534.0, -249500.0], MWs=[18.01528, 16.04246, 142.28168], names=['water', 'methane', 'decane'], omegas=[0.344, 0.008, 0.49], Pcs=[22048320.0, 4599000.0, 2110000.0], Sfgs=[-44.499999999999964, -80.79999999999997, -948.8999999999997], Tbs=[373.124, 111.65, 447.25], Tcs=[647.14, 190.564, 611.7], Vml_STPs=[1.8087205105724903e-05, 5.858784737690099e-05, 0.00019580845677748954], Vml_60Fs=[1.8036021352633123e-05, 5.858784737690099e-05, 0.00019404661845090487])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
    HeatCapacityGas(load_data=False, poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735])),
    ],
    VolumeLiquids=[VolumeLiquid(load_data=False, poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652])),
    VolumeLiquid(load_data=False, poly_fit=(90.8, 180.564, [7.730541828225242e-20, -7.911042356530585e-17, 3.51935763791471e-14, -8.885734012624568e-12, 1.3922694980104743e-09, -1.3860056394382538e-07, 8.560110533953199e-06, -0.00029978743425740123, 0.004589555868318768])),
    VolumeLiquid(load_data=False, poly_fit=(243.51, 607.7, [1.0056823442253386e-22, -3.2166293088353376e-19, 4.442027873447809e-16, -3.4574825216883073e-13, 1.6583965814129937e-10, -5.018203505211133e-08, 9.353680499788552e-06, -0.0009817356348626736, 0.04459313654596568])),
    ],
    )
    
    gas = IdealGas(HeatCapacityGases=correlations.HeatCapacityGases, zs=[.3, .3, .4], Hfs=constants.Hfgs, Gfs=constants.Gfgs, T=298.15, P=101325.0)
    flasher = FlashVLN(constants=constants, correlations=correlations, gas=gas, liquids=[])
    
    state_flash = flasher.flash(T=300, P=1e5, zs=[.5, .3, .2])
    
    
    base = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], n=10, flasher=flasher)
    case_zs_m = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], m=base.m, flasher=flasher)
    case_zs_Ql = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], Ql=base.Ql, flasher=flasher)
    case_zs_Qg = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], Qg=base.Qg, flasher=flasher)
    case_zs_Q = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], Q=base.Q, flasher=flasher)
    
    case_ws_n = EquilibriumStream(T=300.0, P=1e5, ws=base.ws(), n=10, flasher=flasher)
    case_ws_m = EquilibriumStream(T=300.0, P=1e5, ws=base.ws(), m=base.m, flasher=flasher)
    case_ws_Ql = EquilibriumStream(T=300.0, P=1e5, ws=base.ws(), Ql=base.Ql, flasher=flasher)
    case_ws_Qg = EquilibriumStream(T=300.0, P=1e5, ws=base.ws(), Qg=base.Qg, flasher=flasher)
    case_ws_Q = EquilibriumStream(T=300.0, P=1e5, ws=base.ws(), Q=base.Q, flasher=flasher)
    
    case_Vfgs_n = EquilibriumStream(T=300.0, P=1e5, Vfgs=base.Vfgs(), n=10, flasher=flasher)
    case_Vfgs_m = EquilibriumStream(T=300.0, P=1e5, Vfgs=base.Vfgs(), m=base.m, flasher=flasher)
    case_Vfgs_Ql = EquilibriumStream(T=300.0, P=1e5, Vfgs=base.Vfgs(), Ql=base.Ql, flasher=flasher)
    case_Vfgs_Qg = EquilibriumStream(T=300.0, P=1e5, Vfgs=base.Vfgs(), Qg=base.Qg, flasher=flasher)
    case_Vfgs_Q = EquilibriumStream(T=300.0, P=1e5, Vfgs=base.Vfgs(), Q=base.Q, flasher=flasher)
    
    case_Vfls_n = EquilibriumStream(T=300.0, P=1e5, Vfls=base.Vfls(), n=10, flasher=flasher)
    case_Vfls_m = EquilibriumStream(T=300.0, P=1e5, Vfls=base.Vfls(), m=base.m, flasher=flasher)
    case_Vfls_Ql = EquilibriumStream(T=300.0, P=1e5, Vfls=base.Vfls(), Ql=base.Ql, flasher=flasher)
    case_Vfls_Qg = EquilibriumStream(T=300.0, P=1e5, Vfls=base.Vfls(), Qg=base.Qg, flasher=flasher)
    case_Vfls_Q = EquilibriumStream(T=300.0, P=1e5, Vfls=base.Vfls(), Q=base.Q, flasher=flasher)
    
    case_ns = EquilibriumStream(T=300.0, P=1e5, ns=base.ns, flasher=flasher)
    case_ms = EquilibriumStream(T=300.0, P=1e5, ms=base.ms, flasher=flasher)
    case_Qls = EquilibriumStream(T=300.0, P=1e5, Qls=base.Qls, flasher=flasher)
    case_Qgs = EquilibriumStream(T=300.0, P=1e5, Qgs=base.Qgs, flasher=flasher)
    
    case_energy_ns = EquilibriumStream(energy=base.energy, P=1e5, ns=base.ns, flasher=flasher)
    case_energy_ms = EquilibriumStream(energy=base.energy, P=1e5, ms=base.ms, flasher=flasher)
    case_energy_Qls = EquilibriumStream(energy=base.energy, P=1e5, Qls=base.Qls, flasher=flasher)
    case_energy_Qgs = EquilibriumStream(energy=base.energy, P=1e5, Qgs=base.Qgs, flasher=flasher)
    
    base_existing = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], n=10, flasher=flasher, existing_flash=state_flash)
    case_zs_m_existing = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], m=base.m, flasher=flasher, existing_flash=state_flash)
    case_zs_Ql_existing = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], Ql=base.Ql, flasher=flasher, existing_flash=state_flash)
    case_zs_Qg_existing = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], Qg=base.Qg, flasher=flasher, existing_flash=state_flash)
    case_zs_Q_existing = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], Q=base.Q, flasher=flasher, existing_flash=state_flash)
    
    case_ws_n_existing = EquilibriumStream(T=300.0, P=1e5, ws=base.ws(), n=10, flasher=flasher, existing_flash=state_flash)
    case_ws_m_existing = EquilibriumStream(T=300.0, P=1e5, ws=base.ws(), m=base.m, flasher=flasher, existing_flash=state_flash)
    case_ws_Ql_existing = EquilibriumStream(T=300.0, P=1e5, ws=base.ws(), Ql=base.Ql, flasher=flasher, existing_flash=state_flash)
    case_ws_Qg_existing = EquilibriumStream(T=300.0, P=1e5, ws=base.ws(), Qg=base.Qg, flasher=flasher, existing_flash=state_flash)
    case_ws_Q_existing = EquilibriumStream(T=300.0, P=1e5, ws=base.ws(), Q=base.Q, flasher=flasher, existing_flash=state_flash)
    
    case_Vfgs_n_existing = EquilibriumStream(T=300.0, P=1e5, Vfgs=base.Vfgs(), n=10, flasher=flasher, existing_flash=state_flash)
    case_Vfgs_m_existing = EquilibriumStream(T=300.0, P=1e5, Vfgs=base.Vfgs(), m=base.m, flasher=flasher, existing_flash=state_flash)
    case_Vfgs_Ql_existing = EquilibriumStream(T=300.0, P=1e5, Vfgs=base.Vfgs(), Ql=base.Ql, flasher=flasher, existing_flash=state_flash)
    case_Vfgs_Qg_existing = EquilibriumStream(T=300.0, P=1e5, Vfgs=base.Vfgs(), Qg=base.Qg, flasher=flasher, existing_flash=state_flash)
    case_Vfgs_Q_existing = EquilibriumStream(T=300.0, P=1e5, Vfgs=base.Vfgs(), Q=base.Q, flasher=flasher, existing_flash=state_flash)
    
    case_Vfls_n_existing = EquilibriumStream(T=300.0, P=1e5, Vfls=base.Vfls(), n=10, flasher=flasher, existing_flash=state_flash)
    case_Vfls_m_existing = EquilibriumStream(T=300.0, P=1e5, Vfls=base.Vfls(), m=base.m, flasher=flasher, existing_flash=state_flash)
    case_Vfls_Ql_existing = EquilibriumStream(T=300.0, P=1e5, Vfls=base.Vfls(), Ql=base.Ql, flasher=flasher, existing_flash=state_flash)
    case_Vfls_Qg_existing = EquilibriumStream(T=300.0, P=1e5, Vfls=base.Vfls(), Qg=base.Qg, flasher=flasher, existing_flash=state_flash)
    case_Vfls_Q_existing = EquilibriumStream(T=300.0, P=1e5, Vfls=base.Vfls(), Q=base.Q, flasher=flasher, existing_flash=state_flash)
    
    case_ns_existing = EquilibriumStream(T=300.0, P=1e5, ns=base.ns, flasher=flasher, existing_flash=state_flash)
    case_ms_existing = EquilibriumStream(T=300.0, P=1e5, ms=base.ms, flasher=flasher, existing_flash=state_flash)
    case_Qls_existing = EquilibriumStream(T=300.0, P=1e5, Qls=base.Qls, flasher=flasher, existing_flash=state_flash)
    case_Qgs_existing = EquilibriumStream(T=300.0, P=1e5, Qgs=base.Qgs, flasher=flasher, existing_flash=state_flash)
    
    case_energy_ns_existing = EquilibriumStream(energy=base.energy, P=1e5, ns=base.ns, flasher=flasher, existing_flash=state_flash)
    case_energy_ms_existing = EquilibriumStream(energy=base.energy, P=1e5, ms=base.ms, flasher=flasher, existing_flash=state_flash)
    case_energy_Qls_existing = EquilibriumStream(energy=base.energy, P=1e5, Qls=base.Qls, flasher=flasher, existing_flash=state_flash)
    case_energy_Qgs_existing = EquilibriumStream(energy=base.energy, P=1e5, Qgs=base.Qgs, flasher=flasher, existing_flash=state_flash)
    
    all_cases = [base, case_zs_m, case_zs_Ql, case_zs_Qg, case_zs_Q,
                 case_ws_n, case_ws_m, case_ws_Ql, case_ws_Qg, case_ws_Q,
                case_Vfgs_n, case_Vfgs_m, case_Vfgs_Ql, case_Vfgs_Qg, case_Vfgs_Q,
                case_Vfls_n, case_Vfls_m, case_Vfls_Ql, case_Vfls_Qg, case_Vfls_Q,
                case_ns, case_ms, case_Qls, case_Qgs, 
                case_energy_ns, case_energy_ms, case_energy_Qls, case_energy_Qgs,
                 base_existing, case_zs_m_existing, case_zs_Ql_existing, case_zs_Qg_existing, case_zs_Q_existing,
                 case_ws_n_existing, case_ws_m_existing, case_ws_Ql_existing, case_ws_Qg_existing, case_ws_Q_existing,
                case_Vfgs_n_existing, case_Vfgs_m_existing, case_Vfgs_Ql_existing, case_Vfgs_Qg_existing, case_Vfgs_Q_existing,
                case_Vfls_n_existing, case_Vfls_m_existing, case_Vfls_Ql_existing, case_Vfls_Qg_existing, case_Vfls_Q_existing,
                case_ns_existing, case_ms_existing, case_Qls_existing, case_Qgs_existing, 
                case_energy_ns_existing, case_energy_ms_existing, case_energy_Qls_existing, case_energy_Qgs_existing
                ]
    
    for i, case in enumerate(all_cases):
        assert_close1d(case.ns, [5.0, 3.0, 2.0], rtol=1e-13)
        assert_close1d(case.ms, [0.0900764, 0.04812738, 0.28456336], rtol=1e-13)
        assert_close1d(case.Qls, [9.043602552862452e-05, 0.00017576354213070296, 0.0003916169135549791], rtol=1e-13)
        assert_close1d(case.Qgs, [0.11822415018114264, 0.07093449010868558, 0.04728966007245706], rtol=1e-13)
        assert_close(case.n, sum(case.ns), rtol=1e-13)
        assert_close(case.m, sum(case.ms), rtol=1e-13)
        assert_close(case.Ql, sum(case.Qls), rtol=1e-13)
        assert_close(case.Qg, sum(case.Qgs), rtol=1e-13)
    
        assert_close(case.T, 300.0, rtol=1e-13)
        assert_close(case.P, 1e5, rtol=1e-13)
        assert_close(case.VF, 1, rtol=1e-8)
        assert_close(case.V(), 0.02494338785445972, rtol=1e-8)
        assert_close(case.rho(), 40.090785014242016, rtol=1e-8)
        assert_close(case.rho_mass(), 1.6949066520825955, rtol=1e-8)
    
        assert_close(case.H(), 137.38678195289813, rtol=1e-8)
        assert_close(case.S(),  9.129827636854362, rtol=1e-8)
        assert_close(case.G(), -2601.56150910341, rtol=1e-8)
        assert_close(case.U(), -2356.952003493074 , rtol=1e-8)
        assert_close(case.A(), -5095.900294549382, rtol=1e-8)
    
        assert_close(case.H_mass(), property_molar_to_mass(case.H(), case.MW()), rtol=1e-13)
        assert_close(case.S_mass(), property_molar_to_mass(case.S(), case.MW()), rtol=1e-13)
        assert_close(case.G_mass(), property_molar_to_mass(case.G(), case.MW()), rtol=1e-13)
        assert_close(case.U_mass(), property_molar_to_mass(case.U(), case.MW()), rtol=1e-13)
        assert_close(case.A_mass(), property_molar_to_mass(case.A(), case.MW()), rtol=1e-13)
    
        assert_close(case.H_reactive(), -193033.81321804711, rtol=1e-8)
        assert_close(case.energy, case.H()*case.n, rtol=1e-13)
        assert_close(case.energy_reactive, case.H_reactive()*case.n, rtol=1e-13)
    
        assert_close1d(case.zs, [.5, .3, .2], rtol=1e-13)
        assert_close1d(case.ws(), [0.21306386300505759, 0.11383898001154961, 0.6730971569833928], rtol=1e-13)
        assert_close1d(case.Vfls(), [0.13747911174509148, 0.26719236618433384, 0.5953285220705747], rtol=1e-13)
        assert_close1d(case.Vfgs(), [0.5, 0.3, 0.2], rtol=1e-13)
    
        assert_close1d(case.ns, case.bulk.ns, rtol=1e-13)
        assert_close1d(case.ms, case.bulk.ms, rtol=1e-13)
        assert_close1d(case.Qls, case.bulk.Qls, rtol=1e-13)
        assert_close1d(case.Qgs, case.bulk.Qgs, rtol=1e-13)
        assert_close(case.n, case.bulk.n, rtol=1e-13)
        assert_close(case.m, case.bulk.m, rtol=1e-13)
        assert_close(case.Ql, case.bulk.Ql, rtol=1e-13)
        assert_close(case.Qg, case.bulk.Qg, rtol=1e-13)
    
        assert_close1d(case.ns_calc, case.bulk.ns_calc, rtol=1e-13)
        assert_close1d(case.ms_calc, case.bulk.ms_calc, rtol=1e-13)
        assert_close1d(case.Qls_calc, case.bulk.Qls_calc, rtol=1e-13)
        assert_close1d(case.Qgs_calc, case.bulk.Qgs_calc, rtol=1e-13)
        assert_close(case.n_calc, case.bulk.n_calc, rtol=1e-13)
        assert_close(case.m_calc, case.bulk.m_calc, rtol=1e-13)
        assert_close(case.Ql_calc, case.bulk.Ql_calc, rtol=1e-13)
        assert_close(case.Qg_calc, case.bulk.Qg_calc, rtol=1e-13)
    
        
        
        assert_close(case.T, case.bulk.T, rtol=1e-13)
        assert_close(case.P, case.bulk.P, rtol=1e-13)
        assert_close(case.VF, case.bulk.VF, rtol=1e-13)
        assert_close(case.energy, case.bulk.energy, rtol=1e-13)
        assert_close(case.energy_reactive, case.bulk.energy_reactive, rtol=1e-13)
    
        assert_close(case.V(), case.bulk.V(), rtol=1e-8)
        assert_close(case.rho(), case.bulk.rho(), rtol=1e-13)
        assert_close(case.rho_mass(), case.bulk.rho_mass(), rtol=1e-13)
    
        assert_close(case.H(), case.bulk.H(), rtol=1e-13)
        assert_close(case.S(), case.bulk.S(), rtol=1e-13)
        assert_close(case.G(), case.bulk.G(), rtol=1e-13)
        assert_close(case.U(), case.bulk.U(), rtol=1e-13)
        assert_close(case.A(), case.bulk.A(), rtol=1e-13)
    
        assert_close(case.H_mass(), case.bulk.H_mass(), rtol=1e-13)
        assert_close(case.S_mass(), case.bulk.S_mass(), rtol=1e-13)
        assert_close(case.G_mass(), case.bulk.G_mass(), rtol=1e-13)
        assert_close(case.U_mass(), case.bulk.U_mass(), rtol=1e-13)
        assert_close(case.A_mass(), case.bulk.A_mass(), rtol=1e-13)
    
        assert_close(case.H_reactive(), case.bulk.H_reactive(), rtol=1e-13)
    
        assert_close1d(case.zs, case.bulk.zs, rtol=1e-13)
        assert_close1d(case.ws(), case.bulk.ws(), rtol=1e-13)
        assert_close1d(case.Vfls(), case.bulk.Vfls(), rtol=1e-13)
        assert_close1d(case.Vfgs(), case.bulk.Vfgs(), rtol=1e-13)
    
        # Generic volume
        assert_close(case.Q, 0.2494338785445972, rtol=1e-9)
        assert_close(case.Q, case.bulk.Q)
        
        assert_close(case.T_calc, case.bulk.T_calc, rtol=1e-13)
        assert_close(case.T_calc, case.T, rtol=1e-13)
        assert_close(case.P_calc, case.bulk.P_calc, rtol=1e-13)
        assert_close(case.P_calc, case.P, rtol=1e-13)
        assert_close(case.VF_calc, case.bulk.VF_calc, rtol=1e-13)
        assert_close(case.VF_calc, case.VF, rtol=1e-13)
        assert_close(case.energy_calc, case.bulk.energy_calc, rtol=1e-13)
        assert_close(case.energy_calc, case.energy, rtol=1e-13)
        assert_close(case.energy_reactive_calc, case.bulk.energy_reactive_calc, rtol=1e-13)
        assert_close(case.energy_reactive_calc, case.energy_reactive, rtol=1e-13)
        assert_close(case.H_calc, case.bulk.H_calc, rtol=1e-13)
        assert_close(case.H_calc, case.H(), rtol=1e-13)
    
        assert_close1d(case.zs, case.zs_calc, rtol=1e-13)
        assert_close1d(case.zs_calc, case.bulk.zs_calc, rtol=1e-13)
        assert_close1d(case.ws(), case.ws_calc, rtol=1e-13)
        assert_close1d(case.ws_calc, case.bulk.ws_calc, rtol=1e-13)
        assert_close1d(case.Vfls(), case.Vfls_calc, rtol=1e-13)
        assert_close1d(case.Vfls_calc, case.bulk.Vfls_calc, rtol=1e-13)
        assert_close1d(case.Vfgs(), case.Vfgs_calc, rtol=1e-13)
        assert_close1d(case.Vfgs_calc, case.bulk.Vfgs_calc, rtol=1e-13)

    # cases with StreamAargs
    base = EquilibriumStream(T=300.0, P=1e5, zs=[.5, .3, .2], n=10, flasher=flasher)
    base_args = StreamArgs(T=300.0, P=1e5, zs=[.5, .3, .2], n=10, flasher=flasher)
    case_zs_m = StreamArgs(T=300.0, P=1e5, zs=[.5, .3, .2], m=base.m, flasher=flasher)
    case_zs_Ql = StreamArgs(T=300.0, P=1e5, zs=[.5, .3, .2], Ql=base.Ql, flasher=flasher)
    case_zs_Qg = StreamArgs(T=300.0, P=1e5, zs=[.5, .3, .2], Qg=base.Qg, flasher=flasher)
    case_zs_Q = StreamArgs(T=300.0, P=1e5, zs=[.5, .3, .2], Q=base.Q, flasher=flasher)
    
    case_ws_n = StreamArgs(T=300.0, P=1e5, ws=base.ws(), n=10, flasher=flasher)
    case_ws_m = StreamArgs(T=300.0, P=1e5, ws=base.ws(), m=base.m, flasher=flasher)
    case_ws_Ql = StreamArgs(T=300.0, P=1e5, ws=base.ws(), Ql=base.Ql, flasher=flasher)
    case_ws_Qg = StreamArgs(T=300.0, P=1e5, ws=base.ws(), Qg=base.Qg, flasher=flasher)
    case_ws_Q = StreamArgs(T=300.0, P=1e5, ws=base.ws(), Q=base.Q, flasher=flasher)
    
    case_Vfgs_n = StreamArgs(T=300.0, P=1e5, Vfgs=base.Vfgs(), n=10, flasher=flasher)
    case_Vfgs_m = StreamArgs(T=300.0, P=1e5, Vfgs=base.Vfgs(), m=base.m, flasher=flasher)
    case_Vfgs_Ql = StreamArgs(T=300.0, P=1e5, Vfgs=base.Vfgs(), Ql=base.Ql, flasher=flasher)
    case_Vfgs_Qg = StreamArgs(T=300.0, P=1e5, Vfgs=base.Vfgs(), Qg=base.Qg, flasher=flasher)
    case_Vfgs_Q = StreamArgs(T=300.0, P=1e5, Vfgs=base.Vfgs(), Q=base.Q, flasher=flasher)
    
    case_Vfls_n = StreamArgs(T=300.0, P=1e5, Vfls=base.Vfls(), n=10, flasher=flasher)
    case_Vfls_m = StreamArgs(T=300.0, P=1e5, Vfls=base.Vfls(), m=base.m, flasher=flasher)
    case_Vfls_Ql = StreamArgs(T=300.0, P=1e5, Vfls=base.Vfls(), Ql=base.Ql, flasher=flasher)
    case_Vfls_Qg = StreamArgs(T=300.0, P=1e5, Vfls=base.Vfls(), Qg=base.Qg, flasher=flasher)
    case_Vfls_Q = StreamArgs(T=300.0, P=1e5, Vfls=base.Vfls(), Q=base.Q, flasher=flasher)
    
    case_ns = StreamArgs(T=300.0, P=1e5, ns=base.ns, flasher=flasher)
    case_ms = StreamArgs(T=300.0, P=1e5, ms=base.ms, flasher=flasher)
    case_Qls = StreamArgs(T=300.0, P=1e5, Qls=base.Qls, flasher=flasher)
    case_Qgs = StreamArgs(T=300.0, P=1e5, Qgs=base.Qgs, flasher=flasher)
    
    case_energy_ns = StreamArgs(energy=base.energy, P=1e5, ns=base.ns, flasher=flasher)
    case_energy_ms = StreamArgs(energy=base.energy, P=1e5, ms=base.ms, flasher=flasher)
    case_energy_Qls = StreamArgs(energy=base.energy, P=1e5, Qls=base.Qls, flasher=flasher)
    case_energy_Qgs = StreamArgs(energy=base.energy, P=1e5, Qgs=base.Qgs, flasher=flasher)
    
    all_cases_args = [base_args, case_zs_m, case_zs_Ql, case_zs_Qg, case_zs_Q,
                case_ws_n, case_ws_m, case_ws_Ql, case_ws_Qg, case_ws_Q,
                case_Vfgs_n, case_Vfgs_m, case_Vfgs_Ql, case_Vfgs_Qg, case_Vfgs_Q,
                case_Vfls_n, case_Vfls_m, case_Vfls_Ql, case_Vfls_Qg, case_Vfls_Q,
                case_ns, case_ms, case_Qls, case_Qgs, 
                case_energy_ns, case_energy_ms, case_energy_Qls, case_energy_Qgs
                ]
    for i, case in enumerate(all_cases_args):
    #     print(i)
    #     print(case.__dict__)
    #     print(case.ns_calc)
        assert_close1d(case.ns_calc, [5.0, 3.0, 2.0], rtol=1e-13)
        assert_close1d(case.ms_calc, [0.0900764, 0.04812738, 0.28456336], rtol=1e-13)
        assert_close1d(case.Qls_calc, [9.043602552862452e-05, 0.00017576354213070296, 0.0003916169135549791], rtol=1e-13)
        assert_close1d(case.Qgs_calc, [0.11822415018114264, 0.07093449010868558, 0.04728966007245706], rtol=1e-13)
        assert_close(case.n_calc, sum(case.ns_calc), rtol=1e-13)
        assert_close(case.m_calc, sum(case.ms_calc), rtol=1e-13)
        assert_close(case.Ql_calc, sum(case.Qls_calc), rtol=1e-13)
        assert_close(case.Qg_calc, sum(case.Qgs_calc), rtol=1e-13)
    #     assert_close(case.Q_calc, sum(case.Q_calc), rtol=1e-13)
        
        assert_close(case.T_calc, 300.0, rtol=1e-13)
        assert_close(case.P_calc, 1e5, rtol=1e-13)
        assert_close(case.VF_calc, 1, rtol=1e-8)
        
        
        assert case.composition_specified
        assert case.composition_spec
        assert not case.clean
        assert case.state_specs
        assert case.specified_state_vars == 2
        assert case.state_specified
        assert case.non_pressure_spec_specified
        assert case.flow_spec
        assert case.specified_flow_vars == 1
        assert case.flow_specified
        
        for new, do_flow in zip([case.flash(), case.stream, case.flash_state()], [True, True, False]):
    #         new = case.flash()
            assert_close(new.T, 300, rtol=1e-13)
            assert_close(new.P, 1e5, rtol=1e-13)
            assert_close1d(new.zs, [.5, .3, .2], rtol=1e-13)
            if do_flow:
                assert_close1d(new.ns, [5.0, 3.0, 2.0], rtol=1e-13)
    
        
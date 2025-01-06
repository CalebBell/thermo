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
SOFTWARE.
'''

from collections import OrderedDict
from math import *

import pytest
from chemicals import property_molar_to_mass
from chemicals.exceptions import OverspeficiedError
from fluids.numerics import assert_close, assert_close1d
from numpy.testing import assert_allclose
import json

from thermo import (
    PR78MIX,
    PRMIX,
    CEOSGas,
    CEOSLiquid,
    ChemicalConstantsPackage,
    FlashPureVLS,
    FlashVLN,
    HeatCapacityGas,
    IdealGas,
    PropertyCorrelationsPackage,
    VolumeLiquid,
    EquilibriumState,
    VaporPressure,
)
from thermo.stream import EquilibriumStream, Stream, StreamArgs, mole_balance, energy_balance, EnergyStream
from thermo.chemical_package import ChemicalConstantsPackage, PropertyCorrelationsPackage
from thermo.flash import FlashPureVLS, FlashVLN
from thermo.heat_capacity import HeatCapacityGas
from thermo.interface import SurfaceTension, SurfaceTensionMixture
from thermo.phases import CEOSGas, CEOSLiquid, IdealGas, GibbsExcessLiquid
from thermo.thermal_conductivity import ThermalConductivityGas, ThermalConductivityLiquid, ThermalConductivityGasMixture, ThermalConductivityLiquidMixture
from thermo.vapor_pressure import VaporPressure
from thermo.viscosity import ViscosityGas, ViscosityLiquid, ViscosityGasMixture, ViscosityLiquidMixture
from thermo.volume import VolumeLiquid


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
    compositions = {'zs': [0.6, 0.4], 'ws': [0.3697142863397261, 0.6302857136602741],
                   'Vfgs': [0.6, 0.4], 'Vfls': [0.31563000422121396, 0.6843699957787861]}
    inputs = {'m': 2.9236544, 'n': 100, 'Q': 0.0033317638037953824}
    flow_inputs = {'ns': [60.0, 40.0], 'ms': [1.0809168000000002, 1.8427376000000006],
                  'Qls': [0.0010846512497007257, 0.0023518130762336747], 'Qgs': [1.4966032712675834, 0.997735514178389]}


    # Test ordereddict input
    IDs = ['water', 'ethanol']

    for key1, val1 in compositions.items():
        d = OrderedDict()
        for i, j in zip(IDs, val1):
            d.update({i: j})

        for key2, val2 in inputs.items():
            m = Stream(T=300, P=1E5, **{key1:d, key2:val2})
            # Check the composition
            assert_close1d(m.zs, compositions['zs'], rtol=1E-6)
            assert_close1d(m.zs, m.xs)
            assert_close1d(m.Vfls(), compositions['Vfls'], rtol=1E-5)
            assert_close1d(m.Vfgs(), compositions['Vfgs'], rtol=1E-5)

            assert_close(m.n, inputs['n'])
            assert_close(m.m, inputs['m'])
            assert_close(m.Q, inputs['Q'], rtol=1e-5)
            assert_close1d(m.ns, flow_inputs['ns'])
            assert_close1d(m.ms, flow_inputs['ms'])
            assert_close1d(m.Qls, flow_inputs['Qls'], rtol=1e-5)
            assert_close1d(m.Qgs, flow_inputs['Qgs'], rtol=1e-5)

    # Test ordereddict input with flow rates being given as dicts
    for key, val in flow_inputs.items():
        other_tol = 1e-7 if key not in ('Qls', 'Qgs') else 1e-5
        d = OrderedDict()
        for i, j in zip(IDs, val):
            d.update({i: j})

        m = Stream(T=300, P=1E5, **{key:d})
        assert_close(m.n, inputs['n'], rtol=other_tol)
        assert_close(m.m, inputs['m'], rtol=other_tol)
        assert_close(m.Q, inputs['Q'], rtol=1e-5)
        assert_close1d(m.ns, flow_inputs['ns'], rtol=other_tol)
        assert_close1d(m.ms, flow_inputs['ms'], rtol=other_tol)
        assert_close1d(m.Qls, flow_inputs['Qls'], rtol=1e-5)
        assert_close1d(m.Qgs, flow_inputs['Qgs'], rtol=1e-5)


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
            assert_close(m.n, inputs['n'])
            assert_close(m.m, inputs['m'])
            assert_close(m.Q, inputs['Q'], rtol=1e-5)
            assert_close1d(m.ns, flow_inputs['ns'])
            assert_close1d(m.ms, flow_inputs['ms'])
            assert_close1d(m.Qls, flow_inputs['Qls'], rtol=1e-5)
            assert_close1d(m.Qgs, flow_inputs['Qgs'], rtol=1e-5)

    for key, val in flow_inputs.items():
        m = Stream(['water', 'ethanol'], T=300, P=1E5, **{key:val})
        other_tol = 1e-7 if key not in ('Qls', 'Qgs') else 1e-5
        assert_close(m.n, inputs['n'], rtol=other_tol)
        assert_close(m.m, inputs['m'], rtol=other_tol)
        assert_close(m.Q, inputs['Q'], rtol=1e-5)
        assert_close1d(m.ns, flow_inputs['ns'], rtol=other_tol)
        assert_close1d(m.ms, flow_inputs['ms'], rtol=other_tol)
        assert_close1d(m.Qls, flow_inputs['Qls'], rtol=1e-5)
        assert_close1d(m.Qgs, flow_inputs['Qgs'], rtol=1e-5)

@pytest.mark.deprecated
def test_stream_TP_Q():
    n_known = 47.09364637244089
    n = Stream(T=433.0, P=680E3, Q=3800.0/3600, IDs=['CO2'], zs=[1], Q_TP=(273.15, 101325, 'g')).n
    assert_close(n, n_known, rtol=1e-3)
    n = Stream(T=433.0, P=680E3, Q=3800.0/3600, IDs=['CO2'], zs=[1], Q_TP=(273.15, 101325, '')).n
    assert_close(n, n_known, rtol=1e-3)
    n = Stream(T=433.0, P=680E3, Q=3800.0/3600, IDs=['CO2'], zs=[1], Q_TP=(273.15, 101325, None)).n
    assert_close(n, n_known, rtol=1e-3)

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


@pytest.fixture(
    params=[
        (StreamArgs,{}),
        (EnergyStream,dict(Q=0))
    ],
    ids="StreamArgs EnergyStream".split()
)
def any_type_of_stream(request):
    return request.param

def test_stream_copy(any_type_of_stream):
    stream_type, param = any_type_of_stream
    expected_args = stream_type(**param)
    # when copied
    stream_copy = expected_args.copy()
    # then
    assert isinstance(stream_copy, stream_type)

def test_child_class_copy(any_type_of_stream):
    # lets say user wants to enhance StreamArgs class by:
    stream_type, param = any_type_of_stream
    class EnhancedStreamArgs(stream_type):
        pass
    expected_args = EnhancedStreamArgs(**param)
    # when copied
    stream_copy = expected_args.copy()
    # then
    assert isinstance(stream_copy, EnhancedStreamArgs)


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

    json_data = f1.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_obj = StreamArgs.from_json(json_data)
    assert new_obj == f1


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

def test_energy_balance():
    from thermo import IAPWS95Liquid, IAPWS95Gas, iapws_constants, iapws_correlations
    from thermo.stream import energy_balance, EnergyStream
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])


    in_known = EquilibriumStream(m=5, P=1e5, T=340, flasher=flasher, zs=[1])
    in_unknown = StreamArgs(P=1e5, T=330, flasher=flasher, zs=[1])
    out_known = EquilibriumStream(m=4, P=1e5, T=350, flasher=flasher, zs=[1])
    out_unknown = StreamArgs(P=1e5, T=329, flasher=flasher, zs=[1])

    json_data = in_unknown.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_obj = StreamArgs.from_json(json_data)
    assert new_obj == in_unknown

    energy_balance([in_known,in_unknown], [out_unknown, out_known], reactive=False, use_mass=True)
    in_unknown = in_unknown.stream()
    out_unknown = out_unknown.stream()
    assert_close(in_known.energy+in_unknown.energy, out_unknown.energy + out_known.energy, rtol=1e-13)
    assert_close(in_known.m+in_unknown.m ,out_unknown.m + out_known.m, rtol=1e-13)
    assert_close(in_unknown.m, 29.06875321803928)
    assert_close(out_unknown.m, 30.06875321803839)

    json_data = in_unknown.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_obj = StreamArgs.from_json(json_data)
    assert new_obj == in_unknown

    json_data = out_unknown.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_obj = StreamArgs.from_json(json_data)
    assert new_obj == out_unknown

    # Run the same test reactive
    in_known = EquilibriumStream(m=5, P=1e5, T=340, flasher=flasher, zs=[1])
    in_unknown = StreamArgs(P=1e5, T=330, flasher=flasher, zs=[1])
    out_known = EquilibriumStream(m=4, P=1e5, T=350, flasher=flasher, zs=[1])
    out_unknown = StreamArgs(P=1e5, T=329, flasher=flasher, zs=[1])

    energy_balance([in_known,in_unknown], [out_unknown, out_known], reactive=True, use_mass=True)
    in_unknown = in_unknown.stream()
    out_unknown = out_unknown.stream()
    assert_close(in_known.energy+in_unknown.energy, out_unknown.energy + out_known.energy, rtol=1e-13)
    assert_close(in_known.m+in_unknown.m ,out_unknown.m + out_known.m, rtol=1e-13)
    assert_close(in_unknown.m, 29.06875321803928)
    assert_close(out_unknown.m, 30.06875321803839)

    # Test with dummy energy streams
    in_known = EquilibriumStream(m=5, P=1e5, T=340, flasher=flasher, zs=[1])
    in_unknown = StreamArgs(P=1e5, T=330, flasher=flasher, zs=[1])
    out_known = EquilibriumStream(m=4, P=1e5, T=350, flasher=flasher, zs=[1])
    out_unknown = StreamArgs(P=1e5, T=329, flasher=flasher, zs=[1])
    energy_balance([in_known,in_unknown, EnergyStream(Q=1e-100)], [out_unknown, out_known, EnergyStream(Q=1e-100)], reactive=False, use_mass=True)
    in_unknown = in_unknown.stream()
    out_unknown = out_unknown.stream()
    assert_close(in_known.energy+in_unknown.energy, out_unknown.energy + out_known.energy, rtol=1e-13)
    assert_close(in_known.m+in_unknown.m ,out_unknown.m + out_known.m, rtol=1e-13)
    assert_close(in_unknown.m, 29.06875321803928)
    assert_close(out_unknown.m, 30.06875321803839)

    # Run the same test reactive
    in_known = EquilibriumStream(m=5, P=1e5, T=340, flasher=flasher, zs=[1])
    in_unknown = StreamArgs(P=1e5, T=330, flasher=flasher, zs=[1])
    out_known = EquilibriumStream(m=4, P=1e5, T=350, flasher=flasher, zs=[1])
    out_unknown = StreamArgs(P=1e5, T=329, flasher=flasher, zs=[1])

    energy_balance([in_known,in_unknown, EnergyStream(Q=1e-100)], [out_unknown, out_known], reactive=True, use_mass=True)
    in_unknown = in_unknown.stream()
    out_unknown = out_unknown.stream()
    assert_close(in_known.energy+in_unknown.energy, out_unknown.energy + out_known.energy, rtol=1e-13)
    assert_close(in_known.m+in_unknown.m ,out_unknown.m + out_known.m, rtol=1e-13)
    assert_close(in_unknown.m, 29.06875321803928)
    assert_close(out_unknown.m, 30.06875321803839)

def test_energy_balance_mass_two_unknown_inlets():
    from thermo import IAPWS95Liquid, IAPWS95Gas, iapws_constants, iapws_correlations
    from thermo.stream import energy_balance, EnergyStream
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])

    for reactive in (True, False):
        in0 = StreamArgs(P=1e5, T=340, flasher=flasher, zs=[1])
        in1 = StreamArgs(P=1e5, T=330, flasher=flasher, zs=[1])
        out1 = EquilibriumStream(m=4, P=1e5, T=350, flasher=flasher, zs=[1])
        out2 = EquilibriumStream(P=1e5, m=30.06875321803839, T=329, flasher=flasher, zs=[1])
        progress = energy_balance([in0,in1], [out1, out2], reactive=reactive, use_mass=True)
        assert progress

        in0 = in0.stream()
        in1 = in1.stream()
        assert_close(in0.energy+in1.energy, out1.energy + out2.energy, rtol=1e-13)
        assert_close(in0.m+in1.m ,out1.m + out2.m, rtol=1e-13)
        assert_close(in0.m, 5)
        assert_close(in1.m, 29.06875321803928)

        # with energy streams too
        in0 = StreamArgs(P=1e5, T=340, flasher=flasher, zs=[1])
        in1 = StreamArgs(P=1e5, T=330, flasher=flasher, zs=[1])
        out1 = EquilibriumStream(m=4, P=1e5, T=350, flasher=flasher, zs=[1])
        out2 = EquilibriumStream(P=1e5, m=30.06875321803839, T=329, flasher=flasher, zs=[1])
        progress = energy_balance([EnergyStream(Q=1e-300), in0,in1,EnergyStream(Q=1e-100),EnergyStream(Q=1e-20)], [out1, out2,EnergyStream(Q=1e-100)], reactive=reactive, use_mass=True)
        assert progress

        in0 = in0.stream()
        in1 = in1.stream()
        assert_close(in0.energy+in1.energy, out1.energy + out2.energy, rtol=1e-13)
        assert_close(in0.m+in1.m ,out1.m + out2.m, rtol=1e-13)
        assert_close(in0.m, 5)
        assert_close(in1.m, 29.06875321803928)


        json_data = in0.as_json()
        json_data = json.loads(json.dumps(json_data))
        new_obj = StreamArgs.from_json(json_data)
        assert new_obj == in0

def test_energy_balance_mass_two_unknown_outlets():
    from thermo import IAPWS95Liquid, IAPWS95Gas, iapws_constants, iapws_correlations
    from thermo.stream import energy_balance, EnergyStream
    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])

    for reactive in (True, False):
        in0 = EquilibriumStream(P=1e5, T=340, m=5, flasher=flasher, zs=[1])
        in1 = EquilibriumStream(P=1e5, T=330, m=29.06875321803928, flasher=flasher, zs=[1])
        out0 = StreamArgs(P=1e5, T=350, flasher=flasher, zs=[1])
        out1 = StreamArgs(P=1e5, T=329, flasher=flasher, zs=[1])
        progress = energy_balance([in0,in1], [out0, out1], reactive=reactive, use_mass=True)
        assert progress
        out0 = out0.stream()
        out1 = out1.stream()
        assert_close(in0.energy+in1.energy, out0.energy + out1.energy, rtol=1e-13)
        assert_close(in0.m+in1.m ,out0.m + out1.m, rtol=1e-13)
        assert_close(out0.m, 4)
        assert_close(out1.m, 30.06875321803839)

        # with energy streams too
        in0 = EquilibriumStream(P=1e5, T=340, m=5, flasher=flasher, zs=[1])
        in1 = EquilibriumStream(P=1e5, T=330, m=29.06875321803928, flasher=flasher, zs=[1])
        out0 = StreamArgs(P=1e5, T=350, flasher=flasher, zs=[1])
        out1 = StreamArgs(P=1e5, T=329, flasher=flasher, zs=[1])
        progress = energy_balance([in0,in1,EnergyStream(Q=1e-100),EnergyStream(Q=1e-100),EnergyStream(Q=1e-100)], [EnergyStream(Q=1e-100),out0, out1,EnergyStream(Q=1e-100)], reactive=reactive, use_mass=True)
        assert progress
        out0 = out0.stream()
        out1 = out1.stream()
        assert_close(in0.energy+in1.energy, out0.energy + out1.energy, rtol=1e-13)
        assert_close(in0.m+in1.m ,out0.m + out1.m, rtol=1e-13)
        assert_close(out0.m, 4)
        assert_close(out1.m, 30.06875321803839)



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
    identical_stream = EquilibriumStream(T=T, P=P, zs=[1], m=1, flasher=flasher)
    assert stream == identical_stream

    # flow_checks
    stream_different_flow = EquilibriumStream(T=T, P=P, zs=[1], n=1, flasher=flasher)
    assert stream !=stream_different_flow

    check_base = EquilibriumStream(P=P, V=stream.V(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check_base.T)
    assert stream !=check_base

    check = EquilibriumStream(P=P, rho=stream.rho(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    assert stream !=check


    check = EquilibriumStream(P=P, rho_mass=stream.rho_mass(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    assert stream !=check

    check = EquilibriumStream(P=P, H_mass=stream.H_mass(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    assert stream !=check

    check = EquilibriumStream(P=P, S_mass=stream.S_mass(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    assert stream !=check

    check = EquilibriumStream(P=P, U_mass=stream.U_mass(), zs=[1], m=1, flasher=flasher)
    assert_close(stream.T, check.T)
    assert stream !=check

    # Hit up the vapor fractions
    stream = EquilibriumStream(VF=.5, P=P, zs=[1], m=1, flasher=flasher)
    assert stream !=check

    check = EquilibriumStream(VF=stream.VF, A_mass=stream.A_mass(), zs=[1], m=1, flasher=flasher)
    assert stream !=check
    assert_close(stream.T, check.T)

    check = EquilibriumStream(VF=stream.VF, G_mass=stream.G_mass(), zs=[1], m=1, flasher=flasher)
    assert stream !=check
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

        json_data = case.as_json()
        json_data = json.loads(json.dumps(json_data))
        new_obj = StreamArgs.from_json(json_data)
        assert new_obj == case

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

        json_data = case.as_json()
        json_data = json.loads(json.dumps(json_data))
        new_obj = StreamArgs.from_json(json_data)
        assert new_obj == case

        for new, do_flow in zip([case.stream(), case.flash_state()], [True, False]):
            assert_close(new.T, 300, rtol=1e-13)
            assert_close(new.P, 1e5, rtol=1e-13)
            assert_close1d(new.zs, [.5, .3, .2], rtol=1e-13)
            if do_flow:
                assert_close1d(new.ns, [5.0, 3.0, 2.0], rtol=1e-13)


def test_equilibrium_stream():
    constants = ChemicalConstantsPackage(atomss=[{'C': 3, 'H': 8}], CASs=['74-98-6'], Gfgs=[-24008.76000000004], Hcs=[-2219332.0], Hcs_lower=[-2043286.016], Hcs_lower_mass=[-46337618.47548578], Hcs_mass=[-50329987.42278712], Hfgs=[-104390.0], MWs=[44.09562], names=['propane'], omegas=[0.152], Pcs=[4248000.0], Sfgs=[-269.5999999999999], Tbs=[231.04], Tcs=[369.83], Vml_STPs=[8.982551425831519e-05], Vml_60Fs=[8.721932949945705e-05])
    correlations = PropertyCorrelationsPackage(VaporPressures=[VaporPressure(extrapolation="AntoineAB|DIPPR101_ABC", method="EXP_POLY_FIT", exp_poly_fit=(85.53500000000001, 369.88, [-6.614459112569553e-18, 1.3568029167021588e-14, -1.2023152282336466e-11, 6.026039040950274e-09, -1.877734093773071e-06, 0.00037620249872919755, -0.048277894617307984, 3.790545023359657, -137.90784855852178]))], 
                                                VolumeLiquids=[VolumeLiquid(extrapolation="constant", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53500000000001, 359.89, [1.938305828541795e-22, -3.1476633433892663e-19, 2.1728188240044968e-16, -8.304069912036783e-14, 1.9173728759045994e-11, -2.7331397706706945e-09, 2.346460759888426e-07, -1.1005126799030672e-05, 0.00027390337689920513]))], 
                                                HeatCapacityGases=[HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367]))], 
                                                ViscosityLiquids=[ViscosityLiquid(extrapolation="linear", method="EXP_POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, exp_poly_fit=(85.53500000000001, 369.88, [-1.464186051602898e-17, 2.3687106330094663e-14, -1.601177693693127e-11, 5.837768086076859e-09, -1.2292283268696937e-06, 0.00014590412750959653, -0.0081324465457914, -0.005575029473976978, 8.728914946382764]))], 
                                                ViscosityGases=[ViscosityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-3.1840315590772447e-24, 5.632245762287636e-21, -4.211563759618865e-18, 1.7309219264976467e-15, -4.25623447818058e-13, 6.379502491722484e-11, -5.653736202867734e-09, 2.934273667761606e-07, -4.688742520151596e-06]))], 
                                                ThermalConductivityLiquids=[ThermalConductivityLiquid(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-1.119942102768228e-20, 2.0010259140958334e-17, -1.5432664732534744e-14, 6.710420754858951e-12, -1.8195587835583956e-09, 3.271396846047887e-07, -4.022072549142343e-05, 0.0025702260414860677, 0.15009818638364272]))], 
                                                ThermalConductivityGases=[ThermalConductivityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-4.225968132998871e-21, 7.499602824205188e-18, -5.628911367597151e-15, 2.323826706645111e-12, -5.747490655145977e-10, 8.692266787645703e-08, -7.692555396195328e-06, 0.0004075191640226901, -0.009175197596970735]))], 
                                                SurfaceTensions=[SurfaceTension(Tc=369.83, extrapolation="DIPPR106_AB", method="EXP_POLY_FIT_LN_TAU", exp_poly_fit_ln_tau=(193.15, 366.48, 369.83, [-4.69903867038229e-05, -0.001167676479018507, -0.01245104796692622, -0.07449082604785806, -0.27398619941324853, -0.6372368552001203, -0.9215870661729839, 0.4680106704255822, -3.2163790497734346]))], 
                                                ViscosityGasMixtureObj=ViscosityGasMixture(MWs=[44.09562], molecular_diameters=[], Stockmayers=[], CASs=[], correct_pressure_pure=False, method="HERNING_ZIPPERER", ViscosityGases=[ViscosityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-3.1840315590772447e-24, 5.632245762287636e-21, -4.211563759618865e-18, 1.7309219264976467e-15, -4.25623447818058e-13, 6.379502491722484e-11, -5.653736202867734e-09, 2.934273667761606e-07, -4.688742520151596e-06]))]), 
                                                ViscosityLiquidMixtureObj=ViscosityLiquidMixture(MWs=[], CASs=[], correct_pressure_pure=False, method="Logarithmic mixing, molar", ViscosityLiquids=[ViscosityLiquid(extrapolation="linear", method="EXP_POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, exp_poly_fit=(85.53500000000001, 369.88, [-1.464186051602898e-17, 2.3687106330094663e-14, -1.601177693693127e-11, 5.837768086076859e-09, -1.2292283268696937e-06, 0.00014590412750959653, -0.0081324465457914, -0.005575029473976978, 8.728914946382764]))]), 
                                                ThermalConductivityGasMixtureObj=ThermalConductivityGasMixture(MWs=[44.09562], Tbs=[231.04], CASs=[], correct_pressure_pure=False, method="LINDSAY_BROMLEY", ThermalConductivityGases=[ThermalConductivityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-4.225968132998871e-21, 7.499602824205188e-18, -5.628911367597151e-15, 2.323826706645111e-12, -5.747490655145977e-10, 8.692266787645703e-08, -7.692555396195328e-06, 0.0004075191640226901, -0.009175197596970735]))], ViscosityGases=[ViscosityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-3.1840315590772447e-24, 5.632245762287636e-21, -4.211563759618865e-18, 1.7309219264976467e-15, -4.25623447818058e-13, 6.379502491722484e-11, -5.653736202867734e-09, 2.934273667761606e-07, -4.688742520151596e-06]))]), 
                                                ThermalConductivityLiquidMixtureObj=ThermalConductivityLiquidMixture(MWs=[], CASs=[], correct_pressure_pure=False, method="DIPPR_9H", ThermalConductivityLiquids=[ThermalConductivityLiquid(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-1.119942102768228e-20, 2.0010259140958334e-17, -1.5432664732534744e-14, 6.710420754858951e-12, -1.8195587835583956e-09, 3.271396846047887e-07, -4.022072549142343e-05, 0.0025702260414860677, 0.15009818638364272]))]), 
                                                SurfaceTensionMixtureObj=SurfaceTensionMixture(MWs=[44.09562], Tbs=[231.04], Tcs=[369.83], CASs=['74-98-6'], correct_pressure_pure=False, method="Winterfeld, Scriven, and Davis (1978)", SurfaceTensions=[SurfaceTension(Tc=369.83, extrapolation="DIPPR106_AB", method="EXP_POLY_FIT_LN_TAU", exp_poly_fit_ln_tau=(193.15, 366.48, 369.83, [-4.69903867038229e-05, -0.001167676479018507, -0.01245104796692622, -0.07449082604785806, -0.27398619941324853, -0.6372368552001203, -0.9215870661729839, 0.4680106704255822, -3.2163790497734346]))], VolumeLiquids=[SurfaceTension(Tc=369.83, extrapolation="DIPPR106_AB", method="EXP_POLY_FIT_LN_TAU", exp_poly_fit_ln_tau=(193.15, 366.48, 369.83, [-4.69903867038229e-05, -0.001167676479018507, -0.01245104796692622, -0.07449082604785806, -0.27398619941324853, -0.6372368552001203, -0.9215870661729839, 0.4680106704255822, -3.2163790497734346]))]), 
                                                constants=ChemicalConstantsPackage(atomss=[{'C': 3, 'H': 8}], CASs=['74-98-6'], Gfgs=[-24008.76000000004], Hcs=[-2219332.0], Hcs_lower=[-2043286.016], Hcs_lower_mass=[-46337618.47548578], Hcs_mass=[-50329987.42278712], Hfgs=[-104390.0], MWs=[44.09562], names=['propane'], omegas=[0.152], Pcs=[4248000.0], Sfgs=[-269.5999999999999], Tbs=[231.04], Tcs=[369.83], Vml_STPs=[8.982551425831519e-05], Vml_60Fs=[8.721932949945705e-05]), 
                                                skip_missing=True)

    gas = CEOSGas(eos_class=PRMIX, eos_kwargs={"Pcs": [4248000.0], "Tcs": [369.83], "omegas": [0.152]}, HeatCapacityGases=correlations.HeatCapacityGases, Hfs=[-104390.0], Gfs=[-24008.76000000004], Sfs=[-269.5999999999999], T=298.15, P=101325.0, zs=[1.0])
    liquid = CEOSLiquid(eos_class=PRMIX, eos_kwargs={"Pcs": [4248000.0], "Tcs": [369.83], "omegas": [0.152]}, HeatCapacityGases=correlations.HeatCapacityGases, Hfs=[-104390.0], Gfs=[-24008.76000000004], Sfs=[-269.5999999999999], T=298.15, P=101325.0, zs=[1.0])
    flasher = FlashPureVLS(gas=gas, liquids=[liquid], solids=[], constants=constants, correlations=correlations)

    obj = EquilibriumStream(n=714.3446478105737, flasher=flasher, existing_flash=EquilibriumState(T=394.26111111111106, P=689475.7293168361, zs=[1], betas=[1.0],
                            gas=CEOSGas(eos_class=PRMIX, eos_kwargs={"Pcs": [4248000.0], "Tcs": [369.83], "omegas": [0.152]}, 
                            HeatCapacityGases=correlations.HeatCapacityGases, 
                            Hfs=[-104390.0], Gfs=[-24008.76000000004], Sfs=[-269.5999999999999], T=394.26111111111106, P=689475.7293168361, zs=[1]), liquids=[], solids=[]))

    obj.H()
    obj_copy = eval(obj.__repr__())

    # New object doesn't have the same instance of Flasher or LiquidBulk or constants or properties but they are equal by having the same hash
    # Wasteful but so convinient for writting tests 
    assert obj_copy.flasher == obj.flasher
    assert obj_copy.constants == obj.constants
    assert obj_copy.correlations == obj.correlations

    assert obj_copy.gas == obj.gas
    assert obj_copy.bulk == obj.bulk
    assert obj_copy == obj

    json_data = obj.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_obj = StreamArgs.from_json(json_data)
    assert new_obj == obj


def test_StreamArgs_force_unset_other_composition_cases_zs_ws():
    a = StreamArgs(ws=[.5, .5], multiple_composition_basis=False)
    with pytest.raises(ValueError):
        a.zs = [.2, .8]
    assert_close1d(a.ws, [.5, .5])
    assert a.zs is None

    a.ws = None
    a.zs = [.2, .8]
    assert_close1d(a.zs, [.2, .8])
    assert a.ws is None

    # multiple_composition_basis 
    a = StreamArgs(ws=[.5, .5], multiple_composition_basis=True)
    a.zs = [.2, .8]
    assert_close1d(a.ws, [.5, .5])
    assert_close1d(a.zs, [.2, .8])

def test_StreamArgs_force_unset_other_composition_cases_zs_Vfls():
    a = StreamArgs(Vfls=[.5, .5], multiple_composition_basis=False)
    with pytest.raises(ValueError):
        a.zs = [.2, .8]
    assert_close1d(a.Vfls, [.5, .5])
    assert a.zs is None

    a.Vfls = None
    a.zs = [.2, .8]
    assert_close1d(a.zs, [.2, .8])
    assert a.Vfls is None

    # multiple_composition_basis 
    a = StreamArgs(Vfls=[.5, .5], multiple_composition_basis=True)
    a.zs = [.2, .8]
    assert_close1d(a.Vfls, [.5, .5])
    assert_close1d(a.zs, [.2, .8])

def test_StreamArgs_force_unset_other_composition_cases_zs_Vfgs():
    a = StreamArgs(Vfgs=[.5, .5], multiple_composition_basis=False)
    with pytest.raises(ValueError):
        a.zs = [.2, .8]
    assert_close1d(a.Vfgs, [.5, .5])
    assert a.zs is None

    a.Vfgs = None
    a.zs = [.2, .8]
    assert_close1d(a.zs, [.2, .8])
    assert a.Vfgs is None

    # multiple_composition_basis 
    a = StreamArgs(Vfgs=[.5, .5], multiple_composition_basis=True)
    a.zs = [.2, .8]
    assert_close1d(a.Vfgs, [.5, .5])
    assert_close1d(a.zs, [.2, .8])

def test_StreamArgs_force_unset_other_composition_cases_zs_ns():
    a = StreamArgs(ns=[.5, .5], multiple_composition_basis=False)
    with pytest.raises(ValueError):
        a.zs = [.2, .8]
    assert_close1d(a.ns, [.5, .5])
    assert a.zs is None

    a.ns = None
    a.zs = [.2, .8]
    assert_close1d(a.zs, [.2, .8])
    assert a.ns is None

    # multiple_composition_basis 
    a = StreamArgs(ns=[.5, .5], multiple_composition_basis=True)
    a.zs = [.2, .8]
    assert_close1d(a.ns, [.5, .5])
    assert_close1d(a.zs, [.2, .8])

def test_StreamArgs_force_unset_other_composition_cases_zs_ms():
    a = StreamArgs(ms=[.5, .5], multiple_composition_basis=False)
    with pytest.raises(ValueError):
        a.zs = [.2, .8]
    assert_close1d(a.ms, [.5, .5])
    assert a.zs is None

    a.ms = None
    a.zs = [.2, .8]
    assert_close1d(a.zs, [.2, .8])
    assert a.ms is None

    # multiple_composition_basis 
    a = StreamArgs(ms=[.5, .5], multiple_composition_basis=True)
    a.zs = [.2, .8]
    assert_close1d(a.ms, [.5, .5])
    assert_close1d(a.zs, [.2, .8])

def test_StreamArgs_force_unset_other_composition_cases_zs_Qls():
    a = StreamArgs(Qls=[.5, .5], multiple_composition_basis=False)
    with pytest.raises(ValueError):
        a.zs = [.2, .8]
    assert_close1d(a.Qls, [.5, .5])
    assert a.zs is None

    a.Qls = None
    a.zs = [.2, .8]
    assert_close1d(a.zs, [.2, .8])
    assert a.Qls is None

    # multiple_composition_basis 
    a = StreamArgs(Qls=[.5, .5], multiple_composition_basis=True)
    a.zs = [.2, .8]
    assert_close1d(a.Qls, [.5, .5])
    assert_close1d(a.zs, [.2, .8])

def test_StreamArgs_force_unset_other_composition_cases_zs_Qgs():
    a = StreamArgs(Qgs=[.5, .5], multiple_composition_basis=False)
    with pytest.raises(ValueError):
        a.zs = [.2, .8]
    assert_close1d(a.Qgs, [.5, .5])
    assert a.zs is None

    a.Qgs = None
    a.zs = [.2, .8]
    assert_close1d(a.zs, [.2, .8])
    assert a.Qgs is None

    # multiple_composition_basis 
    a = StreamArgs(Qgs=[.5, .5], multiple_composition_basis=True)
    a.zs = [.2, .8]
    assert_close1d(a.Qgs, [.5, .5])
    assert_close1d(a.zs, [.2, .8])



def test_StreamArgs_force_unset_other_composition_cases_other_input():
    # attr = ['zs', 'ws', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls', 'Qqs']
    attr = ['zs', 'ws', 'Vfls', 'Vfgs']
    for attra in attr:
        for attrb in attr:
            if attra == attrb:
                continue
            kwargs = {attra: [.5, .5], 'multiple_composition_basis': False}
            a = StreamArgs(**kwargs)
            with pytest.raises(ValueError):
                setattr(a,attrb, [.2, .8])
            assert_close1d(getattr(a, attra), [.5, .5])
            assert getattr(a, attrb) is None

            setattr(a, attra, None)
            setattr(a, attrb, [.2, .8])
            assert_close1d(getattr(a, attrb), [.2, .8])
            assert getattr(a, attra) is None

            # multiple_composition_basis 
            kwargs = {attra: [.5, .5], 'multiple_composition_basis': True}
            a = StreamArgs(**kwargs)
            setattr(a,attrb, [.2, .8])
            assert_close1d(getattr(a, attra), [.5, .5])
            assert_close1d(getattr(a, attrb), [.2, .8])

def test_energy_balance():
    constants = ChemicalConstantsPackage(atomss=[{'H': 2, 'O': 1}, {'C': 5, 'H': 12}, {'C': 2, 'H': 6, 'O': 1}, {'C': 7, 'H': 8}], CASs=['7732-18-5', '109-66-0', '64-17-5', '108-88-3'], Gfgs=[-228554.325, -8296.02800000002, -167635.325, 122449.00299999998], Hcs=[0.0, -3508820.0, -1367393.0, -3909978.0], Hcs_lower=[44011.496, -3244751.024, -1235358.512, -3733932.016], Hcs_lower_mass=[2443009.2676883177, -44973054.62406987, -26815722.69432175, -40525244.6916281], Hcs_mass=[0.0, -48633116.180204295, -29681773.46573923, -42435913.27049021], Hfgs=[-241822.0, -146900.0, -234570.0, 50410.0], MWs=[18.01528, 72.14878, 46.06844, 92.13842], names=['water', 'pentane', 'ethanol', 'toluene'], omegas=[0.344, 0.251, 0.635, 0.257], Pcs=[22048320.0, 3370000.0, 6137000.0, 4108000.0], Sfgs=[-44.499999999999964, -464.88, -224.49999999999997, -241.61999999999995], Tbs=[373.124, 309.21, 351.39, 383.75], Tcs=[647.14, 469.7, 514.0, 591.75], Vml_STPs=[1.8087205105724903e-05, 0.00011620195905023226, 5.894275025059058e-05, 0.00010680280039288264], Vml_60Fs=[1.8036021352633123e-05, 0.00011460335996635014, 5.830519706995175e-05, 0.00010562240369105897])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
    HeatCapacityGas(load_data=False, poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [-9.48396765770823e-21, 4.444060985512694e-17, -8.628480671647472e-14, 8.883982004570444e-11, -5.0893293251198045e-08, 1.4947108372371731e-05, -0.0015271248410402886, 0.19186172941013854, 30.797883940134057])),
    ],
    VaporPressures=[VaporPressure(load_data=False, exp_poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317])),
    VaporPressure(load_data=False, exp_poly_fit=(144, 469.6900000000001, [-4.615267824440021e-19, 1.2871894720673305e-15, -1.559590276548174e-12, 1.0761461914304972e-09, -4.6539407107989163e-07, 0.0001305817686830925, -0.023686309296601378, 2.64766854437685, -136.66909337025592])),
    VaporPressure(load_data=False, exp_poly_fit=(159.11, 514.7, [-2.3617526481119e-19, 7.318686894378096e-16, -9.835941684445551e-13, 7.518263303343784e-10, -3.598426432676194e-07, 0.00011171481063640762, -0.022458952185007635, 2.802615041941912, -166.43524219017118])),
    VaporPressure(load_data=False, exp_poly_fit=(178.01, 591.74, [-8.638045111752356e-20, 2.995512203611858e-16, -4.5148088801006036e-13, 3.8761537879200513e-10, -2.0856828984716705e-07, 7.279010846673517e-05, -0.01641020023565049, 2.2758331029405516, -146.04484159879843])),
    ],
    VolumeLiquids=[VolumeLiquid(load_data=False, poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652])),
    VolumeLiquid(load_data=False, poly_fit=(144, 459.7000000000001, [1.0839519373491257e-22, -2.420837244222272e-19, 2.318236501104612e-16, -1.241609625841306e-13, 4.0636406847721776e-11, -8.315431504053525e-09, 1.038485128954003e-06, -7.224842789857136e-05, 0.0022328080060137396])),
    VolumeLiquid(load_data=False, poly_fit=(159.11, 504.71000000000004, [5.388587987308587e-23, -1.331077476340645e-19, 1.4083880805283782e-16, -8.327187308842775e-14, 3.006387047487587e-11, -6.781931902982022e-09, 9.331209920256822e-07, -7.153268618320437e-05, 0.0023871634205665524])),
    VolumeLiquid(load_data=False, poly_fit=(178.01, 581.75, [2.2801490297347937e-23, -6.411956871696508e-20, 7.723152902379232e-17, -5.197203733189603e-14, 2.1348482785660093e-11, -5.476649499770259e-09, 8.564670053875876e-07, -7.455178589434267e-05, 0.0028545812080104068])),
    ],                                           
    )
    N = constants.N
    zs = [1.0/N]*N

    gas = IdealGas(HeatCapacityGases=correlations.HeatCapacityGases, T=298.15, P=101325.0, zs=zs, Hfs=constants.Hfgs,
                Gfs=constants.Gfgs)

    liquid = GibbsExcessLiquid(VaporPressures=correlations.VaporPressures, HeatCapacityGases=correlations.HeatCapacityGases,
                        VolumeLiquids=correlations.VolumeLiquids,
                        use_Poynting=True,
                        use_phis_sat=False, eos_pure_instances=None,
                        Hfs=constants.Hfgs, Gfs=constants.Gfgs,
                        T=298.15, P=101325.0, zs=zs)
    flasher = FlashVLN(constants, correlations, liquids=[liquid], gas=gas, solids=[])




    f1 = StreamArgs(n=5, zs=[.1, .1, .1, .7], P=1e5, T=300.0, flasher=flasher)
    f2 = StreamArgs(m=10, zs=[0, .5, .5, 0], P=1e5, T=310.0, flasher=flasher)
    p0 = StreamArgs(P=1e5, flasher=flasher)
    mole_balance([f1, f2], [p0], 4)
    energy_balance([f1.stream(), f2.stream()], [p0])
    assert p0.stream() is not None    
    
        
    f1 = StreamArgs(n=5, zs=[.1, .1, .1, .7], P=1e5, T=300.0, flasher=flasher)
    f2 = StreamArgs(m=10, zs=[0, .5, .5, 0], P=1e5, flasher=flasher)
    p0 = StreamArgs(P=1e5, flasher=flasher, energy=-5844748.163383733)
    mole_balance([f1, f2], [p0], 4)
    energy_balance([f1, f2], [p0])
    assert p0.stream() is not None
    assert f2.stream() is not None
    assert f1.stream() is not None
    assert f2.flash_state() is not None
    
    # Calculate with an energy stream in the outlets
    f1 = StreamArgs(n=5, zs=[.1, .1, .1, .7], P=1e5, T=300.0, flasher=flasher)
    f2 = StreamArgs(m=10, zs=[0, .5, .5, 0], P=1e5, T=310, flasher=flasher)
    p0 = StreamArgs(P=1e5, flasher=flasher, T=320.0)
    p1 = EnergyStream(Q=None)
    assert mole_balance([f1, f2], [p0], 4)
    assert energy_balance([f1, f2], [p0, p1])
    assert not mole_balance([f1, f2], [p0], 4)
    assert not energy_balance([f1, f2], [p0, p1])
    assert p1.energy is not None
    
    # Calculate with energy in outlets
    f1 = StreamArgs(n=5, zs=[.1, .1, .1, .7], P=1e5, T=300.0, flasher=flasher)
    f2 = StreamArgs(m=10, zs=[0, .5, .5, 0], P=1e5, T=310, flasher=flasher)
    f3 = EnergyStream(Q=1e5)
    
    p0 = StreamArgs(P=1e5, flasher=flasher)
    assert mole_balance([f1, f2], [p0], 4)
    assert energy_balance([f1, f2, f3], [p0])
    assert_close(p0.T_calc, 314.0384879751711, rtol=1e-3)
    assert not mole_balance([f1, f2], [p0], 4)
    assert not energy_balance([f1, f2, f3], [p0])
    
    # Energy balance canot do anything two unknown outs
    f1 = StreamArgs(n=5, zs=[.1, .1, .1, .7], P=1e5, T=300.0, flasher=flasher)
    f2 = StreamArgs(m=10, zs=[0, .5, .5, 0], P=1e5, T=310, flasher=flasher)
    f3 = EnergyStream(Q=1e5)
    
    p0 = StreamArgs(P=1e5, flasher=flasher)
    p1 = EnergyStream(Q=None)
    
    assert mole_balance([f1, f2], [p0], 4)
    assert not energy_balance([f1, f2, f3], [p0, p1])


    # Energy balance canot do anything one unknown in and one out
    f1 = StreamArgs(n=5, zs=[.1, .1, .1, .7], P=1e5, T=300.0, flasher=flasher)
    f2 = StreamArgs(m=10, zs=[0, .5, .5, 0], P=1e5, T=310, flasher=flasher)
    f3 = EnergyStream(Q=None)
    
    p0 = StreamArgs(P=1e5, flasher=flasher)
    p1 = EnergyStream(Q=1e4)
    

    json_data = p1.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_obj = StreamArgs.from_json(json_data)
    assert new_obj == p1

    assert mole_balance([f1, f2], [p0], 4)
    assert not energy_balance([f1, f2, f3], [p0, p1])
    
    
    # Two stream, solve for T in second; T specified in first
    f0 = StreamArgs(n=5, zs=[.1, .1, .1, .7], P=1e5, T=300.0, flasher=flasher)
    p0 = StreamArgs(n=5,  zs=[.1, .1, .1, .7], P=9e4, flasher=flasher)
    energy_balance([f0], [p0])
    assert p0.state_specified
    
    # Two stream, solve for T in second; H specified in first
    f0 = StreamArgs(n=5, H=-37898.61493308388, zs=[.1, .1, .1, .7], P=1e5, flasher=flasher)
    p0 = StreamArgs(n=5,  zs=[.1, .1, .1, .7], P=9e4, flasher=flasher)
    energy_balance([f0], [p0])
    assert p0.state_specified    
    
    
    # Two streams, first equilibrium stream
    f0 = EquilibriumStream(n=5, zs=[.1, .1, .1, .7], P=1e5, T=300.0, flasher=flasher)
    p0 = StreamArgs(n=5,  zs=[.1, .1, .1, .7], P=9e4, flasher=flasher)
    assert hash(p0) != hash(f0)
    energy_balance([f0], [p0])
    assert p0.state_specified

    # Two stream, solve for T in first; H specified in second
    f0 = StreamArgs(n=5, zs=[.1, .1, .1, .7], P=1e5, flasher=flasher)
    p0 = StreamArgs(n=5, H=-37898.61493308388, zs=[.1, .1, .1, .7], P=9e4, flasher=flasher)
    energy_balance([f0], [p0])
    assert f0.state_specified
    
    # Misc things
    f0 = StreamArgs(m=0.39060072, zs=[.1, .1, .1, .7], P=1e5, flasher=flasher)
    p0 = StreamArgs(n=5, H_mass=-485132.4254226075, ws=[0.02306099179745496, 0.09235617896454468, 0.05897126866535219, 0.8256115605726482],
                    P=9e4, flasher=flasher)
    assert p0.flash_state() is not None
    assert energy_balance([f0], [p0])
    assert f0.T_calc is not None
    assert not  energy_balance([f0], [p0])
    
    
    
    product = StreamArgs(m=0.39060072, zs=[.1, .1, .1, .7], P=1e5, flasher=flasher)
    feed = StreamArgs(n=5, H_mass=-485132.4254226075, ws=[0.02306099179745496, 0.09235617896454468, 0.05897126866535219, 0.8256115605726482],
                    P=9e4, flasher=flasher)
    assert feed.flash_state() is not None
    assert energy_balance([feed], [product])
    assert product.T_calc is not None
    assert not  energy_balance([feed], [product]) 
    
    
    # Extra fun
    f0 = StreamArgs(n=1, zs=[.1, .2, .3, .4], P=1e5, T=25.0+273.15, flasher=flasher)
    f1 = StreamArgs(m=.01, ws=[.4, .3, .2, .1], P=1e5, T=30.0 + 273.15, flasher=flasher)
    f2 = StreamArgs(Q=.01, Q_TP=(298.15, 101325.0, 'g'), zs=[.25, .25, .25, .25], P=1e5, flasher=flasher)
    p0 = StreamArgs(ns=[.01, .01, .01, .01], P=1e5, T=273.15, flasher=flasher)
    p1 = StreamArgs(P=1e5, T=273.15+2, flasher=flasher)
    
    assert mole_balance([f0, f1, f2], [p0, p1], 4)
    assert p1.stream()
    assert not mole_balance([f0, f1, f2], [p0, p1], 4)
    
    # Add in an energy stream to the fun
    f0 = StreamArgs(n=1, zs=[.1, .2, .3, .4], P=1e5, T=25.0+273.15, flasher=flasher)
    f1 = StreamArgs(m=.01, ws=[.4, .3, .2, .1], P=1e5, flasher=flasher)
    f2 = StreamArgs(Q=.01, T=310.0, Q_TP=(298.15, 101325.0, 'g'), zs=[.25, .25, .25, .25], P=1e5, flasher=flasher)
    f3 = EnergyStream(Q=2e2)
    p0 = StreamArgs(ns=[.01, .01, .01, .01], P=1e5, T=273.15, flasher=flasher)
    p1 = StreamArgs(P=1e5, T=273.15+2, flasher=flasher)
    
    
    assert mole_balance([f0, f1, f2], [p0, p1], 4)
    assert energy_balance([f0, f1, f2, f3], [p0, p1])
    assert f0.energy_calc
    assert f1.energy_calc
    assert f2.energy_calc
    assert p0.energy_calc
    assert p1.energy_calc


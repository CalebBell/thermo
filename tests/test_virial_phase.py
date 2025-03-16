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
SOFTWARE.
'''

import json
import pickle
from math import *

import numpy as np
import pytest
from chemicals.utils import rho_to_Vm, Vm_to_rho
from fluids.constants import *
from fluids.numerics import assert_close, assert_close1d, assert_close2d, assert_close3d, derivative, hessian, jacobian, normalize

from thermo import EXP_POLY_FIT
from thermo.bulk import *
from thermo.chemical_package import ChemicalConstantsPackage, PropertyCorrelationsPackage
from thermo.coolprop import PropsSI, has_CoolProp
from thermo.eos import *
from thermo.eos_mix import *
from thermo.equilibrium import *
from thermo.heat_capacity import *
from thermo.interface import *
from thermo.phase_change import *
from thermo.phases import *
from thermo.phases.phase_utils import fugacities_direct, lnphis_direct
from thermo.regular_solution import RegularSolution
from thermo.thermal_conductivity import *
from thermo.unifac import DOUFIP2016, DOUFSG, PSRKIP, PSRKSG, UFIP, UFSG, UNIFAC, VTPRIP, VTPRSG
from thermo.uniquac import UNIQUAC
from thermo.utils import LINEAR
from thermo.vapor_pressure import VaporPressure
from thermo.viscosity import *
from thermo.volume import *
from thermo.wilson import Wilson
from thermo.phases.virial_phase import VIRIAL_CROSS_B_TARAKAD_DANNER
from thermo.phases.ceos import CEOSGas

def test_store_load_VirialCSP():
    Tcs = [190.564]
    Pcs = [4599000.0]
    omegas = [0.008]
    Vcs = [9.86e-05]
    model = VirialCSP(T=300.0, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas, B_model=VIRIAL_B_PITZER_CURL, C_model=VIRIAL_C_ZERO)

    # Run the various checks for storing/loading
    model_copy = eval(repr(model))
    assert model_copy.B_pures() == model.B_pures()
    assert model_copy.C_pures() == model.C_pures()

    model_pickle = pickle.loads(pickle.dumps(model))
    assert model_pickle == model
    assert model_pickle.model_hash() == model.model_hash()
    assert VirialCSP.from_json(json.loads(json.dumps(model.as_json()))) == model

def test_store_load_VirialGas():
    Tcs = [190.564]
    Pcs = [4599000.0]
    omegas = [0.008]
    Vcs = [9.86e-05]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593]))]
    model = VirialCSP(T=300.0, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas, B_model=VIRIAL_B_PITZER_CURL, C_model=VIRIAL_C_ZERO)
    zs = [1]
    PT = VirialGas(model, HeatCapacityGases=HeatCapacityGases, T=300.0, P=1e5, zs=zs)

    check_virial_temperature_consistency_T_calls(PT, [320, 800])

    # Run the various checks for storing/loading
    phase_copy = eval(repr(PT))
    assert phase_copy.B() == PT.B()
    assert phase_copy.C() == PT.C()
    assert phase_copy == PT
    assert hash(phase_copy) == hash(PT)
    assert phase_copy.model_hash() == PT.model_hash()
    assert phase_copy.state_hash() == PT.state_hash()

    # Pickle checks
    phase_copy = pickle.loads(pickle.dumps(PT))
    assert phase_copy.B() == PT.B()
    assert phase_copy.C() == PT.C()
    assert phase_copy == PT
    assert hash(phase_copy) == hash(PT)
    assert phase_copy.model_hash() == PT.model_hash()
    assert phase_copy.state_hash() == PT.state_hash()

    # Simple json check
    phase_copy = VirialGas.from_json(json.loads(json.dumps(PT.as_json())))
    assert phase_copy.B() == PT.B()
    assert phase_copy.C() == PT.C()
    assert phase_copy == PT
    assert hash(phase_copy) == hash(PT)
    assert phase_copy.model_hash() == PT.model_hash()
    assert phase_copy.state_hash() == PT.state_hash()


def test_virial_phase_pure_B_only_pitzer_curl():
    Tcs = [190.564]
    Pcs = [4599000.0]
    omegas = [0.008]
    Vcs = [9.86e-05]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593]))]
    model = VirialCSP(T=300.0, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas, B_model=VIRIAL_B_PITZER_CURL, C_model=VIRIAL_C_ZERO)
    zs = [1]
    PT = VirialGas(model, HeatCapacityGases=HeatCapacityGases, T=300.0, P=1e5, zs=zs)

    assert_close(PT.rho(), 40.159125929164205, rtol=1e-13)
    assert_close(PT.V(), 0.02490094036817131, rtol=1e-13)
    assert_close(PT.B(), -4.2375251149270624e-05, rtol=1e-14)

    assert_close(PT.dB_dT(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).B(), PT.T, PT.T*3e-7), rtol=1e-7)
    assert_close(PT.d2B_dT2(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).dB_dT(), PT.T, PT.T*3e-7), rtol=1e-7)
    assert_close(PT.d3B_dT3(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).d2B_dT2(), PT.T, PT.T*3e-7), rtol=1e-7)
    assert_close(PT.dC_dT(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).C(), PT.T, PT.T*3e-7), rtol=1e-7)
    assert_close(PT.d2C_dT2(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).dC_dT(), PT.T, PT.T*3e-7), rtol=1e-7)
    assert_close(PT.d3C_dT3(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).d2C_dT2(), PT.T, PT.T*3e-7), rtol=1e-7)


    assert_close(PT.to_TP_zs(T=PT.T, P=PT.P, zs=[1]).V(), PT.V())

    TV = PT.to(T=PT.T, V=PT.V(), zs=[1])
    assert_close(TV.P, PT.P, rtol=1e-13)

    PV = PT.to(P=PT.P, V=PT.V(), zs=[1])
    assert_close(PV.T, 300, rtol=1e-13)

    PT2 = PT.to(P=PT.P, T=PT.T, zs=[1])
    assert_close(PT2.V(), 0.02490094036817131, rtol=1e-13)
    assert_close(PT2.B(), -4.2375251149270624e-05, rtol=1e-14)

    assert_close(PT.dP_dT(), 334.86796331728914, rtol=1e-13)
    assert_close(PT.dP_dT(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).P, PT.T, PT.T*1e-7), rtol=1e-10)

    assert_close(PT.dP_dV(), -4009066.851663165, rtol=1e-13)
    assert_close(PT.dP_dV(), derivative(lambda V: PT.to(T=PT.T, V=V, zs=[1]).P, PT.V(), PT.V()*1e-7), rtol=1e-8)

    assert_close(PT.d2P_dTdV(), -13486.814969771252, rtol=1e-13)
    assert_close(PT.d2P_dTdV(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).dP_dV(), PT.T, PT.T*1e-7), rtol=1e-9)

    assert_close(PT.d2P_dV2(), 321451403.1386218, rtol=1e-13)
    assert_close(PT.d2P_dV2(), derivative(lambda V: PT.to(T=PT.T, V=V, zs=[1]).dP_dV(), PT.V(), PT.V()*1e-7), rtol=1e-8)

    assert_close(PT.d2P_dT2(), -0.0020770847374848075, rtol=1e-13)
    assert_close(PT.d2P_dT2(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).dP_dT(), PT.T, PT.T*8e-7), rtol=1e-6)

    # Poling equation is for negative of H_dep
    assert_close(PT.H_dep(), -15.708867544147836, rtol=1e-13)
    H_dep_Poling = -(-(PT.B() - PT.T*PT.dB_dT())/PT.V())*R*PT.T
    assert_close(PT.H_dep(), H_dep_Poling, rtol=1e-13)


    assert_close(PT.dH_dep_dT(), 0.03958097657787867, rtol=1e-13)
    assert_close(PT.dH_dep_dT(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).H_dep(), PT.T, PT.T*3e-7), rtol=1e-7)

    assert_close(PT.dG_dep_dT(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).G_dep(), PT.T, PT.T*3e-7), rtol=1e-7)

    S_dep_Poling = -((PT.B() + PT.T*PT.dB_dT())/PT.V()- log(PT.Z()))*R
    assert_close(PT.S_dep(), -0.03822578258348812, rtol=1e-13)
    assert_close(PT.S_dep(), S_dep_Poling, rtol=1e-13)

    assert_close(PT.dS_dep_dT(), 0.0001793175995307205, rtol=1e-13)
    assert_close(PT.dS_dep_dT(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).S_dep(), PT.T, PT.T*3e-7), rtol=1e-7)

def test_virial_phase_ternary_B_only_pitzer_curl():
    CASs = ['7727-37-9', '74-82-8', '124-38-9']
    atomss = [{'N': 2}, {'C': 1, 'H': 4}, {'C': 1, 'O': 2}]
    Tcs = [126.2, 190.564, 304.2]
    Tcs_np = np.array(Tcs)
    Pcs = [3394387.5, 4599000.0, 7376460.0]
    Pcs_np = np.array(Pcs)
    Vcs = [8.95e-05, 9.86e-05, 9.4e-05]
    Vcs_np = np.array(Vcs)
    omegas = [0.04, 0.008, 0.2252]
    omegas_np = np.array(omegas)
    N = 3

    HeatCapacityGases = [HeatCapacityGas(CASRN="7727-37-9", MW=28.0134, similarity_variable=0.07139440410660612, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
     HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="124-38-9", MW=44.0095, similarity_variable=0.0681671002851657, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644]))]

    # Specifically test without kijs
    kijs = [[0.0]*3 for _ in range(N)]
    kijs_np = np.array(kijs)
    T = 300

    model = VirialCSP(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                  B_model=VIRIAL_B_ABBOTT,
                                  cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                                  cross_B_model_kijs=kijs,
                                  C_model=VIRIAL_C_ZERO)
    model_np = VirialCSP(T=T, Tcs=Tcs_np, Pcs=Pcs_np, Vcs=Vcs_np, omegas=omegas_np,
                              B_model=VIRIAL_B_ABBOTT,
                              cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                              cross_B_model_kijs=kijs_np,
                              C_model=VIRIAL_C_ZERO)


    P = 1e5
    zs = [.02, .92, .06]
    zs_np = np.array(zs)
    gas = VirialGas(model=model, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='theory',
                T=T, P=P, zs=zs)
    gas_np = VirialGas(model=model_np, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='theory',
                T=T, P=P, zs=zs_np)


    # Test some basics
    B, dB, d2B, d3B =  -4.413104126753512e-05, 3.9664296537847194e-07,  -3.488354169403768e-09,  4.274046794787785e-11
    assert_close(gas.B(), B, rtol=1e-13)
    assert_close(gas.dB_dT(), dB, rtol=1e-13)
    assert_close(gas.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas.d3B_dT3(), d3B, rtol=1e-13)
    assert_close(gas_np.B(), B, rtol=1e-13)
    assert_close(gas_np.dB_dT(), dB, rtol=1e-13)
    assert_close(gas_np.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas_np.d3B_dT3(), d3B, rtol=1e-13)


    for val in (gas.B(), gas.dB_dT(), gas.d2B_dT2(), gas.d3B_dT3(),
                gas_np.B(), gas_np.dB_dT(), gas_np.d2B_dT2(), gas_np.d3B_dT3()):
        assert type(val) is float


    for val in (gas.C(), gas.dC_dT(), gas.d2C_dT2(), gas.d3C_dT3(),
                gas_np.C(), gas_np.dC_dT(), gas_np.d2C_dT2(), gas_np.d3C_dT3()):
        assert val == 0.0
        assert type(val) is float

    V = 0.024899178456922296
    assert_close(gas.V(), V, rtol=1e-13)
    assert_close(gas_np.V(), V, rtol=1e-13)


    # Check the B and T derivatives
    assert_close(gas.dB_dT(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).B(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d2B_dT2(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).dB_dT(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d3B_dT3(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).d2B_dT2(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.dC_dT(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).C(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d2C_dT2(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).dC_dT(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d3C_dT3(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).d2C_dT2(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas_np.dB_dT(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).B(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d2B_dT2(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).dB_dT(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d3B_dT3(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).d2B_dT2(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.dC_dT(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).C(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d2C_dT2(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).dC_dT(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d3C_dT3(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).d2C_dT2(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)


    # Mole fraction derivatives of B, including its temperature derivatives
    dB_dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).B(), zs, perturbation=1e-7)
    dB_dzs_expect = [-4.1226874244236366e-05, -8.550397384968317e-05, -0.0001462314851412832]
    assert_close1d(gas.dB_dzs(), dB_dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.dB_dzs(), dB_dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.dB_dzs(), np.ndarray)
    assert_close1d(dB_dzs, gas.dB_dzs())

    d2B_dTdzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).dB_dT(), zs, perturbation=1e-7)
    d2B_dTdzs_expect = [5.260937128431123e-07, 7.722240212573212e-07, 1.2052992823891033e-06]
    assert_close1d(gas.d2B_dTdzs(), d2B_dTdzs_expect, rtol=1e-13)
    assert_close1d(gas_np.d2B_dTdzs(), d2B_dTdzs_expect, rtol=1e-13)
    assert isinstance(gas_np.d2B_dTdzs(), np.ndarray)
    assert_close1d(d2B_dTdzs, gas.d2B_dTdzs())

    d3B_dT2dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).d2B_dT2(), zs, perturbation=1e-7)
    d3B_dT2dzs_expect = [-4.596636740922754e-09, -6.7485932495824905e-09, -1.1267830239553159e-08]
    assert_close1d(gas.d3B_dT2dzs(), d3B_dT2dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.d3B_dT2dzs(), d3B_dT2dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.d3B_dT2dzs(), np.ndarray)
    assert_close1d(d3B_dT2dzs, gas.d3B_dT2dzs())

    d4B_dT3dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).d3B_dT3(), zs, perturbation=1e-7)
    d4B_dT3dzs_expect = [5.580371109330211e-11, 8.195352679443102e-11, 1.4946028371688516e-10]
    assert_close1d(gas.d4B_dT3dzs(), d4B_dT3dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.d4B_dT3dzs(), d4B_dT3dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.d4B_dT3dzs(), np.ndarray)
    assert_close1d(d4B_dT3dzs, gas.d4B_dT3dzs())


    # Second mole fraction derivatives
    d2B_dzizjs = hessian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).B(), zs, perturbation=125e-5)
    d2B_dzizjs_expect = [[-1.0639357784985337e-05, -3.966321845899801e-05, -7.53987684376414e-05], [-3.966321845899801e-05, -8.286257232134107e-05, -0.00014128571574782375], [-7.53987684376414e-05, -0.00014128571574782375, -0.00024567752140887547]]
    assert_close2d(d2B_dzizjs_expect, gas.d2B_dzizjs(), rtol=1e-13)
    assert_close1d(gas_np.d2B_dzizjs(), d2B_dzizjs_expect, rtol=1e-13)
    assert_close2d(d2B_dzizjs, gas.d2B_dzizjs(), rtol=1e-5)
    assert isinstance(gas_np.d2B_dzizjs(), np.ndarray)

    d3B_dTdzizjs = hessian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).dB_dT(), zs, perturbation=125e-5)
    d3B_dTdzizjs_expect = [[3.4972455964751027e-07, 5.136445106798038e-07, 7.757711970790444e-07], [5.136445106798038e-07, 7.522549586697655e-07, 1.1646094844590136e-06], [7.757711970790444e-07, 1.1646094844590136e-06, 1.9723855457538296e-06]]
    assert_close2d(d3B_dTdzizjs_expect, gas.d3B_dTdzizjs(), rtol=1e-13)
    assert_close1d(gas_np.d3B_dTdzizjs(), d3B_dTdzizjs_expect, rtol=1e-13)
    assert_close2d(d3B_dTdzizjs, gas.d3B_dTdzizjs(), rtol=1e-6)
    assert isinstance(gas_np.d3B_dTdzizjs(), np.ndarray)

    d4B_dT2dzizjs = hessian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).d2B_dT2(), zs, perturbation=125e-5)
    d4B_dT2dzizjs_expect = [[-3.0445377873470256e-09, -4.4720495115515925e-09, -7.024340575805803e-09], [-4.4720495115515925e-09, -6.536647166125688e-09, -1.0757281108597095e-08], [-7.024340575805803e-09, -1.0757281108597095e-08, -2.051074680212858e-08]]
    assert_close2d(d4B_dT2dzizjs_expect, gas.d4B_dT2dzizjs(), rtol=1e-13)
    assert_close1d(gas_np.d4B_dT2dzizjs(), d4B_dT2dzizjs_expect, rtol=1e-13)
    assert_close2d(d4B_dT2dzizjs, gas.d4B_dT2dzizjs(), rtol=1e-6)
    assert isinstance(gas_np.d4B_dT2dzizjs(), np.ndarray)

    # Third mole fraction derivatives
    d3B_dzizjzks_expect = [[[0.0]*N for _ in range(N)] for _ in range(N)]
    assert_close3d(d3B_dzizjzks_expect, gas.d3B_dzizjzks(), atol=0)
    assert_close3d(d3B_dzizjzks_expect, gas.d4B_dTdzizjzks(), atol=0)
    assert_close3d(d3B_dzizjzks_expect, gas.d5B_dT2dzizjzks(), atol=0)
    assert_close3d(d3B_dzizjzks_expect, gas.d6B_dT3dzizjzks(), atol=0)

    assert_close3d(d3B_dzizjzks_expect, gas_np.d3B_dzizjzks(), atol=0)
    assert isinstance(gas_np.d3B_dzizjzks(), np.ndarray)
    assert_close3d(d3B_dzizjzks_expect, gas_np.d4B_dTdzizjzks(), atol=0)
    assert isinstance(gas_np.d4B_dTdzizjzks(), np.ndarray)
    assert_close3d(d3B_dzizjzks_expect, gas_np.d5B_dT2dzizjzks(), atol=0)
    assert isinstance(gas_np.d5B_dT2dzizjzks(), np.ndarray)
    assert_close3d(d3B_dzizjzks_expect, gas_np.d6B_dT3dzizjzks(), atol=0)
    assert isinstance(gas_np.d6B_dT3dzizjzks(), np.ndarray)


    # Mole number derivatives
    def dB_dns_to_jac(ns):
        zs = normalize(ns)
        return gas.to(T=T, P=P, zs=zs).B()

    dB_dns = jacobian(dB_dns_to_jac, zs, perturbation=1e-7)
    dB_dns_expect = [4.7035208290833874e-05, 2.7581086853870686e-06, -5.796940260621295e-05]
    assert_close1d(gas.dB_dns(), dB_dns_expect, rtol=1e-13)
    assert_close1d(gas_np.dB_dns(), dB_dns_expect, rtol=1e-13)
    assert isinstance(gas_np.dB_dns(), np.ndarray)
    assert_close1d(dB_dns, gas.dB_dns())

    # Partial derivatives
    dnB_dns_expect = [2.904167023298754e-06, -4.137293258214805e-05, -0.00010210044387374808]
    assert_close1d(gas.dnB_dns(), dnB_dns_expect, rtol=1e-13)
    assert_close1d(gas_np.dnB_dns(), dnB_dns_expect, rtol=1e-13)
    assert isinstance(gas_np.dnB_dns(), np.ndarray)

def test_virial_phase_ternary_B_only_pitzer_curl_no_interactions():
    CASs = ['7727-37-9', '74-82-8', '124-38-9']
    atomss = [{'N': 2}, {'C': 1, 'H': 4}, {'C': 1, 'O': 2}]
    Tcs = [126.2, 190.564, 304.2]
    Tcs_np = np.array(Tcs)
    Pcs = [3394387.5, 4599000.0, 7376460.0]
    Pcs_np = np.array(Pcs)
    Vcs = [8.95e-05, 9.86e-05, 9.4e-05]
    Vcs_np = np.array(Vcs)
    omegas = [0.04, 0.008, 0.2252]
    omegas_np = np.array(omegas)
    N = 3

    HeatCapacityGases = [HeatCapacityGas(CASRN="7727-37-9", MW=28.0134, similarity_variable=0.07139440410660612, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
     HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="124-38-9", MW=44.0095, similarity_variable=0.0681671002851657, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644]))]

    # Specifically test without kijs
    kijs = [[0.0]*3 for _ in range(N)]
    kijs_np = np.array(kijs)
    T = 300

    model = VirialCSP(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                  B_model=VIRIAL_B_ABBOTT,
                                  cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                                  cross_B_model_kijs=kijs,
                                  C_model=VIRIAL_C_ZERO)
    model_np = VirialCSP(T=T, Tcs=Tcs_np, Pcs=Pcs_np, Vcs=Vcs_np, omegas=omegas_np,
                              B_model=VIRIAL_B_ABBOTT,
                              cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                              cross_B_model_kijs=kijs_np,
                              C_model=VIRIAL_C_ZERO)


    P = 1e5
    zs = [.02, .92, .06]
    zs_np = np.array(zs)
    gas = VirialGas(model=model, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='linear',
                T=T, P=P, zs=zs)
    gas_np = VirialGas(model=model_np, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='linear',
                T=T, P=P, zs=zs_np)


    # Test some basics
    B, dB, d2B, d3B = (-4.559350248793301e-05, 4.0870609295718214e-07, -3.6526254783551437e-09, 4.574694317862753e-11)
    assert_close(gas.B(), B, rtol=1e-13)
    assert_close(gas.dB_dT(), dB, rtol=1e-13)
    assert_close(gas.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas.d3B_dT3(), d3B, rtol=1e-13)
    assert_close(gas_np.B(), B, rtol=1e-13)
    assert_close(gas_np.dB_dT(), dB, rtol=1e-13)
    assert_close(gas_np.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas_np.d3B_dT3(), d3B, rtol=1e-13)


    for val in (gas.B(), gas.dB_dT(), gas.d2B_dT2(), gas.d3B_dT3(),
                gas_np.B(), gas_np.dB_dT(), gas_np.d2B_dT2(), gas_np.d3B_dT3()):
        assert type(val) is float


    for val in (gas.C(), gas.dC_dT(), gas.d2C_dT2(), gas.d3C_dT3(),
                gas_np.C(), gas_np.dC_dT(), gas_np.d2C_dT2(), gas_np.d3C_dT3()):
        assert val == 0.0
        assert type(val) is float

    V = 0.024897710706483875
    assert_close(gas.V(), V, rtol=1e-13)
    assert_close(gas_np.V(), V, rtol=1e-13)


    # Check the B and T derivatives
    assert_close(gas.dB_dT(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).B(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d2B_dT2(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).dB_dT(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d3B_dT3(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).d2B_dT2(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.dC_dT(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).C(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d2C_dT2(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).dC_dT(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d3C_dT3(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).d2C_dT2(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas_np.dB_dT(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).B(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d2B_dT2(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).dB_dT(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d3B_dT3(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).d2B_dT2(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.dC_dT(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).C(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d2C_dT2(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).dC_dT(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d3C_dT3(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).d2C_dT2(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)


    # Mole fraction derivatives of B, including its temperature derivatives
    dB_dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).B(), zs, perturbation=1e-7)
    dB_dzs_expect =  [-5.3196788924926625e-06, -4.143128616067054e-05, -0.00012283876070443773]
    assert_close1d(gas.dB_dzs(), dB_dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.dB_dzs(), dB_dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.dB_dzs(), np.ndarray)
    assert_close1d(dB_dzs, gas.dB_dzs())

    d2B_dTdzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).dB_dT(), zs, perturbation=1e-7)
    d2B_dTdzs_expect =  [1.7486227982375503e-07, 3.761274793348827e-07, 9.861927728769148e-07]
    assert_close1d(gas.d2B_dTdzs(), d2B_dTdzs_expect, rtol=1e-13)
    assert_close1d(gas_np.d2B_dTdzs(), d2B_dTdzs_expect, rtol=1e-13)
    assert isinstance(gas_np.d2B_dTdzs(), np.ndarray)
    assert_close1d(d2B_dTdzs, gas.d2B_dTdzs())

    d3B_dT2dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).d2B_dT2(), zs, perturbation=4e-7)
    d3B_dT2dzs_expect = [-1.5222688936735118e-09, -3.268323583062844e-09, -1.025537340106429e-08]
    assert_close1d(gas.d3B_dT2dzs(), d3B_dT2dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.d3B_dT2dzs(), d3B_dT2dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.d3B_dT2dzs(), np.ndarray)
    assert_close1d(d3B_dT2dzs, gas.d3B_dT2dzs())

    d4B_dT3dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).d3B_dT3(), zs, perturbation=1e-7)
    d4B_dT3dzs_expect = [1.8385020623121148e-11, 3.9368119318647694e-11, 1.5267621655015376e-10]
    assert_close1d(gas.d4B_dT3dzs(), d4B_dT3dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.d4B_dT3dzs(), d4B_dT3dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.d4B_dT3dzs(), np.ndarray)
    assert_close1d(d4B_dT3dzs, gas.d4B_dT3dzs())


    # Second mole fraction derivatives
    d2B_dzizjs = hessian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).B(), zs, perturbation=125e-5)
    d2B_dzizjs_expect = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert_close2d(d2B_dzizjs_expect, gas.d2B_dzizjs(), rtol=1e-13)
    assert_close1d(gas_np.d2B_dzizjs(), d2B_dzizjs_expect, rtol=1e-13)
    assert_close2d(d2B_dzizjs, gas.d2B_dzizjs(), rtol=1e-5)
    assert isinstance(gas_np.d2B_dzizjs(), np.ndarray)

    d3B_dTdzizjs = hessian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).dB_dT(), zs, perturbation=125e-5)
    d3B_dTdzizjs_expect =  [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert_close2d(d3B_dTdzizjs_expect, gas.d3B_dTdzizjs(), rtol=1e-13)
    assert_close1d(gas_np.d3B_dTdzizjs(), d3B_dTdzizjs_expect, rtol=1e-13)
    assert_close2d(d3B_dTdzizjs, gas.d3B_dTdzizjs(), atol=1e-6)
    assert isinstance(gas_np.d3B_dTdzizjs(), np.ndarray)

    d4B_dT2dzizjs = hessian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).d2B_dT2(), zs, perturbation=125e-5)
    d4B_dT2dzizjs_expect =  [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert_close2d(d4B_dT2dzizjs_expect, gas.d4B_dT2dzizjs(), rtol=1e-13)
    assert_close1d(gas_np.d4B_dT2dzizjs(), d4B_dT2dzizjs_expect, rtol=1e-13)
    assert_close2d(d4B_dT2dzizjs, gas.d4B_dT2dzizjs(), atol=1e-6)
    assert isinstance(gas_np.d4B_dT2dzizjs(), np.ndarray)

    # Third mole fraction derivatives
    d3B_dzizjzks_expect = [[[0.0]*N for _ in range(N)] for _ in range(N)]
    assert_close3d(d3B_dzizjzks_expect, gas.d3B_dzizjzks(), atol=0)
    assert_close3d(d3B_dzizjzks_expect, gas.d4B_dTdzizjzks(), atol=0)
    assert_close3d(d3B_dzizjzks_expect, gas.d5B_dT2dzizjzks(), atol=0)
    assert_close3d(d3B_dzizjzks_expect, gas.d6B_dT3dzizjzks(), atol=0)

    assert_close3d(d3B_dzizjzks_expect, gas_np.d3B_dzizjzks(), atol=0)
    assert isinstance(gas_np.d3B_dzizjzks(), np.ndarray)
    assert_close3d(d3B_dzizjzks_expect, gas_np.d4B_dTdzizjzks(), atol=0)
    assert isinstance(gas_np.d4B_dTdzizjzks(), np.ndarray)
    assert_close3d(d3B_dzizjzks_expect, gas_np.d5B_dT2dzizjzks(), atol=0)
    assert isinstance(gas_np.d5B_dT2dzizjzks(), np.ndarray)
    assert_close3d(d3B_dzizjzks_expect, gas_np.d6B_dT3dzizjzks(), atol=0)
    assert isinstance(gas_np.d6B_dT3dzizjzks(), np.ndarray)


    # Mole number derivatives
    def dB_dns_to_jac(ns):
        zs = normalize(ns)
        return gas.to(T=T, P=P, zs=zs).B()

    dB_dns = jacobian(dB_dns_to_jac, zs, perturbation=.5e-7)
    dB_dns_expect = [4.0273823595440354e-05, 4.162216327262471e-06, -7.724525821650472e-05]
    assert_close1d(gas.dB_dns(), dB_dns_expect, rtol=1e-13)
    assert_close1d(gas_np.dB_dns(), dB_dns_expect, rtol=1e-13)
    assert isinstance(gas_np.dB_dns(), np.ndarray)
    assert_close1d(dB_dns, gas.dB_dns())

    # Partial derivatives
    dnB_dns_expect = [-5.3196788924926625e-06, -4.143128616067054e-05, -0.00012283876070443773]
    assert_close1d(gas.dnB_dns(), dnB_dns_expect, rtol=1e-13)
    assert_close1d(gas_np.dnB_dns(), dnB_dns_expect, rtol=1e-13)
    assert isinstance(gas_np.dnB_dns(), np.ndarray)


def test_virial_phase_pure_BC_pitzer_curl_orbey_vera():
    Tcs = [190.564]
    Pcs = [4599000.0]
    omegas = [0.008]
    Vcs = [9.86e-05]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593]))]
    T = 300.0
    model = VirialCSP(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas, B_model=VIRIAL_B_PITZER_CURL, C_model=VIRIAL_C_ORBEY_VERA)
    zs = [1]
    gas = PT = VirialGas(model, HeatCapacityGases=HeatCapacityGases, T=T, P=1e5, zs=zs, B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz')

    check_virial_temperature_consistency_T_calls(gas, [320, 800])

    assert_close(PT.rho(), 40.15896568252847, rtol=1e-13)
    assert_close(PT.V(), 0.02490103973058896, rtol=1e-13)
    assert_close(PT.B(), -4.2375251149270624e-05, rtol=1e-14)
    assert_close(PT.C(), 2.4658163172184193e-09, rtol=1e-14)

    assert_close(PT.dB_dT(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).B(), PT.T, PT.T*3e-7), rtol=1e-7)
    assert_close(PT.d2B_dT2(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).dB_dT(), PT.T, PT.T*3e-7), rtol=1e-7)
    assert_close(PT.d3B_dT3(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).d2B_dT2(), PT.T, PT.T*3e-7), rtol=1e-7)
    assert_close(PT.dC_dT(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).C(), PT.T, PT.T*3e-7), rtol=1e-7)
    assert_close(PT.d2C_dT2(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).dC_dT(), PT.T, PT.T*3e-7), rtol=1e-7)
    assert_close(PT.d3C_dT3(), derivative(lambda T: PT.to(T=T, P=PT.P, zs=zs).d2C_dT2(), PT.T, PT.T*3e-7), rtol=1e-7)


    assert_close(PT.to_TP_zs(T=PT.T, P=PT.P, zs=[1]).V(), PT.V())

    TV = PT.to(T=PT.T, V=PT.V(), zs=[1])
    assert_close(TV.P, PT.P, rtol=1e-13)
    assert_close(TV.B(), PT.B(), rtol=1e-13)
    assert_close(TV.C(), PT.C(), rtol=1e-13)

    PV = PT.to(P=PT.P, V=PT.V(), zs=[1])
    assert_close(PV.T, 300, rtol=1e-13)
    assert_close(PV.B(), PT.B(), rtol=1e-13)
    assert_close(PV.C(), PT.C(), rtol=1e-13)

    PT2 = PT.to(P=PT.P, T=PT.T, zs=[1])
    assert_close(PT2.B(), PT.B(), rtol=1e-13)
    assert_close(PT2.C(), PT.C(), rtol=1e-13)
    assert_close(PT2.V(), 0.02490103973058896, rtol=1e-13)
    assert_close(PT2.B(), -4.2375251149270624e-05, rtol=1e-14)
    assert_close(PT2.C(),  2.4658163172184193e-09, rtol=1e-14)

    assert_close(PT.dP_dT(), 334.8667253234988, rtol=1e-13)
    assert_close(PT.dP_dT(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).P, PT.T, PT.T*1e-7), rtol=1e-7)

    assert_close(PT.dP_dV(), -4009082.903515464, rtol=1e-13)
    assert_close(PT.dP_dV(), derivative(lambda V: PT.to(T=PT.T, V=V, zs=[1]).P, PT.V(), PT.V()*1e-7), rtol=1e-8)

    assert_close(PT.d2P_dTdV(), -13486.719326292223, rtol=1e-13)
    assert_close(PT.d2P_dTdV(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).dP_dV(), PT.T, PT.T*1e-7), rtol=1e-7)

    assert_close(PT.d2P_dV2(), 321455270.893362, rtol=1e-13)
    assert_close(PT.d2P_dV2(), derivative(lambda V: PT.to(T=PT.T, V=V, zs=[1]).dP_dV(), PT.V(), PT.V()*1e-7), rtol=1e-8)

    assert_close(PT.d2P_dT2(), -0.0020701719078863546, rtol=1e-13)
    assert_close(PT.d2P_dT2(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).dP_dT(), PT.T, PT.T*8e-7), rtol=1e-6)

    assert_close(PT.H_dep(), -15.694307202591004, rtol=1e-13)

    assert_close(PT.dH_dep_dT(), 0.03955760203920011, rtol=1e-13)
    assert_close(PT.dH_dep_dT(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).H_dep(), PT.T, PT.T*3e-7), rtol=1e-7)

    assert_close(PT.dG_dep_dT(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).G_dep(), PT.T, PT.T*3e-7), rtol=1e-7)

    assert_close(PT.S_dep(), -0.03819378035089749, rtol=1e-13)

    assert_close(PT.dS_dep_dT(), 0.00017912859867798742, rtol=1e-13)
    assert_close(PT.dS_dep_dT(), derivative(lambda T: PT.to(T=T, V=PT.V(), zs=[1]).S_dep(), PT.T, PT.T*3e-7), rtol=1e-7)


def test_virial_phase_ternary_BC_pitzer_curl_orbey_vera():
    CASs = ['7727-37-9', '74-82-8', '124-38-9']
    atomss = [{'N': 2}, {'C': 1, 'H': 4}, {'C': 1, 'O': 2}]
    Tcs = [126.2, 190.564, 304.2]
    Tcs_np = np.array(Tcs)
    Pcs = [3394387.5, 4599000.0, 7376460.0]
    Pcs_np = np.array(Pcs)
    Vcs = [8.95e-05, 9.86e-05, 9.4e-05]
    Vcs_np = np.array(Vcs)
    omegas = [0.04, 0.008, 0.2252]
    omegas_np = np.array(omegas)
    N = 3

    HeatCapacityGases = [HeatCapacityGas(CASRN="7727-37-9", MW=28.0134, similarity_variable=0.07139440410660612, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
     HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="124-38-9", MW=44.0095, similarity_variable=0.0681671002851657, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644]))]

    # Specifically test without kijs
    kijs = [[0.0]*3 for _ in range(N)]
    kijs_np = np.array(kijs)
    T = 300

    model = VirialCSP(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                  B_model=VIRIAL_B_ABBOTT,
                                  cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                                  cross_B_model_kijs=kijs,
                                  C_model=VIRIAL_C_ORBEY_VERA)
    model_np = VirialCSP(T=T, Tcs=Tcs_np, Pcs=Pcs_np, Vcs=Vcs_np, omegas=omegas_np,
                              B_model=VIRIAL_B_ABBOTT,
                              cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                              cross_B_model_kijs=kijs_np,
                              C_model=VIRIAL_C_ORBEY_VERA)


    P = 1e5
    zs = [.02, .92, .06]
    zs_np = np.array(zs)
    gas = VirialGas(model=model, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                T=T, P=P, zs=zs)
    gas_np = VirialGas(model=model_np, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                T=T, P=P, zs=zs_np)

    with pytest.raises(ValueError):
        gas.to(T=T, zs=zs)
    with pytest.raises(ValueError):
        gas.to(P=P, zs=zs)
    with pytest.raises(ValueError):
        gas.to(V=gas.V(), zs=zs)

    # Test some basics
    B, dB, d2B, d3B =  -4.413104126753512e-05, 3.9664296537847194e-07,  -3.488354169403768e-09,  4.274046794787785e-11
    assert_close(gas.B(), B, rtol=1e-13)
    assert_close(gas.dB_dT(), dB, rtol=1e-13)
    assert_close(gas.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas.d3B_dT3(), d3B, rtol=1e-13)
    assert_close(gas_np.B(), B, rtol=1e-13)
    assert_close(gas_np.dB_dT(), dB, rtol=1e-13)
    assert_close(gas_np.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas_np.d3B_dT3(), d3B, rtol=1e-13)


    for val in (gas.B(), gas.dB_dT(), gas.d2B_dT2(), gas.d3B_dT3(),
                gas_np.B(), gas_np.dB_dT(), gas_np.d2B_dT2(), gas_np.d3B_dT3()):
        assert type(val) is float


    for val in (gas.C(), gas.dC_dT(), gas.d2C_dT2(), gas.d3C_dT3(),
                gas_np.C(), gas_np.dC_dT(), gas_np.d2C_dT2(), gas_np.d3C_dT3()):
        assert type(val) is float

    V = 0.0248992803360438
    assert_close(gas.V(), V, rtol=1e-13)
    assert_close(gas_np.V(), V, rtol=1e-13)


    # Check the B and T derivatives
    assert_close(gas.dB_dT(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).B(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d2B_dT2(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).dB_dT(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d3B_dT3(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).d2B_dT2(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.dC_dT(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).C(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d2C_dT2(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).dC_dT(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d3C_dT3(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).d2C_dT2(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas_np.dB_dT(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).B(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d2B_dT2(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).dB_dT(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d3B_dT3(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).d2B_dT2(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.dC_dT(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).C(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d2C_dT2(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).dC_dT(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d3C_dT3(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).d2C_dT2(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)

    B_mat = gas.model.B_interactions()
    assert_close1d(gas.model.B_pures(), [B_mat[i][i] for i in range(N)], rtol=1e-13)

    assert_close(gas.d2T_dV2(), 6030.193454056163, rtol=1e-11)
    assert_close(gas.d2T_dV2_P(), 6030.193454056163, rtol=1e-11)
    assert_close(gas.d2V_dT2(), -3.5160077601372554e-09, rtol=1e-11)

    # Mole fraction derivatives of B, including its temperature derivatives
    dC_dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).C(), zs, perturbation=1e-7)
    dC_dzs_expect = [6.196410187720199e-09, 7.51935295282256e-09, 9.02453803131555e-09]
    assert_close1d(gas.dC_dzs(), dC_dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.dC_dzs(), dC_dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.dC_dzs(), np.ndarray)
    assert_close1d(dC_dzs, gas.dC_dzs())


    # Second mole fraction derivatives
    d2C_dzizjs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).dC_dzs(), zs, scalar=False, perturbation=1e-7)
    d2C_dzizjs_expect = [[1.0334498744432369e-08, 1.232601605536606e-08, 1.4103260493582947e-08], [1.232601605536606e-08, 1.4914908015449092e-08, 1.7841170172077197e-08], [1.4103260493582947e-08, 1.7841170172077197e-08, 2.255223824080701e-08]]
    assert_close2d(d2C_dzizjs_expect, gas.d2C_dzizjs(), rtol=1e-13)
    assert_close1d(gas_np.d2C_dzizjs(), d2C_dzizjs_expect, rtol=1e-13)
    assert_close2d(d2C_dzizjs, gas.d2C_dzizjs(), rtol=1e-5)
    assert isinstance(gas_np.d2C_dzizjs(), np.ndarray)

    # Third mole fraction derivatives
    d3C_dzizjzks = hessian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).dC_dzs(), zs, scalar=False, perturbation=3e-5)
    d3C_dzizjzks_expect = [[[8.791135059452836e-09, 1.0308422318043785e-08, 1.1248791844050504e-08], [1.0308422318043785e-08, 1.2261620654600401e-08, 1.3985943446213539e-08], [1.1248791844050504e-08, 1.3985943446213539e-08, 1.685361143642469e-08]], [[1.0308422318043785e-08, 1.2261620654600401e-08, 1.3985943446213539e-08], [1.2261620654600401e-08, 1.4794897903310516e-08, 1.7639492188523473e-08], [1.3985943446213539e-08, 1.7639492188523473e-08, 2.2218641495188888e-08]], [[1.1248791844050504e-08, 1.3985943446213539e-08, 1.685361143642469e-08], [1.3985943446213539e-08, 1.7639492188523473e-08, 2.2218641495188888e-08], [1.685361143642469e-08, 2.2218641495188888e-08, 2.9566930608412272e-08]]]
    assert_close2d(d3C_dzizjzks_expect, gas.d3C_dzizjzks(), rtol=1e-13)
    assert_close1d(gas_np.d3C_dzizjzks(), d3C_dzizjzks_expect, rtol=1e-13)
    assert_close2d(d3C_dzizjzks, gas.d3C_dzizjzks(), rtol=1e-5)
    assert isinstance(gas_np.d3C_dzizjzks(), np.ndarray)

    # Mole number derivatives
    def dC_dns_to_jac(ns):
        zs = normalize(ns)
        return gas.to(T=T, P=P, zs=zs).C()

    dC_dns = jacobian(dC_dns_to_jac, zs, perturbation=2e-7)
    dC_dns_expect =[-1.3867950145098947e-09, -6.385224940753337e-11, 1.4413328290854567e-09]
    assert_close1d(gas.dC_dns(), dC_dns_expect, rtol=1e-13)
    assert_close1d(gas_np.dC_dns(), dC_dns_expect, rtol=1e-13)
    assert isinstance(gas_np.dC_dns(), np.ndarray)
    assert_close1d(dC_dns, gas.dC_dns(), rtol=1e-6)

    # Partial derivatives
    dnC_dns_expect = [1.1409400529001358e-09, 2.463882818002497e-09, 3.969067896495487e-09]
    assert_close1d(gas.dnC_dns(), dnC_dns_expect, rtol=1e-13)
    assert_close1d(gas_np.dnC_dns(), dnC_dns_expect, rtol=1e-13)
    assert isinstance(gas_np.dnC_dns(), np.ndarray)

    # Volume derivative
    dV_dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).V(), zs, perturbation=3e-6)
    dV_dzs_expect = [-4.112328459328977e-05, -8.550402887152434e-05, -0.0001463861573387482]
    assert_close1d(gas.dV_dzs(), dV_dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.dV_dzs(), dV_dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.dV_dzs(), np.ndarray)
    assert_close1d(dV_dzs, gas.dV_dzs(), rtol=5e-6)

    d2V_dzizjs_expect = [[-1.039580364021386e-05, -3.958861196399791e-05, -7.55799645242824e-05], [-3.958861196399791e-05, -8.314138850953133e-05, -0.00014207163186822304], [-7.55799645242824e-05, -0.00014207163186822304, -0.0002473595232182522]]
    assert_close1d(gas.d2V_dzizjs(), d2V_dzizjs_expect, rtol=1e-13)
    assert_close1d(gas_np.d2V_dzizjs(), d2V_dzizjs_expect, rtol=1e-13)
    assert isinstance(gas_np.d2V_dzizjs(), np.ndarray)


    # B and C pressure derivatives at constant volume
    dB_dP_V = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).B(), gas.P, dx=gas.P*1e-6)
    assert_close(gas.dB_dP_V(), 1.1842641665356751e-09, rtol=1e-11)
    assert_close(gas.dB_dP_V(), dB_dP_V)

    d2B_dTdP_V = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).dB_dT(), gas.P, dx=gas.P*1e-6)
    assert_close(gas.d2B_dTdP_V(), -1.04152429353394e-11, rtol=1e-11)
    assert_close(gas.d2B_dTdP_V(), d2B_dTdP_V)

    d3B_dT2dP_V = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).d2B_dT2(), gas.P, dx=gas.P*1e-6)
    assert_close(gas.d3B_dT2dP_V(), 1.2761099797481876e-13, rtol=1e-11)
    assert_close(gas.d3B_dT2dP_V(), d3B_dT2dP_V)

    dC_dP_V = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).C(), gas.P, dx=gas.P*1e-6)
    assert_close(gas.dC_dP_V(), -2.5264916071048583e-14)
    assert_close(gas.dC_dP_V(), dC_dP_V)

    d2C_dTdP_V = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).dC_dT(), gas.P, dx=gas.P*1e-6)
    assert_close(gas.d2C_dTdP_V(), 2.9635255492502356e-16, rtol=1e-11)
    assert_close(gas.d2C_dTdP_V(), d2C_dTdP_V)

    d3C_dT2dP_V = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).d2C_dT2(), gas.P, dx=gas.P*1e-6)
    assert_close(gas.d3C_dT2dP_V(),-3.891861385232169e-18, rtol=1e-11)
    assert_close(gas.d3C_dT2dP_V(), d3C_dT2dP_V)

    # B and C volume derivatives at constant pressure
    dB_dV_P = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).B(), gas.V(), dx=gas.V()*1e-6)
    assert_close(gas.dB_dV_P(), 0.00474781253771208, rtol=1e-11)
    assert_close(gas.dB_dV_P(), dB_dV_P)

    d2B_dTdV_P = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).dB_dT(), gas.V(), dx=gas.V()*1e-6)
    assert_close(gas.d2B_dTdV_P(),-4.175556635845567e-05, rtol=1e-11)
    assert_close(gas.d2B_dTdV_P(), d2B_dTdV_P)

    d3B_dT2dV_P = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).d2B_dT2(), gas.V(), dx=gas.V()*1e-6)
    assert_close(gas.d3B_dT2dV_P(), 5.116029964050627e-07, rtol=1e-11)
    assert_close(gas.d3B_dT2dV_P(), d3B_dT2dV_P)

    dC_dV_P = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).C(), gas.V(), dx=gas.V()*1e-6)
    assert_close(gas.dC_dV_P(), -1.0128912845287412e-07)
    assert_close(gas.dC_dV_P(), dC_dV_P)

    d2C_dTdV_P = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).dC_dT(), gas.V(), dx=gas.V()*1e-6)
    assert_close(gas.d2C_dTdV_P(), 1.1881017897991506e-09, rtol=1e-11)
    assert_close(gas.d2C_dTdV_P(), d2C_dTdV_P)

    d3C_dT2dV_P = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).d2C_dT2(), gas.V(), dx=gas.V()*1e-6)
    assert_close(gas.d3C_dT2dV_P(), -1.5602792689316894e-11, rtol=1e-11)
    assert_close(gas.d3C_dT2dV_P(), d3C_dT2dV_P)



    # Enthalpy extra derivatives

    dH_dep_dP_V = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).H_dep(), gas.P, dx=gas.P*1e-6)
    assert_close(gas.dH_dep_dP_V(), 0.000150298196047701, rtol=1e-11)
    assert_close(gas.dH_dep_dP_V(), dH_dep_dP_V)

    dH_dep_dP_T = derivative(lambda P: gas.to(T=gas.T, P=P, zs=gas.zs).H_dep(), gas.P, dx=gas.P*6e-7)
    assert_close(gas.dH_dep_dP_T(), -0.00016339614251889556, rtol=1e-11)
    assert_close(gas.dH_dep_dP_T(), dH_dep_dP_T)

    dH_dep_dV_T = derivative(lambda V: gas.to(T=gas.T, V=V, zs=gas.zs).H_dep(), gas.V(), dx=gas.V()*6e-7)
    assert_close(gas.dH_dep_dV_T(), 655.0685868798632, rtol=1e-11)
    assert_close(gas.dH_dep_dV_T(), dH_dep_dV_T)

    dH_dV_T = derivative(lambda V: gas.to(T=gas.T, V=V, zs=gas.zs).H(), gas.V(), dx=gas.V()*6e-7)
    assert_close(gas.dH_dV_T(), 655.0685868798632, rtol=1e-11)
    assert_close(gas.dH_dV_T(), dH_dV_T)

    dH_dep_dV_P = derivative(lambda V: gas.to(P=gas.P, V=V, zs=gas.zs).H_dep(), gas.V(), dx=gas.V()*6e-7)
    assert_close(gas.dH_dep_dV_P(), 1257.6264280735763, rtol=1e-11)
    assert_close(gas.dH_dep_dV_P(), dH_dep_dV_P)

    # entropy special derivatives

    dS_dep_dP_V = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).S_dep(), gas.P, dx=gas.P*1e-6)
    assert_close(gas.dS_dep_dP_V(), 6.480190482120792e-07, rtol=1e-11)
    assert_close(gas.dS_dep_dP_V(), dS_dep_dP_V)

    dS_dP_V = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).S(), gas.P, dx=gas.P*1e-6)
    assert_close(gas.dS_dP_V(), 0.0002741160790577555, rtol=1e-11)
    assert_close(gas.dS_dP_V(), dS_dP_V)

    dS_dep_dP_T = derivative(lambda P: gas.to(T=gas.T, P=P, zs=gas.zs).S_dep(), gas.P, dx=gas.P*6e-7)
    assert_close(gas.dS_dep_dP_T(),-3.9762874700991293e-07, rtol=1e-11)
    assert_close(gas.dS_dep_dP_T(), dS_dep_dP_T)

    dS_dP_T = derivative(lambda P: gas.to(T=gas.T, P=P, zs=gas.zs).S(), gas.P, dx=gas.P*6e-7)
    assert_close(gas.dS_dP_T(), -8.354225492854231e-05, rtol=1e-11)
    assert_close(gas.dS_dP_T(), dS_dP_T)


    dS_dep_dV_T = derivative(lambda V: gas.to(T=gas.T, V=V, zs=gas.zs).S_dep(), gas.V(), dx=gas.V()*6e-7)
    assert_close(gas.dS_dep_dV_T(), 1.5941263813891449, rtol=1e-11)
    assert_close(gas.dS_dep_dV_T(), dS_dep_dV_T)

    dS_dV_T = derivative(lambda V: gas.to(T=gas.T, V=V, zs=gas.zs).S(), gas.V(), dx=gas.V()*6e-7)
    assert_close(gas.dS_dV_T(), 334.92777759101733, rtol=1e-11)
    assert_close(gas.dS_dV_T(), dS_dV_T)

    dS_dep_dV_P = derivative(lambda V: gas.to(P=gas.P, V=V, zs=gas.zs).S_dep(), gas.V(), dx=gas.V()*6e-7)
    assert_close(gas.dS_dep_dV_P(), 4.192088093578603, rtol=1e-11)
    assert_close(gas.dS_dep_dV_P(), dS_dep_dV_P)

    # This is WRONG, the Cp T integral needs work, TODO
    # dS_dV_P = derivative(lambda V: gas.to(P=gas.P, V=V, zs=gas.zs).S(), gas.V(), dx=gas.V()*6e-3)
    # assert_close(gas.dS_dV_P(), 427649.25865266385, rtol=1e-11)
    # assert_close(gas.dS_dV_P(), dS_dV_P)

    dS_dep_dT_V = derivative(lambda T: gas.to(T=T, V=gas.V(), zs=gas.zs).S_dep(), gas.T, dx=gas.T*6e-7)
    assert_close(gas.dS_dep_dT_V(), 0.00021703957965431793, rtol=1e-11)
    assert_close(gas.dS_dep_dT_V(), dS_dep_dT_V)

    dS_dT_V = derivative(lambda T: gas.to(T=T, V=gas.V(), zs=gas.zs).S(), gas.T, dx=gas.T*6e-7)
    assert_close(gas.dS_dT_V(), 0.09180908916077761, rtol=1e-11)
    assert_close(gas.dS_dT_V(), dS_dT_V)

    # Gibbs composition derivative
    dG_dep_dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).G_dep(), zs, perturbation=3e-6)
    assert_close1d(dG_dep_dzs, gas.dG_dep_dzs(), rtol=5e-6)

    dG_dep_dzs_expect = [-4.117525515170916, -8.55041759275306, -14.630898319262041]
    assert_close1d(gas.dG_dep_dzs(), dG_dep_dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.dG_dep_dzs(), dG_dep_dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.dG_dep_dzs(), np.ndarray)

    dnG_dep_dns_expect = [0.2971307072097611, -4.135761370372383, -10.216242096881363]
    assert_close1d(gas.dnG_dep_dns(), dnG_dep_dns_expect, rtol=1e-13)
    assert_close1d(gas_np.dnG_dep_dns(), dnG_dep_dns_expect, rtol=1e-13)
    assert isinstance(gas_np.dnG_dep_dns(), np.ndarray)

    # lnphis call
    def to_jac(ns):
        zs = [i/sum(ns) for i in ns]
        return sum(ns)*gas.to(T=T, P=P, zs=zs).lnphi()
    lnphis = jacobian(to_jac, zs, perturbation=.7e-6)
    lnphis_expect = [0.00011912203303876263, -0.0016580591996980612, -0.004095771655595198]
    assert_close1d(gas.lnphis(), lnphis_expect, rtol=1e-13)
    assert_close1d(gas_np.lnphis(), lnphis_expect, rtol=1e-13)
    assert isinstance(gas_np.lnphis(), np.ndarray)
    assert_close1d(lnphis, gas.lnphis(), rtol=1.5e-6)


# test_virial_phase_ternary_BC_pitzer_curl_orbey_vera()


def test_virial_phase_ternary_BC_pitzer_curl_orbey_vera_no_interactions():
    CASs = ['7727-37-9', '74-82-8', '124-38-9']
    atomss = [{'N': 2}, {'C': 1, 'H': 4}, {'C': 1, 'O': 2}]
    Tcs = [126.2, 190.564, 304.2]
    Tcs_np = np.array(Tcs)
    Pcs = [3394387.5, 4599000.0, 7376460.0]
    Pcs_np = np.array(Pcs)
    Vcs = [8.95e-05, 9.86e-05, 9.4e-05]
    Vcs_np = np.array(Vcs)
    omegas = [0.04, 0.008, 0.2252]
    omegas_np = np.array(omegas)
    N = 3

    HeatCapacityGases = [HeatCapacityGas(CASRN="7727-37-9", MW=28.0134, similarity_variable=0.07139440410660612, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
     HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="124-38-9", MW=44.0095, similarity_variable=0.0681671002851657, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644]))]

    # Specifically test without kijs
    kijs = [[0.0]*3 for _ in range(N)]
    kijs_np = np.array(kijs)
    T = 300

    model = VirialCSP(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                  B_model=VIRIAL_B_ABBOTT,
                                  cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                                  cross_B_model_kijs=kijs,
                                  C_model=VIRIAL_C_ORBEY_VERA)
    model_np = VirialCSP(T=T, Tcs=Tcs_np, Pcs=Pcs_np, Vcs=Vcs_np, omegas=omegas_np,
                              B_model=VIRIAL_B_ABBOTT,
                              cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                              cross_B_model_kijs=kijs_np,
                              C_model=VIRIAL_C_ORBEY_VERA)


    P = 1e5
    zs = [.02, .92, .06]
    zs_np = np.array(zs)
    gas = VirialGas(model=model, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='linear', C_mixing_rule='linear',
                T=T, P=P, zs=zs)
    gas_np = VirialGas(model=model_np, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='linear',C_mixing_rule='linear',
                T=T, P=P, zs=zs_np)


    # Test some basics
    B, dB, d2B, d3B = (-4.559350248793301e-05, 4.0870609295718214e-07, -3.6526254783551437e-09, 4.574694317862753e-11)
    assert_close(gas.B(), B, rtol=1e-13)
    assert_close(gas.dB_dT(), dB, rtol=1e-13)
    assert_close(gas.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas.d3B_dT3(), d3B, rtol=1e-13)
    assert_close(gas_np.B(), B, rtol=1e-13)
    assert_close(gas_np.dB_dT(), dB, rtol=1e-13)
    assert_close(gas_np.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas_np.d3B_dT3(), d3B, rtol=1e-13)

    B_mat = gas.model.B_interactions()
    assert_close1d(gas.model.B_pures(), [B_mat[i][i] for i in range(N)], rtol=1e-13)

    check_virial_temperature_consistency_T_calls(gas, [320, 800])

    for val in (gas.B(), gas.dB_dT(), gas.d2B_dT2(), gas.d3B_dT3(),
                gas_np.B(), gas_np.dB_dT(), gas_np.d2B_dT2(), gas_np.d3B_dT3()):
        assert type(val) is float


    for val in (gas.C(), gas.dC_dT(), gas.d2C_dT2(), gas.d3C_dT3(),
                gas_np.C(), gas_np.dC_dT(), gas_np.d2C_dT2(), gas_np.d3C_dT3()):
        assert type(val) is float

    V = 0.0248978152556876
    assert_close(gas.V(), V, rtol=1e-13)
    assert_close(gas_np.V(), V, rtol=1e-13)


    # Check the B and T derivatives
    assert_close(gas.dB_dT(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).B(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d2B_dT2(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).dB_dT(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d3B_dT3(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).d2B_dT2(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.dC_dT(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).C(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d2C_dT2(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).dC_dT(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas.d3C_dT3(), derivative(lambda T: gas.to(T=T, P=P, zs=zs).d2C_dT2(), gas.T, gas.T*3e-7), rtol=1e-7)
    assert_close(gas_np.dB_dT(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).B(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d2B_dT2(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).dB_dT(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d3B_dT3(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).d2B_dT2(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.dC_dT(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).C(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d2C_dT2(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).dC_dT(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)
    assert_close(gas_np.d3C_dT3(), derivative(lambda T: gas_np.to(T=T, P=P, zs=zs).d2C_dT2(), gas_np.T, gas_np.T*3e-7), rtol=1e-7)


    # Mole fraction derivatives of C
    dC_dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).C(), zs, perturbation=1e-7)
    dC_dzs_expect =  [1.4651891765754702e-09, 2.4658163172184143e-09, 4.927821768068704e-09]
    assert_close1d(gas.dC_dzs(), dC_dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.dC_dzs(), dC_dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.dC_dzs(), np.ndarray)
    assert_close1d(dC_dzs, gas.dC_dzs())


    # Second mole fraction derivatives
    d2C_dzizjs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).dC_dzs(), zs, scalar=False, perturbation=125e-5)
    d2C_dzizjs_expect = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert_close2d(d2C_dzizjs_expect, gas.d2C_dzizjs(), rtol=1e-13)
    assert_close1d(gas_np.d2C_dzizjs(), d2C_dzizjs_expect, rtol=1e-13)
    assert_close2d(d2C_dzizjs, gas.d2C_dzizjs(), rtol=1e-6)
    assert isinstance(gas_np.d2C_dzizjs(), np.ndarray)


    # Third mole fraction derivatives
    d3C_dzizjzks_expect = [[[0.0]*N for _ in range(N)] for _ in range(N)]
    assert_close3d(d3C_dzizjzks_expect, gas.d3C_dzizjzks(), atol=0)


    # Mole number derivatives
    def dC_dns_to_jac(ns):
        zs = normalize(ns)
        return gas.to(T=T, P=P, zs=zs).C()

    dC_dns = jacobian(dC_dns_to_jac, zs, perturbation=.5e-7)
    dC_dns_expect = [-1.1283349248811026e-09, -1.277077842381585e-10, 2.334297666612131e-09]
    assert_close1d(gas.dC_dns(), dC_dns_expect, rtol=1e-13)
    assert_close1d(gas_np.dC_dns(), dC_dns_expect, rtol=1e-13)
    assert isinstance(gas_np.dC_dns(), np.ndarray)
    assert_close1d(dC_dns, gas.dC_dns(), rtol=5e-7)

    # Partial derivatives
    dnC_dns_expect = [1.4651891765754702e-09, 2.4658163172184143e-09, 4.927821768068704e-09]
    assert_close1d(gas.dnC_dns(), dnC_dns_expect, rtol=1e-13)
    assert_close1d(gas_np.dnC_dns(), dnC_dns_expect, rtol=1e-13)
    assert isinstance(gas_np.dnC_dns(), np.ndarray)

    dV_dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).V(), zs, perturbation=20e-6)
    dV_dzs_expect = [-5.28010263028751e-06, -4.14836598615557e-05, -0.00012309010527128904]
    assert_close1d(gas.dV_dzs(), dV_dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.dV_dzs(), dV_dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.dV_dzs(), np.ndarray)
    assert_close1d(dV_dzs, gas.dV_dzs(), rtol=5e-6)

    dG_dep_dzs = jacobian(lambda zs: gas.to(T=gas.T, P=gas.P, zs=zs).G_dep(), zs, perturbation=3e-6)
    assert_close1d(dG_dep_dzs, gas.dG_dep_dzs(), rtol=5e-6)

    dG_dep_dzs_expect = [-0.5299938046882267, -4.145751201696929, -12.296446092079966]
    assert_close1d(gas.dG_dep_dzs(), dG_dep_dzs_expect, rtol=1e-13)
    assert_close1d(gas_np.dG_dep_dzs(), dG_dep_dzs_expect, rtol=1e-13)
    assert isinstance(gas_np.dG_dep_dzs(), np.ndarray)

    def to_jac(ns):
        zs = [i/sum(ns) for i in ns]
        return sum(ns)*gas.to(T=T, P=P, zs=zs).lnphi()
    lnphis = jacobian(to_jac, zs, perturbation=4e-6)
    lnphis_expect = [-0.00021080760121739046, -0.0016603931258887027, -0.004928070691558975]
    assert_close1d(gas.lnphis(), lnphis_expect, rtol=1e-13)
    assert_close1d(gas_np.lnphis(), lnphis_expect, rtol=1e-13)
    assert isinstance(gas_np.lnphis(), np.ndarray)
    assert_close1d(lnphis, gas.lnphis(), rtol=1e-6)

# test_virial_phase_ternary_BC_pitzer_curl_orbey_vera_no_interactions()

def test_ternary_virial_phase_hashing_repr():
    Tcs=[126.2, 154.58, 150.8]
    Pcs=[3394387.5, 5042945.25, 4873732.5]
    Vcs=[8.95e-05, 7.34e-05, 7.49e-05]
    omegas=[0.04, 0.021, -0.004]
    T = 300.0
    model = VirialCSP(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas, B_model='VIRIAL_B_PITZER_CURL', cross_B_model='Tarakad-Danner', C_model='VIRIAL_C_ORBEY_VERA')
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63])),
                         HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
                         HeatCapacityGas(poly_fit=(50.0, 1000.0, [0,0,0,0, R*2.5]))]
    phase = VirialGas(model=model, T=T, P=1e5, zs=[.78, .21, .01], HeatCapacityGases=HeatCapacityGases, B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz')

    check_virial_temperature_consistency_T_calls(phase, [320, 800])


    model2 = VirialCSP(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas, B_model='VIRIAL_B_PITZER_CURL', cross_B_model='Tarakad-Danner', C_model='VIRIAL_C_ORBEY_VERA')
    phase2 = VirialGas(model=model2, T=T, P=1e5, zs=[.78, .21, .01], HeatCapacityGases=HeatCapacityGases, B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz')

    # print(model.__dict__)
    # print(model2.__dict__)
    assert model.model_hash() == model2.model_hash()

    model_different = VirialCSP(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas, B_model='VIRIAL_B_ABBOTT', cross_B_model='Tarakad-Danner', C_model='VIRIAL_C_ORBEY_VERA')

    # Change T
    phased0 = VirialGas(model=model, T=315.0, P=1e5, zs=[.78, .21, .01], HeatCapacityGases=HeatCapacityGases, B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz')
    # Change P
    phased1 = VirialGas(model=model, T=T, P=1e6, zs=[.78, .21, .01], HeatCapacityGases=HeatCapacityGases, B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz')
    # Change composition
    phased2 = VirialGas(model=model2, T=T, P=1e5, zs=[.77, .22, .01], HeatCapacityGases=HeatCapacityGases, B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz')
    # Change model to the same model
    phased3 = VirialGas(model=model2, T=T, P=1e5, zs=[.78, .21, .01], HeatCapacityGases=HeatCapacityGases, B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz')

    # Make a variety of diffinitely different models
    phased4 = VirialGas(model=model_different, T=315.0, P=1e5, zs=[.78, .21, .01], HeatCapacityGases=HeatCapacityGases, B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz')
    phased5 = VirialGas(model=model_different, T=T, P=1e6, zs=[.78, .21, .01], HeatCapacityGases=HeatCapacityGases, B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz')
    phased6 = VirialGas(model=model_different, T=T, P=1e5, zs=[.77, .22, .01], HeatCapacityGases=HeatCapacityGases, B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz')
    phased7 = VirialGas(model=model_different, T=T, P=1e5, zs=[.78, .21, .01], HeatCapacityGases=HeatCapacityGases, B_mixing_rule='linear', C_mixing_rule='Orentlicher-Prausnitz')


    assert model.state_hash() == model2.state_hash()
    assert model == model2
    assert hash(model) == hash(model2)
    assert eval(repr(model)) == model


    assert phase.model_hash() == phase2.model_hash()
    assert phase.state_hash() == phase2.state_hash()
    assert phase == phase2
    assert hash(phase) == hash(phase2)
    assert eval(repr(phase2)) == phase

    for p in (phased3, phased0, phased1, phased2):
        assert phase.model_hash() == p.model_hash()
    for p in (phased0, phased1, phased2):
        assert not phase.state_hash() == p.state_hash()
        assert not phase == p
        assert not hash(phase) == hash(p)

    for p in (phased4, phased5, phased6, phased7):
        assert not phase.model_hash() == p.model_hash()
        assert not phase.state_hash() == p.state_hash()
        assert not phase == p
        assert not hash(phase) == hash(p)

# test_ternary_virial_phase_hashing_repr()


def test_virial_ternary_vs_ideal_gas():
    constants = ChemicalConstantsPackage(Tcs=[508.1, 536.2, 512.5], Pcs=[4700000.0, 5330000.0, 8084000.0], omegas=[0.309, 0.21600000000000003, 0.5589999999999999], Vcs=[0.000213, 0.000244, 0.000117],
                                    MWs=[58.07914, 119.37764000000001, 32.04186], CASs=['67-64-1', '67-66-3', '67-56-1'], names=['acetone', 'chloroform', 'methanol'])
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.3320002425347943e-21, 6.4063345232664645e-18, -1.251025808150141e-14, 1.2265314167534311e-11, -5.535306305509636e-09, -4.32538332013644e-08, 0.0010438724775716248, -0.19650919978971002, 63.84239495676709])),
                        HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.5389278550737367e-21, -8.289631533963465e-18, 1.9149760160518977e-14, -2.470836671137373e-11, 1.9355882067011222e-08, -9.265600540761629e-06, 0.0024825718663005762, -0.21617464276832307, 48.149539665907696])),
                        HeatCapacityGas(poly_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924]))]
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, HeatCapacityGases=HeatCapacityGases)


    T, P = 350.0, 1e6
    zs = [0.2, 0.3, 0.5]
    N = len(zs)
    phase = IdealGas(T=T, P=P, zs=zs, HeatCapacityGases=HeatCapacityGases)

    model = VirialCSP(T=300.0, Tcs=constants.Tcs, Pcs=constants.Pcs, Vcs=constants.Vcs, omegas=constants.omegas, B_model=VIRIAL_B_ZERO, C_model=VIRIAL_C_ZERO)
    phase_EOS = VirialGas(model, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)

    check_virial_temperature_consistency_T_calls(phase_EOS, [320, 800])

    ### Copy specs
    TV = phase.to(T=T, V=phase.V(), zs=zs)
    assert_close(TV.P, phase.P, rtol=1e-12)
    PV = phase.to(P=P, V=phase.V(), zs=zs)
    assert_close(TV.T, phase.T, rtol=1e-12)

    new = phase.to_TP_zs(T=350.0, P=1e6, zs=zs)
    assert new.T == phase.T
    assert new.P == phase.P

    new = phase.to(P=P, T=T, zs=zs)
    assert new.T == phase.T
    assert new.P == phase.P

    B_mat = phase_EOS.model.B_interactions()
    assert_close1d(phase_EOS.model.B_pures(), [B_mat[i][i] for i in range(N)], rtol=1e-13)
    C_mat = phase_EOS.model.C_interactions()
    assert_close1d(phase_EOS.model.C_pures(), [C_mat[i][i] for i in range(N)], rtol=1e-13)

    assert_close(phase.V(), phase_EOS.V(), rtol=1e-11)
    assert_close(phase.PIP(), phase_EOS.PIP(), rtol=1e-11)

    ### Derivatives of volume
    assert_close(phase.dV_dT(), phase_EOS.dV_dT(), rtol=1e-11)
    assert_close(phase.dV_dT_P(), phase_EOS.dV_dT_P(), rtol=1e-11)

    assert_close(phase.dV_dP(), phase_EOS.dV_dP(), rtol=1e-11)
    assert_close(phase.dV_dP_T(), phase_EOS.dV_dP_T(), rtol=1e-11)

    assert_close(phase.d2V_dT2(), phase_EOS.d2V_dT2(), rtol=1e-11)
    assert_close(phase.d2V_dT2_P(), phase_EOS.d2V_dT2_P(), rtol=1e-11)

    assert_close(phase.d2V_dP2(), phase_EOS.d2V_dP2(), rtol=1e-11)
    assert_close(phase.d2V_dP2_T(), phase_EOS.d2V_dP2_T(), rtol=1e-11)

    assert_close(phase.d2V_dTdP(), phase_EOS.d2V_dTdP(), rtol=1e-11)
    assert_close(phase.d2V_dPdT(), phase_EOS.d2V_dPdT(), rtol=1e-11)

    assert phase.dZ_dT() == 0.0
    assert phase.dZ_dP() == 0.0

    assert_close1d(phase_EOS.dV_dzs(), [0.0, 0.0, 0.0], atol=0.0, rtol=0.0)
    assert_close1d(phase_EOS.dZ_dzs(), [0.0, 0.0, 0.0], atol=0.0, rtol=0.0)
    assert_close1d(phase_EOS.dV_dzs(), [0.0, 0.0, 0.0], atol=0.0, rtol=0.0)
    assert_close1d(phase_EOS.dZ_dzs(), [0.0, 0.0, 0.0], atol=0.0, rtol=0.0)


    ### Derivatives of pressure
    assert_close(phase.dP_dV(), phase_EOS.dP_dV(), rtol=1e-11)
    assert_close(phase.dP_dV_T(), phase_EOS.dP_dV_T(), rtol=1e-11)

    assert_close(phase.dP_dT(), phase_EOS.dP_dT(), rtol=1e-11)
    assert_close(phase.dP_dT_V(), phase_EOS.dP_dT_V(), rtol=1e-11)

    assert_close(phase.d2P_dV2(), phase_EOS.d2P_dV2(), rtol=1e-11)
    assert_close(phase.d2P_dV2_T(), phase_EOS.d2P_dV2_T(), rtol=1e-11)

    assert_close(phase.d2P_dT2(), phase_EOS.d2P_dT2(), rtol=1e-11)
    assert_close(phase.d2P_dT2_V(), phase_EOS.d2P_dT2_V(), rtol=1e-11)

    assert_close(phase.d2P_dVdT(), phase_EOS.d2P_dVdT(), rtol=1e-11)
    assert_close(phase.d2P_dTdV(), phase_EOS.d2P_dTdV(), rtol=1e-11)

    ### Derivatives of Temperature
    assert_close(phase.dT_dV(), phase_EOS.dT_dV(), rtol=1e-11)
    assert_close(phase.dT_dV_P(), phase_EOS.dT_dV_P(), rtol=1e-11)

    assert_close(phase.dT_dP(), phase_EOS.dT_dP(), rtol=1e-11)
    assert_close(phase.dT_dP_V(), phase_EOS.dT_dP_V(), rtol=1e-11)

    assert_close(phase.d2T_dV2(), phase_EOS.d2T_dV2(), rtol=1e-11)
    assert_close(phase.d2T_dV2_P(), phase_EOS.d2T_dV2_P(), rtol=1e-11)

    assert_close(phase.d2T_dP2(), phase_EOS.d2T_dP2(), rtol=1e-11)
    assert_close(phase.d2T_dP2_V(), phase_EOS.d2T_dP2_V(), rtol=1e-11)

    assert_close(phase.d2T_dVdP(), phase_EOS.d2T_dVdP(), rtol=1e-11)
    assert_close(phase.d2T_dPdV(), phase_EOS.d2T_dPdV(), rtol=1e-11)


    ### Phis and derivatives
    assert_close1d(phase.phis(), phase_EOS.phis(), rtol=1e-11)
    assert_close1d(phase.lnphis(), phase_EOS.lnphis(), rtol=1e-11)
    assert_close1d(phase.fugacities(), phase_EOS.fugacities(), rtol=1e-11)

    # Basic thermodynamic quantities
    assert_close(phase.H(), phase_EOS.H(), rtol=1e-11)
    assert_close(phase.S(), phase_EOS.S(), rtol=1e-11)
    assert_close(phase.G(), phase_EOS.G(), rtol=1e-11)
    assert_close(phase.U(), phase_EOS.U(), rtol=1e-11)
    assert_close(phase.A(), phase_EOS.A(), rtol=1e-11)

    ### First temperature derivative

    assert_close(phase.dH_dT(), phase_EOS.dH_dT(), rtol=1e-11)
    assert_close(phase.dS_dT(), phase_EOS.dS_dT(), rtol=1e-11)
    assert_close(phase.dG_dT(), phase_EOS.dG_dT(), rtol=1e-11)
    assert_close(phase.dU_dT(), phase_EOS.dU_dT(), rtol=1e-11)
    assert_close(phase.dA_dT(), phase_EOS.dA_dT(), rtol=1e-11)

    assert_close(phase.dH_dT_P(), phase_EOS.dH_dT_P(), rtol=1e-11)
    assert_close(phase.dS_dT_P(), phase_EOS.dS_dT_P(), rtol=1e-11)
    assert_close(phase.dG_dT_P(), phase_EOS.dG_dT_P(), rtol=1e-11)
    assert_close(phase.dU_dT_P(), phase_EOS.dU_dT_P(), rtol=1e-11)
    assert_close(phase.dA_dT_P(), phase_EOS.dA_dT_P(), rtol=1e-11)

    assert_close(phase.dH_dT_V(), phase_EOS.dH_dT_V(), rtol=1e-11)
    assert_close(phase.dS_dT_V(), phase_EOS.dS_dT_V(), rtol=1e-11)
    assert_close(phase.dG_dT_V(), phase_EOS.dG_dT_V(), rtol=1e-11)
    assert_close(phase.dU_dT_V(), phase_EOS.dU_dT_V(), rtol=1e-11)
    assert_close(phase.dA_dT_V(), phase_EOS.dA_dT_V(), rtol=1e-11)

    ### First pressure derivative
    assert_close(phase.dH_dP(), phase_EOS.dH_dP(), rtol=1e-11, atol=1e-16)
    assert_close(phase.dS_dP(), phase_EOS.dS_dP(), rtol=1e-11)
    assert_close(phase.dG_dP(), phase_EOS.dG_dP(), rtol=1e-11)
    assert_close(phase.dU_dP(), phase_EOS.dU_dP(), rtol=1e-11, atol=1e-16)
    assert_close(phase.dA_dP(), phase_EOS.dA_dP(), rtol=1e-11)

    assert_close(phase.dH_dP_T(), phase_EOS.dH_dP_T(), rtol=1e-11, atol=1e-16)
    assert_close(phase.dS_dP_T(), phase_EOS.dS_dP_T(), rtol=1e-11)
    assert_close(phase.dG_dP_T(), phase_EOS.dG_dP_T(), rtol=1e-11)
    assert_close(phase.dU_dP_T(), phase_EOS.dU_dP_T(), rtol=1e-11, atol=1e-16)
    assert_close(phase.dA_dP_T(), phase_EOS.dA_dP_T(), rtol=1e-11)

    assert_close(phase.dH_dP_V(), phase_EOS.dH_dP_V(), rtol=1e-11)
    assert_close(phase.dS_dP_V(), phase_EOS.dS_dP_V(), rtol=1e-11)
    assert_close(phase.dG_dP_V(), phase_EOS.dG_dP_V(), rtol=1e-11)
    assert_close(phase.dU_dP_V(), phase_EOS.dU_dP_V(), rtol=1e-11, atol=1e-16)
    assert_close(phase.dA_dP_V(), phase_EOS.dA_dP_V(), rtol=1e-11)

    assert_close(phase.dH_dV_T(), phase_EOS.dH_dV_T(), rtol=1e-11)
    assert_close(phase.dS_dV_T(), phase_EOS.dS_dV_T(), rtol=1e-11)
    assert_close(phase.dG_dV_T(), phase_EOS.dG_dV_T(), rtol=1e-11)

    assert phase_EOS.H_dep() == 0
    assert phase_EOS.S_dep() == 0
    assert phase_EOS.G_dep() == 0
    assert phase_EOS.U_dep() == 0
    assert phase_EOS.A_dep() == 0


def check_virial_temperature_consistency_T_calls(model, T_list, rtol=1e-14):
    for T in T_list:
        # Create a new model at the test temperature
        model_at_T = model.to(T=T, P=model.P, zs=model.zs)
        
        # Check B and its derivatives
        assert_close(model_at_T.B(), model.B_at(T, model.zs), rtol=rtol)
        assert_close(model_at_T.dB_dT(), model.dB_dT_at(T, model.zs), rtol=rtol)
        assert_close(model_at_T.d2B_dT2(), model.d2B_dT2_at(T, model.zs), rtol=rtol)
        assert_close(model_at_T.d3B_dT3(), model.d3B_dT3_at(T, model.zs), rtol=rtol) 
        
        # Check C and its derivatives
        assert_close(model_at_T.C(), model.C_at(T, model.zs), rtol=rtol)
        assert_close(model_at_T.dC_dT(), model.dC_dT_at(T, model.zs), rtol=rtol)
        assert_close(model_at_T.d2C_dT2(), model.d2C_dT2_at(T, model.zs), rtol=rtol)
        assert_close(model_at_T.d3C_dT3(), model.d3C_dT3_at(T, model.zs), rtol=rtol) 

def test_virial_easy_B_C_models():
    from chemicals.virial import Meng_virial_a
    CASs = ['7727-37-9', '74-82-8', '124-38-9']
    atomss = [{'N': 2}, {'C': 1, 'H': 4}, {'C': 1, 'O': 2}]
    Tcs = [126.2, 190.564, 304.2]
    Pcs = [3394387.5, 4599000.0, 7376460.0]
    Vcs = [8.95e-05, 9.86e-05, 9.4e-05]
    omegas = [0.04, 0.008, 0.2252]
    N = 3

    HeatCapacityGases = [HeatCapacityGas(CASRN="7727-37-9", MW=28.0134, similarity_variable=0.07139440410660612, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
     HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="124-38-9", MW=44.0095, similarity_variable=0.0681671002851657, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644]))]

    kijs = [[0.0, 0.024878043016556484, 0.0],
     [0.024878043016556484, 0.0, 0.04313694538361394],
     [0.0, 0.04313694538361394, 0.0]]

    # Get VIRIAL_B_TSONOPOULOS and VIRIAL_C_XIANG
    T = 300
    model = VirialCSP(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                  B_model=VIRIAL_B_TSONOPOULOS,
                                  cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                                  cross_B_model_kijs=kijs,
                                  C_model=VIRIAL_C_XIANG)
    P = 1e5
    zs = [.02, .92, .06]
    gas = VirialGas(model=model, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                T=T, P=P, zs=zs)

    B, dB, d2B, d3B = -4.434925694919992e-05, 3.937650137082095e-07, -3.2111514855184275e-09, 3.870025170366373e-11
    assert_close(gas.B(), B, rtol=1e-13)
    assert_close(gas.dB_dT(), dB, rtol=1e-13)
    assert_close(gas.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas.d3B_dT3(), d3B, rtol=1e-13)

    C, dC, d2C, d3C = (2.3672028792220187e-09, -8.499704687487288e-12, 1.014506083545108e-13, -1.3762462745837535e-15)
    assert_close(gas.C(), C, rtol=1e-13)
    assert_close(gas.dC_dT(), dC, rtol=1e-13)
    assert_close(gas.d2C_dT2(), d2C, rtol=1e-13)
    assert_close(gas.d3C_dT3(), d3C, rtol=1e-13)

    B_mat = gas.model.B_interactions()
    assert_close1d(gas.model.B_pures(), [B_mat[i][i] for i in range(N)], rtol=1e-13)
    C_mat = gas.model.C_interactions()
    assert_close1d(gas.model.C_pures(), [C_mat[i][i] for i in range(N)], rtol=1e-13)
    check_virial_temperature_consistency_T_calls(gas, [320, 800])

    # VIRIAL_B_OCONNELL_PRAUSNITZ

    model = VirialCSP(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                  B_model=VIRIAL_B_OCONNELL_PRAUSNITZ,
                                  cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                                  cross_B_model_kijs=kijs,
                                  C_model=VIRIAL_C_ZERO)
    gas = VirialGas(model=model, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                T=T, P=P, zs=zs)

    B, dB, d2B, d3B = -4.4070589761655625e-05, 3.9458357396975057e-07, -3.265768989707745e-09, 4.069493510858296e-11
    assert_close(gas.B(), B, rtol=1e-13)
    assert_close(gas.dB_dT(), dB, rtol=1e-13)
    assert_close(gas.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas.d3B_dT3(), d3B, rtol=1e-13)
    B_mat = gas.model.B_interactions()
    assert_close1d(gas.model.B_pures(), [B_mat[i][i] for i in range(N)], rtol=1e-13)
    check_virial_temperature_consistency_T_calls(gas, [320, 800])


    model = VirialCSP(T=T,Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                  B_model=VIRIAL_B_XIANG,
                                  cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                                  cross_B_model_kijs=kijs,
                                  C_model=VIRIAL_C_ZERO)
    gas = VirialGas(model=model, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                T=T, P=P, zs=zs)

    # VIRIAL_B_XIANG
    B, dB, d2B, d3B = -4.558372700762532e-05, 3.8023980103500797e-07, -2.9686366447752638e-09, 3.4864810676267165e-11
    assert_close(gas.B(), B, rtol=1e-13)
    assert_close(gas.dB_dT(), dB, rtol=1e-13)
    assert_close(gas.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas.d3B_dT3(), d3B, rtol=1e-13)
    B_mat = gas.model.B_interactions()
    assert_close1d(gas.model.B_pures(), [B_mat[i][i] for i in range(N)], rtol=1e-13)

    check_virial_temperature_consistency_T_calls(gas, [320, 800])

    # VIRIAL_B_MENG
    # Made up numbers to get a value of `a`
    dipoles = [[1e-5, 1e-3, 2e-2], [1e-3, 1e-6, 1e-2], [2e-2, 1e-2, 1e-8]]
    Tcijs = [[126.20000000000002, 151.21993365643428, 195.93376431845533], [151.21993365643428, 190.56399999999996, 230.38267751412258], [195.93376431845533, 230.38267751412258, 304.19999999999993]]
    Pcijs = [[3394387.499999999, 3851314.2142062606, 5005217.123514405], [3851314.2142062606, 4598999.999999997, 5573970.154312947], [5005217.123514405, 5573970.154312947, 7376459.999999995]]
    N = 3
    Meng_virial_as = [[Meng_virial_a(Tcijs[i][j], Pcijs[i][j], dipole=dipoles[i][j]) for j in range(N)]
                                     for i in range(N)]
    model = VirialCSP(T=T,Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                  B_model=VIRIAL_B_MENG,
                                  cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                                  cross_B_model_kijs=kijs,
                                  B_model_Meng_as=Meng_virial_as,
                                  C_model=VIRIAL_C_ZERO)
    gas = VirialGas(model=model, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                T=T, P=P, zs=zs)

    B, dB, d2B, d3B = (-4.4032087141036684e-05, 3.8809438374191875e-07, -3.195223834343181e-09, 3.857289293387547e-11)
    assert_close(gas.B(), B, rtol=1e-13)
    assert_close(gas.dB_dT(), dB, rtol=1e-13)
    assert_close(gas.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas.d3B_dT3(), d3B, rtol=1e-13)
    B_mat = gas.model.B_interactions()
    assert_close1d(gas.model.B_pures(), [B_mat[i][i] for i in range(N)], rtol=1e-13)
    check_virial_temperature_consistency_T_calls(gas, [320, 800])

    # VIRIAL_B_TSONOPOULOS_EXTENDED
    # Made up parameters
    # These should always be symmetric
    # No real ideal how to use them except set the diagonal to real values and zero the rest
    BVirial_Tsonopoulos_extended_as = [[-0.0109, 0.0878, -6787.3],
     [0.0878, 0.5, 0.3],
     [-6787.3, .3, -5000.0]]

    BVirial_Tsonopoulos_extended_bs = [[1e-5, 1e-6, 1e-7], [1e-6, .25, 1e-8], [1e-7, 1e-8, .343]]


    model = VirialCSP(T=T,Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                  B_model=VIRIAL_B_TSONOPOULOS_EXTENDED,
                                  cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                                  cross_B_model_kijs=kijs,
                                 B_model_Tsonopoulos_extended_as=BVirial_Tsonopoulos_extended_as,
                                 B_model_Tsonopoulos_extended_bs=BVirial_Tsonopoulos_extended_bs,
                                  C_model=VIRIAL_C_ZERO)
    gas = VirialGas(model=model, HeatCapacityGases=HeatCapacityGases,
                    B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                T=T, P=1e4, zs=zs)

    B, dB, d2B, d3B = (-0.007155125156828212, 0.00014262531996979625, -3.3223750860211125e-06, 8.85625680727924e-08)
    assert_close(gas.B(), B, rtol=1e-13)
    assert_close(gas.dB_dT(), dB, rtol=1e-13)
    assert_close(gas.d2B_dT2(), d2B, rtol=1e-13)
    assert_close(gas.d3B_dT3(), d3B, rtol=1e-13)

    B_mat = gas.model.B_interactions()
    assert_close1d(gas.model.B_pures(), [B_mat[i][i] for i in range(N)], rtol=1e-13)
    check_virial_temperature_consistency_T_calls(gas, [320, 800])


def test_basic_compare_virial_vs_PR():
    T, P, zs = 300, 1e5,  [1]
    eos_kwargs = {'Pcs': [4599200.0], 'Tcs': [190.564], 'omegas': [0.01142], }
    Vcs = [9.86e-05]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 5000.0, [4.8986184697537195e-26, -1.1318000255051273e-21, 1.090383509787202e-17, -5.664719389870236e-14, 1.7090042167602582e-10, -2.9728679808459997e-07, 0.00026565262671378613, -0.054476667747310976, 35.35366254807737]
                                                ))]
    cubic = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)

    virial_csp = VirialCSP(T=T, Vcs=Vcs, B_model=VIRIAL_B_ABBOTT, cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                      C_model=VIRIAL_C_ORBEY_VERA, **eos_kwargs)

    virial = VirialGas(model=virial_csp, HeatCapacityGases=HeatCapacityGases,
                       B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                       T=T, P=P, zs=zs)

    assert_close(cubic.H_dep(), virial.H_dep(), rtol=0.2)
    assert_close(cubic.G_dep(), virial.G_dep(), rtol=0.35)
    assert_close(cubic.S_dep(), virial.S_dep(), rtol=0.15)
    assert_close(cubic.lnphi(), virial.lnphi(), rtol=0.4)
    assert_close1d(cubic.lnphis(), virial.lnphis(), rtol=0.35)

    assert_close(cubic.H(), virial.H(), rtol=0.05)
    assert_close(cubic.S(), virial.S(), rtol=0.05)
    assert_close(cubic.G(), virial.G(), rtol=0.05)
    


def test_basic_compare_H_dep_vs_literature_equations():
    def H_dep_Poling_leiden(self):
        r'''Method to calculate and return the molar departure enthalpy using
        the Poling et al. equation form.
        
        Returns
        -------
        H_dep : float
            Departure enthalpy [J/mol]
            
        Notes
        -----
        '''
        V = self._V  # molar volume
        T = self.T   # temperature
        B = self.B()  # mixture second virial coefficient
        C = self.C()  # mixture third virial coefficient
        dB_dT = self.dB_dT()  # temperature derivative of B
        dC_dT = self.dC_dT()  # temperature derivative of C
        
        # First term: -(B - T*dB/dT)/V
        first_term = -(B - T*dB_dT)/V
        
        # Second term: -(2C - T*dC/dT)/(2V²)
        second_term = -(2.0*C - T*dC_dT)/(2.0*V*V)
        
        # Combine terms and multiply by RT
        # The equation has the negative of H_Dep in it
        return - self.R*T*(first_term + second_term)
        

    def H_dep_Walas(self):
        r'''Method to calculate and return the molar departure enthalpy using
        the Walas formulation with virial coefficients in the pressure form.
        
        .. math::
            H^{id} - H = RT^2 \left[ P \frac{dB'}{dT} + \frac{P^2}{2} \frac{dC'}{dT} + \ldots \right]
            
            = PT \left( \frac{dB}{dT} - \frac{B}{T} \right)
            + \frac{P^2}{2R} \left[ \frac{B^2-C}{T} + \frac{dC}{dT} - 2B\frac{dB}{dT} + \ldots \right]
        
        Where:
            B' = B/RT
            C' = (C - B²)/(RT)²
            dB'/dT = (1/RT)·(dB/dT - B/T)
            
        Returns
        -------
        H_dep : float
            Departure enthalpy [J/mol]
            
        Notes
        -----
        This method uses the virial equation in pressure form truncated after the C' term.
        The equation gives the departure from the ideal gas state.
        '''
        T = self.T   # temperature
        P = self.P   # pressure
        B = self.B()  # mixture second virial coefficient
        C = self.C()  # mixture third virial coefficient
        dB_dT = self.dB_dT()  # temperature derivative of B
        dC_dT = self.dC_dT()  # temperature derivative of C
        R = self.R   # gas constant
        
        # First term: PT(dB/dT - B/T)
        first_term = P*T*(dB_dT - B/T)
        
        # Second term: (P²/2R)·[(B²-C)/T + dC/dT - 2B·dB/dT]
        second_term = (P*P)/(2.0*R)*((B*B-C)/T + dC_dT - 2.0*B*dB_dT)
        
        # H_dep = -(H^id - H)
        H_dep = -(first_term + second_term)
        
        return H_dep

    T, P, zs = 300, 1e5,  [1]
    eos_kwargs = {'Pcs': [4599200.0], 'Tcs': [190.564], 'omegas': [0.01142], }
    Vcs = [9.86e-05]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 5000.0, [4.8986184697537195e-26, -1.1318000255051273e-21, 1.090383509787202e-17, -5.664719389870236e-14, 1.7090042167602582e-10, -2.9728679808459997e-07, 0.00026565262671378613, -0.054476667747310976, 35.35366254807737]
                                                ))]
    virial_csp = VirialCSP(T=T, Vcs=Vcs, B_model=VIRIAL_B_ABBOTT, cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                      C_model=VIRIAL_C_ORBEY_VERA, **eos_kwargs)

    virial = VirialGas(model=virial_csp, HeatCapacityGases=HeatCapacityGases,
                       B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                       T=T, P=P, zs=zs)


    assert_close(virial.H_dep(), H_dep_Poling_leiden(virial), rtol=1e-9)
    # Walas doesn't quite match
    assert_close(virial.H_dep(), H_dep_Walas(virial), rtol=1e-4)


def test_basic_compare_S_dep_vs_literature_equations():
    def S_dep_Poling_leiden(self):
        r'''Method to calculate and return the molar departure entropy using
        the Poling et al. equation form.
        
        .. math::
            \frac{S^{ig} - S}{R} = \frac{\left(B + T\frac{dB}{dT}\right)}{V} 
            + \frac{\left(C - T\frac{dC}{dT}\right)}{2V^2} + \ldots - \ln Z
            
            S - S^{ig} = -R\left[\frac{\left(B + T\frac{dB}{dT}\right)}{V} 
            + \frac{\left(C - T\frac{dC}{dT}\right)}{2V^2} + \ldots - \ln Z\right]
        
        Returns
        -------
        S_dep : float
            Departure entropy [J/(mol*K)]
            
        Notes
        -----
        This method uses the virial equation truncated after the C term.
        The equation gives the departure from the ideal gas state.
        '''
        V = self._V  # molar volume
        T = self.T   # temperature
        B = self.B()  # mixture second virial coefficient
        C = self.C()  # mixture third virial coefficient
        dB_dT = self.dB_dT()  # temperature derivative of B
        dC_dT = self.dC_dT()  # temperature derivative of C
        
        # Calculate Z factor
        Z = self.P*V/(self.R*T)
        ln_Z = log(Z)
        
        # First term: (B + T*dB/dT)/V
        first_term = (B + T*dB_dT)/V
        
        # Second term: (C - T*dC/dT)/(2V²)

        # While implementing this formula and debugging the disrepancy, it was found that the 
        # C - T*dC_dT is incorrect and should be C + T*dC_dT
        second_term = (C + T*dC_dT)/(2.0*V*V)
        
        # Combine terms and multiply by -R
        S_dep = -self.R*(first_term + second_term - ln_Z)
        
        return S_dep

    def S_dep_Walas(self):
        r'''Method to calculate and return the molar departure entropy using
        the Walas formulation with virial coefficients in the pressure form.
        Table 11.2. 

        It has not been found where the error is, but there is one.


        .. math::
            S^{id} - S = R \left[ B'P + \frac{C'P^2}{2} + \ldots
            + T \left( P \frac{dB'}{dT} + \frac{P^2}{2} \frac{dC'}{dT} + \ldots \right) \right]
        
        Where:
            B' = B/RT
            C' = (C - B²)/(RT)²
            dB'/dT = (1/RT)·(dB/dT - B/T)
            dC'/dT = (1/RT)²·((B²-C)/T + dC/dT - 2B·dB/dT)
            
        Returns
        -------
        S_dep : float
            Departure entropy [J/(mol*K)]
            
        Notes
        -----
        '''
        T = self.T   # temperature
        P = self.P   # pressure
        B = self.B()  # mixture second virial coefficient
        C = self.C()  # mixture third virial coefficient
        dB_dT = self.dB_dT()  # temperature derivative of B
        dC_dT = self.dC_dT()  # temperature derivative of C
        R = self.R   # gas constant
        
        # Calculate B' and C'
        B_prime = B/(R*T)
        C_prime = (C - B*B)/((R*T)**2)
        
        # Calculate dB'/dT and dC'/dT
        dB_prime_dT = (1.0/(R*T))*(dB_dT - B/T)
        dC_prime_dT = (1.0/(R*T))**2*((B*B-C)/T + dC_dT - 2.0*B*dB_dT)
        
        # First part: R·[B'P + (C'P²)/2]
        first_part = R*(B_prime*P + (C_prime*P*P)/2.0)
        
        # Second part: R·T·[P·(dB'/dT) + (P²/2)·(dC'/dT)]
        second_part = R*T*(P*dB_prime_dT + (P*P/2.0)*dC_prime_dT)
        
        # S_dep = -(S^id - S)
        S_dep = -(first_part + second_part)
        
        return S_dep
    T, P, zs = 300, 1e5,  [1]
    eos_kwargs = {'Pcs': [4599200.0], 'Tcs': [190.564], 'omegas': [0.01142], }
    Vcs = [9.86e-05]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 5000.0, [4.8986184697537195e-26, -1.1318000255051273e-21, 1.090383509787202e-17, -5.664719389870236e-14, 1.7090042167602582e-10, -2.9728679808459997e-07, 0.00026565262671378613, -0.054476667747310976, 35.35366254807737]
                                                ))]
    virial_csp = VirialCSP(T=T, Vcs=Vcs, B_model=VIRIAL_B_ABBOTT, cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                      C_model=VIRIAL_C_ORBEY_VERA, **eos_kwargs)

    virial = VirialGas(model=virial_csp, HeatCapacityGases=HeatCapacityGases,
                       B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                       T=T, P=P, zs=zs)


    assert_close(virial.S_dep(), S_dep_Poling_leiden(virial), rtol=1e-9)
    # Walas doesn't quite match, some error, unclear
    assert_close(virial.S_dep(), S_dep_Walas(virial), rtol=2e-4)


def test_basic_compare_G_dep_lnphi_lnphis_vs_literature_equations():

    def lnphis_Prausnitz_leiden(self):
        r'''Method to calculate and return the log fugacity coefficients of
        the phase using the Prausnitz-Leiden equation.
        
        .. math::
            \ln \phi_i = \frac{2}{v} \sum_{j=1}^m y_j B_{ij} + \frac{3}{2v^2} \sum_{j=1}^m \sum_{k=1}^m y_j y_k C_{ijk} - \ln z_{mixt}
        
        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
            
        Notes
        -----
        The zs member variable is used as the mole fractions (y_j in the equation).
        This method requires interaction virial coefficients B_ij and C_ijk.
        '''
        V = self._V  # molar volume
        v_inv = 1.0/V
        v2_inv = v_inv*v_inv
        zs = self.zs  # mole fractions (y in the equation)
        
        # Calculate Z factor for the mixture (z_mixt)
        Z = self.P*V/(self.R*self.T)
        ln_Z = log(Z)
        
        # Get interaction coefficients
        B_interactions = self.model.B_interactions()
        C_interactions = self.model.C_interactions() if self.has_cross_C_coefficients else None
        
        N = self.N
        lnphis = [0.0]*N
        
        # Calculate ln phi for each component
        for i in range(N):
            # First term: (2/v) * sum(y_j * B_ij)
            sum_y_B = 0.0
            for j in range(N):
                sum_y_B += zs[j] * B_interactions[i][j]
            
            # Second term: (3/2v²) * sum(y_j * y_k * C_ijk)
            sum_yy_C = 0.0
            if self.has_cross_C_coefficients and self.model.C_model != VIRIAL_C_ZERO:
                for j in range(N):
                    for k in range(N):
                        C_ijk = 0.0
                        C_ij = C_interactions[i][j]
                        C_jk = C_interactions[j][k]
                        C_ik = C_interactions[i][k]
                        
                        # Using Orentlicher-Prausnitz mixing rule
                        C_ijk = (C_ij * C_jk * C_ik)**(1.0/3.0) if (C_ij * C_jk * C_ik) > 0 else 0.0
                        
                        sum_yy_C += zs[j] * zs[k] * C_ijk
            
            # Combine terms for ln phi_i
            lnphis[i] = 2.0 * v_inv * sum_y_B + 1.5 * v2_inv * sum_yy_C - ln_Z
        
        return lnphis
    def lnphis_Gmehling_B_only_leiden(self):
        r'''Method to calculate and return the log fugacity coefficients of
        the phase using the Gmehling B-only equation (Leiden form).
        
        .. math::
            \ln \phi_k = \int_v^{\infty} \frac{2n_r \sum_i y_i B_{ik}}{V^2} dV - \ln z 
            = \left(2n_r \sum_i y_i B_{ik}\right) \int_v^{\infty} \frac{dV}{V^2} - \ln z
            = \frac{2}{v} \sum_i y_i B_{ik} - \ln z
        
        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
            
        Notes
        -----
        This simplified form uses only the second virial coefficient (B) terms.
        The zs member variable is used as the mole fractions (y_i in the equation).
        '''
        V = self._V  # molar volume
        v_inv = 1.0/V
        zs = self.zs  # mole fractions (y in the equation)
        
        # Calculate Z factor for the mixture
        Z = self.P*V/(self.R*self.T)
        ln_Z = log(Z)
        
        # Get interaction coefficients
        B_interactions = self.model.B_interactions()
        
        N = self.N
        lnphis = [0.0]*N
        
        # Calculate ln phi for each component k
        for k in range(N):
            # Calculate sum(y_i * B_ik)
            sum_y_B = 0.0
            for i in range(N):
                sum_y_B += zs[i] * B_interactions[i][k]
            
            # Apply formula: 2/v * sum(y_i * B_ik) - ln z
            lnphis[k] = 2.0 * v_inv * sum_y_B - ln_Z
        
        return lnphis

    def lnphi_Poling_leiden(self):
        r'''Method to calculate and return the log fugacity coefficient of
        the phase using the Poling et al. equation form.
        
        .. math::
            \ln \phi = \ln \left(\frac{f}{P}\right) = 2\frac{B}{V} + \frac{3C}{2V^2} + \ldots - \ln Z
        
        Returns
        -------
        lnphi : float
            Log fugacity coefficient for the mixture, [-]
            
        Notes
        -----
        This method returns the fugacity coefficient for the overall mixture,
        not component-specific fugacity coefficients.
        The equation is truncated after the C term as higher-order terms are
        rarely available.
        '''
        V = self._V  # molar volume
        B = self.B()  # mixture second virial coefficient
        C = self.C()  # mixture third virial coefficient
        
        # Calculate Z factor
        Z = self.P*V/(self.R*self.T)
        ln_Z = log(Z)
        
        # Apply the formula from Poling
        lnphi = 2.0*B/V + 1.5*C/(V*V) - ln_Z
        
        return lnphi

    def G_dep_Poling_leiden(self):
        r'''Method to calculate and return the molar departure Gibbs energy using
        the Poling et al. equation form. The negative of these equations is the departure
        term.
        
        .. math::
            \frac{G^{ig} - G}{RT} = -2\frac{B}{V} - \frac{3C}{2V^2} - \ldots + \ln Z
            
            G - G^{ig} = RT\left[-2\frac{B}{V} - \frac{3C}{2V^2} - \ldots + \ln Z\right]
        
        Returns
        -------
        G_dep : float
            Departure Gibbs energy [J/mol]
            
        Notes
        -----
        This method uses the virial equation truncated after the C term.
        The equation gives the departure from the ideal gas state.
        Mathematically, this is equivalent to RT*ln(φ), where φ is the
        fugacity coefficient.
        '''
        V = self._V  # molar volume
        T = self.T   # temperature
        B = self.B()  # mixture second virial coefficient
        C = self.C()  # mixture third virial coefficient
        
        # Calculate Z factor
        Z = self.P*V/(self.R*T)
        ln_Z = log(Z)
        
        # First term: -2B/V
        first_term = -2.0*B/V
        
        # Second term: -3C/(2V²)
        second_term = -3.0*C/(2.0*V*V)
        
        # Combine terms and multiply by RT
        G_dep = -self.R*T*(first_term + second_term + ln_Z)
        
        return G_dep

    T, P, zs = 300, 1e5,  [1]
    eos_kwargs = {'Pcs': [4599200.0], 'Tcs': [190.564], 'omegas': [0.01142], }
    Vcs = [9.86e-05]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 5000.0, [4.8986184697537195e-26, -1.1318000255051273e-21, 1.090383509787202e-17, -5.664719389870236e-14, 1.7090042167602582e-10, -2.9728679808459997e-07, 0.00026565262671378613, -0.054476667747310976, 35.35366254807737]
                                                ))]
    virial_csp = VirialCSP(T=T, Vcs=Vcs, B_model=VIRIAL_B_ABBOTT, cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                      C_model=VIRIAL_C_ORBEY_VERA, **eos_kwargs)

    virial = VirialGas(model=virial_csp, HeatCapacityGases=HeatCapacityGases,
                       B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                       T=T, P=P, zs=zs)

    assert_close(virial.G_dep(), G_dep_Poling_leiden(virial), rtol=1e-9)
    assert_close(virial.lnphi(), lnphi_Poling_leiden(virial), rtol=1e-9)


    assert_close1d(virial.lnphis(), lnphis_Prausnitz_leiden(virial), rtol=1e-9)


    # This formula only has B
    virial_csp = VirialCSP(T=T, Vcs=Vcs, B_model=VIRIAL_B_ABBOTT, cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                      C_model=VIRIAL_C_ZERO, **eos_kwargs)

    virial = VirialGas(model=virial_csp, HeatCapacityGases=HeatCapacityGases,
                       B_mixing_rule='theory', C_mixing_rule='Orentlicher-Prausnitz',
                       T=T, P=P, zs=zs)

    assert_close1d(virial.lnphis(), lnphis_Gmehling_B_only_leiden(virial), rtol=1e-9)

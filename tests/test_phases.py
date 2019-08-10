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

from numpy.testing import assert_allclose
import pytest

from thermo.phases import *
from thermo.eos_mix import *
from thermo.eos import *
from thermo.vapor_pressure import VaporPressure
from thermo.heat_capacity import *
from thermo.phase_change import *

def test_GibbbsExcessLiquid_VaporPressure():
    # Binary ethanol-water
    VaporPressures = [VaporPressure(best_fit=(159.11, 514.7, [-2.3617526481119e-19, 7.318686894378096e-16, -9.835941684445551e-13, 7.518263303343784e-10, -3.598426432676194e-07, 0.00011171481063640762, -0.022458952185007635, 2.802615041941912, -166.43524219017118])),
                      VaporPressure(best_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))]
    T = 300.0
    P = 1e5
    zs = [.4, .6]
    liquid = GibbsExcessLiquid(VaporPressures=VaporPressures).to_TP_zs(T, P, zs)
    
    # Ingredients
    Psats_expect = [8778.910843769489, 3537.075987237396]
    assert_allclose(liquid.Psats(), Psats_expect, rtol=1e-12)
    
    gammas_expect = [1.0, 1.0]
    assert liquid.gammas() == gammas_expect
    
    Poyntings_expect = [1.0, 1.0]
    assert liquid.Poyntings() == Poyntings_expect
    
    dPoyntings_dT_expect = [0.0, 0.0]
    assert liquid.dPoyntings_dT() == dPoyntings_dT_expect
    
    dPoyntings_dP_expect = [0.0, 0.0]
    assert liquid.dPoyntings_dP() == dPoyntings_dP_expect
    
    phis_sat_expect = [1.0, 1.0]
    assert liquid.phis_sat() == phis_sat_expect
    
    # Fugacities and friends
    phis_expect = [0.0877891084376949, 0.035370759872373966]
    assert_allclose(liquid.phis(), phis_expect, rtol=1e-12)
    
    lnphis_expect = [-2.432817835720433, -3.3418697924678376]
    assert_allclose(liquid.lnphis(), lnphis_expect, rtol=1e-12)
    
    fugacities_expect = [3511.564337507796, 2122.245592342438]
    assert_allclose(liquid.fugacities(), fugacities_expect, rtol=1e-12)
    
    # Temperature derivatives
    dlnphis_dT_expect = [0.05691421137269392, 0.058786419948670225]
    assert_allclose(liquid.dlnphis_dT(), dlnphis_dT_expect, rtol=1e-12)
    
    dphis_dT_expect = [0.004996447873843315, 0.0020793203437609493]
    assert_allclose(liquid.dphis_dT(), dphis_dT_expect, rtol=1e-12)
    
    dfugacities_dT_expect = [199.8579149537326, 124.75922062565697]
    assert_allclose(liquid.dfugacities_dT(), dfugacities_dT_expect, rtol=1e-12)
    
    # Pressure derivatives
    dlnphis_dP_expect = [-1e-05, -1e-05]
    assert_allclose(liquid.dlnphis_dP(), dlnphis_dP_expect, rtol=1e-12)
    
    dphis_dP_expect = [-8.778910843769491e-07, -3.537075987237397e-07]
    assert_allclose(liquid.dphis_dP(), dphis_dP_expect, rtol=1e-12)
    
    dfugacities_dP_expect = [0, 0]
    assert_allclose(liquid.dfugacities_dP(), dfugacities_dP_expect, atol=1e-15)
    
def test_EOSGas_phis():
    # Acetone, chloroform, methanol
    T = 331.42
    P = 90923
    zs = [0.229, 0.175, 0.596]
    
    eos_kwargs = {'Pcs': [4700000.0, 5330000.0, 8084000.0],
     'Tcs': [508.1, 536.2, 512.5],
     'omegas': [0.309, 0.21600000000000003, 0.5589999999999999],
     'kijs': [[0, 0.038, 0.08], [0.038, 0, 0.021], [0.08, 0.021, 0]],
    }
    
    HeatCapacityGases = [HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.3320002425347943e-21, 6.4063345232664645e-18, -1.251025808150141e-14, 1.2265314167534311e-11, -5.535306305509636e-09, -4.32538332013644e-08, 0.0010438724775716248, -0.19650919978971002, 63.84239495676709])),
     HeatCapacityGas(best_fit=(200.0, 1000.0, [1.5389278550737367e-21, -8.289631533963465e-18, 1.9149760160518977e-14, -2.470836671137373e-11, 1.9355882067011222e-08, -9.265600540761629e-06, 0.0024825718663005762, -0.21617464276832307, 48.149539665907696])),
     HeatCapacityGas(best_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924]))]
    
    gas = EOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    
    lnphis_expect = [-0.02360432649642419, -0.024402271514780954, -0.016769813943198587]
    assert_allclose(gas.lnphis(), lnphis_expect, rtol=1e-12)
    
    phis_expect = [0.9766720765776962, 0.9758930568084294, 0.9833700166511827]
    assert_allclose(gas.phis(), phis_expect, rtol=1e-12)
    
    fugacities_expect = [20335.64774507632, 15527.946770733744, 53288.92740628939]
    assert_allclose(gas.fugacities(), fugacities_expect, rtol=1e-12)
    
    dlnphis_dT_expect = [0.0001969437889400412, 0.0001955309568834383, 0.00014847122768410804]
    assert_allclose(gas.dlnphis_dT(), dlnphis_dT_expect, rtol=1e-12)
    
    dphis_dT_expect = [0.00019234949931314953, 0.00019081730321365582, 0.00014600215363994287]
    assert_allclose(gas.dphis_dT(), dphis_dT_expect, rtol=1e-12)
    
    dfugacities_dT_expect = [4.004979517465334, 3.036194290516665, 7.911872473981097]
    assert_allclose(gas.dfugacities_dT(), dfugacities_dT_expect, rtol=1e-12)
    
    dlnphis_dP_expect = [-2.6201216188562436e-07, -2.709988856769895e-07, -1.857038475967749e-07]
    assert_allclose(gas.dlnphis_dP(), dlnphis_dP_expect, rtol=1e-12)
    
    dphis_dP_expect = [-2.558999622374442e-07, -2.644659309349954e-07, -1.8261559570342924e-07]
    assert_allclose(gas.dphis_dP(), dphis_dP_expect, rtol=1e-12)
    
    dfugacities_dP_expect = [0.21832971850726046, 0.16657322866975469, 0.5761925710704517]
    assert_allclose(gas.dfugacities_dP(), dfugacities_dP_expect, rtol=1e-12)

    assert_allclose(gas.H(), 1725.7273210879043, rtol=1e-12)
    assert_allclose(gas.S(), 14.480694885134456, rtol=1e-12)
    assert_allclose(gas.Cp(), 58.748474752042945, rtol=1e-12)
    assert_allclose(gas.dH_dT(), gas.Cp(), rtol=1e-12)
    assert_allclose(gas.dH_dP(), -0.0017158255316092434, rtol=1e-12)
    assert_allclose(gas.dS_dT(), 0.17726291337892383, rtol=1e-12)
    assert_allclose(gas.dS_dP(), -9.480886482495667e-05, rtol=1e-12)
    
    dH_dzs_expect = [2227.672637816117, 1886.132133010868, 1210.163163133309]
    assert_allclose(gas.dH_dzs(), dH_dzs_expect, rtol=1e-12)
    dS_dzs_expect = [11.452747620043832, 12.611417881165302, 0.2036373977480378]
    assert_allclose(gas.dS_dzs(), dS_dzs_expect, rtol=1e-12)
    
    # Volumetric properties - should be implemented in the model only
    assert_allclose(gas.V(), 0.029705728448677898, rtol=1e-12)
    assert_allclose(gas.dP_dT(), 284.342003076555, rtol=1e-12)
    assert_allclose(gas.dP_dV(), -2999107.769105018, rtol=1e-12)
    assert_allclose(gas.d2P_dT2(), -0.009887846156943235, rtol=1e-12)
    assert_allclose(gas.d2P_dV2(), 197784608.40171462, rtol=1e-12)
    assert_allclose(gas.d2P_dTdV(), -9721.251806049266, rtol=1e-12)
    
    # Volumetric properties - base class
    assert_allclose(gas.Z(), 0.9801692315172096, rtol=1e-12)
    assert_allclose(gas.rho(), 33.66354074527018, rtol=1e-12)
    assert_allclose(gas.dT_dP(), 0.0035168915924488455, rtol=1e-12)
    assert_allclose(gas.dV_dT(), 9.480886482495666e-05, rtol=1e-12)
    assert_allclose(gas.dV_dP(), -3.334324995925092e-07, rtol=1e-12)
    assert_allclose(gas.dT_dV(), 10547.536898013452, rtol=1e-12)
    assert_allclose(gas.d2V_dP2(), 7.331895665172319e-12, rtol=1e-12)
    assert_allclose(gas.d2T_dP2(), 4.301091137792481e-10, rtol=1e-12)
    assert_allclose(gas.d2T_dV2(), 29492.455975795572, rtol=1e-12)
    assert_allclose(gas.d2V_dT2(), -2.513377829277684e-08, rtol=1e-12)
    assert_allclose(gas.d2V_dPdT(), -1.003984034506695e-09, rtol=1e-12)
    assert_allclose(gas.d2T_dPdV(), 0.12152750389888099, rtol=1e-12)
    # aliases
    assert_allclose(gas.d2V_dTdP(), gas.d2V_dPdT())
    assert_allclose(gas.d2T_dPdV(), gas.d2T_dVdP())
    assert_allclose(gas.d2P_dVdT(), gas.d2P_dTdV())
    # Compressibility factor
    assert_allclose(gas.dZ_dT(), 0.00017082651132311415, rtol=1e-12)
    assert_allclose(gas.dZ_dP(), -2.2171553318823896e-07, rtol=1e-12)
    
    # Derived properties
    assert_allclose(gas.PIP(), 0.9434309912868786, rtol=1e-12)
    assert_allclose(gas.kappa(), 1.1224518535829717e-05, rtol=1e-12)
    assert_allclose(gas.beta(), 0.0031916020840477414, rtol=1e-12)
    assert_allclose(gas.Joule_Thomson(), 2.9206299207786268e-05, rtol=1e-12)
    assert_allclose(gas.speed_of_sound(), 55.867443841933685, rtol=1e-12)
    
    # Molar density
    assert_allclose(gas.dP_drho(), 2646.5035764210666, rtol=1e-12)
    assert_allclose(gas.drho_dP(), 0.00037785703707694405, rtol=1e-12)
    assert_allclose(gas.d2P_drho2(), -3.2210736519363414, rtol=1e-12)
    assert_allclose(gas.d2rho_dP2(), 1.737733604711968e-10, rtol=1e-12)
    assert_allclose(gas.dT_drho(), -9.30746617730105, rtol=1e-12)
    assert_allclose(gas.d2T_drho2(), 0.5759354067635106, rtol=1e-12)
    assert_allclose(gas.drho_dT(), -0.10744062679903038, rtol=1e-12)
    assert_allclose(gas.d2rho_dT2(), 0.0007142979083006338, rtol=1e-12)
    assert_allclose(gas.d2P_dTdrho(), 8.578327173510202, rtol=1e-12)
    assert_allclose(gas.d2T_dPdrho(), -0.00010723955204780491, rtol=1e-12)
    assert_allclose(gas.d2rho_dPdT(), -1.274189795242708e-06, rtol=1e-12)

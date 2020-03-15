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
from fluids.numerics import derivative

from thermo.chemical_package import ChemicalConstantsPackage, PropertyCorrelationPackage
from thermo import Chemical, Mixture
from thermo.phases import *
from thermo.eos_mix import *
from thermo.eos import *
from thermo.vapor_pressure import VaporPressure
from thermo.volume import *
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


def test_GibbbsExcessLiquid_VolumeLiquids():
    # Binary ethanol-water
    T = 230.0
    P = 1e5
    zs = [.4, .6]
    
    MWs = [18.01528, 46.06844]
    Tcs = [647.14, 514.0]
    Pcs = [22048320.0, 6137000.0]
    Vcs = [5.6e-05, 0.000168]
    Zcs = [0.22947273972184645, 0.24125043269792068]
    omegas = [0.344, 0.635]
    
    eoss = [PR(Tc=Tcs[0], Pc=Pcs[0], omega=omegas[0], T=T, P=P),
            PR(Tc=Tcs[1], Pc=Pcs[1], omega=omegas[1], T=T, P=P)]
    
    m = Mixture(['water', 'ethanol'], zs=zs, T=T, P=P)
    
    VaporPressures = [VaporPressure(best_fit=(159.11, 514.7, [-2.3617526481119e-19, 7.318686894378096e-16, -9.835941684445551e-13, 7.518263303343784e-10, -3.598426432676194e-07, 0.00011171481063640762, -0.022458952185007635, 2.802615041941912, -166.43524219017118])),
                      VaporPressure(best_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))]
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                       HeatCapacityGas(best_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965]))]
    # HBT Pressure dependence needs Psats, Tc, Pc, omegas
    VolumeLiquids = [VolumeLiquid(best_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652]),
                                  Psat=VaporPressures[0], Tc=Tcs[0], Pc=Pcs[0], omega=omegas[0]),
                     VolumeLiquid(best_fit=(159.11, 504.71000000000004, [5.388587987308587e-23, -1.331077476340645e-19, 1.4083880805283782e-16, -8.327187308842775e-14, 3.006387047487587e-11, -6.781931902982022e-09, 9.331209920256822e-07, -7.153268618320437e-05, 0.0023871634205665524]),
                                  Psat=VaporPressures[1], Tc=Tcs[1], Pc=Pcs[1], omega=omegas[1])]
    
    EnthalpyVaporizations = [EnthalpyVaporization(best_fit=(273.17, 647.095, 647.14, [0.010220675607316746, 0.5442323619614213, 11.013674729940819, 110.72478547661254, 591.3170172192005, 1716.4863395285283, 4063.5975524922624, 17960.502354189244, 53916.28280689388])),
                              EnthalpyVaporization(best_fit=(159.11, 513.9999486, 514.0, [-0.002197958699297133, -0.1583773493009195, -4.716256555877727, -74.79765793302774, -675.8449382004112, -3387.5058752252276, -7531.327682252346, 5111.75264050548, 50774.16034043739]))]
    
    #VolumeLiquidMixtureArgs = dict(MWs=MWs, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, Zcs=Zcs, omegas=omegas, VolumeLiquids=VolumeLiquids)
    #obj = VolumeLiquidMixture(**VolumeLiquidMixtureArgs)
    
    liquid = GibbsExcessLiquid(VaporPressures=VaporPressures,HeatCapacityGases=HeatCapacityGases,
                               VolumeLiquids=VolumeLiquids,
                               EnthalpyVaporizations=EnthalpyVaporizations,
    #                           VolumeLiquidMixture=obj,
                               use_phis_sat=False, eos_pure_instances=eoss).to_TP_zs(T, P, zs)
    
    Vms_expect = [1.8214873182761534e-05, 5.44799706327522e-05]
    Vms_calc = liquid.Vms_sat()
    assert_allclose(Vms_expect, Vms_calc, rtol=1e-12)
    
    dVms_sat_dT_expect = [-4.55877309895312e-09, 5.1434298716332116e-08]
    dVms_sat_dT_calc = liquid.dVms_sat_dT()
    assert_allclose(dVms_sat_dT_expect, dVms_sat_dT_calc, rtol=1e-12)

    V_calc = liquid.V()
    assert_allclose(V_calc, 3.997393165275593e-05)
    
    
    
    liq2 = liquid.to_TP_zs(400, 1e6, zs)
    assert_allclose(liq2.V(), 4.8251068646661126e-05)
    
    # Old stuff - needs to be re-implemented
#    assert_allclose(liq2.dP_dV(), -22338535869771.266, rtol=1e-4)
#    assert_allclose(liq2.d2P_dV2(), -5.6652055039363195e+23, rtol=1e-4)
#    assert_allclose(liq2.dP_dT(), 4.420694273557085e-21, rtol=1e-4)
#    assert_allclose(liq2.d2P_dTdV(), 319083206136.0664, rtol=1e-4)
#    assert_allclose(liq2.d2P_dT2(), 1.0758054055476452e-22, rtol=1e-4)
#    
#    assert liq2.PIP() > 1# Yes, liquid
    
    assert_allclose(liquid.H(), -40657.50045812148, rtol=1e-10)
    assert_allclose(liquid.Hvaps(), [46687.6343559442, 45719.87039687816])
    
    # dH_dT needs to be implemented with the revised version(s)
#    dH_dT_numeric = derivative(lambda T: liquid.to_TP_zs(T, P, zs).H(), 300, dx=1e-1, n=1, order=15)
#    dH_dT_implemented = liquid.to_TP_zs(300, P, zs).Cp()
#    assert_allclose(dH_dT_implemented, dH_dT_numeric, rtol=1e-9)
#    assert_allclose(dH_dT_implemented, 103.23645638614174, rtol=1e-12)

    

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
    assert_allclose(gas.speed_of_sound(), (gas.dP_drho_S())**0.5, rtol=1e-11)
    
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
    
    # ideal gas heat capacity functions
    # Special speed-heavy functions are implemented, so tests are good.
    Ts = [25, 75, 200, 500, 1000, 2000]
    Cps_expect = [[40.07347056476225, 33.29375955093512, 31.648314772557875],
     [45.85126998195463, 39.28949465629621, 35.173569709777134],
     [60.29576852493555, 54.27883241969893, 39.860887753971014],
     [107.85718893790086, 80.57497980236016, 59.60897906926993],
     [162.03590079225324, 95.5180597032743, 89.53718866129503],
     [235.64700575314473, 111.94330438696761, 133.75492691926814]]
    
    integrals_expect = [[-15359.743127401309, -13549.348836281308, -10504.31342159649],
     [-13211.624613733387, -11734.767481100524, -8829.670918204232],
     [-6577.434707052751, -5886.747038850828, -4113.998603256549],
     [18460.582821423024, 14891.995003616028, 10416.929377837709],
     [87563.90176282992, 59654.61279312357, 48496.162006301776],
     [286405.35503552883, 163385.2948382445, 160142.21979658338]]
    integrals_over_T_expect = [[-124.11377728159405, -107.78348964807205, -89.75780643024278],
     [-77.48455137296759, -68.50431529726663, -53.31873830843409],
     [-26.568337973571715, -23.79988041589195, -16.67690619528966],
     [46.53789612349661, 37.80485397009687, 26.33993249405455],
     [140.16601782591798, 99.2695080821947, 77.9324862384278],
     [275.0685207089067, 170.51771449925985, 153.56327376798978]]
    
    dCps_expect = [[0.11555598834384742, 0.11991470210722174, 0.07548394100635737], [0.11555598834384742, 0.11991470210722174, 0.056589692276255196], [0.1155559883438474, 0.11991470210722148, 0.0349394761861727], [0.15194986052287612, 0.05385782617451507, 0.08087587202769089], [0.07361110496090048, 0.016425244683690856, 0.044217738257974225], [0.07361110496089149, 0.016425244683693302, 0.04421773825797312]]
    
    
    integrals_calc = []
    integrals_over_T_calc = []
    Cps_calc = []
    dCps_dT_calc = []
    for i, T in enumerate(Ts):
        gas2 = gas.to_TP_zs(T=T, P=P, zs=zs)
        assert gas2.Cpgs_locked
        Cps_calc.append(gas2.Cpigs_pure())
        integrals_calc.append(gas2.Cpig_integrals_pure())
        integrals_over_T_calc.append(gas2.Cpig_integrals_over_T_pure())
        dCps_dT_calc.append(gas2.dCpigs_dT_pure())
    
    assert_allclose(Cps_expect, Cps_calc)
    assert_allclose(integrals_expect, integrals_calc)
    assert_allclose(integrals_over_T_expect, integrals_over_T_calc)
    assert_allclose(dCps_expect, dCps_dT_calc)

def test_chemical_potential():
    T, P = 200.0, 1e5
    zs = [0.229, 0.175, 0.596]
    
    eos_kwargs = {'Pcs': [4700000.0, 5330000.0, 8084000.0],
     'Tcs': [508.1, 536.2, 512.5],
     'omegas': [0.309, 0.21600000000000003, 0.5589999999999999],
    }
    
    HeatCapacityGases = [HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.3320002425347943e-21, 6.4063345232664645e-18, -1.251025808150141e-14, 1.2265314167534311e-11, -5.535306305509636e-09, -4.32538332013644e-08, 0.0010438724775716248, -0.19650919978971002, 63.84239495676709])),
     HeatCapacityGas(best_fit=(200.0, 1000.0, [1.5389278550737367e-21, -8.289631533963465e-18, 1.9149760160518977e-14, -2.470836671137373e-11, 1.9355882067011222e-08, -9.265600540761629e-06, 0.0024825718663005762, -0.21617464276832307, 48.149539665907696])),
     HeatCapacityGas(best_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924]))]
    
    Hfs = [-216070.0, -103510.0, -200700.0]
    Sfs = [-216.5, -110.0, -129.8]
    Gfs = [-151520.525, -70713.5, -162000.13]
    
    liquid = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, Hfs=Hfs,
                       Sfs=Sfs, Gfs=Gfs, T=T, P=P, zs=zs)
    mu_r_exp = [-188705.73988392908, -97907.9761772734, -193308.17514525697]
    mu_r_calc = liquid.chemical_potential()
    assert_allclose(mu_r_exp, mu_r_calc, rtol=1e-9)
    '''#Derived with:
    from sympy import *
    T = symbols('T')
    zs = z0, z1, z2 = symbols('z0, z1, z2')
    Hb = symbols('Hbase', cls=Function)
    Hbase = Hb(z0, z1, z2)
    Hfs = Hf0, Hf1, Hf2 = symbols('Hf0, Hf1, Hf2')
    for zi, Hf in zip(zs, Hfs):
        Hbase += zi*Hf
    Sb = symbols('Sbase', cls=Function)
    Sbase = Sb(z0, z1, z2)
    Sfs = Sf0, Sf1, Sf2 = symbols('Sf0, Sf1, Sf2')
    for zi, Sf in zip(zs, Sfs):
        Sbase += zi*Sf
    
    G = Hbase - T*Sbase
    diff(G, z0)
    '''
    
    
    
    # Random gamma example
    gammas_expect = [1.8877873731435573, 1.52276935445383, 1.5173639948878495]
    assert_allclose(liquid.gammas(), gammas_expect, rtol=1e-12)
    
    gammas_parent = super(EOSLiquid, liquid).gammas()
    assert_allclose(gammas_parent, gammas_expect, rtol=1e-12)
    
    
def test_EOSGas_volume_HSGUA_derivatives():
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 
                                                                  4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924])), ]
    kwargs = dict(eos_kwargs=dict(Tcs=[512.5], Pcs=[8084000.0], omegas=[.559]),
                 HeatCapacityGases=HeatCapacityGases)
    gas = EOSGas(PRMIX, T=330, P=1e5, zs=[1], **kwargs)
    
    dH_dT_V_num = derivative(lambda T: gas.to(V=gas.V(), T=T, zs=gas.zs).H(), gas.T, dx=gas.T*1e-6)
    assert_allclose(dH_dT_V_num, gas.dH_dT_V(), rtol=1e-8)
    
    dH_dP_V_num = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).H(), gas.P, dx=gas.P*1e-7)
    assert_allclose(dH_dP_V_num, gas.dH_dP_V(), rtol=1e-8)
    
    dH_dV_T_num = derivative(lambda V: gas.to(V=V, T=gas.T, zs=gas.zs).H(), gas.V(), dx=gas.V()*2e-7)
    assert_allclose(dH_dV_T_num, gas.dH_dV_T(), rtol=1e-8)
    
    dH_dV_P_num = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).H(), gas.V(), dx=gas.V()*1e-7)
    assert_allclose(dH_dV_P_num, gas.dH_dV_P(), rtol=1e-8)    
    
    
    dS_dT_V_num = derivative(lambda T: gas.to(V=gas.V(), T=T, zs=gas.zs).S(), gas.T, dx=gas.T*1e-8)
    assert_allclose(gas.dS_dT_V(), dS_dT_V_num, rtol=1e-7)
    
    dS_dP_V_num = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).S(), gas.P, dx=gas.P*1e-7)
    assert_allclose(gas.dS_dP_V(), dS_dP_V_num, rtol=1e-7)
    
    dS_dV_T_num = derivative(lambda V: gas.to(V=V, T=gas.T, zs=gas.zs).S(), gas.V(), dx=gas.V()*1e-7)
    assert_allclose(gas.dS_dV_T(), dS_dV_T_num, rtol=1e-7)
    
    dS_dV_P_num = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).S(), gas.V(), dx=gas.V()*1e-7)
    assert_allclose(gas.dS_dV_P(), dS_dV_P_num, rtol=1e-7)
    

    dG_dT_V_num = derivative(lambda T: gas.to(V=gas.V(), T=T, zs=gas.zs).G(), gas.T, dx=gas.T*1e-8)
    assert_allclose(gas.dG_dT_V(), dG_dT_V_num, rtol=1e-7)
    
    dG_dP_V_num = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).G(), gas.P, dx=gas.P*1e-6)
    assert_allclose(gas.dG_dP_V(), dG_dP_V_num, rtol=1e-7)
    
    dG_dV_T_num = derivative(lambda V: gas.to(V=V, T=gas.T, zs=gas.zs).G(), gas.V(), dx=gas.V()*1e-6)
    assert_allclose(gas.dG_dV_T(), dG_dV_T_num, rtol=1e-7)
    
    dG_dV_P_num = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).G(), gas.V(), dx=gas.V()*1e-6)
    assert_allclose(gas.dG_dV_P(), dG_dV_P_num, rtol=1e-7)
    
    dU_dT_V_num = derivative(lambda T: gas.to(V=gas.V(), T=T, zs=gas.zs).U(), gas.T, dx=gas.T*1e-8)
    assert_allclose(gas.dU_dT_V(), dU_dT_V_num)
    
    dU_dP_V_num = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).U(), gas.P, dx=gas.P*1e-8)
    assert_allclose(gas.dU_dP_V(), dU_dP_V_num)
    
    dU_dV_T_num = derivative(lambda V: gas.to(V=V, T=gas.T, zs=gas.zs).U(), gas.V(), dx=gas.V()*1e-8)
    assert_allclose(gas.dU_dV_T(), dU_dV_T_num)
    
    dU_dV_P_num = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).U(), gas.V(), dx=gas.V()*1e-8)
    assert_allclose(gas.dU_dV_P(), dU_dV_P_num)
    
    dA_dT_V_num = derivative(lambda T: gas.to(V=gas.V(), T=T, zs=gas.zs).A(), gas.T, dx=gas.T*1e-8)
    assert_allclose(gas.dA_dT_V(), dA_dT_V_num)
    
    dA_dP_V_num = derivative(lambda P: gas.to(V=gas.V(), P=P, zs=gas.zs).A(), gas.P, dx=gas.P*1e-7)
    assert_allclose(gas.dA_dP_V(), dA_dP_V_num)
    
    dA_dV_T_num = derivative(lambda V: gas.to(V=V, T=gas.T, zs=gas.zs).A(), gas.V(), dx=gas.V()*1e-8)
    assert_allclose(gas.dA_dV_T(), dA_dV_T_num)
    
    dA_dV_P_num = derivative(lambda V: gas.to(V=V, P=gas.P, zs=gas.zs).A(), gas.V(), dx=gas.V()*1e-8)
    assert_allclose(gas.dA_dV_P(), dA_dV_P_num)
    
    
def test_CoolPropPhase_PR_pure():
    T, P = 299.0, 1e5
    CPP = CoolPropPhase('PR', 'nHexane', T=T, P=P)
    # Make a duplicate fluid using the parameters from CoolProp
    Tc, Pc, omega = CPP.AS.T_critical(), CPP.AS.p_critical(), CPP.AS.acentric_factor()

    # These have been tested by varing the PRMIX coefficients - they are correct,
    # and can get much closer as well when that is the case
    eos = PRMIX(Tcs=[Tc], Pcs=[Pc], omegas=[omega], T=T, P=P, zs=[1])
    assert_allclose(CPP.V(), eos.V_l, rtol=1e-4)
    assert_allclose(CPP.dP_dT(), eos.dP_dT_l, rtol=3e-4)
    assert_allclose(CPP.dP_dV(), eos.dP_dV_l, rtol=5e-4)
    assert_allclose(CPP.d2P_dT2(), eos.d2P_dT2_l, rtol=2e-4)
    assert_allclose(CPP.d2P_dV2(), eos.d2P_dV2_l, rtol=4e-4)
    assert_allclose(CPP.d2P_dTdV(), eos.d2P_dTdV_l, rtol=5e-4)


def test_CoolPropPhase_Water():
    T, P = 299.0, 1e5
    CPP = CoolPropPhase('HEOS', 'water', T=T, P=P)
    
    # Test the initialization methods
    CPP_TV = CPP.to(T=T, V=CPP.V(), zs=[1.0])
    assert_allclose(CPP_TV.P, P, rtol=1e-8)
    
    CPP_PV = CPP.to(P=P, V=CPP.V(), zs=[1.0])
    assert_allclose(CPP_PV.T, T, rtol=1e-8)
    
    assert_allclose(CPP.H(), 1954.1678289799822)
    assert_allclose(CPP.S(), 6.829644373073796)
    assert_allclose(CPP.G(), CPP.AS.gibbsmolar(), rtol=1e-11)
    assert_allclose(CPP.U(), CPP.AS.umolar(), rtol=1e-11)
    assert_allclose(CPP.A(), CPP.AS.helmholtzmolar(), rtol=1e-9) # This one cannot go lower prec oddly
    
    dH_dT_num = derivative(lambda T: CPP.to(T=T, P=P, zs=[1]).H(), T, dx=.01)
    assert_allclose(CPP.dH_dT(), dH_dT_num)
    
    dH_dP_num = derivative(lambda P: CPP.to(T=T, P=P, zs=[1]).H(), P, dx=P*2e-5, order=7)
    assert_allclose(CPP.dH_dP(), dH_dP_num, rtol=5e-6)
    
    d2H_dT2_num = derivative(lambda T: CPP.to(T=T, P=P, zs=[1]).dH_dT(), T, dx=T*1e-5, n=1)
    assert_allclose(CPP.d2H_dT2(), d2H_dT2_num)
    
    d2H_dP2_num = derivative(lambda P: CPP.to(T=T, P=P, zs=[1]).dH_dP(), P, dx=P*1e-3, n=1, order=5)
    assert_allclose(CPP.d2H_dP2(), d2H_dP2_num, rtol=5e-6)
    
    d2H_dTdP_num = derivative(lambda T: CPP.to(T=T, P=P, zs=[1]).dH_dP(), T, dx=T*1e-5, n=1)
    assert_allclose(CPP.d2H_dTdP(), d2H_dTdP_num)
    
    dH_dT_V_num = derivative(lambda T: CPP.to(T=T, V=CPP.V(), zs=[1]).H(), T, dx=.001)
    assert_allclose(CPP.dH_dT_V(), dH_dT_V_num)
    
    dH_dP_V_num = derivative(lambda P: CPP.to(P=P, V=CPP.V(), zs=[1]).H(), P, dx=P*3e-5)
    assert_allclose(CPP.dH_dP_V(), dH_dP_V_num, rtol=5e-6)
    
    dH_dV_T_num = derivative(lambda V: CPP.to(T=T, V=V, zs=[1]).H(), CPP.V(), dx=CPP.V()*1e-5)
    assert_allclose(CPP.dH_dV_T(), dH_dV_T_num)
    
    dH_dV_P_num = derivative(lambda V: CPP.to(P=P, V=V, zs=[1]).H(), CPP.V(), dx=CPP.V()*1e-6)
    assert_allclose(CPP.dH_dV_P(), dH_dV_P_num)
    
    dS_dT_num = derivative(lambda T: CPP.to(T=T, P=P, zs=[1]).S(), T, dx=.01)
    assert_allclose(CPP.dS_dT(), dS_dT_num)
    
    dS_dP_num = derivative(lambda P: CPP.to(T=T, P=P, zs=[1]).S(), P, dx=P*2e-4, order=7)
    assert_allclose(CPP.dS_dP(), dS_dP_num, rtol=5e-5)
    
    d2S_dT2_num = derivative(lambda T: CPP.to(T=T, P=P, zs=[1]).dS_dT(), T, dx=T*1e-5, n=1)
    assert_allclose(CPP.d2S_dT2(), d2S_dT2_num)
    
    d2S_dP2_num = derivative(lambda P: CPP.to(T=T, P=P, zs=[1]).dS_dP(), P, dx=P*1e-3, n=1, order=5)
    assert_allclose(CPP.d2S_dP2(), d2S_dP2_num, rtol=5e-6)
    
    d2S_dTdP_num = derivative(lambda T: CPP.to(T=T, P=P, zs=[1]).dS_dP(), T, dx=T*1e-5, n=1)
    assert_allclose(CPP.d2S_dTdP(), d2S_dTdP_num)
    
    dS_dT_V_num = derivative(lambda T: CPP.to(T=T, V=CPP.V(), zs=[1]).S(), T, dx=.001)
    assert_allclose(CPP.dS_dT_V(), dS_dT_V_num)
    
    dS_dP_V_num = derivative(lambda P: CPP.to(P=P, V=CPP.V(), zs=[1]).S(), P, dx=P*3e-5)
    assert_allclose(CPP.dS_dP_V(), dS_dP_V_num, rtol=5e-6)
    
    dS_dV_T_num = derivative(lambda V: CPP.to(T=T, V=V, zs=[1]).S(), CPP.V(), dx=CPP.V()*1e-5)
    assert_allclose(CPP.dS_dV_T(), dS_dV_T_num)
    
    dS_dV_P_num = derivative(lambda V: CPP.to(P=P, V=V, zs=[1]).S(), CPP.V(), dx=CPP.V()*1e-6)
    assert_allclose(CPP.dS_dV_P(), dS_dV_P_num)
    
    
def test_model_hash():
    zs = [0.95, 0.05]
    T, P = 400.0, 1325753.6447835972*.96
    
    constants = ChemicalConstantsPackage(Tcs=[647.14, 615.0], Pcs=[22048320.0, 5100000.0], omegas=[0.344, 0.1869], MWs=[18.01528, 94.19904], CASs=['7732-18-5', '624-92-0'])

    properties = PropertyCorrelationPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                        HeatCapacityGas(best_fit=(273, 1000, [-1.575967061488898e-21, 8.453271073419098e-18, -1.921448640274908e-14, 2.3921686769873392e-11, -1.7525253961492494e-08, 7.512525679465744e-06, -0.0018211688612260338, 0.3869010410224839, 35.590034427486614])),],)
    properties2 = PropertyCorrelationPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                        HeatCapacityGas(best_fit=(273, 1000, [-1.575967061488898e-21, 8.453271073419098e-18, -1.921448640274908e-14, 2.3921686769873392e-11, -1.7525253961492494e-08, 7.512525679465744e-06, -0.0018211688612260338, 0.3869010410224839, 35.590034427486614])),],)
    HeatCapacityGases = properties.HeatCapacityGases
    
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = EOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(PRMIX, dict(eos_kwargs), HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq2 = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=properties2.HeatCapacityGases, T=T, P=P, zs=zs)
    liq3 = EOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties2.HeatCapacityGases, T=T, P=P, zs=zs)
    assert gas.model_hash() != liq.model_hash()
    assert liq2.model_hash() == liq.model_hash()
    assert liq3.model_hash() != liq.model_hash()


def test_dlnfugacities_SRK():
    T = 115.0
    P = 1e6
    zs = [0.4, 0.6]
    
    dlnfugacities_dns_expect = [[1.4378058197970829, -0.9585372131980551],
     [-0.958537213198055, 0.6390248087987035]]
    
    dlnfugacities_dns_l_expect = [[1.1560067098003597, -0.7706711398669063],
     [-0.7706711398669062, 0.5137807599112717]]
    
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                         HeatCapacityGas(best_fit=(273, 1000, [-1.575967061488898e-21, 8.453271073419098e-18, -1.921448640274908e-14, 2.3921686769873392e-11, -1.7525253961492494e-08, 7.512525679465744e-06, -0.0018211688612260338, 0.3869010410224839, 35.590034427486614])),]
    
    eos_kwargs = {'Pcs': [33.94E5, 46.04E5], 'Tcs': [126.1, 190.6], 'omegas': [0.04, 0.011]}
    gas = EOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    assert_allclose(gas.dlnfugacities_dns(), dlnfugacities_dns_expect, rtol=1e-9)
    assert_allclose(liq.dlnfugacities_dns(), dlnfugacities_dns_l_expect, rtol=1e-9)
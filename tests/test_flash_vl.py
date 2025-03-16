'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import *

import numpy as np
from chemicals.utils import *
from fluids.core import C2K
from fluids.numerics import *

from thermo import *
from thermo.phases.phase_utils import lnphis_direct
from thermo.unifac import DOUFIP2006, UFIP
import json

def test_C2_C5_PR():
    T, P = 300, 3e6
    constants = ChemicalConstantsPackage(Tcs=[305.32, 469.7], Pcs=[4872000.0, 3370000.0],
                                         omegas=[0.098, 0.251], Tms=[90.3, 143.15],
                                         Tbs=[184.55, 309.21], CASs=['74-84-0', '109-66-0'],
                                         names=['ethane', 'pentane'], MWs=[30.06904, 72.14878])
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                         HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    zs = ws_to_zs([.5, .5], constants.MWs)


    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)

    flasher = FlashVL(constants, correlations, liquid=liq, gas=gas)
    # Check there are two phases near the dew point. don't bother checking the composition most of the time.

    # When this test was written, case is still valid for a dP of 0.00000001 Pa

    # Issue here was that (sum_criteria < 1e-7) was the check in the stability test result interpretation
    # Fixed it by decreasing the tolerance 10x (1e-8)
    res = flasher.flash(P=5475649.470049857+15, T=123.3+273.15, zs=zs)
    assert_close1d(res.betas, [0.9999995457838572, 4.5421614280893863e-07], rtol=1e-4)
    assert_close1d(res.gas.zs, [0.7058337751720506, 0.29416622482794935], rtol=1e-4)
    assert_close1d(res.liquid0.zs, [0.49517964670906095, 0.504820353290939], rtol=1e-4)


    # # In this case, the tolerance had to be decreased 10x more - to 1e-9! Triggered at a dP of 0.5
    res = flasher.flash(P=5475649.470049857+0.5, T=123.3+273.15, zs=zs)
    assert_close1d(res.betas, [0.999999984859061, 1.5140938947055815e-08], rtol=1e-4)
    assert_close1d(res.gas.zs, [0.7058336826506021, 0.29416631734939785])
    assert_close1d(res.liquid0.zs, [0.4951780663825745, 0.5048219336174254])

    # # This one is too close to the border - the VF from SS is less than 0,
    # # but if the tolerance is increased, it is positive (and should be)
    res = flasher.flash(P=5475649.470049857+0.001, T=123.3+273.15, zs=zs)
    assert_close1d(res.betas, [0.9999999999697144, 3.028555184414472e-11], rtol=3e-3)
    assert_close1d(res.gas.zs, [0.7058336794959247, 0.29416632050407526])
    assert_close1d(res.liquid0.zs, [0.49517801199759515, 0.5048219880024049])

    # This one is presently identified as a LL...  just check the number of phases
    assert flasher.flash(zs=zs, P=6.615e6, T=386).phase_count == 2

    # Check we can store this
    output = json.loads(json.dumps(flasher.as_json()))
    new_flasher = FlashVL.from_json(output)
    assert new_flasher == flasher

    json_data = res.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_eq_state = EquilibriumState.from_json(json_data)
    assert new_eq_state == res

def test_flash_TP_K_composition_idependent_unhappiness():
    constants = ChemicalConstantsPackage(Tcs=[508.1, 536.2, 512.5], Pcs=[4700000.0, 5330000.0, 8084000.0], omegas=[0.309, 0.21600000000000003, 0.5589999999999999],
                                         MWs=[58.07914, 119.37764000000001, 32.04186], CASs=['67-64-1', '67-66-3', '67-56-1'], names=['acetone', 'chloroform', 'methanol'])

    HeatCapacityGases = [HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.3320002425347943e-21, 6.4063345232664645e-18, -1.251025808150141e-14, 1.2265314167534311e-11, -5.535306305509636e-09, -4.32538332013644e-08, 0.0010438724775716248, -0.19650919978971002, 63.84239495676709])),
     HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.5389278550737367e-21, -8.289631533963465e-18, 1.9149760160518977e-14, -2.470836671137373e-11, 1.9355882067011222e-08, -9.265600540761629e-06, 0.0024825718663005762, -0.21617464276832307, 48.149539665907696])),
     HeatCapacityGas(poly_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924]))]
    VolumeLiquids = [VolumeLiquid(poly_fit=(178.51, 498.1, [6.564241965071999e-23, -1.6568522275506375e-19, 1.800261692081815e-16, -1.0988731296761538e-13, 4.118691518070104e-11, -9.701938804617744e-09, 1.4022905458596618e-06, -0.00011362923883050033, 0.0040109650220160956])),
                    VolumeLiquid(poly_fit=(209.63, 509.5799999999999, [2.034047306563089e-23, -5.45567626310959e-20, 6.331811062990084e-17, -4.149759318710192e-14, 1.6788970104955462e-11, -4.291900093120011e-09, 6.769385838271721e-07, -6.0166473220815445e-05, 0.0023740769479069054])),
                    VolumeLiquid(poly_fit=(175.7, 502.5, [3.5725079384600736e-23, -9.031033742820083e-20, 9.819637959370411e-17, -5.993173551565636e-14, 2.2442465416964825e-11, -5.27776114586072e-09, 7.610461006178106e-07, -6.148574498547711e-05, 0.00216398089328537])),]

    VaporPressures = [VaporPressure(exp_poly_fit=(178.51, 508.09000000000003, [-1.3233111115238975e-19, 4.2217134794609376e-16, -5.861832547132719e-13, 4.6488594950801467e-10, -2.3199079844570237e-07, 7.548290741523459e-05, -0.015966705328994194, 2.093003523977292, -125.39006100979816])),
                      VaporPressure(exp_poly_fit=(207.15, 536.4, [-8.714046553871422e-20, 2.910491615051279e-16, -4.2588796020294357e-13, 3.580003116042944e-10, -1.902612144361103e-07, 6.614096470077095e-05, -0.01494801055978542, 2.079082613726621, -130.24643185169472])),
                      VaporPressure(exp_poly_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10, -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708]))]

    liquid = GibbsExcessLiquid(VaporPressures=VaporPressures, VolumeLiquids=VolumeLiquids,
                     HeatCapacityGases=HeatCapacityGases, use_Poynting=True,
                     use_phis_sat=False)

    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, HeatCapacityGases=HeatCapacityGases,
                                               VolumeLiquids=VolumeLiquids, VaporPressures=VaporPressures)

    T, P = 350.0, 1e6
    zs = [0.2, 0.0, 0.8]
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas':constants.omegas}
    gas = IdealGas(HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    flashN = FlashVLN(constants, correlations, liquids=[liquid], gas=gas)

    # Low - all K under zero
    res =  flashN.flash(T=T, P=P, zs=zs)
    assert_close(res.rho_mass(), 733.1047159397776)
    assert 1 == res.phase_count
    assert res.liquid0 is not None

    # High - all K above zero
    res = flashN.flash(T=430, P=1e4, zs=zs)
    assert 1 == res.phase_count
    assert res.gas is not None
    assert_close(res.rho_mass(), 0.10418751067559757)

    # One K value is under 1, rest are above - but that component has mole frac of zero
    res = flashN.flash(T=420, P=1e4, zs=zs)
    assert 1 == res.phase_count
    assert res.gas is not None

    # phis_at for liquids was broken, breaking this calculation
    res = flashN.flash(T=285.5, P=1e4, zs=zs)
    assert_close1d(res.betas, [0.21860038882559643, 0.7813996111744036])
    assert res.phase_count == 2

    # Two cases RR was working on Ks less than 1, and coming up with a made up VF
    # Need to check Ks first
    res = flashN.flash(T=300.0000, P=900000.0000, zs=[0.5, 0.1, 0.4],)
    assert 1 == res.phase_count
    assert res.gas is None
    res = flashN.flash(T=300.0000, P=900000.0000, zs=[.5, 0, .5])
    assert 1 == res.phase_count
    assert res.gas is None

def test_flash_combustion_products():
    P = 1e5
    T = 794.5305048838037
    zs = [0.5939849621247668, 0.112781954982051, 0.0676691730155464, 0.2255639098776358]
    constants = ChemicalConstantsPackage(atomss=[{'N': 2}, {'C': 1, 'O': 2}, {'O': 2}, {'H': 2, 'O': 1}], CASs=['7727-37-9', '124-38-9', '7782-44-7', '7732-18-5'], MWs=[28.0134, 44.0095, 31.9988, 18.01528], names=['nitrogen', 'carbon dioxide', 'oxygen', 'water'], omegas=[0.04, 0.2252, 0.021, 0.344], Pcs=[3394387.5, 7376460.0, 5042945.25, 22048320.0], Tbs=[77.355, 194.67, 90.18799999999999, 373.124], Tcs=[126.2, 304.2, 154.58, 647.14], Tms=[63.15, 216.65, 54.36, 273.15])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
                                               HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                           HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
                           HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
                           HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))])
    kijs = [[0.0, -0.0122, -0.0159, 0.0], [-0.0122, 0.0, 0.0, 0.0952], [-0.0159, 0.0, 0.0, 0.0], [0.0, 0.0952, 0.0, 0.0]]
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)

    flasher = FlashVL(constants, correlations, liquid=liq, gas=gas)
    res = flasher.flash(T=T, P=P, zs=zs)

    assert res.gas
    assert res.phase == 'V'

    json_data = res.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_eq_state = EquilibriumState.from_json(json_data)
    assert new_eq_state == res

def test_bubble_T_PR_VL():
    constants = ChemicalConstantsPackage(CASs=['124-38-9', '110-54-3'], MWs=[44.0095, 86.17536], names=['carbon dioxide', 'hexane'], omegas=[0.2252, 0.2975], Pcs=[7376460.0, 3025000.0], Tbs=[194.67, 341.87], Tcs=[304.2, 507.6], Tms=[216.65, 178.075])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
                                               HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
                       HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998]))])
    zs = [.5, .5]
    T = 300.0
    P = 1e6
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)

    flasher = FlashVL(constants, correlations, liquid=liq, gas=gas)
    res = flasher.flash(P=7.93e6, VF=0, zs=zs)
    assert_close(res.T, 419.0621213529388, rtol=1e-6)

    # Awesome points! Work well with new solver. Slower than wanted however.
    res = flasher.flash(P=8e6, VF=0, zs=zs)
    assert_close(res.T, 454.420461011768, rtol=1e-4)

    # This one is really
    res = flasher.flash(P=8e6, VF=1, zs=zs)
    assert_close(res.T, 422.0034892250092, rtol=1e-4)


    # TPD Tangent Plane Distance checks

    res = flasher.flash(P=1e6, VF=0, zs=zs)
    TPD_calc = TPD(res.T, res.liquid0.zs, res.liquid0.lnphis(), res.gas.zs, res.gas.lnphis())
    assert_close(TPD_calc, 0, atol=1e-6)

    res = flasher.flash(T=200, VF=0, zs=zs)
    TPD_calc = TPD(res.T, res.liquid0.zs, res.liquid0.lnphis(), res.gas.zs, res.gas.lnphis())
    assert_close(TPD_calc, 0, atol=1e-6)

    res = flasher.flash(P=1e6, VF=1, zs=zs)
    TPD_calc = TPD(res.T, res.gas.zs, res.gas.lnphis(), res.liquid0.zs, res.liquid0.lnphis())
    assert_close(TPD_calc, 0, atol=1e-6)


    res = flasher.flash(T=300, VF=1, zs=zs)
    TPD_calc = TPD(res.T, res.gas.zs, res.gas.lnphis(), res.liquid0.zs, res.liquid0.lnphis())
    assert_close(TPD_calc, 0, atol=1e-6)


def test_PR_four_bubble_dew_cases_VL():
    zs=[.5, .5]
    T=300.0
    P=1E6
    constants = ChemicalConstantsPackage(CASs=['98-01-1', '98-00-0'], MWs=[96.08406000000001, 98.09994], names=['2-furaldehyde', 'furfuryl alcohol'], omegas=[0.4522, 0.7340000000000001], Pcs=[5510000.0, 5350000.0], Tbs=[434.65, 441.15], Tcs=[670.0, 632.0], Tms=[235.9, 250.35])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
                                               HeatCapacityGases=[HeatCapacityGas(poly_fit=(298, 1000, [4.245751608816354e-21, -2.470461837781697e-17, 6.221823690784335e-14, -8.847967216702641e-11, 7.749899297737877e-08, -4.250059888737765e-05, 0.013882452355067994, -2.1404621487165327, 185.84988012691903])),
                           HeatCapacityGas(poly_fit=(250.35, 632.0, [-9.534610090167143e-20, 3.4583416772306854e-16, -5.304513883184021e-13, 4.410937690059558e-10, -2.0905505018557675e-07, 5.20661895325169e-05, -0.004134468659764938, -0.3746374641720497, 114.90130267531933]))])
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)

    flasher = FlashVL(constants, correlations, liquid=liq, gas=gas)
    assert_close(flasher.flash(P=1e6, VF=0, zs=zs).T, 539.1838522423529, rtol=1e-6)
    assert_close(flasher.flash(P=1e6, VF=1, zs=zs).T, 540.2081697501809, rtol=1e-6)
    assert_close(flasher.flash(T=600.0, VF=0, zs=zs).P, 2766476.7473238464, rtol=1e-6)
    assert_close(flasher.flash(T=600.0, VF=1, zs=zs).P, 2702616.6490743402, rtol=1e-6)


def test_C1_C10_PT_flash_VL():
    IDs = ['methane', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    zs=[.1]*10
    T=300.0
    P=1E5
    constants = ChemicalConstantsPackage(CASs=['74-82-8', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '110-54-3', '142-82-5', '111-65-9', '111-84-2', '124-18-5'], MWs=[16.04246, 30.06904, 44.09562, 58.1222, 72.14878, 86.17536, 100.20194000000001, 114.22852, 128.2551, 142.28168], names=['methane', 'ethane', 'propane', 'butane', 'pentane', 'hexane', 'heptane', 'octane', 'nonane', 'decane'], omegas=[0.008, 0.098, 0.152, 0.193, 0.251, 0.2975, 0.3457, 0.39399999999999996, 0.444, 0.49], Pcs=[4599000.0, 4872000.0, 4248000.0, 3796000.0, 3370000.0, 3025000.0, 2740000.0, 2490000.0, 2290000.0, 2110000.0], Tbs=[111.65, 184.55, 231.04, 272.65, 309.21, 341.87, 371.53, 398.77, 423.95, 447.25], Tcs=[190.56400000000002, 305.32, 369.83, 425.12, 469.7, 507.6, 540.2, 568.7, 594.6, 611.7], Tms=[90.75, 90.3, 85.5, 135.05, 143.15, 178.075, 182.15, 216.3, 219.9, 243.225])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
                                               HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                        HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                        HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
                        HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
                        HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
                        HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
                        HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.4046935863496273e-21, 5.8024177500786575e-18, -7.977871529098155e-15, 7.331444047402207e-13, 9.954400606484495e-09, -1.2112107913343475e-05, 0.0062964696142858104, -1.0843106737278825, 173.87692850911935])),
                        HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303])),
                        HeatCapacityGas(poly_fit=(200.0, 1000.0, [6.513870466670624e-22, -5.318305817618858e-18, 1.8015815307749625e-14, -3.370046452151828e-11, 3.840755097595374e-08, -2.7203677889897072e-05, 0.011224516822410626, -1.842793858054514, 247.3628627781443])),
                        HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735]))])
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)

    flasher = FlashVL(constants, correlations, liquid=liq, gas=gas)
    res = flasher.flash(T=T, P=P, zs=zs)
    assert_close(res.VF, 0.3933480634014041, rtol=1e-5)

    json_data = res.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_eq_state = EquilibriumState.from_json(json_data)
    assert new_eq_state == res


def test_combustion_products():
    from chemicals.combustion import fuel_air_spec_solver
    IDs = ['methane', 'carbon dioxide', 'ethane', 'propane',
           'isobutane', 'butane', '2-methylbutane', 'pentane',
           'hexane', 'nitrogen', 'oxygen', 'water']

    T = C2K(15)
    P = 1e5
    zs_fuel = [0.9652228316853225, 0.0059558310220860665, 0.018185509193506685, 0.004595963476244076,
               0.0009769695915451998, 0.001006970610302194, 0.000472984762445398, 0.0003239924667435125,
               0.0006639799746946288, 0.002594967217109564, 0.0, 0.0]
    zs_fuel = normalize(zs_fuel)
    zs_air = [0.0]*9 + [0.79, 0.21] + [0.0]

    constants, correlations = ChemicalConstantsPackage.from_IDs(IDs)
    combustion = fuel_air_spec_solver(zs_air=zs_air, zs_fuel=zs_fuel, CASs=constants.CASs,
                                      atomss=constants.atomss, n_fuel=1.0, O2_excess=0.1)
    zs = combustion['zs_out']

    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(PRMIX, eos_kwargs, T=T, P=P, zs=zs, HeatCapacityGases=correlations.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIX, eos_kwargs, T=T, P=P, zs=zs, HeatCapacityGases=correlations.HeatCapacityGases)

    flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)
    res = flasher.flash(T=400.0, P=1e5, zs=zs)
    assert res.phase_count == 1
    assert res.gas is not None

    # Check we can store this
    output = json.loads(json.dumps(flasher.as_json()))
    assert 'constants' not in output
    assert 'correlations' not in output
    assert 'settings' not in output
    new_flasher = FlashVL.from_json(output, {'constants': constants, 'correlations': correlations, 'settings': flasher.settings})
    assert new_flasher == flasher

    json_data = res.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_eq_state = EquilibriumState.from_json(json_data)
    assert new_eq_state == res

def test_furfuryl_alcohol_high_TP():
    # Legacy bug, don't even remember what the original issue was
    constants = ChemicalConstantsPackage(MWs=[98.09994, 18.01528], Tcs=[632.0, 647.14], Pcs=[5350000.0, 22048320.0], omegas=[0.734, 0.344], names=['furfuryl alcohol', 'water'], CASs=['98-00-0', '7732-18-5'])

    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(250.35, 632.0, [-9.534610090167143e-20, 3.4583416772306854e-16, -5.304513883184021e-13, 4.410937690059558e-10, -2.0905505018557675e-07, 5.20661895325169e-05, -0.004134468659764938, -0.3746374641720497, 114.90130267531933])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))])

    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    zs = [0.4444445555555555, 1-0.4444445555555555]
    T, P = 5774.577777777778, 220483199.99999997
    gas = CEOSGas(eos_class=PRMIX, eos_kwargs=eos_kwargs, T=T, P=P, zs=zs, HeatCapacityGases=correlations.HeatCapacityGases)
    liquid = CEOSLiquid(eos_class=PRMIX, eos_kwargs=eos_kwargs, T=T, P=P, zs=zs, HeatCapacityGases=correlations.HeatCapacityGases)
    flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)

    assert_close(flasher.flash(T=T, P=P, zs=zs).rho_mass(), 227.52709151903954)


def test_flash_GibbsExcessLiquid_ideal_Psat():
    # Binary water-ethanol
    T = 230.0
    P = 1e5
    zs = [.4, .6]

    MWs = [18.01528, 46.06844]
    Tcs = [647.086, 514.7]
    Pcs = [22048320.0, 6137000.0]
    omegas = [0.344, 0.635]

    VaporPressures = [VaporPressure(extrapolation='DIPPR101_ABC|DIPPR101_ABC', exp_poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317])),
                    VaporPressure(extrapolation='DIPPR101_ABC|DIPPR101_ABC', exp_poly_fit=(159.11, 514.7, [-2.3617526481119e-19, 7.318686894378096e-16, -9.835941684445551e-13, 7.518263303343784e-10, -3.598426432676194e-07, 0.00011171481063640762, -0.022458952185007635, 2.802615041941912, -166.43524219017118]))]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                       HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965]))]

    VolumeLiquids = [VolumeLiquid(poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652]),
                                  Psat=VaporPressures[0], Tc=Tcs[0], Pc=Pcs[0], omega=omegas[0]),
                     VolumeLiquid(poly_fit=(159.11, 504.71000000000004, [5.388587987308587e-23, -1.331077476340645e-19, 1.4083880805283782e-16, -8.327187308842775e-14, 3.006387047487587e-11, -6.781931902982022e-09, 9.331209920256822e-07, -7.153268618320437e-05, 0.0023871634205665524]),
                                  Psat=VaporPressures[1], Tc=Tcs[1], Pc=Pcs[1], omega=omegas[1])]

    EnthalpyVaporizations = [EnthalpyVaporization(Tc=647.14, poly_fit_ln_tau=(273.17, 647.095, 647.14, [0.010220675607316746, 0.5442323619614213, 11.013674729940819, 110.72478547661254, 591.3170172192005, 1716.4863395285283, 4063.5975524922624, 17960.502354189244, 53916.28280689388])),
                              EnthalpyVaporization(Tc=514.0, poly_fit_ln_tau=(159.11, 513.9999486, 514.0, [-0.002197958699297133, -0.1583773493009195, -4.716256555877727, -74.79765793302774, -675.8449382004112, -3387.5058752252276, -7531.327682252346, 5111.75264050548, 50774.16034043739]))]

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=MWs, CASs=['7732-18-5', '64-17-5'])
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, EnthalpyVaporizations=EnthalpyVaporizations,
                                               VolumeLiquids=VolumeLiquids, VaporPressures=VaporPressures, skip_missing=True)

    liquid = GibbsExcessLiquid(VaporPressures=VaporPressures,
                               HeatCapacityGases=HeatCapacityGases,
                               VolumeLiquids=VolumeLiquids,
                               EnthalpyVaporizations=EnthalpyVaporizations,
                               caloric_basis='Psat', equilibrium_basis='Psat',
                              T=T, P=P, zs=zs)

    gas = IdealGas(T=T, P=P, zs=zs, HeatCapacityGases=HeatCapacityGases)
    flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)

    # All points were missing because G_dep was missing
    res = flasher.flash(T=300, P=1e5, zs=zs)
    assert res.liquid_count == 1

    # Failing when two K values were under 1e-10
    res = flasher.flash(T=100, P=1e5, zs=zs)
    assert res.phase_count == 1
    assert res.liquid_count == 1

    # Wilson guessess are hard zeros
    res = flasher.flash(T=5, P=1e5, zs=zs)
    assert res.phase_count == 1
    assert res.liquid_count == 1

    # Wilson guesses inf, nan, and all zero
    res = flasher.flash(T=6.2, P=5e4, zs=zs)
    assert res.phase_count == 1
    assert res.liquid_count == 1

    # One (but not both) fugacity became zero
    res = flasher.flash(T=8.4, P=1e-5, zs=zs)
    assert res.phase_count == 1
    assert res.liquid_count == 1

    # Vapor fraction flashes
    for VF_value in (0.0, 1e-5, .3, .5, .7, 1-1e-5, 1.0):
        VF = flasher.flash(T=T, VF=VF_value, zs=zs)
        check = flasher.flash(T=T, P=VF.P, zs=zs)
        assert_close(VF.VF, check.VF, rtol=1e-9)

    # Not exactly sure where the numerical challenge is occuring, but this is to be expected.
    # The tolerance decays at very small numbers
    for VF_value in (1e-7, 1e-8, 1-1e-7, 1-1e-8):
        VF = flasher.flash(T=T, VF=VF_value, zs=zs)
        check = flasher.flash(T=T, P=VF.P, zs=zs)
        assert_close(VF.VF, check.VF, rtol=1e-5)

    # Check we can store this
    output = json.loads(json.dumps(flasher.as_json()))
    assert 'constants' not in output
    assert 'correlations' not in output
    assert 'settings' not in output
    new_flasher = FlashVL.from_json(output, {'constants': constants, 'correlations': correlations, 'settings': flasher.settings})
    assert new_flasher == flasher

def test_flash_GibbsExcessLiquid_ideal_PsatPoynting():
    # Binary water-ethanol
    T = 230.0
    P = 1e5
    zs = [.4, .6]

    MWs = [18.01528, 46.06844]
    Tcs = [647.086, 514.7]
    Pcs = [22048320.0, 6137000.0]
    omegas = [0.344, 0.635]

    VaporPressures = [VaporPressure(exp_poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317])),
                    VaporPressure(exp_poly_fit=(159.11, 514.7, [-2.3617526481119e-19, 7.318686894378096e-16, -9.835941684445551e-13, 7.518263303343784e-10, -3.598426432676194e-07, 0.00011171481063640762, -0.022458952185007635, 2.802615041941912, -166.43524219017118]))]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                       HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965]))]

    VolumeLiquids = [VolumeLiquid(poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652]),
                                  Psat=VaporPressures[0], Tc=Tcs[0], Pc=Pcs[0], omega=omegas[0]),
                     VolumeLiquid(poly_fit=(159.11, 504.71000000000004, [5.388587987308587e-23, -1.331077476340645e-19, 1.4083880805283782e-16, -8.327187308842775e-14, 3.006387047487587e-11, -6.781931902982022e-09, 9.331209920256822e-07, -7.153268618320437e-05, 0.0023871634205665524]),
                                  Psat=VaporPressures[1], Tc=Tcs[1], Pc=Pcs[1], omega=omegas[1])]

    EnthalpyVaporizations = [EnthalpyVaporization(Tc=647.14, poly_fit_ln_tau=(273.17, 647.095, 647.14, [0.010220675607316746, 0.5442323619614213, 11.013674729940819, 110.72478547661254, 591.3170172192005, 1716.4863395285283, 4063.5975524922624, 17960.502354189244, 53916.28280689388])),
                              EnthalpyVaporization(Tc=514.0, poly_fit_ln_tau=(159.11, 513.9999486, 514.0, [-0.002197958699297133, -0.1583773493009195, -4.716256555877727, -74.79765793302774, -675.8449382004112, -3387.5058752252276, -7531.327682252346, 5111.75264050548, 50774.16034043739]))]

    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=MWs, CASs=['7732-18-5', '64-17-5'])
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, EnthalpyVaporizations=EnthalpyVaporizations,
                                               VolumeLiquids=VolumeLiquids, VaporPressures=VaporPressures, skip_missing=True)

    eoss = [PR(Tc=Tcs[0], Pc=Pcs[0], omega=omegas[0], T=T, P=P),
            PR(Tc=Tcs[1], Pc=Pcs[1], omega=omegas[1], T=T, P=P)]

    liquid = GibbsExcessLiquid(VaporPressures=VaporPressures,
                               HeatCapacityGases=HeatCapacityGases,
                               VolumeLiquids=VolumeLiquids,
                               EnthalpyVaporizations=EnthalpyVaporizations,
                               caloric_basis='PhiSat', equilibrium_basis='PhiSat',
                               eos_pure_instances=eoss,
                              T=T, P=P, zs=zs)

    gas = IdealGas(T=T, P=P, zs=zs, HeatCapacityGases=HeatCapacityGases)
    flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)

    # This was failing in pypy for a while instead of CPython
    res = flasher.flash(T=15, P=1e5, zs=zs)
    assert res.phase_count == 1
    assert res.liquid_count == 1

    # Check we can store this
    output = json.loads(json.dumps(flasher.as_json()))
    assert 'constants' not in output
    assert 'correlations' not in output
    assert 'settings' not in output
    new_flasher = FlashVL.from_json(output, {'constants': constants, 'correlations': correlations, 'settings': flasher.settings})
    assert new_flasher == flasher

def test_PRTranslated_air_two_phase():
    HeatCapacityGases = [HeatCapacityGas(CASRN="7727-37-9", MW=28.0134, similarity_variable=0.07139440410660612, extrapolation="linear", method="TRCIG"),
     HeatCapacityGas(CASRN="7782-44-7", MW=31.9988, similarity_variable=0.06250234383789392, extrapolation="linear", method="TRCIG"),
     HeatCapacityGas(CASRN="7440-37-1", MW=39.948, similarity_variable=0.025032542304996495, extrapolation="linear", method="WEBBOOK_SHOMATE")]

    constants = ChemicalConstantsPackage(atom_fractions=[{'N': 1.0}, {'O': 1.0}, {'Ar': 1.0}], atomss=[{'N': 2}, {'O': 2}, {'Ar': 1}], MWs=[28.0134, 31.9988, 39.948], omegas=[0.04, 0.021, -0.004], Pcs=[3394387.5, 5042945.25, 4873732.5], phase_STPs=['g', 'g', 'g'], rhocs=[11173.1843575419, 13623.978201634878, 13351.134846461948], similarity_variables=[0.07139440410660612, 0.06250234383789392, 0.025032542304996495], Tbs=[77.355, 90.188, 87.302], Tcs=[126.2, 154.58, 150.8], Tms=[63.15, 54.36, 83.81], Vcs=[8.95e-05, 7.34e-05, 7.49e-05], Zcs=[0.2895282296391198, 0.2880002236716698, 0.29114409080360165])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases)

    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': [[0.0, -0.0159, -0.0004],
      [-0.0159, 0.0, 0.0089],
      [-0.0004, 0.0089, 0.0]],
     'cs': [-3.643e-06, -2.762e-06, -3.293e-06],
     'alpha_coeffs': [(0.1243, 0.8898, 2.0129),
      (0.2339, 0.8896, 1.3053),
      (0.1227, 0.9045, 1.8541)]}

    gas = CEOSGas(PRMIXTranslatedConsistent, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIXTranslatedConsistent, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
    zs = normalize([78.08, 20.95, .93])

    # This test case was finding some infinities in fugacities, very nasty
    res = flasher.flash(P=200000, T=1e-3, zs=zs)
    assert res.phase_count == 1
    assert res.liquid0 is not None

    res = flasher.flash(P=200000, H=-11909.90990990991, zs=zs)
    assert_close(res.gas_beta, 0.0010452858012690303, rtol=1e-5)

    json_data = res.as_json()
    json_data = json.loads(json.dumps(json_data))
    new_eq_state = EquilibriumState.from_json(json_data)
    assert new_eq_state == res

def test_issue106_Michelson_stability_test_log_zero():
    T = 500
    P = 1e12
    zs = [0, 0.81, 0.16, 0.02, 0.01]
    activity_model = NRTL(T=T, xs=zs, tau_as=[[0, -5.1549, 0, 0, 0], [5.8547, 0, -0.40926, 0, 0], [0, -0.39036, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], tau_bs=[[0, 2270.62, 284.966, 0, 0], [229.497, 0, 1479.46, 0, 0], [-216.256, 447.003, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], alpha_cs=[[0, 0.2, 0.3, 0, 0], [0.2, 0, 0.46, 0, 0], [0.3, 0.46, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    chemicals = ['nitrobenzene', 'water', 'aniline', 'hydrogen', 'methane']
    constants, properties = ChemicalConstantsPackage.from_IDs(chemicals)
    kijs = [[0.0, 0, 0, 0, 0], [0, 0.0, 0, 0, 0], [0, 0, 0.0, 0, 0], [0, 0, 0, 0.0, -0.0044], [0, 0, 0, -0.0044, 0.0]]
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, HeatCapacityGases=properties.HeatCapacityGases, eos_kwargs=eos_kwargs)

    liquid = GibbsExcessLiquid(
        VaporPressures=properties.VaporPressures,
        HeatCapacityGases=properties.HeatCapacityGases,
        VolumeLiquids=properties.VolumeLiquids,
        GibbsExcessModel=activity_model,
        equilibrium_basis='Psat', caloric_basis='Psat',
        T=T, P=P, zs=zs)

    conditions = {'T': 500, 'P': 1e12}
    flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
    assert liquid.to(T=T, P=P, zs=zs).G() < gas.to(T=T, P=P, zs=zs).G()
    res = flasher.flash(T=T, P=P, zs=zs)
    assert res.phase_count == 1
    assert res.liquid0 is not None
    assert isinstance(res.liquid0, GibbsExcessLiquid)


    # Also need a test for lnphis direct
    liquid = liquid.to(T=513.994, P=1e4, zs=zs)
    lnphis_args = liquid.lnphis_args()
    lnphis_from_args = lnphis_direct(zs, *lnphis_args)
    assert_close1d(lnphis_from_args, liquid.lnphis(), rtol=1e-13)


def test_NRTL_water_ethanol_sample():
    # 6 coefficients per row.
    # Sample parameters from Understanding Distillation Using Column Profile Maps, First Edition.
    #  Daniel Beneke, Mark Peters, David Glasser, and Diane Hildebrandt.
    # Nice random example except for the poor prediction ! Dew point is good
    # But the bubble point is 10 kPa too high.
    # Still it is a good test of asymmetric values and the required
    # input form.
    chemicals = ['water', 'ethanol']
    T = 273.15+70
    P = 1e5
    zs = [1-.252, .252]

    constants, properties = ChemicalConstantsPackage.from_IDs(chemicals)
    GE = NRTL(T=T, xs=zs, tau_as=[[0.0, 3.458], [-0.801, 0.0]], tau_bs=[[0.0, -586.1], [246.2, 0.0]],
                        alpha_cs=[[0, 0.0], [0.3, 0]])
    liquid = GibbsExcessLiquid(
        VaporPressures=properties.VaporPressures,
        HeatCapacityGases=properties.HeatCapacityGases,
        VolumeLiquids=properties.VolumeLiquids,
        GibbsExcessModel=GE,
        equilibrium_basis='Psat', caloric_basis='Psat',
        T=T, P=P, zs=zs)

    assert_close1d(liquid.gammas(), [1.1114056946393671, 2.5391220022675163], rtol=1e-6)
    assert_close1d(liquid.GibbsExcessModel.alphas(), [[0.0, 0.0], [0.3, 0.0]])
    assert_close1d(liquid.GibbsExcessModel.taus(), [[0.0, 1.7500005828354948], [-0.08352950604691833, 0.0]])


    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(IGMIX, HeatCapacityGases=properties.HeatCapacityGases, eos_kwargs=eos_kwargs)
    flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)

    # vapor pressure curve not included here so low tolerance
    assert_close(flasher.flash(T=T, VF=0, zs=zs).P, 72190.62175687613, rtol=4e-3)
    assert_close(flasher.flash(T=T, VF=1, zs=zs).P, 40542.73708315536, rtol=2e-3)
    assert_close(flasher.flash(T=300, VF=0.5, zs=zs).P,5763.42373196148, atol=20, rtol=1e-4)
    assert_close(flasher.flash(P=5763.42373196148, VF=0.5, zs=zs).T,300, atol=2, rtol=1e-4)

def test_UNIFAC_water_ethanol_sample():
    # As this test loads vapor pressure curves from disk, do not check values just convergence/consistency
    chemicals = ['water', 'ethanol']
    zs=[0.5, 0.5]
    P=6500
    T=298.15
    constants, properties = ChemicalConstantsPackage.from_IDs(chemicals)
    GE_plain = UNIFAC.from_subgroups(T=T, xs=zs, chemgroups=[{1: 1, 2: 1, 14: 1}, {16: 1}], version=0,
                            interaction_data=UFIP, subgroups=UFSG)
    GE_DO = UNIFAC.from_subgroups(T=T, xs=zs, chemgroups=[{1: 1, 2: 1, 14: 1}, {16: 1}], version=1,
                    interaction_data=DOUFIP2006, subgroups=DOUFSG)
    for GE in (GE_plain, GE_DO):
        liquid = GibbsExcessLiquid(
            VaporPressures=properties.VaporPressures,
            HeatCapacityGases=properties.HeatCapacityGases,
            VolumeLiquids=properties.VolumeLiquids,
            GibbsExcessModel=GE,
            equilibrium_basis='Psat', caloric_basis='Psat',
            T=T, P=P, zs=zs)


        if GE is GE_plain:
            assert_close1d(liquid.gammas_infinite_dilution(), [7.623846608529782, 2.662771526958551])
        else:
            gamma_inf = liquid.to(T=298.15, P=1e5, zs=zs).gammas_infinite_dilution()[1]
            assert_close(gamma_inf, 2.611252717452456) # 3.28 in ddbst free data


            gamma_inf = liquid.to(T=283.15, P=1e5, zs=zs).gammas_infinite_dilution()[0]
            assert_close(gamma_inf, 4.401784691406401) # 3.28 in ddbst free data

        eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
        gas = CEOSGas(IGMIX, HeatCapacityGases=properties.HeatCapacityGases, eos_kwargs=eos_kwargs)
        flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)


        base  = flasher.flash(T=T, P=P, zs=zs)
        assert_close1d(base.gas.fugacities(), base.liquid0.fugacities(), rtol=1e-5)

        TVF = flasher.flash(T=T, VF=base.VF, zs=zs)
        assert_close1d(TVF.gas.fugacities(), TVF.liquid0.fugacities(), rtol=1e-5)
        assert_close1d(base.gas.zs, TVF.gas.zs, rtol=1e-5)
        assert_close1d(base.liquid0.zs, TVF.liquid0.zs, rtol=1e-5)
        assert_close(base.VF, TVF.VF, rtol=1e-5)

        PVF = flasher.flash(P=P, VF=base.VF, zs=zs)
        assert_close1d(PVF.gas.fugacities(), PVF.liquid0.fugacities(), rtol=1e-5)
        assert_close1d(base.gas.zs, PVF.gas.zs, rtol=1e-5)
        assert_close1d(base.liquid0.zs, PVF.liquid0.zs, rtol=1e-5)
        assert_close(base.VF, PVF.VF, rtol=1e-5)

        json_data = base.as_json()
        json_data = json.loads(json.dumps(json_data))
        new_eq_state = EquilibriumState.from_json(json_data)
        assert new_eq_state == base
        assert base != TVF
        assert base != PVF


def test_UNIFAC_ternary_basic():
    chemicals = ['pentane', 'hexane', 'octane']
    zs=[.1, .4, .5]
    P=1e5
    T=298.15
    constants, correlations = ChemicalConstantsPackage.from_IDs(chemicals)
    GE = UNIFAC.from_subgroups(T=T, xs=zs, chemgroups=[{1: 2, 2: 3}, {1: 2, 2: 4}, {1: 2, 2: 6}], version=0,
                            interaction_data=UFIP, subgroups=UFSG)

    eoss = [PR(Tc=constants.Tcs[i], Pc=constants.Pcs[i], omega=constants.omegas[i], T=T, P=P) for i in range(constants.N)]
    liquid = GibbsExcessLiquid(
        VaporPressures=correlations.VaporPressures,
        HeatCapacityGases=correlations.HeatCapacityGases,
        VolumeLiquids=correlations.VolumeLiquids,
        GibbsExcessModel=GE,
        eos_pure_instances=eoss,
        equilibrium_basis='Poynting&PhiSat', caloric_basis='Poynting&PhiSat',
        T=T, P=P, zs=zs)


    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(IGMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
    flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)


    base  = flasher.flash(T=400, P=1E7, zs=zs)
    assert base.gas is None
    base.G()
    base.G_dep()
    base.liquid0.GibbsExcessModel.GE()


    res = flasher.flash(zs=zs, T=400, VF=0.5)
    xs_expect = [0.04428613261665119, 0.28125472768746834, 0.6744591396958806]
    ys_expect = [0.15571386738334897, 0.518745272312532, 0.32554086030411905]
    assert res.phase == 'VL'
    assert_close1d(res.liquid0.zs, xs_expect, rtol=5e-2)
    assert_close1d(res.gas.zs, ys_expect, rtol=5e-2)
    assert_close(res.P, 212263.260256, rtol=1e-1)

    # Check we can store this
    output = json.loads(json.dumps(flasher.as_json()))
    assert 'constants' not in output
    assert 'correlations' not in output
    assert 'settings' not in output
    new_flasher = FlashVL.from_json(output, {'constants': constants, 'correlations': correlations, 'settings': flasher.settings})
    assert new_flasher == flasher

def test_UNIFAC_binary_caloric_basic():
    # DDBST test question
    chemicals = ['hexane', '2-Butanone']
    zs=[.5, .5]
    P=1e5
    T=273.15+60
    constants, properties = ChemicalConstantsPackage.from_IDs(chemicals)
    GE = UNIFAC.from_subgroups(T=T, xs=zs, chemgroups=[{1: 2, 2: 4}, {1: 1, 2: 1, 18: 1}], version=0,
                            interaction_data=UFIP, subgroups=UFSG)

    eoss = [PR(Tc=constants.Tcs[i], Pc=constants.Pcs[i], omega=constants.omegas[i], T=T, P=P) for i in range(constants.N)]
    liquid = GibbsExcessLiquid(
        VaporPressures=properties.VaporPressures,
        HeatCapacityGases=properties.HeatCapacityGases,
        VolumeLiquids=properties.VolumeLiquids,
        GibbsExcessModel=GE,
        eos_pure_instances=eoss,
        equilibrium_basis='Psat', caloric_basis='Psat',
        T=T, P=P, zs=zs)


    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(IGMIX, HeatCapacityGases=properties.HeatCapacityGases, eos_kwargs=eos_kwargs)
    flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
    res = flasher.flash(zs=zs, T=273.15 + 60, P=3E5)

    assert_close(res.liquid0.GibbsExcessModel.GE(), 923.6411976689183)
    assert_close(res.liquid0.GibbsExcessModel.HE(), 854.7719336324379)
    assert_close(res.liquid0.GibbsExcessModel.CpE(), 1.2662038866442173)


def test_case_air_Odhran_2022_09_24():
    constants = ChemicalConstantsPackage(atomss=[{'N': 2}, {'O': 2}, {'Ar': 1}], CASs=['7727-37-9', '7782-44-7', '7440-37-1'], Gfgs=[0.0, 0.0, 0.0], Hfgs=[0.0, 0.0, 0.0], MWs=[28.0134, 31.9988, 39.948], names=['nitrogen', 'oxygen', 'argon'], omegas=[0.04, 0.021, -0.004], Pcs=[3394387.5, 5042945.25, 4873732.5], Tbs=[77.355, 90.188, 87.302], Tcs=[126.2, 154.58, 150.8], Tms=[63.15, 54.36, 83.81], Vcs=[8.95e-05, 7.34e-05, 7.49e-05])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 5000.0, [-2.8946147064530805e-27, 5.818275034669907e-23, -4.519291247965907e-19, 1.609185011831327e-15, -1.9359711365420902e-12, -3.146524643467457e-09, 9.702047239977565e-06, -0.0025354183983147998, 29.203539884897914])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 5000.0, [1.2283565107055296e-26, -2.7951451726762067e-22, 2.6334919674696596e-18, -1.3233024602402331e-14, 3.7962829852713573e-11, -6.128657296759334e-08, 4.962798152953606e-05, -0.009971787087565588, 29.48256621272467])),
    HeatCapacityGas(load_data=False, poly_fit=(298.0, 6000.0, [-3.314740108347415e-34, 9.351547491936385e-30, -1.1038315945910408e-25, 7.065840421054809e-22, -2.65775666980227e-18, 5.9597846928676876e-15, -7.7937356614881e-12, 5.408021118723734e-09, 20.785998601382207])),
    ],
    )

    eos_kwargs = {'Pcs': [3394387.5, 5042945.25, 4873732.5], 'Tcs': [126.2, 154.58, 150.8], 'omegas': [0.04, 0.021, -0.004], 'kijs': [[0.0, -0.0159, -0.0004], [-0.0159, 0.0, 0.0089], [-0.0004, 0.0089, 0.0]]}
    gas = CEOSGas(PRMIXTranslatedConsistent, eos_kwargs=eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIXTranslatedConsistent, eos_kwargs=eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases)
    flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)
    zs = normalize([78.08, 20.95, .93])
    res = flasher.flash(zs=zs, T=0.5185328860513607,  P=3000000.0)
    assert_close(res.G(), -25510.432580278408)

    # was going to 0
    res = flasher.flash(zs=zs, T=0.46078924591519266,  P=3000000.0)
    assert_close(res.G(), -26018.476463001498)

    # Test using another flasher
    flasher_gas = FlashVLN(constants, correlations, liquids=[], gas=gas)
    res = flasher_gas.flash(T=130, P=3328741.8516244013*2, zs=zs)
    res2 = flasher_gas.flash(P=res.P, H=res.H(), zs=zs)

    assert_close(res2.flash_convergence['err'], 0, atol=1e-2)

def test_case_water_sodium_and_zeros():
    constants = ChemicalConstantsPackage(atomss=[{'H': 2, 'O': 1}, {'Na': 1}, {'H': 2}, {'H': 1, 'Na': 1, 'O': 1}], CASs=['7732-18-5', '7440-23-5', '1333-74-0', '1310-73-2'], Gfgs=[-228554.325, 76969.44, 0.0, -193906.9625], Hfgs=[-241822.0, 107500.0, 0.0, -191000.0], MWs=[18.01528, 22.98977, 2.01588, 39.99711], names=['water', 'sodium', 'hydrogen', 'sodium hydroxide'], omegas=[0.344, -0.1055, -0.22, 0.477], Pcs=[22048320.0, 35464000.0, 1296960.0, 25000000.0], Tbs=[373.124, 1156.09, 20.271, 1661.15], Tcs=[647.14, 2573.0, 33.2, 2820.0], Tms=[273.15, 370.944, 13.99, 596.15], Vcs=[5.6e-05, 0.000116, 6.5e-05, 0.0002])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 5000.0, [-1.4661028760473505e-27, 2.752229097393049e-23, -1.7636768499746195e-19, 2.2554942241375344e-16, 2.532929281241331e-12, -1.3180823687866557e-08, 2.3666962694229476e-05, -0.005252595370280718, 33.462944538809126])),
    HeatCapacityGas(load_data=False, poly_fit=(1170.525, 6000.0, [-1.3246796108417689e-30, 4.236857111151157e-26, -5.815708502150833e-22, 4.475210625658004e-18, -2.114558525027583e-14, 1.8271286274419583e-10, -5.091880729386989e-07, 0.0004035206079178648, 20.741910980098048])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 5000.0, [5.854271757259617e-26, -1.3261214129640104e-21, 1.2447980660780873e-17, -6.250491551709368e-14, 1.807383550446052e-10, -3.014214088546821e-07, 0.0002737850827612277, -0.11358941399545286, 43.247808225524906])),
    HeatCapacityGas(load_data=False, poly_fit=(2500.0, 6000.0, [-2.1045067223485937e-30, 7.960143727635666e-26, -1.3180729761762297e-21, 1.2515437717053405e-17, -7.486900740763893e-14, 3.8623569890294093e-10, -2.1185736065594167e-06, 0.008091715347661994, 48.668881804783936])),
    ],
    VaporPressures=[VaporPressure(load_data=False, exp_poly_fit=(235.0, 647.096, [-2.2804872185158488e-20, 9.150413938315753e-17, -1.5948089451561768e-13, 1.583762733400446e-10, -9.868214067470077e-08, 3.996161556967839e-05, -0.01048791558325078, 1.7034790943304348, -125.70729281239332])),
    VaporPressure(load_data=False, exp_poly_fit=(924.0, 1118.0, [-4.263479695383464e-22, 3.7401921310828384e-18, -1.4432593653214248e-14, 3.2036171852741314e-11, -4.481802685738435e-08, 4.0568209476817133e-05, -0.02329727534188316, 7.81883748394923, -1179.3804758282047])),
    VaporPressure(load_data=False, exp_poly_fit=(14.0, 33.19, [1.9076671790976024e-10, -3.219485766812894e-08, 2.3032621093911396e-06, -8.984227746760556e-05, 0.0020165500049109265, -0.02373656021803824, 0.05217445467694446, 2.2934522914532782, -11.673258169126537])),
    VaporPressure(load_data=False, exp_poly_fit=(0.01, 2820.0, [-9.530401777801152e-23, 1.229909552121622e-18, -6.607515020735929e-15, 1.9131201752275348e-11, -3.230815306030239e-08, 3.221745270067264e-05, -0.018383895361035442, 5.569322257601897, -744.4957633056008])),
    ],
    )

    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(IGMIX, Hfs=constants.Hfgs, Gfs=constants.Gfgs, eos_kwargs=eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases)

    liquid = GibbsExcessLiquid(VaporPressures=correlations.VaporPressures, VolumeLiquids=correlations.VolumeLiquids,
                     HeatCapacityGases=correlations.HeatCapacityGases, equilibrium_basis='Psat',
                              Hfs=constants.Hfgs, Gfs=constants.Gfgs)

    flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)
    res = flasher.flash(zs=[.9, .1, 0, 0], T=296.85585149700563, P=1e5)
    assert res.phase_count == 1

def test_first_henry_pure_solvent():
    constants = ChemicalConstantsPackage(CASs=['7732-18-5', '74-82-8', '7782-44-7'], MWs=[18.01528, 16.04246, 31.9988], names=['water', 'methane', 'oxygen'], omegas=[0.344, 0.008, 0.021], Pcs=[22048320.0, 4599000.0, 5042945.25], Tbs=[373.124, 111.65, 90.188], Tcs=[647.14, 190.564, 154.58], Tms=[273.15, 90.75, 54.36], Vcs=[5.6e-05, 9.86e-05, 7.34e-05], Zcs=[0.2294727397218464, 0.2861971332411768, 0.2880002236716698])

    properties = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    VaporPressures=[VaporPressure(load_data=False, exp_poly_fit=(235.0, 647.096, [-2.2804872185158488e-20, 9.150413938315753e-17, -1.5948089451561768e-13, 1.583762733400446e-10, -9.868214067470077e-08, 3.996161556967839e-05, -0.01048791558325078, 1.7034790943304348, -125.70729281239332])),
    VaporPressure(load_data=False, exp_poly_fit=(91.0, 190.53, [9.622723229049567e-17, -8.612411529834985e-14, 2.9833595006425225e-11, -4.260777701614599e-09, -1.2594926603862017e-07, 0.00013955339672608612, -0.022179706326222424, 1.7235004480396063, -46.63780124626052])),
    VaporPressure(load_data=False, exp_poly_fit=(54.0, 154.7, [-9.531407063492216e-16, 9.489123009949052e-13, -4.093086833338262e-10, 1.0049035481323629e-07, -1.5482475796101542e-05, 0.0015527861689055723, -0.10130889779718814, 4.125643816700549, -72.5217726309476])),
    ],
    HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 5000.0, [-1.4661028760473505e-27, 2.752229097393049e-23, -1.7636768499746195e-19, 2.2554942241375344e-16, 2.532929281241331e-12, -1.3180823687866557e-08, 2.3666962694229476e-05, -0.005252595370280718, 33.462944538809126])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 5000.0, [4.8986184697537195e-26, -1.1318000255051273e-21, 1.090383509787202e-17, -5.664719389870236e-14, 1.7090042167602582e-10, -2.9728679808459997e-07, 0.00026565262671378613, -0.054476667747310976, 35.35366254807737])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 5000.0, [1.2283565107055296e-26, -2.7951451726762067e-22, 2.6334919674696596e-18, -1.3233024602402331e-14, 3.7962829852713573e-11, -6.128657296759334e-08, 4.962798152953606e-05, -0.009971787087565588, 29.48256621272467])),
    ],
    VolumeLiquids=[VolumeLiquid(load_data=False, poly_fit=(273.17, 637.096, [9.003072610498602e-24, -3.097008950027542e-20, 4.608271228765453e-17, -3.872669284187594e-14, 2.009922021889232e-11, -6.5962047297859515e-09, 1.3368112879131714e-06, -0.0001529876250360836, 0.007589247005014973])),
    VolumeLiquid(load_data=False, poly_fit=(90.8, 180.564, [7.730541828225242e-20, -7.911042356530568e-17, 3.519357637914696e-14, -8.885734012624522e-12, 1.3922694980104644e-09, -1.3860056394382418e-07, 8.560110533953112e-06, -0.0002997874342573979, 0.004589555868318713])),
    VolumeLiquid(load_data=False, poly_fit=(54.370999999999995, 144.58100000000002, [6.457909929992152e-20, -4.7825644162085234e-17, 1.5319533644419177e-14, -2.769251182054237e-12, 3.088256295705138e-10, -2.1749171236451567e-08, 9.448300475892969e-07, -2.3081894336449995e-05, 0.00026558114294435154])),
    ],
    )


    P = 1e5
    T = 300.0
    zs = [0.8, 0.15, 0.05]

    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}

    gas = CEOSGas(IGMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    #henry_params = tuple(IPDB.get_ip_asymmetric_matrix('ChemSep Henry', constants.CASs, p) for p in ('A', 'B', 'C', 'D', 'E', 'F'))
    henry_params = ([[0.0, 0.0, 0.0], [349.743, 0.0, 0.0], [198.51, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-13282.1, 0.0, 0.0], [-8544.73, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-51.9144, 0.0, 0.0], [-26.35, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0425831, 0.0, 0.0], [0.0083538, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


    liquid = GibbsExcessLiquid(VaporPressures=properties.VaporPressures,
                            HeatCapacityGases=properties.HeatCapacityGases,
                            VolumeLiquids=properties.VolumeLiquids,
                            henry_as=henry_params[0],henry_bs=henry_params[1],
                            henry_cs=henry_params[2],henry_ds=henry_params[3],
                            henry_es=henry_params[4],henry_fs=henry_params[5],
                            henry_components=[False, True, True],
                            T=T, P=P, zs=zs,
                            )
    flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
    PT = flasher.flash(T=T, P=P, zs=zs)

    assert_close(PT.VF, 0.20731270079400071, rtol=1e-12)
    assert_close1d(PT.gas.zs, [0.03536138984599165, 0.7234773060969654, 0.2411613040570431])
    assert_close1d(PT.liquid0.zs, [0.9999770849882174, 1.761818296336498e-05, 5.296828819383467e-06])

def test_henry_water_ethanol_solvent_only_water_parameters():
    constants = ChemicalConstantsPackage(CASs=['7732-18-5', '74-82-8', '7782-44-7', '64-17-5'], MWs=[18.01528, 16.04246, 31.9988, 46.06844], names=['water', 'methane', 'oxygen', 'ethanol'], omegas=[0.344, 0.008, 0.021, 0.635], Pcs=[22048320.0, 4599000.0, 5042945.25, 6137000.0], Tbs=[373.124, 111.65, 90.188, 351.39], Tcs=[647.14, 190.564, 154.58, 514.0], Tms=[273.15, 90.75, 54.36, 159.05], Vcs=[5.6e-05, 9.86e-05, 7.34e-05, 0.000168], Zcs=[0.2294727397218464, 0.2861971332411768, 0.2880002236716698, 0.24125043269792065])

    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    VaporPressures=[VaporPressure(load_data=False, exp_poly_fit=(235.0, 647.096, [-2.2804872185158488e-20, 9.150413938315753e-17, -1.5948089451561768e-13, 1.583762733400446e-10, -9.868214067470077e-08, 3.996161556967839e-05, -0.01048791558325078, 1.7034790943304348, -125.70729281239332])),
    VaporPressure(load_data=False, exp_poly_fit=(91.0, 190.53, [9.622723229049567e-17, -8.612411529834985e-14, 2.9833595006425225e-11, -4.260777701614599e-09, -1.2594926603862017e-07, 0.00013955339672608612, -0.022179706326222424, 1.7235004480396063, -46.63780124626052])),
    VaporPressure(load_data=False, exp_poly_fit=(54.0, 154.7, [-9.531407063492216e-16, 9.489123009949052e-13, -4.093086833338262e-10, 1.0049035481323629e-07, -1.5482475796101542e-05, 0.0015527861689055723, -0.10130889779718814, 4.125643816700549, -72.5217726309476])),
    VaporPressure(load_data=False, exp_poly_fit=(293.0, 513.92, [5.3777491325156017e-20, -1.6012106867195758e-16, 2.0458559761660556e-13, -1.4489862063851104e-10, 6.057246220896061e-08, -1.405779300981973e-05, 0.0010708367239337575, 0.2768117325567463, -46.96722914399204])),
    ],
    HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 5000.0, [-1.4661028760473505e-27, 2.752229097393049e-23, -1.7636768499746195e-19, 2.2554942241375344e-16, 2.532929281241331e-12, -1.3180823687866557e-08, 2.3666962694229476e-05, -0.005252595370280718, 33.462944538809126])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 5000.0, [4.8986184697537195e-26, -1.1318000255051273e-21, 1.090383509787202e-17, -5.664719389870236e-14, 1.7090042167602582e-10, -2.9728679808459997e-07, 0.00026565262671378613, -0.054476667747310976, 35.35366254807737])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 5000.0, [1.2283565107055296e-26, -2.7951451726762067e-22, 2.6334919674696596e-18, -1.3233024602402331e-14, 3.7962829852713573e-11, -6.128657296759334e-08, 4.962798152953606e-05, -0.009971787087565588, 29.48256621272467])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 3000.0, [1.6541603267750207e-24, -2.3723762468159123e-20, 1.4267114626956977e-16, -4.653769293350726e-13, 8.856693213126163e-10, -9.695705050605968e-07, 0.0005151923152039144, 0.022054256737010197, 34.05391204132863])),
    ],
    VolumeLiquids=[VolumeLiquid(load_data=False, poly_fit=(273.17, 637.096, [9.003072610498602e-24, -3.097008950027542e-20, 4.608271228765453e-17, -3.872669284187594e-14, 2.009922021889232e-11, -6.5962047297859515e-09, 1.3368112879131714e-06, -0.0001529876250360836, 0.007589247005014973])),
    VolumeLiquid(load_data=False, poly_fit=(90.8, 180.564, [7.730541828225242e-20, -7.911042356530568e-17, 3.519357637914696e-14, -8.885734012624522e-12, 1.3922694980104644e-09, -1.3860056394382418e-07, 8.560110533953112e-06, -0.0002997874342573979, 0.004589555868318713])),
    VolumeLiquid(load_data=False, poly_fit=(54.370999999999995, 144.58100000000002, [6.457909929992152e-20, -4.7825644162085234e-17, 1.5319533644419177e-14, -2.769251182054237e-12, 3.088256295705138e-10, -2.1749171236451567e-08, 9.448300475892969e-07, -2.3081894336449995e-05, 0.00026558114294435154])),
    VolumeLiquid(load_data=False, poly_fit=(159.11, 504.71000000000004, [5.388587987308847e-23, -1.3310774763407132e-19, 1.4083880805284546e-16, -8.327187308843255e-14, 3.006387047487769e-11, -6.781931902982453e-09, 9.331209920257441e-07, -7.153268618320931e-05, 0.00238716342056672])),
    ],
    )


    P = 1e5
    T = 300.0
    zs = [0.4, 0.15, 0.05, 0.4]
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}

    gas = CEOSGas(IGMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases)
    # henry_params = tuple( np.array(IPDB.get_ip_asymmetric_matrix('ChemSep Henry', constants.CASs, p)).tolist() for p in ('A', 'B', 'C', 'D', 'E', 'F'))
    henry_params = ([[0.0, 0.0, 0.0, 0.0], [349.743, 0.0, 0.0, 0.0], [198.51, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-13282.1, 0.0, 0.0, 0.0], [-8544.73, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-51.9144, 0.0, 0.0, 0.0], [-26.35, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0425831, 0.0, 0.0, 0.0], [0.0083538, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])


    liquid = GibbsExcessLiquid(VaporPressures=correlations.VaporPressures,
                            HeatCapacityGases=correlations.HeatCapacityGases,
                            VolumeLiquids=correlations.VolumeLiquids,
                            henry_as=henry_params[0],henry_bs=henry_params[1],
                            henry_cs=henry_params[2],henry_ds=henry_params[3],
                            henry_es=henry_params[4],henry_fs=henry_params[5],
                            henry_components=[False, True, True, False],
                            henry_mode='solvents_with_parameters', # solvents, solvents_with_parameters
                            T=T, P=P, zs=zs,
                            )
    dT = T*1e-8
    assert_close1d(liquid.Psats(), [3536.2200171225227, 4106424071.0937953, 4552937470.331796, 8753.810784136993])
    dlnHenry_matrix_dT_numerical = (np.array(liquid.to(T=T+dT, P=P, zs=zs).lnHenry_matrix()) - liquid.lnHenry_matrix())/dT
    assert_close2d(dlnHenry_matrix_dT_numerical, liquid.dlnHenry_matrix_dT(), rtol=1e-5)

    d2lnHenry_matrix_dT2_numerical = (np.array(liquid.to(T=T+dT, P=P, zs=zs).dlnHenry_matrix_dT()) - liquid.dlnHenry_matrix_dT())/dT
    assert_close2d(d2lnHenry_matrix_dT2_numerical, liquid.d2lnHenry_matrix_dT2(), rtol=1e-5)

    # First vapor pressure derivative
    dPsats_dT = liquid.dPsats_dT()
    dPsats_dT_num = jacobian(lambda T: liquid.to(T=T[0], P=P, zs=zs).Psats(), [T], scalar=False, perturbation=2e-7)
    dPsats_dT_num = [i[0] for i in dPsats_dT_num]
    assert_close1d(dPsats_dT, dPsats_dT_num, rtol=1e-5)

    # Second vapor pressure derivative
    d2Psats_dT2 = liquid.d2Psats_dT2()
    d2Psats_dT2_num = jacobian(lambda T: liquid.to(T=T[0], P=P, zs=zs).dPsats_dT(), [T], scalar=False, perturbation=2e-7)
    d2Psats_dT2_num = [i[0] for i in d2Psats_dT2_num]
    assert_close1d(d2Psats_dT2, d2Psats_dT2_num, rtol=1e-5)

    # Vapor pressure in different ways functions
    assert_close1d(liquid.Psats_T_ref(), liquid.to(T=liquid.T_REF_IG, P=1e5, zs=zs).Psats())
    assert_close1d(liquid.Psats_at(315.0), liquid.to(T=315.0, P=1e5, zs=zs).Psats())
    assert_close1d(liquid.dPsats_dT_at(315.0), liquid.to(T=315.0, P=1e5, zs=zs).dPsats_dT())
    assert_close1d(liquid.lnPsats(), [log(v) for v in liquid.to(T=T, P=P, zs=zs).Psats()])
    assert_close1d(liquid.dPsats_dT_over_Psats(), np.array(liquid.dPsats_dT())/liquid.Psats(), rtol=1e-12)
    assert_close1d(liquid.d2Psats_dT2_over_Psats(), np.array(liquid.d2Psats_dT2())/liquid.Psats(), rtol=1e-12)

    # Nasty function to calculate
    dlnPsats_dT = liquid.dlnPsats_dT()
    dlnPsats_dT_num = jacobian(lambda T: liquid.to(T=T[0], P=P, zs=zs).lnPsats(), [T], scalar=False, perturbation=2e-7)
    dlnPsats_dT_num = [i[0] for i in dlnPsats_dT_num]
    assert_close1d(dlnPsats_dT, dlnPsats_dT_num, rtol=1e-5)

    # Second derivative of same nasty function
    d2lnPsats_dT2 = liquid.d2lnPsats_dT2()
    d2lnPsats_dT2_num = jacobian(lambda T: liquid.to(T=T[0], P=P, zs=zs).dlnPsats_dT(), [T], scalar=False, perturbation=2e-7)
    d2lnPsats_dT2_num = [i[0] for i in d2lnPsats_dT2_num]
    assert_close1d(d2lnPsats_dT2, d2lnPsats_dT2_num, rtol=1e-5)

    flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)
    PT = flasher.flash(T=T, P=P, zs=zs)
    assert_close(PT.VF, 0.21303456001555365, rtol=1e-12)
    assert_close1d(PT.gas.zs, [0.017803525791727007, 0.7040477725323597, 0.2346846611296595, 0.04346404054625393])
    assert_close1d(PT.liquid0.zs, [0.5034620500286068, 1.7145033253831193e-05, 5.154576856346698e-06, 0.49651565036128303])


    # Check we can store this
    output = json.loads(json.dumps(flasher.as_json()))
    assert 'constants' not in output
    assert 'correlations' not in output
    assert 'settings' not in output
    new_flasher = FlashVL.from_json(output, {'constants': constants, 'correlations': correlations, 'settings': flasher.settings})
    assert new_flasher == flasher


def test_PRMIX_basics_H_S():
    chemicals = ['Ethane', 'Heptane']
    # constants, properties = ChemicalConstantsPackage.from_IDs(chemicals)
    # constants.subset(properties=['Tcs', 'Pcs', 'omegas', 'MWs', 'Vcs'])

    constants = ChemicalConstantsPackage(MWs=[30.06904, 100.20194000000001], omegas=[0.099, 0.349],
                                        Pcs=[4872200.0, 2736000.0], Tcs=[305.322, 540.13],
                                        Vcs=[0.0001455, 0.000428])
    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 1500.0, [-1.0480862560578738e-22, 6.795933556773635e-19, -1.752330995156058e-15, 2.1941287956874937e-12, -1.1560515172055718e-09, -1.8163596179818727e-07, 0.00044831921501838854, -0.038785639211185385, 34.10970704595796])),
    HeatCapacityGas(load_data=False, poly_fit=(200.0, 1500.0, [3.92133614210253e-22, -3.159591705025203e-18, 1.0953093194585358e-14, -2.131394945087635e-11, 2.5381451844763867e-08, -1.872671854270201e-05, 0.007985818706468728, -1.3181368580077415, 187.25540686626923])),
    ],
    )

    kij = 0
    kijs = [[0,kij],[kij,0]]
    zs = [0.4, 0.6]

    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases)

    flasher = FlashVL(constants, correlations, liquid=liq, gas=gas)

    res = flasher.flash(T=450, P=400, zs=zs)
    H1 = res.H()
    S1 = res.S()
    assert res.phase == 'V'
    res = flasher.flash(T=450, P=1e6, zs=zs)
    H2 = res.H()
    S2 = res.S()
    assert res.phase == 'V'
    assert_close(H1 - H2, 1638.193586065765, rtol=1e-7)
    assert_close(S1 - S2, 67.59097448573374, rtol=1e-5)

    # Case gas to VF= = 0 at same T
    res = flasher.flash(T=350, P=400, zs=zs)
    assert res.phase == 'V'
    S1 = res.S()
    H1 = res.H()
    res = flasher.flash(T=350, VF=.5, zs=zs)
    assert res.phase == 'VL'
    H2 = res.H()
    S2 = res.S()
    assert_close(H1 - H2, 16445.148662152467, rtol=1e-6)
    assert_close(S1 - S2, 96.84962967388968, rtol=1e-5)

    # Higher pressure (gas constant diff probably; gas-liquid difference! No partial phase.)
    res = flasher.flash(T=450, P=400, zs=zs)
    assert res.phase == 'V'
    H1 = res.H()
    S1 = res.S()
    res = flasher.flash(T=450, P=1e8, zs=zs)
    assert res.phase == 'L'
    H2 = res.H()
    S2 = res.S()
    assert_close(H1 - H2, 13815.671299952719, rtol=1e-7)
    assert_close(S1 - S2, 128.67198457877873, rtol=1e-5)

    # low P fluid to saturation pressure (both gas)
    res = flasher.flash(T=450, P=400, zs=zs)
    assert res.phase == 'V'
    H1 = res.H()
    S1 = res.S()
    res = flasher.flash(T=450, VF=1, zs=zs)
    assert res.phase == 'VL'
    assert res.gas_beta == 1
    H2 = res.H()
    S2 = res.S()
    assert_close(H1 - H2, 2003.8453690354618, rtol=1e-6)
    assert_close(S1 - S2, 69.64347719345321, rtol=1e-5)

    # low pressure gas to liquid saturated
    res = flasher.flash(T=350, P=400, zs=zs)
    assert res.phase == 'V'
    H1 = res.H()
    S1 = res.S()
    res = flasher.flash(T=350, VF=0, zs=zs)
    assert res.phase == 'VL'
    H2 = res.H()
    S2 = res.S()
    assert_close(H1 - H2, 23682.354847696974, rtol=1e-6)
    assert_close(S1 - S2, 124.44424015029476, rtol=1e-6)

    # High pressure liquid to partial evaporation
    res = flasher.flash(T=350, P=3e6, zs=zs)
    assert res.phase == 'L'
    H1 = res.H()
    S1 = res.S()
    res = flasher.flash(T=350, VF=.25, zs=zs)
    assert res.phase == 'VL'
    H2 = res.H()
    S2 = res.S()
    assert_close(H1 - H2, -2328.213423713927, rtol=1e-7)
    assert_close(S1 - S2, -7.913403132740896, rtol=1e-5)

    # High pressure temperature change
    res = flasher.flash(T=300, P=3e6, zs=zs)
    assert res.phase == 'L'
    S1 = res.S()
    H1 = res.H()
    res = flasher.flash(T=400, P=1e7, zs=zs)
    assert res.phase == 'L'
    H2 = res.H()
    S2 = res.S()
    assert_close(H1 - H2, -18470.021914554898, rtol=1e-5)
    assert_close(S1 - S2, -50.38009840930218, rtol=1e-5)

    # High pressure temperature change and phase change
    res = flasher.flash(T=300, P=3e6, zs=zs)
    assert res.phase == 'L'
    H1 = res.H()
    S1 = res.S()
    res = flasher.flash(T=400, P=1e5, zs=zs)
    assert res.phase == 'V'
    H2 = res.H()
    S2 = res.S()
    assert_close(H1 - H2, -39430.44410647196, rtol=1e-5)
    assert_close(S1 - S2, -124.39418852732291, rtol=1e-5)

def test_air_was_calling_wrong_lnphis_liquid():
    constants = ChemicalConstantsPackage(atomss=[{'H': 2, 'O': 1}, {'O': 2}, {'N': 2}, {'Ar': 1}], CASs=['7732-18-5', '7782-44-7', '7727-37-9', '7440-37-1'], Gfgs=[-228554.325, 0.0, 0.0, 0.0], Hfgs=[-241822.0, 0.0, 0.0, 0.0], MWs=[18.01528, 31.9988, 28.0134, 39.948], names=['water', 'oxygen', 'nitrogen', 'argon'], omegas=[0.344, 0.021, 0.04, -0.004], Pcs=[22048320.0, 5042945.25, 3394387.5, 4873732.5], Sfgs=[-44.499999999999964, 0.0, 0.0, 0.0], Tbs=[373.124, 90.188, 77.355, 87.302], Tcs=[647.14, 154.58, 126.2, 150.8], Vml_STPs=[1.8087205105724903e-05, 4.20717054123152e-05, 4.940428399628771e-05, 4.2812201767994384e-05], Vml_60Fs=[1.8036021352633123e-05, 4.20717054123152e-05, 4.940428399628771e-05, 4.2812201767994384e-05])

    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    VaporPressures=[VaporPressure(load_data=False, exp_poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317])),
    VaporPressure(load_data=False, exp_poly_fit=(54.370999999999995, 154.57100000000003, [-9.865296960381724e-16, 9.716055729011619e-13, -4.163287834047883e-10, 1.0193358930366495e-07, -1.57202974507404e-05, 0.0015832482627752501, -0.10389607830776562, 4.24779829961549, -74.89465804494587])),
    VaporPressure(load_data=False, exp_poly_fit=(63.2, 126.18199999999999, [5.490876411024536e-15, -3.709517805130509e-12, 1.0593254238679989e-09, -1.6344291780087318e-07, 1.4129990091975526e-05, -0.0005776268289835264, -0.004489180523814208, 1.511854256824242, -36.95425216567675])),
    VaporPressure(load_data=False, exp_poly_fit=(83.816, 150.67700000000002, [3.156255133278695e-15, -2.788016448186089e-12, 1.065580375727257e-09, -2.2940542608809444e-07, 3.024735996501385e-05, -0.0024702132398995436, 0.11819673125756014, -2.684020790786307, 20.312746972164785])),
    ],
    VolumeLiquids=[VolumeLiquid(load_data=False, poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652])),
    VolumeLiquid(load_data=False, poly_fit=(54.370999999999995, 144.58100000000002, [6.457909929992152e-20, -4.7825644162085234e-17, 1.5319533644419177e-14, -2.7692511820542383e-12, 3.088256295705142e-10, -2.1749171236451626e-08, 9.448300475893009e-07, -2.3081894336450133e-05, 0.00026558114294435354])),
    VolumeLiquid(load_data=False, poly_fit=(63.2, 116.192, [9.50261462694019e-19, -6.351064785670885e-16, 1.8491415360234833e-13, -3.061531642102745e-11, 3.151588109585604e-09, -2.0650965261816766e-07, 8.411110954342014e-06, -0.00019458305886755787, 0.0019857193167955463])),
    VolumeLiquid(load_data=False, poly_fit=(83.816, 140.687, [6.384785376493128e-19, -5.452304094035189e-16, 2.030873013507877e-13, -4.3082371353093367e-11, 5.691865080782158e-09, -4.794806581352254e-07, 2.5147558955587817e-05, -0.000750618147035446, 0.00978545135574999])),
    ],
    HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
    HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [-1.0939921922581918e-31, 4.144146614006628e-28, -6.289942296644484e-25, 4.873620648503505e-22, -2.0309301195845294e-19, 4.3863747689727484e-17, -4.29308508081826e-15, 20.786156545383236])),
    ],
    )

    zs_air = [0, 0.2096, 0.7812, 0.0092]
    gas = IdealGas(T=300, P=3e5, zs=zs_air, HeatCapacityGases=correlations.HeatCapacityGases)

    liquid = GibbsExcessLiquid(equilibrium_basis='Psat', caloric_basis='Psat',
            HeatCapacityGases=correlations.HeatCapacityGases,
            VolumeLiquids=correlations.VolumeLiquids, VaporPressures=correlations.VaporPressures)
    flasher = FlashVL(constants, correlations, liquid=liquid, gas=gas)

    # Bug involves a hard 0 for a compound
    res = flasher.flash(T=300, P=3e5, zs=zs_air)
    assert res.phase == 'V'

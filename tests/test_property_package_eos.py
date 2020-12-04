# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import numpy as np
from numpy.testing import assert_allclose
import pytest
from thermo.utils import TPD
from thermo.eos import *
from thermo.eos_mix import *
from scipy.misc import derivative
from scipy.optimize import minimize, newton
from math import log, exp, sqrt, log10
from thermo import Mixture
from thermo.property_package import *
from fluids.numerics import linspace, logspace, normalize
from thermo.property_package_constants import (PropertyPackageConstants, PR_PKG)


@pytest.mark.deprecated
def test_bubble_T_PR():
    # Copied to VL! Can't get last point to converge.
    Ps = np.logspace(np.log10(1e3), np.log10(8e6), 100).tolist()
    # Value working for sure!
    # A long enough list of points may reveal errors
    # Need to check for singularities in results!
    # Lagrange multiplier is needed.
    T_bubbles_expect = [135.77792634341301, 136.56179975223873, 137.35592304111714, 138.1605125904237, 138.97579118069618, 139.80198815378043, 140.63933971310234, 141.48808915266713, 142.34848716775062, 143.22079210796352, 144.10527026879004, 145.00219623035326, 145.9118531621595, 146.8345331709676, 147.77053765471518, 148.7201776796149, 149.68377437184307, 150.66165932879846, 151.65417505244912, 152.6616753977778, 153.68452605664353, 154.72310505184726, 155.7778032642612, 156.8490249894867, 157.937188514101, 159.04272673536184, 160.16608780166473, 161.30773579673297, 162.46815145564204, 163.64783292476886, 164.84729656230823, 166.06707778415586, 167.30773196086088, 168.56983536585116, 169.8539861804285, 171.16080556094636, 172.49093877035423, 173.84505638241404, 175.22385556194536, 176.6280614293828, 178.058428515323, 179.51574231484207, 181.00082094865053, 182.5145169422077, 184.0577191341151, 185.63135472512306, 187.2363914833706, 188.8738401205766, 190.54475685783353, 192.25024620138348, 193.991463951159, 195.76962046909824, 197.5859842371162, 199.4418857394953, 201.33872170960848, 203.27795978657647, 205.26114363572563, 207.28989859303456, 209.36593790645554, 211.49106965667633, 213.66720445521423, 215.89636403432021, 218.18069086349888, 220.52245895198226, 222.92408602593875, 225.3881473051149, 227.91739114691686, 230.5147568796014, 233.18339521130144, 235.92669168167328, 238.74829372436815, 241.65214202994656, 244.64250705759693, 247.7240317371467, 250.90178165300227, 254.18130431821905, 257.5686995555806, 261.07070353354993, 264.69478970158224, 268.44929079409445, 272.3435473154688, 276.3880896135361, 280.59486299764814, 284.9775086709067, 289.5517180159047, 294.3356847958481, 299.35069043485873, 304.62187400558975, 310.17926492998157, 316.059200210731, 322.3063237832385, 328.97650301847204, 336.14126110695065, 343.8948656757251, 352.36642480869347, 361.7423599546769, 372.31333661508177, 384.5907961800425, 399.6948959805394, 422.0030866468656]

    m = Mixture(['CO2', 'n-hexane'], zs=[.5, .5], T=300, P=1E6)
    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs,
                     Tcs=m.Tcs, Pcs=m.Pcs, omegas=m.omegas, kijs=[[0,0],[0,0]], eos_kwargs=None)

    bubs = []

    for P in Ps:
        bubs.append(pkg.bubble_T(P, m.zs, maxiter=20, xtol=1e-10, maxiter_initial=20, xtol_initial=1e-1)[-3])
    assert_allclose(bubs, T_bubbles_expect, rtol=5e-6)


@pytest.mark.deprecated
def test_PR_four_bubble_dew_cases():
    m = Mixture(['furfural', 'furfuryl alcohol'], zs=[.5, .5], T=300, P=1E6)
    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=[235.9, 250.35], Tbs=[434.65, 441.15],
                    Tcs=[670.0, 632.0], Pcs=[5510000.0, 5350000.0], omegas=[0.4522, 0.734],
                    kijs=[[0,0],[0,0]], eos_kwargs=None,
                 HeatCapacityGases=m.HeatCapacityGases)
    # Strongly believed to be correct!
    assert_allclose(pkg.bubble_T(P=1e6, zs=m.zs)[-3], 539.1838522423355, atol=.1)
    assert_allclose(pkg.dew_T(P=1e6, zs=m.zs)[-3], 540.208169750248, atol=.1)
    assert_allclose(pkg.dew_P(T=600, zs=m.zs)[-3], 2702616.6490743402, rtol=1e-4)
    assert_allclose(pkg.bubble_P(T=600, zs=m.zs)[-3], 2766476.7473238516, rtol=1e-4)


@pytest.mark.deprecated
def test_C1_C10_PT_flash():

    m = Mixture(['methane', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'], zs=[.1]*10, T=300, P=1E6)
    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs,
                     Tcs=m.Tcs, Pcs=m.Pcs, omegas=m.omegas, kijs=None, eos_kwargs=None)
    pkg.flash(m.zs, T=300, P=1e5)
    assert_allclose(pkg.V_over_F, 0.3933480636546702, atol=.001)

@pytest.mark.deprecated
def test_ternary_4_flashes_2_algorithms():
    zs = [0.8168, 0.1501, 0.0331]
    m = Mixture(['n-pentane', 'n-hexane', 'heptane'], zs=zs, T=300, P=1E6)
    kijs = [[0, 0.00076, 0.00171], [0.00076, 0, 0.00061], [0.00171, 0.00061, 0]]
    Tcs = [469.7, 507.6, 540.2]
    Pcs = [3370000.0, 3025000.0, 2740000.0]
    omegas = [0.251, 0.2975, 0.3457]
    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs,
                    Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                    kijs=kijs, eos_kwargs=None,
                 HeatCapacityGases=m.HeatCapacityGases)


    # Test the TVF dew and bubble functions
    Ts = linspace(160, 473) # Can go up to 474 K in some ones
    P_dews = []
    P_bubbles = []

    P_dews_expect = [0.13546805712060028, 0.43921188284030244, 1.2845937999086763, 3.4284388636658223, 8.432934762206317, 19.28325222813278, 41.304893136512625, 83.4340609062355, 159.8684378673581, 292.0910726014042, 511.22209088685435, 860.6178486961774, 1398.6103635474976, 2201.2649842930023, 3365.0325891826838, 5009.183244809382, 7277.928645001543, 10342.16726173515, 14400.815256785363, 19681.7145007659, 26442.1339934372, 34968.90099430364, 45578.21259245172, 58615.18736373379, 74453.22081253518, 93493.20841630214, 116162.69734460281, 142915.02335020038, 174228.483859415, 210605.5927100339, 252572.45694034494, 300678.3120515243, 355495.2497147813, 417618.1714535647, 487665.00399456796, 566277.2176110205, 654120.6992511221, 751887.0498581398, 860295.4040565268, 980094.9175170006, 1112068.1463153881, 1257035.6798685808, 1415862.6391328266, 1589468.1351378255, 1778839.7877655022, 1985057.6930205086, 2209338.09186709, 2453124.6502311486, 2718322.7103527053, 3008161.494037711]
    P_bubbles_expect = [1.6235349125052008, 4.093581157610554, 9.575333470191898, 20.9520396276609, 43.19443917687544, 84.41963574404814, 157.25506477949756, 280.5086157382652, 481.11896195432473, 796.3336480902728, 1276.039624284318, 1985.1548551180522, 3005.982245837768, 4440.428884348238, 6412.003702807469, 9067.523338810985, 12578.476805989876, 17142.0220330821, 22981.609086464254, 30347.244137353322, 39515.42378295007, 50788.78061416531, 64495.487917554194, 80988.47447946836, 100644.50024948097, 123863.1408495057, 151065.7243154966, 182694.25768005836, 219210.37457192744, 261094.32829701997, 308844.0480934111, 362974.2694768574, 424015.74270421825, 492514.5160789749, 569031.2825613177, 654140.7680276675, 748431.1260918825, 852503.2852406674, 966970.1650283068, 1092455.6319079874, 1229592.988381081, 1379022.655075611, 1541388.4595460077, 1717331.4675977635, 1907479.294701055, 2112426.5546414126, 2332696.247164788, 2568654.2410637783, 2820281.571897286, 3086319.669072729]

    for T in Ts[2:]:
        pkg.flash(T=T, VF=0, zs=zs)
        P_bubbles.append(pkg.P)
        pkg.flash(T=T, VF=1, zs=zs)
        P_dews.append(pkg.P)

    assert_allclose(P_bubbles, P_bubbles_expect[2:], rtol=5e-5)
    assert_allclose(P_dews, P_dews_expect[2:], rtol=5e-5)

    # For each point, solve it as a T problem.
    for P, T in zip(P_bubbles, Ts[2:]):
        pkg.flash(P=P, VF=0, zs=zs)
        assert_allclose(pkg.T, T, rtol=5e-5)
    for P, T in zip(P_dews, Ts[2:]):
        pkg.flash(P=P, VF=1, zs=zs)
        assert_allclose(pkg.T, T, rtol=5e-5)


    P_dews_almost = []
    P_bubbles_almost = []
    for T in Ts[4:]:
        # Some convergence issues in sequential_substitution_VL at lower pressures
        pkg.flash(T=T, VF=0+1e-9, zs=zs)
        P_bubbles_almost.append(pkg.P)
        pkg.flash(T=T, VF=1-1e-9, zs=zs)
        P_dews_almost.append(pkg.P)

    assert_allclose(P_bubbles[2:], P_bubbles_almost, rtol=5e-5)
    assert_allclose(P_dews[2:], P_dews_almost, rtol=5e-5)


    # Some points fail here too!
    for P, T in zip(P_dews_expect[4:-1], Ts[4:-1]):
        pkg.flash(P=P, VF=1-1e-9, zs=zs)
        assert_allclose(P, pkg.P)

    for P, T in zip(P_bubbles_expect[2:-2], Ts[2:-2]):
        pkg.flash(P=P, VF=0+1e-9, zs=zs)
        assert_allclose(P, pkg.P)


@pytest.mark.deprecated
@pytest.mark.slow
def test_PVF_parametric_binary_vs_CoolProp():
    import CoolProp.CoolProp as CP
    zs = [0.4, 0.6]
    m = Mixture(['Ethane', 'Heptane'], zs=zs, T=300, P=1E6)


    kij = .0067
    kijs = [[0,kij],[kij,0]]
    Tcs = [305.322, 540.13]
    Pcs = [4872200.0, 2736000.0]
    omegas = [0.099, 0.349]
    c1, c2 = PRMIX.c1, PRMIX.c2

    PRMIX.c1, PRMIX.c2 = 0.45724, 0.07780

    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs,
                    Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                    kijs=kijs, eos_kwargs=None,
                 HeatCapacityGases=m.HeatCapacityGases)
    pkg.FLASH_VF_TOL = 1e-12


    AS = CP.AbstractState("PR", "Ethane&Heptane")
    AS.set_mole_fractions(zs)
    AS.set_binary_interaction_double(0,1,"kij", kij)

    Ps = [10, 100, 1000, 1e4, 5e4, 1e5, 5e5, 1e6, 2e6]
    for P in Ps:
        # Up above 2e6, issues arise in thermo
        VFs = linspace(0, 1)
        CP_Ts = []
        Ts_calc = []
        for VF in VFs:
            try:
                AS.update(CP.PQ_INPUTS, P, VF);
                CP_Ts.append(AS.T())
                pkg.flash(VF=VF, P=P, zs=zs)
                Ts_calc.append(pkg.T)
            except Exception as e:
                print(VF, e)
    #     print(CP_Ts/np.array(Ts_calc))
        # the low pressure and highest pressure regions are the greatest errors
        # can go down to 1e-6 tol for all, most are 1e-12
        assert_allclose(CP_Ts, Ts_calc, rtol=1e-5)

    PRMIX.c1, PRMIX.c2 = c1, c2

@pytest.mark.deprecated
@pytest.mark.slow
def test_PVF_parametric_binary_zs_vs_CoolProp():
    '''More advanced test of the above. Changes mole fractions.
    To get more errors, reduce the mole fractions; and wide the P range.
    '''
    import CoolProp.CoolProp as CP

    zs = [0.4, 0.6]
    m = Mixture(['Ethane', 'Heptane'], zs=zs, T=300, P=1E6)

    kij = .0067
    kijs = [[0,kij],[kij,0]]
    Tcs = [305.322, 540.13]
    Pcs = [4872200.0, 2736000.0]
    omegas = [0.099, 0.349]
    c1, c2 = PRMIX.c1, PRMIX.c2
    PRMIX.c1, PRMIX.c2 = 0.45724, 0.07780

    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs,
                    Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                    kijs=kijs, eos_kwargs=None,
                 HeatCapacityGases=m.HeatCapacityGases)
    pkg.FLASH_VF_TOL = 1e-12


    AS = CP.AbstractState("PR", "Ethane&Heptane")
    AS.set_binary_interaction_double(0,1,"kij", kij)

    zis = linspace(.01, .98, 5)
    for zi in zis:
        zs = [1-zi, zi]
        Ps = [100, 1000, 1e4, 5e4, 1e5, 5e5, 1e6]
        for P in Ps:
            # Up above 2e6, issues arise in thermo
            VFs = linspace(0, 1)
            CP_Ts = []
            Ts_calc = []
            for VF in VFs:
                try:
                    AS.set_mole_fractions(zs)
                    AS.update(CP.PQ_INPUTS, P, VF);
                    CP_Ts.append(AS.T())
                    pkg.flash(VF=VF, P=P, zs=zs)
                    Ts_calc.append(pkg.T)
                except Exception as e:
                    print(zi, P, VF, e)
    #         try:
    #             print(CP_Ts/np.array(Ts_calc))
    #         except:
    #             print('bad shape')
            assert_allclose(CP_Ts, Ts_calc, rtol=1e-5)

    PRMIX.c1, PRMIX.c2 = c1, c2

@pytest.mark.deprecated
@pytest.mark.xfail
def test_failing_sequential_subs():
    zs = [0.8168, 0.1501, 0.0331]
    m = Mixture(['n-pentane', 'n-hexane', 'heptane'], zs=zs, T=300, P=1E6)
    kijs = [[0, 0.00076, 0.00171], [0.00076, 0, 0.00061], [0.00171, 0.00061, 0]]
    Tcs = [469.7, 507.6, 540.2]
    Pcs = [3370000.0, 3025000.0, 2740000.0]
    omegas = [0.251, 0.2975, 0.3457]
    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs,
                    Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                    kijs=kijs, eos_kwargs=None,
                 HeatCapacityGases=m.HeatCapacityGases)

    pkg.to_TP_zs(T=180, P=4, zs=zs).sequential_substitution_VL(maxiter=10,xtol=1E-7)




@pytest.mark.deprecated
def test_PRMIX_pkg_H():
    zs = [0.4, 0.6]
    m = Mixture(['Ethane', 'Heptane'], zs=zs, T=300, P=1E6)

    kij = .0
    kijs = [[0,kij],[kij,0]]
    Tcs = [305.322, 540.13]
    Pcs = [4872200.0, 2736000.0]
    omegas = [0.099, 0.349]

    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs,
                    Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                    kijs=kijs, eos_kwargs=None,
                 HeatCapacityGases=m.HeatCapacityGases)
    pkg.FLASH_VF_TOL = 1e-12

    # Case  gas -gas pressure difference
    pkg.flash(T=450, P=400, zs=m.zs)
    H1 = pkg.Hm
    assert pkg.phase == 'g'
    pkg.flash(T=450, P=1e6, zs=m.zs)
    H2 = pkg.Hm
    assert pkg.phase == 'g'
    assert_allclose(H1 - H2, 1638.19303081, rtol=1e-3)

    # Case gas to VF= = 0 at same T
    pkg.flash(T=350, P=400, zs=m.zs)
    assert pkg.phase == 'g'
    H1 = pkg.Hm
    pkg.flash(T=350, VF=.5, zs=m.zs)
    assert pkg.phase == 'l/g'
    H2 = pkg.Hm
    assert_allclose(H1 - H2, 16445.143155, rtol=1e-3)


    # Higher pressure, less matching (gas constant diff probably; gas-liquid difference! No partial phase.)
    pkg.flash(T=450, P=400, zs=m.zs)
    assert pkg.phase == 'g'
    H1 = pkg.Hm
    pkg.flash(T=450, P=1e8, zs=m.zs)
    assert pkg.phase == 'l'
    H2 = pkg.Hm
    H1 - H2
    assert_allclose(H1 - H2, 13815.6666172, rtol=1e-3)

    # low P fluid to saturation pressure (both gas)
    pkg.flash(T=450, P=400, zs=m.zs)
    assert pkg.phase == 'g'
    H1 = pkg.Hm
    H1 = pkg.Hm
    pkg.flash(T=450, VF=1, zs=m.zs)
    assert pkg.phase == 'g'
    H2 = pkg.Hm
    H2 = pkg.Hm
    assert_allclose(H1 - H2, 2003.84468984, rtol=1e-3)

    # low pressure gas to liquid saturated
    pkg.flash(T=350, P=400, zs=m.zs)
    assert pkg.phase == 'g'
    H1 = pkg.Hm
    pkg.flash(T=350, VF=0, zs=m.zs)
    assert pkg.phase == 'l'
    H2 = pkg.Hm
    assert_allclose(H1 - H2, 23682.3468207, rtol=1e-3)

    # High pressure liquid to partial evaporation
    pkg.flash(T=350, P=3e6, zs=m.zs)
    assert pkg.phase == 'l'
    H1 = pkg.Hm
    pkg.flash(T=350, VF=.25, zs=m.zs)
    assert pkg.phase == 'l/g'
    H2 = pkg.Hm
    assert_allclose(H1 - H2, -2328.21259061, rtol=1e-3)

    # High pressure temperature change
    pkg.flash(T=300, P=3e6, zs=m.zs)
    assert pkg.phase == 'l'
    H1 = pkg.Hm
    pkg.flash(T=400, P=1e7, zs=m.zs)
    assert pkg.phase == 'l'
    H2 = pkg.Hm
    assert_allclose(H1 - H2, -18470.2994798, rtol=1e-3)

    # High pressure temperature change and phase change
    pkg.flash(T=300, P=3e6, zs=m.zs)
    assert pkg.phase == 'l'
    H1 = pkg.Hm
    pkg.flash(T=400, P=1e5, zs=m.zs)
    assert pkg.phase == 'g'
    H2 = pkg.Hm
    H1 - H2
    assert_allclose(H1 - H2, -39430.7145672, rtol=1e-3)

@pytest.mark.deprecated
def test_PRMIX_pkg_S():
    zs = [0.4, 0.6]
    m = Mixture(['Ethane', 'Heptane'], zs=zs, T=300, P=1E6)

    kij = .0
    kijs = [[0,kij],[kij,0]]
    Tcs = [305.322, 540.13]
    Pcs = [4872200.0, 2736000.0]
    omegas = [0.099, 0.349]

    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs,
                    Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                    kijs=kijs, eos_kwargs=None,
                 HeatCapacityGases=m.HeatCapacityGases)
    pkg.FLASH_VF_TOL = 1e-12


    # Case  gas -gas pressure difference
    pkg.flash(T=450, P=400, zs=m.zs)
    S1 = pkg.Sm
    assert pkg.phase == 'g'
    pkg.flash(T=450, P=1e6, zs=m.zs)
    S2 = pkg.Sm
    assert pkg.phase == 'g'
    assert_allclose(S1 - S2, 67.59095157604824, rtol=1e-3)

    # Case gas to VF= = 0 at same T
    pkg.flash(T=350, P=400, zs=m.zs)
    assert pkg.phase == 'g'
    S1 = pkg.Sm
    pkg.flash(T=350, VF=.5, zs=m.zs)
    assert pkg.phase == 'l/g'
    S2 = pkg.Sm
    assert_allclose(S1 - S2, 96.84959621651315, rtol=1e-3)


    # Higher pressure, less matching (gas constant diff probably; gas-liquid difference! No partial phase.)
    pkg.flash(T=450, P=400, zs=m.zs)
    assert pkg.phase == 'g'
    S1 = pkg.Sm
    pkg.flash(T=450, P=1e8, zs=m.zs)
    assert pkg.phase == 'l'
    S2 = pkg.Sm
    S1 - S2
    assert_allclose(S1 - S2, 128.67194096593366, rtol=1e-3)

    # low P fluid to saturation pressure (both gas)
    pkg.flash(T=450, P=400, zs=m.zs)
    assert pkg.phase == 'g'
    H1 = pkg.Hm
    S1 = pkg.Sm
    pkg.flash(T=450, VF=1, zs=m.zs)
    assert pkg.phase == 'g'
    H2 = pkg.Hm
    S2 = pkg.Sm
    assert_allclose(S1 - S2, 69.64345358808025, rtol=1e-3)

    # low pressure gas to liquid saturated
    pkg.flash(T=350, P=400, zs=m.zs)
    assert pkg.phase == 'g'
    S1 = pkg.Sm
    pkg.flash(T=350, VF=0, zs=m.zs)
    assert pkg.phase == 'l'
    S2 = pkg.Sm
    assert_allclose(S1 - S2, 124.44419797042649, rtol=1e-3)

    # High pressure liquid to partial evaporation
    pkg.flash(T=350, P=3e6, zs=m.zs)
    assert pkg.phase == 'l'
    S1 = pkg.Sm
    pkg.flash(T=350, VF=.25, zs=m.zs)
    assert pkg.phase == 'l/g'
    S2 = pkg.Sm
    assert_allclose(S1 - S2, -7.913399921816193, rtol=1e-3)

    # High pressure temperature change
    pkg.flash(T=300, P=3e6, zs=m.zs)
    assert pkg.phase == 'l'
    S1 = pkg.Sm
    pkg.flash(T=400, P=1e7, zs=m.zs)
    assert pkg.phase == 'l'
    S2 = pkg.Sm
    assert_allclose(S1 - S2, -50.38050604000216, atol=1)

    # High pressure temperature change and phase change
    pkg.flash(T=300, P=3e6, zs=m.zs)
    assert pkg.phase == 'l'
    S1 = pkg.Sm
    pkg.flash(T=400, P=1e5, zs=m.zs)
    assert pkg.phase == 'g'
    S2 = pkg.Sm
    S1 - S2
    assert_allclose(S1 - S2, -124.39457107124854, atol=1)

@pytest.mark.deprecated
def test_PRMIX_pkg_extras():
    # TODO add more properties as they are added
    zs = [0.4, 0.6]
    m = Mixture(['Ethane', 'Heptane'], zs=zs, T=300, P=1E6)

    kij = .0
    kijs = [[0,kij],[kij,0]]
    Tcs = [305.322, 540.13]
    Pcs = [4872200.0, 2736000.0]
    omegas = [0.099, 0.349]

    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs,
                    Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                    kijs=kijs, eos_kwargs=None,
                 HeatCapacityGases=m.HeatCapacityGases)


    pkg.flash(T=400, P=1e5, zs=m.zs)
    assert 'g' == pkg.phase

    assert_allclose(pkg.eos_g.H_dep_g, -179.77096245871508, rtol=1e-5)
    assert_allclose(pkg.eos_g.S_dep_g, -0.2971318950892263, rtol=1e-5)
    assert_allclose(pkg.Hgm_dep, -179.77096245871508, rtol=5e-5)
    assert_allclose(pkg.Sgm_dep, -0.2971318950892263, rtol=5e-5)

    assert_allclose(pkg.Cpgm, 153.32126587681677, rtol=1e-3)
    assert_allclose(pkg.Cvgm, 144.3920626710827, rtol=1e-3) # :)
    assert_allclose(pkg.Cpgm_dep, 0.7139646058820279, rtol=1e-5)
    assert_allclose(pkg.Cvgm_dep, 0.09922120014794993, rtol=1e-5) #? maybe issue

    pkg.flash(T=300, P=1e7, zs=m.zs)
    assert 'l' == pkg.phase

    assert_allclose(pkg.eos_l.H_dep_l, -25490.54123032457, rtol=5e-5)
    assert_allclose(pkg.eos_l.S_dep_l, -48.47646403887194, rtol=5e-5)
    assert_allclose(pkg.Hlm_dep, -25490.54123, rtol=1e-4)
    assert_allclose(pkg.Slm_dep, -48.47646403887194, rtol=1e-4)

    assert_allclose(pkg.Cplm, 160.5756363050434, rtol=1e-3)
    assert_allclose(pkg.Cvlm, 133.7943922248561, rtol=1e-3) # :)
    assert_allclose(pkg.Cplm_dep, 39.8813153015303, rtol=5e-5)
    assert_allclose(pkg.Cvlm_dep, 21.414531021342995, rtol=5e-5) #? maybe issue


@pytest.mark.deprecated
def test_azeotrope_Txy_PR():
    IDs = ['ethanol', 'benzene']
    pkg = PropertyPackageConstants(IDs, name=PR_PKG)
    pkg.pkg.kijs = [[0.0, .0728], [0.0728, 0]]

    # Test the pressure in the test

    _, _, Tbubbles, Tdews = pkg.pkg.plot_Txy(P=101325., pts=30, values=True)

    Tbubbles_expect = [353.1524424999673, 351.21711105215405, 349.63220641849136, 348.3290291072549, 347.2552443556649, 346.37022614955663, 345.6419123814478, 345.0446351984003, 344.55759626315887, 344.16377920005266, 343.84916614883053, 343.60217197943285, 343.41323969870245, 343.2745605540422, 343.1798963139651, 343.12449170081203, 343.1050736632354, 343.1199423771055, 343.169167658216, 343.2549149821879, 343.38193882073034, 343.5582990521058, 343.7963805186986, 344.1143278723936, 344.53804741377195, 345.1039685436253, 345.8627772097754, 346.88426937346605, 348.26317130456636, 350.12491594342015]
    Tdews_expect = [353.1524424945457, 352.3912913474467, 351.6262944570331, 350.8588218276585, 350.0906535909099, 349.32409993796796, 348.56216098024134, 347.8087416697709, 347.0689431804551, 346.349459873305, 345.6591224986107, 345.00963438553083, 344.4165436003679, 343.90042076441017, 343.4879384830795, 343.21166686886806, 343.10538604291753, 343.1904450269102, 343.4583142995908, 343.8715382698287, 344.38531268086734, 344.96341038590646, 345.5807576414249, 346.22080282099756, 346.8726671468842, 347.52913516661, 348.18536889289476, 348.83809921197854, 349.4851121234294, 350.1249159362295]
    assert_allclose(Tbubbles, Tbubbles_expect, rtol=5e-5)
    assert_allclose(Tdews, Tdews_expect, rtol=5e-5)

@pytest.mark.deprecated
def test_azeotrope_Txy_PR_multiP():
    IDs = ['ethanol', 'benzene']
    pkg = PropertyPackageConstants(IDs, name=PR_PKG)
    pkg.pkg.kijs = [[0.0, .0728], [0.0728, 0]]
    #Test some more pressures for good measure (do not go too far near the boundaries)
    Tdews_vary_P_expect = [[220.15284322119734, 219.96736090890047, 222.4531025319982, 225.87591713961928, 228.38731541934482, 230.38394741856035, 232.04763019651986, 233.47741573028978, 234.73343380218137, 235.85502051831918, 236.8693632699694, 237.79606282049812, 238.6497311937851, 239.441561771029, 240.18032475929877], [250.29484272442642, 249.8077093695365, 249.42498602603337, 249.28679137901344, 251.96383913156598, 254.37995372490553, 256.413822517376, 258.1732216294687, 259.72617151880036, 261.1180840150342, 262.38075082537034, 263.53729905772974, 264.6050861496727, 265.5974792849115, 266.5249972102388], [291.6640151659878, 290.7705630707953, 289.9138195863271, 289.1364722089608, 288.52440010361823, 288.2911650820978, 289.2699169291151, 291.1074635611929, 292.88383384665804, 294.5345675748379, 296.06323507541447, 297.48280466114016, 298.8066120886574, 300.0464992568524, 301.2125736864664], [352.7187334210476, 351.14318764286776, 349.5582311684951, 347.9830060760723, 346.4505096140636, 345.01844631869784, 343.78971773705734, 342.946979856768, 342.761997697104, 343.3571699367641, 344.44973552643745, 345.7436125329223, 347.09887074851576, 348.456160479165, 349.78950974944104], [452.0244773102955, 448.93347954186527, 445.80843092367013, 442.6587093618919, 439.501298156353, 436.36519982883647, 433.2983005305142, 430.3773519002321, 427.7197877701338, 425.488417876116, 423.8636548918616, 422.9595322281223, 422.7424527930051, 423.0631573964071, 423.755679832123]]
    Tbubbles_vary_P_expect = [[220.15284322260558, 219.9593754659149, 219.97616818101181, 220.06166994291502, 220.11857644484724, 220.07120074079083, 219.85507103807385, 219.41277280716295, 218.69492646979015, 217.668864547681, 216.34458685271593, 214.85030366123252, 213.69173078234607, 215.05360231675624, 240.18032476043962], [250.29484272636603, 249.6034873812954, 249.3302912913737, 249.28015720321142, 249.32976951168072, 249.39258345431227, 249.40319499276376, 249.31112373153306, 249.08166823900348, 248.70647278451116, 248.234802114006, 247.86101246500266, 248.19942715491368, 251.37880207972458, 266.5249972119777], [291.6640151695664, 289.98349864091705, 289.03830617940577, 288.5439451223825, 288.32964841249037, 288.2845537334007, 288.33357406890497, 288.4270830783461, 288.5394805139744, 288.6772197151695, 288.90297364622535, 289.39555977182874, 290.6007278361622, 293.62911489553994, 301.2125736895028], [352.71873342667294, 349.111737745725, 346.7112146623439, 345.1062784039534, 344.0464541688307, 343.37007653346484, 342.96929893428904, 342.772655513814, 342.7375808864912, 342.85043503480443, 343.1348957652109, 343.67328993658566, 344.6514600873893, 346.44748552216527, 349.78950975665305], [452.0244773382353, 444.9894726934088, 439.5814848561124, 435.3968758569498, 432.1385100677904, 429.5895584761499, 427.5915768205421, 426.02882099085946, 424.8179436451578, 423.90205997868895, 423.2487129719721, 422.85201924536585, 422.7401871273592, 422.99064457143066, 423.7556798321766]]
    Tdews_vary_P = []
    Tbubbles_vary_P = []

    # pkg.pkg.plot_Txy(P=100, pts=100) # values=True
    for P in logspace(2, 6, 5):
        _, _, Tbubbles, Tdews = pkg.pkg.plot_Txy(P=P, pts=15, values=True)
        Tbubbles_vary_P.append(Tbubbles)
        Tdews_vary_P.append(Tdews)
    assert_allclose(Tbubbles_vary_P, Tbubbles_vary_P_expect, rtol=1e-5)
    assert_allclose(Tdews_vary_P, Tdews_vary_P_expect, rtol=1e-5)

@pytest.mark.deprecated
def test_azeotrope_Pxy_PR_multiT():
    IDs = ['ethanol', 'benzene']
    pkg = PropertyPackageConstants(IDs, name=PR_PKG)
    pkg.pkg.kijs = [[0.0, .0728], [0.0728, 0]]

    Ts = [220, 250, 300, 350, 400, 450, 475, 450, 500, 505, 507.5]
    Ps_bubble_multi_T_expect = [[2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672], [2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672], [2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672], [2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672], [2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672], [2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672], [2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672], [2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672], [2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672], [2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672], [2413788.5246687443, 2641798.365260112, 2861327.402278104, 3072725.7261532187, 3276386.430826817, 3472735.47268973, 3662217.055850461, 3845273.4025023617, 4022317.494062474, 4193697.0600327696, 4359647.812662887, 4520233.860544795, 4675273.823398787, 4824253.350470867, 4966230.2319129715, 5099749.351949349, 5222801.450719186, 5332871.261916157, 5427095.16804713, 5502455.831709672]]
    Ps_dew_multi_T_expect = [[2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671], [2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671], [2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671], [2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671], [2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671], [2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671], [2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671], [2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671], [2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671], [2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671], [2413788.5246687448, 2528934.820505938, 2653201.630551055, 2787361.571970873, 2932152.5813825456, 3088204.613722783, 3255927.030569813, 3435346.2652946324, 3625890.6578108566, 3826139.6891573607, 4033592.988357248, 4244562.319550634, 4454311.19552944, 4657505.6804501135, 4848882.834798467, 5023886.247109449, 5179004.275036525, 5311703.561411566, 5420071.437922198, 5502455.831709671]]
    Ps_bubble_multi_T, Ps_dew_multi_T = [], []
    for T in Ts:
        _, _, Ps_bubble, Ps_dew = pkg.pkg.plot_Pxy(T=507.5, pts=20, ignore_errors=True, values=True)
        Ps_bubble_multi_T.append(Ps_bubble)
        Ps_dew_multi_T.append(Ps_dew)
    assert_allclose(Ps_bubble_multi_T_expect, Ps_bubble_multi_T, rtol=1e-6)
    assert_allclose(Ps_dew_multi_T_expect, Ps_dew_multi_T, rtol=1e-6)


@pytest.mark.deprecated
def test_phase_envelope_ethane_pentane():

    IDs = ['ethane', 'n-pentane']
    pkg = PropertyPackageConstants(IDs, PR_PKG, kijs=[[0, 7.609447e-003], [7.609447e-003, 0]])
    zs = [0.7058334393128614, 0.2941665606871387] # 50/50 mass basis

    max_step_damping = 100
    P_high = 8e6
    factor = 1.2
    min_step_termination = 1000
    min_factor_termination = 1.001
    pkg.pkg.FLASH_VF_TOL = 1e-8
    max_P_step = 1e5

    P_low = 1e5
    spec_points = linspace(1e5, 6.8e6, 68)

    P_points, Ts_known, xs_known = pkg.pkg.dew_T_envelope(zs, P_low=P_low, P_high=P_high, xtol=1E-10,
                           factor=factor, min_step_termination=min_step_termination,
                          min_factor_termination=min_factor_termination,
                          max_step_damping=max_step_damping,
                          max_P_step=max_P_step,
                          spec_points=spec_points)

    P_points2, Ts_known2, ys_known = pkg.pkg.bubble_T_envelope(zs, P_low=P_low, P_high=P_high, xtol=1E-10,
                           factor=factor, min_step_termination=min_step_termination,
                           max_step_damping=max_step_damping,
                           min_factor_termination=min_factor_termination,
                           max_P_step=max_P_step, spec_points=spec_points)
    Ps_dew_check = []
    Ts_dew_check = []
    Ts_dew_expect = [277.1449361694948, 293.9890986702753, 304.8763147090649, 313.1006603531763, 319.7750626828419, 325.42150966613895, 330.32990856864086, 334.6791912532372, 338.58812791519466, 342.13987634031974, 345.3950895854326, 348.39946023112896, 351.1883247302556, 353.7896091573966, 356.22578835719867, 358.51523195418594, 360.673155009561, 362.71230559820697, 364.64347249145516, 366.47586686424677, 368.21741391309746, 369.8749788315006, 371.4545441481416, 372.96135047685806, 374.40000935978657, 375.774594553273, 377.0887164639959, 378.3455832681606, 379.54805139424366, 380.698667422777, 381.7997029894565, 382.8531839253168, 383.8609145986068, 384.824498212078, 385.74535365141924, 386.6247293397855, 387.4637144549204, 388.2632477701861, 389.02412430395054, 389.7469998909282, 390.4323937166012, 391.08068879770735, 391.69213029960156, 392.2668215113749, 392.8047171889375, 393.30561384008115, 393.76913637985547, 394.19472032380185, 394.58158839582626, 394.9287200011976, 395.2348113354687, 395.4982230372662, 395.71691081859336, 395.8883324260842, 396.0093207511565, 396.0759073750358, 396.0830711573792, 396.024369487178, 395.8913790901176, 395.67280294095485, 395.3529926936849, 394.9092730479461, 394.3067055020046, 393.48636807223045, 392.33342385249546, 390.55261457054587]
    for P_dew, T_dew in zip(P_points, Ts_known):
        if abs(P_dew % 1e5) < 1e-5:
            Ps_dew_check.append(P_dew)
            Ts_dew_check.append(T_dew)

    Ps_bubble_check = []
    Ts_bubble_check = []
    Ts_bubble_expect = [277.1449361694948, 293.9890986702753, 304.8763147090649, 313.1006603531763, 319.7750626828419, 325.42150966613895, 330.32990856864086, 334.6791912532372, 338.58812791519466, 342.13987634031974, 345.3950895854326, 348.39946023112896, 351.1883247302556, 353.7896091573966, 356.22578835719867, 358.51523195418594, 360.673155009561, 362.71230559820697, 364.64347249145516, 366.47586686424677, 368.21741391309746, 369.8749788315006, 371.4545441481416, 372.96135047685806, 374.40000935978657, 375.774594553273, 377.0887164639959, 378.3455832681606, 379.54805139424366, 380.698667422777, 381.7997029894565, 382.8531839253168, 383.8609145986068, 384.824498212078, 385.74535365141924, 386.6247293397855, 387.4637144549204, 388.2632477701861, 389.02412430395054, 389.7469998909282, 390.4323937166012, 391.08068879770735, 391.69213029960156, 392.2668215113749, 392.8047171889375, 393.30561384008115, 393.76913637985547, 394.19472032380185, 394.58158839582626, 394.9287200011976, 395.2348113354687, 395.4982230372662, 395.71691081859336, 395.8883324260842, 396.0093207511565, 396.0759073750358, 396.0830711573792, 396.024369487178, 395.8913790901176, 395.67280294095485, 395.3529926936849, 394.9092730479461, 394.3067055020046, 393.48636807223045, 392.33342385249546, 390.55261457054587]
    for P_bubble, T_bubble in zip(P_points, Ts_known):
        if abs(P_bubble % 1e5) < 1e-5:
            Ps_bubble_check.append(P_bubble)
            Ts_bubble_check.append(T_bubble)

    assert_allclose(Ps_bubble_check, spec_points[:-2])
    assert_allclose(Ps_dew_check, spec_points[:-2])
    assert_allclose(Ts_dew_check, Ts_dew_expect, rtol=1e-5)
    assert_allclose(Ts_bubble_check, Ts_bubble_expect, rtol=1e-5)

@pytest.mark.deprecated
def test_ethane_pentane_TP_Tdew_Tbubble_TP():
    # Takes 9 seconds!
    IDs = ['ethane', 'n-pentane']
    pkg = PropertyPackageConstants(IDs, PR_PKG, kijs=[[0, 7.609447e-003], [7.609447e-003, 0]])
    zs = [0.7058334393128614, 0.2941665606871387] # 50/50 mass basis
    pkg = pkg.pkg

    VFs = []
    all_Ts = []
    all_Ps = []
    P_high = 6.1e6 # goal: 6e6 It worked!
    P_low = 1e3
    Ps = logspace(log10(P_low), log10(P_high), 50)
    T_lows = []
    T_highs = []

    for P in Ps:
        pkg.flash(P=P, VF=0, zs=zs)
        T_low = pkg.T # 129 K
        T_lows.append(T_low)
        pkg.flash(P=P, VF=1, zs=zs)
        T_high = pkg.T # 203 K
        T_highs.append(T_high)

        for Wilson_first in (False, True):
            VFs_working = []
            Ts = linspace(T_low+1e-4, T_high-1e-4, 50)
            for T in Ts:
                ans = pkg.flash_TP_zs(P=P, T=T, zs=zs, Wilson_first=Wilson_first)
                VFs_working.append(ans[-1])
                if ans[0] != 'l/g':
                    raise ValueError("Converged to single phase solution at T=%g K, P=%g Pa" %(T, P))

        VFs.append(VFs_working)
        all_Ts.append(Ts)
        all_Ps.append(Ps)

@pytest.mark.deprecated
@pytest.mark.slow_envelope
def test_phase_envelope_44_components():
    IDs = ['nitrogen', 'carbon dioxide', 'H2S', 'methane', 'ethane', 'propane', 'isobutane', 'butane', 'isopentane', 'pentane', 'Hexane', 'Heptane', 'Octane', 'Nonane', 'Decane', 'Undecane', 'Dodecane', 'Tridecane', 'Tetradecane', 'Pentadecane', 'Hexadecane', 'Heptadecane', 'Octadecane', 'Nonadecane', 'Eicosane', 'Heneicosane', 'Docosane', 'Tricosane', 'Tetracosane', 'Pentacosane', 'Hexacosane', 'Heptacosane', 'Octacosane', 'Nonacosane', 'Triacontane', 'Benzene', 'Toluene', 'Ethylbenzene', 'Xylene', '1,2,4-Trimethylbenzene', 'Cyclopentane', 'Methylcyclopentane', 'Cyclohexane', 'Methylcyclohexane']
    zs = [9.11975115499676e-05, 9.986813065240533e-05, 0.0010137795304828892, 0.019875879000370657, 0.013528874875432457, 0.021392773691700402, 0.00845450438914824, 0.02500218071904368, 0.016114189201071587, 0.027825798446635016, 0.05583179467176313, 0.0703116540769539, 0.07830577180555454, 0.07236459223729574, 0.0774523322851419, 0.057755091407705975, 0.04030134965162674, 0.03967043780553758, 0.03514481759005302, 0.03175471055284055, 0.025411123554079325, 0.029291866298718154, 0.012084986551713202, 0.01641114551124426, 0.01572454598093482, 0.012145363820829673, 0.01103585282423499, 0.010654818322680342, 0.008777712911254239, 0.008732073853067238, 0.007445155260036595, 0.006402875549212365, 0.0052908087849774296, 0.0048199150683177075, 0.015943943854195963, 0.004452253754752775, 0.01711981267072777, 0.0024032720444511282, 0.032178399403544646, 0.0018219517069058137, 0.003403378548794345, 0.01127516775495176, 0.015133143423489698, 0.029483213283483682]

    pkg = PropertyPackageConstants(IDs, PR_PKG)

    max_step_damping = 50
    P_low = 1e4
    factor = 1.2
    min_step_termination = 1000
    min_factor_termination = 1.0002
    pkg.pkg.FLASH_VF_TOL = 1e-8
    P_high = 2e8
    spec_points = linspace(1e5, 4e6, 40)

    P_points, Ts_known, xs_known = pkg.pkg.dew_T_envelope(zs, P_low=P_low, P_high=P_high, xtol=1E-10,
                           factor=factor, min_step_termination=min_step_termination,
                                                          min_factor_termination=min_factor_termination,
                                                          max_step_damping=max_step_damping,
                                                          spec_points=spec_points
                                                         )

    P_points2, Ts_known2, ys_known = pkg.pkg.bubble_T_envelope(zs, P_low=P_low, P_high=P_high, xtol=1E-10,
                           factor=factor, min_step_termination=min_step_termination,
                                                               max_step_damping=max_step_damping,
                                                               min_factor_termination=min_factor_termination,
                                                               spec_points=spec_points
                                                              )
    Ps_dew_check = []
    Ts_dew_check = []
    Ts_dew_expect = [585.1745093521665, 609.5133715138915, 624.6944734390993, 635.7991119723131, 644.5334850169733, 651.6941060581852, 657.7213913216676, 662.8858558611348, 667.3660286752593, 671.2860034847065, 674.7354375617153, 677.7810270676093, 680.4734809440047, 682.8519536806468, 684.9469622199979, 686.7823540873131, 688.3766543470003, 689.7439863506575, 690.8946833742955, 691.8356590318011, 692.5705695910872, 693.0997717010517, 693.4200465117376, 693.5240144469666, 693.399082494406, 693.0255964253895, 692.3734715991103, 691.3954910689196, 690.0119359589117, 688.0668235519908, 685.1543692400655, 679.0864243340858]
    for P_dew, T_dew in zip(P_points, Ts_known):
        if abs(P_dew % 1e5) < 1e-5:
            Ps_dew_check.append(P_dew)
            Ts_dew_check.append(T_dew)
            Ps_bubble_check = []

    Ts_bubble_check = []
    Ts_bubble_expect = [585.1745093521665, 609.5133715138915, 624.6944734390993, 635.7991119723131, 644.5334850169733, 651.6941060581852, 657.7213913216676, 662.8858558611348, 667.3660286752593, 671.2860034847065, 674.7354375617153, 677.7810270676093, 680.4734809440047, 682.8519536806468, 684.9469622199979, 686.7823540873131, 688.3766543470003, 689.7439863506575, 690.8946833742955, 691.8356590318011, 692.5705695910872, 693.0997717010517, 693.4200465117376, 693.5240144469666, 693.399082494406, 693.0255964253895, 692.3734715991103, 691.3954910689196, 690.0119359589117, 688.0668235519908, 685.1543692400655, 679.0864243340858]
    for P_bubble, T_bubble in zip(P_points, Ts_known):
        if abs(P_bubble % 1e5) < 1e-5:
            Ps_bubble_check.append(P_bubble)
            Ts_bubble_check.append(T_bubble)

    assert_allclose(Ps_bubble_check, spec_points[:-8])
    assert_allclose(Ps_dew_check, spec_points[:-8])
    assert_allclose(Ts_dew_check, Ts_dew_expect, rtol=1e-5)
    assert_allclose(Ts_bubble_check, Ts_bubble_expect, rtol=1e-5)

@pytest.mark.deprecated
def test_TPD_bubble_dew():
    IDs = ['ethane', 'n-pentane']
    pkg = PropertyPackageConstants(IDs, PR_PKG, kijs=[[0, 7.609447e-003], [7.609447e-003, 0]])
    zs = [0.7058334393128614, 0.2941665606871387] # 50/50 mass basis
    pkg = pkg.pkg


    pkg.flash(P=1e6, VF=0, zs=zs)
    pkg.eos_l.fugacities()
    pkg.eos_g.fugacities()
    TPD_calc = TPD(pkg.eos_g.T, pkg.eos_l.zs, pkg.eos_l.lnphis_l, pkg.eos_g.zs, pkg.eos_g.lnphis_g,)
    assert_allclose(TPD_calc, 0, atol=1e-6)

    pkg.flash(T=200, VF=0, zs=zs)
    pkg.eos_l.fugacities()
    pkg.eos_g.fugacities()
    TPD_calc = TPD(pkg.eos_g.T, pkg.eos_l.zs, pkg.eos_l.lnphis_l, pkg.eos_g.zs, pkg.eos_g.lnphis_g,)
    assert_allclose(TPD_calc, 0, atol=1e-6)

    pkg.flash(P=1e6, VF=1, zs=zs)
    pkg.eos_l.fugacities()
    pkg.eos_g.fugacities()
    TPD_calc = TPD(pkg.eos_g.T, pkg.eos_g.zs, pkg.eos_g.lnphis_g, pkg.eos_l.zs, pkg.eos_l.lnphis_l)
    assert_allclose(TPD_calc, 0, atol=1e-6)

    pkg.flash(T=300, VF=1, zs=zs)
    pkg.eos_l.fugacities()
    pkg.eos_g.fugacities()
    TPD_calc = TPD(pkg.eos_g.T, pkg.eos_g.zs, pkg.eos_g.lnphis_g, pkg.eos_l.zs, pkg.eos_l.lnphis_l)
    assert_allclose(TPD_calc, 0, atol=1e-6)


@pytest.mark.deprecated
def test_stab_comb_products_need_both_roots():
    comb_IDs = ['N2', 'CO2', 'O2', 'H2O']
    comb_zs = [0.5939849621247668,
      0.112781954982051,
      0.0676691730155464,
      0.2255639098776358]

    pkg2 = PropertyPackageConstants(comb_IDs, PR_PKG)
    kijs = [[0.0, -0.0122, -0.0159, 0.0], [-0.0122, 0.0, 0.0, 0.0952], [-0.0159, 0.0, 0.0, 0.0], [0.0, 0.0952, 0.0, 0.0]]
    pkg2 = PropertyPackageConstants(comb_IDs, PR_PKG, kijs=kijs)

    pkg2.pkg.flash_caloric(P=1e5,T=794.5305048838037, zs=comb_zs)
    assert 'g' == pkg2.pkg.phase
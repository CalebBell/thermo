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
from thermo.utils import normalize
from thermo.eos import *
from thermo.eos_mix import *
from scipy.misc import derivative
from scipy.optimize import minimize, newton
from math import log, exp, sqrt
from thermo import Mixture
from thermo.property_package import *
from fluids.numerics import linspace, logspace
from thermo.property_package_constants import (PropertyPackageConstants, PR_PKG)


def test_bubble_T_PR():
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
        bubs.append(pkg.bubble_T(P, m.zs, maxiter=20, xtol=1e-10, maxiter_initial=20, xtol_initial=1e-1)[-1])
    assert_allclose(bubs, T_bubbles_expect, rtol=5e-6)


def test_PR_four_bubble_dew_cases():
    m = Mixture(['furfural', 'furfuryl alcohol'], zs=[.5, .5], T=300, P=1E6)
    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=[235.9, 250.35], Tbs=[434.65, 441.15], 
                    Tcs=[670.0, 632.0], Pcs=[5510000.0, 5350000.0], omegas=[0.4522, 0.734], 
                    kijs=[[0,0],[0,0]], eos_kwargs=None,
                 HeatCapacityGases=m.HeatCapacityGases)
    # Strongly believed to be correct!
    assert_allclose(pkg.bubble_T(P=1e6, zs=m.zs)[-1], 539.1838522423355, atol=.1)
    assert_allclose(pkg.dew_T(P=1e6, zs=m.zs)[-1], 540.208169750248, atol=.1)
    assert_allclose(pkg.dew_P(T=600, zs=m.zs)[-1], 2702616.6490743402, rtol=1e-4)
    assert_allclose(pkg.bubble_P(T=600, zs=m.zs)[-1], 2766476.7473238516, rtol=1e-4)


def test_C1_C10_PT_flash():

    m = Mixture(['methane', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'], zs=[.1]*10, T=300, P=1E6)
    pkg = GceosBase(eos_mix=PRMIX, VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs, 
                     Tcs=m.Tcs, Pcs=m.Pcs, omegas=m.omegas, kijs=None, eos_kwargs=None)
    pkg.flash(m.zs, T=300, P=1e5)
    assert_allclose(pkg.V_over_F, 0.3933480636546702, atol=.001)

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
    
    P_dews_expect = [0.13550997960772718, 0.43920927301936163, 1.2845936810152798, 3.4284382400291693, 8.4329335064149, 19.283250088575713, 41.30489024588801, 83.43405803268355, 159.8684362297488, 292.0905623857701, 511.2220883589547, 860.617828280894, 1398.61028606488, 2201.2647663425837, 3365.0320695656337, 5009.182128235335, 7277.9264056343845, 10342.162976562995, 14400.807318841986, 19681.70013049728, 26442.108416294817, 34968.90098754035, 45578.21257737363, 58615.187330537265, 74453.22074032361, 93493.20826111095, 116162.69701520355, 142915.0226599925, 174228.48243230282, 210605.5897992026, 252572.4510846651, 300678.3004334014, 355495.226975388, 417618.1275304346, 487664.92020677816, 566277.0596055393, 654120.6993504411, 751887.0501098596, 860295.404689946, 980094.9191067469, 1112068.1503187479, 1257035.6900635846, 1415862.6656668677, 1589468.206763977, 1778839.7877382424, 1985057.692849609, 2209338.0904894564, 2453124.633112388, 2718322.710220342, 3008161.4956615395]
    P_bubbles_expect = [1.6235301866702332, 4.093574584709757, 9.575342178574326, 20.952039209715185, 43.194436834552775, 84.41963115012123, 157.25505678687318, 280.5086025113121, 481.11894146295754, 796.3336201619444, 1276.0395899912198, 1985.1548171878508, 3005.9822090277526, 4440.4288543813955, 6412.003684382402, 9067.523332578297, 12578.477383099947, 17142.02331043844, 22981.609135307823, 30347.244491015474, 39515.42532006434, 50788.78573841474, 64495.48791928443, 80988.47448637952, 100644.50027326142, 123863.14092224094, 151065.72451755867, 182694.2581981442, 219210.3758133033, 261094.3311046259, 308844.0541368579, 362974.28194353415, 424015.7674940148, 492514.5160883044, 569031.2825847857, 654140.7680846398, 748431.126226, 852503.2855480977, 966970.1657168784, 1092455.633419491, 1229592.9916416765, 1379022.6620045549, 1541388.4740836907, 1717331.4977741877, 1907479.3567695513, 2112426.5546544623, 2332696.2472028914, 2568654.2411678717, 2820281.572130601, 3086319.6695971168]
    
    for T in Ts:
        pkg.flash(T=T, VF=0, zs=zs)
        P_bubbles.append(pkg.P)
        pkg.flash(T=T, VF=1, zs=zs)
        P_dews.append(pkg.P)
        
    assert_allclose(P_bubbles, P_bubbles_expect, rtol=5e-5)
    assert_allclose(P_dews, P_dews_expect, rtol=5e-5)
    
    # For each point, solve it as a T problem.
    for P, T in zip(P_bubbles, Ts):
        pkg.flash(P=P, VF=0, zs=zs)
        assert_allclose(pkg.T, T, rtol=5e-5)
    for P, T in zip(P_dews[1:], Ts[1:]):
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
    
    assert_allclose(P_bubbles[4:], P_bubbles_almost, rtol=5e-5)
    assert_allclose(P_dews[4:], P_dews_almost, rtol=5e-5)
    
    
    # Some points fail here too!
    for P, T in zip(P_dews_expect[4:-1], Ts[4:-1]):
        pkg.flash(P=P, VF=1-1e-9, zs=zs)
        assert_allclose(P, pkg.P)
        
    for P, T in zip(P_bubbles_expect[2:-2], Ts[2:-2]):
        pkg.flash(P=P, VF=0+1e-9, zs=zs)
        assert_allclose(P, pkg.P)
    

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
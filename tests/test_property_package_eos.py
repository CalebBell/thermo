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
from fluids.numerics import linspace


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
        
    assert_allclose(P_bubbles, P_bubbles_expect, rtol=5e-6)
    assert_allclose(P_dews, P_dews_expect, rtol=5e-6)
    
    # For each point, solve it as a T problem.
    for P, T in zip(P_bubbles, Ts):
        pkg.flash(P=P, VF=0, zs=zs)
        assert_allclose(pkg.T, T, rtol=5e-6)
    for P, T in zip(P_dews[1:], Ts[1:]):
        pkg.flash(P=P, VF=1, zs=zs)
        assert_allclose(pkg.T, T, rtol=5e-6)
        
        
    P_dews_almost = []
    P_bubbles_almost = []
    for T in Ts[4:]:
        # Some convergence issues in sequential_substitution_VL at lower pressures
        pkg.flash(T=T, VF=0+1e-9, zs=zs)
        P_bubbles_almost.append(pkg.P)
        pkg.flash(T=T, VF=1-1e-9, zs=zs)
        P_dews_almost.append(pkg.P)
    
    assert_allclose(P_bubbles[4:], P_bubbles_almost, rtol=5e-6)
    assert_allclose(P_dews[4:], P_dews_almost, rtol=5e-6)
    
    
    # Some points fail here too!
    for P, T in zip(P_dews_expect[4:-1], Ts[4:-1]):
        pkg.flash(P=P, VF=1-1e-9, zs=zs)
        assert_allclose(P, pkg.P)
        
    for P, T in zip(P_bubbles_expect[2:-2], Ts[2:-2]):
        pkg.flash(P=P, VF=0+1e-9, zs=zs)
        assert_allclose(P, pkg.P)
    

def test_PVF_parametric_binary_vs_CoolProp():
    import CoolProp.CoolProp as CP
    zs = [0.4, 0.6]
    m = Mixture(['Ethane', 'Heptane'], zs=zs, T=300, P=1E6)
    
    
    kij = .0067
    kijs = [[0,kij],[kij,0]]
    Tcs = [305.322, 540.13]
    Pcs = [4872200.0, 2736000.0]
    omegas = [0.099, 0.349]
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



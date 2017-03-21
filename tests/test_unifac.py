# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import numpy as np
import pandas as pd
from thermo.unifac import *


def test_UNIFAC_data():
    # Test the interaction pairs
    assert len(UFIP) == 54
    assert sum([len(i) for i in UFIP.values()]) == 1270
    val_sum = sum([np.sum(np.abs(list(i.values()))) for i in UFIP.values()])
    assert_allclose(val_sum, 449152.27169999998)
    assert_allclose(UFIP[1][2], 86.02)
    
    for G in UFIP.keys():
        assert G in UFMG
        for G2 in UFIP[G].keys():
            assert G2 in UFMG
    
    for G in UFSG.values():
        assert G.main_group_id in UFMG
        assert UFMG[G.main_group_id][0] == G.main_group
        
    # subgroup strings:
    # [i.group for i in UFSG.values()]
    # ['CH3', 'CH2', 'CH', 'C', 'CH2=CH', 'CH=CH', 'CH2=C', 'CH=C', 'ACH', 'AC', 'ACCH3', 'ACCH2', 'ACCH', 'OH', 'CH3OH', 'H2O', 'ACOH', 'CH3CO', 'CH2CO', 'CHO', 'CH3COO', 'CH2COO', 'HCOO', 'CH3O', 'CH2O', 'CHO', 'THF', 'CH3NH2', 'CH2NH2', 'CHNH2', 'CH3NH', 'CH2NH', 'CHNH', 'CH3N', 'CH2N', 'ACNH2', 'C5H5N', 'C5H4N', 'C5H3N', 'CH3CN', 'CH2CN', 'COOH', 'HCOOH', 'CH2CL', 'CHCL', 'CCL', 'CH2CL2', 'CHCL2', 'CCL2', 'CHCL3', 'CCL3', 'CCL4', 'ACCL', 'CH3NO2', 'CH2NO2', 'CHNO2', 'ACNO2', 'CS2', 'CH3SH', 'CH2SH', 'FURFURAL', 'DOH', 'I', 'BR', 'CH=-C', 'C=-C', 'DMSO', 'ACRY', 'CL-(C=C)', 'C=C', 'ACF', 'DMF', 'HCON(..', 'CF3', 'CF2', 'CF', 'COO', 'SIH3', 'SIH2', 'SIH', 'SI', 'SIH2O', 'SIHO', 'SIO', 'NMP', 'CCL3F', 'CCL2F', 'HCCL2F', 'HCCLF', 'CCLF2', 'HCCLF2', 'CCLF3', 'CCL2F2', 'AMH2', 'AMHCH3', 'AMHCH2', 'AM(CH3)2', 'AMCH3CH2', 'AM(CH2)2', 'C2H5O2', 'C2H4O2', 'CH3S', 'CH2S', 'CHS', 'MORPH', 'C4H4S', 'C4H3S', 'C4H2S', 'NCO', '(CH2)2SU', 'CH2CHSU', 'IMIDAZOL', 'BTI']
    # Main group strings:
    # [i[0] for i in UFMG.values()]
    # ['CH2', 'C=C', 'ACH', 'ACCH2', 'OH', 'CH3OH', 'H2O', 'ACOH', 'CH2CO', 'CHO', 'CCOO', 'HCOO', 'CH2O', 'CNH2', 'CNH', '(C)3N', 'ACNH2', 'PYRIDINE', 'CCN', 'COOH', 'CCL', 'CCL2', 'CCL3', 'CCL4', 'ACCL', 'CNO2', 'ACNO2', 'CS2', 'CH3SH', 'FURFURAL', 'DOH', 'I', 'BR', 'C=-C', 'DMSO', 'ACRY', 'CLCC', 'ACF', 'DMF', 'CF2', 'COO', 'SIH2', 'SIO', 'NMP', 'CCLF', 'CON(AM)', 'OCCOH', 'CH2S', 'MORPH', 'THIOPHEN', 'NCO', 'SULFONES', 'IMIDAZOL', 'BTI']

def test_modified_UNIFAC_data():
    assert len(DOUFIP2006) == 61
    assert len(DOUFIP2016) == 65
    assert sum([len(i) for i in DOUFIP2016.values()]) == 1516
    assert sum([len(i) for i in DOUFIP2006.values()]) == 1318
    val_sum = np.sum(np.abs(np.vstack([np.array(list(i.values())) for i in DOUFIP2006.values() if i.values()])), axis=0)
    assert_allclose(val_sum, [831285.1119000008, 2645.011300000001, 3.068751211])          
    val_sum = np.sum(np.abs(np.vstack([np.array(list(i.values())) for i in DOUFIP2016.values() if i.values()])), axis=0)
    assert_allclose(val_sum, [1011296.5521000021, 3170.4274820000005, 3.7356898600000004])          
              
    assert_allclose(DOUFIP2016[1][2], (189.66, -0.2723, 0.0))
    assert_allclose(DOUFIP2006[1][2], (189.66, -0.2723, 0.0))

    for G in DOUFSG.values():
        assert G.main_group_id in DOUFMG
        assert DOUFMG[G.main_group_id][0] == G.main_group
 
    for d in [DOUFIP2006]:
        for G in d.keys():
            assert G in DOUFMG
            for G2 in d[G].keys():
                assert G2 in DOUFMG
    # Missing some of them for DOUFIP2016 - the actual groups are known but not the numbers


    # [i.group for i in DOUFSG.values()]
    # ['CH3', 'CH2', 'CH', 'C', 'CH2=CH', 'CH=CH', 'CH2=C', 'CH=C', 'ACH', 'AC', 'ACCH3', 'ACCH2', 'ACCH', 'OH(P)', 'CH3OH', 'H2O', 'ACOH', 'CH3CO', 'CH2CO', 'CHO', 'CH3COO', 'CH2COO', 'HCOO', 'CH3O', 'CH2O', 'CHO', 'THF', 'CH3NH2', 'CH2NH2', 'CHNH2', 'CH3NH', 'CH2NH', 'CHNH', 'CH3N', 'CH2N', 'ACNH2', 'AC2H2N', 'AC2HN', 'AC2N', 'CH3CN', 'CH2CN', 'COOH', 'HCOOH', 'CH2CL', 'CHCL', 'CCL', 'CH2CL2', 'CHCL2', 'CCL2', 'CHCL3', 'CCL3', 'CCL4', 'ACCL', 'CH3NO2', 'CH2NO2', 'CHNO2', 'ACNO2', 'CS2', 'CH3SH', 'CH2SH', 'FURFURAL', 'DOH', 'I', 'BR', 'CH=-C', 'C=-C', 'DMSO', 'ACRY', 'CL-(C=C)', 'C=C', 'ACF', 'DMF', 'HCON(..', 'CF3', 'CF2', 'CF', 'COO', 'CY-CH2', 'CY-CH', 'CY-C', 'OH(S)', 'OH(T)', 'CY-CH2O', 'TRIOXAN', 'CNH2', 'NMP', 'NEP', 'NIPP', 'NTBP', 'CONH2', 'CONHCH3', 'CONHCH2', 'AM(CH3)2', 'AMCH3CH2', 'AM(CH2)2', 'AC2H2S', 'AC2HS', 'AC2S', 'H2COCH', 'COCH', 'HCOCH', '(CH2)2SU', 'CH2SUCH', '(CH3)2CB', '(CH2)2CB', 'CH2CH3CB', 'H2COCH2', 'CH3S', 'CH2S', 'CHS', 'H2COC', 'C3H2N2+', 'BTI-', 'C3H3N2+', 'C4H8N+', 'BF4-', 'C5H5N+', 'OTF-', '-S-S-']
    # [i[0] for i in UFMG.values()]
    # [i[0] for i in DOUFMG.values()]
    # ['CH2', 'C=C', 'ACH', 'ACCH2', 'OH', 'CH3OH', 'H2O', 'ACOH', 'CH2CO', 'CHO', 'CCOO', 'HCOO', 'CH2O', 'CH2NH2', 'CH2NH', '(C)3N', 'ACNH2', 'PYRIDINE', 'CH2CN', 'COOH', 'CCL', 'CCL2', 'CCL3', 'CCL4', 'ACCL', 'CNO2', 'ACNO2', 'CS2', 'CH3SH', 'FURFURAL', 'DOH', 'I', 'BR', 'C=-C', 'DMSO', 'ACRY', 'CLCC', 'ACF', 'DMF', 'CF2', 'COO', 'CY-CH2', 'CY-CH2O', 'HCOOH', 'CHCL3', 'CY-CONC', 'CONR', 'CONR2', 'HCONR', 'ACS', 'EPOXIDES', 'CARBONAT', 'SULFONE', 'SULFIDES', 'IMIDAZOL', 'BTI', 'PYRROL', 'BF4', 'PYRIDIN', 'OTF', 'DISULFIDES']
def test_modified_UNIFAC_NIST_data():
    pass

def test_UNIFAC():
    # Gmehling
    # 05.22 VLE of Hexane-Butanone-2 Via UNIFAC (p. 289)
    # Mathcad (2001) - Solution (zip) - step by step
    # http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/05.22a%20VLE%20of%20Hexane-Butanone-2%20Via%20UNIFAC%20-%20Step%20by%20Step.xps
    gammas_0522A = UNIFAC(chemgroups=[{1:2, 2:4}, {1:1, 2:1, 18:1}], T=60+273.15, xs=[0.5, 0.5])
    assert_allclose(gammas_0522A, [1.4276025835624173, 1.3646545010104225])
    assert_allclose(gammas_0522A, [1.428, 1.365], atol=0.001)
    assert_allclose(gammas_0522A, [1.4276, 1.36466], atol=0.0001) # Another calculator
    
    
    
    # Example 4.14 Activity Coefficients of Ethanol + Benzene with the UNIFAC method
    # Walas, Phase Equilibria in Chemical Engineering
    gammas_414 = UNIFAC(chemgroups=[{1:1, 2:1, 14:1}, {9:6}], T=345., xs=[0.2, 0.8])
    assert_allclose(gammas_414, [2.90999524962436, 1.1038643452317465])
    # Matches faily closely. Confirmed it uses the same coefficients.
    assert_allclose(gammas_414, [2.9119, 1.10832], atol=0.005)
    
    # Examples from ACTCOEFF.XLS, chethermo, from Introductory Chemical Engineering 
    # Thermodynamics, 2nd Ed.: an undergraduate chemical engineering text,
    #  J.Richard Elliott and Carl T. Lira, http://chethermo.net/
    # All match exactly; not even a different gas constant used
    # isopropyl alcohol-water
    gammas_ACTCOEFF_1 = UNIFAC(chemgroups=[{1:2, 3:1, 14:1}, {16:1}], T=80.37+273.15, xs=[0.5, 0.5])
    gammas_ACTCOEFF_1_expect = [1.2667572876079400, 1.700192255741180]
    assert_allclose(gammas_ACTCOEFF_1, gammas_ACTCOEFF_1_expect)
    gammas_ACTCOEFF_2 = UNIFAC(chemgroups=[{1:2, 3:1, 14:1}, {16:1}], T=80.37+273.15, xs=[0.1, 0.9])
    gammas_ACTCOEFF_2_expect = [5.0971362612830500, 1.058637792621310]
    assert_allclose(gammas_ACTCOEFF_2, gammas_ACTCOEFF_2_expect)
    # Add in Propionic acid
    gammas_ACTCOEFF_3 = UNIFAC(chemgroups=[{1:2, 3:1, 14:1}, {16:1}, {1:1, 2:1, 42:1}], T=80.37+273.15, xs=[0.1, 0.1, 0.8])
    gammas_ACTCOEFF_3_expect = [0.9968890535625640, 2.170957708441830, 1.0011209111895400]
    assert_allclose(gammas_ACTCOEFF_3, gammas_ACTCOEFF_3_expect)
    # Add in ethanol
    gammas_ACTCOEFF_4 = UNIFAC(chemgroups=[{1:2, 3:1, 14:1}, {16:1}, {1:1, 2:1, 42:1}, {1:1, 2:1, 14:1}], T=80.37+273.15, xs=[0.01, 0.01, 0.01, .97])
    gammas_ACTCOEFF_4_expect = [1.0172562157805600, 2.721887947655120, 0.9872477280109450, 1.000190624095510]
    assert_allclose(gammas_ACTCOEFF_4, gammas_ACTCOEFF_4_expect)
    # Add in pentane
    gammas_ACTCOEFF_5 = UNIFAC(chemgroups=[{1:2, 3:1, 14:1}, {16:1}, {1:1, 2:1, 42:1}, {1:1, 2:1, 14:1}, {1:2, 2:3}], T=80.37+273.15, xs=[.1, .05, .1, .25, .5])
    gammas_ACTCOEFF_5_expect = [1.2773557137580500, 8.017146862811100, 1.1282116576861800, 1.485860948162550, 1.757426505841570]
    assert_allclose(gammas_ACTCOEFF_5, gammas_ACTCOEFF_5_expect)


    # Acetone and Pentane at 307 K and x1 = 0.047
    # Example 8-12 in Poling et al., 5E
    gammas_Poling_5e = UNIFAC(chemgroups=[{1:1, 18:1}, {1:2, 2:3}], T=307, xs=[0.047, 0.953])
    gammas_Poling_known = [4.992034311484559, 1.00526021118788]
    assert_allclose(gammas_Poling_5e, gammas_Poling_known)
    assert_allclose(gammas_Poling_5e, [4.99, 1.005], atol=0.003)
    
    gammas_Poling_with_cache = UNIFAC(chemgroups=[{1:1, 18:1}, {1:2, 2:3}], T=307, xs=[0.047, 0.953], cached=([2.5735, 3.8254], [2.336, 3.316], {1: 3, 18: 1, 2: 3}))
    assert_allclose(gammas_Poling_with_cache, gammas_Poling_known)
    # Test the caching
    
    # Another case with the same mixture
    gammas_custom = UNIFAC(chemgroups=[{1:1, 18:1}, {1:2, 2:3}], T=307, xs=[.674747, .325251])
    assert_allclose(gammas_custom, [1.1645751997624518, 2.105331695192004])


def test_UNIFAC_modified_2006():
    # 11.02 Azeotropic Points in the Quaternary System Benzene - Cyclohexane - Acetone - Ethanol Using Mod. UNIFAC-1.xps
    gammas_1102_1 = UNIFAC(chemgroups=[{9:6}, {78:6}, {1:1, 18:1}, {1:1, 2:1, 14:1}], T=373.15, xs=[0.2, 0.3, 0.2, 0.2],
                             subgroup_data=DOUFSG, interaction_data=DOUFIP2006, modified=True)
    # Values in .xps
    gammas_1102_1_known = [1.18643111, 1.44028013, 1.20447983, 1.97207061]
    assert_allclose(gammas_1102_1, gammas_1102_1_known)
    # Recalculated values with more precision, still matching exactly
    gammas_1102_1_known2 = [1.18643111370682970, 1.44028013391119700, 1.20447983349960850, 1.97207060902998130]
    assert_allclose(gammas_1102_1, gammas_1102_1_known2, rtol=1E-14)
    # 290 K, x3=0.3 to balance
    gammas_1102_2 = UNIFAC(chemgroups=[{9:6}, {78:6}, {1:1, 18:1}, {1:1, 2:1, 14:1}], T=290, xs=[0.2, 0.3, 0.3, 0.2],
                             subgroup_data=DOUFSG, interaction_data=DOUFIP2006, modified=True)
    gammas_1102_2_known = [1.2555831362844658, 2.002790560351622, 1.313653013490284, 2.4472442902051923]
    assert_allclose(gammas_1102_2_known, gammas_1102_2, rtol=1E-13)
    
    # 0.01 mole fractions except last, 250 K
    gammas_1102_3 = UNIFAC(chemgroups=[{9:6}, {78:6}, {1:1, 18:1}, {1:1, 2:1, 14:1}], T=250, xs=[0.01, 0.01, 0.01, 0.97], subgroup_data=DOUFSG, interaction_data=DOUFIP2006, modified=True)
    gammas_1102_3_known = [6.233033961983859, 10.01994111294437, 3.376394671321658, 1.00137007335149700]
    assert_allclose(gammas_1102_3_known, gammas_1102_3, rtol=1E-13)


def test_UNIFAC_misc():
    from scipy.misc import derivative
    from scipy.constants import R
    from math import log
    T = 273.15 + 60
    
    def gE_T(T):
        xs = [0.5, 0.5]
        gammas = UNIFAC(chemgroups=[{1:2, 2:4}, {1:1, 2:1, 18:1}], T=T, xs=xs)
        return R*T*sum(xi*log(gamma) for xi, gamma in zip(xs, gammas))
    
    def hE_T(T):
        to_diff = lambda T: gE_T(T)/T
        return -derivative(to_diff, T,dx=1E-5, order=7)*T**2

    # A source gives 854.758 for hE, matching to within a gas constant
    assert_allclose(hE_T(T), 854.771631451345)
    assert_allclose(gE_T(T), 923.6408846044955)



def test_Van_der_Waals_area():
    # DIPPR and YAWS, hexane, units are good
    assert_allclose(Van_der_Waals_area(3.856), 964000.0)

def test_Van_der_Waals_volume():
    # DIPPR and YAWS, hexane, units are good
    assert_allclose(Van_der_Waals_volume(4.4998), 6.826196599999999e-05)
    
    
def test_UNIFAC_psi():
    assert_allclose(UNIFAC_psi(307, 18, 1, UFSG, UFIP), 0.9165248264184787)
    
    assert_allclose(UNIFAC_psi(373.15, 9, 78, DOUFSG, DOUFIP2006, modified=True), 1.3703140538273264)
    


def test_UNIFAC_flash_1():
    from thermo.activity import flash_inner_loop, K_value
    def flash_UNIFAC_sequential_substitution(T, P, zs, Psats, chemgroups):
        gammas = UNIFAC(chemgroups=chemgroups, T=T, xs=zs)
        Ks = [K_value(P=P, Psat=Psat, gamma=gamma) for Psat, gamma in zip(Psats, gammas)]
        V_over_F, xs, ys = flash_inner_loop(zs, Ks)
        for i in range(100):
            gammas = UNIFAC(chemgroups=chemgroups, T=T, xs=xs)
            Ks = [K_value(P=P, Psat=Psat, gamma=gamma) for Psat, gamma in zip(Psats, gammas)]
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
            xs, ys = xs_new, ys_new
            if err < 1E-11:
                break
        return V_over_F, xs, ys

    T = 307
    P = 1E5
    zs = [0.5, 0.5]
    chemgroups = [{1:1, 18:1}, {1:2, 2:3}]
    Psats = [44501.41359963363, 93853.94807811991]
    ans = flash_UNIFAC_sequential_substitution(T=T, P=P, zs=zs, Psats=Psats, chemgroups=chemgroups)
    assert_allclose(ans[0], 0.5101142364235425)
    assert_allclose(ans[1], [0.6594292844045343, 0.34057071559546576])
    assert_allclose(ans[2], [0.3468928503651561, 0.653107149634844])
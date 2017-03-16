# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from thermo.vapor_pressure import *
from thermo.vapor_pressure import VDI_TABULAR
from thermo.identifiers import checkCAS

### Regression equations

def test_Wagner_original():
    # Methane, coefficients from [2]_, at 100 K
    Psat = Wagner_original(100.0, 190.53, 4596420., -6.00435, 1.1885, -0.834082, -1.22833)
    assert_allclose(Psat, 34520.44601450496)


def test_Wagner():
    # Methane, coefficients from [2]_, at 100 K.
    Psat = Wagner(100., 190.551, 4599200, -6.02242, 1.26652, -0.5707, -1.366)
    assert_allclose(Psat, 34415.00476263708)


def test_TRC_Antoine_extended():
    # Tetrafluoromethane, coefficients from [1]_, at 180 K:
    Psat = TRC_Antoine_extended(180.0, 227.51, -120., 8.95894, 510.595, -15.95, 2.41377, -93.74, 7425.9)
    assert_allclose(Psat, 706317.0898414153)

    # Test x is restricted to 0
    Psat = TRC_Antoine_extended(120.0, 227.51, -120., 8.95894, 510.595, -15.95, 2.41377, -93.74, 7425.9)
    assert_allclose(Psat, 11265.018958511126)


def test_Antoine():
    # Methane, coefficients from [1]_, at 100 K:
    Psat = Antoine(100.0, 8.7687, 395.744, -6.469)
    assert_allclose(Psat, 34478.367349639906)
    
    # Tetrafluoromethane, coefficients from [1]_, at 180 K
    Psat = Antoine(180, A=8.95894, B=510.595, C=-15.95)
    assert_allclose(Psat, 702271.0518579542)
    
    # Oxygen at 94.91 K, with coefficients from [3]_ in units of °C, mmHg, log10,
    # showing the conversion of coefficients A (mmHg to Pa) and C (°C to K)
    Psat = Antoine(94.91, 6.83706+2.1249, 339.2095, 268.70-273.15)
    assert_allclose(Psat, 162978.88655572367)


### Data integrity
def test_WagnerMcGarry():
    sums_calc = [WagnerMcGarry[i].abs().sum() for i in ['A', 'B', 'C', 'D', 'Pc', 'Tc', 'Tmin']]
    sums = [1889.3027499999998, 509.57053652899992, 1098.2766456999998, 1258.0866876, 1005210819, 129293.19100000001, 68482]
    assert_allclose(sums_calc, sums)

    assert WagnerMcGarry.index.is_unique
    assert WagnerMcGarry.shape == (245, 8)
    assert all([checkCAS(i) for i in WagnerMcGarry.index])


def test_AntoinePoling():
    sums_calc =  [AntoinePoling[i].abs().sum() for i in ['A', 'B', 'C', 'Tmin', 'Tmax']]
    sums = [2959.75131, 398207.29786, 18732.24601, 86349.09, 120340.66]
    assert_allclose(sums_calc, sums)

    assert AntoinePoling.index.is_unique
    assert AntoinePoling.shape == (325, 6)
    assert all([checkCAS(i) for i in AntoinePoling.index])


def test_WagnerPoling():
    sums_calc =  [WagnerPoling[i].abs().sum() for i in ['A', 'B', 'C', 'D', 'Tmin', 'Tmax', 'Tc', 'Pc']]
    sums = [894.39071999999999, 271.76480999999995, 525.8134399999999, 538.25393000000008, 24348.006000000001, 59970.149999999994, 63016.021000000001, 357635500]
    assert_allclose(sums_calc, sums)

    assert WagnerPoling.index.is_unique
    assert WagnerPoling.shape == (104, 9)
    assert all([checkCAS(i) for i in WagnerPoling.index])


def test_AntoineExtended():
    sums_calc = [AntoineExtended[i].abs().sum() for i in ['A', 'B', 'C', 'Tc', 'to', 'n', 'E', 'F', 'Tmin', 'Tmax']]
    sums = [873.55827000000011, 107160.285, 4699.9650000000001, 47592.470000000001, 7647, 241.56537999999998, 22816.815000000002, 1646509.79, 33570.550000000003, 46510.849999999999]
    assert_allclose(sums_calc, sums)

    assert AntoineExtended.index.is_unique
    assert AntoineExtended.shape == (97, 11)
    assert all([checkCAS(i) for i in AntoineExtended.index])
    
def test_VDI_PPDS_3_data():
    '''I believe there are no errors here. Average temperature deviation
    0.144% vs tabulated values. 
    '''
    assert all([checkCAS(i) for i in VDI_PPDS_3.index])
    tots_calc = [VDI_PPDS_3[i].abs().sum() for i in [u'A', u'B', u'C', u'D', u'Tc', u'Pc', u'Tm']]
    tots = [2171.4607300000002, 694.38631999999996, 931.3604499999999, 919.88944000000004, 150225.16000000003, 1265565000, 56957.849999999991]
    assert_allclose(tots_calc, tots)
    
    assert VDI_PPDS_3.index.is_unique
    assert VDI_PPDS_3.shape == (275, 8)


### CSP relationships
def test_boiling_critical_relation():
    P = boiling_critical_relation(347.2, 409.3, 617.1, 36E5)
    assert_allclose(P, 15209.467273093938)


def test_Lee_Kesler():
    # Example from [2]_; ethylbenzene at 347.2 K.
    # Their result is 0.132 bar.
    P = Lee_Kesler(347.2, 617.1, 36E5, 0.299)
    assert_allclose(P, 13078.694162949312)


def test_Ambrose_Walton():
    # Example from [2]_; ethylbenzene at 347.25 K.
    # Their result is 0.1329 bar.
    Psat = Ambrose_Walton(347.25, 617.15, 36.09E5, 0.304)
    assert_allclose(Psat, 13278.878504306222)

    
def test_Edalat():
    # No check data, but gives the same results as the other CSP relationships
    Psat = Edalat(347.2, 617.1, 36E5, 0.299)
    assert_allclose(Psat, 13461.273080743307)

def test_Sanjari():
    P = Sanjari(347.2, 617.1, 36E5, 0.299)
    assert_allclose(P, 13651.916109552498)

    Ts_dat = [125.45, 126.54775, 127.6455, 128.74325, 129.841, 130.93875, 132.0365, 133.13425, 134.232, 135.32975, 136.4275, 137.52525, 138.623, 139.72075, 140.8185, 141.91625, 143.014, 144.11175, 145.2095, 146.30725, 147.405, 148.50275, 149.6005, 150.69825, 151.796, 152.89375, 153.9915, 155.08925, 156.187, 157.28475, 158.3825, 159.48025, 160.578, 161.67575, 162.7735, 163.87125, 164.969, 166.06675, 167.1645, 168.26225, 169.36, 170.45775, 171.5555, 172.65325, 173.751, 174.84875, 175.9465, 177.04425, 178.142, 179.23975, 180.3375, 181.43525, 182.533, 183.63075, 184.7285, 185.82625, 186.924, 188.02175, 189.1195, 190.21725, 191.315, 192.41275, 193.5105, 194.60825, 195.706, 196.80375, 197.9015, 198.99925, 200.097, 201.19475, 202.2925, 203.39025, 204.488, 205.58575, 206.6835, 207.78125, 208.879, 209.97675, 211.0745, 212.17225, 213.27, 214.36775, 215.4655, 216.56325, 217.661, 218.75875, 219.8565, 220.95425, 222.052, 223.14975, 224.2475, 225.34525, 226.443, 227.54075, 228.6385, 229.73625, 230.834, 231.93175, 233.0295, 234.12725, 235.225, 236.32275, 237.4205, 238.51825, 239.616, 240.71375, 241.8115, 242.90925, 244.007, 245.10475, 246.2025, 247.30025, 248.398, 249.49575, 250.5935, 251.69125, 252.789, 253.88675, 254.9845, 256.08225, 257.18, 258.27775, 259.3755, 260.47325, 261.571, 262.66875, 263.7665, 264.86425, 265.962, 267.05975, 268.1575, 269.25525, 270.353, 271.45075, 272.5485, 273.64625, 274.744, 275.84175, 276.9395, 278.03725, 279.135, 280.23275, 281.3305, 282.42825, 283.526, 284.62375, 285.7215, 286.81925, 287.917, 289.01475, 290.1125, 291.21025, 292.308, 293.40575, 294.5035, 295.60125, 296.699, 297.79675, 298.8945, 299.99225, 301.09, 302.18775, 303.2855, 304.38325, 305.481, 306.57875, 307.6765, 308.77425, 309.872, 310.96975, 312.0675, 313.16525, 314.263, 315.36075, 316.4585, 317.55625, 318.654, 319.75175, 320.8495, 321.94725, 323.045, 324.14275, 325.2405, 326.33825, 327.436, 328.53375, 329.6315, 330.72925, 331.827, 332.92475, 334.0225, 335.12025, 336.218, 337.31575, 338.4135, 339.51125, 340.609, 341.70675, 342.8045, 343.90225]
    Ps_dat = [2.01857353521E-006, 0.000002517, 3.12468960653E-006, 3.86254620966E-006, 4.75480477553E-006, 5.82952636953E-006, 7.11906108993E-006, 8.66057817586E-006, 1.04966307916E-005, 1.2675775303E-005, 1.52532334366E-005, 1.82916090268E-005, 2.18616534529E-005, 2.60430842891E-005, 3.0925458133E-005, 3.66090963473E-005, 4.32060629106E-005, 5.08412011933E-005, 5.96532246339E-005, 6.97958540095E-005, 0.000081439, 9.47700963623E-005, 0.0001099952, 0.0001273406, 0.0001470541, 0.0001694061, 0.000194692, 0.0002232326, 0.0002553766, 0.0002915015, 0.0003320157, 0.00037736, 0.0004280092, 0.0004844737, 0.0005473018, 0.0006170807, 0.0006944384, 0.0007800461, 0.0008746188, 0.0009789181, 0.0010937533, 0.0012199834, 0.0013585185, 0.0015103219, 0.0016764114, 0.0018578609, 0.0020558023, 0.0022714268, 0.0025059864, 0.0027607955, 0.0030372323, 0.0033367399, 0.0036608279, 0.0040110736, 0.0043891231, 0.0047966926, 0.0052355691, 0.0057076118, 0.0062147529, 0.0067589985, 0.0073424292, 0.0079672011, 0.0086355463, 0.0093497735, 0.0101122685, 0.0109254949, 0.011791994, 0.0127143855, 0.0136953675, 0.0147377169, 0.0158442892, 0.0170180188, 0.0182619187, 0.0195790806, 0.0209726748, 0.0224459497, 0.0240022316, 0.0256449245, 0.0273775098, 0.0292035452, 0.0311266652, 0.0331505797, 0.0352790737, 0.0375160069, 0.0398653126, 0.0423309973, 0.0449171398, 0.0476278905, 0.0504674705, 0.0534401708, 0.0565503513, 0.0598024404, 0.0632009333, 0.0667503919, 0.0704554433, 0.074320779, 0.078351154, 0.0825513859, 0.0869263538, 0.0914809975, 0.0962203163, 0.1011493683, 0.1062732693, 0.111597192, 0.1171263649, 0.1228660716, 0.1288216497, 0.1349984903, 0.1414020366, 0.1480377837, 0.1549112773, 0.1620281134, 0.1693939372, 0.1770144428, 0.1848953722, 0.1930425148, 0.2014617071, 0.2101588319, 0.219139818, 0.2284106396, 0.2379773164, 0.2478459127, 0.2580225378, 0.2685133453, 0.2793245335, 0.2904623449, 0.3019330665, 0.3137430302, 0.3258986125, 0.3384062351, 0.3512723653, 0.3645035164, 0.3781062484, 0.3920871688, 0.4064529329, 0.4212102456, 0.4363658616, 0.4519265872, 0.4678992814, 0.4842908574, 0.5011082844, 0.5183585892, 0.5360488584, 0.5541862404, 0.5727779482, 0.5918312616, 0.6113535304, 0.6313521773, 0.6518347014, 0.6728086819, 0.6942817823, 0.7162617546, 0.7387564438, 0.7617737937, 0.785321852, 0.8094087764, 0.8340428416, 0.8592324462, 0.8849861205, 0.9113125351, 0.9382205104, 0.9657190263, 0.9938172336, 1.0225244661, 1.0518502533, 1.0818043358, 1.1123966802, 1.1436374973, 1.1755372609, 1.2081067294, 1.2413569686, 1.2752993784, 1.3099457208, 1.3453081525, 1.3813992602, 1.4182321006, 1.4558202448, 1.4941778289, 1.5333196103, 1.573261032, 1.614018295, 1.6556084416, 1.6980494505, 1.7413603456, 1.7855613233, 1.8306738995, 1.8767210817, 1.9237275734, 1.9717200158, 2.0207272795, 2.0707808186, 2.1219151068, 2.1741681851, 2.2275823605, 2.2822051216, 2.338090372, 2.3953001459, 2.4539070606, 2.5139977584, 2.575676075]

    AARD_calc = sum([abs(Sanjari(T, 345.0, 26.40E5, 0.3170)-P*1E6)/(P*1E6) for T, P in zip(Ts_dat, Ps_dat)])/len(Ts_dat)
    assert_allclose(AARD_calc, 0.006445800342803334)

    # Supposed to be 1.387 %, small difference
    # Functions are identical- data simply must be different.
    # Or different method sof calculating AARD. No worries.
    AARD_calc = sum([abs(Lee_Kesler(T, 345.0, 26.40E5, 0.3170)-P*1E6)/(P*1E6) for T, P in zip(Ts_dat, Ps_dat)])/len(Ts_dat)
    assert_allclose(AARD_calc, 0.01370923047231833)

    # Supposed to be 0.785 %, small difference; plus formula matches
    AARD_calc = sum([abs(Ambrose_Walton(T, 345.0, 26.40E5, 0.3170)-P*1E6)/(P*1E6) for T, P in zip(Ts_dat, Ps_dat)])/len(Ts_dat)
    assert_allclose(AARD_calc, 0.00841629399152493)


### Main predictor
@pytest.mark.meta_T_dept
def test_VaporPressure():
    # Ethanol, test as many methods asa possible at once
    EtOH = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
    EtOH.T_dependent_property(305.)
    methods = EtOH.sorted_valid_methods
    methods.remove(VDI_TABULAR)
    Psat_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(305.))[1] for i in methods]
    Psat_exp = [11579.634014300127, 11698.02742876088, 11590.408779316374, 11659.154222044575, 11592.205263402893, 11593.661615921257, 11612.378633936816, 11350.156640503357, 12081.738947110121, 14088.453409816764, 9210.26200064024]
    assert_allclose(sorted(Psat_calcs), sorted(Psat_exp))
    
    assert_allclose(EtOH.calculate(305, VDI_TABULAR), 11690.81660829924, rtol=1E-4)

    # Use another chemical to get in ANTOINE_EXTENDED_POLING
    a = VaporPressure(CASRN='589-81-1')
    a.T_dependent_property(410)
    Psat_calcs = [(a.set_user_methods(i), a.T_dependent_property(410))[1] for i in a.sorted_valid_methods]
    Psat_exp = [162944.82134710113, 162870.44794192078, 162865.5380455795]
    assert_allclose(Psat_calcs, Psat_exp)

    # Test that methods return None
    EtOH = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
    EtOH.T_dependent_property(298.15)
    Psat_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(5000))[1] for i in EtOH.sorted_valid_methods]
    assert [None]*11 == Psat_calcs

    # Test interpolation, extrapolation
    w = VaporPressure(Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, CASRN='7732-18-5')
    Ts = np.linspace(300, 350, 10)
    Ps = [3533.918074415897, 4865.419832056078, 6612.2351036034115, 8876.854141719203, 11780.097759775277, 15462.98385942125, 20088.570250257424, 25843.747665059742, 32940.95821687677, 41619.81654904555]
    w.set_tabular_data(Ts=Ts, properties=Ps)
    assert_allclose(w.T_dependent_property(305.), 4715.122890601165)
    w.tabular_extrapolation_permitted = True
    assert_allclose(w.T_dependent_property(200.), 0.5364148240126076)
    w.tabular_extrapolation_permitted = False
    assert_allclose(w.T_dependent_property(200.), 0.09934382362141778) # Fall back to ambrose-Walton


    # Get a check for Antoine Extended
    cycloheptane = VaporPressure(Tb=391.95, Tc=604.2, Pc=3820000.0, omega=0.2384, CASRN='291-64-5')
    cycloheptane.set_user_methods('ANTOINE_EXTENDED_POLING', forced=True)
    assert_allclose(cycloheptane.T_dependent_property(410), 161647.35219882353)
    assert None == cycloheptane.T_dependent_property(400)

    with pytest.raises(Exception):
        cycloheptane.test_method_validity(300, 'BADMETHOD')


test_VaporPressure()
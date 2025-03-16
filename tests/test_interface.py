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
SOFTWARE.
'''

import json

import pytest
from chemicals.dippr import EQ106
from chemicals.utils import property_mass_to_molar, zs_to_ws, ws_to_zs
from fluids.numerics import assert_close, assert_close1d, derivative, linspace

from thermo.interface import *
from thermo.interface import DIGUILIOTEJA, JASPER, LINEAR, VDI_TABULAR, WINTERFELDSCRIVENDAVIS, SurfaceTensionMixture
from thermo.utils import POLY_FIT
from thermo.volume import VDI_PPDS, VolumeLiquid


@pytest.mark.meta_T_dept
def test_SurfaceTension():
    # Ethanol, test as many methods as possible at once
    EtOH = SurfaceTension(Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, StielPolar=-0.01266, CASRN='64-17-5')
    methods = list(EtOH.all_methods)
    methods_nontabular = list(methods)
    methods_nontabular.remove(VDI_TABULAR)
    sigma_calcs = []
    for i in methods_nontabular:
        EtOH.method = i
        sigma_calcs.append(EtOH.T_dependent_property(305.))

    EtOH.method = 'PITZER_SIGMA'
    assert_close(EtOH.T_dependent_property(305.), 0.03905907338532846, rtol=1e-13)
    EtOH.method = 'BROCK_BIRD'
    assert_close(EtOH.T_dependent_property(305.), 0.03739257387107131, rtol=1e-13)
    EtOH.method = 'MIQUEU'
    assert_close(EtOH.T_dependent_property(305.), 0.03805573872932236, rtol=1e-13)
    EtOH.method = 'ZUO_STENBY'
    assert_close(EtOH.T_dependent_property(305.), 0.036707332059707456, rtol=1e-13)
    EtOH.method = 'REFPROP_FIT'
    assert_close(EtOH.T_dependent_property(305.), 0.021391980051928626, rtol=1e-13)
    EtOH.method = 'SASTRI_RAO'
    assert_close(EtOH.T_dependent_property(305.), 0.02645171690486363, rtol=1e-13)
    EtOH.method = 'SOMAYAJULU'
    assert_close(EtOH.T_dependent_property(305.), 0.0217115665365073, rtol=1e-13)
    EtOH.method = 'JASPER'
    assert_close(EtOH.T_dependent_property(305.), 0.02140008, rtol=1e-13)
    EtOH.method = 'VDI_PPDS'
    assert_close(EtOH.T_dependent_property(305.), 0.021462066798796146, rtol=1e-13)
    EtOH.method = 'REFPROP'
    assert_close(EtOH.T_dependent_property(305.), 0.021222422444285596, rtol=1e-13)
    EtOH.method = 'SOMAYAJULU2'
    assert_close(EtOH.T_dependent_property(305.), 0.0217115665365073, rtol=1e-13)

    assert_close(EtOH.calculate(305., VDI_TABULAR), 0.021537011904316786, rtol=5E-4)

    # Test that methods return None
    EtOH.extrapolation = None
    for i in methods:
        EtOH.method = i
        assert EtOH.T_dependent_property(5000) is None


    assert SurfaceTension.from_json(EtOH.as_json()) == EtOH


    EtOH.method = 'VDI_TABULAR'
    EtOH.tabular_extrapolation_permitted = False
    assert None is EtOH.T_dependent_property(700.0)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    # Test Aleem

    CH4 = SurfaceTension(Tb=111.65, Cpl=property_mass_to_molar(2465.,16.04246), Hvap_Tb=510870., MW=16.04246, Vml=3.497e-05, method='Aleem')
    assert_close(CH4.T_dependent_property(90), 0.016704545538936296)

    assert not CH4.test_method_validity(600, 'Aleem')
    assert CH4.test_method_validity(100, 'Aleem')

@pytest.mark.meta_T_dept
def test_SurfaceTension_water_iapws():
    water = SurfaceTension(CASRN="7732-18-5", MW=18.01528, Tb=373.124, Tc=647.14, Pc=22048320.0, Vc=5.6e-05, Zc=0.229, omega=0.344, StielPolar=0.0232, Hvap_Tb=2256470., extrapolation="DIPPR106_AB", method="IAPWS_SIGMA")
    assert water.method == 'IAPWS_SIGMA'
    assert_close(water(400), 0.05357792640201927)

    assert water(3) is not None
    assert water(3) > 0

@pytest.mark.meta_T_dept
def test_SurfaceTension_fits_to_data_mercury():
    obj = SurfaceTension(CASRN='7439-97-6', Tc=1735.0)
    assert_close(obj(300), 0.4740500003730662, rtol=1e-5)

    assert obj(1734.9999) < 1e-4
    assert obj(obj.Tc) == 0.0
    assert obj(1) > obj(200)

@pytest.mark.meta_T_dept
def test_SurfaceTension_fits_to_data_sulfur():
    # Just check there is something there
    obj = SurfaceTension(CASRN="7704-34-9")
    assert_close(obj(500), 0.05184181941716021, rtol=0.08)

@pytest.mark.meta_T_dept
def test_SurfaceTension_fits_to_data_H2O2():
    obj = SurfaceTension(CASRN='7722-84-1')
    assert_close(obj(280), 0.07776090102150693, rtol=0.08)

@pytest.mark.meta_T_dept
def test_SurfaceTensionJasperMissingLimits():
    obj = SurfaceTension(CASRN='110-01-0')
    assert_close(obj.calculate(obj.T_limits[JASPER][1], 'JASPER'), 0, atol=1e-10)

    obj = SurfaceTension(CASRN='14901-07-6')
    assert_close(obj.calculate(obj.T_limits[JASPER][1], 'JASPER'), 0, atol=1e-10)

@pytest.mark.meta_T_dept
def test_SurfaceTensionVDITabularMissingZeroLimits():
    obj = SurfaceTension(CASRN='7782-41-4')
    assert_close(obj.calculate(144.41, 'VDI_TABULAR'), 0, atol=1e-10)

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_SurfaceTension_fitting0_yaws():
    A, Tc, n = 117.684, 7326.47, 1.2222
    Ts = linspace(378.4, 7326.47, 10)
    props_calc = [EQ106(T, Tc, A, n, 0.0, 0.0, 0.0, 0) for T in Ts]
    res, stats = SurfaceTension.fit_data_to_model(Ts=Ts, data=props_calc, model='YawsSigma',
                          do_statistics=True, use_numba=False, model_kwargs={'Tc': Tc},
                          fit_method='lm')
    assert stats['MAE'] < 1e-5


@pytest.mark.meta_T_dept
def test_SurfaceTension_no_polyfit():
    sigma_kwargs_no_polyfit = {'CASRN': '7727-37-9', 'MW': 28.0134, 'Tb': 77.355, 'Tc': 126.2, 'Pc': 3394387.5, 'Vc': 8.95e-05, 'Zc': 0.2895282296391198, 'omega': 0.04, 'StielPolar': 0.009288558616105336, 'Hvap_Tb': 199160.50751339202, 'poly_fit': ()}
    created = SurfaceTension(**sigma_kwargs_no_polyfit)
    assert created(100) is not None

@pytest.mark.meta_T_dept
def test_SurfaceTension_exp_poly_fit_ln_tau():
    coeffs = [-1.2616237655927602e-05, -0.0004061873638525952, -0.005563382112542401, -0.04240531802937599, -0.19805733513004808, -0.5905741856310869, -1.1388001144550794, -0.1477584393673108, -2.401287527958821]
    Tc = 647.096
    Tmin, Tmax = 233.22, 646.15

    # Create an object with no CAS, check a value
    good_obj = SurfaceTension(Tc=Tc, exp_poly_fit_ln_tau=(Tmin, Tmax, Tc, coeffs))
    expect = 0.07175344047522199
    assert_close(good_obj(300), expect, rtol=1e-12)

    # Create an object with a CAS, check a value
    good_obj2 = SurfaceTension(Tc=Tc, CASRN='7732-18-5', exp_poly_fit_ln_tau=(Tmin, Tmax, Tc, coeffs))
    assert_close(good_obj2(300), expect, rtol=1e-12)

    expect_der = -0.000154224581713238
    expect_der2 = -5.959538970287795e-07

    assert_close(good_obj.T_dependent_property_derivative(300), expect_der, rtol=1e-13)
    assert_close(good_obj.T_dependent_property_derivative(300, order=2), expect_der2, rtol=1e-13)


    assert SurfaceTension.from_json(good_obj2.as_json()) == good_obj2
    assert SurfaceTension.from_json(good_obj.as_json()) == good_obj
    assert eval(str(good_obj2)) == good_obj2
    assert eval(str(good_obj)) == good_obj

@pytest.mark.meta_T_dept
def test_SurfaceTension_exp_poly_ln_tau_extrapolate():
    coeffs = [1.1624065398371628, -1.9976745939643825]

    Tc = 647.096
    Tmin, Tmax = 233.22, 646.15
    good_obj = SurfaceTension(Tc=Tc, exp_poly_fit_ln_tau=(Tmin, Tmax, Tc, coeffs), extrapolation='EXP_POLY_LN_TAU2')
    assert_close(good_obj.calculate(1, good_obj.method), good_obj(1), rtol=1e-13)
    assert_close(good_obj.calculate(647, good_obj.method), good_obj(647), rtol=1e-13)

    assert_close(derivative(good_obj, 50, dx=50*1e-7), good_obj.T_dependent_property_derivative(50))

    assert_close(derivative(good_obj.T_dependent_property_derivative, 190, dx=190*1e-6),
                 good_obj.T_dependent_property_derivative(190, order=2), rtol=1e-7)

    assert_close(good_obj.T_dependent_property_derivative(50), -0.0002405128589491164, rtol=1e-11)
    assert_close(good_obj.T_dependent_property_derivative(50, 2),6.541805875147306e-08, rtol=1e-11)

    # Floating-point errors pile up in this one
    coeffs = [-0.02235848200899392,  1.0064575672832703,  -2.0629066032890777 ]
    good_obj = SurfaceTension(Tc=Tc, exp_poly_fit_ln_tau=(Tmin, Tmax, Tc, coeffs), extrapolation='EXP_POLY_LN_TAU3')
    assert_close(good_obj.calculate(200, good_obj.method), good_obj(200), rtol=1e-7)
    assert_close(good_obj.calculate(647, good_obj.method), good_obj(647), rtol=1e-7)

    assert_close(derivative(good_obj, 50, dx=50*1e-7), good_obj.T_dependent_property_derivative(50))

    assert_close(derivative(good_obj.T_dependent_property_derivative, 190, dx=190*1e-6),
                 good_obj.T_dependent_property_derivative(190, order=2), rtol=1e-7)


    assert_close(good_obj.T_dependent_property_derivative(50), -0.00019823412229632575, rtol=1e-11)
    assert_close(good_obj.T_dependent_property_derivative(50, 2), -1.136037983710264e-08, rtol=1e-11)

@pytest.mark.meta_T_dept
def test_SurfaceTension_exp_cheb_fit_ln_tau():
    coeffs = [-5.922664830406188, -3.6003367212635444, -0.0989717205896406, 0.05343895281736921, -0.02476759166597864, 0.010447569392539213, -0.004240542036664352, 0.0017273355647560718, -0.0007199858491173661, 0.00030714447101984343, -0.00013315510546685339, 5.832551964424226e-05, -2.5742454514671165e-05, 1.143577875153956e-05, -5.110008470393668e-06, 2.295229193177706e-06, -1.0355920205401548e-06, 4.690917226601865e-07, -2.1322112805921556e-07, 9.721709759435981e-08, -4.4448656630335925e-08, 2.0373327115630335e-08, -9.359475430792408e-09, 4.308620855930645e-09, -1.9872392620357004e-09, 9.181429297400179e-10, -4.2489342599871804e-10, 1.969051449668413e-10, -9.139573819982871e-11, 4.2452263926406886e-11, -1.9768853221080462e-11, 9.190537220149508e-12, -4.2949394041258415e-12, 1.9981863386142606e-12, -9.396025624219817e-13, 4.335282133283158e-13, -2.0410756418343112e-13, 1.0455525334407412e-13, -4.748978987834107e-14, 2.7630675525358583e-14]
    Tmin, Tmax, Tc = 233.22, 646.15, 647.096

    obj2 = SurfaceTension(Tc=Tc, exp_cheb_fit_ln_tau=((Tmin, Tmax, Tc, coeffs)))

    T = 500
    assert_close(obj2(T), 0.031264474019763455, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(T), -0.00023379922039411865, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(T, order=2), -1.0453010755999069e-07, rtol=1e-13)

@pytest.mark.meta_T_dept
def test_SurfaceTension_exp_stablepoly_fit_ln_tau():
    coeffs = [0.011399360373616219, -0.014916568994522095, -0.06881296308711171, 0.0900153056718409, 0.19066633691545576, -0.24937350547406822, -0.3148389292182401, 0.41171834646956995, 0.3440581845934503, -0.44989947455906076, -0.2590532901358529, 0.33869134876113094, 0.1391329435696207, -0.18195230788023764, -0.050437145563137165, 0.06583166394466389, 0.01685157036382634, -0.022266583863000733, 0.003539388708205138, -0.005171064606571463, 0.012264455189935575, -0.018085676249990357, 0.026950795197264732, -0.04077120220662778, 0.05786417011592615, -0.07222889554773304, 0.07433570330647113, -0.05829288696590232, -3.7182636506596722, -5.844828481765601]
    Tmin, Tmax, Tc = 233.22, 646.15, 647.096

    obj2 = SurfaceTension(Tc=Tc, exp_stablepoly_fit_ln_tau=(Tmin, Tmax, Tc, coeffs))

    T = 500
    assert_close(obj2(T), 0.03126447402046822, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(T), -0.0002337992205182661, rtol=1e-13)
    assert_close(obj2.T_dependent_property_derivative(T, order=2), -1.0453011134030858e-07, rtol=1e-13)

@pytest.mark.meta_T_dept
def test_SurfaceTension_EQ106_ABC_extrapolate():
    Tmin, Tc, A, B = 194.0, 592.5, 0.056, 1.32
    C = -0.01
    Tmax = 590.0

    good_obj = SurfaceTension(Tc=Tc, DIPPR106_parameters={'Test': {'Tmin': Tmin,
                                'Tmax': Tmax, 'Tc': Tc, 'A': A, 'B': B, 'C': C}},
                              extrapolation='DIPPR106_ABC')
    assert_close(good_obj(591), good_obj.calculate(591, good_obj.method), rtol=1e-13)
    assert_close(good_obj(50), good_obj.calculate(50, good_obj.method), rtol=1e-13)

    # Three derivatives
    assert_close(derivative(good_obj, 50, dx=50*1e-7), good_obj.T_dependent_property_derivative(50))

    assert_close(derivative(good_obj.T_dependent_property_derivative, 190, dx=190*1e-6),
                 good_obj.T_dependent_property_derivative(190, order=2), rtol=1e-7)
    assert_close(derivative(lambda T:good_obj.T_dependent_property_derivative(T, 2), 190, dx=190*1e-6),
                 good_obj.T_dependent_property_derivative(190, order=3), rtol=1e-7)


    assert_close(good_obj.T_dependent_property_derivative(50), -0.00012114625271322606, rtol=1e-11)
    assert_close(good_obj.T_dependent_property_derivative(50, 2), 7.405594884975754e-08, rtol=1e-11)
    assert_close(good_obj.T_dependent_property_derivative(50, 3), 7.578114176744999e-11, rtol=1e-11)



    good_obj = SurfaceTension(Tc=Tc, DIPPR106_parameters={'Test': {'Tmin': Tmin,
                                'Tmax': Tmax, 'Tc': Tc, 'A': A, 'B': B}},
                              extrapolation='DIPPR106_AB')
    assert_close(good_obj(591), good_obj.calculate(591, good_obj.method), rtol=1e-13)
    assert_close(good_obj(50), good_obj.calculate(50, good_obj.method), rtol=1e-13)

    assert_close(derivative(good_obj, 50, dx=50*1e-7), good_obj.T_dependent_property_derivative(50))

    assert_close(derivative(good_obj.T_dependent_property_derivative, 190, dx=190*1e-6),
                 good_obj.T_dependent_property_derivative(190, order=2), rtol=1e-7)
    assert_close(derivative(lambda T:good_obj.T_dependent_property_derivative(T, 2), 190, dx=190*1e-6),
                 good_obj.T_dependent_property_derivative(190, order=3), rtol=1e-7)


    assert_close(good_obj.T_dependent_property_derivative(50), -0.00012128895314509345, rtol=1e-11)
    assert_close(good_obj.T_dependent_property_derivative(50, 2), 7.154371429756683e-08, rtol=1e-11)
    assert_close(good_obj.T_dependent_property_derivative(50, 3), 8.967691377390834e-11, rtol=1e-11)

def test_SurfaceTensionMixture():
    # ['pentane', 'dichloromethane']
    T = 298.15
    P = 101325.0
    zs = [.1606, .8394]

    MWs = [72.14878, 84.93258]
    ws = zs_to_ws(zs, MWs)

    VolumeLiquids = [VolumeLiquid(CASRN="109-66-0", MW=72.14878, Tb=309.21, Tc=469.7, Pc=3370000.0, Vc=0.000311, Zc=0.26837097540904814, omega=0.251, dipole=0.0, extrapolation="constant", method=POLY_FIT, poly_fit=(144, 459.7000000000001, [1.0839519373491257e-22, -2.420837244222272e-19, 2.318236501104612e-16, -1.241609625841306e-13, 4.0636406847721776e-11, -8.315431504053525e-09, 1.038485128954003e-06, -7.224842789857136e-05, 0.0022328080060137396])),
     VolumeLiquid(CASRN="75-09-2", MW=84.93258, Tb=312.95, Tc=508.0, Pc=6350000.0, Vc=0.000177, Zc=0.26610258553203137, omega=0.2027, dipole=1.6, extrapolation="constant", method=POLY_FIT, poly_fit=(178.01, 484.5, [1.5991056738532454e-23, -3.92303910541969e-20, 4.1522438881104836e-17, -2.473595776587317e-14, 9.064684097377694e-12, -2.0911320815626796e-09, 2.9653069375266426e-07, -2.3580713574913447e-05, 0.0008567355308938564]))]

    SurfaceTensions = [SurfaceTension(CASRN="109-66-0", MW=72.14878, Tb=309.21, Tc=469.7, Pc=3370000.0, Vc=0.000311, Zc=0.26837097540904814, omega=0.251, StielPolar=0.005164116344598568, Hvap_Tb=357736.5860890071, extrapolation=None, method=VDI_PPDS),
                       SurfaceTension(CASRN="75-09-2", MW=84.93258, Tb=312.95, Tc=508.0, Pc=6350000.0, Vc=0.000177, Zc=0.26610258553203137, omega=0.2027, StielPolar=-0.027514125341022044, Hvap_Tb=333985.7240672881, extrapolation=None, method=VDI_PPDS)]

    obj = SurfaceTensionMixture(MWs=MWs, Tbs=[309.21, 312.95], Tcs=[469.7, 508.0], correct_pressure_pure=False, CASs=['109-66-0', '75-09-2'], SurfaceTensions=SurfaceTensions, VolumeLiquids=VolumeLiquids)

    sigma_wsd = obj.calculate(T, P, zs, method=WINTERFELDSCRIVENDAVIS, ws=ws)
    assert_close(sigma_wsd, 0.02388914831076298)

    # Check the default calculation method is still WINTERFELDSCRIVENDAVIS
    sigma = obj.mixture_property(T, P, zs)
    assert_close(sigma, sigma_wsd)

    # Check the other implemented methods
    sigma = obj.calculate(T, P, zs, ws, LINEAR)
    assert_close(sigma, 0.025332871945242523)

    sigma = obj.calculate(T, P, zs, ws, DIGUILIOTEJA)
    assert_close(sigma, 0.025262398831653664)

    with pytest.raises(Exception):
        obj.test_method_validity(T, P, zs, ws, 'BADMETHOD')
    with pytest.raises(Exception):
        obj.calculate(T, P, zs, ws, 'BADMETHOD')

    hash0 = hash(obj)
    obj2 = SurfaceTensionMixture.from_json(json.loads(json.dumps(obj.as_json())))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

    obj2 = eval(str(obj))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0


def test_SurfaceTensionMixture_RK():
    rk_struct = [[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[-0.001524469890834797, 1.0054473481826371], [0.011306977309368058, -3.260062182048661], [0.015341200351987746, -4.484565752157498], [-0.030368567453463967, 8.99609823333603]]], [[[-0.001524469890834797, 1.0054473481826371], [-0.011306977309368058, 3.260062182048661], [0.015341200351987746, -4.484565752157498], [0.030368567453463967, -8.99609823333603]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]]

    redlick_kister_parameters={'fit test': {'coeffs': rk_struct, 'N_terms': 4, 'N_T': 2}}

    SurfaceTensions = [SurfaceTension(CASRN="7732-18-5", MW=18.01528, Tb=373.124295848, Tc=647.096, Pc=22064000.0, Vc=5.59480372671e-05, Zc=0.22943845208106295, omega=0.3443, StielPolar=0.023504366573307456, Hvap_Tb=2256470.315159003, extrapolation="DIPPR106_AB", method="REFPROP_FIT", exp_stable_polynomial_ln_tau_parameters={'REFPROP_FIT': {'Tc': 647.096, 'coeffs': [-0.013397167543380295, 0.09018370971898611, -0.2520912562903041, 0.3433557312810748, -0.13152273913480572, -0.29531952656781263, 0.5090941125808105, -0.3066181788894154, -0.028094295006648968, 0.16299091616169084, -0.10021702497821923, 0.01293602515363503, 0.013931363949354863, -0.009681075363666861, -0.00037868681772088844, -0.0026032409621614305, -0.005187181836159808, -0.008505280528265961, -0.015031573223813882, -0.027341171777993004, -0.049718571565094134, -0.08358188908174445, -0.11579008507191359, 2.3236782516300956, -4.554811076898462], 'Tmax': 638.9178163265306, 'Tmin': 251.165}}),
                    SurfaceTension(CASRN="7722-84-1", MW=34.01468, Tb=423.35, Tc=728.0, Pc=22000000.0, Vc=7.77e-05, Zc=0.28240874135993943, omega=0.3582, StielPolar=-0.03984834320500852, Hvap_Tb=1336355.5708140612, extrapolation="DIPPR106_AB", method="Fit 2023", DIPPR100_parameters={'Fit 2023': {'A': 0.12117476526756533, 'B': -0.00015504951516449433, 'C': 0.0, 'D': 0.0, 'E': 0.0, 'F': 0.0, 'G': 0.0, 'Tmax': 291.34999999999997, 'Tmin': 273.34999999999997}})]

    obj = SurfaceTensionMixture(MWs=[18.01528, 34.01468], Tbs=[373.124295848, 423.35], 
                        Tcs=[647.096, 728.0], CASs=['7732-18-5', '7722-84-1'], 
                        correct_pressure_pure=False, 
                        SurfaceTensions=SurfaceTensions,
                        redlick_kister_parameters=redlick_kister_parameters)
    assert obj.method == 'fit test'
    ws = [1-0.8631, 0.8631]
    MWs = [18.01528, 34.01468]
    value = obj(273.15, 1e5, ws_to_zs(ws, MWs), ws)
    assert_close(value, 0.07840690708312911, rtol=1e-13)

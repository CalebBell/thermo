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
from chemicals.utils import zs_to_ws
from chemicals.viscosity import *
from fluids.numerics import assert_close, assert_close2d, linspace

from thermo import VaporPressure, VolumeLiquid
from thermo.coolprop import has_CoolProp
from thermo.eos import PR
from thermo.mixture import Mixture
from thermo.viscosity import *
from thermo.viscosity import (
    BROKAW,
    COOLPROP,
    DIPPR_PERRY_8E,
    DUTT_PRASAD,
    HERNING_ZIPPERER,
    JOBACK,
    LALIBERTE_MU,
    LETSOU_STIEL,
    LINEAR,
    LUCAS,
    MIXING_LOG_MOLAR,
    PRZEDZIECKI_SRIDHAR,
    VDI_PPDS,
    VDI_TABULAR,
    VISWANATH_NATARAJAN_2E,
    VISWANATH_NATARAJAN_3,
    WILKE,
)
from thermo.volume import VolumeGas


@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_ViscosityLiquid_CoolProp():
    EtOH = ViscosityLiquid(MW=46.06844, Tm=159.05, Tc=514.0, Pc=6137000.0, Vc=0.000168, omega=0.635, Psat=7872.16, Vml=5.8676e-5, CASRN='64-17-5')
    EtOH.method = (COOLPROP)
    assert_close(EtOH.T_dependent_property(298.15), 0.0010823506202025659, rtol=1e-9)

    # Ethanol compressed
    assert [False, True] == [EtOH.test_method_validity_P(300, P, COOLPROP) for P in (1E3, 1E5)]



@pytest.mark.meta_T_dept
def test_ViscosityLiquid():
    EtOH = ViscosityLiquid(MW=46.06844, Tm=159.05, Tc=514.0, Pc=6137000.0, Vc=0.000168, omega=0.635, Psat=7872.16, Vml=5.8676e-5, CASRN='64-17-5')

    # Test json export
    EtOH2 = ViscosityLiquid.from_json(EtOH.as_json())
    assert EtOH.__dict__ == EtOH2.__dict__

    # Test json export with interpolator objects
    EtOH.method = VDI_TABULAR
    EtOH.T_dependent_property(315)
    s = EtOH.as_json()
    EtOH2 = ViscosityLiquid.from_json(s)
    # Do hash checks before interpolation object exists
    assert hash(EtOH) == hash(EtOH2)
    assert EtOH == EtOH2
    assert id(EtOH) != id(EtOH2)

    assert EtOH.T_dependent_property(315) == EtOH2.T_dependent_property(315)

    EtOH.method = (DIPPR_PERRY_8E)
    assert_close(EtOH.T_dependent_property(298.15), 0.0010774308462863267, rtol=1e-9)
    EtOH.method = (VDI_PPDS)
    assert_close(EtOH.T_dependent_property(298.15), 0.0010623746999654108, rtol=1e-9)
    EtOH.method = (DUTT_PRASAD)
    assert_close(EtOH.T_dependent_property(298.15), 0.0010720812586059744, rtol=1e-9)
    EtOH.method = (VISWANATH_NATARAJAN_3)
    assert_close(EtOH.T_dependent_property(298.15), 0.0031157679801337825, rtol=1e-9)
    EtOH.method = (VDI_TABULAR)
    assert_close(EtOH.T_dependent_property(310), 0.0008726933038017184, rtol=1e-9)
    EtOH.method = (LETSOU_STIEL)
    assert_close(EtOH.T_dependent_property(298.15), 0.0004191198228004421, rtol=1e-9)
    EtOH.method = (PRZEDZIECKI_SRIDHAR)
    assert EtOH.T_dependent_property(298.15) is None
    assert_close(EtOH.T_dependent_property(400.0), 0.00039598337518386806, rtol=1e-9)


    EtOH.extrapolation = None
    for i in EtOH.all_methods:
        EtOH.method = i
        if i == JOBACK:
            continue
        assert EtOH.T_dependent_property(600) is None


    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    # Acetic acid to test Viswanath_Natarajan_2_exponential
    acetic_acid = ViscosityLiquid(CASRN='64-19-7', Tc=590.7, method=VISWANATH_NATARAJAN_2E)
    assert_close(acetic_acid.T_dependent_property(350.0), 0.000587027903931889, rtol=1e-12)

    acetic_acid.extrapolation = None
    for i in acetic_acid.all_methods:
        acetic_acid.method = i
        assert acetic_acid.T_dependent_property(650.0) is None

    # Test Viswanath_Natarajan_2 with boron trichloride
    mu = ViscosityLiquid(CASRN='10294-34-5').T_dependent_property(250)
    assert_close(mu, 0.0003389255178814321)
    assert None is ViscosityLiquid(CASRN='10294-34-5', extrapolation=None).T_dependent_property(350)


    # Ethanol compressed
    EtOH = ViscosityLiquid(MW=46.06844, Tm=159.05, Tc=514.0, Pc=6137000.0, Vc=0.000168, omega=0.635, Psat=7872.16, Vml=5.8676e-5, CASRN='64-17-5')
    assert [True, True] == [EtOH.test_method_validity_P(300, P, LUCAS) for P in (1E3, 1E5)]
    EtOH.method = DIPPR_PERRY_8E

    assert_close(EtOH.calculate_P(298.15, 1E6, LUCAS), 0.0010830773668247635)

    EtOH = ViscosityLiquid(MW=46.06844, Tm=159.05, Tc=514.0, Pc=6137000.0, Vc=0.000168, omega=0.635, Psat=7872.16, Vml=5.8676e-5, CASRN='64-17-5')
    # Ethanol data, calculated from CoolProp
    Ts = [275, 300, 350]
    Ps = [1E5, 5E5, 1E6]
    TP_data = [[0.0017455993713216815, 0.0010445175985089377, 0.00045053170256051774], [0.0017495149679815605, 0.0010472128172002075, 0.000452108003076486], [0.0017543973013034444, 0.0010505716944451827, 0.00045406921275411145]]
    EtOH.add_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_close2d(TP_data, recalc_pts)

    # TEst thta we can export to json, create a new instance with P interpolation objects
    EtOH2 = ViscosityLiquid.from_json( EtOH.as_json())
    recalc_pts2 = [[EtOH2.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_close2d(recalc_pts, recalc_pts2, atol=0, rtol=0)


    EtOH.tabular_extrapolation_permitted = False
    assert None is EtOH.TP_dependent_property(300, 90000.0)
    EtOH.tabular_extrapolation_permitted = True
    assert_close(EtOH.TP_dependent_property(300, 9E4), 0.0010443414003675137)

    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')


    #PRZEDZIECKI_SRIDHAR was not calculating due to volume liquid failure
    obj = ViscosityLiquid(CASRN="67-56-1", MW=32.04186, Tm=175.15, Tc=512.5, Pc=8084000.0, Vc=0.000117,
                        omega=0.559, Psat=VaporPressure(CASRN="67-56-1", Tb=337.65, Tc=512.5, Pc=8084000.0, omega=0.559,
                                                        extrapolation="AntoineAB|DIPPR101_ABC", method="WAGNER_MCGARRY"),
                        Vml=VolumeLiquid(CASRN="67-56-1", MW=32.04186, Tb=337.65, Tc=512.5, Pc=8084000.0, Vc=0.000117,
                                        Zc=0.22196480200068586, omega=0.559, dipole=1.7,
                                        Psat=VaporPressure(CASRN="67-56-1",
                                Tb=337.65, Tc=512.5, Pc=8084000.0, omega=0.559,
                            extrapolation="AntoineAB|DIPPR101_ABC", method="WAGNER_MCGARRY"),
                                extrapolation="constant", method="DIPPR_PERRY_8E"), extrapolation="Arrhenius")
    res = obj.calculate(230, PRZEDZIECKI_SRIDHAR)




@pytest.mark.meta_T_dept
def test_ViscosityLiquid_Joback():
    obj = ViscosityLiquid(CASRN='124-18-5', MW=142.28168)
    obj.method = JOBACK
    assert_close(obj.T_dependent_property(300), 0.0008245072864231092)


@pytest.mark.meta_T_dept
def test_ViscosityLiquid_derivative_exp_poly_fit():
    thing = ViscosityLiquid(load_data=False, method='linear', exp_poly_fit=(90.8, 190.554, [-4.2380635484289735e-14, 4.5493522318564076e-11, -2.116492796413691e-08, 5.572829307777471e-06, -0.0009082059294533955, 0.0937979289971011, -5.994475607852222, 216.68461775245618, -3398.7238594522332]))
    # The first derivative should match closely.
    assert_close(thing.T_dependent_property_derivative(190.553999999999),
        thing.T_dependent_property_derivative(190.55400000001), rtol=1e-4)

    assert_close(thing.T_dependent_property_derivative(90.80000000001),
        thing.T_dependent_property_derivative(90.79999999999999), rtol=1e-5)

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_ViscosityLiquid_fitting0():
    ammonia_Ts_mul = [195.41, 206.081, 216.751, 227.422, 238.092, 239.82, 248.763, 259.433, 270.104, 280.774, 291.445, 302.115, 312.786, 323.456, 334.127, 344.797, 355.468, 366.139, 376.809, 387.48, 398.15]
    ammonia_muls = [0.000526102, 0.000427438, 0.000355131, 0.000300457, 0.00025797, 0.000251978, 0.00022415, 0.00019665, 0.000173865, 0.000154667, 0.000138254, 0.000124039, 0.000111591, 0.000100584, 9.07738E-05, 8.19706E-05, 7.4028E-05, 6.68311E-05, 6.02885E-05, 5.43268E-05, 4.88864E-05]

    fit, res = ViscosityLiquid.fit_data_to_model(Ts=ammonia_Ts_mul, data=ammonia_muls, model='DIPPR101',
                          do_statistics=True, use_numba=False,
                          fit_method='lm')
    assert res['MAE'] < 1e-5

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_ViscosityLiquid_fitting1():
    # Argon - was not fitting properly
    obj = ViscosityLiquid(CASRN='7440-37-1')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # benzamide
    obj = ViscosityLiquid(CASRN='55-21-0')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # acetamide
    obj = ViscosityLiquid(CASRN='60-35-5')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101', multiple_tries=True,
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # 2-butanol
    obj = ViscosityLiquid(CASRN='78-92-2')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # benzene
    obj = ViscosityLiquid(CASRN='71-43-2')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # butane
    obj = ViscosityLiquid(CASRN='106-97-8')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    #  1-chloropropane
    obj = ViscosityLiquid(CASRN='540-54-5')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # m-cresol
    obj = ViscosityLiquid(CASRN='108-39-4')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # o-cresol
    obj = ViscosityLiquid(CASRN='95-48-7')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # cyclohexanol
    obj = ViscosityLiquid(CASRN='108-93-0')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # decane
    obj = ViscosityLiquid(CASRN='124-18-5')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # 1,3-dichlorobenzene
    obj = ViscosityLiquid(CASRN='541-73-1')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # diethanolamine
    obj = ViscosityLiquid(CASRN='111-42-2')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # difluoromethane
    obj = ViscosityLiquid(CASRN='75-10-5')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # dodecane
    obj = ViscosityLiquid(CASRN='112-40-3')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # ethylene glycol
    obj = ViscosityLiquid(CASRN='107-21-1')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # 2-ethylhexanoic acid
    obj = ViscosityLiquid(CASRN='149-57-5')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # ethyltrichlorosilane
    obj = ViscosityLiquid(CASRN='115-21-9')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # formamide
    obj = ViscosityLiquid(CASRN='75-12-7')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # helium
    obj = ViscosityLiquid(CASRN='7440-59-7')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # hexane
    obj = ViscosityLiquid(CASRN='110-54-3')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # hydrazine
    obj = ViscosityLiquid(CASRN='302-01-2')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # hydrogen
    obj = ViscosityLiquid(CASRN='1333-74-0')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # hydrochloric acid
    obj = ViscosityLiquid(CASRN='7647-01-0')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # nitric oxide
    obj = ViscosityLiquid(CASRN='10102-43-9')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # toluene
    obj = ViscosityLiquid(CASRN='108-88-3')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

    # 1,2-propanediol
    obj = ViscosityLiquid(CASRN='57-55-6')
    Ts = linspace(obj.T_limits[DIPPR_PERRY_8E][0], obj.T_limits[DIPPR_PERRY_8E][1], 10)
    props_calc = [obj.calculate(T, DIPPR_PERRY_8E) for T in Ts]
    res, stats = obj.fit_data_to_model(Ts=Ts, data=props_calc, model='DIPPR101',
                          do_statistics=True, use_numba=False, fit_method='lm')
    assert stats['MAE'] < 1e-6

@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_ViscosityLiquid_fitting2():
    # yaws points
    A, B, C, D = -25.532, 3747.2, 0.0466, 0.0
    Ts = linspace(489.2, 619.65)
    props = [mu_Yaws(T, A, B, C, D) for T in Ts]
    res, stats = ViscosityLiquid.fit_data_to_model(Ts=Ts, data=props, model='mu_Yaws',
                          do_statistics=True, use_numba=False,
                          fit_method='lm')
    assert stats['MAE'] < 1e-5

@pytest.mark.meta_T_dept
def test_ViscosityLiquid_PPDS9_limits():
    assert ViscosityLiquid(CASRN='7553-56-2').T_limits[VDI_PPDS][1] < 793.019

    # Case where the singularity is at 164.8
    low, high = ViscosityLiquid(CASRN='7440-63-3', Tm=161.4, Tc=289.733).T_limits[VDI_PPDS]
    assert high == 289.733
    assert low > 164.8
    assert low < 174.8

    # Case where missing critical temperature
    obj = ViscosityLiquid(CASRN='10097-32-2', Tm=265.925)
    low, high = obj.T_limits[VDI_PPDS]
    assert low == 265.925
    assert high > 400
    assert high < obj.PPDS9_parameters['VDI_PPDS']['C']
    
    # Case that gets to take the two limits
    obj = ViscosityLiquid(CASRN='7647-01-0', Tm=203.55, Tc=324.6)
    low, high = obj.T_limits[VDI_PPDS]
    assert low, high == (203.55, 324.6)

    # Case where Tc is past negative slope
    obj = ViscosityLiquid(CASRN='7664-41-7', Tm=195.45, Tc=405.6)
    low, high = obj.T_limits[VDI_PPDS]
    assert low == obj.Tm
    assert high <373.726

    # Case where we need a solver to find min T.
    obj = ViscosityLiquid(CASRN='75-15-0', Tm=161.15, Tc=552.0)
    low, high = obj.T_limits[VDI_PPDS]
    assert_close(high, 370.9503355938449)
    assert low == obj.Tm

    # TODO: figure out an alcorithm which does not chop off
    # the coefficients and shrink the range in this case
#    obj = ViscosityLiquid(CASRN='75-71-8', Tm=115.15, Tc=385.0)
#    low, high = obj.T_limits[VDI_PPDS]

@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_ViscosityLiquid_fitting_glycerol_case():
    kwargs = {'Ts': [295.15, 300.15, 305.15, 310.15, 315.15, 320.15, 295.65, 293.15, 293.15, 293.15, 
                    353.15, 423.15, 293.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15, 363.15, 
                    273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15, 363.15, 
                    373.15, 243.14999999999998, 253.14999999999998, 263.15, 273.15, 283.15, 293.15, 
                    303.15, 313.15, 323.15, 333.15, 353.15, 363.15, 373.15, 383.15, 393.15, 403.15, 
                    413.15, 423.15, 431.15, 440.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15, 
                    363.15, 288.15, 303.15, 293.15, 295.65, 298.15, 300.65, 303.15, 263.15, 273.34999999999997, 
                    283.34999999999997, 273.15, 279.15, 288.15, 293.15, 298.15, 303.15, 298.15, 348.15],
            'data': [1.198123, 0.777632, 0.542983, 0.38614, 0.266536, 0.17119700000000002, 1.1300000000000001, 
                    1.49, 1.49, 1.12, 0.0274, 0.00423, 1.41, 0.612, 0.28400000000000003, 0.14200000000000002, 
                    0.0813, 0.0506, 0.0319, 0.0213, 12.07, 3.9, 1.412, 0.612, 0.28400000000000003, 
                    0.14200000000000002, 0.0813, 0.0506, 0.0319, 0.0213, 0.0148, 933.0, 171.0, 38.6, 10.59, 
                    3.44, 1.34, 0.629, 0.289, 0.14100000000000001, 0.07200000000000001, 0.03218, 0.0212,
                    0.0146, 0.010480000000000001, 0.0078, 0.00597, 0.004730000000000001, 0.00382, 0.00328, 
                    0.00281, 0.46430000000000005, 0.2381, 0.13140000000000002, 0.07740000000000001, 0.049, 
                    0.033299999999999996, 0.02325, 2.267, 0.587, 1.499, 1.186, 0.9450000000000001, 0.764,
                    0.624, 44.4, 10.5, 3.34, 12.038, 6.258, 2.325, 1.456, 0.9540000000000001, 0.629, 0.5826, 0.02558], 
            'model': 'DIPPR101', 'model_kwargs': {}, 'params_points_max': 2, 
            'model_selection': 'min(BIC, AICc, KFold(5))', 'do_statistics': True, 'use_numba': False,
            'multiple_tries': True, 'multiple_tries_max_err': 1e-05, 'fit_method': 'lm'}

    res, stats = ViscosityLiquid.fit_data_to_model(**kwargs)
    assert stats['MAE']  < 0.2




@pytest.mark.CoolProp
@pytest.mark.meta_T_dept
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_ViscosityGas_CoolProp():
    EtOH = ViscosityGas(MW=46.06844, Tc=514.0, Pc=6137000.0, Zc=0.2412, dipole=1.44, Vmg=0.02357, CASRN='64-17-5')
    EtOH.method = COOLPROP
    assert_close(EtOH.T_dependent_property(305), 8.982631881778473e-06)

    # Ethanol compressed
    assert [True, False] == [EtOH.test_method_validity_P(300, P, COOLPROP) for P in (1E3, 1E5)]
    assert_close(EtOH.calculate_P(298.15, 1E3, COOLPROP), 8.77706377246337e-06)

@pytest.mark.meta_T_dept
def test_ViscosityGas():
    EtOH = ViscosityGas(MW=46.06844, Tc=514.0, Pc=6137000.0, Zc=0.2412, dipole=1.44, Vmg=0.02357, CASRN='64-17-5')

    EtOH.method = VDI_PPDS
    assert_close(EtOH.T_dependent_property(305), 9.13079342291875e-06)
    EtOH.method = YOON_THODOS
    assert_close(EtOH.T_dependent_property(305), 7.584977640806432e-06)
    EtOH.method = STIEL_THODOS
    assert_close(EtOH.T_dependent_property(305), 7.699272212598415e-06)
    EtOH.method = GHARAGHEIZI
    assert_close(EtOH.T_dependent_property(305), 8.12922040122556e-06)
    EtOH.method = LUCAS_GAS
    assert_close(EtOH.T_dependent_property(305), 9.008160864895763e-06)
    EtOH.method = VDI_TABULAR
    assert_close(EtOH.T_dependent_property(305), 8.761390047621239e-06)
    EtOH.method = DIPPR_PERRY_8E
    assert_close(EtOH.T_dependent_property(305), 9.129674918795814e-06)


    # Test that methods return None
    EtOH.extrapolation = None
    for i in EtOH.all_methods:
        EtOH.method = i
        assert EtOH.T_dependent_property(6000) is None

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')


    s = EtOH.as_json()
    EtOH2 = ViscosityGas.from_json(s)
    # Do hash checks before interpolation object exists
    assert hash(EtOH) == hash(EtOH2)
    assert EtOH == EtOH2
    assert id(EtOH) != id(EtOH2)


    # Ethanol data, calculated from CoolProp
    Ts = [400, 500, 550]
    Ps = [1E3, 1E4, 1E5]
    TP_data = [[1.18634700291489e-05, 1.4762189560203758e-05, 1.6162732753470533e-05], [1.1862505513959454e-05, 1.4762728590964208e-05, 1.6163602669178767e-05], [1.1853229260926176e-05, 1.4768417536555742e-05, 1.617257402798515e-05]]
    EtOH.add_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_close2d(TP_data, recalc_pts)

    EtOH.tabular_extrapolation_permitted = False
    assert None is EtOH.TP_dependent_property(300, 90000.0)
    EtOH.tabular_extrapolation_permitted = True
    assert_close(EtOH.TP_dependent_property(300, 9E4), 8.940734479925292e-06)

    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')


@pytest.mark.fitting
@pytest.mark.meta_T_dept
def test_ViscosityGas_fitting0():
    # ammonia chemsep
    ammonia_Ts_mug = [194.6, 236.989, 239.82, 279.379, 321.768, 364.158, 406.547, 448.937, 491.326, 533.716, 576.105, 618.495, 660.884, 703.274, 745.663, 788.053, 830.442, 872.831, 915.221, 957.61, 1000]
    ammonia_mugs = [6.45166E-06, 7.95475E-06, 8.05584E-06, 9.47483E-06, 1.10043E-05, 1.25388E-05, 1.40761E-05, 1.56145E-05, 1.7153E-05, 1.86911E-05, 2.02283E-05, 2.17643E-05, 2.3299E-05, 2.48323E-05, 2.6364E-05, 2.78941E-05, 2.94227E-05, 3.09497E-05, 3.24751E-05, 3.39989E-05, 3.55212E-05]

    fit, res = ViscosityGas.fit_data_to_model(Ts=ammonia_Ts_mug, data=ammonia_mugs, model='DIPPR102',
                          do_statistics=True, use_numba=False,
                          fit_method='lm')
    fit, res
    assert res['MAE'] < 1e-5



def test_ViscosityLiquidMixture():
    # DIPPR  1983 manual example
    ViscosityLiquids = [ViscosityLiquid(CASRN=CAS) for CAS in ['56-23-5', '67-63-0']]
    for obj in ViscosityLiquids:
        obj.method = DIPPR_PERRY_8E

    T, P, zs, ws = 313.2, 101325.0, [0.5, 0.5], [0.7190741374767832, 0.2809258625232169]
    obj = ViscosityLiquidMixture(ViscosityLiquids=ViscosityLiquids, CASs=['56-23-5', '67-63-0'], MWs=[153.8227, 60.09502])
    mu = obj.mixture_property(T, P, zs, ws)
    assert_close(mu, 0.0009948528627794172)

    mu = obj.calculate(T, P, zs, ws, MIXING_LOG_MOLAR)
    assert_close(mu, 0.0009948528627794172)

    mu = obj.calculate(T, P, zs, ws, LINEAR)
    assert_close(mu, 0.001039155803329608)

    hash0 = hash(obj)
    s = json.dumps(obj.as_json())
    assert 'json_version' in s
    obj2 = ViscosityLiquidMixture.from_json(json.loads(s))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

    obj2 = eval(str(obj))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0


def test_ViscosityLiquidMixture_electrolyte():
    # Test Laliberte
    m = Mixture(['water', 'sulfuric acid'], zs=[0.5, 0.5], T=298.15)
    ViscosityLiquids = [i.ViscosityLiquid for i in m.Chemicals]
    obj = ViscosityLiquidMixture(ViscosityLiquids=ViscosityLiquids, CASs=m.CASs, MWs=m.MWs)
    mu = obj.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_close(mu, 0.024955325569420893)
    assert obj.method == LALIBERTE_MU

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


def test_ViscosityGasMixture():
    # DIPPR  1983 manual example, ['dimethyl ether', 'sulfur dioxide']
    T = 308.2
    P = 101325.0
    zs = [.95, .05]
    MWs = [46.06844, 64.0638]
    ws = zs_to_ws(zs, MWs)
    molecular_diameters = [4.594142622412462, 3.98053]
    Stockmayers = [330.53265973799995, 333.4]
    CASs = ['115-10-6', '7446-09-5']

    VolumeGases = [VolumeGas(CASRN="115-10-6", MW=46.06844, Tc=400.2, Pc=5340000.0, omega=0.2033, dipole=1.3, eos=[PR(Tc=400.2, Pc=5340000.0, omega=0.2033, T=308.2, P=101325.0)], extrapolation=None, method=None, method_P="EOS"),
                   VolumeGas(CASRN="7446-09-5", MW=64.0638, Tc=430.8, Pc=7884098.25, omega=0.251, dipole=1.63, eos=[PR(Tc=430.8, Pc=7884098.25, omega=0.251, T=308.2, P=101325.0)], extrapolation=None, method=None, method_P="EOS")]

    ViscosityGases = [ViscosityGas(CASRN="115-10-6", MW=46.06844, Tc=400.2, Pc=5340000.0, Zc=0.26961203187388905, dipole=1.3, Vmg=VolumeGases[0], extrapolation="linear", method=DIPPR_PERRY_8E, method_P=None),
                      ViscosityGas(CASRN="7446-09-5", MW=64.0638, Tc=430.8, Pc=7884098.25, Zc=0.2685356680541311, dipole=1.63, Vmg=VolumeGases[1], extrapolation="linear", method=DIPPR_PERRY_8E, method_P=None)]

    obj = ViscosityGasMixture(MWs=MWs, molecular_diameters=molecular_diameters, Stockmayers=Stockmayers, CASs=CASs,
                              ViscosityGases=ViscosityGases, correct_pressure_pure=False)

    assert_close(obj.calculate(T, P, zs, ws, BROKAW), 9.758786340336624e-06, rtol=1e-10)
    assert_close(obj.calculate(T, P, zs, ws, HERNING_ZIPPERER), 9.79117166372455e-06, rtol=1e-10)
    assert_close(obj.calculate(T, P, zs, ws, WILKE), 9.764037930004713e-06, rtol=1e-10)
    assert_close(obj.calculate(T, P, zs, ws, LINEAR), 9.759079102058173e-06, rtol=1e-10)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    # json
    hash0 = hash(obj)
    obj2 = ViscosityGasMixture.from_json(json.loads(json.dumps(obj.as_json())))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0

    obj2 = eval(str(obj))
    assert obj == obj2
    assert hash(obj) == hash0
    assert hash(obj2) == hash0


def test_ViscosityLiquid_exp_stable_polynomial_generic():
    Tmin, Tmax = (286.40000000000003, 616.168)
    coeffs = [-665348.6775513209, -382952.99175865226, 4698402.139640011, 2610816.157608245, -14904063.263016937, -7964469.576701106, 28050013.908699483, 14348029.233549928, -34845665.63942717, -16966581.681701113, 30076103.297519933, 13845597.87282422, -18489424.72634786, -7980392.802563844, 8164567.967231155, 3269394.9397244365, -2579012.5275077187, -945177.7114472535, 573377.8324670881, 188870.04313096567, -87044.1524863814, -25131.29955632268, 8585.597421800152, 2093.7678534217353, -507.67980909409175, -98.46408927141526, 15.979546303522653, 2.099783548884737, -0.1934071967398076, -1.1612323883439841, -8.699063371522966]
    obj_polynomial = ViscosityLiquid(exp_stable_polynomial_parameters={'test': {'coeffs': coeffs,
                                                                    'Tmin': Tmin, 'Tmax': Tmax}})
    assert_close(obj_polynomial.calculate(300.0, 'test'), 0.0005929213664710905)


def test_ViscosityLiquidMixture_initialization_str():
    pure_obj = ViscosityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-3.1840315590772447e-24, 5.632245762287636e-21, -4.211563759618865e-18, 1.7309219264976467e-15, -4.25623447818058e-13, 6.379502491722484e-11, -5.653736202867734e-09, 2.934273667761606e-07, -4.688742520151596e-06]))
    obj = ViscosityGasMixture(MWs=[44.09562], molecular_diameters=[], Stockmayers=[], CASs=[], correct_pressure_pure=False, method="HERNING_ZIPPERER", ViscosityGases=[pure_obj])
    copy = eval(str(obj))

    assert copy == obj

def test_ViscosityLiquid_str_repr_extrapolation_min_max():
    a = ViscosityLiquid(extrapolation_min=0, exp_poly_fit=(190.0, 391.9, [1.8379049563136273e-17, -4.5666126233131545e-14, 4.9414486397781785e-11, -3.042378423089263e-08, 1.166244931040138e-05, -0.0028523723735774113, 0.4352378275340892, -37.99358630363772, 1456.8338572042996]))
    s = str(a)
    assert 'extrapolation_min=0' in s
    b = eval(str(a))
    assert a == b
    assert hash(a) == hash(b)

    a = ViscosityLiquid(extrapolation_max=100, exp_poly_fit=(190.0, 391.9, [1.8379049563136273e-17, -4.5666126233131545e-14, 4.9414486397781785e-11, -3.042378423089263e-08, 1.166244931040138e-05, -0.0028523723735774113, 0.4352378275340892, -37.99358630363772, 1456.8338572042996]))
    s = str(a)
    assert 'extrapolation_max=100' in s
    b = eval(str(a))
    assert a == b
    assert hash(a) == hash(b)

    a = ViscosityLiquid(extrapolation_min=0, extrapolation_max=100, exp_poly_fit=(190.0, 391.9, [1.8379049563136273e-17, -4.5666126233131545e-14, 4.9414486397781785e-11, -3.042378423089263e-08, 1.166244931040138e-05, -0.0028523723735774113, 0.4352378275340892, -37.99358630363772, 1456.8338572042996]))
    s = str(a)
    assert 'extrapolation_min=0' in s
    assert 'extrapolation_max=100' in s
    b = eval(str(a))
    assert a == b
    assert hash(a) == hash(b)
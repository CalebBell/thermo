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
import pandas as pd
from thermo.critical import *

def test_data_IUPAC():
    MW_sum = _crit_IUPAC['MW'].sum()
    assert_allclose(MW_sum,122998.43799999992)

    Tc_sum = _crit_IUPAC['Tc'].sum()
    assert_allclose(Tc_sum, 462157.51300000004)

    Pc_sum = _crit_IUPAC['Pc'].sum()
    assert_allclose(Pc_sum, 2063753000.0)

    Vc_sum = _crit_IUPAC['Vc'].sum()
    assert_allclose(Vc_sum, 0.19953190000000001)

    Zc_sum = _crit_IUPAC['Zc'].sum()
    assert_allclose(Zc_sum, 109.892)

    assert _crit_IUPAC.shape == (810, 7)
    assert _crit_IUPAC.index.is_unique


def test_data_Matthews():
    MW_sum = _crit_Matthews['MW'].sum()
    assert_allclose(MW_sum, 19541.760399999999)

    Tc_sum = _crit_Matthews['Tc'].sum()
    assert_allclose(Tc_sum, 65343.210900000005)

    Pc_sum = _crit_Matthews['Pc'].sum()
    assert_allclose(Pc_sum, 579365204.25)

    Vc_sum = _crit_Matthews['Vc'].sum()
    assert_allclose(Vc_sum, 0.014921000000000002)

    Zc_sum = _crit_Matthews['Zc'].sum()
    assert_allclose(Zc_sum, 12.141000000000002)

    assert _crit_Matthews.shape == (120, 6)
    assert _crit_Matthews.index.is_unique


def test_data_CRC():
    Tc_sum = _crit_CRC['Tc'].sum()
    assert_allclose(Tc_sum, 514092.75)

    Pc_sum = _crit_CRC['Pc'].sum()
    assert_allclose(Pc_sum, 2700259000.0)

    Vc_sum = _crit_CRC['Vc'].sum()
    assert_allclose(Vc_sum, 0.38876929999999998)

    Zc_sum = _crit_CRC['Zc'].sum()
    assert_allclose(Zc_sum, 207.98663028416496, 1e-6)

    assert _crit_CRC.shape == (861, 8)
    assert _crit_CRC.index.is_unique

    Tc_error_sum = _crit_CRC['Tc_error'].sum()
    assert_allclose(Tc_error_sum, 2444.24)

    Pc_error_sum = _crit_CRC['Pc_error'].sum()
    assert_allclose(Pc_error_sum, 1.2587e+08)

    Vc_error_sum = _crit_CRC['Vc_error'].sum()
    assert_allclose(Vc_error_sum, 0.014151)


def test_data_PSRKR4():
    Tc_sum = _crit_PSRKR4['Tc'].sum()
    assert_allclose(Tc_sum, 597984.0)

    Pc_sum = _crit_PSRKR4['Pc'].sum()
    assert_allclose(Pc_sum, 3708990509)

    Vc_sum = _crit_PSRKR4['Vc'].sum()
    assert_allclose(Vc_sum, 0.40726849999999998)

    Zc_sum = _crit_PSRKR4['Zc'].sum()
    assert_allclose(Zc_sum, 251.29839643655527, 1e-6)

    omega_sum = _crit_PSRKR4['omega'].sum()
    assert_allclose(omega_sum, 410.50560000000002)

    assert _crit_PSRKR4.shape == (995, 6)
    assert _crit_PSRKR4.index.is_unique


def test_data_PassutDanner():
    Tc_sum = _crit_PassutDanner['Tc'].sum()
    assert_allclose(Tc_sum, 111665.28333333334)

    Pc_sum = _crit_PassutDanner['Pc'].sum()
    assert_allclose(Pc_sum, 579756767.55527318)

    omega_sum = _crit_PassutDanner['omega'].sum()
    assert_allclose(omega_sum, 65.567000000000007)

    assert _crit_PassutDanner.shape == (192, 4)
    assert _crit_PassutDanner.index.is_unique


def test_data_Yaws():
    Tc_sum = _crit_Yaws['Tc'].sum()
    assert_allclose(Tc_sum, 5862006.9500000002)

    Pc_sum = _crit_Yaws['Pc'].sum()
    assert_allclose(Pc_sum, 62251189000.0)

    Vc_sum = _crit_Yaws['Vc'].sum()
    assert_allclose(Vc_sum, 4.65511199)

    Zc_sum = _crit_Yaws['Zc'].sum()
    assert_allclose(Zc_sum, 1859.6389176846883, 1e-6)

    omega_sum = _crit_Yaws['omega'].sum()
    assert_allclose(omega_sum, 3170.3041999999996)

    assert _crit_Yaws.shape == (7549, 6)
    assert _crit_Yaws.index.is_unique


def test_relationships():
    Vc = Ihmels(Tc=599.4, Pc=1.19E6)
    Pc = Ihmels(Tc=599.4, Vc=0.00109273333333)
    Tc = Ihmels(Vc=0.00109273333333, Pc=1.19E6)
    assert_allclose([Tc, Pc, Vc], [599.3999999981714, 1190000.0000037064, 0.0010927333333333334])
    with pytest.raises(Exception):
        Ihmels(Tc=559.4)

    Vc = Meissner(Tc=599.4, Pc=1.19E6)
    Pc = Meissner(Tc=599.4, Vc=0.0010695726588235296)
    Tc = Meissner(Vc=0.0010695726588235296, Pc=1.19E6)
    assert_allclose([Tc, Pc, Vc], [599.4000000000001, 1190000.0, 0.0010695726588235296])
    with pytest.raises(Exception):
        Meissner(Tc=559.4)

    Vc = Grigoras(Tc=599.4, Pc=1.19E6)
    Pc = Grigoras(Tc=599.4, Vc=0.00134532)
    Tc = Grigoras(Vc=0.00134532, Pc=1.19E6)
    assert_allclose([Tc, Pc, Vc], [599.4, 1190000.0, 0.00134532])
    with pytest.raises(Exception):
        Grigoras(Tc=559.4)

    Vc1 = critical_surface(Tc=599.4, Pc=1.19E6, Method='IHMELS')
    Vc2 = critical_surface(Tc=599.4, Pc=1.19E6, Method='MEISSNER')
    Vc3 = critical_surface(Tc=599.4, Pc=1.19E6, Method='GRIGORAS')
    assert_allclose([Vc1, Vc2, Vc3],
                    [0.0010927333333333334, 0.0010695726588235296, 0.00134532])

    methods = critical_surface(Tc=599.4, Pc=1.19E6, AvailableMethods=True)
    methods.sort()
    methods_listed = ['IHMELS', 'MEISSNER', 'GRIGORAS', 'NONE']
    methods_listed.sort()
    assert methods == methods_listed

    assert None == critical_surface()
    with pytest.raises(Exception):
        critical_surface(Tc=599.4, Pc=1.19E6, Method='FAIL')

@pytest.mark.slow
def test_Tc_main():
    sources = [_crit_IUPAC, _crit_Matthews, _crit_CRC, _crit_PSRKR4, _crit_PassutDanner, _crit_Yaws]
    CASs = set()
    [CASs.update(set(k.index.values)) for k in sources]

    # Use the default method for each chemical in this file
    Tcs = [Tc(i) for i in CASs]
    Tcs_default_sum = pd.Series(Tcs).sum()
    assert_allclose(Tcs_default_sum, 6054504.896122223)

    assert_allclose(514.0, Tc(CASRN='64-17-5'))

    assert_allclose(647.3, Tc(CASRN='7732-18-5', Method='PSRK'))

    assert_allclose(126.2, Tc(CASRN='7727-37-9', Method='MATTHEWS'))

    assert_allclose(467.3661399548533, Tc(CASRN='64-17-5', Method='SURF'))

    methods = Tc(CASRN='98-01-1', AvailableMethods=True)
    assert methods == ['IUPAC', 'PSRK', 'YAWS', 'NONE']

    # Error handling
    assert None == Tc(CASRN='BADCAS')
    # TODO: Only list critical surface as a method if it can be calculated!
    with pytest.raises(Exception):
        Tc(CASRN='98-01-1', Method='BADMETHOD')

@pytest.mark.slow
def test_Pc_main():
    sources = [_crit_IUPAC, _crit_Matthews, _crit_CRC, _crit_PSRKR4, _crit_PassutDanner, _crit_Yaws]
    CASs = set()
    [CASs.update(set(k.index.values)) for k in sources]

    # Use the default method for each chemical in this file
    Pcs = [Pc(i) for i in CASs]
    Pcs_default_sum = pd.Series(Pcs).sum()
    assert_allclose(Pcs_default_sum, 63159160396.183258)

    assert_allclose(6137000.0, Pc(CASRN='64-17-5'))

    assert_allclose(22048321.0, Pc(CASRN='7732-18-5', Method='PSRK'))

    assert_allclose(3394387.5, Pc(CASRN='7727-37-9', Method='MATTHEWS'))

    assert_allclose(6751845.238095238, Pc(CASRN='64-17-5', Method='SURF'))

    methods = Pc(CASRN='98-01-1', AvailableMethods=True)
    assert methods == ['IUPAC', 'PSRK', 'YAWS', 'NONE']

    # Error handling
    assert None == Pc(CASRN='BADCAS')
    # TODO: Only list critical surface as a method if it can be calculated!
    with pytest.raises(Exception):
        Pc(CASRN='98-01-1', Method='BADMETHOD')

@pytest.mark.slow
def test_Vc_main():
    sources = [_crit_IUPAC, _crit_Matthews, _crit_CRC, _crit_PSRKR4, _crit_Yaws]
    CASs = set()
    [CASs.update(set(k.index.values)) for k in sources]

    # Use the default method for each chemical in this file
    Vcs = [Vc(i) for i in CASs]
    Vcs_default_sum = pd.Series(Vcs).sum()
    assert_allclose(Vcs_default_sum, 4.7958940200000004)

    assert_allclose(0.000168, Vc(CASRN='64-17-5'))

    assert_allclose(5.600e-05, Vc(CASRN='7732-18-5', Method='PSRK'))

    assert_allclose(8.950e-05, Vc(CASRN='7727-37-9', Method='MATTHEWS'))

    assert_allclose(0.00018476306394027916, Vc(CASRN='64-17-5', Method='SURF'))

    methods = Vc(CASRN='98-01-1', AvailableMethods=True)
    assert methods == ['PSRK', 'YAWS', 'NONE']

    # Error handling
    assert None == Vc(CASRN='BADCAS')
    # TODO: Only list critical surface as a method if it can be calculated!
    with pytest.raises(Exception):
        Vc(CASRN='98-01-1', Method='BADMETHOD')

@pytest.mark.slow
def test_Zc_main():
    sources = [_crit_IUPAC, _crit_Matthews, _crit_CRC, _crit_PSRKR4, _crit_Yaws]
    CASs = set()
    [CASs.update(set(k.index.values)) for k in sources]

    # Use the default method for each chemical in this file
    Zcs = [Zc(i) for i in CASs]
    Zcs_default_sum = pd.Series(Zcs).sum()
    assert_allclose(Zcs_default_sum, 1930.7388004412558, 1e-6)

    assert_allclose(0.241, Zc(CASRN='64-17-5'))

    assert_allclose(0.22941610667800444, Zc(CASRN='7732-18-5', Method='PSRK'))

    assert_allclose(0.29, Zc(CASRN='7727-37-9', Method='MATTHEWS'))

    assert_allclose(0.24125051446879994, Zc(CASRN='64-17-5', Method='COMBINED'))

    methods = Zc(CASRN='98-01-1', AvailableMethods=True)
    assert methods == ['PSRK', 'YAWS', 'NONE']

    # Error handling
    assert None == Zc(CASRN='BADCAS')
    with pytest.raises(Exception):
        Zc(CASRN='98-01-1', Method='BADMETHOD')


def test_mixing_Tc():
    # Nitrogen-Argon 50/50 mixture
    Tcm =  Li([0.5, 0.5], [126.2, 150.8], [8.95e-05, 7.49e-05])
    assert_allclose(Tcm, 137.40766423357667)

    # example is from [2]_, for:
    # butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.
    # Its result is identical to that calculated in the article.
    Tcm = Li([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6], [0.000255, 0.000313, 0.000371])
    assert_allclose(Tcm, 449.68261498555444)

    with pytest.raises(Exception):
        Li([0.2359, 0.1192], [425.12, 469.7, 507.6], [0.000255, 0.000313, 0.000371])

    # First example is for an acetone/n-butane 50/50 mixture. No point is
    # available to compare the calculated value with, but it is believed
    # correct.
    Tcm = Chueh_Prausnitz_Tc([0.5, 0.5], [508.1, 425.12], [0.000213, 0.000255], [[0, -14.2619], [-14.2619, 0]])
    assert_allclose(Tcm, 457.01862919555555)

    # 2rd example is from [2]_, for:
    # butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.
    # Its result is identical to that calculated in the article.
    Tcm = Chueh_Prausnitz_Tc([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6], [0.000255, 0.000313, 0.000371], [[0, 1.92681, 6.80358], [1.92681, 0, 1.89312], [ 6.80358, 1.89312, 0]])
    assert_allclose(Tcm, 450.1225764723492)
    # 3rd example is from [2]_, for ethylene, Benzene, ethylbenzene. This is
    # similar to but not identical to the result from the article. The
    # experimental point is 486.9 K.
    Tcm = Chueh_Prausnitz_Tc([0.5, 0.447, .053], [282.34, 562.05, 617.15], [0.0001311, 0.000256, 0.000374], [[0, 37.9570, 0], [37.9570, 0, 4.2459], [0, 4.2459, 0]])
    assert_allclose(Tcm, 475.3154572323848)

    with pytest.raises(Exception):
        Chueh_Prausnitz_Tc([0.447, .053], [282.34, 562.05, 617.15], [0.0001311, 0.000256, 0.000374], [[0, 37.9570, 0], [37.9570, 0, 4.2459], [0, 4.2459, 0]])



    # First example is for an acetone/n-butane 50/50 mixture. No point is
    # available to compare the calculated value with, but it is believed
    # correct. Parameters here are from [2]_.
    Tcm = Grieves_Thodos([0.5, 0.5], [508.1, 425.12], [[0, 0.7137], [1.6496, 0]])
    assert_allclose(Tcm, 456.9398283342622)

    # 2rd example is from [2]_, for:
    # butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.
    # Its result is identical to that calculated in the article.
    Tcm = Grieves_Thodos([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6], [[0, 1.2503, 1.516], [0.799807, 0, 1.23843], [0.659633, 0.807474, 0]])
    assert_allclose(Tcm, 450.1839618758971)

    with pytest.raises(Exception):
        Grieves_Thodos([0.5, 0.5], [508.1, None], [[0, 0.7137], [1.6496, 0]])


    # First example is Acetone/butane 50/50 mixture.
    Tcm = modified_Wilson_Tc([0.5, 0.5], [508.1, 425.12],  [[0, 0.8359], [0, 1.1963]])
    assert_allclose(Tcm, 456.59176287162256)

    # 2rd example is from [2]_, for:
    # butane/pentane/hexane 0.6449/0.2359/0.1192 mixture, exp: 450.22 K.
    # Its result is identical to that calculated in the article.
    Tcm = modified_Wilson_Tc([0.6449, 0.2359, 0.1192], [425.12, 469.7, 507.6], [[0, 1.174450, 1.274390], [0.835914, 0, 1.21038], [0.746878, 0.80677, 0]])
    assert_allclose(Tcm, 450.0305966823031)

    with pytest.raises(Exception):
        Tcm = modified_Wilson_Tc([0.5, 0.5], [508.1, None],  [[0, 0.8359], [0, 1.1963]])


def test_mixing_Chueh_Prausnitz_Vc():
    Vcm = Chueh_Prausnitz_Vc([0.4271, 0.5729], [0.000273, 0.000256], [[0, 5.61847], [5.61847, 0]])
    assert_allclose(Vcm, 0.00026620503424517445)

    with pytest.raises(Exception):
        Chueh_Prausnitz_Vc([0.4271], [0.000273, 0.000256], [[0, 5.61847], [5.61847, 0]])

def test_mixing_modified_Wilson_Vc():
    Vcm = modified_Wilson_Vc([0.4271, 0.5729], [0.000273, 0.000256], [[0, 0.6671250], [1.3939900, 0]])
    assert_allclose(Vcm, 0.00026643350327068809)

    with pytest.raises(Exception):
        modified_Wilson_Vc([0.4271], [0.000273, 0.000256], [[0, 0.6671250], [1.3939900, 0]])


def test_third_property():
    with pytest.raises(Exception):
        third_property('141-62-8')
    assert third_property('1410-62-8', V=True) is None



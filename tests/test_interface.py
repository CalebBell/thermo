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
from thermo.interface import *
from thermo.interface import VDI_TABULAR
from thermo.identifiers import checkCAS

def test_CSP():
    # p-dichloribenzene at 412.15 K, from DIPPR; value differs due to a slight
    # difference in method.
    sigma1 = Brock_Bird(412.15, 447.3, 685, 3.952E6)
    assert_allclose(sigma1, 0.02208448325192495)

    # Chlorobenzene from Poling, as compared with a % error value at 293 K.
    sigma1 = Brock_Bird(293.15, 404.75, 633.0, 4530000.0)
    assert_allclose(sigma1, 0.032985686413713036)

    # TODO: Find parameters where Brock Bird is negative

    # Chlorobenzene from Poling, as compared with a % error value at 293 K.
    sigma1 = Pitzer(293., 633.0, 4530000.0, 0.249)
    assert_allclose(sigma1, 0.03458453513446387)

    sigma1 = Sastri_Rao(293.15, 404.75, 633.0, 4530000.0)
    sigma2 = Sastri_Rao(293.15, 404.75, 633.0, 4530000.0, chemicaltype='alcohol')
    sigma3 = Sastri_Rao(293.15, 404.75, 633.0, 4530000.0, chemicaltype='acid')
    sigmas = [0.03234567739694441, 0.023255104102733407, 0.02558993464948134]
    assert_allclose([sigma1, sigma2, sigma3], sigmas)

    sigma = Zuo_Stenby(293., 633.0, 4530000.0, 0.249)
    assert_allclose(sigma, 0.03345569011871088)

    # 1-butanol, as compared to value in CRC Handbook of 0.02493.
    sigma = Hakim_Steinberg_Stiel(298.15, 563.0, 4414000.0, 0.59, StielPolar=-0.07872)
    assert_allclose(sigma, 0.021907902575190447)

    sigma_calc = Miqueu(300., 340.1, 0.000199, 0.1687)
    assert_allclose(sigma_calc, 0.003474099603581931)

    
def test_Aleem():
    st = Aleem(T=90, MW=16.04246, Tb=111.6, rhol=458.7, Hvap_Tb=510870., Cpl=2465.)
    assert_allclose(st, 0.01669970221165325)

    # Test the "critical" point (where the correlation predicts sigma=0)
    st_at_Tc = Aleem(T=318.8494929006085, MW=16.04246, Tb=111.6, rhol=458.7, 
                     Hvap_Tb=510870.,  Cpl=2465.)
    assert_allclose(st_at_Tc, 0, atol=1E-12)
    
    
def test_REFPROP():
    sigma = REFPROP(298.15, 647.096, -0.1306, 2.471, 0.2151, 1.233)
    assert_allclose(sigma, 0.07205503890847453)


def test_Somayajulu():
    sigma = Somayajulu(300, 647.126, 232.713514, -140.18645, -4.890098)
    assert_allclose(sigma, 0.07166386387996757)


def test_Jasper():
    sigma = Jasper(298.15, 24, 0.0773)
    assert_allclose(sigma, 0.0220675)


def test_data():
    tot = sum([Mulero_Cachadina_data[i].sum() for i in Mulero_Cachadina_data.columns[1:]])
    assert_allclose(tot, 114350.07371931802)

    assert np.all(Mulero_Cachadina_data.columns == [u'Fluid', u'sigma0', u'n0',
                                                    u'sigma1', u'n1', u'sigma2',
                                                    u'n2', u'Tc', u'Tmin',
                                                    u'Tmax'])
    assert Mulero_Cachadina_data.shape == (115, 10)
    assert Mulero_Cachadina_data.index.is_unique

    tot = sum([Jasper_Lange_data[i].sum() for i in Jasper_Lange_data.columns[1:]])
    assert_allclose(tot, 343153.38953333395)

    assert np.all(Jasper_Lange_data.columns == [u'Name', u'a', u'b', u'Tmin', u'Tmax'])
    assert Jasper_Lange_data.shape == (522, 5)
    assert Jasper_Lange_data.index.is_unique

    tot = sum([Somayajulu_data[i].sum() for i in Somayajulu_data.columns[1:]])
    assert_allclose(tot, 38941.199955999997)

    assert np.all(Somayajulu_data.columns == [u'Chemical', u'Tt', u'Tc', u'A', u'B', u'C'])
    assert Somayajulu_data.shape == (64, 6)
    assert Somayajulu_data.index.is_unique

    tot = sum([Somayajulu_data_2[i].sum() for i in Somayajulu_data_2.columns[1:]])
    assert_allclose(tot, 39471.356771000006)
    assert np.all(Somayajulu_data_2.columns == [u'Chemical', u'Tt', u'Tc', u'A', u'B', u'C'])
    assert Somayajulu_data_2.shape == (64, 6)
    assert Somayajulu_data_2.index.is_unique

def test_VDI_PPDS_11_data():
    '''I believe there are no errors here. 
    '''
    assert all([checkCAS(i) for i in VDI_PPDS_11.index])
    tots_calc = [VDI_PPDS_11[i].abs().sum() for i in [u'A', u'B', u'C', u'D', u'E', u'Tc', u'Tm']]
    tots = [18.495069999999998, 336.69950000000006, 6.5941200000000002, 7.7347200000000003, 6.4262199999999998, 150142.28, 56917.699999999997]
    assert_allclose(tots_calc, tots)
    
    assert VDI_PPDS_11.index.is_unique
    assert VDI_PPDS_11.shape == (272, 8)



@pytest.mark.meta_T_dept
def test_SurfaceTension():
    # Ethanol, test as many methods as possible at once
    EtOH = SurfaceTension(Tb=351.39, Tc=514.0, Pc=6137000.0, Vc=0.000168, Zc=0.24125, omega=0.635, StielPolar=-0.01266, CASRN='64-17-5')
    EtOH.T_dependent_property(305.)
    methods = EtOH.sorted_valid_methods
    methods.remove(VDI_TABULAR)
    sigma_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(305.))[1] for i in methods]
    sigma_exp = [0.021222422444285592, 0.02171156653650729, 0.02171156653650729, 0.021462066798796135, 0.02140008, 0.038055725907414066, 0.03739257387107131, 0.02645171690486362, 0.03905907338532845, 0.03670733205970745]

    assert_allclose(sorted(sigma_calcs), sorted(sigma_exp))
    assert_allclose(EtOH.calculate(305., VDI_TABULAR), 0.021533867879206747, rtol=1E-4)

    # Test that methods return None
    sigma_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(5000))[1] for i in EtOH.sorted_valid_methods]
    assert [None]*11 == sigma_calcs

    EtOH.set_user_methods('VDI_TABULAR', forced=True)
    EtOH.tabular_extrapolation_permitted = False
    assert None == EtOH.T_dependent_property(700.)

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    # Test Aleem
    
    CH4 = SurfaceTension(Tb=111.65, Cpl=2465., Hvap_Tb=510870., MW=16.04246, Vml=3.497e-05)
    assert_allclose(CH4.T_dependent_property(90), 0.016704545538936296)
    
    assert not CH4.test_method_validity(600, 'Aleem')
    assert CH4.test_method_validity(100, 'Aleem')


def test_Winterfeld_Scriven_Davis():
    # The example is from [2]_; all results agree.
    # The original source has not been reviewed.
    # 16.06 mol% n-pentane, 83.94 mol% dichloromethane at 298.15 K.
    # sigmas are 15.47 and 28.77 respectively, rhos 8.61 kmol/m^3 and 15.53 kmol/m^3

    sigma = Winterfeld_Scriven_Davis([0.1606, 0.8394], [0.01547, 0.02877], [8610., 15530.])
    assert_allclose(sigma, 0.024967388450439817)

    with pytest.raises(Exception):
        Winterfeld_Scriven_Davis([0.1606, 0.8394, 0.118], [0.01547, 0.02877], [8610., 15530.])


def test_Diguilio_Teja():
    # Winterfeld_Scriven_Davis example, with Diguilio Teja. Calculated sigmas at Tbs are
    # 0.01424 and 0.02530.
    # Checked and edited 2017-01-30

    sigma = Diguilio_Teja(T=298.15, xs=[0.1606, 0.8394], sigmas_Tb=[0.01424, 0.02530], Tbs=[309.21, 312.95], Tcs=[469.7, 508.0])
    assert_allclose(sigma, 0.025716823875045505)

    with pytest.raises(Exception):
        Diguilio_Teja(T=298.15, xs=[0.1606, 0.8394, 0.118], sigmas_Tb=[0.01424, 0.02530], Tbs=[309.21, 312.95], Tcs=[469.7, 508.0])

    with pytest.raises(Exception):
         Diguilio_Teja(T=501.85, xs=[0.1606, 0.8394], sigmas_Tb=[0.01424, 0.02530], Tbs=[309.21, 312.95], Tcs=[469.7, 508.0])



def test_SurfaceTensionMixture():
    from thermo.chemical import Mixture
    from thermo.interface import SurfaceTensionMixture, DIGUILIOTEJA, SIMPLE, WINTERFELDSCRIVENDAVIS
    m = Mixture(['pentane', 'dichloromethane'], zs=[.1606, .8394], T=298.15)
    SurfaceTensions = [i.SurfaceTension for i in m.Chemicals]
    VolumeLiquids = [i.VolumeLiquid for i in m.Chemicals]
    
    a = SurfaceTensionMixture(MWs=m.MWs, Tbs=m.Tbs, Tcs=m.Tcs, CASs=m.CASs, SurfaceTensions=SurfaceTensions, VolumeLiquids=VolumeLiquids)

    sigma = a.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(sigma, 0.023886472131054423)
    
    sigma = a.calculate(m.T, m.P, m.zs, m.ws, SIMPLE)
    assert_allclose(sigma, 0.025331490604571537)
    
    sigmas = [a.calculate(m.T, m.P, m.zs, m.ws, i) for i in [DIGUILIOTEJA, SIMPLE, WINTERFELDSCRIVENDAVIS]]
    assert_allclose(sigmas, [0.025257338967448677, 0.025331490604571537, 0.023886472131054423])
    
    with pytest.raises(Exception):
        a.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
    with pytest.raises(Exception):
        a.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

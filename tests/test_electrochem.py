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
from thermo.elements import charge_from_formula
from thermo.electrochem import *
from thermo.electrochem import _Laliberte_Density_ParametersDict, _Laliberte_Viscosity_ParametersDict, _Laliberte_Heat_Capacity_ParametersDict
from thermo.identifiers import checkCAS, CAS_from_any, pubchem_db
from math import log10


#@pytest.mark.scipy_019
def test_Laliberte_viscosity_w():
    mu_w = Laliberte_viscosity_w(298)
    assert_allclose(mu_w, 0.0008932264487033279)

def test_Laliberte_viscosity_i():
    d =  _Laliberte_Viscosity_ParametersDict['7647-14-5']
    mu = Laliberte_viscosity_i(273.15+5, 1-0.005810, d["V1"], d["V2"], d["V3"], d["V4"], d["V5"], d["V6"] )
    assert_allclose(mu, 0.0042540255333087936)


def test_Laliberte_viscosity():
    mu_i = Laliberte_viscosity(273.15+5, [0.005810], ['7647-14-5'])
    assert_allclose(mu_i, 0.0015285828581961414)


def test_Laliberte_density_w():
    rho1 = Laliberte_density_w(298.15)
    rho2 = Laliberte_density_w(273.15 + 50)
    assert_allclose([rho1, rho2], [997.0448954179155, 988.0362916114763])


def test_Laliberte_density_i():
    d = _Laliberte_Density_ParametersDict['7647-14-5']
    rho = Laliberte_density_i(273.15+0, 1-0.0037838838, d["C0"], d["C1"], d["C2"], d["C3"], d["C4"])
    assert_allclose(rho, 3761.8917585699983)


def test_Laliberte_density():
    rho = Laliberte_density(273.15, [0.0037838838], ['7647-14-5'])
    assert_allclose(rho, 1002.6250120185854)


def test_Laliberte_heat_capacity_w():
    rhow = Laliberte_heat_capacity_w(273.15+3.56)
    assert_allclose(rhow, 4208.8759205525475, rtol=1E-6)


def test_Laliberte_heat_capacity_i():
    d = _Laliberte_Heat_Capacity_ParametersDict['7647-14-5']
    Cpi = Laliberte_heat_capacity_i(1.5+273.15, 1-0.00398447, d["A1"], d["A2"], d["A3"], d["A4"], d["A5"], d["A6"])
    assert_allclose(Cpi, -2930.7353945880477)


def test_Laliberte_heat_capacity():
    Cp = Laliberte_heat_capacity(273.15+1.5, [0.00398447], ['7647-14-5'])
    assert_allclose(Cp, 4186.566417712068, rtol=1E-6)


def test_cond_pure():
    tots_calc = [Lange_cond_pure[i].sum() for i in ['Conductivity', 'T']]
    tots = [4742961.018575863, 35024.150000000001]
    assert_allclose(tots_calc, tots)

    assert Lange_cond_pure.index.is_unique
    assert Lange_cond_pure.shape == (124, 3)

def test_conductivity():
    tots_calc = list(pd.DataFrame([conductivity(CASRN=CASRN, full_info=True) for CASRN in Lange_cond_pure.index]).sum())
    tots = [4742961.0185758611, 35024.150000000067]
    assert_allclose(tots_calc, tots)


    assert conductivity(CASRN='234-34-44', full_info=False) == None
    with pytest.raises(Exception):
        conductivity(CASRN='7732-18-5', Method='BADMETHOD')

    assert conductivity('7732-18-5', full_info=False) == 4e-06


def test_Marcus_ion_conductivities():
    # Check the CAS numbers are the "canonical" ones
    assert all([CAS_from_any(i) == i for i in Marcus_ion_conductivities.index])

    # Check the charges match up
    for v, CAS in zip(Marcus_ion_conductivities['Charge'], Marcus_ion_conductivities.index):
        assert v == charge_from_formula(pubchem_db.search_CAS(CAS).formula)

    # Even check the formulas work!
    for formula, CAS in zip(Marcus_ion_conductivities['Formula'], Marcus_ion_conductivities.index):
        assert pubchem_db.search_CAS(CAS_from_any(formula)).CASs == CAS
        
    
def test_Magomedovk_thermal_cond():
    assert all([checkCAS(i) for i in Magomedovk_thermal_cond.index])
    assert Magomedovk_thermal_cond.index.is_unique
    assert Magomedovk_thermal_cond.shape == (39, 3)
    tot_calc = Magomedovk_thermal_cond['Ai'].abs().sum()
    tot = 0.10688
    assert_allclose(tot_calc, tot)


def test_thermal_conductivity_Magomedov():
    kl =  thermal_conductivity_Magomedov(293., 1E6, [.25], ['7758-94-3'], k_w=0.59827)
    assert_allclose(kl, 0.548654049375)

    # TODO: reconsider this behavior
    with pytest.raises(Exception):
        thermal_conductivity_Magomedov(293., 1E6, [.25], ['7758-94-3'])



def test_ionic_strength():
    I1 = ionic_strength([0.1393, 0.1393], [1, -1])
    I2 = ionic_strength([0.1393, 0.1393], [2, -3])
    assert_allclose([I1, I2], [0.1393, 0.90545])


def test_Kweq_IAPWS_gas():
    # Tested to give the correct answers for all values in the Kweq_IAPWS check
    Kw_G_calc = [Kweq_IAPWS_gas(T) for T in [300, 600, 800]]
    Kw_G_exp = [8.438044566243019e-162, 1.2831436188429253e-81 ,1.4379721554798815e-61]
    assert_allclose(Kw_G_calc, Kw_G_exp, rtol=1e-10)        


def test_Kweq_IAPWS():
    # All checks in the IAPWS document implemented
    Kws_calc = [-1*log10(Kweq_IAPWS(T, rho)) for T, rho in [(300, 1000), (600, 70), (600, 700), (800, 200), (800, 1200)]]
    Kws_exp = [13.906564500165022, 21.048873829703776, 11.203153057603775, 15.08976501255044, 6.438329619174414]
    assert_allclose(Kws_calc, Kws_exp, rtol=1e-10)        


def test_Kweq_1981():
    # Point from IAPWS formulation, very close despite being different
    pKw = -1*log10(Kweq_1981(600, 700))
    assert_allclose(pKw, 11.274522047458206)
    
    
def test_balance_ions():
    anion_concs = [37561.09, 600.14, 0.3, 2047.49]
    cation_concs = [0.15, 3717.44, 2.61, 364.08, 267.84, 113.34, 18908.04]
    
    anions = ['Cl-', 'HCO3-', 'HS-', 'SO4-2']
    cations = ['Ba+2', 'Ca+2', 'Fe+2', 'K+', 'Mg+2', 'NH4+', 'Na+']
    cations = [pubchem_db.search_name(i) for i in cations]
    anions = [pubchem_db.search_name(i) for i in anions]
    
    
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.01844389123949594, 0.00018264948953265628, 1.6843448929678392e-07, 0.0003957995227824709]
    dominant_cat_zs = [2.0283448144191746e-08, 0.001722453668971278, 8.678922979921716e-07, 0.0001729226579918368, 0.0002046394845036363, 0.00011667568840362263, 0.015272747204245271]
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_allclose(z_water, 0.963487164434, rtol=1E-4)
    
    
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='decrease dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.01844389123949594, 0.00018264948953265628, 1.6843448929678392e-07, 0.0003957995227824709]
    dominant_cat_zs = [2.0283448144191746e-08, 0.001722453668971278, 8.678922979921716e-07, 0.0001729226579918368, 0.0002046394845036363, 0.00011667568840362263, 0.015272747204245271]
    assert_allclose(z_water, 0.963487164434, rtol=1E-4)
    
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019674097453720542, 0.00018264948953265628, 1.6843448929678392e-07, 0.0003957995227824709]
    dominant_cat_zs = [2.0283448144191746e-08, 0.001722453668971278, 8.678922979921716e-07, 0.0001729226579918368, 0.0002046394845036363, 0.00011667568840362263, 0.016502953418469874]
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_allclose(z_water, 0.961026752005, rtol=1E-4)
    
    
    # Proportional
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional insufficient ions increase')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019674097453720542, 0.00018264948953265628, 1.6843448929678392e-07, 0.0003957995227824709]
    dominant_cat_zs = [2.1568463485601134e-08, 0.0018315761107652187, 9.228757953582418e-07, 0.00018387781052887865, 0.00021760399010327137, 0.00012406743208065072, 0.016240320090443242]
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_allclose(z_water, 0.961148895221, rtol=1E-4)
    
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional excess ions decrease')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.018501945479459977, 0.00017176751844061514, 1.583994256997263e-07, 0.0003722184058782681]
    dominant_cat_zs = [2.0283448144191746e-08, 0.001722453668971278, 8.678922979921716e-07, 0.0001729226579918368, 0.0002046394845036363, 0.00011667568840362263, 0.015272747204245271]
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_allclose(z_water, 0.963463583317, rtol=1E-4)
    
    # Proportional anion/cation direct adjustment
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional cation adjustment')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019674097453720542, 0.00018264948953265628, 1.6843448929678392e-07, 0.0003957995227824709]
    dominant_cat_zs = [2.1568463485601134e-08, 0.0018315761107652187, 9.228757953582418e-07, 0.00018387781052887865, 0.00021760399010327137, 0.00012406743208065072, 0.016240320090443242]
    assert_allclose(z_water, 0.961148895221, rtol=1E-4)
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)
    
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional anion adjustment')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.018501945479459977, 0.00017176751844061514, 1.583994256997263e-07, 0.0003722184058782681]
    dominant_cat_zs = [2.0283448144191746e-08, 0.001722453668971278, 8.678922979921716e-07, 0.0001729226579918368, 0.0002046394845036363, 0.00011667568840362263, 0.015272747204245271]
    assert_allclose(z_water, 0.963463583317, rtol=1E-4)
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)
    
    # Make there be too much Na+, back to dominant
    anion_concs = [37561.09, 600.14, 0.3, 2047.49]
    cation_concs = [0.15, 3717.44, 2.61, 364.08, 267.84, 113.34, 78908.04]
    
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019940964959685198, 0.0001851270219252835, 1.7071920361126766e-07, 0.0004011683094195923]
    dominant_cat_zs = [2.055858113218964e-08, 0.0017458177351430204, 8.796647441516932e-07, 0.0001752682516624798, 0.00020741529818351878, 0.00011825832516976795, 0.016726806229517385]
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_allclose(z_water, 0.960498102927, rtol=1E-4)
    
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='decrease dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019940964959685198, 0.0001851270219252835, 1.7071920361126766e-07, 0.0004011683094195923]
    dominant_cat_zs = [2.055858113218964e-08, 0.0017458177351430204, 8.796647441516932e-07, 0.0001752682516624798, 0.00020741529818351878, 0.00011825832516976795, 0.016726806229517385]
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_allclose(z_water, 0.960498102927, rtol=1E-4)
    
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.06781575645277317, 0.0001851270219252835, 1.7071920361126766e-07, 0.0004011683094195923]
    dominant_cat_zs = [2.055858113218964e-08, 0.0017458177351430204, 8.796647441516932e-07, 0.0001752682516624798, 0.00020741529818351878, 0.00011825832516976795, 0.06460159772260538]
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_allclose(z_water, 0.864748519941, rtol=1E-4)
    
    # proportional again
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional insufficient ions increase')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.06555651357021297, 0.0006086105737407381, 5.612444438303e-07, 0.0013188527121718626]
    dominant_cat_zs = [2.055858113218964e-08, 0.0017458177351430204, 8.796647441516932e-07, 0.0001752682516624798, 0.00020741529818351878, 0.00011825832516976795, 0.06460159772260538]
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_allclose(z_water, 0.865666204343, rtol=1E-4)
    
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional excess ions decrease')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019940964959685198, 0.0001851270219252835, 1.7071920361126766e-07, 0.0004011683094195923]
    dominant_cat_zs = [6.253504398746918e-09, 0.0005310424302109641, 2.675762160629857e-07, 5.331305578359546e-05, 6.309153687299677e-05, 3.597178968151982e-05, 0.01965049888057932]
    assert_allclose(z_water, 0.959138377467, rtol=1E-4)
    assert_allclose(an_zs, dominant_an_zs, rtol=1E-4)
    assert_allclose(cat_zs, dominant_cat_zs, rtol=1E-4)



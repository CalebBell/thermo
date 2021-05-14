# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import pytest
import numpy as np
from fluids.numerics import linspace, assert_close, assert_close1d
import pandas as pd
from chemicals.elements import charge_from_formula, nested_formula_parser
from thermo.electrochem import *
from chemicals.identifiers import check_CAS, CAS_from_any, pubchem_db, serialize_formula
from math import log10
from chemicals.iapws import iapws95_Psat, iapws95_rhol_sat, iapws95_rho
from thermo.electrochem import cond_data_Lange, Marcus_ion_conductivities, CRC_ion_conductivities, Magomedovk_thermal_cond, CRC_aqueous_thermodynamics, electrolyte_dissociation_reactions, cond_data_McCleskey, cond_data_Lange, Laliberte_data

from thermo.electrochem import electrolyte_dissociation_reactions as df
from collections import Counter

import thermo
thermo.complete_lazy_loading()

def test_Laliberte_viscosity_w():
    mu_w = Laliberte_viscosity_w(298)
    assert_close(mu_w, 0.0008932264487033279)

def test_Laliberte_viscosity_i():
    mu = Laliberte_viscosity_i(273.15+5, 1-0.005810, 16.221788633396, 1.32293086770011, 1.48485985010431, 0.00746912559657377, 30.7802007540575, 2.05826852322558)
    assert_close(mu, 0.0042540255333087936)

def test_Laliberte_viscosity_mix():
    mu = Laliberte_viscosity_mix(T=278.15, ws=[0.00581, 0.002], v1s=[16.221788633396, 69.5769240055845], v2s=[1.32293086770011, 4.17047793905946], v3s=[1.48485985010431, 3.57817553622189], v4s=[0.00746912559657377, 0.0116677996754397], v5s=[30.7802007540575, 13897.6652650556], v6s=[2.05826852322558, 20.8027689840251])
    assert_close(mu, 0.0015377348091189648, rtol=1e-13)



def test_Laliberte_viscosity():
    mu_i = Laliberte_viscosity(273.15+5, [0.005810], ['7647-14-5'])
    assert_close(mu_i, 0.0015285828581961414)


def test_Laliberte_density_w():
    rho1 = Laliberte_density_w(298.15)
    rho2 = Laliberte_density_w(273.15 + 50)
    assert_close1d([rho1, rho2], [997.0448954179155, 988.0362916114763])


def test_Laliberte_density_i():
    rho = Laliberte_density_i(273.15+0, 1-0.0037838838, -0.00324112223655149, 0.0636354335906616, 1.01371399467365, 0.0145951015210159, 3317.34854426537)
    assert_close(rho, 3761.8917585699983)


def test_Laliberte_density():
    rho = Laliberte_density(273.15, [0.0037838838], ['7647-14-5'])
    assert_close(rho, 1002.6250120185854)

def test_Laliberte_density_mix():
    rho = Laliberte_density_mix(T=278.15, ws=[0.00581, 0.002], c0s=[-0.00324112223655149, 0.967814929691928], c1s=[0.0636354335906616, 5.540434135986], c2s=[1.01371399467365, 1.10374669742622], c3s=[0.0145951015210159, 0.0123340782160061], c4s=[3317.34854426537, 2589.61875022366])
    assert_close(rho, 1005.6947727219127, rtol=1e-13)

def test_Laliberte_heat_capacity_w():
    rhow = Laliberte_heat_capacity_w(273.15+3.56)
    assert_close(rhow, 4208.8759205525475, rtol=1E-6)


def test_Laliberte_heat_capacity_i():
    Cpi = Laliberte_heat_capacity_i(1.5+273.15, 1-0.00398447, -0.0693559668993322, -0.0782134167486952, 3.84798479408635, -11.2762109247072, 8.73187698542672, 1.81245930472755)
    assert_close(Cpi, -2930.7353945880477)


def test_Laliberte_heat_capacity():
    Cp = Laliberte_heat_capacity(273.15+1.5, [0.00398447], ['7647-14-5'])
    assert_close(Cp, 4186.566417712068, rtol=1E-5)

def test_Laliberte_heat_capacity_mix():
    Cp = Laliberte_heat_capacity_mix(T=278.15, ws=[0.00581, 0.002], a1s=[-0.0693559668993322, -0.103713247177424], a2s=[-0.0782134167486952, -0.0647453826944371], a3s=[3.84798479408635, 2.92191453087969], a4s=[-11.2762109247072, -5.48799065938436], a5s=[8.73187698542672, 2.41768600041476], a6s=[1.81245930472755, 1.32062411084408])
    assert_close(Cp, 4154.788562680796, rtol=1e-10)

@pytest.mark.scipy
@pytest.mark.fuzz
def test_Laliberte_heat_capacity_w():
    from scipy.interpolate import interp1d
    _T_array = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]
    _Cp_array = [4294.03, 4256.88, 4233.58, 4219.44, 4204.95, 4195.45, 4189.1, 4184.8, 4181.9, 4180.02, 4178.95, 4178.86, 4178.77, 4179.56, 4180.89, 4182.77, 4185.17, 4188.1, 4191.55, 4195.52, 4200.01, 4205.02, 4210.57, 4216.64, 4223.23, 4230.36, 4238.07, 4246.37, 4255.28, 4264.84, 4275.08, 4286.04]
    Laliberte_heat_capacity_w_interp = interp1d(_T_array, _Cp_array, kind='cubic')
    for T in linspace(_T_array[0], 92.0, 1000):
        assert_close(Laliberte_heat_capacity_w_interp(T),
                     Laliberte_heat_capacity_w(T+273.15),
                     rtol=1e-5)


@pytest.mark.slow
def test_dissociation_reactions():

    # Check there's only one dissociation reaction for each product
    assert len(df['Electrolyte Formula']) == len(set(df['Electrolyte Formula'].values.tolist()))

    # Check the chemicals match up with the database
    for name, CAS, formula in zip(df['Electrolyte name'], df['Electrolyte CAS'], df['Electrolyte Formula']):
        assert CAS_from_any(CAS) == CAS
        assert pubchem_db.search_CAS(CAS).formula == serialize_formula(formula)

    # Check the anions match up with the database
    for formula, CAS, charge in zip(df['Anion formula'], df['Anion CAS'], df['Anion charge']):
        assert CAS_from_any(CAS) == CAS
        assert CAS_from_any(formula) == CAS
        hit = pubchem_db.search_CAS(CAS)
        assert hit.charge == charge
        assert hit.formula == serialize_formula(formula)

    # Check the cations match up with the database
    for formula, CAS, charge in zip(df['Cation formula'], df['Cation CAS'], df['Cation charge']):
        assert CAS_from_any(CAS) == CAS
        assert CAS_from_any(formula) == CAS
        hit = pubchem_db.search_CAS(CAS)
        assert hit.charge == charge
        assert hit.formula == serialize_formula(formula)

    # Check the charges and counts of ions sums to zero
    for an_charge, an_count, cat_charge, cat_count in zip(df['Anion charge'].tolist(), df['Anion count'].tolist(), df['Cation charge'].tolist(), df['Cation count'].tolist()):
    # for index, row in df.iterrows():
    #     an_charge = row['Anion charge']
    #     an_count = row['Anion count']
    #     cat_charge = row['Cation charge']
    #     cat_count = row['Cation count']
        err = an_charge*an_count + cat_charge*cat_count
        assert err == 0

    # Check the reactant counts and product counts sum to be equal and conserve
    # moles
    #for index, row in df.iterrows():
    for elec, cat, cat_count, an, an_count in zip(df['Electrolyte Formula'].tolist(), df['Cation formula'].tolist(),
                                                  df['Cation count'].tolist(), df['Anion formula'].tolist(),
                                                  df['Anion count'].tolist()):
        elec = nested_formula_parser(elec)
        #elec = nested_formula_parser(row['Electrolyte Formula'])
        cat = nested_formula_parser(cat)
        #cat = nested_formula_parser(row['Cation formula'])
        #cat_count = row['Cation count']
        an = nested_formula_parser(an)
        #an = nested_formula_parser(row['Anion formula'])
        #an_count = row['Anion count']
        product_counter = Counter()
        for _ in range(cat_count):
            product_counter.update(cat)
        for _ in range(an_count):
            product_counter.update(an)
        assert dict(product_counter.items()) == elec


def test_cond_pure():
    tots_calc = [cond_data_Lange[i].sum() for i in ['Conductivity', 'T']]
    tots = [4742961.018575863, 35024.150000000001]
    assert_close1d(tots_calc, tots)

    assert cond_data_Lange.index.is_unique
    assert cond_data_Lange.shape == (124, 3)

def test_conductivity():
    tots_calc = list(pd.DataFrame([conductivity(CASRN=CASRN) for CASRN in cond_data_Lange.index]).sum())
    tots = [4742961.0185758611, 35024.150000000067]
    assert_close1d(tots_calc, tots)


    assert conductivity(CASRN='234-34-44') == (None, None)
    with pytest.raises(Exception):
        conductivity(CASRN='7732-18-5', method='BADMETHOD')

    assert conductivity('7732-18-5')[0] == 4e-06


    val, T= conductivity("142-82-5")
    assert T is None
    assert_close(val, 1e-11, rtol=1e-13)


def test_Marcus_ion_conductivities():
    # Check the CAS numbers are the "canonical" ones
    assert all([CAS_from_any(i) == i for i in Marcus_ion_conductivities.index])

    # Check the charges match up
    for v, CAS in zip(Marcus_ion_conductivities['Charge'], Marcus_ion_conductivities.index):
        assert v == charge_from_formula(pubchem_db.search_CAS(CAS).formula)

    # Even check the formulas work!
    for formula, CAS in zip(Marcus_ion_conductivities['Formula'], Marcus_ion_conductivities.index):
        assert pubchem_db.search_CAS(CAS_from_any(formula)).CASs == CAS

@pytest.mark.fuzz
def test_CRC_ion_conductivities():
    # Check CASs match up
    for i in CRC_ion_conductivities.index:
        assert CAS_from_any(i)

    # Check search by formula matches up
    for formula, CAS in zip(CRC_ion_conductivities['Formula'].tolist(), CRC_ion_conductivities.index):
        assert pubchem_db.search_CAS(CAS_from_any(formula)).CASs == CAS
    # Charges weren't stored


def test_CRC_aqueous_thermodynamics():
    assert all([check_CAS(i) for i in CRC_aqueous_thermodynamics.index])

    # Check CASs match up
    assert all([CAS_from_any(i) == i for i in CRC_aqueous_thermodynamics.index])

    # Check search by formula matches up
    for formula, CAS in zip(CRC_aqueous_thermodynamics['Formula'], CRC_aqueous_thermodynamics.index):
        assert pubchem_db.search_CAS(CAS_from_any(formula)).CASs == CAS

    # Check the MWs match up
    for CAS, MW_specified in zip(CRC_aqueous_thermodynamics.index, CRC_aqueous_thermodynamics['MW']):
        c = pubchem_db.search_CAS(CAS)
        assert_close(c.MW, MW_specified, atol=0.05)

    # Checking names is an option too but of 173, only 162 are unique
    # and many of the others have names that seem ambiguous for ions which can
    # have more than one charge

    assert CRC_aqueous_thermodynamics.index.is_unique
    assert CRC_aqueous_thermodynamics.shape == (173, 7)

    Hf_tot = CRC_aqueous_thermodynamics['Hf(aq)'].abs().sum()
    assert_close(Hf_tot, 70592500.0)

    Gf_tot = CRC_aqueous_thermodynamics['Gf(aq)'].abs().sum()
    assert_close(Gf_tot, 80924000.0)

    S_tot = CRC_aqueous_thermodynamics['S(aq)'].abs().sum()
    assert_close(S_tot, 17389.9)

    Cp_tot = CRC_aqueous_thermodynamics['Cp(aq)'].abs().sum()
    assert_close(Cp_tot, 2111.5)


def test_Magomedovk_thermal_cond():
    for i in Magomedovk_thermal_cond.index:
        assert check_CAS(i)
    assert Magomedovk_thermal_cond.index.is_unique
    assert Magomedovk_thermal_cond.shape == (39, 3)
    tot_calc = Magomedovk_thermal_cond['Ai'].abs().sum()
    tot = 0.10688
    assert_close(tot_calc, tot)


def test_thermal_conductivity_Magomedov():
    kl =  thermal_conductivity_Magomedov(293., 1E6, [.25], ['7758-94-3'], k_w=0.59827)
    assert_close(kl, 0.548654049375)

    with pytest.raises(Exception):
        thermal_conductivity_Magomedov(293., 1E6, [.25], ['7758-94-3'])



def test_ionic_strength():
    I1 = ionic_strength([0.1393, 0.1393], [1, -1])
    I2 = ionic_strength([0.1393, 0.1393], [2, -3])
    assert_close1d([I1, I2], [0.1393, 0.90545])


def test_Kweq_IAPWS_gas():
    # Tested to give the correct answers for all values in the Kweq_IAPWS check
    Kw_G_calc = [Kweq_IAPWS_gas(T) for T in [300, 600, 800]]
    Kw_G_exp = [8.438044566243019e-162, 1.2831436188429253e-81 ,1.4379721554798815e-61]
    assert_close1d(Kw_G_calc, Kw_G_exp, rtol=1e-10)


def test_Kweq_IAPWS():
    # All checks in the IAPWS document implemented
    Kws_calc = [-1*log10(Kweq_IAPWS(T, rho)) for T, rho in [(300, 1000), (600, 70), (600, 700), (800, 200), (800, 1200)]]
    Kws_exp = [13.906564500165022, 21.048873829703776, 11.203153057603775, 15.08976501255044, 6.438329619174414]
    assert_close1d(Kws_calc, Kws_exp, rtol=1e-10)


def test_Kweq_1981():
    # Point from IAPWS formulation, very close despite being different
    pKw = -1*log10(Kweq_1981(600, 700))
    assert_close(pKw, 11.274522047458206)

def test_Kweq_Arcis_Tremaine_Bandura_Lvov():
    test_Ts = [273.15, 298.15, 323.15, 348.15, 373.15, 398.15, 423.15, 448.15, 473.15, 498.15, 523.15, 548.15, 573.15, 598.15, 623.15, 648.15, 673.15]
    test_Psats = [iapws95_Psat(T) for T in test_Ts[:-2]]
    test_Ps = [5e6, 10e6, 15e6, 20e6, 25e6, 30e6]
    expect_saturation_Kweqs = [14.945, 13.996, 13.263, 12.687, 12.234, 11.884, 11.621, 11.436, 11.318, 11.262, 11.263, 11.320, 11.434, 11.613, 11.895]
    
    expect_Kweqs = [[14.889, 14.832, 14.775, 14.719, 14.663, 14.608],
    [13.948, 13.899, 13.851, 13.802, 13.754, 13.707],
    [13.219, 13.173, 13.128, 13.083, 13.039, 12.995],
    [12.643, 12.598, 12.554, 12.511, 12.468, 12.425],
    [12.190, 12.145, 12.101, 12.057, 12.013, 11.970],
    [11.839, 11.793, 11.747, 11.702, 11.657, 11.613],
    [11.577, 11.528, 11.480, 11.432, 11.386, 11.340],
    [11.392, 11.339, 11.288, 11.237, 11.187, 11.138],
    [11.277, 11.219, 11.163, 11.108, 11.054, 11.001],
    [11.229, 11.164, 11.101, 11.040, 10.980, 10.922],
    [11.247, 11.171, 11.099, 11.029, 10.961, 10.896],
    [23.534, 11.245, 11.158, 11.075, 10.997, 10.922],
    [23.432, 11.399, 11.287, 11.183, 11.088, 10.999],
    [23.296, 19.208, 11.515, 11.370, 11.244, 11.131],
    [23.150, 19.283, 16.618, 11.698, 11.495, 11.335],
    [23.006, 19.266, 16.920, 14.909, 11.998, 11.659],
    [22.867, 19.210, 17.009, 15.350, 13.883, 12.419]]
    
    
    for i in range(len(test_Psats)):
        # Saturation density is likely not quite as accurate in original paper
        T = test_Ts[i]
        rho_w = iapws95_rhol_sat(T)
        calc = -log10(Kweq_Arcis_Tremaine_Bandura_Lvov(T, rho_w))
        assert_close(calc, expect_saturation_Kweqs[i], atol=.0015)
    
    # These results match exactly
    for i in range(len(test_Ts)):
        T = test_Ts[i]
        for j in range(len(test_Ps)):
            P = test_Ps[j]
            rho_w = iapws95_rho(T, P)
            calc = -log10(Kweq_Arcis_Tremaine_Bandura_Lvov(T, rho_w))
            assert_close(calc, expect_Kweqs[i][j], atol=.0005)




def test_balance_ions():

    def check_charge_balance(an_zs, cat_zs, an_charges, cat_charges):
        an = np.sum(np.array(an_zs)*np.array(an_charges))
        cat = np.sum(np.array(cat_zs)*np.array(cat_charges))
        assert_close(-an, cat)

    Na_ion = pubchem_db.search_formula('Na+')
    Cl_ion = pubchem_db.search_formula('Cl-')


    anion_concs = [37561.09, 600.14, 0.3, 2047.49]
    cation_concs = [0.15, 3717.44, 2.61, 364.08, 267.84, 113.34, 18908.04]

    anions = ['Cl-', 'CHO3-', 'HS-', 'O4S-2']
    cations = ['Ba+2', 'Ca+2', 'Fe+2', 'K+', 'Mg+2', 'H4N+', 'Na+']
    cations = [pubchem_db.search_formula(i) for i in cations]
    anions = [pubchem_db.search_formula(i) for i in anions]


    anion_charges = [i.charge for i in anions]
    cation_charges = [i.charge for i in cations]

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.01844389123949594, 0.00018264948953265628, 1.6843448929678392e-07, 0.0003957995227824709]
    dominant_cat_zs = [2.0283448144191746e-08, 0.001722453668971278, 8.678922979921716e-07, 0.0001729226579918368, 0.0002046394845036363, 0.00011667568840362263, 0.015272747204245271]
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_close(z_water, 0.963487164434, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='decrease dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.01844389123949594, 0.00018264948953265628, 1.6843448929678392e-07, 0.0003957995227824709]
    dominant_cat_zs = [2.0283448144191746e-08, 0.001722453668971278, 8.678922979921716e-07, 0.0001729226579918368, 0.0002046394845036363, 0.00011667568840362263, 0.015272747204245271]
    assert_close(z_water, 0.963487164434, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019674097453720542, 0.00018264948953265628, 1.6843448929678392e-07, 0.0003957995227824709]
    dominant_cat_zs = [2.0283448144191746e-08, 0.001722453668971278, 8.678922979921716e-07, 0.0001729226579918368, 0.0002046394845036363, 0.00011667568840362263, 0.016502953418469874]
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_close(z_water, 0.961026752005, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # Proportional
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional insufficient ions increase')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019674097453720542, 0.00018264948953265628, 1.6843448929678392e-07, 0.0003957995227824709]
    dominant_cat_zs = [2.1568463485601134e-08, 0.0018315761107652187, 9.228757953582418e-07, 0.00018387781052887865, 0.00021760399010327137, 0.00012406743208065072, 0.016240320090443242]
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_close(z_water, 0.961148895221, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional excess ions decrease')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.018501945479459977, 0.00017176751844061514, 1.583994256997263e-07, 0.0003722184058782681]
    dominant_cat_zs = [2.0283448144191746e-08, 0.001722453668971278, 8.678922979921716e-07, 0.0001729226579918368, 0.0002046394845036363, 0.00011667568840362263, 0.015272747204245271]
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_close(z_water, 0.963463583317, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # Proportional anion/cation direct adjustment
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional cation adjustment')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019674097453720542, 0.00018264948953265628, 1.6843448929678392e-07, 0.0003957995227824709]
    dominant_cat_zs = [2.1568463485601134e-08, 0.0018315761107652187, 9.228757953582418e-07, 0.00018387781052887865, 0.00021760399010327137, 0.00012406743208065072, 0.016240320090443242]
    assert_close(z_water, 0.961148895221, rtol=1E-4)
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional anion adjustment')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.018501945479459977, 0.00017176751844061514, 1.583994256997263e-07, 0.0003722184058782681]
    dominant_cat_zs = [2.0283448144191746e-08, 0.001722453668971278, 8.678922979921716e-07, 0.0001729226579918368, 0.0002046394845036363, 0.00011667568840362263, 0.015272747204245271]
    assert_close(z_water, 0.963463583317, rtol=1E-4)
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # Na or Cl Increase
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='Na or Cl increase')
    an_res_2, cat_res_2, an_zs_2, cat_zs_2, z_water_2 = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase dominant')
    assert an_res == an_res_2
    assert cat_res == cat_res_2
    assert_close1d(an_zs, an_zs_2)
    assert_close1d(cat_zs, cat_zs_2)
    assert_close(z_water, z_water_2)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # Na or Cl decrease
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='Na or Cl decrease')
    an_res_2, cat_res_2, an_zs_2, cat_zs_2, z_water_2 = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='decrease dominant')
    assert an_res == an_res_2
    assert cat_res == cat_res_2
    assert_close1d(an_zs, an_zs_2)
    assert_close1d(cat_zs, cat_zs_2)
    assert_close(z_water, z_water_2)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # Adjust
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=Cl_ion, method='adjust')
    an_res_2, cat_res_2, an_zs_2, cat_zs_2, z_water_2 = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='decrease dominant')
    assert an_res == an_res_2
    assert cat_res == cat_res_2
    assert_close1d(an_zs, an_zs_2)
    assert_close1d(cat_zs, cat_zs_2)
    assert_close(z_water, z_water_2)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=Na_ion, method='adjust')
    an_res_2, cat_res_2, an_zs_2, cat_zs_2, z_water_2 = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase dominant')
    assert an_res == an_res_2
    assert cat_res == cat_res_2
    assert_close1d(an_zs, an_zs_2)
    assert_close1d(cat_zs, cat_zs_2)
    assert_close(z_water, z_water_2)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)


    # Increase and decrease

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=Cl_ion, method='decrease')
    an_res_2, cat_res_2, an_zs_2, cat_zs_2, z_water_2 = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='decrease dominant')
    assert an_res == an_res_2
    assert cat_res == cat_res_2
    assert_close1d(an_zs, an_zs_2)
    assert_close1d(cat_zs, cat_zs_2)
    assert_close(z_water, z_water_2)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=Na_ion, method='increase')
    an_res_2, cat_res_2, an_zs_2, cat_zs_2, z_water_2 = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase dominant')
    assert an_res == an_res_2
    assert cat_res == cat_res_2
    assert_close1d(an_zs, an_zs_2)
    assert_close1d(cat_zs, cat_zs_2)
    assert_close(z_water, z_water_2)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)


    # makeup options
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=(Cl_ion, Na_ion), method='makeup')
    an_res_2, cat_res_2, an_zs_2, cat_zs_2, z_water_2 = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase dominant')
    assert an_res == an_res_2
    assert cat_res == cat_res_2
    assert_close1d(an_zs, an_zs_2)
    assert_close1d(cat_zs, cat_zs_2)
    assert_close(z_water, z_water_2)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # A few failure cases
    with pytest.raises(Exception):
        an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=Cl_ion, method='increase')

    with pytest.raises(Exception):
        an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=Na_ion, method='decrease')

    with pytest.raises(Exception):
        HS_ion = pubchem_db.search_formula('HS-')
        an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=HS_ion, method='adjust')

    with pytest.raises(Exception):
        balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='NOTAREALMETHOD dominant')

    with pytest.raises(Exception):
        balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='NOTAMETHOD proportional insufficient ions increase')

    # No ion specified
    with pytest.raises(Exception):
        balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase')

    # Bad method
    with pytest.raises(Exception):
        balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='NOT A METHOD')

    # Make there be too much Na+, back to dominant
    anion_concs = [37561.09, 600.14, 0.3, 2047.49]
    cation_concs = [0.15, 3717.44, 2.61, 364.08, 267.84, 113.34, 78908.04]

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019940964959685198, 0.0001851270219252835, 1.7071920361126766e-07, 0.0004011683094195923]
    dominant_cat_zs = [2.055858113218964e-08, 0.0017458177351430204, 8.796647441516932e-07, 0.0001752682516624798, 0.00020741529818351878, 0.00011825832516976795, 0.016726806229517385]
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_close(z_water, 0.960498102927, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='decrease dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019940964959685198, 0.0001851270219252835, 1.7071920361126766e-07, 0.0004011683094195923]
    dominant_cat_zs = [2.055858113218964e-08, 0.0017458177351430204, 8.796647441516932e-07, 0.0001752682516624798, 0.00020741529818351878, 0.00011825832516976795, 0.016726806229517385]
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_close(z_water, 0.960498102927, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase dominant')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.06781575645277317, 0.0001851270219252835, 1.7071920361126766e-07, 0.0004011683094195923]
    dominant_cat_zs = [2.055858113218964e-08, 0.0017458177351430204, 8.796647441516932e-07, 0.0001752682516624798, 0.00020741529818351878, 0.00011825832516976795, 0.06460159772260538]
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_close(z_water, 0.864748519941, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # proportional again
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional insufficient ions increase')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.06555651357021297, 0.0006086105737407381, 5.612444438303e-07, 0.0013188527121718626]
    dominant_cat_zs = [2.055858113218964e-08, 0.0017458177351430204, 8.796647441516932e-07, 0.0001752682516624798, 0.00020741529818351878, 0.00011825832516976795, 0.06460159772260538]
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_close(z_water, 0.865666204343, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='proportional excess ions decrease')
    assert an_res == anions
    assert cat_res == cations
    dominant_an_zs = [0.019940964959685198, 0.0001851270219252835, 1.7071920361126766e-07, 0.0004011683094195923]
    dominant_cat_zs = [6.253504398746918e-09, 0.0005310424302109641, 2.675762160629857e-07, 5.331305578359546e-05, 6.309153687299677e-05, 3.597178968151982e-05, 0.01965049888057932]
    assert_close(z_water, 0.959138377467, rtol=1E-4)
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # makeup options
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=(Cl_ion, Na_ion), method='makeup')
    an_res_2, cat_res_2, an_zs_2, cat_zs_2, z_water_2 = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase dominant')
    assert an_res == an_res_2
    assert cat_res == cat_res_2
    assert_close1d(an_zs, an_zs_2)
    assert_close1d(cat_zs, cat_zs_2)
    assert_close(z_water, z_water_2)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # Na or Cl Increase
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='Na or Cl increase')
    an_res_2, cat_res_2, an_zs_2, cat_zs_2, z_water_2 = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='increase dominant')
    assert an_res == an_res_2
    assert cat_res == cat_res_2
    assert_close1d(an_zs, an_zs_2)
    assert_close1d(cat_zs, cat_zs_2)
    assert_close(z_water, z_water_2)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # Na or Cl decrease
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='Na or Cl decrease')
    an_res_2, cat_res_2, an_zs_2, cat_zs_2, z_water_2 = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, method='decrease dominant')
    assert an_res == an_res_2
    assert cat_res == cat_res_2
    assert_close1d(an_zs, an_zs_2)
    assert_close1d(cat_zs, cat_zs_2)
    assert_close(z_water, z_water_2)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    # Test a case with adding a Cl not initially present options
    # Note the test cases with adding ions are especially obvious the mole
    # fractions will be different in each case due to the mole/mass fraction
    # conversion
    anion_concs = [600.14, 0.3, 2047.49]
    cation_concs = [0.15, 3717.44, 2.61, 364.08, 267.84, 113.34, 18908.04]

    anions = ['CHO3-', 'HS-', 'O4S-2']
    cations = ['Ba+2', 'Ca+2', 'Fe+2', 'K+', 'Mg+2', 'H4N+', 'Na+']

    cations = [pubchem_db.search_formula(i) for i in cations]
    anions = [pubchem_db.search_formula(i) for i in anions]
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=Cl_ion, method='increase')

    assert an_res == [pubchem_db.search_formula(i) for i in  ['CHO3-', 'HS-', 'O4S-2', 'Cl-', ]]
    assert cat_res == cations
    dominant_an_zs = [0.00017923623007416514, 1.6528687243128162e-07, 0.0003884030254352281, 0.018099221312491646]
    dominant_cat_zs = [1.9904401526508215e-08, 0.001690265343164992, 8.516735743447466e-07, 0.0001696911685445447, 0.00020081528736051808, 0.00011449531331449091, 0.014987337981446901]
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_close(z_water, 0.9641694974733193, rtol=1E-4)


    anion_charges = [i.charge for i in an_res]
    cation_charges = [i.charge for i in cat_res]

    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)


    # Add Na+ to balance it case
    anion_concs = [37561.09, 600.14, 0.3, 2047.49]
    cation_concs = [0.15, 3717.44, 2.61, 364.08, 267.84, 113.34]
    anions = ['Cl-', 'CHO3-', 'HS-', 'O4S-2']
    cations = ['Ba+2', 'Ca+2', 'Fe+2', 'K+', 'Mg+2', 'H4N+']
    cations = [pubchem_db.search_formula(i) for i in cations]
    anions = [pubchem_db.search_formula(i) for i in anions]
    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations, anion_concs=anion_concs, cation_concs=cation_concs, selected_ion=Na_ion, method='increase')

    assert an_res == anions
    assert cat_res == [pubchem_db.search_formula(i) for i in ['Ba+2', 'Ca+2', 'Fe+2', 'K+', 'Mg+2', 'H4N+', 'Na+']]
    anion_charges = [i.charge for i in an_res]
    cation_charges = [i.charge for i in cat_res]
    dominant_an_zs = [0.019591472379087822, 0.00018188241862941595, 1.6772711696208816e-07, 0.0003941372882028963]
    dominant_cat_zs = [2.0198263986663557e-08, 0.0017152199006479827, 8.64247420962186e-07, 0.00017219643674809882, 0.0002037800624783217, 0.00011618568689352288, 0.01643364615997587]
    assert_close1d(an_zs, dominant_an_zs, rtol=1E-4)
    assert_close1d(cat_zs, dominant_cat_zs, rtol=1E-4)
    assert_close(z_water, 0.961190427495, rtol=1E-4)
    check_charge_balance(an_zs, cat_zs, anion_charges, cation_charges)

    an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions=[Na_ion], cations=[Cl_ion], anion_zs=[.1], cation_zs=[.1])
    assert an_res == [Na_ion]
    assert cat_res == [Cl_ion]
    assert_close1d(an_zs, [0.1])
    assert_close1d(an_zs, [0.1])
    assert_close(z_water, 0.8)

    with pytest.raises(Exception):
         balance_ions(anions=[Na_ion], cations=[Cl_ion], anion_zs=[.1])


def test_dilute_ionic_conductivity():
    ionic_conductivities = [0.00764, 0.00445, 0.016, 0.00501, 0.00735, 0.0119, 0.01061]
    zs = [0.03104, 0.00039, 0.00022, 0.02413, 0.0009, 0.0024, 0.00103]
    c = dilute_ionic_conductivity(ionic_conductivities=ionic_conductivities, zs=zs, rhom=53865.9)
    assert_close(c, 22.05246783663)


def test_conductivity_McCleskey():
    cond = conductivity_McCleskey(T=293.15, M=0.045053, A_coeffs=[.03918, 3.905, 137.7], lambda_coeffs=[0.01124, 2.224, 72.36], B=3.8, multiplier=2)
    assert_close(cond, .8482584585108555)

    # CaCl2 max concentration actual point from tablbe
    cond = conductivity_McCleskey(T=298.15, M=0.3773, A_coeffs=[.03918, 3.905, 137.7], lambda_coeffs=[0.01124, 2.224, 72.36], B=3.8, multiplier=2)
    assert_close(cond, 6.5740628852868)

    # 6.531 exp

@pytest.mark.slow
def test_McCleskey_data():
    # Check the CAS lookups
    for CAS in cond_data_McCleskey.index:
        assert pubchem_db.search_CAS(CAS).CASs == CAS

    # Check the formula lookups
    for CAS, formula in zip(cond_data_McCleskey.index, cond_data_McCleskey['formula']):
        assert CAS_from_any(formula) == CAS

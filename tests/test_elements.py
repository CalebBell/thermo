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

from thermo.elements import *
from thermo.elements import periodic_table

def test_molecular_weight():
    MW_calc = molecular_weight({'H': 12, 'C': 20, 'O': 5})
    MW = 332.30628
    assert_allclose(MW_calc, MW)

    MW_calc = molecular_weight({'C': 32, 'Cu': 1, 'H': 12, 'Na': 4, 'S': 4, 'O': 12, 'N': 8})
    MW = 984.24916
    assert_allclose(MW_calc, MW)

    with pytest.raises(Exception):
        molecular_weight({'H': 12, 'C': 20, 'FAIL': 5})


def test_mass_fractions():
    mfs_calc = mass_fractions({'H': 12, 'C': 20, 'O': 5})
    mfs = {'H': 0.03639798802478244, 'C': 0.7228692758981262, 'O': 0.24073273607709128}
    assert_allclose(sorted(mfs.values()), sorted(mfs_calc.values()))

    mfs_calc = mass_fractions({'C': 32, 'Cu': 1, 'H': 12, 'Na': 4, 'S': 4, 'O': 12, 'N': 8})
    mfs = {'C': 0.39049299264832493, 'H': 0.012288839545466314, 'O': 0.19506524140696244, 'N': 0.11384678245496294, 'S': 0.13031253183899086, 'Na': 0.09343069187887344, 'Cu': 0.06456292022641909}
    assert_allclose(sorted(mfs.values()), sorted(mfs_calc.values()))

    # Fail two tests, one without MW, and one with MW
    with pytest.raises(Exception):
        mass_fractions({'FAIL': 12, 'C': 20, 'O': 5})
    with pytest.raises(Exception):
        mass_fractions({'FAIL': 12, 'C': 20, 'O': 5}, MW=120)


def test_atom_fractions():
    fractions_calc = atom_fractions({'H': 12, 'C': 20, 'O': 5})
    fractions = {'H': 0.32432432432432434, 'C': 0.5405405405405406, 'O': 0.13513513513513514}
    assert_allclose(sorted(fractions_calc.values()), sorted(fractions.values()))


def test_similarity_variable():
    sim1 = similarity_variable({'H': 32, 'C': 15})
    sim2 = similarity_variable({'H': 32, 'C': 15}, 212.41458)
    assert_allclose([sim1, sim2], [0.2212654140784498]*2)


def test_elements_data():
    tots_calc = [sum([getattr(i, att) for i in periodic_table if not getattr(i, att) is None]) for att in
    ['number', 'MW', 'period', 'group', 'AReneg', 'rcov', 'rvdw', 'maxbonds', 'elneg', 'ionization', 'elaffinity', 'electrons', 'protons']]
    tots_exp = [7021, 17285.2137652, 620, 895, 109.91, 144.3100000000001, 179.4300000000001, 94, 163.27000000000007, 816.4238999999999, 67.50297235000001, 7021, 7021]
    assert_allclose(tots_calc, tots_exp)

def test_misc_elements():
    assert periodic_table['H'].InChI == 'InChI=1S/H'
    
    assert periodic_table['H'].smiles == '[H]'

def test_Hill_formula():
    Hill_formulas = {'ClNa': {'Na': 1, 'Cl': 1}, 'BrI': {'I': 1, 'Br': 1},
                    'CCl4': {'C': 1, 'Cl': 4}, 'CH3I': {'I': 1, 'H': 3, 'C': 1},
                    'C2H5Br': {'H': 5, 'C': 2, 'Br': 1}, 'H2O4S': {'H': 2, 'S': 1, 'O': 4}}

    for formula, atoms in Hill_formulas.items():
        assert formula == atoms_to_Hill(atoms)


def test_simple_formula_parser():
    assert simple_formula_parser('CO2') == {'C': 1, 'O': 2}
    assert simple_formula_parser('H20OCo2NaClH4P4-132') == {'P': 4, 'Co': 2, 'Cl': 1, 'H': 24, 'Na': 1, 'O': 1}



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
SOFTWARE.
'''

import os

import pytest
from chemicals.identifiers import pubchem_db
from fluids.numerics import assert_close, assert_close1d

from thermo.group_contribution.joback import *
from thermo.group_contribution.joback import JOBACK_GROUPS

folder = os.path.join(os.path.dirname(__file__), 'Data')

try:
    import rdkit
except:
    rdkit = None

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_Joback_acetone():
    from rdkit import Chem
    for i in [Chem.MolFromSmiles('CC(=O)C'), 'CC(=O)C']:
        ex = Joback(i) # Acetone example
        assert_close(ex.Tb(ex.counts), 322.11)
        assert_close(ex.Tm(ex.counts), 173.5)
        assert_close(ex.Tc(ex.counts), 500.5590049525365)
        assert_close(ex.Tc(ex.counts, 322.11), 500.5590049525365)
        assert_close(ex.Pc(ex.counts, ex.atom_count), 4802499.604994407)
        assert_close(ex.Vc(ex.counts), 0.0002095)
        assert_close(ex.Hf(ex.counts), -217830)
        assert_close(ex.Gf(ex.counts), -154540)
        assert_close(ex.Hfus(ex.counts), 5125)
        assert_close(ex.Hvap(ex.counts), 29018)
        assert_close1d(ex.Cpig_coeffs(ex.counts),[7.52, 0.26084, -0.0001207, 1.546e-08] )
        assert_close(ex.Cpig(300.0), 75.32642000000001)
        assert_close1d(ex.mul_coeffs(ex.counts), [839.11, -14.99])
        assert_close(ex.mul(300.0), 0.0002940378347162687)

    with pytest.raises(ValueError):
        # Raise an error if there are no groups matched
        obj = Joback('[Fe]')
        obj.estimate()

    # Test we can handle missing groups
    nitrobenzene = 'C1=CC=C(C=C1)[N+](=O)[O-]'
    obj = Joback(nitrobenzene)
    res = obj.estimate()
    assert res['mul_coeffs'] is None

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_Joback_formic_acid():
    ex = Joback('C(=O)O')
    assert ex.status == 'OK'
    # values obtained from DDBST http://ddbonline.ddbst.com/OnlinePropertyEstimation/OnlinePropertyEstimation.exe
    # needed to update smarts to fix it
    assert_close(ex.Tb(ex.counts), 367.29, rtol=1e-7)
    assert_close(ex.Tm(ex.counts), 278.0, rtol=1e-6)
    assert_close(ex.Tc(ex.counts), 561.541, rtol=1e-6)
    assert_close(ex.Pc(ex.counts, ex.atom_count), 6796390, rtol=1e-6) # Pa from kPa
    assert_close(ex.Vc(ex.counts), 0.0001065, rtol=1e-7) # m³/mol from cm³/mol
    assert_close(ex.Hf(ex.counts), -358430, rtol=1e-7) # J/mol from kJ/mol
    assert_close(ex.Gf(ex.counts), -333990, rtol=1e-7) # J/mol from kJ/mol
    assert_close(ex.Hfus(ex.counts), 10171, rtol=1e-7)
    assert_close(ex.Hvap(ex.counts), 34837, rtol=1e-7) # J/mol from kJ/mol

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_Joback_2_methylphenol_PGL6():
    # has a known Tb
    from thermo import Chemical
    estimator = Joback(Chemical('2-methylphenol').rdkitmol, Tb=464.15)
    assert 'OK' == estimator.status
    estimates = estimator.estimate(callables=False)
    assert_close(estimates['Tc'], 692.639982032995, rtol=1e-13)
    assert_close(estimates['Pc'], 5029928.072028569, rtol=1e-13)
    assert_close(estimates['Vc'], 0.0002855, rtol=1e-13)

@pytest.mark.fuzz
@pytest.mark.slow
@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_Joback_database():
    from thermo.group_contribution.group_contribution_base import smarts_fragment
    from thermo.group_contribution.joback import JOBACK_GROUPS_FOR_FRAGMENTATION
    pubchem_db.autoload_main_db()

    f = open(os.path.join(folder, 'joback_log.txt'), 'w')
    from rdkit import Chem
    lines = []
    for key in sorted(pubchem_db.CAS_index):
        chem_info = pubchem_db.CAS_index[key]
        try:
            mol = Chem.MolFromSmiles(chem_info.smiles)
            parsed = smarts_fragment(rdkitmol=mol, catalog=JOBACK_GROUPS_FOR_FRAGMENTATION, deduplicate=False)
            line = f'{parsed[2]}\t{chem_info.CASs}\t{chem_info.smiles}\t{parsed[0]}\n'
        except Exception as e:
            line = f'{chem_info.CASs}\t{chem_info.smiles}\t{e}\n'
        lines.append(line)

    [f.write(line) for line in sorted(lines)]
    f.close()

# Maybe use this again if more work is done on Joback
del test_Joback_database



@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_DikyJoback_acetone():
    """Test the DikyJoback estimator with acetone, checking coefficients and Cp values"""
    from rdkit import Chem
    for i in [Chem.MolFromSmiles('CC(=O)C'), 'CC(=O)C']:
        ex = DikyJoback(i)  # Acetone example
        assert ex.status == 'OK'
        # Test coefficients are computed correctly
        assert_close1d(ex.coeffs, [28.867, 0.1736, 3.19e-06, -3.932e-08])
        # Test Cp at different temperatures
        assert_close(ex.Cpig(300.0), 80.17246)
        assert_close(ex.Cpig(400.0), 96.30092)
        assert_close(ex.Cpig(500.0), 111.5495)


    
@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_DikyJoback_ring_corrections():
    """Test the DikyJoback estimator with cyclopropane to verify ring corrections"""
    ex = DikyJoback('C1CC1')  # Cyclopropane
    assert ex.status == 'OK'
    # Should include both group contributions and 3-membered ring correction
    assert ex.coeffs is not None
    # Test Cp at room temperature
    assert_close(ex.Cpig(298.15), 58.62719565202786)
    assert ex.counts == {11: 3, 42: 1, 43: 1}

    """Test the DikyJoback estimator with cyclobutane to verify 4-membered ring corrections"""
    ex = DikyJoback('C1CCC1')  # Cyclobutane
    assert ex.status == 'OK'
    # Should include both group contributions and 4-membered ring correction
    assert ex.coeffs is not None
    assert ex.counts == {11: 4, 42: 1, 44: 1}
    # Test Cp at various temperatures
    assert_close(ex.Cpig(298.15), 74.04152705570361)


@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_DikyJoback_invalid():
    """Test handling of invalid molecules"""
    # Test with iron atom which can't be fragmented
    ex = DikyJoback('[Fe]')
    assert not ex.success
    assert ex.coeffs is None
    assert ex.Cpig(300.0) is None

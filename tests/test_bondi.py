'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2024, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import *

import pytest
from fluids.numerics import *

from thermo import Chemical
from thermo.group_contribution.group_contribution_base import smarts_fragment_priority
from thermo.group_contribution.bondi import *
from thermo.group_contribution.bondi import find_methylene_rings_condensed_to_aromatic_rings, count_conjugation_interrupting_bonds, count_dioxane_rings, count_bonds_near_acid_amide, BONDI_GROUPS, count_cis_condensed_naphthenes, count_transcondensed_and_free_cycloalkyl

try:
    import rdkit
    from rdkit import Chem
except:
    rdkit = None
catalog = BONDI_GROUPS.values()

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_bondi_first_fragmentation():
    rdkitmol = Chemical('decane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=catalog, rdkitmol=rdkitmol)
    assert assignment == {4: 2, 3: 8}
    assert success

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_count_bonds_near_acid_amide():
    # Test case 1: Simple carboxylic acid (acetic acid)
    # CH3-COOH has 1 single bond adjacent to the carboxyl group
    acetic_acid = Chem.MolFromSmiles('CC(=O)O')
    assert count_bonds_near_acid_amide(acetic_acid) == 1

    # Test case 2: Propionic acid
    # CH3-CH2-COOH has 1 single bond immediately adjacent to the carboxyl group
    propionic_acid = Chem.MolFromSmiles('CCC(=O)O')
    assert count_bonds_near_acid_amide(propionic_acid) == 1

    # Test case 3: Simple amide (acetamide)
    # CH3-CONH2 has 1 single bond adjacent to the amide group
    acetamide = Chem.MolFromSmiles('CC(=O)N')
    assert count_bonds_near_acid_amide(acetamide) == 1

    # Test case 4: N,N-dimethylacetamide
    # CH3-CON(CH3)2 has 3 single bonds adjacent to the amide group
    dimethylacetamide = Chem.MolFromSmiles('CC(=O)N(C)C')
    assert count_bonds_near_acid_amide(dimethylacetamide) == 3

    # Test case 5
    acid_amide = Chemical('4-aminobutanoic acid').rdkitmol
    assert count_bonds_near_acid_amide(acid_amide) == 1

    # Test case 6: Branched carboxylic acid
    # (CH3)2CH-COOH has 3 single bonds adjacent to the carboxyl group but only 1 is immediate
    isobutyric_acid = Chem.MolFromSmiles('CC(C)C(=O)O')
    assert count_bonds_near_acid_amide(isobutyric_acid) == 1

    # Test case 7: Molecule with no acid or amide groups
    # Should return 0
    hexane = Chem.MolFromSmiles('CCCCCC')
    assert count_bonds_near_acid_amide(hexane) == 0

    # Test case 8: Complex molecule with multiple acid groups, two singles immediate neighbors
    # Succinic acid (HOOC-CH2-CH2-COOH)
    succinic_acid = Chem.MolFromSmiles('C(CC(=O)O)C(=O)O')
    assert count_bonds_near_acid_amide(succinic_acid) == 2

    # Test case 9: N-methylacetamide
    # CH3-CONHCH3 has 2 single bonds adjacent to the amide group
    n_methylacetamide = Chem.MolFromSmiles('CC(=O)NC')
    assert count_bonds_near_acid_amide(n_methylacetamide) == 2

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_count_dioxane_rings():
    # Test case 1: 1,4-dioxane (basic case)
    # A single 6-membered ring with 2 oxygens and 4 carbons
    dioxane = Chem.MolFromSmiles('O1CCOCC1')  # 1,4-dioxane
    assert count_dioxane_rings(dioxane) == 1

    # Test case 2: A molecule with two dioxane rings sharing a common edge
    # This is a fused system with two dioxane rings
    bis_dioxane = Chem.MolFromSmiles('C=CC1OCC2(CO1)COC(OC2)C=C')
    assert count_dioxane_rings(bis_dioxane) == 2

    # Test case 3: 1,3,5-trioxane (control case)
    # Similar ring structure but with 3 oxygens instead of 2
    # Should return 0 as it's not a dioxane ring
    trioxane = Chem.MolFromSmiles('O1COCOC1')
    assert count_dioxane_rings(trioxane) == 0

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_count_conjugation_interrupting_bonds():
    # Test case 1: 1,3-butadiene (control case - one interrupting bonds)
    # C=CC=C has one single bonds between conjugated double bonds
    butadiene = Chem.MolFromSmiles('C=CC=C')
    assert count_conjugation_interrupting_bonds(butadiene) == 1
    
    # Test case 2: 1,4-hexadiene
    # C=CC-C=C has zero single bond interrupting conjugation
    hexadiene = Chem.MolFromSmiles('C=CCC=C')
    assert count_conjugation_interrupting_bonds(hexadiene) == 0
    
    # Test case 3: 1,4,7-octatriene
    # C=CC-C=C-C=C has 0 single bonds interrupting conjugation
    octatriene = Chem.MolFromSmiles('C=CCC=CCC=C')
    assert count_conjugation_interrupting_bonds(octatriene) == 0
    
    # Test case 4: 1,6-diphenyl-1,3,5-hexatriene
    # Ph-C=C-C=C-C=C-Ph has conjugated system, no interrupting bonds
    diphenylhexatriene = Chem.MolFromSmiles('c1ccccc1C=CC=CC=Cc2ccccc2')
    assert count_conjugation_interrupting_bonds(diphenylhexatriene) == 2
    
    # Test case 5: Molecule with no conjugation
    # C-C-C-C has no conjugation to interrupt
    butane = Chem.MolFromSmiles('CCCC')
    assert count_conjugation_interrupting_bonds(butane) == 0
    
    # Test case 6: Complex molecule with multiple conjugated systems
    # Two separate conjugated systems with a single bond between them
    complex_diene = Chem.MolFromSmiles('C=CC=CCC=CC=C')
    assert count_conjugation_interrupting_bonds(complex_diene) == 2
    
    # Test case 7: Cyclic conjugated system
    # Benzene has no interrupting bonds
    benzene = Chem.MolFromSmiles('c1ccccc1')
    assert count_conjugation_interrupting_bonds(benzene) == 0
    
    # Test case 8: Cross-conjugated system
    # 3-Methylhexa-2,4-diene https://pubchem.ncbi.nlm.nih.gov/compound/120063#section=2D-Structure
    cross_conj = Chem.MolFromSmiles('CC=C(C)C=CC')
    assert count_conjugation_interrupting_bonds(cross_conj) == 1

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_find_methylene_rings_condensed_to_aromatic_rings():
    # Test case 1: Indane (one methylene ring with 2 CH2 groups)
    indane = Chem.MolFromSmiles('C1CC2=CC=CC=C2C1')
    rings = find_methylene_rings_condensed_to_aromatic_rings(indane)
    assert len(rings) == 1
    assert len(rings[0]) == 5  # 5 atoms in methylene ring (3 CH2 + 2 shared)
    
    # Test case 2: Tetralin (one methylene ring with 4 CH2 groups)
    tetralin = Chem.MolFromSmiles('C1CCC2=CC=CC=C2C1')
    rings = find_methylene_rings_condensed_to_aromatic_rings(tetralin)
    assert len(rings) == 1
    assert len(rings[0]) == 6  # 6 atoms in methylene ring (4 CH2 + 2 shared)
    
    # Test case 3: Benzene (no methylene rings)
    benzene = Chem.MolFromSmiles('c1ccccc1')
    rings = find_methylene_rings_condensed_to_aromatic_rings(benzene)
    assert len(rings) == 0
    
    # Test case 4: 6,7,8,9-tetrahydro-5H-benzo[7]annulene (7-membered methylene ring)
    benzocycloheptene = Chem.MolFromSmiles('C1CCCC2=CC=CC=C2C1')
    rings = find_methylene_rings_condensed_to_aromatic_rings(benzocycloheptene)
    assert len(rings) == 1
    assert len(rings[0]) == 7  # 7 atoms in methylene ring (5 CH2 + 2 shared)
    
    # Test case 5: Molecule with methyl substituent on the methylene ring
    # Should not be counted as it's not all CH2 groups
    methylindane = Chem.MolFromSmiles('CC1CC2=CC=CC=C2C1')
    rings = find_methylene_rings_condensed_to_aromatic_rings(methylindane)
    assert len(rings) == 0
    
    # Test case 6: 1,2,3,4-tetrahydroisoquinoline
    # Should not be counted as it contains NH instead of CH2
    thiq = Chem.MolFromSmiles('C1CNc2ccccc2C1')
    rings = find_methylene_rings_condensed_to_aromatic_rings(thiq)
    assert len(rings) == 0
    
    # Test case 7: 9,10-dihydroanthracene (two methylene groups between two benzene rings)
    dihydroanthracene = Chem.MolFromSmiles('c1ccc2c(c1)CC3=CC=CC=C3C2')
    rings = find_methylene_rings_condensed_to_aromatic_rings(dihydroanthracene)
    assert len(rings) == 1




@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_count_transcondensed_and_free_cycloalkyl():
    cyclohexyl = Chem.MolFromSmiles('C1CCCCC1')
    cyclopentyl = Chem.MolFromSmiles('C1CCCC1')
    trans_decalin = Chem.MolFromSmiles('[C@]12([C@](CCCC1)(CCCC2)[H])[H]')
    cis_decalin = Chem.MolFromSmiles('[C@]12([C@@](CCCC1)(CCCC2)[H])[H]')

    assert 0 == count_transcondensed_and_free_cycloalkyl(cyclohexyl)
    assert 0 == count_transcondensed_and_free_cycloalkyl(cyclopentyl)
    assert 2 == count_transcondensed_and_free_cycloalkyl(trans_decalin)
    assert 0 == count_transcondensed_and_free_cycloalkyl(cis_decalin)



    perhydroindane_trans = Chem.MolFromSmiles('[C@]12([C@](CCC1)(CC2)[H])[H]')
    assert 1 == count_transcondensed_and_free_cycloalkyl(perhydroindane_trans)
    
    # Three ring systems, one set of trans condensed, once set of cis condensed
    perhydroanthracene_trans = Chem.MolFromSmiles('[C@]12([C@](CCCC1)([C@]3([C@](CCCC3)(CC2)[H])[H])[H])[H]')
    assert 1 == count_transcondensed_and_free_cycloalkyl(perhydroanthracene_trans)
    
    # three trans - two side rings are trans by themselves plus the main center ring counts too
    perhydroanthracene_mixed = Chem.MolFromSmiles('[C@]12([C@](CCCC1)([C@]3([C@@](CCCC3)(CC2)[H])[H])[H])[H]')
    assert 3 == count_transcondensed_and_free_cycloalkyl(perhydroanthracene_mixed)
    
    # Spiro compound - "per cyclohexyl and per cyclo-pentyl ring, free"
    # so botgh count
    spiro = Chem.MolFromSmiles('C1CCC2(CCCCC2)CC1')
    assert 2 == count_transcondensed_and_free_cycloalkyl(spiro)
    
    # Bridged system, 2 5, free?
    bicyclo = Chem.MolFromSmiles('C1CC2CCC1C2')
    assert 2 == count_transcondensed_and_free_cycloalkyl(bicyclo)
    
    # System with heteroatom (should not count)
    morpholine_dimer = Chem.MolFromSmiles('[C@]12([C@](CCOC1)(CCOC2)[H])[H]')
    assert 0 == count_transcondensed_and_free_cycloalkyl(morpholine_dimer)

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_count_cis_condensed_naphthenes():
    # Cannot seem to encode stereochemistry through smiles into rdkit
    return
    # Test Case 1: Simple cis-decalin (should count as 1)
    cis_decalin = Chem.MolFromSmiles('[C@]12([C@@](CCCC1)(CCCC2)[H])[H]')
    assert count_cis_condensed_naphthenes(cis_decalin) == 1, "Failed on cis-decalin"

    # Test Case 2: Trans-decalin (should count as 0)
    trans_decalin = Chem.MolFromSmiles('[C@]12([C@](CCCC1)(CCCC2)[H])[H]')
    assert count_cis_condensed_naphthenes(trans_decalin) == 0, "Failed on trans-decalin"

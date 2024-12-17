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
from thermo.group_contribution.ppr78 import PPR78_kij, PPR78_kijs, PPR78_GROUP_IDS, PPR78_GROUPS, PPR78_INTERACTIONS, PPR78_GROUPS_BY_ID
try:
    import rdkit
    from rdkit import Chem
except:
    rdkit = None

def test_ppr78_kij_propane_butane():
    """
    Test PPR78_kij function against the example calculation from the paper for
    propane(1)/n-butane(2) system at T = 303.15 K.
    
    Values taken from:
    Jaubert, J.-N., and Fabrice Mutelet. "VLE Predictions with the Peng-Robinson 
    Equation of State and Temperature Dependent Kij Calculated through a Group 
    Contribution Method." Fluid Phase Equilibria 224, no. 2 (2004): 285-304.
    """
    # Test conditions from paper
    T = 303.15  # K
    
    # Molecule group compositions
    molecule1_groups = {
        "CH3": 2,  # propane has 2 CH3 groups
        "CH2": 1   # propane has 1 CH2 group
    }
    
    molecule2_groups = {
        "CH3": 2,  # n-butane has 2 CH3 groups
        "CH2": 2   # n-butane has 2 CH2 groups
    }
    
    # Critical properties and acentric factors from paper
    Tc1 = 369.83  # K (propane)
    Pc1 = 42.48e5  # Pa
    omega1 = 0.152
    
    Tc2 = 425.12  # K (n-butane)
    Pc2 = 37.96e5  # Pa
    omega2 = 0.200
    
    # Calculate kij
    kij = PPR78_kij(T, molecule1_groups, molecule2_groups, 
                    Tc1, Pc1, omega1, Tc2, Pc2, omega2)
    
    # Expected value from paper
    expected_kij = 0.0028
    
    # Test with reasonable tolerance due to potential rounding differences
    assert_close(kij, expected_kij, atol=0.00005)
    
    # Check group fractions
    alpha_11 = 2/3  # CH3 fraction in propane
    alpha_12 = 1/3  # CH2 fraction in propane
    alpha_21 = 1/2  # CH3 fraction in butane
    alpha_22 = 1/2  # CH2 fraction in butane
    
    # Calculate fractions from our function inputs for verification
    calc_alpha_11 = molecule1_groups["CH3"] / sum(molecule1_groups.values())
    calc_alpha_12 = molecule1_groups["CH2"] / sum(molecule1_groups.values())
    calc_alpha_21 = molecule2_groups["CH3"] / sum(molecule2_groups.values())
    calc_alpha_22 = molecule2_groups["CH2"] / sum(molecule2_groups.values())
    
    # Verify group fractions
    assert_close(calc_alpha_11, alpha_11, rtol=1e-10)
    assert_close(calc_alpha_12, alpha_12, rtol=1e-10)
    assert_close(calc_alpha_21, alpha_21, rtol=1e-10)
    assert_close(calc_alpha_22, alpha_22, rtol=1e-10)

def test_ppr78_kijs_matrix():
    """
    Test PPR78_kijs matrix calculation for a propane/n-butane/n-pentane system.
    """
    T = 303.15  # K
    
    # Molecule group compositions
    groups = [
        {"CH3": 2, "CH2": 1},    # propane
        {"CH3": 2, "CH2": 2},    # n-butane
        {"CH3": 2, "CH2": 3},    # n-pentane
    ]
    
    # Critical properties and acentric factors
    Tcs = [369.83, 425.12, 469.70]  # K
    Pcs = [42.48e5, 37.96e5, 33.70e5]  # Pa
    omegas = [0.152, 0.200, 0.252]
    
    # Calculate full kij matrix
    kij_matrix = PPR78_kijs(T, groups, Tcs, Pcs, omegas)
    
    # Test matrix properties
    assert len(kij_matrix) == 3, "Matrix should be 3x3"
    assert all(len(row) == 3 for row in kij_matrix), "Matrix should be square"

    assert_close2d(kij_matrix, [[0.0, 0.0027994360072542274, 0.006933513208806096], 
                                [0.0027994360072542274, 0.0, 0.000827218293750849], 
                                [0.006933513208806096, 0.000827218293750849, 0.0]],
                    rtol=1e-7)

def readable_assignment(assignment):
    return {PPR78_GROUPS_BY_ID[i].group : v for i, v in assignment.items()}

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_table_2_2005_paper_compounds():
    '''Jaubert, Jean-Noël, Stéphane Vitu, Fabrice Mutelet, and Jean-Pierre Corriou. 
    "Extension of the PPR78 Model (Predictive 1978, Peng-Robinson EOS with Temperature 
    Dependent Kij Calculated through a Group Contribution Method) to Systems Containing
    Aromatic Compounds." Fluid Phase Equilibria 237, no. 1-2 (October 25, 2005): 
    193-211. doi:10.1016/j.fluid.2005.09.003. 
    '''
    rdkitmol = Chemical('nitrogen').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'N2': 1}
    assert success
    rdkitmol = Chemical('carbon dioxide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CO2': 1}
    assert success

    rdkitmol = Chemical('hydrogen sulfide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'H2S': 1}
    assert success


    rdkitmol = Chemical('methane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH4': 1}
    assert success


    # Ethane - should use the specific C2H6 group
    rdkitmol = Chemical('ethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'C2H6': 1}
    assert success

    # Propane - should be CH3-CH2-CH3
    rdkitmol = Chemical('propane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 2, 'CH2': 1}
    assert success

    # n-Butane - should be CH3-CH2-CH2-CH3
    rdkitmol = Chemical('butane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 2, 'CH2': 2}
    assert success


    # n-Pentane (CH3-CH2-CH2-CH2-CH3)
    rdkitmol = Chemical('pentane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 2, 'CH2': 3}
    assert success

    # 2-Methyl butane (iso-pentane) (CH3-CH(CH3)-CH2-CH3)
    rdkitmol = Chemical('2-methylbutane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 3, 'CH': 1, 'CH2': 1}
    assert success

    # 2,2-Dimethyl propane (neo-pentane) ((CH3)4-C)
    rdkitmol = Chemical('2,2-dimethylpropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 4, 'C': 1}
    assert success

    # n-Hexane (CH3-CH2-CH2-CH2-CH2-CH3)
    rdkitmol = Chemical('hexane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 2, 'CH2': 4}
    assert success

    # 2-Methyl pentane (CH3-CH(CH3)-CH2-CH2-CH3)
    rdkitmol = Chemical('2-methylpentane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 3, 'CH': 1, 'CH2': 2}
    assert success

    # 3-Methyl pentane (CH3-CH2-CH(CH3)-CH2-CH3)
    rdkitmol = Chemical('3-methylpentane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 3, 'CH': 1, 'CH2': 2}
    assert success

    # 2,2-Dimethyl butane (CH3-C(CH3)2-CH2-CH3)
    rdkitmol = Chemical('2,2-dimethylbutane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 4, 'C': 1, 'CH2': 1}
    assert success

    # 2,3-Dimethyl butane (CH3-CH(CH3)-CH(CH3)-CH3)
    rdkitmol = Chemical('2,3-dimethylbutane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 4, 'CH': 2}
    assert success

    # Benzene (C6H6)
    rdkitmol = Chemical('benzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 6}
    assert success

    # n-Heptane (CH3-CH2-CH2-CH2-CH2-CH2-CH3)
    rdkitmol = Chemical('heptane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 2, 'CH2': 5}
    assert success

    # 3-Methyl hexane (CH3-CH2-CH(CH3)-CH2-CH2-CH3)
    rdkitmol = Chemical('3-methylhexane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 3, 'CH': 1, 'CH2': 3}
    assert success

    # 2,3-Dimethyl pentane (CH3-CH(CH3)-CH(CH3)-CH2-CH3)
    rdkitmol = Chemical('2,3-dimethylpentane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 4, 'CH': 2, 'CH2': 1}
    assert success

    # 2,4-Dimethyl pentane (CH3-CH(CH3)-CH2-CH(CH3)-CH3)
    rdkitmol = Chemical('2,4-dimethylpentane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 4, 'CH': 2, 'CH2': 1}
    assert success

    # 2,2,3-Trimethyl butane (CH3-C(CH3)2-CH(CH3)-CH3)
    rdkitmol = Chemical('2,2,3-trimethylbutane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 5, 'CH': 1, 'C': 1}
    assert success

    # 2,2,4-Trimethyl pentane (iso-octane) (CH3-C(CH3)2-CH2-CH(CH3)-CH3)
    rdkitmol = Chemical('2,2,4-trimethylpentane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 5, 'CH': 1, 'C': 1, 'CH2': 1}
    assert success

    # Methyl benzene (toluene) (C6H5-CH3)
    rdkitmol = Chemical('toluene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 5, 'Caro': 1, 'CH3': 1}
    assert success

    # n-Octane (CH3-(CH2)6-CH3)
    rdkitmol = Chemical('octane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 2, 'CH2': 6}
    assert success

    # 2,2,5-Trimethyl hexane (CH3-CH2-CH2-CH(CH3)-C(CH3)2-CH3)
    rdkitmol = Chemical('2,2,5-trimethylhexane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 5, 'CH': 1, 'C': 1, 'CH2': 2}
    assert success

    # Ethyl benzene (C6H5-CH2-CH3)
    rdkitmol = Chemical('ethylbenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 5, 'Caro': 1, 'CH2': 1, 'CH3': 1}
    assert success

    # 1,4-Dimethyl benzene (para-xylene)
    rdkitmol = Chemical('p-xylene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 4, 'Caro': 2, 'CH3': 2}
    assert success

    # 1,3-Dimethyl benzene (meta-xylene)
    rdkitmol = Chemical('m-xylene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 4, 'Caro': 2, 'CH3': 2}
    assert success

    # 1,2-Dimethyl benzene (ortho-xylene)
    rdkitmol = Chemical('o-xylene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 4, 'Caro': 2, 'CH3': 2}
    assert success

    # n-Propyl benzene (C6H5-CH2-CH2-CH3)
    rdkitmol = Chemical('propylbenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 5, 'Caro': 1, 'CH2': 2, 'CH3': 1}
    assert success

    # Isopropyl benzene (cumene) (C6H5-CH(CH3)2)
    rdkitmol = Chemical('isopropylbenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 5, 'Caro': 1, 'CH': 1, 'CH3': 2}
    assert success

    # 1,3,5-Trimethyl benzene (mesitylene)
    rdkitmol = Chemical('1,3,5-trimethylbenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 3, 'Caro': 3, 'CH3': 3}
    assert success

    # 1,2,4-Trimethyl benzene
    rdkitmol = Chemical('1,2,4-trimethylbenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 3, 'Caro': 3, 'CH3': 3}
    assert success

    # Naphthalene (C10H8)
    rdkitmol = Chemical('naphthalene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 8, 'Cfused_aromatic': 2}
    assert success

    # n-Nonane (CH3-(CH2)7-CH3)
    rdkitmol = Chemical('nonane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 2, 'CH2': 7}
    assert success

    # 2,2-Dimethyl heptane (CH3-CH2-CH2-CH2-CH2-C(CH3)2-CH3)
    rdkitmol = Chemical('2,2-dimethylheptane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 4, 'C': 1, 'CH2': 4}
    assert success

    # Butyl benzene (C6H5-CH2-CH2-CH2-CH3)
    rdkitmol = Chemical('butylbenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 5, 'Caro': 1, 'CH2': 3, 'CH3': 1}
    assert success

    # Tertiobutyl benzene (C6H5-C(CH3)3)
    rdkitmol = Chemical('t-butylbenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 5, 'Caro': 1, 'C': 1, 'CH3': 3}
    assert success

    # 1-Methyl naphthalene
    rdkitmol = Chemical('1-methylnaphthalene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 7, 'Cfused_aromatic': 2, 'Caro': 1, 'CH3': 1}
    assert success

    # Long chain n-alkanes
    alkanes = {
        'decane': {'CH3': 2, 'CH2': 8},
        'dodecane': {'CH3': 2, 'CH2': 10},
        'tridecane': {'CH3': 2, 'CH2': 11},
        'tetradecane': {'CH3': 2, 'CH2': 12},
        'pentadecane': {'CH3': 2, 'CH2': 13},
        'hexadecane': {'CH3': 2, 'CH2': 14},
        'heptadecane': {'CH3': 2, 'CH2': 15}
    }
    
    for name, expected in alkanes.items():
        rdkitmol = Chemical(name).rdkitmol
        assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
        assert readable_assignment(assignment) == expected
        assert success

    # Phenanthrene (C14H10)
    rdkitmol = Chemical('phenanthrene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 10, 'Cfused_aromatic': 4}
    assert success


@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_extension_naphtenic():
    '''Vitu, Stéphane, Jean-Noël Jaubert, and Fabrice Mutelet. "Extension of the PPR78 Model 
    (Predictive 1978, Peng–Robinson EOS with Temperature Dependent Kij Calculated through 
    a Group Contribution Method) to Systems Containing Naphtenic Compounds." 
    Fluid Phase Equilibria 243, no. 1-2 (May 10, 2006): 9-28. doi:10.1016/j.fluid.2006.02.004. 
    '''
    # 1,2,3-Trimethyl benzene
    rdkitmol = Chemical('1,2,3-trimethylbenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 3, 'Caro': 3, 'CH3': 3}
    assert success

    # 2,2,4,4,6,8,8-Heptamethyl nonane
    rdkitmol = Chemical('2,2,4,4,6,8,8-heptamethylnonane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 9, 'CH': 1, 'C': 3, 'CH2': 3}
    assert success

    # 3,4-Dimethyl hexane
    rdkitmol = Chemical('3,4-dimethylhexane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 4, 'CH': 2, 'CH2': 2}
    assert success

    # 3-Methyl pentane (already covered in previous tests, but including for completeness)
    rdkitmol = Chemical('3-methylpentane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 3, 'CH': 1, 'CH2': 2}
    assert success

    # 4-Methyl heptane
    rdkitmol = Chemical('4-methylheptane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 3, 'CH': 1, 'CH2': 4}
    assert success

    # Biphenyl (phenyl benzene)
    rdkitmol = Chemical('biphenyl').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 10, 'Caro': 2}
    assert success

    # Cis-decalin
    rdkitmol = Chemical('cis-decalin').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH2cyclic': 8, 'CHcyclic': 2}
    assert success

    # Cycloheptane
    rdkitmol = Chemical('cycloheptane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH2cyclic': 7}
    assert success

    # Cyclohexane
    rdkitmol = Chemical('cyclohexane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH2cyclic': 6}
    assert success

    # Cyclooctane
    rdkitmol = Chemical('cyclooctane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH2cyclic': 8}
    assert success

    # Cyclopentane
    rdkitmol = Chemical('cyclopentane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH2cyclic': 5}
    assert success

    # Ethyl cyclohexane
    rdkitmol = Chemical('ethylcyclohexane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH2cyclic': 5, 'CHcyclic': 1, 'CH2': 1, 'CH3': 1}
    assert success

    # Methyl cyclohexane
    rdkitmol = Chemical('methylcyclohexane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH2cyclic': 5, 'CHcyclic': 1, 'CH3': 1}
    assert success

    # Methyl cyclopentane
    rdkitmol = Chemical('methylcyclopentane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH2cyclic': 4, 'CHcyclic': 1, 'CH3': 1}
    assert success

    # n-Butyl benzene
    rdkitmol = Chemical('butylbenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CHaro': 5, 'Caro': 1, 'CH2': 3, 'CH3': 1}
    assert success

    # n-Propyl cyclohexane
    rdkitmol = Chemical('propylcyclohexane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH2cyclic': 5, 'CHcyclic': 1, 'CH2': 2, 'CH3': 1}
    assert success

    # Tetralin (1,2,3,4-tetrahydronaphthalene)
    rdkitmol = Chemical('tetralin').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'Cfused_aromatic': 2, 'CHaro': 4, 'CH2cyclic': 4}
    assert success

    # Trans-decalin
    rdkitmol = Chemical('trans-decalin').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH2cyclic': 8, 'CHcyclic': 2}
    assert success

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_SH_group():
    """Privat, Romain, Jean-Noël Jaubert, and Fabrice Mutelet. "Addition of
    the Sulfhydryl Group (-SH) to the PPR78 Model (Predictive 1978, 
    Peng-Robinson EOS with Temperature Dependent kIj Calculated through 
    a Group Contribution Method)." The Journal of Chemical 
    Thermodynamics 40, no. 9 (September 1, 2008): 1331-41.
    """
    # Methyl mercaptan (CH3-SH)
    rdkitmol = Chemical('methanethiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 1, 'SH': 1}
    assert success

    # Ethyl mercaptan (CH3-CH2-SH)
    rdkitmol = Chemical('ethanethiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 1, 'CH2': 1, 'SH': 1}
    assert success

    # Propyl mercaptan (CH3-CH2-CH2-SH)
    rdkitmol = Chemical('propane-1-thiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 1, 'CH2': 2, 'SH': 1}
    assert success

    # Butyl mercaptan (CH3-CH2-CH2-CH2-SH)
    rdkitmol = Chemical('butane-1-thiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 1, 'CH2': 3, 'SH': 1}
    assert success

    # Isopropyl mercaptan (CH3-CH(CH3)-SH)
    rdkitmol = Chemical('propane-2-thiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 2, 'CH': 1, 'SH': 1}
    assert success

    # Secbutyl mercaptan (CH3-CH2-CH(CH3)-SH)
    rdkitmol = Chemical('butane-2-thiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 2, 'CH2': 1, 'CH': 1, 'SH': 1}
    assert success

    # Isobutyl mercaptan (CH3-CH(CH3)-CH2-SH)
    rdkitmol = Chemical('2-methylpropane-1-thiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 2, 'CH': 1, 'CH2': 1, 'SH': 1}
    assert success

    # Tertbutyl mercaptan ((CH3)3C-SH)
    rdkitmol = Chemical('2-methylpropane-2-thiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PPR78_GROUPS, rdkitmol=rdkitmol)
    assert readable_assignment(assignment) == {'CH3': 3, 'C': 1, 'SH': 1}
    assert success
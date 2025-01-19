'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2024 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains an implementation of the Bondi
group-contribution method.
This functionality requires the RDKit library to work.

.. contents:: :local:


.. autofunction:: thermo.group_contribution.bondi_van_der_waals_surface_area_volume
.. autofunction:: thermo.group_contribution.R_from_Van_der_Waals_volume
.. autofunction:: thermo.group_contribution.Q_from_Van_der_Waals_area


'''
__all__ = []
from thermo.functional_groups import FG_CARBOXYLIC_ACID, FG_AMIDE, identify_conjugated_bonds, identify_functional_group_atoms, count_rings_by_atom_counts
from thermo.group_contribution.group_contribution_base import priority_from_atoms, SINGLE_BOND, DOUBLE_BOND, TRIPLE_BOND, AROMATIC_BOND

# WIP


GROUP_ID_COUNTER = 0 # temporary

class BondiGroupContribution:
    __slots__ = ('group', 'group_id', 'Vw', 'Aw', 'smarts', 'atoms', 'bonds', 'hydrogen_from_smarts', 'priority', 'smart_rdkit')
    
    def __init__(self, group, Vw, Aw, smarts=None,
                 priority=None, atoms=None, bonds=None, hydrogen_from_smarts=False):
        """
        Initialize the GroupContribution object with group name, volume contribution, and polarizability.
        
        Parameters:
        - group (str): The name or identifier of the group.
        - Vw (float): Volume contribution in cm^3/mole.
        - Aw (float): Polarizability contribution in cm^3/mole * 10^9.
        """
        global GROUP_ID_COUNTER
        self.group = group
        self.Vw = Vw
        self.Aw = Aw
        self.smarts = smarts
        self.atoms = atoms
        self.bonds = bonds
        self.hydrogen_from_smarts = hydrogen_from_smarts
        self.priority = priority
        self.smart_rdkit = None
        self.group_id = GROUP_ID_COUNTER
        GROUP_ID_COUNTER += 1


    def __repr__(self):
        return f"GroupContribution(group={self.group!r}, Vw={self.Vw!r}, Aw={self.Aw!r})"

BONDI_GROUPS_BY_ID = {}
BONDI_GROUPS = {}
# TABLE XV in van der Waals Volumes and Radii, 1964
BONDI_GROUPS['C'] = BondiGroupContribution('C', 3.33, 0.0, smarts='[CX4;H0]', atoms={'C': 1, 'H': 0})
BONDI_GROUPS['CH'] = BondiGroupContribution('CH', 6.78, 0.57, smarts='[CX4;H1]', atoms={'C': 1, 'H': 1})
BONDI_GROUPS['CH2'] = BondiGroupContribution('CH2', 10.23, 1.35, smarts='[CX4;H2]', atoms={'C': 1, 'H': 2})
BONDI_GROUPS['CH3'] = BondiGroupContribution('CH3', 13.67, 2.12, smarts='[CX4;H3]', atoms={'C': 1, 'H': 3})

BONDI_GROUPS['CH4'] = BondiGroupContribution(
    'CH4', 17.12, 2.90,
    atoms={'C': 1, 'H': 4},
    bonds={SINGLE_BOND: 4},
    smarts='[CX4;H4]'
)
# n-paraffins specific correlations: 6.88 + 10.23 Nc for Vw, and 1.54 + 1.35 Nc for Aw

BONDI_GROUPS['=C='] = BondiGroupContribution(
    '=C=', 6.96, None,
    atoms={'C': 2, 'H': 0},
    bonds={DOUBLE_BOND: 2},
    smarts='[C;X2;R0;$(*=,=*)]'
)

# >C=C< (internal double bond with two carbons)
BONDI_GROUPS['>C=C<'] = BondiGroupContribution(
    '>C=C<', 10.02, 0.61,
    atoms={'C': 2, 'H': 0},
    bonds={DOUBLE_BOND: 1, SINGLE_BOND: 4},
    smarts='[C;X3;R0]=[C;X3;R0]'
)

# =CH (terminal double bond with one hydrogen)
BONDI_GROUPS['=CH-'] = BondiGroupContribution(
    '=CH-', 8.47, 1.08,
    atoms={'C': 1, 'H': 1},
    bonds={DOUBLE_BOND: 1, SINGLE_BOND: 1},
    smarts='[C;H1;X3;$(*=*)]'
)

# =CH2 (terminal double bond with two hydrogens)
BONDI_GROUPS['=CH2'] = BondiGroupContribution(
    '=CH2', 11.94, 1.86,
    atoms={'C': 1, 'H': 2},
    bonds={DOUBLE_BOND: 1},
    smarts='[C;H2;X3;R0;$(*=*)]'
)

# >C=CH2 (internal double bond with one terminal CH2)
BONDI_GROUPS['>C=CH2'] = BondiGroupContribution(
    '>C=CH2', 16.95, 2.17,
    atoms={'C': 2, 'H': 2},
    bonds={DOUBLE_BOND: 1, SINGLE_BOND: 2},
    smarts='[C;X3;R0]=[C;H2;R0]'
)

# >C=CH- (internal double bond with one terminal CH)
BONDI_GROUPS['>C=CH-'] = BondiGroupContribution(
    '>C=CH-', 13.49, 1.39,
    atoms={'C': 2, 'H': 1},
    bonds={DOUBLE_BOND: 1, SINGLE_BOND: 3},
    smarts='[C;X3;H0]=[C;X3;H1]'
)

# Acetylenic group definitions
# —C≡ (internal triple bond)
BONDI_GROUPS['-C≡'] = BondiGroupContribution(
    '-C≡', 8.05, 0.98,
    atoms={'C': 1},
    bonds={TRIPLE_BOND: 1, SINGLE_BOND: 1},
    smarts='[C;H0;X2;R0;$(*#*)]'
)


# ≡C—H (terminal acetylenic carbon)
BONDI_GROUPS['≡C-H'] = BondiGroupContribution(
    '≡C-H', 11.55, 1.74,
    atoms={'C': 1, 'H': 1},
    bonds={TRIPLE_BOND: 1},
    smarts='[C;H1;X2;R0;$(*#[C;H1;X2;R0;$(*#[C;H1])])][H]'
)
# There is also a version of ≡C- for diacetylene but we it includes Vw and Aw only
# and the value for diacetylene is likely to come from UNIFAC



# >C— (condensation): carbon in a fused aromatic ring system
BONDI_GROUPS['>C— (condensation)'] = BondiGroupContribution(
    '>C— (condensation)', 4.74, 0.21,
    atoms={'C': 1},
    bonds={AROMATIC_BOND: 2},
    smarts='[c;R2]'
)

# >C— (alkyl): alkyl-substituted aromatic carbon
BONDI_GROUPS['>C— (alkyl)'] = BondiGroupContribution(
    '>C— (alkyl)', 5.54, 0.30,
    atoms={'C': 1},
    bonds={AROMATIC_BOND: 1, SINGLE_BOND: 1},
    smarts='[c;R1;$(C-[C,H])]'
)

# >C—H: terminal aromatic carbon with hydrogen
BONDI_GROUPS['>C—H'] = BondiGroupContribution(
    '>C—H', 8.06, 1.00,
    atoms={'C': 1, 'H': 1},
    bonds={AROMATIC_BOND: 1, SINGLE_BOND: 1},
    smarts='[cH;R1]'
)

# Benzene
BONDI_GROUPS['Benzene'] = BondiGroupContribution(
    'Benzene', 48.36, 6.01,
    atoms={'C': 6},
    bonds={AROMATIC_BOND: 6},
    smarts='c1ccccc1'
)

# Phenyl
BONDI_GROUPS['Phenyl'] = BondiGroupContribution(
    'Phenyl', 45.84, 5.33,
    atoms={'C': 6},
    bonds={AROMATIC_BOND: 5, SINGLE_BOND: 1},
    smarts='[c1ccccc1]'
)

# Naphthalene
BONDI_GROUPS['Naphthalene'] = BondiGroupContribution(
    'Naphthalene', 73.97, 8.44,
    atoms={'C': 10},
    bonds={AROMATIC_BOND: 12},
    smarts='c1c2ccccc2ccc1'
)

# Naphthyl
BONDI_GROUPS['Naphthyl'] = BondiGroupContribution(
    'Naphthyl', 71.45, 7.76,
    atoms={'C': 10},
    bonds={AROMATIC_BOND: 11, SINGLE_BOND: 1},
    smarts='[c]1ccc2ccccc2c1'
)







# TABLE XVI in van der Waals Volumes and Radii, 1964

# -O- (c.e.): heterocycloaliphatic esters
BONDI_GROUPS['-O- (c.e.)'] = BondiGroupContribution(
    '-O- (c.e.)', 5.20, 0.74,
    atoms={'O': 1},
    bonds={SINGLE_BOND: 2},
    smarts='[O;X2;R1]'
)

# -O- (a.e.): polyalkane ethers
BONDI_GROUPS['-O- (a.e.)'] = BondiGroupContribution(
    '-O- (a.e.)', 3.70, 0.60,
    atoms={'O': 1},
    bonds={SINGLE_BOND: 2},
    smarts='[O;X2;R0]'
)

# -O- (ph.e.): polyphenyl ethers
BONDI_GROUPS['-O- (ph.e.)'] = BondiGroupContribution(
    '-O- (ph.e.)', 3.20, 0.54,
    atoms={'O': 1},
    bonds={SINGLE_BOND: 2},
    smarts='[O;X2;$(c1ccccc1)]'
)

# -OH: hydroxyl group
BONDI_GROUPS['-OH'] = BondiGroupContribution(
    '-OH', 8.04, 1.46,
    atoms={'O': 1, 'H': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[O;H1]'
)
# >C=O: carbonyl group (non-aromatic)
BONDI_GROUPS['>C=O'] = BondiGroupContribution(
    '>C=O', 11.70, 1.60,
    atoms={'C': 1, 'O': 1},
    bonds={DOUBLE_BOND: 1, SINGLE_BOND: 2},
    smarts='[C;X3](=O)'
)
# -S- and -SH do not include an Aw contribution
# -S-: sulfur ether
BONDI_GROUPS['-S-'] = BondiGroupContribution(
    '-S-', 10.8, None,
    atoms={'S': 1},
    bonds={SINGLE_BOND: 2},
    smarts='[S;X2]'
)

# -SH: thiol group
BONDI_GROUPS['-SH'] = BondiGroupContribution(
    '-SH', 14.8, None,
    atoms={'S': 1, 'H': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[S;H1]'
)

# -NH2 (amino group)
BONDI_GROUPS['-NH2'] = BondiGroupContribution(
    '-NH2', 10.54, 1.74,
    atoms={'N': 1, 'H': 2},
    bonds={SINGLE_BOND: 1},
    smarts='[N;H2;X3]'
)


# >NH (secondary amine)
BONDI_GROUPS['>NH'] = BondiGroupContribution(
    '>NH', 8.08, 0.99,
    atoms={'N': 1, 'H': 1},
    bonds={SINGLE_BOND: 2},
    smarts='[N;H1;X3]'
)

# NX3H0 (tertiary amine)
BONDI_GROUPS['NX3H0'] = BondiGroupContribution(
    'NX3H0', 4.33, 0.23,
    atoms={'N': 1},
    bonds={SINGLE_BOND: 3},
    smarts='[N;H0;X3]'
)

# -C≡N (nitrile group)
BONDI_GROUPS['-C≡N'] = BondiGroupContribution(
    '-C≡N', 14.70, 2.19,
    atoms={'C': 1, 'N': 1},
    bonds={TRIPLE_BOND: 1, DOUBLE_BOND: 1},
    smarts='[C;X2;H0]#[N;H0;X1]'
)

# -NO2 (nitro group)
BONDI_GROUPS['-NO2'] = BondiGroupContribution(
    '-NO2', 16.8, 2.55,
    atoms={'N': 1, 'O': 2},
    bonds={SINGLE_BOND: 2, DOUBLE_BOND: 1},
    smarts='[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'
)
# phosphorous not sure what they tried to draw, looks wrong, has only Vw


# Fluorine group definitions

# -F (pr): primary aliphatic fluorine, attached to an alkane in the primary position
BONDI_GROUPS['-F (pr)'] = BondiGroupContribution(
    '-F (pr)', 5.72, 1.10,
    atoms={'F': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[F;X1;R0]'
)

# -F (s,t): secondary or tertiary aliphatic fluorine, attached to alkane
BONDI_GROUPS['-F (s,t)'] = BondiGroupContribution(
    '-F (s,t)', 6.20, 1.18,
    atoms={'F': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[F;X1;R0;$(C([#6,#1])([#6,#1]))]'
)

# -F (p): per- or polyhalide of an alkane
BONDI_GROUPS['-F (p)'] = BondiGroupContribution(
    '-F (p)', 6.00, 1.15,
    atoms={'F': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[F$([*;!R]C([F,Cl,Br,I])[#6,F,Cl,Br,I])]'
)

# -F (ph): phenyl fluorine, attached to phenyl ring
BONDI_GROUPS['-F (ph)'] = BondiGroupContribution(
    '-F (ph)', 5.80, 1.10,
    atoms={'F': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[F;$(c1ccccc1)]'
)

# Chlorine group definitions

# -Cl (pr): primary aliphatic chlorine, attached to an alkane in the primary position
BONDI_GROUPS['-Cl (pr)'] = BondiGroupContribution(
    '-Cl (pr)', 11.62, 1.80,
    atoms={'Cl': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[Cl;X1;R0]'
)

# -Cl (s,t,p): secondary, tertiary, or per/polyhalide of an alkane
BONDI_GROUPS['-Cl (s,t,p)'] = BondiGroupContribution(
    '-Cl (s,t,p)', 12.24, 1.82,
    atoms={'Cl': 1},
    bonds={SINGLE_BOND: 1},
    smarts=[
        '[Cl;X1;R0;$(C([#6,#1])([#6,#1]))]',
        '[Cl$([*;!R]C([F,Cl,Br,I])[#6,F,Cl,Br,I])]'
    ]
)

# -Cl (v): chlorine attached to a vinyl group
BONDI_GROUPS['-Cl (v)'] = BondiGroupContribution(
    '-Cl (v)', 11.65, 1.80,
    atoms={'Cl': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[Cl;$([Cl]C=C)]'
)

# -Cl (ph): phenyl chlorine, attached to phenyl ring
BONDI_GROUPS['-Cl (ph)'] = BondiGroupContribution(
    '-Cl (ph)', 12.0, 1.81,
    atoms={'Cl': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[Cl;$(c1ccccc1)]'
)


# -Br (pr): primary aliphatic bromine, attached to an alkane in the primary position
BONDI_GROUPS['-Br (pr)'] = BondiGroupContribution(
    '-Br (pr)', 14.40, 2.08,
    atoms={'Br': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[Br;X1;R0]'
)

# -Br (s,t,p): secondary, tertiary, or per/polyhalide of an alkane
BONDI_GROUPS['-Br (s,t,p)'] = BondiGroupContribution(
    '-Br (s,t,p)', 14.60, 2.09,
    atoms={'Br': 1},
    bonds={SINGLE_BOND: 1},
    smarts=[
        '[Br;X1;R0;$(C([#6,#1])([#6,#1]))]',
        '[Br$([*;!R]C([F,Cl,Br,I])[#6,F,Cl,Br,I])]'
    ]
)

# -Br (ph): phenyl bromine, attached to phenyl ring
BONDI_GROUPS['-Br (pr)'] = BondiGroupContribution(
    '-Br (pr)', 15.12, 2.13,
    atoms={'Br': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[Br;$(c1ccccc1)]'
)


# -I (pr): primary aliphatic iodine, attached to an alkane in the primary position
BONDI_GROUPS['-I (pr)'] = BondiGroupContribution(
    '-I (pr)', 19.18, 2.48,
    atoms={'I': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[I;X1;R0]'
)

# -I (s,t,p): secondary, tertiary, or per/polyhalide of an alkane
BONDI_GROUPS['-I (s,t,p)'] = BondiGroupContribution(
    '-I (s,t,p)', 20.35, 2.54,
    atoms={'I': 1},
    bonds={SINGLE_BOND: 1},
    smarts=[
        '[I;X1;R0;$(C([#6,#1])([#6,#1]))]',
        '[I$([*;!R]C([F,Cl,Br,I])[#6,F,Cl,Br,I])]'
    ]
)

# -I (ph): phenyl iodine, attached to phenyl ring
BONDI_GROUPS['-I (ph)'] = BondiGroupContribution(
    '-I (ph)', 19.64, 2.51,
    atoms={'I': 1},
    bonds={SINGLE_BOND: 1},
    smarts='[I;$(c1ccccc1)]'
)

# TABLE XVII in van der Waals Volumes and Radii, 1964




for group in BONDI_GROUPS.values():
    if group.priority is None:
        if group.atoms is not None:
            group.priority = priority_from_atoms(group.atoms, group.bonds)
    BONDI_GROUPS_BY_ID[group.group_id] = group
catalog = BONDI_GROUPS.values()


def count_bonds_near_acid_amide(mol):
    """Count single bonds adjacent to carboxyl or amide groups.
    Each such bond contributes a decrement of 0.22 to Vw.
    """
    from rdkit import Chem
    # Get all atoms in carboxyl and amide groups
    carboxyl_matches = identify_functional_group_atoms(mol, FG_CARBOXYLIC_ACID)
    amide_matches = identify_functional_group_atoms(mol, FG_AMIDE)
    
    # Combine all group atoms into a set
    group_atoms = set()
    for match in carboxyl_matches + amide_matches:
        group_atoms.update(match)
    
    # Count qualifying single bonds
    count = 0
    for atom_idx in group_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            # Only count bonds to atoms outside the functional group
            if neighbor_idx not in group_atoms:
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    count += 1
    
    return count

def count_dioxane_rings(mol):
    """Count dioxane rings for Bondi decrements.
    Each dioxane ring contributes a decrement of 1.70 to Vw and 0.7 to Aw.
    """
    # Dioxane ring has exactly 2 oxygens and 4 carbons
    return count_rings_by_atom_counts(mol, {'O': 2, 'C': 4})

def count_conjugation_interrupting_bonds(mol):
    """Count single bonds between conjugated double bonds.
    Each such bond contributes a decrement of 0.25 to Vw.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to analyze
        
    Returns
    -------
    int
        Number of single bonds between conjugated double bonds
    """
    conjugated_systems = identify_conjugated_bonds(mol)
    return len(conjugated_systems)

# checked with various groups
def find_methylene_rings_condensed_to_aromatic_rings(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, find all methylene rings that are
    condensed to benzene rings. A methylene ring is defined as a saturated ring where
    all non-shared atoms are CH2 groups.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to analyze

    Returns
    -------
    list
        List of tuples containing the atom indices for each unique methylene ring
        condensed to benzene. The indices are sorted for uniqueness.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> # Indane has one methylene ring
    >>> find_methylene_rings_condensed_to_aromatic_rings(MolFromSmiles('C1CC2=CC=CC=C2C1')) # doctest:+SKIP
    [(0, 1, 2, 7, 8)]
    >>> # Tetralin also has one methylene ring
    >>> find_methylene_rings_condensed_to_aromatic_rings(MolFromSmiles('C1CCC2=CC=CC=C2C1')) # doctest:+SKIP
    [(0, 1, 2, 3, 8, 9)]
    '''
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    ring_count = len(atom_rings)
    
    # Convert to sets for easier intersection operations
    ring_sets = [set(r) for r in atom_rings]
    
    # Find aromatic rings (benzene rings)
    aromatic_rings = []
    for i, ring in enumerate(ring_sets):
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings.append(i)
    
    # Find potential methylene rings (non-aromatic) condensed to benzene
    methylene_ring_candidates = []
    for arom_idx in aromatic_rings:
        arom_ring = ring_sets[arom_idx]
        for i, ring in enumerate(ring_sets):
            if i == arom_idx:
                continue
            
            # Check if rings share exactly 2 atoms
            shared = arom_ring.intersection(ring)
            if len(shared) != 2:
                continue
                
            # Check if shared atoms are adjacent in both rings
            # Convert shared atoms to list for indexing
            shared_list = list(shared)
            for atom1 in shared_list:
                for atom2 in shared_list:
                    if atom1 >= atom2:
                        continue
                    # Check if atoms are bonded
                    bond = mol.GetBondBetweenAtoms(atom1, atom2)
                    if bond is not None:
                        # Quick check - does this ring have at least one CH2?
                        has_ch2 = False
                        for atom_idx in ring:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            if (atom.GetSymbol() == 'C' and
                                atom.GetTotalNumHs() == 2 and 
                                atom.GetDegree() == 2):
                                has_ch2 = True
                                break
                        if has_ch2:
                            methylene_ring_candidates.append(ring)
                        break                        
    # Now check each candidate ring to see if it's a true methylene ring
#     print(methylene_ring_candidates, 'methylene_ring_candidates')
    methylene_rings = []
    for ring in methylene_ring_candidates:
        is_methylene = True
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            # Skip the shared atoms (which will be aromatic)
            if atom.GetIsAromatic():
                continue
            # Check if it's a CH2 group:
            # - Must be carbon
            # - Must have exactly 2 hydrogens
            # - Must have exactly 2 bonds (to other ring carbons)
            if (atom.GetSymbol() != 'C' or 
                atom.GetTotalNumHs() != 2 or
                atom.GetDegree() != 2):
                is_methylene = False
                break
        
        if is_methylene:
            # Add sorted tuple of atom indices for uniqueness
            methylene_rings.append(tuple(sorted(ring)))
    
    # Remove any duplicates and return
    return sorted(list(set(methylene_rings)))



def bondi_van_der_waals_surface_area_volume(rdkitmol):
    """
    Calculate the van der Waals volume `V_vdw` and surface area `A_vdw`
    for a given molecule using Bondi group contributions.

    This function identifies the Bondi groups present in the molecule 
    using SMARTS-based fragmentation and computes the total van der Waals 
    volume and surface area from the bondi group contributions method.

    Parameters
    ----------
    rdkitmol : rdkit.Chem.Mol
        RDKit molecule object representing the chemical structure of the molecule, [-]

    Returns
    -------
    V_vdw : float
        Van der Waals volume, [m^3/mol]
    A_vdw : float
        Van der Waals surface area, [m^2/mol]

    Notes
    -----
    .. warning::
        The corrections have not been implemented and this function has not yet undergone
        validation.

    Examples
    --------
    >>> from rdkit import Chem # doctest:+SKIP
    >>> from thermo import Chemical # doctest:+SKIP
    >>> mol = Chemical('decane').rdkitmol # doctest:+SKIP
    >>> bondi_van_der_waals_surface_area_volume(mol) # doctest:+SKIP
    (0.00010918, 1504000.0)

    References
    ----------
    .. [1] Bondi, A. "Van Der Waals Volumes and Radii." The Journal of Physical
       Chemistry 68, no. 3 (March 1, 1964): 441-451. https://doi.org/10.1021/j100785a001.
    .. [2] Horvath, ARI L., ed. "Chapter 3 - Relationships between Structure and Properties."
       In Studies in Physical and Theoretical Chemistry, 75:575-860. Molecular Design. 
       Elsevier, 1992. https://doi.org/10.1016/B978-0-444-89217-1.50007-7.
    """
    from thermo.group_contribution.group_contribution_base import smarts_fragment_priority
    assignment, _, _, success, status = smarts_fragment_priority(catalog=catalog, rdkitmol=rdkitmol)
    V_vdw = 0.0
    A_vdw = 0.0

    # Iterate over the fragmentation assignment
    for group_id, count in assignment.items():
        group = BONDI_GROUPS_BY_ID.get(group_id)
        if group:
            # Update R and Q using the group's volume and polarizability contributions
            V_vdw += group.Vw * count
            if group.Aw is not None:
                A_vdw += group.Aw * count
    # TODO: Corrections; all main groups have been implemented

    V_vdw *= 1e-6   # Convert R from cm^3/mol to m^3/mol
    A_vdw *= 0.0001*1e9  # Convert Q from 1e9 cm^2/mol to m^2/mol
    return V_vdw, A_vdw


def R_Q_from_bondi(rdkitmol):
    V_vdw, A_vdw = bondi_van_der_waals_surface_area_volume(rdkitmol)
    R = R_from_Van_der_Waals_volume(V_vdw)
    Q = Q_from_Van_der_Waals_area(A_vdw)
    return R, Q


def R_from_Van_der_Waals_volume(V_vdw):
    r'''Calculates the UNIFAC R parameter from a species' Van der Waals molar volume.

    .. math::
        R_k = \frac{V_{wk}}{15.17}

    Parameters
    ----------
    V_vdw : float
        Unnormalized Van der Waals volume, [m^3/mol]

    Returns
    -------
    R : float
        R UNIFAC parameter (normalized Van der Waals Volume)  [-]

    Examples
    --------
    >>> R_from_Van_der_Waals_volume(6.826196599999999e-05)
    4.4998

    Notes
    -----
    This function is the inverse of Van_der_Waals_volume().
    '''
    return V_vdw / 1.517e-05

def Q_from_Van_der_Waals_area(A_vdw):
    r'''Calculates the UNIFAC Q parameter from a species' Van der Waals molar surface area.

    .. math::
        Q_k = \frac{A_{wk}}{2.5 \times 10^9}

    Parameters
    ----------
    A_vdw : float
        Unnormalized Van der Waals surface area, [m^2/mol]

    Returns
    -------
    Q : float
        Q UNIFAC parameter (normalized Van der Waals Area)  [-]

    Examples
    --------
    >>> Q_from_Van_der_Waals_area(964000.0)
    3.856

    Notes
    -----
    This function is the inverse of Van_der_Waals_area().
    '''
    return A_vdw / 250000.0
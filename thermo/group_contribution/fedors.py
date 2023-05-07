'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2022 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains an implementation of the Fedors
group-contribution method.
This functionality requires the RDKit library to work.

.. contents:: :local:


.. autofunction:: thermo.group_contribution.Fedors
'''
__all__ = ['Fedors']

from chemicals.elements import simple_formula_parser

from thermo.functional_groups import alcohol_smarts, amine_smarts, count_rings_attatched_to_rings, smarts_mol_cache

fedors_allowed_atoms = frozenset(['C', 'H', 'O', 'N', 'F', 'Cl', 'Br', 'I', 'S'])
fedors_contributions = {'C': 34.426, 'H': 9.172, 'O': 20.291,
                       'O_alcohol': 18, 'N': 48.855, 'N_amine': 47.422,
                       'F': 22.242, 'Cl': 52.801, 'Br': 71.774, 'I': 96.402,
                       'S': 50.866, '3_ring': -15.824, '4_ring': -17.247,
                       '5_ring': -39.126, '6_ring': -39.508,
                       'double_bond': 5.028, 'triple_bond': 0.7973,
                       'ring_ring_bonds': 35.524}


def Fedors(mol):
    r'''Estimate the critical volume of a molecule
    using the Fedors [1]_ method, which is a basic
    group contribution method that also uses certain
    bond count features and the number of different
    types of rings.

    Parameters
    ----------
    mol : str or rdkit.Chem.rdchem.Mol, optional
        Smiles string representing a chemical or a rdkit molecule, [-]

    Returns
    -------
    Vc : float
        Estimated critical volume, [m^3/mol]
    status : str
        A string holding an explanation of why the molecule failed to be
        fragmented, if it fails; 'OK' if it suceeds, [-]
    unmatched_atoms : bool
        Whether or not all atoms in the molecule were matched successfully;
        if this is True, the results should not be trusted, [-]
    unrecognized_bond : bool
        Whether or not all bonds in the molecule were matched successfully;
        if this is True, the results should not be trusted, [-]
    unrecognized_ring_size : bool
        Whether or not all rings in the molecule were matched successfully;
        if this is True, the results should not be trusted, [-]

    Notes
    -----
    Raises an exception if rdkit is not installed, or `smi` or `rdkitmol` is
    not defined.

    Examples
    --------
    Example for sec-butanol in [2]_:

    >>> Vc, status, _, _, _ = Fedors('CCC(C)O') # doctest:+SKIP
    >>> Vc, status # doctest:+SKIP
    (0.000274024, 'OK')

    References
    ----------
    .. [1] Fedors, R. F. "A Method to Estimate Critical Volumes." AIChE
       Journal 25, no. 1 (1979): 202-202. https://doi.org/10.1002/aic.690250129.
    .. [2] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    from rdkit import Chem
    if type(mol) is Chem.rdchem.Mol:
        rdkitmol = Chem.Mol(mol)
        no_H_mol = mol
    else:
        rdkitmol = Chem.MolFromSmiles(mol)
        no_H_mol = Chem.Mol(rdkitmol)

    # Canont modify the molecule we are given
    rdkitmol = Chem.AddHs(rdkitmol)

    ri = no_H_mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    UNRECOGNIZED_RING_SIZE = False
    three_rings = four_rings = five_rings = six_rings = 0
    for ring in atom_rings:
        ring_size = len(ring)
        if ring_size == 3:
            three_rings += 1
        elif ring_size == 4:
            four_rings += 1
        elif ring_size == 5:
            five_rings += 1
        elif ring_size == 6:
            six_rings += 1
        else:
            UNRECOGNIZED_RING_SIZE = True

    rings_attatched_to_rings = count_rings_attatched_to_rings(no_H_mol, atom_rings=atom_rings)

    UNRECOGNIZED_BOND_TYPE = False
    DOUBLE_BOND = Chem.rdchem.BondType.DOUBLE
    TRIPLE_BOND = Chem.rdchem.BondType.TRIPLE
    SINGLE_BOND = Chem.rdchem.BondType.SINGLE
    AROMATIC_BOND = Chem.rdchem.BondType.AROMATIC

    double_bond_count = triple_bond_count = 0
    # GetBonds is very slow; we can make it a little faster by iterating
    # over a copy without hydrogens
    for bond in no_H_mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type is DOUBLE_BOND:
            double_bond_count += 1
        elif bond_type is TRIPLE_BOND:
            triple_bond_count += 1
        elif bond_type is SINGLE_BOND or bond_type is AROMATIC_BOND:
            pass
        else:
            UNRECOGNIZED_BOND_TYPE = True

    alcohol_matches = rdkitmol.GetSubstructMatches(smarts_mol_cache(alcohol_smarts))
    amine_matches = rdkitmol.GetSubstructMatches(smarts_mol_cache(amine_smarts))

    # This was the fastest way to get the atom counts
    atoms = simple_formula_parser(Chem.rdMolDescriptors.CalcMolFormula(rdkitmol))
    # For the atoms with functional groups, they always have to be there
    if 'N' not in atoms:
        atoms['N'] = 0
    if 'O' not in atoms:
        atoms['O'] = 0
    found_atoms = set(atoms.keys())
    UNKNOWN_ATOMS = bool(not found_atoms.issubset(fedors_allowed_atoms))

    atoms['O_alcohol'] = len(alcohol_matches)
    atoms['O'] -= len(alcohol_matches)
    atoms['N_amine'] = len(amine_matches)
    atoms['N'] -= len(amine_matches)
    atoms['3_ring'] = three_rings
    atoms['4_ring'] = four_rings
    atoms['5_ring'] = five_rings
    atoms['6_ring'] = six_rings
    atoms['double_bond'] = double_bond_count
    atoms['triple_bond'] = triple_bond_count
    atoms['ring_ring_bonds'] = rings_attatched_to_rings

#     print(atoms)
    Vc = 26.6
    for k, v in fedors_contributions.items():
        try:
            Vc += atoms[k]*v
        except KeyError:
            pass

    Vc *= 1e-6

    status = 'errors found' if (UNKNOWN_ATOMS or UNRECOGNIZED_BOND_TYPE or UNRECOGNIZED_RING_SIZE) else 'OK'

    return Vc, status, UNKNOWN_ATOMS, UNRECOGNIZED_BOND_TYPE, UNRECOGNIZED_RING_SIZE


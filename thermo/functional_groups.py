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

This module contains various methods for identifying functional groups in
molecules. This functionality requires the RDKit library to work.

For submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.


.. contents:: :local:

------------------------------------
Specific molecule matching functions
------------------------------------

.. autofunction:: thermo.functional_groups.is_organic
.. autofunction:: thermo.functional_groups.is_inorganic
.. autofunction:: thermo.functional_groups.is_radionuclide

------------------
Hydrocarbon Groups
------------------

.. autofunction:: thermo.functional_groups.is_hydrocarbon
.. autofunction:: thermo.functional_groups.is_alkane
.. autofunction:: thermo.functional_groups.is_cycloalkane
.. autofunction:: thermo.functional_groups.is_branched_alkane
.. autofunction:: thermo.functional_groups.is_alkene
.. autofunction:: thermo.functional_groups.is_alkyne
.. autofunction:: thermo.functional_groups.is_aromatic

-------------
Oxygen Groups
-------------

.. autofunction:: thermo.functional_groups.is_alcohol
.. autofunction:: thermo.functional_groups.is_polyol
.. autofunction:: thermo.functional_groups.is_ketone
.. autofunction:: thermo.functional_groups.is_aldehyde
.. autofunction:: thermo.functional_groups.is_carboxylic_acid
.. autofunction:: thermo.functional_groups.is_ether
.. autofunction:: thermo.functional_groups.is_phenol
.. autofunction:: thermo.functional_groups.is_ester
.. autofunction:: thermo.functional_groups.is_anhydride
.. autofunction:: thermo.functional_groups.is_acyl_halide
.. autofunction:: thermo.functional_groups.is_carbonate
.. autofunction:: thermo.functional_groups.is_carboxylate
.. autofunction:: thermo.functional_groups.is_hydroperoxide
.. autofunction:: thermo.functional_groups.is_peroxide
.. autofunction:: thermo.functional_groups.is_orthoester
.. autofunction:: thermo.functional_groups.is_methylenedioxy
.. autofunction:: thermo.functional_groups.is_orthocarbonate_ester
.. autofunction:: thermo.functional_groups.is_carboxylic_anhydride

---------------
Nitrogen Groups
---------------
.. autofunction:: thermo.functional_groups.is_amide
.. autofunction:: thermo.functional_groups.is_amidine
.. autofunction:: thermo.functional_groups.is_amine
.. autofunction:: thermo.functional_groups.is_primary_amine
.. autofunction:: thermo.functional_groups.is_secondary_amine
.. autofunction:: thermo.functional_groups.is_tertiary_amine
.. autofunction:: thermo.functional_groups.is_quat
.. autofunction:: thermo.functional_groups.is_imine
.. autofunction:: thermo.functional_groups.is_primary_ketimine
.. autofunction:: thermo.functional_groups.is_secondary_ketimine
.. autofunction:: thermo.functional_groups.is_primary_aldimine
.. autofunction:: thermo.functional_groups.is_secondary_aldimine
.. autofunction:: thermo.functional_groups.is_imide
.. autofunction:: thermo.functional_groups.is_azide
.. autofunction:: thermo.functional_groups.is_azo
.. autofunction:: thermo.functional_groups.is_cyanate
.. autofunction:: thermo.functional_groups.is_isocyanate
.. autofunction:: thermo.functional_groups.is_nitrate
.. autofunction:: thermo.functional_groups.is_nitrile
.. autofunction:: thermo.functional_groups.is_isonitrile
.. autofunction:: thermo.functional_groups.is_nitrite
.. autofunction:: thermo.functional_groups.is_nitro
.. autofunction:: thermo.functional_groups.is_nitroso
.. autofunction:: thermo.functional_groups.is_oxime
.. autofunction:: thermo.functional_groups.is_pyridyl
.. autofunction:: thermo.functional_groups.is_carbamate
.. autofunction:: thermo.functional_groups.is_cyanide

-------------
Sulfur Groups
-------------

.. autofunction:: thermo.functional_groups.is_mercaptan
.. autofunction:: thermo.functional_groups.is_sulfide
.. autofunction:: thermo.functional_groups.is_disulfide
.. autofunction:: thermo.functional_groups.is_sulfoxide
.. autofunction:: thermo.functional_groups.is_sulfone
.. autofunction:: thermo.functional_groups.is_sulfinic_acid
.. autofunction:: thermo.functional_groups.is_sulfonic_acid
.. autofunction:: thermo.functional_groups.is_sulfonate_ester
.. autofunction:: thermo.functional_groups.is_thiocyanate
.. autofunction:: thermo.functional_groups.is_isothiocyanate
.. autofunction:: thermo.functional_groups.is_thioketone
.. autofunction:: thermo.functional_groups.is_thial
.. autofunction:: thermo.functional_groups.is_carbothioic_s_acid
.. autofunction:: thermo.functional_groups.is_carbothioic_o_acid
.. autofunction:: thermo.functional_groups.is_thiolester
.. autofunction:: thermo.functional_groups.is_thionoester
.. autofunction:: thermo.functional_groups.is_carbodithioic_acid
.. autofunction:: thermo.functional_groups.is_carbodithio

--------------
Silicon Groups
--------------

.. autofunction:: thermo.functional_groups.is_siloxane
.. autofunction:: thermo.functional_groups.is_silyl_ether

------------
Boron Groups
------------

.. autofunction:: thermo.functional_groups.is_boronic_acid
.. autofunction:: thermo.functional_groups.is_boronic_ester
.. autofunction:: thermo.functional_groups.is_borinic_acid
.. autofunction:: thermo.functional_groups.is_borinic_ester

-----------------
Phosphorus Groups
-----------------

.. autofunction:: thermo.functional_groups.is_phosphine
.. autofunction:: thermo.functional_groups.is_phosphonic_acid
.. autofunction:: thermo.functional_groups.is_phosphodiester
.. autofunction:: thermo.functional_groups.is_phosphate

--------------
Halogen Groups
--------------

.. autofunction:: thermo.functional_groups.is_haloalkane
.. autofunction:: thermo.functional_groups.is_fluoroalkane
.. autofunction:: thermo.functional_groups.is_chloroalkane
.. autofunction:: thermo.functional_groups.is_bromoalkane
.. autofunction:: thermo.functional_groups.is_iodoalkane

--------------------
Organometalic Groups
--------------------

.. autofunction:: thermo.functional_groups.is_alkyllithium
.. autofunction:: thermo.functional_groups.is_alkylaluminium
.. autofunction:: thermo.functional_groups.is_alkylmagnesium_halide

------------
Other Groups
------------

.. autofunction:: thermo.functional_groups.is_acid

-----------------
Utility functions
-----------------

.. autofunction:: thermo.functional_groups.count_ring_ring_attatchments
.. autofunction:: thermo.functional_groups.count_rings_attatched_to_rings
.. autofunction:: thermo.functional_groups.benene_rings

------------------------------------
Functions using group identification
------------------------------------

.. autofunction:: thermo.functional_groups.BVirial_Tsonopoulos_extended_ab



'''

from chemicals.elements import simple_formula_parser

group_names = ['mercaptan', 'sulfide', 'disulfide', 'sulfoxide', 'sulfone',
               'sulfinic_acid', 'sulfonic_acid', 'sulfonate_ester', 'thiocyanate',
               'isothiocyanate', 'thioketone', 'thial', 'carbothioic_s_acid',
               'carbothioic_o_acid', 'thiolester', 'thionoester',
               'carbodithioic_acid', 'carbodithio', 'siloxane',
               'branched_alkane', 'alkane', 'cycloalkane', 'alkene', 'alkyne',
               'aromatic', 'nitrile', 'carboxylic_acid', 'haloalkane',
               'fluoroalkane', 'chloroalkane', 'bromoalkane', 'iodoalkane',
               'amine', 'primary_amine', 'secondary_amine', 'tertiary_amine',
               'quat', 'amide', 'nitro', 'amidine', 'imine', 'primary_ketimine',
               'secondary_ketimine', 'primary_aldimine', 'secondary_aldimine',
               'imide', 'azide', 'azo', 'cyanate', 'isocyanate', 'nitrate',
               'isonitrile', 'nitrite', 'nitroso', 'oxime', 'pyridyl',
               'carbamate', 'acyl_halide', 'alcohol', 'polyol', 'acid',
               'ketone', 'aldehyde', 'anhydride', 'ether', 'phenol', 'carbonate',
               'carboxylate', 'hydroperoxide', 'peroxide', 'orthoester',
               'methylenedioxy', 'orthocarbonate_ester', 'carboxylic_anhydride',
               'ester', 'boronic_acid', 'boronic_ester', 'borinic_acid',
               'borinic_ester', 'phosphine', 'phosphonic_acid', 'phosphodiester',
               'phosphate', 'alkyllithium', 'alkylmagnesium_halide',
               'alkylaluminium', 'silyl_ether', 'organic', 'inorganic',
               'is_hydrocarbon', 'cyanide']
__all__ = [# sulfur
           'is_mercaptan', 'is_sulfide', 'is_disulfide', 'is_sulfoxide',
           'is_sulfone', 'is_sulfinic_acid', 'is_sulfonic_acid',
           'is_sulfonate_ester', 'is_thiocyanate', 'is_isothiocyanate',
           'is_thioketone', 'is_thial', 'is_carbothioic_s_acid',
           'is_carbothioic_o_acid', 'is_thiolester', 'is_thionoester',
           'is_carbodithioic_acid', 'is_carbodithio',

           'is_siloxane',

           # other
           'is_hydrocarbon', 'is_branched_alkane',
           'is_alkane', 'is_cycloalkane', 'is_alkene',
           'is_alkyne', 'is_aromatic', 'is_nitrile', 'is_carboxylic_acid',
           'is_haloalkane', 'is_fluoroalkane', 'is_chloroalkane',
           'is_bromoalkane', 'is_iodoalkane',

           # Nitrogen
           'is_amine', 'is_primary_amine', 'is_secondary_amine',
           'is_tertiary_amine', 'is_quat',
           'is_amide', 'is_nitro', 'is_amidine', 'is_imine',
           'is_primary_ketimine',
           'is_secondary_ketimine', 'is_primary_aldimine',
           'is_secondary_aldimine', 'is_imide', 'is_azide', 'is_azo',
           'is_cyanate', 'is_isocyanate', 'is_nitrate', 'is_isonitrile',
           'is_nitrite', 'is_nitroso', 'is_oxime', 'is_pyridyl',
           'is_carbamate', 'is_cyanide',

           # oxygen
           'is_acyl_halide', 'is_alcohol', 'is_polyol',
           'is_acid', 'is_ketone', 'is_aldehyde', 'is_anhydride',
           'is_ether', 'is_phenol', 'is_carbonate', 'is_carboxylate',
           'is_hydroperoxide', 'is_peroxide', 'is_orthoester',
           'is_methylenedioxy', 'is_orthocarbonate_ester',
           'is_carboxylic_anhydride', 'is_ester',

           'is_boronic_acid', 'is_boronic_ester', 'is_borinic_acid',
           'is_borinic_ester',

           'is_phosphine', 'is_phosphonic_acid', 'is_phosphodiester',
           'is_phosphate',

           'is_alkyllithium', 'is_alkylmagnesium_halide', 'is_alkylaluminium',
           'is_silyl_ether',

           'is_organic', 'is_inorganic', 'is_radionuclide',

           'count_ring_ring_attatchments',
           'count_rings_attatched_to_rings',
           'benene_rings',
           'group_names',

           'BVirial_Tsonopoulos_extended_ab']


rdkit_missing = 'RDKit is not installed; it is required to use this functionality'

loaded_rdkit = False
Chem = Descriptors = AllChem = rdMolDescriptors = CanonSmiles = MolToSmiles = MolFromSmarts = None
def load_rdkit_modules():
    global loaded_rdkit, Chem, Descriptors, AllChem, rdMolDescriptors, CanonSmiles, MolToSmiles, MolFromSmarts
    if loaded_rdkit:
        return
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, CanonSmiles, Descriptors, MolFromSmarts, MolToSmiles, rdMolDescriptors
        loaded_rdkit = True
    except:
        raise Exception(rdkit_missing) # pragma: no cover




def substructures_are_entire_structure(mol, matches, exclude_Hs=True):
    atomIdxs = {atom.GetIdx() for atom in mol.GetAtoms() if (not exclude_Hs or atom.GetAtomicNum() != 1)}
    matched_atoms = []
    for h in matches:
        matched_atoms.extend(h)
    return set(matched_atoms) == atomIdxs




mol_smarts_cache = {}
def smarts_mol_cache(smarts):
    try:
        return mol_smarts_cache[smarts]
    except:
        pass
    if not loaded_rdkit:
        load_rdkit_modules()
    mol = MolFromSmarts(smarts)
    mol_smarts_cache[smarts] = mol
    return mol

amide_smarts_3 = '[NX3][CX3](=[OX1])[#6]'
amide_smarts_2 = 'O=C([c,CX4])[$([NH2]),$([NH][c,CX4]),$(N([c,CX4])[c,CX4])]'
amide_smarts_1 = '[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]'
amide_smarts_4 = '[*][CX3](=[OX1H0])[NX3]([*])([*])' # Doesn't match ones without H
amide_smarts_collection = [amide_smarts_3, amide_smarts_2, amide_smarts_1, amide_smarts_4]
def is_amide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule has a amide RC(=O)NR`R″ group.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_amide : bool
        Whether or not the compound is a amide or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_amide(MolFromSmiles('CN(C)C=O')) # doctest:+SKIP
    True
    '''
    for s in amide_smarts_collection:
        hits = mol.GetSubstructMatches(smarts_mol_cache(s))
        if len(hits):
            return True
    return False

amidine_smarts_1 = '[*][NX2]=[CX3H0]([*])[NX3]([*])([*])'
amidine_smarts_2 = '[NX2H1]=[CX3H0]([*])[NX3]([*])([*])'
amidine_smarts_3 = '[NX2H1]=[CX3H0]([*])[NX3H1]([*])'
amidine_smarts_4 = '[NX2H1]=[CX3H0]([*])[NX3H2]'
amidine_smarts_5 = '[*][NX2]=[CX3H0]([*])[NX3H1]([*])'
amidine_smarts_6 = '[*][NX2]=[CX3H0]([*])[NX3H2]'
amidine_smarts_collection = [amidine_smarts_1, amidine_smarts_2,
                             amidine_smarts_3, amidine_smarts_4,
                             amidine_smarts_5, amidine_smarts_6]

def is_amidine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule has a amidine RC(NR)NR2 group.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_amidine : bool
        Whether or not the compound is a amidine or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_amidine(MolFromSmiles('C1=CC(=CC=C1C(=N)N)OCCCCCOC2=CC=C(C=C2)C(=N)N')) # doctest:+SKIP
    True
    '''
    for s in amidine_smarts_collection:
        hits = mol.GetSubstructMatches(smarts_mol_cache(s))
        if len(hits):
            return True
    return False

primary_ketimine_smarts = '[*][CX3H0](=[NX2H1])([*])'

def is_primary_ketimine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a primary ketimine.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_primary_ketimine : bool
        Whether or not the compound is a primary ketimine, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_primary_ketimine(MolFromSmiles('C1=CC=C(C=C1)C(=N)C2=CC=CC=C2')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(primary_ketimine_smarts))
    return bool(matches)


secondary_ketimine_smarts = '[*][CX3H0]([*])=[NX2H0]([*])'

def is_secondary_ketimine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a secondary ketimine.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_secondary_ketimine : bool
        Whether or not the compound is a secondary ketimine, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_secondary_ketimine(MolFromSmiles('CC(C)CC(=NC1=CC=C(C=C1)CC2=CC=C(C=C2)N=C(C)CC(C)C)C')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(secondary_ketimine_smarts))
    return bool(matches)

primary_aldimine_smarts = '[*][CX3H1]=[NX2H1]'

def is_primary_aldimine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a primary aldimine.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_primary_aldimine : bool
        Whether or not the compound is a primary aldimine, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_primary_aldimine(MolFromSmiles('CC=N')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(primary_aldimine_smarts))
    return bool(matches)

secondary_aldimine_smarts = '[*][CX3H1]=[NX2H0]'

def is_secondary_aldimine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a secondary aldimine.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_secondary_aldimine : bool
        Whether or not the compound is a secondary aldimine, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_secondary_aldimine(MolFromSmiles( 'C1=CC=C(C=C1)/C=N\\O')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(secondary_aldimine_smarts))
    return bool(matches)

def is_imine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a imine.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_imine : bool
        Whether or not the compound is a imine, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_imine(MolFromSmiles('C1=CC=C(C=C1)C(=N)C2=CC=CC=C2')) # doctest:+SKIP
    True
    '''
    return is_primary_ketimine(mol) or is_secondary_ketimine(mol) or is_primary_aldimine(mol) or is_secondary_aldimine(mol)

mercaptan_smarts = '[#16X2H]'
def is_mercaptan(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule has a mercaptan R-SH group. This is also called a thiol.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_mercaptan : bool
        Whether or not the compound is a mercaptan or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_mercaptan(MolFromSmiles("CS")) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/a9b45f9cc6f17d3b5649b77a81f535dfe0729a84fc3ac453c9aaa60286e6
    hits = mol.GetSubstructMatches(smarts_mol_cache(mercaptan_smarts))
    return len(hits) > 0

sulfide_smarts = '[!#16][#16X2H0][!#16]'
def is_sulfide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a sulfide. This group excludes disulfides.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_sulfide : bool
        Whether or not the compound is a sulfide, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_sulfide(MolFromSmiles('CSC')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(sulfide_smarts))
    return bool(matches)

disulfide_smarts = '[#16X2H0][#16X2H0]'
def is_disulfide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a disulfide.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_disulfide : bool
        Whether or not the compound is a disulfide, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_disulfide(MolFromSmiles('CSSC')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(disulfide_smarts))
    return bool(matches)

sulfoxide_smarts = '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]'
def is_sulfoxide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a sulfoxide.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_sulfoxide : bool
        Whether or not the compound is a sulfoxide, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_sulfoxide(MolFromSmiles('CS(=O)C')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(sulfoxide_smarts))
    return bool(matches)

sulfone_smarts = '[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]'
def is_sulfone(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a sulfone.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_sulfone : bool
        Whether or not the compound is a sulfone, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_sulfone(MolFromSmiles('CS(=O)(=O)C')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(sulfone_smarts))
    return bool(matches)

sulfinic_acid_smarts = '[SX3H0](=O)([OX2H])[!H]'
def is_sulfinic_acid(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a sulfinic acid.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_sulfinic_acid : bool
        Whether or not the compound is a sulfinic acid, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_sulfinic_acid(MolFromSmiles('O=S(O)CCN')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(sulfinic_acid_smarts))
    return bool(matches)

sulfonic_acid_smarts = '[SX4H0](=O)(=O)([OX2H])[!H]'

def is_sulfonic_acid(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a sulfonic acid.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_sulfonic_acid : bool
        Whether or not the compound is a sulfonic acid, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_sulfonic_acid(MolFromSmiles('OS(=O)(=O)c1ccccc1')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(sulfonic_acid_smarts))
    return bool(matches)

sulfonate_ester_smarts = '[SX4H0](=O)(=O)([OX2H0])[!H]'
def is_sulfonate_ester(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a sulfonate ester.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_sulfonate_ester : bool
        Whether or not the compound is a sulfonate ester, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_sulfonate_ester(MolFromSmiles('COS(=O)(=O)C(F)(F)F')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(sulfonate_ester_smarts))
    return bool(matches)

thiocyanate_smarts = '[SX2H0]([!H])[CH0]#[NX1H0]'
def is_thiocyanate(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a thiocyanate.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_thiocyanate : bool
        Whether or not the compound is a thiocyanate, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_thiocyanate(MolFromSmiles('C1=CC=C(C=C1)SC#N')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(thiocyanate_smarts))
    return bool(matches)

isothiocyanate_smarts = '[!H][NX2H0]=[CX2H0]=[SX1H0]'

def is_isothiocyanate(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a isothiocyanate.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_isothiocyanate : bool
        Whether or not the compound is a isothiocyanate, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_isothiocyanate(MolFromSmiles('C=CCN=C=S')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(isothiocyanate_smarts))
    return bool(matches)

thioketone_smarts = '[#6X3;H0]([!H])([!H])=[SX1H0]'

def is_thioketone(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a thioketone.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_thioketone : bool
        Whether or not the compound is a thioketone, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_thioketone(MolFromSmiles('C1=CC=C(C=C1)C(=S)C2=CC=CC=C2')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(thioketone_smarts))
    return bool(matches)

thial_smarts = '[#6X3;H1](=[SX1H0])([!H])'

def is_thial(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a thial.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_thial : bool
        Whether or not the compound is a thial, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_thial(MolFromSmiles('CC=S')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(thial_smarts))
    return bool(matches)


carbothioic_s_acid_smarts = '[#6X3;H0](=[OX1H0])([SX2H1])([!H])'

def is_carbothioic_s_acid(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a Carbothioic S-acid.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_carbothioic_s_acid : bool
        Whether or not the compound is a Carbothioic S-acid, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_carbothioic_s_acid(MolFromSmiles('C1=CC=C(C=C1)C(=O)S')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(carbothioic_s_acid_smarts))
    return bool(matches)


carbothioic_o_acid_smarts = '[#6X3;H0]([OX2H1])(=[SX1H0])([!H])'

def is_carbothioic_o_acid(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a Carbothioic S-acid.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_carbothioic_o_acid : bool
        Whether or not the compound is a Carbothioic S-acid, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_carbothioic_o_acid(MolFromSmiles('OC(=S)c1ccccc1O')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(carbothioic_o_acid_smarts))
    return bool(matches)


thiolester_smarts = '[#6X3;H0](=[OX1H0])([*])[SX2H0][!H]'

def is_thiolester(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a thiolester.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_thiolester : bool
        Whether or not the compound is a thiolester, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_thiolester(MolFromSmiles('CSC(=O)C=C')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(thiolester_smarts))
    return bool(matches)

thionoester_smarts = '[#6X3;H0](=[SX1H0])([*])[OX2H0][!H]'

def is_thionoester(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a thionoester.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_thionoester : bool
        Whether or not the compound is a thionoester, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_thionoester(MolFromSmiles('CCOC(=S)S')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(thionoester_smarts))
    return bool(matches)

carbodithioic_acid_smarts = '[#6X3;H0](=[SX1H0])([!H])[SX2H1]'

def is_carbodithioic_acid(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a carbodithioic acid .

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_carbodithioic_acid : bool
        Whether or not the compound is a carbodithioic acid , [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_carbodithioic_acid(MolFromSmiles('C1=CC=C(C=C1)C(=S)S')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(carbodithioic_acid_smarts))
    return bool(matches)

carbodithio_smarts = '[#6X3;H0](=[SX1H0])([!H])[SX2H0]([!H])'

def is_carbodithio(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a carbodithio.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_carbodithio : bool
        Whether or not the compound is a carbodithio, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_carbodithio(MolFromSmiles('C(=S)(N)SSC(=S)N')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(carbodithio_smarts))
    return bool(matches)



siloxane_smarts = '[Si][O][Si]'
def is_siloxane(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a siloxane.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_siloxane : bool
        Whether or not the compound is a siloxane, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_siloxane(MolFromSmiles('C[Si]1(O[Si](O[Si](O[Si](O1)(C)C)(C)C)(C)C)C')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(siloxane_smarts))
    return bool(matches)

def is_hydrocarbon(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is an hydrocarbon (molecule containing hydrogen and carbon only)

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_hydrocarbon : bool
        Whether or not the compound is a hydrocarbon or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_hydrocarbon(MolFromSmiles("CCC")) # doctest:+SKIP
    True
    '''
    if not loaded_rdkit:
        load_rdkit_modules()
    # Check the hardcoded list first
    formula = rdMolDescriptors.CalcMolFormula(mol)
    atoms = simple_formula_parser(formula)
    return ('C' in atoms and 'H' in atoms and len(atoms) == 2)

alkane_smarts = '[CX4]'
def is_alkane(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is an alkane, also refered to as a paraffin. All bonds in the
    molecule must be single carbon-carbon or carbon-hydrogen.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_alkane : bool
        Whether or not the compound is an alkane or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_alkane(MolFromSmiles("CCC")) # doctest:+SKIP
    True
    '''
    # Also parafins
    # Is all single carbon bonds and hydrogens
    # https://smarts.plus/smartsview/66ef20801e42ff2c04658f07ee3c5858864478fe570cfb1813c739c8f15e
    matches = mol.GetSubstructMatches(smarts_mol_cache(alkane_smarts))
    return substructures_are_entire_structure(mol, matches)

def is_cycloalkane(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a cycloalkane, also refered to as a naphthenes.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_cycloalkane : bool
        Whether or not the compound is a cycloalkane or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_cycloalkane(MolFromSmiles('C1CCCCCCCCC1')) # doctest:+SKIP
    True
    '''
    # naphthenes are a synonym
    if is_alkane(mol):
        ri = mol.GetRingInfo()
        if len(ri.AtomRings()):
            return True
    return False

alkene_smarts = '[CX3]=[CX3]'
def is_alkene(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is an alkene. Alkenes are also refered to as olefins.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_alkene : bool
        Whether or not the compound is a alkene or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_alkene(MolFromSmiles('C=C')) # doctest:+SKIP
    True
    '''
    # Has at least one double carbon bond
    # https://smarts.plus/smartsview/fd64df5c2208477a0750fd0a35a3a7d7df9f59273b972dd639eb868f267e
    matches = mol.GetSubstructMatches(smarts_mol_cache(alkene_smarts))
    return bool(matches)


alkyne_smarts = '[CX2]#C'
def is_alkyne(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is an alkyne.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_alkyne : bool
        Whether or not the compound is a alkyne or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_alkyne(MolFromSmiles('CC#C')) # doctest:+SKIP
    True
    '''
    # Has at least one triple carbon bond
    # https://smarts.plus/smartsview/f1d85ad46c8ad7419fdb4c6cb1908d4ca5ea2fcd89ef5d0a6e07c5b799fb
    matches = mol.GetSubstructMatches(smarts_mol_cache(alkyne_smarts))
    return bool(matches)

aromatic_smarts = '[c]'

def is_aromatic(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is aromatic.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_aromatic : bool
        Whether or not the compound is aromatic or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_aromatic(MolFromSmiles('CC1=CC=CC=C1C')) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/6011201068306235ac861ddaa794a4559576e23361a5437373562ae3cc45
    matches = mol.GetSubstructMatches(smarts_mol_cache(aromatic_smarts))
    return bool(matches)

boronic_acid_smarts = '[BX3]([OX2H])([OX2H])'

def is_boronic_acid(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule has any boronic acid functional groups.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_boronic_acid : bool
        Whether or not the compound is an boronic acid, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_boronic_acid(MolFromSmiles('B(C)(O)O')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(boronic_acid_smarts))
    return bool(matches)

boronic_ester_smarts = '[BX3;H0]([OX2H0])([OX2H0])[!O@!H]'

def is_boronic_ester(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a boronic ester.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_boronic_ester : bool
        Whether or not the compound is a boronic ester, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_boronic_ester(MolFromSmiles('B(C)(OC(C)C)OC(C)C')) # doctest:+SKIP
    True
    '''
    hits = mol.GetSubstructMatches(smarts_mol_cache(boronic_ester_smarts))
    return len(hits) > 0

borinic_acid_smarts = '[BX3;H0]([OX2H1])([!O])[!O]'
borinic_acid_smarts_H1 = '[BX3;H1]([OX2H1])[!O]'
borinic_acid_smarts_H2 = '[BX3;H2][OX2H1]'
borinic_acid_smarts = (borinic_acid_smarts, borinic_acid_smarts_H1, borinic_acid_smarts_H2)
def is_borinic_acid(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a borinic acid.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_borinic_acid : bool
        Whether or not the compound is a borinic acid, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_borinic_acid(MolFromSmiles('BO')) # doctest:+SKIP
    True
    '''
    for s in borinic_acid_smarts:
        hits = mol.GetSubstructMatches(smarts_mol_cache(s))
        if len(hits) > 0:
            return True
    return False

borinic_ester_smarts = '[BX3;H0]([OX2H0])([!O])[!O]'

def is_borinic_ester(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a borinic ester.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_borinic_ester : bool
        Whether or not the compound is a borinic ester, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_borinic_ester(MolFromSmiles('B(C1=CC=CC=C1)(C2=CC=CC=C2)OCCN')) # doctest:+SKIP
    True
    '''
    hits = mol.GetSubstructMatches(smarts_mol_cache(borinic_ester_smarts))
    return len(hits) > 0

alcohol_smarts = '[#6][OX2H]'

def is_alcohol(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule any alcohol functional groups.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_alcohol : bool
        Whether or not the compound is an alcohol, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_alcohol(MolFromSmiles('CCO')) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/644c9799417838a70f3c8d126b1d20a3bd96e9a634742bff7f3b67fcaa0a
    matches = mol.GetSubstructMatches(smarts_mol_cache(alcohol_smarts))
    return bool(matches)

def is_polyol(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a polyol (more than 1 alcohol functional groups).

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_polyol : bool
        Whether or not the compound is a polyol, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_polyol(MolFromSmiles('C(C(CO)O)O')) # doctest:+SKIP
    True
    '''
    # More than one alcohol group
    matches = mol.GetSubstructMatches(smarts_mol_cache(alcohol_smarts))
    return len(matches) > 1

acid_smarts = '[!H0;F,Cl,Br,I,N+,$([OH]-*=[!#6]),+]'
def is_acid(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is an acid.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_acid : bool
        Whether or not the compound is a acid, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_acid(MolFromSmiles('CC(=O)O')) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/4878307fc4fa813db953aba8a928e809a256f7cc0080c7c28ebf944e3ce9
    matches = mol.GetSubstructMatches(smarts_mol_cache(acid_smarts))
    return bool(matches)

ketone_smarts = '[#6][CX3](=O)[#6]'
def is_ketone(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a ketone.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_ketone : bool
        Whether or not the compound is a ketone, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_ketone(MolFromSmiles('C1CCC(=O)CC1')) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/947488b11c90e968703137c3faa9ae07a6536644f1de63ce48772e2680c8
    matches = mol.GetSubstructMatches(smarts_mol_cache(ketone_smarts))
    return bool(matches)

aldehyde_smarts = '[CX3H1](=O)[#6]'

def is_aldehyde(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is an aldehyde.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_aldehyde : bool
        Whether or not the compound is an aldehyde, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_aldehyde(MolFromSmiles('C=O')) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/8c5ed80db53e19cc40dcfef58453d90fec96e18d8b7f602d34ff1e3a566c
    matches = mol.GetSubstructMatches(smarts_mol_cache(aldehyde_smarts))
    return bool(matches) or CanonSmiles(MolToSmiles(mol)) == 'C=O'


acyl_halide_smarts = '[#6X3;H0](=[OX1H0])([FX1,ClX1,BrX1,IX1])[!H]'

def is_acyl_halide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a acyl halide.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_acyl_halide : bool
        Whether or not the compound is a acyl halide, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_acyl_halide(MolFromSmiles('C(CCC(=O)Cl)CC(=O)Cl')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(acyl_halide_smarts))
    return bool(matches)

carbonate_smarts =  '[!H][OX2H0][CX3H0](=[OX1H0])[OX2H0][!H]'

def is_carbonate(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a carbonate.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_carbonate : bool
        Whether or not the compound is a carbonate, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_carbonate(MolFromSmiles('C(=O)(OC(Cl)(Cl)Cl)OC(Cl)(Cl)Cl')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(carbonate_smarts))
    return bool(matches)

carboxylate_smarts =  '[C][C](=[OX1H0])[O-X1H0]'

def is_carboxylate(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a carboxylate.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_carboxylate : bool
        Whether or not the compound is a carboxylate, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_carboxylate(MolFromSmiles('CC(=O)[O-].[Na+]')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(carboxylate_smarts))
    return bool(matches)

hydroperoxide_smarts = '[!H][OX2H0][OX2H1]'

def is_hydroperoxide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a hydroperoxide.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_hydroperoxide : bool
        Whether or not the compound is a hydroperoxide, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_hydroperoxide(MolFromSmiles('CC(C)(C)OO')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(hydroperoxide_smarts))
    return bool(matches)


peroxide_smarts = '[!H][OX2H0][OX2H0][!H]'

def is_peroxide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a peroxide.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_peroxide : bool
        Whether or not the compound is a peroxide, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_peroxide(MolFromSmiles('CC(C)(C)OOC(C)(C)C')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(peroxide_smarts))
    return bool(matches) or CanonSmiles(MolToSmiles(mol)) == 'OO'

orthoester_smarts = '[*][CX4]([OX2H0])([OX2H0])([OX2H0])'

def is_orthoester(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a orthoester.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_orthoester : bool
        Whether or not the compound is a orthoester, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_orthoester(MolFromSmiles('CCOC(C)(OCC)OCC')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(orthoester_smarts))
    return bool(matches)

orthocarbonate_ester_smarts = '[CX4H0]([OX2])([OX2])([OX2])([OX2])'

def is_orthocarbonate_ester (mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a orthocarbonate ester .

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_orthocarbonate_ester  : bool
        Whether or not the compound is a orthocarbonate ester , [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_orthocarbonate_ester (MolFromSmiles('COC(OC)(OC)OC') # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(orthocarbonate_ester_smarts))
    return bool(matches)

carboxylic_anhydride_smarts = '[*][CX3H0](=[OX1H0])[OX2H0][CX3H0](=[OX1H0])[*]'

def is_carboxylic_anhydride(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a carboxylic anhydride .

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_carboxylic_anhydride  : bool
        Whether or not the compound is a carboxylic anhydride, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_carboxylic_anhydride (MolFromSmiles('CCCC(=O)OC(=O)CCC') # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(carboxylic_anhydride_smarts))
    return bool(matches)


methylenedioxy_smarts = '[CX4H2;R]([OX2H0;R])([OX2H0;R])'

def is_methylenedioxy(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a methylenedioxy.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_methylenedioxy : bool
        Whether or not the compound is a methylenedioxy, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_methylenedioxy(MolFromSmiles('C1OC2=CC=CC=C2O1')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(methylenedioxy_smarts))
    return bool(matches)


anhydride_smarts = '[CX3](=[OX1])[OX2][CX3](=[OX1])'

def is_anhydride(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is an anhydride.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_anhydride : bool
        Whether or not the compound is an anhydride, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_anhydride(MolFromSmiles('C1=CC(=O)OC1=O')) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/8d10c9a838fc851160958f7b48e6d07eda26e0e544aaeae2d6b8a065bcd8
    matches = mol.GetSubstructMatches(smarts_mol_cache(anhydride_smarts))
    return bool(matches)

ether_smarts = '[OD2]([#6])[#6]'
def is_ether(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is an ether.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_ether : bool
        Whether or not the compound is an ether, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_ether(MolFromSmiles('CC(C)OC(C)C')) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/1e2e9cdb5e1275b0eb4168ef37600b8cb7e9301e08801ac9ae9a6f4a9729
    matches = mol.GetSubstructMatches(smarts_mol_cache(ether_smarts))
    return bool(matches)

phenol_smarts = '[OX2H][cX3]:[c]'
def is_phenol(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a phenol.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_phenol : bool
        Whether or not the compound is a phenol, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_phenol(MolFromSmiles('CC(=O)NC1=CC=C(C=C1)O')) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/0dbfd2ece438a9b9dec68765cc7e0f76ada4193d45dc1b2de95585b4cf2f
    matches = mol.GetSubstructMatches(smarts_mol_cache(phenol_smarts))
    return bool(matches)

nitrile_smarts = '[NX1]#[CX2]'
def is_nitrile(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a nitrile.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_nitrile : bool
        Whether or not the compound is a nitrile, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_nitrile(MolFromSmiles('CC#N')) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/a04d5a51cd03fd469672f34a226fbd049a5d220d3819992fe210bd6d77a7
    matches = mol.GetSubstructMatches(smarts_mol_cache(nitrile_smarts))
    return bool(matches)

isonitrile_smarts = '[*][N+]#[C-]'

def is_isonitrile(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a isonitrile.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_isonitrile : bool
        Whether or not the compound is a isonitrile, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_isonitrile(MolFromSmiles('C[N+]#[C-]')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(isonitrile_smarts))
    return bool(matches)

imide_smarts = '[CX3H0](=[OX1H0])([*])[NX3][CX3H0](=[OX1H0])[*]'
def is_imide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a imide.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_imide : bool
        Whether or not the compound is a imide, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_imide(MolFromSmiles('C1=CC=C2C(=C1)C(=O)NC2=O')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(imide_smarts))
    return bool(matches)

azide_smarts = '[NX2]=[N+X2H0]=[N-X1H0]'

def is_azide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a azide.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_azide : bool
        Whether or not the compound is a azide, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_azide(MolFromSmiles('C1=CC=C(C=C1)N=[N+]=[N-]')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(azide_smarts))
    return bool(matches)

azo_smarts = '[*][NX2]=[NX2][*]'

def is_azo(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a azo.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_azo : bool
        Whether or not the compound is a azo, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_azo(MolFromSmiles('C1=CC=C(C=C1)N=NC2=CC=CC=C2')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(azo_smarts))
    return bool(matches)

cyanate_smarts = '[*][OX2H0][CX2H0]#[NX1H0]'

def is_cyanate(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a cyanate.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_cyanate : bool
        Whether or not the compound is a cyanate, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_cyanate(MolFromSmiles('COC#N')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(cyanate_smarts))
    return bool(matches)

isocyanate_smarts = '[NX2H0]=[CX2H0]=[OX1H0]'

def is_isocyanate(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a isocyanate.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_isocyanate : bool
        Whether or not the compound is a isocyanate, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_isocyanate(MolFromSmiles('CN=C=O')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(isocyanate_smarts))
    return bool(matches)

cyanide_smarts = 'C#N'

def is_cyanide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule contains a cyanide functional group.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_cyanide : bool
        Whether or not the compound contains a cyanide functional group, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_cyanide(MolFromSmiles('CC#N')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(cyanide_smarts))
    return bool(matches)

nitrate_smarts = '[OX2][N+X3H0](=[OX1H0])[O-X1H0]'

def is_nitrate(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a nitrate.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_nitrate : bool
        Whether or not the compound is a nitrate, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_nitrate(MolFromSmiles('CCCCCO[N+](=O)[O-]')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(nitrate_smarts))
    return bool(matches)

nitro_smarts = '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'
def is_nitro(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a nitro.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_nitro : bool
        Whether or not the compound is a nitro, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_nitro(MolFromSmiles('C[N+](=O)[O-]')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(nitro_smarts))
    return bool(matches)

nitrite_smarts = '[OX2][NX2H0]=[OX1H0]'

def is_nitrite(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a nitrite.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_nitrite : bool
        Whether or not the compound is a nitrite, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_nitrite(MolFromSmiles('CC(C)CCON=O')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(nitrite_smarts))
    return bool(matches)

nitroso_smarts = '[*][NX2]=[OX1H0]'

def is_nitroso(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a nitroso.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_nitroso : bool
        Whether or not the compound is a nitroso, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_nitroso(MolFromSmiles('C1=CC=C(C=C1)N=O')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(nitroso_smarts))
    return bool(matches)

oxime_smarts = '[!H][CX3]([*])=[NX2H0][OX2H1]'

def is_oxime(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a oxime.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_oxime : bool
        Whether or not the compound is a oxime, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_oxime(MolFromSmiles('CC(=NO)C')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(oxime_smarts))
    return bool(matches)

pyridyl_smarts = 'c1ccncc1'

def is_pyridyl(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a pyridyl.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_pyridyl : bool
        Whether or not the compound is a pyridyl, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_pyridyl(MolFromSmiles('CN1CCC[C@H]1C1=CC=CN=C1')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(pyridyl_smarts))
    return bool(matches)

carbamate_smarts = '[OX2][CX3H0](=[OX1H0])[NX3]'

def is_carbamate(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a carbamate.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_carbamate : bool
        Whether or not the compound is a carbamate, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_carbamate(MolFromSmiles('CC(C)OC(=O)NC1=CC(=CC=C1)Cl')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(carbamate_smarts))
    return bool(matches)

carboxylic_acid_smarts = '[CX3](=O)[OX2H1]'

def is_carboxylic_acid(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a carboxylic acid.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_carboxylic_acid : bool
        Whether or not the compound is a carboxylic acid, [-].

    Examples
    --------
    Butyric acid (butter)

    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_carboxylic_acid(MolFromSmiles('CCCC(=O)O')) # doctest:+SKIP
    True
    '''
    #https://smarts.plus/smartsview/a33ad72207a43d56151d62438cef247f6dcd3071fa7e3e944eabc2923e53
    matches = mol.GetSubstructMatches(smarts_mol_cache(carboxylic_acid_smarts))
    return bool(matches)

haloalkane_smarts = '[#6][F,Cl,Br,I]'
def is_haloalkane(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a haloalkane.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_haloalkane : bool
        Whether or not the compound is a haloalkane, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_haloalkane(MolFromSmiles('CCCl')) # doctest:+SKIP
    True
    '''
    # https://smarts.plus/smartsview/d0f0d91e09810af5bf2aba9a8498fe264931efb6559430ca44d46a719211
    matches = mol.GetSubstructMatches(smarts_mol_cache(haloalkane_smarts))
    return bool(matches)

fluoroalkane_smarts = '[#6][F]'
def is_fluoroalkane(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a fluoroalkane.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_fluoroalkane : bool
        Whether or not the compound is a fluoroalkane, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_fluoroalkane(MolFromSmiles('CF')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(fluoroalkane_smarts))
    return bool(matches)

chloroalkane_smarts = '[#6][Cl]'
def is_chloroalkane(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a chloroalkane.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_chloroalkane : bool
        Whether or not the compound is a chloroalkane, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_chloroalkane(MolFromSmiles('CCl')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(chloroalkane_smarts))
    return bool(matches)

bromoalkane_smarts = '[#6][Br]'
def is_bromoalkane(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a bromoalkane.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_bromoalkane : bool
        Whether or not the compound is a bromoalkane, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_bromoalkane(MolFromSmiles('CBr')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(bromoalkane_smarts))
    return bool(matches)

iodoalkane_smarts = '[#6][I]'
def is_iodoalkane(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a iodoalkane.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_iodoalkane : bool
        Whether or not the compound is a iodoalkane, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_iodoalkane(MolFromSmiles('CI')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(iodoalkane_smarts))
    return bool(matches)

primary_amine_smarts = '[CX4][NH2]'
primary_amine_smarts_aliphatic = '[NX3H2+0,NX4H3+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]'
primary_amine_smarts_aromatic = '[NX3H2+0,NX4H3+]c'
def is_primary_amine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a primary amine.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_primary_amine : bool
        Whether or not the compound is a primary amine, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_primary_amine(MolFromSmiles('CN')) # doctest:+SKIP
    True
    '''
    for s in (primary_amine_smarts, primary_amine_smarts_aliphatic, primary_amine_smarts_aromatic):
        matches = mol.GetSubstructMatches(smarts_mol_cache(s))
        if bool(matches):
            return True
    return False

secondary_amine_smarts = '[$([NH]([CX4])[CX4]);!$([NH]([CX4])[CX4][O,N]);!$([NH]([CX4])[CX4][O,N])]'
def is_secondary_amine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a secondary amine.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_secondary_amine : bool
        Whether or not the compound is a secondary amine, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_secondary_amine(MolFromSmiles('CNC')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(secondary_amine_smarts))
    return bool(matches)

tertiary_amine_smarts = '[ND3]([CX4])([CX4])[CX4]'
tertiary_amine_smarts_aliphatic = "[NX3H0+0,NX4H1+;!$([N][!c]);!$([N]*~[#7,#8,#15,#16])]"
tertiary_amine_smarts_aromatic = "[NX3H0+0,NX4H1+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]"
tertiary_amine_smarts_mixed = '[NX3H0+0,NX4H1+;$([N]([c])([C])[#6]);!$([N]*~[#7,#8,#15,#16])]'
tertiary_amine_1 = '[#6]-[#7](-[#6])-[#6]'
def is_tertiary_amine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a tertiary amine.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_tertiary_amine : bool
        Whether or not the compound is a tertiary amine, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_tertiary_amine(MolFromSmiles('CN(C)C')) # doctest:+SKIP
    True
    '''
    for s in (tertiary_amine_smarts, tertiary_amine_smarts_aliphatic, tertiary_amine_smarts_aromatic, tertiary_amine_smarts_mixed, tertiary_amine_1):
        matches = mol.GetSubstructMatches(smarts_mol_cache(s))
        if bool(matches):
            return True
    return False

quat_smarts = '[N+X4]([c,C])([c,C])([c,C])[c,C]'

def is_quat(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a quat.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_quat : bool
        Whether or not the compound is a quat, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_quat(MolFromSmiles('CCCCCCCCCCCCCCCCCC[N+](C)(C)CCCCCCCCCCCCCCCCCC.[Cl-]')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(quat_smarts))
    return bool(matches)

amine_smarts = '[$([NH2][CX4]),$([$([NH]([CX4])[CX4]);!$([NH]([CX4])[CX4][O,N]);!$([NH]([CX4])[CX4][O,N])]),$([ND3]([CX4])([CX4])[CX4])]'
amine_smarts = '[NX3+0,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]'
all_amine_smarts = (amine_smarts, tertiary_amine_smarts, tertiary_amine_smarts_aliphatic,
              tertiary_amine_smarts_aromatic, tertiary_amine_smarts_mixed, tertiary_amine_1,
              primary_amine_smarts, primary_amine_smarts_aliphatic, primary_amine_smarts_aromatic,
              quat_smarts)

def is_amine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a amine.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_amine : bool
        Whether or not the compound is a amine, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_amine(MolFromSmiles('CN')) # doctest:+SKIP
    True
    '''
    for s in all_amine_smarts:
        matches = mol.GetSubstructMatches(smarts_mol_cache(s))
        if bool(matches):
            return True
    return False



phosphine_smarts = '[PX3]'
def is_phosphine(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a phosphine.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_phosphine : bool
        Whether or not the compound is a phosphine, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_phosphine(MolFromSmiles('CCCPC')) # doctest:+SKIP
    True
    '''
    hits = mol.GetSubstructMatches(smarts_mol_cache(phosphine_smarts))
    return len(hits) > 0

phosphonic_acid_smarts = '[PX4](=O)([OX2H])[OX2H]'

def is_phosphonic_acid(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a phosphonic_acid.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_phosphonic_acid : bool
        Whether or not the compound is a phosphonic_acid, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_phosphonic_acid(MolFromSmiles('C1=CC=C(C=C1)CP(=O)(O)O')) # doctest:+SKIP
    True
    '''
    hits = mol.GetSubstructMatches(smarts_mol_cache(phosphonic_acid_smarts))
    # phosphonic acid itself only has an H for the last group
    return len(hits) > 0 or CanonSmiles(MolToSmiles(mol)) == 'O=[PH](O)O'

phosphodiester_smarts = '[PX4;H0](=O)([OX2H])([OX2H0])[OX2H0]'
def is_phosphodiester(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a phosphodiester.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_phosphodiester : bool
        Whether or not the compound is a phosphodiester, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_phosphodiester(MolFromSmiles('C(COP(=O)(O)OCC(C(=O)O)N)N=C(N)N')) # doctest:+SKIP
    True
    '''
    hits = mol.GetSubstructMatches(smarts_mol_cache(phosphodiester_smarts))
    return len(hits) > 0

phosphate_smarts = '[PX4;H0](=O)([OX2H])([OX2H])[OX2H0]'
def is_phosphate(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a phosphate.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_phosphate : bool
        Whether or not the compound is a phosphate, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_phosphate(MolFromSmiles('C1=CN(C(=O)N=C1N)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O')) # doctest:+SKIP
    True
    '''
    hits = mol.GetSubstructMatches(smarts_mol_cache(phosphate_smarts))
    return len(hits) > 0

# ester_smarts = '[#6][CX3](=O)[OX2H0][#6]'
# ester_smarts = '[$([#6]);!$(C=[O,S,N])]C(=O)O[$([#6]);!$(C=[O,S,N])]'
# ester_smarts = '[CX3H1,CX3](=O)'
ester_smarts = '[OX2H0][#6;!$([C]=[O])]'
def is_ester(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is an ester.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_ester : bool
        Whether or not the compound is an ester, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_ester(MolFromSmiles('CCOC(=O)C')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(ester_smarts))
    return bool(matches)

alkyllithium_smarts = '[Li+;H0].[C-]'

def is_alkyllithium(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule any alkyllithium functional groups.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_alkyllithium : bool
        Whether or not the compound is an alkyllithium, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_alkyllithium(MolFromSmiles('[Li+].[CH3-]')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(alkyllithium_smarts))
    return bool(matches)

alkylaluminium_smarts = '[Al][C,c]'

def is_alkylaluminium(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule any alkylaluminium functional groups.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_alkylaluminium : bool
        Whether or not the compound is an alkylaluminium, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_alkylaluminium(MolFromSmiles('CC[Al](CC)CC')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(alkylaluminium_smarts))
    return bool(matches)

smarts_alkylmagnesium_halide_Mg1 = '[I-,Br-,Cl-,F-].[Mg+][C,c]'
smarts_alkylmagnesium_halide_Mg0 = '[I,Br,Cl,F][Mg]'
smarts_alkylmagnesium_halide_Mg2 = '[c-,C-].[Mg+2].[I-,Br-,Cl-,F-]'

def is_alkylmagnesium_halide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule any alkylmagnesium_halide functional groups.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_alkylmagnesium_halide : bool
        Whether or not the compound is an alkylmagnesium_halide, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_alkylmagnesium_halide(MolFromSmiles('C1=CC=[C-]C=C1.[Mg+2].[Br-]')) # doctest:+SKIP
    True
    '''
    for s in (smarts_alkylmagnesium_halide_Mg1, smarts_alkylmagnesium_halide_Mg0, smarts_alkylmagnesium_halide_Mg2):
        hits = mol.GetSubstructMatches(smarts_mol_cache(s))
        if len(hits) > 0:
            return True
    return False

silyl_ether_smarts = '[SiX4]([OX2H0])([!H])([!H])[!H]'

def is_silyl_ether(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule any silyl ether functional groups.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_silyl_ether : bool
        Whether or not the compound is an silyl ether, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_silyl_ether(MolFromSmiles('C[Si](C)(C)OS(=O)(=O)C(F)(F)F')) # doctest:+SKIP
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(silyl_ether_smarts))
    return bool(matches)


branched_alkane_smarts = 'CC(C)C'
def is_branched_alkane(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is a branched alkane, also refered to as an isoparaffin. All bonds
    in the molecule must be single carbon-carbon or carbon-hydrogen.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_branched_alkane : bool
        Whether or not the compound is a branched alkane or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_branched_alkane(MolFromSmiles("CC(C)C(C)C(C)C")) # doctest:+SKIP
    True
    '''
    ri = mol.GetRingInfo()
    if len(ri.AtomRings()):
        return False
    alkane_matches = mol.GetSubstructMatches(smarts_mol_cache(alkane_smarts))
    only_aliphatic = substructures_are_entire_structure(mol, alkane_matches)
    return bool(only_aliphatic and mol.GetSubstructMatches(smarts_mol_cache(branched_alkane_smarts)))


hardcoded_organic_smiles = frozenset([
    'C', # methane
    'CO', # methanol
])
hardcoded_controversial_organic_smiles = frozenset([
    'NC(N)=O', # Urea - CRC, history
    'O=C(OC(=O)C(F)(F)F)C(F)(F)F', # Trifluoroacetic anhydride CRC, not a hydrogen in it

    ])
hardcoded_inorganic_smiles = frozenset([
    '[C-]#[O+]', # carbon monoxide
    'O=C=O', # Carbon dioxide
    'S=C=S', # Carbon disulfide
    'BrC(Br)(Br)Br', # Carbon tetrabromide
    'ClC(Cl)(Cl)Cl', # Carbon tetrachloride
    'FC(F)(F)F', # Carbon tetrafluoride
    'IC(I)(I)I', # Carbon tetraiodide
    'O=C(O)O', # Carbonic acid
    'O=C(Cl)Cl', # Carbonyl chloride
    'O=C(F)F', # Carbonyl fluoride
    'O=C=S', # Carbonyl sulfide
    ])
hardcoded_controversial_inorganic_smiles = frozenset([
    'C#N', # hydrogen cyanide
])

default_inorganic_smiles = frozenset().union(hardcoded_controversial_inorganic_smiles, hardcoded_inorganic_smiles)
default_organic_smiles = frozenset().union(hardcoded_organic_smiles, hardcoded_controversial_organic_smiles)

# allowed_organic_atoms = frozenset(['H','C','N','O','F','S','Cl','Br'])


organic_smarts_groups = [alkane_smarts, alkene_smarts, alkyne_smarts,
                         aromatic_smarts, '[C][H]', '[C@H]', '[CR]',
                         ] + amide_smarts_collection


def is_organic(mol, restrict_atoms=None,
               organic_smiles=default_organic_smiles,
               inorganic_smiles=default_inorganic_smiles):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is organic. The definition of organic vs. inorganic compounds is
    arabitrary. The rules implemented here are fairly complex.

    * If a compound has an C-C bond, a C=C bond, a carbon triple bond, a
      carbon attatched however to a hydrogen, a carbon in a ring, or an amide
      group.
    * If a compound is in the list of canonical smiles `organic_smiles`,
      either the defaults in the library or those provided as an input to the
      function, the molecule is considered organic.
    * If a compound is in the list of canonical smiles `inorganic_smiles`,
      either the defaults in the library or those provided as an input to the
      function, the molecule is considered inorganic.
    * If `restrict_atoms` is provided and atoms are present in the molecule
      that are restricted, the compound is considered restricted.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]
    restrict_atoms : Iterable[str]
        Atoms that cannot be found in an organic molecule, [-]
    organic_smiles : Iterable[str]
        Smiles that are hardcoded to be organic, [-]
    inorganic_smiles : Iterable[str]
        Smiles that are hardcoded to be inorganic, [-]

    Returns
    -------
    is_organic : bool
        Whether or not the compound is a organic or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_organic(MolFromSmiles("CC(C)C(C)C(C)C")) # doctest:+SKIP
    True
    '''
    if not loaded_rdkit:
        load_rdkit_modules()
    # Check the hardcoded list first
    smiles = CanonSmiles(MolToSmiles(mol))
    if smiles in organic_smiles:
        return True
    if smiles in default_inorganic_smiles:
        return False

    if restrict_atoms is not None:
        atoms = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            try:
                atoms[symbol] += 1
            except:
                atoms[symbol] = 1
        found_atoms = set(atoms.keys())

        if not found_atoms.issubset(restrict_atoms):
            return False

    for smart in organic_smarts_groups:
        matches = mol.GetSubstructMatches(smarts_mol_cache(smart))
        if matches:
            return True
    return False

def is_inorganic(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule is inorganic.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_inorganic : bool
        Whether or not the compound is inorganic or not, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_inorganic(MolFromSmiles("O=[Zr].Cl.Cl")) # doctest:+SKIP
    True
    '''
    return not is_organic(mol)


def count_ring_ring_attatchments(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, count the number
    of times a ring in the molecule is bonded with another ring
    in the molecule.

    An easy explanation is cubane - each edge of the cube is a ring uniquely
    bonding with another ring; so this function returns twelve.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    ring_ring_attatchments : bool
        The number of ring-ring bonds, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> count_ring_ring_attatchments(MolFromSmiles('C12C3C4C1C5C2C3C45')) # doctest:+SKIP
    12
    '''
    ri =  mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    ring_count = len(atom_rings)
    ring_ids = [frozenset(t) for t in atom_rings]
    ring_ring_attatchments = 0
    for i in range(ring_count):
        for j in range(i+1, ring_count):
#             print(ring_ids[i].intersection(ring_ids[j]))
            shared_atoms = int(len(ring_ids[i].intersection(ring_ids[j])) > 0)
            ring_ring_attatchments += shared_atoms
    return ring_ring_attatchments


def count_rings_attatched_to_rings(mol, allow_neighbors=True, atom_rings=None):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, count the number
    of rings in the molecule that are attatched to another ring.
    if `allow_neighbors` is True, any bond to another atom that is part of a
    ring is allowed; if it is False, the rings have to share a wall.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]
    allow_neighbors : bool
        Whether or not to count neighboring rings or just ones sharing a wall, [-]
    atom_rings : rdkit.Chem.rdchem.RingInfo, optional
        Internal parameter, used for performance only

    Returns
    -------
    rings_attatched_to_rings : bool
        The number of rings bonded to other rings, [-].

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> count_rings_attatched_to_rings(MolFromSmiles('C12C3C4C1C5C2C3C45')) # doctest:+SKIP
    6
    '''
    if atom_rings is None:
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
    ring_count = len(atom_rings)
    ring_ids = [frozenset(t) for t in atom_rings]
    rings_attatched_to_rings = 0
    other_ring_atoms = set()
    for i in range(ring_count):
        other_ring_atoms.clear()
        attatched_to_ring = False
        for j in range(ring_count):
            if i != j:
                other_ring_atoms.update(atom_rings[j])


        for atom in atom_rings[i]:
            if attatched_to_ring:
                break

            if atom in other_ring_atoms:
                attatched_to_ring = True
                break
            if allow_neighbors:
                atom_obj = mol.GetAtomWithIdx(atom)
                neighbors = atom_obj.GetNeighbors()
                for n in neighbors:
                    if n.GetIdx() in other_ring_atoms:
                        attatched_to_ring = True
                        break
                if attatched_to_ring:
                    break
        if attatched_to_ring:
            rings_attatched_to_rings+= 1
    return rings_attatched_to_rings

benzene_smarts = 'c1ccccc1'

def benene_rings(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns the number of benzene rings
    in the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    benene_rings : int
        Number of benzene rings in the molecule, [-]

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> benene_rings(MolFromSmiles('c1ccccc1')) # doctest:+SKIP
    1
    >>> benene_rings(MolFromSmiles('c1ccccc1c1ccccc1')) # doctest:+SKIP
    2
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(benzene_smarts))
    return len(matches)





radionuclides = {
    'H': {3},  # Tritium
    'Be': {10},  # Beryllium-10
    'C': {14},  # Carbon-14
    'F': {18},  # Fluorine-18
    'Al': {26},  # Aluminium-26
    'Cl': {36},  # Chlorine-36
    'K': {40},  # Potassium-40
    'Ca': {41},  # Calcium-41
    'Co': {60},  # Cobalt-60
    'Kr': {81},  # Krypton-81
    'Sr': {90},  # Strontium-90
    'Tc': {99},  # Technetium-99 and Technetium-99m (same isotope number)
    'I': {129, 131},  # Iodine-129 and Iodine-131
    'Xe': {135},  # Xenon-135
    'Cs': {137},  # Caesium-137
    'Gd': {153},  # Gadolinium-153
    'Bi': {209},  # Bismuth-209
    'Po': {210},  # Polonium-210
    'Rn': {222},  # Radon-222
    'Th': {232},  # Thorium-232
    'U': {235, 238},  # Uranium-235 and Uranium-238
    'Pu': {238, 239},  # Plutonium-238 and Plutonium-239
    'Am': {241},  # Americium-241
    'Cf': {252}  # Californium-252
}
# Computer readable version
# https://www.anl.gov/sites/www/files/2022-11/nubase_4.mas20.txt
# https://www.anl.gov/phy/atomic-mass-data-resources

def is_radionuclide(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule contains an unstable isotope (radionuclide).

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule [-]

    Returns
    -------
    is_radionuclide : bool
        Whether or not the compound is a radionuclide, [-]

    Notes
    -----
    The lsit of radionuclide in this function is not complete
    and only contains ~25 common ones. A complete data source is in [1]_.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles # doctest:+SKIP
    >>> is_radionuclide(MolFromSmiles("[131I]C")) # doctest:+SKIP
    True

    References
    ----------
    .. [1] Kondev, F. G., M. Wang, W. J. Huang, S. Naimi, and G. Audi.
       "The NUBASE2020 Evaluation of Nuclear Physics Properties *."
       Chinese Physics C 45, no. 3 (March 2021): 030001.
       https://doi.org/10.1088/1674-1137/abddae.
    '''
    for atom in mol.GetAtoms():
        element = atom.GetSymbol()
        isotope = atom.GetIsotope()
        if element in radionuclides and isotope in radionuclides[element]:
            return True
    return False


### Calculate things using functional groups - basic

def BVirial_Tsonopoulos_extended_ab(Tc, Pc, dipole, smiles):
    r'''Calculates the  of `a` and `b` parameters of the Tsonopoulos (extended)
    second virial coefficient prediction method. These parameters account for
    polarity. This function uses `rdkit` to identify the component type of the
    molecule.

    Parameters
    ----------
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    dipole : float
        dipole moment, optional, [Debye]

    Returns
    -------
    a : float
        Fit parameter matched to one of the supported chemical classes.
    b : float
        Fit parameter matched to one of the supported chemical classes.

    Notes
    -----
    To calculate `a` or `b`, the following rules are used:

    For 'simple' or 'normal' fluids:

    .. math::
        a = 0

    .. math::
        b = 0

    For 'ketone', 'aldehyde', 'alkyl nitrile', 'ether', 'carboxylic acid',
    or 'ester' types of chemicals:

    .. math::
        a = -2.14\times 10^{-4} \mu_r - 4.308 \times 10^{-21} (\mu_r)^8

    .. math::
        b = 0

    For 'alkyl halide', 'mercaptan', 'sulfide', or 'disulfide' types of
    chemicals:

    .. math::
        a = -2.188\times 10^{-4} (\mu_r)^4 - 7.831 \times 10^{-21} (\mu_r)^8

    .. math::
        b = 0

    For 'alkanol' types of chemicals (except methanol):

    .. math::
        a = 0.0878

    .. math::
        b = 0.00908 + 0.0006957 \mu_r

    For methanol:

    .. math::
        a = 0.0878

    .. math::
        b = 0.0525

    For water:

    .. math::
        a = -0.0109

    .. math::
        b = 0

    If required, the form of dipole moment used in the calculation of some
    types of `a` and `b` values is as follows:

    .. math::
        \mu_r = 100000\frac{\mu^2(Pc/101325.0)}{Tc^2}

    References
    ----------
    .. [1] Tsonopoulos, C., and J. L. Heidman. "From the Virial to the Cubic
       Equation of State." Fluid Phase Equilibria 57, no. 3 (1990): 261-76.
       doi:10.1016/0378-3812(90)85126-U
    .. [2] Tsonopoulos, Constantine, and John H. Dymond. "Second Virial
       Coefficients of Normal Alkanes, Linear 1-Alkanols (and Water), Alkyl
       Ethers, and Their Mixtures." Fluid Phase Equilibria, International
       Workshop on Vapour-Liquid Equilibria and Related Properties in Binary
       and Ternary Mixtures of Ethers, Alkanes and Alkanols, 133, no. 1-2
       (June 1997): 11-34. doi:10.1016/S0378-3812(97)00058-7.

    '''
    if smiles == 'CO':
        # methanol
        a, b = 0.0878, 0.0525
    elif smiles == 'O':
        # water
        a, b = -0.0109, 0.0
    else:
        from rdkit.Chem import MolFromSmiles
        mol = MolFromSmiles(smiles)

        dipole_r = 1E5*dipole**2*(Pc/101325.0)/Tc**2

        if (is_ketone(mol)
            or is_aldehyde(mol)
            or is_nitrile(mol)
            or is_ether(mol)
            or is_carboxylic_acid(mol)
            or is_ester(mol)
            # ammonia, H2S, HCN
            or smiles in ('N', 'S', 'C#N')):
            a, b = -2.14E-4*dipole_r -4.308E-21*dipole_r**8, 0.0

        elif (is_haloalkane(mol) or is_mercaptan(mol)
              or is_sulfide(mol) or is_disulfide(mol)):
            a, b = -2.188E-4*dipole_r**4 - 7.831E-21*dipole_r**8, 0.0

        elif (is_alcohol(mol)
              and not is_aromatic(mol)
              and not is_alkyne(mol)
              and not is_alkene(mol)):
            a, b = 0.0878, 0.00908 + 0.0006957*dipole_r
        else:
            a, b = 0.0, 0.0
    return (a, b)

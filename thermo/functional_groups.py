# -*- coding: utf-8 -*-
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

Specific molecule matching functions

.. autofunction:: thermo.functional_groups.is_alkane
.. autofunction:: thermo.functional_groups.is_mercaptan
.. autofunction:: thermo.functional_groups.is_cycloalkane
.. autofunction:: thermo.functional_groups.is_alkene
.. autofunction:: thermo.functional_groups.is_alkyne
.. autofunction:: thermo.functional_groups.is_aromatic
.. autofunction:: thermo.functional_groups.is_alcohol
.. autofunction:: thermo.functional_groups.is_polyol
.. autofunction:: thermo.functional_groups.is_acid
.. autofunction:: thermo.functional_groups.is_ketone
.. autofunction:: thermo.functional_groups.is_aldehyde
.. autofunction:: thermo.functional_groups.is_anhydride
.. autofunction:: thermo.functional_groups.is_ether
.. autofunction:: thermo.functional_groups.is_phenol
.. autofunction:: thermo.functional_groups.is_nitrile
.. autofunction:: thermo.functional_groups.is_carboxylic_acid
.. autofunction:: thermo.functional_groups.is_haloalkane
.. autofunction:: thermo.functional_groups.is_fluoroalkane
.. autofunction:: thermo.functional_groups.is_chloroalkane
.. autofunction:: thermo.functional_groups.is_bromoalkane
.. autofunction:: thermo.functional_groups.is_iodoalkane
.. autofunction:: thermo.functional_groups.is_amine
.. autofunction:: thermo.functional_groups.is_primary_amine
.. autofunction:: thermo.functional_groups.is_secondary_amine
.. autofunction:: thermo.functional_groups.is_tertiary_amine
.. autofunction:: thermo.functional_groups.is_ester
.. autofunction:: thermo.functional_groups.is_branched_alkane

'''

from __future__ import division

__all__ = ['is_mercaptan', 'is_alkane', 'is_cycloalkane', 'is_alkene',
           'is_alkyne', 'is_aromatic', 'is_alcohol', 'is_polyol',
           'is_acid', 'is_ketone', 'is_aldehyde', 'is_anhydride',
           'is_ether', 'is_phenol', 'is_nitrile', 'is_carboxylic_acid',
           'is_haloalkane', 'is_fluoroalkane', 'is_chloroalkane', 
           'is_bromoalkane', 'is_iodoalkane',
           'is_amine', 'is_primary_amine', 'is_secondary_amine',
           'is_tertiary_amine', 'is_ester', 'is_branched_alkane']


rdkit_missing = 'RDKit is not installed; it is required to use this functionality'

loaded_rdkit = False
Chem = Descriptors = AllChem = rdMolDescriptors = CanonSmiles = MolToSmiles = MolFromSmarts = None
def load_rdkit_modules():
    global loaded_rdkit, Chem, Descriptors, AllChem, rdMolDescriptors, CanonSmiles, MolToSmiles, MolFromSmarts
    if loaded_rdkit:
        return
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, CanonSmiles, MolToSmiles, MolFromSmarts
        loaded_rdkit = True
    except:
        if not hasRDKit: # pragma: no cover
            raise Exception(rdkit_missing)




def substructures_are_entire_structure(mol, matches, exclude_Hs=True):
    atomIdxs = set([atom.GetIdx() for atom in mol.GetAtoms() if (not exclude_Hs or atom.GetAtomicNum() != 1)])
    matched_atoms = []
    for h in matches:
        matched_atoms.extend(h)
    return set(matched_atoms) == atomIdxs



mercaptan_smarts = '[#16X2H]'

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


def is_mercaptan(mol):
    r'''Given a `rdkit.Chem.rdchem.Mol` object, returns whether or not the
    molecule has a mercaptan R-SH group.

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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_mercaptan(MolFromSmiles("CS"))
    True
    '''
    # https://smarts.plus/smartsview/a9b45f9cc6f17d3b5649b77a81f535dfe0729a84fc3ac453c9aaa60286e6
    hits = mol.GetSubstructMatches(smarts_mol_cache(mercaptan_smarts))
    return len(hits) > 0


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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_alkane(MolFromSmiles("CCC"))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_cycloalkane(MolFromSmiles('C1CCCCCCCCC1'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_alkene(MolFromSmiles('C=C'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_alkyne(MolFromSmiles('CC#C'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_aromatic(MolFromSmiles('CC1=CC=CC=C1C'))
    True
    '''
    # https://smarts.plus/smartsview/6011201068306235ac861ddaa794a4559576e23361a5437373562ae3cc45
    matches = mol.GetSubstructMatches(smarts_mol_cache(aromatic_smarts))
    return bool(matches)


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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_alcohol(MolFromSmiles('CCO'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_polyol(MolFromSmiles('C(C(CO)O)O'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_acid(MolFromSmiles('CC(=O)O'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_ketone(MolFromSmiles('C1CCC(=O)CC1'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_aldehyde(MolFromSmiles('C=O'))
    True
    '''
    # https://smarts.plus/smartsview/8c5ed80db53e19cc40dcfef58453d90fec96e18d8b7f602d34ff1e3a566c
    matches = mol.GetSubstructMatches(smarts_mol_cache(aldehyde_smarts))
    return bool(matches) or CanonSmiles(MolToSmiles(mol)) == 'C=O' 

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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_anhydride(MolFromSmiles('C1=CC(=O)OC1=O'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_ether(MolFromSmiles('CC(C)OC(C)C'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_phenol(MolFromSmiles('CC(=O)NC1=CC=C(C=C1)O'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_nitrile(MolFromSmiles('CC#N'))
    True
    '''
    # https://smarts.plus/smartsview/a04d5a51cd03fd469672f34a226fbd049a5d220d3819992fe210bd6d77a7
    matches = mol.GetSubstructMatches(smarts_mol_cache(nitrile_smarts))
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
    
    >>> from rdkit.Chem import MolFromSmiles
    >>> is_carboxylic_acid(MolFromSmiles('CCCC(=O)O'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_haloalkane(MolFromSmiles('CCCl'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_fluoroalkane(MolFromSmiles('CF'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_chloroalkane(MolFromSmiles('CCl'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_bromoalkane(MolFromSmiles('CBr'))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_iodoalkane(MolFromSmiles('CI'))
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(iodoalkane_smarts))
    return bool(matches)

amine_smarts = '[$([NH2][CX4]),$([$([NH]([CX4])[CX4]);!$([NH]([CX4])[CX4][O,N]);!$([NH]([CX4])[CX4][O,N])]),$([ND3]([CX4])([CX4])[CX4])]'
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_amine(MolFromSmiles('CN'))
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(amine_smarts))
    return bool(matches)

primary_amine_smarts = '[CX4][NH2]'
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_primary_amine(MolFromSmiles('CN'))
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(primary_amine_smarts))
    return bool(matches)

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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_secondary_amine(MolFromSmiles('CNC'))
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(secondary_amine_smarts))
    return bool(matches)

tertiary_amine_smarts = '[ND3]([CX4])([CX4])[CX4]'
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_tertiary_amine(MolFromSmiles('CN(C)C'))
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(tertiary_amine_smarts))
    return bool(matches)

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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_ester(MolFromSmiles('CCOC(=O)C'))
    True
    '''
    matches = mol.GetSubstructMatches(smarts_mol_cache(ester_smarts))
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

    >>> from rdkit.Chem import MolFromSmiles
    >>> is_branched_alkane(MolFromSmiles("CC(C)C(C)C(C)C"))
    True
    '''
    ri = mol.GetRingInfo()
    if len(ri.AtomRings()):
        return False
    alkane_matches = mol.GetSubstructMatches(smarts_mol_cache(alkane_smarts))
    only_aliphatic = substructures_are_entire_structure(mol, alkane_matches)
    if only_aliphatic and mol.GetSubstructMatches(smarts_mol_cache(branched_alkane_smarts)):
        return True
    return False
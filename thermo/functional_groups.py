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

.. autofunction:: thermo.functional_groups.is_mercaptan


'''

from __future__ import division

__all__ = ['is_mercaptan']


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

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

This module contains an implementation of the Wilson-Jasperson
group-contribution method.
This functionality requires the RDKit library to work.

.. contents:: :local:


.. autofunction:: thermo.group_contribution.Wilson_Jasperson
'''
__all__ = ['Wilson_Jasperson', 'Wilson_Jasperson_Tc_increments',
           'Wilson_Jasperson_Pc_increments',
           'Wilson_Jasperson_Tc_groups', 'Wilson_Jasperson_Pc_groups']
from math import exp

from chemicals.elements import simple_formula_parser

from thermo.functional_groups import (
    alcohol_smarts,
    aldehyde_smarts,
    all_amine_smarts,
    carboxylic_acid_smarts,
    disulfide_smarts,
    ester_smarts,
    ether_smarts,
    is_haloalkane,
    ketone_smarts,
    mercaptan_smarts,
    nitrile_smarts,
    nitro_smarts,
    siloxane_smarts,
    smarts_mol_cache,
    sulfide_smarts,
)

Wilson_Jasperson_Tc_increments = {
'H': 0.002793,
'D': 0.002793,
'T': 0.002793,
'He': 0.32,
'B': 0.019,
'C': 0.008532,
'N': 0.019181,
'O': 0.020341,
'F': 0.00881,
'Ne': 0.0364,
'Al': 0.088,
'Si': 0.02,
'P': 0.012,
'S': 0.007271,
'Cl': 0.011151,
'Ar': 0.0168,
'Ti': 0.014,
'V': 0.0186,
'Ga': 0.059,
'Ge': 0.031,
'As': 0.007,
'Se': 0.0103,
'Br': 0.012447,
'Kr': 0.0133,
'Rb': -0.027,
'Zr': 0.175,
'Nb': 0.0176,
'Mo': 0.007,
'Sn': 0.02,
'Sb': 0.01,
'Te': 0,
'I': 0.0059,
'Xe': 0.017,
'Cs': -0.0275,
'Hf': 0.219,
'Ta': 0.013,
'W': 0.011,
'Re': 0.014,
'Os': -0.05,
'Hg': 0,
'Bi': 0,
'Rn': 0.007,
'U': 0.015,
}

Wilson_Jasperson_Pc_increments = {
'H': 0.1266,
'D': 0.1266,
'T': 0.1266,
'He': 0.434,
'B': 0.91,
'C': 0.72983,
'N': 0.44805,
'O': 0.4336,
'F': 0.32868,
'Ne': 0.126,
'Al': 6.05,
'Si': 1.34,
'P': 1.22,
'S': 1.04713,
'Cl': 0.97711,
'Ar': 0.796,
'Ti': 1.19,
'V': None ,
'Ga': None ,
'Ge': 1.42,
'As': 2.68,
'Se': 1.2,
'Br': 0.97151,
'Kr': 1.11,
'Rb': None ,
'Zr': 1.11,
'Nb': 2.71,
'Mo': 1.69,
'Sn': 1.95,
'Sb': None ,
'Te': 0.43,
'I': 1.31593,
'Xe': 1.66,
'Cs': 6.33,
'Hf': 1.07,
'Ta': None ,
'W': 1.08,
'Re': None ,
'Os': None ,
'Hg': -0.08,
'Bi': 0.69,
'Rn': 2.05,
'U': 2.04,
}

Wilson_Jasperson_Tc_groups = {'OH_large': 0.01, 'OH_small': 0.0350, '-O-': -0.0075, 'amine': -0.004,
                            '-CHO': 0, '>CO': -0.0550, '-COOH': 0.017, '-COO-': -0.015,
                             '-CN': 0.017, '-NO2': -0.02, 'halide': 0.002, 'sulfur_groups': 0.0,
                            'siloxane': -0.025}
Wilson_Jasperson_Pc_groups = {'OH_large': 0, 'OH_small': 0, '-O-': 0, 'amine': 0,
                            '-CHO': 0.5, '>CO': 0, '-COOH': 0.5, '-COO-': 0,
                             '-CN': 1.5, '-NO2': 1.0, 'halide': 0, 'sulfur_groups': 0.0,
                            'siloxane': -0.5}

def Wilson_Jasperson(mol, Tb, second_order=True):
    r'''Estimate the critical temperature and pressure of a molecule using
    the molecule itself, and a known or estimated boiling point
    using the Wilson-Jasperson method.

    Parameters
    ----------
    mol : str or rdkit.Chem.rdchem.Mol, optional
        Smiles string representing a chemical or a rdkit molecule, [-]
    Tb : float
        Known or estimated boiling point, [K]
    second_order : bool
        Whether to use the first order method (False), or the second order
        method, [-]

    Returns
    -------
    Tc : float
        Estimated critical temperature, [K]
    Pc : float
        Estimated critical pressure, [Pa]
    missing_Tc_increments : bool
        Whether or not there were missing atoms for the `Tc` calculation, [-]
    missing_Pc_increments : bool
        Whether or not there were missing atoms for the `Pc` calculation, [-]

    Notes
    -----
    Raises an exception if rdkit is not installed, or `smi` or `rdkitmol` is
    not defined.

    Calculated values were published in [3]_ for 448 compounds, as calculated
    by NIST TDE. There appear to be further modifications to the method in
    NIST TDE, as ~25% of values have differences larger than 5 K.

    Examples
    --------
    Example for 2-ethylphenol in [2]_:

    >>> Tc, Pc, _, _ = Wilson_Jasperson('CCC1=CC=CC=C1O', Tb=477.67) # doctest:+SKIP
    >>> (Tc, Pc) # doctest:+SKIP
    (693.567, 3743819.6667)
    >>> Tc, Pc, _, _ = Wilson_Jasperson('CCC1=CC=CC=C1O', Tb=477.67, second_order=False) # doctest:+SKIP
    >>> (Tc, Pc) # doctest:+SKIP
    (702.883, 3794106.49)

    References
    ----------
    .. [1] Wilson, G. M., and L. V. Jasperson. "Critical Constants Tc, Pc,
       Estimation Based on Zero, First and Second Order Methods." In
       Proceedings of the AIChE Spring Meeting, 21, 1996.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [3] Yan, Xinjian, Qian Dong, and Xiangrong Hong. "Reliability Analysis
       of Group-Contribution Methods in Predicting Critical Temperatures of
       Organic Compounds." Journal of Chemical & Engineering Data 48, no. 2
       (March 1, 2003): 374-80. https://doi.org/10.1021/je025596f.
    '''
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    if type(mol) is Chem.rdchem.Mol:
        rdkitmol = Chem.Mol(mol)
        no_H_mol = mol
    else:
        rdkitmol = Chem.MolFromSmiles(mol)
        no_H_mol = Chem.Mol(rdkitmol)

    ri = no_H_mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    Nr = len(atom_rings)

    atoms = simple_formula_parser(rdMolDescriptors.CalcMolFormula(rdkitmol))

    group_contributions = {}
    OH_matches = rdkitmol.GetSubstructMatches(smarts_mol_cache(alcohol_smarts))
    if 'C' in atoms:
        if atoms['C'] >= 5:
            group_contributions['OH_large'] = len(OH_matches)
        else:
            group_contributions['OH_small'] = len(OH_matches)

    ether_O_matches =  rdkitmol.GetSubstructMatches(smarts_mol_cache(ether_smarts))
    group_contributions['-O-'] = len(ether_O_matches)


    group_contributions['-CN'] = 0
    amine_groups = set()
    if 'N' in atoms:
        nitro_matches =  rdkitmol.GetSubstructMatches(smarts_mol_cache(nitro_smarts))
        group_contributions['-NO2'] = len(nitro_matches)

        nitrile_matches =  rdkitmol.GetSubstructMatches(smarts_mol_cache(nitrile_smarts))
        group_contributions['-CN'] = len(nitrile_matches)

        for s in all_amine_smarts:
            amine_matches =  rdkitmol.GetSubstructMatches(smarts_mol_cache(s))
            for h in amine_matches:
                # Get the N atom and store its index
                for at in h:
                    atom = rdkitmol.GetAtomWithIdx(at)
                    if atom.GetSymbol() == 'N':
                        amine_groups.add(at)
#     print(amine_groups)
    group_contributions['amine'] = len(amine_groups)

    if 'O' in atoms and 'C' in atoms:
        aldehyde_matches =  rdkitmol.GetSubstructMatches(smarts_mol_cache(aldehyde_smarts))
        group_contributions['-CHO'] = len(aldehyde_matches)

        ketone_matches =  rdkitmol.GetSubstructMatches(smarts_mol_cache(ketone_smarts))
        group_contributions['>CO'] = len(ketone_matches)

        carboxylic_acid_matches =  rdkitmol.GetSubstructMatches(smarts_mol_cache(carboxylic_acid_smarts))
        group_contributions['-COOH'] = len(carboxylic_acid_matches)

        ester_matches =  rdkitmol.GetSubstructMatches(smarts_mol_cache(ester_smarts))
        group_contributions['-COO-'] = len(ester_matches)



    group_contributions['halide'] = 1 if is_haloalkane(rdkitmol) else 0

    group_contributions['sulfur_groups'] = 0
    if 'S' in atoms:
        for s in (mercaptan_smarts, sulfide_smarts, disulfide_smarts):
            group_contributions['sulfur_groups'] += len(rdkitmol.GetSubstructMatches(smarts_mol_cache(s)))

    group_contributions['siloxane'] = 0
    if 'Si' in atoms:
        siloxane_matches = rdkitmol.GetSubstructMatches(smarts_mol_cache(siloxane_smarts))
        group_contributions['siloxane'] = len(siloxane_matches)

#     group_contributions = {'OH_large': 0, '-O-': 0, 'amine': 0, '-CHO': 0,
#                            '>CO': 0, '-COOH': 0, '-COO-': 0, '-CN': 0,
#                            '-NO2': 0, 'halide': 0, 'sulfur_groups': 0, 'siloxane': 0}

    missing_Tc_increments = False
    Tc_inc = 0.0
    for k, v in atoms.items():
        try:
            Tc_inc += Wilson_Jasperson_Tc_increments[k]*v
        except KeyError:
            missing_Tc_increments = True

    missing_Pc_increments = False
    Pc_inc = 0.0
    for k, v in atoms.items():
        try:
            Pc_inc += Wilson_Jasperson_Pc_increments[k]*v
        except (KeyError, TypeError):
            missing_Pc_increments = True

    second_order_Pc = 0.0
    second_order_Tc = 0.0
    if second_order:
        for k, v in group_contributions.items():
            second_order_Tc += Wilson_Jasperson_Tc_groups[k]*v
        for k, v in group_contributions.items():
            second_order_Pc += Wilson_Jasperson_Pc_groups[k]*v

#     print(atoms)
#     print(group_contributions)
#     print('rings', Nr)
#     print(Tc_inc, second_order_Tc)
    den = 0.048271 - 0.019846*Nr + Tc_inc + second_order_Tc
    if den >= 0:
        Tc = Tb/(den)**0.2
    else:
        # Can't make a prediction
        missing_Tc_increments = True
        Tc = Tb*2.5

    Y = -0.00922295 - 0.0290403*Nr + 0.041*(second_order_Pc + Pc_inc)

    Pc = 0.0186233*Tc/(-0.96601 + exp(Y))
    return Tc, Pc*1e5, missing_Tc_increments, missing_Pc_increments

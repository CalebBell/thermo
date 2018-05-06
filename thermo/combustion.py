# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from __future__ import division
from pprint import pprint

__all__ = ['Hcombustion', 'combustion_products', 'combustion_products_mixture']



def combustion_products(atoms):
    '''Calculates the combustion products of a molecule, given a dictionary of
    its constituent atoms and their counts.
    Products for non-hydrocarbons may not be correct, but are still 
    calculated.

    Parameters
    ----------
    atoms : dict
        Dictionary of atoms and their counts, [-]

    Returns
    -------
    combustion_producucts : dict
        Dictionary of combustion products and their counts, [-]

    Notes
    -----
    Also included in the results is the moles of O2 required per mole of
    the mixture of the molecule.

    Examples
    --------
    Methanol:

    >>> pprint(combustion_products({'H': 4, 'C': 1, 'O': 1}))
    {'Br2': 0.0,
     'CO2': 1,
     'H2O': 2.0,
     'HCl': 0,
     'HF': 0,
     'I2': 0.0,
     'N2': 0.0,
     'O2_required': 1.5,
     'P4O10': 0.0,
     'SO2': 0}
    '''
    nC, nH, nN, nO, nS, nBr, nI, nCl, nF, nP = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    if 'C' in atoms and atoms['C'] != 0:
        nC = atoms['C']
    if 'H' in atoms:
        nH = atoms['H']
    if 'N' in atoms:
        nN = atoms['N']
    if 'O' in atoms:
        nO = atoms['O']
    if 'S' in atoms:
        nS = atoms['S']
    if 'Br' in atoms:
        nBr = atoms['Br']
    if 'I' in atoms:
        nI = atoms['I']
    if 'Cl' in atoms:
        nCl = atoms['Cl']
    if 'F' in atoms:
        nF = atoms['F']
    if 'P' in atoms:
        nP = atoms['P']

    nO2_req = nC + nS + nH/4. + 5*nP/4. - (nCl + nF)/4. - nO/2.
    nCO2 = nC
    nBr2 = nBr/2.
    nI2 = nI/2.

    nHCl = nCl
    nHF = nF

    nSO2 = nS

    nN2 = nN/2.
    nP4O10 = nP/4.
    nH2O = (nH - nCl - nF)/2.
    products = {'CO2': nCO2, 'Br2': nBr2, 'I2': nI2, 'HCl': nCl, 'HF': nHF, 
                'SO2': nSO2, 'N2': nN2, 'P4O10': nP4O10, 'H2O': nH2O,
                'O2_required': nO2_req}
    return products


# mixture - mole fractions and atoms

def combustion_products_mixture(atoms_list, zs):
    '''Calculates the combustion products of a mixture of molecules and their,
    mole fractions; requires a list of dictionaries of each molecule's 
    constituent atoms and their counts.
    Products for non-hydrocarbons may not be correct, but are still 
    calculated.

    Parameters
    ----------
    atoms_list : list[dict]
        List of dictionaries of atoms and their counts, [-]
    zs : list[float]
        Mole fractions of each molecule in the mixture, [-]

    Returns
    -------
    combustion_producucts : dict
        Dictionary of combustion products and their counts, [-]

    Notes
    -----
    Also included in the results is the moles of O2 required per mole of
    the mixture to be burnt.

    Examples
    --------
    Mixture of methane and ethane.

    >>> combustion_products_mixture([{'H': 4, 'C': 1}, {'H': 6, 'C': 2}],
    ... [.9, .1])
    {'Br2': 0.0,
     'CO2': 1.1,
     'H2O': 2.1,
     'HCl': 0,
     'HF': 0,
     'I2': 0.0,
     'N2': 0.0,
     'O2_required': 2.15,
     'P4O10': 0.0,
     'SO2': 0}
    '''
    products = {'CO2': 0.0, 'Br2': 0.0, 'I2': 0.0, 'HCl': 0.0, 'HF': 0.0, 
                'SO2': 0.0, 'N2': 0.0, 'P4O10': 0.0, 'H2O': 0.0,
                'O2_required': 0.0}
    for atoms, zs_i in zip(atoms_list, zs):
        ans = combustion_products(atoms)
        if ans is not None:
            for key, val in ans.items():
                products[key] += val*zs_i
    return products


def Hcombustion(atoms, Hf=None, HfH2O=-285825, HfCO2=-393474,
                HfSO2=-296800, HfBr2=30880, HfI2=62417, HfHCl=-92173,
                HfHF=-272711, HfP4O10=-3009940, HfO2=0, HfN2=0):
    '''Calculates the heat of combustion, in J/mol.
    Value non-hydrocarbons is not correct, but still calculable.

    Parameters
    ----------
    atoms : dict
        Dictionary of atoms and their counts, []
    Hf : float
        Heat of formation of given chemical, [J/mol]
    HfH2O : float, optional
        Heat of formation of water, [J/mol]
    HfCO2 : float, optional
        Heat of formation of carbon dioxide, [J/mol]
    HfSO2 : float, optional
        Heat of formation of sulfur dioxide, [J/mol]
    HfBr2 : float, optional
        Heat of formation of bromine, [J/mol]
    HfI2 : float, optional
        Heat of formation of iodine, [J/mol]
    HfHCl : float, optional
        Heat of formation of chlorine, [J/mol]
    HfHF : float, optional
        Heat of formation of hydrogen fluoride, [J/mol]
    HfP4O10 : float, optional
        Heat of formation of phosphorus pentoxide, [J/mol]
    HfO2 : float, optional
        Heat of formation of oxygen, [J/mol]
    HfN2 : float, optional
        Heat of formation of nitrogen, [J/mol]

    Returns
    -------
    Hc : float
        Heat of combustion of chemical, [J/mol]

    Notes
    -----
    Default heats of formation for chemicals are at 298 K, 1 atm.

    Examples
    --------
    Liquid methanol burning

    >>> Hcombustion({'H': 4, 'C': 1, 'O': 1}, Hf=-239100)
    -726024.0
    '''
    if not Hf or not atoms:
        return None
    nC, nH, nN, nO, nS, nBr, nI, nCl, nF, nP = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    if 'C' in atoms and atoms['C'] != 0:
        nC = atoms['C']
    else:
        return None  # C is necessary for this formula
    if 'H' in atoms:
        nH = atoms['H']
    if 'N' in atoms:
        nN = atoms['N']
    if 'O' in atoms:
        nO = atoms['O']
    if 'S' in atoms:
        nS = atoms['S']
    if 'Br' in atoms:
        nBr = atoms['Br']
    if 'I' in atoms:
        nI = atoms['I']
    if 'Cl' in atoms:
        nCl = atoms['Cl']
    if 'F' in atoms:
        nF = atoms['F']
    if 'P' in atoms:
        nP = atoms['P']

    nO2_req = nC + nS + nH/4. + 5*nP/4. - (nCl + nF)/4. - nO/2.
    nCO2 = nC
    nBr2 = nBr/2.
    nI2 = nI/2.

    nHCl = nCl
    nHF = nF

    nSO2 = nS

    nN2 = nN/2.
    nP4O10 = nP/4.
    nH2O = (nH - nCl - nF)/2.

    Hc = (nBr2*HfBr2 + nI2*HfI2) + (nHCl*HfHCl + nHF*HfHF) + nSO2*HfSO2 + \
        nN2*HfN2 + nP4O10*HfP4O10 + nH2O*HfH2O - nO2_req*HfO2 + nCO2*HfCO2 - Hf
    return Hc

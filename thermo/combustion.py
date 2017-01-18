# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['Hcombustion']


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

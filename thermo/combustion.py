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
from thermo.utils import property_mass_to_molar, property_molar_to_mass

__all__ = ['Hcombustion', 'combustion_products', 'combustion_products_mixture',
           'air_fuel_ratio_solver']


combustion_atoms = set(['C', 'H', 'N', 'O', 'S', 'Br', 'I', 'Cl', 'F', 'P'])

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
    
    Atoms not in ['C', 'H', 'N', 'O', 'S', 'Br', 'I', 'Cl', 'F', 'P'] are 
    returned as pure species; i.e. sodium hydroxide produces water and pure
    Na.

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

    nO2_req = nC + nS + .25*nH + 1.25*nP - .25*(nCl + nF) - .5*nO
    nCO2 = nC
    nBr2 = .5*nBr
    nI2 = .5*nI

    nHCl = nCl
    nHF = nF

    nSO2 = nS

    nN2 = .5*nN
    nP4O10 = .25*nP
    nH2O = (nH - nCl - nF)/2.
    products = {'CO2': nCO2, 'Br2': nBr2, 'I2': nI2, 'HCl': nCl, 'HF': nHF, 
                'SO2': nSO2, 'N2': nN2, 'P4O10': nP4O10, 'H2O': nH2O,
                'O2_required': nO2_req}
    
    for atom, value in atoms.items():
        if atom not in combustion_atoms:
            products[atom] = value
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
    
    Note that if O2 is in the feed, this will be subtracted from the required
    O2 amount.
    
    Note that if instead of mole fractions, mole flows are given - the results
    are in terms of mole flows as well!

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
    # Attempted to use a .copy() on a base dict but that was slower
    products = {'CO2': 0.0, 'Br2': 0.0, 'I2': 0.0, 'HCl': 0.0, 'HF': 0.0, 
                'SO2': 0.0, 'N2': 0.0, 'P4O10': 0.0, 'H2O': 0.0,
                'O2_required': 0.0}
    for atoms, zs_i in zip(atoms_list, zs):
        ans = combustion_products(atoms)
        if ans is not None:
            for key, val in ans.items():
                if key in products:
                    products[key] += val*zs_i
                else:
                    products[key] = val*zs_i
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


def air_fuel_ratio_solver(ratio, Vm_air, Vm_fuel, MW_air, MW_fuel,
                          n_air=None, n_fuel=None,
                          basis='mass', full_info=False):
    '''Calculates molar flow rate of air or fuel from the other,
    using a specified air-fuel ratio. Supports 'mole', 'mass', and 'volume'
    bases for the ratio variable. The ratio must be of the same units - 
    i.e. kg/kg instead of lb/kg.
    
    If `full_info` is True, the mole, mass, and volume air-fuel ratios will
    all be calculated and returned as well.

    Parameters
    ----------
    ratio : float
        Air-fuel ratio, in the specified `basis`, [-]
    Vm_air : float
        Molar volume of air, [m^3/mol]
    Vm_fuel : float
        Molar volume of fuel, [m^3/mol]
    MW_air : float
        Molecular weight of air, [g/mol]
    MW_fuel : float
        Molecular weight of fuel, [g/mol]
    n_air : float, optional
        Molar flow rate of air, [mol/s]
    n_fuel : float, optional
        Molar flow rate of fuel, [mol/s]
    basis : str, optional
        One of 'mass', 'mole', or 'volume', [-]
    full_info : bool, optional
        Whether to calculate and return all the mole, mass, and volume 
        ratios as well, [-]
        
    Returns
    -------
    n_air : float
        Molar flow rate of air, [mol/s]
    n_fuel : float
        Molar flow rate of fuel, [mol/s]
    mole_ratio : float, only returned if full_info == True
        Air-fuel mole ratio, [-]
    mass_ratio : float, only returned if full_info == True
        Air-fuel mass ratio, [-]
    volume_ratio : float, only returned if full_info == True
        Air-fuel volume ratio, [-]

    Notes
    -----
    The function works so long as the flow rates, molar volumes, and molecular
    weights are in a consistent basis.
    
    The function may also be used to obtain the other ratios, even if both 
    flow rates are known.
    
    Be careful to use standard volumes if the ratio known is at standard
    conditions! 
    
    Examples
    --------
    >>> Vm_air = 0.024936627188566596
    >>> Vm_fuel = 0.024880983160354486
    >>> MW_air = 28.850334
    >>> MW_fuel = 17.86651
    >>> n_fuel = 5.0
    >>> n_air = 25.0
    >>> air_fuel_ratio_solver(ratio=5, Vm_air=Vm_air, Vm_fuel=Vm_fuel,
    ... MW_air=MW_air, MW_fuel=MW_fuel, n_air=n_air,
    ... n_fuel=n_fuel, basis='mole', full_info=True)
    (25.0, 5.0, 5.0, 8.073858296891782, 5.011182039683378)
    
    '''
    if basis == 'mole':
        if n_air is not None:
            n_fuel = n_air/ratio
        elif n_fuel is not None:
            n_air = n_fuel*ratio
    elif basis == 'mass':
        if n_air is not None:
            m_air = property_mass_to_molar(n_air, MW_air)
            m_fuel = m_air/ratio
            n_fuel = property_molar_to_mass(m_fuel, MW_fuel)
        elif n_fuel is not None:
            m_fuel = property_mass_to_molar(n_fuel, MW_fuel)
            m_air = m_fuel*ratio
            n_air = property_molar_to_mass(m_air, MW_air)
    elif basis == 'volume':
        if n_air is not None:
            V_air = n_air*Vm_air
            V_fuel = V_air/ratio
            n_fuel = V_fuel/Vm_fuel
        elif n_fuel is not None:
            V_fuel = n_fuel*Vm_fuel
            V_air = V_fuel*ratio
            n_air = V_air/Vm_air
    if n_air is None or n_fuel is None:
        raise ValueError("Could not convert")
    if full_info:
        mole_ratio = n_air/n_fuel
        mass_ratio, volume_ratio = MW_air/MW_fuel*mole_ratio, Vm_air/Vm_fuel*mole_ratio
        return n_air, n_fuel, mole_ratio, mass_ratio, volume_ratio
        
    return n_air, n_fuel
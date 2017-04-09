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

__all__ = ['PeriodicTable', 'molecular_weight', 'mass_fractions', 
           'atom_fractions', 'similarity_variable', 'atoms_to_Hill', 
           'simple_formula_parser', 'CAS_by_number', 'periods', 'groups', 
           'blocks']
import os
import re
from collections import Counter
from thermo.utils import to_num

folder = os.path.join(os.path.dirname(__file__), 'Misc')


CAS_by_number = ['1333-74-0', '7440-59-7', '7439-93-2', '7440-41-7', '7440-42-8', '7440-44-0', '7727-37-9', '7782-44-7', '7782-41-4', '7440-01-9', '7440-23-5', '7439-95-4', '7429-90-5', '7440-21-3', '7723-14-0', '7704-34-9', '7782-50-5', '7440-37-1', '7440-09-7', '7440-70-2', '7440-20-2', '7440-32-6', '7440-62-2', '7440-47-3', '7439-96-5', '7439-89-6', '7440-48-4', '7440-02-0', '7440-50-8', '7440-66-6', '7440-55-3', '7440-56-4', '7440-38-2', '7782-49-2', '7726-95-6', '7439-90-9', '7440-17-7', '7440-24-6', '7440-65-5', '7440-67-7', '7440-03-1', '7439-98-7', '7440-26-8', '7440-18-8', '7440-16-6', '7440-05-3', '7440-22-4', '7440-43-9', '7440-74-6', '7440-31-5', '7440-36-0', '13494-80-9', '7553-56-2', '7440-63-3', '7440-46-2', '7440-39-3', '7439-91-0', '7440-45-1', '7440-10-0', '7440-00-8', '7440-12-2', '7440-19-9', '7440-53-1', '7440-54-2', '7440-27-9', '7429-91-6', '7440-60-0', '7440-52-0', '7440-30-4', '7440-64-4', '7439-94-3', '7440-58-6', '7440-25-7', '7440-33-7', '7440-15-5', '7440-04-2', '7439-88-5', '7440-06-4', '7440-57-5', '7439-97-6', '7440-28-0', '7439-92-1', '7440-69-9', '7440-08-6', '7440-68-8', '10043-92-2', '7440-73-5', '7440-14-4', '7440-34-8', '7440-29-1', '7440-13-3', '7440-61-1', '7439-99-8', '7440-07-5', '7440-35-9', '7440-51-9', '7440-40-6', '7440-71-3', '7429-92-7', '7440-72-4', '7440-11-1', '10028-14-5', '22537-19-5', '53850-36-5', '53850-35-4', '54038-81-2', '54037-14-8', '54037-57-9', '54038-01-6', '54083-77-1', '54386-24-2', '54084-26-3', '54084-70-7', '54085-16-4', '54085-64-2', '54100-71-9', '54101-14-3', '54144-19-3']
'''CAS numbers of the elements, indexed by atomic numbers off-by-one up to 118.'''

periods = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
'''Periods of the elements, indexed by atomic numbers off-by-one up to 118.'''

groups = [1, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
'''Groups of the elements, indexed by atomic numbers off-by-one up to 118.
Lanthanides and Actinides are set to None.'''

s_block = [1, 2, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88]
d_block = list(range(21, 31)) + list(range(39, 49)) + list(range(71, 81)) + list(range(103, 113))
f_block = list(range(57, 71)) + list(range(89, 103))
p_block = list(range(5, 11)) + list(range(13, 19)) + list(range(31, 37)) + list(range(49, 55)) + list(range(81, 87)) + list(range(113, 119))
blocks = {'s': s_block, 'd': d_block, 'f': f_block, 'p': p_block}
'''Blocks of the elements, stored in a dictionary with four keys and lists.
Indexed by atomic numbers off-by-one up to 118.'''



class PeriodicTable(object):
    '''Periodic Table object for use in dealing with elements.

    Parameters
    ----------
    elements : list[Element]
        List of Element objects

    Notes
    -----
    Can be checked to sese if an element in in this, can be iterated over,
    and as a current length of 118 elements.

    See Also
    --------
    periodic_table
    Element

    References
    ----------
    .. [1] N M O'Boyle, M Banck, C A James, C Morley, T Vandermeersch, and
       G R Hutchison. "Open Babel: An open chemical toolbox." J. Cheminf.
       (2011), 3, 33. DOI:10.1186/1758-2946-3-33
    '''
    __slots__ = ['number_to_elements', 'symbol_to_elements',
                 'name_to_elements', 'CAS_to_elements']
    def __init__(self, elements):
        self.number_to_elements = {}
        self.symbol_to_elements = {}
        self.name_to_elements = {}
        self.CAS_to_elements = {}
    
        for ele in elements:
            self.number_to_elements[ele.number] = ele
            self.symbol_to_elements[ele.symbol] = ele
            self.name_to_elements[ele.name] = ele
            self.name_to_elements[ele.name.lower()] = ele
            self.CAS_to_elements[ele.CAS] = ele
            
    def __contains__(self, key):
        for i in [self.symbol_to_elements, self.number_to_elements,
                  self.name_to_elements, self.CAS_to_elements]:
            if key in i:
                return True
        return False
        

    def __len__(self):
        return 118

    def __iter__(self):
        return iter([self.number_to_elements[i] for i in range(1,119)])

    def __getitem__(self, key):
        for i in [self.symbol_to_elements, self.number_to_elements,
                  self.name_to_elements, self.CAS_to_elements]:
            if key in i:
                return i[key]
        raise KeyError('Key is not in the periodic table.')

    def __getattr__(self, key):
        return self.__getitem__(key)


class Element(object):
    '''Class for storing data on chemical elements. Supports most common
    properties. If a property is not available, it is set to None.

    Attributes
    ----------
    number : int
        Atomic number
    name : str
        name
    symbol : str
        Elemental symbol
    MW : float
        Molecular weight
    CAS : str
        CAS number
    period : str
        Period in the periodic table
    group : str
        Group in the periodic table
    block : str
        Block in the periodic table
    AReneg : float
        Allred and Rochow electronegativity
    rcov : float
        Covalent radius, [Angstrom]
    rvdw : float
        Van der Waals radius, [Angstrom]
    maxbonds : float
        Maximum valence of a bond with this element
    elneg : float
        Pauling electronegativity
    ionization : float
        Ionization potential, [eV]
    ionization : float
        elaffinity affinity, [eV]
    protons : int
        Number of protons
    electrons : int
        Number of electrons of the element in the ground state
    InChI : str
        Standard InChI string of the element
    smiles : str
        Standard smiles string of the element

    '''
    __slots__ = ['number', 'symbol', 'name', 'CAS', 'MW', 'AReneg', 'rcov',
                 'rvdw', 'maxbonds', 'elneg', 'ionization', 'elaffinity',
                 'period', 'group', 'block', 'protons', 'electrons', 'InChI',
                 'smiles']

    def __init__(self, number, symbol, name, MW, CAS, AReneg, rcov, rvdw,
                 maxbonds, elneg, ionization, elaffinity, period, group, block):
        self.number = number
        self.symbol = symbol
        self.name = name
        self.MW = MW
        self.CAS = CAS

        self.period = period
        self.group = group
        self.block = block

        self.AReneg = AReneg
        self.rcov = rcov
        self.rvdw = rvdw
        self.maxbonds = maxbonds
        self.elneg = elneg
        self.ionization = ionization
        self.elaffinity = elaffinity
        
        self.protons = number
        self.electrons = number
        self.InChI = 'InChI=1S/' + self.symbol
        self.smiles = '[' + self.symbol + ']'


element_list = []
with open(os.path.join(folder, 'element.txt'), 'rb') as f:
    '''Load the file from OpenBabel with element data, and store it as both a
    list of elements first, and then as an instance of Periodic Table.'''
    for line in f:
        line = line.decode("utf-8")
        if line[0] != '#':
            values = to_num(line.strip('\n').split('\t'))
            number, symbol, AReneg, rcov, _, rvdw, maxbonds, MW, elneg, ionization, elaffinity, _, _, _, name = values
            number = int(number)
            AReneg = None if AReneg == 0 else AReneg
            rcov = None if rcov == 1.6 else rcov  # in Angstrom
            rvdw = None if rvdw == 2.0 else rvdw  # in Angstrom
            maxbonds = None if maxbonds == 6.0 else int(maxbonds)
            elneg = None if elneg == 0.0 else elneg
            ionization = None if ionization == 0.0 else ionization  # in eV
            elaffinity = None if elaffinity == 0.0 else elaffinity  # in eV
            block = [key for key in blocks.keys() if number in blocks[key]][0]
            period = periods[number-1]
            group = groups[number-1]

            ele = Element(number=number, symbol=symbol, name=name, MW=MW,
                          CAS=CAS_by_number[number-1], AReneg=AReneg,
                          rcov=rcov, rvdw=rvdw, maxbonds=maxbonds, elneg=elneg,
                          ionization=ionization, elaffinity=elaffinity,
                          block=block, period=period, group=group)
            element_list.append(ele)

periodic_table = PeriodicTable(element_list)
'''Single instance of the PeriodicTable class'''


def molecular_weight(atoms):
    r'''Calculates molecular weight of a molecule given a dictionary of its
    atoms and their counts, in the format {symbol: count}.

    .. math::
        MW = \sum_i n_i MW_i

    Parameters
    ----------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]

    Returns
    -------
    MW : float
        Calculated molecular weight [g/mol]

    Notes
    -----
    Elemental data is from rdkit, with CAS numbers added. An exception is
    raised if an incorrect element symbol is given. Elements up to 118 are
    supported.

    Examples
    --------
    >>> molecular_weight({'H': 12, 'C': 20, 'O': 5}) # DNA
    332.30628

    References
    ----------
    .. [1] RDKit: Open-source cheminformatics; http://www.rdkit.org
    '''
    MW = 0
    for i in atoms:
        if i in periodic_table:
            MW += periodic_table[i].MW*atoms[i]
        else:
            raise Exception('Molecule includes unknown atoms')
    return MW


def mass_fractions(atoms, MW=None):
    r'''Calculates the mass fractions of each element in a compound,
    given a dictionary of its atoms and their counts, in the format
    {symbol: count}.

    .. math::
        w_i =  \frac{n_i MW_i}{\sum_i n_i MW_i}

    Parameters
    ----------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]
    MW : float, optional
        Molecular weight, [g/mol]

    Returns
    -------
    mfracs : dict
        dictionary of mass fractions of individual atoms, indexed by symbol
        with proper capitalization, [-]

    Notes
    -----
    Molecular weight is optional, but speeds up the calculation slightly. It
    is calculated using the function `molecular_weight` if not specified.

    Elemental data is from rdkit, with CAS numbers added. An exception is
    raised if an incorrect element symbol is given. Elements up to 118 are
    supported.

    Examples
    --------
    >>> mass_fractions({'H': 12, 'C': 20, 'O': 5})
    {'H': 0.03639798802478244, 'C': 0.7228692758981262, 'O': 0.24073273607709128}

    References
    ----------
    .. [1] RDKit: Open-source cheminformatics; http://www.rdkit.org
    '''
    if not MW:
        MW = molecular_weight(atoms)
    mfracs = {}
    for i in atoms:
        if i in periodic_table:
            mfracs[i] = periodic_table[i].MW*atoms[i]/MW
        else:
            raise Exception('Molecule includes unknown atoms')
    return mfracs


def atom_fractions(atoms):
    r'''Calculates the atomic fractions of each element in a compound,
    given a dictionary of its atoms and their counts, in the format
    {symbol: count}.

    .. math::
        a_i =  \frac{n_i}{\sum_i n_i}

    Parameters
    ----------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]

    Returns
    -------
    afracs : dict
        dictionary of atomic fractions of individual atoms, indexed by symbol
        with proper capitalization, [-]

    Notes
    -----
    No actual data on the elements is used, so incorrect or custom compounds
    would not raise an error.

    Examples
    --------
    >>> atom_fractions({'H': 12, 'C': 20, 'O': 5})
    {'H': 0.32432432432432434, 'C': 0.5405405405405406, 'O': 0.13513513513513514}

    References
    ----------
    .. [1] RDKit: Open-source cheminformatics; http://www.rdkit.org
    '''
    count = sum(atoms.values())
    afracs = {}
    for i in atoms:
        afracs[i] = atoms[i]/count
    return afracs


def similarity_variable(atoms, MW=None):
    r'''Calculates the similarity variable of an compound, as defined in [1]_.
    Currently only applied for certain heat capacity estimation routines.

    .. math::
        \alpha = \frac{N}{MW} = \frac{\sum_i n_i}{\sum_i n_i MW_i}

    Parameters
    ----------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]
    MW : float, optional
        Molecular weight, [g/mol]

    Returns
    -------
    similarity_variable : float
        Similarity variable as defined in [1]_, [mol/g]

    Notes
    -----
    Molecular weight is optional, but speeds up the calculation slightly. It
    is calculated using the function `molecular_weight` if not specified.

    Examples
    --------
    >>> similarity_variable({'H': 32, 'C': 15})
    0.2212654140784498

    References
    ----------
    .. [1] Laštovka, Václav, Nasser Sallamie, and John M. Shaw. "A Similarity
       Variable for Estimating the Heat Capacity of Solid Organic Compounds:
       Part I. Fundamentals." Fluid Phase Equilibria 268, no. 1-2
       (June 25, 2008): 51-60. doi:10.1016/j.fluid.2008.03.019.
    '''
    if not MW:
        MW = molecular_weight(atoms)
    return sum(atoms.values())/MW


def atoms_to_Hill(atoms):
    r'''Determine the Hill formula of a compound, given a dictionary of its
    atoms and their counts, in the format {symbol: count}.

    Parameters
    ----------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]

    Returns
    -------
    Hill_formula : str
        Hill formula, [-]

    Notes
    -----
    The Hill system is as follows:

    If the chemical has 'C' in it, this is listed first, and then if it has
    'H' in it as well as 'C', then that goes next. All elements are sorted
    alphabetically afterwards, including 'H' if 'C' is not present.
    All elements are followed by their count, unless it is 1.

    Examples
    --------
    >>> atoms_to_Hill({'H': 5, 'C': 2, 'Br': 1})
    'C2H5Br'

    References
    ----------
    .. [1] Hill, Edwin A. “ON A SYSTEM OF INDEXING CHEMICAL LITERATURE;
       ADOPTED BY THE CLASSIFICATION DIVISION OF THE U. S. PATENT OFFICE.1.”
       Journal of the American Chemical Society 22, no. 8 (August 1, 1900):
       478-94. doi:10.1021/ja02046a005.
    '''
    def str_ele_count(ele):
        if atoms[ele] == 1:
            count = ''
        else:
            count = str(atoms[ele])
        return count
    atoms = atoms.copy()
    s = ''
    if 'C' in atoms.keys():
        s += 'C' + str_ele_count('C')
        del atoms['C']
        if 'H' in atoms.keys():
            s += 'H' + str_ele_count('H')
            del atoms['H']
        for ele in sorted(atoms.keys()):
            s += ele + str_ele_count(ele)
    else:
        for ele in sorted(atoms.keys()):
            s += ele + str_ele_count(ele)
    return s



_formula_p1 = re.compile(r'([A-Z][a-z]{0,2}\d*)')
_formula_p2 = re.compile(r'([A-Z][a-z]{0,2})')

def simple_formula_parser(formula):
    r'''Basic formula parser, primarily for obtaining element counts from 
    formulas as formated in PubChem. Handles formulas with integer counts, 
    but no brackets, no hydrates, no charges, no isotopes, and no group
    multipliers.
    
    Strips charges from the end of a formula first. Accepts repeated chemical
    units. Performs no sanity checking that elements are actually elements.
    As it uses regular expressions for matching, errors are mostly just ignored.
    
    Parameters
    ----------
    formula : str
        Formula string, very simply formats only.

    Returns
    -------
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]

    Notes
    -----
    Inspiration taken from the thermopyl project, at
    https://github.com/choderalab/thermopyl.

    Examples
    --------
    >>> simple_formula_parser('CO2')
    {'C': 1, 'O': 2}
    '''
    formula = formula.split('+')[0].split('-')[0]
    groups = _formula_p1.split(formula)[1::2]
    cnt = Counter()
    for group in groups:
        ele, count = _formula_p2.split(group)[1:]
        cnt[ele] += int(count) if count.isdigit() else 1
    return dict(cnt)
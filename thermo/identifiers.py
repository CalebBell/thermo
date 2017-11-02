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

__all__ = ['checkCAS', 'CAS_from_any', 'PubChem', 'MW', 'formula', 'smiles', 
           'InChI', 'InChI_Key', 'IUPAC_name', 'name', 'synonyms', 
           '_MixtureDict', 'mixture_from_any', 'cryogenics', 'dippr_compounds',
           'pubchem_db']
import os
from thermo.utils import to_num, CAS2int, int2CAS
from thermo.elements import periodic_table, homonuclear_elemental_gases, charge_from_formula, serialize_formula

folder = os.path.join(os.path.dirname(__file__), 'Identifiers')

def checkCAS(CASRN):
    '''Checks if a CAS number is valid. Returns False if the parser cannot 
    parse the given string..

    Parameters
    ----------
    CASRN : string
        A three-piece, dash-separated set of numbers

    Returns
    -------
    result : bool
        Boolean value if CASRN was valid. If parsing fails, return False also.

    Notes
    -----
    Check method is according to Chemical Abstract Society. However, no lookup
    to their service is performed; therefore, this function cannot detect
    false positives.

    Function also does not support additional separators, apart from '-'.
    
    CAS numbers up to the series 1 XXX XXX-XX-X are now being issued.
    
    A long can hold CAS numbers up to 2 147 483-64-7

    Examples
    --------
    >>> checkCAS('7732-18-5')
    True
    >>> checkCAS('77332-18-5')
    False
    '''
    try:
        check = CASRN[-1]
        CASRN = CASRN[::-1][1:]
        productsum = 0
        i = 1
        for num in CASRN:
            if num == '-':
                pass
            else:
                productsum += i*int(num)
                i += 1
        return (productsum % 10 == int(check))
    except:
        return False


class ChemicalMetadata(object):
    __slots__ = ['pubchemid', 'formula', 'MW', 'smiles', 'InChI', 'InChI_key',
                 'iupac_name', 'common_name', 'all_names', 'CAS', '_charge']
    def __repr__(self):
        return ('<ChemicalMetadata, name=%s, formula=%s, smiles=%s, MW=%g>'
                %(self.common_name, self.formula, self.smiles, self.MW))
        
    @property
    def charge(self):
        '''Charge of the species as an integer. Computed as a property as most
        species do not have a charge and so storing it would be a waste of 
        memory.
        '''
        try:
            return self._charge
        except AttributeError:
            self._charge = charge_from_formula(self.formula)
            return self._charge
        
    @property
    def CASs(self):
        return int2CAS(self.CAS)
    
    def __init__(self, pubchemid, CAS, formula, MW, smiles, InChI, InChI_key,
                 iupac_name, common_name, all_names):
        self.pubchemid = pubchemid
        self.CAS = CAS
        self.formula = formula
        self.MW = MW
        self.smiles = smiles
        self.InChI = InChI
        
        self.InChI_key = InChI_key
        self.iupac_name = iupac_name
        self.common_name = common_name
        self.all_names = all_names
        
        
    

class ChemicalMetadataDB(object):
    exclusion_options = [os.path.join(folder, 'dippr_2014_int.csv'),
                         os.path.join(folder, 'Chemicals with data.csv')]
    
    def __init__(self, create_pubchem_index=True, create_CAS_index=True,
                 create_name_index=True, create_smiles_index=True, 
                 create_InChI_index=True, create_InChI_key_index=True, 
                 create_formula_index=True,
                 restrict_identifiers_file=None, elements=True,
                 main_db=os.path.join(folder, 'chemical identifiers.tsv'),
                 user_dbs=[os.path.join(folder, 'chemical identifiers example user db.tsv'),
                           os.path.join(folder, 'Cation db.tsv'),
                           os.path.join(folder, 'Anion db.tsv'),
                           os.path.join(folder, 'Inorganic db.tsv')]):
        
        
        self.pubchem_index = {}
        self.smiles_index = {}
        self.InChI_index = {}
        self.InChI_key_index = {}
        self.name_index = {}
        self.CAS_index = {}
        self.formula_index = {}

        self.create_CAS_index = create_CAS_index
        self.create_pubchem_index = create_pubchem_index
        self.create_name_index = create_name_index
        self.create_smiles_index = create_smiles_index
        self.create_InChI_index = create_InChI_index
        self.create_InChI_key_index = create_InChI_key_index
        self.create_formula_index = create_formula_index
        self.restrict_identifiers_file = restrict_identifiers_file
        self.main_db = main_db
        self.user_dbs = user_dbs
        self.elements = elements
        
        if restrict_identifiers_file:
            self.load_included_indentifiers(restrict_identifiers_file)
        else:
            self.restrict_identifiers = False
            
        self.load(self.main_db, overwrite=False)
        for db in self.user_dbs:
            self.load(db, overwrite=True)
        self.load_elements()
        
    def load_elements(self):
        if not self.elements:
            return None
        for ele in periodic_table:
            
            CAS = int(ele.CAS.replace('-', '')) # Store as int for easier lookup
            all_names = [ele.name.lower()]
            
            obj = ChemicalMetadata(pubchemid=ele.PubChem, CAS=CAS, 
                                   formula=ele.symbol, MW=ele.MW, smiles=ele.smiles,
                                   InChI=ele.InChI, InChI_key=ele.InChI_key,
                                   iupac_name=ele.name.lower(), 
                                   common_name=ele.name.lower(), 
                                   all_names=all_names)
            
            
            if self.create_InChI_key_index:
                if ele.InChI_key in self.InChI_key_index:
                    if ele.number not in homonuclear_elemental_gases:
                        obj_old = self.InChI_key_index[ele.InChI_key]
                        for name in obj_old.all_names:
                            self.name_index[name] = obj    
                
                self.InChI_key_index[ele.InChI_key] = obj
            
            
            
            if self.create_CAS_index:
                self.CAS_index[CAS] = obj
            if self.create_pubchem_index:
                self.pubchem_index[ele.PubChem] = obj
            if self.create_smiles_index:
                self.smiles_index[ele.smiles] = obj
            if self.create_InChI_index:
                self.InChI_index[ele.InChI] = obj
                
            if self.create_name_index:
                if ele.number in homonuclear_elemental_gases:
                    for name in all_names:
                        self.name_index['monatomic ' + name] = obj    
                else:
                    for name in all_names:
                        self.name_index[name] = obj    

            if self.create_formula_index:
                self.formula_index[obj.formula] = obj


    def load(self, file_name, overwrite=False):
        f = open(file_name)
        for line in f:
            # This is effectively the documentation for the file format of the file
            values = line.rstrip('\n').split('\t')
            (pubchemid, CAS, formula, MW, smiles, InChI, InChI_key, iupac_name, common_name) = values[0:9]
            CAS = int(CAS.replace('-', '')) # Store as int for easier lookup
            # Handle the case of the db having more compounds than a user wants
            # to keep in memory
            if self.restrict_identifiers and CAS not in self.included_identifiers and file_name not in self.user_dbs:
                continue
            
            all_names = values[7:]
            pubchemid = int(pubchemid)

            obj = ChemicalMetadata(pubchemid, CAS, formula, float(MW), smiles,
                                    InChI, InChI_key, iupac_name, common_name, 
                                    all_names)
            
            # Lookup indexes
            if self.create_CAS_index:
                self.CAS_index[CAS] = obj
            if self.create_pubchem_index:
                self.pubchem_index[pubchemid] = obj
            if self.create_smiles_index:
                self.smiles_index[smiles] = obj
            if self.create_InChI_index:
                self.InChI_index[InChI] = obj
            if self.create_InChI_key_index:
                self.InChI_key_index[InChI_key] = obj
            if self.create_name_index:
                if overwrite:
                    for name in all_names:
                        self.name_index[name] = obj
                else:
                    for name in all_names:
                        if name in self.name_index:
                            pass
                        else:
                            self.name_index[name] = obj   
                            
            if self.create_formula_index:
                if obj.formula in self.formula_index:
                    if overwrite:
                        self.formula_index[obj.formula] = obj
                    else:
                        hit = self.formula_index[obj.formula]
                        if type(hit) != list:
                            if hit.CAS == obj.CAS:
                                # Replace repreated chemicals
                                self.formula_index[obj.formula] = hit
                            else:
    #                            self.formula_index[obj.formula] = None
                                self.formula_index[obj.formula] = [hit, obj]
                        else:
    #                        self.formula_index[obj.formula] = None
                            self.formula_index[obj.formula].append(obj)
                else:
                    self.formula_index[obj.formula] = obj
                    
        f.close()
    
    def load_included_indentifiers(self, file_name):
        '''Loads a file with newline-separated integers representing which 
        chemical should be kept in memory; ones not included are ignored.
        '''
        self.restrict_identifiers = True
        included_identifiers = set()       
        with open(file_name) as f:
            [included_identifiers.add(int(line)) for line in f]
        self.included_identifiers = included_identifiers
        
    @property
    def can_autoload(self):
        if not self.restrict_identifiers:
            return False
        if self.restrict_identifiers_file in self.exclusion_options:
            return True
        
        
    def autoload_next(self):
        if self.restrict_identifiers_file == self.exclusion_options[0]:
            self.restrict_identifiers_file = self.exclusion_options[1]
            self.load_included_indentifiers(self.restrict_identifiers_file)
            
        elif self.restrict_identifiers_file == self.exclusion_options[1]:
            self.restrict_identifiers_file = None
            self.restrict_identifiers = False
        else:
            return None
        
        self.load(self.main_db, overwrite=False)
        for db in self.user_dbs:
            self.load(db, overwrite=True)
        self.load_elements()
        return True
        
    def _search_autoload(self, identifier, index, autoload=True):
        if index:
            if identifier in index:
                return index[identifier]
            else:
                if autoload and self.can_autoload:
                    self.autoload_next()
                    return self._search_autoload(identifier, index, autoload)
        return False
    
    def search_pubchem(self, pubchem, autoload=True):
        if type(pubchem) != int:
            pubchem = int(pubchem)
        return self._search_autoload(pubchem, self.pubchem_index, autoload=autoload)
        
    def search_CAS(self, CAS, autoload=True):
        if type(CAS) != int:
            CAS = CAS2int(CAS)
        return self._search_autoload(CAS, self.CAS_index, autoload=autoload)

    def search_smiles(self, smiles, autoload=True):
        return self._search_autoload(smiles, self.smiles_index, autoload=autoload)

    def search_InChI(self, InChI, autoload=True):
        return self._search_autoload(InChI, self.InChI_index, autoload=autoload)

    def search_InChI_key(self, InChI_key, autoload=True):
        return self._search_autoload(InChI_key, self.InChI_key_index, autoload=autoload)

    def search_name(self, name, autoload=True):
        return self._search_autoload(name, self.name_index, autoload=autoload)
    
    def search_formula(self, formula, autoload=True):
        return self._search_autoload(formula, self.formula_index, autoload=autoload)

#pubchem_db = ChemicalMetadataDB(restrict_identifiers_file=os.path.join(folder, 'dippr_2014_int.csv'),
#                                create_pubchem_index=False, create_CAS_index=False,
#                 create_name_index=False, create_smiles_index=False, 
#                 create_InChI_index=False, create_InChI_key_index=False, 
#                 create_formula_index=False)
pubchem_db = ChemicalMetadataDB(restrict_identifiers_file=os.path.join(folder, 'dippr_2014_int.csv'))
#pubchem_db = ChemicalMetadataDB()


def CAS_from_any(ID, autoload=False):
    '''Looks up the CAS number of a chemical by searching and testing for the
    string being any of the following types of chemical identifiers:
    
    * Name, in IUPAC form or common form or a synonym registered in PubChem
    * InChI name, prefixed by 'InChI=1S/' or 'InChI=1/'
    * InChI key, prefixed by 'InChIKey='
    * PubChem CID, prefixed by 'PubChem='
    * SMILES (prefix with 'SMILES=' to ensure smiles parsing; ex.
      'C' will return Carbon as it is an element whereas the SMILES 
      interpretation for 'C' is methane)
    * CAS number (obsolete numbers may point to the current number)    

    If the input is an ID representing an element, the following additional 
    inputs may be specified as 
    
    * Atomic symbol (ex 'Na')
    * Atomic number (as a string)

    Parameters
    ----------
    ID : str
        One of the name formats described above

    Returns
    -------
    CASRN : string
        A three-piece, dash-separated set of numbers

    Notes
    -----
    An exception is raised if the name cannot be identified. The PubChem 
    database includes a wide variety of other synonyms, but these may not be
    present for all chemcials.

    Examples
    --------
    >>> CAS_from_any('water')
    '7732-18-5'
    >>> CAS_from_any('InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3')
    '64-17-5'
    >>> CAS_from_any('CCCCCCCCCC')
    '124-18-5'
    >>> CAS_from_any('InChIKey=LFQSCWFLJHTTHZ-UHFFFAOYSA-N')
    '64-17-5'
    >>> CAS_from_any('pubchem=702')
    '64-17-5'
    >>> CAS_from_any('O') # only elements can be specified by symbol
    '17778-80-2'
    '''
    ID = ID.strip()
    ID_lower = ID.lower()
    if ID in periodic_table:
        if periodic_table[ID].number not in homonuclear_elemental_gases:
            return periodic_table[ID].CAS
        else:
            for i in [periodic_table.symbol_to_elements, 
                      periodic_table.number_to_elements,
                      periodic_table.CAS_to_elements]:
                if i == periodic_table.number_to_elements:
                    if int(ID in i):
                        return periodic_table[int(ID)].CAS
                    
                else:
                    if ID in i:
                        return periodic_table[ID].CAS

    if checkCAS(ID):
        CAS_lookup = pubchem_db.search_CAS(ID, autoload)
        if CAS_lookup:
            return CAS_lookup.CASs
        
        # handle the case of synonyms
        CAS_alternate_loopup = pubchem_db.search_name(ID, autoload)
        if CAS_alternate_loopup:
            return CAS_alternate_loopup.CASs
        if not autoload:
            return CAS_from_any(ID, autoload=True)
        raise Exception('A valid CAS number was recognized, but is not in the database')
        
        
    
    ID_len = len(ID)
    if ID_len > 9:
        inchi_search = False
        # normal upper case is 'InChI=1S/'
        if ID_lower[0:9] == 'inchi=1s/':
            inchi_search = ID[9:]
        elif ID_lower[0:8] == 'inchi=1/':
            inchi_search = ID[8:]
        if inchi_search:
            inchi_lookup = pubchem_db.search_InChI(inchi_search, autoload)
            if inchi_lookup:
                return inchi_lookup.CASs
            else:
                if not autoload:
                    return CAS_from_any(ID, autoload=True)
                raise Exception('A valid InChI name was recognized, but it is not in the database')
        if ID_lower[0:9] == 'inchikey=':
            inchi_key_lookup = pubchem_db.search_InChI_key(ID[9:], autoload)
            if inchi_key_lookup:
                return inchi_key_lookup.CASs
            else:
                if not autoload:
                    return CAS_from_any(ID, autoload=True)
                raise Exception('A valid InChI Key was recognized, but it is not in the database')
    if ID_len > 8:
        if ID_lower[0:8] == 'pubchem=':
            pubchem_lookup = pubchem_db.search_pubchem(ID[8:], autoload)
            if pubchem_lookup:
                return pubchem_lookup.CASs
            else:
                if not autoload:
                    return CAS_from_any(ID, autoload=True)
                raise Exception('A PubChem integer identifier was recognized, but it is not in the database.')
    if ID_len > 7:
        if ID_lower[0:7] == 'smiles=':
            smiles_lookup = pubchem_db.search_smiles(ID[7:], autoload)
            if smiles_lookup:
                return smiles_lookup.CASs
            else:
                if not autoload:
                    return CAS_from_any(ID, autoload=True)
                raise Exception('A SMILES identifier was recognized, but it is not in the database.')

    # Try the smiles lookup anyway
    # Parsing SMILES is an option, but this is faster
    # Pybel API also prints messages to console on failure
    smiles_lookup = pubchem_db.search_smiles(ID, autoload)
    if smiles_lookup:
        return smiles_lookup.CASs
    
    try:
        formula_query = pubchem_db.search_formula(serialize_formula(ID), autoload)
        if formula_query and type(formula_query) == ChemicalMetadata:
            return formula_query.CASs
    except:
        pass
    
    # Try a direct lookup with the name - the fastest
    name_lookup = pubchem_db.search_name(ID, autoload)
    if name_lookup:
        return name_lookup.CASs

#     Permutate through various name options
    ID_no_space = ID.replace(' ', '')
    ID_no_space_dash = ID_no_space.replace('-', '')
    
    for name in [ID, ID_no_space, ID_no_space_dash]:
        for name2 in [name, name.lower()]:
            name_lookup = pubchem_db.search_name(name2, autoload)
            if name_lookup:
                return name_lookup.CASs
            
    
    if ID[-1] == ')' and '(' in ID:#
        # Try to matck in the form 'water (H2O)'
        first_identifier, second_identifier = ID[0:-1].split('(', 1)
        try:
            CAS1 = CAS_from_any(first_identifier)
            CAS2 = CAS_from_any(second_identifier)
            assert CAS1 == CAS2
            return CAS1
        except:
            pass
        
    if not autoload:
        return CAS_from_any(ID, autoload=True)
            
    raise Exception('Chemical name not recognized')





def PubChem(CASRN):
    '''Given a CASRN in the database, obtain the PubChem database
    number of the compound.

    Parameters
    ----------
    CASRN : string
        Valid CAS number in PubChem database [-]

    Returns
    -------
    pubchem : int
        PubChem database id, as an integer [-]

    Notes
    -----
    CASRN must be an indexing key in the pubchem database.

    Examples
    --------
    >>> PubChem('7732-18-5')
    962

    References
    ----------
    .. [1] Pubchem.
    '''
    return pubchem_db.search_CAS(CASRN).pubchemid



def MW(CASRN):
    '''Given a CASRN in the database, obtain the Molecular weight of the
    compound, if it is in the database.

    Parameters
    ----------
    CASRN : string
        Valid CAS number in PubChem database

    Returns
    -------
    MolecularWeight : float

    Notes
    -----
    CASRN must be an indexing key in the pubchem database. No MW Calculation is
    performed; nor are any historical isotopic corrections applied.

    Examples
    --------
    >>> MW('7732-18-5')
    18.01528

    References
    ----------
    .. [1] Pubchem.
    '''
    return pubchem_db.search_CAS(CASRN).MW


def formula(CASRN):
    '''
    >>> formula('7732-18-5')
    'H2O'
    '''
    return pubchem_db.search_CAS(CASRN).formula


def smiles(CASRN):
    '''
    >>> smiles('7732-18-5')
    'O'
    '''
    return pubchem_db.search_CAS(CASRN).smiles


def InChI(CASRN):
    '''
    >>> InChI('7732-18-5')
    'H2O/h1H2'
    '''
    return pubchem_db.search_CAS(CASRN).InChI


def InChI_Key(CASRN):
    '''
    >>> InChI_Key('7732-18-5')
    'XLYOFNOQVPJJNP-UHFFFAOYSA-N'
    '''
    return pubchem_db.search_CAS(CASRN).InChI_key


def IUPAC_name(CASRN):
    '''
    >>> IUPAC_name('7732-18-5')
    'oxidane'
    '''
    return pubchem_db.search_CAS(CASRN).iupac_name

def name(CASRN):
    '''
    >>> name('7732-18-5')
    'water'
    '''
    return pubchem_db.search_CAS(CASRN).common_name


def synonyms(CASRN):
    '''
    >>> synonyms('98-00-0')
    ['furan-2-ylmethanol', 'furfuryl alcohol', '2-furanmethanol', '2-furancarbinol', '2-furylmethanol', '2-furylcarbinol', '98-00-0', '2-furanylmethanol', 'furfuranol', 'furan-2-ylmethanol', '2-furfuryl alcohol', '5-hydroxymethylfuran', 'furfural alcohol', 'alpha-furylcarbinol', '2-hydroxymethylfuran', 'furfuralcohol', 'furylcarbinol', 'furyl alcohol', '2-(hydroxymethyl)furan', 'furan-2-yl-methanol', 'furfurylalcohol', 'furfurylcarb', 'methanol, (2-furyl)-', '2-furfurylalkohol', 'furan-2-methanol', '2-furane-methanol', '2-furanmethanol, homopolymer', '(2-furyl)methanol', '2-hydroxymethylfurane', 'furylcarbinol (van)', '2-furylmethan-1-ol', '25212-86-6', '93793-62-5', 'furanmethanol', 'polyfurfuryl alcohol', 'pffa', 'poly(furfurylalcohol)', 'poly-furfuryl alcohol', '(fur-2-yl)methanol', '.alpha.-furylcarbinol', '2-hydroxymethyl-furan', 'poly(furfuryl alcohol)', '.alpha.-furfuryl alcohol', 'agn-pc-04y237', 'h159', 'omega-hydroxypoly(furan-2,5-diylmethylene)', '(2-furyl)-methanol (furfurylalcohol)', '40795-25-3', '88161-36-8']
    '''
    return pubchem_db.search_CAS(CASRN).all_names


### DIPPR Database, chemical list only
# Obtained via the command:
# list(pd.read_excel('http://www.aiche.org/sites/default/files/docs/pages/sponsor_compound_list-2014.xlsx')['Unnamed: 2'])[2:]
# This is consistently faster than creating a list and then making the set.
dippr_compounds = set()
with open(os.path.join(folder, 'dippr_2014.csv')) as f:
    dippr_compounds.update(f.read().split('\n'))


_MixtureDict = {}
_MixtureDictLookup = {}
with open(os.path.join(folder, 'Mixtures Compositions.tsv')) as f:
    '''Read in a dict of 90 or so mixutres, their components, and synonyms.
    Small errors in mole fractions not adding to 1 are known.
    Errors in adding mass fraction are less common, present at the 5th decimal.
    TODO: Normalization
    Mass basis is assumed for all mixtures.
    '''
    next(f)
    for line in f:
        values = to_num(line.strip('\n').strip('\t').split('\t'))
        _name, _source, N = values[0:3]
        N = int(N)
        _CASs, _names, _ws, _zs = values[3:3+N], values[3+N:3+2*N], values[3+2*N:3+3*N], values[3+3*N:3+4*N]
        _syns = values[3+4*N:]
        if _syns:
            _syns = [i.lower() for i in _syns]
        _syns.append(_name.lower())
        _MixtureDict[_name] = {"CASs": _CASs, "N": N, "Source": _source,
                               "Names": _names, "ws": _ws, "zs": _zs,
                               "Synonyms": _syns}
        for syn in _syns:
            _MixtureDictLookup[syn] = _name


def mixture_from_any(ID):
    '''Looks up a string which may represent a mixture in the database of 
    thermo to determine the key by which the composition of that mixture can
    be obtained in the dictionary `_MixtureDict`.

    Parameters
    ----------
    ID : str
        A string or 1-element list containing the name which may represent a
        mixture.

    Returns
    -------
    key : str
        Key for access to the data on the mixture in `_MixtureDict`.

    Notes
    -----
    White space, '-', and upper case letters are removed in the search.

    Examples
    --------
    >>> mixture_from_any('R512A')
    'R512A'
    >>> mixture_from_any([u'air'])
    'Air'
    '''
    if type(ID) == list:
        if len(ID) == 1:
            ID = ID[0]
        else:
            raise Exception('If the input is a list, the list must contain only one item.')
    ID = ID.lower().strip()
    ID2 = ID.replace(' ', '')
    ID3 = ID.replace('-', '')
    for i in [ID, ID2, ID3]:
        if i in _MixtureDictLookup:
            return _MixtureDictLookup[i]
    raise Exception('Mixture name not recognized')


# TODO LIST OF REFRIGERANTS FOR USE IN HEAT TRANSFER CORRELATIONS

cryogenics = {'132259-10-0': 'Air', '7440-37-1': 'Argon', '630-08-0':
'carbon monoxide', '7782-39-0': 'deuterium', '7782-41-4': 'fluorine',
'7440-59-7': 'helium', '1333-74-0': 'hydrogen', '7439-90-9': 'krypton',
'74-82-8': 'methane', '7440-01-9': 'neon', '7727-37-9': 'nitrogen',
'7782-44-7': 'oxygen', '7440-63-3': 'xenon'}

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

__all__ = ['checkCAS', 'CASfromAny', 'PubChem', 'MW', 'formula', 'smiles', 
           'InChI', 'InChI_Key', 'IUPAC_name', 'name', 'synonyms', 
           '_MixtureDict', 'mixture_from_any', 'cryogenics', 'dippr_compounds',
           'pubchem_dict']
import os
from thermo.utils import to_num

folder = os.path.join(os.path.dirname(__file__), 'Identifiers')

def checkCAS(CASRN):
    '''Checks if a CAS number is valid. Crashes if not.

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

    Examples
    --------
    >>> checkCAS('7732-18-5')
    True
    >>> checkCAS('77332-18-5')
    False

    References
    ----------
    TODO
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




_cas_from_pubchem_dict = {}
_cas_from_smiles_dict = {}
_cas_from_inchi_dict = {}
_cas_from_inchikey_dict = {}
_cas_from_name_dict = {}
_cas_from_iupacname_dict = {}


pubchem_dict = {}


with open(os.path.join(folder, 'chemical identifiers.tsv')) as f:
    for line in f:
        values = line.rstrip('\n').split('\t')
        (pubchemid, CAS, formula, mw, smiles, inchi, inchikey, iupac_name, common_name) = values[0:9]
        allnames = values[7:]
        pubchemid = int(pubchemid)
        mw = float(mw)
        # Create lookup dictionaries
        _cas_from_pubchem_dict[pubchemid] = CAS
        _cas_from_smiles_dict[smiles] = CAS
        _cas_from_inchi_dict[inchi] = CAS
        _cas_from_inchikey_dict[inchikey] = CAS
        if iupac_name in _cas_from_iupacname_dict:
            # TODO: make unnecessary by removing previously unique identifiers,
            # which are no longer unique after making them lower case.
            pass
        else:
            _cas_from_iupacname_dict[iupac_name] = CAS
        for name in allnames:
            # TODO: make unnecessary by removing previously unique identifiers,
            # which are no longer unique after making them lower case.
            if name in _cas_from_name_dict:
                pass
            else:
                _cas_from_name_dict[name] = CAS

        pubchem_dict[CAS] = {'Pubchem ID': pubchemid, 'formula': formula,
        'MW': mw, 'SMILES': smiles, 'InChI': inchi, 'InChI Key': inchikey,
        'IUPAC name': iupac_name, 'common name': common_name, 'Names': allnames}
#        _pubchem_info(int(pubchemid), formula, float(mw), smiles, inchi, inchikey, iupac_name, common_name, allnames)

del pubchemid, formula, mw, smiles, inchi, inchikey, iupac_name, \
    common_name, allnames, name, line, f, CAS, values
#print len(_cas_from_name_dict)/float(len(_cas_from_pubchem_dict))
#print _cas_from_name_dict['Water'.lower()]
#print _pubchem_dict['7732-18-5']


def CASfromAny(ID):
    '''Input must be string
    First check if input is InChI, best defined format
    TODO: if int, format as rest-2digits-1digit and see if in dict. All CASs are in there.
    '''
    CASRN = None
    ID = ID.strip()
    if checkCAS(ID):
#        print 'CAS'
        try:
            a = pubchem_dict[ID]
            CASRN = ID
        except:
            try:
                CASRN = _cas_from_name_dict[ID]
            except:
                pass

    if type(ID) == type('') and len(ID) > 9 and not CASRN:
        # InChI strings and keys  are not stored with format for space reasons
        if ID[0:9] == 'InChI=1S/':
            try:
                CASRN = _cas_from_inchi_dict[ID[9:]]
            except:
                pass

        if ID[0:8] == 'InChI=1/' and not CASRN:
            try:
                CASRN =  _cas_from_inchi_dict[ID[8:]]
            except:
                pass

        if ID[0:9] == 'InChIKey=' and not CASRN:
            try:
                CASRN = _cas_from_inchikey_dict[ID[9:]]
            except:
                pass
    if type(ID) == type('') and not CASRN:
        # Parsing SMILES is an option, but this is faster
        # Pybel API also prints messages to console on failure
        try:
            CASRN = _cas_from_smiles_dict[ID]
        except:
            pass

    if type(ID) == type('') or type(ID) == type(1324) and not CASRN:
        try:
            intID = int(ID)
            CASRN = _cas_from_pubchem_dict[intID]
        except:
            pass
    if type(ID) == type('') and not CASRN:
#        print 'Early error'
        try:
            CASRN = _cas_from_iupacname_dict[ID.lower()]

#            print 'CASRN'
        except:
            pass
    if type(ID) == type('') and not CASRN:
        try:
            CASRN = _cas_from_name_dict[ID.lower()]
        except:
            try:
                ID = ID.replace(' ', '')
                CASRN = _cas_from_name_dict[ID.lower()]
            except:
                try:
                    ID = ID.replace('-', '')
                    CASRN = _cas_from_name_dict[ID.lower()]
                except:
                    CASRN = None
#            raise Exception('Not Found')
    return CASRN





def PubChem(CASRN):
    '''
    Given a CASRN in the database, obtain the PubChem database
    number of the compound.

    Parameters
    ----------
        CASRN : string
            Valid CAS number in PubChem database

    Returns
    -------
        pubchem : int
            PubChem database id, as an integer

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
    pubchem = pubchem_dict[CASRN]['Pubchem ID']
    return pubchem


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

    MolecularWeight = pubchem_dict[CASRN]['MW']
    return MolecularWeight


def formula(CASRN):
    '''
    >>> formula('7732-18-5')
    'H2O'
    '''
    Formula = pubchem_dict[CASRN]['formula']
    return Formula


def smiles(CASRN):
    '''
    >>> smiles('7732-18-5')
    'O'
    '''
    Smiles = pubchem_dict[CASRN]['SMILES']
    return Smiles


def InChI(CASRN):
    '''
    >>> InChI('7732-18-5')
    'H2O/h1H2'
    '''
    inchi = pubchem_dict[CASRN]['InChI']
    return inchi


def InChI_Key(CASRN):
    '''
    >>> InChI_Key('7732-18-5')
    'XLYOFNOQVPJJNP-UHFFFAOYSA-N'
    '''
    inchikey = pubchem_dict[CASRN]['InChI Key']
    return inchikey


def IUPAC_name(CASRN):
    '''
    >>> IUPAC_name('7732-18-5')
    'oxidane'
    '''
    iupac_name = pubchem_dict[CASRN]['IUPAC name']
    return iupac_name

def name(CASRN):
    '''
    >>> name('7732-18-5')
    'water'
    '''
    common_name = pubchem_dict[CASRN]['common name']
    return common_name


def synonyms(CASRN):
    '''
    >>> synonyms('98-00-0')
    ['furan-2-ylmethanol', 'furfuryl alcohol', '2-furanmethanol', '2-furancarbinol', '2-furylmethanol', '2-furylcarbinol', '98-00-0', '2-furanylmethanol', 'furfuranol', 'furan-2-ylmethanol', '2-furfuryl alcohol', '5-hydroxymethylfuran', 'furfural alcohol', 'alpha-furylcarbinol', '2-hydroxymethylfuran', 'furfuralcohol', 'furylcarbinol', 'furyl alcohol', '2-(hydroxymethyl)furan', 'furan-2-yl-methanol', 'furfurylalcohol', 'furfurylcarb', 'methanol, (2-furyl)-', '2-furfurylalkohol', 'furan-2-methanol', '2-furane-methanol', '2-furanmethanol, homopolymer', '(2-furyl)methanol', '2-hydroxymethylfurane', 'furylcarbinol (van)', '2-furylmethan-1-ol', '25212-86-6', '93793-62-5', 'furanmethanol', 'polyfurfuryl alcohol', 'pffa', 'poly(furfurylalcohol)', 'poly-furfuryl alcohol', '(fur-2-yl)methanol', '.alpha.-furylcarbinol', '2-hydroxymethyl-furan', 'poly(furfuryl alcohol)', '.alpha.-furfuryl alcohol', 'agn-pc-04y237', 'h159', 'omega-hydroxypoly(furan-2,5-diylmethylene)', '(2-furyl)-methanol (furfurylalcohol)', '40795-25-3', '88161-36-8']
    '''
    _synonyms = pubchem_dict[CASRN]['Names']
    return _synonyms





_MixtureDict = {}
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

def mixture_from_any(ID):
    if type(ID) == type([]):
        if len(ID) == 1:
            ID = ID[0]
        else:
            raise Exception('Input cannot be multiple components')
    ID = ID.lower()
    ID = ID.strip()
    ID2 = ID.replace(' ', '')
    ID3 = ID.replace('-', '')

    for i in _MixtureDict:
        d = _MixtureDict[i]["Synonyms"]
        if ID in d or ID2 in d or ID3 in d:
            return i

# TODO LIST OF REFRIGERANTS FOR USE IN HEAT TRANSFER CORRELATIONS

cryogenics = {'132259-10-0': 'Air', '7440-37-1': 'Argon', '630-08-0':
'carbon monoxide', '7782-39-0': 'deuterium', '7782-41-4': 'fluorine',
'7440-59-7': 'helium', '1333-74-0': 'hydrogen', '7439-90-9': 'krypton',
'74-82-8': 'methane', '7440-01-9': 'neon', '7727-37-9': 'nitrogen',
'7782-44-7': 'oxygen', '7440-63-3': 'xenon'}



### DIPPR Database, chemical list only
# Obtained via the command:
# list(pd.read_excel('http://www.aiche.org/sites/default/files/docs/pages/sponsor_compound_list-2014.xlsx')['Unnamed: 2'])[2:]
dippr_compounds = set()
with open(os.path.join(folder, 'dippr_2014.csv')) as f:
    dippr_compounds.update(f.read().split('\n'))











### Temporary functions which served a purpose, once


#def mid(ID):
#    print(CASfromAny(ID) + '\t' + ID + '\n')*20
#
#
##def paste(str):
##    from subprocess import Popen, PIPE
##    p = Popen(['xsel', '-bi'], stdin=PIPE)
##    p.communicate(input=str)
#
#
##def cid(ID):
##    paste((CASfromAny(ID) + '\t' + ID + '\n')*60)
#
#def fid(ID):
#    a = CASfromAny(ID)
#    paste((a + '\t' + IUPAC_name(a) + '\n')*60)
#
#def mancas(ID):
#    paste((ID + '\n')*60)
#
#def cas(ID):
#    a = CASfromAny(ID)
#    paste(a)
#
#def autocas():
#    while True:
#        data = raw_input()
#        cas(data)


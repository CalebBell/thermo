# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016-2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from numpy.testing import assert_allclose
import pytest
from thermo.identifiers import *
from thermo.utils import CAS2int
from thermo.elements import periodic_table, serialize_formula
import os

# Force the whole db to load
try:
    Chemical('asdfadsfasdfasdf')
except:
    pass

def test_dippr_list():
    assert 12916928773 == sum([CAS2int(i) for i in dippr_compounds])
    assert all([checkCAS(i) for i in dippr_compounds])


#@pytest.mark.slow
def test_pubchem_dict():
    assert all([checkCAS(i.CASs) for i in pubchem_db.CAS_index.values()])

def test_database_formulas():
    assert all([i.formula == serialize_formula(i.formula) for i in pubchem_db.CAS_index.values()])



def test_mixture_from_any():
    with pytest.raises(Exception):
        mixture_from_any(['water', 'methanol'])
    with pytest.raises(Exception):
        mixture_from_any('NOTAMIXTURE')

    for name in ['Air', 'air', u'Air', ['air']]:
        assert mixture_from_any(name) == 'Air'

    names = ['R-401A ', ' R-401A ', 'R401A ', 'r401a', 'r-401A', 'refrigerant-401A', 'refrigerant 401A']
    for name in names:
        assert mixture_from_any(name) == 'R401A'
        
    assert mixture_from_any('R512A') == 'R512A'
    assert mixture_from_any([u'air']) == 'Air'
    
    
def test_CAS_from_any():
    assert CAS_from_any('7732-18-5 ') == '7732-18-5'
    assert CAS_from_any('   7732  -18-5 ') == '7732-18-5'
    # direct in dictionary case
    assert CAS_from_any('water') == '7732-18-5'

    # Not in the main dict, but a synonym cas exists in the database
    assert CAS_from_any('136-16-3') == '582-36-5'

    assert CAS_from_any('inchi=1s/C2H6O/c1-2-3/h3H,2H2,1H3') == '64-17-5'
    assert CAS_from_any('inchi=1/C2H6O/c1-2-3/h3H,2H2,1H3') == '64-17-5'
    assert CAS_from_any('InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3') == '64-17-5'
    assert CAS_from_any('InChI=1/C2H6O/c1-2-3/h3H,2H2,1H3') == '64-17-5'
    assert CAS_from_any('InChIKey=LFQSCWFLJHTTHZ-UHFFFAOYSA-N') == '64-17-5'
    assert CAS_from_any('inchikey=LFQSCWFLJHTTHZ-UHFFFAOYSA-N') == '64-17-5'
    assert CAS_from_any(' inchikey=LFQSCWFLJHTTHZ-UHFFFAOYSA-N') == '64-17-5'

    assert CAS_from_any('InChI=1S/C6H15N/c1-5-6(2)7(3)4/h6H,5H2,1-4H3') == '921-04-0'
    
    assert CAS_from_any('pubchem=702') == '64-17-5'
    
    assert CAS_from_any('oxidane') == '7732-18-5'
    
    assert CAS_from_any('CCCCCCCCCC') == '124-18-5'
    assert CAS_from_any('SMILES=CCCCCCCCCC') == '124-18-5'

    assert CAS_from_any('S') == '7704-34-9'
    assert CAS_from_any('O') == '17778-80-2'

    assert CAS_from_any('InChiKey=QVGXLLKOCUKJST-UHFFFAOYSA-N') == '17778-80-2'
    # Just because it's an element does not mean the CAS number refers to the 
    # monatomic form unfortunately - this is the CAS for Monooxygen
    
    assert CAS_from_any('1') == '12385-13-6'
    
    
    assert CAS_from_any('HC2O4-') == '920-52-5'
    
    assert CAS_from_any('water (H2O)') == '7732-18-5'


    # Unknown inchi
    with pytest.raises(Exception):
        CAS_from_any('InChI=1S/C13H14N2O2S/c18-13-15-14-12(17-13)8-16-11-6-5-9-3-1-2-4-10(9)7-11/h5-7H,1-4,8H2,(H,15,18)')
    with pytest.raises(Exception):
        CAS_from_any('InChI=1/C13H14N2O2S/c18-13-15-14-12(17-13)8-16-11-6-5-9-3-1-2-4-10(9)7-11/h5-7H,1-4,8H2,(H,15,18)')
    with pytest.raises(Exception):
        CAS_from_any('InChIKey=QHHWJJGJTYTIPM-UHFFFAOYSA-N')
    # unknown pubchem
    with pytest.raises(Exception):
        CAS_from_any('pubchem=902100')
    # unknown CAS
    with pytest.raises(Exception):
        CAS_from_any('1411769-41-9')
        
    with pytest.raises(Exception):
        # This was parsed as Cerium for a little while
        CAS_from_any('Cellulose')
        
        
def test_periodic_table_variants():
    '''Do a lookup in the periodic table and compare vs CAS_from_any.
    '''
    ids = [periodic_table.CAS_to_elements, periodic_table.name_to_elements, periodic_table.symbol_to_elements]
    failed_CASs = []
    for thing in ids:
        for i in thing.keys():
            try:
                CAS_from_any(i)
            except:
                failed_CASs.append(periodic_table[i].name)
    assert 0 == len(set(failed_CASs))
    
    # Check only the 5 known diatomics have a diff case
    failed_CASs = []
    for thing in ids:
        for i in thing.keys():
            try:
                assert CAS_from_any(i) == periodic_table[i].CAS
            except:
                failed_CASs.append(periodic_table[i].name)
    assert set(['Chlorine', 'Fluorine', 'Hydrogen', 'Nitrogen', 'Oxygen']) == set(failed_CASs)     
    
    
    for CAS, d in periodic_table.CAS_to_elements.items():
        assert CAS_from_any(d.smiles) == CAS
        
    for CAS, d in periodic_table.CAS_to_elements.items():
        assert CAS_from_any('SMILES=' + d.smiles) == CAS
        
    for CAS, d in periodic_table.CAS_to_elements.items():
        assert CAS_from_any('InChI=1S/' + d.InChI) == CAS
        
    for CAS, d in periodic_table.CAS_to_elements.items():
        assert CAS_from_any('InChIKey=' + d.InChI_key) == CAS
        
        
    fail = 0
    for CAS, d in periodic_table.CAS_to_elements.items():
        
        if d.PubChem != None:
            assert CAS_from_any('PubChem=' + str(d.PubChem)) == CAS
        else:
            fail += 1
    assert fail == 9
    # 111 - 118 aren't in pubchem
    
            
    
def test_db_vs_ChemSep():
    import xml.etree.ElementTree as ET
    folder = os.path.join(os.path.dirname(__file__), 'Data')

    tree = ET.parse(os.path.join(folder, 'chemsep1.xml'))
    root = tree.getroot()

    data = {}
    for child in root:
        CAS = [i.attrib['value'] for i in child if i.tag == 'CAS'][0]
        name = [i.attrib['value'] for i in child if i.tag == 'CompoundID'][0]
        smiles = [i.attrib['value'] for i in child if i.tag == 'Smiles']
        if smiles:
            smiles = smiles[0]
        else:
            smiles = None
        
        data[CAS] = {'name': name, 'smiles': smiles}        
    
    for CAS, d in data.items():
        hit = pubchem_db.search_CAS(CAS)
        assert hit.CASs == CAS

    for CAS, d in data.items():
        assert CAS_from_any(CAS) == CAS

    # in an ideal world, the names would match too but ~15 don't. Adding more synonyms
    # might help.
#    try:
#        assert CAS_from_any(name) == CAS
#    except:
#        print(CAS, name)
#

    # In an ideal world we could also validate against their smiles
    # but that's proving difficult due to things like 1-hexene - 
    # is it 'CCCCC=C' or 'C=CCCCC'?
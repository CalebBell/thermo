# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import os
import pytest
from fluids.numerics import assert_close, assert_close1d
from thermo.group_contribution.joback import *
from thermo.group_contribution.joback import J_BIGGS_JOBACK_SMARTS_id_dict

from chemicals.identifiers import pubchem_db

folder = os.path.join(os.path.dirname(__file__), 'Data')

try:
    import rdkit
    from rdkit import Chem
except:
    rdkit = None
from thermo import Chemical

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_Joback_acetone():
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolDescriptors
    for i in [Chem.MolFromSmiles('CC(=O)C'), 'CC(=O)C']:
        ex = Joback(i) # Acetone example
        assert_close(ex.Tb(ex.counts), 322.11)
        assert_close(ex.Tm(ex.counts), 173.5)
        assert_close(ex.Tc(ex.counts), 500.5590049525365)
        assert_close(ex.Tc(ex.counts, 322.11), 500.5590049525365)
        assert_close(ex.Pc(ex.counts, ex.atom_count), 4802499.604994407)
        assert_close(ex.Vc(ex.counts), 0.0002095)
        assert_close(ex.Hf(ex.counts), -217830)
        assert_close(ex.Gf(ex.counts), -154540)
        assert_close(ex.Hfus(ex.counts), 5125)
        assert_close(ex.Hvap(ex.counts), 29018)
        assert_close1d(ex.Cpig_coeffs(ex.counts),[7.52, 0.26084, -0.0001207, 1.546e-08] )
        assert_close(ex.Cpig(300.0), 75.32642000000001)
        assert_close1d(ex.mul_coeffs(ex.counts), [839.11, -14.99])
        assert_close(ex.mul(300.0), 0.0002940378347162687)

    with pytest.raises(ValueError):
        # Raise an error if there are no groups matched
        obj = Joback('[Fe]')
        obj.estimate()

    # Test we can handle missing groups
    nitrobenzene = 'C1=CC=C(C=C1)[N+](=O)[O-]'
    obj = Joback(nitrobenzene)
    res = obj.estimate()
    assert res['mul_coeffs'] is None


@pytest.mark.fuzz
@pytest.mark.slow
@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_Joback_database():
    pubchem_db.autoload_main_db()

    f = open(os.path.join(folder, 'joback_log.txt'), 'w')
    from rdkit import Chem
    catalog = unifac_smarts = {i: Chem.MolFromSmarts(j) for i, j in J_BIGGS_JOBACK_SMARTS_id_dict.items()}
    lines = []
    for key in sorted(pubchem_db.CAS_index):
        chem_info = pubchem_db.CAS_index[key]
        try:
            mol = Chem.MolFromSmiles(chem_info.smiles)
            parsed = smarts_fragment(rdkitmol=mol, catalog=catalog, deduplicate=False)
            line = '%s\t%s\t%s\t%s\n' %(parsed[2], chem_info.CASs, chem_info.smiles, parsed[0])
        except Exception as e:
            line = '%s\t%s\t%s\n' %(chem_info.CASs, chem_info.smiles, e)
        lines.append(line)

    [f.write(line) for line in sorted(lines)]
    f.close()

# Maybe use this again if more work is done on Joback
del test_Joback_database


@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_Fedors():
    Vc, status, _, _, _ = Fedors('CCC(C)O')
    assert_close(Vc, 0.000274024)
    assert status == 'OK'

    Vc, status, _, _, _ = Fedors('C1=CC=C2C(=C1)C=CC3=CC4=C(C=CC5=CC=CC=C54)C=C32')
    assert_close(Vc, 0.00089246)
    assert status == 'OK'

    Vc, status, _, _, _ = Fedors('C1=CC=C(C=C1)O')
    assert_close(Vc, 0.00026668)
    assert status == 'OK'

    Vc, status, _, _, _ = Fedors('C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12C5=C%11C4=C3C3=C5C(=C81)C%10=C23')
    assert_close(Vc, 0.001969256)
    assert status == 'OK'

    Vc, status, _, _, _ = Fedors('C12C3C4C1C5C2C3C45')
    assert_close(Vc, 0.000485046)
    assert status == 'OK'

    Vc, status, _, _, _ = Fedors('O=[U](=O)=O')
    assert status != 'OK'


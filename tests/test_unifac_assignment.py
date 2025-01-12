'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2022, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
'''

from math import *

import pytest
from fluids.numerics import *

from thermo import Chemical
from chemicals.identifiers import search_chemical
from thermo.group_contribution.group_contribution_base import smarts_fragment_priority
from thermo.unifac import *
from thermo.unifac import DOUFSG, UNIFAC_SUBGROUPS, DOUFSG_SUBGROUPS, PSRK_SUBGROUPS, UNIFAC_LLE_SUBGROUPS, VTPRSG_SUBGROUPS

try:
    import rdkit
    from rdkit import Chem
except:
    rdkit = None

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_UNIFAC_original():
    # http://www.aim.env.uea.ac.uk/aim/info/UNIFACgroups.html was very helpful in this development

    rdkitmol = Chemical('17059-44-8').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {11: 2, 12: 1, 9: 3, 1: 1, 2: 1}
    assert success

    rdkitmol = Chemical('methanol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {15: 1}
    assert success

    rdkitmol = Chemical('water').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {16: 1}
    assert success

    rdkitmol = Chemical('4-hydroxybenzaldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {17: 1, 9: 4, 10: 1, 20: 1}
    assert success

    rdkitmol = Chemical('acetaldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {20: 1, 1: 1}
    assert success

    rdkitmol = Chemical('camphor').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {19: 1, 1: 3, 2: 2, 3: 1, 4: 2}
    assert success

    rdkitmol = Chemical('butyraldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {20: 1, 1: 1, 2: 2}
    assert success

    rdkitmol = Chemical('glutaraldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {20: 2, 2: 3}
    assert success

    rdkitmol = Chemical('triacetin').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {21: 3, 2: 2, 3: 1}
    assert success

    rdkitmol = Chemical('1,4-dioxin').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {27: 2, 6: 1}
    assert success

    rdkitmol = Chemical('phthalan').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {12: 1, 9: 4, 10: 1, 27: 1}
    assert success

    rdkitmol = Chemical('butyl propionate').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {22: 1, 1: 2, 2: 3}
    assert success

    rdkitmol = Chemical('ethyl formate').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {23: 1, 1: 1, 2: 1}
    assert success

    rdkitmol = Chemical('dimethyl ether').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {24: 1, 1: 1}
    assert success

    rdkitmol = Chemical('diethyl ether').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {25: 1, 1: 2, 2: 1}
    assert success

    rdkitmol = Chemical('diisopropyl ether').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {26: 1, 1: 4, 3: 1}
    assert success

    rdkitmol = Chemical('tetrahydrofuran').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {27: 1, 2: 3}
    assert success

    rdkitmol = Chemical('malondialdehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {2: 1, 20: 2}
    assert success

    rdkitmol = Chemical('glycolaldehyde').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {20: 1, 14: 1, 2: 1}
    assert success

    rdkitmol = Chemical('methylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {28: 1}
    assert success

    rdkitmol = Chemical('ethylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {29: 1, 1: 1}
    assert success

    rdkitmol = Chemical('isopropyl amine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {30: 1, 1: 2}
    assert success

    rdkitmol = Chemical('dimethylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {31: 1, 1: 1}
    assert success


    rdkitmol = Chemical('diethylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {32: 1, 1: 2, 2: 1}
    assert success

    rdkitmol = Chemical('diisopropyl amine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {33: 1, 1: 4, 3: 1}
    assert success

    rdkitmol = Chemical('trimethylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 34: 1}
    assert success

    rdkitmol = Chemical('triethylamine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 3, 2: 2, 35: 1}
    assert success

    rdkitmol = Chemical('aniline').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {9: 5, 36: 1}
    assert success

    rdkitmol = Chemical('pyridine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {37: 1}
    assert success


    rdkitmol = Chemical('2-methylpyridine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {38: 1, 1: 1}
    assert success

    rdkitmol = Chemical('2,3-Dimethylpyridine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 39: 1}
    assert success

    rdkitmol = Chemical('acetonitrile').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {40:1}
    assert success

    rdkitmol = Chemical('propionitrile').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 41: 1}
    assert success


    rdkitmol = Chemical('acetic acid').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 42: 1}
    assert success

    rdkitmol = Chemical('formic acid').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {43: 1}
    assert success

    rdkitmol = Chemical('Butyl chloride').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 2: 2, 44: 1}
    assert success

    rdkitmol = Chemical('2-chloropropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 45: 1}
    assert success

    rdkitmol = Chemical('2-Chloro-2-methylpropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 3, 46: 1}
    assert success

    rdkitmol = Chemical('Dichloromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {47: 1}
    assert success
    rdkitmol = Chemical('Ethylidene chloride').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 48: 1}
    assert success

    rdkitmol = Chemical('2,2-Dichloropropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 49: 1}
    assert success

    rdkitmol = Chemical('chloroform').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {50: 1}
    assert success
    rdkitmol = Chemical('1,1,1-Trichloroethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 51: 1}
    assert success

    rdkitmol = Chemical('tetrachloromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {52: 1}
    assert success

    rdkitmol = Chemical('chlorobenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {9: 5, 53: 1}
    assert success


    rdkitmol = Chemical('nitromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {54: 1}
    assert success

    rdkitmol = Chemical('1-nitropropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {55: 1, 1: 1, 2: 1}
    assert success

    rdkitmol = Chemical('2-nitropropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {56: 1, 1: 2}
    assert success

    rdkitmol = Chemical('nitrobenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {9: 5, 57: 1}
    assert success

    rdkitmol = Chemical('carbon disulphide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {58: 1}
    assert success

    rdkitmol = Chemical('methanethiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {59: 1}
    assert success

    rdkitmol = Chemical('ethanethiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 60: 1}
    assert success

    rdkitmol = Chemical('FURFURAL').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {61: 1}
    assert success

    rdkitmol = Chemical('1,2-ethanediol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {62: 1}
    assert success

    rdkitmol = Chemical('iodoethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 2: 1, 63: 1}
    assert success

    rdkitmol = Chemical('bromoethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 2: 1, 64: 1}
    assert success


    rdkitmol = Chemical('1-hexyne').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 2: 3, 65: 1}
    assert success

    rdkitmol = Chemical('2-hexyne').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 2: 2, 66: 1}
    assert success

    rdkitmol = Chemical('3-methylbut-1-yne').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 3: 1, 65: 1}
    assert success


    rdkitmol = Chemical('dimethylsulfoxide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {67: 1}
    assert success

    rdkitmol = Chemical('Acrylonitrile').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {68: 1}
    assert success

    rdkitmol = Chemical('Trichloroethylene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {8: 1, 69: 3}
    assert success

    rdkitmol = Chemical('2,3-Dimethyl-2-butene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 4, 70: 1}
    assert success

    rdkitmol = Chemical('hexafluorobenzene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {71: 6}
    assert success

    rdkitmol = Chemical('Dimethylformamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {72: 1}
    assert success

    rdkitmol = Chemical('n,n-diethylformamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 73: 1}
    assert success

    rdkitmol = Chemical('Perfluorohexane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {74: 2, 75: 4}
    assert success

    rdkitmol = Chemical('perfluoromethylcyclohexane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {74: 1, 75: 5, 76: 1}
    assert success

    rdkitmol = Chemical('methyl acrylate').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {77: 1, 5: 1, 1: 1}
    assert success

    # Si compounds not in database - TODO
    rdkitmol = Chem.MolFromSmiles('C[SiH3]') # methylsilane
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 78: 1}
    assert success


    # diethylsilane with H2 on the Si - pubchem disagrees but common chemistry is right
    # https://commonchemistry.cas.org/detail?cas_rn=542-91-6&search=diethylsilane
    rdkitmol = Chem.inchi.MolFromInchi('InChI=1S/C4H12Si/c1-3-5-4-2/h3-5H2,1-2H3')
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 2: 2, 79: 1}
    assert success

    # heptamethyltrisiloxane
    # https://commonchemistry.cas.org/detail?cas_rn=2895-07-0&search=heptamethyltrisiloxane
    rdkitmol = Chem.MolFromSmiles('O([SiH](C)C)[Si](O[Si](C)(C)C)(C)C')
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {83: 1, 84: 1, 81: 1, 1: 7}
    assert success

    # https://commonchemistry.cas.org/detail?cas_rn=1873-88-7&search=1,1,1,3,5,5,5-Heptamethyltrisiloxane
    # Different groups matching, don't test the results speficically may change
    rdkitmol = Chem.MolFromSmiles('O([Si](C)(C)C)[SiH](O[Si](C)(C)C)C')
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert success

    # 1,3-dimethyldisiloxane
    rdkitmol = Chem.MolFromSmiles('O([SiH2]C)[SiH2]C')
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {82: 1, 79: 1, 1: 2}
    assert success


    # 1,1,3,3-tetramethyldisiloxane
    rdkitmol = Chem.MolFromSmiles('O([SiH](C)C)[SiH](C)C')
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {83: 1, 80: 1, 1: 4}
    assert success

    # octamethylcyclotetrasiloxane
    rdkitmol = Chem.MolFromSmiles('C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1')
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {84: 4, 1: 8}
    assert success


    # N-Methyl-2-pyrrolidone NMP
    rdkitmol = Chemical('872-50-4').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {85: 1}
    assert success

    rdkitmol = Chemical('Trichlorofluoromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {86: 1}
    assert success

    rdkitmol = Chemical('tetrachloro-1,2-difluoroethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {87: 2}
    assert success

    rdkitmol = Chemical('dichlorofluoromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {88: 1}
    assert success

    rdkitmol = Chemical('1-chloro-1,2,2,2-tetrafluoroethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {74: 1, 89: 1}
    assert success

    rdkitmol = Chemical('1,2-dichlorotetrafluoroethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {90: 2}
    assert success

    rdkitmol = Chemical('chlorodifluoromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {91: 1}
    assert success

    rdkitmol = Chemical('chlorotrifluoromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {92: 1}
    assert success

    rdkitmol = Chemical('dichlorodifluoromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {93: 1}
    assert success

    # Issue with this at one point but fixed the smiles
    rdkitmol = Chemical('2-chloranyl-2,2-bis(fluoranyl)ethanoic acid').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {90: 1, 42: 1}
    assert success

    rdkitmol = Chemical('acetamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {94: 1, 1:1}
    assert success

    rdkitmol = Chemical('n-methylacetamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {95: 1, 1:1}
    assert success

    rdkitmol = Chemical('n-ethylacetamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {96: 1, 1: 2}
    assert success

    rdkitmol = Chemical('n,n-dimethylacetamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 97: 1}
    assert success

    # N-ethyl-N-methylacetamide
    rdkitmol = Chem.MolFromSmiles('CCN(C)C(=O)C')
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {98: 1, 1: 2}
    assert success

    rdkitmol = Chemical('n,n-diethylacetamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {99: 1, 1: 3}
    assert success

    rdkitmol = Chemical('2-ethoxyethanol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {100: 1, 1: 1, 2: 1}
    assert success

    rdkitmol = Chemical('1-Propanol, 2-ethoxy-').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {101: 1, 1: 2, 2: 1}
    assert success


    # Investigated this one wondering what was causing a difference, I like this match better
    rdkitmol = Chemical('2-hydroxyethyl ethanoate').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert success


    rdkitmol = Chemical('dimethylsulphide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {102: 1, 1: 1}
    assert success

    rdkitmol = Chemical('diethyl thioether').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {103: 1, 1: 2, 2: 1}
    assert success

    rdkitmol = Chemical('Diisopropyl sulfide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {104: 1, 1: 4, 3: 1}
    assert success

    rdkitmol = Chemical('morpholine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {105: 1}
    assert success


    rdkitmol = Chemical('thiophene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {106: 1}
    assert success

    rdkitmol = Chemical('2-methylthiophene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {107: 1, 1: 1}
    assert success


    rdkitmol = Chemical('2,3-dimethylthiophene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 2, 108: 1}
    assert success

    rdkitmol = Chemical('Methyl Isocyanate').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 1, 109: 1}
    assert success

    rdkitmol = Chemical('sulfolane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {118: 1, 2: 2}
    assert success

    rdkitmol = Chemical('2,4-dimethylsulfolane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {119: 1, 1: 2, 2: 1, 3: 1}
    assert success

    # Mistake on my part, wrongly identified this earlier
    rdkitmol = Chemical('1-ethylsulfonylethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {118: 1, 1: 2}
    assert success

    # Mistake on my part, wrongly identified this earlier
    rdkitmol = Chemical('2-ethylsulfonylpropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1: 3, 119: 1}
    assert success


    # Node that Imidazole itself does not match this definition; it is only imidazolium ionic liquids
    rdkitmol = Chem.MolFromSmiles('CCN1C=C[N+](=C1)C')
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {178: 1, 1: 2, 2: 1}
    assert success


    rdkitmol = Chem.MolFromSmiles('CCn1cc[n+](C)c1.FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F')
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {178: 1, 179: 1, 1: 2, 2: 1}
    assert success




@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_UNIFAC_failures():
    rdkitmol = Chemical('5-Methylfurfuryl alcohol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {11: 1, 12: 1, 9: 2, 14: 1}
    assert not success


"""
The following compounds were investigated and found to have a different fragmentation.
hydroxyacetone
"""

"""The following compounds need isotopic help
c = Chemical('n,n-dideuterio-1-phenyl-methanamine')
"""

"""The following compounds need a better tria-and-error algorithm
c = Chemical('5-[1,3-bis(oxidanyl)propan-2-ylamino]-1-(hydroxymethyl)cyclohexane-1,2,3,4-tetrol')
2-[4-[bis(2-hydroxyethyl)amino]-2,6-bis[4-[bis(2-hydroxyethyl)amino]phenyl]phenyl]ethanenitrile
"""

"""
The following compounds were processed by the DDBST web application but don't
actually have UNIFAC groups.
4,5-didehydropyridine

"""

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_UNIFAC_LLE_success():
    rdkitmol = Chemical('1-propanol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_LLE_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {15: 1}
    assert success

    rdkitmol = Chemical('2-propanol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_LLE_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {16: 1}
    assert success


    rdkitmol = Chemical('water').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_LLE_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {17: 1}
    assert success

    rdkitmol = Chemical('diethylene glycol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_LLE_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {49: 1}
    assert success

    rdkitmol = Chemical('trichloroethylene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_LLE_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {53: 1}
    assert success

    rdkitmol = Chemical('methylformamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_LLE_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {54: 1}
    assert success

    rdkitmol = Chemical('dimethylformamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_LLE_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {55: 1}
    assert success

    rdkitmol = Chemical('tetramethylene sulfone').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_LLE_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {56: 1}
    assert success

    rdkitmol = Chemical('DMSO').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=UNIFAC_LLE_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {57: 1}
    assert success


@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_PSRK_group_detection():
    rdkitmol = Chemical('ammonia').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {111: 1}
    assert success

    rdkitmol = Chemical('carbon monoxide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {112: 1}
    assert success

    rdkitmol = Chemical('hydrogen').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {113: 1}
    assert success

    rdkitmol = Chemical('deuterium').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {120: 1}
    assert success

    rdkitmol = Chemical('hydrogen sulfide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {114: 1}
    assert success

    rdkitmol = Chemical('nitrogen').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {115: 1}
    assert success

    rdkitmol = Chemical('argon').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {116: 1}
    assert success
    rdkitmol = Chemical('carbon dioxide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {117: 1}
    assert success

    rdkitmol = Chemical('methane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {118: 1}
    assert success

    rdkitmol = Chemical('oxygen').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {119: 1}
    assert success

    rdkitmol = Chemical('SO2').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {121: 1}
    assert success

    rdkitmol = Chemical('Nitric oxide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {122: 1}
    assert success

    rdkitmol = Chemical('N2O').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {123: 1}
    assert success

    rdkitmol = Chemical('SF6').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {124: 1}
    assert success

    rdkitmol = Chemical('He').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {125: 1}
    assert success

    rdkitmol = Chemical('Ne').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {126: 1}
    assert success

    rdkitmol = Chemical('Kr').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {127: 1}
    assert success

    rdkitmol = Chemical('Xe').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {128: 1}
    assert success

    rdkitmol = Chemical('HF').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {129: 1}
    assert success

    rdkitmol = Chemical('hydrogen chloride').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {130: 1}
    assert success

    rdkitmol = Chemical('hydrogen bromide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {131: 1}
    assert success

    rdkitmol = Chemical('hydrogen iodide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {132: 1}
    assert success

    rdkitmol = Chemical('Carbonyl sulfide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {133: 1}
    assert success

    rdkitmol = Chemical('2-propanethiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {134: 1, 1: 2}
    assert success

    rdkitmol = Chemical('2-methyl-2-propanethiol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {135: 1, 1: 3}
    assert success

    rdkitmol = Chemical('propyleneoxide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {136: 1, 1: 1}
    assert success

    rdkitmol = Chemical('ethylene').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {109: 1}
    assert success

    rdkitmol = Chemical('ethyne').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {110: 1}
    assert success

    rdkitmol = Chemical('2,3-epoxybutane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {137: 1, 1: 2}
    assert success

    rdkitmol = Chemical('2,2,3-trimethyloxirane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {138: 1, 1: 3}
    assert success

    rdkitmol = Chemical('ethylene Oxide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {139: 1}
    assert success

    rdkitmol = Chemical('2-methyl-1,2-epoxypropane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {140: 1, 1: 2}
    assert success

    rdkitmol = Chemical('2,3-dimethyl-2,3-epoxybutane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {141: 1, 1: 4}
    assert success

    rdkitmol = Chemical('fluorine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {142: 1}
    assert success

    rdkitmol = Chemical('Chlorine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {143: 1}
    assert success

    rdkitmol = Chemical('bromine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {144: 1}
    assert success

    rdkitmol = Chemical('hydrogen cyanide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {145: 1}
    assert success

    rdkitmol = Chemical('NO2').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {146: 1}
    assert success

    rdkitmol = Chemical('tetrafluoromethane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {147: 1}
    assert success

    rdkitmol = Chemical('ozone').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {148: 1}
    assert success

    rdkitmol = Chemical('nitrosyl chloride').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=PSRK_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {149: 1}
    assert success

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_DOUFSG_group_detection():
    rdkitmol = Chemical('1-propanol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1:1, 2:2, 14:1}
    assert success

    rdkitmol = Chemical('2-propanol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {3:1, 1:2, 81:1}
    assert success

    rdkitmol = Chemical('2-methylpropan-2-ol').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {1:3, 4:1, 82:1}
    assert success

    rdkitmol = Chemical('4-Methylpyridine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {37: 1, 11:1, 9:2}
    assert success

    rdkitmol = Chemical('2-Methylpyridine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {38: 1, 9: 3, 1: 1}
    assert success

    # Typo in article, this one does not actually match 39
    rdkitmol = Chemical('2,5-dimethylpyridine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {38: 1, 11: 1, 9: 2, 1: 1}
    assert success

    rdkitmol = Chemical('2,6-dimethylpyridine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {39: 1, 9: 3, 1: 2}
    assert success

    rdkitmol = Chemical('1,3,5-Trioxane').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    # This might become 83, 84 and 83 are a total dumpster fire
    assert assignment == {84:3}
    assert success

    rdkitmol = Chemical('cycloleucine').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {42: 1, 78: 4, 85:1}
    assert success

    rdkitmol = Chemical('N-Methylcaprolactam').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {86: 1, 78: 5}
    assert success


    rdkitmol = Chemical('oxiracetam').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    # print(assignment)
    # Disagrees with the assesment
    # assert assignment == {78: 2, 79: 1, 81: 1, 87: 1, 91:1}
    assert success

    # disagrees
    rdkitmol = Chemical('piracetam').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert success


    rdkitmol = Chemical('1-octyl-2-pyrrolidone').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {87: 1, 78: 3, 2: 6, 1:1}
    assert success

    rdkitmol = Chemical('1-(isopropyl)pyrrolidin-2-one').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    # May prefer 87 and a different fragmentation
    assert assignment == {1: 2, 78: 3, 88:1}
    assert success

    rdkitmol = Chemical('aziridinone, 1,3-bis(1,1-dimethylethyl)-').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {89: 1, 79: 1, 4: 1, 1:6}
    assert success

    rdkitmol = Chemical('2,2-Dichloroacetamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {91: 1, 48: 1}
    assert success

    # This one would be good for a solver, not a ton of matches
    rdkitmol = Chemical('N-Methylacetamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    # print(assignment)
    # assert assignment == {92: 1, 1: 1}
    assert success

    rdkitmol = Chemical('n-ethylacetamide').rdkitmol
    assignment, _, _, success, status = smarts_fragment_priority(catalog=DOUFSG_SUBGROUPS, rdkitmol=rdkitmol)
    assert assignment == {100: 1, 1:2}
    assert success


@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_VTPRSG_gaseous():
    test_cases = [
        (search_chemical('tetrafluoromethane').smiles, {146: 1}),  # CF4
        (search_chemical('ammonia').smiles, {300: 1}),  # NH3
        (search_chemical('carbon dioxide').smiles, {306: 1}),  # CO2
        (search_chemical('methane').smiles, {307: 1}),  # CH4
        (search_chemical('oxygen').smiles, {308: 1}),  # O2
        (search_chemical('argon').smiles, {305: 1}),
        (search_chemical('nitrogen').smiles, {304: 1}),  # N2
        (search_chemical('hydrogen sulfide').smiles, {303: 1}),  # H2S
        (search_chemical('hydrogen').smiles, {302: 1}),  # H2
        (search_chemical('carbon monoxide').smiles, {301: 1}),  # CO
        (search_chemical('sulfur dioxide').smiles, {310: 1}),  # SO2
        (search_chemical('nitrous oxide').smiles, {312: 1}),  # N2O
        (search_chemical('helium').smiles, {314: 1}),
        (search_chemical('neon').smiles, {315: 1}),
        (search_chemical('hydrogen chloride').smiles, {319: 1}),  # HCl
        (search_chemical('mercury').smiles, {345: 1}),
        (search_chemical('deuterium').smiles, {309: 1}),  # D2
    ]
    
    for smiles, expected in test_cases:
        rdkitmol = Chem.MolFromSmiles(smiles)
        assignment, *_, success, status = smarts_fragment_priority(
            catalog=VTPRSG_SUBGROUPS,
            rdkitmol=rdkitmol
        )
        assert assignment == expected, f"Failed for {smiles}: got {assignment}, expected {expected}"
        assert success, f"Assignment failed for {smiles}: {status}"


@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_VTPRSG_allene():
    rdkitmol = Chem.MolFromSmiles('C=C=C')
    assignment, *_, success, status = smarts_fragment_priority(
        catalog=VTPRSG_SUBGROUPS,
        rdkitmol=rdkitmol
    )
    assert assignment == {97: 1}
    assert success


    rdkitmol = Chem.MolFromSmiles('C=C')
    assignment, *_, success, status = smarts_fragment_priority(
        catalog=VTPRSG_SUBGROUPS,
        rdkitmol=rdkitmol
    )
    assert assignment == {250: 1}
    assert success


@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_VTPR_allene_unknown_correct_fragmentations():
    # can't find any references for what should match, making assumptions only
    group_98_smarts = VTPRSG[98].smarts
    group_98_pattern = Chem.MolFromSmarts(group_98_smarts)

    # Create test molecule
    rdkitmol = Chem.MolFromSmiles('CC=CC=C')

    # Check for matches
    matches = rdkitmol.GetSubstructMatches(group_98_pattern)
    assert len(matches) == 1

    group_99_smarts = VTPRSG[99].smarts
    group_99_pattern = Chem.MolFromSmarts(group_99_smarts)

    # Create test molecule
    rdkitmol = Chem.MolFromSmiles('CC=CC(=CC)C')

    # Check for matches
    matches = rdkitmol.GetSubstructMatches(group_99_pattern)
    assert len(matches) == 1



@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_ester_assignments():
    # Dictionary of test cases: chemical name -> expected assignment
    test_cases = {
        'methyl 2-methylpropanoate': {129: 1, 1: 3},
        'methyl 2-hydroxypropanoate': {129: 1, 81: 1, 1: 2},
        'methyl pivalate': {180: 1, 1: 4},
        'methyl 2-hydroxy-2-methylpropanoate': {180: 1, 82: 1, 1: 3}
    }
    
    for name, expected_assignment in test_cases.items():
        # Get SMILES and create RDKit molecule
        smiles = search_chemical(name).smiles
        rdkitmol = Chem.MolFromSmiles(smiles)
        
        # Get assignment
        assignment, *_, success, status = smarts_fragment_priority(
            catalog=VTPRSG_SUBGROUPS,
            rdkitmol=rdkitmol
        )
        
        # Assert both the assignment and success
        assert assignment == expected_assignment, f"Failed for {name}: got {assignment}, expected {expected_assignment}"
        assert success, f"Failed for {name}: matching was not successful"


@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_VTPR_fluorocarbon_groups():
    test_cases = {
        'difluoromethane': {140: 1},  # CF2H2 (pure R-32)
        '1,1-difluoroethane': {139: 1, 1: 1}  # CH3-CF2H
    }
    
    for name, expected_assignment in test_cases.items():
        smiles = search_chemical(name).smiles
        rdkitmol = Chem.MolFromSmiles(smiles)
        
        assignment, *_, success, status = smarts_fragment_priority(
            catalog=VTPRSG_SUBGROUPS,
            rdkitmol=rdkitmol
        )
        
        assert assignment == expected_assignment, f"Failed for {name}: got {assignment}, expected {expected_assignment}"
        assert success, f"Failed for {name}: matching was not successful"

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_VTPR_halogenated_methanes():
    test_cases = {
        'dichlorodifluoromethane': {143: 1},  # CF2Cl2 (R-12)
        'bromotrifluoromethane': {148: 1}     # CF3Br (R-13B1)
    }
    
    for name, expected_assignment in test_cases.items():
        smiles = search_chemical(name).smiles
        rdkitmol = Chem.MolFromSmiles(smiles)
        
        assignment, *_, success, status = smarts_fragment_priority(
            catalog=VTPRSG_SUBGROUPS,
            rdkitmol=rdkitmol
        )
        
        assert assignment == expected_assignment, f"Failed for {name}: got {assignment}, expected {expected_assignment}"
        assert success, f"Failed for {name}: matching was not successful"



@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_VTPR_CF2Cl_group():
    rdkitmol = Chem.MolFromSmiles('CCC(F)(F)Cl')
    
    assignment, *_, success, status = smarts_fragment_priority(
        catalog=VTPRSG_SUBGROUPS,
        rdkitmol=rdkitmol
    )
    
    assert assignment == {142: 1, 1: 1, 2: 1}, f"Got {assignment}, expected {{142: 1, 1: 1, 2: 1}}"
    assert success

@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_VTPR_AC_CHO():
    rdkitmol = Chem.MolFromSmiles('O=Cc1ccccc1')
    
    assignment, *_, success, status = smarts_fragment_priority(
        catalog=VTPRSG_SUBGROUPS,
        rdkitmol=rdkitmol
    )
    
    assert assignment == {116: 1, 9: 5}, f"Got {assignment}, expected {{116: 1, 9: 5}}"
    assert success

# def test_VTPR_smarts_assigned_to_all_groups():
#     # TODO, a few references to Dortmund that don't actually have groups
#     none_priority_groups = [
#         "Group {} ({})".format(i.group_id, i.group)
#         for i in VTPRSG_SUBGROUPS 
#         if i.priority is None
#     ]
    
#     assert len(none_priority_groups) == 0, (
#         "Found {} groups with None priority:\n"
#         "{}".format(
#             len(none_priority_groups), 
#             "\n".join(none_priority_groups)
#         )
#     )
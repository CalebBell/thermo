'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import json
from math import *

import pytest
from chemicals import *
from chemicals.utils import hash_any_primitive
from fluids.numerics import *
import pickle
from thermo import *

assorted_IDs = ['1,2-ethanediol', '1,3-butadiene', '1,3-propanediol', '1,4-Dioxane', '2-Nitrotoluene', 
'2-methylbutane', '3-Nitrotoluene', '4-Nitrotoluene', 'Acetic Acid', 'Acetic anhydride', 'Ammonium Chloride',
 'Aniline', 'CO2', 'COS', 'CS2', 'Carbon monoxide', 'Chloroform', 'Cumene', 'DMSO', 'Diethanolamine',
  'Dimethyl Carbonate', 'ETBE', 'Ethyl Lactate', 'Ethylamine', 'H2S', 'Isoamyl Acetate', 'Isoamyl Alcohol', 
  'MTBE', 'Methyl Acetate', 'Methyl Isobutyl Ketone', 'Methyl Methacrylate', 'Methylal', 'MonoChlorobenzene', 
  'Monoethanolamine', 'N2H4', 'Nitrobenzene', 'Phenol', 'Propylene oxide', 'Pyridine', 'SO2', 'Sulfur Trioxide', 
  'Triethanolamine', 'Vinyl Chloride', 'acetanilide', 'ammonia', 'argon', 'benzene', 'benzoic acid', 'carbon', 
  'cyclohexane', 'decane', 'dimethyl ether', 'eicosane', 'ethane', 'ethanol', 'ethyl mercaptan', 'ethylbenzene',
   'ethylene', 'ethylene oxide', 'ethylenediamine', 'furfural', 'furfuryl alcohol', 'glycerol', 'heptane',
    'hexane', 'hydrochloric acid', 'hydrogen', 'hydrogen cyanide', 'hydroxylamine hydrochloride', 'isobutane',
     'isopropanol', 'ketene', 'maleic anhydride', 'methane', 'methanol', 'methyl mercaptan', 'n-Propyl Acetate', 
     'n-butane', 'n-octane', 'nitric acid', 'nitrogen', 'oxygen', 'p-Phenetidine', 'pentane', 'potassium iodide', 
     'propane', 'propylene', 'propylene glycol', 'styrene', 'tetrahydrofuran', 'toluene', 'triethylene glycol', 'water']

@pytest.mark.fuzz
@pytest.mark.slow
def test_ChemicalConstantsPackage_from_json_as_json_large():
    create_compounds = []
    for k in dippr_compounds():
        try:
            if search_chemical(k) is not None:
                create_compounds.append(k)
        except:
            pass

    # Test constants_from_IDs
    obj = ChemicalConstantsPackage.constants_from_IDs(create_compounds)
    obj2 = ChemicalConstantsPackage.from_json(json.loads(json.dumps(obj.as_json())))

    assert hash(obj) == hash(obj2)
    assert obj == obj2
    assert id(obj) != id(obj2)

    # Test correlations_from_IDs
    obj = ChemicalConstantsPackage.correlations_from_IDs(create_compounds)
    obj2 = PropertyCorrelationsPackage.from_json(json.loads(json.dumps(obj.as_json())))
    assert hash(obj) == hash(obj2)
    assert obj == obj2
    assert id(obj) != id(obj2)

    assert obj != int
    assert obj != float

def test_ChemicalConstantsPackage_json_version_exported():
    constants = ChemicalConstantsPackage(MWs=[18.01528, 106.165], names=['water', 'm-xylene'])
    string = json.dumps(constants.as_json())
    c2 = ChemicalConstantsPackage.from_json(json.loads(string))
    assert 'py/object' in string
    assert 'json_version' in string
    assert not hasattr(c2, 'json_version')

def test_ChemicalConstantsPackage_json_export_does_not_change_hashes():
    # There was a nasty bug where the hashing function was changing its result
    # every call
    obj = ChemicalConstantsPackage.correlations_from_IDs(['hexane'])
    hashes_orig = [hash_any_primitive(getattr(obj, k)) for k in obj.correlations]
    copy = obj.as_json()
    hashes_after = [hash_any_primitive(getattr(obj, k)) for k in obj.correlations]
    assert hashes_orig == hashes_after


def test_ChemicalConstantsPackage_json_export_sane_recursion():

    # It might be nice to do something about the duplicate EOSs, but they could be different
    # Really still feels like a different structure for that would be better.
    obj = ChemicalConstantsPackage.correlations_from_IDs(['methane', 'ethane'])
    assert 3 == json.dumps(obj.as_json()).count('VaporPressure')

def test_ChemicalConstantsPackage_json_export_same_output():
    obj = ChemicalConstantsPackage.correlations_from_IDs(['hexane'])
    obj2 = PropertyCorrelationsPackage.from_json(json.loads(json.dumps(obj.as_json())))

    assert hash_any_primitive(obj.constants) == hash_any_primitive(obj2.constants)
    for prop in obj.pure_correlations:
        assert hash_any_primitive(getattr(obj, prop)) ==  hash_any_primitive(getattr(obj2, prop))
    assert hash_any_primitive(obj.VaporPressures) == hash_any_primitive(obj2.VaporPressures)
    assert hash_any_primitive(obj.ViscosityGases) == hash_any_primitive(obj2.ViscosityGases)
    assert hash(obj.SurfaceTensionMixture) == hash(obj2.SurfaceTensionMixture)
    assert hash(obj.VolumeGasMixture) == hash(obj2.VolumeGasMixture)
    for prop in obj.mixture_correlations:
        assert hash_any_primitive(getattr(obj, prop)) ==  hash_any_primitive(getattr(obj2, prop))


    assert hash(obj) == hash(obj2)
    assert obj == obj2

def test_ChemicalConstantsPackage_wrong_behaviors():
    obj = ChemicalConstantsPackage.correlations_from_IDs(['7647-19-0'])
    obj.VolumeLiquids[0].eos is None
    assert obj != int
    assert obj != float


def test_lemmon2000_package():
    Ts = (150.0, 200.0, 300.0, 1000.0, 2000.0)
    CoolProp_Cps = [29.030484473246823, 29.03511836728048, 29.103801681330573, 33.046833525551676, 36.210748112152906]
    for T, Cp in zip(Ts, CoolProp_Cps):
        assert_close(Cp, lemmon2000_correlations.HeatCapacityGases[0](T), rtol=2e-7)


def test_compound_index():
    obj = ChemicalConstantsPackage(MWs=[18.01528, 106.165], names=['water', 'm-xylene'],
                             CASs=['7732-18-5', '108-38-3'],
                             InChI_Keys=['XLYOFNOQVPJJNP-UHFFFAOYSA-N', 'IVSZLXZYQVIEFR-UHFFFAOYSA-N'],
                             InChIs=['H2O/h1H2', 'C8H10/c1-7-4-3-5-8(2)6-7/h3-6H,1-2H3'],
                             smiless=['O', 'CC1=CC(=CC=C1)C'], PubChems=[962, 7929],)
    assert 0 == obj.compound_index(name='water')
    assert 1 == obj.compound_index(name='m-xylene')
    assert 1 == obj.compound_index(PubChem=7929)
    assert 0 == obj.compound_index(smiles='O')
    assert 0 == obj.compound_index(CAS='7732-18-5')
    assert 0 == obj.compound_index(CAS='7732-18-5')
    assert 0 == obj.compound_index(InChI='H2O/h1H2')
    assert 1 == obj.compound_index(InChI_Key='IVSZLXZYQVIEFR-UHFFFAOYSA-N')


def test_add_ChemicalConstantsPackage():
    a = ChemicalConstantsPackage.constants_from_IDs(IDs=['water', 'hexane'])
    b = ChemicalConstantsPackage.constants_from_IDs(IDs=['toluene'])
    c = a + b

    c_good = ChemicalConstantsPackage.constants_from_IDs(IDs=['water', 'hexane', 'toluene'])
    assert c == c_good

def test_add_PropertyCorrelationsPackage():
    a = ChemicalConstantsPackage.correlations_from_IDs(IDs=['water', 'hexane'])
    b = ChemicalConstantsPackage.correlations_from_IDs(IDs=['toluene'])
    c = a + b

    c_good = ChemicalConstantsPackage.correlations_from_IDs(IDs=['water', 'hexane', 'toluene'])
    assert c == c_good


def test_ChemicalConstantsPackage_pickle():
    # Pickle checks
    model = ChemicalConstantsPackage.constants_from_IDs(IDs=['water', 'hexane'])
    model_pickle = pickle.loads(pickle.dumps(model))
    assert model_pickle == model

def test_correlations_from_IDs_pickle():
    # Pickle checks
    model = ChemicalConstantsPackage.correlations_from_IDs(IDs=['water', 'hexane'])
    model_pickle = pickle.loads(pickle.dumps(model))
    assert model_pickle == model

def test_correlations_from_IDs_larger_pickle():
    # Pickle checks
    model = ChemicalConstantsPackage.correlations_from_IDs(IDs=assorted_IDs)
    model_pickle = pickle.loads(pickle.dumps(model))
    assert model_pickle == model

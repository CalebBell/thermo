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
                             CASs=['7732-18-5', '108-38-3'],atomss=[{'H': 2, 'O': 1}, {'C': 8, 'H': 10}],
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

    assert ('C', 'H', 'O') == obj.unique_atoms


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
    model.unique_atoms
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


def test_chemical_package_recreation():
    obj = PropertyCorrelationsPackage(VaporPressures=[VaporPressure(extrapolation="AntoineAB|DIPPR101_ABC", method="EXP_POLY_FIT", exp_poly_fit=(85.53500000000001, 369.88, [-6.614459112569553e-18, 1.3568029167021588e-14, -1.2023152282336466e-11, 6.026039040950274e-09, -1.877734093773071e-06, 0.00037620249872919755, -0.048277894617307984, 3.790545023359657, -137.90784855852178]))], 
                                        VolumeLiquids=[VolumeLiquid(extrapolation="constant", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53500000000001, 359.89, [1.938305828541795e-22, -3.1476633433892663e-19, 2.1728188240044968e-16, -8.304069912036783e-14, 1.9173728759045994e-11, -2.7331397706706945e-09, 2.346460759888426e-07, -1.1005126799030672e-05, 0.00027390337689920513]))], 
                                        HeatCapacityGases=[HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367]))], 
                                        ViscosityLiquids=[ViscosityLiquid(extrapolation="linear", method="EXP_POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, exp_poly_fit=(85.53500000000001, 369.88, [-1.464186051602898e-17, 2.3687106330094663e-14, -1.601177693693127e-11, 5.837768086076859e-09, -1.2292283268696937e-06, 0.00014590412750959653, -0.0081324465457914, -0.005575029473976978, 8.728914946382764]))], 
                                        ViscosityGases=[ViscosityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-3.1840315590772447e-24, 5.632245762287636e-21, -4.211563759618865e-18, 1.7309219264976467e-15, -4.25623447818058e-13, 6.379502491722484e-11, -5.653736202867734e-09, 2.934273667761606e-07, -4.688742520151596e-06]))], 
                                        ThermalConductivityLiquids=[ThermalConductivityLiquid(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-1.119942102768228e-20, 2.0010259140958334e-17, -1.5432664732534744e-14, 6.710420754858951e-12, -1.8195587835583956e-09, 3.271396846047887e-07, -4.022072549142343e-05, 0.0025702260414860677, 0.15009818638364272]))], 
                                        ThermalConductivityGases=[ThermalConductivityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-4.225968132998871e-21, 7.499602824205188e-18, -5.628911367597151e-15, 2.323826706645111e-12, -5.747490655145977e-10, 8.692266787645703e-08, -7.692555396195328e-06, 0.0004075191640226901, -0.009175197596970735]))], 
                                        SurfaceTensions=[SurfaceTension(Tc=369.83, extrapolation="DIPPR106_AB", method="EXP_POLY_FIT_LN_TAU", exp_poly_fit_ln_tau=(193.15, 366.48, 369.83, [-4.69903867038229e-05, -0.001167676479018507, -0.01245104796692622, -0.07449082604785806, -0.27398619941324853, -0.6372368552001203, -0.9215870661729839, 0.4680106704255822, -3.2163790497734346]))], 
                                        ViscosityGasMixtureObj=ViscosityGasMixture(MWs=[44.09562], molecular_diameters=[], Stockmayers=[], CASs=[], correct_pressure_pure=False, method="HERNING_ZIPPERER", ViscosityGases=[ViscosityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-3.1840315590772447e-24, 5.632245762287636e-21, -4.211563759618865e-18, 1.7309219264976467e-15, -4.25623447818058e-13, 6.379502491722484e-11, -5.653736202867734e-09, 2.934273667761606e-07, -4.688742520151596e-06]))]), 
                                        ViscosityLiquidMixtureObj=ViscosityLiquidMixture(MWs=[], CASs=[], correct_pressure_pure=False, method="Logarithmic mixing, molar", ViscosityLiquids=[ViscosityLiquid(extrapolation="linear", method="EXP_POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, exp_poly_fit=(85.53500000000001, 369.88, [-1.464186051602898e-17, 2.3687106330094663e-14, -1.601177693693127e-11, 5.837768086076859e-09, -1.2292283268696937e-06, 0.00014590412750959653, -0.0081324465457914, -0.005575029473976978, 8.728914946382764]))]), 
                                        ThermalConductivityGasMixtureObj=ThermalConductivityGasMixture(MWs=[44.09562], Tbs=[231.04], CASs=[], correct_pressure_pure=False, method="LINDSAY_BROMLEY", ThermalConductivityGases=[ThermalConductivityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-4.225968132998871e-21, 7.499602824205188e-18, -5.628911367597151e-15, 2.323826706645111e-12, -5.747490655145977e-10, 8.692266787645703e-08, -7.692555396195328e-06, 0.0004075191640226901, -0.009175197596970735]))], ViscosityGases=[ViscosityGas(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-3.1840315590772447e-24, 5.632245762287636e-21, -4.211563759618865e-18, 1.7309219264976467e-15, -4.25623447818058e-13, 6.379502491722484e-11, -5.653736202867734e-09, 2.934273667761606e-07, -4.688742520151596e-06]))]), 
                                        ThermalConductivityLiquidMixtureObj=ThermalConductivityLiquidMixture(MWs=[], CASs=[], correct_pressure_pure=False, method="DIPPR_9H", ThermalConductivityLiquids=[ThermalConductivityLiquid(extrapolation="linear", method="POLY_FIT", method_P="NEGLECT_P", tabular_extrapolation_permitted=True, poly_fit=(85.53, 350.0, [-1.119942102768228e-20, 2.0010259140958334e-17, -1.5432664732534744e-14, 6.710420754858951e-12, -1.8195587835583956e-09, 3.271396846047887e-07, -4.022072549142343e-05, 0.0025702260414860677, 0.15009818638364272]))]), 
                                        SurfaceTensionMixtureObj=SurfaceTensionMixture(MWs=[44.09562], Tbs=[231.04], Tcs=[369.83], CASs=['74-98-6'], correct_pressure_pure=False, method="Winterfeld, Scriven, and Davis (1978)", SurfaceTensions=[SurfaceTension(Tc=369.83, extrapolation="DIPPR106_AB", method="EXP_POLY_FIT_LN_TAU", exp_poly_fit_ln_tau=(193.15, 366.48, 369.83, [-4.69903867038229e-05, -0.001167676479018507, -0.01245104796692622, -0.07449082604785806, -0.27398619941324853, -0.6372368552001203, -0.9215870661729839, 0.4680106704255822, -3.2163790497734346]))], VolumeLiquids=[SurfaceTension(Tc=369.83, extrapolation="DIPPR106_AB", method="EXP_POLY_FIT_LN_TAU", exp_poly_fit_ln_tau=(193.15, 366.48, 369.83, [-4.69903867038229e-05, -0.001167676479018507, -0.01245104796692622, -0.07449082604785806, -0.27398619941324853, -0.6372368552001203, -0.9215870661729839, 0.4680106704255822, -3.2163790497734346]))]), 
                                        constants=ChemicalConstantsPackage(atomss=[{'C': 3, 'H': 8}], CASs=['74-98-6'], Gfgs=[-24008.76000000004], Hcs=[-2219332.0], Hcs_lower=[-2043286.016], Hcs_lower_mass=[-46337618.47548578], Hcs_mass=[-50329987.42278712], Hfgs=[-104390.0], MWs=[44.09562], names=['propane'], omegas=[0.152], Pcs=[4248000.0], Sfgs=[-269.5999999999999], Tbs=[231.04], Tcs=[369.83], Vml_STPs=[8.982551425831519e-05], Vml_60Fs=[8.721932949945705e-05]), 
                                        skip_missing=True)

    copy = eval(str(obj))
    assert obj == copy

    obj = ChemicalConstantsPackage.correlations_from_IDs(IDs=['water', 'hexane', 'toluene'])
    copy = eval(str(obj))
    assert obj == copy

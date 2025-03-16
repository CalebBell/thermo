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
from thermo.functional_groups import FG_ALCOHOL, FG_ORGANIC

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

def test_chemical_package_recreation_another_issue():
    constants = ChemicalConstantsPackage(atom_fractions=[{'H': 0.6666666666666666, 'C': 0.16666666666666666, 'O': 0.16666666666666666}], atomss=[{'H': 4, 'C': 1, 'O': 1}],
                                        CASs=['67-56-1'], charges=[0], conductivities=[4.4e-05], dipoles=[1.7],
                                        formulas=['CH4O'], Gfgs=[-162000.13000000003], Gfgs_mass=[-5055890.325967345], Hcs=[-764464.0], Hcs_lower=[-676489.1], Hcs_lower_mass=[-21112666.368306957], Hcs_mass=[-23858290.373904638],
                                        Hfgs=[-200700.0], Hfgs_mass=[-6263681.321870828], Hfus_Tms=[3215.0000000000005], Hfus_Tms_mass=[100337.49601302797], Hvap_298s=[37486.47944178592], Hvap_298s_mass=[1169922.078237216],
                                        Hvap_Tbs=[35170.873821707064], Hvap_Tbs_mass=[1097653.9383702152], LFLs=[0.06], logPs=[-0.74], molecular_diameters=[3.7995699999999997],
                                        MWs=[32.04186], names=['methanol'], omegas=[0.5589999999999999], Parachors=[1.574169046057019e-05], Pcs=[8084000.0], phase_STPs=['l'], Psat_298s=[16905.960312551426], PSRK_groups=[{15: 1}],
                                        Pts=[0.1758862695025245], PubChems=[887], rhocs=[8547.008547008547], rhocs_mass=[273.86205128205125], rhol_STPs=[24494.78614922483], rhol_STPs_mass=[784.8585085234012], RIs=[1.3288], S0gs=[239.9],
                                        S0gs_mass=[7487.080962216301], Sfgs=[-129.7999999999999], Sfgs_mass=[-4050.950849919446], similarity_variables=[0.18725504699165404], Skins=[True], smiless=['CO'], STELs=[(250.0, 'ppm')],
                                        StielPolars=[0.027243902847492674], Stockmayers=[685.96], Tautoignitions=[713.15], Tbs=[337.65], Tcs=[512.5], Tflashs=[282.15], Tms=[175.15], Tts=[175.59], UFLs=[0.36],
                                        UNIFAC_Dortmund_groups=[{15: 1}], UNIFAC_groups=[{15: 1}], Van_der_Waals_areas=[358000.0], Van_der_Waals_volumes=[2.1709787e-05], Vcs=[0.000117], Vml_STPs=[4.0825014511573776e-05],
                                        Vml_Tms=[3.541058756059562e-05], Zcs=[0.22196480200068586], UNIFAC_Rs=[1.4311], UNIFAC_Qs=[1.432], rhos_Tms=[1013.0221439813405], Vms_Tms=[3.162996997683616e-05], solubility_parameters=[29315.58469262365],
                                        Vml_60Fs=[4.033573571273147e-05], rhol_60Fs_mass=[794.3789652976724], rhol_60Fs=[24791.91174599953])

    VaporPressures = [VaporPressure(exp_poly_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10,
                                                            -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708])), ]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12,
                                                                4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924])), ]
    HeatCapacityLiquids = [HeatCapacityLiquid(poly_fit=(180.0, 503.1, [-5.042130764341761e-17, 1.3174414379504284e-13, -1.472202211288266e-10, 9.19934288272021e-08,
                                                                    -3.517841445216993e-05, 0.008434516406617465, -1.2381765320848312, 101.71442569958393, -3508.6245143327947])),]
    VolumeLiquids = [VolumeLiquid(poly_fit=(175.7, 502.5, [3.5725079384600736e-23, -9.031033742820083e-20, 9.819637959370411e-17, -5.993173551565636e-14,
                                                        2.2442465416964825e-11, -5.27776114586072e-09, 7.610461006178106e-07, -6.148574498547711e-05, 0.00216398089328537])), ]
    EnthalpyVaporizations = [EnthalpyVaporization(Tc=512.5, poly_fit_ln_tau=(175.7, 512.499, 512.5, [-0.004536133852590396, -0.2817551666837462, -7.344529282245696, -104.02286881045083,
                                                                                    -860.5796142607192, -4067.8897875259267, -8952.300062896637, 2827.0089241465225, 44568.12528999141])),]
    HeatCapacitySolids = [HeatCapacitySolid(poly_fit=(1.0, 5000.0, [-4.2547351607351175e-26, 9.58204543572984e-22, -8.928062818728625e-18, 4.438942190507877e-14,
                                                                    -1.2656161406049876e-10, 2.0651464217978594e-07, -0.0001691371394823046, 0.2038633833421581, -0.07254973910767148])),]
    SublimationPressures = [SublimationPressure(exp_poly_fit=(61.5, 179.375, [-1.9972190661146383e-15, 2.1648606414769645e-12, -1.0255776193312338e-09, 2.7846062954442135e-07,
                                                                        -4.771529410705124e-05, 0.005347189071525987, -0.3916553642749777, 18.072103851054266, -447.1556383160345])),]
    EnthalpySublimations = [EnthalpySublimation(poly_fit=(1.7515, 175.59, [2.3382707698778188e-17, -2.03890965442551e-12, 1.5374109464154768e-09, -4.640933157748743e-07,
                                                                        6.931187040484687e-05, -0.004954625422589015, 0.045058888152305354, 32.52432385785916, 42213.605713250145])),]
    VolumeSolids = [VolumeSolid(poly_fit=(52.677, 175.59, [3.9379562779372194e-30, 1.4859309728437516e-27, 3.897856765862211e-24, 5.012758300685479e-21, 7.115820892078097e-18,
                                                        9.987967202910477e-15, 1.4030825662633013e-11, 1.970935889948393e-08, 2.7686131179275174e-05])),]

    SurfaceTensions = [SurfaceTension(CASRN='67-56-1', method="SOMAYAJULU2")]

    ThermalConductivityLiquids=[ThermalConductivityLiquid(poly_fit=(390.65, 558.9, [-1.7703926719478098e-31, 5.532831178371296e-28, -7.157706109850407e-25, 4.824017093238245e-22, -1.678132299010268e-19, 1.8560214447222824e-17, 6.274769714658382e-15, -0.00020340000228224661, 0.21360000021862866])),]
    ThermalConductivityGases=[ThermalConductivityGas(poly_fit=(390.65, 558.9, [1.303338742188738e-26, -5.948868042722525e-23, 1.2393384322893673e-19, -1.5901481819379786e-16, 1.4993659486913432e-13, -1.367840742416352e-10, 1.7997602278525846e-07, 3.5456258123020795e-06, -9.803647813554084e-05])),]
    ViscosityLiquids=[ViscosityLiquid(extrapolation_min=0, exp_poly_fit=(190.0, 391.9, [1.8379049563136273e-17, -4.5666126233131545e-14, 4.9414486397781785e-11, -3.042378423089263e-08, 1.166244931040138e-05, -0.0028523723735774113, 0.4352378275340892, -37.99358630363772, 1456.8338572042996])),]
    ViscosityGases=[ViscosityGas(extrapolation_min=0, poly_fit=(390.65, 558.9, [4.166385860107714e-29, -1.859399624586853e-25, 3.723945144634823e-22, -4.410000193606962e-19, 3.412270901850386e-16, -1.7666632565075753e-13, 5.266250837132718e-11, 1.8202807683935545e-08, -3.7907568022643496e-07])),]

    correlations = PropertyCorrelationsPackage(constants, VaporPressures=VaporPressures, HeatCapacityGases=HeatCapacityGases, HeatCapacityLiquids=HeatCapacityLiquids, VolumeLiquids=VolumeLiquids,
                                            EnthalpyVaporizations=EnthalpyVaporizations, HeatCapacitySolids=HeatCapacitySolids, SublimationPressures=SublimationPressures,
                                            SurfaceTensions=SurfaceTensions,
                                            ViscosityLiquids=ViscosityLiquids, ViscosityGases=ViscosityGases,
                                            ThermalConductivityGases=ThermalConductivityGases, ThermalConductivityLiquids=ThermalConductivityLiquids,
                                            EnthalpySublimations=EnthalpySublimations, VolumeSolids=VolumeSolids)
    correlations2 = eval(repr(correlations))
    for name in correlations.correlations:
        assert getattr(correlations, name) == getattr(correlations2, name)
        assert hash_any_primitive(getattr(correlations, name)) == hash_any_primitive(getattr(correlations2, name))
            
    assert correlations == correlations2


def test_functional_groups_json_serialization():
    """Test that functional groups serialize and deserialize correctly via JSON"""
    # Create a simple package with two components - one with functional groups, one without
    obj = ChemicalConstantsPackage(
        MWs=[18.01528, 32.04186],  # Water and methanol
        names=['water', 'methanol'],
        functional_groups=[
            None,  # Water has no groups identified
            {FG_ALCOHOL, FG_ORGANIC}  # Methanol has two groups
        ]
    )
    
    # Test JSON serialization and deserialization
    json_str = json.dumps(obj.as_json())
    obj2 = ChemicalConstantsPackage.from_json(json.loads(json_str))
    
    # Verify the objects are equal
    assert obj == obj2
    assert hash(obj) == hash(obj2)
    assert id(obj) != id(obj2)
    
    # Verify the functional groups specifically
    assert obj2.functional_groups[0] is None
    assert obj2.functional_groups[1] == {FG_ALCOHOL, FG_ORGANIC}

def test_functional_groups_addition():
    """Test that functional groups combine correctly when adding packages"""
    a = ChemicalConstantsPackage(
        MWs=[18.01528],
        names=['water'],
        functional_groups=[{FG_ORGANIC}]
    )
    b = ChemicalConstantsPackage(
        MWs=[32.04186],
        names=['methanol'], 
        functional_groups=[{FG_ALCOHOL, FG_ORGANIC}]
    )
    
    c = a + b
    
    assert c.functional_groups[0] == {FG_ORGANIC}
    assert c.functional_groups[1] == {FG_ALCOHOL, FG_ORGANIC}
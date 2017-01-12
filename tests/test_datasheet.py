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

import pytest
import pandas as pd
from thermo.datasheet import *
from thermo.chemical import Chemical
from thermo.identifiers import pubchem_dict

# These tests slow down implementation of new methods too much.
#@pytest.mark.meta_Chemical
#def test_tabulate_solid():
#    df = tabulate_solid('sodium hydroxide', pts=2)
#    df_as_dict = {'Constant-pressure heat capacity, J/kg/K': {496.14999999999998: 1267.9653086278533, 596.14999999999998: 1582.2714391628249}, 'Density, kg/m^3': {496.14999999999998: 2130.0058046853483, 596.14999999999998: 2130.0058046853483}}
#    pd.util.testing.assert_frame_equal(pd.DataFrame(df_as_dict), pd.DataFrame(df.to_dict()))
#
#
#@pytest.mark.meta_Chemical
#def test_tabulate_gas():
#    df = tabulate_gas('hexane', pts=2)
#    df_as_dict = {'Constant-pressure heat capacity, J/kg/K': {178.07499999999999: 1206.4098393032568, 507.60000000000002: 2551.4044899160472}, 'Viscosity, Pa*S': {178.07499999999999: 3.6993265691382959e-06, 507.60000000000002: 1.0598974706090609e-05}, 'Isentropic exponent': {178.07499999999999: 1.0869273799268073, 507.60000000000002: 1.0393018803424154}, 'Joule-Thompson expansion coefficient, K/Pa': {178.07499999999999: 0.00016800664986363302, 507.60000000000002: 7.8217064543503734e-06}, 'Isobaric expansion, 1/K': {178.07499999999999: 0.015141550023997695, 507.60000000000002: 0.0020523335027846585}, 'Prandtl number': {178.07499999999999: 0.69678226644585661, 507.60000000000002: 0.74170212695888871}, 'Density, kg/m^3': {178.07499999999999: 8.3693048957953522, 507.60000000000002: 2.0927931856300876}, 'Constant-volume heat capacity, J/kg/K': {178.07499999999999: 1109.9268098154776, 507.60000000000002: 2454.9214604282679}, 'Thermal diffusivity, m^2/s': {178.07499999999999: 6.3436058798806709e-07, 507.60000000000002: 6.8282280730497638e-06}, 'Thermal consuctivity, W/m/K': {178.07499999999999: 0.0064050194540236464, 507.60000000000002: 0.036459746670141478}}
#    pd.util.testing.assert_frame_equal(pd.DataFrame(df_as_dict), pd.DataFrame(df.to_dict()))
#
#
#@pytest.mark.meta_Chemical
#def test_tabulate_liq():
#    df = tabulate_liq('hexane', Tmin=280, Tmax=350, pts=2)
#    df_as_dict = {'Constant-pressure heat capacity, J/kg/K': {280.0: 2199.5376248501448, 350.0: 2509.3959378687496}, 'Viscosity, Pa*S': {280.0: 0.0003595695325135477, 350.0: 0.00018618849649397316}, 'Saturation pressure, Pa': {280.0: 8624.370564055087, 350.0: 129801.09838575375}, 'Joule-Thompson expansion coefficient, K/Pa': {280.0: 3.4834926941752087e-05, 350.0: 3.066272687922139e-05}, 'Surface tension, N/m': {280.0: 0.019794991465879444, 350.0: 0.01261221127458579}, 'Prandtl number': {280.0: 6.2861632870484234, 350.0: 4.5167171403747597}, 'Isobaric expansion, 1/K': {280.0: 0.001340989794772991, 350.0: 0.0016990766161286714}, 'Density, kg/m^3': {280.0: 671.28561912698535, 350.0: 606.36768482956563}, 'Thermal diffusivity, m^2/s': {280.0: 8.5209866345631262e-08, 350.0: 6.7981994628212491e-08}, 'Heat of vaporization, J/kg': {280.0: 377182.42886698805, 350.0: 328705.97080247721}, 'Permittivity': {280.0: 1.8865000000000001, 350.0: 1.802808}, 'Thermal consuctivity, W/m/K': {280.0: 0.12581389941664639, 350.0: 0.10344253187860687}}
#    pd.util.testing.assert_frame_equal(pd.DataFrame(df_as_dict), pd.DataFrame(df.to_dict()))
#
#
#@pytest.mark.meta_Chemical
#def test_constants():
#    # TODO: Hsub again so that works
#    df = tabulate_constants('hexane')
#    df_as_dict = {'Heat of vaporization at Tb, J/mol': {'hexane': 28862.311605415733}, 'Time-weighted average exposure limit': {'hexane': "(50.0, 'ppm')"}, 'Tc, K': {'hexane': 507.60000000000002}, 'Short-term exposure limit': {'hexane': 'None'}, 'Molecular Diameter, Angstrom': {'hexane': 5.6184099999999999}, 'Zc': {'hexane': 0.26376523052422041}, 'Tm, K': {'hexane': 178.07499999999999}, 'Heat of fusion, J/mol': {'hexane': 13080.0}, 'Tb, K': {'hexane': 341.87}, 'Stockmayer parameter, K': {'hexane': 434.75999999999999}, 'MW, g/mol': {'hexane': 86.175359999999998}, 'Refractive index': {'hexane': 1.3727}, 'rhoC, kg/m^3': {'hexane': 234.17217391304345}, 'Heat of formation, J/mol': {'hexane':  -166950.0}, 'Pc, Pa': {'hexane': 3025000.0}, 'Lower flammability limit, fraction': {'hexane': 0.01}, 'logP': {'hexane': 4.0}, 'Upper flammability limit, fraction': {'hexane': 0.08900000000000001}, 'Dipole moment, debye': {'hexane': 0.0}, 'Triple temperature, K': {'hexane': 177.84}, 'Acentric factor': {'hexane': 0.29749999999999999}, 'Triple pressure, Pa': {'hexane': 1.1747772750450831}, 'Autoignition temperature, K': {'hexane': 498.14999999999998}, 'Vc, m^3/mol': {'hexane': 0.000368}, 'CAS': {'hexane': '110-54-3'}, 'Formula': {'hexane': 'C6H14'}, 'Flash temperature, K': {'hexane': 251.15000000000001}, 'Heat of sublimation, J/mol': {'hexane': None}}
#
#    pd.util.testing.assert_frame_equal(pd.DataFrame(df_as_dict), pd.DataFrame(df.to_dict()))
#
#    df = tabulate_constants(['hexane', 'toluene'], full=True, vertical=True)
#    df_as_dict = {'hexane': {'Electrical conductivity, S/m': 1e-16, 'Global warming potential': None, 'InChI key': 'VLKZOEOYAKHREP-UHFFFAOYSA-N', 'Heat of vaporization at Tb, J/mol': 28862.311605415733, 'Time-weighted average exposure limit': "(50.0, 'ppm')", 'Tc, K': 507.6, 'Short-term exposure limit': 'None', 'Molecular Diameter, Angstrom': 5.61841, 'Formula': 'C6H14', 'InChI': 'C6H14/c1-3-5-6-4-2/h3-6H2,1-2H3', 'Parachor': 272.1972168105559, 'Heat of fusion, J/mol': 13080.0, 'Tb, K': 341.87, 'Stockmayer parameter, K': 434.76, 'IUPAC name': 'hexane', 'Refractive index': 1.3727, 'Tm, K': 178.075, 'solubility parameter, Pa^0.5': 14848.17694628013, 'Heat of formation, J/mol': -166950.0, 'Pc, Pa': 3025000.0, 'Lower flammability limit, fraction': 0.01, 'Vc, m^3/mol': 0.000368, 'Upper flammability limit, fraction': 0.08900000000000001, 'Dipole moment, debye': 0.0, 'MW, g/mol': 86.17536, 'Acentric factor': 0.2975, 'rhoC, kg/m^3': 234.17217391304345, 'Zc': 0.2637652305242204, 'Triple pressure, Pa': 1.1747772750450831, 'Autoignition temperature, K': 498.15, 'CAS': '110-54-3', 'smiles': 'CCCCCC', 'Flash temperature, K': 251.15, 'Ozone depletion potential': None, 'logP': 4.0, 'Heat of sublimation, J/mol': None, 'Triple temperature, K': 177.84}, 'toluene': {'Electrical conductivity, S/m': 1e-12, 'Global warming potential': None, 'InChI key': 'YXFVVABEGXRONW-UHFFFAOYSA-N', 'Heat of vaporization at Tb, J/mol': 33233.94544167449, 'Time-weighted average exposure limit': "(20.0, 'ppm')", 'Tc, K': 591.75, 'Short-term exposure limit': 'None', 'Molecular Diameter, Angstrom': 5.4545, 'Formula': 'C7H8', 'InChI': 'C7H8/c1-7-5-3-2-4-6-7/h2-6H,1H3', 'Parachor': 246.76008384965857, 'Heat of fusion, J/mol': 6639.9999999999991, 'Tb, K': 383.75, 'Stockmayer parameter, K': 350.74, 'IUPAC name': 'methylbenzene', 'Refractive index': 1.4941, 'Tm, K': 179.2, 'solubility parameter, Pa^0.5': 18242.232319337778, 'Heat of formation, J/mol': 50170.0, 'Pc, Pa': 4108000.0, 'Lower flammability limit, fraction': 0.01, 'Vc, m^3/mol': 0.00031600000000000004, 'Upper flammability limit, fraction': 0.078, 'Dipole moment, debye': 0.33, 'MW, g/mol': 92.13842, 'Acentric factor': 0.257, 'rhoC, kg/m^3': 291.5772784810126, 'Zc': 0.26384277925843774, 'Triple pressure, Pa': 0.04217711401906639, 'Autoignition temperature, K': 803.15, 'CAS': '108-88-3', 'smiles': 'CC1=CC=CC=C1', 'Flash temperature, K': 277.15, 'Ozone depletion potential': None, 'logP': 2.73, 'Heat of sublimation, J/mol': None, 'Triple temperature, K': 179.2}}
#    pd.util.testing.assert_frame_equal(pd.DataFrame(df_as_dict), pd.DataFrame(df.to_dict()))

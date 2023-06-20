"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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
"""

import json
import os
import xml.etree.ElementTree as ET

folder = os.path.dirname(__file__)
tree = ET.parse(os.path.join(folder, 'ChemSep8.32.xml'))
root = tree.getroot()
parameter_tags = ['ApiSrkS1',
 'ApiSrkS2',
 'RacketParameter',
  'SolubilityParameter',
   'COSTALDVolume',
    'CostaldAcentricFactor',
     'ChaoSeaderAcentricFactor',
  'ChaoSeaderLiquidVolume',
  'ChaoSeaderSolubilityParameter',
  'MatthiasCopemanC1',
 'MatthiasCopemanC2',
 'MatthiasCopemanC3',
 'WilsonVolume',
 ]
parameter_tags_set = set(parameter_tags)

parameter_dicts = {k: {} for k in parameter_tags}
all_tags = set()
data = {}
for child in root:
    CAS = None
    # Identify the CAS number first
    for i in child:
        tag = i.tag
#         all_tags.add(tag)
        if CAS is None and tag == 'CAS':
            CAS = i.attrib['value']
    for i in child:
        tag = i.tag
        if tag in parameter_tags_set:
            parameter_dicts[tag][CAS] = float(i.attrib['value'] )

            #        CAS = [i.attrib['value'] if  ][0]
#        name = [i.attrib['value'] for i in child if i.tag ][0]
#        smiles = [i.attrib['value'] for i in child if i.tag == ]
#        formula = [i.attrib['value'] for i in child if i.tag == 'StructureFormula'][0]


folder = os.path.join(os.path.dirname(__file__), '..', 'thermo', 'Scalar Parameters')

article_source = "ChemSep 8.26"
"""
"""
PSRK_metadata = {
  "metadata": {
    "source": article_source,
    "necessary keys": [
      "MCSRKC1", "MCSRKC2", "MCSRKC3",
    ],
    "missing": {
      "MCSRKC1": None, "MCSRKC2": None, "MCSRKC3": None,
    }
  }
}

data = {}
for CAS in parameter_dicts['MatthiasCopemanC1'].keys():
    data[CAS] = {"name": CAS, "MCSRKC1": parameter_dicts['MatthiasCopemanC1'][CAS], "MCSRKC2": parameter_dicts['MatthiasCopemanC2'][CAS], "MCSRKC3": parameter_dicts['MatthiasCopemanC3'][CAS]}
PSRK_metadata['data'] = data

f = open(os.path.join(folder, 'chemsep_PSRK_matthias_copeman.json'), 'w')
f.write(json.dumps(PSRK_metadata, sort_keys=True, indent=2))
f.close()


APISRK_metadata = {
  "metadata": {
    "source": article_source,
    "necessary keys": [
      "APISRKS1", "APISRKS2"
    ],
    "missing": {
      "APISRKS1": None, "APISRKS2": None,
    }
  }
}

data = {}
for CAS in parameter_dicts['ApiSrkS1'].keys():
    data[CAS] = {"name": CAS, "APISRKS1": parameter_dicts['ApiSrkS1'][CAS], "APISRKS2": parameter_dicts['ApiSrkS2'][CAS]}
APISRK_metadata['data'] = data

f = open(os.path.join(folder, 'chemsep_APISRK.json'), 'w')
f.write(json.dumps(APISRK_metadata, sort_keys=True, indent=2))
f.close()



RegularSolution_metadata = {
  "metadata": {
    "source": article_source,
    "necessary keys": [
      "RegularSolutionV", "RegularSolutionSP"
    ],
    "missing": {
      "RegularSolutionV": None, "RegularSolutionSP": None,
    }
  }
}


data = {}
for CAS in parameter_dicts['WilsonVolume'].keys():
    if CAS in parameter_dicts['SolubilityParameter']:
        data[CAS] = {"name": CAS, "RegularSolutionV": parameter_dicts['WilsonVolume'][CAS]*0.001, "RegularSolutionSP": parameter_dicts['SolubilityParameter'][CAS]}
RegularSolution_metadata['data'] = data

f = open(os.path.join(folder, 'chemsep_regular_solution.json'), 'w')
f.write(json.dumps(RegularSolution_metadata, sort_keys=True, indent=2))
f.close()

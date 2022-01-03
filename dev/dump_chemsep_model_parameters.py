# -*- coding: utf-8 -*-
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

import os
import xml.etree.cElementTree as ET
folder = os.path.dirname(__file__),
tree = ET.parse(os.path.join(folder, 'ChemSep8.26.xml'))
root = tree.getroot()
parameter_tags = ['ApiSrkS1',
 'ApiSrkS2',
 'RacketParameter',
  'SolubilityParameter',
#   'COSTALDVolume',
#    'CostaldAcentricFactor',
#     'ChaoSeaderAcentricFactor',
#  'ChaoSeaderLiquidVolume',
#  'ChaoSeaderSolubilityParameter',
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



article_source = "ChemSep 8.26"
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

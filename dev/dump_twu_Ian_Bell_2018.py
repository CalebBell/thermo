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

import pandas as pd
from chemicals import *

file_name = 'je7b00967_si_001.xlsx'
df2 = pd.read_excel(file_name)

folder = os.path.join(os.path.dirname(__file__), '..', 'thermo', 'Scalar Parameters')

webbook_metadata = pd.read_csv('webbook_metadata.tsv', index_col=0, sep='\t')
webbook_inchikey_to_CAS = {k: v for k, v in zip(webbook_metadata['InChI_key'], webbook_metadata.index)}
webbook_inchi_to_CAS = {k: v for k, v in zip(webbook_metadata['InChI'], webbook_metadata.index)}
#print(webbook_inchi_to_CAS)

article_source = "Consistent Twu Parameters for More than 2500 Pure Fluids from Critically Evaluated Experimental Data"
PR_Twu_metadata = {
  "metadata": {
    "source": article_source,
    "necessary keys": [
      "TwuPRL", "TwuPRM", "TwuPRN", "TwuPRc",
    ],
    "missing": {
      "TwuPRL": None, "TwuPRM": None, "TwuPRN": None, "TwuPRc": 0.0,
    }
  }
}


PRLs = df2['c0'].values.tolist()
PRMs = df2['c1'].values.tolist()
PRNs = df2['c2'].values.tolist()
inchis = df2['inchi'].values.tolist()
inchikeys = df2['inchikey'].values.tolist()
names = df2['name'].values.tolist()
formulas = df2['formula'].values.tolist()

CASs = []
for i in range(len(inchikeys)):
    CAS = None
    try:
        try:
            # Prefer close NIST data
            try:
                CAS = webbook_inchikey_to_CAS[inchikeys[i]]
            except:
                # Doesn't seem to find any others
                CAS = webbook_inchi_to_CAS[inchis[i]]
        except:
            # Try to locate it in Chemicals
            CAS = CAS_from_any('InChIKey=%s'%(inchikeys[i]))
    except:
        pass
        #print(names[i], formulas[i], inchikeys[i], inchis[i])
    CASs.append(CAS)
#print(CASs)


data = {}
for CAS, L, M, N in zip(CASs, PRLs, PRMs, PRNs):
    if CAS is not None:
        data[CAS] = {"name": CAS, "TwuPRL": L, "TwuPRM": M, "TwuPRN": N}
PR_Twu_metadata['data'] = data

f = open(os.path.join(folder, 'PRTwu_ibell_2018.json'), 'w')
f.write(json.dumps(PR_Twu_metadata, sort_keys=True, indent=2))
f.close()

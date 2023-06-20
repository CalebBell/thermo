"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2021, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

file_name = 'Use of 300,000 pseudo-experimental data over 1800 pure fluids to assess the performance of four cubic equations of state parameters.xlsx'
df2 = pd.read_excel(file_name, sheet_name='export', index_col='CAS')

folder = os.path.join(os.path.dirname(__file__), '..', 'thermo', 'Scalar Parameters')


article_source = "Use of 300,000 pseudo-experimental data over 1800 pure fluids to assess the performance of four cubic equations of state: SRK, PR, tc-RK, and tc-PR"
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

PR_C_metadata = {
  "metadata": {
    "source": article_source,
    "necessary keys": [
      "PRc",
    ],
    "missing": {
      "PRc": 0.0,
    }
  }
}

SRK_Twu_metadata = {
  "metadata": {
    "source": article_source,
    "necessary keys": [
      "TwuSRKL", "TwuSRKM", "TwuSRKN", "TwuSRKc",
    ],
    "missing": {
      "TwuSRKL": None, "TwuSRKM": None, "TwuSRKN": None, "TwuSRKc": 0.0,
    }
  }
}

SRK_C_metadata = {
  "metadata": {
    "source": article_source,
    "necessary keys": [
      "SRKc",
    ],
    "missing": {
      "SRKc": 0.0,
    }
  }
}

CASs = df2.index.tolist()
PRLs = df2['PRL'].values.tolist()
PRMs = df2['PRM'].values.tolist()
PRNs = df2['PRN'].values.tolist()
PRCs = df2['PRC'].values.tolist()
SRKLs = df2['SRKL'].values.tolist()
SRKMs = df2['SRKM'].values.tolist()
SRKNs = df2['SRKN'].values.tolist()
SRKCs = df2['SRKC'].values.tolist()


data = {}
for CAS, L, M, N, c in zip(CASs, PRLs, PRMs, PRNs, PRCs):
    data[CAS] = {"name": CAS, "TwuPRL": L, "TwuPRM": M, "TwuPRN": N, "TwuPRc": c}
PR_Twu_metadata['data'] = data

data = {}
for CAS, c in zip(CASs, PRCs):
    data[CAS] = {"name": CAS, "PRc": c}
PR_C_metadata['data'] = data

f = open(os.path.join(folder, 'PRTwu_PinaMartinez.json'), 'w')
f.write(json.dumps(PR_Twu_metadata, sort_keys=True, indent=2))
f.close()

f = open(os.path.join(folder, 'PRVolumeTranslation_PinaMartinez.json'), 'w')
f.write(json.dumps(PR_C_metadata, sort_keys=True, indent=2))
f.close()


data = {}
for CAS, L, M, N, c in zip(CASs, SRKLs, SRKMs, SRKNs, SRKCs):
    data[CAS] = {"name": CAS, "TwuSRKL": L, "TwuSRKM": M, "TwuSRKN": N, "TwuSRKc": c}
SRK_Twu_metadata['data'] = data

data = {}
for CAS, c in zip(CASs, SRKCs):
    data[CAS] = {"name": CAS, "SRKc": c}
SRK_C_metadata['data'] = data

f = open(os.path.join(folder, 'SRKTwu_PinaMartinez.json'), 'w')
f.write(json.dumps(SRK_Twu_metadata, sort_keys=True, indent=2))
f.close()

f = open(os.path.join(folder, 'SRKVolumeTranslation_PinaMartinez.json'), 'w')
f.write(json.dumps(SRK_C_metadata, sort_keys=True, indent=2))
f.close()



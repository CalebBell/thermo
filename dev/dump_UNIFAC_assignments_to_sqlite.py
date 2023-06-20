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

import numpy as np
import pandas as pd
from chemicals import *
from chemicals.identifiers import pubchem_db

import thermo
from thermo import *
from thermo import Chemical
from thermo.group_contribution.group_contribution_base import group_assignment_to_str

try:
    Chemical('asdfasdaf')
except:
    pass
thermo.unifac.load_group_assignments_DDBST()
thermo.unifac.DDBST_UNIFAC_assignments

CASs = []
UNIFACs = []
MODFACs = []
PSRKs = []
all_InChIs = list(set(list(thermo.unifac.DDBST_UNIFAC_assignments.keys()) +
               list(thermo.unifac.DDBST_MODIFIED_UNIFAC_assignments.keys()) +
               list(thermo.unifac.DDBST_PSRK_assignments.keys())))

dev_folder = os.path.dirname(__file__)
to_location = os.path.join(dev_folder, '..', 'thermo', 'Phase Change', 'DDBST_UNIFAC_assignments.sqlite')

for inchi in all_InChIs:
    try:
        CASi = pubchem_db.InChI_key_index[inchi].CAS
    except:
        continue
    CASs.append(CASi)

    groups = thermo.unifac.DDBST_UNIFAC_assignments.get(inchi, {})
    UNIFACs.append(group_assignment_to_str(groups))


    groups = thermo.unifac.DDBST_MODIFIED_UNIFAC_assignments.get(inchi, {})
    MODFACs.append(group_assignment_to_str(groups))

    groups = thermo.unifac.DDBST_PSRK_assignments.get(inchi, {})
    PSRKs.append(group_assignment_to_str(groups))



prop_array_T = np.array([UNIFACs, MODFACs,PSRKs]).T

# Would not be good if there were multiple values
assert len(CASs) == len(set(CASs))


df = pd.DataFrame(prop_array_T, columns=['UNIFAC', 'MODIFIED_UNIFAC', 'PSRK'], index=CASs)

# Does not save memory except when compressed and causes many test failures
#df = df.fillna(value=np.nan).astype(np.float32)
df.sort_index(inplace=True)

from sqlalchemy import create_engine

engine = create_engine('sqlite:///' + to_location, echo=False)
if os.path.exists(to_location):
    os.remove(to_location)
df.to_sql('DDBST', con=engine)

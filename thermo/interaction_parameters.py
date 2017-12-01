  # -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from __future__ import division

__all__ = ['InteractionParameterDB', 'IPDB']

import os
import json
import numpy as np
from thermo.utils import sorted_CAS_key
from thermo.identifiers import checkCAS

'''Need to be able to add NRTL parameters sets (T dept)
Need to add single kijis, and kijs with T dept
'''

folder = os.path.join(os.path.dirname(__file__), 'Phase Change')
chemsep_db_path = os.path.join(folder, 'ChemSep')


        
class InteractionParameterDB(object):
    
    def __init__(self):
        self.tables = {}
        self.metadata = {}
    
    def load_json(self, file, name):
        f = open(file).read()
        dat = json.loads(f)
        self.tables[name] = dat['data']
        self.metadata[name] = dat['metadata']
        
    def validate_table(self, name):
        table = self.tables[name]
        meta = self.metadata[name]
        components = meta['components']
        necessary_keys = meta['necessary keys']
        # Check the CASs
        for key in table:
            CASs = key.split(' ')
            # Check the key is the right length
            assert len(CASs) == components
            # Check all CAS number keys are valid
            assert all(checkCAS(i) for i in CASs)
            
            values = table[key]
            for i in necessary_keys:
                # Assert all necessary keys are present
                assert i in values
                val = values[i]
                # Check they are not None
                assert val is not None
                # Check they are not nan
                assert not np.isnan(val)
                
                
    def has_ip_specific(self, name, CASs, ip):
        if self.metadata[name]['symmetric']:
            key = ' '.join(sorted_CAS_key(CASs))
        else:
            key = ' '.join(CASs)
        table = self.tables[name]
        if key not in table:
            return False
        return ip in table[key]

    def get_ip_specific(self, name, CASs, ip):
        if self.metadata[name]['symmetric']:
            key = ' '.join(sorted_CAS_key(CASs))
        else:
            key = ' '.join(CASs)
        try:
            return self.tables[name][key][ip]
        except KeyError:
            return self.metadata[name]['missing'][ip]
    
    def get_tables_with_type(self, ip_type):
        tables = []
        for key, d in self.metadata.items():
            if d['type'] == ip_type:
                tables.append(key)
        return tables
    
    def get_ip_automatic(self, CASs, ip_type, ip):
        table = self.get_tables_with_type(ip_type)[0]
        return self.get_ip_specific(table, CASs, ip)
    
    def get_ip_symmetric_matrix(self, name, CASs, ip):
        table = self.tables[name]
        N = len(CASs)
        values = [[None for i in range(N)] for j in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    i_ip = 0.0
                elif values[j][i] is not None:
                    continue # already set
                else:
                    i_ip = self.get_ip_specific(name, [CASs[i], CASs[j]], ip)
                values[i][j] = values[j][i] = i_ip
        return np.array(values)
        
    
ip_files = {'ChemSep PR': os.path.join(chemsep_db_path, 'pr.json'),
            'ChemSep NRTL': os.path.join(chemsep_db_path, 'nrtl.json')}

IPDB = InteractionParameterDB()
for name, file in ip_files.items():
    IPDB.load_json(file, name)



    
IPDB.validate_table('ChemSep PR')
IPDB.validate_table('ChemSep NRTL')

IPDB.get_ip_specific('ChemSep PR', ['124-38-9', '67-56-1'], 'kij')
IPDB.get_ip_specific('ChemSep PR', ['1249-38-9', '67-56-1'], 'kij')
IPDB.has_ip_specific('ChemSep PR', ['1249-38-9', '67-56-1'], 'kij')
IPDB.has_ip_specific('ChemSep PR', ['124-38-9', '67-56-1'], 'kij')
IPDB.get_tables_with_type('PR kij')

IPDB.get_ip_automatic(['124-38-9', '67-56-1'], 'PR kij', 'kij')
# C1 - C4 IPs
ans = IPDB.get_ip_symmetric_matrix('ChemSep PR', ['74-82-8', '74-84-0', '74-98-6', '106-97-8'], 'kij')
print(ans)

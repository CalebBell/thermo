# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020, 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
Copyright (C) 2021 Yoel Cortes-Pena <yoelcortes@gmail.com>

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
from thermo import ChemicalConstantsPackage
from chemicals import CAS_from_any

class FlashDefaultSettings:
    
    __slots__ = (
        'IDs',
        'L',
        'G',
        'Gkw',
        'Lkw',
        'constants',
        'properties',
        'index',
    )
    
    def __init__(self, IDs, L, G, Gkw, Lkw):
        self.IDs = IDs
        self.L = L
        self.G = G
        self.Gkw = Gkw
        self.Lkw = Lkw
        self.constants, self.properties = ChemicalConstantsPackage.from_IDs(IDs)
        self.index = index = {j: i for i, j in enumerate(IDs)}
        for i, j in enumerate(self.constants.CASs): index[j] = i

    def get_index(self, ID):
        if ID in self.index:
            return self.index[ID]
        try:
            CAS = CAS_from_any(ID)
            self.index[ID] = index = self.index[CAS]
            return index
        except:
            raise LookupError("ID '%s' not in default settings" %ID)
        
    def get_indices(self, IDs):
        return [self.get_index(i) for i in IDs]
        
    def subset(self, IDs):
        indices = self.get_indices(IDs)
        cls = self.__class__
        new = cls.__new__(cls)
        new.IDs = [self.IDs[i] for i in indices]
        new.L = self.L
        new.G = self.G
        new.Gkw = self.Gkw
        new.Lkw = self.Lkw
        new.constants = self.constants.subset(indices)
        new.properties = self.properties.subset(indices)
        new.index = {}
        for index, ID in enumerate(IDs):
            original_index = self.IDs.index(ID)
            for i, j in self.index.items():
                if j == original_index:
                    new.index[i] = index
        return new
    
    def get_aliases(self, ID):
        k = self.index[ID]
        return [i for i, j in self.index.items() if j==k] 
        
    def set_alias(self, ID, alias):
        self.index[alias] = self.index[ID]
        
        
    
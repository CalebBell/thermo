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
from chemicals import CAS_from_any
from ..chemical_package import ChemicalConstantsPackage
from .flash_pure_vls import FlashPureVLS
from .flash_vln import FlashVLN
from .flash_vl import FlashVL
from ..bulk import default_settings

__all__ = (
    'ConstantsAndCorrelations',
    'FlashConstructor',
)

class ConstantsAndCorrelations(tuple):
    """
    Create a ConstantsAndCorrelations object which is a tuple of
    a ChemicalConstantsPackage and a PropertyCorrelationsPackage.
    
    Parameters
    ----------
    cc : tuple[ChemicalConstantsPackage, PropertyCorrelationsPackage]
    index : dict[str, float], optional
        Dictionary to get the index of a chemical by ID.
    
    Examples
    --------
    >>> cc = constants, correlations = tmo.ConstantsAndCorrelations.from_IDs(['Water', 'Ethanol'])
    >>> constants.CASs
    ['7732-18-5', '64-17-5']
    >>> cc.get_index('h2o') # Find index with any identifier
    0
    >>> cc.get_index('etoh')
    1
    >>> cc_e = cc.subset(['ethanol']) # Subset with any identifiers
    >>> cc_e.get_index('etoh')
    0
    >>> cc_w = cc.subset([0]) # Lists of integers also work
    >>> cc_w.get_index('h2o') 
    0
    >>> cc_copy = cc.subset(slice(None)) # Slices work too
    >>> cc_copy.get_index('ethanol') 
    1
    >>> cc.get_index('propanol') # Undefined component raises a value error
    Traceback (most recent call last):
    LookupError: no component with ID 'propanol'
    >>> cc.get_index(1.5) # Invalid type
    Traceback (most recent call last):
    ValueError: ID must be either a string or an integer; not a 'float' object
    >>> cc.set_alias('Ethanol', 'CH3CH2OH') # Allows you to index with other names
    >>> cc.get_index('CH3CH2OH')
    1
    >>> cc.get_aliases('Ethanol') # Get all aliases used
    ['Ethanol', 'etoh', 'ethanol', '64-17-5']
    
    """
    @classmethod
    def from_IDs(cls, IDs):
        cc = ChemicalConstantsPackage.from_IDs(IDs)
        index = {j: i for i, j in enumerate(IDs)}
        return cls(cc, index)
    
    def __new__(cls, cc, index=None):
        self = super().__new__(cls, cc)
        self._index = index or {}
        return self
    
    @property
    def constants(self): return self[0]
    
    @property
    def correlations(self): return self[1]
    
    def get_index(self, ID):
        if isinstance(ID, str):
            index = self._index
            if ID in index:
                return index[ID]
            try:
                CAS = CAS_from_any(ID)
                index[ID] = value = self.constants.CASs.index(CAS)
                return value
            except:
                raise LookupError("no component with ID '%s'" %ID)
        else:
            try:
                value = int(ID)
                assert value == ID
                return value
            except:
                raise ValueError("ID must be either a string or an integer; "
                                 "not a '%s' object" %type(ID).__name__)
        
    def get_indices(self, IDs):
        return [self.get_index(i) for i in IDs]
     
    def get_aliases(self, ID):
        k = self.get_index(ID)
        return [i for i, j in self._index.items() if j==k] 
        
    def set_alias(self, ID, alias):
        self._index[alias] = self._index[ID]

    def subset(self, IDs, aliases=True):
        if isinstance(IDs, slice):
            IDs = self.constants.CASs[IDs]
        index = {}
        if aliases:
            for i, ID in enumerate(IDs):
                index[ID] = i
                for j in self.get_aliases(self.get_index(ID)): index[j] = i
        cls = self.__class__
        indices = self.get_indices(IDs)
        return cls.__new__(cls, [i.subset(indices) for i in self], index)


class FlashConstructor:
    """
    Create a FlashConstructor object that predefines flash algorithms
    for easier creation of Flash and Phase objects.
    
    Parameters
    ----------
    cc : tuple[ChemicalConstantsPackage, PropertyCorrelationsPackage]
    G : Phase subclass
        Class create gas phase object.
    L : Phase subclass
        Class create liquid phase object.
    S : Phase subclass
        Class create solid phase object.
    GE : GibbsExcessModel subclass
        Class create GibbsExcessModel object.
    Gkw : dict
        Key word arguments to initialize `G`.
    Lkw : Phase subclass
        Key word arguments to initialize `L`.
    Skw : Phase subclass
        Key word arguments to initialize `S`.
    GEkw : GibbsExcessModel subclass
        Key word arguments to initialize `GE`.
    settings : :obj:`BulkSettings <thermo.bulk.BulkSettings>`, optional
        Object containing settings for calculating bulk and transport
        properties, [-]
    
    Examples
    --------
    >>> import thermo as tmo
    >>> flashpkg = tmo.FlashConstructor.from_IDs(
    ...     IDs=['Water', 'Ethanol', 'Hexanol'], 
    ...     G=tmo.CEOSGas, L=tmo.GibbsExcessLiquid, S=tmo.GibbsExcessSolid,
    ...     GE=tmo.UNIFAC, GEkw=dict(version=1), Gkw=dict(eos_class=tmo.PRMIX),
    ... )
    >>> flasher = flashpkg.flasher(N_liquid=2)
    >>> type(flasher).__name__
    'FlashVLN'
    >>> PT = flasher.flash(zs=[0.3, 0.2, 0.5], T=330, P=101325)
    >>> (PT.VF, PT.betas, PT.liquid0.zs, PT.H())
    (0.0,
     [0.027302, 0.972697],
     [0.943926, 0.051944, 0.004129],
     -46646.90)
    >>> flasher = flashpkg.flasher(['Water', 'Ethanol'])
    >>> type(flasher).__name__
    'FlashVL'
    >>> PT = flasher.flash(zs=[0.5, 0.5], T=353, P=101325)
    >>> (PT.VF, PT.gas.zs, PT.H())
    (0.312, [0.363, 0.636], -25473.987)
    >>> flasher = flashpkg.flasher(['Ethanol'])
    >>> type(flasher).__name__
    'FlashPureVLS'
    >>> PT = flasher.flash(T=353, P=101325)
    >>> (PT.VF, PT.gas.zs, PT.H())
    (1.0, [1.0], 3619.78)
    
    """
    __slots__ = (
        'cc', 'settings',
        'G', 'Gkw',
        'L', 'Lkw', 
        'S', 'Skw',
        'GE', 'GEkw',
    )
    
    @classmethod
    def from_IDs(cls, IDs, G, L, S, GE=None,
                 Gkw=None, Lkw=None, Skw=None, GEkw=None):
        return cls(
            ConstantsAndCorrelations.from_IDs(IDs),
            G, L, S, GE, Gkw, Lkw, Skw, GEkw
        )
    
    def __init__(self, cc, G, L, S=None, GE=None,
                 Gkw=None, Lkw=None, Skw=None, GEkw=None,
                 settings=None):
        self.cc = ConstantsAndCorrelations(cc) if not isinstance(cc, ConstantsAndCorrelations) else cc
        self.G = G
        self.L = L
        self.S = S
        self.GE = GE
        self.Gkw = Gkw or {}
        self.Lkw = Lkw or {}
        self.Skw = Skw or {}
        self.GEkw = GEkw or {}
        self.settings = settings or default_settings
        
    def solid(self, IDs):
        return self._solid_from_cc(self.cc.subset(IDs, False) if IDs else self.cc)
        
    def liquid(self, IDs):
        return self._liquid_from_cc(self.cc.subset(IDs, False) if IDs else self.cc)

    def gas(self, IDs):
        return self._gas_from_cc(self.cc.subset(IDs, False) if IDs else self.cc)
    
    def flasher(self, IDs=None, N_liquid=None, N_solid=None):
        return self._flash_from_cc(self.cc.subset(IDs, False) if IDs else self.cc,
                                  N_liquid, N_solid)

    def _solid_from_cc(self, cc):
        raise NotImplementedError("this method is not implemented yet")
        
    def _liquid_from_cc(self, cc):
        if self.GE:
            GE = self.GE.from_cc(cc, **self.GEkw)
            return self.L.from_cc(cc, GibbsExcessModel=GE, **self.Lkw)
        else:
            return self.L.from_cc(cc, **self.Lkw)

    def _gas_from_cc(self, cc):
        return self.G.from_cc(cc, **self.Gkw)

    def _flash_from_cc(self, cc, N_liquid, N_solid):
        constants, correlations = cc
        N = len(constants.CASs)
        if N_solid is None: N_solid = 0
        if N_liquid is None: N_liquid = 1
        if N == 0:
            raise ValueError(
                "IDs cannot be empty; at least one component ID must be given"
            )
        elif N == 1: # Pure component
            return FlashPureVLS(
                constants, 
                correlations,
                self._gas_from_cc(cc),
                [self._liquid_from_cc(cc)
                  for i in range(N_liquid)],
                [self._solid_from_cc(cc)
                  for i in range(N_solid)],
                self.settings,
            )
        elif N_liquid == 1:
            if N_solid:
                raise NotImplementedError(
                    'multi-component flasher with solid phases '
                    'not implemented (yet)'
                )
            return FlashVL(
                constants, 
                correlations,
                self._gas_from_cc(cc),
                self._liquid_from_cc(cc),
                self.settings,
            )
        else:
            if N_solid:
                raise NotImplementedError(
                    'multi-component flasher with solid phases '
                    'not implemented (yet)'
                )
            return FlashVLN(
                constants, 
                correlations,
                [self._liquid_from_cc(cc)
                  for i in range(N_liquid)],
                self._gas_from_cc(cc),
                [],
                self.settings,
            )
        


    

        
        
    
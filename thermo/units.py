# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['u']
import types
import re
import inspect
import functools
import collections
import thermo
import numpy as np
try:
    import pint
    from pint import _DEFAULT_REGISTRY as u
    from pint import DimensionalityError

except ImportError: # pragma: no cover
    raise ImportError('The unit handling in fluids requires the installation '
                      'of the package pint, available on pypi or from '
                      'https://github.com/hgrecco/pint')
from fluids.units import wraps_numpydoc, wrap_numpydoc_obj

__funcs = {}

failed_wrapping = False


for name in dir(thermo):
    if name == '__getattr__' or name == '__test__':
        continue
    obj = getattr(thermo, name)
    if isinstance(obj, types.FunctionType):
        pass
#        obj = wraps_numpydoc(u)(obj)
    elif type(obj) == type and (obj in [thermo.Chemical, thermo.Mixture, thermo.Stream]
                                 or thermo.eos.GCEOS in obj.__mro__
                                 or thermo.activity.GibbsExcess in obj.__mro__):
        if obj in (thermo.eos_mix.PSRKMixingRules, thermo.eos_mix.PSRK):
            # Not yet implemented
            continue
        try:
            obj = wrap_numpydoc_obj(obj)
        except Exception as e:
            failed_wrapping = True
            print('Current implementation of %s contains documentation not '
                  'parseable and cound not be wrapped to use pint:' %str(obj))
            print(e)
    elif isinstance(obj, str):
        continue
    if name == '__all__':
        continue
    __all__.append(name)
    __funcs.update({name: obj})

globals().update(__funcs)

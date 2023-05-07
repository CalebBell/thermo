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

Basic module which wraps all thermo functions with numpy's np.vectorize
function.

All other object - dicts, classes, etc - are not wrapped. Supports star
imports; so the same objects exported when importing from the main library
will be imported from here.

>>> from thermo.vectorized import *
>>> Antoine(np.linspace(100, 200, 5), A=8.95894, B=510.595, C=-15.95)
array([7.65674361e+02, 1.89116754e+04, 1.41237759e+05, 5.60609191e+05,
       1.53010431e+06])

Inputs do not need to be numpy arrays; they can be any iterable:

>>> import thermo.vectorized
>>> thermo.vectorized.Tc(['108-88-3', '7732-18-5'])
array([591.75, 647.14])

.. warning::
    This module does not replace the functions in the `thermo` module; it
    copies all the functions into the `thermo.vectorized` module and makes
    them vectorized there.

    For example by importing `thermo.vectorized`,
    `thermo.Antoine` won't become vectorized,
    but `thermo.vectorized.Antoine` will become available and is vectorized.

.. warning:: `np.vectorize` does not use NumPy to accelerate any computations;
   it is a convenience wrapper. If you are working on a problem large enough for
   speed to be an issue and Numba is compatible with your version of Python,
   an interface to that library is available at :obj:`thermo.numba` which does
   accelerate NumPy array computations and is normally faster than using numpy
   directly.

"""

from fluids.numerics import FakePackage
from fluids.numerics import numpy as np

import thermo

__all__ = []

bad_names = {'__file__', '__name__', '__package__', '__cached__'}

_new_globals = {}

if isinstance(np, FakePackage):
    pass
else:
    import types
    for name in dir(thermo):
        obj = getattr(thermo, name)
        if isinstance(obj, types.FunctionType):
            obj = np.vectorize(obj)
        elif isinstance(obj, str):
            if name in bad_names:
                continue
        __all__.append(name)
        _new_globals.update({name: obj})
globals().update(_new_globals)





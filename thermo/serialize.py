# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains functionality for serializing classes.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/chemicals/>`_.

.. contents:: :local:
'''
# This module SHOULD NOT import anything from thermo
from fluids.numerics import numpy as np
from chemicals.utils import PY37
try:
    array = np.array
except:
    pass

BasicNumpyEncoder = None
def build_numpy_encoder():
    '''Create a basic numpy encoder for json applications. All np ints become
    Python ints; numpy floats become python floats; and all arrays become
    lists-of-lists of floats, ints, bools, or complexes according to Python's
    rules.
    '''
    global BasicNumpyEncoder, json

    import json
    int_types = frozenset([np.short, np.ushort, np.intc, np.uintc, np.int_,
                           np.uint, np.longlong, np.ulonglong])
    float_types = frozenset([np.float16, np.single, np.double, np.longdouble,
                             np.csingle, np.cdouble, np.clongdouble])
    ndarray = np.ndarray
    JSONEncoder = json.JSONEncoder

    class BasicNumpyEncoder(JSONEncoder):
        def default(self, obj):
            t = type(obj)
            if t is ndarray:
                return obj.tolist()
            elif t in int_types:
                return int(obj)
            elif t in float_types:
                return float(obj)
            return JSONEncoder.default(self, obj)


def naive_lists_to_arrays(obj):
    t = type(obj)
    if t is dict:
        for k, v in obj.items():
            obj[k] = naive_lists_to_arrays(v)
    if t is tuple:
        return tuple(naive_lists_to_arrays(v) for v in obj)
    if t is set:
        return set(naive_lists_to_arrays(v) for v in obj)
    if t is list:
        return array(obj)
    return obj

BasicNumpyDecoder = None
def build_numpy_decoder():
    global BasicNumpyDecoder, json
    import json

    class BasicNumpyDecoder(json.JSONDecoder):
        def __init__(self, *args, **kwargs):
            json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

        def object_hook(self, obj):
            return naive_lists_to_arrays(obj)


def _load_orjson():
    global orjson
    import orjson

def dump_json_np(obj, library='json'):
    '''Serialization tool that handles numpy arrays. By default this will
    use the standard library json, but this can also use orjson.'''
    if library == 'json':
        if BasicNumpyEncoder is None:
            build_numpy_encoder()
        return json.dumps(obj, cls=BasicNumpyEncoder)
    elif library == 'orjson':
        if orjson is None:
            _load_orjson()
        opt = orjson.OPT_SERIALIZE_NUMPY
        return orjson.dumps(obj, option=opt)

def load_json_np(obj, library='json'):
    '''Serialization tool that handles numpy arrays. By default this will
    use the standard library json, but this can also use orjson.'''
    if library == 'json':
        if BasicNumpyDecoder is None:
            build_numpy_decoder()
        return json.loads(obj, cls=BasicNumpyDecoder)
    elif library == 'orjson':
        if orjson is None:
            _load_orjson()
        opt = orjson.OPT_SERIALIZE_NUMPY
        return orjson.loads(obj, option=opt)

json_loaded = False
def _load_json():
    global json
    import json
    json_loaded = True

global json, orjson
orjson = None

if PY37:
    def __getattr__(name):
        global json, json_loaded
        if name == 'json':
            import json
            json_loaded = True
            return json
        raise AttributeError("module %s has no attribute %s" %(__name__, name))
else:
    import json
    json_loaded = True


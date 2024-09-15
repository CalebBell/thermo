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
import sys

from chemicals.utils import PY37, object_data
from fluids.numerics import numpy as np

__all__ = ['object_from_json', 'json_default']
try:
    array = np.array
    int_types = frozenset([np.short, np.ushort, np.intc, np.uintc, np.int_,
                           np.uint, np.longlong, np.ulonglong])
    float_types = frozenset([np.float16, np.single, np.double, np.longdouble,
                             np.csingle, np.cdouble, np.clongdouble])
    ndarray = np.ndarray

except:
    pass

object_lookups = {}

# These are what e.g. orjson can serialize. note tuples become lists.
primitive_serialization_types = {type(None), bool, int, float, str, dict, list, tuple}
primitive_serialization_types_no_containers = {type(None), bool, int, float, str}


BasicNumpyEncoder = None
def build_numpy_encoder():
    '''Create a basic numpy encoder for json applications. All np ints become
    Python ints; numpy floats become python floats; and all arrays become
    lists-of-lists of floats, ints, bools, or complexes according to Python's
    rules.
    '''
    global BasicNumpyEncoder, json, int_types, float_types

    import json
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

def arrays_to_lists(obj):
    t = type(obj)
    if t is dict:
        # Do not modify objects in place
        obj = obj.copy()
        for k, v in obj.items():
            obj[k] = arrays_to_lists(v)
    elif t is tuple:
        return tuple(arrays_to_lists(v) for v in obj)
    elif t is set:
        # Can't put arrays in sets
        return obj
#        return set(arrays_to_lists(v) for v in obj)
    elif t is ndarray:
        return obj.tolist()
    elif t in int_types:
        return int(obj)
    elif t in float_types:
        return float(obj)
    return obj


def naive_lists_to_arrays(obj):
    t = type(obj)
    if t is dict:
        # Do not modify objects in place
        obj = obj.copy()
        for k, v in obj.items():
            obj[k] = naive_lists_to_arrays(v)
    elif t is tuple:
        return tuple(naive_lists_to_arrays(v) for v in obj)
    elif t is set:
        return {naive_lists_to_arrays(v) for v in obj}
    elif t is list:
        if len(obj) >= 2 and type(obj[0]) is list:
            # Handle tuples of different sized arrays
            try:
                length = len(obj[0])
                for v in obj[1:]:
                    if len(v) != length:
                        return tuple(array(v) for v in obj)
            except:
                pass
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
    use the standard library json, but this can also use orjson.
    '''
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
    use the standard library json, but this can also use orjson.
    '''
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

orjson = None

if PY37:
    def __getattr__(name):
        global json, json_loaded
        if name == 'json':
            import json
            json_loaded = True
            return json
        raise AttributeError(f"module {__name__} has no attribute {name}")
else:
    import json
    json_loaded = True





def object_from_json(json_object):
    if 'py/object' in json_object:
        pth = json_object['py/object']
        # TODO: Cache these lookups
        bits = pth.split('.')
        mod = '.'.join(bits[:-1])
        cls_name = bits[-1]
        obj = getattr(sys.modules[mod], cls_name)

        return obj.from_json(json_object)
    raise ValueError("Could not recognize object")


def json_default(obj):
    if hasattr(obj, 'as_json'):
        return obj.as_json()
    raise TypeError()

JSON_DEFAULT = 0
JSON_DROP_RECALCULABLE = 1

class JsonOptEncodable:
    json_version = 1
    """This attribute will be encoded into the produced json blob.
    It is specific to each object. When backwards incompatible changes are made
    to an object's structure, be sure to increment this to avoid deserializations
    producing broken objects."""

    obj_references = None
    """If this attribute is not None, instead of inspecting each object for whether it is a json-supported type,
    only these attribute names are inspected for recursion. These are also the only references
    subject to deduplication.
    """

    non_json_attributes = []
    """List of attributes to remove from a dict
    """

    def _custom_as_json(self, cache):
        # Handle anything custom
        pass
    def as_json(self, cache=None, option=0):
        # id_to_num_str: id: "pyid_{num}" where num is the length of the array plus 1 at the time of creation
        # I also want another dict, num_to_object, which contains num: object_serialization
        # When encountering an object, add it to id_to_num_str. Store the string reference to it.
        # Also add it to num_to_object.
        # Hope this works!

        base_serializer = cache is None
        if base_serializer:
            num_to_object = {"pyid_0": None} # will get replaced, this contains the output dictionary
            id_to_num_str = {id(self): "pyid_0"}
            cache = (num_to_object, id_to_num_str)
        else:
            num_to_object, id_to_num_str = cache

        d = object_data(self)
        for attr in self.non_json_attributes:
            try:
                del d[attr]
            except:
                pass
        if option & JSON_DROP_RECALCULABLE:
            try:
                for attr in self.recalculable_attributes:
                    try:
                        del d[attr]
                    except:
                        pass
            except:
                pass


        search_recurse = self.obj_references if self.obj_references is not None else list(d.keys())
        # search_recurse = set(self.obj_references).intersection(d)

        for obj_name in search_recurse:
            try:
                o = d[obj_name]
            except:
                continue
            t = type(o)
            if t in primitive_serialization_types_no_containers:
                continue
            # We support recursing into lists of expected object references but no further
            if t is list:
                # print(obj_name, o)
                json_obj_list = []
                for v in o:
                    if id(v) in id_to_num_str:
                        num_str = id_to_num_str[id(v)]
                    else:
                        num = len(id_to_num_str) # May change as we do as_json so we must re-check the length
                        num_str = f'pyid_{num}'
                        id_to_num_str[id(v)] = num_str # we must claim the number first
                        num_to_object[num_str] = v.as_json(cache)
                    json_obj_list.append(num_str)
                d[obj_name] = json_obj_list
            elif t is dict:
                json_obj_dict = {}
                for k, v in o.items():
                    if id(v) in id_to_num_str:
                        num_str = id_to_num_str[id(v)]
                    else:
                        num = len(id_to_num_str) # May change as we do as_json so we must re-check the length
                        num_str = f'pyid_{num}'
                        id_to_num_str[id(v)] = num_str # we must claim the number first
                        num_to_object[num_str] = v.as_json(cache)
                    json_obj_dict[k] = num_str
                d[obj_name] = json_obj_dict
            else:
                # Assume it's one item for performance, only other type is tuple and that's not json-able
                if id(o) in id_to_num_str:
                    num_str = id_to_num_str[id(o)]
                else:
                    num = len(id_to_num_str) # May change as we do as_json so we must re-check the length
                    num_str = f'pyid_{num}'
                    id_to_num_str[id(o)] = num_str
                    num_to_object[num_str] = o.as_json(cache)
                d[obj_name] = num_str

        if self.vectorized:
            d = arrays_to_lists(d)
        d["py/object"] = self.__full_path__
        d['json_version'] = self.json_version

        if hasattr(self, '_custom_as_json'):
            self._custom_as_json(d, cache)

        if base_serializer:
            num_to_object["pyid_0"] = d
            return num_to_object
        return d

    @classmethod
    def from_json(cls, json_repr, cache=None, ref_name='pyid_0'):
        if cache is None:
            cache = {}
        d = json_repr[ref_name]
        num_to_object = json_repr
        class_name = d['py/object']
        json_version = d['json_version']
        del d['py/object']
        del d['json_version']
        original_obj = object_lookups[class_name]
        try:
            new = original_obj.__new__(original_obj)
            cache[ref_name] = new
        except:
            new = original_obj.from_json(d)
            cache[ref_name] = new
            return new
        search_recurse = new.obj_references if new.obj_references is not None else list(d.keys())
        if d.get('vectorized'):
            d = naive_lists_to_arrays(d)

        for obj_name in search_recurse:
            try:
                o = d[obj_name]
            except:
                continue
            t = type(o)
            if t is str and o.startswith('pyid_'):
                if o not in cache:
                    JsonOptEncodable.from_json(json_repr, cache, ref_name=o)
                d[obj_name] = cache[o]
            elif t is list and len(o):
                initial_list_item = o[0]
                if type(initial_list_item) is str and initial_list_item.startswith('pyid_'):
                    created_objs = []
                    for v in o:
                        if v not in cache:
                            JsonOptEncodable.from_json(json_repr, cache, ref_name=v)
                        created_objs.append(cache[v])
                    d[obj_name] = created_objs
            elif t is dict and len(o):
                initial_list_item = next(iter(o.values()))
                if type(initial_list_item) is str and initial_list_item.startswith('pyid_'):
                    created_objs = {}
                    for k, v in o.items():
                        if v not in cache:
                            JsonOptEncodable.from_json(json_repr, cache, ref_name=v)
                        created_objs[k] = cache[v]
                    d[obj_name] = created_objs

        # Cannot use dict update because of slots
        for k, v in d.items():
            setattr(new, k, v)
        if hasattr(new, '_custom_from_json'):
            new._custom_from_json(num_to_object)
        return new

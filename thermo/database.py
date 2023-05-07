'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
'''


__all__ = []

import marshal
import os

from thermo.chemical import ChemicalConstants

folder = os.path.join(os.path.dirname(__file__), 'Misc')





def loadChemicalConstants(data, rows=True):
    '''Accepts either a marshal-style list-of-lists with fixed indexes, or
    takes in the json-style dict-of-dicts-of-dicts.
    Returns a dictionary of ChemicalConstants indexed by their CASs.
    '''
    loaded_chemicals = {}
    # Question: What if every database is a per-datasource one
    # And I fit all methods to apolynom to within their range?
    # Then we have one additional database which holds the best data source.
    # That way, all coefficients are precisely sourced.

    def add_chemical(kwargs):
        # TODO: remove to skip a function call
        constants = ChemicalConstants(**kwargs)
        loaded_chemicals[constants.CAS] = constants


    if rows:
        for row in data:
            kwargs = dict(CAS=row[0], Tc=row[1], Pc=row[2], Vc=row[3], omega=row[4], Tb=row[5],
                         Tm=row[6], Tt=row[7], Pt=row[8], Hfus=row[9], Hsub=row[10], Hfg=row[11],
                         dipole=row[12],
                         HeatCapacityGas=row[13], HeatCapacityLiquid=row[14],
                         HeatCapacitySolid=row[15],
                         ThermalConductivityLiquid=row[16], ThermalConductivityGas=row[17],
                         ViscosityLiquid=row[18], ViscosityGas=row[19],
                         EnthalpyVaporization=row[20], VaporPressure=row[21], VolumeLiquid=row[22],
                         SublimationPressure=row[23], EnthalpySublimation=row[24],
                         VolumeSolid=row[25], VolumeSupercriticalLiquid=row[26])
            add_chemical(kwargs)
    else:
        for CAS, item in data.items():
            kwargs= dict(CAS=CAS, Tc=item['Tc']['value'],
                    Pc=item['Pc']['value'],
                    Vc=item['Vc']['value'],
                    omega=item['omega']['value'],
                    Tb=item['Tb']['value'],
                    Tm=item['Tm']['value'],
                    Tt=item['Tt']['value'],
                    Pt=item['Pt']['value'],
                    Hfus=item['Hfus']['value'],
                    Hsub=item['Hsub']['value'],
                    Hfg=item['Hfg']['value'],
                    dipole=item['dipole']['value'])

            for prop_key, store in marshal_properties:
                try:
                    prop_data = item[prop_key]
                    Tmin, Tmax = prop_data['Tmin'], prop_data['Tmax']
                    coefficients = prop_data['coefficients']
                    if 'Tc' in prop_data:
                        kwargs[prop_key] = (Tmin, Tmax, prop_data['Tc'], coefficients)
                    else:
                        kwargs[prop_key] = (Tmin, Tmax, coefficients)
                except KeyError:
                    pass
#                    Tmin, Tmax, coefficients = None, None, None
#                kwargs[prop_key] = (Tmin, Tmax, coefficients)
            add_chemical(kwargs)
    return loaded_chemicals


def load_json_data(json_path):
    f = open(json_path)
    import json
    full_data = json.load(f)
    f.close()
    return full_data

def marshal_json_data(full_data, path):
    marshal_rows = []
    for CAS, data in full_data.items():
        row = [CAS]
        row.append(data['Tc']['value'])
        row.append(data['Pc']['value'])
        row.append(data['Vc']['value'])
        row.append(data['omega']['value'])
        row.append(data['Tb']['value'])
        row.append(data['Tm']['value'])
        row.append(data['Tt']['value'])
        row.append(data['Pt']['value'])
        row.append(data['Hfus']['value'])
        row.append(data['Hsub']['value'])
        row.append(data['Hfg']['value'])
        row.append(data['dipole']['value'])

        for prop_key, store in marshal_properties:
            try:
                prop_data = data[prop_key]
                Tmin, Tmax = prop_data['Tmin'], prop_data['Tmax']
                coefficients = prop_data['coefficients']
                if 'Tc' in prop_data:
                    Tc = prop_data['Tc']
                    row = (Tmin, Tmax, Tc, coefficients)
                else:
                    row = (Tmin, Tmax, coefficients)
            except KeyError:
                row = (None, None, None)

            row.append(row)

        marshal_rows.append(row)

    f = open(path, 'wb')
    marshal.dump(marshal_rows, f, 2)
    f.close()
    return marshal_rows



marshal_properties = [('HeatCapacityGas', True),
           ('HeatCapacityLiquid', True),
           ('HeatCapacitySolid', True),

           ('ThermalConductivityLiquid', True),
           ('ThermalConductivityGas', True),

           ('ViscosityLiquid', True),
           ('ViscosityGas', True),

           ('EnthalpyVaporization', True),
           ('VaporPressure', True),
           ('VolumeLiquid', True),
           ('SublimationPressure', True),
           ('EnthalpySublimation', True),
           ('VolumeSolid', True),
           ('SurfaceTension', True),
           ]


json_path = os.path.join(folder, 'constants dump.json')
binary_path = os.path.join(folder, 'binary dump.marshal')
skip = not os.path.exists(json_path)

loaded_chemicals = {}

if not skip:
    from_json = True
    if os.path.exists(binary_path):
        # get the changed dates for each file and only load from binary if
        # the binary file is newer
        json_mtime = os.path.getmtime(json_path)
        binary_mtime = os.path.getmtime(binary_path)

        if binary_mtime > json_mtime and os.path.getsize(binary_path) > 10000:
            from_json = False


    full_data = {}
    marshal_rows = []


    if from_json:
        full_data = load_json_data(json_path)
        loaded_chemicals = loadChemicalConstants(full_data, rows=False)


    marshal_data = from_json
    if marshal_data:
        try:
            marshal_rows = marshal_json_data(full_data, binary_path)
        except:
            pass

    if not from_json:
        marshal_rows = marshal.load(open(binary_path, 'rb'))
        loaded_chemicals = loadChemicalConstants(marshal_rows, rows=True)

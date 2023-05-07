'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains a small database of interaction parameters.
Only two data sets are currently included, both from ChemSep. If you would
like to add parameters to this project please make a referenced compilation of
values and submit them on GitHub.


For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

.. autoclass:: InteractionParameterDB
    :members:

.. autodata:: IPDB

   Exmple database with NRTL and PR values from ChemSep. This is lazy-loaded,
   access it as `thermo.interaction_parameters.IPDB`.

.. autoclass:: ScalarParameterDB

SPDB

    Example scalar parameters for models. This is lazy-loaded,
    access it as `thermo.interaction_parameters.SPDB`.
'''


__all__ = ['InteractionParameterDB', 'ScalarParameterDB']

import os
from math import isnan

from chemicals.identifiers import check_CAS, sorted_CAS_key
from chemicals.utils import PY37, can_load_data


class InteractionParameterDB:
    '''Basic database framework for interaction parameters.
    '''

    def __init__(self):
        self.tables = {}
        self.metadata = {}

    def load_json(self, file, name):
        '''Load a json file from disk containing interaction
        coefficients.

        The format for the file is as follows:

        A `data` key containing a dictionary with a key:

            * `CAS1 CAS2` : str
               The CAS numbers of both components, sorted from small to high
               as integers; they should have the '-' symbols still in them
               and have a single space between them; if these are ternary or
               higher parameters, follow the same format for the other CAS
               numbers, [-]

            * values : dict[str : various]
                All of the values listed in the metadata element
                `necessary keys`; they are None if missing.

        A `metadata` key containing:

            * `symmetric` : bool
               Whether or not the interaction coefficients are missing.
            * `source` : str
               Where the data came from.
            * `components` : int
               The number of components each interaction parameter is for;
               2 for binary, 3 for ternary, etc.
            * `necessary keys` : list[str]
               Which elements are required in the data.
            * `P dependent` : bool
               Whether or not the interaction parameters are pressure dependent.
            * `missing` : dict[str : float]
               Values which are missing are returned with these values
            * `type` : One of 'PR kij', 'SRK kij', etc; used to group data but not
               tied into anything else.
            * `T dependent` : bool
               Whether or not the data is T-dependent.

        Parameters
        ----------
        file : str
            Path to json file on disk which contains interaction coefficients, [-]
        name : str
            Name that the data read should be referred to by, [-]
        '''
        import json
        f = open(file).read()
        dat = json.loads(f)
        self.tables[name] = dat['data']
        self.metadata[name] = dat['metadata']

    def validate_table(self, name):
        '''Basic method which checks that all CAS numbers are valid, and that
        all elements of the data have non-nan values.
        Raises an exception if any of the data is missing or is a nan value.
        '''
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
            assert all(check_CAS(i) for i in CASs)

            values = table[key]
            for i in necessary_keys:
                # Assert all necessary keys are present
                assert i in values
                val = values[i]
                # Check they are not None
                assert val is not None
                # Check they are not nan
                assert not isnan(val)


    def has_ip_specific(self, name, CASs, ip):
        '''Check if a bip exists in a table.

        Parameters
        ----------
        name : str
            Name of the data table, [-]
        CASs : Iterable[str]
            CAS numbers; they do not need to be sorted, [-]
        ip : str
            Name of the parameter to retrieve, [-]

        Returns
        -------
        present : bool
            Whether or not the data is included in the table, [-]

        Examples
        --------
        Check if nitrogen-ethane as a PR BIP:

        >>> from thermo.interaction_parameters import IPDB
        >>> IPDB.has_ip_specific('ChemSep PR', ['7727-37-9', '74-84-0'], 'kij')
        True
        '''
        if self.metadata[name]['symmetric']:
            key = ' '.join(sorted_CAS_key(CASs))
        else:
            key = ' '.join(CASs)
        table = self.tables[name]
        if key not in table:
            return False
        return ip in table[key]

    def get_ip_specific(self, name, CASs, ip):
        '''Get an interaction parameter from a table. If the specified
        parameter is missing, the default `missing` value as defined in
        the data file is returned instead.

        Parameters
        ----------
        name : str
            Name of the data table, [-]
        CASs : Iterable[str]
            CAS numbers; they do not need to be sorted, [-]
        ip : str
            Name of the parameter to retrieve, [-]

        Returns
        -------
        value : float
            Interaction parameter specified by `ip`, [-]

        Examples
        --------
        Check if nitrogen-ethane as a PR BIP:

        >>> from thermo.interaction_parameters import IPDB
        >>> IPDB.get_ip_specific('ChemSep PR', ['7727-37-9', '74-84-0'], 'kij')
        0.0533
        '''
        if self.metadata[name]['symmetric']:
            key = ' '.join(sorted_CAS_key(CASs))
        else:
            key = ' '.join(CASs)
        try:
            return self.tables[name][key][ip]
        except KeyError:
            return self.metadata[name]['missing'][ip]

    def get_tables_with_type(self, ip_type):
        '''Get a list of tables which have a type of a parameter.

        Parameters
        ----------
        ip_type : str
            Name of the parameter type, [-]

        Returns
        -------
        table_names : list[str]
            Interaction parameter tables including `ip`, [-]

        Examples
        --------
        >>> from thermo.interaction_parameters import IPDB
        >>> IPDB.get_tables_with_type('PR kij')
        ['ChemSep PR']
        '''
        tables = []
        for key, d in self.metadata.items():
            if d['type'] == ip_type:
                tables.append(key)
        return tables

    def get_ip_automatic(self, CASs, ip_type, ip):
        '''Get an interaction parameter for the first table containing the
        value.

        Parameters
        ----------
        CASs : Iterable[str]
            CAS numbers; they do not need to be sorted, [-]
        ip_type : str
            Name of the parameter type, [-]
        ip : str
            Name of the parameter to retrieve, [-]

        Returns
        -------
        value : float
            Interaction parameter specified by `ip`, [-]

        Examples
        --------
        >>> from thermo.interaction_parameters import IPDB
        >>> IPDB.get_ip_automatic(CASs=['7727-37-9', '74-84-0'], ip_type='PR kij', ip='kij')
        0.0533
        '''
        table = self.get_tables_with_type(ip_type)[0]
        return self.get_ip_specific(table, CASs, ip)

    def get_ip_symmetric_matrix(self, name, CASs, ip, T=298.15):
        '''Get a table of interaction parameters from a specified source
        for the specified parameters. This method assumes symmetric
        parameters for speed.

        Parameters
        ----------
        name : str
            Name of the data table, [-]
        CASs : Iterable[str]
            CAS numbers; they do not need to be sorted, [-]
        ip : str
            Name of the parameter to retrieve, [-]
        T : float, optional
            Temperature of the system, [-]

        Returns
        -------
        values : list[list[float]]
            Interaction parameters specified by `ip`, [-]

        Examples
        --------
        >>> from thermo.interaction_parameters import IPDB
        >>> IPDB.get_ip_symmetric_matrix(name='ChemSep PR', CASs=['7727-37-9', '74-84-0', '74-98-6'], ip='kij')
        [[0.0, 0.0533, 0.0878], [0.0533, 0.0, 0.0011], [0.0878, 0.0011, 0.0]]
        '''
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
        return values

    def get_ip_asymmetric_matrix(self, name, CASs, ip, T=298.15):
        '''Get a table of interaction parameters from a specified source
        for the specified parameters.

        Parameters
        ----------
        name : str
            Name of the data table, [-]
        CASs : Iterable[str]
            CAS numbers; they do not need to be sorted, [-]
        ip : str
            Name of the parameter to retrieve, [-]
        T : float, optional
            Temperature of the system, [-]

        Returns
        -------
        values : list[list[float]]
            Interaction parameters specified by `ip`, [-]

        Examples
        --------
        >>> from thermo.interaction_parameters import IPDB
        >>> IPDB.get_ip_symmetric_matrix(name='ChemSep NRTL', CASs=['64-17-5', '7732-18-5', '67-56-1'], ip='alphaij')
        [[0.0, 0.2937, 0.3009], [0.2937, 0.0, 0.2999], [0.3009, 0.2999, 0.0]]
        '''
        table = self.tables[name]
        N = len(CASs)
        values = [[None for i in range(N)] for j in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    i_ip = 0.0
                else:
                    i_ip = self.get_ip_specific(name, [CASs[i], CASs[j]], ip)
                values[i][j] = i_ip
        return values

class ScalarParameterDB:
    '''Basic database framework for scalar parameters of various thermodynamic
    models. The following keys are used:

    **Peng-Robinson**

    :obj:`Twu Volume-translated Peng-Robinson <thermo.eos_mix.PRMIXTranslatedConsistent>`:
    `TwuPRL`, `TwuPRM`, `TwuPRN`, `TwuPRc`

    :obj:`Volume-translated Peng-Robinson <thermo.eos_mix.PRMIXTranslated>`:
    `PRc`

    :obj:`Peng-Robinson-Stryjek-Vera  <thermo.eos_mix.PRSVMIX>`:
    `PRSVkappa1`

    :obj:`Peng-Robinson-Stryjek-Vera 2  <thermo.eos_mix.PRSV2MIX>`:
    `PRSV2kappa1`, `PRSV2kappa2`, `PRSV2kappa3`

    **SRK**

    :obj:`Twu Volume-translated Peng-Robinson <thermo.eos_mix.SRKMIXTranslatedConsistent>`:
    `TwuSRKL`, `TwuSRKM`, `TwuSRKN`, `TwuSRKc`

    :obj:`Volume-translated Peng-Robinson <thermo.eos_mix.SRKMIXTranslated>`:
    `SRKc`

    :obj:`Refinery Soave-Redlich-Kwong  <thermo.eos_mix.APISRKMIX>`:
    `APISRKS1`, `APISRKS2`

    :obj:`MSRK  <thermo.eos_mix.MSRKMIXTranslated>`:
    `MSRKM`, `MSRKN`, `MSRKc`

    :obj:`Predictive Soave-Redlich-Kwong <thermo.eos_mix.PSRK>`:
    `MCSRKC1`, `MCSRKC2`, `MCSRKC3`


    **Excess Gibbs Energy Models**

    :obj:`Regular Solution <thermo.regular_solution.RegularSolution>`:
    `RegularSolutionV`, `RegularSolutionSP`


    '''

    def __init__(self):
        self.tables = {}
        self.metadata = {}
    def load_json(self, file, name):

        import json
        f = open(file).read()
        dat = json.loads(f)
        self.tables[name] = dat['data']
        self.metadata[name] = dat['metadata']

    def get_parameter_specific(self, name, CAS, parameter):
        '''Get a parameter from a table. If the specified
        parameter is missing, the default `missing` value as defined in
        the data file is returned instead.

        Parameters
        ----------
        name : str
            Name of the data table, [-]
        CAS : str
            CAS number, [-]
        parameter : str
            Name of the parameter to retrieve, [-]

        Returns
        -------
        value : float
            Interaction parameter specified by `ip`, [-]

        Examples
        --------
        Get the `L` Twu parameter for PR from Piña-Martinez (2021):

        >>> from thermo.interaction_parameters import SPDB
        >>> SPDB.get_parameter_specific('PRTwu_PinaMartinez', '7727-37-9', 'TwuPRL')
        0.1243
        '''
        try:
            return self.tables[name][CAS][parameter]
        except KeyError:
            return self.metadata[name]['missing'][parameter]

    def has_parameter_specific(self, name, CAS, parameter):
        '''Check if a parameter exists in a table.

        Parameters
        ----------
        name : str
            Name of the data table, [-]
        CAS : str
            CAS number, [-]
        parameter : str
            Name of the parameter to retrieve, [-]

        Returns
        -------
        present : bool
            Whether or not the data is included in the table, [-]

        Examples
        --------
        >>> from thermo.interaction_parameters import SPDB
        >>> SPDB.has_parameter_specific('PRTwu_PinaMartinez', '7727-37-9', 'TwuPRL')
        True
        >>> SPDB.has_parameter_specific('PRTwu_PinaMartinez', '7439-89-6', 'TwuPRL')
        False
        '''
        table = self.tables[name]
        if CAS not in table:
            return False
        return parameter in table[CAS]

    def get_tables_with_type(self, parameter):
        '''Get a list of tables which have a parameter.

        Parameters
        ----------
        parameter : str
            Name of the parameter, [-]

        Returns
        -------
        table_names : list[str]
            Interaction parameter tables including `parameter`, [-]

        Examples
        --------
        >>> from thermo.interaction_parameters import SPDB
        >>> SPDB.get_tables_with_type('TwuPRL')
        ['PRTwu_PinaMartinez', 'PRTwu_ibell_2018']
        '''
        tables = []
        for key, d in self.metadata.items():
            if parameter in d['missing']:
                tables.append(key)
        return tables

    def get_parameter_automatic(self, CAS, parameter):
        '''Get an interaction parameter for the first table containing the
        value.

        Parameters
        ----------
        CAS : str
            CAS number, [-]
        parameter : str
            Name of the parameter to retrieve, [-]

        Returns
        -------
        value : float
            Parameter value specified by `parameter`, [-]

        Examples
        --------
        >>> from thermo.interaction_parameters import SPDB
        >>> SPDB.get_parameter_automatic('7727-37-9', parameter='TwuPRL')
        0.1243
        '''
        table = self.get_tables_with_type(parameter)[0]
        return self.get_parameter_specific(table, CAS, parameter)

    def get_parameter_vector(self, name, CASs, parameter):
        '''Get a list of parameters from a specified source
        for the specified parameter.

        Parameters
        ----------
        name : str
            Name of the data table, [-]
        CASs : Iterable[str]
            CAS numbers; the returned values will be in this order, [-]
        parameter : str
            Name of the parameter to retrieve, [-]

        Returns
        -------
        values : list[float]
            Parameter specified by `parameter`, [-]

        Examples
        --------
        >>> from thermo.interaction_parameters import SPDB
        >>> SPDB.get_parameter_vector(name='PRTwu_PinaMartinez', CASs=['7727-37-9', '74-84-0', '74-98-6'], parameter='TwuPRL')
        [0.1243, 0.3053, 0.7455]
        >>> SPDB.get_parameter_vector(name='PRVolumeTranslation_PinaMartinez', CASs=['7727-37-9', '74-84-0', '74-98-6'], parameter='PRc')
        [-3.643e-06, -3.675e-06, -3.735e-06]
        '''
        table = self.tables[name]
        N = len(CASs)
        values = [None]*N
        for i in range(N):
            values[i] = self.get_parameter_specific(name, CASs[i], parameter)
        return values


_loaded_interactions = False
def load_all_interaction_parameters():
    global IPDB, _loaded_interactions

    folder = os.path.join(os.path.dirname(__file__), 'Interaction Parameters')
    chemsep_db_path = os.path.join(folder, 'ChemSep')
    ip_files = {'ChemSep PR': os.path.join(chemsep_db_path, 'pr.json'),
                'ChemSep NRTL': os.path.join(chemsep_db_path, 'nrtl.json'),
                'ChemSep UNIQUAC': os.path.join(chemsep_db_path, 'uniquac.json'),
                'ChemSep Wilson': os.path.join(chemsep_db_path, 'wilson.json'),

                'ChemSep Henry': os.path.join(chemsep_db_path, 'henry.json'),
                'Sander Const': os.path.join(folder, 'Sander_henry_const.json'),
                'Sander T dep': os.path.join(folder, 'Sander_henry_T_dep.json'),
                'PRTranslated Henry Best': os.path.join(folder, 'PRTranslated_best_henry_T_dep.json'),
                }


    IPDB = InteractionParameterDB()
    for name, file in ip_files.items():
        IPDB.load_json(file, name)

    _loaded_interactions = True

_loaded_scalars = False
def load_all_scalar_parameters():
    global SPDB, _loaded_scalars

    folder = os.path.join(os.path.dirname(__file__), 'Scalar Parameters')
    sp_files = {'PRTwu_PinaMartinez': os.path.join(folder, 'PRTwu_PinaMartinez.json'),
                'SRKTwu_PinaMartinez': os.path.join(folder, 'SRKTwu_PinaMartinez.json'),
                'PRVolumeTranslation_PinaMartinez': os.path.join(folder, 'PRVolumeTranslation_PinaMartinez.json'),
                'SRKVolumeTranslation_PinaMartinez': os.path.join(folder, 'SRKVolumeTranslation_PinaMartinez.json'),
                'PRTwu_ibell_2018': os.path.join(folder, 'PRTwu_ibell_2018.json'),

                'chemsep_regular_solution': os.path.join(folder, 'chemsep_regular_solution.json'),
                'chemsep_PSRK_matthias_copeman': os.path.join(folder, 'chemsep_PSRK_matthias_copeman.json'),
                'chemsep_APISRK': os.path.join(folder, 'chemsep_APISRK.json'),
                }


    SPDB = ScalarParameterDB()
    for name, file in sp_files.items():
        SPDB.load_json(file, name)

    _loaded_scalars = True

if PY37:
    def __getattr__(name):
        if name in ('IPDB',):
            load_all_interaction_parameters()
            return globals()[name]
        if name in ('SPDB',):
            load_all_scalar_parameters()
            return globals()[name]
        raise AttributeError(f"module {__name__} has no attribute {name}")
else:
    if can_load_data:
        load_all_interaction_parameters()
        load_all_scalar_parameters()

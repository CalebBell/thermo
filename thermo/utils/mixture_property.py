'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['MixtureProperty']

from chemicals.utils import hash_any_primitive, mixing_simple, normalize, ws_to_zs, zs_to_ws
from fluids.numerics import derivative, linspace, trunc_exp, trunc_log
from fluids.numerics import numpy as np

from thermo.redlich_kister import redlich_kister_build_structure, redlich_kister_excess_inner, redlich_kister_T_dependence
from thermo.serialize import JsonOptEncodable
from thermo.utils.functional import has_matplotlib
from thermo.utils.names import LINEAR, MIXING_LOG_MASS, MIXING_LOG_MOLAR
from thermo.utils.t_dependent_property import ENABLE_MIXTURE_JSON, json_mixture_correlation_lookup

try:
    from itertools import product
except:
    pass


class MixtureProperty:
    RAISE_PROPERTY_CALCULATION_ERROR = False
    name = 'Test'
    units = 'test units'
    property_min = 0.0
    property_max = 10.0
    ranked_methods = []
    TP_zs_ws_cached = (None, None, None, None)
    prop_cached = None
    _correct_pressure_pure = True
    _method = None

    pure_references = ()
    pure_reference_types = ()
    obj_references = ()
    json_version = 1
    non_json_attributes = ['TP_zs_ws_cached', 'prop_cached']
    vectorized = False
    skip_prop_validity_check = False
    """Flag to disable checking the output of the value. Saves a little time.
    """
    skip_method_validity_check = False
    """Flag to disable checking the validity of the method at the
    specified conditions. Saves a little time.
    """

    def __init_subclass__(cls):
        cls.__full_path__ = f"{cls.__module__}.{cls.__qualname__}"

    def __repr__(self):
        clsname = self.__class__.__name__
        base = f'{clsname}('
        for k in self.custom_args:
            v = getattr(self, k)
            if v is not None:
                base += f'{k}={v}, '
        base += f'CASs={self.CASs}, '
        base += f'correct_pressure_pure={self._correct_pressure_pure}, '
        base += f'method="{self.method}", '
        for attr in self.pure_references:
            base += f'{attr}={getattr(self, attr)}, '

        if base[-2:] == ', ':
            base = base[:-2]
        return base + ')'


    def __eq__(self, other):
        return self.__hash__() == hash(other)

    def __hash__(self):
        d = self.__dict__
        ans = hash_any_primitive((self.__class__, d))
        return ans

    def pure_objs(self):
        return getattr(self, self.pure_references[0])

    def __init__(self, **kwargs):
        if self.CASs:
            self.N = N = len(self.CASs)
        else:
            for attr in self.pure_constants:
                value = getattr(self, attr)
                if value:
                    self.N = N = len(value)
                    break

        self.T_limits = T_limits = {}
        self.P_limits = P_limits = {}

        self._correct_pressure_pure = kwargs.get('correct_pressure_pure', self._correct_pressure_pure)

        self.Tmin = None
        """Minimum temperature at which no method can calculate the
        property under."""
        self.Tmax = None
        """Maximum temperature at which no method can calculate the
        property above."""

        self.mixture_correlations = {}
        """Dictionary containing lookups for coefficient-based mixture excess models."""

        self.all_methods = set()
        """Set of all methods available for a given set of information;
        filled by :obj:`load_all_methods`."""
        self.load_all_methods()

        # Attempt to load json data
        CASs = self.CASs
        if CASs and None not in CASs and ENABLE_MIXTURE_JSON:
            cls_name = self.__class__.__name__
            outer = []
            for CAS1 in CASs:
                row = []
                for CAS2 in CASs:
                    found = json_mixture_correlation_lookup(CAS1, CAS2, cls_name)
                    row.append(found)
                outer.append(row)
            # Load method by method and see if we have the data we need
            redlich_kister_coeffs = []
            redlich_kister_indexes = []
            N_T = 1
            N_terms = 1

            for i in range(N):
                for j in range(N):
                    rk_dicts_ij = outer[i][j].get('redlick_kister_parameters', {})
                    if rk_dicts_ij:
                        # What to do about other data sets?
                        first_data = next(iter(rk_dicts_ij.values()))
                        if first_data['N_T'] > N_T:
                            N_T = first_data['N_T']
                        if first_data['N_terms'] > N_terms:
                            N_terms = first_data['N_terms']
                        redlich_kister_coeffs.append(first_data['coeffs'])
                        # TODO figure out order of storing i,j with data
                        if j > i:
                            redlich_kister_indexes.append((i,j))
                        else:
                            redlich_kister_indexes.append((j,i))
            # assemble the parameters
            # Only add them if anyone had them
            if redlich_kister_coeffs:
                rk_struct = redlich_kister_build_structure(N, (N_terms, N_T), redlich_kister_coeffs, redlich_kister_indexes)
                if 'redlick_kister_parameters' not in kwargs:
                    kwargs['redlick_kister_parameters']= {}
                kwargs['redlick_kister_parameters']['Combined Json'] = {'N_T': N_T, 'N_terms': N_terms, 'coeffs': rk_struct}

        if kwargs:
            mixture_excess_models = {'redlick_kister_parameters'}
            # Iterate over all the dictionaries in reverse such that the first one is left as the default
            for key in reversed(list(kwargs.keys())):
                if key in mixture_excess_models:
                    correlation_name = key.replace('_parameters', '')
                    correlation_dict = kwargs[key]
                    for corr_i in reversed(list(correlation_dict.keys())):
                        corr_kwargs = correlation_dict[corr_i]
                    # for corr_i, corr_kwargs in correlation_dict.items():
                        self.add_excess_correlation(name=corr_i, model=correlation_name, **corr_kwargs)

        try:
            method = kwargs['method']
        except:
            try:
                method = self._method
            except:
                method = None

        if method is None:
            all_methods = self.all_methods
            for i in self.ranked_methods:
                if i in all_methods:
                    method = i
                    break
        self.method = method


    def add_excess_correlation(self, name, model, **kwargs):
        d = getattr(self, model + '_parameters', None)
        if d is None:
            d = {}
            setattr(self, model + '_parameters', d)

        full_kwargs = kwargs.copy()
        d[name] = full_kwargs
        self.all_methods.add(name)
        self.method = name

        args = (kwargs['coeffs'], kwargs['N_terms'], kwargs['N_T'])

        self.mixture_correlations[name] = args

    def calculate(self, T, P, zs, ws, method):
        pure_props = self.calculate_pures_corrected(T, P, fallback=True)
        if method == LINEAR:
            return mixing_simple(zs, pure_props)
        if method == MIXING_LOG_MOLAR:
            ln_prop = 0.0
            for i in range(len(zs)):
                ln_prop += zs[i]*trunc_log(pure_props[i])
            return trunc_exp(ln_prop)
        elif method == MIXING_LOG_MASS:
            ln_prop = 0.0
            for i in range(len(ws)):
                ln_prop += ws[i]*trunc_log(pure_props[i])
            return trunc_exp(ln_prop)
        if method in self.mixture_correlations:
            rk_struct, N_terms, N_T = self.mixture_correlations[method]
            Ais_matrix_for_calc = redlich_kister_T_dependence(rk_struct, T=T, N=len(zs), N_T=N_T, N_terms=N_terms)
            excess = redlich_kister_excess_inner(N_T, N_terms, Ais_matrix_for_calc, zs)
            base_property = mixing_simple(zs, pure_props)
            return base_property + excess
        raise ValueError(f"Unknown method; methods are {self.all_methods}")

    def calculate_pures_corrected(self, T, P, fallback=False, objs=None):
        if self._correct_pressure_pure:
            pure_props = self.calculate_pures_P(T, P, fallback, objs=objs)
        else:
            pure_props = self.calculate_pures(T, objs=objs)
        return pure_props

    def calculate_pures(self, T, objs=None):
        objs = getattr(self, self.pure_references[0]) if objs is None else objs
        values = [i.T_dependent_property(T) for i in objs]
        return values

    def calculate_pures_P(self, T, P, fallback=False, objs=None):
        objs = getattr(self, self.pure_references[0]) if objs is None else objs
        if objs[0].P_dependent:
            values = []
            for o in objs:
                v = o.TP_dependent_property(T, P)
                if v is None and fallback:
                    v = o.T_dependent_property(T)
                values.append(v)
        else:
            values = [i(T) for i in objs]
        return values

    def test_method_validity(self, T, P, zs, ws, method):
        r'''Method to test the validity of a specified method for the given
        conditions.

        Parameters
        ----------
        T : float
            Temperature at which to check method validity, [K]
        P : float
            Pressure at which to check method validity, [Pa]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Method name to use

        Returns
        -------
        validity : bool
            Whether or not a specifid method is valid
        '''
        T_low, T_high = self.T_limits.get(method, (None, None))
        P_low, P_high = self.P_limits.get(method, (None, None))
        if T_low is not None and T < T_low:
            return False
        if P_low is not None and P < P_low:
            return False
        if T_high is not None and T > T_high:
            return False
        if P_high is not None and P > P_high:
            return False
        if method in self.mixture_correlations:
            return True
        raise ValueError("No check implemented for sepcified method")

    def as_json(self, cache=None, option=0):
        r'''Method to create a JSON serialization of the mixture property
        which can be stored, and reloaded later.

        Parameters
        ----------
        references : int
            How to handle references to other objects; internal parameter, [-]

        Returns
        -------
        json_repr : dict
            JSON-friendly representation, [-]

        Notes
        -----

        Examples
        --------
        '''
        return JsonOptEncodable.as_json(self, cache, option)

    def _custom_as_json(self, d, cache):
        d['all_methods'] = list(d['all_methods'])

    @classmethod
    def from_json(cls, json_repr, cache=None):
        r'''Method to create a MixtureProperty from a JSON
        serialization of another MixtureProperty.

        Parameters
        ----------
        json_repr : dict
            JSON-friendly representation, [-]

        Returns
        -------
        constants : :obj:`MixtureProperty`
            Newly created object from the json serialization, [-]

        Notes
        -----
        It is important that the input string be in the same format as that
        created by :obj:`MixtureProperty.as_json`.

        Examples
        --------
        '''
        return JsonOptEncodable.from_json(json_repr, cache)

    def _custom_from_json(self, *args):
        self.all_methods = set(self.all_methods)
        self.TP_zs_ws_cached = [None, None, None, None]

    @property
    def method(self):
        r'''Method to set the T, P, and composition dependent property method
        desired. See the :obj:`all_methods` attribute for a list of methods valid
        for the specified chemicals and inputs.
        '''
        return self._method

    @method.setter
    def method(self, method):
        self._method = method
        self.TP_zs_ws_cached = [None, None, None, None]

    @property
    def correct_pressure_pure(self):
        r'''Method to set the pressure-dependence of the model;
        if set to False, only temperature dependence is used, and if
        True, temperature and pressure dependence are used.
        '''
        return self._correct_pressure_pure

    @correct_pressure_pure.setter
    def correct_pressure_pure(self, v):
        if v != self._correct_pressure_pure:
            self._correct_pressure_pure = v
            self.TP_zs_ws_cached = [None, None, None, None]

    def _complete_zs_ws(self, zs, ws):
        if zs is None and ws is None:
            raise Exception('No Composition Specified')
        elif zs is None:
            return ws_to_zs(ws, self.MWs), ws
        elif ws is None:
            return zs, zs_to_ws(zs, self.MWs)

    def __call__(self, T, P, zs=None, ws=None):
        r'''Convenience method to calculate the property; calls
        :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`. Caches previously calculated value,
        which is an overhead when calculating many different values of
        a property. See :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>` for more details as to the
        calculation procedure. One or both of `zs` and `ws` are required.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        if (T, P, zs, ws) == self.TP_zs_ws_cached:
            return self.prop_cached
        else:
            self.prop_cached = self.mixture_property(T, P, zs, ws)
            self.TP_zs_ws_cached = [T, P, zs, ws]
            return self.prop_cached

    @classmethod
    def test_property_validity(self, prop):
        r'''Method to test the validity of a calculated property. Normally,
        this method is used by a given property class, and has maximum and
        minimum limits controlled by the variables :obj:`property_min` and
        :obj:`property_max`.

        Parameters
        ----------
        prop : float
            property to be tested, [`units`]

        Returns
        -------
        validity : bool
            Whether or not a specifid method is valid
        '''
        if isinstance(prop, complex):
            return False
        elif prop < self.property_min:
            return False
        elif prop > self.property_max:
            return False
        return True


    def mixture_property(self, T, P, zs=None, ws=None):
        r'''Method to calculate the property with sanity checking and without
        specifying a specific method. :obj:`valid_methods` is used to obtain
        a sorted list of methods to try. Methods are then tried in order until
        one succeeds. The methods are allowed to fail, and their results are
        checked with :obj:`test_property_validity`. On success, the used method
        is stored in the variable :obj:`method`.

        If :obj:`method` is set, this method is first checked for validity with
        :obj:`test_method_validity` for the specified temperature, and if it is
        valid, it is then used to calculate the property. The result is checked
        for validity, and returned if it is valid. If either of the checks fail,
        the function retrieves a full list of valid methods with
        :obj:`valid_methods` and attempts them as described above.

        If no methods are found which succeed, returns None.
        One or both of `zs` and `ws` are required.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        if zs is None or ws is None: zs, ws = self._complete_zs_ws(zs, ws)
        method = self._method
        if not self.skip_method_validity_check or self.test_method_validity(T, P, zs, ws, method):
            try:
                prop = self.calculate(T, P, zs, ws, self._method)
            except Exception as e:
                if self.RAISE_PROPERTY_CALCULATION_ERROR: raise e
            else:
                if self.skip_prop_validity_check or self.test_property_validity(prop):
                    return prop
                elif self.RAISE_PROPERTY_CALCULATION_ERROR:
                    raise RuntimeError(f"{self.name} method '{method}' computed an invalid value of {prop} {self.units}")
        elif self.RAISE_PROPERTY_CALCULATION_ERROR:
            raise RuntimeError(f"{self.name} method '{method}' is not valid at T={T} K and P={P} Pa")

    def calculate_excess_property(self, T, P, zs, ws, method):
        N = len(zs)
        prop = self.calculate(T, P, zs, ws, method)
        tot = 0.0
        for i in range(N):
            zs2, ws2 = [0.0]*N, [0.0]*N
            zs2[i], ws2[i] = 1.0, 1.0
            tot += zs[i]*self.calculate(T, P, zs2, ws2, method)
        return prop - tot

    def excess_property(self, T, P, zs=None, ws=None):
        r'''Method to calculate the excess property with sanity checking and
        without specifying a specific method. This requires the calculation of
        the property as a function of composition at the limiting concentration
        of each component. One or both of `zs` and `ws` are required.

        .. math::
            m^E = m_{mixing} = m - \sum_i m_{i, pure}\cdot z_i

        Parameters
        ----------
        T : float
            Temperature at which to calculate the excess property, [K]
        P : float
            Pressure at which to calculate the excess property, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]

        Returns
        -------
        excess_prop : float
            Calculated excess property, [`units`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        return self.calculate_excess_property(T, P, zs, ws, self._method)

    def partial_property(self, T, P, i, zs=None, ws=None):
        r'''Method to calculate the partial molar property with sanity checking
        and without specifying a specific method for the specified compound
        index and composition.

        .. math::
            \bar m_i = \left( \frac{\partial (n_T m)} {\partial n_i}
            \right)_{T, P, n_{j\ne i}}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the partial property, [K]
        P : float
            Pressure at which to calculate the partial property, [Pa]
        i : int
            Compound index, [-]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]

        Returns
        -------
        partial_prop : float
            Calculated partial property, [`units`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        def prop_extensive(ni, ns, i):
            ns[i] = ni
            n_tot = sum(ns)
            zs = normalize(ns)
            prop = self.mixture_property(T, P, zs)
            return prop*n_tot
        return derivative(prop_extensive, zs[i], dx=1E-6, args=(list(zs), i))


    def calculate_derivative_T(self, T, P, zs, ws, method, order=1):
        r'''Method to calculate a derivative of a mixture property with respect
        to temperature at constant pressure and composition
        of a given order using a specified  method. Uses SciPy's derivative
        function, with a delta of 1E-6 K and a number of points equal to
        2*order + 1.

        This method can be overwritten by subclasses who may perfer to add
        analytical methods for some or all methods as this is much faster.

        If the calculation does not succeed, returns the actual error
        encountered.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        P : float
            Pressure at which to calculate the derivative, [Pa]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Method for which to find the derivative
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        d_prop_d_T_at_P : float
            Calculated derivative property at constant pressure,
            [`units/K^order`]
        '''
        return derivative(self.calculate, T, dx=1e-6, args=(P, zs, ws, method), n=order, order=1+order*2)

    def calculate_derivative_P(self, P, T, zs, ws, method, order=1):
        r'''Method to calculate a derivative of a mixture property with respect
        to pressure at constant temperature and composition
        of a given order using a specified method. Uses SciPy's derivative
        function, with a delta of 0.01 Pa and a number of points equal to
        2*order + 1.

        This method can be overwritten by subclasses who may perfer to add
        analytical methods for some or all methods as this is much faster.

        If the calculation does not succeed, returns the actual error
        encountered.

        Parameters
        ----------
        P : float
            Pressure at which to calculate the derivative, [Pa]
        T : float
            Temperature at which to calculate the derivative, [K]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Method for which to find the derivative
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        d_prop_d_P_at_T : float
            Calculated derivative property at constant temperature,
            [`units/Pa^order`]
        '''
        f = lambda P: self.calculate(T, P, zs, ws, method)
        return derivative(f, P, dx=1e-2, n=order, order=1+order*2)


    def property_derivative_T(self, T, P, zs=None, ws=None, order=1):
        r'''Method to calculate a derivative of a mixture property with respect
        to temperature at constant pressure and composition,
        of a given order. Methods found valid by :obj:`valid_methods` are
        attempted until a method succeeds. If no methods are valid and succeed,
        None is returned.

        Calls :obj:`calculate_derivative_T` internally to perform the actual
        calculation.

        .. math::
            \text{derivative} = \frac{d (\text{property})}{d T}|_{P, z}

        One or both of `zs` and `ws` are required.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        P : float
            Pressure at which to calculate the derivative, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        d_prop_d_T_at_P : float
            Calculated derivative property, [`units/K^order`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        try:
            return self.calculate_derivative_T(T, P, zs, ws, self._method, order)
        except:
            pass
        return None


    def property_derivative_P(self, T, P, zs=None, ws=None, order=1):
        r'''Method to calculate a derivative of a mixture property with respect
        to pressure at constant temperature and composition,
        of a given order. Methods found valid by :obj:`valid_methods` are
        attempted until a method succeeds. If no methods are valid and succeed,
        None is returned.

        Calls :obj:`calculate_derivative_P` internally to perform the actual
        calculation.

        .. math::
            \text{derivative} = \frac{d (\text{property})}{d P}|_{T, z}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        P : float
            Pressure at which to calculate the derivative, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        d_prop_d_P_at_T : float
            Calculated derivative property, [`units/Pa^order`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        try:
            return self.calculate_derivative_P(P, T, zs, ws, self._method, order)
        except:
            pass
        return None

    def plot_isotherm(self, T, zs=None, ws=None, Pmin=None, Pmax=None,
                      methods=[], pts=50, only_valid=True):  # pragma: no cover
        r'''Method to create a plot of the property vs pressure at a specified
        temperature and composition according to either a specified list of
        methods, or the set method. User-selectable
        number of  points, and pressure range. If only_valid is set,
        :obj:`test_method_validity` will be used to check if each condition in
        the specified range is valid, and :obj:`test_property_validity` will be used
        to test the answer, and the method is allowed to fail; only the valid
        points will be plotted. Otherwise, the result will be calculated and
        displayed as-is. This will not suceed if the method fails.
        One or both of `zs` and `ws` are required.

        Parameters
        ----------
        T : float
            Temperature at which to create the plot, [K]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]
        Pmin : float
            Minimum pressure, to begin calculating the property, [Pa]
        Pmax : float
            Maximum pressure, to stop calculating the property, [Pa]
        methods : list, optional
            List of methods to consider
        pts : int, optional
            A list of points to calculate the property at; if Pmin to Pmax
            covers a wide range of method validities, only a few points may end
            up calculated for a given method so this may need to be large
        only_valid : bool
            If True, only plot successful methods and calculated properties,
            and handle errors; if False, attempt calculation without any
            checking and use methods outside their bounds
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        # This function cannot be tested
        if has_matplotlib():
            import matplotlib.pyplot as plt
        else:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if Pmin is None:
            if self.Pmin is not None:
                Pmin = self.Pmin
            else:
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Pmax is None:
            if self.Pmax is not None:
                Pmax = self.Pmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')

        if not methods:
            methods = [self._method]
        Ps = linspace(Pmin, Pmax, pts)
        for method in methods:
            if only_valid:
                properties, Ps2 = [], []
                for P in Ps:
                    if self.test_method_validity(T, P, zs, ws, method):
                        try:
                            p = self.calculate(T, P, zs, ws, method)
                            if self.test_property_validity(p):
                                properties.append(p)
                                Ps2.append(P)
                        except:
                            pass
                plt.plot(Ps2, properties, label=method)
            else:
                properties = [self.calculate(T, P, zs, ws, method) for P in Ps]
                plt.plot(Ps, properties, label=method)
        plt.legend(loc='best')
        plt.ylabel(self.name + ', ' + self.units)
        plt.xlabel('Pressure, Pa')
        plt.title(self.name + ' of a mixture of ' + ', '.join(self.CASs)
                  + ' at mole fractions of ' + ', '.join(str(round(i, 4)) for i in zs) + '.')
        plt.show()


    def plot_isobar(self, P, zs=None, ws=None, Tmin=None, Tmax=None,
                    methods=[], pts=50, only_valid=True):  # pragma: no cover
        r'''Method to create a plot of the property vs temperature at a
        specific pressure and composition according to
        either a specified list of methods, or the selected method. User-selectable number of points, and temperature range. If
        only_valid is set,:obj:`test_method_validity` will be used to check if
        each condition in the specified range is valid, and
        :obj:`test_property_validity` will be used to test the answer, and the
        method is allowed to fail; only the valid points will be plotted.
        Otherwise, the result will be calculated and displayed as-is. This will
        not suceed if the method fails. One or both of `zs` and `ws` are
        required.

        Parameters
        ----------
        P : float
            Pressure for the isobar, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]
        Tmin : float
            Minimum temperature, to begin calculating the property, [K]
        Tmax : float
            Maximum temperature, to stop calculating the property, [K]
        methods : list, optional
            List of methods to consider
        pts : int, optional
            A list of points to calculate the property at; if Tmin to Tmax
            covers a wide range of method validities, only a few points may end
            up calculated for a given method so this may need to be large
        only_valid : bool
            If True, only plot successful methods and calculated properties,
            and handle errors; if False, attempt calculation without any
            checking and use methods outside their bounds
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        if has_matplotlib():
            import matplotlib.pyplot as plt
        else:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if Tmin is None:
            if self.Tmin is not None:
                Tmin = self.Tmin
            else:
                raise Exception('Minimum temperature could not be auto-detected; please provide it')
        if Tmax is None:
            if self.Tmax is not None:
                Tmax = self.Tmax
            else:
                raise Exception('Maximum temperature could not be auto-detected; please provide it')

        if not methods:
            methods = [self._method]
        Ts = linspace(Tmin, Tmax, pts)
        for method in methods:
            if only_valid:
                properties, Ts2 = [], []
                for T in Ts:
                    if self.test_method_validity(T, P, zs, ws, method):
                        try:
                            p = self.calculate(T, P, zs, ws, method)
                            if self.test_property_validity(p):
                                properties.append(p)
                                Ts2.append(T)
                        except:
                            pass
                plt.plot(Ts2, properties, label=method)
            else:
                properties = [self.calculate(T, P, zs, ws, method) for T in Ts]
                plt.plot(Ts, properties, label=method)
        plt.legend(loc='best')
        plt.ylabel(self.name + ', ' + self.units)
        plt.xlabel('Temperature, K')
        plt.title(self.name + ' of a mixture of ' + ', '.join(self.CASs)
                  + ' at mole fractions of ' + ', '.join(str(round(i, 4)) for i in zs) + '.')
        plt.show()

    def plot_isobaric_isothermal(self, T, P, methods=[], pts=50, only_valid=True,
                                 plot='property'):  # pragma: no cover
        if has_matplotlib():
            import matplotlib.pyplot as plt
        else:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if not methods:
            methods = [self._method]
        if self.N != 2:
            raise ValueError("Only binary systems are supported")

        if plot is None or plot == 'property':
            func = self.calculate
            prop_name = self.name
        elif plot == 'excess':
            func = self.calculate_excess_property
            only_valid = False
            prop_name = 'Excess ' + self.name
        if isinstance(T, (tuple, list, np.ndarray)):
            iter_Ts = T
        else:
            iter_Ts = [T]
        if isinstance(P, (tuple, list, np.ndarray)):
            iter_Ps = P
        else:
            iter_Ps = [P]
        if len(iter_Ps) != len(iter_Ts):
            # Create a product of the lists of they are not the same size
            combined = list(product(iter_Ts, iter_Ps))
            iter_Ts = [T for T, _ in combined]
            iter_Ps = [P for _, P in combined]


        xs = linspace(0, 1, pts)
        for T, P in zip(iter_Ts, iter_Ps):
            for method in methods:
                if only_valid:
                    properties = []
                    xs_plot = []
                    for x0 in xs:
                        comp = [x0, 1.0 - x0]
                        if self.test_method_validity(T, P, comp, None, method):
                            try:
                                p = func(T, P, comp, None, method)
                                if self.test_property_validity(p):
                                    properties.append(p)
                                    xs_plot.append(x0)
                            except:
                                pass
                    plt.plot(xs_plot, properties, label=method + f' at {T:g} K and {P:g} Pa' )
                else:
                    properties = [func(T, P, [x0, 1.0 - x0], None, method) for x0 in xs]
                    plt.plot(xs, properties, label=method + f' at {T:g} K and {P:g} Pa')
        plt.legend(loc='best')
        plt.ylabel(prop_name + ', ' + self.units)
        plt.xlabel('Mole fraction x0')
        plt.title(prop_name + ' of a mixture of ' + ', '.join(self.CASs))
        plt.show()

    def plot_property(self, zs=None, ws=None, Tmin=None, Tmax=None, Pmin=1E5,
                      Pmax=1E6, methods=[], pts=15, only_valid=True):  # pragma: no cover
        r'''Method to create a plot of the property vs temperature and pressure
        according to either a specified list of methods, or the selected method.
        User-selectable number of points for each
        variable. If only_valid is set,:obj:`test_method_validity` will be used to
        check if each condition in the specified range is valid, and
        :obj:`test_property_validity` will be used to test the answer, and the
        method is allowed to fail; only the valid points will be plotted.
        Otherwise, the result will be calculated and displayed as-is. This will
        not suceed if the any method fails for any point. One or both of `zs`
        and `ws` are required.

        Parameters
        ----------
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]
        Tmin : float
            Minimum temperature, to begin calculating the property, [K]
        Tmax : float
            Maximum temperature, to stop calculating the property, [K]
        Pmin : float
            Minimum pressure, to begin calculating the property, [Pa]
        Pmax : float
            Maximum pressure, to stop calculating the property, [Pa]
        methods : list, optional
            List of methods to consider
        pts : int, optional
            A list of points to calculate the property at for both temperature
            and pressure; pts^2 points will be calculated.
        only_valid : bool
            If True, only plot successful methods and calculated properties,
            and handle errors; if False, attempt calculation without any
            checking and use methods outside their bounds
        '''
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        else:
            import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        from numpy import ma
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        if Pmin is None:
            if self.Pmin is not None:
                Pmin = self.Pmin
            else:
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Pmax is None:
            if self.Pmax is not None:
                Pmax = self.Pmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')
        if Tmin is None:
            if self.Tmin is not None:
                Tmin = self.Tmin
            else:
                raise Exception('Minimum temperature could not be auto-detected; please provide it')
        if Tmax is None:
            if self.Tmax is not None:
                Tmax = self.Tmax
            else:
                raise Exception('Maximum temperature could not be auto-detected; please provide it')

        if not methods:
            methods = [self._method]
        Ps = np.linspace(Pmin, Pmax, pts)
        Ts = np.linspace(Tmin, Tmax, pts)
        Ts_mesh, Ps_mesh = np.meshgrid(Ts, Ps)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection="3d")

        handles = []
        for method in methods:
            if only_valid:
                properties = []
                for T in Ts:
                    T_props = []
                    for P in Ps:
                        if self.test_method_validity(T, P, zs, ws, method):
                            try:
                                p = self.calculate(T, P, zs, ws, method)
                                if self.test_property_validity(p):
                                    T_props.append(p)
                                else:
                                    T_props.append(None)
                            except:
                                T_props.append(None)
                        else:
                            T_props.append(None)
                    properties.append(T_props)
                properties = ma.masked_invalid(np.array(properties, dtype=np.float64).T)
                handles.append(ax.plot_surface(Ts_mesh, Ps_mesh, properties, cstride=1, rstride=1, alpha=0.5))
            else:
                properties = [[self.calculate(T, P, zs, ws, method) for P in Ps] for T in Ts]
                handles.append(ax.plot_surface(Ts_mesh, Ps_mesh, properties, cstride=1, rstride=1, alpha=0.5))

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.set_xlabel('Temperature, K')
        ax.set_ylabel('Pressure, Pa')
        ax.set_zlabel(self.name + ', ' + self.units)
        plt.title(self.name + ' of a mixture of ' + ', '.join(self.CASs)
                  + ' at mole fractions of ' + ', '.join(str(round(i, 4)) for i in zs) + '.')
        plt.show(block=False)


    def plot_binary(self, P=None, T=None, pts=30, Tmin=None, Tmax=None, Pmin=1E5,
                    Pmax=1E6, methods=[], only_valid=True): # pragma: no cover
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        else:
            import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        from numpy import ma



        if T is None:
            # vary T
            if Tmin is None:
                if self.Tmin is not None:
                    Tmin = self.Tmin
                else:
                    raise Exception('Minimum temperature could not be auto-detected; please provide it')
            if Tmax is None:
                if self.Tmax is not None:
                    Tmax = self.Tmax
                else:
                    raise Exception('Maximum temperature could not be auto-detected; please provide it')
            vary = Ts = linspace(Tmin, Tmax, pts)
        if P is None:
            if Pmin is None:
                if self.Pmin is not None:
                    Pmin = self.Pmin
                else:
                    raise Exception('Minimum pressure could not be auto-detected; please provide it')
            if Pmax is None:
                if self.Pmax is not None:
                    Pmax = self.Pmax
                else:
                    raise Exception('Maximum pressure could not be auto-detected; please provide it')
            vary = Ps = linspace(Pmin, Pmax, pts)
        if not methods:
            methods = [self._method]

        xs = linspace(0, 1, pts)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection="3d")

        vary_mesh, xs_mesh = np.meshgrid(vary, xs)
        handles = []
        for method in methods:
            if only_valid:
                properties = []
                for v in vary:
                    v_props = []
                    T_set, P_set = (T, v) if T is not None else (v, P)
                    for x0 in xs:
                        if self.test_method_validity(T_set, P_set, [x0, 1.0 - x0], None, method):
                            try:
                                p = self.calculate(T_set, P_set, [x0, 1.0 - x0], None, method)
                                if self.test_property_validity(p):
                                    v_props.append(p)
                                else:
                                    v_props.append(None)
                            except:
                                v_props.append(None)
                        else:
                            v_props.append(None)
                    properties.append(v_props)
                properties = ma.masked_invalid(np.array(properties, dtype=np.float64).T)
                handles.append(ax.plot_surface(vary_mesh, xs_mesh, properties, cstride=1, rstride=1, alpha=0.5))
            else:
                properties = []
                for v in vary:
                    row = []
                    for x0 in xs:
                        T_set, P_set = (T, v) if T is not None else (v, P)
                        row.append(self.calculate(T_set, P_set, [x0, 1.0-x0], None, method))
                    properties.append(row)
                handles.append(ax.plot_surface(vary_mesh, xs_mesh, np.array(properties).T, cstride=1, rstride=1, alpha=0.5))

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.set_xlabel('Temperature, K' if T is None else 'Pressure, Pa')
        ax.set_ylabel('Mole Fraction x0')
        ax.set_zlabel(self.name + ', ' + self.units)
        plt.title(self.name + ' binary system of ' + ', '.join(self.CASs))
        plt.show(block=False)

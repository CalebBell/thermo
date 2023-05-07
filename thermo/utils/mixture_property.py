# -*- coding: utf-8 -*-
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
SOFTWARE.'''

__all__ = ['MixtureProperty']

from chemicals.utils import (hash_any_primitive, ws_to_zs, zs_to_ws,
                             normalize)
from fluids.numerics import linspace, derivative, numpy as np
from thermo.utils import has_matplotlib, POLY_FIT
from thermo.eos_mix import GCEOSMIX

class MixtureProperty(object):
    RAISE_PROPERTY_CALCULATION_ERROR = False
    name = 'Test'
    units = 'test units'
    property_min = 0.0
    property_max = 10.0
    ranked_methods = []
    TP_zs_ws_cached = (None, None, None, None)
    prop_cached = None
    all_poly_fit = False
    _correct_pressure_pure = True
    _method = None

    pure_references = ()
    pure_reference_types = ()

    skip_prop_validity_check = False
    """Flag to disable checking the output of the value. Saves a little time.
    """
    skip_method_validity_check = False
    """Flag to disable checking the validity of the method at the
    specified conditions. Saves a little time.
    """

    def __init_subclass__(cls):
        cls.__full_path__ = "%s.%s" %(cls.__module__, cls.__qualname__)

    def set_poly_fit_coeffs(self):
        pure_objs = self.pure_objs()
        if all(i.method == POLY_FIT for i in pure_objs):
            self.all_poly_fit = True
            self.poly_fit_data = [[i.poly_fit_Tmin for i in pure_objs],
                               [i.poly_fit_Tmin_slope for i in pure_objs],
                               [i.poly_fit_Tmin_value for i in pure_objs],
                               [i.poly_fit_Tmax for i in pure_objs],
                               [i.poly_fit_Tmax_slope for i in pure_objs],
                               [i.poly_fit_Tmax_value for i in pure_objs],
                               [i.poly_fit_coeffs for i in pure_objs]]


    def __repr__(self):
        clsname = self.__class__.__name__
        base = '%s(' % (clsname)
        for k in self.custom_args:
            v = getattr(self, k)
            if v is not None:
                base += '%s=%s, ' %(k, v)
        base += 'CASs=%s, ' %(self.CASs)
        base += 'correct_pressure_pure=%s, ' %(self._correct_pressure_pure)
        base += 'method="%s", ' %(self.method)
        for attr in self.pure_references:
            base += '%s=%s, ' %(attr, getattr(self, attr))

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
        self._correct_pressure_pure = kwargs.get('correct_pressure_pure', self._correct_pressure_pure)

        self.Tmin = None
        """Minimum temperature at which no method can calculate the
        property under."""
        self.Tmax = None
        """Maximum temperature at which no method can calculate the
        property above."""

        self.all_methods = set()
        """Set of all methods available for a given set of information;
        filled by :obj:`load_all_methods`."""
        self.load_all_methods()

        self.set_poly_fit_coeffs()

        if 'method' in kwargs: self.method = kwargs['method']

    def as_json(self, references=1):
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

        d = self.__dict__.copy() # Not a the real object dictionary
        if references == 1:
            for k in self.pure_references:
                d[k] = [v.as_json() for v in d[k]]

        try:
            eos = getattr(self, 'eos')
            if eos:
                d['eos'] = eos[0].as_json()
        except:
            pass

        d['json_version'] = 1
        d["py/object"] = self.__full_path__
        d['all_methods'] = list(d['all_methods'])
        return d

    @classmethod
    def from_json(cls, string):
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
        d = string
        d['all_methods'] = set(d['all_methods'])
        for k, sub_cls in zip(cls.pure_references, cls.pure_reference_types):
            sub_jsons = d[k]
            d[k] = [sub_cls.from_json(j) if j is not None else None
                    for j in sub_jsons]

        try:
            eos = d['eos']
            if eos is not None:
                d['eos'] = [GCEOSMIX.from_json(eos)]
        except:
            pass
        del d['py/object']
        del d["json_version"]
        new = cls.__new__(cls)
        new.__dict__ = d
        return new

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
                    raise RuntimeError("%s method '%s' computed an invalid value of %s %s" %(self.name, method, prop, self.units))
        elif self.RAISE_PROPERTY_CALCULATION_ERROR:
            raise RuntimeError("%s method '%s' is not valid at T=%s K and P=%s Pa" %(self.name, method, T, P))

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
        N = len(zs)
        prop = self.mixture_property(T, P, zs, ws)
        tot = 0.0
        for i in range(N):
            zs2, ws2 = [0.0]*N, [0.0]*N
            zs2[i], ws2[i] = 1.0, 1.0
            tot += zs[i]*self.mixture_property(T, P, zs2, ws2)
        return prop - tot

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
        return derivative(prop_extensive, zs[i], dx=1E-6, args=[list(zs), i])


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
        return derivative(self.calculate, T, dx=1e-6, args=[P, zs, ws, method], n=order, order=1+order*2)

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
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Tmax is None:
            if self.Tmax is not None:
                Tmax = self.Tmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')

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
        if has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        else:
            import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib.ticker import FormatStrFormatter
        import numpy.ma as ma
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
        ax = fig.gca(projection='3d')

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
                properties = ma.masked_invalid(np.array(properties, dtype=np.float).T)
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
        # The below is a workaround for a matplotlib bug
        ax.legend(handles, methods)
        plt.show(block=False)

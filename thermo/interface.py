'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 207, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains implementations of :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
representing liquid-air surface tension. A variety of estimation
and data methods are available as included in the `chemicals` library.
Additionally a liquid mixture surface tension predictor objects
are implemented subclassing :obj:`MixtureProperty <thermo.utils.MixtureProperty>`.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.


.. contents:: :local:

Pure Liquid Surface Tension
===========================
.. autoclass:: SurfaceTension
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: surface_tension_methods

Mixture Surface Tension
=======================
.. autoclass:: SurfaceTensionMixture
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: surface_tension_mixture_methods


'''


__all__ = ['surface_tension_methods', 'SurfaceTension',
           'surface_tension_mixture_methods', 'SurfaceTensionMixture']

from chemicals import interface, miscdata
from chemicals.dippr import EQ106
from chemicals.iapws import iapws95_Tc
from chemicals.interface import (
    Aleem,
    Brock_Bird,
    Diguilio_Teja,
    Jasper,
    Miqueu,
    Pitzer_sigma,
    REFPROP_sigma,
    Sastri_Rao,
    Somayajulu,
    Winterfeld_Scriven_Davis,
    Zuo_Stenby,
    sigma_IAPWS,
)
from chemicals.miscdata import lookup_VDI_tabular_data
from chemicals.utils import Vm_to_rho, none_and_length_check, property_molar_to_mass
from fluids.numerics import isnan

from thermo.heat_capacity import HeatCapacityLiquid
from thermo.utils import IAPWS, LINEAR, REFPROP_FIT, VDI_TABULAR, MixtureProperty, TDependentProperty
from thermo.volume import VolumeLiquid

STREFPROP = 'REFPROP'
SOMAYAJULU2 = 'SOMAYAJULU2'
SOMAYAJULU = 'SOMAYAJULU'
JASPER = 'JASPER'
MIQUEU = 'MIQUEU'
BROCK_BIRD = 'BROCK_BIRD'
SASTRI_RAO = 'SASTRI_RAO'
PITZER = 'PITZER'
ZUO_STENBY = 'ZUO_STENBY'
HAKIM_STEINBERG_STIEL = 'HAKIM_STEINBERG_STIEL'
ALEEM = 'Aleem'
VDI_PPDS = 'VDI_PPDS'


surface_tension_methods = [IAPWS, REFPROP_FIT, STREFPROP, SOMAYAJULU2, SOMAYAJULU, VDI_PPDS, VDI_TABULAR,
                           JASPER, MIQUEU, BROCK_BIRD, SASTRI_RAO, PITZER,
                           ZUO_STENBY, ALEEM]
"""Holds all methods available for the :obj:`SurfaceTension` class, for use in
iterating over them."""


class SurfaceTension(TDependentProperty):
    '''Class for dealing with surface tension as a function of temperature.
    Consists of three coefficient-based methods and four data sources, one
    source of tabular information, five corresponding-states estimators,
    and one substance-specific method.

    Parameters
    ----------
    Tb : float, optional
        Boiling point, [K]
    MW : float, optional
        Molecular weight, [g/mol]
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    Vc : float, optional
        Critical volume, [m^3/mol]
    Zc : float, optional
        Critical compressibility
    omega : float, optional
        Acentric factor, [-]
    StielPolar : float, optional
        Stiel polar factor
    Hvap_Tb : float
        Mass enthalpy of vaporization at the normal boiling point [kg/m^3]
    CASRN : str, optional
        The CAS number of the chemical
    Vml : float or callable, optional
        Liquid molar volume at a given temperature and pressure or callable
        for the same, [m^3/mol]
    Cpl : float or callable, optional
        Molar heat capacity of the fluid at a pressure and temperature or
        or callable for the same, [J/mol/K]
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`surface_tension_methods`.

    **IAPWS**:
        The IAPWS formulation for water,
        :obj:`REFPROP_sigma <chemicals.interface.sigma_IAPWS>`
    **STREFPROP**:
        The REFPROP coefficient-based method, documented in the function
        :obj:`REFPROP_sigma <chemicals.interface.REFPROP_sigma>` for 115 fluids from [5]_.
    **SOMAYAJULU and SOMAYAJULU2**:
        The Somayajulu coefficient-based method,
        documented in the function :obj:`Somayajulu <chemicals.interface.Somayajulu>`. Both methods have data
        for 64 fluids. The first data set if from [1]_, and the second
        from [2]_. The later, revised coefficients should be used prefered.
    **JASPER**:
        Fit with a single temperature coefficient from Jaspen (1972)
        as documented in the function :obj:`Jasper <chemicals.interface.Jasper>`. Data for 522 fluids is
        available, as shown in [4]_ but originally in [3]_.
    **BROCK_BIRD**:
        CSP method documented in :obj:`Brock_Bird <chemicals.interface.Brock_Bird>`.
        Most popular estimation method; from 1955.
    **SASTRI_RAO**:
        CSP method documented in :obj:`Sastri_Rao <chemicals.interface.Sastri_Rao>`.
        Second most popular estimation method; from 1995.
    **PITZER**:
        CSP method documented in :obj:`Pitzer_sigma <chemicals.interface.Pitzer_sigma>`; from 1958.
    **ZUO_STENBY**:
        CSP method documented in :obj:`Zuo_Stenby <chemicals.interface.Zuo_Stenby>`; from 1997.
    **MIQUEU**:
        CSP method documented in :obj:`Miqueu <chemicals.interface.Miqueu>`.
    **ALEEM**:
        CSP method documented in :obj:`Aleem <chemicals.interface.Aleem>`.
    **VDI_TABULAR**:
        Tabular data in [6]_ along the saturation curve; interpolation is as
        set by the user or the default.
    **REFPROP_FIT**:
        A series of higher-order polynomial fits to the calculated results from
        the equations implemented in REFPROP.

    See Also
    --------
    chemicals.interface.REFPROP_sigma
    chemicals.interface.Somayajulu
    chemicals.interface.Jasper
    chemicals.interface.Brock_Bird
    chemicals.interface.Sastri_Rao
    chemicals.interface.Pitzer
    chemicals.interface.Zuo_Stenby
    chemicals.interface.Miqueu
    chemicals.interface.Aleem
    chemicals.interface.sigma_IAPWS

    References
    ----------
    .. [1] Somayajulu, G. R. "A Generalized Equation for Surface Tension from
       the Triple Point to the Critical Point." International Journal of
       Thermophysics 9, no. 4 (July 1988): 559-66. doi:10.1007/BF00503154.
    .. [2] Mulero, A., M. I. Parra, and I. Cachadina. "The Somayajulu
       Correlation for the Surface Tension Revisited." Fluid Phase
       Equilibria 339 (February 15, 2013): 81-88.
       doi:10.1016/j.fluid.2012.11.038.
    .. [3] Jasper, Joseph J. "The Surface Tension of Pure Liquid Compounds."
       Journal of Physical and Chemical Reference Data 1, no. 4
       (October 1, 1972): 841-1010. doi:10.1063/1.3253106.
    .. [4] Speight, James. Lange's Handbook of Chemistry. 16 edition.
       McGraw-Hill Professional, 2005.
    .. [5] Mulero, A., I. Cachadiña, and M. I. Parra. “Recommended
       Correlations for the Surface Tension of Common Fluids.” Journal of
       Physical and Chemical Reference Data 41, no. 4 (December 1, 2012):
       043105. doi:10.1063/1.4768782.
    .. [6] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''

    name = 'Surface tension'
    units = 'N/m'
    interpolation_T = None
    """No interpolation transformation by default."""
    interpolation_property = None
    """No interpolation transformation by default."""
    interpolation_property_inv = None
    """No interpolation transformation by default."""
    tabular_extrapolation_permitted = True
    """Allow tabular extrapolation by default; values below 0 will be obtained
    at high temperatures."""
    property_min = 0
    """Mimimum valid value of surface tension. This occurs at the critical
    point exactly."""
    property_max = 4.0
    """Maximum valid value of surface tension. Set to roughly twice that of
    cobalt at its melting point."""
    critical_zero = True
    """Whether or not the property is declining and reaching zero at the
    critical point."""

    ranked_methods = [IAPWS, REFPROP_FIT, STREFPROP, SOMAYAJULU2, SOMAYAJULU, VDI_PPDS, VDI_TABULAR,
                      JASPER, MIQUEU, BROCK_BIRD, SASTRI_RAO, PITZER,
                      ZUO_STENBY, ALEEM]
    """Default rankings of the available methods."""

    _fit_force_n = {}
    """Dictionary containing method: fit_n, for use in methods which should
    only ever be fit to a specific `n` value"""
    _fit_force_n[JASPER] = 2

    custom_args = ('MW', 'Tb', 'Tc', 'Pc', 'Vc', 'Zc', 'omega', 'StielPolar',
                   'Hvap_Tb', 'Vml', 'Cpl')





    obj_references = pure_references = ('Vml', 'Cpl')
    obj_references_types = pure_reference_types = (VolumeLiquid, HeatCapacityLiquid)


    def __init__(self, MW=None, Tb=None, Tc=None, Pc=None, Vc=None, Zc=None,
                 omega=None, StielPolar=None, Hvap_Tb=None, CASRN='', Vml=None,
                 Cpl=None, extrapolation='DIPPR106_AB', **kwargs):
        self.MW = MW
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.Vc = Vc
        self.Zc = Zc
        self.omega = omega
        self.StielPolar = StielPolar
        self.Hvap_Tb = Hvap_Tb
        self.CASRN = CASRN
        self.Vml = Vml
        self.Cpl = Cpl

        super().__init__(extrapolation, **kwargs)


    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        return {STREFPROP: interface.sigma_data_Mulero_Cachadina.index,
                SOMAYAJULU2: interface.sigma_data_Somayajulu2.index,
                SOMAYAJULU: interface.sigma_data_Somayajulu.index,
                VDI_TABULAR: list(miscdata.VDI_saturation_dict.keys()),
                JASPER: interface.sigma_data_Jasper_Lange.index,
                VDI_PPDS: interface.sigma_data_VDI_PPDS_11.index,
                }

    def load_all_methods(self, load_data):
        r'''Method which picks out coefficients for the specified chemical
        from the various dictionaries and DataFrames storing it. All data is
        stored as attributes. This method also sets :obj:`Tmin`, :obj:`Tmax`,
        and :obj:`all_methods` as a set of methods for which the data exists for.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods = []
        self.all_methods = set()
        self.T_limits = T_limits = {}
        CASRN = self.CASRN
        if load_data and CASRN:
            if CASRN == '7732-18-5':
                methods.append(IAPWS)
                T_limits[IAPWS] = (273.15-25.0, iapws95_Tc)
            if CASRN in interface.sigma_data_Mulero_Cachadina.index:
                methods.append(STREFPROP)
                sigma0, n0, sigma1, n1, sigma2, n2, Tc, self.STREFPROP_Tmin, self.STREFPROP_Tmax = interface.sigma_values_Mulero_Cachadina[interface.sigma_data_Mulero_Cachadina.index.get_loc(CASRN)].tolist()
                self.STREFPROP_coeffs = [sigma0, n0, sigma1, n1, sigma2, n2, Tc]
                T_limits[STREFPROP] = (self.STREFPROP_Tmin, self.STREFPROP_Tmax)
            if CASRN in interface.sigma_data_Somayajulu2.index:
                methods.append(SOMAYAJULU2)
                self.SOMAYAJULU2_Tt, self.SOMAYAJULU2_Tc, A, B, C = interface.sigma_values_Somayajulu2[interface.sigma_data_Somayajulu2.index.get_loc(CASRN)].tolist()
                self.SOMAYAJULU2_coeffs = [A, B, C]
                T_limits[SOMAYAJULU2] = (self.SOMAYAJULU2_Tt, self.SOMAYAJULU2_Tc)
            if CASRN in interface.sigma_data_Somayajulu.index:
                methods.append(SOMAYAJULU)
                self.SOMAYAJULU_Tt, self.SOMAYAJULU_Tc, A, B, C = interface.sigma_values_Somayajulu[interface.sigma_data_Somayajulu.index.get_loc(CASRN)].tolist()
                self.SOMAYAJULU_coeffs = [A, B, C]
                T_limits[SOMAYAJULU] = (self.SOMAYAJULU_Tt, self.SOMAYAJULU_Tc)
            if CASRN in miscdata.VDI_saturation_dict:
                Ts, props = lookup_VDI_tabular_data(CASRN, 'sigma')
                # mercury missing values
                if Ts:
                    self.add_tabular_data(Ts, props, VDI_TABULAR, check_properties=False)
                    del self._method
            if CASRN in interface.sigma_data_Jasper_Lange.index:
                methods.append(JASPER)
                a, b, self.JASPER_Tmin, self.JASPER_Tmax = interface.sigma_values_Jasper_Lange[interface.sigma_data_Jasper_Lange.index.get_loc(CASRN)].tolist()
                if isnan(self.JASPER_Tmax) or self.JASPER_Tmax == self.JASPER_Tmin:
                    # Some data is missing; and some is on a above the limit basis
                    self.JASPER_Tmax = a/b + 273.15
                if isnan(self.JASPER_Tmin):
                    self.JASPER_Tmin = 0.0
                self.JASPER_coeffs = [a, b]
                T_limits[JASPER] = (self.JASPER_Tmin, self.JASPER_Tmax)
            if CASRN in interface.sigma_data_VDI_PPDS_11.index:
                Tm, Tc, A, B, C, D, E = interface.sigma_values_VDI_PPDS_11[interface.sigma_data_VDI_PPDS_11.index.get_loc(CASRN)].tolist()
                self.VDI_PPDS_coeffs = [A, B, C, D, E]
                self.VDI_PPDS_Tc = Tc
                self.VDI_PPDS_Tm = Tm
                methods.append(VDI_PPDS)
                T_limits[VDI_PPDS] = (self.VDI_PPDS_Tm, self.VDI_PPDS_Tc)
        if all((self.Tc, self.Vc, self.omega)):
            methods.append(MIQUEU)
            T_limits[MIQUEU] = (0.0, self.Tc)
        if all((self.Tb, self.Tc, self.Pc)):
            methods.append(BROCK_BIRD)
            methods.append(SASTRI_RAO)
            T_limits[BROCK_BIRD] = T_limits[SASTRI_RAO] = (0.0, self.Tc)
        if all((self.Tc, self.Pc, self.omega)):
            methods.append(PITZER)
            methods.append(ZUO_STENBY)
            T_limits[PITZER] = T_limits[ZUO_STENBY] = (1e-10, self.Tc)
        if all((self.Tb, self.Hvap_Tb, self.MW)):
            # Cache Cpl at Tb for ease of calculation of Tmax
            self.Cpl_Tb = self.Cpl(self.Tb) if hasattr(self.Cpl, '__call__') else self.Cpl
            if self.Cpl_Tb:
                self.Cpl_Tb = property_molar_to_mass(self.Cpl_Tb, self.MW)
                methods.append(ALEEM)
                # Tmin and Tmax for this method is known
                Tmax_possible = self.Tb + self.Hvap_Tb/self.Cpl_Tb
                # This method will ruin solve_property as it is typically valid
                # well above Tc. If Tc is available, limit it to that.
                if self.Tc:
                    Tmax_possible = min(self.Tc, Tmax_possible)
                T_limits[ALEEM] = (0.0, Tmax_possible)
        self.all_methods.update(methods)

    def calculate(self, T, method):
        r'''Method to calculate surface tension of a liquid at temperature `T`
        with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate surface tension, [K]
        method : str
            Name of the method to use

        Returns
        -------
        sigma : float
            Surface tension of the liquid at T, [N/m]
        '''
        if method == STREFPROP:
            sigma0, n0, sigma1, n1, sigma2, n2, Tc = self.STREFPROP_coeffs
            sigma = REFPROP_sigma(T, Tc=Tc, sigma0=sigma0, n0=n0, sigma1=sigma1, n1=n1,
                                  sigma2=sigma2, n2=n2)
        elif method == VDI_PPDS:
            sigma = EQ106(T, self.VDI_PPDS_Tc, *self.VDI_PPDS_coeffs)
        elif method == SOMAYAJULU2:
            A, B, C = self.SOMAYAJULU2_coeffs
            sigma = Somayajulu(T, Tc=self.SOMAYAJULU2_Tc, A=A, B=B, C=C)
        elif method == SOMAYAJULU:
            A, B, C = self.SOMAYAJULU_coeffs
            sigma = Somayajulu(T, Tc=self.SOMAYAJULU_Tc, A=A, B=B, C=C)
        elif method == JASPER:
            sigma = Jasper(T, a=self.JASPER_coeffs[0], b=self.JASPER_coeffs[1])
        elif method == IAPWS:
            sigma = sigma_IAPWS(T)
        elif method == BROCK_BIRD:
            sigma = Brock_Bird(T, self.Tb, self.Tc, self.Pc)
        elif method == SASTRI_RAO:
            sigma = Sastri_Rao(T, self.Tb, self.Tc, self.Pc)
        elif method == PITZER:
            sigma = Pitzer_sigma(T, self.Tc, self.Pc, self.omega)
        elif method == ZUO_STENBY:
            sigma = Zuo_Stenby(T, self.Tc, self.Pc, self.omega)
        elif method == MIQUEU:
            sigma = Miqueu(T, self.Tc, self.Vc, self.omega)
        elif method == ALEEM:
            Cpl = self.Cpl(T) if hasattr(self.Cpl, '__call__') else self.Cpl
            Cpl = property_molar_to_mass(Cpl, self.MW)
            try:
                Vml = self.Vml.T_dependent_property(T)
            except:
                try:
                    Vml = self.Vml(T)
                except:
                    Vml = self.Vml
            rhol = Vm_to_rho(Vml, self.MW)
            sigma = Aleem(T=T, MW=self.MW, Tb=self.Tb, rhol=rhol, Hvap_Tb=self.Hvap_Tb, Cpl=Cpl)
        else:
            return self._base_calculate(T, method)
        return sigma

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For CSP methods, the models
        are considered valid from 0 K to the critical point. For tabular data,
        extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the extrapolation
        is considered valid for all temperatures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        Parameters
        ----------
        T : float
            Temperature at which to test the method, [K]
        method : str
            Name of the method to test

        Returns
        -------
        validity : bool
            Whether or not a method is valid
        '''
        return super().test_method_validity(T, method)



WINTERFELDSCRIVENDAVIS = 'Winterfeld, Scriven, and Davis (1978)'
DIGUILIOTEJA = 'Diguilio and Teja (1988)'

surface_tension_mixture_methods = [WINTERFELDSCRIVENDAVIS, DIGUILIOTEJA, LINEAR]
"""Holds all methods available for the :obj:`SurfaceTensionMixture` class, for use in
iterating over them."""


class SurfaceTensionMixture(MixtureProperty):
    '''Class for dealing with surface tension of a mixture as a function of
    temperature, pressure, and composition.
    Consists of two mixing rules specific to surface tension, and mole
    weighted averaging.

    Prefered method is :obj:`Winterfeld_Scriven_Davis <chemicals.interface.Winterfeld_Scriven_Davis>` which requires mole
    fractions, pure component surface tensions, and the molar density of each
    pure component. :obj:`Diguilio_Teja <chemicals.interface.Diguilio_Teja>` is of similar accuracy, but requires
    the surface tensions of pure components at their boiling points, as well
    as boiling points and critical points and mole fractions. An ideal mixing
    rule based on mole fractions, **LINEAR**, is also available and is still
    relatively accurate.

    Parameters
    ----------
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    Tbs : list[float], optional
        Boiling points of all species in the mixture, [K]
    Tcs : list[float], optional
        Critical temperatures of all species in the mixture, [K]
    CASs : list[str], optional
        The CAS numbers of all species in the mixture, [-]
    SurfaceTensions : list[SurfaceTension], optional
        SurfaceTension objects created for all species in the mixture [-]
    VolumeLiquids : list[VolumeLiquid], optional
        VolumeLiquid objects created for all species in the mixture [-]
    correct_pressure_pure : bool, optional
        Whether to try to use the better pressure-corrected pure component
        models or to use only the T-only dependent pure species models, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`surface_tension_mixture_methods`.

    **WINTERFELDSCRIVENDAVIS**:
        Mixing rule described in :obj:`Winterfeld_Scriven_Davis <chemicals.interface.Winterfeld_Scriven_Davis>`.
    **DIGUILIOTEJA**:
        Mixing rule described in :obj:`Diguilio_Teja <chemicals.interface.Diguilio_Teja>`.
    **LINEAR**:
        Mixing rule described in :obj:`mixing_simple <chemicals.utils.mixing_simple>`.

    See Also
    --------
    chemicals.interface.Winterfeld_Scriven_Davis
    chemicals.interface.Diguilio_Teja

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''

    name = 'Surface tension'
    units = 'N/m'
    property_min = 0
    """Mimimum valid value of surface tension. This occurs at the critical
    point exactly."""
    property_max = 4.0
    """Maximum valid value of surface tension. Set to roughly twice that of
    cobalt at its melting point."""

    ranked_methods = [WINTERFELDSCRIVENDAVIS, DIGUILIOTEJA, LINEAR]

    pure_references = ('SurfaceTensions', 'VolumeLiquids')
    pure_reference_types = (SurfaceTension, VolumeLiquid)
    obj_references = ('SurfaceTensions', 'VolumeLiquids')

    pure_constants = ('MWs', 'Tbs', 'Tcs')
    custom_args = pure_constants

    def __init__(self, MWs=[], Tbs=[], Tcs=[], CASs=[], SurfaceTensions=[],
                 VolumeLiquids=[], correct_pressure_pure=False, **kwargs):
        self.MWs = MWs
        self.Tbs = Tbs
        self.Tcs = Tcs
        self.CASs = CASs
        self.SurfaceTensions = SurfaceTensions
        self.VolumeLiquids = VolumeLiquids
        super().__init__(correct_pressure_pure=correct_pressure_pure, **kwargs)

    def load_all_methods(self):
        r'''Method to initialize the object by precomputing any values which
        may be used repeatedly and by retrieving mixture-specific variables.
        All data are stored as attributes. This method also sets :obj:`Tmin`,
        :obj:`Tmax`, and :obj:`all_methods` as a set of methods which should
        work to calculate the property.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods = set()
        methods.add(LINEAR) # Needs sigma
        methods.add(WINTERFELDSCRIVENDAVIS) # Nothing to load, needs rhoms, sigma
        if none_and_length_check([self.Tbs, self.Tcs]):
            self.sigmas_Tb = [i(Tb) for i, Tb in zip(self.SurfaceTensions, self.Tbs)]
            if none_and_length_check([self.sigmas_Tb]):
                methods.add(DIGUILIOTEJA)
        self.all_methods = methods

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate surface tension of a liquid mixture at
        temperature `T`, pressure `P`, mole fractions `zs` and weight fractions
        `ws` with a given method.

        This method has no exception handling; see :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Name of the method to use

        Returns
        -------
        sigma : float
            Surface tension of the liquid at given conditions, [N/m]
        '''
        if method == DIGUILIOTEJA:
            return Diguilio_Teja(T=T, xs=zs, sigmas_Tb=self.sigmas_Tb,
                                 Tbs=self.Tbs, Tcs=self.Tcs)
        elif method == WINTERFELDSCRIVENDAVIS:
            sigmas = self.calculate_pures(T)
            Vms = self.calculate_pures_corrected(T, P, fallback=False, objs=self.VolumeLiquids)
            rhoms = [1.0/v for v in Vms]
            return Winterfeld_Scriven_Davis(zs, sigmas, rhoms)
        return super().calculate(T, P, zs, ws, method)


    def test_method_validity(self, T, P, zs, ws, method):
        # LINEAR and WINTERFELDSCRIVENDAVIS need to calculate sigma for pure
        # species - doesn't work above Tc for any compound.
        # DIGUILIOTEJA needs Tcs, not sure.
        if method in [LINEAR, DIGUILIOTEJA, WINTERFELDSCRIVENDAVIS]:
            return True
        return super().test_method_validity(T, P, zs, ws, method)


'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020, 2021, 2022 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
__all__ = ['CEOSLiquid', 'CEOSGas']
from fluids.numerics import numpy as np
from fluids.numerics import trunc_exp, trunc_exp_numpy

from thermo.eos_mix import IGMIX, eos_mix_full_path_dict, eos_mix_full_path_reverse_dict
from thermo.heat_capacity import HeatCapacityGas
from thermo.phases.phase import IdealGasDeparturePhase
from thermo.phases.phase_utils import lnphis_direct

try:
    zeros, ndarray, full, array = np.zeros, np.ndarray, np.full, np.array
except:
    pass

class CEOSPhase(IdealGasDeparturePhase):
    r'''Class for representing a cubic equation of state gas phase
    as a phase object. All departure
    properties are actually calculated by the code in :obj:`thermo.eos` and
    :obj:`thermo.eos_mix`.

    .. math::
        P=\frac{RT}{V-b}-\frac{a\alpha(T)}{V^2 + \delta V + \epsilon}

    Parameters
    ----------
    eos_class : GCEOSMIX
        EOS class, [-]
    eos_kwargs : dict
        Parameters to be passed to the created EOS, [-]
    HeatCapacityGases : list[HeatCapacityGas]
        Objects proiding pure-component heat capacity correlations, [-]
    Hfs : list[float]
        Molar ideal-gas standard heats of formation at 298.15 K and 1 atm,
        [J/mol]
    Gfs : list[float]
        Molar ideal-gas standard Gibbs energies of formation at 298.15 K and
        1 atm, [J/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    zs : list[float], optional
        Mole fractions of each component, [-]

    Examples
    --------
    T-P initialization for oxygen and nitrogen with the PR EOS, using Poling's
    polynomial heat capacities:

    >>> from scipy.constants import R
    >>> from thermo import HeatCapacityGas, PRMIX, CEOSGas
    >>> eos_kwargs = dict(Tcs=[154.58, 126.2], Pcs=[5042945.25, 3394387.5], omegas=[0.021, 0.04], kijs=[[0.0, -0.0159], [-0.0159, 0.0]])
    >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
    ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
    >>> phase = CEOSGas(eos_class=PRMIX, eos_kwargs=eos_kwargs, T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
    >>> phase.Cp()
    29.2285050

    '''

    __slots__ = ('eos_class', 'eos_kwargs', 'vectorized', 'HeatCapacityGases', 'N',
    'Hfs', 'Gfs', 'Sfs', 'Cpgs_poly_fit', '_Cpgs_data', 'composition_independent',
     'eos_mix', 'T', 'P', 'zs', '_model_hash_ignore_phase', '_model_hash')
    ideal_gas_basis = True

    pure_references = ('HeatCapacityGases',)
    pure_reference_types = (HeatCapacityGas,)
    obj_references = ('eos_mix', 'result', 'constants', 'correlations', 'HeatCapacityGases')


    pointer_references = ('eos_class',)
    pointer_reference_dicts = (eos_mix_full_path_dict,)
    """Tuple of dictionaries for string -> object
    """
    reference_pointer_dicts = (eos_mix_full_path_reverse_dict,)

    model_attributes = ('Hfs', 'Gfs', 'Sfs', 'eos_class',
                        'eos_kwargs') + pure_references

    def _custom_as_json(self, d, cache):
        d['eos_class'] = d['eos_class'].__full_path__

    def _custom_from_json(self, *args):
        self.eos_class = eos_mix_full_path_dict[self.eos_class]

    def __repr__(self):
        r'''Method to create a string representation of the phase object, with
        the goal of making it easy to obtain standalone code which reproduces
        the current state of the phase. This is extremely helpful in creating
        new test cases.

        Returns
        -------
        recreation : str
            String which is valid Python and recreates the current state of
            the object if ran, [-]

        '''
        eos_kwargs = str(self.eos_kwargs).replace("'", '"')
        try:
            Cpgs = ', '.join(str(o) for o in self.HeatCapacityGases)
        except:
            Cpgs = ''
        base = f'{self.__class__.__name__}(eos_class={self.eos_class.__name__}, eos_kwargs={eos_kwargs}, HeatCapacityGases=[{Cpgs}], '
        for s in ('Hfs', 'Gfs', 'Sfs', 'T', 'P', 'zs'):
            if hasattr(self, s) and getattr(self, s) is not None:
                base += f'{s}={getattr(self, s)}, '
        if base[-2:] == ', ':
            base = base[:-2]
        base += ')'
        return base

    def __init__(self, eos_class, eos_kwargs, HeatCapacityGases=None, Hfs=None,
                 Gfs=None, Sfs=None, T=IdealGasDeparturePhase.T_DEFAULT, P=IdealGasDeparturePhase.P_DEFAULT, zs=None):
        self.eos_class = eos_class
        self.eos_kwargs = eos_kwargs
        self.vectorized = vectorized = (
            any(type(v) is ndarray for v in eos_kwargs.values())
            or any(type(v) is ndarray for v in (zs, Hfs, Gfs, Sfs))
        )
        self.HeatCapacityGases = HeatCapacityGases
        if HeatCapacityGases is not None:
            self.N = N = len(HeatCapacityGases)
            for obj in HeatCapacityGases:
                if not isinstance(obj, HeatCapacityGas):
                    raise ValueError("A HeatCapacityGas object is required")
        elif 'Tcs' in eos_kwargs:
            self.N = N = len(eos_kwargs['Tcs'])
        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs
        self.Cpgs_poly_fit, self._Cpgs_data = self._setup_Cpigs(HeatCapacityGases)
        self.composition_independent = eos_class is IGMIX
        if T is None: T = 298.15
        if P is None: P = 101325.0
        if zs is None:
            if vectorized:
                v = 1.0 / N
                zs = full(N, v)
            else:
                zs = [1.0 / N] * N
        self.T = T
        self.P = P
        self.zs = zs
        self.eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)

    def to_TP_zs(self, T, P, zs, other_eos=None):
        r'''Method to create a new Phase object with the same constants as the
        existing Phase but at a different `T` and `P`. This method has a
        special parameter `other_eos`.

        This is added to allow a gas-type phase to be created from
        a liquid-type phase at the same conditions (and vice-versa),
        as :obj:`GCEOSMIX <thermo.eos_mix.GCEOSMIX>` objects were designed to
        have vapor and liquid properties in the same phase. This argument is
        mostly for internal use.

        Parameters
        ----------
        zs : list[float]
            Molar composition of the new phase, [-]
        T : float
            Temperature of the new phase, [K]
        P : float
            Pressure of the new phase, [Pa]
        other_eos : obj:`GCEOSMIX <thermo.eos_mix.GCEOSMIX>` object
            Other equation of state object at the same conditions, [-]

        Returns
        -------
        new_phase : Phase
            New phase at the specified conditions, [-]

        Notes
        -----
        This method is marginally faster than :obj:`Phase.to` as it does not
        need to check what the inputs are.

        Examples
        --------
        >>> from thermo.eos_mix import PRMIX
        >>> eos_kwargs = dict(Tcs=[305.32, 369.83], Pcs=[4872000.0, 4248000.0], omegas=[0.098, 0.152])
        >>> gas = CEOSGas(PRMIX, T=300.0, P=1e6, zs=[.2, .8], eos_kwargs=eos_kwargs)
        >>> liquid = CEOSLiquid(PRMIX, T=500.0, P=1e7, zs=[.3, .7], eos_kwargs=eos_kwargs)
        >>> new_liq = liquid.to_TP_zs(T=gas.T, P=gas.P, zs=gas.zs, other_eos=gas.eos_mix)
        >>> new_liq
        CEOSLiquid(eos_class=PRMIX, eos_kwargs={"Tcs": [305.32, 369.83], "Pcs": [4872000.0, 4248000.0], "omegas": [0.098, 0.152]}, HeatCapacityGases=[], T=300.0, P=1000000.0, zs=[0.2, 0.8])
        >>> new_liq.eos_mix is gas.eos_mix
        True
        '''
        # Why so slow
        new = self.__class__.__new__(self.__class__)
        new.vectorized = self.vectorized
        if other_eos is not None:
            other_eos.solve_missing_volumes()
            new.eos_mix = other_eos
        else:
            try:
                new.eos_mix = self.eos_mix.to_TP_zs_fast(T=T, P=P, zs=zs, only_g=self.is_gas, only_l=self.is_liquid,
                                                         full_alphas=True) # optimize alphas?
                                                         # Be very careful doing this in the future - wasted
                                                         # 1 hour on this because the heat capacity calculation was wrong
            except AttributeError:
                new.eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
        new.eos_class = self.eos_class
        new.eos_kwargs = self.eos_kwargs
        new.HeatCapacityGases = self.HeatCapacityGases
        new._Cpgs_data = self._Cpgs_data
        new.Cpgs_poly_fit = self.Cpgs_poly_fit
        new.composition_independent = self.composition_independent
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        new.T = T
        new.P = P
        new.zs = zs
        if new.composition_independent:
            new.force_phase = 'g'

        try:
            new.N = self.N
        except:
            pass

        return new


    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        # temporary TODO remove this statement
        if self.vectorized and type(zs) is not ndarray:
            zs = array(zs)

        if T is not None:
            if P is not None:
                try:
                    new.eos_mix = self.eos_mix.to_TP_zs_fast(T=T, P=P, zs=zs, only_g=self.is_gas, only_l=self.is_liquid,
                                                             full_alphas=True)
                except AttributeError:
                    new.eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
            elif V is not None:
                try:
                    new.eos_mix = self.eos_mix.to(T=T, V=V, zs=zs, fugacities=False)
                except AttributeError:
                    new.eos_mix = self.eos_class(T=T, V=V, zs=zs, **self.eos_kwargs)
                P = new.eos_mix.P
        elif P is not None and V is not None:
            try:
                new.eos_mix = self.eos_mix.to_PV_zs(P=P, V=V, zs=zs, only_g=self.is_gas, only_l=self.is_liquid, fugacities=False)
            except AttributeError:
                new.eos_mix = self.eos_class(P=P, V=V, zs=zs, only_g=self.is_gas, only_l=self.is_liquid, **self.eos_kwargs)
            T = new.eos_mix.T
        else:
            raise ValueError("Two of T, P, or V are needed")
        new.eos_class, new.eos_kwargs = self.eos_class, self.eos_kwargs
        new.HeatCapacityGases, new._Cpgs_data, new.Cpgs_poly_fit = self.HeatCapacityGases, self._Cpgs_data, self.Cpgs_poly_fit
        new.composition_independent, new.vectorized = self.composition_independent, self.vectorized
        new.Hfs, new.Gfs, new.Sfs = self.Hfs, self.Gfs, self.Sfs
        new.T = T
        new.P = P
        new.zs = zs
        if new.composition_independent:
            new.force_phase = 'g'
        try:
            new.N = self.N
        except:
            pass

        return new
    def lnphis_G_min(self):
        eos_mix = self.eos_mix
        if eos_mix.phase == 'l/g':
            # Check both phases are solved, and complete if not
            eos_mix.solve_missing_volumes()
            if eos_mix.G_dep_l < eos_mix.G_dep_g:
                return eos_mix.fugacity_coefficients(eos_mix.Z_l)
            return eos_mix.fugacity_coefficients(eos_mix.Z_g)
        try:
            return eos_mix.fugacity_coefficients(eos_mix.Z_g)
        except AttributeError:
            return eos_mix.fugacity_coefficients(eos_mix.Z_l)

    supports_lnphis_args = True

    def lnphis_args(self, most_stable=False):
        # VTPR, PSRK, anything with GE not yet supported
        # Could save time by allowing T, P as an argument, and getting a new eos_mix at that
        N = self.N
        eos_mix = self.eos_mix
        if self.vectorized:
            a_alpha_j_rows, vec0, lnphis = zeros(N), zeros(N), zeros(N)
        else:
            a_alpha_j_rows, vec0, lnphis = [0.0]*N, [0.0]*N, [0.0]*N
        l, g = (self.is_liquid, self.is_gas) #if not most_stable else (True, True)
        if eos_mix.translated:
            return (self.eos_class.model_id, self.T, self.P, self.N, eos_mix.one_minus_kijs, l, g,
                   eos_mix.b0s, eos_mix.bs, eos_mix.cs, eos_mix.a_alphas, eos_mix.a_alpha_roots, a_alpha_j_rows, vec0, lnphis)
        else:
            return (self.eos_class.model_id, self.T, self.P, self.N, eos_mix.one_minus_kijs, l, g,
                   eos_mix.bs, eos_mix.a_alphas, eos_mix.a_alpha_roots, a_alpha_j_rows, vec0, lnphis)

    def lnphis_at_zs(self, zs, most_stable=False):
        # eos_mix = self.eos_mix
        # if eos_mix.__class__.__name__ in ('PRMIX', 'VDWMIX', 'SRKMIX', 'RKMIX'):
        return lnphis_direct(zs, *self.lnphis_args(most_stable))
        # return self.to_TP_zs(self.T, self.P, zs).lnphis()

    def fugacities_at_zs(self, zs, most_stable=False):
        P = self.P
        lnphis = lnphis_direct(zs, *self.lnphis_args(most_stable))
        if self.vectorized:
            return trunc_exp_numpy(lnphis)*P*zs
        else:
            return [P*zs[i]*trunc_exp(lnphis[i]) for i in range(len(zs))]

    def T_max_at_V(self, V):
        T_max = self.eos_mix.T_max_at_V(V)
        if T_max is not None:
            T_max = T_max*(1.0-1e-12)
        return T_max

    def P_max_at_V(self, V):
        P_max = self.eos_mix.P_max_at_V(V)
        if P_max is not None:
            P_max = P_max*(1.0-1e-12)
        return P_max

    def mu(self):
        try:
            return self._mu
        except AttributeError:
            pass
        try:
            phase = self.assigned_phase
        except:
            phase = self.phase
            if phase == 'l/g': phase = 'g'
        try:
            ws = self._ws
        except:
            ws = self.ws()
        if phase == 'g':
            mu = self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, ws)
        else:
            mu = self.correlations.ViscosityLiquidMixture.mixture_property(self.T, self.P, self.zs, ws)
        self._mu = mu
        return mu

    def k(self):
        try:
            return self._k
        except AttributeError:
            pass
        try:
            phase = self.assigned_phase
        except:
            phase = self.phase
            if phase == 'l/g': phase = 'g'
        if phase == 'g':
            k = self.correlations.ThermalConductivityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        elif phase == 'l':
            k = self.correlations.ThermalConductivityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._k = k
        return k

    def _set_mechanical_critical_point(self):
        new = self.eos_mix.to_mechanical_critical_point()
        self._mechanical_critical_T = new.T
        self._mechanical_critical_P = new.P
        try:
            V = new.V_l
        except:
            V = new.V_g
        self._mechanical_critical_V = V
        return new.T, new.P, V

    def P_transitions(self):
        e = self.eos_mix
        return e.P_discriminant_zeros_analytical(e.T, e.b, e.delta, e.epsilon, e.a_alpha, valid=True)
        # EOS is guaranteed to be at correct temperature
        try:
            return [self.eos_mix.P_discriminant_zero_l()]
        except:
            return [self.eos_mix.P_discriminant_zero_g()]

    def T_transitions(self):
        try:
            return [self.eos_mix.T_discriminant_zero_l()]
        except:
            return [self.eos_mix.T_discriminant_zero_g()]

    def d2P_dTdV(self):
        r'''Method to calculate and return the second derivative of
        pressure with respect to temperature and volume of the phase.

        .. math::
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) = - \frac{
            R}{\left(V - b\right)^{2}} + \frac{a \left(2 V + \delta\right)
            \frac{d \alpha{\left (T \right )}}{d T}}{\left(V^{2} + V \delta
            + \epsilon\right)^{2}}

        Returns
        -------
        d2P_dTdV : float
            Second volume derivative of pressure, [mol*Pa^2/(J*K)]
        '''

    def lnphis(self):
        r'''Method to calculate and return the log of fugacity coefficients of
        each component in the phase. The calculation is performed by
        :obj:`thermo.eos_mix.GCEOSMIX.fugacity_coefficients` or a simpler formula in the case
        of most specific models.

        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
        '''
    def dlnphis_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        the log of fugacity coefficients of
        each component in the phase. The calculation is performed by
        :obj:`thermo.eos_mix.GCEOSMIX.dlnphis_dT` or a simpler formula in the
        case of most specific models.

        Returns
        -------
        dlnphis_dT : list[float]
            First temperature derivative of log fugacity coefficients, [1/K]
        '''

    def dlnphis_dP(self):
        r'''Method to calculate and return the first pressure derivative of
        the log of fugacity coefficients of
        each component in the phase. The calculation is performed by
        :obj:`thermo.eos_mix.GCEOSMIX.dlnphis_dP` or a simpler formula in the
        case of most specific models.

        Returns
        -------
        dlnphis_dP : list[float]
            First pressure derivative of log fugacity coefficients, [1/Pa]
        '''

    def V(self):
        r'''Method to calculate and return the molar volume of the phase.

        Returns
        -------
        V : float
            Molar volume, [m^3/mol]
        '''

    def dP_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        pressure of the phase.

        .. math::
            \left(\frac{\partial P}{\partial T}\right)_V = \frac{R}{V - b}
            - \frac{a \frac{d \alpha{\left (T \right )}}{d T}}{V^{2} + V \delta
            + \epsilon}

        Returns
        -------
        dP_dT : float
            First temperature derivative of pressure, [Pa/K]
        '''


    def dP_dV(self):
        r'''Method to calculate and return the first volume derivative of
        pressure of the phase.

        .. math::
            \left(\frac{\partial P}{\partial V}\right)_T = - \frac{R T}{\left(
            V - b\right)^{2}} - \frac{a \left(- 2 V - \delta\right) \alpha{
            \left (T \right )}}{\left(V^{2} + V \delta + \epsilon\right)^{2}}

        Returns
        -------
        dP_dV : float
            First volume derivative of pressure, [Pa*mol/m^3]
        '''


    def d2P_dT2(self):
        r'''Method to calculate and return the second temperature derivative of
        pressure of the phase.

        .. math::
            \left(\frac{\partial^2  P}{\partial T^2}\right)_V =  - \frac{a
            \frac{d^{2} \alpha{\left (T \right )}}{d T^{2}}}{V^{2} + V \delta
            + \epsilon}

        Returns
        -------
        d2P_dT2 : float
            Second temperature derivative of pressure, [Pa/K^2]
        '''

    def d2P_dV2(self):
        r'''Method to calculate and return the second volume derivative of
        pressure of the phase.

        .. math::
            \left(\frac{\partial^2  P}{\partial V^2}\right)_T = 2 \left(\frac{
            R T}{\left(V - b\right)^{3}} - \frac{a \left(2 V + \delta\right)^{
            2} \alpha{\left (T \right )}}{\left(V^{2} + V \delta + \epsilon
            \right)^{3}} + \frac{a \alpha{\left (T \right )}}{\left(V^{2} + V
            \delta + \epsilon\right)^{2}}\right)

        Returns
        -------
        d2P_dV2 : float
            Second volume derivative of pressure, [Pa*mol^2/m^6]
        '''


    def lnphis_lowest_Gibbs(self):
        try:
            return self._lnphis_lowest_Gibbs
        except:
            pass
        eos_mix = self.eos_mix
        # A bad bug was discovered where the other root just wasn't solved for
        # however, more tests work it isn't solved for. Highly odd.
        # eos_mix.solve_missing_volumes()
        try:
            if eos_mix.G_dep_g < eos_mix.G_dep_l:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_g)
            else:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_l)
        except:
            # Do not have both phases, only one will work,
            # order of attempt does not matter
            try:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_g)
            except:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_l)

        self._lnphis_lowest_Gibbs = lnphis
        return lnphis

    def fugacities_lowest_Gibbs(self):
        P = self.P
        zs = self.zs
        lnphis = self.lnphis_lowest_Gibbs()
        if self.vectorized:
            return trunc_exp_numpy(lnphis)*P*zs
        else:
            return [P*zs[i]*trunc_exp(lnphis[i]) for i in range(len(zs))]

    def V_iter(self, force=False):
        # Can be some severe issues in the very low pressure/temperature range
        # For that reason, consider not doing TV iterations.
        # Cal occur also with PV iterations

        T, P = self.T, self.P
#        if 0 and ((P < 1.0 or T < 1.0) or (P/T < 500.0 and T < 50.0)):
        eos_mix = self.eos_mix
        V = self.V()
        P_err = abs((self.R*T/(V-eos_mix.b) - eos_mix.a_alpha/(V*V + eos_mix.delta*V + eos_mix.epsilon)) - P)
        if (P_err/P) < 1e-9 and not force:
            return V
        if self.phase == 'g':
            try:
                return eos_mix.V_g_mpmath.real
            except:
                return eos_mix.V_l_mpmath.real
        else:
            try:
                return eos_mix.V_l_mpmath.real
            except:
                return eos_mix.V_g_mpmath.real

class CEOSGas(CEOSPhase):
    is_gas = True
    is_liquid = False
    __slots__ = ()

    @property
    def phase(self):
        phase = self.eos_mix.phase
        if phase in ('l', 'g'):
            return phase
        return 'g'

    def PIP(self):
        try:
            return self.eos_mix.PIP_g
        except AttributeError:
            return self.eos_mix.PIP_l

    def lnphis(self):
        try:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_g)
        except AttributeError:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_l)

    def dlnphis_dT(self):
        try:
            return self.eos_mix.dlnphis_dT('g')
        except:
            return self.eos_mix.dlnphis_dT('l')

    def dlnphis_dP(self):
        try:
            return self.eos_mix.dlnphis_dP('g')
        except:
            return self.eos_mix.dlnphis_dP('l')

    def dlnphis_dns(self):
        eos_mix = self.eos_mix
        try:
            return eos_mix.dlnphis_dns(eos_mix.Z_g)
        except:
            return eos_mix.dlnphis_dns(eos_mix.Z_l)

    def dlnphis_dzs(self):
        eos_mix = self.eos_mix
        try:
            return eos_mix.dlnphis_dzs(eos_mix.Z_g)
        except:
            return eos_mix.dlnphis_dzs(eos_mix.Z_l)

    def phi_pures(self):
        try:
            return self._phi_pures
        except:
            pass
        phis_pure = zeros(self.N) if self.vectorized else [0.0]*self.N
        for i, o in enumerate(self.eos_mix.pures()):
            try:
                phis_pure[i] = o.phi_g
            except AttributeError:
                phis_pure[i] = o.phi_l
        self._phi_pures = phis_pure
        return phis_pure


    def H_dep(self):
        try:
            return self.eos_mix.H_dep_g
        except AttributeError:
            return self.eos_mix.H_dep_l

    def S_dep(self):
        try:
            return self.eos_mix.S_dep_g
        except AttributeError:
            return self.eos_mix.S_dep_l

    def G_dep(self):
        try:
            return self.eos_mix.G_dep_g
        except AttributeError:
            return self.eos_mix.G_dep_l

    def Cp_dep(self):
        try:
            return self.eos_mix.Cp_dep_g
        except AttributeError:
            return self.eos_mix.Cp_dep_l

    def dS_dep_dT(self):
        try:
            return self.eos_mix.dS_dep_dT_g
        except AttributeError:
            return self.eos_mix.dS_dep_dT_l

    def dS_dep_dP_V(self):
        try:
            dS_dP_V = self.eos_mix.dS_dep_dP_g_V
        except AttributeError:
            dS_dP_V = self.eos_mix.dS_dep_dP_l_V
        return dS_dP_V

    def dS_dep_dP_T(self):
        try:
            return self.eos_mix.dS_dep_dP_g
        except AttributeError:
            return self.eos_mix.dS_dep_dP_l

    def dS_dep_dT_V(self):
        try:
            return self.eos_mix.dS_dep_dT_g_V
        except AttributeError:
            return self.eos_mix.dS_dep_dT_l_V

    def dH_dep_dP_V(self):
        try:
            return self.eos_mix.dH_dep_dP_g_V
        except AttributeError:
            return self.eos_mix.dH_dep_dP_l_V

    def dH_dep_dP_T(self):
        try:
            return self.eos_mix.dH_dep_dP_g
        except AttributeError:
            return self.eos_mix.dH_dep_dP_l

    def dH_dep_dV_T(self):
        try:
            return self.eos_mix.dH_dep_dV_g_T
        except AttributeError:
            return self.eos_mix.dH_dep_dV_l_T

    def dH_dep_dV_P(self):
        try:
            return self.eos_mix.dH_dep_dV_g_P
        except AttributeError:
            return self.eos_mix.dH_dep_dV_l_P

    def V(self):
        try:
            return self.eos_mix.V_g
        except AttributeError:
            return self.eos_mix.V_l

    def dP_dT(self):
        try:
            return self.eos_mix.dP_dT_g
        except AttributeError:
            return self.eos_mix.dP_dT_l

    dP_dT_V = dP_dT

    def dP_dV(self):
        try:
            return self.eos_mix.dP_dV_g
        except AttributeError:
            return self.eos_mix.dP_dV_l

    dP_dV_T = dP_dV

    def d2P_dT2(self):
        try:
            return self.eos_mix.d2P_dT2_g
        except AttributeError:
            return self.eos_mix.d2P_dT2_l

    d2P_dT2_V = d2P_dT2

    def d2P_dV2(self):
        try:
            return self.eos_mix.d2P_dV2_g
        except AttributeError:
            return self.eos_mix.d2P_dV2_l

    d2P_dV2_T = d2P_dV2

    def d2P_dTdV(self):
        try:
            return self.eos_mix.d2P_dTdV_g
        except AttributeError:
            return self.eos_mix.d2P_dTdV_l

    # The following methods are implemented to provide numerically precise answers
    # for the ideal gas equation of state only, the rest of the EOSs are fine without
    # these methods
    def d2T_dV2(self):
        try:
            return self.eos_mix.d2T_dV2_g
        except AttributeError:
            return self.eos_mix.d2T_dV2_l

    d2T_dV2_P = d2T_dV2

    def d2V_dT2(self):
        try:
            return self.eos_mix.d2V_dT2_g
        except AttributeError:
            return self.eos_mix.d2V_dT2_l

    d2V_dT2_P = d2V_dT2

    def dV_dzs(self):
        eos_mix = self.eos_mix
        try:
            dV_dzs = self.eos_mix.dV_dzs(eos_mix.Z_g)
        except AttributeError:
            dV_dzs = self.eos_mix.dV_dzs(eos_mix.Z_l)
        return dV_dzs

    def dV_dns(self):
        eos_mix = self.eos_mix
        try:
            dV_dns = self.eos_mix.dV_dns(eos_mix.Z_g)
        except AttributeError:
            dV_dns = self.eos_mix.dV_dns(eos_mix.Z_l)
        return dV_dns

    def dnV_dns(self):
        eos_mix = self.eos_mix
        try:
            dnV_dns = self.eos_mix.dnV_dns(eos_mix.Z_g)
        except AttributeError:
            dnV_dns = self.eos_mix.dnV_dns(eos_mix.Z_l)
        return dnV_dns

    def d2H_dep_dT2(self):
        try:
            return self.eos_mix.d2H_dep_dT2_g
        except AttributeError:
            return self.eos_mix.d2H_dep_dT2_l

    def d2H_dep_dT2_V(self):
        try:
            return self.eos_mix.d2H_dep_dT2_g_V
        except AttributeError:
            return self.eos_mix.d2H_dep_dT2_l_V

    def d2H_dTdP(self):
        try:
            return self.eos_mix.d2H_dep_dTdP_g
        except AttributeError:
            return self.eos_mix.d2H_dep_dTdP_l

    def dH_dep_dT_V(self):
        try:
            return self.eos_mix.dH_dep_dT_g_V
        except:
            return self.eos_mix.dH_dep_dT_l_V

    def dH_dep_dzs(self):
        try:
            return self.eos_mix.dH_dep_dzs(self.eos_mix.Z_g)
        except AttributeError:
            return self.eos_mix.dH_dep_dzs(self.eos_mix.Z_l)


    def d2P_dTdP(self):
        try:
            return self.eos_mix.d2P_dTdP_g
        except AttributeError:
            return self.eos_mix.d2P_dTdP_l

    def d2P_dVdP(self):
        try:
            return self.eos_mix.d2P_dVdP_g
        except AttributeError:
            return self.eos_mix.d2P_dVdP_l

    def d2P_dVdT_TP(self):
        try:
            return self.eos_mix.d2P_dVdT_TP_g
        except AttributeError:
            return self.eos_mix.d2P_dVdT_TP_l

    def d2P_dT2_PV(self):
        try:
            return self.eos_mix.d2P_dT2_PV_g
        except AttributeError:
            return self.eos_mix.d2P_dT2_PV_l

    def dS_dep_dzs(self):
        try:
            return self.eos_mix.dS_dep_dzs(self.eos_mix.Z_g)
        except AttributeError:
            return self.eos_mix.dS_dep_dzs(self.eos_mix.Z_l)


class CEOSLiquid(CEOSPhase):
    is_gas = False
    is_liquid = True
    __slots__ = ()

    @property
    def phase(self):
        phase = self.eos_mix.phase
        if phase in ('g', 'l'):
            return phase
        return 'l'

    def PIP(self):
        try:
            return self.eos_mix.PIP_l
        except AttributeError:
            return self.eos_mix.PIP_g

    def lnphis(self):
        try:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_l)
        except AttributeError:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_g)

    def dlnphis_dT(self):
        try:
            return self.eos_mix.dlnphis_dT('l')
        except:
            return self.eos_mix.dlnphis_dT('g')

    def dlnphis_dP(self):
        try:
            return self.eos_mix.dlnphis_dP('l')
        except:
            return self.eos_mix.dlnphis_dP('g')

    def dlnphis_dns(self):
        eos_mix = self.eos_mix
        try:
            return eos_mix.dlnphis_dns(eos_mix.Z_l)
        except:
            return eos_mix.dlnphis_dns(eos_mix.Z_g)

    def dlnphis_dzs(self):
        eos_mix = self.eos_mix
        try:
            return eos_mix.dlnphis_dzs(eos_mix.Z_l)
        except:
            return eos_mix.dlnphis_dzs(eos_mix.Z_g)

    def phi_pures(self):
        try:
            return self._phi_pures
        except:
            pass
        phis_pure = zeros(self.N) if self.vectorized else [0.0]*self.N
        for i, o in enumerate(self.eos_mix.pures()):
            try:
                phis_pure[i] = (o.phi_l)
            except AttributeError:
                phis_pure[i] = (o.phi_g)
        self._phi_pures = phis_pure
        return phis_pure


    def H_dep(self):
        try:
            return self.eos_mix.H_dep_l
        except AttributeError:
            return self.eos_mix.H_dep_g

    def S_dep(self):
        try:
            return self.eos_mix.S_dep_l
        except AttributeError:
            return self.eos_mix.S_dep_g

    def G_dep(self):
        try:
            return self.eos_mix.G_dep_l
        except AttributeError:
            return self.eos_mix.G_dep_g

    def Cp_dep(self):
        try:
            return self.eos_mix.Cp_dep_l
        except AttributeError:
            return self.eos_mix.Cp_dep_g

    def dS_dep_dT(self):
        try:
            return self.eos_mix.dS_dep_dT_l
        except AttributeError:
            return self.eos_mix.dS_dep_dT_g

    def dS_dep_dP_V(self):
        try:
            dS_dP_V = self.eos_mix.dS_dep_dP_l_V
        except AttributeError:
            dS_dP_V = self.eos_mix.dS_dep_dP_g_V
        return dS_dP_V

    def dS_dep_dP_T(self):
        try:
            return self.eos_mix.dS_dep_dP_l
        except AttributeError:
            return self.eos_mix.dS_dep_dP_g

    def dS_dep_dT_V(self):
        try:
            return self.eos_mix.dS_dep_dT_l_V
        except AttributeError:
            return self.eos_mix.dS_dep_dT_g_V

    def dH_dep_dP_V(self):
        try:
            return self.eos_mix.dH_dep_dP_l_V
        except AttributeError:
            return self.eos_mix.dH_dep_dP_g_V

    def dH_dep_dP_T(self):
        try:
            return self.eos_mix.dH_dep_dP_l
        except AttributeError:
            return self.eos_mix.dH_dep_dP_g

    def dH_dep_dV_T(self):
        try:
            return self.eos_mix.dH_dep_dV_l_T
        except AttributeError:
            return self.eos_mix.dH_dep_dV_g_T

    def dH_dep_dV_P(self):
        try:
            return self.eos_mix.dH_dep_dV_l_P
        except AttributeError:
            return self.eos_mix.dH_dep_dV_g_P

    def V(self):
        try:
            return self.eos_mix.V_l
        except AttributeError:
            return self.eos_mix.V_g

    def dP_dT(self):
        try:
            return self.eos_mix.dP_dT_l
        except AttributeError:
            return self.eos_mix.dP_dT_g

    dP_dT_V = dP_dT

    def dP_dV(self):
        try:
            return self.eos_mix.dP_dV_l
        except AttributeError:
            return self.eos_mix.dP_dV_g

    dP_dV_T = dP_dV

    def d2P_dT2(self):
        try:
            return self.eos_mix.d2P_dT2_l
        except AttributeError:
            return self.eos_mix.d2P_dT2_g

    d2P_dT2_V = d2P_dT2

    def d2P_dV2(self):
        try:
            return self.eos_mix.d2P_dV2_l
        except AttributeError:
            return self.eos_mix.d2P_dV2_g

    d2P_dV2_T = d2P_dV2

    def d2P_dTdV(self):
        try:
            return self.eos_mix.d2P_dTdV_l
        except AttributeError:
            return self.eos_mix.d2P_dTdV_g

    # The following methods are implemented to provide numerically precise answers
    # for the ideal gas equation of state only, the rest of the EOSs are fine without
    # these methods
    def d2T_dV2(self):
        try:
            return self.eos_mix.d2T_dV2_l
        except AttributeError:
            return self.eos_mix.d2T_dV2_g

    d2T_dV2_P = d2T_dV2

    def d2V_dT2(self):
        try:
            return self.eos_mix.d2V_dT2_l
        except AttributeError:
            return self.eos_mix.d2V_dT2_g

    d2V_dT2_P = d2V_dT2

    def dV_dzs(self):
        eos_mix = self.eos_mix
        try:
            dV_dzs = self.eos_mix.dV_dzs(eos_mix.Z_l)
        except AttributeError:
            dV_dzs = self.eos_mix.dV_dzs(eos_mix.Z_g)
        return dV_dzs

    def dV_dns(self):
        eos_mix = self.eos_mix
        try:
            dV_dns = self.eos_mix.dV_dns(eos_mix.Z_l)
        except AttributeError:
            dV_dns = self.eos_mix.dV_dns(eos_mix.Z_g)
        return dV_dns

    def dnV_dns(self):
        eos_mix = self.eos_mix
        try:
            dnV_dns = self.eos_mix.dnV_dns(eos_mix.Z_l)
        except AttributeError:
            dnV_dns = self.eos_mix.dnV_dns(eos_mix.Z_g)
        return dnV_dns

    def d2H_dep_dT2(self):
        try:
            return self.eos_mix.d2H_dep_dT2_l
        except AttributeError:
            return self.eos_mix.d2H_dep_dT2_g

    def d2H_dep_dT2_V(self):
        try:
            return self.eos_mix.d2H_dep_dT2_l_V
        except AttributeError:
            return self.eos_mix.d2H_dep_dT2_g_V



    def d2H_dTdP(self):
        try:
            return self.eos_mix.d2H_dep_dTdP_l
        except AttributeError:
            return self.eos_mix.d2H_dep_dTdP_g

    def dH_dep_dT_V(self):
        try:
            return self.eos_mix.dH_dep_dT_l_V
        except:
            return self.eos_mix.dH_dep_dT_g_V

    def dH_dep_dzs(self):
        try:
            return self.eos_mix.dH_dep_dzs(self.eos_mix.Z_l)
        except AttributeError:
            return self.eos_mix.dH_dep_dzs(self.eos_mix.Z_g)


    def d2P_dTdP(self):
        try:
            return self.eos_mix.d2P_dTdP_l
        except AttributeError:
            return self.eos_mix.d2P_dTdP_g

    def d2P_dVdP(self):
        try:
            return self.eos_mix.d2P_dVdP_l
        except AttributeError:
            return self.eos_mix.d2P_dVdP_g

    def d2P_dVdT_TP(self):
        try:
            return self.eos_mix.d2P_dVdT_TP_l
        except AttributeError:
            return self.eos_mix.d2P_dVdT_TP_g

    def d2P_dT2_PV(self):
        try:
            return self.eos_mix.d2P_dT2_PV_l
        except AttributeError:
            return self.eos_mix.d2P_dT2_PV_g

    def dS_dep_dzs(self):
        try:
            return self.eos_mix.dS_dep_dzs(self.eos_mix.Z_l)
        except AttributeError:
            return self.eos_mix.dS_dep_dzs(self.eos_mix.Z_g)

try:
    CEOSGas.__doc__ = CEOSPhase.__doc__
    CEOSLiquid.__doc__ = CEOSPhase.__doc__
except:
    pass

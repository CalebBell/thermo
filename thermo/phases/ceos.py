# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import os
from fluids.numerics import trunc_exp, numpy as np
from chemicals.utils import log
from thermo.eos_mix import IGMIX, eos_mix_full_path_dict, eos_mix_full_path_reverse_dict
from thermo.phases.phase_utils import PR_lnphis_fastest, lnphis_direct
from thermo.heat_capacity import HeatCapacityGas
from thermo.phases.phase import Phase
try:
    zeros = np.zeros
except:
    pass

class CEOSGas(Phase):
    r'''Class for representing a cubic equation of state gas phase
    as a phase object. All departure
    properties are actually calculated by the code in :obj:`thermo.eos` and
    :obj:`thermo.eos_mix`.

    .. math::
        P=\frac{RT}{V-b}-\frac{a\alpha(T)}{V^2 + \delta V + \epsilon}

    Parameters
    ----------
    eos_class : :obj:`thermo.eos_mix.GCEOSMIX`
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

    >>> from thermo import HeatCapacityGas, PRMIX, CEOSGas
    >>> R = CEOSGas.R
    >>> eos_kwargs = dict(Tcs=[154.58, 126.2], Pcs=[5042945.25, 3394387.5], omegas=[0.021, 0.04], kijs=[[0.0, -0.0159], [-0.0159, 0.0]])
    >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
    ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
    >>> phase = CEOSGas(eos_class=PRMIX, eos_kwargs=eos_kwargs, T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
    >>> phase.Cp()
    29.2285050

    '''
    is_gas = True
    is_liquid = False
    ideal_gas_basis = True

    pure_references = ('HeatCapacityGases',)
    pure_reference_types = (HeatCapacityGas,)
    obj_references = ('eos_mix',)


    pointer_references = ('eos_class',)
    pointer_reference_dicts = (eos_mix_full_path_dict,)
    '''Tuple of dictionaries for string -> object
    '''
    reference_pointer_dicts = (eos_mix_full_path_reverse_dict,)

    model_attributes = ('Hfs', 'Gfs', 'Sfs', 'eos_class',
                        'eos_kwargs') + pure_references

    @property
    def phase(self):
        phase = self.eos_mix.phase
        if phase in ('l', 'g'):
            return phase
        return 'g'

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

        Examples
        --------
        >>> from thermo import HeatCapacityGas, PRMIX, CEOSGas
        >>> R = CEOSGas
        >>> eos_kwargs = dict(Tcs=[154.58, 126.2], Pcs=[5042945.25, 3394387.5], omegas=[0.021, 0.04], kijs=[[0.0, -0.0159], [-0.0159, 0.0]])
        >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
        ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
        >>> phase = CEOSGas(eos_class=PRMIX, eos_kwargs=eos_kwargs, T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
        >>> phase
        CEOSGas(eos_class=PRMIX, eos_kwargs={"Tcs": [154.58, 126.2], "Pcs": [5042945.25, 3394387.5], "omegas": [0.021, 0.04], "kijs": [[0.0, -0.0159], [-0.0159, 0.0]]}, HeatCapacityGases=[HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-8.231317991971707e-12, 1.3053706310500586e-08, 5.820123832707268e-07, -0.0021700747433379955, 29.424883205644317])), HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [1.48828880864943e-11, -4.9886775708919434e-08, 5.4709164027448316e-05, -0.014916145936966912, 30.18149930389626]))], T=300, P=100000.0, zs=[0.79, 0.21])

        '''
        eos_kwargs = str(self.eos_kwargs).replace("'", '"')
        try:
            Cpgs = ', '.join(str(o) for o in self.HeatCapacityGases)
        except:
            Cpgs = ''
        base = '%s(eos_class=%s, eos_kwargs=%s, HeatCapacityGases=[%s], '  %(self.__class__.__name__, self.eos_class.__name__, eos_kwargs, Cpgs)
        for s in ('Hfs', 'Gfs', 'Sfs', 'T', 'P', 'zs'):
            if hasattr(self, s) and getattr(self, s) is not None:
                base += '%s=%s, ' %(s, getattr(self, s))
        if base[-2:] == ', ':
            base = base[:-2]
        base += ')'
        return base

    def __init__(self, eos_class, eos_kwargs, HeatCapacityGases=None, Hfs=None,
                 Gfs=None, Sfs=None,
                 T=None, P=None, zs=None):
        self.eos_class = eos_class
        self.eos_kwargs = eos_kwargs




        self.HeatCapacityGases = HeatCapacityGases
        if HeatCapacityGases is not None:
            self.N = N = len(HeatCapacityGases)
        elif 'Tcs' in eos_kwargs:
            self.N = N = len(eos_kwargs['Tcs'])

        self.Hfs = Hfs
        self.Gfs = Gfs
        self.Sfs = Sfs
        self.Cpgs_poly_fit, self._Cpgs_data = self._setup_Cpigs(HeatCapacityGases)
        self.composition_independent = ideal_gas = eos_class is IGMIX
        if ideal_gas:
            self.force_phase = 'g'


        if T is not None and P is not None and zs is not None:
            self.T = T
            self.P = P
            self.zs = zs
            self.eos_mix = eos_mix = self.eos_class(T=T, P=P, zs=zs, **self.eos_kwargs)
        else:
            zs = [1.0/N]*N
            self.eos_mix = eos_mix = self.eos_class(T=298.15, P=101325.0, zs=zs, **self.eos_kwargs)
            self.T = 298.15
            self.P = 101325.0
            self.zs = zs

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
        new.T = T
        new.P = P
        new.zs = zs
        if other_eos is not None:
            other_eos.solve_missing_volumes()
            new.eos_mix = other_eos
        else:
            try:
                new.eos_mix = self.eos_mix.to_TP_zs_fast(T=T, P=P, zs=zs, only_g=True,
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
        if new.composition_independent:
            new.force_phase = 'g'

        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs

        try:
            new.N = self.N
        except:
            pass

        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs

        if T is not None:
            if P is not None:
                try:
                    new.eos_mix = self.eos_mix.to_TP_zs_fast(T=T, P=P, zs=zs, only_g=True,
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
                new.eos_mix = self.eos_mix.to_PV_zs(P=P, V=V, zs=zs, only_g=True, fugacities=False)
            except AttributeError:
                new.eos_mix = self.eos_class(P=P, V=V, zs=zs, only_g=True, **self.eos_kwargs)
            T = new.eos_mix.T
        else:
            raise ValueError("Two of T, P, or V are needed")
        new.P = P
        new.T = T

        new.eos_class = self.eos_class
        new.eos_kwargs = self.eos_kwargs

        new.HeatCapacityGases = self.HeatCapacityGases
        new._Cpgs_data = self._Cpgs_data
        new.Cpgs_poly_fit = self.Cpgs_poly_fit

        new.composition_independent = self.composition_independent
        if new.composition_independent:
            new.force_phase = 'g'

        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs

        try:
            new.N = self.N
        except:
            pass

        return new

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
        try:
            return eos_mix.V_g_mpmath.real
        except:
            return eos_mix.V_l_mpmath.real
#        else:
#            return self.V()

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


    def lnphis_args(self):
        N = self.N
        eos_mix = self.eos_mix
        if self.eos_mix.scalar:
            a_alpha_j_rows, vec0 = [0.0]*N, [0.0]*N
        else:
            a_alpha_j_rows, vec0 = zeros(N), zeros(N)
        if eos_mix.translated:
            return (self.eos_class.model_id, self.T, self.P, self.N, eos_mix.kijs, self.is_liquid, self.is_gas,
                   eos_mix.b0s, eos_mix.bs, eos_mix.cs, eos_mix.a_alphas, eos_mix.a_alpha_roots, a_alpha_j_rows, vec0)
        else:
            return (self.eos_class.model_id, self.T, self.P, self.N, eos_mix.kijs, self.is_liquid, self.is_gas,
                   eos_mix.bs, eos_mix.a_alphas, eos_mix.a_alpha_roots, a_alpha_j_rows, vec0)

    def lnphis_at_zs(self, zs):
        eos_mix = self.eos_mix
        # if eos_mix.__class__.__name__ in ('PRMIX', 'VDWMIX', 'SRKMIX', 'RKMIX'):
        return lnphis_direct(zs, *self.lnphis_args())
        # return self.to_TP_zs(self.T, self.P, zs).lnphis()



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
        try:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_g)
        except AttributeError:
            return self.eos_mix.fugacity_coefficients(self.eos_mix.Z_l)


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
        try:
            return self.eos_mix.dlnphis_dT('g')
        except:
            return self.eos_mix.dlnphis_dT('l')

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
        # Confirmed to be mole fraction derivatives - taked with sum not 1 -
        # of the log fugacity coefficients!
        eos_mix = self.eos_mix
        try:
            return eos_mix.dlnphis_dzs(eos_mix.Z_g)
        except:
            return eos_mix.dlnphis_dzs(eos_mix.Z_l)

    def fugacities_lowest_Gibbs(self):
        eos_mix = self.eos_mix
        P = self.P
        zs = self.zs
        try:
            if eos_mix.G_dep_g < eos_mix.G_dep_l:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_g)
            else:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_l)
        except:
            try:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_g)
            except:
                lnphis = eos_mix.fugacity_coefficients(eos_mix.Z_l)
        return [P*zs[i]*trunc_exp(lnphis[i]) for i in range(len(zs))]


    def gammas(self):
        #         liquid.phis()/np.array([i.phi_l for i in liquid.eos_mix.pures()])
        phis = self.phis()
        phis_pure = []
        for i in self.eos_mix.pures():
            try:
                phis_pure.append(i.phi_g)
            except AttributeError:
                phis_pure.append(i.phi_l)
        return [phis[i]/phis_pure[i] for i in range(self.N)]


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

    def V(self):
        r'''Method to calculate and return the molar volume of the phase.

        Returns
        -------
        V : float
            Molar volume, [m^3/mol]
        '''
        try:
            return self.eos_mix.V_g
        except AttributeError:
            return self.eos_mix.V_l

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
        try:
            return self.eos_mix.dP_dT_g
        except AttributeError:
            return self.eos_mix.dP_dT_l

    dP_dT_V = dP_dT

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
        try:
            return self.eos_mix.dP_dV_g
        except AttributeError:
            return self.eos_mix.dP_dV_l

    dP_dV_T = dP_dV

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
        try:
            return self.eos_mix.d2P_dT2_g
        except AttributeError:
            return self.eos_mix.d2P_dT2_l

    d2P_dT2_V = d2P_dT2

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
        try:
            return self.eos_mix.d2P_dV2_g
        except AttributeError:
            return self.eos_mix.d2P_dV2_l

    d2P_dV2_T = d2P_dV2

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
        try:
            return self.eos_mix.d2P_dTdV_g
        except AttributeError:
            return self.eos_mix.d2P_dTdV_l

    # because of the ideal gas model, for some reason need to use the right ones
    # FOR THIS MODEL ONLY
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

    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        H = self.H_dep()
        for zi, Cp_int in zip(self.zs, self.Cpig_integrals_pure()):
            H += zi*Cp_int
        self._H = H
        return H

    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        log_zs = self.log_zs()
        P, zs, cmps = self.P, self.zs, range(self.N)
        P_REF_IG_INV = self.P_REF_IG_INV
        R = self.R
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)

        for i in cmps:
            S += zs[i]*Cpig_integrals_over_T_pure[i]
        S += self.S_dep()
        self._S = S
        return S



    def Cp(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        for i in range(self.N):
            Cp += zs[i]*Cpigs_pure[i]
        Cp += self.Cp_dep()
        self._Cp = Cp
        return Cp

    dH_dT = dH_dT_P = Cp

    def dH_dP(self):
        try:
            return self._dH_dP
        except AttributeError:
            pass
        try:
            self._dH_dP = dH_dP = self.eos_mix.dH_dep_dP_g
        except AttributeError:
            self._dH_dP = dH_dP = self.eos_mix.dH_dep_dP_l
        return dH_dP

    def d2H_dT2(self):
        try:
            return self._d2H_dT2
        except AttributeError:
            pass
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        for i in range(self.N):
            dCp += zs[i]*dCpigs_pure[i]
        try:
            dCp += self.eos_mix.d2H_dep_dT2_g
        except AttributeError:
            dCp += self.eos_mix.d2H_dep_dT2_l
        self._d2H_dT2 = dCp
        return dCp

    def d2H_dT2_V(self):
        # Turned out not to be needed when I thought it was - ignore this!
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        for i in range(self.N):
            dCp += zs[i]*dCpigs_pure[i]

        try:
            dCp += self.eos_mix.d2H_dep_dT2_g_V
        except AttributeError:
            dCp += self.eos_mix.d2H_dep_dT2_l_V
        return dCp


    def d2H_dP2(self):
        try:
            return self.eos_mix.d2H_dep_dP2_g
        except AttributeError:
            return self.eos_mix.d2H_dep_dP2_l

    def d2H_dTdP(self):
        try:
            return self.eos_mix.d2H_dep_dTdP_g
        except AttributeError:
            return self.eos_mix.d2H_dep_dTdP_l


    def dH_dT_V(self):
        dH_dT_V = self.Cp_ideal_gas()
        try:
            dH_dT_V += self.eos_mix.dH_dep_dT_g_V
        except AttributeError:
            dH_dT_V += self.eos_mix.dH_dep_dT_l_V
        return dH_dT_V

    def dH_dP_V(self):
        dH_dP_V = self.Cp_ideal_gas()*self.dT_dP()
        try:
            dH_dP_V += self.eos_mix.dH_dep_dP_g_V
        except AttributeError:
            dH_dP_V += self.eos_mix.dH_dep_dP_l_V
        return dH_dP_V

    def dH_dV_T(self):
        dH_dV_T = 0.0
        try:
            dH_dV_T += self.eos_mix.dH_dep_dV_g_T
        except AttributeError:
            dH_dV_T += self.eos_mix.dH_dep_dV_l_T
        return dH_dV_T

    def dH_dV_P(self):
        dH_dV_P = self.dT_dV()*self.Cp_ideal_gas()
        try:
            dH_dV_P += self.eos_mix.dH_dep_dV_g_P
        except AttributeError:
            dH_dV_P += self.eos_mix.dH_dep_dV_l_P
        return dH_dV_P

    def dH_dzs(self):
        try:
            return self._dH_dzs
        except AttributeError:
            pass
        eos_mix = self.eos_mix
        try:
            dH_dep_dzs = self.eos_mix.dH_dep_dzs(eos_mix.Z_g)
        except AttributeError:
            dH_dep_dzs = self.eos_mix.dH_dep_dzs(eos_mix.Z_l)
        Cpig_integrals_pure = self.Cpig_integrals_pure()
        self._dH_dzs = [dH_dep_dzs[i] + Cpig_integrals_pure[i] for i in range(self.N)]
        return self._dH_dzs

    def dS_dT(self):
        HeatCapacityGases = self.HeatCapacityGases
        cmps = range(self.N)
        T, zs = self.T, self.zs
        T_REF_IG = self.T_REF_IG
        P_REF_IG_INV = self.P_REF_IG_INV

        dS_dT = self.Cp_ideal_gas()/T
        try:
            dS_dT += self.eos_mix.dS_dep_dT_g
        except AttributeError:
            dS_dT += self.eos_mix.dS_dep_dT_l
        return dS_dT

    def dS_dP(self):
        dS = 0.0
        P = self.P
        dS -= self.R/P
        try:
            dS += self.eos_mix.dS_dep_dP_g
        except AttributeError:
            dS += self.eos_mix.dS_dep_dP_l
        return dS

    def d2S_dP2(self):
        P = self.P
        d2S = self.R/(P*P)
        try:
            d2S += self.eos_mix.d2S_dep_dP_g
        except AttributeError:
            d2S += self.eos_mix.d2S_dep_dP_l
        return d2S

    def dS_dT_P(self):
        return self.dS_dT()

    def dS_dT_V(self):
        r'''Method to calculate and return the first temperature derivative of
        molar entropy at constant volume of the phase.

        .. math::
            \left(\frac{\partial S}{\partial T}\right)_V =
            \frac{C_p^{ig}}{T} - \frac{R}{P}\frac{\partial P}{\partial T}
            + \left(\frac{\partial S_{dep}}{\partial T}\right)_V

        Returns
        -------
        dS_dT_V : float
            First temperature derivative of molar entropy at constant volume,
            [J/(mol*K^2)]
        '''
        # Good
        '''
        # Second last bit from
        from sympy import *
        T, R = symbols('T, R')
        P = symbols('P', cls=Function)
        expr =-R*log(P(T)/101325)
        diff(expr, T)
        '''
        dS_dT_V = self.Cp_ideal_gas()/self.T - self.R/self.P*self.dP_dT()
        try:
            dS_dT_V += self.eos_mix.dS_dep_dT_g_V
        except AttributeError:
            dS_dT_V += self.eos_mix.dS_dep_dT_l_V
        return dS_dT_V

    def dS_dP_V(self):
        dS_dP_V = -self.R/self.P + self.Cp_ideal_gas()/self.T*self.dT_dP()
        try:
            dS_dP_V += self.eos_mix.dS_dep_dP_g_V
        except AttributeError:
            dS_dP_V += self.eos_mix.dS_dep_dP_l_V
        return dS_dP_V

    # The following - likely should be reimplemented generically
    # http://www.coolprop.org/_static/doxygen/html/class_cool_prop_1_1_abstract_state.html#a0815380e76a7dc9c8cc39493a9f3df46

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

    def dS_dzs(self):
        try:
            return self._dS_dzs
        except AttributeError:
            pass
        cmps, eos_mix = range(self.N), self.eos_mix

        log_zs = self.log_zs()
        integrals = self.Cpig_integrals_over_T_pure()

        try:
            dS_dep_dzs = self.eos_mix.dS_dep_dzs(eos_mix.Z_g)
        except AttributeError:
            dS_dep_dzs = self.eos_mix.dS_dep_dzs(eos_mix.Z_l)
        R = self.R
        self._dS_dzs = [integrals[i] - R*(log_zs[i] + 1.0) + dS_dep_dzs[i]
                        for i in cmps]
        return self._dS_dzs

    def _set_mechanical_critical_point(self):
        zs = self.zs
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
#        try:
#            return self._mu
#        except AttributeError:
#            pass
        try:
            phase = self.assigned_phase
        except:
            phase = self.eos_mix.phase
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
            phase = self.eos_mix.phase
            if phase == 'l/g': phase = 'g'
        if phase == 'g':
            k = self.correlations.ThermalConductivityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        elif phase == 'l':
            k = self.correlations.ThermalConductivityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._k = k
        return k

def build_CEOSLiquid():
    import inspect
    source = inspect.getsource(CEOSGas)
    source = source.replace('CEOSGas', 'CEOSLiquid').replace('only_g', 'only_l')
    source = source.replace("'g'", "'gORIG'")
    source = source.replace("'l'", "'g'")
    source = source.replace("'gORIG'", "'l'")
    source = source.replace("ViscosityGasMixture", "gViscosityGasMixture")
    source = source.replace("ViscosityLiquidMixture", "ViscosityGasMixture")
    source = source.replace("gViscosityGasMixture", "ViscosityLiquidMixture")
    source = source.replace("ThermalConductivityGasMixture", "gThermalConductivityGasMixture")
    source = source.replace("ThermalConductivityLiquidMixture", "ThermalConductivityGasMixture")
    source = source.replace("gThermalConductivityGasMixture", "ThermalConductivityLiquidMixture")
    # TODO add new volume derivatives
    swap_strings = ('Cp_dep', 'd2P_dT2', 'd2P_dTdV', 'd2P_dV2', 'd2T_dV2',
                    'd2V_dT2', 'dH_dep_dP', 'dP_dT', 'dP_dV', 'phi',
                    'dS_dep_dP', 'dS_dep_dT', 'G_dep', 'H_dep', 'S_dep', '.V', '.Z',
                    'd2P_dVdT_TP', 'd2P_dT2_PV', 'd2P_dVdP', 'd2P_dTdP',
                    'd2S_dep_dP', 'dH_dep_dV', 'dH_dep_dT', 'd2H_dep_dTdP',
                    'd2H_dep_dP2', 'd2H_dep_dT2')
    for s in swap_strings:
        source = source.replace(s+'_g', 'gORIG')
        source = source.replace(s+'_l', s+'_g')
        source = source.replace('gORIG', s+'_l')
    return source

from fluids.numerics import is_micropython
if is_micropython:
    class CEOSLiquid(object): 
        __full_path__ = None
else:
    try:
        CEOSLiquid
    except:
        loaded_data = False
        # Cost is ~10 ms - must be pasted in the future!
        try:  # pragma: no cover
            from appdirs import user_data_dir, user_config_dir
            data_dir = user_config_dir('thermo')
        except ImportError:  # pragma: no cover
            data_dir = ''
        if data_dir:
            import marshal
            try:
                1/0
                f = open(os.path.join(data_dir, 'CEOSLiquid.dat'), 'rb')
                compiled_CEOSLiquid = marshal.load(f)
                f.close()
                loaded_data = True
            except:
                pass
            if not loaded_data:
                compiled_CEOSLiquid = compile(build_CEOSLiquid(), '<string>', 'exec')
                f = open(os.path.join(data_dir, 'CEOSLiquid.dat'), 'wb')
                marshal.dump(compiled_CEOSLiquid, f)
                f.close()
        else:
            compiled_CEOSLiquid = compile(build_CEOSLiquid(), '<string>', 'exec')
        exec(compiled_CEOSLiquid)
        # exec(build_CEOSLiquid())

CEOSLiquid.is_gas = False
CEOSLiquid.is_liquid = True

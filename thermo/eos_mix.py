# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020, 2021
Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains implementations of most cubic equations of state for
mixtures. This includes Peng-Robinson, SRK, Van der Waals, PRSV, TWU and
many other variants.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Base Class
==========
.. autoclass:: thermo.eos_mix.GCEOSMIX
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: a_alpha_and_derivatives_numpy, a_alpha_and_derivatives_py, main_derivatives_and_departures, derivatives_and_departures,
                      sequential_substitution_3P, sequential_substitution_VL, stability_Michelsen, stability_iteration_Michelsen, newton_VL, broyden2_VL,
                      d2A_dep_dninjs, d2A_dep_dninjs_Vt, d2A_dninjs_Vt, d2A_dninjs_Vt_another, d2P_dninjs_Vt, d2nA_dninjs_Vt, d3P_dninjnks_Vt,
                      dScomp_dns, d2Scomp_dninjs, dA_dep_dns_Vt, dP_dns_Vt

Peng-Robinson Family EOSs
=========================

Standard Peng Robinson
----------------------
.. autoclass:: thermo.eos_mix.PRMIX
   :show-inheritance:
   :members: eos_pure, a_alphas_vectorized, a_alpha_and_derivatives_vectorized,
             d3a_alpha_dT3, d3a_alpha_dT3_vectorized, fugacity_coefficients,
             dlnphis_dT, dlnphis_dP, dlnphis_dzs, ddelta_dzs, ddelta_dns,
             d2delta_dzizjs, d2delta_dninjs, d3delta_dninjnks, depsilon_dzs,
             depsilon_dns, d2epsilon_dzizjs, d2epsilon_dninjs,
             d3epsilon_dninjnks

Peng Robinson (1978)
--------------------
.. autoclass:: thermo.eos_mix.PR78MIX
   :show-inheritance:
   :members: eos_pure

Peng Robinson Stryjek-Vera
--------------------------
.. autoclass:: thermo.eos_mix.PRSVMIX
   :show-inheritance:
   :members: eos_pure, a_alphas_vectorized, a_alpha_and_derivatives_vectorized

Peng Robinson Stryjek-Vera 2
----------------------------
.. autoclass:: thermo.eos_mix.PRSV2MIX
   :show-inheritance:
   :members: eos_pure, a_alphas_vectorized, a_alpha_and_derivatives_vectorized

Peng Robinson Twu (1995)
------------------------
.. autoclass:: thermo.eos_mix.TWUPRMIX
   :show-inheritance:
   :members: eos_pure

Peng Robinson Translated
------------------------
.. autoclass:: thermo.eos_mix.PRMIXTranslated
   :show-inheritance:
   :members: eos_pure, ddelta_dzs, d2delta_dzizjs, d3delta_dzizjzks, ddelta_dns,
             d2delta_dninjs, d3delta_dninjnks, depsilon_dzs, depsilon_dns,
             d2epsilon_dzizjs, d3epsilon_dzizjzks, d2epsilon_dninjs,
             d3epsilon_dninjnks

Peng Robinson Translated-Consistent
-----------------------------------
.. autoclass:: thermo.eos_mix.PRMIXTranslatedConsistent
   :show-inheritance:
   :members: eos_pure

Peng Robinson Translated (Pina-Martinez, Privat, and Jaubert Variant)
---------------------------------------------------------------------
.. autoclass:: thermo.eos_mix.PRMIXTranslatedPPJP
   :show-inheritance:
   :members: eos_pure

SRK Family EOSs
===============

Standard SRK
------------
.. autoclass:: thermo.eos_mix.SRKMIX
   :show-inheritance:
   :members: eos_pure, dlnphis_dT, dlnphis_dP, a_alphas_vectorized,
             a_alpha_and_derivatives_vectorized, fugacity_coefficients
   :exclude-members:

Twu SRK (1995)
--------------
.. autoclass:: thermo.eos_mix.TWUSRKMIX
   :show-inheritance:
   :members: eos_pure

API SRK
-------
.. autoclass:: thermo.eos_mix.APISRKMIX
   :show-inheritance:
   :members: eos_pure

SRK Translated
--------------
.. autoclass:: thermo.eos_mix.SRKMIXTranslated
   :show-inheritance:
   :members: eos_pure, ddelta_dzs, d2delta_dzizjs, d3delta_dzizjzks, ddelta_dns,
             d2delta_dninjs, d3delta_dninjnks, depsilon_dzs, depsilon_dns,
             d2epsilon_dzizjs, d3epsilon_dzizjzks, d2epsilon_dninjs,
             d3epsilon_dninjnks

SRK Translated-Consistent
-------------------------
.. autoclass:: thermo.eos_mix.SRKMIXTranslatedConsistent
   :show-inheritance:
   :members: eos_pure

MSRK Translated
---------------
.. autoclass:: thermo.eos_mix.MSRKMIXTranslated
   :show-inheritance:
   :members: eos_pure

Cubic Equation of State with Activity Coefficients
==================================================
.. autoclass:: thermo.eos_mix.PSRK
   :show-inheritance:
   :members: eos_pure

Van der Waals Equation of State
===============================
.. autoclass:: thermo.eos_mix.VDWMIX
   :show-inheritance:
   :members: eos_pure, dlnphis_dT, dlnphis_dP, a_alphas_vectorized,
             a_alpha_and_derivatives_vectorized, fugacity_coefficients,
             ddelta_dzs, ddelta_dns, d2delta_dzizjs, d2delta_dninjs,
             d3delta_dninjnks

Redlich-Kwong Equation of State
===============================
.. autoclass:: thermo.eos_mix.RKMIX
   :show-inheritance:
   :members: eos_pure, a_alphas_vectorized, a_alpha_and_derivatives_vectorized,
             ddelta_dzs, ddelta_dns, d2delta_dzizjs, d2delta_dninjs,
             d3delta_dninjnks

Ideal Gas Equation of State
===========================
.. autoclass:: thermo.eos_mix.IGMIX
   :show-inheritance:
   :members: eos_pure, a_alphas_vectorized, a_alpha_and_derivatives_vectorized

Different Mixing Rules
======================
.. autoclass:: thermo.eos_mix.EpsilonZeroMixingRules
.. autoclass:: thermo.eos_mix.PSRKMixingRules
    :members: u, A, a_alpha_and_derivatives
    :undoc-members:
    :show-inheritance:

Lists of Equations of State
===========================
.. autodata:: thermo.eos_mix.eos_mix_list
.. autodata:: thermo.eos_mix.eos_mix_no_coeffs_list

'''
from __future__ import division


__all__ = ['GCEOSMIX', 'PRMIX', 'SRKMIX', 'PR78MIX', 'VDWMIX', 'PRSVMIX',
'PRSV2MIX', 'TWUPRMIX', 'TWUSRKMIX', 'APISRKMIX', 'IGMIX', 'RKMIX',
'PRMIXTranslatedConsistent', 'PRMIXTranslatedPPJP', 'PRMIXTranslated',
'SRKMIXTranslatedConsistent', 'PSRK', 'MSRKMIXTranslated',
'eos_mix_list', 'eos_mix_no_coeffs_list', 'SRKMIXTranslated']

import sys
from cmath import log as clog

from fluids.numerics import numpy as np, IS_PYPY, newton_system, broyden2, UnconvergedError, trunc_exp, solve_2_direct, catanh
from fluids.numerics.arrays import det, subset_matrix
from fluids.constants import R

from chemicals.utils import normalize, dxs_to_dn_partials, dxs_to_dns, dns_to_dn_partials, d2xs_to_dxdn_partials, d2ns_to_dn2_partials
from chemicals.utils import log, exp, sqrt
from chemicals.rachford_rice import flash_inner_loop, Rachford_Rice_flash_error, Rachford_Rice_solution2
from chemicals.flash_basic import K_value, Wilson_K_value

from thermo import serialize
from thermo.eos_mix_methods import (a_alpha_aijs_composition_independent,
    a_alpha_aijs_composition_independent_support_zeros, a_alpha_and_derivatives, a_alpha_and_derivatives_full,
    a_alpha_quadratic_terms, a_alpha_and_derivatives_quadratic_terms,
    G_dep_lnphi_d_helper, eos_mix_dV_dzs, VDW_lnphis, SRK_lnphis, eos_mix_db_dns, PR_translated_ddelta_dns,
    PR_translated_depsilon_dns, PR_depsilon_dns, PR_translated_d2epsilon_dzizjs,
    PR_d2epsilon_dninjs, PR_d3epsilon_dninjnks, PR_d2delta_dninjs, PR_d3delta_dninjnks,
    PR_ddelta_dzs, PR_ddelta_dns, PR_d2epsilon_dzizjs, PR_depsilon_dzs,
    RK_d3delta_dninjnks, SRK_translated_d2epsilon_dzizjs, SRK_translated_depsilon_dzs,
    PR_translated_ddelta_dzs, PR_translated_depsilon_dzs, PR_translated_d2epsilon_dninjs,
    PR_translated_d2delta_dninjs, PR_translated_d3delta_dninjnks, PR_translated_d3epsilon_dninjnks,
    SRK_translated_ddelta_dns, SRK_translated_depsilon_dns, SRK_translated_d2delta_dninjs,
    SRK_translated_d2epsilon_dninjs, SRK_translated_d3epsilon_dninjnks,
    SRK_translated_d3delta_dninjnks)
from thermo.eos_alpha_functions import (TwuPR95_a_alpha, TwuSRK95_a_alpha, Twu91_a_alpha, Mathias_Copeman_a_alpha,
                                    Soave_79_a_alpha, PR_a_alpha_and_derivatives_vectorized, PR_a_alphas_vectorized,
                                    RK_a_alpha_and_derivatives_vectorized, RK_a_alphas_vectorized,
                                    SRK_a_alpha_and_derivatives_vectorized, SRK_a_alphas_vectorized,
                                    PRSV_a_alphas_vectorized, PRSV_a_alpha_and_derivatives_vectorized,
                                    PRSV2_a_alphas_vectorized, PRSV2_a_alpha_and_derivatives_vectorized,
                                    APISRK_a_alphas_vectorized, APISRK_a_alpha_and_derivatives_vectorized)
from thermo.eos import *

try:
    (zeros, array, npexp, npsqrt, empty, full, npwhere, npmin, npmax) = (
        np.zeros, np.array, np.exp, np.sqrt, np.empty, np.full, np.where, np.min, np.max)
except:
    pass

R2 = R*R
R_inv = 1.0/R
R2_inv = R_inv*R_inv

two_root_two = 2*2**0.5
root_two = sqrt(2.)
root_two_m1 = root_two - 1.0
root_two_p1 = root_two + 1.0

c1R2_PR = PR.c1R2
c2R_PR = PR.c2R


class GCEOSMIX(GCEOS):
    r'''Class for solving a generic pressure-explicit three-parameter cubic
    equation of state for a mixture. Does not implement any parameters itself;
    must be subclassed by a mixture equation of state class which subclasses it.

    .. math::
        P=\frac{RT}{V-b}-\frac{a\alpha(T)}{V^2 + \delta V + \epsilon}
    '''
    nonstate_constants = ('N', 'cmps', 'Tcs', 'Pcs', 'omegas', 'kijs', 'kwargs', 'ais', 'bs')
    mix_kwargs_to_pure = {}
    kwargs_square = ('kijs',)
    '''Tuple of 2D arguments used by the specific EOS.
    '''
    kwargs_linear = tuple()
    '''Tuple of 1D arguments used by the specific EOS in addition to the conventional ones.
    '''
    multicomponent = True
    '''All inherited classes of GCEOSMIX are multicomponent.
    '''
    scalar = True
    '''Whether the model is implemented using pure-Python lists of floats,
    or numpy arrays of float64.
    '''
    translated = False
    '''Whether or not the model implements volume translation.
    '''

    def subset(self, idxs, **state_specs):
        r'''Method to construct a new :obj:`GCEOSMIX` that removes all components
        not specified in the `idxs` argument.

        Parameters
        ----------
        idxs : list[int] or Slice
            Indexes of components that should be included, [-]

        Returns
        -------
        subset_eos : :obj:`GCEOSMIX`
            Multicomponent :obj:`GCEOSMIX` at the same specified specs but with a
            composition normalized to 1 and with fewer components, [-]
        state_specs : float
            Keyword arguments which can be any of `T`, `P`, `V`, `zs`; `zs`
            is optional, as are (`T`, `P`, `V`), but if any of (`T`, `P`, `V`)
            are specified, a second one is required as well, [various]

        Notes
        -----
        Subclassing equations of state require their :obj:`kwargs_linear <GCEOSMIX.kwargs_linear>` and
        :obj:`kwargs_square <GCEOSMIX.kwargs_square>` attributes to be correct for this to work.
        `Tcs`, `Pcs`, and `omegas` are always assumed to be used.

        Examples
        --------
        >>> kijs = [[0.0, 0.00076, 0.00171], [0.00076, 0.0, 0.00061], [0.00171, 0.00061, 0.0]]
        >>> PR3 = PRMIX(Tcs=[469.7, 507.4, 540.3], zs=[0.8168, 0.1501, 0.0331], omegas=[0.249, 0.305, 0.349], Pcs=[3.369E6, 3.012E6, 2.736E6], T=322.29, P=101325.0, kijs=kijs)
        >>> PR3.subset([1,2])
        PRMIX(Tcs=[507.4, 540.3], Pcs=[3012000.0, 2736000.0], omegas=[0.305, 0.349], kijs=[[0.0, 0.00061], [0.00061, 0.0]], zs=[0.8193231441048036, 0.1806768558951965], T=322.29, P=101325.0)
        >>> PR3.subset([1,2], T=500.0, P=1e5, zs=[.2, .8])
        PRMIX(Tcs=[507.4, 540.3], Pcs=[3012000.0, 2736000.0], omegas=[0.305, 0.349], kijs=[[0.0, 0.00061], [0.00061, 0.0]], zs=[0.2, 0.8], T=500.0, P=100000.0)
        >>> PR3.subset([1,2], zs=[.2, .8])
        PRMIX(Tcs=[507.4, 540.3], Pcs=[3012000.0, 2736000.0], omegas=[0.305, 0.349], kijs=[[0.0, 0.00061], [0.00061, 0.0]], zs=[0.2, 0.8], T=322.29, P=101325.0)
        '''
        is_slice = isinstance(idxs, slice)

        if is_slice:
            def atindexes(values):
                return values[idxs]
        else:
            def atindexes(values):
                return [values[i] for i in idxs]

        if state_specs:
            kwargs = state_specs
            if len(kwargs) == 1 and 'zs' in kwargs:
                kwargs.update(self.state_specs)
        else:
            kwargs = self.state_specs

        if 'zs' not in kwargs:
            zs = atindexes(self.zs)
            if not zs:
                raise ValueError("Cannot create an EOS without any components selected")
            zs_tot_inv = 1.0/sum(zs)
            for i in range(len(zs)):
                zs[i] *= zs_tot_inv
            kwargs['zs'] = zs
        kwargs['Tcs'] = atindexes(self.Tcs)
        kwargs['Pcs'] = atindexes(self.Pcs)
        kwargs['omegas'] = atindexes(self.omegas)
        local_kwargs = self.kwargs
        for k in self.kwargs_linear:
            kwargs[k] = atindexes(local_kwargs[k])
        for k in self.kwargs_square:
            kwargs[k] = subset_matrix(local_kwargs[k], idxs)
        return self.__class__(**kwargs)




    def __repr__(self):
        s = '%s(Tcs=%s, Pcs=%s, omegas=%s, ' %(self.__class__.__name__, repr(self.Tcs), repr(self.Pcs), repr(self.omegas))
        for k, v in self.kwargs.items():
            s += '%s=%s, ' %(k, repr(v))

        s += 'zs=%s, ' %(repr(self.zs))
        if hasattr(self, 'no_T_spec') and self.no_T_spec:
            s += 'P=%s, V=%s' %(repr(self.P), repr(self.V))
        elif self.V is not None:
            s += 'T=%s, V=%s' %(repr(self.T), repr(self.V))
        else:
            s += 'T=%s, P=%s' %(repr(self.T), repr(self.P))
        s += ')'
        return s

    @classmethod
    def from_json(cls, json_repr):
        r'''Method to create a mixture cubic equation of state from a JSON
        friendly serialization of another mixture cubic equation of state.

        Parameters
        ----------
        json_repr : dict
            Json representation, [-]

        Returns
        -------
        eos_mix : :obj:`GCEOSMIX`
            Newly created object from the json serialization, [-]

        Notes
        -----
        It is important that the input string be in the same format as that
        created by :obj:`GCEOS.as_json`.

        Examples
        --------
        >>> import pickle
        >>> eos = PRSV2MIX(Tcs=[507.6], Pcs=[3025000], omegas=[0.2975], zs=[1], T=299., P=1E6, kappa1s=[0.05104], kappa2s=[0.8634], kappa3s=[0.460])
        >>> json_stuff = pickle.dumps(eos.as_json())
        >>> new_eos = GCEOSMIX.from_json(pickle.loads(json_stuff))
        >>> assert new_eos == eos
        '''
        d = json_repr

        eos_name = d['py/object']
        del d['py/object']
        del d['json_version']
        if not d['scalar']:
            d = serialize.naive_lists_to_arrays(d)

        try:
            d['raw_volumes'] = tuple(d['raw_volumes'])
        except:
            pass

        try:
            alpha_coeffs = [tuple(v) for v in d['alpha_coeffs']]
            d['alpha_coeffs'] = alpha_coeffs
        except:
            pass

        eos = eos_mix_full_path_dict[eos_name]

        if eos.kwargs_keys:
            d['kwargs'] = {k: d[k] for k in eos.kwargs_keys}
            try:
                d['kwargs']['alpha_coeffs'] = alpha_coeffs
            except:
                pass

        new = eos.__new__(eos)
        new.__dict__ = d
        return new

    def to_TP_zs_fast(self, T, P, zs, only_l=False, only_g=False, full_alphas=True):
        r'''Method to construct a new :obj:`GCEOSMIX` instance with the same
        parameters as the existing object. If both instances are at the same
        temperature, `a_alphas` and `da_alpha_dTs` and `d2a_alpha_dT2s` are
        shared between the instances. It is always assumed the new object has
        a differet composition. Optionally, only one set of phase properties
        can be solved for, increasing speed. Additionally, if `full_alphas`
        is set to False no temperature derivatives of `a_alpha` will be
        computed. Those derivatives are not needed in the context of a
        PT or PVF flash.

        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]
        zs : list[float]
            Mole fractions of each component, [-]
        only_l : bool
            When true, if there is a liquid and a vapor root, only the liquid
            root (and properties) will be set.
        only_g : bool
            When true, if there is a liquid and a vapor root, only the vapor
            root (and properties) will be set.

        Returns
        -------
        eos : :obj:`GCEOSMIX`
            Multicomponent :obj:`GCEOSMIX` at the specified conditions [-]

        Notes
        -----

        Examples
        --------
        >>> base = RKMIX(T=500.0, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> base.to_TP_zs_fast(T=300, P=1e5, zs=base.zs)
        RKMIX(Tcs=[126.1, 190.6], Pcs=[3394000.0, 4604000.0], omegas=[0.04, 0.011], kijs=[[0.0, 0.0], [0.0, 0.0]], zs=[0.6, 0.4], T=300, P=100000.0)
        '''
        copy_alphas = T == self.T
        new = self.__class__.__new__(self.__class__)
        new.N = self.N
        new.Tcs = self.Tcs
        new.Pcs = self.Pcs
        new.omegas = self.omegas
        new.kijs = self.kijs
        new.kwargs = self.kwargs
        new.ais = self.ais
        new.bs = self.bs
        new.scalar = self.scalar

        if copy_alphas:
            new.a_alphas = self.a_alphas
            try:
                new.da_alpha_dTs = self.da_alpha_dTs
                new.d2a_alpha_dT2s = self.d2a_alpha_dT2s
            except:
                pass

        new.zs = zs
        new.T = T
        new.P = P
        new.V = None
        new._fast_init_specific(self)
        new.solve(pure_a_alphas=(not copy_alphas), only_l=only_l,
                  only_g=only_g, full_alphas=full_alphas)
        return new


    def to_TP_zs(self, T, P, zs, fugacities=True, only_l=False, only_g=False):
        r'''Method to construct a new :obj:`GCEOSMIX` instance at `T`, `P`, and `zs`
        with the same parameters as the existing object. Optionally, only one
        set of phase properties can be solved for, increasing speed. The
        fugacities calculation can be be skipped by by setting `fugacities` to
        False.

        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]
        zs : list[float]
            Mole fractions of each component, [-]
        fugacities : bool
            Whether or not to calculate and set the fugacities of each
            component, [-]
        only_l : bool
            When true, if there is a liquid and a vapor root, only the liquid
            root (and properties) will be set.
        only_g : bool
            When true, if there is a liquid and a vapor root, only the vapor
            root (and properties) will be set.

        Returns
        -------
        eos : :obj:`GCEOSMIX`
            Multicomponent :obj:`GCEOSMIX` at the specified conditions [-]

        Notes
        -----
        A check for whether or not `T`, `P`, and `zs` are the same as the
        existing instance is performed; if it is, the existing object is
        returned.

        Examples
        --------
        >>> base = RKMIX(T=500.0, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> base.to_TP_zs(T=300, P=1e5, zs=[.1, 0.9])
        RKMIX(Tcs=[126.1, 190.6], Pcs=[3394000.0, 4604000.0], omegas=[0.04, 0.011], kijs=[[0.0, 0.0], [0.0, 0.0]], zs=[0.1, 0.9], T=300, P=100000.0)
        '''
        if T != self.T or P != self.P or zs != self.zs:
            return self.__class__(T=T, P=P, zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, only_l=only_l, only_g=only_g, fugacities=fugacities, **self.kwargs)
        else:
            return self

    def to_PV_zs(self, P, V, zs, fugacities=True, only_l=False, only_g=False):
        r'''Method to construct a new :obj:`GCEOSMIX` instance at `P`, `V`, and `zs`
        with the same parameters as the existing object. Optionally, only one
        set of phase properties can be solved for, increasing speed. The
        fugacities calculation can be be skipped by by setting `fugacities` to
        False.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        zs : list[float]
            Mole fractions of each component, [-]
        fugacities : bool
            Whether or not to calculate and set the fugacities of each
            component, [-]
        only_l : bool
            When true, if there is a liquid and a vapor root, only the liquid
            root (and properties) will be set.
        only_g : bool
            When true, if there is a liquid and a vapor root, only the vapor
            root (and properties) will be set.

        Returns
        -------
        eos : :obj:`GCEOSMIX`
            Multicomponent :obj:`GCEOSMIX` at the specified conditions [-]

        Notes
        -----
        A check for whether or not `P`, `V`, and `zs` are the same as the
        existing instance is performed; if it is, the existing object is
        returned.

        Examples
        --------
        >>> base = RKMIX(T=500.0, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> base.to_PV_zs(V=0.004162, P=1e5, zs=[.1, 0.9])
        RKMIX(Tcs=[126.1, 190.6], Pcs=[3394000.0, 4604000.0], omegas=[0.04, 0.011], kijs=[[0.0, 0.0], [0.0, 0.0]], zs=[0.1, 0.9], P=100000.0, V=0.004162)
        '''
        if P == self.P and V == self.V and zs == self.zs:
            return self
        return self.__class__(P=P, V=V, zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, only_l=only_l, only_g=only_g, fugacities=fugacities, **self.kwargs)

    def to(self, zs=None, T=None, P=None, V=None, fugacities=True):
        r'''Method to construct a new :obj:`GCEOSMIX` object at two of `T`, `P` or `V`
        with the specified composition.
        In the event the specs match those of the current object, it will be
        returned unchanged.

        Parameters
        ----------
        zs : list[float], optional
            Mole fractions of EOS, [-]
        T : float or None, optional
            Temperature, [K]
        P : float or None, optional
            Pressure, [Pa]
        V : float or None, optional
            Molar volume, [m^3/mol]
        fugacities : bool
            Whether or not to calculate fugacities, [-]

        Returns
        -------
        obj : :obj:`GCEOSMIX`
            Pure component :obj:`GCEOSMIX` at the two specified specs, [-]

        Notes
        -----
        Constructs the object with parameters `Tcs`, `Pcs`, `omegas`, and
        `kwargs`.

        Examples
        --------
        >>> base = PRMIX(T=500.0, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> base.to(T=300.0, P=1e9).state_specs
        {'T': 300.0, 'P': 1000000000.0}
        >>> base.to(T=300.0, V=1.0).state_specs
        {'T': 300.0, 'V': 1.0}
        >>> base.to(P=1e5, V=1.0).state_specs
        {'P': 100000.0, 'V': 1.0}
        '''
        if zs is None:
            zs = self.zs
        if T is not None and P is not None:
            try:
                sln = self.to_TP_zs_fast(T, P, zs)
                if fugacities:
                    sln.fugacities()
                return sln
            except:
                return self.to_TP_zs(T, P, zs, fugacities)
        elif T is not None and V is not None:
            if T == self.T and V == self.V and zs == self.zs:
                return self
            return self.__class__(T=T, V=V, zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, fugacities=fugacities, **self.kwargs)
        elif P is not None and V is not None:
            return self.to_PV_zs(P, V, zs, fugacities)
        else:
            return self.__class__(T=T, P=P, V=V, zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, fugacities=fugacities, **self.kwargs)

    def to_TP(self, T, P):
        r'''Method to construct a new :obj:`GCEOSMIX` object at the spcified `T` and `P`
        with the current composition. In the event the `T` and `P` match the
        current object's `T` and `P`, it will be returned unchanged.

        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]

        Returns
        -------
        obj : :obj:`GCEOSMIX`
            Pure component :obj:`GCEOSMIX` at specified `T` and `P`, [-]

        Notes
        -----
        Constructs the object with parameters `Tcs`, `Pcs`, `omegas`, and
        `kwargs`.

        Examples
        --------

        >>> base = RKMIX(T=500.0, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> new = base.to_TP(T=10.0, P=2000.0)
        >>> base.state_specs, new.state_specs
        ({'T': 500.0, 'P': 1000000.0}, {'T': 10.0, 'P': 2000.0})
        '''
        return self.to_TP_zs(T, P, zs=self.zs)

    def to_TV(self, T, V):
        r'''Method to construct a new :obj:`GCEOSMIX` object at the spcified `T` and `V`
        with the current composition. In the event the `T` and `V` match the
        current object's `T` and `V`, it will be returned unchanged.

        Parameters
        ----------
        T : float
            Temperature, [K]
        V : float
            Molar volume, [m^3/mol]

        Returns
        -------
        obj : :obj:`GCEOSMIX`
            Pure component :obj:`GCEOSMIX` at specified `T` and `V`, [-]

        Notes
        -----
        Constructs the object with parameters `Tcs`, `Pcs`, `omegas`, and
        `kwargs`.

        Examples
        --------

        >>> base = RKMIX(T=500.0, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> new = base.to_TV(T=1000000.0, V=1.0)
        >>> base.state_specs, new.state_specs
        ({'T': 500.0, 'P': 1000000.0}, {'T': 1000000.0, 'V': 1.0})
        '''
        if T == self.T and V == self.V:
            return self
        return self.__class__(T=T, V=V, zs=self.zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, fugacities=True, **self.kwargs)

    def to_PV(self, P, V):
        r'''Method to construct a new :obj:`GCEOSMIX` object at the spcified `P` and `V`
        with the current composition. In the event the `P` and `V` match the
        current object's `P` and `V`, it will be returned unchanged.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]

        Returns
        -------
        obj : :obj:`GCEOSMIX`
            Pure component :obj:`GCEOSMIX` at specified `P` and `V`, [-]

        Notes
        -----
        Constructs the object with parameters `Tcs`, `Pcs`, `omegas`, and
        `kwargs`.

        Examples
        --------

        >>> base = RKMIX(T=500.0, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> new = base.to_PV(P=1000000.0, V=1.0)
        >>> base.state_specs, new.state_specs
        ({'T': 500.0, 'P': 1000000.0}, {'P': 1000000.0, 'V': 1.0})
        '''
        if V == self.V and P == self.P:
            return self
        return self.__class__(V=V, P=P, zs=self.zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, fugacities=True, **self.kwargs)

    def to_mechanical_critical_point(self):
        r'''Method to construct a new :obj:`GCEOSMIX` object at the current object's
        properties and composition, but which is at the mechanical critical
        point.

        Returns
        -------
        obj : :obj:`GCEOSMIX`
            Pure component :obj:`GCEOSMIX` at mechanical critical point [-]

        Examples
        --------
        >>> base = RKMIX(T=500.0, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> base.to_mechanical_critical_point()
        RKMIX(Tcs=[126.1, 190.6], Pcs=[3394000.0, 4604000.0], omegas=[0.04, 0.011], kijs=[[0.0, 0.0], [0.0, 0.0]], zs=[0.6, 0.4], T=151.861, P=3908737.9)
        '''
        T, P = self.mechanical_critical_point()
        return self.to_TP_zs(T=T, P=P, zs=self.zs)

    def to_TPV_pure(self, i, T=None, P=None, V=None):
        r'''Helper method which returns a pure `EOSs` at the specs (two of `T`,
        `P` and `V`) and base EOS as the mixture for a particular index.

        Parameters
        ----------
        i : int
            Index of specified compound, [-]
        T : float or None, optional
            Specified temperature, [K]
        P : float or None, optional
            Specified pressure, [Pa]
        V : float or None, optional
            Specified volume, [m^3/mol]

        Returns
        -------
        eos_pure : eos
            A pure-species EOSs at the two specified  `T`, `P`, and `V` for
            component `i`, [-]

        Notes
        -----
        '''
        kwargs = {}
        mix_kwargs_to_pure = self.mix_kwargs_to_pure
        for k, v in self.kwargs.items():
            if k in mix_kwargs_to_pure:
                kwargs[mix_kwargs_to_pure[k]] = v[i]
        return self.eos_pure(T=T, P=P, V=V, Tc=self.Tcs[i], Pc=self.Pcs[i],
                             omega=self.omegas[i], **kwargs)

    def pures(self):
        r'''Helper method which returns a list of pure `EOSs` at the same `T`
        and `P` and base EOS as the mixture.

        Returns
        -------
        eos_pures : list[eos]
            A list of pure-species EOSs at the same `T` and `P` as the system,
            [-]

        Notes
        -----
        This is useful for i.e. comparing mixture fugacities with the
        Lewis-Randall rule or when using an activity coefficient model which
        require pure component fugacities.
        '''
        T, P, N = self.T, self.P, self.N
        return [self.to_TPV_pure(T=T, P=P, V=None, i=i) for i in range(N)]


    @property
    def pseudo_Tc(self):
        '''Apply a linear mole-fraction mixing rule to compute the average
        critical temperature, [K].

        Examples
        --------
        >>> base = RKMIX(T=150.0, P=4e6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> base.pseudo_Tc
        151.9
        '''
        zs = self.zs
        Tcs = self.Tcs
        Tc = 0.0
        for i in range(self.N):
            Tc += zs[i]*Tcs[i]
        return Tc

    @property
    def pseudo_Pc(self):
        '''Apply a linear mole-fraction mixing rule to compute the average
        critical pressure, [Pa].

        Examples
        --------
        >>> base = RKMIX(T=150.0, P=4e6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> base.pseudo_Pc
        3878000.0
        '''
        zs = self.zs
        Pcs = self.Pcs
        Pc = 0.0
        for i in range(self.N):
            Pc += zs[i]*Pcs[i]
        return Pc

    @property
    def pseudo_omega(self):
        '''Apply a linear mole-fraction mixing rule to compute the average
        `omega`, [-].

        Examples
        --------
        >>> base = RKMIX(T=150.0, P=4e6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> base.pseudo_omega
        0.0284
        '''
        zs = self.zs
        omegas = self.omegas
        omega = 0.0
        for i in range(self.N):
            omega += zs[i]*omegas[i]
        return omega

    @property
    def pseudo_a(self):
        '''Apply a linear mole-fraction mixing rule to compute the average
        `a` coefficient, [-].

        Examples
        --------
        >>> base = RKMIX(T=150.0, P=4e6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.6, 0.4])
        >>> base.pseudo_a
        0.17634464184
        '''
        zs = self.zs
        ais = self.ais
        a = 0.0
        for i in range(self.N):
            a += zs[i]*ais[i]
        return a

    def Psat(self, T, polish=False):
        r'''Generic method to calculate vapor pressure of a pure-component
        equation of state for a specified `T`. An explicit solution is used
        unless `polish` is True. 
        
        The result of this function has no physical meaning for multicomponent
        mixtures, and does not represent either a dew point or a bubble point!

        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to attempt to use a numerical solver to make the solution
            more precise or not

        Returns
        -------
        Psat : float
            Vapor pressure using the pure-component approach, [Pa]

        Notes
        -----
        For multicomponent mixtures this may serve as a useful guess
        for the dew and the bubble pressure.

        '''
        if self.N == 1:
            Tc, Pc, omega, a = self.Tcs[0], self.Pcs[0], self.omegas[0], self.ais[0]
        else:
            zs = self.zs
            Tcs, Pcs, omegas, ais = self.Tcs, self.Pcs, self.omegas, self.ais
            Tc, Pc, omega, a = 0.0, 0.0, 0.0, 0.0
            for i in range(self.N):
                Tc += Tcs[i]*zs[i]
                Pc += Pcs[i]*zs[i]
                omega += omegas[i]*zs[i]
                a += ais[i]*zs[i]
        self.Tc, self.Pc, self.omega = Tc, Pc, omega
        self.a = a

        Psat = GCEOS.Psat(self, T, polish=False)
        del self.Tc, self.Pc, self.omega
        return Psat

    def a_alpha_and_derivatives(self, T, full=True, quick=True,
                                pure_a_alphas=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for an EOS with the Van der Waals mixing rules. Uses the
        parent class's interface to compute pure component values. Returns
        `a_alpha`, `da_alpha_dT`, and `d2a_alpha_dT2`.

        For use in :obj:`solve_T <GCEOSMIX.solve_T>` this returns only
        `a_alpha` if `full` is False.

        .. math::
            a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

        .. math::
            (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

        Parameters
        ----------
        T : float
            Temperature, [K]
        full : bool, optional
            If False, calculates and returns only `a_alpha`
        quick : bool, optional
            Only the quick variant is implemented; it is little faster anyhow
        pure_a_alphas : bool, optional
            Whether or not to recalculate the a_alpha terms of pure components
            (for the case of mixtures only) which stay the same as the
            composition changes (i.e in a PT flash), [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K**2]

        Notes
        -----
        The exact expressions can be obtained with the following SymPy
        expression below, commented out for brevity.

        >>> from sympy import *  # doctest:+SKIP
        >>> kij, T = symbols('kij, T ')  # doctest:+SKIP
        >>> a_alpha_i, a_alpha_j = symbols('a_alpha_i, a_alpha_j', cls=Function)  # doctest:+SKIP
        >>> a_alpha_ij = (1-kij)*sqrt(a_alpha_i(T)*a_alpha_j(T))  # doctest:+SKIP
        >>> diff(a_alpha_ij, T)  # doctest:+SKIP
        >>> diff(a_alpha_ij, T, T)  # doctest:+SKIP
        '''
        if pure_a_alphas:
            if full:
                a_alphas, da_alpha_dTs, d2a_alpha_dT2s = self.a_alpha_and_derivatives_vectorized(T)
                self.a_alphas, self.da_alpha_dTs, self.d2a_alpha_dT2s = a_alphas, da_alpha_dTs, d2a_alpha_dT2s
            else:
                self.a_alphas = a_alphas = self.a_alphas_vectorized(T)
                da_alpha_dTs = d2a_alpha_dT2s = None
        else:
            try:
                a_alphas, da_alpha_dTs, d2a_alpha_dT2s = self.a_alphas, self.da_alpha_dTs, self.d2a_alpha_dT2s
            except:
                if full:
                    a_alphas, da_alpha_dTs, d2a_alpha_dT2s = self.a_alpha_and_derivatives_vectorized(T)
                    self.a_alphas, self.da_alpha_dTs, self.d2a_alpha_dT2s = a_alphas, da_alpha_dTs, d2a_alpha_dT2s
                else:
                    self.a_alphas = a_alphas = self.a_alphas_vectorized(T)
                    da_alpha_dTs = d2a_alpha_dT2s = None
        if not IS_PYPY and self.N > 2000:
            return self.a_alpha_and_derivatives_numpy(a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, full=full, quick=quick)
        return self.a_alpha_and_derivatives_py(a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, full=full, quick=quick)




    def a_alpha_and_derivatives_py(self, a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, full=True, quick=True):
        # For 44 components, takes 150 us in PyPy.; 95 in pythran. Much of that is type conversions.
        # 4 ms pypy for 44*4, 1.3 ms for pythran, 10 ms python with numpy
        # 2 components 1.89 pypy, pythran 1.75 us, regular python 12.7 us.
        # 10 components - regular python 148 us, 9.81 us PyPy, 8.37 pythran in PyPy (flags have no effect; 14.3 us in regular python)
        zs, kijs, N = self.zs, self.kijs, self.N

        same_T = T == self.T
        if quick:
            try:
                assert same_T
                a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = self.a_alpha_ijs, self.a_alpha_roots, self.a_alpha_ij_roots_inv
            except (AttributeError, AssertionError):
                try:
                    a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, kijs)
                except ZeroDivisionError:
                    a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent_support_zeros(a_alphas, kijs)
                self.a_alpha_ijs, self.a_alpha_roots, self.a_alpha_ij_roots_inv = a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv
        else:
            try:
                a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, kijs)
            except:
                a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent_support_zeros(a_alphas, kijs)

            if same_T:
                self.a_alpha_ijs, self.a_alpha_roots, self.a_alpha_ij_roots_inv = a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv

        if full:
            try:
                a_alpha, da_alpha_dT, d2a_alpha_dT2, a_alpha_ijs, da_alpha_dT_ijs, d2a_alpha_dT2_ijs = a_alpha_and_derivatives_full(a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, zs, kijs,
                                                                                                                                    a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv)
            except:
                if self.N == 1:
                    a_alpha, da_alpha_dT, d2a_alpha_dT2 = a_alphas[0], da_alpha_dTs[0], d2a_alpha_dT2s[0]
                    d2a_alpha_dT2_ijs, da_alpha_dT_ijs, a_alpha_ijs = [[d2a_alpha_dT2s[0]]], [[da_alpha_dTs[0]]], [[a_alphas[0]]]

            self.d2a_alpha_dT2_ijs = d2a_alpha_dT2_ijs
            self.da_alpha_dT_ijs = da_alpha_dT_ijs
            self.a_alpha_ijs = a_alpha_ijs
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
        else:
            # Priority - test, fix, and validate
            a_alpha, _, a_alpha_ijs = a_alpha_and_derivatives(a_alphas, T, zs, kijs, a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv)
            self.da_alpha_dT_ijs = []
            self.a_alpha_ijs = a_alpha_ijs
            return a_alpha








#        # DO NOT REMOVE THIS CODE! IT MAKES TIHNGS SLOWER IN PYPY, even though it never runs
#        cmps = self.cmps
#        da_alpha_dT, d2a_alpha_dT2 = 0.0, 0.0
#
#        a_alpha_ijs = [[None]*N for _ in cmps]
#        a_alpha_roots = [a_alpha_i**0.5 for a_alpha_i in a_alphas]
#
#        if full:
#            a_alpha_ij_roots = [[None]*N for _ in cmps]
#            for i in cmps:
#                kijs_i = kijs[i]
#                a_alpha_i = a_alphas[i]
#                a_alpha_ijs_is = a_alpha_ijs[i]
#                a_alpha_ij_roots_i = a_alpha_ij_roots[i]
#                for j in cmps:
#                    if j < i:
#                        continue
#                    a_alpha_ij_roots_i[j] = a_alpha_roots[i]*a_alpha_roots[j]#(a_alpha_i*a_alphas[j])**0.5
#                    a_alpha_ijs_is[j] = a_alpha_ijs[j][i] = (1. - kijs_i[j])*a_alpha_ij_roots_i[j]
#        else:
#            for i in cmps:
#                kijs_i = kijs[i]
#                a_alpha_i = a_alphas[i]
#                a_alpha_ijs_is = a_alpha_ijs[i]
#                for j in cmps:
#                    if j < i:
#                        continue
#                    a_alpha_ijs_is[j] = a_alpha_ijs[j][i] = (1. - kijs_i[j])*a_alpha_roots[i]*a_alpha_roots[j]
#
#        # Faster than an optimized loop in pypy even
#        z_products = [[zs[i]*zs[j] for j in cmps] for i in cmps]
#
#        a_alpha = 0.0
#        for i in cmps:
#            a_alpha_ijs_i = a_alpha_ijs[i]
#            z_products_i = z_products[i]
#            for j in cmps:
#                if j < i:
#                    continue
#                elif i != j:
#                    a_alpha += 2.0*a_alpha_ijs_i[j]*z_products_i[j]
#                else:
#                    a_alpha += a_alpha_ijs_i[j]*z_products_i[j]
#
#        # List comprehension tested to be faster in CPython not pypy
##        a_alpha = sum([a_alpha_ijs[i][j]*z_products[i][j]
##                      for j in self.cmps for i in self.cmps])
#        self.a_alpha_ijs = a_alpha_ijs
#
#        da_alpha_dT_ijs = self.da_alpha_dT_ijs = [[None]*N for _ in cmps]
#
#        if full:
#            for i in cmps:
#                kijs_i = kijs[i]
#                a_alphai = a_alphas[i]
#                z_products_i = z_products[i]
#                da_alpha_dT_i = da_alpha_dTs[i]
#                d2a_alpha_dT2_i = d2a_alpha_dT2s[i]
#                a_alpha_ij_roots_i = a_alpha_ij_roots[i]
#                for j in cmps:
#                    if j < i:
#                        # skip the duplicates
#                        continue
#                    a_alphaj = a_alphas[j]
#                    x0 = a_alphai*a_alphaj
#                    x0_05 = a_alpha_ij_roots_i[j]
#                    zi_zj = z_products_i[j]
#
#                    x1 = a_alphai*da_alpha_dTs[j]
#                    x2 = a_alphaj*da_alpha_dT_i
#                    x1_x2 = x1 + x2
#                    x3 = 2.0*x1_x2
#
#                    kij_m1 = kijs_i[j] - 1.0
#
#                    da_alpha_dT_ij = -0.5*kij_m1*x1_x2/x0_05
#
#                    # For temperature derivatives of fugacities
#                    da_alpha_dT_ijs[i][j] = da_alpha_dT_ijs[j][i] = da_alpha_dT_ij
#
#                    da_alpha_dT_ij *= zi_zj
#
#                    d2a_alpha_dT2_ij = zi_zj*kij_m1*(-0.25*x0_05*(x0*(
#                    2.0*(a_alphai*d2a_alpha_dT2s[j] + a_alphaj*d2a_alpha_dT2_i)
#                    + 4.*da_alpha_dT_i*da_alpha_dTs[j]) - x1*x3 - x2*x3 + x1_x2*x1_x2)/(x0*x0))
#
#                    if i != j:
#                        da_alpha_dT += da_alpha_dT_ij + da_alpha_dT_ij
#                        d2a_alpha_dT2 += d2a_alpha_dT2_ij + d2a_alpha_dT2_ij
#                    else:
#                        da_alpha_dT += da_alpha_dT_ij
#                        d2a_alpha_dT2 += d2a_alpha_dT2_ij
#
#            return a_alpha, da_alpha_dT, d2a_alpha_dT2
#        else:
#            return a_alpha

    def a_alpha_and_derivatives_py(self, a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, full=True, quick=True):
        zs, kijs, scalar, N = self.zs, self.kijs, self.scalar, self.N
        if scalar:
            self.a_alpha_roots = a_alpha_roots = [sqrt(i) for i in a_alphas]
        else:
            self.a_alpha_roots = a_alpha_roots = npsqrt(a_alphas)
        if full:
            # Converting kijs into a matrix kills the performance! 5x slower than the performance of the functions.
            # converting the 1d arrays also takes as long as the function.
#            a_alpha, da_alpha_dT, d2a_alpha_dT2, self.a_alpha_j_rows, self.da_alpha_dT_j_rows = (
#                    a_alpha_and_derivatives_quadratic_terms(np.array(a_alphas), np.array(a_alpha_roots), np.array(da_alpha_dTs),
#                                                            np.array(d2a_alpha_dT2s), T, np.array(zs), np.array(kijs)))
            if scalar:
                a_alpha_j_rows, da_alpha_dT_j_rows = [0.0]*N, [0.0]*N
            else:
                a_alpha_j_rows, da_alpha_dT_j_rows = zeros(N), zeros(N)
            a_alpha, da_alpha_dT, d2a_alpha_dT2, self.a_alpha_j_rows, self.da_alpha_dT_j_rows = (
                    a_alpha_and_derivatives_quadratic_terms(a_alphas, a_alpha_roots, da_alpha_dTs,
                                                            d2a_alpha_dT2s, T, zs, kijs,
                                                            a_alpha_j_rows=a_alpha_j_rows, da_alpha_dT_j_rows=da_alpha_dT_j_rows))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

        else:
#            a_alpha, self.a_alpha_j_rows = a_alpha_quadratic_terms(np.array(a_alphas), np.array(a_alpha_roots), T, np.array(zs), np.array(kijs))
            a_alpha_j_rows = [0.0]*N if scalar else zeros(N)
            a_alpha, self.a_alpha_j_rows = a_alpha_quadratic_terms(a_alphas, a_alpha_roots, T, zs, kijs, a_alpha_j_rows=a_alpha_j_rows)
            return a_alpha



    def a_alpha_and_derivatives_numpy(self, a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, full=True, quick=True):
        zs, kijs = self.zs, np.array(self.kijs)
        a_alphas = np.array(a_alphas)
        da_alpha_dTs = np.array(da_alpha_dTs)
        one_minus_kijs = 1.0 - kijs

        x0 = np.einsum('i,j', a_alphas, a_alphas)
        x0_05 = npsqrt(x0)
        a_alpha_ijs = (one_minus_kijs)*x0_05
        z_products = np.einsum('i,j', zs, zs)
        a_alpha = np.einsum('ij,ji', a_alpha_ijs, z_products)
        
        if self.scalar:
            self.a_alpha_ijs = a_alpha_ijs.tolist()
        else:
            self.a_alpha_ijs = a_alpha_ijs

        if full:
            term0 = np.einsum('j,i', a_alphas, da_alpha_dTs)
            term7 = (one_minus_kijs)/(x0_05)
            da_alpha_dT = (z_products*term7*(term0)).sum()

            term1 = -x0_05/x0*(one_minus_kijs)

            term2 = np.einsum('i, j', a_alphas, da_alpha_dTs)

            main3 = da_alpha_dTs/(2.0*a_alphas)*term2
            main4 = -np.einsum('i, j', a_alphas, d2a_alpha_dT2s)
            main6 = -0.5*np.einsum('i, j', da_alpha_dTs, da_alpha_dTs)

            # Needed for fugacity temperature derivative
            self.da_alpha_dT_ijs = (0.5*(term7)*(term2 + term0)).tolist()

            d2a_alpha_dT2 = (z_products*(term1*(main3 + main4 + main6))).sum()

            return float(a_alpha), float(da_alpha_dT), float(d2a_alpha_dT2)
        else:
            return float(a_alpha)

    def _spinodal_f(self, TPV):
        # TODO - use `self`, do not create new instance
        # Work to do - ethane', 'heptane
        # Specify V, solve P; increase V and keep going
        # After Effective utilization of equations of state for thermodynamic properties in process simulation
        '''eos = PRMIX(P=6e6, T=500, Tcs=[305.32, 540.2], Pcs=[4872000.0, 2740000.0], omegas=[0.098, 0.3457], zs=[.5, .5])
        def to_solve(T):
            return eos.to(T=T, P=eos.P, zs=eos.zs)._spinodal_f([T, eos.P])

        # Very well could be right
        eos.to(T=secant(to_solve, eos.T), P=eos.P, zs=eos.zs).rho_l  # 3004.715984610371
        '''
        T, P, V = TPV
        eos_instance = self.to(T=T, P=P, V=V, zs=self.zs)
        RT_inv = 1.0/(R*eos_instance.T)
        if eos_instance.phase == 'l/g':
            if eos_instance.G_dep_l < eos_instance.G_dep_g:
                v = eos_instance.d2nA_dninjs_Vt('l')
            else:
                v = eos_instance.d2nA_dninjs_Vt('g')
        elif eos_instance.phase == 'g':
            v = eos_instance.d2nA_dninjs_Vt('g')
        else:
            v = eos_instance.d2nA_dninjs_Vt('l')
        dGs = [[i*RT_inv for i in row] for row in v]
        return det(dGs)

    def _spinodal_at(self, T=None, P=None, V=None):
        # TODO finish
        if T is not None:
            def to_solve(V):
                return self._spinodal_f((T, None, V))

            if 1:
                from fluids.numerics import linspace
                Vs = linspace(self.b*(1+1e-7), self.b*1000, 1000)
                errs = []
                for Vi in Vs:
                    try:
                        errs.append(abs(to_solve(Vi)))
                    except:
                        errs.append(1e5)
                import matplotlib.pyplot as plt
                plt.semilogy(Vs, errs)
                plt.show()
                a = 1
        elif P is not None:
            def to_solve(V):
                return self._spinodal_f((None, P, V))
        elif V is not None:
            def to_solve(T):
                return self._spinodal_f((T, None, V))


    def _mechanical_critical_point_f_jac(self, TP):
        '''The criteria for c_goal and d_goal come from a cubic
        'roots_cubic', which uses a `f`, `g`, and `h` parameter. When all of
        them are zero, all three roots are equal. For the eos (a=1), this
        results in the following system of equations:

        from sympy import *
        a = 1
        b, c, d = symbols('b, c, d')
        f = ((3* c / a) - ((b ** 2) / (a ** 2))) / 3
        g = (((2 * (b ** 3)) / (a ** 3)) - ((9* b * c) / (a **2)) + (27 * d / a)) /27
        h = ((g ** 2) / 4 + (f ** 3) / 27)z
        solve([Eq(f, 0), Eq(g, 0), Eq(h, 0)], [b, c, d])

        The solution (sympy struggled) is:
        c = b^2/3
        d = b^3/27

        These two variables switch sign at the criteria, so they work well with
        a root finding approach.


        Derived with:

        from sympy import *
        P, T, V, R, b_eos, alpha = symbols('P, T, V, R, b_eos, alpha')
        Tc, Pc, omega = symbols('Tc, Pc, omega')
        delta, epsilon = symbols('delta, epsilon')

        a_alpha = alpha(T)

        eta = b_eos
        B = b_eos*P/(R*T)
        deltas = delta*P/(R*T)
        thetas = a_alpha*P/(R*T)**2
        epsilons = epsilon*(P/(R*T))**2
        etas = eta*P/(R*T)

        b = (deltas - B - 1)
        c = (thetas + epsilons - deltas*(B + 1))
        d = -(epsilons*(B + 1) + thetas*etas)

        c_goal = b*b/3
        d_goal = b*b*b/27

        F1 = c - c_goal
        F2 = d - d_goal

        cse([F1, F2, diff(F1, T), diff(F1, P), diff(F2, T), diff(F2, P)], optimizations='basic')



        Performance analysis:

        77% of this is getting a_alpha and da_alpha_dT.
        71% of the outer solver is getting f and this Jacobian.
        Limited results from optimizing the below code, which was derived with
        sympy.
        '''
        T, P = float(TP[0]), float(TP[1])
        b_eos, delta, epsilon = self.b, self.delta, self.epsilon
        eta = b_eos

        try:
            del self.a_alpha_ijs
            del self.a_alpha_roots
            del self.a_alpha_ij_roots_inv
        except:
            pass
        a_alpha, da_alpha_dT, _ = self.a_alpha_and_derivatives(T, full=True)


        x6 = R_inv
        x7 = 1.0/T
        x0 = a_alpha
        x1 = R_inv*R_inv
        x2 = x7*x7
        x3 = x1*x2
        x4 = P*P
        x5 = epsilon*x3*x4
        x8 = P*x6*x7
        x9 = delta*x8
        x10 = b_eos*x8
        x11 = x10 + 1.0
        x12 = x11 - x9
        x13 = x12*x12
        x14 = P*x2*x6
        x15 = da_alpha_dT
        x16 = x6*x7
        x17 = x0*x16
        x18 = 2.0*epsilon*x8
        x19 = delta*x10
        x20 = delta*x11
        x21 = b_eos - delta
        x22 = 2.0/3.0*x12*x21
        x23 = P*b_eos*x0*x1*x2
        x24 = b_eos*x5
        x25 = x11*x18
        x26 = x13*x21/9.0


        F1 = P*x0*x3 - x11*x9 - x13/3.0 + x5
        F2 = -x11*x5 + x13*x12/27.0 - b_eos*x0*x4*x6*x1*x7*x2
        dF1_dT = x14*(x15*x6 - 2.0*x17 - x18 + x19 + x20 + x22)
        dF1_dP = x16*(x17 + x18 - x19 - x20 - x22)
        dF2_dT = x14*(-P*b_eos*x1*x15*x7 + 3.0*x23 + x24 + x25 - x26)
        dF2_dP = x16*(-2.0*x23 - x24 - x25 + x26)
        return [F1, F2], [[dF1_dT, dF1_dP], [dF2_dT, dF2_dP]]


    def mechanical_critical_point(self):
        r'''Method to calculate the mechanical critical point of a mixture
        of defined composition.

        The mechanical critical point is where:

        .. math::
            \frac{\partial P}{\partial \rho}|_T =
            \frac{\partial^2 P}{\partial \rho^2}|_T =  0

        Returns
        -------
        T : float
            Mechanical critical temperature, [K]
        P : float
            Mechanical critical temperature, [Pa]

        Notes
        -----
        One useful application of the mechanical critical temperature is that
        the phase identification approach of Venkatarathnam is valid only up to
        it.

        Note that the equation of state, when solved at these conditions, will
        have fairly large (1e-3 - 1e-6) results for the derivatives; but they
        are the minimum. This is just from floating point precision.

        It can also be checked looking at the calculated molar volumes - all
        three (available with :obj:`sorted_volumes <GCEOSMIX.sorted_volumes>`) will be very close (1e-5
        difference in practice), again differing because of floating point
        error.

        The algorithm here is a custom implementation, using Newton-Raphson's
        method with the initial guesses described in [1] (mole-weighted
        critical pressure average, critical temperature average using a
        quadratic mixing rule). Normally ~4 iterations are needed to solve the
        system. It is relatively fast, as only one evaluation of `a_alpha`
        and `da_alpha_dT` are needed per call to function and its jacobian.

        References
        ----------
        .. [1] Watson, Harry A. J., and Paul I. Barton. "Reliable Flash
           Calculations: Part 3. A Nonsmooth Approach to Density Extrapolation
           and Pseudoproperty Evaluation." Industrial & Engineering Chemistry
           Research, November 11, 2017.
           https://doi.org/10.1021/acs.iecr.7b03233.
        .. [2] Mathias P. M., Boston J. F., and Watanasiri S. "Effective
           Utilization of Equations of State for Thermodynamic Properties in
           Process Simulation." AIChE Journal 30, no. 2 (June 17, 2004):
           182-86. https://doi.org/10.1002/aic.690300203.
        '''
        zs, Tcs, Pcs, N = self.zs, self.Tcs, self.Pcs, self.N
        Pmc = sum([Pcs[i]*zs[i] for i in range(N)])
        Tmc = sum([sqrt(Tcs[i]*Tcs[j])*zs[j]*zs[i] for i in range(N)
                  for j in range(N)])
        TP, iterations = newton_system(self._mechanical_critical_point_f_jac,
                                       x0=[Tmc, Pmc], jac=True, ytol=1e-10,
                                       xtol=1e-12,
                                       solve_func=solve_2_direct)
        T, P = float(TP[0]), float(TP[1])
        return T, P

    def fugacities(self, only_l=False, only_g=False):
        r'''Helper method for calculating fugacity coefficients for any
        phases present, using either the overall mole fractions for both phases
        or using specified mole fractions for each phase.

        Requires :obj:`fugacity_coefficients <GCEOSMIX.fugacity_coefficients>` to be implemented by each subclassing
        EOS.

        In addition to setting `fugacities_l` and/or `fugacities_g`, this also
        sets the fugacity coefficients `phis_l` and/or `phis_g`.

        .. math::
            \hat \phi_i^g = \frac{\hat f_i^g}{y_i P}

        .. math::
            \hat \phi_i^l = \frac{\hat f_i^l}{x_i P}

        Note that in a flash calculation, each phase requires their own EOS
        object.

        Parameters
        ----------
        only_l : bool
            When true, if there is a liquid and a vapor root, only the liquid
            root (and properties) will be set.
        only_g : bool
            When true, if there is a liquid and a vapor root, only the vapor
            root (and properties) will be set.

        Notes
        -----
        It is helpful to check that :obj:`fugacity_coefficients <GCEOSMIX.fugacity_coefficients>` has been
        implemented correctly using the following expression, from [1]_.

        .. math::
            \ln \hat \phi_i = \left[\frac{\partial (n\ln \phi)}{\partial
            n_i}\right]_{T,P,n_j,V_t}

        For reference, several expressions for fugacity of a component are as
        follows, shown in [1]_ and [2]_.

        .. math::
             \ln \hat \phi_i = \int_{0}^P\left(\frac{\hat V_i}
             {RT} - \frac{1}{P}\right)dP

        .. math::
             \ln \hat \phi_i = \int_V^\infty \left[
             \frac{1}{RT}\frac{\partial P}{ \partial n_i}
             - \frac{1}{V}\right] d V - \ln Z

        References
        ----------
        .. [1] Hu, Jiawen, Rong Wang, and Shide Mao. "Some Useful Expressions
           for Deriving Component Fugacity Coefficients from Mixture Fugacity
           Coefficient." Fluid Phase Equilibria 268, no. 1-2 (June 25, 2008):
           7-13. doi:10.1016/j.fluid.2008.03.007.
        .. [2] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
           Butterworth-Heinemann, 1985.
        '''
        P, zs, scalar = self.P, self.zs, self.scalar
        if not only_g and hasattr(self, 'V_l'):
            self.lnphis_l = lnphis_l = self.fugacity_coefficients(self.Z_l)
            if scalar:
                try:
                    self.phis_l = [exp(i) for i in lnphis_l]
                except:
                    self.phis_l = [trunc_exp(i, trunc=1e308) for i in lnphis_l]
                self.fugacities_l = [phi*x*P for phi, x in zip(self.phis_l, zs)]
            else:
                self.phis_l = phis_l = npexp(lnphis_l)
                self.fugacities_l = zs*P*phis_l

        if not only_l and hasattr(self, 'V_g'):
            self.lnphis_g = lnphis_g = self.fugacity_coefficients(self.Z_g)
            if scalar:
                try:
                    self.phis_g = phis_g = [exp(i) for i in lnphis_g]
                except:
                    self.phis_g = phis_g = [trunc_exp(i, trunc=1e308) for i in lnphis_g]
                self.fugacities_g = [phi*y*P for phi, y in zip(phis_g, zs)]
            else:
                self.phis_g = phis_g = npexp(lnphis_g)
                self.fugacities_g = zs*P*phis_g



    def _eos_lnphis_lowest_Gibbs(self):
        try:
            try:
                if self.G_dep_l < self.G_dep_g:
                    return self.lnphis_l, 'l'
                else:
                    return self.lnphis_g, 'g'
            except:
                # Only one root - take it and set the prefered other phase to be a different type
                return (self.lnphis_g, 'g') if hasattr(self, 'Z_g') else (self.lnphis_l, 'l')
        except:
            self.fugacities()
            return self._eos_fugacities_lowest_Gibbs()


    def _eos_fugacities_lowest_Gibbs(self):
        # TODO delete with property_package.py
        try:
            try:
                if self.G_dep_l < self.G_dep_g:
                    return self.fugacities_l, 'l'
                else:
                    return self.fugacities_g, 'g'
            except:
                # Only one root - take it and set the prefered other phase to be a different type
                return (self.fugacities_g, 'g') if hasattr(self, 'Z_g') else (self.fugacities_l, 'l')
        except:
            self.fugacities()
            return self._eos_fugacities_lowest_Gibbs()


    def _dphi_dn(self, zi, i, phase):
        # obsolete, should be deleted
        z_copy = list(self.zs)
        z_copy.pop(i)
        z_sum = sum(z_copy) + zi
        z_copy = [j/z_sum if j else 0 for j in z_copy]
        z_copy.insert(i, zi)

        eos = self.to_TP_zs(self.T, self.P, z_copy)
        if phase == 'g':
            return eos.phis_g[i]
        elif phase == 'l':
            return eos.phis_l[i]

    def _dfugacity_dn(self, zi, i, phase):
        # obsolete, should be deleted
        z_copy = list(self.zs)
        z_copy.pop(i)
        z_sum = sum(z_copy) + zi
        z_copy = [j/z_sum if j else 0 for j in z_copy]
        z_copy.insert(i, zi)

        eos = self.to_TP_zs(self.T, self.P, z_copy)
        if phase == 'g':
            return eos.fugacities_g[i]
        elif phase == 'l':
            return eos.fugacities_l[i]

    def _Stateva_Tsvetkov_TPDF_broken(self, Zz, Zy, zs, ys):
        # TODO: delete
        z_log_fugacity_coefficients = self.fugacity_coefficients(Zz)
        y_log_fugacity_coefficients = self.fugacity_coefficients(Zy)

        kis = []
        for yi, phi_yi, zi, phi_zi in zip(ys, y_log_fugacity_coefficients, zs, z_log_fugacity_coefficients):
            di = log(zi) + phi_zi
            try:
                ki = phi_yi + log(yi) - di
            except ValueError:
                ki = phi_yi + log(1e-200) - di
            kis.append(ki)
        kis.append(kis[0])

        tot = 0.0
        for i in range(self.N):
            t = kis[i+1] - kis[i]
            tot += t*t
        return tot

    def _d_TPD_Michelson_modified(self, Zz, Zy, zs, alphas):
        r'''Modified objective function for locating the minima of the
        Tangent Plane Distance function according to [1]_, also shown in [2]_
        [2]_. The stationary points of a system are all zeros of this function;
        so once all zeroes have been located, the stability can be evaluated
        at the stationary points only. It may be required to use multiple
        guesses to find all stationary points, and there is no method of
        confirming all points have been found.

        This method does not alter the state of the object.

        .. math::
            \frac{\partial \; TPD^*}{\partial \alpha_i} = \sqrt{Y_i} \left[
            \ln \phi_i(Y) + \ln(Y_i) - h_i\right]

        .. math::
            \alpha_i = 2 \sqrt{Y_i}

        .. math::
            d_i(z) = \ln z_i + \ln \phi_i(z)

        Parameters
        ----------
        Zz : float
            Compressibility factor of the phase undergoing stability testing,
             (`test` phase), [-]
        Zy : float
            Compressibility factor of the trial phase, [-]
        zs : list[float]
            Mole fraction composition of the phase undergoing stability
            testing  (`test` phase), [-]
        alphas : list[float]
            Twice the square root of the mole numbers of each component,
            [mol^0.5]

        Returns
        -------
        err : float
            Error in solving for stationary points according to the modified
            TPD method in [1]_, [-]

        Notes
        -----
        This method is particularly useful because it is not a constrained
        objective function. This has been verified to return the same roots as
        other stationary point methods.

        References
        ----------
        .. [1] Michelsen, Michael L. "The Isothermal Flash Problem. Part I.
           Stability." Fluid Phase Equilibria 9, no. 1 (December 1982): 1-19.
        .. [2] Qiu, Lu, Yue Wang, Qi Jiao, Hu Wang, and Rolf D. Reitz.
           "Development of a Thermodynamically Consistent, Robust and Efficient
           Phase Equilibrium Solver and Its Validations." Fuel 115 (January 1,
           2014): 1-16
        '''
        # TODO: delete
        Ys = [(alpha/2.)**2 for alpha in alphas]
        ys = normalize(Ys)
        z_log_fugacity_coefficients = self.fugacity_coefficients(Zz)
        y_log_fugacity_coefficients = self.fugacity_coefficients(Zy)
        tot = 0
        for Yi, phi_yi, zi, phi_zi in zip(Ys, y_log_fugacity_coefficients, zs, z_log_fugacity_coefficients):
            di = log(zi) + phi_zi
            if Yi != 0:
                diff = Yi**0.5*(log(Yi) + phi_yi - di)
                tot += abs(diff)
        return tot


#    def TDP_Michelsen(self, phase):
#
#        z_log_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
#        y_log_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
#        tot = 0
#        for yi, phi_yi, zi, phi_zi in zip(ys, y_log_fugacity_coefficients, zs, z_log_fugacity_coefficients):
#            hi = di = log(zi) + phi_zi # same as di
#
#            k = log(yi) + phi_yi - hi
#            # Michaelsum doesn't do the exponents.
#            Yi = exp(-k)*yi
#            tot += Yi*(log(Yi) + phi_yi - hi - 1.)
#
#        return 1. + tot

#    def TDP_Michelsen_modified(self, Zz, Zy, zs, Ys):
#        # https://www.e-education.psu.edu/png520/m17_p7.html
#        # Might as well continue
#        Ys = [abs(float(Yi)) for Yi in Ys]
#        # Ys only need to be positive
#        ys = normalize(Ys)
#
#        z_log_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
#        y_log_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
#
#        tot = 0
#        for Yi, phi_yi, yi, zi, phi_zi in zip(Ys, y_log_fugacity_coefficients, ys, zs, z_log_fugacity_coefficients):
#            hi = di = log(zi) + phi_zi # same as di
#            tot += Yi*(log(Yi) + phi_yi - di - 1.)
#        return (1. + tot)
#    # Another formulation, returns the same answers.
##            tot += yi*(log(sum(Ys)) +log(yi)+ log(phi_yi) - di - 1.)
##        return (1. + sum(Ys)*tot)*1e15


    def solve_T(self, P, V, quick=True, solution=None):
        r'''Generic method to calculate `T` from a specified `P` and `V`.
        Provides SciPy's `newton` solver, and iterates to solve the general
        equation for `P`, recalculating `a_alpha` as a function of temperature
        using :obj:`a_alpha_and_derivatives <GCEOSMIX.a_alpha_and_derivatives>` each iteration.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        quick : bool, optional
            Unimplemented, although it may be possible to derive explicit
            expressions as done for many pure-component EOS
        solution : str or None, optional
            'l' or 'g' to specify a liquid of vapor solution (if one exists);
            if None, will select a solution more likely to be real (closer to
            STP, attempting to avoid temperatures like 60000 K or 0.0001 K).

        Returns
        -------
        T : float
            Temperature, [K]
        '''
        # -4 goes back from object, GCEOS
        return super(type(self).__mro__[-3], self).solve_T(P=P, V=V, solution=solution)


    def _err_VL_jacobian(self, lnKsVF, T, P, zs, near_critical=False,
                         err_also=False, info=None):
        if info is None:
            info = []
        N = self.N
        lnKs = lnKsVF[:-1]
        Ks = [exp(lnKi) for lnKi in lnKs]
        VF = float(lnKsVF[-1])


        xs = [zi/(1.0 + VF*(Ki - 1.0)) for zi, Ki in zip(zs, Ks)]
        ys = [Ki*xi for Ki, xi in zip(Ks, xs)]

        eos_g = self.to_TP_zs_fast(T=T, P=P, zs=ys, only_g=True) #
        eos_l = self.to_TP_zs_fast(T=T, P=P, zs=xs, only_l=True) #

#        eos_g = self.to_TP_zs(T=T, P=P, zs=ys)
#        eos_l = self.to_TP_zs(T=T, P=P, zs=xs)
        if not near_critical:
#            lnphis_g = eos_g.lnphis_g
#            lnphis_l = eos_l.lnphis_l
            Z_g = eos_g.Z_g
            Z_l = eos_l.Z_l
        else:
            try:
#                lnphis_g = eos_g.lnphis_g
                Z_g = eos_g.Z_g
            except AttributeError:
#                lnphis_g = eos_g.lnphis_l
                Z_g = eos_g.Z_l
            try:
#                lnphis_l = eos_l.lnphis_l
                Z_l = eos_l.Z_l
            except AttributeError:
#                lnphis_l = eos_l.lnphis_g
                Z_l = eos_l.Z_g

        lnphis_g = eos_g.fugacity_coefficients(Z_g)
        lnphis_l = eos_l.fugacity_coefficients(Z_l)

        size = N + 1
        J = [[None]*size for i in range(size)]


#        d_lnphi_dzs_basic_num
#        d_lnphi_dxs = eos_l.d_lnphi_dzs_basic_num(Z_l, xs)
#        d_lnphi_dys = eos_g.d_lnphi_dzs_basic_num(Z_g, ys)
        d_lnphi_dxs = eos_l.dlnphis_dzs(Z_l)
        d_lnphi_dys = eos_g.dlnphis_dzs(Z_g)



#        # Handle the zeros and the ones
        # Half of this is probably wrong! Only gets set for one set of variables?
        # Numerical jacobian not good enough to tell
#        for i in range(self.N):
#            J[i][-2] = 0.0
#            J[-2][i] = 0.0

        J[N][N] = 1.0

        # Last column except last value; believed correct
        # Was not correct when compared to numerical solution
        Ksm1 = [Ki - 1.0 for Ki in Ks]
        RR_denoms_inv2 = []
        for i in range(N):
            t = 1.0 + VF*Ksm1[i]
            RR_denoms_inv2.append(1.0/(t*t))

        RR_terms = [zs[k]*Ksm1[k]*RR_denoms_inv2[k] for k in range(N)]
        for i in range(N):
            value = 0.0
            d_lnphi_dxs_i, d_lnphi_dys_i = d_lnphi_dxs[i], d_lnphi_dys[i]
            for k in range(N):
                # pretty sure indexing is right in the below expression
                value += RR_terms[k]*(d_lnphi_dxs_i[k] - Ks[k]*d_lnphi_dys_i[k])
            J[i][-1] = value
#            print(value)

#        def delta(k, j):
#            if k == j:
#                return 1.0
#            return 0.0



        # Main body - expensive to compute! Lots of elements
        # Can flip around the indexing of i, j on the d_lnphi_ds but still no fix
        # unsure of correct order!
        # Reveals bugs in d_lnphi_dxs though.
        zsKsRRinvs2 = [zs[j]*Ks[j]*RR_denoms_inv2[j] for j in range(N)]
        one_m_VF = 1.0 - VF
        for i in range(N): # to N is CORRECT/MATCHES JACOBIAN NUMERICALLY
            Ji = J[i]
            d_lnphi_dxs_is, d_lnphi_dys_is = d_lnphi_dxs[i], d_lnphi_dys[i]
            for j in range(N): # to N is CORRECT/MATCHES JACOBIAN NUMERICALLY
                value = 1.0 if i == j else 0.0
#                value = 0.0
#                value += delta(i, j)
#                print(i, j, value)
                # Maybe if i == j, can skip the bit below? Tried it once and the solver never converged
#                term = zs[j]*Ks[j]*RR_denoms_inv2[j]
                value += zsKsRRinvs2[j]*(VF*d_lnphi_dxs_is[j] + one_m_VF*d_lnphi_dys_is[j])
                Ji[j] = value

        # Last row except last value  - good, working
        # Diff of RR w.r.t each log K
        bottom_row = J[-1]
        for j in range(N):
#            value = 0.0
#            RR_l =
#            RR_l = -Ks[j]*zs[j]*VF/(1.0 + VF*(Ks[j] - 1.0))**2.0
#            RR_g = Ks[j]*(1.0 - VF)*zs[j]/(1.0 + VF*(Ks[j] - 1.0))**2.0
#            value +=  #  -RR_l
            bottom_row[j] = zsKsRRinvs2[j]*(one_m_VF) + VF*zsKsRRinvs2[j]
        # Last row except last value  - good, working
#        bottom_row = J[-1]
#        for j in range(self.N):
#            value = 0.0
#            for k in range(self.N):
#                if k == j:
#                    RR_l = -Ks[j]*zs[k]*VF/(1.0 + VF*(Ks[k] - 1.0))**2.0
#                    RR_g = Ks[j]*(1.0 - VF)*zs[k]/(1.0 + VF*(Ks[k] - 1.0))**2.0
#                    value += RR_g - RR_l
#            bottom_row[j] = value
#
        # Last value - good, working, being overwritten
        dF_ncp1_dB = 0.0
        for i in range(N):
            dF_ncp1_dB -= RR_terms[i]*Ksm1[i]
        J[-1][-1] = dF_ncp1_dB

        info[:] = VF, xs, ys, eos_l, eos_g

        if err_also:
            err_RR = Rachford_Rice_flash_error(VF, zs, Ks)
            Fs = [lnKi - lnphi_l + lnphi_g for lnphi_l, lnphi_g, lnKi in zip(lnphis_l, lnphis_g, lnKs)]
            Fs.append(err_RR)
            return Fs, J

        return J

    def _err_VL(self, lnKsVF, T, P, zs, near_critical=False, info=None):
#        import numpy as np
        # tried autograd without luck
        lnKs = lnKsVF[:-1]
#        if isinstance(lnKs, np.ndarray):
#            lnKs = lnKs.tolist()
#        Ks = np.exp(lnKs)
        Ks = [exp(lnKi) for lnKi in lnKs]
        VF = float(lnKsVF[-1])
#        VF = lnKsVF[-1]
        if info is None:
            info = []
        xs = [zi/(1.0 + VF*(Ki - 1.0)) for zi, Ki in zip(zs, Ks)]
        ys = [Ki*xi for Ki, xi in zip(Ks, xs)]

        err_RR = Rachford_Rice_flash_error(VF, zs, Ks)

        eos_g = self.to_TP_zs_fast(T=T, P=P, zs=ys, only_g=True) #
        eos_g.fugacities()
        eos_l = self.to_TP_zs_fast(T=T, P=P, zs=xs, only_l=True) #
        eos_l.fugacities()

        if not near_critical:
            lnphis_g = eos_g.lnphis_g
            lnphis_l = eos_l.lnphis_l
        else:
            try:
                lnphis_g = eos_g.lnphis_g
            except AttributeError:
                lnphis_g = eos_g.lnphis_l
            try:
                lnphis_l = eos_l.lnphis_l
            except AttributeError:
                lnphis_l = eos_l.lnphis_g
#        Fs = [fl/fg-1.0 for fl, fg in zip(fugacities_l, fugacities_g)]
        Fs = [lnKi - lnphi_l + lnphi_g for lnphi_l, lnphi_g, lnKi in zip(lnphis_l, lnphis_g, lnKs)]
        Fs.append(err_RR)
        info[:] = VF, xs, ys, eos_l, eos_g
        return Fs

    def sequential_substitution_3P(self, Ks_y, Ks_z, beta_y, beta_z=0.0,

                                   maxiter=1000,
                                   xtol=1E-13, near_critical=True,
                                   xs=None, ys=None, zs=None,
                                   trivial_solution_tol=1e-5):


        print(Ks_y, Ks_z, beta_y, beta_z)
        beta_y, beta_z, xs_new, ys_new, zs_new = Rachford_Rice_solution2(ns=self.zs,
                                                                         Ks_y=Ks_y, Ks_z=Ks_z,
                                                                         beta_y=beta_y, beta_z=beta_z)
        print(beta_y, beta_z, xs_new, ys_new, zs_new)

        Ks_y = [exp(lnphi_x - lnphi_y) for lnphi_x, lnphi_y in zip(lnphis_x, lnphis_y)]
        Ks_z = [exp(lnphi_x - lnphi_z) for lnphi_x, lnphi_z in zip(lnphis_x, lnphis_z)]

    def newton_VL(self, Ks_initial=None, maxiter=30,
                  ytol=1E-7, near_critical=True,
                  xs=None, ys=None, V_over_F=None):
        T, P, zs = self.T, self.P, self.zs
        if xs is not None and ys is not None and V_over_F is not None:
            pass
        else:
            if Ks_initial is None:
                Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
            else:
                Ks = Ks_initial
            V_over_F, xs, ys = flash_inner_loop(zs, Ks)




        lnKs_guess = [log(yi/xi) for yi, xi in zip(ys, xs)] + [V_over_F]

        info = []
        def err_and_jacobian(lnKs_guess):
            err =  self._err_VL_jacobian(lnKs_guess, T, P, zs, near_critical=True, err_also=True, info=info)
#            print(lnKs_guess[-1], err[0])
            return err

        ans, count = newton_system(err_and_jacobian, jac=True, x0=lnKs_guess, ytol=ytol, maxiter=maxiter)
        V_over_F, xs, ys, eos_l, eos_g = info
        return V_over_F, xs, ys, eos_l, eos_g



    def broyden2_VL(self, Ks_initial=None, maxiter=30,
                  ytol=1E-7, xtol=1e-8, near_critical=True,
                  xs=None, ys=None, V_over_F=None):
        T, P, zs = self.T, self.P, self.zs
        if xs is not None and ys is not None and V_over_F is not None:
            pass
        else:
            if Ks_initial is None:
                Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
            else:
                Ks = Ks_initial
            V_over_F, xs, ys = flash_inner_loop(zs, Ks)

        lnKs_guess = [log(yi/xi) for yi, xi in zip(ys, xs)] + [V_over_F]

        info = []
        def err_and_jacobian(lnKs_guess):
            err =  self._err_VL_jacobian(lnKs_guess, T, P, zs, near_critical=near_critical, err_also=True, info=info)
#            print(lnKs_guess[-1], err[0])
            return err[0], err[1]
        def err(lnKs_guess):
            err = self._err_VL(lnKs_guess, T, P, zs, near_critical=near_critical, info=info)
#            print(lnKs_guess[-1], err[0])
            return err

        ans, count = broyden2(fun=err, jac=err_and_jacobian, xs=lnKs_guess, xtol=xtol, maxiter=maxiter, jac_has_fun=True, skip_J=True)
        V_over_F, xs, ys, eos_l, eos_g = info
        return V_over_F, xs, ys, eos_l, eos_g, count

    def sequential_substitution_VL(self, Ks_initial=None, maxiter=1000,
                                   xtol=1E-13, near_critical=True, Ks_extra=None,
                                   xs=None, ys=None, trivial_solution_tol=1e-5, info=None,
                                   full_alphas=False):
#        print(self.zs, Ks)
        T, P, zs = self.T, self.P, self.zs
        V_over_F = None
        if xs is not None and ys is not None:
            pass
        else:
            # TODO use flash_wilson here
            if Ks_initial is None:
                Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
            else:
                Ks = Ks_initial
            xs = None
            try:
                V_over_F, xs, ys = flash_inner_loop(zs, Ks)
            except ValueError as e:
                if Ks_extra is not None:
                    for Ks in Ks_extra:
                        try:
                            V_over_F, xs, ys = flash_inner_loop(zs, Ks)
                            break
                        except ValueError as e:
                            pass
            if xs is None:
                raise(e)

#        print(xs, ys, 'innerloop')
#        Z_l_prev = None
#        Z_g_prev = None

        for i in range(maxiter):
            if not near_critical:
                eos_g = self.to_TP_zs_fast(T=T, P=P, zs=ys, only_l=False, only_g=True, full_alphas=full_alphas)
                eos_l = self.to_TP_zs_fast(T=T, P=P, zs=xs, only_l=True, only_g=False, full_alphas=full_alphas)
                lnphis_g = eos_g.fugacity_coefficients(eos_g.Z_g)
                lnphis_l = eos_l.fugacity_coefficients(eos_l.Z_l)
            else:
                eos_g = self.to_TP_zs_fast(T=T, P=P, zs=ys, only_l=False, only_g=True, full_alphas=full_alphas)
                eos_l = self.to_TP_zs_fast(T=T, P=P, zs=xs, only_l=True, only_g=False, full_alphas=full_alphas)
                try:
                    lnphis_g = eos_g.fugacity_coefficients(eos_g.Z_g)
                except AttributeError:
                    lnphis_g = eos_g.fugacity_coefficients(eos_g.Z_l)
                try:
                    lnphis_l = eos_l.fugacity_coefficients(eos_l.Z_l)
                except AttributeError:
                    lnphis_l = eos_l.fugacity_coefficients(eos_l.Z_g)


#                eos_g = self.to_TP_zs(T=self.T, P=self.P, zs=ys)
#                eos_l = self.to_TP_zs(T=self.T, P=self.P, zs=xs)
#                if 0:
#                    if hasattr(eos_g, 'lnphis_g') and hasattr(eos_g, 'lnphis_l'):
#                        if Z_l_prev is not None and Z_g_prev is not None:
#                            if abs(eos_g.Z_g - Z_g_prev) < abs(eos_g.Z_l - Z_g_prev):
#                                lnphis_g = eos_g.lnphis_g
#                                fugacities_g = eos_g.fugacities_g
#                                Z_g_prev = eos_g.Z_g
#                            else:
#                                lnphis_g = eos_g.lnphis_l
#                                fugacities_g = eos_g.fugacities_l
#                                Z_g_prev = eos_g.Z_l
#                        else:
#                            if eos_g.G_dep_g < eos_g.lnphis_l:
#                                lnphis_g = eos_g.lnphis_g
#                                fugacities_g = eos_g.fugacities_g
#                                Z_g_prev = eos_g.Z_g
#                            else:
#                                lnphis_g = eos_g.lnphis_l
#                                fugacities_g = eos_g.fugacities_l
#                                Z_g_prev = eos_g.Z_l
#                    else:
#                        try:
#                            lnphis_g = eos_g.lnphis_g#fugacity_coefficients(eos_g.Z_g, ys)
#                            fugacities_g = eos_g.fugacities_g
#                            Z_g_prev = eos_g.Z_g
#                        except AttributeError:
#                            lnphis_g = eos_g.lnphis_l#fugacity_coefficients(eos_g.Z_l, ys)
#                            fugacities_g = eos_g.fugacities_l
#                            Z_g_prev = eos_g.Z_l
#                    if hasattr(eos_l, 'lnphis_g') and hasattr(eos_l, 'lnphis_l'):
#                        if Z_l_prev is not None and Z_g_prev is not None:
#                            if abs(eos_l.Z_l - Z_l_prev) < abs(eos_l.Z_g - Z_l_prev):
#                                lnphis_l = eos_l.lnphis_g
#                                fugacities_l = eos_l.fugacities_g
#                                Z_l_prev = eos_l.Z_g
#                            else:
#                                lnphis_l = eos_l.lnphis_l
#                                fugacities_l = eos_l.fugacities_l
#                                Z_l_prev = eos_l.Z_l
#                        else:
#                            if eos_l.G_dep_g < eos_l.lnphis_l:
#                                lnphis_l = eos_l.lnphis_g
#                                fugacities_l = eos_l.fugacities_g
#                                Z_l_prev = eos_l.Z_g
#                            else:
#                                lnphis_l = eos_l.lnphis_l
#                                fugacities_l = eos_l.fugacities_l
#                                Z_l_prev = eos_l.Z_l
#                    else:
#                        try:
#                            lnphis_l = eos_l.lnphis_g#fugacity_coefficients(eos_l.Z_g, ys)
#                            fugacities_l = eos_l.fugacities_g
#                            Z_l_prev = eos_l.Z_g
#                        except AttributeError:
#                            lnphis_l = eos_l.lnphis_l#fugacity_coefficients(eos_l.Z_l, ys)
#                            fugacities_l = eos_l.fugacities_l
#                            Z_l_prev = eos_l.Z_l
#                elif 0:
#                    if hasattr(eos_g, 'lnphis_g') and hasattr(eos_g, 'lnphis_l'):
#                        if eos_g.G_dep_g < eos_g.lnphis_l:
#                            lnphis_g = eos_g.lnphis_g
#                            fugacities_g = eos_g.fugacities_g
#                        else:
#                            lnphis_g = eos_g.lnphis_l
#                            fugacities_g = eos_g.fugacities_l
#                    else:
#                        try:
#                            lnphis_g = eos_g.lnphis_g#fugacity_coefficients(eos_g.Z_g, ys)
#                            fugacities_g = eos_g.fugacities_g
#                        except AttributeError:
#                            lnphis_g = eos_g.lnphis_l#fugacity_coefficients(eos_g.Z_l, ys)
#                            fugacities_g = eos_g.fugacities_l
#
#                    if hasattr(eos_l, 'lnphis_g') and hasattr(eos_l, 'lnphis_l'):
#                        if eos_l.G_dep_g < eos_l.lnphis_l:
#                            lnphis_l = eos_l.lnphis_g
#                            fugacities_l = eos_l.fugacities_g
#                        else:
#                            lnphis_l = eos_l.lnphis_l
#                            fugacities_l = eos_l.fugacities_l
#                    else:
#                        try:
#                            lnphis_l = eos_l.lnphis_g#fugacity_coefficients(eos_l.Z_g, ys)
#                            fugacities_l = eos_l.fugacities_g
#                        except AttributeError:
#                            lnphis_l = eos_l.lnphis_l#fugacity_coefficients(eos_l.Z_l, ys)
#                            fugacities_l = eos_l.fugacities_l
#
#                else:
#            print(phis_l, phis_g, 'phis')
            Ks = [exp(l - g) for l, g in zip(lnphis_l, lnphis_g)] # K_value(phi_l=l, phi_g=g)
#            print(Ks)
            # Hack - no idea if this will work
#            maxK = max(Ks)
#            if maxK < 1:
#                Ks[Ks.index(maxK)] = 1.1
#            minK = min(Ks)
#            if minK >= 1:
#                Ks[Ks.index(minK)] = .9


#            print(Ks, 'Ks into RR')
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F)
#            if any(i < 0 for i in xs_new):
#                print('hil', xs_new)
#
#            if any(i < 0 for i in ys_new):
#                print('hig', ys_new)

            for xi in xs_new:
                if xi < 0.0:
                    xs_new_sum = sum(abs(i) for i in xs_new)
                    xs_new = [abs(i)/xs_new_sum for i in xs_new]
                    break
            for yi in ys_new:
                if yi < 0.0:
                    ys_new_sum = sum(abs(i) for i in ys_new)
                    ys_new = [abs(i)/ys_new_sum for i in ys_new]
                    break

            # Claimed error function in CONVENTIONAL AND RAPID FLASH CALCULATIONS FOR THE SOAVE-REDLICH-KWONG AND PENG-ROBINSON EQUATIONS OF STATE

            err3 = 0.0
            # Suggested tolerance 1e-15
            for Ki, xi, yi in zip(Ks, xs, ys):
                # equivalent of fugacity ratio
                # Could divide by the old Ks as well.
                err_i = Ki*xi/yi - 1.0
                err3 += err_i*err_i
                # or use absolute for tolerance...

#            err2 = sum([(exp(l-g)-1.0)**2  ])
#            err2 = 0.0
#            for l, g in zip(fugacities_l, fugacities_g):
#                err_i = (l/g-1.0)
#                err2 += err_i*err_i
           # Suggested tolerance 1e-15
            # This is a better metric because it does not involve  hysterisis
#            print(err3, err2)

#            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
#                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
#            print(err, err2)
            xs, ys = xs_new, ys_new
#            print(i, 'err', err, err2, 'xs, ys', xs, ys, 'VF', V_over_F)
            if near_critical:
                comp_difference = sum([abs(xi - yi) for xi, yi in zip(xs, ys)])
                if comp_difference < trivial_solution_tol:
                    raise ValueError("Converged to trivial condition, compositions of both phases equal")
#            print(xs)
            if err3 < xtol:
                break
            if i == maxiter-1:
                raise ValueError('End of SS without convergence')

        if info is not None:
            info[:] = (i, err3)
        return V_over_F, xs, ys, eos_l, eos_g

    def stabiliy_iteration_Michelsen(self, T, P, zs, Ks_initial=None,
                                     maxiter=20, xtol=1E-12, liq=True):
        # checks stability vs. the current zs, mole fractions
        # liq: whether adding a test liquid phase to see if is stable or not

        eos_ref = self#.to_TP_zs(T=T, P=P, zs=zs)
        # If one phase is present - use that phase as the reference phase.
        # Otherwise, consider the phase with the lowest Gibbs excess energy as
        # the stable phase
        fugacities_ref, fugacities_ref_phase = eos_ref._eos_fugacities_lowest_Gibbs()
#        print(fugacities_ref, fugacities_ref_phase, 'fugacities_ref, fugacities_ref_phase')

        if Ks_initial is None:
            Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        else:
            Ks = Ks_initial

        same_phase_count = 0.0
        for _ in range(maxiter):
            if liq:
                zs_test = [zi/Ki for zi, Ki in zip(zs, Ks)]
            else:
                zs_test = [zi*Ki for zi, Ki in zip(zs, Ks)]

            sum_zs_test = sum(zs_test)
            zs_test_normalized = [zi/sum_zs_test for zi in zs_test]
#            if liq:
#                print(zs_test_normalized, sum_zs_test)

#            to_TP_zs_fast(self, T, P, zs, only_l=False, only_g=False)

            # IT IS NOT PERMISSIBLE TO DO ONLY ONE ROOT! 2019-03-20
            # Breaks lots of stabilities.
            eos_test = self.to_TP_zs_fast(T=T, P=P, zs=zs_test_normalized, only_l=False, only_g=False, full_alphas=False)
            fugacities_test, fugacities_phase = eos_test._eos_fugacities_lowest_Gibbs()

            if fugacities_ref_phase == fugacities_phase:
                same_phase_count += 1.0
            else:
                same_phase_count = 0

#            if liq:
#                print(fugacities_test, fugacities_ref_phase, fugacities_phase)

            if liq:
                corrections = [fi/f_ref*sum_zs_test for fi, f_ref in zip(fugacities_test, fugacities_ref)]
            else:
                corrections = [f_ref/(fi*sum_zs_test) for fi, f_ref in zip(fugacities_test, fugacities_ref)]
            Ks = [Ki*corr for Ki, corr in zip(Ks, corrections)]

            corrections_minus_1 = [corr - 1.0 for corr in corrections]
            err = sum([ci*ci for ci in corrections_minus_1])
#            print(err, xtol, Ks, corrections)
#            print('MM iter Ks =', Ks, 'zs', zs_test_normalized, 'MM err', err, xtol, _)
            if err < xtol:
                break
#            elif same_phase_count > 5:
#                break
            # It is possible to break if the trivial solution is being approached here also
            if _ == maxiter-1 and fugacities_ref_phase != fugacities_phase:
                raise UnconvergedError('End of stability_iteration_Michelsen without convergence')

        # Fails directly if fugacities_ref_phase == fugacities_phase
        # Fugacity error:
        # no, the fugacities are not supposed to be equal
#        err_equifugacity = 0
#        for fi, fref in zip(fugacities_test, fugacities_ref):
#            err_equifugacity += abs(fi - fref)
#        if err_equifugacity/P > 1e-3:
#            sum_zs_test = 1

        return sum_zs_test, Ks, fugacities_ref_phase == fugacities_phase

    def stability_Michelsen(self, T, P, zs, Ks_initial=None, maxiter=20,
                            xtol=1E-12, trivial_criteria=1E-4,
                            stable_criteria=1E-7):
#        print('MM starting, Ks=', Ks_initial)
        if Ks_initial is None:
            Ks = [Wilson_K_value(T, P, Tci, Pci, omega)  for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        else:
            Ks = Ks_initial

        zs_sum_g, Ks_g, phase_failure_g = self.stabiliy_iteration_Michelsen(T=T, P=P, zs=zs, Ks_initial=Ks,
                                                                            maxiter=maxiter, xtol=xtol, liq=False)
        zs_sum_l, Ks_l, phase_failure_l = self.stabiliy_iteration_Michelsen(T=T, P=P, zs=zs, Ks_initial=Ks,
                                                                            maxiter=maxiter, xtol=xtol, liq=True)

        log_Ks_g = [log(Ki) for Ki in Ks_g]
        log_Ks_l = [log(Ki) for Ki in Ks_l]

        lnK_2_tot_g = sum(log_Ki*log_Ki for log_Ki in log_Ks_g)
        lnK_2_tot_l = sum(log_Ki*log_Ki for log_Ki in log_Ks_l)

        sum_g_criteria = zs_sum_g - 1.0
        sum_l_criteria = zs_sum_l - 1.0

        trivial_g, trivial_l = False, False
        if lnK_2_tot_g < trivial_criteria:
            trivial_g = True
        if lnK_2_tot_l < trivial_criteria:
            trivial_l = True

        stable = False
#        print(Ks_l, Ks_g, 'Ks_l, Ks_g')

        # Table 4.6 Summary of Possible Phase Stability Test Results,
        # Phase Behavior, Whitson and Brule
        # There is a typo where Sl appears in the vapor column; this should be
        # liquid; as shown in https://www.e-education.psu.edu/png520/m17_p7.html
        g_pass, l_pass = False, False # pass means this phase cannot form another phase

        if phase_failure_g:
            g_pass = True
        if phase_failure_l:
            l_pass = True
        if trivial_g:
            g_pass = True
        if trivial_l:
            l_pass = True
        if sum_g_criteria < stable_criteria:
            g_pass = True
        if sum_l_criteria < stable_criteria:
            l_pass = True
#        print(l_pass, g_pass, 'l, g test show stable')



        if phase_failure_g and phase_failure_l:
            stable = True
        elif trivial_g and trivial_l:
            stable = True
        elif sum_g_criteria < stable_criteria and trivial_l:
            stable = True
        elif trivial_g and sum_l_criteria < stable_criteria:
            stable = True
        elif sum_g_criteria < stable_criteria and sum_l_criteria < stable_criteria:
            stable = True
        # These last two are custom, and it is apparent since they are bad
        # Also did not document well enough the cases they fail in
        # Disabled 2018-12-29
#        elif trivial_l and sum_l_criteria < stable_criteria:
#            stable = True
#        elif trivial_g and sum_g_criteria < stable_criteria:
#            stable = True
#        else:
#            print('lnK_2_tot_g', lnK_2_tot_g , 'lnK_2_tot_l', lnK_2_tot_l,
#                  'sum_g_criteria', sum_g_criteria, 'sum_l_criteria', sum_l_criteria)
#        print('stable', stable, 'phase_failure_g', phase_failure_g, 'phase_failure_l', phase_failure_l,
#              'sum_g_criteria', sum_g_criteria, 'sum_l_criteria', sum_l_criteria,
#              'trivial_g', trivial_g, 'trivial_l', trivial_l)


        # No need to enumerate unstable results

        if not stable: # One set may be trivial, which means the other set is approx
            # the only use used
            Ks = [K_g*K_l for K_g, K_l in zip(Ks_g, Ks_l)]
#        print('MM ended', Ks, stable, Ks_g, Ks_l)
        return stable, Ks, [Ks_g, Ks_l]


    def _V_over_F_bubble_T_inner(self, T, P, zs, maxiter=20, xtol=1E-3):
        eos_l = self.to_TP_zs(T=T, P=P, zs=zs)

        if not hasattr(eos_l, 'V_l'):
            raise ValueError('At the specified temperature, there is no liquid root')

        Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F, xs, ys = flash_inner_loop(zs, Ks)
        for i in range(maxiter):
            eos_g = self.to_TP_zs(T=T, P=P, zs=ys)

            if not hasattr(eos_g, 'V_g'):
                phis_g = eos_g.phis_l
                fugacities_g = eos_g.fugacities_l
            else:
                phis_g = eos_g.phis_g
                fugacities_g = eos_g.fugacities_g

            Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(eos_l.phis_l, phis_g)]
            V_over_F, xs, ys = flash_inner_loop(zs, Ks)
            err = sum([abs(i-j) for i, j in zip(eos_l.fugacities_l, fugacities_g)])
            if err < xtol:
                break
        if not hasattr(eos_g, 'V_g'):
            raise ValueError('At the specified temperature, the solver did not converge to a vapor root')
        return V_over_F
#        raise Exception('Could not converge to desired tolerance')

    def _V_over_F_dew_T_inner(self, T, P, zs, maxiter=20, xtol=1E-10):
        eos_g = self.to_TP_zs(T=T, P=P, zs=zs)
        if not hasattr(eos_g, 'V_g'):
            raise ValueError('At the specified temperature, there is no vapor root')

        Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F, xs, ys = flash_inner_loop(zs, Ks)
        for i in range(maxiter):
            eos_l = self.to_TP_zs(T=T, P=P, zs=xs)

            if not hasattr(eos_l, 'V_l'):
                phis_l = eos_l.phis_g
                fugacities_l = eos_l.fugacities_g
            else:
                phis_l = eos_l.phis_l
                fugacities_l = eos_l.fugacities_l


            Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(phis_l, eos_g.phis_g)]
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
            xs, ys = xs_new, ys_new
            if xtol < 1E-10:
                break
        if not hasattr(eos_l, 'V_l'):
            raise ValueError('At the specified temperature, the solver did not converge to a liquid root')
        return V_over_F-1.0
#        return abs(V_over_F-1)

    def _V_over_F_dew_T_inner_accelerated(self, T, P, zs, maxiter=20, xtol=1E-10):
        '''This is not working.
        '''
        eos_g = self.to_TP_zs(T=T, P=P, zs=zs)
        if not hasattr(eos_g, 'V_g'):
            raise ValueError('At the specified temperature, there is no vapor root')

        Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F_new, xs, ys = flash_inner_loop(zs, Ks)
        for i in range(maxiter):
            eos_l = self.to_TP_zs(T=T, P=P, zs=xs)

            if not hasattr(eos_l, 'V_l'):
                phis_l = eos_l.phis_g
                fugacities_l = eos_l.fugacities_g
            else:
                phis_l = eos_l.phis_l
                fugacities_l = eos_l.fugacities_l

            if 0.0 < V_over_F_new < 1.0 and i > 2:
                Rs = [K_value(phi_l=l, phi_g=g) for l, g in zip(phis_l, eos_g.phis_g)]
                lambdas = [(Ki - 1.0)/(Ki - Rri) for Rri, Ki in zip(Rs, Ks)]
                Ks = [Ki*Ri**lambda_i for Ki, Ri, lambda_i in zip(Ks, Rs, lambdas)]
            else:
                Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(phis_l, eos_g.phis_g)]

            V_over_F_new, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err_new = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
            xs, ys = xs_new, ys_new
            V_over_F_old = V_over_F_new
            if i == 0:
                err_old = err_new

            err_old = err_new
            if err_new < xtol:
                break
        if not hasattr(eos_l, 'V_l'):
            raise ValueError('At the specified temperature, the solver did not converge to a liquid root')
        return V_over_F_new-1.0
#        return abs(V_over_F-1)

#    def _a_alpha_j_rows(self):
##        try:
##            return self.a_alpha_j_rows
##        except:
##            pass
#        zs = self.zs
#        N = self.N
#        a_alpha_ijs = self.a_alpha_ijs
#        a_alpha_j_rows = []
#        for i in range(N):
#            l = a_alpha_ijs[i]
#            sum_term = 0.0
#            for j in range(N):
#                sum_term += zs[j]*l[j]
#            a_alpha_j_rows.append(sum_term)
#        self.a_alpha_j_rows = a_alpha_j_rows
#        return a_alpha_j_rows


    @property
    def _a_alpha_j_rows(self):
        try:
            return self.a_alpha_j_rows
        except:
            pass
        zs, N = self.zs, self.N
        a_alpha_ijs = self.a_alpha_ijs
        if self.scalar:
            a_alpha_j_rows = [0.0]*N
        else:
            a_alpha_j_rows = zeros(N)
            
        for i in range(N):
            l = a_alpha_ijs[i]
            for j in range(i):
                a_alpha_j_rows[j] += zs[i]*l[j]
                a_alpha_j_rows[i] += zs[j]*l[j]
            a_alpha_j_rows[i] += zs[i]*l[i]
        self.a_alpha_j_rows = a_alpha_j_rows
        return a_alpha_j_rows

    def _set_alpha_matrices(self):
        try:
            a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(self.a_alphas, self.kijs)
        except ZeroDivisionError:
            a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent_support_zeros(self.a_alphas, self.kijs)


        _, _, _, a_alpha_ijs, da_alpha_dT_ijs, d2a_alpha_dT2_ijs = a_alpha_and_derivatives_full(
                self.a_alphas, self.da_alpha_dTs, self.d2a_alpha_dT2s, self.T, self.zs, self.kijs,
                a_alpha_ijs, self.a_alpha_roots, a_alpha_ij_roots_inv)
        self._d2a_alpha_dT2_ijs = d2a_alpha_dT2_ijs
        self._da_alpha_dT_ijs = da_alpha_dT_ijs
        self._a_alpha_ijs = a_alpha_ijs

    @property
    def a_alpha_ijs(self):
        r'''Calculate and return the matrix
        :math:`(a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}`.

        Returns
        -------
        a_alpha_ijs : list[list[float]]
            `a_alpha` terms for each component with every other component,
            [J^2/mol^2/Pa]

        Notes
        -----
        In an earlier implementation this matrix was stored each EOS solve;
        however, allocating that much memory becomes quite expensive for large
        number of component cases and this is now calculated on-demand only.
        '''
        try:
            return self._a_alpha_ijs
        except:
            self._set_alpha_matrices()
            return self._a_alpha_ijs

    @property
    def da_alpha_dT_ijs(self):
        r'''Calculate and return the matrix for the temperature derivatives of
        the alpha terms.

        .. math::
            \frac{\partial (a\alpha)_{ij}}{\partial T} =
            \frac{\sqrt{\operatorname{a\alpha_{i}}{\left(T \right)} \operatorname{a\alpha_{j}}
            {\left(T \right)}} \left(1 - k_{ij}\right) \left(\frac{\operatorname{a\alpha_{i}}
            {\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}}{2}
            + \frac{\operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T} \operatorname{
            a\alpha_{i}}{\left(T \right)}}{2}\right)}{\operatorname{a\alpha_{i}}{\left(T \right)}
            \operatorname{a\alpha_{j}}{\left(T \right)}}

        Returns
        -------
        da_alpha_dT_ijs : list[list[float]]
            First temperature derivative of `a_alpha` terms for each component
            with every other component, [J^2/mol^2/Pa/K]

        Notes
        -----
        In an earlier implementation this matrix was stored each EOS solve;
        however, allocating that much memory becomes quite expensive for large
        number of component cases and this is now calculated on-demand only.
        '''
        try:
            return self._da_alpha_dT_ijs
        except:
            self._set_alpha_matrices()
            return self._da_alpha_dT_ijs

    @property
    def d2a_alpha_dT2_ijs(self):
        r'''Calculate and return the matrix of the second temperature
        derivatives of the alpha terms.

        .. math::
            \frac{\partial^2 (a\alpha)_{ij}}{\partial T^2} =
            - \frac{\sqrt{\operatorname{a\alpha_{i}}{\left(T \right)} \operatorname{a\alpha_{j}}
            {\left(T \right)}} \left(k_{ij} - 1\right) \left(\frac{\left(\operatorname{
            a\alpha_{i}}{\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}
            + \operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{i}}
            {\left(T \right)}\right)^{2}}{4 \operatorname{a\alpha_{i}}{\left(T \right)}
            \operatorname{a\alpha_{j}}{\left(T \right)}} - \frac{\left(\operatorname{a\alpha_{i}}
            {\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}
            + \operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T}
            \operatorname{a\alpha_{i}}{\left(T \right)}\right) \frac{d}{d T}
            \operatorname{a\alpha_{j}}{\left(T \right)}}{2 \operatorname{a\alpha_{j}}
            {\left(T \right)}} - \frac{\left(\operatorname{a\alpha_{i}}{\left(T \right)}
            \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}
            + \operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T}
            \operatorname{a\alpha_{i}}{\left(T \right)}\right) \frac{d}{d T}
            \operatorname{a\alpha_{i}}{\left(T \right)}}{2 \operatorname{a\alpha_{i}}
            {\left(T \right)}} + \frac{\operatorname{a\alpha_{i}}{\left(T \right)}
            \frac{d^{2}}{d T^{2}} \operatorname{a\alpha_{j}}{\left(T \right)}}{2}
            + \frac{\operatorname{a\alpha_{j}}{\left(T \right)} \frac{d^{2}}{d T^{2}}
            \operatorname{a\alpha_{i}}{\left(T \right)}}{2} + \frac{d}{d T}
            \operatorname{a\alpha_{i}}{\left(T \right)} \frac{d}{d T}
            \operatorname{a\alpha_{j}}{\left(T \right)}\right)}
            {\operatorname{a\alpha_{i}}{\left(T \right)} \operatorname{a\alpha_{j}}
            {\left(T \right)}}

        Returns
        -------
        d2a_alpha_dT2_ijs : list[list[float]]
            Second temperature derivative of `a_alpha` terms for each component
            with every other component, [J^2/mol^2/Pa/K^2]

        Notes
        -----
        In an earlier implementation this matrix was stored each EOS solve;
        however, allocating that much memory becomes quite expensive for large
        number of component cases and this is now calculated on-demand only.
        '''
        try:
            return self._d2a_alpha_dT2_ijs
        except:
            self._set_alpha_matrices()
            return self._d2a_alpha_dT2_ijs

    @property
    def _da_alpha_dT_j_rows(self):
        try:
            return self.da_alpha_dT_j_rows
        except:
            pass
        zs, N, scalar = self.zs, self.N, self.N
        da_alpha_dT_ijs = self.da_alpha_dT_ijs

        # Handle the case of attempting to avoid a full alpha derivative matrix evaluation
        if not da_alpha_dT_ijs:
            self.resolve_full_alphas()
            da_alpha_dT_ijs = self.da_alpha_dT_ijs

        if scalar:
            da_alpha_dT_j_rows = [0.0]*N
        else:
            da_alpha_dT_j_rows = zeros(N)

        for i in range(N):
            l = da_alpha_dT_ijs[i]
            for j in range(i):
                da_alpha_dT_j_rows[j] += zs[i]*l[j]
                da_alpha_dT_j_rows[i] += zs[j]*l[j]
            da_alpha_dT_j_rows[i] += zs[i]*l[i]

        self.da_alpha_dT_j_rows = da_alpha_dT_j_rows
        return da_alpha_dT_j_rows

    @property
    def _d2a_alpha_dT2_j_rows(self):
        try:
            return self.d2a_alpha_dT2_j_rows
        except AttributeError:
            pass
        d2a_alpha_dT2_ijs, N, scalar = self.d2a_alpha_dT2_ijs, self.N, self.scalar

        # Handle the case of attempting to avoid a full alpha derivative matrix evaluation
        if d2a_alpha_dT2_ijs is None:
            self.resolve_full_alphas()
            d2a_alpha_dT2_ijs = self.d2a_alpha_dT2_ijs

        zs = self.zs
        if scalar:
            d2a_alpha_dT2_j_rows = [0.0]*N
        else:
            d2a_alpha_dT2_j_rows = zeros(N)
        for i in range(N):
            l = d2a_alpha_dT2_ijs[i]
            for j in range(i):
                d2a_alpha_dT2_j_rows[j] += zs[i]*l[j]
                d2a_alpha_dT2_j_rows[i] += zs[j]*l[j]
            d2a_alpha_dT2_j_rows[i] += zs[i]*l[i]

        self.d2a_alpha_dT2_j_rows = d2a_alpha_dT2_j_rows
        return d2a_alpha_dT2_j_rows



    @property
    def db_dzs(self):
        r'''Helper method for calculating the composition derivatives of `b`.
        Note this is independent of the phase.

        .. math::
            \left(\frac{\partial b}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = b_i

        Returns
        -------
        db_dzs : list[float]
            Composition derivative of `b` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        return self.bs

    @property
    def db_dns(self):
        r'''Helper method for calculating the mole number derivatives of `b`.
        Note this is independent of the phase.

        .. math::
            \left(\frac{\partial b}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = b_i - b

        Returns
        -------
        db_dns : list[float]
            Composition derivative of `b` of each component, [m^3/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        b = self.b
        if self.scalar:
            return [bi - b for bi in self.bs]
        else:
            return self.bs - b

    @property
    def dnb_dns(self):
        r'''Helper method for calculating the partial molar derivative of `b`.
        Note this is independent of the phase.

        .. math::
            \left(\frac{\partial n \cdot b}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = b_i

        Returns
        -------
        dnb_dns : list[float]
            Partial molar derivative of `b` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        return self.bs

    @property
    def d2b_dzizjs(self):
        r'''Helper method for calculating the second partial mole fraction
        derivatives of `b`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 b}{\partial x_i \partial x_j}
            \right)_{T, P,
            n_{k \ne i,j}} = 0

        Returns
        -------
        d2b_dzizjs : list[list[float]]
            Second mole fraction derivatives of `b` of each component,
            [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            return [[0.0]*N for i in range(N)]
        return zeros((N, N))

    @property
    def d2b_dninjs(self):
        r'''Helper method for calculating the second partial mole number
        derivatives of `b`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 b}{\partial n_i \partial n_j}\right)_{T, P,
            n_{k\ne i,k}} = 2b - b_i - b_j

        Returns
        -------
        d2b_dninjs : list[list[float]]
            Second Composition derivative of `b` of each component,
            [m^3/mol^3]

        Notes
        -----
        This derivative is checked numerically.
        '''
        bb = 2.0*self.b
        bs = self.bs
        if self.scalar:
            d2b_dninjs = []
            for bi in bs:
                d2b_dninjs.append([bb - bi - bj for bj in bs])
        else:
            N = self.N
            d2b_dninjs = full((N, N), bb)
            d2b_dninjs -= bs
            d2b_dninjs = d2b_dninjs.transpose()
            d2b_dninjs -= bs
        return d2b_dninjs

    @property
    def d3b_dzizjzks(self):
        r'''Helper method for calculating the third partial mole fraction
        derivatives of `b`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 b}{\partial x_i \partial x_j \partial x_k}
            \right)_{T, P,
            n_{k \ne i,j,k}} = 0

        Returns
        -------
        d3b_dzizjzks : list[list[list[float]]]
            Third mole fraction derivatives of `b` of each component,
            [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            return [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            return zeros((N, N, N))

    @property
    def d3b_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `b`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 b}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = 2(-3b + b_i + b_j + b_k)

        Returns
        -------
        d3b_dninjnks : list[list[list[float]]]
            Third mole number derivative of `b` of each component,
            [m^3/mol^4]

        Notes
        -----
        This derivative is checked numerically.
        '''
        bs = self.bs
        n6b = -6.0*self.b
        if self.scalar:
            bs2 = [bi + bi for bi in bs]
            d3b_dninjnks = []
            for bi2 in bs2:
                d3b_dnjnks = []
                for bj2 in bs2:
                    base = n6b + bi2 + bj2
                    d3b_dnjnks.append([base + bk2 for bk2 in bs2])
                d3b_dninjnks.append(d3b_dnjnks)
        else:
            bs2 = 2.0*self.bs
            N = self.N
            d3b_dninjnks = full((N, N, N), n6b)
            d3b_dninjnks += bs2
            d3b_dninjnks = d3b_dninjnks.transpose((2, 1, 0))
            d3b_dninjnks += bs2
            d3b_dninjnks = d3b_dninjnks.transpose((0, 2, 1))
            d3b_dninjnks += bs2
        return d3b_dninjnks

    @property
    def d3epsilon_dzizjzks(self):
        r'''Helper method for calculating the third composition derivatives
        of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \epsilon}{\partial x_i \partial x_j
            \partial x_k }\right)_{T, P, x_{m\ne i,j,k}} =  0

        Returns
        -------
        d2epsilon_dzizjzks : list[list[list[float]]]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            return [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            return zeros((N, N, N))

    @property
    def d3delta_dzizjzks(self):
        r'''Helper method for calculating the third composition derivatives
        of `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \delta}{\partial x_i \partial x_j
            \partial x_k }\right)_{T, P, x_{m\ne i,j,k}} =  0

        Returns
        -------
        d3delta_dzizjzks : list[list[list[float]]]
            Third composition derivative of `epsilon` of each component,
            [m^6/mol^5]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            return [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            return zeros((N, N, N))

    @property
    def da_alpha_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `a_alpha`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial a \alpha}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = 2 \cdot \sum_j z_{j} (1 - k_{ij}) \sqrt{ (a \alpha)_i (a \alpha)_j}

        Returns
        -------
        da_alpha_dzs : list[float]
            Composition derivative of `alpha` of each component,
            [kg*m^5/(mol^2*s^2)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            a_alpha_j_rows = self.a_alpha_j_rows
        except:
            a_alpha_j_rows = self._a_alpha_j_rows
        if self.scalar:
            return [i + i for i in a_alpha_j_rows]
        return 2.0*a_alpha_j_rows

    @property
    def da_alpha_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `a_alpha`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial a \alpha}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = 2 (-a\alpha + \sum_j z_{j} (1 - k_{ij}) \sqrt{ (a \alpha)_i (a \alpha)_j})

        Returns
        -------
        da_alpha_dns : list[float]
            Mole number derivative of `alpha` of each component,
            [kg*m^5/(mol^3*s^2)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            a_alpha_j_rows = self.a_alpha_j_rows
        except:
            a_alpha_j_rows = self._a_alpha_j_rows
        a_alpha_n_2 = -2.0*self.a_alpha
        if self.scalar:
            return [2.0*t + a_alpha_n_2 for t in a_alpha_j_rows]
        return 2.0*a_alpha_j_rows + a_alpha_n_2

    @property
    def dna_alpha_dns(self):
        r'''Helper method for calculating the partial molar derivatives of
        `a_alpha`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial a \alpha}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = 2 (-0.5 a\alpha + \sum_j z_{j} (1 - k_{ij}) \sqrt{ (a \alpha)_i (a \alpha)_j})

        Returns
        -------
        dna_alpha_dns : list[float]
            Partial molar derivative of `alpha` of each component,
            [kg*m^5/(mol^2*s^2)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            a_alpha_j_rows = self.a_alpha_j_rows
        except:
            a_alpha_j_rows = self._a_alpha_j_rows
        a_alpha = self.a_alpha
        if self.scalar:
            return [t + t - a_alpha for t in a_alpha_j_rows]
        return 2.0*a_alpha_j_rows - a_alpha

    @property
    def d2a_alpha_dzizjs(self):
        r'''Helper method for calculating the second composition derivatives of
        `a_alpha` (hessian). Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 a \alpha}{\partial x_i \partial
            x_j}\right)_{T, P, x_{k\ne i,j}}
            = 2 (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

        Returns
        -------
        d2a_alpha_dzizjs : list[float]
            Second composition derivative of `alpha` of each component,
            [kg*m^5/(mol^2*s^2)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        a_alpha_ijs = self.a_alpha_ijs
        if self.scalar:
            return [[i+i for i in row] for row in a_alpha_ijs]
        else:
            return 2.0*a_alpha_ijs

    @property
    def d2a_alpha_dninjs(self):
        r'''Helper method for calculating the second partial molar derivatives
        of `a_alpha` (hessian). Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 a \alpha}{\partial n_i \partial n_j }\right)_{T, P, n_{k\ne i,j}}
            = 2\left[3(a \alpha) + (a\alpha)_{ij}  -2 (\text{term}_{i,j})
            \right]

        .. math::
            \text{term}_{i,j} = \sum_k z_k\left((a\alpha)_{ik} + (a\alpha)_{jk}
            \right)

        .. math::
            (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}


        Returns
        -------
        d2a_alpha_dninjs : list[float]
            Second partial molar derivative of `alpha` of each component,
            [kg*m^5/(mol^4*s^2)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            a_alpha_j_rows = self.a_alpha_j_rows
        except:
            a_alpha_j_rows = self._a_alpha_j_rows
        a_alpha = self.a_alpha
        a_alpha_ijs = self.a_alpha_ijs
        N = self.N
        zs = self.zs
        a_alpha3 = 3.0*a_alpha
        
        if self.scalar:
            hessian = [[0.0]*N for _ in range(N)]
        else:
            hessian = zeros((N, N))
        for i in range(N):
            for j in range(i+1):
                if i == j:
                    term = 2.0*a_alpha_j_rows[i]
                else:
                    term = 0.0
                    for k in range(N):
                        term += zs[k]*(a_alpha_ijs[i][k] + a_alpha_ijs[j][k])

                hessian[i][j] = hessian[j][i] = 2.0*(a_alpha3 + a_alpha_ijs[i][j] -2.0*term)
#                row.append(2.0*(a_alpha3 + a_alpha_ijs[i][j] -2.0*term))
#            hessian.append(row)
        return hessian

    @property
    def d3a_alpha_dzizjzks(self):
        r'''Helper method for calculating the third composition derivatives of
        `a_alpha`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 a \alpha}{\partial x_i \partial x_j
            \partial x_k}\right)_{T, P, x_{m\ne i,j,k}}
            = 0

        Returns
        -------
        d3a_alpha_dzizjzks : list[float]
            Third composition derivative of `alpha` of each component,
            [kg*m^5/(mol^2*s^2)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        return [[[0.0]*N for _ in range(N)] for _ in range(N)]

    @property
    def d3a_alpha_dninjnks(self):
        r'''Helper method for calculating the third mole number derivatives of
        `a_alpha`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 a \alpha}{\partial n_i \partial n_j
            \partial n_k}\right)_{T, P, n_{m\ne i,j,k}}
            = 4\left(-6 (a \alpha) -  [(a \alpha)_{i,j} +  (a \alpha)_{i,k}
            +  (a \alpha)_{j,k}]
            + 3\sum_m z_m[(a \alpha)_{i,m} +  (a \alpha)_{j,m}
            +  (a \alpha)_{k,m}]\right)

        Returns
        -------
        d3a_alpha_dninjnks : list[float]
            Third mole number derivative of `alpha` of each component,
            [kg*m^5/(mol^5*s^2)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        # Seems correct across diagonal
        # Each term is of similar magnitude, so likely would notice if brokwn
        a_alpha = self.a_alpha
        a_alpha_ijs = self.a_alpha_ijs
        N = self.N
        zs = self.zs
        a_alpha6 = -6.0*a_alpha
        matrix = []
        for i in range(N):
            l = []
            for j in range(N):
                row = []
                for k in range(N):
                    mid = a_alpha_ijs[i][j] + a_alpha_ijs[i][k] + a_alpha_ijs[j][k]
                    last = sum(zs[m]*(a_alpha_ijs[i][m] + a_alpha_ijs[j][m] + a_alpha_ijs[k][m]) for m in range(N))
                    ele = 4.0*(a_alpha6 - mid + 3.0*last)
                    row.append(ele)
                l.append(row)
            matrix.append(l)
        return matrix


    @property
    def da_alpha_dT_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `da_alpha_dT`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 a \alpha}{\partial x_i \partial T}
            \right)_{P, x_{i\ne j}}
            = 2 \sum_j -z_{j} (k_{ij} - 1) (a \alpha)_i (a \alpha)_j
            \frac{\partial (a \alpha)_i}{\partial T} \frac{\partial (a \alpha)_j}{\partial T}
            \left({ (a \alpha)_i (a \alpha)_j}\right)^{-0.5}

        Returns
        -------
        da_alpha_dT_dzs : list[float]
            Composition derivative of `da_alpha_dT` of each component,
            [kg*m^5/(mol^2*s^2*K)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            da_alpha_dT_j_rows = self.da_alpha_dT_j_rows
        except:
            da_alpha_dT_j_rows = self._da_alpha_dT_j_rows
        return [i + i for i in da_alpha_dT_j_rows]

    @property
    def da_alpha_dT_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `da_alpha_dT`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 a \alpha}{\partial n_i \partial T}
            \right)_{P, n_{i\ne j}}
            = 2 \left[\sum_j -z_{j} (k_{ij} - 1) (a \alpha)_i (a \alpha)_j
            \frac{\partial (a \alpha)_i}{\partial T} \frac{\partial (a \alpha)_j}{\partial T}
            \left({ (a \alpha)_i (a \alpha)_j}\right)^{-0.5}
             - \frac{\partial a \alpha}{\partial T} \right]

        Returns
        -------
        da_alpha_dT_dns : list[float]
            Composition derivative of `da_alpha_dT` of each component,
            [kg*m^5/(mol^3*s^2*K)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            da_alpha_dT_j_rows = self.da_alpha_dT_j_rows
        except:
            da_alpha_dT_j_rows = self._da_alpha_dT_j_rows
        da_alpha_dT = self.da_alpha_dT
        return [2.0*(t - da_alpha_dT) for t in da_alpha_dT_j_rows]

    @property
    def dna_alpha_dT_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `da_alpha_dT`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 n a \alpha}{\partial n_i \partial T}
            \right)_{P, n_{i\ne j}}
            = 2 \left[\sum_j -z_{j} (k_{ij} - 1) (a \alpha)_i (a \alpha)_j
            \frac{\partial (a \alpha)_i}{\partial T} \frac{\partial (a \alpha)_j}{\partial T}
            \left({ (a \alpha)_i (a \alpha)_j}\right)^{-0.5}
             - 0.5 \frac{\partial a \alpha}{\partial T} \right]

        Returns
        -------
        dna_alpha_dT_dns : list[float]
            Composition derivative of `da_alpha_dT` of each component,
            [kg*m^5/(mol^2*s^2*K)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            da_alpha_dT_j_rows = self.da_alpha_dT_j_rows
        except:
            da_alpha_dT_j_rows = self._da_alpha_dT_j_rows
        da_alpha_dT = self.da_alpha_dT
        return [t + t - da_alpha_dT for t in da_alpha_dT_j_rows]


    @property
    def d2a_alpha_dT2_dzs(self):
        r'''Helper method for calculating the mole number derivatives of
        `d2a_alpha_dT2`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 a \alpha}{\partial z_i \partial T^2}
            \right)_{P, z_{i\ne j}}
            = \text{large expression}

        Returns
        -------
        d2a_alpha_dT2_dzs : list[float]
            Composition derivative of `d2a_alpha_dT2` of each component,
            [kg*m^5/(mol^2*s^2*K^2)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            d2a_alpha_dT2_j_rows = self.d2a_alpha_dT2_j_rows
        except:
            d2a_alpha_dT2_j_rows = self._d2a_alpha_dT2_j_rows
        return [i + i for i in d2a_alpha_dT2_j_rows]

    @property
    def d2a_alpha_dT2_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `d2a_alpha_dT2`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 a \alpha}{\partial n_i \partial T^2}
            \right)_{P, n_{i\ne j}}
            = f\left(\left(\frac{\partial^3 a\alpha}{\partial z_i \partial T^2}
            \right)_{P, z_{i\ne j}}   \right)

        Returns
        -------
        d2a_alpha_dT2_dns : list[float]
            Mole number derivative of `d2a_alpha_dT2` of each component,
            [kg*m^5/(mol^3*s^2*K^2)]

        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            d2a_alpha_dT2_j_rows = self.d2a_alpha_dT2_j_rows
        except:
            d2a_alpha_dT2_j_rows = self._d2a_alpha_dT2_j_rows
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        return [2.0*(t - d2a_alpha_dT2) for t in d2a_alpha_dT2_j_rows]

    def dV_dzs(self, Z):
        r'''Calculates the molar volume composition derivative
        (where the mole fractions do not sum to 1). Verified numerically.
        Used in many other derivatives, and for the molar volume mole number
        derivative and partial molar volume calculation.

        .. math::
            \left(\frac{\partial V}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}  =
            \frac{- R T \left(V^{2}{\left(x \right)} + V{\left(x \right)} \delta{\left(x \right)}
            + \epsilon{\left(x \right)}\right)^{3} \frac{d}{d x} b{\left(x \right)} + \left(V{\left(x \right)}
            - b{\left(x \right)}\right)^{2} \left(V^{2}{\left(x \right)} + V{\left(x \right)} \delta{\left(x \right)}
            + \epsilon{\left(x \right)}\right)^{2} \frac{d}{d x} \operatorname{a \alpha}{\left(x \right)}
            - \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} V^{3}{\left(x \right)} \operatorname{a
            \alpha}{\left(x \right)} \frac{d}{d x} \delta{\left(x \right)} - \left(V{\left(x \right)} - b{\left(x
            \right)}\right)^{2} V^{2}{\left(x \right)} \operatorname{a \alpha}{\left(x \right)} \delta{\left(x
            \right)} \frac{d}{d x} \delta{\left(x \right)} - \left(V{\left(x \right)} - b{\left(x \right)}
            \right)^{2} V^{2}{\left(x \right)} \operatorname{a \alpha}{\left(x \right)} \frac{d}{d x} \epsilon{
            \left(x \right)} - \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} V{\left(x \right)}
            \operatorname{a \alpha}{\left(x \right)} \delta{\left(x \right)} \frac{d}{d x} \epsilon{\left(x
            \right)} - \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} V{\left(x \right)} \operatorname{a
            \alpha}{\left(x \right)} \epsilon{\left(x \right)} \frac{d}{d x} \delta{\left(x \right)}
            - \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} \operatorname{a \alpha}{\left(x \right)}
            \epsilon{\left(x \right)} \frac{d}{d x} \epsilon{\left(x \right)}}{- R T \left(V^{2}{\left(x \right)}
            + V{\left(x \right)} \delta{\left(x \right)} + \epsilon{\left(x \right)}\right)^{3}
            + 2 \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} V^{3}{\left(x \right)}
            \operatorname{a \alpha}{\left(x \right)} + 3 \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2}
            V^{2}{\left(x \right)} \operatorname{a \alpha}{\left(x \right)} \delta{\left(x \right)}
            + \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} V{\left(x \right)} \operatorname{a
            \alpha}{\left(x \right)} \delta^{2}{\left(x \right)} + 2 \left(V{\left(x \right)} - b{\left(x
            \right)}\right)^{2} V{\left(x \right)} \operatorname{a \alpha}{\left(x \right)} \epsilon{\left(x
            \right)} + \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} \operatorname{a \alpha}{\left(x
            \right)} \delta{\left(x \right)} \epsilon{\left(x \right)}}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dV_dzs : float
            Molar volume composition derivatives, [m^3/mol]

        Notes
        -----
        The derivation for the derivative is performed as follows using SymPy.
        The function source code is an optimized variant created with the `cse`
        SymPy function, and hand optimized further.

        >>> from sympy import * # doctest:+SKIP
        >>> P, T, R, x = symbols('P, T, R, x') # doctest:+SKIP
        >>> V, delta, epsilon, a_alpha, b = symbols('V, delta, epsilon, a\ \\alpha, b', cls=Function) # doctest:+SKIP
        >>> CUBIC = R*T/(V(x) - b(x)) - a_alpha(x)/(V(x)*V(x) + delta(x)*V(x) + epsilon(x)) - P # doctest:+SKIP
        >>> solve(diff(CUBIC, x), Derivative(V(x), x)) # doctest:+SKIP
        [(-R*T*(V(x)**2 + V(x)*delta(x) + epsilon(x))**3*Derivative(b(x), x) + (V(x) - b(x))**2*(V(x)**2 + V(x)*delta(x) + epsilon(x))**2*Derivative(a \alpha(x), x) - (V(x) - b(x))**2*V(x)**3*a \alpha(x)*Derivative(delta(x), x) - (V(x) - b(x))**2*V(x)**2*a \alpha(x)*delta(x)*Derivative(delta(x), x) - (V(x) - b(x))**2*V(x)**2*a \alpha(x)*Derivative(epsilon(x), x) - (V(x) - b(x))**2*V(x)*a \alpha(x)*delta(x)*Derivative(epsilon(x), x) - (V(x) - b(x))**2*V(x)*a \alpha(x)*epsilon(x)*Derivative(delta(x), x) - (V(x) - b(x))**2*a \alpha(x)*epsilon(x)*Derivative(epsilon(x), x))/(-R*T*(V(x)**2 + V(x)*delta(x) + epsilon(x))**3 + 2*(V(x) - b(x))**2*V(x)**3*a \alpha(x) + 3*(V(x) - b(x))**2*V(x)**2*a \alpha(x)*delta(x) + (V(x) - b(x))**2*V(x)*a \alpha(x)*delta(x)**2 + 2*(V(x) - b(x))**2*V(x)*a \alpha(x)*epsilon(x) + (V(x) - b(x))**2*a \alpha(x)*delta(x)*epsilon(x))]
        '''
        return eos_mix_dV_dzs(self.T, self.P, Z, self.b, self.delta, self.epsilon,
                              self.a_alpha, self.db_dzs, self.ddelta_dzs,
                              self.depsilon_dzs, self.da_alpha_dzs, self.N)

    def dV_dns(self, Z):
        r'''Calculates the molar volume mole number derivatives
        (where the mole fractions sum to 1). No specific formula is implemented
        for this property - it is calculated from the mole fraction derivative.

        .. math::
            \left(\frac{\partial V}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left( \left(\frac{\partial V}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dV_dns : float
            Molar volume mole number derivatives, [m^3/mol^2]
        '''
        dV_dns = dxs_to_dns(self.dV_dzs(Z), self.zs)
        if not self.scalar:
            dV_dns = array(dV_dns)
        return dV_dns

    def dnV_dns(self, Z):
        r'''Calculates the partial molar volume of the specified phase
        No specific formula is implemented
        for this property - it is calculated from the molar
        volume mole fraction derivative.

        .. math::
            \left(\frac{\partial n V}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left( \left(\frac{\partial V}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dnV_dns : float
            Partial molar volume of the mixture of the specified phase,
            [m^3/mol]
        '''
        V = Z*R*self.T/self.P
        return dxs_to_dn_partials(self.dV_dzs(Z), self.zs, V)

    def _d2V_dij_wrapper(self, V, d_Vs, dbs, d2bs, d_epsilons, d2_epsilons,
                         d_deltas, d2_deltas, da_alphas, d2a_alphas):
        T = self.T

        x0 = V
        x3 = self.b
        x4 = x0 - x3
        x5 = self.epsilon
        x6 = x0*x0
        x7 = self.delta
        x8 = x0*x7
        x9 = x5 + x6 + x8
        x10 = self.a_alpha
        x11 = x10*x4*x4
        x12 = x0 + x0
        x13 = x9*x9
        x14 = R*T
        x17 = x4*x4*x4
        x18 = x10*x17
        x19 = 2*x18
        x22 = 4*x18
        x27 = x12*x18
        x33 = x14*x13*x9
        x34 = x33 + x33
        x37 = x19*x8
        x38 = x17*x9
        x39 = x10*x38

        hessian = []
        N = self.N
        for i in range(N):


            row = []
            for j in range(N):

                # TODO optimize this - symmetric, others
                x15 = d_epsilons[i]
                x16 = d_epsilons[j]
                x20 = x16*x19

                x21 = d_Vs[i]
                x24 = d_Vs[j]

                x23 = x21*x22
                x25 = x15*x24
                x26 = d_deltas[i]
                x28 = d_deltas[j]
                x29 = x21*x24
                x30 = 8*x18*x29
                x31 = x28*x6

                x32 = x24*x26
                x35 = x34*dbs[j]
                x36 = dbs[i]
                x40 = x38*da_alphas[i]
                x41 = x38*da_alphas[j]
                x42 = x21*x41
                x43 = x24*x40
                x44 = x21*x39

                d1 = d2_deltas[i][j] # Derivative(x7, x1, x2)
                d2 = d2a_alphas[i][j] # Derivative(x10, x1, x2)
                d3 = d2bs[i][j] # Derivative(x3, x1, x2)
                d4 = d2_epsilons[i][j] # Derivative(x5, x1, x2)

                v = ((x0*x16*x23 + x0*x22*x25 - x0*x26*x41 - x0*x28*x40
                      - x0*x39*d1 - x12*x42 - x12*x43 + x13*x17*d2 + x15*x20
                      + x15*x27*x28 - x15*x41 + x16*x26*x27 - x16*x40
                      + x19*x25*x7 + x19*x26*x31 + x19*x29*x7**2 + x20*x21*x7
                      + x21*x28*x37 + x21*x35 + x22*x32*x6 + x23*x31
                      + x24*x34*x36 - 2*x24*x44 - x28*x44 - x29*x34 + x30*x6
                      + x30*x8 + x32*x37 - x32*x39 - x33*x4*d3 - x35*x36
                      - x39*d4 - x42*x7 - x43*x7)/(x4*x9*(x11*x12 + x11*x7 - x13*x14)))
                row.append(v)

            hessian.append(row)
        return hessian

    def d2V_dzizjs(self, Z):
        r'''Calculates the molar volume second composition derivative
        (where the mole fractions do not sum to 1). Verified numerically.
        Used in many other derivatives, and for the molar volume second mole
        number derivative.

        .. math::
            \left(\frac{\partial^2 V}{\partial x_i \partial x_j}\right)_{T, P,
            x_{k \ne i,j}} = \text{run SymPy code to obtain - very long!}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        d2V_dzizjs : float
            Molar volume second composition derivatives, [m^3/mol]

        Notes
        -----
        The derivation for the derivative is performed as follows using SymPy.
        The function source code is an optimized variant created with the `cse`
        SymPy function, and hand optimized further.

        >>> from sympy import * # doctest:+SKIP
        >>> P, T, R, x1, x2 = symbols('P, T, R, x1, x2') # doctest:+SKIP
        >>> V, delta, epsilon, a_alpha, b = symbols('V, delta, epsilon, a\ \\alpha, b', cls=Function) # doctest:+SKIP
        >>> CUBIC = R*T/(V(x1, x2) - b(x1, x2)) - a_alpha(x1, x2)/(V(x1, x2)*V(x1, x2) + delta(x1, x2)*V(x1, x2) + epsilon(x1, x2)) - P # doctest:+SKIP
        >>> solve(diff(CUBIC, x1, x2), Derivative(V(x1, x2), x1, x2)) # doctest:+SKIP
        '''
        V = Z*self.T*R/self.P
        dV_dzs = self.dV_dzs(Z)

        depsilon_dzs = self.depsilon_dzs
        d2epsilon_dzizjs = self.d2epsilon_dzizjs

        ddelta_dzs = self.ddelta_dzs
        d2delta_dzizjs = self.d2delta_dzizjs

        db_dzs = self.db_dzs
        d2bs = self.d2b_dzizjs

        da_alpha_dzs = self.da_alpha_dzs
        d2a_alpha_dzizjs = self.d2a_alpha_dzizjs

        return self._d2V_dij_wrapper(V=V, d_Vs=dV_dzs, dbs=db_dzs, d2bs=d2bs,
                                     d_epsilons=depsilon_dzs, d2_epsilons=d2epsilon_dzizjs,
                                     d_deltas=ddelta_dzs, d2_deltas=d2delta_dzizjs,
                                     da_alphas=da_alpha_dzs, d2a_alphas=d2a_alpha_dzizjs)

    def d2V_dninjs(self, Z):
        r'''Calculates the molar volume second mole number derivatives
        (where the mole fractions sum to 1). No specific formula is implemented
        for this property - it is calculated from the second mole fraction
        derivatives.

        .. math::
            \left(\frac{\partial^2 V}{\partial n_i \partial n_j}\right)_{T, P,
            n_{k\ne i,j}} = f\left( \left(\frac{\partial^2 V}{\partial
            x_i\partial x_j}\right)_{T, P, x_{k\ne i,j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        d2V_dninjs : float
            Molar volume second mole number derivatives, [m^3/mol^3]
        '''
        V = Z*self.T*R/self.P
        dV_dns = self.dV_dns(Z)

        depsilon_dns = self.depsilon_dns
        d2epsilon_dninjs = self.d2epsilon_dninjs

        ddelta_dns = self.ddelta_dns
        d2delta_dninjs = self.d2delta_dninjs

        db_dns = self.db_dns
        d2bs = self.d2b_dninjs

        da_alpha_dns = self.da_alpha_dns
        d2a_alpha_dninjs = self.d2a_alpha_dninjs

        return self._d2V_dij_wrapper(V=V, d_Vs=dV_dns, dbs=db_dns, d2bs=d2bs,
                                     d_epsilons=depsilon_dns, d2_epsilons=d2epsilon_dninjs,
                                     d_deltas=ddelta_dns, d2_deltas=d2delta_dninjs,
                                     da_alphas=da_alpha_dns, d2a_alphas=d2a_alpha_dninjs)

    def dZ_dzs(self, Z):
        r'''Calculates the compressibility composition derivatives
        (where the mole fractions do not sum to 1). No specific formula is
        implemented for this property - it is calculated from the
        composition derivative of molar volume, which does have its formula
        implemented.

        .. math::
            \left(\frac{\partial Z}{\partial x_i}\right)_{T, P,
            x_{i\ne j}} = \frac{P }{RT}
            \left(\frac{\partial V}{\partial x_i}\right)_{T, P, x_{i\ne j}}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dZ_dzs : float
            Compressibility composition derivative, [-]
        '''
        factor = self.P/(self.T*R)
        return [dV*factor for dV in self.dV_dzs(Z)]

    def dZ_dns(self, Z):
        r'''Calculates the compressibility mole number derivatives
        (where the mole fractions sum to 1). No specific formula is implemented
        for this property - it is calculated from the mole fraction derivative.

        .. math::
            \left(\frac{\partial Z}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left( \left(\frac{\partial Z}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dZ_dns : float
            Compressibility number derivatives, [1/mol]
        '''
        return dxs_to_dns(self.dZ_dzs(Z), self.zs)

    def dnZ_dns(self, Z):
        r'''Calculates the partial compressibility of the specified phase
        No specific formula is implemented
        for this property - it is calculated from the compressibility
        mole fraction derivative.

        .. math::
            \left(\frac{\partial n Z}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left( \left(\frac{\partial Z}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dnZ_dns : float
            Partial compressibility of the mixture of the specified phase,
            [-]
        '''
        return dxs_to_dn_partials(self.dZ_dzs(Z), self.zs, Z)

    def dH_dep_dzs(self, Z):
        r'''Calculates the molar departure enthalpy composition derivative
        (where the mole fractions do not sum to 1). Verified numerically.
        Useful in solving for enthalpy specifications in newton-type methods,
        and forms the basis for the molar departure enthalpy mole number
        derivative and molar partial departure enthalpy.

        .. math::
            \left(\frac{\partial H_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}  =
            P \frac{d}{d x} V{\left(x \right)} + \frac{2 \left(T \frac{\partial}{\partial T}
            \operatorname{a \alpha}{\left(T,x \right)} - \operatorname{a \alpha}{\left(x
            \right)}\right) \left(- \delta{\left(x \right)} \frac{d}{d x} \delta{\left(x
            \right)} + 2 \frac{d}{d x} \epsilon{\left(x \right)}\right) \operatorname{atanh}
            {\left(\frac{2 V{\left(x \right)} + \delta{\left(x \right)}}{\sqrt{\delta^{2}
            {\left(x \right)} - 4 \epsilon{\left(x \right)}}} \right)}}{\left(\delta^{2}
            {\left(x \right)} - 4 \epsilon{\left(x \right)}\right)^{\frac{3}{2}}}
            + \frac{2 \left(T \frac{\partial}{\partial T} \operatorname{a \alpha}
            {\left(T,x \right)} - \operatorname{a \alpha}{\left(x \right)}\right)
            \left(\frac{\left(- \delta{\left(x \right)} \frac{d}{d x} \delta{\left(x
            \right)} + 2 \frac{d}{d x} \epsilon{\left(x \right)}\right) \left(2
            V{\left(x \right)} + \delta{\left(x \right)}\right)}{\left(\delta^{2}{\left(x
            \right)} - 4 \epsilon{\left(x \right)}\right)^{\frac{3}{2}}} + \frac{2
            \frac{d}{d x} V{\left(x \right)} + \frac{d}{d x} \delta{\left(x \right)}}
            {\sqrt{\delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}}}\right)}{\left(
            - \frac{\left(2 V{\left(x \right)} + \delta{\left(x \right)}\right)^{2}}{
            \delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}} + 1\right) \sqrt{
            \delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}}} + \frac{2
            \left(T \frac{\partial^{2}}{\partial x\partial T} \operatorname{a \alpha}
            {\left(T,x \right)} - \frac{d}{d x} \operatorname{a \alpha}{\left(x \right)}
            \right) \operatorname{atanh}{\left(\frac{2 V{\left(x \right)} + \delta{\left(x
            \right)}}{\sqrt{\delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}}}
            \right)}}{\sqrt{\delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}}}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dH_dep_dzs : float
            Departure enthalpy composition derivatives, [J/mol]

        Notes
        -----
        The derivation for the derivative is performed as follows using SymPy.
        The function source code is an optimized variant created with the `cse`
        SymPy function, and hand optimized further.

        >>> from sympy import * # doctest:+SKIP
        >>> P, T, V, R, b, a, delta, epsilon, x = symbols('P, T, V, R, b, a, delta, epsilon, x') # doctest:+SKIP
        >>> V, delta, epsilon, a_alpha, b = symbols('V, delta, epsilon, a_alpha, b', cls=Function) # doctest:+SKIP
        >>> H_dep = (P*V(x) - R*T + 2/sqrt(delta(x)**2 - 4*epsilon(x))*(T*Derivative(a_alpha(T, x), T) # doctest:+SKIP
        ... - a_alpha(x))*atanh((2*V(x)+delta(x))/sqrt(delta(x)**2-4*epsilon(x))))
        >>> diff(H_dep, x) # doctest:+SKIP
        P*Derivative(V(x), x) + 2*(T*Derivative(a \alpha(T, x), T) - a \alpha(x))*(-delta(x)*Derivative(delta(x), x) + 2*Derivative(epsilon(x), x))*atanh((2*V(x) + delta(x))/sqrt(delta(x)**2 - 4*epsilon(x)))/(delta(x)**2 - 4*epsilon(x))**(3/2) + 2*(T*Derivative(a \alpha(T, x), T) - a \alpha(x))*((-delta(x)*Derivative(delta(x), x) + 2*Derivative(epsilon(x), x))*(2*V(x) + delta(x))/(delta(x)**2 - 4*epsilon(x))**(3/2) + (2*Derivative(V(x), x) + Derivative(delta(x), x))/sqrt(delta(x)**2 - 4*epsilon(x)))/((-(2*V(x) + delta(x))**2/(delta(x)**2 - 4*epsilon(x)) + 1)*sqrt(delta(x)**2 - 4*epsilon(x))) + 2*(T*Derivative(a \alpha(T, x), T, x) - Derivative(a \alpha(x), x))*atanh((2*V(x) + delta(x))/sqrt(delta(x)**2 - 4*epsilon(x)))/sqrt(delta(x)**2 - 4*epsilon(x))
        '''
        P = self.P
        T = self.T
        ddelta_dzs = self.ddelta_dzs
        depsilon_dzs = self.depsilon_dzs
        da_alpha_dzs = self.da_alpha_dzs
        da_alpha_dT_dzs = self.da_alpha_dT_dzs
        dV_dzs = self.dV_dzs(Z)

        x0 = V = Z*R*T/P
        x2 = self.delta
        x3 = x0 + x0 + x2
        x4 = self.epsilon
        x5 = x2*x2 - 4.0*x4
        try:
            x6 = x5**-0.5
        except:
            # VDW has x5 as zero as delta, epsilon = 0
            x6 = 1e50
        x7 = 2.0*catanh(x3*x6).real
        x8 = x9 = self.a_alpha

        x10 = T*self.da_alpha_dT - x8
        x13 = x6*x6# 1.0/x5

        t0 = x6*x7
        t1 = x10*t0*x13
        t2 = 2.0*x10*x13/(x13*x3*x3 - 1.0)
        x3_x13 = x3*x13
        dH_dzs = []
        for i in range(self.N):
            x1 = dV_dzs[i]
            x11 = ddelta_dzs[i]
            x12 = x11*x2 - 2.0*depsilon_dzs[i]

            value = (P*x1 - x12*t1 + t2*(x12*x3_x13 - x1 - x1 - x11)
                     + t0*(T*da_alpha_dT_dzs[i] - da_alpha_dzs[i]))
            dH_dzs.append(value)
        return dH_dzs

    def dS_dep_dzs(self, Z):
        r'''Calculates the molar departure entropy composition derivative
        (where the mole fractions do not sum to 1). Verified numerically.
        Useful in solving for entropy specifications in newton-type methods,
        and forms the basis for the molar departure entropy mole number
        derivative and molar partial departure entropy.

        .. math::
            \left(\frac{\partial S_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}  = \frac{1}{T}\left(
             \left(\frac{\partial H_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
             -  \left(\frac{\partial G_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dS_dep_dzs : float
            Departure entropy composition derivatives, [J/mol/K]

        Notes
        -----
        '''
        dH_dep_dzs = self.dH_dep_dzs(Z)
        dG_dep_dzs = self.dG_dep_dzs(Z)
        T_inv = 1.0/self.T
        return [T_inv*(dH_dep_dzs[i] - dG_dep_dzs[i]) for i in range(self.N)]

    def dS_dep_dns(self, Z):
        r'''Calculates the molar departure entropy mole number derivatives
        (where the mole fractions sum to 1). No specific formula is implemented
        for this property - it is calculated from the mole fraction derivative.

        .. math::
            \left(\frac{\partial S_{dep}}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left(  \left(\frac{\partial S_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dS_dep_dns : float
            Departure entropy mole number derivatives, [J/mol^2/K]
        '''
        return dxs_to_dns(self.dS_dep_dzs(Z), self.zs)

    def dP_dns_Vt(self, phase):
        # Checked numerically, working. Evaluated at constant temperature and total volume.
        r'''from sympy import *
        Vt, P, T, R, n1, n2, n3, no = symbols('Vt, P, T, R, n1, n2, n3, no') # doctest:+SKIP
        n, P, V, a_alpha, delta, epsilon, b = symbols('n, P, V, a\ \\alpha, delta, epsilon, b', cls=Function) # doctest:+SKIP
        da_alpha_dT, d2a_alpha_dT2 = symbols('da_alpha_dT, d2a_alpha_dT2', cls=Function) # doctest:+SKIP
        n = no + n1 + n2 + n3

        P = R*T/(Vt/n-b(n1, n2, n3)) - a_alpha(T, n1, n2, n3)/((Vt/n)**2 + delta(n1, n2, n3)*(Vt/n)+epsilon(n1, n2, n3))
        V = Vt/n
        cse(diff(P, n1))

        '''
        if phase == 'g':
            Vt = self.V_g
        else:
            Vt = self.V_l

        T = self.T
        b = self.b
        a_alpha = self.a_alpha
        epsilon = self.epsilon
        Vt2 = Vt*Vt
        delta = self.delta
        x9 = Vt2 + Vt*delta + epsilon

        depsilon_dns = self.depsilon_dns
        ddelta_dns = self.ddelta_dns
        db_dns = self.db_dns
        da_alpha_dns = self.da_alpha_dns

        t1 = R*T*1.0/((Vt - b)*(Vt - b))
        t2 = 1.0/x9
        t3 = a_alpha*t2*t2
        t4 = t1*Vt -t3*(Vt*delta + Vt2 + Vt2)

        dP_dns_Vt = []
        for i in range(self.N):
            v = (t4 + t1*db_dns[i] + t3*(Vt*ddelta_dns[i] + depsilon_dns[i]) - t2*da_alpha_dns[i])
            dP_dns_Vt.append(v)
        return dP_dns_Vt


    def d2P_dninjs_Vt(self, phase):
        if phase == 'g':
            Vt = self.V_g
        else:
            Vt = self.V_l

        T, N = self.T, self.N
        b = self.b
        a_alpha = self.a_alpha
        epsilon = self.epsilon

        depsilon_dns = self.depsilon_dns
        ddelta_dns = self.ddelta_dns
        db_dns = self.db_dns
        da_alpha_dns = self.da_alpha_dns

        d2delta_dninjs = self.d2delta_dninjs
        d2epsilon_dninjs = self.d2epsilon_dninjs
        d2bs = self.d2b_dninjs
        d2a_alpha_dninjs = self.d2a_alpha_dninjs

        x0 = self.a_alpha
        x1 = self.epsilon
        x2 = Vt*Vt
        x5 = self.delta
        x7 = x1 + x2 + x5*Vt
        x7_inv = 1.0/x7
        x8 = self.b
        x9 = Vt - x8
        x11 = Vt + Vt
        x12 = R*T
        x13 = Vt
        x14 = x7_inv*x7_inv
        x16 = x2 + x2 + x13*x5

        t1 = x0*x14

        x9_inv = 1.0/x9
        x9_inv2 = x9_inv*x9_inv
        x9_inv3 = x9_inv*x9_inv2


        t2 = t1*(x11*x5 + 6.0*x2) - x12*x11*x9_inv2
        t3 = x12*x9_inv2
        t4 = 2.0*x12*x9_inv3
        t5 = 2.0*x0*x7_inv*x7_inv*x7_inv

        hess = [[0.0]*N for _ in range(N)]
        for i in range(N):
            x15 = ddelta_dns[i]
            x17 = -x15*Vt + x16 - depsilon_dns[i]

            t50 = -x13*x15
            t51 = t5*x17
            t52 = t4*(x13 + db_dns[i])
            t53 = x14*x17
            t54 = x14*da_alpha_dns[i]
            t55 = (t51 + t54)

            iadd = t1*t50 + t52*x13 - x16*t55

            for j in range(i+1):
                x18 = ddelta_dns[j]
                x19 = x18*Vt + depsilon_dns[j]

                v = (t2 + iadd + t1*(Vt*d2delta_dninjs[i][j] + d2epsilon_dninjs[i][j] - x13*x18)
                     + t52*db_dns[j] - t53*da_alpha_dns[j]  + t55*x19
                     + t3*d2bs[i][j] - x7_inv*d2a_alpha_dninjs[i][j])
                hess[i][j] = hess[j][i] = v
        return hess

    def d3P_dninjnks_Vt(self, phase):
        if phase == 'g':
            Vt = self.V_g
        else:
            Vt = self.V_l

        T, N = self.T, self.N
        b = self.b
        a_alpha = self.a_alpha
        epsilon = self.epsilon

        depsilon_dns = self.depsilon_dns
        ddelta_dns = self.ddelta_dns
        db_dns = self.db_dns
        da_alpha_dns = self.da_alpha_dns

        d2delta_dninjs = self.d2delta_dninjs
        d2epsilon_dninjs = self.d2epsilon_dninjs
        d2bs = self.d2b_dninjs
        d2a_alpha_dninjs = self.d2a_alpha_dninjs

        d3epsilon_dninjnks = self.d3epsilon_dninjnks
        d3delta_dninjnks = self.d3delta_dninjnks
        d3a_alpha_dninjnks = self.d3a_alpha_dninjnks
        d3b_dninjnks = self.d3b_dninjnks

        mat = [[[0.0]*N for _ in range(N)] for _ in range(N)]

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    x0 = self.b
                    x1 = 1.0
                    x2 = Vt/x1
                    x3 = -x0 + x2
                    x4 = 6/x1**4
                    x5 = Vt*x4
                    x6 = R*T
                    x7 = self.a_alpha
                    x8 = self.epsilon
                    x9 = Vt**2
                    x10 = x1**(-2)
                    x11 = self.delta
                    x12 = x10*x9 + x11*x2 + x8
                    x13 = 2/x1**3
                    x14 = Vt*x13
                    x15 = Vt*x10
                    x16 = x6*(x15 + db_dns[k])
                    x17 = 2/x3**3
                    x18 = x15 + db_dns[j]
                    x19 = x17*x6
                    x20 = x15 + db_dns[i]
                    x21 = x12**(-2)
                    x22 = ddelta_dns[i]
                    x23 = x11*x15 + x13*x9
                    x24 = -x2*x22 + x23 - depsilon_dns[i]
                    x25 = ddelta_dns[j]
                    x26 = -x2*x25 + x23 - depsilon_dns[j]
                    x27 = ddelta_dns[k]
                    x28 = -x2*x27 + x23 - depsilon_dns[j]
                    x29 = da_alpha_dns[k]
                    x30 = d2delta_dninjs[i][j]
                    x31 = -x15*x25
                    x32 = x4*x9
                    x33 = x11*x14
                    x34 = -x15*x22 + x32 + x33
                    x35 = x2*x30 + x31 + x34 + d2epsilon_dninjs[i][j]
                    x36 = da_alpha_dns[j]
                    x37 = d2delta_dninjs[i][k]
                    x38 = -x15*x27
                    x39 = x2*x37 + x34 + x38 + d2epsilon_dninjs[i][k]
                    x40 = da_alpha_dns[i]
                    x41 = d2delta_dninjs[j][k]
                    x42 = x2*x41 + x31 + x32 + x33 + x38 + d2epsilon_dninjs[j][k]
                    x43 = 2/x12**3
                    x44 = x24*x26
                    x45 = x28*x43
                    x46 = x43*x7

                    v = (-x16*x17*(x14 - d2bs[i][j]) + 6*x16*x18*x20/x3**4 - x18*x19*(x14 -d2bs[i][k])
                        - x19*x20*(x14 - d2bs[j][k]) - x21*x24*d2a_alpha_dninjs[j][k]
                        - x21*x26*d2a_alpha_dninjs[i][k] - x21*x28*d2a_alpha_dninjs[i][j]
                        + x21*x29*x35 + x21*x36*x39 + x21*x40*x42
                        - x21*x7*(x11*x5 - x14*x22 - x14*x25 - x14*x27 + x15*x30 + x15*x37 + x15*x41
                                 - x2*d3delta_dninjnks[i][j][k] - d3epsilon_dninjnks[i][j][k] + 24*x9/x1**5)
                        - x24*x36*x45 + x24*x42*x46 + x26*x39*x46 - x26*x40*x45 - x29*x43*x44 + x35*x45*x7
                        + x6*(x5 + d3b_dninjnks[i][j][k])/x3**2 - d3a_alpha_dninjnks[i][j][k]/x12 - 6*x28*x44*x7/x12**4)


                    mat[i][j][k] = v
        return mat






    def dH_dep_dns(self, Z):
        r'''Calculates the molar departure enthalpy mole number derivatives
        (where the mole fractions sum to 1). No specific formula is implemented
        for this property - it is calculated from the mole fraction derivative.

        .. math::
            \left(\frac{\partial H_{dep}}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left(  \left(\frac{\partial H_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dH_dep_dns : float
            Departure enthalpy mole number derivatives, [J/mol^2]
        '''
        return dxs_to_dns(self.dH_dep_dzs(Z), self.zs)

    def dnH_dep_dns(self, Z):
        r'''Calculates the partial molar departure enthalpy. No specific
        formula is implemented for this property - it is calculated from the
        mole fraction derivative.

        .. math::
            \left(\frac{\partial n H_{dep}}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left(  \left(\frac{\partial H_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dnH_dep_dns : float
            Partial molar departure enthalpies of the phase, [J/mol]
        '''
        try:
            if Z == self.Z_l:
                F = self.H_dep_l
            else:
                F = self.H_dep_g
        except:
            F = self.H_dep_g
        return dxs_to_dn_partials(self.dH_dep_dzs(Z), self.zs, F)

    def _G_dep_lnphi_d_helper(self, Z, dbs, depsilons, ddelta, dVs, da_alphas,
                              G=True):
        return G_dep_lnphi_d_helper(self.T, self.P, self.b, self.delta,
                                    self.epsilon, self.a_alpha, self.N,
                                    Z, dbs, depsilons, ddelta, dVs, da_alphas,
                                    G)

    def dlnphi_dzs(self, Z):
        r'''Calculates the mixture log *fugacity coefficient* mole fraction
        derivatives (where the mole fractions do not sum to 1). No specific
        formula is implemented for this property - it is calculated from the
        mole fraction  derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial \ln \phi }{\partial x_i}\right)_{T, P,
            x_{i\ne j}} = \frac{1}{RT}\left(  \left(\frac{\partial G_{dep}}
            {\partial x_i}\right)_{T, P, x_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dlnphi_dzs : float
            Mixture log fugacity coefficient mole fraction derivatives, [-]
        '''
        return self._G_dep_lnphi_d_helper(Z, dbs=self.db_dzs, depsilons=self.depsilon_dzs,
                                          ddelta=self.ddelta_dzs, dVs=self.dV_dzs(Z),
                                          da_alphas=self.da_alpha_dzs, G=False)

    def dlnphi_dns(self, Z):
        r'''Calculates the mixture log *fugacity coefficient* mole number
        derivatives (where the mole fractions sum to 1). No specific formula is
        implemented for this property - it is calculated from the mole fraction
        derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial \ln \phi }{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left(  \left(\frac{\partial G_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        This property can be converted into a partial molar property to obtain
        the individual fugacity coefficients.

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dlnphi_dns : float
            Mixture log fugacity coefficient mole number derivatives, [1/mol]
        '''
        return self._G_dep_lnphi_d_helper(Z, dbs=self.db_dns, depsilons=self.depsilon_dns,
                                          ddelta=self.ddelta_dns, dVs=self.dV_dns(Z),
                                          da_alphas=self.da_alpha_dns, G=False)

    def dG_dep_dzs(self, Z):
        r'''Calculates the molar departure Gibbs energy composition derivative
        (where the mole fractions do not sum to 1). Verified numerically.
        Useful in solving for gibbs minimization calculations or for solving
        for the true critical point. Also forms the basis for the molar
        departure Gibbs energy mole number derivative and molar partial
        departure Gibbs energy.

        .. math::
            \left(\frac{\partial G_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}  =
            P \frac{d}{d x} V{\left(x \right)} - \frac{R T \left(\frac{d}{d x}
            V{\left(x \right)} - \frac{d}{d x} b{\left(x \right)}\right)}{
            V{\left(x \right)} - b{\left(x \right)}} - \frac{2 \left(- \delta{
            \left(x \right)} \frac{d}{d x} \delta{\left(x \right)} + 2 \frac{d}
            {d x} \epsilon{\left(x \right)}\right) \operatorname{a \alpha}{
            \left(x \right)} \operatorname{atanh}{\left(\frac{2 V{\left(x
            \right)}}{\sqrt{\delta^{2}{\left(x \right)} - 4 \epsilon{\left(x
            \right)}}} + \frac{\delta{\left(x \right)}}{\sqrt{\delta^{2}{\left(
            x \right)} - 4 \epsilon{\left(x \right)}}} \right)}}{\left(
            \delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}\right)^{
            \frac{3}{2}}} - \frac{2 \operatorname{atanh}{\left(\frac{2 V{\left(
            x \right)}}{\sqrt{\delta^{2}{\left(x \right)} - 4 \epsilon{\left(x
            \right)}}} + \frac{\delta{\left(x \right)}}{\sqrt{\delta^{2}{\left(
            x \right)} - 4 \epsilon{\left(x \right)}}} \right)} \frac{d}{d x}
            \operatorname{a \alpha}{\left(x \right)}}{\sqrt{\delta^{2}{\left(x
            \right)} - 4 \epsilon{\left(x \right)}}} - \frac{2 \left(\frac{2
            \left(- \delta{\left(x \right)} \frac{d}{d x} \delta{\left(x
            \right)} + 2 \frac{d}{d x} \epsilon{\left(x \right)}\right)
            V{\left(x \right)}}{\left(\delta^{2}{\left(x \right)} - 4 \epsilon{
            \left(x \right)}\right)^{\frac{3}{2}}} + \frac{\left(- \delta{\left
            (x \right)} \frac{d}{d x} \delta{\left(x \right)} + 2 \frac{d}{d x}
            \epsilon{\left(x \right)}\right) \delta{\left(x \right)}}{\left(
            \delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}\right)^{
            \frac{3}{2}}} + \frac{2 \frac{d}{d x} V{\left(x \right)}}{\sqrt{
            \delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}}}
            + \frac{\frac{d}{d x} \delta{\left(x \right)}}{\sqrt{\delta^{2}{
            \left(x \right)} - 4 \epsilon{\left(x \right)}}}\right)
            \operatorname{a \alpha}{\left(x \right)}}{\left(1 - \left(\frac{2
            V{\left(x \right)}}{\sqrt{\delta^{2}{\left(x \right)} - 4 \epsilon{
            \left(x \right)}}} + \frac{\delta{\left(x \right)}}{\sqrt{
            \delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}}}\right
            )^{2}\right) \sqrt{\delta^{2}{\left(x \right)}
            - 4 \epsilon{\left(x \right)}}}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dG_dep_dzs : float
            Departure Gibbs free energy composition derivatives, [J/mol]

        Notes
        -----
        The derivation for the derivative is performed as follows using SymPy.
        The function source code is an optimized variant created with the `cse`
        SymPy function, and hand optimized further.

        >>> from sympy import * # doctest:+SKIP
        >>> P, T, R, x = symbols('P, T, R, x') # doctest:+SKIP
        >>> a_alpha, a, delta, epsilon, V, b, da_alpha_dT = symbols('a\ \\alpha, a, delta, epsilon, V, b, da_alpha_dT', cls=Function) # doctest:+SKIP
        >>> S_dep = R*log(P*V(x)/(R*T)) + R*log(V(x)-b(x))+2*da_alpha_dT(x)*atanh((2*V(x)+delta(x))/sqrt(delta(x)**2-4*epsilon(x)))/sqrt(delta(x)**2-4*epsilon(x))-R*log(V(x)) # doctest:+SKIP
        >>> H_dep = P*V(x) - R*T + 2*atanh((2*V(x)+delta(x))/sqrt(delta(x)**2-4*epsilon(x)))*(da_alpha_dT(x)*T-a_alpha(x))/sqrt(delta(x)**2-4*epsilon(x)) # doctest:+SKIP
        >>> G_dep = simplify(H_dep - T*S_dep) # doctest:+SKIP
        >>> diff(G_dep, x) # doctest:+SKIP
        P*Derivative(V(x), x) - R*T*(Derivative(V(x), x) - Derivative(b(x), x))/(V(x) - b(x)) - 2*(-delta(x)*Derivative(delta(x), x) + 2*Derivative(epsilon(x), x))*a \alpha(x)*atanh(2*V(x)/sqrt(delta(x)**2 - 4*epsilon(x)) + delta(x)/sqrt(delta(x)**2 - 4*epsilon(x)))/(delta(x)**2 - 4*epsilon(x))**(3/2) - 2*atanh(2*V(x)/sqrt(delta(x)**2 - 4*epsilon(x)) + delta(x)/sqrt(delta(x)**2 - 4*epsilon(x)))*Derivative(a \alpha(x), x)/sqrt(delta(x)**2 - 4*epsilon(x)) - 2*(2*(-delta(x)*Derivative(delta(x), x) + 2*Derivative(epsilon(x), x))*V(x)/(delta(x)**2 - 4*epsilon(x))**(3/2) + (-delta(x)*Derivative(delta(x), x) + 2*Derivative(epsilon(x), x))*delta(x)/(delta(x)**2 - 4*epsilon(x))**(3/2) + 2*Derivative(V(x), x)/sqrt(delta(x)**2 - 4*epsilon(x)) + Derivative(delta(x), x)/sqrt(delta(x)**2 - 4*epsilon(x)))*a \alpha(x)/((1 - (2*V(x)/sqrt(delta(x)**2 - 4*epsilon(x)) + delta(x)/sqrt(delta(x)**2 - 4*epsilon(x)))**2)*sqrt(delta(x)**2 - 4*epsilon(x)))
        '''
        return self._G_dep_lnphi_d_helper(Z, dbs=self.db_dzs, depsilons=self.depsilon_dzs,
                                          ddelta=self.ddelta_dzs, dVs=self.dV_dzs(Z),
                                          da_alphas=self.da_alpha_dzs, G=True)

    def dG_dep_dns(self, Z):
        r'''Calculates the molar departure Gibbs energy mole number derivatives
        (where the mole fractions sum to 1). No specific formula is implemented
        for this property - it is calculated from the mole fraction derivative.

        .. math::
            \left(\frac{\partial G_{dep}}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left(  \left(\frac{\partial G_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        Apart from the ideal term, this is the formulation for chemical
        potential.

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dG_dep_dns : float
            Departure Gibbs energy mole number derivatives, [J/mol^2]
        '''
        return self._G_dep_lnphi_d_helper(Z, dbs=self.db_dns, depsilons=self.depsilon_dns,
                                          ddelta=self.ddelta_dns, dVs=self.dV_dns(Z),
                                          da_alphas=self.da_alpha_dns, G=True)

    def dnG_dep_dns(self, Z):
        r'''Calculates the partial molar departure Gibbs energy. No specific
        formula is implemented for this property - it is calculated from the
        mole fraction derivative.

        .. math::
            \left(\frac{\partial n G_{dep}}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left(  \left(\frac{\partial G_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dnG_dep_dns : float
            Partial molar departure Gibbs energy of the phase, [J/mol]
        '''
        try:
            if Z == self.Z_l:
                F = self.G_dep_l
            else:
                F = self.G_dep_g
        except:
            F = self.G_dep_g
        dG_dns = self.dG_dep_dns(Z)
        return dns_to_dn_partials(dG_dns, F)

    def fugacity_coefficients(self, Z):
        r'''Generic formula for calculating log fugacity coefficients for each
        species in a mixture. Verified numerically. Applicable to all cubic
        equations of state which can be cast in the form used here.
        Normally this routine is slower than EOS-specific ones, as it does not
        make assumptions that certain parameters are zero or equal to other
        parameters.

        .. math::
            \left(\frac{\partial n \ln \phi}{\partial n_i}
            \right)_{n_{k \ne i}} = \ln \phi _i = \ln \phi +
            n \left(\frac{\partial \ln \phi}{\partial n_i}
            \right)_{n_{k\ne i}}

        .. math::
            \left(\frac{\partial \ln \phi }{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = \frac{1}{RT}\left(  \left(\frac{\partial G_{dep}}
            {\partial n_i}\right)_{T, P, n_{i\ne j}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        log_phis : float
            Log fugacity coefficient for each species, [-]
        '''
        zs = self.zs
        try:
            if Z == self.Z_l:
                F = self.phi_l
            else:
                F = self.phi_g
        except:
            F = self.phi_g
        # This conversion seems numerically safe anyway
        try:
            logF = log(F)
        except:
            logF = -690.7755278982137
        log_phis = dns_to_dn_partials(self.dlnphi_dns(Z), logF)
        return log_phis if self.scalar else array(log_phis)


    def _d2_G_dep_lnphi_d2_helper(self, V, d_Vs, d2Vs, dbs, d2bs, d_epsilons, d2_epsilons,
                          d_deltas, d2_deltas, da_alphas, d2a_alphas, G=True):
        T, P = self.T, self.P
        N = self.N
        RT = T*R
        RT_inv = 1.0/RT
        hess = []
        for i in range(N):
            row = []
            for j in range(N):
                # x1: i
                # x2: j
                x0 = V# V(x1, x2)
                x3 = d2Vs[i][j] #Derivative(x0, x1, x2)
                x4 = self.b#b(x1, x2)
                x5 = x0 - x4
                x6 = R*T
                x7 = d_Vs[i] #Derivative(x0, x1)
                x8 = d_Vs[j] #Derivative(x0, x2)
                x9 = self.delta#delta(x1, x2)
                x10 = self.epsilon#epsilon(x1, x2)
                x11 = -4*x10 + x9**2
                if x11 == 0.0:
                    x11 = 1e-100
                x12 = 1/sqrt(x11)
                x13 = self.a_alpha#alpha(x1, x2)
                x14 = 2*x0
                x15 = x14 + x9
                x16 = catanh(x12*x15).real
                x17 = 2*x16
                x18 = d_deltas[i] #Derivative(x9, x1)
                x19 = x18*x9 - 2*d_epsilons[i]#Derivative(x10, x1)
                x20 = da_alphas[j]#Derivative(x13, x2)
                x21 = x17/x11**(3/2)
                x22 = d_deltas[j]#Derivative(x9, x2)
                x23 = x22*x9 - 2*d_epsilons[j]#Derivative(x10, x2)
                x24 = da_alphas[i]#Derivative(x13, x1)
                x25 = d2_deltas[i][j]#Derivative(x9, x1, x2)
                x26 = x18*x22 + x25*x9 - 2*d2_epsilons[i][j]#Derivative(x10, x1, x2)
                x27 = x13*x23
                x28 = 2*x7
                x29 = 1/x11
                x30 = x29*x9
                x31 = x19*x29
                x32 = x14*x31 - x18 + x19*x30 - x28
                x33 = x15**2*x29 - 1
                x34 = 2/x33
                x35 = x29*x34
                x36 = 2*x8
                x37 = x23*x29
                x38 = x14*x37 - x22 + x23*x30 - x36
                x39 = x11**(-2)
                x40 = x19*x39
                x41 = x13*x38
                x42 = x32*x39
                x43 = x23*x40
                v = (P*x3 - x12*x17*d2a_alphas[i][j] + x13*x21*x26
                     - x13*x35*(-6*x0*x43 + x14*x26*x29 + x18*x37 + x22*x31
                                - x25 + x26*x30 + x28*x37 - 2*x3 + x31*x36
                            - 3*x43*x9) - 4*x15*x41*x42/x33**2 + x19*x20*x21
                    - x20*x32*x35 + x21*x23*x24 - x24*x35*x38 + x27*x34*x42
                    + x34*x40*x41 - x6*(x3 - d2bs[i][j])/x5
                    + x6*(x7 - dbs[i])*(x8 - dbs[j])/x5**2
                    - 6*x16*x19*x27/x11**(5/2))
                if not G:
                    v *= RT_inv
                row.append(v)
            hess.append(row)
        return hess



    def d2lnphi_dzizjs(self, Z):
        r'''Calculates the mixture log *fugacity coefficient* second mole
        fraction derivatives (where the mole fractions do not sum to 1). No
        specific formula is implemented for this property - it is calculated
        from the second mole fraction derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial^2 \ln \phi }{\partial x_i\partial x_j}\right)_{T, P,
            x_{i,j\ne k}} = \frac{1}{RT}\left( \left(\frac{\partial^2 G_{dep}}
            {\partial x_j \partial x_i}\right)_{T, P, x_{i,j\ne k}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        d2lnphi_dzizjs : float
            Mixture log fugacity coefficient second mole fraction derivatives,
            [-]
        '''
        V = Z*self.T*R/self.P
        dV_dzs = self.dV_dzs(Z)
        d2Vs = self.d2V_dzizjs(Z)

        depsilon_dzs = self.depsilon_dzs
        d2epsilon_dzizjs = self.d2epsilon_dzizjs

        ddelta_dzs = self.ddelta_dzs
        d2delta_dzizjs = self.d2delta_dzizjs

        db_dzs = self.db_dzs
        d2bs = self.d2b_dzizjs
        da_alpha_dzs = self.da_alpha_dzs
        d2a_alpha_dzizjs = self.d2a_alpha_dzizjs
        return self._d2_G_dep_lnphi_d2_helper(V=V, d_Vs=dV_dzs, d2Vs=d2Vs, dbs=db_dzs, d2bs=d2bs,
                                     d_epsilons=depsilon_dzs, d2_epsilons=d2epsilon_dzizjs,
                                     d_deltas=ddelta_dzs, d2_deltas=d2delta_dzizjs,
                                     da_alphas=da_alpha_dzs, d2a_alphas=d2a_alpha_dzizjs,
                                     G=False)

    def d2lnphi_dninjs(self, Z):
        r'''Calculates the mixture log *fugacity coefficient* second mole
        number derivatives (where the mole fraction sum to 1). No
        specific formula is implemented for this property - it is calculated
        from the second mole fraction derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial^2 \ln \phi }{\partial n_i\partial n_j}\right)_{T, P,
            n_{i,j\ne k}}  f\left( \left(\frac{\partial^2 G_{dep}}
            {\partial x_j \partial x_i}\right)_{T, P, x_{i,j\ne k}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        d2lnphi_dninjs : float
            Mixture log fugacity coefficient second mole number derivatives,
            [-]
        '''
        V = Z*self.T*R/self.P
        dV_dns = self.dV_dns(Z)
        d2Vs = self.d2V_dninjs(Z)

        depsilon_dns = self.depsilon_dns
        d2epsilon_dninjs = self.d2epsilon_dninjs

        ddelta_dns = self.ddelta_dns
        d2delta_dninjs = self.d2delta_dninjs

        db_dns = self.db_dns
        d2bs = self.d2b_dninjs
        da_alpha_dns = self.da_alpha_dns
        d2a_alpha_dninjs = self.d2a_alpha_dninjs
        return self._d2_G_dep_lnphi_d2_helper(V=V, d2Vs=d2Vs, d_Vs=dV_dns, dbs=db_dns, d2bs=d2bs,
                                     d_epsilons=depsilon_dns, d2_epsilons=d2epsilon_dninjs,
                                     d_deltas=ddelta_dns, d2_deltas=d2delta_dninjs,
                                     da_alphas=da_alpha_dns, d2a_alphas=d2a_alpha_dninjs,
                                     G=False)


    def d2G_dep_dzizjs(self, Z):
        r'''Calculates the molar departure Gibbs energy second composition
        derivative (where the mole fractions do not sum to 1). Verified numerically.
        Useful in solving for gibbs minimization calculations or for solving
        for the true critical point. Also forms the basis for the molar
        departure Gibbs energy mole second number derivative.

        .. math::
            \left(\frac{\partial^2 G_{dep}}{\partial x_j \partial x_i}\right)_{T, P,
            x_{i,j\ne k}}  = \text{run SymPy code to obtain - very long!}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        d2G_dep_dzizjs : float
            Departure Gibbs free energy second composition derivatives, [J/mol]

        Notes
        -----
        The derivation for the derivative is performed as follows using SymPy.
        The function source code is an optimized variant created with the `cse`
        SymPy function, and hand optimized further.

        >>> from sympy import * # doctest:+SKIP
        >>> P, T, R, x1, x2 = symbols('P, T, R, x1, x2') # doctest:+SKIP
        >>> a_alpha, delta, epsilon, V, b = symbols('a\ \\alpha, delta, epsilon, V, b', cls=Function) # doctest:+SKIP
        >>> da_alpha_dT, d2a_alpha_dT2 = symbols('da_alpha_dT, d2a_alpha_dT2', cls=Function) # doctest:+SKIP
        >>> S_dep = R*log(P*V(x1, x2)/(R*T)) + R*log(V(x1, x2)-b(x1, x2))+2*da_alpha_dT(x1, x2)*atanh((2*V(x1, x2)+delta(x1, x2))/sqrt(delta(x1, x2)**2-4*epsilon(x1, x2)))/sqrt(delta(x1, x2)**2-4*epsilon(x1, x2))-R*log(V(x1, x2)) # doctest:+SKIP
        >>> H_dep = P*V(x1, x2) - R*T + 2*atanh((2*V(x1, x2)+delta(x1, x2))/sqrt(delta(x1, x2)**2-4*epsilon(x1, x2)))*(da_alpha_dT(x1, x2)*T-a_alpha(x1, x2))/sqrt(delta(x1, x2)**2-4*epsilon(x1, x2)) # doctest:+SKIP
        >>> G_dep = simplify(H_dep - T*S_dep) # doctest:+SKIP
        >>> diff(G_dep, x1, x2) # doctest:+SKIP
        '''
        V = Z*self.T*R/self.P
        dV_dzs = self.dV_dzs(Z)
        d2Vs = self.d2V_dzizjs(Z)

        depsilon_dzs = self.depsilon_dzs
        d2epsilon_dzizjs = self.d2epsilon_dzizjs

        ddelta_dzs = self.ddelta_dzs
        d2delta_dzizjs = self.d2delta_dzizjs

        db_dzs = self.db_dzs
        d2bs = self.d2b_dzizjs
        da_alpha_dzs = self.da_alpha_dzs
        d2a_alpha_dzizjs = self.d2a_alpha_dzizjs
        return self._d2_G_dep_lnphi_d2_helper(V=V, d_Vs=dV_dzs, d2Vs=d2Vs, dbs=db_dzs, d2bs=d2bs,
                                     d_epsilons=depsilon_dzs, d2_epsilons=d2epsilon_dzizjs,
                                     d_deltas=ddelta_dzs, d2_deltas=d2delta_dzizjs,
                                     da_alphas=da_alpha_dzs, d2a_alphas=d2a_alpha_dzizjs,
                                     G=True)
    def dlnphis_dns(self, Z):
        r'''Generic formula for calculating the mole number derivaitves of
        log fugacity coefficients for each species in a mixture. Verified
        numerically. Applicable to all cubic equations of state which can be
        cast in the form used here.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial n_i}\right)_{P,
            n_{j \ne i}}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dlnphis_dns : list[list[float]]
            Mole number derivatives of log fugacity coefficient for each
            species, [-]

        Notes
        -----
        '''
        dns = self.dlnphi_dns(Z)
        d2ns = self.d2lnphi_dninjs(Z)
        return d2ns_to_dn2_partials(d2ns, dns)

    def dlnfugacities_dns(self, phase):
        r'''Generic formula for calculating the mole number derivaitves of
        log fugacities for each species in a mixture. Verified
        numerically. Applicable to all cubic equations of state which can be
        cast in the form used here.

        .. math::
            \left(\frac{\partial \ln f_i}{\partial n_i}\right)_{P,
            n_{j \ne i}}

        Parameters
        ----------
        phase : str
            One of 'l' or 'g', [-]

        Returns
        -------
        dlnfugacities_dns : list[list[float]]
            Mole number derivatives of log fugacities for each
            species, [-]

        Notes
        -----
        '''
        zs, N = self.zs, self.N
        if phase == 'l':
            Z = self.Z_l
            try:
                fugacities = self.fugacities_l
            except AttributeError:
                self.fugacities()
                fugacities = self.fugacities_l
        else:
            Z = self.Z_g
            try:
                fugacities = self.fugacities_g
            except AttributeError:
                self.fugacities()
                fugacities = self.fugacities_g
        dlnfugacities_dns = [list(i) for i in self.dfugacities_dns(phase)]
        fugacities_inv = [1.0/fi for fi in fugacities]
        for i in range(N):
            r = dlnfugacities_dns[i]
            for j in range(N):
                r[j]*= fugacities_inv[i]
        return dlnfugacities_dns

    def dfugacities_dns(self, phase):
        r'''Generic formula for calculating the mole number derivaitves of
        fugacities for each species in a mixture. Verified
        numerically. Applicable to all cubic equations of state which can be
        cast in the form used here.

        .. math::
            \left(\frac{\partial f_i}{\partial n_i}\right)_{P,
            n_{j \ne i}}

        Parameters
        ----------
        phase : str
            One of 'l' or 'g', [-]

        Returns
        -------
        dfugacities_dns : list[list[float]]
            Mole number derivatives of fugacities for each species, [-]

        Notes
        -----
        '''
        '''
        from sympy import *
        phifun1, phifun2 = symbols('phifun1, phifun2', cls=Function)
        n1, n2, P = symbols('n1, n2, P')

        x1 = n1/(n1+n2)
        x2 = n2/(n1+n2)

        to_diff = x2*P*exp(phifun1(n1))
        diff(to_diff, n1).subs({n1+n1: 1})
        '''
        zs = self.zs
        if phase == 'l':
            Z = self.Z_l
            try:
                phis = self.phis_l
            except AttributeError:
                self.fugacities()
                phis = self.phis_l
        else:
            Z = self.Z_g
            try:
                phis = self.phis_g
            except AttributeError:
                self.fugacities()
                phis = self.phis_g


        dlnphis_dns = self.dlnphis_dns(Z)

        P = self.P
        N = self.N
        matrix = []
        for i in range(N):
            phi_P = P*phis[i]
            ziPphi = phi_P*zs[i]
            r = dlnphis_dns[i]
#            row = [ziPphi*(r[j] - 1.0) for j in range(N)]
            row = [ziPphi*(dlnphis_dns[j][i] - 1.0) for j in range(N)]
            row[i] += phi_P
            matrix.append(row)
        return matrix



    def d2G_dep_dninjs(self, Z):
        r'''Calculates the molar departure Gibbs energy mole number derivatives
        (where the mole fractions sum to 1). No specific formula is implemented
        for this property - it is calculated from the mole fraction derivative.

        .. math::
             \left(\frac{\partial^2 G_{dep}}{\partial n_j \partial n_i}\right)_{T, P,
            n_{i,j\ne k}}   = f\left(   \left(\frac{\partial^2 G_{dep}}{\partial x_j \partial x_i}\right)_{T, P,
            x_{i,j\ne k}}
            \right)

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        d2G_dep_dninjs : float
            Departure Gibbs energy second mole number derivatives, [J/mol^3]
        '''
        V = Z*self.T*R/self.P
        dV_dns = self.dV_dns(Z)
        d2Vs = self.d2V_dninjs(Z)

        depsilon_dns = self.depsilon_dns
        d2epsilon_dninjs = self.d2epsilon_dninjs

        ddelta_dns = self.ddelta_dns
        d2delta_dninjs = self.d2delta_dninjs

        db_dns = self.db_dns
        d2bs = self.d2b_dninjs
        da_alpha_dns = self.da_alpha_dns
        d2a_alpha_dninjs = self.d2a_alpha_dninjs
        return self._d2_G_dep_lnphi_d2_helper(V=V, d2Vs=d2Vs, d_Vs=dV_dns, dbs=db_dns, d2bs=d2bs,
                                     d_epsilons=depsilon_dns, d2_epsilons=d2epsilon_dninjs,
                                     d_deltas=ddelta_dns, d2_deltas=d2delta_dninjs,
                                     da_alphas=da_alpha_dns, d2a_alphas=d2a_alpha_dninjs,
                                     G=True)



    def _d2_A_dep_d2_helper(self, V, d_Vs, d2Vs, dbs, d2bs, d_epsilons,
                            d2_epsilons, d_deltas, d2_deltas, da_alphas,
                            d2a_alphas):
        # pass
        r'''from sympy import * # doctest:+SKIP
        P, T, R, x1, x2 = symbols('P, T, R, x1, x2') # doctest:+SKIP
        a_alpha, delta, epsilon, V, b = symbols('a\ \\alpha, delta, epsilon, V, b', cls=Function) # doctest:+SKIP
        da_alpha_dT, d2a_alpha_dT2 = symbols('da_alpha_dT, d2a_alpha_dT2', cls=Function) # doctest:+SKIP
        S_dep = R*log(P*V(x1, x2)/(R*T)) + R*log(V(x1, x2)-b(x1, x2))+2*da_alpha_dT(x1, x2)*atanh((2*V(x1, x2)+delta(x1, x2))/sqrt(delta(x1, x2)**2-4*epsilon(x1, x2)))/sqrt(delta(x1, x2)**2-4*epsilon(x1, x2))-R*log(V(x1, x2)) # doctest:+SKIP
        H_dep = P*V(x1, x2) - R*T + 2*atanh((2*V(x1, x2)+delta(x1, x2))/sqrt(delta(x1, x2)**2-4*epsilon(x1, x2)))*(da_alpha_dT(x1, x2)*T-a_alpha(x1, x2))/sqrt(delta(x1, x2)**2-4*epsilon(x1, x2)) # doctest:+SKIP
        G_dep = simplify(H_dep - T*S_dep) # doctest:+SKIP


        V_dep = V(x1, x2) - R*T/P
        U_dep = H_dep - P*V_dep

        A_dep = simplify(U_dep - T*S_dep)
        '''
        T, P = self.T, self.P
        b = self.b
        N = self.N
        RT = T*R
        hess = []

        for i in range(N):
            row = []
            for j in range(N):
                x0 = V
                x3 = b
                x4 = x0 - x3
                x5 = d2Vs[i][j]
                x6 = R*T
                x7 = d_Vs[i]
                x8 = d_Vs[j]
                x9 = self.delta
                x10 = self.epsilon
                x11 = -4*x10 + x9**2
                x12 = 1/sqrt(x11)
                x13 = self.a_alpha
                x14 = 2*x0
                x15 = x14 + x9
                x16 = catanh(x12*x15).real
                x17 = 2*x16
                x18 = d_deltas[i]
                x19 = x18*x9 - 2*d_epsilons[i]
                x20 = da_alphas[j]
                x21 = x17/x11**(3/2)
                x22 = d_deltas[j]
                x23 = x22*x9 - 2*d_epsilons[j]
                x24 = da_alphas[i]
                x25 = d2_deltas[i][j]
                x26 = x18*x22 + x25*x9 - 2*d2_epsilons[i][j]
                x27 = x13*x23
                x28 = 2*x7
                x29 = 1/x11
                x30 = x29*x9
                x31 = x19*x29
                x32 = x14*x31 - x18 + x19*x30 - x28
                x33 = x15**2*x29 - 1
                x34 = 2/x33
                x35 = x29*x34
                x36 = 2*x8
                x37 = x23*x29
                x38 = x14*x37 - x22 + x23*x30 - x36
                x39 = x11**(-2)
                x40 = x19*x39
                x41 = x13*x38
                x42 = x32*x39
                x43 = x23*x40
                v = (-x12*x17*d2a_alphas[i][j] + x13*x21*x26 - x13*x35*(-6*x0*x43
                     + x14*x26*x29 + x18*x37 + x22*x31 - x25 + x26*x30 + x28*x37
                     + x31*x36 - 3*x43*x9 - 2*x5) - 4*x15*x41*x42/x33**2
        + x19*x20*x21 - x20*x32*x35 + x21*x23*x24 - x24*x35*x38 + x27*x34*x42
        + x34*x40*x41 - x6*(x5 - d2bs[i][j])/x4 + x6*(x7 - dbs[i])*(x8 - dbs[j])/x4**2 - 6*x16*x19*x27/x11**(5/2.))
                row.append(v)
            hess.append(row)

        return hess



    def d2A_dep_dninjs(self, Z):
        V = Z*self.T*R/self.P
        dV_dns = self.dV_dns(Z)
        d2Vs = self.d2V_dninjs(Z)

        depsilon_dns = self.depsilon_dns
        d2epsilon_dninjs = self.d2epsilon_dninjs

        ddelta_dns = self.ddelta_dns
        d2delta_dninjs = self.d2delta_dninjs

        db_dns = self.db_dns
        d2bs = self.d2b_dninjs
        da_alpha_dns = self.da_alpha_dns
        d2a_alpha_dninjs = self.d2a_alpha_dninjs
        return self._d2_A_dep_d2_helper(V=V, d2Vs=d2Vs, d_Vs=dV_dns, dbs=db_dns, d2bs=d2bs,
                                     d_epsilons=depsilon_dns, d2_epsilons=d2epsilon_dninjs,
                                     d_deltas=ddelta_dns, d2_deltas=d2delta_dninjs,
                                     da_alphas=da_alpha_dns, d2a_alphas=d2a_alpha_dninjs)

    def dA_dep_dns_Vt(self, phase):
        # pass
        r'''
        from sympy import *
        Vt, P, T, R, n1, n2, n3 = symbols('Vt, P, T, R, n1, n2, n3') # doctest:+SKIP
        P, V, a_alpha, delta, epsilon, b = symbols('P, V, a\ \\alpha, delta, epsilon, b', cls=Function) # doctest:+SKIP
        da_alpha_dT, d2a_alpha_dT2 = symbols('da_alpha_dT, d2a_alpha_dT2', cls=Function) # doctest:+SKIP
        ns = [n1, n2, n3]

        S_dep = R*log(P(n1, n2, n3)*V(n1, n2, n3)/(R*T)) + R*log(V(n1, n2, n3)-b(n1, n2, n3))+2*da_alpha_dT(n1, n2, n3)*atanh((2*V(n1, n2, n3)+delta(n1, n2, n3))/sqrt(delta(n1, n2, n3)**2-4*epsilon(n1, n2, n3)))/sqrt(delta(n1, n2, n3)**2-4*epsilon(n1, n2, n3))-R*log(V(n1, n2, n3))
        H_dep = P(n1, n2, n3)*V(n1, n2, n3) - R*T + 2*atanh((2*V(n1, n2, n3)+delta(n1, n2, n3))/sqrt(delta(n1, n2, n3)**2-4*epsilon(n1, n2, n3)))*(da_alpha_dT(n1, n2, n3)*T-a_alpha(n1, n2, n3))/sqrt(delta(n1, n2, n3)**2-4*epsilon(n1, n2, n3))
        G_dep = simplify(H_dep - T*S_dep)
        V_dep = V(n1, n2, n3) - R*T/P(n1, n2, n3)
        U_dep = H_dep - P(n1, n2, n3)*V_dep
        A_dep = simplify(U_dep - T*S_dep)
        expr = diff(A_dep, n1)

        for ni in ns:
            expr = expr.subs(Derivative(V(n1, n2, n3), ni), -Vt)

        expr = simplify(expr)
        cse(expr, optimizations='basic')
        '''
        if phase == 'g':
            Vt = self.V_g
        else:
            Vt = self.V_l

        T, N = self.T, self.N
        b = self.b
        a_alpha = self.a_alpha
        epsilon = self.epsilon

        depsilon_dns = self.depsilon_dns
        ddelta_dns = self.ddelta_dns
        db_dns = self.db_dns
        da_alpha_dns = self.da_alpha_dns
        dP_dns_Vt = self.dP_dns_Vt(phase)

        x0 = self.P
        x1 = Vt
        x2 = self.b
        x3 = x1 - x2
        x4 = self.delta
        x5 = x4**2
        x6 = self.epsilon
        x7 = 4*x6
        x8 = x5 - x7
        x9 = x8**(7/2)
        x10 = 2*x1
        x11 = x10 + x4
        x12 = x11**2 - x5 + x7
        x13 = Vt*x0
        x14 = x12*x3
        x15 = R*T*x9
        x16 = x14*x15
        x17 = self.a_alpha
        x18 = x0*x10
        x19 = x14*catanh(x11*x8**-0.5).real

        jac = []
        for i in range(N):
            x20 = ddelta_dns[i]
            x21 = x20*x4 - 2*depsilon_dns[i]
            x22 = x17*x18

            v = (-(-x0*x1*x12*x15*(Vt + db_dns[i]) + x13*x16 - x16*(-x1*dP_dns_Vt[i] + x13)
                + x18*x19*x8**3*da_alpha_dns[i] - x19*x21*x22*x8**2
                + x22*x3*x8**(5/2)*(x11*x21 + x8*(2*Vt - x20)))/(x0*x1*x12*x3*x9))
            jac.append(v)
        return jac



    def d2A_dep_dninjs_Vt(self, phase):
        if phase == 'g':
            Vt = self.V_g
        else:
            Vt = self.V_l

        T, N = self.T, self.N
        b = self.b
        a_alpha = self.a_alpha
        epsilon = self.epsilon

        depsilon_dns = self.depsilon_dns
        ddelta_dns = self.ddelta_dns
        db_dns = self.db_dns
        da_alpha_dns = self.da_alpha_dns

        d2delta_dninjs = self.d2delta_dninjs
        d2epsilon_dninjs = self.d2epsilon_dninjs
        d2bs = self.d2b_dninjs
        d2a_alpha_dninjs = self.d2a_alpha_dninjs

        dP_dns_Vt = self.dP_dns_Vt(phase)
        d2P_dninjs_Vt = self.d2P_dninjs_Vt(phase)

        hess = [[0.0]*N for i in range(N)]

        for i in range(N):
            for j in range(i+1):
                x0 = self.P
                x1 = x0**2
                x2 = Vt#V(n1, n2, n3)
                x3 = x2**2
                x4 = self.b
                x5 = x2 - x4
                x6 = x5**2
                x7 = self.delta
                x8 = x7**2
                x9 = self.epsilon
                x10 = 4*x9
                x11 = -x10 + x8
                x12 = x11**(25/2)
                x13 = 2*x2
                x14 = x13 + x7
                x15 = x10 + x14**2 - x8
                x16 = x15**2
                x17 = x1*x6
                x18 = R*T*x12*x16
                x19 = x17*x18
                x20 = x1*x18*x3
                x21 = Vt*x0
                x22 = dP_dns_Vt[i]
                x23 = -x2*x22 + x21
                x24 = 2*Vt
                x25 = dP_dns_Vt[j]
                x26 = x18*x2*x6
                x27 = self.a_alpha
                x28 = x17*x3
                x29 = 2*x28
                x30 = x16*catanh(x14/sqrt(x11)).real
                x31 = x29*x30
                x32 = ddelta_dns[i]
                x33 = x32*x7 - 2*depsilon_dns[i]
                x34 = ddelta_dns[j]
                x35 = x34*x7 - 2*depsilon_dns[j]
                x36 = x33*x35
                x37 = da_alpha_dns[j]
                x38 = da_alpha_dns[i]
                x39 = d2delta_dninjs[i][j]
                x40 = x32*x34 + x39*x7 - 2*d2epsilon_dninjs[i][j]
                x41 = x11*(x24 - x32) + x13*x33 + x33*x7
                x42 = x11*(x24 - x34) + x13*x35 + x35*x7
                x43 = x11**(21/2)*x27
                x44 = x15*x29
                x45 = x43*x44
                v = (-(Vt**2*x19 - Vt*x13*x19 + x0*x26*(-Vt*x22 - Vt*x25 + x0*x24
                                                        + x2*d2P_dninjs_Vt[i][j]) + x11**(23/2)*x44*(x37*x41 + x38*x42)
                    + x11**12*x31*d2a_alpha_dninjs[i][j] - x11**11*x31*(x27*x40 + x33*x37 + x35*x38)
                    + 6*x11**10*x27*x28*x30*x36 + 4*x14*x28*x41*x42*x43 - x18*x21*x23*x6
                    + x20*x5*(x24 - d2bs[i][j]) - x20*(Vt + db_dns[i])*(Vt + db_dns[j]) + x23*x25*x26
                    - x45*(x33*x42 + x35*x41) - x45*(x11**2*(4*Vt + x39) - x11*(x13*x40 - x24*x33
                           - x24*x35 + x32*x35 + x33*x34 + x40*x7) + 3*x14*x36))/(x1*x12*x16*x3*x6))
                hess[i][j] = hess[j][i] = v
        return hess

#    @property
#    def SCp0_l(self):
#        S_dep = self.S_dep_l
#        S_dep -= R*sum([zi*log(zi) for zi in self.zs if zi > 0.0]) # ideal composition entropy composition
#        S_dep -= R*log(self.P/101325.0)
#        return S_dep
#
#    @property
#    def ACp0_l(self):
#        return self.A_dep_l - self.T*(self.SCp0_l - self.S_dep_l)
#
#    @property
#    def SCp0_g(self):
#        S_dep = self.S_dep_g
#        S_dep -= R*sum([zi*log(zi) for zi in self.zs if zi > 0.0]) # ideal composition entropy composition
#        S_dep -= R*log(self.P/101325.0)
#        return S_dep
#
#    @property
#    def ACp0_g(self):
#        return self.A_dep_g - self.T*(self.SCp0_g - self.S_dep_g)
#
#    def Scomp(self, phase):
#        v = self.T*R*sum([zi*log(zi) for zi in self.zs if zi > 0.0]) # ideal composition entropy composition
#        v += R*self.T*log(self.P/101325.0)
#        return v
#
#    @property
#    def HCp0_g(self):
#        return self.H_dep_g
#
#    @property
#    def HCp0_l(self):
#        return self.H_dep_l
#
#    @property
#    def GCp0_g(self):
#        return self.HCp0_g - self.T*self.SCp0_g
#
#    @property
#    def GCp0_l(self):
#        return self.HCp0_l - self.T*self.SCp0_l

    def dScomp_dns(self, phase):
        dP_dns_Vt = self.dP_dns_Vt(phase)

        mRT = -R*self.T
        zs, N = self.zs, self.N

        logzs = [log(zi) for zi in zs]
        tot = 0.0
        for i in range(N):
            tot += zs[i]*logzs[i]

        const = R*self.T/self.P
        return [mRT*(tot - logzs[i]) + const*dP_dns_Vt[i] for i in range(N)]

    def d2Scomp_dninjs(self, phase):
        '''P_ref = symbols('P_ref')
            diff(R*T*log(P(n1, n2, n3)/P_ref), n1, n2)
        '''
        dP_dns_Vt = self.dP_dns_Vt(phase)
        d2P_dninjs_Vt = self.d2P_dninjs_Vt(phase)

        P = self.P
        RT = R*self.T
        const = RT/P
        zs, N = self.zs, self.N

        logzs = [log(zi) for zi in zs]

        hess = []
        for i in range(N):
            row = []
            for j in range(N):
                t = sum(2.0*zs[i]*logzs[i] + 3.0*zs[i] for i in range(N))
                if i != j:
                    v = RT*(t - logzs[i] - logzs[j] -4.0)
                else:
                    v = RT*(t - 2*logzs[i] - 3 - (zs[i] - 1.0)/zs[i])

                v += const*(d2P_dninjs_Vt[i][j] - dP_dns_Vt[i]*dP_dns_Vt[j]/P)

                row.append(v)
            hess.append(row)
        return hess



        # TODO fix the implementation below, make it work
        tot = 0.0
        for i in range(N):
            tot += zs[i]*logzs[i]

        tot2m1 = tot + tot - 1.0
        hess = [[RT*(tot2m1 - logzs[i] - logzs[j]) for i in range(N)] for j in range(N)]
        return hess
#        return d2xs_to_dxdn_partials(hess, zs)
#        return d2ns_to_dn2_partials(hess, self.dScomp_dns)

    def d2A_dninjs_Vt(self, phase):
        if phase == 'g':
            Vt = self.V_g
        else:
            Vt = self.V_l
        N, zs = self.N, self.zs

        d2A_dep_dninjs_Vt = self.d2A_dep_dninjs_Vt(phase)
        d2Scomp_dninjs = self.d2Scomp_dninjs

        hess = [[0.0]*N for i in range(N)]
        for i in range(N):
            for j in range(N):
                hess[i][j] = d2Scomp_dninjs[i][j] + d2A_dep_dninjs_Vt[i][j]
        return hess


    def d2nA_dninjs_Vt(self, phase):
        d2ns = [[i+j for i, j in zip(r1, r2)] for r1, r2 in zip(self.d2A_dep_dninjs_Vt(phase), self.d2Scomp_dninjs(phase))]
        dns = [i+j for i, j in zip(self.dA_dep_dns_Vt(phase), self.dScomp_dns(phase))]
        return d2ns_to_dn2_partials(d2ns, dns)

    def d2A_dninjs_Vt_another(self, phase):
        d2ns = [[i+j for i, j in zip(r1, r2)] for r1, r2 in zip(self.d2A_dep_dninjs_Vt(phase), self.d2Scomp_dninjs(phase))]
        return d2ns
#        dns = [i+j for i, j in zip(self.dA_dep_dns_Vt(phase), self.dScomp_dns(phase))]
#        return d2ns_to_dn2_partials(d2ns, dns)

    def _d_main_derivatives_and_departures_dnx(self, V, db_dns, ddelta_dns,
                                               depsilon_dns, da_alpha_dns,
                                               da_alpha_dT_dns,
                                               d2a_alpha_dT2_dns, dV_dns):
        T = self.T
        Z = (self.P*V)/(R*T)

        x0 = self.a_alpha
        x2 = self.epsilon
        x3 = V
        x4 = self.delta
        x5 = x2 + x3**2 + x3*x4
        x6 = 1/x5
        x7 = self.b
        x8 = x3 - x7
        x14 = x5**(-2)
        x15 = self.da_alpha_dT
        x16 = x14*x15
        x18 = 2*x3 + x4
        x23 = x5**(-3)
        x24 = 2*x23
        x27 = x18**2
        x28 = x18*x24


        dndP_dT_dsn = []
        dndP_dV_dns = []
        dnd2P_dT2_dns = []
        dnd2P_dV2_dns = []
        dnd2P_dTdV_dns = []

        for i in range(self.N):
            x1 = da_alpha_dT_dns[i]
            x9 = dV_dns[i]
            x10 = R*(x9 - db_dns[i])
            x17 = 2*x10/x8**3
            x12 = 2*x9

            x11 = ddelta_dns[i]
            x21 = x11 + x12
            x22 = x0*x21

            x13 = x11*x3 + x12*x3 + x4*x9 + depsilon_dns[i]
            x25 = x0*x13
            x26 = x24*x25

            x19 = da_alpha_dns[i]
            x20 = x14*x19

            dndP_dT = -x1*x6 - x10/x8**2 + x13*x16
            dndP_dT_dsn.append(dndP_dT)

            dndP_dV = T*x17 + x14*x22 + x18*x20 - x18*x26
            dndP_dV_dns.append(dndP_dV)

            d2a_alpha_dT2_dn = d2a_alpha_dT2_dns[i]
            dnd2P_dT2 = x6*(x13*x6*self.d2a_alpha_dT2 - d2a_alpha_dT2_dn)
            dnd2P_dT2_dns.append(dnd2P_dT2)

            dnd2P_dV2 = -6*T*x10/x8**4 - 2*x19*x23*x27 + 2*x20 - 2*x22*x28 + 6*x25*x27/x5**4 - 2*x26
            dnd2P_dV2_dns.append(dnd2P_dV2)

            dnd2P_dTdV = x1*x14*x18 - x13*x15*x28 + x16*x21 + x17
            dnd2P_dTdV_dns.append(dnd2P_dTdV)

        return dndP_dT_dsn, dndP_dV_dns, dnd2P_dT2_dns, dnd2P_dV2_dns, dnd2P_dTdV_dns

    def _d_main_derivatives_and_departures_dn(self, V):
        Z = (self.P*V)/(R*self.T)
        db_dns = self.db_dns
        ddelta_dns = self.ddelta_dns
        depsilon_dns = self.depsilon_dns
        dV_dns = self.dV_dns(Z)

        da_alpha_dns = self.da_alpha_dns
        da_alpha_dT_dns = self.da_alpha_dT_dns
        d2a_alpha_dT2_dns = self.d2a_alpha_dT2_dns
        return self._d_main_derivatives_and_departures_dnx(V, db_dns, ddelta_dns,
                                               depsilon_dns, da_alpha_dns,
                                               da_alpha_dT_dns, d2a_alpha_dT2_dns,
                                               dV_dns)

    def _d_main_derivatives_and_departures_dz(self, V):
        Z = (self.P*V)/(R*self.T)
        db_dzs = self.db_dzs
        ddelta_dzs = self.ddelta_dzs
        depsilon_dzs = self.depsilon_dzs
        dV_dzs = self.dV_dzs(Z)

        da_alpha_dzs = self.da_alpha_dzs
        da_alpha_dT_dzs = self.da_alpha_dT_dzs
        d2a_alpha_dT2_dzs = self.d2a_alpha_dT2_dzs
        return self._d_main_derivatives_and_departures_dnx(V, db_dzs, ddelta_dzs,
                                               depsilon_dzs, da_alpha_dzs,
                                               da_alpha_dT_dzs, d2a_alpha_dT2_dzs,
                                               dV_dzs)


    def _dnz_derivatives_and_departures(self, V, n=True):
        try:
            if V == self.V_l:
                l = True
            else:
                l = False
        except:
            l = False

        if n:
            f = self._d_main_derivatives_and_departures_dn
        else:
            f = self._d_main_derivatives_and_departures_dz

        d2P_dTdns, d2P_dVdns, d3P_dT2dns, d3P_dV2dns, d3P_dTdVdns = f(V)

        # Needed in calculation routines
        if l:
            (dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP, d2P_dT2, d2P_dV2, d2V_dT2,
             d2V_dP2, d2T_dV2, d2T_dP2, d2V_dPdT, d2P_dTdV, d2T_dPdV) = (self.dP_dT_l,
            self.dP_dV_l, self.dV_dT_l, self.dV_dP_l, self.dT_dV_l, self.dT_dP_l,
            self.d2P_dT2_l, self.d2P_dV2_l, self.d2V_dT2_l, self.d2V_dP2_l, self.d2T_dV2_l,
            self.d2T_dP2_l, self.d2V_dPdT_l, self.d2P_dTdV_l, self.d2T_dPdV_l)
        else:
            (dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP, d2P_dT2, d2P_dV2, d2V_dT2,
             d2V_dP2, d2T_dV2, d2T_dP2, d2V_dPdT, d2P_dTdV, d2T_dPdV) = (self.dP_dT_g,
            self.dP_dV_g, self.dV_dT_g, self.dV_dP_g, self.dT_dV_g, self.dT_dP_g,
            self.d2P_dT2_g, self.d2P_dV2_g, self.d2V_dT2_g, self.d2V_dP2_g, self.d2T_dV2_g,
            self.d2T_dP2_g, self.d2V_dPdT_g, self.d2P_dTdV_g, self.d2T_dPdV_g)

        d2V_dTdns = []
        d2V_dPdns = []
        d2T_dVdns = []
        d2T_dPdns = []
        d3T_dP2dns = []
        d3V_dP2dns = []
        d3T_dV2dns = []
        d3V_dT2dns = []
        d3T_dPdVdns = []
        d3V_dPdTdns = []
        for i in range(self.N):
            d2P_dTdn, d2P_dVdn, d3P_dT2dn, d3P_dV2dn, d3P_dTdVdn = (
                    d2P_dTdns[i], d2P_dVdns[i], d3P_dT2dns[i], d3P_dV2dns[i], d3P_dTdVdns[i])

            # First derivative - one over the other
            d2V_dTdn = dP_dT*d2P_dVdn/dP_dV**2 - d2P_dTdn/dP_dV
            d2V_dTdns.append(d2V_dTdn)
    #        dP_dT # f
    #        dP_dV # g

            # Second derivative - one over the other
            d2V_dPdn = dV_dT*d2P_dTdn/dP_dT**2 - d2V_dTdn/dP_dT
            d2V_dPdns.append(d2V_dPdn)
    #        f = dV_dT
    #        g = dP_dT

            # Third derivative - inverse of other expression
            d2T_dVdn = -d2V_dTdn/dV_dT**2
            d2T_dVdns.append(d2T_dVdn)

            # Fourth derivative - inverse of other expression
            d2T_dPdn = -d2P_dTdn/dP_dT**2
            d2T_dPdns.append(d2T_dPdn)

            # Fifth derivative - starting to get big
            f = d2P_dT2
            df = d3P_dT2dn
            g = dP_dT
            dg = d2P_dTdn
            d3T_dP2dn = 3*f*dg/g**4 - df/g**3
            d3T_dP2dns.append(d3T_dP2dn)

            # Sixth derivative
            f = d2P_dV2
            df = d3P_dV2dn
            g = dP_dV
            dg = d2P_dVdn
            d3V_dP2dn = 3*f*dg/g**4 - df/g**3
            d3V_dP2dns.append(d3V_dP2dn)

            # Seventh - crazy
            f = d2P_dV2
            df = d3P_dV2dn
            g = dP_dT
            dg = d2P_dTdn
            h = dP_dV
            dh = d2P_dVdn
            k = d2P_dTdV
            dk = d3P_dTdVdn
            j = d2P_dT2
            dj = d3P_dT2dn

            d3T_dV2dn = (f*g**2*dg - g**3*df + 2*g**2*h*dk + 2*g**2*k*dh - g*h**2*dj - 4*g*h*k*dg - 2*g*h*j*dh + 3*h**2*j*dg)/g**4
            d3T_dV2dns.append(d3T_dV2dn)

            # ekghth - crazy
            f = d2P_dT2
            df = d3P_dT2dn
            g = dP_dV
            dg = d2P_dVdn
            h = dP_dT
            dh = d2P_dTdn
            k = d2P_dTdV
            dk = d3P_dTdVdn
            j = d2P_dV2
            dj = d3P_dV2dn
            d3V_dT2dn = (f*g**2*dg - g**3*df + 2*g**2*h*dk + 2*g**2*k*dh - g*h**2*dj - 4*g*h*k*dg - 2*g*h*j*dh + 3*h**2*j*dg)/g**4
            d3V_dT2dns.append(d3V_dT2dn)

            # nknth
            f = d2P_dTdV
            df = d3P_dTdVdn
            g = dP_dT
            dg = d2P_dTdn
            h = dP_dV
            dh = d2P_dVdn
            k = d2P_dT2
            dk = d3P_dT2dn
            j = dP_dT
            dj = d2P_dTdn
            d3T_dPdVdn = 3*(f*g - h*k)*dj/j**4 - (f*dg + g*df - h*dk- k*dh)/j**3
            d3T_dPdVdns.append(d3T_dPdVdn)

            # tenth
            f = d2P_dTdV
            df = d3P_dTdVdn
            g = dP_dV
            dg = d2P_dVdn
            h = dP_dT
            dh = d2P_dTdn
            k = d2P_dV2
            dk = d3P_dV2dn
            j = dP_dV
            dj = d2P_dVdn
            d3V_dPdTdn = 3*(f*g - h*k)*dj/j**4 - (f*dg + g*df - h*dk- k*dh)/j**3
            d3V_dPdTdns.append(d3V_dPdTdn)





        return (d2P_dTdns, d2P_dVdns, d2V_dTdns, d2V_dPdns, d2T_dVdns, d2T_dPdns,
                d3P_dT2dns, d3P_dV2dns, d3V_dT2dns, d3V_dP2dns, d3T_dV2dns, d3T_dP2dns,
                d3V_dPdTdns, d3P_dTdVdns, d3T_dPdVdns)

    def set_dnzs_derivatives_and_departures(self, n=True, x=True, only_l=False,
                                           only_g=False):
        r'''Sets a number of mole number and/or composition partial derivatives
        of thermodynamic partial derivatives.

        The list of properties set is as follows, with all properties suffixed
        with '_l' or '_g'

        if `n` is True:
        d2P_dTdns, d2P_dVdns, d2V_dTdns, d2V_dPdns, d2T_dVdns, d2T_dPdns,
        d3P_dT2dns, d3P_dV2dns, d3V_dT2dns, d3V_dP2dns, d3T_dV2dns, d3T_dP2dns,
        d3V_dPdTdns, d3P_dTdVdns, d3T_dPdVdns, dV_dep_dns, dG_dep_dns,
        dH_dep_dns, dU_dep_dns, dS_dep_dns, dA_dep_dns

        if `x` is True:
        d2P_dTdzs, d2P_dVdzs, d2V_dTdzs, d2V_dPdzs, d2T_dVdzs, d2T_dPdzs,
        d3P_dT2dzs, d3P_dV2dzs, d3V_dT2dzs, d3V_dP2dzs, d3T_dV2dzs, d3T_dP2dzs,
        d3V_dPdTdzs, d3P_dTdVdzs, d3T_dPdVdzs, dV_dep_dzs, dG_dep_dzs,
        dH_dep_dzs, dU_dep_dzs, dS_dep_dzs, dA_dep_dzs

        Parameters
        ----------
        n : bool, optional
            Whether or not to set the mole number derivatives (sums up to one),
            [-]
        x : bool, optional
            Whether or not to set the composition derivatives (does not sum up
            to one), [-]
        only_l : bool, optional
            Whether or not to set only the liquid-like phase properties (if
            there are two phases), [-]
        only_g : bool, optional
            Whether or not to set only the gas-like phase properties (if
            there are two phases), [-]

        Notes
        -----
        '''
        N = self.N
        zs = self.zs
        T, P = self.T, self.P
        if n and x:
            ns = [True, False]
        elif n:
            ns = [True]
        elif x:
            ns = [False]
        else:
            return

        if only_l:
            phases = ['l']
        elif only_g:
            phases = ['g']
        else:
            phases = ['l', 'g']


        for n in ns:
            for phase in phases:
                if phase == 'g':
                    Z, V = self.Z_g, self.V_g
                else:
                    Z, V = self.Z_l, self.V_l

                if n:
                    V_fun, G_fun, H_fun = self.dV_dns, self.dG_dep_dns, self.dH_dep_dns
                else:
                    V_fun, G_fun, H_fun = self.dV_dzs, self.dG_dep_dzs, self.dH_dep_dzs

                (d2P_dTdns, d2P_dVdns, d2V_dTdns, d2V_dPdns, d2T_dVdns, d2T_dPdns,
                 d3P_dT2dns, d3P_dV2dns, d3V_dT2dns, d3V_dP2dns, d3T_dV2dns, d3T_dP2dns,
                 d3V_dPdTdns, d3P_dTdVdns, d3T_dPdVdns) = self._dnz_derivatives_and_departures(V, n=n)

                # V
                dV_dep_dns = V_fun(Z)
                # G
                dG_dep_dns = G_fun(Z)
                # H
                dH_dep_dns = H_fun(Z)
                # U
                dU_dep_dns = [dH_dep_dns[i] - P*dV_dep_dns[i] for i in range(N)]
                # S
                dS_dep_dns = [(dG_dep_dns[i] - dH_dep_dns[i])/-T for i in range(N)]
                # A
                dA_dep_dns = [dU_dep_dns[i] - T*dS_dep_dns[i] for i in range(N)]

                if n and phase == 'l':
                    self.d2P_dTdns_l, self.d2P_dVdns_l, self.d2V_dTdns_l = d2P_dTdns, d2P_dVdns, d2V_dTdns
                    self.d2V_dPdns_l, self.d2T_dVdns_l, self.d2T_dPdns_l = d2V_dPdns, d2T_dVdns, d2T_dPdns
                    self.d3P_dT2dns_l, self.d3P_dV2dns_l, self.d3V_dT2dns_l = d3P_dT2dns, d3P_dV2dns, d3V_dT2dns
                    self.d3V_dP2dns_l, self.d3T_dV2dns_l, self.d3T_dP2dns_l = d3V_dP2dns, d3T_dV2dns, d3T_dP2dns
                    self.d3V_dPdTdns_l, self.d3P_dTdVdns_l, self.d3T_dPdVdns_l = d3V_dPdTdns, d3P_dTdVdns, d3T_dPdVdns

                    self.dV_dep_dns_l, self.dG_dep_dns_l, self.dH_dep_dns_l = dV_dep_dns, dG_dep_dns, dH_dep_dns
                    self.dU_dep_dns_l, self.dS_dep_dns_l, self.dA_dep_dns_l = dU_dep_dns, dS_dep_dns, dA_dep_dns
                if n and phase == 'g':
                    self.d2P_dTdns_g, self.d2P_dVdns_g, self.d2V_dTdns_g = d2P_dTdns, d2P_dVdns, d2V_dTdns
                    self.d2V_dPdns_g, self.d2T_dVdns_g, self.d2T_dPdns_g = d2V_dPdns, d2T_dVdns, d2T_dPdns
                    self.d3P_dT2dns_g, self.d3P_dV2dns_g, self.d3V_dT2dns_g = d3P_dT2dns, d3P_dV2dns, d3V_dT2dns
                    self.d3V_dP2dns_g, self.d3T_dV2dns_g, self.d3T_dP2dns_g = d3V_dP2dns, d3T_dV2dns, d3T_dP2dns
                    self.d3V_dPdTdns_g, self.d3P_dTdVdns_g, self.d3T_dPdVdns_g = d3V_dPdTdns, d3P_dTdVdns, d3T_dPdVdns

                    self.dV_dep_dns_g, self.dG_dep_dns_g, self.dH_dep_dns_g = dV_dep_dns, dG_dep_dns, dH_dep_dns
                    self.dU_dep_dns_g, self.dS_dep_dns_g, self.dA_dep_dns_g = dU_dep_dns, dS_dep_dns, dA_dep_dns
                if not n and phase == 'g':
                    self.d2P_dTdzs_g, self.d2P_dVdzs_g, self.d2V_dTdzs_g = d2P_dTdns, d2P_dVdns, d2V_dTdns
                    self.d2V_dPdzs_g, self.d2T_dVdzs_g, self.d2T_dPdzs_g = d2V_dPdns, d2T_dVdns, d2T_dPdns
                    self.d3P_dT2dzs_g, self.d3P_dV2dzs_g, self.d3V_dT2dzs_g = d3P_dT2dns, d3P_dV2dns, d3V_dT2dns
                    self.d3V_dP2dzs_g, self.d3T_dV2dzs_g, self.d3T_dP2dzs_g = d3V_dP2dns, d3T_dV2dns, d3T_dP2dns
                    self.d3V_dPdTdzs_g, self.d3P_dTdVdzs_g, self.d3T_dPdVdzs_g = d3V_dPdTdns, d3P_dTdVdns, d3T_dPdVdns

                    self.dV_dep_dzs_g, self.dG_dep_dzs_g, self.dH_dep_dzs_g = dV_dep_dns, dG_dep_dns, dH_dep_dns
                    self.dU_dep_dzs_g, self.dS_dep_dzs_g, self.dA_dep_dzs_g = dU_dep_dns, dS_dep_dns, dA_dep_dns
                if not n and phase == 'l':
                    self.d2P_dTdzs_l, self.d2P_dVdzs_l, self.d2V_dTdzs_l = d2P_dTdns, d2P_dVdns, d2V_dTdns
                    self.d2V_dPdzs_l, self.d2T_dVdzs_l, self.d2T_dPdzs_l = d2V_dPdns, d2T_dVdns, d2T_dPdns
                    self.d3P_dT2dzs_l, self.d3P_dV2dzs_l, self.d3V_dT2dzs_l = d3P_dT2dns, d3P_dV2dns, d3V_dT2dns
                    self.d3V_dP2dzs_l, self.d3T_dV2dzs_l, self.d3T_dP2dzs_l = d3V_dP2dns, d3T_dV2dns, d3T_dP2dns
                    self.d3V_dPdTdzs_l, self.d3P_dTdVdzs_l, self.d3T_dPdVdzs_l = d3V_dPdTdns, d3P_dTdVdns, d3T_dPdVdns

                    self.dV_dep_dzs_l, self.dG_dep_dzs_l, self.dH_dep_dzs_l = dV_dep_dns, dG_dep_dns, dH_dep_dns
                    self.dU_dep_dzs_l, self.dS_dep_dzs_l, self.dA_dep_dzs_l = dU_dep_dns, dS_dep_dns, dA_dep_dns

    def dlnphis_dP(self, phase):
        r'''Generic formula for calculating the pressure derivaitve of
        log fugacity coefficients for each species in a mixture. Verified
        numerically. Applicable to all cubic equations of state which can be
        cast in the form used here.

        Normally this routine is slower than EOS-specific ones, as it does not
        make assumptions that certain parameters are zero or equal to other
        parameters.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial P}\right)_{T,
            nj \ne i} = \frac{G_{dep}}{\partial P}_{T, n}
            +  \left(\frac{\partial^2 \ln \phi}{\partial P \partial n_i}
            \right)_{T, P, n_{j \ne i}}

        Parameters
        ----------
        phase : str
            One of 'l' or 'g', [-]

        Returns
        -------
        dlnphis_dP : float
            Pressure derivatives of log fugacity coefficient for each species,
            [1/Pa]

        Notes
        -----
        This expression for the partial derivative of the mixture `lnphi` with
        respect to pressure and mole number can be derived as follows; to
        convert to the partial molar `lnphi` pressure and temperature
        derivative, add ::math::`\frac{G_{dep}/(RT)}{\partial P}_{T, n}`.

        >>> from sympy import * # doctest:+SKIP
        >>> P, T, R, n = symbols('P, T, R, n') # doctest:+SKIP
        >>> a_alpha, a, delta, epsilon, V, b, da_alpha_dT, d2a_alpha_dT2 = symbols('a_alpha, a, delta, epsilon, V, b, da_alpha_dT, d2a_alpha_dT2', cls=Function) # doctest:+SKIP
        >>> S_dep = R*log(P*V(n, P)/(R*T)) + R*log(V(n, P)-b(n))+2*da_alpha_dT(n, T)*atanh((2*V(n, P)+delta(n))/sqrt(delta(n)**2-4*epsilon(n)))/sqrt(delta(n)**2-4*epsilon(n))-R*log(V(n, P)) # doctest:+SKIP
        >>> H_dep = P*V(n, P) - R*T + 2*atanh((2*V(n, P)+delta(n))/sqrt(delta(n)**2-4*epsilon(n)))*(da_alpha_dT(n, T)*T-a_alpha(n, T))/sqrt(delta(n)**2-4*epsilon(n)) # doctest:+SKIP
        >>> G_dep = H_dep - T*S_dep # doctest:+SKIP
        >>> lnphi = simplify(G_dep/(R*T)) # doctest:+SKIP
        >>> diff(diff(lnphi, P), n) # doctest:+SKIP
        P*Derivative(V(n, P), P, n)/(R*T) + Derivative(V(n, P), P, n)/V(n, P) - Derivative(V(n, P), P)*Derivative(V(n, P), n)/V(n, P)**2 - Derivative(V(n, P), P, n)/(V(n, P) - b(n)) - (-Derivative(V(n, P), n) + Derivative(b(n), n))*Derivative(V(n, P), P)/(V(n, P) - b(n))**2 + Derivative(V(n, P), n)/(R*T) - 4*(-2*delta(n)*Derivative(delta(n), n) + 4*Derivative(epsilon(n), n))*a_alpha(n, T)*Derivative(V(n, P), P)/(R*T*(1 - (2*V(n, P)/sqrt(delta(n)**2 - 4*epsilon(n)) + delta(n)/sqrt(delta(n)**2 - 4*epsilon(n)))**2)*(delta(n)**2 - 4*epsilon(n))**2) - 4*a_alpha(n, T)*Derivative(V(n, P), P, n)/(R*T*(1 - (2*V(n, P)/sqrt(delta(n)**2 - 4*epsilon(n)) + delta(n)/sqrt(delta(n)**2 - 4*epsilon(n)))**2)*(delta(n)**2 - 4*epsilon(n))) - 4*Derivative(V(n, P), P)*Derivative(a_alpha(n, T), n)/(R*T*(1 - (2*V(n, P)/sqrt(delta(n)**2 - 4*epsilon(n)) + delta(n)/sqrt(delta(n)**2 - 4*epsilon(n)))**2)*(delta(n)**2 - 4*epsilon(n))) - 4*(2*V(n, P)/sqrt(delta(n)**2 - 4*epsilon(n)) + delta(n)/sqrt(delta(n)**2 - 4*epsilon(n)))*(4*(-delta(n)*Derivative(delta(n), n) + 2*Derivative(epsilon(n), n))*V(n, P)/(delta(n)**2 - 4*epsilon(n))**(3/2) + 2*(-delta(n)*Derivative(delta(n), n) + 2*Derivative(epsilon(n), n))*delta(n)/(delta(n)**2 - 4*epsilon(n))**(3/2) + 4*Derivative(V(n, P), n)/sqrt(delta(n)**2 - 4*epsilon(n)) + 2*Derivative(delta(n), n)/sqrt(delta(n)**2 - 4*epsilon(n)))*a_alpha(n, T)*Derivative(V(n, P), P)/(R*T*(1 - (2*V(n, P)/sqrt(delta(n)**2 - 4*epsilon(n)) + delta(n)/sqrt(delta(n)**2 - 4*epsilon(n)))**2)**2*(delta(n)**2 - 4*epsilon(n))) + R*T*(P*Derivative(V(n, P), P)/(R*T) + V(n, P)/(R*T))*Derivative(V(n, P), n)/(P*V(n, P)**2) - R*T*(P*Derivative(V(n, P), P, n)/(R*T) + Derivative(V(n, P), n)/(R*T))/(P*V(n, P))
        '''
        if phase == 'g':
            V = self.V_g
            Z = self.Z_g
            dV_dP = self.dV_dP_g
            dG_dep_dP = (self.dH_dep_dP_g  - self.T*self.dS_dep_dP_g)/(R*self.T)

        else:
            V = self.V_l
            Z = self.Z_l
            dV_dP = self.dV_dP_l
            dG_dep_dP = (self.dH_dep_dP_l  - self.T*self.dS_dep_dP_l)/(R*self.T)

        T = self.T
        P = self.P
        dV_dns = self.dV_dns(Z)
        ddelta_dns = self.ddelta_dns
        depsilon_dns = self.depsilon_dns
        da_alpha_dns = self.da_alpha_dns
        db_dns = self.db_dns
        d2V_dPdns = self._dnz_derivatives_and_departures(V)[3]# self.d2V_dPdn

        x0 = V
        x2 = 1/(R*T)
        x3 = 1/x0
        x6 = dV_dP
        x8 = self.b
        x9 = x0 - x8
        x10 = 1/P
        x11 = self.delta
        x12 = 2*x0
        x13 = x11 + x12
        x14 = self.epsilon
        x15 = x11**2 - 4*x14
        try:
            x16 = 1/x15
        except ZeroDivisionError:
            x16 = 1e50
        x17 = x13**2*x16 - 1
        x18 = 1/x17
        x19 = self.a_alpha
        x20 = 4*x16
        x21 = x2*x6
        x22 = x18*x21
        x25 = 8*x19*x16*x16

        t50 = 1.0/(x0*x0)

        dlnphis_dPs = []
        for i in range(self.N):
            # number dependent calculations
            x1 = dV_dns[i] # Derivative(x0, n)
            x7 = x1*t50

            x4 = d2V_dPdns[i] #Derivative(x0, P, n) # TODO calculate only this - d2V_dPdn; the T one wants d2V_dTdn
            x5 = P*x4

            x23 = ddelta_dns[i]# Derivative(x11, n)
            x24 = x11*x23 - 2.0*depsilon_dns[i]#Derivative(x14, n)
            x26 = x16*x24

            dlnphi_dP = (x1*x2 - x10*x3*(x1 + x5) + x10*x7*(P*x6 + x0)
            - x13*x21*x25*(2*x1 - x11*x26 - x12*x26 + x23)/x17**2
            + x18*x19*x2*x20*x4 + x2*x5 + x20*x22*da_alpha_dns[i]
            - x22*x24*x25 + x3*x4 - x4/x9 - x6*x7 + x6*(x1 - db_dns[i])/x9**2)
            dlnphis_dPs.append(dlnphi_dP + dG_dep_dP)
        return dlnphis_dPs


    def dlnphis_dT(self, phase):
        r'''Generic formula for calculating the temperature derivaitve of
        log fugacity coefficients for each species in a mixture. Verified
        numerically. Applicable to all cubic equations of state which can be
        cast in the form used here.

        Normally this routine is slower than EOS-specific ones, as it does not
        make assumptions that certain parameters are zero or equal to other
        parameters.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial T}\right)_{P,
            nj \ne i} = \frac{\frac{G_{dep}}{RT}}{\partial T}_{P, n}
            +  \left(\frac{\partial^2 \ln \phi}{\partial T \partial n_i}
            \right)_{P, n_{j \ne i}}

        Parameters
        ----------
        phase : str
            One of 'l' or 'g', [-]

        Returns
        -------
        dlnphis_dT : float
            Temperature derivatives of log fugacity coefficient for each species,
            [1/K]

        Notes
        -----
        This expression for the partial derivative of the mixture `lnphi` with
        respect to pressure and mole number can be derived as follows; to
        convert to the partial molar `lnphi` pressure and temperature
        derivative, add ::math::`\frac{G_{dep}/(RT)}{\partial T}_{P, n}`.

        >>> from sympy import * # doctest:+SKIP
        >>> P, T, R, n = symbols('P, T, R, n') # doctest:+SKIP
        >>> a_alpha, a, delta, epsilon, V, b, da_alpha_dT, d2a_alpha_dT2 = symbols('a_alpha, a, delta, epsilon, V, b, da_alpha_dT, d2a_alpha_dT2', cls=Function) # doctest:+SKIP
        >>> S_dep = R*log(P*V(n, T)/(R*T)) + R*log(V(n, T)-b(n))+2*da_alpha_dT(n, T)*atanh((2*V(n, T)+delta(n))/sqrt(delta(n)**2-4*epsilon(n)))/sqrt(delta(n)**2-4*epsilon(n))-R*log(V(n, T)) # doctest:+SKIP
        >>> H_dep = P*V(n, T) - R*T + 2*atanh((2*V(n, T)+delta(n))/sqrt(delta(n)**2-4*epsilon(n)))*(da_alpha_dT(n, T)*T-a_alpha(n, T))/sqrt(delta(n)**2-4*epsilon(n)) # doctest:+SKIP
        >>> G_dep = H_dep - T*S_dep # doctest:+SKIP
        >>> lnphi = simplify(G_dep/(R*T)) # doctest:+SKIP
        >>> diff(diff(lnphi, T), n) # doctest:+SKIP
        '''
        T, P, zs, N = self.T, self.P, self.zs, self.N
        if phase == 'g':
            V = self.V_g
            Z = self.Z_g
            dV_dT = self.dV_dT_g
            dG_dep_dT = (-T*self.dS_dep_dT_g - self.S_dep_g + self.dH_dep_dT_g)/(R*self.T)
            dG_dep_dT -= (-T*self.S_dep_g + self.H_dep_g)/(R*self.T*self.T)
        else:
            V = self.V_l
            Z = self.Z_l
            dV_dT = self.dV_dT_l
            dG_dep_dT = (-T*self.dS_dep_dT_l - self.S_dep_l + self.dH_dep_dT_l)/(R*self.T)
            dG_dep_dT -= (-T*self.S_dep_l + self.H_dep_l)/(R*self.T*self.T)
        '''R, T = symbols('R, T')
        H, S = symbols('H, S', cls=Function)
        print(diff((H(T) - T*S(T))/(R*T), T))
        # (-T*Derivative(S(T), T) - S(T) + Derivative(H(T), T))/(R*T) - (-T*S(T) + H(T))/(R*T**2)
        '''

        d2V_dTdns = self._dnz_derivatives_and_departures(V, n=True)[2]
        dV_dns = self.dV_dns(Z)
        db_dns = self.db_dns
        da_alpha_dns = self.da_alpha_dns
        da_alpha_dT_dns = self.da_alpha_dT_dns
        ddelta_dns = self.ddelta_dns
        depsilon_dns = self.depsilon_dns

        x0 = V
        x1 = 1/x0
        x4 = T**(-2)
        x5 = 1/R
        x6 = P*x5
        x7 = 1/T
        x9 = dV_dT
        x11 = self.b
        x12 = x0 - x11
        x13 = self.a_alpha
        x15 = self.delta
        x16 = self.epsilon
        x17 = x15*x15 - 4.0*x16
        if x17 == 0.0:
            x17 = 1e-100
        x18 = 1/sqrt(x17)
        x19 = 2*x0
        x20 = x15 + x19
        x21 = 2*x5
        x22 = x21*catanh(x18*x20).real
        x23 = x18*x22
        x24 = 1/x17
        x25 = x20**2*x24 - 1
        x26 = 1/x25
        x27 = x24*x26
        x28 = 4*x27*x5
        x29 = x7*x9
        x30 = x13*x4
        x34 = x7*self.da_alpha_dT
        x35 = 8*x13*x29*x5/x17**2

        dlnphis_dTs = []
        for i in range(N):
            x2 = d2V_dTdns[i]
            x8 = x2*x7

            x3 = dV_dns[i]
            x10 = x3/x0**2
            x14 = da_alpha_dns[i]
            x31 = ddelta_dns[i]
            x32 = x15*x31 - 2.0*depsilon_dns[i]
            x33 = x22*x32/x17**(3/2)
            x36 = x24*x32

            x37 = -x15*x36 - x19*x36 + 2.0*x3 + x31
            x38 = x21*x27*x37

            dlnphi_dT = (x1*x2 - x1*(x2 - x3*x7) - x10*x9 + x10*(-x0*x7 + x9)
            + x13*x28*x8 + x14*x23*x4 + x14*x28*x29 - x20*x35*x37/x25**2
            - x23*x7*da_alpha_dT_dns[i] - x26*x32*x35 - x3*x4*x6 - x30*x33
            - x30*x38 + x33*x34 + x34*x38 + x6*x8 - x2/x12 + x9*(x3 - db_dns[i])/x12**2)
            dlnphis_dTs.append(dlnphi_dT + dG_dep_dT)
        return dlnphis_dTs

    def dlnphis_dzs(self, Z):
        r'''Generic formula for calculating the mole fraction derivaitves of
        log fugacity coefficients for each species in a mixture. Verified
        numerically. Applicable to all cubic equations of state which can be
        cast in the form used here.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial z_i}\right)_{P,
            z_{j \ne i}}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dlnphis_dzs : list[list[float]]
            Mole fraction derivatives of log fugacity coefficient for each
            species (such that the mole fractions do not sum to 1), [-]

        Notes
        -----
        '''
        d2dxs = self.d2lnphi_dzizjs(Z)
        d2ns = d2xs_to_dxdn_partials(d2dxs, self.zs)
        if sefl.scalar:
            return d2ns
        return array(d2ns)

class EpsilonZeroMixingRules(object):
    @property
    def depsilon_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \epsilon}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = 0

        Returns
        -------
        depsilon_dzs : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        if self.scalar:
            return [0.0]*self.N
        return zeros(self.N)

    @property
    def depsilon_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \epsilon}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = 0

        Returns
        -------
        depsilon_dns : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^3]

        Notes
        -----
        This derivative is checked numerically.
        '''
        if self.scalar:
            return [0.0]*self.N
        return zeros(self.N)

    @property
    def d2epsilon_dzizjs(self):
        r'''Helper method for calculating the second composition derivatives (hessian)
        of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial x_i \partial x_j}\right)_{T, P, x_{k\ne i,j}}
            = 0

        Returns
        -------
        d2epsilon_dzizjs : list[list[float]]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            return [[0.0]*N for i in range(N)]
        return zeros((N, N))

    @property
    def d2epsilon_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial n_i n_j}\right)_{T, P, n_{k\ne i,j}}
            = 0

        Returns
        -------
        d2epsilon_dninjs : list[list[float]]
            Second composition derivative of `epsilon` of each component, [m^6/mol^4]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            return [[0.0]*N for i in range(N)]
        return zeros((N, N))

    @property
    def d3epsilon_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \epsilon}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = 0

        Returns
        -------
        d3epsilon_dninjnks : list[list[list[float]]]
            Third mole number derivative of `epsilon` of each component,
            [m^6/mol^5]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            return [[[0.0]*N for _ in range(N)] for _ in range(N)]
        return zeros((N, N, N))

#    # Python 2/3 compatibility
#    try:
#        eos.__dict__['d3epsilon_dninjnks'] = d3epsilon_dninjnks
#        eos.__dict__['d2epsilon_dninjs'] = d2epsilon_dninjs
#        eos.__dict__['d2epsilon_dzizjs'] = d2epsilon_dzizjs
#        eos.__dict__['depsilon_dns'] = depsilon_dns
#        eos.__dict__['depsilon_dzs'] = depsilon_dzs
#    except:
#        setattr(eos, 'd3epsilon_dninjnks', d3epsilon_dninjnks)
#        setattr(eos, 'd2epsilon_dninjs', d2epsilon_dninjs)
#        setattr(eos, 'd2epsilon_dzizjs', d2epsilon_dzizjs)
#        setattr(eos, 'depsilon_dns', depsilon_dns)
#        setattr(eos, 'depsilon_dzs', depsilon_dzs)




class PSRKMixingRules(object):
    u = 1.1
    A = -0.6466271649250525 # log(1.1/(1.1+1))
    A_inv = 1.0/A
    def a_alpha_and_derivatives(self, T, full=True, quick=True,
                                pure_a_alphas=True):

        r'''Method to calculate `a_alpha` and its first and second
        derivatives for an EOS with the PSRK mixing rules. Returns
        `a_alpha`, `da_alpha_dT`, and `d2a_alpha_dT2`.

        For use in some methods, this returns only `a_alpha` if `full` is False.

        .. math::
            \alpha = bRT \left[ \sum_i \frac{z_i \alpha_i}{b_i RT}
            + \frac{1}{A}\left(\frac{G^E}{RT} + \sum_i z_i \ln
            \left(\frac{b}{b_i}\right) \right)\right]

        .. math::
            \frac{\partial \alpha}{\partial T} = RTb\left[
            \sum_i \left(\frac{z_i \frac{\partial \alpha_i}{\partial T}}{RTb_i}
            -\frac{z_i\alpha_i}{RT^2b_i} \right)
            + \frac{1}{A}\left(\frac{\frac{\partial G^E}{\partial T}}{RT}
            - \frac{G^E}{RT^2} \right)
            \right] + \frac{\alpha}{T}

        .. math::
            \frac{\partial^2 \alpha}{\partial T^2} = b\left[\sum_i
            \left(\frac{z_i\frac{\partial^2 \alpha_i}{\partial T^2}}{b_i}
            - \frac{2z_i \frac{\partial \alpha_i}{\partial T}}{T b_i}
            + \frac{2z_i\alpha_i}{T^2 b_i}
            \right)
            + \frac{2}{T}\left[\sum_i \left(\frac{z_i\frac{\partial \alpha_i}
            {\partial T}}{b_i}
            - \frac{z_i \alpha_i}{T b_i}
            \right)
            + \frac{1}{A}\left(\frac{\partial G^E}{\partial T} - \frac{G^E}{T}
            \right)
            \right]
            + \frac{1}{A}\left(
            \frac{\partial^2 G^E}{\partial T^2} - \frac{2}{T}
            \frac{\partial G^E}{\partial T} + 2\frac{G^E}{T^2}
            \right)
            \right]

        Parameters
        ----------
        T : float
            Temperature, [K]
        full : bool, optional
            If False, calculates and returns only `a_alpha`
        quick : bool, optional
            Only the quick variant is implemented; it is little faster anyhow
        pure_a_alphas : bool, optional
            Whether or not to recalculate the a_alpha terms of pure components
            (for the case of mixtures only) which stay the same as the
            composition changes (i.e in a PT flash), [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by PSRK-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by PSRK-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            PSRK-specific method, [J^2/mol^2/Pa/K**2]

        Notes
        -----
        '''
        if pure_a_alphas:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = self.a_alpha_and_derivatives_vectorized(T)
            self.a_alphas, self.da_alpha_dTs, self.d2a_alpha_dT2s = a_alphas, da_alpha_dTs, d2a_alpha_dT2s
        else:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = self.a_alphas, self.da_alpha_dTs, self.d2a_alpha_dT2s

        b, zs, bs = self.b, self.zs, self.bs

        ge_model = self.ge_model

        if T != ge_model.T:
            # TODO make sure this gets set when solve_T is called
            ge_model = ge_model.to_T_xs(T, zs)
            self._last_ge = ge_model

        GE = ge_model.GE()
        if full:
            dGE_dT = ge_model.dGE_dT()
            d2GE_dT2 = ge_model.d2GE_dT2()

        T_inv = 1.0/T
        T2_inv = T_inv*T_inv
        RT_inv = R_inv*T_inv
        RT2_inv = R_inv*T2_inv

        A_inv = self.A_inv
        N = self.N


        tot0, tot1, d1tot, d2tot, other = 0.0, 0.0, 0.0, 0.0, 0.0
        if full:
            for i in range(N):
                bi_inv = 1.0/bs[i]
                # Main component
                tot0 += zs[i]*a_alphas[i]*bi_inv*RT_inv
                tot1 += zs[i]*log(b*bi_inv)


                d1tot += zs[i]*da_alpha_dTs[i]*RT_inv*bi_inv - zs[i]*a_alphas[i]*RT2_inv*bi_inv

                # TODO go back to just using d1tot
                # TODO optimize all of this
                other += zs[i]*da_alpha_dTs[i]*bi_inv - zs[i]*a_alphas[i]*bi_inv*T_inv

                d2tot += (zs[i]*d2a_alpha_dT2s[i]*bi_inv
                          - 2.0*zs[i]*da_alpha_dTs[i]*T_inv*bi_inv
                          + 2.0*zs[i]*a_alphas[i]*T2_inv*bi_inv)
        else:
            for i in range(N):
                bi_inv = 1.0/bs[i]
                tot0 += zs[i]*a_alphas[i]*bi_inv*RT_inv
                tot1 += zs[i]*log(b*bi_inv)


        a_alpha = R*T*b*(tot0 + A_inv*(GE*RT_inv + tot1))
        if full:
            da_alpha_dT = R*T*b*(d1tot + A_inv*(dGE_dT*RT_inv - GE*RT2_inv)) + a_alpha*T_inv
            d2a_alpha_dT2 = b*(d2tot + 2.0*T_inv*(other + A_inv*(dGE_dT - GE*T_inv))
                               + A_inv*(d2GE_dT2 - 2.0*T_inv*dGE_dT + 2.0*GE*T2_inv))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
        return a_alpha

    def solve_T(self, P, V, quick=True, solution=None):
        T = GCEOS.solve_T(self, P, V, solution=solution)
        if hasattr(self, '_last_ge') and self._last_ge.T == T:
            self.ge_model = self._last_ge
            del self._last_ge
        else:
            self.ge_model = self.ge_model.to_T_xs(T, self.zs)
        return T

    @property
    def da_alpha_dzs(self):
        raise NotImplementedError("TODO")

    @property
    def da_alpha_dns(self):
        raise NotImplementedError("TODO")

    @property
    def dna_alpha_dns(self):
        raise NotImplementedError("TODO")

    @property
    def d2a_alpha_dzizjs(self):
        raise NotImplementedError("TODO")

    @property
    def d2a_alpha_dninjs(self):
        raise NotImplementedError("TODO")

    @property
    def d3a_alpha_dzizjzks(self):
        raise NotImplementedError("TODO")

    @property
    def d3a_alpha_dninjnks(self):
        raise NotImplementedError("TODO")

    @property
    def da_alpha_dT_dzs(self):
        raise NotImplementedError("TODO")

    @property
    def da_alpha_dT_dns(self):
        raise NotImplementedError("TODO")

    @property
    def dna_alpha_dT_dns(self):
        raise NotImplementedError("TODO")

    @property
    def d2a_alpha_dT2_dzs(self):
        raise NotImplementedError("TODO")

    @property
    def d2a_alpha_dT2_dns(self):
        raise NotImplementedError("TODO")


class IGMIX(EpsilonZeroMixingRules, GCEOSMIX, IG):
    r'''Class for solving the ideal gas [1]_ [2]_ equation of state for a
    mixture of any number of compounds. Subclasses :obj:`thermo.eos.IG`. Solves
    the EOS on initialization.
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P =\frac{RT}{V}

    Parameters
    ----------
    zs : list[float]
        Overall mole fractions of all species, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    Tcs : list[float], optional
        Critical temperatures of all compounds, [K]
    Pcs : list[float], optional
        Critical pressures of all compounds, [Pa]
    omegas : list[float], optional
        Acentric factors of all compounds - Not used in this equation of
        state!, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 and not used[-]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = IGMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, .008], zs=[0.5, 0.5])
    >>> eos.phase, eos.V_g
    ('g', 0.0009561632010876225)

    Notes
    -----
    Many properties of this object are zero. Many of the arguments are not used
    and are provided for consistency only.

    References
    ----------
    .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
       Butterworth-Heinemann, 1985.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th
       edition. New York: McGraw-Hill Professional, 2000.
    '''
    eos_pure = IG
    a_alphas = None
    da_alpha_dTs = None
    d2a_alpha_dT2s = None
    nonstate_constants_specific = ()
    kwargs_keys = ('kijs',)

    model_id = 0

    def _zeros1d(self):
        return self.zeros1d

    def _zeros2d(self):
        return self.zeros2d

    def _zeros3d(self):
        N = self.N
        if self.scalar:
            return [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            return zeros((N, N, N))
    
    @property
    def a_alpha_roots(self):
        return self.zeros1d

    @property
    def ddelta_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \delta}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = 0

        Returns
        -------
        ddelta_dzs : list[float]
            Composition derivative of `delta` of each component, [m^3/mol]
        '''
        return self.zeros1d

    @property
    def ddelta_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \delta}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = 0

        Returns
        -------
        ddelta_dns : list[float]
            Mole number derivative of `delta` of each component, [m^3/mol^2]
        '''
        return self.zeros1d

    @property
    def depsilon_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \epsilon}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = 0

        Returns
        -------
        depsilon_dzs : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]
        '''
        return self.zeros1d

    @property
    def depsilon_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \epsilon}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = 0

        Returns
        -------
        depsilon_dns : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^3]
        '''
        return self.zeros1d

    @property
    def d2delta_dzizjs(self):
        r'''Helper method for calculating the second composition derivatives (hessian) of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \delta}{\partial x_i\partial x_j}\right)_{T, P, x_{k\ne i,j}}
            = 0

        Returns
        -------
        d2delta_dzizjs : list[float]
            Second Composition derivative of `delta` of each component, [m^3/mol]
        '''
        return self.zeros2d

    @property
    def d2delta_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \delta}{\partial n_i \partial n_j}\right)_{T, P, n_{k\ne i,j}}
            = 0

        Returns
        -------
        d2delta_dninjs : list[list[float]]
            Second mole number derivative of `delta` of each component, [m^3/mol^3]

        '''
        return self.zeros2d

    @property
    def d2epsilon_dzizjs(self):
        r'''Helper method for calculating the second composition derivatives (hessian)
        of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial x_i \partial x_j}\right)_{T, P, x_{k\ne i,j}}
            = 0

        Returns
        -------
        d2epsilon_dzizjs : list[list[float]]
            Second composition derivative of `epsilon` of each component, [m^6/mol^2]
        '''
        return self.zeros2d

    @property
    def d2epsilon_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives
        (hessian) of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial n_i n_j}\right)_{T, P,
            n_{k\ne i,j}}  = 0

        Returns
        -------
        d2epsilon_dninjs : list[list[float]]
            Second mole number derivative of `epsilon` of each component,
            [m^6/mol^4]
        '''
        return self.zeros2d

    @property
    def d3delta_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \delta}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = 0

        Returns
        -------
        d3delta_dninjnks : list[list[list[float]]]
            Third mole number derivative of `delta` of each component,
            [m^3/mol^4]
        '''
        return self._zeros3d()

    @property
    def d3epsilon_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \epsilon}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = 0

        Returns
        -------
        d3epsilon_dninjnks : list[list[list[float]]]
            Third mole number derivative of `epsilon` of each component,
            [m^6/mol^5]
        '''
        return self._zeros3d()

    def __init__(self, zs, T=None, P=None, V=None,
                 Tcs=None, Pcs=None, omegas=None, kijs=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(zs)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if scalar:
            self.zeros2d = zeros2d = [[0.0]*N for _ in range(N)]
        else:
            self.zeros2d = zeros2d = zeros((N, N))
        if kijs is None:
            kijs = zeros2d
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        self.b = 0.0
        self.bs = self.ais = self.zeros1d = self.a_alphas = self.da_alpha_dTs = self.d2a_alpha_dT2s = zeros2d[0]

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        self.bs = other.bs
        self.ais = other.ais
        self.b = other.b
        self.zeros1d = self.a_alphas = self.da_alpha_dTs = self.d2a_alpha_dT2s = other.zeros1d
        self.zeros2d = other.zeros2d

    def a_alphas_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` for the Ideal Gas
        EOS. This vectorized implementation is added for extra speed.

        .. math::
            a\alpha = 0

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        '''
        return self.zeros1d

    def a_alpha_and_derivatives_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` and their first
        and second derivatives for the Ideal Gas EOS. This vectorized
        implementation is added for extra speed.

        .. math::
            a\alpha = 0

        .. math::
            \frac{d a\alpha}{dT} = 0

        .. math::
            \frac{d^2 a\alpha}{dT^2} = 0

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dTs : list[float]
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2s : list[float]
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K**2]
        '''
        return self.zeros1d, self.zeros1d, self.zeros1d

    def a_alpha_and_derivatives(self, T, full=True, quick=True,
                                pure_a_alphas=True):
        # Saves time
        if full:
            return 0.0, 0.0, 0.0
        return 0.0
    try:
        a_alpha_and_derivatives.__doc__ = GCEOSMIX.a_alpha_and_derivatives.__doc__
    except:
        pass

    def fugacity_coefficients(self, Z):
        r'''Calculate and return the fugacity coefficients of the ideal-gas
        phase (0 by definition).

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        log_phis : float
            Log fugacity coefficient for each species, [-]
        '''
        return self.zeros1d

    def dlnphis_dT(self, phase):
        r'''Calculate and return the temperature derivative of fugacity
        coefficients of the ideal-gas phase (0 by definition).

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dlnphis_dT : float
            Temperature derivatives of log fugacity coefficient for each
            species, [1/K]
        '''
        return self.zeros1d

    def dlnphis_dP(self, phase):
        r'''Calculate and return the pressure derivative of fugacity
        coefficients of the ideal-gas phase (0 by definition).

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dlnphis_dP : float
            Pressure derivatives of log fugacity coefficient for each
            species, [1/Pa]
        '''
        return self.zeros1d

    @property
    def a_alpha_ijs(self):
        return self.zeros2d

    try:
        a_alpha_ijs.__doc__ = GCEOSMIX.a_alpha_ijs.__doc__
    except:
        pass

    @property
    def da_alpha_dT_ijs(self):
        return self.zeros2d
    try:
        da_alpha_dT_ijs.__doc__ = GCEOSMIX.da_alpha_dT_ijs.__doc__
    except:
        pass

    @property
    def d2a_alpha_dT2_ijs(self):
        return self.zeros2d
    try:
        d2a_alpha_dT2_ijs.__doc__ = GCEOSMIX.d2a_alpha_dT2_ijs.__doc__
    except:
        pass


class RKMIX(EpsilonZeroMixingRules, GCEOSMIX, RK):
    r'''Class for solving the Redlich Kwong [1]_ [2]_ cubic equation of state for a
    mixture of any number of compounds. Subclasses :obj:`thermo.eos.RK` . Solves the EOS on
    initialization and calculates fugacities for all components in all phases.
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P =\frac{RT}{V-b}-\frac{a}{V\sqrt{T}(V+b)}

    .. math::
        a = \sum_i \sum_j z_i z_j {a}_{ij}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_{ij} = (1-k_{ij})\sqrt{a_{i}a_{j}}

    .. math::
        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}

    .. math::
        b_i=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    omegas : float, optional
        Acentric factors of all compounds - Not used in this equation of
        state!, [-]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = RKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (4.048414781e-05, 0.00070060605863)

    Notes
    -----
    The PV solution for `T` is iterative.

    References
    ----------
    .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
       Butterworth-Heinemann, 1985.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th
       edition. New York: McGraw-Hill Professional, 2000.
    '''
    eos_pure = RK

    kwargs_keys = ('kijs',)
    model_id = 10002

    def __init__(self, Tcs, Pcs, zs, omegas=None, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in cmps]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        c1R2_c2R, c2R = self.c1R2_c2R, self.c2R
        if scalar:
            self.bs = bs = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*bs[i] for i in cmps]
            b = 0.0
            for i in cmps:
                b += bs[i]*zs[i]
        else:
            self.bs = bs = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*bs
            b = float((bs*zs).sum())

        self.b = self.delta = b


        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        b = 0.0
        if self.scalar:
            for bi, zi in zip(self.bs, self.zs):
                b += bi*zi
        else:
            b = float((self.bs*self.zs).sum())
        self.b = self.delta = b

    def a_alphas_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` for the RK EOS.
        This vectorized implementation is added for extra speed.

        .. math::
            a\alpha = \frac{a}{\sqrt{\frac{T}{Tc}}}

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

        Examples
        --------
        >>> eos = RKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        >>> eos.a_alphas_vectorized(115)
        [0.1449810919468, 0.30019773677]
        '''
        return RK_a_alphas_vectorized(T, self.Tcs, self.ais,
                                       a_alphas=[0.0]*self.N if self.scalar else zeros(self.N))

    def a_alpha_and_derivatives_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` and their first
        and second derivatives for the RK EOS. This vectorized implementation
        is added for extra speed.

        .. math::
            a\alpha = \frac{a}{\sqrt{\frac{T}{Tc}}}

        .. math::
            \frac{d a\alpha}{dT} = - \frac{a}{2 T\sqrt{\frac{T}{Tc}}}

        .. math::
            \frac{d^2 a\alpha}{dT^2} = \frac{3 a}{4 T^{2}\sqrt{\frac{T}{Tc}}}

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dTs : list[float]
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2s : list[float]
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K**2]

        Examples
        --------

        >>> eos = RKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        >>> eos.a_alpha_and_derivatives_vectorized(115)
        ([0.1449810919468, 0.30019773677], [-0.000630352573681, -0.00130520755121], [8.2219900915e-06, 1.7024446320e-05])
        '''
        N = self.N
        if self.scalar:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [0.0]*N, [0.0]*N, [0.0]*N
        else:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = zeros(N), zeros(N), zeros(N)
        return RK_a_alpha_and_derivatives_vectorized(T, self.Tcs, self.ais,
                                                     a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs, 
                                                     d2a_alpha_dT2s=d2a_alpha_dT2s)

    def solve_T(self, P, V, solution=None):
        if self.N == 1 and type(self) is RKMIX:
            self.Tc = self.Tcs[0]
            self.Pc = self.Pcs[0]
            self.a = self.ais[0]
            T = super(type(self).__mro__[-4], self).solve_T(P=P, V=V, solution=solution)
            del self.Tc
            del self.Pc
            del self.a
            return T
        else:
            return super(type(self).__mro__[-3], self).solve_T(P=P, V=V, solution=solution)


    @property
    def ddelta_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \delta}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = b_i

        Returns
        -------
        ddelta_dzs : list[float]
            Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        return self.bs

    @property
    def ddelta_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \delta}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = (b_i - b)

        Returns
        -------
        ddelta_dns : list[float]
            Mole number derivative of `delta` of each component, [m^3/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        b = self.b
        if self.scalar:
            return [(bi - b) for bi in self.bs]
        return self.bs - b

    @property
    def d2delta_dzizjs(self):
        r'''Helper method for calculating the second composition derivatives (hessian) of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \delta}{\partial x_i\partial x_j}\right)_{T, P, x_{k\ne i,j}}
            = 0

        Returns
        -------
        d2delta_dzizjs : list[float]
            Second Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            return [[0.0]*N for i in range(N)]
        else:
            return zeros((N, N))

    @property
    def d2delta_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \delta}{\partial n_i \partial n_j}\right)_{T, P, n_{k\ne i,j}}
            = 2b - b_i - b_j

        Returns
        -------
        d2delta_dninjs : list[list[float]]
            Second mole number derivative of `delta` of each component, [m^3/mol^3]

        Notes
        -----
        This derivative is checked numerically.
        '''
        return self.d2b_dninjs

    @property
    def d3delta_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \delta}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = 2(-3b + b_i + b_j + b_k)

        Returns
        -------
        d3delta_dninjnks : list[list[list[float]]]
            Third mole number derivative of `delta` of each component,
            [m^3/mol^4]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[[0.0]*N for _ in range(N) ] for _ in range(N)] if self.scalar else zeros((N, N, N))
        return RK_d3delta_dninjnks(self.b, self.bs, N, out)


class PRMIX(GCEOSMIX, PR):
    r'''Class for solving the Peng-Robinson [1]_ [2]_ cubic equation of state
    for a mixture of any number of compounds. Subclasses `PR`. Solves the EOS
    on initialization and calculates fugacities for all components in all
    phases.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}

    .. math::
	    b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}

    .. math::
        \alpha(T)_i=[1+\kappa_i(1-\sqrt{T_{r,i}})]^2

    .. math::
        \kappa_i=0.37464+1.54226\omega_i-0.26992\omega^2_i

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (3.6257362939e-05, 0.00070066592313)
    >>> eos.fugacities_l, eos.fugacities_g
    ([793860.83821, 73468.552253], [436530.92470, 358114.63827])

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    References
    ----------
    .. [1] Peng, Ding-Yu, and Donald B. Robinson. "A New Two-Constant Equation
       of State." Industrial & Engineering Chemistry Fundamentals 15, no. 1
       (February 1, 1976): 59-64. doi:10.1021/i160057a011.
    .. [2] Robinson, Donald B., Ding-Yu Peng, and Samuel Y-K Chung. "The
       Development of the Peng - Robinson Equation and Its Application to Phase
       Equilibrium in a System Containing Methanol." Fluid Phase Equilibria 24,
       no. 1 (January 1, 1985): 25-41. doi:10.1016/0378-3812(85)87035-7.
    '''
    eos_pure = PR

    nonstate_constants_specific = ('kappas', )
    kwargs_keys = ('kijs',)
    model_id = 10200

    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in cmps]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        # optimization, unfortunately
        c1R2_c2R, c2R = self.c1R2_c2R, self.c2R
        # Also tried to store the inverse of Pcs, without success - slows it down
        self.scalar = scalar = type(Tcs) is list
        if scalar:
            self.bs = bs = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*bs[i] for i in cmps]
            self.kappas = [omega*(-0.26992*omega + 1.54226) + 0.37464 for omega in omegas]
            b = 0.0
            for i in cmps:
                b += bs[i]*zs[i]
        else:
            self.bs = bs = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*bs
            self.kappas = omegas*(-0.26992*omegas + 1.54226) + 0.37464
            b = float((bs*zs).sum())
        self.b = b


        self.delta = 2.0*b
        self.epsilon = -b*b

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        self.kappas = other.kappas
        if self.scalar:
            b = 0.0
            for bi, zi in zip(self.bs, self.zs):
                b += bi*zi
        else:
            b = float((self.bs*self.zs).sum())
        self.b = b
        self.delta = 2.0*b
        self.epsilon = -b*b

    def a_alphas_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` for the PR EOS.
        This vectorized implementation is added for extra speed.

        .. math::
            a\alpha = a \left(\kappa \left(- \frac{T^{0.5}}{Tc^{0.5}}
            + 1\right) + 1\right)^{2}

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        '''
        return PR_a_alphas_vectorized(T, self.Tcs, self.ais, self.kappas,
                                      a_alphas=[0.0]*self.N if self.scalar else zeros(self.N))

    def a_alpha_and_derivatives_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` and their first
        and second derivatives for the PR EOS. This vectorized implementation
        is added for extra speed.

        .. math::
            a\alpha = a \left(\kappa \left(- \frac{T^{0.5}}{Tc^{0.5}}
            + 1\right) + 1\right)^{2}

        .. math::
            \frac{d a\alpha}{dT} = - \frac{1.0 a \kappa}{T^{0.5} Tc^{0.5}}
            \left(\kappa \left(- \frac{T^{0.5}}{Tc^{0.5}} + 1\right) + 1\right)

        .. math::
            \frac{d^2 a\alpha}{dT^2} = 0.5 a \kappa \left(- \frac{1}{T^{1.5}
            Tc^{0.5}} \left(\kappa \left(\frac{T^{0.5}}{Tc^{0.5}} - 1\right)
            - 1\right) + \frac{\kappa}{T^{1.0} Tc^{1.0}}\right)

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dTs : list[float]
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2s : list[float]
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K**2]
        '''
        N = self.N
        if self.scalar:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [0.0]*N, [0.0]*N, [0.0]*N
        else:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = zeros(N), zeros(N), zeros(N)
        return PR_a_alpha_and_derivatives_vectorized(T, self.Tcs, self.ais, self.kappas, a_alphas=a_alphas,
                                                     da_alpha_dTs=da_alpha_dTs, d2a_alpha_dT2s=d2a_alpha_dT2s)

    @property
    def d3a_alpha_dT3(self):
        r'''Method to calculate approximately the third temperature derivative
        of `a_alpha` for the PR EOS. A rigorous calculation has not been
        implemented.

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        d3a_alpha_dT3 : float
            Third temperature derivative :math:`a \alpha`, [J^2/mol^2/Pa/K^3]
        '''
        try:
            return self._d3a_alpha_dT3
        except AttributeError:
            pass
        tot = 0.0
        zs = self.zs
        vs = self.d3a_alpha_dT3_vectorized(self.T)
        for i in range(self.N):
            tot += zs[i]*vs[i]
        self._d3a_alpha_dT3 = tot
        return tot

    def d3a_alpha_dT3_vectorized(self, T):
        r'''Method to calculate the third temperature derivative of
        pure-component `a_alphas` for the PR EOS. This vectorized implementation
        is added for extra speed.

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        d3a_alpha_dT3s : list[float]
            Third temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^3]
        '''
        ais, kappas, Tcs = self.ais, self.kappas, self.Tcs
        T_inv = 1.0/T
        N = self.N

        d3a_alpha_dT3s = [0.0]*N if self.scalar else zeros(N)
        for i in range(N):
            kappa = kappas[i]

            x0 = 1.0/Tcs[i]
            x1 = sqrt(T*x0)
            v = (-ais[i]*0.75*kappa*(kappa*x0 - x1*(kappa*(x1 - 1.0) - 1.0)*T_inv)*T_inv*T_inv)
            d3a_alpha_dT3s[i] = v
        return d3a_alpha_dT3s

    def fugacity_coefficients(self, Z):
        r'''Literature formula for calculating fugacity coefficients for each
        species in a mixture. Verified numerically. Applicable to most
        derivatives of the Peng-Robinson equation of state as well.
        Called by :obj:`fugacities <GCEOSMIX.fugacities>` on initialization, or by a solver routine
        which is performing a flash calculation.

        .. math::
            \ln \hat \phi_i = \frac{B_i}{B}(Z-1)-\ln(Z-B) + \frac{A}{2\sqrt{2}B}
            \left[\frac{B_i}{B} - \frac{2}{a\alpha}\sum_i y_i(a\alpha)_{ij}\right]
            \ln\left[\frac{Z + (1+\sqrt{2})B}{Z-(\sqrt{2}-1)B}\right]

        .. math::
            A = \frac{(a\alpha)P}{R^2 T^2}

        .. math::
            B = \frac{b P}{RT}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        log_phis : float
            Log fugacity coefficient for each species, [-]
        '''
        a_alpha = self.a_alpha
#        a_alpha_ijs = self.a_alpha_ijs
        T_inv = 1.0/self.T
        bs, b = self.bs, self.b
        P_T = self.P*T_inv

        A = a_alpha*P_T*R2_inv*T_inv
        B = b*P_T*R_inv
        # The two log terms need to use a complex log; typically these are
        # calculated at "liquid" volume solutions which are unstable
        # and cannot exist
        try:
            x0 = log(Z - B)
        except ValueError:
            # less than zero
            x0 = 0.0

        root_two_B = B*root_two
        two_root_two_B = root_two_B + root_two_B

        ZB = Z + B

        try:
            x4 = A*log((ZB + root_two_B)/(ZB - root_two_B))
        except ValueError:
            # less than zero
            x4 = 0.0

        a_alpha_j_rows = self._a_alpha_j_rows
        try:
            t50 = 2.0*x4/(a_alpha*two_root_two_B)
        except ZeroDivisionError:
            return [0.0]*self.N
        t51 = (x4 + (Z - 1.0)*two_root_two_B)/(b*two_root_two_B)

        if self.scalar:
            return [bs[i]*t51 - x0 - t50*a_alpha_j_rows[i]
                    for i in range(self.N)]
        else:
            return bs*t51 - x0 - t50*a_alpha_j_rows

    def dlnphis_dT(self, phase):
        r'''Formula for calculating the temperature derivaitve of
        log fugacity coefficients for each species in a mixture for the
        Peng-Robinson equation of state. Verified numerically.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial T}\right)_{P,
            nj \ne i}

        Parameters
        ----------
        phase : str
            One of 'l' or 'g', [-]

        Returns
        -------
        dlnphis_dT : float
            Temperature derivatives of log fugacity coefficient for each species,
            [1/K]

        Notes
        -----
        This expression was derived using SymPy and optimized with the `cse`
        technique.
        '''
        zs = self.zs
        if phase == 'g':
            Z = self.Z_g
            dZ_dT = self.dZ_dT_g
        else:
            Z = self.Z_l
            dZ_dT = self.dZ_dT_l

        bs, b = self.bs, self.b

        T_inv = 1.0/self.T

        A = self.a_alpha*self.P*R2_inv*T_inv*T_inv

        B = b*self.P*R_inv*T_inv

        x2 = T_inv*T_inv
        x3 = R_inv

        x4 = self.P*b*x3
        x5 = x2*x4
        x8 = x4*T_inv

        x10 = self.a_alpha
        x11 = 1.0/self.a_alpha
        x12 = self.da_alpha_dT

        x13 = root_two
        x14 = 1.0/b

        x15 = x13 + 1.0 # root_two plus 1
        x16 = Z + x15*x8
        x17 = x13 - 1.0 # root two minus one
        x18 = x16/(x17*x8 - Z)
        x19 = log(-x18)

        x13x14 = x13*x14
        x10x13x14_4 = 0.25*x10*x13x14
        x19x3 = x19*x3

        x24 = x10x13x14_4*x19x3*x2
        x25 = 0.25*x12*x13x14*x19x3*T_inv
        x26 = x10x13x14_4*x3*T_inv*(-dZ_dT + x15*x5 - x18*(dZ_dT + x17*x5))/(x16)
        x50 = -0.5*x13x14*x19x3*T_inv
        x51 = -x11*x12
        x52 = (dZ_dT + x5)/(x8 - Z)
        x53 = 2.0*x11
        x54 = x52/x50
        x55 = x24 - x25 + x26

        x56 = dZ_dT/x55
        x57 = x53*x55
        x58 = x14*(dZ_dT - x55)
        x59 = x57/x50 + x51

        # Composition stuff

        a_alpha_j_rows = self._a_alpha_j_rows
        da_alpha_dT_j_rows = self._da_alpha_dT_j_rows

        d_lnphis_dTs = [x52 + bs[i]*x58 + x50*(x59*a_alpha_j_rows[i] + da_alpha_dT_j_rows[i])
                        for i in range(self.N)]
        return d_lnphis_dTs

    def dlnphis_dP(self, phase):
        r'''Generic formula for calculating the pressure derivaitve of
        log fugacity coefficients for each species in a mixture for the
        Peng-Robinson EOS. Verified numerically.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial P}\right)_{T,
            nj \ne i}

        Parameters
        ----------
        phase : str
            One of 'l' or 'g', [-]

        Returns
        -------
        dlnphis_dP : float
            Pressure derivatives of log fugacity coefficient for each species,
            [1/Pa]

        Notes
        -----
        This expression was derived using SymPy and optimized with the `cse`
        technique.
        '''
        zs = self.zs
        if phase == 'l':
            Z, dZ_dP = self.Z_l, self.dZ_dP_l
        else:
            Z, dZ_dP = self.Z_g, self.dZ_dP_g
        a_alpha = self.a_alpha
        bs, b = self.bs, self.b
        T_inv = 1.0/self.T

        x2 = 1.0/b
        x6 = b*R_inv*T_inv
        x8 = self.P*x6
        x9 = (dZ_dP - x6)/(x8 - Z)
        x13 = Z + root_two_p1*x8
        x15 = (a_alpha*root_two*x2*R_inv*T_inv*(dZ_dP + root_two_p1*x6
                + x13*(dZ_dP - root_two_m1*x6)/(root_two_m1*x8 - Z))/(4.0*x13))
        x16 = dZ_dP + x15

        a_alpha_j_rows = self._a_alpha_j_rows

        x50 = -2.0/a_alpha
        d_lnphi_dPs = []
        for i in range(self.N):
            x3 = bs[i]*x2
            x10 = x50*a_alpha_j_rows[i]
#            d_lnphi_dP = dZ_dP*x3 + x15*(x10 + x3) + x9
            d_lnphi_dP = x16*x3 + x15*x10 + x9
            d_lnphi_dPs.append(d_lnphi_dP)
        return d_lnphi_dPs



    def d_lnphi_dzs_analytical0(self, Z, zs):

        # TODO try to follow "B.5.2.1 Derivatives of Fugacity Coefficient with Respect to Mole Fraction"
        # "Development of an Equation-of-State Thermal Flooding Simulator"
        N = self.N
        cmps_m1 = range(N-1)

        a_alpha = self.a_alpha
        a_alpha_ijs = self.a_alpha_ijs
        T2 = self.T*self.T
        b = self.b

        A = a_alpha*self.P/(R2*T2)
        B = b*self.P/(R*self.T)
        B2 = B*B
        Z2 = Z*Z
        A_B = A/B
        ZmB = Z - B


        dZ_dA = (B - Z)/(3.0*Z2 - 2.0*(1.0 - B)*Z + (A - 2.0*B - 3.0*B2))

        # 2*(3.0*B + 1)*Z may or may not have Z
        # Simple phase stability-testing algorithm in the reduction method.
        dZ_dB = ((-Z2 + 2*(3.0*B + 1)*Z) + (A - 2.0*B - 3.0*B2))/(
                3.0*Z2 - 2.0*(1.0 - B)*Z + (A - 2.0*B - 3.0*B2))


        Sis = []
        for i in range(N):
            tot = 0.0
            for j in range(N):
                tot += zs[j]*a_alpha_ijs[i][j]
            Sis.append(tot)

        Sais = [val/a_alpha for val in Sis]
        Sbis = [bi/b for bi in self.bs]

        Snc = Sis[-1]
        const_A = 2.0*self.P/(R2*T2)
        dA_dzis = [const_A*(Si - Snc) for Si in Sis[:-1]]

        const_B = 2.0*self.P/(R*self.T)
        bnc = self.bs[-1]
        dB_dzis = [const_B*(self.bs[i] - bnc) for i in range(N)] # Probably wrong, missing

        dZ_dzs = [dZ_dA*dA_dz_i + dZ_dB*dB_dzi for dA_dz_i, dB_dzi in zip(dA_dzis, dB_dzis)]

        t1 = (Z2 + 2.0*Z*B - B2)
        t2 = clog((Z + (root_two + 1.)*B)/(Z - (root_two - 1.)*B)).real
        t3 = t2*-A/(B*two_root_two)
        t4 = -t2/(two_root_two*B)

        a_nc = a_alpha_ijs[-1][-1] # no idea if this is right

        # Have some converns of what Snc really is
        dlnphis_dzs_all = []
        for i in range(self.N):
            Diks = [-A_B*(2.0*Sais[i] - Sbis[i])*(Z*dB_dzis[k] - B*dZ_dzs[k])/t1
                    for k in cmps_m1]

            Ciks = [t3*(2.0*(a_alpha_ijs[i][k] - a_nc)/a_alpha
                        - 4.0*Sais[i]*(Sais[k] - Snc)
                        + Sbis[i]*(Sbis[k] - Snc))
                    for k in cmps_m1]


            x5 = t4*(2.0*Sais[i] - Sbis[i])
            Biks = [x5*(dA_dzis[k] - A_B*dB_dzis[k])
                    for k in cmps_m1 ]

            Aiks = [Sbis[i]*(dZ_dzs[k] - (Sbis[k] - Snc)*(Z - 1.0))
                    - (dZ_dzs[k] - dB_dzis[k])/ZmB
                    for k in cmps_m1 ]

            dlnphis_dzs = [Aik + Bik + Cik + Dik for Aik, Bik, Cik, Dik in zip(Aiks, Biks, Ciks, Diks)]
            dlnphis_dzs_all.append(dlnphis_dzs)
        return dlnphis_dzs_all


    def d_lnphi_dzs_basic_num(self, Z, zs):
        all_diffs = []

        try:
            if self.G_dep_l < self.G_dep_g:
                lnphis_ref = self.lnphis_l
            else:
                lnphis_ref = self.lnphis_g
        except:
           lnphis_ref = self.lnphis_l if hasattr(self, 'G_dep_l') else self.lnphis_g




        for i in range(len(zs)):
            zs2 = list(zs)
            dz = 1e-7#zs2[i]*3e-
            zs2[i] = zs2[i]+dz
#            sum_one = sum(zs2)
#            zs2 = normalize(zs2)
            eos2 = self.to_TP_zs(T=self.T, P=self.P, zs=zs2)



            diffs = []
            for j in range(len(zs)):
                try:
                    dlnphis = (eos2.lnphis_g[j] - lnphis_ref[j])/dz
                except:
                    dlnphis = (eos2.lnphis_l[j] - lnphis_ref[j])/dz
                diffs.append(dlnphis)
            all_diffs.append(diffs)
        import numpy as np
        return np.array(all_diffs).T.tolist()


    def d_lnphi_dzs_numdifftools(self, Z, zs):
        import numpy as np
        import numdifftools as nd

        def lnphis_from_zs(zs2):
            if isinstance(zs2, np.ndarray):
                zs2 = zs2.tolist()
                zs2 = normalize(zs2)
            # Last row suggests the normalization breaks everything!
#            zs2 = normalize(zs2)

#            if Z == self.Z_l


            try:
                return np.array(self.to_TP_zs(T=self.T, P=self.P, zs=zs2).lnphis_l)
            except:
                return np.array(self.to_TP_zs(T=self.T, P=self.P, zs=zs2).lnphis_g)

        Jfun_partial = nd.Jacobian(lnphis_from_zs, step=1e-4, order=2, method='central')
        return Jfun_partial(zs)

    def dlnphis_dzs(self, Z):
        r'''Calculate and return the mole fraction derivaitves of
        log fugacity coefficients for each species in a mixture. This formula
        is specific to the Peng-Robinson equation of state.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial z_i}\right)_{P,
            z_{j \ne i}}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        dlnphis_dzs : list[list[float]]
            Mole fraction derivatives of log fugacity coefficient for each
            species (such that the mole fractions do not sum to 1), [-]

        Notes
        -----
        This formula is from [1]_ but is validated to match the generic
        implementation.

        Examples
        --------

        >>> kijs = [[0, 0.00076, 0.00171], [0.00076, 0, 0.00061], [0.00171, 0.00061, 0]]
        >>> eos = PRMIX(Tcs=[469.7, 507.4, 540.3], zs=[0.8168, 0.1501, 0.0331], omegas=[0.249, 0.305, 0.349], Pcs=[3.369E6, 3.012E6, 2.736E6], T=322.29, P=101325, kijs=kijs)
        >>> eos.dlnphis_dzs(eos.Z_l)
        [[0.009938069276, 0.0151503498382, 0.018297235797], [-0.038517738793, -0.05958926042, -0.068438990795], [-0.07057106923, -0.10363920720, -0.14116283024]]

        References
        ----------
        .. [1] Chang, Yih-Bor. "Development and Application of an Equation of
           State Compositional Simulator" 1990.
           https://repositories.lib.utexas.edu/handle/2152/80585.
        '''
        T, P, zs = self.T, self.P, self.zs
        T2 = T*T
        T_inv = 1.0/T
        RT_inv = R_inv*T_inv
        bs, b = self.bs, self.b
        a_alpha = self.a_alpha
        a_alpha_ijs = self.a_alpha_ijs
        a_alphas = self.a_alphas
        a_alpha_j_rows = self.a_alpha_j_rows
        N = len(zs)

        b2 = b*b
        b_inv = 1.0/b
        b2_inv = b_inv*b_inv
        a_alpha2 = a_alpha*a_alpha

        A = a_alpha*P*RT_inv*RT_inv
        B = b*P*RT_inv
        B_inv = 1.0/B
        C = 1.0/(Z - B)

        Zm1 = Z - 1.0
        G = (Z + (1.0 + root_two)*B)/(Z + (1.0 - root_two)*B)

        t4 = 2.0/a_alpha
        t5 = -A/(two_root_two*B)
        Eis = [t5*(t4*a_alpha_j_rows[i] - bs[i]*b_inv) for i in range(N)]
#        ln_phis = []
#        for i in range(N):
#            ln_phis.append(log(C) + Dis[i] + Eis[i]*log(G))
#        return ln_phis

#        Bis = [bi*P/(R*T) for bi in bs]
        # maybe with a 2 constant?
        t6 = P*RT_inv
        dB_dxks = [t6*bk for bk in bs]


        # THIS IS WRONG - the sum changes w.r.t (or does it?)
        # Believed right now?
        const = (P+P)*RT_inv*RT_inv
        dA_dxks = [const*term_i for term_i in a_alpha_j_rows]

        dF_dZ_inv = 1.0/(3.0*Z*Z - 2.0*Z*(1.0 - B) + (A - 3.0*B*B - 2.0*B))

        t15 = (A - 2.0*B - 3.0*B*B + 2.0*(3.0*B + 1.0)*Z - Z*Z)
        BmZ = (B - Z)
        dZ_dxs = [(BmZ*dA_dxks[i] + t15*dB_dxks[i])*dF_dZ_inv for i in range(N)]

        # function only of k
        ZmB = Z - B
        t20 = -1.0/(ZmB*ZmB)
        dC_dxs = [t20*(dZ_dxs[k] - dB_dxks[k]) for k in range(N)]

        dD_dxs = []
#        dD_dxs = [[0.0]*N for _ in cmps]
        t55s = [b*dZ_dxs[k] - bs[k]*Zm1 for k in range(N)]
        for i in range(N):
#            dD_dxs_i = dD_dxs[i]
            b_term_ratio = bs[i]*b2_inv
            dD_dxs.append([b_term_ratio*t55s[k] for k in range(N)])
#            for k in range(N):
#                dD_dxs_i[k] = b_term_ratio*t55s[k]
#        dD_dxs = []
#        for i in range(N):
#            term = bs[i]/(b*b)*(b*dZ_dxs[i] - b*(Z - 1.0))
#            dD_dxs.append(term)

        # ? Believe this is the only one with multi indexes?
        t1 = 1.0/(two_root_two*a_alpha*b*B)
        t2 = t1*A/(a_alpha*b)
        t50s = [B*dA_dxks[k] - A*dB_dxks[k] for k in range(N)]

        # problem is in here, tested numerically
        b_two = b + b
        t32 = 2.0*a_alpha*b2
        t33 = 4.0*b2
        t34 = t1*B_inv*a_alpha
        t35 = -t1*B_inv*b_two

        # Symmetric matrix!
        dE_dxs = [[0.0]*N for _ in range(N)] # TODO - makes little sense. Too many i indexes.
        for i in range(N):
            zm_aim_tot = a_alpha_j_rows[i]
            t30 = t34*bs[i] + t35*zm_aim_tot
            t31 = t33*zm_aim_tot

            dE_dxs_i = []
            a_alpha_ijs_i = a_alpha_ijs[i]
            for k in range(0, i+1):
                # Sign was wrong in article - should be a plus
                second = t2*(t31*a_alpha_j_rows[k] - t32*a_alpha_ijs_i[k] - bs[i]*bs[k]*a_alpha2)
                dE_dxs[i][k] = dE_dxs[k][i] = t30*t50s[k] + second

#                dE_dxs_i.append(t1*(first + second))
#            dE_dxs.append(dE_dxs_i)

        t59 = (Z + (1.0 - root_two)*B)
        t60 = two_root_two/(t59*t59)
        dG_dxs = [t60*(Z*dB_dxks[k] - B*dZ_dxs[k]) for k in range(N)]


        G_inv = 1.0/G
        logG = log(G)
        C_inv = 1.0/C
        dlnphis_dxs = []
#        dlnphis_dxs = [[0.0]*N for _ in range(N)]
        t61s = [C_inv*dC_dxi for dC_dxi in dC_dxs]
        for i in range(N):
            dD_dxs_i = dD_dxs[i]
            dE_dxs_i = dE_dxs[i]
            E_G = Eis[i]*G_inv
#            dlnphis_dxs_i = dlnphis_dxs[i]
            dlnphis_dxs_i = [t61s[k] + dD_dxs_i[k] + logG*dE_dxs_i[k] + E_G*dG_dxs[k]
                             for k in range(N)]
            dlnphis_dxs.append(dlnphis_dxs_i)

#        return dlnphis_dxs
        return dlnphis_dxs#, dZ_dxs, dA_dxks, dB_dxks, dC_dxs, dD_dxs, dE_dxs, dG_dxs

    @property
    def ddelta_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \delta}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = 2 b_i

        Returns
        -------
        ddelta_dzs : list[float]
            Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        return PR_ddelta_dzs(self.bs, N, out=[0.0]*N if self.scalar else zeros(N))

    @property
    def ddelta_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \delta}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = 2 (b_i - b)

        Returns
        -------
        ddelta_dns : list[float]
            Mole number derivative of `delta` of each component, [m^3/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        return PR_ddelta_dns(self.bs, self.b, N, out=[0.0]*N if self.scalar else zeros(N))

    @property
    def d2delta_dzizjs(self):
        r'''Helper method for calculating the second composition derivatives (hessian) of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \delta}{\partial x_i\partial x_j}\right)_{T, P, x_{k\ne i,j}}
            = 0

        Returns
        -------
        d2delta_dzizjs : list[float]
            Second Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            return [[0.0]*N for i in range(N)]
        return zeros((N, N))

    @property
    def d2delta_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \delta}{\partial n_i \partial n_j}\right)_{T, P, n_{k\ne i,j}}
            = 4b - 2b_i - 2b_j

        Returns
        -------
        d2delta_dninjs : list[list[float]]
            Second mole number derivative of `delta` of each component, [m^3/mol^3]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[0.0]*N for _ in range(N)] if self.scalar else zeros((N, N))
        return PR_d2delta_dninjs(self.b, self.bs, N, out)

    @property
    def d3delta_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \delta}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = 4(-3b + b_i + b_j + b_k)

        Returns
        -------
        d3delta_dninjnks : list[list[list[float]]]
            Third mole number derivative of `delta` of each component,
            [m^3/mol^4]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[[0.0]*N for _ in range(N) ] for _ in range(N)] if self.scalar else zeros((N, N, N))
        return PR_d3delta_dninjnks(self.b, self.bs, N, out)

    @property
    def depsilon_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \epsilon}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = -2 b_i\cdot b

        Returns
        -------
        depsilon_dzs : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        return PR_depsilon_dzs(self.b, self.bs, N, out=[0.0]*N if self.scalar else zeros(N))

    @property
    def depsilon_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \epsilon}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = 2b(b - b_i)

        Returns
        -------
        depsilon_dns : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^3]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        return PR_depsilon_dns(self.b, self.bs, N, out=[0.0]*N if self.scalar else zeros(N))

    @property
    def d2epsilon_dzizjs(self):
        r'''Helper method for calculating the second composition derivatives (hessian)
        of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial x_i \partial x_j}\right)_{T, P, x_{k\ne i,j}}
            =  2 b_i b_j

        Returns
        -------
        d2epsilon_dzizjs : list[list[float]]
            Second composition derivative of `epsilon` of each component, [m^6/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[0.0]*N for _ in range(N)] if self.scalar else zeros((N, N))
        return PR_d2epsilon_dzizjs(self.b, self.bs, N, out)

    @property
    def d2epsilon_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives
        (hessian) of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial n_i n_j}\right)_{T, P,
            n_{k\ne i,j}}  = -2b(2b - b_i - b_j) - 2(b - b_i)(b - b_j)

        Returns
        -------
        d2epsilon_dninjs : list[list[float]]
            Second mole number derivative of `epsilon` of each component,
            [m^6/mol^4]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[0.0]*N for _ in range(N)] if self.scalar else zeros((N, N))
        return PR_d2epsilon_dninjs(self.b, self.bs, N, out)
    


    @property
    def d3epsilon_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \epsilon}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = 24b^2 - 12b(b_i + b_j + b_k)
                 + 4(b_i b_j + b_i b_k + b_j b_k)

        Returns
        -------
        d3epsilon_dninjnks : list[list[list[float]]]
            Third mole number derivative of `epsilon` of each component,
            [m^6/mol^5]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[[0.0]*N for _ in range(N) ] for _ in range(N)] if self.scalar else zeros((N, N, N))
        return PR_d3epsilon_dninjnks(self.b, self.bs, N, out)

    def solve_T(self, P, V, quick=True, solution=None):
        if self.N == 1 and type(self) is PRMIX:
            self.Tc = self.Tcs[0]
            self.Pc = self.Pcs[0]
            self.kappa = self.kappas[0]
            self.a = self.ais[0]
            T = super(type(self).__mro__[-4], self).solve_T(P=P, V=V, solution=solution)
            del self.Tc
            del self.Pc
            del self.kappa
            del self.a
            return T
        else:
            return super(type(self).__mro__[-3], self).solve_T(P=P, V=V, solution=solution)

class PRMIXTranslated(PRMIX):
    r'''Class for solving the Peng-Robinson [1]_ [2]_ translated cubic equation
    of state for a mixture of any number of compounds. Solves the EOS
    on initialization and calculates fugacities for all components in all
    phases.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v + c - b} - \frac{a\alpha(T)}{(v+c)(v + c + b)+b(v
        + c - b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}

    .. math::
	    b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}

    .. math::
        \alpha(T)_i=[1+\kappa_i(1-\sqrt{T_{r,i}})]^2

    .. math::
        \kappa_i=0.37464+1.54226\omega_i-0.26992\omega^2_i

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    cs : list[float], optional
        Volume translation parameters; always zero in the original
        implementation, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = PRMIXTranslated(T=115, P=1E6, cs=[-4.4e-6, -4.35e-6], Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.2, 0.8], kijs=[[0,0.03],[0.03,0]])
    >>> eos.V_l, eos.V_g
    (3.9079056337e-05, 0.00060231393016)
    >>> eos.fugacities_l, eos.fugacities_g
    ([442838.8615, 108854.48589], [184396.972, 565531.7709])

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    References
    ----------
    .. [1] Peng, Ding-Yu, and Donald B. Robinson. "A New Two-Constant Equation
       of State." Industrial & Engineering Chemistry Fundamentals 15, no. 1
       (February 1, 1976): 59-64. doi:10.1021/i160057a011.
    .. [2] Robinson, Donald B., Ding-Yu Peng, and Samuel Y-K Chung. "The
       Development of the Peng - Robinson Equation and Its Application to Phase
       Equilibrium in a System Containing Methanol." Fluid Phase Equilibria 24,
       no. 1 (January 1, 1985): 25-41. doi:10.1016/0378-3812(85)87035-7.
    '''
    translated = True
    eos_pure = PRTranslated
    mix_kwargs_to_pure = {'cs': 'c'}
    kwargs_linear = ('cs',)
    fugacity_coefficients = GCEOSMIX.fugacity_coefficients
    dlnphis_dT = GCEOSMIX.dlnphis_dT
    dlnphis_dP = GCEOSMIX.dlnphis_dP
    d_lnphi_dzs = GCEOSMIX.dlnphis_dzs
    P_max_at_V = GCEOSMIX.P_max_at_V
    model_id = 10202

    # All the b derivatives happen to work out to be the same, and are checked numerically
    solve_T = GCEOS.solve_T
    kwargs_keys = ('kijs', 'cs')

    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, cs=None,
                 T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):

        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in range(N)]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.T = T
        self.P = P
        self.V = V

        c2R, c1R2_c2R = self.c2R, self.c1R2_c2R
        if scalar:
            b0s = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*b0s[i] for i in cmps]
        else:
            b0s = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*b0s

        if cs is None:
            if scalar:
                cs = [0.0]*N
            else:
                cs = zeros(N)
        if scalar:
            self.kappas = [omega*(-0.26992*omega + 1.54226) + 0.37464 for omega in omegas]
            b0, c = 0.0, 0.0
            for i in range(N):
                b0 += b0s[i]*zs[i]
                c += cs[i]*zs[i]
            bs = [b0s[i] - cs[i] for i in range(N)]
        else:
            self.kappas = omegas*(-0.26992*omegas + 1.54226) + 0.37464
            b0 = float((b0s*zs).sum())
            c = float((cs*zs).sum())
            bs = b0s - cs
        
        self.kwargs = {'kijs': kijs, 'cs': cs}
        self.cs = cs
        
        self.b0s = b0s
        self.bs = bs
        self.c = c
        self.b = b = b0 - c
        self.delta = 2.0*(c + b0)
        self.epsilon = -b0*b0 + c*(c + b0 + b0)
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        self.cs = cs = other.cs
        self.kappas = other.kappas
        zs = self.zs
        self.b0s = b0s = other.b0s
        if self.scalar:
            b0, c = 0.0, 0.0
            for i in range(self.N):
                b0 += b0s[i]*zs[i]
                c += cs[i]*zs[i]
        else:
            b0 = float((b0s*zs).sum())
            c = float((cs*zs).sum())
        self.c = c
        self.b = b0 - c
        self.delta = 2.0*(c + b0)
        self.epsilon = -b0*b0 + c*(c + b0 + b0) # Very important to be calculated exactly the same way as the other implementation

    @property
    def ddelta_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `delta`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial \delta}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = 2 (c_i + b^0_i)

        Returns
        -------
        ddelta_dzs : list[float]
            Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        return PR_translated_ddelta_dzs(self.b0s, self.cs, N, 
                                        [0.0]*N if self.scalar else zeros(N))

    # Zero in both cases
    d2delta_dzizjs = PRMIX.d2delta_dzizjs
    d3delta_dzizjzks = PRMIX.d3delta_dzizjzks

    @property
    def ddelta_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `delta`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial \delta}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = 2 (c_i + b^0_i) - \delta

        Returns
        -------
        ddelta_dns : list[float]
            Mole number derivative of `delta` of each component, [m^3/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        return PR_translated_ddelta_dns(self.b0s, self.cs, self.delta, N, [0.0]*N if self.scalar else zeros(N))


    @property
    def d2delta_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `delta`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial^2 \delta}{\partial n_i \partial n_j}\right)_{T, P, n_{k\ne i,j}}
            = 2\left(\delta - b^0_i - b^0_j - c_i - c_j \right)

        Returns
        -------
        d2delta_dninjs : list[list[float]]
            Second mole number derivative of `delta` of each component, [m^3/mol^3]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[0.0]*N for _ in range(N)] if self.scalar else zeros((N, N))
        return PR_translated_d2delta_dninjs(self.b0s, self.cs, self.b, self.c, self.delta, N, out)

    @property
    def d3delta_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `delta`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial^3 \delta}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = 4\left(b^0_i + b^0_j + b^0_k + c_i + c_j
                + c_k \right) - 6 \delta

        Returns
        -------
        d3delta_dninjnks : list[list[list[float]]]
            Third mole number derivative of `delta` of each component,
            [m^3/mol^4]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)] if self.scalar else zeros((N, N, N))
        return PR_translated_d3delta_dninjnks(self.b0s, self.cs, self.delta, N, out)

    @property
    def depsilon_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `epsilon`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial \epsilon}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = c_i(2b^0_i + c) + c(2b^0_i + c_i) - 2b^0 b^0_i

        Returns
        -------
        depsilon_dzs : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        return PR_translated_depsilon_dzs(self.epsilon, self.c, self.b, self.b0s, self.cs, N,
                                          [0.0]*N if self.scalar else zeros(N))

    @property
    def depsilon_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `epsilon`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial \epsilon}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = 2b^0(b^0 - b^0_i) - c(2b^0 - 2b_i^0 + c - c_i) - (c - c_i)(2b^0 + c)


        Returns
        -------
        depsilon_dns : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^3]

        Notes
        -----
        This derivative is checked numerically.
        '''
        epsilon, c, b = self.epsilon, self.c, self.b
        N, b0s, cs = self.N, self.b0s, self.cs
        return PR_translated_depsilon_dns(epsilon, c, b, b0s, cs, N, out=([0.0]*N if self.scalar else zeros(N)))

    @property
    def d2epsilon_dzizjs(self):
        r'''Helper method for calculating the second composition derivatives (hessian)
        of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial x_i \partial x_j}\right)_{T, P, x_{k\ne i,j}}
            =  -2 b^0_i b^0_j + 2b^0_i c_j + 2b^0_j c_i + 2c_i c_j

        Returns
        -------
        d2epsilon_dzizjs : list[list[float]]
            Second composition derivative of `epsilon` of each component, [m^6/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[0.0]*N for _ in range(N)] if self.scalar else zeros((N, N))
        return PR_translated_d2epsilon_dzizjs(self.b0s, self.cs, N=N, out=out)

    d3epsilon_dzizjzks = GCEOSMIX.d3epsilon_dzizjzks # Zeros


    @property
    def d2epsilon_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial n_i n_j}\right)_{T, P, n_{k\ne i,j}}
            = -2b^0(2b^0 - b_i^0 - b_j^0) + c(4b^0 - 2b^0_i - 2b^0_j + 2c - c_i - c_j)
            -2(b^0 - b_i^0)(b^0 - b^0_j)
            + (c - c_i)(2b^0 - 2b^0_j - c_j + c)
            + (c - c_j)(2b^0 - 2b^0_i - c_i + c)
            + (2b^0 + c)(2c-c_i - c_j)

        Returns
        -------
        d2epsilon_dninjs : list[list[float]]
            Second mole number derivative of `epsilon` of each component, [m^6/mol^4]

        Notes
        -----
        This derivative is checked numerically.
        '''
        # Not trusted yet - numerical check does not have enough digits
        N = self.N
        out = [[0.0]*N for _ in range(N)] if self.scalar else zeros((N, N))
        return PR_translated_d2epsilon_dninjs(self.b0s, self.cs, self.b, self.c, N, out=out)

    @property
    def d3epsilon_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \epsilon}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = 4b^0(3b^0 - b_i^0 - b_j^0 - b_k^0)
                -2c(6b^0 - 2(b_i^0 + b_j^0 + b_k^0) + 3c - (c_i + c_j + c_k))
                +2(b^0-b_i^0)(2b^0 - b_j^0 - b_k^0) + 2(b^0 - b^0_j)(2b^0 - b_i^0 - b_k^0)
                +2(b^0-b^0_k)(2b^0 - b^0_i-b^0_j)
            -(c-c_i)(4b^0 - 2b^0_j - 2b^0_k + 2c - c_j - c_k)
            -(c-c_j)(4b^0 - 2b^0_i - 2b^0_k + 2c - c_i - c_k)
            -(c-c_k)(4b^0 - 2b^0_j - 2b^0_i + 2c - c_j - c_i)
            -2(c + 2b^0)(3c - c_i - c_j - c_k)
            -(2c - c_i - c_j)(2b^0 + c - 2b^0_k - c_k)
            -(2c - c_i - c_k)(2b^0 + c - 2b^0_j - c_j)
            -(2c - c_j - c_k)(2b^0 + c - 2b^0_i - c_i)

        Returns
        -------
        d3epsilon_dninjnks : list[list[list[float]]]
            Third mole number derivative of `epsilon` of each component,
            [m^6/mol^5]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)] if self.scalar else zeros((N, N, N))
        return PR_translated_d3epsilon_dninjnks(self.b0s, self.cs, self.b, self.c, self.epsilon, N, out)



class PRMIXTranslatedPPJP(PRMIXTranslated):
    r'''Class for solving the Pina-Martinez, Privat, Jaubert,
    and Peng revision of the Peng-Robinson equation of state.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v + c - b} - \frac{a\alpha(T)}{(v+c)(v + c + b)+b(v
        + c - b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}

    .. math::
	    b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}

    .. math::
        \alpha(T)_i=[1+\kappa_i(1-\sqrt{T_{r,i}})]^2

    .. math::
        \kappa_i=0.3919 + 1.4996 \omega - 0.2721\omega^2 + 0.1063\omega^3

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    cs : list[float], optional
        Volume translation parameters, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = PRMIXTranslatedPPJP(T=115, P=1E6, cs=[-4.4e-6, -4.35e-6], Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.2, 0.8], kijs=[[0,0.03],[0.03,0]])
    >>> eos.V_l, eos.V_g
    (3.8989032701e-05, 0.00059686183724)
    >>> eos.fugacities_l, eos.fugacities_g
    ([444791.13707, 104520.280997], [184782.600238, 563352.147])

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    References
    ----------
    .. [1] Pina-Martinez, Andrés, Romain Privat, Jean-Noël Jaubert, and
       Ding-Yu Peng. "Updated Versions of the Generalized Soave α-Function
       Suitable for the Redlich-Kwong and Peng-Robinson Equations of State."
       Fluid Phase Equilibria, December 7, 2018.
       https://doi.org/10.1016/j.fluid.2018.12.007.
    '''
    eos_pure = PRTranslatedPPJP
    mix_kwargs_to_pure = {'cs': 'c'}
    kwargs_linear = ('cs',)
    kwargs_keys = ('kijs', 'cs')
    model_id = 10207
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, cs=None,
                 T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):

        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in cmps]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.T = T
        self.P = P
        self.V = V
        c2R, c1R2_c2R = self.c2R, self.c1R2_c2R

        if scalar:
            b0s = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*b0s[i] for i in cmps]
        else:
            b0s = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*b0s

        if cs is None:
            if scalar:
                cs = [0.0]*N
            else:
                cs = zeros(N)
        
        if scalar:
            self.kappas = [omega*(omega*(0.1063*omega - 0.2721) + 1.4996) + 0.3919 for omega in omegas]
            b0, c = 0.0, 0.0
            for i in range(N):
                b0 += b0s[i]*zs[i]
                c += cs[i]*zs[i]
            bs = [b0s[i] - cs[i] for i in range(N)]
        else:   
            self.kappas = omegas*(omegas*(0.1063*omegas - 0.2721) + 1.4996) + 0.3919
            b0 = float((b0s*zs).sum())
            c = float((cs*zs).sum())
            bs = b0s - cs
            
        self.kwargs = {'kijs': kijs, 'cs': cs}
        self.cs = cs


        self.b0s = b0s
        self.bs = bs
        self.c = c
        self.b = b = b0 - c
        self.delta = 2.0*(c + b0)
        self.epsilon = -b0*b0 + c*(c + b0 + b0)
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

class PRMIXTranslatedConsistent(Twu91_a_alpha, PRMIXTranslated):
    r'''Class for solving the volume translated Le Guennec, Privat, and Jaubert
    revision of the Peng-Robinson equation of state according to [1]_.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v + c - b} - \frac{a\alpha(T)}{(v+c)(v + c + b)+b(v
        + c - b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}

    .. math::
	    b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}

    .. math::
        \alpha_i = \left(\frac{T}{T_{c}}\right)^{c_{3} \left(c_{2}
        - 1\right)} e^{c_{1} \left(- \left(\frac{T}{T_{c}}
        \right)^{c_{2} c_{3}} + 1\right)}

    If `c` is not provided, they are estimated as:

    .. math::
        c =\frac{R T_c}{P_c}(0.0198\omega - 0.0065)

    If `alpha_coeffs` is not provided, the parameters `L` and `M` are estimated
    from the acentric factor as follows:

    .. math::
        L = 0.1290\omega^2 + 0.6039\omega + 0.0877

    .. math::
        M = 0.1760\omega^2 - 0.2600\omega + 0.8884

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    cs : list[float], optional
        Volume translation parameters, [m^3/mol]
    alpha_coeffs : list[tuple(float[3])], optional
        Coefficients L, M, N (also called C1, C2, C3) of TWU 1991 form, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = PRMIXTranslatedConsistent(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.2, 0.8], kijs=[[0,0.03],[0.03,0]])
    >>> eos.V_l, eos.V_g
    (3.675235812e-05, 0.00059709319879)
    >>> eos.fugacities_l, eos.fugacities_g
    ([443454.9336, 106184.004057], [184122.74082, 563037.785])

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    References
    ----------
    .. [1] Le Guennec, Yohann, Romain Privat, and Jean-Noël Jaubert.
       "Development of the Translated-Consistent Tc-PR and Tc-RK Cubic
       Equations of State for a Safe and Accurate Prediction of Volumetric,
       Energetic and Saturation Properties of Pure Compounds in the Sub- and
       Super-Critical Domains." Fluid Phase Equilibria 429 (December 15, 2016):
       301-12. https://doi.org/10.1016/j.fluid.2016.09.003.
    '''
    eos_pure = PRTranslatedConsistent
    kwargs_linear = ('cs', 'alpha_coeffs')
    mix_kwargs_to_pure = {'cs': 'c', 'alpha_coeffs': 'alpha_coeffs'}
    kwargs_keys = ('kijs', 'alpha_coeffs', 'cs')
    model_id = 10203

    # There is an updated set of correlations - which means a revision flag is needed
    # Analysis of the Combinations of Property Data That Are Suitable for a Safe Estimation of Consistent Twu α-Function Parameters: Updated Parameter Values for the Translated-Consistent tc-PR and tc-RK Cubic Equations of State
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, cs=None,
                 alpha_coeffs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in cmps]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.T = T
        self.P = P
        self.V = V

        c1R2_c2R, c2R = self.c1R2_c2R, self.c2R

        if scalar:
            b0s = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*b0s[i] for i in cmps]
        else:
            b0s = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*b0s

        if cs is None:
            if scalar:
                cs = [R*Tcs[i]/Pcs[i]*(0.0198*min(max(omegas[i], -0.01), 1.48) - 0.0065)
                    for i in range(N)]
            else:
                cs = R*Tcs/Pcs*(0.0198*npmin(npmax(omegas, -0.01), 1.48) - 0.0065)
        if alpha_coeffs is None:
            alpha_coeffs = []
            for i in range(N):
                o = min(max(omegas[i], -0.01), 1.48)
                L = o*(0.1290*o + 0.6039) + 0.0877
                M = o*(0.1760*o - 0.2600) + 0.8884
                alpha_coeffs.append((L, M, 2.0))

        self.kwargs = {'kijs': kijs, 'alpha_coeffs': alpha_coeffs, 'cs': cs}
        self.alpha_coeffs = alpha_coeffs
        self.cs = cs
        
        if scalar:
            b0, c = 0.0, 0.0
            for i in range(N):
                b0 += b0s[i]*zs[i]
                c += cs[i]*zs[i]
            bs = [b0s[i] - cs[i] for i in range(N)]
        else:
            b0 = float((b0s*zs).sum())
            c = float((cs*zs).sum())
            bs = b0s - cs

        self.b0s = b0s
        self.bs = bs
        self.c = c
        self.b = b = b0 - c
        self.delta = 2.0*(c + b0)
        self.epsilon = -b0*b0 + c*(c + b0 + b0)
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        self.cs = cs = other.cs
        self.alpha_coeffs = other.alpha_coeffs
        zs = self.zs
        self.b0s = b0s = other.b0s
        
        if self.scalar:
            b0, c = 0.0, 0.0
            for i in range(self.N):
                b0 += b0s[i]*zs[i]
                c += cs[i]*zs[i]
        else:
            b0 = float((zs*b0s).sum())
            c = float((zs*cs).sum())

        self.c = c
        self.b = b0 - c
        self.delta = 2.0*(c + b0)
        self.epsilon = -b0*b0 + c*(c + b0 + b0) # Very important to be calculated exactly the same way as the other implementation


class SRKMIX(EpsilonZeroMixingRules, GCEOSMIX, SRK):
    r'''Class for solving the Soave-Redlich-Kwong cubic equation of state for a
    mixture of any number of compounds. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.

    The implemented method here is :obj:`fugacity_coefficients <SRKMIX.fugacity_coefficients>`, which implements
    the formula for fugacity coefficients in a mixture as given in [1]_.
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}

    .. math::
        b_i =\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}

    .. math::
        \alpha(T)_i = \left[1 + m_i\left(1 - \sqrt{\frac{T}{T_{c,i}}}\right)\right]^2

    .. math::
        m_i = 0.480 + 1.574\omega_i - 0.176\omega_i^2

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> SRK_mix = SRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> SRK_mix.V_l, SRK_mix.V_g
    (4.1047569614e-05, 0.0007110158049)

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    References
    ----------
    .. [1] Soave, Giorgio. "Equilibrium Constants from a Modified Redlich-Kwong
       Equation of State." Chemical Engineering Science 27, no. 6 (June 1972):
       1197-1203. doi:10.1016/0009-2509(72)80096-4.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th
       edition. New York: McGraw-Hill Professional, 2000.
    .. [3] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
       Butterworth-Heinemann, 1985.
    '''
    eos_pure = SRK
    nonstate_constants_specific = ('ms',)
    kwargs_keys = ('kijs', )
    model_id = 10100

    ddelta_dzs = RKMIX.ddelta_dzs
    ddelta_dns = RKMIX.ddelta_dns
    d2delta_dzizjs = RKMIX.d2delta_dzizjs
    d2delta_dninjs = RKMIX.d2delta_dninjs
    d3delta_dninjnks = RKMIX.d3delta_dninjnks

    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in cmps]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V
        if self.scalar:
            self.ais = [self.c1*R2*Tc*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
            self.bs = [self.c2*R*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
            ms = [omega*(1.574 - 0.176*omega) + 0.480 for omega in omegas]
            b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        else:
            Tc_Pc_ratio = Tcs/Pcs
            self.ais = self.c1R2*Tcs*Tc_Pc_ratio
            self.bs = bs = self.c2R*Tc_Pc_ratio
            ms = omegas*(1.574 - 0.176*omegas) + 0.480
            b =  float((bs*zs).sum())
        
        self.b = b
        self.ms = ms
        self.delta = self.b

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        self.ms = other.ms
        if self.scalar:
            self.b = b = sum([bi*zi for bi, zi in zip(self.bs, self.zs)])
        else:
            self.b = b = float((self.bs*self.zs).sum())
        self.delta = b

    def a_alphas_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` for the SRK EOS.
        This vectorized implementation is added for extra speed.

        .. math::
            a\alpha = a \left(m \left(- \sqrt{\frac{T}{Tc}} + 1\right)
            + 1\right)^{2}

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        '''
        return SRK_a_alphas_vectorized(T, self.Tcs, self.ais, self.ms,
                                       a_alphas=[0.0]*self.N if self.scalar else zeros(self.N))

    def a_alpha_and_derivatives_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` and their first
        and second derivatives for the SRK EOS. This vectorized implementation
        is added for extra speed.

        .. math::
            a\alpha = a \left(m \left(- \sqrt{\frac{T}{Tc}} + 1\right)
            + 1\right)^{2}

        .. math::
            \frac{d a\alpha}{dT} = \frac{a m}{T} \sqrt{\frac{T}{Tc}} \left(m
            \left(\sqrt{\frac{T}{Tc}} - 1\right) - 1\right)

        .. math::
            \frac{d^2 a\alpha}{dT^2} = \frac{a m \sqrt{\frac{T}{Tc}}}{2 T^{2}}
            \left(m + 1\right)

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dTs : list[float]
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2s : list[float]
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K**2]
        '''
        N = self.N
        if self.scalar:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [0.0]*N, [0.0]*N, [0.0]*N
        else:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = zeros(N), zeros(N), zeros(N)
        return SRK_a_alpha_and_derivatives_vectorized(T, self.Tcs, self.ais, self.ms,
                                                      a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs, d2a_alpha_dT2s=d2a_alpha_dT2s)

    def fugacity_coefficients(self, Z):
        r'''Literature formula for calculating fugacity coefficients for each
        species in a mixture. Verified numerically. Applicable to most
        derivatives of the SRK equation of state as well.
        Called by :obj:`fugacities <GCEOSMIX.fugacities>` on initialization, or by a solver routine
        which is performing a flash calculation.

        .. math::
            \ln \hat \phi_i = \frac{B_i}{B}(Z-1) - \ln(Z-B) + \frac{A}{B}
            \left[\frac{B_i}{B} - \frac{2}{a \alpha}\sum_i y_i(a\alpha)_{ij}
            \right]\ln\left(1+\frac{B}{Z}\right)

        .. math::
            A=\frac{a\alpha P}{R^2T^2}

        .. math::
            B = \frac{bP}{RT}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        log_phis : float
            Log fugacity coefficient for each species, [-]
        '''
        N = self.N
        return SRK_lnphis(self.T, self.P, Z, self.b, self.a_alpha, self.bs, self.a_alpha_j_rows, N,
                          lnphis=[0.0]*N if self.scalar else zeros(N))


    def dlnphis_dT(self, phase):
        r'''Formula for calculating the temperature derivaitve of
        log fugacity coefficients for each species in a mixture for the
        SRK equation of state. Verified numerically.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial T}\right)_{P,
            nj \ne i}

        Parameters
        ----------
        phase : str
            One of 'l' or 'g', [-]

        Returns
        -------
        dlnphis_dT : float
            Temperature derivatives of log fugacity coefficient for each
            species, [1/K]

        Notes
        -----
        This expression was derived using SymPy and optimized with the `cse`
        technique.
        '''
        zs = self.zs
        if phase == 'g':
            Z = self.Z_g
            dZ_dT = self.dZ_dT_g
        else:
            Z = self.Z_l
            dZ_dT = self.dZ_dT_l

        da_alpha_dT_j_rows = self._da_alpha_dT_j_rows
        N = self.N
        P, bs, b = self.P, self.bs, self.b

        T_inv = 1.0/self.T
        A = self.a_alpha*P*R2_inv*T_inv*T_inv
        B = b*P*R_inv*T_inv

        x2 = T_inv*T_inv
        x4 = P*b*R_inv
        x6 = x4*T_inv

        x8 = self.a_alpha
        x9 = 1.0/x8
        x10 = self.da_alpha_dT
        x11 = 1.0/b
        x12 = 1.0/Z
        x13 = x12*x6 + 1.0
        x14 = log(x13)
        x19 = x11*x14*x2*R_inv*x8
        x20 = x10*x11*x14*R_inv*T_inv
        x21 = P*x12*x2*x8*(dZ_dT*x12 + T_inv)/(R2*x13)

        x50 = -x11*x14*R_inv*T_inv
        x51 = -2.0*x10
        x52 = (dZ_dT + x2*x4)/(x6 - Z)

        # Composition stuff
        d_lnphis_dTs = []

        a_alpha_j_rows = self.a_alpha_j_rows

        for i in range(N):
            x7 = a_alpha_j_rows[i]
            x15 = (x50*(x51*x7*x9 + 2.0*da_alpha_dT_j_rows[i]) + x52)

            x16 = bs[i]*x11
            x18 = -x16 + 2.0*x7*x9

            d_lhphi_dT = dZ_dT*x16 + x15 + x18*(x19 - x20 + x21)
            d_lnphis_dTs.append(d_lhphi_dT)
        return d_lnphis_dTs

    def dlnphis_dP(self, phase):
        r'''Generic formula for calculating the pressure derivaitve of
        log fugacity coefficients for each species in a mixture for the
        SRK EOS. Verified numerically.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial P}\right)_{T,
            nj \ne i}

        Parameters
        ----------
        phase : str
            One of 'l' or 'g', [-]

        Returns
        -------
        dlnphis_dP : float
            Pressure derivatives of log fugacity coefficient for each species,
            [1/Pa]

        Notes
        -----
        This expression was derived using SymPy and optimized with the `cse`
        technique.
        '''
        zs = self.zs
        if phase == 'l':
            Z, dZ_dP = self.Z_l, self.dZ_dP_l
        else:
            Z, dZ_dP = self.Z_g, self.dZ_dP_g
        a_alpha = self.a_alpha
        N = self.N
        bs, b = self.bs, self.b
        T_inv = 1.0/self.T
        a_alpha_j_rows = self._a_alpha_j_rows

        RT_inv = T_inv*R_inv
        x0 = Z
        x1 = dZ_dP
        x2 = 1.0/b
        x4 = b*RT_inv
        x5 = self.P*x4
        x6 = (dZ_dP - x4)/(x5 - Z)
        x7 = a_alpha
        x9 = 1./Z
        x10 = a_alpha*x9*(self.P*dZ_dP*x9 - 1.0)*RT_inv*RT_inv/((x5*x9 + 1.0))

        x50 = 2.0/a_alpha
        d_lnphi_dPs = []
        for i in range(N):
            x8 = x50*a_alpha_j_rows[i]
            x3 = bs[i]*x2
            d_lnphi_dP = dZ_dP*x3 + x10*(x8 - x3) + x6
            d_lnphi_dPs.append(d_lnphi_dP)
        return d_lnphi_dPs


class SRKMIXTranslated(SRKMIX):
    r'''Class for solving the volume translated Soave-Redlich-Kwong cubic equation of state for a
    mixture of any number of compounds. Subclasses :obj:`SRKMIX`. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V + c - b} - \frac{a\alpha(T)}{(V + c)(V + c + b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}

    .. math::
        b_i =\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}

    .. math::
        \alpha(T)_i = \left[1 + m_i\left(1 - \sqrt{\frac{T}{T_{c,i}}}\right)\right]^2

    .. math::
        m_i = 0.480 + 1.574\omega_i - 0.176\omega_i^2

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    cs : list[float], optional
        Volume translation parameters; always zero in the original
        implementation, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = SRKMIXTranslated(T=115, P=1E6, cs=[-4.4e-6, -4.35e-6], Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.2, 0.8], kijs=[[0,0.03],[0.03,0]])
    >>> eos.V_l, eos.V_g
    (4.35928920e-05, 0.00060927202)

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    '''
    fugacity_coefficients = GCEOSMIX.fugacity_coefficients
    dlnphis_dT = GCEOSMIX.dlnphis_dT
    dlnphis_dP = GCEOSMIX.dlnphis_dP
    d_lnphi_dzs = GCEOSMIX.dlnphis_dzs
    P_max_at_V = GCEOSMIX.P_max_at_V
    solve_T = GCEOS.solve_T
    model_id = 10101

    eos_pure = SRKTranslated
    translated = True
    mix_kwargs_to_pure = {'cs': 'c'}
    kwargs_linear = ('cs',)
    kwargs_keys = ('kijs', 'cs')

    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, cs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if cs is None:
            if scalar:
                cs = [0.0]*N
            else:
                cs = zeros(N)
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in cmps]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.kwargs = {'kijs': kijs, 'cs': cs}
        self.T = T
        self.P = P
        self.V = V
        
        
        c2R, c1R2_c2R = self.c2R, self.c1R2_c2R
        
        if scalar:
            b0s = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*b0s[i] for i in cmps]
            self.ms = [0.480 + omega*(1.574 - 0.176*omega) for omega in omegas]
        else:
            b0s = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*b0s
            self.ms = 0.480 + omegas*(1.574 - 0.176*omegas)
        self.cs = cs
        
        if scalar:
            b0, c = 0.0, 0.0
            for i in range(N):
                b0 += b0s[i]*zs[i]
                c += cs[i]*zs[i]
            bs = [b0s[i] - cs[i] for i in range(N)]
        else:
            b0 = float((b0s*zs).sum())
            c = float((cs*zs).sum())
            bs = b0s - cs

        self.b0s = b0s
        self.bs = bs
        self.c = c
        self.b = b = b0 - c
        self.delta = c + c + b0
        self.epsilon = c*(b0 + c)
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        self.cs = cs = other.cs
        self.ms = other.ms
        zs = self.zs
        self.b0s = b0s = other.b0s
        
        if self.scalar:
            b0, c = 0.0, 0.0
            for i in range(self.N):
                b0 += b0s[i]*zs[i]
                c += cs[i]*zs[i]
        else:
            b0 = float((b0s*zs).sum())
            c = float((cs*zs).sum())

        self.c = c
        self.b = b0 - c
        self.delta = c + c + b0
        self.epsilon = c*(b0 + c)

    @property
    def ddelta_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `delta`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial \delta}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = 2 (c_i + b^0_i)

        Returns
        -------
        ddelta_dzs : list[float]
            Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        b0s, cs = self.b0s, self.cs
        if self.scalar:
            return [(2.0*cs[i] + b0s[i]) for i in range(self.N)]
        return 2.0*cs + b0s

    # Zero in both cases
    d2delta_dzizjs = PRMIX.d2delta_dzizjs
    d3delta_dzizjzks = PRMIX.d3delta_dzizjzks

    @property
    def ddelta_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `delta`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial \delta}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = (2 c_i + b^0_i) - \delta

        Returns
        -------
        ddelta_dns : list[float]
            Mole number derivative of `delta` of each component, [m^3/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        return SRK_translated_ddelta_dns(self.b0s, self.cs, self.delta, N, out=[0.0]*N if self.scalar else zeros(N))

    @property
    def d2delta_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `delta`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial^2 \delta}{\partial n_i \partial n_j}\right)_{T, P, n_{k\ne i,j}}
            = \left(\2(b^0 - c_i - c_j) + 4c - b_i^0 - b_j^0\right)

        Returns
        -------
        d2delta_dninjs : list[list[float]]
            Second mole number derivative of `delta` of each component, [m^3/mol^3]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[0.0]*N for _ in range(N)] if self.scalar else zeros((N, N))
        return SRK_translated_d2delta_dninjs(self.b0s, self.cs, self.b, self.c, self.delta, N, out)

    @property
    def d3delta_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `delta`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial^3 \delta}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = -6b^0 + 2(b^0_i + b^0_j + b^0_k) + -12c
                +4(c_i + c_j + c_k)

        Returns
        -------
        d3delta_dninjnks : list[list[list[float]]]
            Third mole number derivative of `delta` of each component,
            [m^3/mol^4]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)] if self.scalar else zeros((N, N, N))
        return SRK_translated_d3delta_dninjnks(self.b0s, self.cs, self.b, self.c, self.delta, N, out)

    @property
    def depsilon_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `epsilon`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial \epsilon}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = c_i b^0 + 2c c_i + b_i c

        Returns
        -------
        depsilon_dzs : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [0.0]*N if self.scalar else zeros(N)
        return SRK_translated_depsilon_dzs(self.b0s, self.cs, self.b, self.c, N, out)

    @property
    def depsilon_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `epsilon`. Note this is independent of the phase. :math:`b^0` refers to
        the original `b` parameter not involving any translation.

        .. math::
            \left(\frac{\partial \epsilon}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = -b^0(c - c_i) - c(b^0 - b_i^0) - 2c(c - c_i)


        Returns
        -------
        depsilon_dns : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^3]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        return SRK_translated_depsilon_dns(self.b0s, self.cs, self.b, self.c, N, out=[0.0]*N if self.scalar else zeros(N))

    @property
    def d2epsilon_dzizjs(self):
        r'''Helper method for calculating the second composition derivatives (hessian)
        of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial x_i \partial x_j}\right)_{T, P, x_{k\ne i,j}}
            = b^0_i c_j + b^0_j c_i + 2c_i c_j

        Returns
        -------
        d2epsilon_dzizjs : list[list[float]]
            Second composition derivative of `epsilon` of each component, [m^6/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[0.0]*N for _ in range(N)] if self.scalar else zeros((N, N))
        return SRK_translated_d2epsilon_dzizjs(self.b0s, self.cs, self.b, self.c, N, out=out)

    d3epsilon_dzizjzks = GCEOSMIX.d3epsilon_dzizjzks # Zeros

    @property
    def d2epsilon_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial n_i n_j}\right)_{T, P, n_{k\ne i,j}}
            = b^0(2c - c_i - c_j) + c(2b^0 - b_i^0 - b_j^0) + 2c(2c - c_i - c_j)
            +(b^0 - b^0_i)(c - c_j) + (b^0 - b_j^0)(c - c_i) + 2(c - c_i)(c - c_j)

        Returns
        -------
        d2epsilon_dninjs : list[list[float]]
            Second mole number derivative of `epsilon` of each component, [m^6/mol^4]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[0.0]*N for _ in range(N)] if self.scalar else zeros((N, N))
        return SRK_translated_d2epsilon_dninjs(self.b0s, self.cs, self.b, self.c, N, out)
        
    @property
    def d3epsilon_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `epsilon`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \epsilon}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = -2b^0(3c - c_i - c_j - c_k)
                - 2c(3b^0 - b^0_i - b^0_j - b^0_k)
                - 4c(3c - c_i - c_j - c_k)
                -(b^0 - b^0_i)(2c - c_j - c_k)
                -(b^0 - b^0_j)(2c - c_i - c_k)
                -(b^0 - b^0_k)(2c - c_i - c_j)
                - (c - c_i)(2b^0 - b^0_j - b^0_k)
                - (c - c_j)(2b^0 - b^0_i - b^0_k)
                - (c - c_k)(2b^0 - b^0_i - b^0_j)
                -2(c - c_i)(2c - c_j - c_k)
                -2(c - c_j)(2c - c_i - c_k)
                -2(c - c_k)(2c - c_i - c_j)

        Returns
        -------
        d3epsilon_dninjnks : list[list[list[float]]]
            Third mole number derivative of `epsilon` of each component,
            [m^6/mol^5]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)] if self.scalar else zeros((N, N, N))
        return SRK_translated_d3epsilon_dninjnks(self.b0s, self.cs, self.b, self.c, self.epsilon, N, out)


class SRKMIXTranslatedConsistent(Twu91_a_alpha, SRKMIXTranslated):
    r'''Class for solving the volume translated Le Guennec, Privat, and Jaubert
    revision of the SRK equation of state according to [1]_.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V + c - b} - \frac{a\alpha(T)}{(V + c)(V + c + b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        \alpha_i = \left(\frac{T}{T_{c,i}}\right)^{c_{3} \left(c_{2}
        - 1\right)} e^{c_{1} \left(- \left(\frac{T}{T_{c,i}}
        \right)^{c_{2} c_{3}} + 1\right)}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}

    .. math::
        b_i =\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}


    If `cs` is not provided, they are estimated as:

    .. math::
        c =\frac{R T_c}{P_c}(0.0172\omega - 0.0096)

    If `alpha_coeffs` is not provided, the parameters `L` and `M` are estimated
    from each of the acentric factors as follows:

    .. math::
        L = 0.0947\omega^2 + 0.6871\omega + 0.1508

    .. math::
        M = 0.1615\omega^2 - 0.2349\omega + 0.8876

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    cs : list[float], optional
        Volume translation parameters, [m^3/mol]
    alpha_coeffs : list[list[float]]
        Coefficients for
        :obj:`thermo.eos_alpha_functions.Twu91_a_alpha`, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = SRKMIXTranslatedConsistent(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.2, 0.8], kijs=[[0,0.03],[0.03,0]])
    >>> eos.V_l, eos.V_g
    (3.591044498e-05, 0.0006020501621)

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    References
    ----------
    .. [1] Le Guennec, Yohann, Romain Privat, and Jean-Noël Jaubert.
       "Development of the Translated-Consistent Tc-PR and Tc-RK Cubic
       Equations of State for a Safe and Accurate Prediction of Volumetric,
       Energetic and Saturation Properties of Pure Compounds in the Sub- and
       Super-Critical Domains." Fluid Phase Equilibria 429 (December 15, 2016):
       301-12. https://doi.org/10.1016/j.fluid.2016.09.003.

    '''
    eos_pure = SRKTranslatedConsistent
    mix_kwargs_to_pure = {'cs': 'c', 'alpha_coeffs': 'alpha_coeffs'}
    kwargs_linear = ('cs', 'alpha_coeffs')
    kwargs_keys = ('kijs', 'alpha_coeffs', 'cs')
    model_id = 10102

    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, cs=None,
                 alpha_coeffs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in range(N)]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.T = T
        self.P = P
        self.V = V

        c2R, c1R2_c2R = self.c2R, self.c1R2_c2R
        if scalar:
            b0s = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*b0s[i] for i in cmps]
        else:
            b0s = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*b0s

        if cs is None:
            if scalar:
                cs = [R*Tcs[i]/Pcs[i]*(0.0172*min(max(omegas[i], -0.01), 1.46) + 0.0096)
                    for i in range(N)]
            else:
                cs = R*Tcs/Pcs*(0.0172*npmin(npmax(omegas, -0.01), 1.46) + 0.0096)
                
        if alpha_coeffs is None:
            alpha_coeffs = []
            for i in range(N):
                o = min(max(omegas[i], -0.01), 1.46)
                L = o*(0.0947*o + 0.6871) + 0.1508
                M = o*(0.1615*o - 0.2349) + 0.8876
                alpha_coeffs.append((L, M, 2.0))

        self.kwargs = {'kijs': kijs, 'alpha_coeffs': alpha_coeffs, 'cs': cs}
        self.alpha_coeffs = alpha_coeffs
        self.cs = cs
        
        if scalar:
            b0, c = 0.0, 0.0
            for i in range(N):
                b0 += b0s[i]*zs[i]
                c += cs[i]*zs[i]
            bs = [b0s[i] - cs[i] for i in range(N)]
        else:
            b0 = float((b0s*zs).sum())
            c = float((cs*zs).sum())
            bs = b0s - cs

        self.b0s = b0s
        self.bs = bs
        self.c = c
        self.b = b = b0 - c
        self.delta = c + c + b0
        self.epsilon = c*(b0 + c)
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        self.cs = cs = other.cs
        self.alpha_coeffs = other.alpha_coeffs
        zs = self.zs
        self.b0s = b0s = other.b0s

        if self.scalar:
            b0, c = 0.0, 0.0
            for i in range(self.N):
                b0 += b0s[i]*zs[i]
                c += cs[i]*zs[i]
        else:
            b0 = float((b0s*zs).sum())
            c = float((cs*zs).sum())

        self.c = c
        self.b = b0 - c
        self.delta = c + c + b0
        self.epsilon = c*(b0 + c)


class MSRKMIXTranslated(Soave_79_a_alpha, SRKMIXTranslatedConsistent):
    r'''Class for solving the volume translated Soave (1980) alpha function,
    revision of the Soave-Redlich-Kwong equation of state
    for a pure compound according to [1]_. Uses two fitting parameters `N` and
    `M` to more accurately fit the vapor pressure of pure species.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V + c - b} - \frac{a\alpha(T)}{(V + c)(V + c + b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        \alpha(T)_i = 1 + (1 - T_{r,i})(M + \frac{N}{T_{r,i}})

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}

    .. math::
        b_i =\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}

    This is an older correlation that offers lower accuracy on many properties
    which were sacrificed to obtain the vapor pressure accuracy. The alpha
    function of this EOS does not meet any of the consistency requriements for
    alpha functions.

    Coefficients can be found in [2]_, or estimated with the method in [3]_.
    The estimation method in [3]_ works as follows, using the acentric factor
    and true critical compressibility:

    .. math::
        M = 0.4745 + 2.7349(\omega Z_c) + 6.0984(\omega Z_c)^2

    .. math::
        N = 0.0674 + 2.1031(\omega Z_c) + 3.9512(\omega Z_c)^2

    An alternate estimation scheme is provided in [1]_, which provides
    analytical solutions to calculate the parameters `M` and `N` from two
    points on the vapor pressure curve, suggested as 10 mmHg and 1 atm.
    This is used as an estimation method here if the parameters are not
    provided, and the two vapor pressure points are obtained from the original
    SRK equation of state.

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    cs : list[float], optional
        Volume translation parameters, [m^3/mol]
    alpha_coeffs : list[list[float]]
        Coefficients for
        :obj:`thermo.eos_alpha_functions.Soave_79_a_alpha`, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = MSRKMIXTranslated(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.2, 0.8], kijs=[[0,0.03],[0.03,0]])
    >>> eos.V_l, eos.V_g
    (3.9222990198e-05, 0.00060438075638)

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    References
    ----------
    .. [1] Soave, G. "Rigorous and Simplified Procedures for Determining
       the Pure-Component Parameters in the Redlich—Kwong—Soave Equation of
       State." Chemical Engineering Science 35, no. 8 (January 1, 1980):
       1725-30. https://doi.org/10.1016/0009-2509(80)85007-X.
    .. [2] Sandarusi, Jamal A., Arthur J. Kidnay, and Victor F. Yesavage.
       "Compilation of Parameters for a Polar Fluid Soave-Redlich-Kwong
       Equation of State." Industrial & Engineering Chemistry Process Design
       and Development 25, no. 4 (October 1, 1986): 957-63.
       https://doi.org/10.1021/i200035a020.
    .. [3] Valderrama, Jose O., Héctor De la Puente, and Ahmed A. Ibrahim.
       "Generalization of a Polar-Fluid Soave-Redlich-Kwong Equation of State."
       Fluid Phase Equilibria 93 (February 11, 1994): 377-83.
       https://doi.org/10.1016/0378-3812(94)87021-7.
    '''
    kwargs_keys = ('kijs', 'alpha_coeffs', 'cs')
    eos_pure = MSRKTranslated
    model_id = 10103

    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, cs=None,
                 alpha_coeffs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            kijs = [[0.0]*N for i in cmps]
        self.kijs = kijs
        self.T = T
        self.P = P
        self.V = V

        c1R2, c2R = self.c1*R2, self.c2*R
        self.ais = [c1R2*Tcs[i]*Tcs[i]/Pcs[i] for i in cmps]
        b0s = [c2R*Tcs[i]/Pcs[i] for i in cmps]

        if cs is None:
            cs = [0.0]*N # TODO peneloux? Inherit?
        if alpha_coeffs is None:
            alpha_coeffs = []
            for i in cmps:
                alpha_coeffs.append(MSRKTranslated.estimate_MN(Tcs[i], Pcs[i], omegas[i], cs[i]))

        self.kwargs = {'kijs': kijs, 'alpha_coeffs': alpha_coeffs, 'cs': cs}
        self.alpha_coeffs = alpha_coeffs
        self.cs = cs

        b0, c = 0.0, 0.0
        for i in cmps:
            b0 += b0s[i]*zs[i]
            c += cs[i]*zs[i]

        self.b0s = b0s
        self.bs = [b0s[i] - cs[i] for i in cmps]
        self.c = c
        self.b = b = b0 - c
        self.delta = c + c + b0
        self.epsilon = c*(b0 + c)
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

class PSRK(Mathias_Copeman_a_alpha, PSRKMixingRules, SRKMIXTranslated):
    r'''Class for solving the Predictive Soave-Redlich-Kwong [1]_ equation of
    state for a mixture of any number of compounds.
    Solves the EOS on initialization.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. warning::
        This class is not complete! Fugacities and their derivatives among
        others are not yet implemented.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}

    .. math::
        b_i =\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    alpha_coeffs : list[list[float]]
        Coefficients for
        :obj:`thermo.eos_alpha_functions.Mathias_Copeman_a_alpha`, [-]
    ge_model : :obj:`thermo.activity.GibbsExcess` object
        Excess Gibbs free energy model; to match the `PSRK` model, this is
        a :obj:`thermo.unifac.UNIFAC` object, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    cs : list[float], optional
        Volume translation parameters; always zero in the original
        implementation, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, equimolar CO2, n-hexane:


    >>> from thermo.unifac import UNIFAC, PSRKIP, PSRKSG
    >>> Tcs = [304.2, 507.4]
    >>> Pcs = [7.37646e6, 3.014419e6]
    >>> omegas = [0.2252, 0.2975]
    >>> zs = [0.5, 0.5]
    >>> Mathias_Copeman_coeffs = [[-1.7039, 0.2515, 0.8252, 1.0], [2.9173, -1.4411, 1.1061, 1.0]]
    >>> T = 313.
    >>> P = 1E6
    >>> ge_model = UNIFAC.from_subgroups(T=T, xs=zs, chemgroups=[{117: 1}, {1:2, 2:4}], subgroups=PSRKSG, interaction_data=PSRKIP, version=0)
    >>> eos = PSRK(Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, ge_model=ge_model, alpha_coeffs=Mathias_Copeman_coeffs, T=T, P=P)
    >>> eos
    PSRK(Tcs=[304.2, 507.4], Pcs=[7376460.0, 3014419.0], omegas=[0.2252, 0.2975], kijs=[[0.0, 0.0], [0.0, 0.0]], alpha_coeffs=[[-1.7039, 0.2515, 0.8252, 1.0], [2.9173, -1.4411, 1.1061, 1.0]], cs=[0.0, 0.0], ge_model=UNIFAC(T=313.0, xs=[0.5, 0.5], rs=[1.3, 4.4998000000000005], qs=[0.982, 3.856], Qs=[0.848, 0.54, 0.982], vs=[[0, 2], [0, 4], [1, 0]], psi_abc=([[0.0, 0.0, 919.8], [0.0, 0.0, 919.8], [-38.672, -38.672, 0.0]], [[0.0, 0.0, -3.9132], [0.0, 0.0, -3.9132], [0.8615, 0.8615, 0.0]], [[0.0, 0.0, 0.0046309], [0.0, 0.0, 0.0046309], [-0.0017906, -0.0017906, 0.0]]), version=0), zs=[0.5, 0.5], T=313.0, P=1000000.0)
    >>> eos.phase, eos.V_l, eos.V_g
    ('l/g', 0.000110889753959, 0.00197520225546)

    Notes
    -----

    References
    ----------
    .. [1] Holderbaum, T., and J. Gmehling. "PSRK: A Group Contribution
       Equation of State Based on UNIFAC.” Fluid Phase Equilibria 70, no. 2-3
       (December 30, 1991): 251-65.
       https://doi.org/10.1016/0378-3812(91)85038-V.
    '''
    eos_pure = SRKTranslated
    mix_kwargs_to_pure = {'cs': 'c', 'alpha_coeffs': 'alpha_coeffs'}
    kwargs_linear = ('cs', 'alpha_coeffs')
    kwargs_keys = ('kijs', 'alpha_coeffs', 'cs', 'ge_model')
    model_id = 10300

    def __init__(self, Tcs, Pcs, omegas, zs, alpha_coeffs, ge_model,
                 kijs=None, cs=None,
                 T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            kijs = [[0.0]*N for i in cmps]
        if cs is None:
            cs = [0.0]*N
        self.kijs = kijs
        self.T = T
        self.P = P
        self.V = V

        c1R2, c2R = self.c1*R2, self.c2*R
        self.ais = [c1R2*Tcs[i]*Tcs[i]/Pcs[i] for i in cmps]
        b0s = [c2R*Tcs[i]/Pcs[i] for i in cmps]


        self.kwargs = {'kijs': kijs, 'alpha_coeffs': alpha_coeffs, 'cs': cs,
                       'ge_model': ge_model}
        self.alpha_coeffs = alpha_coeffs
        self.cs = cs

        if zs != ge_model.xs or ge_model.T != T:
            if T is None:
                T = 298.15 # default value, need to check in a_alpha call
            ge_model = ge_model.to_T_xs(T, zs)
        self.ge_model = ge_model

        b0, c = 0.0, 0.0
        for i in cmps:
            b0 += b0s[i]*zs[i]
            c += cs[i]*zs[i]

        self.b0s = b0s
        self.bs = [b0s[i] - cs[i] for i in cmps]
        self.c = c
        self.b = b = b0 - c
        self.delta = c + c + b0
        self.epsilon = c*(b0 + c)
        self.solve(only_l=only_l, only_g=only_g)
#        if fugacities:
#            self.fugacities()

    def _fast_init_specific(self, other):
        zs = self.zs
        self.ge_model = other.ge_model.to_T_xs(self.T, zs)
        self.cs = cs = other.cs
        self.alpha_coeffs = other.alpha_coeffs
        self.b0s = b0s = other.b0s

        b0, c = 0.0, 0.0
        for i in range(self.N):
            b0 += b0s[i]*zs[i]
            c += cs[i]*zs[i]

        self.c = c
        self.b = b0 - c
        self.delta = c + c + b0
        self.epsilon = c*(b0 + c)


class PR78MIX(PRMIX):
    r'''Class for solving the Peng-Robinson cubic equation of state for a
    mixture of any number of compounds according to the 1978 variant.
    Subclasses `PR`. Solves the EOS on initialization and calculates fugacities
    for all components in all phases.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}

    .. math::
	    b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}

    .. math::
        \alpha(T)_i=[1+\kappa_i(1-\sqrt{T_{r,i}})]^2

    .. math::
        \kappa_i = 0.37464+1.54226\omega_i-0.26992\omega_i^2 \text{ if } \omega_i
        \le 0.491

    .. math::
        \kappa_i = 0.379642 + 1.48503 \omega_i - 0.164423\omega_i^2 + 0.016666
        \omega_i^3 \text{ if } \omega_i > 0.491

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa, with modified
    acentric factors to show the difference between :obj:`PRMIX`

    >>> eos = PR78MIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.6, 0.7], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (3.2396438915e-05, 0.00050433802024)
    >>> eos.fugacities_l, eos.fugacities_g
    ([833048.45119, 6160.9088153], [460717.27767, 279598.90103])

    Notes
    -----
    This variant is recommended over the original.

    References
    ----------
    .. [1] Peng, Ding-Yu, and Donald B. Robinson. "A New Two-Constant Equation
       of State." Industrial & Engineering Chemistry Fundamentals 15, no. 1
       (February 1, 1976): 59-64. doi:10.1021/i160057a011.
    .. [2] Robinson, Donald B., Ding-Yu Peng, and Samuel Y-K Chung. "The
       Development of the Peng - Robinson Equation and Its Application to Phase
       Equilibrium in a System Containing Methanol." Fluid Phase Equilibria 24,
       no. 1 (January 1, 1985): 25-41. doi:10.1016/0378-3812(85)87035-7.
    '''
    eos_pure = PR78
    model_id = 10201
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in range(N)]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V
        
        c1R2_c2R, c2R = self.c1R2_c2R, self.c2R

        if scalar:
            self.bs = bs = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*bs[i] for i in cmps]
            self.kappas = kappas = [omega*(-0.26992*omega + 1.54226) + 0.37464 for omega in omegas]
            for i, omega in enumerate(omegas):
                if omega > 0.491:
                    kappas[i] = omega*(omega*(0.016666*omega - 0.164423) + 1.48503) + 0.379642
            b = 0.0
            for i in cmps:
                b += bs[i]*zs[i]
            
        else:
            self.bs = bs = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*bs
            self.kappas = kappas = omegas*(-0.26992*omegas + 1.54226) + 0.37464
            b = float((bs*zs).sum())
            high_omega_idxs = npwhere(omegas > 0.491)
            high_omegas = omegas[high_omega_idxs]
            kappas[high_omega_idxs] = high_omegas*(high_omegas*(0.016666*high_omegas - 0.164423) + 1.48503) + 0.379642
        self.b = b

        self.delta = 2.*b
        self.epsilon = -b*b

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()



class VDWMIX(EpsilonZeroMixingRules, GCEOSMIX, VDW):
    r'''Class for solving the Van der Waals [1]_ [2]_ cubic equation of state for a
    mixture of any number of compounds. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P=\frac{RT}{V-b}-\frac{a}{V^2}

    .. math::
        a = \sum_i \sum_j z_i z_j {a}_{ij}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_{ij} = (1-k_{ij})\sqrt{a_{i}a_{j}}

    .. math::
        a_i=\frac{27}{64}\frac{(RT_{c,i})^2}{P_{c,i}}

    .. math::
        b_i=\frac{RT_{c,i}}{8P_{c,i}}

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    omegas : float, optional
        Acentric factors of all compounds - Not used in equation of state!, [-]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = VDWMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (5.881369844883e-05, 0.00077708723758)
    >>> eos.fugacities_l, eos.fugacities_g
    ([854533.266920, 207126.8497276], [448470.736338, 397826.543999])

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    References
    ----------
    .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
       Butterworth-Heinemann, 1985.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th
       edition. New York: McGraw-Hill Professional, 2000.
    '''
    eos_pure = VDW
    nonstate_constants_specific = tuple()
    kwargs_keys = ('kijs',)
    model_id = 10001

    def __init__(self, Tcs, Pcs, zs, kijs=None, T=None, P=None, V=None,
                 omegas=None, fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*self.N for i in range(N)]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V
        
        c1R2, c2R = self.c1R2, self.c2R
        if self.scalar:
            self.ais = [c1R2*Tc*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
            self.bs = [c2R*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
            self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        else:
            Tc_Pc_ratio = Tcs/Pcs
            self.ais = c1R2*Tcs*Tc_Pc_ratio
            self.bs = bs = c2R*Tc_Pc_ratio
            self.b = float((bs*zs).sum())
    
        self.omegas = omegas
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        if self.scalar:
            self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        else:
            self.b = float((self.bs*self.zs).sum())

    def a_alphas_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` for the VDW EOS.
        This vectorized implementation is added for extra speed.

        .. math::
            a\alpha = a

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        '''
        return self.ais

    def a_alpha_and_derivatives_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` and their first
        and second derivatives for the VDW EOS. This vectorized implementation
        is added for extra speed.

        .. math::
            a\alpha = a

        .. math::
            \frac{d a\alpha}{dT} = 0

        .. math::
            \frac{d^2 a\alpha}{dT^2} = 0

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dTs : list[float]
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2s : list[float]
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K**2]
        '''
        if self.scalar:
            zero_array = [0.0]*self.N
        else:
            zero_array = zeros(self.N)
        return self.ais, zero_array, zero_array

    def fugacity_coefficients(self, Z):
        r'''Literature formula for calculating fugacity coefficients for each
        species in a mixture. Verified numerically.
        Called by `fugacities` on initialization, or by a solver routine
        which is performing a flash calculation.

        .. math::
            \ln \hat \phi_i = \frac{b_i}{V-b} - \ln\left[Z\left(1
            - \frac{b}{V}\right)\right] - \frac{2\sqrt{aa_i}}{RTV}

        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]

        Returns
        -------
        log_phis : float
            Log fugacity coefficient for each species, [-]

        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
           Butterworth-Heinemann, 1985.
        '''
        N = self.N
        return VDW_lnphis(self.T, self.P, Z, self.b, self.a_alpha, self.bs, self.a_alpha_roots, N,
                          lnphis=[0.0]*N if self.scalar else zeros(N))

    def dlnphis_dT(self, phase):
        r'''Formula for calculating the temperature derivaitve of
        log fugacity coefficients for each species in a mixture for the
        VDW equation of state. Verified numerically.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial T}\right)_{P,
            nj \ne i}

        Parameters
        ----------
        phase : str
            One of 'l' or 'g', [-]

        Returns
        -------
        dlnphis_dT : float
            Temperature derivatives of log fugacity coefficient for each
            species, [1/K]

        Notes
        -----
        This expression was derived using SymPy and optimized with the `cse`
        technique.
        '''
        zs = self.zs
        if phase == 'g':
            Z = self.Z_g
            dZ_dT = self.dZ_dT_g
        else:
            Z = self.Z_l
            dZ_dT = self.dZ_dT_l

        N = self.N
        T, P, ais, bs, b = self.T, self.P, self.ais, self.bs, self.b

        T_inv = 1.0/T
        T_inv2 = T_inv*T_inv
        A = self.a_alpha*P*R2_inv*T_inv2
        B = b*P*R_inv*T_inv

        x0 = self.a_alpha
        x4 = 1.0/Z
        x5 = 4.0*P*R2_inv*x4*T_inv2*T_inv

        x8 = 2*P*R2_inv*T_inv2*dZ_dT/Z**2
        x9 = P*R2_inv*x4*T_inv2*self.da_alpha_dT/x0
        x10 = 1.0/P
        x11 = R*x10*(T*dZ_dT + Z)/(-R*T*x10*Z + b)**2
        x13 = b*T_inv*R_inv
        x14 = P*x13*x4 - 1.0
        x15 = x4*(P*x13*(T_inv + x4*dZ_dT) - x14*dZ_dT)/x14

        # Composition stuff
        d_lnphis_dTs = []
        for i in range(N):
            x1 = (ais[i]*x0)**0.5
            d_lhphi_dT = -bs[i]*x11 + x1*x5 + x1*x8 - x1*x9 + x15
            d_lnphis_dTs.append(d_lhphi_dT)
        return d_lnphis_dTs

    def dlnphis_dP(self, phase):
        r'''Generic formula for calculating the pressure derivaitve of
        log fugacity coefficients for each species in a mixture for the
        VDW EOS. Verified numerically.

        .. math::
            \left(\frac{\partial \ln \phi_i}{\partial P}\right)_{T,
            nj \ne i}

        Parameters
        ----------
        phase : str
            One of 'l' or 'g', [-]

        Returns
        -------
        dlnphis_dP : float
            Pressure derivatives of log fugacity coefficient for each species,
            [1/Pa]

        Notes
        -----
        This expression was derived using SymPy and optimized with the `cse`
        technique.
        '''
        zs = self.zs
        if phase == 'l':
            Z, dZ_dP = self.Z_l, self.dZ_dP_l
        else:
            Z, dZ_dP = self.Z_g, self.dZ_dP_g
        a_alpha = self.a_alpha
        N = self.N
        T, P, bs, b, ais = self.T, self.P, self.bs, self.b, self.ais

        T_inv = 1.0/T
        RT_inv = T_inv*R_inv

        x3 = T_inv*T_inv
        x5 = 1.0/Z
        x6 = 2.0*R2_inv*x3*x5
        x8 = 2.0*P*R2_inv*x3*dZ_dP*x5*x5
        x9 = 1./P
        x10 = Z*x9
        x11 = R*T*x9*(-x10 + dZ_dP)/(-R*T*x10 + b)**2
        x12 = P*x5
        x13 = b*RT_inv
        x14 = x12*x13 - 1.0
        x15 = -x5*(-x13*(x12*dZ_dP - 1.0) + x14*dZ_dP)/x14

        d_lnphi_dPs = []
        for i in range(N):
            x1 = (ais[i]*a_alpha)**0.5
            d_lnphi_dP = -bs[i]*x11 - x1*x6 + x1*x8 + x15
            d_lnphi_dPs.append(d_lnphi_dP)
        return d_lnphi_dPs

    @property
    def ddelta_dzs(self):
        r'''Helper method for calculating the composition derivatives of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \delta}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            = 0

        Returns
        -------
        ddelta_dzs : list[float]
            Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        if self.scalar:
            zero_array = [0.0]*self.N
        else:
            zero_array = zeros(self.N)
        return zero_array

    @property
    def ddelta_dns(self):
        r'''Helper method for calculating the mole number derivatives of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial \delta}{\partial n_i}\right)_{T, P, n_{i\ne j}}
            = 0

        Returns
        -------
        ddelta_dns : list[float]
            Mole number derivative of `delta` of each component, [m^3/mol^2]

        Notes
        -----
        This derivative is checked numerically.
        '''
        if self.scalar:
            zero_array = [0.0]*self.N
        else:
            zero_array = zeros(self.N)
        return zero_array


    @property
    def d2delta_dzizjs(self):
        r'''Helper method for calculating the second composition derivatives (hessian) of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \delta}{\partial x_i\partial x_j}\right)_{T, P, x_{k\ne i,j}}
            = 0

        Returns
        -------
        d2delta_dzizjs : list[float]
            Second Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            zero_array = [[0.0]*N for i in range(N)]
        else:
            zero_array = zeros((N, N))
        return zero_array

    @property
    def d2delta_dninjs(self):
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^2 \delta}{\partial n_i \partial n_j}\right)_{T, P, n_{k\ne i,j}}
            = 0

        Returns
        -------
        d2delta_dninjs : list[list[float]]
            Second mole number derivative of `delta` of each component, [m^3/mol^3]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            zero_array = [[0.0]*N for i in range(N)]
        else:
            zero_array = zeros((N, N))
        return zero_array

    @property
    def d3delta_dninjnks(self):
        r'''Helper method for calculating the third partial mole number
        derivatives of `delta`. Note this is independent of the phase.

        .. math::
            \left(\frac{\partial^3 \delta}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P,
            n_{m \ne i,j,k}} = 0

        Returns
        -------
        d3delta_dninjnks : list[list[list[float]]]
            Third mole number derivative of `delta` of each component,
            [m^3/mol^4]

        Notes
        -----
        This derivative is checked numerically.
        '''
        N = self.N
        if self.scalar:
            zero_array = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            zero_array = zeros((N, N, N))
        return zero_array


class PRSVMIX(PRMIX, PRSV):
    r'''Class for solving the Peng-Robinson-Stryjek-Vera equations of state for
    a mixture as given in [1]_.  Subclasses :obj:`PRMIX` and :obj:`PRSV <thermo.eos.PRSV>`.
    Solves the EOS on initialization and calculates fugacities for all
    components in all phases.

    Inherits the method of calculating fugacity coefficients from :obj:`PRMIX`.
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}

    .. math::
	    b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}

    .. math::
        \alpha(T)_i=[1+\kappa_i(1-\sqrt{T_{r,i}})]^2

    .. math::
        \kappa_i = \kappa_{0,i} + \kappa_{1,i}(1 + T_{r,i}^{0.5})(0.7 - T_{r,i})

    .. math::
        \kappa_{0,i} = 0.378893 + 1.4897153\omega_i - 0.17131848\omega_i^2
        + 0.0196554\omega_i^3

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    kappa1s : list[float], optional
        Fit parameter; available in [1]_ for over 90 compounds, SRKMIXTranslated[-]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    P-T initialization, two-phase, nitrogen and methane

    >>> eos = PRSVMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l/g', 3.6235536165e-05, -6349.0055583, -49.1240502472)

    Notes
    -----
    [1]_ recommends that `kappa1` be set to 0 for Tr > 0.7. This is not done by
    default; the class boolean `kappa1_Tr_limit` may be set to True and the
    problem re-solved with that specified if desired. `kappa1_Tr_limit` is not
    supported for P-V inputs.

    For P-V initializations, a numerical solver is used to find T.

    [2]_ and [3]_ are two more resources documenting the PRSV EOS. [4]_ lists
    `kappa` values for 69 additional compounds. See also :obj:`PRSV2MIX`. Note that
    tabulated `kappa` values should be used with the critical parameters used
    in their fits. Both [1]_ and [4]_ only considered vapor pressure in fitting
    the parameter.

    References
    ----------
    .. [1] Stryjek, R., and J. H. Vera. "PRSV: An Improved Peng-Robinson
       Equation of State for Pure Compounds and Mixtures." The Canadian Journal
       of Chemical Engineering 64, no. 2 (April 1, 1986): 323-33.
       doi:10.1002/cjce.5450640224.
    .. [2] Stryjek, R., and J. H. Vera. "PRSV - An Improved Peng-Robinson
       Equation of State with New Mixing Rules for Strongly Nonideal Mixtures."
       The Canadian Journal of Chemical Engineering 64, no. 2 (April 1, 1986):
       334-40. doi:10.1002/cjce.5450640225.
    .. [3] Stryjek, R., and J. H. Vera. "Vapor-liquid Equilibrium of
       Hydrochloric Acid Solutions with the PRSV Equation of State." Fluid
       Phase Equilibria 25, no. 3 (January 1, 1986): 279-90.
       doi:10.1016/0378-3812(86)80004-8.
    .. [4] Proust, P., and J. H. Vera. "PRSV: The Stryjek-Vera Modification of
       the Peng-Robinson Equation of State. Parameters for Other Pure Compounds
       of Industrial Interest." The Canadian Journal of Chemical Engineering
       67, no. 1 (February 1, 1989): 170-73. doi:10.1002/cjce.5450670125.
    '''
    eos_pure = PRSV
    nonstate_constants_specific = ('kappa0s', 'kappa1s', 'kappas')
    mix_kwargs_to_pure = {'kappa1s': 'kappa1'}
    kwargs_linear = ('kappa1s',)
    kwargs_keys = ('kijs', 'kappa1s')
    model_id = 10205

    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 kappa1s=None, fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list

        if kijs is None:
            if scalar:
                kijs = [[0]*self.N for i in range(N)]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs

        if kappa1s is None:
            if scalar:
                kappa1s = [0.0 for i in range(N)]
            else:
                kappa1s = zeros(N)
        self.kwargs = {'kijs': kijs, 'kappa1s': kappa1s}
        self.T = T
        self.P = P
        self.V = V
        
        c1R2_c2R, c2R = self.c1R2_c2R, self.c2R

        
        if scalar:
            self.kappa0s = [omega*(omega*(0.0196554*omega - 0.17131848) + 1.4897153) + 0.378893 for omega in omegas]
            self.bs = bs = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*bs[i] for i in cmps]
            b = 0.0
            for i in cmps:
                b += bs[i]*zs[i]
        else:
            self.kappa0s = omegas*(omegas*(0.0196554*omegas - 0.17131848) + 1.4897153) + 0.378893
            self.bs = bs = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*bs
            b = float((bs*zs).sum())
            
        self.b = b

        self.delta = 2.0*b
        self.epsilon = -b*b

        self.check_sufficient_inputs()
        if self.V and self.P:
            # Deal with T-solution here; does NOT support kappa1_Tr_limit.
            self.kappa1s = kappa1s
            solution = 'g' if (only_g and not only_l) else ('l' if only_l else None)
            self.T = self.solve_T(self.P, self.V, solution=solution)
        else:
            self.kappa1s = [(0 if (T/Tc > 0.7 and self.kappa1_Tr_limit) else kappa1) for kappa1, Tc in zip(kappa1s, Tcs)]

        self.kappas = [kappa0 + kappa1*(1 + (self.T/Tc)**0.5)*(0.7 - (self.T/Tc)) for kappa0, kappa1, Tc in zip(self.kappa0s, self.kappa1s, self.Tcs)]

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()



    def _fast_init_specific(self, other):
        self.kappa0s = other.kappa0s
        self.kappa1s = other.kappa1s
        self.kappas = other.kappas
        if self.scalar:
            b = 0.0
            for bi, zi in zip(self.bs, self.zs):
                b += bi*zi
        else:
            b = float((self.bs*self.zs).sum())
        self.b = b
        self.delta = 2.0*b
        self.epsilon = -b*b

    def a_alphas_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` for the PRSV EOS.
        This vectorized implementation is added for extra speed.

        .. math::
            a\alpha = a \left(\left(\kappa_{0} + \kappa_{1} \left(\sqrt{\frac{
            T}{Tc}} + 1\right) \left(- \frac{T}{Tc} + \frac{7}{10}\right)
            \right) \left(- \sqrt{\frac{T}{Tc}} + 1\right) + 1\right)^{2}

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        '''
        return PRSV_a_alphas_vectorized(T, self.Tcs, self.ais, self.kappa0s, self.kappa1s,
                                        a_alphas=[0.0]*self.N if self.scalar else zeros(self.N))

    def a_alpha_and_derivatives_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` and their first
        and second derivatives for the PRSV EOS. This vectorized implementation
        is added for extra speed.

        .. math::
            a\alpha = a \left(\left(\kappa_{0} + \kappa_{1} \left(\sqrt{\frac{
            T}{Tc}} + 1\right) \left(- \frac{T}{Tc} + \frac{7}{10}\right)
            \right) \left(- \sqrt{\frac{T}{Tc}} + 1\right) + 1\right)^{2}

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dTs : list[float]
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2s : list[float]
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K**2]
        '''
        N = self.N
        if self.scalar:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [0.0]*N, [0.0]*N, [0.0]*N
        else:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = zeros(N), zeros(N), zeros(N)
        return PRSV_a_alpha_and_derivatives_vectorized(T, self.Tcs, self.ais, self.kappa0s, self.kappa1s,
                                                       a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs, d2a_alpha_dT2s=d2a_alpha_dT2s)


class PRSV2MIX(PRMIX, PRSV2):
    r'''Class for solving the Peng-Robinson-Stryjek-Vera 2 equations of state
    for a Mixture as given in [1]_.  Subclasses :obj:`PRMIX` and `PRSV2 <thermo.eos.PRSV2>`.
    Solves the EOS on initialization and calculates fugacities for all
    components in all phases.

    Inherits the method of calculating fugacity coefficients from :obj:`PRMIX`.
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}

    .. math::
	    b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}

    .. math::
        \alpha(T)_i=[1+\kappa_i(1-\sqrt{T_{r,i}})]^2

    .. math::
        \kappa_i = \kappa_{0,i} + [\kappa_{1,i} + \kappa_{2,i}(\kappa_{3,i} - T_{r,i})(1-T_{r,i}^{0.5})]
        (1 + T_{r,i}^{0.5})(0.7 - T_{r,i})

    .. math::
        \kappa_{0,i} = 0.378893 + 1.4897153\omega_i - 0.17131848\omega_i^2
        + 0.0196554\omega_i^3

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    kappa1s : list[float], optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]
    kappa2s : list[float], optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]
    kappa3s : list[float], optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = PRSV2MIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (3.6235536165e-05, 0.00070024238654)
    >>> eos.fugacities_l, eos.fugacities_g
    ([794057.58318, 72851.22327], [436553.65618, 357878.11066])

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    Note that tabulated `kappa` values should be used with the critical
    parameters used in their fits. [1]_ considered only vapor
    pressure in fitting the parameter.

    References
    ----------
    .. [1] Stryjek, R., and J. H. Vera. "PRSV2: A Cubic Equation of State for
       Accurate Vapor-liquid Equilibria Calculations." The Canadian Journal of
       Chemical Engineering 64, no. 5 (October 1, 1986): 820-26.
       doi:10.1002/cjce.5450640516.
    '''
    eos_pure = PRSV2
    nonstate_constants_specific = ('kappa1s', 'kappa2s', 'kappa3s', 'kappa0s', 'kappas')
    mix_kwargs_to_pure = {'kappa1s': 'kappa1', 'kappa2s': 'kappa2', 'kappa3s': 'kappa3'}
    kwargs_linear = ('kappa1s', 'kappa2s', 'kappa3s')
    kwargs_keys = ('kijs', 'kappa1s', 'kappa2s', 'kappa3s')
    model_id = 10206

    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 kappa1s=None, kappa2s=None, kappa3s=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list

        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in cmps]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        
        if scalar:
            if kappa1s is None:
                kappa1s = [0.0]*N
            if kappa2s is None:
                kappa2s = [0.0]*N
            if kappa3s is None:
                kappa3s = [0.0]*N
        else:
            if kappa1s is None:
                kappa1s = zeros(N)
            if kappa2s is None:
                kappa2s = zeros(N)
            if kappa3s is None:
                kappa3s = zeros(N)
        
        self.kwargs = {'kijs': kijs, 'kappa1s': kappa1s, 'kappa2s': kappa2s, 'kappa3s': kappa3s}
        self.kappa1s = kappa1s
        self.kappa2s = kappa2s
        self.kappa3s = kappa3s

        self.T = T
        self.P = P
        self.V = V
        
        c2R, c1R2_c2R = self.c2R, self.c1R2_c2R

        if scalar:
            self.bs = bs = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*bs[i] for i in cmps]
            self.kappa0s = kappa0s = [omega*(omega*(0.0196554*omega - 0.17131848) + 1.4897153) + 0.378893 for omega in omegas]
            b = 0.0
            for i in cmps:
                b += bs[i]*zs[i]
        else:
            self.bs = bs = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*bs
            self.kappa0s = kappa0s = omegas*(omegas*(0.0196554*omegas - 0.17131848) + 1.4897153) + 0.378893
            b = float((bs*zs).sum())
        self.b = b

        self.delta = 2.0*b
        self.epsilon = -b*b


        if self.V and self.P:
            solution = 'g' if (only_g and not only_l) else ('l' if only_l else None)
            self.T = T = self.solve_T(self.P, self.V, solution=solution)

        
        if scalar:
            kappas = [0.0]*N
            for i in cmps:
                Tr = T/Tcs[i]
                sqrtTr = sqrt(Tr)
                kappas[i] = kappa0s[i] + ((kappa1s[i] + kappa2s[i]*(kappa3s[i] - Tr)*(1. - sqrtTr))*(1. + sqrtTr)*(0.7 - Tr))
        else:
            Trs = T/Tcs
            sqrtTrs = npsqrt(Trs)
            kappas = kappa0s + ((kappa1s + kappa2s*(kappa3s - Trs)*(1. - sqrtTrs))*(1. + sqrtTrs)*(0.7 - Trs))
        
        self.kappas = kappas
        
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        self.kappa0s = other.kappa0s
        self.kappa1s = other.kappa1s
        self.kappa2s = other.kappa2s
        self.kappa3s = other.kappa3s
        self.kappas = other.kappas
        if self.scalar:
            b = 0.0
            for bi, zi in zip(self.bs, self.zs):
                b += bi*zi
        else:
            b = float((self.bs*self.zs).sum())
        self.b = b
        self.delta = b + b
        self.epsilon = -b*b

    def a_alphas_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` for the PRSV2
        EOS. This vectorized implementation is added for extra speed.

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

        Examples
        --------
        >>> eos = PRSV2MIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
        >>> eos.a_alphas_vectorized(300)
        [0.0860568595, 0.20174345803]
        '''
        return PRSV2_a_alphas_vectorized(T, self.Tcs, self.ais, self.kappa0s, self.kappa1s, self.kappa2s, self.kappa3s,
                                         a_alphas=([0.0]*self.N if self.scalar else zeros(self.N)))

    def a_alpha_and_derivatives_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` and their first
        and second derivatives for the PRSV2 EOS. This vectorized
        implementation is added for extra speed.

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dTs : list[float]
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2s : list[float]
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K**2]
        '''
        N = self.N
        if self.scalar:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [0.0]*N, [0.0]*N, [0.0]*N
        else:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = zeros(N), zeros(N), zeros(N)
        return PRSV2_a_alpha_and_derivatives_vectorized(T, self.Tcs, self.ais, self.kappa0s, self.kappa1s, self.kappa2s, self.kappa3s,
                                                        a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs, d2a_alpha_dT2s=d2a_alpha_dT2s)


class TWUPRMIX(TwuPR95_a_alpha, PRMIX):
    r'''Class for solving the Twu [1]_ variant of the Peng-Robinson cubic
    equation of state for a mixture. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}

    .. math::
	    b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}

    .. math::
       \alpha_i = \alpha_i^{(0)} + \omega_i(\alpha_i^{(1)}-\alpha_i^{(0)})

    .. math::
       \alpha^{(\text{0 or 1})} = T_{r,i}^{N(M-1)}\exp[L(1-T_{r,i}^{NM})]

    For sub-critical conditions:

    L0, M0, N0 =  0.125283, 0.911807,  1.948150;

    L1, M1, N1 = 0.511614, 0.784054, 2.812520

    For supercritical conditions:

    L0, M0, N0 = 0.401219, 4.963070, -0.2;

    L1, M1, N1 = 0.024955, 1.248089, -8.

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = TWUPRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (3.624571041e-05, 0.0007004401318)
    >>> eos.fugacities_l, eos.fugacities_g
    ([792155.022163, 73305.88829], [436468.967764, 358049.2495573])

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.
    Claimed to be more accurate than the PR, PR78 and PRSV equations.

    References
    ----------
    .. [1] Twu, Chorng H., John E. Coon, and John R. Cunningham. "A New
       Generalized Alpha Function for a Cubic Equation of State Part 1.
       Peng-Robinson Equation." Fluid Phase Equilibria 105, no. 1 (March 15,
       1995): 49-59. doi:10.1016/0378-3812(94)02601-V.
    '''
    eos_pure = TWUPR
    P_max_at_V = GCEOS.P_max_at_V
    solve_T = GCEOS.solve_T
    kwargs_keys = ('kijs', )
    model_id = 10204

    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in cmps]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V
        
        c2R, c1R2_c2R = self.c2R, self.c1R2_c2R
        if scalar:
            self.bs = bs = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*bs[i] for i in cmps]
            b = 0.0
            for i in cmps:
                b += bs[i]*zs[i]
        else:
            self.bs = bs = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*bs
            b = float((bs*zs).sum())
        self.b = b

        self.delta = 2.*b
        self.epsilon = -b*b
        self.check_sufficient_inputs()

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        if self.scalar:
            b = 0.0
            for bi, zi in zip(self.bs, self.zs):
                b += bi*zi
        else:
            b = float((self.bs*self.zs).sum())
        self.b = b
        self.delta = 2.0*b
        self.epsilon = -b*b

class TWUSRKMIX(TwuSRK95_a_alpha, SRKMIX):
    r'''Class for solving the Twu variant of the Soave-Redlich-Kwong cubic
    equation of state for a mixture. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}

    .. math::
        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}

    .. math::
        b_i =\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        \alpha_i = \alpha^{(0,i)} + \omega_i(\alpha^{(1,i)}-\alpha^{(0,i)})

    .. math::
        \alpha^{(\text{0 or 1, i})} = T_{r,i}^{N(M-1)}\exp[L(1-T_{r,i}^{NM})]

    For sub-critical conditions:

    L0, M0, N0 =  0.141599, 0.919422, 2.496441

    L1, M1, N1 = 0.500315, 0.799457, 3.291790

    For supercritical conditions:

    L0, M0, N0 = 0.441411, 6.500018, -0.20

    L1, M1, N1 = 0.032580,  1.289098, -8.0

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = TWUSRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (4.1087927542e-05, 0.00071170732525)
    >>> eos.fugacities_l, eos.fugacities_g
    ([809692.830826, 74093.6388157], [441783.431489, 362470.3174107])

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.
    Claimed to be more accurate than the SRK equation.

    References
    ----------
    .. [1] Twu, Chorng H., John E. Coon, and John R. Cunningham. "A New
       Generalized Alpha Function for a Cubic Equation of State Part 2.
       Redlich-Kwong Equation." Fluid Phase Equilibria 105, no. 1 (March 15,
       1995): 61-69. doi:10.1016/0378-3812(94)02602-W.
    '''
#    a_alpha_mro = -5
    kwargs_keys = ('kijs', )
    eos_pure = TWUSRK
    P_max_at_V = GCEOS.P_max_at_V
    solve_T = GCEOS.solve_T
    model_id = 10104
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in range(N)]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        c2R, c1R2_c2R = self.c2R, self.c1R2_c2R
        if scalar:
            self.bs = bs = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*bs[i] for i in cmps]
            b = 0.0
            for i in cmps:
                b += bs[i]*zs[i]
        else:
            self.bs = bs = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*bs
            b = float((bs*zs).sum())

        self.delta = self.b = b
        self.check_sufficient_inputs()

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        b = 0.0
        bs, zs = self.bs, self.zs
        if self.scalar:
            for i in range(self.N):
                b += bs[i]*zs[i]
        else:
            b = float((bs*zs).sum())
        self.delta = self.b = b

class APISRKMIX(SRKMIX, APISRK):
    r'''Class for solving the Refinery Soave-Redlich-Kwong cubic
    equation of state for a mixture of any number of compounds, as shown in the
    API Databook [1]_. Subclasses :obj:`APISRK <thermo.eos.APISRK>`. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        b = \sum_i z_i b_i

    .. math::
        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}

    .. math::
        b_i =\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}

    .. math::
        \alpha(T)_i = \left[1 + S_{1,i}\left(1-\sqrt{T_{r,i}}\right) + S_{2,i}
        \frac{1- \sqrt{T_{r,i}}}{\sqrt{T_{r,i}}}\right]^2

    .. math::
        S_{1,i} = 0.48508 + 1.55171\omega_i - 0.15613\omega_i^2 \text{ if S1 is not tabulated }

    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    S1s : float, optional
        Fit constant or estimated from acentric factor if not provided [-]
    S2s : float, optional
        Fit constant or 0 if not provided [-]
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optional
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

    Notes
    -----
    For P-V initializations, a numerical solver is used to find T.

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:

    >>> eos = APISRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (4.101592310e-05, 0.00071046883030)
    >>> eos.fugacities_l, eos.fugacities_g
    ([817882.3033, 71620.4823812], [442158.29113, 361519.79877])

    References
    ----------
    .. [1] API Technical Data Book: General Properties & Characterization.
       American Petroleum Institute, 7E, 2005.
    '''
    eos_pure = APISRK
    nonstate_constants_specific = ('S1s', 'S2s')
    mix_kwargs_to_pure = {'S1s': 'S1', 'S2s': 'S2'}
    kwargs_linear = ('S1s', 'S2s')
    kwargs_keys = ('kijs', 'S1s', 'S2s')
    model_id = 10105

    def __init__(self, Tcs, Pcs, zs, omegas=None, kijs=None, T=None, P=None, V=None,
                 S1s=None, S2s=None, fugacities=True, only_l=False, only_g=False):
        self.N = N = len(Tcs)
        cmps = range(N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        self.scalar = scalar = type(zs) is list
        if kijs is None:
            if scalar:
                kijs = [[0.0]*N for i in cmps]
            else:
                kijs = zeros((N, N))
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        self.check_sufficient_inputs()

        # Setup S1s and S2s
        if S1s is None and omegas is None:
            raise ValueError('Either acentric factor of S1 is required')
        if S1s is None:
            if scalar:
                self.S1s = [omega*(1.55171 - 0.15613*omega) + 0.48508 for omega in omegas]
            else:
                self.S1s = omegas*(1.55171 - 0.15613*omegas) + 0.48508
        else:
            self.S1s = S1s
        if S2s is None:
            if scalar:
                S2s = [0.0]*N
            else:
                S2s = zeros(N)
        self.S2s = S2s
        self.kwargs = {'S1s': self.S1s, 'S2s': self.S2s}
        c2R, c1R2_c2R = self.c2R, self.c1R2_c2R
        if scalar:
            self.bs = bs = [c2R*Tcs[i]/Pcs[i] for i in cmps]
            self.ais = [c1R2_c2R*Tcs[i]*bs[i] for i in cmps]
            b = 0.0
            for i in cmps:
                b += bs[i]*zs[i]
        else:
            self.bs = bs = c2R*Tcs/Pcs
            self.ais = c1R2_c2R*Tcs*bs
            b = float((bs*zs).sum())
        self.b = self.delta = b
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def _fast_init_specific(self, other):
        self.S1s = other.S1s
        self.S2s = other.S2s
        if self.scalar:
            self.delta = self.b = sum([bi*zi for bi, zi in zip(self.bs, self.zs)])
        else:
            self.delta = self.b = float((self.bs*self.zs).sum())

    def a_alphas_vectorized(self, T):
        a_alphas = [0.0]*self.N if self.scalar else zeros(self.N)
        return APISRK_a_alphas_vectorized(T, self.Tcs, self.ais, self.S1s, self.S2s, a_alphas=a_alphas)

    def a_alpha_and_derivatives_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` and their first
        and second derivatives for the API SRK EOS. This vectorized implementation
        is added for extra speed.

        .. math::
            a\alpha(T) = a\left[1 + S_1\left(1-\sqrt{T_r}\right) + S_2\frac{1
            - \sqrt{T_r}}{\sqrt{T_r}}\right]^2

        .. math::
            \frac{d a\alpha}{dT} = a\frac{Tc}{T^{2}} \left(- S_{2} \left(\sqrt{
            \frac{T}{Tc}} - 1\right) + \sqrt{\frac{T}{Tc}} \left(S_{1} \sqrt{
            \frac{T}{Tc}} + S_{2}\right)\right) \left(S_{2} \left(\sqrt{\frac{
            T}{Tc}} - 1\right) + \sqrt{\frac{T}{Tc}} \left(S_{1} \left(\sqrt{
            \frac{T}{Tc}} - 1\right) - 1\right)\right)

        .. math::
            \frac{d^2 a\alpha}{dT^2} = a\frac{1}{2 T^{3}} \left(S_{1}^{2} T
            \sqrt{\frac{T}{Tc}} - S_{1} S_{2} T \sqrt{\frac{T}{Tc}} + 3 S_{1}
            S_{2} Tc \sqrt{\frac{T}{Tc}} + S_{1} T \sqrt{\frac{T}{Tc}}
            - 3 S_{2}^{2} Tc \sqrt{\frac{T}{Tc}} + 4 S_{2}^{2} Tc + 3 S_{2}
            Tc \sqrt{\frac{T}{Tc}}\right)

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alphas : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dTs : list[float]
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2s : list[float]
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K**2]
        '''
        N = self.N
        if self.scalar:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [0.0]*N, [0.0]*N, [0.0]*N
        else:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = zeros(N), zeros(N), zeros(N)
        return APISRK_a_alpha_and_derivatives_vectorized(T, self.Tcs, self.ais, self.S1s, self.S2s,
                                                         a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs,
                                                         d2a_alpha_dT2s=d2a_alpha_dT2s)


    def P_max_at_V(self, V):
        if self.N == 1 and self.S2s[0] == 0:
            self.ms = self.S1s
            P_max_at_V = SRK.P_max_at_V(self, V)
            del self.ms
            return P_max_at_V
        return GCEOSMIX.P_max_at_V(self, V)




eos_mix_list = [PRMIX, SRKMIX, PR78MIX, VDWMIX, PRSVMIX, PRSV2MIX, TWUPRMIX,
                TWUSRKMIX, APISRKMIX, IGMIX, RKMIX, PRMIXTranslatedConsistent,
                PRMIXTranslatedPPJP, SRKMIXTranslatedConsistent,
                PRMIXTranslated, SRKMIXTranslated]
'''List of all exported EOS classes.
'''
eos_mix_no_coeffs_list = [PRMIX, SRKMIX, PR78MIX, VDWMIX, TWUPRMIX, TWUSRKMIX,
                          IGMIX, RKMIX, PRMIXTranslatedConsistent, PRMIXTranslated,
                          SRKMIXTranslated,
                          PRMIXTranslatedPPJP, SRKMIXTranslatedConsistent]
'''List of all exported EOS classes that do not require special parameters
or can fill in their special parameters from other specified parameters.
'''

eos_mix_dict = {c.__name__: c for c in eos_mix_list}
'''dict : Dict of all cubic mixture equation of state classes, indexed by their class name.
'''

eos_mix_full_path_dict = {c.__full_path__: c for c in eos_mix_list}
'''dict : Dict of all cubic mixture equation of state classes, indexed by their module path and class name.
'''

eos_mix_full_path_reverse_dict = {c: c.__full_path__ for c in eos_mix_list}
'''dict : Dict of all cubic mixture equation of state classes, indexed by their module path and class name.
'''

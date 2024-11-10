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

This module contains a base class :obj:`GibbsExcess` for handling activity
coefficient based
models. The design is for a sub-class to provide the minimum possible number of
derivatives of Gibbs energy, and for this base class to provide the rest of the
methods.  An ideal-liquid class with no excess Gibbs energy
:obj:`IdealSolution` is also available.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Base Class
==========

.. autoclass:: GibbsExcess
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: d2GE_dT2_numerical, d2GE_dTdxs_numerical, d2GE_dxixjs_numerical, d3GE_dT3_numerical, dGE_dT_numerical, dGE_dxs_numerical
    :special-members: __hash__, __eq__, __repr__

Ideal Liquid Class
==================

.. autoclass:: IdealSolution
    :members: to_T_xs, GE, dGE_dT, d2GE_dT2, d3GE_dT3, d2GE_dTdxs, dGE_dxs, d2GE_dxixjs, d3GE_dxixjxks
    :undoc-members:
    :show-inheritance:
    :exclude-members: gammas

Notes
-----
=====
Excellent references for working with activity coefficient models are [1]_ and
[2]_.

References
----------
.. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
   Butterworth-Heinemann, 1985.
.. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process
   Simulation. Weinheim, Germany: Wiley-VCH, 2012.

'''


__all__ = ['GibbsExcess', 'IdealSolution']
from chemicals.utils import d2xs_to_dxdn_partials, dns_to_dn_partials, dxs_to_dn_partials, dxs_to_dns, hash_any_primitive, normalize, object_data
from fluids.constants import R, R_inv
from fluids.numerics import derivative, exp, hessian, jacobian, log, trunc_exp
from fluids.numerics import numpy as np

from thermo.fitting import fit_customized
from thermo.serialize import JsonOptEncodable

try:
    npexp, ones, zeros, array, ndarray = np.exp, np.ones, np.zeros, np.array, np.ndarray
except:
    pass

def gibbs_excess_gammas(xs, dG_dxs, GE, T, gammas=None):
    xdx_totF = GE
    N = len(xs)
    for i in range(N):
        xdx_totF -= xs[i]*dG_dxs[i]
    RT_inv = R_inv/T
    if gammas is None:
        gammas = [0.0]*N
    for i in range(N):
        gammas[i] = exp((dG_dxs[i] + xdx_totF)*RT_inv)
    return gammas

def gibbs_excess_dHE_dxs(dGE_dxs, d2GE_dTdxs, N, T, dHE_dxs=None):
    if dHE_dxs is None:
        dHE_dxs = [0.0]*N
    for i in range(N):
        dHE_dxs[i] = -T*d2GE_dTdxs[i] + dGE_dxs[i]
    return dHE_dxs


def gibbs_excess_dgammas_dns(xs, gammas, d2GE_dxixjs, N, T, dgammas_dns=None, vec0=None):
    if vec0 is None:
        vec0 = [0.0]*N
    if dgammas_dns is None:
        dgammas_dns = [[0.0]*N for _ in range(N)] # numba : delete
#        dgammas_dns = zeros((N, N)) # numba : uncomment

    for j in range(N):
        tot = 0.0
        row = d2GE_dxixjs[j]
        for k in range(N):
            tot += xs[k]*row[k]
        vec0[j] = tot

    RT_inv = R_inv/(T)

    for i in range(N):
        gammai_RT = gammas[i]*RT_inv
        for j in range(N):
            dgammas_dns[i][j] = gammai_RT*(d2GE_dxixjs[i][j] - vec0[j])

    return dgammas_dns

def gibbs_excess_dgammas_dT(xs, GE, dGE_dT, dG_dxs, d2GE_dTdxs, N, T, dgammas_dT=None):
    if dgammas_dT is None:
        dgammas_dT = [0.0]*N

    xdx_totF0 = dGE_dT
    for j in range(N):
        xdx_totF0 -= xs[j]*d2GE_dTdxs[j]
    xdx_totF1 = GE
    for j in range(N):
        xdx_totF1 -= xs[j]*dG_dxs[j]

    T_inv = 1.0/T
    RT_inv = R_inv*T_inv
    for i in range(N):
        dG_dni = xdx_totF1 + dG_dxs[i]
        dgammas_dT[i] = RT_inv*(d2GE_dTdxs[i] - dG_dni*T_inv + xdx_totF0)*exp(dG_dni*RT_inv)
    return dgammas_dT

def interaction_exp(T, N, A, B, C, D, E, F, lambdas=None):
    if lambdas is None:
        lambdas = [[0.0]*N for i in range(N)] # numba: delete
#        lambdas = zeros((N, N)) # numba: uncomment

#        # 87% of the time of this routine is the exponential.
    T2 = T*T
    Tinv = 1.0/T
    T2inv = Tinv*Tinv
    logT = log(T)
    for i in range(N):
        Ai = A[i]
        Bi = B[i]
        Ci = C[i]
        Di = D[i]
        Ei = E[i]
        Fi = F[i]
        lambdais = lambdas[i]
        # Might be more efficient to pass over this matrix later,
        # and compute all the exps
        # Spoiler: it was not.

        # Also - it was tested the impact of using fewer terms
        # there was very little, to no impact from that
        # the exp is the huge time sink.
        for j in range(N):
            lambdais[j] = exp(Ai[j] + Bi[j]*Tinv
                    + Ci[j]*logT + Di[j]*T
                    + Ei[j]*T2inv + Fi[j]*T2)
#            lambdas[i][j] = exp(A[i][j] + B[i][j]*Tinv
#                    + C[i][j]*logT + D[i][j]*T
#                    + E[i][j]*T2inv + F[i][j]*T2)
#    135 µs ± 1.09 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each) # without out specified numba
#    129 µs ± 2.45 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each) # with out specified numba
#    118 µs ± 2.67 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each) # without out specified numba 1 term
#    115 µs ± 1.77 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each) # with out specified numba 1 term

    return lambdas


def dinteraction_exp_dT(T, N, B, C, D, E, F, lambdas, dlambdas_dT=None):
    if dlambdas_dT is None:
        dlambdas_dT = [[0.0]*N for i in range(N)] # numba: delete
#        dlambdas_dT = zeros((N, N)) # numba: uncomment

    T2 = T + T
    Tinv = 1.0/T
    nT2inv = -Tinv*Tinv
    nT3inv2 = 2.0*nT2inv*Tinv
    for i in range(N):
        lambdasi = lambdas[i]
        Bi = B[i]
        Ci = C[i]
        Di = D[i]
        Ei = E[i]
        Fi = F[i]
        dlambdas_dTi = dlambdas_dT[i]
        for j in range(N):
            dlambdas_dTi[j] = (T2*Fi[j] + Di[j] + Ci[j]*Tinv + Bi[j]*nT2inv
                             + Ei[j]*nT3inv2)*lambdasi[j]
    return dlambdas_dT

def d2interaction_exp_dT2(T, N, B, C, E, F, lambdas, dlambdas_dT, d2lambdas_dT2=None):
    if d2lambdas_dT2 is None:
        d2lambdas_dT2 = [[0.0]*N for i in range(N)] # numba: delete
#        d2lambdas_dT2 = zeros((N, N)) # numba: uncomment

    Tinv = 1.0/T
    nT2inv = -Tinv*Tinv
    T3inv2 = -2.0*nT2inv*Tinv
    T4inv6 = 3.0*T3inv2*Tinv
    for i in range(N):
        lambdasi = lambdas[i]
        dlambdas_dTi = dlambdas_dT[i]
        Bi = B[i]
        Ci = C[i]
        Ei = E[i]
        Fi = F[i]
        d2lambdas_dT2i = d2lambdas_dT2[i]
        for j in range(N):
            d2lambdas_dT2i[j] = ((2.0*Fi[j] + nT2inv*Ci[j]
                             + T3inv2*Bi[j] + T4inv6*Ei[j]
                               )*lambdasi[j] + dlambdas_dTi[j]*dlambdas_dTi[j]/lambdasi[j])
    return d2lambdas_dT2

def d3interaction_exp_dT3(T, N, B, C, E, F, lambdas, dlambdas_dT, d3lambdas_dT3=None):
    if d3lambdas_dT3 is None:
        d3lambdas_dT3 = [[0.0]*N for i in range(N)] # numba: delete
#        d3lambdas_dT3 = zeros((N, N)) # numba: uncomment

    Tinv = 1.0/T
    Tinv3 = 3.0*Tinv
    nT2inv = -Tinv*Tinv
    nT2inv05 = 0.5*nT2inv
    T3inv = -nT2inv*Tinv
    T3inv2 = T3inv+T3inv
    T4inv3 = 1.5*T3inv2*Tinv
    T2_12 = -12.0*nT2inv

    for i in range(N):
        lambdasi = lambdas[i]
        dlambdas_dTi = dlambdas_dT[i]
        Bi = B[i]
        Ci = C[i]
        Ei = E[i]
        Fi = F[i]
        d3lambdas_dT3i = d3lambdas_dT3[i]
        for j in range(N):
            term2 = (Fi[j] + nT2inv05*Ci[j] + T3inv*Bi[j] + T4inv3*Ei[j])

            term3 = dlambdas_dTi[j]/lambdasi[j]

            term4 = (T3inv2*(Ci[j] - Tinv3*Bi[j] - T2_12*Ei[j]))

            d3lambdas_dT3i[j] = ((term3*(6.0*term2 + term3*term3) + term4)*lambdasi[j])

    return d3lambdas_dT3

class GibbsExcess:
    r'''Class for representing an activity coefficient model.
    While these are typically presented as tools to compute activity
    coefficients, in truth they are excess Gibbs energy models and activity
    coefficients are just one derived aspect of them.

    This class does not implement any activity coefficient models itself; it
    must be subclassed by another model. All properties are
    derived with the CAS SymPy, not relying on any derivations previously
    published, and checked numerically for consistency.

    Different subclasses have different parameter requirements for
    initialization; :obj:`IdealSolution` is
    available as a simplest model with activity coefficients of 1 to show
    what needs to be implemented in subclasses. It is also intended subclasses
    implement the method `to_T_xs`, which creates a new object at the
    specified temperature and composition but with the same parameters.

    These objects are intended to lazy-calculate properties as much as
    possible, and for the temperature and composition of an object to be
    immutable.

    '''
    T_DEFAULT = 298.15
    _x_infinite_dilution = 0.0
    """When set, this will be the limiting mole fraction used to approximate
    the :obj:`gammas_infinite_dilution` calculation. This is important
    as not all models can mathematically be evaluated at zero mole-fraction."""


    __slots__ = ('T', 'N', 'xs', 'vectorized', '_GE', '_dGE_dT', '_SE','_d2GE_dT2', '_d2GE_dTdxs', '_dGE_dxs',
                  '_gammas', '_dgammas_dns', '_dgammas_dT', '_d2GE_dxixjs',  '_dHE_dxs', '_dSE_dxs',
                  '_model_hash')

    recalculable_attributes = ('_GE', '_dGE_dT', '_SE','_d2GE_dT2', '_d2GE_dTdxs', '_dGE_dxs',
                  '_gammas', '_dgammas_dns', '_dgammas_dT', '_d2GE_dxixjs',  '_dHE_dxs', '_dSE_dxs')

    _point_properties = ('CpE', 'GE', 'HE', 'SE', 'd2GE_dT2', 'd2GE_dTdns',
                         'd2GE_dTdxs', 'd2GE_dxixjs', 'd2nGE_dTdns', 'd2nGE_dninjs',
                         'dGE_dT', 'dGE_dns', 'dGE_dxs', 'dHE_dT', 'dHE_dns', 'dHE_dxs',
                         'dSE_dT', 'dSE_dns', 'dSE_dxs', 'dgammas_dT', 'dgammas_dns',
                         'dnGE_dns', 'dnHE_dns', 'dnSE_dns', 'gammas')
    """These are all methods which take no arguments. For use in testing."""

    def __init_subclass__(cls):
        cls.__full_path__ = f"{cls.__module__}.{cls.__qualname__}"

    json_version = 1
    obj_references = []
    non_json_attributes = ['_model_hash']

    def __repr__(self):
        r'''Method to create a string representation of the state of the model.
        Included is `T`, `xs`, and all constants necessary to create the model.
        This can be passed into :py:func:`exec` to re-create the
        model. Note that parsing strings like this can be slow.

        Returns
        -------
        repr : str
            String representation of the object, [-]

        Examples
        --------
        >>> IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
        IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
        '''
        # Other classes with different parameters should expose them here too
        s = f'{self.__class__.__name__}(T={self.T!r}, xs={self.xs!r})'
        return s

    def __eq__(self, other):
        return self.__hash__() == hash(other)

    def model_hash(self):
        r'''Basic method to calculate a hash of the non-state parts of the model
        This is useful for comparing to models to
        determine if they are the same, i.e. in a VLL flash it is important to
        know if both liquids have the same model.

        Note that the hashes should only be compared on the same system running
        in the same process!

        Returns
        -------
        model_hash : int
            Hash of the object's model parameters, [-]
        '''
        try:
            return self._model_hash
        except AttributeError:
            pass
        to_hash = [self.__class__.__name__, self.N]
        for k in self._model_attributes:
            v = getattr(self, k)
            if type(v) is ndarray:
                v = v.tolist()
            to_hash.append(v)
        self._model_hash = hash_any_primitive(to_hash)
        return self._model_hash

    def state_hash(self):
        r'''Basic method to calculate a hash of the state of the model and its
        model parameters.

        Note that the hashes should only be compared on the same system running
        in the same process!

        Returns
        -------
        state_hash : int
            Hash of the object's model parameters and state, [-]
        '''
        xs = self.xs if not self.vectorized else self.xs.tolist()
        return hash_any_primitive((self.model_hash(), float(self.T), xs))

    __hash__ = state_hash

    def exact_hash(self):
        r'''Method to calculate and return a hash representing the exact state
        of the object. This includes `T`, `xs`,
        the model class, and which values have already been calculated.

        Returns
        -------
        hash : int
            Hash of the object, [-]
        '''
        d = object_data(self)
        ans = hash_any_primitive((self.__class__.__name__, d))
        return ans

    def as_json(self, cache=None, option=0):
        r'''Method to create a JSON-friendly representation of the Gibbs Excess
        model which can be stored, and reloaded later.

        Returns
        -------
        json_repr : dict
            JSON-friendly representation, [-]

        Notes
        -----

        Examples
        --------
        >>> import json
        >>> model = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
        >>> json_view = model.as_json()
        >>> json_str = json.dumps(json_view)
        >>> assert type(json_str) is str
        >>> model_copy = IdealSolution.from_json(json.loads(json_str))
        >>> assert model_copy == model
        '''
        return JsonOptEncodable.as_json(self, cache, option)

    @classmethod
    def from_json(cls, json_repr, cache=None):
        r'''Method to create a Gibbs Excess model from a JSON-friendly
        serialization of another Gibbs Excess model.

        Parameters
        ----------
        json_repr : dict
            JSON-friendly representation, [-]

        Returns
        -------
        model : :obj:`GibbsExcess`
            Newly created object from the json serialization, [-]

        Notes
        -----
        It is important that the input string be in the same format as that
        created by :obj:`GibbsExcess.as_json`.

        Examples
        --------
        >>> model = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
        >>> json_view = model.as_json()
        >>> new_model = IdealSolution.from_json(json_view)
        >>> assert model == new_model
        '''
        return JsonOptEncodable.from_json(json_repr, cache)

    def _custom_from_json(self, *args):
        vectorized = self.vectorized
        if vectorized and hasattr(self, 'cmp_group_idx'):
            self.cmp_group_idx = tuple(array(v) for v in self.cmp_group_idx)
        if vectorized and hasattr(self, 'group_cmp_idx'):
            self.group_cmp_idx = tuple(array(v) for v in self.group_cmp_idx)

    def HE(self):
        r'''Calculate and return the excess entropy of a liquid phase using an
        activity coefficient model.

        .. math::
            h^E = -T \frac{\partial g^E}{\partial T} + g^E

        Returns
        -------
        HE : float
            Excess enthalpy of the liquid phase, [J/mol]

        Notes
        -----
        '''
        """f = symbols('f', cls=Function)
        T = symbols('T')
        simplify(-T**2*diff(f(T)/T, T))
        """
        return -self.T*self.dGE_dT() + self.GE()

    def dHE_dT(self):
        r'''Calculate and return the first temperature derivative of excess
        enthalpy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial h^E}{\partial T} = -T \frac{\partial^2 g^E}
            {\partial T^2}

        Returns
        -------
        dHE_dT : float
            First temperature derivative of excess enthalpy of the liquid
            phase, [J/mol/K]

        Notes
        -----
        '''
        return -self.T*self.d2GE_dT2()

    CpE = dHE_dT

    def dHE_dxs(self):
        r'''Calculate and return the mole fraction derivative of excess
        enthalpy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial h^E}{\partial x_i} = -T \frac{\partial^2 g^E}
            {\partial T \partial x_i} + \frac{\partial g^E}{\partial x_i}

        Returns
        -------
        dHE_dxs : list[float]
            First mole fraction derivative of excess enthalpy of the liquid
            phase, [J/mol]

        Notes
        -----
        '''
        try:
            return self._dHE_dxs
        except:
            pass
        # Derived by hand taking into account the expression for excess enthalpy
        d2GE_dTdxs = self.d2GE_dTdxs()
        try:
            dGE_dxs = self._dGE_dxs
        except:
            dGE_dxs = self.dGE_dxs()
        dHE_dxs = gibbs_excess_dHE_dxs(dGE_dxs, d2GE_dTdxs, self.N, self.T)
        if self.vectorized and type(dHE_dxs) is list:
            dHE_dxs = array(dHE_dxs)
        self._dHE_dxs = dHE_dxs
        return dHE_dxs

    def dHE_dns(self):
        r'''Calculate and return the mole number derivative of excess
        enthalpy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial h^E}{\partial n_i}

        Returns
        -------
        dHE_dns : list[float]
            First mole number derivative of excess enthalpy of the liquid
            phase, [J/mol^2]

        Notes
        -----
        '''
        out = [0.0]*self.N if not self.vectorized else zeros(self.N)
        dHE_dns = dxs_to_dns(self.dHE_dxs(), self.xs, out)
        return dHE_dns

    def dnHE_dns(self):
        r'''Calculate and return the partial mole number derivative of excess
        enthalpy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial n h^E}{\partial n_i}

        Returns
        -------
        dnHE_dns : list[float]
            First partial mole number derivative of excess enthalpy of the
            liquid phase, [J/mol]

        Notes
        -----
        '''
        out = [0.0]*self.N if not self.vectorized else zeros(self.N)
        dnHE_dns = dxs_to_dn_partials(self.dHE_dxs(), self.xs, self.HE(), out)
        return dnHE_dns

    def SE(self):
        r'''Calculates the excess entropy of a liquid phase using an
        activity coefficient model.

        .. math::
            s^E = \frac{h^E - g^E}{T}

        Returns
        -------
        SE : float
            Excess entropy of the liquid phase, [J/mol/K]

        Notes
        -----
        Note also the relationship of the expressions for partial excess
        entropy:

        .. math::
            S_i^E = -R\left(T \frac{\partial \ln \gamma_i}{\partial T}
            + \ln \gamma_i\right)


        '''
        try:
            return self._SE
        except:
            self._SE = (self.HE() - self.GE())/self.T
        return self._SE

    def dSE_dT(self):
        r'''Calculate and return the first temperature derivative of excess
        entropy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial s^E}{\partial T} = \frac{1}{T}
            \left(\frac{-\partial g^E}{\partial T} + \frac{\partial h^E}{\partial T}
            - \frac{(G + H)}{T}\right)

        Returns
        -------
        dSE_dT : float
            First temperature derivative of excess entropy of the liquid
            phase, [J/mol/K]

        Notes
        -----

        '''
        """from sympy import *
        T = symbols('T')
        G, H = symbols('G, H', cls=Function)
        S = (H(T) - G(T))/T
        print(diff(S, T))
        # (-Derivative(G(T), T) + Derivative(H(T), T))/T - (-G(T) + H(T))/T**2
        """
        # excess entropy temperature derivative
        dHE_dT = self.dHE_dT()
        try:
            HE = self._HE
        except:
            HE = self.HE()
        try:
            dGE_dT = self._dGE_dT
        except:
            dGE_dT = self.dGE_dT()
        try:
            GE = self._GE
        except:
            GE = self.GE()
        T_inv = 1.0/self.T
        return T_inv*(-dGE_dT + dHE_dT - (HE - GE)*T_inv)

    def dSE_dxs(self):
        r'''Calculate and return the mole fraction derivative of excess
        entropy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial S^E}{\partial x_i} = \frac{1}{T}\left( \frac{\partial h^E}
            {\partial x_i} - \frac{\partial g^E}{\partial x_i}\right)
            = -\frac{\partial^2 g^E}{\partial x_i \partial T}

        Returns
        -------
        dSE_dxs : list[float]
            First mole fraction derivative of excess entropy of the liquid
            phase, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._dSE_dxs
        except:
            pass
        try:
            d2GE_dTdxs = self._d2GE_dTdxs
        except:
            d2GE_dTdxs = self.d2GE_dTdxs()
        if not self.vectorized:
            dSE_dxs = [-v for v in d2GE_dTdxs]
        else:
            dSE_dxs = -d2GE_dTdxs
        self._dSE_dxs = dSE_dxs
        return dSE_dxs

    def dSE_dns(self):
        r'''Calculate and return the mole number derivative of excess
        entropy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial S^E}{\partial n_i}

        Returns
        -------
        dSE_dns : list[float]
            First mole number derivative of excess entropy of the liquid
            phase, [J/(mol^2*K)]

        Notes
        -----
        '''
        out = [0.0]*self.N if not self.vectorized else zeros(self.N)
        dSE_dns = dxs_to_dns(self.dSE_dxs(), self.xs, out)
        return dSE_dns

    def dnSE_dns(self):
        r'''Calculate and return the partial mole number derivative of excess
        entropy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial n S^E}{\partial n_i}

        Returns
        -------
        dnSE_dns : list[float]
            First partial mole number derivative of excess entropy of the liquid
            phase, [J/(mol*K)]

        Notes
        -----
        '''
        out = [0.0]*self.N if not self.vectorized else zeros(self.N)
        dnSE_dns = dxs_to_dn_partials(self.dSE_dxs(), self.xs, self.SE(), out)
        return dnSE_dns

    def dGE_dns(self):
        r'''Calculate and return the mole number derivative of excess
        Gibbs energy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial G^E}{\partial n_i}

        Returns
        -------
        dGE_dns : list[float]
            First mole number derivative of excess Gibbs entropy of the liquid
            phase, [J/(mol^2*K)]

        Notes
        -----
        '''
        out = [0.0]*self.N if not self.vectorized else zeros(self.N)
        dGE_dns = dxs_to_dns(self.dGE_dxs(), self.xs, out)
        return dGE_dns

    def dnGE_dns(self):
        r'''Calculate and return the partial mole number derivative of excess
        Gibbs energy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial n G^E}{\partial n_i}

        Returns
        -------
        dnGE_dns : list[float]
            First partial mole number derivative of excess Gibbs entropy of the
            liquid phase, [J/(mol)]

        Notes
        -----
        '''
        out = [0.0]*self.N if not self.vectorized else zeros(self.N)
        dnGE_dns = dxs_to_dn_partials(self.dGE_dxs(), self.xs, self.GE(), out)
        return dnGE_dns

    def d2GE_dTdns(self):
        r'''Calculate and return the mole number derivative of the first
        temperature derivative of excess Gibbs energy of a liquid phase using
        an activity coefficient model.

        .. math::
            \frac{\partial^2 G^E}{\partial n_i \partial T}

        Returns
        -------
        d2GE_dTdns : list[float]
            First mole number derivative of the temperature derivative of
            excess Gibbs entropy of the liquid phase, [J/(mol^2*K)]

        Notes
        -----
        '''
        out = [0.0]*self.N if not self.vectorized else zeros(self.N)
        d2GE_dTdns = dxs_to_dns(self.d2GE_dTdxs(), self.xs, out)
        return d2GE_dTdns


    def d2nGE_dTdns(self):
        r'''Calculate and return the partial mole number derivative of the first
        temperature derivative of excess Gibbs energy of a liquid phase using
        an activity coefficient model.

        .. math::
            \frac{\partial^2 n G^E}{\partial n_i \partial T}

        Returns
        -------
        d2nGE_dTdns : list[float]
            First partial mole number derivative of the temperature derivative
            of excess Gibbs entropy of the liquid phase, [J/(mol*K)]

        Notes
        -----
        '''
        # needed in gammas temperature derivatives
        dGE_dT = self.dGE_dT()
        d2GE_dTdns = self.d2GE_dTdns()
        out = [0.0]*self.N if not self.vectorized else zeros(self.N)
        d2nGE_dTdns = dns_to_dn_partials(d2GE_dTdns, dGE_dT, out)
        return d2nGE_dTdns


    def d2nGE_dninjs(self):
        r'''Calculate and return the second partial mole number derivative of
        excess Gibbs energy of a liquid phase using
        an activity coefficient model.

        .. math::
            \frac{\partial^2 n G^E}{\partial n_i \partial n_i}

        Returns
        -------
        d2nGE_dninjs : list[list[float]]
            Second partial mole number derivative of excess Gibbs energy of a
            liquid phase, [J/(mol^2)]

        Notes
        -----
        '''
        # This one worked out
        d2nGE_dninjs = d2xs_to_dxdn_partials(self.d2GE_dxixjs(), self.xs)
        if self.vectorized and type(d2nGE_dninjs) is list:
            d2nGE_dninjs = array(d2nGE_dninjs)
        return d2nGE_dninjs

    def gammas_infinite_dilution(self):
        r'''Calculate and return the infinite dilution activity coefficients
        of each component.

        Returns
        -------
        gammas_infinite : list[float]
            Infinite dilution activity coefficients, [-]

        Notes
        -----
        The algorithm is as follows. For each component, set its composition to
        zero. Normalize the remaining compositions to 1. Create a new object
        with that composition, and calculate the activity coefficient of the
        component whose concentration was set to zero.
        '''
        T, N = self.T, self.N
        xs_base = self.xs
        x_infinite_dilution = self._x_infinite_dilution
        if not self.vectorized:
            gammas_inf = [0.0]*N
            copy_fun = list
        else:
            gammas_inf = zeros(N)
            copy_fun = array
        for i in range(N):
            xs = copy_fun(xs_base)
            xs[i] = x_infinite_dilution
            xs = normalize(xs)
            gammas_inf[i] = self.to_T_xs(T, xs=xs).gammas()[i]
        return gammas_inf

    def gammas(self):
        r'''Calculate and return the activity coefficients of a liquid phase
        using an activity coefficient model.

        .. math::
            \gamma_i = \exp\left(\frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}\right)

        Returns
        -------
        gammas : list[float]
            Activity coefficients, [-]

        Notes
        -----
        '''
        try:
            return self._gammas
        except:
            pass
        # Matches the gamma formulation perfectly
        GE = self.GE()
        dG_dxs = self.dGE_dxs()
        if not self.vectorized:
            dG_dns = dxs_to_dn_partials(dG_dxs, self.xs, GE)
            RT_inv = 1.0/(R*self.T)
            gammas = [trunc_exp(i*RT_inv) for i in dG_dns]
        else:
            gammas = gibbs_excess_gammas(self.xs, dG_dxs, GE, self.T)
            if type(gammas) is list:
                gammas = array(gammas)
        self._gammas = gammas
        return gammas

    def gammas_dGE_dxs(self):
        try:
            del self._gammas
        except:
            pass
        return GibbsExcess.gammas(self)
    
    def gammas_numerical(self):
        # for testing purposes
        def nGE_func(ns):
            total_n = sum(ns)
            xs = [n / total_n for n in ns]
            return total_n * self.to_T_xs(T=self.T, xs=xs).GE()
        dnGE_dns = jacobian(nGE_func, self.xs, perturbation=1e-7)
        
        RT_inv = 1.0/(self.T *R)
        gammas = np.exp(np.array(dnGE_dns)*RT_inv) if self.vectorized else [exp(v*RT_inv) for v in dnGE_dns]
        return gammas

    def lngammas(self):
        r'''Calculate and return the natural logarithm of the activity coefficients
        of a liquid phase using an activity coefficient model.

        .. math::
            \ln \gamma_i = \frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}

        Returns
        -------
        log_gammas : list[float]
            Natural logarithm of activity coefficients, [-]

        Notes
        -----
        '''
        GE = self.GE()
        dG_dxs = self.dGE_dxs()
        dG_dns = dxs_to_dn_partials(dG_dxs, self.xs, GE)
        RT_inv = 1.0/(R * self.T)
        if not self.vectorized:
            return [dG_dn * RT_inv for dG_dn in dG_dns]
        else:
            return array(dG_dns) * RT_inv

    def dlngammas_dT(self):
        r'''Calculate and return the temperature derivatives of the natural logarithm
        of activity coefficients of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial \ln \gamma_i}{\partial T} = \frac{1}{\gamma_i} \frac{\partial \gamma_i}{\partial T}

        Returns
        -------
        dlog_gammas_dT : list[float]
            Temperature derivatives of the natural logarithm of activity coefficients, [1/K]

        Notes
        -----
        This method uses the chain rule to calculate the temperature derivative
        of log activity coefficients.
        '''
        gammas = self.gammas()
        dgammas_dT = self.dgammas_dT()
        
        if not self.vectorized:
            return [dgamma_dT / gamma for gamma, dgamma_dT in zip(gammas, dgammas_dT)]
        else:
            return dgammas_dT / gammas
    
    def dgammas_dns(self):
        r'''Calculate and return the mole number derivative of activity
        coefficients of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial \gamma_i}{\partial n_i} = \gamma_i
            \left(\frac{\frac{\partial^2 G^E}{\partial x_i \partial x_j}}{RT}\right)

        Returns
        -------
        dgammas_dns : list[list[float]]
            Mole number derivatives of activity coefficients, [1/mol]

        Notes
        -----
        '''
        try:
            return self._dgammas_dns
        except AttributeError:
            pass
        gammas = self.gammas()
        N = self.N
        xs = self.xs
        d2GE_dxixjs = self.d2GE_dxixjs()

        dgammas_dns = [[0.0]*N for _ in range(N)] if not self.vectorized else zeros((N, N))

        dgammas_dns = gibbs_excess_dgammas_dns(xs, gammas, d2GE_dxixjs, N, self.T, dgammas_dns)

        if self.vectorized and type(dgammas_dns) is list:
            dgammas_dns = array(dgammas_dns)

        self._dgammas_dns = dgammas_dns
        return dgammas_dns

#    def dgammas_dxs(self):
        # TODO - compare with UNIFAC, which has a dx derivative working
#        # NOT WORKING
#        gammas = self.gammas()
#        cmps = self.cmps
#        RT_inv = 1.0/(R*self.T)
#        d2GE_dxixjs = self.d2GE_dxixjs() # Thi smatrix is symmetric
#
#        def thing(d2xs, xs):
#            cmps = range(len(xs))
#
#            double_sums = []
#            for j in cmps:
#                tot = 0.0
#                for k in cmps:
#                    tot += xs[k]*d2xs[j][k]
#                double_sums.append(tot)
#
#            mat = []
#            for i in cmps:
#                row = []
#                for j in cmps:
#                    row.append(d2xs[i][j] - double_sums[i])
#                mat.append(row)
#            return mat
#
#            return [[d2xj - tot for (d2xj, tot) in zip(d2xsi, double_sums)]
#                     for d2xsi in d2xs]
#
#        d2nGE_dxjnis = thing(d2GE_dxixjs, self.xs)
#
#        matrix = []
#        for i in cmps:
#            row = []
#            gammai = gammas[i]
#            for j in cmps:
#                v = gammai*d2nGE_dxjnis[i][j]*RT_inv
#                row.append(v)
#            matrix.append(row)
#        return matrix


    def dgammas_dT(self):
        r'''Calculate and return the temperature derivatives of activity
        coefficients of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial \gamma_i}{\partial T} =
            \left(\frac{\frac{\partial^2 n G^E}{\partial T \partial n_i}}{RT} -
            \frac{{\frac{\partial n_i G^E}{\partial n_i }}}{RT^2}\right)
             \exp\left(\frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}\right)

        Returns
        -------
        dgammas_dT : list[float]
            Temperature derivatives of activity coefficients, [1/K]

        Notes
        -----
        '''
        r"""
        from sympy import *
        R, T = symbols('R, T')
        f = symbols('f', cls=Function)
        diff(exp(f(T)/(R*T)), T)
        """
        try:
            return self._dgammas_dT
        except AttributeError:
            pass
        N, T, xs = self.N, self.T, self.xs
        dGE_dT = self.dGE_dT()
        GE = self.GE()
        dG_dxs = self.dGE_dxs()
        d2GE_dTdxs = self.d2GE_dTdxs()
        dgammas_dT = gibbs_excess_dgammas_dT(xs, GE, dGE_dT, dG_dxs, d2GE_dTdxs, N, T)
        if self.vectorized and type(dgammas_dT) is list:
            dgammas_dT = array(dgammas_dT)
        self._dgammas_dT = dgammas_dT
        return dgammas_dT

    @classmethod
    def _regress_binary_parameters(cls, gammas, xs, fitting_func, fit_parameters,
                                   use_fit_parameters, initial_guesses=None, analytical_jac=None,
                                   **kwargs):

        fit_kwargs = dict(fit_method='lm',
                    # fit_method='differential_evolution',
                   objective='MeanSquareErr', multiple_tries_max_objective='MeanRelErr',
                   initial_guesses=initial_guesses, analytical_jac=analytical_jac,
                   solver_kwargs=None, use_numba=False, multiple_tries=False,
                   do_statistics=True, multiple_tries_max_err=1e-5)
        fit_kwargs.update(kwargs)


        res = fit_customized(xs, data=gammas, fitting_func=fitting_func, fit_parameters=fit_parameters, use_fit_parameters=use_fit_parameters,
                    **fit_kwargs)
        return res


derivatives_added = [('dGE_dT', 'GE', 1),
 ('d2GE_dT2', 'GE', 2),
 ('d3GE_dT3', 'GE', 3),
 ('d4GE_dT4', 'GE', 4),
]
for create_derivative, derive_attr, order in derivatives_added:
    def numerical_derivative(self, derive_attr=derive_attr, n=order, ):
        order = 2*n+1
        perturbation = 1e-7
        xs = self.xs
        def func(T):
            if T == self.T:
                obj = self
            else:
                obj = self.to_T_xs(xs=xs, T=T)
            return getattr(obj, derive_attr)()
        return derivative(func, x0=self.T, dx=self.T*perturbation, lower_limit=0.0, n=n, order=order)
    setattr(GibbsExcess, create_derivative+'_numerical', numerical_derivative)

first_comp_derivatives = [
    ('dGE_dxs', 'GE'),
    ('d2GE_dTdxs', 'dGE_dT'),
    ('d3GE_dT2dxs', 'd2GE_dT2'),
    ('d4GE_dT3dxs', 'd3GE_dT3'),
]
for create_derivative, derive_attr in first_comp_derivatives:
    def numerical_derivative(self, derive_attr=derive_attr):
        perturbation = 1e-7
        def func(xs):
            if not self.vectorized and xs == self.xs:
                obj = self
            else:
                obj = self.to_T_xs(xs=xs, T=self.T)
            return getattr(obj, derive_attr)()
        return jacobian(func, self.xs, perturbation=perturbation)
    setattr(GibbsExcess, create_derivative+'_numerical', numerical_derivative)

second_comp_derivatives = [
    ('d2GE_dxixjs', 'GE'),
    ('d3GE_dTdxixjs', 'dGE_dT'),
    ('d4GE_dT2dxixjs', 'd2GE_dT2'),
    ('d5GE_dT3dxixjs', 'd3GE_dT3'),
]
for create_derivative, derive_attr in second_comp_derivatives:
    def numerical_derivative(self, derive_attr=derive_attr):
        perturbation = 1e-5
        def func(xs):
            if not self.vectorized and xs == self.xs:
                obj = self
            else:
                obj = self.to_T_xs(xs=xs, T=self.T)
            return getattr(obj, derive_attr)()
        return hessian(func, self.xs, perturbation=perturbation)
    setattr(GibbsExcess, create_derivative+'_numerical', numerical_derivative)

class IdealSolution(GibbsExcess):
    r'''Class for  representing an ideal liquid, with no excess gibbs energy
    and thus activity coefficients of 1.

    Parameters
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]

    Attributes
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]

    Examples
    --------
    >>> model = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
    >>> model.GE()
    0.0
    >>> model.gammas()
    [1.0, 1.0, 1.0, 1.0]
    >>> model.dgammas_dT()
    [0.0, 0.0, 0.0, 0.0]
    '''

    _model_attributes = ()

    model_id = 0
    __slots__ = GibbsExcess.__slots__

    def gammas_args(self, T=None):
        N = self.N
        return (N,)

    def __init__(self, *, xs, T=GibbsExcess.T_DEFAULT):
        self.T = T
        self.xs = xs
        self.N = len(xs)
        self.vectorized = type(xs) is not list

    def to_T_xs(self, T, xs):
        r'''Method to construct a new :obj:`IdealSolution` instance at
        temperature `T`, and mole fractions `xs`
        with the same parameters as the existing object.

        Parameters
        ----------
        T : float
            Temperature, [K]
        xs : list[float]
            Mole fractions of each component, [-]

        Returns
        -------
        obj : IdealSolution
            New :obj:`IdealSolution` object at the specified conditions [-]

        Notes
        -----

        Examples
        --------
        >>> p = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
        >>> p.to_T_xs(T=500.0, xs=[.25, .25, .25, .25])
        IdealSolution(T=500.0, xs=[0.25, 0.25, 0.25, 0.25])
        '''
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.vectorized = self.vectorized
        new.N = len(xs)
        return new

    def GE(self):
        r'''Calculate and return the excess Gibbs energy of a liquid phase
        using an activity coefficient model.

        .. math::
            g^E = 0

        Returns
        -------
        GE : float
            Excess Gibbs energy of an ideal liquid, [J/mol]

        Notes
        -----
        '''
        return 0.0

    def dGE_dT(self):
        r'''Calculate and return the temperature derivative of excess Gibbs
        energy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial g^E}{\partial T} = 0

        Returns
        -------
        dGE_dT : float
            First temperature derivative of excess Gibbs energy of an
            ideal liquid, [J/(mol*K)]

        Notes
        -----
        '''
        return 0.0

    def d2GE_dT2(self):
        r'''Calculate and return the second temperature derivative of excess
        Gibbs energy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial^2 g^E}{\partial T^2} = 0

        Returns
        -------
        d2GE_dT2 : float
            Second temperature derivative of excess Gibbs energy of an
            ideal liquid, [J/(mol*K^2)]

        Notes
        -----
        '''
        return 0.0

    def d3GE_dT3(self):
        r'''Calculate and return the third temperature derivative of excess
        Gibbs energy of a liquid phase using an activity coefficient model.

        .. math::
            \frac{\partial^3 g^E}{\partial T^3} = 0

        Returns
        -------
        d3GE_dT3 : float
            Third temperature derivative of excess Gibbs energy of an ideal
            liquid, [J/(mol*K^3)]

        Notes
        -----
        '''
        return 0.0

    def d2GE_dTdxs(self):
        r'''Calculate and return the temperature derivative of mole fraction
        derivatives of excess Gibbs energy of an ideal liquid.

        .. math::
            \frac{\partial^2 g^E}{\partial x_i \partial T} = 0

        Returns
        -------
        d2GE_dTdxs : list[float]
            Temperature derivative of mole fraction derivatives of excess Gibbs
            energy of an ideal liquid, [J/(mol*K)]

        Notes
        -----
        '''
        if not self.vectorized:
            return [0.0]*self.N
        return zeros(self.N)

    def dGE_dxs(self):
        r'''Calculate and return the mole fraction derivatives of excess Gibbs
        energy of an ideal liquid.

        .. math::
            \frac{\partial g^E}{\partial x_i} = 0

        Returns
        -------
        dGE_dxs : list[float]
            Mole fraction derivatives of excess Gibbs energy of an ideal
            liquid, [J/mol]

        Notes
        -----
        '''
        if not self.vectorized:
            return [0.0]*self.N
        return zeros(self.N)

    def d2GE_dxixjs(self):
        r'''Calculate and return the second mole fraction derivatives of excess
        Gibbs energy of an ideal liquid.

        .. math::
            \frac{\partial^2 g^E}{\partial x_i \partial x_j} = 0

        Returns
        -------
        d2GE_dxixjs : list[list[float]]
            Second mole fraction derivatives of excess Gibbs energy of an ideal
            liquid, [J/mol]

        Notes
        -----
        '''
        N = self.N
        if not self.vectorized:
            return [[0.0]*N for i in range(self.N)]
        return zeros((N, N))

    def d3GE_dxixjxks(self):
        r'''Calculate and return the third mole fraction derivatives of excess
        Gibbs energy of an ideal liquid.

        .. math::
            \frac{\partial^3 g^E}{\partial x_i \partial x_j \partial x_k} = 0

        Returns
        -------
        d3GE_dxixjxks : list[list[list[float]]]
            Third mole fraction derivatives of excess Gibbs energy of an ideal
            liquid, [J/mol]

        Notes
        -----
        '''
        N = self.N
        if not self.vectorized:
            return [[[0.0]*N for i in range(N)] for j in range(N)]
        return zeros((N, N, N))

    def gammas(self):
        if not self.vectorized:
            return [1.0]*self.N
        else:
            return ones(self.N)

    try:
        gammas.__doc__ = GibbsExcess.__doc__
    except:
        pass

# -*- coding: utf-8 -*-
# pylint: disable=E1101
r'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains implementations of the calculation of pure-component
EOS :math:`a \alpha` parameters in a vectorized way. Functions for calculating their
temperature derivatives as may be necessary are included as well.

For certain alpha functions, a class is available to provide these functions to
and class that inherits from it.

A mixing rule must be used on the `a_alphas` to get the overall `a_alpha`
term.

.. contents:: :local:

Vectorized Alpha Functions
--------------------------
.. autofunction:: thermo.eos_alpha_functions.PR_a_alphas_vectorized
.. autofunction:: thermo.eos_alpha_functions.SRK_a_alphas_vectorized
.. autofunction:: thermo.eos_alpha_functions.PRSV_a_alphas_vectorized
.. autofunction:: thermo.eos_alpha_functions.PRSV2_a_alphas_vectorized
.. autofunction:: thermo.eos_alpha_functions.APISRK_a_alphas_vectorized
.. autofunction:: thermo.eos_alpha_functions.RK_a_alphas_vectorized

Vectorized Alpha Functions With Derivatives
-------------------------------------------
.. autofunction:: thermo.eos_alpha_functions.PR_a_alpha_and_derivatives_vectorized
.. autofunction:: thermo.eos_alpha_functions.SRK_a_alpha_and_derivatives_vectorized
.. autofunction:: thermo.eos_alpha_functions.PRSV_a_alpha_and_derivatives_vectorized
.. autofunction:: thermo.eos_alpha_functions.PRSV2_a_alpha_and_derivatives_vectorized
.. autofunction:: thermo.eos_alpha_functions.APISRK_a_alpha_and_derivatives_vectorized
.. autofunction:: thermo.eos_alpha_functions.RK_a_alpha_and_derivatives_vectorized

Class With Alpha Functions
--------------------------
The class-based ones van save a little code when implementing a new EOS.
If there is not a standalone function available for an alpha function, it has
not yet been accelerated in a nice vectorized way.

.. autoclass:: thermo.eos_alpha_functions.a_alpha_base
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Almeida_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Androulakis_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Chen_Yang_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Coquelet_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Gasem_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Gibbons_Laughton_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Haghtalab_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Harmens_Knapp_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Heyen_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Mathias_1983_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Mathias_Copeman_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Mathias_Copeman_untruncated_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Melhem_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Poly_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Saffari_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Schwartzentruber_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Soave_1972_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Soave_1984_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Soave_79_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Soave_93_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Trebble_Bishnoi_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Twu91_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.TwuPR95_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.TwuSRK95_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:
.. autoclass:: thermo.eos_alpha_functions.Yu_Lu_a_alpha
    :members:
    :undoc-members:
    :show-inheritance:

'''

from __future__ import division, print_function

__all__ = [
           'PR_a_alphas_vectorized', 'PR_a_alpha_and_derivatives_vectorized',
           'RK_a_alphas_vectorized', 'RK_a_alpha_and_derivatives_vectorized',
           'SRK_a_alphas_vectorized', 'SRK_a_alpha_and_derivatives_vectorized',
           'PRSV_a_alphas_vectorized', 'PRSV_a_alpha_and_derivatives_vectorized',
           'PRSV2_a_alphas_vectorized', 'PRSV2_a_alpha_and_derivatives_vectorized',
           'APISRK_a_alphas_vectorized', 'APISRK_a_alpha_and_derivatives_vectorized',

'a_alpha_base', 'Poly_a_alpha', 'Soave_1972_a_alpha', 'Heyen_a_alpha',
'Harmens_Knapp_a_alpha', 'Mathias_1983_a_alpha', 'Mathias_Copeman_untruncated_a_alpha',
 'Mathias_Copeman_a_alpha', 'Gibbons_Laughton_a_alpha', 'Soave_1984_a_alpha',
 'Yu_Lu_a_alpha', 'Trebble_Bishnoi_a_alpha', 'Melhem_a_alpha', 'Androulakis_a_alpha',
 'Schwartzentruber_a_alpha', 'Almeida_a_alpha', 'Twu91_a_alpha', 'Soave_93_a_alpha',
 'Gasem_a_alpha', 'Coquelet_a_alpha', 'Haghtalab_a_alpha', 'Saffari_a_alpha',
 'Chen_Yang_a_alpha', 'TwuSRK95_a_alpha', 'TwuPR95_a_alpha', 'Soave_79_a_alpha']


from fluids.numerics import (horner, horner_and_der2, numpy as np)
from chemicals.utils import log, exp, sqrt, copysign

try:
    array = np.array
except:
    pass

def PR_a_alphas_vectorized(T, Tcs, ais, kappas, a_alphas=None):
    r'''Calculates the `a_alpha` terms for the Peng-Robinson equation of state
    given the critical temperatures `Tcs`, constants `ais`, and
    `kappas`.

    .. math::
        a_i\alpha(T)_i=a_i [1+\kappa_i(1-\sqrt{T_{r,i}})]^2

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}`, [Pa*m^6/mol^2]
    kappas : list[float]
        `kappa` parameters of Peng-Robinson EOS; formulas vary, but
        the original form uses
        :math:`\kappa_i=0.37464+1.54226\omega_i-0.26992\omega^2_i`, [-]
    a_alphas : list[float], optional
        Vector for pure component `a_alpha` terms in the cubic EOS to be
        calculated and stored in, [Pa*m^6/mol^2]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]

    Notes
    -----

    Examples
    --------
    >>> Tcs = [469.7, 507.4, 540.3]
    >>> ais = [2.0698956357716662, 2.7018068455659545, 3.3725793885832323]
    >>> kappas = [0.74192743008, 0.819919992, 0.8800122140799999]
    >>> PR_a_alphas_vectorized(322.29, Tcs=Tcs, ais=ais, kappas=kappas)
    [2.6306811679, 3.6761503348, 4.8593286234]
    '''
    N = len(Tcs)
    x0_inv = 1.0/sqrt(T)
    x0 = T*x0_inv
    if a_alphas is None:
        a_alphas = [0.0]*N
    for i in range(N):
        x1 = 1.0/sqrt(Tcs[i])
        x2 = kappas[i]*(x0*x1 - 1.) - 1.
        a_alphas[i] = ais[i]*x2*x2
    return a_alphas

def PR_a_alpha_and_derivatives_vectorized(T, Tcs, ais, kappas, a_alphas=None,
                                          da_alpha_dTs=None, d2a_alpha_dT2s=None):
    r'''Calculates the `a_alpha` terms and their first two temperature
    derivatives for the Peng-Robinson equation of state
    given the critical temperatures `Tcs`, constants `ais`, and
    `kappas`.

    .. math::
        a_i\alpha(T)_i=a_i[1+\kappa_i(1-\sqrt{T_{r,i}})]^2

    .. math::
        \frac{d a_i\alpha_i}{dT} = - \frac{a_i \kappa_i}{T^{0.5} {T_c}_i^{0.5}}
        \left(\kappa_i \left(- \frac{T^{0.5}}{{T_c}_i^{0.5}} + 1\right) + 1\right)

    .. math::
        \frac{d^2 a_i\alpha_i}{dT^2} = 0.5 a_i \kappa_i \left(- \frac{1}{T^{1.5}
        {T_c}_i^{0.5}} \left(\kappa_i \left(\frac{T^{0.5}}{{T_c}_i^{0.5}} - 1\right)
        - 1\right) + \frac{\kappa_i}{T {T_c}_i}\right)

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}` [Pa*m^6/mol^2]
    kappas : list[float]
        `kappa` parameters of Peng-Robinson EOS; formulas vary, but
        the original form uses
        :math:`\kappa_i=0.37464+1.54226\omega_i-0.26992\omega^2_i`, [-]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]
    da_alpha_dTs : list[float]
        First temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K)]
    d2a_alpha_dT2s : list[float]
        Second temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K^2)]

    Notes
    -----

    Examples
    --------
    >>> Tcs = [469.7, 507.4, 540.3]
    >>> ais = [2.0698956357716662, 2.7018068455659545, 3.3725793885832323]
    >>> kappas = [0.74192743008, 0.819919992, 0.8800122140799999]
    >>> PR_a_alpha_and_derivatives_vectorized(322.29, Tcs=Tcs, ais=ais, kappas=kappas)
    ([2.63068116797, 3.67615033489, 4.859328623453], [-0.0044497546430, -0.00638993749167, -0.0085372308846], [1.066668360e-05, 1.546687574587e-05, 2.07440632117e-05])
    '''
    N = len(Tcs)
    x0_inv = 1.0/sqrt(T)
    x0 = T*x0_inv
    T_inv = x0_inv*x0_inv
    x0T_inv = x0_inv*T_inv
    x5, x6 = 0.5*T_inv, 0.5*x0T_inv
    
    if a_alphas is None:
        a_alphas = [0.0]*N
    if da_alpha_dTs is None:
        da_alpha_dTs = [0.0]*N
    if d2a_alpha_dT2s is None:
        d2a_alpha_dT2s = [0.0]*N
    for i in range(N):
        x1 = 1.0/sqrt(Tcs[i])
        x2 = kappas[i]*(x0*x1 - 1.) - 1.
        x3 = ais[i]*kappas[i]
        x4 = x1*x2
        a_alphas[i] = ais[i]*x2*x2
        da_alpha_dTs[i] = x4*x3*x0_inv
        d2a_alpha_dT2s[i] = x3*(x5*x1*x1*kappas[i] - x4*x6)
    return a_alphas, da_alpha_dTs, d2a_alpha_dT2s

def SRK_a_alphas_vectorized(T, Tcs, ais, ms, a_alphas=None):
    r'''Calculates the `a_alpha` terms for the SRK equation of state
    given the critical temperatures `Tcs`, constants `ais`, and
    `kappas`.

    .. math::
        a_i\alpha(T)_i = \left[1 + m_i\left(1 - \sqrt{\frac{T}{T_{c,i}}}
        \right)\right]^2

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}`, [Pa*m^6/mol^2]
    ms : list[float]
        `m` parameters of SRK EOS; formulas vary, but
        the original form uses
        :math:`m_i = 0.480 + 1.574\omega_i - 0.176\omega_i^2`, [-]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]

    Notes
    -----

    Examples
    --------
    >>> Tcs = [469.7, 507.4, 540.3]
    >>> ais = [1.9351940385541342, 2.525982668162287, 3.1531036708059315]
    >>> ms = [0.8610138239999999, 0.9436976, 1.007889024]
    >>> SRK_a_alphas_vectorized(322.29, Tcs=Tcs, ais=ais, ms=ms)
    [2.549485814512, 3.586598245260, 4.76614806648]
    '''
    sqrtT = sqrt(T)
    N = len(Tcs)
    if a_alphas is None:
        a_alphas = [0.0]*N
    for i in range(N):
        x0 = ms[i]*(1. - sqrtT/sqrt(Tcs[i])) + 1.0
        a_alphas[i] = ais[i]*x0*x0
    return a_alphas

def SRK_a_alpha_and_derivatives_vectorized(T, Tcs, ais, ms, a_alphas=None,
                                           da_alpha_dTs=None, 
                                           d2a_alpha_dT2s=None):
    r'''Calculates the `a_alpha` terms and their first and second temperature
    derivatives for the SRK equation of state
    given the critical temperatures `Tcs`, constants `ais`, and
    `kappas`.

    .. math::
        a_i\alpha(T)_i = \left[1 + m_i\left(1 - \sqrt{\frac{T}{T_{c,i}}}
        \right)\right]^2

    .. math::
        \frac{d a_i\alpha_i}{dT} = \frac{a_i m_i}{T} \sqrt{\frac{T}{T_{c,i}}}
         \left(m_i \left(\sqrt{\frac{T}{T{c,i}}} - 1\right) - 1\right)

    .. math::
        \frac{d^2 a_i\alpha_i}{dT^2} = \frac{a_i m_i \sqrt{\frac{T}{T_{c,i}}}}
        {2 T^{2}} \left(m_i + 1\right)

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}`, [Pa*m^6/mol^2]
    ms : list[float]
        `m` parameters of SRK EOS; formulas vary, but
        the original form uses
        :math:`m_i = 0.480 + 1.574\omega_i - 0.176\omega_i^2`, [-]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]
    da_alpha_dTs : list[float]
        First temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K)]
    d2a_alpha_dT2s : list[float]
        Second temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K^2)]

    Notes
    -----

    Examples
    --------
    >>> Tcs = [469.7, 507.4, 540.3]
    >>> ais = [1.9351940385541342, 2.525982668162287, 3.1531036708059315]
    >>> ms = [0.8610138239999999, 0.9436976, 1.007889024]
    >>> SRK_a_alpha_and_derivatives_vectorized(322.29, Tcs=Tcs, ais=ais, ms=ms)
    ([2.549485814512, 3.586598245260, 4.76614806648], [-0.004915469296196, -0.00702410108423, -0.00936320876945], [1.236441916324e-05, 1.77752796719e-05, 2.37231823137e-05])
    '''
    N = len(Tcs)
    sqrtnT = 1.0/sqrt(T)
    sqrtT = T*sqrtnT
    T_inv = sqrtnT*sqrtnT
    x10 = 0.5*T_inv*T_inv
    nT_inv = -T_inv
    if a_alphas is None:
        a_alphas = [0.0]*N
    if da_alpha_dTs is None:
        da_alpha_dTs = [0.0]*N
    if d2a_alpha_dT2s is None:
        d2a_alpha_dT2s = [0.0]*N

    for i in range(N):
        x1 = sqrtT/sqrt(Tcs[i])
        x2 = ais[i]*ms[i]*x1
        x3 = ms[i]*(1.0 - x1) + 1.

        a_alphas[i] = ais[i]*x3*x3
        da_alpha_dTs[i] = x2*nT_inv*x3
        d2a_alpha_dT2s[i] = x2*x10*(ms[i] + 1.)
    return a_alphas, da_alpha_dTs, d2a_alpha_dT2s

def RK_a_alphas_vectorized(T, Tcs, ais, a_alphas=None):
    r'''Calculates the `a_alpha` terms for the RK equation of state
    given the critical temperatures `Tcs`, and `a` parameters `ais`.

    .. math::
         a_i\alpha_i = \frac{a_i}{\sqrt{\frac{T}{T_{c,i}}}}

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}`, [Pa*m^6/mol^2]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]

    Notes
    -----

    Examples
    --------
    >>> Tcs = [469.7, 507.4, 540.3]
    >>> ais = [1.9351940385541342, 2.525982668162287, 3.1531036708059315]
    >>> RK_a_alphas_vectorized(322.29, Tcs=Tcs, ais=ais)
    [2.3362073307, 3.16943743055, 4.0825575798]
    '''
    N = len(ais)
    if a_alphas is None:
        a_alphas = [0.0]*N
    T_root_inv = 1.0/sqrt(T)
    for i in range(N):
        a_alphas[i] = ais[i]*sqrt(Tcs[i])*T_root_inv
    return a_alphas

def RK_a_alpha_and_derivatives_vectorized(T, Tcs, ais, a_alphas=None,
                                          da_alpha_dTs=None, d2a_alpha_dT2s=None):
    r'''Calculates the `a_alpha` terms and their first and second temperature
    derivatives for the RK equation of state
    given the critical temperatures `Tcs`, and `a` parameters `ais`.

    .. math::
         a_i\alpha_i = \frac{a_i}{\sqrt{\frac{T}{T_{c,i}}}}

    .. math::
        \frac{d a_i\alpha_i}{dT} = - \frac{a_i}{2 T\sqrt{\frac{T}{T_{c,i}}}}

    .. math::
        \frac{d^2 a_i\alpha_i}{dT^2} = \frac{3 a_i}{4 T^{2}\sqrt{\frac{T}{T_{c,i}}}}

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}`, [Pa*m^6/mol^2]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]
    da_alpha_dTs : list[float]
        First temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K)]
    d2a_alpha_dT2s : list[float]
        Second temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K^2)]

    Notes
    -----

    Examples
    --------
    >>> Tcs = [469.7, 507.4, 540.3]
    >>> ais = [1.9351940385541342, 2.525982668162287, 3.1531036708059315]
    >>> RK_a_alpha_and_derivatives_vectorized(322.29, Tcs=Tcs, ais=ais)
    ([2.3362073307, 3.16943743055, 4.08255757984], [-0.00362438693525, -0.0049170582868, -0.00633367088622], [1.6868597855e-05, 2.28849403652e-05, 2.94781294155e-05])
    '''
    N = len(ais)
    if a_alphas is None:
        a_alphas = [0.0]*N
    if da_alpha_dTs is None:
        da_alpha_dTs = [0.0]*N
    if d2a_alpha_dT2s is None:
        d2a_alpha_dT2s = [0.0]*N
    T_root_inv = 1.0/sqrt(T)
    T_inv = T_root_inv*T_root_inv
    T_15_inv = T_inv*T_root_inv
    T_25_inv = T_inv*T_15_inv
    x0 = -0.5*T_15_inv
    x1 = 0.75*T_25_inv

    for i in range(N):
        Tc_05 = sqrt(Tcs[i])
        aiTc_05 = ais[i]*Tc_05
        a_alphas[i] = aiTc_05*T_root_inv
        da_alpha_dTs[i] = aiTc_05*x0
        d2a_alpha_dT2s[i] = aiTc_05*x1
    return a_alphas, da_alpha_dTs, d2a_alpha_dT2s

def PRSV_a_alphas_vectorized(T, Tcs, ais, kappa0s, kappa1s, a_alphas=None):
    r'''Calculates the `a_alpha` terms for the Peng-Robinson-Stryjek-Vera
    equation of state given the critical temperatures `Tcs`, constants `ais`, PRSV
    parameters `kappa0s` and `kappa1s`.

    .. math::
        a_i\alpha_i = a_i \left(\left(\kappa_{0} + \kappa_{1} \left(\sqrt{\frac{
        T}{T_{c,i}}} + 1\right) \left(- \frac{T}{T_{c,i}} + \frac{7}{10}\right)
        \right) \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right) + 1\right)^{2}

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}`, [Pa*m^6/mol^2]
    kappa0s : list[float]
        `kappa0` parameters of PRSV EOS;
        the original form uses
        :math:`\kappa_{0,i} = 0.378893 + 1.4897153\omega_i - 0.17131848\omega_i^2 + 0.0196554\omega_i^3`, [-]
    kappa1s : list[float]
        Fit parameters, can be set to 0 if unknown [-]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]

    Notes
    -----

    Examples
    --------
    >>> Tcs = [507.6]
    >>> ais = [2.6923169620277805]
    >>> kappa0s = [0.8074380841890093]
    >>> kappa1s = [0.05104]
    >>> PRSV_a_alphas_vectorized(299.0, Tcs=Tcs, ais=ais, kappa0s=kappa0s, kappa1s=kappa1s)
    [3.81298569831]
    '''
    sqrtT = sqrt(T)
    N = len(Tcs)
    if a_alphas is None:
        a_alphas = [0.0]*N
    for i in range(N):
        Tc_inv_root = 1.0/sqrt(Tcs[i])
        Tc_inv = Tc_inv_root*Tc_inv_root
        x0 = Tc_inv_root*sqrtT
        x2 = (1.0 + (kappa0s[i] + kappa1s[i]*(x0 + 1.0)*(0.7 - T*Tc_inv))*(1.0 - x0))
        a_alphas[i] = ais[i]*x2*x2
    return a_alphas

def PRSV_a_alpha_and_derivatives_vectorized(T, Tcs, ais, kappa0s, kappa1s,
                                            a_alphas=None, da_alpha_dTs=None,
                                            d2a_alpha_dT2s=None):
    r'''Calculates the `a_alpha` terms and their first and second derivative
    for the Peng-Robinson-Stryjek-Vera
    equation of state given the critical temperatures `Tcs`, constants `ais`, PRSV
    parameters `kappa0s` and `kappa1s`.

    .. math::
        a_i\alpha_i = a_i \left(\left(\kappa_{0} + \kappa_{1} \left(\sqrt{\frac{
        T}{T_{c,i}}} + 1\right) \left(- \frac{T}{T_{c,i}} + \frac{7}{10}\right)
        \right) \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right) + 1\right)^{2}

    .. math::
        \frac{d a_i\alpha_i}{dT} =a_{i} \left(\left(1 - \sqrt{\frac{T}{T_{c,i}}}
        \right) \left(\kappa_{0,i} + \kappa_{1,i} \left(\sqrt{\frac{T}{T_{c,i}}}
        + 1\right) \left(- \frac{T}{T_{c,i}} + \frac{7}{10}\right)\right)
        + 1\right) \left(2 \left(1 - \sqrt{\frac{T}{T_{c,i}}}\right) \left(
        - \frac{\kappa_{1,i} \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right)}{T_{c,i}}
        + \frac{\kappa_{1,i} \sqrt{\frac{T}{T_{c,i}}} \left(- \frac{T}{T_{c,i}}
        + \frac{7}{10}\right)}{2 T}\right) - \frac{\sqrt{\frac{T}{T_{c,i}}}
        \left(\kappa_{0,i} + \kappa_{1,i} \left(\sqrt{\frac{T}{T_{c,i}}}
        + 1\right) \left(- \frac{T}{T_{c,i}} + \frac{7}{10}\right)\right)}{T}
        \right)

    .. math::
        \frac{d^2 a_i\alpha_i}{dT^2} = \frac{a_{i} \left(\left(\kappa_{1,i}
        \left(\sqrt{\frac{T}{T_{c,i}}} - 1\right) \left(\frac{20 \left(\sqrt{
        \frac{T}{T_{c,i}}} + 1\right)}{T_{c,i}} + \frac{\sqrt{\frac{T}{T_{c,i}}}
        \left(\frac{10 T}{T_{c,i}} - 7\right)}{T}\right) - \frac{\sqrt{\frac{T}
        {T_{c,i}}} \left(10 \kappa_{0,i} - \kappa_{1,i} \left(\sqrt{\frac{T}
        {T_{c,i}}} + 1\right) \left(\frac{10 T}{T_{c,i}} - 7\right)\right)}{T}
        \right)^{2} - \frac{\sqrt{\frac{T}{T_{c,i}}} \left(\left(10 \kappa_{0,i}
        - \kappa_{1,i} \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right)
        \left(\frac{10 T}{T_{c,i}} - 7\right)\right) \left(\sqrt{\frac{T}
        {T_{c,i}}} - 1\right) - 10\right) \left(\kappa_{1,i} \left(\frac{40}
        {T_{c,i}} - \frac{\frac{10 T}{T_{c,i}} - 7}{T}\right) \left(\sqrt{
        \frac{T}{T_{c,i}}} - 1\right) + 2 \kappa_{1,i} \left(\frac{20 \left(
        \sqrt{\frac{T}{T_{c,i}}} + 1\right)}{T_{c,i}} + \frac{\sqrt{\frac{T}
        {T_{c,i}}} \left(\frac{10 T}{T_{c,i}} - 7\right)}{T}\right) + \frac{10
        \kappa_{0,i} - \kappa_{1,i} \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right)
        \left(\frac{10 T}{T_{c,i}} - 7\right)}{T}\right)}{T}\right)}{200}

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}`, [Pa*m^6/mol^2]
    kappa0s : list[float]
        `kappa0` parameters of PRSV EOS; the original form uses
        :math:`\kappa_{0,i} = 0.378893 + 1.4897153\omega_i - 0.17131848\omega_i^2 + 0.0196554\omega_i^3`, [-]
    kappa1s : list[float]
        Fit parameters, can be set to 0 if unknown [-]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]
    da_alpha_dTs : list[float]
        First temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K)]
    d2a_alpha_dT2s : list[float]
        Second temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K^2)]

    Notes
    -----

    Examples
    --------
    >>> Tcs = [507.6]
    >>> ais = [2.6923169620277805]
    >>> kappa0s = [0.8074380841890093]
    >>> kappa1s = [0.05104]
    >>> PRSV_a_alpha_and_derivatives_vectorized(299.0, Tcs=Tcs, ais=ais, kappa0s=kappa0s, kappa1s=kappa1s)
    ([3.8129856983], [-0.0069769034748], [2.00265608110e-05])
    '''
    r'''
    Formula derived with:
    from sympy import *
    Tc = symbols('T_{c\,i}')
    T, a, kappa0, kappa1 = symbols('T, a_i, \kappa_{0\,i}, \kappa_{1\,i}')
    kappa = kappa0 + kappa1*(1 + sqrt(T/Tc))*(Rational(7, 10)-T/Tc)
    a_alpha = a*(1 + kappa*(1-sqrt(T/Tc)))**2
    diff(a_alpha, T, 2)
    '''
    sqrtT = sqrt(T)
    T_inv = 1.0/T
    N = len(Tcs)
    if a_alphas is None:
        a_alphas = [0.0]*N
    if da_alpha_dTs is None:
        da_alpha_dTs = [0.0]*N
    if d2a_alpha_dT2s is None:
        d2a_alpha_dT2s = [0.0]*N
    for i in range(N):
        Tc_inv_root = 1.0/sqrt(Tcs[i])
        Tc_inv = Tc_inv_root*Tc_inv_root

        x1 = T*Tc_inv
        x2 = sqrtT*Tc_inv_root

        x3 = x2 - 1.
        x4 = 10.*x1 - 7.
        x5 = x2 + 1.
        x6 = 10.*kappa0s[i] - kappa1s[i]*x4*x5
        x7 = x3*x6
        x8 = x7*0.1 - 1.
        x10 = x6*T_inv
        x11 = kappa1s[i]*x3
        x12 = x4*T_inv
        x13 = 20.*Tc_inv*x5 + x12*x2
        x14 = -x10*x2 + x11*x13
        a_alpha = ais[i]*x8*x8
        da_alpha_dT = -ais[i]*x14*x8*0.1
        d2a_alpha_dT2 = ais[i]*0.005*(x14*x14 - x2*T_inv*(x7 - 10.)*(2.*kappa1s[i]*x13 + x10 + x11*(40.*Tc_inv - x12)))

        a_alphas[i] = a_alpha
        da_alpha_dTs[i] = da_alpha_dT
        d2a_alpha_dT2s[i] = d2a_alpha_dT2
    return a_alphas, da_alpha_dTs, d2a_alpha_dT2s

def PRSV2_a_alphas_vectorized(T, Tcs, ais, kappa0s, kappa1s, kappa2s, kappa3s,
                              a_alphas=None):
    r'''Calculates the `a_alpha` terms for the Peng-Robinson-Stryjek-Vera 2
    equation of state given the critical temperatures `Tcs`, constants `ais`,
    PRSV2 parameters `kappa0s, `kappa1s`, `kappa2s`, and `kappa3s`.

    .. math::
        a_i\alpha_i = a_{i} \left(\left(1 - \sqrt{\frac{T}{T_{c,i}}}\right)
        \left(\kappa_{0,i} + \left(\kappa_{1,i} + \kappa_{2,i} \left(1
        - \sqrt{\frac{T}{T_{c,i}}}\right) \left(- \frac{T}{T_{c,i}}
        + \kappa_{3,i}\right)\right) \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right)
        \left(- \frac{T}{T_{c,i}} + \frac{7}{10}\right)\right) + 1\right)^{2}

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}`, [Pa*m^6/mol^2]
    kappa0s : list[float]
        `kappa0` parameters of PRSV EOS; the original form uses
        :math:`\kappa_{0,i} = 0.378893 + 1.4897153\omega_i - 0.17131848\omega_i^2 + 0.0196554\omega_i^3`, [-]
    kappa1s : list[float]
        Fit parameters, can be set to 0 if unknown [-]
    kappa2s : list[float]
        Fit parameters, can be set to 0 if unknown [-]
    kappa3s : list[float]
        Fit parameters, can be set to 0 if unknown [-]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]

    Notes
    -----

    Examples
    --------
    >>> PRSV2_a_alphas_vectorized(400.0, Tcs=[507.6], ais=[2.6923169620277805], kappa0s=[0.8074380841890093], kappa1s=[0.05104], kappa2s=[0.8634], kappa3s=[0.460])
    [3.2005700986984]
    '''
    sqrtT = sqrt(T)
    N = len(Tcs)
    if a_alphas is None:
        a_alphas = [0.0]*N
    for i in range(N):
        Tc_inv_root = 1.0/sqrt(Tcs[i])
        Tr_sqrt = sqrtT*Tc_inv_root
        Tr = T*Tc_inv_root*Tc_inv_root
        kappa = (kappa0s[i] + ((kappa1s[i] + kappa2s[i]*(kappa3s[i] - Tr)
                 *(1.0 - Tr_sqrt))*(1.0 + Tr_sqrt)*(0.7 - Tr)))
        x0 = (1.0 + kappa*(1.0 - Tr_sqrt))
        a_alphas[i] = ais[i]*x0*x0
    return a_alphas

def PRSV2_a_alpha_and_derivatives_vectorized(T, Tcs, ais, kappa0s, kappa1s, kappa2s, kappa3s,
                                             a_alphas=None, da_alpha_dTs=None, d2a_alpha_dT2s=None):
    r'''Calculates the `a_alpha` terms and their first and second derivatives
    for the Peng-Robinson-Stryjek-Vera 2
    equation of state given the critical temperatures `Tcs`, constants `ais`,
    PRSV2 parameters `kappa0s, `kappa1s`, `kappa2s`, and `kappa3s`.

    .. math::
        a_i\alpha_i = a_{i} \left(\left(1 - \sqrt{\frac{T}{T_{c,i}}}\right)
        \left(\kappa_{0,i} + \left(\kappa_{1,i} + \kappa_{2,i} \left(1
        - \sqrt{\frac{T}{T_{c,i}}}\right) \left(- \frac{T}{T_{c,i}}
        + \kappa_{3,i}\right)\right) \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right)
        \left(- \frac{T}{T_{c,i}} + \frac{7}{10}\right)\right) + 1\right)^{2}

    .. math::
        \frac{d a_i\alpha_i}{dT} = a_{i} \left(\left(1 - \sqrt{\frac{T}{T_{c,i}
        }}\right) \left(\kappa_{0,i} + \left(\kappa_{1,i} + \kappa_{2,i} \left(
        1 - \sqrt{\frac{T}{T_{c,i}}}\right) \left(- \frac{T}{T_{c,i}}
        + \kappa_{3,i}\right)\right) \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right)
        \left(- \frac{T}{T_{c,i}} + \frac{7}{10}\right)\right) + 1\right)
        \left(2 \left(1 - \sqrt{\frac{T}{T_{c,i}}}\right) \left(\left(\sqrt{
        \frac{T}{T_{c,i}}} + 1\right) \left(- \frac{T}{T_{c,i}} + \frac{7}{10}
        \right) \left(- \frac{\kappa_{2,i} \left(1 - \sqrt{\frac{T}{T_{c,i}}}
        \right)}{T_{c,i}} - \frac{\kappa_{2,i} \sqrt{\frac{T}{T_{c,i}}} \left(
        - \frac{T}{T_{c,i}} + \kappa_{3,i}\right)}{2 T}\right) - \frac{\left(
        \kappa_{1,i} + \kappa_{2,i} \left(1 - \sqrt{\frac{T}{T_{c,i}}}\right)
        \left(- \frac{T}{T_{c,i}} + \kappa_{3,i}\right)\right) \left(\sqrt{
        \frac{T}{T_{c,i}}} + 1\right)}{T_{c,i}} + \frac{\sqrt{\frac{T}{T_{c,i}
        }} \left(\kappa_{1,i} + \kappa_{2,i} \left(1 - \sqrt{\frac{T}{T_{c,i}}}
        \right) \left(- \frac{T}{T_{c,i}} + \kappa_{3,i}\right)\right) \left(
        - \frac{T}{T_{c,i}} + \frac{7}{10}\right)}{2 T}\right) - \frac{\sqrt{
        \frac{T}{T_{c,i}}} \left(\kappa_{0,i} + \left(\kappa_{1,i}
        + \kappa_{2,i} \left(1 - \sqrt{\frac{T}{T_{c,i}}}\right) \left(
        - \frac{T}{T_{c,i}} + \kappa_{3,i}\right)\right) \left(\sqrt{\frac{T}
        {T_{c,i}}} + 1\right) \left(- \frac{T}{T_{c,i}} + \frac{7}{10}\right)
        \right)}{T}\right)

    .. math::
        \frac{d^2 a_i\alpha_i}{dT^2} = - \frac{a_{i} \left(\left(\left(10
        \kappa_{0,i} - \left(\kappa_{1,i} + \kappa_{2,i} \left(\sqrt{\frac{T}
        {T_{c,i}}} - 1\right) \left(\frac{T}{T_{c,i}} - \kappa_{3,i}\right)
        \right) \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right) \left(\frac{10 T}
        {T_{c,i}} - 7\right)\right) \left(\sqrt{\frac{T}{T_{c,i}}} - 1\right)
        - 10\right) \left(\left(\sqrt{\frac{T}{T_{c,i}}} - 1\right) \left(
        \frac{40 \kappa_{2,i} \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right) \left(
        \frac{2 \left(\sqrt{\frac{T}{T_{c,i}}} - 1\right)}{T_{c,i}} + \frac{
        \sqrt{\frac{T}{T_{c,i}}} \left(\frac{T}{T_{c,i}} - \kappa_{3,i}\right)}
        {T}\right)}{T_{c,i}} + \frac{\kappa_{2,i} \sqrt{\frac{T}{T_{c,i}}}
        \left(\frac{4}{T_{c,i}} - \frac{\frac{T}{T_{c,i}} - \kappa_{3,i}}{T}
        \right) \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right) \left(\frac{10 T}
        {T_{c,i}} - 7\right)}{T} + \frac{2 \kappa_{2,i} \sqrt{\frac{T}{T_{c,i}}}
        \left(\frac{10 T}{T_{c,i}} - 7\right) \left(\frac{2 \left(\sqrt{\frac
        {T}{T_{c,i}}} - 1\right)}{T_{c,i}} + \frac{\sqrt{\frac{T}{T_{c,i}}}
        \left(\frac{T}{T_{c,i}} - \kappa_{3,i}\right)}{T}\right)}{T} + \frac{40
        \sqrt{\frac{T}{T_{c,i}}} \left(\kappa_{1,i} + \kappa_{2,i} \left(\sqrt{
        \frac{T}{T_{c,i}}} - 1\right) \left(\frac{T}{T_{c,i}} - \kappa_{3,i}
        \right)\right)}{T T_{c,i}} - \frac{\sqrt{\frac{T}{T_{c,i}}} \left(
        \kappa_{1,i} + \kappa_{2,i} \left(\sqrt{\frac{T}{T_{c,i}}} - 1\right)
        \left(\frac{T}{T_{c,i}} - \kappa_{3,i}\right)\right) \left(\frac{10 T}
        {T_{c,i}} - 7\right)}{T^{2}}\right) + \frac{2 \sqrt{\frac{T}{T_{c,i}}}
        \left(\kappa_{2,i} \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right)
        \left(\frac{10 T}{T_{c,i}} - 7\right) \left(\frac{2 \left(\sqrt{
        \frac{T}{T_{c,i}}} - 1\right)}{T_{c,i}} + \frac{\sqrt{\frac{T}
        {T_{c,i}}} \left(\frac{T}{T_{c,i}} - \kappa_{3,i}\right)}{T}\right)
        + \frac{20 \left(\kappa_{1,i} + \kappa_{2,i} \left(\sqrt{\frac{T}
        {T_{c,i}}} - 1\right) \left(\frac{T}{T_{c,i}} - \kappa_{3,i}\right)
        \right) \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right)}{T_{c,i}}
        + \frac{\sqrt{\frac{T}{T_{c,i}}} \left(\kappa_{1,i} + \kappa_{2,i}
        \left(\sqrt{\frac{T}{T_{c,i}}} - 1\right) \left(\frac{T}{T_{c,i}}
        - \kappa_{3,i}\right)\right) \left(\frac{10 T}{T_{c,i}} - 7\right)}
        {T}\right)}{T} + \frac{\sqrt{\frac{T}{T_{c,i}}} \left(10 \kappa_{0,i}
        - \left(\kappa_{1,i} + \kappa_{2,i} \left(\sqrt{\frac{T}{T_{c,i}}}
        - 1\right) \left(\frac{T}{T_{c,i}} - \kappa_{3,i}\right)\right)
        \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right) \left(\frac{10 T}{T_{c,i}}
        - 7\right)\right)}{T^{2}}\right) - \left(\left(\sqrt{\frac{T}{T_{c,i}}}
        - 1\right) \left(\kappa_{2,i} \left(\sqrt{\frac{T}{T_{c,i}}} + 1\right)
        \left(\frac{10 T}{T_{c,i}} - 7\right) \left(\frac{2 \left(\sqrt{
        \frac{T}{T_{c,i}}} - 1\right)}{T_{c,i}} + \frac{\sqrt{\frac{T}{T_{c,i}}}
        \left(\frac{T}{T_{c,i}} - \kappa_{3,i}\right)}{T}\right) + \frac{20
        \left(\kappa_{1,i} + \kappa_{2,i} \left(\sqrt{\frac{T}{T_{c,i}}}
        - 1\right) \left(\frac{T}{T_{c,i}} - \kappa_{3,i}\right)\right) \left(
        \sqrt{\frac{T}{T_{c,i}}} + 1\right)}{T_{c,i}} + \frac{\sqrt{\frac{T}
        {T_{c,i}}} \left(\kappa_{1,i} + \kappa_{2,i} \left(\sqrt{\frac{T}
        {T_{c,i}}} - 1\right) \left(\frac{T}{T_{c,i}} - \kappa_{3,i}\right)
        \right) \left(\frac{10 T}{T_{c,i}} - 7\right)}{T}\right) - \frac{
        \sqrt{\frac{T}{T_{c,i}}} \left(10 \kappa_{0,i} - \left(\kappa_{1,i}
        + \kappa_{2,i} \left(\sqrt{\frac{T}{T_{c,i}}} - 1\right) \left(\frac{T}
        {T_{c,i}} - \kappa_{3,i}\right)\right) \left(\sqrt{\frac{T}{T_{c,i}}}
        + 1\right) \left(\frac{10 T}{T_{c,i}} - 7\right)\right)}{T}\right)^{2}
        \right)}{200}

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}`, [Pa*m^6/mol^2]
    kappa0s : list[float]
        `kappa0` parameters of PRSV EOS; the original form uses
        :math:`\kappa_{0,i} = 0.378893 + 1.4897153\omega_i - 0.17131848\omega_i^2 + 0.0196554\omega_i^3`, [-]
    kappa1s : list[float]
        Fit parameters, can be set to 0 if unknown [-]
    kappa2s : list[float]
        Fit parameters, can be set to 0 if unknown [-]
    kappa3s : list[float]
        Fit parameters, can be set to 0 if unknown [-]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]
    da_alpha_dTs : list[float]
        First temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K)]
    d2a_alpha_dT2s : list[float]
        Second temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K^2)]

    Notes
    -----

    Examples
    --------
    >>> PRSV2_a_alpha_and_derivatives_vectorized(400.0, Tcs=[507.6], ais=[2.6923169620277805], kappa0s=[0.8074380841890093], kappa1s=[0.05104], kappa2s=[0.8634], kappa3s=[0.460])
    ([3.2005700986], [-0.005301195971], [1.11181477576e-05])
    '''
    sqrtT = sqrt(T)
    T_inv = 1.0/T
    N = len(Tcs)
    if a_alphas is None:
        a_alphas = [0.0]*N
    if da_alpha_dTs is None:
        da_alpha_dTs = [0.0]*N
    if d2a_alpha_dT2s is None:
        d2a_alpha_dT2s = [0.0]*N
    for i in range(N):
        Tc_inv_root = 1.0/sqrt(Tcs[i])
        Tc_inv = Tc_inv_root*Tc_inv_root
        x1 = T*Tc_inv
        x2 = sqrtT*Tc_inv_root

        x3 = x2 - 1.
        x4 = x2 + 1.
        x5 = 10.*x1 - 7.
        x6 = -kappa3s[i] + x1
        x7 = kappa1s[i] + kappa2s[i]*x3*x6
        x8 = x5*x7
        x9 = 10.*kappa0s[i] - x4*x8
        x10 = x3*x9
        x11 = x10*0.1 - 1.0
        x13 = x2*T_inv
        x14 = x7*Tc_inv
        x15 = kappa2s[i]*x4*x5
        x16 = 2.*(-x2 + 1.)*Tc_inv + x13*(kappa3s[i] - x1)
        x17 = -x13*x8 - x14*(20.*x2 + 20.) + x15*x16
        x18 = x13*x9 + x17*x3
        x19 = x2*T_inv*T_inv
        x20 = 2.*x2*T_inv

        a_alpha = ais[i]*x11*x11
        da_alpha_dT = ais[i]*x11*x18*0.1
        d2a_alpha_dT2 = ais[i]*(x18*x18 + (x10 - 10.)*(x17*x20 - x19*x9
                           + x3*(40.*kappa2s[i]*Tc_inv*x16*x4
                         + kappa2s[i]*x16*x20*x5 - 40.*T_inv*x14*x2
                         - x15*T_inv*x2*(4.0*Tc_inv - x6*T_inv)
                         + x19*x8)))*0.005

        a_alphas[i] = a_alpha
        da_alpha_dTs[i] = da_alpha_dT
        d2a_alpha_dT2s[i] = d2a_alpha_dT2
    return a_alphas, da_alpha_dTs, d2a_alpha_dT2s

def APISRK_a_alphas_vectorized(T, Tcs, ais, S1s, S2s, a_alphas=None):
    r'''Calculates the `a_alpha` terms for the API SRK equation of state
    given the critical temperatures `Tcs`, constants `ais`, and
    API parameters `S1s` and `S2s`.

    .. math::
        a_i\alpha(T)_i = a_i \left[1 + S_{1,i}\left(1-\sqrt{T_{r,i}}\right)
         + S_{2,i} \frac{1- \sqrt{T_{r,i}}}{\sqrt{T_{r,i}}}\right]^2

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}`, [Pa*m^6/mol^2]
    S1s : list[float]
        `S1` parameters of API SRK EOS; regressed or estimated with
        :math:`S_{1,i} = 0.48508 + 1.55171\omega_i - 0.15613\omega_i^2`, [-]
    S2s : list[float]
        `S2` parameters of API SRK EOS; regressed or set to zero, [-]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]

    Notes
    -----

    Examples
    --------
    >>> APISRK_a_alphas_vectorized(T=430.0, Tcs=[514.0], ais=[1.2721974560809934],  S1s=[1.678665], S2s=[-0.216396])
    [1.60465652994097]
    '''
    N = len(Tcs)
    sqrtT = sqrt(T)
    if a_alphas is None:
        a_alphas = [0.0]*N
    for i in range(N):
        rtTr = 1.0/sqrt(Tcs[i])
        x0 = (-rtTr*sqrtT + 1.)
        x1 = 1.0/(rtTr*sqrtT)
        x2 = (S1s[i]*x0 + S2s[i]*(x0)*x1 + 1.0)
        a_alphas[i] = ais[i]*x2*x2
    return a_alphas

def APISRK_a_alpha_and_derivatives_vectorized(T, Tcs, ais, S1s, S2s, a_alphas=None, 
                                              da_alpha_dTs=None, d2a_alpha_dT2s=None):
    r'''Calculates the `a_alpha` terms and their first two temperature
    derivatives for the API SRK equation of state
    given the critical temperatures `Tcs`, constants `ais`, and
    API parameters `S1s` and `S2s`.

    .. math::
        a_i\alpha(T)_i = a_i \left[1 + S_{1,i}\left(1-\sqrt{T_{r,i}}\right)
         + S_{2,i} \frac{1- \sqrt{T_{r,i}}}{\sqrt{T_{r,i}}}\right]^2

    .. math::
        \frac{d a_i\alpha_i}{dT} = a_i\frac{T_{c,i}}{T^{2}} \left(- S_{2,i} \left(\sqrt{
        \frac{T}{T_{c,i}}} - 1\right) + \sqrt{\frac{T}{T_{c,i}}} \left(S_{1,i} \sqrt{
        \frac{T}{T_{c,i}}} + S_{2,i}\right)\right) \left(S_{2,i} \left(\sqrt{\frac{
        T}{T_{c,i}}} - 1\right) + \sqrt{\frac{T}{T_{c,i}}} \left(S_{1,i} \left(\sqrt{
        \frac{T}{T_{c,i}}} - 1\right) - 1\right)\right)

    .. math::
        \frac{d^2 a_i\alpha_i}{dT^2} = a_i\frac{1}{2 T^{3}} \left(S_{1,i}^{2} T
        \sqrt{\frac{T}{T_{c,i}}} - S_{1,i} S_{2,i} T \sqrt{\frac{T}{T_{c,i}}} + 3 S_{1,i}
        S_{2,i} T_{c,i} \sqrt{\frac{T}{T_{c,i}}} + S_{1,i} T \sqrt{\frac{T}{T_{c,i}}}
        - 3 S_{2,i}^{2} T_{c,i} \sqrt{\frac{T}{T_{c,i}}} + 4 S_{2,i}^{2} T_{c,i} + 3 S_{2,i}
        T_{c,i} \sqrt{\frac{T}{T_{c,i}}}\right)

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tcs : list[float]
        Critical temperatures of components, [K]
    ais : list[float]
        `a` parameters of cubic EOS,
        :math:`a_i=\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}`, [Pa*m^6/mol^2]
    S1s : list[float]
        `S1` parameters of API SRK EOS; regressed or estimated with
        :math:`S_{1,i} = 0.48508 + 1.55171\omega_i - 0.15613\omega_i^2`, [-]
    S2s : list[float]
        `S2` parameters of API SRK EOS; regressed or set to zero, [-]

    Returns
    -------
    a_alphas : list[float]
        Pure component `a_alpha` terms in the cubic EOS, [Pa*m^6/mol^2]
    da_alpha_dTs : list[float]
        First temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K)]
    d2a_alpha_dT2s : list[float]
        Second temperature derivative of pure component `a_alpha`,
        [Pa*m^6/(mol^2*K^2)]


    Notes
    -----

    Examples
    --------
    >>> APISRK_a_alpha_and_derivatives_vectorized(T=430.0, Tcs=[514.0], ais=[1.2721974560809934],  S1s=[1.678665], S2s=[-0.216396])
    ([1.60465652994], [-0.0043155855337], [8.9931026263e-06])
    '''
    N = len(Tcs)
    T_inv = 1.0/T
    c0 = T_inv*T_inv*0.5
    if a_alphas is None:
        a_alphas = [0.0]*N
    if da_alpha_dTs is None:
        da_alpha_dTs = [0.0]*N
    if d2a_alpha_dT2s is None:
        d2a_alpha_dT2s = [0.0]*N
    for i in range(N):
        x0 = sqrt(T/Tcs[i])
        x1 = x0 - 1.
        x2 = x1/x0
        x3 = S2s[i]*x2
        x4 = S1s[i]*x1 + x3 - 1.
        x5 = S1s[i]*x0
        x6 = S2s[i] - x3 + x5
        x7 = 3.*S2s[i]
        a_alphas[i] = ais[i]*x4*x4
        da_alpha_dTs[i] = ais[i]*x4*x6*T_inv
        d2a_alpha_dT2s[i] = ais[i]*(-x4*(-x2*x7 + x5 + x7) + x6*x6)*c0
    return a_alphas, da_alpha_dTs, d2a_alpha_dT2s

def TWU_a_alpha_common(T, Tc, omega, a, full=True, quick=True, method='PR'):
    r'''Function to calculate `a_alpha` and optionally its first and second
    derivatives for the TWUPR or TWUSRK EOS. Returns 'a_alpha', and
    optionally 'da_alpha_dT' and 'd2a_alpha_dT2'.
    Used by `TWUPR` and `TWUSRK`; has little purpose on its own.
    See either class for the correct reference, and examples of using the EOS.

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tc : float
        Critical temperature, [K]
    omega : float
        Acentric factor, [-]
    a : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
    full : float
        Whether or not to return its first and second derivatives
    quick : bool, optional
        Whether to use a SymPy cse-derived expression (3x faster) or
        individual formulas
    method : str
        Either 'PR' or 'SRK'

    Notes
    -----
    The derivatives are somewhat long and are not described here for
    brevity; they are obtainable from the following SymPy expression.

    >>> from sympy import *
    >>> T, Tc, omega, N1, N0, M1, M0, L1, L0 = symbols('T, Tc, omega, N1, N0, M1, M0, L1, L0')
    >>> Tr = T/Tc
    >>> alpha0 = Tr**(N0*(M0-1))*exp(L0*(1-Tr**(N0*M0)))
    >>> alpha1 = Tr**(N1*(M1-1))*exp(L1*(1-Tr**(N1*M1)))
    >>> alpha = alpha0 + omega*(alpha1-alpha0)
    >>> # diff(alpha, T)
    >>> # diff(alpha, T, T)
    '''
    # e-10 works
    min_a_alpha = 1e-3 # There are a LOT of formulas, and they do not like having zeros
    Tr = T/Tc
    if Tr < 5e-3:
        # not enough: Tr from (x) 0 to 2e-4 to (y) 1e-4 2e-4
        # trying: Tr from (x) 0 to 1e-3 to (y) 5e-4 1e-3
#        Tr = 1e-3 + (Tr - 0.0)*(1e-3 - 5e-4)/1e-3
#        Tr = 5e-4 + (Tr - 0.0)*(5e-4)/1e-3
        Tr = 4e-3 + (Tr - 0.0)*(1e-3)/5e-3
        T = Tc*Tr

    if method == 'PR':
        if Tr < 1.0:
            L0, M0, N0 = 0.125283, 0.911807, 1.948150
            L1, M1, N1 = 0.511614, 0.784054, 2.812520
        else:
            L0, M0, N0 = 0.401219, 4.963070, -0.2
            L1, M1, N1 = 0.024955, 1.248089, -8.
    elif method == 'SRK':
        if Tr < 1.0:
            L0, M0, N0 = 0.141599, 0.919422, 2.496441
            L1, M1, N1 = 0.500315, 0.799457, 3.291790
        else:
            L0, M0, N0 = 0.441411, 6.500018, -0.20
            L1, M1, N1 = 0.032580,  1.289098, -8.0
    else:
        raise ValueError('Only `PR` and `SRK` are accepted as method')

    if not full:
        alpha0 = Tr**(N0*(M0-1.))*exp(L0*(1.-Tr**(N0*M0)))
        alpha1 = Tr**(N1*(M1-1.))*exp(L1*(1.-Tr**(N1*M1)))
        alpha = alpha0 + omega*(alpha1 - alpha0)
        a_alpha = a*alpha
        if a_alpha < min_a_alpha:
            a_alpha = min_a_alpha
        return a_alpha
    else:
        if quick:
            x0 = Tr
            x1 = M0 - 1
            x2 = N0*x1
            x3 = x0**x2
            x4 = M0*N0
            x5 = x0**x4
            x6 = exp(-L0*(x5 - 1.))
            x7 = x3*x6
            x8 = M1 - 1.
            x9 = N1*x8
            x10 = x0**x9
            x11 = M1*N1
            x12 = x0**x11
            x13 = x2*x7
            x14 = L0*M0*N0*x3*x5*x6
            x15 = x13 - x14
            x16 = exp(-L1*(x12 - 1))
            x17 = -L1*M1*N1*x10*x12*x16 + x10*x16*x9 - x13 + x14
            x18 = N0*N0
            x19 = x18*x3*x6
            x20 = x1**2*x19
            x21 = M0**2
            x22 = L0*x18*x3*x5*x6
            x23 = x21*x22
            x24 = 2*M0*x1*x22
            x25 = L0**2*x0**(2*x4)*x19*x21
            x26 = N1**2
            x27 = x10*x16*x26
            x28 = M1**2
            x29 = L1*x10*x12*x16*x26
            a_alpha = a*(-omega*(-x10*exp(L1*(-x12 + 1)) + x3*exp(L0*(-x5 + 1))) + x7)
            da_alpha_dT = a*(omega*x17 + x15)/T
            d2a_alpha_dT2 = a*(-(omega*(-L1**2*x0**(2.*x11)*x27*x28 + 2.*M1*x29*x8 + x17 + x20 - x23 - x24 + x25 - x27*x8**2 + x28*x29) + x15 - x20 + x23 + x24 - x25)/T**2)
        else:
            alpha0 = Tr**(N0*(M0-1.))*exp(L0*(1.-Tr**(N0*M0)))
            alpha1 = Tr**(N1*(M1-1.))*exp(L1*(1.-Tr**(N1*M1)))
            alpha = alpha0 + omega*(alpha1 - alpha0)
            a_alpha = a*alpha
#            a_alpha = TWU_a_alpha_common(T=T, Tc=Tc, omega=omega, a=a, full=False, quick=quick, method=method)
            da_alpha_dT = a*(-L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T + N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T + omega*(L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T - L1*M1*N1*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(L1*(-(T/Tc)**(M1*N1) + 1))/T - N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T + N1*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)*exp(L1*(-(T/Tc)**(M1*N1) + 1))/T))
            d2a_alpha_dT2 = a*((L0**2*M0**2*N0**2*(T/Tc)**(2*M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - L0*M0**2*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - 2*L0*M0*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) + L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) + N0**2*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)**2*exp(-L0*((T/Tc)**(M0*N0) - 1)) - N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) - omega*(L0**2*M0**2*N0**2*(T/Tc)**(2*M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - L0*M0**2*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - 2*L0*M0*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) + L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - L1**2*M1**2*N1**2*(T/Tc)**(2*M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(-L1*((T/Tc)**(M1*N1) - 1)) + L1*M1**2*N1**2*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(-L1*((T/Tc)**(M1*N1) - 1)) + 2*L1*M1*N1**2*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)*exp(-L1*((T/Tc)**(M1*N1) - 1)) - L1*M1*N1*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(-L1*((T/Tc)**(M1*N1) - 1)) + N0**2*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)**2*exp(-L0*((T/Tc)**(M0*N0) - 1)) - N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) - N1**2*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)**2*exp(-L1*((T/Tc)**(M1*N1) - 1)) + N1*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)*exp(-L1*((T/Tc)**(M1*N1) - 1))))/T**2)
        if a_alpha < min_a_alpha:
            a_alpha = min_a_alpha
            da_alpha_dT = d2a_alpha_dT2 = 0.0
            # Hydrogen at low T
#            a_alpha = da_alpha_dT = d2a_alpha_dT2 = 0.0
        return a_alpha, da_alpha_dT, d2a_alpha_dT2


class a_alpha_base(object):
    def _init_test(self, Tc, a, alpha_coeffs, **kwargs):
        self.Tc = Tc
        self.a = a
        self.alpha_coeffs = alpha_coeffs
        self.__dict__.update(kwargs)

class Poly_a_alpha(object):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives given that there is a polynomial equation for
        :math:`\alpha`.

        .. math::
            a \alpha = a\cdot \text{poly}(T)

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
        return horner_and_der2(self.alpha_coeffs, T)

    def a_alpha_pure(self, T):
        r'''Method to calculate `a_alpha` given that there is a polynomial
        equation for :math:`\alpha`.

        .. math::
            a \alpha = a\cdot \text{poly}(T)

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        '''
        return horner(self.alpha_coeffs, T)

class Soave_1972_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1972) [1]_. Returns `a_alpha`, `da_alpha_dT`, and
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more
        documentation. Same as `SRK.a_alpha_and_derivatives` but slower and
        requiring `alpha_coeffs` to be set. One coefficient needed.

        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)
            + 1\right)^{2}

        References
        ----------
        .. [1] Soave, Giorgio. "Equilibrium Constants from a Modified Redlich-
           Kwong Equation of State." Chemical Engineering Science 27, no. 6
           (June 1972): 1197-1203. doi:10.1016/0009-2509(72)80096-4.
        .. [2] Young, André F., Fernando L. P. Pessoa, and Victor R. R. Ahón.
           "Comparison of 20 Alpha Functions Applied in the Peng–Robinson
           Equation of State for Vapor Pressure Estimation." Industrial &
           Engineering Chemistry Research 55, no. 22 (June 8, 2016): 6506-16.
           doi:10.1021/acs.iecr.6b00721.
        '''
        c1 = self.alpha_coeffs[0]
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) + 1)**2
        da_alpha_dT = -a*c1*sqrt(T/Tc)*(c1*(-sqrt(T/Tc) + 1) + 1)/T
        d2a_alpha_dT2 = a*c1*(c1/Tc - sqrt(T/Tc)*(c1*(sqrt(T/Tc) - 1) - 1)/T)/(2*T)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2
    def a_alpha_pure(self, T):
        c1 = self.alpha_coeffs[0]
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) + 1)**2
        return a_alpha

class Heyen_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Heyen (1980) [1]_. Returns `a_alpha`, `da_alpha_dT`,
        and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more
        documentation. Two coefficients needed.

        .. math::
            \alpha = e^{c_{1} \left(- \left(\frac{T}{T_{c,i}}\right)^{c_{2}}
            + 1\right)}

        References
        ----------
        .. [1] Heyen, G. Liquid and Vapor Properties from a Cubic Equation of
           State. In "Proceedings of the 2nd International Conference on Phase
           Equilibria and Fluid Properties in the Chemical Industry". DECHEMA:
           Frankfurt, 1980; p 9-13.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp(c1*(1 -(T/Tc)**c2))
        da_alpha_dT = -a*c1*c2*(T/Tc)**c2*exp(c1*(-(T/Tc)**c2 + 1))/T
        d2a_alpha_dT2 = a*c1*c2*(T/Tc)**c2*(c1*c2*(T/Tc)**c2 - c2 + 1)*exp(-c1*((T/Tc)**c2 - 1))/T**2
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp(c1*(1 -(T/Tc)**c2))
        return a_alpha

class Harmens_Knapp_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Harmens and Knapp (1980) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Two coefficients needed.

        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)
            - c_{2} \left(1 - \frac{T_{c,i}}{T}\right) + 1\right)^{2}

        References
        ----------
        .. [1] Harmens, A., and H. Knapp. "Three-Parameter Cubic Equation of
           State for Normal Substances." Industrial & Engineering Chemistry
           Fundamentals 19, no. 3 (August 1, 1980): 291-94.
           doi:10.1021/i160075a010.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) - c2*(1 - Tc/T) + 1)**2
        da_alpha_dT = a*(-c1*sqrt(T/Tc)/T - 2*Tc*c2/T**2)*(c1*(-sqrt(T/Tc) + 1) - c2*(1 - Tc/T) + 1)
        d2a_alpha_dT2 = a*((c1*sqrt(T/Tc) + 2*Tc*c2/T)**2 - (c1*sqrt(T/Tc) + 8*Tc*c2/T)*(c1*(sqrt(T/Tc) - 1) + c2*(1 - Tc/T) - 1))/(2*T**2)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2
    def a_alpha_pure(self, T):
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) - c2*(1 - Tc/T) + 1)**2
        return a_alpha

class Mathias_1983_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Mathias (1983) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Two coefficients needed.

        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)
            - c_{2} \left(- \frac{T}{T_{c,i}} + 0.7\right) \left(- \frac{T}{T_{c,i}}
            + 1\right) + 1\right)^{2}

        References
        ----------
        .. [1] Mathias, Paul M. "A Versatile Phase Equilibrium Equation of
           State." Industrial & Engineering Chemistry Process Design and
           Development 22, no. 3 (July 1, 1983): 385-91.
           doi:10.1021/i200022a008.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        Tr = T/Tc
        a_alpha = a*(1 + c1*(1-sqrt(Tr)) -c2*(1-Tr)*(0.7-Tr))**2
        da_alpha_dT = a*(c1*(-sqrt(T/Tc) + 1) - c2*(-T/Tc + 0.7)*(-T/Tc + 1) + 1)*(2*c2*(-T/Tc + 0.7)/Tc + 2*c2*(-T/Tc + 1)/Tc - c1*sqrt(T/Tc)/T)
        d2a_alpha_dT2 = a*((8*c2/Tc**2 - c1*sqrt(T/Tc)/T**2)*(c1*(sqrt(T/Tc) - 1) + c2*(T/Tc - 1)*(T/Tc - 0.7) - 1) + (2*c2*(T/Tc - 1)/Tc + 2*c2*(T/Tc - 0.7)/Tc + c1*sqrt(T/Tc)/T)**2)/2
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        Tr = T/Tc
        return a*(1 + c1*(1-sqrt(Tr)) -c2*(1-Tr)*(0.7-Tr))**2

class Mathias_Copeman_untruncated_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Mathias and Copeman (1983) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Three coefficients needed.

        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)
            + c_{2} \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)^{2} + c_{3} \left(
            - \sqrt{\frac{T}{T_{c,i}}} + 1\right)^{3} + 1\right)^{2}

        References
        ----------
        .. [1] Mathias, Paul M., and Thomas W. Copeman. "Extension of the
           Peng-Robinson Equation of State to Complex Mixtures: Evaluation of
           the Various Forms of the Local Composition Concept." Fluid Phase
           Equilibria 13 (January 1, 1983): 91-108.
           doi:10.1016/0378-3812(83)80084-3.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) + c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2
        da_alpha_dT = a*(-c1*sqrt(T/Tc)/T - 2*c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T - 3*c3*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)**2/T)*(c1*(-sqrt(T/Tc) + 1) + c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)
        d2a_alpha_dT2 = a*(T*(c1 - 2*c2*(sqrt(T/Tc) - 1) + 3*c3*(sqrt(T/Tc) - 1)**2)**2 - (2*T*(c2 - 3*c3*(sqrt(T/Tc) - 1)) + Tc*sqrt(T/Tc)*(c1 - 2*c2*(sqrt(T/Tc) - 1) + 3*c3*(sqrt(T/Tc) - 1)**2))*(c1*(sqrt(T/Tc) - 1) - c2*(sqrt(T/Tc) - 1)**2 + c3*(sqrt(T/Tc) - 1)**3 - 1))/(2*T**2*Tc)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*(c1*(-sqrt(T/Tc) + 1) + c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2


class Mathias_Copeman_a_alpha(a_alpha_base):

    def a_alphas_vectorized(self, T):
        ais, alpha_coeffs, Tcs = self.ais, self.alpha_coeffs, self.Tcs
        a_alphas = []
        for i in range(self.N):
            tau = 1.0 - (T/Tcs[i])**0.5
            if T < Tcs[i]:
                x0 = horner(alpha_coeffs[i], tau)
                a_alpha = x0*x0*ais[i]
            else:
                x = (1.0 + alpha_coeffs[i][-2]*tau)
                a_alpha = ais[i]*x*x
            a_alphas.append(a_alpha)
        return a_alphas

    def a_alpha_and_derivatives_vectorized(self, T):
        ais, alpha_coeffs, Tcs = self.ais, self.alpha_coeffs, self.Tcs
        a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [], [], []
        for i in range(self.N):
            a = ais[i]
            Tc = Tcs[i]
            rt = (T/Tc)**0.5
            tau = 1.0 - rt
            if T < Tc:
                x0, x1, x2 = horner_and_der2(alpha_coeffs[i], tau)
                a_alpha = x0*x0*a
                da_alpha_dT = -a*(rt*x0*x1/T)
                d2a_alpha_dT2 = a*((x0*x2/Tc + x1*x1/Tc + rt*x0*x1/T)/(2.0*T))
            else:
                c1 = alpha_coeffs[i][-2]
                x0 = 1.0/T
                x1 = 1.0/Tc
                x2 = rt#sqrt(T*x1)
                x3 = c1*(x2 - 1.0) - 1.0
                x4 = x0*x2*x3
                a_alpha = a*x3*x3
                da_alpha_dT = a*c1*x4
                d2a_alpha_dT2 = a*0.5*c1*x0*(c1*x1 - x4)
            a_alphas.append(a_alpha)
            da_alpha_dTs.append(da_alpha_dT)
            d2a_alpha_dT2s.append(d2a_alpha_dT2)
        return a_alphas, da_alpha_dTs, d2a_alpha_dT2s

    def a_alpha_pure(self, T):
        Tc = self.Tc
        a = self.a
        rt = (T/Tc)**0.5
        tau = 1.0 - rt
        alpha_coeffs = self.alpha_coeffs
#        alpha_coeffs [c3, c2, c1, 1] always
        if T < Tc:
            x0 = horner(alpha_coeffs, tau)
            a_alpha = x0*x0*a
            return a_alpha
        else:
            x = (1.0 + alpha_coeffs[-2]*tau)
            return a*x*x

    def a_alpha_and_derivatives_pure(self, T):
        Tc = self.Tc
        a = self.a
        rt = (T/Tc)**0.5
        tau = 1.0 - rt
        alpha_coeffs = self.alpha_coeffs
        if T < Tc:
            # Do not optimize until unit tests are in place
            x0, x1, x2 = horner_and_der2(alpha_coeffs, tau)
            a_alpha = x0*x0*a

            da_alpha_dT = -a*(rt*x0*x1/T)
            d2a_alpha_dT2 = a*((x0*x2/Tc + x1*x1/Tc + rt*x0*x1/T)/(2.0*T))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
        else:
            '''
            from sympy import *
            T, Tc, c1 = symbols('T, Tc, c1')
            tau = 1 - sqrt(T/Tc)
            alpha = (1 + c1*tau)**2
            cse([alpha, diff(alpha, T), diff(alpha, T, T)], optimizations='basic')
            '''
            c1 = alpha_coeffs[-2]
            x0 = 1.0/T
            x1 = 1.0/Tc
            x2 = rt#sqrt(T*x1)
            x3 = c1*(x2 - 1.0) - 1.0
            x4 = x0*x2*x3
            a_alpha = a*x3*x3
            da_alpha_dT = a*c1*x4
            d2a_alpha_dT2 = 0.5*a*c1*x0*(c1*x1 - x4)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class Gibbons_Laughton_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Gibbons and Laughton (1984) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Two coefficients needed.

        .. math::
            \alpha = c_{1} \left(\frac{T}{T_{c,i}} - 1\right) + c_{2}
            \left(\sqrt{\frac{T}{T_{c,i}}} - 1\right) + 1

        References
        ----------
        .. [1] Gibbons, Richard M., and Andrew P. Laughton. "An Equation of
           State for Polar and Non-Polar Substances and Mixtures" 80, no. 9
           (January 1, 1984): 1019-38. doi:10.1039/F29848001019.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(T/Tc - 1) + c2*(sqrt(T/Tc) - 1) + 1)
        da_alpha_dT = a*(c1/Tc + c2*sqrt(T/Tc)/(2*T))
        d2a_alpha_dT2 = a*(-c2*sqrt(T/Tc)/(4*T**2))
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*(c1*(T/Tc - 1) + c2*(sqrt(T/Tc) - 1) + 1)


class Soave_1984_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1984) [1]_. Returns `a_alpha`, `da_alpha_dT`, and
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more
        documentation. Two coefficients needed.

        .. math::
            \alpha = c_{1} \left(- \frac{T}{T_{c,i}} + 1\right) + c_{2} \left(-1
            + \frac{T_{c,i}}{T}\right) + 1

        References
        ----------
        .. [1] Soave, G. "Improvement of the Van Der Waals Equation of State."
           Chemical Engineering Science 39, no. 2 (January 1, 1984): 357-69.
           doi:10.1016/0009-2509(84)80034-2.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-T/Tc + 1) + c2*(-1 + Tc/T) + 1)
        da_alpha_dT = a*(-c1/Tc - Tc*c2/T**2)
        d2a_alpha_dT2 = a*(2*Tc*c2/T**3)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2
    # "Stryjek-Vera" skipped, doesn't match PRSV or PRSV2
    def a_alpha_pure(self, T):
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*(c1*(-T/Tc + 1) + c2*(-1 + Tc/T) + 1)


class Yu_Lu_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Yu and Lu (1987) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Four coefficients needed.

        .. math::
            \alpha = 10^{c_{4} \left(- \frac{T}{T_{c,i}} + 1\right) \left(
            \frac{T^{2} c_{3}}{Tc^{2}} + \frac{T c_{2}}{T_{c,i}} + c_{1}\right)}

        References
        ----------
        .. [1] Yu, Jin-Min, and Benjamin C. -Y. Lu. "A Three-Parameter Cubic
           Equation of State for Asymmetric Mixture Density Calculations."
           Fluid Phase Equilibria 34, no. 1 (January 1, 1987): 1-19.
           doi:10.1016/0378-3812(87)85047-1.
        '''
        c1, c2, c3, c4 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*10**(c4*(-T/Tc + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1))
        da_alpha_dT = a*(10**(c4*(-T/Tc + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1))*(c4*(-T/Tc + 1)*(2*T*c3/Tc**2 + c2/Tc) - c4*(T**2*c3/Tc**2 + T*c2/Tc + c1)/Tc)*log(10))
        d2a_alpha_dT2 = a*(10**(-c4*(T/Tc - 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1))*c4*(-4*T*c3/Tc - 2*c2 - 2*c3*(T/Tc - 1) + c4*(T**2*c3/Tc**2 + T*c2/Tc + c1 + (T/Tc - 1)*(2*T*c3/Tc + c2))**2*log(10))*log(10)/Tc**2)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2
    def a_alpha_pure(self, T):
        c1, c2, c3, c4 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*10**(c4*(-T/Tc + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1))


class Trebble_Bishnoi_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Trebble and Bishnoi (1987) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. One coefficient needed.

        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{T_{c,i}} + 1\right)}

        References
        ----------
        .. [1] Trebble, M. A., and P. R. Bishnoi. "Development of a New Four-
           Parameter Cubic Equation of State." Fluid Phase Equilibria 35, no. 1
           (September 1, 1987): 1-18. doi:10.1016/0378-3812(87)80001-8.
        '''
        c1 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp(c1*(-T/Tc + 1))
        da_alpha_dT = a*-c1*exp(c1*(-T/Tc + 1))/Tc
        d2a_alpha_dT2 = a*c1**2*exp(-c1*(T/Tc - 1))/Tc**2
        return a_alpha, da_alpha_dT, d2a_alpha_dT2
    def a_alpha_pure(self, T):
        c1 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*exp(c1*(-T/Tc + 1))
class Melhem_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Melhem et al. (1989) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Two coefficients needed.

        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{T_{c,i}} + 1\right) + c_{2}
            \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)^{2}}

        References
        ----------
        .. [1] Melhem, Georges A., Riju Saini, and Bernard M. Goodwin. "A
           Modified Peng-Robinson Equation of State." Fluid Phase Equilibria
           47, no. 2 (August 1, 1989): 189-237.
           doi:10.1016/0378-3812(89)80176-1.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2)
        da_alpha_dT = a*((-c1/Tc - c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T)*exp(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2))
        d2a_alpha_dT2 = a*(((c1/Tc - c2*sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T)**2 + c2*(1/Tc - sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T)/(2*T))*exp(-c1*(T/Tc - 1) + c2*(sqrt(T/Tc) - 1)**2))
        return a_alpha, da_alpha_dT, d2a_alpha_dT2
    def a_alpha_pure(self, T):
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*exp(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2)

class Androulakis_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Androulakis et al. (1989) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Three coefficients needed.

        .. math::
            \alpha = c_{1} \left(- \left(\frac{T}{T_{c,i}}\right)^{\frac{2}{3}}
            + 1\right) + c_{2} \left(- \left(\frac{T}{T_{c,i}}\right)^{\frac{2}{3}}
            + 1\right)^{2} + c_{3} \left(- \left(\frac{T}{T_{c,i}}\right)^{
            \frac{2}{3}} + 1\right)^{3} + 1

        References
        ----------
        .. [1] Androulakis, I. P., N. S. Kalospiros, and D. P. Tassios.
           "Thermophysical Properties of Pure Polar and Nonpolar Compounds with
           a Modified VdW-711 Equation of State." Fluid Phase Equilibria 45,
           no. 2 (April 1, 1989): 135-63. doi:10.1016/0378-3812(89)80254-7.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-(T/Tc)**(2/3) + 1) + c2*(-(T/Tc)**(2/3) + 1)**2 + c3*(-(T/Tc)**(2/3) + 1)**3 + 1)
        da_alpha_dT = a*(-2*c1*(T/Tc)**(2/3)/(3*T) - 4*c2*(T/Tc)**(2/3)*(-(T/Tc)**(2/3) + 1)/(3*T) - 2*c3*(T/Tc)**(2/3)*(-(T/Tc)**(2/3) + 1)**2/T)
        d2a_alpha_dT2 = a*(2*(T/Tc)**(2/3)*(c1 + 4*c2*(T/Tc)**(2/3) - 2*c2*((T/Tc)**(2/3) - 1) - 12*c3*(T/Tc)**(2/3)*((T/Tc)**(2/3) - 1) + 3*c3*((T/Tc)**(2/3) - 1)**2)/(9*T**2))
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*(c1*(-(T/Tc)**(2/3) + 1) + c2*(-(T/Tc)**(2/3) + 1)**2 + c3*(-(T/Tc)**(2/3) + 1)**3 + 1)

class Schwartzentruber_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Schwartzentruber et al. (1990) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Three coefficients needed.

        .. math::
            \alpha = \left(c_{4} \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)
            - \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right) \left(\frac{T^{2} c_{3}}
            {Tc^{2}} + \frac{T c_{2}}{T_{c,i}} + c_{1}\right) + 1\right)^{2}

        References
        ----------
        .. [1] J. Schwartzentruber, H. Renon, and S. Watanasiri, "K-values for
           Non-Ideal Systems:An Easier Way," Chem. Eng., March 1990, 118-124.
        '''
        c1, c2, c3, c4 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*((c4*(-sqrt(T/Tc) + 1) - (-sqrt(T/Tc) + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1) + 1)**2)
        da_alpha_dT = a*((c4*(-sqrt(T/Tc) + 1) - (-sqrt(T/Tc) + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1) + 1)*(-2*(-sqrt(T/Tc) + 1)*(2*T*c3/Tc**2 + c2/Tc) - c4*sqrt(T/Tc)/T + sqrt(T/Tc)*(T**2*c3/Tc**2 + T*c2/Tc + c1)/T))
        d2a_alpha_dT2 = a*(((-c4*(sqrt(T/Tc) - 1) + (sqrt(T/Tc) - 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1) + 1)*(8*c3*(sqrt(T/Tc) - 1)/Tc**2 + 4*sqrt(T/Tc)*(2*T*c3/Tc + c2)/(T*Tc) + c4*sqrt(T/Tc)/T**2 - sqrt(T/Tc)*(T**2*c3/Tc**2 + T*c2/Tc + c1)/T**2) + (2*(sqrt(T/Tc) - 1)*(2*T*c3/Tc + c2)/Tc - c4*sqrt(T/Tc)/T + sqrt(T/Tc)*(T**2*c3/Tc**2 + T*c2/Tc + c1)/T)**2)/2)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        c1, c2, c3, c4 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*((c4*(-sqrt(T/Tc) + 1) - (-sqrt(T/Tc) + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1) + 1)**2)


class Almeida_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Almeida et al. (1991) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Three coefficients needed.

        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{T_{c,i}} + 1\right) \left|{
            \frac{T}{T_{c,i}} - 1}\right|^{c_{2} - 1} + c_{3} \left(-1
            + \frac{T_{c,i}}{T}\right)}

        References
        ----------
        .. [1] Almeida, G. S., M. Aznar, and A. S. Telles. "Uma Nova Forma de
           Dependência Com a Temperatura Do Termo Atrativo de Equações de
           Estado Cúbicas." RBE, Rev. Bras. Eng., Cad. Eng. Quim 8 (1991): 95.
        '''
        # Note: For the second derivative, requires the use a CAS which can
        # handle the assumption that Tr-1 != 0.
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp(c1*(-T/Tc + 1)*abs(T/Tc - 1)**(c2 - 1) + c3*(-1 + Tc/T))
        da_alpha_dT = a*((c1*(c2 - 1)*(-T/Tc + 1)*abs(T/Tc - 1)**(c2 - 1)*copysign(1, T/Tc - 1)/(Tc*abs(T/Tc - 1)) - c1*abs(T/Tc - 1)**(c2 - 1)/Tc - Tc*c3/T**2)*exp(c1*(-T/Tc + 1)*abs(T/Tc - 1)**(c2 - 1) + c3*(-1 + Tc/T)))
        d2a_alpha_dT2 = a*exp(c3*(Tc/T - 1) - c1*abs(T/Tc - 1)**(c2 - 1)*(T/Tc - 1))*((c1*abs(T/Tc - 1)**(c2 - 1))/Tc + (Tc*c3)/T**2 + (c1*abs(T/Tc - 1)**(c2 - 2)*copysign(1, T/Tc - 1)*(c2 - 1)*(T/Tc - 1))/Tc)**2 - exp(c3*(Tc/T - 1) - c1*abs(T/Tc - 1)**(c2 - 1)*(T/Tc - 1))*((2*c1*abs(T/Tc - 1)**(c2 - 2)*copysign(1, T/Tc - 1)*(c2 - 1))/Tc**2 - (2*Tc*c3)/T**3 + (c1*abs(T/Tc - 1)**(c2 - 3)*copysign(1, T/Tc - 1)**2*(c2 - 1)*(c2 - 2)*(T/Tc - 1))/Tc**2)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2
    def a_alpha_pure(self, T):
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*exp(c1*(-T/Tc + 1)*abs(T/Tc - 1)**(c2 - 1) + c3*(-1 + Tc/T))


class Twu91_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Twu et al. (1991) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Three coefficients needed.

        .. math::
            \alpha = \left(\frac{T}{T_{c,i}}\right)^{c_{3} \left(c_{2}
            - 1\right)} e^{c_{1} \left(- \left(\frac{T}{T_{c,i}}
            \right)^{c_{2} c_{3}} + 1\right)}

        References
        ----------
        .. [1] Twu, Chorng H., David Bluck, John R. Cunningham, and John E.
           Coon. "A Cubic Equation of State with a New Alpha Function and a
           New Mixing Rule." Fluid Phase Equilibria 69 (December 10, 1991):
           33-50. doi:10.1016/0378-3812(91)90024-2.
        '''
        c0, c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        Tr = T/Tc
        T_inv = 1.0/T
        x1 = c1 - 1.0
        x2 = c2*x1
        x3 = c1*c2
        x4 = Tr**x3
        x5 = a*Tr**x2*exp(-c0*(x4 - 1.0))
        x6 = c0*x4
        x7 = c1*x6
        x8 = c2*x5
        x9 = c1*c1*c2
        d2a_alpha_dT2 = (x8*(c0*c0*x4*x4*x9 - c1 + c2*x1*x1
                             - 2.0*x2*x7 - x6*x9 + x7 + 1.0)*T_inv*T_inv)
        return x5, x8*(x1 - x7)*T_inv, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        c0, c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        Tr = T/Tc
        a_alpha = a*(Tr**(c2*(c1 - 1.0))*exp(c0*(1.0 - (Tr)**(c1*c2))))
        return a_alpha

    def a_alphas_vectorized(self, T):
        ais, alpha_coeffs, Tcs = self.ais, self.alpha_coeffs, self.Tcs
        a_alphas = []
        for i in range(self.N):
            coeffs = alpha_coeffs[i]
            Tr = T/Tcs[i]
            a_alpha = ais[i]*(Tr**(coeffs[2]*(coeffs[1] - 1.0))*exp(coeffs[0]*(1.0 - (Tr)**(coeffs[1]*coeffs[2]))))
            a_alphas.append(a_alpha)
        if self.scalar:
            return a_alphas
        return array(a_alphas)

    def a_alpha_and_derivatives_vectorized(self, T):
        r'''Method to calculate the pure-component `a_alphas` and their first
        and second derivatives for TWU91 alpha function EOS. This vectorized
        implementation is added for extra speed.

        .. math::
            \alpha = \left(\frac{T}{T_{c,i}}\right)^{c_{3} \left(c_{2}
            - 1\right)} e^{c_{1} \left(- \left(\frac{T}{T_{c,i}}
            \right)^{c_{2} c_{3}} + 1\right)}

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
        ais, alpha_coeffs, Tcs = self.ais, self.alpha_coeffs, self.Tcs
        N = len(ais)
        a_alphas = [0.0]*N
        da_alpha_dTs = [0.0]*N
        d2a_alpha_dT2s = [0.0]*N
        T_inv = 1.0/T
        for i in range(N):
            coeffs = alpha_coeffs[i]
            c0, c1, c2 = coeffs[0], coeffs[1], coeffs[2]
            Tr = T/Tcs[i]

            x1 = c1 - 1.0
            x2 = c2*x1
            x3 = c1*c2
            x4 = Tr**x3
            x5 = ais[i]*Tr**x2*exp(-c0*(x4 - 1.0))
            x6 = c0*x4
            x7 = c1*x6
            x8 = c2*x5
            x9 = c1*c1*c2

            d2a_alpha_dT2 = (x8*(c0*c0*x4*x4*x9 - c1 + c2*x1*x1
                                 - 2.0*x2*x7 - x6*x9 + x7 + 1.0)*T_inv*T_inv)
            a_alphas[i] = x5
            da_alpha_dTs[i] = x8*(x1 - x7)*T_inv
            d2a_alpha_dT2s[i] = d2a_alpha_dT2

        if self.scalar:
            return a_alphas, da_alpha_dTs, d2a_alpha_dT2s
        return array(a_alphas), array(da_alpha_dTs), array(d2a_alpha_dT2s)


class Soave_93_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1983) [1]_. Returns `a_alpha`, `da_alpha_dT`, and
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more
        documentation. Two coefficient needed.

        .. math::
            \alpha = c_{1} \left(- \frac{T}{T_{c,i}} + 1\right) + c_{2}
            \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)^{2} + 1

        References
        ----------
        .. [1] Soave, G. "Improving the Treatment of Heavy Hydrocarbons by the
           SRK EOS." Fluid Phase Equilibria 84 (April 1, 1993): 339-42.
           doi:10.1016/0378-3812(93)85131-5.
        '''
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2 + 1)
        da_alpha_dT = a*(-c1/Tc - c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T)
        d2a_alpha_dT2 = a*(c2*(1/Tc - sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T)/(2*T))
        return a_alpha, da_alpha_dT, d2a_alpha_dT2
    def a_alpha_pure(self, T):
        c1, c2 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2 + 1)


class Gasem_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Gasem (2001) [1]_. Returns `a_alpha`, `da_alpha_dT`, and
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more
        documentation. Three coefficients needed.

        .. math::
            \alpha = e^{\left(- \left(\frac{T}{T_{c,i}}\right)^{c_{3}} + 1\right)
            \left(\frac{T c_{2}}{T_{c,i}} + c_{1}\right)}

        References
        ----------
        .. [1] Gasem, K. A. M, W Gao, Z Pan, and R. L Robinson Jr. "A Modified
           Temperature Dependence for the Peng-Robinson Equation of State."
           Fluid Phase Equilibria 181, no. 1–2 (May 25, 2001): 113-25.
           doi:10.1016/S0378-3812(01)00488-5.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(exp((-(T/Tc)**c3 + 1)*(T*c2/Tc + c1)))
        da_alpha_dT = a*((c2*(-(T/Tc)**c3 + 1)/Tc - c3*(T/Tc)**c3*(T*c2/Tc + c1)/T)*exp((-(T/Tc)**c3 + 1)*(T*c2/Tc + c1)))
        d2a_alpha_dT2 = a*(((c2*((T/Tc)**c3 - 1)/Tc + c3*(T/Tc)**c3*(T*c2/Tc + c1)/T)**2 - c3*(T/Tc)**c3*(2*c2/Tc + c3*(T*c2/Tc + c1)/T - (T*c2/Tc + c1)/T)/T)*exp(-((T/Tc)**c3 - 1)*(T*c2/Tc + c1)))
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*(exp((-(T/Tc)**c3 + 1)*(T*c2/Tc + c1)))


class Coquelet_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Coquelet et al. (2004) [1]_. Returns `a_alpha`, `da_alpha_dT`, and
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more
        documentation. Three coefficients needed.

        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{T_{c,i}} + 1\right) \left(c_{2}
            \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)^{2} + c_{3}
            \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)^{3} + 1\right)^{2}}

        References
        ----------
        .. [1] Coquelet, C., A. Chapoy, and D. Richon. "Development of a New
           Alpha Function for the Peng–Robinson Equation of State: Comparative
           Study of Alpha Function Models for Pure Gases (Natural Gas
           Components) and Water-Gas Systems." International Journal of
           Thermophysics 25, no. 1 (January 1, 2004): 133-58.
           doi:10.1023/B:IJOT.0000022331.46865.2f.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(exp(c1*(-T/Tc + 1)*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2))
        da_alpha_dT = a*((c1*(-T/Tc + 1)*(-2*c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T - 3*c3*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)**2/T)*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1) - c1*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2/Tc)*exp(c1*(-T/Tc + 1)*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2))
        d2a_alpha_dT2 = a*(c1*(c1*(-(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)/Tc + sqrt(T/Tc)*(-2*c2 + 3*c3*(sqrt(T/Tc) - 1))*(sqrt(T/Tc) - 1)*(T/Tc - 1)/T)**2*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)**2 - ((T/Tc - 1)*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)*(2*c2/Tc - 6*c3*(sqrt(T/Tc) - 1)/Tc - 2*c2*sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T + 3*c3*sqrt(T/Tc)*(sqrt(T/Tc) - 1)**2/T) + 4*sqrt(T/Tc)*(2*c2 - 3*c3*(sqrt(T/Tc) - 1))*(sqrt(T/Tc) - 1)*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)/Tc + (2*c2 - 3*c3*(sqrt(T/Tc) - 1))**2*(sqrt(T/Tc) - 1)**2*(T/Tc - 1)/Tc)/(2*T))*exp(-c1*(T/Tc - 1)*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)**2))
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*(exp(c1*(-T/Tc + 1)*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2))


class Haghtalab_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Haghtalab et al. (2010) [1]_. Returns `a_alpha`, `da_alpha_dT`, and
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more
        documentation. Three coefficients needed.

        .. math::
            \alpha = e^{\left(- c_{3}^{\ln{\left (\frac{T}{T_{c,i}} \right )}}
            + 1\right) \left(- \frac{T c_{2}}{T_{c,i}} + c_{1}\right)}

        References
        ----------
        .. [1] Haghtalab, A., M. J. Kamali, S. H. Mazloumi, and P. Mahmoodi.
           "A New Three-Parameter Cubic Equation of State for Calculation
           Physical Properties and Vapor-liquid Equilibria." Fluid Phase
           Equilibria 293, no. 2 (June 25, 2010): 209-18.
           doi:10.1016/j.fluid.2010.03.029.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*exp((-c3**log(T/Tc) + 1)*(-T*c2/Tc + c1))
        da_alpha_dT = a*((-c2*(-c3**log(T/Tc) + 1)/Tc - c3**log(T/Tc)*(-T*c2/Tc + c1)*log(c3)/T)*exp((-c3**log(T/Tc) + 1)*(-T*c2/Tc + c1)))
        d2a_alpha_dT2 = a*(((c2*(c3**log(T/Tc) - 1)/Tc + c3**log(T/Tc)*(T*c2/Tc - c1)*log(c3)/T)**2 + c3**log(T/Tc)*(2*c2/Tc + (T*c2/Tc - c1)*log(c3)/T - (T*c2/Tc - c1)/T)*log(c3)/T)*exp((c3**log(T/Tc) - 1)*(T*c2/Tc - c1)))
        return a_alpha, da_alpha_dT, d2a_alpha_dT2
    def a_alpha_pure(self, T):
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*exp((-c3**log(T/Tc) + 1)*(-T*c2/Tc + c1))


class Saffari_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Saffari and Zahedi (2013) [1]_. Returns `a_alpha`, `da_alpha_dT`, and
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more
        documentation. Three coefficients needed.

        .. math::
            \alpha = e^{\frac{T c_{1}}{T_{c,i}} + c_{2} \ln{\left (\frac{T}{T_{c,i}}
            \right )} + c_{3} \left(- \sqrt{\frac{T}{T_{c,i}}} + 1\right)}

        References
        ----------
        .. [1] Saffari, Hamid, and Alireza Zahedi. "A New Alpha-Function for
           the Peng-Robinson Equation of State: Application to Natural Gas."
           Chinese Journal of Chemical Engineering 21, no. 10 (October 1,
           2013): 1155-61. doi:10.1016/S1004-9541(13)60581-9.
        '''
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        a_alpha = a*(exp(T*c1/Tc + c2*log(T/Tc) + c3*(-sqrt(T/Tc) + 1)))
        da_alpha_dT = a*((c1/Tc + c2/T - c3*sqrt(T/Tc)/(2*T))*exp(T*c1/Tc + c2*log(T/Tc) + c3*(-sqrt(T/Tc) + 1)))
        d2a_alpha_dT2 = a*(((2*c1/Tc + 2*c2/T - c3*sqrt(T/Tc)/T)**2 - (4*c2 - c3*sqrt(T/Tc))/T**2)*exp(T*c1/Tc + c2*log(T/Tc) - c3*(sqrt(T/Tc) - 1))/4)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2
    def a_alpha_pure(self, T):
        c1, c2, c3 = self.alpha_coeffs
        Tc, a = self.Tc, self.a
        return a*(exp(T*c1/Tc + c2*log(T/Tc) + c3*(-sqrt(T/Tc) + 1)))


class Chen_Yang_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Hamid and Yang (2017) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Seven coefficients needed.

        .. math::
            \alpha = e^{\left(- c_{3}^{\ln{\left (\frac{T}{T_{c,i}} \right )}}
            + 1\right) \left(- \frac{T c_{2}}{T_{c,i}} + c_{1}\right)}

        References
        ----------
        .. [1] Chen, Zehua, and Daoyong Yang. "Optimization of the Reduced
           Temperature Associated with Peng–Robinson Equation of State and
           Soave-Redlich-Kwong Equation of State To Improve Vapor Pressure
           Prediction for Heavy Hydrocarbon Compounds." Journal of Chemical &
           Engineering Data, August 31, 2017. doi:10.1021/acs.jced.7b00496.
        '''
        c1, c2, c3, c4, c5, c6, c7 = self.alpha_coeffs
        Tc, a, omega = self.Tc, self.a, self.omega
        a_alpha = a*exp(c4*log((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)**2 + (-T/Tc + 1)*(c1 + c2*omega + c3*omega**2))
        da_alpha_dT = a*(-(c1 + c2*omega + c3*omega**2)/Tc - c4*sqrt(T/Tc)*(c5 + c6*omega + c7*omega**2)*log((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)/(T*((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)))*exp(c4*log((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)**2 + (-T/Tc + 1)*(c1 + c2*omega + c3*omega**2))
        d2a_alpha_dT2 = a*(((c1 + c2*omega + c3*omega**2)/Tc - c4*sqrt(T/Tc)*(c5 + c6*omega + c7*omega**2)*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)/(T*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)))**2 - c4*(c5 + c6*omega + c7*omega**2)*((c5 + c6*omega + c7*omega**2)*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)/(Tc*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)) - (c5 + c6*omega + c7*omega**2)/(Tc*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)) + sqrt(T/Tc)*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)/T)/(2*T*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)))*exp(c4*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)**2 - (T/Tc - 1)*(c1 + c2*omega + c3*omega**2))
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        c1, c2, c3, c4, c5, c6, c7 = self.alpha_coeffs
        Tc, a, omega = self.Tc, self.a, self.omega
        return a*exp(c4*log((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)**2 + (-T/Tc + 1)*(c1 + c2*omega + c3*omega**2))

class TwuSRK95_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives for the Twu alpha function. Uses the set values of `Tc`,
        `omega` and `a`.

        .. math::
            \alpha = \alpha^{(0)} + \omega(\alpha^{(1)}-\alpha^{(0)})

        .. math::
            \alpha^{(i)} = T_r^{N(M-1)}\exp[L(1-T_r^{NM})]

        For sub-critical conditions:

        L0, M0, N0 =  0.141599, 0.919422, 2.496441

        L1, M1, N1 = 0.500315, 0.799457, 3.291790

        For supercritical conditions:

        L0, M0, N0 = 0.441411, 6.500018, -0.20

        L1, M1, N1 = 0.032580,  1.289098, -8.0


        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]

        Notes
        -----
        This method does not alter the object's state and the temperature
        provided can be a different than that of the object.

        The derivatives are somewhat long and are not described here for
        brevity; they are obtainable from the following SymPy expression.

        >>> from sympy import *   # doctest:+SKIP
        >>> T, Tc, omega, N1, N0, M1, M0, L1, L0 = symbols('T, Tc, omega, N1, N0, M1, M0, L1, L0')  # doctest:+SKIP
        >>> Tr = T/Tc  # doctest:+SKIP
        >>> alpha0 = Tr**(N0*(M0-1))*exp(L0*(1-Tr**(N0*M0)))  # doctest:+SKIP
        >>> alpha1 = Tr**(N1*(M1-1))*exp(L1*(1-Tr**(N1*M1)))  # doctest:+SKIP
        >>> alpha = alpha0 + omega*(alpha1-alpha0)  # doctest:+SKIP
        >>> diff(alpha, T)  # doctest:+SKIP
        >>> diff(alpha, T, T)  # doctest:+SKIP
        '''
        return TWU_a_alpha_common(T, self.Tc, self.omega, self.a, full=True, quick=True, method='SRK')

    def a_alpha_pure(self, T):
        r'''Method to calculate :math:`a \alpha` for the Twu alpha function.
        Uses the set values of `Tc`, `omega` and `a`.

        .. math::
            \alpha = \alpha^{(0)} + \omega(\alpha^{(1)}-\alpha^{(0)})

        .. math::
            \alpha^{(i)} = T_r^{N(M-1)}\exp[L(1-T_r^{NM})]

        For sub-critical conditions:

        L0, M0, N0 =  0.141599, 0.919422, 2.496441

        L1, M1, N1 = 0.500315, 0.799457, 3.291790

        For supercritical conditions:

        L0, M0, N0 = 0.441411, 6.500018, -0.20

        L1, M1, N1 = 0.032580,  1.289098, -8.0


        Parameters
        ----------
        T : float
            Temperature at which to calculate the value, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

        Notes
        -----
        This method does not alter the object's state and the temperature
        provided can be a different than that of the object.
        '''
        return TWU_a_alpha_common(T, self.Tc, self.omega, self.a, full=False, quick=True, method='SRK')

    def a_alphas_vectorized(self, T):
        Tcs, omegas, ais = self.Tcs, self.omegas, self.ais
        a_alphas = [TWU_a_alpha_common(T, Tcs[i], omegas[i], ais[i], full=False, quick=True, method='SRK')
                for i in range(self.N)]
        if self.scalar:
            return a_alphas
        return array(a_alphas)

    def a_alpha_and_derivatives_vectorized(self, T):
        Tcs, omegas, ais = self.Tcs, self.omegas, self.ais
        r0, r1, r2 = [], [], []
        for i in range(self.N):
            v0, v1, v2 = TWU_a_alpha_common(T, Tcs[i], omegas[i], ais[i], full=True, quick=True, method='SRK')
            r0.append(v0)
            r1.append(v1)
            r2.append(v2)
        if self.scalar:
            return r0, r1, r2
        return array(r0), array(r1), array(r2)



class TwuPR95_a_alpha(a_alpha_base):

    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives for the Twu alpha function. Uses the set values of `Tc`,
        `omega` and `a`.

        .. math::
            \alpha = \alpha^{(0)} + \omega(\alpha^{(1)}-\alpha^{(0)})

        .. math::
            \alpha^{(i)} = T_r^{N(M-1)}\exp[L(1-T_r^{NM})]

        For sub-critical conditions:

        L0, M0, N0 =  0.125283, 0.911807,  1.948150;

        L1, M1, N1 = 0.511614, 0.784054, 2.812520

        For supercritical conditions:

        L0, M0, N0 = 0.401219, 4.963070, -0.2;

        L1, M1, N1 = 0.024955, 1.248089, -8.


        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]

        Notes
        -----
        This method does not alter the object's state and the temperature
        provided can be a different than that of the object.

        The derivatives are somewhat long and are not described here for
        brevity; they are obtainable from the following SymPy expression.

        >>> from sympy import *   # doctest:+SKIP
        >>> T, Tc, omega, N1, N0, M1, M0, L1, L0 = symbols('T, Tc, omega, N1, N0, M1, M0, L1, L0')  # doctest:+SKIP
        >>> Tr = T/Tc  # doctest:+SKIP
        >>> alpha0 = Tr**(N0*(M0-1))*exp(L0*(1-Tr**(N0*M0)))  # doctest:+SKIP
        >>> alpha1 = Tr**(N1*(M1-1))*exp(L1*(1-Tr**(N1*M1)))  # doctest:+SKIP
        >>> alpha = alpha0 + omega*(alpha1-alpha0)  # doctest:+SKIP
        >>> diff(alpha, T)  # doctest:+SKIP
        >>> diff(alpha, T, T)  # doctest:+SKIP
        '''
        return TWU_a_alpha_common(T, self.Tc, self.omega, self.a, full=True, quick=True, method='PR')

    def a_alpha_pure(self, T):
        r'''Method to calculate :math:`a \alpha` for the Twu alpha function.
        Uses the set values of `Tc`, `omega` and `a`.

        .. math::
            \alpha = \alpha^{(0)} + \omega(\alpha^{(1)}-\alpha^{(0)})

        .. math::
            \alpha^{(i)} = T_r^{N(M-1)}\exp[L(1-T_r^{NM})]

        For sub-critical conditions:

        L0, M0, N0 =  0.125283, 0.911807,  1.948150;

        L1, M1, N1 = 0.511614, 0.784054, 2.812520

        For supercritical conditions:

        L0, M0, N0 = 0.401219, 4.963070, -0.2;

        L1, M1, N1 = 0.024955, 1.248089, -8.


        Parameters
        ----------
        T : float
            Temperature at which to calculate the value, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

        Notes
        -----
        This method does not alter the object's state and the temperature
        provided can be a different than that of the object.
        '''
        return TWU_a_alpha_common(T, self.Tc, self.omega, self.a, full=False, quick=True, method='PR')

    def a_alphas_vectorized(self, T):
        Tcs, omegas, ais = self.Tcs, self.omegas, self.ais
        a_alphas = [TWU_a_alpha_common(T, Tcs[i], omegas[i], ais[i], full=False, quick=True, method='PR')
                for i in range(self.N)]
        if self.scalar:
            return a_alphas
        return array(a_alphas)

    def a_alpha_and_derivatives_vectorized(self, T):
        Tcs, omegas, ais = self.Tcs, self.omegas, self.ais
        r0, r1, r2 = [], [], []
        for i in range(self.N):
            v0, v1, v2 = TWU_a_alpha_common(T, Tcs[i], omegas[i], ais[i], full=True, quick=True, method='PR')
            r0.append(v0)
            r1.append(v1)
            r2.append(v2)
        if self.scalar:
            return r0, r1, r2
        return array(r0), array(r1), array(r2)


class Soave_79_a_alpha(a_alpha_base):
    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1979) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. Three coefficients are needed.

        .. math::
            \alpha = 1 + (1 - T_r)(M + \frac{N}{T_r})

        References
        ----------
        .. [1] Soave, G. "Rigorous and Simplified Procedures for Determining
           the Pure-Component Parameters in the Redlich—Kwong—Soave Equation of
           State." Chemical Engineering Science 35, no. 8 (January 1, 1980):
           1725-30. https://doi.org/10.1016/0009-2509(80)85007-X.
        '''
        M, N = self.alpha_coeffs#self.M, self.N
        Tc, a = self.Tc, self.a
        T_inv = 1.0/T
        x0 = 1.0/Tc
        x1 = T*x0 - 1.0
        x2 = Tc*T_inv
        x3 = M + N*x2
        x4 = N*T_inv*T_inv
        return (a*(1.0 - x1*x3), a*(Tc*x1*x4 - x0*x3), a*(2.0*x4*(1.0 - x1*x2)))

    def a_alpha_pure(self, T):
        M, N = self.alpha_coeffs#self.M, self.N
        Tc, a = self.Tc, self.a
        Tr = T/Tc
        return a*(1.0 + (1.0 - Tr)*(M + N/Tr))

    def a_alphas_vectorized(self, T):
        ais, alpha_coeffs, Tcs = self.ais, self.alpha_coeffs, self.Tcs
        a_alphas = []
        for i in range(self.N):
            Tr = T/Tcs[i]
            M, N = alpha_coeffs[i]
            a_alphas.append(ais[i]*(1.0 + (1.0 - Tr)*(M + N/Tr)))
        return a_alphas

    def a_alpha_and_derivatives_vectorized(self, T):
        ais, alpha_coeffs, Tcs = self.ais, self.alpha_coeffs, self.Tcs
        T_inv = 1.0/T
        a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [], [], []
        for i in range(self.N):
            a = ais[i]
            M, N = alpha_coeffs[i]
            x0 = 1.0/Tcs[i]
            x1 = T*x0 - 1.0
            x2 = Tcs[i]*T_inv
            x3 = M + N*x2
            x4 = N*T_inv*T_inv

            a_alphas.append(a*(1.0 - x1*x3))
            da_alpha_dTs.append(a*(Tcs[i]*x1*x4 - x0*x3))
            d2a_alpha_dT2s.append(a*(2.0*x4*(1.0 - x1*x2)))
        return a_alphas, da_alpha_dTs, d2a_alpha_dT2s


a_alpha_bases = [Soave_1972_a_alpha, Heyen_a_alpha, Harmens_Knapp_a_alpha, Mathias_1983_a_alpha,
                 Mathias_Copeman_untruncated_a_alpha, Gibbons_Laughton_a_alpha, Soave_1984_a_alpha, Yu_Lu_a_alpha,
                 Trebble_Bishnoi_a_alpha, Melhem_a_alpha, Androulakis_a_alpha, Schwartzentruber_a_alpha,
                 Almeida_a_alpha, Twu91_a_alpha, Soave_93_a_alpha, Gasem_a_alpha,
                 Coquelet_a_alpha, Haghtalab_a_alpha, Saffari_a_alpha, Chen_Yang_a_alpha,
                 Mathias_Copeman_a_alpha,
                 TwuSRK95_a_alpha, TwuPR95_a_alpha, Soave_79_a_alpha]


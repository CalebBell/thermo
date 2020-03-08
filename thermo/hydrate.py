# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from __future__ import division
from math import log, exp, log10
from fluids.constants import psi, psi_inv
from fluids.numerics import roots_quartic
from thermo.utils import SG
__all__ = ['Caroll_hydrate_formation_P_pure', 'Motiee_hydrate_formation_T']


# Table 13.2 in Phase Behavior of Petroleum Reservoir Fluids with additinos for all cases
# Type I
typeI_formers_small_names = ['nitrogen', 'CO2', 'H2S', 'methane']
typeI_formers_small_CASs = ['7727-37-9', '124-38-9', '7783-06-4', '74-82-8']
typeI_formers_large_names = ['nitrogen', 'CO2', 'H2S', 'methane', 'ethane']
typeI_formers_large_CASs = ['7727-37-9', '124-38-9', '7783-06-4', '74-82-8', '74-84-0']
typeI_formers_names = typeI_formers_large_names
typeI_formers_CASs = typeI_formers_large_CASs

# Type II
# type II are more stable than type I
typeII_formers_small_names = typeI_formers_small_names
typeII_formers_small_CASs = typeI_formers_small_CASs

typeII_formers_large_names =  ['nitrogen', 'CO2', 'H2S', 'methane', 'ethane', 'propane',
                              'isobutane', 'n-butane', '2,2-dimethylpropane',
                              'cyclopentane', 'cyclohexane', 'benzene']
typeII_formers_large_CASs = ['7727-37-9', '124-38-9', '7783-06-4', '74-82-8',
                             '74-84-0', '74-98-6', '75-28-5', '106-97-8', '463-82-1',
                             '287-92-3', '110-82-7', '71-43-2']
# print([Chemical(i).CAS for i in typeII_formers_large_names])
typeII_formers_names = typeII_formers_large_names
typeII_formers_CASs = typeII_formers_large_CASs

# Type H
typeH_formers_small_names = ['nitrogen', 'methane']
typeH_formers_small_CASs = ['7727-37-9', '74-82-8']

typeH_formers_huge_CASs = []
typeH_fomers_huge_names = ['2-methylbutane',
'2,2-dimethylbutane', '2,3-dimethylbutane', '2,2,3-trimethylbutane',
'2,2-dimethyl-pentane', '3,3-dimethylpentane', 'methylcyclopentane',
'ethylcyclopentane', 'methyl-cyclohexane', 'cycloheptane',
'cyclooctane']


# Carrol mentions the following also form hydrates:
# argon, krypton, xenon, oxygen, SO2, ethylene oxide,
# N2O, H2Se, SF6, pH3, AsH3, SbH3, ClO3F




Caroll_hydrate_antoine_coeffs = {
    '74-82-8': (-146.1094, 0.3165, 16556.78, 0.0),
    '74-84-0': (-278.8474, 0.5626, 33996.53, 0.0),
    '74-98-6': (-259.5822, 0.5800, 27150.70, 0.0),
    '75-28-5': (469.1248, -0.7523, -72608.26, 0.0),
    '115-07-1': (63.2863, 0.0, -17486.30, 0.0),
    '74-86-2' : (34.0727, 0.0, -9428.80, 0.0),
    '124-38-9': (-304.7103, 0.6138, 37486.96, 0.0),
    '7727-37-9': (26.1193, 0.0103, -7141.92, 0.0),
    '7783-06-4' : (-19.9874, 0.1514, 2788.88, -3.5786),
}


def Caroll_hydrate_formation_P_pure(T, CAS):
    # No temperature limits are specified
    if CAS in Caroll_hydrate_antoine_coeffs:
        A, B, C, D = Caroll_hydrate_antoine_coeffs[CAS]
        P = 1e6*exp(A + B*T + C/T + D*log(T))
    elif CAS == '74-85-1': # Ethylene
        # T range 0 to 55 C
        # Pressura range 0.5 to 500 MPa
        a0 = 3.585253810E-3
        a1 = -1.2413537E-4
        a2 = 3.0907775E-5
        a3 = -3.4162547E-6
        a4 = -1.1210772E-7
        Tinv = 1.0/T
#         ais = (a4, a3, a2, a1, a0 - Tinv)
        logP = roots_quartic(a4, a3, a2, a1, a0 - Tinv)[1].real
#         logP = np.roots(a4, a3, a2, a1, a0 - Tinv)[1].real
        P = 1e6*exp(logP)
    return P




psi_inv = 1.0/psi

def Motiee_hydrate_formation_T(P, SG):
    r'''Calculates the hydrate formation temperature at a specified operating 
    pressure and gas specific gravity using the correlation of Motiee (1991) 
    [1]_.

    .. math::
        T [^\circ F] = (-238.24469 + 78.99667 (\log_{10} P \text{[psi]})
        - 5.352544 (\log_{10} P \text{[psi]})^2
          + 349.473877\text{SG} - 150.854675\text{SG}^2 - 27.604065\text{SG}
          (\log_{10} P \text{[psi]}))

    Parameters
    ----------
    P : float
        Gas pressure, [Pa]
    SG : float
        Specific gravity of the gas with respect to air; no specific
        density is recommended so a molecular weight of 28.96 is recommended
        for simplicity, [-]

    Returns
    -------
    formation_T : float
        Hydrate formation temperature, [K]

    Notes
    -----
    
    **Confusion about constants and units in publications**
    
    [2]_ shows the form used here, as does [3]_ which provides a varity of 
    sample points used to confirm the correlation's correctness.
    [4]_ shows the same formula used here except a sign is negated on the
    constant 150.854675.
    [5]_ shows a different constant of -283.24469 instead of -238.24469 
    and also claims the units are Kelvin and kPa, which is assumed to be an
    error.

    Examples
    --------
    >>> Motiee_hydrate_formation_T(600.0*psi, 0.555)
    280.28189396992724
    
    References
    ----------
    .. [1] Motiee, M. "Estimate Possibility of Hydrates," 1991.
    .. [2] Chavoshi, Sakineh, Mani Safamirzaei, and F. Pajoum Shariati. 
       "Evaluation of Empirical Correlations for Predicting Gas Hydrate
       Formation Temperature." Gas Processing 6, no. 2 (October 1, 2018): 
       15-36. https://doi.org/10.22108/gpj.2018.112052.1036.    
    .. [3] Fattah, Khaled Ahmed Abdel. "Evaluation of Empirical Correlations
       for Natural Gas Hydrate Predictions." Сетевое Издание «Нефтегазовое 
       Дело», no. 2 (2004).
    .. [4] Mogbolu, Peter O., and John Madu. "Prediction of Onset of Gas
       Hydrate Formation in Offshore Operations." In SPE Nigeria Annual 
       International Conference and Exhibition. Society of Petroleum Engineers, 
       2014.
    .. [5] Carroll, John. Natural Gas Hydrates: A Guide for Engineers. Gulf
       Professional Publishing, 2014.
    '''
    P *= psi_inv # Convert to psi
    x = log10(P)
    # Calculate formation T in deg F
    T = (-238.24469 + 78.99667*x - 5.352544*x*x
          + 349.473877*SG - 150.854675*SG*SG - 27.604065*SG*x)
    # Convert to K
    T = (T - 32.0)/1.8 + 273.15
    return T
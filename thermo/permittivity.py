# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['CRC_Permittivity_data', 'permittivity_IAPWS', 'Permittivity']

import os
import numpy as np
import pandas as pd
from thermo.utils import N_A, epsilon_0, k
from thermo.utils import TDependentProperty

folder = os.path.join(os.path.dirname(__file__), 'Electrolytes')


CRC_Permittivity_data = pd.read_csv(os.path.join(folder, 'Permittivity (Dielectric Constant) of Liquids.tsv'),
                                    sep='\t', index_col=0)
_CRC_Permittivity_data_values = CRC_Permittivity_data.values


def permittivity_IAPWS(T, rho):
    r'''Calculate the relative permittivity of pure water as a function of.
    temperature and density. Assumes the 1997 IAPWS [1]_ formulation.

    .. math::
        \epsilon(\rho, T) =\frac{1 + A + 5B + (9 + 2A + 18B + A^2 + 10AB + 
        9B^2)^{0.5}}{4(1-B)}
        
        A(\rho, T) = \frac{N_A\mu^2\rho g}{M\epsilon_0 kT}
        
        B(\rho) = \frac{N_A\alpha\rho}{3M\epsilon_0}
        
        g(\delta,\tau) = 1 + \sum_{i=1}^{11}n_i\delta^{I_i}\tau^{J_i} 
        + n_{12}\delta\left(\frac{647.096}{228}\tau^{-1} - 1\right)^{-1.2}

        \delta = \rho/(322 \text{ kg/m}^3)
        
        \tau = T/647.096\text{K}

    Parameters
    ----------
    T : float
        Temperature of water [K]
    rho : float
        Mass density of water at T and P [kg/m^3]

    Returns
    -------
    epsilon : float
        Relative permittivity of water at T and rho, [-]

    Notes
    -----
    Validity:
    
    273.15 < T < 323.15 K for 0 < P < iceVI melting pressure at T or 1000 MPa,
    whichever is smaller.
    
    323.15 < T < 873.15 K 0 < p < 600 MPa.
    
    Coefficients:
    
    ih = [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 10];
    jh = [0.25, 1, 2.5, 1.5, 1.5, 2.5, 2, 2, 5, 0.5, 10];
    Nh = [0.978224486826, -0.957771379375, 0.237511794148, 0.714692244396,
          -0.298217036956, -0.108863472196, 0.949327488264E-1, 
          -.980469816509E-2, 0.165167634970E-4, 0.937359795772E-4, 
          -0.12317921872E-9];
    polarizability = 1.636E-40
    dipole = 6.138E-30
    
    Examples
    --------
    >>> permittivity_IAPWS(373., 958.46)
    55.56584297721836

    References
    ----------
    .. [1] IAPWS. 1997. Release on the Static Dielectric Constant of Ordinary 
       Water Substance for Temperatures from 238 K to 873 K and Pressures up 
       to 1000 MPa.
    '''
    dipole = 6.138E-30 # actual molecular dipole moment of water, in C*m
    polarizability = 1.636E-40 # actual mean molecular polarizability of water, C^2/J*m^2
    MW = 0.018015268 # molecular weight of water, kg/mol
    ih = [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 10]
    jh = [0.25, 1, 2.5, 1.5, 1.5, 2.5, 2, 2, 5, 0.5, 10]
    Nh = [0.978224486826, -0.957771379375, 0.237511794148, 0.714692244396,
          -0.298217036956, -0.108863472196, 0.949327488264E-1, 
          -.980469816509E-2, 0.165167634970E-4, 0.937359795772E-4, 
          -0.12317921872E-9]
    
    delta = rho/322.
    tau = 647.096/T
    
    g = (1 + sum([Nh[h]*delta**ih[h]*tau**jh[h] for h in range(11)])
        + 0.196096504426E-2*delta*(T/228. - 1)**-1.2)
    
    A = N_A*dipole**2*(rho/MW)*g/epsilon_0/k/T
    B = N_A*polarizability*(rho/MW)/3./epsilon_0
    epsilon = (1. + A + 5.*B + (9. + 2.*A + 18.*B + A**2 + 10.*A*B + 9.*B**2
        )**0.5)/(4. - 4.*B)
    return epsilon


CRC = 'CRC'
CRC_CONSTANT = 'CRC_CONSTANT'
permittivity_methods = [CRC, CRC_CONSTANT]
'''Holds all methods available for the Permittivity class, for use in
iterating over them.'''


class Permittivity(TDependentProperty):
    r'''Class for dealing with liquid permittivity as a function of temperature.
    Consists of one temperature-dependent simple expression and one constant
    value source.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`permittivity_methods`.

    **CRC**:
        Simple polynomials for calculating permittivity over a specified
        temperature range only. The full expression is:

        .. math::
            \epsilon_r = A + BT + CT^2 + DT^3

        Not all chemicals use all terms; in fact, few do. Data is available
        for 759 liquids, from [1]_.
    **CRC_CONSTANT**:
        Constant permittivity values at specified temperatures only.
        Data is from [1]_, and is available for 1303 liquids.

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    '''
    name = 'relative permittivity'
    units = '-'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default.'''
    property_min = 1
    '''Relative permittivity must always be larger than 1; nothing is better
    than a vacuum.'''
    property_max = 1000
    '''Maximum valid of permittivity; highest in the data available is ~240.'''

    ranked_methods = [CRC, CRC_CONSTANT]
    '''Default rankings of the available methods.'''

    def __init__(self, CASRN=''):
        self.CASRN = CASRN

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        permittivity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        permittivity above.'''

        self.tabular_data = {}
        '''tabular_data, dict: Stored (Ts, properties) for any
        tabular data; indexed by provided or autogenerated name.'''
        self.tabular_data_interpolators = {}
        '''tabular_data_interpolators, dict: Stored (extrapolator,
        spline) tuples which are interp1d instances for each set of tabular
        data; indexed by tuple of (name, interpolation_T,
        interpolation_property, interpolation_property_inv) to ensure that
        if an interpolation transform is altered, the old interpolator which
        had been created is no longer used.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by `T_dependent_property`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `T_dependent_property`.'''

        self.all_methods = set()
        '''Set of all methods available for a given CASRN and properties;
        filled by :obj:`load_all_methods`.'''

        self.load_all_methods()

    def load_all_methods(self):
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
        Tmins, Tmaxs = [], []
        if self.CASRN in CRC_Permittivity_data.index:
            methods.append(CRC_CONSTANT)
            _, self.CRC_CONSTANT_T, self.CRC_permittivity, A, B, C, D, Tmin, Tmax = _CRC_Permittivity_data_values[CRC_Permittivity_data.index.get_loc(self.CASRN)].tolist()
            self.CRC_Tmin = Tmin
            self.CRC_Tmax = Tmax
            self.CRC_coeffs = [0 if np.isnan(x) else x for x in [A, B, C, D] ]
            if not np.isnan(Tmin):
                Tmins.append(Tmin); Tmaxs.append(Tmax)
            if self.CRC_coeffs[0]:
                methods.append(CRC)
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin = min(Tmins)
            self.Tmax = max(Tmaxs)

    def calculate(self, T, method):
        r'''Method to calculate permittivity of a liquid at temperature `T`
        with a given method.

        This method has no exception handling; see `T_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate relative permittivity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        epsilon : float
            Relative permittivity of the liquid at T, [-]
        '''
        if method == CRC:
            A, B, C, D = self.CRC_coeffs
            epsilon = A + B*T + C*T**2 + D*T**3
        elif method == CRC_CONSTANT:
            epsilon = self.CRC_permittivity
        elif method in self.tabular_data:
            epsilon = self.interpolate(T, method)
        return epsilon

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For tabular data,
        extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the
        extrapolation is considered valid for all temperatures.

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
        validity = True
        if method == CRC:
            if T < self.CRC_Tmin or T > self.CRC_Tmax:
                validity = False
        elif method == CRC_CONSTANT:
            # Arbitraty choice of temperature limits
            if T < self.CRC_CONSTANT_T - 20 or T > self.CRC_CONSTANT_T + 20:
                validity = False
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity


#from scipy.constants import pi, N_A, k
##from scipy.optimize import fsolve
#
#def calc_molecular_polarizability(T, Vm, dipole, permittivity):
#    dipole *= 3.33564E-30
#    rhom = 1./Vm
#    alpha = (-4*N_A*permittivity*dipole**2*pi*rhom - 8*N_A*dipole**2*pi*rhom + 9*T*permittivity*k - 9*T*k)/(12*N_A*T*k*pi*rhom*(permittivity + 2))
#
##    def to_solve(alpha):
##        ans = rhom*(4*pi*N_A*alpha/3. + 4*pi*N_A*dipole**2/9/k/T) - (permittivity-1)/(permittivity+2)
##        return ans
##
##    alpha = fsolve(to_solve, 1e-30)
#
#    return alpha

#
#print(calc_molecular_polarizability(T=293.15, Vm=0.023862, dipole=0.827, permittivity=1.00279))
#3.61E-24


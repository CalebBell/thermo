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

__all__ = ['T_converter', 'T_scales', 'ITS90_68_difference', 'Ts_68', 
'diffs_68', 'Ts_48', 'diffs_48', 'Ts_76', 'diffs_76', 'Ts_27', 'diffs_27']

import numpy as np
from scipy.constants import C2K
from scipy.interpolate import UnivariateSpline

'''Tabulated values of T68 vs. difference as in [2]_'''
Ts_68 = np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                  29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                  44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                  59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
                  74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                  89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 110, 120,
                  130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
                  250, 260, 270, 273.15, 280, 290, 300, 310, 320, 330, 340,
                  350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460,
                  470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580,
                  590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700,
                  710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820,
                  830, 840, 850, 860, 870, 880, 890, 900, 903.89, 910, 920,
                  930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030,
                  1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130,
                  1140, 1150, 1160, 1170, 1180, 1190, 1200, 1210, 1220, 1230,
                  1235.08, 1240, 1250, 1260, 1270, 1280, 1290, 1300, 1310,
                  1320, 1330, 1337.58, 1340, 1350, 1360, 1400, 1500, 1600,
                  1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                  2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,
                  3700, 3800, 3900, 4000, 4100, 4200, 4300])
diffs_68 = np.array([-0.006, -0.003, -0.004, -0.006, -0.008, -0.009, -0.009,
                     -0.008, -0.007, -0.007, -0.006, -0.005, -0.004, -0.004,
                     -0.005, -0.006, -0.006, -0.007, -0.008, -0.008, -0.008,
                     -0.007, -0.007, -0.007, -0.006, -0.006, -0.006, -0.006,
                     -0.006, -0.006, -0.006, -0.007, -0.007, -0.007, -0.006,
                     -0.006, -0.006, -0.005, -0.005, -0.004, -0.003, -0.002,
                     -0.001, 0, 0.001, 0.002, 0.003, 0.003, 0.004, 0.004,
                     0.005, 0.005, 0.006, 0.006, 0.007, 0.007, 0.007, 0.007,
                     0.007, 0.007, 0.007, 0.008, 0.008, 0.008, 0.008, 0.008,
                     0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008,
                     0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008,
                     0.008, 0.009, 0.009, 0.009, 0.009, 0.011, 0.013, 0.014,
                     0.014, 0.014, 0.014, 0.013, 0.012, 0.012, 0.011, 0.01,
                     0.009, 0.008, 0.007, 0.005, 0.003, 0.001, 0, -0.001,
                     -0.004, -0.006, -0.009, -0.012, -0.015, -0.017, -0.02,
                     -0.023, -0.025, -0.027, -0.029, -0.031, -0.033, -0.035,
                     -0.037, -0.038, -0.039, -0.039, -0.04, -0.04, -0.04,
                     -0.04, -0.04, -0.04, -0.04, -0.039, -0.039, -0.039,
                     -0.039, -0.039, -0.039, -0.04, -0.04, -0.041, -0.042,
                     -0.043, -0.044, -0.046, -0.047, -0.05, -0.052, -0.055,
                     -0.058, -0.061, -0.064, -0.067, -0.071, -0.074, -0.078,
                     -0.082, -0.086, -0.089, -0.093, -0.097, -0.1, -0.104,
                     -0.107, -0.111, -0.114, -0.117, -0.121, -0.124, -0.125,
                     -0.12, -0.1, -0.09, -0.07, -0.06, -0.04, -0.03, -0.01, 0,
                     0.01, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05,
                     0.04, 0.04, 0.03, 0.02, 0.01, 0, -0.01, -0.03, -0.04,
                     -0.06, -0.08, -0.09, -0.11, -0.13, -0.14, -0.15, -0.16,
                     -0.17, -0.19, -0.2, -0.21, -0.22, -0.23, -0.24, -0.25,
                     -0.25, -0.25, -0.25, -0.26, -0.26, -0.27, -0.31, -0.36,
                     -0.4, -0.45, -0.5, -0.56, -0.62, -0.68, -0.74, -0.81,
                     -0.87, -0.95, -1.02, -1.09, -1.17, -1.26, -1.34, -1.43,
                     -1.52, -1.62, -1.71, -1.81, -1.92, -2.02, -2.13, -2.24,
                     -2.35, -2.46, -2.58])
Ts_48 = C2K(np.array([-180, -170, -160, -150, -140, -130, -120, -110, -100, -90,
                      -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40,
                      50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
                      180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
                      300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410,
                      420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530,
                      540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650,
                      660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770,
                      780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890,
                      900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010,
                      1020, 1030, 1040, 1050, 1060, 1070, 1100, 1200, 1300, 1400,
                      1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                      2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400,
                      3500, 3600, 3700, 3800, 3900, 4000]))
diffs_48 = np.array([0.02, 0.017, 0.007, 0, 0.001, 0.008, 0.017, 0.026, 0.035,
                     0.041, 0.045, 0.045, 0.042, 0.038, 0.032, 0.024, 0.016,
                     0.008, 0, -0.006, -0.012, -0.016, -0.02, -0.023, -0.026,
                     -0.026, -0.027, -0.027, -0.026, -0.024, -0.023, -0.02,
                     -0.018, -0.016, -0.012, -0.009, -0.005, -0.001, 0.003,
                     0.007, 0.011, 0.014, 0.018, 0.021, 0.024, 0.028, 0.03,
                     0.032, 0.034, 0.035, 0.036, 0.036, 0.037, 0.036, 0.035,
                     0.034, 0.032, 0.03, 0.028, 0.024, 0.022, 0.019, 0.015,
                     0.012, 0.009, 0.007, 0.004, 0.002, 0, -0.001, -0.002,
                     -0.001, 0, 0.002, 0.007, 0.011, 0.018, 0.025, 0.035,
                     0.047, 0.06, 0.075, 0.15, 0.22, 0.3, 0.37, 0.45, 0.52,
                     0.59, 0.66, 0.73, 0.78, 0.83, 0.88, 0.92, 0.94, 0.97,
                     0.99, 1.01, 1.02, 1.01, 1, 1, 0.99, 0.98, 0.97, 0.95,
                     0.95, 0.94, 0.95, 0.95, 0.96, 0.97, 0.98, 0.98, 0.99,
                     1.01, 1.03, 1.05, 1.07, 1.09, 1.11, 1.13, 1.15, 1.17,
                     1.19, 1.2, 1.4, 1.5, 1.6, 1.8, 1.9, 2.1, 2.2, 2.3, 2.5,
                     2.7, 2.9, 3.1, 3.2, 3.4, 3.7, 3.8, 4, 4.2, 4.4, 4.6, 4.8,
                     5.1, 5.3, 5.5, 5.8, 6, 6.3, 6.6, 6.8])
Ts_76 = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27])
diffs_76 = np.array([-0.0001, -0.0002, -0.0003, -0.0004, -0.0005, -0.0006,
                     -0.0007, -0.0008, -0.001, -0.0011, -0.0013, -0.0014,
                     -0.0016, -0.0018, -0.002, -0.0022, -0.0025, -0.0027,
                     -0.003, -0.0032, -0.0035, -0.0038, -0.0041])
Ts_27 = C2K(np.array([630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730,
                      740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840,
                      850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950,
                      960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050,
                      1060, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
                      1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700,
                      2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,
                      3700, 3800, 3900, 4000]))
diffs_27 = np.array([0.08, 0.19, 0.3, 0.42, 0.52, 0.63, 0.73, 0.83, 0.93, 1.02,
                     1.09, 1.16, 1.23, 1.29, 1.32, 1.37, 1.4, 1.42, 1.44, 1.44,
                     1.43, 1.43, 1.42, 1.41, 1.39, 1.36, 1.36, 1.34, 1.33,
                     1.32, 1.32, 1.31, 1.3, 1.28, 1.27, 1.27, 1.26, 1.25, 1.25,
                     1.24, 1.22, 1.21, 1.2, 1.18, 1.04, 0.9, 0.35, -0.09,
                     -0.54, -1.09, -1.64, -2.4, -3.06, -3.92, -4.69, -5.55,
                     -6.53, -7.6, -8.57, -9.75, -11, -12.2, -13.6, -15.1,
                     -16.6, -18.3, -19.9, -21.7, -23.7, -25.7, -27.9, -30.1,
                     -32.4, -35.1])

Ts_90_from_68 = Ts_68 + diffs_68
Ts_90_from_48 = Ts_48 + diffs_48
Ts_90_from_76 = Ts_76 + diffs_76
Ts_90_from_27 = Ts_27 + diffs_27

T68_to_T90 = UnivariateSpline(Ts_68, Ts_90_from_68, s=0)
T48_to_T90 = UnivariateSpline(Ts_48, Ts_90_from_48, s=0)
T76_to_T90 = UnivariateSpline(Ts_76, Ts_90_from_76, s=0)
T27_to_T90 = UnivariateSpline(Ts_27, Ts_90_from_27, s=0)

T90_to_T68 = UnivariateSpline(Ts_90_from_68, Ts_68, s=0)
T90_to_T48 = UnivariateSpline(Ts_90_from_48, Ts_48, s=0)
T90_to_T76 = UnivariateSpline(Ts_90_from_76, Ts_76, s=0)
T90_to_T27 = UnivariateSpline(Ts_90_from_27, Ts_27, s=0)


def ITS90_68_difference(T):
    r'''Calculates the difference between ITS-90 and ITS-68 scales using a
    series of models listed in [1]_, [2]_, and [3]_.

    The temperature difference is given by the following equations:

    From 13.8 K to 73.15 K:

    .. math::
        T_{90} - T_{68} = a_0 + \sum_{i=1}^{12} a_i[(T_{90}/K-40)/40]^i

    From 83.8 K to 903.75 K:

    .. math::
        T_{90} - T_{68} = \sum_{i=1}^8 b_i[(T_{90}/K - 273.15)/630]^i

    From 903.75 K to 1337.33 K:

    .. math::
        T_{90} - T_{68} = \sum_{i=0}^5 c_i[T_{90}/^\circ C]^i

    Above 1337.33 K:

    .. math::
        T_{90} - T_{68} = -1.398\cdot 10^{-7}\left(\frac{T_{90}}{K}\right)^2


    Parameters
    ----------
        T : float
            Temperature, ITS-90, or approximately ITS-68 [K]

    Returns
    -------
        dT : float
            Temperature, difference between ITS-90 and ITS-68 at T [K]

    Notes
    -----
    The conversion is straightforward when T90 is known. Theoretically, the
    model should be solved numerically to convert the reverse way. However,
    according to [4]_, the difference is under 0.05 mK from 73.15 K to
    903.15 K, and under 0.26 mK up to 1337.33 K.

    For temperatures under 13.8 K, no conversion is performed.

    The first set of coefficients are:
    -0.005903, 0.008174, -0.061924, -0.193388, 1.490793, 1.252347, -9.835868,
    1.411912, 25.277595, -19.183815, -18.437089, 27.000895, -8.716324.

    The second set of coefficients are:
    0, -0.148759, -0.267408, 1.08076, 1.269056, -4.089591, -1.871251,
    7.438081, -3.536296.

    The third set of coefficients are:
    7.8687209E1, -4.7135991E-1, 1.0954715E-3, -1.2357884E-6, 6.7736583E-10,
    -1.4458081E-13.
    These last coefficients use the temperature in degrees Celcius. A slightly
    older model used the following coefficients but a different equation over
    the same range:
    -0.00317, -0.97737, 1.2559, 2.03295, -5.91887, -3.23561, 7.23364,
    5.04151. The model for these coefficients was:

    .. math::
        T_{90} - T_{68} = c_0 + \sum_{i=1}^7 c_i[(T_{90}/K - 1173.15)/300]^i

    For temperatures larger than several thousand K, the differences have no
    meaning and grows quadratically.

    Examples
    --------
    >>> ITS90_68_difference(1000.)
    0.01231818956580355

    References
    ----------
    .. [1] Bedford, R. E., G. Bonnier, H. Maas, and F. Pavese. "Techniques for
       Approximating the International Temperature Scale of 1990." Bureau
       International Des Poids et Mesures, Sfievres, 1990.
    .. [2] Wier, Ron D., and Robert N. Goldberg. "On the Conversion of
       Thermodynamic Properties to the Basis of the International Temperature
       Scale of 1990." The Journal of Chemical Thermodynamics 28, no. 3
       (March 1996): 261-76. doi:10.1006/jcht.1996.0026.
    .. [3] Goldberg, Robert N., and R. D. Weir. "Conversion of Temperatures
       and Thermodynamic Properties to the Basis of the International
       Temperature Scale of 1990 (Technical Report)." Pure and Applied
       Chemistry 64, no. 10 (1992): 1545-1562. doi:10.1351/pac199264101545.
    .. [4] Code10.info. "Conversions among International Temperature Scales."
       Accessed May 22, 2016. http://www.code10.info/index.php%3Foption%3Dcom_content%26view%3Darticle%26id%3D83:conversions-among-international-temperature-scales%26catid%3D60:temperature%26Itemid%3D83.
    '''
    ais = [-0.005903, 0.008174, -0.061924, -0.193388, 1.490793, 1.252347,
           -9.835868, 1.411912, 25.277595, -19.183815, -18.437089, 27.000895,
           -8.716324]
    bis = [0, -0.148759, -0.267408, 1.08076, 1.269056, -4.089591, -1.871251,
           7.438081, -3.536296]
#    cis = [-0.00317, -0.97737, 1.2559, 2.03295, -5.91887, -3.23561, 7.23364,
#           5.04151]
    new_cs = [7.8687209E1, -4.7135991E-1, 1.0954715E-3, -1.2357884E-6,
              6.7736583E-10, -1.4458081E-13]
    dT = 0
    if T < 13.8:
        dT = 0
    elif T >= 13.8 and T <= 73.15:
        for i in range(13):
            dT += ais[i]*((T - 40.)/40.)**i
    elif T > 73.15 and T < 83.8:
        dT = 0
    elif T >= 83.8 and T <= 903.75:
        for i in range(9):
            dT += bis[i]*((T - 273.15)/630.)**i
    elif T > 903.75 and T <= 1337.33:
        # Revised function exists, but does not match the tabulated data
        # for i in range(8):
        #    dT += cis[i]*((T - 1173.15)/300.)**i
        for i in range(6):
            dT += new_cs[i]*(T-273.15)**i
    elif T > 1337.33:
        dT = -1.398E-7*T**2

    return dT


T_scales = ['ITS-90', 'ITS-68', 'ITS-27', 'ITS-48', 'ITS-76']


def T_converter(T, current, desired):
    r'''Converts the a temperature reading made in any of the scales
    'ITS-90', 'ITS-68','ITS-48', 'ITS-76', or 'ITS-27' to any of the other
    scales. Not all temperature ranges can be converted to other ranges; for
    instance, 'ITS-76' is purely for low temperatures, and 5 K on it has no
    conversion to 'ITS-90' or any other scale. Both a conversion to ITS-90 and
    to the desired scale must be possible for the conversion to occur.
    The conversion uses cubic spline interpolation.

    ITS-68 conversion is valid from 14 K to 4300 K.
    ITS-48 conversion is valid from 93.15 K to 4273.15 K
    ITS-76 conversion is valid from 5 K to 27 K.
    ITS-27 is valid from 903.15 K to 4273.15 k.

    Parameters
    ----------
        T : float
            Temperature, on `current` scale [K]
        current : str
            String representing the scale T is in, 'ITS-90', 'ITS-68',
            'ITS-48', 'ITS-76', or 'ITS-27'.
        desired : str
            String representing the scale T will be returned in, 'ITS-90',
            'ITS-68', 'ITS-48', 'ITS-76', or 'ITS-27'.

    Returns
    -------
        T : float
            Temperature, on scale `desired` [K]

    Notes
    -----
    Because the conversion is performed by spline functions, a re-conversion
    of a value will not yield exactly the original value. However, it is quite
    close.

    The use of splines is quite quick (20 micro seconds/calculation). While
    just a spline for one-way conversion could be used, a numerical solver
    would have to be used to obtain an exact result for the reverse conversion.
    This was found to take approximately 1 ms/calculation, depending on the
    region.

    Examples
    --------
    >>> T_converter(500, 'ITS-68', 'ITS-48')
    499.9470092992346

    References
    ----------
    .. [1] Wier, Ron D., and Robert N. Goldberg. "On the Conversion of
       Thermodynamic Properties to the Basis of the International Temperature
       Scale of 1990." The Journal of Chemical Thermodynamics 28, no. 3
       (March 1996): 261-76. doi:10.1006/jcht.1996.0026.
    .. [2] Goldberg, Robert N., and R. D. Weir. "Conversion of Temperatures
       and Thermodynamic Properties to the Basis of the International
       Temperature Scale of 1990 (Technical Report)." Pure and Applied
       Chemistry 64, no. 10 (1992): 1545-1562. doi:10.1351/pac199264101545.
    '''
    def range_check(T, Tmin, Tmax):
        if T < Tmin or T > Tmax:
            raise Exception('Temperature conversion is outside one or both scales')

    try:
        if current == 'ITS-90':
            pass
        elif current == 'ITS-68':
            range_check(T, 13.999, 4300.0001)
            T = T68_to_T90(T)
        elif current == 'ITS-76':
            range_check(T, 4.9999, 27.0001)
            T = T76_to_T90(T)
        elif current == 'ITS-48':
            range_check(T, 93.149999, 4273.15001)
            T = T48_to_T90(T)
        elif current == 'ITS-27':
            range_check(T, 903.15, 4273.15)
            T = T27_to_T90(T)
        else:
            raise Exception('Current scale not supported')
        # T should be in ITS-90 now

        if desired == 'ITS-90':
            pass
        elif desired == 'ITS-68':
            range_check(T, 13.999, 4300.0001)
            T = T90_to_T68(T)
        elif desired == 'ITS-76':
            range_check(T, 4.9999, 27.0001)
            T = T90_to_T76(T)
        elif desired == 'ITS-48':
            range_check(T, 93.149999, 4273.15001)
            T = T90_to_T48(T)
        elif desired == 'ITS-27':
            range_check(T, 903.15, 4273.15)
            T = T90_to_T27(T)
        else:
            raise Exception('Desired scale not supported')
    except ValueError:
        raise Exception('Temperature could not be converted to desired scale')
    return float(T)

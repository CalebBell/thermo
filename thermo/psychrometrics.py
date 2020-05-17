# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['x_w_to_humidity_ratio', 'water_saturation',
           'water_dew_T']

import os
from fluids.constants import R
from fluids.numerics import bisplev, implementation_optimize_tck, horner, horner_and_der, derivative, py_newton as newton, linspace
from chemicals.utils import log, exp, isnan
from thermo.utils import TDependentProperty
from thermo.coolprop import has_CoolProp, HAPropsSI
from thermo.vapor_pressure import Psat_IAPWS, dPsat_IAPWS_dT
from scipy.interpolate import bisplev as sp_bisplev


def x_w_to_humidity_ratio(x_w, MW_water=18.015268, MW_dry_air=28.96546):
    r'''Convert the mole fraction of water in air to a humidity ratio.
    This is a trivial equation that does not depend on any saturation routines.
    
    .. math::
        \text{HR} = \frac{x_{water} \text{MW}_{water}}{MW_{dry air}(1-x_{water})}

    Parameters
    ----------
    x_w : float
        Mole fraction of water in air at a condition; this can be saturated,
        subsaturated, or even supersaturated, [-]
    MW_water : float, optional
        Molecular weight of water, [g/mol]
    MW_dry_air : float, optional
        Molecular weight of dry air, [g/mol]
    Returns
    -------
    HR : float
        Humidity ratio - mass of water vapor per mass of dry air, [-]
        
    Notes
    -----
    The ASHRAE handbooko simplifies the MW ratio to a number of 0.621945.
    
    This calculation is equivalent to taking the mole fractions of water and
    dry air, converting them to mass fractions, and dividing the mass fraction 
    of water divided by the mass fraction of dry air.

    Examples
    --------
    >>> x_w_to_humidity_ratio(0.031417, MW_water=18.015268, MW_dry_air=28.96546)
    0.020173821183401795

    References
    ----------
    .. [1] American Society of Heating, Refrigerating and Air-Conditioning 
       Engineers. ASHRAE Handbook - Fundamentals. 2015.
    '''
    return MW_water*x_w/(MW_dry_air*(1.0 - x_w))
    # Test code - run with SymPy to check the math.
#     zs = [1-x_w, x_w]
#     MWs = [MW_dry_air, MW_water]
#     ws = zs_to_ws(zs, MWs)
#     return ws[1]/(ws[0])

ASHREA_RP1485_saturation_tck = [
  [200.0, 200.0, 200.0, 200.0, 209.38306912362552, 218.0623105268494, 232.42562347143374, 257.55966756173217, 272.2735209468338, 272.64377682403403, 273.6083277487549, 275.0997462823168, 280.4723694978397, 300.1465858289642, 347.9907658157456, 385.26613728766364, 411.424878588552, 445.0469487088845, 478.35736647440103, 520.8663555270089, 575.9315450643755, 575.9315450643755, 575.9315450643755, 575.9315450643755],
  [1000.0, 1000.0, 1000.0, 1000.0, 20343.38446820347, 89665.11889240403, 426337.6478082888, 1060634.4322609215, 2571959.692525264, 4095447.417024138, 6352353.105549088, 8752292.338454487, 10000000.0, 10000000.0, 10000000.0, 10000000.0],
  [0.4627277083396064, 0.4629738446778962, 0.4641029986626697, 0.4695515831607978, 0.4832037526650565, 0.5177824851690456, 0.5759311541811574, 0.6776251403221288, 0.8285462854877181, 1.0254728914405484, 1.1841119952078434, 1.2437918047579763, 0.48783604628696503, 0.48808491427890066, 0.48922663742876293, 0.4947355996914216, 0.5085258148241742, 0.5433565121771444, 0.6015925365121437, 0.7025070017150353, 0.8505510939813861, 1.0406648179620503, 1.1915023795539847, 1.2479197146467746, 0.529118987654894, 0.5293686243535869, 0.5305136981304099, 0.5360325263162657, 0.549819086840929, 0.5844760426343885, 0.6418342820970092, 0.7396386144374153, 0.8802968494273326, 1.0560383486598677, 1.1921141551509298, 1.242539844913444, 0.5946055573098428, 0.594850463658792, 0.5959726413376808, 0.6013832826024383, 0.6148644449994451, 0.648504584500176, 0.7033317983329711, 0.7946254120188792, 0.9220718320725811, 1.075099052475319, 1.1893073380204535, 1.2310632705016438, 0.692324077626783, 0.6925603563616018, 0.6936495513279188, 0.6988785777145251, 0.7118596499261651, 0.7439823222917762, 0.7953823531481103, 0.8785790301259014, 0.9907488627931078, 1.1192322345715802, 1.2114694936163457, 1.2447099821617709, 0.8249328377578005, 0.8251687209421086, 0.8262309066638432, 0.831378223798958, 0.8441215911916481, 0.8754432244692837, 0.9248597797946733, 1.0032225630009621, 1.1061399456611407, 1.2202909957982686, 1.299622433732241, 1.3279002414913008, 0.9457275162182328, 0.946011917214968, 0.947082838336995, 0.9522434713623142, 0.964980924026783, 0.9961980968703379, 1.0450602163085898, 1.1216107462176679, 1.2208085490888614, 1.3287281811950316, 1.403205261856891, 1.4296706247419642, 0.9974149071774054, 0.997761748293906, 0.9989119025889782, 1.0040298465723736, 1.016851071605947, 1.048078094437447, 1.0968779644740119, 1.1730392384248214, 1.2711079666083283, 1.3772477031049244, 1.44912370142445, 1.474514479050117, 1.000549638533106, 1.000877862438133, 1.0020261720307335, 1.007001569875874, 1.0192816501186561, 1.0495606519586043, 1.0966511038140956, 1.1700561308136894, 1.2645786508458623, 1.3664114122436577, 1.4371073059849193, 1.4622776878052255, 1.0002755523500142, 1.0007120662010858, 1.001782779102461, 1.0065955014746166, 1.0188743847581738, 1.0482083010054706, 1.0942276499803827, 1.1655376796081207, 1.257141702651426, 1.3555804280387482, 1.4222152393307896, 1.4457156258790007, 1.0003807088640233, 1.0012285270328964, 1.0020918323808548, 1.0067095921081337, 1.0174554369600386, 1.0447396613866098, 1.08647257939444, 1.1513687296873032, 1.2336455673137734, 1.3213096352911744, 1.3803766122765988, 1.401247823180217, 0.9997987905253182, 1.000800167234422, 1.0026195437477587, 1.0057676837412632, 1.01563390180904, 1.036623552353288, 1.0702516337248895, 1.120312513492834, 1.183476813449897, 1.2486369444283762, 1.2918461159743448, 1.3068676991936872, 1.000412677714676, 1.0017784456987138, 1.0063901380749594, 1.0094789082113194, 1.0153642580473783, 1.0330679405011185, 1.0582697227342763, 1.0976283767758477, 1.1452965230466579, 1.1946065432448303, 1.2264999945200905, 1.2377451974017064, 0.9481518085421287, 0.9578610968141993, 0.9935841203913752, 1.0133667704441942, 1.0212530874770427, 1.0369369287074108, 1.0587146520456767, 1.09124497813308, 1.1309893715421209, 1.1713507573104982, 1.1975294182038771, 1.2066307408788497, 0.9296043541848459, 0.9324682348101169, 0.944925644301309, 0.9915460209782152, 1.0222584104420236, 1.042204160904465, 1.0642281743046598, 1.0948430623940557, 1.1311218320946306, 1.1675949550668532, 1.1910574176234385, 1.1992471198421655, -0.06532498742926328, -0.0205674721228374, 0.17553995998292451, 0.9262720613577713, 0.9943094396462986, 1.0434667231343895, 1.06829773322922, 1.1027115671252412, 1.1379470424755855, 1.174549039921881, 1.1969916241619296, 1.205164916712533, 0.09772021087307314, 0.11193561528622106, 0.17597427256882395, 0.4589686387744209, 0.8978715651729539, 0.994828041137959, 1.0540943964918845, 1.1008215723973303, 1.14529226792399, 1.1840246256921245, 1.2086073089615224, 1.216908021231967, 0.06343622922079302, 0.07342634712477496, 0.11850841463581972, 0.3194320666449265, 0.6500048388427533, 0.8245888460181597, 0.9495329229299483, 1.0430416501194197, 1.1113065671145954, 1.1664436727324212, 1.198032658766441, 1.2090444573751322, -1.3378786816554726, -1.3235950095126015, -1.2585864705634866, -0.9568633778916943, -0.3293492410456366, 0.6214166845218695, 0.756128587095944, 0.8924011192134779, 0.9945593355387533, 1.064667067221918, 1.104426767680698, 1.1174389142382923, -3.0458921000380728, -3.01829217811769, -2.89253394896887, -2.305714976561856, -1.0510705303130077, 1.0154662339684686, 1.6774894550408632, 0.8232286743824879, 0.9009569456895993, 0.9794019239087943, 1.021508579626634, 1.0357082665178725],
    3,3,
]
ASHRAE_RP1485_saturation_tck = implementation_optimize_tck(ASHREA_RP1485_saturation_tck)

def water_saturation(T, P, method='ideal'):
    r'''Computes the equilibrium saturation mole fraction of water in standard
    air as a function of temperature and pressure.
    
    .. math::
        x_{water} = \frac{P^{sat}_{water}(T)\cdot f(T, P)}{P}
        
    Where :math:`f(T, P)` is the enhancement factor, a correction for 
    non-ideality.

    Parameters
    ----------
    T : float
        Temperature of air, [K]
    P : float
        Pressure of air, [Pa]
    method : str
        One of 'ideal' (no enhancement factor) or 'ASHRAE1485_2020' for a
        spline fit to the ASHRAE 1485 values. 
        
    Returns
    -------
    x_w : float
        Calculated saturation mole fraction of water in air , [-]
        
    Notes
    -----
    For the 'ASHRAE1485_2020' method, the average relative error is around 1e-5.
    However, some places have relative error up to 0.005 in mole fraction. 
    This is still substantially better than the ideal formuation. It is around
    20 times slower than the ideal formula.
    
    The range of validity of ASHRAE1485 is 100 Pa to 10 MPa, and -143.15 °C 
    to 350 °C. The 2020 fit range is slightly less, starting at 200 K and 1 kPa
    but it does extend to the same high limits.

    Examples
    --------
    >>> water_saturation(T=300.0, P=1e5, method='ideal')
    0.0353658941301301
    >>> water_saturation(T=300.0, P=1e5, method='ASHRAE1485')
    0.03551758880618026
    
    >>> T, P  = 330, 92000.0
    >>> 1-water_saturation_ASHRAE1485(T, P)/HAPropsSI('psi_w', 'T', T, 'P', P, 'RH', 1)
    -7.299148780326448e-05
    
    References
    ----------
    .. [1] American Society of Heating, Refrigerating and Air-Conditioning 
       Engineers. ASHRAE Handbook - Fundamentals. 2015.
    '''
    Psat = Psat_IAPWS(T)
    x_w_ideal = Psat/P
    if method == 'ideal':
        factor = 1.0
    elif method == 'ASHRAE1485_2020' or method == 'ASHRAE1485':
        factor = float(bisplev(T, P, ASHRAE_RP1485_saturation_tck))
    elif method == 'CoolProp':
        return HAPropsSI('psi_w', 'T', T, 'P', P, 'RH', 1.0)
    else:
        raise ValueError("Unsupported method")
    return factor*x_w_ideal

def water_saturation_and_der(T, P, method='ideal'):
    Psat = Psat_IAPWS(T)
    der = dPsat_IAPWS_dT(T)/P
    x_w_ideal = Psat/P
    if method == 'ideal':
        factor = 1.0
        dfactor_dT = 0.0
    elif method == 'ASHRAE1485_2020' or method == 'ASHRAE1485':
        factor = float(bisplev(T, P, ASHRAE_RP1485_saturation_tck))
        dfactor_dT = float(sp_bisplev(T, P, ASHRAE_RP1485_saturation_tck, dx=1))
    elif method == 'CoolProp':
        x_w = HAPropsSI('psi_w', 'T', T, 'P', P, 'RH', 1.0)
        dx_w_dT = derivative(lambda T: HAPropsSI('psi_w', 'T', T, 'P', P, 'RH', 1.0), T, dx=T*1e-6, order=7)
        return x_w, dx_w_dT
    else:
        raise ValueError("Unsupported method")
    return x_w_ideal*factor, factor*der + dfactor_dT*x_w_ideal

def water_dew_T(T, P, RH, method='ideal'):
    x_water_actual = water_saturation(T, P)*RH
    def to_solve(T_dew):
        xw_calc, dxw_calc_dT = water_saturation_and_der(T_dew, P, method)
        return xw_calc - x_water_actual, dxw_calc_dT
    given_method = method
    method = 'ideal' # for initial guess
    T_dew = newton(to_solve, T, fprime=True)
    if given_method != 'ideal':
        method = given_method
        T_dew = newton(to_solve, T_dew, fprime=True)
    return T_dew

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
__all__ = ['CRC_inorganic_data', 'CRC_organic_data', '_VDISaturationDict', 
           'VDI_tabular_data']
import os
import copy
import pandas as pd
from thermo.utils import to_num, rho_to_Vm

folder = os.path.join(os.path.dirname(__file__), 'Misc')

### CRC Handbook general tables

CRC_inorganic_data = pd.read_csv(os.path.join(folder,
'Physical Constants of Inorganic Compounds.csv'), sep='\t', index_col=0)


CRC_organic_data = pd.read_csv(os.path.join(folder,
'Physical Constants of Organic Compounds.csv'), sep='\t', index_col=0)


### VDI Saturation

emptydict = {"Name": None, "MW": None, "Tc": None, "T": [], "P": [],
             "Density (l)": [], "Density (g)": [], "Hvap": [], "Cp (l)": [],
            "Cp (g)": [], "Mu (l)": [], "Mu (g)": [], "K (l)": [], "K (g)": [],
            "Pr (l)": [], "Pr (g)": [], "sigma": [], "Beta": [],
            "Volume (l)": [], "Volume (g)": []}

# After some consideration, it has been devided to keep this load method as is.

_VDISaturationDict = {}
with open(os.path.join(folder, 'VDI Saturation Compounds Data.csv')) as f:
    '''Read in a dict of assorted chemical properties at saturation for 58
    industrially important chemicals, from:
    Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2E. Berlin : Springer, 2010.
    This listing is the successor to that in:
    Schlunder, Ernst U, and International Center for Heat and Mass Transfer.
    Heat Exchanger Design Handbook. Washington: Hemisphere Pub. Corp., 1983.
    '''
    next(f)
    for line in f:
        values = to_num(line.strip('\n').split('\t'))
        (CASRN, _name, _MW, _Tc, T, P, rhol, rhog, Hvap, cpl, cpg, mul, mug, kl, kg, prl, prg, sigma, Beta) = values
        newdict = (_VDISaturationDict[CASRN] if CASRN in _VDISaturationDict else copy.deepcopy(emptydict))
        newdict["Name"] = _name
        newdict["MW"] = _MW
        newdict["Tc"] = _Tc
        newdict["T"].append(T)
        newdict["P"].append(P)
        newdict["Density (l)"].append(rhol)
        newdict["Density (g)"].append(rhog)  # Not actually used
        newdict["Hvap"].append(Hvap)
        newdict["Cp (l)"].append(cpl)  # Molar
        newdict["Cp (g)"].append(cpg)  # Molar
        newdict["Mu (l)"].append(mul)
        newdict["Mu (g)"].append(mug)
        newdict["K (l)"].append(kl)
        newdict["K (g)"].append(kg)
        newdict["Pr (l)"].append(prl)
        newdict["Pr (g)"].append(prl)
        newdict["sigma"].append(sigma)
        newdict["Beta"].append(Beta)
        newdict["Volume (l)"].append(rho_to_Vm(rhol, _MW))
        newdict["Volume (g)"].append(rho_to_Vm(rhog, _MW))
        _VDISaturationDict[CASRN] = newdict


def VDI_tabular_data(CASRN, prop):
    r'''This function retrieves the tabular data available for a given chemical
    and a given property. Lookup is based on CASRNs. Length of data returned
    varies between chemicals. All data is at saturation condition from [1]_.

    Function has data for 58 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]
    prop : string
        Property [-]

    Returns
    -------
    Ts : list
        Temperatures where property data is available, [K]
    props : list
        Properties at each temperature, [various]

    Notes
    -----
    The available properties are 'P', 'Density (l)', 'Density (g)', 'Hvap',
    'Cp (l)', 'Cp (g)', 'Mu (l)', 'Mu (g)', 'K (l)', 'K (g)', 'Pr (l)',
    'Pr (g)', 'sigma', 'Beta', 'Volume (l)', and 'Volume (g)'.

    Data is available for all properties and all chemicals; surface tension
    data was missing for mercury, but added as estimated from the a/b
    coefficients listed in Jasper (1972) to simplify the function.

    Examples
    --------
    >>> VDI_tabular_data('67-56-1', 'Mu (g)')
    ([337.63, 360.0, 385.0, 410.0, 435.0, 460.0, 500.0], [1.11e-05, 1.18e-05, 1.27e-05, 1.36e-05, 1.46e-05, 1.59e-05, 2.04e-05])

    References
    ----------
    .. [1] Gesellschaft, VDI, ed. VDI Heat Atlas. 2E. Berlin : Springer, 2010.
    '''
    try:
        d = _VDISaturationDict[CASRN]
    except KeyError:
        raise Exception('CASRN not in VDI tabulation')
    try:
        props, Ts = d[prop], d['T']
    except:
        raise Exception('Proprty not specified correctly')
    Ts = [T for p, T in zip(props, Ts) if p]
    props = [p for p in props if p]

    # Not all data series convererge to correct values
    if prop == 'sigma':
        Ts.append(d['Tc'])
        props.append(0)
    return Ts, props
#print VDI_tabular_data('67-56-1', 'Mu (g)')
# Mercury surface tension is missing.

#'''
#Linear: k(l) Done
#linear in logarighmic, x = ln(Tc-T), for mu(g) and k(g). Done
#Log in Log, x = ln(Tc-T) and y = ln(property) for rho(l), Hvap, Cp(l), cp(g), Pr(l), Pr(g), sigma, Beta # 5 done
#Linear in inverse temperature x = T^-1 for mu(l) # Done
#Logrithmic in inverse temperature for vapor pressure, vapor density # Done
#'''


#def interp_VDI_lin(T, CASRN, prop=None):
#    '''
#    Linear: k(l)
#    '''
#    d = _VDISaturationDict[CASRN]
#    dT = d["T"]
#    dA = d[prop]
#    Ts = [dT[i] for i in range(len(dA)) if dA[i]]
#    if T < Ts[0] or T > Ts[-1]:
#        return None # Outside of bounds, no interpolation possible
#    As = [dA[i] for i in range(len(dA)) if dA[i]]
#    A_interp = interp1d(Ts, As)
#    A = float(A_interp(T))
#    return A
#
#
#def interp_VDI_logx(T, CASRN, prop=None):
#    '''linear in logarighmic, x = ln(Tc-T), for mu(g) and k(g).
#
#    >>> interp_VDI_logx(450, CASRN='67-56-1', prop='K (g)')
#    0.0382872097941
#    '''
#    d = _VDISaturationDict[CASRN]
#    Tc = d["Tc"]+1E-6 # Allow interpolation to the critical point without math error
#    dT = d["T"]
#    dA = d[prop]
#    Ts = [dT[i] for i in range(len(dA)) if dA[i]]
#    if T < Ts[0] or T > Ts[-1] or T >= Tc:
#        return None # Outside of bounds, no interpolation possible
#    mod_Ts = [log(Tc - Ti) for Ti in Ts]
#    As = [dA[i] for i in range(len(dA)) if dA[i]]
#    A_interp = interp1d(mod_Ts, As)
#    A = float(A_interp(log(Tc-T)))
#    return A


#def interp_VDI_logx_logy(T, CASRN, prop=None):
#    '''Log in Log, x = ln(Tc-T) and y = ln(property) for rho(l), Hvap, Cp(l), cp(g), Pr(l), Pr(g), sigma, Beta
#
#    Sigma for mercury doesn't work.
#
#    >>> interp_VDI_logx_logy(245, CASRN='67-56-1', prop='Density (l)')
#    837.167593918
#    '''
#    d = _VDISaturationDict[CASRN]
#    Tc = d["Tc"]+1E-6 # Allow interpolation to the critical point without math error
#    dT = d["T"]
#    dA = d[prop]
#    Ts = [dT[i] for i in range(len(dA)) if dA[i]]
#    if not Ts or T < Ts[0] or T > Ts[-1]:
#        return None # Outside of bounds, no interpolation possible, or no values
#    mod_Ts = [log(Tc-Ti) for Ti in Ts]
#    As = [dA[i] for i in range(len(dA)) if dA[i]]
#    mod_As = [log(i) for i in As]
#    modA_interp = interp1d(mod_Ts, mod_As)
#    A = float(exp(modA_interp(log(Tc-T))))
#    return A


#def interp_VDI_invx(T, CASRN, prop=None):
#    '''Linear in inverse temperature x = T^-1 for mu(l)
#
#    >>> interp_VDI_invx(250, CASRN='67-56-1', prop='Mu (l)') #230 to 500
#    0.001362704
#    '''
#    d = _VDISaturationDict[CASRN]
#    dT = d["T"]
#    dA = d[prop]
#    Ts = [dT[i] for i in range(len(dA)) if dA[i]]
#    if T < Ts[0] or T > Ts[-1]:
#        return None # Outside of bounds, no interpolation possible
#    mod_Ts = [1./Ti for Ti in Ts]
#    As = [dA[i] for i in range(len(dA)) if dA[i]]
#    A_interp = interp1d(mod_Ts, As)
#    A = float(A_interp(1./T))
#    return A
#
#
#def interp_VDI_loginvx(T, CASRN, prop=None):
#    '''Logrithmic in inverse temperature for vapor pressure, vapor density
#
#    >>> interp_VDI_loginvx(250, CASRN='67-56-1', prop='P')
#    796.567071759
#    '''
#    d = _VDISaturationDict[CASRN]
#    dT = d["T"]
#    dA = d[prop]
#    Ts = [dT[i] for i in range(len(dA)) if dA[i]]
#    if T < Ts[0] or T > Ts[-1]:
#        return None # Outside of bounds, no interpolation possible
#    mod_Ts = [(1./Ti) for Ti in Ts]
#    As = [dA[i] for i in range(len(dA)) if dA[i]]
#    mod_As = [log(i) for i in As]
#    A_interp = interp1d(mod_Ts, mod_As)
#    A = float(exp(A_interp((1./T))))
#    return A





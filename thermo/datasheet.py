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

__all__ = ['tabulate_solid', 'tabulate_liq', 'tabulate_gas', 
           'tabulate_constants']
           
from collections import OrderedDict
import numpy as np
import pandas as pd
from thermo.chemical import Chemical


def tabulate_solid(chemical, Tmin=None, Tmax=None, pts=10):
    chem = Chemical(chemical)

    (rhos, Cps) = [[] for i in range(2)]
    if not Tmin:  # pragma: no cover
        if chem.Tm:
            Tmin = chem.Tm-100
        else:
            Tmin = 150.
    if not Tmax:  # pragma: no cover
        if chem.Tm:
            Tmax = chem.Tm
        else:
            Tmax = 350

    Ts = np.linspace(Tmin, Tmax, pts)
    for T in Ts:
        chem = Chemical(chemical, T=T)
        rhos.append(chem.rhos)
        Cps.append(chem.Cps)

    data = OrderedDict()
    data['Density, kg/m^3'] = rhos
    data['Constant-pressure heat capacity, J/kg/K'] = Cps

    df = pd.DataFrame(data, index=Ts)
    df.index.name = 'T, K'
    return df


def tabulate_liq(chemical, Tmin=None, Tmax=None, pts=10):
    chem = Chemical(chemical)

    (rhos, Cps, mugs, kgs, Prs, alphas, isobarics, JTs, Psats, sigmas, Hvaps,
     permittivities) = [[] for i in range(12)]
    if not Tmin:  # pragma: no cover
        if chem.Tm:
            Tmin = chem.Tm
        else:
            Tmin = 273.15
    if not Tmax:  # pragma: no cover
        if chem.Tc:
            Tmax = chem.Tc
        else:
            Tmax = 450

    Ts = np.linspace(Tmin, Tmax, pts)
    for T in Ts:
        chem = Chemical(chemical, T=T)

        rhos.append(chem.rhol)
        Cps.append(chem.Cpl)
        mugs.append(chem.mul)
        kgs.append(chem.kl)
        Prs.append(chem.Prl)
        alphas.append(chem.alphal)
        isobarics.append(chem.isobaric_expansion_l)
        JTs.append(chem.JTg)
        Psats.append(chem.Psat)
        Hvaps.append(chem.Hvap)
        sigmas.append(chem.sigma)
        permittivities.append(chem.permittivity)

    data = OrderedDict()
    data['Saturation pressure, Pa'] = Psats
    data['Density, kg/m^3'] = rhos
    data['Constant-pressure heat capacity, J/kg/K'] = Cps
    data['Heat of vaporization, J/kg'] = Hvaps
    data['Viscosity, Pa*S'] = mugs
    data['Thermal consuctivity, W/m/K'] = kgs
    data['Surface tension, N/m'] = sigmas
    data['Prandtl number'] = Prs
    data['Thermal diffusivity, m^2/s'] = alphas
    data['Isobaric expansion, 1/K'] = isobarics
    data['Joule-Thompson expansion coefficient, K/Pa'] = JTs
    data['Permittivity'] = permittivities

    df = pd.DataFrame(data, index=Ts)
    df.index.name = 'T, K'
    return df


def tabulate_gas(chemical, Tmin=None, Tmax=None, pts=10):
    chem = Chemical(chemical)

    (rhos, Cps, Cvs, mugs, kgs, Prs, alphas, isobarics, isentropics, JTs) = [[] for i in range(10)]
    if not Tmin:  # pragma: no cover
        if chem.Tm:
            Tmin = chem.Tm
        else:
            Tmin = 273.15
    if not Tmax:  # pragma: no cover
        if chem.Tc:
            Tmax = chem.Tc
        else:
            Tmax = 450

    Ts = np.linspace(Tmin, Tmax, pts)
    for T in Ts:
        chem = Chemical(chemical, T=T)

        rhos.append(chem.rhog)
        Cps.append(chem.Cpg)
        Cvs.append(chem.Cvg)
        mugs.append(chem.mug)
        kgs.append(chem.kg)
        Prs.append(chem.Prg)
        alphas.append(chem.alphag)
        isobarics.append(chem.isobaric_expansion_g)
        isentropics.append(chem.isentropic_exponent)
        JTs.append(chem.JTg)
    data = OrderedDict()
    data['Density, kg/m^3'] = rhos
    data['Constant-pressure heat capacity, J/kg/K'] = Cps
    data['Constant-volume heat capacity, J/kg/K'] = Cvs
    data['Viscosity, Pa*S'] = mugs
    data['Thermal consuctivity, W/m/K'] = kgs
    data['Prandtl number'] = Prs
    data['Thermal diffusivity, m^2/s'] = alphas
    data['Isobaric expansion, 1/K'] = isobarics
    data['Isentropic exponent'] = isentropics
    data['Joule-Thompson expansion coefficient, K/Pa'] = JTs

    df = pd.DataFrame(data, index=Ts)  # add orient='index'
    df.index.name = 'T, K'
    return df


def tabulate_constants(chemical, full=False, vertical=False):
    pd.set_option('display.max_rows', 100000)
    pd.set_option('display.max_columns', 100000)

    all_chemicals = OrderedDict()

    if isinstance(chemical, str):
        cs = [chemical]
    else:
        cs = chemical

    for chemical in cs:
        chem = Chemical(chemical)
        data = OrderedDict()
        data['CAS'] = chem.CAS
        data['Formula'] = chem.formula
        data['MW, g/mol'] = chem.MW
        data['Tm, K'] = chem.Tm
        data['Tb, K'] = chem.Tb
        data['Tc, K'] = chem.Tc
        data['Pc, Pa'] = chem.Pc
        data['Vc, m^3/mol'] = chem.Vc
        data['Zc'] = chem.Zc
        data['rhoc, kg/m^3'] = chem.rhoc
        data['Acentric factor'] = chem.omega
        data['Triple temperature, K'] = chem.Tt
        data['Triple pressure, Pa'] = chem.Pt
        data['Heat of vaporization at Tb, J/mol'] = chem.Hvap_Tbm
        data['Heat of fusion, J/mol'] = chem.Hfusm
        data['Heat of sublimation, J/mol'] = chem.Hsubm
        data['Heat of formation, J/mol'] = chem.Hf
        data['Dipole moment, debye'] = chem.dipole
        data['Molecular Diameter, Angstrom'] = chem.molecular_diameter
        data['Stockmayer parameter, K'] = chem.Stockmayer
        data['Refractive index'] = chem.RI
        data['Lower flammability limit, fraction'] = chem.LFL
        data['Upper flammability limit, fraction'] = chem.UFL
        data['Flash temperature, K'] = chem.Tflash
        data['Autoignition temperature, K'] = chem.Tautoignition
        data['Time-weighted average exposure limit'] = str(chem.TWA)
        data['Short-term exposure limit'] = str(chem.STEL)
        data['logP'] = chem.logP

        if full:
            data['smiles'] = chem.smiles
            data['InChI'] = chem.InChI
            data['InChI key'] = chem.InChI_Key
            data['IUPAC name'] = chem.IUPAC_name
            data['solubility parameter, Pa^0.5'] = chem.solubility_parameter
            data['Parachor'] = chem.Parachor
            data['Global warming potential'] = chem.GWP
            data['Ozone depletion potential'] = chem.ODP
            data['Electrical conductivity, S/m'] = chem.conductivity

        all_chemicals[chem.name] = data

    if vertical:
        df = pd.DataFrame.from_dict(all_chemicals)
    else:
        df = pd.DataFrame.from_dict(all_chemicals, orient='index')
    return df




#chemicals = ['Sodium Hydroxide', 'sodium chloride', 'methanol',
#'hydrogen sulfide', 'methyl mercaptan', 'Dimethyl disulfide', 'dimethyl sulfide',
# 'alpha-pinene', 'chlorine dioxide', 'sulfuric acid', 'SODIUM CHLORATE', 'carbon dioxide', 'Cl2', 'formic acid',
# 'sodium sulfate']
#for i in chemicals:
#    print tabulate_solid(i)
#    print tabulate_liq(i)
#    print tabulate_gas(i)
#    tabulate_constants(i)

#tabulate_constants('Methylene blue')
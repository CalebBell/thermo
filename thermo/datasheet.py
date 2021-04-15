# -*- coding: utf-8 -*-
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
SOFTWARE.'''

from __future__ import division

__all__ = ['tabulate_solid', 'tabulate_liq', 'tabulate_gas',
           'tabulate_constants', 'tabulate_streams']

from collections import OrderedDict
from fluids.numerics import numpy as np
from thermo.chemical import Chemical

global _pd
_pd = None
def pandas():
    global _pd
    if _pd is None:
        import pandas as _pd
    return _pd
def tabulate_solid(chemical, Tmin=None, Tmax=None, pts=10):
    pd = pandas()
    chem = Chemical(chemical)

    (rhos, Cps) = [[] for i in range(2)]
    if not Tmin:  # pragma: no cover
        if chem.Tm:
            Tmin = min(chem.Tm-100, 1e-2)
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
    pd = pandas()
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
    data['Viscosity, Pa*s'] = mugs
    data['Thermal conductivity, W/m/K'] = kgs
    data['Surface tension, N/m'] = sigmas
    data['Prandtl number'] = Prs
    data['Thermal diffusivity, m^2/s'] = alphas
    data['Isobaric expansion, 1/K'] = isobarics
    data['Joule-Thompson expansion coefficient, K/Pa'] = JTs
    data['PermittivityLiquid'] = permittivities

    df = pd.DataFrame(data, index=Ts)
    df.index.name = 'T, K'
    return df


def tabulate_gas(chemical, Tmin=None, Tmax=None, pts=10):
    chem = Chemical(chemical)
    pd = pandas()

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
    data['Viscosity, Pa*s'] = mugs
    data['Thermal conductivity, W/m/K'] = kgs
    data['Prandtl number'] = Prs
    data['Thermal diffusivity, m^2/s'] = alphas
    data['Isobaric expansion, 1/K'] = isobarics
    data['Isentropic exponent'] = isentropics
    data['Joule-Thompson expansion coefficient, K/Pa'] = JTs

    df = pd.DataFrame(data, index=Ts)  # add orient='index'
    df.index.name = 'T, K'
    return df


def tabulate_constants(chemical, full=False, vertical=False):
    pd = pandas()
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


def tabulate_streams(names=None, *args, **kwargs):
    # Names are the names of the streams to be displayed; input
    # strings for each of them or bad things happen!
    pd = pandas()
    Ts = [i.T for i in args]
    Ps = [i.P for i in args]
    VFs = [i.V_over_F for i in args]
    phases = [i.phase for i in args]
    Hs = [i.H for i in args]
    ms = [i.m for i in args]
    ns = [i.n for i in args]
    CASs = set()
    IDs = {}
    for stream in args:
        CASs.update(stream.CASs)
        for CAS, i in zip(stream.CASs, stream.names):
            IDs[CAS] = i
    CASs = list(CASs) # So it can be indexed

    mole_fractions = []
    mole_flows = []
    mass_fractions = []
    mass_flows = []
    for stream in args:
        mole_fractions_i = []
        mass_fractions_i = []
        mole_flows_i = []
        mass_flows_i = []
        for CAS in CASs:
            if CAS in stream.CASs:
                ind = stream.CASs.index(CAS)
                zi = stream.zs[ind]
                wi = stream.ws[ind]
                n = stream.ns[ind]
                m = stream.ms[ind]
            else:
                zi, wi, n, m = 0, 0, 0, 0
            mole_fractions_i.append(zi)
            mass_fractions_i.append(wi)
            mole_flows_i.append(n)
            mass_flows_i.append(m)
        mole_fractions.append(mole_fractions_i)
        mass_fractions.append(mass_fractions_i)
        mass_flows.append(mass_flows_i)
        mole_flows.append(mole_flows_i)

    dat = OrderedDict([['Temperature, K', Ts],
                      ['Pressure, Pa', Ps],
                       ['Phase', phases],
                      ['Vapor fraction', VFs],
                      ['Enthalpy, J', Hs],
                      ['Mass flows, kg/s', ms],
                      ['Mole flows, mol/s', ns]])

    if kwargs.get('Mole flows', True):
        for i, CAS in enumerate(CASs):
            s = 'Mole flow, mol/s %s' %IDs[CAS]
            vals = [j[i] for j in mole_flows]
            dat[s] = vals

    if kwargs.get('Mass flows', True):
        for i, CAS in enumerate(CASs):
            s = 'Mass flow, kg/s %s' %IDs[CAS]
            vals = [j[i] for j in mass_flows]
            dat[s] = vals

    if kwargs.get('Mass fractions', True):
        for i, CAS in enumerate(CASs):
            s = 'Mass fraction %s' %IDs[CAS]
            vals = [j[i] for j in mass_fractions]
            dat[s] = vals

    if kwargs.get('Mole fractions', True):
        for i, CAS in enumerate(CASs):
            s = 'Mole fraction %s' %IDs[CAS]
            vals = [j[i] for j in mole_fractions]
            dat[s] = vals

#    print(dat, names)
    if names is None:
        df = pd.DataFrame(dat)
    else:
        df = pd.DataFrame(dat, index=names)
    return df.transpose()




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

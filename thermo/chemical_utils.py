'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
'''


__all__ = ['standard_entropy', 'S0_basis_converter', 'standard_state_ideal_gas_formation']

from fluids.numerics import quad
from chemicals.reaction import standard_formation_reaction
from thermo.heat_capacity import HeatCapacitySolid, HeatCapacityLiquid, HeatCapacityGas
from chemicals.elements import periodic_table

def standard_entropy(c=None, dS_trans_s=None, dH_trans_s=None, T_trans_s=None,
                     Cp_s_fun=None,
                     Sfusm=None, Hfusm=None, Tm=None, Cp_l_fun=None,
                     Svapm=None, Hvapm=None, Tb=None, Cp_g_fun=None,
                     T_ref=298.15, T_low=0.5, force_gas=True):
    if Tm is None:
        Tm = c.Tm
    if Tb is None:
        Tb = c.Tb
    if Hfusm is None:
        Hfusm = c.Hfusm
    if Hvapm is None:
        Hvapm = c.EnthalpyVaporization(Tb)

    # Misc crystalline transitions
    tot = 0.0
    if dS_trans_s is not None:
        tot += sum(dS_trans_s)
    if dH_trans_s is not None and T_trans_s is not None:
        for dH, T in zip(dH_trans_s, T_trans_s):
            if T < T_ref or force_gas:
                tot += dH/T

    # Solid heat capacity integral
    if Cp_s_fun is not None:
        tot += float(quad(lambda T: Cp_s_fun(T)/T, T_low, Tm)[0])
    else:
        tot += c.HeatCapacitySolid.T_dependent_property_integral_over_T(T_low, Tm)

    # Heat of fusion
    if force_gas or Tm < T_ref:
        if Sfusm is not None:
            tot += Sfusm
        else:
            tot += Hfusm/Tm

    # Liquid heat capacity
    if not force_gas and Tb > T_ref:
        T_liquid_int = T_ref
    else:
        T_liquid_int = Tb
    if force_gas or Tm < T_ref:
        if Cp_l_fun is not None:
            tot += float(quad(lambda T: Cp_l_fun(T)/T, Tm, T_liquid_int)[0])
        else:
            tot += c.HeatCapacityLiquid.T_dependent_property_integral_over_T(Tm, T_liquid_int)

    # Heat of vaporization
    if force_gas or Tb < T_ref:
        if Svapm is not None:
            tot += Svapm
        else:
            tot += Hvapm/Tb

    if force_gas or Tb < T_ref:
        # gas heat capacity
        if Cp_g_fun is not None:
            tot += float(quad(lambda T: Cp_g_fun(T)/T, Tb, T_ref)[0])
        else:
            tot += c.HeatCapacityGas.T_dependent_property_integral_over_T(Tb, T_ref)

    return tot


def S0_basis_converter(c, S0_liq=None, S0_gas=None, T_ref=298.15):
    r'''This function converts a liquid or gas standard entropy to the
    other. This is useful, as thermodynamic packages often work with ideal-
    gas as the reference state and require ideal-gas Gibbs energies of
    formation.

    Parameters
    ----------
    c : Chemical
        Chemical object, [-]
    S0_liq : float, optional
        Liquid absolute entropy of the compound at the reference temperature
        [J/mol/K]
    S0_gas : float, optional
        Gas absolute entropy of the compound at the reference temperature
        [J/mol/K]
    T_ref : float, optional
        The standard state temperature, default 298.15 K; few values are
        tabulated at other temperatures, [-]

    Returns
    -------
    S0_calc : float
        Standard absolute entropy of the compound at the reference temperature
        in the other state to the one provided, [J/mol]

    Notes
    -----
    This function relies in accurate heat capacity curves for both the liquid
    and gas state.

    Examples
    --------
    >>> from thermo.chemical import Chemical
    >>> S0_basis_converter(Chemical('decane'), S0_liq=425.89) # doctest:+SKIP
    544.6792
    >>> S0_basis_converter(Chemical('decane'), S0_gas=545.7) # doctest:+SKIP
    426.9107
    '''
    if S0_liq is None and S0_gas is None:
        raise ValueError("Provide either a liquid or a gas standard absolute entropy")
    if S0_liq is None:
        dS = c.HeatCapacityGas.T_dependent_property_integral_over_T(T_ref, c.Tb)
        dS -= c.EnthalpyVaporization(c.Tb)/c.Tb
        dS += c.HeatCapacityLiquid.T_dependent_property_integral_over_T(c.Tb, T_ref)
        return S0_gas + dS
    else:
        dS = c.HeatCapacityLiquid.T_dependent_property_integral_over_T(T_ref, c.Tb)
        dS += c.EnthalpyVaporization(c.Tb)/c.Tb
        dS += c.HeatCapacityGas.T_dependent_property_integral_over_T(c.Tb, T_ref)
        return S0_liq + dS


standard_state_transitions = {
    'Si': {'Tm': 1687.15, 'Tb': 3504.616, 'Hfus': 50210.0, 'Hvap': 384548.0},
    'Br': {'Tm': 265.900, 'Tb': 332.503, 'Hfus': 10571, 'Hvap': 29563},
    'I': {'Tm': 386.750, 'Tb': 457.666, 'Hfus': 15517, 'Hvap': 41960},
    'B': {'Tm': 2350.000, 'Tb': 4139.449, 'Hfus': 50208, 'Hvap': 480509},
    'Mg': {'Tm': 923.000, 'Tb': 1366.104, 'Hfus': 8477, 'Hvap': 127867},
    'Hg': {'Tm': 234.290, 'Tb': 629.839, 'Hfus': 2295, 'Hvap': 59205},
    'Pb': {'Tm': 600.600, 'Tb': 2019.022, 'Hfus': 4774, 'Hvap': 177582},
    'Li': {'Tm': 453.690, 'Tb': 1620.120, 'Hfus': 3000, 'Hvap': 145843},
    'Na': {'Tm': 370.980, 'Tb': 1170.525, 'Hfus': 2603, 'Hvap': 97022},
    'Al': {'Tm': 933.450, 'Tb': 2790.812, 'Hfus': 10711, 'Hvap': 294002},
    'K': {'Tm': 336.350, 'Tb': 1039.540, 'Hfus': 2334, 'Hvap': 79556},
    'V': {'Tm': 2190.000, 'Tb': 3690.080, 'Hfus': 22845, 'Hvap': 446977},
    'Cr': {'Tm': 2130.000, 'Tb': 2952.078, 'Hfus': 20502, 'Hvap': 339537},
    # 'Mn': {'Tm': 1519.000, 'Tb': 2334.526, 'Hfus': 12058, 'Hvap': 225980,  'solid_T_trans': [980.000, 1361.000, 1412.000], 'solid_H_trans': [2226, 2122, 1879]}, # Double TODO,
    # 'Ti': {'Tm': 1939.000, 'Tb': 3630.956, 'Hfus': 14146, 'Hvap': 409984,  'solid_T_trans': [1166.000], 'solid_H_trans': [4172]}, # TODO
    # 'Ca': {'Tm': 1115.000, 'Tb': 1773.658, 'Hfus': 8540, 'Hvap': 149047,  'solid_T_trans': [716.000], 'solid_H_trans': [930]}, # TODO
    # 'S': {'Tm': 388.360, 'Tb': 882.117, 'Hfus': 1721, 'Hvap': 53326,  'solid_T_trans': [368.300], 'solid_H_trans': [401]}, # TODO
    # 'Be': {'Tm': 1560.000, 'Tb': 2741.437, 'Hfus': 7895, 'Hvap': 291572, 'solid_T_trans': [1527.000], 'solid_H_trans': [6849]}, # TODO
    # 'Fe': {'Tm': 3133.345, 'Hfus': 349585, 'solid_T_trans': [1184.000, 1665.000], 'solid_H_trans': [900, 837]}, # TODO
    
    }

def standard_state_ideal_gas_formation(c, T, Hf=None, Sf=None, T_ref=298.15):
    # Whatever the compound is, it is assumed to be in the standard state
    # not that this should not be called on elements
    # Can check against JANAF
    Hf_ref = Hf if Hf is not None else c.Hfgm
    Sf_ref = Sf if Sf is not None else c.Sfgm
    atoms = c.atoms
    reactant_coeff, elemental_counts, elemental_composition = standard_formation_reaction(atoms)
    
    dH_compound = c.HeatCapacityGas.T_dependent_property_integral(T_ref, T)
    dS_compound = c.HeatCapacityGas.T_dependent_property_integral_over_T(T_ref, T)
    
    H_calc = reactant_coeff*Hf_ref + reactant_coeff*dH_compound
    S_calc = reactant_coeff*Sf_ref + reactant_coeff*dS_compound
    # if the compound is an element it will need special handling to go from solid liquid to gas if needed

    solid_ele = set(['C'])
    liquid_ele = set([''])
    
    for coeff, ele_data in zip(elemental_counts, elemental_composition):
        ele = list(ele_data.keys())[0]
        element_obj = periodic_table[ele]
#         element = Chemical(element_obj.CAS_standard)
        solid_obj = HeatCapacitySolid(CASRN=element_obj.CAS_standard)
        liquid_obj = HeatCapacityLiquid(CASRN=element_obj.CAS_standard)
        gas_obj = HeatCapacityGas(CASRN=element_obj.CAS_standard)
        if ele in ('H', 'O', 'N', 'F', 'P', 'Cl', 'Br', 'I', 'Mg', 'B', 'Pb', 'Li', 'Na', 'Al', 'K', 'V', 'Cr'):
            gas_obj.method = 'WEBBOOK_SHOMATE'
        elif ele == 'Si':
            solid_obj.method = 'JANAF'
            liquid_obj.method = 'JANAF'
            gas_obj.method = 'JANAF'

        if ele in standard_state_transitions:
            dat = standard_state_transitions[ele]
            Tm = dat['Tm']
            Tb = dat['Tb']
            Hfus = dat['Hfus']
            Hvap = dat['Hvap']
            Tm_solid_int = min(T, Tm)
            T_liquid_int = min(T, Tb)
            dH_ele = solid_obj.T_dependent_property_integral(T_ref, Tm_solid_int)
            dS_ele = solid_obj.T_dependent_property_integral_over_T(T_ref, Tm_solid_int)
            if T > Tm:
                dH_ele += Hfus
                dS_ele += Hfus/Tm
                dH_ele += liquid_obj.T_dependent_property_integral(Tm, T_liquid_int)
                dS_ele += liquid_obj.T_dependent_property_integral_over_T(Tm, T_liquid_int)
            if T > Tb:
                dH_ele += Hvap
                dS_ele += Hvap/Tb
                dH_ele += gas_obj.T_dependent_property_integral(Tb, T)
                dS_ele += gas_obj.T_dependent_property_integral_over_T(Tb, T)
        elif ele == 'P':
            # White phosphorus is the basis here
            T_alpha_beta_P = 195.400
            Htrans_alpha_beta_P = 521.0 # 525.5104 reported in 
            # The thermodynamic properties of elementary phosphorus The heat capacities of two crystalline modifications of red phosphorus, of α and β white phosphorus, and of black phosphorus from 15 to 300 K
            Tm_P = 317.300
            Hfus_P = 659
            Tb_P = 1180.008
            Hvap_P = 63728.0
            # https://janaf.nist.gov/tables/P-001.html
            # ALPHA <--> BETA 195.4 K, BETA <--> LIQUID 317.3 K, LIQUID <--> IDEAL GAS 1180.008 K
            T_solid_int0 = min(T, T_alpha_beta_P)
            T_solid_int1 = min(T, Tm_P)
            T_liquid_int = min(T, Tb_P)
            if T < T_alpha_beta_P:
                dH_ele = solid_obj.T_dependent_property_integral(T_ref, T)
                dS_ele = solid_obj.T_dependent_property_integral_over_T(T_ref, T)

                dH_ele -= Htrans_alpha_beta_P
                dS_ele -= Htrans_alpha_beta_P/T_alpha_beta_P
                # dH_ele -= solid_obj.T_dependent_property_integral(T_alpha_beta_P, T_liquid_int)
                # dS_ele -= solid_obj.T_dependent_property_integral_over_T(Tm_P, T_liquid_int)
            else:
                dH_ele = solid_obj.T_dependent_property_integral(T_ref, T_solid_int1)
                dS_ele = solid_obj.T_dependent_property_integral_over_T(T_ref, T_solid_int1)
                if T > Tm_P:
                    dH_ele += Hfus_P
                    dS_ele += Hfus_P/Tm_P
                    dH_ele += liquid_obj.T_dependent_property_integral(Tm_P, T_liquid_int)
                    dS_ele += liquid_obj.T_dependent_property_integral_over_T(Tm_P, T_liquid_int)
                if T > Tb_P:
                    dH_ele += Hvap_P
                    dS_ele += Hvap_P/Tb_P
                    dH_ele += gas_obj.T_dependent_property_integral(Tb_P, T)
                    dS_ele += gas_obj.T_dependent_property_integral_over_T(Tb_P, T)
        elif ele == 'S':
            # CRystal II to Crystal 1 at 368 K
            # crystal I to liquid at 388 K
            # 432 K liquid-liquid lambda transition
            # 882 K liquid to ideal gas transition
            raise NotImplementedError
        # Need to do all the metals with no fancy phases at once generically
        # https://janaf.nist.gov/tables/Ni-001.html
        # https://janaf.nist.gov/tables/Cu-001.html
        # https://janaf.nist.gov/tables/Zn-001.html

        elif ele in solid_ele:
            dH_ele = solid_obj.T_dependent_property_integral(T_ref, T)
            dS_ele = solid_obj.T_dependent_property_integral_over_T(T_ref, T)
        elif ele in liquid_ele:
            dH_ele = liquid_obj.T_dependent_property_integral(T_ref, T)
            dS_ele = liquid_obj.T_dependent_property_integral_over_T(T_ref, T)
        else:
            dH_ele = gas_obj.T_dependent_property_integral(T_ref, T)
            dS_ele = gas_obj.T_dependent_property_integral_over_T(T_ref, T)
        H_calc -= coeff*dH_ele
        S_calc -= coeff*dS_ele
    G_calc = H_calc - T*S_calc
    
    H_calc, S_calc, G_calc = H_calc/reactant_coeff, S_calc/reactant_coeff, G_calc/reactant_coeff
    
    return H_calc, S_calc, G_calc

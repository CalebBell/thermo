'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2024 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains an implementation of the PPR78
group-contribution method.
This functionality requires the RDKit library to work.

.. contents:: :local:

.. autofunction:: thermo.group_contribution.PPR78_kij
.. autofunction:: thermo.group_contribution.PPR78_kijs

'''
__all__ = ['PPR78_kij', 'PPR78_kijs', 'PPR78_GROUP_IDS', 'PPR78_INTERACTIONS', 'PPR78_GROUPS']

from thermo.unifac import priority_from_atoms
from math import sqrt
from fluids.constants import R

SINGLE_BOND = 'single'
DOUBLE_BOND = 'double'
TRIPLE_BOND = 'triple '
AROMATIC_BOND = 'aromatic'

GROUP_ID_COUNTER = 1



class PPR78GroupContribution:
    __slots__ = ('group', 'group_id', 'atoms', 'bonds', 'smarts', 'priority', 'hydrogen_from_smarts', 'smart_rdkit')

    def __init__(self, group, atoms=None, bonds=None, smarts=None, priority=None, hydrogen_from_smarts=False):
        global GROUP_ID_COUNTER
        self.group = group
        self.atoms = atoms
        self.bonds = bonds
        self.smarts = smarts
        self.priority = priority
        self.group_id = GROUP_ID_COUNTER
        self.hydrogen_from_smarts = hydrogen_from_smarts
        self.smart_rdkit = None
        GROUP_ID_COUNTER += 1

    def __repr__(self):
        return f"GroupContribution(group={self.group!r}, atoms={self.atoms!r}, bonds={self.bonds!r})"

PPR78_GROUPS_BY_ID = {}
PPR78_GROUPS = {}
# The order of these statements is sensitive
PPR78_GROUPS['CH3'] = PPR78GroupContribution('CH3',  smarts='[CX4;H3]', atoms={'C': 1, 'H': 3}, bonds={})
PPR78_GROUPS['CH2'] = PPR78GroupContribution('CH2', smarts='[CX4;H2]', atoms={'C': 1, 'H': 2}, bonds={})
PPR78_GROUPS['CH'] = PPR78GroupContribution('CH', smarts='[CX4;H1]', atoms={'C': 1, 'H': 1}, bonds={})
PPR78_GROUPS['C'] = PPR78GroupContribution('C', smarts='[CX4;H0]', atoms={'C': 1, 'H': 0}, bonds={})
# Specifically methane
PPR78_GROUPS['CH4'] = PPR78GroupContribution('CH4', atoms={'C': 1, 'H': 4}, bonds={}, smarts='[CX4H4]')
# Specifically ethane
PPR78_GROUPS['C2H6'] = PPR78GroupContribution('C2H6', atoms={'C': 2, 'H': 6}, bonds={}, smarts='[CH3X4]-[CH3X4]')
# messy ones, did some examples with all of these and they did make sense to me
PPR78_GROUPS['CHaro'] = PPR78GroupContribution('CHaro', atoms={'C': 1, 'H': 1}, bonds={AROMATIC_BOND: 2}, smarts='[cH]')
PPR78_GROUPS['Caro'] = PPR78GroupContribution('Caro', atoms={'C': 1}, bonds={AROMATIC_BOND: 2}, smarts='[c]')
PPR78_GROUPS['Cfused_aromatic'] = PPR78GroupContribution('Cfused_aromatic', atoms={'C': 1}, bonds={AROMATIC_BOND: 3}, smarts='[cR2]', priority=10000)
PPR78_GROUPS['CH2cyclic'] = PPR78GroupContribution('CH2cyclic', atoms={'C': 1, 'H': 2}, bonds={}, smarts='[CH2R]')
PPR78_GROUPS['CHcyclic'] = PPR78GroupContribution('CHcyclic', atoms={'C': 1, 'H': 1}, bonds={}, smarts='[CHR]')
# CO2, N2, H2S from PSRK
PPR78_GROUPS['CO2'] = PPR78GroupContribution('CO2', atoms={'C': 1, 'O': 2}, bonds={DOUBLE_BOND: 2}, smarts='[CX2H0](=[OX1H0])=[OX1H0]')
PPR78_GROUPS['N2'] = PPR78GroupContribution('N2', atoms={'N': 2}, bonds={TRIPLE_BOND: 1}, smarts='N#N')
PPR78_GROUPS['H2S'] = PPR78GroupContribution('H2S', atoms={'S': 1, 'H': 2}, bonds={}, smarts='[SH2]')

PPR78_GROUPS['SH'] = PPR78GroupContribution('SH', atoms={'S': 1, 'H': 1}, bonds={}, smarts='[S;H1]')

# Populate PPR78_GROUPS_BY_ID for easy lookup
for group in PPR78_GROUPS.values():
    if group.priority is None:
        if group.atoms is not None:
            group.priority = priority_from_atoms(group.atoms, group.bonds)
    PPR78_GROUPS_BY_ID[group.group_id] = group

catalog = PPR78_GROUPS.values()


# format A, B value indexed by tuple of groups
PPR78_INTERACTIONS = {}
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['CH3'])] = (0.0, 0.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['CH2'])] = (74.81, 165.7)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['CH'])] = (261.5, 388.8)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['C'])] = (396.7, 804.3)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['CH4'])] = (32.94, -35.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['C2H6'])] = (8.579, -29.51)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['CHaro'])] = (90.25, 146.1)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['Caro'])] = (62.80, 41.86)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['Cfused_aromatic'])] = (62.80, 41.86)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['CH2cyclic'])] = (40.38, 95.90)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['CHcyclic'])] = (98.48, 231.6)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['CO2'])] = (164.0, 269.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['N2'])] = (52.74, 87.19)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['H2S'])] = (158.4, 241.2)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH3'], PPR78_GROUPS['SH'])] = (799.9, 2109.0)

PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['CH2'])] = (0.0, 0.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['CH'])] = (51.47, 79.61)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['C'])] = (88.53, 315.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['CH4'])] = (36.72, 108.4)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['C2H6'])] = (31.23, 84.76)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['CHaro'])] = (29.78, 58.17)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['Caro'])] = (3.775, 144.8)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['Cfused_aromatic'])] = (3.775, 144.8)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['CH2cyclic'])] = (12.78, 28.37)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['CHcyclic'])] = (-54.90, -319.5)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['CO2'])] = (136.9, 254.6)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['N2'])] = (82.28, 202.8)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['H2S'])] = (134.6, 138.3)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2'], PPR78_GROUPS['SH'])] = (459.5, 627.3)

PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['CH'])] = (0.0, 0.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['C'])] = (-305.7, -250.8)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['CH4'])] = (145.2, 301.6)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['C2H6'])] = (174.3, 352.1)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['CHaro'])] = (103.3, 191.8)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['Caro'])] = (6.177, -33.97)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['Cfused_aromatic'])] = (6.177, -33.97)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['CH2cyclic'])] = (101.9, -90.93)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['CHcyclic'])] = (-226.5, -51.47)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['CO2'])] = (184.3, 762.1)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['N2'])] = (365.4, 521.9)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['H2S'])] = (193.9, 307.8)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH'], PPR78_GROUPS['SH'])] = (425.5, 514.7)

PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['C'])] = (0.0, 0.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['CH4'])] = (263.9, 531.5)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['C2H6'])] = (333.2, 203.8)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['CHaro'])] = (158.9, 613.2)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['Caro'])] = (79.61, -326.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['Cfused_aromatic'])] = (79.61, -326.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['CH2cyclic'])] = (177.1, 601.9)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['CHcyclic'])] = (17.84, -109.5)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['CO2'])] = (287.9, 346.2)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['N2'])] = (263.9, 772.6)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['H2S'])] = (305.1, -143.1)
PPR78_INTERACTIONS[(PPR78_GROUPS['C'], PPR78_GROUPS['SH'])] = (682.9, 1544.0)

PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['CH4'])] = (0.0, 0.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['C2H6'])] = (13.04, 6.863)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['CHaro'])] = (67.26, 167.5)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['Caro'])] = (139.3, 464.3)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['Cfused_aromatic'])] = (139.3, 464.3)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['CH2cyclic'])] = (36.37, 26.42)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['CHcyclic'])] = (40.15, 255.3)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['CO2'])] = (137.3, 194.2)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['N2'])] = (37.90, 37.2)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['H2S'])] = (181.2, 288.9)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH4'], PPR78_GROUPS['SH'])] = (706.0, 1483.0)

PPR78_INTERACTIONS[(PPR78_GROUPS['C2H6'], PPR78_GROUPS['C2H6'])] = (0.0, 0.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['C2H6'], PPR78_GROUPS['CHaro'])] = (41.18, 50.79)
PPR78_INTERACTIONS[(PPR78_GROUPS['C2H6'], PPR78_GROUPS['Caro'])] = (-3.088, 13.04)
PPR78_INTERACTIONS[(PPR78_GROUPS['C2H6'], PPR78_GROUPS['Cfused_aromatic'])] = (-3.088, 13.04)
PPR78_INTERACTIONS[(PPR78_GROUPS['C2H6'], PPR78_GROUPS['CH2cyclic'])] = (8.579, 76.86)
PPR78_INTERACTIONS[(PPR78_GROUPS['C2H6'], PPR78_GROUPS['CHcyclic'])] = (10.29, -52.84)
PPR78_INTERACTIONS[(PPR78_GROUPS['C2H6'], PPR78_GROUPS['CO2'])] = (135.5, 239.5)
PPR78_INTERACTIONS[(PPR78_GROUPS['C2H6'], PPR78_GROUPS['N2'])] = (61.59, 84.92)
PPR78_INTERACTIONS[(PPR78_GROUPS['C2H6'], PPR78_GROUPS['H2S'])] = (157.2, 217.1)

PPR78_INTERACTIONS[(PPR78_GROUPS['CHaro'], PPR78_GROUPS['CHaro'])] = (0.0, 0.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHaro'], PPR78_GROUPS['Caro'])] = (-13.38, 20.25)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHaro'], PPR78_GROUPS['Cfused_aromatic'])] = (-13.38, 20.25)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHaro'], PPR78_GROUPS['CH2cyclic'])] = (29.17, 69.32)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHaro'], PPR78_GROUPS['CHcyclic'])] = (-26.42, -789.2)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHaro'], PPR78_GROUPS['CO2'])] = (102.6, 161.3)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHaro'], PPR78_GROUPS['N2'])] = (185.2, 490.6)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHaro'], PPR78_GROUPS['H2S'])] = (21.96, 13.04)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHaro'], PPR78_GROUPS['SH'])] = (285.5, 392.0)

PPR78_INTERACTIONS[(PPR78_GROUPS['Caro'], PPR78_GROUPS['Caro'])] = (0.0, 0.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['Caro'], PPR78_GROUPS['Cfused_aromatic'])] = (0.0, 0.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['Caro'], PPR78_GROUPS['CH2cyclic'])] = (34.31, 95.39)
PPR78_INTERACTIONS[(PPR78_GROUPS['Caro'], PPR78_GROUPS['CHcyclic'])] = (-105.7, -286.5)
PPR78_INTERACTIONS[(PPR78_GROUPS['Caro'], PPR78_GROUPS['CO2'])] = (110.1, 637.6)
PPR78_INTERACTIONS[(PPR78_GROUPS['Caro'], PPR78_GROUPS['N2'])] = (284.0, 1892.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['Caro'], PPR78_GROUPS['H2S'])] = (1.029, -8.579)
PPR78_INTERACTIONS[(PPR78_GROUPS['Caro'], PPR78_GROUPS['SH'])] = (1072.0, 1094.0)

PPR78_INTERACTIONS[(PPR78_GROUPS['Cfused_aromatic'], PPR78_GROUPS['CH2cyclic'])] = (34.31, 95.39)
PPR78_INTERACTIONS[(PPR78_GROUPS['Cfused_aromatic'], PPR78_GROUPS['CHcyclic'])] = (-105.7, -286.5)
PPR78_INTERACTIONS[(PPR78_GROUPS['Cfused_aromatic'], PPR78_GROUPS['CO2'])] = (267.3, 444.4)
PPR78_INTERACTIONS[(PPR78_GROUPS['Cfused_aromatic'], PPR78_GROUPS['N2'])] = (718.1, 1892.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['Cfused_aromatic'], PPR78_GROUPS['H2S'])] = (1.029, -8.579)
PPR78_INTERACTIONS[(PPR78_GROUPS['Cfused_aromatic'], PPR78_GROUPS['SH'])] = (1072.0, 1094.0)

PPR78_INTERACTIONS[(PPR78_GROUPS['CH2cyclic'], PPR78_GROUPS['CHcyclic'])] = (-50.10, -891.1)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2cyclic'], PPR78_GROUPS['CO2'])] = (130.1, 225.8)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2cyclic'], PPR78_GROUPS['N2'])] = (179.5, 546.6)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2cyclic'], PPR78_GROUPS['H2S'])] = (120.8, 163.0)
PPR78_INTERACTIONS[(PPR78_GROUPS['CH2cyclic'], PPR78_GROUPS['SH'])] = (446.1, 549.0)

PPR78_INTERACTIONS[(PPR78_GROUPS['CHcyclic'], PPR78_GROUPS['CO2'])] = (91.28, 82.01)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHcyclic'], PPR78_GROUPS['N2'])] = (100.9, 249.8)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHcyclic'], PPR78_GROUPS['H2S'])] = (-16.13, -147.6)
PPR78_INTERACTIONS[(PPR78_GROUPS['CHcyclic'], PPR78_GROUPS['SH'])] = (411.8, -308.8)

PPR78_INTERACTIONS[(PPR78_GROUPS['CO2'], PPR78_GROUPS['N2'])] = (98.42, 221.4)
PPR78_INTERACTIONS[(PPR78_GROUPS['CO2'], PPR78_GROUPS['H2S'])] = (134.9, 201.4)
# PPR78_INTERACTIONS[(PPR78_GROUPS['CO2'], PPR78_GROUPS['SH'])] = (None, None)  # N.A.

PPR78_INTERACTIONS[(PPR78_GROUPS['N2'], PPR78_GROUPS['H2S'])] = (319.5, 550.1)
# PPR78_INTERACTIONS[(PPR78_GROUPS['N2'], PPR78_GROUPS['SH'])] = (None, None)  # N.A.

PPR78_INTERACTIONS[(PPR78_GROUPS['H2S'], PPR78_GROUPS['SH'])] = (-77.21, 156.1)


# Symmetrize and fill missing interactions
PPR78_ZERO_PARAMETER = (0.0, 0.0)
# Symmetrize the PPR78_INTERACTIONS
groups = list(PPR78_GROUPS.values())
# Loop through all pairs of groups to ensure symmetry and fill defaults
for group1 in groups:
    for group2 in groups:
        key = (group1, group2)
        reverse_key = (group2, group1)
        # Symmetrize: if key exists but reverse_key does not, copy the value
        if key in PPR78_INTERACTIONS and reverse_key not in PPR78_INTERACTIONS:
            PPR78_INTERACTIONS[reverse_key] = PPR78_INTERACTIONS[key]
        # Missing parameters are zero
        if key not in PPR78_INTERACTIONS and reverse_key not in PPR78_INTERACTIONS:
            PPR78_INTERACTIONS[key] = PPR78_ZERO_PARAMETER
            PPR78_INTERACTIONS[reverse_key] = PPR78_ZERO_PARAMETER

PPR78_INTERACTIONS_BY_STR = {}
for (group1, group2), value in PPR78_INTERACTIONS.items():
    PPR78_INTERACTIONS_BY_STR[(group1.group, group2.group)] = value

PPR78_GROUP_IDS = [k.group_id for k in PPR78_GROUPS.values()]
PPR78_GROUPS = list(PPR78_GROUPS.values())

def PPR78_kij(T, molecule1_groups, molecule2_groups, Tc1, Pc1, omega1, Tc2, Pc2, omega2):
    r'''Calculate binary interaction parameter kij(T) between two molecules using the PPR78 method.

    This function implements the PPR78 (Predictive Peng-Robinson 1978) method to calculate
    the binary interaction parameter kij between two molecules, including the calculation
    of their equation of state parameters.

    .. math::
        k_{ij}(T) = \frac{
        -\frac{1}{2} \left[ \sum_{k=1}^{N_g} \sum_{l=1}^{N_g} \left( \alpha_{ik} - \alpha_{jk} \right) \left( \alpha_{il} - \alpha_{jl} \right) A_{kl} \times \left( \frac{298.15}{T/K} \right) \left( \frac{B_{kl}}{A_{kl}} - 1 \right) \right] - \left( \frac{\sqrt{a_i(T)}}{b_i} - \frac{\sqrt{a_j(T)}}{b_j} \right)^2
        }{
        2 \sqrt{\frac{a_i(T) \cdot a_j(T)}{b_i \cdot b_j}}
        }

    .. math::
        b_i = 0.0777960739 \frac{RT_{c,i}}{P_{c,i}}

    .. math::
        a_i = 0.457235529 \frac{R^2T_{c,i}^2}{P_{c,i}} \left[1 + m_i\left(1 - \sqrt{\frac{T}{T_{c,i}}}\right)\right]^2

    .. math::
        m_i = \begin{cases}
        0.37464 + 1.54226\omega_i - 0.26992\omega_i^2 & \text{if } \omega_i \leq 0.491 \\
        0.379642 + 1.48503\omega_i - 0.164423\omega_i^2 + 0.016666\omega_i^3 & \text{if } \omega_i > 0.491
        \end{cases}

    Parameters
    ----------
    T : float
        Temperature [K]
    molecule1_groups : dict
        Dictionary of group counts for molecule 1 {group: count}
    molecule2_groups : dict
        Dictionary of group counts for molecule 2 {group: count}
    Tc1 : float
        Critical temperature of molecule 1 [K]
    Pc1 : float
        Critical pressure of molecule 1 [Pa]
    omega1 : float
        Acentric factor of molecule 1 [-]
    Tc2 : float
        Critical temperature of molecule 2 [K]
    Pc2 : float
        Critical pressure of molecule 2 [Pa]
    omega2 : float
        Acentric factor of molecule 2 [-]

    Returns
    -------
    float
        Binary interaction parameter kij [-]

    Examples
    --------
    >>> # Example for methane-ethane system
    >>> PPR78_kij(298.15, {"CH3": 1}, {"CH3": 2}, Tc1=190.564, Pc1=4599200, omega1=0.01142, Tc2=305.322, Pc2=4872200, omega2=0.0995)
    -0.0096176

    Notes
    -----
    Confirmed with an example in [2]_.
    
    References
    ----------
    .. [1] Jaubert, Jean-Noël, Romain Privat, and Fabrice Mutelet. "Predicting 
       the Phase Equilibria of Synthetic Petroleum Fluids with the PPR78 
       Approach." AIChE Journal 56, no. 12 (2010): 3225-35. 
       https://doi.org/10.1002/aic.12232.
    .. [2] Jaubert, Jean-Noël, and Fabrice Mutelet. "VLE Predictions with the
       Peng-Robinson Equation of State and Temperature Dependent Kij Calculated
       through a Group Contribution Method." Fluid Phase Equilibria 224, 
       no. 2 (October 1, 2004): 285-304. doi:10.1016/j.fluid.2004.06.059. 
    '''
    OMEGA_A = 0.4572355289213821893834601962251837888504
    OMEGA_B = 0.0777960739038884559718447100373331839711

    # Calculate m parameters based on acentric factors
    def calculate_m(omega):
        if omega <= 0.491:
            return 0.37464 + 1.54226*omega - 0.26992*omega**2
        else:
            return 0.379642 + 1.48503*omega - 0.164423*omega**2 + 0.016666*omega**3

    # Calculate b parameters
    bi_1 = OMEGA_B * (R * Tc1 / Pc1)
    bi_2 = OMEGA_B * (R * Tc2 / Pc2)

    # Calculate a parameters
    m1 = calculate_m(omega1)
    m2 = calculate_m(omega2)
    
    # a_alpha in thermo, matches PR78
    ai_T1 = (OMEGA_A * R**2 * Tc1**2 / Pc1) * (1 + m1 * (1 - sqrt(T/Tc1)))**2
    ai_T2 = (OMEGA_A * R**2 * Tc2**2 / Pc2) * (1 + m2 * (1 - sqrt(T/Tc2)))**2

    # Calculate group fractions
    total_groups1 = sum(molecule1_groups.values())
    total_groups2 = sum(molecule2_groups.values())
    
    alpha_i = {group: count/total_groups1 for group, count in molecule1_groups.items()}
    alpha_j = {group: count/total_groups2 for group, count in molecule2_groups.items()}
    
    # Calculate first term (group contribution)
    term1 = 0.0
    for group_k in set(alpha_i.keys()) | set(alpha_j.keys()):
        for group_l in set(alpha_i.keys()) | set(alpha_j.keys()):
            alpha_ik = alpha_i.get(group_k, 0.0)
            alpha_jk = alpha_j.get(group_k, 0.0)
            alpha_il = alpha_i.get(group_l, 0.0)
            alpha_jl = alpha_j.get(group_l, 0.0)
            
            A_kl, B_kl = PPR78_INTERACTIONS_BY_STR[(group_k, group_l)]
            
            if A_kl != 0:
                delta_k = alpha_ik - alpha_jk
                delta_l = alpha_il - alpha_jl
                term1 += delta_k * delta_l * A_kl * (298.15/T)**(B_kl/A_kl - 1)
    term1 *= -0.5
    term1 *= 1e6 # Didn't catch that the first time around
    
    # Calculate second term (EOS parameters)
    sqrt_ai_T1 = sqrt(ai_T1)
    sqrt_ai_T2 = sqrt(ai_T2)
    term2 = ((sqrt_ai_T1/bi_1) - (sqrt_ai_T2/bi_2))**2
    
    # Calculate denominator
    denominator = 2 * sqrt(ai_T1 * ai_T2)/(bi_1 * bi_2)
    
    # Final calculation
    kij_value = (term1 - term2)/denominator
    return kij_value

def PPR78_kijs(T, groups, Tcs, Pcs, omegas):
    r"""Calculate the binary interaction parameter (kij) matrix for a mixture of components 
    at a specified temperature using the PPR78 method at a specified temperature.
    
    Parameters
    ----------
    T : float
        System temperature [K]
    groups : list[dict]
        List of dictionaries containing group counts for each component
        Each dict has format {group_name: count}
    Tcs : list[float]
        Critical temperatures for each component [K]
    Pcs : list[float]
        Critical pressures for each component [Pa]
    omegas : list[float]
        Acentric factors for each component [-]
        
    Returns
    -------
    list[list[float]]
        Square matrix of kij values where matrix[i][j] gives the interaction
        parameter between components i and j
        
    Examples
    --------
    >>> # Calculate kij matrix for methane-ethane-propane mixture
    >>> groups = [
    ...     {"CH3": 1},              # methane
    ...     {"CH3": 2},              # ethane  
    ...     {"CH3": 2, "CH2": 1}     # propane
    ... ]
    >>> Tc = [190.564, 305.322, 369.83]  
    >>> Pc = [4599200, 4872200, 4248000]
    >>> omega = [0.01142, 0.0995, 0.1523]
    >>> matrix = PPR78_kijs(298.15, groups, Tc, Pc, omega)
    """
    n_components = len(groups)
    kij_matrix = [[0.0 for _ in range(n_components)] for _ in range(n_components)]
    
    # Calculate upper triangle
    for i in range(n_components):
        for j in range(i+1, n_components):
            kij = PPR78_kij(
                T,
                groups[i],
                groups[j],
                Tcs[i],
                Pcs[i],
                omegas[i],
                Tcs[j],
                Pcs[j],
                omegas[j]
            )
            # Set both (i,j) and (j,i) due to symmetry
            kij_matrix[i][j] = kij
            kij_matrix[j][i] = kij
            
    return kij_matrix

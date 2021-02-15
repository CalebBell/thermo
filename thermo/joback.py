# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains an implementation of the Joback group-contribution method.
This functionality requires the RDKit library to work.

For submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. warning::
    The Joback class method does not contain all the groups for every chemical.
    There are often multiple ways of fragmenting a chemical. Other times, the
    fragmentation algorithm will fail. These limitations are present in both
    the implementation and the method itself. You are welcome to seek to
    improve this code but no to little help can be offered.

.. contents:: :local:


.. autoclass:: thermo.joback.Joback
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: thermo.joback.smarts_fragment
.. autodata:: J_BIGGS_JOBACK_SMARTS
.. autodata:: J_BIGGS_JOBACK_SMARTS_id_dict

'''

from __future__ import division

__all__ = ['smarts_fragment', 'Joback', 'J_BIGGS_JOBACK_SMARTS',
           'J_BIGGS_JOBACK_SMARTS_id_dict']

from chemicals.utils import to_num, horner, exp

rdkit_missing = 'RDKit is not installed; it is required to use this functionality'

loaded_rdkit = False
Chem, Descriptors, AllChem, rdMolDescriptors = None, None, None, None
def load_rdkit_modules():
    global loaded_rdkit, Chem, Descriptors, AllChem, rdMolDescriptors
    if loaded_rdkit:
        return
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdMolDescriptors
        loaded_rdkit = True
    except:
        if not hasRDKit: # pragma: no cover
            raise Exception(rdkit_missing)

# See https://www.atmos-chem-phys.net/16/4401/2016/acp-16-4401-2016.pdf for more
# smarts patterns


J_BIGGS_JOBACK_SMARTS = [["Methyl","-CH3", "[CX4H3]"],
["Secondary acyclic", "-CH2-", "[!R;CX4H2]"],
["Tertiary acyclic",">CH-", "[!R;CX4H]"],
["Quaternary acyclic", ">C<", "[!R;CX4H0]"],

["Primary alkene", "=CH2", "[CX3H2]"],
["Secondary alkene acyclic", "=CH-", "[!R;CX3H1;!$([CX3H1](=O))]"],
["Tertiary alkene acyclic", "=C<", "[$([!R;CX3H0]);!$([!R;CX3H0]=[#8])]"],
["Cumulative alkene", "=C=", "[$([CX2H0](=*)=*)]"],
["Terminal alkyne", u"≡CH","[$([CX2H1]#[!#7])]"],
["Internal alkyne",u"≡C-","[$([CX2H0]#[!#7])]"],

["Secondary cyclic", "-CH2- (ring)", "[R;CX4H2]"],
["Tertiary cyclic", ">CH- (ring)", "[R;CX4H]"],
["Quaternary cyclic", ">C< (ring)", "[R;CX4H0]"],

["Secondary alkene cyclic", "=CH- (ring)", "[R;CX3H1,cX3H1]"],
["Tertiary alkene cyclic", "=C< (ring)","[$([R;CX3H0]);!$([R;CX3H0]=[#8])]"],

["Fluoro", "-F", "[F]"],
["Chloro", "-Cl", "[Cl]"],
["Bromo", "-Br", "[Br]"],
["Iodo", "-I", "[I]"],

["Alcohol","-OH (alcohol)", "[OX2H;!$([OX2H]-[#6]=[O]);!$([OX2H]-a)]"],
["Phenol","-OH (phenol)", "[$([OX2H]-a)]"],
["Ether acyclic", "-O- (nonring)", "[OX2H0;!R;!$([OX2H0]-[#6]=[#8])]"],
["Ether cyclic", "-O- (ring)", "[#8X2H0;R;!$([#8X2H0]~[#6]=[#8])]"],
["Carbonyl acyclic", ">C=O (nonring)","[$([CX3H0](=[OX1]));!$([CX3](=[OX1])-[OX2]);!R]=O"],
["Carbonyl cyclic", ">C=O (ring)","[$([#6X3H0](=[OX1]));!$([#6X3](=[#8X1])~[#8X2]);R]=O"],
["Aldehyde","O=CH- (aldehyde)","[CX3H1](=O)"],
["Carboxylic acid", "-COOH (acid)", "[OX2H]-[C]=O"],
["Ester", "-COO- (ester)", "[#6X3H0;!$([#6X3H0](~O)(~O)(~O))](=[#8X1])[#8X2H0]"],
["Oxygen double bond other", "=O (other than above)","[OX1H0;!$([OX1H0]~[#6X3]);!$([OX1H0]~[#7X3]~[#8])]"],

["Primary amino","-NH2", "[NX3H2]"],
["Secondary amino acyclic",">NH (nonring)", "[NX3H1;!R]"],
["Secondary amino cyclic",">NH (ring)", "[#7X3H1;R]"],
["Tertiary amino", ">N- (nonring)","[#7X3H0;!$([#7](~O)~O)]"],
["Imine acyclic","-N= (nonring)","[#7X2H0;!R]"],
["Imine cyclic","-N= (ring)","[#7X2H0;R]"],
["Aldimine", "=NH", "[#7X2H1]"],
["Cyano", "-CN","[#6X2]#[#7X1H0]"],
["Nitro", "-NO2", "[$([#7X3,#7X3+][!#8])](=[O])~[O-]"],

["Thiol", "-SH", "[SX2H]"],
["Thioether acyclic", "-S- (nonring)", "[#16X2H0;!R]"],
["Thioether cyclic", "-S- (ring)", "[#16X2H0;R]"]]
'''Metadata for the Joback groups. The first element is the group name; the
second is the group symbol; and the third is the SMARTS matching string.
'''

J_BIGGS_JOBACK_SMARTS_id_dict = {i+1: j[2] for i, j in enumerate(J_BIGGS_JOBACK_SMARTS)}
J_BIGGS_JOBACK_SMARTS_str_dict = {i[1]: i[2] for i in J_BIGGS_JOBACK_SMARTS}

# Shi Chenyang's JRGUI code indicates he left the following list of smarts in
# favor of those above by J Biggs
SHI_CHENYANG_JOBACK_SMARTS =  [
("-CH3", "[CH3;A;X4;!R]"),
("-CH2-", "[CH2;A;X4;!R]"),
(">CH-", "[CH1;A;X4;!R]"),
(">C<", "[CH0;A;X4;!R]"),
("=CH2", "[CH2;A;X3;!R]"),
("=CH-", "[CH1;A;X3;!R]"),
("=C<", "[CH0;A;X3;!R]"),
("=C=", "[$([CH0;A;X2;!R](=*)=*)]"),
("≡CH", "[$([CH1;A;X2;!R]#*)]"),
("≡C-", "[$([CH0;A;X2;!R]#*)]"),
("-CH2- (ring)", "[CH2;A;X4;R]"),
(">CH- (ring)", "[CH1;A;X4;R]"),
(">C< (ring)", "[CH0;A;X4;R]"),
("=CH- (ring)", "[CH1;X3;R]"),
("=C< (ring)", "[CH0;X3;R]"),
("-F", "[F]"),
("-Cl", "[Cl]"),
("-Br", "[Br]"),
("-I", "[I]"),
("-OH (alcohol)", "[O;H1;$(O-!@[C;!$(C=!@[O,N,S])])]"),
("-OH (phenol)", "[O;H1;$(O-!@c)]"),
("-O- (nonring)", "[OH0;X2;!R]"),
("-O- (ring)", "[OH0;X2;R]"),
(">C=O (nonring)", "[CH0;A;X3;!R]=O"),
(">C=O (ring)", "[CH0;A;X3;R]=O"),
("O=CH- (aldehyde)", "[CH;D2;$(C-!@C)](=O)"),
("-COOH (acid)", "[$(C-!@[A;!O])](=O)([O;H,-])"),
("-COO- (ester)", "C(=O)[OH0]"),
("=O (other than above)", "[OX1]"),
("-NH2", "[NH2;X3]"),
(">NH (nonring)", "[NH1;X3;!R]"),
(">NH (ring)", "[NH1;X3;R]"),
(">N- (nonring)", "[NH0;X3;!R]"),
("-N= (nonring)", "[NH0;X2;!R]"),
("-N= (ring)", "[NH0;X2;R]"),
("=NH", "[NH1;X2]"),
("-CN", "C#N"),
("-NO2", "N(=O)=O"),
("-SH", "[SH1]"),
("-S- (nonring)", "[SH0;!R]"),
("-S- (ring)", "[SH0;R]")]
SHI_CHENYANG_JOBACK_SMARTS_id_dict = {i+1: j[1] for i, j in enumerate(SHI_CHENYANG_JOBACK_SMARTS)}
SHI_CHENYANG_JOBACK_SMARTS_str_dict = {i[0]: i[1] for i in SHI_CHENYANG_JOBACK_SMARTS}

joback_data_txt = u'''-CH3 	0.0141 	-0.0012 	65 	23.58 	-5.10 	-76.45 	-43.96 	1.95E+1 	-8.08E-3 	1.53E-4 	-9.67E-8 	0.908 	2.373 	548.29 	-1.719
-CH2- 	0.0189 	0.0000 	56 	22.88 	11.27 	-20.64 	8.42 	-9.09E-1 	9.50E-2 	-5.44E-5 	1.19E-8 	2.590 	2.226 	94.16 	-0.199
>CH- 	0.0164 	0.0020 	41 	21.74 	12.64 	29.89 	58.36 	-2.30E+1 	2.04E-1 	-2.65E-4 	1.20E-7 	0.749 	1.691 	-322.15 	1.187
>C< 	0.0067 	0.0043 	27 	18.25 	46.43 	82.23 	116.02 	-6.62E+1 	4.27E-1 	-6.41E-4 	3.01E-7 	-1.460 	0.636 	-573.56 	2.307
=CH2 	0.0113 	-0.0028 	56 	18.18 	-4.32 	-9.630 	3.77 	2.36E+1 	-3.81E-2 	1.72E-4 	-1.03E-7 	-0.473 	1.724 	495.01 	-1.539
=CH- 	0.0129 	-0.0006 	46 	24.96 	8.73 	37.97 	48.53 	-8.00 	1.05E-1 	-9.63E-5 	3.56E-8 	2.691 	2.205 	82.28 	-0.242
=C< 	0.0117 	0.0011 	38 	24.14 	11.14 	83.99 	92.36 	-2.81E+1 	2.08E-1 	-3.06E-4 	1.46E-7 	3.063 	2.138 	n. a. 	n. a.
=C= 	0.0026 	0.0028 	36 	26.15 	17.78 	142.14 	136.70 	2.74E+1 	-5.57E-2 	1.01E-4 	-5.02E-8 	4.720 	2.661 	n. a. 	n. a.
≡CH 	0.0027 	-0.0008 	46 	9.20 	-11.18 	79.30 	77.71 	2.45E+1 	-2.71E-2 	1.11E-4 	-6.78E-8 	2.322 	1.155 	n. a. 	n. a.
≡C- 	0.0020 	0.0016 	37 	27.38 	64.32 	115.51 	109.82 	7.87 	2.01E-2 	-8.33E-6 	1.39E-9 	4.151 	3.302 	n. a. 	n. a.
-CH2- (ring)	0.0100 	0.0025 	48 	27.15 	7.75 	-26.80 	-3.68 	-6.03 	8.54E-2 	-8.00E-6 	-1.80E-8 	0.490 	2.398 	307.53 	-0.798
>CH- (ring)	0.0122 	0.0004 	38 	21.78 	19.88 	8.67 	40.99 	-2.05E+1 	1.62E-1 	-1.60E-4 	6.24E-8 	3.243 	1.942 	-394.29 	1.251
>C< (ring)	0.0042 	0.0061 	27 	21.32 	60.15 	79.72 	87.88 	-9.09E+1 	5.57E-1 	-9.00E-4 	4.69E-7 	-1.373 	0.644 	n. a. 	n. a.
=CH- (ring)	0.0082 	0.0011 	41 	26.73 	8.13 	2.09 	11.30 	-2.14 	5.74E-2 	-1.64E-6 	-1.59E-8 	1.101 	2.544 	259.65 	-0.702
=C< (ring)	0.0143 	0.0008 	32 	31.01 	37.02 	46.43 	54.05 	-8.25 	1.01E-1 	-1.42E-4 	6.78E-8 	2.394 	3.059 	-245.74 	0.912
-F 	0.0111 	-0.0057 	27 	-0.03 	-15.78 	-251.92 	-247.19 	2.65E+1 	-9.13E-2 	1.91E-4 	-1.03E-7 	1.398 	-0.670 	n. a. 	n. a.
-Cl 	0.0105 	-0.0049 	58 	38.13 	13.55 	-71.55 	-64.31 	3.33E+1 	-9.63E-2 	1.87E-4 	-9.96E-8 	2.515 	4.532 	625.45 	-1.814
-Br 	0.0133 	0.0057 	71 	66.86 	43.43 	-29.48 	-38.06 	2.86E+1 	-6.49E-2 	1.36E-4 	-7.45E-8 	3.603 	6.582 	738.91 	-2.038
-I 	0.0068 	-0.0034 	97 	93.84 	41.69 	21.06 	5.74 	3.21E+1 	-6.41E-2 	1.26E-4 	-6.87E-8 	2.724 	9.520 	809.55 	-2.224
-OH (alcohol) 	0.0741 	0.0112 	28 	92.88 	44.45 	-208.04 	-189.20 	2.57E+1 	-6.91E-2 	1.77E-4 	-9.88E-8 	2.406 	16.826 	2173.72 	-5.057
-OH (phenol) 	0.0240 	0.0184 	-25 	76.34 	82.83 	-221.65 	-197.37 	-2.81 	1.11E-1 	-1.16E-4 	4.94E-8 	4.490 	12.499 	3018.17 	-7.314
-O- (nonring) 	0.0168 	0.0015 	18 	22.42 	22.23 	-132.22 	-105.00 	2.55E+1 	-6.32E-2 	1.11E-4 	-5.48E-8 	1.188 	2.410 	122.09 	-0.386
-O- (ring) 	0.0098 	0.0048 	13 	31.22 	23.05 	-138.16 	-98.22 	1.22E+1 	-1.26E-2 	6.03E-5 	-3.86E-8 	5.879 	4.682 	440.24 	-0.953
>C=O (nonring) 	0.0380 	0.0031 	62 	76.75 	61.20 	-133.22 	-120.50 	6.45 	6.70E-2 	-3.57E-5 	2.86E-9 	4.189 	8.972 	340.35 	-0.350
>C=O (ring) 	0.0284 	0.0028 	55 	94.97 	75.97 	-164.50 	-126.27 	3.04E+1 	-8.29E-2 	2.36E-4 	-1.31E-7 	0. 	6.645 	n. a. 	n. a.
O=CH- (aldehyde) 	0.0379 	0.0030 	82 	72.24 	36.90 	-162.03 	-143.48 	3.09E+1 	-3.36E-2 	1.60E-4 	-9.88E-8 	3.197 	9.093 	740.92 	-1.713
-COOH (acid) 	0.0791 	0.0077 	89 	169.09 	155.50 	-426.72 	-387.87 	2.41E+1 	4.27E-2 	8.04E-5 	-6.87E-8 	11.051 	19.537 	1317.23 	-2.578
-COO- (ester) 	0.0481 	0.0005 	82 	81.10 	53.60 	-337.92 	-301.95 	2.45E+1 	4.02E-2 	4.02E-5 	-4.52E-8 	6.959 	9.633 	483.88 	-0.966
=O (other than above) 	0.0143 	0.0101 	36 	-10.50 	2.08 	-247.61 	-250.83 	6.82 	1.96E-2 	1.27E-5 	-1.78E-8 	3.624 	5.909 	675.24 	-1.340
-NH2 	0.0243 	0.0109 	38 	73.23 	66.89 	-22.02 	14.07 	2.69E+1 	-4.12E-2 	1.64E-4 	-9.76E-8 	3.515 	10.788 	n. a. 	n. a.
>NH (nonring) 	0.0295 	0.0077 	35 	50.17 	52.66 	53.47 	89.39 	-1.21 	7.62E-2 	-4.86E-5 	1.05E-8 	5.099 	6.436 	n. a. 	n. a.
>NH (ring) 	0.0130 	0.0114 	29 	52.82 	101.51 	31.65 	75.61 	1.18E+1 	-2.30E-2 	1.07E-4 	-6.28E-8 	7.490 	6.930 	n. a. 	n. a.
>N- (nonring) 	0.0169 	0.0074 	9 	11.74 	48.84 	123.34 	163.16 	-3.11E+1 	2.27E-1 	-3.20E-4 	1.46E-7 	4.703 	1.896 	n. a. 	n. a.
-N= (nonring) 	0.0255 	-0.0099 	n. a. 	74.60 	n. a. 	23.61 	n. a. 	n. a. 	n. a. 	n. a. 	n. a. 	n. a. 	3.335 	n. a. 	n. a.
-N= (ring) 	0.0085 	0.0076 	34 	57.55 	68.40 	55.52 	79.93 	8.83 	-3.84E-3 	4.35E-5 	-2.60E-8 	3.649 	6.528 	n. a. 	n. a.
=NH 	n. a. 	n. a. 	n. a. 	83.08 	68.91 	93.70 	119.66 	5.69 	-4.12E-3 	1.28E-4 	-8.88E-8 	n. a. 	12.169 	n. a. 	n. a.
-CN 	0.0496 	-0.0101 	91 	125.66 	59.89 	88.43 	89.22 	3.65E+1 	-7.33E-2 	1.84E-4 	-1.03E-7 	2.414 	12.851 	n. a. 	n. a.
-NO2 	0.0437 	0.0064 	91 	152.54 	127.24 	-66.57 	-16.83 	2.59E+1 	-3.74E-3 	1.29E-4 	-8.88E-8 	9.679 	16.738 	n. a. 	n. a.
-SH 	0.0031 	0.0084 	63 	63.56 	20.09 	-17.33 	-22.99 	3.53E+1 	-7.58E-2 	1.85E-4 	-1.03E-7 	2.360 	6.884 	n. a. 	n. a.
-S- (nonring) 	0.0119 	0.0049 	54 	68.78 	34.40 	41.87 	33.12 	1.96E+1 	-5.61E-3 	4.02E-5 	-2.76E-8 	4.130 	6.817 	n. a. 	n. a.
-S- (ring) 	0.0019 	0.0051 	38 	52.10 	79.93 	39.10 	27.76 	1.67E+1 	4.81E-3 	2.77E-5 	-2.11E-8 	1.557 	5.984 	n. a. 	n. a.'''

joback_groups_str_dict = {}
joback_groups_id_dict = {}
#JOBACK = namedtuple('JOBACK', 'i, name, Tc, Pc, Vc, Tb, Tm, Hform, Gform, Cpa, Cpb, Cpc, Cpd, Hfus, Hvap, mua, mub')

class JOBACK(object):
    __slots__ = ('i', 'name', 'Tc', 'Pc', 'Vc', 'Tb', 'Tm', 'Hform',
                 'Gform', 'Cpa', 'Cpb', 'Cpc', 'Cpd', 'Hfus', 'Hvap',
                 'mua', 'mub')
    def __init__(self, i, name, Tc, Pc, Vc, Tb, Tm, Hform, Gform, Cpa, Cpb,
                 Cpc, Cpd, Hfus, Hvap, mua, mub):
        self.i = i
        self.name = name
        self.Tc = Tc
        self.Pc = Pc
        self.Vc = Vc
        self.Tb = Tb
        self.Tm = Tm
        self.Hform = Hform
        self.Gform = Gform
        self.Cpa = Cpa
        self.Cpb = Cpb
        self.Cpc = Cpc
        self.Cpd = Cpd
        self.Hfus = Hfus
        self.Hvap = Hvap
        self.mua = mua
        self.mub = mub

    def __repr__(self):
        return '''JOBACK(i=%r, name=%r, Tc=%r, Pc=%r, Vc=%r, Tb=%r, Tm=%r, Hform=%r, Gform=%r,
Cpa=%r, Cpb=%r, Cpc=%r, Cpd=%r, Hfus=%r, Hvap=%r, mua=%r, mub=%r)''' % (
        self.i, self.name, self.Tc, self.Pc, self.Vc, self.Tb, self.Tm,
        self.Hform, self.Gform, self.Cpa, self.Cpb, self.Cpc, self.Cpd,
        self.Hfus, self.Hvap, self.mua, self.mub)

for i, line in enumerate(joback_data_txt.split('\n')):
    parsed = to_num(line.split('\t'))
    j = JOBACK(i+1, *parsed)
    joback_groups_str_dict[parsed[0]] = j
    joback_groups_id_dict[i+1] = j


def smarts_fragment(catalog, rdkitmol=None, smi=None, deduplicate=True):
    r'''Fragments a molecule into a set of unique groups and counts as
    specified by the `catalog`. The molecule can either be an rdkit
    molecule object, or a smiles string which will be parsed by rdkit.
    Returns a dictionary of groups and their counts according to the
    indexes of the catalog provided.

    Parameters
    ----------
    catalog : dict
        Dictionary indexed by keys pointing to smarts strings, [-]
    rdkitmol : mol, optional
        Molecule as rdkit object, [-]
    smi : str, optional
        Smiles string representing a chemical, [-]

    Returns
    -------
    counts : dict
        Dictionaty of integer counts of the found groups only, indexed by
        the same keys used by the catalog [-]
    success : bool
        Whether or not molecule was fully and uniquely fragmented, [-]
    status : str
        A string holding an explanation of why the molecule failed to be
        fragmented, if it fails; 'OK' if it suceeds.

    Notes
    -----
    Raises an exception if rdkit is not installed, or `smi` or `rdkitmol` is
    not defined.

    Examples
    --------
    Acetone:

    >>> smarts_fragment(catalog=J_BIGGS_JOBACK_SMARTS_id_dict, smi='CC(=O)C')
    ({1: 2, 24: 1}, True, 'OK')

    Sodium sulfate, (Na2O4S):

    >>> smarts_fragment(catalog=J_BIGGS_JOBACK_SMARTS_id_dict, smi='[O-]S(=O)(=O)[O-].[Na+].[Na+]')
    ({29: 4}, False, 'Did not match all atoms present')

    Propionic anhydride (C6H10O3):

    >>> smarts_fragment(catalog=J_BIGGS_JOBACK_SMARTS_id_dict, smi='CCC(=O)OC(=O)CC')
    ({1: 2, 2: 2, 28: 2}, False, 'Matched some atoms repeatedly: [4]')
    '''
    if not loaded_rdkit:
        load_rdkit_modules()
    if rdkitmol is None and smi is None:
        raise Exception('Either an rdkit mol or a smiles string is required')
    if smi is not None:
        rdkitmol = Chem.MolFromSmiles(smi)
        if rdkitmol is None:
            status = 'Failed to construct mol'
            success = False
            return {}, success, status
    from collections import Counter

    atom_count = len(rdkitmol.GetAtoms())
    status = 'OK'
    success = True

    counts = {}
    all_matches = {}
    for key, smart in catalog.items():
        if isinstance(smart, str):
            patt = Chem.MolFromSmarts(smart)
        else:
            patt = smart
        hits = list(rdkitmol.GetSubstructMatches(patt))
        if hits:
            all_matches[key] = hits
            counts[key] = len(hits)

    # Duplicate group cleanup
    matched_atoms = []
    for i in all_matches.values():
        for j in i:
            matched_atoms.extend(j)

    if deduplicate:
        dups = [i for i, c in Counter(matched_atoms).items() if c > 1]
        iteration = 0
        while (dups and iteration < 100):
            dup = dups[0]

            dup_smart_matches = []
            for group, group_match_list in all_matches.items():
                for i, group_match_i in enumerate(group_match_list):
                    if dup in group_match_i:
                        dup_smart_matches.append((group, i, group_match_i, len(group_match_i)))


            sizes = [i[3] for i in dup_smart_matches]
            max_size = max(sizes)
#            print(sizes, 'sizes', 'dup', dup, 'working_data', dup_smart_matches)
            if sizes.count(max_size) > 1:
                iteration += 1
#                print('BAD')
                # Two same size groups, continue, can't do anything
                continue
            else:
                # Remove matches that are not the largest
                max_idx = sizes.index(max_size)
                for group, idx, positions, size in dup_smart_matches:
                    if size != max_size:
                        # Not handling the case of multiple duplicate matches right, indexes changing!!!
                        del all_matches[group][idx]
                        continue

            matched_atoms = []
            for i in all_matches.values():
                for j in i:
                    matched_atoms.extend(j)

            dups = [i for i, c in Counter(matched_atoms).items() if c > 1]
            iteration += 1

    matched_atoms = set()
    for i in all_matches.values():
        for j in i:
            matched_atoms.update(j)
    if len(matched_atoms) != atom_count:
        status = 'Did not match all atoms present'
        success = False

    # Check the atom aount again, this time looking for duplicate matches (only if have yet to fail)
    if success:
        matched_atoms = []
        for i in all_matches.values():
            for j in i:
                matched_atoms.extend(j)
        if len(matched_atoms) < atom_count:
            status = 'Matched %d of %d atoms only' %(len(matched_atoms), atom_count)
            success = False
        elif len(matched_atoms) > atom_count:
            status = 'Matched some atoms repeatedly: %s' %( [i for i, c in Counter(matched_atoms).items() if c > 1])
            success = False

    return counts, success, status


class Joback(object):
    r'''Class for performing chemical property estimations with the Joback
    group contribution method as described in [1]_ and [2]_. This is a very
    common method with low accuracy but wide applicability. This routine can be
    used with either its own automatic fragmentation routine, or user specified
    groups. It is applicable to organic compounds only, and has only 41 groups
    with no interactions between them. Each method's documentation describes
    its accuracy. The automatic fragmentation routine is possible only because
    of the development of SMARTS expressions to match the Joback groups by
    Dr. Jason Biggs. The list of SMARTS expressions
    was posted publically on the
    `RDKit mailing list <https://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg07446.html>`_.

    Parameters
    ----------
    mol : rdkitmol or smiles str
        Input molecule for the analysis, [-]
    atom_count : int, optional
        The total number of atoms including hydrogen in the molecule; this will
        be counted by rdkit if it not provided, [-]
    MW : float, optional
        Molecular weight of the molecule; this will be calculated by rdkit if
        not provided, [g/mol]
    Tb : float, optional
        An experimentally known boiling temperature for the chemical; this
        increases the accuracy of the calculated critical point if provided.
        [K]

    Notes
    -----
    Be sure to check the status of the automatic fragmentation; not all
    chemicals with the Joback method are applicable.

    Approximately 68% of chemcials in the thermo database seem to be able to
    be estimated with the Joback method.

    If a group which was identified is missign a regressed contribution, the
    estimated property will be None. However, if not all atoms of a molecule
    are identified as particular groups, property estimation will go ahead
    with heavily reduced accuracy. Check the `status` attribute to be sure
    a molecule was properly fragmented.

    Examples
    --------
    Analysis of Acetone:

    >>> J = Joback('CC(=O)C')
    >>> J.Hfus(J.counts)
    5125.0
    >>> J.Cpig(350)
    84.69109750000001
    >>> J.status
    'OK'

    All properties can be obtained in one go with the `estimate` method:

    >>> J.estimate(callables=False)
    {'Tb': 322.11, 'Tm': 173.5, 'Tc': 500.5590049525365, 'Pc': 4802499.604994407, 'Vc': 0.0002095, 'Hf': -217829.99999999997, 'Gf': -154540.00000000003, 'Hfus': 5125.0, 'Hvap': 29018.0, 'mul_coeffs': [839.1099999999998, -14.99], 'Cpig_coeffs': [7.520000000000003, 0.26084, -0.0001207, 1.545999999999998e-08]}


    The results for propionic anhydride (if the status is not OK) should not be
    used.

    >>> J = Joback('CCC(=O)OC(=O)CC')
    >>> J.status
    'Matched some atoms repeatedly: [4]'
    >>> J.Cpig(300)
    175.85999999999999

    None of the routines need to use the automatic routine; they can be used
    manually too:

    >>> Joback.Tb({1: 2, 24: 1})
    322.11

    References
    ----------
    .. [1] Joback, Kevin G. "A Unified Approach to Physical Property Estimation
       Using Multivariate Statistical Techniques." Thesis, Massachusetts
       Institute of Technology, 1984.
    .. [2] Joback, K.G., and R.C. Reid. "Estimation of Pure-Component
       Properties from Group-Contributions." Chemical Engineering
       Communications 57, no. 1-6 (July 1, 1987): 233-43.
       doi:10.1080/00986448708960487.
    '''
    calculated_Cpig_coeffs = None
    calculated_mul_coeffs = None

    def __init__(self, mol, atom_count=None, MW=None, Tb=None):
        if not loaded_rdkit:
            load_rdkit_modules()

        if type(mol) == Chem.rdchem.Mol:
            self.rdkitmol = mol
        else:
            self.rdkitmol = Chem.MolFromSmiles(mol)
        if atom_count is None:
            self.rdkitmol_Hs = Chem.AddHs(self.rdkitmol)
            self.atom_count = len(self.rdkitmol_Hs.GetAtoms())
        else:
            self.atom_count = atom_count
        if MW is None:
            self.MW = rdMolDescriptors.CalcExactMolWt(self.rdkitmol_Hs)
        else:
            self.MW = MW

        self.counts, self.success, self.status = smarts_fragment(J_BIGGS_JOBACK_SMARTS_id_dict, rdkitmol=self.rdkitmol)

        if Tb is not None:
            self.Tb_estimated = self.Tb(self.counts)
        else:
            self.Tb_estimated = Tb

    def estimate(self, callables=True):
        '''Method to compute all available properties with the Joback method;
        returns their results as a dict. For the tempearture dependent values
        Cpig and mul, both the coefficients and objects to perform calculations
        are returned.
        '''
        if not self.counts:
            raise ValueError("Zero matching groups identified")
        # Pre-generate the coefficients or they will not be returned
        self.mul(300)
        self.Cpig(300)
        estimates = {'Tb': self.Tb(self.counts),
                     'Tm': self.Tm(self.counts),
                     'Tc': self.Tc(self.counts, self.Tb_estimated),
                     'Pc': self.Pc(self.counts, self.atom_count),
                     'Vc': self.Vc(self.counts),
                     'Hf': self.Hf(self.counts),
                     'Gf': self.Gf(self.counts),
                     'Hfus': self.Hfus(self.counts),
                     'Hvap': self.Hvap(self.counts),
                     'mul_coeffs': self.calculated_mul_coeffs,
                     'Cpig_coeffs': self.calculated_Cpig_coeffs}
        if callables:
            estimates['mul'] = self.mul
            estimates['Cpig'] = self.Cpig
        return estimates

    @staticmethod
    def Tb(counts):
        r'''Estimates the normal boiling temperature of an organic compound
        using the Joback method as a function of chemical structure only.

        .. math::
            T_b = 198.2 + \sum_i {T_{b,i}}

        For 438 compounds tested by Joback, the absolute average error was
        12.91 K  and standard deviation was 17.85 K; the average relative error
        was 3.6%.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]

        Returns
        -------
        Tb : float
            Estimated normal boiling temperature, [K]

        Examples
        --------
        >>> Joback.Tb({1: 2, 24: 1})
        322.11
        '''
        try:
            tot = 0.0
            for group, count in counts.items():
                tot += joback_groups_id_dict[group].Tb*count
            Tb = 198.2 + tot
            return Tb
        except:
            return None

    @staticmethod
    def Tm(counts):
        r'''Estimates the melting temperature of an organic compound using the
        Joback method as a function of chemical structure only.

        .. math::
            T_m = 122.5 + \sum_i {T_{m,i}}

        For 388 compounds tested by Joback, the absolute average error was
        22.6 K  and standard deviation was 24.68 K; the average relative error
        was 11.2%.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]

        Returns
        -------
        Tm : float
            Estimated melting temperature, [K]

        Examples
        --------
        >>> Joback.Tm({1: 2, 24: 1})
        173.5
        '''
        try:
            tot = 0.0
            for group, count in counts.items():
                tot += joback_groups_id_dict[group].Tm*count
            Tm = 122.5 + tot
            return Tm
        except:
            return None

    @staticmethod
    def Tc(counts, Tb=None):
        r'''Estimates the critcal temperature of an organic compound using the
        Joback method as a function of chemical structure only, or optionally
        improved by using an experimental boiling point. If the experimental
        boiling point is not provided it will be estimated with the Joback
        method as well.

        .. math::
            T_c = T_b \left[0.584 + 0.965 \sum_i {T_{c,i}}
            - \left(\sum_i {T_{c,i}}\right)^2 \right]^{-1}

        For 409 compounds tested by Joback, the absolute average error was
        4.76 K  and standard deviation was 6.94 K; the average relative error
        was 0.81%.

        Appendix BI of Joback's work lists 409 estimated critical temperatures.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]
        Tb : float, optional
            Experimental normal boiling temperature, [K]

        Returns
        -------
        Tc : float
            Estimated critical temperature, [K]

        Examples
        --------
        >>> Joback.Tc({1: 2, 24: 1}, Tb=322.11)
        500.5590049525365
        '''
        try:
            if Tb is None:
                Tb = Joback.Tb(counts)
            tot = 0.0
            for group, count in counts.items():
                tot += joback_groups_id_dict[group].Tc*count
            Tc = Tb/(0.584 + 0.965*tot - tot*tot)
            return Tc
        except:
            return None

    @staticmethod
    def Pc(counts, atom_count):
        r'''Estimates the critcal pressure of an organic compound using the
        Joback method as a function of chemical structure only. This
        correlation was developed using the actual number of atoms forming
        the molecule as well.

        .. math::
            P_c = \left [0.113 + 0.0032N_A - \sum_i {P_{c,i}}\right ]^{-2}

        In the above equation, critical pressure is calculated in bar; it is
        converted to Pa here.

        392 compounds were used by Joback in this determination, with an
        absolute average error of 2.06 bar, standard devaition 3.2 bar, and
        AARE of 5.2%.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]
        atom_count : int
            Total number of atoms (including hydrogens) in the molecule, [-]

        Returns
        -------
        Pc : float
            Estimated critical pressure, [Pa]

        Examples
        --------
        >>> Joback.Pc({1: 2, 24: 1}, 10)
        4802499.604994407
        '''
        try:
            tot = 0.0
            for group, count in counts.items():
                tot += joback_groups_id_dict[group].Pc*count
            Pc = (0.113 + 0.0032*atom_count - tot)**-2
            return Pc*1E5 # bar to Pa
        except:
            return None

    @staticmethod
    def Vc(counts):
        r'''Estimates the critcal volume of an organic compound using the
        Joback method as a function of chemical structure only.

        .. math::
            V_c = 17.5 + \sum_i {V_{c,i}}

        In the above equation, critical volume is calculated in cm^3/mol; it
        is converted to m^3/mol here.

        310 compounds were used by Joback in this determination, with an
        absolute average error of 7.54 cm^3/mol, standard devaition 13.16
        cm^3/mol, and AARE of 2.27%.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]

        Returns
        -------
        Vc : float
            Estimated critical volume, [m^3/mol]

        Examples
        --------
        >>> Joback.Vc({1: 2, 24: 1})
        0.0002095
        '''
        try:
            tot = 0.0
            for group, count in counts.items():
                tot += joback_groups_id_dict[group].Vc*count
            Vc = 17.5 + tot
            return Vc*1E-6 # cm^3/mol to m^3/mol
        except:
            return None

    @staticmethod
    def Hf(counts):
        r'''Estimates the ideal-gas enthalpy of formation at 298.15 K of an
        organic compound using the Joback method as a function of chemical
        structure only.

        .. math::
            H_{formation} = 68.29 + \sum_i {H_{f,i}}

        In the above equation, enthalpy of formation is calculated in kJ/mol;
        it is converted to J/mol here.

        370 compounds were used by Joback in this determination, with an
        absolute average error of 2.2 kcal/mol, standard devaition 2.0
        kcal/mol, and AARE of 15.2%.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]

        Returns
        -------
        Hf : float
            Estimated ideal-gas enthalpy of formation at 298.15 K, [J/mol]

        Examples
        --------
        >>> Joback.Hf({1: 2, 24: 1})
        -217829.99999999997
        '''
        try:
            tot = 0.0
            for group, count in counts.items():
                tot += joback_groups_id_dict[group].Hform*count
            Hf = 68.29 + tot
            return Hf*1000 # kJ/mol to J/mol
        except:
            return None

    @staticmethod
    def Gf(counts):
        r'''Estimates the ideal-gas Gibbs energy of formation at 298.15 K of an
        organic compound using the Joback method as a function of chemical
        structure only.

        .. math::
            G_{formation} = 53.88 + \sum {G_{f,i}}

        In the above equation, Gibbs energy of formation is calculated in
        kJ/mol; it is converted to J/mol here.

        328 compounds were used by Joback in this determination, with an
        absolute average error of 2.0 kcal/mol, standard devaition 4.37
        kcal/mol, and AARE of 15.7%.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]

        Returns
        -------
        Gf : float
            Estimated ideal-gas Gibbs energy of formation at 298.15 K, [J/mol]

        Examples
        --------
        >>> Joback.Gf({1: 2, 24: 1})
        -154540.00000000003
        '''
        try:
            tot = 0.0
            for group, count in counts.items():
                tot += joback_groups_id_dict[group].Gform*count
            Gf = 53.88 + tot
            return Gf*1000 # kJ/mol to J/mol
        except:
            return None

    @staticmethod
    def Hfus(counts):
        r'''Estimates the enthalpy of fusion of an organic compound at its
        melting point using the Joback method as a function of chemical
        structure only.

        .. math::
            \Delta H_{fus} = -0.88 + \sum_i H_{fus,i}

        In the above equation, enthalpy of fusion is calculated in
        kJ/mol; it is converted to J/mol here.

        For 155 compounds tested by Joback, the absolute average error was
        485.2 cal/mol  and standard deviation was 661.4 cal/mol; the average
        relative error was 38.7%.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]

        Returns
        -------
        Hfus : float
            Estimated enthalpy of fusion of the compound at its melting point,
            [J/mol]

        Examples
        --------
        >>> Joback.Hfus({1: 2, 24: 1})
        5125.0
        '''
        try:
            tot = 0.0
            for group, count in counts.items():
                tot += joback_groups_id_dict[group].Hfus*count
            Hfus = -0.88 + tot
            return Hfus*1000 # kJ/mol to J/mol
        except:
            return None

    @staticmethod
    def Hvap(counts):
        r'''Estimates the enthalpy of vaporization of an organic compound at
        its normal boiling point using the Joback method as a function of
        chemical structure only.

        .. math::
            \Delta H_{vap} = 15.30 + \sum_i H_{vap,i}

        In the above equation, enthalpy of fusion is calculated in
        kJ/mol; it is converted to J/mol here.

        For 368 compounds tested by Joback, the absolute average error was
        303.5 cal/mol  and standard deviation was 429 cal/mol; the average
        relative error was 3.88%.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]

        Returns
        -------
        Hvap : float
            Estimated enthalpy of vaporization of the compound at its normal
            boiling point, [J/mol]

        Examples
        --------
        >>> Joback.Hvap({1: 2, 24: 1})
        29018.0
        '''
        try:
            tot = 0.0
            for group, count in counts.items():
                tot += joback_groups_id_dict[group].Hvap*count
            Hvap = 15.3 + tot
            return Hvap*1000 # kJ/mol to J/mol
        except:
            return None

    @staticmethod
    def Cpig_coeffs(counts):
        r'''Computes the ideal-gas polynomial heat capacity coefficients
        of an organic compound using the Joback method as a function of
        chemical structure only.

        .. math::
            C_p^{ig} = \sum_i a_i - 37.93 + \left[ \sum_i b_i + 0.210 \right] T
            + \left[ \sum_i c_i - 3.91 \cdot 10^{-4} \right] T^2
            + \left[\sum_i d_i + 2.06 \cdot 10^{-7}\right] T^3

        288 compounds were used by Joback in this determination. No overall
        error was reported.

        The ideal gas heat capacity values used in developing the heat
        capacity polynomials used 9 data points between 298 K and 1000 K.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]

        Returns
        -------
        coefficients : list[float]
            Coefficients which will result in a calculated heat capacity in
            in units of J/mol/K, [-]

        Examples
        --------
        >>> c = Joback.Cpig_coeffs({1: 2, 24: 1})
        >>> c
        [7.520000000000003, 0.26084, -0.0001207, 1.545999999999998e-08]
        >>> Cp = lambda T : c[0] + c[1]*T + c[2]*T**2 + c[3]*T**3
        >>> Cp(300)
        75.32642000000001
        '''
        try:
            a, b, c, d = 0.0, 0.0, 0.0, 0.0
            for group, count in counts.items():
                a += joback_groups_id_dict[group].Cpa*count
                b += joback_groups_id_dict[group].Cpb*count
                c += joback_groups_id_dict[group].Cpc*count
                d += joback_groups_id_dict[group].Cpd*count
            a -= 37.93
            b += 0.210
            c -= 3.91E-4
            d += 2.06E-7
            return [a, b, c, d]
        except:
            return None

    @staticmethod
    def mul_coeffs(counts):
        r'''Computes the liquid phase viscosity Joback coefficients
        of an organic compound using the Joback method as a function of
        chemical structure only.

        .. math::
            \mu_{liq} = \text{MW} \exp\left( \frac{ \sum_i \mu_a - 597.82}{T}
            + \sum_i \mu_b - 11.202 \right)

        288 compounds were used by Joback in this determination. No overall
        error was reported.

        The liquid viscosity data used was specified to be at "several
        temperatures for each compound" only. A small and unspecified number
        of compounds were used in this estimation.

        Parameters
        ----------
        counts : dict
            Dictionary of Joback groups present (numerically indexed) and their
            counts, [-]

        Returns
        -------
        coefficients : list[float]
            Coefficients which will result in a liquid viscosity in
            in units of Pa*s, [-]

        Examples
        --------
        >>> mu_ab = Joback.mul_coeffs({1: 2, 24: 1})
        >>> mu_ab
        [839.1099999999998, -14.99]
        >>> MW = 58.041864812
        >>> mul = lambda T : MW*exp(mu_ab[0]/T + mu_ab[1])
        >>> mul(300)
        0.0002940378347162687
        '''
        try:
            a, b = 0.0, 0.0
            for group, count in counts.items():
                a += joback_groups_id_dict[group].mua*count
                b += joback_groups_id_dict[group].mub*count
            a -= 597.82
            b -= 11.202
            return [a, b]
        except:
            return None

    def Cpig(self, T):
        r'''Computes ideal-gas heat capacity at a specified temperature
        of an organic compound using the Joback method as a function of
        chemical structure only.

        .. math::
            C_p^{ig} = \sum_i a_i - 37.93 + \left[ \sum_i b_i + 0.210 \right] T
            + \left[ \sum_i c_i - 3.91 \cdot 10^{-4} \right] T^2
            + \left[\sum_i d_i + 2.06 \cdot 10^{-7}\right] T^3

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        Cpig : float
            Ideal-gas heat capacity, [J/mol/K]

        Examples
        --------
        >>> J = Joback('CC(=O)C')
        >>> J.Cpig(300)
        75.32642000000001
        '''
        try:
            if self.calculated_Cpig_coeffs is None:
                self.calculated_Cpig_coeffs = Joback.Cpig_coeffs(self.counts)
            return horner(reversed(self.calculated_Cpig_coeffs), T)
        except:
            return None

    def mul(self, T):
        r'''Computes liquid viscosity at a specified temperature
        of an organic compound using the Joback method as a function of
        chemical structure only.

        .. math::
            \mu_{liq} = \text{MW} \exp\left( \frac{ \sum_i \mu_a - 597.82}{T}
            + \sum_i \mu_b - 11.202 \right)

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        mul : float
            Liquid viscosity, [Pa*s]

        Examples
        --------
        >>> J = Joback('CC(=O)C')
        >>> J.mul(300)
        0.0002940378347162687
        '''
        try:
            if self.calculated_mul_coeffs is None:
                self.calculated_mul_coeffs = Joback.mul_coeffs(self.counts)
            a, b = self.calculated_mul_coeffs
            return self.MW*exp(a/T + b)
        except:
            return None

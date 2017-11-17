# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = []
from collections import namedtuple
from thermo import to_num

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    hasRDKit = True
except:
    # pragma: no cover
    hasRDKit = False


J_BIGGS_JOBACK_SMARTS = [["Methyl","-CH3", "[CX4H3]"],
["SecondaryAcyclic", "-CH2-", "[!R;CX4H2]"],
["TertiaryAcyclic",">CH-", "[!R;CX4H]"],
["QuaternaryAcyclic", ">C<", "[!R;CX4H0]"],
["PrimaryAlkene", "=CH2", "[CX3H2]"],
["SecondaryAlkeneAcyclic", "=CH-", "[!R;CX3H1;!$([CX3H1](=O))]"],
["TertiaryAlkeneAcyclic", "=C<", "[$([!R;#6X3H0]);!$([!R;#6X3H0]=[#8])]"],
["CumulativeAlkene", "=C=", "[$([CX2H0](=*)=*)]"],
["TerminalAlkyne", u"≡CH","[$([CX2H1]#[!#7])]"],
["InternalAlkyne",u"≡C-","[$([CX2H0]#[!#7])]"],
["SecondaryCyclic", "-CH2- (ring)", "[R;CX4H2]"],
["TertiaryCyclic", ">CH- (ring)", "[R;CX4H]"],
["QuaternaryCyclic", ">C< (ring)", "[R;CX4H0]"],
["SecondaryAlkeneCyclic", "=CH- (ring)", "[R;CX3H1,cX3H1]"],
["TertiaryAlkeneCyclic", "=C< (ring)","[$([R;#6X3H0]);!$([R;#6X3H0]=[#8])]"],
["Fluoro", "-F", "[F]"],
["Chloro", "-Cl", "[Cl]"],
["Bromo", "-Br", "[Br]"],
["Iodo", "-I", "[I]"],
["Alcohol","-OH (alcohol)", "[OX2H;!$([OX2H]-[#6]=[O]);!$([OX2H]-a)]"],
["Phenol","-OH (phenol)", "[$([OX2H]-a)]"],
["EtherAcyclic", "-O- (nonring)", "[OX2H0;!R;!$([OX2H0]-[#6]=[#8])]"],
["EtherCyclic", "-O- (ring)", "[#8X2H0;R;!$([#8X2H0]~[#6]=[#8])]"],
["CarbonylAcyclic", ">C=O (nonring)","[$([CX3H0](=[OX1]));!$([CX3](=[OX1])-[OX2]);!R]=O"],
["CarbonylCyclic", ">C=O (ring)","[$([#6X3H0](=[OX1]));!$([#6X3](=[#8X1])~[#8X2]);R]=O"],
["Aldehyde","O=CH- (aldehyde)","[CX3H1](=O)"],
["CarboxylicAcid", "-COOH (acid)", "[OX2H]-[C]=O"],
["Ester", "-COO- (ester)", "[#6X3H0;!$([#6X3H0](~O)(~O)(~O))](=[#8X1])[#8X2H0]"],
["OxygenDoubleBondOther", "=O (other than above)","[OX1H0;!$([OX1H0]~[#6X3]);!$([OX1H0]~[#7X3]~[#8])]"],
["PrimaryAmino","-NH2", "[NX3H2]"],
["SecondaryAminoAcyclic",">NH (nonring)", "[NX3H1;!R]"],
["SecondaryAminoCyclic",">NH (ring)", "[#7X3H1;R]"],
["TertiaryAmino", ">N- (nonring)","[#7X3H0;!$([#7](~O)~O)]"], 
["ImineAcyclic","-N= (nonring)","[#7X2H0;!R]"],
["ImineCyclic","-N= (ring)","[#7X2H0;R]"],
["Aldimine", "=NH", "[#7X2H1]"],
["Cyano", "-CN","[#6X2]#[#7X1H0]"],
["Nitro", "-NO2", "[$([#7X3,#7X3+][!#8])](=[O])~[O-]"],
["Thiol", "-SH", "[SX2H]"],
["ThioetherAcyclic", "-S- (nonring)", "[#16X2H0;!R]"],
["ThioetherCyclic", "-S- (ring)", "[#16X2H0;R]"]]


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

joback_groups_list = []
joback_groups_dict = {}
JOBACK = namedtuple('JOBACK', 'i, name, Tc, Pc, Vc, Tb, Tm, Hform, Gform, Cpa, Cpb, Cpc, Cpd, Hfusion, Hvap, mua, mub')
for i, line in enumerate(joback_data_txt.split('\n')):
    parsed = to_num(line.split('\t'))
    j = JOBACK(i+1, *parsed)
    joback_groups_list.append(j)
    joback_groups_dict[parsed[0]] = j

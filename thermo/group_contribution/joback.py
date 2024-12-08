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


.. autoclass:: thermo.group_contribution.joback.Joback
    :members:
    :undoc-members:
    :show-inheritance:

.. autodata:: J_BIGGS_JOBACK_SMARTS
.. autodata:: J_BIGGS_JOBACK_SMARTS_id_dict



'''


__all__ = ['Joback', 'J_BIGGS_JOBACK_SMARTS',
           'J_BIGGS_JOBACK_SMARTS_id_dict']

from fluids.numerics import exp, horner

from thermo.group_contribution.group_contribution_base import smarts_fragment

rdkit_missing = 'RDKit is not installed; it is required to use this functionality'

loaded_rdkit = False
Chem, Descriptors, AllChem, rdMolDescriptors = None, None, None, None
def load_rdkit_modules():
    global loaded_rdkit, Chem, Descriptors, AllChem, rdMolDescriptors
    if loaded_rdkit:
        return
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
        loaded_rdkit = True
    except:
        if not loaded_rdkit: # pragma: no cover
            raise Exception(rdkit_missing)

# See https://www.atmos-chem-phys.net/16/4401/2016/acp-16-4401-2016.pdf for more
# smarts patterns
# TODO switch to https://www.ncbi.nlm.nih.gov/labs/pmc/articles/PMC6645593/

J_BIGGS_JOBACK_SMARTS = [["Methyl","-CH3", "[CX4H3]"],
["Secondary acyclic", "-CH2-", "[!R;CX4H2]"],
["Tertiary acyclic",">CH-", "[!R;CX4H]"],
["Quaternary acyclic", ">C<", "[!R;CX4H0]"],

["Primary alkene", "=CH2", "[CX3H2]"],
["Secondary alkene acyclic", "=CH-", "[!R;CX3H1;!$([CX3H1](=O))]"],
["Tertiary alkene acyclic", "=C<", "[$([!R;#6X3H0]);!$([!R;#6X3H0]=[#8])]"], # corrected to be that of JRgui, match C element instead of aliphatic C (still non ring)
["Cumulative alkene", "=C=", "[$([CX2H0](=*)=*)]"],
["Terminal alkyne", "≡CH","[$([CX2H1]#[!#7])]"],
["Internal alkyne","≡C-","[$([CX2H0]#[!#7])]"],

["Secondary cyclic", "-CH2- (ring)", "[R;CX4H2]"],
["Tertiary cyclic", ">CH- (ring)", "[R;CX4H]"],
["Quaternary cyclic", ">C< (ring)", "[R;CX4H0]"],

["Secondary alkene cyclic", "=CH- (ring)", "[R;CX3H1,cX3H1]"],
["Tertiary alkene cyclic", "=C< (ring)","[$([R;#6X3H0]);!$([R;#6X3H0]=[#8])]"], # corrected to be that of JRgui, match C element instead of aliphatic C (still ring)

["Fluoro", "-F", "[F]"],
["Chloro", "-Cl", "[Cl]"],
["Bromo", "-Br", "[Br]"],
["Iodo", "-I", "[I]"],

["Alcohol","-OH (alcohol)", "[OX2H;!$([OX2H]-[#6]=[O]);!$([OX2H]-a)]"],
["Phenol","-OH (phenol)", "[$([OX2H]-a)]"], # [O;H1;$(O-!@c)] is suggested in JRGui which is slightly less permissive, see https://pubchem.ncbi.nlm.nih.gov/compound/4-Hydroxyaminoquinoline-1-oxide-hydrochloride as example of compound that won't match anymore
["Ether acyclic", "-O- (nonring)", "[OX2H0;!R;!$([OX2H0]-[#6]=[#8])]"],
["Ether cyclic", "-O- (ring)", "[#8X2H0;R;!$([#8X2H0]~[#6]=[#8])]"],
["Carbonyl acyclic", ">C=O (nonring)","[$([CX3H0](=[OX1]));!$([CX3](=[OX1])-[OX2]);!R]=O"],
["Carbonyl cyclic", ">C=O (ring)","[$([#6X3H0](=[OX1]));!$([#6X3](=[#8X1])~[#8X2]);R]=O"],
["Aldehyde","O=CH- (aldehyde)","[CH;D2;$(C-!@C)](=O)"], # Updated to the one in JRGui which is much more correct, e.g. formic acid needs the updated change
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
"""Metadata for the Joback groups. The first element is the group name; the
second is the group symbol; and the third is the SMARTS matching string.
"""

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


class JOBACK:
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
        return f"""JOBACK(i={self.i!r}, name={self.name!r}, Tc={self.Tc!r}, Pc={self.Pc!r}, Vc={self.Vc!r}, Tb={self.Tb!r}, Tm={self.Tm!r}, Hform={self.Hform!r}, Gform={self.Gform!r},
Cpa={self.Cpa!r}, Cpb={self.Cpb!r}, Cpc={self.Cpc!r}, Cpd={self.Cpd!r}, Hfus={self.Hfus!r}, Hvap={self.Hvap!r}, mua={self.mua!r}, mub={self.mub!r})"""

joback_groups_str_dict = {'-CH3': JOBACK(i=1, name='-CH3', Tc=0.0141, Pc=-0.0012, Vc=65.0, Tb=23.58, Tm=-5.1, Hform=-76.45, Gform=-43.96,
 Cpa=19.5, Cpb=-0.00808, Cpc=0.000153, Cpd=-9.67e-08, Hfus=0.908, Hvap=2.373, mua=548.29, mub=-1.719),
 '-CH2-': JOBACK(i=2, name='-CH2-', Tc=0.0189, Pc=0.0, Vc=56.0, Tb=22.88, Tm=11.27, Hform=-20.64, Gform=8.42,
 Cpa=-0.909, Cpb=0.095, Cpc=-5.44e-05, Cpd=1.19e-08, Hfus=2.59, Hvap=2.226, mua=94.16, mub=-0.199),
 '>CH-': JOBACK(i=3, name='>CH-', Tc=0.0164, Pc=0.002, Vc=41.0, Tb=21.74, Tm=12.64, Hform=29.89, Gform=58.36,
 Cpa=-23.0, Cpb=0.204, Cpc=-0.000265, Cpd=1.2e-07, Hfus=0.749, Hvap=1.691, mua=-322.15, mub=1.187),
 '>C<': JOBACK(i=4, name='>C<', Tc=0.0067, Pc=0.0043, Vc=27.0, Tb=18.25, Tm=46.43, Hform=82.23, Gform=116.02,
 Cpa=-66.2, Cpb=0.427, Cpc=-0.000641, Cpd=3.01e-07, Hfus=-1.46, Hvap=0.636, mua=-573.56, mub=2.307),
 '=CH2': JOBACK(i=5, name='=CH2', Tc=0.0113, Pc=-0.0028, Vc=56.0, Tb=18.18, Tm=-4.32, Hform=-9.63, Gform=3.77,
 Cpa=23.6, Cpb=-0.0381, Cpc=0.000172, Cpd=-1.03e-07, Hfus=-0.473, Hvap=1.724, mua=495.01, mub=-1.539),
 '=CH-': JOBACK(i=6, name='=CH-', Tc=0.0129, Pc=-0.0006, Vc=46.0, Tb=24.96, Tm=8.73, Hform=37.97, Gform=48.53,
 Cpa=-8.0, Cpb=0.105, Cpc=-9.63e-05, Cpd=3.56e-08, Hfus=2.691, Hvap=2.205, mua=82.28, mub=-0.242),
 '=C<': JOBACK(i=7, name='=C<', Tc=0.0117, Pc=0.0011, Vc=38.0, Tb=24.14, Tm=11.14, Hform=83.99, Gform=92.36,
 Cpa=-28.1, Cpb=0.208, Cpc=-0.000306, Cpd=1.46e-07, Hfus=3.063, Hvap=2.138, mua=None, mub=None),
 '=C=': JOBACK(i=8, name='=C=', Tc=0.0026, Pc=0.0028, Vc=36.0, Tb=26.15, Tm=17.78, Hform=142.14, Gform=136.7,
 Cpa=27.4, Cpb=-0.0557, Cpc=0.000101, Cpd=-5.02e-08, Hfus=4.72, Hvap=2.661, mua=None, mub=None),
 '≡CH': JOBACK(i=9, name='≡CH', Tc=0.0027, Pc=-0.0008, Vc=46.0, Tb=9.2, Tm=-11.18, Hform=79.3, Gform=77.71,
 Cpa=24.5, Cpb=-0.0271, Cpc=0.000111, Cpd=-6.78e-08, Hfus=2.322, Hvap=1.155, mua=None, mub=None),
 '≡C-': JOBACK(i=10, name='≡C-', Tc=0.002, Pc=0.0016, Vc=37.0, Tb=27.38, Tm=64.32, Hform=115.51, Gform=109.82,
 Cpa=7.87, Cpb=0.0201, Cpc=-8.33e-06, Cpd=1.39e-09, Hfus=4.151, Hvap=3.302, mua=None, mub=None),
 '-CH2- (ring)': JOBACK(i=11, name='-CH2- (ring)', Tc=0.01, Pc=0.0025, Vc=48.0, Tb=27.15, Tm=7.75, Hform=-26.8, Gform=-3.68,
 Cpa=-6.03, Cpb=0.0854, Cpc=-8e-06, Cpd=-1.8e-08, Hfus=0.49, Hvap=2.398, mua=307.53, mub=-0.798),
 '>CH- (ring)': JOBACK(i=12, name='>CH- (ring)', Tc=0.0122, Pc=0.0004, Vc=38.0, Tb=21.78, Tm=19.88, Hform=8.67, Gform=40.99,
 Cpa=-20.5, Cpb=0.162, Cpc=-0.00016, Cpd=6.24e-08, Hfus=3.243, Hvap=1.942, mua=-394.29, mub=1.251),
 '>C< (ring)': JOBACK(i=13, name='>C< (ring)', Tc=0.0042, Pc=0.0061, Vc=27.0, Tb=21.32, Tm=60.15, Hform=79.72, Gform=87.88,
 Cpa=-90.9, Cpb=0.557, Cpc=-0.0009, Cpd=4.69e-07, Hfus=-1.373, Hvap=0.644, mua=None, mub=None),
 '=CH- (ring)': JOBACK(i=14, name='=CH- (ring)', Tc=0.0082, Pc=0.0011, Vc=41.0, Tb=26.73, Tm=8.13, Hform=2.09, Gform=11.3,
 Cpa=-2.14, Cpb=0.0574, Cpc=-1.64e-06, Cpd=-1.59e-08, Hfus=1.101, Hvap=2.544, mua=259.65, mub=-0.702),
 '=C< (ring)': JOBACK(i=15, name='=C< (ring)', Tc=0.0143, Pc=0.0008, Vc=32.0, Tb=31.01, Tm=37.02, Hform=46.43, Gform=54.05,
 Cpa=-8.25, Cpb=0.101, Cpc=-0.000142, Cpd=6.78e-08, Hfus=2.394, Hvap=3.059, mua=-245.74, mub=0.912),
 '-F': JOBACK(i=16, name='-F', Tc=0.0111, Pc=-0.0057, Vc=27.0, Tb=-0.03, Tm=-15.78, Hform=-251.92, Gform=-247.19,
 Cpa=26.5, Cpb=-0.0913, Cpc=0.000191, Cpd=-1.03e-07, Hfus=1.398, Hvap=-0.67, mua=None, mub=None),
 '-Cl': JOBACK(i=17, name='-Cl', Tc=0.0105, Pc=-0.0049, Vc=58.0, Tb=38.13, Tm=13.55, Hform=-71.55, Gform=-64.31,
 Cpa=33.3, Cpb=-0.0963, Cpc=0.000187, Cpd=-9.96e-08, Hfus=2.515, Hvap=4.532, mua=625.45, mub=-1.814),
 '-Br': JOBACK(i=18, name='-Br', Tc=0.0133, Pc=0.0057, Vc=71.0, Tb=66.86, Tm=43.43, Hform=-29.48, Gform=-38.06,
 Cpa=28.6, Cpb=-0.0649, Cpc=0.000136, Cpd=-7.45e-08, Hfus=3.603, Hvap=6.582, mua=738.91, mub=-2.038),
 '-I': JOBACK(i=19, name='-I', Tc=0.0068, Pc=-0.0034, Vc=97.0, Tb=93.84, Tm=41.69, Hform=21.06, Gform=5.74,
 Cpa=32.1, Cpb=-0.0641, Cpc=0.000126, Cpd=-6.87e-08, Hfus=2.724, Hvap=9.52, mua=809.55, mub=-2.224),
 '-OH (alcohol)': JOBACK(i=20, name='-OH (alcohol)', Tc=0.0741, Pc=0.0112, Vc=28.0, Tb=92.88, Tm=44.45, Hform=-208.04, Gform=-189.2,
 Cpa=25.7, Cpb=-0.0691, Cpc=0.000177, Cpd=-9.88e-08, Hfus=2.406, Hvap=16.826, mua=2173.72, mub=-5.057),
 '-OH (phenol)': JOBACK(i=21, name='-OH (phenol)', Tc=0.024, Pc=0.0184, Vc=-25.0, Tb=76.34, Tm=82.83, Hform=-221.65, Gform=-197.37,
 Cpa=-2.81, Cpb=0.111, Cpc=-0.000116, Cpd=4.94e-08, Hfus=4.49, Hvap=12.499, mua=3018.17, mub=-7.314),
 '-O- (nonring)': JOBACK(i=22, name='-O- (nonring)', Tc=0.0168, Pc=0.0015, Vc=18.0, Tb=22.42, Tm=22.23, Hform=-132.22, Gform=-105.0,
 Cpa=25.5, Cpb=-0.0632, Cpc=0.000111, Cpd=-5.48e-08, Hfus=1.188, Hvap=2.41, mua=122.09, mub=-0.386),
 '-O- (ring)': JOBACK(i=23, name='-O- (ring)', Tc=0.0098, Pc=0.0048, Vc=13.0, Tb=31.22, Tm=23.05, Hform=-138.16, Gform=-98.22,
 Cpa=12.2, Cpb=-0.0126, Cpc=6.03e-05, Cpd=-3.86e-08, Hfus=5.879, Hvap=4.682, mua=440.24, mub=-0.953),
 '>C=O (nonring)': JOBACK(i=24, name='>C=O (nonring)', Tc=0.038, Pc=0.0031, Vc=62.0, Tb=76.75, Tm=61.2, Hform=-133.22, Gform=-120.5,
 Cpa=6.45, Cpb=0.067, Cpc=-3.57e-05, Cpd=2.86e-09, Hfus=4.189, Hvap=8.972, mua=340.35, mub=-0.35),
 '>C=O (ring)': JOBACK(i=25, name='>C=O (ring)', Tc=0.0284, Pc=0.0028, Vc=55.0, Tb=94.97, Tm=75.97, Hform=-164.5, Gform=-126.27,
 Cpa=30.4, Cpb=-0.0829, Cpc=0.000236, Cpd=-1.31e-07, Hfus=0.0, Hvap=6.645, mua=None, mub=None),
 'O=CH- (aldehyde)': JOBACK(i=26, name='O=CH- (aldehyde)', Tc=0.0379, Pc=0.003, Vc=82.0, Tb=72.24, Tm=36.9, Hform=-162.03, Gform=-143.48,
 Cpa=30.9, Cpb=-0.0336, Cpc=0.00016, Cpd=-9.88e-08, Hfus=3.197, Hvap=9.093, mua=740.92, mub=-1.713),
 '-COOH (acid)': JOBACK(i=27, name='-COOH (acid)', Tc=0.0791, Pc=0.0077, Vc=89.0, Tb=169.09, Tm=155.5, Hform=-426.72, Gform=-387.87,
 Cpa=24.1, Cpb=0.0427, Cpc=8.04e-05, Cpd=-6.87e-08, Hfus=11.051, Hvap=19.537, mua=1317.23, mub=-2.578),
 '-COO- (ester)': JOBACK(i=28, name='-COO- (ester)', Tc=0.0481, Pc=0.0005, Vc=82.0, Tb=81.1, Tm=53.6, Hform=-337.92, Gform=-301.95,
 Cpa=24.5, Cpb=0.0402, Cpc=4.02e-05, Cpd=-4.52e-08, Hfus=6.959, Hvap=9.633, mua=483.88, mub=-0.966),
 '=O (other than above)': JOBACK(i=29, name='=O (other than above)', Tc=0.0143, Pc=0.0101, Vc=36.0, Tb=-10.5, Tm=2.08, Hform=-247.61, Gform=-250.83,
 Cpa=6.82, Cpb=0.0196, Cpc=1.27e-05, Cpd=-1.78e-08, Hfus=3.624, Hvap=5.909, mua=675.24, mub=-1.34),
 '-NH2': JOBACK(i=30, name='-NH2', Tc=0.0243, Pc=0.0109, Vc=38.0, Tb=73.23, Tm=66.89, Hform=-22.02, Gform=14.07,
 Cpa=26.9, Cpb=-0.0412, Cpc=0.000164, Cpd=-9.76e-08, Hfus=3.515, Hvap=10.788, mua=None, mub=None),
 '>NH (nonring)': JOBACK(i=31, name='>NH (nonring)', Tc=0.0295, Pc=0.0077, Vc=35.0, Tb=50.17, Tm=52.66, Hform=53.47, Gform=89.39,
 Cpa=-1.21, Cpb=0.0762, Cpc=-4.86e-05, Cpd=1.05e-08, Hfus=5.099, Hvap=6.436, mua=None, mub=None),
 '>NH (ring)': JOBACK(i=32, name='>NH (ring)', Tc=0.013, Pc=0.0114, Vc=29.0, Tb=52.82, Tm=101.51, Hform=31.65, Gform=75.61,
 Cpa=11.8, Cpb=-0.023, Cpc=0.000107, Cpd=-6.28e-08, Hfus=7.49, Hvap=6.93, mua=None, mub=None),
 '>N- (nonring)': JOBACK(i=33, name='>N- (nonring)', Tc=0.0169, Pc=0.0074, Vc=9.0, Tb=11.74, Tm=48.84, Hform=123.34, Gform=163.16,
 Cpa=-31.1, Cpb=0.227, Cpc=-0.00032, Cpd=1.46e-07, Hfus=4.703, Hvap=1.896, mua=None, mub=None),
 '-N= (nonring)': JOBACK(i=34, name='-N= (nonring)', Tc=0.0255, Pc=-0.0099, Vc=None, Tb=74.6, Tm=None, Hform=23.61, Gform=None,
 Cpa=None, Cpb=None, Cpc=None, Cpd=None, Hfus=None, Hvap=3.335, mua=None, mub=None),
 '-N= (ring)': JOBACK(i=35, name='-N= (ring)', Tc=0.0085, Pc=0.0076, Vc=34.0, Tb=57.55, Tm=68.4, Hform=55.52, Gform=79.93,
 Cpa=8.83, Cpb=-0.00384, Cpc=4.35e-05, Cpd=-2.6e-08, Hfus=3.649, Hvap=6.528, mua=None, mub=None),
 '=NH': JOBACK(i=36, name='=NH', Tc=None, Pc=None, Vc=None, Tb=83.08, Tm=68.91, Hform=93.7, Gform=119.66,
 Cpa=5.69, Cpb=-0.00412, Cpc=0.000128, Cpd=-8.88e-08, Hfus=None, Hvap=12.169, mua=None, mub=None),
 '-CN': JOBACK(i=37, name='-CN', Tc=0.0496, Pc=-0.0101, Vc=91.0, Tb=125.66, Tm=59.89, Hform=88.43, Gform=89.22,
 Cpa=36.5, Cpb=-0.0733, Cpc=0.000184, Cpd=-1.03e-07, Hfus=2.414, Hvap=12.851, mua=None, mub=None),
 '-NO2': JOBACK(i=38, name='-NO2', Tc=0.0437, Pc=0.0064, Vc=91.0, Tb=152.54, Tm=127.24, Hform=-66.57, Gform=-16.83,
 Cpa=25.9, Cpb=-0.00374, Cpc=0.000129, Cpd=-8.88e-08, Hfus=9.679, Hvap=16.738, mua=None, mub=None),
 '-SH': JOBACK(i=39, name='-SH', Tc=0.0031, Pc=0.0084, Vc=63.0, Tb=63.56, Tm=20.09, Hform=-17.33, Gform=-22.99,
 Cpa=35.3, Cpb=-0.0758, Cpc=0.000185, Cpd=-1.03e-07, Hfus=2.36, Hvap=6.884, mua=None, mub=None),
 '-S- (nonring)': JOBACK(i=40, name='-S- (nonring)', Tc=0.0119, Pc=0.0049, Vc=54.0, Tb=68.78, Tm=34.4, Hform=41.87, Gform=33.12,
 Cpa=19.6, Cpb=-0.00561, Cpc=4.02e-05, Cpd=-2.76e-08, Hfus=4.13, Hvap=6.817, mua=None, mub=None),
 '-S- (ring)': JOBACK(i=41, name='-S- (ring)', Tc=0.0019, Pc=0.0051, Vc=38.0, Tb=52.1, Tm=79.93, Hform=39.1, Gform=27.76,
 Cpa=16.7, Cpb=0.00481, Cpc=2.77e-05, Cpd=-2.11e-08, Hfus=1.557, Hvap=5.984, mua=None, mub=None)}

joback_groups_id_dict = {}
for j in joback_groups_str_dict.values():
    joback_groups_id_dict[j.i] = j


class Joback:
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

    >>> J = Joback('CC(=O)C') # doctest:+SKIP
    >>> J.Hfus(J.counts) # doctest:+SKIP
    5125.0
    >>> J.Cpig(350) # doctest:+SKIP
    84.69109750000001
    >>> J.status # doctest:+SKIP
    'OK'

    All properties can be obtained in one go with the `estimate` method:

    >>> J.estimate(callables=False) # doctest:+SKIP
    {'Tb': 322.11, 'Tm': 173.5, 'Tc': 500.5590049525365, 'Pc': 4802499.604994407, 'Vc': 0.0002095, 'Hf': -217829.99999999997, 'Gf': -154540.00000000003, 'Hfus': 5125.0, 'Hvap': 29018.0, 'mul_coeffs': [839.1099999999998, -14.99], 'Cpig_coeffs': [7.520000000000003, 0.26084, -0.0001207, 1.545999999999998e-08]}


    The results for propionic anhydride (if the status is not OK) should not be
    used.

    >>> J = Joback('CCC(=O)OC(=O)CC') # doctest:+SKIP
    >>> J.status # doctest:+SKIP
    'Matched some atoms repeatedly: [4]'
    >>> J.Cpig(300) # doctest:+SKIP
    175.85999999999999

    None of the routines need to use the automatic routine; they can be used
    manually too:

    >>> Joback.Tb({1: 2, 24: 1})
    322.11

    Example from [3]_ comparing against manually calculated literature values for 2-methylphenol
    (using a known `Tb` values for extra accuracy):

    >>> J = Joback('CC1=CC=CC=C1O',  Tb=464.15) # doctest:+SKIP
    >>> res = J.estimate(callables=False) # doctest:+SKIP
    >>> [res['Tc'], res['Pc']/1e5, res['Vc']*1e6] # doctest:+SKIP
    [692.64, 50.30, 285.5]
    >>> J.status # doctest:+SKIP
    'OK'

    This matches exactly with the values given in [3]_ from Joback's method. They also mention
    experimental values of Tc = 697.55 K, Pc = 50.10 bar, and Vc = 282 cm³/mol,
    values all within 1.5%.


    References
    ----------
    .. [1] Joback, Kevin G. "A Unified Approach to Physical Property Estimation
       Using Multivariate Statistical Techniques." Thesis, Massachusetts
       Institute of Technology, 1984.
    .. [2] Joback, K.G., and R.C. Reid. "Estimation of Pure-Component
       Properties from Group-Contributions." Chemical Engineering
       Communications 57, no. 1-6 (July 1, 1987): 233-43.
       doi:10.1080/00986448708960487.
    .. [3] Elliott, J. Richard, Vladimir Diky, Thomas A. Knotts IV, and 
       W. Vincent Wilding. The Properties of Gases and Liquids, Sixth 
       Edition. 6th edition. New York: McGraw Hill, 2023.
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

        if Tb is None:
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
        >>> J = Joback('CC(=O)C') # doctest:+SKIP
        >>> J.Cpig(300) # doctest:+SKIP
        75.32642000000001
        '''
        try:
            if self.calculated_Cpig_coeffs is None:
                self.calculated_Cpig_coeffs = Joback.Cpig_coeffs(self.counts)
            coeffs = list(reversed(self.calculated_Cpig_coeffs))
            return horner(coeffs, T)
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
        >>> J = Joback('CC(=O)C') # doctest:+SKIP
        >>> J.mul(300) # doctest:+SKIP
        0.0002940378347162687
        '''
        try:
            if self.calculated_mul_coeffs is None:
                self.calculated_mul_coeffs = Joback.mul_coeffs(self.counts)
            a, b = self.calculated_mul_coeffs
            return self.MW*exp(a/T + b)
        except:
            return None


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

__all__ = ['UNIFAC', 'UNIFAC_psi', 'DOUFMG', 'DOUFSG', 'UFSG', 'UFMG', 
           'DOUFIP2016', 'DOUFIP2006', 'UFIP']
import os
from thermo.utils import log, exp

folder = os.path.join(os.path.dirname(__file__), 'Phase Change')

def UNIFAC_psi(T, subgroup1, subgroup2, UFSG, UFIP, modified=False):
    main1 = UFSG[subgroup1].group_id
    main2 = UFSG[subgroup2].group_id
    if modified:
        try:
            a, b, c = UFIP[main1][main2]
        except:
            return 1.
        return exp((-a -b*T - c*T**2)/T)
    else:
        try:
            return exp(-UFIP[main1][main2]/T)
        except:
            return 1.


class UNIFAC_subgroup(object):
    __slots__ = ['group', 'group_id', 'subgroup', 'R', 'Q']
    def __init__(self, group, group_id, subgroup, R, Q):
        self.group = group
        self.group_id = group_id
        self.subgroup = subgroup
        self.R = R
        self.Q = Q

UFIP = {i: {} for i in list(range(1, 52)) + [55, 84, 85]}
with open(os.path.join(folder, 'UNIFAC original interaction parameters.tsv')) as f:
    for line in f:
        maingroup1, maingroup2, interaction_parameter = line.strip('\n').split('\t')
        # Index by both int, order maters, to only one parameter.
        UFIP[int(maingroup1)][int(maingroup2)] = float(interaction_parameter)

DOUFIP2006 = {i: {} for i in list(range(1, 51)) + [52, 53, 55, 56, 61, 77, 84, 85, 87, 89, 90, 91, 93, 98, 99]}
# Some of the groups have no public parameters unfortunately
with open(os.path.join(folder, 'UNIFAC modified Dortmund interaction parameters 2006.tsv')) as f:
    for line in f:
        maingroup1, maingroup2, a, b, c = line.strip('\n').split('\t')
        DOUFIP2006[int(maingroup1)][int(maingroup2)] = (float(a), float(b), float(c))

DOUFIP2016 = {i: {} for i in list(range(1, 51)) + [52, 53, 55, 56, 61, 77, 84, 85, 87, 89, 90, 91, 93, 98, 99]}
# Some of the groups have no public parameters unfortunately
with open(os.path.join(folder, 'UNIFAC modified Dortmund interaction parameters.tsv')) as f:
    for line in f:
        maingroup1, maingroup2, a, b, c = line.strip('\n').split('\t')
        DOUFIP2016[int(maingroup1)][int(maingroup2)] = (float(a), float(b), float(c))


# http://www.ddbst.com/published-parameters-unifac.html#ListOfMainGroups
#UFMG[No.] = ('Maingroup Name', subgroups)
UFMG = {}
UFMG[1] = ('CH2', (1, 2, 3, 4))
UFMG[2] = ('C=C', (5, 6, 7, 8, 70))
UFMG[3] = ('ACH', (9, 10))
UFMG[4] = ('ACCH2', (11, 12, 13))
UFMG[5] = ('OH', (14))
UFMG[6] = ('CH3OH', (15))
UFMG[7] = ('H2O', (16))
UFMG[8] = ('ACOH', (17))
UFMG[9] = ('CH2CO', (18, 19))
UFMG[10] = ('CHO', (20))
UFMG[11] = ('CCOO', (21, 22))
UFMG[12] = ('HCOO', (23))
UFMG[13] = ('CH2O', (24, 25, 26, 27))
UFMG[14] = ('CNH2', (28, 29, 30))
UFMG[15] = ('CNH', (31, 32, 33))
UFMG[16] = ('(C)3N', (34, 35))
UFMG[17] = ('ACNH2', (36))
UFMG[18] = ('PYRIDINE', (37, 38, 39))
UFMG[19] = ('CCN', (40, 41))
UFMG[20] = ('COOH', (42, 43))
UFMG[21] = ('CCL', (44, 45, 46))
UFMG[22] = ('CCL2', (47, 48, 49))
UFMG[23] = ('CCL3', (50, 51))
UFMG[24] = ('CCL4', (52))
UFMG[25] = ('ACCL', (53))
UFMG[26] = ('CNO2', (54, 55, 56))
UFMG[27] = ('ACNO2', (57))
UFMG[28] = ('CS2', (58))
UFMG[29] = ('CH3SH', (59, 60))
UFMG[30] = ('FURFURAL', (61))
UFMG[31] = ('DOH', (62))
UFMG[32] = ('I', (63))
UFMG[33] = ('BR', (64))
UFMG[34] = ('C=-C', (65, 66))
UFMG[35] = ('DMSO', (67))
UFMG[36] = ('ACRY', (68))
UFMG[37] = ('CLCC', (69))
UFMG[38] = ('ACF', (71))
UFMG[39] = ('DMF', (72, 73))
UFMG[40] = ('CF2', (74, 75, 76))
UFMG[41] = ('COO', (77))
UFMG[42] = ('SIH2', (78, 79, 80, 81))
UFMG[43] = ('SIO', (82, 83, 84))
UFMG[44] = ('NMP', (85))
UFMG[45] = ('CCLF', (86, 87, 88, 89, 90, 91, 92, 93))
UFMG[46] = ('CON(AM)', (94, 95, 96, 97, 98, 99))
UFMG[47] = ('OCCOH', (100, 101))
UFMG[48] = ('CH2S', (102, 103, 104))
UFMG[49] = ('MORPH', (105))
UFMG[50] = ('THIOPHEN', (106, 107, 108))
UFMG[51] = ('NCO', (109))
UFMG[55] = ('SULFONES', (118, 119))
UFMG[84] = ('IMIDAZOL', (178))
UFMG[85] = ('BTI', (179))


UFSG = {}
# UFSG[subgroup ID] = (subgroup formula, main group ID, subgroup R, subgroup Q)
# http://www.ddbst.com/published-parameters-unifac.html
UFSG[1] = UNIFAC_subgroup('CH3', 1, 'CH2', 0.9011, 0.848)
UFSG[2] = UNIFAC_subgroup('CH2', 1, 'CH2', 0.6744, 0.54)
UFSG[3] = UNIFAC_subgroup('CH', 1, 'CH2', 0.4469, 0.228)
UFSG[4] = UNIFAC_subgroup('C', 1, 'CH2', 0.2195, 0)
UFSG[5] = UNIFAC_subgroup('CH2=CH', 2, 'C=C', 1.3454, 1.176)
UFSG[6] = UNIFAC_subgroup('CH=CH', 2, 'C=C', 1.1167, 0.867)
UFSG[7] = UNIFAC_subgroup('CH2=C', 2, 'C=C', 1.1173, 0.988)
UFSG[8] = UNIFAC_subgroup('CH=C', 2, 'C=C', 0.8886, 0.676)
UFSG[9] = UNIFAC_subgroup('ACH', 3, 'ACH', 0.5313, 0.4)
UFSG[10] = UNIFAC_subgroup('AC', 3, 'ACH', 0.3652, 0.12)
UFSG[11] = UNIFAC_subgroup('ACCH3', 4, 'ACCH2', 1.2663, 0.968)
UFSG[12] = UNIFAC_subgroup('ACCH2', 4, 'ACCH2', 1.0396, 0.66)
UFSG[13] = UNIFAC_subgroup('ACCH', 4, 'ACCH2', 0.8121, 0.348)
UFSG[14] = UNIFAC_subgroup('OH', 5, 'OH', 1, 1.2)
UFSG[15] = UNIFAC_subgroup('CH3OH', 6, 'CH3OH', 1.4311, 1.432)
UFSG[16] = UNIFAC_subgroup('H2O', 7, 'H2O', 0.92, 1.4)
UFSG[17] = UNIFAC_subgroup('ACOH', 8, 'ACOH', 0.8952, 0.68)
UFSG[18] = UNIFAC_subgroup('CH3CO', 9, 'CH2CO', 1.6724, 1.488)
UFSG[19] = UNIFAC_subgroup('CH2CO', 9, 'CH2CO', 1.4457, 1.18)
UFSG[20] = UNIFAC_subgroup('CHO', 10, 'CHO', 0.998, 0.948)
UFSG[21] = UNIFAC_subgroup('CH3COO', 11, 'CCOO', 1.9031, 1.728)
UFSG[22] = UNIFAC_subgroup('CH2COO', 11, 'CCOO', 1.6764, 1.42)
UFSG[23] = UNIFAC_subgroup('HCOO', 12, 'HCOO', 1.242, 1.188)
UFSG[24] = UNIFAC_subgroup('CH3O', 13, 'CH2O', 1.145, 1.088)
UFSG[25] = UNIFAC_subgroup('CH2O', 13, 'CH2O', 0.9183, 0.78)
UFSG[26] = UNIFAC_subgroup('CHO', 13, 'CH2O', 0.6908, 0.468)
UFSG[27] = UNIFAC_subgroup('THF', 13, 'CH2O', 0.9183, 1.1)
UFSG[28] = UNIFAC_subgroup('CH3NH2', 14, 'CNH2', 1.5959, 1.544)
UFSG[29] = UNIFAC_subgroup('CH2NH2', 14, 'CNH2', 1.3692, 1.236)
UFSG[30] = UNIFAC_subgroup('CHNH2', 14, 'CNH2', 1.1417, 0.924)
UFSG[31] = UNIFAC_subgroup('CH3NH', 15, 'CNH', 1.4337, 1.244)
UFSG[32] = UNIFAC_subgroup('CH2NH', 15, 'CNH', 1.207, 0.936)
UFSG[33] = UNIFAC_subgroup('CHNH', 15, 'CNH', 0.9795, 0.624)
UFSG[34] = UNIFAC_subgroup('CH3N', 16, '(C)3N', 1.1865, 0.94)
UFSG[35] = UNIFAC_subgroup('CH2N', 16, '(C)3N', 0.9597, 0.632)
UFSG[36] = UNIFAC_subgroup('ACNH2', 17, 'ACNH2', 1.06, 0.816)
UFSG[37] = UNIFAC_subgroup('C5H5N', 18, 'PYRIDINE', 2.9993, 2.113)
UFSG[38] = UNIFAC_subgroup('C5H4N', 18, 'PYRIDINE', 2.8332, 1.833)
UFSG[39] = UNIFAC_subgroup('C5H3N', 18, 'PYRIDINE', 2.667, 1.553)
UFSG[40] = UNIFAC_subgroup('CH3CN', 19, 'CCN', 1.8701, 1.724)
UFSG[41] = UNIFAC_subgroup('CH2CN', 19, 'CCN', 1.6434, 1.416)
UFSG[42] = UNIFAC_subgroup('COOH', 20, 'COOH', 1.3013, 1.224)
UFSG[43] = UNIFAC_subgroup('HCOOH', 20, 'COOH', 1.528, 1.532)
UFSG[44] = UNIFAC_subgroup('CH2CL', 21, 'CCL', 1.4654, 1.264)
UFSG[45] = UNIFAC_subgroup('CHCL', 21, 'CCL', 1.238, 0.952)
UFSG[46] = UNIFAC_subgroup('CCL', 21, 'CCL', 1.0106, 0.724)
UFSG[47] = UNIFAC_subgroup('CH2CL2', 22, 'CCL2', 2.2564, 1.988)
UFSG[48] = UNIFAC_subgroup('CHCL2', 22, 'CCL2', 2.0606, 1.684)
UFSG[49] = UNIFAC_subgroup('CCL2', 22, 'CCL2', 1.8016, 1.448)
UFSG[50] = UNIFAC_subgroup('CHCL3', 23, 'CCL3', 2.87, 2.41)
UFSG[51] = UNIFAC_subgroup('CCL3', 23, 'CCL3', 2.6401, 2.184)
UFSG[52] = UNIFAC_subgroup('CCL4', 24, 'CCL4', 3.39, 2.91)
UFSG[53] = UNIFAC_subgroup('ACCL', 25, 'ACCL', 1.1562, 0.844)
UFSG[54] = UNIFAC_subgroup('CH3NO2', 26, 'CNO2', 2.0086, 1.868)
UFSG[55] = UNIFAC_subgroup('CH2NO2', 26, 'CNO2', 1.7818, 1.56)
UFSG[56] = UNIFAC_subgroup('CHNO2', 26, 'CNO2', 1.5544, 1.248)
UFSG[57] = UNIFAC_subgroup('ACNO2', 27, 'ACNO2', 1.4199, 1.104)
UFSG[58] = UNIFAC_subgroup('CS2', 28, 'CS2', 2.057, 1.65)
UFSG[59] = UNIFAC_subgroup('CH3SH', 29, 'CH3SH', 1.877, 1.676)
UFSG[60] = UNIFAC_subgroup('CH2SH', 29, 'CH3SH', 1.651, 1.368)
UFSG[61] = UNIFAC_subgroup('FURFURAL', 30, 'FURFURAL', 3.168, 2.484)
UFSG[62] = UNIFAC_subgroup('DOH', 31, 'DOH', 2.4088, 2.248)
UFSG[63] = UNIFAC_subgroup('I', 32, 'I', 1.264, 0.992)
UFSG[64] = UNIFAC_subgroup('BR', 33, 'BR', 0.9492, 0.832)
UFSG[65] = UNIFAC_subgroup('CH=-C', 34, 'C=-C', 1.292, 1.088)
UFSG[66] = UNIFAC_subgroup('C=-C', 34, 'C=-C', 1.0613, 0.784)
UFSG[67] = UNIFAC_subgroup('DMSO', 35, 'DMSO', 2.8266, 2.472)
UFSG[68] = UNIFAC_subgroup('ACRY', 36, 'ACRY', 2.3144, 2.052)
UFSG[69] = UNIFAC_subgroup('CL-(C=C)', 37, 'CLCC', 0.791, 0.724)
UFSG[70] = UNIFAC_subgroup('C=C', 2, 'C=C', 0.6605, 0.485)
UFSG[71] = UNIFAC_subgroup('ACF', 38, 'ACF', 0.6948, 0.524)
UFSG[72] = UNIFAC_subgroup('DMF', 39, 'DMF', 3.0856, 2.736)
UFSG[73] = UNIFAC_subgroup('HCON(..', 39, 'DMF', 2.6322, 2.12)
UFSG[74] = UNIFAC_subgroup('CF3', 40, 'CF2', 1.406, 1.38)
UFSG[75] = UNIFAC_subgroup('CF2', 40, 'CF2', 1.0105, 0.92)
UFSG[76] = UNIFAC_subgroup('CF', 40, 'CF2', 0.615, 0.46)
UFSG[77] = UNIFAC_subgroup('COO', 41, 'COO', 1.38, 1.2)
UFSG[78] = UNIFAC_subgroup('SIH3', 42, 'SIH2', 1.6035, 1.2632)
UFSG[79] = UNIFAC_subgroup('SIH2', 42, 'SIH2', 1.4443, 1.0063)
UFSG[80] = UNIFAC_subgroup('SIH', 42, 'SIH2', 1.2853, 0.7494)
UFSG[81] = UNIFAC_subgroup('SI', 42, 'SIH2', 1.047, 0.4099)
UFSG[82] = UNIFAC_subgroup('SIH2O', 43, 'SIO', 1.4838, 1.0621)
UFSG[83] = UNIFAC_subgroup('SIHO', 43, 'SIO', 1.303, 0.7639)
UFSG[84] = UNIFAC_subgroup('SIO', 43, 'SIO', 1.1044, 0.4657)
UFSG[85] = UNIFAC_subgroup('NMP', 44, 'NMP', 3.981, 3.2)
UFSG[86] = UNIFAC_subgroup('CCL3F', 45, 'CCLF', 3.0356, 2.644)
UFSG[87] = UNIFAC_subgroup('CCL2F', 45, 'CCLF', 2.2287, 1.916)
UFSG[88] = UNIFAC_subgroup('HCCL2F', 45, 'CCLF', 2.406, 2.116)
UFSG[89] = UNIFAC_subgroup('HCCLF', 45, 'CCLF', 1.6493, 1.416)
UFSG[90] = UNIFAC_subgroup('CCLF2', 45, 'CCLF', 1.8174, 1.648)
UFSG[91] = UNIFAC_subgroup('HCCLF2', 45, 'CCLF', 1.967, 1.828)
UFSG[92] = UNIFAC_subgroup('CCLF3', 45, 'CCLF', 2.1721, 2.1)
UFSG[93] = UNIFAC_subgroup('CCL2F2', 45, 'CCLF', 2.6243, 2.376)
UFSG[94] = UNIFAC_subgroup('AMH2', 46, 'CON(AM)', 1.4515, 1.248)
UFSG[95] = UNIFAC_subgroup('AMHCH3', 46, 'CON(AM)', 2.1905, 1.796)
UFSG[96] = UNIFAC_subgroup('AMHCH2', 46, 'CON(AM)', 1.9637, 1.488)
UFSG[97] = UNIFAC_subgroup('AM(CH3)2', 46, 'CON(AM)', 2.8589, 2.428)
UFSG[98] = UNIFAC_subgroup('AMCH3CH2', 46, 'CON(AM)', 2.6322, 2.12)
UFSG[99] = UNIFAC_subgroup('AM(CH2)2', 46, 'CON(AM)', 2.4054, 1.812)
UFSG[100] = UNIFAC_subgroup('C2H5O2', 47, 'OCCOH', 2.1226, 1.904)
UFSG[101] = UNIFAC_subgroup('C2H4O2', 47, 'OCCOH', 1.8952, 1.592)
UFSG[102] = UNIFAC_subgroup('CH3S', 48, 'CH2S', 1.613, 1.368)
UFSG[103] = UNIFAC_subgroup('CH2S', 48, 'CH2S', 1.3863, 1.06)
UFSG[104] = UNIFAC_subgroup('CHS', 48, 'CH2S', 1.1589, 0.748)
UFSG[105] = UNIFAC_subgroup('MORPH', 49, 'MORPH', 3.474, 2.796)
UFSG[106] = UNIFAC_subgroup('C4H4S', 50, 'THIOPHEN', 2.8569, 2.14)
UFSG[107] = UNIFAC_subgroup('C4H3S', 50, 'THIOPHEN', 2.6908, 1.86)
UFSG[108] = UNIFAC_subgroup('C4H2S', 50, 'THIOPHEN', 2.5247, 1.58)
UFSG[109] = UNIFAC_subgroup('NCO', 51, 'NCO', 1.0567, 0.732)
UFSG[118] = UNIFAC_subgroup('(CH2)2SU', 55, 'SULFONES', 2.6869, 2.12)
UFSG[119] = UNIFAC_subgroup('CH2CHSU', 55, 'SULFONES', 2.4595, 1.808)
UFSG[178] = UNIFAC_subgroup('IMIDAZOL', 84, 'IMIDAZOL', 2.026, 0.868)
UFSG[179] = UNIFAC_subgroup('BTI', 85, 'BTI', 5.774, 4.932)


# http://www.ddbst.com/PublishedParametersUNIFACDO.html#ListOfSubGroupsAndTheirGroupSurfacesAndVolumes
#  subgroup = (subgroup, #maingroup, maingroup, R, Q)
DOUFSG = {}
DOUFSG[1] = UNIFAC_subgroup('CH3', 1, 'CH2', 0.6325, 1.0608)
DOUFSG[2] = UNIFAC_subgroup('CH2', 1, 'CH2', 0.6325, 0.7081)
DOUFSG[3] = UNIFAC_subgroup('CH', 1, 'CH2', 0.6325, 0.3554)
DOUFSG[4] = UNIFAC_subgroup('C', 1, 'CH2', 0.6325, 0)
DOUFSG[5] = UNIFAC_subgroup('CH2=CH', 2, 'C=C', 1.2832, 1.6016)
DOUFSG[6] = UNIFAC_subgroup('CH=CH', 2, 'C=C', 1.2832, 1.2489)
DOUFSG[7] = UNIFAC_subgroup('CH2=C', 2, 'C=C', 1.2832, 1.2489)
DOUFSG[8] = UNIFAC_subgroup('CH=C', 2, 'C=C', 1.2832, 0.8962)
DOUFSG[9] = UNIFAC_subgroup('ACH', 3, 'ACH', 0.3763, 0.4321)
DOUFSG[10] = UNIFAC_subgroup('AC', 3, 'ACH', 0.3763, 0.2113)
DOUFSG[11] = UNIFAC_subgroup('ACCH3', 4, 'ACCH2', 0.91, 0.949)
DOUFSG[12] = UNIFAC_subgroup('ACCH2', 4, 'ACCH2', 0.91, 0.7962)
DOUFSG[13] = UNIFAC_subgroup('ACCH', 4, 'ACCH2', 0.91, 0.3769)
DOUFSG[14] = UNIFAC_subgroup('OH(P)', 5, 'OH', 1.2302, 0.8927)
DOUFSG[15] = UNIFAC_subgroup('CH3OH', 6, 'CH3OH', 0.8585, 0.9938)
DOUFSG[16] = UNIFAC_subgroup('H2O', 7, 'H2O', 1.7334, 2.4561)
DOUFSG[17] = UNIFAC_subgroup('ACOH', 8, 'ACOH', 1.08, 0.975)
DOUFSG[18] = UNIFAC_subgroup('CH3CO', 9, 'CH2CO', 1.7048, 1.67)
DOUFSG[19] = UNIFAC_subgroup('CH2CO', 9, 'CH2CO', 1.7048, 1.5542)
DOUFSG[20] = UNIFAC_subgroup('CHO', 10, 'CHO', 0.7173, 0.771)
DOUFSG[21] = UNIFAC_subgroup('CH3COO', 11, 'CCOO', 1.27, 1.6286)
DOUFSG[22] = UNIFAC_subgroup('CH2COO', 11, 'CCOO', 1.27, 1.4228)
DOUFSG[23] = UNIFAC_subgroup('HCOO', 12, 'HCOO', 1.9, 1.8)
DOUFSG[24] = UNIFAC_subgroup('CH3O', 13, 'CH2O', 1.1434, 1.6022)
DOUFSG[25] = UNIFAC_subgroup('CH2O', 13, 'CH2O', 1.1434, 1.2495)
DOUFSG[26] = UNIFAC_subgroup('CHO', 13, 'CH2O', 1.1434, 0.8968)
DOUFSG[27] = UNIFAC_subgroup('THF', 43, 'CY-CH2O', 1.7023, 1.8784)
DOUFSG[28] = UNIFAC_subgroup('CH3NH2', 14, 'CH2NH2', 1.6607, 1.6904)
DOUFSG[29] = UNIFAC_subgroup('CH2NH2', 14, 'CH2NH2', 1.6607, 1.3377)
DOUFSG[30] = UNIFAC_subgroup('CHNH2', 14, 'CH2NH2', 1.6607, 0.985)
DOUFSG[31] = UNIFAC_subgroup('CH3NH', 15, 'CH2NH', 1.368, 1.4332)
DOUFSG[32] = UNIFAC_subgroup('CH2NH', 15, 'CH2NH', 1.368, 1.0805)
DOUFSG[33] = UNIFAC_subgroup('CHNH', 15, 'CH2NH', 1.368, 0.7278)
DOUFSG[34] = UNIFAC_subgroup('CH3N', 16, '(C)3N', 1.0746, 1.176)
DOUFSG[35] = UNIFAC_subgroup('CH2N', 16, '(C)3N', 1.0746, 0.824)
DOUFSG[36] = UNIFAC_subgroup('ACNH2', 17, 'ACNH2', 1.1849, 0.8067)
DOUFSG[37] = UNIFAC_subgroup('AC2H2N', 18, 'PYRIDINE', 1.4578, 0.9022)
DOUFSG[38] = UNIFAC_subgroup('AC2HN', 18, 'PYRIDINE', 1.2393, 0.633)
DOUFSG[39] = UNIFAC_subgroup('AC2N', 18, 'PYRIDINE', 1.0731, 0.353)
DOUFSG[40] = UNIFAC_subgroup('CH3CN', 19, 'CH2CN', 1.5575, 1.5193)
DOUFSG[41] = UNIFAC_subgroup('CH2CN', 19, 'CH2CN', 1.5575, 1.1666)
DOUFSG[42] = UNIFAC_subgroup('COOH', 20, 'COOH', 0.8, 0.9215)
DOUFSG[43] = UNIFAC_subgroup('HCOOH', 44, 'HCOOH', 0.8, 1.2742)
DOUFSG[44] = UNIFAC_subgroup('CH2CL', 21, 'CCL', 0.9919, 1.3654)
DOUFSG[45] = UNIFAC_subgroup('CHCL', 21, 'CCL', 0.9919, 1.0127)
DOUFSG[46] = UNIFAC_subgroup('CCL', 21, 'CCL', 0.9919, 0.66)
DOUFSG[47] = UNIFAC_subgroup('CH2CL2', 22, 'CCL2', 1.8, 2.5)
DOUFSG[48] = UNIFAC_subgroup('CHCL2', 22, 'CCL2', 1.8, 2.1473)
DOUFSG[49] = UNIFAC_subgroup('CCL2', 22, 'CCL2', 1.8, 1.7946)
DOUFSG[50] = UNIFAC_subgroup('CHCL3', 45, 'CHCL3', 2.45, 2.8912)
DOUFSG[51] = UNIFAC_subgroup('CCL3', 23, 'CCL3', 2.65, 2.3778)
DOUFSG[52] = UNIFAC_subgroup('CCL4', 24, 'CCL4', 2.618, 3.1836)
DOUFSG[53] = UNIFAC_subgroup('ACCL', 25, 'ACCL', 0.5365, 0.3177)
DOUFSG[54] = UNIFAC_subgroup('CH3NO2', 26, 'CNO2', 2.644, 2.5)
DOUFSG[55] = UNIFAC_subgroup('CH2NO2', 26, 'CNO2', 2.5, 2.304)
DOUFSG[56] = UNIFAC_subgroup('CHNO2', 26, 'CNO2', 2.887, 2.241)
DOUFSG[57] = UNIFAC_subgroup('ACNO2', 27, 'ACNO2', 0.4656, 0.3589)
DOUFSG[58] = UNIFAC_subgroup('CS2', 28, 'CS2', 1.24, 1.068)
DOUFSG[59] = UNIFAC_subgroup('CH3SH', 29, 'CH3SH', 1.289, 1.762)
DOUFSG[60] = UNIFAC_subgroup('CH2SH', 29, 'CH3SH', 1.535, 1.316)
DOUFSG[61] = UNIFAC_subgroup('FURFURAL', 30, 'FURFURAL', 1.299, 1.289)
DOUFSG[62] = UNIFAC_subgroup('DOH', 31, 'DOH', 2.088, 2.4)
DOUFSG[63] = UNIFAC_subgroup('I', 32, 'I', 1.076, 0.9169)
DOUFSG[64] = UNIFAC_subgroup('BR', 33, 'BR', 1.209, 1.4)
DOUFSG[65] = UNIFAC_subgroup('CH=-C', 34, 'C=-C', 0.9214, 1.3)
DOUFSG[66] = UNIFAC_subgroup('C=-C', 34, 'C=-C', 1.303, 1.132)
DOUFSG[67] = UNIFAC_subgroup('DMSO', 35, 'DMSO', 3.6, 2.692)
DOUFSG[68] = UNIFAC_subgroup('ACRY', 36, 'ACRY', 1, 0.92)
DOUFSG[69] = UNIFAC_subgroup('CL-(C=C)', 37, 'CLCC', 0.5229, 0.7391)
DOUFSG[70] = UNIFAC_subgroup('C=C', 2, 'C=C', 1.2832, 0.4582)
DOUFSG[71] = UNIFAC_subgroup('ACF', 38, 'ACF', 0.8814, 0.7269)
DOUFSG[72] = UNIFAC_subgroup('DMF', 39, 'DMF', 2, 2.093)
DOUFSG[73] = UNIFAC_subgroup('HCON(..', 39, 'DMF', 2.381, 1.522)
DOUFSG[74] = UNIFAC_subgroup('CF3', 40, 'CF2', 1.284, 1.266)
DOUFSG[75] = UNIFAC_subgroup('CF2', 40, 'CF2', 1.284, 1.098)
DOUFSG[76] = UNIFAC_subgroup('CF', 40, 'CF2', 0.8215, 0.5135)
DOUFSG[77] = UNIFAC_subgroup('COO', 41, 'COO', 1.6, 0.9)
DOUFSG[78] = UNIFAC_subgroup('CY-CH2', 42, 'CY-CH2', 0.7136, 0.8635)
DOUFSG[79] = UNIFAC_subgroup('CY-CH', 42, 'CY-CH2', 0.3479, 0.1071)
DOUFSG[80] = UNIFAC_subgroup('CY-C', 42, 'CY-CH2', 0.347, 0)
DOUFSG[81] = UNIFAC_subgroup('OH(S)', 5, 'OH', 1.063, 0.8663)
DOUFSG[82] = UNIFAC_subgroup('OH(T)', 5, 'OH', 0.6895, 0.8345)
DOUFSG[83] = UNIFAC_subgroup('CY-CH2O', 43, 'CY-CH2O', 1.4046, 1.4)
DOUFSG[84] = UNIFAC_subgroup('TRIOXAN', 43, 'CY-CH2O', 1.0413, 1.0116)
DOUFSG[85] = UNIFAC_subgroup('CNH2', 14, 'CH2NH2', 1.6607, 0.985)
DOUFSG[86] = UNIFAC_subgroup('NMP', 46, 'CY-CONC', 3.981, 3.2)
DOUFSG[87] = UNIFAC_subgroup('NEP', 46, 'CY-CONC', 3.7543, 2.892)
DOUFSG[88] = UNIFAC_subgroup('NIPP', 46, 'CY-CONC', 3.5268, 2.58)
DOUFSG[89] = UNIFAC_subgroup('NTBP', 46, 'CY-CONC', 3.2994, 2.352)
DOUFSG[91] = UNIFAC_subgroup('CONH2', 47, 'CONR', 1.4515, 1.248)
DOUFSG[92] = UNIFAC_subgroup('CONHCH3', 47, 'CONR', 1.5, 1.08)
DOUFSG[100] = UNIFAC_subgroup('CONHCH2', 47, 'CONR', 1.5, 1.08)
DOUFSG[101] = UNIFAC_subgroup('AM(CH3)2', 48, 'CONR2', 2.4748, 1.9643)
DOUFSG[102] = UNIFAC_subgroup('AMCH3CH2', 48, 'CONR2', 2.2739, 1.5754)
DOUFSG[103] = UNIFAC_subgroup('AM(CH2)2', 48, 'CONR2', 2.0767, 1.1866)
DOUFSG[104] = UNIFAC_subgroup('AC2H2S', 52, 'ACS', 1.7943, 1.34)
DOUFSG[105] = UNIFAC_subgroup('AC2HS', 52, 'ACS', 1.6282, 1.06)
DOUFSG[106] = UNIFAC_subgroup('AC2S', 52, 'ACS', 1.4621, 0.78)
DOUFSG[107] = UNIFAC_subgroup('H2COCH', 53, 'EPOXIDES', 1.3601, 1.8031)
DOUFSG[108] = UNIFAC_subgroup('COCH', 53, 'EPOXIDES', 0.683, 0.3418)
DOUFSG[109] = UNIFAC_subgroup('HCOCH', 53, 'EPOXIDES', 0.9104, 0.6538)
DOUFSG[110] = UNIFAC_subgroup('(CH2)2SU', 56, 'SULFONE', 2.687, 2.12)
DOUFSG[111] = UNIFAC_subgroup('CH2SUCH', 56, 'SULFONE', 2.46, 1.808)
DOUFSG[112] = UNIFAC_subgroup('(CH3)2CB', 55, 'CARBONAT', 2.42, 2.4976)
DOUFSG[113] = UNIFAC_subgroup('(CH2)2CB', 55, 'CARBONAT', 2.42, 2.0018)
DOUFSG[114] = UNIFAC_subgroup('CH2CH3CB', 55, 'CARBONAT', 2.42, 2.2497)
DOUFSG[119] = UNIFAC_subgroup('H2COCH2', 53, 'EPOXIDES', 1.063, 1.123)
DOUFSG[153] = UNIFAC_subgroup('H2COC', 53, 'EPOXIDES', 0.9104, 0.6538)
DOUFSG[178] = UNIFAC_subgroup('C3H2N2+', 84, 'IMIDAZOL', 1.7989, 0.64)
DOUFSG[179] = UNIFAC_subgroup('BTI-', 85, 'BTI', 5.8504, 5.7513)
DOUFSG[184] = UNIFAC_subgroup('C3H3N2+', 84, 'IMIDAZOL', 2.411, 2.409)
DOUFSG[189] = UNIFAC_subgroup('C4H8N+', 87, 'PYRROL', 2.7986, 2.7744)
DOUFSG[195] = UNIFAC_subgroup('BF4-', 89, 'BF4', 4.62, 1.1707)
DOUFSG[196] = UNIFAC_subgroup('C5H5N+', 90, 'PYRIDIN', 2.4878, 2.474)
DOUFSG[197] = UNIFAC_subgroup('OTF-', 91, 'OTF', 3.3854, 2.009)

#  subgroup = (group, (subgroup ids))
# http://www.ddbst.com/PublishedParametersUNIFACDO.html#ListOfMainGroups
DOUFMG = {}
DOUFMG[1] = ('CH2', (1, 2, 3, 4))
DOUFMG[2] = ('C=C', (5, 6, 7, 8, 70))
DOUFMG[3] = ('ACH', (9, 10))
DOUFMG[4] = ('ACCH2', (11, 12, 13))
DOUFMG[5] = ('OH', (14, 81, 82))
DOUFMG[6] = ('CH3OH', (15))
DOUFMG[7] = ('H2O', (16))
DOUFMG[8] = ('ACOH', (17))
DOUFMG[9] = ('CH2CO', (18, 19))
DOUFMG[10] = ('CHO', (20))
DOUFMG[11] = ('CCOO', (21, 22))
DOUFMG[12] = ('HCOO', (23))
DOUFMG[13] = ('CH2O', (24, 25, 26))
DOUFMG[14] = ('CH2NH2', (28, 29, 30, 85))
DOUFMG[15] = ('CH2NH', (31, 32, 33))
DOUFMG[16] = ('(C)3N', (34, 35))
DOUFMG[17] = ('ACNH2', (36))
DOUFMG[18] = ('PYRIDINE', (37, 38, 39))
DOUFMG[19] = ('CH2CN', (40, 41))
DOUFMG[20] = ('COOH', (42))
DOUFMG[21] = ('CCL', (44, 45, 46))
DOUFMG[22] = ('CCL2', (47, 48, 49))
DOUFMG[23] = ('CCL3', (51))
DOUFMG[24] = ('CCL4', (52))
DOUFMG[25] = ('ACCL', (53))
DOUFMG[26] = ('CNO2', (54, 55, 56))
DOUFMG[27] = ('ACNO2', (57))
DOUFMG[28] = ('CS2', (58))
DOUFMG[29] = ('CH3SH', (59, 60))
DOUFMG[30] = ('FURFURAL', (61))
DOUFMG[31] = ('DOH', (62))
DOUFMG[32] = ('I', (63))
DOUFMG[33] = ('BR', (64))
DOUFMG[34] = ('C=-C', (65, 66))
DOUFMG[35] = ('DMSO', (67))
DOUFMG[36] = ('ACRY', (68))
DOUFMG[37] = ('CLCC', (69))
DOUFMG[38] = ('ACF', (71))
DOUFMG[39] = ('DMF', (72, 73))
DOUFMG[40] = ('CF2', (74, 75, 76))
DOUFMG[41] = ('COO', (77))
DOUFMG[42] = ('CY-CH2', (78, 79, 80))
DOUFMG[43] = ('CY-CH2O', (27, 83, 84))
DOUFMG[44] = ('HCOOH', (43))
DOUFMG[45] = ('CHCL3', (50))
DOUFMG[46] = ('CY-CONC', (86, 87, 88, 89))
DOUFMG[47] = ('CONR', (91, 92, 100))
DOUFMG[48] = ('CONR2', (101, 102, 103))
DOUFMG[52] = ('ACS', (104, 105, 106))
DOUFMG[53] = ('EPOXIDES', (107, 108, 109, 119, 153))
DOUFMG[55] = ('CARBONAT', (112, 113, 114))
DOUFMG[56] = ('SULFONE', (110, 111))
DOUFMG[84] = ('IMIDAZOL', (178, 184))
DOUFMG[85] = ('BTI', (179))
DOUFMG[87] = ('PYRROL', (189))
DOUFMG[89] = ('BF4', (195))
DOUFMG[90] = ('PYRIDIN', (196))
DOUFMG[91] = ('OTF', (197))


def UNIFAC(T, xs, chemgroups, cached=None, UFSG=UFSG, UFIP=UFIP, modified=False):
    cmps = range(len(xs))

    # Obtain r and q values using the subgroup values
    if not cached:
        rs = []
        qs = []
        for groups in chemgroups:
            ri = 0.
            qi = 0.
            for group, count in groups.items():
                ri += UFSG[group].R*count
                qi += UFSG[group].Q*count
            rs.append(ri)
            qs.append(qi)
        

        group_counts = {}
        for groups in chemgroups:
            for group, count in groups.items():
                if group in group_counts:
                    group_counts[group] += count
                else:
                    group_counts[group] = count
    else:
        rs, qs, group_counts = cached

    # Sum the denominator for calculating Xs
    group_sum = sum(count*xs[i] for i in cmps for count in chemgroups[i].values())

    # Caclulate each numerator for calculating Xs
    group_count_xs = {}
    for group in group_counts:
        tot_numerator = sum(chemgroups[i][group]*xs[i] for i in cmps if group in chemgroups[i])        
        group_count_xs[group] = tot_numerator/group_sum

    rsxs = sum([rs[i]*xs[i] for i in cmps])
    Vis = [rs[i]/rsxs for i in cmps]
    qsxs = sum([qs[i]*xs[i] for i in cmps])
    Fis = [qs[i]/qsxs for i in cmps]

    if modified:
        rsxs2 = sum([rs[i]**0.75*xs[i] for i in cmps])
        Vis2 = [rs[i]**0.75/rsxs2 for i in cmps]
        loggammacs = [1. - Vis2[i] + log(Vis2[i]) - 5.*qs[i]*(1. - Vis[i]/Fis[i]
                      + log(Vis[i]/Fis[i])) for i in cmps]
    else:
        loggammacs = [1. - Vis[i] + log(Vis[i]) - 5.*qs[i]*(1. - Vis[i]/Fis[i]
                      + log(Vis[i]/Fis[i])) for i in cmps]

    Q_sum_term = sum([UFSG[group].Q*group_count_xs[group] for group in group_counts])
    area_fractions = {group: UFSG[group].Q*group_count_xs[group]/Q_sum_term
                      for group in group_counts.keys()}

    UNIFAC_psis = {k: {m:(UNIFAC_psi(T, m, k, UFSG, UFIP, modified=modified))
                   for m in group_counts} for k in group_counts}

    loggamma_groups = {}
    for k in group_counts:
        sum1, sum2 = 0., 0.
        for m in group_counts:
            sum1 += area_fractions[m]*UNIFAC_psis[k][m]
            sum3 = sum(area_fractions[n]*UNIFAC_psis[m][n] for n in group_counts)
            sum2 -= area_fractions[m]*UNIFAC_psis[m][k]/sum3
        loggamma_groups[k] = UFSG[k].Q*(1. - log(sum1) + sum2)


    loggammars = []
    for groups in chemgroups:
        chem_loggamma_groups = {}
        chem_group_sum = sum(groups.values())
        chem_group_count_xs = {group: count/chem_group_sum for group, count in groups.items()}
                               
        Q_sum_term = sum([UFSG[group].Q*chem_group_count_xs[group] for group in groups])
        chem_area_fractions = {group: UFSG[group].Q*chem_group_count_xs[group]/Q_sum_term
                               for group in groups.keys()}
        for k in groups:
            sum1, sum2 = 0., 0.
            for m in groups:
                sum1 += chem_area_fractions[m]*UNIFAC_psis[k][m]
                sum3 = sum(chem_area_fractions[n]*UNIFAC_psis[m][n] for n in groups)
                sum2 -= chem_area_fractions[m]*UNIFAC_psis[m][k]/sum3

            chem_loggamma_groups[k] = UFSG[k].Q*(1. - log(sum1) + sum2)

        tot = sum([count*(loggamma_groups[group] - chem_loggamma_groups[group])
                   for group, count in groups.items()])
        loggammars.append(tot)

    return [exp(loggammacs[i]+loggammars[i]) for i in cmps]



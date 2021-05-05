# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019, 2020, 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains functions and classes related to the UNIFAC and its many
variants. The bulk of the code relates to calculating derivativies, or
is tables of data.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_
or contact the author at Caleb.Andrew.Bell@gmail.com.

.. contents:: :local:

Main Model (Object-Oriented)
----------------------------
.. autoclass:: UNIFAC
    :members:

Main Model (Functional)
-----------------------
.. autofunction:: UNIFAC_gammas
.. autofunction:: UNIFAC_psi

Misc Functions
--------------
.. autofunction:: UNIFAC_RQ
.. autofunction:: Van_der_Waals_volume
.. autofunction:: Van_der_Waals_area
.. autofunction:: chemgroups_to_matrix
.. autofunction:: load_group_assignments_DDBST

Data for Original UNIFAC
------------------------
.. autodata:: UFSG
.. autodata:: UFMG
.. py:data:: UFIP

    Interaction parameters for the original unifac model.

    :type: dict[int: dict[int: float]]

Data for Dortmund UNIFAC
------------------------
.. autodata:: DOUFSG
.. autodata:: DOUFMG
.. py:data:: DOUFIP2016

    Interaction parameters for the Dornmund unifac model.

    :type: dict[int: dict[int: tuple(float, 3)]]


Data for NIST UNIFAC (2015)
---------------------------
.. autodata:: NISTUFSG
.. autodata:: NISTUFMG
.. py:data:: NISTUFIP

    Interaction parameters for the NIST (2015) unifac model.

    :type: dict[int: dict[int: tuple(float, 3)]]

Data for NIST KT UNIFAC (2011)
------------------------------
.. autodata:: NISTKTUFSG
.. autodata:: NISTKTUFMG
.. py:data:: NISTKTUFIP

    Interaction parameters for the NIST KT UNIFAC (2011) model.

    :type: dict[int: dict[int: tuple(float, 3)]]

Data for UNIFAC LLE
-------------------
.. autodata:: LLEUFSG
.. autodata:: LLEMG
.. py:data:: LLEUFIP

    Interaction parameters for the LLE unifac model.

    :type: dict[int: dict[int: float]]

Data for Lyngby UNIFAC
----------------------
.. autodata:: LUFSG
.. autodata:: LUFMG
.. py:data:: LUFIP

    Interaction parameters for the Lyngby UNIFAC model.

    :type: dict[int: dict[int: tuple(float, 3)]]

Data for PSRK UNIFAC
--------------------
.. autodata:: PSRKSG
.. autodata:: PSRKMG
.. py:data:: PSRKIP

    Interaction parameters for the PSRKIP UNIFAC model.

    :type: dict[int: dict[int: tuple(float, 3)]]

Data for VTPR UNIFAC
--------------------
.. autodata:: VTPRSG
.. autodata:: VTPRMG
.. py:data:: VTPRIP

    Interaction parameters for the VTPRIP UNIFAC model.

    :type: dict[int: dict[int: tuple(float, 3)]]
'''

from __future__ import division

__all__ = ['UNIFAC_gammas','UNIFAC', 'UNIFAC_psi', 'DOUFMG', 'DOUFSG', 'UFSG', 'UFMG',

           'DDBST_UNIFAC_assignments',
           'DDBST_MODIFIED_UNIFAC_assignments', 'DDBST_PSRK_assignments',

           'UNIFAC_RQ', 'Van_der_Waals_volume', 'Van_der_Waals_area',
           'load_group_assignments_DDBST',

           'PSRKSG', 'LLEUFSG', 'LLEMG',
            'LUFSG', 'NISTUFSG', 'NISTUFMG',
           'VTPRSG', 'VTPRMG', 'NISTKTUFSG', 'NISTKTUFMG',
           'LUFMG', 'PSRKMG',
           'unifac_gammas_at_T']
import os
from fluids.constants import R
from fluids.numerics import numpy as np
from chemicals.utils import log, exp, dxs_to_dns, can_load_data, PY37
from thermo.activity import GibbsExcess

try:
    array, zeros, npexp, array_equal = np.array, np.zeros, np.exp, np.array_equal
except (ImportError, AttributeError):
    pass


class UNIFAC_subgroup(object):
    __slots__ = ['group', 'main_group_id', 'main_group', 'R', 'Q', 'smarts']

    def __repr__(self):   # pragma: no cover
        return '<%s>' %self.group

    def __init__(self, group, main_group_id, main_group, R, Q, smarts=None):
        self.group = group
        self.main_group_id = main_group_id
        self.main_group = main_group
        self.R = R
        self.Q = Q
        self.smarts = smarts



# http://www.ddbst.com/published-parameters-unifac.html#ListOfMainGroups
#UFMG[No.] = ('Maingroup Name', subgroups)
UFMG = {}
UFMG[1] = ('CH2', [1, 2, 3, 4])
UFMG[2] = ('C=C', [5, 6, 7, 8, 70])
UFMG[3] = ('ACH', [9, 10])
UFMG[4] = ('ACCH2', [11, 12, 13])
UFMG[5] = ('OH', [14])
UFMG[6] = ('CH3OH', [15])
UFMG[7] = ('H2O', [16])
UFMG[8] = ('ACOH', [17])
UFMG[9] = ('CH2CO', [18, 19])
UFMG[10] = ('CHO', [20])
UFMG[11] = ('CCOO', [21, 22])
UFMG[12] = ('HCOO', [23])
UFMG[13] = ('CH2O', [24, 25, 26, 27])
UFMG[14] = ('CNH2', [28, 29, 30])
UFMG[15] = ('CNH', [31, 32, 33])
UFMG[16] = ('(C)3N', [34, 35])
UFMG[17] = ('ACNH2', [36])
UFMG[18] = ('PYRIDINE', [37, 38, 39])
UFMG[19] = ('CCN', [40, 41])
UFMG[20] = ('COOH', [42, 43])
UFMG[21] = ('CCL', [44, 45, 46])
UFMG[22] = ('CCL2', [47, 48, 49])
UFMG[23] = ('CCL3', [50, 51])
UFMG[24] = ('CCL4', [52])
UFMG[25] = ('ACCL', [53])
UFMG[26] = ('CNO2', [54, 55, 56])
UFMG[27] = ('ACNO2', [57])
UFMG[28] = ('CS2', [58])
UFMG[29] = ('CH3SH', [59, 60])
UFMG[30] = ('FURFURAL', [61])
UFMG[31] = ('DOH', [62])
UFMG[32] = ('I', [63])
UFMG[33] = ('BR', [64])
UFMG[34] = ('C=-C', [65, 66])
UFMG[35] = ('DMSO', [67])
UFMG[36] = ('ACRY', [68])
UFMG[37] = ('CLCC', [69])
UFMG[38] = ('ACF', [71])
UFMG[39] = ('DMF', [72, 73])
UFMG[40] = ('CF2', [74, 75, 76])
UFMG[41] = ('COO', [77])
UFMG[42] = ('SIH2', [78, 79, 80, 81])
UFMG[43] = ('SIO', [82, 83, 84])
UFMG[44] = ('NMP', [85])
UFMG[45] = ('CCLF', [86, 87, 88, 89, 90, 91, 92, 93])
UFMG[46] = ('CON(AM)', [94, 95, 96, 97, 98, 99])
UFMG[47] = ('OCCOH', [100, 101])
UFMG[48] = ('CH2S', [102, 103, 104])
UFMG[49] = ('MORPH', [105])
UFMG[50] = ('THIOPHEN', [106, 107, 108])
UFMG[51] = ('NCO', [109])
UFMG[55] = ('SULFONES', [118, 119])
UFMG[84] = ('IMIDAZOL', [178])
UFMG[85] = ('BTI', [179])


UFSG = {}
# UFSG[subgroup ID] = (subgroup formula, main group ID, subgroup R, subgroup Q)
# http://www.ddbst.com/published-parameters-unifac.html
UFSG[1] = UNIFAC_subgroup('CH3', 1, 'CH2', 0.9011, 0.848, smarts='[CX4;H3]')
UFSG[2] = UNIFAC_subgroup('CH2', 1, 'CH2', 0.6744, 0.54, smarts='[CX4;H2]')
UFSG[3] = UNIFAC_subgroup('CH', 1, 'CH2', 0.4469, 0.228, smarts='[CX4;H1]')
UFSG[4] = UNIFAC_subgroup('C', 1, 'CH2', 0.2195, 0, smarts='[CX4;H0]')
UFSG[5] = UNIFAC_subgroup('CH2=CH', 2, 'C=C', 1.3454, 1.176, smarts='[CX3;H2]=[CX3;H1]')
UFSG[6] = UNIFAC_subgroup('CH=CH', 2, 'C=C', 1.1167, 0.867, smarts='[CX3;H1]=[CX3;H1]') # Could restrict the next connection from being  H
UFSG[7] = UNIFAC_subgroup('CH2=C', 2, 'C=C', 1.1173, 0.988, smarts='[CX3;H2]=[CX3;H0]')
UFSG[8] = UNIFAC_subgroup('CH=C', 2, 'C=C', 0.8886, 0.676, smarts='[CX3;H1]=[CX3;H0]')
UFSG[9] = UNIFAC_subgroup('ACH', 3, 'ACH', 0.5313, 0.4, smarts='[cX3;H1]')
UFSG[10] = UNIFAC_subgroup('AC', 3, 'ACH', 0.3652, 0.12, smarts='[cX3;H0]')
UFSG[11] = UNIFAC_subgroup('ACCH3', 4, 'ACCH2', 1.2663, 0.968, smarts='[cX3;H0][CX4;H3]')
UFSG[12] = UNIFAC_subgroup('ACCH2', 4, 'ACCH2', 1.0396, 0.66, smarts='[cX3;H0][CX4;H2]')
UFSG[13] = UNIFAC_subgroup('ACCH', 4, 'ACCH2', 0.8121, 0.348, smarts='[cX3;H0][CX4;H1]')
UFSG[14] = UNIFAC_subgroup('OH', 5, 'OH', 1, 1.2, smarts='[OX2;H1]')
UFSG[15] = UNIFAC_subgroup('CH3OH', 6, 'CH3OH', 1.4311, 1.432, smarts='[CX4;H3][OX2;H1]') # One extra radical; otherwise perfect, six matches
UFSG[16] = UNIFAC_subgroup('H2O', 7, 'H2O', 0.92, 1.4, smarts='[OH2]')
UFSG[17] = UNIFAC_subgroup('ACOH', 8, 'ACOH', 0.8952, 0.68, smarts='[cX3;H0;R][OX2;H1]') # pretty good 5 extra
UFSG[18] = UNIFAC_subgroup('CH3CO', 9, 'CH2CO', 1.6724, 1.488, smarts='[CX4;H3][CX3](=O)')
UFSG[19] = UNIFAC_subgroup('CH2CO', 9, 'CH2CO', 1.4457, 1.18, smarts='[CX4;H2;!$([CX4,CX3;H0,H1][CX3](=O)[CX4;H2][CX3](=O)[CX4,CX3;H0,H1])][CX3;!$([CX4,CX3;H0,H1][CX3](=O)[CX4;H2][CX3](=O)[CX4,CX3;H0,H1])](=O)[#6;!$([CX4;H3]);!$([CX4,CX3;H0,H1][CX3](=O)[CX4;H2][CX3](=O)[CX4,CX3;H0,H1])]') # '[CX4;H2][CX3](=O)'
UFSG[20] = UNIFAC_subgroup('CHO', 10, 'CHO', 0.998, 0.948, smarts='[CX3H1](=O)')
UFSG[21] = UNIFAC_subgroup('CH3COO', 11, 'CCOO', 1.9031, 1.728, smarts='C(=O)[OH0]') # can't distinguish between 21, 22
UFSG[22] = UNIFAC_subgroup('CH2COO', 11, 'CCOO', 1.6764, 1.42, smarts='[CX4;H2][CX3](=[OX1])[OX2]') # Custom, lots of extra matches
UFSG[23] = UNIFAC_subgroup('HCOO', 12, 'HCOO', 1.242, 1.188, smarts='[CX3;H1](=[OX1])[OX2][#6;!$([CX3]=[#8])]')
UFSG[24] = UNIFAC_subgroup('CH3O', 13, 'CH2O', 1.145, 1.088, smarts='[CH3][O]')
UFSG[25] = UNIFAC_subgroup('CH2O', 13, 'CH2O', 0.9183, 0.78, smarts='[CH2][O]')
UFSG[26] = UNIFAC_subgroup('CHO', 13, 'CH2O', 0.6908, 0.468, smarts='[C;H1][O]')
UFSG[27] = UNIFAC_subgroup('THF', 13, 'CH2O', 0.9183, 1.1, smarts='[CX4,CX3;H2,H1;R][OX2;R]') # CX3, H1 needed to allow 290-67-5 and 255-37-8 but adds a lot of false positives;
UFSG[28] = UNIFAC_subgroup('CH3NH2', 14, 'CNH2', 1.5959, 1.544, smarts='[CX4;H3][NX3;H2]') # Perfect
UFSG[29] = UNIFAC_subgroup('CH2NH2', 14, 'CNH2', 1.3692, 1.236, smarts='[CX4;H2][NX3;H2]')
UFSG[30] = UNIFAC_subgroup('CHNH2', 14, 'CNH2', 1.1417, 0.924, smarts='[CX4;H1][NX3;H2]')
UFSG[31] = UNIFAC_subgroup('CH3NH', 15, 'CNH', 1.4337, 1.244, smarts='[CX4;H3][NX3;H1]')
UFSG[32] = UNIFAC_subgroup('CH2NH', 15, 'CNH', 1.207, 0.936, smarts='[CX4;H2][NX3;H1]')
UFSG[33] = UNIFAC_subgroup('CHNH', 15, 'CNH', 0.9795, 0.624, smarts='[CX4;H1][NX3;H1]')
UFSG[34] = UNIFAC_subgroup('CH3N', 16, '(C)3N', 1.1865, 0.94, smarts='[CX4;H3][NX3;H0]')
UFSG[35] = UNIFAC_subgroup('CH2N', 16, '(C)3N', 0.9597, 0.632, smarts='[CX4;H2][NX3;H0]')
UFSG[36] = UNIFAC_subgroup('ACNH2', 17, 'ACNH2', 1.06, 0.816, smarts='[c][NX3;H2]')
UFSG[37] = UNIFAC_subgroup('C5H5N', 18, 'PYRIDINE', 2.9993, 2.113, smarts='[cX3;H1][cX3;H1][cX3;H1][cX3;H1][nX2;H0][cX3;H1]') # There is another match from ddbst 3,4-Didehydropyridine  but it is C5H3N and so wrong; only one real hit
UFSG[38] = UNIFAC_subgroup('C5H4N', 18, 'PYRIDINE', 2.8332, 1.833, smarts='[$([nX2;H0]1[cX3;H0][cX3;H1][cX3;H1][cX3;H1][cX3;H1]1),$([nX2;H0]1[cX3;H1][cX3;H0][cX3;H1][cX3;H1][cX3;H1]1),$([nX2;H0]1[cX3;H1][cX3;H1][cX3;H0][cX3;H1][cX3;H1]1),$([nX2;H0]1[cX3;H1][cX3;H1][cX3;H1][cX3;H0][cX3;H1]1),$([nX2;H0]1[cX3;H1][cX3;H1][cX3;H1][cX3;H1][cX3;H0]1)]') # Perfect hand made
UFSG[39] = UNIFAC_subgroup('C5H3N', 18, 'PYRIDINE', 2.667, 1.553, smarts='[$([nX2;H0]1[cX3;H0][cX3;H0][cX3;H1][cX3;H1][cX3;H1]1),$([nX2;H0]1[cX3;H0][cX3;H1][cX3;H0][cX3;H1][cX3;H1]1),$([nX2;H0]1[cX3;H0][cX3;H1][cX3;H1][cX3;H0][cX3;H1]1),$([nX2;H0]1[cX3;H0][cX3;H1][cX3;H1][cX3;H1][cX3;H0]1),$([nX2;H0]1[cX3;H1][cX3;H0][cX3;H0][cX3;H1][cX3;H1]1),$([nX2;H0]1[cX3;H1][cX3;H0][cX3;H1][cX3;H0][cX3;H1]1),$([nX2;H0]1[cX3;H1][cX3;H0][cX3;H1][cX3;H1][cX3;H0]1),$([nX2;H0]1[cX3;H1][cX3;H1][cX3;H0][cX3;H0][cX3;H1]1),$([nX2;H0]1[cX3;H1][cX3;H1][cX3;H0][cX3;H1][cX3;H0]1),$([nX2;H0]1[cX3;H1][cX3;H1][cX3;H1][cX3;H0][cX3;H0]1)]') # Very much not fun to make, 200 matches perfectly, 10 different ring combos
UFSG[40] = UNIFAC_subgroup('CH3CN', 19, 'CCN', 1.8701, 1.724, smarts='[CX4;H3][CX2]#[NX1]')
UFSG[41] = UNIFAC_subgroup('CH2CN', 19, 'CCN', 1.6434, 1.416, smarts='[CX4;H2][CX2]#[NX1]')
UFSG[42] = UNIFAC_subgroup('COOH', 20, 'COOH', 1.3013, 1.224, smarts='[CX3](=[OX1])[O;H1]') # Tried '[C][CX3](=[OX1])[OH1]' at first but fails for a few hundred
UFSG[43] = UNIFAC_subgroup('HCOOH', 20, 'COOH', 1.528, 1.532, smarts='[CX3;H1](=[OX1])[OX2;H1]') # effortlessly web - missing one hit
UFSG[44] = UNIFAC_subgroup('CH2CL', 21, 'CCL', 1.4654, 1.264, smarts='[CX4;H2;!$(C(Cl)(Cl))](Cl)') # Matches perfectly effortlessly web
UFSG[45] = UNIFAC_subgroup('CHCL', 21, 'CCL', 1.238, 0.952, smarts='[CX4;H1;!$(C(Cl)(Cl))](Cl)') # effortlessly web
UFSG[46] = UNIFAC_subgroup('CCL', 21, 'CCL', 1.0106, 0.724, smarts='[CX4;H0;!$(C(Cl)(Cl))](Cl)') # effortlessly web
UFSG[47] = UNIFAC_subgroup('CH2CL2', 22, 'CCL2', 2.2564, 1.988, smarts='[CX4;H2;!$(C(Cl)(Cl)(Cl))](Cl)(Cl)') # effortlessly web
UFSG[48] = UNIFAC_subgroup('CHCL2', 22, 'CCL2', 2.0606, 1.684, smarts='[CX4;H1;!$(C(Cl)(Cl)(Cl))](Cl)(Cl)') # effortlessly web
UFSG[49] = UNIFAC_subgroup('CCL2', 22, 'CCL2', 1.8016, 1.448, smarts='[CX4;H0;!$(C(Cl)(Cl)(Cl))](Cl)(Cl)') # effortlessly web
UFSG[50] = UNIFAC_subgroup('CHCL3', 23, 'CCL3', 2.87, 2.41, smarts='[CX4;H1;!$([CX4;H0](Cl)(Cl)(Cl)(Cl))](Cl)(Cl)(Cl)') # effortlessly web
UFSG[51] = UNIFAC_subgroup('CCL3', 23, 'CCL3', 2.6401, 2.184, smarts='[CX4;H0;!$([CX4;H0](Cl)(Cl)(Cl)(Cl))](Cl)(Cl)(Cl)') # effortlessly web
UFSG[52] = UNIFAC_subgroup('CCL4', 24, 'CCL4', 3.39, 2.91, smarts='[CX4;H0]([Cl])([Cl])([Cl])([Cl])')
UFSG[53] = UNIFAC_subgroup('ACCL', 25, 'ACCL', 1.1562, 0.844, smarts='aCl') # Does take up one of the carbon spaces on the ring
UFSG[54] = UNIFAC_subgroup('CH3NO2', 26, 'CNO2', 2.0086, 1.868, smarts='[CX4;H3][NX3](=[OX1])(=[OX1])') # untested
UFSG[55] = UNIFAC_subgroup('CH2NO2', 26, 'CNO2', 1.7818, 1.56, smarts='[CX4;H2][NX3](=[OX1])(=[OX1])') # untested
UFSG[56] = UNIFAC_subgroup('CHNO2', 26, 'CNO2', 1.5544, 1.248, smarts='[CX4;H1][NX3](=[OX1])(=[OX1])') # untested
UFSG[57] = UNIFAC_subgroup('ACNO2', 27, 'ACNO2', 1.4199, 1.104, smarts='[cX3][NX3](=[OX1])(=[OX1])')
UFSG[58] = UNIFAC_subgroup('CS2', 28, 'CS2', 2.057, 1.65, smarts='C(=S)=S') # Easy, compount smarts
UFSG[59] = UNIFAC_subgroup('CH3SH', 29, 'CH3SH', 1.877, 1.676, smarts='[SX2H][CX4;H3]') # perfect match
UFSG[60] = UNIFAC_subgroup('CH2SH', 29, 'CH3SH', 1.651, 1.368, smarts='[SX2H][CX4;H2]')
UFSG[61] = UNIFAC_subgroup('FURFURAL', 30, 'FURFURAL', 3.168, 2.484, smarts='c1cc(oc1)C=O') # Easy, compound smarts, 1 hit only
UFSG[62] = UNIFAC_subgroup('DOH', 31, 'DOH', 2.4088, 2.248, smarts='C(CO)O') # Probably going to cause problems, match too much
UFSG[63] = UNIFAC_subgroup('I', 32, 'I', 1.264, 0.992, smarts='[I]')
UFSG[64] = UNIFAC_subgroup('BR', 33, 'BR', 0.9492, 0.832, smarts='[Br]')
UFSG[65] = UNIFAC_subgroup('CH=-C', 34, 'C=-C', 1.292, 1.088, smarts='[CX2;H1]#[CX2;H0]')
UFSG[66] = UNIFAC_subgroup('C=-C', 34, 'C=-C', 1.0613, 0.784, smarts='[CX2;H0]#[CX2;H0]')
UFSG[67] = UNIFAC_subgroup('DMSO', 35, 'DMSO', 2.8266, 2.472, smarts='CS(=O)C') # Compound smarts; some extra matches
UFSG[68] = UNIFAC_subgroup('ACRY', 36, 'ACRY', 2.3144, 2.052, smarts='N#CC=C') # Easy, compount smarts
UFSG[69] = UNIFAC_subgroup('CL-(C=C)', 37, 'CLCC', 0.791, 0.724, smarts='Cl[CX3]=[CX3]')
UFSG[70] = UNIFAC_subgroup('C=C', 2, 'C=C', 0.6605, 0.485, smarts='[CX3;H0]=[CX3;H0]') # ddbst matches some of these into rings incorrectly
UFSG[71] = UNIFAC_subgroup('ACF', 38, 'ACF', 0.6948, 0.524, smarts='[cX3][F]') # Perfect for many, except 71671-89-1
UFSG[72] = UNIFAC_subgroup('DMF', 39, 'DMF', 3.0856, 2.736, smarts='CN(C)C=O') # Probably going to cause problems, match too much
UFSG[73] = UNIFAC_subgroup('HCON(CH2)2', 39, 'DMF', 2.6322, 2.12, smarts='[CX4;H2,H1,H3][CX4;H2,H1][NX3][CX3;H1](=[OX1])') # No idea why extra H3 was needed  to match one compound
UFSG[74] = UNIFAC_subgroup('CF3', 40, 'CF2', 1.406, 1.38, smarts='C(F)(F)F')
UFSG[75] = UNIFAC_subgroup('CF2', 40, 'CF2', 1.0105, 0.92, smarts='C(F)F') # Probably going to cause problems, match too much
UFSG[76] = UNIFAC_subgroup('CF', 40, 'CF2', 0.615, 0.46, smarts='C(F)') # Probably going to cause problems, match too much
UFSG[77] = UNIFAC_subgroup('COO', 41, 'COO', 1.38, 1.2, smarts='[CX3,cX3](=[OX1])[OX2,oX2]') # ddbst wants match into rings, thus the cX3, oX2
UFSG[78] = UNIFAC_subgroup('SIH3', 42, 'SIH2', 1.6035, 1.2632, smarts='[SiX4,SiX3,SiX5;H3]') # some db smiles compounds missing Hs, not matched not due to smarts
UFSG[79] = UNIFAC_subgroup('SIH2', 42, 'SIH2', 1.4443, 1.0063, smarts='[SiX4,SiX3,SiX5,SiX2;H2]') # some db smiles compounds missing Hs, not matched not due to smarts
UFSG[80] = UNIFAC_subgroup('SIH', 42, 'SIH2', 1.2853, 0.7494, smarts='[SiX4,SiX3,SiX5,SiX2,SiX1;H1]') # some db smiles compounds missing Hs, not matched not due to smarts
UFSG[81] = UNIFAC_subgroup('SI', 42, 'SIH2', 1.047, 0.4099, smarts='[Si]')
UFSG[82] = UNIFAC_subgroup('SIH2O', 43, 'SIO', 1.4838, 1.0621) # Cannot find; Si problems
UFSG[83] = UNIFAC_subgroup('SIHO', 43, 'SIO', 1.303, 0.7639) # Cannot find; Si problems
UFSG[84] = UNIFAC_subgroup('SIO', 43, 'SIO', 1.1044, 0.4657) # Cannot find; Si problems
UFSG[85] = UNIFAC_subgroup('NMP', 44, 'NMP', 3.981, 3.2, smarts='CN1CCCC1=O') # 1 compound
UFSG[86] = UNIFAC_subgroup('CCL3F', 45, 'CCLF', 3.0356, 2.644, smarts='C(F)(Cl)(Cl)Cl') # pure compound?
UFSG[87] = UNIFAC_subgroup('CCL2F', 45, 'CCLF', 2.2287, 1.916, smarts='C(F)(Cl)Cl')
UFSG[88] = UNIFAC_subgroup('HCCL2F', 45, 'CCLF', 2.406, 2.116, smarts='ClC(Cl)F')
UFSG[89] = UNIFAC_subgroup('HCCLF', 45, 'CCLF', 1.6493, 1.416, smarts='C(Cl)F')
UFSG[90] = UNIFAC_subgroup('CCLF2', 45, 'CCLF', 1.8174, 1.648, smarts='ClC(F)(F)')
UFSG[91] = UNIFAC_subgroup('HCCLF2', 45, 'CCLF', 1.967, 1.828, smarts='ClC(F)F')
UFSG[92] = UNIFAC_subgroup('CCLF3', 45, 'CCLF', 2.1721, 2.1, smarts='ClC(F)(F)F') # perfect
UFSG[93] = UNIFAC_subgroup('CCL2F2', 45, 'CCLF', 2.6243, 2.376, smarts='ClC(Cl)(F)F') # perfect
UFSG[94] = UNIFAC_subgroup('AMH2', 46, 'CON(AM)', 1.4515, 1.248, smarts='[CX3;H0](=[OX1])[NX3;H2]')
UFSG[95] = UNIFAC_subgroup('AMHCH3', 46, 'CON(AM)', 2.1905, 1.796, smarts='[CX3;H0](=[OX1])[NX3;H1][CX4;H3]') # 3 extra hits, effortlessly web
UFSG[96] = UNIFAC_subgroup('AMHCH2', 46, 'CON(AM)', 1.9637, 1.488, smarts='[CX3;H0](=[OX1])[NX3;H1][CX4;H2]') # 4 extra hits, effortlessly web
UFSG[97] = UNIFAC_subgroup('AM(CH3)2', 46, 'CON(AM)', 2.8589, 2.428, smarts='[CX3;H0](=[OX1])[NX3;H0]([CX4;H3])[CX4;H3]') # effortlessly web
UFSG[98] = UNIFAC_subgroup('AMCH3CH2', 46, 'CON(AM)', 2.6322, 2.12, smarts='[CX3;H0](=[OX1])[NX3;H0]([CX4;H3])[CX4;H2]') # 1 extra hits, effortlessly web
UFSG[99] = UNIFAC_subgroup('AM(CH2)2', 46, 'CON(AM)', 2.4054, 1.812, smarts='[CX3;H0](=[OX1])[NX3;H0]([CX4;H2])[CX4;H2]') # 2 extra hits, effortlessly web
UFSG[100] = UNIFAC_subgroup('C2H5O2', 47, 'OCCOH', 2.1226, 1.904, smarts='[CX4;H2]([OX2;H1])[CX4;H2][OX2;H0]') # Matches all; 53 extra hits
UFSG[101] = UNIFAC_subgroup('C2H4O2', 47, 'OCCOH', 1.8952, 1.592, smarts='[$([CX4;H1]([OX2;H1])[CX4;H2][OX2;H0]),$([CX4;H2]([OX2;H1])[CX4;H1][OX2;H0])]') # custom expression
UFSG[102] = UNIFAC_subgroup('CH3S', 48, 'CH2S', 1.613, 1.368, smarts='[CX4;H3][SX2]')
UFSG[103] = UNIFAC_subgroup('CH2S', 48, 'CH2S', 1.3863, 1.06, smarts='[CX4;H2][SX2]')
UFSG[104] = UNIFAC_subgroup('CHS', 48, 'CH2S', 1.1589, 0.748, smarts='[CX4,CX3,CX2;H1][S]') # S bond might need to be more restricted; C bond might need to be more restricted
UFSG[105] = UNIFAC_subgroup('MORPH', 49, 'MORPH', 3.474, 2.796, smarts='C1CNCCO1') # lots of extra hits 100
UFSG[106] = UNIFAC_subgroup('C4H4S', 50, 'THIOPHEN', 2.8569, 2.14, smarts='[cX3;H1][cX3;H1][cX3;H1][sX2;H0][cX3;H1]') # Custom tuned - matches perfectly no extras, might need to be more slack in the future
UFSG[107] = UNIFAC_subgroup('C4H3S', 50, 'THIOPHEN', 2.6908, 1.86, smarts='[$([cX3;H1][cX3;H1][cX3;H1][sX2;H0][cX3;H0]),$([cX3;H1][cX3;H0][cX3;H1][sX2;H0][cX3;H1])]') # 1 extra - custom tuned
UFSG[108] = UNIFAC_subgroup('C4H2S', 50, 'THIOPHEN', 2.5247, 1.58, smarts='[cX3;H1,H0;r5][cX3;H1,H0;r5][cX3;H0,H1;r5][sX2;H0;r5][cX3;H0,H1;r5]') # Not sure if this is right - probably not!!!
UFSG[109] = UNIFAC_subgroup('NCO', 51, 'NCO', 1.0567, 0.732, smarts='[NX2]=[CX2]=[OX1]') # Bonds might need to be different - but this smarts matches them all so far
UFSG[118] = UNIFAC_subgroup('(CH2)2SU', 55, 'SULFONES', 2.6869, 2.12, smarts='CC[SX4](=O)(=O)') # TYPO on their part; Makes no sense for there to be CH3 groups
UFSG[119] = UNIFAC_subgroup('CH2CHSU', 55, 'SULFONES', 2.4595, 1.808, smarts='[CX4;H2][CX4;H1][SX4](=O)(=O)') # 3 missing of 6
UFSG[178] = UNIFAC_subgroup('IMIDAZOL', 84, 'IMIDAZOL', 2.026, 0.868, smarts='c1cnc[nH]1') # untested
UFSG[179] = UNIFAC_subgroup('BTI', 85, 'BTI', 5.774, 4.932, smarts='C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F') # untested



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
# Is 90 missing?
DOUFSG[91] = UNIFAC_subgroup('CONH2', 47, 'CONR', 1.4515, 1.248)
DOUFSG[92] = UNIFAC_subgroup('CONHCH3', 47, 'CONR', 1.5, 1.08)
# 93, 98, 99 missing but inteaction parameters are available.
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
# 122, 123, 124, 201 Added Rev. 6
DOUFSG[122] = UNIFAC_subgroup('CH3S', 61, 'SULFIDES', 1.6130, 1.3680)
DOUFSG[123] = UNIFAC_subgroup('CH2S', 61, 'SULFIDES', 1.3863, 1.0600)
DOUFSG[124] = UNIFAC_subgroup('CHS', 61, 'SULFIDES', 1.1589, 0.7480)
DOUFSG[201] = UNIFAC_subgroup('-S-S-', 93, 'DISULFIDES', 1.0678, 2.2440)

#  subgroup = (group, (subgroup ids))
# http://www.ddbst.com/PublishedParametersUNIFACDO.html#ListOfMainGroups
DOUFMG = {}
DOUFMG[1] = ('CH2', [1, 2, 3, 4])
DOUFMG[2] = ('C=C', [5, 6, 7, 8, 70])
DOUFMG[3] = ('ACH', [9, 10])
DOUFMG[4] = ('ACCH2', [11, 12, 13])
DOUFMG[5] = ('OH', [14, 81, 82])
DOUFMG[6] = ('CH3OH', [15])
DOUFMG[7] = ('H2O', [16])
DOUFMG[8] = ('ACOH', [17])
DOUFMG[9] = ('CH2CO', [18, 19])
DOUFMG[10] = ('CHO', [20])
DOUFMG[11] = ('CCOO', [21, 22])
DOUFMG[12] = ('HCOO', [23])
DOUFMG[13] = ('CH2O', [24, 25, 26])
DOUFMG[14] = ('CH2NH2', [28, 29, 30, 85])
DOUFMG[15] = ('CH2NH', [31, 32, 33])
DOUFMG[16] = ('(C)3N', [34, 35])
DOUFMG[17] = ('ACNH2', [36])
DOUFMG[18] = ('PYRIDINE', [37, 38, 39])
DOUFMG[19] = ('CH2CN', [40, 41])
DOUFMG[20] = ('COOH', [42])
DOUFMG[21] = ('CCL', [44, 45, 46])
DOUFMG[22] = ('CCL2', [47, 48, 49])
DOUFMG[23] = ('CCL3', [51])
DOUFMG[24] = ('CCL4', [52])
DOUFMG[25] = ('ACCL', [53])
DOUFMG[26] = ('CNO2', [54, 55, 56])
DOUFMG[27] = ('ACNO2', [57])
DOUFMG[28] = ('CS2', [58])
DOUFMG[29] = ('CH3SH', [59, 60])
DOUFMG[30] = ('FURFURAL', [61])
DOUFMG[31] = ('DOH', [62])
DOUFMG[32] = ('I', [63])
DOUFMG[33] = ('BR', [64])
DOUFMG[34] = ('C=-C', [65, 66])
DOUFMG[35] = ('DMSO', [67])
DOUFMG[36] = ('ACRY', [68])
DOUFMG[37] = ('CLCC', [69])
DOUFMG[38] = ('ACF', [71])
DOUFMG[39] = ('DMF', [72, 73])
DOUFMG[40] = ('CF2', [74, 75, 76])
DOUFMG[41] = ('COO', [77])
DOUFMG[42] = ('CY-CH2', [78, 79, 80])
DOUFMG[43] = ('CY-CH2O', [27, 83, 84])
DOUFMG[44] = ('HCOOH', [43])
DOUFMG[45] = ('CHCL3', [50])
DOUFMG[46] = ('CY-CONC', [86, 87, 88, 89])
DOUFMG[47] = ('CONR', [91, 92, 100])
DOUFMG[48] = ('CONR2', [101, 102, 103])
DOUFMG[49] = ('HCONR', [93, 94]) # Added in Further Development of Modified UNIFAC (Dortmund):  Revision and Extension 5
DOUFMG[52] = ('ACS', [104, 105, 106])
DOUFMG[53] = ('EPOXIDES', [107, 108, 109, 119, 153])
DOUFMG[55] = ('CARBONAT', [112, 113, 114])
DOUFMG[56] = ('SULFONE', [110, 111])
DOUFMG[84] = ('IMIDAZOL', [178, 184])
DOUFMG[85] = ('BTI', [179])
DOUFMG[87] = ('PYRROL', [189])
DOUFMG[89] = ('BF4', [195])
DOUFMG[90] = ('PYRIDIN', [196])
DOUFMG[91] = ('OTF', [197])
# Added Rev 6
DOUFMG[61] = ('SULFIDES', [122, 123, 124])
DOUFMG[93] = ('DISULFIDES', [201])


VTPRSG = {}

#  subgroup = (subgroup, #maingroup, maingroup, R, Q)
VTPRSG = {}
VTPRSG[1] = UNIFAC_subgroup('CH3', 1, 'CH2', None, 1.2958, smarts=DOUFSG[1].smarts)
VTPRSG[2] = UNIFAC_subgroup('CH2', 1, 'CH2', None, 0.9471, smarts=DOUFSG[2].smarts)
VTPRSG[3] = UNIFAC_subgroup('CH', 1, 'CH2', None, 0.2629, smarts=DOUFSG[3].smarts)
VTPRSG[4] = UNIFAC_subgroup('C', 1, 'CH2', None, 0, smarts=DOUFSG[4].smarts)

VTPRSG[5] = UNIFAC_subgroup('CH2=CH', 2, 'C=C', None, 1.1507, smarts=DOUFSG[5].smarts)
VTPRSG[6] = UNIFAC_subgroup('CH=CH', 2, 'C=C', None, 1.3221, smarts=DOUFSG[6].smarts)
VTPRSG[7] = UNIFAC_subgroup('CH2=C', 2, 'C=C', None, 0.9889, smarts=DOUFSG[7].smarts)
VTPRSG[8] = UNIFAC_subgroup('CH=C', 2, 'C=C', None, 0.6760, smarts=DOUFSG[8].smarts)
VTPRSG[70] = UNIFAC_subgroup('C=C', 2, 'C=C', None, 0.4850, smarts=DOUFSG[70].smarts)
VTPRSG[97] = UNIFAC_subgroup('Allene', 2, 'Allene', None, 1.1287, smarts=None)
VTPRSG[98] = UNIFAC_subgroup('=CHCH=', 2, '=CHCH=', None, 1.7345, smarts=None)
VTPRSG[99] = UNIFAC_subgroup('=CCH=', 2, '=CCH=', None, 3.5331, smarts=None)
VTPRSG[250] = UNIFAC_subgroup('H2C=CH2', 2, 'H2C=CH2', None, 0.6758, smarts=None)

VTPRSG[9] = UNIFAC_subgroup('ACH', 3, 'ACH', None, 0.4972, smarts=DOUFSG[9].smarts)
VTPRSG[10] = UNIFAC_subgroup('AC', 3, 'ACH', None, 0.1885, smarts=DOUFSG[10].smarts)

VTPRSG[11] = UNIFAC_subgroup('ACCH3', 4, 'ACCH2', None, 1.4843, smarts=DOUFSG[11].smarts)
VTPRSG[12] = UNIFAC_subgroup('ACCH2', 4, 'ACCH2', None, 1.1356, smarts=DOUFSG[12].smarts)
VTPRSG[13] = UNIFAC_subgroup('ACCH', 4, 'ACCH2', None, 0.4514, smarts=DOUFSG[13].smarts)

VTPRSG[14] = UNIFAC_subgroup('OH(P)', 5, 'OH', None, 1.0189, smarts=DOUFSG[14].smarts)
VTPRSG[81] = UNIFAC_subgroup('OH(S)', 5, 'OH', None, 0.9326, smarts=DOUFSG[81].smarts)
VTPRSG[82] = UNIFAC_subgroup('OH(T)', 5, 'OH', None, 0.8727, smarts=DOUFSG[82].smarts)

VTPRSG[15] = UNIFAC_subgroup('CH3OH', 6, 'CH3OH', None, 0.8779, smarts=DOUFSG[15].smarts)

VTPRSG[16] = UNIFAC_subgroup('H2O', 7, 'H2O', None, 1.5576, smarts=DOUFSG[16].smarts)

VTPRSG[17] = UNIFAC_subgroup('ACOH', 8, 'ACOH', None, 0.9013, smarts=DOUFSG[17].smarts)

VTPRSG[18] = UNIFAC_subgroup('CH3CO', 9, 'CH2CO', None, 1.448, smarts=DOUFSG[18].smarts)
VTPRSG[19] = UNIFAC_subgroup('CH2CO', 9, 'CH2CO', None, 1.18, smarts=DOUFSG[19].smarts)

VTPRSG[20] = UNIFAC_subgroup('CHO', 10, 'CHO', None, 0.948, smarts=DOUFSG[20].smarts)

VTPRSG[21] = UNIFAC_subgroup('CH3COO', 11, 'CCOO', None, 1.728, smarts=DOUFSG[21].smarts)
VTPRSG[22] = UNIFAC_subgroup('CH2COO', 11, 'CCOO', None, 1.42, smarts=DOUFSG[22].smarts)
VTPRSG[129] = UNIFAC_subgroup('CHCOO', 11, 'CCOO', None, 1.221, smarts=None)
VTPRSG[180] = UNIFAC_subgroup('CHCOO', 11, 'CCOO', None, 0.88, smarts=None)

VTPRSG[23] = UNIFAC_subgroup('HCOO', 12, 'HCOO', None, 1.1880, smarts=DOUFSG[23].smarts)

VTPRSG[24] = UNIFAC_subgroup('CH3O', 13, 'CH2O', None, 1.088, smarts=DOUFSG[24].smarts)
VTPRSG[25] = UNIFAC_subgroup('CH2O', 13, 'CH2O', None, 0.78, smarts=DOUFSG[25].smarts)
VTPRSG[26] = UNIFAC_subgroup('CHO', 13, 'CH2O', None, 0.468, smarts=DOUFSG[26].smarts)

VTPRSG[28] = UNIFAC_subgroup('CH3NH2', 14, 'CH2NH2', None, 1.2260, smarts=DOUFSG[28].smarts)
VTPRSG[29] = UNIFAC_subgroup('CH2NH2', 14, 'CH2NH2', None, 1.2360, smarts=DOUFSG[29].smarts)
VTPRSG[30] = UNIFAC_subgroup('CHNH2', 14, 'CH2NH2', None, 1.1868, smarts=DOUFSG[30].smarts)
VTPRSG[85] = UNIFAC_subgroup('CNH2', 14, 'CH2NH2', None, 1.1527, smarts=DOUFSG[85].smarts)

VTPRSG[31] = UNIFAC_subgroup('CH3NH', 15, 'CH2NH', None, 1.2440, smarts=DOUFSG[31].smarts)
VTPRSG[32] = UNIFAC_subgroup('CH2NH', 15, 'CH2NH', None, 0.936, smarts=DOUFSG[32].smarts)
VTPRSG[33] = UNIFAC_subgroup('CHNH', 15, 'CH2NH', None, 0.6240, smarts=DOUFSG[33].smarts)
VTPRSG[34] = UNIFAC_subgroup('CH3N', 16, '(C)3N', None, 0.94, smarts=DOUFSG[34].smarts)
VTPRSG[35] = UNIFAC_subgroup('CH2N', 16, '(C)3N', None, 0.632, smarts=DOUFSG[35].smarts)
VTPRSG[36] = UNIFAC_subgroup('ACNH2', 17, 'ACNH2', None, 0.8160, smarts=DOUFSG[36].smarts)
VTPRSG[40] = UNIFAC_subgroup('CH3CN', 19, 'CH2CN', None, 1.5302, smarts=DOUFSG[40].smarts)
VTPRSG[41] = UNIFAC_subgroup('CH2CN', 19, 'CH2CN', None, 1.4492, smarts=DOUFSG[41].smarts)
VTPRSG[44] = UNIFAC_subgroup('CH2CL', 21, 'CCL', None, 1.264, smarts=DOUFSG[44].smarts)
VTPRSG[45] = UNIFAC_subgroup('CHCL', 21, 'CCL', None, 0.952, smarts=DOUFSG[45].smarts)
VTPRSG[46] = UNIFAC_subgroup('CCL', 21, 'CCL', None, 0.724, smarts=DOUFSG[46].smarts)
VTPRSG[47] = UNIFAC_subgroup('CH2CL2', 22, 'CCL2', None, 1.9880, smarts=DOUFSG[47].smarts)
VTPRSG[48] = UNIFAC_subgroup('CHCL2', 22, 'CCL2', None, 1.6840, smarts=DOUFSG[48].smarts)
VTPRSG[49] = UNIFAC_subgroup('CCL2', 22, 'CCL2', None, 1.4480, smarts=DOUFSG[49].smarts)
VTPRSG[51] = UNIFAC_subgroup('CCL3', 23, 'CCL3', None, 2.1840, smarts=DOUFSG[51].smarts)
VTPRSG[52] = UNIFAC_subgroup('CCL4', 24, 'CCL4', None, 3.1836, smarts=DOUFSG[52].smarts)

VTPRSG[53] = UNIFAC_subgroup('ACCL', 25, 'ACCL', None, 0.3177, smarts=DOUFSG[53].smarts) # Q not verified - from DO, not listed, in 2016 paper

VTPRSG[59] = UNIFAC_subgroup('CH3SH', 29, 'CH3SH', None, 1.762, smarts=DOUFSG[59].smarts)# Q not verified - from DO, not listed, in 2016 paper
VTPRSG[60] = UNIFAC_subgroup('CH2SH', 29, 'CH3SH', None, 1.316, smarts=DOUFSG[60].smarts)# Q not verified - from DO, not listed, in 2016 paper

VTPRSG[58] = UNIFAC_subgroup('CS2', 28, 'CS2', None, 1.65, smarts=DOUFSG[58].smarts)
VTPRSG[61] = UNIFAC_subgroup('FURFURAL', 30, 'FURFURAL', None, 2.0363, smarts=DOUFSG[61].smarts)
VTPRSG[62] = UNIFAC_subgroup('DOH', 31, 'DOH', None, 2.2480, smarts=DOUFSG[62].smarts)
VTPRSG[63] = UNIFAC_subgroup('I', 32, 'I', None, 0.9920, smarts=DOUFSG[63].smarts)
VTPRSG[64] = UNIFAC_subgroup('BR', 33, 'BR', None, 0.8320, smarts=DOUFSG[64].smarts)
VTPRSG[67] = UNIFAC_subgroup('DMSO', 35, 'DMSO', None, 2.4720, smarts=DOUFSG[67].smarts)
VTPRSG[72] = UNIFAC_subgroup('DMF', 39, 'DMF', None, 2.7360, smarts=DOUFSG[72].smarts)
VTPRSG[73] = UNIFAC_subgroup('HCON(..', 39, 'DMF', None, 2.1200, smarts=DOUFSG[73].smarts)
VTPRSG[78] = UNIFAC_subgroup('CY-CH2', 42, 'CY-CH2', None, 0.8635, smarts=DOUFSG[78].smarts)
VTPRSG[79] = UNIFAC_subgroup('CY-CH', 42, 'CY-CH2', None, 0.1071, smarts=DOUFSG[79].smarts)
VTPRSG[80] = UNIFAC_subgroup('CY-C', 42, 'CY-CH2', None, 0, smarts=DOUFSG[80].smarts)
VTPRSG[27] = UNIFAC_subgroup('THF', 43, 'CY-CH2O', None, 2.3637, smarts=DOUFSG[27].smarts)
VTPRSG[83] = UNIFAC_subgroup('CY-CH2O', 43, 'CY-CH2O', None, 1.4, smarts=DOUFSG[83].smarts)
VTPRSG[84] = UNIFAC_subgroup('TRIOXAN', 43, 'CY-CH2O', None, 1.0116, smarts=DOUFSG[84].smarts)
VTPRSG[50] = UNIFAC_subgroup('CHCL3', 45, 'CHCL3', None, 2.4100, smarts=DOUFSG[50].smarts)
VTPRSG[86] = UNIFAC_subgroup('NMP', 46, 'CY-CONC', None, 3.2, smarts=DOUFSG[86].smarts)
VTPRSG[87] = UNIFAC_subgroup('NEP', 46, 'CY-CONC', None, 2.892, smarts=DOUFSG[87].smarts)
VTPRSG[88] = UNIFAC_subgroup('NIPP', 46, 'CY-CONC', None, 2.58, smarts=DOUFSG[88].smarts)
VTPRSG[89] = UNIFAC_subgroup('NTBP', 46, 'CY-CONC', None, 2.352, smarts=DOUFSG[89].smarts)
VTPRSG[107] = UNIFAC_subgroup('H2COCH', 53, 'EPOXIDES', None, 1.8031, smarts=DOUFSG[107].smarts)
VTPRSG[108] = UNIFAC_subgroup('COCH', 53, 'EPOXIDES', None, 0.3418, smarts=DOUFSG[108].smarts)
VTPRSG[109] = UNIFAC_subgroup('HCOCH', 53, 'EPOXIDES', None, 0.6538, smarts=DOUFSG[109].smarts)
VTPRSG[119] = UNIFAC_subgroup('H2COCH2', 53, 'EPOXIDES', None, 1.123, smarts=DOUFSG[119].smarts)
VTPRSG[153] = UNIFAC_subgroup('H2COC', 53, 'EPOXIDES', None, 0.6538, smarts=DOUFSG[153].smarts)
VTPRSG[116] = UNIFAC_subgroup('AC-CHO', 57, 'AC-CHO', None, 1.0, smarts=None)
VTPRSG[139] = UNIFAC_subgroup('CF2H', 68, 'CF2H', None, 1.6643, smarts=None)
VTPRSG[140] = UNIFAC_subgroup('CF2H2', 68, 'CF2H', None, 1.3304, smarts=None)
VTPRSG[142] = UNIFAC_subgroup('CF2Cl', 70, 'CF2Cl2', None, 1.8506, smarts=None)
VTPRSG[143] = UNIFAC_subgroup('CF2Cl2', 70, 'CF2Cl2', None, 2.5974, smarts=None)
VTPRSG[148] = UNIFAC_subgroup('CF3Br', 70, 'CF2Cl2', None, 2.5104, smarts=None)

VTPRSG[146] = UNIFAC_subgroup('CF4', 73, 'CF4', None, 1.8400, smarts=None)
VTPRSG[300] = UNIFAC_subgroup('NH3', 150, 'NH3', None, 0.7780, smarts=None)
VTPRSG[306] = UNIFAC_subgroup('CO2', 151, 'CO2', None, 0.982, smarts=None)
VTPRSG[307] = UNIFAC_subgroup('CH4', 152, 'CH4', None, 1.124, smarts=None)
VTPRSG[308] = UNIFAC_subgroup('O2', 153, 'O2', None, 0.849, smarts=None)
VTPRSG[305] = UNIFAC_subgroup('Ar', 154, 'Ar', None, 1.116, smarts=None)
VTPRSG[304] = UNIFAC_subgroup('N2', 155, 'N2', None, 0.93, smarts=None)
VTPRSG[303] = UNIFAC_subgroup('H2S', 156, 'H2S', None, 1.202, smarts=None)
VTPRSG[302] = UNIFAC_subgroup('H2', 157, 'H2', None, 0.571, smarts=None)
VTPRSG[309] = UNIFAC_subgroup('D2', 157, 'D2', None, 0.527, smarts=None)
VTPRSG[301] = UNIFAC_subgroup('CO', 158, 'CO', None, 0.8280, smarts=None)
VTPRSG[310] = UNIFAC_subgroup('SO2', 160, 'SO2', None, 1.1640, smarts=None)
VTPRSG[312] = UNIFAC_subgroup('N2O', 162, 'N2O', None, 0.8880, smarts=None)
VTPRSG[314] = UNIFAC_subgroup('He', 164, 'He', None, 0.9850, smarts=None)
VTPRSG[315] = UNIFAC_subgroup('Ne', 165, 'Ne', None, 0.9860, smarts=None)
VTPRSG[319] = UNIFAC_subgroup('HCl', 169, 'HCl', None, 1.2560, smarts=None)
VTPRSG[345] = UNIFAC_subgroup('Hg', 185, 'Hg', None, 7.9616, smarts=None)

# From Present Status of the Group Contribution Equation of State VTPR and Typical Applications for Process Development
VTPRSG[54] = UNIFAC_subgroup('CH3NO2', 26, 'CNO2', None, 1.8285, smarts=DOUFSG[54].smarts)
VTPRSG[55] = UNIFAC_subgroup('CH2NO2', 26, 'CNO2', None, 1.56, smarts=DOUFSG[55].smarts)
VTPRSG[56] = UNIFAC_subgroup('CHNO2', 26, 'CNO2', None, 1.248, smarts=DOUFSG[56].smarts)


VTPRMG = {1: ("CH2", [1, 2, 3, 4]),
2: ("H2C=CH2", [5, 6, 7, 8, 70, 97, 98, 99, 250]),
3: ("ACH", [9, 10]),
4: ("ACCH2", [11, 12, 13]),
5: ("OH", [14, 81, 82]),
6: ("CH3OH", [15]),
7: ("H2O", [16]),
8: ("ACOH", [17]),
9: ("CH2CO", [18, 19]),
10: ("CHO", [20]),
11: ("CCOO", [21, 22, 129, 180]),
12: ("HCOO", [23]),
13: ("CH2O", [24, 25, 26]),
14: ("CH2NH2", [28, 29, 30, 85]),
15: ("CH2NH", [31, 32, 33]),
16: ("(C)3N", [34, 35]),
17: ("ACNH2", [36]),
19: ("CH2CN", [40, 41]),
21: ("CCL", [44, 45, 46]),
22: ("CCL2", [47, 48, 49]),
23: ("CCL3", [51]),
24: ("CCL4", [52]),
25: ("ACCL", [53]),
26: ("CNO2", [54, 55, 56]),
28: ("CS2", [58]),
29: ("CH3SH", [59, 60]),
30: ("FURFURAL", [61]),
31: ("DOH", [62]),
32: ("I", [63]),
33: ("BR", [64]),
35: ("DMSO", [67]),
39: ("DMF", [72, 73]),
42: ("CY-CH2", [78, 79, 80]),
43: ("CY-CH2O", [27, 83, 84]),
45: ("CHCL3", [50]),
46: ("CY-CONC", [86, 87, 88, 89]),
53: ("EPOXIDES", [107, 108, 109, 119, 153]),
57: ("AC-CHO", [116]),
68: ("CF2H", [139, 140]),
70: ("CF2Cl2", [142, 143, 148]),
73: ("CF4", [146]),
150: ("NH3", [300]),
151: ("CO2", [306]),
152: ("CH4", [307]),
153: ("O2", [308]),
154: ("Ar", [305]),
155: ("N2", [304]),
156: ("H2S", [303]),
157: ("D2", [302, 309]),
158: ("CO", [301]),
160: ("SO2", [310]),
162: ("N2O", [312]),
164: ("He", [314]),
165: ("Ne", [315]),
169: ("HCl", [319]),
185: ("Hg", [345]),
}


NISTUFMG = {}
# From Kang and Diky and Chirico and Magee and Muzny and Abdulagatov and Kazakov and Frenkel - 2011 - A new method for evaluation of UNIFAC interaction parameters
# + Some information extracted from below
NISTUFMG[1] = ('CH2', [1, 2, 3, 4], 'Alkyl chains')
NISTUFMG[2] = ('C=C', [5, 6, 7, 8, 9], 'Double bonded alkyl chains')
NISTUFMG[3] = ('ACH', [15, 16, 17], 'Aromatic carbon')
NISTUFMG[4] = ('ACCH2', [18, 19, 20, 21], 'Aromatic carbon plus alkyl chain')
NISTUFMG[5] = ('OH', [34, 204, 205], 'Alcohols')
NISTUFMG[6] = ('CH3OH', [35], 'Methanol')
NISTUFMG[7] = ('H2O', [36], 'Water')
NISTUFMG[8] = ('ACOH', [37], 'Phenolic –OH groups ')
NISTUFMG[9] = ('CH2CO', [42, 43, 44, 45], 'Ketones')
NISTUFMG[10] = ('CHO', [48], 'Aldehydes')
NISTUFMG[11] = ('CCOO', [51, 52, 53, 54], 'Esters')
NISTUFMG[12] = ('HCOO', [55], 'Formates')
NISTUFMG[13] = ('CH2O', [59, 60, 61, 62, 63], 'Ethers')
NISTUFMG[14] = ('CNH2', [66, 67, 68, 69], 'Amines with 1-alkyl group')
NISTUFMG[15] = ('(C)2NH', [71, 72, 73], 'Amines with 2-alkyl groups')
NISTUFMG[16] = ('(C)3N', [74, 75], 'Amines with 3-alkyl groups')
NISTUFMG[17] = ('ACNH2', [79, 80, 81], 'Anilines')
NISTUFMG[18] = ('PYRIDINE', [76, 77, 78], 'Pyridines')
NISTUFMG[19] = ('CCN', [85, 86, 87, 88], 'Nitriles')
NISTUFMG[20] = ('COOH', [94, 95], 'Acids')
NISTUFMG[21] = ('CCl', [99, 100, 101], 'Chlorocarbons')
NISTUFMG[22] = ('CCl2', [102, 103, 104], 'Dichlorocarbons')
NISTUFMG[23] = ('CCl3', [105, 106], 'Trichlorocarbons')
NISTUFMG[24] = ('CCl4', [107], 'Tetrachlorocarbons')
NISTUFMG[25] = ('ACCl', [109], 'Chloroaromatics')
NISTUFMG[26] = ('CNO2', [132, 133, 134, 135], 'Nitro alkanes')
NISTUFMG[27] = ('ACNO2', [136], 'Nitroaromatics')
NISTUFMG[28] = ('CS2', [146], 'Carbon disulfide')
NISTUFMG[29] = ('CH3SH', [138, 139, 140, 141], 'Mercaptans')
NISTUFMG[30] = ('FURFURAL', [50], 'Furfural')
NISTUFMG[31] = ('DOH', [38], 'Ethylene Glycol')
NISTUFMG[32] = ('I', [128], 'Iodides')
NISTUFMG[33] = ('BR', [130], 'Bromides')
NISTUFMG[34] = ('C≡C', [13, 14], 'Triplebonded alkyl chains')
NISTUFMG[35] = ('DMSO', [153], 'Dimethylsulfoxide')
NISTUFMG[36] = ('ACRY', [90], 'Acrylic')
NISTUFMG[37] = ('ClC=C', [108], 'Chlorine attached to double bonded alkyl chain')
NISTUFMG[38] = ('ACF', [118], 'Fluoroaromatics')
NISTUFMG[39] = ('DMF', [161, 162, 163, 164, 165], 'Amides')
NISTUFMG[40] = ('CF2', [111, 112, 113, 114, 115, 116, 117], 'Fluorines')
NISTUFMG[41] = ('COO', [58], 'Esters')
NISTUFMG[42] = ('SiH2', [197, 198, 199, 200], 'Silanes')
NISTUFMG[43] = ('SiO', [201, 202, 203], 'Siloxanes')
NISTUFMG[44] = ('NMP', [195], 'N-Methyl-2-pyrrolidone')
NISTUFMG[45] = ('CClF', [120, 121, 122, 123, 124, 125, 126, 127], 'Chloro-Fluorides')
NISTUFMG[46] = ('CONCH2', [166, 167, 168, 169], 'Amides')
NISTUFMG[47] = ('OCCOH', [39, 40, 41], 'Oxygenated Alcohols')
NISTUFMG[48] = ('CH2S', [142, 143, 144, 145], 'Sulfides')
NISTUFMG[49] = ('MORPHOLIN', [196], 'Morpholine')
NISTUFMG[50] = ('THIOPHENE', [147, 148, 149], 'Thiophene')
NISTUFMG[51] = ('CH2(cy)', [27, 28, 29], 'Cyclic hydrocarbon chains')
NISTUFMG[52] = ('C=C(cy)', [30, 31, 32], 'Cyclic unsaturated hydrocarbon chains')
# Added

NISTUFSG = {}
NISTUFSG[1] = UNIFAC_subgroup('CH3', 1, 'CH2', 0.6325, 1.0608)
NISTUFSG[2] = UNIFAC_subgroup('CH2', 1, 'CH2', 0.6325, 0.7081)
NISTUFSG[3] = UNIFAC_subgroup('CH', 1, 'CH2', 0.6325, 0.3554)
NISTUFSG[4] = UNIFAC_subgroup('C', 1, 'CH2', 0.6325, 0)
NISTUFSG[5] = UNIFAC_subgroup('CH2=CH', 2, 'C=C', 1.2832, 1.6016)
NISTUFSG[6] = UNIFAC_subgroup('CH=CH', 2, 'C=C', 1.2832, 1.2489)
NISTUFSG[7] = UNIFAC_subgroup('CH2=C', 2, 'C=C', 1.2832, 1.2489)
NISTUFSG[8] = UNIFAC_subgroup('CH=C', 2, 'C=C', 1.2832, 0.8962)
NISTUFSG[70] = UNIFAC_subgroup('C=C', 2, 'C=C', 1.2832, 0.4582)
NISTUFSG[9] = UNIFAC_subgroup('ACH', 3, 'ACH', 0.3763, 0.4321)
NISTUFSG[10] = UNIFAC_subgroup('AC', 3, 'ACH', 0.3763, 0.2113)
NISTUFSG[11] = UNIFAC_subgroup('ACCH3', 4, 'ACCH2', 0.91, 0.949)
NISTUFSG[12] = UNIFAC_subgroup('ACCH2', 4, 'ACCH2', 0.91, 0.7962)
NISTUFSG[13] = UNIFAC_subgroup('ACCH', 4, 'ACCH2', 0.91, 0.3769)
NISTUFSG[195] = UNIFAC_subgroup('ACC', 4, 'ACCH2', 0.5847, 0.12)
NISTUFSG[14] = UNIFAC_subgroup('OH prim', 5, 'OH', 1.2302, 0.8927)
NISTUFSG[81] = UNIFAC_subgroup('OH sec', 5, 'OH', 1.063, 0.8663)
NISTUFSG[82] = UNIFAC_subgroup('OH tert', 5, 'OH', 0.6895, 0.8345)
NISTUFSG[15] = UNIFAC_subgroup('CH3OH', 6, 'CH3OH', 0.8585, 0.9938)
NISTUFSG[16] = UNIFAC_subgroup('H2O', 7, 'H2O', 1.7334, 2.4561)
NISTUFSG[17] = UNIFAC_subgroup('ACOH', 8, 'ACOH', 1.08, 0.975)
NISTUFSG[18] = UNIFAC_subgroup('CH3CO', 9, 'CH2CO', 1.7048, 1.67)
NISTUFSG[19] = UNIFAC_subgroup('CH2CO', 9, 'CH2CO', 1.7048, 1.5542)
NISTUFSG[301] = UNIFAC_subgroup('CHCO', 9, 'CH2CO', 1.7048, 1.5542)
NISTUFSG[302] = UNIFAC_subgroup('CCO', 9, 'CH2CO', 1.7048, 1.5542)
NISTUFSG[20] = UNIFAC_subgroup('CHO', 10, 'CHO', 0.7173, 0.771)
NISTUFSG[308] = UNIFAC_subgroup('HCHO', 10, 'CHO', 0.7173, 0.771)
NISTUFSG[21] = UNIFAC_subgroup('CH3COO', 11, 'CCOO', 1.27, 1.6286)
NISTUFSG[22] = UNIFAC_subgroup('CH2COO', 11, 'CCOO', 1.27, 1.4228)
NISTUFSG[23] = UNIFAC_subgroup('HCOO', 12, 'HCOO', 1.9, 1.8)
NISTUFSG[24] = UNIFAC_subgroup('CH3O', 13, 'CH2O', 1.1434, 1.6022)
NISTUFSG[25] = UNIFAC_subgroup('CH2O', 13, 'CH2O', 1.1434, 1.2495)
NISTUFSG[26] = UNIFAC_subgroup('CHO', 13, 'CH2O', 1.1434, 0.8968)
NISTUFSG[28] = UNIFAC_subgroup('CH3NH2', 14, 'CNH2', 1.6607, 1.6904)
NISTUFSG[29] = UNIFAC_subgroup('CH2NH2', 14, 'CNH2', 1.6607, 1.3377)
NISTUFSG[30] = UNIFAC_subgroup('CHNH2', 14, 'CNH2', 1.6607, 0.985)
NISTUFSG[85] = UNIFAC_subgroup('CNH2', 14, 'CNH2', 1.6607, 0.985)
NISTUFSG[31] = UNIFAC_subgroup('CH3NH', 15, 'CNH', 1.368, 1.4332)
NISTUFSG[32] = UNIFAC_subgroup('CH2NH', 15, 'CNH', 1.368, 1.0805)
NISTUFSG[33] = UNIFAC_subgroup('CHNH', 15, 'CNH', 1.368, 0.7278)
NISTUFSG[34] = UNIFAC_subgroup('CH3N', 16, '(C)3N', 1.0746, 1.176)
NISTUFSG[35] = UNIFAC_subgroup('CH2N', 16, '(C)3N', 1.0746, 0.824)
NISTUFSG[36] = UNIFAC_subgroup('ACNH2', 17, 'ACNH2', 1.1849, 0.8067)
NISTUFSG[306] = UNIFAC_subgroup('ACNH', 17, 'ACNH2', 1.1849, 0.732)
NISTUFSG[307] = UNIFAC_subgroup('ACN', 17, 'ACNH2', 1.1849, 0.61)
NISTUFSG[37] = UNIFAC_subgroup('AC2H2N', 18, 'Pyridine', 1.4578, 0.9022)
NISTUFSG[38] = UNIFAC_subgroup('AC2HN', 18, 'Pyridine', 1.2393, 0.633)
NISTUFSG[39] = UNIFAC_subgroup('AC2N', 18, 'Pyridine', 1.0731, 0.353)
NISTUFSG[196] = UNIFAC_subgroup('AC2H2NH', 94, 'Pyrrole', 1.325, 0.752)
NISTUFSG[197] = UNIFAC_subgroup('AC2HNH', 94, 'Pyrrole', 1.0976, 0.44)
NISTUFSG[198] = UNIFAC_subgroup('AC2NH', 94, 'Pyrrole', 0.8701, 0.212)
NISTUFSG[40] = UNIFAC_subgroup('CH3CN', 19, 'CCN', 1.5575, 1.5193)
NISTUFSG[41] = UNIFAC_subgroup('CH2CN', 19, 'CCN', 1.5575, 1.1666)
NISTUFSG[303] = UNIFAC_subgroup('CHCN', 19, 'CCN', 1.5575, 1.1666)
NISTUFSG[304] = UNIFAC_subgroup('CCN', 19, 'CCN', 1.5575, 1.1666)
NISTUFSG[42] = UNIFAC_subgroup('COOH', 20, 'COOH', 0.8, 0.9215)
NISTUFSG[44] = UNIFAC_subgroup('CH2Cl', 21, 'CCl', 0.9919, 1.3654)
NISTUFSG[45] = UNIFAC_subgroup('CHCl', 21, 'CCl', 0.9919, 1.0127)
NISTUFSG[46] = UNIFAC_subgroup('CCl', 21, 'CCl', 0.9919, 0.66)
NISTUFSG[47] = UNIFAC_subgroup('CH2Cl2', 22, 'CCl2', 1.8, 2.5)
NISTUFSG[48] = UNIFAC_subgroup('CHCl2', 22, 'CCl2', 1.8, 2.1473)
NISTUFSG[49] = UNIFAC_subgroup('CCl2', 22, 'CCl2', 1.8, 1.7946)
NISTUFSG[51] = UNIFAC_subgroup('CCl3', 23, 'CCl3', 2.65, 2.3778)
NISTUFSG[52] = UNIFAC_subgroup('CCl4', 52, 'CCl4', 2.618, 3.1863)
NISTUFSG[53] = UNIFAC_subgroup('ACCl', 53, 'ACCl', 0.5365, 0.3177)
NISTUFSG[54] = UNIFAC_subgroup('CH3NO2', 26, 'CNO2', 2.644, 2.5)
NISTUFSG[55] = UNIFAC_subgroup('CH2NO2', 26, 'CNO2', 2.5, 2.304)
NISTUFSG[56] = UNIFAC_subgroup('CHNO2', 26, 'CNO2', 2.887, 2.241)
NISTUFSG[305] = UNIFAC_subgroup('CNO2', 26, 'CNO2', 2.887, 2.241)
NISTUFSG[57] = UNIFAC_subgroup('ACNO2', 27, 'ACNO2', 0.4656, 0.3589)
NISTUFSG[58] = UNIFAC_subgroup('CS2', 28, 'CS2', 1.24, 1.068)
NISTUFSG[59] = UNIFAC_subgroup('CH3SH', 29, 'CH2SH', 1.289, 1.762)
NISTUFSG[60] = UNIFAC_subgroup('CH2SH', 29, 'CH2SH', 1.535, 1.316)
NISTUFSG[192] = UNIFAC_subgroup('CHSH', 29, 'CH2SH', 1.4232, 1.21)
NISTUFSG[193] = UNIFAC_subgroup('CSH', 29, 'CH2SH', 1.1958, 1.1401)
NISTUFSG[194] = UNIFAC_subgroup('ACSH', 29, 'CH2SH', 1.2887, 1.2)
NISTUFSG[61] = UNIFAC_subgroup('Furfural', 30, 'Furfural', 1.299, 1.289)
NISTUFSG[62] = UNIFAC_subgroup('CH2(OH)-CH2(OH)', 31, 'DOH', 3.7374, 3.2016)
NISTUFSG[205] = UNIFAC_subgroup('-CH(OH)-CH2(OH)', 31, 'DOH', 3.5642, 2.8225)
NISTUFSG[206] = UNIFAC_subgroup('-CH(OH)-CH(OH)-', 31, 'DOH', 3.391, 2.4434)
NISTUFSG[207] = UNIFAC_subgroup('>C(OH)-CH2(OH)', 31, 'DOH', 3.1847, 2.1144)
NISTUFSG[208] = UNIFAC_subgroup('>C(OH)-CH(OH)-', 31, 'DOH', 3.0175, 2.0562)
NISTUFSG[209] = UNIFAC_subgroup('>C(OH)-C(OH)<', 31, 'DOH', 2.644, 1.669)
NISTUFSG[63] = UNIFAC_subgroup('I', 32, 'I', 1.076, 0.9169)
NISTUFSG[64] = UNIFAC_subgroup('Br', 33, 'Br', 1.209, 1.4)
NISTUFSG[65] = UNIFAC_subgroup('CH#C', 34, 'C#C', 0.9214, 1.3)
NISTUFSG[66] = UNIFAC_subgroup('C#C', 34, 'C#C', 1.303, 1.132)
NISTUFSG[67] = UNIFAC_subgroup('DMSO', 35, 'DMSO', 3.6, 2.692)
NISTUFSG[68] = UNIFAC_subgroup('Acrylonitrile', 36, 'Acrylonitrile', 1, 0.92)
NISTUFSG[69] = UNIFAC_subgroup('Cl-(C=C)', 37, 'Cl-(C=C)', 0.5229, 0.7391)
NISTUFSG[71] = UNIFAC_subgroup('ACF', 38, 'ACF', 0.8814, 0.7269)
NISTUFSG[72] = UNIFAC_subgroup('DMF', 39, 'DMF', 2, 2.093)
NISTUFSG[73] = UNIFAC_subgroup('HCON(CH2)2', 39, 'DMF', 2.381, 1.522)
NISTUFSG[74] = UNIFAC_subgroup('CF3', 40, 'CF2', 2.7489, 2.7769)
NISTUFSG[75] = UNIFAC_subgroup('CF2', 40, 'CF2', 1.4778, 1.4738)
NISTUFSG[76] = UNIFAC_subgroup('CF', 40, 'CF2', 0.8215, 0.5135)
NISTUFSG[77] = UNIFAC_subgroup('COO', 41, 'COO', 1.6, 0.9)
NISTUFSG[78] = UNIFAC_subgroup('c-CH2', 42, 'c-CH2', 0.7136, 0.8635)
NISTUFSG[79] = UNIFAC_subgroup('c-CH', 42, 'c-CH2', 0.3479, 0.1071)
NISTUFSG[80] = UNIFAC_subgroup('c-C', 42, 'c-CH2', 0.347, 0)
NISTUFSG[27] = UNIFAC_subgroup('CH2-O-CH2', 43, 'c-CH2O', 1.7023, 1.8784)
NISTUFSG[83] = UNIFAC_subgroup('CH2-O-[CH2-O]1/2', 43, 'c-CH2O', 1.4046, 1.4)
NISTUFSG[84] = UNIFAC_subgroup('[O-CH2]1/2-O-[CH2-O]1/2', 43, 'c-CH2O', 1.0413, 1.0116)
NISTUFSG[43] = UNIFAC_subgroup('HCOOH', 44, 'HCOOH', 0.8, 1.2742)
NISTUFSG[50] = UNIFAC_subgroup('CHCl3', 45, 'CHCl3', 2.45, 2.8912)
NISTUFSG[86] = UNIFAC_subgroup('c-CON-CH3', 46, 'c-CONC', 3.981, 3.2)
NISTUFSG[87] = UNIFAC_subgroup('c-CON-CH2', 46, 'c-CONC', 3.7543, 2.892)
NISTUFSG[88] = UNIFAC_subgroup('c-CON-CH', 46, 'c-CONC', 3.5268, 2.58)
NISTUFSG[89] = UNIFAC_subgroup('c-CON-C', 46, 'c-CONC', 3.2994, 2.352)
NISTUFSG[92] = UNIFAC_subgroup('CONHCH3', 47, 'CONR', 1.5, 1.08)
NISTUFSG[100] = UNIFAC_subgroup('CONHCH2', 47, 'CONR', 1.5, 1.08)
NISTUFSG[101] = UNIFAC_subgroup('CON(CH3)2', 48, 'CONR2', 2.4748, 1.9643)
NISTUFSG[102] = UNIFAC_subgroup('CON(CH3)CH2', 48, 'CONR2', 2.2739, 1.5754)
NISTUFSG[103] = UNIFAC_subgroup('CON(CH2)2', 48, 'CONR2', 2.0767, 1.1866)
NISTUFSG[93] = UNIFAC_subgroup('HCONHCH3', 49, 'HCONR', 2.4617, 2.192)
NISTUFSG[94] = UNIFAC_subgroup('HCONHCH2', 49, 'HCONR', 2.4617, 1.842)
NISTUFSG[116] = UNIFAC_subgroup('ACCN', 50, 'ACCN', 1.2815, 0.96)
NISTUFSG[117] = UNIFAC_subgroup('CH3NCO', 51, 'NCO', 1.9578, 1.58)
NISTUFSG[118] = UNIFAC_subgroup('CH2NCO', 51, 'NCO', 1.731, 1.272)
NISTUFSG[119] = UNIFAC_subgroup('CHNCO', 51, 'NCO', 1.5036, 0.96)
NISTUFSG[120] = UNIFAC_subgroup('ACNCO', 51, 'NCO', 1.4219, 0.852)
NISTUFSG[104] = UNIFAC_subgroup('AC2H2S', 52, 'ACS', 1.7943, 1.34)
NISTUFSG[105] = UNIFAC_subgroup('AC2HS', 52, 'ACS', 1.6282, 1.06)
NISTUFSG[106] = UNIFAC_subgroup('AC2S', 52, 'ACS', 1.4621, 0.78)
NISTUFSG[107] = UNIFAC_subgroup('H2COCH', 53, 'Epoxy', 1.3601, 1.8031)
NISTUFSG[109] = UNIFAC_subgroup('HCOCH', 53, 'Epoxy', 0.9104, 0.6538)
NISTUFSG[121] = UNIFAC_subgroup('COOCO', 54, 'Anhydride', 1.7732, 1.52)
NISTUFSG[112] = UNIFAC_subgroup('(CH3O)2CO', 55, 'Carbonate', 3.0613, 2.816)
NISTUFSG[113] = UNIFAC_subgroup('(CH2O)2CO', 55, 'Carbonate', 2.6078, 2.2)
NISTUFSG[114] = UNIFAC_subgroup('(CH3O)COOCH2', 55, 'Carbonate', 2.8214, 2.508)
NISTUFSG[199] = UNIFAC_subgroup('(ACO)COOCH2', 55, 'Carbonate', 2.2854, 1.78)
NISTUFSG[200] = UNIFAC_subgroup('(ACO)CO(OAC)', 55, 'Carbonate', 1.9895, 1.36)
NISTUFSG[110] = UNIFAC_subgroup('CH2SuCH2', 56, 'Sulfone', 2.687, 2.12)
NISTUFSG[111] = UNIFAC_subgroup('CH2SuCH ', 56, 'Sulfone', 2.46, 1.808)
NISTUFSG[122] = UNIFAC_subgroup('ACSO2', 56, 'Sulfone', 1.7034, 1.16)
NISTUFSG[123] = UNIFAC_subgroup('ACCHO', 57, 'ACCHO', 1.3632, 1.068)
NISTUFSG[124] = UNIFAC_subgroup('ACCOOH', 58, 'ACCOOH', 1.6664, 1.344)
NISTUFSG[127] = UNIFAC_subgroup('AC-O-CO-CH3 ', 59, 'AC-O-CO', 2.2815, 1.848)
NISTUFSG[128] = UNIFAC_subgroup('AC-O-CO-CH2', 59, 'AC-O-CO', 2.0547, 1.54)
NISTUFSG[129] = UNIFAC_subgroup('AC-O-CO-CH', 59, 'AC-O-CO', 1.8273, 1.228)
NISTUFSG[130] = UNIFAC_subgroup('AC-O-CO-C', 59, 'AC-O-CO', 1.5999, 1)
NISTUFSG[131] = UNIFAC_subgroup('-O-CH2-CH2-OH', 60, 'OCCOH', 2.1226, 1.904)
NISTUFSG[132] = UNIFAC_subgroup('-O-CH-CH2-OH', 60, 'OCCOH', 1.8952, 1.592)
NISTUFSG[133] = UNIFAC_subgroup('-O-CH2-CH-OH', 60, 'OCCOH', 1.8952, 1.592)
NISTUFSG[134] = UNIFAC_subgroup('CH3-S-', 61, 'CH2S', 1.6131, 1.368)
NISTUFSG[135] = UNIFAC_subgroup('-CH2-S-', 61, 'CH2S', 1.3863, 1.06)
NISTUFSG[136] = UNIFAC_subgroup('>CH-S-', 61, 'CH2S', 1.1589, 0.748)
NISTUFSG[137] = UNIFAC_subgroup('->C-S-', 61, 'CH2S', 0.9314, 0.52)
NISTUFSG[187] = UNIFAC_subgroup('ACS', 61, 'CH2S', 1.0771, 0.64)
NISTUFSG[125] = UNIFAC_subgroup('c-CO-NH', 62, 'Lactam', 1.3039, 1.036)
NISTUFSG[126] = UNIFAC_subgroup('c-CO-O', 63, 'Lactone', 1.0152, 0.88)
NISTUFSG[138] = UNIFAC_subgroup('CH3O-(O)', 64, 'Peroxide', 1.3889, 1.328)
NISTUFSG[139] = UNIFAC_subgroup('CH2O-(O)', 64, 'Peroxide', 1.1622, 1.02)
NISTUFSG[140] = UNIFAC_subgroup('CHO-(O)', 64, 'Peroxide', 0.9347, 0.708)
NISTUFSG[141] = UNIFAC_subgroup('CO-(O)', 64, 'Peroxide', 1.0152, 0.88)
NISTUFSG[142] = UNIFAC_subgroup('ACO-(O)', 64, 'Peroxide', 0.853, 0.6)
NISTUFSG[143] = UNIFAC_subgroup('CFH', 65, 'CFH', 0.5966, 0.44)
NISTUFSG[144] = UNIFAC_subgroup('CFCl', 66, 'CFCl', 1.4034, 1.168)
NISTUFSG[145] = UNIFAC_subgroup('CFCl2', 67, 'CFCl2', 2.2103, 1.896)
NISTUFSG[146] = UNIFAC_subgroup('CF2H', 68, 'CF2H', 0.9736, 0.88)
NISTUFSG[147] = UNIFAC_subgroup('CF2ClH', 69, 'CF2ClH', 1.7396, 1.6)
NISTUFSG[148] = UNIFAC_subgroup('CF2Cl2', 70, 'CF2Cl2', 2.5873, 2.336)
NISTUFSG[149] = UNIFAC_subgroup('CF3H', 71, 'CF3H', 1.3507, 1.32)
NISTUFSG[150] = UNIFAC_subgroup('CF3Cl', 72, 'CF3Cl', 2.1575, 2.048)
NISTUFSG[151] = UNIFAC_subgroup('CF4', 73, 'CF4', 1.7278, 1.76)
NISTUFSG[152] = UNIFAC_subgroup('C(O)2', 74, 'Acetal', 0.7073, 0.48)
NISTUFSG[186] = UNIFAC_subgroup('CH(O)2', 74, 'Acetal', 0.9347, 0.708)
NISTUFSG[309] = UNIFAC_subgroup('CH2(O)2', 74, 'Acetal', 0.9347, 0.708)
NISTUFSG[153] = UNIFAC_subgroup('ACN(CH3)2', 75, 'ACNR2', 2.4529, 1.908)
NISTUFSG[154] = UNIFAC_subgroup('ACN(CH3)CH2', 75, 'ACNR2', 2.2261, 1.6)
NISTUFSG[155] = UNIFAC_subgroup('ACN(CH2)2', 75, 'ACNR2', 1.9993, 1.292)
NISTUFSG[156] = UNIFAC_subgroup('ACNHCH3', 76, 'ACNR', 1.7989, 1.364)
NISTUFSG[157] = UNIFAC_subgroup('ACNHCH2', 76, 'ACNR', 1.5722, 1.056)
NISTUFSG[158] = UNIFAC_subgroup('ACNHCH', 76, 'ACNR', 1.3448, 0.744)
NISTUFSG[159] = UNIFAC_subgroup('AC2H2O', 77, 'Furan', 1.3065, 1.04)
NISTUFSG[160] = UNIFAC_subgroup('AC2HO', 77, 'Furan', 1.1404, 0.76)
NISTUFSG[161] = UNIFAC_subgroup('AC2O', 77, 'Furan', 0.9743, 0.48)
NISTUFSG[188] = UNIFAC_subgroup('c-CH2-NH', 78, 'c-CNH', 1.207, 0.936)
NISTUFSG[162] = UNIFAC_subgroup('c-CH-NH', 78, 'c-CNH', 0.9796, 0.624)
NISTUFSG[163] = UNIFAC_subgroup('c-C-NH', 78, 'c-CNH', 0.7521, 0.396)
NISTUFSG[189] = UNIFAC_subgroup('c-CH2-NCH3', 79, 'c-CNR', 1.8609, 1.48)
NISTUFSG[190] = UNIFAC_subgroup('c-CH2-NCH2', 79, 'c-CNR', 1.6341, 1.172)
NISTUFSG[191] = UNIFAC_subgroup('c-CH2-NCH', 79, 'c-CNR', 1.4067, 0.86)
NISTUFSG[164] = UNIFAC_subgroup('c-CH-NCH3', 79, 'c-CNR', 1.6335, 1.168)
NISTUFSG[165] = UNIFAC_subgroup('c-CH-NCH2', 79, 'c-CNR', 1.4067, 0.86)
NISTUFSG[166] = UNIFAC_subgroup('c-CH-NCH', 79, 'c-CNR', 1.1793, 0.548)
NISTUFSG[170] = UNIFAC_subgroup('SiH3-', 80, 'SiH', 1.6035, 1.263)
NISTUFSG[171] = UNIFAC_subgroup('-SiH2-', 80, 'SiH', 1.4443, 1.006)
NISTUFSG[172] = UNIFAC_subgroup('>SiH-', 80, 'SiH', 1.2853, 0.749)
NISTUFSG[173] = UNIFAC_subgroup('>Si<', 80, 'SiH', 1.047, 0.41)
NISTUFSG[174] = UNIFAC_subgroup('-SiH2-O-', 81, 'SiO', 1.4838, 1.062)
NISTUFSG[175] = UNIFAC_subgroup('>SiH-O-', 81, 'SiO', 1.303, 0.764)
NISTUFSG[176] = UNIFAC_subgroup('->Si-O-', 81, 'SiO', 1.1044, 0.466)
NISTUFSG[309] = UNIFAC_subgroup('CH=NOH', 82, 'Oxime', 1.499, 1.46)
NISTUFSG[177] = UNIFAC_subgroup('C=NOH', 82, 'Oxime', 1.499, 1.46)
NISTUFSG[178] = UNIFAC_subgroup('ACCO', 83, 'ACCO', 1.1365, 0.76)
NISTUFSG[179] = UNIFAC_subgroup('C2Cl4', 86, 'C2Cl4', 3.381, 3.5845)
NISTUFSG[180] = UNIFAC_subgroup('c-CHH2', 92, 'c-CHNH2', 1.2261, 1.096)
NISTUFSG[201] = UNIFAC_subgroup('c-CH=CH', 95, 'c-C=C', 1.0897, 0.832)
NISTUFSG[202] = UNIFAC_subgroup('c-CH=C', 95, 'c-C=C', 0.8616, 0.644)
NISTUFSG[203] = UNIFAC_subgroup('c-C=C', 95, 'c-C=C', 0.5498, 0.244)
NISTUFSG[204] = UNIFAC_subgroup('Glycerol', 96, 'Glycerol', 5.4209, 4.4227)


PSRKSG = {}
PSRKSG[1] = UNIFAC_subgroup('CH3', 1, 'CH2', 0.9011, 0.8480)
PSRKSG[2] = UNIFAC_subgroup('CH2', 1, 'CH2', 0.6744, 0.5400)
PSRKSG[3] = UNIFAC_subgroup('CH', 1, 'CH2', 0.4469, 0.2280)
PSRKSG[4] = UNIFAC_subgroup('C', 1, 'CH2', 0.2195, 0.0000)
PSRKSG[5] = UNIFAC_subgroup('CH2=CH', 2, 'C=C', 1.3454, 1.1760)
PSRKSG[6] = UNIFAC_subgroup('CH=CH', 2, 'C=C', 1.1167, 0.8670)
PSRKSG[7] = UNIFAC_subgroup('CH2=C', 2, 'C=C', 1.1173, 0.9880)
PSRKSG[8] = UNIFAC_subgroup('CH=C', 2, 'C=C', 0.8886, 0.6760)
PSRKSG[9] = UNIFAC_subgroup('ACH', 3, 'ACH', 0.5313, 0.4000)
PSRKSG[10] = UNIFAC_subgroup('AC', 3, 'ACH', 0.3652, 0.1200)
PSRKSG[11] = UNIFAC_subgroup('ACCH3', 4, 'ACCH2', 1.2663, 0.9680)
PSRKSG[12] = UNIFAC_subgroup('ACCH2', 4, 'ACCH2', 1.0396, 0.6600)
PSRKSG[13] = UNIFAC_subgroup('ACCH', 4, 'ACCH2', 0.8121, 0.3480)
PSRKSG[14] = UNIFAC_subgroup('OH', 5, 'OH', 1.0000, 1.2000)
PSRKSG[15] = UNIFAC_subgroup('CH3OH', 6, 'CH3OH', 1.4311, 1.4320)
PSRKSG[16] = UNIFAC_subgroup('H2O', 7, 'H2O', 0.9200, 1.4000)
PSRKSG[17] = UNIFAC_subgroup('ACOH', 8, 'ACOH', 0.8952, 0.6800)
PSRKSG[18] = UNIFAC_subgroup('CH3CO', 9, 'CH2CO', 1.6724, 1.4880)
PSRKSG[19] = UNIFAC_subgroup('CH2CO', 9, 'CH2CO', 1.4457, 1.1800)
PSRKSG[20] = UNIFAC_subgroup('CHO', 10, 'CHO', 0.9980, 0.9480)
PSRKSG[21] = UNIFAC_subgroup('CH3COO', 11, 'CCOO', 1.9031, 1.7280)
PSRKSG[22] = UNIFAC_subgroup('CH2COO', 11, 'CCOO', 1.6764, 1.4200)
PSRKSG[23] = UNIFAC_subgroup('HCOO', 12, 'HCOO', 1.2420, 1.1880)
PSRKSG[24] = UNIFAC_subgroup('CH3O', 13, 'CH2O', 1.1450, 1.0880)
PSRKSG[25] = UNIFAC_subgroup('CH2O', 13, 'CH2O', 0.9183, 0.7800)
PSRKSG[26] = UNIFAC_subgroup('CHO', 13, 'CH2O', 0.6908, 0.4680)
PSRKSG[27] = UNIFAC_subgroup('THF', 13, 'CH2O', 0.9183, 1.1000)
PSRKSG[28] = UNIFAC_subgroup('CH3NH2', 14, 'CNH2', 1.5959, 1.5440)
PSRKSG[29] = UNIFAC_subgroup('CH2NH2', 14, 'CNH2', 1.3692, 1.2360)
PSRKSG[30] = UNIFAC_subgroup('CHNH2', 14, 'CNH2', 1.1417, 0.9240)
PSRKSG[31] = UNIFAC_subgroup('CH3NH', 15, 'CNH', 1.4337, 1.2440)
PSRKSG[32] = UNIFAC_subgroup('CH2NH', 15, 'CNH', 1.2070, 0.9360)
PSRKSG[33] = UNIFAC_subgroup('CHNH', 15, 'CNH', 0.9795, 0.6240)
PSRKSG[34] = UNIFAC_subgroup('CH3N', 16, '(C)3N', 1.1865, 0.9400)
PSRKSG[35] = UNIFAC_subgroup('CH2N', 16, '(C)3N', 0.9597, 0.6320)
PSRKSG[36] = UNIFAC_subgroup('ACNH2', 17, 'ACNH2', 1.0600, 0.8160)
PSRKSG[37] = UNIFAC_subgroup('C5H5N', 18, 'PYRIDINE', 2.9993, 2.1130)
PSRKSG[38] = UNIFAC_subgroup('C5H4N', 18, 'PYRIDINE', 2.8332, 1.8330)
PSRKSG[39] = UNIFAC_subgroup('C5H3N', 18, 'PYRIDINE', 2.6670, 1.5530)
PSRKSG[40] = UNIFAC_subgroup('CH3CN', 19, 'CCN', 1.8701, 1.7240)
PSRKSG[41] = UNIFAC_subgroup('CH2CN', 19, 'CCN', 1.6434, 1.4160)
PSRKSG[42] = UNIFAC_subgroup('COOH', 20, 'COOH', 1.3013, 1.2240)
PSRKSG[43] = UNIFAC_subgroup('HCOOH', 20, 'COOH', 1.5280, 1.5320)
PSRKSG[44] = UNIFAC_subgroup('CH2CL', 21, 'CCL', 1.4654, 1.2640)
PSRKSG[45] = UNIFAC_subgroup('CHCL', 21, 'CCL', 1.2380, 0.9520)
PSRKSG[46] = UNIFAC_subgroup('CCL', 21, 'CCL', 1.0106, 0.7240)
PSRKSG[47] = UNIFAC_subgroup('CH2CL2', 22, 'CCL2', 2.2564, 1.9880)
PSRKSG[48] = UNIFAC_subgroup('CHCL2', 22, 'CCL2', 2.0606, 1.6840)
PSRKSG[49] = UNIFAC_subgroup('CCL2', 22, 'CCL2', 1.8016, 1.4480)
PSRKSG[50] = UNIFAC_subgroup('CHCL3', 23, 'CCL3', 2.8700, 2.4100)
PSRKSG[51] = UNIFAC_subgroup('CCL3', 23, 'CCL3', 2.6401, 2.1840)
PSRKSG[52] = UNIFAC_subgroup('CCL4', 24, 'CCL4', 3.3900, 2.9100)
PSRKSG[53] = UNIFAC_subgroup('ACCL', 25, 'ACCL', 1.1562, 0.8440)
PSRKSG[54] = UNIFAC_subgroup('CH3NO2', 26, 'CNO2', 2.0086, 1.8680)
PSRKSG[55] = UNIFAC_subgroup('CH2NO2', 26, 'CNO2', 1.7818, 1.5600)
PSRKSG[56] = UNIFAC_subgroup('CHNO2', 26, 'CNO2', 1.5544, 1.2480)
PSRKSG[57] = UNIFAC_subgroup('ACNO2', 27, 'ACNO2', 1.4199, 1.1040)
PSRKSG[58] = UNIFAC_subgroup('CS2', 28, 'CS2', 2.0570, 1.6500)
PSRKSG[59] = UNIFAC_subgroup('CH3SH', 29, 'CH3SH', 1.8770, 1.6760)
PSRKSG[60] = UNIFAC_subgroup('CH2SH', 29, 'CH3SH', 1.6510, 1.3680)
PSRKSG[61] = UNIFAC_subgroup('FURFURAL', 30, 'FURFURAL', 3.1680, 2.4840)
PSRKSG[62] = UNIFAC_subgroup('DOH', 31, 'DOH', 2.4088, 2.2480)
PSRKSG[63] = UNIFAC_subgroup('I', 32, 'I', 1.2640, 0.9920)
PSRKSG[64] = UNIFAC_subgroup('BR', 33, 'BR', 0.9492, 0.8320)
PSRKSG[65] = UNIFAC_subgroup('CH=-C', 34, 'C=-C', 1.2920, 1.0880)
PSRKSG[66] = UNIFAC_subgroup('C=-C', 34, 'C=-C', 1.0613, 0.7840)
PSRKSG[67] = UNIFAC_subgroup('DMSO', 35, 'DMSO', 2.8266, 2.4720)
PSRKSG[68] = UNIFAC_subgroup('ACRY', 36, 'ACRY', 2.3144, 2.0520)
PSRKSG[69] = UNIFAC_subgroup('CL-(C=C)', 37, 'CLCC', 0.7910, 0.7240)
PSRKSG[70] = UNIFAC_subgroup('C=C', 2, 'C=C', 0.6605, 0.4850)
PSRKSG[71] = UNIFAC_subgroup('ACF', 38, 'ACF', 0.6948, 0.5240)
PSRKSG[72] = UNIFAC_subgroup('DMF', 39, 'DMF', 3.0856, 2.7360)
PSRKSG[73] = UNIFAC_subgroup('HCON(..', 39, 'DMF', 2.6322, 2.1200)
PSRKSG[74] = UNIFAC_subgroup('CF3', 40, 'CF2', 1.4060, 1.3800)
PSRKSG[75] = UNIFAC_subgroup('CF2', 40, 'CF2', 1.0105, 0.9200)
PSRKSG[76] = UNIFAC_subgroup('CF', 40, 'CF2', 0.6150, 0.4600)
PSRKSG[77] = UNIFAC_subgroup('COO', 41, 'COO', 1.3800, 1.2000)
PSRKSG[78] = UNIFAC_subgroup('SIH3', 42, 'SIH2', 1.6035, 1.2632)
PSRKSG[79] = UNIFAC_subgroup('SIH2', 42, 'SIH2', 1.4443, 1.0063)
PSRKSG[80] = UNIFAC_subgroup('SIH', 42, 'SIH2', 1.2853, 0.7494)
PSRKSG[81] = UNIFAC_subgroup('SI', 42, 'SIH2', 1.0470, 0.4099)
PSRKSG[82] = UNIFAC_subgroup('SIH2O', 43, 'SIO', 1.4838, 1.0621)
PSRKSG[83] = UNIFAC_subgroup('SIHO', 43, 'SIO', 1.3030, 0.7639)
PSRKSG[84] = UNIFAC_subgroup('SIO', 43, 'SIO', 1.1044, 0.4657)
PSRKSG[85] = UNIFAC_subgroup('NMP', 44, 'NMP', 3.9810, 3.2000)
PSRKSG[86] = UNIFAC_subgroup('CCL3F', 45, 'CCLF', 3.0356, 2.6440)
PSRKSG[87] = UNIFAC_subgroup('CCL2F', 45, 'CCLF', 2.2287, 1.9160)
PSRKSG[88] = UNIFAC_subgroup('HCCL2F', 45, 'CCLF', 2.4060, 2.1160)
PSRKSG[89] = UNIFAC_subgroup('HCCLF', 45, 'CCLF', 1.6493, 1.4160)
PSRKSG[90] = UNIFAC_subgroup('CCLF2', 45, 'CCLF', 1.8174, 1.6480)
PSRKSG[91] = UNIFAC_subgroup('HCCLF2', 45, 'CCLF', 1.9670, 1.8280)
PSRKSG[92] = UNIFAC_subgroup('CCLF3', 45, 'CCLF', 2.1721, 2.1000)
PSRKSG[93] = UNIFAC_subgroup('CCL2F2', 45, 'CCLF', 2.6243, 2.3760)
PSRKSG[94] = UNIFAC_subgroup('AMH2', 46, 'CON (AM)', 1.4515, 1.2480)
PSRKSG[95] = UNIFAC_subgroup('AMHCH3', 46, 'CON (AM)', 2.1905, 1.7960)
PSRKSG[96] = UNIFAC_subgroup('AMHCH2', 46, 'CON (AM)', 1.9637, 1.4880)
PSRKSG[97] = UNIFAC_subgroup('AM(CH3)2', 46, 'CON (AM)', 2.8589, 2.4280)
PSRKSG[98] = UNIFAC_subgroup('AMCH3CH2', 46, 'CON (AM)', 2.6322, 2.1200)
PSRKSG[99] = UNIFAC_subgroup('AM(CH2)2', 46, 'CON (AM)', 2.4054, 1.8120)
PSRKSG[100] = UNIFAC_subgroup('C2H5O2', 47, 'OCCOH', 2.1226, 1.9040)
PSRKSG[101] = UNIFAC_subgroup('C2H4O2', 47, 'OCCOH', 1.8952, 1.5920)
PSRKSG[102] = UNIFAC_subgroup('CH3S', 48, 'CH2S', 1.6130, 1.3680)
PSRKSG[103] = UNIFAC_subgroup('CH2S', 48, 'CH2S', 1.3863, 1.0600)
PSRKSG[104] = UNIFAC_subgroup('CHS', 48, 'CH2S', 1.1589, 0.7480)
PSRKSG[105] = UNIFAC_subgroup('MORPH', 49, 'MORPH', 3.4740, 2.7960)
PSRKSG[106] = UNIFAC_subgroup('C4H4S', 50, 'THIOPHEN', 2.8569, 2.1400)
PSRKSG[107] = UNIFAC_subgroup('C4H3S', 50, 'THIOPHEN', 2.6908, 1.8600)
PSRKSG[108] = UNIFAC_subgroup('C4H2S', 50, 'THIOPHEN', 2.5247, 1.5800)
PSRKSG[109] = UNIFAC_subgroup('H2C=CH2', 2, 'C=C', 1.3564, 1.3098)
PSRKSG[110] = UNIFAC_subgroup('CH=-CH', 34, 'C=-C', 0.7910, 0.7200)
PSRKSG[111] = UNIFAC_subgroup('NH3', 55, 'NH3', 0.8510, 0.7780)
PSRKSG[112] = UNIFAC_subgroup('CO', 63, 'CO', 0.7110, 0.8280)
PSRKSG[113] = UNIFAC_subgroup('H2', 62, 'H2', 0.4160, 0.5710)
PSRKSG[114] = UNIFAC_subgroup('H2S', 61, 'H2S', 1.2350, 1.2020)
PSRKSG[115] = UNIFAC_subgroup('N2', 60, 'N2', 0.8560, 0.9300)
PSRKSG[116] = UNIFAC_subgroup('AR', 59, 'AR', 1.1770, 1.1160)
PSRKSG[117] = UNIFAC_subgroup('CO2', 56, 'CO2', 1.3000, 0.9820)
PSRKSG[118] = UNIFAC_subgroup('CH4', 57, 'CH4', 1.1292, 1.1240)
PSRKSG[119] = UNIFAC_subgroup('O2', 58, 'O2', 0.7330, 0.8490)
PSRKSG[120] = UNIFAC_subgroup('D2', 62, 'H2', 0.3700, 0.5270)
PSRKSG[121] = UNIFAC_subgroup('SO2', 65, 'SO2', 1.3430, 1.1640)
PSRKSG[122] = UNIFAC_subgroup('NO', 66, 'NO', 0.7160, 0.6200)
PSRKSG[123] = UNIFAC_subgroup('N2O', 67, 'N2O', 0.9800, 0.8880)
PSRKSG[124] = UNIFAC_subgroup('SF6', 68, 'SF6', 2.3740, 2.0560)
PSRKSG[125] = UNIFAC_subgroup('HE', 69, 'HE', 0.8850, 0.9850)
PSRKSG[126] = UNIFAC_subgroup('NE', 70, 'NE', 0.8860, 0.9860)
PSRKSG[127] = UNIFAC_subgroup('KR', 71, 'KR', 1.1200, 1.1200)
PSRKSG[128] = UNIFAC_subgroup('XE', 72, 'XE', 1.1300, 1.1300)
PSRKSG[129] = UNIFAC_subgroup('HF', 73, 'HF', 1.0160, 1.2160)
PSRKSG[130] = UNIFAC_subgroup('HCL', 74, 'HCL', 1.0560, 1.2560)
PSRKSG[131] = UNIFAC_subgroup('HBR', 75, 'HBR', 1.0580, 1.2580)
PSRKSG[132] = UNIFAC_subgroup('HI', 76, 'HI', 1.3930, 1.2080)
PSRKSG[133] = UNIFAC_subgroup('COS', 77, 'COS', 1.6785, 1.3160)
PSRKSG[134] = UNIFAC_subgroup('CHSH', 29, 'CH3SH', 1.4250, 1.0600)
PSRKSG[135] = UNIFAC_subgroup('CSH', 29, 'CH3SH', 1.1990, 0.7520)
PSRKSG[136] = UNIFAC_subgroup('H2COCH', 51, 'EPOXY', 1.3652, 1.0080)
PSRKSG[137] = UNIFAC_subgroup('HCOCH', 51, 'EPOXY', 1.1378, 0.6960)
PSRKSG[138] = UNIFAC_subgroup('HCOC', 51, 'EPOXY', 0.9104, 0.4680)
PSRKSG[139] = UNIFAC_subgroup('H2COCH2', 51, 'EPOXY', 1.5926, 1.3200)
PSRKSG[140] = UNIFAC_subgroup('H2COC', 51, 'EPOXY', 1.1378, 0.7800)
PSRKSG[141] = UNIFAC_subgroup('COC', 51, 'EPOXY', 0.6829, 0.2400)
PSRKSG[142] = UNIFAC_subgroup('F2', 78, 'F2', 0.7500, 0.8800)
PSRKSG[143] = UNIFAC_subgroup('CL2', 79, 'CL2', 1.5300, 1.4400)
PSRKSG[144] = UNIFAC_subgroup('BR2', 80, 'BR2', 1.9000, 1.6600)
PSRKSG[145] = UNIFAC_subgroup('HCN', 81, 'HCN', 1.2000, 1.1900)
PSRKSG[146] = UNIFAC_subgroup('NO2', 82, 'NO2', 1.0000, 1.1000)
PSRKSG[147] = UNIFAC_subgroup('CF4', 83, 'CF4', 1.7800, 1.8200)
PSRKSG[148] = UNIFAC_subgroup('O3', 84, 'O3', 1.1000, 1.2700)
PSRKSG[149] = UNIFAC_subgroup('CLNO', 85, 'CLNO', 1.4800, 1.3400)
PSRKSG[152] = UNIFAC_subgroup('CNH2', 14, 'CNH2', 0.9147, 0.6140)

PSRKMG = {1: ("CH2", [1, 2, 3, 4]),
2: ("C=C", [5, 6, 7, 8, 70, 109]),
3: ("ACH", [9, 10]),
4: ("ACCH2", [11, 12, 13]),
5: ("OH", [14]),
6: ("CH3OH", [15]),
7: ("H2O", [16]),
8: ("ACOH", [17]),
9: ("CH2CO", [18, 19]),
10: ("CHO", [20]),
11: ("CCOO", [21, 22]),
12: ("HCOO", [23]),
13: ("CH2O", [24, 25, 26, 27]),
14: ("CNH2", [28, 29, 30, 152]),
15: ("CNH", [31, 32, 33]),
16: ("(C)3N", [34, 35]),
17: ("ACNH2", [36]),
18: ("PYRIDINE", [37, 38, 39]),
19: ("CCN", [40, 41]),
20: ("COOH", [42, 43]),
21: ("CCL", [44, 45, 46]),
22: ("CCL2", [47, 48, 49]),
23: ("CCL3", [50, 51]),
24: ("CCL4", [52]),
25: ("ACCL", [53]),
26: ("CNO2", [54, 55, 56]),
27: ("ACNO2", [57]),
28: ("CS2", [58]),
29: ("CH3SH", [59, 60, 134, 135]),
30: ("FURFURAL", [61]),
31: ("DOH", [62]),
32: ("I", [63]),
33: ("BR", [64]),
34: ("C=-C", [65, 66, 110]),
35: ("DMSO", [67]),
36: ("ACRY", [68]),
37: ("CLCC", [69]),
38: ("ACF", [71]),
39: ("DMF", [72, 73]),
40: ("CF2", [74, 75, 76]),
41: ("COO", [77]),
42: ("SIH2", [78, 79, 80, 81]),
43: ("SIO", [82, 83, 84]),
44: ("NMP", [85]),
45: ("CCLF", [86, 87, 88, 89, 90, 91, 92, 93]),
46: ("CON (AM)", [94, 95, 96, 97, 98, 99]),
47: ("OCCOH", [100, 101]),
48: ("CH2S", [102, 103, 104]),
49: ("MORPH", [105]),
50: ("THIOPHEN", [106, 107, 108]),
51: ("EPOXY", [136, 137, 138, 139, 140, 141]),
55: ("NH3", [111]),
56: ("CO2", [117]),
57: ("CH4", [118]),
58: ("O2", [119]),
59: ("AR", [116]),
60: ("N2", [115]),
61: ("H2S", [114]),
62: ("H2", [113, 120]),
63: ("CO", [112]),
65: ("SO2", [121]),
66: ("NO", [122]),
67: ("N2O", [123]),
68: ("SF6", [124]),
69: ("HE", [125]),
70: ("NE", [126]),
71: ("KR", [127]),
72: ("XE", [128]),
73: ("HF", [129]),
74: ("HCL", [130]),
75: ("HBR", [131]),
76: ("HI", [132]),
77: ("COS", [133]),
78: ("F2", [142]),
79: ("CL2", [143]),
80: ("BR2", [144]),
81: ("HCN", [145]),
82: ("NO2", [146]),
83: ("CF4", [147]),
84: ("O3", [148]),
85: ("CLNO", [149]),
}

LLEUFSG = {}
# LLEUFSG[subgroup ID] = (subgroup formula, main group ID, subgroup R, subgroup Q)
LLEUFSG[1] = UNIFAC_subgroup('CH3', 1, 'CH2', 0.9011, 0.848, smarts=UFSG[1].smarts)
LLEUFSG[2] = UNIFAC_subgroup('CH2', 1, 'CH2', 0.6744, 0.54, smarts=UFSG[2].smarts)
LLEUFSG[3] = UNIFAC_subgroup('CH', 1, 'CH2', 0.4469, 0.228, smarts=UFSG[3].smarts)
LLEUFSG[4] = UNIFAC_subgroup('C', 1, 'CH2', 0.2195, 0, smarts=UFSG[4].smarts)

LLEUFSG[5] = UNIFAC_subgroup('CH2=CH', 2, 'C=C', 1.3454, 1.176, smarts=UFSG[5].smarts)
LLEUFSG[6] = UNIFAC_subgroup('CH=CH', 2, 'C=C', 1.1167, 0.867, smarts=UFSG[6].smarts)
LLEUFSG[7] = UNIFAC_subgroup('CH=C', 2, 'C=C', 0.8886, 0.676, smarts=UFSG[8].smarts) # 7, 8 diff order than UFSG
LLEUFSG[8] = UNIFAC_subgroup('CH2=C', 2, 'C=C', 1.1173, 0.988, smarts=UFSG[7].smarts)

LLEUFSG[9] = UNIFAC_subgroup('ACH', 3, 'ACH', 0.5313, 0.4, smarts=UFSG[9].smarts)
LLEUFSG[10] = UNIFAC_subgroup('AC', 3, 'ACH', 0.3652, 0.12, smarts=UFSG[10].smarts)

LLEUFSG[11] = UNIFAC_subgroup('ACCH3', 4, 'ACCH2', 1.2663, 0.968, smarts=UFSG[11].smarts)
LLEUFSG[12] = UNIFAC_subgroup('ACCH2', 4, 'ACCH2', 1.0396, 0.66, smarts=UFSG[12].smarts)
LLEUFSG[13] = UNIFAC_subgroup('ACCH', 4, 'ACCH2', 0.8121, 0.348, smarts=UFSG[13].smarts)

LLEUFSG[14] = UNIFAC_subgroup('OH', 5, 'OH', 1, 1.2, smarts=UFSG[14].smarts)

LLEUFSG[15] = UNIFAC_subgroup('P1', 6, 'P1', 3.2499, 3.128, smarts='CCCO') # 1-propanol, hopefully smarts works
LLEUFSG[16] = UNIFAC_subgroup('P2', 7, 'P2', 3.2491, 3.124, smarts='CC(C)O') # 2-propanol, hopefully smarts works
LLEUFSG[17] = UNIFAC_subgroup('H2O', 8, 'H2O', 0.92, 1.4, smarts=UFSG[16].smarts)

LLEUFSG[18] = UNIFAC_subgroup('ACOH', 9, 'ACOH', 0.8952, 0.68, smarts=UFSG[17].smarts)

LLEUFSG[19] = UNIFAC_subgroup('CH3CO', 10, 'CH2CO', 1.6724, 1.488, smarts=UFSG[18].smarts)
LLEUFSG[20] = UNIFAC_subgroup('CH2CO', 10, 'CH2CO', 1.4457, 1.18, smarts=UFSG[19].smarts)
LLEUFSG[21] = UNIFAC_subgroup('CHO', 11, 'CHO', 0.998, 0.948, smarts=UFSG[20].smarts)
LLEUFSG[22] = UNIFAC_subgroup('Furfural', 12, 'Furfural', 3.168, 2.484, smarts=UFSG[61].smarts)

LLEUFSG[23] = UNIFAC_subgroup('COOH', 13, 'COOH', 1.3013, 1.224, smarts=UFSG[42].smarts)
LLEUFSG[24] = UNIFAC_subgroup('HCOOH', 13, 'COOH', 1.528, 1.532, smarts=UFSG[43].smarts)

LLEUFSG[25] = UNIFAC_subgroup('CH3COO', 14, 'CCOO', 1.9031, 1.728, smarts=UFSG[21].smarts)
LLEUFSG[26] = UNIFAC_subgroup('CH2COO', 14, 'CCOO', 1.6764, 1.42, smarts=UFSG[22].smarts)

LLEUFSG[27] = UNIFAC_subgroup('CH3O', 15, 'CH2O', 1.145, 1.088, smarts=UFSG[24].smarts)
LLEUFSG[28] = UNIFAC_subgroup('CH2O', 15, 'CH2O', 0.9183, 0.78, smarts=UFSG[25].smarts)
LLEUFSG[29] = UNIFAC_subgroup('CHO', 15, 'CH2O', 0.6908, 0.468, smarts=UFSG[26].smarts)
LLEUFSG[30] = UNIFAC_subgroup('FCH2O', 15, 'CH2O', 9183, 1.1, smarts=UFSG[27].smarts) # THF in original and others, FCH2O here

LLEUFSG[31] = UNIFAC_subgroup('CH2CL', 16, 'CCL', 1.4654, 1.264, smarts=UFSG[44].smarts)
LLEUFSG[32] = UNIFAC_subgroup('CHCL', 16, 'CCL', 1.238, 0.952, smarts=UFSG[45].smarts)
LLEUFSG[33] = UNIFAC_subgroup('CCL', 16, 'CCL', 1.0106, 0.724, smarts=UFSG[46].smarts)

LLEUFSG[34] = UNIFAC_subgroup('CH2CL2', 17, 'CCL2', 2.2564, 1.988, smarts=UFSG[47].smarts)
LLEUFSG[35] = UNIFAC_subgroup('CHCL2', 17, 'CCL2', 2.0606, 1.684, smarts=UFSG[48].smarts)
LLEUFSG[36] = UNIFAC_subgroup('CCL2', 17, 'CCL2', 1.8016, 1.448, smarts=UFSG[49].smarts)

LLEUFSG[37] = UNIFAC_subgroup('CHCL3', 18, 'CCL3', 2.87, 2.41, smarts=UFSG[50].smarts)
LLEUFSG[38] = UNIFAC_subgroup('CCL3', 18, 'CCL3', 2.6401, 2.184, smarts=UFSG[51].smarts)

LLEUFSG[39] = UNIFAC_subgroup('CCL4', 19, 'CCL4', 3.39, 2.91, smarts=UFSG[52].smarts)

LLEUFSG[40] = UNIFAC_subgroup('ACCL', 20, 'ACCL', 1.1562, 0.844, smarts=UFSG[53].smarts)

LLEUFSG[41] = UNIFAC_subgroup('CH3CN', 21, 'CCN', 1.8701, 1.724, smarts=UFSG[40].smarts)
LLEUFSG[42] = UNIFAC_subgroup('CH2CN', 21, 'CCN', 1.6434, 1.416, smarts=UFSG[41].smarts)

LLEUFSG[43] = UNIFAC_subgroup('ACNH2', 22, 'ACNH2', 1.06, 0.816, smarts=UFSG[36].smarts)

LLEUFSG[44] = UNIFAC_subgroup('CH3NO2', 23, 'CNO2', 2.0086, 1.868, smarts=UFSG[54].smarts)
LLEUFSG[45] = UNIFAC_subgroup('CH2NO2', 23, 'CNO2', 1.7818, 1.56, smarts=UFSG[55].smarts)
LLEUFSG[46] = UNIFAC_subgroup('CHNO2', 23, 'CNO2', 1.5544, 1.248, smarts=UFSG[56].smarts)

LLEUFSG[47] = UNIFAC_subgroup('ACNO2', 24, 'ACNO2', 1.4199, 1.104, smarts=UFSG[57].smarts)

LLEUFSG[48] = UNIFAC_subgroup('DOH', 25, 'DOH', 2.4088, 2.248, smarts=UFSG[62].smarts)

LLEUFSG[49] = UNIFAC_subgroup('(HOCH2CH2)2O', 26, 'DEOH', 4.0013, 3.568, smarts='C(COCCO)O') # diethylene glycol

LLEUFSG[50] = UNIFAC_subgroup('C5H5N', 27, 'PYRIDINE', 2.9993, 2.113, smarts=UFSG[37].smarts)
LLEUFSG[51] = UNIFAC_subgroup('C5H4N', 27, 'PYRIDINE', 2.8332, 1.833, smarts=UFSG[38].smarts)
LLEUFSG[52] = UNIFAC_subgroup('C5H3N', 27, 'PYRIDINE', 2.667, 1.553, smarts=UFSG[39].smarts)

LLEUFSG[53] = UNIFAC_subgroup('CCl2=CHCl', 28, 'TCE', 3.3092, 2.860, smarts='C(=C(Cl)Cl)Cl') # trichloroethylene
LLEUFSG[54] = UNIFAC_subgroup('HCONHCH3', 29, 'MFA', 2.4317, 2.192, smarts='C(=C(Cl)Cl)Cl') # methylformamide
LLEUFSG[55] = UNIFAC_subgroup('DMF', 30, 'DMFA', 3.0856, 2.736, smarts=UFSG[72].smarts) # DMFA is same as DMF - dimethylformamide
LLEUFSG[56] = UNIFAC_subgroup('(CH2)4SO2', 31, 'TMS', 4.0358, 3.20, smarts='C1CCS(=O)(=O)C1') # tetramethylene sulfone
LLEUFSG[57] = UNIFAC_subgroup('DMSO', 32, 'DMSO', 2.8266, 2.472, smarts=UFSG[67].smarts)

# Generated with the following:
'''
t = 'LLEMG = {'
for main_key in range(1, 33):
    main_group = None
    hits = []
    for subgroup_key, o in LLEUFSG.items():
        if o.main_group_id == main_key:
            main_group = o.main_group
            hits.append(subgroup_key)
    t += '%d: ("%s", %s),\n' %(main_key, main_group, list(sorted(hits)))
t += '}'
'''

LLEMG = {   1: ("CH2", [1, 2, 3, 4]),
            2: ("C=C", [5, 6, 7, 8]),
            3: ("ACH", [9, 10]),
            4: ("ACCH2", [11, 12, 13]),
            5: ("OH", [14]),
            6: ("P1", [15]),
            7: ("P2", [16]),
            8: ("H2O", [17]),
            9: ("ACOH", [18]),
            10: ("CH2CO", [19, 20]),
            11: ("CHO", [21]),
            12: ("Furfural", [22]),
            13: ("COOH", [23, 24]),
            14: ("CCOO", [25, 26]),
            15: ("CH2O", [27, 28, 29, 30]),
            16: ("CCL", [31, 32, 33]),
            17: ("CCL2", [34, 35, 36]),
            18: ("CCL3", [37, 38]),
            19: ("CCL4", [39]),
            20: ("ACCL", [40]),
            21: ("CCN", [41, 42]),
            22: ("ACNH2", [43]),
            23: ("CNO2", [44, 45, 46]),
            24: ("ACNO2", [47]),
            25: ("DOH", [48]),
            26: ("DEOH", [49]),
            27: ("PYRIDINE", [50, 51, 52]),
            28: ("TCE", [53]),
            29: ("MFA", [54]),
            30: ("DMFA", [55]),
            31: ("TMS", [56]),
            32: ("DMSO", [57]),
}

'''
Larsen, Bent L., Peter Rasmussen, and Aage Fredenslund. "A Modified UNIFAC
Group-Contribution Model for Prediction of Phase Equilibria and Heats of Mixing."
Industrial & Engineering Chemistry Research 26, no. 11 (November 1, 1987):
2274-86. https://doi.org/10.1021/ie00071a018.
'''
LUFSG = {}
LUFSG[1] = UNIFAC_subgroup('CH3', 1, 'CH2', 0.9011, 0.848, smarts=UFSG[1].smarts)
LUFSG[2] = UNIFAC_subgroup('CH2', 1, 'CH2', 0.6744, 0.54, smarts=UFSG[2].smarts)
LUFSG[3] = UNIFAC_subgroup('CH', 1, 'CH2', 0.4469, 0.228, smarts=UFSG[3].smarts)
LUFSG[4] = UNIFAC_subgroup('C', 1, 'CH2', 0.2195, 0, smarts=UFSG[4].smarts)

LUFSG[5] = UNIFAC_subgroup('CH2=CH', 2, 'C=C', 1.3454, 1.176, smarts=UFSG[5].smarts)
LUFSG[6] = UNIFAC_subgroup('CH=CH', 2, 'C=C', 1.1167, 0.867, smarts=UFSG[6].smarts)
LUFSG[7] = UNIFAC_subgroup('CH2=C', 2, 'C=C', 1.1173, 0.988, smarts=UFSG[7].smarts)
LUFSG[8] = UNIFAC_subgroup('CH=C', 2, 'C=C', 0.8886, 0.676, smarts=UFSG[8].smarts)
LUFSG[9] = UNIFAC_subgroup('C=C', 2, 'C=C', 0.6605, 0.485, smarts=UFSG[70].smarts)

LUFSG[10] = UNIFAC_subgroup('ACH', 3, 'ACH', 0.5313, 0.4, smarts=UFSG[9].smarts)
LUFSG[11] = UNIFAC_subgroup('AC', 3, 'ACH', 0.3652, 0.12, smarts=UFSG[10].smarts)

LUFSG[12] = UNIFAC_subgroup('OH', 4, 'OH', 1, 1.2, smarts=UFSG[14].smarts)

LUFSG[13] = UNIFAC_subgroup('CH3OH', 5, 'CH3OH', 1.0, 1.0, smarts=UFSG[15].smarts)

LUFSG[14] = UNIFAC_subgroup('H2O', 6, 'H2O', 0.92, 1.4, smarts=UFSG[16].smarts)

LUFSG[15] = UNIFAC_subgroup('CH3CO', 7, 'CH2CO', 1.6724, 1.488, smarts=UFSG[18].smarts)
LUFSG[16] = UNIFAC_subgroup('CH2CO', 7, 'CH2CO', 1.4457, 1.488, smarts=UFSG[19].smarts)

LUFSG[17] = UNIFAC_subgroup('CHO', 8, 'CHO', 0.998, 0.948, smarts=UFSG[20].smarts)

LUFSG[18] = UNIFAC_subgroup('CH3COO', 9, 'CCOO', 1.9031, 1.728, smarts=UFSG[21].smarts)
LUFSG[19] = UNIFAC_subgroup('CH2COO', 9, 'CCOO', 1.6764, 1.42, smarts=UFSG[22].smarts)

LUFSG[20] = UNIFAC_subgroup('CH3O', 10, 'CH2O', 1.145, 0.9, smarts=UFSG[24].smarts)
LUFSG[21] = UNIFAC_subgroup('CH2O', 10, 'CH2O', 0.9183, 0.78, smarts=UFSG[25].smarts)
LUFSG[22] = UNIFAC_subgroup('CHO', 10, 'CH2O', 0.6908, 0.65, smarts=UFSG[26].smarts)
LUFSG[23] = UNIFAC_subgroup('THF', 10, 'CH2O', 0.9183, 1.1, smarts=UFSG[27].smarts)

LUFSG[24] = UNIFAC_subgroup('NH2', 11, 'NH2', 0.6948, 1.150, smarts='[NH2-]')

LUFSG[25] = UNIFAC_subgroup('CH3NH', 12, 'CNH2NG', 1.4337, 1.050, smarts=UFSG[31].smarts)
LUFSG[26] = UNIFAC_subgroup('CH2NH', 12, 'CNH2NG', 1.207, 0.936, smarts=UFSG[32].smarts)
LUFSG[27] = UNIFAC_subgroup('CHNH', 12, 'CNH2NG', 0.9795, 0.624, smarts=UFSG[33].smarts)

LUFSG[28] = UNIFAC_subgroup('CH3N', 13, 'CH2N', 1.1865, 0.94, smarts=UFSG[34].smarts)
LUFSG[29] = UNIFAC_subgroup('CH2N', 13, 'CH2N', 0.9597, 0.632, smarts=UFSG[35].smarts)

LUFSG[30] = UNIFAC_subgroup('ANH2', 14, 'ANH2', 0.6948, 1.4, smarts=None) # NH2 attached to aromatic ring

LUFSG[31] = UNIFAC_subgroup('C5H5N', 15, 'PYRIDINE', 2.9993, 2.113, smarts=UFSG[37].smarts)
LUFSG[32] = UNIFAC_subgroup('C5H4N', 15, 'PYRIDINE', 2.8332, 1.833, smarts=UFSG[38].smarts)
LUFSG[33] = UNIFAC_subgroup('C5H3N', 15, 'PYRIDINE', 2.667, 1.553, smarts=UFSG[39].smarts)

LUFSG[34] = UNIFAC_subgroup('CH3CN', 16, 'CCN', 1.8701, 1.724, smarts=UFSG[40].smarts)
LUFSG[35] = UNIFAC_subgroup('CH2CN', 16, 'CCN', 1.6434, 1.416, smarts=UFSG[41].smarts)

LUFSG[36] = UNIFAC_subgroup('COOH', 17, 'COOH', 1.3013, 1.224, smarts=UFSG[42].smarts)

LUFSG[37] = UNIFAC_subgroup('CH2CL', 18, 'CCL', 1.4654, 1.264, smarts=UFSG[44].smarts)
LUFSG[38] = UNIFAC_subgroup('CHCL', 18, 'CCL', 1.238, 0.952, smarts=UFSG[45].smarts)
LUFSG[39] = UNIFAC_subgroup('CCL', 18, 'CCL', 1.0106, 0.724, smarts=UFSG[46].smarts)

LUFSG[40] = UNIFAC_subgroup('CH2CL2', 19, 'CCL2', 2.2564, 1.988, smarts=UFSG[47].smarts)
LUFSG[41] = UNIFAC_subgroup('CHCL2', 19, 'CCL2', 2.0606, 1.684, smarts=UFSG[48].smarts)
LUFSG[42] = UNIFAC_subgroup('CCL2', 19, 'CCL2', 1.8016, 1.448, smarts=UFSG[49].smarts)

LUFSG[43] = UNIFAC_subgroup('CHCL3', 20, 'CCL3', 2.87, 2.41, smarts=UFSG[50].smarts)
LUFSG[44] = UNIFAC_subgroup('CCL3', 20, 'CCL3', 2.6401, 2.184, smarts=UFSG[51].smarts)

LUFSG[45] = UNIFAC_subgroup('CCL4', 21, 'CCL4', 3.39, 2.91, smarts=UFSG[52].smarts)

LUFMG = {1: ("CH2", [1, 2, 3, 4]),
2: ("C=C", [5, 6, 7, 8, 9]),
3: ("ACH", [10, 11]),
4: ("OH", [12]),
5: ("CH3OH", [13]),
6: ("H2O", [14]),
7: ("CH2CO", [15, 16]),
8: ("CHO", [17]),
9: ("CCOO", [18, 19]),
10: ("CH2O", [20, 21, 22, 23]),
11: ("NH2", [24]),
12: ("CNH2NG", [25, 26, 27]),
13: ("CH2N", [28, 29]),
14: ("ANH2", [30]),
15: ("PYRIDINE", [31, 32, 33]),
16: ("CCN", [34, 35]),
17: ("COOH", [36]),
18: ("CCL", [37, 38, 39]),
19: ("CCL2", [40, 41, 42]),
20: ("CCL3", [43, 44]),
21: ("CCL4", [45]),
}

NISTKTUFSG = {}
NISTKTUFSG[1] = UNIFAC_subgroup("CH3-", 1, 'C', 0.9011, 0.848)
NISTKTUFSG[2] = UNIFAC_subgroup("-CH2-", 1, 'C', 0.6744, 0.54)
NISTKTUFSG[3] = UNIFAC_subgroup("-CH<", 1, 'C', 0.4469, 0.228)
NISTKTUFSG[4] = UNIFAC_subgroup(">C<", 1, 'C', 0.2195, 0)
NISTKTUFSG[5] = UNIFAC_subgroup("CH2=CH-", 2, 'C=C', 1.3454, 1.176)
NISTKTUFSG[6] = UNIFAC_subgroup("-CH=CH-", 2, 'C=C', 1.1167, 0.867)
NISTKTUFSG[7] = UNIFAC_subgroup("CH2=C<", 2, 'C=C', 1.1173, 0.988)
NISTKTUFSG[8] = UNIFAC_subgroup("-CH=C<", 2, 'C=C', 0.8886, 0.676)
NISTKTUFSG[9] = UNIFAC_subgroup(">C=C<", 2, 'C=C', 0.6605, 0.485)
NISTKTUFSG[15] = UNIFAC_subgroup("-ACH-", 3, 'ACH', 0.5313, 0.4)
NISTKTUFSG[16] = UNIFAC_subgroup(">AC- (link)", 3, 'ACH', 0.3652, 0.12)
NISTKTUFSG[17] = UNIFAC_subgroup(">AC- (cond)", 3, 'ACH', 0.3125, 0.084)
NISTKTUFSG[18] = UNIFAC_subgroup(">AC-CH3", 4, 'ACCH2', 1.2663, 0.968)
NISTKTUFSG[19] = UNIFAC_subgroup(">AC-CH2-", 4, 'ACCH2', 1.0396, 0.66)
NISTKTUFSG[20] = UNIFAC_subgroup(">AC-CH<", 4, 'ACCH2', 0.8121, 0.348)
NISTKTUFSG[21] = UNIFAC_subgroup(">AC-C<-", 4, 'ACCH2', 0.5847, 0.084)
NISTKTUFSG[34] = UNIFAC_subgroup("-OH(primary)", 5, 'OH', 1, 1.2)
NISTKTUFSG[204] = UNIFAC_subgroup("-OH(secondary)", 5, 'OH', 1, 1.2)
NISTKTUFSG[205] = UNIFAC_subgroup("-OH(tertiary)", 5, 'OH', 1, 1.2)
NISTKTUFSG[35] = UNIFAC_subgroup("CH3OH", 6, 'CH2OH', 1.4311, 1.432)
NISTKTUFSG[36] = UNIFAC_subgroup("H2O", 7, 'H2O', 0.92, 1.4)
NISTKTUFSG[37] = UNIFAC_subgroup(">AC-OH", 8, 'ACOH', 0.8952, 0.68)
NISTKTUFSG[42] = UNIFAC_subgroup("CH3-CO-", 9, 'CH2CO', 1.6724, 1.488)
NISTKTUFSG[43] = UNIFAC_subgroup("-CH2-CO-", 9, 'CH2CO', 1.4457, 1.18)
NISTKTUFSG[44] = UNIFAC_subgroup(">CH-CO-", 9, 'CH2CO', 1.2182, 0.868)
NISTKTUFSG[45] = UNIFAC_subgroup("->C-CO-", 9, 'CH2CO', 0.9908, 0.64)
NISTKTUFSG[48] = UNIFAC_subgroup("-CHO", 10, 'CHO', 0.998, 0.948)
NISTKTUFSG[51] = UNIFAC_subgroup("CH3-COO-", 11, 'CCOO', 1.9031, 1.728)
NISTKTUFSG[52] = UNIFAC_subgroup("-CH2-COO-", 11, 'CCOO', 1.6764, 1.42)
NISTKTUFSG[53] = UNIFAC_subgroup(">CH-COO-", 11, 'CCOO', 1.4489, 1.108)
NISTKTUFSG[54] = UNIFAC_subgroup("->C-COO-", 11, 'CCOO', 1.2215, 0.88)
NISTKTUFSG[55] = UNIFAC_subgroup("HCOO-", 12, 'HCOO', 1.242, 1.188)
NISTKTUFSG[59] = UNIFAC_subgroup("CH3-O-", 13, 'CH2O', 1.145, 1.088)
NISTKTUFSG[60] = UNIFAC_subgroup("-CH2-O-", 13, 'CH2O', 0.9183, 0.78)
NISTKTUFSG[61] = UNIFAC_subgroup(">CH-O-", 13, 'CH2O', 0.6908, 0.468)
NISTKTUFSG[62] = UNIFAC_subgroup("->CO-", 13, 'CH2O', 0.9183, 0.24)
NISTKTUFSG[63] = UNIFAC_subgroup("-CH2-O- (cy)", 'CH2O', None, 0.9183, 1.1)
NISTKTUFSG[66] = UNIFAC_subgroup("CH3-NH2", 14, 'CNH2', 1.5959, 1.544)
NISTKTUFSG[67] = UNIFAC_subgroup("-CH2-NH2", 14, 'CNH2', 1.3692, 1.236)
NISTKTUFSG[68] = UNIFAC_subgroup(">CH-NH2", 14, 'CNH2', 1.1417, 0.924)
NISTKTUFSG[69] = UNIFAC_subgroup("->C-NH2", 14, 'CNH2', 0.9275, 0.696)
NISTKTUFSG[71] = UNIFAC_subgroup("CH3-NH-", 15, '(C)2NH', 1.4337, 1.244)
NISTKTUFSG[72] = UNIFAC_subgroup("-CH2-NH-", 15, '(C)2NH', 1.207, 0.936)
NISTKTUFSG[73] = UNIFAC_subgroup(">CH-NH-", 15, '(C)2NH', 0.9795, 0.624)
NISTKTUFSG[74] = UNIFAC_subgroup("CH3-N<", 16, '(C)3N', 1.1865, 0.94)
NISTKTUFSG[75] = UNIFAC_subgroup("-CH2-N<", 16, '(C)3N', 0.9597, 0.632)
NISTKTUFSG[79] = UNIFAC_subgroup(">AC-NH2", 17, 'ACNH2', 1.06, 0.816)
NISTKTUFSG[80] = UNIFAC_subgroup(">AC-NH-", 17, 'ACNH2', 0.8978, 0.516)
NISTKTUFSG[81] = UNIFAC_subgroup(">AC-N<", 17, 'ACNH2', 0.6506, 0.212)
NISTKTUFSG[76] = UNIFAC_subgroup("C5H5N", 18, 'Pyridine', 2.9993, 2.113)
NISTKTUFSG[77] = UNIFAC_subgroup("C5H4N-", 18, 'Pyridine', 2.8332, 1.833)
NISTKTUFSG[78] = UNIFAC_subgroup("C5H3N<", 18, 'Pyridine', 2.667, 1.553)
NISTKTUFSG[85] = UNIFAC_subgroup("CH3-CN", 19, 'CCN', 1.8701, 1.724)
NISTKTUFSG[86] = UNIFAC_subgroup("-CH2-CN", 19, 'CCN', 1.6434, 1.416)
NISTKTUFSG[87] = UNIFAC_subgroup(">CH-CN", 19, 'CCN', 1.416, 1.104)
NISTKTUFSG[88] = UNIFAC_subgroup("->C-CN", 19, 'CCN', 1.1885, 0.876)
NISTKTUFSG[94] = UNIFAC_subgroup("-COOH", 20, 'COOH', 1.3013, 1.224)
NISTKTUFSG[95] = UNIFAC_subgroup("HCOOH", 20, 'COOH', 1.528, 1.532)
NISTKTUFSG[99] = UNIFAC_subgroup("-CH2-Cl", 21, 'CCl', 1.4654, 1.264)
NISTKTUFSG[100] = UNIFAC_subgroup(">CH-Cl", 21, 'CCl', 1.238, 0.952)
NISTKTUFSG[101] = UNIFAC_subgroup("->CCl", 21, 'CCl', 1.0106, 0.724)
NISTKTUFSG[102] = UNIFAC_subgroup("CH2Cl2", 22, 'CCl2', 2.2564, 1.988)
NISTKTUFSG[103] = UNIFAC_subgroup("-CHCl2", 22, 'CCl2', 2.0606, 1.684)
NISTKTUFSG[104] = UNIFAC_subgroup(">CCl2", 22, 'CCl2', 1.8016, 1.448)
NISTKTUFSG[105] = UNIFAC_subgroup("CHCl3", 23, 'CCl3', 2.87, 2.41)
NISTKTUFSG[106] = UNIFAC_subgroup("-CCl3", 23, 'CCl3', 2.6401, 2.184)
NISTKTUFSG[107] = UNIFAC_subgroup("CCl4", 24, 'CCl4', 3.39, 2.91)
NISTKTUFSG[109] = UNIFAC_subgroup(">AC-Cl", 25, 'ACCl', 1.1562, 0.844)
NISTKTUFSG[132] = UNIFAC_subgroup("CH3-NO2", 26, 'CNO2', 2.0086, 1.868)
NISTKTUFSG[133] = UNIFAC_subgroup("-CH2-NO2", 26, 'CNO2', 1.7818, 1.56)
NISTKTUFSG[134] = UNIFAC_subgroup(">CH-NO2", 26, 'CNO2', 1.5544, 1.248)
NISTKTUFSG[135] = UNIFAC_subgroup("->C-NO2", 26, 'CNO2', 1.327, 1.02)
NISTKTUFSG[136] = UNIFAC_subgroup(">AC-NO2", 27, 'ACNO2', 1.4199, 1.104)
NISTKTUFSG[146] = UNIFAC_subgroup("CS2", 28, 'CS2', 2.057, 1.65)
NISTKTUFSG[138] = UNIFAC_subgroup("CH3-SH", 29, 'CH3SH', 1.877, 1.676)
NISTKTUFSG[139] = UNIFAC_subgroup("-CH2-SH", 29, 'CH3SH', 1.651, 1.368)
NISTKTUFSG[140] = UNIFAC_subgroup(">CH-SH", 29, 'CH3SH', 1.4232, 0.228)
NISTKTUFSG[141] = UNIFAC_subgroup("->C-SH", 29, 'CH3SH', 1.1958, 0)
NISTKTUFSG[50] = UNIFAC_subgroup("C5H4O2", 30, 'Furfural', 3.168, 2.484)
NISTKTUFSG[38] = UNIFAC_subgroup("(CH2OH)2", 31, 'DOH', 2.4088, 2.248)
NISTKTUFSG[128] = UNIFAC_subgroup("-I", 32, 'I', 1.264, 0.992)
NISTKTUFSG[130] = UNIFAC_subgroup("-Br", 33, 'Br', 0.9492, 0.832)
NISTKTUFSG[13] = UNIFAC_subgroup("CH≡C-", 34, 'C=-C', 1.292, 1.088)
NISTKTUFSG[14] = UNIFAC_subgroup("-C≡C-", 34, 'C=-C', 1.0613, 0.784)
NISTKTUFSG[153] = UNIFAC_subgroup("DMSO", 35, 'DMSO', 2.8266, 2.472)
NISTKTUFSG[90] = UNIFAC_subgroup("CH2=CH-CN", 36, 'ACRY', 2.3144, 2.052)
NISTKTUFSG[108] = UNIFAC_subgroup("Cl(C=C)", 37, 'Cl(C=C)', 0.791, 0.724)
NISTKTUFSG[118] = UNIFAC_subgroup(">AC-F", 38, 'ACF', 0.6948, 0.524)
NISTKTUFSG[161] = UNIFAC_subgroup("DMF", 39, 'DMF', 3.0856, 2.736)
NISTKTUFSG[162] = UNIFAC_subgroup("-CON(CH3)2", 39, 'DMF', 2.8589, 2.428)
NISTKTUFSG[163] = UNIFAC_subgroup("-CON(CH2)(CH3)-", 39, 'DMF', 2.6322, 2.12)
NISTKTUFSG[164] = UNIFAC_subgroup("HCON(CH2)2<", 39, 'DMF', 2.6322, 2.12)
NISTKTUFSG[165] = UNIFAC_subgroup("-CON(CH2)2<", 39, 'DMF', 2.4054, 1.812)
NISTKTUFSG[111] = UNIFAC_subgroup("CHF3", 40, 'CF2', 1.5781, 1.548)
NISTKTUFSG[112] = UNIFAC_subgroup("-CF3", 40, 'CF2', 1.406, 1.38)
NISTKTUFSG[113] = UNIFAC_subgroup("-CHF2", 40, 'CF2', 1.2011, 1.108)
NISTKTUFSG[114] = UNIFAC_subgroup(">CF2", 40, 'CF2', 1.0105, 0.92)
NISTKTUFSG[115] = UNIFAC_subgroup("-CH2F", 40, 'CF2', 1.0514, 0.98)
NISTKTUFSG[116] = UNIFAC_subgroup(">CH-F", 40, 'CF2', 0.824, 0.668)
NISTKTUFSG[117] = UNIFAC_subgroup("->CF", 40, 'CF2', 0.615, 0.46)
NISTKTUFSG[58] = UNIFAC_subgroup("-COO-", 41, 'COO', 1.38, 1.2)
NISTKTUFSG[197] = UNIFAC_subgroup("SiH3-", 42, 'SiH2', 1.6035, 1.263)
NISTKTUFSG[198] = UNIFAC_subgroup("-SiH2-", 42, 'SiH2', 1.4443, 1.006)
NISTKTUFSG[199] = UNIFAC_subgroup(">SiH-", 42, 'SiH2', 1.2853, 0.749)
NISTKTUFSG[200] = UNIFAC_subgroup(">Si<", 42, 'SiH2', 1.047, 0.41)
NISTKTUFSG[201] = UNIFAC_subgroup("-SiH2-O-", 43, 'SiO', 1.4838, 1.062)
NISTKTUFSG[202] = UNIFAC_subgroup(">SiH-O-", 43, 'SiO', 1.303, 0.764)
NISTKTUFSG[203] = UNIFAC_subgroup("->Si-O-", 43, 'SiO', 1.1044, 0.466)
NISTKTUFSG[195] = UNIFAC_subgroup("NMP", 44, 'NMP', 3.981, 3.2)
NISTKTUFSG[120] = UNIFAC_subgroup("CCl3F", 45, 'CClF', 3.0356, 2.644)
NISTKTUFSG[121] = UNIFAC_subgroup("-CCl2F", 45, 'CClF', 2.2287, 1.916)
NISTKTUFSG[122] = UNIFAC_subgroup("HCCl2F", 45, 'CClF', 2.406, 2.116)
NISTKTUFSG[123] = UNIFAC_subgroup("-HCClF", 45, 'CClF', 1.6493, 1.416)
NISTKTUFSG[124] = UNIFAC_subgroup("-CClF2", 45, 'CClF', 1.8174, 1.648)
NISTKTUFSG[125] = UNIFAC_subgroup("HCClF2", 45, 'CClF', 1.967, 1.828)
NISTKTUFSG[126] = UNIFAC_subgroup("CClF3", 45, 'CClF', 2.1721, 2.1)
NISTKTUFSG[127] = UNIFAC_subgroup("CCl2F2", 45, 'CClF', 2.6243, 2.376)
NISTKTUFSG[166] = UNIFAC_subgroup("-CONH(CH3)", 46, 'CONCH2', 2.205, 1.884)
NISTKTUFSG[167] = UNIFAC_subgroup("HCONH(CH2)-", 46, 'CONCH2', 2.205, 1.884)
NISTKTUFSG[168] = UNIFAC_subgroup("-CONH(CH2)-", 46, 'CONCH2', 1.9782, 1.576)
NISTKTUFSG[169] = UNIFAC_subgroup("-CONH2", 46, 'CONCH2', 1.4661, 1.336)
NISTKTUFSG[39] = UNIFAC_subgroup("-O-CH2-CH2-OH", 47, 'OCCOH', 2.1226, 1.904)
NISTKTUFSG[40] = UNIFAC_subgroup("-O-CH-CH2-OH", 47, 'OCCOH', 1.8952, 1.592)
NISTKTUFSG[41] = UNIFAC_subgroup("-O-CH2-CH-OH", 47, 'OCCOH', 1.8952, 1.592)
NISTKTUFSG[142] = UNIFAC_subgroup("CH3-S-", 48, 'CH2S', 1.613, 1.368)
NISTKTUFSG[143] = UNIFAC_subgroup("-CH2-S-", 48, 'CH2S', 1.3863, 1.06)
NISTKTUFSG[144] = UNIFAC_subgroup(">CH-S-", 48, 'CH2S', 1.1589, 0.748)
NISTKTUFSG[145] = UNIFAC_subgroup("->C-S-", 48, 'CH2S', 0.9314, 0.52)
NISTKTUFSG[196] = UNIFAC_subgroup("MORPHOLIN", 49, 'Morpholin', 3.474, 2.796)
NISTKTUFSG[147] = UNIFAC_subgroup("THIOPHENE", 50, 'THIOPHENE', 2.8569, 2.14)
NISTKTUFSG[148] = UNIFAC_subgroup("C4H3S-", 50, 'THIOPHENE', 2.6908, 1.86)
NISTKTUFSG[149] = UNIFAC_subgroup("C4H2S<", 50, 'THIOPHENE', 2.5247, 1.58)
NISTKTUFSG[27] = UNIFAC_subgroup("-CH2- (cy)", 51, 'CH2(cyc)', 0.6744, 0.54)
NISTKTUFSG[28] = UNIFAC_subgroup(">CH- (cy)", 51, 'CH2(cyc)', 0.4469, 0.228)
NISTKTUFSG[29] = UNIFAC_subgroup(">C< (cy)", 51, 'CH2(cyc)', 0.2195, 0)
NISTKTUFSG[30] = UNIFAC_subgroup("-CH=CH- (cy)", 52, 'C=C(cyc)', 1.1167, 0.867)
NISTKTUFSG[31] = UNIFAC_subgroup("CH2=C< (cy)", 52, 'C=C(cyc)', 1.1173, 0.988)
NISTKTUFSG[32] = UNIFAC_subgroup("-CH=C< (cy)", 52, 'C=C(cyc)', 0.8886, 0.676)

NISTKTUFMG = {1: ("C", [1, 2, 3, 4]),
2: ("C=C", [5, 6, 7, 8, 9]),
3: ("ACH", [15, 16, 17]),
4: ("ACCH2", [18, 19, 20, 21]),
5: ("OH", [34, 204, 205]),
6: ("CH2OH", [35]),
7: ("H2O", [36]),
8: ("ACOH", [37]),
9: ("CH2CO", [42, 43, 44, 45]),
10: ("CHO", [48]),
11: ("CCOO", [51, 52, 53, 54]),
12: ("HCOO", [55]),
13: ("CH2O", [59, 60, 61, 62]),
14: ("CNH2", [66, 67, 68, 69]),
15: ("(C)2NH", [71, 72, 73]),
16: ("(C)3N", [74, 75]),
17: ("ACNH2", [79, 80, 81]),
18: ("Pyridine", [76, 77, 78]),
19: ("CCN", [85, 86, 87, 88]),
20: ("COOH", [94, 95]),
21: ("CCl", [99, 100, 101]),
22: ("CCl2", [102, 103, 104]),
23: ("CCl3", [105, 106]),
24: ("CCl4", [107]),
25: ("ACCl", [109]),
26: ("CNO2", [132, 133, 134, 135]),
27: ("ACNO2", [136]),
28: ("CS2", [146]),
29: ("CH3SH", [138, 139, 140, 141]),
30: ("Furfural", [50]),
31: ("DOH", [38]),
32: ("I", [128]),
33: ("Br", [130]),
34: ("C=-C", [13, 14]),
35: ("DMSO", [153]),
36: ("ACRY", [90]),
37: ("Cl(C=C)", [108]),
38: ("ACF", [118]),
39: ("DMF", [161, 162, 163, 164, 165]),
40: ("CF2", [111, 112, 113, 114, 115, 116, 117]),
41: ("COO", [58]),
42: ("SiH2", [197, 198, 199, 200]),
43: ("SiO", [201, 202, 203]),
44: ("NMP", [195]),
45: ("CClF", [120, 121, 122, 123, 124, 125, 126, 127]),
46: ("CONCH2", [166, 167, 168, 169]),
47: ("OCCOH", [39, 40, 41]),
48: ("CH2S", [142, 143, 144, 145]),
49: ("Morpholin", [196]),
50: ("THIOPHENE", [147, 148, 149]),
51: ("CH2(cyc)", [27, 28, 29]),
52: ("C=C(cyc)", [30, 31, 32]),
}

'''Compared to storing the values in dict[(int1, int2)] = (values),
the dict-in-dict structure is found emperically to take 111608 bytes vs.
79096 bytes, or 30% less memory.
'''

global _unifac_ip_loaded
_unifac_ip_loaded = False
def load_unifac_ip():
    global _unifac_ip_loaded, UFIP, LLEUFIP, LUFIP, DOUFIP2006, DOUFIP2016, NISTUFIP, NISTKTUFIP, PSRKIP, VTPRIP
    folder = os.path.join(os.path.dirname(__file__), 'Phase Change')

    UFIP = {i: {} for i in list(range(1, 52)) + [55, 84, 85]}
    with open(os.path.join(folder, 'UNIFAC original interaction parameters.tsv')) as f:
        for line in f:
            maingroup1, maingroup2, interaction_parameter = line.strip('\n').split('\t')
            # Index by both int, order maters, to only one parameter.
            UFIP[int(maingroup1)][int(maingroup2)] = float(interaction_parameter)


    LLEUFIP = {i: {} for i in list(range(1, 33))}
    with open(os.path.join(folder, 'UNIFAC LLE interaction parameters.tsv')) as f:
        for line in f:
            maingroup1, maingroup2, interaction_parameter = line.strip('\n').split('\t')
            LLEUFIP[int(maingroup1)][int(maingroup2)] = float(interaction_parameter)

    LUFIP = {i: {} for i in list(range(1, 22))}
    with open(os.path.join(folder, 'UNIFAC Lyngby interaction parameters.tsv')) as f:
        for line in f:
            maingroup1, maingroup2, a, b, c = line.strip('\n').split('\t')
            LUFIP[int(maingroup1)][int(maingroup2)] = (float(a), float(b), float(c))


    DOUFIP2006 = {i: {} for i in DOUFMG.keys()}
    with open(os.path.join(folder, 'UNIFAC modified Dortmund interaction parameters 2006.tsv')) as f:
        for line in f:
            maingroup1, maingroup2, a, b, c = line.strip('\n').split('\t')
            DOUFIP2006[int(maingroup1)][int(maingroup2)] = (float(a), float(b), float(c))

    DOUFIP2016 = {i: {} for i in list(DOUFMG.keys())+[50, 77, 98, 99]}
    # Some of the groups have no public parameters unfortunately
    with open(os.path.join(folder, 'UNIFAC modified Dortmund interaction parameters.tsv')) as f:
        for line in f:
            maingroup1, maingroup2, a, b, c = line.strip('\n').split('\t')
            DOUFIP2016[int(maingroup1)][int(maingroup2)] = (float(a), float(b), float(c))


    #NISTUFIP = {i: {} for i in list(NISTUFMG.keys())}
    NISTUFIP = {i: {} for i in list(range(87)) + [92, 94, 95, 96] }

    with open(os.path.join(folder, 'UNIFAC modified NIST 2015 interaction parameters.tsv')) as f:
        for line in f:
            maingroup1, maingroup2, a, b, c, Tmin, Tmax = line.strip('\n').split('\t')
            NISTUFIP[int(maingroup1)][int(maingroup2)] = (float(a), float(b), float(c))

    NISTKTUFIP = {i: {} for i in range(1, 53) }
    with open(os.path.join(folder, 'NIST KT 2011 interaction parameters.tsv')) as f:
        for line in f:
            maingroup1, maingroup2, a, b, c = line.strip('\n').split('\t')
            NISTKTUFIP[int(maingroup1)][int(maingroup2)] = (float(a), float(b), float(c))


    PSRKIP = {i: {} for i in range(1, 86)}

    with open(os.path.join(folder, 'PSRK interaction parameters.tsv')) as f:
        for line in f:
            maingroup1, maingroup2, a, b, c = line.strip('\n').split('\t')
            PSRKIP[int(maingroup1)][int(maingroup2)] = (float(a), float(b), float(c))


    VTPRIP = {i: {} for i in range(1, 200)}
    # Three existing documents
    for name in ('VTPR 2012 interaction parameters.tsv', 'VTPR 2014 interaction parameters.tsv', 'VTPR 2016 interaction parameters.tsv'):
        with open(os.path.join(folder, name)) as f:
            for line in f:
                maingroup1, maingroup2, a, b, c = line.strip('\n').split('\t')
                VTPRIP[int(maingroup1)][int(maingroup2)] = (float(a), float(b), float(c))

    _unifac_ip_loaded = True


if PY37:
    def __getattr__(name):
        if name in ('UFIP', 'LLEUFIP', 'LUFIP', 'DOUFIP2006', 'DOUFIP2016',
                    'NISTUFIP', 'NISTKTUFIP', 'PSRKIP', 'VTPRIP'):
            load_unifac_ip()
            return globals()[name]
        raise AttributeError("module %s has no attribute %s" %(__name__, name))
else:
    if can_load_data:
        load_unifac_ip()



DDBST_UNIFAC_assignments = {}
DDBST_MODIFIED_UNIFAC_assignments = {}
DDBST_PSRK_assignments = {}

def load_group_assignments_DDBST():
    '''Data is stored in the format
    InChI key\tbool bool bool \tsubgroup count ...\tsubgroup count \tsubgroup count...
    where the bools refer to whether or not the original UNIFAC, modified
    UNIFAC, and PSRK group assignments were completed correctly.
    The subgroups and their count have an indefinite length.
    '''
    # Do not allow running multiple times
    if DDBST_UNIFAC_assignments:
        return None
    folder = os.path.join(os.path.dirname(__file__), 'Phase Change')
    with open(os.path.join(folder, 'DDBST UNIFAC assignments.tsv')) as f:
        _group_assignments = [DDBST_UNIFAC_assignments, DDBST_MODIFIED_UNIFAC_assignments, DDBST_PSRK_assignments]
        for line in f.readlines():
            key, valids, original, modified, PSRK = line.split('\t')
            # list of whether or not each method was correctly identified or not
            valids = [True if i == '1' else False for i in valids.split(' ')]
            for groups, storage, valid in zip([original, modified, PSRK], _group_assignments, valids):
                if valid:
                    groups = groups.rstrip().split(' ')
                    d_data = {}
                    for i in range(int(len(groups)/2)):
                        d_data[int(groups[i*2])] = int(groups[i*2+1])
                    storage[key] = d_data


def UNIFAC_RQ(groups, subgroup_data=None):
    r'''Calculates UNIFAC parameters R and Q for a chemical, given a dictionary
    of its groups, as shown in [1]_. Most UNIFAC methods use the same subgroup
    values; however, a dictionary of `UNIFAC_subgroup` instances may be
    specified as an optional second parameter.

    .. math::
        r_i = \sum_{k=1}^{n} \nu_k R_k

    .. math::
        q_i = \sum_{k=1}^{n}\nu_k Q_k

    Parameters
    ----------
    groups : dict[count]
        Dictionary of numeric subgroup IDs : their counts
    subgroup_data : None or dict[UNIFAC_subgroup]
        Optional replacement for standard subgroups; leave as None to use the
        original UNIFAC subgroup r and q values.

    Returns
    -------
    R : float
        R UNIFAC parameter (normalized Van der Waals Volume)  [-]
    Q : float
        Q UNIFAC parameter (normalized Van der Waals Area)  [-]

    Notes
    -----
    These parameters have some predictive value for other chemical properties.

    Examples
    --------
    Hexane

    >>> UNIFAC_RQ({1:2, 2:4})
    (4.4998000000000005, 3.856)

    References
    ----------
    .. [1] Gmehling, Jurgen. Chemical Thermodynamics: For Process Simulation.
       Weinheim, Germany: Wiley-VCH, 2012.
    '''
    if subgroup_data is not None:
        subgroups = subgroup_data
    else:
        subgroups = UFSG
    ri = 0.
    qi = 0.
    for group, count in groups.items():
        ri += subgroups[group].R*count
        qi += subgroups[group].Q*count
    return ri, qi


def Van_der_Waals_volume(R):
    r'''Calculates a species Van der Waals molar volume with the UNIFAC method,
    given a species's R parameter.

    .. math::
        V_{wk} = 15.17R_k

    Parameters
    ----------
    R : float
        R UNIFAC parameter (normalized Van der Waals Volume)  [-]

    Returns
    -------
    V_vdw : float
        Unnormalized Van der Waals volume, [m^3/mol]

    Notes
    -----
    The volume was originally given in cm^3/mol, but is converted to SI here.

    Examples
    --------
    >>> Van_der_Waals_volume(4.4998)
    6.826196599999999e-05

    References
    ----------
    .. [1] Wei, James, Morton M. Denn, John H. Seinfeld, Arup Chakraborty,
       Jackie Ying, Nicholas Peppas, and George Stephanopoulos. Molecular
       Modeling and Theory in Chemical Engineering. Academic Press, 2001.
    '''
    return R*1.517e-05


def Van_der_Waals_area(Q):
    r'''Calculates a species Van der Waals molar surface area with the UNIFAC
    method, given a species's Q parameter.

    .. math::
        A_{wk} = 2.5\times 10^9 Q_k

    Parameters
    ----------
    Q : float
        Q UNIFAC parameter (normalized Van der Waals Area)  [-]

    Returns
    -------
    A_vdw : float
        Unnormalized Van der Waals surface area, [m^2/mol]

    Notes
    -----
    The volume was originally given in cm^2/mol, but is converted to SI here.

    Examples
    --------
    >>> Van_der_Waals_area(3.856)
    964000.0

    References
    ----------
    .. [1] Wei, James, Morton M. Denn, John H. Seinfeld, Arup Chakraborty,
       Jackie Ying, Nicholas Peppas, and George Stephanopoulos. Molecular
       Modeling and Theory in Chemical Engineering. Academic Press, 2001.
    '''
    return Q*250000.0


def UNIFAC_psi(T, subgroup1, subgroup2, subgroup_data, interaction_data,
               modified=False):
    r'''Calculates the interaction parameter psi(m, n) for two UNIFAC
    subgroups, given the system temperature, the UNIFAC subgroups considered
    for the variant of UNIFAC used, the interaction parameters for the
    variant of UNIFAC used, and whether or not the temperature dependence is
    modified from the original form, as shown below.

    Original temperature dependence:

    .. math::
        \Psi_{mn} = \exp\left(\frac{-a_{mn}}{T}\right)

    Modified temperature dependence:

    .. math::
        \Psi_{mn} = \exp\left(\frac{-a_{mn} - b_{mn}T - c_{mn}T^2}{T}\right)

    Parameters
    ----------
    T : float
        Temperature of the system, [K]
    subgroup1 : int
        First UNIFAC subgroup for identifier, [-]
    subgroup2 : int
        Second UNIFAC subgroup for identifier, [-]
    subgroup_data : dict[UNIFAC_subgroup]
        Normally provided as inputs to `UNIFAC`.
    interaction_data : dict[dict[tuple(a_mn, b_mn, c_mn)]]
        Normally provided as inputs to `UNIFAC`.
    modified : bool
        True if the modified temperature dependence is used by the interaction
        parameters, otherwise False

    Returns
    -------
    psi : float
        UNIFAC interaction parameter term, [-]

    Notes
    -----
    UNIFAC interaction parameters are asymmetric. No warning is raised if an
    interaction parameter is missing.

    Examples
    --------
    >>> from thermo.unifac import UFSG, UFIP, DOUFSG, DOUFIP2006

    >>> UNIFAC_psi(307, 18, 1, UFSG, UFIP)
    0.9165248264184787

    >>> UNIFAC_psi(373.15, 9, 78, DOUFSG, DOUFIP2006, modified=True)
    1.3703140538273264

    References
    ----------
    .. [1] Gmehling, Jurgen. Chemical Thermodynamics: For Process Simulation.
       Weinheim, Germany: Wiley-VCH, 2012.
    .. [2] Fredenslund, Aage, Russell L. Jones, and John M. Prausnitz. "Group
       Contribution Estimation of Activity Coefficients in Nonideal Liquid
       Mixtures." AIChE Journal 21, no. 6 (November 1, 1975): 1086-99.
       doi:10.1002/aic.690210607.
    '''
    main1 = subgroup_data[subgroup1].main_group_id
    main2 = subgroup_data[subgroup2].main_group_id
    if modified:
        try:
            a, b, c = interaction_data[main1][main2]
        except:
            return 1.
        return exp((-a/T -b - c*T))
    else:
        try:
            return exp(-interaction_data[main1][main2]/T)
        except:
            return 1.


def UNIFAC_gammas(T, xs, chemgroups, cached=None, subgroup_data=None,
                  interaction_data=None, modified=False):
    r'''Calculates activity coefficients using the UNIFAC model (optionally
    modified), given a mixture's temperature, liquid mole fractions,
    and optionally the subgroup data and interaction parameter data of your
    choice. The default is to use the original UNIFAC model, with the latest
    parameters published by DDBST. The model supports modified forms (Dortmund,
    NIST) when the `modified` parameter is True.

    Parameters
    ----------
    T : float
        Temperature of the system, [K]
    xs : list[float]
        Mole fractions of all species in the system in the liquid phase, [-]
    chemgroups : list[dict]
        List of dictionaries of subgroup IDs and their counts for all species
        in the mixture, [-]
    subgroup_data : dict[UNIFAC_subgroup]
        UNIFAC subgroup data; available dictionaries in this module are UFSG
        (original), DOUFSG (Dortmund), or NISTUFSG ([4]_).
    interaction_data : dict[dict[tuple(a_mn, b_mn, c_mn)]]
        UNIFAC interaction parameter data; available dictionaries in this
        module are UFIP (original), DOUFIP2006 (Dortmund parameters as
        published by 2006), DOUFIP2016 (Dortmund parameters as published by
        2016), and NISTUFIP ([4]_).
    modified : bool
        True if using the modified form and temperature dependence, otherwise
        False.

    Returns
    -------
    gammas : list[float]
        Activity coefficients of all species in the mixture, [-]

    Notes
    -----
    The actual implementation of UNIFAC is formulated slightly different than
    the formulas above for computational efficiency. DDBST switched to using
    the more efficient forms in their publication, but the numerical results
    are identical.

    The model is as follows:

    .. math::
        \ln \gamma_i =  \ln \gamma_i^c + \ln \gamma_i^r

    **Combinatorial component**

    .. math::
        \ln \gamma_i^c = \ln \frac{\phi_i}{x_i} + \frac{z}{2} q_i
        \ln\frac{\theta_i}{\phi_i} + L_i - \frac{\phi_i}{x_i}
        \sum_{j=1}^{n} x_j L_j

    .. math::
        \theta_i = \frac{x_i q_i}{\sum_{j=1}^{n} x_j q_j}

    .. math::
         \phi_i = \frac{x_i r_i}{\sum_{j=1}^{n} x_j r_j}

    .. math::
         L_i = 5(r_i - q_i)-(r_i-1)

    **Residual component**

    .. math::
        \ln \gamma_i^r = \sum_{k}^n \nu_k^{(i)} \left[ \ln \Gamma_k
        - \ln \Gamma_k^{(i)} \right]

    .. math::
        \ln \Gamma_k = Q_k \left[1 - \ln \sum_m \Theta_m \Psi_{mk} - \sum_m
        \frac{\Theta_m \Psi_{km}}{\sum_n \Theta_n \Psi_{nm}}\right]

    .. math::
        \Theta_m = \frac{Q_m X_m}{\sum_{n} Q_n X_n}

    .. math::
        X_m = \frac{ \sum_j \nu^j_m x_j}{\sum_j \sum_n \nu_n^j x_j}

    **R and Q**

    .. math::
        r_i = \sum_{k=1}^{n} \nu_k R_k

    .. math::
        q_i = \sum_{k=1}^{n}\nu_k Q_k

    The newer forms of UNIFAC (Dortmund, NIST) calculate the combinatorial
    part slightly differently:

    .. math::
        \ln \gamma_i^c = 1 - {V'}_i + \ln({V'}_i) - 5q_i \left(1
        - \frac{V_i}{F_i}+ \ln\left(\frac{V_i}{F_i}\right)\right)

    .. math::
        V'_i = \frac{r_i^{3/4}}{\sum_j r_j^{3/4}x_j}

    .. math::
        V_i = \frac{r_i}{\sum_j r_j x_j}

    .. math::
        F_i = \frac{q_i}{\sum_j q_j x_j}

    Although this form looks substantially different than the original, it
    infact reverts to the original form if only :math:`V'_i` is replaced by
    :math:`V_i`. This is more clear when looking at the full rearranged form as
    in [3]_.

    In some publications such as [5]_, the nomenclature is such that
    :math:`\theta_i` and :math:`\phi` do not contain the top :math:`x_i`,
    making :math:`\theta_i = F_i` and  :math:`\phi_i = V_i`. [5]_ is also
    notable for having supporting information containing very nice sets of
    analytical derivatives.

    UNIFAC LLE uses the original formulation of UNIFAC, and otherwise only
    different interaction parameters.

    Examples
    --------
    >>> UNIFAC_gammas(T=333.15, xs=[0.5, 0.5], chemgroups=[{1:2, 2:4}, {1:1, 2:1, 18:1}])
    [1.427602583562, 1.364654501010]

    >>> from thermo.unifac import DOUFIP2006
    >>> UNIFAC_gammas(373.15, [0.2, 0.3, 0.2, 0.2],
    ... [{9:6}, {78:6}, {1:1, 18:1}, {1:1, 2:1, 14:1}],
    ... subgroup_data=DOUFSG, interaction_data=DOUFIP2006, modified=True)
    [1.1864311137, 1.44028013391, 1.20447983349, 1.972070609029]

    References
    ----------
    .. [1] Gmehling, Jurgen. Chemical Thermodynamics: For Process Simulation.
       Weinheim, Germany: Wiley-VCH, 2012.
    .. [2] Fredenslund, Aage, Russell L. Jones, and John M. Prausnitz. "Group
       Contribution Estimation of Activity Coefficients in Nonideal Liquid
       Mixtures." AIChE Journal 21, no. 6 (November 1, 1975): 1086-99.
       doi:10.1002/aic.690210607.
    .. [3] Jakob, Antje, Hans Grensemann, Jürgen Lohmann, and Jürgen Gmehling.
       "Further Development of Modified UNIFAC (Dortmund):  Revision and
       Extension 5." Industrial & Engineering Chemistry Research 45, no. 23
       (November 1, 2006): 7924-33. doi:10.1021/ie060355c.
    .. [4] Kang, Jeong Won, Vladimir Diky, and Michael Frenkel. "New Modified
       UNIFAC Parameters Using Critically Evaluated Phase Equilibrium Data."
       Fluid Phase Equilibria 388 (February 25, 2015): 128-41.
       doi:10.1016/j.fluid.2014.12.042.
    .. [5] Jäger, Andreas, Ian H. Bell, and Cornelia Breitkopf. "A
       Theoretically Based Departure Function for Multi-Fluid Mixture Models."
       Fluid Phase Equilibria 469 (August 15, 2018): 56-69.
       https://doi.org/10.1016/j.fluid.2018.04.015.
    '''
    cmps = range(len(xs))
    if subgroup_data is None:
        subgroups = UFSG
    else:
        subgroups = subgroup_data
    if interaction_data is None:
        if not _unifac_ip_loaded: load_unifac_ip()
        interactions = UFIP
    else:
        interactions = interaction_data

    # Obtain r and q values using the subgroup values
    if not cached:
        rs = []
        qs = []
        for groups in chemgroups:
            ri = 0.
            qi = 0.
            for group, count in groups.items():
                ri += subgroups[group].R*count
                qi += subgroups[group].Q*count
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
    # Xms stored in group_count_xs, length number of independent groups
    group_count_xs = {}
    for group in group_counts:
        tot_numerator = sum(chemgroups[i][group]*xs[i] for i in cmps if group in chemgroups[i])
        group_count_xs[group] = tot_numerator/group_sum
#    print(group_count_xs, 'group_count_xs')

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
#    print(loggammacs)

    Q_sum_term = sum([subgroups[group].Q*group_count_xs[group] for group in group_counts])

    # theta(m) for an overall mixture composition
    area_fractions = {group: subgroups[group].Q*group_count_xs[group]/Q_sum_term
                      for group in group_counts.keys()}
#    print('theta(m) for an overall mixture ', area_fractions)

    # This needs to not be a dictionary
    UNIFAC_psis = {k: {m:(UNIFAC_psi(T, m, k, subgroups, interactions, modified=modified))
                   for m in group_counts} for k in group_counts}

    loggamma_groups = {}
    # This is for the total mixture bit of the residual
    for k in group_counts:
        sum1, sum2 = 0., 0.
        for m in group_counts:
            sum1 += area_fractions[m]*UNIFAC_psis[k][m]
            # This term can be pre-calculated
            sum3 = sum(area_fractions[n]*UNIFAC_psis[m][n] for n in group_counts)
            sum2 -= area_fractions[m]*UNIFAC_psis[m][k]/sum3
        loggamma_groups[k] = subgroups[k].Q*(1. - log(sum1) + sum2)

    loggammars = []
    for groups in chemgroups:
        # Most of this is for the pure-component bit of the residual
        chem_loggamma_groups = {}
        chem_group_sum = sum(groups.values())

        # Xm = chem_group_count_xs
        chem_group_count_xs = {group: count/chem_group_sum for group, count in groups.items()}
#        print('Xm', chem_group_count_xs)

        # denominator of term used to compute Theta(m)
        Q_sum_term = sum([subgroups[group].Q*chem_group_count_xs[group] for group in groups])

        # Theta(m) = chem_area_fractions (dict indexed by main group)
        chem_area_fractions = {group: subgroups[group].Q*chem_group_count_xs[group]/Q_sum_term
                               for group in groups.keys()}
#        print('Theta(m)', chem_area_fractions)

        for k in groups:
            sum1, sum2 = 0., 0.
            for m in groups:
                sum1 += chem_area_fractions[m]*UNIFAC_psis[k][m]

                # sum3 should be cached
                sum3 = sum(chem_area_fractions[n]*UNIFAC_psis[m][n] for n in groups)
                sum2 -= chem_area_fractions[m]*UNIFAC_psis[m][k]/sum3

            chem_loggamma_groups[k] = subgroups[k].Q*(1. - log(sum1) + sum2)

        tot = sum([count*(loggamma_groups[group] - chem_loggamma_groups[group])
                   for group, count in groups.items()])
        loggammars.append(tot)

    return [exp(loggammacs[i]+loggammars[i]) for i in cmps]

def chemgroups_to_matrix(chemgroups):
    r'''
    Index by [group index][compound index]

    >>> chemgroups_to_matrix([{9: 6}, {2: 6}, {1: 1, 18: 1}, {1: 1, 2: 1, 14: 1}])
    [[0, 0, 1, 1], [0, 6, 0, 1], [6, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    '''
    matrix = []
    keys = []
    all_keys = set()
    [all_keys.update(i.keys()) for i in chemgroups]
    for k in sorted(list(all_keys)):
        matrix.append([l[k] if k in l else 0 for l in chemgroups])
#        matrix.append([float(l[k]) if k in l else 0.0 for l in chemgroups]) # Cannot notice performance improvement
    return matrix

def unifac_psis(T, N_groups, version, psi_a, psi_b, psi_c, psis=None):
    if psis is None:
        psis = [[0.0]*N_groups for _ in range(N_groups)] # numba: delete
#        psis = zeros((N_groups, N_groups)) # numba: uncomment

    mT_inv = -1.0/T

    if version == 4 or version == 5:
        T0 = 298.15
        TmT0 = T - T0
        B = T*log(T0/T) + T - T0
        for i in range(N_groups):
            a_row, b_row, c_row = psi_a[i], psi_b[i], psi_c[i]
            psisi = psis[i]
            for j in range(N_groups):
                psisi[j] = exp(mT_inv*(a_row[j] + b_row[j]*TmT0 + c_row[j]*B))
    else:
        for i in range(N_groups):
            a_row, b_row, c_row = psi_a[i], psi_b[i], psi_c[i]
            psisi = psis[i]
            for j in range(N_groups):
                psisi[j] = exp(a_row[j]*mT_inv - b_row[j] - c_row[j]*T)

    return psis


def unifac_dpsis_dT(T, N_groups, version, psi_a, psi_b, psi_c, psis, dpsis_dT=None):
    if dpsis_dT is None:
        dpsis_dT = [[0.0]*N_groups for _ in range(N_groups)] # numba: delete
#        dpsis_dT = zeros((N_groups, N_groups)) # numba: uncomment

    T2_inv = 1.0/(T*T)

    if version == 4 or version == 5:
        T0 = 298.15
        mT_inv = -1.0/T
        T2_inv = mT_inv*mT_inv
        TmT0 = T - T0
        x0 = log(T0/T)
        B = T*x0 + T - T0

        x1 = mT_inv + TmT0*T2_inv
        x2 = x0*mT_inv + B*T2_inv

        for i in range(N_groups):
            a_row, b_row, c_row, psis_row = psi_a[i], psi_b[i], psi_c[i], psis[i]
            dpsis_dTi = dpsis_dT[i]
            for j in range(N_groups):
                dpsis_dTi[j] = psis_row[j]*(x1*b_row[j] + c_row[j]*x2 + a_row[j]*T2_inv)
    else:
        for i in range(N_groups):
            a_row, c_row, psis_row = psi_a[i], psi_c[i], psis[i]
            dpsis_dTi = dpsis_dT[i]
            for j in range(N_groups):
                dpsis_dTi[j] = psis_row[j]*(a_row[j]*T2_inv - c_row[j])

    return dpsis_dT

def unifac_d2psis_dT2(T, N_groups, version, psi_a, psi_b, psi_c, psis, d2psis_dT2=None):
    if d2psis_dT2 is None:
        d2psis_dT2 = [[0.0]*N_groups for _ in range(N_groups)] # numba: delete
#        d2psis_dT2 = zeros((N_groups, N_groups)) # numba: uncomment

    mT2_inv = -1.0/(T*T)
    T3_inv_m2 = -2.0/(T*T*T)

    if version == 4 or version == 5:
        T0 = 298.15
        T_inv = 1.0/T
        T2_inv = T_inv*T_inv
        TmT0 = T - T0
        x0 = log(T0/T)
        B = T*x0 + T - T0
        for i in range(N_groups):
            psis_row, a_row, b_row, c_row = psis[i], psi_a[i], psi_b[i], psi_c[i]
            row = d2psis_dT2[i]
            for j in range(N_groups):
                # TODO: optimize
                a1, a2, a3 = a_row[j], b_row[j], c_row[j]
                tf2 = a1 + a2*(T - T0) + a3*(T*log(T0/T) + T - T0)
                tf3 = b_row[j] + c_row[j]*x0

                x1 = (tf3 - tf2*T_inv)
                v = T2_inv*psis_row[j]*(a3 + 2.0*tf3 + x1*x1 - 2.0*tf2*T_inv)
                row[j] = v
    else:
        for i in range(N_groups):
            a_row, c_row, psis_row = psi_a[i], psi_c[i], psis[i]
            d2psis_dT2i = d2psis_dT2[i]
            for j in range(N_groups):
                x0 = c_row[j] + mT2_inv*a_row[j]

                d2psis_dT2i[j] = (x0*x0 + T3_inv_m2*a_row[j])*psis_row[j]

    return d2psis_dT2

def unifac_d3psis_dT3(T, N_groups, version, psi_a, psi_b, psi_c, psis, d3psis_dT3=None):
    if d3psis_dT3 is None:
        d3psis_dT3 = [[0.0]*N_groups for _ in range(N_groups)] # numba: delete
#        d3psis_dT3 = zeros((N_groups, N_groups)) # numba: uncomment

    nT2_inv = -1.0/(T*T)
    T3_inv_6 = 6.0/(T*T*T)
    T4_inv_6 = 6.0/(T*T*T*T)

    if version == 4 or version == 5:
        T0 = 298.15
        T_inv = 1.0/T
        nT3_inv = -T_inv*T_inv*T_inv
        TmT0 = T - T0
        x0 = log(T0/T)
        B = T*x0 + T - T0
        for i in range(N_groups):
            psis_row, a_row, b_row, c_row = psis[i], psi_a[i], psi_b[i], psi_c[i]
            row = d3psis_dT3[i]
            for j in range(N_groups):
                a1, a2, a3 = a_row[j], b_row[j], c_row[j]
                tf2 = a1 + a2*TmT0 + a3*B
                tf3 = b_row[j] + c_row[j]*x0
                x6 = tf2*T_inv
                x5 = (tf3 - x6)
                v = nT3_inv*psis_row[j]*(4.0*a3 + 6.0*tf3 + x5*x5*x5
                                    + 3.0*(x5)*(a3 + tf3 + tf3 - 2.0*x6)
                                    - 6.0*x6)
                row[j] = v
    else:
        for i in range(N_groups):
            psis_row, a_row, c_row = psis[i], psi_a[i], psi_c[i]
            row = d3psis_dT3[i]
            for j in range(N_groups):
                x0 = c_row[j] + nT2_inv*a_row[j]
                row[j] = (x0*(T3_inv_6*a_row[j] - x0*x0) + T4_inv_6*a_row[j])*psis_row[j]

    return d3psis_dT3

def unifac_Vis(rs, xs, N, Vis=None):
    if Vis is None:
        Vis = [0.0]*N
    rx_sum_inv = 0.0
    for i in range(N):
        rx_sum_inv += rs[i]*xs[i]
    rx_sum_inv = 1.0/rx_sum_inv # actually inverse it
    for i in range(N):
        Vis[i] = rs[i]*rx_sum_inv
    return Vis, rx_sum_inv

def unifac_dVis_dxs(rs, rx_sum_inv, N, dVis_dxs=None):
    if dVis_dxs is None:
        dVis_dxs = [[0.0]*N for _ in range(N)] # numba: delete
#        dVis_dxs = zeros((N, N)) # numba: uncomment

    mrx_sum_inv2 = -rx_sum_inv*rx_sum_inv

    for i in range(N):
        v = rs[i]*mrx_sum_inv2
        dVijs = dVis_dxs[i]
        for j in range(N):
            dVijs[j] = v*rs[j]
    return dVis_dxs

def unifac_d2Vis_dxixjs(rs, rx_sum_inv, N, d2Vis_dxixjs=None):
    if d2Vis_dxixjs is None:
        d2Vis_dxixjs = [[[0.0]*N for _ in range(N)] for _ in range(N)] # numba: delete
#        d2Vis_dxixjs = zeros((N, N, N)) # numba: uncomment

    rx_sum_inv3_2 = 2.0*rx_sum_inv*rx_sum_inv*rx_sum_inv

    for i in range(N):
        mat = d2Vis_dxixjs[i]
        x0 = rs[i]*rx_sum_inv3_2
        for j in range(N):
            row = mat[j]
            x1 = x0*rs[j]
            for k in range(N):
                row[k] = x1*rs[k]
    return d2Vis_dxixjs

def unifac_d3Vis_dxixjxks(rs, rx_sum_inv, N, d3Vis_dxixjxks=None):
    if d3Vis_dxixjxks is None:
        d3Vis_dxixjxks = [[[[0.0]*N for _ in range(N)] for _ in range(N)] for _ in range(N)] # numba: delete
#        d3Vis_dxixjxks = zeros((N, N, N, N)) # numba: uncomment

    mrx_sum_inv4_6 = -6.0*rx_sum_inv*rx_sum_inv*rx_sum_inv*rx_sum_inv

    for i in range(N):
        cube = d3Vis_dxixjxks[i]
        x0 = rs[i]*mrx_sum_inv4_6
        for j in range(N):
            mat = cube[j]
            x1 = x0*rs[j]
            for k in range(N):
                row = mat[k]
                x2 = x1*rs[k]
                for l in range(N):
                    row[l] = x2*rs[l]
    return d3Vis_dxixjxks

def unifac_Xs(N, N_groups, xs, vs, Xs=None):
    if Xs is None:
        Xs = [0.0]*N_groups

    Xs_sum_inv = 0.0
    for i in range(N_groups):
        tot = 0.0
        vsi = vs[i]
        for j in range(N):
            tot += vsi[j]*xs[j]
        Xs[i] = tot
        Xs_sum_inv += tot
    Xs_sum_inv = 1.0/Xs_sum_inv

    for i in range(N_groups):
        Xs[i] *= Xs_sum_inv
    return Xs, Xs_sum_inv

def unifac_Thetas(N_groups, Xs, Qs, Thetas=None):
    if Thetas is None:
        Thetas = [0.0]*N_groups

    Thetas_sum_inv = 0.0
    for i in range(N_groups):
        Thetas_sum_inv += Xs[i]*Qs[i]

    Thetas_sum_inv = 1.0/Thetas_sum_inv
    for i in range(N_groups):
        Thetas[i] = Qs[i]*Xs[i]*Thetas_sum_inv
    return Thetas, Thetas_sum_inv

def unifac_dThetas_dxs(N_groups, N, Qs, vs, VS, VSXS, F, G, dThetas_dxs=None, vec0=None):
    if dThetas_dxs is None:
        dThetas_dxs = [[0.0]*N for _ in range(N_groups)] # numba: delete
#        dThetas_dxs = zeros((N_groups, N)) # numba: uncomment
    if vec0 is None:
        vec0 = [0.0]*N

    tot0 = 0.0
    for k in range(N_groups):
        tot0 += Qs[k]*VSXS[k]
    tot0*= F

    for j in range(N):
        tot1 = 0.0
        for k in range(N_groups):
            tot1 -= Qs[k]*vs[k][j]
        vec0[j] = F*(G*(tot0*VS[j] + tot1) - VS[j])

    FG = F*G

    for i in range(N_groups):
        c = FG*Qs[i]
        row = dThetas_dxs[i]
        for j in range(N):
            row[j] = c*(VSXS[i]*vec0[j] + vs[i][j])
    return dThetas_dxs

def unifac_d2Thetas_dxixjs(N_groups, N, Qs, vs, VS, VSXS, F, G, d2Thetas_dxixjs=None, vec0=None):
    if d2Thetas_dxixjs is None:
        d2Thetas_dxixjs = [[[0.0]*N_groups for _ in range(N)] for _ in range(N)] # numba: delete
#        d2Thetas_dxixjs = zeros((N, N, N_groups)) # numba: uncomment
    if vec0 is None:
        vec0 = [0.0]*N

    QsVSXS = 0.0
    for i in range(N_groups):
        QsVSXS += Qs[i]*VSXS[i]
    QsVSXS_sum_inv = 1.0/QsVSXS

    for j in range(N):
        nffVSj = -F*VS[j]
        v = 0.0
        for n in range(N_groups):
            v += Qs[n]*(nffVSj*VSXS[n] + vs[n][j])
        vec0[j] = v

    n2F = -2.0*F
    F2_2 = 2.0*F*F
    QsVSXS_sum_inv2 = 2.0*QsVSXS_sum_inv

    for j in range(N):
        matrix = d2Thetas_dxixjs[j]
        for k in range(N):
            row = matrix[k]
            n2FVsK = n2F*VS[k]
            tot0 = 0.0
            for n in range(N_groups):
                tot0 += Qs[n]*(VS[j]*(n2FVsK*VSXS[n] + vs[n][k]) + VS[k]*vs[n][j])
            tot0 = tot0*F*QsVSXS_sum_inv

            for i in range(N_groups):
#                tot0, tot1, tot2 = 0.0, 0.0, 0.0
#                for n in range(N_groups):
#                    # dep on k, j only; some sep
#                    tot0 += -2.0*F*Qs[n]*VS[j]*VS[k]*VSXS[n] + Qs[n]*VS[j]*vs[n][k] + Qs[n]*VS[k]*vs[n][j]
#                 These are each used in three places
#                    tot1 += -F*Qs[n]*VS[j]*VSXS[n] + Qs[n]*vs[n][j]
#                    tot2 += -F*Qs[n]*VS[k]*VSXS[n] + Qs[n]*vs[n][k]
                v = -F*(VS[j]*vs[i][k] + VS[k]*vs[i][j]) + VSXS[i]*tot0 + F2_2*VS[j]*VS[k]*VSXS[i]

#                v = -F*VS[j]*vs[i][k] - F*VS[k]*vs[i][j]
#                v += F2_2*VS[j]*VS[k]*VSXS[i]
#                v += VSXS[i]*tot0
#
#                v += QsVSXS_sum_inv2*VSXS[i]*vec0[j]*vec0[k]*QsVSXS_sum_inv
#
#                # For both of these duplicate terms, j goes with k; k with j
#                v -= vs[i][j]*vec0[k]*QsVSXS_sum_inv
#                v -= vs[i][k]*vec0[j]*QsVSXS_sum_inv
#
#                v += F*VS[j]*VSXS[i]*vec0[k]*QsVSXS_sum_inv
#                v += F*VS[k]*VSXS[i]*vec0[j]*QsVSXS_sum_inv

                v += QsVSXS_sum_inv*(QsVSXS_sum_inv2*VSXS[i]*vec0[j]*vec0[k]
                     - vs[i][j]*vec0[k] - vs[i][k]*vec0[j]
                     + F*VSXS[i]*(VS[j]*vec0[k] + VS[k]*vec0[j]))

                row[i] = v*Qs[i]*QsVSXS_sum_inv
    return d2Thetas_dxixjs

def unifac_VSXS(N, N_groups, vs, xs, VSXS=None):
    if VSXS is None:
        VSXS = [0.0]*N_groups
    for i in range(N_groups):
        v = 0.0
        for j in range(N):
            v += vs[i][j]*xs[j]
        VSXS[i] = v
    return VSXS

def unifac_Theta_Psi_sums(N_groups, Thetas, psis, Theta_Psi_sums=None):
    if Theta_Psi_sums is None:
        Theta_Psi_sums = [0.0]*N_groups
    for k in range(N_groups):
        tot = 0.0
        for m in range(N_groups):
            tot += Thetas[m]*psis[m][k]
        Theta_Psi_sums[k] = tot
    return Theta_Psi_sums

def unifac_ws(N, N_groups, psis, dThetas_dxs, Ws=None):
    if Ws is None:
        Ws = [[0.0]*N for _ in range(N_groups)] # numba: delete
#        Ws = zeros((N_groups, N)) # numba: uncomment

    for k in range(N_groups):
        row0 = Ws[k]
        for i in range(N):
            tot0 = 0.0
            for m in range(N_groups):
                tot0 += psis[m][k]*dThetas_dxs[m][i]
            row0[i] = tot0
    return Ws

def unifac_Theta_pure_Psi_sums(N, N_groups, psis, Thetas_pure, Theta_pure_Psi_sums=None):
    if Theta_pure_Psi_sums is None:
        Theta_pure_Psi_sums = [[0.0]*N_groups for _ in range(N)] # numba: delete
#        Theta_pure_Psi_sums = zeros((N, N_groups)) # numba: uncomment

    for i in range(N):
        row = Theta_pure_Psi_sums[i]
        Thetas_pure_i = Thetas_pure[i]
        for k in range(N_groups):
            tot = 0.0
            for m in range(N_groups):
                tot += Thetas_pure_i[m]*psis[m][k]
            row[k] = (tot)
    return Theta_pure_Psi_sums

def unifac_lnGammas_subgroups(N, N_groups, Qs, psis, Thetas, Theta_Psi_sums, Theta_Psi_sum_invs, lnGammas_subgroups=None):
    if lnGammas_subgroups is None:
        lnGammas_subgroups = [0.0]*N_groups

    for k in range(N_groups):
        psisk = psis[k]
        last = 1.0
        for m in range(N_groups):
            last -= Thetas[m]*Theta_Psi_sum_invs[m]*psisk[m]

        lnGammas_subgroups[k] = Qs[k]*(last - log(Theta_Psi_sums[k]))
    return lnGammas_subgroups

def unifac_dlnGammas_subgroups_dxs(N, N_groups, Qs, Ws, psis, Thetas, Theta_Psi_sum_invs,
                                   dThetas_dxs, dlnGammas_subgroups_dxs=None):
    if dlnGammas_subgroups_dxs is None:
        dlnGammas_subgroups_dxs = [[0.0]*N for _ in range(N_groups)] # numba : delete
#        dlnGammas_subgroups_dxs = zeros((N_groups, N)) # numba : uncomment

    for k in range(N_groups):
        row = dlnGammas_subgroups_dxs[k]
        for i in range(N):
            tot = -Ws[k][i]*Theta_Psi_sum_invs[k]
            for m in range(N_groups):
                tot -= psis[k][m]*Theta_Psi_sum_invs[m]*(dThetas_dxs[m][i]
                        - Ws[m][i]*Theta_Psi_sum_invs[m]*Thetas[m])
            row[i] = tot*Qs[k]

    return dlnGammas_subgroups_dxs

def unifac_d2lnGammas_subgroups_dTdxs(N, N_groups, Qs, Fs, Zs, Ws, psis, dpsis_dT, Thetas, dThetas_dxs, d2lnGammas_subgroups_dTdxs=None, vec0=None, vec1=None, mat0=None):
    if d2lnGammas_subgroups_dTdxs is None:
        d2lnGammas_subgroups_dTdxs = [[0.0]*N for _ in range(N_groups)] # numba : delete
#        d2lnGammas_subgroups_dTdxs = zeros((N_groups, N)) # numba : uncomment

    if mat0 is None:
        mat0 = [[0.0]*N for _ in range(N_groups)] # numba : delete
#        mat0 = zeros((N_groups, N)) # numba : uncomment
    if vec0 is None:
        vec0 = [0.0]*N_groups
    if vec1 is None:
        vec1 = [0.0]*N_groups

    # Could save this matrix but it is not used anywhere else
    for k in range(N_groups):
        row = mat0[k]
        for j in range(N):
            tot = 0.0
            for m in range(N_groups):
                tot += dThetas_dxs[m][j]*dpsis_dT[m][k]
            row[j] = tot

    for i in range(N_groups):
        vec0[i] = Zs[i]*Zs[i]
    for m in range(N_groups):
        vec1[m] = 2.0*Fs[m]*vec0[m]*Zs[m]*Thetas[m]

    for k in range(N_groups):
        row = d2lnGammas_subgroups_dTdxs[k]
        for i in range(N):
            v = Zs[k]*(mat0[k][i] - Fs[k]*Ws[k][i]*Zs[k])
            for m in range(N_groups):
                v += dThetas_dxs[m][i]*Zs[m]*(dpsis_dT[k][m] - Fs[m]*Zs[m]*psis[k][m])
                v -= vec0[m]*Thetas[m]*(mat0[m][i]*psis[k][m] + Ws[m][i]*dpsis_dT[k][m])
                v += vec1[m]*Ws[m][i]*psis[k][m]
            row[i] = -v*Qs[k]

    return d2lnGammas_subgroups_dTdxs

def unifac_d2lnGammas_subgroups_dxixjs(N, N_groups, Qs, Zs, Ws, psis, Thetas,
                                       dThetas_dxs, d2Thetas_dxixjs,
                                       d2lnGammas_subgroups_dxixjs=None, vec0=None):
    if d2lnGammas_subgroups_dxixjs is None:
        d2lnGammas_subgroups_dxixjs = [[[0.0]*N_groups for _ in range(N)] for _ in range(N)] # numba : delete
#        d2lnGammas_subgroups_dxixjs = zeros((N, N, N_groups)) # numba : uncomment

    if vec0 is None:
        vec0 = [0.0]*N_groups

    for i in range(N):
        matrix = d2lnGammas_subgroups_dxixjs[i]
        for j in range(N):
            d2Thetas_dxixjs_ij = d2Thetas_dxixjs[i][j]

            row = matrix[j]
            for k in range(N_groups):
                # d2Thetas_dxixjs_ij is why this loop can't be moved out
                totK = 0.0
                for m in range(N_groups):
                    totK += psis[m][k]*d2Thetas_dxixjs_ij[m]
                vec0[k] = totK   #K(k, i, j)
#                Krow = [K(k, i, j) for k in range(N_groups)]

            for k in range(N_groups):
                v = 0.0
                for m in range(N_groups):
                    d = d2Thetas_dxixjs_ij[m]
#                        d += (2.0*Ws[m][i]*Ws[m][j]*Zs[m] - vec0[m])*Zs[m]*Thetas[m]
#                        d -= Zs[m]*(Ws[m][j]*dThetas_dxs[m][i] + Ws[m][i]*dThetas_dxs[m][j])
                    d += Zs[m]*((2.0*Ws[m][i]*Ws[m][j]*Zs[m] - vec0[m])*Thetas[m]
                                - (Ws[m][j]*dThetas_dxs[m][i] + Ws[m][i]*dThetas_dxs[m][j]))
                    v += d*psis[k][m]*Zs[m]

                # psis[k][m] can be factored here
                v += Zs[k]*(vec0[k] - Ws[k][i]*Ws[k][j]*Zs[k])
                row[k] = (-v*Qs[k])


    return d2lnGammas_subgroups_dxixjs

def unifac_dlnGammas_subgroups_dT(N_groups, Qs, psis, dpsis_dT, Thetas,
                                  Theta_Psi_sum_invs, Theta_dPsidT_sum,
                                  dlnGammas_subgroups_dT=None):
    r'''

    .. math::
        \frac{\partial \ln \Gamma_i}{\partial T} = Q_i\left(
        \sum_j^{gr} Z(j) \left[{\theta_j \frac{\partial \psi_{i,j}}{\partial T}}
        + {\theta_j \psi_{i,j} F(j)}Z(j) \right]- F(i) Z(i)
        \right)

    .. math::
        F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}

    .. math::
        Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}

    '''
    # Theta_Psi_sum_invs = Z
    # Theta_dPsidT_sum = F
    if dlnGammas_subgroups_dT is None:
        dlnGammas_subgroups_dT = [0.0]*N_groups
    for i in range(N_groups):
        psisi, dpsis_dTi = psis[i], dpsis_dT[i]
        tot = 0.0
        for j in range(N_groups):
            tot += (psisi[j]*Theta_dPsidT_sum[j]*Theta_Psi_sum_invs[j]
                   - dpsis_dTi[j])*Theta_Psi_sum_invs[j]*Thetas[j]

        v = Qs[i]*(tot - Theta_dPsidT_sum[i]*Theta_Psi_sum_invs[i])
        dlnGammas_subgroups_dT[i] = v
    return dlnGammas_subgroups_dT

def unifac_d2lnGammas_subgroups_dT2(N_groups, Qs, psis, dpsis_dT, d2psis_dT2,
                                    Thetas, Theta_Psi_sum_invs, Theta_dPsidT_sum,
                                    Theta_d2PsidT2_sum, d2lnGammas_subgroups_dT2=None, vec0=None):
    r'''
    .. math::
        \frac{\partial^2 \ln \Gamma_i}{\partial T^2} = -Q_i\left[
        Z(i)G(i) - F(i)^2 Z(i)^2 + \sum_j\left(
        \theta_j Z(j)\frac{\partial^2 \psi_{i,j}}{\partial T}
        - Z(j)^2 \left(G(j)\theta_j \psi_{i,j} + 2 F_j \theta_j \frac{\partial \psi_{i,j}}{\partial T}\right)
        + 2Z(j)^3F(j)^2 \theta_j \psi_{i,j}
        \right)\right]

    .. math::
        F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}

    .. math::
        G(k) = \sum_m^{gr} \theta_m \frac{\partial^2 \psi_{m,k}}{\partial T^2}

    .. math::
        Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}

    '''
    if d2lnGammas_subgroups_dT2 is None:
        d2lnGammas_subgroups_dT2 = [0.0]*N_groups
    if vec0 is None:
        vec0 = [0.0]*N_groups
    Zs, Fs, Gs = Theta_Psi_sum_invs, Theta_dPsidT_sum, Theta_d2PsidT2_sum
    for j in range(N_groups):
        vec0[j] = 2.0*Fs[j]*Fs[j]*Thetas[j]*Zs[j]*Zs[j]*Zs[j]

    for i in range(N_groups):
        tot0 = 0.0
        for j in range(N_groups):
            tot0 += Zs[j]*Thetas[j]*(d2psis_dT2[i][j] - (Gs[j]*psis[i][j] + 2.0*Fs[j]*dpsis_dT[i][j])*Zs[j])
            tot0 += vec0[j]*psis[i][j]
        v = Qs[i]*(Zs[i]*(Fs[i]*Fs[i]*Zs[i] - Gs[i]) - tot0)
        d2lnGammas_subgroups_dT2[i] = v
    return d2lnGammas_subgroups_dT2

def unifac_d3lnGammas_subgroups_dT3(N_groups, Qs, psis, dpsis_dT, d2psis_dT2, d3psis_dT3,
                                  Thetas, Theta_Psi_sum_invs, Theta_dPsidT_sum,
                                  Theta_d2PsidT2_sum, Theta_d3PsidT3_sum,
                                  d3lnGammas_subgroups_dT3=None):
    r'''

    .. math::
        \frac{\partial^3 \ln \Gamma_i}{\partial T^3} =Q_i\left[-H(i) Z(i)
        - 2F(i)^3 Z(i)^3 + 3F(i) G(i) Z(i)^2+ \left(
        -\theta_j Z(j) \frac{\partial^3 \psi}{\partial T^3}
        + H(j) Z(j)^2 \theta(j)\psi_{i,j}
        - 6F(j)^2 Z(j)^3 \theta_j \frac{\partial \psi_{i,j}}{\partial T}
        + 3 F(j) Z(j)^2 \theta(j) \frac{\partial^2 \psi_{i,j}}{\partial T^2}
        ++ 3G(j) \theta(j) Z(j)^2 \frac{\partial \psi_{i,j}}{\partial T}
        + 6F(j)^3 \theta(j) Z(j)^4 \psi_{i,j}
        - 6F(j) G(j) \theta(j) Z(j)^3 \psi_{i,j}
        \right)
        \right]

    .. math::
        F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}

    .. math::
        G(k) = \sum_m^{gr} \theta_m \frac{\partial^2 \psi_{m,k}}{\partial T^2}

    .. math::
        H(k) = \sum_m^{gr} \theta_m \frac{\partial^3 \psi_{m,k}}{\partial T^3}

    .. math::
        Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}

    '''
    if d3lnGammas_subgroups_dT3 is None:
        d3lnGammas_subgroups_dT3 = [0.0]*N_groups
    # TODO optimize
    Us_inv, Fs, Gs, Hs = Theta_Psi_sum_invs, Theta_dPsidT_sum, Theta_d2PsidT2_sum, Theta_d3PsidT3_sum
    for i in range(N_groups):
        tot = 0.0
        for j in range(N_groups):
            # There is a bug in numba related to the three lines below.
            # If Theta_U is not separated out, an error is raised
            Theta_U = Thetas[j]*Us_inv[j]
            tot -= Theta_U*d3psis_dT3[i][j]
            tot += Hs[j]*Theta_U*psis[i][j]*Us_inv[j]
            
            tot -= 6.0*Fs[j]*Fs[j]*Thetas[j]*dpsis_dT[i][j]*Us_inv[j]*Us_inv[j]*Us_inv[j]
            tot += 3.0*Fs[j]*Thetas[j]*d2psis_dT2[i][j]*Us_inv[j]*Us_inv[j]

            tot += 3.0*Gs[j]*Thetas[j]*dpsis_dT[i][j]*Us_inv[j]*Us_inv[j]
            tot += 6.0*Fs[j]**3*Thetas[j]*psis[i][j]*Us_inv[j]**4
            tot -= 6.0*Fs[j]*Gs[j]*Thetas[j]*psis[i][j]*Us_inv[j]**3


        v = Qs[i]*(-Hs[i]*Us_inv[i] - 2.0*Fs[i]**3*Us_inv[i]**3 + 3.0*Fs[i]*Gs[i]*Us_inv[i]**2 + tot)
        d3lnGammas_subgroups_dT3[i] = v
    return d3lnGammas_subgroups_dT3

def unifac_Xs_pure(N, N_groups, vs, cmp_v_count_inv, Xs_pure=None):
    if Xs_pure is None:
        Xs_pure = [[0.0]*N for _ in range(N_groups)] # numba: delete
#        Xs_pure = zeros((N_groups, N)) # numba: uncomment
    for i in range(N_groups):
        row = Xs_pure[i]
        vsi = vs[i]
        for j in range(N):
            row[j] = vsi[j]*cmp_v_count_inv[j]
    return Xs_pure

def unifac_Thetas_pure(N, N_groups, Xs_pure, Qs, Thetas_pure=None):
    if Thetas_pure is None:
        Thetas_pure = [[0.0]*N_groups for _ in range(N)] # numba: delete
#        Thetas_pure = zeros((N, N_groups)) # numba: uncomment

    for i in range(N):
        # groups = self.cmp_group_idx[i]
        tot = 0.0
        row = Thetas_pure[i]
        for j in range(N_groups):
            tot += Qs[j]*Xs_pure[j][i]

        tot_inv = 1.0/tot
        for j in range(N_groups):
            row[j] = Qs[j]*Xs_pure[j][i]*tot_inv
    return Thetas_pure


def unifac_lnGammas_subgroups_pure(N, N_groups, Qs, Thetas_pure, cmp_group_idx, group_cmp_idx, psis, lnGammas_subgroups_pure=None):
    if lnGammas_subgroups_pure is None:
        lnGammas_subgroups_pure = [[0.0]*N for _ in range(N_groups)] # numba: delete
#        lnGammas_subgroups_pure = zeros((N_groups, N)) # numba: uncomment

    for k in range(N_groups):
        row = lnGammas_subgroups_pure[k]
        for i in group_cmp_idx[k]:
            groups2 = cmp_group_idx[i]
            Thetas_purei = Thetas_pure[i]

            psisk = psis[k]
            log_sum = 0.0
            for m in groups2:
                log_sum += Thetas_purei[m]*psis[m][k]
            log_sum = log(log_sum)

            last = 0.0
            for m in groups2:
                sub_subs = 0.0
                for n in range(N_groups):
                    sub_subs += Thetas_purei[n]*psis[n][m]
                last += Thetas_purei[m]*psisk[m]/sub_subs

            v = Qs[k]*(1.0 - log_sum - last)
            row[i] = v
    return lnGammas_subgroups_pure

def unifac_dlnGammas_subgroups_pure_dT(N, N_groups, Qs, psis, dpsis_dT,
                                       Thetas_pure, Theta_pure_Psi_sum_invs, Fs_pure, cmp_group_idx,
                                       dlnGammas_subgroups_pure_dT=None, vec0=None):
    if dlnGammas_subgroups_pure_dT is None:
        dlnGammas_subgroups_pure_dT = [[0.0]*N for _ in range(N_groups)] # numba: delete
#        dlnGammas_subgroups_pure_dT = zeros((N_groups, N)) # numba: uncomment

    if vec0 is None:
        vec0 = [0.0]*N_groups
    for m in range(N):
        groups2 = cmp_group_idx[m]
        Thetas = Thetas_pure[m]
        Theta_Psi_sum_invs = Theta_pure_Psi_sum_invs[m]
        Theta_dPsidT_sum = Fs_pure[m]

        vec0 = unifac_dlnGammas_subgroups_dT(N_groups, Qs, psis, dpsis_dT,
                                                    Thetas, Theta_Psi_sum_invs,
                                                    Theta_dPsidT_sum, vec0)
        for k in range(N_groups):
            if k in groups2:
                dlnGammas_subgroups_pure_dT[k][m] = vec0[k]
    return dlnGammas_subgroups_pure_dT

def unifac_d2lnGammas_subgroups_pure_dT2(N, N_groups, Qs, psis, dpsis_dT, d2psis_dT2, Thetas_pure, Theta_pure_Psi_sum_invs, Fs_pure, Gs_pure, cmp_group_idx, d2lnGammas_subgroups_pure_dT2=None, vec0=None):
    if d2lnGammas_subgroups_pure_dT2 is None:
        d2lnGammas_subgroups_pure_dT2 = [[0.0]*N for _ in range(N_groups)] # numba: delete
#        d2lnGammas_subgroups_pure_dT2 = zeros((N_groups, N)) # numba: uncomment

    if vec0 is None:
        vec0 = [0.0]*N
    for m in range(N):
        groups2 = cmp_group_idx[m]
        Thetas = Thetas_pure[m]
        Theta_Psi_sum_invs = Theta_pure_Psi_sum_invs[m]
        Theta_dPsidT_sum = Fs_pure[m]
        Theta_d2PsidT2_sum = Gs_pure[m]

        vec0 = unifac_d2lnGammas_subgroups_dT2(N_groups, Qs, psis, dpsis_dT, d2psis_dT2, Thetas, Theta_Psi_sum_invs, Theta_dPsidT_sum, Theta_d2PsidT2_sum)


        for k in range(N_groups):
            if k in groups2:
                d2lnGammas_subgroups_pure_dT2[k][m] = vec0[k]
    return d2lnGammas_subgroups_pure_dT2

def unifac_d3lnGammas_subgroups_pure_dT3(N, N_groups, Qs, psis, dpsis_dT, d2psis_dT2, d3psis_dT3, Thetas_pure, Theta_pure_Psi_sum_invs, Fs_pure, Gs_pure, Hs_pure, cmp_group_idx, d3lnGammas_subgroups_pure_dT3=None, vec0=None):
    if d3lnGammas_subgroups_pure_dT3 is None:
        d3lnGammas_subgroups_pure_dT3 = [[0.0]*N for _ in range(N_groups)] # numba: delete
#        d3lnGammas_subgroups_pure_dT3 = zeros((N_groups, N)) # numba: uncomment

    if vec0 is None:
        vec0 = [0.0]*N
    for m in range(N):
        groups2 = cmp_group_idx[m]
        Thetas = Thetas_pure[m]
        Theta_Psi_sum_invs = Theta_pure_Psi_sum_invs[m]
        Theta_dPsidT_sum = Fs_pure[m]
        Theta_d2PsidT2_sum = Gs_pure[m]
        Theta_d3PsidT3_sum = Hs_pure[m]

        row = unifac_d3lnGammas_subgroups_dT3(N_groups, Qs, psis, dpsis_dT, d2psis_dT2, d3psis_dT3, Thetas, Theta_Psi_sum_invs, Theta_dPsidT_sum, Theta_d2PsidT2_sum, Theta_d3PsidT3_sum)

        for k in range(N_groups):
            if k in groups2:
                d3lnGammas_subgroups_pure_dT3[k][m] = row[k]
    return d3lnGammas_subgroups_pure_dT3

def unifac_lngammas_r(N, N_groups, lnGammas_subgroups_pure, lnGammas_subgroups, vs, lngammas_r=None):
    if lngammas_r is None:
        lngammas_r = [0.0]*N

    for i in range(N):
        tot = 0.0
        for k in range(N_groups):
            tot += vs[k][i]*(lnGammas_subgroups[k] - lnGammas_subgroups_pure[k][i])
        lngammas_r[i] = tot
    return lngammas_r

def unifac_dlngammas_r_dxs(N, N_groups, vs, dlnGammas_subgroups_dxs, dlngammas_r_dxs=None):
    if dlngammas_r_dxs is None:
        dlngammas_r_dxs = [[0.0]*N for _ in range(N)] # numba : delete
#        dlngammas_r_dxs = zeros((N, N)) # numba : uncomment
    for i in range(N):
        row = dlngammas_r_dxs[i]
        for j in range(N):
            tot = 0.0
            for m in range(N_groups):
                tot += vs[m][i]*dlnGammas_subgroups_dxs[m][j]
            row[j] = tot
    return dlngammas_r_dxs


def unifac_d2lngammas_r_dxixjs(N, N_groups, vs, d2lnGammas_subgroups_dxixjs, d2lngammas_r_dxixjs=None):
    if d2lngammas_r_dxixjs is None:
        d2lngammas_r_dxixjs = [[[0.0]*N for _ in range(N)] for _ in range(N)] # numba : delete
#        d2lngammas_r_dxixjs = zeros((N, N, N)) # numba : uncomment

    for i in range(N):
        matrix = d2lngammas_r_dxixjs[i]
        for j in range(N):
            row = matrix[j]
            for k in range(N):
                tot = 0.0
                r = d2lnGammas_subgroups_dxixjs[j][k]
                for m in range(N_groups):
                    tot += vs[m][i]*r[m]
                row[k] = tot
    return d2lngammas_r_dxixjs

def unifac_GE(T, xs, N, lngammas_r, lngammas_c):
    GE = 0.0
    for i in range(N):
        GE += xs[i]*(lngammas_c[i] + lngammas_r[i])
    GE *= R*T
    return GE

def unifac_GE_skip_comb(T, xs, N, lngammas_r):
    GE = 0.0
    for i in range(N):
        GE += xs[i]*lngammas_r[i]
    GE *= R*T
    return GE

def unifac_dGE_dxs(T, xs, N, lngammas_r, dlngammas_r_dxs, lngammas_c, dlngammas_c_dxs, dGE_dxs=None):
    if dGE_dxs is None:
        dGE_dxs = [0.0]*N
    RT = R*T
    for i in range(N):
        dGE = lngammas_r[i] + lngammas_c[i]
        for j in range(N):
            dGE += xs[j]*(dlngammas_c_dxs[j][i] + dlngammas_r_dxs[j][i])
        dGE_dxs[i] = dGE*RT
    return dGE_dxs

def unifac_dGE_dxs_skip_comb(T, xs, N, lngammas_r, dlngammas_r_dxs, dGE_dxs=None):
    if dGE_dxs is None:
        dGE_dxs = [0.0]*N
    RT = R*T
    for i in range(N):
        dGE = lngammas_r[i]
        for j in range(N):
            dGE += xs[j]*(dlngammas_r_dxs[j][i])
        dGE_dxs[i] = dGE*RT
    return dGE_dxs


def unifac_d2GE_dTdxs(T, xs, N, lngammas_r, dlngammas_r_dxs, dlngammas_r_dT, d2lngammas_r_dTdxs, lngammas_c, dlngammas_c_dxs, d2GE_dTdxs=None):
    if d2GE_dTdxs is None:
        d2GE_dTdxs = [0.0]*N
    for i in range(N):
        dGE = lngammas_r[i] + lngammas_c[i] + T*dlngammas_r_dT[i]
        for j in range(N):
            dGE += xs[j]*(dlngammas_c_dxs[j][i] + dlngammas_r_dxs[j][i])
            dGE += T*xs[j]*d2lngammas_r_dTdxs[j][i] # ji should be consistent in all of them

        d2GE_dTdxs[i] = dGE*R
    return d2GE_dTdxs

def unifac_d2GE_dTdxs_skip_comb(T, xs, N, lngammas_r, dlngammas_r_dxs,dlngammas_r_dT, d2lngammas_r_dTdxs, d2GE_dTdxs=None):
    if d2GE_dTdxs is None:
        d2GE_dTdxs = [0.0]*N
    for i in range(N):
        dGE = lngammas_r[i] + T*dlngammas_r_dT[i]
        for j in range(N):
            dGE += xs[j]*(dlngammas_r_dxs[j][i] + T*d2lngammas_r_dTdxs[j][i])
        d2GE_dTdxs[i] = dGE*R
    return d2GE_dTdxs

def unifac_d2GE_dxixjs(T, xs, N, dlngammas_r_dxs, d2lngammas_r_dxixjs, dlngammas_c_dxs, d2lngammas_c_dxixjs, d2GE_dxixjs=None):
    if d2GE_dxixjs is None:
        d2GE_dxixjs = [[0.0]*N for _ in range(N)] # numba: delete
#        d2GE_dxixjs = zeros((N, N)) # numba: uncomment
    RT = R*T
    for i in range(N):
        row = d2GE_dxixjs[i]
        for j in range(N):
            dGE = dlngammas_c_dxs[i][j] + dlngammas_r_dxs[i][j]
            dGE += dlngammas_c_dxs[j][i] + dlngammas_r_dxs[j][i]

            for k in range(N):
                dGE += xs[k]*(d2lngammas_c_dxixjs[k][i][j] + d2lngammas_r_dxixjs[k][i][j])
            row[j] = dGE*RT

    return d2GE_dxixjs

def unifac_d2GE_dxixjs_skip_comb(T, xs, N, dlngammas_r_dxs, d2lngammas_r_dxixjs, d2GE_dxixjs=None):
    if d2GE_dxixjs is None:
        d2GE_dxixjs = [[0.0]*N for _ in range(N)] # numba: delete
#        d2GE_dxixjs = zeros((N, N)) # numba: uncomment
    RT = R*T
    for i in range(N):
        row = d2GE_dxixjs[i]
        for j in range(N):
            dGE =  dlngammas_r_dxs[i][j] + dlngammas_r_dxs[j][i]
            for k in range(N):
                dGE += xs[k]*d2lngammas_r_dxixjs[k][i][j]
            row[j] = dGE*RT

    return d2GE_dxixjs

def unifac_dGE_dT(N, T, xs, dlngammas_r_dT, GE):
    tot = 0.0
    for i in range(N):
        tot += xs[i]*dlngammas_r_dT[i]
    return R*T*tot + GE/T

def unifac_d2GE_dT2(N, T, xs, dlngammas_r_dT, d2lngammas_r_dT2):
    tot0, tot1 = 0.0, 0.0
    for i in range(N):
        tot0 += xs[i]*d2lngammas_r_dT2[i]
        tot1 += xs[i]*dlngammas_r_dT[i] # This line same as the dGE_dT

    return R*(T*tot0 + (tot1 + tot1))

def unifac_d3GE_dT3(N, T, xs, d2lngammas_r_dT2, d3lngammas_r_dT3):
    tot0, tot1 = 0.0, 0.0
    for i in range(N):
        tot0 += xs[i]*d3lngammas_r_dT3[i]
        tot1 += xs[i]*d2lngammas_r_dT2[i] # This line same as the d2GE_dT2

    return R*(T*tot0 + 3.0*tot1)

def unifac_gammas(N, xs, lngammas_r, lngammas_c, gammas=None):
    if gammas is None:
        gammas = [0.0]*N

    for i in range(N):
        gammas[i] = exp(lngammas_r[i] + lngammas_c[i])
    return gammas

def unifac_gammas_at_T(xs, N, N_groups, vs, rs, qs, Qs, 
                         psis, lnGammas_subgroups_pure,# Depends on T only
                         version, rs_34,
                         gammas=None):
    skip_comb = version == 3

    Xs, Xs_sum_inv = unifac_Xs(N=N, N_groups=N_groups, xs=xs, vs=vs)
    Thetas, Thetas_sum_inv = unifac_Thetas(N_groups=N_groups, Xs=Xs, Qs=Qs)
    Theta_Psi_sums = unifac_Theta_Psi_sums(N_groups=N_groups, Thetas=Thetas, psis=psis)
    
    Theta_Psi_sum_invs = [0.0]*N_groups
    for i in range(N_groups):
        Theta_Psi_sum_invs[i] = 1.0/Theta_Psi_sums[i]
    lnGammas_subgroups = unifac_lnGammas_subgroups(N=N, N_groups=N_groups, Qs=Qs, psis=psis, Thetas=Thetas, 
                                                   Theta_Psi_sums=Theta_Psi_sums, Theta_Psi_sum_invs=Theta_Psi_sum_invs)
    lngammas_r = unifac_lngammas_r(N=N, N_groups=N_groups, lnGammas_subgroups_pure=lnGammas_subgroups_pure,
                                   lnGammas_subgroups=lnGammas_subgroups, vs=vs)
    
    if gammas is None:
        gammas = [0.0]*N
    if skip_comb:
        for i in range(N):
            gammas[i] = exp(lngammas_r[i])
    else:
        Vis, rx_sum_inv = unifac_Vis(rs=rs, xs=xs, N=N)
        Fis, qx_sum_inv = unifac_Vis(rs=qs, xs=xs, N=N)
        
        if version == 1 or version == 3 or version == 4:
            Vis_modified, r34x_sum_inv = unifac_Vis(rs=rs_34, xs=xs, N=N)
        else:
            Vis_modified = Vis
        
        lngammas_c = unifac_lngammas_c(N=N, version=version, qs=qs, Fis=Fis, Vis=Vis, Vis_modified=Vis_modified)
     
        for i in range(N):
            gammas[i] = exp(lngammas_r[i] + lngammas_c[i])
    return gammas

def unifac_dgammas_dxs(N, xs, gammas, dlngammas_r_dxs, dlngammas_c_dxs, dgammas_dxs=None):
    if dgammas_dxs is None:
        dgammas_dxs = [[0.0]*N for _ in range(N)] # numba: delete
#        dgammas_dxs = zeros((N, N)) # numba: uncomment

    for i in range(N):
        dlngammas_r_dxsi = dlngammas_r_dxs[i]
        dlngammas_c_dxsi = dlngammas_c_dxs[i]
        dgammas_dxsi = dgammas_dxs[i]
        for j in range(N):
            dgammas_dxsi[j] = (dlngammas_r_dxsi[j] + dlngammas_c_dxsi[j])*gammas[i]
    return dgammas_dxs


def unifac_dgammas_dxs_skip_comb(N, xs, gammas, dlngammas_r_dxs, dgammas_dxs=None):
    if dgammas_dxs is None:
        dgammas_dxs = [[0.0]*N for _ in range(N)] # numba: delete
#        dgammas_dxs = zeros((N, N)) # numba: uncomment

    for i in range(N):
        dlngammas_r_dxsi = dlngammas_r_dxs[i]
        dgammas_dxsi = dgammas_dxs[i]
        for j in range(N):
            dgammas_dxsi[j] = dlngammas_r_dxsi[j]*gammas[i]
    return dgammas_dxs

def unifac_dgammas_dns(N, xs, dgammas_dxs, dgammas_dns=None):
    if dgammas_dns is None:
        dgammas_dns = [[0.0]*N for _ in range(N)] # numba: delete
#        dgammas_dns = zeros((N, N)) # numba: uncomment

    for i in range(N):
        row = dgammas_dns[i]
        dgammas_dxsi = dgammas_dxs[i]
        xdx_tot = 0.0
        for j in range(N):
            xdx_tot += xs[j]*dgammas_dxsi[j]
        for j in range(N):
            row[j] = dgammas_dxsi[j] - xdx_tot
    return dgammas_dns


def unifac_lngammas_c(N, version, qs, Fis, Vis, Vis_modified, lngammas_c=None):
    if lngammas_c is None:
        lngammas_c = [0.0]*N

    if version == 4:
        for i in range(N):
            r = Vis_modified[i] # In the definition of V' used here, there is no mole fraction division needed
            val = log(r) + 1.0 - r
            lngammas_c[i] = val
    else:
        for i in range(N):
            Vi_Fi = Vis[i]/Fis[i]
            val = (1.0 - Vis_modified[i] + log(Vis_modified[i])
                    - 5.0*qs[i]*(1.0 - Vi_Fi + log(Vi_Fi)))
            lngammas_c[i] = val
    return lngammas_c

def unifac_dlngammas_c_dxs(N, version, qs, Fis, dFis_dxs, Vis, dVis_dxs, Vis_modified, dVis_modified_dxs, dlngammas_c_dxs=None):
    if dlngammas_c_dxs is None:
        dlngammas_c_dxs = [[0.0]*N for _ in range(N)] # numba: delete
#        dlngammas_c_dxs = zeros((N, N)) # numba: uncomment

    if version == 4:
        for i in range(N):
            row = dlngammas_c_dxs[i]
            for j in range(N):
                v = -dVis_modified_dxs[i][j] + dVis_modified_dxs[i][j]/Vis_modified[i]
                row[j] = v
    else:
        for i in range(N):
            row = dlngammas_c_dxs[i]
            Fi_inv = 1.0/Fis[i]
            for j in range(N):
                val = -5.0*qs[i]*((dVis_dxs[i][j] - Vis[i]*dFis_dxs[i][j]*Fi_inv)/Vis[i]
                - dVis_dxs[i][j]*Fi_inv + Vis[i]*dFis_dxs[i][j]*Fi_inv*Fi_inv
                ) - dVis_modified_dxs[i][j] + dVis_modified_dxs[i][j]/Vis_modified[i]
                row[j] = val
    return dlngammas_c_dxs


def unifac_d2lngammas_c_dxixjs(N, version, qs, Fis, dFis_dxs, d2Fis_dxixjs, Vis, dVis_dxs, d2Vis_dxixjs, Vis_modified, dVis_modified_dxs, d2Vis_modified_dxixjs, d2lngammas_c_dxixjs=None):
    if d2lngammas_c_dxixjs is None:
        d2lngammas_c_dxixjs = [[[0.0]*N for _ in range(N)] for _ in range(N)]# numba: delete
#        d2lngammas_c_dxixjs = zeros((N, N, N)) # numba: uncomment

    if version == 4:
        for i in range(N):
            Vi = Vis_modified[i]
            matrix = d2lngammas_c_dxixjs[i]
            for j in range(N):
                row = matrix[j]
                for k in range(N):
                    val = -d2Vis_modified_dxixjs[i][j][k] + 1.0/Vi*d2Vis_modified_dxixjs[i][j][k]
                    val -= 1.0/Vi**2*dVis_modified_dxs[i][j]*dVis_modified_dxs[i][k]
                    row[k] = val

    else:
        for i in range(N):
            Vi = Vis[i]
            qi = qs[i]
            ViD = Vis_modified[i]
            ViD_inv2 = 1.0/(ViD*ViD)
            Fi = Fis[i]
            x1 = 1.0/Fi
            x4 = x1*x1
            Fi_inv3 = x1*x1*x1
            x5 = Vis[i]*x4
            x15 = 1.0/Vi
            Vi_inv2 = x15*x15
            matrix = d2lngammas_c_dxixjs[i]
            for j in range(N):
                x6 = dFis_dxs[i][j]
                x10 = dVis_dxs[i][j]
                dViD_dxj = dVis_modified_dxs[i][j]
                row = matrix[j]
                for k in range(N):
                    x0 = d2Vis_modified_dxixjs[i][j][k]
                    x2 = d2Vis_dxixjs[i][j][k]
                    x3 = d2Fis_dxixjs[i][j][k]
                    x7 = dVis_dxs[i][k]
                    dViD_dxk = dVis_modified_dxs[i][k]
                    x8 = x6*x7
                    x9 = dFis_dxs[i][k]
                    x11 = x10*x9
                    x12 = 2.0*x6*x9

                    x13 = Vi*x1
                    x14 = x10 - x13*x6

                    val = (5.0*qi*(-x1*x14*x15*x9 + x1*x2 - x11*x4
                                    + x15*(x1*x11 + x1*x8 - x12*x5 + x13*x3 - x2)
                                    - x3*x5 - x4*x8 + x14*x7*Vi_inv2 + Vi*x12*Fi_inv3)
                            - x0 + x0/ViD - dViD_dxj*dViD_dxk*ViD_inv2
                            )
                    row[k] = val
    return d2lngammas_c_dxixjs

def unifac_d3lngammas_c_dxixjxks(N, version, qs, Fis, dFis_dxs, d2Fis_dxixjs, d3Fis_dxixjxks, Vis, dVis_dxs, d2Vis_dxixjs, d3Vis_dxixjxks, Vis_modified, dVis_modified_dxs, d2Vis_modified_dxixjs, d3Vis_modified_dxixjxks, d3lngammas_c_dxixjxks=None):
    if d3lngammas_c_dxixjxks is None:
        d3lngammas_c_dxixjxks = [[[[0.0]*N for _ in range(N)] for _ in range(N)] for _ in range(N)] # numba: delete
#        d3lngammas_c_dxixjxks = zeros((N, N, N, N)) # numba: uncomment

    if version == 4:
        for i in range(N):
            Vi = Vis_modified[i]
            third = d3lngammas_c_dxixjxks[i]
            for j in range(N):
                hess = third[j]
                for k in range(N):
                    row = hess[k]
                    for m in range(N):
                        val = d3Vis_modified_dxixjxks[i][j][k][m]*(1.0/Vi - 1.0)
                        val-= 1.0/Vi**2*  (dVis_modified_dxs[i][j]*d2Vis_modified_dxixjs[i][k][m]
                                         + dVis_modified_dxs[i][k]*d2Vis_modified_dxixjs[i][j][m]
                                         + dVis_modified_dxs[i][m]*d2Vis_modified_dxixjs[i][j][k])

                        val += 2.0/Vi**3*dVis_modified_dxs[i][j]*dVis_modified_dxs[i][k]*dVis_modified_dxs[i][m]

                        row[m] = val
    else:
        for i in range(N):
            Vi = Vis[i]
            ViD = Vis_modified[i]
            Fi = Fis[i]
            qi = qs[i]
            third = d3lngammas_c_dxixjxks[i]
            for j in range(N):
                hess = third[j]
                for k in range(N):
                    row = hess[k]
                    for m in range(N):
                        x0 = d3Vis_modified_dxixjxks[i][j][k][m]#Derivative(ViD, xj, xk, xm)
                        x1 = 1/Fis[i]#1/Fi
                        x2 = 5.0*qs[i]
                        x3 = x2*d3Fis_dxixjxks[i][j][k][m]#Derivative(Fi, xj, xk, xm)
                        x4 = x2*d3Vis_dxixjxks[i][j][k][m]#Derivative(Vi, xj, xk, xm)
                        x5 = Vis_modified[i]**-2#ViD**(-2)
                        x6 = dVis_modified_dxs[i][j]#Derivative(ViD, xj)
                        x7 = dVis_modified_dxs[i][k]#Derivative(ViD, xk)
                        x8 = dVis_modified_dxs[i][m]#Derivative(ViD, xm)
                        x9 = Fis[i]**-2#Fi**(-2)
                        x10 = x2*x9
                        x11 = dFis_dxs[i][j]#Derivative(Fi, xj)
                        x12 = d2Fis_dxixjs[i][k][m]#Derivative(Fi, xk, xm)
                        x13 = x11*x12
                        x14 = d2Vis_dxixjs[i][k][m]#Derivative(Vi, xk, xm)
                        x15 = d2Fis_dxixjs[i][j][m]#Derivative(Fi, xj, xm)
                        x16 = dFis_dxs[i][k]#Derivative(Fi, xk)
                        x17 = x10*x16
                        x18 = d2Vis_dxixjs[i][j][m]#Derivative(Vi, xj, xm)
                        x19 = d2Fis_dxixjs[i][j][k]#Derivative(Fi, xj, xk)
                        x20 = dFis_dxs[i][m]#Derivative(Fi, xm)
                        x21 = x10*x20
                        x22 = d2Vis_dxixjs[i][j][k]#Derivative(Vi, xj, xk)
                        x23 = dVis_dxs[i][j]#Derivative(Vi, xj)
                        x24 = dVis_dxs[i][k]#Derivative(Vi, xk)
                        x25 = dVis_dxs[i][m]#Derivative(Vi, xm)
                        x26 = x2/Vis[i]**2
                        x27 = Fis[i]**(-3)
                        x28 = 10*qs[i]
                        x29 = x27*x28
                        x30 = Vi*x29
                        x31 = x11*x16
                        x32 = x20*x29
                        x33 = x25*x28
                        val = (-Vi*x3*x9 - x0 + x1*x3 + x1*x4 - x10*x11*x14 - x10*x12*x23
                               - x10*x13 - x10*x15*x24 - x10*x19*x25 + x11*x24*x32 + x13*x30
                               + x14*x23*x26 + x15*x16*x30 - x15*x17 + x16*x23*x32 - x17*x18
                               + x18*x24*x26 + x19*x20*x30 - x19*x21 - x21*x22 + x22*x25*x26
                               + x27*x31*x33 + x31*x32 - x5*x6*d2Vis_modified_dxixjs[i][k][m]
                               - x5*x7*d2Vis_modified_dxixjs[i][j][m]
                               - x5*x8*d2Vis_modified_dxixjs[i][j][k]
                               + x0/ViD + 2*x6*x7*x8/ViD**3 - x4/Vi - x23*x24*x33/Vi**3 - 30*Vi*qi*x20*x31/Fi**4)

                        row[m] = val
    return d3lngammas_c_dxixjxks

class UNIFAC(GibbsExcess):
    r'''Class for representing an a liquid with excess gibbs energy represented
    by the UNIFAC equation. This model is capable of representing VL and LL
    behavior, provided the correct interaction parameters are used. [1]_ and
    [2]_ are good references on this model.

    Parameters
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]
    rs : list[float]
        `r` parameters :math:`r_i = \sum_{k=1}^{n} \nu_k R_k`, [-]
    qs : list[float]
        `q` parameters :math:`q_i = \sum_{k=1}^{n}\nu_k Q_k`, [-]
    Qs : list[float]
        `Q` parameter for each subgroup; subgroups are not required to but are
        suggested to be sorted from lowest number to highest number, [-]
    vs : list[list[float]]
        Indexed by [subgroup][count], this variable is the count of each
        subgroups in each compound, [-]
    psi_abc : tuple(list[list[float]], 3), optional
        `psi` interaction parameters between each subgroup; indexed
        [subgroup][subgroup], not symmetrical; first arg is the matrix for `a`,
        then `b`, and then `c`. Only one of `psi_abc` or `psi_coeffs` is
        required, [-]
    psi_coeffs : list[list[tuple(float, 3)]], optional
        `psi` interaction parameters between each subgroup; indexed
        [subgroup][subgroup][letter], not symmetrical. Only one of `psi_abc`
        or `psi_coeffs` is required, [-]
    version : int, optional
        Which version of the model to use [-]

        * 0 - original UNIFAC, OR UNIFAC LLE
        * 1 - Dortmund UNIFAC (adds T dept, 3/4 power)
        * 2 - PSRK (original with T dept function)
        * 3 - VTPR (drops combinatorial term, Dortmund UNIFAC otherwise)
        * 4 - Lyngby/Larsen has different combinatorial, 2/3 power
        * 5 - UNIFAC KT (2 params for psi, Lyngby/Larsen formulation;
          otherwise same as original)

    Attributes
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]

    Notes
    -----
    In addition to the methods presented here, the methods of its base class
    :obj:`thermo.activity.GibbsExcess` are available as well.

    Examples
    --------
    The DDBST has published numerous sample problems using UNIFAC; a simple
    binary system from example P05.22a in [2]_ with n-hexane and butanone-2
    is shown below:

    >>> from thermo.unifac import UFIP, UFSG
    >>> GE = UNIFAC.from_subgroups(chemgroups=[{1:2, 2:4}, {1:1, 2:1, 18:1}], T=60+273.15, xs=[0.5, 0.5], version=0, interaction_data=UFIP, subgroups=UFSG)
    >>> GE.gammas()
    [1.4276025835, 1.3646545010]
    >>> GE.GE(), GE.dGE_dT(), GE.d2GE_dT2()
    (923.641197, 0.206721488, -0.00380070204)
    >>> GE.HE(), GE.SE(), GE.dHE_dT(), GE.dSE_dT()
    (854.77193363, -0.2067214889, 1.266203886, 0.0038007020460)

    The solution given by the DDBST has the same values [1.428, 1.365],
    and can be found here:
    http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/05.22a%20VLE%20of%20Hexane-Butanone-2%20Via%20UNIFAC%20-%20Step%20by%20Step.xps

    References
    ----------
    .. [1] Poling, Bruce E., John M. Prausnitz, and John P. O’Connell. The
       Properties of Gases and Liquids. 5th edition. New York: McGraw-Hill
       Professional, 2000.
    .. [2] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
       Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
    '''

    '''
    List of functions not fully optimized (for Python anyway)
    psis second derivative

    lnGammas_subgroups_pure - "in" array check is approx. 2x as slow in numba
    as coding the flor loop to check it

    unifac_d3lnGammas_subgroups_dT3 has no optimization
    d2lnGammas_subgroups_dT2 can have one more vec to save time
    unifac_dlnGammas_subgroups_dT can have one more vec to save time
    unifac_dlnGammas_subgroups_dxs can have one more vec to save time


    '''
    @property
    def model_id(self):
        '''A unique numerical identifier refering to the thermodynamic model
        being implemented. For internal use.
        '''
        return self.version + 500
    
    def lnphis_args(self):
        try:
            rs_34 = self.rs_34
        except:
            rs_34 = self.rs
        return (self.N_groups, self.vs, self.rs, self.qs, self.Qs, 
             self.psis(), self.lnGammas_subgroups_pure(),# Depends on T only
             self.version, rs_34)

    @staticmethod
    def from_subgroups(T, xs, chemgroups, subgroups=None,
                       interaction_data=None, version=0):
        r'''Method to construct a UNIFAC object from a dictionary of
        interaction parameters parameters and a list of dictionaries of UNIFAC keys.
        As the actual implementation is matrix based not dictionary based, this method
        can be quite convenient.

        Parameters
        ----------
        T : float
            Temperature, [K]
        xs : list[float]
            Mole fractions, [-]
        chemgroups : list[dict]
            List of dictionaries of subgroup IDs and their counts for all species
            in the mixture, [-]
        subgroups : dict[int: UNIFAC_subgroup], optional
            UNIFAC subgroup data; available dictionaries in this module include
            UFSG (original), DOUFSG (Dortmund), or NISTUFSG. The default depends
            on the given `version`, [-]
        interaction_data : dict[int: dict[int: tuple(a_mn, b_mn, c_mn)]], optional
            UNIFAC interaction parameter data; available dictionaries in this
            module include UFIP (original), DOUFIP2006 (Dortmund parameters
            published in 2006), DOUFIP2016 (Dortmund parameters published in
            2016), and NISTUFIP. The default depends on the given `version`, [-]
        version : int, optional
            Which version of the model to use. Defaults to 0, [-]

            * 0 - original UNIFAC, OR UNIFAC LLE
            * 1 - Dortmund UNIFAC (adds T dept, 3/4 power)
            * 2 - PSRK (original with T dept function)
            * 3 - VTPR (drops combinatorial term, Dortmund UNIFAC otherwise)
            * 4 - Lyngby/Larsen has different combinatorial, 2/3 power
            * 5 - UNIFAC KT (2 params for psi, Lyngby/Larsen formulation;
              otherwise same as original)

        Returns
        -------
        UNIFAC : UNIFAC
            Object for performing calculations with the UNIFAC activity
            coefficient model, [-]

        Warning
        -------
        For version 0, the interaction data and subgroups default to the 
        original UNIFAC model (not LLE). 
        
        For version 1, the interaction data defaults to the Dortmund parameters
        publshed in 2016 (not 2006).

        Examples
        --------
        Mixture of ['benzene', 'cyclohexane', 'acetone', 'ethanol']
        according to the Dortmund UNIFAC model:

        >>> from thermo.unifac import DOUFIP2006, DOUFSG
        >>> T = 373.15
        >>> xs = [0.2, 0.3, 0.1, 0.4]
        >>> chemgroups = [{9: 6}, {78: 6}, {1: 1, 18: 1}, {1: 1, 2: 1, 14: 1}]
        >>> GE = UNIFAC.from_subgroups(T=T, xs=xs, chemgroups=chemgroups, version=1, interaction_data=DOUFIP2006, subgroups=DOUFSG)
        >>> GE
        UNIFAC(T=373.15, xs=[0.2, 0.3, 0.1, 0.4], rs=[2.2578, 4.2816, 2.3373, 2.4951999999999996], qs=[2.5926, 5.181, 2.7308, 2.6616], Qs=[1.0608, 0.7081, 0.4321, 0.8927, 1.67, 0.8635], vs=[[0, 0, 1, 1], [0, 0, 0, 1], [6, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 6, 0, 0]], psi_abc=([[0.0, 0.0, 114.2, 2777.0, 433.6, -117.1], [0.0, 0.0, 114.2, 2777.0, 433.6, -117.1], [16.07, 16.07, 0.0, 3972.0, 146.2, 134.6], [1606.0, 1606.0, 3049.0, 0.0, -250.0, 3121.0], [199.0, 199.0, -57.53, 653.3, 0.0, 168.2], [170.9, 170.9, -2.619, 2601.0, 464.5, 0.0]], [[0.0, 0.0, 0.0933, -4.674, 0.1473, 0.5481], [0.0, 0.0, 0.0933, -4.674, 0.1473, 0.5481], [-0.2998, -0.2998, 0.0, -13.16, -1.237, -1.231], [-4.746, -4.746, -12.77, 0.0, 2.857, -13.69], [-0.8709, -0.8709, 1.212, -1.412, 0.0, -0.8197], [-0.8062, -0.8062, 1.094, -1.25, 0.1542, 0.0]], [[0.0, 0.0, 0.0, 0.001551, 0.0, -0.00098], [0.0, 0.0, 0.0, 0.001551, 0.0, -0.00098], [0.0, 0.0, 0.0, 0.01208, 0.004237, 0.001488], [0.0009181, 0.0009181, 0.01435, 0.0, -0.006022, 0.01446], [0.0, 0.0, -0.003715, 0.000954, 0.0, 0.0], [0.001291, 0.001291, -0.001557, -0.006309, 0.0, 0.0]]), version=1)
        '''
        if subgroups is None:
            if version == 0:
                subgroups = UFSG
            elif version == 1:
                subgroups = DOUFSG
            elif version == 2:
                subgroups = PSRKSG
            elif version == 3:
                subgroups = VTPRSG
            elif version == 4:
                subgroups = LUFSG
            elif version == 5:
                subgroups = NISTKTUFSG
            else:
                raise ValueError("'version' must be a number from 0 to 5")
        if interaction_data is None:
            if not _unifac_ip_loaded: load_unifac_ip()
            if version == 0:
                interaction_data = UFIP
            elif version == 1:
                interaction_data = DOUFIP2016
            elif version == 2:
                interaction_data = PSRKIP
            elif version == 3:
                interaction_data = VTPRIP
            elif version == 4:
                interaction_data = LUFIP
            elif version == 5:
                interaction_data = NISTKTUFIP
            else:
                raise ValueError("'version' must be a number from 0 to 5")

        scalar = type(xs) is list
        rs = []
        qs = []
        for groups in chemgroups:
            ri = 0.
            qi = 0.
            for subgroup_idx, count in groups.items():
                if version != 3:
                    ri += subgroups[subgroup_idx].R*count

                qi += subgroups[subgroup_idx].Q*count
            rs.append(ri)
            qs.append(qi)

        # Make a dictionary containing the subgroups as keys and their count - the total number
        # of them in ALL the compounds, regardless of mole fractions
        group_counts = {}
        for compound_subgroups in chemgroups:
            for subgroup, count in compound_subgroups.items():
                try:
                    group_counts[subgroup] += count
                except KeyError:
                    group_counts[subgroup] = count

        # Convert group counts into a list, sorted by index (lowest subgroup index is first element, highest subgroup index is the last)
        subgroup_list = list(sorted(group_counts.keys()))
        group_counts_list = [c for _, c in sorted(zip(group_counts.keys(), group_counts.values()))]

        Qs = [subgroups[group].Q for group in subgroup_list]
        vs = chemgroups_to_matrix(chemgroups)

        psi_a, psi_b, psi_c = [], [], []
        for sub1 in subgroup_list:
            a_row, b_row, c_row = [], [], []
            for sub2 in subgroup_list:
                main1 = subgroups[sub1].main_group_id
                main2 = subgroups[sub2].main_group_id
                try:
                    v = interaction_data[main1][main2]
                    try:
                        a_row.append(v[0])
                        b_row.append(v[1])
                        c_row.append(v[2])
                    except:
                        a_row.append(v)
                        b_row.append(0.0)
                        c_row.append(0.0)
                except KeyError:
                        a_row.append(0.0)
                        b_row.append(0.0)
                        c_row.append(0.0)
            psi_a.append(a_row), psi_b.append(b_row), psi_c.append(c_row)


#        debug = (rs, qs, Qs, vs, (psi_a, psi_b, psi_c))
        if scalar:
            return UNIFAC(T=T, xs=xs, rs=rs, qs=qs, Qs=Qs, vs=vs, psi_abc=(psi_a, psi_b, psi_c), version=version)
        return UNIFAC(T=T, xs=xs, rs=array(rs), qs=array(qs), Qs=array(Qs), vs=array(vs), psi_abc=(array(psi_a), array(psi_b), array(psi_c)), version=version)

    model_attriubtes = ('rs', 'qs', 'psi_a', 'psi_b', 'psi_c', 'version')

    def __repr__(self):  # pragma: no cover

        psi_abc = (self.psi_a, self.psi_b, self.psi_c)
        s = 'UNIFAC('
        s += 'T=%s, xs=%s, rs=%s, qs=%s' %(self.T, self.xs, self.rs, self.qs)
        s += ', Qs=%s, vs=%s, psi_abc=%s, version=%s' %(self.Qs, self.vs,
                                                        psi_abc, self.version)
        s += ')'
        return s


    def __init__(self, T, xs, rs, qs, Qs, vs, psi_coeffs=None, psi_abc=None,
                 version=0):
        self.T = T
        self.xs = xs
        self.scalar = scalar = type(xs) is list

        # rs - 1d index by [component] parameter, calculated using the chemical's subgroups and their count
        self.rs = rs
        # qs - 1d index by [component] parameter, calculated using the chemical's subgroups and their count
        self.qs = qs
        self.Qs = Qs

        # [subgroup][component] = number of subgroup in component where subgroup
        # is an index, numbered sequentially by the number of subgroups in the mixture
        self.vs = vs


        # each psi_letter is a matrix of [subgroup_length][subgroups_length]
        # the diagonal is zero
        # Indexed by index of the subgroup in the mixture, again sorted lowest first
        if psi_abc is not None:
            self.psi_a, self.psi_b, self.psi_c = psi_abc

        else:
            if psi_coeffs is None:
                raise ValueError("Missing psis")
            self.psi_a = [[i[0] for i in l] for l in psi_coeffs]
            self.psi_b = [[i[1] for i in l] for l in psi_coeffs]
            self.psi_c = [[i[2] for i in l] for l in psi_coeffs]
        self.N_groups = N_groups = len(self.psi_a)
        self.N = N = len(rs)
        self.version = version
        self.skip_comb = version == 3

        if self.version == 1:
            power = 0.75
            if scalar:
                self.rs_34 = [ri**power for ri in rs]
            else:
                self.rs_34 = rs**0.75
        elif self.version == 4:
            power = 2.0/3.0 # Lyngby
            # works in the various functions without change as never taking the der w.r.t. r
            if scalar:
                self.rs_34 = [ri**power for ri in rs]
            else:
                self.rs_34 = rs**power

        self.cmp_v_count = cmp_v_count = []
        for i in range(N):
            tot = 0
            for group in range(N_groups):
                tot += vs[group][i]
            cmp_v_count.append(tot)
        if scalar:
            cmp_v_count_inv = [1.0/ni for ni in cmp_v_count]
        else:
            self.cmp_v_count = cmp_v_count = array(cmp_v_count)
            cmp_v_count_inv = 1.0/cmp_v_count
        self.cmp_v_count_inv = cmp_v_count_inv

        # Matrix of [component][list(indexes to groups in component)], list of list
        cmp_group_idx = [[j for j in range(N_groups) if vs[j][i]] for i in range(N)]
        # TODO figure out the best way to handle this with numba
        # as each array was supposedly a different shape
        if not scalar:
            cmp_group_idx = tuple(array(v) for v in cmp_group_idx)
        self.cmp_group_idx = cmp_group_idx


        group_cmp_idx = []
        for k in range(N_groups):
            temp = []
            for i in range(N):
                groups2 = cmp_group_idx[i]
                if k in groups2:
                    temp.append(i)
            group_cmp_idx.append(temp)
        if not scalar:
            group_cmp_idx = tuple(array(v) for v in group_cmp_idx)
        self.group_cmp_idx = group_cmp_idx

        # Calculate the composition and temperature independent parameters on initialization
        self.Thetas_pure()
        self.Xs_pure()

    def to_T_xs(self, T, xs):
        r'''Method to construct a new :obj:`UNIFAC` instance at
        temperature `T`, and mole fractions `xs`
        with the same parameters as the existing object.

        Parameters
        ----------
        T : float
            Temperature, [K]
        xs : list[float]
            Mole fractions of each component, [-]

        Returns
        -------
        obj : UNIFAC
            New :obj:`UNIFAC` object at the specified conditions [-]

        Notes
        -----
        If the new temperature is the same temperature as the existing
        temperature, if the `psi` terms or their derivatives have been
        calculated, they will be set to the new object as well.
        If the mole fractions are the same, various subgroup terms are also
        kept.
        '''
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.N = self.N
        new.scalar = self.scalar

        new.N_groups = self.N_groups

        new.rs = self.rs
        new.qs = self.qs
        new.Qs = self.Qs
        new.vs = self.vs
        new.cmp_v_count = self.cmp_v_count
        new.cmp_v_count_inv = self.cmp_v_count_inv
        new.cmp_group_idx = self.cmp_group_idx
        new.group_cmp_idx = self.group_cmp_idx

        new.version = self.version
        new.skip_comb = self.skip_comb

        new.psi_a, new.psi_b, new.psi_c = self.psi_a, self.psi_b, self.psi_c

        try:
            new.rs_34 = self.rs_34
        except AttributeError:
            pass

        new._Thetas_pure = self._Thetas_pure
        new._Xs_pure = self._Xs_pure
        if T == self.T:
            # interaction parameters that depend on T only
            try:
                new._psis = self._psis
            except AttributeError:
                pass
            try:
                new._dpsis_dT = self._dpsis_dT
            except AttributeError:
                pass
            try:
                new._d2psis_dT2 = self._d2psis_dT2
            except AttributeError:
                pass
            try:
                new._d3psis_dT3 = self._d3psis_dT3
            except AttributeError:
                pass

            # pure parameters that depend on T only
            try:
                new._lnGammas_subgroups_pure = self._lnGammas_subgroups_pure
            except AttributeError:
                pass
            try:
                new._dlnGammas_subgroups_pure_dT = self._dlnGammas_subgroups_pure_dT
            except AttributeError:
                pass
            try:
                new._d2lnGammas_subgroups_pure_dT2 = self._d2lnGammas_subgroups_pure_dT2
            except AttributeError:
                pass
            try:
                new._d3lnGammas_subgroups_pure_dT3 = self._d3lnGammas_subgroups_pure_dT3
            except AttributeError:
                pass
        if (self.scalar and xs == self.xs) or (not self.scalar and array_equal(xs, self.xs)):
            try:
                new._Fis = self._Fis
            except AttributeError:
                pass
            try:
                new.qx_sum_inv = self.qx_sum_inv
            except AttributeError:
                pass
            try:
                new._dFis_dxs = self._dFis_dxs
            except AttributeError:
                pass
            try:
                new._d2Fis_dxixjs = self._d2Fis_dxixjs
            except AttributeError:
                pass
            try:
                new._d3Fis_dxixjxks = self._d3Fis_dxixjxks
            except AttributeError:
                pass
            try:
                new._Vis_modified = self._Vis_modified
                new.r34x_sum_inv = self.r34x_sum_inv
            except AttributeError:
                pass
            try:
                new._dVis_modified_dxs = self._dVis_modified_dxs
            except AttributeError:
                pass
            try:
                new._d2Vis_modified_dxixjs = self._d2Vis_modified_dxixjs
            except AttributeError:
                pass
            try:
                new._d3Vis_modified_dxixjxks = self._d3Vis_modified_dxixjxks
            except AttributeError:
                pass
            # composition dependent - parameters not using psis
            try:
                new._Xs = self._Xs
            except AttributeError:
                pass
            try:
                new._Thetas = self._Thetas
                new.Thetas_sum_inv = self.Thetas_sum_inv
            except AttributeError:
                pass
            try:
                new.Xs_sum_inv = self.Xs_sum_inv
            except AttributeError:
                pass
            try:
                new._dThetas_dxs = self._dThetas_dxs
            except AttributeError:
                pass
            try:
                new._d2Thetas_dxixjs = self._d2Thetas_dxixjs
            except AttributeError:
                pass
            try:
                new._lngammas_c = self._lngammas_c
            except AttributeError:
                pass
            try:
                new._dlngammas_c_dxs = self._dlngammas_c_dxs
            except AttributeError:
                pass
            try:
                new._d2lngammas_c_dxixjs = self._d2lngammas_c_dxixjs
            except AttributeError:
                pass
            try:
                new._d3lngammas_c_dxixjxks = self._d3lngammas_c_dxixjxks
            except AttributeError:
                pass

        # gammas, theta_psi_sums, _theta_psi_sum_inv, lngammas_subgroups, lngammas_r
        # SHOULD NOT be moved to a new class - use the same class if T and x is the same!

        return new


    def psis(self):
        r'''Calculate the :math:`\Psi` term matrix for all groups interacting
        with all other groups.

        The main model calculates it as a function of three coefficients;

        .. math::
            \Psi_{mn} = \exp\left(\frac{-a_{mn} - b_{mn}T - c_{mn}T^2}{T}\right)

        Only the first, `a` coefficient, is used in the original UNIFAC model
        as well as the UNIFAC-LLE model, so the expression simplifies to:

        .. math::
            \Psi_{mn} = \exp\left(\frac{-a_{mn}}{T}\right)

        For the Lyngby model, the temperature dependence is modified slightly,
        as follows:

        .. math::
            \Psi_{mk} = e^{\frac{- a_{1} - a_{2} \left(T - T_{0}\right) - a_{3}
            \left(T \ln{\left(\frac{T_{0}}{T} \right)} + T - T_{0}\right)}{T}}

        with :math:`T_0 = 298.15` K and the `a` coefficients are specific to
        each pair of main groups, and they are asymmetric, so
        :math:`a_{0,mk} \ne a_{0,km}`.

        Returns
        -------
        psis : list[list[float]]
            `psi` terms, size subgroups x subgroups [-]
        '''
        try:
            return self._psis
        except AttributeError:
            pass
        T, N_groups = self.T, self.N_groups
#        mT_inv = -1.0/T
        psi_a, psi_b, psi_c = self.psi_a, self.psi_b, self.psi_c
        if self.scalar:
            psis = [[0.0]*N_groups for _ in range(N_groups)]
        else:
            psis = zeros((N_groups, N_groups))

        self._psis = unifac_psis(T, N_groups, self.version, psi_a, psi_b, psi_c, psis)
        return psis

    def dpsis_dT(self):
        r'''Calculate the :math:`\Psi` term first temperature derivative
        matrix for all groups interacting with all other groups.

        The main model calculates the derivative as a function of three
        coefficients;

        .. math::
            \frac{\partial \Psi_{mn}}{\partial T} = \left(\frac{- 2 T c_{mn}
            - b_{mn}}{T} - \frac{- T^{2} c_{mn} - T b_{mn} - a_{mn}}{T^{2}}
            \right) e^{\frac{- T^{2} c_{mn} - T b_{mn} - a_{mn}}{T}}

        Only the first, `a` coefficient, is used in the original UNIFAC model
        as well as the UNIFAC-LLE model, so the expression simplifies to:

        .. math::
            \frac{\partial \Psi_{mn}}{\partial T} = \frac{a_{mn}
            e^{- \frac{a_{mn}}{T}}}{T^{2}}

        For the Lyngby model, the first temperature derivative is:

        .. math::
            \frac{\partial \Psi_{mk}}{\partial T} = \left(\frac{- a_{2} - a_{3}
            \ln{\left(\frac{T_{0}}{T} \right)}}{T} - \frac{- a_{1} - a_{2}
            \left(T - T_{0}\right) - a_{3} \left(T \ln{\left(\frac{T_{0}}{T}
            \right)} + T - T_{0}\right)}{T^{2}}\right) e^{\frac{- a_{1} - a_{2}
            \left(T - T_{0}\right) - a_{3} \left(T \ln{\left(\frac{T_{0}}{T}
            \right)} + T - T_{0}\right)}{T}}

        with :math:`T_0 = 298.15` K and the `a` coefficients are specific to
        each pair of main groups, and they are asymmetric, so
        :math:`a_{0,mk} \ne a_{0,km}`.

        Returns
        -------
        dpsis_dT : list[list[float]]
            First temperature derivative of`psi` terms, size subgroups x
            subgroups [-]
        '''
        try:
            return self._dpsis_dT
        except AttributeError:
            pass
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()

        T, N_groups = self.T, self.N_groups
        psi_a, psi_b, psi_c = self.psi_a, self.psi_b, self.psi_c

        if self.scalar:
            dpsis_dT = [[0.0]*N_groups for _ in range(N_groups)]
        else:
            dpsis_dT = zeros((N_groups, N_groups))

        self._dpsis_dT = unifac_dpsis_dT(T, N_groups, self.version, psi_a, psi_b, psi_c, psis, dpsis_dT)
        return dpsis_dT

    def d2psis_dT2(self):
        r'''Calculate the :math:`\Psi` term second temperature derivative
        matrix for all groups interacting with all other groups.

        The main model calculates the derivative as a function of three
        coefficients;

        .. math::
            \frac{\partial^2 \Psi_{mn}}{\partial T^2} = \frac{\left(- 2 c_{mn}
            + \frac{2 \left(2 T c_{mn} + b_{mn}\right)}{T} + \frac{\left(2 T
            c_{mn} + b_{mn} - \frac{T^{2} c_{mn} + T b_{mn} + a_{mn}}{T}
            \right)^{2}}{T} - \frac{2 \left(T^{2} c_{mn} + T b_{mn} + a_{mn}
            \right)}{T^{2}}\right) e^{- \frac{T^{2} c_{mn} + T b_{mn} + a_{mn}}
            {T}}}{T}

        Only the first, `a` coefficient, is used in the original UNIFAC model
        as well as the UNIFAC-LLE model, so the expression simplifies to:

        .. math::
            \frac{\partial^2 \Psi_{mn}}{\partial T^2} = \frac{a_{mn} \left(-2
            + \frac{a_{mn}}{T}\right) e^{- \frac{a_{mn}}{T}}}{T^{3}}

        For the Lyngby model, the second temperature derivative is:

        .. math::
            \frac{\partial^2 \Psi_{mk}}{\partial T^2} = \frac{\left(2 a_{2}
            + 2 a_{3} \ln{\left(\frac{T_{0}}{T} \right)} + a_{3} + \left(a_{2}
            + a_{3} \ln{\left(\frac{T_{0}}{T} \right)} - \frac{a_{1} + a_{2}
            \left(T - T_{0}\right) + a_{3} \left(T \ln{\left(\frac{T_{0}}{T}
            \right)} + T - T_{0}\right)}{T}\right)^{2} - \frac{2 \left(a_{1}
            + a_{2} \left(T - T_{0}\right) + a_{3} \left(T \ln{\left(
            \frac{T_{0}}{T} \right)} + T - T_{0}\right)\right)}{T}\right)
            e^{- \frac{a_{1} + a_{2} \left(T - T_{0}\right) + a_{3} \left(
            T \ln{\left(\frac{T_{0}}{T} \right)} + T - T_{0}\right)}{T}}}
            {T^{2}}

        with :math:`T_0 = 298.15` K and the `a` coefficients are specific to
        each pair of main groups, and they are asymmetric, so
        :math:`a_{0,mk} \ne a_{0,km}`.

        Returns
        -------
        d2psis_dT2 : list[list[float]]
            Second temperature derivative of`psi` terms, size subgroups x
            subgroups [-]
        '''
        try:
            return self._d2psis_dT2
        except AttributeError:
            pass
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()

        T, N_groups = self.T, self.N_groups
        psi_a, psi_b, psi_c = self.psi_a, self.psi_b, self.psi_c

        if self.scalar:
            d2psis_dT2 = [[0.0]*N_groups for _ in range(N_groups)]
        else:
            d2psis_dT2 = zeros((N_groups, N_groups))

        self._d2psis_dT2 = unifac_d2psis_dT2(T, N_groups, self.version, psi_a, psi_b, psi_c, psis, d2psis_dT2)
        return d2psis_dT2


    def d3psis_dT3(self):
        r'''Calculate the :math:`\Psi` term third temperature derivative
        matrix for all groups interacting with all other groups.

        The main model calculates the derivative as a function of three
        coefficients;

        .. math::
            \frac{\partial^3 \Psi_{mn}}{\partial T^3} = \frac{\left(6 c_{mn}
            + 6 \left(c_{mn} - \frac{2 T c_{mn} + b_{mn}}{T} + \frac{T^{2}
            c_{mn} + T b_{mn} + a_{mn}}{T^{2}}\right) \left(2 T c_{mn} + b_{mn}
                - \frac{T^{2} c_{mn} + T b_{mn} + a_{mn}}{T}\right) - \frac{6
            \left(2 T c_{mn} + b_{mn}\right)}{T} - \frac{\left(2 T c_{mn}
            + b_{mn} - \frac{T^{2} c_{mn} + T b_{mn} + a_{mn}}{T}\right)^{3}}
            {T} + \frac{6 \left(T^{2} c_{mn} + T b_{mn} + a_{mn}\right)}{T^{2}}
            \right) e^{- \frac{T^{2} c_{mn} + T b_{mn} + a_{mn}}{T}}}{T^{2}}

        Only the first, `a` coefficient, is used in the original UNIFAC model
        as well as the UNIFAC-LLE model, so the expression simplifies to:

        .. math::
            \frac{\partial^3 \Psi_{mn}}{\partial T^3} = \frac{a_{mn} \left(6
            - \frac{6 a_{mn}}{T} + \frac{a_{mn}^{2}}{T^{2}}\right) e^{-
            \frac{a_{mn}}{T}}}{T^{4}}

        For the Lyngby model, the third temperature derivative is:

        .. math::
            \frac{\partial^3 \Psi_{mk}}{\partial T^3} =
            - \frac{\left(6 a_{2} + 6 a_{3} \ln{\left(\frac{T_{0}}{T} \right)}
            + 4 a_{3} + \left(a_{2} + a_{3} \ln{\left(\frac{T_{0}}{T} \right)}
            - \frac{a_{1} + a_{2} \left(T - T_{0}\right) + a_{3} \left(T \ln{
            \left(\frac{T_{0}}{T} \right)} + T - T_{0}\right)}{T}\right)^{3}
            + 3 \left(a_{2} + a_{3} \ln{\left(\frac{T_{0}}{T} \right)}
            - \frac{a_{1} + a_{2} \left(T - T_{0}\right) + a_{3} \left(T \ln{
            \left(\frac{T_{0}}{T} \right)} + T - T_{0}\right)}{T}\right) \left(
            2 a_{2} + 2 a_{3} \ln{\left(\frac{T_{0}}{T} \right)} + a_{3}
            - \frac{2 \left(a_{1} + a_{2} \left(T - T_{0}\right) + a_{3} \left(
            T \ln{\left(\frac{T_{0}}{T} \right)} + T - T_{0}\right)\right)}{T}
            \right) - \frac{6 \left(a_{1} + a_{2} \left(T - T_{0}\right)
            + a_{3} \left(T \ln{\left(\frac{T_{0}}{T} \right)} + T - T_{0}
            \right)\right)}{T}\right) e^{- \frac{a_{1} + a_{2} \left(T - T_{0}
            \right) + a_{3} \left(T \ln{\left(\frac{T_{0}}{T} \right)}
            + T - T_{0}\right)}{T}}}{T^{3}}

        with :math:`T_0 = 298.15` K and the `a` coefficients are specific to
        each pair of main groups, and they are asymmetric, so
        :math:`a_{0,mk} \ne a_{0,km}`.

        Returns
        -------
        d3psis_dT3 : list[list[float]]
            Third temperature derivative of`psi` terms, size subgroups x
            subgroups [-]
        '''
        try:
            return self._d3psis_dT3
        except AttributeError:
            pass
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()

        T, N_groups = self.T, self.N_groups
        psi_a, psi_b, psi_c = self.psi_a, self.psi_b, self.psi_c

        if self.scalar:
            d3psis_dT3 = [[0.0]*N_groups for _ in range(N_groups)]
        else:
            d3psis_dT3 = zeros((N_groups, N_groups))

        self._d3psis_dT3 = unifac_d3psis_dT3(T, N_groups, self.version, psi_a, psi_b, psi_c, psis, d3psis_dT3)
        return d3psis_dT3

    def Vis(self):
        r'''Calculate the :math:`V_i` terms used in calculating the
        combinatorial part. A function of mole fractions and the parameters
        `r` only.

        .. math::
            V_i = \frac{r_i}{\sum_j r_j x_j}

        This is used in the UNIFAC, UNIFAC-LLE, UNIFAC Dortmund, UNIFAC-NIST,
        and PSRK models.

        Returns
        -------
        Vis : list[float]
            `V` terms size number of components, [-]
        '''
        try:
            return self._Vis
        except:
            pass
        rs, xs, N = self.rs, self.xs, self.N
        if self.scalar:
            Vis = [0.0]*N
        else:
            Vis = zeros(N)

        self._Vis, self.rx_sum_inv = unifac_Vis(rs, xs, N, Vis)
        return Vis

    def dVis_dxs(self):
        r'''Calculate the mole fraction derivative of the :math:`V_i` terms
        used in calculating the combinatorial part. A function of mole
        fractions and the parameters `r` only.

        .. math::
            \frac{\partial V_i}{\partial x_j} = -r_i r_j V_{sum}^2

        .. math::
            V_{sum} = \frac{1}{\sum_j r_j x_j}

        This is used in the UNIFAC, UNIFAC-LLE, UNIFAC Dortmund, UNIFAC-NIST,
        and PSRK models.

        Returns
        -------
        dVis_dxs : list[list[float]]
            `V` terms size number of components by number of components, [-]
        '''
        try:
            return self._dVis_dxs
        except AttributeError:
            pass
        try:
            rx_sum_inv = self.rx_sum_inv
        except AttributeError:
            self.Vis()
            rx_sum_inv = self.rx_sum_inv

        rs, N = self.rs, self.N
        if self.scalar:
            dVis_dxs = [[0.0]*N for _ in range(N)]
        else:
            dVis_dxs = zeros((N, N))
        self._dVis_dxs = unifac_dVis_dxs(rs, rx_sum_inv, N, dVis_dxs)
        return dVis_dxs

    def d2Vis_dxixjs(self):
        r'''Calculate the second mole fraction derivative of the :math:`V_i`
        terms used in calculating the combinatorial part. A function of mole
        fractions and the parameters `r` only.

        .. math::
            \frac{\partial V_i}{\partial x_j \partial x_k} =
            2 r_i r_j r_k V_{sum}^3

        .. math::
            V_{sum} = \frac{1}{\sum_j r_j x_j}

        This is used in the UNIFAC, UNIFAC-LLE, UNIFAC Dortmund, UNIFAC-NIST,
        and PSRK models.

        Returns
        -------
        d2Vis_dxixjs : list[list[list[float]]]
            `V` terms size number of components by number of components by
            number of components, [-]
        '''
        try:
            return self._d2Vis_dxixjs
        except AttributeError:
            pass
        try:
            rx_sum_inv = self.rx_sum_inv
        except AttributeError:
            self.Vis()
            rx_sum_inv = self.rx_sum_inv
        rs, N = self.rs, self.N
        if self.scalar:
            d2Vis_dxixjs = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d2Vis_dxixjs = zeros((N, N, N))

        self._d2Vis_dxixjs = unifac_d2Vis_dxixjs(rs, rx_sum_inv, N, d2Vis_dxixjs)
        return d2Vis_dxixjs

    def d3Vis_dxixjxks(self):
        r'''Calculate the third mole fraction derivative of the :math:`V_i`
        terms used in calculating the combinatorial part. A function of mole
        fractions and the parameters `r` only.

        .. math::
            \frac{\partial V_i}{\partial x_j \partial x_k \partial x_m} =
            -6 r_i r_j r_k r_m V_{sum}^4

        .. math::
            V_{sum} = \frac{1}{\sum_j r_j x_j}

        This is used in the UNIFAC, UNIFAC-LLE, UNIFAC Dortmund, UNIFAC-NIST,
        and PSRK models.

        Returns
        -------
        d3Vis_dxixjxks : list[list[list[list[float]]]]
            `V` terms size number of components by number of components by
            number of components by number of components, [-]
        '''
        try:
            return self._d3Vis_dxixjxks
        except AttributeError:
            pass
        try:
            rx_sum_inv = self.rx_sum_inv
        except AttributeError:
            self.Vis()
            rx_sum_inv = self.rx_sum_inv
        rs, N = self.rs, self.N
        if self.scalar:
            d3Vis_dxixjxks = [[[[0.0]*N for _ in range(N)] for _ in range(N)] for _ in range(N)]
        else:
            d3Vis_dxixjxks = zeros((N, N, N, N))
        self._d3Vis_dxixjxks = unifac_d3Vis_dxixjxks(rs, rx_sum_inv, N, d3Vis_dxixjxks)
        return d3Vis_dxixjxks

    def Fis(self):
        r'''Calculate the :math:`F_i` terms used in calculating the
        combinatorial part. A function of mole fractions and the parameters
        `q` only.

        .. math::
            F_i = \frac{q_i}{\sum_j q_j x_j}

        This is used in the UNIFAC, UNIFAC-LLE, UNIFAC Dortmund, UNIFAC-NIST,
        and PSRK models.

        Returns
        -------
        Fis : list[float]
            `F` terms size number of components, [-]
        '''
        try:
            return self._Fis
        except AttributeError:
            pass
        qs, xs, N = self.qs, self.xs, self.N
        if self.scalar:
            Fis = [0.0]*N
        else:
            Fis = zeros(N)

        self._Fis, self.qx_sum_inv = unifac_Vis(qs, xs, N, Fis)
        return Fis

    def dFis_dxs(self):
        r'''Calculate the mole fraction derivative of the :math:`F_i` terms
        used in calculating the combinatorial part. A function of mole
        fractions and the parameters `q` only.

        .. math::
            \frac{\partial F_i}{\partial x_j} = -q_i q_j G_{sum}^2

        .. math::
            G_{sum} = \frac{1}{\sum_j q_j x_j}

        This is used in the UNIFAC, UNIFAC-LLE, UNIFAC Dortmund, UNIFAC-NIST,
        and PSRK models.

        Returns
        -------
        dFis_dxs : list[list[float]]
            `F` terms size number of components by number of components, [-]
        '''
        try:
            return self._dFis_dxs
        except AttributeError:
            pass
        try:
            qx_sum_inv = self.qx_sum_inv
        except AttributeError:
            self.Fis()
            qx_sum_inv = self.qx_sum_inv

        qs, N = self.qs, self.N


        if self.scalar:
            dFis_dxs  = [[0.0]*N for _ in range(N)]
        else:
            dFis_dxs  = zeros((N, N))
        self._dFis_dxs = unifac_dVis_dxs(qs, qx_sum_inv, N, dFis_dxs)
        return dFis_dxs

    def d2Fis_dxixjs(self):
        r'''Calculate the second mole fraction derivative of the :math:`F_i`
        terms used in calculating the combinatorial part. A function of mole
        fractions and the parameters `q` only.

        .. math::
            \frac{\partial F_i}{\partial x_j \partial x_k} =
            2 q_i q_j q_k G_{sum}^3

        .. math::
            G_{sum} = \frac{1}{\sum_j q_j x_j}

        This is used in the UNIFAC, UNIFAC-LLE, UNIFAC Dortmund, UNIFAC-NIST,
        and PSRK models.

        Returns
        -------
        d2Fis_dxixjs : list[list[list[float]]]
            `F` terms size number of components by number of components by
            number of components, [-]
        '''
        try:
            return self._d2Fis_dxixjs
        except AttributeError:
            pass
        try:
            qx_sum_inv = self.qx_sum_inv
        except AttributeError:
            self.Fis()
            qx_sum_inv = self.qx_sum_inv

        qs, N = self.qs, self.N

        if self.scalar:
            d2Fis_dxixjs = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d2Fis_dxixjs = zeros((N, N, N))

        self._d2Fis_dxixjs  = unifac_d2Vis_dxixjs(qs, qx_sum_inv, N, d2Fis_dxixjs )
        return d2Fis_dxixjs

    def d3Fis_dxixjxks(self):
        r'''Calculate the third mole fraction derivative of the :math:`F_i`
        terms used in calculating the combinatorial part. A function of mole
        fractions and the parameters `q` only.

        .. math::
            \frac{\partial F_i}{\partial x_j \partial x_k \partial x_m} =
            -6 q_i q_j q_k q_m G_{sum}^4

        .. math::
            G_{sum} = \frac{1}{\sum_j q_j x_j}

        This is used in the UNIFAC, UNIFAC-LLE, UNIFAC Dortmund, UNIFAC-NIST,
        and PSRK models.

        Returns
        -------
        d3Fis_dxixjxks : list[list[list[list[float]]]]
            `F` terms size number of components by number of components by
            number of components by number of components, [-]
        '''
        try:
            return self._d3Fis_dxixjxks
        except AttributeError:
            pass
        try:
            qx_sum_inv = self.qx_sum_inv
        except AttributeError:
            self.Fis()
            qx_sum_inv = self.qx_sum_inv
        qs, N = self.qs, self.N

        if self.scalar:
            d3Fis_dxixjxks = [[[[0.0]*N for _ in range(N)] for _ in range(N)] for _ in range(N)]
        else:
            d3Fis_dxixjxks = zeros((N, N, N, N))
        self._d3Fis_dxixjxks = unifac_d3Vis_dxixjxks(qs, qx_sum_inv, N, d3Fis_dxixjxks)
        return d3Fis_dxixjxks

    def Vis_modified(self):
        r'''Calculate the :math:`V_i'` terms used in calculating the
        combinatorial part. A function of mole fractions and the parameters
        `r` only.

        .. math::
            V_i' = \frac{r_i^n}{\sum_j r_j^n x_j}

        This is used in the UNIFAC Dortmund and UNIFAC-NIST model with
        n=0.75, and the Lyngby model with n=2/3.

        Returns
        -------
        Vis_modified : list[float]
            Modified `V` terms size number of components, [-]
        '''
        try:
            return self._Vis_modified
        except:
            pass
        rs_34, xs, N = self.rs_34, self.xs, self.N

        if self.scalar:
            Vis_modified = [0.0]*N
        else:
            Vis_modified = zeros(N)

        self._Vis_modified, self.r34x_sum_inv = unifac_Vis(rs_34, xs, N, Vis_modified)
        return Vis_modified

    def dVis_modified_dxs(self):
        r'''Calculate the mole fraction derivative of the :math:`V_i'` terms
        used in calculating the combinatorial part. A function of mole
        fractions and the parameters `r` only.

        .. math::
            \frac{\partial V_i'}{\partial x_j} = -r_i^n r_j^n V_{sum}^2

        .. math::
            V_{sum} = \frac{1}{\sum_j r_j^n x_j}

        This is used in the UNIFAC Dortmund and UNIFAC-NIST model with
        n=0.75, and the Lyngby model with n=2/3.

        Returns
        -------
        dVis_modified_dxs : list[list[float]]
            `V'` terms size number of components by number of components, [-]
        '''
        try:
            return self._dVis_modified_dxs
        except AttributeError:
            pass
        try:
            r34x_sum_inv = self.r34x_sum_inv
        except AttributeError:
            self.Vis_modified()
            r34x_sum_inv = self.r34x_sum_inv

        rs_34, N = self.rs_34, self.N
        if self.scalar:
            dVis_modified = [[0.0]*N for _ in range(N)]
        else:
            dVis_modified = zeros((N, N))
        self._dVis_modified_dxs = unifac_dVis_dxs(rs_34, r34x_sum_inv, N, dVis_modified)
        return dVis_modified

    def d2Vis_modified_dxixjs(self):
        r'''Calculate the second mole fraction derivative of the :math:`V_i'`
        terms used in calculating the combinatorial part. A function of mole
        fractions and the parameters `r` only.

        .. math::
            \frac{\partial V_i'}{\partial x_j \partial x_k} =
            2 r_i^n r_j^n r_k^n V_{sum}^3

        .. math::
            V_{sum} = \frac{1}{\sum_j r_j^n x_j}

        This is used in the UNIFAC Dortmund and UNIFAC-NIST model with
        n=0.75, and the Lyngby model with n=2/3.

        Returns
        -------
        d2Vis_modified_dxixjs : list[list[list[float]]]
            `V'` terms size number of components by number of components by
            number of components, [-]
        '''
        try:
            return self._d2Vis_modified_dxixjs
        except AttributeError:
            pass
        try:
            r34x_sum_inv = self.r34x_sum_inv
        except AttributeError:
            self.Vis_modified()
            r34x_sum_inv = self.r34x_sum_inv
        rs_34, N = self.rs_34, self.N

        if self.scalar:
            d2Vis_modified = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d2Vis_modified = zeros((N, N, N))

        self._d2Vis_modified_dxixjs = unifac_d2Vis_dxixjs(rs_34, r34x_sum_inv, N, d2Vis_modified)
        return d2Vis_modified

    def d3Vis_modified_dxixjxks(self):
        r'''Calculate the third mole fraction derivative of the :math:`V_i'`
        terms used in calculating the combinatorial part. A function of mole
        fractions and the parameters `r` only.

        .. math::
            \frac{\partial V_i'}{\partial x_j \partial x_k \partial x_m} =
            -6 r_i^n r_j^n r_k^n r_m^n V_{sum}^4

        .. math::
            V_{sum} = \frac{1}{\sum_j r_j x_j}

        This is used in the UNIFAC Dortmund and UNIFAC-NIST model with
        n=0.75, and the Lyngby model with n=2/3.

        Returns
        -------
        d3Vis_modified_dxixjxks : list[list[list[list[float]]]]
            `V'` terms size number of components by number of components by
            number of components by number of components, [-]
        '''
        try:
            return self._d3Vis_modified_dxixjxks
        except AttributeError:
            pass
        try:
            r34x_sum_inv = self.r34x_sum_inv
        except AttributeError:
            self.Vis_modified()
            r34x_sum_inv = self.r34x_sum_inv
        rs_34, N = self.rs_34, self.N

        if self.scalar:
            d3Vis_modified = [[[[0.0]*N for _ in range(N)] for _ in range(N)] for _ in range(N)]
        else:
            d3Vis_modified = zeros((N, N, N, N))
        self._d3Vis_modified_dxixjxks = unifac_d3Vis_dxixjxks(rs_34, r34x_sum_inv, N, d3Vis_modified)
        return d3Vis_modified

    def Xs(self):
        r'''Calculate the :math:`X_m` parameters
        used in calculating the residual part. A function of mole
        fractions and group counts only.

        .. math::
            X_m = \frac{ \sum_j \nu^j_m x_j}{\sum_j \sum_n \nu_n^j x_j}

        Returns
        -------
        Xs : list[float]
           :math:`X_m` terms, size number of subgroups, [-]
        '''
        try:
            return self._Xs
        except AttributeError:
            pass
        # [subgroup][component] = number of subgroup in component where subgroup
        # is an index, numbered sequentially by the number of subgroups in the mixture
        vs, xs = self.vs, self.xs
        N, N_groups = self.N, self.N_groups
        if self.scalar:
            Xs = [0.0]*N_groups
        else:
            Xs = zeros(N_groups)

        self._Xs, self.Xs_sum_inv = unifac_Xs(N, N_groups, xs, vs, Xs)
        return Xs

    def _Xs_sum_inv(self):
        try:
            return self.Xs_sum_inv
        except AttributeError:
            self.Xs()
            return self.Xs_sum_inv

    def Thetas(self):
        r'''Calculate the :math:`\Theta_m` parameters
        used in calculating the residual part. A function of mole
        fractions and group counts only.

        .. math::
            \Theta_m = \frac{Q_m X_m}{\sum_{n} Q_n X_n}

        Returns
        -------
        Thetas : list[float]
           :math:`\Theta_m` terms, size number of subgroups, [-]
        '''
        try:
            return self._Thetas
        except AttributeError:
            pass
        Qs, N_groups = self.Qs, self.N_groups
        try:
            Xs = self._Xs
        except AttributeError:
            Xs = self.Xs()

        if self.scalar:
            Thetas = [0.0]*N_groups
        else:
            Thetas = zeros(N_groups)
        self._Thetas, self.Thetas_sum_inv = unifac_Thetas(N_groups, Xs, Qs, Thetas)
        return Thetas

    def _Thetas_sum_inv(self):
        try:
            return self.Thetas_sum_inv
        except AttributeError:
            self.Thetas()
            return self.Thetas_sum_inv

    def dThetas_dxs(self):
        r'''Calculate the mole fraction derivatives of the :math:`\Theta_m`
        parameters. A function of mole fractions and group counts only.

        .. math::
            \frac{\partial \Theta_i}{\partial x_j} =
            FGQ_i\left[FG (\nu x)_{sum,i}
            \left(\sum_k^{gr} FQ_k  (\nu)_{sum,j} (\nu x)_{sum,k}
            -\sum_k^{gr} Q_k \nu_{k,j}
            \right)
            - F (\nu)_{sum,j}(\nu x)_{sum,i} + \nu_{ij}
            \right]

        .. math::
            G = \frac{1}{\sum_j Q_j X_j}

        .. math::
            F = \frac{1}{\sum_j \sum_n \nu_n^j x_j}

        .. math::
            (\nu)_{sum,i} = \sum_j \nu_{j,i}

        .. math::
            (\nu x)_{sum,i} = \sum_j \nu_{i,j}x_j

        Returns
        -------
        dThetas_dxs : list[list[float]]
           Mole fraction derivatives of :math:`\Theta_m` terms, size number of
           subgroups by mole fractions and indexed in that order, [-]
        '''
        try:
            return self._dThetas_dxs
        except AttributeError:
            pass

        try:
            F = self.Xs_sum_inv
        except AttributeError:
            F = self._Xs_sum_inv()
        try:
            G = self.Thetas_sum_inv
        except AttributeError:
            G = self._Thetas_sum_inv()
        Qs, N, N_groups = self.Qs, self.N, self.N_groups
        # Xs_sum_inv and Thetas_sum_inv have already calculated _Xs, _Thetas
        vs = self.vs

        VS = self.cmp_v_count

        if self.scalar:
            dThetas_dxs = [[0.0]*N for _ in range(N_groups)]
        else:
            dThetas_dxs = zeros((N_groups, N))

        try:
            VSXS = self.VSXS
        except AttributeError:
            VSXS = self._VSXS()
#        # Index [subgroup][component]
        self._dThetas_dxs = unifac_dThetas_dxs(N_groups, N, Qs, vs, VS, VSXS, F, G, dThetas_dxs)
        return dThetas_dxs

    def d2Thetas_dxixjs(self):
        r'''Calculate the mole fraction derivatives of the :math:`\Theta_m`
        parameters. A function of mole fractions and group counts only.

        .. math::
            \frac{\partial^2 \Theta_i}{\partial x_j \partial x_k} =
            \frac{Q_i}{\sum_n Q_n (\nu x)_{sum,n}}\left[
            -F(\nu)_{sum,j} \nu_{i,k} - F (\nu)_{sum,k}\nu_{i,j}
            + 2F^2(\nu)_{sum,j} (\nu)_{sum,k} (\nu x)_{sum,i}
            + \frac{F (\nu x)_{sum,i}\left[
            \sum_n(-2 F Q_n (\nu)_{sum,j} (\nu)_{sum,k}
            (\nu x)_{sum,n} + Q_n (\nu)_{sum,j} \nu_{n,k} + Q_n (\nu)_{sum,k}\nu_{n,j}
            )\right] }
            {\sum_n^{gr} Q_n (\nu x)_{sum,n} }
            + \frac{2(\nu x)_{sum,i}(\sum_n^{gr}[-FQ_n (\nu)_{sum,j} (\nu x)_{sum,n} + Q_n \nu_{n,j}])
            (\sum_n^{gr}[-FQ_n (\nu)_{sum,k} (\nu x)_{sum,n} + Q_n \nu_{n,k}])  }
            {\left( \sum_n^{gr} Q_n (\nu x)_{sum,n} \right)^2}
            - \frac{\nu_{i,j}(\sum_n^{gr} -FQ_n (\nu)_{sum,k} (\nu x)_{sum,n} + Q_n \nu_{n,k} )}
            {\left( \sum_n^{gr} Q_n (\nu x)_{sum,n} \right)}
            - \frac{\nu_{i,k}(\sum_n^{gr} -FQ_n (\nu)_{sum,j} (\nu x)_{sum,n} + Q_n \nu_{n,j} )}
            {\left( \sum_n^{gr} Q_n (\nu x)_{sum,n} \right)}
            + \frac{F(\nu)_{sum,j} (\nu x)_{sum,i} (\sum_n^{gr} -FQ_n (\nu)_{sum,k}
            (\nu x)_{sum,n} + Q_n \nu_{n,k})}
            {\left(\sum_n^{gr} Q_n (\nu x)_{sum,n} \right)}
            + \frac{F(\nu)_{sum,k} (\nu x)_{sum,i} (\sum_n^{gr} -FQ_n (\nu)_{sum,j}
            (\nu x)_{sum,n} + Q_n \nu_{n,j})}
            {\left(\sum_n^{gr} Q_n (\nu x)_{sum,n} \right)}
            \right]

        .. math::
            G = \frac{1}{\sum_j Q_j X_j}

        .. math::
            F = \frac{1}{\sum_j \sum_n \nu_n^j x_j}

        .. math::
            (\nu)_{sum,i} = \sum_j \nu_{j,i}

        .. math::
            (\nu x)_{sum,i} = \sum_j \nu_{i,j}x_j

        Returns
        -------
        d2Thetas_dxixjs : list[list[list[float]]]
           :math:`\Theta_m` terms, size number of subgroups by mole fractions
           and indexed in that order, [-]
        '''
        try:
            return self._d2Thetas_dxixjs
        except AttributeError:
            pass

        try:
            F = self.Xs_sum_inv
        except AttributeError:
            F = self._Xs_sum_inv()
        try:
            G = self.Thetas_sum_inv
        except AttributeError:
            G = self._Thetas_sum_inv()
        Qs, N, N_groups = self.Qs, self.N, self.N_groups
        vs = self.vs

        VS = self.cmp_v_count
        try:
            VSXS = self.VSXS
        except AttributeError:
            VSXS = self._VSXS()

        if self.scalar:
            d2Thetas_dxixjs = [[[0.0]*N_groups for _ in range(N)] for _ in range(N)]
        else:
            d2Thetas_dxixjs = zeros((N, N, N_groups))
        self._d2Thetas_dxixjs = unifac_d2Thetas_dxixjs(N_groups, N, Qs, vs, VS, VSXS, F, G, d2Thetas_dxixjs)
        return d2Thetas_dxixjs

    def _VSXS(self):
        try:
            return self.VSXS
        except AttributeError:
            pass
        N_groups = self.N_groups
        if self.scalar:
            VSXS = [0.0]*N_groups
        else:
            VSXS = zeros(N_groups)

        self.VSXS = unifac_VSXS(self.N, N_groups, self.vs, self.xs, VSXS)
        return VSXS

    def _Theta_Psi_sums(self):
        r'''
        Computes the following term for each group `k`, size number of groups.

        .. math::
            \sum_m \Theta_m \Psi_{mk}
        '''
        try:
            return self.Theta_Psi_sums
        except AttributeError:
            pass

        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        N_groups = self.N_groups

        if self.scalar:
            Theta_Psi_sums = [0.0]*N_groups
        else:
            Theta_Psi_sums = zeros(N_groups)

        self.Theta_Psi_sums = unifac_Theta_Psi_sums(N_groups, Thetas, psis, Theta_Psi_sums)
        return Theta_Psi_sums

    def _Theta_Psi_sum_invs(self):
        r'''
        Computes the following term for each group `k`, size number of groups.

        .. math::
            U(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}
        '''
        try:
            return self.Theta_Psi_sum_invs
        except AttributeError:
            try:
                Theta_Psi_sums = self.Theta_Psi_sums
            except AttributeError:
                Theta_Psi_sums = self._Theta_Psi_sums()
        if self.scalar:
            self.Theta_Psi_sum_invs = [1.0/v for v in Theta_Psi_sums]
        else:
            self.Theta_Psi_sum_invs = 1.0/Theta_Psi_sums
        return self.Theta_Psi_sum_invs

    def _Ws(self):
        r'''
        Computes the following for each `k` and each `i`, indexed by [k][i]
        `k` is in groups, and `i` is in components.

        .. math::
            W(k,i) = \sum_m^{gr} \psi_{m,k} \frac{\partial \theta_m}{\partial x_i}
        '''
        try:
            return self.Ws
        except AttributeError:
            pass

        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        try:
            dThetas_dxs = self._dThetas_dxs
        except AttributeError:
            dThetas_dxs = self.dThetas_dxs()
        N, N_groups = self.N, self.N_groups

        if self.scalar:
            Ws = [[0.0]*N for _ in range(N_groups)]
        else:
            Ws = zeros((N_groups, N))

        self.Ws = unifac_ws(N, N_groups, psis, dThetas_dxs, Ws)
        return Ws

    def _Fs(self):
        r'''Computes the following:

        .. math::
            F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}
        '''
        try:
            return self.Fs
        except AttributeError:
            pass
        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            dpsis_dT = self._dpsis_dT
        except AttributeError:
            dpsis_dT = self.dpsis_dT()

        N_groups = self.N_groups

        if self.scalar:
            Fs = [0.0]*N_groups
        else:
            Fs = zeros(N_groups)

        self.Fs = unifac_Theta_Psi_sums(N_groups, Thetas, dpsis_dT, Fs)
        return Fs

    def _Gs(self):
        r'''Computes the following:

        .. math::
            G(k) = \sum_m^{gr} \theta_m \frac{\partial^2 \psi_{m,k}}{\partial T^2}
        '''
        try:
            return self.Gs
        except AttributeError:
            pass
        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            d2psis_dT2 = self._d2psis_dT2
        except AttributeError:
            d2psis_dT2 = self.d2psis_dT2()

        N_groups = self.N_groups

        if self.scalar:
            Gs = [0.0]*N_groups
        else:
            Gs = zeros(N_groups)

        self.Gs = unifac_Theta_Psi_sums(N_groups, Thetas, d2psis_dT2, Gs)
#        self.Gs = Gs = []
#        for k in range(N_groups):
#            tot = 0.0
#            for m in range(N_groups):
#                tot += Thetas[m]*d2psis_dT2[m][k]
#            Gs.append(tot)
        return Gs

    def _Hs(self):
        r'''Computes the following:

        .. math::
            H(k) = \sum_m^{gr} \theta_m \frac{\partial^3 \psi_{m,k}}{\partial T^3}
        '''
        try:
            return self.Hs
        except AttributeError:
            pass
        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            d3psis_dT3 = self._d3psis_dT3
        except AttributeError:
            d3psis_dT3 = self.d3psis_dT3()

        N_groups = self.N_groups

        if self.scalar:
            Hs = [0.0]*N_groups
        else:
            Hs = zeros(N_groups)

        self.Hs = unifac_Theta_Psi_sums(N_groups, Thetas, d3psis_dT3, Hs)
        return Hs

    def _Theta_pure_Psi_sums(self):
        try:
            return self.Theta_pure_Psi_sums
        except AttributeError:
            pass
        Thetas_pure = self._Thetas_pure
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()

        N_groups, N = self.N_groups, self.N

        if self.scalar:
            Theta_pure_Psi_sums = [[0.0]*N_groups for _ in range(N)]
        else:
            Theta_pure_Psi_sums = zeros((N, N_groups))

        self.Theta_pure_Psi_sums = unifac_Theta_pure_Psi_sums(N, N_groups, psis, Thetas_pure, Theta_pure_Psi_sums)
        return Theta_pure_Psi_sums

    def _Theta_pure_Psi_sum_invs(self):
        r'''
        Computes the following term for each group `k`, size number of groups.

        .. math::
            U(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}
        '''
        try:
            return self.Theta_pure_Psi_sum_invs
        except AttributeError:
            try:
                Theta_pure_Psi_sums = self.Theta_pure_Psi_sums
            except AttributeError:
                Theta_pure_Psi_sums = self._Theta_pure_Psi_sums()
        if self.scalar:
            self.Theta_pure_Psi_sum_invs = [[1.0/v for v in row] for row in Theta_pure_Psi_sums]
        else:
            self.Theta_pure_Psi_sum_invs = 1.0/Theta_pure_Psi_sums
        return self.Theta_pure_Psi_sum_invs

    def _Fs_pure(self):
        r'''Computes the following:

        .. math::
            F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}
        '''
        try:
            return self.Fs_pure
        except AttributeError:
            pass
        Thetas_pure = self._Thetas_pure
        try:
            dpsis_dT = self._dpsis_dT
        except AttributeError:
            dpsis_dT = self.dpsis_dT()

        N_groups, N = self.N_groups, self.N

        if self.scalar:
            Fs_pure = [[0.0]*N_groups for _ in range(N)]
        else:
            Fs_pure = zeros((N, N_groups))

        self.Fs_pure = unifac_Theta_pure_Psi_sums(N, N_groups, dpsis_dT, Thetas_pure, Fs_pure)
        return Fs_pure

    def _Gs_pure(self):
        r'''Computes the following:

        .. math::
            G(k) = \sum_m^{gr} \theta_m \frac{\partial^2 \psi_{m,k}}{\partial T^2}
        '''
        try:
            return self.Gs_pure
        except AttributeError:
            pass
        Thetas_pure = self._Thetas_pure
        try:
            d2psis_dT2 = self._d2psis_dT2
        except AttributeError:
            d2psis_dT2 = self.d2psis_dT2()

        N_groups, N = self.N_groups, self.N

        if self.scalar:
            Gs_pure = [[0.0]*N_groups for _ in range(N)]
        else:
            Gs_pure = zeros((N, N_groups))

        self.Gs_pure = unifac_Theta_pure_Psi_sums(N, N_groups, d2psis_dT2, Thetas_pure, Gs_pure)
        return Gs_pure

    def _Hs_pure(self):
        r'''Computes the following:

        .. math::
            H(k) = \sum_m^{gr} \theta_m \frac{\partial^3 \psi_{m,k}}{\partial T^3}
        '''
        try:
            return self.Hs_pure
        except AttributeError:
            pass
        Thetas_pure = self._Thetas_pure
        try:
            d3psis_dT3 = self._d3psis_dT3
        except AttributeError:
            d3psis_dT3 = self.d3psis_dT3()

        N_groups, N = self.N_groups, self.N

        if self.scalar:
            Hs_pure = [[0.0]*N_groups for _ in range(N)]
        else:
            Hs_pure = zeros((N, N_groups))

        self.Hs_pure = unifac_Theta_pure_Psi_sums(N, N_groups, d3psis_dT3, Thetas_pure, Hs_pure)
        return Hs_pure

    def lnGammas_subgroups(self):
        r'''Calculate the :math:`\ln \Gamma_k` parameters for the phase;
        depends on the phases's composition and temperature.

        .. math::
            \ln \Gamma_k = Q_k \left[1 - \ln \sum_m \Theta_m \Psi_{mk} - \sum_m
            \frac{\Theta_m \Psi_{km}}{\sum_n \Theta_n \Psi_{nm}}\right]

        Returns
        -------
        lnGammas_subgroups : list[float]
           Gamma parameters for each subgroup, size number of subgroups, [-]
        '''
        try:
            return self._lnGammas_subgroups
        except AttributeError:
            pass
        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        try:
            Theta_Psi_sums = self.Theta_Psi_sums
        except AttributeError:
            Theta_Psi_sums = self._Theta_Psi_sums()
        try:
            Theta_Psi_sum_invs = self.Theta_Psi_sum_invs
        except AttributeError:
            Theta_Psi_sum_invs = self._Theta_Psi_sum_invs()

        N, N_groups, Qs = self.N, self.N_groups, self.Qs

        if self.scalar:
            lnGammas_subgroups = [0.0]*N_groups
        else:
            lnGammas_subgroups = zeros(N_groups)


        self._lnGammas_subgroups = unifac_lnGammas_subgroups(N, N_groups, Qs, psis, Thetas, Theta_Psi_sums, Theta_Psi_sum_invs, lnGammas_subgroups)
        return lnGammas_subgroups

    def dlnGammas_subgroups_dxs(self):
        r'''Calculate the mole fraction derivatives of the :math:`\ln \Gamma_k`
        parameters for the phase; depends on the phases's composition and
        temperature.

        .. math::
            \frac{\partial \ln \Gamma_k}{\partial x_i} = Q_k\left(
            -\frac{\sum_m^{gr} \psi_{m,k} \frac{\partial \theta_m}{\partial x_i}}{\sum_m^{gr} \theta_m \psi_{m,k}}
            - \sum_m^{gr} \frac{\psi_{k,m} \frac{\partial \theta_m}{\partial x_i}}{\sum_n^{gr} \theta_n \psi_{n,m}}
            + \sum_m^{gr}  \frac{(\sum_n^{gr} \psi_{n,m}\frac{\partial \theta_n}{\partial x_i})\theta_m \psi_{k,m}}{(\sum_n^{gr} \theta_n \psi_{n,m})^2}
            \right)

        The group W is used internally as follows to simplfy the number of
        evaluations.

        .. math::
            W(k,i) = \sum_m^{gr} \psi_{m,k} \frac{\partial \theta_m}{\partial x_i}

        Returns
        -------
        dlnGammas_subgroups_dxs : list[list[float]]
           Mole fraction derivatives of Gamma parameters for each subgroup,
           size number of subgroups by number of components and indexed in
           that order, [-]
        '''
        try:
            return self._dlnGammas_subgroups_dxs
        except AttributeError:
            pass

        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        try:
            Theta_Psi_sum_invs = self.Theta_Psi_sum_invs
        except AttributeError:
            Theta_Psi_sum_invs = self._Theta_Psi_sum_invs()
        try:
            dThetas_dxs = self._dThetas_dxs
        except AttributeError:
            dThetas_dxs = self.dThetas_dxs()
        try:
            Ws = self.Ws
        except AttributeError:
            Ws = self._Ws()
        N, N_groups, Qs = self.N, self.N_groups, self.Qs

        if self.scalar:
            dlnGammas_subgroups_dxs = [[0.0]*N for _ in range(N_groups)]
        else:
            dlnGammas_subgroups_dxs = zeros((N_groups, N))


        self._dlnGammas_subgroups_dxs = unifac_dlnGammas_subgroups_dxs(N, N_groups, Qs, Ws, psis, Thetas, Theta_Psi_sum_invs,
                                   dThetas_dxs, dlnGammas_subgroups_dxs)

        return dlnGammas_subgroups_dxs

    def d2lnGammas_subgroups_dTdxs(self):
        r'''Calculate the temperature and mole fraction derivatives of the
        :math:`\ln \Gamma_k` parameters for the phase; depends on the phases's
        composition and temperature.

        .. math::
            \frac{\partial^2 \ln \Gamma_k}{\partial x_i \partial T} = -Q_k\left(
            D(k,i) Z(k) - B(k)W(k,i) Z(k)^2
            + \sum_m^{gr} (Z(m) \frac{\partial \theta_m}{\partial x_i}\frac{\partial \psi_{k,m}}{\partial T})
            -\sum_m^{gr} (B(m) Z(m)^2 \psi_{k,m} \frac{\partial \theta_m}{\partial x_i})
            -\sum_m^{gr}(D(m,i) Z(m)^2 \theta_m \psi_{k,m})
            - \sum_m^{gr} (W(m,i) Z(m)^2 \theta_m \frac{\partial \psi_{k,m}}{\partial T})
            + \sum_m^{gr} 2 B(m) W(m,i) Z(m)^3 \theta_m \psi_{k,m}
            \right)

        The following groups are used as follows to simplfy the number of
        evaluations:

        .. math::
            W(k,i) = \sum_m^{gr} \psi_{m,k} \frac{\partial \theta_m}{\partial x_i}

        .. math::
            Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{mk}}

        .. math::
            F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}

        In the below expression, k` refers to a group, and `i` refers to a
        component.

        .. math::
            D(k,i) = \sum_m^{gr} \frac{\partial \theta_m}{\partial x_i}
            \frac{\partial \psi_{m,k}}{\partial T}

        Returns
        -------
        d2lnGammas_subgroups_dTdxs : list[list[float]]
           Temperature and mole fraction derivatives of Gamma parameters for
           each subgroup, size number of subgroups by number of components and
           indexed in that order, [1/K]
        '''
        try:
            return self._d2lnGammas_subgroups_dTdxs
        except:
            pass
        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        try:
            dpsis_dT = self._dpsis_dT
        except AttributeError:
            dpsis_dT = self.dpsis_dT()
        try:
            dThetas_dxs = self._dThetas_dxs
        except AttributeError:
            dThetas_dxs = self.dThetas_dxs()
        try:
            Zs = self.Theta_Psi_sum_invs
        except AttributeError:
            Zs = self._Theta_Psi_sum_invs()
        try:
            Ws = self.Ws
        except AttributeError:
            Ws = self._Ws()
        N, N_groups, Qs = self.N, self.N_groups, self.Qs

        try:
            Fs = self.Fs
        except AttributeError:
            Fs = self._Fs()

        if self.scalar:
            d2lnGammas_subgroups_dTdxs = [[0.0]*N for _ in range(N_groups)]
        else:
            d2lnGammas_subgroups_dTdxs = zeros((N_groups, N))

        self._d2lnGammas_subgroups_dTdxs = unifac_d2lnGammas_subgroups_dTdxs(N, N_groups, Qs, Fs, Zs,
                                                                             Ws, psis, dpsis_dT, Thetas, dThetas_dxs,
                                                                             d2lnGammas_subgroups_dTdxs=d2lnGammas_subgroups_dTdxs)


        return d2lnGammas_subgroups_dTdxs


    def d2lnGammas_subgroups_dxixjs(self):
        r'''Calculate the second mole fraction derivatives of the
        :math:`\ln \Gamma_k`  parameters for the phase; depends on the phases's
        composition and temperature.

        .. math::
            \frac{\partial^2 \ln \Gamma_k}{\partial x_i \partial x_j} = -Q_k\left(
            -Z(k) K(k,i,j) - \sum_m^{gr} Z(m)^2 K(m,i,j)\theta_m \psi_{k,m}
            -W(k,i) W(k,j) Z(k)^2
            + \sum_m^{gr} Z_m \psi_{k,m} \frac{\partial^2 \theta_m}{\partial x_i \partial x_j}
            - \sum_m \left(W(m,j) Z(m)^2 \psi_{k,m} \frac{\partial \theta_m}{\partial x_i}
            + W(m,i) Z(m)^2 \psi(k,m) \frac{\partial \theta_m}{\partial x_j}\right)
            + \sum_m^{gr} 2 W(m,i) W(m,j) Z(m)^3 \theta_m \psi_{k,m}\right)


        The following groups are used as follows to simplfy the number of
        evaluations:

        .. math::
            W(k,i) = \sum_m^{gr} \psi_{m,k} \frac{\partial \theta_m}{\partial x_i}

        .. math::
            Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{mk}}

        .. math::
            K(k, i, j) = \sum_m^{gr} \psi_{m,k} \frac{\partial^2 \theta_m}{\partial x_i \partial x_j}

        Returns
        -------
        d2lnGammas_subgroups_dxixjs : list[list[list[float]]]
           Second mole fraction derivatives of Gamma parameters for each
           subgroup, size number of components by number of components by
           number of subgroups and indexed in that order, [-]
        '''
        try:
            return self._d2lnGammas_subgroups_dxixjs
        except:
            pass

        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        try:
            dThetas_dxs = self._dThetas_dxs
        except AttributeError:
            dThetas_dxs = self.dThetas_dxs()
        try:
            d2Thetas_dxixjs = self._d2Thetas_dxixjs
        except AttributeError:
            d2Thetas_dxixjs = self.d2Thetas_dxixjs()
        try:
            Zs = self.Theta_Psi_sum_invs
        except AttributeError:
            Zs = self._Theta_Psi_sum_invs()
        try:
            Ws = self.Ws
        except AttributeError:
            Ws = self._Ws()

        N, N_groups, Qs = self.N, self.N_groups, self.Qs

        if self.scalar:
            d2lnGammas_subgroups_dxixjs = [[[0.0]*N_groups for _ in range(N)] for _ in range(N)]
        else:
            d2lnGammas_subgroups_dxixjs = zeros((N, N, N_groups))

        self._d2lnGammas_subgroups_dxixjs = unifac_d2lnGammas_subgroups_dxixjs(N, N_groups, Qs, Zs, Ws, psis, Thetas, dThetas_dxs, d2Thetas_dxixjs, d2lnGammas_subgroups_dxixjs)
        return d2lnGammas_subgroups_dxixjs

    def dlnGammas_subgroups_dT(self):
        r'''Calculate the first temperature derivative of the
        :math:`\ln \Gamma_k`  parameters for the phase; depends on the phases's
        composition and temperature.

        .. math::
            \frac{\partial \ln \Gamma_i}{\partial T} = Q_i\left(
            \sum_j^{gr} Z(j) \left[{\theta_j \frac{\partial \psi_{i,j}}{\partial T}}
            + {\theta_j \psi_{i,j} F(j)}Z(j) \right]- F(i) Z(i)
            \right)

        .. math::
            F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}

        .. math::
            Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}

        Returns
        -------
        dlnGammas_subgroups_dT : list[float]
           First temperature derivative of ln Gamma parameters for each
           subgroup, size number of subgroups, [1/K]
        '''
        try:
            return self._dlnGammas_subgroups_dT
        except:
            pass
        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        try:
            dpsis_dT = self._dpsis_dT
        except AttributeError:
            dpsis_dT = self.dpsis_dT()
        try:
            Zs = self.Theta_Psi_sum_invs
        except AttributeError:
            Zs = self._Theta_Psi_sum_invs()
        try:
            Fs = self.Fs
        except AttributeError:
            Fs = self._Fs()
        N, N_groups, Qs = self.N, self.N_groups, self.Qs
        if self.scalar:
            dlnGammas_subgroups_dT = [0.0]*N_groups
        else:
            dlnGammas_subgroups_dT = zeros(N_groups)

        self._dlnGammas_subgroups_dT = unifac_dlnGammas_subgroups_dT(N_groups, Qs, psis, dpsis_dT, Thetas, Zs, Fs, dlnGammas_subgroups_dT)
        return dlnGammas_subgroups_dT

    def d2lnGammas_subgroups_dT2(self):
        r'''Calculate the second temperature derivative of the
        :math:`\ln \Gamma_k`  parameters for the phase; depends on the phases's
        composition and temperature.

        .. math::
            \frac{\partial^2 \ln \Gamma_i}{\partial T^2} = -Q_i\left[
            Z(i)G(i) - F(i)^2 Z(i)^2 + \sum_j\left(
            \theta_j Z(j)\frac{\partial^2 \psi_{i,j}}{\partial T}
            - Z(j)^2 \left(G(j)\theta_j \psi_{i,j} + 2 F_j \theta_j \frac{\partial \psi_{i,j}}{\partial T}\right)
            + 2Z(j)^3F(j)^2 \theta_j \psi_{i,j}
            \right)\right]

        .. math::
            F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}

        .. math::
            G(k) = \sum_m^{gr} \theta_m \frac{\partial^2 \psi_{m,k}}{\partial T^2}

        .. math::
            Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}

        Returns
        -------
        d2lnGammas_subgroups_dT2 : list[float]
           Second temperature derivative of ln Gamma parameters for each
           subgroup, size number of subgroups, [1/K^2]
        '''
        try:
            return self._d2lnGammas_subgroups_dT2
        except:
            pass
        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        try:
            dpsis_dT = self._dpsis_dT
        except AttributeError:
            dpsis_dT = self.dpsis_dT()
        try:
            d2psis_dT2 = self._d2psis_dT2
        except AttributeError:
            d2psis_dT2 = self.d2psis_dT2()
        try:
            Zs = self.Theta_Psi_sum_invs
        except AttributeError:
            Zs = self._Theta_Psi_sum_invs()
        try:
            Fs = self.Fs
        except AttributeError:
            Fs = self._Fs()
        try:
            Gs = self.Gs
        except AttributeError:
            Gs = self._Gs()
        N, N_groups, Qs = self.N, self.N_groups, self.Qs

        if self.scalar:
            d2lnGammas_subgroups_dT2 = [0.0]*N_groups
        else:
            d2lnGammas_subgroups_dT2 = zeros(N_groups)

        self._d2lnGammas_subgroups_dT2 = row = unifac_d2lnGammas_subgroups_dT2(
                        N_groups, Qs, psis, dpsis_dT, d2psis_dT2, Thetas, Zs, Fs, Gs, d2lnGammas_subgroups_dT2)
        return row


    def d3lnGammas_subgroups_dT3(self):
        r'''Calculate the third temperature derivative of the
        :math:`\ln \Gamma_k`  parameters for the phase; depends on the phases's
        composition and temperature.

        .. math::
            \frac{\partial^3 \ln \Gamma_i}{\partial T^3} =Q_i\left[-H(i) Z(i)
            - 2F(i)^3 Z(i)^3 + 3F(i) G(i) Z(i)^2+ \left(
            -\theta_j Z(j) \frac{\partial^3 \psi}{\partial T^3}
            + H(j) Z(j)^2 \theta(j)\psi_{i,j}
            - 6F(j)^2 Z(j)^3 \theta_j \frac{\partial \psi_{i,j}}{\partial T}
            + 3 F(j) Z(j)^2 \theta(j) \frac{\partial^2 \psi_{i,j}}{\partial T^2}
            ++ 3G(j) \theta(j) Z(j)^2 \frac{\partial \psi_{i,j}}{\partial T}
            + 6F(j)^3 \theta(j) Z(j)^4 \psi_{i,j}
            - 6F(j) G(j) \theta(j) Z(j)^3 \psi_{i,j}
            \right)
            \right]

        .. math::
            F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}

        .. math::
            G(k) = \sum_m^{gr} \theta_m \frac{\partial^2 \psi_{m,k}}{\partial T^2}

        .. math::
            H(k) = \sum_m^{gr} \theta_m \frac{\partial^3 \psi_{m,k}}{\partial T^3}

        .. math::
            Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}

        Returns
        -------
        d3lnGammas_subgroups_dT3 : list[float]
           Third temperature derivative of ln Gamma parameters for each
           subgroup, size number of subgroups, [1/K^3]
        '''
        try:
            return self._d3lnGammas_subgroups_dT3
        except:
            pass
        try:
            Thetas = self._Thetas
        except AttributeError:
            Thetas = self.Thetas()
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        try:
            dpsis_dT = self._dpsis_dT
        except AttributeError:
            dpsis_dT = self.dpsis_dT()
        try:
            d2psis_dT2 = self._d2psis_dT2
        except AttributeError:
            d2psis_dT2 = self.d2psis_dT2()
        try:
            d3psis_dT3 = self._d3psis_dT3
        except AttributeError:
            d3psis_dT3 = self.d3psis_dT3()
        try:
            Zs = self.Theta_Psi_sum_invs
        except AttributeError:
            Zs = self._Theta_Psi_sum_invs()
        try:
            Fs = self.Fs
        except AttributeError:
            Fs = self._Fs()
        try:
            Gs = self.Gs
        except AttributeError:
            Gs = self._Gs()
        try:
            Hs = self.Hs
        except AttributeError:
            Hs = self._Hs()
        N, N_groups, Qs = self.N, self.N_groups, self.Qs

        if self.scalar:
            d3lnGammas_subgroups_dT3 = [0.0]*N_groups
        else:
            d3lnGammas_subgroups_dT3 = zeros(N_groups)

        self._d3lnGammas_subgroups_dT3 = unifac_d3lnGammas_subgroups_dT3(N_groups, Qs, psis, dpsis_dT, d2psis_dT2, d3psis_dT3,
                                                   Thetas, Zs, Fs, Gs, Hs, d3lnGammas_subgroups_dT3)

        return d3lnGammas_subgroups_dT3

    def Xs_pure(self):
        r'''Calculate the :math:`X_m` parameters for each chemical in the
        mixture as a pure species, used in calculating the residual part. A
        function of group counts only, not even mole fractions or temperature.

        .. math::
            X_m = \frac{\nu_m}{\sum^{gr}_n \nu_n}

        Returns
        -------
        Xs_pure : list[list[float]]
           :math:`X_m` terms, size number of subgroups by number of components
           and indexed in that order, [-]
        '''
        try:
            return self._Xs_pure
        except AttributeError:
            pass
        # Independent of mole fractions and temperature
        vs, cmp_v_count_inv = self.vs, self.cmp_v_count_inv
        N, N_groups = self.N, self.N_groups

        if self.scalar:
            Xs_pure = [[0.0]*N for _ in range(N_groups)]
        else:
            Xs_pure = zeros((N_groups, N))
        self._Xs_pure = unifac_Xs_pure(N, N_groups, vs, cmp_v_count_inv, Xs_pure)
        return Xs_pure

    def Thetas_pure(self):
        r'''Calculate the :math:`\Theta_m` parameters for each chemical in the
        mixture as a pure species, used in calculating the residual part.
        A function of group counts only.

        .. math::
            \Theta_m = \frac{Q_m X_m}{\sum_{n} Q_n X_n}

        Returns
        -------
        Thetas_pure : list[list[float]]
           :math:`\Theta_m` terms, size number of components by number of
           subgroups and indexed in that order, [-]
        '''
        # Composition and temperature independent
        try:
            return self._Thetas_pure
        except AttributeError:
            pass

        Xs_pure, Qs = self.Xs_pure(), self.Qs
        N, N_groups = self.N, self.N_groups

        if self.scalar:
            Thetas_pure = [[0.0]*N_groups for _ in range(N)]
        else:
            Thetas_pure = zeros((N, N_groups))

        # Revised! Keep in order [component][subgroup]
        self._Thetas_pure = unifac_Thetas_pure(N, N_groups, Xs_pure, Qs, Thetas_pure)
        return Thetas_pure


    def lnGammas_subgroups_pure(self):
        r'''Calculate the :math:`\ln \Gamma_k` pure component parameters for
        the phase; depends on the phases's temperature only.

        .. math::
            \ln \Gamma_k = Q_k \left[1 - \ln \sum_m \Theta_m \Psi_{mk} - \sum_m
            \frac{\Theta_m \Psi_{km}}{\sum_n \Theta_n \Psi_{nm}}\right]

        In this model, the :math:`\Theta` values come from the
        :obj:`UNIFAC.Thetas_pure` method, where each compound is assumed to be
        pure.

        Returns
        -------
        lnGammas_subgroups_pure : list[list[float]]
           Gamma parameters for each subgroup, size number of subgroups by
           number of components and indexed in that order, [-]
        '''
        try:
            return self._lnGammas_subgroups_pure
        except AttributeError:
            pass
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        N, N_groups, Qs = self.N, self.N_groups, self.Qs
        Thetas_pure, cmp_group_idx = self._Thetas_pure, self.cmp_group_idx

        if self.scalar:
            lnGammas_subgroups_pure = [[0.0]*N for _ in range(N_groups)]
        else:
            lnGammas_subgroups_pure = zeros((N_groups, N))
        # Future note: lnGammas_subgroups_pure will not zero all arrays
        self._lnGammas_subgroups_pure = unifac_lnGammas_subgroups_pure(N, N_groups, Qs, Thetas_pure, cmp_group_idx, self.group_cmp_idx, psis, lnGammas_subgroups_pure)
        return lnGammas_subgroups_pure

    def dlnGammas_subgroups_pure_dT(self):
        r'''Calculate the first temperature derivative of :math:`\ln \Gamma_k`
        pure component parameters for the phase; depends on the phases's
        temperature only.

        .. math::
            \frac{\partial \ln \Gamma_i}{\partial T} = Q_i\left(
            \sum_j^{gr} Z(j) \left[{\theta_j \frac{\partial \psi_{i,j}}{\partial T}}
            + {\theta_j \psi_{i,j} F(j)}Z(j) \right]- F(i) Z(i)
            \right)

        .. math::
            F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}

        .. math::
            Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}

        In this model, the :math:`\Theta` values come from the
        :obj:`UNIFAC.Thetas_pure` method, where each compound is assumed to be
        pure.

        Returns
        -------
        dlnGammas_subgroups_pure_dT : list[list[float]]
           First temperature derivative of ln Gamma parameters for each
           subgroup, size number of subgroups by number of components and
           indexed in that order, [1/K]
        '''
        # Temperature dependent only!
        try:
            return self._dlnGammas_subgroups_pure_dT
        except:
            pass
        # The followign are calculated on initialization - no caching needed
        Xs_pure = self._Xs_pure
        Thetas_pure = self._Thetas_pure
        try:
            psis = self._psis
        except AttributeError:
            psis = self.psis()
        try:
            dpsis_dT = self._dpsis_dT
        except AttributeError:
            dpsis_dT = self.dpsis_dT()
        N, N_groups, Qs = self.N, self.N_groups, self.Qs
        cmp_group_idx = self.cmp_group_idx

        try:
            Theta_pure_Psi_sum_invs = self.Theta_pure_Psi_sum_invs
        except AttributeError:
            Theta_pure_Psi_sum_invs = self._Theta_pure_Psi_sum_invs()
        try:
            Fs_pure = self.Fs_pure
        except AttributeError:
            Fs_pure = self._Fs_pure()

        if self.scalar:
            dlnGammas_subgroups_pure_dT = [[0.0]*N for _ in range(N_groups)]
        else:
            dlnGammas_subgroups_pure_dT = zeros((N_groups, N))

        self._dlnGammas_subgroups_pure_dT = unifac_dlnGammas_subgroups_pure_dT(N, N_groups, Qs, psis, dpsis_dT,
                                                                               Thetas_pure, Theta_pure_Psi_sum_invs,
                                                                               Fs_pure, cmp_group_idx,
                                                                               dlnGammas_subgroups_pure_dT)
        return dlnGammas_subgroups_pure_dT

    def d2lnGammas_subgroups_pure_dT2(self):
        r'''Calculate the second temperature derivative of :math:`\ln \Gamma_k`
        pure component parameters for the phase; depends on the phases's
        temperature only.

        .. math::
            \frac{\partial^2 \ln \Gamma_i}{\partial T^2} = -Q_i\left[
            Z(i)G(i) - F(i)^2 Z(i)^2 + \sum_j\left(
            \theta_j Z(j)\frac{\partial^2 \psi_{i,j}}{\partial T}
            - Z(j)^2 \left(G(j)\theta_j \psi_{i,j} + 2 F_j \theta_j \frac{\partial \psi_{i,j}}{\partial T}\right)
            + 2Z(j)^3F(j)^2 \theta_j \psi_{i,j}
            \right)\right]

        .. math::
            F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}

        .. math::
            G(k) = \sum_m^{gr} \theta_m \frac{\partial^2 \psi_{m,k}}{\partial T^2}

        .. math::
            Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}

        In this model, the :math:`\Theta` values come from the
        :obj:`UNIFAC.Thetas_pure` method, where each compound is assumed to be
        pure.

        Returns
        -------
        d2lnGammas_subgroups_pure_dT2 : list[list[float]]
           Second temperature derivative of ln Gamma parameters for each
           subgroup, size number of subgroups by number of components and
           indexed in that order, [1/K^2]
        '''
        try:
            return self._d2lnGammas_subgroups_pure_dT2
        except:
            pass

        Thetas_pure, Qs = self.Thetas_pure(), self.Qs
        psis, dpsis_dT, d2psis_dT2 = self.psis(), self.dpsis_dT(), self.d2psis_dT2()
        N, N_groups = self.N, self.N_groups
        cmp_group_idx = self.cmp_group_idx

        # Index by [component][subgroup]
        try:
            Theta_pure_Psi_sum_invs = self.Theta_pure_Psi_sum_invs
        except AttributeError:
            Theta_pure_Psi_sum_invs = self._Theta_pure_Psi_sum_invs()
        try:
            Fs_pure = self.Fs_pure
        except AttributeError:
            Fs_pure = self._Fs_pure()
        try:
            Gs_pure = self.Gs_pure
        except AttributeError:
            Gs_pure = self._Gs_pure()
        if self.scalar:
            d2lnGammas_subgroups_pure_dT2 = [[0.0]*N for _ in range(N_groups)]
        else:
            d2lnGammas_subgroups_pure_dT2 = zeros((N_groups, N))
        # Index by [subgroup][component]
        self._d2lnGammas_subgroups_pure_dT2 = unifac_d2lnGammas_subgroups_pure_dT2(N, N_groups, Qs, psis, dpsis_dT, d2psis_dT2, Thetas_pure, Theta_pure_Psi_sum_invs, Fs_pure, Gs_pure, cmp_group_idx, d2lnGammas_subgroups_pure_dT2)
        return d2lnGammas_subgroups_pure_dT2

    def d3lnGammas_subgroups_pure_dT3(self):
        r'''Calculate the third temperature derivative of :math:`\ln \Gamma_k`
        pure component parameters for the phase; depends on the phases's
        temperature only.

        .. math::
            \frac{\partial^3 \ln \Gamma_i}{\partial T^3} =Q_i\left[-H(i) Z(i)
            - 2F(i)^3 Z(i)^3 + 3F(i) G(i) Z(i)^2+ \left(
            -\theta_j Z(j) \frac{\partial^3 \psi}{\partial T^3}
            + H(j) Z(j)^2 \theta(j)\psi_{i,j}
            - 6F(j)^2 Z(j)^3 \theta_j \frac{\partial \psi_{i,j}}{\partial T}
            + 3 F(j) Z(j)^2 \theta(j) \frac{\partial^2 \psi_{i,j}}{\partial T^2}
            ++ 3G(j) \theta(j) Z(j)^2 \frac{\partial \psi_{i,j}}{\partial T}
            + 6F(j)^3 \theta(j) Z(j)^4 \psi_{i,j}
            - 6F(j) G(j) \theta(j) Z(j)^3 \psi_{i,j}
            \right)
            \right]

        .. math::
            F(k) = \sum_m^{gr} \theta_m \frac{\partial \psi_{m,k}}{\partial T}

        .. math::
            G(k) = \sum_m^{gr} \theta_m \frac{\partial^2 \psi_{m,k}}{\partial T^2}

        .. math::
            H(k) = \sum_m^{gr} \theta_m \frac{\partial^3 \psi_{m,k}}{\partial T^3}

        .. math::
            Z(k) = \frac{1}{\sum_m \Theta_m \Psi_{m,k}}

        In this model, the :math:`\Theta` values come from the
        :obj:`UNIFAC.Thetas_pure` method, where each compound is assumed to be
        pure.

        Returns
        -------
        d3lnGammas_subgroups_pure_dT3 : list[list[float]]
           Third temperature derivative of ln Gamma parameters for each
           subgroup, size number of subgroups by number of components and
           indexed in that order, [1/K^3]
        '''
        try:
            return self._d3lnGammas_subgroups_pure_dT3
        except:
            pass
        Thetas_pure, Qs = self.Thetas_pure(), self.Qs
        psis, dpsis_dT, d2psis_dT2, d3psis_dT3 = self.psis(), self.dpsis_dT(), self.d2psis_dT2(), self.d3psis_dT3()
        N, N_groups = self.N, self.N_groups
        cmp_group_idx = self.cmp_group_idx
        try:
            Theta_pure_Psi_sum_invs = self.Theta_pure_Psi_sum_invs
        except AttributeError:
            Theta_pure_Psi_sum_invs = self._Theta_pure_Psi_sum_invs()
        try:
            Fs_pure = self.Fs_pure
        except AttributeError:
            Fs_pure = self._Fs_pure()
        try:
            Gs_pure = self.Gs_pure
        except AttributeError:
            Gs_pure = self._Gs_pure()
        try:
            Hs_pure = self.Hs_pure
        except AttributeError:
            Hs_pure = self._Hs_pure()


        if self.scalar:
            d3lnGammas_subgroups_pure_dT3 = [[0.0]*N for _ in range(N_groups)]
        else:
            d3lnGammas_subgroups_pure_dT3 = zeros((N_groups, N))
        # Index by [subgroup][component]
        self._d3lnGammas_subgroups_pure_dT3 = unifac_d3lnGammas_subgroups_pure_dT3(N, N_groups, Qs, psis, dpsis_dT, d2psis_dT2, d3psis_dT3, Thetas_pure, Theta_pure_Psi_sum_invs, Fs_pure, Gs_pure, Hs_pure, cmp_group_idx, d3lnGammas_subgroups_pure_dT3)
        return d3lnGammas_subgroups_pure_dT3

    def lngammas_r(self):
        r'''Calculates the residual part of the UNIFAC model.

        .. math::
            \ln \gamma_i^r = \sum_{k}^{gr} \nu_k^{(i)} \left[ \ln \Gamma_k
            - \ln \Gamma_k^{(i)} \right]

        where the second Gamma is the pure-component Gamma of group `k` in
        component `i`.

        Returns
        -------
        lngammas_r : list[float]
            Residual lngammas terms, size number of components [-]
        '''
        try:
            return self._lngammas_r
        except AttributeError:
            pass
        lnGammas_subgroups_pure = self.lnGammas_subgroups_pure()
        lnGammas_subgroups = self.lnGammas_subgroups()
        vs = self.vs
        N, N_groups = self.N, self.N_groups

        if self.scalar:
            lngammas_r = [0.0]*N
        else:
            lngammas_r = zeros(N)

        self._lngammas_r = unifac_lngammas_r(N, N_groups, lnGammas_subgroups_pure, lnGammas_subgroups, vs, lngammas_r)

        return lngammas_r

    def dlngammas_r_dT(self):
        r'''Calculates the first temperature derivative of the residual part of
        the UNIFAC model.

        .. math::
            \frac{\partial \ln \gamma_i^r}{\partial T} = \sum_{k}^{gr}
            \nu_k^{(i)} \left[ \frac{\partial \ln \Gamma_k}{\partial T}
            - \frac{\partial \ln \Gamma_k^{(i)}}{\partial T} \right]

        where the second Gamma is the pure-component Gamma of group `k` in
        component `i`.

        Returns
        -------
        dlngammas_r_dT : list[float]
            Residual lngammas terms first temperature derivative, size number
            of components [1/K]
        '''
        try:
            return self._dlngammas_r_dT
        except AttributeError:
            pass
        dlnGammas_subgroups_pure_dT = self.dlnGammas_subgroups_pure_dT()
        dlnGammas_subgroups_dT = self.dlnGammas_subgroups_dT()
        vs = self.vs
        N, N_groups = self.N, self.N_groups

        if self.scalar:
            dlngammas_r_dT = [0.0]*N
        else:
            dlngammas_r_dT = zeros(N)

        self._dlngammas_r_dT = unifac_lngammas_r(N, N_groups, dlnGammas_subgroups_pure_dT, dlnGammas_subgroups_dT, vs, dlngammas_r_dT)
        return dlngammas_r_dT
    dlngammas_dT = dlngammas_r_dT

    def d2lngammas_r_dT2(self):
        r'''Calculates the second temperature derivative of the residual part of
        the UNIFAC model.

        .. math::
            \frac{\partial^2 \ln \gamma_i^r}{\partial T^2} = \sum_{k}^{gr}
            \nu_k^{(i)} \left[ \frac{\partial^2 \ln \Gamma_k}{\partial T^2}
            - \frac{\partial^2 \ln \Gamma_k^{(i)}}{\partial T^2} \right]

        where the second Gamma is the pure-component Gamma of group `k` in
        component `i`.

        Returns
        -------
        d2lngammas_r_dT2 : list[float]
            Residual lngammas terms second temperature derivative, size number
            of components [1/K^2]
        '''
        try:
            return self._d2lngammas_r_dT2
        except AttributeError:
            pass
        d2lnGammas_subgroups_pure_dT2 = self.d2lnGammas_subgroups_pure_dT2()
        d2lnGammas_subgroups_dT2 = self.d2lnGammas_subgroups_dT2()
        vs = self.vs
        N, N_groups = self.N, self.N_groups


        if self.scalar:
            d2lngammas_r_dT2 = [0.0]*N
        else:
            d2lngammas_r_dT2 = zeros(N)

        self._d2lngammas_r_dT2 = unifac_lngammas_r(N, N_groups, d2lnGammas_subgroups_pure_dT2, d2lnGammas_subgroups_dT2, vs, d2lngammas_r_dT2)
        return d2lngammas_r_dT2
    d2lngammas_dT2 = d2lngammas_r_dT2

    def d3lngammas_r_dT3(self):
        r'''Calculates the third temperature derivative of the residual part of
        the UNIFAC model.

        .. math::
            \frac{\partial^3 \ln \gamma_i^r}{\partial T^3} = \sum_{k}^{gr}
            \nu_k^{(i)} \left[ \frac{\partial^23\ln \Gamma_k}{\partial T^3}
            - \frac{\partial^3 \ln \Gamma_k^{(i)}}{\partial T^3} \right]

        where the second Gamma is the pure-component Gamma of group `k` in
        component `i`.

        Returns
        -------
        d3lngammas_r_dT3 : list[float]
            Residual lngammas terms third temperature derivative, size number
            of components [1/K^3]
        '''
        try:
            return self._d3lngammas_r_dT3
        except AttributeError:
            pass
        d3lnGammas_subgroups_pure_dT3 = self.d3lnGammas_subgroups_pure_dT3()
        d3lnGammas_subgroups_dT3 = self.d3lnGammas_subgroups_dT3()
        vs = self.vs
        N, N_groups = self.N, self.N_groups
        if self.scalar:
            d3lngammas_r_dT3 = [0.0]*N
        else:
            d3lngammas_r_dT3 = zeros(N)
        self._d3lngammas_r_dT3 = unifac_lngammas_r(N, N_groups, d3lnGammas_subgroups_pure_dT3, d3lnGammas_subgroups_dT3, vs, d3lngammas_r_dT3)
        return d3lngammas_r_dT3
    d3lngammas_dT3 = d3lngammas_r_dT3

    def dlngammas_r_dxs(self):
        r'''Calculates the first mole fraction derivative of the residual part
        of the UNIFAC model.

        .. math::
            \frac{\partial \ln \gamma_i^r}{\partial x_j} = \sum_{m}^{gr} \nu_m^{(i)}
            \frac{\partial \ln \Gamma_m}{\partial x_j}

        Returns
        -------
        dlngammas_r_dxs : list[list[float]]
            First mole fraction derivative of residual lngammas terms, size
            number of components by number of components [-]
        '''
        try:
            return self._dlngammas_r_dxs
        except AttributeError:
            pass
        vs, N, N_groups = self.vs, self.N, self.N_groups
        dlnGammas_subgroups_dxs = self.dlnGammas_subgroups_dxs()

        if self.scalar:
            dlngammas_r_dxs = [[0.0]*N for _ in range(N)]
        else:
            dlngammas_r_dxs = zeros((N, N))

        self._dlngammas_r_dxs = unifac_dlngammas_r_dxs(N, N_groups, vs, dlnGammas_subgroups_dxs, dlngammas_r_dxs)
        for i in range(N):
            row = dlngammas_r_dxs[i]
            for j in range(N):
                tot = 0.0
                for m in range(N_groups):
                    tot += vs[m][i]*dlnGammas_subgroups_dxs[m][j]
                row[j] = tot

        return dlngammas_r_dxs

    def d2lngammas_r_dTdxs(self):
        r'''Calculates the first mole fraction derivative of the temperature
        derivative of the residual part of the UNIFAC model.

        .. math::
            \frac{\partial^2 \ln \gamma_i^r}{\partial x_j \partial T} =
            \sum_{m}^{gr} \nu_m^{(i)} \frac{\partial^2 \ln \Gamma_m}
            {\partial x_j \partial T}

        Returns
        -------
        d2lngammas_r_dTdxs : list[list[float]]
            First mole fraction derivative and temperature derivative of
            residual lngammas terms, size number of components by number of
            components [-]
        '''
        try:
            return self._d2lngammas_r_dTdxs
        except AttributeError:
            pass
        vs = self.vs
        N, N_groups = self.N, self.N_groups
        d2lnGammas_subgroups_dTdxs = self.d2lnGammas_subgroups_dTdxs()

        if self.scalar:
            d2lngammas_r_dTdxs = [[0.0]*N for _ in range(N)]
        else:
            d2lngammas_r_dTdxs = zeros((N, N))

        self._d2lngammas_r_dTdxs = unifac_dlngammas_r_dxs(N, N_groups, vs, d2lnGammas_subgroups_dTdxs, d2lngammas_r_dTdxs)
        return d2lngammas_r_dTdxs

    def d2lngammas_r_dxixjs(self):
        r'''Calculates the second mole fraction derivative of the residual part
        of the UNIFAC model.

        .. math::
            \frac{\partial^2 \ln \gamma_i^r}{\partial x_j^2} = \sum_{m}^{gr}
            \nu_m^{(i)} \frac{\partial^2 \ln \Gamma_m}{\partial x_j^2}

        Returns
        -------
        d2lngammas_r_dxixjs : list[list[list[float]]]
            Second mole fraction derivative of the residual lngammas terms,
            size number of components by number of components by number of
            components [-]
        '''
        try:
            return self._d2lngammas_r_dxixjs
        except AttributeError:
            pass
        vs = self.vs
        N, N_groups = self.N, self.N_groups
        d2lnGammas_subgroups_dxixjs = self.d2lnGammas_subgroups_dxixjs()

        if self.scalar:
            d2lngammas_r_dxixjs = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d2lngammas_r_dxixjs = zeros((N, N, N))

        self._d2lngammas_r_dxixjs = unifac_d2lngammas_r_dxixjs(N, N_groups, vs, d2lnGammas_subgroups_dxixjs, d2lngammas_r_dxixjs)
        return d2lngammas_r_dxixjs

    def GE(self):
        r'''Calculate the excess Gibbs energy with the UNIFAC model.

        .. math::
            G^E = RT\sum_i x_i \left(\ln \gamma_i^c + \ln \gamma_i^r \right)

        For the VTPR model, the combinatorial component is set to zero.

        Returns
        -------
        GE : float
            Excess Gibbs energy, [J/mol]
        '''
        try:
            return self._GE
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N
        lngammas_r = self.lngammas_r()
        if self.skip_comb:
            GE = unifac_GE_skip_comb(T, xs, N, lngammas_r)
        else:
            lngammas_c = self.lngammas_c()
            GE = unifac_GE(T, xs, N, lngammas_r, lngammas_c)
        self._GE = GE
        return GE

    def dGE_dxs(self):
        r'''Calculate the first composition derivative of excess Gibbs energy
        with the UNIFAC model.

        .. math::
            \frac{\partial G^E}{\partial x_i} = RT\left(\ln \gamma_i^c
            + \ln \gamma_i^r \right)
            + RT\sum_j x_j \left(\frac{\partial \ln \gamma_j^c}{\partial x_i}
            + \frac{\partial \ln \gamma_j^r}{\partial x_i} \right)

        Returns
        -------
        dGE_dxs : list[float]
            First composition derivative of excess Gibbs energy, [J/mol]
        '''
        try:
            return self._dGE_dxs
        except AttributeError:
            pass
        T, xs, N, skip_comb = self.T, self.xs, self.N, self.skip_comb
        lngammas_r = self.lngammas_r()
        dlngammas_r_dxs = self.dlngammas_r_dxs()
        if self.scalar:
            dGE_dxs = [0.0]*N
        else:
            dGE_dxs = zeros(N)

        if skip_comb:
            self._dGE_dxs = unifac_dGE_dxs_skip_comb(T, xs, N, lngammas_r, dlngammas_r_dxs, dGE_dxs)
        else:
            lngammas_c = self.lngammas_c()
            dlngammas_c_dxs = self.dlngammas_c_dxs()
            self._dGE_dxs = unifac_dGE_dxs(T, xs, N, lngammas_r, dlngammas_r_dxs, lngammas_c, dlngammas_c_dxs, dGE_dxs)

        return dGE_dxs

    def d2GE_dTdxs(self):
        r'''Calculate the first composition derivative and temperature
        derivative of excess Gibbs energy with the UNIFAC model.

        .. math::
            \frac{\partial^2 G^E}{\partial T\partial x_i} =
            RT\left(\frac{\partial \ln \gamma_i^r}{\partial T}
            + \sum_j x_j \frac{\partial \ln \gamma_j^r}{\partial x_i}  \right)
            + R\left[ \frac{\partial \ln \gamma_i^c}{\partial x_i}
            + \frac{\partial \ln \gamma_i^r}{\partial x_i}
            + \sum_j x_j \left( \frac{\partial \ln \gamma_j^c}{\partial x_i}
            + \frac{\partial \ln \gamma_j^r}{\partial x_i}\right)\right]

        Returns
        -------
        dGE_dxs : list[float]
            First composition derivative and first temperature derivative of
            excess Gibbs energy, [J/mol/K]
        '''
        try:
            return self._d2GE_dTdxs
        except AttributeError:
            pass
        T, xs, N, skip_comb = self.T, self.xs, self.N, self.skip_comb
        lngammas_r = self.lngammas_r()
        dlngammas_r_dxs = self.dlngammas_r_dxs()
        dlngammas_r_dT = self.dlngammas_r_dT()
        d2lngammas_r_dTdxs = self.d2lngammas_r_dTdxs()

        if self.scalar:
            d2GE_dTdxs = [0.0]*N
        else:
            d2GE_dTdxs = zeros(N)

        if skip_comb:
            self._d2GE_dTdxs = unifac_d2GE_dTdxs_skip_comb(T, xs, N, lngammas_r, dlngammas_r_dxs,dlngammas_r_dT, d2lngammas_r_dTdxs, d2GE_dTdxs)
        else:
            lngammas_c = self.lngammas_c()
            dlngammas_c_dxs = self.dlngammas_c_dxs()
            self._d2GE_dTdxs = unifac_d2GE_dTdxs(T, xs, N, lngammas_r, dlngammas_r_dxs, dlngammas_r_dT, d2lngammas_r_dTdxs, lngammas_c, dlngammas_c_dxs, d2GE_dTdxs)
        return d2GE_dTdxs

    def d2GE_dxixjs(self):
        r'''Calculate the second composition derivative of excess Gibbs energy
        with the UNIFAC model.

        .. math::
            \frac{\partial^2 G^E}{\partial x_j \partial x_k} = RT
            \left[\sum_i \left(
            \frac{\partial \ln \gamma_i^c}{\partial x_j \partial x_k}
            + \frac{\partial \ln \gamma_i^r}{\partial x_j \partial x_k}
            \right)
            + \frac{\partial \ln \gamma_j^c}{\partial x_k}
            + \frac{\partial \ln \gamma_j^r}{\partial x_k}
            + \frac{\partial \ln \gamma_k^c}{\partial x_j}
            + \frac{\partial \ln \gamma_k^r}{\partial x_j}\right]

        Returns
        -------
        d2GE_dxixjs : list[list[float]]
            Second composition derivative of excess Gibbs energy, [J/mol]
        '''
        try:
            return self._d2GE_dxixjs
        except AttributeError:
            pass
        T, xs, N, skip_comb = self.T, self.xs, self.N, self.skip_comb

        dlngammas_r_dxs = self.dlngammas_r_dxs()
        d2lngammas_r_dxixjs = self.d2lngammas_r_dxixjs()
        RT = R*T

        if self.scalar:
            d2GE_dxixjs = [[0.0]*N for _ in range(N)]
        else:
            d2GE_dxixjs = zeros((N, N))



        if skip_comb:
            self._d2GE_dxixjs = unifac_d2GE_dxixjs_skip_comb(T, xs, N, dlngammas_r_dxs, d2lngammas_r_dxixjs, d2GE_dxixjs)
        else:
            dlngammas_c_dxs = self.dlngammas_c_dxs()
            d2lngammas_c_dxixjs = self.d2lngammas_c_dxixjs()
            self._d2GE_dxixjs = unifac_d2GE_dxixjs(T, xs, N, dlngammas_r_dxs, d2lngammas_r_dxixjs, dlngammas_c_dxs, d2lngammas_c_dxixjs, d2GE_dxixjs)
        return d2GE_dxixjs

    def dGE_dT(self):
        r'''Calculate the first temperature derivative of excess Gibbs energy
        with the UNIFAC model.

        .. math::
            \frac{\partial G^E}{\partial T} =
            RT\sum_i x_i \frac{\partial \ln \gamma_i^r}{\partial T}
            + \frac{G^E}{T}

        Returns
        -------
        dGE_dT : float
            First temperature derivative of excess Gibbs energy, [J/mol/K]
        '''
        try:
            return self._dGE_dT
        except AttributeError:
            pass
        self._dGE_dT = dGE_dT = unifac_dGE_dT(self.N, self.T, self.xs, self.dlngammas_r_dT(), self.GE())
        return dGE_dT

    def d2GE_dT2(self):
        r'''Calculate the second temperature derivative of excess Gibbs energy
        with the UNIFAC model.

        .. math::
            \frac{\partial^2 G^E}{\partial T^2} =
            RT\sum_i x_i \frac{\partial^2 \ln \gamma_i^r}{\partial T^2}
            + 2R\sum_i x_i \frac{\partial \ln \gamma_i^r}{\partial T}

        Returns
        -------
        d2GE_dT2 : float
            Second temperature derivative of excess Gibbs energy, [J/mol/K^2]
        '''
        try:
            return self._d2GE_dT2
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N
        dlngammas_r_dT = self.dlngammas_r_dT()
        d2lngammas_r_dT2 = self.d2lngammas_r_dT2()
        self._d2GE_dT2 = d2GE_dT2 = unifac_d2GE_dT2(N, T, xs, dlngammas_r_dT, d2lngammas_r_dT2)
        return d2GE_dT2

    def d3GE_dT3(self):
        r'''Calculate the third temperature derivative of excess Gibbs energy
        with the UNIFAC model.

        .. math::
            \frac{\partial^3 G^E}{\partial T^3} =
            RT\sum_i x_i \frac{\partial^3 \ln \gamma_i^r}{\partial T^3}
            + 3R\sum_i x_i \frac{\partial^2 \ln \gamma_i^r}{\partial T^2}

        Returns
        -------
        d3GE_dT3 : float
            Third temperature derivative of excess Gibbs energy, [J/mol/K^3]
        '''
        try:
            return self._d3GE_dT3
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N
        d2lngammas_r_dT2 = self.d2lngammas_r_dT2()
        d3lngammas_r_dT3 = self.d3lngammas_r_dT3()
        self._d3GE_dT3 = d3GE_dT3 = unifac_d3GE_dT3(N, T, xs, d2lngammas_r_dT2, d3lngammas_r_dT3)
        return d3GE_dT3

    def gammas(self):
        r'''Calculates the activity coefficients with the UNIFAC model.

        .. math::
            \gamma_i =  \exp\left(\ln \gamma_i^c + \ln \gamma_i^r \right)

        For the VTPR variant, the combinatorial part is skipped:

        .. math::
            \gamma_i = \exp(\ln \gamma_i^r)

        Returns
        -------
        gammas : list[float]
            Activity coefficients, size number of components [-]
        '''
        try:
            return self._gammas
        except:
            pass
        xs, N = self.xs, self.N
        try:
            lngammas_r = self._lngammas_r
        except AttributeError:
            lngammas_r = self.lngammas_r()
        if self.skip_comb:
            if self.scalar:
                self._gammas = gammas = [exp(ri) for ri in lngammas_r]
            else:
                self._gammas = gammas = npexp(lngammas_r)
        else:
            try:
                lngammas_c = self._lngammas_c
            except AttributeError:
                lngammas_c = self.lngammas_c()
            if self.scalar:
                gammas = [0.0]*N
            else:
                gammas = zeros(N)
            self._gammas = unifac_gammas(N, xs, lngammas_r, lngammas_c, gammas)

        self._gammas = gammas
        return gammas

    def dgammas_dT(self):
        r'''Calculates the first temperature derivative of activity
        coefficients with the UNIFAC model.

        .. math::
            \frac{\partial \gamma_i}{\partial T} = \gamma_i\frac{\partial \ln
            \gamma_i^r}{\partial T}

        Returns
        -------
        dgammas_dT : list[float]
            First temperature derivative of activity coefficients, size number
            of components [1/K]
        '''
        try:
            return self._dgammas_dT
        except:
            pass
        try:
            gammas = self._gammas
        except AttributeError:
            gammas = self.gammas()
        try:
            dlngammas_r_dT = self._dlngammas_r_dT
        except AttributeError:
            dlngammas_r_dT = self.dlngammas_r_dT()
        if self.scalar:
            self._dgammas_dT = dgammas_dT = [dlngammas_r_dT[i]*gammas[i] for i in range(self.N)]
        else:
            self._dgammas_dT = dgammas_dT = dlngammas_r_dT*gammas
        return dgammas_dT

    def dgammas_dns(self):
        try:
            return self._dgammas_dns
        except AttributeError:
            pass
        dgammas_dxs = self.dgammas_dxs()
        N = self.N
        if self.scalar:
            dgammas_dns = [[0.0]*N for _ in range(N)]
        else:
            dgammas_dns = zeros((N, N))
        self._dgammas_dns = unifac_dgammas_dns(N, self.xs, dgammas_dxs, dgammas_dns)
        return dgammas_dns

    def dgammas_dxs(self):
        r'''Calculates the first mole fraction derivative of activity
        coefficients with the UNIFAC model.

        .. math::
            \frac{\partial \gamma_i}{\partial x_j} = \gamma_i
            \left(\frac{\partial \ln \gamma_i^r}{\partial x_j}
            + \frac{\partial \ln \gamma_i^c}{\partial x_j}
            \right)

        For the VTPR variant, the combinatorial part is skipped:

        .. math::
            \frac{\partial \gamma_i}{\partial x_j} = \gamma_i
            \left(\frac{\partial \ln \gamma_i^r}{\partial x_j}
            \right)

        Returns
        -------
        dgammas_dxs : list[list[float]]
            First mole fraction derivative of activity coefficients, size
            number of components by number of components [-]
        '''
        try:
            return self._dgammas_dxs
        except:
            pass
        try:
            gammas = self._gammas
        except AttributeError:
            gammas = self.gammas()


        xs, N = self.xs, self.N
        try:
            dlngammas_r_dxs = self._dlngammas_r_dxs
        except AttributeError:
            dlngammas_r_dxs = self.dlngammas_r_dxs()

        if self.scalar:
            dgammas_dxs = [[0.0]*N for _ in range(N)]
        else:
            dgammas_dxs = zeros((N, N))

        if self.skip_comb:
            self._dgammas_dxs = unifac_dgammas_dxs_skip_comb(N, xs, gammas, dlngammas_r_dxs, dgammas_dxs)
        else:
            try:
                dlngammas_c_dxs = self._dlngammas_c_dxs
            except AttributeError:
                dlngammas_c_dxs = self.dlngammas_c_dxs()
            self._dgammas_dxs = unifac_dgammas_dxs(N, xs, gammas, dlngammas_r_dxs, dlngammas_c_dxs, dgammas_dxs)
        return dgammas_dxs

    def lngammas_c(self):
        r'''Calculates the combinatorial part of the UNIFAC model. For the
        modified UNIFAC model, the equation is as follows; for the original
        UNIFAC and UNIFAC LLE, replace :math:`V_i'` with :math:`V_i`.

        .. math::
            \ln \gamma_i^c = 1 - {V'}_i + \ln({V'}_i) - 5q_i \left(1
            - \frac{V_i}{F_i}+ \ln\left(\frac{V_i}{F_i}\right)\right)

        For the Lyngby model:

        .. math::
            \ln \gamma_i^c = \ln \left(V_i'\right) + 1
            - V_i'

        Returns
        -------
        lngammas_c : list[float]
            Combinatorial lngammas terms, size number of components [-]
        '''
        try:
            return self._lngammas_c
        except AttributeError:
            pass
        try:
            Vis = self._Vis
        except AttributeError:
            Vis = self.Vis()
        try:
            Fis = self._Fis
        except AttributeError:
            Fis = self.Fis()
        N, version, qs = self.N, self.version, self.qs
        if self.version in (1, 4):
            try:
                Vis_modified = self._Vis_modified
            except AttributeError:
                Vis_modified = self.Vis_modified()
        else:
            Vis_modified = Vis

        if self.scalar:
            lngammas_c = [0.0]*N
        else:
            lngammas_c = zeros(N)

        self._lngammas_c = unifac_lngammas_c(N, version, qs, Fis, Vis, Vis_modified, lngammas_c)
        return lngammas_c

    def dlngammas_c_dT(self):
        r'''Temperature derivatives of the combinatorial part of the UNIFAC
        model. Zero in all variations.

        .. math::
            \frac{\partial \ln \gamma_i^c}{\partial T} = 0

        Returns
        -------
        dlngammas_c_dT : list[float]
            Combinatorial lngammas term temperature derivatives, size number of
            components, [-]
        '''
        if self.scalar:
            return [0.0]*self.N
        return zeros(self.N)

    def d2lngammas_c_dT2(self):
        r'''Second temperature derivatives of the combinatorial part of the
        UNIFAC model. Zero in all variations.

        .. math::
            \frac{\partial^2 \ln \gamma_i^c}{\partial T^2} = 0

        Returns
        -------
        d2lngammas_c_dT2 : list[float]
            Combinatorial lngammas term second temperature derivatives, size
            number of components, [-]
        '''
        if self.scalar:
            return [0.0]*self.N
        return zeros(self.N)

    def d3lngammas_c_dT3(self):
        r'''Third temperature derivatives of the combinatorial part of the
        UNIFAC model. Zero in all variations.

        .. math::
            \frac{\partial^3 \ln \gamma_i^c}{\partial T^3} = 0

        Returns
        -------
        d3lngammas_c_dT3 : list[float]
            Combinatorial lngammas term second temperature derivatives, size
            number of components, [-]
        '''
        if self.scalar:
            return [0.0]*self.N
        return zeros(self.N)

    def d2lngammas_c_dTdx(self):
        r'''Second temperature derivative and first mole fraction derivative of
        the combinatorial part of the UNIFAC model. Zero in all variations.

        .. math::
            \frac{\partial^3 \ln \gamma_i^c}{\partial T^2 \partial x_j} = 0

        Returns
        -------
        d2lngammas_c_dTdx : list[list[float]]
            Combinatorial lngammas term second temperature derivatives, size
            number of components by number of components, [-]
        '''
        if self.scalar:
            return [0.0]*self.N
        return zeros(self.N)

    def dlngammas_c_dxs(self):
        r'''First composition derivative of
        the combinatorial part of the UNIFAC model. For the modified UNIFAC
        model, the equation is as follows; for the original UNIFAC and UNIFAC
        LLE, replace :math:`V_i'` with :math:`V_i`.

        .. math::
            \frac{\partial \ln \gamma^c_i}{\partial x_j} =
            -5q_i\left[ \left( \frac{\frac{\partial V_i}{\partial x_j}}{F_i}
            - \frac{V_i \frac{\partial F_i}{\partial x_j}}{F_i^2}
            \right)\frac{F_i}{V_i} - \frac{\frac{\partial V_i}{\partial x_j}}{F_i}
            + \frac{V_i\frac{\partial F_i}{\partial x_j}}{F_i^2}
            \right]
            - \frac{\partial V_i'}{\partial x_j}
            + \frac{\frac{\partial V_i'}{\partial x_j}}{V_i'}

        For the Lyngby model, the following equations are used:

        .. math::
            \frac{\partial \ln \gamma^c_i}{\partial x_j} =
            \frac{-\partial V_i'}{\partial x_j} + \frac{1}{V_i'}
            \frac{\partial V_i'}{\partial x_j}

        Returns
        -------
        dlngammas_c_dxs : list[list[float]]
            Combinatorial lngammas term first composition derivative, size
            number of components by number of components, [-]
        '''
        try:
            return self._dlngammas_c_dxs
        except AttributeError:
            pass
        N, version, qs = self.N, self.version, self.qs
        try:
            Vis = self._Vis
        except AttributeError:
            Vis = self.Vis()
        try:
            dVis_dxs = self._dVis_dxs
        except AttributeError:
            dVis_dxs = self.dVis_dxs()
        try:
            Fis = self._Fis
        except AttributeError:
            Fis = self.Fis()
        try:
            dFis_dxs = self._dFis_dxs
        except AttributeError:
            dFis_dxs = self.dFis_dxs()

        if version in (1, 4):
            try:
                Vis_modified = self._Vis_modified
            except AttributeError:
                Vis_modified = self.Vis_modified()
            try:
                dVis_modified_dxs = self._dVis_modified_dxs
            except AttributeError:
                dVis_modified_dxs = self.dVis_modified_dxs()
        else:
            Vis_modified = Vis
            dVis_modified_dxs = dVis_dxs

        if self.scalar:
            dlngammas_c_dxs = [[0.0]*N for _ in range(N)]
        else:
            dlngammas_c_dxs = zeros((N, N))

        # index style - [THE GAMMA FOR WHICH THE DERIVATIVE IS BEING CALCULATED][THE VARIABLE BEING CHANGED CAUsING THE DIFFERENCE]
        self._dlngammas_c_dxs = unifac_dlngammas_c_dxs(N, version, qs, Fis, dFis_dxs, Vis, dVis_dxs, Vis_modified, dVis_modified_dxs, dlngammas_c_dxs)
        return dlngammas_c_dxs

    ''' Sympy code used to get these derivatives - not yet validated with numerical values from SymPy!
        Second and third derivative formulas generated with SymPy.
    from sympy import *

    N = 3
    N = range(N)
    xs = x0, x1, x2 = symbols('x0, x1, x2')
    rs = r0, r1, r2 = symbols('r0, r1, r2')
    qs = q0, q1, q2 = symbols('q0, q1, q2') # Pure component property (made from subgroups, but known)

    rsxs = sum([rs[i]*xs[i] for i in range(N)])
    Vis = [rs[i]/rsxs for i in range(N)]

    qsxs = sum([qs[i]*xs[i] for i in range(N)])
    Fis = [qs[i]/qsxs for i in range(N)]

    Vis = V0, V1, V2 = symbols('V0, V1, V2', cls=Function)
    VisD = V0D, V1D, V2D = symbols('V0D, V1D, V2D', cls=Function)
    Fis = F0, F1, F2 = symbols('F0, F1, F2', cls=Function)
    Vis = [Vis[i](x0, x1, x2) for i in range(N)]
    VisD = [VisD[i](x0, x1, x2) for i in range(N)]
    Fis = [Fis[i](x0, x1, x2) for i in range(N)]

    loggammacs = [1 - VisD[i] + log(VisD[i]) - 5*qs[i]*(1 - Vis[i]/Fis[i]
                  + log(Vis[i]/Fis[i])) for i in range(N)]
    # Variable to use for substitutions
    Vi, ViD, Fi, xj, xk, xm, qi = symbols('V_i, Vi\', F_i, x_j, x_k, x_m, q_i')

    # First derivative
    good_first = diff(loggammacs[0], x1).subs(V0(x0, x1, x2), Vi).subs(F0(x0, x1, x2), Fi).subs(V0D(x0, x1, x2), ViD).subs(x1, xj).subs(q0, qi)
    good_first = simplify(expand(simplify(good_first)))

    # Second derivative
    good_second = diff(loggammacs[0], x1, x2).subs(V0(x0, x1, x2), Vi).subs(F0(x0, x1, x2), Fi).subs(V0D(x0, x1, x2), ViD).subs(x1, xj).subs(x2, xk).subs(q0, qi)

    # Third derivative
    good_third = diff(loggammacs[0], x0, x1, x2).subs(V0(x0, x1, x2), Vi).subs(F0(x0, x1, x2), Fi).subs(V0D(x0, x1, x2), ViD).subs(x0, xj).subs(x1, xk).subs(x2, xm).subs(q0, qi)
    good_third = simplify(good_third)
    '''

    '''For the Lyngby model composition derivatives remaining:

    from sympy import *
    N = 4
    N = range(N)
    xs = x0, x1, x2, x3 = symbols('x0, x1, x2, x3')
    Vis = V0, V1, V2, V3 = symbols('V0, V1, V2, V3', cls=Function)
    Vis = [Vis[i](x0, x1, x2, x3) for i in range(N)]
    loggammacs = [1 + log(Vis[i]/xs[i]) - Vis[i]/xs[i] for i in range(N)]
    diff(loggammacs[0], xs[1], xs[2])
    '''

    def d2lngammas_c_dxixjs(self):
        r'''Second composition derivative of
        the combinatorial part of the UNIFAC model. For the modified UNIFAC
        model, the equation is as follows; for the original UNIFAC and UNIFAC
        LLE, replace :math:`V_i'` with :math:`V_i`.

        .. math::
            \frac{\partial \ln \gamma^c_i}{\partial x_j \partial x_k} =
            5 q_{i} \left(\frac{- \frac{d^{2}}{d x_{k}d x_{j}} V_{i} + \frac{V_{i}
            \frac{d^{2}}{d x_{k}d x_{j}} F_{i}}{F_{i}} + \frac{\frac{d}{d x_{j}} F_{i}
            \frac{d}{d x_{k}} V_{i}}{F_{i}} + \frac{\frac{d}{d x_{k}} F_{i}
            \frac{d}{d x_{j}} V_{i}}{F_{i}} - \frac{2 V_{i} \frac{d}{d x_{j}}
            F_{i} \frac{d}{d x_{k}} F_{i}}{F_{i}^{2}}}{V_{i}} + \frac{\left(
            \frac{d}{d x_{j}} V_{i} - \frac{V_{i} \frac{d}{d x_{j}} F_{i}}
            {F_{i}}\right) \frac{d}{d x_{k}} V_{i}}{V_{i}^{2}}
            + \frac{\frac{d^{2}}{d x_{k}d x_{j}} V_{i}}{F_{i}} - \frac{\left(
            \frac{d}{d x_{j}} V_{i} - \frac{V_{i} \frac{d}{d x_{j}} F_{i}}{
            F_{i}}\right) \frac{d}{d x_{k}} F_{i}}{F_{i} V_{i}} - \frac{V_{i}
            \frac{d^{2}}{d x_{k}d x_{j}} F_{i}}{F_{i}^{2}} - \frac{\frac{d}
            {d x_{j}} F_{i} \frac{d}{d x_{k}} V_{i}}{F_{i}^{2}}
            - \frac{\frac{d}{d x_{k}} F_{i} \frac{d}{d x_{j}} V_{i}}{F_{i}^{2}}
            + \frac{2 V_{i} \frac{d}{d x_{j}} F_{i} \frac{d}{d x_{k}} F_{i}}
            {F_{i}^{3}}\right) - \frac{d^{2}}{d x_{k}d x_{j}} Vi'
            + \frac{\frac{d^{2}}{d x_{k}d x_{j}} Vi'}{Vi'} - \frac{\frac{d}
            {d x_{j}} Vi' \frac{d}{d x_{k}} Vi'}{Vi'^{2}}

        For the Lyngby model, the following equations are used:

        .. math::
            \frac{\partial^2 \ln \gamma^c_i}{\partial x_j \partial x_k} =
            -\frac{\partial^2 V_i'}{\partial x_j \partial x_k}
            + \frac{1}{V_i'} \frac{\partial^2 V_i'}{\partial x_j \partial x_k}
            - \frac{1}{\left(V_i'\right)^2} \frac{\partial V_i'}{\partial x_j}
             \frac{\partial V_i'}{\partial x_k}

        Returns
        -------
        d2lngammas_c_dxixjs : list[list[list[float]]]
            Combinatorial lngammas term second composition derivative, size
            number of components by number of components by number of
            components, [-]
        '''
        try:
            return self._d2lngammas_c_dxixjs
        except AttributeError:
            pass
        N, version, qs = self.N, self.version, self.qs
        try:
            Vis = self._Vis
        except AttributeError:
            Vis = self.Vis()
        try:
            dVis_dxs = self._dVis_dxs
        except AttributeError:
            dVis_dxs = self.dVis_dxs()
        try:
            d2Vis_dxixjs = self._d2Vis_dxixjs
        except AttributeError:
            d2Vis_dxixjs = self.d2Vis_dxixjs()
        try:
            Fis = self._Fis
        except AttributeError:
            Fis = self.Fis()
        try:
            dFis_dxs = self._dFis_dxs
        except AttributeError:
            dFis_dxs = self.dFis_dxs()
        try:
            d2Fis_dxixjs = self._d2Fis_dxixjs
        except AttributeError:
            d2Fis_dxixjs = self.d2Fis_dxixjs()

        if self.version in (1, 4):
            try:
                Vis_modified = self._Vis_modified
            except AttributeError:
                Vis_modified = self.Vis_modified()
            try:
                dVis_modified_dxs = self._dVis_modified_dxs
            except AttributeError:
                dVis_modified_dxs = self.dVis_modified_dxs()
            try:
                d2Vis_modified_dxixjs = self._d2Vis_modified_dxixjs
            except AttributeError:
                d2Vis_modified_dxixjs = self.d2Vis_modified_dxixjs()
        else:
            Vis_modified = Vis
            dVis_modified_dxs = dVis_dxs
            d2Vis_modified_dxixjs = d2Vis_dxixjs

        if self.scalar:
            d2lngammas_c_dxixjs = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d2lngammas_c_dxixjs = zeros((N, N, N))


        self._d2lngammas_c_dxixjs = unifac_d2lngammas_c_dxixjs(N, version, qs, Fis, dFis_dxs, d2Fis_dxixjs, Vis, dVis_dxs, d2Vis_dxixjs, Vis_modified, dVis_modified_dxs, d2Vis_modified_dxixjs, d2lngammas_c_dxixjs)
        return d2lngammas_c_dxixjs

    def d3lngammas_c_dxixjxks(self):
        r'''Third composition derivative of
        the combinatorial part of the UNIFAC model. For the modified UNIFAC
        model, the equation is as follows; for the original UNIFAC and UNIFAC
        LLE, replace :math:`V_i'` with :math:`V_i`.

        .. math::
            \frac{\partial \ln \gamma^c_i}{\partial x_j \partial x_k
            \partial x_m} = - \frac{d^{3}}{d x_{m}d x_{k}d x_{j}} Vi'
            + \frac{\frac{d^{3}}{d x_{m}d x_{k}d x_{j}} Vi'}{Vi'}
            - \frac{\frac{d}{d x_{j}} Vi' \frac{d^{2}}{d x_{m}d x_{k}} Vi'}
            {Vi'^{2}} - \frac{\frac{d}{d x_{k}} Vi' \frac{d^{2}}{d x_{m}d
            x_{j}} Vi'}{Vi'^{2}} - \frac{\frac{d}{d x_{m}} Vi' \frac{d^{2}}
            {d x_{k}d x_{j}} Vi'}{Vi'^{2}} + \frac{2 \frac{d}{d x_{j}} Vi'
            \frac{d}{d x_{k}} Vi' \frac{d}{d x_{m}} Vi'}{Vi'^{3}} - \frac{5
            q_{i} \frac{d^{3}}{d x_{m}d x_{k}d x_{j}} V_{i}}{V_{i}}
            + \frac{5 q_{i} \frac{d}{d x_{j}} V_{i} \frac{d^{2}}{d x_{m}d x_{k}} V_{i}}{V_{i}^{2}}
            + \frac{5 q_{i} \frac{d}{d x_{k}} V_{i} \frac{d^{2}}{d x_{m}d x_{j}} V_{i}}{V_{i}^{2}}
            + \frac{5 q_{i} \frac{d}{d x_{m}} V_{i} \frac{d^{2}}{d x_{k}d x_{j}} V_{i}}{V_{i}^{2}}
            - \frac{10 q_{i} \frac{d}{d x_{j}} V_{i} \frac{d}{d x_{k}} V_{i} \frac{d}{d x_{m}} V_{i}}{V_{i}^{3}}
            + \frac{5 q_{i} \frac{d^{3}}{d x_{m}d x_{k}d x_{j}} F_{i}}{F_{i}}
            + \frac{5 q_{i} \frac{d^{3}}{d x_{m}d x_{k}d x_{j}} V_{i}}{F_{i}}
            - \frac{5 V_{i} q_{i} \frac{d^{3}}{d x_{m}d x_{k}d x_{j}} F_{i}}{F_{i}^{2}}
            - \frac{5 q_{i} \frac{d}{d x_{j}} F_{i} \frac{d^{2}}{d x_{m}d x_{k}} F_{i}}{F_{i}^{2}}
            - \frac{5 q_{i} \frac{d}{d x_{j}} F_{i} \frac{d^{2}}{d x_{m}d x_{k}} V_{i}}{F_{i}^{2}}
            - \frac{5 q_{i} \frac{d}{d x_{k}} F_{i} \frac{d^{2}}{d x_{m}d x_{j}} F_{i}}{F_{i}^{2}}
            - \frac{5 q_{i} \frac{d}{d x_{k}} F_{i} \frac{d^{2}}{d x_{m}d x_{j}} V_{i}}{F_{i}^{2}}
            - \frac{5 q_{i} \frac{d}{d x_{m}} F_{i} \frac{d^{2}}{d x_{k}d x_{j}} F_{i}}{F_{i}^{2}}
            - \frac{5 q_{i} \frac{d}{d x_{m}} F_{i} \frac{d^{2}}{d x_{k}d x_{j}} V_{i}}{F_{i}^{2}}
            - \frac{5 q_{i} \frac{d}{d x_{j}} V_{i} \frac{d^{2}}{d x_{m}d x_{k}} F_{i}}{F_{i}^{2}}
            - \frac{5 q_{i} \frac{d}{d x_{k}} V_{i} \frac{d^{2}}{d x_{m}d x_{j}} F_{i}}{F_{i}^{2}}
            - \frac{5 q_{i} \frac{d}{d x_{m}} V_{i} \frac{d^{2}}{d x_{k}d x_{j}} F_{i}}{F_{i}^{2}}
            + \frac{10 V_{i} q_{i} \frac{d}{d x_{j}} F_{i} \frac{d^{2}}{d x_{m}d x_{k}} F_{i}}{F_{i}^{3}}
            + \frac{10 V_{i} q_{i} \frac{d}{d x_{k}} F_{i} \frac{d^{2}}{d x_{m}d x_{j}} F_{i}}{F_{i}^{3}}
            + \frac{10 V_{i} q_{i} \frac{d}{d x_{m}} F_{i} \frac{d^{2}}{d x_{k}d x_{j}} F_{i}}{F_{i}^{3}}
            + \frac{10 q_{i} \frac{d}{d x_{j}} F_{i} \frac{d}{d x_{k}} F_{i} \frac{d}{d x_{m}} F_{i}}{F_{i}^{3}}
            + \frac{10 q_{i} \frac{d}{d x_{j}} F_{i} \frac{d}{d x_{k}} F_{i} \frac{d}{d x_{m}} V_{i}}{F_{i}^{3}}
            + \frac{10 q_{i} \frac{d}{d x_{j}} F_{i} \frac{d}{d x_{m}} F_{i} \frac{d}{d x_{k}} V_{i}}{F_{i}^{3}}
            + \frac{10 q_{i} \frac{d}{d x_{k}} F_{i} \frac{d}{d x_{m}} F_{i} \frac{d}{d x_{j}} V_{i}}{F_{i}^{3}}
            - \frac{30 V_{i} q_{i} \frac{d}{d x_{j}} F_{i} \frac{d}{d x_{k}} F_{i} \frac{d}{d x_{m}} F_{i}}{F_{i}^{4}}

        For the Lyngby model, the following equations are used:

        .. math::
            \frac{\partial^3 \ln \gamma^c_i}{\partial x_j \partial x_k \partial
            x_m} = \frac{\partial^3 V_i'}{\partial x_j \partial x_k \partial
            x_m}\left(\frac{1}{V_i'} - 1\right)
            - \frac{1}{(V_i')^2}\left(
            \frac{\partial V_i'}{\partial x_j}\frac{\partial V_i'}{\partial x_k \partial x_m}
            + \frac{\partial V_i'}{\partial x_k}\frac{\partial V_i'}{\partial x_j \partial x_m}
            + \frac{\partial V_i'}{\partial x_m}\frac{\partial V_i'}{\partial x_j \partial x_k}
            \right)
            + \frac{2}{(V_i')^3}\frac{\partial V_i'}{\partial x_j}
            \frac{\partial V_i'}{\partial x_k}\frac{\partial V_i'}{\partial x_m}


        Returns
        -------
        d3lngammas_c_dxixjxks : list[list[list[list[float]]]]
            Combinatorial lngammas term third composition derivative, size
            number of components by number of components by number of
            components by number of components, [-]
        '''
        try:
            return self._d3lngammas_c_dxixjxks
        except AttributeError:
            pass
        N, version, qs = self.N, self.version, self.qs
        Vis = self.Vis()
        dVis_dxs = self.dVis_dxs()
        d2Vis_dxixjs = self.d2Vis_dxixjs()
        d3Vis_dxixjxks = self.d3Vis_dxixjxks()

        Fis = self.Fis()
        dFis_dxs = self.dFis_dxs()
        d2Fis_dxixjs = self.d2Fis_dxixjs()
        d3Fis_dxixjxks = self.d3Fis_dxixjxks()

        if version in (1, 4):
            Vis_modified = self.Vis_modified()
            dVis_modified_dxs = self.dVis_modified_dxs()
            d2Vis_modified_dxixjs = self.d2Vis_modified_dxixjs()
            d3Vis_modified_dxixjxks = self.d3Vis_modified_dxixjxks()
        else:
            Vis_modified = Vis
            dVis_modified_dxs = dVis_dxs
            d2Vis_modified_dxixjs = d2Vis_dxixjs
            d3Vis_modified_dxixjxks = d3Vis_dxixjxks

        if self.scalar:
            d3lngammas_c_dxixjxks = [[[[0.0]*N for _ in range(N)] for _ in range(N)] for _ in range(N)]
        else:
            d3lngammas_c_dxixjxks = zeros((N, N, N, N))

        self._d3lngammas_c_dxixjxks = unifac_d3lngammas_c_dxixjxks(N, version, qs, Fis, dFis_dxs, d2Fis_dxixjs, d3Fis_dxixjxks, Vis, dVis_dxs, d2Vis_dxixjs, d3Vis_dxixjxks, Vis_modified, dVis_modified_dxs, d2Vis_modified_dxixjs, d3Vis_modified_dxixjxks, d3lngammas_c_dxixjxks)
        return d3lngammas_c_dxixjxks

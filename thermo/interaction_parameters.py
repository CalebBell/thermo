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

import numpy as np

'''Need to be able to add NRTL parameters sets (T dept)
Need to add single kijis, and kijs with T dept
'''

import re
file_name = '/home/caleb/Documents/University/CHE3123/thermo/thermo/Phase Change/chemsep/pr.ipd'
opened_file = open(file_name)
file_lines = opened_file.readlines()
file_lines = file_lines[18:]
to_match = re.compile(' +')
all_comment_match = re.compile(' T=[^ ]+| P=[^ ]+|p[0-9]+')
T_comment_match = re.compile(' T=[^ ]+')
P_comment_match = re.compile(' P=[^ ]+')
page_comment_match = re.compile('p[0-9]+')

def comment_parse_condition(s, init='T=', unit='K'):
    Tmin, Tmax = None, None
    if s:
        T = s[0].strip().replace(',', '-').replace(init, '').replace(unit, '')
        if '-' in T:
            Tmin = float(T.split('-')[0])
            Tmax = float(T.split('-')[1])
            T = None
        else:
            T = float(T)
    else:
        T = None
    return T, Tmin, Tmax


for line in file_lines:
    splits = to_match.split(line)
    CAS1, CAS2, kij = splits[0:3]
    kij = float(kij)
    comment = ' '.join(splits[3:]).strip()
    chemical_names = all_comment_match.split(comment)[0].strip()
    s = T_comment_match.findall(comment)
    T, Tmin, Tmax, = comment_parse_condition(s, init='T=', unit='K')
    s = P_comment_match.findall(comment)
    P, Pmin, Pmax, = comment_parse_condition(s, init='P=', unit='bar')
    page_comment = page_comment_match.findall(comment)

    
class InteractionParameterDB(object):
    pass
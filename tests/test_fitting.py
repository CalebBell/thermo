# -*- coding: utf-8 -*-
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
SOFTWARE.'''

from numpy.testing import assert_allclose
import pytest
from thermo import fitting
from thermo.fitting import *
import os
import pandas as pd

def test_Twu91_check_params():
    assert Twu91_check_params((0.694911381318495, 0.919907783415812, 1.70412689631515)) # Ian Bell, methanol
    assert not Twu91_check_params((0.81000842, 0.94790489, 1.49618907)) # Fit without constraints for methanol

    # CH4
    # Twu91_check_params((0.1471, 0.9074, 1.8253))  # Should be consistent - probably a decimal problem
    assert not  Twu91_check_params((0.0777, 0.9288, 3.0432)) # Should be inconsistent

    # N2
    assert Twu91_check_params((0.1240, 0.8897, 2.0138))# consistent
    assert not Twu91_check_params((0.0760, 0.9144, 2.9857)) # inconsistent


def test_Twu91_check_params_Bell():
    folder = os.path.join(os.path.dirname(fitting.__file__), 'Phase Change')

    Bell_2018_data = pd.read_csv(os.path.join(folder, 'Bell 2018 je7b00967_si_001.tsv'),
                                        sep='\t', index_col=6)
    v = Bell_2018_data_values = Bell_2018_data.values

    for (c0, c1, c2) in zip(v[:, 2], v[:, 3], v[:, 4]):
        assert Twu91_check_params((c0, c1, c2))

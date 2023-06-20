'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import os

import pytest

"""
Tests that run aspects of the documentation should go in here.
The only bit included right now are the plots, which should run without an
error; no contents checking is performed.
"""
plots_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'plots')
plot_files = [i for i in os.listdir(plots_folder) if i.endswith('.py')]
print(plot_files)

@pytest.mark.parametrize("file", plot_files)
def test_documentation_plots(file):
    import matplotlib
    matplotlib.use('Agg')
    exec(open(os.path.join(plots_folder, file)).read(), globals())
#

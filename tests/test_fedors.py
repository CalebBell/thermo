'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2022, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import pytest
from fluids.numerics import assert_close

try:
    import rdkit
except:
    rdkit = None
from thermo.group_contribution import Fedors


@pytest.mark.rdkit
@pytest.mark.skipif(rdkit is None, reason="requires rdkit")
def test_Fedors():
    Vc, status, _, _, _ = Fedors('CCC(C)O')
    assert_close(Vc, 0.000274024)
    assert status == 'OK'

    Vc, status, _, _, _ = Fedors('C1=CC=C2C(=C1)C=CC3=CC4=C(C=CC5=CC=CC=C54)C=C32')
    assert_close(Vc, 0.00089246)
    assert status == 'OK'

    Vc, status, _, _, _ = Fedors('C1=CC=C(C=C1)O')
    assert_close(Vc, 0.00026668)
    assert status == 'OK'

    Vc, status, _, _, _ = Fedors('C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12C5=C%11C4=C3C3=C5C(=C81)C%10=C23')
    assert_close(Vc, 0.001969256)
    assert status == 'OK'

    Vc, status, _, _, _ = Fedors('C12C3C4C1C5C2C3C45')
    assert_close(Vc, 0.000485046)
    assert status == 'OK'

    Vc, status, _, _, _ = Fedors('O=[U](=O)=O')
    assert status != 'OK'


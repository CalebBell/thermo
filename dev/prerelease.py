"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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
"""

import os
import shutil
from datetime import datetime


def set_file_modification_time(filename, mtime):
    atime = os.stat(filename).st_atime
    os.utime(filename, times=(atime, mtime.timestamp()))

now = datetime.now()

main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

remove_folders = ('__pycache__', '.mypy_cache', '_build', '.cache', '.ipynb_checkpoints')
bad_extensions = ('.pyc', '.nbi', '.nbc')


paths = [main_dir]

for p in paths:
    for (dirpath, dirnames, filenames) in os.walk(p):
        for bad_folder in remove_folders:
            if dirpath.endswith(bad_folder):
                shutil.rmtree(dirpath)
                continue
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if not os.path.exists(full_path):
                continue
            set_file_modification_time(full_path, now)
            for bad_extension in bad_extensions:
                if full_path.endswith(bad_extension):
                    os.remove(full_path)

# import pytest
# os.chdir(main_dir)
# pytest.main(["--doctest-glob='*.rst'", "--doctest-modules", "--nbval", "-n", "8", "--dist", "loadscope", "-v"])

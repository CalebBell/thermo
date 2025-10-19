"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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
"""
import os
import sys

try:  # pragma: no cover
    from appdirs import user_config_dir
    data_dir = user_config_dir("thermo")
    if not os.path.exists(data_dir):
        try:
            os.mkdir(data_dir)
        except FileNotFoundError:
            os.makedirs(data_dir) # Recursive
except:  # pragma: no cover
    data_dir = os.path.dirname(__file__)

def _get_source_path():
    """Get the base path for package resources.

    Works with PyInstaller, py2exe, cx_Freeze, and normal Python.

    Returns
    -------
    str
        Absolute path to the thermo package directory
    """
    if getattr(sys, "frozen", False):
        # Running as compiled executable
        if hasattr(sys, "_MEIPASS"):
            # PyInstaller >= 2.0
            path = os.path.join(sys._MEIPASS, "thermo")
            return path
        else:
            # py2exe, cx_Freeze - they copy package structure to executable directory
            exe_dir = os.path.dirname(sys.executable)
            # Look for thermo package in lib directory (cx_Freeze pattern)
            lib_path = os.path.join(exe_dir, "lib", "thermo")
            if os.path.exists(lib_path):
                return lib_path
            # Fallback to dist directory (py2exe pattern)
            fallback_path = os.path.join(exe_dir, "thermo")
            return fallback_path
    else:
        # Running in normal Python environment
        path = os.path.dirname(__file__)
        return path

source_path = _get_source_path()

if os.name == "nt":
    def os_path_join(*args) -> str:
        return "\\".join(args)
else:
    def os_path_join(*args) -> str:
        return "/".join(args)

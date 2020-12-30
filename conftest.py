import sys
import platform

def pytest_ignore_collect(path):
    path = str(path)
    if 'manual_runner' in path or 'make_test_stubs' in path or 'plot' in path or 'prerelease' in path:
        return True
    if platform.python_version_tuple()[0:2] < ('3', '6'):
        if 'numba' in path:
            return True
    if sys.version[0] == '2':
        if 'numba' in path or 'typing_utils' in path:
            return True
        #if 'rst' in path:
        #    if platform.python_version_tuple()[0:2] != ('3', '7'):
        #        return True
        if 'test' not in path:
            return True
    if 'ipynb' in path and 'bench' in path:
        return True

def pytest_configure(config):
    import os
    os._called_from_test = True
    
    if sys.version[0] == '3':
        import pytest
        if pytest.__version__.split('.')[0] >= '6':
            #config.addinivalue_line("addopts", '--doctest-modules')
            #config.option.doctestmodules = True
            config.addinivalue_line("doctest_optionflags", "NUMBER")
#        config.addinivalue_line("addopts", config.inicfg['addopts'].replace('//', '') + ' --doctest-modules')
        #config.inicfg['addopts'] = config.inicfg['addopts'] + ' --doctest-modules'
        #
        config.addinivalue_line("doctest_optionflags", "NORMALIZE_WHITESPACE")

import platform
import sys

# Detect Python implementation and version
is_pypy = "PyPy" in sys.version
is_graal = "Graal" in sys.version
is_free_threaded = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()
ver_tup = tuple(int(x) for x in platform.python_version_tuple()[:2])
is_x86_or_x86_64 = platform.machine().lower() in ("i386", "i686", "x86", "x86_64", "amd64")


def pytest_ignore_collect( collection_path, config):
    """Determine which paths pytest should ignore during test collection."""
    # Normalize path to string
    path = str(collection_path)

    # Skip virtual environments and ASV benchmark environments
    if "venv" in path or "site-packages" in path or ".asv" in path:
        return True

    # Serious technical debt - skip certain core files unless in tests
    if ((path.endswith(("chemical.py", "mixture.py", "stream.py")) ) and "test" not in path) or path.endswith("README.rst"):
        return True

    # Skip utility and development directories
    skip_paths = ("cx_freeze", "py2exe", "manual_runner", "make_test_stubs", "plot", "prerelease", "benchmarks", "benchmark", "conf.py", "dev", "dump")
    if any(skip_path in path for skip_path in skip_paths):
        return True

    # Skip notebook benchmarks
    if "ipynb" in path and "bench" in path:
        return True

    # PyPy/GraalVM compatibility exclusions
    if (is_pypy or is_graal) and "test_spa" in path:
        return True

    if is_graal and "units" in path:
        return True

    # Skip numba and .rst tests for unsupported configurations
    # Numba requires: CPython 3.8-3.13, x86/x86_64 architecture, GIL-enabled
    unsupported_for_numba = (
        ver_tup < (3, 8) or
        ver_tup >= (3, 14) or
        is_pypy or
        is_graal or
        is_free_threaded or
        not is_x86_or_x86_64
    )
    if unsupported_for_numba:
        if "numba" in path:
            return True
        # Skip .rst tests due to rendering differences and missing NUMBER flag support
        if ".rst" in path:
            return True

    return False


def pytest_configure(config):
    """Configure pytest options for doctest support."""
    # Only configure for Python 3
    if sys.version[0] != "3":
        return

    import pytest
    pytest_major_version = int(pytest.__version__.split(".")[0])

    # Enable doctest modules for pytest >= 6
    if pytest_major_version >= 6:
        config.addinivalue_line("addopts", "--doctest-modules")
        config.option.doctestmodules = True
        config.addinivalue_line("doctest_optionflags", "NUMBER")

    # Always normalize whitespace in doctests
    config.addinivalue_line("doctest_optionflags", "NORMALIZE_WHITESPACE")

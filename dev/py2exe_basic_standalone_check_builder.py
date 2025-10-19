from setuptools import setup
import py2exe
import sys
import os
import glob

# Get all data files from thermo package (local)
thermo_data = []
thermo_data_dirs = ["Critical Properties", "Density", "Electrolytes", "Environment",
                    "Heat Capacity", "Identifiers", "Law", "Misc", "Phase Change",
                    "Reactions", "Safety", "Solubility", "Interface", "Triple Properties",
                    "Thermal Conductivity", "Interaction Parameters", "Scalar Parameters",
                    "Vapor Pressure", "Viscosity"]

for data_dir in thermo_data_dirs:
    dir_path = os.path.join('..', 'thermo', data_dir)
    if os.path.exists(dir_path):
        files = glob.glob(os.path.join(dir_path, '*'))
        files = [f for f in files if os.path.isfile(f)]
        if files:
            thermo_data.append((f'thermo\\{data_dir}', files))

# Handle Interaction Parameters/ChemSep subdirectory for thermo
chemsep_dir = os.path.join('..', 'thermo', 'Interaction Parameters', 'ChemSep')
if os.path.exists(chemsep_dir):
    files = glob.glob(os.path.join(chemsep_dir, '*'))
    files = [f for f in files if os.path.isfile(f)]
    if files:
        thermo_data.append((r'thermo\Interaction Parameters\ChemSep', files))

# Get all data files from installed chemicals package
chemicals_data = []
try:
    import chemicals
    chemicals_path = os.path.dirname(chemicals.__file__)
    chemicals_data_dirs = ["Critical Properties", "Density", "Electrolytes", "Environment",
                           "Heat Capacity", "Identifiers", "Law", "Misc", "Phase Change",
                           "Reactions", "Safety", "Solubility", "Interface", "Triple Properties",
                           "Thermal Conductivity", "Vapor Pressure", "Viscosity"]

    for data_dir in chemicals_data_dirs:
        dir_path = os.path.join(chemicals_path, data_dir)
        if os.path.exists(dir_path):
            files = glob.glob(os.path.join(dir_path, '*'))
            files = [f for f in files if os.path.isfile(f)]
            if files:
                chemicals_data.append((f'chemicals\\{data_dir}', files))
                print(f"Adding {len(files)} files from chemicals/{data_dir}")
except ImportError:
    print("Warning: chemicals package not found, data files may be missing")

# Combine all data files
all_data_files = thermo_data + chemicals_data

setup(
    console=['basic_standalone_thermo_check.py'],
    packages=[],
    py_modules=[],
    data_files=all_data_files,
    options={
        'py2exe': {
            'packages': ['thermo', 'fluids', 'chemicals', 'numpy', 'scipy'],
            'bundle_files': 3,  # Don't bundle - scipy/numpy have compiled extensions
            'compressed': True,
            'optimize': 0,
            'excludes': ["cairo", "locket", "setproctitle", "bcrypt", "beniget",
                "curses", "et_xmlfile", "google", "imagesize", "olefile", "pyasn1_modules", "pytest",
                "tabulate", "tlz", "xxhash", "_pydevd_frame_eval", "astunparse", "backcall", "constantly",
                "cssselect", "greenlet", "incremental", "iniconfig", "ipywidgets",
                "matplotlib_inline", "ply", "pydoc_data", "pygtkcompat", "pyximport", "tblib",
                "typed_ast", "yapf", "zope", "asgiref", "blib2to3", "certifi", "cloudpickle",
                "dbm", "jupyter_core", "kiwisolver", "lz4", "ptyprocess", "PySide2", "snappy",
                "sortedcontainers", "toml", "tomli", "tomllib", "zoneinfo", "blosc", "ephem",
                "exceptiongroup", "gast", "jacobi", "lazy_object_proxy", "llvmlite",
                "mpi4py", "mpl_toolkits", "msgpack", "OpenSSL", "past", "pydevd_plugins", "smmap",
                "wrapt", "wsgiref", "argcomplete", "bs4", "executing", "ipython_genutils",
                "markupsafe", "mdurl", "pure_eval", "pyasn1", "PyQt5", "qtpy", "service_identity",
                "zstandard", "asttokens", "bytecode", "colorama", "contourpy", "idna", "numexpr",
                "PyQt6", "soupsieve", "stack_data", "wcwidth", "nacl", "pycparser", "traitlets",
                "alabaster", "cvxopt", "fastjsonschema", "pexpect", "pluggy", "simplejson",
                "tkinter", "torchgen", "defusedxml", "monkeytype", "av", "charset_normalizer",
                "IPython", "opt_einsum", "psutil", "sphinxcontrib", "toolz", "torchvision",
                "xlrd", "zict", "docutils", "girepository-1.0", "gitdb", "jedi", "jsonschema",
                "parso", "pyparsing", "wx", "html5lib", "lib2to3", "dill", "partd",
                "tqdm", "babel", "click", "gi", "git", "pydev_ipython", "pyrsistent", "zmq",
                "nbformat", "odf", "torchaudio", "Cython", "fsspec", "pygments", "requests",
                "yaml", "django", "invoke", "markdown_it", "black", "graphviz", "jaxlib",
                "sqlalchemy", "hypothesis", "openpyxl", "attr", "_pydev_bundle", "fontTools",
                "jinja2", "jupyter_client", "pyglet", "joblib", "twisted", "patsy", "ipykernel",
                "pvlib", "statsmodels", "tornado", "pythran", "snowballstemmer",
                "tables", "h5py", "prompt_toolkit", "sphinx", "coverage", "dask", "jax",
                "setuptools", "numba", "sympy", "chardet", "paramiko", "distributed",
                "gevent", "rich", "torch", "matplotlib", "pyarrow", "PIL"],
        }
    },
    zipfile=None,
)

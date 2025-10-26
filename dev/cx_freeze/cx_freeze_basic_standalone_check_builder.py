from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": ["numpy", "scipy", "thermo", "fluids", "chemicals"],
    "excludes": ["cairo", "locket", "setproctitle", "bcrypt", "beniget", "concurrent", 
    "curses", "et_xmlfile", "google", "imagesize", "olefile", "pyasn1_modules", "pytest", 
    "tabulate", "tlz", "xxhash", "_pydevd_frame_eval", "astunparse", "backcall", "constantly",
    "cssselect", "greenlet", "html", "incremental", "iniconfig", "ipywidgets", 
    "matplotlib_inline", "ply", "pydoc_data", "pygtkcompat", "pyximport", "tblib",
    "typed_ast", "xmlrpc", "yapf", "zope", "asgiref", "blib2to3", "certifi", "cloudpickle",
    "dbm", "jupyter_core", "kiwisolver", "lz4", "ptyprocess", "PySide2", "snappy", 
    "sortedcontainers", "toml", "tomli", "tomllib", "zoneinfo", "blosc", "ephem", 
    "exceptiongroup", "gast", "http", "jacobi", "lazy_object_proxy", "llvmlite",
    "mpi4py", "mpl_toolkits", "msgpack", "OpenSSL", "past", "pydevd_plugins", "smmap",
    "wrapt", "wsgiref", "xml", "argcomplete", "bs4", "executing", "ipython_genutils",
    "markupsafe", "mdurl", "pure_eval", "pyasn1", "PyQt5", "qtpy", "service_identity", 
    "zstandard", "asttokens", "bytecode", "colorama", "contourpy", "idna", "numexpr", 
    "PyQt6", "soupsieve", "stack_data", "wcwidth", "nacl", "pycparser", "traitlets", 
    "alabaster", "cvxopt", "fastjsonschema", "pexpect", "pluggy", "simplejson", 
    "tkinter", "torchgen", "defusedxml", "monkeytype", "av", "charset_normalizer", 
    "IPython", "opt_einsum", "psutil", "sphinxcontrib", "toolz", "torchvision", 
    "xlrd", "zict", "docutils", "girepository-1.0", "gitdb", "jedi", "jsonschema", 
    "parso", "pyparsing", "wx", "html5lib", "lib2to3", "cffi", "dill", "partd", 
    "tqdm", "babel", "click", "gi", "git", "pydev_ipython", "pyrsistent", "zmq", 
    "nbformat", "odf", "torchaudio", "Cython", "fsspec", "pygments", "requests", 
    "yaml", "django", "invoke", "markdown_it", "black", "graphviz", "jaxlib", 
    "sqlalchemy", "hypothesis", "openpyxl", "attr", "_pydev_bundle", "fontTools", 
    "jinja2", "jupyter_client", "pyglet", "joblib", "twisted", "patsy", "ipykernel", 
    "pvlib", "statsmodels", "tornado", "pythran", "snowballstemmer", "asyncio", 
    "tables", "h5py", "prompt_toolkit", "sphinx", "coverage", "dask", "jax", 
    "setuptools", "numba", "sympy", "chardet", "paramiko", "distributed", 
    "gevent", "rich", "torch", "matplotlib", "pyarrow", "PIL"],
    "include_files": []
}

base = None

executables = [
    Executable("../basic_standalone_thermo_check.py", base=base)
]

setup(
    name="basic_thermo_check",
    version="0.1",
    description="A simple cx_Freeze script that includes numpy and scipy",
    options={"build_exe": build_exe_options},
    executables=executables
)

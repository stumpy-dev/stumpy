import ast
import importlib
import os.path
import pathlib
import types
from importlib.metadata import distribution
from site import getsitepackages

from numba import cuda

from . import cache, config

# Define which functions belong to which module
# Key: function name to expose at top level
# Value: name of the module
_lazy_imports = {
    "aamp": "aamp",
    "aamp_mmotifs": "aamp_mmotifs",
    "aamp_match": "aamp_motifs",
    "aamp_motifs": "aamp_motifs",
    "aamp_ostinato": "aamp_ostinato",
    "aamp_ostinatoed": "aamp_ostinato",
    "aamp_stimp": "aamp_stimp",
    "aamp_stimped": "aamp_stimp",
    "aampdist": "aampdist",
    "aampdisted": "aampdist",
    "aampdist_snippets": "aampdist_snippets",
    "aamped": "aamped",
    "aampi": "aampi",
    "allc": "chains",
    "atsc": "chains",
    "mass": "core",
    "floss": "floss",
    "fluss": "floss",
    "maamp": "maamp",
    "maamp_mdl": "maamp",
    "maamp_subspace": "maamp",
    "maamped": "maamped",
    "mmotifs": "mmotifs",
    "match": "motifs",
    "motifs": "motifs",
    "mpdist": "mpdist",
    "mpdisted": "mpdist",
    "mdl": "mstump",
    "mstump": "mstump",
    "subspace": "mstump",
    "mstumped": "mstumped",
    "ostinato": "ostinato",
    "ostinatoed": "ostinato",
    "prescraamp": "scraamp",
    "scraamp": "scraamp",
    "prescrump": "scrump",
    "scrump": "scrump",
    "snippets": "snippets",
    "stimp": "stimp",
    "stimped": "stimp",
    "stump": "stump",
    "stumped": "stumped",
    "stumpi": "stumpi",
}

# Get the default fastmath flags for all njit functions
# and update the _STUMPY_DEFAULTS dictionary


def _get_fastmath_value(module_name, func_name):  # pragma: no cover
    fname = module_name + ".py"
    fname = pathlib.Path(__file__).parent / fname
    with open(fname, "r", encoding="utf-8") as f:
        src = f.read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                for dec in node.decorator_list:
                    for kw in dec.keywords:
                        if kw.arg == "fastmath":
                            fastmath_flag = ast.get_source_segment(src, kw.value)
                            return eval(fastmath_flag)


njit_funcs = cache.get_njit_funcs()
for module_name, func_name in njit_funcs:
    key = module_name + "." + func_name  # e.g., core._mass
    key = "STUMPY_FASTMATH_" + key.upper()  # e.g., STUMPY_FASTHMATH_CORE._MASS
    config._STUMPY_DEFAULTS[key] = _get_fastmath_value(module_name, func_name)

if cuda.is_available():
    _lazy_imports.update(
        {
            "gpu_aamp": "gpu_aamp",
            "gpu_aamp_ostinato": "gpu_aamp_ostinato",
            "gpu_aamp_stimp": "gpu_aamp_stimp",
            "gpu_aampdist": "gpu_aampdist",
            "gpu_mpdist": "gpu_mpdist",
            "gpu_ostinato": "gpu_ostinato",
            "gpu_stimp": "gpu_stimp",
            "gpu_stump": "gpu_stump",
        }
    )
else:  # pragma: no cover
    from . import core
    from .core import _gpu_aamp_driver_not_found as gpu_aamp  # noqa: F401
    from .core import (  # noqa: F401
        _gpu_aamp_ostinato_driver_not_found as gpu_aamp_ostinato,
    )
    from .core import _gpu_aamp_stimp_driver_not_found as gpu_aamp_stimp  # noqa: F401
    from .core import _gpu_aampdist_driver_not_found as gpu_aampdist  # noqa: F401
    from .core import _gpu_mpdist_driver_not_found as gpu_mpdist  # noqa: F401
    from .core import _gpu_ostinato_driver_not_found as gpu_ostinato  # noqa: F401
    from .core import _gpu_stimp_driver_not_found as gpu_stimp  # noqa: F401
    from .core import _gpu_stump_driver_not_found as gpu_stump  # noqa: F401

    core._gpu_searchsorted_left = core._gpu_searchsorted_left_driver_not_found
    core._gpu_searchsorted_right = core._gpu_searchsorted_right_driver_not_found

    # Fix GPU-STUMP Docs
    gpu_stump.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_stump.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_stump":
            gpu_stump.__doc__ = ast.get_docstring(fd)

    # Fix GPU-AAMP Docs
    gpu_aamp.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_aamp.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_aamp":
            gpu_aamp.__doc__ = ast.get_docstring(fd)

    # Fix GPU-OSTINATO Docs
    gpu_ostinato.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_ostinato.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_ostinato":
            gpu_ostinato.__doc__ = ast.get_docstring(fd)

    # Fix GPU-AAMP-OSTINATO Docs
    gpu_aamp_ostinato.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_aamp_ostinato.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_aamp_ostinato":
            gpu_aamp_ostinato.__doc__ = ast.get_docstring(fd)

    # Fix GPU-MPDIST Docs
    gpu_mpdist.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_mpdist.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_mpdist":
            gpu_mpdist.__doc__ = ast.get_docstring(fd)

    # Fix GPU-AAMPDIST Docs
    gpu_aampdist.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_aampdist.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    function_definitions = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    for fd in function_definitions:
        if fd.name == "gpu_aampdist":
            gpu_aampdist.__doc__ = ast.get_docstring(fd)

    # Fix GPU-STIMP Docs
    # Note that this is a special case for class definitions.
    # See above for function definitions.
    # Also, please update docs/api.rst
    gpu_stimp.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_stimp.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
    for cd in class_definitions:
        if cd.name == "gpu_stimp":
            gpu_stimp.__doc__ = ast.get_docstring(cd)

    # Fix GPU-AAMP-STIMP Docs
    # Note that this is a special case for class definitions.
    # See above for function definitions.
    # Also, please update docs/api.rst
    gpu_aamp_stimp.__doc__ = ""
    filepath = pathlib.Path(__file__).parent / "gpu_aamp_stimp.py"

    file_contents = ""
    with open(filepath, encoding="utf8") as f:
        file_contents = f.read()
    module = ast.parse(file_contents)
    class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
    for cd in class_definitions:
        if cd.name == "gpu_aamp_stimp":
            gpu_aamp_stimp.__doc__ = ast.get_docstring(cd)

try:
    # _dist = get_distribution("stumpy")
    _dist = distribution("stumpy")
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(getsitepackages()[0])
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, "stumpy")):
        # not installed, but there is another version that *is*
        raise ModuleNotFoundError  # pragma: no cover
except ModuleNotFoundError:  # pragma: no cover
    __version__ = "Please install this project with setup.py"
else:  # pragma: no cover
    __version__ = _dist.version


# PEP 562: module-level __getattr__ for lazy imports
def __getattr__(name):  # pragma: no cover
    if name in _lazy_imports:
        mod_name = _lazy_imports[name]
        module = importlib.import_module(f"{__package__}.{mod_name}")
        # Retrieve the attribute from the loaded module and cache it
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Ensure that if a module was imported during package import
# (causing the package attribute to point to the module object), we
# replace that entry with the actual attribute (e.g., function) so that
# users get the expected callable at `stumpy.aamp` rather than the module.
for _name, _sub in _lazy_imports.items():  # pragma: no cover
    val = globals().get(_name)
    if isinstance(val, types.ModuleType):
        try:
            replacement = getattr(val, _name)
        except AttributeError:
            # Nothing to do if the module doesn't define the attribute
            continue
        globals()[_name] = replacement


# Eagerly import exports that would otherwise collide with
# same-named modules. This keeps lazy imports for most names but
# ensures that when a top-level exported name exactly matches its
# module (e.g., `stump` -> `stump.py`), the exported attribute is
# available immediately so REPL completers prefer the callable/class
# instead of the module.
for _name, _sub in _lazy_imports.items():  # pragma: no cover
    try:
        if _name == _sub:
            filepath = pathlib.Path(__file__).parent / f"{_sub}.py"
            if filepath.exists():
                module = importlib.import_module(f"{__package__}.{_sub}")
                try:
                    globals()[_name] = getattr(module, _name)
                except AttributeError:
                    # If the module doesn't define the attribute, keep it lazy
                    pass
    except Exception:
        # Be conservative: don't let eager-import attempts raise during package import
        pass


def __dir__():  # pragma: no cover
    # Expose lazy names in dir() for discoverability
    # Also include __all__ so tools that consult it will see the intended
    # top-level exports (this helps some REPL completers prefer the
    # callable/class exports over same-named modules).
    all_names = list(globals().keys()) + list(_lazy_imports.keys())
    all_names += list(globals().get("__all__", []))
    return sorted(all_names)


# Make the lazy-exported names explicit for tools that respect __all__.
# This helps REPL tab-completion prefer functions/classes over modules
# when names collide (e.g., `stumpy.stump` should point to the function
# rather than the module during completion).
__all__ = sorted(list(_lazy_imports.keys()))

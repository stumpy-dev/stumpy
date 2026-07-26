import importlib

import numba
import numpy as np
import pytest

from stumpy import _get_fastmath_value, cache, fastmath


def test_set():
    # The test is done by changing the value of fastmath flag for
    # the fastmath._add_assoc function, taken from the following link:
    # https://numba.pydata.org/numba-doc/dev/user/performance-tips.html#fastmath

    # case1: flag=False
    fastmath._set("fastmath", "_add_assoc", flag=False)
    cache._recompile()
    out = fastmath._add_assoc(0, np.inf)
    assert np.isnan(out)

    # case2: flag={'reassoc', 'nsz'}
    fastmath._set("fastmath", "_add_assoc", flag={"reassoc", "nsz"})
    cache._recompile()
    out = fastmath._add_assoc(0, np.inf)
    if numba.config.DISABLE_JIT:
        assert np.isnan(out)
    else:  # pragma: no cover
        assert out == 0.0

    # case3: flag={'reassoc'}
    fastmath._set("fastmath", "_add_assoc", flag={"reassoc"})
    cache._recompile()
    out = fastmath._add_assoc(0, np.inf)
    assert np.isnan(out)

    # case4: flag={'nsz'}
    fastmath._set("fastmath", "_add_assoc", flag={"nsz"})
    cache._recompile()
    out = fastmath._add_assoc(0, np.inf)
    assert np.isnan(out)


def test_reset():
    # The test is done by changing the value of fastmath flag for
    # the fastmath._add_assoc function, taken from the following link:
    # https://numba.pydata.org/numba-doc/dev/user/performance-tips.html#fastmath
    # and then reset it to the default value, i.e. `True`
    fastmath._set("fastmath", "_add_assoc", False)
    cache._recompile()
    fastmath._reset("fastmath", "_add_assoc")
    cache._recompile()
    if numba.config.DISABLE_JIT:
        assert np.isnan(fastmath._add_assoc(0.0, np.inf))
    else:  # pragma: no cover
        assert fastmath._add_assoc(0.0, np.inf) == 0.0


@pytest.mark.skipif(numba.config.DISABLE_JIT, reason="JIT Disabled")
def test_get_fastmath_value():  # pragma: no cover
    njit_funcs = cache.get_njit_funcs()
    for module_name, func_name in njit_funcs:
        module = importlib.import_module(f".{module_name}", package="stumpy")
        func = getattr(module, func_name)
        ref = func.targetoptions["fastmath"]
        cmp = _get_fastmath_value(module_name, func_name)
        assert ref == cmp

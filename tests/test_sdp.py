import inspect
import warnings
from operator import eq, lt

import naive
import pytest
from numpy import testing as npt

from stumpy import rng, sdp

# README
# Real FFT algorithm performs more efficiently when the length
# of the input array `arr` is composed of small prime factors.
# The next_fast_len(arr, real=True) function from Scipy returns
# the same length if len(arr) is composed of a subset of
# prime numbers 2, 3, 5. Therefore, these radices are
# considered as the most efficient for the real FFT algorithm.

# To ensure that the tests cover different cases, the following cases
# are considered:
# 1. len(T) is even, and len(T) == next_fast_len(len(T), real=True)
# 2. len(T) is odd, and len(T) == next_fast_len(len(T), real=True)
# 3. len(T) is even, and len(T) < next_fast_len(len(T), real=True)
# 4. len(T) is odd, and len(T) < next_fast_len(len(T), real=True)
# And 5. a special case of 1, where len(T) is power of 2.

# Therefore:
# 1. len(T) is composed of 2 and a subset of {3, 5}
# 2. len(T) is composed of a subset of {3, 5}
# 3. len(T) is composed of a subset of {7, 11, 13, ...} and 2
# 4. len(T) is composed of a subset of {7, 11, 13, ...}
# 5. len(T) is power of 2

# In some cases, the prime factors are raised to a power of
# certain degree to increase the length of array to be around
# 1000-2000. This allows us to test sliding_dot_product for
# wider range of query lengths.

# test cases 1-4
test_inputs = [
    # Input format:
    # (
    #     len(T),
    #     remainder,  #  from `len(T) % 2`
    #     comparator,  # for len(T) comparator next_fast_len(len(T), real=True)
    # )
    (
        2 * (3**2) * (5**3),
        0,
        eq,
    ),  # = 2250, Even `len(T)`, and `len(T) == next_fast_len(len(T), real=True)`
    (
        (3**2) * (5**3),
        1,
        eq,
    ),  # = 1125, Odd `len(T)`, and `len(T) == next_fast_len(len(T), real=True)`.
    (
        2 * 7 * 11 * 13,
        0,
        lt,
    ),  # = 2002, Even `len(T)`, and `len(T) < next_fast_len(len(T), real=True)`
    (
        7 * 11 * 13,
        1,
        lt,
    ),  # = 1001, Odd `len(T)`, and `len(T) < next_fast_len(len(T), real=True)`
]


def get_sdp_function_names():
    out = []
    for func_name, func in inspect.getmembers(sdp, callable):
        if func_name.endswith("sliding_dot_product") and inspect.signature(
            func
        ).parameters.keys() >= {"Q", "T"}:
            out.append(func_name)

    return out


@pytest.mark.parametrize("n_T, remainder, comparator", test_inputs)
def test_sdp(n_T, remainder, comparator):
    # test_sdp for cases 1-4

    n_Q_prime = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
    n_Q_power2 = [2, 4, 8, 16, 32, 64]
    n_Q_values = n_Q_prime + n_Q_power2 + [n_T]
    n_Q_values = sorted(n_Q for n_Q in set(n_Q_values) if n_Q <= n_T)

    for n_Q in n_Q_values:
        Q = rng.RNG.rand(n_Q)
        T = rng.RNG.rand(n_T)
        ref = naive.rolling_window_dot_product(Q, T)
        for func_name in get_sdp_function_names():
            func = getattr(sdp, func_name)
            try:
                comp = func(Q, T)
                npt.assert_allclose(comp, ref)
            except Exception as e:  # pragma: no cover
                msg = f"Error in {func_name}, with n_Q={n_Q} and n_T={n_T}"
                warnings.warn(msg)
                raise e


def test_sdp_power2():
    # test for case 5. len(T) is power of 2
    pmin = 3
    pmax = 13

    for func_name in get_sdp_function_names():
        func = getattr(sdp, func_name)
        try:
            for q in range(pmin, pmax + 1):
                n_Q = 2**q
                for p in range(q, pmax + 1):
                    n_T = 2**p
                    Q = rng.RNG.rand(n_Q)
                    T = rng.RNG.rand(n_T)

                    ref = naive.rolling_window_dot_product(Q, T)
                    comp = func(Q, T)
                    npt.assert_allclose(comp, ref)

        except Exception as e:  # pragma: no cover
            msg = f"Error in {func_name}, with q={q} and p={p}"
            warnings.warn(msg)
            raise e

    return


def test_pyfftw_sdp_init_len():
    if not sdp.PYFFTW_IS_AVAILABLE:  # pragma: no cover
        pytest.skip("Skipping Test pyFFTW Not Installed")

    # When `len(T)` larger than original `init_len`,
    # the callable object returned by `_make_pyfftw_sliding_dot_product`
    # should still work correctly. This test checks that functionality.

    init_len = 2**10
    _pyfftw_sliding_dot_product = sdp._make_pyfftw_sliding_dot_product(
        init_len=init_len, real_dtype="float64"
    )

    # len(T) > init_len to trigger internal array resizing
    T = np.random.rand(init_len + 1)
    Q = np.random.rand(2**2)

    comp = _pyfftw_sliding_dot_product(Q, T)
    ref = naive.rolling_window_dot_product(Q, T)

    np.testing.assert_allclose(comp, ref)

    return


def test_pyfftw_sdp_longdoube():
    if not sdp.PYFFTW_IS_AVAILABLE:  # pragma: no cover
        pytest.skip("Skipping Test pyFFTW Not Installed")

    # This test checks that the callable object
    # returned by `_make_pyfftw_sliding_dot_product`
    # can support `real_dtype="longdouble"`
    init_len = 2**10
    sdp_func = sdp._make_pyfftw_sliding_dot_product(
        init_len=init_len, real_dtype="longdouble"
    )

    T = np.random.rand(init_len)
    Q = np.random.rand(2**8)

    comp = sdp_func(Q, T)
    ref = naive.rolling_window_dot_product(Q, T)

    np.testing.assert_allclose(comp, ref)

    return


def test_pyfftw_sdp_multithreaded():
    if not sdp.PYFFTW_IS_AVAILABLE:  # pragma: no cover
        pytest.skip("Skipping Test pyFFTW Not Installed")

    # This test checks that the callable object
    # returned by `_make_pyfftw_sliding_dot_product`
    # can support multithreading.
    T = np.random.rand(2**5)
    Q = np.random.rand(2**4)

    comp = sdp._pyfftw_sliding_dot_product(Q, T, n_threads=2)
    ref = naive.rolling_window_dot_product(Q, T)

    np.testing.assert_allclose(comp, ref)

    return


def test_pyfftw_sdp_multithreaded_longdouble():
    if not sdp.PYFFTW_IS_AVAILABLE:  # pragma: no cover
        pytest.skip("Skipping Test pyFFTW Not Installed")

    # This test checks that the callable object
    # returned by `_make_pyfftw_sliding_dot_product`
    # can support  multithreading and `real_dtype="longdouble"`
    init_len = 2**10
    sdp_func = sdp._make_pyfftw_sliding_dot_product(
        init_len=init_len, real_dtype="longdouble"
    )

    T = np.random.rand(init_len)
    Q = np.random.rand(2**8)

    comp = sdp_func(Q, T, n_threads=2)
    ref = naive.rolling_window_dot_product(Q, T)

    np.testing.assert_allclose(comp, ref)

    return

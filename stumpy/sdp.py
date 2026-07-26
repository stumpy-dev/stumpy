import numpy as np
from numba import njit
from scipy.fft import next_fast_len
from scipy.signal import convolve, oaconvolve

from . import config

# _duccfft replaced _pocketfft in scipy 1.18
try:
    from scipy.fft._duccfft.basic import c2r, r2c
except ModuleNotFoundError:  # pragma: no cover
    from scipy.fft._pocketfft.basic import c2r, r2c


@njit(fastmath=config.STUMPY_FASTMATH_TRUE)
def _njit_sliding_dot_product(Q, T):
    """
    A Numba JIT-compiled implementation of the sliding window dot product.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    out : numpy.ndarray
        Sliding dot product between `Q` and `T`.
    """
    m = len(Q)
    l = len(T) - m + 1
    out = np.empty(l)
    for i in range(l):
        result = 0.0
        for j in range(m):
            result += Q[j] * T[i + j]
        out[i] = result

    return out


def _convolve_sliding_dot_product(Q, T):
    """
    Use (direct or FFT) convolution to calculate the sliding window dot product.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    output : numpy.ndarray
        Sliding dot product between `Q` and `T`.
    """
    return convolve(np.flipud(Q), T, mode="valid")


def _oaconvolve_sliding_dot_product(Q, T):
    """
    Use scipy's oaconvolve to calculate the sliding dot product.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    output : numpy.ndarray
        Sliding dot product between `Q` and `T`.
    """
    return oaconvolve(np.ascontiguousarray(Q[::-1]), T, mode="valid")


def _pocketfft_sliding_dot_product(Q, T):
    """
    Use scipy.fft._pocketfft to compute
    the sliding dot product.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    output : numpy.ndarray
        Sliding dot product between `Q` and `T`.
    """
    n = len(T)
    m = len(Q)
    next_fast_n = next_fast_len(n, real=True)

    tmp = np.empty((2, next_fast_n))
    tmp[0, :m] = Q[::-1]
    tmp[0, m:] = 0.0
    tmp[1, :n] = T
    tmp[1, n:] = 0.0
    fft_2d = r2c(True, tmp, axis=-1)

    return c2r(False, np.multiply(fft_2d[0], fft_2d[1]), n=next_fast_n)[m - 1 : n]


def _sliding_dot_product(Q, T):
    """
    Compute the sliding dot product between `Q` and `T`

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence

    T : numpy.ndarray
        Time series or sequence

    Returns
    -------
    out : numpy.ndarray
        Sliding dot product between `Q` and `T`.
    """
    return _convolve_sliding_dot_product(Q, T)

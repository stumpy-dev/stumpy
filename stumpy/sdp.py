# This file contains different implementations of
# the sliding dot product (sdp). The name of any
# callable object that computes the sliding dot product
# should end with 'sliding_dot_product'.

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

try:  # pragma: no cover
    import pyfftw

    PYFFTW_IS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYFFTW_IS_AVAILABLE = False


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


def _make_pyfftw_sliding_dot_product(max_n=2**20, real_dtype="float64"):
    """
    A closure to compute the sliding dot product using FFTW via pyfftw.

    This closure uses FFTW (via pyfftw) to efficiently compute the sliding dot product
    between a query sequence, Q, and a time series, T. It preallocates arrays and caches
    FFTW objects to optimize repeated computations with similar-sized inputs.

    Parameters
    ----------
    max_n : int, default 2**20
        Maximum length to preallocate arrays for. This will be the size of the
        real-valued array. A complex-valued array of size 1 + (max_n // 2)
        will also be preallocated. If inputs exceed this size, arrays will be
        reallocated to accommodate larger sizes.

    real_dtype : str, default "float64"
        The real data type to use for the preallocated arrays. Must be either
        "float64" or "longdouble". The complex data type will be set to
        "complex128" or "clongdouble", respectively.

    Returns
    -------
    sliding_dot_product : callable
        A callable that computes the sliding dot product between Q and T using FFTW
        via pyfftw, and caches FFTW objects if not already cached. The callable
        automatically resizes the preallocated arrays if the inputs exceed
        the current maximum size. In addition, the callable has a method
        `reset_arr(max_n)` to reset the preallocated arrays to a new maximum length.

    Notes
    -----
    The closure maintains internal caches of FFTW objects to avoid redundant planning
    operations when called multiple times with similar-sized inputs and parameters.
    When planning_flag == "FFTW_ESTIMATE", there will be no planning operation.
    However, caching FFTW objects is still beneficial as the overhead of creating
    those objects can be avoided in subsequent calls.

    References
    ----------
    FFTW documentation: http://www.fftw.org/
    pyfftw documentation: https://pyfftw.readthedocs.io/
    """
    REAL_TO_COMPLEX_MAP = {
        "float64": "complex128",
        "longdouble": "clongdouble",
    }
    if real_dtype not in ["float64", "longdouble"]:  # pragma: no cover
        raise ValueError(
            f"Invalid real_dtype: {real_dtype}. Must be 'float64' or 'longdouble'."
        )
    complex_dtype = REAL_TO_COMPLEX_MAP[real_dtype]

    # Preallocate arrays
    real_arr = pyfftw.empty_aligned(max_n, dtype=real_dtype)
    complex_arr = pyfftw.empty_aligned(1 + (max_n // 2), dtype=complex_dtype)

    # Store FFTW objects, keyed by (next_fast_n, n_threads, planning_flag)
    rfft_objects = {}
    irfft_objects = {}

    def sliding_dot_product(Q, T, n_threads=1, planning_flag="FFTW_ESTIMATE"):
        """
        Compute the sliding dot product between Q and T using FFTW via pyfftw,
        and cache FFTW objects if not already cached.

        Parameters
        ----------
        Q : numpy.ndarray
            Query array or subsequence.

        T : numpy.ndarray
            Time series or sequence.

        n_threads : int, default=1
            Number of threads to use for FFTW computations.

        planning_flag : str, default="FFTW_ESTIMATE"
            The planning flag that will be used in FFTW for planning.
            See pyfftw documentation for details. Current options, ordered
            ascendingly by the level of aggressiveness in planning, are:
            "FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", and "FFTW_EXHAUSTIVE".
            The more aggressive the planning, the longer the planning time, but
            the faster the execution time.

        Returns
        -------
        out : numpy.ndarray
            Sliding dot product between Q and T.

        Notes
        -----
        The planning_flag is defaulted to "FFTW_ESTIMATE" to be aligned with
        MATLAB's FFTW usage (as of version R2025b)
        See: https://www.mathworks.com/help/matlab/ref/fftw.html

        This implementation is inspired by the answer on StackOverflow:
        https://stackoverflow.com/a/30615425/2955541
        """
        nonlocal real_arr, complex_arr

        m = Q.shape[0]
        n = T.shape[0]
        next_fast_n = pyfftw.next_fast_len(n)

        # Update preallocated arrays if needed
        if next_fast_n > len(real_arr):
            real_arr = pyfftw.empty_aligned(next_fast_n, dtype=real_arr.dtype)
            complex_arr = pyfftw.empty_aligned(
                1 + (next_fast_n // 2), dtype=complex_arr.dtype
            )

        real_view = real_arr[:next_fast_n]
        complex_view = complex_arr[: 1 + (next_fast_n // 2)]

        # Get or create FFTW objects
        key = (next_fast_n, n_threads, planning_flag)

        rfft_obj = rfft_objects.get(key, None)
        irfft_obj = irfft_objects.get(key, None)

        if rfft_obj is None or irfft_obj is None:
            rfft_obj = pyfftw.FFTW(
                input_array=real_view,
                output_array=complex_view,
                direction="FFTW_FORWARD",
                flags=(planning_flag,),
                threads=n_threads,
            )
            irfft_obj = pyfftw.FFTW(
                input_array=complex_view,
                output_array=real_view,
                direction="FFTW_BACKWARD",
                flags=(planning_flag, "FFTW_DESTROY_INPUT"),
                threads=n_threads,
            )
            rfft_objects[key] = rfft_obj
            irfft_objects[key] = irfft_obj
        else:
            # Update the input and output arrays of the cached FFTW objects
            # in case their original input and output arrays were reallocated
            # in a previous call
            rfft_obj.update_arrays(real_view, complex_view)
            irfft_obj.update_arrays(complex_view, real_view)

        # Compute the (circular) convolution between T and Q[::-1],
        # each zero-padded to length next_fast_n by performing
        # the following three steps:

        # Step 1
        # Compute RFFT of T (zero-padded)
        # Must make a copy of output to avoid losing it when the array is
        # overwritten when computing the RFFT of Q in the next step
        rfft_obj.input_array[:n] = T
        rfft_obj.input_array[n:] = 0.0
        rfft_obj.execute()
        rfft_T = rfft_obj.output_array.copy()

        # Step 2
        # Compute RFFT of Q (reversed, scaled, and zero-padded)
        # Scaling is required because the thin wrapper execute
        # that will be called below does not perform normalization
        np.multiply(Q[::-1], 1.0 / next_fast_n, out=rfft_obj.input_array[:m])
        rfft_obj.input_array[m:] = 0.0
        rfft_obj.execute()
        rfft_Q = rfft_obj.output_array

        # Step 3
        # Convert back to time domain by taking the inverse RFFT
        np.multiply(rfft_T, rfft_Q, out=irfft_obj.input_array)
        irfft_obj.execute()

        return irfft_obj.output_array[m - 1 : n]  # valid portion

    def reset_arr(max_n):  # pragma: no cover
        """
        Reset the preallocated arrays to a new maximum length.

        Parameters
        ----------
        max_n : int
            New maximum length for the preallocated arrays.

        Returns
        -------
        None
        """
        nonlocal real_arr, complex_arr
        complex_dtype = REAL_TO_COMPLEX_MAP[real_dtype]
        real_arr = pyfftw.empty_aligned(max_n, dtype=real_dtype)
        complex_arr = pyfftw.empty_aligned(1 + (max_n // 2), dtype=complex_dtype)

    sliding_dot_product.reset_arr = reset_arr
    return sliding_dot_product


if PYFFTW_IS_AVAILABLE:  # pragma: no cover
    _pyfftw_sliding_dot_product = _make_pyfftw_sliding_dot_product(
        max_n=2**20, real_dtype="float64"
    )


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

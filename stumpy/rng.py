import os
from contextlib import contextmanager

import numpy as np

# Note that an initial SEED = 0 is disallowed
# in order to account for unit testing
if os.getenv("STUMPY_SEED") is not None:  # pragma: no cover
    SEED = int(os.getenv("STUMPY_SEED"))
    if SEED == 0:
        raise ValueError("STUMPY_SEED must be greater than zero!")
else:
    SEED = np.random.randint(1, 4_294_967_296, dtype=np.uint32)
RNG = np.random.RandomState(seed=SEED)


@contextmanager
def fix_seed(seed):
    """
    A context manager for setting the RNG seed to a fixed, hardcoded, safe seed
    and then returning the RNG back to its previous state prior to the seed change

    This is typically used when you want to generate a specific random sequence once.
    To repeat the same random sequence, use `fix_state` instead. If you are picking
    a random seed directly before calling `fix_seed` then you probably want to use
    `fix_state` instead!

    Parameters
    ----------
    seed : int
        The random seed for (temporarily) setting the random number generator to

    Returns
    -------
    None
    """
    curr_state = RNG.get_state()
    RNG.seed(seed)
    try:
        yield
    finally:
        RNG.set_state(curr_state)


@contextmanager
def fix_state(state=None):
    """
    A context manager for setting the RNG state to a fixed, hardcoded, safe state
    and then returning the RNG back to its previous state prior to the state change

    This is typically used when you want to repeat the same random sequence more than
    once. Alternatively, if a specific Mersenne Twister state is needed, it
    can be passed in temporarily and then the state will return back to the state
    prior to the change upon completion.

    Parameters
    ----------
    state : tuple, default None
        A NumPy legacy mersenne twister state

    Returns
    -------
    None
    """
    curr_state = RNG.get_state()
    if state is not None:
        RNG.set_state(state)
    try:
        yield
    finally:
        RNG.set_state(curr_state)

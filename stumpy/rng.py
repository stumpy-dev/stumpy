import json
import os
import warnings
from contextlib import contextmanager

import numpy as np

# Note that an initial SEED = 0 is disallowed
# in order to account for unit testing
if os.getenv("STUMPY_SEED") is not None:  # pragma: no cover
    SEED = int(os.getenv("STUMPY_SEED"))
else:
    SEED = np.random.randint(1, 4_294_967_296, dtype=np.uint32)
RNG = np.random.RandomState(seed=SEED)

if os.getenv("STUMPY_STATE") is not None:  # pragma: no cover
    if os.getenv("STUMPY_SEED") is not None:  # pragma: no cover
        warnings.warn("STUMPY_SEED was ignored in lieu of STUMPY_STATE")
    state_str = os.getenv("STUMPY_STATE")
    state = json.loads(state_str)
    STATE = (
        state[0],
        np.array(state[1], dtype=np.uint32),
        state[2],
        state[3],
        state[4],
    )
    RNG.set_state(STATE)
else:
    STATE = RNG.get_state()

# seed = RNG.get_state()[1][0]


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
def fix_state():
    """
    A context manager for setting the RNG state to a fixed, hardcoded, safe state
    and then returning the RNG back to its previous state prior to the state change

    This is typically used when you want to repeat the same random sequence more than
    once.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    curr_state = RNG.get_state()
    try:
        yield
    finally:
        RNG.set_state(curr_state)

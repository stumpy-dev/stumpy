from contextlib import contextmanager

import numpy as np

# Note that an initial SEED = 0 is disallowed
# in order to account for unit testing
SEED = np.random.randint(1, 4_294_967_295, dtype=np.uint32)
RNG = np.random.RandomState(seed=SEED)


def set_seed(seed):
    """
    Permanently set the RNG seed to a different value

    Parameters
    ----------
    seed : int
        The random seed for (permanently) setting the random number generator to

    Returns
    -------
    None
    """
    global SEED
    global RNG
    SEED = seed
    RNG = np.random.RandomState(seed=SEED)


@contextmanager
def fix_seed(seed):
    """
    A context manager for setting the RNG seed to a fixed, hardcoded, safe seed
    and then returning the RNG back to its previous state prior to the seed change

    Parameters
    ----------
    seed : int
        The random seed for (temporarily) setting the random number generator to

    Returns
    -------
    None
    """
    state = RNG.get_state()
    RNG.seed(seed)
    try:
        yield
    finally:
        RNG.set_state(state)

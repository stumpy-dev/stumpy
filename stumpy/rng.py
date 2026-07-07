import numpy as np

bit_gen = np.random.PCG64()
RNG = np.random.Generator(bit_gen)
# RNG.bit_generator.state = {
#     Set/paste RNG state here
# }

PREV_STATE = None
FIXED_STATE = {
    # DO NOT CHANGE/ALTER!!
    "bit_generator": "PCG64",
    "state": {
        "state": 195349167630453735115769518810051464980,
        "inc": 247589055400886363559049235690497450585,
    },
    "has_uint32": 0,
    "uinteger": 0,
}


def _get_state():
    """
    Get a copy of the current RNG state

    Parameters
    ----------
    None

    Returns
    -------
    state : dict
        A copy of the current RNG state
    """
    return RNG.bit_generator.state.copy()


def _set_state(state):
    """
    Store existing RNG state and set RNG state

    Parameters
    ----------
    state : dict
        The RNG state to set

    Returns
    -------
    None
    """
    global PREV_STATE
    PREV_STATE = RNG.bit_generator.state.copy()
    RNG.bit_generator.state = state


def _reset_state():
    """
    Restore the RNG state to the last recorded RNG state

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    global PREV_STATE
    RNG.bit_generator.state = PREV_STATE.copy()
    PREV_STATE = None


def _fix_state():
    """
    Set the RNG state to a fixed, hardcoded, safe state

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    _set_state(FIXED_STATE)


get_state = _get_state
set_state = _set_state
fix_state = _fix_state
unfix_state = _reset_state

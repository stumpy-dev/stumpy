# This file acts as an entry point for pytest.

# Its root directory is added to `sys.path` when pytest is executed
# to fix eventual module import errors that can arise, for example when
# running tests from inside VS code.
# See https://stackoverflow.com/a/34520971
import json

import numpy as np
import pytest

pytest.RNG = np.random.default_rng()
# pytest.RNG.bit_generator.state = {
#     Set RNG state here
# }

pytest.LAST_RNG_STATE = None
pytest.FIXED_RNG_STATE = {
    # DO NOT CHANGE/ALTER!!
    "bit_generator": "PCG64",
    "state": {
        "state": 195349167630453735115769518810051464980,
        "inc": 247589055400886363559049235690497450585,
    },
    "has_uint32": 0,
    "uinteger": 0,
}


def pytest_configure(config):
    """
    Called after command line options have been parsed
    and all plugins and initial conftest files been loaded.
    """
    state = json.dumps(pytest.RNG.bit_generator.state, indent=4)
    print(f"conftest.py: pytest.RNG.bit_generator.state = {state}")


def _get_rng_state():
    """
    Get a copy of the current RNG state
    """
    return pytest.RNG.bit_generator.state.copy()


def _set_rng_state(state):
    """
    Store existing RNG state and set RNG state
    """
    pytest.LAST_RNG_STATE = pytest.RNG.bit_generator.state.copy()
    pytest.RNG.bit_generator.state = state


def _reset_rng_state():
    """
    Restore the RNG state to the last recorded RNG state
    """
    pytest.RNG.bit_generator.state = pytest.LAST_RNG_STATE.copy()
    pytest.LAST_RNG_STATE = None


def _fix_rng_state():
    """
    Set the RNG state to a fixed, hardcoded, safe state
    """
    _set_rng_state(pytest.FIXED_RNG_STATE)


pytest.get_rng_state = _get_rng_state
pytest.set_rng_state = _set_rng_state
pytest.fix_rng_state = _fix_rng_state
pytest.unfix_rng_state = _reset_rng_state

# This file acts as an entry point for pytest.

# Its root directory is added to `sys.path` when pytest is executed
# to fix eventual module import errors that can arise, for example when
# running tests from inside VS code.
# See https://stackoverflow.com/a/34520971

import json

import pytest

from stumpy import rng


def pytest_configure(config):
    """
    Called after command line options have been parsed
    and all plugins and initial conftest files been loaded.
    """
    state = rng.STATE
    state_str = json.dumps(
        (
            state[0],
            state[1].tolist(),
            state[2],
            state[3],
            state[4],
        )
    )

    # Store details of starting random state in case of failure
    pytest.STUMPY_MSG = (
        f"\n\nSTUMPY_STATE='{state_str}' pixi run tests custom {config.args[0]}"
    )


def pytest_sessionfinish(session, exitstatus):
    """
    Upon test failure, print additional seed and state
    for reproducing test failure
    """
    # exitstatus 1 or greater usually indicates failures/errors
    if session.testsfailed > 0:  # pragma: no cover
        print(f"{pytest.STUMPY_MSG}")

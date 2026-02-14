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


def pytest_configure(config):
    """
    Called after command line options have been parsed
    and all plugins and initial conftest files been loaded.
    """
    state = json.dumps(pytest.RNG.bit_generator.state, indent=4)
    print(f"conftest.py: pytest.RNG.bit_generator.state = {state}")

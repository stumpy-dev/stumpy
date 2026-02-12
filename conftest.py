# This file acts as an entry point for pytest.

# Its root directory is added to `sys.path` when pytest is executed
# to fix eventual module import errors that can arise, for example when
# running tests from inside VS code.
# See https://stackoverflow.com/a/34520971

import os

import numpy as np

# Set STUMPY_SEED for reproducible test failures.
# If not already set (e.g., by test.sh), generate a random seed.
# The seed is printed so it can be noted and reused to reproduce failures:
#   STUMPY_SEED=12345 pytest tests/test_stump.py
if "STUMPY_SEED" not in os.environ:
    os.environ["STUMPY_SEED"] = str(np.random.randint(2**31))


def pytest_configure(config):
    """Print STUMPY_SEED so failed tests can be reproduced."""
    print(f"\nSTUMPY_SEED={os.environ['STUMPY_SEED']}")

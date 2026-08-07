# This file acts as an entry point for pytest.

# Its root directory is added to `sys.path` when pytest is executed
# to fix eventual module import errors that can arise, for example when
# running tests from inside VS code.
# See https://stackoverflow.com/a/34520971

import os
import platform
from importlib.metadata import PackageNotFoundError, version

import pytest

from stumpy import rng


def get_specs():
    """
    Find and return all  package versions
    """
    pkgs = [
        # Alphabetical Order
        "black",
        "coverage",
        "dask",
        "distributed",
        "flake8",
        "isort",
        "numba",
        "numpy",
        "pandas",
        "polars",
        "python",
        "pytest",
        "ray",
        "scipy",
    ]

    specs = []
    for pkg in pkgs:
        try:  # pragma: no cover
            pkg_version = version(pkg)
            specs.append(f"--spec {pkg}={pkg_version}")
        except PackageNotFoundError:
            if pkg == "python":
                specs.append(f"--spec {pkg}={platform.python_version()}")
            pass

    return " ".join(specs)


def get_env_vars():
    """
    Find and return all environment variables
    """
    keys = [
        "NUMBA_DISABLE_JIT",
        "NUMBA_ENABLE_CUDASIM",
    ]

    env_vars = []
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            env_vars.append(f"{key}={value}")

    return " ".join(env_vars)


def pytest_configure(config):
    """
    Called after command line options have been parsed
    and all plugins and initial conftest files been loaded.
    """
    # Store details of starting random seed in case of failure
    env_vars = get_env_vars()
    specs = get_specs()
    pytest.STUMPY_MSG = f"\n\nSTUMPY_SEED={rng.SEED} {env_vars} "
    pytest.STUMPY_MSG += f"pixi exec {specs} ./test.sh custom 1 {config.args[0]}"


def pytest_sessionfinish(session, exitstatus):
    """
    Upon test failure, print additional seed and state
    for reproducing test failure
    """
    # exitstatus 1 or greater usually indicates failures/errors
    if session.testsfailed > 0:  # pragma: no cover
        print(f"{pytest.STUMPY_MSG}")

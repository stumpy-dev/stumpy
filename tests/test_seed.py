import os

import numpy as np
import numpy.testing as npt


def test_stumpy_seed_env_var_is_set():
    """conftest.py should always set STUMPY_SEED in the environment."""
    assert "STUMPY_SEED" in os.environ
    seed_val = os.environ["STUMPY_SEED"]
    assert seed_val.isdigit()
    assert 0 <= int(seed_val) < 2**31


def test_stumpy_seed_produces_deterministic_rng():
    """Same STUMPY_SEED should produce identical random sequences."""
    seed = int(os.environ["STUMPY_SEED"])
    np.random.seed(seed)
    arr1 = np.random.uniform(-1000, 1000, [64])
    np.random.seed(seed)
    arr2 = np.random.uniform(-1000, 1000, [64])
    npt.assert_array_equal(arr1, arr2)

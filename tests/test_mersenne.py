from itertools import combinations

import mersenne
import numpy as np
import pytest


@pytest.mark.parametrize("keys", list(combinations(mersenne.STATE_DICT, 2)))
def test_unique_states(keys):
    assert not np.array_equal(
        mersenne.seed_to_state(keys[0])[1], mersenne.seed_to_state(keys[1])[1]
    )

import mersenne
import numpy as np
import numpy.testing as npt

from stumpy import rng


def test_fix_seed():
    init_state = rng.RNG.get_state()

    with rng.fix_seed(0):
        ref_state = mersenne.seed_to_state(0)
        cmp_state = rng.RNG.get_state()

        npt.assert_array_equal(cmp_state[1], ref_state[1])
        assert np.any(
            cmp_state[1] != init_state[1]
        )  # Assert at least one element is different


def test_random():
    with rng.fix_seed(0):
        assert rng.RNG.rand() == 0.5488135039273248
        assert rng.RNG.randint(1_000_000) == 435829
        assert rng.RNG.uniform(0, 1_000_000) == 844265.7485810174
        npt.assert_almost_equal(
            rng.RNG.permutation([10, 20, 30, 40, 50]), [10, 30, 20, 50, 40]
        )
        npt.assert_almost_equal(
            rng.RNG.choice([10, 20, 30, 40, 50], 10, replace=True),
            [30, 50, 10, 10, 50, 30, 20, 10, 20, 20],
        )
        assert rng.RNG.normal() == 0.44386323274542566

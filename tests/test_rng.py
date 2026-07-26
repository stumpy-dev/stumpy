import numpy.testing as npt

from stumpy import rng


def test_set_seed():
    init_state = rng.RNG.get_state()

    rng.set_seed(0)
    seed = rng.RNG.get_state()[1][0]
    assert seed == 0

    rng.RNG.set_state(init_state)


def test_fix_seed():
    init_state = rng.RNG.get_state()
    init_seed = init_state[1][0]  # This returns the seed

    with rng.fix_seed(0):
        state = rng.RNG.get_state()
        seed = state[1][0]
        assert seed == 0
        assert seed != init_seed

    state = rng.RNG.get_state()
    seed = state[1][0]
    assert seed == init_seed


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

import numpy.testing as npt

from stumpy import rng


def test_fix_state():
    rng._fix_state()
    state = rng.get_state()
    assert state == rng.FIXED_STATE

    rng.unfix_state()


def test_random():
    rng.fix_state()

    assert rng.RNG.random() == 0.1442355276650238
    assert rng.RNG.integers(1_000_000) == 616778
    assert rng.RNG.uniform(0, 1_000_000) == 945097.9509917531
    npt.assert_almost_equal(
        rng.RNG.permutation([10, 20, 30, 40, 50]), [20, 40, 10, 50, 30]
    )
    npt.assert_almost_equal(
        rng.RNG.choice([10, 20, 30, 40, 50], 10, replace=True),
        [40, 10, 50, 20, 50, 10, 20, 10, 20, 50],
    )
    assert rng.RNG.normal() == -0.1655864933503086

    rng.unfix_state()

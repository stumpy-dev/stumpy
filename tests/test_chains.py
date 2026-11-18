import numpy as np
import numpy.testing as npt
import pytest

from stumpy.chains import allc, atsc

test_data = [
    (
        np.array([47, 32, 1, 22, 2, 58, 3, 36, 4, -5, 5, 40], dtype=np.int64),
        np.array([11, 7, 4, 7, 6, 11, 8, 11, 10, 10, 11, -1], dtype=np.int64),
        np.array([-1, 0, 1, 1, 2, 0, 4, 1, 6, 2, 8, 7], dtype=np.int64),
    )
]


@pytest.mark.parametrize("Value, IR, IL", test_data)
def test_atsc(Value, IR, IL):
    j = 2
    ref_chain = np.array([2, 4, 6, 8, 10], np.int64)
    cmp_chain = atsc(IL, IR, j)
    npt.assert_array_equal(cmp_chain, ref_chain)


@pytest.mark.parametrize("Value, IR, IL", test_data)
def test_allc(Value, IR, IL):
    ref_chains = [
        np.array([1, 7, 11], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([3], dtype=np.int64),
        np.array([9], dtype=np.int64),
        np.array([2, 4, 6, 8, 10], dtype=np.int64),
        np.array([5], dtype=np.int64),
    ]
    ref_longest_chain = np.array([2, 4, 6, 8, 10], dtype=np.int64)
    cmp_chains, cmp_longest_chain = allc(IL, IR)

    ref_chains = sorted(ref_chains, key=lambda x: (len(x), list(x)))
    cmp_chains = sorted(cmp_chains, key=lambda x: (len(x), list(x)))

    npt.assert_equal(cmp_chains, ref_chains)
    npt.assert_array_equal(cmp_longest_chain, ref_longest_chain)

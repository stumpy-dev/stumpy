import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import config
from stumpy.mmotifs import mmotifs

test_data = [
    np.array(
        [
            [5.2, 0.1, 3.5, 3.4, 7.1, 9.8, 3.7, 5.0, 2.1, 4.3, 7.5, 6.8, 8.0, 8.1, 1.2],
            [
                7.3,
                3.2,
                5.0,
                9.1,
                8.2,
                7.3,
                4.8,
                8.2,
                10.0,
                0.0,
                4.1,
                3.2,
                2.3,
                0.1,
                1.4,
            ],
            [6.2, 7.6, 7.6, 8.4, 1.1, 5.9, 9.2, 8.5, 9.3, 4.6, 3.5, 0.0, 3.1, 5.3, 0.9],
            [
                0.1,
                1.3,
                3.0,
                2.1,
                6.2,
                1.3,
                9.5,
                10.0,
                1.8,
                2.0,
                2.1,
                5.2,
                1.3,
                0.5,
                4.3,
            ],
        ]
    )
]


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_with_default_parameters(T):
    ref_motif_distances = np.array([[0.0000000e00, 1.1151008e-07]])
    ref_motif_indices = np.array([[2, 9]])
    ref_motif_subspaces = [np.array([1])]
    ref_motif_mdls = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = mmotifs(T, P, I)

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, rtol=0, atol=1.5e-06)
    npt.assert_array_equal(cmp_motif_indices, ref_motif_indices)
    npt.assert_array_equal(cmp_motif_subspaces, ref_motif_subspaces)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, rtol=0, atol=1.5e-06)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_max_matches_none(T):
    ref_motif_distances = np.array([[0.0000000e00, 1.1151008e-07]])
    ref_motif_indices = np.array([[2, 9]])
    ref_motif_subspaces = [np.array([1])]
    ref_motif_mdls = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = mmotifs(T, P, I, max_matches=None)

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, rtol=0, atol=1.5e-06)
    npt.assert_array_equal(cmp_motif_indices, ref_motif_indices)
    npt.assert_array_equal(cmp_motif_subspaces, ref_motif_subspaces)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, rtol=0, atol=1.5e-06)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_more_motifs_when_cutoffs_3(T):
    ref_motif_distances = np.array([[0.0000000e00, 1.1151008e-07]])
    ref_motif_indices = np.array([[2, 9]])
    ref_motif_subspaces = [np.array([1])]
    ref_motif_mdls = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = mmotifs(T, P, I, cutoffs=3, max_motifs=10)

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, rtol=0, atol=1.5e-06)
    npt.assert_array_equal(cmp_motif_indices, ref_motif_indices)
    npt.assert_array_equal(cmp_motif_subspaces, ref_motif_subspaces)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, rtol=0, atol=1.5e-06)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_more_motifs_cutoffs_is_list(T):
    ref_motif_distances = np.array([[0.0000000e00, 1.1151008e-07]])
    ref_motif_indices = np.array([[2, 9]])
    ref_motif_subspaces = [np.array([1])]
    ref_motif_mdls = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    cutoffs = [2, 3, 4, 5]
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = mmotifs(T, P, I, cutoffs=cutoffs, max_motifs=10)

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, rtol=0, atol=1.5e-06)
    npt.assert_array_equal(cmp_motif_indices, ref_motif_indices)
    npt.assert_array_equal(cmp_motif_subspaces, ref_motif_subspaces)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, rtol=0, atol=1.5e-06)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_max_matches_2_k_1(T):
    ref_motif_distances = np.array([[0.0, 0.20948156]])
    ref_motif_indices = np.array([[2, 9]])
    ref_motif_subspaces = [np.array([1, 3])]
    ref_motif_mdls = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = mmotifs(T, P, I, max_distance=np.inf, max_matches=2, k=1)

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, rtol=0, atol=1.5e-06)
    npt.assert_array_equal(cmp_motif_indices, ref_motif_indices)
    npt.assert_array_equal(cmp_motif_subspaces, ref_motif_subspaces)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, rtol=0, atol=1.5e-06)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_two_motif_pairs_max_motifs_2(T):
    ref_motif_distances = np.array(
        [[0.00000000e00, 1.11510080e-07], [1.68587394e-07, 2.58694429e-01]]
    )
    ref_motif_indices = np.array([[2, 9], [6, 1]])
    ref_motif_subspaces = [np.array([1]), np.array([2])]
    ref_motif_mdls = [
        np.array([232.0, 250.57542476, 260.0, 271.3509059]),
        np.array([264.0, 280.0, 299.01955001, 310.51024953]),
    ]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.mstump(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = mmotifs(
        T, P, I, cutoffs=np.inf, max_motifs=2, max_distance=np.inf, max_matches=2
    )

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, rtol=0, atol=1.5e-06)
    npt.assert_array_equal(cmp_motif_indices, ref_motif_indices)
    npt.assert_array_equal(cmp_motif_subspaces, ref_motif_subspaces)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, rtol=0, atol=1.5e-06)


@pytest.mark.parametrize("T", test_data)
def test_mmotifs_with_default_parameters_with_isconstant(T):
    ref_motif_distances = np.array([[0.0000000e00, 1.1151008e-07]])
    ref_motif_indices = np.array([[2, 9]])
    ref_motif_subspaces = [np.array([1])]
    ref_motif_mdls = [np.array([232.0, 250.57542476, 260.0, 271.3509059])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    # The following `T_subseq_isconstant` is basically equivalent to
    # `T_subseq_isconstant=None` (default). The goal is to test its
    # functionality.
    T_subseq_isconstant = [
        None,
        naive.rolling_isconstant(T[1], m),
        None,
        naive.is_ptp_zero_1d,
    ]

    P, I = naive.mstump(T, m, excl_zone, T_subseq_isconstant=T_subseq_isconstant)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = mmotifs(T, P, I, T_subseq_isconstant=T_subseq_isconstant)

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, rtol=0, atol=1.5e-06)
    npt.assert_array_equal(cmp_motif_indices, ref_motif_indices)
    npt.assert_array_equal(cmp_motif_subspaces, ref_motif_subspaces)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, rtol=0, atol=1.5e-06)

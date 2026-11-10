import mersenne
import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import config, rng
from stumpy.aamp_mmotifs import aamp_mmotifs

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


def test_aamp_mmotifs_default_parameters():
    ref_motif_distances = np.array(
        [[0.0, 0.06315749, 0.25275899, 0.34087884, 0.3452315]]
    )
    ref_motif_indices = np.array([[19, 77, 63, 52, 71]])
    ref_motif_subspaces = [np.array([2])]
    ref_motif_mdls = [
        np.array([411.60964047, 423.69925001, 449.11032383, 476.95855027, 506.62406252])
    ]

    state = mersenne.seed_to_state(0)
    with rng.fix_state(state):
        T = rng.RNG.rand(500).reshape(5, 100)

        m = 5
        excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
        P, I = naive.maamp(T, m, excl_zone)
        (
            cmp_motif_distances,
            cmp_motif_indices,
            cmp_motif_subspaces,
            cmp_motif_mdls,
        ) = aamp_mmotifs(T, P, I)

        npt.assert_allclose(cmp_motif_distances, ref_motif_distances, atol=1.5e-07)
        npt.assert_allclose(cmp_motif_indices, ref_motif_indices, atol=1.5e-07)
        npt.assert_allclose(cmp_motif_subspaces, ref_motif_subspaces, atol=1.5e-07)
        npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_max_distance(T):
    ref_motif_distances = np.array(
        [[0.0, 1.41421356, 4.46430286, 6.85346628, 8.207923, 8.50529247]]
    )
    ref_motif_indices = np.array([[2, 9, 0, 11, 7, 5]])
    ref_motif_subspaces = [np.array([3])]
    ref_motif_mdls = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = aamp_mmotifs(T, P, I, max_distance=np.inf)

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_indices, ref_motif_indices, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_subspaces, ref_motif_subspaces, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_max_distance_max_matches_none(T):
    ref_motif_distances = np.array(
        [[0.0, 1.41421356, 4.46430286, 6.85346628, 8.207923, 8.50529247]]
    )
    ref_motif_indices = np.array([[2, 9, 0, 11, 7, 5]])
    ref_motif_subspaces = [np.array([3])]
    ref_motif_mdls = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = aamp_mmotifs(T, P, I, max_distance=np.inf, max_matches=None)

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_indices, ref_motif_indices, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_subspaces, ref_motif_subspaces, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_max_motifs_1_max_matches_2_k_1(T):
    ref_motif_distances = np.array([[0.0, 2.87778559]])
    ref_motif_indices = np.array([[0, 5]])
    ref_motif_subspaces = [np.array([2, 1])]
    ref_motif_mdls = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = aamp_mmotifs(T, P, I, max_distance=np.inf, max_matches=2, k=1)

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_indices, ref_motif_indices, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_subspaces, ref_motif_subspaces, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_more_motif_pairs_cutoffs_3(T):
    ref_motif_distances = np.array([[0.0, 1.41421356], [0.0, 2.06639783]])
    ref_motif_indices = np.array([[2, 9], [0, 5]])
    ref_motif_subspaces = [np.array([3]), np.array([2])]
    ref_motif_mdls = [
        np.array([244.0, 260.67970001, 279.86313714, 281.35940001]),
        np.array([254.33985, 260.67970001, 279.86313714, 291.20703549]),
    ]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = aamp_mmotifs(
        T, P, I, max_distance=np.inf, cutoffs=3, max_matches=2, max_motifs=10
    )

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_indices, ref_motif_indices, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_subspaces, ref_motif_subspaces, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_more_motif_pairs_cutoffs_as_list(T):
    ref_motif_distances = np.array([[0.0, 1.41421356]])
    ref_motif_indices = np.array([[2, 9]])
    ref_motif_subspaces = [np.array([3])]
    ref_motif_mdls = [np.array([244.0, 260.67970001, 279.86313714, 281.35940001])]

    m = 4
    cutoffs = [2, 3, 4, 5]
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = aamp_mmotifs(
        T, P, I, max_distance=np.inf, cutoffs=cutoffs, max_matches=2, max_motifs=10
    )

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_indices, ref_motif_indices, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_subspaces, ref_motif_subspaces, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
def test_aamp_mmotifs_two_motif_pairs_max_motifs_2(T):
    ref_motif_distances = np.array([[0.0, 1.41421356], [0.0, 2.06639783]])
    ref_motif_indices = np.array([[2, 9], [0, 5]])
    ref_motif_subspaces = [np.array([3]), np.array([2])]
    ref_motif_mdls = [
        np.array([244.0, 260.67970001, 279.86313714, 281.35940001]),
        np.array([254.33985, 260.67970001, 279.86313714, 291.20703549]),
    ]

    m = 4
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    P, I = naive.maamp(T, m, excl_zone)
    (
        cmp_motif_distances,
        cmp_motif_indices,
        cmp_motif_subspaces,
        cmp_motif_mdls,
    ) = aamp_mmotifs(
        T,
        P,
        I,
        max_distance=np.inf,
        cutoffs=np.inf,
        max_matches=2,
        max_motifs=2,
    )

    npt.assert_allclose(cmp_motif_distances, ref_motif_distances, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_indices, ref_motif_indices, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_subspaces, ref_motif_subspaces, atol=1.5e-07)
    npt.assert_allclose(cmp_motif_mdls, ref_motif_mdls, atol=1.5e-07)

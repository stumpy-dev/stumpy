import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import config, rng
from stumpy.aampdist_snippets import _get_all_aampdist_profiles, aampdist_snippets

test_data = [rng.RNG.uniform(-1000, 1000, size=64).astype(np.float64)]
s = [6, 7, 8]
percentage = [0.7, 0.8, 0.9]
m = [8, 9, 10]
k = [1, 2, 3]


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
def test_get_all_aampdist_profiles(T, m):
    ref_profiles = naive.get_all_aampdist_profiles(T, m)
    cmp_profiles = _get_all_aampdist_profiles(T, m)

    npt.assert_almost_equal(
        ref_profiles, cmp_profiles, decimal=config.STUMPY_TEST_PRECISION
    )


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("s", s)
def test_get_all_aampdist_profiles_s(T, m, s):
    ref_profiles = naive.get_all_aampdist_profiles(T, m, s=s)
    cmp_profiles = _get_all_aampdist_profiles(T, m, s=s)

    npt.assert_almost_equal(
        ref_profiles, cmp_profiles, decimal=config.STUMPY_TEST_PRECISION
    )


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("k", k)
def test_aampdist_snippets(T, m, k):
    for p in [1.0, 2.0, 3.0]:
        D = _get_all_aampdist_profiles(
            T,
            m,
            p=p,
        )
        (
            ref_aampdist_snippets,
            ref_indices,
            ref_profiles,
            ref_fractions,
            ref_areas,
            ref_regimes,
        ) = naive.aampdist_snippets(
            T,
            m,
            k,
            p=p,
            D=D,
        )
        (
            cmp_aampdist_snippets,
            cmp_indices,
            cmp_profiles,
            cmp_fractions,
            cmp_areas,
            cmp_regimes,
        ) = aampdist_snippets(T, m, k, p=p)

        npt.assert_allclose(
            cmp_aampdist_snippets,
            ref_aampdist_snippets,
            atol=1.5 * 10**-config.STUMPY_TEST_PRECISION,
        )
        npt.assert_allclose(
            cmp_indices, ref_indices, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
        )
        # npt.assert_allclose(
        #     cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
        # )
        npt.assert_allclose(
            cmp_fractions, ref_fractions, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
        )
        # npt.assert_allclose(
        #     cmp_areas, ref_areas, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
        # )
        npt.assert_allclose(cmp_regimes, ref_regimes, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("k", k)
@pytest.mark.parametrize("percentage", percentage)
def test_aampdist_snippets_percentage(T, m, k, percentage):
    D = _get_all_aampdist_profiles(
        T,
        m,
        percentage=percentage,
    )
    (
        ref_aampdist_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.aampdist_snippets(T, m, k, percentage=percentage, D=D)
    (
        cmp_aampdist_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = aampdist_snippets(T, m, k, percentage=percentage)

    npt.assert_allclose(
        cmp_aampdist_snippets,
        ref_aampdist_snippets,
        atol=1.5 * 10**-config.STUMPY_TEST_PRECISION,
    )
    npt.assert_allclose(
        cmp_indices, ref_indices, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    # npt.assert_allclose(
    #     cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    # )
    npt.assert_allclose(
        cmp_fractions, ref_fractions, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    # npt.assert_allclose(
    #     cmp_areas, ref_areas, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    # )
    npt.assert_allclose(cmp_regimes, ref_regimes, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("k", k)
@pytest.mark.parametrize("s", s)
def test_aampdist_snippets_s(T, m, k, s):
    D = _get_all_aampdist_profiles(
        T,
        m,
        s=s,
    )
    (
        ref_aampdist_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.aampdist_snippets(T, m, k, s=s, D=D)
    (
        cmp_aampdist_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = aampdist_snippets(T, m, k, s=s)

    npt.assert_allclose(
        cmp_aampdist_snippets,
        ref_aampdist_snippets,
        atol=1.5 * 10**-config.STUMPY_TEST_PRECISION,
    )
    npt.assert_allclose(
        cmp_indices, ref_indices, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    # npt.assert_allclose(
    #     cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    # )
    npt.assert_allclose(
        cmp_fractions, ref_fractions, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    # npt.assert_allclose(
    #     cmp_areas, ref_areas, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    # )
    npt.assert_allclose(cmp_regimes, ref_regimes, atol=1.5e-07)

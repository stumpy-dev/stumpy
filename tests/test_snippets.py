import functools

import naive
import numpy as np
import numpy.testing as npt
import pytest

from stumpy import config, rng
from stumpy.snippets import _get_all_profiles, snippets

test_data = [rng.RNG.uniform(-1000, 1000, [64]).astype(np.float64)]
s = [3, 6, 7, 8]
percentage = [0.4, 0.7, 0.8, 0.9]
m = [8, 9, 10]
k = [1, 2, 3]


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
def test_get_all_mpdist_profiles(T, m):
    ref_profiles = naive.get_all_mpdist_profiles(T, m)
    cmp_profiles = _get_all_profiles(T, m)

    npt.assert_allclose(
        cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("percentage", percentage)
def test_get_all_mpdist_profiles_percentage(T, m, percentage):
    ref_profiles = naive.get_all_mpdist_profiles(T, m, percentage=percentage)
    cmp_profiles = _get_all_profiles(T, m, percentage=percentage)

    npt.assert_allclose(
        cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("s", s)
def test_get_all_mpdist_profiles_s(T, m, s):
    ref_profiles = naive.get_all_mpdist_profiles(T, m, s=s)
    cmp_profiles = _get_all_profiles(T, m, s=s)

    npt.assert_allclose(
        cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
def test_get_all_mpdist_profiles_with_isconstant(T, m):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )
    ref_profiles = naive.get_all_mpdist_profiles(
        T, m, mpdist_T_subseq_isconstant=isconstant_custom_func
    )
    cmp_profiles = _get_all_profiles(
        T, m, mpdist_T_subseq_isconstant=isconstant_custom_func
    )

    npt.assert_allclose(
        cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("k", k)
def test_mpdist_snippets(T, m, k):
    D = _get_all_profiles(
        T,
        m,
    )
    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.mpdist_snippets(
        T,
        m,
        k,
        D=D,
    )
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = snippets(T, m, k)

    npt.assert_allclose(
        cmp_snippets, ref_snippets, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_indices, ref_indices, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_fractions, ref_fractions, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_areas, ref_areas, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(cmp_regimes, ref_regimes, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("k", k)
@pytest.mark.parametrize("percentage", percentage)
def test_mpdist_snippets_percentage(T, m, k, percentage):
    D = _get_all_profiles(
        T,
        m,
        percentage=percentage,
    )
    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.mpdist_snippets(
        T,
        m,
        k,
        percentage=percentage,
        D=D,
    )
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = snippets(T, m, k, percentage=percentage)

    npt.assert_allclose(
        cmp_snippets, ref_snippets, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_indices, ref_indices, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_fractions, ref_fractions, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_areas, ref_areas, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(cmp_regimes, ref_regimes, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("k", k)
@pytest.mark.parametrize("s", s)
def test_mpdist_snippets_s(T, m, k, s):
    D = _get_all_profiles(
        T,
        m,
        s=s,
    )
    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.mpdist_snippets(
        T,
        m,
        k,
        s=s,
        D=D,
    )
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = snippets(T, m, k, s=s)

    npt.assert_allclose(
        cmp_snippets, ref_snippets, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_indices, ref_indices, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_fractions, ref_fractions, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_areas, ref_areas, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(cmp_regimes, ref_regimes, atol=1.5e-07)


@pytest.mark.parametrize("T", test_data)
@pytest.mark.parametrize("m", m)
@pytest.mark.parametrize("k", k)
@pytest.mark.parametrize("s", s)
def test_mpdist_snippets_s_with_isconstant(T, m, k, s):
    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )
    D = _get_all_profiles(
        T,
        m,
        s=s,
        mpdist_T_subseq_isconstant=isconstant_custom_func,
    )
    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.mpdist_snippets(
        T,
        m,
        k,
        s=s,
        mpdist_T_subseq_isconstant=isconstant_custom_func,
        D=D,
    )
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = snippets(T, m, k, s=s, mpdist_T_subseq_isconstant=isconstant_custom_func)

    npt.assert_allclose(
        cmp_snippets, ref_snippets, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_indices, ref_indices, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_profiles, ref_profiles, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_fractions, ref_fractions, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(
        cmp_areas, ref_areas, atol=1.5 * 10**-config.STUMPY_TEST_PRECISION
    )
    npt.assert_allclose(cmp_regimes, ref_regimes, atol=1.5e-07)


def test_mpdist_snippets_tied_areas():
    # In GH-984, windows starting at 40 and 56 tie for the third snippet.
    with rng.fix_seed(455):
        T = rng.RNG.uniform(-1000, 1000, 64)
    m = 8
    isconstant = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )
    ref_profiles = naive.get_all_mpdist_profiles(
        T, m, s=3, mpdist_T_subseq_isconstant=isconstant
    )
    cmp_snippets, cmp_indices, cmp_profiles, _, cmp_areas, _ = snippets(
        T, m, k=3, s=3, mpdist_T_subseq_isconstant=isconstant
    )

    npt.assert_array_equal(cmp_indices[:2], [16, 32])
    assert cmp_indices[2] in (40, 56)
    npt.assert_array_equal(cmp_snippets, [T[i : i + m] for i in cmp_indices])
    # Check each selected profile against the independent reference at its own index.
    selected = ref_profiles[cmp_indices // m]
    atol = 1.5 * 10**-config.STUMPY_TEST_PRECISION
    npt.assert_allclose(cmp_profiles, selected, atol=atol)
    npt.assert_allclose(
        cmp_areas, np.minimum.accumulate(selected).sum(axis=1), atol=atol
    )

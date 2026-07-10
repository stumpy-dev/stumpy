import functools
from unittest.mock import patch

import naive
import numba
import numpy as np
import numpy.testing as npt
import pytest
from numba import cuda

from stumpy import cache, config, core, fastmath, rng, sdp

if cuda.is_available():
    from stumpy.gpu_stump import gpu_stump
else:  # pragma: no cover
    from stumpy.core import _gpu_stump_driver_not_found as gpu_stump  # noqa: F401
from stumpy.snippets import snippets

try:
    from numba.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
    from numba.core.errors import NumbaPerformanceWarning


test_data = [
    # seed = 332
    # np.random.seed(seed)
    # T = np.random.uniform(-1000.0, 1000.0, [64])
    np.array(
        [
            836.349989350273290256,
            526.800862055943639461,
            -429.339375142462358781,
            860.078846179842571473,
            -596.812914971213785975,
            593.127160844603395162,
            633.075798396083314401,
            -750.776773161861342487,
            -665.810204117144508018,
            -174.607860313390034435,
            -710.208321817786327301,
            753.532886560284850930,
            -838.186757410647146571,
            173.299013001693140268,
            703.222990535601979900,
            -807.064168431504754153,
            962.883899197580944929,
            38.810424278810096155,
            -83.862617990669846790,
            18.550959664470177302,
            -662.512127934688805908,
            -747.060775456810347350,
            760.462217296095332131,
            -130.112948062053078502,
            683.272031839069086345,
            -908.349670788853018166,
            557.546937977237575979,
            640.792333789786312082,
            391.562294309980586604,
            -911.469920938469613247,
            743.180478392348163652,
            -887.714941208484106028,
            -286.934569476950514400,
            -562.818879750450264510,
            306.910774071569505850,
            -830.856058225298170328,
            208.087981330164382143,
            -651.135621003526352979,
            235.249465940220005677,
            600.462506051327750356,
            -843.324044229922378690,
            -722.536288038352608964,
            876.710973289016123999,
            -336.923085822701693814,
            521.587179728839601012,
            872.169364663994997500,
            -56.626714461904150255,
            719.145878188693018274,
            -851.663239330660871929,
            680.558711336155511162,
            61.668445663497273301,
            600.716760574291583907,
            93.433236590898076201,
            380.439820449711930905,
            -341.580795812605344963,
            455.065177017377891389,
            -92.403701334716984661,
            -321.681699561015022937,
            731.415976214936563338,
            -694.499688446164896050,
            21.217923163572738332,
            243.395272972420485758,
            889.609495142733749162,
            -743.441543901967293095,
        ]
    )
]

TEST_THREADS_PER_BLOCK = 10


def test_mpdist_snippets_s():
    # This test function raises an error if the distance between
    # a subsequence (of length `s`) and itelf becomes non-zero
    # in the performant version. Fixing this loss-of-precision can
    # result in this test being passed.
    # seed = 0
    # np.random.seed(seed)
    # T = np.random.uniform(-1000, 1000, [64]).astype(np.float64)
    T = np.array(
        [
            97.627007854649505703,
            430.378732744838941926,
            205.526752143287751551,
            89.766365993793726830,
            -152.690401322190581368,
            291.788226133312207367,
            -124.825577474614973994,
            783.546001564159496411,
            927.325521002058508202,
            -233.116962348444587860,
            583.450076165329164724,
            57.789839505808956233,
            136.089122187864631996,
            851.193276585322109895,
            -857.927883604226167336,
            -825.741400596918538213,
            -959.563205119348594963,
            665.239691095875969040,
            556.313501899701009279,
            740.024296493638303218,
            957.236684465528014698,
            598.317128433447237512,
            -77.041275494136300495,
            561.058352572910962408,
            -763.451148262133528988,
            279.842042655047634980,
            -713.293425181907196020,
            889.337834099167821478,
            43.696643500143352412,
            -170.676120018952815371,
            -470.888775790746080929,
            548.467378868433343087,
            -87.699335566902902883,
            136.867897737297028016,
            -962.420399127289670105,
            235.270994151754109680,
            224.191445444842827328,
            233.867993749513829016,
            887.496157029248365689,
            363.640598206966842554,
            -280.984198852427994098,
            -125.936092401317097256,
            395.262391854529710145,
            -879.549056741460390185,
            333.533430891335342494,
            341.275739236318827352,
            -579.234877852318163605,
            -742.147404690293342355,
            -369.143298151632279769,
            -272.578458114754766939,
            140.393540835759267793,
            -122.796973075359304062,
            976.747676118452318406,
            -795.910378503943888973,
            -582.246487810330563661,
            -677.380964230007521110,
            306.216650930796845387,
            -493.416794920435791028,
            -67.378454287387427257,
            -511.148815996794496641,
            -682.060832708960560922,
            -779.249717671389703355,
            312.659178930546829633,
            -723.634097302772374860,
        ]
    )
    m = 10
    k = 3
    s = 3

    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.mpdist_snippets(T, m, k, s=s)
    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = snippets(T, m, k, s=s)

    npt.assert_almost_equal(
        ref_fractions, cmp_fractions, decimal=config.STUMPY_TEST_PRECISION
    )


def test_distace_profile():
    # This test function raises an error when the distance profile between
    # the query `Q = T[i: i+m]`  and `T` becomes non-zero at index `i`.
    T = rng.RNG.random(64)
    m = 3
    T, M_T, Σ_T, T_subseq_isconstant = core.preprocess(T, m)

    for i in range(len(T) - m + 1):
        Q = T[i : i + m]
        D_ref = naive.distance_profile(Q, T, m)
        D_comp = core.mass(
            Q, T, M_T=M_T, Σ_T=Σ_T, T_subseq_isconstant=T_subseq_isconstant, query_idx=i
        )

        npt.assert_almost_equal(D_ref, D_comp)


@pytest.mark.parametrize("T", test_data)
def test_calculate_squared_distance(T):
    # This test function raises an error if the distance between a subsequence
    # and another does not satisfy the symmetry property.
    m = 3

    T_subseq_isconstant = core.rolling_isconstant(T, m)
    M_T, Σ_T = core.compute_mean_std(T, m)

    n = len(T)
    k = n - m + 1
    for i in range(k):
        for j in range(k):
            QT_i = sdp._njit_sliding_dot_product(T[i : i + m], T)
            dist_ij = core._calculate_squared_distance(
                m,
                QT_i[j],
                M_T[i],
                Σ_T[i],
                M_T[j],
                Σ_T[j],
                T_subseq_isconstant[i],
                T_subseq_isconstant[j],
            )

            QT_j = sdp._njit_sliding_dot_product(T[j : j + m], T)
            dist_ji = core._calculate_squared_distance(
                m,
                QT_j[i],
                M_T[j],
                Σ_T[j],
                M_T[i],
                Σ_T[i],
                T_subseq_isconstant[j],
                T_subseq_isconstant[i],
            )

            comp = dist_ij - dist_ji
            ref = 0.0

            npt.assert_almost_equal(ref, comp, decimal=14)


@pytest.mark.parametrize("T", test_data)
def test_snippets(T):
    # This test function raises an error if there is a considerable loss of precision
    # that violates the symmetry property of a distance measure.
    m = 10
    k = 3
    s = 3

    isconstant_custom_func = functools.partial(
        naive.isconstant_func_stddev_threshold, quantile_threshold=0.05
    )
    (
        ref_snippets,
        ref_indices,
        ref_profiles,
        ref_fractions,
        ref_areas,
        ref_regimes,
    ) = naive.mpdist_snippets(
        T, m, k, s=s, mpdist_T_subseq_isconstant=isconstant_custom_func
    )

    (
        cmp_snippets,
        cmp_indices,
        cmp_profiles,
        cmp_fractions,
        cmp_areas,
        cmp_regimes,
    ) = snippets(T, m, k, s=s, mpdist_T_subseq_isconstant=isconstant_custom_func)

    if (
        not np.allclose(ref_snippets, cmp_snippets) and not numba.config.DISABLE_JIT
    ):  # pragma: no cover
        # Revise fastmath flags by removing reassoc (to improve precision),
        # recompile njit functions, and re-compute snippets.
        fastmath._set(
            "core", "_calculate_squared_distance", {"nsz", "arcp", "contract", "afn"}
        )
        cache._recompile()

        (
            cmp_snippets,
            cmp_indices,
            cmp_profiles,
            cmp_fractions,
            cmp_areas,
            cmp_regimes,
        ) = snippets(T, m, k, s=s, mpdist_T_subseq_isconstant=isconstant_custom_func)

    npt.assert_almost_equal(
        ref_snippets, cmp_snippets, decimal=config.STUMPY_TEST_PRECISION
    )
    npt.assert_almost_equal(
        ref_indices, cmp_indices, decimal=config.STUMPY_TEST_PRECISION
    )
    npt.assert_almost_equal(
        ref_profiles, cmp_profiles, decimal=config.STUMPY_TEST_PRECISION
    )
    npt.assert_almost_equal(
        ref_fractions, cmp_fractions, decimal=config.STUMPY_TEST_PRECISION
    )
    npt.assert_almost_equal(ref_areas, cmp_areas, decimal=config.STUMPY_TEST_PRECISION)
    npt.assert_almost_equal(ref_regimes, cmp_regimes)

    if not numba.config.DISABLE_JIT:  # pragma: no cover
        # Revert fastmath flag back to their default values
        fastmath._reset("core", "_calculate_squared_distance")
        cache._recompile()


@pytest.mark.filterwarnings("ignore", category=NumbaPerformanceWarning)
@patch("stumpy.config.STUMPY_THREADS_PER_BLOCK", TEST_THREADS_PER_BLOCK)
@pytest.mark.parametrize("T", test_data)
def test_distance_symmetry_property_in_gpu(T):
    if not cuda.is_available():  # pragma: no cover
        pytest.skip("Skipping Tests No GPUs Available")

    # This test function raises an error if the distance between a subsequence
    # and another one does not satisfy the symmetry property.
    m = 3

    i, j = 2, 10
    # M_T, Σ_T = core.compute_mean_std(T, m)
    # Σ_T[i] is `650.912209452633`
    # Σ_T[j] is `722.0717285148525`

    # This test raises an error if arithmetic operation in ...
    # ... `gpu_stump._compute_and_update_PI_kernel` does not
    # generates the same result if values of variable for mean and std
    # are swapped.

    T_A = T[i : i + m]
    T_B = T[j : j + m]

    mp_AB = gpu_stump(T_A, m, T_B)
    mp_BA = gpu_stump(T_B, m, T_A)

    d_ij = mp_AB[0, 0]
    d_ji = mp_BA[0, 0]

    comp = d_ij - d_ji
    ref = 0.0

    npt.assert_almost_equal(comp, ref, decimal=15)

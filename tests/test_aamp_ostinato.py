import mersenne
import naive
import numpy as np
import numpy.testing as npt
import pytest
import tornado.ioloop
from dask.distributed import Client, LocalCluster

from stumpy import core, rng
from stumpy.aamp_ostinato import aamp_ostinato, aamp_ostinatoed


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        dashboard_address=None,
        worker_dashboard_address=None,
    )
    yield cluster.scheduler_address
    try:
        cluster.close(timeout=60)
    except tornado.ioloop.TimeoutError:  # pragma: no cover
        pass


@pytest.mark.parametrize("runs", range(25))
def test_random_ostinato(runs):
    m = 50
    Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]

    ref_radius, ref_Ts_idx, ref_subseq_idx = naive.aamp_ostinato(Ts, m)
    cmp_radius, cmp_Ts_idx, cmp_subseq_idx = aamp_ostinato(Ts, m)

    npt.assert_allclose(cmp_radius, ref_radius, atol=1.5e-07)
    npt.assert_allclose(cmp_Ts_idx, ref_Ts_idx, atol=1.5e-07)
    npt.assert_allclose(cmp_subseq_idx, ref_subseq_idx, atol=1.5e-07)


@pytest.mark.parametrize("seed", [41, 88, 290, 292, 310, 328, 538, 556, 563, 570])
def test_deterministic_ostinato(seed):
    m = 50
    state = mersenne.seed_to_state(seed)
    with rng.fix_state(state):
        Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]

        for p in [1.0, 2.0, 3.0]:
            ref_radius, ref_Ts_idx, ref_subseq_idx = naive.aamp_ostinato(Ts, m, p=p)
            cmp_radius, cmp_Ts_idx, cmp_subseq_idx = aamp_ostinato(Ts, m, p=p)

            npt.assert_allclose(cmp_radius, ref_radius, atol=1.5e-07)
            npt.assert_allclose(cmp_Ts_idx, ref_Ts_idx, atol=1.5e-07)
            npt.assert_allclose(cmp_subseq_idx, ref_subseq_idx, atol=1.5e-07)


@pytest.mark.parametrize("runs", range(25))
def test_random_ostinatoed(runs, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 50
        Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]

        ref_radius, ref_Ts_idx, ref_subseq_idx = naive.aamp_ostinato(Ts, m)
        cmp_radius, cmp_Ts_idx, cmp_subseq_idx = aamp_ostinatoed(dask_client, Ts, m)

        npt.assert_allclose(cmp_radius, ref_radius, atol=1.5e-07)
        npt.assert_allclose(cmp_Ts_idx, ref_Ts_idx, atol=1.5e-07)
        npt.assert_allclose(cmp_subseq_idx, ref_subseq_idx, atol=1.5e-07)


@pytest.mark.parametrize("seed", [41, 88, 290, 292, 310, 328, 538, 556, 563, 570])
def test_deterministic_ostinatoed(seed, dask_cluster):
    with Client(dask_cluster) as dask_client:
        m = 50
        state = mersenne.seed_to_state(seed)
        with rng.fix_state(state):
            Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]

            for p in [1.0, 2.0, 3.0]:
                ref_radius, ref_Ts_idx, ref_subseq_idx = naive.aamp_ostinato(Ts, m, p=p)
                cmp_radius, cmp_Ts_idx, cmp_subseq_idx = aamp_ostinatoed(
                    dask_client, Ts, m, p=p
                )

                npt.assert_allclose(cmp_radius, ref_radius, atol=1.5e-07)
                npt.assert_allclose(cmp_Ts_idx, ref_Ts_idx, atol=1.5e-07)
                npt.assert_allclose(cmp_subseq_idx, ref_subseq_idx, atol=1.5e-07)


def test_input_not_overwritten_ostinato():
    # aamp_ostinato preprocesses its input, a list of time series,
    # by replacing nan value with 0 in each time series.
    # This test ensures that the original input is not overwritten
    m = 50
    Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]
    for T in Ts:
        T[0] = np.nan

    # raise error if aamp_ostinato overwrite its input
    Ts_input = [T.copy() for T in Ts]
    aamp_ostinato(Ts_input, m)
    for i in range(len(Ts)):
        T_ref = Ts[i]
        T_cmp = Ts_input[i]
        npt.assert_allclose(
            T_cmp[np.isfinite(T_cmp)], T_ref[np.isfinite(T_ref)], atol=1.5e-07
        )


def test_extract_several_consensus_ostinato():
    # This test is to further ensure that the function `aamp_ostinato`
    # does not tamper with the original data.
    Ts = [rng.RNG.rand(n) for n in [256, 512, 1024]]
    Ts_ref = [T.copy() for T in Ts]
    Ts_cmp = [T.copy() for T in Ts]

    m = 20

    k = 5  # Get the first `k` consensus motifs
    for _ in range(k):
        # Find consensus motif and its NN in each time series in Ts_cmp
        # Remove them from Ts_cmp as well as Ts_ref, and assert that the
        # two time series are the same
        radius, Ts_idx, subseq_idx = aamp_ostinato(Ts_cmp, m)
        consensus_motif = Ts_cmp[Ts_idx][subseq_idx : subseq_idx + m].copy()
        for i in range(len(Ts_cmp)):
            if i == Ts_idx:
                query_idx = subseq_idx
            else:
                query_idx = None

            idx = np.argmin(
                core.mass(
                    consensus_motif, Ts_cmp[i], normalize=False, query_idx=query_idx
                )
            )
            Ts_cmp[i][idx : idx + m] = np.nan
            Ts_ref[i][idx : idx + m] = np.nan

            npt.assert_allclose(
                Ts_cmp[i][np.isfinite(Ts_cmp[i])],
                Ts_ref[i][np.isfinite(Ts_ref[i])],
                atol=1.5e-07,
            )


def test_input_not_overwritten_ostinatoed(dask_cluster):
    # ostinatoed preprocesses its input, a list of time series,
    # by replacing nan value with 0 in each time series.
    # This test ensures that the original input is not overwritten
    with Client(dask_cluster) as dask_client:
        m = 50
        Ts = [rng.RNG.rand(n) for n in [64, 128, 256]]
        for T in Ts:
            T[0] = np.nan

        # raise error if ostinato overwrite its input
        Ts_input = [T.copy() for T in Ts]
        aamp_ostinatoed(dask_client, Ts_input, m)
        for i in range(len(Ts)):
            T_ref = Ts[i]
            T_cmp = Ts_input[i]
            npt.assert_allclose(
                T_cmp[np.isfinite(T_cmp)], T_ref[np.isfinite(T_ref)], atol=1.5e-07
            )


def test_extract_several_consensus_ostinatoed(dask_cluster):
    # This test is to further ensure that the function `ostinatoed`
    # does not tamper with the original data.
    Ts = [rng.RNG.rand(n) for n in [256, 512, 1024]]
    Ts_ref = [T.copy() for T in Ts]
    Ts_cmp = [T.copy() for T in Ts]

    m = 20

    with Client(dask_cluster) as dask_client:
        k = 5  # Get the first `k` consensus motifs
        for _ in range(k):
            # Find consensus motif and its NN in each time series in Ts_cmp
            # Remove them from Ts_cmp as well as Ts_ref, and assert that the
            # two time series are the same
            radius, Ts_idx, subseq_idx = aamp_ostinatoed(dask_client, Ts_cmp, m)
            consensus_motif = Ts_cmp[Ts_idx][subseq_idx : subseq_idx + m].copy()
            for i in range(len(Ts_cmp)):
                if i == Ts_idx:
                    query_idx = subseq_idx
                else:
                    query_idx = None

                idx = np.argmin(
                    core.mass(
                        consensus_motif,
                        Ts_cmp[i],
                        normalize=False,
                        query_idx=query_idx,
                    )
                )
                Ts_cmp[i][idx : idx + m] = np.nan
                Ts_ref[i][idx : idx + m] = np.nan

                npt.assert_allclose(
                    Ts_cmp[i][np.isfinite(Ts_cmp[i])],
                    Ts_ref[i][np.isfinite(Ts_ref[i])],
                    atol=1.5e-07,
                )

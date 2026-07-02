# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.
import math
import multiprocessing as mp
import os

import numpy as np
from numba import cuda

from . import config, core
from .gpu_aamp import gpu_aamp
from .mparray import mparray


@cuda.jit(
    "(i8, f8[:], f8[:], i8, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], b1[:], b1[:], i8, b1, i8, f8[:, :], f8[:], f8[:], i8[:, :], i8[:], i8[:], b1, i8[:], i8, i8)"
)
def _compute_and_update_PI_kernel(
    idx,
    T_A,
    T_B,
    m,
    cov_even,
    cov_odd,
    cov_first,
    cov_a,
    cov_b,
    cov_c,
    cov_d,
    μ_Q,
    σ_Q,
    M_T,
    Σ_T,
    Q_subseq_isconstant,
    T_subseq_isconstant,
    w,
    ignore_trivial,
    excl_zone,
    profile,
    profile_L,
    profile_R,
    indices,
    indices_L,
    indices_R,
    compute_cov,
    bfs,
    nlevel,
    k,
):
    """
    A Numba CUDA kernel to update the matrix profile and matrix profile indices
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    j = idx
    m_inverse = 1.0 / m
    constant = (m - 1) * m_inverse * m_inverse

    if j % 2 == 0:
        cov_out = cov_even
        cov_in = cov_odd
    else:
        cov_out = cov_odd
        cov_in = cov_even

    for i in range(start, w, stride):
        zone_start = max(0, i - excl_zone)
        zone_stop = min(w, i + excl_zone)

        if compute_cov:
            if i == 0:
                cov_out[0] = cov_first[j]
            else:
                cov_out[i] = cov_in[i - 1] + constant * (
                    cov_a[j] * cov_b[i] - cov_c[j] * cov_d[i]
                )

        if math.isinf(μ_Q[i]) or math.isinf(M_T[j]):
            p_norm = np.inf
        elif Q_subseq_isconstant[i] and T_subseq_isconstant[j]:
            p_norm = 0.0
        elif Q_subseq_isconstant[i] or T_subseq_isconstant[j]:
            p_norm = m
        else:
            pearson = cov_out[i] / max(σ_Q[i] * Σ_T[j], config.STUMPY_DENOM_THRESHOLD)
            pearson = min(pearson, 1.0)
            p_norm = 2 * m * (1.0 - pearson)

        if ignore_trivial:
            if j <= zone_stop and j >= zone_start:
                p_norm = np.inf
            if p_norm < profile_L[i] and j < i:
                profile_L[i] = p_norm
                indices_L[i] = j
            if p_norm < profile_R[i] and j > i:
                profile_R[i] = p_norm
                indices_R[i] = j

        if p_norm < profile[i, -1]:
            idx_pos = core._gpu_searchsorted_right(profile[i], p_norm, bfs, nlevel)
            for g in range(k - 1, idx_pos, -1):
                profile[i, g] = profile[i, g - 1]
                indices[i, g] = indices[i, g - 1]

            profile[i, idx_pos] = p_norm
            indices[i, idx_pos] = j


def _gpu_stump(
    T_A_fname,
    T_B_fname,
    m,
    range_stop,
    excl_zone,
    μ_Q_fname,
    σ_Q_fname,
    cov_fname,
    cov_first_fname,
    cov_a_fname,
    cov_b_fname,
    cov_c_fname,
    cov_d_fname,
    M_T_fname,
    Σ_T_fname,
    Q_subseq_isconstant_fname,
    T_subseq_isconstant_fname,
    w,
    ignore_trivial=True,
    range_start=1,
    device_id=0,
    k=1,
):
    """
    A Numba CUDA version of STOMP for parallel computation of the
    matrix profile, matrix profile indices, left matrix profile indices,
    and right matrix profile indices.
    """
    threads_per_block = config.STUMPY_THREADS_PER_BLOCK
    blocks_per_grid = math.ceil(w / threads_per_block)

    T_A = np.load(T_A_fname, allow_pickle=False)
    T_B = np.load(T_B_fname, allow_pickle=False)
    cov = np.load(cov_fname, allow_pickle=False)
    cov_first = np.load(cov_first_fname, allow_pickle=False)
    cov_a = np.load(cov_a_fname, allow_pickle=False)
    cov_b = np.load(cov_b_fname, allow_pickle=False)
    cov_c = np.load(cov_c_fname, allow_pickle=False)
    cov_d = np.load(cov_d_fname, allow_pickle=False)
    μ_Q = np.load(μ_Q_fname, allow_pickle=False)
    σ_Q = np.load(σ_Q_fname, allow_pickle=False)
    M_T = np.load(M_T_fname, allow_pickle=False)
    Σ_T = np.load(Σ_T_fname, allow_pickle=False)
    Q_subseq_isconstant = np.load(Q_subseq_isconstant_fname, allow_pickle=False)
    T_subseq_isconstant = np.load(T_subseq_isconstant_fname, allow_pickle=False)

    nlevel = np.floor(np.log2(k) + 1).astype(np.int64)

    with cuda.gpus[device_id]:
        device_T_A = cuda.to_device(T_A)
        device_cov_odd = cuda.to_device(cov)
        device_cov_even = cuda.to_device(cov)
        device_cov_first = cuda.to_device(cov_first)
        device_cov_a = cuda.to_device(cov_a)
        device_cov_b = cuda.to_device(cov_b)
        device_cov_c = cuda.to_device(cov_c)
        device_cov_d = cuda.to_device(cov_d)
        device_μ_Q = cuda.to_device(μ_Q)
        device_σ_Q = cuda.to_device(σ_Q)
        device_Q_subseq_isconstant = cuda.to_device(Q_subseq_isconstant)

        if ignore_trivial:
            device_T_B = device_T_A
            device_M_T = device_μ_Q
            device_Σ_T = device_σ_Q
            device_T_subseq_isconstant = device_Q_subseq_isconstant
        else:
            device_T_B = cuda.to_device(T_B)
            device_M_T = cuda.to_device(M_T)
            device_Σ_T = cuda.to_device(Σ_T)
            device_T_subseq_isconstant = cuda.to_device(T_subseq_isconstant)

        profile = np.full((w, k), np.inf, dtype=np.float64)
        indices = np.full((w, k), -1, dtype=np.int64)

        profile_L = np.full(w, np.inf, dtype=np.float64)
        indices_L = np.full(w, -1, dtype=np.int64)

        profile_R = np.full(w, np.inf, dtype=np.float64)
        indices_R = np.full(w, -1, dtype=np.int64)

        device_profile = cuda.to_device(profile)
        device_profile_L = cuda.to_device(profile_L)
        device_profile_R = cuda.to_device(profile_R)
        device_indices = cuda.to_device(indices)
        device_indices_L = cuda.to_device(indices_L)
        device_indices_R = cuda.to_device(indices_R)
        device_bfs = cuda.to_device(core._bfs_indices(k, fill_value=-1))

        _compute_and_update_PI_kernel[blocks_per_grid, threads_per_block](
            range_start - 1,
            device_T_A,
            device_T_B,
            m,
            device_cov_even,
            device_cov_odd,
            device_cov_first,
            device_cov_a,
            device_cov_b,
            device_cov_c,
            device_cov_d,
            device_μ_Q,
            device_σ_Q,
            device_M_T,
            device_Σ_T,
            device_Q_subseq_isconstant,
            device_T_subseq_isconstant,
            w,
            ignore_trivial,
            excl_zone,
            device_profile,
            device_profile_L,
            device_profile_R,
            device_indices,
            device_indices_L,
            device_indices_R,
            False,
            device_bfs,
            nlevel,
            k,
        )

        for i in range(range_start, range_stop):
            _compute_and_update_PI_kernel[blocks_per_grid, threads_per_block](
                i,
                device_T_A,
                device_T_B,
                m,
                device_cov_even,
                device_cov_odd,
                device_cov_first,
                device_cov_a,
                device_cov_b,
                device_cov_c,
                device_cov_d,
                device_μ_Q,
                device_σ_Q,
                device_M_T,
                device_Σ_T,
                device_Q_subseq_isconstant,
                device_T_subseq_isconstant,
                w,
                ignore_trivial,
                excl_zone,
                device_profile,
                device_profile_L,
                device_profile_R,
                device_indices,
                device_indices_L,
                device_indices_R,
                True,
                device_bfs,
                nlevel,
                k,
            )

        profile = device_profile.copy_to_host()
        profile_L = device_profile_L.copy_to_host()
        profile_R = device_profile_R.copy_to_host()
        indices = device_indices.copy_to_host()
        indices_L = device_indices_L.copy_to_host()
        indices_R = device_indices_R.copy_to_host()

        profile[:, :] = np.sqrt(profile)
        profile_L[:] = np.sqrt(profile_L)
        profile_R[:] = np.sqrt(profile_R)

        profile_fname = core.array_to_temp_file(profile)
        profile_L_fname = core.array_to_temp_file(profile_L)
        profile_R_fname = core.array_to_temp_file(profile_R)
        indices_fname = core.array_to_temp_file(indices)
        indices_L_fname = core.array_to_temp_file(indices_L)
        indices_R_fname = core.array_to_temp_file(indices_R)

    return (
        profile_fname,
        profile_L_fname,
        profile_R_fname,
        indices_fname,
        indices_L_fname,
        indices_R_fname,
    )


@core.non_normalized(gpu_aamp)
def gpu_stump(
    T_A,
    m,
    T_B=None,
    ignore_trivial=True,
    device_id=0,
    normalize=True,
    p=2.0,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    """
    Compute the z-normalized matrix profile with one or more GPU devices
    """
    if T_B is None:  # Self join!
        T_B = T_A
        core.check_self_join(ignore_trivial)
        ignore_trivial = True
        T_B_subseq_isconstant = T_A_subseq_isconstant

    T_A, μ_Q, σ_Q, Q_subseq_isconstant = core.preprocess(
        T_A, m, T_subseq_isconstant=T_A_subseq_isconstant
    )
    T_B, M_T, Σ_T, T_subseq_isconstant = core.preprocess(
        T_B, m, T_subseq_isconstant=T_B_subseq_isconstant
    )

    if T_A.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_A is {T_A.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )

    if T_B.ndim != 1:  # pragma: no cover
        raise ValueError(
            f"T_B is {T_B.ndim}-dimensional and must be 1-dimensional. "
            "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
        )

    ignore_trivial = core.check_ignore_trivial(T_A, T_B, ignore_trivial)
    if ignore_trivial:  # self-join
        core.check_window_size(
            m, max_size=min(T_A.shape[0], T_B.shape[0]), n=T_A.shape[0]
        )
    else:  # AB-join
        core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))

    n = T_B.shape[0]
    w = T_A.shape[0] - m + 1
    l = n - m + 1
    excl_zone = int(
        np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)
    )  # See Definition 3 and Figure 3

    # Precalculate for sliding covariance
    M_T_clean, _ = core.compute_mean_std(T_B, m)
    μ_Q_clean, _ = core.compute_mean_std(T_A, m)

    M_T_m_1, Σ_T_m_1 = core.compute_mean_std(T_B, m - 1)
    μ_Q_m_1, σ_Q_m_1 = core.compute_mean_std(T_A, m - 1)

    cov_a = T_B[m - 1 :] - M_T_m_1[:-1]
    cov_b = T_A[m - 1 :] - μ_Q_m_1[:-1]

    cov_c = np.empty(M_T_m_1.shape[0], dtype=np.float64)
    cov_c[1:] = T_B[: M_T_m_1.shape[0] - 1]
    cov_c[0] = T_B[-1]
    cov_c[:] = cov_c - M_T_m_1

    cov_d = np.empty(μ_Q_m_1.shape[0], dtype=np.float64)
    cov_d[1:] = T_A[: μ_Q_m_1.shape[0] - 1]
    cov_d[0] = T_A[-1]
    cov_d[:] = cov_d - μ_Q_m_1

    T_A_fname = core.array_to_temp_file(T_A)
    T_B_fname = core.array_to_temp_file(T_B)
    cov_a_fname = core.array_to_temp_file(cov_a)
    cov_b_fname = core.array_to_temp_file(cov_b)
    cov_c_fname = core.array_to_temp_file(cov_c)
    cov_d_fname = core.array_to_temp_file(cov_d)
    μ_Q_fname = core.array_to_temp_file(μ_Q)
    σ_Q_fname = core.array_to_temp_file(σ_Q)
    M_T_fname = core.array_to_temp_file(M_T)
    Σ_T_fname = core.array_to_temp_file(Σ_T)
    Q_subseq_isconstant_fname = core.array_to_temp_file(Q_subseq_isconstant)
    T_subseq_isconstant_fname = core.array_to_temp_file(T_subseq_isconstant)

    if isinstance(device_id, int):
        device_ids = [device_id]
    else:
        device_ids = device_id

    profile = [None] * len(device_ids)
    indices = [None] * len(device_ids)

    profile_L = [None] * len(device_ids)
    indices_L = [None] * len(device_ids)

    profile_R = [None] * len(device_ids)
    indices_R = [None] * len(device_ids)

    for _id in device_ids:
        with cuda.gpus[_id]:
            if (
                cuda.current_context().__class__.__name__ != "FakeCUDAContext"
            ):  # pragma: no cover
                cuda.current_context().deallocations.clear()

    step = 1 + l // len(device_ids)

    # Start process pool for multi-GPU request
    if len(device_ids) > 1:  # pragma: no cover
        mp.set_start_method("spawn", force=True)
        pool = mp.Pool(processes=len(device_ids))
        results = [None] * len(device_ids)

    cov_fnames = []
    cov_first_fnames = []

    for idx, start in enumerate(range(0, l, step)):
        stop = min(l, start + step)

        QT, QT_first = core._get_QT(start, T_A, T_B, m)
        cov = (QT / m) - (μ_Q_clean * M_T_clean[start])
        cov_first = (QT_first / m) - (μ_Q_clean[0] * M_T_clean)
        
        cov_fname = core.array_to_temp_file(cov)
        cov_first_fname = core.array_to_temp_file(cov_first)
        cov_fnames.append(cov_fname)
        cov_first_fnames.append(cov_first_fname)

        if len(device_ids) > 1 and idx < len(device_ids) - 1:  # pragma: no cover
            # Spawn and execute in child process for multi-GPU request
            results[idx] = pool.apply_async(
                _gpu_stump,
                (
                    T_A_fname,
                    T_B_fname,
                    m,
                    stop,
                    excl_zone,
                    μ_Q_fname,
                    σ_Q_fname,
                    cov_fname,
                    cov_first_fname,
                    cov_a_fname,
                    cov_b_fname,
                    cov_c_fname,
                    cov_d_fname,
                    M_T_fname,
                    Σ_T_fname,
                    Q_subseq_isconstant_fname,
                    T_subseq_isconstant_fname,
                    w,
                    ignore_trivial,
                    start + 1,
                    device_ids[idx],
                    k,
                ),
            )
        else:
            # Execute last chunk in parent process
            # Only parent process is executed when a single GPU is requested
            (
                profile[idx],
                profile_L[idx],
                profile_R[idx],
                indices[idx],
                indices_L[idx],
                indices_R[idx],
            ) = _gpu_stump(
                T_A_fname,
                T_B_fname,
                m,
                stop,
                excl_zone,
                μ_Q_fname,
                σ_Q_fname,
                cov_fname,
                cov_first_fname,
                cov_a_fname,
                cov_b_fname,
                cov_c_fname,
                cov_d_fname,
                M_T_fname,
                Σ_T_fname,
                Q_subseq_isconstant_fname,
                T_subseq_isconstant_fname,
                w,
                ignore_trivial,
                start + 1,
                device_ids[idx],
                k,
            )

    # Clean up process pool for multi-GPU request
    if len(device_ids) > 1:  # pragma: no cover
        pool.close()
        pool.join()

        # Collect results from spawned child processes if they exist
        for idx, result in enumerate(results):
            if result is not None:
                (
                    profile[idx],
                    profile_L[idx],
                    profile_R[idx],
                    indices[idx],
                    indices_L[idx],
                    indices_R[idx],
                ) = result.get()

    os.remove(T_A_fname)
    os.remove(T_B_fname)
    os.remove(cov_a_fname)
    os.remove(cov_b_fname)
    os.remove(cov_c_fname)
    os.remove(cov_d_fname)
    os.remove(μ_Q_fname)
    os.remove(σ_Q_fname)
    os.remove(M_T_fname)
    os.remove(Σ_T_fname)
    os.remove(Q_subseq_isconstant_fname)
    os.remove(T_subseq_isconstant_fname)
    for cov_fname in cov_fnames:
        os.remove(cov_fname)
    for cov_first_fname in cov_first_fnames:
        os.remove(cov_first_fname)

    for idx in range(len(device_ids)):
        profile_fname = profile[idx]
        profile_L_fname = profile_L[idx]
        profile_R_fname = profile_R[idx]
        indices_fname = indices[idx]
        indices_L_fname = indices_L[idx]
        indices_R_fname = indices_R[idx]

        profile[idx] = np.load(profile_fname, allow_pickle=False)
        profile_L[idx] = np.load(profile_L_fname, allow_pickle=False)
        profile_R[idx] = np.load(profile_R_fname, allow_pickle=False)
        indices[idx] = np.load(indices_fname, allow_pickle=False)
        indices_L[idx] = np.load(indices_L_fname, allow_pickle=False)
        indices_R[idx] = np.load(indices_R_fname, allow_pickle=False)

        os.remove(profile_fname)
        os.remove(profile_L_fname)
        os.remove(profile_R_fname)
        os.remove(indices_fname)
        os.remove(indices_L_fname)
        os.remove(indices_R_fname)

    for i in range(1, len(device_ids)):  # pragma: no cover
        # Update (top-k) matrix profile and matrix profile indices
        core._merge_topk_PI(profile[0], profile[i], indices[0], indices[i])

        # Update (top-1) left matrix profile and matrix profile indices
        mask = profile_L[0] > profile_L[i]
        profile_L[0][mask] = profile_L[i][mask]
        indices_L[0][mask] = indices_L[i][mask]

        # Update (top-1) right matrix profile and matrix profile indices
        mask = profile_R[0] > profile_R[i]
        profile_R[0][mask] = profile_R[i][mask]
        indices_R[0][mask] = indices_R[i][mask]

    out = np.empty((w, 2 * k + 2), dtype=object)  # last two columns are to store
    # (top-1) left/right matrix profile indices
    out[:, :k] = profile[0]
    out[:, k:] = np.column_stack((indices[0], indices_L[0], indices_R[0]))

    core._check_P(out[:, 0])

    return mparray(out, m, k, config.STUMPY_EXCL_ZONE_DENOM)

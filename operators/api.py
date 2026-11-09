# anislap/api.py

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


__all__ = [
    "compute_aniso_features",
    "compute_iso_features",
    "compute_hks",
]


def compute_aniso_features(
    V: np.ndarray,
    F: np.ndarray,
    *,
    tensor_inputs: dict,
    num_eigs: int = 128,
    num_hks_times: int = 16,
    hks_times: Optional[np.ndarray] = None,
    hks_time_mode: str = "repo",
    compute_descriptors: bool = True,
) -> Dict[str, np.ndarray]:

    from .assembly import anisotropic_laplacian, vertex_field_to_face_tensor
    from .solvers import compute_eigenpairs
    from .descriptors import compute_hks as _compute_hks

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)

    if tensor_inputs is None:
        raise ValueError(
            "compute_aniso_features now requires tensor_inputs={'T_v': ..., 'D_v': ...}. "
            "Construct boundary/curvature/transferred tensors outside the API."
        )

    if not all(k in tensor_inputs for k in ("T_v", "D_v")):
        raise ValueError("tensor_inputs must contain {'T_v', 'D_v'}.")

    T_v = np.asarray(tensor_inputs["T_v"], dtype=np.float64)
    D_v = np.asarray(tensor_inputs["D_v"], dtype=np.float64)

    Umin, Umax, D = vertex_field_to_face_tensor(V, F, T_v, D_v)

    W, M = anisotropic_laplacian(V, F, Umin, Umax, D)
    evals, evecs = compute_eigenpairs(W, M, k=num_eigs)

    result = {
        "evals": evals.astype(np.float32),
        "evecs": evecs.astype(np.float32),
        "mass": M.diagonal().astype(np.float32),
    }

    if compute_descriptors:
        hks, hks_times = _compute_hks(
            evals,
            evecs,
            num_times=num_hks_times,
            times=hks_times,
            time_mode=hks_time_mode,
        )
        result["hks"] = hks
        result["hks_times"] = hks_times

    return result


def compute_iso_features(
    V: np.ndarray,
    F: np.ndarray,
    *,
    num_eigs: int = 128,
    num_hks_times: int = 16,
    hks_times: Optional[np.ndarray] = None,
    hks_time_mode: str = "repo",
    compute_descriptors: bool = True,
) -> Dict[str, np.ndarray]:

    from .assembly import cotangent_laplacian
    from .solvers import compute_eigenpairs
    from .descriptors import compute_hks as _compute_hks

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)

    W, M = cotangent_laplacian(V, F)
    evals, evecs = compute_eigenpairs(W, M, k=num_eigs)

    result = {
        "evals": evals.astype(np.float32),
        "evecs": evecs.astype(np.float32),
        "mass": M.diagonal().astype(np.float32),
    }

    if compute_descriptors:
        hks, hks_times = _compute_hks(
            evals,
            evecs,
            num_times=num_hks_times,
            times=hks_times,
            time_mode=hks_time_mode,
        )
        result["hks"] = hks
        result["hks_times"] = hks_times

    return result


def compute_hks(
    evals: np.ndarray,
    evecs: np.ndarray,
    num_times: int = 16,
    **kwargs,
):
    """Small compatibility wrapper for HKS."""
    from .descriptors import compute_hks as _compute_hks

    return _compute_hks(evals, evecs, num_times=num_times, **kwargs)
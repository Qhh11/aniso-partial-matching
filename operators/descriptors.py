# anislap/descriptors.py
"""
Spectral shape descriptors: Heat Kernel Signature (HKS) and Wave Kernel Signature (WKS).

This module provides functions to compute spectral descriptors from
both isotropic and anisotropic Laplacian eigenpairs.
"""

import numpy as np
from typing import Optional, Tuple, Union

def compute_hks(
    evals: np.ndarray,
    evecs: np.ndarray,
    num_times: int = 16,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    log_scale: bool = True,
    times: Optional[np.ndarray] = None,
    eps: float = 1e-8,
    time_mode: str = "legacy",   # "legacy", "repo", "standard" "auto"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HKS(x,t) = sum_k exp(-lambda_k * t) * phi_k(x)^2

    time_mode:
      - "legacy": current 4/lambda automatic range
      - "repo":   use logspace(-2, 0), following the referenced FM repo
      - "standard": use log(10)/lambda automatic range
    """
    evecs = np.asarray(evecs, dtype=np.float64)
    evals = np.asarray(evals, np.float64).reshape(-1)

    neg_ratio = np.mean(evals[1:] < 0) if evals.size > 1 else float(evals[0] < 0)
    lam = (-evals) if neg_ratio > 0.5 else evals

    lam = np.real(lam)
    lam[~np.isfinite(lam)] = 0.0
    lam[lam < 0.0] = 0.0

    if times is not None:
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if times.size == 0:
            raise ValueError("times is empty")

    else:
        valid_mask = lam > eps
        lam_pos = lam[valid_mask]

        if lam_pos.size == 0:
            times = np.ones(num_times, dtype=np.float64)
            return (
                np.zeros((evecs.shape[0], num_times), dtype=np.float32),
                times.astype(np.float32),
            )

        if time_mode == "repo":
            # Same idea as:
            # torch.logspace(-2.0, 0.0, steps=count)
            times = np.logspace(-2.0, 0.0, num_times)

        elif time_mode == "standard":
            lam_min = float(lam_pos.min())
            lam_max = float(lam_pos.max())

            if t_min is None:
                t_min = np.log(10.0) / lam_max
            if t_max is None:
                t_max = np.log(10.0) / max(lam_min, eps)

            if t_min <= 0 or t_max <= 0 or t_min >= t_max:
                t_min = np.log(10.0) / lam_max
                t_max = np.log(10.0) / max(lam_min, eps)

            if log_scale:
                times = np.logspace(np.log10(t_min), np.log10(t_max), num_times)
            else:
                times = np.linspace(t_min, t_max, num_times)

        elif time_mode == "legacy":
            if t_min is None:
                t_min = 4.0 / lam_pos.max()
            if t_max is None:
                t_max = 4.0 / max(lam_pos.min(), eps)

            if t_min >= t_max:
                t_max = t_min * 10.0

            if log_scale:
                times = np.logspace(np.log10(t_min), np.log10(t_max), num_times)
            else:
                times = np.linspace(t_min, t_max, num_times)

        elif time_mode == "auto":
            """Sun et al. 2009: log-spaced from 4ln10/lam_max to 4ln10/lam_min."""
            evals = np.asarray(evals, np.float64)
            abs_e = np.abs(evals)
            nz = abs_e[abs_e > eps]
            if nz.size < 2:
                return np.geomspace(1e-2, 1.0, num_times)
            lam_min = nz.min()
            lam_max = nz.max()
            t_min = 4.0 * np.log(10.0) / lam_max
            t_max = 4.0 * np.log(10.0) / lam_min
            times = np.geomspace(t_min, t_max, num_times)
        else:
            raise ValueError(f"Unknown HKS time_mode: {time_mode}")

    weights = np.exp(-lam[None, :] * times[:, None])      # (T,K)
    phi_sq = evecs ** 2                                   # (N,K)
    hks = phi_sq @ weights.T                              # (N,T)

    return hks.astype(np.float32), times.astype(np.float32)

def compute_hks_autoscale(evals: np.ndarray,evecs: np.ndarray,num_times: int = 16,) -> np.ndarray:
    hks, _ = compute_hks(evals, evecs, num_times=num_times)
    return hks


def compute_wks(
    evals,
    evecs,
    num_energies=100,
    sigma=None,
    eps=1e-8,
):
    """
    WKS(x, e) = sum_k exp(-(log lam_k - e)^2 / (2 sigma^2)) * phi_k(x)^2
    Aubry et al. 2011.
    """
    evals = np.asarray(evals, np.float64).reshape(-1)
    evecs = np.asarray(evecs, np.float64)

    # same sign-fix as in HKS
    neg_ratio = np.mean(evals[1:] < 0) if evals.size > 1 else float(evals[0] < 0)
    lam = (-evals) if neg_ratio > 0.5 else evals
    lam = np.real(lam)
    lam[~np.isfinite(lam)] = 0.0
    lam[lam < eps] = eps  # WKS needs strictly positive eigenvalues for log

    log_lam = np.log(lam)
    e_min = float(log_lam.min())
    e_max = float(log_lam.max())

    # standard recipe from Aubry et al.
    delta = (e_max - e_min) / num_energies
    if sigma is None:
        sigma = 7.0 * delta   # paper recommends 6-7 delta

    energies = np.linspace(e_min + 2 * sigma, e_max - 2 * sigma, num_energies)

    # weights: (num_energies, K)
    diff = log_lam[None, :] - energies[:, None]
    W = np.exp(- diff**2 / (2.0 * sigma**2))

    # No per-energy weight normalization here.
    # fmap_preprocess_descriptors(...) does mass-normalization on each
    # WKS column afterwards, which is what fmap pipelines actually need.

    phi_sq = evecs ** 2                 # (N, K)
    wks = phi_sq @ W.T                  # (N, num_energies)

    return wks.astype(np.float32), energies.astype(np.float32)


def compute_wks_autoscale( evals: np.ndarray,evecs: np.ndarray, num_scales: int = 100,) -> np.ndarray:
    """Compute WKS with automatic scale selection."""
    wks, _ = compute_wks(evals, evecs, num_scales=num_scales)
    return wks

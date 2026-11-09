# anislap/solvers.py
import numpy as np
import igl
from scipy.sparse.linalg import spsolve, eigsh, splu, LinearOperator
from scipy.sparse import diags, identity
from operators.geometry import *


def implicit_heat_step(L, M, u0, t):
    A = M - t * L
    b = M @ u0
    u1 = spsolve(A, b)
    return u1


def heat_geodesic(V, F, W_heat, M_heat, W_poisson, u0, t, precomputed=None, src_idx=None, eps=1e-12):
    A = M_heat - t * W_heat
    b = M_heat @ u0
    u = spsolve(A, b)

    X = compute_vertex_gradient(V, F, u, eps=eps)
    bad_grad = np.isnan(X).sum()
    if bad_grad > 0:
        print(f"[X] WARNING: {bad_grad} NaNs in gradient!!")

    if precomputed is None:
        A_v = vertex_areas_barycentric(V, F)
        edges, cot_sum, e_ij = build_edge_data_for_divergence(V, F, eps=eps)
    else:
        A_v, edges, cot_sum, e_ij = precomputed

    divX = divergence_cot(V, F, X, edges, cot_sum, e_ij, A_v, eps=eps)
    phi = solve_poisson_cot(W_poisson, divX, src_idx)

    phi -= phi[src_idx]
    if np.sum(phi < 0) > np.sum(phi > 0):
        phi = -phi

    return phi


def solve_poisson_cot(L, divX, src_idx):
    L_mod = L.tolil()
    b = divX.copy()

    L_mod[src_idx, :] = 0.0
    L_mod[src_idx, src_idx] = 1.0
    b[src_idx] = 0.0

    L_mod = L_mod.tocsc()
    phi = spsolve(L_mod, b)
    return phi


def compute_eigenpairs(
    L,
    M,
    k: int = 10,
    which: str = "SM",
    method: str = "auto",
    sigma: float = -1e-4,
    tol: float = 1e-5,
    maxiter: int = 20000,
    verbose: bool = True,
):
    """
    Robust generalized eigensolver for L x = lambda M x.

    Strategy:
      - default: shift-invert (fast)
      - cheap matrix sanity checks before entering ARPACK
      - try several sigmas
      - fallback to mass_diag only if needed
    """
    from scipy.sparse.linalg import ArpackNoConvergence

    # ---------------------------------------------------------
    # sanitize
    # ---------------------------------------------------------
    L = L.tocsc().astype(np.float64)
    M = M.tocsc().astype(np.float64)

    n = L.shape[0]
    if L.shape != M.shape:
        raise ValueError(f"[eigs] shape mismatch: L{L.shape} vs M{M.shape}")
    if n < 3:
        raise ValueError(f"[eigs] matrix too small: n={n}")

    k = min(int(k), n - 2)
    if k < 1:
        raise ValueError(f"[eigs] invalid k={k}, n={n}")

    if L.nnz == 0:
        raise RuntimeError("[eigs] L has no nonzeros")
    if M.nnz == 0:
        raise RuntimeError("[eigs] M has no nonzeros")

    if not np.isfinite(L.data).all():
        raise RuntimeError("[eigs] L has NaN/Inf entries")
    if not np.isfinite(M.data).all():
        raise RuntimeError("[eigs] M has NaN/Inf entries")

    m_diag = np.asarray(M.diagonal()).reshape(-1)
    if m_diag.size != n:
        raise RuntimeError("[eigs] failed to read M diagonal")
    if not np.isfinite(m_diag).all():
        raise RuntimeError("[eigs] M diagonal has NaN/Inf")
    if np.any(m_diag <= 0):
        bad = int(np.sum(m_diag <= 0))
        raise RuntimeError(f"[eigs] M diagonal has {bad} non-positive entries")

    # optional symmetry diagnostics
    L_skew = L - L.T
    M_skew = M - M.T
    asym_L = float(np.max(np.abs(L_skew.data))) if L_skew.nnz > 0 else 0.0
    asym_M = float(np.max(np.abs(M_skew.data))) if M_skew.nnz > 0 else 0.0

    if verbose:
        #print(
        #   f"[eigs] n={n}, k={k}, nnz(L)={L.nnz}, nnz(M)={M.nnz}, "
        #  f"L_range=[{L.data.min():.3e}, {L.data.max():.3e}], "
        #    f"Mdiag_range=[{m_diag.min():.3e}, {m_diag.max():.3e}]"
        #)
        if asym_L > 1e-8:
            print(f"[eigs] WARNING: L not symmetric enough, max|L-L^T|={asym_L:.3e}")
        if asym_M > 1e-8:
            print(f"[eigs] WARNING: M not symmetric enough, max|M-M^T|={asym_M:.3e}")

    if method == "auto":
        method = "shift_invert"

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------
    def _sort_pairs(evals, evecs):
        # Sort by |λ| ascending. For a PSD Laplacian this equals sorting by λ asc;
        # for an NSD Laplacian (cotangent convention with L_ii negative) this gives
        # low-frequency modes (|λ| small, starting at 0) first — which is what HKS
        # and all downstream code expect.
        idx = np.argsort(np.abs(evals))
        evals = evals[idx]
        evecs = evecs[:, idx]
        # Force non-negative Laplacian convention so HKS exp(-λt) is a decay.
        evals = np.abs(evals)
        return evals, evecs

    def _mass_diag(k_override=None):
        kk = k if k_override is None else min(int(k_override), n - 2)
        if kk < 1:
            raise RuntimeError(f"[eigs] invalid fallback k={kk}")

        inv_sqrt_m = 1.0 / np.sqrt(m_diag)
        Dm = diags(inv_sqrt_m)
        A = Dm @ L @ Dm
        A = A.tocsc()

        if not np.isfinite(A.data).all():
            raise RuntimeError("[eigs] mass_diag: A = DLD has NaN/Inf")

        evals, evecs_std = eigsh(A, k=kk, which="SM", tol=tol, maxiter=maxiter)
        evecs = Dm @ evecs_std
        return _sort_pairs(evals, evecs)

    def _shift_invert(sig):
        K = (L - sig * M).tocsc()

        if K.nnz == 0:
            raise RuntimeError(f"[eigs] K = L - {sig} M has no nonzeros")
        if not np.isfinite(K.data).all():
            raise RuntimeError(f"[eigs] K = L - {sig} M has NaN/Inf")

        # cheap conditioning proxy
        K_abs = np.abs(K.data)
        kmin = float(np.min(K_abs))
        kmax = float(np.max(K_abs))
        
        #print(
        #    f"[eigs] shift-invert sigma={sig:.3e}, "
        #    f"K_range=[{K.data.min():.3e}, {K.data.max():.3e}], "
        #    f"|K|_min={kmin:.3e}, |K|_max={kmax:.3e}"
        #)

        try:
            lu = splu(K)
        except Exception as e:
            raise RuntimeError(f"[eigs] splu failed (sigma={sig}): {e}")

        nan_count = [0]

        def matvec(v):
            x = lu.solve(v)
            if not np.isfinite(x).all():
                nan_count[0] += 1
                if nan_count[0] <= 3:
                    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    raise RuntimeError(
                        f"[eigs] lu.solve produced NaN/Inf repeatedly "
                        f"({nan_count[0]} times, sigma={sig})"
                    )
            return x

        OPinv = LinearOperator(shape=K.shape, matvec=matvec, dtype=np.float64)

        evals, evecs = eigsh(
            L,
            k=k,
            M=M,
            sigma=sig,
            OPinv=OPinv,
            which="LM",
            tol=tol,
            maxiter=maxiter,
        )

        if nan_count[0] > 0 and verbose:
            print(
                f"[eigs] WARNING: lu.solve produced NaN/Inf {nan_count[0]} time(s) "
                f"(sigma={sig})"
            )

        return _sort_pairs(evals, evecs)

    # ---------------------------------------------------------
    # solve
    # ---------------------------------------------------------
    sigmas = [sigma, sigma * 10.0, sigma * 100.0, -1e-3, -1e-2]

    if method == "shift_invert":
        last_err = None
        for s in sigmas:
            try:
                return _shift_invert(s)
            except ArpackNoConvergence as e:
                last_err = e
                got = 0 if e.eigenvalues is None else len(e.eigenvalues)
                print(f"[eigs] shift-invert no convergence with sigma={s}: got {got}/{k}")
            except Exception as e:
                last_err = e
                print(f"[eigs] shift-invert failed with sigma={s}: {repr(e)}")

        print("[eigs] shift-invert failed for all sigma. Falling back to mass_diag.")
        try:
            # fallback can use smaller k to avoid killing runtime too much
            return _mass_diag(k_override=min(k, 48))
        except Exception as e:
            raise RuntimeError(
                f"[eigs] shift_invert and mass_diag both failed. "
                f"Last shift-invert error: {repr(last_err)} | "
                f"mass_diag error: {repr(e)}"
            )

    elif method == "mass_diag":
        try:
            return _mass_diag()
        except ArpackNoConvergence as e:
            got = 0 if e.eigenvalues is None else len(e.eigenvalues)
            print(f"[eigs] mass_diag no convergence: got {got}/{k}")
            print("[eigs] Falling back to shift-invert.")
            last_err = e
            for s in sigmas:
                try:
                    return _shift_invert(s)
                except Exception as ee:
                    last_err = ee
                    print(f"[eigs] shift-invert failed with sigma={s}: {repr(ee)}")
            raise RuntimeError(f"[eigs] mass_diag and shift_invert both failed: {repr(last_err)}")
        except Exception as e:
            print(f"[eigs] mass_diag failed: {repr(e)}")
            print("[eigs] Falling back to shift-invert.")
            last_err = e
            for s in sigmas:
                try:
                    return _shift_invert(s)
                except Exception as ee:
                    last_err = ee
                    print(f"[eigs] shift-invert failed with sigma={s}: {repr(ee)}")
            raise RuntimeError(f"[eigs] mass_diag and shift_invert both failed: {repr(last_err)}")

    else:
        raise ValueError(f"Unknown method: {method}")
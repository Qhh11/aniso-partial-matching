from __future__ import annotations

import numpy as np
import igl

from collections import deque
from scipy.spatial import cKDTree

# ============================================================
# Basic geometry / shared helpers
# ============================================================

def _build_vertex_adjacency(F: np.ndarray, N: int):
    adj = [set() for _ in range(N)]
    for a, b, c in np.asarray(F, dtype=np.int64):
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return [np.fromiter(s, dtype=np.int64) for s in adj]


def mean_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    V = np.asarray(V, float)
    F = np.asarray(F, int)

    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    L = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    return float(np.mean(L))


def arbitrary_tangent(n, eps=1e-12):
    n = np.asarray(n, dtype=np.float64)
    n = n / (np.linalg.norm(n) + eps)

    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    t = a - np.dot(a, n) * n
    nt = np.linalg.norm(t)
    if nt < eps:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t = a - np.dot(a, n) * n
        nt = np.linalg.norm(t)

    return t / max(nt, eps)


def _normalize_rows(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    good = n[:, 0] > eps
    Y = X.copy()
    Y[good] /= n[good]
    return Y


def _project_tangent(v, n, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    nn = np.linalg.norm(n)
    if nn < eps:
        return None

    n = n / nn
    out = v - np.dot(v, n) * n
    no = np.linalg.norm(out)
    if no < eps:
        return None
    return out / no

def _line_angle_rad(u, v, eps=1e-12):
    """
    Sign-invariant angle between two line-field directions.
    Range: [0, pi/2].
    """
    u = np.asarray(u, dtype=np.float64).reshape(3)
    v = np.asarray(v, dtype=np.float64).reshape(3)

    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        return np.inf

    u = u / nu
    v = v / nv

    c = float(np.clip(abs(np.dot(u, v)), 0.0, 1.0))
    return float(np.arccos(c))

def _dbg_stats(name, x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"[dbg:{name}] empty")
        return

    print(
        f"[dbg:{name}] "
        f"mean={x.mean():.6e}, "
        f"p01={np.percentile(x, 1):.6e}, "
        f"p05={np.percentile(x, 5):.6e}, "
        f"p50={np.percentile(x, 50):.6e}, "
        f"p90={np.percentile(x, 90):.6e}, "
        f"p95={np.percentile(x, 95):.6e}, "
        f"max={x.max():.6e}, "
        f"min={x.min():.6e}"
    )

def vertex_normals(V: np.ndarray, F: np.ndarray):
    vn = igl.per_vertex_normals(V, F)
    if isinstance(vn, tuple):
        vn = vn[0]
    return np.asarray(vn, float)


def face_normals(V: np.ndarray, F: np.ndarray, eps=1e-12):
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)

    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nrm = np.linalg.norm(n, axis=1, keepdims=True)
    good = nrm[:, 0] > eps
    n[good] /= nrm[good]
    return n


def precompute_face_mean_edge_lengths(V, F):
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)

    P0 = V[F[:, 0]]
    P1 = V[F[:, 1]]
    P2 = V[F[:, 2]]

    l01 = np.linalg.norm(P1 - P0, axis=1)
    l12 = np.linalg.norm(P2 - P1, axis=1)
    l20 = np.linalg.norm(P0 - P2, axis=1)

    return (l01 + l12 + l20) / 3.0


def project_points_to_mesh_with_face_batch(X, V, F):
    X = np.asarray(X, float).reshape(-1, 3)
    if X.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )

    _, I, C = igl.point_mesh_squared_distance(X, V, F)
    face_ids = np.asarray(I, np.int64).reshape(-1)
    X_proj = np.asarray(C, float).reshape(-1, 3)
    return X_proj, face_ids


def barycentric_on_triangle_projected(x, P, eps=1e-12):
    x = np.asarray(x, float).reshape(3)
    P = np.asarray(P, float).reshape(3, 3)

    a, b, c = P
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < eps:
        return None
    n = n / nn

    x_proj = x - np.dot(x - a, n) * n

    v0 = b - a
    v1 = c - a
    v2 = x_proj - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < eps:
        return None

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return np.array([u, v, w], dtype=np.float64)


def _project_point_to_plane_at_anchor(B, A, n, eps=1e-12):
    B = np.asarray(B, dtype=np.float64).reshape(3)
    A = np.asarray(A, dtype=np.float64).reshape(3)
    n = np.asarray(n, dtype=np.float64).reshape(3)

    nn = np.linalg.norm(n)
    if nn < eps:
        return B.copy()

    n = n / nn
    return B - np.dot(B - A, n) * n


def _normalize_vec(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    nv = np.linalg.norm(v)
    if nv < eps:
        return None
    return v / nv


def _build_U_from_u1(u1, n, eps=1e-12):
    n = np.asarray(n, dtype=np.float64).reshape(3)
    nn = np.linalg.norm(n)
    if nn < eps:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = n / nn

    u1 = _project_tangent(u1, n, eps=eps)
    if u1 is None:
        u1 = arbitrary_tangent(n, eps=eps)
    u1 = u1 / (np.linalg.norm(u1) + eps)

    u2 = np.cross(n, u1)
    nu2 = np.linalg.norm(u2)
    if nu2 < eps:
        u1 = arbitrary_tangent(n, eps=eps)
        u1 = u1 / (np.linalg.norm(u1) + eps)
        u2 = np.cross(n, u1)
        nu2 = np.linalg.norm(u2)

    u2 = u2 / (nu2 + eps)
    U = np.column_stack([u1, u2])
    return U, u1, u2


def _coords_in_U(v, U):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    U = np.asarray(U, dtype=np.float64).reshape(3, 2)
    return (U.T @ v).astype(np.float64)


def _reconstruct_from_U(c, U, n, eps=1e-12):
    c = np.asarray(c, dtype=np.float64).reshape(2)
    U = np.asarray(U, dtype=np.float64).reshape(3, 2)

    v = U @ c
    v = _project_tangent(v, n, eps=eps)
    if v is None:
        v = U[:, 0].copy()
    v = v / (np.linalg.norm(v) + eps)
    return v


def _choose_ordered_face_edge_for_u(V, F, f_src, u_ref, n_ref, eps=1e-12):
    tri = np.asarray(F[int(f_src)], np.int64)

    best = None
    best_score = -np.inf

    for ia in range(3):
        for ib in range(3):
            if ia == ib:
                continue

            a = int(tri[ia])
            b = int(tri[ib])

            t_ab = _project_tangent(V[b] - V[a], n_ref, eps=eps)
            if t_ab is None:
                continue

            score = float(np.dot(t_ab, u_ref))
            if score > best_score:
                best_score = score
                best = (a, b, t_ab)

    return best


def smooth_vector_field_on_mesh(
    V, F, T_in,
    n_iter=5,
    anchor_mask=None,
    anchor_weight=10.0,
    project_to_tangent=True,
    flip_sign_for_line_field=True,
    eps=1e-12,
):
    """
    Graph-Laplacian smoothing for a vertex tangent line field T_in (N,3).
    - At each iteration: T[i] <- mix(T_in[i], mean of neighbors after sign-align)
    - anchor vertices (alpha>thresh) keep more of T_in
    - re-project to tangent plane each iter
    """
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    T = np.asarray(T_in, np.float64).copy()
    N = V.shape[0]
    VN = vertex_normals(V, F)

    # Build 1-ring adjacency
    adj = [set() for _ in range(N)]
    for a, b, c in F:
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    adj = [list(s) for s in adj]

    if anchor_mask is None:
        anchor_mask = np.zeros(N, dtype=bool)
    aw = np.where(anchor_mask, anchor_weight, 1.0)

    T_orig = T.copy()

    for it in range(n_iter):
        T_new = np.zeros_like(T)
        for i in range(N):
            nbrs = adj[i]
            if len(nbrs) == 0:
                T_new[i] = T[i]
                continue

            ref = T[i]
            acc = aw[i] * ref            # self with weight
            wsum = aw[i]
            for j in nbrs:
                t_j = T[j]
                if flip_sign_for_line_field and np.dot(t_j, ref) < 0.0:
                    t_j = -t_j
                acc = acc + t_j
                wsum += 1.0

            T_new[i] = acc / max(wsum, eps)

        if project_to_tangent:
            dot = np.sum(T_new * VN, axis=1, keepdims=True)
            T_new = T_new - dot * VN

        # Soft pull-back toward anchor's original direction
        T_new[anchor_mask] = 0.5 * T_new[anchor_mask] + 0.5 * T_orig[anchor_mask]

        # renormalize
        nrm = np.linalg.norm(T_new, axis=1, keepdims=True)
        good = nrm[:, 0] > eps
        T_new[good] /= nrm[good]
        T_new[~good] = T_orig[~good]   # fall back

        T = T_new

    return T

def map_segment_src_to_tgt_via_fixed_triplet_affine3d(
    A_src,
    B_src,
    X_src,
    X_tgt,
    triplet_ids,
    eps=1e-12,
):
    """
    Map an AB segment from src to tgt using the FULL 3x3 affine map defined by
    a triplet of corresponding anchor points. Unlike the barycentric version,
    this does NOT project A or B onto the triplet's plane, so the segment's
    perpendicular-to-triplet-plane component is preserved (and rotated into
    the corresponding component of the target triplet).

    For src == tgt this map is EXACT identity, so A_tgt = A_src and
    B_tgt = B_src bit-for-bit.

    Returns (A_tgt, B_tgt, lam_A) where lam_A is the barycentric coords of
    A_src in the src triangle (kept for back-compat with callers that use it).
    """
    ids = np.asarray(triplet_ids, np.int64).reshape(3)

    a_s = X_src[ids[0]]
    b_s = X_src[ids[1]]
    c_s = X_src[ids[2]]

    a_t = X_tgt[ids[0]]
    b_t = X_tgt[ids[1]]
    c_t = X_tgt[ids[2]]

    e1_s = b_s - a_s
    e2_s = c_s - a_s
    n_s_raw = np.cross(e1_s, e2_s)
    n_s_norm = np.linalg.norm(n_s_raw)
    if n_s_norm < eps:
        raise RuntimeError("Degenerate src triplet (collinear anchors).")
    n_s = n_s_raw / n_s_norm

    e1_t = b_t - a_t
    e2_t = c_t - a_t
    n_t_raw = np.cross(e1_t, e2_t)
    n_t_norm = np.linalg.norm(n_t_raw)
    if n_t_norm < eps:
        raise RuntimeError("Degenerate tgt triplet (collinear anchors).")
    n_t = n_t_raw / n_t_norm

    F_s = np.column_stack([e1_s, e2_s, n_s]).astype(np.float64)
    F_t = np.column_stack([e1_t, e2_t, n_t]).astype(np.float64)

    F_s_inv = np.linalg.inv(F_s)
    J = F_t @ F_s_inv   # 3x3

    A_tgt = a_t + J @ (np.asarray(A_src, np.float64) - a_s)
    B_tgt = a_t + J @ (np.asarray(B_src, np.float64) - a_s)

    # Also return lam_A for any caller that wants it (back-compat).
    # Use the 2D in-plane part (drop normal coord).
    coord_A = F_s_inv @ (np.asarray(A_src, np.float64) - a_s)
    alpha, beta = float(coord_A[0]), float(coord_A[1])
    lam_A = np.array([1.0 - alpha - beta, alpha, beta], dtype=np.float64)

    return A_tgt, B_tgt, lam_A

# ============================================================
# Template / canonical helpers for the new transfer path
# ============================================================

def pick_local_template_triplet(
    X_embed,
    x_query,
    kdtree=None,
    k=10,
    area_eps=1e-10,
):
    X_embed = np.asarray(X_embed, float)
    x_query = np.asarray(x_query, float).reshape(3)

    if kdtree is None:
        kdtree = cKDTree(X_embed)

    kk = min(int(k), X_embed.shape[0])
    _, idx = kdtree.query(x_query, k=kk)
    idx = np.atleast_1d(idx).astype(np.int64)

    if idx.size < 3:
        return None

    P = X_embed[idx]

    i0 = 0
    v = P - P[i0]
    j_rel = np.argmax(np.linalg.norm(v, axis=1))
    if j_rel == i0:
        cand = np.argsort(np.linalg.norm(v, axis=1))
        j_rel = cand[-2]

    p0 = P[i0]
    p1 = P[j_rel]

    base = p1 - p0
    best_rel = -1
    best_area = -np.inf

    for r in range(P.shape[0]):
        if r == i0 or r == j_rel:
            continue
        area2 = np.linalg.norm(np.cross(base, P[r] - p0))
        if area2 > best_area:
            best_area = area2
            best_rel = r

    if best_rel < 0 or best_area < area_eps:
        for a in range(P.shape[0]):
            for b in range(a + 1, P.shape[0]):
                for c in range(b + 1, P.shape[0]):
                    area2 = np.linalg.norm(np.cross(P[b] - P[a], P[c] - P[a]))
                    if area2 > area_eps:
                        return idx[[a, b, c]]
        return None

    return idx[[i0, j_rel, best_rel]]


def map_point_src_to_tgt_via_fixed_triplet(
    x_src,
    X_src,
    X_tgt,
    triplet_ids,
    eps=1e-12,
):
    ids = np.asarray(triplet_ids, np.int64).reshape(3)
    lam = barycentric_on_triangle_projected(x_src, X_src[ids], eps=eps)
    if lam is None:
        raise RuntimeError("Fixed-triplet barycentric map failed.")

    x_tgt = (
        lam[0] * X_tgt[ids[0]]
        + lam[1] * X_tgt[ids[1]]
        + lam[2] * X_tgt[ids[2]]
    )
    return x_tgt, lam


def build_common_template_embeddings_from_v2tid(
    V_src,
    src_v2tid,
    V_tgt,
    tgt_v2tid,
):
    src_v2tid = np.asarray(src_v2tid, np.int64).reshape(-1)
    tgt_v2tid = np.asarray(tgt_v2tid, np.int64).reshape(-1)

    tid2src = {}
    for vs, tid in enumerate(src_v2tid):
        if tid >= 0 and tid not in tid2src:
            tid2src[int(tid)] = int(vs)

    tid2tgt = {}
    for vt, tid in enumerate(tgt_v2tid):
        if tid >= 0 and tid not in tid2tgt:
            tid2tgt[int(tid)] = int(vt)

    common_tids = np.array(
        sorted(set(tid2src.keys()) & set(tid2tgt.keys())),
        dtype=np.int64,
    )
    if common_tids.size < 3:
        raise RuntimeError("Too few common template ids for template-pull transfer.")

    src_ids = np.array([tid2src[int(t)] for t in common_tids], dtype=np.int64)
    tgt_ids = np.array([tid2tgt[int(t)] for t in common_tids], dtype=np.int64)

    X_src = np.asarray(V_src, float)[src_ids].copy()
    X_tgt = np.asarray(V_tgt, float)[tgt_ids].copy()

    return X_src, X_tgt, common_tids, src_ids, tgt_ids


def build_target_driven_donor_query_from_vertices(
    V_tgt,
    X_tgt,
    X_src,
    k_triplet=24,
    eps=1e-12,
):
    V_tgt = np.asarray(V_tgt, float)
    X_tgt = np.asarray(X_tgt, float)
    X_src = np.asarray(X_src, float)

    N_tgt = V_tgt.shape[0]
    tree_tgt = cKDTree(X_tgt)

    donor_triplet = -np.ones((N_tgt, 3), dtype=np.int64)
    donor_lam = np.full((N_tgt, 3), np.nan, dtype=np.float64)
    donor_point_src = np.full((N_tgt, 3), np.nan, dtype=np.float64)
    donor_point_tgt = np.full((N_tgt, 3), np.nan, dtype=np.float64)
    valid = np.zeros(N_tgt, dtype=bool)

    for i_tgt in range(N_tgt):
        x_tgt = V_tgt[i_tgt]

        ids = pick_local_template_triplet(
            X_tgt,
            x_tgt,
            kdtree=tree_tgt,
            k=k_triplet,
        )
        if ids is None:
            continue

        lam = barycentric_on_triangle_projected(x_tgt, X_tgt[ids], eps=eps)
        if lam is None:
            continue

        donor_triplet[i_tgt] = ids
        donor_lam[i_tgt] = lam
        donor_point_tgt[i_tgt] = (
            lam[0] * X_tgt[ids[0]]
            + lam[1] * X_tgt[ids[1]]
            + lam[2] * X_tgt[ids[2]]
        )
        donor_point_src[i_tgt] = (
            lam[0] * X_src[ids[0]]
            + lam[1] * X_src[ids[1]]
            + lam[2] * X_src[ids[2]]
        )
        valid[i_tgt] = True

    return donor_triplet, donor_lam, donor_point_src, donor_point_tgt, valid


def build_p2p_donor_query_from_vertices(
    V_tgt,
    X_tgt,
    X_src,
    tgt_to_anchor,      # (N_tgt,) int: for each full vertex, the anchor-cloud index it maps to (-1 if none)
    eps=1e-12,
):
    """
    DIRECT p2p donor query (no barycentric interpolation).

    For each full vertex i, its exact source is the single anchor tgt_to_anchor[i].
    We produce the SAME output format as the barycentric version so the downstream
    AB-arrow direction mapping is unchanged:
        donor_point_src[i] = X_src[k]        (exact source anchor position)
        donor_point_tgt[i] = X_tgt[k]        (exact target anchor position == this vertex)
        donor_triplet[i]   = [k, k2, k3]     (k plus its 2 nearest anchors, to keep a
                                              non-degenerate local frame for the affine map)
        donor_lam[i]       = [1, 0, 0]       (all weight on the exact p2p anchor)
    """
    V_tgt = np.asarray(V_tgt, float)
    X_tgt = np.asarray(X_tgt, float)
    X_src = np.asarray(X_src, float)
    tgt_to_anchor = np.asarray(tgt_to_anchor, np.int64).reshape(-1)

    N_tgt = V_tgt.shape[0]
    n_anchor = X_tgt.shape[0]
    tree_tgt = cKDTree(X_tgt)

    donor_triplet = -np.ones((N_tgt, 3), dtype=np.int64)
    donor_lam = np.full((N_tgt, 3), np.nan, dtype=np.float64)
    donor_point_src = np.full((N_tgt, 3), np.nan, dtype=np.float64)
    donor_point_tgt = np.full((N_tgt, 3), np.nan, dtype=np.float64)
    valid = np.zeros(N_tgt, dtype=bool)

    kq = min(3, n_anchor)
    for i_tgt in range(N_tgt):
        k = int(tgt_to_anchor[i_tgt])
        if k < 0 or k >= n_anchor:
            continue
        # nearest 2 other anchors to form a non-degenerate frame (for the affine map)
        _, nbr = tree_tgt.query(X_tgt[k], k=kq)
        nbr = np.atleast_1d(nbr).astype(np.int64)
        k2 = int(nbr[1]) if nbr.size > 1 and nbr[1] != k else k
        k3 = int(nbr[2]) if nbr.size > 2 and nbr[2] not in (k, k2) else k2

        donor_triplet[i_tgt] = np.array([k, k2, k3], dtype=np.int64)
        donor_lam[i_tgt] = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # exact p2p, no interp
        donor_point_tgt[i_tgt] = X_tgt[k]     # == this full vertex (anchor is exact)
        donor_point_src[i_tgt] = X_src[k]     # exact source anchor position
        valid[i_tgt] = True

    return donor_triplet, donor_lam, donor_point_src, donor_point_tgt, valid


# ============================================================
# Continuous vertex-tensor evaluation
# ============================================================

def evaluate_vertex_tensor_at_projected_point(
    x_proj,
    face_id,
    V,
    F,
    VN,
    T_v,
    D_v,
    eps=1e-12,
):
    face_id = int(face_id)
    tri = np.asarray(F[face_id], np.int64)
    P = V[tri]

    lam = barycentric_on_triangle_projected(x_proj, P, eps=eps)
    if lam is None:
        lam = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    n_p = lam[0] * VN[tri[0]] + lam[1] * VN[tri[1]] + lam[2] * VN[tri[2]]
    nn = np.linalg.norm(n_p)
    if nn < eps:
        n_p = np.cross(P[1] - P[0], P[2] - P[0])
        nn = np.linalg.norm(n_p)
        if nn < eps:
            n_p = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            nn = 1.0
    n_p = n_p / (nn + eps)

    T_tri = np.asarray(T_v[tri], dtype=np.float64).copy()
    ref = T_tri[0].copy()
    nr = np.linalg.norm(ref)
    if nr < eps:
        ref = arbitrary_tangent(n_p, eps=eps)
    else:
        ref = ref / nr

    for k in [1, 2]:
        tk = T_tri[k]
        nk = np.linalg.norm(tk)
        if nk < eps:
            continue
        tk = tk / nk
        if np.dot(tk, ref) < 0.0:
            T_tri[k] = -T_tri[k]

    t_p = lam[0] * T_tri[0] + lam[1] * T_tri[1] + lam[2] * T_tri[2]
    t_p = _project_tangent(t_p, n_p, eps=eps)
    if t_p is None:
        e = P[1] - P[0]
        t_p = _project_tangent(e, n_p, eps=eps)
    if t_p is None:
        t_p = arbitrary_tangent(n_p, eps=eps)
    t_p = t_p / (np.linalg.norm(t_p) + eps)

    r_p = np.cross(n_p, t_p)
    nrp = np.linalg.norm(r_p)
    if nrp < eps:
        r_p = arbitrary_tangent(n_p, eps=eps)
        nrp = np.linalg.norm(r_p)
    r_p = r_p / (nrp + eps)

    D_p = lam[0] * D_v[tri[0]] + lam[1] * D_v[tri[1]] + lam[2] * D_v[tri[2]]
    D_p = np.asarray(D_p, dtype=np.float64)

    return n_p, t_p, r_p, D_p


def batch_build_target_directions_from_local_correspondence_vertex(
    donor_valid,
    donor_triplet,
    donor_point_src,
    donor_point_tgt,
    X_src,
    X_tgt,
    V_src,
    F_src,
    VN_src,
    T_src_v,
    D_src_v,
    V_tgt,
    F_tgt,
    VN_tgt,
    T_tgt_self_v,
    D_tgt_self_v,
    eps=1e-12,
    debug_trace=False,
    debug_trace_n=5,
):
    donor_valid = np.asarray(donor_valid, bool).reshape(-1)
    donor_triplet = np.asarray(donor_triplet, np.int64)
    donor_point_src = np.asarray(donor_point_src, np.float64)
    donor_point_tgt = np.asarray(donor_point_tgt, np.float64)

    N_tgt = donor_valid.shape[0]

    out = { "ok": np.zeros(N_tgt, dtype=bool), "A_src_m": np.full((N_tgt, 3), np.nan, dtype=np.float64), "faceA_src": -np.ones(N_tgt, dtype=np.int64), "B_src_m": np.full((N_tgt, 3), np.nan, dtype=np.float64), "B_src_t": np.full((N_tgt, 3), np.nan, dtype=np.float64), "v_src_t": np.full((N_tgt, 3), np.nan, dtype=np.float64), "v_u_y": np.full((N_tgt, 2), np.nan, dtype=np.float64), "D_src_point": np.full((N_tgt, 2), np.nan, dtype=np.float64), "A_tgt_raw": np.full((N_tgt, 3), np.nan, dtype=np.float64), "B_tgt_raw": np.full((N_tgt, 3), np.nan, dtype=np.float64), "A_tgt_m": np.full((N_tgt, 3), np.nan, dtype=np.float64), "faceA_tgt": -np.ones(N_tgt, dtype=np.int64), "B_tgt_m": np.full((N_tgt, 3), np.nan, dtype=np.float64), "B_tgt_t": np.full((N_tgt, 3), np.nan, dtype=np.float64), "v_tgt_obs": np.full((N_tgt, 3), np.nan, dtype=np.float64), "v_tilde_u_x": np.full((N_tgt, 2), np.nan, dtype=np.float64), "v_tilde": np.full((N_tgt, 3), np.nan, dtype=np.float64), "consistency_error": np.full(N_tgt, np.inf, dtype=np.float64), }

    idx = np.where(donor_valid)[0]
    if idx.size == 0:
        return out

    face_step_src = precompute_face_mean_edge_lengths(V_src, F_src)
    n_src_f = face_normals(V_src, F_src)

    A_src_m_batch, faceA_src_batch = project_points_to_mesh_with_face_batch(
        donor_point_src[idx], V_src, F_src
    )

    B_src_t_guess_batch = np.zeros((idx.size, 3), dtype=np.float64)

    for k, i_global in enumerate(idx):
        face_id = int(faceA_src_batch[k])
        A_src_m = A_src_m_batch[k]

        n_A_src, u1_src, _, D_src_point = evaluate_vertex_tensor_at_projected_point(
            x_proj=A_src_m,
            face_id=face_id,
            V=V_src,
            F=F_src,
            VN=VN_src,
            T_v=T_src_v,
            D_v=D_src_v,
            eps=eps,
        )

        step = float(face_step_src[face_id])
        if step < eps:
            step = mean_edge_length(V_src, F_src)
        step = max(step, eps)

        B_src_t_guess_batch[k] = A_src_m + step * u1_src

        out["A_src_m"][i_global] = A_src_m
        out["faceA_src"][i_global] = face_id
        out["D_src_point"][i_global] = D_src_point

    B_src_m_batch, _ = project_points_to_mesh_with_face_batch(
        B_src_t_guess_batch, V_src, F_src
    )
    n_edge_fallback = 0
    
    for k, i_global in enumerate(idx):
        A_src_m = out["A_src_m"][i_global]
        face_id = int(out["faceA_src"][i_global])
        B_src_m = B_src_m_batch[k]

        n_A_src, u1_src, _, _ = evaluate_vertex_tensor_at_projected_point(
            x_proj=A_src_m,
            face_id=face_id,
            V=V_src,
            F=F_src,
            VN=VN_src,
            T_v=T_src_v,
            D_v=D_src_v,
            eps=eps,
        )

        step = float(face_step_src[face_id])
        if np.linalg.norm(B_src_m - A_src_m) < 0.25 * step:
            edge_choice = _choose_ordered_face_edge_for_u(
                V_src, F_src, face_id, u1_src, n_src_f[face_id], eps=eps
            )
            if edge_choice is not None:
                n_edge_fallback += 1

                a, b, _ = edge_choice
                A_src_m = V_src[a].copy()
                B_src_m = V_src[b].copy()

                n_A_src, u1_src, _, D_src_point = evaluate_vertex_tensor_at_projected_point(
                    x_proj=A_src_m,
                    face_id=face_id,
                    V=V_src,
                    F=F_src,
                    VN=VN_src,
                    T_v=T_src_v,
                    D_v=D_src_v,
                    eps=eps,
                )

                out["A_src_m"][i_global] = A_src_m
                out["D_src_point"][i_global] = D_src_point

        B_src_t = _project_point_to_plane_at_anchor(B_src_m, A_src_m, n_A_src, eps=eps)
        v_src_raw = B_src_t - A_src_m
        v_src_t = _normalize_vec(v_src_raw, eps=eps)
        if v_src_t is None:
            continue

        U_y, _, _ = _build_U_from_u1(u1_src, n_A_src, eps=eps)
        v_u_y = _coords_in_U(v_src_raw, U_y)

        out["B_src_m"][i_global] = B_src_m
        out["B_src_t"][i_global] = B_src_t
        out["v_src_t"][i_global] = v_src_t
        out["v_u_y"][i_global] = v_u_y

    valid_AB = np.isfinite(out["A_src_m"][:, 0]) & np.isfinite(out["B_src_m"][:, 0])
    idx_ab = np.where(valid_AB)[0]
    if idx_ab.size == 0:
        return out

    A_tgt_raw_batch = np.full((idx_ab.size, 3), np.nan, dtype=np.float64)
    B_tgt_raw_batch = np.full((idx_ab.size, 3), np.nan, dtype=np.float64)
    valid_map = np.ones(idx_ab.size, dtype=bool)

    for k, i_global in enumerate(idx_ab):
        ids_A = donor_triplet[i_global]

        try:
            A_tgt_raw_batch[k], B_tgt_raw_batch[k], _ = (
                map_segment_src_to_tgt_via_fixed_triplet_affine3d(
                    A_src=out["A_src_m"][i_global],
                    B_src=out["B_src_m"][i_global],
                    X_src=X_src,
                    X_tgt=X_tgt,
                    triplet_ids=ids_A,
                    eps=eps,
                )
            )

        except Exception:
            valid_map[k] = False

    idx_map = idx_ab[valid_map]
    if idx_map.size == 0:
        return out

    A_tgt_raw_valid = A_tgt_raw_batch[valid_map]
    B_tgt_raw_valid = B_tgt_raw_batch[valid_map]

    A_tgt_m_sub, faceA_tgt_sub = project_points_to_mesh_with_face_batch(
        A_tgt_raw_valid, V_tgt, F_tgt
    )
    B_tgt_m_sub, _ = project_points_to_mesh_with_face_batch(
        B_tgt_raw_valid, V_tgt, F_tgt
    )

    for j, i_global in enumerate(idx_map):
        out["A_tgt_raw"][i_global] = A_tgt_raw_valid[j]
        out["B_tgt_raw"][i_global] = B_tgt_raw_valid[j]
        out["A_tgt_m"][i_global] = A_tgt_m_sub[j]
        out["faceA_tgt"][i_global] = int(faceA_tgt_sub[j])
        out["B_tgt_m"][i_global] = B_tgt_m_sub[j]

    valid_tgt = np.isfinite(out["A_tgt_m"][:, 0]) & np.isfinite(out["B_tgt_m"][:, 0])
    idx_tgt = np.where(valid_tgt)[0]
    
    for i_global in idx_tgt:
        A_tgt_m = out["A_tgt_m"][i_global]
        B_tgt_m = out["B_tgt_m"][i_global]
        face_id = int(out["faceA_tgt"][i_global])

        n_A_tgt, u1_tgt_self, _, _ = evaluate_vertex_tensor_at_projected_point(
            x_proj=A_tgt_m,
            face_id=face_id,
            V=V_tgt,
            F=F_tgt,
            VN=VN_tgt,
            T_v=T_tgt_self_v,
            D_v=D_tgt_self_v,
            eps=eps,
        )

        B_tgt_t = _project_point_to_plane_at_anchor(B_tgt_m, A_tgt_m, n_A_tgt, eps=eps)
        v_tgt_obs_raw = B_tgt_t - A_tgt_m
        v_tgt_obs = _normalize_vec(v_tgt_obs_raw, eps=eps)
        if v_tgt_obs is None:
            continue

        U_x, _, _ = _build_U_from_u1(u1_tgt_self, n_A_tgt, eps=eps)
        v_tilde_u_x = _coords_in_U(v_tgt_obs_raw, U_x)
        v_tilde = _reconstruct_from_U(v_tilde_u_x, U_x, n_A_tgt, eps=eps)

        out["B_tgt_t"][i_global] = B_tgt_t
        out["v_tgt_obs"][i_global] = v_tgt_obs
        out["v_tilde_u_x"][i_global] = v_tilde_u_x
        out["v_tilde"][i_global] = v_tilde
        
        e_i = np.linalg.norm(A_tgt_m - donor_point_tgt[i_global])
        out["consistency_error"][i_global] = float(e_i)

        out["ok"][i_global] = True
    
    # ============================================================
    # DIAGNOSTIC: per-vertex trace for first few OK vertices.
    # Triggers only when debug_trace=True (passed by caller).
    # ============================================================
    if debug_trace:
        ok_mask = out["ok"]
        ok_ids = np.where(ok_mask)[0][:debug_trace_n]

        for i_global in ok_ids:
            i_global = int(i_global)
            face_src = int(out["faceA_src"][i_global])

            # Recompute u1_src at A_src_m to compare with what was used
            n_dbg, u1_src_dbg, _, _ = evaluate_vertex_tensor_at_projected_point(
                x_proj=out["A_src_m"][i_global],
                face_id=face_src,
                V=V_src, F=F_src, VN=VN_src,
                T_v=T_src_v, D_v=D_src_v, eps=eps,
            )
            u1_src_n = u1_src_dbg / (np.linalg.norm(u1_src_dbg) + 1e-12)

            T_ref_i = T_tgt_self_v[i_global]
            T_ref_n = T_ref_i / (np.linalg.norm(T_ref_i) + 1e-12)

            v_obs   = out["v_tgt_obs"][i_global]
            v_tilde = out["v_tilde"][i_global]
            v_obs_n   = v_obs   / (np.linalg.norm(v_obs)   + 1e-12)
            v_tilde_n = v_tilde / (np.linalg.norm(v_tilde) + 1e-12)

            def _line_ang(a, b):
                c = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
                return float(np.degrees(np.arccos(c)))

            err_donor      = float(np.linalg.norm(donor_point_src[i_global] - donor_point_tgt[i_global]))
            err_A_src_tgt  = float(np.linalg.norm(out["A_src_m"][i_global] - out["A_tgt_m"][i_global]))
            err_B_src_tgt  = float(np.linalg.norm(out["B_src_m"][i_global] - out["B_tgt_m"][i_global]))
            err_A_to_V     = float(np.linalg.norm(out["A_tgt_m"][i_global] - V_tgt[i_global]))

            ang_usrc_ref   = _line_ang(u1_src_n, T_ref_n)
            ang_vobs_ref   = _line_ang(v_obs_n,  T_ref_n)
            ang_vtilde_ref = _line_ang(v_tilde_n, T_ref_n)
            ang_vobs_usrc  = _line_ang(v_obs_n,  u1_src_n)

            print(f"\n[trace] i_tgt={i_global}  faceA_src={face_src}")
            print(f"  donor_point_src - donor_point_tgt  ={err_donor:.3e}")
            print(f"  A_src_m - A_tgt_m                  ={err_A_src_tgt:.3e}  (should be 0 for src=tgt)")
            print(f"  B_src_m - B_tgt_m                  ={err_B_src_tgt:.3e}  (should be 0 for src=tgt)")
            print(f"  A_tgt_m - V_tgt[i] (anchor_err)    ={err_A_to_V:.3e}")
            print(f"  angle(u1_src,  T_ref[i]) = {ang_usrc_ref:.3f} deg")
            print(f"  angle(v_obs,   T_ref[i]) = {ang_vobs_ref:.3f} deg")
            print(f"  angle(v_obs,   u1_src ) = {ang_vobs_usrc:.3f} deg")
            print(f"  angle(v_tilde, T_ref[i]) = {ang_vtilde_ref:.3f} deg  <-- assigned to T_out[i]")

            print(f"  u1_src         = {u1_src_n}")
            print(f"  T_ref[i]       = {T_ref_n}")
            print(f"  v_obs          = {v_obs_n}")
            print(f"  v_tilde        = {v_tilde_n}")

    #print(
    #    f"[debug-transfer] edge_fallback={n_edge_fallback}/{idx.size} "
    #    f"rate={n_edge_fallback / max(idx.size, 1):.4f}"
    #)
    return out

def _normalize_soft_column_weights(w, topk=None, eps=1e-12):
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)

    if topk is not None and topk < w.size:
        idx = np.argpartition(w, -topk)[-topk:]
        keep = np.zeros_like(w, dtype=bool)
        keep[idx] = True
        w = np.where(keep, w, 0.0)

    s = float(w.sum())
    if s <= eps:
        return None
    return w / s


def _soft_columns_to_points(V_src, W_src_to_tgt, topk=20, eps=1e-12):
    """
    Vectorized version.

    W_src_to_tgt: (N_src, N_tgt)
    For each target column i:
        X[i] = sum_j W[j, i] * V_src[j]

    Returns:
        X:     (N_tgt, 3)
        valid: (N_tgt,)
    """
    V_src = np.asarray(V_src, np.float64)
    W = np.asarray(W_src_to_tgt, np.float64)
    W = np.where(np.isfinite(W), W, 0.0)
    W = np.maximum(W, 0.0)

    N_src, N_tgt = W.shape
    X = np.full((N_tgt, 3), np.nan, dtype=np.float64)
    valid = np.zeros(N_tgt, dtype=bool)

    kk = min(int(topk), N_src)

    if kk <= 0:
        return X, valid

    if kk == N_src:
        idx = np.tile(np.arange(N_src, dtype=np.int64)[:, None], (1, N_tgt))
        vals = W.copy()
    else:
        kth = N_src - kk
        idx = np.argpartition(W, kth, axis=0)[-kk:, :]
        vals = np.take_along_axis(W, idx, axis=0)

    vals = np.maximum(vals, 0.0)
    s = vals.sum(axis=0)
    ok = s > eps

    if np.any(ok):
        vals[:, ok] /= s[ok][None, :]
        # V_src[idx] -> (kk, N_tgt, 3)
        X[ok] = np.einsum(
            "kn,knc->nc",
            vals[:, ok],
            V_src[idx[:, ok]],
        )
        valid[ok] = True

    return X, valid


def _map_part_surface_point_to_full_linear(
    x_part,
    face_id_part,
    V_part,
    F_part,
    part_full_pos,
    part_full_valid,
    eps=1e-12,
):
    face_id_part = int(face_id_part)
    tri = np.asarray(F_part[face_id_part], np.int64)

    lam = barycentric_on_triangle_projected(
        x_part,
        V_part[tri],
        eps=eps,
    )
    if lam is None:
        lam = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    if not np.all(part_full_valid[tri]):
        return None

    x_full = (
        lam[0] * part_full_pos[tri[0]]
        + lam[1] * part_full_pos[tri[1]]
        + lam[2] * part_full_pos[tri[2]]
    )

    if not np.all(np.isfinite(x_full)):
        return None

    return x_full


def _make_default_reference_tensor(V_tgt, F_tgt, eps=1e-12):
    VN_tgt = vertex_normals(V_tgt, F_tgt)
    T_ref = np.zeros((V_tgt.shape[0], 3), dtype=np.float64)
    for i in range(V_tgt.shape[0]):
        T_ref[i] = arbitrary_tangent(VN_tgt[i], eps=eps)
    D_ref = np.ones((V_tgt.shape[0], 2), dtype=np.float64)
    return VN_tgt, T_ref, D_ref


def _soft_column_to_point(V_part, Pyx_part_to_full, i_full, topk=20, eps=1e-12):
    w = _normalize_soft_column_weights(Pyx_part_to_full[:, i_full], topk=topk, eps=eps)
    if w is None:
        return None, None
    x = w @ np.asarray(V_part, dtype=np.float64)
    return x, w

def _precompute_part_vertex_full_pos_from_pyx(
    Pyx_part_to_full,
    V_full,
    topk_row_soft=20,
    eps=1e-12,
):
    """
    For each part vertex j:
        part_full_pos[j] = sum_i Pyx[j, i] * V_full[i]

    This builds a soft full-space position for every part vertex.
    Pyx_part_to_full should have shape (N_part, N_full).
    """
    Pyx = np.asarray(Pyx_part_to_full, np.float64)
    V_full = np.asarray(V_full, np.float64)

    N_part, N_full = Pyx.shape

    part_full_pos = np.full((N_part, 3), np.nan, dtype=np.float64)
    part_full_valid = np.zeros(N_part, dtype=bool)

    for j in range(N_part):
        w = _normalize_soft_column_weights(
            Pyx[j, :],
            topk=topk_row_soft,
            eps=eps,
        )
        if w is None:
            continue

        part_full_pos[j] = w @ V_full
        part_full_valid[j] = True

    return part_full_pos, part_full_valid


def _precompute_part_vertex_full_pos_from_p2p(
    p2p_part_to_full,
    V_full,
):
    """
    Hard version:
        part_full_pos[j] = V_full[p2p[j]]
    """
    p2p = np.asarray(p2p_part_to_full, np.int64).reshape(-1)
    V_full = np.asarray(V_full, np.float64)

    part_full_pos = np.full((p2p.shape[0], 3), np.nan, dtype=np.float64)
    part_full_valid = (p2p >= 0) & (p2p < V_full.shape[0])

    part_full_pos[part_full_valid] = V_full[p2p[part_full_valid]]

    return part_full_pos, part_full_valid

def _precompute_full_as_part_cloud_from_pyx(
    Pyx_part_to_full,
    V_part,
    topk_col=50,
    eps=1e-12,
):
    """
    Build inverse cloud from Pyx columns.

    For each full vertex i:
        full_as_part[i] = sum_j Pyx[j, i] * V_part[j]

    This gives a point cloud in PART space whose point i represents
    where full vertex i lands on the part surface.
    """
    Pyx = np.asarray(Pyx_part_to_full, np.float64)
    V_part = np.asarray(V_part, np.float64)

    N_part, N_full = Pyx.shape

    full_as_part = np.full((N_full, 3), np.nan, dtype=np.float64)
    valid_full = np.zeros(N_full, dtype=bool)

    for i_full in range(N_full):
        x_part, w = _soft_column_to_point(
            V_part,
            Pyx,
            i_full,
            topk=topk_col,
            eps=eps,
        )

        if x_part is None:
            continue

        full_as_part[i_full] = x_part
        valid_full[i_full] = True

    valid_full_ids = np.where(valid_full)[0].astype(np.int64)

    if valid_full_ids.size == 0:
        raise RuntimeError("No valid full_as_part points from Pyx columns.")

    inv_tree = cKDTree(full_as_part[valid_full_ids])

    return full_as_part, valid_full, valid_full_ids, inv_tree


def _map_part_point_to_full_by_inverse_cloud(
    x_part,
    inv_tree,
    valid_full_ids,
    V_full,
    k_inv=8,
    inv_power=2.0,
    eps=1e-12,
):
    """
    Map a point in PART space back to FULL space by KNN search
    in the column-built inverse cloud.

    x_part -> nearest full_as_part points -> weighted average of V_full.
    """
    x_part = np.asarray(x_part, np.float64).reshape(3)
    V_full = np.asarray(V_full, np.float64)

    k = min(int(k_inv), len(valid_full_ids))
    if k <= 0:
        return None

    d, nn = inv_tree.query(x_part, k=k)

    d = np.atleast_1d(d).astype(np.float64)
    nn = np.atleast_1d(nn).astype(np.int64)

    full_ids = valid_full_ids[nn]

    # exact or near-exact inverse
    if d[0] < 1e-10:
        return V_full[full_ids[0]].copy()

    w = 1.0 / np.maximum(d, eps) ** float(inv_power)
    w = w / (np.sum(w) + eps)

    x_full = w @ V_full[full_ids]

    if not np.all(np.isfinite(x_full)):
        return None

    return x_full

def _transfer_tensor_to_full_templatepull_vertex(
    V_full, F_full,
    V_part, F_part,
    T_part_v, D_part_v,
    part_v2tid=None,
    full_v2tid=None,
    T_full_ref_v=None,
    D_full_ref_v=None,
    k_triplet=24,
    consistency_mul=2.0,
    fallback_mode="soft",
    seg_tau=0.5,
    anchor_src_ids=None,
    anchor_tgt_ids=None,
    use_p2p_donor=False,
    return_debug=False,
    eps=1e-12,
):
    V_full = np.asarray(V_full, float)
    F_full = np.asarray(F_full, np.int64)
    V_part = np.asarray(V_part, float)
    F_part = np.asarray(F_part, np.int64)

    T_part_v = np.asarray(T_part_v, float)
    D_part_v = np.asarray(D_part_v, float)
    T_part_v = _normalize_rows(T_part_v, eps=eps)
    #D_part_v = _renorm_trace2(D_part_v, eps=eps)

    use_anchor_pairs = (anchor_src_ids is not None) and (anchor_tgt_ids is not None)

    if (not use_anchor_pairs) and (part_v2tid is None or full_v2tid is None):
        raise ValueError(
            "template_pull_vertex mode requires either "
            "{part_v2tid, full_v2tid} or {anchor_src_ids, anchor_tgt_ids}."
        )
    VN_part = vertex_normals(V_part, F_part)

    if T_full_ref_v is None or D_full_ref_v is None:
        VN_full, T_full_ref_v_default, D_full_ref_v_default = _make_default_reference_tensor(
            V_full, F_full, eps=eps
        )
        if T_full_ref_v is None:
            T_full_ref_v = T_full_ref_v_default
        if D_full_ref_v is None:
            D_full_ref_v = D_full_ref_v_default
    else:
        VN_full = vertex_normals(V_full, F_full)

    T_full_ref_v = np.asarray(T_full_ref_v, float)
    D_full_ref_v = np.asarray(D_full_ref_v, float)
    T_full_ref_v = _normalize_rows(T_full_ref_v, eps=eps)
    #D_full_ref_v = _renorm_trace2(D_full_ref_v, eps=eps)

    if use_anchor_pairs:
        X_src, X_tgt, src_ids, tgt_ids = build_anchor_cloud_from_corr_pairs(
            V_src=V_part,
            V_tgt=V_full,
            corr_src=anchor_src_ids,
            corr_tgt=anchor_tgt_ids,
            dedup_tgt=True,
        )
        common_tids = np.arange(X_src.shape[0], dtype=np.int64)
        transfer_name = "anchor_pair_barycentric"

    else:
        X_src, X_tgt, common_tids, src_ids, tgt_ids = build_common_template_embeddings_from_v2tid(
            V_src=V_part,
            src_v2tid=part_v2tid,
            V_tgt=V_full,
            tgt_v2tid=full_v2tid,
        )
        transfer_name = "template_pull_vertex"

    if use_p2p_donor:
        # DIRECT p2p: each full vertex takes its single exact anchor (no barycentric).
        # anchor k sits at full vertex tgt_ids[k]; invert to full-vertex -> anchor index.
        tgt_to_anchor = -np.ones(V_full.shape[0], dtype=np.int64)
        tgt_to_anchor[np.asarray(tgt_ids, np.int64)] = np.arange(len(tgt_ids), dtype=np.int64)
        donor_triplet, donor_lam, donor_point_src, donor_point_tgt, donor_valid = \
            build_p2p_donor_query_from_vertices(
                V_tgt=V_full,
                X_tgt=X_tgt,
                X_src=X_src,
                tgt_to_anchor=tgt_to_anchor,
                eps=eps,
            )
        transfer_name = transfer_name + "_p2p"
    else:
        donor_triplet, donor_lam, donor_point_src, donor_point_tgt, donor_valid = \
            build_target_driven_donor_query_from_vertices(
                V_tgt=V_full,
                X_tgt=X_tgt,
                X_src=X_src,
                k_triplet=k_triplet,
                eps=eps,
            )

    dir_batch = batch_build_target_directions_from_local_correspondence_vertex( donor_valid=donor_valid, donor_triplet=donor_triplet, donor_point_src=donor_point_src, donor_point_tgt=donor_point_tgt, X_src=X_src, X_tgt=X_tgt, V_src=V_part, F_src=F_part, VN_src=VN_part, T_src_v=T_part_v, D_src_v=D_part_v, V_tgt=V_full, F_tgt=F_full, VN_tgt=VN_full, T_tgt_self_v=T_full_ref_v, D_tgt_self_v=D_full_ref_v, eps=eps, debug_trace=False,
    debug_trace_n=5,)

    N_full = V_full.shape[0]
    mel_full = mean_edge_length(V_full, F_full)
    tau_conf = float(max(consistency_mul * mel_full, eps))

    T_out = T_full_ref_v.copy()
    D_out = np.ones((N_full, 2), dtype=np.float64)

    alpha_v = np.zeros(N_full, dtype=np.float64)
    consistency_error_v = np.full(N_full, np.inf, dtype=np.float64)
    transport_ok = np.zeros(N_full, dtype=bool)
    anchor_error_v = np.full(N_full, np.inf, dtype=np.float64)
    segment_error_v = np.full(N_full, np.inf, dtype=np.float64)
    alpha_A_v = np.zeros(N_full, dtype=np.float64)
    alpha_seg_v = np.zeros(N_full, dtype=np.float64)

    for i in range(N_full):
        if not dir_batch["ok"][i]:
            continue

        v_tilde = dir_batch["v_tilde"][i].copy()
        u1 = _project_tangent(v_tilde, VN_full[i], eps=eps)
        if u1 is None:
            continue

        u1 = u1 / (np.linalg.norm(u1) + eps)

        D_src_point = dir_batch["D_src_point"][i].copy()

        e_A = float(dir_batch["consistency_error"][i])
        A_src = dir_batch["A_src_m"][i]
        B_src = dir_batch["B_src_m"][i]
        A_tgt = dir_batch["A_tgt_m"][i]
        B_tgt = dir_batch["B_tgt_m"][i]

        if (
            np.all(np.isfinite(A_src)) and np.all(np.isfinite(B_src))
            and np.all(np.isfinite(A_tgt)) and np.all(np.isfinite(B_tgt))
        ):
            src_len = float(np.linalg.norm(B_src - A_src))
            tgt_len = float(np.linalg.norm(B_tgt - A_tgt))
            ratio = tgt_len / max(src_len, eps)
            e_seg = abs(float(np.log(max(ratio, eps))))
        else:
            e_seg = float("inf")

        if fallback_mode == "soft":
            alpha_A = float(np.exp(- (e_A / tau_conf) ** 2)) if np.isfinite(e_A) else 0.0
            alpha_seg = float(np.exp(- (e_seg / seg_tau) ** 2)) if np.isfinite(e_seg) else 0.0
            alpha_i = alpha_A
            # fallback to isotropic tensor; then direction no longer matters
            D_mix = alpha_i * D_src_point + (1.0 - alpha_i) * np.array([1.0, 1.0], dtype=np.float64)

        elif fallback_mode == "hard":
            anchor_ok = np.isfinite(e_A) and e_A <= tau_conf
            segment_ok = np.isfinite(e_seg) and e_seg <= seg_tau

            alpha_A = 1.0 if anchor_ok else 0.0
            alpha_seg = 1.0 if segment_ok else 0.0
            alpha_i = 1.0 if anchor_ok  else 0.0

            if alpha_i > 0.5:
                D_mix = D_src_point
            else:
                D_mix = np.array([1.0, 1.0], dtype=np.float64)

        elif fallback_mode == "none":
            alpha_A = 1.0
            alpha_seg = 1.0
            alpha_i = 1.0
            D_mix = D_src_point

        else:
            raise ValueError(f"Unknown fallback_mode: {fallback_mode}")
        #D_mix = _renorm_trace2(D_mix[None, :], eps=eps)[0]

        T_out[i] = u1
        D_out[i] = D_mix

        alpha_v[i] = alpha_i
        anchor_error_v[i] = e_A
        segment_error_v[i] = e_seg
        alpha_A_v[i] = alpha_A
        alpha_seg_v[i] = alpha_seg

        # store combined normalized error for reporting
        consistency_error_v[i] = max(
            e_A / max(tau_conf, eps),
            e_seg / max(seg_tau, eps),
        )

        transport_ok[i] = True
    finite_A = np.isfinite(anchor_error_v)
    finite_seg = np.isfinite(segment_error_v)

    mean_anchor_err = float(np.mean(anchor_error_v[finite_A])) if np.any(finite_A) else float("inf")
    mean_segment_err = float(np.mean(segment_error_v[finite_seg])) if np.any(finite_seg) else float("inf")
    med_segment_err = float(np.median(segment_error_v[finite_seg])) if np.any(finite_seg) else float("inf")
    p90_segment_err = float(np.percentile(segment_error_v[finite_seg], 90)) if np.any(finite_seg) else float("inf")
    p95_segment_err = float(np.percentile(segment_error_v[finite_seg], 95)) if np.any(finite_seg) else float("inf")
    finite_mask = np.isfinite(consistency_error_v)
    mean_consistency_err = (
        float(np.mean(consistency_error_v[finite_mask]))
        if np.any(finite_mask) else float("inf")
    )
    median_consistency_err = (
        float(np.median(consistency_error_v[finite_mask]))
        if np.any(finite_mask) else float("inf")
    )

    stats = { "transfer_mode": transfer_name, "n_anchor": int(X_src.shape[0]), "n_common_template": int(len(common_tids)), "donor_valid_ratio": float(np.mean(donor_valid)), "missing_ratio": float(np.mean(~donor_valid)), "tau_consistency": float(tau_conf), "mean_consistency_err": mean_consistency_err, "median_consistency_err": median_consistency_err, "mean_consistency_weight": float(np.mean(alpha_v)), "reliable_ratio": float(np.mean(alpha_v > 0.5)), "transport_ok_ratio": float(np.mean(transport_ok)), "n_transport_ok": int(transport_ok.sum()), "n_vertices_full": int(N_full), "n_valid_face_map": int(np.count_nonzero(dir_batch["ok"])),"mean_anchor_err": mean_anchor_err, "mean_segment_err": mean_segment_err, "median_segment_err": med_segment_err, "p90_segment_err": p90_segment_err, "p95_segment_err": p95_segment_err, "mean_alpha_A": float(np.mean(alpha_A_v)), "mean_alpha_seg": float(np.mean(alpha_seg_v)), "seg_tau": float(seg_tau), }
    """
    print(
        f"[transfer:{transfer_name}] "
        f"n_anchor={stats['n_anchor']}, "
        f"donor_valid={stats['donor_valid_ratio']:.3f}, "
        f"transport_ok={stats['n_transport_ok']}/{N_full}, "
        f"anchor_err={stats['mean_anchor_err']:.6g}, "
        f"seg_err mean/med/p90/p95="
        f"{stats['mean_segment_err']:.3f}/"
        f"{stats['median_segment_err']:.3f}/"
        f"{stats['p90_segment_err']:.3f}/"
        f"{stats['p95_segment_err']:.3f}, "
        f"alpha_A={stats['mean_alpha_A']:.3f}, "
        f"alpha_seg={stats['mean_alpha_seg']:.3f}, "
        f"mean_w={stats['mean_consistency_weight']:.3f}, "
        f"reliable={stats['reliable_ratio']:.3f}, "
        f"tau={stats['tau_consistency']:.6g}, "
        f"seg_tau={stats['seg_tau']:.3f}"
    )
    """
    if return_debug:
        debug = { "X_src": X_src.astype(np.float32), "X_tgt": X_tgt.astype(np.float32), "common_tids": common_tids.astype(np.int32), "src_ids": src_ids.astype(np.int32), "tgt_ids": tgt_ids.astype(np.int32), "donor_triplet": donor_triplet.astype(np.int32), "donor_lam": donor_lam.astype(np.float32), "donor_point_src": donor_point_src.astype(np.float32), "donor_point_tgt": donor_point_tgt.astype(np.float32), "donor_valid": donor_valid, "A_src_m": dir_batch["A_src_m"].astype(np.float32), "faceA_src": dir_batch["faceA_src"].astype(np.int32), "B_src_m": dir_batch["B_src_m"].astype(np.float32), "B_src_t": dir_batch["B_src_t"].astype(np.float32), "v_src_t": dir_batch["v_src_t"].astype(np.float32), "v_u_y": dir_batch["v_u_y"].astype(np.float32), "D_src_point": dir_batch["D_src_point"].astype(np.float32), "A_tgt_raw": dir_batch["A_tgt_raw"].astype(np.float32), "B_tgt_raw": dir_batch["B_tgt_raw"].astype(np.float32), "A_tgt_m": dir_batch["A_tgt_m"].astype(np.float32), "faceA_tgt": dir_batch["faceA_tgt"].astype(np.int32), "B_tgt_m": dir_batch["B_tgt_m"].astype(np.float32), "B_tgt_t": dir_batch["B_tgt_t"].astype(np.float32), "v_tgt_obs": dir_batch["v_tgt_obs"].astype(np.float32), "v_tilde_u_x": dir_batch["v_tilde_u_x"].astype(np.float32), "v_tilde": dir_batch["v_tilde"].astype(np.float32), "alpha_v": alpha_v.astype(np.float32), "consistency_error_v": consistency_error_v.astype(np.float32), "transport_ok": transport_ok,"anchor_error_v": anchor_error_v.astype(np.float32), "segment_error_v": segment_error_v.astype(np.float32), "alpha_A_v": alpha_A_v.astype(np.float32), "alpha_seg_v": alpha_seg_v.astype(np.float32), }
        return (
            T_out.astype(np.float32),
            D_out.astype(np.float32),
            alpha_v.astype(np.float32),
            consistency_error_v.astype(np.float32),
            stats,
            debug,
        )

    return (
        T_out.astype(np.float32),
        D_out.astype(np.float32),
        alpha_v.astype(np.float32),
        consistency_error_v.astype(np.float32),
        stats,
    )

def _transfer_tensor_to_full_wormhole_soft_linear(
    V_full, F_full,
    V_part, F_part,
    T_part_v, D_part_v,
    Pyx_part_to_full,
    p2p_part_to_full=None,
    T_full_ref_v=None,
    D_full_ref_v=None,
    topk_soft=20,
    topk_row_soft=20,
    consistency_mul=2.0,
    seg_tau=0.5,
    fallback_mode="soft",
    pyx_map_mode="row_soft",   # "row_soft", "hard_p2p", or "col_inverse_cloud"
    k_inv=8,
    inv_power=2.0,
    return_debug=False,
    eps=1e-12,
):
    """
    Wormhole Pyx-based tensor transfer using the same A/B segment transfer idea.

    For each FULL vertex i:
      1. Use Pyx column i to get soft source anchor A_src on PART.
      2. Project A_src to PART surface.
      3. Evaluate PART tensor at A_src.
      4. Build B_src = A_src + step * u_src, then project B_src to PART.
      5. Map both A_src and B_src to FULL using a PART->FULL embedding:
           - row_soft: each part vertex is mapped by Pyx row weights
           - hard_p2p: each part vertex is mapped by p2p_part_to_full
      6. Project mapped A/B to FULL surface.
      7. Use A_tgt -> B_tgt as transferred direction.
      8. Use A-anchor round-trip consistency only as fallback confidence.

    ``seg_tau`` is accepted for legacy call compatibility but is ignored.
    A/B lengths are used only to reject degenerate direction constructions;
    they are not required to agree across differently shaped surfaces.
    """

    V_full = np.asarray(V_full, float)
    F_full = np.asarray(F_full, np.int64)
    V_part = np.asarray(V_part, float)
    F_part = np.asarray(F_part, np.int64)

    T_part_v = _normalize_rows(np.asarray(T_part_v, float), eps=eps)
    D_part_v = np.asarray(D_part_v, float)

    Pyx_part_to_full = np.asarray(Pyx_part_to_full, np.float64)
    if Pyx_part_to_full.shape != (V_part.shape[0], V_full.shape[0]):
        raise ValueError(
            f"Pyx_part_to_full shape mismatch: got {Pyx_part_to_full.shape}, "
            f"expected {(V_part.shape[0], V_full.shape[0])}"
        )

    VN_part = vertex_normals(V_part, F_part)
    VN_full = vertex_normals(V_full, F_full)

    N_full = V_full.shape[0]

    if T_full_ref_v is None:
        T_full_ref_v = np.zeros((N_full, 3), dtype=np.float64)
        for i in range(N_full):
            T_full_ref_v[i] = arbitrary_tangent(VN_full[i], eps=eps)
    else:
        T_full_ref_v = _normalize_rows(np.asarray(T_full_ref_v, float), eps=eps)

    if D_full_ref_v is None:
        D_full_ref_v = np.ones((N_full, 2), dtype=np.float64)
    else:
        D_full_ref_v = np.asarray(D_full_ref_v, float)

    # ------------------------------------------------------------
    # Build PART vertex -> FULL position embedding.
    # This replaces old normal-based parallel transport.
    # ------------------------------------------------------------
    part_full_pos = None
    part_full_valid = None

    full_as_part = None
    full_as_part_valid = None
    inv_full_ids = None
    inv_tree = None

    if pyx_map_mode == "row_soft":
        part_full_pos, part_full_valid = _precompute_part_vertex_full_pos_from_pyx(
            Pyx_part_to_full,
            V_full,
            topk_row_soft=topk_row_soft,
            eps=eps,
        )
        map_valid_ratio = float(np.mean(part_full_valid))

    elif pyx_map_mode == "hard_p2p":
        if p2p_part_to_full is None:
            raise ValueError("pyx_map_mode='hard_p2p' requires p2p_part_to_full.")

        part_full_pos, part_full_valid = _precompute_part_vertex_full_pos_from_p2p(
            p2p_part_to_full,
            V_full,
        )
        map_valid_ratio = float(np.mean(part_full_valid))

    elif pyx_map_mode == "col_inverse_cloud":
        full_as_part, full_as_part_valid, inv_full_ids, inv_tree = (
            _precompute_full_as_part_cloud_from_pyx(
                Pyx_part_to_full,
                V_part,
                topk_col=topk_soft,
                eps=eps,
            )
        )
        map_valid_ratio = float(np.mean(full_as_part_valid))

    else:
        raise ValueError(f"Unknown pyx_map_mode: {pyx_map_mode}")

    mel_full = mean_edge_length(V_full, F_full)
    mel_part = mean_edge_length(V_part, F_part)

    tau_conf = float(max(consistency_mul * mel_full, eps))
    face_step_part = precompute_face_mean_edge_lengths(V_part, F_part)

    # ------------------------------------------------------------
    # Output init
    # ------------------------------------------------------------
    T_out = np.ones((N_full, 3), dtype=np.float64)          
    for i in range(N_full):
        t = _project_tangent(T_out[i], VN_full[i], eps=eps)
        if t is None:
            t = arbitrary_tangent(VN_full[i], eps=eps)
        T_out[i] = t / (np.linalg.norm(t) + eps)
    D_out = np.ones((N_full, 2), dtype=np.float64)

    alpha_v = np.zeros(N_full, dtype=np.float64)
    alpha_A_v = np.zeros(N_full, dtype=np.float64)

    anchor_error_v = np.full(N_full, np.inf, dtype=np.float64)
    consistency_error_v = np.full(N_full, np.inf, dtype=np.float64)
    transport_ok = np.zeros(N_full, dtype=bool)

    debug_A_src = np.full((N_full, 3), np.nan, dtype=np.float64)
    debug_B_src = np.full((N_full, 3), np.nan, dtype=np.float64)
    debug_A_tgt = np.full((N_full, 3), np.nan, dtype=np.float64)
    debug_B_tgt = np.full((N_full, 3), np.nan, dtype=np.float64)

    # ============================================================
    # 1) Batch build A_src_raw from Pyx columns
    # ============================================================
    A_src_raw_all, A_valid = _soft_columns_to_points(
        V_src=V_part,
        W_src_to_tgt=Pyx_part_to_full,
        topk=topk_soft,
        eps=eps,
    )

    idx_A = np.where(A_valid)[0]
    if idx_A.size == 0:
        raise RuntimeError("No valid A_src from Pyx columns.")

    # Batch project all A_src_raw to PART
    A_src_m_sub, faceA_sub = project_points_to_mesh_with_face_batch(
        A_src_raw_all[idx_A],
        V_part,
        F_part,
    )

    A_src_m_all = np.full((N_full, 3), np.nan, dtype=np.float64)
    faceA_all = -np.ones(N_full, dtype=np.int64)

    A_src_m_all[idx_A] = A_src_m_sub
    faceA_all[idx_A] = faceA_sub

    # ============================================================
    # 2) Evaluate source tensor and build B_src guesses
    # ============================================================
    B_src_guess_all = np.full((N_full, 3), np.nan, dtype=np.float64)
    D_src_point_all = np.full((N_full, 2), np.nan, dtype=np.float64)

    for i in idx_A:
        faceA_src = int(faceA_all[i])
        if faceA_src < 0:
            continue

        A_src_m = A_src_m_all[i]
        if not np.all(np.isfinite(A_src_m)):
            continue

        n_src, u1_src, _, D_src_point = evaluate_vertex_tensor_at_projected_point(
            x_proj=A_src_m,
            face_id=faceA_src,
            V=V_part,
            F=F_part,
            VN=VN_part,
            T_v=T_part_v,
            D_v=D_part_v,
            eps=eps,
        )

        step = float(face_step_part[faceA_src])
        if step < eps:
            step = mel_part

        B_src_guess_all[i] = A_src_m + step * u1_src
        D_src_point_all[i] = D_src_point

    # ============================================================
    # 3) Batch project B_src guesses to PART
    # ============================================================
    idx_B = np.where(np.isfinite(B_src_guess_all[:, 0]))[0]

    B_src_m_all = np.full((N_full, 3), np.nan, dtype=np.float64)
    faceB_all = -np.ones(N_full, dtype=np.int64)

    if idx_B.size > 0:
        B_src_m_sub, faceB_sub = project_points_to_mesh_with_face_batch(
            B_src_guess_all[idx_B],
            V_part,
            F_part,
        )

        B_src_m_all[idx_B] = B_src_m_sub
        faceB_all[idx_B] = faceB_sub

    # ============================================================
    # 4) Map A_src / B_src to FULL raw positions
    # ============================================================
    A_tgt_raw_all = np.full((N_full, 3), np.nan, dtype=np.float64)
    B_tgt_raw_all = np.full((N_full, 3), np.nan, dtype=np.float64)

    idx_map = np.where(
        np.isfinite(A_src_m_all[:, 0])
        & np.isfinite(B_src_m_all[:, 0])
        & (faceA_all >= 0)
        & (faceB_all >= 0)
    )[0]

    for i in idx_map:
        A_src_m = A_src_m_all[i]
        B_src_m = B_src_m_all[i]
        faceA_src = int(faceA_all[i])
        faceB_src = int(faceB_all[i])

        if pyx_map_mode in ("row_soft", "hard_p2p"):
            A_tgt_raw = _map_part_surface_point_to_full_linear(
                A_src_m,
                faceA_src,
                V_part,
                F_part,
                part_full_pos,
                part_full_valid,
                eps=eps,
            )

            B_tgt_raw = _map_part_surface_point_to_full_linear(
                B_src_m,
                faceB_src,
                V_part,
                F_part,
                part_full_pos,
                part_full_valid,
                eps=eps,
            )

        elif pyx_map_mode == "col_inverse_cloud":
            A_tgt_raw = _map_part_point_to_full_by_inverse_cloud(
                A_src_m,
                inv_tree,
                inv_full_ids,
                V_full,
                k_inv=k_inv,
                inv_power=inv_power,
                eps=eps,
            )

            B_tgt_raw = _map_part_point_to_full_by_inverse_cloud(
                B_src_m,
                inv_tree,
                inv_full_ids,
                V_full,
                k_inv=k_inv,
                inv_power=inv_power,
                eps=eps,
            )

        else:
            raise ValueError(f"Unknown pyx_map_mode: {pyx_map_mode}")

        if A_tgt_raw is None or B_tgt_raw is None:
            continue

        A_tgt_raw_all[i] = A_tgt_raw
        B_tgt_raw_all[i] = B_tgt_raw

    # ============================================================
    # 5) Batch project mapped A/B to FULL mesh
    # ============================================================
    idx_tgt = np.where(
        np.isfinite(A_tgt_raw_all[:, 0])
        & np.isfinite(B_tgt_raw_all[:, 0])
    )[0]

    A_tgt_m_all = np.full((N_full, 3), np.nan, dtype=np.float64)
    B_tgt_m_all = np.full((N_full, 3), np.nan, dtype=np.float64)

    if idx_tgt.size > 0:
        A_tgt_m_sub, _ = project_points_to_mesh_with_face_batch(
            A_tgt_raw_all[idx_tgt],
            V_full,
            F_full,
        )

        B_tgt_m_sub, _ = project_points_to_mesh_with_face_batch(
            B_tgt_raw_all[idx_tgt],
            V_full,
            F_full,
        )

        A_tgt_m_all[idx_tgt] = A_tgt_m_sub
        B_tgt_m_all[idx_tgt] = B_tgt_m_sub

    # ============================================================
    # 6) Final tensor transfer + fallback
    # ============================================================
    for i in idx_tgt:
        A_src_m = A_src_m_all[i]
        B_src_m = B_src_m_all[i]
        A_tgt_m = A_tgt_m_all[i]
        B_tgt_m = B_tgt_m_all[i]

        if (
            not np.all(np.isfinite(A_src_m))
            or not np.all(np.isfinite(B_src_m))
            or not np.all(np.isfinite(A_tgt_m))
            or not np.all(np.isfinite(B_tgt_m))
        ):
            continue

        src_len = float(np.linalg.norm(B_src_m - A_src_m))
        tgt_vec = B_tgt_m - A_tgt_m
        tgt_len = float(np.linalg.norm(tgt_vec))

        if src_len < eps or tgt_len < eps:
            continue

        u1_tgt = _project_tangent(tgt_vec, VN_full[i], eps=eps)
        if u1_tgt is None:
            continue
        u1_tgt = u1_tgt / (np.linalg.norm(u1_tgt) + eps)

        D_src_point = D_src_point_all[i]
        if not np.all(np.isfinite(D_src_point)):
            continue

        # A-anchor consistency: mapped A should return near target full vertex i
        e_A = float(np.linalg.norm(A_tgt_m - V_full[i]))

        if fallback_mode == "soft":
            alpha_A = float(np.exp(- (e_A / tau_conf) ** 2)) if np.isfinite(e_A) else 0.0
            alpha_i = alpha_A

            D_mix = (
                alpha_i * D_src_point
                + (1.0 - alpha_i) * np.array([1.0, 1.0], dtype=np.float64)
            )

        elif fallback_mode == "hard":
            anchor_ok = np.isfinite(e_A) and e_A <= tau_conf
            alpha_A = 1.0 if anchor_ok else 0.0
            alpha_i = 1.0 if anchor_ok else 0.0

            if alpha_i > 0.5:
                D_mix = D_src_point
            else:
                D_mix = np.array([1.0, 1.0], dtype=np.float64)

        elif fallback_mode == "none":
            alpha_A = 1.0
            alpha_i = 1.0
            D_mix = D_src_point

        else:
            raise ValueError(f"Unknown fallback_mode: {fallback_mode}")

        T_out[i] = u1_tgt
        D_out[i] = D_mix

        alpha_v[i] = alpha_i
        alpha_A_v[i] = alpha_A

        anchor_error_v[i] = e_A
        consistency_error_v[i] = e_A / max(tau_conf, eps)

        transport_ok[i] = True

        debug_A_src[i] = A_src_m
        debug_B_src[i] = B_src_m
        debug_A_tgt[i] = A_tgt_m
        debug_B_tgt[i] = B_tgt_m

    # ------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------
    finite_A = np.isfinite(anchor_error_v)
    finite_cons = np.isfinite(consistency_error_v)

    stats = {
        "transfer_mode": "wormhole_ab_soft_linear",
        "pyx_map_mode": pyx_map_mode,
        "topk_soft": int(topk_soft),
        "topk_row_soft": int(topk_row_soft),

        "tau_consistency": float(tau_conf),
        "consistency_metric": "anchor_roundtrip_only",
        "segment_consistency_used": False,

        "mean_anchor_err": float(np.mean(anchor_error_v[finite_A])) if np.any(finite_A) else float("inf"),
        "median_anchor_err": float(np.median(anchor_error_v[finite_A])) if np.any(finite_A) else float("inf"),

        "mean_consistency_err": float(np.mean(consistency_error_v[finite_cons])) if np.any(finite_cons) else float("inf"),
        "median_consistency_err": float(np.median(consistency_error_v[finite_cons])) if np.any(finite_cons) else float("inf"),

        "mean_consistency_weight": float(np.mean(alpha_v)),
        "reliable_ratio": float(np.mean(alpha_v > 0.5)),

        "mean_alpha_A": float(np.mean(alpha_A_v)),

        "transport_ok_ratio": float(np.mean(transport_ok)),
        "n_transport_ok": int(transport_ok.sum()),
        "n_vertices_full": int(N_full),

        "part_full_valid_ratio": float(map_valid_ratio),
        "k_inv": int(k_inv),
        "inv_power": float(inv_power),
        "A_valid_ratio": float(np.mean(A_valid)),
        "n_idx_A": int(idx_A.size),
        "n_idx_B": int(idx_B.size),
        "n_idx_map": int(idx_map.size),
        "n_idx_tgt": int(idx_tgt.size),
    }

    print(
        f"[transfer:wormhole-ab-soft] "
        f"map={stats['pyx_map_mode']}, "
        f"topk_col={stats['topk_soft']}, "
        f"topk_row={stats['topk_row_soft']}, "
        f"part_full_valid={stats['part_full_valid_ratio']:.3f}, "
        f"A_valid={stats['A_valid_ratio']:.3f}, "
        f"transport_ok={stats['n_transport_ok']}/{N_full}, "
        f"anchor_err={stats['mean_anchor_err']:.6g}, "
        f"map={stats['pyx_map_mode']}, "
        f"alpha_A={stats['mean_alpha_A']:.3f}, "
        f"mean_w={stats['mean_consistency_weight']:.3f}, "
        f"reliable={stats['reliable_ratio']:.3f}, "
        f"tau={stats['tau_consistency']:.6g}, consistency=anchor_only"
    )

    if return_debug:
        debug = {
            "A_src_m": debug_A_src.astype(np.float32),
            "B_src_m": debug_B_src.astype(np.float32),
            "A_tgt_m": debug_A_tgt.astype(np.float32),
            "B_tgt_m": debug_B_tgt.astype(np.float32),

            "alpha_v": alpha_v.astype(np.float32),
            "alpha_A_v": alpha_A_v.astype(np.float32),

            "anchor_error_v": anchor_error_v.astype(np.float32),
            "consistency_error_v": consistency_error_v.astype(np.float32),

            "transport_ok": transport_ok,
            "part_full_pos": part_full_pos.astype(np.float32),
            "part_full_valid": part_full_valid,
        }

        return (
            T_out.astype(np.float32),
            D_out.astype(np.float32),
            alpha_v.astype(np.float32),
            consistency_error_v.astype(np.float32),
            stats,
            debug,
        )

    return (
        T_out.astype(np.float32),
        D_out.astype(np.float32),
        alpha_v.astype(np.float32),
        consistency_error_v.astype(np.float32),
        stats,
    )

def build_anchor_cloud_from_corr_pairs(
    V_src,
    V_tgt,
    corr_src,
    corr_tgt,
    dedup_tgt=True,
):
    V_src = np.asarray(V_src, float)
    V_tgt = np.asarray(V_tgt, float)

    corr_src = np.asarray(corr_src, np.int64).reshape(-1)
    corr_tgt = np.asarray(corr_tgt, np.int64).reshape(-1)

    ok = (
        (corr_src >= 0) & (corr_src < V_src.shape[0]) &
        (corr_tgt >= 0) & (corr_tgt < V_tgt.shape[0])
    )

    src_ids = corr_src[ok]
    tgt_ids = corr_tgt[ok]

    if src_ids.size == 0:
        raise RuntimeError("No valid anchor pairs.")

    # remove exact duplicate pairs first
    pairs = np.unique(np.stack([src_ids, tgt_ids], axis=1), axis=0)
    src_ids = pairs[:, 0]
    tgt_ids = pairs[:, 1]

    if dedup_tgt:
        # If several part vertices map to the same full vertex,
        # keep the source with smallest ambient residual.
        resid = np.linalg.norm(V_src[src_ids] - V_tgt[tgt_ids], axis=1)
        order = np.argsort(resid)

        keep_src = []
        keep_tgt = []
        seen_tgt = set()

        for k in order:
            t = int(tgt_ids[k])
            if t in seen_tgt:
                continue
            seen_tgt.add(t)
            keep_src.append(int(src_ids[k]))
            keep_tgt.append(int(tgt_ids[k]))

        src_ids = np.asarray(keep_src, np.int64)
        tgt_ids = np.asarray(keep_tgt, np.int64)

    X_src = V_src[src_ids].copy()
    X_tgt = V_tgt[tgt_ids].copy()

    return X_src, X_tgt, src_ids, tgt_ids

# ====================================
# Full to Full transfer
# ===================================
def build_tid_to_vertex(v2tid: np.ndarray):
    """
    Build template-id -> mesh-vertex map from mesh vertex -> template id.
    If duplicate template ids exist, keep the first vertex.
    """
    v2tid = np.asarray(v2tid, np.int64).reshape(-1)

    valid = v2tid >= 0
    if not np.any(valid):
        raise RuntimeError("No valid template ids in v2tid.")

    n_tid = int(v2tid[valid].max()) + 1
    tid2v = -np.ones(n_tid, dtype=np.int64)

    for v, tid in enumerate(v2tid):
        if tid < 0:
            continue
        if tid2v[tid] < 0:
            tid2v[tid] = int(v)

    return tid2v


def k_ring_vertices(adj, start, k=1):
    """
    Return vertices within k-ring around start, excluding start.
    """
    visited = {int(start)}
    frontier = {int(start)}

    for _ in range(int(k)):
        new_frontier = set()
        for v in frontier:
            for nb in adj[v]:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    new_frontier.add(nb)
        frontier = new_frontier
        if len(frontier) == 0:
            break

    visited.remove(int(start))
    return np.array(sorted(visited), dtype=np.int64)


def choose_B_src_by_arrow_alignment(
    j_src,
    V_src,
    F_src,
    VN_src,
    T_src_v,
    src_v2tid,
    tgt_tid2v,
    adj_src,
    k_ring=2,
    eps=1e-12,
):
    """
    Choose B_src around A_src=j_src.

    We treat T_src_v as a line field, so the best edge can be aligned with
    either +T or -T. However, we also return the sign, so that the target-side
    arrow can be oriented consistently with the source direction.
    """
    n_src = VN_src[j_src]
    u_src = _project_tangent(T_src_v[j_src], n_src, eps=eps)
    if u_src is None:
        return -1, 1.0

    candidates = k_ring_vertices(adj_src, j_src, k=k_ring)
    if candidates.size == 0:
        return -1, 1.0

    best_q = -1
    best_score = -np.inf
    best_sign = 1.0

    for q in candidates:
        tid_q = int(src_v2tid[q])
        if tid_q < 0 or tid_q >= tgt_tid2v.shape[0]:
            continue

        q_tgt = int(tgt_tid2v[tid_q])
        if q_tgt < 0:
            continue

        e = V_src[q] - V_src[j_src]
        e = _project_tangent(e, n_src, eps=eps)
        if e is None:
            continue

        raw = float(np.dot(e, u_src))

        # line-field matching: use absolute alignment
        score = abs(raw)

        # but keep orientation sign for target-side arrow
        sign = 1.0 if raw >= 0.0 else -1.0

        if score > best_score:
            best_score = score
            best_q = int(q)
            best_sign = sign

    return best_q, best_sign

def transfer_tensor_full2full_continuous_template(
    V_src,
    F_src,
    V_tgt,
    F_tgt,
    T_src_v,
    D_src_v,
    src_tmpl2mesh,
    tgt_tmpl2mesh,
    T_tgt_ref_v=None,
    D_tgt_ref_v=None,
    k_triplet=24,
    consistency_mul=2.0,
    fallback_mode="none",   # "none", "soft", "hard"
    eps=1e-12,
):
    V_src = np.asarray(V_src, np.float64)
    F_src = np.asarray(F_src, np.int64)
    V_tgt = np.asarray(V_tgt, np.float64)
    F_tgt = np.asarray(F_tgt, np.int64)

    T_src_v = np.asarray(T_src_v, np.float64)
    D_src_v = np.asarray(D_src_v, np.float64)
    T_src_v = _normalize_rows(T_src_v, eps=eps)

    N_tgt = V_tgt.shape[0]

    VN_src = vertex_normals(V_src, F_src)
    VN_tgt = vertex_normals(V_tgt, F_tgt)

    if T_tgt_ref_v is None:
        T_tgt_ref_v = np.zeros((N_tgt, 3), dtype=np.float64)
        for i in range(N_tgt):
            T_tgt_ref_v[i] = arbitrary_tangent(VN_tgt[i], eps=eps)
    else:
        T_tgt_ref_v = np.asarray(T_tgt_ref_v, np.float64)
        T_tgt_ref_v = _normalize_rows(T_tgt_ref_v, eps=eps)

    if D_tgt_ref_v is None:
        D_tgt_ref_v = np.ones((N_tgt, 2), dtype=np.float64)
    else:
        D_tgt_ref_v = np.asarray(D_tgt_ref_v, np.float64)

    src_tmpl2mesh = np.asarray(src_tmpl2mesh, np.int64).reshape(-1)
    tgt_tmpl2mesh = np.asarray(tgt_tmpl2mesh, np.int64).reshape(-1)

    # ------------------------------------------------------------
    # Build canonical sample clouds:
    # X_src[tau] and X_tgt[tau] are corresponding template samples.
    # ------------------------------------------------------------
    valid_src = (src_tmpl2mesh >= 0) & (src_tmpl2mesh < V_src.shape[0])
    valid_tgt = (tgt_tmpl2mesh >= 0) & (tgt_tmpl2mesh < V_tgt.shape[0])
    common_valid = valid_src & valid_tgt

    if np.count_nonzero(common_valid) < 3:
        raise RuntimeError("Too few valid common template samples for continuous full2full transfer.")

    common_tids = np.where(common_valid)[0].astype(np.int64)

    X_src = V_src[src_tmpl2mesh[common_tids]].copy()
    X_tgt = V_tgt[tgt_tmpl2mesh[common_tids]].copy()

    # ------------------------------------------------------------
    # Target-first donor construction.
    # donor_triplet indices are into X_src / X_tgt arrays,
    # not original template ids.
    # ------------------------------------------------------------
    donor_triplet, donor_lam, donor_point_src, donor_point_tgt, donor_valid = (
        build_target_driven_donor_query_from_vertices(
            V_tgt=V_tgt,
            X_tgt=X_tgt,
            X_src=X_src,
            k_triplet=k_triplet,
            eps=eps,
        )
    )

    # ------------------------------------------------------------
    # Continuous A/B transfer.
    # ------------------------------------------------------------
    dir_batch = batch_build_target_directions_from_local_correspondence_vertex(
        donor_valid=donor_valid,
        donor_triplet=donor_triplet,
        donor_point_src=donor_point_src,
        donor_point_tgt=donor_point_tgt,
        X_src=X_src,
        X_tgt=X_tgt,
        V_src=V_src,
        F_src=F_src,
        VN_src=VN_src,
        T_src_v=T_src_v,
        D_src_v=D_src_v,
        V_tgt=V_tgt,
        F_tgt=F_tgt,
        VN_tgt=VN_tgt,
        T_tgt_self_v=T_tgt_ref_v,
        D_tgt_self_v=D_tgt_ref_v,
        eps=eps,
        debug_trace=False,
        debug_trace_n=5,
    )

    mel_tgt = mean_edge_length(V_tgt, F_tgt)
    tau_conf = float(max(consistency_mul * mel_tgt, eps))

    T_out = T_tgt_ref_v.copy()
    D_out = np.ones((N_tgt, 2), dtype=np.float64)

    alpha_v = np.zeros(N_tgt, dtype=np.float64)
    consistency_error_v = np.full(N_tgt, np.inf, dtype=np.float64)
    valid = np.zeros(N_tgt, dtype=bool)

    for i in range(N_tgt):
        if not dir_batch["ok"][i]:
            continue

        v_tilde = dir_batch["v_tilde"][i].copy()
        u1 = _project_tangent(v_tilde, VN_tgt[i], eps=eps)
        if u1 is None:
            continue

        u1 = u1 / (np.linalg.norm(u1) + eps)

        D_src_point = dir_batch["D_src_point"][i].copy()

        e_A = float(dir_batch["consistency_error"][i])

        # A/B segment consistency
        A_src = dir_batch["A_src_m"][i]
        B_src = dir_batch["B_src_m"][i]
        A_tgt = dir_batch["A_tgt_m"][i]
        B_tgt = dir_batch["B_tgt_m"][i]

        if (
            np.all(np.isfinite(A_src)) and np.all(np.isfinite(B_src))
            and np.all(np.isfinite(A_tgt)) and np.all(np.isfinite(B_tgt))
        ):
            src_len = float(np.linalg.norm(B_src - A_src))
            tgt_len = float(np.linalg.norm(B_tgt - A_tgt))
            ratio = tgt_len / max(src_len, eps)
            e_seg = abs(float(np.log(max(ratio, eps))))
        else:
            e_seg = float("inf")

        # combine both consistency errors
        if fallback_mode == "soft":
            alpha_A = float(np.exp(- (e_A / tau_conf) ** 2)) if np.isfinite(e_A) else 0.0
            #alpha_seg = float(np.exp(- (e_seg / 0.6) ** 2)) if np.isfinite(e_seg) else 0.0
            alpha_i = alpha_A

            D_mix = alpha_i * D_src_point + (1.0 - alpha_i) * np.array([1.0, 1.0], dtype=np.float64)

        elif fallback_mode == "hard":
            #ratio_ok = np.isfinite(e_seg) and e_seg <= np.log(2.0)
            anchor_ok = np.isfinite(e_A) and e_A <= tau_conf
            alpha_i = 1.0 if anchor_ok  else 0.0

            if alpha_i > 0.5:
                D_mix = D_src_point
            else:
                D_mix = np.array([1.0, 1.0], dtype=np.float64)

        elif fallback_mode == "none":
            alpha_i = 1.0
            D_mix = D_src_point

        else:
            raise ValueError(f"Unknown fallback_mode: {fallback_mode}")

        T_out[i] = u1
        D_out[i] = D_mix
        alpha_v[i] = alpha_i
        consistency_error_v[i] = e_A
        valid[i] = True

    finite = np.isfinite(consistency_error_v)
    mean_consistency_err = (
        float(np.mean(consistency_error_v[finite]))
        if np.any(finite) else float("inf")
    )
    median_consistency_err = (
        float(np.median(consistency_error_v[finite]))
        if np.any(finite) else float("inf")
    )

    # ------------------------------------------------------------
    # Debug quality: continuous A/B segment length.
    # ------------------------------------------------------------
    ok_len = (
        valid
        & np.isfinite(dir_batch["A_src_m"][:, 0])
        & np.isfinite(dir_batch["B_src_m"][:, 0])
        & np.isfinite(dir_batch["A_tgt_m"][:, 0])
        & np.isfinite(dir_batch["B_tgt_m"][:, 0])
    )

    if np.any(ok_len):
        src_len = np.linalg.norm(
            dir_batch["B_src_m"][ok_len] - dir_batch["A_src_m"][ok_len],
            axis=1,
        )
        tgt_len = np.linalg.norm(
            dir_batch["B_tgt_m"][ok_len] - dir_batch["A_tgt_m"][ok_len],
            axis=1,
        )
        ratio = tgt_len / np.maximum(src_len, eps)

        print(
            "[transfer:full2full-continuous dbg] "
            f"src_len mean/p50/p95={src_len.mean():.4e}/"
            f"{np.percentile(src_len,50):.4e}/"
            f"{np.percentile(src_len,95):.4e}, "
            f"tgt_len mean/p50/p95={tgt_len.mean():.4e}/"
            f"{np.percentile(tgt_len,50):.4e}/"
            f"{np.percentile(tgt_len,95):.4e}, "
            f"ratio mean/p50/p95/max={ratio.mean():.4f}/"
            f"{np.percentile(ratio,50):.4f}/"
            f"{np.percentile(ratio,95):.4f}/"
            f"{ratio.max():.4f}"
        )

    stats = {
        "mode": "full2full_continuous_template",
        "fallback_mode": fallback_mode,
        "n_common_template": int(common_tids.size),
        "donor_valid_ratio": float(np.mean(donor_valid)),
        "valid_ratio": float(np.mean(valid)),
        "n_valid": int(valid.sum()),
        "n_tgt": int(N_tgt),
        "tau_consistency": float(tau_conf),
        "mean_consistency_err": mean_consistency_err,
        "median_consistency_err": median_consistency_err,
        "mean_consistency_weight": float(np.mean(alpha_v)),
        "reliable_ratio": float(np.mean(alpha_v > 0.5)),
    }

    print(
        f"[transfer:full2full-continuous] "
        f"mode={fallback_mode}, "
        f"common_tids={stats['n_common_template']}, "
        f"donor_valid={stats['donor_valid_ratio']:.3f}, "
        f"valid={stats['n_valid']}/{N_tgt} ({stats['valid_ratio']:.3f}), "
        f"mean_err={stats['mean_consistency_err']:.6g}, "
        f"median_err={stats['median_consistency_err']:.6g}, "
        f"mean_w={stats['mean_consistency_weight']:.6g}, "
        f"reliable={stats['reliable_ratio']:.3f}, "
        f"tau={stats['tau_consistency']:.6g}"
    )

    # donor_A_src / donor_B_src are not discrete donors in the continuous case.
    # For compatibility, return nearest source query vertex if available.
    donor_A_src = -np.ones(N_tgt, dtype=np.int64)
    donor_B_src = -np.ones(N_tgt, dtype=np.int64)

    if "src_query_vid" in dir_batch:
        donor_A_src = np.asarray(dir_batch["src_query_vid"], np.int64).copy()

    return (
        T_out.astype(np.float32),
        D_out.astype(np.float32),
        valid,
        donor_A_src,
        donor_B_src,
        stats,
    )

def transfer_tensor_full2full_arrow_template(
    V_src,
    F_src,
    V_tgt,
    F_tgt,
    T_src_v,
    D_src_v,
    src_v2tid,
    tgt_v2tid,
    T_tgt_ref_v=None,
    D_tgt_ref_v=None,
    k_ring=2,
    fallback_mode="soft",       # "soft", "hard", "none"
    consistency_mul=2.0,
    fallback_to_normal=True,
    fallback_to_nearest=True,
    eps=1e-12,
):

    V_src = np.asarray(V_src, np.float64)
    F_src = np.asarray(F_src, np.int64)
    V_tgt = np.asarray(V_tgt, np.float64)
    F_tgt = np.asarray(F_tgt, np.int64)

    T_src_v = np.asarray(T_src_v, np.float64)
    D_src_v = np.asarray(D_src_v, np.float64)

    src_v2tid = np.asarray(src_v2tid, np.int64).reshape(-1)
    tgt_v2tid = np.asarray(tgt_v2tid, np.int64).reshape(-1)

    N_tgt = V_tgt.shape[0]

    VN_src = vertex_normals(V_src, F_src)
    VN_tgt = vertex_normals(V_tgt, F_tgt)

    T_src_v = _normalize_rows(T_src_v, eps=eps)

    # ------------------------------------------------------------
    # Target fallback/reference tensor
    # ------------------------------------------------------------
    if T_tgt_ref_v is None:
        T_tgt_ref_v = np.zeros((N_tgt, 3), dtype=np.float64)
        for i in range(N_tgt):
            T_tgt_ref_v[i] = arbitrary_tangent(VN_tgt[i], eps=eps)
    else:
        T_tgt_ref_v = np.asarray(T_tgt_ref_v, np.float64)
        T_tgt_ref_v = _normalize_rows(T_tgt_ref_v, eps=eps)

    if D_tgt_ref_v is None:
        D_tgt_ref_v = np.ones((N_tgt, 2), dtype=np.float64)
    else:
        D_tgt_ref_v = np.asarray(D_tgt_ref_v, np.float64)

    src_tid2v = build_tid_to_vertex(src_v2tid)
    tgt_tid2v = build_tid_to_vertex(tgt_v2tid)
    adj_src = _build_vertex_adjacency(F_src, V_src.shape[0])
    adj_tgt = _build_vertex_adjacency(F_tgt, V_tgt.shape[0])

    # Initialize by reference tensor, not zeros.
    # This means missing/unreliable vertices naturally fall back to self tensor.
    T_out = T_tgt_ref_v.copy()
    D_out = D_tgt_ref_v.copy()

    valid = np.zeros(N_tgt, dtype=bool)
    donor_A_src = -np.ones(N_tgt, dtype=np.int64)
    donor_B_src = -np.ones(N_tgt, dtype=np.int64)

    # Debug arrays for full2full discrete arrow transfer
    donor_B_tgt = -np.ones(N_tgt, dtype=np.int64)
    arrow_sign_v = np.full(N_tgt, np.nan, dtype=np.float64)

    src_arrow_len_v = np.full(N_tgt, np.nan, dtype=np.float64)
    tgt_arrow_len_v = np.full(N_tgt, np.nan, dtype=np.float64)
    src_arrow_angle_v = np.full(N_tgt, np.nan, dtype=np.float64)

    tgt_B_is_1ring_v = np.zeros(N_tgt, dtype=bool)
    tgt_B_is_kring_v = np.zeros(N_tgt, dtype=bool)

    tid_A_v = -np.ones(N_tgt, dtype=np.int64)
    tid_B_v = -np.ones(N_tgt, dtype=np.int64)

    alpha_v = np.zeros(N_tgt, dtype=np.float64)
    consistency_error_v = np.full(N_tgt, np.inf, dtype=np.float64)

    A_tgt_back_v = -np.ones(N_tgt, dtype=np.int64)
    B_tgt_back_v = -np.ones(N_tgt, dtype=np.int64)

    fallback_normal_count = 0
    fallback_ref_count = 0
    transfer_ok_count = 0

    mel_tgt = mean_edge_length(V_tgt, F_tgt)
    tau_conf = float(max(consistency_mul * mel_tgt, eps))

    for i_tgt in range(N_tgt):
        tid_A = int(tgt_v2tid[i_tgt])
        if tid_A < 0 or tid_A >= src_tid2v.shape[0]:
            continue

        j_src = int(src_tid2v[tid_A])
        if j_src < 0:
            continue
        # ------------------------------------------------------------
        # A-anchor round-trip consistency:
        # target i_tgt -> source j_src -> target i_back
        # ------------------------------------------------------------
        tid_back_A = int(src_v2tid[j_src])
        i_back_A = int(tgt_tid2v[tid_back_A]) if 0 <= tid_back_A < tgt_tid2v.shape[0] else -1
        A_tgt_back_v[i_tgt] = i_back_A

        if i_back_A >= 0:
            e_round_A = float(np.linalg.norm(V_tgt[i_back_A] - V_tgt[i_tgt]))
        else:
            e_round_A = float("inf")
            
        q_src, arrow_sign = choose_B_src_by_arrow_alignment(
            j_src=j_src,
            V_src=V_src,
            F_src=F_src,
            VN_src=VN_src,
            T_src_v=T_src_v,
            src_v2tid=src_v2tid,
            tgt_tid2v=tgt_tid2v,
            adj_src=adj_src,
            k_ring=k_ring,
            eps=eps,
        )

        tid_A_v[i_tgt] = tid_A
        arrow_sign_v[i_tgt] = arrow_sign

        # Debug source-side arrow: is A_src -> B_src really aligned with T_src[j_src]?
        if q_src >= 0:
            e_src_raw = V_src[q_src] - V_src[j_src]
            e_src = _project_tangent(e_src_raw, VN_src[j_src], eps=eps)
            u_src_dbg = _project_tangent(T_src_v[j_src], VN_src[j_src], eps=eps)

            src_arrow_len_v[i_tgt] = float(np.linalg.norm(e_src_raw))

            if e_src is not None and u_src_dbg is not None:
                # sign-invariant angle, because T is a line field
                src_arrow_angle_v[i_tgt] = np.degrees(
                    _line_angle_rad(e_src, u_src_dbg, eps=eps)
                )

        u_trans = None

        if q_src >= 0:
            tid_B = int(src_v2tid[q_src])
            tid_B_v[i_tgt] = tid_B

            q_tgt = int(tgt_tid2v[tid_B]) if 0 <= tid_B < tgt_tid2v.shape[0] else -1
            donor_B_tgt[i_tgt] = q_tgt

            if q_tgt >= 0:
                e_tgt = V_tgt[q_tgt] - V_tgt[i_tgt]
                tgt_arrow_len_v[i_tgt] = float(np.linalg.norm(e_tgt))

                # Is q_tgt still local around i_tgt on the target mesh?
                tgt_B_is_1ring_v[i_tgt] = q_tgt in set(map(int, adj_tgt[i_tgt]))

                kring_tgt = k_ring_vertices(adj_tgt, i_tgt, k=k_ring)
                if kring_tgt.size > 0:
                    tgt_B_is_kring_v[i_tgt] = bool(np.any(kring_tgt == q_tgt))

                u_trans = _project_tangent(arrow_sign * e_tgt, VN_tgt[i_tgt], eps=eps)

        # Only if A/B arrow completely fails, use normal transport fallback.
        # This is not the main quality fallback.
        if u_trans is None and fallback_to_normal:
            u_src = _project_tangent(T_src_v[j_src], VN_src[j_src], eps=eps)
            if u_src is not None:
                u_trans = _parallel_transport_chord(
                    u_src,
                    VN_src[j_src],
                    VN_tgt[i_tgt],
                    eps=eps,
                )
                fallback_normal_count += 1

        if u_trans is None:
            continue

        u_trans = u_trans / (np.linalg.norm(u_trans) + eps)
        D_trans = D_src_v[j_src].copy()

        # ------------------------------------------------------------
        # Round-trip consistency, same idea as part-to-full:
        # target anchor -> source anchor -> target anchor.
        # ------------------------------------------------------------
        e_i = e_round_A
        consistency_error_v[i_tgt] = e_i

        if fallback_mode == "soft":
            alpha_i = float(np.exp(- (e_i / tau_conf) ** 2)) if np.isfinite(e_i) else 0.0

            # Keep transferred direction, but attenuate anisotropy.
            # When alpha -> 0, D becomes isotropic, so direction no longer matters.
            T_mix = u_trans
            D_mix = alpha_i * D_trans + (1.0 - alpha_i) * np.array([1.0, 1.0], dtype=np.float64)

            if alpha_i > 0.5:
                transfer_ok_count += 1
            else:
                fallback_ref_count += 1

        elif fallback_mode == "hard":
            alpha_i = 1.0 if (np.isfinite(e_i) and e_i <= tau_conf) else 0.0

            T_mix = u_trans
            if alpha_i > 0.5:
                D_mix = D_trans
                transfer_ok_count += 1
            else:
                D_mix = np.array([1.0, 1.0], dtype=np.float64)
                fallback_ref_count += 1

        elif fallback_mode == "none":
            alpha_i = 1.0
            T_mix = u_trans
            D_mix = D_trans
            transfer_ok_count += 1

        else:
            raise ValueError(f"Unknown fallback_mode: {fallback_mode}")

        T_out[i_tgt] = T_mix
        D_out[i_tgt] = D_mix

        alpha_v[i_tgt] = alpha_i
        valid[i_tgt] = True
        donor_A_src[i_tgt] = j_src
        donor_B_src[i_tgt] = q_src

    # Fill missing target vertices by nearest valid target vertex.
    # Since T_out/D_out already start from reference tensor, this is optional.
    # If enabled, missing vertices inherit nearby transferred/ref-mixed tensor.
    if fallback_to_nearest and np.any(~valid) and np.any(valid):
        valid_ids = np.where(valid)[0]
        missing_ids = np.where(~valid)[0]

        tree = cKDTree(V_tgt[valid_ids])
        _, nn = tree.query(V_tgt[missing_ids], k=1)
        ref_ids = valid_ids[nn]

        T_out[missing_ids] = T_out[ref_ids]
        D_out[missing_ids] = D_out[ref_ids]
        donor_A_src[missing_ids] = donor_A_src[ref_ids]
        donor_B_src[missing_ids] = donor_B_src[ref_ids]
        donor_B_tgt[missing_ids] = donor_B_tgt[ref_ids]

        alpha_v[missing_ids] = alpha_v[ref_ids]
        consistency_error_v[missing_ids] = consistency_error_v[ref_ids]
        A_tgt_back_v[missing_ids] = A_tgt_back_v[ref_ids]
        B_tgt_back_v[missing_ids] = B_tgt_back_v[ref_ids]

        arrow_sign_v[missing_ids] = arrow_sign_v[ref_ids]
        src_arrow_len_v[missing_ids] = src_arrow_len_v[ref_ids]
        tgt_arrow_len_v[missing_ids] = tgt_arrow_len_v[ref_ids]
        src_arrow_angle_v[missing_ids] = src_arrow_angle_v[ref_ids]
        tgt_B_is_1ring_v[missing_ids] = tgt_B_is_1ring_v[ref_ids]
        tgt_B_is_kring_v[missing_ids] = tgt_B_is_kring_v[ref_ids]
        tid_A_v[missing_ids] = tid_A_v[ref_ids]
        tid_B_v[missing_ids] = tid_B_v[ref_ids]

        valid[missing_ids] = True

    # Final target tangent cleanup
    for i in range(N_tgt):
        u = _project_tangent(T_out[i], VN_tgt[i], eps=eps)
        if u is None:
            u = arbitrary_tangent(VN_tgt[i], eps=eps)
        T_out[i] = u / (np.linalg.norm(u) + eps)

    """"
    # ------------------------------------------------------------
    # Debug: discrete A/B arrow transfer quality
    # ------------------------------------------------------------
    ok_dbg = valid & (donor_A_src >= 0) & (donor_B_src >= 0) & (donor_B_tgt >= 0)

    if np.any(ok_dbg):
        print("\n[dbg:full2full discrete arrow]")

        print(
            f"[dbg:donor] "
            f"ok={ok_dbg.sum()}/{N_tgt}, "
            f"unique_A_src={np.unique(donor_A_src[ok_dbg]).size}/{ok_dbg.sum()}, "
            f"unique_B_src={np.unique(donor_B_src[ok_dbg]).size}/{ok_dbg.sum()}, "
            f"unique_B_tgt={np.unique(donor_B_tgt[ok_dbg]).size}/{ok_dbg.sum()}"
        )

        print(
            f"[dbg:donor collapse] "
            f"AeqB_src={np.mean(donor_A_src[ok_dbg] == donor_B_src[ok_dbg]):.6f}, "
            f"iEqB_tgt={np.mean(np.where(ok_dbg)[0] == donor_B_tgt[ok_dbg]):.6f}"
        )

        _dbg_stats("source arrow len |q_src-j_src|", src_arrow_len_v[ok_dbg])
        _dbg_stats("target arrow len |q_tgt-i_tgt|", tgt_arrow_len_v[ok_dbg])
        _dbg_stats("source arrow angle to T_src[j_src] deg", src_arrow_angle_v[ok_dbg])

        ratio_len = tgt_arrow_len_v[ok_dbg] / np.maximum(src_arrow_len_v[ok_dbg], eps)
        _dbg_stats("target/source arrow length ratio", ratio_len)

        print(
            f"[dbg:target locality] "
            f"B_tgt_in_1ring={np.mean(tgt_B_is_1ring_v[ok_dbg]):.6f}, "
            f"B_tgt_in_{k_ring}ring={np.mean(tgt_B_is_kring_v[ok_dbg]):.6f}"
        )

        print(
            f"[dbg:arrow sign] "
            f"positive={np.mean(arrow_sign_v[ok_dbg] > 0):.6f}, "
            f"negative={np.mean(arrow_sign_v[ok_dbg] < 0):.6f}"
        )

        # Optional: inspect worst target/source length ratios
        ratio_abs = np.abs(np.log(np.maximum(ratio_len, eps)))
        worst_local = np.argsort(-ratio_abs)[:10]
        worst_ids = np.where(ok_dbg)[0][worst_local]

        print("[dbg:worst length-ratio vertices]")
        for ii in worst_ids:
            print(
                f"  i_tgt={int(ii)} "
                f"A_src={int(donor_A_src[ii])} "
                f"B_src={int(donor_B_src[ii])} "
                f"B_tgt={int(donor_B_tgt[ii])} "
                f"tid_A={int(tid_A_v[ii])} "
                f"tid_B={int(tid_B_v[ii])} "
                f"src_len={src_arrow_len_v[ii]:.4e} "
                f"tgt_len={tgt_arrow_len_v[ii]:.4e} "
                f"ratio={tgt_arrow_len_v[ii] / max(src_arrow_len_v[ii], eps):.4e} "
                f"src_ang={src_arrow_angle_v[ii]:.2f} "
                f"in_kring={bool(tgt_B_is_kring_v[ii])}"
            )
    else:
        print("[dbg:full2full discrete arrow] no valid debug arrows")
        
    finite = np.isfinite(consistency_error_v)

    if np.any(finite):
        cons = consistency_error_v[finite]
        cons_mean = float(np.mean(cons))
        cons_med = float(np.median(cons))
        cons_p90 = float(np.percentile(cons, 90))
        cons_p95 = float(np.percentile(cons, 95))
    else:
        cons_mean = cons_med = cons_p90 = cons_p95 = float("nan")
    stats = {
        "mode": "full2full_arrow_template_roundtrip_fallback",
        "fallback_mode": fallback_mode,
        "valid_ratio": float(np.mean(valid)),
        "n_valid": int(valid.sum()),
        "n_tgt": int(N_tgt),
        "n_unique_A_src": int(np.unique(donor_A_src[donor_A_src >= 0]).size),
        "n_unique_B_src": int(np.unique(donor_B_src[donor_B_src >= 0]).size),
        "fallback_normal_count": int(fallback_normal_count),
        "fallback_ref_count": int(fallback_ref_count),
        "transfer_ok_count": int(transfer_ok_count),
        "mean_consistency_weight": float(np.mean(alpha_v)),
        "reliable_ratio": float(np.mean(alpha_v > 0.5)),
        "tau_consistency": float(tau_conf),
        "mean_consistency_err": cons_mean,
        "median_consistency_err": cons_med,
        "consistency_p90": cons_p90,
        "consistency_p95": cons_p95,
        "n_unique_B_tgt": int(np.unique(donor_B_tgt[donor_B_tgt >= 0]).size),
        "dbg_target_B_in_1ring_ratio": float(np.mean(tgt_B_is_1ring_v[ok_dbg])) if np.any(ok_dbg) else float("nan"),
        "dbg_target_B_in_kring_ratio": float(np.mean(tgt_B_is_kring_v[ok_dbg])) if np.any(ok_dbg) else float("nan"),
        "dbg_src_arrow_angle_mean_deg": float(np.nanmean(src_arrow_angle_v[ok_dbg])) if np.any(ok_dbg) else float("nan"),
        "dbg_src_arrow_len_mean": float(np.nanmean(src_arrow_len_v[ok_dbg])) if np.any(ok_dbg) else float("nan"),
        "dbg_tgt_arrow_len_mean": float(np.nanmean(tgt_arrow_len_v[ok_dbg])) if np.any(ok_dbg) else float("nan"),
    }
    print(
        f"[transfer:full2full-arrow-roundtrip] "
        f"mode={fallback_mode}, "
        f"valid={stats['n_valid']}/{stats['n_tgt']} ({stats['valid_ratio']:.3f}), "
        f"unique_A={stats['n_unique_A_src']}, "
        f"unique_B={stats['n_unique_B_src']}, "
        f"mean_err={stats['mean_consistency_err']:.6g}, "
        f"median_err={stats['median_consistency_err']:.6g}, "
        f"mean_w={stats['mean_consistency_weight']:.3f}, "
        f"reliable={stats['reliable_ratio']:.3f}, "
        f"fallback_ref={stats['fallback_ref_count']}, "
        f"fallback_normal={stats['fallback_normal_count']}, "
        f"tau={stats['tau_consistency']:.6g}"
    )

    return (
        T_out.astype(np.float32),
        D_out.astype(np.float32),
        valid,
        donor_A_src,
        donor_B_src,
        stats,
    )
    """
# ============================================================
# Public API
# ============================================================
def transfer_tensor_to_full_fullcover(
    V_full, F_full,
    V_part, F_part,
    T_part_v, D_part_v,
    p2p_part_to_full=None,
    consistency_mul=2.0,
    return_debug=False,
    eps=1e-12,
    seg_tau=0.35,
    *,
    part_v2tid=None,
    full_v2tid=None,
    T_full_ref_v=None,
    D_full_ref_v=None,
    transfer_mode="template_pull_vertex",
    fallback_mode="soft",
    k_triplet=24,
    Pyx_part_to_full=None,
    topk_soft=20,
    topk_row_soft=20,
    pyx_map_mode="row_soft",
    anchor_src_ids=None,
    anchor_tgt_ids=None,
    k_inv=8,
    inv_power=2.0,
):

    if transfer_mode in ("template_pull_vertex", "anchor_pair_barycentric", "anchor_pair_p2p"):
        return _transfer_tensor_to_full_templatepull_vertex(
            V_full, F_full,
            V_part, F_part,
            T_part_v, D_part_v,
            part_v2tid=part_v2tid,
            full_v2tid=full_v2tid,
            T_full_ref_v=T_full_ref_v,
            D_full_ref_v=D_full_ref_v,
            k_triplet=k_triplet,
            consistency_mul=consistency_mul,
            seg_tau=seg_tau,
            fallback_mode=fallback_mode,
            anchor_src_ids=anchor_src_ids if transfer_mode in ("anchor_pair_barycentric", "anchor_pair_p2p") else None,
            anchor_tgt_ids=anchor_tgt_ids if transfer_mode in ("anchor_pair_barycentric", "anchor_pair_p2p") else None,
            use_p2p_donor=(transfer_mode == "anchor_pair_p2p"),
            return_debug=return_debug,
            eps=eps,
        )

    if transfer_mode == "wormhole_soft_linear":
        if Pyx_part_to_full is None:
            raise ValueError("wormhole_soft_linear mode requires Pyx_part_to_full.")

        return _transfer_tensor_to_full_wormhole_soft_linear(
            V_full, F_full,
            V_part, F_part,
            T_part_v, D_part_v,
            Pyx_part_to_full=Pyx_part_to_full,
            p2p_part_to_full=p2p_part_to_full,
            T_full_ref_v=T_full_ref_v,
            D_full_ref_v=D_full_ref_v,
            topk_soft=topk_soft,
            topk_row_soft=topk_row_soft,
            consistency_mul=consistency_mul,
            seg_tau=seg_tau,
            fallback_mode=fallback_mode,
            pyx_map_mode=pyx_map_mode,
            k_inv=k_inv,
            inv_power=inv_power,
            return_debug=return_debug,
            eps=eps,
        )

    raise ValueError(f"Unknown transfer_mode: {transfer_mode}")

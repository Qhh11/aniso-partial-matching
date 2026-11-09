# anislap/boundary.py

import heapq

import numpy as np
import igl
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve

from .assembly import vertex_field_to_face_tensor_spd


# ===================================================================
# Utilities
# ===================================================================

def _unwrap_igl_output(x):
    return x[0] if isinstance(x, tuple) else x


def _project_to_tangent(v: np.ndarray, n: np.ndarray, eps: float = 1e-12):
    v = v - np.dot(v, n) * n
    nv = np.linalg.norm(v)
    if nv < eps:
        return None
    return v / nv


def _normalize_rows(X: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    good = n[:, 0] > eps
    Y = X.copy()
    Y[good] /= n[good]
    return Y


# ===================================================================
# Boundary falloff
# ===================================================================

def boundary_falloff_sigmoid(
    d_v,
    radius_percentile: float = 40.0,
    center_frac: float = 0.06,
    width_frac: float = 0.02,
    clip: float = 12.0,
    eps: float = 1e-12,
):
    d_v = np.asarray(d_v, dtype=float)
    valid = d_v > eps
    if not np.any(valid):
        return np.zeros_like(d_v)

    r = float(np.percentile(d_v[valid], radius_percentile))
    r = max(r, eps)

    center = max(center_frac * r, eps)
    width = max(width_frac * r, eps)

    z = (d_v - center) / width
    z = np.clip(z, -clip, clip)

    w_v = 1.0 / (1.0 + np.exp(z))
    return w_v


# ===================================================================
# Graph / boundary helpers
# ===================================================================

def _get_boundary_edges_and_vertices(F: np.ndarray):
    F_int = F.astype(np.int32, copy=False)
    be_out = igl.boundary_facets(F_int)
    boundary_edges = be_out[0] if isinstance(be_out, tuple) else be_out
    boundary_edges = np.asarray(boundary_edges, dtype=int)

    if boundary_edges.size == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=int)

    boundary_vertices = np.unique(boundary_edges)
    return boundary_edges, boundary_vertices


def _get_unique_mesh_edges(F: np.ndarray):
    F_int = F.astype(np.int32, copy=False)
    edges = np.vstack([
        F_int[:, [0, 1]],
        F_int[:, [1, 2]],
        F_int[:, [2, 0]],
    ])
    edges_sorted = np.sort(edges, axis=1)
    _, uniq_idx = np.unique(edges_sorted, axis=0, return_index=True)
    unique_edges = edges[uniq_idx]
    return unique_edges.astype(int)


def _build_boundary_adjacency(boundary_edges: np.ndarray, N: int):
    adj = [[] for _ in range(N)]
    for i, j in boundary_edges:
        i = int(i)
        j = int(j)
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _extract_ordered_boundary_components(adj, boundary_vertices):
    N = len(adj)
    visited = np.zeros(N, dtype=bool)
    components = []
    closed_flags = []

    for seed in boundary_vertices:
        seed = int(seed)
        if visited[seed]:
            continue

        stack = [seed]
        comp = []
        visited[seed] = True

        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        comp = list(comp)
        deg = {u: len(adj[u]) for u in comp}
        endpoints = [u for u in comp if deg[u] == 1]

        # closed loop
        if len(endpoints) == 0 and all(deg[u] == 2 for u in comp):
            start = comp[0]
            ordered = [start]
            prev = -1
            curr = start

            for _ in range(len(comp) + 5):
                nbrs = adj[curr]
                nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
                if nxt == start:
                    break
                ordered.append(nxt)
                prev, curr = curr, nxt

            components.append(np.array(ordered, dtype=int))
            closed_flags.append(True)

        # open chain
        elif len(endpoints) == 2 and all(deg[u] in (1, 2) for u in comp):
            start = endpoints[0]
            ordered = [start]
            prev = -1
            curr = start

            for _ in range(len(comp) + 5):
                nbrs = [v for v in adj[curr] if v != prev]
                if len(nbrs) == 0:
                    break
                nxt = nbrs[0]
                ordered.append(nxt)
                prev, curr = curr, nxt
                if curr == endpoints[1]:
                    break

            components.append(np.array(ordered, dtype=int))
            closed_flags.append(False)

        # irregular fallback
        else:
            components.append(np.array(comp, dtype=int))
            closed_flags.append(None)

    return components, closed_flags


# ===================================================================
# Boundary distance + inward direction
# ===================================================================

def compute_boundary_distance_with_parents(V: np.ndarray, F: np.ndarray, eps: float = 1e-12):
    """
    Multi-source Dijkstra on the mesh graph.

    Returns:
        dist   : (N,) geodesic distance to the nearest boundary vertex
        parent : (N,) parent pointer toward the boundary along the shortest path tree
        source : (N,) boundary seed assigned to each vertex
    """
    V = np.asarray(V, dtype=float)
    F_int = F.astype(np.int32, copy=False)
    N = V.shape[0]

    boundary_edges, boundary_vertices = _get_boundary_edges_and_vertices(F_int)
    if boundary_edges.size == 0:
        print("[boundary] No boundary found; returning zeros.")
        return np.zeros(N), -np.ones(N, dtype=int), -np.ones(N, dtype=int)

    unique_edges = _get_unique_mesh_edges(F_int)
    p0 = V[unique_edges[:, 0]]
    p1 = V[unique_edges[:, 1]]
    lengths = np.linalg.norm(p0 - p1, axis=1)
    lengths = np.maximum(lengths, eps)

    adj = [[] for _ in range(N)]
    for (i, j), w in zip(unique_edges, lengths):
        i = int(i)
        j = int(j)
        adj[i].append((j, float(w)))
        adj[j].append((i, float(w)))

    INF = np.inf
    dist = np.full(N, INF)
    parent = -np.ones(N, dtype=int)
    source = -np.ones(N, dtype=int)
    visited = np.zeros(N, dtype=bool)
    heap = []

    for s in boundary_vertices:
        s = int(s)
        dist[s] = 0.0
        parent[s] = -1
        source[s] = s
        heapq.heappush(heap, (0.0, s))

    while heap:
        d_cur, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True

        for v, w in adj[u]:
            nd = d_cur + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                source[v] = source[u] if source[u] >= 0 else u
                heapq.heappush(heap, (nd, v))

    mask_finite = np.isfinite(dist)
    max_val = dist[mask_finite].max() if np.any(mask_finite) else 0.0
    dist[~mask_finite] = max_val
    dist = dist - dist.min()

    return dist, parent, source


def compute_shortest_path_direction_vertex(
    V: np.ndarray,
    parent: np.ndarray,
    n_v: np.ndarray,
    eps: float = 1e-12,
):
    """
    R_v points from v to parent[v] (toward the nearest boundary),
    then projected to the tangent plane.
    """
    V = np.asarray(V, dtype=float)
    N = V.shape[0]

    R_v = np.zeros((N, 3), dtype=float)

    valid = parent >= 0
    idx = np.where(valid)[0]
    R_v[idx] = V[parent[idx]] - V[idx]

    dot_rn = np.sum(R_v * n_v, axis=1, keepdims=True)
    R_v = R_v - dot_rn * n_v

    nr = np.linalg.norm(R_v, axis=1, keepdims=True)
    good = nr[:, 0] > eps
    R_v[good] /= nr[good]

    # robust fallback on boundary seeds / degenerate cases
    bad_idx = np.where(~good)[0]
    basis = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    for i in bad_idx:
        n = n_v[i]
        dots = np.abs(basis @ n)
        axis = basis[np.argmin(dots)]
        r = axis - np.dot(axis, n) * n
        nr = np.linalg.norm(r)
        if nr < eps:
            continue
        R_v[i] = r / nr

    return R_v

def compute_boundary_distance_fast_marching(
    V: np.ndarray,
    F: np.ndarray,
    eps: float = 1e-12,
):
    """
    Fast marching distance from mesh boundary.

    Unlike graph Dijkstra, this distance is computed on the triangle surface,
    so it can propagate across faces instead of only along mesh edges.

    Returns:
        dist              : (N,) distance to boundary
        boundary_vertices : boundary vertex indices
        boundary_edges    : boundary edge list
    """
    try:
        import potpourri3d as pp3d
    except ImportError as e:
        raise ImportError(
            "potpourri3d is required for fast marching distance. "
            "Install it with: pip install potpourri3d"
        ) from e

    V = np.asarray(V, dtype=float)
    F_int = F.astype(np.int32, copy=False)
    N = V.shape[0]

    boundary_edges, boundary_vertices = _get_boundary_edges_and_vertices(F_int)

    if boundary_edges.size == 0:
        print("[boundary] No boundary found; returning zeros.")
        return np.zeros(N), np.zeros(0, dtype=int), np.zeros((0, 2), dtype=int)

    # Use each boundary edge as a zero-distance curve source.
    # This is better than only using boundary vertices as sources.
    curves = [
        [(int(i), []), (int(j), [])]
        for i, j in boundary_edges
    ]

    solver = pp3d.MeshFastMarchingDistanceSolver(V, F_int)
    dist = np.asarray(solver.compute_distance(curves, sign=False), dtype=float)

    # Robust cleanup
    finite = np.isfinite(dist)
    if not np.any(finite):
        print("[boundary] Fast marching returned no finite distances; returning zeros.")
        return np.zeros(N), boundary_vertices, boundary_edges

    max_val = float(dist[finite].max())
    dist[~finite] = max_val
    dist = np.maximum(dist - dist.min(), 0.0)

    # Force exact zero on boundary vertices
    dist[boundary_vertices] = 0.0

    return dist, boundary_vertices, boundary_edges

def compute_distance_gradient_direction_vertex(
    V: np.ndarray,
    F: np.ndarray,
    d_v: np.ndarray,
    n_v: np.ndarray,
    eps: float = 1e-12,
):

    V = np.asarray(V, np.float64)
    Fi = np.asarray(F, np.int32)
    d_v = np.asarray(d_v, np.float64).reshape(-1)
    N = V.shape[0]

    edges = _get_unique_mesh_edges(Fi)
    adj = [[] for _ in range(N)]
    for i, j in edges:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))

    is_boundary = d_v <= eps

    R_v = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        if is_boundary[i]:
            # --- Boundary branch: R perpendicular to local boundary tangent ---
            bnbrs = [j for j in adj[i] if is_boundary[j]]
            t_local = None
            if len(bnbrs) >= 2:
                t_local = V[bnbrs[1]] - V[bnbrs[0]]
            elif len(bnbrs) == 1:
                t_local = V[bnbrs[0]] - V[i]

            if t_local is not None:
                t_local = t_local - np.dot(t_local, n_v[i]) * n_v[i]
                tn = np.linalg.norm(t_local)
                if tn > eps:
                    t_local = t_local / tn
                    R_v[i] = np.cross(n_v[i], t_local)
        else:
            # --- Interior branch: steepest descent along d_v ---
            best_j = -1
            best_drop = 0.0
            for j in adj[i]:
                drop = d_v[i] - d_v[j]
                if drop > best_drop:
                    best_drop = drop
                    best_j = j
            if best_j >= 0:
                R_v[i] = V[best_j] - V[i]

    # Project to tangent plane + normalize
    R_v = R_v - np.sum(R_v * n_v, axis=1, keepdims=True) * n_v
    nr = np.linalg.norm(R_v, axis=1, keepdims=True)
    good = nr[:, 0] > eps
    R_v[good] /= nr[good]

    basis = np.eye(3)
    for i in np.where(~good)[0]:
        n = n_v[i]
        axis = basis[np.argmin(np.abs(basis @ n))]
        r = axis - np.dot(axis, n) * n
        rn = np.linalg.norm(r)
        if rn > eps:
            R_v[i] = r / rn

    return R_v

def compute_distance_gradient_direction_vertex_fe(
    V: np.ndarray,
    F: np.ndarray,
    d_v: np.ndarray,
    n_v: np.ndarray,
    eps: float = 1e-12,
):

    V = np.asarray(V, dtype=float)
    F_int = F.astype(np.int32, copy=False)
    d_v = np.asarray(d_v, dtype=float)

    N = V.shape[0]
    grad_v = np.zeros((N, 3), dtype=float)
    area_v = np.zeros(N, dtype=float)

    for tri in F_int:
        i0, i1, i2 = map(int, tri)
        p0, p1, p2 = V[i0], V[i1], V[i2]
        d0, d1, d2 = d_v[i0], d_v[i1], d_v[i2]

        e1 = p1 - p0
        e2 = p2 - p0

        G = np.array([
            [np.dot(e1, e1), np.dot(e1, e2)],
            [np.dot(e2, e1), np.dot(e2, e2)],
        ], dtype=float)

        rhs = np.array([d1 - d0, d2 - d0], dtype=float)

        det = np.linalg.det(G)
        if abs(det) < eps:
            continue

        a, b = np.linalg.solve(G, rhs)
        grad_f = a * e1 + b * e2

        area = 0.5 * np.linalg.norm(np.cross(e1, e2))
        if area < eps:
            continue

        grad_v[i0] += area * grad_f
        grad_v[i1] += area * grad_f
        grad_v[i2] += area * grad_f

        area_v[i0] += area
        area_v[i1] += area
        area_v[i2] += area

    good_area = area_v > eps
    grad_v[good_area] /= area_v[good_area, None]

    # Direction toward boundary
    R_v = -grad_v

    # Project to tangent plane
    R_v = R_v - np.sum(R_v * n_v, axis=1, keepdims=True) * n_v

    nr = np.linalg.norm(R_v, axis=1, keepdims=True)
    good = nr[:, 0] > eps
    R_v[good] /= nr[good]

    # Fallback: use steepest neighbor descent in the distance field
    bad_idx = np.where(~good)[0]
    unique_edges = _get_unique_mesh_edges(F_int)

    adj = [[] for _ in range(N)]
    for i, j in unique_edges:
        i = int(i)
        j = int(j)
        adj[i].append(j)
        adj[j].append(i)

    basis = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    for i in bad_idx:
        # Prefer a neighbor with smaller distance
        best_j = None
        best_drop = 0.0

        for j in adj[i]:
            drop = d_v[i] - d_v[j]
            if drop > best_drop:
                best_drop = drop
                best_j = j

        if best_j is not None:
            r = V[best_j] - V[i]
            r = r - np.dot(r, n_v[i]) * n_v[i]
            nr_i = np.linalg.norm(r)
            if nr_i > eps:
                R_v[i] = r / nr_i
                continue

        # Final fallback: arbitrary tangent direction
        n = n_v[i]
        dots = np.abs(basis @ n)
        axis = basis[np.argmin(dots)]
        r = axis - np.dot(axis, n) * n
        nr_i = np.linalg.norm(r)
        if nr_i > eps:
            R_v[i] = r / nr_i

    return R_v

# ===================================================================
# Simple boundary tangent (NO corner logic)
# ===================================================================

def compute_boundary_tangent_simple(
    V: np.ndarray,
    F: np.ndarray,
    boundary_tangent_hops: int = 3,
    eps: float = 1e-12,
):
    """
    Clean boundary tangent estimate:
    - ordered boundary chains / loops
    - k-hop secant tangent
    - endpoint or irregular fallback to local edge direction
    No corner detection, no anchor weights, no strip override.
    """
    V = np.asarray(V, dtype=float)
    F_int = F.astype(np.int32, copy=False)
    N = V.shape[0]

    boundary_edges, boundary_vertices = _get_boundary_edges_and_vertices(F_int)
    if boundary_edges.size == 0:
        return (
            np.zeros((N, 3), dtype=float),
            np.zeros(0, dtype=int),
            np.zeros((0, 2), dtype=int),
        )

    n_v = _unwrap_igl_output(igl.per_vertex_normals(V, F_int))
    adj = _build_boundary_adjacency(boundary_edges, N)
    components, closed_flags = _extract_ordered_boundary_components(adj, boundary_vertices)

    T_bnd = np.zeros((N, 3), dtype=float)

    def longest_local_boundary_edge_tangent(i: int):
        n_i = n_v[i]
        best_t = None
        best_len = -1.0
        for j in adj[i]:
            d = V[j] - V[i]
            t = _project_to_tangent(d, n_i, eps=eps)
            if t is None:
                continue
            l = np.linalg.norm(d)
            if l > best_len:
                best_len = l
                best_t = t
        return best_t

    for comp, is_closed in zip(components, closed_flags):
        if len(comp) == 0:
            continue

        # irregular component: use local longest edge direction
        if is_closed is None:
            for i in comp:
                i = int(i)
                t = longest_local_boundary_edge_tangent(i)
                if t is not None:
                    T_bnd[i] = t
            continue

        L = len(comp)

        for pos, i in enumerate(comp):
            i = int(i)
            n_i = n_v[i]

            # local forward direction for consistent sign
            forward_t = None
            if is_closed or pos < L - 1:
                j = int(comp[(pos + 1) % L]) if is_closed else int(comp[pos + 1])
                forward_t = _project_to_tangent(V[j] - V[i], n_i, eps=eps)
            elif pos > 0:
                j = int(comp[pos - 1])
                forward_t = _project_to_tangent(V[i] - V[j], n_i, eps=eps)

            # k-hop secant
            t = None
            if is_closed:
                lo = int(comp[(pos - boundary_tangent_hops) % L])
                hi = int(comp[(pos + boundary_tangent_hops) % L])
                if lo != hi:
                    t = _project_to_tangent(V[hi] - V[lo], n_i, eps=eps)
            else:
                lo_idx = max(0, pos - boundary_tangent_hops)
                hi_idx = min(L - 1, pos + boundary_tangent_hops)
                if lo_idx != hi_idx:
                    lo = int(comp[lo_idx])
                    hi = int(comp[hi_idx])
                    t = _project_to_tangent(V[hi] - V[lo], n_i, eps=eps)

            # fallback
            if t is None:
                t = forward_t

            if t is None:
                t = longest_local_boundary_edge_tangent(i)

            if t is None:
                continue

            # orient consistently with local traversal
            if forward_t is not None and np.dot(t, forward_t) < 0:
                t = -t

            T_bnd[i] = t

    #print(f"[boundary] boundary vertices: {len(boundary_vertices)}")
    return T_bnd, boundary_vertices, boundary_edges


# ===================================================================
# Smoothing
# ===================================================================

def smooth_vector_field_vertex_dirichlet_boundary(
    V,
    F,
    X0_v,
    bmask,
    X_bnd,
    mu_smooth=3.0,
    eps=1e-12,
):
    """
    Simple Laplacian smoothing with hard boundary:
    boundary vertices are fixed to X_bnd, only interior is solved.
    """
    V = np.asarray(V, dtype=float)
    F_int = F.astype(np.int32, copy=False)
    N = V.shape[0]

    L_cot = _unwrap_igl_output(igl.cotmatrix(V, F_int))
    L = -L_cot  # PSD
    I = sp_diags(np.ones(N))
    A = I + mu_smooth * L

    bmask = np.asarray(bmask, dtype=bool)
    idx_b = np.where(bmask)[0]
    idx_i = np.where(~bmask)[0]

    X = np.asarray(X0_v, dtype=float).copy()
    X[idx_b] = X_bnd[idx_b]

    if len(idx_i) == 0:
        return X

    A_II = A[idx_i][:, idx_i]
    A_IB = A[idx_i][:, idx_b]

    for k in range(3):
        rhs = X0_v[idx_i, k] - A_IB @ X[idx_b, k]
        X[idx_i, k] = spsolve(A_II, rhs)

    return X


# ===================================================================
# MAIN
# ===================================================================

def compute_boundary_aware_tensor(
    V,
    F,
    beta_tangent: float = 3.0,
    gamma_normal: float = 3.0,
    radius_percentile: float = 40.0,
    sigmoid_center_frac: float = 0.03,
    sigmoid_width_frac: float = 0.005,
    normalize_tensor: bool = False,
    mu_smooth: float = 0.0,
    eigenvalue_mode: str = "sigmoid",
    return_on: str = "vertices",
    distance_method: str = "fast_marching",
    direction_method: str = "auto",
    boundary_tangent_hops: int = 3,
    corner_dot_threshold: float = -0.6,
    corner_anchor_weight: float = 0.25,
    endpoint_anchor_weight: float = 0.15,
    hard_boundary_strip_faces: bool = False,
):
    del corner_dot_threshold
    del corner_anchor_weight
    del endpoint_anchor_weight
    del hard_boundary_strip_faces

    if direction_method == "auto":
        direction_method = "parent" if distance_method in ("dijkstra", "graph") else "backtrack"

    if direction_method == "parent" and distance_method not in ("dijkstra", "graph"):
        raise ValueError("direction_method='parent' requires distance_method='dijkstra'.")
    if direction_method in ("fe_grad", "backtrack") and distance_method not in ("fast_marching", "fmm"):
        raise ValueError(f"direction_method={direction_method!r} expects distance_method='fast_marching'.")


    V = np.asarray(V, dtype=float)
    F_int = F.astype(np.int32, copy=False)
    N = V.shape[0]
    eps = 1e-12

    # 1) distance
    if distance_method in ("dijkstra", "graph"):
        d_v, parent, _ = compute_boundary_distance_with_parents(V, F_int, eps=eps)
        d_v = np.maximum(d_v, 0.0)

    elif distance_method in ("fast_marching", "fmm"):
        d_v, _, _ = compute_boundary_distance_fast_marching(V, F_int, eps=eps)
        d_v = np.maximum(d_v, 0.0)
        parent = None

    else:
        raise ValueError(
            f"Unknown distance_method={distance_method!r}. "
            "Use 'dijkstra' or 'fast_marching'."
        )

    # 2) vertex normals
    n_v = _unwrap_igl_output(igl.per_vertex_normals(V, F_int))

    # 3) boundary-normal direction
    if direction_method == "parent":
        R_v = compute_shortest_path_direction_vertex(V, parent, n_v, eps=eps)
    elif direction_method == "fe_grad":
        R_v = compute_distance_gradient_direction_vertex_fe(V, F_int, d_v, n_v, eps=eps) 
    elif direction_method == "backtrack":
        R_v = compute_distance_gradient_direction_vertex(V, F_int, d_v, n_v, eps=eps)   

    # 4) tangent by 90-degree rotation in tangent plane
    T_v = np.cross(n_v, R_v)
    T_v = _normalize_rows(T_v, eps=eps)

    # 5) overwrite boundary tangent with SIMPLE ordered-boundary estimate
    T_bnd, boundary_vertices, boundary_edges = compute_boundary_tangent_simple(
        V,
        F_int,
        boundary_tangent_hops=boundary_tangent_hops,
        eps=eps,
    )
    _ = boundary_edges  # kept for debug symmetry if needed later

    bmask = np.zeros(N, dtype=bool)
    bmask[boundary_vertices] = True

    anchor_mask = bmask & (np.linalg.norm(T_bnd, axis=1) > eps)
    T_v[anchor_mask] = T_bnd[anchor_mask]
    T_v = T_v - np.sum(T_v * n_v, axis=1, keepdims=True) * n_v
    T_v = _normalize_rows(T_v, eps=eps)

    # enforce (T, R, n) consistency
    R_v = np.cross(T_v, n_v)
    R_v = _normalize_rows(R_v, eps=eps)

    # 6) optional simple smoothing
    if mu_smooth > 0.0 and np.any(anchor_mask):
        T_anchor = T_v.copy()
        T_anchor[anchor_mask] = T_bnd[anchor_mask]

        T_v_sm = smooth_vector_field_vertex_dirichlet_boundary(
            V,
            F_int,
            X0_v=T_v,
            bmask=anchor_mask,
            X_bnd=T_anchor,
            mu_smooth=mu_smooth,
            eps=eps,
        )

        T_v_sm = T_v_sm - np.sum(T_v_sm * n_v, axis=1, keepdims=True) * n_v
        T_v = _normalize_rows(T_v_sm, eps=eps)

        R_v = np.cross(T_v, n_v)
        R_v = _normalize_rows(R_v, eps=eps)

    # 7) eigenvalues
    if eigenvalue_mode != "sigmoid":
        raise ValueError("Only 'sigmoid' mode is supported in the cleaned boundary version.")

    w_v = boundary_falloff_sigmoid(
        d_v,
        radius_percentile=radius_percentile,
        center_frac=sigmoid_center_frac,
        width_frac=sigmoid_width_frac,
        eps=eps,
    )

    lam_t_v = 1.0 + beta_tangent * w_v
    lam_n_v = 1.0 / (1.0 + gamma_normal * w_v)

    # !!! D convention this way to match D in anisotropic laplacian 
    D_v = np.zeros((N, 2), dtype=float)
    D_v[:, 1] = lam_t_v
    D_v[:, 0] = lam_n_v

    if normalize_tensor == "trace" or normalize_tensor is True:
        s = np.maximum(D_v.sum(axis=1, keepdims=True), eps)
        D_v = D_v * (2.0 / s)
    elif normalize_tensor == "det":
        det = np.maximum(D_v[:, 0] * D_v[:, 1], 1e-12)
        D_v = D_v * (1.0 / np.sqrt(det))[:, None]
    elif normalize_tensor in (False, "none", None):
        pass
    else:
        raise ValueError(f"Unknown normalize_tensor mode: {normalize_tensor!r}")

    #print(f"[boundary] anchors used: {int(np.count_nonzero(anchor_mask))}/{int(np.count_nonzero(bmask))}")
    #print(f"[boundary] lam_n range: [{lam_n_v.min():.6f}, {lam_n_v.max():.6f}], mean={lam_n_v.mean():.4f}")
    #print(f"[boundary] lam_t range: [{lam_t_v.min():.6f}, {lam_t_v.max():.6f}], mean={lam_t_v.mean():.4f}")

    if return_on == "vertices":
        return T_v, R_v, D_v, d_v

    if return_on != "faces":
        raise ValueError(f"[boundary] return_on must be 'faces' or 'vertices', got {return_on}")

    # face tensor via the same downstream converter
    Umin_f, Umax_f, D_f = vertex_field_to_face_tensor_spd(
        V,
        F_int,
        T_v=T_v,
        R_v=R_v,
        D_v=D_v,
        eps=eps,
    )

    if normalize_tensor:
        det = np.maximum(D_f[:, 0] * D_f[:, 1], 1e-12)
        scale = 1.0 / np.sqrt(det)
        D_f *= scale[:, None]

    print("[boundary] face tensor stats:")
    print("  D_f[:,0] range:", float(D_f[:, 0].min()), float(D_f[:, 0].max()))
    print("  D_f[:,1] range:", float(D_f[:, 1].min()), float(D_f[:, 1].max()))

    return Umin_f, Umax_f, D_f, d_v
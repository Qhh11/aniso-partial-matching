# anislap/geometry.py
import numpy as np
import igl


def compute_face_principal_directions_and_D(
    V,
    F,
    alpha_min: float = 10.0,
    alpha_max: float = 10.0,
    curvature_smoothing: int = 5,  
    use_sign: bool = False,
    normalize_tensor: bool = False,
    d_min: float = 0.01,
):

    F_int = F.astype(np.int32, copy=False)


    pd1, pd2, pv1, pv2 = igl.principal_curvature( V, F_int, curvature_smoothing, True)[:4]

    Umax_v = pd1  
    Umin_v = pd2 
    kmax_v = pv1 
    kmin_v = pv2  

    normals_v = igl.per_vertex_normals(V, F_int) 

    # Align signs to avoid ± flips canceling out
    # U3 = Umin × Umax
    U3 = np.cross(Umin_v, Umax_v)
    d = np.einsum("ij,ij->i", U3, normals_v)
    flip = d < 0
    if np.any(flip):
        Umax_v[flip] *= -1.0
        U3 = np.cross(Umin_v, Umax_v)

    def average_to_faces(X_v):
        return (
            X_v[F_int[:, 0]]
            + X_v[F_int[:, 1]]
            + X_v[F_int[:, 2]]
        ) / 3.0

    Umin_f = average_to_faces(Umin_v)
    Umax_f = average_to_faces(Umax_v)

    def normalize_rows(X, eps=1e-12):
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm = np.maximum(nrm, eps)
        return X / nrm

    Umin = normalize_rows(Umin_f)
    Umax = normalize_rows(Umax_f)

    kmin_f = (
        kmin_v[F_int[:, 0]]
        + kmin_v[F_int[:, 1]]
        + kmin_v[F_int[:, 2]]
    ) / 3.0
    kmax_f = (
        kmax_v[F_int[:, 0]]
        + kmax_v[F_int[:, 1]]
        + kmax_v[F_int[:, 2]]
    ) / 3.0

    def psi(k, alpha, use_sign=False):
        k_abs = np.abs(k)
        base = 1.0 / (1.0 + alpha * k_abs)
        if use_sign:
            return np.sign(k) * base
        else:
            return base

    D = np.zeros((F_int.shape[0], 2), dtype=float)

    D[:, 1] = psi(kmin_f, alpha_min, use_sign)
    D[:, 0] = psi(kmax_f, alpha_max, use_sign)

    if normalize_tensor:
        det = D[:, 0] * D[:, 1]
        det = np.maximum(det, 1e-12)
        scale = 1.0 / np.sqrt(det)
        D *= scale[:, None]

    #D_v = np.clip(D_v, 1e-3, 1e3)

    #print("D_min range:", D[:, 0].min(), D[:, 0].max())
    #print("D_max range:", D[:, 1].min(), D[:, 1].max())
    #print("mean D_min/D_max:", float(np.mean(D[:, 0] / D[:, 1])))

    return Umin, Umax, D

def compute_vertex_principal_directions_and_D(
    V,
    F,
    alpha_min: float = 10.0,
    alpha_max: float = 10.0,
    curvature_smoothing: int = 5,
    use_sign: bool = False,
    normalize_tensor: bool = False,
    d_min: float = 0.01,
):
    """
    Direct vertex-based curvature tensor.

    Returns
    -------
    T_v : (N,3)
        First local reference axis on vertices (same role as Umin->vertex in old pipeline)
    R_v : (N,3)
        Second in-plane orthogonal axis on vertices
    D_v : (N,2)
        Per-vertex eigenvalues
    """
    V = np.asarray(V, float)
    F_int = F.astype(np.int32, copy=False)

    pd1, pd2, pv1, pv2 = igl.principal_curvature(
        V, F_int, curvature_smoothing, True
    )[:4]

    # keep the same convention as your current face version
    Umax_v = np.asarray(pd1, float)
    Umin_v = np.asarray(pd2, float)
    kmax_v = np.asarray(pv1, float)
    kmin_v = np.asarray(pv2, float)

    normals_v = igl.per_vertex_normals(V, F_int)
    if isinstance(normals_v, tuple):
        normals_v = normals_v[0]
    normals_v = np.asarray(normals_v, float)

    # enforce right-handed consistency: cross(Umin, Umax) aligned with normal
    U3 = np.cross(Umin_v, Umax_v)
    d = np.einsum("ij,ij->i", U3, normals_v)
    flip = d < 0
    if np.any(flip):
        Umax_v[flip] *= -1.0

    def normalize_rows(X, eps=1e-12):
        X = np.asarray(X, float)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm = np.maximum(nrm, eps)
        return X / nrm

    T_v = normalize_rows(Umin_v)   # same role as old target reference first axis
    R_v = normalize_rows(Umax_v)

    def psi(k, alpha, use_sign=False):
        k_abs = np.abs(k)
        base = 1.0 / (1.0 + alpha * k_abs)
        if use_sign:
            return np.sign(k) * base
        return base

    D_v = np.zeros((V.shape[0], 2), dtype=float)

    # keep same ordering convention as old face-based function
    D_v[:, 1] = psi(kmin_v, alpha_min, use_sign)
    D_v[:, 0] = psi(kmax_v, alpha_max, use_sign)

    if normalize_tensor:
        det = np.maximum(D_v[:, 0] * D_v[:, 1], 1e-12)
        scale = 1.0 / np.sqrt(det)
        D_v *= scale[:, None]

    #D_v = np.clip(D_v, 1e-3, 1e3)
    return T_v, R_v, D_v

def vertex_areas_barycentric(V, F):
    """
    Compute barycentric vertex areas: each triangle area is equally
    distributed to its three vertices.

    Parameters
    ----------
    V : (N, 3) array
        Vertex positions.
    F : (M, 3) array
        Triangle indices.

    Returns
    -------
    A_v : (N,) array
        Per-vertex area.
    """
    N = V.shape[0]
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    v0, v1, v2 = V[i0], V[i1], V[i2]

    e01 = v1 - v0
    e02 = v2 - v0
    face_area2 = np.linalg.norm(np.cross(e01, e02), axis=1)  # 2 * area
    face_area = 0.5 * face_area2

    A_v = np.zeros(N)
    for tri, a in zip(F, face_area):
        A_v[tri] += a / 3.0
    return A_v


def build_edge_data_for_divergence(V, F, eps=1e-12):
    """
    Precompute geometric data for the cotangent-based divergence
    used in the heat method.

    Returns
    -------
    edges : (E, 2) int array
        Vertex indices (i, j) with i < j.
    cot_sum : (E,) float array
        Sum of opposite cotangents for each edge.
    e_ij : (E, 3) float array
        Edge vectors v_j - v_i.
    """
    V = np.asarray(V)
    F = np.asarray(F)
    M = F.shape[0]

    # Compute interior angles and their cotangents per face
    angles = np.zeros((M, 3))
    for k in range(3):
        i1 = k
        i2 = (k + 1) % 3
        i3 = (k + 2) % 3

        p = V[F[:, i2]] - V[F[:, i1]]
        q = V[F[:, i3]] - V[F[:, i1]]

        p_norm = np.linalg.norm(p, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        p_norm = np.maximum(p_norm, eps)
        q_norm = np.maximum(q_norm, eps)

        p_unit = p / p_norm
        q_unit = q / q_norm

        cos_a = np.sum(p_unit * q_unit, axis=1)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angles[:, i1] = np.arccos(cos_a)

    sin_a = np.sin(angles)
    sin_a = np.maximum(sin_a, eps)
    cos_a = np.cos(angles)
    cot_a = cos_a / sin_a  # (M,3)

    # Accumulate cotangent contributions per undirected edge
    edge_dict = {}

    for f in range(M):
        tri = F[f]
        for local_e in range(3):
            i = tri[local_e]
            j = tri[(local_e + 1) % 3]
            k_local = (local_e + 2) % 3
            cot_opp = cot_a[f, k_local]

            a, b = (i, j) if i < j else (j, i)
            key = (a, b)
            if key not in edge_dict:
                edge_dict[key] = {
                    "cot_sum": 0.0,
                    "e": V[b] - V[a]
                }
            edge_dict[key]["cot_sum"] += cot_opp

    E = len(edge_dict)
    edges = np.zeros((E, 2), dtype=int)
    cot_sum = np.zeros(E)
    e_ij = np.zeros((E, 3))

    for idx, ((a, b), data) in enumerate(edge_dict.items()):
        edges[idx, 0] = a
        edges[idx, 1] = b
        cot_sum[idx] = data["cot_sum"]
        e_ij[idx] = data["e"]

    return edges, cot_sum, e_ij

def compute_vertex_gradient(V, F, u, eps=1e-12):
    """
    Given a scalar function u at vertices, compute an approximate
    per-vertex unit vector field X by:
        1) computing per-face gradients via libigl's grad operator,
        2) averaging face gradients to vertices,

    Returns
    -------
    X : (N, 3) array
        Per-vertex unit vector field (approximation of -∇u / |∇u|).
    """
    G = igl.grad(V, F)         # (3M, N)
    gu = G @ u                 # (3M,)
    gu_f = gu.reshape((-1, 3)) # (M,3), per-face gradients

    N = V.shape[0]
    grad_v = np.zeros((N, 3))
    count_v = np.zeros(N)

    # Simple averaging of face gradients to vertices
    for f in range(F.shape[0]):
        tri = F[f]
        for vid in tri:
            grad_v[vid] += gu_f[f]
            count_v[vid] += 1.0

    count_v[count_v == 0] = 1.0
    grad_v /= count_v[:, None]
    """
    norms = np.linalg.norm(grad_v, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    X = -grad_v / norms
    """
    return grad_v

from scipy.sparse.linalg import spsolve

def divergence_cot(V, F, X, edges, cot_sum, e_ij, A_v, eps=1e-12):
    N = V.shape[0]
    div = np.zeros(N)

    for k in range(edges.shape[0]):
        i, j = edges[k]
        w = cot_sum[k]    # cot(angle1)+cot(angle2)
        e = e_ij[k]       # v_j - v_i

        # (X_i - X_j)·e_ij
        dot_ij = np.dot(X[i] - X[j], e)

        div[i] += w * dot_ij
        div[j] -= w * dot_ij

    return div / (2 * A_v)





def compute_mean_edge_length(V, F):
    # F: (M, 3) face indices
    # V: (N, 3) vertices

    edges = np.vstack([
        F[:, [0, 1]],
        F[:, [1, 2]],
        F[:, [2, 0]],
    ])

    e0 = V[edges[:, 0]]
    e1 = V[edges[:, 1]]

    edge_lengths = np.linalg.norm(e0 - e1, axis=1)

    edges_sorted = np.sort(edges, axis=1)
    _, unique_idx = np.unique(edges_sorted, axis=0, return_index=True)
    
    unique_edge_lengths = edge_lengths[unique_idx]
    return unique_edge_lengths.mean()



''' below is the version of first week (buggy version)
def compute_face_principal_directions_and_D(V, F, alpha_min=10.0, alpha_max=10.0):
    """
    Compute per-face principal curvature directions Umin, Umax and
    diffusion eigenvalues D from a given mesh (V, F), using libigl.
    """
    F_int = F.astype(np.int32, copy=False)

    res = igl.principal_curvature(V, F_int)
    pd1, pd2, pv1, pv2 = res[:4]
    is_pv1_min = (pv1 <= pv2)

    kmin = np.where(is_pv1_min, pv1, pv2)
    kmax = np.where(is_pv1_min, pv2, pv1)

    dmin = np.where(is_pv1_min[:, None], pd1, pd2)
    dmax = np.where(is_pv1_min[:, None], pd2, pd1)

    dmin_f = (dmin[F[:, 0]] + dmin[F[:, 1]] + dmin[F[:, 2]]) / 3.0
    dmax_f = (dmax[F[:, 0]] + dmax[F[:, 1]] + dmax[F[:, 2]]) / 3.0

    M_tris = F.shape[0]
    dmin_f = np.zeros((M_tris, 3))
    dmax_f = np.zeros((M_tris, 3))

    for fi in range(M_tris):
        ids = F[fi]
        v0, v1, v2 = ids

        v0_dmin, v1_dmin, v2_dmin = dmin[v0], dmin[v1], dmin[v2]
        # Align signs to avoid ± flips canceling out
        if np.dot(v1_dmin, v0_dmin) < 0:
            v1_dmin = -v1_dmin
        if np.dot(v2_dmin, v0_dmin) < 0:
            v2_dmin = -v2_dmin
        dmin_f[fi] = (v0_dmin + v1_dmin + v2_dmin) / 3.0

        v0_dmax, v1_dmax, v2_dmax = dmax[v0], dmax[v1], dmax[v2]
        if np.dot(v1_dmax, v0_dmax) < 0:
            v1_dmax = -v1_dmax
        if np.dot(v2_dmax, v0_dmax) < 0:
            v2_dmax = -v2_dmax
        dmax_f[fi] = (v0_dmax + v1_dmax + v2_dmax) / 3.0

    def normalize_rows(X, eps=1e-12):
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm = np.maximum(nrm, eps)
        return X / nrm

    Umin = normalize_rows(dmin_f)
    Umax = normalize_rows(dmax_f)

    kmin_f = (kmin[F[:, 0]] + kmin[F[:, 1]] + kmin[F[:, 2]]) / 3.0
    kmax_f = (kmax[F[:, 0]] + kmax[F[:, 1]] + kmax[F[:, 2]]) / 3.0

    # Use 1 / (1 + α|k|) for diffusion tensor eigenvalues
    D = np.zeros((F.shape[0], 2), dtype=float)
    D[:, 0] = 1.0 / (1.0 + alpha_min * np.abs(kmin_f))
    D[:, 1] = 1.0 / (1.0 + alpha_max * np.abs(kmax_f))

    # Optional: clamp values to avoid extreme diffusion suppression
    d_min = 0.01
    D = np.clip(D, d_min, 1.0)
    print("D_min range:", D[:, 0].min(), D[:, 0].max())
    print("D_max range:", D[:, 1].min(), D[:, 1].max())
    print("ratio D_min/D_max:", np.mean(D[:,0]/D[:,1]))

    return Umin, Umax, D
'''
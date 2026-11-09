# anislap/assembly.py

import numpy as np
from scipy.sparse import coo_matrix, diags
import igl


def cotangent_laplacian(V, F):
    """
    Compute the cotangent Laplace–Beltrami operator on a triangle mesh.

    Given:
        V: (N, 3) array of vertex positions
        F: (M, 3) array of triangle indices

    Returns:
        L: (N, N) sparse matrix (csr), the stiffness matrix approximating Δ
        M: (N, N) sparse diagonal mass matrix (csr), with vertex areas on the diagonal

    """
    N = V.shape[0]

    # Indices of the three vertices of each triangle
    i0 = F[:, 0]
    i1 = F[:, 1]
    i2 = F[:, 2]

    # Vertex positions of each triangle
    v0 = V[i0]
    v1 = V[i1]
    v2 = V[i2]

    # Edge vectors:
    e0 = v2 - v1 
    e1 = v2 - v0 
    e2 = v1 - v0 

    l0 = np.sum(e0**2, axis=1)
    l1 = np.sum(e1**2, axis=1)
    l2 = np.sum(e2**2, axis=1)

    # Triangle areas: A = 0.5 * || e1 x e2 ||
    face_normal = np.cross(e1, e2)
    area2 = np.linalg.norm(face_normal, axis=1)
    area = 0.5 * area2 + 1e-16  # small epsilon to avoid division by zero

    # Cotangent of angles at the three vertices of each triangle.
    cot0 = (l1 + l2 - l0) / (4.0 * area)  # angle at v0
    cot1 = (l0 + l2 - l1) / (4.0 * area)  # angle at v1
    cot2 = (l0 + l1 - l2) / (4.0 * area)  # angle at v2

    # For each triangle, contributions to off-diagonal entries:
    # edge (i1, i2) gets weight 0.5 * cot(angle at v0), etc.
    I = np.concatenate([i1, i2, i0, i2, i0, i1])
    J = np.concatenate([i2, i1, i2, i0, i1, i0])
    W = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])

    # Assemble the sparse off-diagonal matrix
    L_off = coo_matrix((W, (I, J)), shape=(N, N))

    # Diagonal entries: L_ii = - sum_{j ≠ i} L_ij
    diag_vals = -np.array(L_off.sum(axis=1)).ravel()
    L = L_off + diags(diag_vals)

    # Lumped mass matrix: each triangle area is evenly split among its three vertices 
    vertex_area = np.zeros(N)
    for tri, a in zip(F, area):
        vertex_area[tri] += a / 3.0
    
    M = diags(vertex_area)
    return L.tocsr(), M



def anisotropic_laplacian(V, F, Umin, Umax, D, eps=1e-12):
    V = np.asarray(V, float)
    F = np.asarray(F, np.int32)
    Umin = np.asarray(Umin, float)
    Umax = np.asarray(Umax, float)
    D = np.asarray(D, float)
    N = V.shape[0]

    # vertex-based input -> convert internally for face assembly
    if Umin.shape[0] == V.shape[0]:
        Umin, Umax, D = vertex_field_to_face_tensor_directional(
            V, F, Umin, D, eps=eps
        )
    # ------------------------------------------------------------------
    # 1) Compute triangle angles at each vertex
    #    angles[k, i] = angle at corner i of face k
    # ------------------------------------------------------------------
    angles = np.zeros((F.shape[0], 3), dtype=float)

    for i in range(3):
        i1 = i  
        i2 = (i + 1) % 3
        i3 = (i + 2) % 3

        # For each triangle, vectors from vertex i1 to i2 and i3
        p = V[F[:, i2]] - V[F[:, i1]]
        q = V[F[:, i3]] - V[F[:, i1]]

        p_norm = np.linalg.norm(p, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        p_norm = np.maximum(p_norm, eps)
        q_norm = np.maximum(q_norm, eps)
        p_unit = p / p_norm
        q_unit = q / q_norm

        # Angle at i1: arccos( dot(p_unit, q_unit) )
        cos_angle = np.sum(p_unit * q_unit, axis=1)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles[:, i1] = np.arccos(cos_angle)

    sin_angles = np.sin(angles)
    sin_angles = np.maximum(sin_angles, eps)
    cos_angles = np.cos(angles)
    cot_angles = cos_angles / sin_angles

    # ------------------------------------------------------------------
    # 2) Assemble anisotropic stiffness W
    #      e1, e2: edge directions inside the triangle
    #      factore: off-diagonal contribution W(i1, i2) = W(i2, i1)
    #      factord: diagonal contribution W(i1, i1)
    # ------------------------------------------------------------------
    rows = []
    cols = []
    vals = []

    #!!! Here the convention is the opposite: D[:, 0] for U1 (second direction), D[:, 1] for U0 (first direction)
    lam_min = D[:, 0]            
    lam_max = D[:, 1]

    for i in range(3):
        i1 = i
        i2 = (i + 1) % 3
        i3 = (i + 2) % 3

        e1 = V[F[:, i3]] - V[F[:, i2]]
        e2 = V[F[:, i1]] - V[F[:, i3]]

        e1_norm = np.linalg.norm(e1, axis=1, keepdims=True)
        e2_norm = np.linalg.norm(e2, axis=1, keepdims=True)
        e1_norm = np.maximum(e1_norm, eps)
        e2_norm = np.maximum(e2_norm, eps)
        e1_unit = e1 / e1_norm
        e2_unit = e2 / e2_norm

        # Projections onto principal curvature directions (per face)
        proj_e1_min = np.sum(e1_unit * Umin, axis=1)
        proj_e2_min = np.sum(e2_unit * Umin, axis=1)
        proj_e1_max = np.sum(e1_unit * Umax, axis=1)
        proj_e2_max = np.sum(e2_unit * Umax, axis=1)

        sin_opposite = sin_angles[:, i3]

        # Off-diagonal contribution 
        factore = -0.5 * (
            lam_min * proj_e1_min * proj_e2_min
            + lam_max * proj_e1_max * proj_e2_max
        ) / sin_opposite

        # Diagonal contribution
        cot_i2 = cot_angles[:, i2]
        cot_i3 = cot_angles[:, i3]
        factord = -0.5 * (
            lam_min * proj_e1_min**2
            + lam_max * proj_e1_max**2
        ) * (cot_i2 + cot_i3)

        # Accumulate symmetric off-diagonal and diagonal terms
        rows.append(F[:, i1])
        cols.append(F[:, i2])
        vals.append(factore)

        rows.append(F[:, i2])
        cols.append(F[:, i1])
        vals.append(factore)

        rows.append(F[:, i1])
        cols.append(F[:, i1])
        vals.append(factord)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    W = coo_matrix((vals, (rows, cols)), shape=(N, N))

    # Enforce symmetry explicitly (optional)
    W = 0.5 * (W + W.T)

    # ------------------------------------------------------------------
    # 3) Lumped mass matrix 
    # ------------------------------------------------------------------
    i0 = F[:, 0]
    i1 = F[:, 1]
    i2 = F[:, 2]

    v0 = V[i0]
    v1 = V[i1]
    v2 = V[i2]

    e01 = v1 - v0
    e02 = v2 - v0
    face_normal = np.cross(e01, e02)
    area2 = np.linalg.norm(face_normal, axis=1)
    area = 0.5 * area2

    vertex_area = np.zeros(N)
    for tri, a in zip(F, area):
        vertex_area[tri] += a / 3.0

    M = diags(vertex_area)

    return W.tocsr(), M

def _vertex_tensor_to_face(V, F, Umin_v, Umax_v, D_v, eps=1e-12):
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    Umin_v = np.asarray(Umin_v, float)
    Umax_v = np.asarray(Umax_v, float)
    D_v = np.asarray(D_v, float)

    # 1) vertex SPD tensor 
    A_v = (
        D_v[:, 0][:, None, None] * (Umin_v[:, :, None] * Umin_v[:, None, :]) +
        D_v[:, 1][:, None, None] * (Umax_v[:, :, None] * Umax_v[:, None, :])
    )   # (N,3,3)

    # 2) face average
    A_f = (
        A_v[F[:, 0]] +
        A_v[F[:, 1]] +
        A_v[F[:, 2]]
    ) / 3.0

    # 3) project to face tangent plane
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    n_f = np.cross(v1 - v0, v2 - v0)
    nrm = np.linalg.norm(n_f, axis=1, keepdims=True)
    n_f = n_f / np.maximum(nrm, eps)

    I3 = np.eye(3)[None, :, :]
    P_f = I3 - n_f[:, :, None] * n_f[:, None, :]
    A_f = P_f @ A_f @ P_f

    # 4) eigendecompose on each face
    evals, evecs = np.linalg.eigh(A_f)   # ascending

    U1_f = evecs[:, :, 1]
    U2_f = evecs[:, :, 2]
    D_f = np.stack([evals[:, 1], evals[:, 2]], axis=1)

    return U1_f.astype(np.float64), U2_f.astype(np.float64), D_f.astype(np.float64)

def vertex_normals(V: np.ndarray, F: np.ndarray):
    vn = igl.per_vertex_normals(V, F)
    if isinstance(vn, tuple):
        vn = vn[0]
    return np.asarray(vn, dtype=np.float64)


def face_normals(V: np.ndarray, F: np.ndarray, eps=1e-12):
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    good = nn[:, 0] > eps
    n[good] /= nn[good]
    return n


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
    Y = X.copy()
    good = n[:, 0] > eps
    Y[good] /= n[good]
    return Y


def _project_rows_to_tangent(X, N, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)

    Y = X - np.sum(X * N, axis=1, keepdims=True) * N
    nY = np.linalg.norm(Y, axis=1, keepdims=True)
    good = nY[:, 0] > eps
    Y[good] /= nY[good]

    bad = np.where(~good)[0]
    for i in bad:
        Y[i] = arbitrary_tangent(N[i], eps=eps)
    return Y

def vertex_field_to_face_tensor_directional(V, F, T_v, D_v, eps=1e-12):
    """
    Convert a vertex line-field tensor to a face tensor.

    Clean rule:
    - Interpolate the direction field on each face.
    - Project directions to the face tangent plane.
    - Average eigenvalues to the face.
    - Reconstruct an orthonormal face basis.

    Important:
    If D_v[:, 0] == D_v[:, 1], the resulting face tensor is scalar identity
    on the face tangent plane, independent of T_v.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    T_v = np.asarray(T_v, dtype=np.float64)
    D_v = np.asarray(D_v, dtype=np.float64)

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    # Face normals.
    n_face = np.cross(v1 - v0, v2 - v0)
    n_face = n_face / np.maximum(
        np.linalg.norm(n_face, axis=1, keepdims=True),
        eps,
    )

    # Project vertex directions to the corresponding face tangent plane.
    T0 = T_v[F[:, 0]].copy()
    T1 = T_v[F[:, 1]].copy()
    T2 = T_v[F[:, 2]].copy()

    def project_to_face(T):
        T = T - np.sum(T * n_face, axis=1, keepdims=True) * n_face
        nT = np.linalg.norm(T, axis=1, keepdims=True)
        good = nT[:, 0] > eps
        T[good] /= nT[good]
        return T, good

    T0, good0 = project_to_face(T0)
    T1, good1 = project_to_face(T1)
    T2, good2 = project_to_face(T2)

    # Fallback for degenerate projected directions.
    edge01 = v1 - v0
    edge01 = edge01 - np.sum(edge01 * n_face, axis=1, keepdims=True) * n_face
    edge01 = edge01 / np.maximum(
        np.linalg.norm(edge01, axis=1, keepdims=True),
        eps,
    )

    T0[~good0] = edge01[~good0]
    T1[~good1] = edge01[~good1]
    T2[~good2] = edge01[~good2]

    # Sign-align the line field before averaging.
    flip1 = np.sum(T0 * T1, axis=1) < 0.0
    flip2 = np.sum(T0 * T2, axis=1) < 0.0

    T1[flip1] *= -1.0
    T2[flip2] *= -1.0

    # Average direction on each face.
    U_tan = (T0 + T1 + T2) / 3.0
    U_tan = U_tan - np.sum(U_tan * n_face, axis=1, keepdims=True) * n_face

    nt = np.linalg.norm(U_tan, axis=1, keepdims=True)
    good = nt[:, 0] > eps
    U_tan[good] /= nt[good]

    # Fallback when three directions cancel.
    U_tan[~good] = edge01[~good]

    # Orthogonal tangent direction.
    U_ortho = np.cross(n_face, U_tan)
    U_ortho = U_ortho / np.maximum(
        np.linalg.norm(U_ortho, axis=1, keepdims=True),
        eps,
    )

    # Average eigenvalues to face.
    D_f = (
        D_v[F[:, 0]] +
        D_v[F[:, 1]] +
        D_v[F[:, 2]]
    ) / 3.0

    return U_tan, U_ortho, D_f

def vertex_field_to_face_tensor(V_full, F_full, T_v, D_v, eps=1e-12):
    V = np.asarray(V_full, float)
    F = np.asarray(F_full, int)
    T_v = np.asarray(T_v, float)
    D_v = np.asarray(D_v, float)

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    n_face = np.cross(v1 - v0, v2 - v0)
    nn = np.linalg.norm(n_face, axis=1, keepdims=True)
    n_face = n_face / np.maximum(nn, eps)

    U_tan = (T_v[F[:, 0]] + T_v[F[:, 1]] + T_v[F[:, 2]]) / 3.0

    U_tan = U_tan - np.sum(U_tan * n_face, axis=1, keepdims=True) * n_face

    nt = np.linalg.norm(U_tan, axis=1, keepdims=True)
    good = nt[:, 0] > eps
    U_tan[good] /= nt[good]

    bad = ~good
    if np.any(bad):
        e = v1[bad] - v0[bad]
        e = e - np.sum(e * n_face[bad], axis=1, keepdims=True) * n_face[bad]
        ne = np.linalg.norm(e, axis=1, keepdims=True)
        e = e / np.maximum(ne, eps)
        U_tan[bad] = e

    U_ortho = np.cross(n_face, U_tan)
    no = np.linalg.norm(U_ortho, axis=1, keepdims=True)
    U_ortho /= np.maximum(no, eps)

    D_f = (D_v[F[:, 0]] + D_v[F[:, 1]] + D_v[F[:, 2]]) / 3.0

    Umin_f = U_tan.astype(np.float32)
    Umax_f = U_ortho.astype(np.float32)
    D_f = D_f.astype(np.float32)

    return Umin_f, Umax_f, D_f

def vertex_field_to_face_tensor_spd(V, F, T_v, R_v=None, D_v=None, eps=1e-12):
    if D_v is None:
        D_v = R_v
        R_v = None

    if D_v is None:
        raise ValueError("D_v must be provided.")

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    T_v = np.asarray(T_v, dtype=np.float64)
    D_v = np.asarray(D_v, dtype=np.float64)

    VN_v = vertex_normals(V, F)
    T_v = _project_rows_to_tangent(T_v, VN_v, eps=eps)

    R_v = np.cross(VN_v, T_v)
    R_v = _normalize_rows(R_v, eps=eps)

    lam_fast = np.maximum(D_v[:, 0], 1e-8)
    lam_slow = np.maximum(D_v[:, 1], 1e-8)

    TT = T_v[:, :, None] * T_v[:, None, :]
    RR = R_v[:, :, None] * R_v[:, None, :]
    A_v = lam_fast[:, None, None] * TT + lam_slow[:, None, None] * RR

    A_f = (A_v[F[:, 0]] + A_v[F[:, 1]] + A_v[F[:, 2]]) / 3.0

    NF = face_normals(V, F, eps=eps)
    I3 = np.eye(3, dtype=np.float64)[None, :, :]
    P = I3 - NF[:, :, None] * NF[:, None, :]

    A_f = P @ A_f @ P
    A_f = 0.5 * (A_f + np.swapaxes(A_f, 1, 2))

    evals, evecs = np.linalg.eigh(A_f)

    lam_slow_f = np.maximum(evals[:, 1], 1e-8)
    lam_fast_f = np.maximum(evals[:, 2], 1e-8)

    Ufast_f = evecs[:, :, 2]
    Ufast_f = _project_rows_to_tangent(Ufast_f, NF, eps=eps)

    Uslow_f = np.cross(NF, Ufast_f)
    Uslow_f = _normalize_rows(Uslow_f, eps=eps)

    D_f = np.stack([lam_fast_f, lam_slow_f], axis=1)

    return (
        Ufast_f.astype(np.float64),
        Uslow_f.astype(np.float64),
        D_f.astype(np.float64),
    )

def face_tensor_to_vertex_field_spd(V, F, Umin_f, Umax_f, D_f, eps=1e-12):
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    Umin_f = np.asarray(Umin_f, float)
    Umax_f = np.asarray(Umax_f, float)
    D_f = np.asarray(D_f, float)

    nV = V.shape[0]
    n_v = igl.per_vertex_normals(V, F)

    lam_fast_f = D_f[:, 0]
    lam_slow_f = D_f[:, 1]

    UU = Umin_f[:, :, None] * Umin_f[:, None, :]
    VV = Umax_f[:, :, None] * Umax_f[:, None, :]
    A_f = lam_fast_f[:, None, None] * UU + lam_slow_f[:, None, None] * VV  # (M,3,3)

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    A_v = np.zeros((nV, 3, 3), dtype=np.float64)
    w_v = np.zeros(nV, dtype=np.float64)

    for c in range(3):
        idx = F[:, c]
        np.add.at(A_v, idx, area[:, None, None] * A_f)
        np.add.at(w_v, idx, area)

    A_v /= np.maximum(w_v[:, None, None], eps)

    I3 = np.eye(3)[None, :, :]
    P = I3 - n_v[:, :, None] * n_v[:, None, :]
    A_v = P @ A_v @ P

    evals, evecs = np.linalg.eigh(A_v)  # ascending
    lam_slow_v = evals[:, 1]
    lam_fast_v = evals[:, 2]

    R_v = evecs[:, :, 1]   # slow
    T_v = evecs[:, :, 2]   # fast

    # re-project / re-orthonormalize for robustness
    T_v = T_v - np.sum(T_v * n_v, axis=1, keepdims=True) * n_v
    T_v /= np.maximum(np.linalg.norm(T_v, axis=1, keepdims=True), eps)

    R_v = R_v - np.sum(R_v * n_v, axis=1, keepdims=True) * n_v
    R_v = R_v - np.sum(R_v * T_v, axis=1, keepdims=True) * T_v
    R_v /= np.maximum(np.linalg.norm(R_v, axis=1, keepdims=True), eps)

    D_v = np.stack([lam_fast_v, lam_slow_v], axis=1)

    return T_v.astype(np.float32), R_v.astype(np.float32), D_v.astype(np.float32)
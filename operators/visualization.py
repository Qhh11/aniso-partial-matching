import polyscope as ps
import numpy as np
import polyscope.imgui as psim
import heapq
import igl

_ps_initialized = False


def _ensure_init():
    global _ps_initialized
    if not _ps_initialized:
        ps.init()
        _ps_initialized = True


def _get_or_register_mesh(V, F, mesh_name="mesh"):
    _ensure_init()
    if ps.has_surface_mesh(mesh_name):
        return ps.get_surface_mesh(mesh_name)
    return ps.register_surface_mesh(mesh_name, V, F, smooth_shade=True)


def show_scalar_on_mesh(
    V, F, scalar,
    mesh_name="mesh",
    quantity_name="scalar",
    show_now=True,
    symmetric=True,
):
    """
    Generic scalar visualization.
    symmetric=True  -> for signed quantities
    symmetric=False -> for nonnegative quantities (robust p95 clipping)
    """
    ps_mesh = _get_or_register_mesh(V, F, mesh_name)
    scalar = np.asarray(scalar, dtype=np.float64).reshape(-1)

    finite = np.isfinite(scalar)
    scalar_clean = scalar.copy()
    scalar_clean[~finite] = 0.0

    if symmetric:
        v_max = np.max(np.abs(scalar_clean)) if scalar_clean.size else 1.0
        if v_max < 1e-12:
            scalar_vis = np.zeros_like(scalar_clean)
        else:
            scalar_vis = scalar_clean / v_max
    else:
        pos = scalar_clean[finite]
        pos = pos[pos >= 0]
        if pos.size == 0:
            scalar_vis = np.zeros_like(scalar_clean)
        else:
            p95 = np.percentile(pos, 95)
            if p95 < 1e-12:
                scalar_vis = np.zeros_like(scalar_clean)
            else:
                scalar_clip = np.clip(scalar_clean, 0.0, p95)
                scalar_vis = scalar_clip / p95

    ps_mesh.add_scalar_quantity(
        quantity_name,
        scalar_vis,
        defined_on="vertices",
        cmap="coolwarm",
        enabled=True,
    )

    if show_now:
        ps.show()

def show_scalar_and_directions_on_mesh(
    V,
    F,
    scalar,
    Umin,
    Umax,
    mesh_name="mesh",
    scalar_name="temperature",
    show_now=True,
    symmetric=True,
    vec_on="faces",   # "faces" or "vertices"
):
    _ensure_init()

    if ps.has_surface_mesh(mesh_name):
        ps_mesh = ps.get_surface_mesh(mesh_name)
    else:
        ps_mesh = ps.register_surface_mesh(mesh_name, V, F, smooth_shade=True)

    scalar = np.asarray(scalar)

    if symmetric:
        v_max = np.max(np.abs(scalar))
        scalar_vis = scalar / v_max if v_max > 1e-12 else scalar
    else:
        scalar_vis = scalar

    if scalar_vis.shape[0] == V.shape[0]:
        scalar_on = "vertices"
    elif scalar_vis.shape[0] == F.shape[0]:
        scalar_on = "faces"
    else:
        raise ValueError(
            f"[viz] scalar has length {scalar_vis.shape[0]}, "
            f"expected N={V.shape[0]} or M={F.shape[0]}"
        )

    ps_mesh.add_scalar_quantity(
        scalar_name,
        scalar_vis,
        defined_on=scalar_on,
        cmap="coolwarm",
        enabled=True,
    )

    if vec_on not in ("faces", "vertices"):
        raise ValueError(f"[viz] vec_on must be 'faces' or 'vertices', got {vec_on}")

    expected_len = F.shape[0] if vec_on == "faces" else V.shape[0]

    if Umin.shape[0] != expected_len:
        raise ValueError(
            f"[viz] Umin has shape {Umin.shape}, but vec_on='{vec_on}' "
            f"expects length {expected_len}"
        )

    if Umax.shape[0] != expected_len:
        raise ValueError(
            f"[viz] Umax has shape {Umax.shape}, but vec_on='{vec_on}' "
            f"expects length {expected_len}"
        )

    ps_mesh.add_vector_quantity(
        "U_0",
        Umin,
        defined_on=vec_on,
        enabled=True,
        color=(0.2, 0.8, 0.2),  # green
    )

    ps_mesh.add_vector_quantity(
        "U_1",
        Umax,
        defined_on=vec_on,
        enabled=True,
        color=(0.2, 0.1, 0.8),  # blue
        length=0.02,
    )

    if show_now:
        ps.show()

def show_error_on_mesh(
    V, F, err,
    mesh_name="mesh",
    quantity_name="error",
    show_now=True,
    clip_percentile=95.0,
    cmap="viridis",
):
    """
    Specialized visualization for nonnegative error fields.

    - removes NaN/Inf from scaling
    - clips to percentile to avoid a few huge outliers dominating
    - normalizes to [0,1]
    """
    ps_mesh = _get_or_register_mesh(V, F, mesh_name)
    err = np.asarray(err, dtype=np.float64).reshape(-1)

    finite = np.isfinite(err)
    err_clean = err.copy()
    err_clean[~finite] = 0.0
    err_clean = np.maximum(err_clean, 0.0)

    valid = err_clean[finite]
    valid = valid[valid >= 0]

    if valid.size == 0:
        err_vis = np.zeros_like(err_clean)
        emax = 1.0
    else:
        emax = np.percentile(valid, clip_percentile)
        if emax < 1e-12:
            err_vis = np.zeros_like(err_clean)
            emax = 1.0
        else:
            err_vis = np.clip(err_clean, 0.0, emax) / emax

    q = ps_mesh.add_scalar_quantity(
        quantity_name,
        err_vis,
        defined_on="vertices",
        cmap=cmap,
        enabled=True,
    )

    print(
        f"[viz:error] {quantity_name}: "
        f"raw min={np.nanmin(err):.6g}, raw max={np.nanmax(err):.6g}, "
        f"clip@p{clip_percentile}={emax:.6g}"
    )

    if show_now:
        ps.show()

    return q


def show_weight_on_mesh(
    V, F, weight,
    mesh_name="mesh",
    quantity_name="weight",
    show_now=True,
    cmap="coolwarm",
):
    """
    Specialized visualization for confidence weights in [0,1].
    """
    ps_mesh = _get_or_register_mesh(V, F, mesh_name)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)

    weight_clean = weight.copy()
    weight_clean[~np.isfinite(weight_clean)] = 0.0
    weight_clean = np.clip(weight_clean, 0.0, 1.0)

    ps_mesh.add_scalar_quantity(
        quantity_name,
        weight_clean,
        defined_on="vertices",
        cmap=cmap,
        enabled=True,
    )

    print(
        f"[viz:weight] {quantity_name}: "
        f"min={weight_clean.min():.6g}, max={weight_clean.max():.6g}, "
        f"mean={weight_clean.mean():.6g}"
    )

    if show_now:
        ps.show()



def collect_local_shells_by_hops(F, src_idx, max_hops=1, max_points=300):
    """
    BFS shells on mesh connectivity.

    Returns
    -------
    ids  : (K,) vertex ids
    hop  : (K,) shell index, 0 for src, 1 for 1-ring, etc.
    """
    F = np.asarray(F, int)
    src_idx = int(src_idx)

    adj = _build_vertex_adj_unweighted(F)
    N = len(adj)

    visited = np.zeros(N, dtype=bool)
    hop = np.full(N, -1, dtype=np.int32)

    q = [src_idx]
    visited[src_idx] = True
    hop[src_idx] = 0

    head = 0
    out = [src_idx]

    while head < len(q):
        u = q[head]
        head += 1

        if hop[u] >= max_hops:
            continue

        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                hop[v] = hop[u] + 1
                q.append(v)
                out.append(v)
                if len(out) >= max_points:
                    ids = np.asarray(out, dtype=np.int64)
                    return ids, hop[ids]

    ids = np.asarray(out, dtype=np.int64)
    return ids, hop[ids]


def show_local_correspondence_shells(
    V_full, F_full,
    V_part, F_part,
    src_part,
    p2p_part_to_full,
    T_part_v=None,
    T_full_v=None,
    consistency_w=None,
    consistency_err=None,
    V_part_vis=None,
    max_hops=4,
    max_points=300,
    draw_lines=True,
    show_now=True,
):
    """
    Visualize local correspondence grouped by graph-hop shells.

    Important:
    - This function does NOT register/remove surface meshes.
    - It only adds overlay point clouds + curve networks.
    - Pass V_part_vis from the caller if PART is displayed with an offset.
    """
    _ensure_init()

    V_full = np.asarray(V_full, float)
    F_full = np.asarray(F_full, int)
    V_part = np.asarray(V_part, float)
    F_part = np.asarray(F_part, int)
    p2p = np.asarray(p2p_part_to_full, np.int64).reshape(-1)

    if V_part_vis is None:
        V_part_vis = V_part
    else:
        V_part_vis = np.asarray(V_part_vis, float)

    src_part = int(src_part)

    # -------- collect shells on PART --------
    ring_ids, hop_ids = collect_local_shells_by_hops(
        F_part, src_idx=src_part, max_hops=max_hops, max_points=max_points
    )

    valid = (ring_ids >= 0) & (ring_ids < len(p2p))
    ring_ids = ring_ids[valid]
    hop_ids = hop_ids[valid]

    mapped = p2p[ring_ids]
    valid_map = (mapped >= 0) & (mapped < V_full.shape[0])

    ring_ids = ring_ids[valid_map]
    hop_ids = hop_ids[valid_map]
    mapped = mapped[valid_map]

    if ring_ids.size == 0:
        print("[viz] local shells have no valid mapped points.")
        return

    # -------- clean old overlay structures --------
    # remove old shell overlays if rerunning many times
    for h in range(max_hops + 1):
        for name in [
            f"shell_{h}_part",
            f"shell_{h}_full",
            f"shell_{h}_lines",
        ]:
            if ps.has_point_cloud(name) or ps.has_curve_network(name):
                try:
                    ps.remove_structure(name)
                except Exception:
                    pass

    for name in ["src_part_point_local", "src_full_point_local"]:
        try:
            ps.remove_structure(name)
        except Exception:
            pass

    # -------- shell colors --------
    palette = [
        (1.00, 0.20, 0.20),  # red
        (0.20, 0.70, 1.00),  # blue
        (0.20, 0.85, 0.35),  # green
        (1.00, 0.65, 0.15),  # orange
        (0.70, 0.35, 1.00),  # purple
        (1.00, 0.85, 0.20),  # yellow
        (0.20, 0.85, 0.85),  # cyan
    ]

    # optional local direction diagnostic
    dir_err_deg_all = None
    dir_agree_all = None
    if T_part_v is not None and T_full_v is not None:
        T_part_v = np.asarray(T_part_v, float)
        T_full_v = np.asarray(T_full_v, float)
        n_full = igl.per_vertex_normals(V_full, F_full)

        t_src = T_part_v[ring_ids].copy()
        n_tgt = n_full[mapped]

        t_proj = t_src - np.sum(t_src * n_tgt, axis=1, keepdims=True) * n_tgt
        nt = np.linalg.norm(t_proj, axis=1, keepdims=True)
        good = nt[:, 0] > 1e-12
        t_proj[good] /= nt[good]

        t_full = T_full_v[mapped].copy()
        ntf = np.linalg.norm(t_full, axis=1, keepdims=True)
        goodf = ntf[:, 0] > 1e-12
        t_full[goodf] /= ntf[goodf]

        cos_abs = np.abs(np.sum(t_proj * t_full, axis=1))
        cos_abs = np.clip(cos_abs, 0.0, 1.0)

        dir_agree_all = cos_abs
        dir_err_deg_all = np.degrees(np.arccos(cos_abs))

    # -------- per-shell draw --------
    unique_hops = np.unique(hop_ids)

    for h in unique_hops:
        mask = (hop_ids == h)
        ids_h = ring_ids[mask]
        mapped_h = mapped[mask]
        color = palette[int(h) % len(palette)]

        P_part_h = V_part_vis[ids_h]
        P_full_h = V_full[mapped_h]

        pc_part = ps.register_point_cloud(
            f"shell_{h}_part",
            P_part_h,
            enabled=True,
            color=color,
        )

        pc_full = ps.register_point_cloud(
            f"shell_{h}_full",
            P_full_h,
            enabled=True,
            color=color,
        )

        if consistency_w is not None:
            cw = np.asarray(consistency_w).reshape(-1)[mapped_h]
            pc_full.add_scalar_quantity(
                f"shell_{h}_consistency_weight",
                cw,
                enabled=False,
                cmap="coolwarm",
            )

        if consistency_err is not None:
            ce = np.asarray(consistency_err).reshape(-1)[mapped_h]
            ce = np.where(np.isfinite(ce), ce, np.nan)
            pc_full.add_scalar_quantity(
                f"shell_{h}_consistency_err",
                ce,
                enabled=False,
                cmap="viridis",
            )

        if dir_err_deg_all is not None:
            pc_full.add_scalar_quantity(
                f"shell_{h}_dir_err_deg",
                dir_err_deg_all[mask],
                enabled=False,
                cmap="viridis",
            )
            pc_full.add_scalar_quantity(
                f"shell_{h}_dir_agreement_absdot",
                dir_agree_all[mask],
                enabled=False,
                cmap="coolwarm",
            )

        if draw_lines:
            nodes = np.vstack([P_part_h, P_full_h])
            k = P_part_h.shape[0]
            edges = np.column_stack([np.arange(k), np.arange(k) + k])

            cn = ps.register_curve_network(
                f"shell_{h}_lines",
                nodes,
                edges,
                enabled=True,
                color=color,
            )
            cn.set_radius(0.0007, relative=False)

    # -------- source markers --------
    ps.register_point_cloud(
        "src_part_point_local",
        V_part_vis[src_part:src_part+1],
        enabled=True,
        color=(1.0, 1.0, 1.0),
    )

    src_full = int(p2p[src_part]) if 0 <= src_part < len(p2p) else -1
    if 0 <= src_full < V_full.shape[0]:
        ps.register_point_cloud(
            "src_full_point_local",
            V_full[src_full:src_full+1],
            enabled=True,
            color=(1.0, 1.0, 1.0),
        )

    print(
        f"[viz] local shells: total={ring_ids.size}, "
        f"max_hops={max_hops}, shells={list(unique_hops)}"
    )

    if show_now:
        ps.show()

def _build_vertex_adj_unweighted(F):
    F = np.asarray(F, int)
    N = int(F.max()) + 1
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    adj = [[] for _ in range(N)]
    for i, j in edges:
        i = int(i)
        j = int(j)
        adj[i].append(j)
        adj[j].append(i)
    return adj


def collect_shells_by_hops(F, src_idx, max_hops=2, max_points=200):
    src_idx = int(src_idx)
    adj = _build_vertex_adj_unweighted(F)
    N = len(adj)

    visited = np.zeros(N, dtype=bool)
    hop = np.full(N, -1, dtype=np.int32)

    q = [src_idx]
    visited[src_idx] = True
    hop[src_idx] = 0

    head = 0
    out = [src_idx]

    while head < len(q):
        u = q[head]
        head += 1

        if hop[u] >= max_hops:
            continue

        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                hop[v] = hop[u] + 1
                q.append(v)
                out.append(v)
                if len(out) >= max_points:
                    ids = np.asarray(out, dtype=np.int64)
                    return ids, hop[ids]

    ids = np.asarray(out, dtype=np.int64)
    return ids, hop[ids]

def _extract_transfer_donor_points(debug, V_part=None, F_part=None):
    """
    Return donor point on PART for every FULL vertex.
    Preferred:
      debug["closest_part_point"] : (N_full, 3)
    Fallback:
      debug["part_face_idx"] + debug["part_bc"]
      or
      debug["proj_face_idx"] + debug["proj_bc"]
    """
    if debug is None:
        raise ValueError("[viz] debug is None")

    if "closest_part_point" in debug:
        P = np.asarray(debug["closest_part_point"], dtype=np.float64)
        if P.ndim == 2 and P.shape[1] == 3:
            return P

    # fallback: reconstruct donor point from barycentric data
    face_key = None
    bc_key = None
    for fk, bk in [("part_face_idx", "part_bc"), ("proj_face_idx", "proj_bc")]:
        if fk in debug and bk in debug:
            face_key = fk
            bc_key = bk
            break

    if face_key is None or bc_key is None:
        raise KeyError(
            "[viz] Cannot find donor point info in debug. "
            "Need either 'closest_part_point' or ('part_face_idx' + 'part_bc')."
        )

    if V_part is None or F_part is None:
        raise ValueError("[viz] Need V_part/F_part to reconstruct donor points from barycentric data.")

    face_idx = np.asarray(debug[face_key], dtype=np.int64).reshape(-1)
    bc = np.asarray(debug[bc_key], dtype=np.float64)
    tri = F_part[face_idx]

    P = (
        bc[:, 0:1] * V_part[tri[:, 0]] +
        bc[:, 1:2] * V_part[tri[:, 1]] +
        bc[:, 2:3] * V_part[tri[:, 2]]
    )
    return P


def show_full_to_part_transport_shells(
    V_full, F_full,
    V_part, F_part,
    src_full,
    debug,
    consistency_w=None,
    consistency_err=None,
    V_part_vis=None,
    max_hops=2,
    max_points=120,
    show_now=True,
):
    """
    Correct transfer visualization:

    - choose a local shell on FULL around src_full
    - for each FULL vertex x_i in that shell:
          donor_i = transfer donor point on PART
    - draw:
          donor_i on LEFT partial  --->  x_i on RIGHT full

    This visualizes transfer correspondence, NOT p2p.
    """
    _ensure_init()

    V_full = np.asarray(V_full, float)
    F_full = np.asarray(F_full, int)
    V_part = np.asarray(V_part, float)
    F_part = np.asarray(F_part, int)
    src_full = int(src_full)

    if V_part_vis is None:
        bbox = V_full.max(axis=0) - V_full.min(axis=0)
        part_offset = np.array([-1.5 * bbox[0], 0.0, 0.0], dtype=np.float64)
        V_part_vis = V_part + part_offset[None, :]
    else:
        V_part_vis = np.asarray(V_part_vis, float)
        # safer than taking vertex 0 only
        part_offset = np.mean(V_part_vis - V_part, axis=0)

    # ---- donor point on PART for each FULL vertex ----
    donor_part = _extract_transfer_donor_points(debug, V_part=V_part, F_part=F_part)

    valid = np.isfinite(donor_part).all(axis=1)
    if "valid_face_map" in debug:
        valid &= np.asarray(debug["valid_face_map"], dtype=bool).reshape(-1)
    if "transport_ok" in debug:
        valid &= np.asarray(debug["transport_ok"], dtype=bool).reshape(-1)

    # ---- shell on FULL ----
    shell_ids, hop_ids = collect_shells_by_hops(
        F_full, src_idx=src_full, max_hops=max_hops, max_points=max_points
    )

    keep = valid[shell_ids]
    shell_ids = shell_ids[keep]
    hop_ids = hop_ids[keep]

    if shell_ids.size == 0:
        print("[viz] no valid transfer donor points in selected full shell")
        return

    # shifted donor points on LEFT partial
    P_part = donor_part[shell_ids] + part_offset[None, :]
    P_full = V_full[shell_ids]

    # ---- clean old overlay ----
    for h in range(max_hops + 1):
        for name in [
            f"transfer_full_shell_{h}",
            f"transfer_part_donor_shell_{h}",
            f"transfer_part_to_full_lines_{h}",
        ]:
            try:
                ps.remove_structure(name)
            except Exception:
                pass

    for name in [
        "transfer_src_full_point",
        "transfer_src_part_donor_point",
    ]:
        try:
            ps.remove_structure(name)
        except Exception:
            pass

    palette = [
        (1.00, 0.20, 0.20),  # red
        (0.20, 0.70, 1.00),  # blue
        (0.20, 0.85, 0.35),  # green
        (1.00, 0.65, 0.15),  # orange
        (0.70, 0.35, 1.00),  # purple
        (1.00, 0.85, 0.20),  # yellow
    ]

    unique_hops = np.unique(hop_ids)

    for h in unique_hops:
        m = (hop_ids == h)
        ids_h = shell_ids[m]
        color = palette[int(h) % len(palette)]

        P_full_h = V_full[ids_h]
        P_part_h = donor_part[ids_h] + part_offset[None, :]

        pc_full = ps.register_point_cloud(
            f"transfer_full_shell_{h}",
            P_full_h,
            enabled=True,
            color=color,
        )

        pc_part = ps.register_point_cloud(
            f"transfer_part_donor_shell_{h}",
            P_part_h,
            enabled=True,
            color=color,
        )

        if consistency_err is not None:
            ce = np.asarray(consistency_err, float).reshape(-1)[ids_h]
            pc_full.add_scalar_quantity(
                f"transfer_consistency_err_h{h}",
                ce,
                enabled=False,
                cmap="viridis",
            )

        if consistency_w is not None:
            cw = np.asarray(consistency_w, float).reshape(-1)[ids_h]
            pc_full.add_scalar_quantity(
                f"transfer_consistency_w_h{h}",
                cw,
                enabled=False,
                cmap="coolwarm",
            )

        # donor on PART -> FULL vertex
        nodes = np.vstack([P_part_h, P_full_h])
        k = P_part_h.shape[0]
        edges = np.column_stack([np.arange(k), np.arange(k) + k])

        cn = ps.register_curve_network(
            f"transfer_part_to_full_lines_{h}",
            nodes,
            edges,
            enabled=True,
            color=color,
        )
        cn.set_radius(0.0005, relative=False)

    # ---- source markers for transfer, not p2p ----
    ps.register_point_cloud(
        "transfer_src_full_point",
        V_full[src_full:src_full+1],
        enabled=True,
        color=(1.0, 1.0, 1.0),
    )

    donor_src = donor_part[src_full:src_full+1] + part_offset[None, :]
    ps.register_point_cloud(
        "transfer_src_part_donor_point",
        donor_src,
        enabled=True,
        color=(1.0, 1.0, 1.0),
    )

    print(
        f"[viz] transfer donor shells: total={shell_ids.size}, "
        f"max_hops={max_hops}, shells={list(unique_hops)}"
    )

    if show_now:
        ps.show()
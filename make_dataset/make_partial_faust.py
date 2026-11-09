#!/usr/bin/env python3
"""
Make a CLEAN partial-to-full dataset from ORIGINAL MPI-FAUST registrations.
libigl-free version: pure numpy + scipy only.

All 100 FAUST registrations share the SAME 6890-vertex topology, so vertex id i
is the same anatomical point everywhere. The partial->full GT is therefore
TRIVIAL and DENSE: partial vertex (orig id i) -> full vertex (orig id i), for any
full shape (same or cross subject). 100% coverage, zero noise.

Hole cutting (PFAUST/SHREC'16 style): pick n_holes seed vertices, remove all
vertices within geodesic radius r of any seed, keep the largest connected comp.

Outputs per shape:
    {out}/{s}_part.ply         partial mesh (re-indexed 0..Np-1)
    {out}/{s}_part2full.npy    (Np,) int: original-6890 id of each partial vertex = dense GT
    {out}/{s}_full.ply         (optional) copy of full mesh
"""

import argparse
import struct
import zlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, connected_components

try:
    import potpourri3d as pp3d
    HAVE_PP3D = True
except ImportError:
    HAVE_PP3D = False


def read_ply(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply", "not a ply"
        fmt = None; n_v = n_f = 0; v_props = []
        in_vertex = in_face = False
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("unexpected EOF in header")
            t = line.strip().split()
            if t[0] == b"format":
                fmt = t[1].decode()
            elif t[0] == b"element":
                if t[1] == b"vertex":
                    n_v = int(t[2]); in_vertex, in_face = True, False
                elif t[1] == b"face":
                    n_f = int(t[2]); in_vertex, in_face = False, True
                else:
                    in_vertex = in_face = False
            elif t[0] == b"property" and in_vertex:
                v_props.append((t[1].decode(), t[2].decode()))
            elif t[0] == b"end_header":
                break
        names = [p[1] for p in v_props]
        xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
        if fmt == "ascii":
            V = np.empty((n_v, 3), np.float64)
            for i in range(n_v):
                vals = f.readline().split()
                V[i] = (float(vals[xi]), float(vals[yi]), float(vals[zi]))
            F = np.empty((n_f, 3), np.int64)
            for i in range(n_f):
                vals = f.readline().split()
                F[i] = (int(vals[1]), int(vals[2]), int(vals[3]))
                if int(vals[0]) != 3:
                    raise RuntimeError("non-triangle face")
            return V, F
        type_map = {"float": "f", "float32": "f", "double": "d", "float64": "d",
                    "uchar": "B", "uint8": "B", "char": "b", "int8": "b",
                    "ushort": "H", "uint16": "H", "short": "h", "int16": "h",
                    "uint": "I", "uint32": "I", "int": "i", "int32": "i"}
        v_fmt = "<" + "".join(type_map[p[0]] for p in v_props)
        v_size = struct.calcsize(v_fmt)
        V = np.empty((n_v, 3), np.float64)
        for i in range(n_v):
            rec = struct.unpack(v_fmt, f.read(v_size))
            V[i] = (rec[xi], rec[yi], rec[zi])
        F = np.empty((n_f, 3), np.int64)
        for i in range(n_f):
            cnt = struct.unpack("<B", f.read(1))[0]
            idx = struct.unpack("<" + "i" * cnt, f.read(4 * cnt))
            if cnt != 3:
                raise RuntimeError("non-triangle face in binary ply")
            F[i] = idx
        return V, F


def write_ply(path, V, F):
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {V.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {F.shape[0]}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in V:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F:
            f.write(f"3 {int(t[0])} {int(t[1])} {int(t[2])}\n")


def edge_graph(V, F):
    N = V.shape[0]
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    w = np.linalg.norm(V[rows] - V[cols], axis=1)
    A = sp.csr_matrix((w, (rows, cols)), shape=(N, N))
    return A.maximum(A.T)


def geodesic_multi_source(V, F, seeds, A=None):
    """Smooth geodesic distance from a set of seeds.
    Uses potpourri3d heat method (smooth isolines -> round holes) if available,
    else falls back to edge-weighted dijkstra (jagged)."""
    N = V.shape[0]
    if HAVE_PP3D:
        solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
        D = np.full(N, np.inf)
        for s in seeds:
            d = np.asarray(solver.compute_distance(int(s)), np.float64)
            D = np.minimum(D, d)
        return D
    D = dijkstra(A, indices=seeds, min_only=True)
    return np.asarray(D, np.float64).reshape(-1)


def cut_holes(V, F, A, n_holes, radius, rng):
    N = V.shape[0]
    seeds = rng.choice(N, size=n_holes, replace=False)
    D = geodesic_multi_source(V, F, seeds, A=A)
    keep = D > radius
    if keep.sum() < 100:
        keep = D > (radius * 0.5)
    face_keep = keep[F].all(axis=1)
    Fk = F[face_keep]
    if Fk.shape[0] == 0:
        return keep
    r = np.concatenate([Fk[:, 0], Fk[:, 1], Fk[:, 2]])
    c = np.concatenate([Fk[:, 1], Fk[:, 2], Fk[:, 0]])
    Ak = sp.csr_matrix((np.ones(r.shape[0]), (r, c)), shape=(N, N))
    Ak = Ak.maximum(Ak.T)
    _, labels = connected_components(Ak, directed=False)
    kept_ids = np.where(keep)[0]
    vals, counts = np.unique(labels[kept_ids], return_counts=True)
    best = vals[np.argmax(counts)]
    keep_final = (labels == best) & keep
    return keep_final   # boolean mask now (smoothing applied by caller)



def _finalize_keep(V, F, A, keep_mask, mask_smooth_iters, mask_smooth_step,
                   min_bridge_width_abs):
    """Apply smoothing + thin-bridge removal + manifold enforcement. Returns the
    final keep mask (what actually gets written)."""
    if mask_smooth_iters > 0:
        keep_mask = smooth_keep_mask(keep_mask, F, V.shape[0],
                                     n_iter=mask_smooth_iters, step=mask_smooth_step)
    if min_bridge_width_abs > 0:
        keep_mask = remove_thin_bridges(V, F, keep_mask, A,
                                        min_bridge_width=min_bridge_width_abs)
    keep_mask = enforce_manifold(V, F, keep_mask)
    return keep_mask


def cut_holes_target(V, F, A, n_holes, target_keep, rng, tol=0.03, max_iter=16,
                     max_seed_tries=25, finalize=None, target_metric="vertex"):
    """Cut n_holes holes and binary-search the geodesic radius so the kept fraction
    ~= target_keep. If binary search over radius cannot hit the target within tol
    (e.g. seeds clustered so the largest component is always too small), RESAMPLE the
    seeds and retry, up to max_seed_tries. Returns the mask closest to target."""
    N = V.shape[0]
    if target_metric not in ("vertex", "area"):
        raise ValueError(f"unknown target_metric={target_metric!r}")
    tri_area = None
    full_area = None
    if target_metric == "area":
        tri_area = 0.5 * np.linalg.norm(
            np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]),
            axis=1,
        )
        full_area = float(tri_area.sum())

    def keep_fraction(keep):
        if target_metric == "vertex":
            return float(keep.mean())
        face_keep = keep[F].all(axis=1)
        return float(tri_area[face_keep].sum() / full_area)

    def eval_seeds(seeds):
        D = geodesic_multi_source(V, F, seeds, A=A)
        dmax = float(D[np.isfinite(D)].max())

        def keep_frac_for_radius(radius):
            keep = D > radius
            if keep.sum() < 100:
                return keep, keep_fraction(keep)
            face_keep = keep[F].all(axis=1)
            Fk = F[face_keep]
            if Fk.shape[0] == 0:
                return keep, keep_fraction(keep)
            r = np.concatenate([Fk[:, 0], Fk[:, 1], Fk[:, 2]])
            c = np.concatenate([Fk[:, 1], Fk[:, 2], Fk[:, 0]])
            Ak = sp.csr_matrix((np.ones(r.shape[0]), (r, c)), shape=(N, N))
            Ak = Ak.maximum(Ak.T)
            _, labels = connected_components(Ak, directed=False)
            kept_ids = np.where(keep)[0]
            vals, counts = np.unique(labels[kept_ids], return_counts=True)
            best = vals[np.argmax(counts)]
            keep_cc = (labels == best) & keep
            if finalize is not None:
                keep_cc = finalize(keep_cc)
            return keep_cc, keep_fraction(keep_cc)

        lo, hi = 0.0, dmax
        best_keep, best_frac = None, None
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            keep_cc, frac = keep_frac_for_radius(mid)
            if best_keep is None or abs(frac - target_keep) < abs(best_frac - target_keep):
                best_keep, best_frac = keep_cc, frac
            if abs(frac - target_keep) <= tol:
                break
            if frac > target_keep:
                lo = mid
            else:
                hi = mid
        return best_keep, best_frac

    # retry seeds until we hit target within tol, keeping the globally-best attempt
    global_best_keep, global_best_frac = None, None
    for _ in range(max_seed_tries):
        seeds = rng.choice(N, size=n_holes, replace=False)
        keep_cc, frac = eval_seeds(seeds)
        if global_best_keep is None or abs(frac - target_keep) < abs(global_best_frac - target_keep):
            global_best_keep, global_best_frac = keep_cc, frac
        if abs(frac - target_keep) <= tol:
            break
    return global_best_keep



def _vertex_adjacency(F, N):
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(N, N))
    A = (A + A.T)
    A.data[:] = 1.0
    return A


def smooth_boundary(keep_mask, F, N, n_iter=3, prune_thresh=0.5, fill_thresh=0.65):
    """
    Morphological smoothing of the kept-vertex region to de-jag the hole boundary.
    Each iteration:
      - PRUNE: a kept vertex whose fraction of kept neighbors < prune_thresh is removed
               (removes thin spikes / jagged protrusions).
      - FILL:  a removed vertex whose fraction of kept neighbors > fill_thresh is added back
               (fills tiny notches, making the boundary rounder).
    Then keep the largest connected component again.
    """
    A = _vertex_adjacency(F, N)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg = np.maximum(deg, 1)
    keep = keep_mask.copy()

    for _ in range(n_iter):
        kept_frac = np.asarray(A.dot(keep.astype(np.float64))).ravel() / deg
        prune = keep & (kept_frac < prune_thresh)
        fill = (~keep) & (kept_frac > fill_thresh)
        new_keep = keep.copy()
        new_keep[prune] = False
        new_keep[fill] = True
        if not (prune.any() or fill.any()):
            break
        keep = new_keep

    # largest connected component of the smoothed region
    face_keep = keep[F].all(axis=1)
    Fk = F[face_keep]
    if Fk.shape[0] == 0:
        return keep
    r = np.concatenate([Fk[:, 0], Fk[:, 1], Fk[:, 2]])
    c = np.concatenate([Fk[:, 1], Fk[:, 2], Fk[:, 0]])
    Ak = sp.csr_matrix((np.ones(r.shape[0]), (r, c)), shape=(N, N))
    Ak = Ak.maximum(Ak.T)
    _, labels = connected_components(Ak, directed=False)
    kept_ids = np.where(keep)[0]
    vals, counts = np.unique(labels[kept_ids], return_counts=True)
    best = vals[np.argmax(counts)]
    out = np.zeros(N, dtype=bool)
    out[(labels == best) & keep] = True
    return out



def _avg_operator(F, N):
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(N, N))
    A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)
    return sp.diags(1.0 / deg) @ A


def smooth_keep_mask(keep_mask, F, N, n_iter=8, step=0.5, thresh=None):
    """De-jag the cut boundary by diffusing the 0/1 KEEP indicator on the mesh and
    re-thresholding. Coordinate-preserving: vertex positions NEVER change, only the
    SELECTION of kept original vertices. part2full / identity GT stay exact.

    AREA-PRESERVING: instead of a fixed 0.5 cut (which shrinks/grows the kept
    region depending on curvature), we threshold at the value that keeps the SAME
    number of vertices as the input mask. So smoothing only trades jagged spikes
    for filled notches, without changing the overall keep fraction."""
    W = _avg_operator(F, N)
    x = keep_mask.astype(np.float64)
    for _ in range(n_iter):
        x = (1.0 - step) * x + step * (W @ x)
    n_keep = int(keep_mask.sum())
    if thresh is not None:
        keep = x > thresh
    else:
        # keep the top-n_keep vertices by diffused value -> preserves area
        if n_keep <= 0 or n_keep >= N:
            keep = keep_mask.copy()
        else:
            cutoff = np.partition(x, N - n_keep)[N - n_keep]
            keep = x >= cutoff
    # largest connected component of the smoothed selection
    face_keep = keep[F].all(axis=1)
    Fk = F[face_keep]
    if Fk.shape[0] == 0:
        return keep
    r = np.concatenate([Fk[:, 0], Fk[:, 1], Fk[:, 2]])
    c = np.concatenate([Fk[:, 1], Fk[:, 2], Fk[:, 0]])
    Ak = sp.csr_matrix((np.ones(r.shape[0]), (r, c)), shape=(N, N)).maximum(
         sp.csr_matrix((np.ones(r.shape[0]), (c, r)), shape=(N, N)))
    _, lab = connected_components(Ak, directed=False)
    kept = np.where(keep)[0]
    vals, cnts = np.unique(lab[kept], return_counts=True)
    best = vals[np.argmax(cnts)]
    return (lab == best) & keep



def _get_boundary_loops_distance(V, F, keep_mask, A):
    """Geodesic distance field from the CURRENT cut boundary (kept vertices adjacent
    to removed vertices). Used to measure how thin the kept region is."""
    N = V.shape[0]
    # cut-boundary = kept vertices that have a removed neighbor
    W = _vertex_adjacency(F, N)
    removed = ~keep_mask
    has_removed_nbr = np.asarray(W.dot(removed.astype(np.float64))).ravel() > 0
    cut_bnd = keep_mask & has_removed_nbr
    if cut_bnd.sum() == 0:
        return np.full(N, np.inf)
    seeds = np.where(cut_bnd)[0]
    D = geodesic_multi_source(V, F, seeds, A=A)
    return D


def remove_thin_bridges(V, F, keep_mask, A, min_bridge_width):
    """
    Eliminate thin structures: any kept vertex whose geodesic distance to the cut
    boundary is < min_bridge_width/2 is in a thin neck. We remove kept vertices that
    are within min_bridge_width/2 of the boundary ON BOTH SIDES (i.e. the local kept
    strip is thinner than min_bridge_width). Practically: erode the kept region by
    min_bridge_width/2, keep the largest CC of the eroded region, then the surviving
    'core' defines which parts are thick enough; thin bridges vanish.
    Coordinate-preserving (only selection changes).
    """
    N = V.shape[0]
    half = 0.5 * min_bridge_width
    Dbnd = _get_boundary_loops_distance(V, F, keep_mask, A)
    # erode: core = kept vertices far enough from boundary
    core = keep_mask & (Dbnd >= half)
    if core.sum() < 50:
        return keep_mask  # too aggressive, skip
    # largest CC of the core (thin bridges disconnect here)
    face_core = core[F].all(axis=1)
    Fc = F[face_core]
    if Fc.shape[0] == 0:
        return keep_mask
    r = np.concatenate([Fc[:, 0], Fc[:, 1], Fc[:, 2]])
    c = np.concatenate([Fc[:, 1], Fc[:, 2], Fc[:, 0]])
    Ac = sp.csr_matrix((np.ones(r.shape[0]), (r, c)), shape=(N, N)).maximum(
         sp.csr_matrix((np.ones(r.shape[0]), (c, r)), shape=(N, N)))
    _, lab = connected_components(Ac, directed=False)
    core_ids = np.where(core)[0]
    vals, cnts = np.unique(lab[core_ids], return_counts=True)
    best = vals[np.argmax(cnts)]
    core_main = (lab == best) & core
    # dilate core_main back by half to restore thickness, but stay within original keep
    W = _vertex_adjacency(F, N)
    grown = core_main.copy()
    n_dilate = max(1, int(np.ceil(half / (_mean_edge_len(V, F) + 1e-12))))
    for _ in range(n_dilate):
        grown = grown | (np.asarray(W.dot(grown.astype(np.float64))).ravel() > 0)
        grown = grown & keep_mask
    return grown


def _mean_edge_len(V, F):
    e = np.concatenate([V[F[:,0]]-V[F[:,1]], V[F[:,1]]-V[F[:,2]], V[F[:,2]]-V[F[:,0]]])
    return float(np.mean(np.linalg.norm(e, axis=1)))


def enforce_manifold(V, F, keep_mask):
    """From a kept-vertex mask, build faces, then remove non-manifold configurations:
    edges shared by >2 faces, and keep the largest edge-manifold connected component.
    Returns a cleaned keep_mask (coordinate-preserving)."""
    N = V.shape[0]
    face_keep = keep_mask[F].all(axis=1)
    Fk = F[face_keep]
    if Fk.shape[0] == 0:
        return keep_mask
    # count faces per undirected edge
    from collections import defaultdict
    edge_faces = defaultdict(list)
    for fi, tri in enumerate(Fk):
        for a, b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            e = (a,b) if a<b else (b,a)
            edge_faces[e].append(fi)
    # faces touching a non-manifold edge (>2 faces) are dropped
    bad_faces = set()
    for e, fl in edge_faces.items():
        if len(fl) > 2:
            bad_faces.update(fl)
    good = np.ones(Fk.shape[0], bool)
    if bad_faces:
        good[list(bad_faces)] = False
    Fk2 = Fk[good]
    if Fk2.shape[0] == 0:
        return keep_mask
    # vertices still used
    used = np.zeros(N, bool)
    used[np.unique(Fk2)] = True
    # largest connected component of the cleaned face set
    r = np.concatenate([Fk2[:,0],Fk2[:,1],Fk2[:,2]])
    c = np.concatenate([Fk2[:,1],Fk2[:,2],Fk2[:,0]])
    Am = sp.csr_matrix((np.ones(r.shape[0]),(r,c)),shape=(N,N)).maximum(
         sp.csr_matrix((np.ones(r.shape[0]),(c,r)),shape=(N,N)))
    _, lab = connected_components(Am, directed=False)
    used_ids = np.where(used)[0]
    vals,cnts = np.unique(lab[used_ids], return_counts=True)
    best = vals[np.argmax(cnts)]
    return used & (lab == best)


def reindex_partial(V, F, keep_ids):
    keep_ids = np.sort(np.asarray(keep_ids, np.int64))
    old2new = -np.ones(V.shape[0], np.int64)
    old2new[keep_ids] = np.arange(keep_ids.shape[0])
    face_keep = np.all(np.isin(F, keep_ids), axis=1)
    Fp = old2new[F[face_keep]]
    # drop degenerate faces (any two equal vertex ids)
    nondegen = (Fp[:,0]!=Fp[:,1]) & (Fp[:,1]!=Fp[:,2]) & (Fp[:,0]!=Fp[:,2])
    Fp = Fp[nondegen]
    Vp = V[keep_ids]
    return Vp, Fp, keep_ids.copy()


def stable_shape_seed(base_seed, shape_name):
    """Derive a process-independent RNG seed from the shape name.

    Python's built-in ``hash()`` is randomized between interpreter processes,
    so it cannot be used for a reproducible dataset generator.
    """
    name_code = zlib.crc32(str(shape_name).encode("utf-8"))
    return (int(base_seed) + int(name_code)) % (2**32)




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--faust_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shapes", nargs="+", required=True)
    p.add_argument("--n_holes", type=int, default=6)
    p.add_argument("--radius_frac", type=float, default=0.10)
    target_group = p.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target_keep", type=float, default=None,
        help=(
            "binary-search radius so the final kept VERTEX fraction matches this; "
            "overrides radius_frac"
        ),
    )
    target_group.add_argument(
        "--target_area_keep", type=float, default=None,
        help=(
            "binary-search radius so the final kept SURFACE-AREA fraction matches "
            "this; preferred when controlling partiality"
        ),
    )
    p.add_argument(
        "--target_tol", type=float, default=0.03,
        help="absolute tolerance for --target_keep/--target_area_keep",
    )
    p.add_argument("--min_bridge_width", type=float, default=0.08,
                   help="min geodesic width between holes as frac of bbox diag; thinner necks are cut (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_full", action="store_true")
    p.add_argument("--mask_smooth_iters", type=int, default=8,
                   help="indicator-diffusion boundary smoothing iterations (0 = off)")
    p.add_argument("--mask_smooth_step", type=float, default=0.5,
                   help="diffusion step per iteration (0..1)")
    args = p.parse_args()

    faust = Path(args.faust_dir).expanduser()
    out = Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

    for s in args.shapes:
        ply = faust / f"{s}.ply"
        if not ply.exists():
            print(f"[skip] {ply} not found"); continue
        V, F = read_ply(ply)
        A = edge_graph(V, F)
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        radius = args.radius_frac * diag
        rng = np.random.default_rng(stable_shape_seed(args.seed, s))

        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        mbw_abs = args.min_bridge_width * diag if args.min_bridge_width > 0 else 0.0
        finalize = lambda km: _finalize_keep(V, F, A, km, args.mask_smooth_iters,
                                             args.mask_smooth_step, mbw_abs)
        if args.target_keep is not None:
            # target search optimizes the FINAL kept fraction (after full cleanup)
            keep_mask = cut_holes_target(V, F, A, args.n_holes, args.target_keep, rng,
                                         tol=args.target_tol, finalize=finalize,
                                         target_metric="vertex")
        elif args.target_area_keep is not None:
            keep_mask = cut_holes_target(
                V, F, A, args.n_holes, args.target_area_keep, rng,
                tol=args.target_tol, finalize=finalize, target_metric="area",
            )
        else:
            keep_mask = cut_holes(V, F, A, args.n_holes, radius, rng)
            keep_mask = finalize(keep_mask)
        keep_ids = np.where(keep_mask)[0]
        Vp, Fp, part2full = reindex_partial(V, F, keep_ids)

        write_ply(out / f"{s}_part.ply", Vp, Fp)
        np.save(out / f"{s}_part2full.npy", part2full)
        if args.save_full:
            write_ply(out / f"{s}_full.ply", V, F)

        full_area = 0.5 * np.linalg.norm(
            np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1
        ).sum()
        part_area = 0.5 * np.linalg.norm(
            np.cross(Vp[Fp[:, 1]] - Vp[Fp[:, 0]], Vp[Fp[:, 2]] - Vp[Fp[:, 0]]), axis=1
        ).sum()
        print(f"[ok] {s}: full={V.shape[0]} -> part={Vp.shape[0]} "
              f"(vertex_keep={100*Vp.shape[0]/V.shape[0]:.1f}%, "
              f"area_keep={100*part_area/full_area:.1f}%), "
              f"holes={args.n_holes}, nominal_r={radius:.4f}")

    print(f"\nDone -> {out}/")
    print("Dense GT: part2full.npy gives orig-id of each partial vertex; corr_part=arange(Np)")


if __name__ == "__main__":
    main()

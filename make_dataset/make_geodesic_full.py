#!/usr/bin/env python3
"""
Precompute full-mesh geodesic distance matrices for the CLEAN dataset, so the
cross script can eval with --eval_metric geodesic_area (official metric).

For each {shape}_full.ply we compute an all-pairs geodesic distance matrix
D (N x N, raw mesh units) via potpourri3d heat method, and save it as
    {out_dist_dir}/{shape}_full.mat   with key 'D'
matching what load_full_geodesic_mat() in the cross script expects
(it looks for dist/{full_stem}.mat and reads key D / D_geo / dist).

NOTE: full is original-6890 FAUST, so N=6890 -> D is 6890x6890 float
(~380 MB float64, ~190 MB float32). We save float32 to halve disk.
All-pairs heat solve is one factorization + N right-hand sides; fast.

Run in the venv that has potpourri3d + scipy:
  python make_geodesic_full.py \
    --clean_dir data/clean_pfaust \
    --out_dist_dir data/clean_pfaust/dist \
    --shapes tr_reg_080 tr_reg_081 ... tr_reg_099
(You only need geodesic mats for the TEST full shapes you evaluate on.)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import scipy.io as sio

import potpourri3d as pp3d


def read_ply_ascii(path):
    with open(path, "r") as f:
        assert f.readline().strip() == "ply"
        n_v = n_f = 0
        while True:
            line = f.readline()
            t = line.split()
            if t and t[0] == "element":
                if t[1] == "vertex": n_v = int(t[2])
                elif t[1] == "face": n_f = int(t[2])
            elif t and t[0] == "end_header":
                break
        V = np.empty((n_v, 3), np.float64)
        for i in range(n_v):
            v = f.readline().split(); V[i] = (float(v[0]), float(v[1]), float(v[2]))
        F = np.empty((n_f, 3), np.int64)
        for i in range(n_f):
            v = f.readline().split(); F[i] = (int(v[1]), int(v[2]), int(v[3]))
    return V, F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean_dir", required=True)
    p.add_argument("--out_dist_dir", required=True)
    p.add_argument("--shapes", nargs="+", required=True)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = p.parse_args()

    clean = Path(args.clean_dir)
    out = Path(args.out_dist_dir); out.mkdir(parents=True, exist_ok=True)
    dt = np.float32 if args.dtype == "float32" else np.float64

    for s in args.shapes:
        ply = clean / f"{s}_full.ply"
        manifest_path = clean / f"{s}_manifest.json"
        if not ply.exists() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            declared_full = manifest.get("shared_full")
            if declared_full:
                ply = clean / declared_full
        if not ply.exists():
            print(f"[skip] {ply} not found"); continue
        outmat = out / f"{s}_full.mat"
        if outmat.exists():
            print(f"[exists] {outmat.name}"); continue

        V, F = read_ply_ascii(ply)
        N = V.shape[0]
        solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
        D = np.empty((N, N), dtype=dt)
        for i in range(N):
            D[i] = solver.compute_distance(i).astype(dt)
        # Symmetrize (heat method is not perfectly symmetric).  The numerical
        # solve can also produce a handful of tiny negative values around zero;
        # geodesic distances are non-negative by definition, so clamp them and
        # make the diagonal exactly zero before serialization.
        D = np.minimum(D, D.T)
        np.maximum(D, 0.0, out=D)
        np.fill_diagonal(D, 0.0)
        sio.savemat(str(outmat), {"D": D}, do_compression=True)
        print(f"[ok] {s}: N={N} -> {outmat.name} ({D.nbytes/1e6:.0f} MB in-mem)")

    print(f"\nDone -> {out}/")


if __name__ == "__main__":
    main()

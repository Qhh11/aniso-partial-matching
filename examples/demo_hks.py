# examples/demo_hks.py
"""
HKS/WKS descriptor visualization demo.

Usage examples (run from project root):

    # Isotropic HKS on FAUST mesh
    python -m examples.demo_hks --mode hks_iso --mesh faust_off/tr_reg_000.off --num-eigs 100

    # Anisotropic HKS on FAUST mesh
    python -m examples.demo_hks --mode hks_aniso --mesh faust_off/tr_reg_000.off --num-eigs 100 --alpha-min 1 --alpha-max 10

    # Isotropic WKS on FAUST mesh
    python -m examples.demo_hks --mode wks_iso --mesh faust_off/tr_reg_000.off --num-eigs 100

    # Anisotropic WKS on FAUST mesh
    python -m examples.demo_hks --mode wks_aniso --mesh faust_off/tr_reg_000.off --num-eigs 100 --alpha-min 1 --alpha-max 10

    # Compare isotropic vs anisotropic HKS
    python -m examples.demo_hks --mode compare_hks --mesh faust_off/tr_reg_000.off --num-eigs 100 --alpha-min 1 --alpha-max 10

    # Boundary-aware HKS on partial mesh
    python -m examples.demo_hks --mode hks_boundary --mesh PFAUST-M/off/tr_reg_000.off --num-eigs 100
"""

import argparse
import numpy as np

from operators.io import load_mesh, save_vertex_scalar
from operators.assembly import cotangent_laplacian, anisotropic_laplacian
from operators.solvers import compute_eigenpairs
from operators.geometry import compute_face_principal_directions_and_D
from operators.boundary import compute_boundary_aware_tensor
from operators.descriptors import compute_hks, compute_wks
from operators.visualization import show_scalar_on_mesh, show_scalar_and_directions_on_mesh


def run_hks_iso(mesh_name: str, num_eigs: int, num_times: int, time_indices: list):
    """
    Compute and visualize isotropic HKS.
    """
    V, F = load_mesh(mesh_name)
    print(f"[hks_iso] Loaded mesh: {mesh_name}")
    print(f"          #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    # Compute Laplacian and eigenpairs
    L, M = cotangent_laplacian(V, F)
    print(f"[hks_iso] Computing {num_eigs} eigenpairs...")
    evals, evecs = compute_eigenpairs(L, M, k=num_eigs, method="auto", maxiter=500)
    print(f"[hks_iso] Eigenvalue range: [{np.abs(evals).min():.4e}, {np.abs(evals).max():.4e}]")

    # Compute HKS
    hks, times = compute_hks(evals, evecs, num_times=num_times)
    print(f"[hks_iso] HKS shape: {hks.shape}")
    print(f"[hks_iso] Time range: [{times[0]:.4e}, {times[-1]:.4e}]")
    print(f"[hks_iso] HKS value range: [{hks.min():.6f}, {hks.max():.6f}]")

    # Visualize selected time scales
    for idx in time_indices:
        if 0 <= idx < num_times:
            scalar = hks[:, idx]
            print(f"[hks_iso] t={times[idx]:.4e}: min={scalar.min():.4f}, max={scalar.max():.4f}")
            show_scalar_on_mesh(
                V, F, scalar,
                mesh_name="mesh_hks_iso",
                quantity_name=f"hks_t{idx}",
                show_now=False,
                symmetric=True
            )

    # Mean HKS across all times
    hks_mean = hks.mean(axis=1)
    show_scalar_on_mesh(
        V, F, hks_mean,
        mesh_name="mesh_hks_iso",
        quantity_name="hks_mean",
        show_now=True,
        symmetric=True
    )


def run_hks_aniso( mesh_name: str, num_eigs: int, num_times: int,time_indices: list,alpha_min: float,alpha_max: float,):
    """
    Compute and visualize anisotropic HKS.
    """
    V, F = load_mesh(mesh_name)
    print(f"[hks_aniso] Loaded mesh: {mesh_name}")
    print(f"            #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    # Compute anisotropic Laplacian
    Umin, Umax, D = compute_face_principal_directions_and_D(
        V, F, alpha_min=alpha_min, alpha_max=alpha_max
    )
    W, M = anisotropic_laplacian(V, F, Umin, Umax, D)

    print(f"[hks_aniso] Computing {num_eigs} eigenpairs...")
    evals, evecs = compute_eigenpairs(W, M, k=num_eigs, method="auto", maxiter=500)
    print(f"[hks_aniso] Eigenvalue range: [{np.abs(evals).min():.4e}, {np.abs(evals).max():.4e}]")

    # Compute HKS
    hks, times = compute_hks(evals, evecs, num_times=num_times)
    print(f"[hks_aniso] HKS shape: {hks.shape}")
    print(f"[hks_aniso] Time range: [{times[0]:.4e}, {times[-1]:.4e}]")
    print(f"[hks_aniso] HKS value range: [{hks.min():.6f}, {hks.max():.6f}]")

    # Visualize with directions
    for idx in time_indices:
        if 0 <= idx < num_times:
            scalar = hks[:, idx]
            print(f"[hks_aniso] t={times[idx]:.4e}: min={scalar.min():.4f}, max={scalar.max():.4f}")
            show_scalar_on_mesh(
                V, F, scalar,
                mesh_name="mesh_hks_aniso",
                quantity_name=f"hks_aniso_t{idx}",
                show_now=False,
                symmetric=True
            )

    hks_mean = hks.mean(axis=1)
    show_scalar_and_directions_on_mesh(
        V, F, hks_mean, Umin, Umax,
        mesh_name="mesh_hks_aniso",
        scalar_name="hks_aniso_mean",
        show_now=True,
        symmetric=True
    )


def run_wks_iso(mesh_name: str, num_eigs: int, num_scales: int, scale_indices: list):
    """
    Compute and visualize isotropic WKS.
    """
    V, F = load_mesh(mesh_name)
    print(f"[wks_iso] Loaded mesh: {mesh_name}")
    print(f"          #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    L, M = cotangent_laplacian(V, F)
    print(f"[wks_iso] Computing {num_eigs} eigenpairs...")
    evals, evecs = compute_eigenpairs(L, M, k=num_eigs, method="auto", maxiter=500)

    # Compute WKS
    wks, energies = compute_wks(evals, evecs, num_scales=num_scales)
    print(f"[wks_iso] WKS shape: {wks.shape}")
    print(f"[wks_iso] Energy range: [{energies[0]:.4e}, {energies[-1]:.4e}]")
    print(f"[wks_iso] Row sum check (should be ~1): {wks.sum(axis=1)[:5]}")

    # Visualize
    for idx in scale_indices:
        if 0 <= idx < num_scales:
            scalar = wks[:, idx]
            print(f"[wks_iso] e={energies[idx]:.4e}: min={scalar.min():.6f}, max={scalar.max():.6f}")
            show_scalar_on_mesh(
                V, F, scalar,
                mesh_name="mesh_wks_iso",
                quantity_name=f"wks_e{idx}",
                show_now=False,
                symmetric=True
            )

    # Mean WKS
    wks_mean = wks.mean(axis=1)
    show_scalar_on_mesh(
        V, F, wks_mean,
        mesh_name="mesh_wks_iso",
        quantity_name="wks_mean",
        show_now=True,
        symmetric=True
    )


def run_wks_aniso(
    mesh_name: str,
    num_eigs: int,
    num_scales: int,
    scale_indices: list,
    alpha_min: float,
    alpha_max: float,
):
    """
    Compute and visualize anisotropic WKS.
    """
    V, F = load_mesh(mesh_name)
    print(f"[wks_aniso] Loaded mesh: {mesh_name}")
    print(f"            #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    Umin, Umax, D = compute_face_principal_directions_and_D(
        V, F, alpha_min=alpha_min, alpha_max=alpha_max
    )
    W, M = anisotropic_laplacian(V, F, Umin, Umax, D)

    print(f"[wks_aniso] Computing {num_eigs} eigenpairs...")
    evals, evecs = compute_eigenpairs(W, M, k=num_eigs, method="auto", maxiter=500)

    wks, energies = compute_wks(evals, evecs, num_scales=num_scales)
    print(f"[wks_aniso] WKS shape: {wks.shape}")
    print(f"[wks_aniso] Energy range: [{energies[0]:.4e}, {energies[-1]:.4e}]")

    for idx in scale_indices:
        if 0 <= idx < num_scales:
            scalar = wks[:, idx]
            print(f"[wks_aniso] e={energies[idx]:.4e}: min={scalar.min():.6f}, max={scalar.max():.6f}")
            show_scalar_on_mesh(
                V, F, scalar,
                mesh_name="mesh_wks_aniso",
                quantity_name=f"wks_aniso_e{idx}",
                show_now=False,
                symmetric=True
            )

    wks_mean = wks.mean(axis=1)
    show_scalar_and_directions_on_mesh(
        V, F, wks_mean, Umin, Umax,
        mesh_name="mesh_wks_aniso",
        scalar_name="wks_aniso_mean",
        show_now=True,
        symmetric=True
    )


def run_compare_hks(mesh_name: str,num_eigs: int,num_times: int, alpha_min: float,alpha_max: float,):
    """
    Compare isotropic vs anisotropic HKS side by side.
    """
    V, F = load_mesh(mesh_name)
    print(f"[compare_hks] Loaded mesh: {mesh_name}")
    print(f"              #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    # Isotropic
    print("[compare_hks] Computing isotropic eigenpairs...")
    L_iso, M_iso = cotangent_laplacian(V, F)
    evals_iso, evecs_iso = compute_eigenpairs(L_iso, M_iso, k=num_eigs, method="auto", maxiter=500)
    hks_iso, times_iso = compute_hks(evals_iso, evecs_iso, num_times=num_times)

    # Anisotropic
    print("[compare_hks] Computing anisotropic eigenpairs...")
    Umin, Umax, D = compute_face_principal_directions_and_D(
        V, F, alpha_min=alpha_min, alpha_max=alpha_max
    )
    W_aniso, M_aniso = anisotropic_laplacian(V, F, Umin, Umax, D)
    evals_aniso, evecs_aniso = compute_eigenpairs(W_aniso, M_aniso, k=num_eigs, method="auto", maxiter=500)
    hks_aniso, times_aniso = compute_hks(evals_aniso, evecs_aniso, num_times=num_times)

    # Compare
    diff = np.abs(hks_iso - hks_aniso)
    print(f"[compare_hks] HKS difference:")
    print(f"              Mean: {diff.mean():.6f}")
    print(f"              Max:  {diff.max():.6f}")
    print(f"              Std:  {diff.std():.6f}")

    # Visualize mean HKS
    hks_iso_mean = hks_iso.mean(axis=1)
    hks_aniso_mean = hks_aniso.mean(axis=1)
    diff_mean = np.abs(hks_iso_mean - hks_aniso_mean)

    show_scalar_on_mesh(
        V, F, hks_iso_mean,
        mesh_name="mesh_compare",
        quantity_name="hks_iso_mean",
        show_now=False,
        symmetric=True
    )
    show_scalar_on_mesh(
        V, F, hks_aniso_mean,
        mesh_name="mesh_compare",
        quantity_name="hks_aniso_mean",
        show_now=False,
        symmetric=True
    )
    show_scalar_on_mesh(
        V, F, diff_mean,
        mesh_name="mesh_compare",
        quantity_name="hks_diff",
        show_now=True,
        symmetric=True
    )


def run_hks_boundary( mesh_name: str,num_eigs: int,num_times: int,time_indices: list,alpha_min: float, alpha_max: float,beta_tangent: float,gamma_normal: float,):
    """
    Compute and visualize boundary-aware anisotropic HKS on partial meshes.
    """
    V, F = load_mesh(mesh_name)
    print(f"[hks_boundary] Loaded mesh: {mesh_name}")
    print(f"               #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    # Build boundary-aware tensor
    Umin, Umax, D, d_v, = compute_boundary_aware_tensor(
        V, F,
        beta_tangent=beta_tangent,
        gamma_normal=gamma_normal,
    )

    W, M = anisotropic_laplacian(V, F, Umin, Umax, D)

    print(f"[hks_boundary] Computing {num_eigs} eigenpairs...")
    evals, evecs = compute_eigenpairs(W, M, k=num_eigs, method="auto", maxiter=500)
    print(f"[hks_boundary] Eigenvalue range: [{np.abs(evals).min():.4e}, {np.abs(evals).max():.4e}]")

    # Compute HKS
    hks, times = compute_hks(evals, evecs, num_times=num_times)
    print(f"[hks_boundary] HKS shape: {hks.shape}")
    print(f"[hks_boundary] Time range: [{times[0]:.4e}, {times[-1]:.4e}]")

    # Show boundary distance first
    show_scalar_on_mesh(
        V, F, d_v,
        mesh_name="mesh_hks_boundary",
        quantity_name="dist_to_boundary",
        show_now=False,
        symmetric=True
    )

    # Visualize HKS at different times
    for idx in time_indices:
        if 0 <= idx < num_times:
            scalar = hks[:, idx]
            print(f"[hks_boundary] t={times[idx]:.4e}: min={scalar.min():.4f}, max={scalar.max():.4f}")
            show_scalar_on_mesh(
                V, F, scalar,
                mesh_name="mesh_hks_boundary",
                quantity_name=f"hks_boundary_t{idx}",
                show_now=False,
                symmetric=True
            )

    hks_mean = hks.mean(axis=1)
    show_scalar_and_directions_on_mesh(
        V, F, hks_mean, Umin, Umax,
        mesh_name="mesh_hks_boundary",
        scalar_name="hks_boundary_mean",
        show_now=True,
        symmetric=True
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="HKS/WKS descriptor visualization demo."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "hks_iso", "hks_aniso", "wks_iso", "wks_aniso",
            "compare_hks", "hks_boundary"
        ],
        required=True,
        help="Which demo to run.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to an existing triangle mesh.",
    )
    parser.add_argument(
        "--num-eigs",
        type=int,
        default=100,
        help="Number of eigenpairs to compute.",
    )
    parser.add_argument(
        "--num-times",
        type=int,
        default=16,
        help="Number of time scales for HKS.",
    )
    parser.add_argument(
        "--num-scales",
        type=int,
        default=100,
        help="Number of energy scales for WKS.",
    )
    parser.add_argument(
        "--time-indices",
        type=str,
        default="0,4,8,12,15",
        help="Comma-separated time indices to visualize (for HKS).",
    )
    parser.add_argument(
        "--scale-indices",
        type=str,
        default="0,25,50,75,99",
        help="Comma-separated scale indices to visualize (for WKS).",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=1.0,
        help="Anisotropy parameter along minimum curvature direction.",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=10.0,
        help="Anisotropy parameter along maximum curvature direction.",
    )
    parser.add_argument(
        "--beta-tangent",
        type=float,
        default=2.0,
        help="Anisotropy parameter along boundary tangent (for partial meshes).",
    )
    parser.add_argument(
        "--gamma-normal",
        type=float,
        default=2.0,
        help="Anisotropy parameter along boundary normal (for partial meshes).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mesh_name = args.mesh

    # Parse indices
    time_indices = [int(x) for x in args.time_indices.split(",")]
    scale_indices = [int(x) for x in args.scale_indices.split(",")]

    if args.mode == "hks_iso":
        run_hks_iso(mesh_name, args.num_eigs, args.num_times, time_indices)

    elif args.mode == "hks_aniso":
        run_hks_aniso(
            mesh_name, args.num_eigs, args.num_times, time_indices,
            args.alpha_min, args.alpha_max
        )

    elif args.mode == "wks_iso":
        run_wks_iso(mesh_name, args.num_eigs, args.num_scales, scale_indices)

    elif args.mode == "wks_aniso":
        run_wks_aniso(
            mesh_name, args.num_eigs, args.num_scales, scale_indices,
            args.alpha_min, args.alpha_max
        )

    elif args.mode == "compare_hks":
        run_compare_hks(
            mesh_name, args.num_eigs, args.num_times,
            args.alpha_min, args.alpha_max
        )

    elif args.mode == "hks_boundary":
        run_hks_boundary(
            mesh_name, args.num_eigs, args.num_times, time_indices,
            args.alpha_min, args.alpha_max,
            args.beta_tangent, args.gamma_normal
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()


"""
Usage examples (run from project root):

    # Simple mesh + scalar visualization
    python -m examples.demo_faust --mode view

    # Visualize a isotropic Laplacian eigenfunction with the <index> eigenvalue
    python -m examples.demo_faust --mode eigen --eigen-index 2 --num-eigs 5

    # Visualize a isotropic Laplacian eigenfunction with first four eigenvalues
    python -m examples.demo_faust --mode show_eigen --num-eigs 5

    # Visualize anisotropic Laplacian eigenfunction with first four eigenvalues
    python -m examples.demo_faust  --mode eigen_aniso  --num-eigs 5   --alpha-min 0  --alpha-max 0  

    # Isotropic heat diffusion from the head
    python -m examples.demo_faust --mode heat_iso --time-step 0.005

    # Anisotropic heat diffusion
    python -m examples.demo_faust --mode heat_aniso --time-step 0.05 --alpha-min 0 --alpha-max 10 --src-vertex 4117

    # Anisitropic geodesic distance
    python -m examples.demo_faust --mode geo_aniso --time-step 0.005 --alpha-min 10 --alpha-max 1 --src-vertex 1022


This script assumes:
    - FAUST registrations have been converted to .off and stored as:
        data/meshes/faust_off/tr_reg_000.off
"""

import argparse
import igl
import numpy as np
import polyscope as ps
from operators.io import load_mesh, save_eigs, save_vertex_scalar
from operators.assembly import cotangent_laplacian, anisotropic_laplacian
from operators.solvers import *
from operators.visualization import *
from operators.geometry import compute_face_principal_directions_and_D, vertex_areas_barycentric, build_edge_data_for_divergence


def run_view(mesh_name: str):
    """
    Simple visualization of a FAUST mesh with a trivial scalar field (z-coordinate as height).
    """
    V, F = load_mesh(mesh_name)
    print(f"[view] Loaded mesh: {mesh_name}")
    print(f"       #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    scalar = V[:, 2]  # height as scalar quantity

    show_scalar_on_mesh(
        V,
        F,
        scalar,
        mesh_name="faust_view",
        quantity_name="z_height",
    )


def run_eigen(mesh_name: str, num_eigs: int, eigen_index: int):
    """
    Compute and visualize Laplacian eigenfunctions on a FAUST mesh.
    """
    V, F = load_mesh(mesh_name)
    print(f"[eigen] Loaded mesh: {mesh_name}")
    print(f"        #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    L, M = cotangent_laplacian(V, F)

    # Sanity check: row sums should be ~0 for a Laplacian
    row_sums = np.array(L.sum(axis=1)).ravel()
    print(f"[eigen] Row sum max abs = {np.abs(row_sums).max():.3e} (should be ~0)")

    print(f"[eigen] Solving for {num_eigs} smallest eigenvalues...")
    evals, evecs = compute_eigenpairs(L, M, k=num_eigs, method="mass_diag")
    print("[eigen] First eigenvalues:")
    for i, lam in enumerate(evals):
        print(f"         {i}: {lam:.6e}")

    if not (0 <= eigen_index < num_eigs):
        raise ValueError(
            f"eigen_index={eigen_index} is out of range for num_eigs={num_eigs}"
        )

    scalar = evecs[:, eigen_index]
    print(f"[eigen] Visualizing eigenfunction index {eigen_index}")
    show_scalar_on_mesh(V, F, scalar, mesh_name="faust_eigen", quantity_name=f"eigen_{eigen_index}")

    # Save eigenpairs for later analysis
    base_name = mesh_name.split("/")[-1].replace(".off", "")
    save_eigs(f"{base_name}_top{num_eigs}", evals, evecs)

def run_show_eigen(mesh_name: str, num_eigs: int = 10, num_show: int = 4):
    """
    Visualize multiple eigenfunctions at once on the same FAUST mesh.
    """
    V, F = load_mesh(mesh_name)
    print(f"[show] Loaded mesh: {mesh_name}")
    print(f"       #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    L, M = cotangent_laplacian(V, F)
    evals, evecs = compute_eigenpairs(L, M, k=num_eigs)
    print(f"[show] Computed {num_eigs} eigenpairs.")
    print(f"[show] First few eigenvalues: {evals[:num_show]}")


    for i in range(1, num_show + 1):
        print(f"[show] Registering mode_{i}")
        scalar = evecs[:, i]
        print(f"mode_{i}: min={scalar.min():.3e}, max={scalar.max():.3e}, std={scalar.std():.3e}")
        show_scalar_on_mesh(V, F, scalar, mesh_name="faust_modes", quantity_name=f"mode_{i}", show_now=False)
    ps.show()

def run_eigen_aniso(mesh_name: str, num_eigs: int,alpha_min: float, alpha_max: float, num_show: int = 4, use_fe: bool = False):
    V, F = load_mesh(mesh_name)
    print(f"[eigen_aniso] Loaded mesh: {mesh_name}")
    print(f"              #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    Umin, Umax, D = compute_face_principal_directions_and_D(
        V, F,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    W, M = anisotropic_laplacian(V, F, Umin, Umax, D)

    row_sums = np.array(W.sum(axis=1)).ravel()
    print(f"[eigen_aniso] Row sum max abs = {np.abs(row_sums).max():.3e} (should be ~0)")

    print(f"[eigen_aniso] Solving for {num_eigs} smallest eigenvalues...")
    evals, evecs = compute_eigenpairs(W, M, k=num_eigs, method="mass_diag")
    print("[eigen_aniso] First eigenvalues:")
    for i, lam in enumerate(evals):
        print(f"              {i}: {lam:.6e}")

    for i in range(1, num_show + 1):
        print(f"[eigen_aniso] Registering mode_{i}")
        scalar = evecs[:, i]
        print(f"mode_{i}: min={scalar.min():.3e}, max={scalar.max():.3e}, std={scalar.std():.3e}")
        show_scalar_on_mesh(
            V, F, scalar,
            mesh_name="faust_eigen_aniso_modes",
            quantity_name=f"eigen_aniso_{i}",
            show_now=False
        )

    ps.show()


def run_heat_iso(mesh_name: str, time_step: float, src_vertex: int = -1):
    """
    Isotropic heat diffusion demo on a FAUST mesh.
    """
    V, F = load_mesh(mesh_name)
    print(f"[heat] Loaded mesh: {mesh_name}")
    print(f"       #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    L, M = cotangent_laplacian(V, F)
    
    if 0 <= src_vertex < V.shape[0]:
        print(f"[heat_aniso] Using picked source vertex index = {src_vertex}")
        center = V[src_vertex]
        dist2 = np.sum((V - center) ** 2, axis=1)
        u0 = np.exp(-dist2 / 0.004)
        u0 /= u0.max()

    else:
        print("[heat_aniso] No valid source vertex given, using top in y as default.")
        head_y = np.max(V[:, 1])
        dist2 = (V[:, 1] - head_y) ** 2 
        u0 = np.exp(-dist2 / 0.004)
        u0 /= u0.max()

    print(f"[heat] Performing 1 implicit heat steps with t = {time_step}")
    u1 = u0.copy()
    for _ in range(1):
      u1 = implicit_heat_step(L, M, u1, time_step)
    base_name = mesh_name.split("/")[-1].replace(".off", "")
    save_vertex_scalar(
        name=f"{base_name}_iso_head_t{time_step}",
        V=V,
        F=F,
        scalar=u1,
        subdir="heatmaps",
    )

    show_scalar_on_mesh(
        V,
        F,
        u1,
        mesh_name="faust_heat_iso",
        quantity_name="temperature",
    )


def run_heat_aniso(mesh_name: str, time_step: float, alpha_min: float, alpha_max: float, src_vertex: int = -1):
    """
    Anisotropic heat diffusion demo on a FAUST mesh.
    """
    V, F = load_mesh(mesh_name)
    print(f"[heat_aniso] Loaded mesh: {mesh_name}")
    print(f"             #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    Umin, Umax, D = compute_face_principal_directions_and_D( V, F,alpha_min=alpha_min,alpha_max=alpha_max,)

    W, M = anisotropic_laplacian(V, F, Umin, Umax, D)

    if 0 <= src_vertex < V.shape[0]:
        print(f"[heat_aniso] Using picked source vertex index = {src_vertex}")

        center = V[src_vertex]
        dist2 = np.sum((V - center) ** 2, axis=1)
        u0 = np.exp(-dist2 / 0.001)
        u0 /= u0.max()

    else:
        print("[heat_aniso] No valid source vertex given, using top in y as default.")
        head_y = np.max(V[:, 1])
        dist2 = (V[:, 1] - head_y) ** 2 
        u0 = np.exp(-dist2 / 0.001)
        u0 /= u0.max()


    print(f"[heat_aniso] Performing 1 implicit heat steps with t = {time_step}")
    u1 = u0.copy()
    for _ in range(1):
        u1 = implicit_heat_step(W, M, u1, time_step)


    base_name = mesh_name.split("/")[-1].replace(".off", "")
    save_vertex_scalar(name=f"{base_name}_aniso_head_t{time_step}", V=V, F=F,scalar=u1,subdir="heatmaps",)
    #show_scalar_on_mesh(V,F,u1,mesh_name="faust_heat_aniso",quantity_name="temperature_aniso",)
    show_scalar_and_directions_on_mesh(V, F, u1, Umin, Umax,mesh_name="faust_heat_aniso",scalar_name="temperature_aniso")


def run_geo_iso(mesh_name: str, time_step: float, src_vertex: int = -1):
    V, F = load_mesh(mesh_name)
    print(f"[geo_iso] Loaded mesh: {mesh_name}")
    print(f"         #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    W_iso, M_iso = cotangent_laplacian(V, F)

    N = V.shape[0]

    if 0 <= src_vertex < N:
        print(f"[geo_iso_debug] Using picked source vertex index = {src_vertex}")
        u0 = np.zeros(N)
        u0[src_vertex] = 1.0
        src_from_u0 = src_vertex
    else:
        print("[geo_iso_debug] No valid source; using top in y as default.")
        top_y = np.max(V[:, 1])
        dist2 = (V[:, 1] - top_y) ** 2
        u0 = np.exp(-dist2 / 0.01)
        u0 /= u0.max()
        src_from_u0 = int(np.argmax(u0))
        print("Default src_from_u0:", src_from_u0, "coord:", V[src_from_u0])

    print("u0 stats: min =", float(u0.min()), "max =", float(u0.max()))
    src_from_u0 = int(np.argmax(u0))
    print("argmax u0 index:", src_from_u0, "coord:", V[src_from_u0])

    edge_lengths = igl.edge_lengths(V, F)
    h = edge_lengths.mean()
    t = time_step if time_step > 0 else h**2
    print(f"[geo_iso] Using t = {t:.3e} (h^2 = {h**2:.3e})")

    A_v = vertex_areas_barycentric(V, F)
    edges, cot_sum, e_ij = build_edge_data_for_divergence(V, F)

    phi = heat_geodesic( V, F, W_heat=W_iso,M_heat=M_iso, W_poisson=W_iso, u0=u0,t=t,precomputed=(A_v, edges, cot_sum, e_ij),src_idx=src_from_u0,)
    print("phi stats: min =", float(phi.min()), "max =", float(phi.max()))
    print("phi at src_from_u0:", float(phi[src_from_u0]))
    idx_max = int(np.argmax(phi))
    print("argmax phi index:", idx_max, "coord:", V[idx_max])

    base_name = mesh_name.split("/")[-1].replace(".off", "")
    save_vertex_scalar( name=f"{base_name}_geo_iso_t{t:.3e}", V=V,F=F,scalar=phi,subdir="distances",)
    show_scalar_on_mesh( V, F, phi,mesh_name="faust_geo_iso",quantity_name="geodesic_iso",symmetric=False)


def run_geo_aniso(mesh_name: str, time_step: float, alpha_min: float, alpha_max: float, src_vertex: int = -1):
    V, F = load_mesh(mesh_name)
    print(f"[geo_aniso] Loaded mesh: {mesh_name}")
    print(f"           #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    Umin, Umax, D = compute_face_principal_directions_and_D( V, F, alpha_min=alpha_min, alpha_max=alpha_max)
    W_aniso, M_aniso = anisotropic_laplacian(V, F, Umin, Umax, D)

    # Isotropic Laplacian used in the Poisson step
    W_iso, M_iso = cotangent_laplacian(V, F)

    N = V.shape[0]

    if 0 <= src_vertex < N:
        print(f"[geo_aniso] Using picked source vertex index = {src_vertex}")
        u0 = np.zeros(N)
        u0[src_vertex] = 1.0
        src_from_u0 = src_vertex
    else:
        print("[geo_aniso] No valid source vertex given, using top in y as default.")
        head_y = np.max(V[:, 1])
        dist2 = (V[:, 1] - head_y) ** 2
        u0 = np.exp(-dist2 / 0.001)
        u0 /= u0.max()
        src_from_u0 = int(np.argmax(u0))
        print("Default src_from_u0:", src_from_u0, "coord:", V[src_from_u0])


    print("u0 stats: min =", float(u0.min()), "max =", float(u0.max()))
    src_from_u0 = int(np.argmax(u0))
    print("argmax u0 index:", src_from_u0, "coord:", V[src_from_u0])
    edge_lengths = igl.edge_lengths(V, F)
    h = edge_lengths.mean()
    t = time_step if time_step > 0 else h**2
    print(f"[geo_aniso] Using t = {t:.3e} (h^2 = {h**2:.3e})")

    A_v = vertex_areas_barycentric(V, F)
    edges, cot_sum, e_ij = build_edge_data_for_divergence(V, F)

    phi = heat_geodesic(  V, F,W_heat=W_aniso,M_heat=M_aniso, W_poisson=W_iso, u0=u0, t=t,precomputed=(A_v, edges, cot_sum, e_ij),src_idx=src_from_u0, )
    print("phi stats: min =", float(phi.min()), "max =", float(phi.max()))
    print("phi at src_from_u0:", float(phi[src_from_u0]))
    idx_max = int(np.argmax(phi))
    print("argmax phi index:", idx_max, "coord:", V[idx_max])
    base_name = mesh_name.split("/")[-1].replace(".off", "")
    save_vertex_scalar( name=f"{base_name}_geo_aniso_t{t:.3e}", V=V, F=F,scalar=phi, subdir="distances",)
    show_scalar_on_mesh(V, F, phi ,mesh_name="faust_geo_aniso",quantity_name="geodesic_aniso",symmetric=False)




def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified FAUST demo: view / eigen / heat /geodesic."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["view", "eigen", "heat_iso", "show_eigen", "heat_aniso", "eigen_aniso", "eigen_aniso_fe", 'geo_iso', 'geo_aniso',],
        required=True,
        help="Which demo to run: 'view', 'eigen', 'show_eigen', 'eigen_aniso', 'eigen_aniso_fe','heat_iso', 'heat_aniso','geo_iso', 'geo_aniso',",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to an existing full triangle mesh.",
    )
    parser.add_argument(
        "--num-eigs",
        type=int,
        default=10,
        help="Number of eigenpairs to compute in 'eigen' mode.",
    )
    parser.add_argument(
        "--eigen-index",
        type=int,
        default=2,
        help="Which eigenfunction to visualize in 'eigen' mode (0-based index).",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.005,
        help="Time step t for the implicit heat step in 'heat' mode.",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=10.0,
        help="Anisotropy parameter along minimum curvature direction.",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=10.0,
        help="Anisotropy parameter along maximum curvature direction.",
    )
    parser.add_argument(
        "--src-vertex",
        type=int,
        default=-1,
        help="Source vertex index for the heat / geodesic methods. "
             "If < 0, a default source (e.g. top in y) is used."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mesh_name = args.mesh

    if args.mode == "view":
        run_view(mesh_name)
    elif args.mode == "eigen":
        run_eigen(mesh_name, num_eigs=args.num_eigs, eigen_index=args.eigen_index)
    elif args.mode == "eigen_aniso":
        run_eigen_aniso( mesh_name,num_eigs=args.num_eigs,alpha_min=args.alpha_min,alpha_max=args.alpha_max,use_fe=False,   )

    elif args.mode == "eigen_aniso_fe":
        run_eigen_aniso( mesh_name, num_eigs=args.num_eigs,alpha_min=args.alpha_min,alpha_max=args.alpha_max,use_fe=True,)
    elif args.mode == "heat_iso":
        run_heat_iso(mesh_name, time_step=args.time_step, src_vertex=args.src_vertex)
    elif args.mode == "show_eigen":
        run_show_eigen(mesh_name, num_eigs=args.num_eigs, num_show=4)
    elif args.mode == "heat_aniso":
        run_heat_aniso(mesh_name,time_step=args.time_step, alpha_min=args.alpha_min,alpha_max=args.alpha_max,src_vertex= args.src_vertex)
    elif args.mode == "geo_iso":
        run_geo_iso( mesh_name, time_step=args.time_step, src_vertex= args.src_vertex)
    elif args.mode == "geo_aniso":
        run_geo_aniso(mesh_name,time_step=args.time_step, alpha_min=args.alpha_min,alpha_max=args.alpha_max,src_vertex=args.src_vertex)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

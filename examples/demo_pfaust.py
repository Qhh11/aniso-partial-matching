#examples/demo_pfaust.py

"""
 python -m examples.demo_pfaust   --mode heat_aniso_boundary --time-step 0.0005  --mesh PFAUST-M/off/tr_reg_000.off --beta-tangent 2 --gamma-normal 2  --src-vertex 2461   
 #alpha min max not used in boundary version
"""
import heapq
import argparse
import numpy as np
import igl

from scipy.sparse.linalg import eigsh, spsolve
from scipy.sparse import diags

from operators.io import load_mesh, save_vertex_scalar
from operators.assembly import cotangent_laplacian, anisotropic_laplacian
from operators.visualization import show_scalar_on_mesh, show_scalar_and_directions_on_mesh
from operators.solvers import *
from operators.boundary import (
    compute_boundary_aware_tensor,
    compute_boundary_distance_fast_marching,
    compute_distance_gradient_direction_vertex,
)



def implicit_heat_step(L, M, u0, t):
    """
    One implicit heat step:
        (M - t L) u1 = M u0
    """
    A = M - t * L
    b = M @ u0
    u1 = spsolve(A, b)

    return u1


def run_view_boundary(mesh_name: str, vec_on: str = "faces"):
    V, F = load_mesh(mesh_name)
    print(f"[view_boundary] Loaded mesh: {mesh_name}")
    print(f"               #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    if vec_on == "vertices":
        U_tan, U_ortho, D, d_v = compute_boundary_aware_tensor(V, F, distance_method = "dijkstra", return_on="vertices")
    else:
        U_tan, U_ortho, D, d_v = compute_boundary_aware_tensor(V, F, distance_method = "dijkstra",return_on="faces")

    show_scalar_and_directions_on_mesh(
        V,
        F,
        scalar=d_v,
        Umin=U_tan,
        Umax=U_ortho,
        mesh_name="pfaust_boundary",
        scalar_name="dist_to_boundary",
        vec_on=vec_on,
    )

def _unwrap_igl_output(x):
    return x[0] if isinstance(x, tuple) else x


def _normalize_rows(X: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    good = n[:, 0] > eps
    Y = X.copy()
    Y[good] /= n[good]
    return Y


def run_view_fmm_direction(mesh_name: str, direction_sign: str = "to_boundary"):
    """
    Visualize the raw fast-marching direction field on a partial PFAUST mesh.

    d_v is zero at the boundary and increases inward.
    grad(d_v) points inward.
    -grad(d_v) points back to the boundary.

    direction_sign:
        "to_boundary" : visualize R_v = -grad(d_v)
        "inward"      : visualize R_v =  grad(d_v)
    """
    V, F = load_mesh(mesh_name)
    F = F.astype(np.int32, copy=False)

    print(f"[view_fmm_direction] Loaded mesh: {mesh_name}")
    print(f"                     #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    # 1) Fast marching distance from mesh boundary
    d_v, boundary_vertices, boundary_edges = compute_boundary_distance_fast_marching(V, F)

    print(f"[view_fmm_direction] boundary vertices = {len(boundary_vertices)}")
    print(f"[view_fmm_direction] boundary edges    = {len(boundary_edges)}")
    print(
        "[view_fmm_direction] d_v range = "
        f"[{float(d_v.min()):.6e}, {float(d_v.max()):.6e}]"
    )

    # 2) Vertex normals
    n_v = _unwrap_igl_output(igl.per_vertex_normals(V, F))

    # 3) Raw FMM gradient-based direction
    # This returns -grad(d_v), so it points toward the nearest boundary.
    R_to_boundary = compute_distance_gradient_direction_vertex(
        V,
        F,
        d_v=d_v,
        n_v=n_v,
    )

    if direction_sign == "to_boundary":
        R_v = R_to_boundary
        direction_name = "fmm_R_to_boundary"
    elif direction_sign == "inward":
        R_v = -R_to_boundary
        direction_name = "fmm_R_inward"
    else:
        raise ValueError(
            f"Unknown direction_sign={direction_sign!r}. "
            "Use 'to_boundary' or 'inward'."
        )

    # 4) Tangential companion direction
    T_v = np.cross(n_v, R_v)
    T_v = _normalize_rows(T_v)

    # 5) Debug checks
    r_norm = np.linalg.norm(R_v, axis=1)
    t_norm = np.linalg.norm(T_v, axis=1)
    rn_dot = np.abs(np.sum(R_v * n_v, axis=1))
    rt_dot = np.abs(np.sum(R_v * T_v, axis=1))

    print(
        "[view_fmm_direction] |R| range = "
        f"[{float(r_norm.min()):.6f}, {float(r_norm.max()):.6f}], "
        f"mean={float(r_norm.mean()):.6f}"
    )
    print(
        "[view_fmm_direction] |T| range = "
        f"[{float(t_norm.min()):.6f}, {float(t_norm.max()):.6f}], "
        f"mean={float(t_norm.mean()):.6f}"
    )
    print(
        "[view_fmm_direction] max |R dot n| = "
        f"{float(rn_dot.max()):.3e}  should be close to 0"
    )
    print(
        "[view_fmm_direction] max |R dot T| = "
        f"{float(rt_dot.max()):.3e}  should be close to 0"
    )

    # 6) Visualize on vertices
    # Umin = T_v, Umax = R_v only for visualization here.
    show_scalar_and_directions_on_mesh(
        V,
        F,
        scalar=d_v,
        Umin=T_v,
        Umax=R_v,
        mesh_name="pfaust_fmm_direction",
        scalar_name=direction_name,
        vec_on="vertices",
    )


def run_eigen_aniso_boundary(mesh_name: str, num_eigs: int, num_show: int = 10):
    """
    Compute anisotropic Laplacian eigenpairs on a partial mesh using
    boundary-aware tensor.
    """
    V, F = load_mesh(mesh_name)
    print(f"[eigen_aniso_boundary] Loaded mesh: {mesh_name}")
    print(f"                       #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    Umin, Umax, D, _ =  compute_boundary_aware_tensor(V, F)
    W, M = anisotropic_laplacian(V, F, Umin, Umax, D)

    row_sums = np.array(W.sum(axis=1)).ravel()
    print(f"[eigen_aniso_boundary] Row sum max abs = {np.abs(row_sums).max():.3e} (should be ~0)")

    print(f"[eigen_aniso_boundary] Solving for {num_eigs} smallest eigenvalues...")
    evals, evecs = compute_eigenpairs(W, M, k=num_eigs, which="SM")
    print("[eigen_aniso_boundary] First eigenvalues:")
    for i, lam in enumerate(evals):
        print(f"    {i}: {lam:.6e}")

    # visualize a few non-trivial modes (skip index 0 if constant)
    for i in range(1, min(num_show, num_eigs)):
        scalar = evecs[:, i]
        print(f"[eigen_aniso_boundary] mode_{i}: min={scalar.min():.3e}, max={scalar.max():.3e}, std={scalar.std():.3e}")
        show_scalar_on_mesh(
            V,
            F,
            scalar,
            mesh_name="pfaust_eigen_aniso_boundary",
            quantity_name=f"eigen_aniso_boundary_{i}",
            show_now=False,
        )

    import polyscope as ps
    ps.show()


def run_heat_aniso_boundary(mesh_name: str, time_step: float, beta_tangent: float = 2.0, gamma_normal: float = 2.0, src_vertex: int = -1, vec_on: str = "vertices",):
    """
    nisotropic heat diffusion on a PARTIAL mesh using curvature + boundary blended tensor.
    """
    V, F = load_mesh(mesh_name)
    print(f"[heat_aniso_boundary] Loaded mesh: {mesh_name}")
    print(f"                      #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    Umin, Umax, D, d_v, = compute_boundary_aware_tensor( V, F, beta_tangent=beta_tangent,gamma_normal=gamma_normal, return_on = vec_on)

    W, M = anisotropic_laplacian(V, F, Umin, Umax, D)

    N = V.shape[0]
    if 0 <= src_vertex < N:
        print(f"[heat_aniso_boundary] Using picked source vertex index = {src_vertex}")
        center = V[src_vertex]
        dist2 = np.sum((V - center) ** 2, axis=1)
        u0 = np.exp(-dist2 / 0.004)
        u0 /= u0.max()
    else:
        print("[heat_aniso_boundary] No valid source; using top in y as default.")
        top_y = np.max(V[:, 1])
        dist2 = (V[:, 1] - top_y) ** 2
        u0 = np.exp(-dist2 / 0.004)
        u0 /= u0.max()

    h = igl.edge_lengths(V, F).mean()
    time_step = time_step if time_step > 0 else 1.0 * h * h

    print(f"[heat_aniso_boundary] Performing 1 implicit heat step with t = {time_step}")
    u1 = implicit_heat_step(W, M, u0, time_step)

    base_name = mesh_name.split("/")[-1].rsplit(".", 1)[0]
    save_vertex_scalar(
        name=f"{base_name}_aniso_boundary_heat_t{time_step}",
        V=V,
        F=F,
        scalar=u1,
        subdir="heatmaps",
    )

    show_scalar_and_directions_on_mesh(
        V,
        F,
        scalar=u1,
        Umin=Umin,
        Umax=Umax,
        mesh_name="pfaust_heat_aniso_boundary",
        scalar_name="temperature_aniso_boundary",
        vec_on=vec_on,
    )

def run_heat_iso_boundary(mesh_name: str, time_step: float, src_vertex: int = -1):
    """
    Isotropic heat diffusion on a PARTIAL mesh (i.e., with boundary).
    Uses standard cotangent Laplacian (no boundary-aware tensor).
    """
    V, F = load_mesh(mesh_name)
    print(f"[heat_iso_boundary] Loaded mesh: {mesh_name}")
    print(f"                     #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    L_iso, M_iso = cotangent_laplacian(V, F)

    N = V.shape[0]
    if 0 <= src_vertex < N:
        print(f"[heat_iso_boundary] Using picked source vertex index = {src_vertex}")
        center = V[src_vertex]
        dist2 = np.sum((V - center) ** 2, axis=1)
        u0 = np.exp(-dist2 / 0.004)
        u0 /= u0.max()
    else:
        print("[heat_iso_boundary] No valid source; using top in y as default.")
        top_y = np.max(V[:, 1])
        dist2 = (V[:, 1] - top_y) ** 2
        u0 = np.exp(-dist2 / 0.004)
        u0 /= u0.max()

    h = igl.edge_lengths(V, F).mean()
    time_step = time_step if time_step > 0 else 1.0 * h * h

    print(f"[heat_iso_boundary] Performing 1 implicit heat step with t = {time_step}")
    u1 = implicit_heat_step(L_iso, M_iso, u0, time_step)

    base_name = mesh_name.split("/")[-1].rsplit(".", 1)[0]
    save_vertex_scalar(
        name=f"{base_name}_iso_boundary_heat_t{time_step}",
        V=V,
        F=F,
        scalar=u1,
        subdir="heatmaps",
    )

    show_scalar_on_mesh(
        V,
        F,
        u1,
        mesh_name="pfaust_heat_iso_boundary",
        quantity_name="temperature_iso_boundary",
    )

def run_geo_aniso_boundary(mesh_name: str,time_step: float, beta_tangent: float = 2.0,gamma_normal: float = 2.0, src_vertex: int = -1,):
    """
    Anisotropic geodesic distance on a PARTIAL mesh using the
    curvature + boundary blended tensor.

    This is the boundary-aware counterpart of `run_geo_aniso` on full FAUST.
    """
    V, F = load_mesh(mesh_name)
    print(f"[geo_aniso_boundary] Loaded mesh: {mesh_name}")
    print(f"                    #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    Umin, Umax, D, d_v, = compute_boundary_aware_tensor( V, F, beta_tangent=beta_tangent, gamma_normal=gamma_normal,)

    W_aniso, M_aniso = anisotropic_laplacian(V, F, Umin, Umax, D)

    W_iso, M_iso = cotangent_laplacian(V, F)

    N = V.shape[0]

    if 0 <= src_vertex < N:
        print(f"[geo_aniso_boundary] Using picked source vertex index = {src_vertex}")
        center = V[src_vertex]
        dist2 = np.sum((V - center) ** 2, axis=1)
        u0 = np.exp(-dist2 / 0.001)
        u0 /= u0.max()
    else:
        print("[geo_aniso_boundary] No valid source; using top in y as default.")
        top_y = np.max(V[:, 1])
        dist2 = (V[:, 1] - top_y) ** 2
        u0 = np.exp(-dist2 / 0.001)
        u0 /= u0.max()
        src_vertex = int(np.argmax(u0))
        print(f"[geo_aniso_boundary] Implied source index = {src_vertex}, coord = {V[src_vertex]}")

    print("u0 stats: min =", float(u0.min()), "max =", float(u0.max()))
    src_from_u0 = int(np.argmax(u0))
    print("argmax u0 index:", src_from_u0, "coord:", V[src_from_u0])

    edge_lengths = igl.edge_lengths(V, F)
    h = edge_lengths.mean()
    t = time_step if time_step > 0 else h**2
    print(f"[geo_aniso_boundary] Using t = {t:.3e} (h^2 = {h**2:.3e})")

    A_v = vertex_areas_barycentric(V, F)
    edges, cot_sum, e_ij = build_edge_data_for_divergence(V, F)

    phi = heat_geodesic( V, F,W_heat=W_aniso, M_heat=M_aniso,W_poisson=W_iso, u0=u0, t=t, precomputed=(A_v, edges, cot_sum, e_ij), src_idx=src_from_u0,)

    print("phi stats: min =", float(phi.min()), "max =", float(phi.max()))
    print("phi at src_from_u0:", float(phi[src_from_u0]))
    idx_max = int(np.argmax(phi))
    print("argmax phi index:", idx_max, "coord:", V[idx_max])

    base_name = mesh_name.split("/")[-1].rsplit(".", 1)[0]
    save_vertex_scalar(
        name=f"{base_name}_geo_aniso_boundary_t{t:.3e}",
        V=V,
        F=F,
        scalar=phi,
        subdir="distances",
    )

    show_scalar_on_mesh( V, F, phi,mesh_name="pfaust_geo_aniso_boundary",quantity_name="geodesic_aniso_boundary",symmetric=False,)

def run_geo_iso_partial(mesh_name: str, time_step: float, src_vertex: int = -1):
    V, F = load_mesh(mesh_name)
    print(f"[geo_iso_partial] Loaded mesh: {mesh_name}")
    print(f"                 #vertices = {V.shape[0]}, #faces = {F.shape[0]}")

    W_iso, M_iso = cotangent_laplacian(V, F)
    N = V.shape[0]

    if 0 <= src_vertex < N:
        print(f"[geo_iso_partial] Using picked source vertex index = {src_vertex}")
        center = V[src_vertex]
        dist2 = np.sum((V - center) ** 2, axis=1)
        u0 = np.exp(-dist2 / 0.01)
        u0 /= u0.max()
    else:
        print("[geo_iso_partial] No valid source; using top in y as default.")
        top_y = np.max(V[:, 1])
        dist2 = (V[:, 1] - top_y) ** 2
        u0 = np.exp(-dist2 / 0.01)
        u0 /= u0.max()

    print("u0 stats: min =", float(u0.min()), "max =", float(u0.max()))
    src_from_u0 = int(np.argmax(u0))
    print("argmax u0 index:", src_from_u0, "coord:", V[src_from_u0])

    edge_lengths = igl.edge_lengths(V, F)
    h = edge_lengths.mean()
    t = time_step if time_step > 0 else h * h
    print(f"[geo_iso_partial] Using t = {t:.3e} (h^2 = {h**2:.3e})")

    A_v = vertex_areas_barycentric(V, F)
    edges, cot_sum, e_ij = build_edge_data_for_divergence(V, F)

    phi = heat_geodesic(
        V, F,
        W_heat=W_iso,
        M_heat=M_iso,
        W_poisson=W_iso,
        u0=u0,
        t=t,
        precomputed=(A_v, edges, cot_sum, e_ij),
        src_idx=src_from_u0,
    )
    print("phi stats: min =", float(phi.min()), "max =", float(phi.max()))
    print("phi at src_from_u0:", float(phi[src_from_u0]))

    base_name = mesh_name.split("/")[-1].rsplit(".", 1)[0]
    save_vertex_scalar(
        name=f"{base_name}_geo_iso_partial_t{t:.3e}",
        V=V,
        F=F,
        scalar=phi,
        subdir="distances",
    )

    show_scalar_on_mesh(
        V,
        F,
        phi,
        mesh_name="pfaust_geo_iso_partial",
        quantity_name="geodesic_iso_partial",
        symmetric=False,
    )



def parse_args():
    parser = argparse.ArgumentParser(
        description="PFAUST demo: boundary-aware anisotropic Laplacian on partial meshes."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["view_boundary", "view_fmm_direction","eigen_aniso_boundary", "heat_aniso_boundary", "heat_iso_boundary", "geo_aniso_boundary", "geo_iso_boundary",],
        required=True,
        help="Which demo to run.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to an existing partial triangle mesh.",
    )
    parser.add_argument(
        "--num-eigs",
        type=int,
        default=10,
        help="Number of eigenpairs in 'eigen_aniso_boundary' mode.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.005,
        help="Time step t for implicit heat in 'heat_aniso_boundary' mode.",
    )
    parser.add_argument(
        "--src-vertex",
        type=int,
        default=-1,
        help="Source vertex for heat; if <0, use default at top in y.",
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
        "--beta-tangent",
        type=float,
        default=2.0,
        help="Anisotropy parameter along boundary tangent direction.",
    )
    parser.add_argument(
        "--gamma-normal",
        type=float,
        default=2.0,
        help="Anisotropy parameter along boundary normal direction.",
    )
    parser.add_argument(
        "--direction-sign",
        type=str,
        default="to_boundary",
        choices=["to_boundary", "inward"],
        help=(
            "Direction sign for view_fmm_direction. "
            "'to_boundary' visualizes -grad(d), "
            "'inward' visualizes grad(d)."
        ),
    )
    parser.add_argument(
        "--vec-on",
        type=str,
        default="faces",
        choices=["faces", "vertices"],
        help="Visualize direction field on faces or vertices.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    mesh_name = args.mesh

    if args.mode == "view_boundary":
        run_view_boundary(mesh_name, vec_on=args.vec_on)
    elif args.mode == "view_fmm_direction":
        run_view_fmm_direction(
            mesh_name,
            direction_sign=args.direction_sign,
        )
    elif args.mode == "eigen_aniso_boundary":
        run_eigen_aniso_boundary(mesh_name, num_eigs=args.num_eigs)
    elif args.mode == "heat_aniso_boundary":
        run_heat_aniso_boundary(mesh_name, time_step=args.time_step, beta_tangent=args.beta_tangent, gamma_normal=args.gamma_normal, src_vertex=args.src_vertex, vec_on=args.vec_on)
    elif args.mode == "geo_aniso_boundary":
        run_geo_aniso_boundary( mesh_name,time_step=args.time_step, beta_tangent=args.beta_tangent, gamma_normal=args.gamma_normal,src_vertex=args.src_vertex,)
    elif args.mode == "geo_iso_boundary":
        run_geo_iso_partial( mesh_name,time_step=args.time_step, src_vertex=args.src_vertex,)
    elif args.mode == "heat_iso_boundary":
        run_heat_iso_boundary( mesh_name,time_step=args.time_step,src_vertex=args.src_vertex,)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

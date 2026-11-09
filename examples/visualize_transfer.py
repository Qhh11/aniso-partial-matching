"""Inspect a partial boundary field and the transported full tensor."""
import argparse
from pathlib import Path
import numpy as np
import yaml
from functional_correspondence.dataset import read_pairs, load_pair
from operators.pipeline import pair_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--pair-index', type=int, default=0)
    parser.add_argument('--config', default='configs/fm_mse.yaml')
    parser.add_argument('--output', default='outputs/transfer.npz')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    row = read_pairs(args.pairs)[args.pair_index]
    vp, fp, vf, ff, gt, soft, hard = load_pair(row)
    cfg = yaml.safe_load(Path(args.config).read_text())
    (tp, dp), (tf, df), debug = pair_tensors(vp, fp, vf, ff, cfg['operators'], gt, soft, hard)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, partial_directions=tp, full_directions=tf,
                        partial_eigenvalues=dp, full_eigenvalues=df,
                        boundary_distance=debug['distance'], coverage=debug['coverage'])
    if not args.no_show:
        import polyscope as ps
        ps.init()
        for name, v, f, t, d in [('partial', vp, fp, tp, dp), ('full', vf, ff, tf, df)]:
            mesh = ps.register_surface_mesh(name, v, f)
            mesh.add_vector_quantity('tangent direction', t, enabled=True)
            mesh.add_scalar_quantity('normal diffusivity', d[:, 0])
            mesh.add_scalar_quantity('tangent diffusivity', d[:, 1])
        ps.show()


if __name__ == '__main__':
    main()

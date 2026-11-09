"""Correspondence colors, checkerboard transfer and reference-sample errors."""
import argparse
import numpy as np
from functional_correspondence.dataset import read_pairs, load_pair, array, mesh_area
from functional_correspondence.metrics import geodesic_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--pair-index', type=int, default=0)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--comparison', help='Optional second p2p map for signed error difference')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    row = read_pairs(args.pairs)[args.pair_index]
    vp, fp, vf, ff, gt, _, _ = load_pair(row)
    prediction = array(args.prediction).astype('int64').reshape(-1)
    if len(prediction) != len(vp) or prediction.min() < 0 or prediction.max() >= len(vf):
        raise ValueError('Prediction must map each partial vertex to a full vertex')
    colors = (vf-vf.min(0)) / np.maximum(np.ptp(vf, axis=0), 1e-12)
    checker = (np.floor(colors[:, 0]*12) + np.floor(colors[:, 1]*12)) % 2
    errors = delta = None
    if gt is not None and row.get('full_dist'):
        distances = array(row['full_dist'])
        errors = geodesic_errors(prediction, gt, distances, mesh_area(vf, ff))
        print(f'Mean normalized geodesic error: {errors.mean():.6f}')
        if args.comparison:
            other = geodesic_errors(array(args.comparison), gt, distances, mesh_area(vf, ff))
            delta = errors-other
    if args.no_show:
        return
    import polyscope as ps
    ps.init()
    full = ps.register_surface_mesh('full', vf, ff)
    part = ps.register_surface_mesh('partial', vp, fp)
    full.add_color_quantity('position colors', colors, enabled=True)
    part.add_color_quantity('transferred colors', colors[prediction], enabled=True)
    full.add_scalar_quantity('checkerboard', checker)
    part.add_scalar_quantity('transferred checkerboard', checker[prediction])
    if errors is not None:
        # Display exactly the evaluated reference samples; never fill unknown
        # partial vertices with apparently measured errors.
        samples = ps.register_point_cloud('GT reference samples', vp[gt[0]])
        samples.add_scalar_quantity('geodesic error', errors, enabled=True)
        if delta is not None:
            samples.add_scalar_quantity('prediction minus comparison', delta)
    ps.show()


if __name__ == '__main__':
    main()

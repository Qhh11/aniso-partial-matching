"""Shared non-learning comparison driver for same-pose and cross-pose examples."""
import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from operators.descriptors import compute_hks
from functional_correspondence.dataset import read_pairs, load_pair, array, mesh_area
from functional_correspondence.preparation import prepare_pair, read_prepared
from functional_correspondence.decoding import cosine_nn, solve_feature_fmap, fm_decode
from functional_correspondence.metrics import geodesic_errors, summarize, pck


def main(pairing):
    parser = argparse.ArgumentParser(description=f'Non-learning ISO/ANISO HKS comparison: {pairing} pose')
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--config', default='configs/fm_mse.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', default='cache/examples')
    parser.add_argument('--decoder', choices=('nn', 'fm'), default='nn')
    parser.add_argument('--t-min', type=float, default=.01)
    parser.add_argument('--t-max', type=float, default=1.)
    parser.add_argument('--times', type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.t_min < args.t_max or args.times < 2:
        raise ValueError('Use 0 < t_min < t_max and at least two times')
    rows = read_pairs(args.pairs)
    if any(row.get('pairing') != pairing for row in rows):
        raise ValueError(f'Explicitly mark every manifest row with pairing={pairing}')
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    times = np.geomspace(args.t_min, args.t_max, args.times)
    np.save(out / 'hks_times.npy', times)
    summaries, per_pair = [], []
    for kind in ('iso', 'aniso'):
        current = copy.deepcopy(cfg)
        current['operators']['kind'] = kind
        collected = []
        for row in rows:
            data = read_prepared(prepare_pair(row, current, args.cache), 'cpu')
            p, f = data['partial'], data['full']
            hp, _ = compute_hks(p['evals'].numpy(), p['evecs'].numpy(), times=times)
            hf, _ = compute_hks(f['evals'].numpy(), f['evecs'].numpy(), times=times)
            hp, hf = torch.from_numpy(hp), torch.from_numpy(hf)
            if args.decoder == 'nn':
                prediction = cosine_nn(hp, hf).numpy()
            else:
                k = cfg['model']['fmap_k']
                C = solve_feature_fmap(hp, hf, p['evals'][:k], f['evals'][:k],
                    p['evecs'][:, :k], f['evecs'][:, :k], p['mass'], f['mass'])
                prediction = fm_decode(C, p['evecs'][:, :k], f['evecs'][:, :k], 'partial_to_full')
            np.save(out / f'{row["id"]}_{kind}_p2p.npy', prediction)
            _, _, vf, ff, gt, _, _ = load_pair(row)
            if gt is None or not row.get('full_dist'):
                raise ValueError('Comparison metrics require GT and full_dist in every row')
            errors = geodesic_errors(prediction, gt, array(row['full_dist']), mesh_area(vf, ff))
            np.save(out / f'{row["id"]}_{kind}_errors.npy', errors)
            per_pair.append(dict(pair=row['id'], method=kind, **summarize(errors)))
            collected.append(errors)
        errors = np.concatenate(collected)
        summaries.append(dict(method=kind, **summarize(errors)))
        thresholds = np.linspace(0, .5, 501)
        np.savetxt(out / f'{kind}_pck.csv', np.column_stack((thresholds, pck(errors, thresholds))),
                   delimiter=',', header='threshold,pck', comments='')
    for name, values in (('summary.csv', summaries), ('pairs.csv', per_pair)):
        with (out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (out / 'settings.json').write_text(json.dumps(dict(config=cfg, decoder=args.decoder,
        pairing=pairing, candidates='all_full_vertices', hks_times=times.tolist()), indent=2)+'\n')
    print(json.dumps(summaries, indent=2))

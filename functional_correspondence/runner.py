"""Small single-pair training/evaluation driver, independent of experiment folders."""
import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from .dataset import read_pairs, load_pair, array, mesh_area
from .model import FeatureModel
from .preparation import prepare_pair, read_prepared, source_digest
from .losses import Objective, fmap_prediction
from .decoding import cosine_nn, project_correspondence, fm_decode
from .metrics import geodesic_errors, summarize, pck


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_geometry(row, p, f, device):
    result = {}
    for name in ('full_dist', 'partial_dist', 'partial_mass', 'mask'):
        key = 'train_' + name
        if not row.get(key):
            raise ValueError(f'WH training needs {key}; do not substitute raw evaluation distances')
        result[name] = torch.tensor(array(row[key]), dtype=torch.float32, device=device)
    npart, nfull = len(p['vertices']), len(f['vertices'])
    expected = dict(full_dist=(nfull,nfull), partial_dist=(npart,npart),
                    partial_mass=(npart,), mask=(npart,npart))
    for name, value in result.items():
        if tuple(value.shape) != expected[name] or not torch.isfinite(value).all():
            raise ValueError(f'Invalid WH input {name}: expected finite {expected[name]}')
    return result


def features(model, p, f, cfg):
    if cfg['objective'] == 'wormhole':
        ff = model(f)
        fp = model(p)
    else:
        fp = model(p)
        ff = model(f)
    return fp, ff


def train(cfg, rows, prepared, out, device):
    # Geometry preparation must not consume network initialization RNG.
    seed_everything(cfg.get('seed', 1234))
    model = FeatureModel(cfg).to(device)
    objective = Objective(cfg)
    opt = cfg['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.get('lr', 1e-3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.get('scheduler_period', 300), eta_min=opt.get('min_lr', 1e-4))
    if (out / 'last.pt').exists():
        raise FileExistsError('Choose a fresh output directory; this command never overwrites a trained run')
    with (out / 'loss.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'loss', 'lr'])
        for epoch in range(1, opt['epochs']+1):
            model.train()
            # Reproducible shuffling of the explicitly supplied manifest order.
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(epoch)).tolist()
            total = 0.
            for index in order:
                row = rows[index]
                data = read_prepared(prepared[index], device)
                p, f = data['partial'], data['full']
                _, _, _, _, gt, _, _ = load_pair(row)
                refs = None if gt is None else tuple(torch.as_tensor(i, device=device) for i in gt)
                geometry = train_geometry(row, p, f, device) if cfg['objective'] == 'wormhole' else None
                optimizer.zero_grad(set_to_none=True)
                fp, ff = features(model, p, f, cfg)
                terms = objective(fp, ff, p, f, refs, geometry)
                loss = sum(terms.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss on {row["id"]}')
                loss.backward()
                if any(x.grad is not None and not torch.isfinite(x.grad).all() for x in model.parameters()):
                    raise FloatingPointError(f'Non-finite gradient on {row["id"]}')
                optimizer.step()
                total += loss.detach().item()
            writer.writerow([epoch, total/len(rows), optimizer.param_groups[0]['lr']])
            stream.flush()
            scheduler.step()
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                              scheduler=scheduler.state_dict(), epoch=epoch, config=cfg,
                              source_sha256=source_digest())
            temporary = out / 'last.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(out / 'last.pt')
            print(f'Epoch {epoch}/{opt["epochs"]}: loss={total/len(rows):.6g}', flush=True)


def evaluate(cfg, rows, prepared, out, device, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for key in ('model', 'operators', 'objective'):
        if cfg[key] != state['config'][key]:
            raise ValueError(f'Evaluation {key} differs from training checkpoint')
    model = FeatureModel(cfg).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    metrics, all_errors = [], []
    with torch.no_grad():
        for row, path in zip(rows, prepared):
            data = read_prepared(path, device)
            p, f = data['partial'], data['full']
            fp, ff = features(model, p, f, cfg)
            pred_nn = cosine_nn(fp, ff).cpu().numpy()
            if cfg['objective'] == 'fm_mse':
                k = cfg['model']['fmap_k']
                C = fmap_prediction(fp, ff, p, f, cfg)
                pred = fm_decode(C, p['evecs'][:, :k], f['evecs'][:, :k], 'partial_to_full')
            else:
                # Official WH evaluation: hard feature correspondence -> FM -> p2p.
                C = (p['iso_evecs'].T * p['iso_mass'][None]) @ f['iso_evecs'][pred_nn]
                pred = fm_decode(C, p['iso_evecs'], f['iso_evecs'])
            np.save(out / f'{row["id"]}_p2p.npy', pred)
            np.save(out / f'{row["id"]}_nn.npy', pred_nn)
            _, _, vf, ff_faces, gt, _, _ = load_pair(row)
            if gt is not None and row.get('full_dist'):
                errors = geodesic_errors(pred, gt, array(row['full_dist']), mesh_area(vf, ff_faces))
                np.save(out / f'{row["id"]}_errors.npy', errors)
                metrics.append(dict(pair=row['id'], **summarize(errors)))
                all_errors.append(errors)
    if metrics:
        with (out / 'pairs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        errors = np.concatenate(all_errors)
        summary = summarize(errors)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        thresholds = np.linspace(0, .5, 501)
        np.savetxt(out / 'pck.csv', np.column_stack((thresholds, pck(errors, thresholds))),
                   delimiter=',', header='threshold,pck', comments='')
        print(json.dumps(summary, indent=2))
    else:
        print('Saved predictions. Metrics require GT references and raw target distances.')


def main(mode):
    parser = argparse.ArgumentParser(description=f'{mode.capitalize()} partial-to-full matching')
    parser.add_argument('--config', required=True)
    parser.add_argument('--pairs', required=True, help='Explicit CSV manifest; paths are relative to its directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cache', default='cache/operators')
    parser.add_argument('--prepare-only', action='store_true')
    if mode == 'inference':
        parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rows = read_pairs(args.pairs)
    if cfg['model']['fmap_k'] > cfg['operators']['k_eig'] and cfg['objective'] == 'fm_mse':
        raise ValueError('FM basis cannot exceed the number of diffusion eigenpairs')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new, empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.get('seed', 1234))
    prepared = [prepare_pair(row, cfg, args.cache) for row in rows]
    (out / 'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    if args.prepare_only:
        print(f'Prepared {len(prepared)} pairs; no optimization performed')
    elif mode == 'train':
        train(cfg, rows, prepared, out, torch.device(args.device))
    else:
        evaluate(cfg, rows, prepared, out, torch.device(args.device), args.checkpoint)

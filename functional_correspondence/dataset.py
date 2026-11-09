"""Explicit pair manifests: no machine-specific dataset roots or implicit splits."""
import csv
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from operators.io import load_mesh

PATH_FIELDS = ('partial_mesh', 'full_mesh', 'gt', 'soft', 'hard', 'full_dist',
               'train_full_dist', 'train_partial_dist', 'train_partial_mass', 'train_mask')


def read_pairs(path):
    path = Path(path).resolve()
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty pair manifest: {path}')
    for i, row in enumerate(rows):
        for key in ('partial_mesh', 'full_mesh'):
            if not row.get(key):
                raise ValueError(f'Row {i+1}: missing {key}')
        for key in PATH_FIELDS:
            if row.get(key):
                row[key] = str((path.parent / row[key]).resolve())
        row['id'] = row.get('id') or f'pair_{i:05d}'
        if not row['id'].replace('_', '').replace('-', '').isalnum():
            raise ValueError('Pair IDs must be alphanumeric, with optional _ and -')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Pair IDs must be unique')
    return rows


def array(path, key='D'):
    path = Path(path)
    if path.suffix == '.mat':
        return np.asarray(loadmat(path)[key])
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data[key])
    return np.asarray(np.load(path, allow_pickle=False))


def anchors(path, npart, nfull):
    if Path(path).suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            p, f = data['partial'], data['full']
    else:
        f = array(path).reshape(-1)
        if len(f) != npart:
            raise ValueError('Dense GT must contain one target index per partial vertex')
        p = np.arange(npart)
    p, f = np.asarray(p), np.asarray(f)
    if p.shape != f.shape or p.ndim != 1 or not len(p):
        raise ValueError('GT arrays must be nonempty, equally sized vectors')
    if not (np.isfinite(p).all() and np.isfinite(f).all()):
        raise ValueError('Non-finite GT indices')
    if not (np.equal(p, p.astype('int64')).all() and np.equal(f, f.astype('int64')).all()):
        raise ValueError('GT indices must be integers')
    p, f = p.astype('int64'), f.astype('int64')
    if p.min() < 0 or f.min() < 0 or p.max() >= npart or f.max() >= nfull:
        raise ValueError('GT indices must be valid zero-based vertex IDs')
    return p, f  # Repeated remeshed reference samples are intentionally retained.


def load_pair(row):
    vp, fp = load_mesh(row['partial_mesh'])
    vf, ff = load_mesh(row['full_mesh'])
    gt = anchors(row['gt'], len(vp), len(vf)) if row.get('gt') else None
    soft = array(row['soft'], key='P') if row.get('soft') else None
    hard = array(row['hard']).reshape(-1) if row.get('hard') else None
    if hard is not None:
        if len(hard) != len(vp) or not np.equal(hard, hard.astype('int64')).all():
            raise ValueError('Invalid hard correspondence vector')
        hard = hard.astype('int64')
        if hard.min() < 0 or hard.max() >= len(vf):
            raise ValueError('Hard correspondence must contain zero-based full vertex IDs')
    return vp, fp, vf, ff, gt, soft, hard


def mesh_area(v, f):
    return .5 * np.linalg.norm(np.cross(v[f[:, 1]]-v[f[:, 0]],
                                       v[f[:, 2]]-v[f[:, 0]]), axis=1).sum()

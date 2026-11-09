"""Precompute pair operators before optimization, with content-addressed caches."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import igl
from operators.pipeline import pair_tensors, spectrum
from .dataset import load_pair
from .model import wormhole_module


def source_digest():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for folder in ('operators', 'diffusion_net', 'functional_correspondence'):
        for path in sorted((root / folder).glob('*.py')):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def pair_key(row, cfg):
    h = hashlib.sha256(source_digest().encode())
    h.update(json.dumps(dict(operators=cfg['operators'], model=cfg['model'],
                             wormhole_root=cfg.get('wormhole_root')), sort_keys=True).encode())
    for field in ('partial_mesh', 'full_mesh', 'gt', 'soft', 'hard'):
        if row.get(field):
            h.update(field.encode())
            h.update(Path(row[field]).read_bytes())
    if cfg['model'].get('backbone') == 'wormhole':
        root = Path(cfg['wormhole_root'])
        for field in ('utils/geometry_util.py', 'networks/diffusion_network.py'):
            h.update((root / field).read_bytes())
    return h.hexdigest()


def prepare_shape(v, f, tensor, cfg):
    from diffusion_net import geometry as dn
    from diffusion_net.utils import sparse_np_to_torch
    opt = cfg['operators']
    vertices = torch.tensor(np.ascontiguousarray(v), dtype=torch.float32)
    faces = torch.tensor(f, dtype=torch.long)
    normalization = opt.get('normalization', 'max_radius')
    if normalization == 'max_radius':
        vertices = dn.normalize_positions(vertices)
    elif normalization != 'none':
        raise ValueError('normalization must be max_radius or none')
    k = max(opt['k_eig'], opt.get('basis_k', opt['k_eig']), cfg['model']['fmap_k'])
    if k >= len(v)-1:
        raise ValueError('Use fewer eigenpairs than the number of vertices minus one')
    if cfg['model'].get('backbone') == 'wormhole':
        geom = wormhole_module(cfg['wormhole_root'], 'utils.geometry_util')
        _, mass, L, ev, phi, gx, gy = geom.get_operators(vertices, faces, k=k, cache_dir=None)
    else:
        _, mass, L, ev, phi, gx, gy = dn.get_operators(vertices, faces, k_eig=k, op_cache_dir=None)
    shape = dict(vertices=vertices, faces=faces, gradX=gx, gradY=gy,
                 iso_mass=mass, iso_evals=ev, iso_evecs=phi,
                 normals=torch.tensor(igl.per_vertex_normals(v, f.astype('int32')), dtype=torch.float32))
    if tensor is not None:
        lap, m, values, vectors = spectrum(vertices.numpy(), f, tensor, opt['k_eig'])
        mass = torch.tensor(np.asarray(m), dtype=torch.float32)
        L = sparse_np_to_torch(lap).float()
        ev = torch.tensor(values, dtype=torch.float32)
        phi = torch.tensor(np.ascontiguousarray(vectors), dtype=torch.float32)
    shape.update(mass=mass, L=L, evals=ev[:opt['k_eig']], evecs=phi[:, :opt['k_eig']].contiguous())
    return shape


def prepare_pair(row, cfg, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f'{pair_key(row, cfg)}.pt'
    if target.exists():
        return target
    vp, fp, vf, ff, gt, soft, hard = load_pair(row)
    tp = tf = None
    if cfg['operators']['kind'] == 'aniso':
        tp, tf, _ = pair_tensors(vp, fp, vf, ff, cfg['operators'], gt, soft, hard)
    elif cfg['operators']['kind'] != 'iso':
        raise ValueError('Operator kind must be iso or aniso')
    data = dict(partial=prepare_shape(vp, fp, tp, cfg), full=prepare_shape(vf, ff, tf, cfg))
    temporary = target.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(target)
    return target


def read_prepared(path, device):
    data = torch.load(path, map_location='cpu', weights_only=True)
    return {role: {key: value.to(device) for key, value in shape.items()} for role, shape in data.items()}

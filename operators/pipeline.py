"""Portable pair preparation using the original tensor and transport routines."""
import numpy as np
from .boundary import compute_boundary_aware_tensor
from .transfer import transfer_tensor_to_full_fullcover
from .assembly import vertex_field_to_face_tensor, anisotropic_laplacian
from .solvers import compute_eigenpairs


def pair_tensors(vp, fp, vf, ff, cfg, anchors=None, soft=None, hard=None):
    """Construct on the partial mesh and transport to the full mesh.

    Both hard-map sources use the same anchor-p2p transporter. Soft maps
    use the existing row/column top-k transporter. No GT mask restricts NN.
    """
    boundary = cfg.get('boundary', {})
    tp, _, dp, distance = compute_boundary_aware_tensor(vp, fp, **boundary)
    transfer = dict(cfg.get('transfer', {}))
    source = transfer.pop('source', 'gt')
    transfer.setdefault('fallback_mode', 'none')
    transfer.setdefault('seg_tau', .5)
    if source in ('gt', 'hard'):
        if source == 'hard':
            if hard is None:
                raise ValueError('Hard transfer requires a partial-to-full map')
            anchors = (np.arange(len(vp)), hard)
        if anchors is None:
            raise ValueError('GT transfer is an oracle and requires GT anchors')
        transfer.update(transfer_mode='anchor_pair_p2p',
                        anchor_src_ids=anchors[0], anchor_tgt_ids=anchors[1])
    elif source in ('soft', 'reconstructed'):
        if soft is None or soft.shape != (len(vp), len(vf)):
            raise ValueError('Expected correspondence matrix [n_partial, n_full]')
        if not np.isfinite(soft).all():
            raise ValueError('Non-finite correspondence weights')
        if source == 'soft' and (np.any(soft < 0) or not np.allclose(soft.sum(1), 1, atol=1e-4)):
            raise ValueError('Raw soft probabilities must be nonnegative with unit row sums')
        transfer.update(transfer_mode='wormhole_soft_linear',
                        Pyx_part_to_full=soft, p2p_part_to_full=hard)
    else:
        raise ValueError(f'Unknown transfer source: {source}')
    tf, df, alpha, error, stats = transfer_tensor_to_full_fullcover(
        vf, ff, vp, fp, tp, dp, **transfer)
    return (tp, dp), (tf, df), dict(distance=distance, coverage=alpha,
                                   transfer_error=error, stats=stats)


def spectrum(v, f, tensor, k):
    """Independent tensor assembly, including when beta=gamma=0.

    Convert the generator sign to the positive-semidefinite stiffness sign
    expected by DiffusionNet. Never substitute cached ISO eigenpairs.
    """
    directions, eigenvalues = tensor
    u, w, d = vertex_field_to_face_tensor(v, f, directions, eigenvalues)
    lap, mass = anisotropic_laplacian(v, f, u, w, d)
    positive = -lap if np.mean(lap.diagonal()) < 0 else lap
    values, vectors = compute_eigenpairs(positive, mass, k=k, verbose=False)
    return positive, mass.diagonal(), values, vectors

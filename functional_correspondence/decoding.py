"""NN and functional-map decoders. All candidates are full-mesh vertices."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


def cosine_nn(partial, full, chunk=1024):
    p, f = F.normalize(partial, dim=-1), F.normalize(full, dim=-1)
    return torch.cat([(x @ f.T).argmax(-1) for x in p.split(chunk)])


def soft_correspondence(partial, full, tau=.01):
    if tau <= 0:
        raise ValueError('Temperature must be positive')
    return ((F.normalize(partial, dim=-1) @ F.normalize(full, dim=-1).T) / tau).softmax(-1)


def project_correspondence(P, phi_p, mass_p, phi_f):
    return (phi_p.T * mass_p[None]) @ P @ phi_f


def fm_decode(C, phi_p, phi_f, convention='full_to_partial'):
    # FM-MSE solves C: partial coefficients -> full coefficients. WH projects
    # the reverse map. Keep the convention explicit rather than transposing
    # to make the matrix dimensions happen to agree.
    if convention == 'full_to_partial':
        target, query = phi_f @ C.T, phi_p
    elif convention == 'partial_to_full':
        target, query = phi_f @ C, phi_p
    else:
        raise ValueError(convention)
    target, query = target.detach().cpu().numpy(), query.detach().cpu().numpy()
    return cKDTree(target).query(query)[1].astype(np.int64)


def solve_feature_fmap(fp, ff, ep, ef, pp, pf, mp, mf, regularization=1e-3):
    """Original row-wise FM solver with Laplacian commutativity penalty."""
    a, b = (pp.T * mp[None]) @ fp, (pf.T * mf[None]) @ ff
    gram, rhs = a @ a.T, b @ a.T
    rows = []
    for i in range(len(ef)):
        penalty = torch.diag((ep-ef[i]).square())
        rows.append((torch.inverse(gram + regularization*penalty) @ rhs[i, :, None]).squeeze(-1))
    return torch.stack(rows)

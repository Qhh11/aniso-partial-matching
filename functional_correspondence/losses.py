"""FM-MSE supervision and an adapter to the unmodified external WH losses."""
import torch
from .decoding import project_correspondence, solve_feature_fmap, soft_correspondence
from .model import wormhole_module


def fmap_prediction(fp, ff, p, f, cfg):
    k = cfg['model']['fmap_k']
    return solve_feature_fmap(fp, ff, p['evals'][:k], f['evals'][:k],
                             p['evecs'][:, :k], f['evecs'][:, :k], p['mass'], f['mass'],
                             cfg.get('fm_regularization', 1e-3))


class Objective:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg['objective'] == 'wormhole':
            geo = wormhole_module(cfg['wormhole_root'], 'losses.geodesic_loss')
            ortho = wormhole_module(cfg['wormhole_root'], 'losses.fmap_loss')
            self.geometric = geo.PartialGeodesicLoss(loss_weight=cfg.get('geodesic_weight', 5000))
            self.orthogonal = ortho.PartialOrthoLoss(w_orth=cfg.get('orthogonal_weight', 1))
        elif cfg['objective'] != 'fm_mse':
            raise ValueError('Unknown objective')

    def __call__(self, fp, ff, p, f, references, training_geometry=None):
        cfg = self.cfg
        if cfg['objective'] == 'fm_mse':
            if references is None:
                raise ValueError('FM-MSE training requires GT references')
            ip, jf = references
            k = cfg['model']['fmap_k']
            target = torch.linalg.lstsq(p['evecs'][ip, :k], f['evecs'][jf, :k]).solution.T
            return {'fm_mse': (fmap_prediction(fp, ff, p, f, cfg)-target).square().mean()}
        if training_geometry is None:
            raise ValueError('WH requires preprocessed training distances, mass and mask')
        P = soft_correspondence(fp, ff, cfg.get('temperature', .01))
        # WH's spectral regularizer uses the common ISO basis, independently
        # of the operator used inside the feature extractor.
        C = project_correspondence(P, p['iso_evecs'], p['iso_mass'], f['iso_evecs'])
        loss = self.geometric(P[None], training_geometry['full_dist'][None],
                              training_geometry['partial_dist'][None],
                              training_geometry['partial_mass'][None], training_geometry['mask'][None])
        loss.update(self.orthogonal(C[None], f['iso_evals'][None], p['iso_evals'][None]))
        return loss

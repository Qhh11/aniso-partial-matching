"""DiffusionNet feature extraction; Wormhole remains an external dependency."""
import importlib
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn


def wormhole_module(root, module):
    root = Path(root).expanduser().resolve()
    if not (root / 'networks/diffusion_network.py').is_file():
        raise FileNotFoundError('Set wormhole_root to a checkout of ABracha/Wormhole')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    result = importlib.import_module(module)
    if not Path(result.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f'Conflicting external module: {module}')
    return result


class MagnitudeGradient(nn.Module):
    """Explicit A=I energy: tanh(gx^2 + gy^2), without learned mixing."""
    def forward(self, gradient):
        return torch.tanh(gradient.square().sum(-1))


class FeatureModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model = cfg['model']
        self.features = model.get('input', 'xyz')
        cin = {'xyz': 3, 'xyz_normals': 6, 'hks': 16}[self.features]
        self.external = model.get('backbone', 'diffusionnet') == 'wormhole'
        if self.external:
            module = wormhole_module(cfg['wormhole_root'], 'networks.diffusion_network')
            # Features/geometry are passed explicitly below; do not recompute
            # operators inside the external model's high-level forward().
            self.net = module.DiffusionNet(in_channels=cin, out_channels=model['channels'],
                hidden_channels=model.get('width', 128), n_block=model.get('blocks', 4),
                use_normals=False, dropout=model.get('dropout', True))
        else:
            from diffusion_net.layers import DiffusionNet
            self.net = DiffusionNet(C_in=cin, C_out=model['channels'],
                C_width=model.get('width', 128), N_block=model.get('blocks', 4),
                dropout=model.get('dropout', True))
        if model.get('gradient', 'full') == 'identity':
            count = 0
            for block in self.net.blocks:
                for name, child in list(block.named_children()):
                    if child.__class__.__name__ == 'SpatialGradientFeatures':
                        setattr(block, name, MagnitudeGradient())
                        count += 1
            if count != model.get('blocks', 4):
                raise RuntimeError('Unrecognized gradient-module layout')
        elif model.get('gradient', 'full') != 'full':
            raise ValueError('gradient must be full or identity')

    def forward(self, shape):
        x = shape['vertices']
        if self.features == 'hks':
            from diffusion_net.geometry import compute_hks_autoscale
            x = compute_hks_autoscale(shape['evals'], shape['evecs'], 16)
        else:
            normals = shape['normals']
            if self.training and self.external:
                augment = wormhole_module(self.cfg['wormhole_root'], 'utils.geometry_util').data_augmentation
                if self.features == 'xyz_normals':
                    x, normals = augment(x[None], normals[None])
                    x, normals = x[0], normals[0]
                else:
                    x = augment(x[None])[0]
            if self.features == 'xyz_normals':
                x = torch.cat((x, normals), -1)
        if not self.external:
            return self.net(x, shape['mass'], L=shape['L'], evals=shape['evals'],
                            evecs=shape['evecs'], gradX=shape['gradX'], gradY=shape['gradY'],
                            faces=shape['faces'])
        x = self.net.first_linear(x[None])
        for block in self.net.blocks:
            # In the original intervention only the diffusion spectral path
            # changes; gradient interactions and residual mass remain ISO.
            x = block(x, shape['mass'][None], shape['L'][None],
                      shape['evals'][None], shape['evecs'][None],
                      shape['gradX'][None], shape['gradY'][None])
        return self.net.last_linear(x)[0]

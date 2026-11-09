"""CPU smoke tests on generated meshes; no licensed dataset is required."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
import yaml
import igl
from operators.assembly import cotangent_laplacian, anisotropic_laplacian, vertex_field_to_face_tensor
from operators.boundary import compute_boundary_aware_tensor
from operators.solvers import implicit_heat_step
from functional_correspondence.decoding import cosine_nn, project_correspondence
from functional_correspondence.metrics import geodesic_errors, summarize
from functional_correspondence.model import MagnitudeGradient

ROOT = Path(__file__).resolve().parents[1]


def mesh():
    n = 6
    v = np.array([(x/(n-1), y/(n-1), .06*np.sin(x)*np.cos(y))
                  for y in range(n) for x in range(n)], dtype=float)
    f = []
    for y in range(n-1):
        for x in range(n-1):
            a = y*n+x
            f += [(a, a+1, a+n), (a+1, a+n+1, a+n)]
    return v, np.array(f, dtype=np.int32)


def config():
    cfg = yaml.safe_load((ROOT/'configs/fm_mse.yaml').read_text())
    cfg['operators'].update(k_eig=8, kind='aniso')
    cfg['operators']['boundary'].update(beta_tangent=0., gamma_normal=0.,
        distance_method='dijkstra', direction_method='parent')
    cfg['model'].update(channels=8, width=8, blocks=2, fmap_k=4, dropout=False)
    cfg['training']['epochs'] = 2
    return cfg


class NumericalTests(unittest.TestCase):
    def test_independent_isotropic_assembly(self):
        v, f = mesh()
        t, _, d, _ = compute_boundary_aware_tensor(v, f, beta_tangent=0., gamma_normal=0.,
            distance_method='dijkstra', direction_method='parent')
        np.testing.assert_array_equal(d, np.ones_like(d))
        u, w, df = vertex_field_to_face_tensor(v, f, t, d)
        la, ma = anisotropic_laplacian(v, f, u, w, df)
        li, mi = cotangent_laplacian(v, f)
        # The retained vertex-to-face converter returns float32 frames. Check
        # its error at that precision, not bitwise/double-precision identity.
        relative = np.linalg.norm((la-li).toarray()) / np.linalg.norm(li.toarray())
        self.assertLess(relative, np.finfo(np.float32).eps)
        np.testing.assert_allclose(ma.diagonal(), mi.diagonal(), atol=1e-12)

    def test_double_precision_identity_frames(self):
        v, f = mesh()
        u = v[f[:, 1]]-v[f[:, 0]]
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        normals = np.cross(u, v[f[:, 2]]-v[f[:, 0]])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        w = np.cross(normals, u)
        la, _ = anisotropic_laplacian(v, f, u, w, np.ones((len(f), 2)))
        li, _ = cotangent_laplacian(v, f)
        np.testing.assert_allclose(la.toarray(), li.toarray(), rtol=1e-10, atol=1e-10)

    def test_heat_constant(self):
        v, f = mesh()
        lap, mass = cotangent_laplacian(v, f)
        np.testing.assert_allclose(implicit_heat_step(lap, mass, np.ones(len(v)), .1), 1., atol=1e-9)

    def test_gradient_identity(self):
        x = torch.randn(1, 7, 4, 2, requires_grad=True)
        y = MagnitudeGradient()(x)
        torch.testing.assert_close(y, torch.tanh(x[..., 0]**2+x[..., 1]**2))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_correspondence_convention(self):
        full = torch.eye(5)
        partial = full[[3, 0, 2]]
        np.testing.assert_array_equal(cosine_nn(partial, full).numpy(), [3, 0, 2])
        p = torch.eye(5)[[3, 0, 2]]
        C = project_correspondence(p, torch.eye(3), torch.ones(3), full)
        torch.testing.assert_close(C, p)

    def test_metrics(self):
        d = np.abs(np.arange(4)[:, None]-np.arange(4)[None, :]).astype(float)
        err = geodesic_errors(np.array([2, 1]), (np.array([0, 1]), np.array([2, 1])), d, 4.)
        self.assertEqual(summarize(err)['p05'], 1.)

    def test_train_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            v, f = mesh()
            igl.write_triangle_mesh(str(tmp/'mesh.off'), v, f)
            np.save(tmp/'gt.npy', np.arange(len(v)))
            np.save(tmp/'distance.npy', np.linalg.norm(v[:, None]-v[None, :], axis=-1))
            # Euclidean distances are only a synthetic plumbing fixture here,
            # never reported as a real geodesic benchmark.
            with (tmp/'pairs.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id', 'partial_mesh', 'full_mesh', 'gt', 'full_dist', 'pairing'])
                writer.writerow(['synthetic', 'mesh.off', 'mesh.off', 'gt.npy', 'distance.npy', 'same'])
            cfg = config()
            (tmp/'config.yaml').write_text(yaml.safe_dump(cfg))
            common = ['--config', str(tmp/'config.yaml'), '--pairs', str(tmp/'pairs.csv'),
                      '--cache', str(tmp/'cache'), '--device', 'cpu']
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            for script, extra in [('train.py', ['--output', str(tmp/'train')]),
                                  ('inference.py', ['--output', str(tmp/'eval'),
                                   '--checkpoint', str(tmp/'train/last.pt')])]:
                result = subprocess.run([sys.executable, script, *common, *extra], cwd=ROOT,
                                        env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(np.load(tmp/'eval/synthetic_p2p.npy').shape, (len(v),))
            result = json.loads((tmp/'eval/summary.json').read_text())
            self.assertTrue(np.isfinite(result['mean']))
            self.assertTrue((tmp/'eval/pck.csv').is_file())

    @unittest.skipUnless(os.environ.get('WORMHOLE_ROOT'), 'Optional external Wormhole checkout not configured')
    def test_external_wh_backward(self):
        from functional_correspondence.preparation import prepare_shape
        from functional_correspondence.model import FeatureModel
        from functional_correspondence.losses import Objective
        cfg = config()
        cfg.update(objective='wormhole', wormhole_root=os.environ['WORMHOLE_ROOT'])
        cfg['model'].update(backbone='wormhole', input='xyz_normals', fmap_k=8)
        cfg['operators']['basis_k'] = 8
        v, f = mesh()
        t, _, d, _ = compute_boundary_aware_tensor(v, f, **cfg['operators']['boundary'])
        shape = prepare_shape(v, f, (t, d), cfg)
        model = FeatureModel(cfg)
        ff, fp = model(shape), model(shape)
        dist = torch.tensor(np.linalg.norm(v[:, None]-v[None, :], axis=-1), dtype=torch.float32)
        geo = dict(full_dist=dist, partial_dist=dist, partial_mass=shape['iso_mass'],
                   mask=torch.ones_like(dist))
        loss = sum(Objective(cfg)(fp, ff, shape, shape, None, geo).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

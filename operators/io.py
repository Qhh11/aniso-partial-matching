"""Small mesh I/O helpers shared by the examples."""
from pathlib import Path
import numpy as np
import igl


def load_mesh(name):
    path = Path(name).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Provide an existing mesh path: {path}')
    v, f = igl.read_triangle_mesh(str(path))
    if not len(v) or not len(f) or not np.isfinite(v).all():
        raise ValueError(f'Invalid triangle mesh: {path}')
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def save_vertex_scalar(name, V, F, scalar, subdir='heatmaps'):
    folder = Path('outputs') / subdir
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f'{Path(name).stem}.npz', V=V, F=F, scalar=scalar)


def save_eigs(name, evals, evecs):
    folder = Path('outputs/eigs')
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f'{Path(name).stem}.npz', evals=evals, evecs=evecs)

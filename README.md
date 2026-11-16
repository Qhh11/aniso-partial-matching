# Anisotropic partial shape matching

Boundary-dependent diffusion tensors for partial-to-full surface correspondence.
The tensor is built on the partial mesh, transferred to the full mesh, and used
to assemble anisotropic Laplace–Beltrami operators. The repository includes
geometric examples and a portable DiffusionNet training/inference interface.

## Layout

```text
make_dataset/                 clean registered partial-FAUST generation
operators/                    assembly, boundary tensors, transport, heat and spectra
diffusion_net/                core DiffusionNet implementation
functional_correspondence/    pair loading, models, objectives, decoding and metrics
configs/                      FM-MSE and external-Wormhole example configurations
examples/                     diffusion, HKS, matching and visualization
tests/                        synthetic CPU sanity and end-to-end smoke tests
train.py / inference.py       command-line entry points
```

## Installation

Use Python 3.10 or newer. Install PyTorch for your platform, then:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run commands from the repository root. Interactive examples require a graphical
display. Numerical comparisons and PCK plotting also work without one.

The zero-anisotropy tests distinguish double-precision assembly from the
retained float32 vertex-to-face frames. They check the appropriate numerical
tolerance, not equality of complete training trajectories. Set
`WORMHOLE_ROOT=/path/to/Wormhole` when running the tests to also exercise an
external WH forward/backward step on synthetic inputs.

## Data

Obtain FAUST/PFAUST under their original data licenses. See
[make_dataset/README.md](make_dataset/README.md) for clean partial-mesh generation.
Do not redistribute the generated meshes unless the underlying license allows it.

Pairs are explicit CSV manifests. Paths are relative to the CSV file, not to a
machine-specific root. Example:

```csv
id,partial_mesh,full_mesh,gt,full_dist,pairing
source_to_parent,clean/tr_reg_000_part.ply,clean/tr_reg_000_full.ply,clean/tr_reg_000_part2full.npy,dist/tr_reg_000.mat,same
source_to_other,clean/tr_reg_000_part.ply,clean/tr_reg_010_full.ply,clean/tr_reg_000_part2full.npy,dist/tr_reg_010.mat,cross
```

The second row is valid only for registered meshes sharing the same template
vertex indexing. 

- `gt`: a zero-based dense partial-to-full `.npy` vector, or an `.npz` containing
  equally sized `partial` and `full` reference-index arrays. Repeated evaluation
  samples are retained; vertex counts are not assumed equal.
- `full_dist`: raw target geodesic distances, `.mat` with key `D`, or `.npy`.
  Evaluation divides distances by the square root of the raw target area.
- Optional `soft`: a `.npy` matrix or `.npz` with key `P`, with partial vertices
  as rows and full vertices as columns.
- Optional `hard`: a zero-based partial-to-full index vector.
- `pairing`: `same` or `cross`, required by the corresponding HKS examples.


## Training and inference

```bash
python train.py --config configs/fm_mse.yaml --pairs data/train.csv \
    --output outputs/fm_train --cache cache/fm --device cuda
python inference.py --config configs/fm_mse.yaml --pairs data/test.csv \
    --checkpoint outputs/fm_train/last.pt --output outputs/fm_test \
    --cache cache/fm --device cuda
```

Use a new output directory for each command. `--prepare-only` builds operators
without optimization. Pair caches are keyed by mesh/correspondence contents,
configuration and implementation. Preparation finishes before model
initialization. Checkpoints contain model, optimizer, scheduler and configuration;
the command-line driver currently starts fresh runs rather than resuming them.

Configuration switches:

- `operators.kind: iso | aniso` selects the spectral diffusion path.
- `operators.boundary` controls beta, gamma, sigmoid falloff and tensor normalization.
  Setting beta=gamma=0 still uses independent tensor assembly and eigensolving;
  it does not silently substitute an ISO cache.
- `operators.transfer.source: gt | hard | soft | reconstructed` selects the map.
  GT is an oracle control, including during inference if explicitly selected.
  For deployment use externally estimated maps; GT is not required for prediction.
- `model.gradient: full | identity` selects learned gradient-channel mixing or
  the explicit energy `tanh(gx^2 + gy^2)`. 
- The FM backbone is the included DiffusionNet. The WH backbone and losses are
  loaded from an external Wormhole checkout.

FM-MSE uses the feature-fitted functional-map decoder. WH inference follows
hard feature NN, spectral projection, then functional-map decoding. Both also
save direct-NN predictions. Metrics use all full vertices as candidates, with
no GT-visible candidate restriction. PCK files contain fractions, plotted as
percentages. The global mean is weighted by reference samples; `pairs.csv`
also exposes pair-level results.

The portable driver preserves the core tensor routines but reorganizes data
loading and configuration. Do not treat it as a bitwise replay of archived
training scripts. Independent eigensolves need not produce identical bases.

## External Wormhole integration

Use the [official Wormhole repository](https://github.com/ABracha/Wormhole).
Its source and weights are not bundled here. Install its dependencies and set
`wormhole_root` in `configs/wormhole.yaml` to your checkout.

### 1. Export a teacher correspondence

Load the teacher using the external repository's evaluation setup. For each
pair, save either raw soft probabilities or a clearly identified hard map.
For already loaded `feat_full` and `feat_partial` tensors of shape `[1,N,C]`:

```python
# Run the feature extractor in evaluation mode, without gradients.
model.networks['permutation'].hard = False
P = model.compute_permutation_matrix(feat_partial, feat_full, bidirectional=False)
np.save('pair_soft.npy', P[0].detach().cpu().numpy())
np.save('pair_argmax.npy', P[0].argmax(-1).detach().cpu().numpy())
```

This is a snippet to insert in the external evaluation loop. A hard map returned by
`validate_single()` is FM-decoded and is not generally the same as raw-soft
argmax. Its returned `Pyx` is spectrally reconstructed, not raw softmax
probabilities. Store the map kind explicitly; use `source: reconstructed` for
that matrix. Reconstruction can produce negative entries, whereas `source:
soft` requires nonnegative, row-normalized probabilities.

### 2. Read the export and transfer the tensor

Add the `soft` or `hard` path to the pair manifest and choose the corresponding
transfer source. The partial tensor is transported to the full surface before
assembling either diffusion operator.

### WH-objective training

The adapter calls the external `PartialGeodesicLoss` and `PartialOrthoLoss`
directly; it does not patch their area weighting. Supply four additional
manifest columns: `train_full_dist`, `train_partial_dist`,
`train_partial_mass`, and `train_mask` (`.npy` files). These must contain the
already-preprocessed inputs expected by that external implementation:
`[Nf,Nf]`, `[Np,Np]`, `[Np]`, and `[Np,Np]` respectively. These are distinct
from raw evaluation distances and GT visibility masks.

Use the same training command with `configs/wormhole.yaml`. The WH regularizer
and FM decoder retain the common ISO basis; the ANISO spectrum is used inside
feature diffusion. 

## Examples

See [examples/README.md](examples/README.md) for the diffusion/HKS demos,
same-pose and cross-pose comparison, tensor inspection, correspondence colors,
and PCK curves.

## Attribution

The included DiffusionNet core is adapted from
[nmwsharp/diffusion-net](https://github.com/nmwsharp/diffusion-net).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the original license
included with that package. Dataset and external-repository licenses apply
independently.

This private snapshot does not add an open-source license for the
project-specific code. Existing third-party licenses and attribution remain
in effect. Public release and licensing of the project-specific code are
separate decisions.

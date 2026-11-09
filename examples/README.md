# Examples

Run from the repository root. Replace example paths with your own meshes.
Interactive viewers require a display; use `--no-show` where available for
numerical export only. No training is performed by these examples.

```bash
python -m examples.demo_faust --help
python -m examples.demo_pfaust --help
python -m examples.demo_hks --help
```

- `demo_faust`: inspect a full mesh, eigenfunctions and heat propagation.
  Its curvature-oriented anisotropy is a geometry demonstration, not the
  boundary-tensor partial matching method.
- `demo_pfaust`: inspect boundary distance, directions, eigenfunctions and
  ISO/ANISO heat propagation on a partial mesh.
- `demo_hks`: inspect ISO, curvature-ANISO or boundary-ANISO HKS fields.
  Choose the boundary mode for the partial-matching construction.

## Matching comparison

Use the pair-manifest format in the root README, with `pairing=same` for
partial-to-parent pairs and `pairing=cross` for different-pose pairs.

```bash
python -m examples.compare_pfaust --pairs data/same.csv \
    --config configs/fm_mse.yaml --output outputs/hks_same --decoder nn
python -m examples.compare_pfaust_cross --pairs data/cross.csv \
    --config configs/fm_mse.yaml --output outputs/hks_cross --decoder nn
```

Both compare ISO and ANISO, using a shared explicit HKS time grid and all full
vertices as candidates. The default NN is cosine NN. `--decoder fm` enables
feature-fitted functional-map matching. Configure transfer through the same
operator configuration used for training; do not confuse oracle GT transfer
with estimated transfer. The output includes point maps, pairwise metrics,
sample-weighted mean error and PCK curves. These are non-learning descriptors,
not neural-network evaluation results.

## Inspect transfer and predictions

```bash
python -m examples.visualize_transfer --pairs data/same.csv --pair-index 0 \
    --config configs/fm_mse.yaml --output outputs/transfer.npz
python -m examples.visualize_correspondence --pairs data/test.csv \
    --prediction outputs/fm_test/source_to_parent_p2p.npy
```

`visualize_correspondence` shows position colors and checkerboard transfer.
With GT and target distances it also shows errors at reference samples. Add
`--comparison another_map.npy` to show signed error differences. Unknown
vertices are not assigned fabricated evaluation errors.

## Plot curves

```bash
python -m examples.plot_pck outputs/hks_cross/iso_pck.csv outputs/hks_cross/aniso_pck.csv \
    --labels ISO ANISO --colors '#1f77b4' '#d62728' --linestyles solid dashed \
    --output outputs/comparison.png
```

Reuse the same labels, colors and line styles across comparable experiments.
Plotting reads CSV curves only; it never scans historical result folders.

# Clean registered partial FAUST

These scripts cut partial meshes from registered FAUST and precompute raw
full-mesh geodesic distances. They do not reproduce a separately remeshed
PFAUST dataset. Obtain the original registrations under their data license.

```bash
python make_dataset/make_partial_faust.py --faust_dir /path/to/registrations \
    --out data/clean --shapes tr_reg_000 tr_reg_010 --seed 0 --save_full
python make_dataset/make_geodesic_full.py --clean_dir data/clean \
    --out_dist_dir data/dist --shapes tr_reg_000 tr_reg_010
```

For each shape, the first script writes a partial mesh, a dense
`*_part2full.npy` mapping into the original registered vertex indexing, and
optionally the full mesh. Use `--help` for hole size, number of holes,
retention and mesh-cleanup controls. Keep the generation arguments with your
own dataset; different cuts are different experiments.

The second script computes full-mesh distance matrices with the heat method
and writes `.mat` files with key `D`, in raw mesh units. It can be expensive
in time and disk space. Area normalization belongs to evaluation, not this
export. Neither generated meshes nor matrices belong in Git.

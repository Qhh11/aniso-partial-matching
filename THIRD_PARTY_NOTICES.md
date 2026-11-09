# Third-party notices

## DiffusionNet

The `diffusion_net/` package originates from
https://github.com/nmwsharp/diffusion-net, by Nicholas Sharp and coauthors.
It is distributed under the MIT license reproduced in
`diffusion_net/LICENSE`. Functional-map fitting in
`functional_correspondence/decoding.py` follows the upstream functional
correspondence example and retains the same attribution.

## Wormhole

https://github.com/ABracha/Wormhole is an optional external dependency for
teacher correspondences, its DiffusionNet variant, and WH-objective losses.
Its implementation is not vendored in this repository. Follow its license
and citation instructions when using it. The adapter here is not an official
Wormhole release and does not modify its loss implementation.

## Geometry libraries and data

libigl, potpourri3d, robust-laplacian, Polyscope and other dependencies retain
their respective licenses. FAUST/PFAUST meshes, correspondences, distances
and model weights are not distributed here. Obtain datasets from their
providers and comply with their separate licenses.

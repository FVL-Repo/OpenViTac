# Third-party notices

## Taxim (vendored torch backend)

The files in this directory are derived from the Taxim project's torch
simulation backend:

- `taxim_torch.py`
- `taxim_impl.py`
- `calibration.py`
- `LICENSE`

Upstream: `https://github.com/Robo-Touch/Taxim`

License: MIT

Copyright (c) 2021 CMURoboTouch

The upstream `LICENSE` file is preserved verbatim in this directory.  The
shadow-rendering branch was removed because OpenViTac uses
`with_shadow=False` and does not want a `torch_scatter` dependency.  The JAX
backend and the Isaac/TacEx wrapper are not vendored; the OpenViTac adapter
lives in `openvitac_sim.optical`.

## TacEx calibration assets

The GsMini/Xense 640x480 calibration files used with Taxim were obtained from
the TacEx assets tree (`tacex_assets/data/Sensors/GelSight_Mini/calibs/640x480`).
TacEx declares MIT in its package metadata, but the upstream repository does
not contain a root LICENSE file.  Keep the TacEx attribution and confirm
redistribution terms before publishing a public asset copy.  The Xense
400x700 `real_*_blur_bg.png` files are old-stack-only assets with unclear
licensing and must not be redistributed.

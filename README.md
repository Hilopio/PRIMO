# PRIMO: Panoramic Reconstruction with Integrated Microscopy-Specific Optimization

[![PyPI](https://img.shields.io/pypi/v/primo-stitch)](https://pypi.org/project/primo-stitch/)
[![Python](https://img.shields.io/pypi/pyversions/primo-stitch)](https://pypi.org/project/primo-stitch/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

PRIMO stitches a set of overlapping tiles into a single 2D panorama. The
package is published on PyPI as
[`primo-stitch`](https://pypi.org/project/primo-stitch/) — install it and run
the `primo-stitch` command.

## Install

**Python 3.10–3.12** is required (3.13 is not supported yet). With
[`uv`](https://docs.astral.sh/uv/) you can provision a compatible interpreter
without touching your system Python:

```bash
uv venv --python 3.12
```

Install from PyPI:

```bash
pip install primo-stitch
# or:  uv pip install primo-stitch
```

Pre-built wheels are also available on the
[Releases](https://github.com/Hilopio/PRIMO/releases) page.

Notes:
- PRIMO depends on **PyTorch**. The default install pulls the CPU build; for GPU,
  install the CUDA build of `torch`/`torchvision` first (see https://pytorch.org),
  then install PRIMO.
- Matcher weights download automatically on first use (Hugging Face Hub / Torch
  Hub) and are cached locally.

After installing you get the `primo-stitch` command and the importable `primo`
package.

## Run

Point `--tile_dir` at a folder of overlapping tiles:

```bash
# minimal
primo-stitch --tile_dir path/to/tiles --output_file panorama.png

# advanced
primo-stitch \
  --tile_dir path/to/tiles \
  --output_file panorama.png \
  --matcher "efficient loftr" \
  --device cuda:0 \
  --blending_mode full \
  --inference_size 0.5 \
  --batch_size 8
```

> By default (`full` blending with `--save_alpha_channel` on), the panorama is
> written as **`.png`** (RGBA) regardless of the requested extension. Pass
> `--no-save_alpha_channel` to save a `.jpg`.

Add `--online` to write a live `status.json` and `preview.jpg` while processing;
a streaming variant tailored to the web demo is also available as
`primo-stitch-online` (see below).

### Options

| Flag | Default | Description |
|---|---|---|
| `--tile_dir` | *(required)* | Directory with the input tiles |
| `--output_file` | `panorama.jpg` | Output panorama path (extension may be adjusted, e.g. `.png` when alpha is saved) |
| `--cache_dir` | `cache/` | Cache directory (created automatically) |
| `--matcher` | `xfeat` | `xfeat` \| `efficient loftr` \| `loftr` |
| `--device` | `cpu` | `cpu`, `cuda`, `cuda:0`, ... |
| `--blending_mode` | `full` | `collage` \| `mosaic` \| `full` |
| `--inference_size` | `0.3` | Matcher input scale relative to the original (`0.25`, `0.5`, `1`, ...) |
| `--batch_size` | `1` | Matcher batch size (higher = faster, more memory) |
| `--save_alpha_channel` / `--no-save_alpha_channel` | on | Save the transparency channel; forces `.png` output in `full` mode |
| `--online` | off | Write a live status file and geometric preview while processing |
| `--status_file` | `status.json` | Path to the machine-readable status JSON (online mode) |
| `--preview_file` | `preview.jpg` | Path to the geometric preview image (online mode) |
| `--logfile` | *(none)* | Write a debug log to this file |

### `primo-stitch-online`

`primo-stitch-online` is a streaming variant tailored to the web demo: online
reporting is always on, the output must be `.jpg` (so `--save_alpha_channel`
is off and unsupported), and the preview must be `.webp`
(`--preview_file` defaults to `preview.webp`). It accepts the same options as
above (minus `--online`) plus:

| Flag | Default | Description |
|---|---|---|
| `--use_grid_info` | `false` | Use the grid (row/col) layout to match only neighbouring tile pairs; faster matching |
| `--panorama_quality` | `95` | JPEG quality of the final panorama (0–100) |
| `--preview_quality` | `70` | WebP quality of the geometric preview (0–100) |
| `--preview_scale` | `0.25` | Downscale factor for the geometric preview |
| `--memlog_file` | *(none)* | Sample process memory and save a usage chart to this image file |

Its `--inference_size` also accepts an explicit pixel size as
`"(WIDTH, HEIGHT)"` in addition to a scale factor.

## Python API

```python
from primo import Matcher, Stitcher

matcher = Matcher(
    model='xfeat',            # 'xfeat' | 'efficient loftr' | 'loftr'
    device='cuda:0',          # or 'cpu'
    inference_size=0.5,
    batch_size=8,
)

# the alignment device is taken from the matcher
stitcher = Stitcher(
    matcher,
    blending_mode='full',     # 'collage' | 'mosaic' | 'full'
    save_alpha_channel=True,
)

stitcher.stitch(
    input_dir='path/to/tiles',
    output_file='panorama.png',
    cache_dir='cache',
)
```

`Matcher` and `Stitcher` expose additional keyword arguments (alignment,
photometric correction, blending) — see their signatures for the full set.

## Authors

- Gleb Nikolaev — Lomonosov Moscow State University
- Savelii Shashkov — Lomonosov Moscow State University
- Dmitriy Korshunov — Geological Institute of the Russian Academy of Sciences
- Andrey Krylov — Lomonosov Moscow State University
- Alexander Khvostikov (corresponding author) — Lomonosov Moscow State University

## Citation

A paper describing PRIMO is currently under review; full citation details
(venue, year, DOI) will be added once it is published. Until then, please
credit the authors:

```bibtex
@unpublished{primo,
  title  = {PRIMO: Panoramic Reconstruction with Integrated Microscopy-Specific Optimization},
  author = {Nikolaev, Gleb and Shashkov, Savelii and Korshunov, Dmitriy and Krylov, Andrey and Khvostikov, Alexander},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

PRIMO's source code is licensed under the [Apache License 2.0](LICENSE). It
bundles third-party components under their own licenses (Apache-2.0, MIT); see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

Any accompanying publication is licensed separately from this software.

# PRIMO: Panoramic Reconstruction with Integrated Microscopy-Specific Optimization

Python library and command-line tool for stitching a set of overlapping tiles
into a single 2D panorama.

## Installation

```bash
pip install primo-stitch
```

- Requires **Python 3.10–3.12**.
- PRIMO depends on **PyTorch**. The default install pulls the CPU build; for GPU
  use, install the CUDA build of `torch`/`torchvision` first (see
  https://pytorch.org), then install PRIMO.
- Matcher weights are downloaded automatically on first use (Hugging Face Hub /
  Torch Hub) and cached locally.

After installation you get the `primo-stitch` command and the importable
`primo` package (install name `primo-stitch`, import name `primo`).

## Command-line usage

`primo-stitch` is the main entry point.

```bash
# minimal — stitch a folder of tiles into a panorama
primo-stitch --tile_dir path/to/tiles --output_file panorama.png

# advanced
primo-stitch \
  --tile_dir path/to/tiles \
  --output_file panorama.png \
  --matcher "efficient loftr" \
  --device cuda:0 \
  --blending_mode full \
  --inference_size 0.5 \
  --batch_size 8 \
  --cache_dir cache/ \
  --logfile run.log
```

> By default (`full` blending with `--save_alpha_channel` on), the panorama is
> written as **`.png`** (RGBA) regardless of the requested extension. Pass
> `--no-save_alpha_channel` to save a `.jpg`.

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
| `--logfile` | *(none)* | Write a debug log to this file |

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
    output_file='panorama.jpg',
    cache_dir='cache',
)
```

`Matcher` and `Stitcher` expose additional keyword arguments (alignment,
photometric correction, blending) — see their signatures for the full set.

## Demo

The interactive demo (live preview + machine-readable progress) is driven by
`primo-stitch-online` and the `run_online.sh` wrapper. For normal use they are
not needed; `primo-stitch` can also emit progress by adding `--online`, which
writes a `status.json` and a `preview.jpg` while processing.

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

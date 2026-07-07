# VALIS register

Rigidly align an H&E or multi-channel immunofluorescence (IF) image to a
10x Xenium DAPI image using [VALIS](https://github.com/MathOnco/valis), and
export the resulting transform as a 3x3 affine matrix (CSV).

The moving image's modality is auto-detected from its reader metadata:

- **H&E** (brightfield RGB): registration matches tissue texture/folds
  (via VALIS's `ColorfulStandardizer`) against the DAPI nuclei texture.
- **IF** (multi-channel fluorescence): registration is DAPI-to-DAPI. The
  DAPI channel is located by name (case-insensitive substring match, e.g.
  "DAPI", "Channel1-DAPI", "Nuclear-DAPI-405", "Violet-Z") and extracted for
  registration against the Xenium DAPI channel.

## Requirements

- macOS (Apple Silicon), Linux, or Windows, all x86-64/arm64 as applicable
  (`osx-arm64`, `linux-64`, `win-64`). Intel Macs (`osx-64`) aren't
  supported: VALIS's `torchvision` dependency no longer ships wheels for
  that platform.
- Either [pixi](https://pixi.sh) or [conda](https://docs.conda.io)/
  [miniconda](https://docs.anaconda.com/miniconda/) for dependency
  management (Python, Java, Maven are all installed through it — no
  system-wide Java/Maven install needed).

VALIS depends on [Bio-Formats](https://www.openmicroscopy.org/bio-formats/)
(Java) for reading proprietary microscopy formats such as the Xenium
OME-TIFFs used here. `align_to_xenium.py` auto-detects `JAVA_HOME` from the
`java` binary on `PATH` at runtime, so as long as you run it inside the
pixi/conda environment (which puts its own `openjdk` on `PATH`), no further
Java configuration is needed.

## Setup

### Option A: pixi (recommended)

1. Install pixi if you don't already have it: https://pixi.prefix.dev/latest/installation/


2. From the project root, install all dependencies (Python, OpenJDK,
   Maven, and the `valis-wsi` package) into a local environment:

   ```bash
   pixi install
   ```

3. Run commands with `pixi run`, e.g. `pixi run python align_to_xenium.py ...`.

### Option B: conda

If you'd rather not install pixi, a plain conda environment works too:

```bash
conda create -n valis_align -c conda-forge "openjdk>=25,<26" "python>=3.14,<3.15" maven
conda activate valis_align
pip install "valis-wsi>=1.2.0,<2"
```

Then run commands with `conda activate valis_align` active, e.g.
`python align_to_xenium.py ...`.

## Usage

Run the script through pixi (or with your conda environment activated) so
it picks up the managed Python/Java environment:

```bash
pixi run python align_to_xenium.py <moving_image> <dapi_image> [out_dir] [--channel CHANNEL]

# or, with conda:
conda activate valis_align
python align_to_xenium.py <moving_image> <dapi_image> [out_dir] [--channel CHANNEL]
```

**Arguments:**

- `moving_image` — path to the image being aligned: an H&E slide, or a
  multi-channel IF image with a channel name containing "dapi" (or
  "violet-z").
- `dapi_image` — path to the static 10x Xenium DAPI OME-TIFF. Must be left
  in its original directory (Bio-Formats needs the sibling per-channel
  files alongside it to resolve the multi-file OME group).
- `out_dir` (optional) — where to write the output CSV and VALIS's
  intermediate registration results. Defaults to
  `<moving_image_name>_aligned` next to the moving image.
- `--channel CHANNEL` (optional) — exact name of the moving image's
  DAPI/nuclear channel, bypassing autopick. Also forces IF handling even if
  the image would otherwise be detected as brightfield RGB.

**Example:**

```bash
pixi run python align_to_xenium.py \
  he_slide.ome.tif \
  xenium_output/morphology_focus/morphology_focus_0000.ome.tif \
  results/
```

### Output

The script writes `transform_3x3_xy_crop_to_full.csv` in `out_dir`: a 3x3
affine matrix (comma-separated, CRLF line endings) that maps full-resolution
pixel coordinates in the moving image to full-resolution pixel coordinates
in the static Xenium DAPI image. VALIS's own intermediate registration
artifacts (feature matches, diagnostic overlays, etc.) are written alongside
it under `out_dir/valis_results/`.

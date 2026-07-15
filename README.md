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
pixi run python align_to_xenium.py <moving_image> <dapi_image> [out_dir] [--channel CHANNEL] [--localize]

# or, with conda:
conda activate valis_align
python align_to_xenium.py <moving_image> <dapi_image> [out_dir] [--channel CHANNEL] [--localize]
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
- `--localize` (optional) — enable the small-moving-in-large-static pre-pass
  (see [below](#aligning-a-small-core-to-a-whole-tma-scan---localize)). Use it
  when the moving image covers only a small sub-region of the static scan,
  e.g. a single TMA core vs. a whole-TMA DAPI scan.
- `--localize-candidates N` (optional, default 3) — with `--localize`, how
  many candidate locations to fine-register before keeping the best. Higher
  is more robust to look-alike sibling cores but slower.

**Example:**

```bash
pixi run python align_to_xenium.py \
  he_slide.ome.tif \
  xenium_output/morphology_focus/morphology_focus_0000.ome.tif \
  results/
```

### Aligning a small core to a whole-TMA scan (`--localize`)

By default the moving and static images are assumed to cover roughly the same
field of view (e.g. serial sections of the same tissue). VALIS scales *each*
image independently so its longest side is a fixed pixel budget, which only
makes sense when their physical extents are comparable.

When the moving image is a **small sub-region** of a much larger static
image — for example a single H&E TMA core (~2 mm) aligned to a whole-slide
DAPI TMA scan (~1–2 cm, containing dozens of cores) — that assumption breaks:
the tiny core and the huge scan get squashed to the same pixel size, so the
core's tissue ends up at a completely different micron/pixel than the same
tissue inside the scan, surrounded by look-alike sibling cores. Registration
then fails or locks onto the wrong core.

`--localize` fixes this with a coarse-to-fine pre-pass:

1. Render both images at a **shared micron/pixel** and template-match the
   moving image's nuclei texture (hematoxylin for H&E, DAPI for IF) against
   the static scan to locate which core it corresponds to.
2. Crop the static scan to that core at full resolution.
3. Run the normal fine registration against the crop (now comparable in
   extent, so it just works), trying the top few candidate cores and keeping
   the one with the most feature matches.
4. Compose the crop offset back in, so the exported transform still maps
   moving pixels to **full** static-scan pixels — identical output contract to
   a normal run.

```bash
pixi run python align_to_xenium.py \
  core_11-B4.ome.tiff \
  xenium_output/morphology_focus/morphology_focus_0000.ome.tif \
  results/ \
  --localize
```

### Output

The script writes `transform_3x3_xy_crop_to_full.csv` in `out_dir`: a 3x3
affine matrix (comma-separated, CRLF line endings) that maps full-resolution
pixel coordinates in the moving image to full-resolution pixel coordinates
in the static Xenium DAPI image. VALIS's own intermediate registration
artifacts (feature matches, diagnostic overlays, etc.) are written alongside
it under `out_dir/valis_results/`.

## Running many samples in parallel

Each alignment is single-image and single-process, so batches of samples
can be fanned out with [GNU parallel](https://www.gnu.org/software/parallel/)
instead of looping serially. Put one sample per line in a CSV with the
moving image and DAPI image paths as the first two columns:

```csv
sample1/he_slide.ome.tif,sample1/morphology_focus_0000.ome.tif
sample2/if_image.ome.tif,sample2/morphology_focus_0000.ome.tif
sample3/he_slide.ome.tif,sample3/morphology_focus_0000.ome.tif
```

(no header row — `parallel` treats every line as a job)

Then run:

```bash
parallel --joblog run.log --resume --colsep ',' \
  pixi run python align_to_xenium.py {1} {2} :::: sample.csv
```

- `--colsep ','` splits each CSV row into `{1}`, `{2}`, ... for substitution
  into the command.
- `--joblog run.log` records the exit status of every job.
- `--resume` (paired with `--joblog`) skips jobs already recorded as
  successfully completed in `run.log`, so a failed or interrupted batch can
  be safely re-run without redoing finished alignments.

By default `parallel` runs as many jobs at once as there are CPU cores; pass
`-j N` to cap concurrency (useful since each job also starts its own JVM for
Bio-Formats and can be memory-hungry for large slides). If you'd rather keep
a header row in `sample.csv` for readability, skip it with
`tail -n +2 sample.csv | parallel --joblog run.log --resume --colsep ',' pixi run python align_to_xenium.py {1} {2}`.

If you need to set a per-sample `out_dir` (the script's third, optional
argument) rather than relying on the default location next to each moving
image, add it as a third CSV column and reference it as `{3}` in the
command.

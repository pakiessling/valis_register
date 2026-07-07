#!/usr/bin/env python
"""Rigidly align an H&E or multi-channel IF image to a 10x Xenium DAPI image using VALIS.

The moving image's modality (brightfield H&E vs. immunofluorescence) is
auto-detected from its reader metadata (RGB vs. multi-channel), which is
what determines how it needs to be preprocessed before registration:

* H&E: brightfield RGB. Registration matches the tissue's texture/folds
  (via ColorfulStandardizer) against the DAPI nuclei texture.
* IF: multi-channel fluorescence. Registration is DAPI-to-DAPI: the IF
  image's DAPI channel is located by name (case-insensitive substring
  match, e.g. "DAPI", "Channel1-DAPI", "Nuclear-DAPI-405") and extracted
  via ChannelGetter to match the Xenium DAPI channel.

Computes a 3x3 affine transformation matrix that maps full-resolution
pixel coordinates in the moving image to full-resolution pixel
coordinates in the static Xenium DAPI image, and saves it as a CSV.
"""
import argparse
import os
import shutil
import subprocess
import warnings

import numpy as np

# Xenium morphology_focus OME-TIFFs are part of a multi-file OME group and
# their planes are JPEG2000-compressed. libvips (VALIS's default reader for
# OME-TIFFs) silently decodes that to all-zero pixels here, so force VALIS to
# use its BioFormats reader for this file instead, which handles both the
# compression and the OME channel grouping correctly (and already knows how
# to pick out the "dapi" channel by name via VALIS's default preprocessor).
if not os.environ.get("JAVA_HOME"):
    java_bin = shutil.which("java")
    if java_bin:
        os.environ["JAVA_HOME"] = subprocess.run(
            ["/usr/libexec/java_home"], capture_output=True, text=True
        ).stdout.strip() or os.path.dirname(os.path.dirname(os.path.realpath(java_bin)))

from valis import registration, slide_io, preprocessing, micro_rigid_registrar, valtils, feature_detectors

# These are all expected, benign noise from VALIS's tiled micro-rigid
# refinement: many tiles (edge slivers, background-only regions) legitimately
# have too few keypoints to matter, and the aggregate result across all tiles
# is what determines registration quality (checked separately, see
# "micro rigid registration improved/did not improve alignments" in output).
#
# skimage's deprecation notices for the (still fully working) `.estimate()`
# API, and numpy's "empty slice"/"invalid value" warnings from the same
# empty/near-empty tiles, are equally irrelevant here.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                         message=".*(Mean of empty slice|invalid value encountered).*")

# VALIS's own warning helper (valtils.print_warning) calls
# `warnings.simplefilter('always', ...)` on every invocation, which stomps on
# filters set via the `warnings` module above for the UserWarning category it
# uses -- so its noisiest, benign message (which also triggers a real VALIS
# bug: it calls traceback.format_exc() outside of an except block, which
# prints the literal text "NoneType: None" alongside it) has to be silenced
# by wrapping the function directly instead.
_BENIGN_VALTILS_WARNING_SNIPPETS = (
    "Need at least 4 keypoints for RANSAC filtering",
)
_orig_print_warning = valtils.print_warning


def _quiet_print_warning(msg, *args, **kwargs):
    if any(s in str(msg) for s in _BENIGN_VALTILS_WARNING_SNIPPETS):
        return
    return _orig_print_warning(msg, *args, **kwargs)


valtils.print_warning = _quiet_print_warning

# feature_detectors.py has a single, unconditional `print("detecting features
# in level ... with image shape ...")` per pyramid level, per tile -- with
# hundreds of micro-rigid tiles this is hundreds of lines that convey nothing
# actionable. It's the module's only print() call, so shadowing the builtin
# in its namespace (module globals take precedence over builtins) silences
# just this line without touching real warnings/errors.
feature_detectors.print = lambda *args, **kwargs: None


def get_reader(img_path):
    """Resolve a working SlideReader (class, instance) pair for img_path.

    VALIS picks its default reader (usually the libvips-backed
    VipsSlideReader) based on file extension/format alone, without
    verifying it can actually open the file -- some TIFFs (e.g. certain
    tile sizes/layouts) make libvips's tiff loader raise
    "tile size out of range" at read time. BioFormats reads these fine, so
    fall back to it whenever the "preferred" reader fails to construct.
    """
    reader_cls = slide_io.get_slide_reader(img_path)
    try:
        return reader_cls, reader_cls(img_path)
    except Exception:
        if reader_cls is slide_io.BioFormatsSlideReader:
            raise
        return slide_io.BioFormatsSlideReader, slide_io.BioFormatsSlideReader(img_path)


# Substrings (case-insensitive) that identify a DAPI/nuclear channel by
# name for autopick. "violet-z" covers panels (e.g. some Akoya/Cytek
# configurations) that label the DAPI-equivalent nuclear channel by its
# detector/laser line instead of the stain name.
DAPI_AUTOPICK_SUBSTRINGS = ("dapi", "violet-z")


def find_dapi_channel(reader, requested=None):
    """Resolve the name of the reader's DAPI/nuclear channel.

    If `requested` is given, it must exactly match (case-insensitive) one
    of the reader's channel names -- this lets a caller bypass autopick
    entirely for panels that don't use any recognized naming convention.

    Otherwise, autopick by substring: VALIS's own channel lookup (used for
    the default fluorescence preprocessor) picks the closest-matching
    *whole* channel name via difflib, which only succeeds if the channel
    is named exactly "DAPI". Real IF panels commonly embed it in a longer
    name (e.g. "DAPI (Cy5)", "Channel1-DAPI", "Nuclear-DAPI-405") or use a
    different convention entirely (e.g. "Violet-Z"), so it has to be found
    here via substring search and passed to the preprocessor explicitly.
    """
    channel_names = reader.metadata.channel_names or []

    if requested is not None:
        matches = [c for c in channel_names if c.lower() == requested.lower()]
        if not matches:
            raise ValueError(
                f"Requested channel '{requested}' not found in {reader.src_f}. "
                f"Available channels: {channel_names}"
            )
        return matches[0]

    matches = [c for c in channel_names
               if any(s in c.lower() for s in DAPI_AUTOPICK_SUBSTRINGS)]
    if not matches:
        raise ValueError(
            f"No DAPI/nuclear channel found by name in {reader.src_f} "
            f"(looked for {DAPI_AUTOPICK_SUBSTRINGS}). "
            f"Available channels: {channel_names}. "
            f"Use --channel to specify one explicitly."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple DAPI/nuclear channels found by name in {reader.src_f}: {matches}. "
            f"Use --channel to specify which one to use."
        )
    return matches[0]


def strip_known_suffix(path):
    """Basename of path with its extension removed.

    Handles the compound ".ome.tif"/".ome.tiff" extension as a single unit
    (a plain os.path.splitext would otherwise leave the ".ome" in place).
    """
    name = os.path.basename(path)
    for suffix in (".ome.tiff", ".ome.tif"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def fit_affine(src_xy, dst_xy):
    """Least-squares fit of a 3x3 affine matrix mapping src_xy -> dst_xy."""
    n = src_xy.shape[0]
    ones = np.ones((n, 1))
    src_h = np.hstack([src_xy, ones])  # [n, 3]

    # Solve for each row of the 2x3 affine part independently.
    coeffs, _, _, _ = np.linalg.lstsq(src_h, dst_xy, rcond=None)  # [3, 2]
    A = np.eye(3)
    A[:2, :] = coeffs.T
    return A


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("moving_path", help="Path to the moving image: either an H&E slide, "
                                             "or a multi-channel IF image (with a channel "
                                             "with 'dapi' in its name)")
    parser.add_argument("dapi_path", help="Path to the static 10x Xenium DAPI image")
    parser.add_argument("out_dir", nargs="?", default=None,
                         help="Directory in which to save the transform CSV "
                              "(and VALIS intermediate results). If omitted, defaults to "
                              "'<moving_image_name>_aligned' next to the moving image.")
    parser.add_argument("--channel", default=None,
                         help="Exact name of the moving image's DAPI/nuclear channel to use "
                              "for registration, bypassing autopick (which looks for 'dapi' "
                              "or 'violet-z' in channel names). Forces IF handling even if "
                              "the image would otherwise be detected as brightfield RGB.")
    args = parser.parse_args()

    moving_path = os.path.abspath(args.moving_path)
    dapi_path = os.path.abspath(args.dapi_path)

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(os.path.dirname(moving_path), f"{strip_known_suffix(moving_path)}_aligned")
    os.makedirs(out_dir, exist_ok=True)

    moving_reader_cls, moving_reader = get_reader(moving_path)
    if args.channel is None and moving_reader.metadata.is_rgb:
        modality = "he"
        print(f"{moving_path} detected as brightfield (RGB); using H&E preprocessing")
        moving_processor_dict = None  # let VALIS's default brightfield_processing_cls apply
    else:
        modality = "if"
        moving_dapi_channel = find_dapi_channel(moving_reader, requested=args.channel)
        if args.channel is not None:
            print(f"Using explicitly requested channel '{moving_dapi_channel}' from {moving_path}")
        else:
            print(f"{moving_path} detected as immunofluorescence; "
                  f"using '{moving_dapi_channel}' as its DAPI channel")
        moving_processor_dict = {
            moving_path: [preprocessing.ChannelGetter, {"channel": moving_dapi_channel, "adaptive_eq": True}],
        }

    results_dir = os.path.join(out_dir, "valis_results")

    try:
        # img_list (rather than a src_dir VALIS scans) lets the two images live
        # in their own original directories. This matters for the Xenium DAPI
        # file: it's part of a multi-file OME group (one file per channel),
        # and BioFormats needs the sibling channel files physically present
        # alongside it to resolve the group -- which only works if we read it
        # from its original location instead of a copy/symlink elsewhere.
        registrar = registration.Valis(
            os.path.dirname(dapi_path),
            results_dir,
            name=f"{modality}_to_dapi",
            img_list=[moving_path, dapi_path],
            reference_img_f=dapi_path,
            align_to_reference=True,
            non_rigid_registrar_cls=None,  # rigid/affine only
            micro_rigid_registrar_cls=micro_rigid_registrar.MicroRigidRegistrar,
            # MicroRigidRegistrar refines the rigid/affine fit using tiled,
            # much higher-resolution crops (default 1/8 of full res, in 512px
            # tiles) after the initial low-res (512px thumbnail) rigid pass.
            # It only keeps the refined result if it has at least as many
            # matches as the low-res pass, so it's safe to always enable.
            # Needs the same per-image processor override as the main
            # registration below (for IF: it defaults to the wrong,
            # fuzzy-name-matched DAPI channel; for H&E: it defaults to OD,
            # see brightfield_processing_cls below).
            micro_rigid_registrar_params={
                "processor_dict": moving_processor_dict,
            } if moving_processor_dict else {},
        )
        # VALIS only initializes non_rigid_reg_kwargs when a non-rigid
        # registrar class is given. With it set to None (as above), VALIS's
        # own register() -> cleanup() step crashes with an AttributeError on
        # this missing attribute. Pre-set it so cleanup() has nothing to do.
        registrar.non_rigid_reg_kwargs = {registration.NON_RIGID_REG_CLASS_KEY: None}

        # reader_dict value must be a (class, kwargs) tuple, not a bare
        # instance -- passing an instance hits a code path in VALIS's
        # create_img_reader_dict that reuses a stale reader class left over
        # from the previous image in the loop, silently discarding this
        # override.
        register_kwargs = dict(reader_dict={
            dapi_path: (slide_io.BioFormatsSlideReader, {}),
            moving_path: (moving_reader_cls, {}),
        })
        if modality == "he":
            # brightfield_processing_cls: VALIS's current default brightfield
            # preprocessor (OD) produces a near-blank image for H&E slides with
            # little contrast, which starves feature detection and yields a
            # bad/near-random rigid fit (e.g. only 4 feature matches).
            # ColorfulStandardizer (VALIS's previous default, pre-OD) instead
            # recovers the tissue's texture/folds, matching the DAPI nuclei
            # texture much better -- hundreds to thousands of matches instead of 4.
            register_kwargs["brightfield_processing_cls"] = preprocessing.ColorfulStandardizer
        else:
            register_kwargs["processor_dict"] = moving_processor_dict

        registrar.register(**register_kwargs)

        moving_slide = registrar.get_slide(moving_path)
        dapi_slide = registrar.get_slide(dapi_path)

        w, h = moving_slide.slide_dimensions_wh[0]
        src_pts = np.array([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h],
            [w / 2, h / 2],
        ], dtype=float)

        dst_pts = moving_slide.warp_xy_from_to(
            src_pts, to_slide_obj=dapi_slide, non_rigid=False
        )

        A = fit_affine(src_pts, dst_pts)

        # Match the reference format exactly: shortest round-trip float repr,
        # comma-separated, CRLF row endings, no trailing newline.
        csv_text = "\r\n".join(",".join(repr(float(v)) for v in row) for row in A)
        out_csv = os.path.join(out_dir, "transform_3x3_xy_crop_to_full.csv")
        with open(out_csv, "w", newline="") as fo:
            fo.write(csv_text)

        print(f"Saved affine transform to {out_csv}")
    finally:
        registration.kill_jvm()


if __name__ == "__main__":
    main()

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

# Imported after VALIS on purpose: importing pyvips/cv2 (which load libvips /
# OpenCV native libs) *before* VALIS starts its JVM + torch runtime segfaults
# the interpreter on macOS/Apple Silicon. Keep VALIS's import first.
import cv2
import tifffile
from skimage.color import rgb2hed
from skimage.transform import rotate as sk_rotate

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


# ---------------------------------------------------------------------------
# Small-moving-in-large-static localization (the --localize path)
#
# VALIS scales *each* image independently so its longest side equals
# max_processed_image_dim_px (registration.py). When the moving image is a
# small crop of a much larger static image (e.g. a single H&E TMA core vs. a
# whole-slide DAPI TMA scan), that normalization collapses the tiny core and
# the huge slide to the same pixel size -- so the core's tissue ends up at a
# wildly different micron/pixel than the same tissue inside the static
# thumbnail, surrounded by ~dozens of look-alike sibling cores. Feature
# matching can't survive that scale + distractor mismatch and the fit is
# garbage.
#
# The fix is a coarse-to-fine pre-pass: (1) render both images at a *shared*
# micron/pixel and template-match the moving image's nuclei texture against the
# static one to locate which sub-region it corresponds to; (2) crop the static
# image to that sub-region at full resolution; (3) run the normal fine
# registration against the crop (comparable extent -> it just works); and
# (4) compose the crop offset back in so the exported transform still maps
# moving full-res px -> *full* static full-res px.
# ---------------------------------------------------------------------------

# Coarse localization renders both images near this micron/pixel. Fine enough
# to preserve disambiguating tissue texture, coarse enough that the whole
# static slide is a few thousand pixels (a cheap, exhaustive rotation search).
COARSE_TARGET_UM = 7.0
# The static crop side is the moving image's diagonal (so it covers the core at
# any rotation) times this padding factor, to absorb coarse-localization slop.
CROP_PAD_FACTOR = 1.6
# If a candidate's fine registration yields at least this many feature matches
# it's accepted immediately, without registering the remaining (lower-NCC,
# more likely wrong, and occasionally VALIS-crashing) candidates. A correct
# core produces hundreds to many thousands of matches; wrong cores produce
# only a handful, so this cutoff sits comfortably between the two regimes.
STRONG_MATCH_COUNT = 300


def _nuclei_texture(img, kind):
    """Reduce an image to a band-pass nuclei-density texture map (float32).

    Both modalities are mapped to "where are the nuclei, at tissue scale":
    H&E via hematoxylin color-deconvolution, fluorescence/DAPI via raw
    intensity. Band-pass filtering keeps internal tissue structure (which
    disambiguates the correct core) while discarding the core's disk
    silhouette (which correlates with *every* core) and pixel noise.
    """
    if kind == "he":
        chan = rgb2hed(img[..., :3])[..., 0]  # hematoxylin (nuclei) channel
    else:
        chan = img.astype(np.float32)

    chan = chan.astype(np.float32)
    lo = np.percentile(chan, 1)
    hi = np.percentile(chan, 99) or 1.0
    chan = np.clip((chan - lo) / (hi - lo), 0, 1)

    sig_lo, sig_hi = 1.5, 12.0
    band = cv2.GaussianBlur(chan, (0, 0), sig_lo) - cv2.GaussianBlur(chan, (0, 0), sig_hi)
    support = (cv2.GaussianBlur(chan, (0, 0), sig_hi) > 0.05).astype(np.float32)
    return (band * support).astype(np.float32)


def _dapi_channel_index(reader):
    """Index of the static image's DAPI channel (for slicing coarse/crop reads)."""
    names = reader.metadata.channel_names or []
    for i, c in enumerate(names):
        if any(s in c.lower() for s in DAPI_AUTOPICK_SUBSTRINGS):
            return i
    return 0  # single-channel or unnamed -> first plane


def localize_moving_in_static(moving_reader, static_reader, modality,
                              moving_channel, n_candidates):
    """Find where the moving image sits within the larger static image.

    Returns (candidates, static_um) where candidates is a list of
    (center_x, center_y, score) in full-resolution static pixels, ordered
    best-first, and static_um is the static image's micron/pixel.
    """
    static_dims = static_reader.metadata.slide_dimensions  # [level][w, h]
    static_um = float(static_reader.metadata.pixel_physical_size_xyu[0])
    full_h = static_dims[0][1]

    # Coarsest static pyramid level still finer than COARSE_TARGET_UM.
    lvl = max(i for i, (w, h) in enumerate(static_dims)
              if static_um * (full_h / h) <= COARSE_TARGET_UM)
    coarse_ds = full_h / static_dims[lvl][1]      # full-res px per coarse px
    common_um = static_um * coarse_ds
    dapi_idx = _dapi_channel_index(static_reader)

    static_img = np.asarray(static_reader.slide2image(level=lvl))
    static_img = static_img[..., dapi_idx] if static_img.ndim == 3 else static_img
    static_map = _nuclei_texture(static_img, "dapi")

    # Moving image at the same micron/pixel. For IF, slice the DAPI channel
    # *before* resizing (cv2.resize handles at most 4 channels).
    moving_um = float(moving_reader.metadata.pixel_physical_size_xyu[0])
    moving_img = np.asarray(moving_reader.slide2image(level=0))
    if modality != "he" and moving_img.ndim == 3:
        names = moving_reader.metadata.channel_names or []
        mv_idx = names.index(moving_channel) if moving_channel in names else 0
        moving_img = moving_img[..., mv_idx]
    m_scale = moving_um / common_um
    moving_small = cv2.resize(moving_img, (max(1, round(moving_img.shape[1] * m_scale)),
                                           max(1, round(moving_img.shape[0] * m_scale))),
                              interpolation=cv2.INTER_AREA)
    moving_map = _nuclei_texture(moving_small, "he" if modality == "he" else "dapi")

    print(f"Coarse localization at {common_um:.2f} um/px: "
          f"static {static_map.shape[::-1]} px, moving {moving_map.shape[::-1]} px")

    # Rotation + translation search: for each angle, normalized cross-correlate
    # the rotated moving nuclei map against the static one. Record the global
    # peak per angle, then keep the top-N peaks that sit on *distinct* cores.
    peaks = []  # (score, center_x, center_y) in coarse static px
    for ang in range(0, 360, 4):
        rot = sk_rotate(moving_map, ang, resize=True, order=1,
                        preserve_range=True).astype(np.float32)
        if rot.shape[0] >= static_map.shape[0] or rot.shape[1] >= static_map.shape[1]:
            continue
        res = cv2.matchTemplate(static_map, rot, cv2.TM_CCOEFF_NORMED)
        res[~np.isfinite(res)] = -1.0
        _, score, _, (lx, ly) = cv2.minMaxLoc(res)
        peaks.append((float(score), lx + rot.shape[1] / 2.0, ly + rot.shape[0] / 2.0))

    peaks.sort(key=lambda p: p[0], reverse=True)
    # Non-max suppression (in coarse px) so the N candidates are different
    # cores, not the same core at neighbouring angles. A core spans
    # ~min(moving_map dims).
    min_sep = 0.5 * min(moving_map.shape)
    kept = []  # (score, cx, cy) in coarse px
    for score, cx, cy in peaks:
        if all((cx - kx) ** 2 + (cy - ky) ** 2 > min_sep ** 2 for _, kx, ky in kept):
            kept.append((score, cx, cy))
        if len(kept) >= n_candidates:
            break
    candidates = [(cx * coarse_ds, cy * coarse_ds, score) for score, cx, cy in kept]

    print(f"Top {len(candidates)} candidate location(s) (full-res static px):")
    for i, (cx, cy, score) in enumerate(candidates):
        print(f"  #{i + 1}: center=({cx:.0f}, {cy:.0f})  NCC={score:.3f}")
    return candidates, static_um


def read_static_dapi_region(static_reader, x0, y0, w, h):
    """Read one full-res region of *just* the static DAPI channel via BioFormats.

    VALIS's slide2image reads every channel of the region (4x the I/O here);
    we only need DAPI, so read that single plane directly through the
    BioFormats reader -- meaningfully faster for the large (~20k px) crops.
    """
    dapi_idx = _dapi_channel_index(static_reader)
    rdr, _ = static_reader._get_bf_objects()
    try:
        rdr.setSeries(static_reader.series or 0)
        rdr.setResolution(0)  # full resolution
        np_dtype, _ = slide_io.bf_to_numpy_dtype(rdr.getPixelType(), rdr.isLittleEndian())
        plane = rdr.getIndex(0, dapi_idx, 0)  # (z, c, t)
        buffer = rdr.openBytes(plane, int(x0), int(y0), int(w), int(h))
        return np.frombuffer(bytes(buffer), np_dtype).reshape((int(h), int(w)))
    finally:
        rdr.close()


def crop_static_dapi(static_reader, center_xy, moving_reader, static_um, out_path):
    """Crop a full-res square of the static DAPI channel around center_xy.

    The side covers the moving image's diagonal (any rotation) times
    CROP_PAD_FACTOR. Written as a plain single-channel OME-TIFF named "DAPI"
    (so VALIS's default fluorescence preprocessing picks it up) preserving the
    static micron/pixel. Returns (x0, y0) -- the crop's top-left in full-res
    static px, needed to compose the transform back to full-static coords.
    """
    static_w, static_h = static_reader.metadata.slide_dimensions[0]
    moving_um = float(moving_reader.metadata.pixel_physical_size_xyu[0])
    mw, mh = moving_reader.metadata.slide_dimensions[0]
    moving_diag_um = np.hypot(mw, mh) * moving_um
    side = int(round(moving_diag_um / static_um * CROP_PAD_FACTOR))
    side = min(side, int(static_w), int(static_h))

    cx, cy = center_xy
    x0 = int(round(min(max(cx - side / 2, 0), static_w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), static_h - side)))

    region = read_static_dapi_region(static_reader, x0, y0, side, side)
    tifffile.imwrite(out_path, np.ascontiguousarray(region), ome=True, photometric="minisblack",
                     metadata={"axes": "YX", "PhysicalSizeX": static_um, "PhysicalSizeY": static_um,
                               "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
                               "Channel": {"Name": ["DAPI"]}})
    print(f"  cropped static region {side}x{side} px at ({x0}, {y0}) -> {os.path.basename(out_path)}")
    return x0, y0


def register_pair(moving_path, moving_reader_cls, moving_processor_dict, modality,
                  static_path, static_reader_cls, results_dir, name):
    """Rigid/affine-register moving onto static; return the VALIS registrar.

    Shared by the normal path (static = the whole Xenium DAPI) and the
    --localize path (static = a cropped DAPI region).
    """
    registrar = registration.Valis(
        os.path.dirname(static_path),
        results_dir,
        name=name,
        img_list=[moving_path, static_path],
        reference_img_f=static_path,
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

    # reader_dict value must be a (class, kwargs) tuple, not a bare
    # instance -- passing an instance hits a code path in VALIS's
    # create_img_reader_dict that reuses a stale reader class left over
    # from the previous image in the loop, silently discarding this
    # override.
    register_kwargs = dict(reader_dict={
        static_path: (static_reader_cls, {}),
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
    return registrar


def n_rigid_matches(registrar, moving_path):
    """Number of feature matches the rigid fit found between moving and static.

    A high count means the fit locked onto real, geometrically-consistent
    structure; a wrong core yields only a handful. Used to pick among
    candidate crops in the --localize path.
    """
    matched = registrar.get_slide(moving_path).xy_matched_to_prev
    return 0 if matched is None else len(matched)


def moving_to_static_affine(registrar, moving_path, static_path, offset_xy=(0.0, 0.0)):
    """3x3 affine mapping moving full-res px -> static full-res px.

    offset_xy shifts the result by the static crop's top-left, so a transform
    fit against a crop is expressed in full-static coordinates.
    """
    moving_slide = registrar.get_slide(moving_path)
    static_slide = registrar.get_slide(static_path)

    w, h = moving_slide.slide_dimensions_wh[0]
    src_pts = np.array([[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2]], dtype=float)
    dst_pts = moving_slide.warp_xy_from_to(src_pts, to_slide_obj=static_slide, non_rigid=False)
    dst_pts = dst_pts + np.asarray(offset_xy, dtype=float)
    return fit_affine(src_pts, dst_pts)


def write_transform_csv(A, out_dir):
    # Match the reference format exactly: shortest round-trip float repr,
    # comma-separated, CRLF row endings, no trailing newline.
    csv_text = "\r\n".join(",".join(repr(float(v)) for v in row) for row in A)
    out_csv = os.path.join(out_dir, "transform_3x3_xy_crop_to_full.csv")
    with open(out_csv, "w", newline="") as fo:
        fo.write(csv_text)
    print(f"Saved affine transform to {out_csv}")
    return out_csv


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
    parser.add_argument("--localize", action="store_true",
                         help="Enable the small-moving-in-large-static pre-pass: when the "
                              "moving image covers only a small sub-region of the static image "
                              "(e.g. a single H&E/IF TMA core vs. a whole-slide DAPI TMA scan), "
                              "coarsely locate the core within the scan, crop the static image "
                              "to it, and register against the crop. Without this, VALIS's "
                              "per-image downscaling makes the two images incomparable and the "
                              "fit fails.")
    parser.add_argument("--localize-candidates", type=int, default=3,
                         help="How many candidate locations the --localize pre-pass registers "
                              "against before keeping the best (by feature-match count). Higher "
                              "is more robust to look-alike sibling cores but slower. Default 3.")
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
        moving_channel_name = None
        print(f"{moving_path} detected as brightfield (RGB); using H&E preprocessing")
        moving_processor_dict = None  # let VALIS's default brightfield_processing_cls apply
    else:
        modality = "if"
        moving_dapi_channel = find_dapi_channel(moving_reader, requested=args.channel)
        moving_channel_name = moving_dapi_channel
        if args.channel is not None:
            print(f"Using explicitly requested channel '{moving_dapi_channel}' from {moving_path}")
        else:
            print(f"{moving_path} detected as immunofluorescence; "
                  f"using '{moving_dapi_channel}' as its DAPI channel")
        moving_processor_dict = {
            moving_path: [preprocessing.ChannelGetter, {"channel": moving_dapi_channel, "adaptive_eq": True}],
        }

    results_dir = os.path.join(out_dir, "valis_results")
    # The static Xenium DAPI is JPEG2000-compressed and part of a multi-file
    # OME group; only BioFormats reads it correctly (see get_reader / the
    # module docstring), so it's always read with BioFormatsSlideReader.
    dapi_reader_cls = slide_io.BioFormatsSlideReader

    try:
        if args.localize:
            static_reader = dapi_reader_cls(dapi_path)  # BioFormats (JPEG2000 group)
            candidates, static_um = localize_moving_in_static(
                moving_reader, static_reader, modality,
                moving_channel_name, args.localize_candidates,
            )
            if not candidates:
                raise RuntimeError("Coarse localization found no candidate locations.")

            # Phase 1 -- crop every candidate region while the JVM is alive.
            # Cropping needs BioFormats (the JVM), but VALIS kills the JVM
            # whenever a registration errors out (e.g. a wrong core yields a
            # degenerate match set -> "SVD did not converge"), after which no
            # further crop can be read ("JVM cannot be restarted"). Doing all
            # BioFormats reads *before* any registration means a later crash
            # can't stop us from evaluating the remaining candidates -- one of
            # which may be the correct core even if an earlier one failed.
            crops = []  # (index, crop_path, x0, y0, score)
            for i, (cx, cy, score) in enumerate(candidates):
                print(f"\n=== Candidate #{i + 1} (NCC={score:.3f}) ===")
                crop_path = os.path.join(out_dir, f"static_crop_{i + 1}.ome.tif")
                x0, y0 = crop_static_dapi(static_reader, (cx, cy), moving_reader,
                                          static_um, crop_path)
                crops.append((i, crop_path, x0, y0, score))

            # Phase 2 -- fine-register the moving image against each crop (best
            # NCC first) and keep the fit with the most feature matches. The
            # crop containing the true counterpart core produces many matches;
            # wrong (look-alike) cores produce only a handful, or fail outright
            # (caught below). Registration reads only plain TIFFs (Vips), so it
            # no longer needs the JVM -- a crash on one candidate can't stop the
            # others. A strong result short-circuits the remaining candidates.
            best = None  # (n_matches, A, index)
            for i, crop_path, x0, y0, score in crops:
                print(f"\n=== Registering candidate #{i + 1} (NCC={score:.3f}) ===")
                cand_results = os.path.join(results_dir, f"candidate_{i + 1}")
                try:
                    registrar = register_pair(
                        moving_path, moving_reader_cls, moving_processor_dict, modality,
                        crop_path, slide_io.VipsSlideReader, cand_results,
                        name=f"{modality}_to_dapi_crop{i + 1}",
                    )
                    n = n_rigid_matches(registrar, moving_path)
                    A = moving_to_static_affine(registrar, moving_path, crop_path,
                                                offset_xy=(x0, y0))
                except Exception as e:
                    print(f"  registration against candidate #{i + 1} failed: {e}")
                    continue
                print(f"  candidate #{i + 1}: {n} rigid feature matches")
                if best is None or n > best[0]:
                    best = (n, A, i + 1)
                if n >= STRONG_MATCH_COUNT:
                    print(f"  strong match (>= {STRONG_MATCH_COUNT}); accepting candidate #{i + 1}")
                    break

            # The full-res DAPI crops are large (hundreds of MB each) and only
            # intermediate -- the exported CSV is the deliverable.
            for _, p, _, _, _ in crops:
                try:
                    os.remove(p)
                except OSError:
                    pass

            if best is None:
                raise RuntimeError("All candidate registrations failed.")
            print(f"\nBest candidate: #{best[2]} ({best[0]} matches)")
            write_transform_csv(best[1], out_dir)
        else:
            # img_list (rather than a src_dir VALIS scans) lets the two images
            # live in their own original directories. This matters for the
            # Xenium DAPI file: it's part of a multi-file OME group (one file
            # per channel), and BioFormats needs the sibling channel files
            # physically present alongside it to resolve the group -- which only
            # works if we read it from its original location.
            registrar = register_pair(
                moving_path, moving_reader_cls, moving_processor_dict, modality,
                dapi_path, dapi_reader_cls, results_dir, name=f"{modality}_to_dapi",
            )
            A = moving_to_static_affine(registrar, moving_path, dapi_path)
            write_transform_csv(A, out_dir)
    finally:
        registration.kill_jvm()


if __name__ == "__main__":
    main()

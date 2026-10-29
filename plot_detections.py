"""Overlay SourceXtractor++ detections on their image.

Two colorings:
  --by flags  (default if no star/galaxy catalog) green = clean, red = flagged (blended/boundary)
      writes <name>_detections.png, <name>_detections_zoom.png, <name>.reg
  --by class  point vs extended sources by SPREAD_MODEL (EXTENDED_CLASS) from <name>_stargal.fits
      writes <name>_classes.png, <name>_classes_zoom.png, <name>_classes.reg
  --by class_star  point vs extended sources by CLASS_STAR from <name>_stargal.fits
      writes <name>_classstar.png, <name>_classstar_zoom.png, <name>_classstar.reg
  --by combined  high-confidence stars/galaxies where SPREAD_MODEL and CLASS_STAR agree, mixed otherwise
      writes <name>_combined.png, <name>_combined_zoom.png, <name>_combined.reg
Without --by, all three class colorings are drawn if <name>_stargal.fits exists, else flags.

The .reg files are DS9 region files: open the image (the path printed by images.py, HDU 1) in DS9, then load them.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D

from images import find_images

# Ellipse a/b are 1-sigma moments of the light profile; scale them up so they cover the object.
ELLIPSE_SCALE = 3.0


def flag_groups(cat):
    """(mask, color, linestyle, label) per group, coloring by SourceXtractor++ flags."""
    flagged = cat["source_flags"] != 0
    return [(~flagged, "lime", "solid", "clean"), (flagged, "red", "solid", "flagged")]


def class_groups(cat, max_mag):
    """(mask, color, linestyle, label) per group, coloring by EXTENDED_CLASS (spread_model.py).

    SPREAD_MODEL stops separating stars from galaxies near the detection limit, so sources fainter
    than max_mag (or without a class) are drawn as unclassified.
    """
    cls = np.asarray(cat["extended_class"])
    faint = ~(np.asarray(cat["auto_mag"]) <= max_mag) | (cls < 0)
    return [
        (~faint & (cls == 0), "cyan", "solid", "point source, high confidence"),
        (~faint & (cls == 1), "cyan", "dashed", "point source, likely"),
        (~faint & (cls == 2), "orange", "dashed", "extended, likely"),
        (~faint & (cls == 3), "orange", "solid", "extended, high confidence"),
        (faint, "0.6", "dotted", f"unclassified (mag > {max_mag:g})"),
    ]


def class_star_groups(cat, max_mag):
    """(mask, color, linestyle, label) per group, coloring by CLASS_STAR (class_star.py).

    CLASS_STAR drifts to ~0.4 near the detection limit, so faint sources are drawn as unclassified.
    """
    cs = np.asarray(cat["class_star"])
    faint = ~(np.asarray(cat["auto_mag"]) <= max_mag) | ~np.isfinite(cs)
    return [
        (~faint & (cs >= 0.9), "cyan", "solid", "point source, CLASS_STAR >= 0.9"),
        (~faint & (cs >= 0.5) & (cs < 0.9), "cyan", "dashed", "point source, likely (0.5-0.9)"),
        (~faint & (cs >= 0.1) & (cs < 0.5), "orange", "dashed", "extended, likely (0.1-0.5)"),
        (~faint & (cs < 0.1), "orange", "solid", "extended, CLASS_STAR < 0.1"),
        (faint, "0.6", "dotted", f"unclassified (mag > {max_mag:g})"),
    ]


def combined_groups(cat, max_mag):
    """(mask, color, linestyle, label) per group: both indicators confident and agreeing, or mixed."""
    cls = np.asarray(cat["extended_class"])
    cs = np.asarray(cat["class_star"])
    faint = ~(np.asarray(cat["auto_mag"]) <= max_mag) | (cls < 0) | ~np.isfinite(cs)
    star = ~faint & (cls == 0) & (cs >= 0.9)
    galaxy = ~faint & (cls == 3) & (cs < 0.1)
    return [
        (star, "cyan", "solid", "star: EXTENDED_CLASS 0 and CLASS_STAR >= 0.9"),
        (galaxy, "orange", "solid", "galaxy: EXTENDED_CLASS 3 and CLASS_STAR < 0.1"),
        (~faint & ~star & ~galaxy, "magenta", "solid", "mixed"),
        (faint, "0.6", "dotted", f"unclassified (mag > {max_mag:g})"),
    ]


def draw(ax, image, cat, groups, x0=0, y0=0):
    norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=AsinhStretch(0.1))
    ax.imshow(image, origin="lower", cmap="gray", norm=norm,
              extent=(x0 - 0.5, x0 + image.shape[1] - 0.5, y0 - 0.5, y0 + image.shape[0] - 0.5))

    # Catalog pixel coordinates follow the FITS convention (first pixel centered at 1).
    x = cat["pixel_centroid_x"] - 1
    y = cat["pixel_centroid_y"] - 1
    a = np.maximum(cat["ellipse_a"], 0.5) * ELLIPSE_SCALE
    b = np.maximum(cat["ellipse_b"], 0.5) * ELLIPSE_SCALE
    handles = []
    for mask, color, style, label in groups:
        ax.add_collection(EllipseCollection(
            2 * a[mask], 2 * b[mask], cat["ellipse_theta"][mask],
            units="xy", offsets=np.column_stack([x[mask], y[mask]]),
            offset_transform=ax.transData, facecolors="none",
            edgecolors=color, linewidths=0.8, linestyles=style,
        ))
        handles.append(Line2D([], [], color=color, linestyle=style, label=f"{label} ({mask.sum()})"))
    ax.set_xlim(x0 - 0.5, x0 + image.shape[1] - 0.5)
    ax.set_ylim(y0 - 0.5, y0 + image.shape[0] - 0.5)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.legend(handles=handles, loc="upper right", fontsize=8)


def write_ds9(path: Path, cat, groups):
    ds9_colors = {"lime": "green", "0.6": "gray"}
    ds9_dash = {"solid": "", "dashed": " dash=1", "dotted": " dash=1 width=1"}
    lines = ["# Region file format: DS9", "image"]
    for mask, color, style, _ in groups:
        for row in cat[mask]:
            a = max(row["ellipse_a"], 0.5) * ELLIPSE_SCALE
            b = max(row["ellipse_b"], 0.5) * ELLIPSE_SCALE
            lines.append(
                f"ellipse({row['pixel_centroid_x']:.2f},{row['pixel_centroid_y']:.2f},"
                f"{a:.2f},{b:.2f},{row['ellipse_theta']:.1f}) "
                f"# color={ds9_colors.get(color, color)}{ds9_dash[style]} text={{{row['source_id']}}}"
            )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--by", choices=["flags", "class", "class_star", "combined"], default=None,
                        help="coloring (default: class, class_star and combined if <name>_stargal.fits exists, else flags)")
    parser.add_argument("--max-mag", type=float, default=24.5,
                        help="with class colorings, sources fainter than this are drawn as unclassified")
    parser.add_argument("--zoom", type=int, default=600, help="side of the zoomed crop, in pixels")
    parser.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"),
                        help="FITS pixel center of the zoomed crop (default: image center)")
    parser.add_argument("--only", default="", help="only plot images whose name starts with this")
    args = parser.parse_args()

    for im in find_images(args.samples):
        name = im.name
        catalog_path = args.output / f"{name}_cat.fits"
        if not name.startswith(args.only) or not catalog_path.exists():
            continue
        stargal_path = args.output / f"{name}_stargal.fits"
        modes = [args.by] if args.by else (["class", "class_star", "combined"] if stargal_path.exists() else ["flags"])
        if any(m != "flags" for m in modes) and not stargal_path.exists():
            print(f"{name}: no {stargal_path.name}, run ./run_stargal.sh first")
            continue
        for by in modes:
            plot_image(args, im, catalog_path, stargal_path, by)


def plot_image(args, im, catalog_path, stargal_path, by):
    name = im.name
    if by == "class":
        cat = fits.getdata(stargal_path, 1)
        groups_for = lambda c: class_groups(c, args.max_mag)
        suffix, title = "classes", "SPREAD_MODEL classes"
    elif by == "class_star":
        cat = fits.getdata(stargal_path, 1)
        groups_for = lambda c: class_star_groups(c, args.max_mag)
        suffix, title = "classstar", "CLASS_STAR classes"
    elif by == "combined":
        cat = fits.getdata(stargal_path, 1)
        groups_for = lambda c: combined_groups(c, args.max_mag)
        suffix, title = "combined", "SPREAD_MODEL + CLASS_STAR"
    else:
        cat = fits.getdata(catalog_path, 1)
        groups_for = flag_groups
        suffix, title = "detections", "flags"
    reg_name = f"{name}.reg" if by == "flags" else f"{name}_{suffix}.reg"
    image = im.data()
    weight = im.weight()
    if weight is not None:
        image = np.where(weight > 0, image, np.median(image[weight > 0]))  # keep bad pixels out of the stretch

    fig, ax = plt.subplots(figsize=(12, 12))
    draw(ax, image, cat, groups_for(cat))
    ax.set_title(f"{name}: {len(cat)} detections, {title}")
    fig.savefig(args.output / f"{name}_{suffix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if args.center:
        cx, cy = int(round(args.center[0] - 1)), int(round(args.center[1] - 1))
    else:
        cy, cx = (s // 2 for s in image.shape)
    half = args.zoom // 2
    y0 = int(np.clip(cy - half, 0, image.shape[0] - args.zoom))
    x0 = int(np.clip(cx - half, 0, image.shape[1] - args.zoom))
    crop = image[y0:y0 + args.zoom, x0:x0 + args.zoom]
    x = cat["pixel_centroid_x"] - 1
    y = cat["pixel_centroid_y"] - 1
    inside = (x >= x0 - 20) & (x < x0 + args.zoom + 20) & (y >= y0 - 20) & (y < y0 + args.zoom + 20)
    sub = cat[inside]
    fig, ax = plt.subplots(figsize=(10, 10))
    draw(ax, crop, sub, groups_for(sub), x0, y0)
    ax.set_title(f"{name}: {args.zoom}x{args.zoom} px around ({x0 + half + 1}, {y0 + half + 1}), {title}")
    fig.savefig(args.output / f"{name}_{suffix}_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    write_ds9(args.output / reg_name, cat, groups_for(cat))
    print(f"{name}: wrote {name}_{suffix}.png, {name}_{suffix}_zoom.png and {reg_name}")


if __name__ == "__main__":
    main()

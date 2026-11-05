"""Draw external star/galaxy labels next to the CLASS_STAR, SPREAD_MODEL and MU_MAX calls on each image.

Reads output/<prefix>_truth.fits (compare_truth.py) and each image's <name>_stargal.fits. Only sources
with an external label are drawn: star = Gaia-confirmed (proper motion or parallax) or SDSS type 6,
galaxy = SDSS type 3 (Legacy Surveys labels are used too when compare_truth.py fetched them).

Per image, four panels on the same pixels:
  1. external label: Gaia star, SDSS / Legacy Surveys star, galaxy
  2. CLASS_STAR (>= 0.5 point source), 3. SPREAD_MODEL (EXTENDED_CLASS 0-1 point source) and
  4. MU_MAX vs MAG_AUTO (mu_class 0 point source, mu_max.py), colored by agreement with the external
     label; sources fainter than --max-mag (or without a value) are unclassified.

Writes output/<name>_truth_map.png (whole image, markers) and <name>_truth_map_zoom.png (the --zoom crop
holding the most labeled sources, ellipses), output/<prefix>_mu_max.png (MU_MAX vs MAG_AUTO with the
external labels, stellar locus and cut), and prints the agreement per magnitude bin.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from matplotlib.lines import Line2D

from compare_truth import LABELS
from images import find_images
from plot_detections import draw

MAG_BINS = [(14, 21), (21, 22), (22, 23), (23, 24.5)]
METHODS = {"class_star": "CLASS_STAR (>= 0.5 point)", "spread_model": "SPREAD_MODEL (EXTENDED_CLASS 0-1 point)",
           "mu_max": "MU_MAX vs MAG_AUTO (mu_class 0 point)"}
MU_K = 3.0  # mu_max.mu_max_class default


def as_str(col):
    values = col.filled("") if hasattr(col, "filled") else col
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def as_float(col):
    return np.asarray(col.filled(np.nan) if hasattr(col, "filled") else col, dtype=float)


def external_labels(t):
    """'star', 'galaxy' or '' per source; Gaia astrometry wins over morphology-based survey types."""
    gaia = np.asarray(t["gaia_star"].filled(False) if hasattr(t["gaia_star"], "filled") else t["gaia_star"], bool)
    star, galaxy = gaia.copy(), np.zeros(len(t), bool)
    for survey, (star_types, gal_types) in LABELS.items():
        types = as_str(t[f"{survey}_type"])
        star |= np.isin(types, list(star_types))
        galaxy |= np.isin(types, list(gal_types))
    label = np.full(len(t), "", dtype=object)
    label[galaxy & ~star] = "galaxy"
    label[star] = "star"
    return label, gaia


def calls(t, cat, max_mag):
    """'point', 'extended' or '' (unclassified) per source, for CLASS_STAR, SPREAD_MODEL and MU_MAX."""
    mag = as_float(t["auto_mag"])
    bright = mag <= max_mag
    cs, ec, mc = as_float(t["class_star"]), np.asarray(t["extended_class"]), np.asarray(cat["mu_class"])
    out = {}
    for key, ok, point in [("class_star", np.isfinite(cs), cs >= 0.5), ("spread_model", ec >= 0, ec <= 1),
                           ("mu_max", mc >= 0, mc == 0)]:
        c = np.full(len(t), "", dtype=object)
        c[bright & ok & point] = "point"
        c[bright & ok & ~point] = "extended"
        out[key] = c
    return out


def truth_groups(label, gaia):
    return [
        (gaia, "cyan", "solid", "star (Gaia proper motion / parallax)"),
        ((label == "star") & ~gaia, "dodgerblue", "dashed", "star (SDSS / LS morphology)"),
        (label == "galaxy", "orange", "solid", "galaxy (SDSS / LS)"),
    ]


def call_groups(label, call, saturated):
    labeled = label != ""
    groups = [
        (labeled & (call == "point") & (label == "star"), "cyan", "solid", "point source, agrees"),
        (labeled & (call == "extended") & (label == "galaxy"), "orange", "solid", "extended, agrees"),
        (labeled & (call == "point") & (label == "galaxy"), "red", "solid", "point source, external says galaxy"),
        (labeled & (call == "extended") & (label == "star"), "magenta", "solid", "extended, external says star"),
        (labeled & (call == ""), "0.6", "dotted", "unclassified (faint or no value)"),
    ]
    return [(m, c, s, f"{lab}, {(m & saturated).sum()} saturated" if (m & saturated).any() else lab)
            for m, c, s, lab in groups]


def agreement(label, call):
    both = (label != "") & (call != "")
    hit = ((call == "point") & (label == "star")) | ((call == "extended") & (label == "galaxy"))
    return both.sum(), hit[both].mean() if both.any() else np.nan


def draw_markers(ax, image, cat, groups):
    norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=AsinhStretch(0.1))
    ax.imshow(image, origin="lower", cmap="gray", norm=norm)
    x, y = cat["pixel_centroid_x"] - 1, cat["pixel_centroid_y"] - 1
    handles = []
    for mask, color, style, label in groups:
        ax.scatter(x[mask], y[mask], s=40, facecolors="none", edgecolors=color, linewidths=0.9,
                   linestyles=style)
        handles.append(Line2D([], [], color=color, linestyle=style, label=f"{label} ({mask.sum()})"))
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(-0.5, image.shape[0] - 0.5)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.legend(handles=handles, loc="upper right", fontsize=7)


def busiest_crop(x, y, shape, size, step=100):
    """Lower-left corner of the size x size crop holding the most of the given positions."""
    ny, nx = shape
    best, corner = -1, (0, 0)
    for y0 in range(0, max(ny - size, 0) + 1, step):
        for x0 in range(0, max(nx - size, 0) + 1, step):
            n = np.sum((x >= x0) & (x < x0 + size) & (y >= y0) & (y < y0 + size))
            if n > best:
                best, corner = n, (x0, y0)
    return corner


def panels(name, image, cat, label, gaia, call, saturated, path, zoom):
    x, y = np.asarray(cat["pixel_centroid_x"]) - 1, np.asarray(cat["pixel_centroid_y"]) - 1
    if zoom:
        x0, y0 = busiest_crop(x[label != ""], y[label != ""], image.shape, zoom)
        crop = image[y0:y0 + zoom, x0:x0 + zoom]
        inside = (x >= x0 - 20) & (x < x0 + zoom + 20) & (y >= y0 - 20) & (y < y0 + zoom + 20)
        label, gaia = np.where(inside, label, ""), gaia & inside
        fig, axes = plt.subplots(2, 2, figsize=(22, 22))
    else:
        fig, axes = plt.subplots(1, 4, figsize=(28, 15))

    titles = ["external label (Gaia / SDSS)"] + list(METHODS.values())
    group_sets = [truth_groups(label, gaia)] + [call_groups(label, call[key], saturated) for key in METHODS]
    keys = [None] + list(METHODS)
    for ax, title, groups, key in zip(axes.ravel(), titles, group_sets, keys):
        if key:
            n, acc = agreement(np.where(saturated, "", label), call[key])
            title = f"{title}\n{acc:.1%} of {n} unsaturated agree with the external label"
        if zoom:
            draw(ax, crop, cat, groups, x0, y0)
        else:
            draw_markers(ax, image, cat, groups)
        ax.set_title(title, fontsize=12)
        ax.legend(handles=ax.get_legend().legend_handles, loc="upper right", fontsize=9, framealpha=0.85)
    where = f", {zoom}x{zoom} px from ({x0 + 1}, {y0 + 1})" if zoom else ""
    fig.suptitle(f"{name}: {(label != '').sum()} sources with an external label{where}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110 if zoom else 130, bbox_inches="tight")
    plt.close(fig)


def summary(rows, title):
    print(f"\n{title} (agreement = point/star or extended/galaxy matches the external label; "
          "star ok / gal ok = fraction of external stars / galaxies classified as such)")
    print("  mag            | stars galaxies |" + "|".join(f" {name.split()[0]:12s} agree star ok gal ok "
                                                         for name in METHODS.values()))
    label, mag, sat = rows["label"], rows["mag"], rows["saturated"]
    for lo, hi, saturated in [(lo, hi, False) for lo, hi in MAG_BINS + [(14, 24.5)]] + [(0, 99, True)]:
        b = (mag >= lo) & (mag < hi) & (sat == saturated) & (label != "")
        s, g = b & (label == "star"), b & (label == "galaxy")
        cells = []
        for key in METHODS:
            c = rows[key]
            f = lambda sel, want: f"{np.mean(c[sel] == want):6.1%}" if sel.sum() else "    - "
            n, acc = agreement(label[b], c[b])
            cells.append(f"             {acc:6.1%} {f(s & (c != ''), 'point')} {f(g & (c != ''), 'extended')} "
                         if n else "                -      -      -  ")
        span = "saturated" if saturated else ("all unsat." if (lo, hi) == (14, 24.5) else f"{lo:4.1f}-{hi:4.1f}")
        print(f"  {span:14s} | {s.sum():5d} {g.sum():8d} |" + "|".join(cells))


def plot_mu_max(rows, prefix, path):
    """MU_MAX vs MAG_AUTO and the offset from the stellar locus, colored by the external labels."""
    mag, mu, off, thr = rows["mag"], rows["mu"], rows["mu_offset"], rows["mu_threshold"]
    label, gaia, sat = rows["label"], rows["gaia"], rows["saturated"]
    locus, scatter, det = (np.median(rows[k]) for k in ("mu_locus", "mu_scatter", "det_thresh"))
    m = np.linspace(14, 25.5, 300)
    cut = MU_K * np.hypot(scatter, 2.5 / np.log(10) / det * 10 ** (-0.4 * (np.median(thr) - (m + locus))))

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, y in zip(axes, (mu, off)):
        for sel, color, name, size in [(~sat & (label == "galaxy"), "orange", "galaxy (SDSS)", 2),
                                       (~sat & (label == "star") & ~gaia, "dodgerblue", "star (SDSS)", 3),
                                       (~sat & gaia, "cyan", "star (Gaia)", 9),
                                       (sat & (label != ""), "red", "saturated, labeled", 9)]:
            ax.scatter(mag[sel], y[sel], s=size, c=color, alpha=0.6, label=f"{name} ({sel.sum()})",
                       edgecolors="k" if color == "cyan" else "none", linewidths=0.3)
        ax.set(xlabel="auto_mag (SE++)", xlim=(14, 25.5))
    axes[0].plot(m, m + locus, "k-", lw=0.8, label=f"stellar locus (mu_max - mag = {locus:+.2f})")
    axes[0].plot(m, m + locus + cut, "k--", lw=0.8, label=f"point / extended cut ({MU_K:g} sigma)")
    axes[0].axhline(np.median(thr), color="0.4", ls=":", label=f"mu_threshold ({np.median(thr):.2f})")
    axes[0].set(ylabel="mu_max [mag/arcsec$^2$]", ylim=(26.5, 15.5),
                title=f"{prefix}: MU_MAX vs MAG_AUTO, external labels")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].plot(m, cut, "k--", lw=0.8)
    axes[1].set(ylabel="mu_max - auto_mag - stellar locus [mag]", ylim=(3, -1),
                title="offset from the stellar locus (above the dashed line = point source)")
    axes[0].legend(fontsize=9, markerscale=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--only", default="SUPA", help="images whose name starts with this (default: SUPA)")
    parser.add_argument("--max-mag", type=float, default=24.5, help="fainter sources are drawn as unclassified")
    parser.add_argument("--zoom", type=int, default=500, help="side of the zoomed crop, in pixels")
    args = parser.parse_args()

    truth_path = args.output / f"{args.only}_truth.fits"
    if not truth_path.exists():
        print(f"no {truth_path.name}, run compare_truth.py first")
        return
    truth = Table.read(truth_path)
    images = as_str(truth["image"])
    rows = {}

    for im in find_images(args.samples):
        sel = images == im.name
        if not im.name.startswith(args.only) or not sel.any():
            continue
        t = truth[sel]
        cat = Table.read(args.output / f"{im.name}_stargal.fits")
        if len(cat) != len(t) or not np.all(np.asarray(cat["source_id"]) == np.asarray(t["source_id"])):
            raise ValueError(f"{im.name}: {truth_path.name} is out of date, rerun compare_truth.py")
        if "mu_class" not in cat.colnames:
            raise ValueError(f"{im.name}: no mu_class in {im.name}_stargal.fits, rerun spread_model.py")
        label, gaia = external_labels(t)
        call = calls(t, cat, args.max_mag)
        saturated = (np.asarray(t["source_flags"]) & 4) > 0
        n = len(t)
        for key, value in [("label", label), ("gaia", gaia), ("mag", as_float(t["auto_mag"])), ("saturated", saturated),
                           ("mu", as_float(cat["mu_max"])), ("mu_offset", as_float(cat["mu_locus_offset"])),
                           ("mu_threshold", as_float(cat["mu_threshold"])),
                           ("mu_locus", np.full(n, cat.meta["MULOCUS"])), ("mu_scatter", np.full(n, cat.meta["MUSCATT"])),
                           ("det_thresh", np.full(n, cat.meta["DETTHR"]))] + list(call.items()):
            rows.setdefault(key, []).append(value)

        image = im.data()
        weight = im.weight()
        if weight is not None:
            image = np.where(weight > 0, image, np.median(image[weight > 0]))
        cat = cat.as_array()
        panels(im.name, image, cat, label, gaia, call, saturated, args.output / f"{im.name}_truth_map.png", None)
        panels(im.name, image, cat, label, gaia, call, saturated, args.output / f"{im.name}_truth_map_zoom.png",
               args.zoom)
        print(f"{im.name}: wrote {im.name}_truth_map.png and {im.name}_truth_map_zoom.png")

    if rows:
        rows = {k: np.concatenate(v) for k, v in rows.items()}
        plot_mu_max(rows, args.only, args.output / f"{args.only}_mu_max.png")
        print(f"wrote {args.only}_mu_max.png")
        summary(rows, f"{args.only}: all images")


if __name__ == "__main__":
    main()

"""Score SPREAD_MODEL / CLASS_STAR against external star/galaxy labels over each image's footprint.

Truth samples (downloaded once into output/truth/):
  - Gaia DR3: stars confirmed by proper motion or parallax at > 5 sigma (bright, G < ~20.5).
  - SDSS DR17 PhotoPrimary: type 6 = star, 3 = galaxy (reliable to r ~ 21).
  - DESI Legacy Surveys DR10 Tractor: TYPE = PSF is a point source, REX/EXP/DEV/SER are galaxies
    (reliable to r ~ 22.5; at the faint end compact galaxies drift into PSF). Skipped, with a
    warning, when the NOIRLab Data Lab service is unavailable; rerun later to add it.

Writes output/<prefix>_truth.fits (all matched sources of the selected images) and <prefix>_truth.png.
"""
import argparse
import csv
import io
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack
from astropy.wcs import WCS, FITSFixedWarning

from images import find_images

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
LS_TAP = "https://datalab.noirlab.edu/tap/sync"
SDSS_SQL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
LABELS = {"sdss": ({"6"}, {"3"}), "ls": ({"PSF"}, {"REX", "EXP", "DEV", "SER"})}
MATCH_RADIUS_ARCSEC = 0.7
MAG_BINS = [(14, 18), (18, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25)]


def footprint(im):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        header = im.header()
        w = WCS(header).celestial
    ny, nx = header["NAXIS2"], header["NAXIS1"]
    corners = w.all_pix2world([[0, 0], [nx, 0], [0, ny], [nx, ny]], 0)
    margin = 10 / 3600
    dec0 = np.radians(corners[:, 1].mean())
    return (corners[:, 0].min() - margin / np.cos(dec0), corners[:, 0].max() + margin / np.cos(dec0),
            corners[:, 1].min() - margin, corners[:, 1].max() + margin)


def tap(url, query, path):
    if not path.exists():
        data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}).encode()
        path.write_bytes(urllib.request.urlopen(url, data, timeout=600).read())
    return list(csv.DictReader(io.StringIO(path.read_text())))


def sdss(query, path):
    if not path.exists():
        url = SDSS_SQL + "?" + urllib.parse.urlencode({"cmd": query, "format": "csv"})
        path.write_bytes(urllib.request.urlopen(url, timeout=600).read())
    return list(csv.DictReader(io.StringIO(path.read_text().removeprefix("#Table1\n"))))


def column(rows, key, cast=float):
    return np.array([cast(r[key]) if r[key] not in ("", None) else np.nan for r in rows]) if cast is float \
        else np.array([cast(r[key]) for r in rows])


def nearest(ra, dec, ref_ra, ref_dec, radius_arcsec):
    """Index into the reference of the nearest source within radius for each source (-1 = none)."""
    cosd = np.cos(np.radians(np.median(dec)))
    x, y = ra * cosd * 3600, dec * 3600
    rx, ry = ref_ra * cosd * 3600, ref_dec * 3600
    order = np.argsort(ry)
    rys = ry[order]
    lo, hi = np.searchsorted(rys, y - radius_arcsec), np.searchsorted(rys, y + radius_arcsec)
    idx = np.full(len(ra), -1)
    for i in range(len(ra)):
        cand = order[lo[i]:hi[i]]
        if len(cand):
            d = np.hypot(rx[cand] - x[i], ry[cand] - y[i])
            j = np.argmin(d)
            if d[j] < radius_arcsec:
                idx[i] = cand[j]
    return idx


def label_image(im, output):
    t = Table.read(output / f"{im.name}_stargal.fits")
    ra0, ra1, de0, de1 = footprint(im)
    truth = output / "truth"
    truth.mkdir(exist_ok=True)
    box = f"ra BETWEEN {ra0} AND {ra1} AND dec BETWEEN {de0} AND {de1}"
    gaia = tap(GAIA_TAP, "SELECT ra, dec, phot_g_mean_mag, pmra, pmdec, pmra_error, pmdec_error, parallax_over_error "
               f"FROM gaiadr3.gaia_source WHERE {box}", truth / f"{im.name}_gaia.csv")
    surveys = {"sdss": sdss("SELECT ra, dec, type, psfMag_r AS mag_r FROM PhotoPrimary WHERE "
                            f"ra BETWEEN {ra0} AND {ra1} AND dec BETWEEN {de0} AND {de1}", truth / f"{im.name}_sdss.csv")}
    try:
        surveys["ls"] = tap(LS_TAP, "SELECT ra, dec, type, mag_r FROM ls_dr10.tractor WHERE brick_primary = 1 AND "
                            f"{box}", truth / f"{im.name}_ls.csv")
    except urllib.error.URLError as err:
        print(f"{im.name}: Legacy Surveys unavailable ({err}); continuing without it")
        surveys["ls"] = []

    ra, dec = np.asarray(t["world_centroid_alpha"]), np.asarray(t["world_centroid_delta"])
    g_ra, g_dec = column(gaia, "ra"), column(gaia, "dec")
    pm_sig = np.hypot(column(gaia, "pmra") / column(gaia, "pmra_error"), column(gaia, "pmdec") / column(gaia, "pmdec_error"))
    g_star = (pm_sig > 5) | (column(gaia, "parallax_over_error") > 5)
    gi = nearest(ra, dec, g_ra, g_dec, MATCH_RADIUS_ARCSEC)
    t["gaia_star"] = (gi >= 0) & g_star[np.maximum(gi, 0)]

    t["image"] = im.name
    summary = [f"{im.name}: {len(t)} SE++ sources, {t['gaia_star'].sum()} Gaia-confirmed stars"]
    cosd = np.cos(np.radians(np.median(dec)))
    for survey, rows in surveys.items():
        if not rows:
            t[f"{survey}_type"], t[f"{survey}_mag_r"] = "", np.nan
            continue
        s_ra, s_dec, s_type = column(rows, "ra"), column(rows, "dec"), column(rows, "type", str)
        si = nearest(ra, dec, s_ra, s_dec, MATCH_RADIUS_ARCSEC)
        m = si >= 0
        t[f"{survey}_type"] = np.where(m, s_type[np.maximum(si, 0)], "")
        t[f"{survey}_mag_r"] = np.where(m, column(rows, "mag_r")[np.maximum(si, 0)], np.nan)
        dra = (ra[m] - s_ra[si[m]]) * cosd * 3600
        ddec = (dec[m] - s_dec[si[m]]) * 3600
        summary.append(f"{m.sum()} matched to {survey.upper()} (offset {np.median(dra):+.3f}\" RA, "
                       f"{np.median(ddec):+.3f}\" Dec)")
    print(", ".join(summary))
    keep = ["image", "source_id", "auto_mag", "source_flags", "spread_model", "spreaderr_model", "extended_class",
            "class_star", "mu_max", "sex2_matched", "sex2_class_star", "sex2_extended_class", "gaia_star",
            "sdss_type", "sdss_mag_r", "ls_type", "ls_mag_r"]
    return t[keep]


def report(t, prefix):
    m = np.asarray(t["auto_mag"])
    ec, cs = np.asarray(t["extended_class"]), np.asarray(t["class_star"])
    sat = (np.asarray(t["source_flags"]) & 4) > 0
    sp_point = (ec >= 0) & (ec <= 1)
    classic = np.asarray(t["sex2_matched"])
    c_point = (np.asarray(t["sex2_extended_class"]) >= 0) & (np.asarray(t["sex2_extended_class"]) <= 1)
    c_cs = np.asarray(t["sex2_class_star"])

    def pct(sel, x):
        return f"{np.mean(x[sel]):6.1%}" if sel.sum() else "    - "

    for survey, (star_types, gal_types) in LABELS.items():
        types = np.asarray(t[f"{survey}_type"])
        star, gal = np.isin(types, list(star_types)), np.isin(types, list(gal_types))
        if not (star | gal).any():
            continue
        print(f"\n{prefix}: {survey.upper()} labels (star {sorted(star_types)}, galaxy {sorted(gal_types)}); "
              "saturated (SE++ flag 4) excluded")
        print("  mag (SE++) | stars: n  spread  CS>=0.5  CS>=0.9 | galaxies: n  spread  CS>=0.5  CS>=0.9 "
              "|| classic, same objects: stars spread/CS>=0.5, galaxies spread/CS>=0.5")
        for lo, hi in MAG_BINS:
            b = (m >= lo) & (m < hi) & ~sat
            s, g = b & star, b & gal
            sc, gc = s & classic, g & classic
            print(f"  {lo:4.1f}-{hi:4.1f}  |  {s.sum():5d}  {pct(s, sp_point)} {pct(s, cs >= 0.5)}  {pct(s, cs >= 0.9)} |"
                  f"  {g.sum():6d}  {pct(g, sp_point)} {pct(g, cs >= 0.5)}  {pct(g, cs >= 0.9)} || "
                  f"{pct(sc, c_point)} {pct(sc, c_cs >= 0.5)}   {pct(gc, c_point)} {pct(gc, c_cs >= 0.5)}")
    print("  (spread = fraction called point source by EXTENDED_CLASS 0-1; for galaxies that is the contamination)")

    gs = np.asarray(t["gaia_star"])
    print(f"\n{prefix}: Gaia-confirmed stars called point source")
    for lo, hi in [(10, 16), (16, 18), (18, 20), (20, 22)]:
        for label, extra in [("unsaturated", ~sat), ("saturated", sat)]:
            b = gs & (m >= lo) & (m < hi) & extra
            if b.sum():
                print(f"  mag {lo}-{hi} {label:11s}: n={b.sum():4d}  spread {pct(b, sp_point)}  CS>=0.5 {pct(b, cs >= 0.5)}"
                      f"  CS>=0.9 {pct(b, cs >= 0.9)}")


def plot(t, prefix, path):
    m = np.asarray(t["auto_mag"])
    survey = "ls" if (np.asarray(t["ls_type"]) != "").any() else "sdss"
    star_types, gal_types = LABELS[survey]
    types = np.asarray(t[f"{survey}_type"])
    sat = (np.asarray(t["source_flags"]) & 4) > 0
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, col, ylim in [(axes[0], "spread_model", (-0.02, 0.05)), (axes[1], "class_star", (-0.05, 1.05))]:
        v = np.asarray(t[col])
        g = np.isin(types, list(gal_types)) & ~sat
        s = np.isin(types, list(star_types)) & ~sat
        ax.scatter(m[g], v[g], s=1, c="orange", alpha=0.4, label=f"{survey.upper()} galaxy ({g.sum()})")
        ax.scatter(m[s], v[s], s=2, c="blue", alpha=0.6, label=f"{survey.upper()} star ({s.sum()})")
        ax.scatter(m[sat], v[sat], s=6, c="red", marker="x", label=f"saturated ({sat.sum()})")
        ax.set(xlabel="auto_mag (SE++)", ylabel=col, xlim=(14, 26), ylim=ylim, title=f"{prefix}: {col}")
        ax.legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--only", default="SUPA", help="images whose name starts with this (default: SUPA)")
    args = parser.parse_args()

    tables = [label_image(im, args.output) for im in find_images(args.samples)
              if im.name.startswith(args.only) and (args.output / f"{im.name}_stargal.fits").exists()]
    if not tables:
        print("no star/galaxy catalogs, run ./run_stargal.sh first")
        return
    t = vstack(tables)
    t.write(args.output / f"{args.only}_truth.fits", overwrite=True)
    report(t, args.only)
    plot(t, args.only, args.output / f"{args.only}_truth.png")


if __name__ == "__main__":
    main()

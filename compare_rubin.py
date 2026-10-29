"""Compare SPREAD_MODEL / CLASS_STAR with Rubin's per-visit extendedness (samples/testResults_<small>.fits).

The Rubin tables hold one row per detection per visit. Each detection is matched to the nearest
SourceXtractor++ object within MATCH_RADIUS_ARCSEC, and the per-object Rubin label is the fraction of its
visits flagged extended (extendedness is 0 = point, 1 = extended in each visit).

Writes output/<name>_rubin.fits (stargal catalog + rubin_nvisits, rubin_ext_frac) and <name>_rubin.png.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

MATCH_RADIUS_ARCSEC = 0.5
MIN_VISITS = 5
MAG_BINS = [(14, 18), (18, 20), (20, 22), (22, 23), (23, 24), (24, 24.5), (24.5, 25.5)]


def match_visits(t, rubin, radius_arcsec):
    """Index into t of the nearest object for each Rubin row (-1 = none within radius)."""
    dec0 = np.median(t["world_centroid_delta"])
    cosd = np.cos(np.radians(dec0))
    ox = (np.asarray(t["world_centroid_alpha"]) - np.median(t["world_centroid_alpha"])) * cosd * 3600
    oy = (np.asarray(t["world_centroid_delta"]) - dec0) * 3600
    rx = (np.asarray(rubin["ra"]) - np.median(t["world_centroid_alpha"])) * cosd * 3600
    ry = (np.asarray(rubin["dec"]) - dec0) * 3600

    order = np.argsort(ry)
    rys = ry[order]
    best = np.full(len(rubin), -1)
    best_d = np.full(len(rubin), np.inf)
    lo = np.searchsorted(rys, oy - radius_arcsec)
    hi = np.searchsorted(rys, oy + radius_arcsec)
    for i in range(len(t)):
        cand = order[lo[i]:hi[i]]
        d = np.hypot(rx[cand] - ox[i], ry[cand] - oy[i])
        closer = d < np.minimum(best_d[cand], radius_arcsec)
        best[cand[closer]] = i
        best_d[cand[closer]] = d[closer]
    return best


def plot(t, name, path):
    m = np.asarray(t["auto_mag"])
    ok = np.asarray(t["rubin_nvisits"]) >= MIN_VISITS
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    ax.hist(t["rubin_ext_frac"][ok], bins=40, color="k")
    ax.set(xlabel="fraction of Rubin visits with extendedness = 1", ylabel="objects", yscale="log",
           title=f"{name}: Rubin per-object vote (>= {MIN_VISITS} visits)")
    for ax, col, label, ylim in [(axes[1], "spread_model", "spread_model", (-0.02, 0.05)),
                                 (axes[2], "class_star", "class_star", (-0.05, 1.05))]:
        sc = ax.scatter(m[ok], np.asarray(t[col])[ok], s=2, c=np.asarray(t["rubin_ext_frac"])[ok],
                        cmap="coolwarm", vmin=0, vmax=1)
        ax.set(xlabel="auto_mag (SE++ coadd)", ylabel=label, xlim=(14, 26), ylim=ylim,
               title=f"{label}, colored by Rubin extended fraction")
        fig.colorbar(sc, ax=ax, label="Rubin extended fraction (blue = point)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()

    for rubin_path in sorted(args.samples.glob("testResults_*.fits")):
        tag = rubin_path.stem.removeprefix("testResults_")
        name = f"cutout-circle_{tag}"
        stargal = args.output / f"{name}_stargal.fits"
        if not stargal.exists():
            print(f"{name}: no {stargal.name}, run ./run_stargal.sh first")
            continue
        t = Table.read(stargal)
        rubin = Table.read(rubin_path, hdu=1)
        idx = match_visits(t, rubin, MATCH_RADIUS_ARCSEC)
        ext = np.asarray(rubin["extendedness"], float)
        print(f"{name}: {np.isnan(ext).sum()} of {len(rubin)} Rubin detections have no extendedness")
        hit = (idx >= 0) & np.isfinite(ext)
        nvis = np.bincount(idx[hit], minlength=len(t))
        next_ = np.bincount(idx[hit], weights=ext[hit], minlength=len(t))
        t["rubin_nvisits"] = nvis
        t["rubin_ext_frac"] = np.where(nvis > 0, next_ / np.maximum(nvis, 1), np.nan)
        t.write(args.output / f"{name}_rubin.fits", overwrite=True)
        plot(t, name, args.output / f"{name}_rubin.png")

        m = np.asarray(t["auto_mag"])
        ec = np.asarray(t["extended_class"])
        cs = np.asarray(t["class_star"])
        frac = np.asarray(t["rubin_ext_frac"])
        ok = (nvis >= MIN_VISITS) & (ec >= 0) & np.isfinite(cs)
        r_point, r_ext = ok & (frac < 0.5), ok & (frac >= 0.5)
        print(f"\n{name}: {len(rubin)} Rubin detections, {np.mean(idx >= 0):.0%} matched to {np.sum(nvis > 0)} SE++ objects; "
              f"{ok.sum()} objects with >= {MIN_VISITS} visits ({r_point.sum()} Rubin point, {r_ext.sum()} Rubin extended)")
        print("  mag        | Rubin point: n  SPREAD point  CS>=0.5 | Rubin ext: n  SPREAD ext  CS<0.5 "
              "| agreement SPREAD  CS   | median visits")
        for lo, hi in MAG_BINS:
            b = (m >= lo) & (m < hi)
            p, e, a = r_point & b, r_ext & b, ok & b
            f = lambda sel, x: f"{np.mean(x[sel]):6.1%}" if sel.sum() else "    - "
            agree_s = (ec <= 1) == (frac < 0.5)
            agree_c = (cs >= 0.5) == (frac < 0.5)
            print(f"  {lo:4.1f}-{hi:4.1f}  |  {p.sum():5d}  {f(p, ec <= 1)}      {f(p, cs >= 0.5)} |  {e.sum():5d}   "
                  f"{f(e, ec >= 2)}    {f(e, cs < 0.5)} |  {f(a, agree_s)}  {f(a, agree_c)} | "
                  f"{int(np.median(nvis[a])) if a.sum() else 0}")


if __name__ == "__main__":
    main()

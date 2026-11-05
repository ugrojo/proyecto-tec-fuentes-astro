"""Cutouts (and the SPREAD_MODEL / CLASS_STAR / MU_MAX calls) for the CNN comparison in cnn.py.

  python cnn_dataset.py injected output/inject/r<k> [--cleanup]
      After inject.py and run_sourcextractor.sh (SAMPLES=output/inject/r<k>/images,
      OUTPUT=output/inject/r<k>): measures every SE++ source with
      spread_model.measure() (timed), matches the detections to the injected truth (within MATCH_PIX,
      dropping detections within BLEND_PIX of a real source of the original catalog) and writes
      output/cnn/injected_r<k>.npz. --cleanup deletes the injected images and SE++ maps afterwards.
  python cnn_dataset.py real
      Cutouts of every SE++ source of the original images, with the external labels of
      output/<prefix>_truth.fits (compare_truth.py) and the stargal calls: output/cnn/real.npz.
  python cnn_dataset.py benchmark
      Times SPREAD_MODEL, CLASS_STAR, MU_MAX and the cutouts on the original images with 1, 4 and 16
      worker processes (run it on an idle machine): output/cnn/benchmark_classic.json.

A cutout is STAMP x STAMP pixels around the SE++ centroid, two channels: asinh of the background-
subtracted pixels in units of the local noise (bad pixels 0) and the local PSFEx PSF (unit peak).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from class_star import NeuralNetwork
from images import find_images
from plot_truth import as_float, as_str, external_labels
from spread_model import PSFExModel, match, measure

STAMP = 32
MATCH_PIX, BLEND_PIX = 1.5, 3.0
DETECT_THRESH = 1.5


def stamps(data, sigma, good, psf: PSFExModel, x, y, chunk=8192):
    """(n, 2, STAMP, STAMP) float16 cutouts centered on the FITS (1-based) positions x, y."""
    ny, nx = data.shape
    ix, iy = np.round(np.asarray(x) - 1).astype(int), np.round(np.asarray(y) - 1).astype(int)
    h = STAMP // 2
    oy, ox = np.mgrid[-h:h, -h:h]
    out = np.zeros((len(ix), 2, STAMP, STAMP), np.float16)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(good & (sigma > 0), data / sigma, 0.0)
    for lo in range(0, len(ix), chunk):
        sl = slice(lo, lo + chunk)
        gx, gy = ix[sl, None, None] + ox, iy[sl, None, None] + oy
        inside = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
        out[sl, 0] = np.where(inside, np.arcsinh(snr[np.clip(gy, 0, ny - 1), np.clip(gx, 0, nx - 1)]), 0)
        raster = psf.at(np.asarray(x)[sl], np.asarray(y)[sl])
        p = (STAMP - psf.size) // 2
        out[sl, 1, p:p + psf.size, p:p + psf.size] = raster / raster.max(axis=(1, 2), keepdims=True)
    return out


def maps(im, output: Path):
    """Background-subtracted image, noise sigma and usable-pixel mask of im, from the SE++ check images."""
    weight = im.weight()
    good = np.ones(im.data().shape, bool) if weight is None else weight > 0
    data = im.data() - fits.getdata(output / f"{im.name}_bkg.fits")
    sigma = np.sqrt(np.maximum(fits.getdata(output / f"{im.name}_var.fits"), 0))
    return data, sigma, good


def calls(t):
    return {"auto_mag": as_float(t["auto_mag"]), "saturated": (np.asarray(t["source_flags"]) & 4) > 0,
            "class_star": as_float(t["class_star"]), "spread_model": as_float(t["spread_model"]),
            "spreaderr_model": as_float(t["spreaderr_model"]), "extended_class": np.asarray(t["extended_class"]),
            "mu_class": np.asarray(t["mu_class"]), "mu_locus_offset": as_float(t["mu_locus_offset"])}


def injected(directory: Path, output: Path, only, nnw, jobs, cleanup):
    parts, timing = [], []
    for im in find_images(directory / "images"):
        if not im.name.startswith(only):
            continue
        name = im.name
        t = Table.read(directory / f"{name}_cat.fits")
        psf = PSFExModel(output / "psf" / f"{name}_prepsfex.psf")
        _, seconds = measure(t, im, directory, psf, nnw, DETECT_THRESH, jobs)

        truth = Table.read(directory / f"{name}_truth.fits")
        real = Table.read(output / f"{name}_cat.fits")
        dx, dy = np.asarray(t["pixel_centroid_x"]), np.asarray(t["pixel_centroid_y"])
        idx = match(np.asarray(truth["x"]), np.asarray(truth["y"]), dx, dy, MATCH_PIX)
        blend = match(dx, dy, np.asarray(real["pixel_centroid_x"]), np.asarray(real["pixel_centroid_y"]), BLEND_PIX) >= 0
        keep = (idx >= 0) & ~blend[np.maximum(idx, 0)]
        det = idx[keep]

        start = time.perf_counter()
        data, sigma, good = maps(im, directory)
        cut = stamps(data, sigma, good, psf, dx[det], dy[det])
        seconds["stamps"] = (time.perf_counter() - start) * len(t) / max(len(det), 1)
        timing.append({"image": name, "n_sources": len(t), **seconds})

        part = {k: v[det] for k, v in calls(t).items()}
        part.update(stamps=cut, label=np.asarray(truth["is_star"])[keep].astype(np.int8),
                    true_mag=np.asarray(truth["mag"])[keep], re_arcsec=np.asarray(truth["re_arcsec"])[keep],
                    image=np.full(keep.sum(), name))
        parts.append(part)
        recovered = idx >= 0
        print(f"{name}: {recovered.sum()}/{len(truth)} injected sources detected, {keep.sum()} unblended kept; "
              f"spread_model {seconds['spread_model']:.1f} s for {len(t)} sources")
        if cleanup:
            (directory / "images" / f"{name}.fits").unlink(missing_ok=True)
            for suffix in ("_seg.fits", "_bkg.fits", "_var.fits"):
                (directory / f"{name}{suffix}").unlink(missing_ok=True)

    out = output / "cnn" / f"injected_{directory.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    arrays["timing"] = np.array([tuple(r.values()) for r in timing],
                                dtype=[(k, "U32" if k == "image" else "f8") for k in timing[0]])
    np.savez(out, **arrays)
    print(f"wrote {out}: {len(arrays['label'])} injected sources")


def real(samples: Path, output: Path, only):
    truth = Table.read(output / f"{only}_truth.fits")
    images = as_str(truth["image"])
    parts, seconds, n = [], 0.0, 0
    for im in find_images(samples):
        sel = images == im.name
        if not im.name.startswith(only) or not sel.any():
            continue
        t = Table.read(output / f"{im.name}_stargal.fits")
        lab, gaia = external_labels(truth[sel])
        psf = PSFExModel(output / "psf" / f"{im.name}_prepsfex.psf")
        start = time.perf_counter()
        data, sigma, good = maps(im, output)
        cut = stamps(data, sigma, good, psf, np.asarray(t["pixel_centroid_x"]), np.asarray(t["pixel_centroid_y"]))
        seconds += time.perf_counter() - start
        n += len(t)
        part = calls(t)
        part.update(stamps=cut, label=np.where(lab == "star", 1, np.where(lab == "galaxy", 0, -1)).astype(np.int8),
                    gaia=gaia, source_id=np.asarray(t["source_id"]), image=np.full(len(t), im.name))
        parts.append(part)
        print(f"{im.name}: {len(t)} cutouts")
    out = output / "cnn" / "real.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(out, stamp_seconds=seconds, **arrays)
    print(f"wrote {out}: {n} sources, cutouts {1e3 * seconds / n:.3f} ms per source")


def benchmark(samples: Path, output: Path, only, nnw, jobs_list):
    """Wall-clock time of each classical measurement and of the cutouts on the real images, per job count."""
    results = []
    for jobs in jobs_list:
        total, n = {}, 0
        for im in find_images(samples):
            if not im.name.startswith(only):
                continue
            t = Table.read(output / f"{im.name}_cat.fits")
            psf = PSFExModel(output / "psf" / f"{im.name}_prepsfex.psf")
            _, seconds = measure(t, im, output, psf, nnw, DETECT_THRESH, jobs)
            start = time.perf_counter()
            data, sigma, good = maps(im, output)
            stamps(data, sigma, good, psf, np.asarray(t["pixel_centroid_x"]), np.asarray(t["pixel_centroid_y"]))
            seconds["stamps"] = time.perf_counter() - start
            for k, v in seconds.items():
                total[k] = total.get(k, 0.0) + v
            n += len(t)
        results.append({"jobs": jobs, "n_sources": n, **{k: 1e3 * v / n for k, v in total.items()}})
        print(f"jobs={jobs}: " + ", ".join(f"{k} {1e3 * v / n:.4f} ms" for k, v in total.items())
              + f" per source ({n} sources)")
    out = output / "cnn" / "benchmark_classic.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["injected", "real", "benchmark"])
    parser.add_argument("directory", type=Path, nargs="?", help="output/inject/r<k> (injected only)")
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--only", default="SUPA")
    parser.add_argument("--jobs", type=int, default=4, help="SPREAD_MODEL worker processes")
    parser.add_argument("--nnw", type=Path, default=Path("config/default.nnw"))
    parser.add_argument("--cleanup", action="store_true", help="delete the injected images and SE++ maps")
    args = parser.parse_args()
    if args.kind == "injected":
        injected(args.directory, args.output, args.only, NeuralNetwork(args.nnw), args.jobs, args.cleanup)
    elif args.kind == "real":
        real(args.samples, args.output, args.only)
    else:
        benchmark(args.samples, args.output, args.only, NeuralNetwork(args.nnw), [1, 4, 16])


if __name__ == "__main__":
    main()

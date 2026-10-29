"""SExtractor-style SPREAD_MODEL for SourceXtractor++ catalogs.

Re-implements SExtractor's SPREAD_MODEL (profit_fit() in sextractor/src/profit.c; Desai et al.
2012, Mohr et al. 2012). For each source, phi is the local PSF placed at the best-fit point-source
position, G is phi convolved with a circular exponential of scale length FWHM/16, and t are the
background-subtracted pixels in a circular window around the object:

    SPREAD_MODEL    = (G.t)/(phi.t) - (G.phi)/(phi.phi)
    SPREADERR_MODEL = sqrt(GG*Pt^2 + PP*Gt^2 - 2*GP*Gt*Pt) / Pt^2   (GG, PP, GP weighted by sigma^2)

Point sources give ~0, resolved sources > 0, features sharper than the PSF < 0. The PSF is the
PSFEx model built by config/psf_pipeline.sh, and the classic SExtractor catalog from that pipeline
is matched in as a reference.

Writes output/<name>_stargal.fits (SE++ catalog + spread_model, spreaderr_model and the
matched classic SPREAD_MODEL/CLASS_STAR) and output/<name>_stargal.png (diagnostics).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from class_star import NeuralNetwork, class_star, isophotal_inputs
from images import find_images

MATCH_RADIUS_PIX = 1.0
MAX_WINDOW = 511  # SExtractor subsamples beyond 512 px; such objects are far from point-like anyway
# SExtractor rasterizes the sub-pixel exponential slightly broader than the exact (Fourier) profile;
# scaling FWHM/16 by 1.06 reproduces its SPREAD_MODEL to ~1e-4 on both samples (PSF FWHM 5.5 and 6.2 px).
CLASSIC_SCALE_FACTOR = 1.06


class PSFExModel:
    def __init__(self, path: Path):
        with fits.open(path) as h:
            hdr = h[1].header
            self.basis = h[1].data["PSF_MASK"][0].astype(np.float64)
        if hdr["PSF_SAMP"] != 1.0:
            raise ValueError("expected PSF_SAMPLING 1.0 (PSF pixels = image pixels)")
        self.fwhm = hdr["PSF_FWHM"]
        self.zero = (hdr["POLZERO1"], hdr["POLZERO2"])
        self.scale = (hdr["POLSCAL1"], hdr["POLSCAL2"])
        self.degree = hdr["POLDEG1"]
        self.size = self.basis.shape[-1]

    def at(self, x, y):
        """PSF rasters centered on pixel size//2 for FITS (1-based) positions; x, y scalars or arrays."""
        dx = (np.asarray(x, dtype=np.float64) - self.zero[0]) / self.scale[0]
        dy = (np.asarray(y, dtype=np.float64) - self.zero[1]) / self.scale[1]
        terms = [dx**i * dy**j for j in range(self.degree + 1) for i in range(self.degree + 1 - j)]
        return np.tensordot(np.stack(terms, axis=-1), self.basis, axes=1)


def window_size(cat, psf: PSFExModel):
    """Side of SExtractor's model-fitting window: (object extent + PSF raster size) * 1.2, odd."""
    extent = np.maximum(np.asarray(cat["pixel_boundaries_max_x"]) - np.asarray(cat["pixel_boundaries_min_x"]),
                        np.asarray(cat["pixel_boundaries_max_y"]) - np.asarray(cat["pixel_boundaries_min_y"])) + 1
    n = ((extent + psf.size + 0.499) * 1.2).astype(int) // 2 * 2 + 1
    return np.minimum(n, MAX_WINDOW)


def spread_batch(raster, stamp, sig2, valid, fx, fy, exp_scale, iterations=10, max_shift=2.0, max_step=1.0,
                 tol=1e-3):
    """SPREAD_MODEL for a batch of sources sharing one odd window size n.

    raster: (b, s, s) PSF at each source; stamp, sig2, valid: (b, n, n) background-subtracted pixels,
    variance and window mask; fx, fy: sub-pixel offset of each source from the window center.
    The point-source position fit is a Newton maximization of (phi.t)^2/(phi.phi), with exact
    derivatives of the shifted PSF taken in Fourier space (d/dfx <-> -2*pi*i*kx).
    """
    b, n = len(raster), stamp.shape[-1]
    pad = (n - raster.shape[-1]) // 2
    pk = np.fft.rfft2(np.pad(raster, ((0, 0), (pad, pad), (pad, pad))))
    kx, ky = np.fft.rfftfreq(n), np.fft.fftfreq(n)
    dx, dy = -2j * np.pi * kx[None, :], -2j * np.pi * ky[:, None]
    expo = (1.0 + (2 * np.pi * exp_scale) ** 2 * (ky[:, None] ** 2 + kx[None, :] ** 2)) ** -1.5
    w = valid.astype(np.float64)
    t = stamp * w

    def shifted(idx, px, py, multipliers):
        """Shifted PSF (times Fourier multipliers) on the window, zeroed outside the mask."""
        ramp = np.exp(-2j * np.pi * ky[None, :, None] * py[:, None, None]) \
            * np.exp(-2j * np.pi * kx[None, None, :] * px[:, None, None])
        sk = (pk[idx] * ramp)[:, None]
        return np.fft.irfft2(sk * np.stack(np.broadcast_arrays(*multipliers)), s=(n, n)) * w[idx, None]

    px, py = fx.astype(np.float64), fy.astype(np.float64)
    active = np.ones(b, bool)
    for _ in range(iterations):
        idx = np.flatnonzero(active)
        if not idx.size:
            break
        f = shifted(idx, px[idx], py[idx], [np.ones_like(expo), dx, dy, dx * dx, dy * dy, dx * dy])
        a = np.einsum("bkij,bij->bk", f, t[idx])
        p = np.einsum("bij,bkij->bk", f[:, 0], f)
        xx = np.einsum("bij,bij->b", f[:, 1], f[:, 1])
        yy = np.einsum("bij,bij->b", f[:, 2], f[:, 2])
        xy = np.einsum("bij,bij->b", f[:, 1], f[:, 2])

        A, B = a[:, 0], p[:, 0]
        Ai = np.stack([a[:, 1], a[:, 2]], -1)
        Aij = np.stack([np.stack([a[:, 3], a[:, 5]], -1), np.stack([a[:, 5], a[:, 4]], -1)], -2)
        Bi = 2 * np.stack([p[:, 1], p[:, 2]], -1)
        Bij = 2 * (np.stack([np.stack([xx + p[:, 3], xy + p[:, 5]], -1),
                             np.stack([xy + p[:, 5], yy + p[:, 4]], -1)], -2))
        A_, B_ = A[:, None], B[:, None]
        g = 2 * A_ * Ai / B_ - A_**2 * Bi / B_**2
        A_, B_ = A_[..., None], B_[..., None]
        H = (2 * (Ai[:, :, None] * Ai[:, None, :] + A_ * Aij) / B_
             - 2 * A_ * (Ai[:, :, None] * Bi[:, None, :] + Bi[:, :, None] * Ai[:, None, :]) / B_**2
             - A_**2 * Bij / B_**2 + 2 * A_**2 * Bi[:, :, None] * Bi[:, None, :] / B_**3)

        # Full Newton step where the objective is locally concave, otherwise a per-axis Newton step
        # (still uphill as long as both diagonal curvatures are negative); stop if neither applies.
        hxx, hyy, hxy = H[:, 0, 0], H[:, 1, 1], H[:, 0, 1]
        det = hxx * hyy - hxy**2
        diagonal = (hxx < 0) & (hyy < 0)
        full = diagonal & (det > 0)
        ok = (A > 0) & diagonal
        safe_det = np.where(full, det, 1.0)
        safe_xx, safe_yy = np.where(diagonal, hxx, -1.0), np.where(diagonal, hyy, -1.0)
        sx = np.where(full, -(hyy * g[:, 0] - hxy * g[:, 1]) / safe_det, -g[:, 0] / safe_xx)
        sy = np.where(full, -(hxx * g[:, 1] - hxy * g[:, 0]) / safe_det, -g[:, 1] / safe_yy)

        # Far from the peak the quadratic model overshoots: cap the step (trust region) and keep
        # the position within max_shift of the centroid instead of giving up.
        cap = np.minimum(1.0, max_step / np.maximum(np.hypot(sx, sy), 1e-12))
        ox, oy = px[idx] + sx * cap - fx[idx], py[idx] + sy * cap - fy[idx]
        clip = np.minimum(1.0, max_shift / np.maximum(np.hypot(ox, oy), 1e-12))
        nx_, ny_ = fx[idx] + ox * clip, fy[idx] + oy * clip
        moved = np.hypot(nx_ - px[idx], ny_ - py[idx])
        px[idx[ok]], py[idx[ok]] = nx_[ok], ny_[ok]
        active[idx[~ok | (moved < tol)]] = False

    phi, gm = np.moveaxis(shifted(np.arange(b), px, py, [np.ones_like(expo), expo]), 1, 0)
    pt, gt = np.einsum("bij,bij->b", phi, t), np.einsum("bij,bij->b", gm, t)
    pp, gp = np.einsum("bij,bij->b", phi, phi), np.einsum("bij,bij->b", gm, phi)
    pp2 = np.einsum("bij,bij,bij->b", phi, phi, sig2)
    gg2 = np.einsum("bij,bij,bij->b", gm, gm, sig2)
    gp2 = np.einsum("bij,bij,bij->b", gm, phi, sig2)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = np.where(pt > 0, gt / pt - gp / pp, np.nan)
        err2 = gg2 * pt**2 + pp2 * gt**2 - 2 * gp2 * pt * gt
        spread_err = np.where((pt > 0) & (err2 > 0), np.sqrt(np.maximum(err2, 0)) / pt**2, np.nan)
    return spread, spread_err, px, py


def _spread_batch_star(args):
    return spread_batch(*args)


def spread_model(image, background, variance, cat, psf: PSFExModel, good=None, jobs=None,
                 pixels_per_batch=2_000_000):
    """good: optional boolean map of usable pixels (weight > 0); bad pixels are left out of the fit,
    as SExtractor does for zero-weight pixels."""
    x = np.asarray(cat["pixel_centroid_x"], dtype=np.float64)
    y = np.asarray(cat["pixel_centroid_y"], dtype=np.float64)
    ix, iy = np.round(x - 1).astype(int), np.round(y - 1).astype(int)
    sizes = window_size(cat, psf)
    data = image - background
    ny, nx = image.shape
    exp_scale = CLASSIC_SCALE_FACTOR * psf.fwhm / 16.0

    tasks, order = [], []
    for n in np.unique(sizes):
        members = np.flatnonzero(sizes == n)
        h = n // 2
        yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
        circle = xx**2 + yy**2 < (n / 2) ** 2
        for chunk in np.array_split(members, -(-len(members) * n * n // pixels_per_batch)):
            gx, gy = ix[chunk, None, None] + xx, iy[chunk, None, None] + yy
            valid = circle & (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
            gx, gy = np.clip(gx, 0, nx - 1), np.clip(gy, 0, ny - 1)
            if good is not None:
                valid &= good[gy, gx]
            stamp = np.where(valid, data[gy, gx], 0.0)
            sig2 = np.where(valid, variance[gy, gx], 0.0)
            tasks.append((psf.at(x[chunk], y[chunk]), stamp, sig2, valid,
                          (x[chunk] - 1) - ix[chunk], (y[chunk] - 1) - iy[chunk], exp_scale))
            order.append(chunk)

    if jobs == 1:
        results = map(_spread_batch_star, tasks)
    else:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=jobs)
        results = pool.map(_spread_batch_star, tasks)

    spread = np.full(len(cat), np.nan)
    spread_err = np.full(len(cat), np.nan)
    for chunk, (s, e, _, _) in zip(order, results):
        spread[chunk], spread_err[chunk] = s, e
    if jobs != 1:
        pool.shutdown()
    return spread, spread_err


def extended_class(spread, err):
    """DES Y1 Gold EXTENDED_CLASS (Drlica-Wagner et al. 2018): 0 high-confidence star, 1 likely star,
    2 likely galaxy, 3 high-confidence galaxy, -1 undefined. Thresholds were tuned on DES; re-tune them
    against a truth sample (e.g. HST in COSMOS) before relying on them for this data."""
    spread, err = np.asarray(spread), np.asarray(err)
    cls = ((spread + 3 * err > 0.005).astype(int) + (spread + err > 0.003) + (spread - err > 0.003))
    return np.where(np.isfinite(spread) & np.isfinite(err), cls, -1)


def surface_brightness(value, pixel_scale, zero_point):
    """SExtractor MU_MAX / MU_THRESHOLD: mag/arcsec^2 of a per-pixel flux; 99 where value <= 0."""
    value = np.asarray(value, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = -2.5 * np.log10(value / pixel_scale**2) + zero_point
    return np.where(value > 0, mu, 99.0)


def match(x1, y1, x2, y2, radius):
    """Index into catalog 2 of the nearest source within radius for each source of 1 (-1 = none)."""
    idx = np.full(len(x1), -1)
    order = np.argsort(x2)
    xs = x2[order]
    for i in range(len(x1)):
        lo, hi = np.searchsorted(xs, [x1[i] - radius, x1[i] + radius])
        cand = order[lo:hi]
        if len(cand):
            d2 = (x2[cand] - x1[i]) ** 2 + (y2[cand] - y1[i]) ** 2
            j = np.argmin(d2)
            if d2[j] <= radius**2:
                idx[i] = cand[j]
    return idx


def plot(t: Table, name: str, path: Path):
    m = t["auto_mag"]
    ok = t["sex2_matched"] & np.isfinite(t["spread_model"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    ax = axes[0, 0]
    ax.scatter(m, t["spread_model"], s=1, c="k", alpha=0.4)
    ax.set(xlabel="auto_mag", ylabel="spread_model", xlim=(14, 28), ylim=(-0.02, 0.05),
           title=f"{name}: SPREAD_MODEL (SE++ sources)")
    ax.axhline(0, color="r", lw=0.5)

    ax = axes[0, 1]
    sc = ax.scatter(t["sex2_spread_model"][ok], t["spread_model"][ok], s=2, c=m[ok], cmap="viridis",
                    vmin=18, vmax=27)
    ax.plot([-0.02, 0.05], [-0.02, 0.05], "r-", lw=0.5)
    ax.set(xlabel="classic SExtractor SPREAD_MODEL", ylabel="spread_model (SE++)", xlim=(-0.02, 0.05),
           ylim=(-0.02, 0.05), title="SPREAD_MODEL vs classic SExtractor")
    fig.colorbar(sc, ax=ax, label="auto_mag")

    ax = axes[1, 0]
    ax.scatter(m, t["class_star"], s=1, c="k", alpha=0.4)
    ax.set(xlabel="auto_mag", ylabel="class_star", xlim=(14, 28), ylim=(-0.05, 1.05),
           title=f"{name}: CLASS_STAR (SE++ sources)")

    ax = axes[1, 1]
    okc = t["sex2_matched"] & np.isfinite(t["class_star"])
    sc = ax.scatter(t["sex2_class_star"][okc], t["class_star"][okc], s=2, c=m[okc], cmap="viridis",
                    vmin=18, vmax=27)
    ax.plot([0, 1], [0, 1], "r-", lw=0.5)
    ax.set(xlabel="classic SExtractor CLASS_STAR", ylabel="class_star (SE++)", xlim=(-0.05, 1.05),
           ylim=(-0.05, 1.05), title="CLASS_STAR vs classic SExtractor")
    fig.colorbar(sc, ax=ax, label="auto_mag")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--jobs", type=int, default=None, help="worker processes (default: all CPUs)")
    parser.add_argument("--detect-thresh", type=float, default=1.5,
                        help="SE++ detection threshold in sigma (CLASS_STAR isophotes start there)")
    parser.add_argument("--only", default="", help="only process images whose name starts with this")
    parser.add_argument("--nnw", type=Path, default=Path("config/default.nnw"), help="CLASS_STAR network weights")
    args = parser.parse_args()
    nnw = NeuralNetwork(args.nnw)

    for im in find_images(args.samples):
        name = im.name
        cat_path = args.output / f"{name}_cat.fits"
        if not name.startswith(args.only) or not cat_path.exists():
            continue
        psf_path = args.output / "psf" / f"{name}_prepsfex.psf"
        if not psf_path.exists():
            print(f"{name}: no PSF model, run config/psf_pipeline.sh first")
            continue

        t = Table.read(cat_path)
        image = im.data()
        pixel_scale = im.pixel_scale
        weight = im.weight()
        good = None if weight is None else weight > 0
        bkg = fits.getdata(args.output / f"{name}_bkg.fits")
        var = fits.getdata(args.output / f"{name}_var.fits")
        psf = PSFExModel(psf_path)

        t["spread_model"], t["spreaderr_model"] = spread_model(image, bkg, var, t, psf, good, jobs=args.jobs)
        # SExtractor's analysis threshold uses the global background noise, not the local map.
        thresh = args.detect_thresh * float(np.sqrt(np.median(var if good is None else var[good])))
        seg = fits.getdata(args.output / f"{name}_seg.fits")
        data = image - bkg if good is None else np.where(good, image - bkg, -np.inf)
        peak, iso = isophotal_inputs(data, seg, np.asarray(t["detection_id"]), thresh)
        t["class_star"] = class_star(peak, iso, thresh, psf.fwhm * pixel_scale, pixel_scale, nnw)
        t["mu_max"] = surface_brightness(peak, pixel_scale, im.zero_point)
        t["mu_threshold"] = surface_brightness(np.full(len(t), thresh), pixel_scale, im.zero_point)

        ref = Table.read(args.output / f"{name}_sex2_cat.fits")
        idx = match(np.asarray(t["pixel_centroid_x"]), np.asarray(t["pixel_centroid_y"]),
                    np.asarray(ref["X_IMAGE"]), np.asarray(ref["Y_IMAGE"]), MATCH_RADIUS_PIX)
        t["sex2_matched"] = idx >= 0
        for src, dst in [("SPREAD_MODEL", "sex2_spread_model"), ("SPREADERR_MODEL", "sex2_spreaderr_model"),
                         ("CLASS_STAR", "sex2_class_star"), ("MAG_AUTO", "sex2_mag_auto"),
                         ("MU_MAX", "sex2_mu_max"), ("MU_THRESHOLD", "sex2_mu_threshold")]:
            t[dst] = np.where(idx >= 0, np.asarray(ref[src])[idx], np.nan)
        t["extended_class"] = extended_class(t["spread_model"], t["spreaderr_model"])
        t["sex2_extended_class"] = extended_class(t["sex2_spread_model"], t["sex2_spreaderr_model"])

        t.write(args.output / f"{name}_stargal.fits", overwrite=True)
        plot(t, name, args.output / f"{name}_stargal.png")

        ok = t["sex2_matched"] & np.isfinite(t["spread_model"])
        diff = t["spread_model"][ok] - t["sex2_spread_model"][ok]
        both = ok & (t["extended_class"] >= 0) & (t["sex2_extended_class"] >= 0)
        agree = np.mean((t["extended_class"][both] >= 2) == (t["sex2_extended_class"][both] >= 2))
        print(f"{name}: PSF FWHM {psf.fwhm:.2f} px, {ok.sum()}/{len(t)} matched to classic; "
              f"spread diff median {np.median(diff):+.5f}, MAD {np.median(np.abs(diff - np.median(diff))):.5f}; "
              f"star/galaxy agreement {agree:.1%}")
        okc = t["sex2_matched"] & np.isfinite(t["class_star"])
        dcs = np.abs(t["class_star"][okc] - t["sex2_class_star"][okc])
        print(f"{name}: class_star vs classic CLASS_STAR |diff| median {np.median(dcs):.3f}, "
              f"within 0.1 for {np.mean(dcs < 0.1):.1%}; seeing {psf.fwhm * pixel_scale:.3f}\", "
              f"threshold {thresh:.3f} (image units)")
        okm = t["sex2_matched"] & (t["mu_max"] < 99) & (t["sex2_mu_max"] < 99)
        dmu = t["mu_max"][okm] - t["sex2_mu_max"][okm]
        print(f"{name}: mu_max vs classic MU_MAX diff median {np.median(dmu):+.4f}, |diff| < 0.05 for "
              f"{np.mean(np.abs(dmu) < 0.05):.1%}; mu_threshold {t['mu_threshold'][0]:.3f} "
              f"(classic {np.nanmedian(t['sex2_mu_threshold']):.3f}) mag/arcsec^2")


if __name__ == "__main__":
    main()

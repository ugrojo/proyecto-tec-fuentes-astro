"""Inject synthetic stars and galaxies with known labels into the Subaru CCDs (training set for cnn.py).

Stars are the local PSFEx PSF (config/psf_pipeline.sh) shifted to a random sub-pixel position. Galaxies are
Sersic profiles (n, half-light radius, axis ratio, angle drawn from SIZE_* / SERSIC_* below), rendered
with sub-pixel oversampling and convolved with the same PSF. Each injected source gets the photon noise
of its own flux (gain from the header); the image already holds the real background, its noise, real
neighbours and artifacts. Sources avoid the image borders, masked and saturated pixels (bleed trails),
real detections and each other.

Writes output/inject/r<k>/images/<name>.fits (+ <name>.weight.fits, a link to the original weight map)
for run_sourcextractor.sh (SAMPLES=output/inject/r<k>/images OUTPUT=output/inject/r<k>), and
output/inject/r<k>/<name>_truth.fits with the injected x, y (FITS pixels), mag, is_star and galaxy
shape of every source.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from images import find_images
from spread_model import PSFExModel

MAG_RANGE = (19.0, 25.0)  # brighter than ~18.7 the Suprime-Cam frames saturate, which injection can't model
# Field galaxies get smaller as they get fainter: median half-light radius in arcsec at magnitude m, with a
# log-normal scatter (roughly HST size-magnitude relations, shifted to V).
SIZE_MEDIAN_ARCSEC = lambda m: 0.5 * 10 ** (-0.1 * (m - 21.0))
SIZE_SCATTER_DEX = 0.25
SERSIC_DISK, SERSIC_BULGE, BULGE_FRACTION = (0.6, 1.5), (1.5, 4.0), 0.4
EDGE, MIN_SEPARATION, REAL_CLEARANCE, SATURATION_CLEARANCE = 30, 30, 8, 15  # pixels
OVERSAMPLE = 5


def sersic_b(n):
    """b_n with half the light inside r_e (Ciotti & Bertin 1999 expansion)."""
    return 2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n**2)


def sersic_stamp(n, re_pix, q, theta, dx, dy, size):
    """Unit-flux Sersic image on a size x size pixel grid (size odd), centered (dx, dy) from the central pixel."""
    s = (np.arange(size * OVERSAMPLE) + 0.5) / OVERSAMPLE - size / 2
    x, y = np.meshgrid(s - dx, s - dy)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    r = np.sqrt(q * xr**2 + yr**2 / q)
    img = np.exp(-sersic_b(n) * ((r / re_pix) ** (1 / n) - 1))
    img = img.reshape(size, OVERSAMPLE, size, OVERSAMPLE).sum(axis=(1, 3))
    return img / img.sum()


def convolve(img, psf):
    """Full linear convolution (both odd-sized and centered), via FFT."""
    n = img.shape[0] + psf.shape[0] - 1
    out = np.fft.irfft2(np.fft.rfft2(img, s=(n, n)) * np.fft.rfft2(psf, s=(n, n)), s=(n, n))
    return out / out.sum()


def shifted_psf(raster, dx, dy, pad=4):
    """PSF raster shifted by a sub-pixel (dx, dy) in Fourier space, unit flux."""
    r = np.pad(raster, pad)
    k = np.fft.fftfreq(r.shape[0])
    phase = np.exp(-2j * np.pi * (k[:, None] * dy + k[None, :] * dx))
    out = np.fft.ifft2(np.fft.fft2(r) * phase).real
    return out / out.sum()


def place(image, stamp, ix, iy):
    """Add an odd-sized stamp to image with its central pixel on (ix, iy), clipped at the borders."""
    h = stamp.shape[0] // 2
    ny, nx = image.shape
    y0, y1, x0, x1 = iy - h, iy + h + 1, ix - h, ix + h + 1
    sy0, sx0 = max(0, -y0), max(0, -x0)
    sy1, sx1 = stamp.shape[0] - max(0, y1 - ny), stamp.shape[1] - max(0, x1 - nx)
    image[max(y0, 0):min(y1, ny), max(x0, 0):min(x1, nx)] += stamp[sy0:sy1, sx0:sx1]


def dilate(mask, r):
    """Binary dilation with a (2r+1)^2 box, by shifted maxima."""
    out = mask.copy()
    for axis in (0, 1):
        grown = out.copy()
        for s in range(1, r + 1):
            grown |= np.roll(out, s, axis) | np.roll(out, -s, axis)
        out = grown
    return out


def positions(rng, n, good, saturated, real_x, real_y):
    """n random 0-based positions on usable pixels, away from borders, saturated pixels (bleed trails),
    real sources and each other."""
    ny, nx = good.shape
    blocked = ~good | dilate(saturated, SATURATION_CLEARANCE)
    r = REAL_CLEARANCE
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    disk = ox**2 + oy**2 <= r * r
    rx, ry = np.round(real_x).astype(int), np.round(real_y).astype(int)
    for dy, dx in zip(oy[disk], ox[disk]):
        blocked[np.clip(ry + dy, 0, ny - 1), np.clip(rx + dx, 0, nx - 1)] = True

    cell = MIN_SEPARATION
    taken = {}
    out = []
    while len(out) < n:
        cx = rng.uniform(EDGE, nx - EDGE, 4 * n)
        cy = rng.uniform(EDGE, ny - EDGE, 4 * n)
        for x, y in zip(cx, cy):
            if blocked[int(round(y)), int(round(x))]:
                continue
            gx, gy = int(x // cell), int(y // cell)
            near = [p for i in (-1, 0, 1) for j in (-1, 0, 1) for p in taken.get((gx + i, gy + j), [])]
            if any((x - px) ** 2 + (y - py) ** 2 < MIN_SEPARATION**2 for px, py in near):
                continue
            taken.setdefault((gx, gy), []).append((x, y))
            out.append((x, y))
            if len(out) == n:
                break
    return np.array(out)


def inject(im, psf, real_cat, rng, n_sources, out_dir: Path):
    image = im.data()
    weight = im.weight()
    good = weight > 0 if weight is not None else np.ones(image.shape, bool)
    saturated = image >= im.saturation if im.saturation > 0 else np.zeros(image.shape, bool)
    xy = positions(rng, n_sources, good, saturated, np.asarray(real_cat["pixel_centroid_x"]) - 1,
                   np.asarray(real_cat["pixel_centroid_y"]) - 1)
    mag = rng.uniform(*MAG_RANGE, n_sources)
    is_star = rng.random(n_sources) < 0.5
    bulge = rng.random(n_sources) < BULGE_FRACTION
    sersic_n = np.where(bulge, rng.uniform(*SERSIC_BULGE, n_sources), rng.uniform(*SERSIC_DISK, n_sources))
    re_arcsec = np.clip(SIZE_MEDIAN_ARCSEC(mag) * 10 ** rng.normal(0, SIZE_SCATTER_DEX, n_sources), 0.05, 3.0)
    q = rng.uniform(0.2, 1.0, n_sources)
    theta = rng.uniform(0, np.pi, n_sources)
    flux = 10 ** (-0.4 * (mag - im.zero_point))

    model = np.zeros_like(image)
    for k, (x, y) in enumerate(xy):
        ix, iy = int(round(x)), int(round(y))
        dx, dy = x - ix, y - iy
        raster = psf.at(x + 1, y + 1)
        if is_star[k]:
            stamp = shifted_psf(raster, dx, dy)
        else:
            re_pix = re_arcsec[k] / im.pixel_scale
            size = int(np.clip(2 * np.ceil(8 * re_pix) + 1, 9, 81))
            stamp = convolve(sersic_stamp(sersic_n[k], re_pix, q[k], theta[k], dx, dy, size), raster)
        place(model, flux[k] * stamp, ix, iy)

    noise = rng.normal(0, 1, image.shape) * np.sqrt(np.maximum(model, 0) / im.gain) if im.gain > 0 else 0
    image += model + noise

    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(image.astype(np.float32), im.header())]).writeto(
        image_dir / f"{im.name}.fits", overwrite=True)
    link = image_dir / f"{im.name}.weight.fits"
    if im.weight_path is not None and not link.is_symlink():
        os.symlink(os.path.relpath(im.weight_path, image_dir), link)
    Table({"x": xy[:, 0] + 1, "y": xy[:, 1] + 1, "mag": mag, "is_star": is_star, "sersic_n": sersic_n,
           "re_arcsec": np.where(is_star, 0.0, re_arcsec), "q": q, "theta": theta}).write(
        out_dir / f"{im.name}_truth.fits", overwrite=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--realization", type=int, default=0, help="seed and output/inject/r<k> directory")
    parser.add_argument("--per-image", type=int, default=1200, help="sources injected per CCD")
    parser.add_argument("--only", default="SUPA", help="images whose name starts with this")
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()

    out_dir = args.output / "inject" / f"r{args.realization}"
    for i, im in enumerate(find_images(args.samples)):
        if not im.name.startswith(args.only):
            continue
        psf = PSFExModel(args.output / "psf" / f"{im.name}_prepsfex.psf")
        real = Table.read(args.output / f"{im.name}_cat.fits")
        rng = np.random.default_rng(1000 * args.realization + i)
        inject(im, psf, real, rng, args.per_image, out_dir)
        print(f"{im.name}: injected {args.per_image} sources into {out_dir / 'images' / (im.name + '.fits')}")


if __name__ == "__main__":
    main()

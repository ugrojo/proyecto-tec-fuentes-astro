"""The images in samples/ that the pipeline processes, with their per-image calibration.

Two kinds of inputs:
  - <name>.fits: one image in HDU 1 (Rubin coadd cutouts: nJy pixels, so the AB zero point is 31.4).
  - <base>.fits.fz: multi-extension mosaic exposure (Subaru Suprime-Cam), one image per CCD in HDUs
    1..N, with an optional <base>.weight.fits.fz holding the matching weight maps. Each CCD is
    processed as its own image named <base>_ccd<DET-ID>, extracted to output/inputs/<name>.fits
    (+ <name>.weight.fits) with a header wcslib accepts: the astrometry is a TPV polynomial written
    as RA---TAN + PV terms, and CUNIT 'degree' is not a valid unit.

Zero point, gain and saturation come from the headers. `python images.py` prints one line per image
for the shell scripts: name, path, hdu, weight path ("-" if none), weight hdu, zero point, gain,
saturation level (0 = none) and pixel scale (arcsec).
"""
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

NJY_ZERO_POINT = 31.4
# Suprime-Cam frames are flat-fielded, so the saturation plateau moves with CCD and amplifier
# (~48-66 kADU on SUPA0155974p); flag everything above the lowest plateau.
ADU_SATURATION = {"SuprimeCam": 45000.0}


@dataclass
class Image:
    name: str
    path: Path
    hdu: int
    weight_path: Path | None
    weight_hdu: int
    zero_point: float
    gain: float
    saturation: float
    pixel_scale: float

    def data(self):
        return fits.getdata(self.path, self.hdu).astype(np.float64)

    def header(self):
        return fits.getheader(self.path, self.hdu)

    def weight(self):
        return None if self.weight_path is None else fits.getdata(self.weight_path, self.weight_hdu)


def _calibration(header):
    if header.get("BUNIT", "").strip() == "nJy":
        return NJY_ZERO_POINT, 0.0, 0.0
    zero_point = float(header["PHOTZP"])
    gain = float(header.get("GAIN", 0.0))
    saturation = float(header.get("SATURATE", ADU_SATURATION.get(header.get("INSTRUME", "").strip(), 0.0)))
    return zero_point, gain, saturation


def _pixel_scale(header):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return float(np.mean(proj_plane_pixel_scales(WCS(header).celestial)) * 3600)


def _standard_header(header):
    header = header.copy()
    if any(k.startswith("PV1_") for k in header) and header.get("CTYPE1", "").endswith("-TAN"):
        header["CTYPE1"], header["CTYPE2"] = "RA---TPV", "DEC--TPV"
    for k in ("CUNIT1", "CUNIT2"):
        if header.get(k, "").strip() == "degree":
            header[k] = "deg"
    if "RADECSYS" in header:
        header["RADESYS"] = header.pop("RADECSYS")
    return header


def _extract(hdul, hdu, weight_hdul, path, weight_path):
    """Write one CCD (and its weight map) as a plain image in HDU 1, like the Rubin cutouts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _standard_header(hdul[hdu].header)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(hdul[hdu].data, header)]).writeto(path, overwrite=True)
    if weight_hdul is not None:
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(weight_hdul[hdu].data.astype(np.float32), header)]
                     ).writeto(weight_path, overwrite=True)


def find_images(samples: Path = Path("samples"), inputs: Path = Path("output/inputs")):
    images = []
    for path in sorted(samples.glob("*.fits")):
        if path.name.startswith("testResults_"):
            continue
        header = fits.getheader(path, 1)
        images.append(Image(path.stem, path, 1, None, 0, *_calibration(header), _pixel_scale(header)))
    for path in sorted(samples.glob("*.fits.fz")):
        if path.name.endswith(".weight.fits.fz"):
            continue
        base = path.name.removesuffix(".fits.fz")
        weight = path.with_name(f"{base}.weight.fits.fz")
        with fits.open(path) as hdul, (fits.open(weight) if weight.exists() else _nothing()) as whdul:
            for hdu in range(1, len(hdul)):
                header = _standard_header(hdul[hdu].header)
                name = f"{base}_ccd{header.get('DET-ID', hdu - 1)}"
                image_path, weight_path = inputs / f"{name}.fits", inputs / f"{name}.weight.fits"
                if not image_path.exists() or image_path.stat().st_mtime < path.stat().st_mtime:
                    _extract(hdul, hdu, whdul, image_path, weight_path)
                images.append(Image(name, image_path, 1, weight_path if whdul is not None else None, 1,
                                    *_calibration(header), _pixel_scale(header)))
    return images


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def find_image(name, samples: Path = Path("samples")):
    for image in find_images(samples):
        if image.name == name:
            return image
    raise KeyError(name)


if __name__ == "__main__":
    import sys

    for im in find_images(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples")):
        print(im.name, im.path, im.hdu, im.weight_path or "-", im.weight_hdu,
              f"{im.zero_point:.4f}", f"{im.gain:.3f}", f"{im.saturation:g}", f"{im.pixel_scale:.4f}")

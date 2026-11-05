"""SExtractor's MU_MAX / MU_THRESHOLD for SourceXtractor++ catalogs, and a star/galaxy class from them.

MU_MAX is the peak surface brightness of a source above the background and MU_THRESHOLD that of the
analysis threshold, both in mag/arcsec^2 (SExtractor's analyse.c). A point source has the PSF's
peak-to-total flux ratio, so MU_MAX - MAG_AUTO is constant for stars (the stellar locus) and larger for
extended sources, whose light is spread out (MU_MAX vs MAG_AUTO separation, e.g. Leauthaud et al. 2007).
"""
import numpy as np


def surface_brightness(value, pixel_scale, zero_point):
    """SExtractor MU_MAX / MU_THRESHOLD: mag/arcsec^2 of a per-pixel flux; 99 where value <= 0."""
    value = np.asarray(value, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = -2.5 * np.log10(value / pixel_scale**2) + zero_point
    return np.where(value > 0, mu, 99.0)


def mu_max_class(mag, mu_max, mu_threshold, detect_thresh, saturated, k=3.0, locus_mags=(17.5, 21.0)):
    """Point source (0) / extended (1) / undefined (-1) from MU_MAX vs MAG_AUTO.

    The locus is the median of mu_max - mag around its mode among bright unsaturated sources (stars pile
    up there; galaxies spread out above it), its intrinsic scatter (pixel
    phase) the robust width around it, and the noise of the peak pixel comes from mu_threshold
    (= detect_thresh * sigma_bkg). Point source: offset < k * sqrt(scatter^2 + noise^2). Saturated
    sources are point sources when their mu_max is on the saturation plateau (capped peak), undefined
    otherwise (bleed-trail pieces). Returns (offset from the locus in mag, class, locus, scatter).
    """
    mag, mu_max = np.asarray(mag, float), np.asarray(mu_max, float)
    saturated = np.asarray(saturated, bool)
    d = mu_max - mag
    usable = np.isfinite(d) & (mu_max < 99)
    bright = usable & ~saturated & (mag > locus_mags[0]) & (mag < locus_mags[1])
    hist, edges = np.histogram(d[bright], bins=np.arange(-4, 2, 0.02))
    mode = edges[np.argmax(hist)] + 0.01
    locus = np.median(d[bright & (np.abs(d - mode) < 0.15)])
    near = bright & (np.abs(d - locus) < 0.3)
    scatter = 1.4826 * np.median(np.abs(d[near] - np.median(d[near])))
    offset = d - locus
    noise = 2.5 / np.log(10) / detect_thresh * 10 ** (-0.4 * (np.asarray(mu_threshold, float) - mu_max))
    cls = np.where(offset < k * np.hypot(scatter, noise), 0, 1)
    if (saturated & usable).any():
        plateau = np.median(mu_max[saturated & usable])
        cls = np.where(saturated, np.where(mu_max < plateau + 0.5, 0, -1), cls)
    return offset, np.where(usable, cls, -1), locus, scatter

"""SExtractor's CLASS_STAR (stellarity index) for SourceXtractor++ catalogs.

Re-implements SExtractor's neural-network star/galaxy classifier (analyse.c and neurro.c in
astromatic/sextractor) on SourceXtractor++ footprints. For each source the network sees:

  - 8 isophotal areas: pixels of the footprint above thresholds spaced geometrically from the
    analysis threshold to the peak, in units of the seeing disk (FWHM/pixel scale)^2, as log10
  - the peak, relative to the global analysis threshold, as log10
  - the seeing FWHM in arcsec, as log10

and returns ~1 for point sources and ~0 for extended ones, using SExtractor's default.nnw weights.
Those weights were trained for 1.5 < FWHM < 5 pixels; Rubin coadds are slightly above that.
"""
from pathlib import Path

import numpy as np

NISO = 8


class NeuralNetwork:
    """SExtractor .nnw network: sigmoid layers with input scaling and output rescaling."""

    def __init__(self, path: Path):
        rows = [line.split() for line in Path(path).read_text().splitlines()[1:]
                if line.strip() and not line.lstrip().startswith("#")]
        values = iter([float(v) for row in rows for v in row])
        nlayers = int(next(values))
        sizes = [int(next(values)) for _ in range(nlayers)]
        self.inbias = np.array([next(values) for _ in range(sizes[0])])
        self.inscale = np.array([next(values) for _ in range(sizes[0])])
        self.weights, self.biases = [], []
        for nin, nout in zip(sizes[:-1], sizes[1:]):
            w = np.empty((nin, nout))
            b = np.empty(nout)
            for j in range(nout):
                w[:, j] = [next(values) for _ in range(nin)]
                b[j] = next(values)
            self.weights.append(w)
            self.biases.append(b)
        self.outbias = np.array([next(values) for _ in range(sizes[-1])])
        self.outscale = np.array([next(values) for _ in range(sizes[-1])])

    def __call__(self, inputs):
        """inputs: (n, n_in) -> (n, n_out)."""
        layer = inputs * self.inscale + self.inbias
        for w, b in zip(self.weights, self.biases):
            layer = 1.0 / (1.0 + np.exp(-(layer @ w + b)))
        return (layer - self.outbias) / self.outscale


def isophotal_inputs(data, segmentation, labels, thresh):
    """Peak and the NISO isophotal areas of each footprint (segmentation == label).

    data is the background-subtracted, unfiltered image; thresh the analysis threshold in data
    units. Thresholds are thresh * (peak/thresh)^(i/NISO), SExtractor's CCD-mode spacing.
    """
    seg = segmentation.ravel()
    inside = np.flatnonzero(seg > 0)
    order = np.argsort(seg[inside], kind="stable")
    pix_labels = seg[inside][order]
    pix_values = data.ravel()[inside][order]
    starts = np.searchsorted(pix_labels, labels, side="left")
    ends = np.searchsorted(pix_labels, labels, side="right")

    peak = np.full(len(labels), np.nan)
    iso = np.zeros((len(labels), NISO))
    steps = np.arange(NISO) / NISO
    for k, (lo, hi) in enumerate(zip(starts, ends)):
        if hi == lo:
            continue
        values = pix_values[lo:hi]
        peak[k] = values.max()
        if peak[k] > 0:
            levels = thresh * (peak[k] / thresh) ** steps
            iso[k] = (values[:, None] > levels[None, :]).sum(axis=0)
    return peak, iso


def class_star(peak, iso, thresh, seeing_fwhm_arcsec, pixel_scale, nnw: NeuralNetwork):
    """CLASS_STAR from isophotal_inputs(); NaN where the footprint has no pixels in the map."""
    fac2 = (seeing_fwhm_arcsec / pixel_scale) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        area = np.where(iso > 0, np.log10(np.maximum(iso, 1) / fac2), np.log10(0.01))
        peak_in = np.where(peak > 0, np.log10(np.maximum(peak, 1e-30) / thresh), np.log10(0.1))
    inputs = np.column_stack([area[:, 0], peak_in, area[:, 1:], np.full(len(peak), np.log10(seeing_fwhm_arcsec))])
    out = nnw(inputs)[:, 0]
    return np.where(np.isfinite(peak), out, np.nan)

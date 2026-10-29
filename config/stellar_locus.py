"""Print PSFEx's SAMPLE_FWHMRANGE bracketing the stellar locus of a prepsfex catalog.

PSFEx auto-selects stars around the most common 2*FLUX_RADIUS of all candidates; in galaxy-rich
fields (e.g. cluster fields) that mode can land on compact galaxies and broaden the PSF. The
stellar locus is the narrow peak of 2*FLUX_RADIUS among bright, round, unflagged sources.
"""
import sys

import numpy as np
from astropy.table import Table

t = Table.read(sys.argv[1], hdu=2)
ok = (t["FLAGS"] == 0) & (t["SNR_WIN"] > 50) & (t["ELONGATION"] < 1.3) & (t["FLUX_RADIUS"] > 0)
size = 2 * np.asarray(t["FLUX_RADIUS"])[ok]
counts, edges = np.histogram(size, bins=np.arange(1.0, 20.0, 0.1))
smooth = np.convolve(counts, [1, 2, 3, 2, 1], mode="same")
mode = edges[np.argmax(smooth)] + 0.05
print(f"{0.85 * mode:.2f},{1.15 * mode:.2f}")

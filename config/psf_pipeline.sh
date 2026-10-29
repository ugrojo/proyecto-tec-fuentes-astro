#!/usr/bin/env bash
# Runs inside the conda image. For one image listed by images.py:
#   1. decompress it into a plain FITS file (classic SExtractor wants a simple image)
#   2. SExtractor (bright, VIGNET) -> PSFEx -> <name>.psf
#   3. SExtractor with that PSF -> reference catalog with SPREAD_MODEL and CLASS_STAR
# Zero point, gain, saturation level, weight map and pixel scale come from images.py.
set -euo pipefail

name=$1
work=output/psf
share=/opt/conda/share
mkdir -p "$work"

read -r _ path hdu weight weight_hdu zp gain satur scale < <(python images.py | awk -v n="$name" '$1 == n')
[[ -n ${path:-} ]] || { echo "$name: not listed by images.py" >&2; exit 1; }

python - "$path" "$hdu" "$work/$name.fits" "$weight" "$weight_hdu" "$work/$name.weight.fits" <<'EOF'
import sys
from astropy.io import fits
path, hdu, out, weight, weight_hdu, weight_out = sys.argv[1:]
with fits.open(path) as h:
    fits.PrimaryHDU(h[int(hdu)].data, h[int(hdu)].header).writeto(out, overwrite=True)
if weight != "-":
    with fits.open(weight) as h:
        fits.PrimaryHDU(h[int(weight_hdu)].data, h[int(weight_hdu)].header).writeto(weight_out, overwrite=True)
EOF

# Rubin coadds (nJy) have no meaningful gain or saturation level: gain 0 = infinite, 1e9 = never saturated.
[[ $satur == 0 ]] && satur=1e9
common=(-c "$share/sextractor/default.sex" -FILTER_NAME "$share/sextractor/default.conv"
        -MAG_ZEROPOINT "$zp" -GAIN "$gain" -SATUR_LEVEL "$satur" -PIXEL_SCALE 0 -VERBOSE_TYPE QUIET)
if [[ $weight != - ]]; then
  common+=(-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE "$work/$name.weight.fits")
fi

source-extractor "$work/$name.fits" "${common[@]}" \
  -PARAMETERS_NAME config/prepsfex.param -CATALOG_TYPE FITS_LDAC \
  -CATALOG_NAME "$work/${name}_prepsfex.cat" \
  -DETECT_THRESH 5 -ANALYSIS_THRESH 5 -DETECT_MINAREA 5 -PHOT_APERTURES 15

# PSF_SAMPLING 1: one PSF pixel per image pixel; the PSF (FWHM ~4 px) is well sampled.
# PSF stars come from a size range around the stellar locus, so compact galaxies stay out of the model.
fwhm_range=$(python config/stellar_locus.py "$work/${name}_prepsfex.cat")
psfex "$work/${name}_prepsfex.cat" -c "$share/psfex/default.psfex" \
  -PSF_DIR "$work" -PSF_SAMPLING 1.0 -PSF_SIZE 25,25 -PSFVAR_DEGREES 2 -SAMPLE_FWHMRANGE "$fwhm_range" \
  -CHECKPLOT_TYPE NONE -CHECKIMAGE_TYPE NONE \
  -WRITE_XML Y -XML_NAME "$work/${name}_psfex.xml" -NTHREADS 4 -VERBOSE_TYPE QUIET

fwhm_arcsec=$(python - "$work/${name}_prepsfex.psf" "$scale" <<'EOF'
import sys
from astropy.io import fits
h = fits.getheader(sys.argv[1], 1)
print(f"{h['PSF_FWHM'] * float(sys.argv[2]):.3f}")  # PSF_FWHM is in image pixels
EOF
)
echo "$name: PSF FWHM = ${fwhm_arcsec}\""

source-extractor "$work/$name.fits" "${common[@]}" \
  -PARAMETERS_NAME config/reference.param -CATALOG_TYPE FITS_1.0 \
  -CATALOG_NAME "output/${name}_sex2_cat.fits" \
  -PSF_NAME "$work/${name}_prepsfex.psf" \
  -SEEING_FWHM "$fwhm_arcsec" -STARNNW_NAME "$share/sextractor/default.nnw"

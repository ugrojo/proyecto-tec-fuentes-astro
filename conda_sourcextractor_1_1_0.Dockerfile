# astrorama only publishes linux-64 builds, so arm64 hosts run this under emulation
# numpy<2.4: the resolved astropy 6.1 still uses np.in1d, removed in numpy 2.4
# classic SExtractor + PSFEx build the PSF model and the reference SPREAD_MODEL/CLASS_STAR
FROM --platform=linux/amd64 mambaorg/micromamba:2.0
RUN micromamba install -y -n base -c astrorama -c conda-forge sourcextractor=1.1.0 "numpy<2.4" matplotlib-base \
        astromatic-source-extractor=2.28 astromatic-psfex=3.24 \
    && micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN sourcextractor++ --help > /dev/null && source-extractor -v && psfex -v

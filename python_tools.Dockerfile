# Native (multi-arch) image for the pure-Python steps (spread_model.py, plot_detections.py),
# so they don't run under amd64 emulation like the SourceXtractor++ image.
FROM python:3.12-slim
RUN pip install --no-cache-dir "numpy<2.4" astropy matplotlib

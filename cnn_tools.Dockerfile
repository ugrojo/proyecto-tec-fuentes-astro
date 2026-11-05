# CPU-only PyTorch for cnn.py (training and the speed/accuracy comparison against SPREAD_MODEL & co.).
# No GPU: this host has no NVIDIA container toolkit, and CPU is the like-for-like comparison anyway.
FROM python:3.12-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "numpy<2.4" matplotlib

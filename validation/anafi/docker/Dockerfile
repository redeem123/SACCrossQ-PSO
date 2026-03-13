FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    libssl-dev libffi-dev \
    net-tools iproute2 \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Parrot Olympe SDK (Linux only — run inside this container on macOS)
RUN pip3 install --upgrade pip
RUN pip3 install \
    parrot-olympe \
    numpy \
    scipy \
    matplotlib \
    pandas \
    requests

WORKDIR /workspace

# Default: open shell for interactive use
CMD ["/bin/bash"]

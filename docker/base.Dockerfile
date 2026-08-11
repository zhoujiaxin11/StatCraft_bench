# Base image: general Python + pandas/numpy scientific stack
# All domain images inherit from this.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System packages needed for scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pinned scientific stack (version lock — critical for reproducibility).
# NOTE: openai is REQUIRED at runtime by framework/agent_adapter.py::ModelClient,
# otherwise every containerized model run aborts with RuntimeError(
# "openai package required"). Host-mode runs worked only because the dev box had
# it installed globally — don't let that mask the packaging gap.
RUN pip install --upgrade pip==24.2 && pip install \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scipy==1.13.1 \
        scikit-learn==1.5.1 \
        statsmodels==0.14.2 \
        matplotlib==3.9.2 \
        seaborn==0.13.2 \
        openpyxl==3.1.5 \
        pyarrow==17.0.0 \
        pytest==8.3.2 \
        pytest-timeout==2.3.1 \
        pyyaml==6.0.2 \
        openai==1.55.0

WORKDIR /workspace

# Run as an unprivileged user (defense-in-depth; agent code should never need root).
# The bind-mounted /workspace is shared from the host — on Docker Desktop the
# sharing layer makes it writable regardless of in-container UID.
RUN useradd --create-home --uid 1000 runner \
    && chown -R runner:runner /workspace
USER runner


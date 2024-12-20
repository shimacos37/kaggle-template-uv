
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/root/.cache/uv

FROM base AS builder

WORKDIR /app

# Install Python
RUN --mount=from=ghcr.io/astral-sh/uv:0.5.5,source=/uv,target=/bin/uv \
    --mount=type=bind,source=./uv.lock,target=uv.lock \
    --mount=type=bind,source=./pyproject.toml,target=pyproject.toml \
    --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync

FROM base

WORKDIR /app
# Install dependencies and OpenSearch
RUN --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt update && apt-get install --yes --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    g++ \
    git \
    gnupg2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libmecab-dev \
    libbz2-dev \
    libdb-dev \
    libreadline-dev \
    libffi-dev \
    libgdbm-dev \
    libgomp1 \
    libopencv-dev \
    liblapack-dev \
    liblzma-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    libssl-dev \
    lsb-release \
    uuid-dev \
    make \
    mecab \
    mecab-ipadic-utf8 \
    systemctl \
    wget \
    xz-utils \
    zlib1g-dev \
    gpg \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/share/opensearch/bin/opensearch-plugin install --batch analysis-icu

COPY --from=builder /app/.venv /app/.venv
# uvによってinstallされたpythonのパスを通すために必要
COPY --from=builder /root/.local /root/.local
ENV PATH="/app/.venv/bin:$PATH"

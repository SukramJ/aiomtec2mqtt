# Multi-stage container build for aiomtec2mqtt.
#
# Stage 1 (builder) compiles the package into a wheel against a slim Python
# base image. Stage 2 installs only the resulting wheel into a fresh slim
# image so the runtime layer carries no build toolchain. The container runs
# as a non-root user and reads its config from /config (mount this in).

# ---------- Stage 1: build wheel ----------
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY aiomtec2mqtt/ ./aiomtec2mqtt/

RUN python -m pip install --upgrade pip build \
    && python -m build --wheel --outdir /wheels

# ---------- Stage 2: minimal runtime ----------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    AIOMTEC_CONFIG_DIR=/config \
    XDG_CONFIG_HOME=/config

# Create a dedicated non-root user. UID/GID 1000 matches the default on
# common single-board setups (Raspberry Pi OS, Debian) so bind-mounted
# config files keep their ownership without manual chown.
RUN groupadd --system --gid 1000 mtec \
    && useradd --system --uid 1000 --gid 1000 \
       --home-dir /app --shell /usr/sbin/nologin mtec \
    && mkdir -p /app /config \
    && chown -R mtec:mtec /app /config

COPY --from=builder /wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm -f /tmp/*.whl

USER mtec
WORKDIR /app

# /config is the canonical mount point for config.yaml. The path is also
# exported as XDG_CONFIG_HOME so platformdirs picks it up automatically.
VOLUME ["/config"]

# No network ports are exposed — the process is an outbound MQTT publisher
# and Modbus client.

ENTRYPOINT ["aiomtec2mqtt"]

# Image labels (OCI standard) populated at build-time by the publish workflow.
ARG VERSION=dev
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="aiomtec2mqtt" \
      org.opencontainers.image.description="Read M-TEC Energybutler Modbus data, publish to MQTT" \
      org.opencontainers.image.source="https://github.com/sukramj/aiomtec2mqtt" \
      org.opencontainers.image.licenses="LGPL-3.0-or-later" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

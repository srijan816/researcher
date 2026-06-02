# syntax=docker/dockerfile:1

# Websurfx's upstream Dockerfile builds a static musl binary. On Oracle ARM64
# that currently fails while linking LuaJIT. This app2 image builds Websurfx
# natively against Debian/glibc instead, keeping the service private in Docker.

FROM rust:1.94-bookworm AS builder

ARG WEBSURFX_GIT_REF=rolling

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    git \
    libssl-dev \
    patch \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${WEBSURFX_GIT_REF}" https://github.com/neon-mmd/websurfx.git .
COPY deploy/websurfx/patches/ /tmp/websurfx-patches/
RUN for patch_file in /tmp/websurfx-patches/*.patch; do patch -p1 < "$patch_file"; done
RUN cargo build --release --no-default-features --features redis-cache

FROM debian:bookworm-slim AS runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/public /opt/websurfx/public
COPY --from=builder /src/target/release/websurfx /usr/local/bin/websurfx

VOLUME ["/etc/xdg/websurfx/"]
EXPOSE 8080

CMD ["websurfx"]

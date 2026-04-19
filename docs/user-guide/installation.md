# Installation

## Requirements

- Python **3.13+**
- Linux, macOS, or Windows
- Optional: `systemd` (Linux) for running as a service

## Install with `uv` (recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Install with `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Optional extras

| Extra  | Installs | Purpose                        |
| ------ | -------- | ------------------------------ |
| `fast` | `orjson` | accelerated JSON serialization |

```bash
uv pip install -e ".[fast]"
```

## First run

```bash
aiomtec2mqtt
```

On first launch a configuration wizard creates `config.yaml` under
`~/.config/aiomtec2mqtt/` (Linux) or `%APPDATA%\aiomtec2mqtt\` (Windows).

## Run as systemd service

```bash
sudo bin/install_systemd_service.sh
sudo systemctl enable --now aiomtec2mqtt
journalctl -u aiomtec2mqtt -f
```

## Run as a container (GHCR)

A multi-arch image (`linux/amd64`, `linux/arm64`) is published to GitHub
Container Registry on every release.

```bash
docker pull ghcr.io/sukramj/aiomtec2mqtt:latest
```

Available tags:

| Tag      | What you get                          |
| -------- | ------------------------------------- |
| `latest` | Most recent stable release            |
| `1`      | Latest patch on major version 1       |
| `1.0`    | Latest patch on minor version 1.0     |
| `1.0.6`  | Pinned exact version                  |
| `edge`   | Tip of `main` (no stability promises) |

The image expects your `config.yaml` at `/config/aiomtec2mqtt/config.yaml`
inside the container. Bind-mount the host directory you keep it in:

```yaml
# docker-compose.yml
services:
  aiomtec2mqtt:
    image: ghcr.io/sukramj/aiomtec2mqtt:latest
    container_name: aiomtec2mqtt
    restart: unless-stopped
    volumes:
      - ./config:/config/aiomtec2mqtt:ro
    # The container runs as UID 1000 — chown the host directory accordingly.
```

Bring it up:

```bash
docker compose up -d
docker compose logs -f
```

The container exposes no ports — it makes outbound connections to the
inverter (Modbus TCP) and the MQTT broker only.

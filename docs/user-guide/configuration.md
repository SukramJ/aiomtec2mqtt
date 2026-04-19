# Configuration

`aiomtec2mqtt` is configured via `config.yaml` (loaded from the current
directory or `~/.config/aiomtec2mqtt/`) and/or environment variables prefixed
with `MTEC_`. All fields are validated at startup via Pydantic
(`aiomtec2mqtt.config_schema.ConfigSchema`).

## Example `config.yaml`

```yaml
DEBUG: False

MODBUS_IP: "espressif.fritz.box"
MODBUS_PORT: 502 # 5743 for firmware < V27.52.4.0
MODBUS_SLAVE: 1
MODBUS_TIMEOUT: 5
MODBUS_FRAMER: "rtu" # rtu | socket | ascii | tls
MODBUS_RETRIES: 3

MQTT_SERVER: "localhost"
MQTT_PORT: 1883
MQTT_LOGIN: ""
MQTT_PASSWORD: ""
MQTT_TOPIC: "MTEC"
MQTT_FLOAT_FORMAT: "{:.3f}"

HASS_ENABLE: True
HASS_BASE_TOPIC: "homeassistant"
HASS_BIRTH_GRACETIME: 15

REFRESH_NOW: 10
REFRESH_CONFIG: 30
REFRESH_DAY: 300
REFRESH_STATIC: 3600
REFRESH_TOTAL: 300
```

## Environment-variable overrides

Every configuration key can be overridden with `MTEC_<KEY>`:

```bash
export MTEC_MODBUS_IP=192.168.1.20
export MTEC_MQTT_PASSWORD=my-secret
export MTEC_REFRESH_NOW=5
aiomtec2mqtt
```

Env values are parsed as `bool`/`int`/`float` when possible, otherwise kept as
string. This is especially useful for secrets (keep them out of YAML) and for
container deployments.

## Field reference

| Field                  | Type | Default         | Constraint                   |
| ---------------------- | ---- | --------------- | ---------------------------- |
| `MODBUS_IP`            | str  | — (required)    | non-empty                    |
| `MODBUS_PORT`          | int  | — (required)    | 1–65535                      |
| `MODBUS_SLAVE`         | int  | — (required)    | 0–247                        |
| `MODBUS_TIMEOUT`       | int  | — (required)    | 1–600 s                      |
| `MODBUS_FRAMER`        | str  | `rtu`           | `rtu`/`socket`/`ascii`/`tls` |
| `MODBUS_RETRIES`       | int  | `3`             | 0–20                         |
| `MQTT_SERVER`          | str  | — (required)    | non-empty                    |
| `MQTT_PORT`            | int  | — (required)    | 1–65535                      |
| `MQTT_LOGIN`           | str  | `""`            | —                            |
| `MQTT_PASSWORD`        | str  | `""`            | —                            |
| `MQTT_TOPIC`           | str  | — (required)    | non-empty                    |
| `MQTT_FLOAT_FORMAT`    | str  | `.3f`           | valid Python format spec     |
| `HASS_ENABLE`          | bool | `False`         | —                            |
| `HASS_BASE_TOPIC`      | str  | `homeassistant` | —                            |
| `HASS_BIRTH_GRACETIME` | int  | `15`            | 0–600 s                      |
| `REFRESH_NOW`          | int  | `10`            | 1–3600 s                     |
| `REFRESH_CONFIG`       | int  | `30`            | 1–3600 s                     |
| `REFRESH_DAY`          | int  | `300`           | 1–86400 s                    |
| `REFRESH_STATIC`       | int  | `3600`          | 1–86400 s                    |
| `REFRESH_TOTAL`        | int  | `300`           | 1–86400 s                    |
| `DEBUG`                | bool | `False`         | —                            |

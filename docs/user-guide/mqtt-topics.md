# MQTT Topic Structure

All topics share the base prefix `<MQTT_TOPIC>/<serial_number>/`, where
`<MQTT_TOPIC>` is the value configured in `config.yaml` (default: `MTEC`) and
`<serial_number>` is read from the inverter at startup.

| Topic suffix   | Payload                         | Refreshed        |
| -------------- | ------------------------------- | ---------------- |
| `config`       | Static inverter config          | `REFRESH_CONFIG` |
| `now-base`     | Current power/SOC/status        | `REFRESH_NOW`    |
| `now-grid`     | Grid voltage/current/frequency  | `REFRESH_NOW`    |
| `now-inverter` | Inverter temp + per-phase power | `REFRESH_NOW`    |
| `now-backup`   | Backup power data               | `REFRESH_NOW`    |
| `now-battery`  | Battery temp/cell voltages      | `REFRESH_NOW`    |
| `now-pv`       | PV voltage/current/power        | `REFRESH_NOW`    |
| `day`          | Daily energy totals             | `REFRESH_DAY`    |
| `total`        | Lifetime energy totals          | `REFRESH_TOTAL`  |

See the full [register reference](../reference/registers.md) for every field
published under these topics.

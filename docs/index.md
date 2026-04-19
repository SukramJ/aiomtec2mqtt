# aiomtec2mqtt

**aiomtec2mqtt** is an async Python bridge between M-TEC Energybutler hybrid
inverters (Modbus RTU/TCP) and MQTT brokers. It reads 80+ parameters from the
inverter, publishes them with configurable refresh intervals, and integrates
seamlessly with Home Assistant, evcc, and other energy-management tools.

## Highlights

- **Async-first**: built on `asyncio`, `pymodbus`, and `aiomqtt` — no blocking I/O.
- **Home Assistant auto-discovery** out of the box.
- **Runs anywhere**: Raspberry Pi, NAS, Linux server, container.
- **Offline-capable**: no cloud connection required.
- **Resilient**: retry, circuit-breaker, and health checks included.
- **Typed & tested**: PEP 561, strict mypy, high coverage.

## Quick links

- [Installation](user-guide/installation.md)
- [Configuration reference](user-guide/configuration.md)
- [MQTT topic structure](user-guide/mqtt-topics.md)
- [Register reference](reference/registers.md)
- [Developer setup](developer-guide/setup.md)

## Compatibility

| Device                                             | Status              |
| -------------------------------------------------- | ------------------- |
| M-TEC Energybutler GEN3 (8kW-3P-3G25 and siblings) | supported           |
| Wattsonic / Sunways / Daxtromn (similar firmware)  | probably compatible |

| Firmware        | Required port       |
| --------------- | ------------------- |
| `< V27.52.4.0`  | `MODBUS_PORT: 5743` |
| `>= V27.52.4.0` | `MODBUS_PORT: 502`  |

## License

Licensed under the [LGPL](https://github.com/SukramJ/aiomtec2mqtt/blob/main/LICENSE).

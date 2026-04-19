# Troubleshooting

## Cannot connect to the inverter

- Check `MODBUS_IP` — it should resolve to the **espressif** device on your
  LAN, not the inverter's display unit.
- Firmware **V27.52.4.0** and later require `MODBUS_PORT: 502`. Earlier firmware
  uses `5743`.
- Try a manual connection: `nc -zv <ip> 502` or `nc -zv <ip> 5743`.

## `ConfigValidationError` at startup

The Pydantic schema rejects invalid or missing values. Read the first few
lines of the error — it lists every offending field. Typical cases:

- `MODBUS_PORT` outside `1..65535`
- `MODBUS_FRAMER` not one of `rtu`, `socket`, `ascii`, `tls`
- `MQTT_FLOAT_FORMAT` not a valid Python format spec (try `.3f` or `{:.3f}`)

## Home Assistant doesn't see the devices

- Make sure `HASS_ENABLE: True` and `HASS_BASE_TOPIC` matches your HA config
  (default `homeassistant`).
- Check the MQTT broker is reachable from both HA and `aiomtec2mqtt`.
- Restart HA after the first `aiomtec2mqtt` run so discovery re-synchronises.

## Enable debug logs

```yaml
DEBUG: True
```

or via env var:

```bash
MTEC_DEBUG=true aiomtec2mqtt
```

This switches the root logger to `DEBUG` and logs every Modbus read/write and
every MQTT publish.

## Ask for help

- GitHub issues: <https://github.com/SukramJ/aiomtec2mqtt/issues>
- GitHub discussions: <https://github.com/SukramJ/aiomtec2mqtt/discussions>
- German forum thread:
  <https://www.photovoltaikforum.com/thread/206243->

# evcc Integration

`aiomtec2mqtt` publishes all data needed by [evcc](https://evcc.io) (smart
EV-charging controller) over MQTT.

## Template

A ready-to-paste meter template is in
[`templates/evcc.yaml`](https://github.com/SukramJ/aiomtec2mqtt/tree/main/templates).
Adjust the broker address, topic, and `serial_number` to match your
deployment. Drop the snippet into your `evcc.yaml` under `meters:`.

## Which values are used

evcc typically reads these fields from `now-base`:

- `grid_power` — import (+) / export (−) in W
- `pv` — PV generation in W
- `battery` — battery charge (+) / discharge (−) in W
- `battery_soc` — battery state of charge in %

All values are published with the format string from `MQTT_FLOAT_FORMAT`
(default `.3f`). Make sure evcc's `scale` settings match.

## Tips

- Prefer `REFRESH_NOW: 10` (the default) — evcc polls its meters every 10 s.
- If you see stale meter values in evcc, check the MQTT broker logs; clients
  behind NAT or on flaky WiFi may lose the connection silently.

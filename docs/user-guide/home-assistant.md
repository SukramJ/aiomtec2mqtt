# Home Assistant Integration

`aiomtec2mqtt` supports Home Assistant's **MQTT Discovery** protocol, so all
registered entities appear automatically once the MQTT bridge is running.

## Prerequisites

1. A running MQTT broker reachable from both Home Assistant and `aiomtec2mqtt`
   (Mosquitto is recommended).
2. Home Assistant's [MQTT integration](https://www.home-assistant.io/integrations/mqtt/)
   configured with discovery enabled.

## Enable discovery

Set these two values in `config.yaml`:

```yaml
HASS_ENABLE: True
HASS_BASE_TOPIC: "homeassistant"
```

On startup, `aiomtec2mqtt` publishes discovery messages under
`<HASS_BASE_TOPIC>/sensor/<unique_id>/config`. Home Assistant reads them and
creates the corresponding sensors, numbers, selects, and switches.

## Dashboards

Pre-made Lovelace dashboards are bundled in
[`templates/`](https://github.com/SukramJ/aiomtec2mqtt/tree/main/templates). Copy
them into your Home Assistant config and adjust the `<serial_number>` and
`<MQTT_TOPIC>` placeholders to match your deployment.

## Birth & will messages

HA's birth/will messages (`<HASS_BASE_TOPIC>/status online` / `offline`) are
honoured. When Home Assistant reports `online`, `aiomtec2mqtt` waits
`HASS_BIRTH_GRACETIME` seconds before republishing the discovery payloads to
ensure HA has finished its own boot.

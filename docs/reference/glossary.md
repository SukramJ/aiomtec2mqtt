# Glossary

Compact definitions for the terminology that shows up around `aiomtec2mqtt`
and its ecosystem. Ordering is alphabetical inside each section.

## M-TEC and photovoltaics

**Backup power**
: The inverter's emergency power path: when the grid fails, selected loads
keep being supplied from PV plus battery. Has its own Modbus register
group `now-backup`.

**Energybutler**
: Product line of M-TEC GmbH for hybrid inverters (PV + battery + grid,
optionally with backup power). The code primarily targets GEN3 devices
(e.g. `8kW-3P-3G25`); other vendors with the same firmware base
(Wattsonic, Sunways, Daxtromn) are potentially compatible.

**Espressif adapter**
: Communication module (ESP32-based) inside the inverter that translates
Modbus RTU internally to TCP and listens on the LAN as a Modbus TCP
server. Usually reachable at `espressif.fritz.box` or a DHCP-assigned
IP. The firmware version determines the TCP port (see **Modbus port**).

**Hybrid inverter**
: An inverter that simultaneously couples PV strings, a battery storage
system, and the public grid. Enables self-consumption, feed-in, battery
charging, and backup power from a single device.

**SOC** (State of Charge)
: Battery state of charge in percent (0–100). Held in the `battery_soc`
register and exposed under the MQTT topic `now-base/battery_soc`.

## Modbus

**Address / register address**
: 16-bit word address used by Modbus operations to reference individual
words (16 bit) or word ranges. With M-TEC, config registers live in
the 10000–52505 range, real-time registers from 10100, day stats from 31000.

**Coil / discrete input**
: Single-bit Modbus data types. `aiomtec2mqtt` does not use them — every
M-TEC value lives in **holding registers**.

**Framer**
: pymodbus' encoding layer: `rtu` (binary RTU frame inside a TCP
tunnel), `socket` (plain Modbus TCP), `ascii` (rare). Default in this
bridge: `rtu`. External adapters may need a different framer —
configurable via `MODBUS_FRAMER`.

**Holding register**
: Read/write 16-bit word. Read-only values on M-TEC are nevertheless
read as holding registers (no separate discrete-input system).

**Modbus framer tunnelling**
: M-TEC sends RTU frames (with CRC) over a TCP socket. Hence
`framer=rtu` over a TCP connection — _not_ standard Modbus TCP.

**Modbus port**
: TCP port of the Espressif adapter. Up to firmware **V27.52.4.0** it
was `5743`, after that `502`. The firmware version must be checked
before initial setup.

**Slave ID** (unit ID)
: Modbus addressing of the end device. M-TEC defaults to `255` (0xFF)
or `1` — depending on the adapter setup. Configurable via
`MODBUS_SLAVE`.

**Word vs. register**
: One Modbus register = 1 word = 2 bytes = 16 bits. Larger values
(`uint32`, `float32`) consume 2 registers. Word-order endianness is
M-TEC-specific and handled in `register_processors.py`.

## MQTT

**Auto-discovery**
: Home Assistant mechanism by which devices publish their sensor
definitions as retained messages on
`<discovery_prefix>/<component>/<id>/config`. The HA broker then
creates entities without manual YAML configuration. Code path:
`hass_int.py`.

**Birth message**
: Retained message Home Assistant publishes at startup to
`homeassistant/status` with payload `online`. The bridge waits
`HASS_BIRTH_GRACETIME` seconds (default 15 s) before re-publishing
its discovery configs — prevents race conditions during HA boot.

**LWT** (Last Will and Testament)
: MQTT feature that lets a client register a topic + payload at connect
time which the broker publishes automatically when the connection
dies unexpectedly. Used for `<MQTT_TOPIC>/availability`.

**QoS** (Quality of Service)
: MQTT delivery guarantee level: `0` (fire-and-forget), `1`
(at-least-once), `2` (exactly-once). The bridge mostly uses `0` for
telemetry and `1` for discovery + availability.

**Retained message**
: Message cached by the broker that _every new subscriber_ on the topic
receives immediately at subscribe time. Discovery configs and
availability are retained, telemetry topics are not.

**Topic hierarchy**
: Slash-separated paths. Bridge layout:
`MTEC/<serial>/<group>` — e.g. `MTEC/EBG3-12345/now-base`. Group is
one of `config`, `now-base`, `now-grid`, `now-inverter`,
`now-backup`, `now-battery`, `now-pv`, `day`, `total`.

## Bridge-internal terms

**Calculated register**
: Pseudo-register whose value is not read directly from Modbus but
computed from a formula over other registers. Defined in
`registers.yaml`, evaluated in `formula_evaluator.py`.

**Circuit breaker**
: Resilience pattern (see ADR-002): after `failure_threshold`
consecutive failures the breaker opens and blocks further calls for
`recovery_timeout` seconds. Prevents hammering of dead peers.

**Coordinator**
: `AsyncMtecCoordinator` — orchestrates Modbus polling, MQTT
publishing, health tracking. One instance per inverter. See ADR-003.

**Health stream**
: Logical data channel with its own refresh interval. Examples:
`now`, `day`, `total`, `config`, `static`. Each stream has a
`HealthCheck` entry with `last_success` + `stale_threshold`.

**Refresh interval**
: Polling period per stream in seconds. Configurable via `REFRESH_NOW`,
`REFRESH_DAY`, `REFRESH_TOTAL`, `REFRESH_CONFIG`, `REFRESH_STATIC`.
Defaults in `const.REFRESH_DEFAULTS`.

**Register cluster**
: Optimization in the Modbus client: consecutive register addresses are
collapsed into _one_ read call (max ~125 words per read). Reduces
Modbus traffic and latency.

**Register group**
: Logical grouping of registers that are read together and written to
the same MQTT topic. Defined in `registers.yaml` under `groups:`.

**Stale status**
: `HealthCheck` state when the most recent successful read is older
than `stale_threshold` seconds. Exported via Prometheus and
aggregated into `aiomtec_health`.

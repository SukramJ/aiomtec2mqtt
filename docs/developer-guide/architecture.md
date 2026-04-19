# Architecture

## Data flow

```
M-TEC Energybutler (Modbus RTU/TCP)
          │
          ▼
   AsyncModbusClient  ──►  RegisterProcessorRegistry
          │
          ▼
  AsyncMtecCoordinator  ──►  HealthCheck
          │
          ▼
    AsyncMqttClient  ──►  Home Assistant / evcc / EMS
```

Everything is `async`/`await` on top of `asyncio.TaskGroup`. Each concern
(Modbus read, MQTT publish, health check, backup polling) runs as a separate
task; the coordinator owns the task group and coordinates shutdown.

## Key modules

| Module                           | Responsibility                                     |
| -------------------------------- | -------------------------------------------------- |
| `async_coordinator.py`           | Entry point, task orchestration, shutdown handling |
| `async_modbus_client.py`         | Modbus RTU/TCP client with retry + circuit breaker |
| `async_mqtt_client.py`           | MQTT client using `aiomqtt`                        |
| `hass_int.py`                    | Home Assistant discovery payload builder           |
| `register_models.py`             | Pydantic models for register validation            |
| `register_processors.py`         | Value transformation pipeline                      |
| `config.py` / `config_schema.py` | YAML + env loading with Pydantic validation        |
| `resilience.py`                  | Backoff + circuit-breaker primitives               |
| `health.py`                      | Liveness/health tracking                           |
| `shutdown.py`                    | Signal handling, graceful teardown                 |
| `_json.py`                       | `orjson`-accelerated JSON helpers                  |

## Resilience

- **Retry**: `resilience.BackoffConfig` (exponential with jitter).
- **Circuit breaker**: `resilience.CircuitBreakerConfig`, opens after consecutive
  failures, half-opens on a timer.
- **Health**: `health.HealthCheck` tracks the last-successful timestamp per
  stream and reports `stale` after `stale_threshold` seconds.

## Configuration

All config flows through `config.init_config()` → `config_schema.validate_config()`.
The result is a plain dict keyed by `Config` enum members so consumers can
continue to write `cfg[Config.MODBUS_IP]`. Env overrides (`MTEC_*`) are applied
before validation.

## Sub-package layout

Since 2026-04-17 the public API is grouped into cohesive sub-packages. The
existing flat modules (`async_coordinator.py`, `async_modbus_client.py`, …)
remain in place for backwards compatibility; the sub-packages re-export the
same objects so both import paths work.

| Sub-package                 | Contents                                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `aiomtec2mqtt.client`       | `AsyncModbusClient`, `AsyncMqttClient`                                                                                         |
| `aiomtec2mqtt.model`        | `RegisterDefinition`, `CalculatedRegister`, `RegisterMap`, `ConfigSchema`, `CoordinatorConfig*`, processors, formula evaluator |
| `aiomtec2mqtt.coordinator`  | `AsyncMtecCoordinator`, `CoordinatorConfigBuilder`                                                                             |
| `aiomtec2mqtt.integrations` | `HassIntegration`, `PrometheusMetrics`                                                                                         |
| `aiomtec2mqtt.support`      | exceptions, resilience, health, structured logging, shutdown, DI container                                                     |
| `aiomtec2mqtt.interfaces`   | `Protocol` classes defining seams between components                                                                           |

Prefer the sub-package paths in new code:

```python
from aiomtec2mqtt.client import AsyncModbusClient
from aiomtec2mqtt.model import CoordinatorConfigBuilder
from aiomtec2mqtt.support import CircuitBreaker, HealthCheck
from aiomtec2mqtt.interfaces import ModbusClientProtocol
```

## Register YAML

`registers.yaml` defines the contract between the inverter and the rest of
the system. Invariants (unique MQTT topics per group, known HA device classes,
etc.) are enforced by `tests/test_registers_contract.py`.

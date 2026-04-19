# aiomtec2mqtt-test-support

Reusable testing utilities for projects that integrate with
[`aiomtec2mqtt`](https://github.com/sukramj/aiomtec2mqtt). Provides pytest
fixtures, fakes for the Modbus and MQTT clients, replay helpers, and a small
collection of assertion helpers — everything kept stable across `aiomtec2mqtt`
patch releases so your test suite does not have to track internal moves.

## Install

```bash
pip install aiomtec2mqtt-test-support
```

The package depends on `aiomtec2mqtt>=1.0.6,<2`. Use the same major as the
version of `aiomtec2mqtt` your code targets.

## Quick start

The package registers itself as a pytest plugin (entry point `pytest11`), so
any fixture defined in `aiomtec2mqtt_test_support.fixtures` is available
without an explicit import.

```python
import pytest


@pytest.mark.asyncio
async def test_my_integration(mock_modbus, fake_mqtt_client):
    mock_modbus.register_values[10100] = [4660, 22136]

    # ... drive your code under test, then assert ...

    assert any(
        topic.endswith("/now-base") for topic, *_ in fake_mqtt_client.published_messages
    )
```

## Public surface

```python
from aiomtec2mqtt_test_support.fakes import (
    FakeConfigProvider,
    FakeHealthMonitor,
    FakeModbusClient,
    FakeMqttClient,
)
from aiomtec2mqtt_test_support.transports import (
    Frame,
    MockModbusResponse,
    MockModbusTransport,
    SessionMetadata,
    SessionPlayer,
    SessionRecorder,
)
from aiomtec2mqtt_test_support.assertions import (
    assert_mqtt_topic_seen,
    assert_register_published,
)
```

## Fixtures

| Name                 | What it gives you                                       |
| -------------------- | ------------------------------------------------------- |
| `mock_modbus`        | A fresh `MockModbusTransport` per test                  |
| `fake_modbus_client` | A `FakeModbusClient` with no preloaded data             |
| `fake_mqtt_client`   | A `FakeMqttClient` that records every publish           |
| `fake_config`        | An empty `FakeConfigProvider` you can populate per test |
| `fake_health`        | A `FakeHealthMonitor` with no registered components     |
| `replay_session`     | Factory: `replay_session(frames=...) -> SessionPlayer`  |

## Versioning

`aiomtec2mqtt-test-support` ships with the same version as the matching
`aiomtec2mqtt` release.

## License

LGPL — see `LICENSE`.

# Testing

## Running tests

```bash
pytest                       # parallel (xdist) with coverage gate (fail_under = 85)
pytest tests/test_config.py  # focused run
pytest -m "not slow"         # skip slow markers
pytest -m integration        # only integration tests
```

Coverage configuration lives in `pyproject.toml`
(`[tool.coverage.run]`, `[tool.coverage.report]`).

## Layout

```
tests/
├── conftest.py                    # shared fixtures + patch.stopall teardown
├── test_async_coordinator.py
├── test_async_modbus_client.py
├── test_async_mqtt_client.py
├── test_config.py
├── test_config_schema.py          # Pydantic validation
├── test_hass_int.py
├── test_registers_contract.py     # registers.yaml invariants
└── ...
```

## Markers

| Marker        | Meaning                                       |
| ------------- | --------------------------------------------- |
| `slow`        | Takes > 1 s; excluded from fast pre-push runs |
| `benchmark`   | Timing-sensitive, uses `pytest-benchmark`     |
| `integration` | Requires MQTT broker or Modbus mock           |

Define new markers under `[tool.pytest.ini_options].markers`.

## Writing good tests

- Prefer **fakes** over mocks for framework boundaries. Existing fakes:
  `FakePahoMessage`, `FakeConfigProvider` in `aiomtec2mqtt.testing`.
- Use `freezegun` for time-sensitive behaviour — real sleeps are banned in
  the test suite.
- If you call `patch(...).start()` manually, rely on the autouse
  `_stop_all_patches` fixture in `tests/conftest.py` to clean up.
- **Bug fix = test first**: write a failing test, then fix the bug.

## Contract tests

`tests/test_registers_contract.py` enforces invariants between
`registers.yaml` and the consuming code (groups, MQTT topics, HA classes).
Update `_KNOWN_GROUPS` / `_KNOWN_HASS_DEVICE_CLASSES` in that file when adding
new categories.

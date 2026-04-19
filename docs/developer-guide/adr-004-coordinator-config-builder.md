# ADR-004: Builder pattern for `CoordinatorConfig`

- **Status**: Accepted
- **Date**: 2026-04-17
- **Context**: ROADMAP 1.6 — analogous to `CentralConfigBuilder` from
  `aiohomematic`.

## Context

`AsyncMtecCoordinator` has historically been configured via a
`dict[str, Any]` produced from `config.yaml` plus `MTEC_*` environment
overrides. That path is fine for YAML-driven deployments but awkward for:

- Programmatic setups (tests, notebooks, third-party integrations)
- Partial overrides ("take my YAML, but override only `MQTT_PORT` to 1884")
- IDE autocomplete and type checking
- Self-documenting calls in tests
  (`with_modbus(ip=..., port=...)` instead of
  `cfg[Config.MODBUS_IP] = ...`)

## Decision

`aiomtec2mqtt/coordinator_config.py` introduces two classes:

- **`CoordinatorConfig`** (frozen, slotted dataclass): immutable,
  schema-validated config container.
- **`CoordinatorConfigBuilder`** (fluent): incremental construction via
  `with_modbus`, `with_mqtt`, `with_refresh`, `with_home_assistant`,
  `with_debug`. `build()` validates through `validate_config` and
  returns a `CoordinatorConfig`.

The coordinator stays **dict-based** (`config[Config.MODBUS_IP]`) until a
separate refactor migrates it. `CoordinatorConfig.as_dict()` produces the
legacy format with `Config`-enum keys.

## Rationale

- **Frozen dataclass with `slots=True`**: guarantees immutability +
  memory efficiency, prevents accidental mutations in long-running
  processes.
- **`Self` returns in the builder**: yields natural method chaining.
- **`from_dict` / `from_env` / `from_config`**: three entry points for
  the three real-world use cases — YAML bootstrap, test setups, partial
  clones.
- **Schema validation in `build()`**: one path, one failure source.
  Pydantic owns validation; the builder is a schema-agnostic
  constructor.
- **Legacy adapter via `as_dict`**: enables incremental migration
  without a big-bang coordinator refactor.

## Constraints

- **No** business logic in the builder — only validation and dict
  building.
- Builder state is **mutable internally**, but the produced object is
  immutable. That is intentional: the same builder can be cloned and
  varied further.
- New config fields: first add them to `ConfigSchema` (Pydantic), then
  to `CoordinatorConfig` (dataclass field), then to the matching
  `with_*` method bundle. That keeps the three layers in sync.

## Alternatives rejected

- **A Pydantic model directly as the config**: would obviate the
  builder, but Pydantic models are sub-optimal for method chaining
  (every mutation needs cloning or `model_copy()`). The explicit
  builder is more ergonomic.
- **`attrs` instead of `dataclass`**: extra dependency without payoff
  for our use case.
- **Migrate the coordinator straight to the new object**: higher
  refactor risk because `cfg[Config.MODBUS_IP]` accesses are scattered
  across the codebase. Incremental is safer.

## Consequences

- Tests can compose config cleanly instead of maintaining YAML
  fixtures.
- Third parties can import `CoordinatorConfigBuilder` as a stable
  public API — from `aiomtec2mqtt.coordinator` or
  `aiomtec2mqtt.model`.
- Coordinator refactor (step 2): make `AsyncMtecCoordinator.__init__`
  accept a `CoordinatorConfig` directly instead of a `dict`. That is a
  separate PR, not part of this ADR.

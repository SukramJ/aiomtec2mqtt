# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-20

### Added - Asynchronous Architecture

- **Async Coordinator** (`async_coordinator.py`): Complete async rewrite of the main coordinator
  - Non-blocking I/O operations using Python's `asyncio`
  - Improved responsiveness on low-power devices (Raspberry Pi, NAS)
  - Parallel register group reading for better performance
  - Graceful shutdown handling with proper cleanup
- **Async Modbus Client** (`async_modbus_client.py`): Asynchronous Modbus communication
  - Non-blocking register reads and writes
  - Connection pooling and management
  - Automatic reconnection with exponential backoff
  - Circuit breaker pattern for fault tolerance
- **Async MQTT Client** (`async_mqtt_client.py`): Asynchronous MQTT publishing
  - Non-blocking message publishing
  - Automatic connection management
  - Parallel publishing of multiple topics
  - QoS support with retry logic

### Added - Resilience & Reliability

- **Resilience Patterns** (`resilience.py`): Production-grade error handling
  - Exponential backoff with configurable jitter
  - Circuit breaker implementation (closed, open, half-open states)
  - Connection state machine for robust state transitions
  - Configurable retry policies and thresholds
- **Health Monitoring** (`health.py`): Component and system health tracking
  - Individual component health status (healthy, degraded, unhealthy)
  - System-wide health aggregation
  - Stale component detection
  - Custom health check functions
  - Uptime tracking
- **Shutdown Management** (`shutdown.py`): Graceful shutdown coordination
  - Thread-safe shutdown signaling
  - Signal handler registration (SIGTERM, SIGINT)
  - Cleanup callback coordination
  - Timeout support for blocking operations

### Added - Observability & Monitoring

- **Prometheus Metrics** (`prometheus_metrics.py`): Comprehensive metrics collection
  - Modbus operation metrics (reads, writes, latency, errors)
  - MQTT operation metrics (publishes, connection status)
  - Health status metrics (component health, circuit breaker state)
  - System metrics (uptime, error counters)
  - Register data metrics (battery SOC, power values)
  - HTTP endpoint on port 9090 for Prometheus scraping
  - Context managers for automatic operation timing
- **Structured Logging** (`structured_logging.py`): JSON-formatted logging
  - StructuredLogger class with context field support
  - JSONFormatter for log aggregation systems
  - Exception tracking with full tracebacks
  - Dual mode: JSON or standard text output
- **Grafana Dashboard** (`templates/grafana-dashboard.json`): Production-ready dashboard
  - 8 monitoring panels (connections, battery SOC, power flow, latency, errors, health)
  - Auto-refresh every 10 seconds
  - Time series visualizations
- **Prometheus Alerts** (`templates/prometheus-alerts.yaml`): Alerting rules
  - 11 alert rules across 6 categories
  - Connection alerts (Modbus/MQTT disconnected)
  - Health alerts (component unhealthy/degraded)
  - Error rate alerts (high error rate, no data received)
  - Performance alerts (high latency warnings)
  - Circuit breaker alerts (breaker open state)
  - Battery alerts (SOC critical/low)

### Added - Advanced Features

- **Formula Evaluator** (`formula_evaluator.py`): Calculated register support
  - Safe evaluation of mathematical formulas
  - Support for basic arithmetic operators (+, -, \*, /, %, \*\*)
  - Built-in functions (abs, round, min, max)
  - Dependency resolution and ordering
  - Formula validation with AST parsing
  - Protection against dangerous operations
- **Register Models** (`register_models.py`): Type-safe register definitions
  - RegisterDefinition dataclass with validation
  - CalculatedRegister for formula-based registers
  - RegisterMap with dependency checking
  - Circular dependency detection
  - Min/max value validation
- **Register Processors** (`register_processors.py`): Specialized value processing
  - TemperatureProcessor (validation, unusual value warnings)
  - EquipmentProcessor (code to name mapping)
  - PercentageProcessor (clamping to 0-100%)
  - PowerProcessor (W/kW conversion)
  - EnergyProcessor (Wh/kWh conversion)
  - Chainable processor registry pattern

### Added - Dependency Injection & Testing

- **Service Container** (`container.py`): Dependency injection framework
  - Singleton and factory service registration
  - Automatic dependency resolution
  - Service lifecycle management
  - Configuration-driven container setup
- **Testing Utilities** (`testing.py`): Comprehensive test helpers
  - MockModbusClient for Modbus testing
  - MockMqttClient for MQTT testing
  - Mock health check and metrics collectors
  - Dataclass factories for test data generation
  - Fixture builders for complex test scenarios

### Added - Type Safety & Protocols

- **Protocol Definitions** (`protocols.py`): Structural typing interfaces
  - ModbusClientProtocol for client implementations
  - MqttClientProtocol for MQTT clients
  - HealthCheckProtocol for health monitoring
  - MetricsProtocol for metrics collection
  - Enables duck typing with type safety

### Added - Backward Compatibility Layer

- **Sync Coordinator Wrapper** (`sync_coordinator_wrapper.py`): Synchronous API
  - Wraps async coordinator for backward compatibility
  - Maintains existing synchronous interface
  - Enables gradual migration path

### Added - Documentation

- **Architecture Guide** (`ARCHITECTURE.md`): Comprehensive design documentation
  - 3,665 lines covering all architectural decisions
  - Current state analysis and future roadmap
  - Phase-by-phase migration plan (6 phases)
  - Design patterns and best practices
  - Code examples and usage guidelines
- **Baseline Documentation** (`BASELINE.md`): Pre-async state capture
  - 792 lines documenting synchronous baseline
  - Coverage metrics and test results
  - Performance benchmarks
  - Migration reference point

### Added - Test Coverage

- **243 comprehensive tests** across 16 test modules:
  - `test_async_integration.py` (7 tests): End-to-end async integration
  - `test_async_modbus_client.py` (15 tests): Async Modbus client
  - `test_async_mqtt_client.py` (15 tests): Async MQTT client
  - `test_backward_compatibility.py` (13 tests): MQTT topic/payload compatibility
  - `test_container.py` (10 tests): Dependency injection
  - `test_formula_evaluator.py` (27 tests): Formula evaluation and calculated registers
  - `test_health.py` (21 tests): Health monitoring system
  - `test_prometheus_metrics.py` (21 tests): Metrics collection
  - `test_register_models.py` (18 tests): Register models and validation
  - `test_register_processors.py` (28 tests): Register value processors
  - `test_resilience.py` (24 tests): Resilience patterns
  - `test_shutdown.py` (13 tests): Shutdown management
  - `test_structured_logging.py` (16 tests): Structured logging
  - Plus existing tests for config, coordinator, Modbus, MQTT, Home Assistant

### Changed - Core Modules

- **modbus_client.py**: Enhanced synchronous client
  - Added health check integration
  - Circuit breaker support
  - Connection state machine
  - Improved error handling and recovery
  - +306 lines of improvements
- **mqtt_client.py**: Enhanced synchronous client
  - Health check integration
  - Connection state tracking
  - Better reconnection logic
  - +283 lines of improvements
- **mtec_coordinator.py**: Improved synchronous coordinator
  - Integration with new health and metrics systems
  - Better shutdown handling
  - Cleanup and refactoring
- **exceptions.py**: Complete exception hierarchy rewrite
  - Typed exception classes for all error scenarios
  - ConfigException family (validation, file not found, parse errors)
  - ModbusException family (connection, timeout, read/write errors)
  - MqttException family (connection, publish, subscription errors)
  - FormulaException for calculated register errors
  - +228 lines with comprehensive coverage

### Changed - Configuration & Dependencies

- **pyproject.toml**: Updated project metadata
  - Version bump to support async features
  - New dependencies for async support
  - Enhanced test configuration
  - Updated coverage settings
- **requirements.txt**: Additional dependencies
  - `prometheus-client>=0.21.0` for metrics
  - `pydantic>=2.12.0` for data validation
  - Updated async library versions

### Changed - Documentation

- **CLAUDE.md**: Updated project guide
  - Async architecture documentation
  - New testing guidelines
  - Observability setup instructions
- **README.md**: Minor updates
  - Async feature highlights
  - Updated compatibility notes

### Changed - Development Tooling

- **.pre-commit-config.yaml**: Enhanced linting rules
- **.gitignore**: Additional exclusions for development artifacts

### Fixed

- Copyright notices aligned with LGPL license requirements
- Type safety improvements across all modules
- Proper resource cleanup in all async operations
- Memory leaks in connection handling

### Technical Metrics

- **Total Changes**: 47 files changed, 15,158 insertions(+), 143 deletions(-)
- **Test Coverage**: 243 tests with >85% code coverage
- **Code Quality**: All pre-commit hooks passing (ruff, mypy, pylint, yamllint)
- **Documentation**: 4,457+ lines of architecture and baseline documentation

### Breaking Changes

None - Full backward compatibility maintained through sync wrapper layer.

### Migration Notes

The async migration is fully backward compatible. The original synchronous coordinator continues to work unchanged. Async features can be adopted incrementally:

1. Use existing `mtec_coordinator.py` for synchronous operation (default)
2. Switch to `async_coordinator.py` for full async benefits
3. Use `sync_coordinator_wrapper.py` for hybrid approach

See `ARCHITECTURE.md` for complete migration guide and roadmap.

---

## [0.1.0] - 2026-01-20

### Added

- Initial release forked from [croedel/MTECmqtt](https://github.com/croedel/MTECmqtt)
- Synchronous Modbus RTU client for M-TEC Energybutler
- MQTT publishing with configurable refresh rates
- Home Assistant auto-discovery support
- 80+ register definitions for inverter parameters
- Configuration management via YAML
- Interactive utility tools (mtec_util, mtec_export)
- Basic test suite
- Systemd service installation script
- Documentation and examples

[1.0.0]: https://github.com/sukramj/aiomtec2mqtt/compare/c0dc8a6..8c04c55
[0.1.0]: https://github.com/sukramj/aiomtec2mqtt/releases/tag/v0.1.0

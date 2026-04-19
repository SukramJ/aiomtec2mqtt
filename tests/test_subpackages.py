"""Smoke tests for the re-export sub-packages (ROADMAP 1.4)."""

from __future__ import annotations


def test_client_subpackage_exports() -> None:
    from aiomtec2mqtt import async_modbus_client, async_mqtt_client
    from aiomtec2mqtt.client import AsyncModbusClient, AsyncMqttClient

    assert AsyncModbusClient is async_modbus_client.AsyncModbusClient
    assert AsyncMqttClient is async_mqtt_client.AsyncMqttClient


def test_model_subpackage_exports() -> None:
    from aiomtec2mqtt import config_schema, register_models
    from aiomtec2mqtt.model import ConfigSchema, CoordinatorConfigBuilder, RegisterDefinition

    assert ConfigSchema is config_schema.ConfigSchema
    assert RegisterDefinition is register_models.RegisterDefinition
    assert CoordinatorConfigBuilder.__name__ == "CoordinatorConfigBuilder"


def test_coordinator_subpackage_exports() -> None:
    from aiomtec2mqtt import async_coordinator, coordinator_config
    from aiomtec2mqtt.coordinator import AsyncMtecCoordinator, CoordinatorConfigBuilder

    assert AsyncMtecCoordinator is async_coordinator.AsyncMtecCoordinator
    assert CoordinatorConfigBuilder is coordinator_config.CoordinatorConfigBuilder


def test_integrations_subpackage_exports() -> None:
    from aiomtec2mqtt import hass_int, prometheus_metrics
    from aiomtec2mqtt.integrations import HassIntegration, PrometheusMetrics

    assert HassIntegration is hass_int.HassIntegration
    assert PrometheusMetrics is prometheus_metrics.PrometheusMetrics


def test_support_subpackage_exports() -> None:
    from aiomtec2mqtt import exceptions, health, resilience, shutdown
    from aiomtec2mqtt.support import (
        CircuitBreaker,
        HealthCheck,
        ModbusConnectionError,
        ShutdownManager,
    )

    assert CircuitBreaker is resilience.CircuitBreaker
    assert HealthCheck is health.HealthCheck
    assert ModbusConnectionError is exceptions.ModbusConnectionError
    assert ShutdownManager is shutdown.ShutdownManager

"""
Integrations subpackage — Home Assistant + Prometheus wiring.

Re-exports external-system integrations under a cohesive namespace::

    from aiomtec2mqtt.integrations import HassIntegration, PrometheusMetrics

(c) 2026 by SukramJ
"""

from __future__ import annotations

from aiomtec2mqtt.hass_int import HassIntegration
from aiomtec2mqtt.prometheus_metrics import PrometheusMetrics

__all__ = ["HassIntegration", "PrometheusMetrics"]

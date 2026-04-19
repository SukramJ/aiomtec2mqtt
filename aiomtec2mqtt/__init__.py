"""
Top-level package for aiomtec2mqtt.

Provides MQTT publication of values read from an M-TEC Energybutler via
Modbus, optional Home Assistant discovery integration, and small utilities.
"""

from __future__ import annotations

__all__ = ["__version__", "config", "hass_int"]
__version__ = "1.1.0"  # x-release-please-version

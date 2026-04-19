"""
Reusable testing utilities for projects that integrate with ``aiomtec2mqtt``.

The package re-exports the most useful test doubles, replay helpers, and
assertion utilities under a stable public namespace, and registers a pytest
plugin that exposes ready-to-use fixtures.

Importing this top-level module is intentionally cheap: heavy imports happen
only when a sub-module is actually used.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.1.0"  # x-release-please-version

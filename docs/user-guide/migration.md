# Migration Guide

This guide tracks breaking changes between major versions of `aiomtec2mqtt`
and documents the steps required to upgrade an existing deployment.

Each section describes:

- what changed (configuration keys, MQTT topic layout, Python API)
- why the change was made
- how to migrate safely (config edits, downstream adjustments, rollback path)

!!! note
Until `aiomtec2mqtt` ships its first major release (`2.0.0`), this page is
intentionally a placeholder. Patch and minor releases on the `1.x` line are
expected to stay backward-compatible. Any behavioural change that might
require user attention will still be listed below so you can review it
before upgrading.

## General upgrade checklist

Regardless of the target version, the following steps apply to every upgrade:

1. Read the [CHANGELOG](https://github.com/SukramJ/aiomtec2mqtt/blob/main/CHANGELOG.md)
   entry for the version you are upgrading to (and every intermediate version).
2. Back up your current configuration file
   (`~/.config/aiomtec2mqtt/config.yaml` on Linux) and your systemd unit if you
   use one.
3. Snapshot your Home Assistant or evcc integration if you rely on it — so you
   can roll back quickly if the MQTT topic layout changes.
4. Pin the previous version in `requirements.txt` as a safety net, e.g.
   `aiomtec2mqtt==<previous-version>`.
5. Upgrade: `pip install --upgrade aiomtec2mqtt` (or `uv pip install --upgrade`).
6. Run the application once in the foreground and watch the log for warnings
   or deprecation notices before re-enabling the systemd service.

## Version-specific notes

### `1.x` → `1.x` (minor / patch updates)

No breaking changes between minor releases on the `1.x` line so far. Refer to
the generated release notes (GitHub → Releases) for highlights.

### `1.x` → `2.0.0` (planned)

_To be filled in when `2.0.0` is released._

Anticipated topics this section will cover:

- Python version floor (possible move to 3.14+).
- Optional removal of the legacy `sync_coordinator_wrapper` facade.
- Any configuration keys that move from `mqtt:` to `integrations.mqtt:` after
  the sub-package refactor.
- Home Assistant auto-discovery payload adjustments (if any).

## Reporting migration issues

If you hit an upgrade problem that is not covered here, please open an issue
at [github.com/SukramJ/aiomtec2mqtt/issues](https://github.com/SukramJ/aiomtec2mqtt/issues)
with:

- the source and target version,
- your redacted `config.yaml`,
- a log excerpt showing the first warning or error after the upgrade.

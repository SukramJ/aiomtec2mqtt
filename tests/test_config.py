"""Tests for configuration loading and register map parsing."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestConfigLoading:
    """Tests for configuration file loading."""

    def test_init_config_invalid_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """init_config should ignore invalid YAML and return empty when not found elsewhere."""
        from aiomtec2mqtt.config import init_config

        tmp_path.joinpath("config.yaml").write_text("not: [valid\n yaml")
        monkeypatch.chdir(tmp_path)

        assert init_config() == {}

    def test_init_config_reads_from_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """init_config should read a config.yaml from the current working directory."""
        from aiomtec2mqtt.config import init_config

        cfg = {
            "DEBUG": True,
            "HASS_BASE_TOPIC": "homeassistant",
            "HASS_BIRTH_GRACETIME": 5,
            "HASS_ENABLE": False,
            "MODBUS_FRAMER": "rtu",
            "MODBUS_IP": "127.0.0.1",
            "MODBUS_PORT": 502,
            "MODBUS_RETRIES": 1,
            "MODBUS_SLAVE": 1,
            "MODBUS_TIMEOUT": 3,
            "MQTT_FLOAT_FORMAT": "{:.1f}",
            "MQTT_LOGIN": "u",
            "MQTT_PASSWORD": "p",
            "MQTT_PORT": 1883,
            "MQTT_SERVER": "localhost",
            "MQTT_TOPIC": "MTEC",
            "REFRESH_CONFIG": 1,
            "REFRESH_DAY": 2,
            "REFRESH_NOW": 3,
            "REFRESH_STATIC": 4,
            "REFRESH_TOTAL": 5,
        }
        tmp_path.joinpath("config.yaml").write_text(
            "\n".join(f"{k}: {v!r}" for k, v in cfg.items())
        )
        monkeypatch.chdir(tmp_path)

        loaded = init_config()
        # a couple of sanity checks
        assert loaded["DEBUG"] is True
        assert loaded["MQTT_SERVER"] == "localhost"


class TestConfigFileCreation:
    """Tests for configuration file creation."""

    def test_create_config_file_writes_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_config_file should write a file under XDG_CONFIG_HOME or APPDATA using the template."""
        from aiomtec2mqtt.config import CONFIG_FILE, CONFIG_PATH, create_config_file

        # ensure no DNS resolution path taken and Home Assistant disabled
        monkeypatch.setattr("socket.gethostbyname", lambda n: (_ for _ in ()).throw(OSError()))
        inputs: list[str] = ["192.0.2.1", "n"]  # ip, hass disabled
        monkeypatch.setattr("builtins.input", lambda *a, **k: inputs.pop(0))

        cfg_base = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_base))

        assert create_config_file() is True
        out_file = cfg_base / CONFIG_PATH / CONFIG_FILE
        assert out_file.exists(), f"expected {out_file} to be created"
        content = out_file.read_text()
        assert "HASS_ENABLE : False" in content
        assert "MODBUS_IP" in content and "192.0.2.1" in content


class TestRegisterMap:
    """Tests for register map initialization."""

    def test_init_register_map_loads_and_validates(self) -> None:
        """init_register_map should parse bundled YAML and return groups and cleaned map."""
        from aiomtec2mqtt.config import init_register_map
        from aiomtec2mqtt.const import Register

        reg_map, groups = init_register_map()
        assert isinstance(reg_map, dict) and isinstance(groups, list)
        # a few known keys from the repo registers.yaml
        assert "10000" in reg_map
        assert Register.NAME in reg_map["10000"]
        assert "static" in groups

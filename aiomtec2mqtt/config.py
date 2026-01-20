"""
Read YAML config files.

(c) 2024 by Christian Rödel
(c) 2024 by SukramJ
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from typing import Any, Final, cast

import yaml

from aiomtec2mqtt.const import (
    CONFIG_FILE,
    CONFIG_PATH,
    CONFIG_ROOT,
    CONFIG_TEMPLATE,
    ENV_APPDATA,
    ENV_XDG_CONFIG_HOME,
    FILE_REGISTERS,
    MANDATORY_PARAMETERS,
    OPTIONAL_PARAMETERS,
    UTF8,
    Config,
    Register,
)

_LOGGER: Final = logging.getLogger(__name__)


# Create new config file
def create_config_file() -> bool:
    """Read the config file."""
    _LOGGER.info("Creating %s", CONFIG_FILE)

    # Resolve hostname
    try:
        ip_addr = socket.gethostbyname("espressif")
        _LOGGER.info("Found espressif server: %s", ip_addr)
    except OSError:
        _LOGGER.info("Couldn't find espressif server")
        ip_addr = input("Please enter IP address of espressif server: ")

    opt = input("Enable HomeAssistant support? (y/N): ")
    hass_cfg = (
        f"{Config.HASS_ENABLE} : True" if opt.lower() == "y" else f"{Config.HASS_ENABLE} : False"
    )

    # Read template
    try:
        BASE_DIR = os.path.dirname(__file__)  # Base installation directory
        templ_fname = os.path.join(BASE_DIR, CONFIG_TEMPLATE)
        with open(file=templ_fname, encoding=UTF8) as file:
            data = file.read()
    except Exception as ex:
        _LOGGER.info("ERROR - Couldn't read '%s': %s", CONFIG_TEMPLATE, ex)
        return False

    # Customize: normalize two key lines regardless of template formatting
    new_lines: list[str] = []
    for line in data.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{Config.HASS_ENABLE}"):
            new_lines.append(hass_cfg)
        elif stripped.startswith(f"{Config.MODBUS_IP}"):
            new_lines.append(f"{Config.MODBUS_IP} : '{ip_addr}'")
        else:
            new_lines.append(line)
    data = "\n".join(new_lines) + "\n"

    # Write customized config
    # Usually something like ~/.config/mtec2mqtt/config.yaml resp. 'C:\\Users\\xxxx\\AppData\\Roaming'
    if cfg_path := os.environ.get(ENV_XDG_CONFIG_HOME) or os.environ.get(ENV_APPDATA):
        cfg_fname = os.path.join(cfg_path, CONFIG_PATH, CONFIG_FILE)
    else:
        cfg_fname = os.path.join(
            os.path.expanduser("~"), CONFIG_ROOT, CONFIG_PATH, CONFIG_FILE
        )  # ~/.config/mtec2mqtt/config.yaml

    try:
        os.makedirs(os.path.dirname(cfg_fname), exist_ok=True)
        with open(file=cfg_fname, mode="w", encoding=UTF8) as file:
            file.write(data)
    except Exception as ex:
        _LOGGER.error("ERROR - Couldn't write %s: %s", cfg_fname, ex)
        return False

    _LOGGER.info("Successfully created %s", cfg_fname)
    return True


def init_config() -> dict[str, Any]:
    """
    Read configuration from YAML file.

    Behavior note:
    - If a config.yaml exists in the current working directory, only that file is
      considered. If it cannot be parsed (invalid YAML), an empty dict is returned
      without falling back to user config locations. This makes local development
      and tests deterministic.
    - If there is no config.yaml in the current directory, search common user
      config locations and return the first successfully parsed config; otherwise
      return an empty dict.
    """
    # 1) Prefer config.yaml in the current working directory if present
    cwd_conf = os.path.join(os.getcwd(), CONFIG_FILE)
    try:
        with open(file=cwd_conf, encoding=UTF8) as f_conf:
            config = cast(dict[str, Any], yaml.safe_load(f_conf))
            _LOGGER.info("Using config YAML file: %s", cwd_conf)
            return config
    except OSError:
        # Not found in CWD; continue to look in user locations
        pass
    except yaml.YAMLError as err:
        # Invalid YAML in CWD: do not fall back to other locations
        _LOGGER.debug("Couldn't read config YAML file %s : %s", cwd_conf, str(err))
        return {}

    # 2) Look in user config locations
    conf_files: list[str] = []
    if cfg_path := os.environ.get(ENV_XDG_CONFIG_HOME) or os.environ.get(ENV_APPDATA):
        conf_files.append(os.path.join(cfg_path, CONFIG_PATH, CONFIG_FILE))
    else:
        conf_files.append(
            os.path.join(os.path.expanduser("~"), CONFIG_ROOT, CONFIG_PATH, CONFIG_FILE)
        )

    for fname_conf in conf_files:
        try:
            with open(file=fname_conf, encoding=UTF8) as f_conf:
                config = cast(dict[str, Any], yaml.safe_load(f_conf))
                _LOGGER.info("Using config YAML file: %s", fname_conf)
                return config
        except OSError as err:
            _LOGGER.debug("Couldn't open config YAML file: %s", str(err))
        except yaml.YAMLError as err:
            _LOGGER.debug("Couldn't read config YAML file %s : %s", fname_conf, str(err))

    return {}


def init_register_map() -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Read inverter registers and their mapping from YAML file."""
    BASE_DIR = os.path.dirname(__file__)  # Base installation directory
    try:
        fname_regs = os.path.join(BASE_DIR, FILE_REGISTERS)
        with open(fname_regs, encoding=UTF8) as f_regs:
            r_map = cast(dict[str, dict[str, Any]], yaml.safe_load(f_regs))
    except OSError as err:
        _LOGGER.fatal("Couldn't open registers YAML file: %s", str(err))
        sys.exit(1)
    except yaml.YAMLError as err:
        _LOGGER.fatal("Couldn't read config YAML file %s: %s", fname_regs, str(err))
        sys.exit(1)

    # Syntax checks
    reg_map: dict[str, dict[str, Any]] = {}

    reg_groups: list[str] = []

    error = False
    for key, val in r_map.items():
        # Check for mandatory parameters
        for p in MANDATORY_PARAMETERS:
            if not val.get(p):
                _LOGGER.warning(
                    "Skipping invalid register config: %s. Missing mandatory parameter: %s.",
                    key,
                    p,
                )
                error = True
                break

        if not error:  # All mandatory parameters found
            item = val.copy()
            # Check optional parameters and add defaults, if not found
            for param, default in OPTIONAL_PARAMETERS.items():
                if param not in item:
                    item[param] = default
            reg_map[key] = item  # Append to reg_map

            if (group := item[Register.GROUP]) and group not in reg_groups:
                reg_groups.append(group)  # Append to group list
    return reg_map, reg_groups


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(filename)s: %(message)s")

# Avoid running side effects (I/O, sys.exit) on import so tests can safely import this module.
if __name__ == "__main__":
    if not init_config():
        if create_config_file():  # Create a new config
            if not init_config():
                _LOGGER.fatal("Couldn't open config YAML file")
                sys.exit(1)
        else:
            _LOGGER.fatal("Couldn't create config YAML file")
            sys.exit(1)

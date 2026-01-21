"""
A test utility for MTEC Modbus API.

(c) 2023 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Final

from aiomtec2mqtt.async_modbus_client import AsyncModbusClient
from aiomtec2mqtt.config import init_config, init_register_map
from aiomtec2mqtt.const import Register, RegisterGroup

if TYPE_CHECKING:
    pass

_LOGGER: Final = logging.getLogger(__name__)


def read_register(*, api: AsyncModbusClient) -> None:
    """Read register."""
    _LOGGER.info("-------------------------------------")
    register = input("Register: ")

    async def _read() -> dict[str, Any] | None:
        return await api.read_register(register=register)

    if data := asyncio.run(_read()):
        _LOGGER.info(
            "Register %s (%s): %s %s",
            register,
            data.get(Register.NAME, "Unknown"),
            data.get(Register.VALUE, "N/A"),
            data.get(Register.UNIT, ""),
        )


def read_register_group(*, api: AsyncModbusClient) -> None:
    """Read register group."""
    _LOGGER.info("-------------------------------------")
    line = "Groups: "
    for g in sorted(api.register_groups):
        line += g + ", "
    _LOGGER.info("%s %s", line, "all")

    group_input = input("Register group (or RETURN for all): ")

    async def _read_group() -> dict[str, Any]:
        if group_input in ("", "all"):
            # Read all groups
            all_data: dict[str, Any] = {}
            for grp in api.register_groups:
                try:
                    if grp_data := await api.read_register_group(group_name=RegisterGroup(grp)):
                        all_data.update(grp_data)
                except Exception as ex:
                    _LOGGER.warning("Failed to read group %s: %s", grp, ex)
            return all_data
        # Read specific group
        return await api.read_register_group(group_name=RegisterGroup(group_input)) or {}

    _LOGGER.info("Reading...")
    if data := asyncio.run(_read_group()):
        for reg_name, value in data.items():
            # Find register info from register_map
            reg_info: dict[str, Any] = {}
            reg_addr = ""
            for addr, info in api.register_map.items():
                if info.get(Register.NAME) == reg_name:
                    reg_info = info
                    reg_addr = addr
                    break
            unit = reg_info.get(Register.UNIT, "")
            _LOGGER.info(
                "- %s; %s; %s; %s;",
                reg_addr,
                reg_name,
                value,
                unit,
            )


def write_register(*, api: AsyncModbusClient) -> None:
    """Write register."""
    _LOGGER.info("-------------------------------------")
    _LOGGER.info("Current settings of writable registers:")
    _LOGGER.info("Reg   Name                           Value  Unit")
    _LOGGER.info("----- ------------------------------ ------ ----")

    register_map_sorted = dict(sorted(api.register_map.items()))

    async def _read_single(register: str) -> Any:
        return await api.read_register(register=register)

    for register, item in register_map_sorted.items():
        if item.get(Register.WRITABLE):
            data = asyncio.run(_read_single(register))
            value = ""
            if data:
                value = data.get(Register.VALUE, "")
            unit = item.get(Register.UNIT, "")
            _LOGGER.info("%s; %s; %s; %s", register, item[Register.NAME], str(value), unit)

    _LOGGER.info("")
    register = input("Register: ")
    value = input("Value: ")

    _LOGGER.info("WARNING: Be careful when writing registers to your Inverter!")
    yn = input(f"Do you really want to set register {register} to '{value}'? (y/N)")
    if yn in ("y", "Y"):

        async def _write() -> bool:
            return await api.write_register(register=register, value=value)

        if asyncio.run(_write()):
            _LOGGER.info("New value successfully set")
        else:
            _LOGGER.info("Writing failed")
    else:
        _LOGGER.info("Write aborted by user")


def list_register_config(*, api: AsyncModbusClient) -> None:
    """List register config."""
    _LOGGER.info("-------------------------------------")
    _LOGGER.info(
        "Reg   MQTT Parameter                 Unit Mode Group           Name                   "
    )
    _LOGGER.info(
        "----- ------------------------------ ---- ---- --------------- -----------------------"
    )
    register_map_sorted = dict(sorted(api.register_map.items()))
    for register, item in register_map_sorted.items():
        if (
            not register.isnumeric()
        ):  # non-numeric registers are deemed to be calculated pseudo-registers
            register = ""
        mqtt = item.get(Register.MQTT, "")
        unit = item.get(Register.UNIT, "")
        group = item.get(Register.GROUP, "")
        mode = "RW" if item.get(Register.WRITABLE) else "R"
        _LOGGER.info(
            "%s; %s; %s; %s; %s; %s", register, mqtt, unit, mode, group, item[Register.NAME]
        )


def list_register_config_by_groups(*, api: AsyncModbusClient) -> None:
    """List register config by groups."""
    for group in api.register_groups:
        _LOGGER.info("-------------------------------------")
        _LOGGER.info("Group %s:", group)
        _LOGGER.info("")
        _LOGGER.info("Reg   MQTT Parameter                 Unit Mode Name                   ")
        _LOGGER.info("----- ------------------------------ ---- ---- -----------------------")
        register_map_sorted = dict(sorted(api.register_map.items()))
        for register, item in register_map_sorted.items():
            if item.get(Register.GROUP) == group:
                if (
                    not register.isnumeric()
                ):  # non-numeric registers are deemed to be calculated pseudo-registers
                    register = ""
                mqtt = item.get(Register.MQTT, "")
                unit = item.get(Register.UNIT, "")
                mode = "RW" if item.get(Register.WRITABLE) else "R"
                _LOGGER.info(
                    "%s; %s; %s; %s; %s; %s",
                    group,
                    register,
                    mqtt,
                    unit,
                    mode,
                    item[Register.NAME],
                )
        _LOGGER.info("")


def main() -> None:
    """Start the mtec utilities."""
    register_map, register_groups = init_register_map()
    config = init_config()
    api = AsyncModbusClient(
        config=config, register_map=register_map, register_groups=register_groups
    )

    async def _connect() -> bool:
        return await api.connect()

    async def _disconnect() -> None:
        await api.disconnect()

    asyncio.run(_connect())

    while True:
        print("=====================================")  # noqa: T201
        print("Menu:")  # noqa: T201
        print("  1: List all known registers")  # noqa: T201
        print("  2: List register configuration by groups")  # noqa: T201
        print("  3: Read register group from Inverter")  # noqa: T201
        print("  4: Read single register from Inverter")  # noqa: T201
        print("  5: Write register to Inverter")  # noqa: T201
        print("  x: Exit")  # noqa: T201
        opt = input("Please select: ")
        if opt == "1":
            list_register_config(api=api)
        elif opt == "2":
            list_register_config_by_groups(api=api)
        if opt == "3":
            read_register_group(api=api)
        elif opt == "4":
            read_register(api=api)
        elif opt == "5":
            write_register(api=api)
        elif opt in ("x", "X"):
            break

    asyncio.run(_disconnect())
    print("Bye!")  # noqa: T201


# -------------------------------
if __name__ == "__main__":
    main()

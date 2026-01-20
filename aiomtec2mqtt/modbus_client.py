"""
Modbus API for M-TEC Energybutler.

(c) 2023 by Christian Rödel
(c) 2026 by SukramJ
"""

from __future__ import annotations

import logging
from typing import Any, Final, cast

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
from pymodbus.framer import FramerType
from pymodbus.pdu.register_message import ReadHoldingRegistersResponse

from aiomtec2mqtt.const import DEFAULT_FRAMER, Config, Register, RegisterGroup
from aiomtec2mqtt.exceptions import (
    ModbusConnectionError,
    ModbusReadError,
    ModbusTimeoutError,
    ModbusWriteError,
)
from aiomtec2mqtt.health import HealthCheck
from aiomtec2mqtt.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    ConnectionState,
    ConnectionStateMachine,
)

_LOGGER: Final = logging.getLogger(__name__)


class MTECModbusClient:
    """Modbus API for MTEC Energy Butler."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        register_map: dict[str, dict[str, Any]],
        register_groups: list[str],
        health_check: HealthCheck | None = None,
    ) -> None:
        """
        Init the modbus client.

        Args:
            config: Configuration dictionary
            register_map: Register mapping dictionary
            register_groups: List of register groups
            health_check: Optional health check manager for monitoring

        """
        self._error_count = 0
        self._register_map: Final = register_map
        self._register_groups: Final = register_groups
        self._modbus_client: ModbusTcpClient | None = None
        # Cache for computed register clusters. Keyed by a normalized tuple of numeric register addresses.
        self._cluster_cache: Final[dict[tuple[int, ...], list[dict[str, Any]]]] = {}
        # Precompute frequently used lookups to reduce per-call overhead
        # Numeric registers (as strings) used when reading "all" registers
        self._all_numeric_registers: Final[list[str]] = [r for r in register_map if r.isnumeric()]
        # Mapping from MQTT name to numeric register string for quick writes
        self._mqtt_name_to_register: Final[dict[str, str]] = {
            cast(str, item.get(Register.MQTT)): reg
            for reg, item in register_map.items()
            if reg.isnumeric() and item.get(Register.MQTT)
        }

        self._modbus_framer: Final[str] = config.get(Config.MODBUS_FRAMER, DEFAULT_FRAMER)
        self._modbus_host: Final[str] = config[Config.MODBUS_IP]
        self._modbus_port: Final[int] = config[Config.MODBUS_PORT]
        self._modbus_retries: Final[int] = config[Config.MODBUS_RETRIES]
        self._modbus_slave: Final[int] = config[Config.MODBUS_SLAVE]
        self._modbus_timeout: Final[int] = config[Config.MODBUS_TIMEOUT]

        # Resilience patterns
        self._circuit_breaker: Final = CircuitBreaker(
            name="modbus",
            config=CircuitBreakerConfig(
                failure_threshold=5,  # Open circuit after 5 failures
                success_threshold=2,  # Close after 2 successes in HALF_OPEN
                timeout=30.0,  # Try recovery after 30s
            ),
        )
        self._state_machine: Final = ConnectionStateMachine(name="modbus")
        self._health_check = health_check

        # Register with health check if provided
        if self._health_check:
            self._health_check.register_component(name="modbus")

        _LOGGER.debug("Modbus client initialized")

    def __del__(self) -> None:
        """Cleanup the modbus client."""
        self.disconnect()

    @property
    def error_count(self) -> int:
        """Return the error count."""
        return self._error_count

    @property
    def register_groups(self) -> list[str]:
        """Return the register groups."""
        return self._register_groups

    @property
    def register_map(self) -> dict[str, dict[str, Any]]:
        """Return the register map."""
        return self._register_map

    def connect(self) -> bool:
        """
        Connect to modbus server.

        Returns:
            True if connection successful, False otherwise

        Raises:
            ModbusConnectionError: If connection fails

        """
        # Update state machine
        if self._state_machine.state == ConnectionState.CONNECTED:
            self._state_machine.transition_to(new_state=ConnectionState.RECONNECTING)
        else:
            self._state_machine.transition_to(new_state=ConnectionState.CONNECTING)

        self._error_count = 0
        _LOGGER.debug(
            "Connecting to server %s:%i (framer=%s)",
            self._modbus_host,
            self._modbus_port,
            self._modbus_framer,
        )

        try:
            self._modbus_client = ModbusTcpClient(
                host=self._modbus_host,
                port=self._modbus_port,
                framer=FramerType(self._modbus_framer),
                timeout=self._modbus_timeout,
                retries=self._modbus_retries,
            )

            if self._modbus_client.connect():  # type: ignore[no-untyped-call]
                _LOGGER.info(
                    "Successfully connected to server %s:%i", self._modbus_host, self._modbus_port
                )
                self._state_machine.transition_to(new_state=ConnectionState.CONNECTED)
                if self._health_check:
                    self._health_check.record_success(name="modbus")
                return True
            # Connection failed
            error_msg = f"Couldn't connect to server {self._modbus_host}:{self._modbus_port}"
            _LOGGER.error(error_msg)
            self._state_machine.transition_to(new_state=ConnectionState.ERROR, error=error_msg)
            if self._health_check:
                self._health_check.record_failure(name="modbus", error=error_msg)
            return False  # noqa: TRY300

        except Exception as ex:
            error_msg = f"Connection error to {self._modbus_host}:{self._modbus_port}: {ex}"
            _LOGGER.exception(error_msg)
            self._state_machine.transition_to(new_state=ConnectionState.ERROR, error=error_msg)
            if self._health_check:
                self._health_check.record_failure(name="modbus", error=error_msg)
            raise ModbusConnectionError(
                message=error_msg,
                details={"host": self._modbus_host, "port": self._modbus_port},
            ) from ex

    def disconnect(self) -> None:
        """Disconnect from Modbus server."""
        if self._modbus_client and self._modbus_client.is_socket_open():
            self._modbus_client.close()  # type: ignore[no-untyped-call]
            _LOGGER.debug("Successfully disconnected from server")
        self._state_machine.transition_to(new_state=ConnectionState.DISCONNECTED)

    def get_register_list(self, *, group: RegisterGroup) -> list[str]:
        """Get a list of all registers which belong to a given group."""
        registers: list[str] = []
        for register, item in self._register_map.items():
            if item[Register.GROUP] == group:
                registers.append(register)

        if len(registers) == 0:
            _LOGGER.error("Unknown or empty register group: %s", group)
            return []
        return registers

    def read_modbus_data(self, *, registers: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """
        Read modbus data.

        This is the main API function. It either fetches all registers or a list of given registers.
        """
        data: dict[str, dict[str, Any]] = {}
        _LOGGER.debug("Retrieving data...")

        if registers is None:  # Create a list of all (numeric) registers
            # non-numeric registers are deemed to be calculated pseudo-registers
            registers = self._all_numeric_registers

        cluster_list = self._get_register_clusters(registers=registers)
        for reg_cluster in cluster_list:
            offset = 0
            _LOGGER.debug(
                "Fetching data for cluster start %s, length %s, items %s",
                reg_cluster["start"],
                reg_cluster[Register.LENGTH],
                len(reg_cluster["items"]),
            )
            if rawdata := self._read_registers(
                register=reg_cluster["start"], length=reg_cluster[Register.LENGTH]
            ):
                for item in reg_cluster["items"]:
                    if item.get(Register.TYPE):  # type==None means dummy
                        register = str(reg_cluster["start"] + offset)
                        if data_decoded := self._decode_rawdata(
                            rawdata=rawdata, offset=offset, item=item
                        ):
                            data.update({register: data_decoded})
                        else:
                            _LOGGER.error("Decoding error while decoding register %s", register)
                    offset += item[Register.LENGTH]

        _LOGGER.debug("Data retrieval completed")
        return data

    def write_register(self, *, register: str, value: Any) -> bool:
        """
        Write a value to a register.

        Args:
            register: Register address as string
            value: Value to write

        Returns:
            True if successful, False otherwise

        Raises:
            ModbusWriteError: On write errors

        """
        # Check if connected
        if self._modbus_client is None:
            _LOGGER.error("Can't write register - not connected")
            return False

        # Lookup register
        if not (item := self._register_map.get(str(register), None)):
            _LOGGER.error("Can't write unknown register: %s", register)
            return False
        if item.get(Register.WRITABLE, False) is False:
            _LOGGER.error("Can't write register which is marked read-only: %s", register)
            return False

        # check value
        try:
            if isinstance(value, str):
                value = float(value) if "." in value else int(value)
        except Exception:
            _LOGGER.error("Invalid numeric value: %s", value)
            return False

        # adjust scale
        if (item_scale := int(item.get(Register.SCALE, 1))) > 1:
            value *= item_scale

        try:
            # Use circuit breaker for write operations
            def _do_write() -> bool:
                """Perform the write operation."""
                assert self._modbus_client is not None  # Already checked above
                result = self._modbus_client.write_register(
                    address=int(register), value=int(value), device_id=self._modbus_slave
                )
                if result.isError():
                    error_msg = f"Error writing register {register} value {value}"
                    _LOGGER.error(error_msg)
                    if self._health_check:
                        self._health_check.record_failure(name="modbus", error=error_msg)
                    raise ModbusWriteError(
                        message=error_msg,
                        address=int(register),
                        slave_id=self._modbus_slave,
                        details={"value": value},
                    )
                return True

            # Call through circuit breaker
            result: bool = cast(bool, self._circuit_breaker.call(_do_write))
        except ModbusException as ex:
            error_msg = f"Modbus exception writing register {register}: {ex}"
            _LOGGER.error(error_msg)
            if self._health_check:
                self._health_check.record_failure(name="modbus", error=error_msg)
            raise ModbusWriteError(
                message=error_msg,
                address=int(register),
                slave_id=self._modbus_slave,
                details={"value": value},
            ) from ex
        except ModbusWriteError:
            # Re-raise our typed exception
            raise
        except Exception as ex:
            error_msg = f"Unexpected error writing register {register}: {ex}"
            _LOGGER.exception(error_msg)
            if self._health_check:
                self._health_check.record_failure(name="modbus", error=error_msg)
            raise ModbusWriteError(
                message=error_msg,
                address=int(register),
                slave_id=self._modbus_slave,
                details={"value": value},
            ) from ex
        else:
            if self._health_check:
                self._health_check.record_success(name="modbus")
            return result

    def write_register_by_name(self, *, name: str, value: Any) -> bool:
        """Write a value to a register with a given name."""
        if (register := self._mqtt_name_to_register.get(name)) is None:
            _LOGGER.error("Can't write unknown register with name: %s", name)
            return False
        item = self._register_map[register]
        if value_items := item.get(Register.VALUE_ITEMS):
            for value_modbus, value_display in value_items.items():
                if value_display == value:
                    value = value_modbus
                    break
        return self.write_register(register=register, value=value)

    def _decode_rawdata(
        self, *, rawdata: ReadHoldingRegistersResponse, offset: int, item: dict[str, Any]
    ) -> dict[str, Any]:
        """Decode the result from rawdata, starting at offset."""
        assert self._modbus_client is not None  # Called after successful read
        dt = self._modbus_client.DATATYPE
        try:
            val = None
            item_type = str(item[Register.TYPE])
            item_length = int(item[Register.LENGTH])

            # sanity check: ensure we have enough data
            if offset < 0 or item_length <= 0 or offset + item_length > len(rawdata.registers):
                _LOGGER.error(
                    "Decoding bounds error (type=%s, offset=%s, length=%s, available=%s)",
                    item_type,
                    offset,
                    item_length,
                    len(rawdata.registers),
                )
                return {}

            if item_type == "U16":
                reg = rawdata.registers[offset : offset + 1]
                val = self._modbus_client.convert_from_registers(
                    registers=reg, data_type=dt.UINT16
                )
            elif item_type == "I16":
                reg = rawdata.registers[offset : offset + 1]
                val = self._modbus_client.convert_from_registers(registers=reg, data_type=dt.INT16)
            elif item_type == "U32":
                reg = rawdata.registers[offset : offset + 2]
                val = self._modbus_client.convert_from_registers(
                    registers=reg, data_type=dt.UINT32
                )
            elif item_type == "I32":
                reg = rawdata.registers[offset : offset + 2]
                val = self._modbus_client.convert_from_registers(registers=reg, data_type=dt.INT32)
            elif item_type == "BYTE":
                if item_length == 1:
                    reg1 = int(rawdata.registers[offset])
                    val = f"{reg1 >> 8:02d} {reg1 & 0xFF:02d}"
                elif item_length == 2:
                    reg1 = int(rawdata.registers[offset])
                    reg2 = int(rawdata.registers[offset + 1])
                    val = f"{reg1 >> 8:02d} {reg1 & 0xFF:02d}  {reg2 >> 8:02d} {reg2 & 0xFF:02d}"
                elif item_length == 4:
                    reg1 = int(rawdata.registers[offset])
                    reg2 = int(rawdata.registers[offset + 1])
                    reg3 = int(rawdata.registers[offset + 2])
                    reg4 = int(rawdata.registers[offset + 3])
                    val = (
                        f"{reg1 >> 8:02d} {reg1 & 0xFF:02d} {reg2 >> 8:02d} {reg2 & 0xFF:02d}  "
                        f"{reg3 >> 8:02d} {reg3 & 0xFF:02d} {reg4 >> 8:02d} {reg4 & 0xFF:02d}"
                    )
                else:
                    _LOGGER.error("Unsupported BYTE length: %s", item_length)
                    return {}
            elif item_type == "BIT":
                if item_length == 1:
                    reg1 = int(rawdata.registers[offset])
                    val = f"{reg1:016b}"
                elif item_length == 2:
                    reg1 = int(rawdata.registers[offset])
                    reg2 = int(rawdata.registers[offset + 1])
                    val = f"{reg1:016b} {reg2:016b}"
                else:
                    # support generic N registers as concatenated 16-bit groups
                    bits = [
                        f"{int(rawdata.registers[offset + i]):016b}" for i in range(item_length)
                    ]
                    val = " ".join(bits)
            elif item_type == "DAT":
                if offset + 3 > len(rawdata.registers):
                    _LOGGER.error("DAT requires 3 registers but not enough data available")
                    return {}
                reg1 = int(rawdata.registers[offset])
                reg2 = int(rawdata.registers[offset + 1])
                reg3 = int(rawdata.registers[offset + 2])
                val = (
                    f"{reg1 >> 8:02d}-{reg1 & 0xFF:02d}-{reg2 >> 8:02d} "
                    f"{reg2 & 0xFF:02d}:{reg3 >> 8:02d}:{reg3 & 0xFF:02d}"
                )
            elif item_type == "STR":
                # item_length defines number of 16-bit registers to read
                reg = rawdata.registers[offset : offset + item_length]
                sval = self._modbus_client.convert_from_registers(
                    registers=reg, data_type=dt.STRING
                )
                # strip trailing null bytes and spaces without using multi-character rstrip (B005)
                if isinstance(sval, str):
                    # First remove spaces, then nulls, then spaces again to catch sequences like " \x00 "
                    val = sval.rstrip(" ").rstrip("\x00").rstrip(" ")
                else:
                    val = sval
            else:
                _LOGGER.error("Unknown type %s to decode", item_type)
                return {}

            # apply scaling to numeric values
            item_scale = int(item.get(Register.SCALE, 1))
            if item_scale > 1 and isinstance(val, (int, float)):
                val = float(val) / item_scale

            return {
                Register.NAME: item[Register.NAME],
                Register.VALUE: val,
                Register.UNIT: item.get(Register.UNIT, ""),
            }
        except Exception as ex:
            _LOGGER.error(
                "Exception while decoding data (type=%s, offset=%s, length=%s): %s",
                item.get(Register.TYPE),
                offset,
                item.get(Register.LENGTH),
                ex,
            )
            return {}

    def _generate_register_clusters(self, *, registers: list[str]) -> list[dict[str, Any]]:
        """
        Create clusters.

        Optimizations:
        - Sort numerically instead of lexicographically to ensure proper clustering.
        - Ignore non-numeric and unknown registers early to reduce loop work.
        """
        cluster: dict[str, Any] = {"start": 0, Register.LENGTH: 0, "items": []}
        cluster_list: list[dict[str, Any]] = []

        numeric_regs = sorted(
            {int(r) for r in registers if r.isnumeric() and r in self._register_map}
        )
        for reg in numeric_regs:
            item = self._register_map[str(reg)]
            # if there is a gap to the current cluster, start a new one
            if reg > cluster["start"] + cluster[Register.LENGTH]:
                if cluster["start"] > 0:  # append previous cluster (not the initial dummy)
                    cluster_list.append(cluster)
                cluster = {"start": reg, Register.LENGTH: 0, "items": []}
            # extend current cluster by item length and append the item
            cluster[Register.LENGTH] += item[Register.LENGTH]
            cluster["items"].append(item)

        if cluster["start"] > 0:  # append last cluster
            cluster_list.append(cluster)

        return cluster_list

    def _get_register_clusters(self, *, registers: list[str]) -> list[dict[str, Any]]:
        """Cluster registers in order to optimize modbus traffic."""
        # Normalize key: use sorted unique numeric registers that exist in the map
        key_tuple: tuple[int, ...] = tuple(
            sorted({int(r) for r in registers if r.isnumeric() and r in self._register_map})
        )
        if key_tuple not in self._cluster_cache:
            # Simple cache size guard to avoid unbounded growth in long-running processes
            if len(self._cluster_cache) > 256:
                self._cluster_cache.clear()
            self._cluster_cache[key_tuple] = self._generate_register_clusters(registers=registers)
        return self._cluster_cache[key_tuple]

    def _read_registers(
        self, *, register: str, length: int
    ) -> ReadHoldingRegistersResponse | None:
        """
        Do the actual reading from modbus.

        Args:
            register: Register address as string
            length: Number of registers to read

        Returns:
            ReadHoldingRegistersResponse if successful, None on error

        Raises:
            ModbusReadError: On read errors
            ModbusTimeoutError: On timeout

        """
        try:
            # Use circuit breaker for read operations
            def _do_read() -> ReadHoldingRegistersResponse:
                """Perform the read operation."""
                assert self._modbus_client is not None  # Set during connect()
                result = cast(
                    ReadHoldingRegistersResponse,
                    self._modbus_client.read_holding_registers(
                        address=int(register), count=length, device_id=self._modbus_slave
                    ),
                )
                if result.isError():
                    error_msg = f"Error reading register {register}, length {length}"
                    _LOGGER.error(error_msg)
                    self._error_count += 1
                    if self._health_check:
                        self._health_check.record_failure(name="modbus", error=error_msg)
                    raise ModbusReadError(
                        message=error_msg,
                        address=int(register),
                        slave_id=self._modbus_slave,
                    )
                if len(result.registers) != length:
                    error_msg = (
                        f"Register {register} length mismatch: "
                        f"requested {length}, received {len(result.registers)}"
                    )
                    _LOGGER.error(error_msg)
                    raise ModbusReadError(
                        message=error_msg,
                        address=int(register),
                        slave_id=self._modbus_slave,
                        details={"requested": length, "received": len(result.registers)},
                    )
                return result

            # Call through circuit breaker
            result: ReadHoldingRegistersResponse = cast(
                ReadHoldingRegistersResponse, self._circuit_breaker.call(_do_read)
            )
        except ModbusException as ex:
            error_msg = f"Modbus exception reading register {register}, length {length}: {ex}"
            _LOGGER.error(error_msg)
            self._error_count += 1
            if self._health_check:
                self._health_check.record_failure(name="modbus", error=error_msg)

            # Determine if it's a timeout or read error
            if "timeout" in str(ex).lower():
                raise ModbusTimeoutError(
                    message=error_msg,
                    address=int(register),
                    slave_id=self._modbus_slave,
                ) from ex
            raise ModbusReadError(
                message=error_msg,
                address=int(register),
                slave_id=self._modbus_slave,
            ) from ex
        except (ModbusReadError, ModbusTimeoutError):
            # Re-raise our typed exceptions
            raise
        except Exception as ex:
            error_msg = f"Unexpected error reading register {register}, length {length}: {ex}"
            _LOGGER.exception(error_msg)
            self._error_count += 1
            if self._health_check:
                self._health_check.record_failure(name="modbus", error=error_msg)
            raise ModbusReadError(
                message=error_msg,
                address=int(register),
                slave_id=self._modbus_slave,
            ) from ex
        else:
            if self._health_check:
                self._health_check.record_success(name="modbus")
            return result

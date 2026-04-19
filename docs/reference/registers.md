# Register Reference

Auto-generated from `aiomtec2mqtt/registers.yaml`. Do not edit by hand.
Run `python script/gen_register_reference.py` to regenerate.

Total registers: **94** across **10** groups.

## Group `now-base`

| Register | Name | MQTT | Unit | HA device_class | HA state_class |
| --- | --- | --- | --- | --- | --- |
| `10100` | Inverter date | `inverter_date` | — | — | — |
| `10105` | Inverter status | `inverter_status` | — | enum | — |
| `10112` | Fault flag1 | `fault_flag_1` | — | enum | — |
| `10114` | Fault flag2 | `fault_flag_2` | — | enum | — |
| `11000` | Grid power | `grid_power` | W | power | measurement |
| `11016` | Inverter AC power | `inverter` | W | power | measurement |
| `11028` | PV power | `pv` | W | power | measurement |
| `30230` | Backup power total | `backup` | W | power | measurement |
| `30254` | Battery voltage | `battery_voltage` | V | voltage | measurement |
| `30255` | Battery current | `battery_current` | A | current | measurement |
| `30256` | Battery mode | `battery_mode` | — | enum | — |
| `30258` | Battery power | `battery` | W | power | measurement |
| `33000` | Battery SOC | `battery_soc` | % | battery | measurement |
| `33002` | BMS Status | `bms_status` | — | enum | — |
| `50000` | Inverter operation mode | `mode` | — | enum | — |
| `53509` | BMS Error Code | `bms_error_code` | — | enum | — |
| `53511` | BMS Protection Code | `bms_protection_code` | — | enum | — |
| `53513` | BMS Alarm Code | `bms_alarm_code` | — | enum | — |
| `api-date` | API date | `api_date` | — | — | — |
| `consumption` | Household consumption | `consumption` | W | power | measurement |

## Group `day`

| Register | Name | MQTT | Unit | HA device_class | HA state_class |
| --- | --- | --- | --- | --- | --- |
| `31000` | Grid injection energy (day) | `grid_feed_day` | kWh | energy | total_increasing |
| `31001` | Grid purchased energy (day) | `grid_purchase_day` | kWh | energy | total_increasing |
| `31002` | Backup energy (day) | `backup_day` | kWh | energy | total_increasing |
| `31003` | Battery charge energy (day) | `battery_charge_day` | kWh | energy | total_increasing |
| `31004` | Battery discharge energy (day) | `battery_discharge_day` | kWh | energy | total_increasing |
| `31005` | PV energy generated (day) | `pv_day` | kWh | energy | total_increasing |
| `autarky-day` | Household autarky (day) | `autarky_rate_day` | % | power_factor | measurement |
| `consumption-day` | Household consumption (day) | `consumption_day` | kWh | energy | total_increasing |
| `ownconsumption-day` | Own consumption rate (day) | `own_consumption_day` | % | power_factor | measurement |

## Group `total`

| Register | Name | MQTT | Unit | HA device_class | HA state_class |
| --- | --- | --- | --- | --- | --- |
| `31102` | Grid energy injected (total) | `grid_feed_total` | kWh | energy | total_increasing |
| `31104` | Grid energy purchased (total) | `grid_purchase_total` | kWh | energy | total_increasing |
| `31106` | Backup energy (total) | `backup_total` | kWh | energy | total_increasing |
| `31108` | Battery energy charged (total) | `battery_charge_total` | kWh | energy | total_increasing |
| `31110` | Battery energy discharged (total) | `battery_discharge_total` | kWh | energy | total_increasing |
| `31112` | PV energy generated (total) | `pv_total` | kWh | energy | total_increasing |
| `autarky-total` | Household autarky (total) | `autarky_rate_total` | % | power_factor | measurement |
| `consumption-total` | Household consumption (total) | `consumption_total` | kWh | energy | total_increasing |
| `ownconsumption-total` | Own consumption rate (total) | `own_consumption_total` | % | power_factor | measurement |

## Group `static`

| Register | Name | MQTT | Unit | HA device_class | HA state_class |
| --- | --- | --- | --- | --- | --- |
| `10000` | Inverter serial number | `serial_no` | — | — | — |
| `10008` | Equipment info | `equipment_info` | — | — | — |
| `10011` | Firmware version | `firmware_version` | — | — | — |

## Group `now-grid`

| Register | Name | MQTT | Unit | HA device_class | HA state_class |
| --- | --- | --- | --- | --- | --- |
| `10994` | Grid power phase A | `grid_a` | W | power | measurement |
| `10996` | Grid power phase B | `grid_b` | W | power | measurement |
| `10998` | Grid power phase C | `grid_c` | W | power | measurement |
| `11006` | Inverter AC voltage lines A/B | `ac_voltage_a_b` | V | voltage | measurement |
| `11007` | Inverter AC voltage lines B/C | `ac_voltage_b_c` | V | voltage | measurement |
| `11008` | Inverter AC voltage lines C/A | `ac_voltage_c_a` | V | voltage | measurement |
| `11009` | Inverter AC voltage phase A | `ac_voltage_a` | V | voltage | measurement |
| `11010` | Inverter AC current phase A | `ac_current_a` | A | current | measurement |
| `11011` | Inverter AC voltage phase B | `ac_voltage_b` | V | voltage | measurement |
| `11012` | Inverter AC current phase B | `ac_current_b` | A | current | measurement |
| `11013` | Inverter AC voltage phase C | `ac_voltage_c` | V | voltage | measurement |
| `11014` | Inverter AC current phase C | `ac_current_c` | A | current | measurement |
| `11015` | Grid frequency | `grid_fequency` | Hz | frequency | measurement |

## Group `now-pv`

| Register | Name | MQTT | Unit | HA device_class | HA state_class |
| --- | --- | --- | --- | --- | --- |
| `11022` | PV generation time total | `pv_generation_duration` | h | duration | measurement |
| `11038` | PV1 voltage | `pv_voltage_1` | V | voltage | measurement |
| `11039` | PV1 current | `pv_current_1` | A | current | measurement |
| `11040` | PV2 voltage | `pv_voltage_2` | V | voltage | measurement |
| `11041` | PV2 current | `pv_current_2` | A | current | measurement |
| `11062` | PV1 power | `pv_1` | W | power | measurement |
| `11064` | PV2 power | `pv_2` | W | power | measurement |

## Group `now-inverter`

| Register | Name | MQTT | Unit | HA device_class | HA state_class |
| --- | --- | --- | --- | --- | --- |
| `11032` | Inverter temperature sensor 1 | `inverter_temp1` | °C | temperature | measurement |
| `11033` | Inverter temperature sensor 2 | `inverter_temp2` | °C | temperature | measurement |
| `11034` | Inverter temperature sensor 3 | `inverter_temp3` | °C | temperature | measurement |
| `11035` | Inverter temperature sensor 4 | `inverter_temp4` | °C | temperature | measurement |
| `30236` | Inverter power phase A | `inverter_a` | W | power | measurement |
| `30242` | Inverter power phase B | `inverter_b` | W | power | measurement |
| `30248` | Inverter power phase C | `inverter_c` | W | power | measurement |

## Group `now-backup`

| Register | Name | MQTT | Unit | HA device_class | HA state_class |
| --- | --- | --- | --- | --- | --- |
| `30200` | Backup voltage phase A | `backup_voltage_a` | V | voltage | measurement |
| `30201` | Backup current phase A | `backup_current_a` | A | current | measurement |
| `30202` | Backup frequency phase A | `backup_frequency_a` | Hz | frequency | measurement |
| `30204` | Backup power phase A | `backup_a` | W | power | measurement |
| `30210` | Backup voltage phase B | `backup_voltage_b` | V | voltage | measurement |
| `30211` | Backup current phase B | `backup_current_b` | A | current | measurement |
| `30212` | Backup frequency phase B | `backup_frequency_b` | Hz | frequency | measurement |
| `30214` | Backup power phase B | `backup_b` | W | power | measurement |
| `30220` | Backup voltage phase C | `backup_voltage_c` | V | voltage | measurement |
| `30221` | Backup current phase C | `backup_current_c` | A | current | measurement |
| `30222` | Backup frequency phase C | `backup_frequency_c` | Hz | frequency | measurement |
| `30224` | Backup power phase C | `backup_c` | W | power | measurement |

## Group `now-battery`

| Register | Name | MQTT | Unit | HA device_class | HA state_class |
| --- | --- | --- | --- | --- | --- |
| `33001` | Battery SOH | `battery_soh` | % | power_factor | measurement |
| `33003` | Battery temperature | `battery_temp` | °C | temperature | measurement |
| `33009` | Battery cell temperature max. | `battery_cell_t_max` | °C | temperature | measurement |
| `33011` | Battery cell temperature min. | `battery_cell_t_min` | °C | temperature | measurement |
| `33013` | Battery cell voltage max. | `battery_cell_v_max` | V | voltage | measurement |
| `33015` | Battery cell voltage min. | `battery_cell_v_min` | V | voltage | measurement |

## Group `config`

| Register | Name | MQTT | Unit | HA device_class | HA state_class |
| --- | --- | --- | --- | --- | --- |
| `25100` | Grid injection limit switch | `grid_inject_switch` | — | — | — |
| `25103` | Grid injection power limit | `grid_inject_limit` | % | — | measurement |
| `52502` | On-grid SOC limit switch | `on_grid_soc_switch` | — | — | — |
| `52503` | On-grid SOC limit | `on_grid_soc_limit` | % | — | measurement |
| `52504` | Off-grid SOC limit switch | `off_grid_soc_switch` | — | — | — |
| `52505` | Off-grid SOC limit | `off_grid_soc_limit` | % | — | measurement |
| `52601` | Charge limit | `charge_limit` | A | — | measurement |
| `52603` | Discharge limit | `discharge_limit` | A | — | measurement |

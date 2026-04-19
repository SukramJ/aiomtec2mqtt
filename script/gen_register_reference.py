#!/usr/bin/env python3
"""
Generate Markdown register reference from ``aiomtec2mqtt/registers.yaml``.

Produces a documentation table that lists every register, grouped by its
polling bucket (``now-base``, ``day``, ``total``, etc.). The output is used by
the documentation site (MkDocs) and must stay in sync with the YAML, so this
script is safe to re-run and idempotent.

Usage:
    python script/gen_register_reference.py [--output docs/reference/registers.md]

Exit codes:
    0 on success
    1 on IO failure (YAML missing, cannot write output)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from aiomtec2mqtt.config import init_register_map
from aiomtec2mqtt.const import Register

_DEFAULT_OUTPUT = Path("docs/reference/registers.md")


def _fmt_cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", r"\|").replace("\n", " ")


def _format_group(group_name: str, entries: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines: list[str] = [f"## Group `{group_name}`", ""]
    lines.append("| Register | Name | MQTT | Unit | HA device_class | HA state_class |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key, val in sorted(entries):
        lines.append(
            "| `{key}` | {name} | `{mqtt}` | {unit} | {dc} | {sc} |".format(
                key=_fmt_cell(key),
                name=_fmt_cell(val.get(Register.NAME)),
                mqtt=_fmt_cell(val.get(Register.MQTT)),
                unit=_fmt_cell(val.get("unit")),
                dc=_fmt_cell(val.get(Register.DEVICE_CLASS)),
                sc=_fmt_cell(val.get("hass_state_class")),
            )
        )
    lines.append("")
    return lines


def render() -> str:
    """Render the full Markdown document from the register map."""
    reg_map, groups = init_register_map()

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {g: [] for g in groups}
    for key, val in reg_map.items():
        group = val.get(Register.GROUP)
        if group in grouped:
            grouped[group].append((key, val))

    header = [
        "# Register Reference",
        "",
        "Auto-generated from `aiomtec2mqtt/registers.yaml`. Do not edit by hand.",
        "Run `python script/gen_register_reference.py` to regenerate.",
        "",
        f"Total registers: **{len(reg_map)}** across **{len(groups)}** groups.",
        "",
    ]

    body: list[str] = []
    for group in groups:
        body.extend(_format_group(group, grouped[group]))

    return "\n".join([*header, *body]).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the rendered Markdown to ``--output`` (default ``docs/reference/registers.md``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the on-disk output differs from the rendered content.",
    )
    args = parser.parse_args(argv)

    rendered = render()

    if args.check:
        if not args.output.exists():
            print(f"::error::{args.output} is missing; run without --check to create it.")
            return 1
        existing = args.output.read_text(encoding="utf-8")
        if existing != rendered:
            print(f"::error::{args.output} is out of date. Re-run gen_register_reference.py.")
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

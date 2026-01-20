"""Install systemd service for aiomtec2mqtt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

SERVICE_TEMPLATE = """[Unit]
Description=M-TEC MQTT service
After=multi-user.target

[Service]
Type=simple
User={user}
ExecStart={exec_path}
Restart=always

[Install]
WantedBy=multi-user.target
"""

SERVICE_NAME = "aiomtec2mqtt.service"
SERVICE_PATH = Path("/etc/systemd/system") / SERVICE_NAME


def get_executable_path() -> Path:
    """Get the path to the aiomtec2mqtt executable."""
    # The executable is in the same bin directory as the current Python interpreter
    return Path(sys.executable).parent / "aiomtec2mqtt"


def install_service(*, user: str | None = None) -> None:
    """Install the systemd service."""
    if os.geteuid() != 0:
        sys.exit(1)

    exec_path = get_executable_path()
    if not exec_path.exists():
        sys.exit(1)

    # Get the user who invoked sudo, or fall back to current user
    if user is None:
        user = os.environ.get("SUDO_USER", os.environ.get("USER", "root"))

    service_content = SERVICE_TEMPLATE.format(
        user=user,
        exec_path=exec_path,
    )

    # Write service file
    SERVICE_PATH.write_text(service_content)

    # Reload systemd and enable service
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", SERVICE_NAME], check=True)
    subprocess.run(["systemctl", "start", SERVICE_NAME], check=True)


def uninstall_service() -> None:
    """Uninstall the systemd service."""
    if os.geteuid() != 0:
        sys.exit(1)

    if not SERVICE_PATH.exists():
        sys.exit(1)

    subprocess.run(["systemctl", "stop", SERVICE_NAME], check=False)
    subprocess.run(["systemctl", "disable", SERVICE_NAME], check=False)
    SERVICE_PATH.unlink()
    subprocess.run(["systemctl", "daemon-reload"], check=True)


def main() -> None:
    """Execute the install/uninstall command."""
    parser = argparse.ArgumentParser(
        description="Install or uninstall aiomtec2mqtt systemd service"
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Uninstall the systemd service",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User to run the service as (default: user who invoked sudo)",
    )

    args = parser.parse_args()

    if args.uninstall:
        uninstall_service()
    else:
        install_service(user=args.user)


if __name__ == "__main__":
    main()

"""Install systemd service for aiomtec2mqtt."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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


def install_service(user: str | None = None) -> None:
    """Install the systemd service."""
    if os.geteuid() != 0:
        print("Error: This script requires root privileges.")
        print("Please run with: sudo aiomtec2mqtt-install")
        sys.exit(1)

    exec_path = get_executable_path()
    if not exec_path.exists():
        print(f"Error: Executable not found at {exec_path}")
        print("Make sure aiomtec2mqtt is properly installed.")
        sys.exit(1)

    # Get the user who invoked sudo, or fall back to current user
    if user is None:
        user = os.environ.get("SUDO_USER", os.environ.get("USER", "root"))

    service_content = SERVICE_TEMPLATE.format(
        user=user,
        exec_path=exec_path,
    )

    print(f"Installing systemd service for user '{user}'...")
    print(f"Executable: {exec_path}")

    # Write service file
    SERVICE_PATH.write_text(service_content)
    print(f"Created {SERVICE_PATH}")

    # Reload systemd and enable service
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", SERVICE_NAME], check=True)
    subprocess.run(["systemctl", "start", SERVICE_NAME], check=True)

    print(f"Service '{SERVICE_NAME}' installed and started.")
    print(f"Check status with: systemctl status {SERVICE_NAME}")


def uninstall_service() -> None:
    """Uninstall the systemd service."""
    if os.geteuid() != 0:
        print("Error: This script requires root privileges.")
        print("Please run with: sudo aiomtec2mqtt-install --uninstall")
        sys.exit(1)

    if not SERVICE_PATH.exists():
        print(f"Service file {SERVICE_PATH} does not exist.")
        sys.exit(1)

    print(f"Uninstalling {SERVICE_NAME}...")

    subprocess.run(["systemctl", "stop", SERVICE_NAME], check=False)
    subprocess.run(["systemctl", "disable", SERVICE_NAME], check=False)
    SERVICE_PATH.unlink()
    subprocess.run(["systemctl", "daemon-reload"], check=True)

    print(f"Service '{SERVICE_NAME}' uninstalled.")


def main() -> None:
    """Main entry point."""
    import argparse

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
#!/bin/sh

DIR=$(dirname "$0")
BASE_DIR=$(readlink -f $DIR)

SVC_TXT="
[Unit]
Description=M-TEC MQTT service 
After=multi-user.target

[Service]
Type=simple
User=USER
WorkingDirectory=BASE_DIR
ExecStart=BASE_DIR/venv/bin/aiomtec2mqtt
Restart=always

[Install]
WantedBy=multi-user.target
"

echo "aiomtec2mqtt: Installing systemd service to auto-start aiomtec2mqtt"

if [ $(id -u) != "0" ]; then
  echo "This script required root rights. Please restart using 'sudo'"
else
  echo "$SVC_TXT" | sed "s!BASE_DIR!$BASE_DIR!g" | sed "s/USER/$SUDO_USER/g" > /tmp/aiomtec2mqtt.service
  chmod 666 /tmp/aiomtec2mqtt.service
  mv /tmp/aiomtec2mqtt.service /etc/systemd/system
  systemctl daemon-reload
  systemctl enable aiomtec2mqtt.service
  systemctl start aiomtec2mqtt.service
  echo "==> systemd service '/etc/systemd/system/aiomtec2mqtt.service' installed"
fi


# esp_firmware/mqtt_config.example.py
# Copy to mqtt_config.py and fill in your WiFi credentials.
# Do NOT commit mqtt_config.py with real passwords.

TEAM = "your-team"

WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PASS = "YOUR_WIFI_PASSWORD"

MQTT_BROKER = "broker.benax.rw"
MQTT_PORT = 1883

PAN_MIN_DUTY = 22
PAN_MAX_DUTY = 128
PAN_PIN = 14

PAN_MIN_ANGLE = 10
PAN_MAX_ANGLE = 170
HOME_PAN = 90

T_CMD = b"LaTeam/eye/servo/cmd"
T_STATUS = b"LaTeam/eye/servo/status"

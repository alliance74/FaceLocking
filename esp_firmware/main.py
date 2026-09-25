# esp_firmware/main.py
# Flash to ESP8266 (MicroPython) alongside mqtt_config.py and umqtt/simple.py.
#
# Subscribes to LaTeam/eye/servo/cmd and expects PLAIN TEXT commands
# (matching the team's established protocol, same as the ESP32 Arduino
# version):
#   ANGLE:95
#   STOP
#   HOME
#
# Moves the servo smoothly toward the latest target angle rather than
# jumping, and only acts on the newest message if several arrive faster
# than the servo can physically move (important for continuous tracking,
# which publishes far more often than a simple scan-and-hold design would).

import time
import network
import machine
import ubinascii
import ujson
from umqtt.simple import MQTTClient

from mqtt_config import (
    WIFI_SSID, WIFI_PASS, MQTT_BROKER, MQTT_PORT,
    PAN_MIN_DUTY, PAN_MAX_DUTY,
    PAN_PIN,
    PAN_MIN_ANGLE, PAN_MAX_ANGLE,
    HOME_PAN,
    T_CMD, T_STATUS,
)

CLIENT_ID = b"falcon_eye_" + ubinascii.hexlify(machine.unique_id())

led = machine.Pin(2, machine.Pin.OUT)
led.value(1)

pan_pwm = machine.PWM(machine.Pin(PAN_PIN), freq=50)

client = None

# current + target angle (smoothing state)
current_pan = HOME_PAN
target_pan = HOME_PAN

# how many degrees to step per loop iteration toward the target -- tune for smoothness
STEP_DEG = 3


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def angle_to_duty(angle, min_angle, max_angle, min_duty, max_duty):
    angle = clamp(angle, min_angle, max_angle)
    span_a = max_angle - min_angle
    span_d = max_duty - min_duty
    return int(min_duty + (angle - min_angle) * (span_d / span_a))


def apply_servo_angle(pan_angle):
    pan_pwm.duty(angle_to_duty(pan_angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE, PAN_MIN_DUTY, PAN_MAX_DUTY))


def wifi_connect():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        print("WiFi OK:", wlan.ifconfig()[0])
        return True
    print("Connecting WiFi...")
    wlan.connect(WIFI_SSID, WIFI_PASS)
    for _ in range(30):
        if wlan.isconnected():
            print("WiFi OK:", wlan.ifconfig()[0])
            led.value(1)
            return True
        led.value(not led.value())
        time.sleep(0.5)
    print("WiFi FAILED")
    return False


def mqtt_callback(topic, msg):
    global target_pan
    try:
        command = msg.decode().strip().upper()
    except Exception as e:
        print("Bad payload:", msg, e)
        return

    if command == "HOME":
        target_pan = HOME_PAN
        print("HOME")
    elif command == "STOP":
        target_pan = current_pan  # hold exactly where it currently is
        print("STOP / HOLDING:", current_pan)
    elif command.startswith("ANGLE:"):
        try:
            angle = float(command[len("ANGLE:"):])
            target_pan = clamp(angle, PAN_MIN_ANGLE, PAN_MAX_ANGLE)
        except ValueError:
            print("Invalid angle command:", command)
    else:
        print("Unknown command:", command)


def publish_status(status="MOVING"):
    try:
        client.publish(T_STATUS, ujson.dumps({"status": status, "angle": current_pan}))
    except Exception as e:
        print("Status publish failed:", e)


def step_toward(current, target, step):
    if current < target:
        return min(current + step, target)
    if current > target:
        return max(current - step, target)
    return current


def main():
    global client, current_pan

    print("\n" + "=" * 40)
    print("Falcon Eye - Servo Controller (ESP8266, pan only)")
    print("=" * 40 + "\n")

    if not wifi_connect():
        print("WiFi FAIL - Restarting...")
        time.sleep(5)
        machine.reset()

    apply_servo_angle(current_pan)

    time.sleep(1.5)  # let WiFi/network fully settle before opening the MQTT socket

    mqtt_connected = False
    for attempt in range(5):
        try:
            client = MQTTClient(CLIENT_ID, MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.set_callback(mqtt_callback)
            client.connect()
            client.subscribe(T_CMD)
            print("MQTT OK, subscribed to", T_CMD)
            mqtt_connected = True
            break
        except Exception as e:
            print(f"MQTT connect attempt {attempt + 1}/5 failed:", e)
            time.sleep(2)

    if not mqtt_connected:
        print("MQTT FAIL after 5 attempts - Restarting...")
        time.sleep(5)
        machine.reset()

    print("\nReady - listening for pan commands\n")

    last_status_pub = 0

    while True:
        try:
            client.check_msg()

            current_pan = step_toward(current_pan, target_pan, STEP_DEG)
            apply_servo_angle(current_pan)

            now = time.time()
            if now - last_status_pub > 1:
                publish_status()
                last_status_pub = now

            time.sleep(0.03)  # ~30 steps/sec smoothing rate

        except OSError:
            print("Reconnecting MQTT...")
            try:
                client = MQTTClient(CLIENT_ID, MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.set_callback(mqtt_callback)
                client.connect()
                client.subscribe(T_CMD)
            except Exception:
                time.sleep(5)
                machine.reset()
        except Exception as e:
            print("Error:", e)
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print("Fatal:", e)
        time.sleep(5)
        machine.reset()
# esp_firmware/calibrate.py
# Run this on the ESP8266 FIRST, before main.py, to find the correct
# min/max duty values for your specific servo (they vary by model).
#
# Your wiring: servo signal wire -> D5 (GPIO14), VIN -> 5V/VIN, GND -> GND.
#
# Watch the physical servo and the printed duty value at each step; note
# where it reaches its true 0 deg and 180 deg limits (or starts
# buzzing/straining -- back off from that point, don't force it).

import machine
import time

SERVO_PIN = 14  # D5 on NodeMCU/Wemos D1 mini boards = GPIO14

servo_pwm = machine.PWM(machine.Pin(SERVO_PIN), freq=50)

print("Servo calibration starting. Ctrl+C to stop early.")
print("Watch the servo and note the duty value at each physical limit.\n")

for duty in range(20, 140, 5):
    servo_pwm.duty(duty)
    print("duty =", duty)
    time.sleep(0.6)

print("\nDone. Note which duty value corresponded to 0 deg (min) and")
print("180 deg (max) on the physical servo, then update PAN_MIN_DUTY")
print("and PAN_MAX_DUTY in mqtt_config.py with those values.")
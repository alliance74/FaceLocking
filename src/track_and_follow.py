# src/track_and_follow.py
"""
Phase 6+7: turn the face-center offset into a servo angle correction and
publish it over MQTT, live, so the turret actually follows your face
left/right in real time.

Tuning knobs are at the top -- adjust GAIN and DEAD_ZONE_PX first if the
tracking feels too twitchy, too sluggish, or moves the wrong direction.

Run:
    python -m src.track_and_follow

Keys:
    q : quit
"""
import time

import cv2
import paho.mqtt.client as mqtt

from .haar_5pt import Haar5ptDetector

# -------------------------
# MQTT
# -------------------------
MQTT_BROKER = "broker.benax.rw"
MQTT_PORT = 1883
T_CMD = "LaTeam/eye/servo/cmd"
T_STATUS = "LaTeam/eye/servo/status"

# -------------------------
# Servo / control tuning
# -------------------------
PAN_MIN_ANGLE = 10
PAN_MAX_ANGLE = 170
HOME_PAN = 90

DEAD_ZONE_PX = 15      # ignore offsets smaller than this (prevents jitter when ~centered)
GAIN = 0.04             # degrees of correction per pixel of offset -- tune this first
PUBLISH_MIN_DELTA = 1.0  # only publish if target angle changed by at least this many degrees
MAX_STEP_PER_FRAME = 1.5  # caps how much target_pan can change in one frame -- smooths first-catch snap

# If the servo turns the WRONG way (moves away from your face instead of
# toward it), flip this to -1 rather than rewriting the math.
PAN_DIRECTION = 1

LOST_FACE_HOLD_FRAMES = 30  # after this many consecutive frames with no face, stop adjusting


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[track_and_follow] MQTT connected (rc={rc})")
    client.subscribe(T_STATUS)


def on_message(client, userdata, msg):
    # optional: could parse and display board-reported angle; kept quiet for now
    pass


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Tracking camera (index 0) not opened.")

    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()
    time.sleep(1.0)

    target_pan = HOME_PAN
    last_published_pan = None
    lost_face_count = 0

    print("Tracking + following (index 0). Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            H, W = frame.shape[:2]
            frame_cx = W // 2
            vis = frame.copy()
            cv2.line(vis, (frame_cx, 0), (frame_cx, H), (255, 0, 0), 1)

            faces = det.detect(frame, max_faces=1)

            if faces:
                lost_face_count = 0
                f = faces[0]
                face_cx = (f.x1 + f.x2) // 2
                face_cy = (f.y1 + f.y2) // 2
                offset_x = face_cx - frame_cx

                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                cv2.circle(vis, (face_cx, face_cy), 5, (0, 0, 255), -1)

                if abs(offset_x) > DEAD_ZONE_PX:
                    raw_delta = PAN_DIRECTION * offset_x * GAIN
                    delta = clamp(raw_delta, -MAX_STEP_PER_FRAME, MAX_STEP_PER_FRAME)
                    target_pan = clamp(target_pan + delta, PAN_MIN_ANGLE, PAN_MAX_ANGLE)

                cv2.putText(vis, f"offset_x: {offset_x:+d}px  target_pan: {target_pan:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                lost_face_count += 1
                status = "no face" if lost_face_count < LOST_FACE_HOLD_FRAMES else "no face (holding)"
                cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # target_pan intentionally left unchanged -- holds last known position

            # only publish when the target actually moved meaningfully -- avoids flooding
            if last_published_pan is None or abs(target_pan - last_published_pan) >= PUBLISH_MIN_DELTA:
                client.publish(T_CMD, f"ANGLE:{int(round(target_pan))}")
                last_published_pan = target_pan

            cv2.imshow("Track and Follow (index 0)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
# Falcon Eye — Face Recognition + Servo-Tracking Turret

A CPU-only face recognition pipeline (Haar detection -> 5-point MediaPipe landmarks ->
ArcFace ONNX embedding -> enrollment/recognition) combined with an ESP8266-driven
servo turret, MQTT communication, live face tracking, and identity-locked
expression/blink detection.

```
Camera -> Haar detection -> 5-point landmarks -> alignment (112x112)
       -> ArcFace ONNX embedding -> face database -> Known / Stranger
       -> (identity lock) -> expression + blink analysis
       -> (tracking, optional) -> MQTT -> ESP8266 -> servo
```

---

## Repository Layout

```
face-recognition-5pt/
|-- data/
|   |-- enroll/<name>/          # aligned enrollment crops per identity
|   |-- debug_aligned/          # alignment debug snapshots
|   `-- db/                     # face_db.npz / face_db.json
|-- models/
|   |-- embedder_arcface.onnx   # ~167 MB -- downloaded, not committed
|   `-- haarcascade_frontalface_default.xml  # only needed if cv2.data.haarcascades is empty
|-- esp_firmware/
|   |-- mqtt_config.py          # WiFi/MQTT/servo config -- has real credentials, gitignore this
|   |-- main.py                 # ESP8266 MicroPython firmware (flash this to the board)
|   `-- calibrate.py            # one-off servo duty-cycle calibration script
|-- src/
|   |-- camera.py                 # webcam smoke test (laptop cam, index 1)
|   |-- detect.py                 # Haar detection test
|   |-- landmarks.py              # 5-point landmark test
|   |-- align.py                  # alignment test (112x112 warp)
|   |-- embed.py                  # ArcFace ONNX embedding test
|   |-- enroll.py                 # multi-identity enrollment tool
|   |-- evaluate.py               # threshold tuning (FAR/FRR sweep)
|   |-- recognize.py              # full recognition demo (laptop cam)
|   |-- haar_5pt.py               # Haar + FaceMesh 5-point detector, alignment math
|   |-- track_camera_check.py     # webcam smoke test (turret cam, index 0)
|   |-- face_center_track.py      # visualizes face-center offset (no servo)
|   |-- servo_link.py             # standalone MQTT servo command test
|   |-- track_and_follow.py       # live face tracking + servo control (any face)
|   |-- expression.py             # smile/neutral/sad + blink detection module
|   |-- calibrate_expression.py   # tune expression thresholds against your own face
|   `-- identity_lock_track.py    # recognition + identity lock + expression (servo off)
`-- init_project.py
```

---

## Setup

### 1. Virtual environment
```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\Activate.ps1       # PowerShell
```

### 2. Dependencies
```bash
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install "opencv-python==4.10.0.84" numpy onnxruntime scipy tqdm "mediapipe==0.10.21" paho-mqtt
```
> Pin these exact versions. Newer `opencv-python` (5.x) prerelease builds don't bundle the
> Haar cascade XML files, and `mediapipe` 1.x removed the `mp.solutions.face_mesh` API this
> project depends on.

### 3. Download the ArcFace ONNX model
```bash
curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
unzip -o buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
```

### 4. Haar cascade (only if missing)
Check first:
```bash
ls "$(.venv/Scripts/python.exe -c "import cv2; print(cv2.data.haarcascades)")"
```
If `haarcascade_frontalface_default.xml` isn't listed, the opencv-python pin above (step 2)
already fixes this -- reinstall opencv-python at the pinned version.

### 5. Cameras
- **Index 1** -- laptop webcam, used for `enroll.py` / `recognize.py`.
- **Index 0** -- turret-mounted camera, used for tracking / identity lock scripts.
- All camera opens use `cv2.CAP_DSHOW` explicitly (Windows MSMF backend returns black frames
  on this hardware otherwise).

---

## Hardware -- ESP8266 Servo Turret

- **Wiring**: servo signal -> D5 (GPIO14), VIN -> 5V, GND -> GND.
- **Board**: ESP8266, MicroPython firmware.
- **Broker**: `broker.benax.rw:1883` (shared team broker).
- **Protocol**: plain text commands on `LaTeam/eye/servo/cmd` -- `ANGLE:N`, `STOP`, `HOME`.
  Status published back on `LaTeam/eye/servo/status`.

### Servo calibration (one-time, per servo)
```bash
mpremote connect COM6 run esp_firmware/calibrate.py
```
Watch the physical servo through the full duty sweep (20->135), note the duty values where it
hits its true min/max rotation, and update `PAN_MIN_DUTY` / `PAN_MAX_DUTY` in
`esp_firmware/mqtt_config.py` accordingly (small safety margin recommended -- don't use the
exact strain point).

### Flash firmware
```bash
mpremote connect COM6 fs cp esp_firmware/mqtt_config.py :mqtt_config.py
mpremote connect COM6 fs cp esp_firmware/main.py :main.py
mpremote connect COM6 reset
```
Watch it boot:
```bash
mpremote connect COM6
```
Expect: `WiFi OK: <ip>` -> `MQTT OK, subscribed to b'LaTeam/eye/servo/cmd'` -> `Ready - listening for pan commands`.
Detach without resetting: `Ctrl+]`.

> **Known quirk**: `mpremote`'s REPL viewer can crash with a `UnicodeDecodeError` when the
> ESP8266's WiFi radio activity causes brief UART noise during boot. This is a display bug in
> `mpremote`, not a firmware problem -- the board keeps running fine. Just reconnect, or verify
> the board's alive via `servo_link.py`'s status output instead.

---

## Recognition Pipeline -- Validation Order

Run each in order; each depends on the previous stage working.

```bash
.venv/Scripts/python.exe -m src.camera        # validate laptop webcam
.venv/Scripts/python.exe -m src.detect        # validate Haar face detection
.venv/Scripts/python.exe -m src.landmarks     # validate 5-point landmarks
.venv/Scripts/python.exe -m src.align         # validate alignment (112x112 warp)
.venv/Scripts/python.exe -m src.embed         # validate ArcFace embedding (needs the ONNX model)
```

## Enrollment

```bash
.venv/Scripts/python.exe -m src.enroll
```
- Prompts for a name, opens the camera.
- **SPACE** = capture one sample, **A** = auto-capture toggle, **S** = save to database, **R** = reset new samples, **Q** = quit.
- Enroll at least 2 identities for evaluation to mean anything.
- Prefer manual SPACE captures over auto-mode for cleaner samples -- good even lighting,
  small deliberate head variation (not wide pivots), face fully inside the box each capture.

## Threshold Evaluation

```bash
.venv/Scripts/python.exe -m src.evaluate
```
Prints genuine vs. impostor distance distributions and suggests a threshold. Look for genuine
`p95` sitting clearly below impostor `p05` -- that gap is what makes a threshold reliable.

## Recognition Demo

```bash
.venv/Scripts/python.exe -m src.recognize
```
Labels known enrolled faces by name (green) or "Unknown" (red), live, multi-face capable.
`+`/`-` adjust threshold live, `r` reloads DB, `d` toggles debug overlay.

---

## Tracking (any face, servo moves)

```bash
.venv/Scripts/python.exe -m src.track_camera_check   # confirm turret camera (index 0) opens
.venv/Scripts/python.exe -m src.face_center_track     # visualize offset, no servo movement
.venv/Scripts/python.exe -m src.servo_link            # standalone MQTT/servo sanity test
.venv/Scripts/python.exe -m src.track_and_follow       # live: any detected face moves the servo
```
Tuning knobs live at the top of `track_and_follow.py`: `GAIN`, `DEAD_ZONE_PX`,
`MAX_STEP_PER_FRAME`, `PAN_DIRECTION` (flip to `-1` if the servo turns the wrong way).

---

## Identity Lock (registered person only, expression + blinks, servo off)

```bash
.venv/Scripts/python.exe -m src.calibrate_expression   # tune Sad/Smiling thresholds first
.venv/Scripts/python.exe -m src.identity_lock_track     # the actual demo script
```
- Only an **enrolled/registered** person is "locked" -- shows their name, expression
  (Smiling / Neutral / Sad), and a running blink count.
- A **stranger** is detected and boxed in red, labeled "Stranger" -- no expression analysis
  runs on them, and they are not locked.
- Servo movement is intentionally **disabled** in this script (`ENABLE_SERVO = False` at the
  top) -- this is a visual-lock-only demo for now. Re-enabling motion means porting the control
  loop from `track_and_follow.py` back in.
- Recognition distance for each frame prints to the **terminal** (not the on-screen overlay,
  which is kept clean for the demo) -- useful for debugging if recognition misses.

> **Camera consistency matters**: enrollment (`enroll.py`) runs on the laptop webcam
> (index 1); `identity_lock_track.py` runs on the turret camera (index 0). Different cameras
> can shift embedding distances enough to affect matching. If recognition seems unreliable in
> `identity_lock_track.py`, check the printed `dist=` values in the terminal -- if they're
> hovering just above the threshold, either raise `DIST_THRESH` in that file, or re-enroll
> using the turret camera (temporarily point `enroll.py` at index 0) so enrollment and
> recognition use the same camera.

---

## Full Presentation-Day Command Sequence

```bash
cd "/d/NOTES/year3/ROBOTICS/TERM 1/face-recognition-5pt"
source .venv/Scripts/activate

# 1. Face recognition
.venv/Scripts/python.exe -m src.recognize

# 2. Face tracking (any face, servo moves)
.venv/Scripts/python.exe -m src.track_and_follow

# 3. Identity lock (registered person only, expression + blinks, servo off)
.venv/Scripts/python.exe -m src.identity_lock_track
```

---

## Team Notes

- MQTT protocol is **plain text** (`ANGLE:N`, `STOP`, `HOME`), matching the shared team
  convention used by the ESP32 Arduino firmware variant -- not JSON.
- `esp_firmware/mqtt_config.py` contains real WiFi credentials -- gitignore this file, don't
  commit it. Provide a `mqtt_config.example.py` template for teammates instead.
- `data/enroll/`, `data/debug_aligned/`, and `data/db/` contain real people's face data --
  gitignore these, don't publish them even for assignment documentation purposes.
- `models/embedder_arcface.onnx` (~167 MB) exceeds GitHub's 100 MB file limit -- gitignore it,
  document the download steps instead (see Setup above).

---

## References
- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019.
- InsightFace Project -- 2D & 3D Face Analysis.
- ONNX / ONNX Runtime documentation.
- Lugaresi, C., Tang, J., Nash, H., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines.*

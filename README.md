# Face Locking (Falcon Eye)

CPU-only face recognition with identity lock, expression/blink detection, multi-face handling, and optional ESP8266 servo tracking over MQTT.

```
Camera → Haar detect → MediaPipe 5-point landmarks → 112×112 align
      → ArcFace ONNX embed → face DB → Known / Unknown
      → expression (smile, laugh, frown, blink, …) + coaching messages
      → (optional) MQTT → ESP8266 → pan servo
```

## Features

- Enroll and recognize people by face (ArcFace embeddings)
- **Identity lock** on enrolled faces; other people labeled **UNKNOWN**
- Expressions: smiling, laughing, frowning, sad, grimacing, surprised
- Blink / wink counting with adaptive eye-aspect ratio
- Nose tip offset and direction vs frame center
- Orange → red warning when no face is in frame
- Single **menu launcher** so you don’t run modules one-by-one
- Optional ESP8266 pan turret via MQTT

## Requirements

- Windows (OpenCV uses `CAP_DSHOW`)
- **Python 3.12** (MediaPipe 0.10.21 needs it)
- Webcam / external HD USB camera
- ~170 MB disk for the ArcFace ONNX model (not in git)

## Quick start

### 1. Clone and venv

```powershell
git clone https://github.com/BYIRINGIRO-Samuel/FaceLocking.git
cd FaceLocking
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install packages

Pin these versions (newer OpenCV / MediaPipe break this project):

```powershell
python -m pip install --upgrade pip
python -m pip install "opencv-python==4.10.0.84" "numpy<2" onnxruntime scipy tqdm "mediapipe==0.10.21" paho-mqtt
```

### 3. Download ArcFace model

```powershell
mkdir models
curl.exe -L -o models\embedder_arcface.onnx "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
```

Expected size ~166–174 MB.

### 4. Camera index

On many PCs the working USB cam is **index 2** (0/1 can be black or virtual devices). Find yours:

```powershell
python -m src.camera
```

Or force an index:

```powershell
python -m src.camera --cam 2
```

### 5. Launch the app (GUI)

```powershell
python -m src
```

Flow:
1. **Welcome screen** (animated, black / gold / white · Poppins) — click or press Enter  
2. **Live camera** fills the window  
3. Use the **bottom action bar**: LIVE · ENROLL · RECOGNIZE · FULL DEMO · QUIT  

**ENROLL:** type the name in a dialog → **CAPTURE** / **SAVE** appear above the bar (mouse clicks, not keyboard).  

Data is stored in `data/db/` and `data/enroll/<Name>/`.

## Enrollment tips

1. GUI → **Enroll Face** → type the name → Start Enrollment  
2. Face the camera in good light  
3. Click **CAPTURE** several times (~10–15), then click **SAVE**  
4. Click **QUIT** (or press Q) to return to the GUI  

Database files live under `data/db/` (local; face crops under `data/enroll/` are gitignored).

## Full demo checklist (GUI → **Full Demo**)

| Check | What you should see |
|-------|---------------------|
| External HD cam | Live feed (request 1280×720) |
| Your face | Green **LOCKED: YourName** |
| Other people | Red **UNKNOWN** (do not steal the lock) |
| Nose vs center | Yellow nose dot, line to center, dx/dy/direction |
| Smile / laugh / frown / blink | Labels + bottom message banner |
| Leave the frame | Orange then red **face missing** warning |

## ESP8266 servo (optional)

1. Copy config and edit WiFi:

```powershell
copy esp_firmware\mqtt_config.example.py esp_firmware\mqtt_config.py
```

2. Flash with `mpremote` (example COM port):

```powershell
mpremote connect COMx fs cp esp_firmware/mqtt_config.py :mqtt_config.py
mpremote connect COMx fs cp esp_firmware/main.py :main.py
mpremote connect COMx reset
```

3. PC side: menu `6` (MQTT angle test) or `7` (face follow).

Topics (plain text): `LaTeam/eye/servo/cmd` ← `ANGLE:N` / `STOP` / `HOME`.

## Project layout

```
face-locking/
├── src/
│   ├── gui_app.py / menu.py    # decorative GUI launcher
│   ├── enroll.py / recognize.py
│   ├── identity_lock_track.py  # full demo
│   ├── expression.py
│   ├── haar_5pt.py / embed.py
│   └── …
├── esp_firmware/
│   ├── main.py
│   ├── mqtt_config.example.py  # template (real mqtt_config.py is gitignored)
│   └── calibrate.py
├── data/db/                    # face database (local)
├── models/                     # embedder_arcface.onnx (download, gitignored)
└── README.md
```

## Notes

- Prefer **Python 3.12**. 3.13/3.14 often lack matching MediaPipe wheels.
- Same camera for enroll + demo improves match scores (`--cam 2` everywhere).
- Do not commit WiFi passwords or personal face data.

## License / credit

Built on the Falcon Eye / face-recognition pipeline (Haar + MediaPipe + InsightFace ArcFace ONNX). ArcFace: Deng et al., CVPR 2019.

# src/enroll.py
"""
Enrollment tool: camera → Haar → FaceMesh 5pt → align → ArcFace → face DB.

UI controls (mouse click on on-screen buttons):
    CAPTURE · SAVE · AUTO · RESET · QUIT
Keyboard shortcuts still work as fallback (SPACE / s / a / r / q).

Run:
    python -m src.enroll --cam 2 --name Alice
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX
from .haar_5pt import Haar5ptDetector, align_face_5pt


@dataclass
class EnrollConfig:
    out_db_npz: Path = Path("data/db/face_db.npz")
    out_db_json: Path = Path("data/db/face_db.json")
    save_crops: bool = True
    crops_dir: Path = Path("data/enroll")
    samples_needed: int = 15
    auto_capture_every_s: float = 0.25
    max_existing_crops: int = 300
    window_main: str = "enroll"
    window_aligned: str = "aligned_112"


def ensure_dirs(cfg: EnrollConfig) -> None:
    cfg.out_db_npz.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_db_json.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_crops:
        cfg.crops_dir.mkdir(parents=True, exist_ok=True)


def load_db(cfg: EnrollConfig) -> Dict[str, np.ndarray]:
    if cfg.out_db_npz.exists():
        data = np.load(cfg.out_db_npz, allow_pickle=True)
        return {k: data[k].astype(np.float32) for k in data.files}
    return {}


def save_db(cfg: EnrollConfig, db: Dict[str, np.ndarray], meta: dict) -> None:
    ensure_dirs(cfg)
    np.savez(cfg.out_db_npz, **{k: v.astype(np.float32) for k, v in db.items()})
    cfg.out_db_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    E = np.stack([e.reshape(-1) for e in embeddings], axis=0).astype(np.float32)
    m = E.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-12)
    return m.astype(np.float32)


def _list_existing_crops(person_dir: Path, max_count: int) -> List[Path]:
    if not person_dir.exists():
        return []
    files = sorted([p for p in person_dir.glob("*.jpg") if p.is_file()])
    if len(files) > max_count:
        files = files[-max_count:]
    return files


def load_existing_samples_from_crops(
    cfg: EnrollConfig,
    emb: ArcFaceEmbedderONNX,
    person_dir: Path,
) -> List[np.ndarray]:
    if not cfg.save_crops:
        return []
    crops = _list_existing_crops(person_dir, cfg.max_existing_crops)
    base: List[np.ndarray] = []
    for p in crops:
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            r = emb.embed(img)
            base.append(r.embedding)
        except Exception:
            continue
    return base


# -------------------------
# On-screen button bar
# -------------------------
Button = Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int]]  # id, (x1,y1,x2,y2), color


def build_buttons(frame_w: int, frame_h: int, auto_on: bool) -> List[Button]:
    bar_h = 64
    y1 = frame_h - bar_h
    y2 = frame_h - 8
    labels = [
        ("CAPTURE", (40, 180, 80)),
        ("SAVE", (0, 165, 255)),
        ("AUTO", (0, 200, 255) if auto_on else (90, 90, 90)),
        ("RESET", (80, 80, 200)),
        ("QUIT", (60, 60, 220)),
    ]
    n = len(labels)
    gap = 10
    margin = 12
    usable = frame_w - 2 * margin - gap * (n - 1)
    bw = max(70, usable // n)
    buttons: List[Button] = []
    x = margin
    for bid, color in labels:
        buttons.append((bid, (x, y1, x + bw, y2), color))
        x += bw + gap
    return buttons


def draw_button_bar(img: np.ndarray, buttons: List[Button], flash: Optional[str] = None) -> None:
    H, W = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, H - 72), (W, H), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    for bid, (x1, y1, x2, y2), color in buttons:
        fill = color
        if flash == bid:
            fill = (255, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        text = "AUTO ON" if bid == "AUTO" and color[1] > 150 else bid
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        tx = x1 + (x2 - x1 - ts[0]) // 2
        ty = y1 + (y2 - y1 + ts[1]) // 2
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)


def hit_test(buttons: List[Button], x: int, y: int) -> Optional[str]:
    for bid, (x1, y1, x2, y2), _ in buttons:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return bid
    return None


def draw_status(
    frame: np.ndarray,
    name: str,
    base_count: int,
    new_count: int,
    needed: int,
    auto: bool,
    msg: str = "",
) -> None:
    total = base_count + new_count
    lines = [
        f"ENROLL: {name}",
        f"Existing: {base_count}  |  New: {new_count}  |  Total: {total} / {needed}",
        f"Auto: {'ON' if auto else 'OFF'}   —  use the buttons below (or keys)",
    ]
    if msg:
        lines.insert(0, msg)

    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=2)
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    cfg = EnrollConfig()
    ensure_dirs(cfg)

    name = (args.name or "").strip()
    if not name:
        try:
            name = input("Enter person name to enroll (e.g., Alice): ").strip()
        except EOFError:
            name = ""
    if not name:
        print("No name provided. Exiting.")
        return

    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)
    emb = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112), debug=False)

    db = load_db(cfg)
    person_dir = cfg.crops_dir / name
    if cfg.save_crops:
        person_dir.mkdir(parents=True, exist_ok=True)

    base_samples: List[np.ndarray] = load_existing_samples_from_crops(cfg, emb, person_dir)
    new_samples: List[np.ndarray] = []

    status_msg = f"Loaded {len(base_samples)} existing samples." if base_samples else "Click CAPTURE when your face is green-boxed."
    auto = False
    last_auto = 0.0
    flash_btn: Optional[str] = None
    flash_until = 0.0
    pending_action: Optional[str] = None
    latest_aligned: Optional[np.ndarray] = None
    buttons: List[Button] = []

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {args.cam}.")

    cv2.namedWindow(cfg.window_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(cfg.window_aligned, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.window_aligned, 240, 240)
    sized = False

    def on_mouse(event, x, y, _flags, _userdata):
        nonlocal pending_action
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = hit_test(buttons, x, y)
            if hit:
                pending_action = hit

    cv2.setMouseCallback(cfg.window_main, on_mouse)

    def do_capture():
        nonlocal status_msg, flash_btn, flash_until
        if latest_aligned is None:
            status_msg = "No face detected — center your face first."
            return
        r = emb.embed(latest_aligned)
        new_samples.append(r.embedding)
        status_msg = f"Captured NEW ({len(new_samples)})"
        if cfg.save_crops:
            fn = person_dir / f"{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(fn), latest_aligned)
        flash_btn = "CAPTURE"
        flash_until = time.time() + 0.2

    def do_save():
        nonlocal status_msg, base_samples, flash_btn, flash_until, db
        total = len(base_samples) + len(new_samples)
        if total < max(3, cfg.samples_needed // 2):
            status_msg = f"Need more samples (have {total}). Keep capturing."
            return
        all_samples = base_samples + new_samples
        template = mean_embedding(all_samples)
        db[name] = template
        meta = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_dim": int(template.size),
            "names": sorted(db.keys()),
            "samples_existing_used": int(len(base_samples)),
            "samples_new_used": int(len(new_samples)),
            "samples_total_used": int(len(all_samples)),
            "note": "Embeddings are L2-normalized vectors. Matching uses cosine similarity.",
        }
        save_db(cfg, db, meta)
        status_msg = f"Saved '{name}' to DB. Identities: {len(db)}"
        print(status_msg)
        base_samples = load_existing_samples_from_crops(cfg, emb, person_dir)
        new_samples.clear()
        flash_btn = "SAVE"
        flash_until = time.time() + 0.25

    print(f"Enrollment UI started (cam {args.cam}). Click on-screen buttons.")

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    running = True

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                break

            if not sized:
                h, w = frame.shape[:2]
                cv2.resizeWindow(cfg.window_main, w, h)
                sized = True

            vis = frame.copy()
            faces = det.detect(frame, max_faces=1)

            latest_aligned = None
            if faces:
                f = faces[0]
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                latest_aligned = aligned
                cv2.imshow(cfg.window_aligned, aligned)
            else:
                cv2.imshow(cfg.window_aligned, np.zeros((112, 112, 3), dtype=np.uint8))

            now = time.time()
            if auto and latest_aligned is not None and (now - last_auto) >= cfg.auto_capture_every_s:
                r = emb.embed(latest_aligned)
                new_samples.append(r.embedding)
                last_auto = now
                status_msg = f"Auto captured NEW ({len(new_samples)})"
                if cfg.save_crops:
                    fn = person_dir / f"{int(now * 1000)}.jpg"
                    cv2.imwrite(str(fn), latest_aligned)

            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            draw_status(
                vis,
                name=name,
                base_count=len(base_samples),
                new_count=len(new_samples),
                needed=cfg.samples_needed,
                auto=auto,
                msg=status_msg,
            )
            if fps is not None:
                cv2.putText(
                    vis,
                    f"FPS: {fps:.1f}",
                    (10, vis.shape[0] - 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            H, W = vis.shape[:2]
            buttons = build_buttons(W, H, auto_on=auto)
            draw_button_bar(vis, buttons, flash=flash_btn if now < flash_until else None)

            cv2.imshow(cfg.window_main, vis)

            # Handle UI button clicks
            action = pending_action
            pending_action = None
            if action == "CAPTURE":
                do_capture()
            elif action == "SAVE":
                do_save()
            elif action == "AUTO":
                auto = not auto
                status_msg = f"Auto mode {'ON' if auto else 'OFF'}"
                flash_btn = "AUTO"
                flash_until = time.time() + 0.2
            elif action == "RESET":
                new_samples.clear()
                status_msg = "NEW samples reset (existing kept)."
                flash_btn = "RESET"
                flash_until = time.time() + 0.2
            elif action == "QUIT":
                running = False

            # Keyboard fallback
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                running = False
            elif key == ord("a"):
                auto = not auto
                status_msg = f"Auto mode {'ON' if auto else 'OFF'}"
            elif key == ord("r"):
                new_samples.clear()
                status_msg = "NEW samples reset (existing kept)."
            elif key == ord(" "):
                do_capture()
            elif key == ord("s"):
                do_save()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

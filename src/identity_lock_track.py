# src/identity_lock_track.py
"""
Demo-ready Identity Lock with multi-face handling, expression coaching,
nose-to-center tracking, and missing-face warning.

Checklist coverage:
  - External HD camera (--cam 2, requests 1280x720)
  - Enrolled face locked; other faces labeled UNKNOWN (not confused with lock)
  - Nose tip offset + direction vs frame center
  - Smile / frown / sad / grimace / surprised / blink / wink
  - Orange/red warning when no face in frame

Run:
    python -m src.identity_lock_track --cam 2

Keys:
    q   quit
    r   reload DB
    +/- threshold
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .expression import ExpressionAnalyzer
from .haar_5pt import align_face_5pt
from .recognize import ArcFaceEmbedderONNX, FaceDBMatcher, HaarFaceMesh5pt, load_db_npz

DB_PATH = Path("data/db/face_db.npz")
DEFAULT_DIST_THRESH = 0.40
MAX_FACES = 8
LOOK_AWAY_THRESH = 0.22


def draw_label(img, text, pos, scale=0.65, color=(0, 255, 0), thickness=2):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_banner(img, text, color=(0, 220, 255), warn: bool = False):
    H, W = img.shape[:2]
    bar_h = 58
    y0 = H - bar_h
    overlay = img.copy()
    bg = (0, 60, 200) if warn else (20, 20, 20)  # orange-ish BGR when warn
    if warn and "missing" in text.lower():
        bg = (0, 0, 180)  # red-ish
    cv2.rectangle(overlay, (0, y0), (W, H), bg, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    draw_label(img, text, (14, H - 18), scale=0.72, color=color, thickness=2)


def nose_vs_center(kps: np.ndarray, frame_cx: int, frame_cy: int) -> dict:
    """Nose tip distance + direction relative to frame center."""
    nose = kps[2].astype(np.float32)  # [Leye, Reye, Nose, Lmouth, Rmouth]
    dx = float(nose[0] - frame_cx)
    dy = float(nose[1] - frame_cy)
    dist = float(np.hypot(dx, dy))

    if abs(dx) < 12 and abs(dy) < 12:
        direction = "CENTER"
    else:
        horiz = "RIGHT" if dx > 0 else "LEFT"
        vert = "DOWN" if dy > 0 else "UP"
        if abs(dx) >= abs(dy) * 1.4:
            direction = horiz
        elif abs(dy) >= abs(dx) * 1.4:
            direction = vert
        else:
            direction = f"{vert}-{horiz}"

    return {
        "nose_xy": (int(nose[0]), int(nose[1])),
        "dx": dx,
        "dy": dy,
        "dist_px": dist,
        "direction": direction,
    }


def gaze_from_5pt(kps: np.ndarray) -> str:
    le, re, nose, _, _ = kps.astype(np.float32)
    eye_mid_x = 0.5 * (le[0] + re[0])
    eye_dist = float(np.linalg.norm(re - le)) + 1e-6
    offset = float((nose[0] - eye_mid_x) / eye_dist)
    if abs(offset) > LOOK_AWAY_THRESH:
        return "Looking away"
    return "Looking at camera"


def expr_color(name: str):
    return {
        "Smiling": (0, 255, 255),
        "Laughing": (0, 255, 200),
        "Frowning": (0, 140, 255),
        "Sad": (255, 100, 255),
        "Grimacing": (0, 200, 255),
        "Surprised": (255, 200, 100),
        "Neutral": (220, 220, 220),
    }.get(name, (220, 220, 220))


class Coach:
    def __init__(self):
        self.locked_name: str | None = None
        self.greeted = False
        self.last_blinks = 0
        self.msg = "Stand in front of the HD camera"
        self.msg_color = (200, 200, 200)
        self.warn = False
        self.msg_until = 0.0
        self.missing_since: float | None = None

    def _set(self, text, color=(0, 220, 255), hold_s=2.0, warn=False):
        self.msg = text
        self.msg_color = color
        self.warn = warn
        self.msg_until = time.time() + hold_s

    def update(self, *, no_face: bool, locked_name: str | None, expr, gaze: str, strangers: int):
        now = time.time()

        if no_face:
            if self.missing_since is None:
                self.missing_since = now
            self.locked_name = None
            self.greeted = False
            # Orange first, then red if still missing
            if now - self.missing_since < 1.5:
                self._set("WARNING: Face missing from frame", (0, 165, 255), 0.6, warn=True)
            else:
                self._set("ALERT: No face detected - return to camera", (0, 0, 255), 0.6, warn=True)
            return

        self.missing_since = None

        if locked_name is None:
            self.locked_name = None
            self.greeted = False
            if strangers > 0:
                self._set(
                    f"{strangers} unknown face(s) — enroll to unlock",
                    (80, 80, 255),
                    1.2,
                    warn=False,
                )
            elif now >= self.msg_until:
                self._set("No enrolled identity locked yet", (180, 180, 255), 1.0)
            return

        if locked_name != self.locked_name:
            self.locked_name = locked_name
            self.greeted = False
            self.last_blinks = expr.blink_count if expr else 0

        if not self.greeted:
            self.greeted = True
            extra = f" (+{strangers} unknown nearby)" if strangers else ""
            self._set(f"Welcome, {locked_name}! Identity locked.{extra}", (0, 255, 120), 3.0)
            return

        if expr is None:
            if now >= self.msg_until:
                self._set(f"{locked_name}: analyzing expression...", (200, 200, 200), 1.0)
            return

        if expr.blink_count > self.last_blinks:
            self.last_blinks = expr.blink_count
            self._set(f"Blink detected! Count = {expr.blink_count}", (0, 180, 255), 1.8)
            return
        if expr.wink:
            self._set(f"Wink ({expr.wink} eye) — cool!", (255, 180, 0), 1.8)
            return
        if expr.eyes_closed:
            self._set("Eyes closed", (0, 140, 255), 0.7)
            return
        if gaze.startswith("Looking away"):
            self._set(f"{locked_name}, look at the camera", (100, 200, 255), 1.2)
            return

        reactions = {
            "Laughing": (f"Haha! Laughing detected, {locked_name}!", (0, 255, 200)),
            "Smiling": (f"Smiling detected - nice one, {locked_name}!", (0, 255, 255)),
            "Frowning": ("Frowning detected", (0, 140, 255)),
            "Sad": ("Sadness detected - cheer up!", (255, 120, 255)),
            "Grimacing": ("Grimacing detected", (0, 200, 255)),
            "Surprised": ("Surprised face!", (255, 200, 100)),
        }
        if expr.expression in reactions and now >= self.msg_until - 0.3:
            text, color = reactions[expr.expression]
            self._set(text, color, 1.4)
            return

        if strangers > 0 and now >= self.msg_until:
            self._set(
                f"Locked on {locked_name} — ignoring {strangers} unknown face(s)",
                (180, 255, 180),
                2.0,
            )
            return

        if now >= self.msg_until:
            self._set(f"Hi {locked_name} — smile, blink, or frown to interact", (180, 255, 180), 2.5)


def open_hd_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    # Prefer HD on external cams
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def main():
    parser = argparse.ArgumentParser(description="Demo Identity Lock")
    parser.add_argument("--cam", type=int, default=2, help="External HD camera index (usually 2)")
    parser.add_argument("--thresh", type=float, default=DEFAULT_DIST_THRESH)
    args = parser.parse_args()

    cap = open_hd_camera(args.cam)
    if cap is None:
        for fb in (0, 1, 2, 3):
            if fb == args.cam:
                continue
            cap = open_hd_camera(fb)
            if cap is not None:
                print(f"[demo] fallback camera index {fb}")
                args.cam = fb
                break
        if cap is None:
            raise RuntimeError("No camera opened. Plug in the external HD cam and retry.")

    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX("models/embedder_arcface.onnx", (112, 112), debug=False)
    expr_analyzer = ExpressionAnalyzer(debug=False)
    coach = Coach()

    db = load_db_npz(DB_PATH)
    if not db:
        raise RuntimeError("Empty face DB. Run: python -m src.enroll --cam 2")
    matcher = FaceDBMatcher(db=db, dist_thresh=args.thresh)

    print(f"[demo] identities={matcher._names} cam={args.cam} thr={matcher.dist_thresh:.2f}")
    print("[demo] Multi-face: enrolled=LOCK (green), others=UNKNOWN (red)")
    print("[demo] Keys: q quit | r reload | +/- threshold")

    cv2.namedWindow("Falcon Eye Demo", cv2.WINDOW_NORMAL)
    sized = False
    t0 = time.time()
    frames = 0
    fps = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            if not sized:
                cv2.resizeWindow("Falcon Eye Demo", W, H)
                sized = True

            vis = frame.copy()
            frame_cx, frame_cy = W // 2, H // 2
            # Crosshair at frame center
            cv2.drawMarker(vis, (frame_cx, frame_cy), (255, 120, 0), cv2.MARKER_CROSS, 24, 1)

            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            faces = det.detect(frame, max_faces=MAX_FACES)

            locked_name = None
            strangers = 0
            primary_expr = None
            primary_gaze = "Looking at camera"
            nose_info = None

            if not faces:
                coach.update(no_face=True, locked_name=None, expr=None, gaze="", strangers=0)
            else:
                # Match every face; prefer enrolled with best (lowest) distance for LOCK
                matches = []
                for f in faces:
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned)
                    mr = matcher.match(emb)
                    matches.append((f, mr, aligned))

                # Sort: accepted first, then by distance ascending
                matches.sort(key=lambda t: (not t[1].accepted, t[1].distance))

                lock_assigned = False
                for f, mr, _aligned in matches:
                    x1 = max(0, min(W - 1, f.x1))
                    y1 = max(0, min(H - 1, f.y1))
                    x2 = max(0, min(W - 1, f.x2))
                    y2 = max(0, min(H - 1, f.y2))

                    if mr.accepted and not lock_assigned:
                        lock_assigned = True
                        locked_name = mr.name
                        color = (0, 255, 0)
                        label = f"LOCKED: {mr.name}"

                        # Pad ROI so mouth/eyes are not clipped (helps smile + blink)
                        pad_x = int(0.18 * (x2 - x1))
                        pad_y = int(0.22 * (y2 - y1))
                        rx1 = max(0, x1 - pad_x)
                        ry1 = max(0, y1 - pad_y)
                        rx2 = min(W, x2 + pad_x)
                        ry2 = min(H, y2 + pad_y)
                        face_roi = frame[ry1:ry2, rx1:rx2]
                        primary_expr = (
                            expr_analyzer.analyze(face_roi)
                            if (ry2 - ry1 > 40 and rx2 - rx1 > 40)
                            else None
                        )
                        primary_gaze = gaze_from_5pt(f.kps)
                        nose_info = nose_vs_center(f.kps, frame_cx, frame_cy)

                        # Nose marker + line to center
                        nx, ny = nose_info["nose_xy"]
                        cv2.circle(vis, (nx, ny), 6, (0, 255, 255), -1)
                        cv2.line(vis, (nx, ny), (frame_cx, frame_cy), (0, 200, 255), 2)

                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        for kx, ky in f.kps.astype(int):
                            cv2.circle(vis, (int(kx), int(ky)), 2, color, -1)

                        y0 = y1 - 10 if y1 >= 90 else min(H - 140, y2 + 20)
                        draw_label(vis, label, (x1, y0 - 66), 0.7, color, 2)
                        draw_label(vis, f"dist={mr.distance:.3f} | {primary_gaze}", (x1, y0 - 44), 0.5, (200, 255, 200), 1)
                        draw_label(
                            vis,
                            f"Nose: {nose_info['direction']}  "
                            f"dx={nose_info['dx']:+.0f}px dy={nose_info['dy']:+.0f}px  "
                            f"|r|={nose_info['dist_px']:.0f}px",
                            (x1, y0 - 24),
                            0.5,
                            (0, 255, 255),
                            1,
                        )
                        if primary_expr:
                            eye = "CLOSED" if primary_expr.eyes_closed else "OPEN"
                            if primary_expr.wink:
                                eye = f"WINK-{primary_expr.wink.upper()}"
                            draw_label(
                                vis,
                                f"Expr: {primary_expr.expression}",
                                (x1, y0 - 4),
                                0.6,
                                expr_color(primary_expr.expression),
                                2,
                            )
                            draw_label(
                                vis,
                                f"Eyes: {eye} | Blinks: {primary_expr.blink_count}  "
                                f"(EAR {primary_expr.ear_avg:.2f}/{primary_expr.ear_thresh:.2f})",
                                (x1, y0 + 18),
                                0.5,
                                (0, 255, 128),
                                1,
                            )
                            draw_label(
                                vis,
                                f"mouth lift={primary_expr.mouth_norm_diff:+.3f} "
                                f"width={primary_expr.mouth_width_norm:.2f} "
                                f"open={primary_expr.mouth_open_norm:.2f}",
                                (x1, y0 + 38),
                                0.45,
                                (200, 220, 255),
                                1,
                            )
                    elif mr.accepted:
                        # Other enrolled faces keep their names (don't steal primary lock)
                        color = (0, 200, 0)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        y_text = y1 - 8 if y1 >= 24 else y2 + 20
                        draw_label(vis, f"{mr.name}", (x1, y_text), 0.65, color, 2)
                        draw_label(vis, f"dist={mr.distance:.3f}", (x1, y_text + 20), 0.5, (180, 255, 180), 1)
                    else:
                        # True strangers only
                        strangers += 1
                        color = (0, 0, 255)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        for kx, ky in f.kps.astype(int):
                            cv2.circle(vis, (int(kx), int(ky)), 2, color, -1)
                        y_text = y1 - 8 if y1 >= 24 else y2 + 20
                        who = f"UNKNOWN (near {mr.name})" if mr.name else "UNKNOWN"
                        draw_label(vis, who, (x1, y_text), 0.65, color, 2)
                        draw_label(vis, f"dist={mr.distance:.3f}", (x1, y_text + 20), 0.5, (180, 180, 255), 1)

                if not lock_assigned:
                    expr_analyzer.reset_blink_count()

                coach.update(
                    no_face=False,
                    locked_name=locked_name,
                    expr=primary_expr,
                    gaze=primary_gaze,
                    strangers=strangers,
                )

            # Header
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (W, 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
            fps_s = f"{fps:.1f}" if fps else "--"
            hdr = (
                f"Falcon Eye Demo | IDs:{len(matcher._names)} | thr={matcher.dist_thresh:.2f} "
                f"| fps={fps_s} | cam={args.cam} | {W}x{H}"
            )
            cv2.putText(vis, hdr, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)

            draw_banner(vis, coach.msg, coach.msg_color, warn=coach.warn)
            cv2.imshow("Falcon Eye Demo", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                matcher.reload_from(DB_PATH)
                print(f"[demo] reloaded: {matcher._names}")
            elif key in (ord("+"), ord("=")):
                matcher.dist_thresh = min(1.20, matcher.dist_thresh + 0.02)
                print(f"[demo] thr={matcher.dist_thresh:.2f}")
            elif key == ord("-"):
                matcher.dist_thresh = max(0.05, matcher.dist_thresh - 0.02)
                print(f"[demo] thr={matcher.dist_thresh:.2f}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

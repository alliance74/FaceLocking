# src/gui_app.py
"""
Face Locking — animated welcome + live camera shell
(black / white / gold · Poppins)

    python -m src
    python -m src.gui_app --cam 2
"""
from __future__ import annotations

import json
import math
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from .gui_theme import (
    BLACK,
    BLACK_2,
    DANGER,
    FONT_DIR,
    GOLD,
    GOLD_DIM,
    GOLD_HOT,
    OK,
    WHITE,
    WHITE_DIM,
    load_poppins,
)

DB_PATH = Path("data/db/face_db.npz")
DB_JSON = Path("data/db/face_db.json")
ENROLL_DIR = Path("data/enroll")


class WelcomeScreen(tk.Frame):
    """Welcome screen: crazy moving welcome text only (no rings/waves)."""

    def __init__(self, master, fonts, on_done):
        super().__init__(master, bg=BLACK)
        self.on_done = on_done
        self.fonts = fonts
        self._t = 0.0
        self._done = False
        self._phrases = [
            "WELCOME",
            "WELCOME TO FACE LOCKING",
            "WELCOME — LOCK IN",
            "WELCOME BACK",
            "WELCOME · LET'S GO",
        ]

        self.canvas = tk.Canvas(self, bg=BLACK, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.bind("<Button-1>", lambda _e: self._finish())
        self.canvas.bind("<Button-1>", lambda _e: self._finish())
        master.bind("<Return>", lambda _e: self._finish())
        master.bind("<space>", lambda _e: self._finish())

        self.after(16, self._animate)
        self.after(5000, self._finish)

    def _animate(self):
        if self._done:
            return
        self._t += 0.055
        w = max(self.winfo_width(), 900)
        h = max(self.winfo_height(), 560)
        self.canvas.delete("all")
        cx, cy = w / 2, h / 2

        # horizontal racing welcome ribbons (text only)
        ribbon = "  WELCOME  ·  WELCOME  ·  WELCOME  ·  WELCOME  ·  "
        mx = int((-self._t * 160) % 420)
        self.canvas.create_text(
            w - mx, 48, text=ribbon * 4, fill=GOLD_HOT, font=self.fonts["subtitle"], anchor="w"
        )
        self.canvas.create_text(
            mx - 200, h - 48, text=ribbon * 4, fill=WHITE_DIM, font=self.fonts["small"], anchor="w"
        )

        phrase = self._phrases[int(self._t / 1.2) % len(self._phrases)]
        letter_gap = 34 if len(phrase) < 16 else 26
        total_w = max(len(phrase) - 1, 1) * letter_gap
        start_x = cx - total_w / 2

        for i, ch in enumerate(phrase):
            if ch == " ":
                continue
            wave = 32 * math.sin(self._t * 5.8 + i * 0.6)
            bounce = 22 * abs(math.sin(self._t * 4.2 + i * 0.4))
            swirl_x = 14 * math.cos(self._t * 3.1 + i * 0.45)
            flash = 0.5 + 0.5 * abs(math.sin(self._t * 7.5 + i * 0.8))
            color = _blend(GOLD_DIM, GOLD_HOT, flash) if i % 2 == 0 else _blend(WHITE_DIM, WHITE, flash)
            x = start_x + i * letter_gap + swirl_x
            y = cy + wave - bounce
            self.canvas.create_text(x + 2, y + 3, text=ch, fill="#1a1a1a", font=self.fonts["hero"])
            self.canvas.create_text(x, y, text=ch, fill=color, font=self.fonts["hero"])

        # floating helper line
        self.canvas.create_text(
            cx,
            cy + 110 + 10 * math.sin(self._t * 2.5),
            text="Click or press Enter to continue",
            fill=_blend(BLACK, WHITE_DIM, min(1.0, self._t / 1.5)),
            font=self.fonts["subtitle"],
        )

        self.after(16, self._animate)

    def _finish(self):
        if self._done:
            return
        self._done = True
        try:
            self.master.unbind("<Return>")
            self.master.unbind("<space>")
        except Exception:
            pass
        self.on_done()


def _blend(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))

    def rgb(c):
        c = c.lstrip("#")
        return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))

    a, b = rgb(c1), rgb(c2)
    m = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return f"#{m[0]:02x}{m[1]:02x}{m[2]:02x}"


class FaceLockingApp(tk.Tk):
    def __init__(self, cam: int = 2):
        super().__init__()
        self.cam_index = cam
        self.title("Face Locking")
        self.configure(bg=BLACK)
        self.geometry("1280x760")
        self.minsize(1100, 640)

        self.fonts = load_poppins(self)
        self._photo = None
        self._cap = None
        self._running = True
        self._mode = "live"  # live | enroll | recognize | demo
        self._enroll_name = ""
        self._status = "Welcome"
        self._banner = ""
        self._frame_lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None
        self._pipe = None  # lazy PipelineHub
        self._enroll_new = []
        self._enroll_base_count = 0
        self._last_fun_push = 0.0

        self.container = tk.Frame(self, bg=BLACK)
        self.container.pack(fill="both", expand=True)

        self.welcome = WelcomeScreen(self.container, self.fonts, on_done=self._show_main)
        self.welcome.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.bind("<Escape>", lambda _e: self._quit())

    # ---------- screens ----------
    def _show_main(self):
        self.welcome.destroy()
        self._build_main()
        self._start_camera()
        self.after(30, self._tick)

    def _build_main(self):
        self.main = tk.Frame(self.container, bg=BLACK)
        self.main.pack(fill="both", expand=True)

        self._auto = False
        self._last_auto = 0.0
        self._last_aligned = None
        self._msg_i = 0
        self._fun_messages = [
            "Looking good — try not to break the camera.",
            "Pro tip: smiling boosts your face-lock karma.",
            "If you blink twice, we count it. We're watching (politely).",
            "Unknown faces get the red-carpet rejection.",
            "Gold buttons. Black vibes. Zero chill for strangers.",
            "Your nose knows where center is. Does it though?",
            "Enroll like a celebrity. Demo like a spy movie.",
            "Camera says cheese. AI says 'identity confirmed'.",
            "Multi-face mode: one VIP lock, extras stay UNKNOWN.",
            "Don't leave the frame — we get dramatic about it.",
            "Poppins font. Fancy face. Main character energy.",
            "Save early, save often… especially after CAPTURE spam.",
        ]

        # IMPORTANT: pack bottom bar FIRST so it never gets crushed by video
        self.bottom_bar = tk.Frame(self.main, bg=BLACK_2, height=100)
        self.bottom_bar.pack(side="bottom", fill="x")
        self.bottom_bar.pack_propagate(False)
        tk.Frame(self.bottom_bar, bg=GOLD, height=3).pack(fill="x", side="top")

        # Few actions: enroll + recognize, then TRACK runs the full checklist in one mode
        actions = [
            ("LIVE", lambda: self._set_mode("live"), False),
            ("ENROLL", self._start_enroll, True),
            ("RECOGNIZE", lambda: self._set_mode("recognize"), False),
            ("TRACK", lambda: self._set_mode("demo"), True),
            ("QUIT", self._quit, False),
        ]
        inner = tk.Frame(self.bottom_bar, bg=BLACK_2)
        inner.pack(expand=True, fill="both")
        btn_row = tk.Frame(inner, bg=BLACK_2)
        btn_row.pack(expand=True)
        for text, cmd, filled in actions:
            self._bar_btn(btn_row, text, cmd, filled=filled).pack(side="left", padx=12, pady=22)

        # Enroll tools sit just above bottom bar
        self.enroll_bar = tk.Frame(self.main, bg=BLACK, height=64)
        # not packed until enroll mode
        for text, cmd, filled in (
            ("CAPTURE", self._enroll_capture, True),
            ("SAVE", self._enroll_save, True),
            ("AUTO", self._enroll_toggle_auto, False),
            ("RESET", self._enroll_reset, False),
        ):
            self._bar_btn(self.enroll_bar, text, cmd, filled=filled).pack(
                side="left", padx=8, pady=12
            )
        self.enroll_count = tk.Label(
            self.enroll_bar, text="", bg=BLACK, fg=GOLD, font=self.fonts["status"]
        )
        self.enroll_count.pack(side="right", padx=16)

        # Fun message + status strip (above enroll/bottom)
        self.msg_bar = tk.Frame(self.main, bg=BLACK, height=36)
        self.msg_bar.pack(side="bottom", fill="x")
        self.msg_bar.pack_propagate(False)
        self.fun_label = tk.Label(
            self.msg_bar,
            text=self._fun_messages[0],
            bg=BLACK,
            fg=GOLD_HOT,
            font=self.fonts["status"],
            anchor="w",
            padx=20,
        )
        self.fun_label.pack(side="left", fill="x", expand=True)
        self.status_bar = tk.Label(
            self.msg_bar,
            text="Camera warming up…",
            bg=BLACK,
            fg=WHITE_DIM,
            font=self.fonts["small"],
            anchor="e",
            padx=16,
        )
        self.status_bar.pack(side="right")

        # Top bar
        top = tk.Frame(self.main, bg=BLACK_2, height=52)
        top.pack(side="top", fill="x")
        top.pack_propagate(False)
        tk.Frame(top, bg=GOLD, height=2).pack(fill="x", side="top")

        tk.Label(
            top, text="FACE LOCKING", bg=BLACK_2, fg=GOLD, font=self.fonts["title"]
        ).pack(side="left", padx=20)

        self.mode_label = tk.Label(
            top, text="LIVE", bg=BLACK_2, fg=WHITE, font=self.fonts["status"]
        )
        self.mode_label.pack(side="left", padx=8)

        cam_wrap = tk.Frame(top, bg=BLACK_2)
        cam_wrap.pack(side="right", padx=16)
        tk.Label(cam_wrap, text="CAM", bg=BLACK_2, fg=WHITE_DIM, font=self.fonts["small"]).pack(
            side="left", padx=(0, 6)
        )
        self.cam_var = tk.IntVar(value=self.cam_index)
        spin = tk.Spinbox(
            cam_wrap,
            from_=0,
            to=5,
            width=3,
            textvariable=self.cam_var,
            font=self.fonts["body"],
            bg=BLACK,
            fg=WHITE,
            buttonbackground=BLACK_2,
            relief="flat",
            justify="center",
        )
        spin.pack(side="left")
        self._gold_btn(cam_wrap, "Apply", self._switch_camera, compact=True).pack(
            side="left", padx=(8, 0)
        )

        # Video fills remaining space
        stage = tk.Frame(self.main, bg=BLACK)
        stage.pack(side="top", fill="both", expand=True)
        self.video = tk.Label(stage, bg=BLACK)
        self.video.pack(fill="both", expand=True)

        self.after(3500, self._rotate_fun_message)

    def _bar_btn(self, parent, text, command, filled=False):
        if filled:
            bg, fg, active = GOLD, BLACK, GOLD_HOT
        else:
            bg, fg, active = BLACK, WHITE, BLACK_2
        btn = tk.Label(
            parent,
            text=f"  {text}  ",
            bg=bg,
            fg=fg,
            font=self.fonts["button"],
            cursor="hand2",
            padx=14,
            pady=10,
            highlightbackground=GOLD,
            highlightthickness=1,
        )

        def enter(_e):
            btn.configure(bg=active if filled else GOLD_DIM, fg=BLACK if filled else GOLD_HOT)

        def leave(_e):
            btn.configure(bg=bg, fg=fg)

        btn.bind("<Enter>", enter)
        btn.bind("<Leave>", leave)
        btn.bind("<Button-1>", lambda _e: command())
        return btn

    def _gold_btn(self, parent, text, command, compact=False):
        pad = (8, 4) if compact else (14, 8)
        btn = tk.Label(
            parent,
            text=f" {text} ",
            bg=GOLD,
            fg=BLACK,
            font=self.fonts["small"] if compact else self.fonts["button"],
            cursor="hand2",
            padx=pad[0],
            pady=pad[1],
        )
        btn.bind("<Button-1>", lambda _e: command())
        return btn

    # ---------- camera ----------
    def _start_camera(self):
        if self._cap is not None:
            self._cap.release()
        idx = int(self.cam_var.get()) if hasattr(self, "cam_var") else self.cam_index
        self.cam_index = idx
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._set_status(f"Camera {idx} failed to open", DANGER)
            self._cap = None
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap = cap
        self._set_status(f"Camera {idx} live", OK)

        def reader():
            while self._running and self._cap is not None:
                ok, frame = self._cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                with self._frame_lock:
                    self._latest_bgr = frame
                time.sleep(0.001)

        threading.Thread(target=reader, daemon=True).start()

    def _switch_camera(self):
        self._start_camera()

    def _tick(self):
        if not self._running:
            return
        with self._frame_lock:
            frame = None if self._latest_bgr is None else self._latest_bgr.copy()
        if frame is not None:
            try:
                vis = self._process(frame)
            except Exception as e:
                vis = frame
                self._set_status(f"Process error: {e}", DANGER)
            self._show_frame(vis)
        self.after(33, self._tick)

    def _show_frame(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Fit into label size
        lw = max(self.video.winfo_width(), 640)
        lh = max(self.video.winfo_height(), 360)
        h, w = rgb.shape[:2]
        scale = min(lw / w, lh / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (nw, nh))
        img = Image.fromarray(resized)
        self._photo = ImageTk.PhotoImage(image=img)
        self.video.configure(image=self._photo)

    # ---------- modes / pipeline ----------
    def _ensure_pipe(self):
        if self._pipe is None:
            self._set_status("Loading AI models…", GOLD)
            self.update_idletasks()
            self._pipe = PipelineHub()
            self._set_status("Models ready", OK)

    def _set_mode(self, mode: str):
        self._mode = mode
        self.mode_label.configure(text=mode.upper())
        if mode == "enroll":
            self.enroll_bar.pack_forget()
            self.enroll_bar.pack(side="bottom", fill="x", before=self.msg_bar)
            self._update_enroll_count()
            self._set_status(f"Enrolling {self._enroll_name}", GOLD)
            self._set_fun(f"Alright {self._enroll_name}, give us your best angles. CAPTURE away!")
        else:
            self.enroll_bar.pack_forget()
            self._auto = False
            fun = {
                "live": "Live HD preview — check the external camera looks crisp.",
                "recognize": "Recognize — enrolled faces get names; others stay Unknown.",
                "demo": "TRACK — lock, multi-face, nose, expressions, and missing-face alert in one.",
            }
            self._set_status(
                {"live": "Live", "recognize": "Recognize", "demo": "Track"}.get(mode, mode),
                WHITE_DIM,
            )
            self._set_fun(fun.get(mode, "Let's go."))
            if mode in ("recognize", "demo"):
                self._ensure_pipe()
                self._pipe.reload_db()
                ids = ", ".join(self._pipe.matcher._names) or "(empty)"
                if mode == "recognize":
                    self._set_fun(f"DB: {ids} · thr={self._pipe.matcher.dist_thresh:.2f}")
                else:
                    self._set_fun(
                        f"TRACK on ({ids}) — smile, blink, bring a stranger, step out for the alert."
                    )

    def _set_fun(self, text: str):
        if hasattr(self, "fun_label"):
            self.fun_label.configure(text=text)

    def _push_fun_throttled(self, text: str, every_s: float = 1.2):
        now = time.time()
        if now - self._last_fun_push < every_s:
            return
        self._last_fun_push = now
        self.after(0, lambda t=text: self._set_fun(t))

    def _rotate_fun_message(self):
        if not self._running or not hasattr(self, "fun_label"):
            return
        # Don't override mode-specific prompts too aggressively during enroll
        if self._mode != "enroll":
            self._msg_i = (self._msg_i + 1) % len(self._fun_messages)
            self.fun_label.configure(text=self._fun_messages[self._msg_i])
        self.after(4000, self._rotate_fun_message)

    def _start_enroll(self):
        name = self._ask_name()
        if not name:
            return
        self._enroll_name = name
        self._enroll_new = []
        self._ensure_pipe()
        person_dir = ENROLL_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        self._enroll_base_count = len(list(person_dir.glob("*.jpg")))
        self._set_mode("enroll")

    def _ask_name(self) -> Optional[str]:
        dlg = tk.Toplevel(self)
        dlg.title("Enroll")
        dlg.configure(bg=BLACK)
        dlg.geometry("440x240")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)
        tk.Frame(dlg, bg=GOLD, height=3).pack(fill="x")
        box = tk.Frame(dlg, bg=BLACK)
        box.pack(fill="both", expand=True, padx=28, pady=22)
        tk.Label(box, text="NEW IDENTITY", bg=BLACK, fg=GOLD, font=self.fonts["small"]).pack(anchor="w")
        tk.Label(
            box, text="Enter the person's name", bg=BLACK, fg=WHITE, font=self.fonts["title"]
        ).pack(anchor="w", pady=(6, 12))
        entry = tk.Entry(
            box,
            font=self.fonts["subtitle"],
            bg=BLACK_2,
            fg=WHITE,
            insertbackground=GOLD,
            relief="flat",
            highlightthickness=1,
            highlightbackground=GOLD_DIM,
            highlightcolor=GOLD,
        )
        entry.pack(fill="x", ipady=8)
        entry.focus_set()
        out: dict[str, Optional[str]] = {"v": None}

        def ok(_e=None):
            out["v"] = entry.get().strip()
            dlg.destroy()

        def cancel(_e=None):
            out["v"] = None
            dlg.destroy()

        row = tk.Frame(box, bg=BLACK)
        row.pack(fill="x", pady=(18, 0))
        self._bar_btn(row, "CANCEL", cancel, filled=False).pack(side="left")
        self._bar_btn(row, "CONTINUE", ok, filled=True).pack(side="right")
        entry.bind("<Return>", ok)
        dlg.bind("<Escape>", cancel)
        self.wait_window(dlg)
        if out["v"] is None:
            return None
        if not out["v"]:
            messagebox.showwarning("Name required", "Please type a name.")
            return None
        return out["v"]

    def _process(self, frame: np.ndarray) -> np.ndarray:
        mode = self._mode
        if mode == "live":
            return self._draw_live(frame)
        if mode == "enroll":
            return self._draw_enroll(frame)
        if mode == "recognize":
            return self._draw_recognize(frame)
        if mode == "demo":
            return self._draw_demo(frame)
        return frame

    def _draw_hud(self, vis, text, color=(212, 175, 55)):
        cv2.putText(vis, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    def _draw_live(self, frame):
        vis = frame.copy()
        self._draw_hud(vis, "LIVE PREVIEW", (212, 175, 55))
        return vis

    def _draw_enroll(self, frame):
        self._ensure_pipe()
        vis = frame.copy()
        faces = self._pipe.det.detect(frame, max_faces=1)
        self._last_aligned = None
        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 200, 0), 2)
            from .haar_5pt import align_face_5pt

            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            self._last_aligned = aligned
            th = 112
            vis[10 : 10 + th, vis.shape[1] - th - 10 : vis.shape[1] - 10] = aligned

        if self._auto and self._last_aligned is not None:
            now = time.time()
            if now - self._last_auto > 0.3:
                self._enroll_capture(silent=True)
                self._last_auto = now

        self._draw_hud(vis, f"ENROLL  {self._enroll_name}", (212, 175, 55))
        return vis

    def _draw_recognize(self, frame):
        self._ensure_pipe()
        vis = frame.copy()
        from .haar_5pt import align_face_5pt

        faces = self._pipe.det.detect(frame, max_faces=6)
        if not faces:
            self._draw_hud(vis, "RECOGNIZE — no face", (0, 140, 255))
            return vis

        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = self._pipe.embedder.embed(aligned)
            mr = self._pipe.matcher.match(emb)
            color = (0, 200, 0) if mr.accepted else (0, 0, 220)
            label = mr.name if mr.accepted else "Unknown"
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            cv2.putText(
                vis,
                f"{label}  dist={mr.distance:.3f}",
                (f.x1, max(20, f.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            if not mr.accepted:
                # show nearest identity hint for debugging threshold
                best_name = None
                best_dist = 1.0
                for n, v in self._pipe.matcher.db.items():
                    d = float(1.0 - float(np.dot(emb.reshape(-1), v.reshape(-1))))
                    if d < best_dist:
                        best_dist, best_name = d, n
                cv2.putText(
                    vis,
                    f"nearest={best_name} ({best_dist:.3f}) thr={self._pipe.matcher.dist_thresh:.2f}",
                    (f.x1, min(vis.shape[0] - 12, f.y2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 255),
                    1,
                )

        ids = ",".join(self._pipe.matcher._names) or "(empty DB)"
        self._draw_hud(vis, f"RECOGNIZE | DB: {ids}", (212, 175, 55))
        return vis

    def _draw_demo(self, frame):
        self._ensure_pipe()
        from .haar_5pt import align_face_5pt
        from .identity_lock_track import gaze_from_5pt, nose_vs_center

        vis = frame.copy()
        H, W = vis.shape[:2]
        cx, cy = W // 2, H // 2
        cv2.drawMarker(vis, (cx, cy), (0, 180, 255), cv2.MARKER_CROSS, 22, 1)

        faces = self._pipe.det.detect(frame, max_faces=8)
        faces_sorted = []
        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            mr = self._pipe.matcher.match(self._pipe.embedder.embed(aligned))
            faces_sorted.append((f, mr, aligned))
        # Primary lock = best accepted match (lowest distance); enrolled others keep their names
        faces_sorted.sort(key=lambda t: (not t[1].accepted, t[1].distance))

        locked = False
        banner = "Looking for faces…"
        for f, mr, aligned in faces_sorted:
            if mr.accepted and not locked:
                locked = True
                color = (0, 200, 0)
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                nose = nose_vs_center(f.kps, cx, cy)
                nx, ny = nose["nose_xy"]
                cv2.circle(vis, (nx, ny), 5, (0, 255, 255), -1)
                cv2.line(vis, (nx, ny), (cx, cy), (0, 200, 255), 2)
                pad = 20
                roi = frame[
                    max(0, f.y1 - pad) : min(H, f.y2 + pad),
                    max(0, f.x1 - pad) : min(W, f.x2 + pad),
                ]
                expr = self._pipe.expr.analyze(roi) if roi.size else None
                gaze = gaze_from_5pt(f.kps)
                cv2.putText(
                    vis,
                    f"LOCKED: {mr.name}",
                    (f.x1, max(24, f.y1 - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                cv2.putText(
                    vis,
                    f"Nose {nose['direction']} dx={nose['dx']:+.0f} dy={nose['dy']:+.0f} |r|={nose['dist_px']:.0f}",
                    (f.x1, max(44, f.y1 - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
                if expr:
                    cv2.putText(
                        vis,
                        f"{expr.expression} | Blinks {expr.blink_count} | {gaze}",
                        (f.x1, max(64, f.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
                    banner = f"{mr.name}: {expr.expression} · blinks {expr.blink_count}"
                else:
                    banner = f"Welcome, {mr.name}"
            elif mr.accepted:
                # Second enrolled person — name them, don't force Unknown
                color = (0, 180, 0)
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                cv2.putText(
                    vis,
                    f"{mr.name}  {mr.distance:.2f}",
                    (f.x1, max(20, f.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )
            else:
                color = (0, 0, 220)
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                nearest = mr.name or "?"
                cv2.putText(
                    vis,
                    f"UNKNOWN  nearest={nearest} ({mr.distance:.2f})",
                    (f.x1, max(20, f.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        if not faces:
            # missing-face warning band
            cv2.rectangle(vis, (0, H - 54), (W, H), (0, 80, 200), -1)
            cv2.putText(
                vis,
                "WARNING: Face missing — move closer / face a light",
                (20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            banner = "No face lock yet — brighter light + closer to cam helps"
            self._push_fun_throttled("Detector can't see a face yet. Step closer and face the light.")
        else:
            cv2.rectangle(vis, (0, H - 48), (W, H), (20, 20, 20), -1)
            cv2.putText(vis, banner, (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (212, 175, 55), 2)
            self._push_fun_throttled(banner)

        self._draw_hud(vis, "TRACK", (212, 175, 55))
        self._banner = banner
        return vis

    # ---------- enroll actions ----------
    def _update_enroll_count(self):
        n = len(self._enroll_new)
        self.enroll_count.configure(
            text=f"New captures: {n}   |   Existing crops: {self._enroll_base_count}"
        )

    def _enroll_capture(self, silent=False):
        if self._last_aligned is None:
            if not silent:
                self._set_status("No face to capture — center your face", DANGER)
            return
        self._ensure_pipe()
        emb = self._pipe.embedder.embed(self._last_aligned)
        self._enroll_new.append(emb)
        person_dir = ENROLL_DIR / self._enroll_name
        person_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(person_dir / f"{int(time.time() * 1000)}.jpg"), self._last_aligned)
        self._update_enroll_count()
        if not silent:
            self._set_status(f"Captured sample #{len(self._enroll_new)}", OK)
            jokes = [
                f"Snap #{len(self._enroll_new)} in the bag. Looking legendary.",
                "Nice. The database is collecting main-character energy.",
                "Captured. Do one more with a tiny head turn.",
                "Sheesh — that crop is clean.",
            ]
            self._set_fun(jokes[len(self._enroll_new) % len(jokes)])

    def _enroll_save(self):
        if len(self._enroll_new) + self._enroll_base_count < 3:
            self._set_status("Need at least a few captures before SAVE", DANGER)
            return
        self._ensure_pipe()
        # reload disk crops into embeddings if needed
        from .enroll import load_db, mean_embedding, save_db, EnrollConfig, load_existing_samples_from_crops

        cfg = EnrollConfig()
        db = load_db(cfg)
        person_dir = ENROLL_DIR / self._enroll_name
        base = load_existing_samples_from_crops(cfg, self._pipe.embedder_enroll, person_dir)
        # Prefer freshly captured + re-embedded crops
        samples = base if base else list(self._enroll_new)
        if self._enroll_new and not base:
            samples = list(self._enroll_new)
        if base and self._enroll_new:
            # disk already has the jpg we wrote; re-embed disk as source of truth
            samples = base
        template = mean_embedding(samples)
        db[self._enroll_name] = template
        meta = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_dim": int(template.size),
            "names": sorted(db.keys()),
            "samples_total_used": len(samples),
            "note": "Saved from Face Locking GUI",
        }
        save_db(cfg, db, meta)
        # refresh matcher
        self._pipe.reload_db()
        self._enroll_new.clear()
        self._enroll_base_count = len(list(person_dir.glob("*.jpg")))
        self._update_enroll_count()
        self._set_status(f"Saved '{self._enroll_name}' to data/db/", OK)
        self._set_fun(f"Boom — {self._enroll_name} is in the vault. Hit TRACK!")
        messagebox.showinfo("Saved", f"Identity '{self._enroll_name}' saved.\nDB now: {', '.join(sorted(db.keys()))}")

    def _enroll_toggle_auto(self):
        self._auto = not self._auto
        self._set_status(f"Auto capture {'ON' if self._auto else 'OFF'}", GOLD)

    def _enroll_reset(self):
        self._enroll_new.clear()
        self._update_enroll_count()
        self._set_status("New captures cleared", WHITE_DIM)

    def _set_status(self, text: str, color: str = WHITE_DIM):
        if hasattr(self, "status_bar"):
            self.status_bar.configure(text=text, fg=color)

    def _quit(self):
        self._running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self.destroy()


class PipelineHub:
    """Lazy-loaded detection / embedding / matching / expression stack."""

    def __init__(self):
        from .embed import ArcFaceEmbedderONNX as EnrollEmbedder
        from .expression import ExpressionAnalyzer
        from .recognize import ArcFaceEmbedderONNX, FaceDBMatcher, HaarFaceMesh5pt, load_db_npz

        self.det = HaarFaceMesh5pt(min_size=(50, 50), debug=False)
        self.embedder = ArcFaceEmbedderONNX("models/embedder_arcface.onnx", (112, 112), debug=False)
        self.embedder_enroll = EnrollEmbedder("models/embedder_arcface.onnx", (112, 112), debug=False)
        self.expr = ExpressionAnalyzer(debug=False)
        self._FaceDBMatcher = FaceDBMatcher
        self._load_db = load_db_npz
        self.reload_db()

    def reload_db(self):
        db = self._load_db(DB_PATH)
        self.matcher = self._FaceDBMatcher(db=db, dist_thresh=0.50)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=2)
    args, _ = parser.parse_known_args()
    app = FaceLockingApp(cam=args.cam)
    app.mainloop()


if __name__ == "__main__":
    main()

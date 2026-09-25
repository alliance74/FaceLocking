# src/expression.py
"""
Expression + eye-state estimation (MediaPipe FaceMesh geometry).

Expressions: Smiling, Laughing, Frowning, Sad, Grimacing, Surprised, Neutral
Also: blinks (adaptive EAR), winks, eyes open/closed
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# 6-point eyes for a more stable EAR
L_EYE = (33, 160, 158, 133, 153, 144)  # outer, top1, top2, inner, bot2, bot1
R_EYE = (263, 387, 385, 362, 380, 373)

L_BROW_MID, R_BROW_MID = 105, 334
L_EYE_TOP, R_EYE_TOP = 159, 386

MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14

# Soft absolute floor (adaptive threshold does most of the work)
BLINK_ABS_FLOOR = 0.18
BLINK_REL_RATIO = 0.75          # closed if EAR < 75% of open baseline
BLINK_MIN_CONSEC = 1
BLINK_MAX_CONSEC = 12           # longer = eyes shut, not a blink

# Smile / laugh (loosened for HD webcam + laughing with open mouth)
SMILE_LIFT_THRESH = 0.008
SMILE_WIDTH_THRESH = 0.70
LAUGH_OPEN_THRESH = 0.16
MOUTH_OPEN_THRESH = 0.22
SAD_LIFT_THRESH = -0.055
FROWN_BROW_THRESH = 0.24
GRIMACE_WIDTH_THRESH = 0.78


@dataclass
class ExpressionResult:
    expression: str
    eyes_closed: bool
    ear_avg: float
    ear_left: float
    ear_right: float
    ear_thresh: float
    mouth_norm_diff: float
    mouth_width_norm: float
    mouth_open_norm: float
    mouth_open: bool
    brow_norm: float
    wink: Optional[str]
    blink_count: int
    is_blinking: bool


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def _ear(pts) -> float:
    """pts: (outer, top1, top2, inner, bot2, bot1)"""
    p1, p2, p3, p4, p5, p6 = pts
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * max(1e-6, _dist(p1, p4)))


class ExpressionAnalyzer:
    def __init__(self, debug: bool = False):
        self.debug = debug
        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe==0.10.21"
            )
        # static_image_mode=False keeps temporal continuity for blinks when ROI is stable
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._blink_count = 0
        self._consec_closed = 0
        self._was_closed = False
        self._ear_open_ema: Optional[float] = None

    def reset_blink_count(self):
        self._blink_count = 0
        self._consec_closed = 0
        self._was_closed = False
        # keep open-ear baseline so thresholds stay personalized

    def analyze(self, roi_bgr: np.ndarray) -> Optional[ExpressionResult]:
        H, W = roi_bgr.shape[:2]
        if H < 20 or W < 20:
            return None

        # Upscale small ROIs so FaceMesh landmarks are more stable
        scale = 1.0
        work = roi_bgr
        short = min(H, W)
        if short < 256:
            scale = 256.0 / short
            work = cv2.resize(roi_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_LINEAR)
        wh, ww = work.shape[:2]

        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark

        def pt(i):
            p = lm[i]
            return (p.x * ww, p.y * wh)

        l_ear = _ear([pt(i) for i in L_EYE])
        r_ear = _ear([pt(i) for i in R_EYE])
        ear_avg = 0.5 * (l_ear + r_ear)

        # Adaptive open-eye baseline
        if self._ear_open_ema is None:
            self._ear_open_ema = ear_avg
        # Update baseline only while likely open
        if ear_avg > (self._ear_open_ema * 0.9):
            self._ear_open_ema = 0.90 * self._ear_open_ema + 0.10 * ear_avg

        ear_thresh = max(BLINK_ABS_FLOOR, float(self._ear_open_ema) * BLINK_REL_RATIO)

        left_closed = l_ear < ear_thresh
        right_closed = r_ear < ear_thresh
        is_closed_now = left_closed and right_closed

        wink = None
        if left_closed and not right_closed and (r_ear - l_ear) > 0.05:
            wink = "Left"
        elif right_closed and not left_closed and (l_ear - r_ear) > 0.05:
            wink = "Right"

        if is_closed_now:
            self._consec_closed += 1
        else:
            if (
                self._was_closed
                and BLINK_MIN_CONSEC <= self._consec_closed <= BLINK_MAX_CONSEC
            ):
                self._blink_count += 1
            self._consec_closed = 0
        self._was_closed = is_closed_now

        eye_dist = _dist(pt(L_EYE[0]), pt(R_EYE[0]))
        mid_lip_y = (pt(MOUTH_TOP)[1] + pt(MOUTH_BOTTOM)[1]) / 2.0
        corner_avg_y = (pt(MOUTH_LEFT)[1] + pt(MOUTH_RIGHT)[1]) / 2.0
        mouth_norm_diff = (mid_lip_y - corner_avg_y) / max(1e-6, eye_dist)
        mouth_width_norm = _dist(pt(MOUTH_LEFT), pt(MOUTH_RIGHT)) / max(1e-6, eye_dist)
        mouth_open_norm = _dist(pt(MOUTH_TOP), pt(MOUTH_BOTTOM)) / max(1e-6, eye_dist)
        mouth_open = mouth_open_norm >= MOUTH_OPEN_THRESH

        brow_gap = (
            _dist(pt(L_BROW_MID), pt(L_EYE_TOP)) + _dist(pt(R_BROW_MID), pt(R_EYE_TOP))
        ) / 2.0
        brow_norm = brow_gap / max(1e-6, eye_dist)

        is_smiling = (
            (mouth_norm_diff >= SMILE_LIFT_THRESH)
            or (mouth_width_norm >= SMILE_WIDTH_THRESH)
            or (mouth_norm_diff >= 0.004 and mouth_width_norm >= 0.66)
        )
        # Laugh = open mouth + smile cues (or clearly open joyful mouth)
        is_laughing = mouth_open_norm >= LAUGH_OPEN_THRESH and (
            is_smiling or mouth_width_norm >= 0.68 or mouth_open_norm >= 0.24
        )
        is_sad = mouth_norm_diff < SAD_LIFT_THRESH
        is_frowning = (brow_norm < FROWN_BROW_THRESH) and (not is_smiling) and (mouth_norm_diff < 0.01)
        is_grimacing = (
            (mouth_width_norm >= GRIMACE_WIDTH_THRESH)
            and (mouth_norm_diff < SMILE_LIFT_THRESH)
            and (mouth_open_norm < LAUGH_OPEN_THRESH)
        )

        if is_laughing:
            expression = "Laughing"
        elif is_smiling:
            expression = "Smiling"
        elif is_grimacing:
            expression = "Grimacing"
        elif mouth_open and not is_smiling:
            expression = "Surprised"
        elif is_frowning:
            expression = "Frowning"
        elif is_sad:
            expression = "Sad"
        else:
            expression = "Neutral"

        if self.debug:
            print(
                f"[expr] ear={ear_avg:.3f}/{ear_thresh:.3f} closed={is_closed_now} "
                f"lift={mouth_norm_diff:+.3f} width={mouth_width_norm:.3f} "
                f"open={mouth_open_norm:.3f} -> {expression} blinks={self._blink_count}"
            )

        return ExpressionResult(
            expression=expression,
            eyes_closed=is_closed_now,
            ear_avg=ear_avg,
            ear_left=l_ear,
            ear_right=r_ear,
            ear_thresh=ear_thresh,
            mouth_norm_diff=mouth_norm_diff,
            mouth_width_norm=mouth_width_norm,
            mouth_open_norm=mouth_open_norm,
            mouth_open=mouth_open,
            brow_norm=brow_norm,
            wink=wink,
            blink_count=self._blink_count,
            is_blinking=is_closed_now,
        )

"""
Biological Motion Analysis Engine
────────────────────────────────
Validates that facial motion obeys biological physics laws.
"""
from __future__ import annotations
import numpy as np
import mediapipe as mp
from typing import List

from utils.face_utils import get_landmarks, get_jaw_center, get_lip_opening, get_eye_aspect_ratio

mp_face_mesh = mp.solutions.face_mesh

MAX_JERK           = 120.0
FREEZE_THRESHOLD   = 0.8
TELEPORT_THRESHOLD = 30.0
BLINK_EAR_THRESHOLD = 0.20


class MotionEngine:

    def __init__(self):
        self.name = "Motion"
        self.violations: List[str] = []

    def analyze(self, frames: List[np.ndarray]) -> float:
        """Returns fake probability score 0.0→1.0."""
        self.violations = []

        if not frames or len(frames) < 5:
            return 0.35

        landmarks_seq, jaw_centers, lip_openings, ear_values = [], [], [], []

        try:
            with mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.4
            ) as mesh:
                for frame in frames:
                    if frame is None or frame.size == 0:
                        continue
                    lm = get_landmarks(frame, mesh)
                    if lm is not None:
                        landmarks_seq.append(lm)
                        jaw_centers.append(get_jaw_center(lm))
                        lip_openings.append(get_lip_opening(lm))
                        ear_left  = get_eye_aspect_ratio(lm, "left")
                        ear_right = get_eye_aspect_ratio(lm, "right")
                        ear_values.append((ear_left + ear_right) / 2.0)
        except Exception:
            return 0.35

        if len(landmarks_seq) < 4:
            self.violations.append("[Motion] Insufficient face detections for motion analysis")
            return 0.45

        penalty = 0.0

        # ── Jaw motion physics ─────────────────────────────────────────────
        jaw_arr = np.array(jaw_centers)
        jaw_vel = np.linalg.norm(np.diff(jaw_arr, axis=0), axis=1)
        jaw_acc = np.diff(jaw_vel)
        jaw_jerk = np.abs(np.diff(jaw_acc)) if len(jaw_acc) > 1 else np.array([0.0])

        freeze_ratio = float(np.sum(jaw_vel < FREEZE_THRESHOLD) / len(jaw_vel))
        teleport_frames = int(np.sum(jaw_vel > TELEPORT_THRESHOLD))
        max_jerk = float(jaw_jerk.max()) if len(jaw_jerk) > 0 else 0.0

        if freeze_ratio > 0.60:
            self.violations.append(
                f"[Motion] FREEZE: Face static for {freeze_ratio*100:.0f}% of frames — deepfakes often freeze mid-motion")
            penalty += 0.30
        if teleport_frames > 0:
            self.violations.append(
                f"[Motion] TELEPORT: {teleport_frames} frame(s) with impossible motion (>{TELEPORT_THRESHOLD}px/frame)")
            penalty += min(0.40, teleport_frames * 0.10)
        if max_jerk > MAX_JERK:
            self.violations.append(
                f"[Motion] Excessive jerk ({max_jerk:.1f} px/frame³) — physically impossible facial acceleration")
            penalty += 0.20

        # ── Lip-jaw correlation ────────────────────────────────────────────
        lip_arr = np.array(lip_openings)
        if len(lip_arr) >= 5:
            lip_vel = np.abs(np.diff(lip_arr))
            lips_moving  = np.sum(lip_arr > 0.02)
            jaws_moving  = np.sum(jaw_vel > 1.0)
            corr_ratio = min(lips_moving, jaws_moving) / (max(lips_moving, jaws_moving) + 1e-9)
            if corr_ratio < 0.40 and lips_moving > 5:
                self.violations.append("[Motion] Lips moving but jaw not — face-swap boundary artifact")
                penalty += 0.20

        # ── Blink rate ─────────────────────────────────────────────────────
        if len(ear_values) >= 10:
            ear_arr = np.array(ear_values)
            blinks = 0
            in_blink = False
            for ear in ear_arr:
                if ear < BLINK_EAR_THRESHOLD:
                    if not in_blink:
                        blinks += 1
                        in_blink = True
                else:
                    in_blink = False
            fps_est = 25.0
            bpm = blinks / (len(ear_arr) / fps_est / 60.0)
            if bpm < 5:
                self.violations.append(
                    f"[Motion] Low blink rate ({bpm:.1f}/min) — deepfakes typically under-blink")
                penalty += 0.25

        return float(np.clip(penalty, 0.0, 1.0))

"""
Rule-Based Causal Consistency Engine
─────────────────────────────────────
Validates 8 biological cause-effect rules all real humans obey.
"""
from __future__ import annotations
import numpy as np
import cv2
import mediapipe as mp
from typing import List

from utils.face_utils import (
    get_landmarks, get_lip_opening, get_eye_aspect_ratio,
    LEFT_EYE, RIGHT_EYE, JAW
)

mp_face_mesh = mp.solutions.face_mesh

LIP_OPEN_SPEECH_MIN = 0.025
LIP_SPEED_MAX       = 0.08
BLINK_EAR_CLOSED    = 0.20
SYMMETRY_THRESHOLD  = 0.15
GAZE_JUMP_THRESHOLD = 0.12

LEFT_IRIS  = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]


def _iris_center(lm: np.ndarray, indices: List[int]) -> np.ndarray:
    return lm[indices, :2].mean(axis=0)


def _face_symmetry(lm: np.ndarray) -> float:
    pairs = [(33,263),(61,291),(70,300),(105,334),(159,386),(145,374)]
    face_cx = float(lm[1, 0])
    face_w  = abs(float(lm[234, 0]) - float(lm[454, 0]))
    if face_w < 1:
        return 0.0
    diffs = []
    for li, ri in pairs:
        left_dist  = abs(float(lm[li, 0]) - face_cx) / face_w
        right_dist = abs(float(lm[ri, 0]) - face_cx) / face_w
        diffs.append(abs(left_dist - right_dist))
    return float(np.mean(diffs))


class CausalEngine:

    def __init__(self):
        self.name = "Causal"
        self.violations: List[str] = []

    def analyze(self, frames: List[np.ndarray]) -> float:
        """Returns fake probability score 0.0→1.0."""
        self.violations = []

        if not frames or len(frames) < 8:
            return 0.30

        lip_arr, ear_l, ear_r, sym_arr = [], [], [], []
        landmarks_seq, left_iris_seq, right_iris_seq = [], [], []

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
                    if lm is None:
                        continue
                    landmarks_seq.append(lm)
                    lip_arr.append(get_lip_opening(lm))
                    ear_l.append(get_eye_aspect_ratio(lm, "left"))
                    ear_r.append(get_eye_aspect_ratio(lm, "right"))
                    sym_arr.append(_face_symmetry(lm))
                    try:
                        left_iris_seq.append(_iris_center(lm, LEFT_IRIS))
                        right_iris_seq.append(_iris_center(lm, RIGHT_IRIS))
                    except Exception:
                        pass
        except Exception:
            return 0.30

        n = len(lip_arr)
        if n < 4:
            return 0.30

        lip_arr = np.array(lip_arr)
        ear_l   = np.array(ear_l)
        ear_r   = np.array(ear_r)
        sym_arr = np.array(sym_arr)
        total_penalty = 0.0

        # ── RULE 1: Speech → Jaw opens ────────────────────────────────────
        speech_frames = np.sum(lip_arr > LIP_OPEN_SPEECH_MIN)
        jaw_open_rate = speech_frames / n
        if jaw_open_rate < 0.15 and np.mean(lip_arr) < LIP_OPEN_SPEECH_MIN:
            self.violations.append("[Causal] RULE 1: Jaw barely opens — no visible speech motion")
            total_penalty += 0.15
        elif np.max(lip_arr) > LIP_OPEN_SPEECH_MIN:
            # some speech detected — jaw should match
            pass

        # ── RULE 2: Speech pause → Blink ──────────────────────────────────
        lip_delta = np.diff(lip_arr)
        pause_frames = np.where((lip_arr[:-1] > 0.02) & (lip_delta < -0.01))[0]
        if len(pause_frames) > 3:
            blink_after = sum(
                1 for pf in pause_frames
                if pf < n - 5 and np.any(ear_l[pf:pf+5] < BLINK_EAR_CLOSED)
            )
            rate = blink_after / len(pause_frames)
            if rate < 0.15:
                self.violations.append(
                    f"[Causal] RULE 2: Only {rate*100:.0f}% of pauses followed by blink (natural ~30–50%)")
                total_penalty += 0.15

        # ── RULE 3: Facial symmetry stability ─────────────────────────────
        sym_std = float(np.std(sym_arr))
        if sym_std > 0.05:
            self.violations.append(
                f"[Causal] RULE 3: Symmetry fluctuating (σ={sym_std:.3f}) — real faces maintain stable symmetry")
            total_penalty += 0.15

        # ── RULE 4: Lip speed physical limit ──────────────────────────────
        lip_speed = np.abs(np.diff(lip_arr))
        overspeed = int(np.sum(lip_speed > LIP_SPEED_MAX))
        if overspeed > 2:
            self.violations.append(
                f"[Causal] RULE 4: {overspeed} frames exceed lip speed limit ({LIP_SPEED_MAX}/frame) — physically impossible")
            total_penalty += min(0.30, overspeed * 0.07)

        # ── RULE 5: Gaze continuity ────────────────────────────────────────
        if len(left_iris_seq) > 3:
            iris_arr = np.array(left_iris_seq)
            face_h = float(abs(landmarks_seq[0][10, 1] - landmarks_seq[0][152, 1]))
            gaze_vel = np.linalg.norm(np.diff(iris_arr, axis=0), axis=1) / max(face_h, 1)
            gaze_jumps = int(np.sum(gaze_vel > GAZE_JUMP_THRESHOLD))
            if gaze_jumps > 1:
                self.violations.append(
                    f"[Causal] RULE 5: {gaze_jumps} impossible gaze jumps — deepfake eye tracking failure")
                total_penalty += min(0.25, gaze_jumps * 0.08)

        # ── RULE 6: Bilateral symmetry within limit ────────────────────────
        mean_sym = float(np.mean(sym_arr))
        if mean_sym > SYMMETRY_THRESHOLD:
            self.violations.append(
                f"[Causal] RULE 6: Mean asymmetry={mean_sym:.3f} exceeds natural limit")
            total_penalty += 0.20

        # ── RULE 7: Sustained intermediate lip position ────────────────────
        partial = lip_arr[(lip_arr > LIP_OPEN_SPEECH_MIN * 0.5) & (lip_arr < LIP_OPEN_SPEECH_MIN * 1.5)]
        if len(partial) > n * 0.40:
            self.violations.append(
                f"[Causal] RULE 7: Lip held at intermediate position {len(partial)/n*100:.0f}% of frames — GAN interpolation")
            total_penalty += 0.15

        # ── RULE 8: Blink EAR completeness ────────────────────────────────
        combined_ear = (ear_l + ear_r) / 2
        blink_frames = combined_ear[combined_ear < BLINK_EAR_CLOSED]
        if len(blink_frames) > 0:
            incomplete = np.sum(blink_frames > 0.12)
            incomplete_ratio = incomplete / len(blink_frames)
            if incomplete_ratio > 0.6:
                self.violations.append(
                    f"[Causal] RULE 8: {incomplete_ratio*100:.0f}% of blinks are incomplete — GAN partial synthesis")
                total_penalty += 0.15

        return float(np.clip(total_penalty, 0.0, 1.0))

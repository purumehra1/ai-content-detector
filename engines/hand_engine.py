"""
Hand / Finger Analysis Engine
─────────────────────────────
Validates hand anatomy and motion physics via MediaPipe 21-point landmarks.
Deepfakes often hallucinate wrong finger counts, fused fingers, impossible joint angles.
"""
from __future__ import annotations
import numpy as np
import cv2
import mediapipe as mp
from typing import List

mp_hands = mp.solutions.hands

FINGERTIPS     = [4, 8, 12, 16, 20]
FINGER_MCP     = [2, 5, 9, 13, 17]
MIN_TIP_SEP    = 0.04
MIN_JOINT_ANG  = 10.0
MAX_JOINT_ANG  = 175.0


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _hand_size(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[0] - lm[9]))


def _analyze_hand(lm: np.ndarray) -> List[str]:
    issues = []
    size = _hand_size(lm)
    # Finger tip separation
    tips = lm[FINGERTIPS, :2]
    for i in range(len(FINGERTIPS) - 1):
        dist = float(np.linalg.norm(tips[i] - tips[i+1])) / max(size, 1)
        if dist < MIN_TIP_SEP:
            issues.append(f"Fused fingers {i+1}&{i+2} (dist={dist:.3f})")
    # Joint angles
    finger_joints = [(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
    for a, b, c in finger_joints:
        ang = _angle(lm[a], lm[b], lm[c])
        if ang < MIN_JOINT_ANG or ang > MAX_JOINT_ANG:
            issues.append(f"Unnatural joint angle {ang:.0f}° at landmark {b}")
    return issues


class HandEngine:

    def __init__(self):
        self.name = "Hand"
        self.violations: List[str] = []

    def analyze(self, frames: List[np.ndarray]) -> float:
        """Returns fake probability score 0.0→1.0."""
        self.violations = []

        if not frames:
            return 0.30

        hand_frames, all_issues, velocity_scores = 0, [], []
        frame_hand_counts = []
        prev_lm = None

        try:
            with mp_hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.6, min_tracking_confidence=0.5
            ) as hands:
                for frame in frames:
                    if frame is None or frame.size == 0:
                        frame_hand_counts.append(0)
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)
                    if not res.multi_hand_landmarks:
                        frame_hand_counts.append(0)
                        prev_lm = None
                        continue
                    hand_frames += 1
                    frame_hand_counts.append(len(res.multi_hand_landmarks))
                    for hlm in res.multi_hand_landmarks:
                        h, w = frame.shape[:2]
                        lm = np.array([[l.x*w, l.y*h, l.z*w] for l in hlm.landmark])
                        all_issues.extend(_analyze_hand(lm))
                        if prev_lm is not None and lm.shape == prev_lm.shape:
                            vel = float(np.mean(np.linalg.norm(lm[:,:2] - prev_lm[:,:2], axis=1)))
                            velocity_scores.append(vel)
                        prev_lm = lm
        except Exception:
            return 0.30

        if hand_frames == 0:
            return 0.30  # no hands — neutral score

        penalty = 0.0

        # Anatomy violations
        if all_issues:
            unique = list(dict.fromkeys(all_issues))[:3]
            for v in unique:
                self.violations.append(f"[Hand] {v}")
            penalty += min(0.50, len(all_issues) * 0.08)

        # Velocity discontinuity
        if velocity_scores:
            mean_v = float(np.mean(velocity_scores))
            std_v  = float(np.std(velocity_scores))
            if std_v > mean_v * 3:
                self.violations.append(
                    f"[Hand] Discontinuous hand motion (vel std={std_v:.1f} >> mean={mean_v:.1f})")
                penalty += 0.25

        # Hand count instability
        non_zero = [c for c in frame_hand_counts if c > 0]
        if non_zero and (max(non_zero) - min(non_zero)) > 1:
            self.violations.append(
                f"[Hand] Hand count varies {min(non_zero)}–{max(non_zero)} across frames")
            penalty += 0.20

        return float(np.clip(penalty, 0.0, 1.0))

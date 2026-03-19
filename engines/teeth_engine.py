"""
Teeth Consistency Engine
────────────────────────
Teeth are biologically stable — they don't change shape between frames.
GANs consistently fail to maintain teeth structure across frames.
"""
from __future__ import annotations
import numpy as np
import cv2
import mediapipe as mp
from typing import List

from utils.face_utils import get_landmarks, get_teeth_region, get_lip_opening

mp_face_mesh = mp.solutions.face_mesh

PHASH_SIZE = 16
MIN_TEETH_BRIGHTNESS = 180
MAX_BRIGHTNESS_VAR   = 25.0


def _phash(img: np.ndarray, hash_size: int = PHASH_SIZE) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (hash_size, hash_size))
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size // 4, :hash_size // 4]
    return (dct_low > np.median(dct_low)).flatten()


def _hamming(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(np.sum(h1 != h2)) / len(h1)


class TeethEngine:

    def __init__(self):
        self.name = "Teeth"
        self.violations: List[str] = []

    def analyze(self, frames: List[np.ndarray]) -> float:
        """Returns fake probability score 0.0→1.0."""
        self.violations = []

        if not frames or len(frames) < 4:
            return 0.35

        teeth_regions, brightness_vals, phash_vals, edge_vals = [], [], [], []
        lip_openings_all, teeth_visible = [], []

        try:
            with mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5
            ) as mesh:
                for frame in frames:
                    if frame is None or frame.size == 0:
                        continue
                    lm = get_landmarks(frame, mesh)
                    if lm is None:
                        continue
                    lip_open = get_lip_opening(lm)
                    lip_openings_all.append(lip_open)
                    if lip_open < 0.015:
                        teeth_visible.append(False)
                        continue
                    region = get_teeth_region(frame, lm)
                    if region is None or region.size == 0:
                        teeth_visible.append(False)
                        continue
                    gray_r = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
                    bright = float(np.mean(gray_r))
                    if bright < MIN_TEETH_BRIGHTNESS * 0.6:
                        teeth_visible.append(False)
                        continue
                    teeth_visible.append(True)
                    teeth_regions.append(region)
                    brightness_vals.append(bright)
                    sobel = cv2.Sobel(gray_r, cv2.CV_64F, 1, 1, ksize=3)
                    edge_vals.append(float(np.mean(np.abs(sobel))))
                    phash_vals.append(_phash(region))
        except Exception:
            return 0.35

        n_open = sum(1 for l in lip_openings_all if l > 0.015)
        if n_open == 0:
            return 0.20

        penalty = 0.0
        show_rate = sum(teeth_visible) / max(n_open, 1)

        if n_open > 4 and show_rate < 0.40:
            self.violations.append(
                f"[Teeth] Teeth visible in only {show_rate*100:.0f}% of open-mouth frames — GAN flickering")
            penalty += 0.35

        if len(brightness_vals) < 3:
            return float(np.clip(penalty, 0.0, 1.0))

        b_std = float(np.std(brightness_vals))
        if b_std > MAX_BRIGHTNESS_VAR:
            self.violations.append(
                f"[Teeth] Brightness variance σ={b_std:.1f} — real teeth brightness is stable")
            penalty += 0.25

        if len(phash_vals) >= 3:
            dists = [_hamming(phash_vals[i], phash_vals[i+1]) for i in range(len(phash_vals)-1)]
            mean_dist = float(np.mean(dists))
            if mean_dist > 0.22:
                self.violations.append(
                    f"[Teeth] Structural pHash distance={mean_dist:.3f} — teeth shape changing between frames")
                penalty += 0.30

        if len(edge_vals) >= 3:
            e_std = float(np.std(edge_vals))
            if e_std > 8.0:
                self.violations.append(
                    f"[Teeth] Texture edge variance={e_std:.1f} — GAN regenerating teeth texture")
                penalty += 0.15

        return float(np.clip(penalty, 0.0, 1.0))

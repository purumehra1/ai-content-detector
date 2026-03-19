"""
Active Stability Testing Engine
────────────────────────────────
Applies controlled perturbations to frames and measures score sensitivity.
Real videos degrade gracefully; deepfakes show erratic brittleness.
Returns: (score 0→1, violations list, stability_modifier ±0.03)
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple


def _perturb_brightness(frame: np.ndarray, factor: float = 1.25) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _perturb_compress(frame: np.ndarray, quality: int = 40) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_COLOR)


def _perturb_blur(frame: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    k = max(3, int(sigma * 3) | 1)
    return cv2.GaussianBlur(frame, (k, k), sigma)


def _perturb_noise(frame: np.ndarray, std: float = 12.0) -> np.ndarray:
    noise = np.random.normal(0, std, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _perturb_contrast(frame: np.ndarray, alpha: float = 1.4, beta: float = -20) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _quick_score(frame: np.ndarray) -> float:
    """Lightweight proxy score via edge density + color variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    color_var = float(np.var(frame.astype(np.float32)))
    return float(np.clip(lap_var / 2000.0 + color_var / 50000.0, 0, 1))


PERTURBATIONS = {
    "brightness_up":    lambda f: _perturb_brightness(f, 1.25),
    "brightness_down":  lambda f: _perturb_brightness(f, 0.75),
    "jpeg_compress":    lambda f: _perturb_compress(f, 40),
    "gaussian_blur":    lambda f: _perturb_blur(f, 1.5),
    "gaussian_noise":   lambda f: _perturb_noise(f, 12),
    "contrast_stretch": lambda f: _perturb_contrast(f, 1.4, -20),
}

STABILITY_THRESHOLD = 0.12
ERRATIC_THRESHOLD   = 0.25


class StabilityEngine:
    """Active stability / perturbation testing engine."""

    def __init__(self, base_engine=None, **kwargs):
        """
        Args:
            base_engine: optional reference to CNN engine (not used — kept for API compat)
        """
        self.base_engine = base_engine
        self.name = "Stability"
        self.violations: List[str] = []

    def analyze(self, frames: List[np.ndarray]) -> Tuple[float, List[str], float]:
        """
        Returns:
            (fake_score 0→1, violations, stability_modifier ±0.03)
        """
        self.violations = []

        if not frames or len(frames) < 3:
            return 0.5, [], 0.0

        sample = frames[::max(1, len(frames) // 10)][:10]
        sample = [f for f in sample if f is not None and f.size > 0]
        if not sample:
            return 0.5, [], 0.0

        baseline_scores = [_quick_score(f) for f in sample]
        baseline_mean = float(np.mean(baseline_scores))

        perturbation_deltas = {}
        penalty = 0.0

        for name, perturb_fn in PERTURBATIONS.items():
            perturbed_scores = []
            for frame in sample:
                try:
                    p = perturb_fn(frame)
                    perturbed_scores.append(_quick_score(p))
                except Exception:
                    continue
            if not perturbed_scores:
                continue
            delta = abs(float(np.mean(perturbed_scores)) - baseline_mean)
            perturbation_deltas[name] = round(delta, 4)

            if delta > ERRATIC_THRESHOLD:
                self.violations.append(
                    f"[Stability] EXTREME sensitivity to {name}: delta={delta:.3f} — deepfakes are brittle")
                penalty += 0.25
            elif delta > STABILITY_THRESHOLD:
                self.violations.append(
                    f"[Stability] Sensitive to {name}: delta={delta:.3f}")
                penalty += 0.10

        if len(perturbation_deltas) >= 3:
            delta_std = float(np.std(list(perturbation_deltas.values())))
            if delta_std > 0.12:
                self.violations.append(
                    f"[Stability] Erratic perturbation response (std={delta_std:.3f})")
                penalty += 0.15

        fake_score = float(np.clip(penalty, 0.0, 1.0))
        modifier = float(np.clip((fake_score - 0.5) * 0.06, -0.03, 0.03))
        return fake_score, self.violations, modifier

"""
Active Stability Testing Engine
────────────────────────────────
Applies controlled perturbations to video frames and measures
how consistently the detection behaves. Real videos tend to degrade
gracefully under perturbation; deepfakes show erratic sensitivity.

Perturbations tested:
  • Brightness variation (±20%)
  • JPEG compression (quality 40)
  • Gaussian blur (σ=1.5)
  • Contrast stretching
  • Random noise injection

Output: stability_score (0=stable/real, 1=unstable/fake)
"""
import numpy as np
import cv2
import io


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
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def _perturb_contrast(frame: np.ndarray, alpha: float = 1.4, beta: float = -20) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


# Simple frame-level "AI score" proxy using edge density + color variance
# (used as a lightweight stability metric without re-running the heavy CNN)
def _quick_score(frame: np.ndarray) -> float:
    """Lightweight proxy score — edge density + color variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Laplacian variance (sharpness)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Color variance
    color_var = float(np.var(frame.astype(np.float32)))
    # Normalize to 0–1 range (empirical scale)
    score = np.clip(lap_var / 2000.0 + color_var / 50000.0, 0, 1)
    return score


PERTURBATIONS = {
    "brightness_up": lambda f: _perturb_brightness(f, 1.25),
    "brightness_down": lambda f: _perturb_brightness(f, 0.75),
    "jpeg_compress": lambda f: _perturb_compress(f, 40),
    "gaussian_blur": lambda f: _perturb_blur(f, 1.5),
    "gaussian_noise": lambda f: _perturb_noise(f, 12),
    "contrast_stretch": lambda f: _perturb_contrast(f, 1.4, -20),
}

STABILITY_THRESHOLD = 0.12   # score delta above this = unstable
ERRATIC_THRESHOLD = 0.25     # extreme instability


class StabilityEngine:
    WEIGHT = 0.0  # included in fusion but not separately weighted (embedded in final formula)

    def analyze(self, frames: list) -> dict:
        if len(frames) < 3:
            return {"score": 0.5, "violations": [], "confidence": 0.0}

        # Sample up to 10 frames for perturbation testing
        sample = frames[::max(1, len(frames) // 10)][:10]

        baseline_scores = [_quick_score(f) for f in sample]
        baseline_mean = float(np.mean(baseline_scores))

        perturbation_deltas = {}
        violations = []
        penalty = 0.0

        for name, perturb_fn in PERTURBATIONS.items():
            perturbed_scores = []
            for frame in sample:
                try:
                    perturbed = perturb_fn(frame)
                    perturbed_scores.append(_quick_score(perturbed))
                except Exception:
                    continue
            if not perturbed_scores:
                continue

            delta = abs(float(np.mean(perturbed_scores)) - baseline_mean)
            perturbation_deltas[name] = round(delta, 4)

            if delta > ERRATIC_THRESHOLD:
                violations.append(f"EXTREME INSTABILITY under {name}: score changed by {delta:.3f} (real video should degrade gracefully)")
                penalty += 0.25
            elif delta > STABILITY_THRESHOLD:
                violations.append(f"SENSITIVITY to {name}: score shift={delta:.3f} (deepfakes are often brittle to compression/noise)")
                penalty += 0.10

        # Real videos: scores should vary smoothly with perturbation strength
        # Deepfakes: often show non-monotonic, erratic responses
        if len(perturbation_deltas) >= 3:
            delta_vals = list(perturbation_deltas.values())
            delta_std = float(np.std(delta_vals))
            if delta_std > 0.12:
                violations.append(f"ERRATIC RESPONSE PATTERN: Score variance across perturbation types={delta_std:.3f} — deepfakes show inconsistent robustness")
                penalty += 0.15

        fake_score = min(1.0, penalty)
        return {
            "score": fake_score,
            "label": "FAKE" if fake_score > 0.5 else "REAL",
            "baseline_score": round(baseline_mean, 4),
            "perturbation_deltas": perturbation_deltas,
            "violations": violations,
            "confidence": min(1.0, abs(fake_score - 0.5) * 2),
            "frames_tested": len(sample),
        }

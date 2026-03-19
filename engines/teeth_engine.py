"""
Teeth Consistency Engine
────────────────────────
Teeth are biologically stable — they don't change shape, alignment, or brightness
during normal speech. GANs struggle to maintain teeth consistency across frames.

Analyzes:
  • Structural hash similarity (pHash) between teeth regions
  • Brightness variance (GANs sometimes flicker teeth region)
  • Texture gradient consistency (Sobel edge map comparison)
  • Teeth presence/absence consistency (flickering in/out = fake indicator)
Output: teeth_consistency_score (0=consistent/real, 1=inconsistent/fake)
"""
import numpy as np
import cv2
import mediapipe as mp
from utils.face_utils import get_landmarks, get_teeth_region, get_lip_opening

mp_face_mesh = mp.solutions.face_mesh

PHASH_SIZE = 16
MIN_TEETH_BRIGHTNESS = 180   # teeth should be bright (white/off-white)
MAX_BRIGHTNESS_VAR = 25.0    # max acceptable brightness variance across frames


def phash(img: np.ndarray, hash_size: int = PHASH_SIZE) -> np.ndarray:
    """Perceptual hash of an image region."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (hash_size, hash_size))
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size // 4, :hash_size // 4]
    median = np.median(dct_low)
    return (dct_low > median).flatten()


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(np.sum(h1 != h2)) / len(h1)


def mean_brightness(region: np.ndarray) -> float:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    return float(np.mean(gray))


def edge_density(region: np.ndarray) -> float:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    return float(np.mean(np.abs(sobel)))


class TeethEngine:
    WEIGHT = 0.15

    def analyze(self, frames: list) -> dict:
        if len(frames) < 4:
            return {"score": 0.5, "violations": [], "confidence": 0.0}

        teeth_regions = []
        brightness_vals = []
        phash_vals = []
        edge_vals = []
        teeth_visible_flags = []
        lip_openings = []

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as mesh:
            for frame in frames:
                lm = get_landmarks(frame, mesh)
                if lm is None:
                    continue

                lip_open = get_lip_opening(lm)
                lip_openings.append(lip_open)

                if lip_open < 0.015:  # mouth closed — skip teeth check
                    teeth_visible_flags.append(False)
                    continue

                region = get_teeth_region(frame, lm)
                if region is None or region.size == 0:
                    teeth_visible_flags.append(False)
                    continue

                # Check if actual bright region (teeth = bright pixels)
                bright = mean_brightness(region)
                if bright < MIN_TEETH_BRIGHTNESS * 0.6:  # too dark = gums/tongue
                    teeth_visible_flags.append(False)
                    continue

                teeth_visible_flags.append(True)
                teeth_regions.append(region)
                brightness_vals.append(bright)
                edge_vals.append(edge_density(region))

                h = phash(region)
                phash_vals.append(h)

        violations = []
        penalty = 0.0

        n_open_frames = sum(lip_openings[i] > 0.015 for i in range(len(lip_openings)))
        if n_open_frames == 0:
            return {"score": 0.2, "violations": ["Mouth mostly closed — teeth analysis not applicable"], "confidence": 0.3}

        teeth_show_rate = sum(teeth_visible_flags) / max(n_open_frames, 1)

        # ── 1. Teeth flickering ───────────────────────────────────────────────
        # If mouth is open but teeth keep appearing/disappearing = GAN glitch
        if n_open_frames > 4 and teeth_show_rate < 0.4:
            violations.append(f"TEETH FLICKER: Teeth visible in only {teeth_show_rate*100:.0f}% of open-mouth frames — GAN-typical instability")
            penalty += 0.35

        if len(brightness_vals) < 3:
            return {"score": max(0.0, min(1.0, penalty)), "violations": violations, "confidence": 0.3}

        # ── 2. Brightness variance ────────────────────────────────────────────
        brightness_std = float(np.std(brightness_vals))
        if brightness_std > MAX_BRIGHTNESS_VAR:
            violations.append(f"TEETH BRIGHTNESS VARIANCE: σ={brightness_std:.1f} — real teeth brightness is stable (GAN often flickers this region)")
            penalty += 0.25

        # ── 3. Structural consistency via pHash ───────────────────────────────
        if len(phash_vals) >= 3:
            distances = []
            for i in range(len(phash_vals) - 1):
                d = hamming_distance(phash_vals[i], phash_vals[i + 1])
                distances.append(d)
            mean_dist = float(np.mean(distances))
            if mean_dist > 0.22:
                violations.append(f"STRUCTURAL CHANGE: Teeth pHash distance={mean_dist:.3f} across frames — real teeth structure is immutable")
                penalty += 0.30

        # ── 4. Edge density variance ──────────────────────────────────────────
        edge_std = float(np.std(edge_vals))
        if edge_std > 8.0:
            violations.append(f"TEXTURE INSTABILITY: Teeth edge variance={edge_std:.1f} — texture structure changing between frames (GAN regenerating teeth)")
            penalty += 0.15

        fake_score = min(1.0, penalty)
        return {
            "score": fake_score,
            "label": "FAKE" if fake_score > 0.5 else "REAL",
            "teeth_visibility_rate": round(teeth_show_rate, 3),
            "brightness_std": round(float(np.std(brightness_vals)) if brightness_vals else 0, 2),
            "mean_phash_distance": round(float(np.mean([hamming_distance(phash_vals[i], phash_vals[i+1]) for i in range(len(phash_vals)-1)])) if len(phash_vals) >= 2 else 0, 3),
            "violations": violations,
            "confidence": min(1.0, abs(fake_score - 0.5) * 2),
            "frames_with_teeth": len(teeth_regions),
        }

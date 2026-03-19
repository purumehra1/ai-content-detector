from __future__ import annotations
"""
Skin Texture & Boundary Engine
────────────────────────────────
GAN-generated faces have characteristic skin rendering artifacts:
  1. Over-smoothed skin (GANs generate low-frequency skin, losing pore detail)
  2. Boundary artifacts at face swap seam (blend boundary unnatural)
  3. Local Binary Pattern (LBP) distribution anomalies
  4. Facial region texture inconsistency (forehead vs cheek vs nose)
  5. Gradient magnitude statistics deviating from natural skin

References:
  - Li & Lyu (2019) "Exposing DeepFake Videos By Detecting Face Warping Artifacts" CVPRW
  - Yang et al. (2019) "Exposing Deep Fakes Using Inconsistent Head Poses"
  - Matern et al. (2019) "Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations"
"""
import numpy as np
import cv2
from typing import List


class SkinTextureEngine:
    """Detects skin rendering artifacts unique to deepfake generation."""

    def __init__(self):
        self.name = "Skin & Boundary"
        self.violations: List[str] = []

    # ── public API ─────────────────────────────────────────────────────────
    def analyze(self, frames: List[np.ndarray]) -> float:
        self.violations = []
        if not frames:
            return 0.5

        scores = []
        for frame in frames[:40]:
            if frame is None or frame.size == 0:
                continue
            s = self._analyze_frame(frame)
            if s is not None:
                scores.append(s)

        if not scores:
            return 0.5

        # Check temporal consistency of texture
        temporal_score = self._temporal_texture_consistency(frames[:30])

        mean_score = float(np.mean(scores))
        return float(np.clip(0.65 * mean_score + 0.35 * temporal_score, 0.0, 1.0))

    # ── per-frame analysis ─────────────────────────────────────────────────
    def _analyze_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
        gray = cv2.resize(gray, (256, 256))

        sub_scores = [
            self._lbp_analysis(gray),
            self._gradient_statistics(gray),
            self._smoothness_check(gray),
            self._boundary_artifacts(frame),
        ]
        sub_scores = [s for s in sub_scores if s is not None]
        return float(np.mean(sub_scores)) if sub_scores else None

    def _lbp_analysis(self, gray: np.ndarray) -> float:
        """
        Local Binary Pattern texture analysis.
        Natural skin has characteristic LBP distribution;
        GAN-generated skin has fewer micro-textures (more uniform LBP).
        """
        # Manual LBP computation (8 neighbors, radius 1)
        lbp = np.zeros_like(gray, dtype=np.uint8)
        h, w = gray.shape
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            shifted = np.zeros_like(gray)
            shifted[max(0,dy):h+min(0,dy), max(0,dx):w+min(0,dx)] = \
                gray[max(0,-dy):h+min(0,-dy), max(0,-dx):w+min(0,-dx)]
            lbp = (lbp << 1) | (shifted >= gray).astype(np.uint8)

        # LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-9)

        # Entropy: natural skin has higher LBP entropy (more texture variety)
        entropy = float(-np.sum(hist * np.log(hist + 1e-9)))
        max_entropy = np.log(32)  # uniform distribution

        normalized_entropy = entropy / max_entropy

        # Low entropy = over-smooth = GAN artifact
        if normalized_entropy < 0.70:
            self.violations.append(f"[Skin] Low LBP texture entropy ({normalized_entropy:.2f}) — skin is over-smoothed, typical of GAN generation")
            return float(1.0 - normalized_entropy)
        return float(np.clip((0.82 - normalized_entropy) * 2.0, 0.0, 0.4))

    def _gradient_statistics(self, gray: np.ndarray) -> float:
        """
        Natural skin gradient magnitude follows a characteristic distribution.
        GAN skin has suppressed gradient magnitude (too smooth).
        """
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        mean_grad = float(magnitude.mean())
        std_grad = float(magnitude.std())
        # Coefficient of variation
        cv = std_grad / (mean_grad + 1e-9)

        # Natural faces: mean_grad typically 15–50, CV > 1.5
        if mean_grad < 8.0:
            self.violations.append(f"[Skin] Extremely smooth skin (mean gradient={mean_grad:.1f}) — GAN over-smoothing")
            return 0.75
        if cv < 0.8:
            self.violations.append(f"[Skin] Uniform gradient distribution (CV={cv:.2f}) — lacks natural skin micro-texture")
            return 0.55
        return float(np.clip((1.5 - cv) * 0.3, 0.0, 0.4))

    def _smoothness_check(self, gray: np.ndarray) -> float:
        """
        Laplacian variance measures image sharpness/texture.
        Very high → noisy (some deepfakes); very low → over-smooth (GANs).
        """
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(lap.var())

        # Real faces typically: 100–2000 Laplacian variance
        if variance < 40:
            self.violations.append(f"[Skin] Face is unusually smooth (Laplacian var={variance:.0f}) — may be GAN-synthesized")
            return float(np.clip((80 - variance) / 80.0 * 0.7, 0.0, 0.7))
        if variance > 5000:
            self.violations.append(f"[Skin] Excessive noise in face texture (Laplacian var={variance:.0f}) — reconstruction artifact")
            return float(np.clip((variance - 5000) / 5000.0 * 0.5, 0.0, 0.5))
        return float(np.clip((200 - variance) / 400.0, 0.0, 0.25)) if variance < 200 else 0.05

    def _boundary_artifacts(self, frame: np.ndarray) -> float:
        """
        Face-swap deepfakes have seam artifacts at the blend boundary.
        Check for abnormal edge patterns at the face perimeter.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame

        # Face boundary typically falls at 10-15% from edges
        # Check gradient asymmetry at these regions
        top_band    = gray[int(h*0.05):int(h*0.15), int(w*0.2):int(w*0.8)]
        bottom_band = gray[int(h*0.85):int(h*0.95), int(w*0.2):int(w*0.8)]
        center_band = gray[int(h*0.40):int(h*0.60), int(w*0.3):int(w*0.7)]

        def band_gradient_energy(band):
            if band.size == 0:
                return 0.0
            gx = cv2.Sobel(band, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)
            return float(np.sqrt(gx**2 + gy**2).mean())

        top_g = band_gradient_energy(top_band)
        bot_g = band_gradient_energy(bottom_band)
        ctr_g = band_gradient_energy(center_band) + 1e-9

        # Boundary / center ratio
        boundary_ratio = (top_g + bot_g) / (2 * ctr_g)

        if boundary_ratio > 2.5:
            self.violations.append(f"[Skin] Boundary gradient ratio={boundary_ratio:.1f} — face-swap seam artifacts detected")
            return min(0.75, boundary_ratio * 0.2)
        if boundary_ratio < 0.3:
            self.violations.append(f"[Skin] Suspiciously smooth face boundary ({boundary_ratio:.2f}) — possible blending mask artifact")
            return 0.45
        return float(np.clip((boundary_ratio - 1.5) * 0.15, 0.0, 0.3))

    # ── temporal analysis ──────────────────────────────────────────────────
    def _temporal_texture_consistency(self, frames: List[np.ndarray]) -> float:
        """Texture should be consistent across frames for real faces."""
        lap_vars = []
        for frame in frames:
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
            gray = cv2.resize(gray, (128, 128))
            lap_vars.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        if len(lap_vars) < 3:
            return 0.3

        arr = np.array(lap_vars)
        cv = arr.std() / (arr.mean() + 1e-9)

        if cv > 0.8:
            self.violations.append(f"[Skin] Highly inconsistent skin sharpness across frames (CV={cv:.2f}) — texture regeneration artifact")
            return min(0.70, cv * 0.55)
        return float(np.clip(cv * 0.3, 0.0, 0.35))

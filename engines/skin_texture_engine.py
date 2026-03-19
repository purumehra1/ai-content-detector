"""
Skin Texture & GAN Artifact Engine — v3.2
───────────────────────────────────────────
GAN-generated skin has highly characteristic rendering artifacts.
This engine uses 6 independent checks — all pure numpy/opencv, no ML deps.
Very reliable for obvious deepfakes.
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import List, Optional


class SkinTextureEngine:

    def __init__(self):
        self.name = "SkinTexture"
        self.violations: List[str] = []

    def analyze(self, frames: List[np.ndarray]) -> float:
        self.violations = []
        if not frames:
            return 0.5

        per_frame = []
        for frame in frames[:40]:
            if frame is None or frame.size == 0:
                continue
            s = self._analyze_frame(frame)
            if s is not None:
                per_frame.append(s)

        if not per_frame:
            return 0.5

        temporal = self._temporal_consistency(frames[:30])
        mean_s   = float(np.mean(per_frame))
        return float(np.clip(0.65 * mean_s + 0.35 * temporal, 0.0, 1.0))

    def _analyze_frame(self, frame: np.ndarray) -> Optional[float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
        gray256 = cv2.resize(gray, (256, 256))

        scores = [
            self._lbp_entropy(gray256),
            self._laplacian_check(gray256),
            self._gradient_stats(gray256),
            self._noise_analysis(frame),
            self._color_coherence(frame),
            self._boundary_seam(frame),
        ]
        valid = [s for s in scores if s is not None]
        return float(np.mean(valid)) if valid else None

    def _lbp_entropy(self, gray: np.ndarray) -> float:
        """LBP texture entropy — real skin has high entropy (>0.72)."""
        lbp = np.zeros_like(gray, dtype=np.uint8)
        offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        h, w = gray.shape
        for dy, dx in offsets:
            sl_src = gray[max(0,-dy):h+min(0,-dy), max(0,-dx):w+min(0,-dx)]
            sl_dst = np.zeros_like(gray)
            sl_dst[max(0,dy):h+min(0,dy), max(0,dx):w+min(0,dx)] = sl_src
            lbp = (lbp << 1) | (sl_dst >= gray).astype(np.uint8)

        hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0,256))
        hist = hist.astype(float) / (hist.sum() + 1e-9)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-9))) / 5.0  # normalize

        if entropy < 0.62:
            self.violations.append(
                f"[Skin] Low texture entropy ({entropy:.2f}) — over-smooth GAN skin")
            return float(1.0 - entropy)
        return float(np.clip((0.78 - entropy) * 1.5, 0.0, 0.35))

    def _laplacian_check(self, gray: np.ndarray) -> float:
        """Laplacian variance — real face: 100–2500."""
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 40:
            self.violations.append(
                f"[Skin] Extremely smooth face (Laplacian={lap_var:.0f}) — GAN over-smoothing")
            return float(np.clip((80 - lap_var) / 80.0 * 0.75, 0.0, 0.75))
        if lap_var > 8000:
            self.violations.append(
                f"[Skin] Excessive noise (Laplacian={lap_var:.0f}) — reconstruction artifact")
            return float(np.clip((lap_var - 8000) / 8000.0 * 0.60, 0.0, 0.60))
        if lap_var < 120:
            return float(np.clip((200 - lap_var) / 200.0 * 0.45, 0.0, 0.45))
        return float(np.clip((200 - lap_var) / 500.0, 0.0, 0.20)) if lap_var < 200 else 0.05

    def _gradient_stats(self, gray: np.ndarray) -> float:
        """Gradient distribution statistics."""
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        mean_g = float(mag.mean())
        cv_g   = float(mag.std()) / (mean_g + 1e-9)

        score = 0.0
        if mean_g < 6.0:
            self.violations.append(
                f"[Skin] Mean gradient={mean_g:.1f} — extremely smooth, typical of GAN generation")
            score += 0.70
        elif mean_g < 12.0:
            score += 0.45
        if cv_g < 0.7:
            self.violations.append(
                f"[Skin] Uniform gradient distribution (CV={cv_g:.2f}) — lacks natural micro-texture")
            score += 0.30
        return float(np.clip(score, 0.0, 0.85))

    def _noise_analysis(self, frame: np.ndarray) -> float:
        """
        Real photos have sensor noise; GANs generate artificially clean images.
        Estimate noise level via high-pass filtered residual.
        """
        if frame is None or frame.size == 0:
            return 0.30
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
        gray_f = gray.astype(np.float32)
        blurred = cv2.GaussianBlur(gray_f, (5, 5), 0)
        residual = gray_f - blurred
        noise_std = float(residual.std())

        # Real photos: noise_std ≈ 3–25
        if noise_std < 1.5:
            self.violations.append(
                f"[Skin] Unnaturally clean image (noise σ={noise_std:.2f}) — GAN-synthesized faces lack sensor noise")
            return float(np.clip((3.0 - noise_std) / 3.0 * 0.70, 0.0, 0.70))
        if noise_std > 35:
            return float(np.clip((noise_std - 35) / 35.0 * 0.45, 0.0, 0.45))
        return float(np.clip((5.0 - noise_std) / 5.0 * 0.30, 0.0, 0.30)) if noise_std < 5 else 0.10

    def _color_coherence(self, frame: np.ndarray) -> float:
        """
        Color channel coherence — in natural faces, channels are correlated.
        GANs sometimes generate channels inconsistently.
        """
        if frame is None or len(frame.shape) < 3:
            return 0.25
        frame_f = frame.astype(np.float32)
        b, g, r = frame_f[:,:,0], frame_f[:,:,1], frame_f[:,:,2]

        def safe_corr(a, b):
            if a.std() < 1e-6 or b.std() < 1e-6: return 1.0
            return float(np.corrcoef(a.ravel(), b.ravel())[0,1])

        rg = safe_corr(r, g)
        rb = safe_corr(r, b)
        gb = safe_corr(g, b)
        mean_corr = (abs(rg) + abs(rb) + abs(gb)) / 3.0

        if mean_corr < 0.70:
            self.violations.append(
                f"[Skin] Low color channel coherence ({mean_corr:.2f}) — abnormal GAN color generation")
            return float(np.clip((0.85 - mean_corr) * 1.8, 0.0, 0.65))
        return float(np.clip((0.90 - mean_corr) * 1.2, 0.0, 0.20))

    def _boundary_seam(self, frame: np.ndarray) -> float:
        """Face-swap seam at perimeter."""
        if frame is None or frame.size == 0:
            return 0.20
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
        gray_f = gray.astype(np.float32)
        h, w = gray_f.shape

        def band_energy(y1p, y2p, x1p, x2p):
            band = gray_f[int(h*y1p):int(h*y2p), int(w*x1p):int(w*x2p)]
            if band.size == 0: return 0.0
            gx = cv2.Sobel(band, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)
            return float(np.sqrt(gx**2 + gy**2).mean())

        top_e = band_energy(0.05,0.15, 0.2,0.8)
        bot_e = band_energy(0.85,0.95, 0.2,0.8)
        ctr_e = band_energy(0.40,0.60, 0.3,0.7)

        if ctr_e < 1e-6:
            return 0.20
        ratio = (top_e + bot_e) / (2 * ctr_e)

        if ratio > 3.0:
            self.violations.append(
                f"[Skin] Face boundary seam (gradient ratio={ratio:.1f}) — face-swap blend artifact")
            return float(np.clip(0.50 + (ratio - 3.0) * 0.08, 0.0, 0.80))
        if ratio < 0.20:
            return 0.40
        return float(np.clip((ratio - 1.5) * 0.12, 0.0, 0.25))

    def _temporal_consistency(self, frames: List[np.ndarray]) -> float:
        lap_vars = []
        for frame in frames:
            if frame is None: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
            gray = cv2.resize(gray, (128,128))
            lap_vars.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        if len(lap_vars) < 3:
            return 0.30
        arr = np.array(lap_vars)
        cv  = arr.std() / (arr.mean() + 1e-9)
        if cv > 0.80:
            self.violations.append(
                f"[Skin] Sharpness inconsistency CV={cv:.2f} — texture regeneration")
            return float(np.clip(cv * 0.55, 0.0, 0.70))
        return float(np.clip(cv * 0.25, 0.0, 0.30))

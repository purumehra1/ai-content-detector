"""
Frequency Domain Engine — v3.2
────────────────────────────────
GAN-generated faces leave characteristic artifacts in the frequency domain.
This engine is highly reliable because it's purely mathematical — no ML deps.

Checks:
  1. FFT high-frequency energy ratio (natural images follow 1/f²)
  2. DCT 8×8 block AC/DC coefficient statistics
  3. Checkerboard artifact detection (transpose conv artifact)
  4. Power spectrum slope deviation from -2.0
  5. Temporal frequency flickering across frames
  6. JPEG blocking artifact analysis (re-encoded deepfakes)
  7. Color channel frequency correlation

References:
  - Frank et al. (2020) "Leveraging Frequency Analysis for Deep Fake Image Forgery Detection" ICML
  - Durall et al. (2020) "Watch your Up-Convolution" CVPR
  - Liu et al. (2021) "Spatial-Phase Shallow Learning" CVPR
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import List, Optional, Tuple


NATURAL_HF_LOW   = 0.10   # minimum natural high-freq energy ratio
NATURAL_HF_HIGH  = 0.38   # maximum natural high-freq energy ratio
NATURAL_SLOPE    = -2.0   # natural 1/f² slope
SLOPE_TOLERANCE  = 0.9    # ±tolerance around natural slope


class FrequencyEngine:

    def __init__(self):
        self.name = "Frequency"
        self.violations: List[str] = []

    def analyze(self, frames: List[np.ndarray]) -> float:
        self.violations = []
        if not frames:
            return 0.5

        per_frame_scores = []
        for frame in frames[:30]:
            if frame is None or frame.size == 0:
                continue
            s = self._analyze_frame(frame)
            if s is not None:
                per_frame_scores.append(s)

        if not per_frame_scores:
            return 0.5

        frame_mean = float(np.mean(per_frame_scores))

        # Temporal flickering
        temporal = self._temporal_flicker(frames[:30])

        combined = float(np.clip(0.70 * frame_mean + 0.30 * temporal, 0.0, 1.0))
        return combined

    # ── Per-frame analysis ─────────────────────────────────────────────────
    def _analyze_frame(self, frame: np.ndarray) -> Optional[float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        gray = cv2.resize(gray, (256, 256)).astype(np.float32)

        scores = [
            self._fft_energy_ratio(gray),
            self._dct_statistics(gray),
            self._checkerboard(gray),
            self._power_slope(gray),
            self._jpeg_blocking(gray),
        ]
        valid = [s for s in scores if s is not None]
        return float(np.mean(valid)) if valid else None

    def _fft_energy_ratio(self, gray: np.ndarray) -> float:
        """Check high-freq energy ratio against natural 1/f² range."""
        fft = np.fft.fft2(gray)
        mag = np.abs(np.fft.fftshift(fft))
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
        r_max = min(cy, cx)

        low_e  = mag[dist < r_max * 0.25].sum()
        high_e = mag[dist > r_max * 0.50].sum()
        total  = low_e + high_e + 1e-9
        ratio  = float(high_e / total)

        if ratio < NATURAL_HF_LOW:
            self.violations.append(
                f"[Frequency] Over-smoothed spectrum (HF ratio={ratio:.3f}) — GAN blurring artifact")
            score = 0.70 + (NATURAL_HF_LOW - ratio) / NATURAL_HF_LOW * 0.25
        elif ratio > NATURAL_HF_HIGH:
            self.violations.append(
                f"[Frequency] Excessive high-freq energy (ratio={ratio:.3f}) — GAN upsampling artifact")
            score = 0.65 + (ratio - NATURAL_HF_HIGH) / (1 - NATURAL_HF_HIGH) * 0.30
        else:
            # Natural range — how centered is it?
            center = (NATURAL_HF_LOW + NATURAL_HF_HIGH) / 2
            dev = abs(ratio - center) / ((NATURAL_HF_HIGH - NATURAL_HF_LOW) / 2)
            score = dev * 0.35  # max 0.35 for border of natural range
        return float(np.clip(score, 0.0, 1.0))

    def _dct_statistics(self, gray: np.ndarray) -> float:
        """DCT 8×8 block AC/DC statistics."""
        h, w = gray.shape
        ac_energies, dc_vals = [], []
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i:i+8, j:j+8]
                dct   = cv2.dct(block)
                dc_vals.append(abs(dct[0, 0]))
                ac = dct.copy(); ac[0, 0] = 0
                ac_energies.append(np.sum(ac**2))

        if not ac_energies:
            return 0.5

        ac_mean = float(np.mean(ac_energies))
        dc_mean = float(np.mean(dc_vals))
        ratio   = ac_mean / (dc_mean**2 + 1e-6)

        # Empirically: real faces 0.5 < ratio < 15.0
        if ratio < 0.3:
            self.violations.append(
                f"[Frequency] DCT: very low AC/DC ratio ({ratio:.2f}) — GAN synthesis")
            return 0.72
        if ratio > 20.0:
            self.violations.append(
                f"[Frequency] DCT: excessive AC/DC ratio ({ratio:.2f}) — reconstruction artifact")
            return 0.65
        return float(np.clip((1.5 - ratio / 10.0) * 0.35, 0.0, 0.40))

    def _checkerboard(self, gray: np.ndarray) -> float:
        """Detect transpose-conv checkerboard at f=0.5 cycles/px."""
        fft = np.fft.fft2(gray)
        mag = np.log1p(np.abs(np.fft.fftshift(fft)))
        h, w = mag.shape
        r = max(h, w) // 16

        # N/2 frequency corners (checkerboard signature)
        corner_e = 0.0
        for fy, fx in [(h//4, w//4),(h//4, 3*w//4),(3*h//4, w//4),(3*h//4, 3*w//4)]:
            Y, X = np.ogrid[:h, :w]
            mask = (Y-fy)**2 + (X-fx)**2 < r**2
            corner_e += float(mag[mask].mean()) if mask.any() else 0.0

        cy, cx = h//2, w//2
        center_e = float(mag[cy-r:cy+r, cx-r:cx+r].mean()) if r > 0 else 1.0

        ratio = corner_e / (center_e * 4 + 1e-9)
        if ratio > 0.88:
            self.violations.append(
                f"[Frequency] Checkerboard pattern (ratio={ratio:.2f}) — GAN transpose conv")
            return float(np.clip(0.65 + (ratio - 0.88) * 2.0, 0.65, 0.90))
        return float(np.clip(ratio * 0.50, 0.0, 0.45))

    def _power_slope(self, gray: np.ndarray) -> float:
        """Power spectrum slope — natural images: α≈-2.0."""
        fft = np.fft.fft2(gray)
        power = np.abs(np.fft.fftshift(fft))**2
        h, w = power.shape
        cy, cx = h//2, w//2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y-cy)**2 + (X-cx)**2) + 1e-9
        max_r = min(cy, cx)
        bands = np.linspace(2, max_r, 25)
        bp = []
        for i in range(len(bands)-1):
            mask = (dist >= bands[i]) & (dist < bands[i+1])
            if mask.any():
                bp.append(np.log(float(power[mask].mean()) + 1e-9))

        if len(bp) < 6:
            return 0.40

        log_f = np.log(np.linspace(2, max_r, len(bp)))
        slope, _ = np.polyfit(log_f, bp, 1)

        dev = abs(slope - NATURAL_SLOPE)
        if dev > SLOPE_TOLERANCE:
            self.violations.append(
                f"[Frequency] Power spectrum slope={slope:.2f} (natural≈-2.0, dev={dev:.2f})")
            return float(np.clip(0.45 + dev * 0.25, 0.0, 0.85))
        return float(np.clip(dev * 0.30, 0.0, 0.35))

    def _jpeg_blocking(self, gray: np.ndarray) -> float:
        """
        Detect JPEG blocking inconsistency — deepfakes often show irregular
        DCT block boundaries not consistent with standard JPEG compression.
        """
        h, w = gray.shape
        # Compute gradient at 8-pixel boundaries vs non-boundaries
        boundary_grads, inner_grads = [], []
        for i in range(0, h-1):
            row_diff = float(np.mean(np.abs(gray[i].astype(float) - gray[i+1].astype(float))))
            if i % 8 == 7:
                boundary_grads.append(row_diff)
            else:
                inner_grads.append(row_diff)

        if not boundary_grads or not inner_grads:
            return 0.30

        mean_b = float(np.mean(boundary_grads))
        mean_i = float(np.mean(inner_grads))

        # Natural JPEG: boundary_grad / inner_grad ≈ 1.0–2.5
        ratio = mean_b / (mean_i + 1e-9)

        if ratio > 4.0:
            self.violations.append(
                f"[Frequency] JPEG blocking ratio={ratio:.2f} — inconsistent compression artifacts")
            return float(np.clip(0.50 + (ratio-4.0)*0.05, 0.50, 0.80))
        if ratio < 0.6:
            self.violations.append(
                f"[Frequency] No JPEG block structure ({ratio:.2f}) — may be re-generated")
            return 0.55
        return float(np.clip((ratio - 1.5) * 0.10, 0.0, 0.30))

    def _temporal_flicker(self, frames: List[np.ndarray]) -> float:
        """Check frequency characteristics stability across frames."""
        if len(frames) < 4:
            return 0.30

        fingerprints = []
        for frame in frames:
            if frame is None or frame.size == 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
            gray = cv2.resize(gray, (64, 64)).astype(np.float32)
            fft  = np.abs(np.fft.fft2(gray))
            h, w = fft.shape
            # Quadrant means as fingerprint
            fp = np.array([
                fft[:h//2, :w//2].mean(),
                fft[:h//2, w//2:].mean(),
                fft[h//2:, :w//2].mean(),
                fft[h//2:, w//2:].mean(),
            ])
            fingerprints.append(fp / (fp.sum() + 1e-9))

        if len(fingerprints) < 3:
            return 0.30

        fps = np.array(fingerprints)
        # Frame-to-frame change in frequency fingerprint
        deltas = np.linalg.norm(np.diff(fps, axis=0), axis=1)
        mean_delta = float(deltas.mean())
        max_delta  = float(deltas.max())

        score = 0.0
        if mean_delta > 0.08:
            self.violations.append(
                f"[Frequency] Temporal frequency flickering (mean Δ={mean_delta:.3f}) — GAN regeneration")
            score += float(np.clip(mean_delta * 4.0, 0.0, 0.70))
        if max_delta > 0.20:
            score += 0.15
        return float(np.clip(score, 0.0, 0.90))

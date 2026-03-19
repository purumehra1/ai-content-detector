from __future__ import annotations
"""
Frequency Domain Engine
────────────────────────
GAN-generated faces leave characteristic artifacts in the frequency domain.
FFT analysis of face crops reveals:
- Checkerboard patterns (upsampling artifacts)
- Abnormal high-frequency energy distribution
- DCT coefficient statistics anomalies
- Power spectrum deviation from natural images

References:
  - Frank et al. (2020) "Leveraging Frequency Analysis for Deep Fake Image Forgery Detection" ICML
  - Durall et al. (2020) "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks..."
  - Liu et al. (2021) "Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain" CVPR
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional


class FrequencyEngine:
    """
    Detects deepfake artifacts via FFT and DCT analysis.
    GANs cannot reproduce the natural frequency distribution of real photos.
    """

    def __init__(self):
        self.name = "Frequency Domain"
        self.violations: List[str] = []

    # ── public API ─────────────────────────────────────────────────────────
    def analyze(self, frames: List[np.ndarray]) -> float:
        """
        Args:
            frames: list of BGR uint8 face-crop arrays (any size)
        Returns:
            score 0.0→1.0  (higher = more likely FAKE)
        """
        self.violations = []
        if not frames:
            return 0.5

        scores = []
        for f in frames:
            s = self._analyze_single(f)
            if s is not None:
                scores.append(s)

        if not scores:
            return 0.5

        mean_score = float(np.mean(scores))
        temporal = self._temporal_consistency(frames)
        combined = 0.70 * mean_score + 0.30 * temporal

        return float(np.clip(combined, 0.0, 1.0))

    # ── per-frame analysis ─────────────────────────────────────────────────
    def _analyze_single(self, frame: np.ndarray) -> Optional[float]:
        if frame is None or frame.size == 0:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        gray = cv2.resize(gray, (256, 256)).astype(np.float32)

        subscores = [
            self._fft_high_freq_ratio(gray),
            self._dct_coefficient_analysis(gray),
            self._checkerboard_detection(gray),
            self._power_spectrum_slope(gray),
        ]
        return float(np.mean(subscores))

    def _fft_high_freq_ratio(self, gray: np.ndarray) -> float:
        """
        GAN upsampling creates abnormal high-frequency energy.
        Real images have 1/f² power spectrum; GANs deviate significantly.
        """
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Low-freq energy (central 25%)
        r_low = min(cy, cx) // 4
        mask_low = np.zeros_like(magnitude, dtype=bool)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
        mask_low = dist < r_low
        low_energy = magnitude[mask_low].sum()

        # High-freq energy (outer 50% radius)
        mask_high = dist > (min(cy, cx) * 0.5)
        high_energy = magnitude[mask_high].sum()

        total = low_energy + high_energy + 1e-9
        ratio = high_energy / total

        # Real images: ratio ~0.15–0.30; GANs: often >0.40 or <0.08 (over-smoothed)
        if ratio > 0.42:
            self.violations.append(f"[Frequency] Excessive high-freq energy ({ratio:.2f}) — GAN upsampling artifact")
            return 0.80
        if ratio < 0.08:
            self.violations.append(f"[Frequency] Over-smoothed spectrum ({ratio:.2f}) — GAN blurring artifact")
            return 0.70
        # Normalize deviation from natural range [0.15, 0.30]
        natural_center = 0.225
        deviation = abs(ratio - natural_center) / natural_center
        return float(np.clip(deviation * 2.0, 0.0, 1.0))

    def _dct_coefficient_analysis(self, gray: np.ndarray) -> float:
        """
        Analyze 8×8 DCT block coefficients (JPEG-style).
        GAN images show characteristic DCT coefficient distributions.
        Reference: Frank et al. (2020)
        """
        h, w = gray.shape
        block_size = 8
        ac_energies = []
        dc_values = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                dct_block = cv2.dct(block)
                dc_values.append(abs(dct_block[0, 0]))
                # AC energy (all except DC)
                ac = dct_block.copy()
                ac[0, 0] = 0
                ac_energies.append(np.sum(ac**2))

        if not ac_energies:
            return 0.5

        ac_std = float(np.std(ac_energies))
        dc_std = float(np.std(dc_values))

        # Real images: consistent AC/DC ratio; GANs show anomalies
        ratio = ac_std / (dc_std + 1e-6)

        # Anomaly detection: ratio < 2.0 or > 25.0
        if ratio < 2.0:
            self.violations.append(f"[Frequency] DCT AC/DC ratio too low ({ratio:.1f}) — face may be synthesized")
            return 0.72
        if ratio > 25.0:
            self.violations.append(f"[Frequency] DCT AC/DC ratio excessive ({ratio:.1f}) — GAN artifact pattern")
            return 0.68
        return float(np.clip((1 - (ratio - 2.0) / 23.0), 0.0, 1.0) * 0.4)

    def _checkerboard_detection(self, gray: np.ndarray) -> float:
        """
        Detect checkerboard/grid artifacts from transpose convolution in GANs.
        These appear as periodic patterns at specific frequencies (f=0.5 cycles/pixel).
        """
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Check for energy peaks at N/2 frequency (checkerboard signature)
        half_y, half_x = cy + h // 4, cx + w // 4
        corner_energy = 0.0
        corner_radius = max(h, w) // 16
        Y, X = np.ogrid[:h, :w]

        for fy, fx in [(h // 4, w // 4), (h // 4, 3*w // 4),
                       (3*h // 4, w // 4), (3*h // 4, 3*w // 4)]:
            mask = (Y - fy)**2 + (X - fx)**2 < corner_radius**2
            corner_energy += magnitude[mask].mean() if mask.any() else 0

        center_energy = magnitude[cy-corner_radius:cy+corner_radius, cx-corner_radius:cx+corner_radius].mean()

        if center_energy > 0:
            ratio = corner_energy / (center_energy * 4 + 1e-9)
            if ratio > 0.85:
                self.violations.append(f"[Frequency] Checkerboard pattern detected (ratio={ratio:.2f}) — GAN transpose conv artifact")
                return 0.82
            return float(np.clip(ratio * 0.6, 0.0, 1.0))
        return 0.3

    def _power_spectrum_slope(self, gray: np.ndarray) -> float:
        """
        Natural images follow 1/f^alpha power law (alpha≈2.0).
        GANs deviate: face-swap often has alpha > 2.5 (over-smooth) or < 1.5 (noisy).
        """
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        power = np.abs(fft_shift)**2
        h, w = power.shape
        cy, cx = h // 2, w // 2

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy)**2 + (X - cx)**2) + 1e-9

        # Bin by distance and compute mean power per band
        max_r = min(cy, cx)
        bands = np.linspace(1, max_r, 20)
        band_powers = []
        for i in range(len(bands) - 1):
            mask = (dist >= bands[i]) & (dist < bands[i+1])
            if mask.any():
                band_powers.append(np.log(power[mask].mean() + 1e-9))

        if len(band_powers) < 5:
            return 0.4

        # Fit slope via linear regression on log-log scale
        log_freqs = np.log(bands[:-len(band_powers)] + 1) if len(bands) > len(band_powers) else np.log(np.arange(1, len(band_powers)+1))
        log_freqs = np.log(np.linspace(1, max_r, len(band_powers)))
        slope, _ = np.polyfit(log_freqs, band_powers, 1)

        # Natural: slope ≈ -2.0; GANs deviate significantly
        expected_slope = -2.0
        deviation = abs(slope - expected_slope)

        if deviation > 1.2:
            self.violations.append(f"[Frequency] Power spectrum slope={slope:.2f} (expected≈-2.0) — unnatural frequency distribution")
            return min(0.75, deviation * 0.35)
        return float(np.clip(deviation * 0.25, 0.0, 0.5))

    def _temporal_consistency(self, frames: List[np.ndarray]) -> float:
        """
        Check if frequency characteristics are consistent across frames.
        Deepfakes often have flickering frequency artifacts.
        """
        if len(frames) < 3:
            return 0.3

        frame_scores = []
        for f in frames[:20]:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
            gray = cv2.resize(gray, (128, 128)).astype(np.float32)
            fft = np.abs(np.fft.fft2(gray))
            # Flatten and take mean energy per quadrant as fingerprint
            h, w = fft.shape
            q1 = fft[:h//2, :w//2].mean()
            q2 = fft[:h//2, w//2:].mean()
            q3 = fft[h//2:, :w//2].mean()
            q4 = fft[h//2:, w//2:].mean()
            frame_scores.append([q1, q2, q3, q4])

        frame_scores = np.array(frame_scores)
        # High variance across frames = frequency flickering
        variance = float(np.std(frame_scores, axis=0).mean())
        normalized = float(np.clip(variance / 1000.0, 0.0, 1.0))

        if normalized > 0.6:
            self.violations.append(f"[Frequency] Temporal frequency flickering (variance={variance:.0f}) — GAN regeneration artifact")
        return normalized

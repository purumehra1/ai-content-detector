from __future__ import annotations
"""
rPPG (Remote PhotoPlethysmoGraphy) Engine
──────────────────────────────────────────
Real human faces contain subtle, periodic color variations caused by blood flow.
This signal — called rPPG — pulses at ~60–100 BPM and is visible in RGB channels.

Deepfake-generated faces CANNOT replicate this physiological signal because:
  1. GANs generate each frame independently — no temporal heartbeat model
  2. Face-swaps mix source+target frequency bands incoherently
  3. Neural renderers don't model haemodynamic skin color changes

This engine:
  1. Extracts the green channel signal from the forehead/cheek ROI
  2. Bandpass filters to 0.75–3.0 Hz (45–180 BPM range)
  3. Computes signal quality via SNR and periodicity
  4. Validates that signal follows physiological plausibility

References:
  - Li et al. (2014) "Remote Heart Rate Measurement from Face Videos under Realistic Situations" CVPR
  - Hernandez-Ortega et al. (2020) "DeepFakesON-Phys: DeepFakes Detection based on Heart Rate Estimation" CVPRW
  - Lin et al. (2022) "Detecting Deepfake Videos Using Biological Signals" IEEE
"""
import numpy as np
import cv2
from typing import List, Optional


class RPPGEngine:
    """
    Detects absence of physiological rPPG signal — a unique signature of deepfakes.
    """
    SAMPLE_RATE = 25.0       # assumed FPS for rPPG processing
    LOW_BPM     = 45.0       # minimum realistic HR
    HIGH_BPM    = 180.0      # maximum realistic HR (exercise)
    NORMAL_BPM  = (55, 100)  # resting HR normal range

    def __init__(self):
        self.name = "rPPG Heart Rate"
        self.violations: List[str] = []
        self._bpm: Optional[float] = None
        self._snr: Optional[float] = None

    @property
    def detected_bpm(self) -> Optional[float]:
        return self._bpm

    # ── public API ─────────────────────────────────────────────────────────
    def analyze(self, frames: List[np.ndarray]) -> float:
        """
        Args:
            frames: list of BGR uint8 face-crop arrays
        Returns:
            score 0.0→1.0 (higher = more likely FAKE)
        """
        self.violations = []
        self._bpm = None
        self._snr = None

        if len(frames) < 30:
            # Need at least ~1.2 seconds of video at 25fps for meaningful rPPG
            return 0.35  # neutral — too short to assess

        # 1. Extract green-channel ROI signal
        signal = self._extract_rppg_signal(frames)
        if signal is None or len(signal) < 20:
            return 0.45

        # 2. Detrend and normalize
        signal = self._detrend(signal)
        signal = (signal - signal.mean()) / (signal.std() + 1e-9)

        # 3. FFT-based heart rate estimation
        hr_score, bpm, snr = self._estimate_heart_rate(signal)

        # 4. Check signal quality
        quality_score = self._check_signal_quality(signal, bpm, snr)

        # 5. Check channel correlation (R, G, B should correlate for real faces)
        channel_score = self._channel_correlation(frames)

        self._bpm = bpm
        self._snr = snr

        combined = 0.45 * hr_score + 0.35 * quality_score + 0.20 * channel_score
        return float(np.clip(combined, 0.0, 1.0))

    # ── signal extraction ──────────────────────────────────────────────────
    def _extract_rppg_signal(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """Extract mean green channel from forehead/upper-face ROI."""
        signals_r, signals_g, signals_b = [], [], []

        for frame in frames:
            if frame is None or frame.size == 0:
                continue
            h, w = frame.shape[:2]
            # Forehead ROI: upper 15-35% of face, center 30-70% width
            y1 = int(h * 0.10)
            y2 = int(h * 0.35)
            x1 = int(w * 0.25)
            x2 = int(w * 0.75)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            # BGR
            signals_b.append(roi[:, :, 0].mean())
            signals_g.append(roi[:, :, 1].mean())
            signals_r.append(roi[:, :, 2].mean())

        if len(signals_g) < 20:
            return None

        # Use CHROM method: project onto chrominance plane
        signals_g = np.array(signals_g)
        signals_r = np.array(signals_r)
        signals_b = np.array(signals_b)

        # Normalize by mean (remove illumination changes)
        r_norm = signals_r / (signals_r.mean() + 1e-9) - 1
        g_norm = signals_g / (signals_g.mean() + 1e-9) - 1
        b_norm = signals_b / (signals_b.mean() + 1e-9) - 1

        # CHROM: X = 3R - 2G, Y = 1.5R + G - 1.5B
        X = 3 * r_norm - 2 * g_norm
        Y = 1.5 * r_norm + g_norm - 1.5 * b_norm

        # Project to remove specular reflection
        alpha = X.std() / (Y.std() + 1e-9)
        rppg = X - alpha * Y
        return rppg

    def _detrend(self, signal: np.ndarray) -> np.ndarray:
        """Remove slow drift by subtracting polynomial fit."""
        x = np.arange(len(signal))
        p = np.polyfit(x, signal, deg=3)
        trend = np.polyval(p, x)
        return signal - trend

    # ── heart rate estimation ──────────────────────────────────────────────
    def _estimate_heart_rate(self, signal: np.ndarray) -> tuple:
        """
        Returns (fake_score, bpm, snr).
        Uses FFT to find dominant frequency in physiological HR band.
        """
        n = len(signal)
        fft = np.fft.rfft(signal * np.hanning(n))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.SAMPLE_RATE)
        power = np.abs(fft) ** 2

        low_hz = self.LOW_BPM / 60.0
        high_hz = self.HIGH_BPM / 60.0
        mask = (freqs >= low_hz) & (freqs <= high_hz)

        if not mask.any():
            return 0.6, None, 0.0

        hr_power = power[mask]
        hr_freqs = freqs[mask]
        peak_idx = np.argmax(hr_power)
        peak_freq = hr_freqs[peak_idx]
        bpm = peak_freq * 60.0

        # SNR: ratio of peak power to mean power in the HR band
        total_band_power = hr_power.sum() + 1e-9
        snr = float(hr_power[peak_idx] / (total_band_power / len(hr_power)))
        self._snr = snr

        # Score based on SNR and physiological plausibility
        score = 0.0

        # If SNR < 1.5, signal is basically noise — no periodic heartbeat detected
        if snr < 1.5:
            score += 0.65
            self.violations.append(f"[rPPG] No periodic heartbeat detected (SNR={snr:.1f}) — deepfake likely lacks physiological signal")
        elif snr < 3.0:
            score += 0.35
            self.violations.append(f"[rPPG] Weak heartbeat signal (SNR={snr:.1f}, BPM={bpm:.0f}) — may be synthetic")
        else:
            # Good signal — check if BPM is in normal range
            if not (self.NORMAL_BPM[0] <= bpm <= self.NORMAL_BPM[1]):
                score += 0.50
                self.violations.append(f"[rPPG] Detected BPM={bpm:.0f} outside normal range ({self.NORMAL_BPM[0]}–{self.NORMAL_BPM[1]}) — physiologically implausible")
            else:
                score = 0.10  # clean signal, normal BPM = likely real

        return float(np.clip(score, 0.0, 1.0)), float(bpm), float(snr)

    def _check_signal_quality(self, signal: np.ndarray, bpm: Optional[float], snr: Optional[float]) -> float:
        """Check signal periodicity, stationarity, and noise level."""
        if len(signal) < 10:
            return 0.5

        # Split signal into 3 segments and check consistency
        segs = np.array_split(signal, 3)
        seg_vars = [s.var() for s in segs if len(s) > 3]

        if not seg_vars:
            return 0.5

        # Variance should be stable (not wildly fluctuating) for real rPPG
        var_cv = np.std(seg_vars) / (np.mean(seg_vars) + 1e-9)
        if var_cv > 1.5:
            self.violations.append(f"[rPPG] Highly non-stationary signal (CV={var_cv:.2f}) — unstable facial skin")
            return float(np.clip(var_cv * 0.4, 0.0, 0.8))

        # Autocorrelation: real heartbeat signal has peaks at lag=HR_period
        if bpm and bpm > 0:
            period_frames = int(self.SAMPLE_RATE * 60.0 / bpm)
            if 3 <= period_frames < len(signal) // 2:
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[len(signal)-1:]
                autocorr /= (autocorr[0] + 1e-9)
                peak_corr = autocorr[period_frames] if period_frames < len(autocorr) else 0
                if peak_corr < 0.15:
                    self.violations.append(f"[rPPG] Low autocorrelation at HR period (r={peak_corr:.2f}) — non-periodic signal")
                    return 0.55

        return float(np.clip(var_cv * 0.2, 0.0, 0.4))

    def _channel_correlation(self, frames: List[np.ndarray]) -> float:
        """
        In real faces, R, G, B channels are correlated during heartbeat.
        Deepfakes generated per-channel lack this physiological coupling.
        """
        r_means, g_means, b_means = [], [], []
        for frame in frames[:60]:
            if frame is None:
                continue
            h, w = frame.shape[:2]
            roi = frame[int(h*0.1):int(h*0.4), int(w*0.25):int(w*0.75)]
            if roi.size == 0:
                continue
            b_means.append(roi[:, :, 0].mean())
            g_means.append(roi[:, :, 1].mean())
            r_means.append(roi[:, :, 2].mean())

        if len(r_means) < 10:
            return 0.3

        r = np.array(r_means)
        g = np.array(g_means)
        b = np.array(b_means)

        # Compute pairwise correlations
        def safe_corr(a, b):
            if a.std() < 1e-6 or b.std() < 1e-6:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        rg = safe_corr(r, g)
        rb = safe_corr(r, b)
        gb = safe_corr(g, b)
        mean_corr = (abs(rg) + abs(rb) + abs(gb)) / 3.0

        # Real faces: mean inter-channel corr > 0.6 (all respond to same heartbeat)
        if mean_corr < 0.35:
            self.violations.append(f"[rPPG] Low RGB channel correlation ({mean_corr:.2f}) — channels not driven by common physiological signal")
            return float(np.clip((0.35 - mean_corr) * 2.0, 0.0, 0.8))

        return float(np.clip((0.6 - mean_corr) * 1.5, 0.0, 0.5))

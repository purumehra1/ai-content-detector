"""
Audio-Visual Consistency Engine
────────────────────────────────
Verifies synchronization between speech audio and lip/jaw movement.
Accepts audio as numpy array (not file path).
"""
from __future__ import annotations
import numpy as np
import cv2
import mediapipe as mp
from typing import List, Optional

from utils.face_utils import get_landmarks, get_lip_opening

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

mp_face_mesh = mp.solutions.face_mesh

SILENCE_RMS     = 0.01
SPEECH_LIP_MIN  = 0.02
MAX_ONSET_DELAY = 3


class AudioVisualEngine:

    def __init__(self):
        self.name = "AudioVisual"
        self.violations: List[str] = []

    def analyze(self,
                frames: List[np.ndarray],
                audio_array: Optional[np.ndarray] = None,
                sample_rate: int = 16000) -> float:
        """
        Args:
            frames: BGR video frames
            audio_array: 1-D numpy float32 audio signal (may be None)
            sample_rate: sample rate of audio_array
        Returns:
            fake probability score 0.0→1.0
        """
        self.violations = []

        # ── Extract lip openings ───────────────────────────────────────────
        lip_openings = []
        try:
            with mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                min_detection_confidence=0.5
            ) as mesh:
                for frame in frames:
                    if frame is None or frame.size == 0:
                        lip_openings.append(0.0)
                        continue
                    lm = get_landmarks(frame, mesh)
                    lip_openings.append(get_lip_opening(lm) if lm is not None else 0.0)
        except Exception:
            return 0.35

        lip_arr = np.array(lip_openings)

        # ── No audio path ─────────────────────────────────────────────────
        if audio_array is None or not LIBROSA_AVAILABLE:
            penalty = 0.0
            if float(np.var(lip_arr)) < 0.0001 and float(np.mean(lip_arr)) > 0.02:
                self.violations.append("[AV] Lips static but open — unnatural stiffness")
                penalty += 0.20
            return float(np.clip(penalty, 0.0, 1.0))

        # ── Build RMS timeline aligned to frames ──────────────────────────
        rms_arr = self._build_rms_timeline(audio_array, sample_rate, len(frames))
        if rms_arr is None:
            return 0.30

        n = min(len(lip_arr), len(rms_arr))
        lip_arr = lip_arr[:n]
        rms_arr = rms_arr[:n]
        has_speech = float(np.sqrt(np.mean(audio_array**2))) > SILENCE_RMS

        penalty = 0.0

        # 1. Cross-correlation
        if np.std(lip_arr) > 1e-6 and np.std(rms_arr) > 1e-6:
            corr = float(np.corrcoef(lip_arr, rms_arr)[0, 1])
        else:
            corr = 0.0
        if corr < 0.15 and has_speech:
            self.violations.append(
                f"[AV] Audio-lip correlation={corr:.3f} (expected >0.30) — desynchronized")
            penalty += 0.35

        # 2. Onset delay
        speech_onset = next((i for i, r in enumerate(rms_arr) if r > SILENCE_RMS * 3), None)
        lip_onset    = next((i for i, l in enumerate(lip_arr) if l > SPEECH_LIP_MIN), None)
        if speech_onset is not None and lip_onset is not None:
            delay = abs(lip_onset - speech_onset)
            if delay > MAX_ONSET_DELAY:
                self.violations.append(
                    f"[AV] Onset delay={delay} frames (natural ≤{MAX_ONSET_DELAY}) — voice dubbing indicator")
                penalty += min(0.30, delay * 0.06)

        # 3. Silent periods — lips should be still
        silent_mask = rms_arr < SILENCE_RMS
        if np.sum(silent_mask) > 5:
            active_in_silence = np.sum(lip_arr[silent_mask] > SPEECH_LIP_MIN)
            ratio = active_in_silence / np.sum(silent_mask)
            if ratio > 0.40:
                self.violations.append(
                    f"[AV] Lips moving in {ratio*100:.0f}% of silent frames — animation mismatch")
                penalty += 0.25

        # 4. Loud audio, mouth closed
        loud_mask = rms_arr > SILENCE_RMS * 5
        if np.sum(loud_mask) > 5:
            closed_in_speech = np.sum(lip_arr[loud_mask] < SPEECH_LIP_MIN * 0.5)
            ratio = closed_in_speech / np.sum(loud_mask)
            if ratio > 0.50:
                self.violations.append(
                    f"[AV] Mouth closed in {ratio*100:.0f}% of loud audio frames — possible dubbing")
                penalty += 0.30

        return float(np.clip(penalty, 0.0, 1.0))

    def _build_rms_timeline(self,
                            audio: np.ndarray,
                            sr: int,
                            n_frames: int,
                            video_fps: float = 25.0) -> Optional[np.ndarray]:
        try:
            frame_dur = 1.0 / video_fps
            samp_per_frame = max(1, int(sr * frame_dur))
            rms_vals = []
            for i in range(n_frames):
                start = i * samp_per_frame
                end   = start + samp_per_frame
                seg   = audio[start:end] if end <= len(audio) else audio[start:]
                rms_vals.append(float(np.sqrt(np.mean(seg**2))) if len(seg) > 0 else 0.0)
            return np.array(rms_vals)
        except Exception:
            return None

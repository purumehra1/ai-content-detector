"""
Audio-Visual Consistency Engine
────────────────────────────────
Verifies synchronization between audio speech and visual lip/jaw movement.
Checks:
  • RMS speech energy timeline vs lip opening timeline (correlation)
  • Speech onset-to-jaw-movement delay (should be ≤2 frames)
  • Silence periods vs face motion (face should be relaxed when silent)
  • Audio presence with no visual mouth movement (mismatch = fake)
Output: av_consistency_score (0=consistent/real, 1=inconsistent/fake)
"""
import numpy as np
import cv2
import mediapipe as mp
import os
from utils.face_utils import get_landmarks, get_lip_opening

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

mp_face_mesh = mp.solutions.face_mesh

SILENCE_THRESHOLD_RMS = 0.01
SPEECH_LIP_OPEN_MIN = 0.02     # lip opening during speech
MAX_ONSET_DELAY_FRAMES = 3     # max frames between audio onset and jaw movement


def _extract_rms_timeline(audio_path: str, frame_count: int, video_fps: float) -> np.ndarray | None:
    """Extract per-frame RMS energy from audio. Returns array of shape (frame_count,)."""
    if not LIBROSA_AVAILABLE or audio_path is None:
        return None
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Split audio into frame-aligned segments
        frame_duration = 1.0 / video_fps
        samples_per_frame = int(sr * frame_duration)
        rms_vals = []
        for i in range(frame_count):
            start = i * samples_per_frame
            end = start + samples_per_frame
            segment = y[start:end] if end <= len(y) else y[start:]
            rms = float(np.sqrt(np.mean(segment ** 2))) if len(segment) > 0 else 0.0
            rms_vals.append(rms)
        return np.array(rms_vals)
    except Exception:
        return None


def _extract_spectral_features(audio_path: str) -> dict:
    """Extract spectral features for scene-audio consistency check."""
    if not LIBROSA_AVAILABLE or audio_path is None:
        return {}
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        features = {}
        features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = mfcc.mean(axis=1).tolist()
        features["rms_mean"] = float(np.sqrt(np.mean(y ** 2)))
        features["has_speech"] = features["rms_mean"] > SILENCE_THRESHOLD_RMS
        return features
    except Exception:
        return {}


class AudioVisualEngine:
    WEIGHT = 0.10

    def analyze(self, frames: list, audio_path: str | None, video_fps: float = 25.0) -> dict:
        violations = []
        penalty = 0.0

        # ── Extract lip openings ──────────────────────────────────────────────
        lip_openings = []
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
        ) as mesh:
            for frame in frames:
                lm = get_landmarks(frame, mesh)
                lip_openings.append(get_lip_opening(lm) if lm is not None else 0.0)

        lip_arr = np.array(lip_openings)

        # ── No audio — limited analysis ───────────────────────────────────────
        if audio_path is None or not LIBROSA_AVAILABLE:
            violations.append("AUDIO UNAVAILABLE: Could not extract audio track — audio-visual sync check skipped")
            # Still check lip motion consistency with itself
            lip_var = float(np.var(lip_arr))
            if lip_var < 0.0001 and np.mean(lip_arr) > 0.02:
                violations.append("LIP FREEZE: Lips are open but not moving — unnatural stiffness")
                penalty += 0.2
            return {"score": max(0.0, penalty), "violations": violations, "confidence": 0.3, "av_correlation": None}

        # ── Extract audio RMS timeline ────────────────────────────────────────
        rms_arr = _extract_rms_timeline(audio_path, len(frames), video_fps)
        spectral = _extract_spectral_features(audio_path)

        if rms_arr is None:
            violations.append("AUDIO PROCESSING ERROR: librosa failed to process audio")
            return {"score": 0.5, "violations": violations, "confidence": 0.0, "av_correlation": None}

        # Align lengths
        min_len = min(len(lip_arr), len(rms_arr))
        lip_arr = lip_arr[:min_len]
        rms_arr = rms_arr[:min_len]

        # ── 1. Cross-correlation: audio energy vs lip opening ─────────────────
        if np.std(lip_arr) > 1e-6 and np.std(rms_arr) > 1e-6:
            corr = np.corrcoef(lip_arr, rms_arr)[0, 1]
        else:
            corr = 0.0

        if corr < 0.15 and spectral.get("has_speech", False):
            violations.append(f"AUDIO-LIP DESYNC: Speech energy ↔ lip movement correlation={corr:.3f} (expected >0.30 for real video)")
            penalty += 0.35

        # ── 2. Speech onset → lip delay check ────────────────────────────────
        speech_onset = None
        lip_onset = None
        for i, r in enumerate(rms_arr):
            if r > SILENCE_THRESHOLD_RMS * 3:
                speech_onset = i
                break
        for i, l in enumerate(lip_arr):
            if l > SPEECH_LIP_OPEN_MIN:
                lip_onset = i
                break

        if speech_onset is not None and lip_onset is not None:
            delay = abs(lip_onset - speech_onset)
            if delay > MAX_ONSET_DELAY_FRAMES:
                violations.append(f"ONSET DELAY: {delay} frames between audio speech onset and lip opening (natural ≤{MAX_ONSET_DELAY_FRAMES} frames)")
                penalty += min(0.30, delay * 0.06)

        # ── 3. Silent periods — face should be relaxed ────────────────────────
        silent_mask = rms_arr < SILENCE_THRESHOLD_RMS
        if np.sum(silent_mask) > 5:
            lip_during_silence = lip_arr[silent_mask]
            active_during_silence = np.sum(lip_during_silence > SPEECH_LIP_OPEN_MIN)
            ratio = active_during_silence / len(lip_during_silence)
            if ratio > 0.4:
                violations.append(f"SPEECH DURING SILENCE: Lips actively moving in {ratio*100:.0f}% of silent audio segments — face animation not matching audio")
                penalty += 0.25

        # ── 4. High-energy audio with no mouth movement ───────────────────────
        loud_mask = rms_arr > SILENCE_THRESHOLD_RMS * 5
        if np.sum(loud_mask) > 5:
            lip_during_speech = lip_arr[loud_mask]
            silent_mouth = np.sum(lip_during_speech < SPEECH_LIP_OPEN_MIN * 0.5)
            ratio = silent_mouth / len(lip_during_speech)
            if ratio > 0.5:
                violations.append(f"MOUTH CLOSED DURING SPEECH: Mouth closed in {ratio*100:.0f}% of loud audio frames — possible voice dubbing")
                penalty += 0.30

        fake_score = min(1.0, penalty)
        result = {
            "score": fake_score,
            "label": "FAKE" if fake_score > 0.5 else "REAL",
            "av_correlation": round(float(corr), 4),
            "has_speech": spectral.get("has_speech", False),
            "rms_mean": round(spectral.get("rms_mean", 0.0), 4),
            "violations": violations,
            "confidence": min(1.0, abs(fake_score - 0.5) * 2),
        }
        if spectral.get("has_speech"):
            result["spectral_centroid_hz"] = round(spectral.get("spectral_centroid", 0), 1)
        return result

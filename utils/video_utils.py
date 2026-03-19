from __future__ import annotations
"""Video utility functions — frame extraction, audio separation."""
import cv2
import numpy as np
import subprocess
import os
import tempfile
from pathlib import Path


def extract_frames(video_path: str, max_frames: int = 60, target_fps: int = 10) -> list[np.ndarray]:
    """Extract evenly-spaced frames from video. Returns list of BGR frames."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    step = max(1, int(src_fps / target_fps))
    indices = list(range(0, total, step))[:max_frames]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def extract_all_frames(video_path: str, max_frames: int = 120) -> tuple[list[np.ndarray], float]:
    """Extract consecutive frames for temporal analysis. Returns (frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    frames = []
    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += step
    cap.release()
    return frames, fps


def extract_audio(video_path: str) -> str | None:
    """Extract audio track to a temp WAV file. Returns path or None."""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            tmp.name
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and os.path.getsize(tmp.name) > 0:
            return tmp.name
        os.unlink(tmp.name)
        return None
    except Exception:
        return None


def extract_audio_array(video_path: str, sr: int = 16000):
    """
    Extract audio as numpy array directly (no temp file needed).
    Returns (audio_array, sample_rate) or (None, sr) on failure.
    Uses imageio[ffmpeg] if available, then falls back to temp-file approach.
    """
    # Method 1: imageio / ffmpeg pipe
    try:
        import imageio
        reader = imageio.get_reader(video_path, "ffmpeg")
        meta = reader.get_meta_data()
        audio_fps = meta.get("audio_fps", sr)
        # imageio doesn't expose raw audio easily — fall through to method 2
        reader.close()
    except Exception:
        pass

    # Method 2: librosa (handles most formats via soundfile / ffmpeg backend)
    try:
        import librosa
        audio_path = extract_audio(video_path)
        if audio_path:
            y, sr_detected = librosa.load(audio_path, sr=sr, mono=True)
            try:
                os.unlink(audio_path)
            except Exception:
                pass
            return y, sr_detected
    except Exception:
        pass

    # Method 3: soundfile on extracted wav
    try:
        import soundfile as sf
        audio_path = extract_audio(video_path)
        if audio_path:
            y, sr_detected = sf.read(audio_path, dtype="float32", always_2d=False)
            try:
                os.unlink(audio_path)
            except Exception:
                pass
            return y, sr_detected
    except Exception:
        pass

    return None, sr


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["total_frames"] / max(info["fps"], 1)
    cap.release()
    return info

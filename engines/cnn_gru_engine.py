"""
CNN + GRU Engine
────────────────
• CNN (ViT / EfficientNet via HuggingFace): per-frame deepfake probability
• GRU temporal layer: analyzes sequence of CNN scores for temporal consistency
• Output: deepfake_score (0=real, 1=fake)
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import pipeline
import cv2
from utils.face_utils import detect_face_crop


class TemporalGRU(nn.Module):
    """Lightweight GRU to model temporal evolution of per-frame CNN scores."""
    def __init__(self, input_size=4, hidden_size=16, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))  # use last hidden state


class CNNGRUEngine:
    """
    Combines a pre-trained deepfake frame classifier (CNN proxy via ViT)
    with a GRU that analyzes temporal patterns in frame-level scores.
    """

    MODEL_ID = "dima806/deepfake_vs_real_image_detection"
    WEIGHT = 0.40

    def __init__(self, device: str = "auto"):
        self.device_str = device
        self._pipe = None
        self._gru = None
        self._init_gru()

    def _lazy_load(self):
        if self._pipe is None:
            dev = 0 if torch.cuda.is_available() else -1
            try:
                self._pipe = pipeline(
                    "image-classification",
                    model=self.MODEL_ID,
                    device=dev,
                )
            except Exception:
                # Fallback to lighter model
                self._pipe = pipeline(
                    "image-classification",
                    model="umm-maybe/AI-image-detector",
                    device=dev,
                )

    def _init_gru(self):
        self._gru = TemporalGRU(input_size=4, hidden_size=16)
        self._gru.eval()

    def _classify_frame(self, frame_bgr: np.ndarray) -> dict:
        """Classify a single frame. Returns {fake_prob, real_prob, label}."""
        self._lazy_load()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        results = self._pipe(img)
        scores = {r["label"].lower(): r["score"] for r in results}
        fake_prob = scores.get("fake", scores.get("artificial", scores.get("deepfake", 0.0)))
        real_prob = scores.get("real", 1.0 - fake_prob)
        # Normalize
        total = fake_prob + real_prob + 1e-9
        return {"fake_prob": fake_prob / total, "real_prob": real_prob / total}

    def _temporal_analysis(self, frame_scores: list[dict]) -> float:
        """
        GRU analysis on sequence of per-frame features:
        [fake_prob, real_prob, delta_fake, variance_running]
        Returns temporal inconsistency score (0=consistent, 1=inconsistent).
        """
        if len(frame_scores) < 3:
            return 0.5

        fakes = np.array([s["fake_prob"] for s in frame_scores])
        reals = np.array([s["real_prob"] for s in frame_scores])
        deltas = np.diff(fakes, prepend=fakes[0])
        running_var = np.array([np.var(fakes[max(0, i-4):i+1]) for i in range(len(fakes))])

        # Feature matrix: (seq_len, 4)
        features = np.stack([fakes, reals, deltas, running_var], axis=1).astype(np.float32)
        x = torch.tensor(features).unsqueeze(0)  # (1, seq, 4)

        with torch.no_grad():
            temporal_score = self._gru(x).item()

        # Spike detection: sudden large swings indicate instability
        spike_score = float(np.mean(np.abs(deltas) > 0.15))

        return (temporal_score + spike_score) / 2.0

    def analyze(self, frames: list[np.ndarray]) -> dict:
        """
        Run CNN+GRU analysis on a list of video frames.
        Returns: {score, label, frame_scores, temporal_score, violations}
        """
        if not frames:
            return {"score": 0.5, "label": "UNKNOWN", "violations": [], "confidence": 0.0}

        frame_scores = []
        face_detection_rate = 0

        for frame in frames:
            # Prefer face-cropped region for better accuracy
            face = detect_face_crop(frame)
            target = face if face is not None else frame
            if face is not None:
                face_detection_rate += 1
            score = self._classify_frame(target)
            frame_scores.append(score)

        avg_fake = float(np.mean([s["fake_prob"] for s in frame_scores]))
        avg_real = float(np.mean([s["real_prob"] for s in frame_scores]))
        temporal_score = self._temporal_analysis(frame_scores)
        face_rate = face_detection_rate / len(frames)

        # Combined score: frame-level avg + temporal inconsistency
        combined = avg_fake * 0.7 + temporal_score * 0.3

        violations = []
        if avg_fake > 0.65:
            violations.append(f"CNN detected deepfake artifacts in {avg_fake*100:.0f}% confidence")
        if temporal_score > 0.4:
            violations.append(f"Temporal inconsistency detected (score={temporal_score:.2f})")
        if face_rate < 0.5:
            violations.append(f"Face detected in only {face_rate*100:.0f}% of frames — possible occlusion or face-swap boundary")

        return {
            "score": combined,
            "label": "FAKE" if combined > 0.5 else "REAL",
            "avg_fake_prob": avg_fake,
            "temporal_score": temporal_score,
            "frame_count": len(frames),
            "face_detection_rate": face_rate,
            "violations": violations,
            "confidence": abs(combined - 0.5) * 2,
        }

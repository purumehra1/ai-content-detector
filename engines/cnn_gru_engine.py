"""
CNN + GRU Engine
────────────────
• CNN (ViT via HuggingFace): per-frame deepfake probability
• GRU temporal layer: analyzes sequence of CNN scores for temporal consistency
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import List, Optional
import cv2

from utils.face_utils import detect_face_crop


class TemporalGRU(nn.Module):
    """Lightweight GRU to model temporal evolution of per-frame CNN scores."""
    def __init__(self, input_size: int = 4, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


class CNNGRUEngine:
    MODEL_ID = "dima806/deepfake_vs_real_image_detection"
    FALLBACK_ID = "umm-maybe/AI-image-detector"

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self._pipe = None
        self.violations: List[str] = []
        self._gru = TemporalGRU(input_size=4, hidden_size=16)
        self._gru.eval()

    def _lazy_load(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
            dev = 0 if torch.cuda.is_available() else -1
            try:
                self._pipe = pipeline("image-classification", model=self.MODEL_ID, device=dev)
            except Exception:
                self._pipe = pipeline("image-classification", model=self.FALLBACK_ID, device=dev)
        except Exception as e:
            self._pipe = None

    def _classify_frame(self, frame_bgr: np.ndarray) -> dict:
        if self._pipe is None:
            return {"fake_prob": 0.5, "real_prob": 0.5}
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            results = self._pipe(img)
            scores = {r["label"].lower(): r["score"] for r in results}
            fake_prob = scores.get("fake", scores.get("artificial", scores.get("deepfake", 0.5)))
            real_prob = scores.get("real", 1.0 - fake_prob)
            total = fake_prob + real_prob + 1e-9
            return {"fake_prob": fake_prob / total, "real_prob": real_prob / total}
        except Exception:
            return {"fake_prob": 0.5, "real_prob": 0.5}

    def _temporal_analysis(self, frame_scores: List[dict]) -> float:
        if len(frame_scores) < 3:
            return 0.5
        fakes = np.array([s["fake_prob"] for s in frame_scores])
        reals = np.array([s["real_prob"] for s in frame_scores])
        deltas = np.diff(fakes, prepend=fakes[0])
        running_var = np.array([np.var(fakes[max(0, i-4):i+1]) for i in range(len(fakes))])
        features = np.stack([fakes, reals, deltas, running_var], axis=1).astype(np.float32)
        x = torch.tensor(features).unsqueeze(0)
        with torch.no_grad():
            temporal_score = float(self._gru(x).item())
        spike_score = float(np.mean(np.abs(deltas) > 0.15))
        return float(np.clip((temporal_score + spike_score) / 2.0, 0.0, 1.0))

    def analyze(self, frames: List[np.ndarray]) -> float:
        """Returns fake probability score 0.0→1.0."""
        self.violations = []
        if not frames:
            return 0.5

        self._lazy_load()

        frame_scores = []
        face_hits = 0
        for frame in frames:
            if frame is None or frame.size == 0:
                continue
            face = detect_face_crop(frame)
            if face is not None:
                face_hits += 1
                target = face
            else:
                target = frame
            score = self._classify_frame(target)
            frame_scores.append(score)

        if not frame_scores:
            return 0.5

        avg_fake = float(np.mean([s["fake_prob"] for s in frame_scores]))
        temporal_score = self._temporal_analysis(frame_scores)
        face_rate = face_hits / len(frames)

        combined = float(np.clip(avg_fake * 0.70 + temporal_score * 0.30, 0.0, 1.0))

        if avg_fake > 0.65:
            self.violations.append(f"[CNN-GRU] CNN detected deepfake artifacts — {avg_fake*100:.0f}% confidence")
        if temporal_score > 0.40:
            self.violations.append(f"[CNN-GRU] Temporal score inconsistency (score={temporal_score:.2f})")
        if face_rate < 0.50:
            self.violations.append(f"[CNN-GRU] Face detected in only {face_rate*100:.0f}% of frames")

        return combined

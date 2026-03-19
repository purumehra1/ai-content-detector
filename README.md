# 🛡️ DeepFake Guardian

**Hybrid multi-modal deepfake detection using nature-aligned causal consistency.**

Instead of asking *"Does this look fake?"*, this system asks **"Does this behave like a real human?"**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)](https://mediapipe.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-orange)](https://streamlit.io)

---

## 🔬 Detection Architecture

```
Input Video
    │
    ├─ Frame Extraction (OpenCV + FFmpeg)
    ├─ Audio Extraction (FFmpeg → WAV)
    │
    ├─────────────────── PARALLEL ENGINES ───────────────────────┐
    │  [1] CNN+GRU Engine      → visual artifacts + temporal (40%) │
    │  [2] Biological Motion   → velocity, jerk, freeze (20%)      │
    │  [3] Teeth Consistency   → structure stability (15%)         │
    │  [4] Hand/Finger         → anatomy, joint angles (15%)       │
    │  [5] Audio-Visual Sync   → lip-speech correlation (5%)       │
    │  [6] Causal Rules        → 8 cause-effect laws (5%)          │
    │  [7] Active Stability    → perturbation testing (modifier)   │
    └─────────────────────────────────────────────────────────────┘
    │
    └─ Weighted Fusion Layer
           │
           └─ VERDICT: REAL / SUSPICIOUS / FAKE + Explanation
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Or CLI
python deepfake_detector.py path/to/video.mp4
```

## 🧠 Engine Details

### 1. CNN + GRU Engine (Weight: 40%)
- **CNN**: `dima806/deepfake_vs_real_image_detection` ViT model — per-frame deepfake probability
- **GRU**: Custom temporal layer analyzes sequence of CNN scores for inconsistency spikes
- Detects: GAN artifacts, temporal score instability, low face detection rate

### 2. Biological Motion Engine (Weight: 20%)
- Tracks 478 MediaPipe Face Mesh landmarks across frames
- Computes velocity, acceleration, jerk per frame
- Detects: Freeze phases (>60% near-zero velocity), teleport motion (>30px/frame jumps), impossible jerk values, abnormal blink rates

### 3. Teeth Consistency Engine (Weight: 15%)
- Analyzes teeth region using lip landmark polygon
- Perceptual hash (pHash) comparison across frames
- Detects: Structural flickering, brightness variance >25, texture instability — GANs struggle to maintain teeth consistency

### 4. Hand / Finger Engine (Weight: 15%)
- MediaPipe 21-point hand landmarks per hand
- Checks anatomical joint angles (10°–175° natural range)
- Detects: Fused fingers, impossible joint angles, velocity discontinuities

### 5. Audio-Visual Sync Engine (Weight: 5%)
- librosa RMS energy timeline vs lip opening correlation
- Speech onset to jaw movement delay check
- Detects: Low AV correlation (<0.15), onset delay >3 frames, mouth closed during loud speech

### 6. Causal Consistency Engine (Weight: 5%)
8 natural cause-effect rules:
| Rule | Check |
|---|---|
| 1 | Speech → Jaw Movement |
| 2 | Speech Pause → Blink |
| 3 | Facial Symmetry Stability |
| 4 | Lip Speed Physical Limit |
| 5 | Gaze Continuity |
| 6 | Bilateral Symmetry |
| 7 | Lip Opening Rate |
| 8 | Micro-Expression Duration |

### 7. Active Stability Engine (Modifier)
Applies 6 perturbations (brightness ±25%, JPEG quality 40, Gaussian blur, noise, contrast) and measures score sensitivity. Real videos degrade gracefully; deepfakes show erratic sensitivity.

## 📊 Fusion Formula

```
Final Score = 0.40 × CNN-GRU
            + 0.20 × Motion
            + 0.15 × Teeth
            + 0.15 × Hand
            + 0.05 × Audio-Visual
            + 0.05 × Causal
            + stability_modifier (±0.025)

FAKE       if score ≥ 0.50
SUSPICIOUS if score ≥ 0.40
REAL       if score < 0.40
```

## 📁 Project Structure

```
deepfake-guardian/
├── app.py                      # Streamlit web app
├── deepfake_detector.py        # Main pipeline (CLI + API)
├── requirements.txt
├── engines/
│   ├── cnn_gru_engine.py       # Visual + temporal detection
│   ├── motion_engine.py        # Biological motion physics
│   ├── teeth_engine.py         # Teeth structural consistency
│   ├── audio_visual_engine.py  # Audio-lip synchronization
│   ├── hand_engine.py          # Hand/finger anatomy
│   ├── stability_engine.py     # Perturbation testing
│   └── causal_engine.py        # Rule-based causal verification
├── fusion/
│   └── weighted_fusion.py      # Weighted score fusion
└── utils/
    ├── video_utils.py           # Frame + audio extraction
    └── face_utils.py            # MediaPipe landmark utilities
```

## 🔑 Key Innovations

1. **Nature-Aligned Causal Consistency** — biological cause-effect rules that all real humans obey
2. **Teeth as Immutable Evidence** — pHash structural comparison; GANs fail to maintain teeth
3. **Motion Physics Validation** — velocity/acceleration/jerk within biomechanical limits
4. **Active Stability Testing** — novel perturbation-based robustness measurement
5. **Explainable AI** — every violation named, every engine scored, no black box

## 📝 Output

```json
{
  "verdict": "FAKE",
  "confidence": "HIGH",
  "confidence_pct": 82.4,
  "final_score": 0.9120,
  "engine_scores": {
    "CNN-GRU (0.40×)": 0.87,
    "Motion (0.20×)": 0.73,
    "Teeth (0.15×)": 0.65,
    "Hand (0.15×)": 0.30,
    "Audio-Visual (0.05×)": 0.61,
    "Causal (0.05×)": 0.55
  },
  "violations": [
    "[CNN-GRU] CNN detected deepfake artifacts in 87% confidence",
    "[Motion] FREEZE: Face nearly static for 68% of frames",
    "[Teeth] TEETH FLICKER: Visible in only 23% of open-mouth frames",
    "[Causal] RULE 4 VIOLATED — Lip Physics: 5 frames with impossible speed"
  ]
}
```

## 👤 Author

**Puru Mehra** · [github.com/purumehra1](https://github.com/purumehra1)

*Upgraded from [ai-content-detector](https://github.com/purumehra1/ai-content-detector)*

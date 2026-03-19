"""
Weighted Score Fusion Engine
─────────────────────────────
Combines scores from all 10 detection engines into a single verdict.

Fusion Formula:
  Final = Σ (weight_i × score_i) + stability_modifier

Weights (v3.1 — revised with 10 engines):
  CNN+GRU         : 0.25  (primary spatial+temporal classifier)
  Frequency Domain: 0.15  (GAN artifact detection)
  Biological Motion: 0.12 (physics-based validation)
  Teeth           : 0.10  (structural biological evidence)
  rPPG            : 0.10  (physiological heartbeat signal)
  Eye Consistency : 0.08  (blink, pupil, corneal reflection)
  Head Pose       : 0.07  (3D consistency)
  Hand/Finger     : 0.06  (anatomy validation)
  Skin Texture    : 0.05  (GAN skin rendering)
  Audio-Visual    : 0.04  (lip-sync)
  Causal Rules    : 0.04  (rule-based)
  Stability       : modifier ±0.03
  ─────────────────────────────────────────
  Total base      : 1.06 → normalized

Classification thresholds:
  ≥ 0.72  →  FAKE     HIGH confidence
  ≥ 0.55  →  FAKE     MEDIUM confidence
  ≥ 0.42  →  SUSPICIOUS (manual review recommended)
  < 0.42  →  REAL
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Engine weights (must sum to ~1.0 before normalization)
ENGINE_WEIGHTS: Dict[str, float] = {
    "CNN-GRU":          0.25,
    "Frequency":        0.15,
    "Motion":           0.12,
    "Teeth":            0.10,
    "rPPG":             0.10,
    "Eye":              0.08,
    "HeadPose":         0.07,
    "Hand":             0.06,
    "SkinTexture":      0.05,
    "AudioVisual":      0.04,
    "Causal":           0.04,
}
_WEIGHT_SUM = sum(ENGINE_WEIGHTS.values())
NORMALIZED_WEIGHTS = {k: v / _WEIGHT_SUM for k, v in ENGINE_WEIGHTS.items()}

# Classification thresholds
THRESHOLD_FAKE_HIGH   = 0.72
THRESHOLD_FAKE_MEDIUM = 0.55
THRESHOLD_SUSPICIOUS  = 0.42


@dataclass
class FusionResult:  # noqa: E302
    label: str                        # REAL | SUSPICIOUS | FAKE
    confidence: str                   # HIGH | MEDIUM | LOW
    confidence_pct: float             # 0–100
    final_score: float                # 0.0–1.0
    engine_scores: Dict[str, float]   # raw score per engine
    engine_contributions: Dict[str, float]  # weighted contribution per engine
    all_violations: List[str]
    summary: str
    elapsed_seconds: float = 0.0
    video_info: dict = field(default_factory=dict)
    bpm_detected: Optional[float] = None
    blink_rate: Optional[float] = None
    dominant_engine: str = ""
    # XAI fields (populated by DeepFakeDetector after fuse())
    xai_region_scores: dict = field(default_factory=dict)
    xai_heatmap: object = None        # numpy ndarray or None
    xai_explanation: str = ""
    face_crops: list = field(default_factory=list)


def fuse(engine_results: Dict[str, float],
         violations: List[str],
         stability_modifier: float = 0.0,
         elapsed: float = 0.0,
         video_info: dict = None,
         bpm: Optional[float] = None,
         blink_rate: Optional[float] = None) -> FusionResult:
    """
    Args:
        engine_results: {engine_name: score_0_to_1}
        violations: flat list of violation strings from all engines
        stability_modifier: ±0.03 from stability engine
        elapsed: analysis time in seconds
        video_info: dict with width/height/fps/duration/total_frames
        bpm: detected heart rate BPM (from rPPG engine)
        blink_rate: detected blink rate per minute
    Returns:
        FusionResult
    """
    # Compute weighted sum using available engines
    weighted_sum = 0.0
    weight_used = 0.0
    contributions = {}

    for engine_key, weight in NORMALIZED_WEIGHTS.items():
        if engine_key in engine_results:
            score = float(engine_results[engine_key])
            contribution = weight * score
            contributions[engine_key] = contribution
            weighted_sum += contribution
            weight_used += weight
        else:
            contributions[engine_key] = 0.0

    # Normalize for missing engines
    if weight_used > 0:
        weighted_sum = weighted_sum / weight_used
    else:
        weighted_sum = 0.5

    # Apply stability modifier
    final_score = float(weighted_sum + stability_modifier)
    final_score = max(0.0, min(1.0, final_score))

    # Determine verdict
    if final_score >= THRESHOLD_FAKE_HIGH:
        label = "FAKE"
        confidence = "HIGH"
        conf_pct = 50.0 + (final_score - THRESHOLD_FAKE_HIGH) / (1.0 - THRESHOLD_FAKE_HIGH) * 49.0 + 1.0
    elif final_score >= THRESHOLD_FAKE_MEDIUM:
        label = "FAKE"
        confidence = "MEDIUM"
        conf_pct = 50.0 + (final_score - THRESHOLD_FAKE_MEDIUM) / (THRESHOLD_FAKE_HIGH - THRESHOLD_FAKE_MEDIUM) * 28.0
    elif final_score >= THRESHOLD_SUSPICIOUS:
        label = "SUSPICIOUS"
        confidence = "LOW"
        conf_pct = 40.0 + (final_score - THRESHOLD_SUSPICIOUS) / (THRESHOLD_FAKE_MEDIUM - THRESHOLD_SUSPICIOUS) * 20.0
    else:
        label = "REAL"
        confidence = "HIGH" if final_score < 0.25 else "MEDIUM"
        conf_pct = 50.0 + (0.42 - final_score) / 0.42 * 49.0

    conf_pct = float(max(1.0, min(99.0, conf_pct)))

    # Dominant engine (highest contributor)
    dominant = max(contributions, key=contributions.get) if contributions else "N/A"

    # Build display engine scores with display names
    display_scores = {}
    name_map = {
        "CNN-GRU":    "🧠 CNN+GRU (0.25×)",
        "Frequency":  "📡 Frequency Domain (0.15×)",
        "Motion":     "🏃 Biological Motion (0.12×)",
        "Teeth":      "🦷 Teeth Consistency (0.10×)",
        "rPPG":       "❤️ rPPG Heart Rate (0.10×)",
        "Eye":        "👁️ Eye Consistency (0.08×)",
        "HeadPose":   "🗿 Head Pose (0.07×)",
        "Hand":       "🤚 Hand Anatomy (0.06×)",
        "SkinTexture":"🔬 Skin Texture (0.05×)",
        "AudioVisual":"🔊 Audio-Visual (0.04×)",
        "Causal":     "⚖️ Causal Rules (0.04×)",
    }
    for k, v in engine_results.items():
        display_name = name_map.get(k, k)
        display_scores[display_name] = round(float(v), 4)
    display_scores["⚡ FINAL SCORE"] = round(final_score, 4)

    # Build summary message
    top_violations = violations[:3]
    if label == "FAKE":
        if top_violations:
            reason = top_violations[0].split("]")[-1].strip() if "]" in top_violations[0] else top_violations[0]
            summary = f"Video is {confidence.lower()}-confidence fake. Primary indicator: {reason}"
        else:
            summary = f"Video classified as FAKE with {confidence.lower()} confidence (score: {final_score:.3f})."
    elif label == "SUSPICIOUS":
        summary = f"Video shows suspicious patterns. Manual review recommended. Score: {final_score:.3f}."
    else:
        summary = f"No significant deepfake indicators detected. Video appears authentic (score: {final_score:.3f})."

    return FusionResult(
        label=label,
        confidence=confidence,
        confidence_pct=conf_pct,
        final_score=final_score,
        engine_scores=display_scores,
        engine_contributions=contributions,
        all_violations=violations,
        summary=summary,
        elapsed_seconds=elapsed,
        video_info=video_info or {},
        bpm_detected=bpm,
        blink_rate=blink_rate,
        dominant_engine=name_map.get(dominant, dominant),
    )

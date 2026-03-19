"""
Weighted Score Fusion Engine — v3.2
─────────────────────────────────────
Smart fusion that:
  1. Excludes engines that failed / returned neutral scores
  2. Applies agreement boost when multiple engines agree on FAKE
  3. Applies CNN anchor — if CNN says >0.70 fake, floor is raised
  4. Properly calibrated thresholds
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# Engine weights
ENGINE_WEIGHTS: Dict[str, float] = {
    "CNN-GRU":     0.28,
    "Frequency":   0.14,
    "Motion":      0.12,
    "Teeth":       0.09,
    "rPPG":        0.09,
    "Eye":         0.08,
    "HeadPose":    0.07,
    "Hand":        0.05,
    "SkinTexture": 0.04,
    "AudioVisual": 0.02,
    "Causal":      0.02,
}

# Engines that return neutral when they can't run (exclude from avg if neutral)
# "Neutral range" = engine returned without running meaningfully
NEUTRAL_RANGE = (0.28, 0.38)   # scores in this tight band = likely "no data" neutral

# Engines with known neutral returns when input is missing
ALWAYS_INCLUDE = {"CNN-GRU", "Frequency", "SkinTexture"}  # always reliable

# Classification thresholds (calibrated)
THRESHOLD_FAKE_HIGH   = 0.68
THRESHOLD_FAKE_MEDIUM = 0.50
THRESHOLD_SUSPICIOUS  = 0.38


@dataclass
class FusionResult:
    label:              str
    confidence:         str
    confidence_pct:     float
    final_score:        float
    engine_scores:      Dict[str, float]
    engine_contributions: Dict[str, float]
    all_violations:     List[str]
    summary:            str
    elapsed_seconds:    float = 0.0
    video_info:         dict  = field(default_factory=dict)
    bpm_detected:       Optional[float] = None
    blink_rate:         Optional[float] = None
    dominant_engine:    str   = ""
    engines_ran:        int   = 0
    # XAI fields
    xai_region_scores:  dict  = field(default_factory=dict)
    xai_heatmap:        object = None
    xai_explanation:    str   = ""
    face_crops:         list  = field(default_factory=list)


def _is_neutral(score: float, engine_name: str) -> bool:
    """Return True if the score looks like a neutral/failed-engine default."""
    if engine_name in ALWAYS_INCLUDE:
        return False
    # Scores very close to 0.3, 0.35, 0.4, 0.45, 0.5 = likely default returns
    known_neutrals = {0.30, 0.35, 0.40, 0.45, 0.50}
    for n in known_neutrals:
        if abs(score - n) < 0.02:
            return True
    return False


def fuse(engine_results:    Dict[str, float],
         violations:        List[str],
         stability_modifier: float = 0.0,
         elapsed:           float  = 0.0,
         video_info:        dict   = None,
         bpm:               Optional[float] = None,
         blink_rate:        Optional[float] = None) -> FusionResult:

    if not engine_results:
        return _make_result("REAL", "LOW", 50.0, 0.5, {}, {}, violations,
                            "No engines ran — cannot determine authenticity.",
                            elapsed, video_info or {}, bpm, blink_rate, 0)

    # ── 1. Filter out neutral/failed engines ─────────────────────────────
    active_scores: Dict[str, float] = {}
    skipped_engines = []

    for name, raw_score in engine_results.items():
        score = float(raw_score)
        if _is_neutral(score, name):
            skipped_engines.append(name)
        else:
            active_scores[name] = score

    # Always include CNN-GRU, Frequency, SkinTexture even if neutral
    for name in ALWAYS_INCLUDE:
        if name in engine_results and name not in active_scores:
            active_scores[name] = float(engine_results[name])

    if not active_scores:
        active_scores = {k: float(v) for k, v in engine_results.items()}

    # ── 2. Weighted average over ACTIVE engines only ──────────────────────
    weighted_sum  = 0.0
    weight_used   = 0.0
    contributions = {}

    # Normalize weights to active engines only
    active_weight_sum = sum(ENGINE_WEIGHTS.get(k, 0.03) for k in active_scores)
    if active_weight_sum == 0:
        active_weight_sum = len(active_scores)

    for name, score in active_scores.items():
        w = ENGINE_WEIGHTS.get(name, 0.03)
        norm_w = w / active_weight_sum
        weighted_sum += norm_w * score
        weight_used  += norm_w
        contributions[name] = round(norm_w * score, 4)

    # Also add zero contributions for inactive engines (for display)
    for name in engine_results:
        if name not in contributions:
            contributions[name] = 0.0

    base_score = float(weighted_sum / max(weight_used, 1e-9))

    # ── 3. CNN Anchor — strong single-engine evidence ─────────────────────
    cnn_score = float(active_scores.get("CNN-GRU", 0.5))
    freq_score = float(active_scores.get("Frequency", 0.5))

    # If CNN is very confident fake, raise the floor
    if cnn_score > 0.75:
        cnn_floor = 0.55 + (cnn_score - 0.75) * 0.60
        base_score = max(base_score, cnn_floor)
    elif cnn_score > 0.65:
        cnn_floor = 0.45 + (cnn_score - 0.65) * 0.50
        base_score = max(base_score, cnn_floor)

    # Frequency engine strong signal
    if freq_score > 0.70:
        base_score = max(base_score, 0.50)

    # ── 4. Agreement Boost ────────────────────────────────────────────────
    # Count how many active engines strongly flag fake
    strong_fake_votes = sum(1 for s in active_scores.values() if s > 0.60)
    mild_fake_votes   = sum(1 for s in active_scores.values() if s > 0.45)
    n_active = len(active_scores)

    agreement_boost = 0.0
    if n_active >= 4:
        if strong_fake_votes >= 4:
            agreement_boost = 0.08  # strong consensus
        elif strong_fake_votes >= 3:
            agreement_boost = 0.05
        elif strong_fake_votes >= 2:
            agreement_boost = 0.03
        if mild_fake_votes >= int(n_active * 0.65):
            agreement_boost += 0.03  # majority mild agreement

    base_score += agreement_boost

    # ── 5. Apply stability modifier ───────────────────────────────────────
    final_score = float(np.clip(base_score + stability_modifier, 0.0, 1.0))

    # ── 6. Classify ───────────────────────────────────────────────────────
    if final_score >= THRESHOLD_FAKE_HIGH:
        label      = "FAKE"
        confidence = "HIGH"
        conf_pct   = 60.0 + (final_score - THRESHOLD_FAKE_HIGH) / (1.0 - THRESHOLD_FAKE_HIGH) * 38.0
    elif final_score >= THRESHOLD_FAKE_MEDIUM:
        label      = "FAKE"
        confidence = "MEDIUM"
        conf_pct   = 50.0 + (final_score - THRESHOLD_FAKE_MEDIUM) / \
                     (THRESHOLD_FAKE_HIGH - THRESHOLD_FAKE_MEDIUM) * 22.0
    elif final_score >= THRESHOLD_SUSPICIOUS:
        label      = "SUSPICIOUS"
        confidence = "LOW"
        conf_pct   = 40.0 + (final_score - THRESHOLD_SUSPICIOUS) / \
                     (THRESHOLD_FAKE_MEDIUM - THRESHOLD_SUSPICIOUS) * 20.0
    else:
        label      = "REAL"
        confidence = "HIGH" if final_score < 0.22 else "MEDIUM"
        conf_pct   = 50.0 + (THRESHOLD_SUSPICIOUS - final_score) / THRESHOLD_SUSPICIOUS * 48.0

    conf_pct = float(max(1.0, min(99.0, conf_pct)))

    # ── 7. Build display scores ───────────────────────────────────────────
    name_map = {
        "CNN-GRU":     "🧠 CNN+GRU (0.28×)",
        "Frequency":   "📡 Frequency Domain (0.14×)",
        "Motion":      "🏃 Biological Motion (0.12×)",
        "Teeth":       "🦷 Teeth Consistency (0.09×)",
        "rPPG":        "❤️  rPPG Heart Rate (0.09×)",
        "Eye":         "👁️  Eye Consistency (0.08×)",
        "HeadPose":    "🗿 Head Pose 3D (0.07×)",
        "Hand":        "🤚 Hand Anatomy (0.05×)",
        "SkinTexture": "🔬 Skin Texture (0.04×)",
        "AudioVisual": "🔊 Audio-Visual (0.02×)",
        "Causal":      "⚖️  Causal Rules (0.02×)",
    }
    display_scores: Dict[str, float] = {}
    for k, v in engine_results.items():
        display_name = name_map.get(k, k)
        display_scores[display_name] = round(float(v), 4)
    display_scores["⚡ FINAL SCORE"] = round(final_score, 4)

    # Dominant engine
    dominant = max(contributions, key=lambda k: contributions[k]) if contributions else "N/A"
    dominant_display = name_map.get(dominant, dominant)

    # Summary
    top_viols = violations[:3]
    if label == "FAKE":
        if top_viols:
            reason = top_viols[0].split("]")[-1].strip() if "]" in top_viols[0] else top_viols[0]
            summary = (f"Video classified as {confidence.lower()}-confidence FAKE "
                       f"(score: {final_score:.3f}). "
                       f"Primary indicator: {reason}")
        else:
            summary = (f"Video classified as FAKE — score {final_score:.3f}. "
                       f"Dominant engine: {dominant_display}.")
    elif label == "SUSPICIOUS":
        summary = (f"Suspicious patterns detected (score: {final_score:.3f}). "
                   f"Manual review recommended. "
                   f"{strong_fake_votes}/{n_active} engines flagged anomalies.")
    else:
        summary = (f"No significant deepfake indicators. "
                   f"Video appears authentic (score: {final_score:.3f}). "
                   f"{n_active} engines ran, {strong_fake_votes} flagged minor anomalies.")

    return _make_result(label, confidence, conf_pct, final_score,
                        display_scores, contributions, violations, summary,
                        elapsed, video_info or {}, bpm, blink_rate,
                        len(active_scores), dominant_display)


def _make_result(label, confidence, conf_pct, final_score,
                 display_scores, contributions, violations, summary,
                 elapsed, video_info, bpm, blink_rate, engines_ran,
                 dominant="") -> FusionResult:
    return FusionResult(
        label=label,
        confidence=confidence,
        confidence_pct=float(conf_pct),
        final_score=float(final_score),
        engine_scores=display_scores,
        engine_contributions=contributions,
        all_violations=violations,
        summary=summary,
        elapsed_seconds=float(elapsed),
        video_info=video_info,
        bpm_detected=bpm,
        blink_rate=blink_rate,
        dominant_engine=dominant,
        engines_ran=engines_ran,
    )

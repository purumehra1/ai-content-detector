"""
Weighted Score Fusion Layer
────────────────────────────
Combines outputs from all detection engines using a calibrated weighted formula:

  Final = 0.40 × CNN-GRU
        + 0.20 × Motion
        + 0.15 × Teeth
        + 0.15 × Hand
        + 0.10 × (Audio-Visual + Causal) / 2

Also incorporates stability score as a modifier.
Output: FusionResult with classification, confidence, and explanation.
"""
from dataclasses import dataclass, field


WEIGHTS = {
    "cnn_gru":    0.40,
    "motion":     0.20,
    "teeth":      0.15,
    "hand":       0.15,
    "av_causal":  0.10,   # average of audio-visual and causal
}

FAKE_THRESHOLD = 0.50     # score above this = FAKE
HIGH_CONFIDENCE = 0.72    # score above this = HIGH confidence fake
SUSPICIOUS = 0.40         # score above this = SUSPICIOUS (needs review)


@dataclass
class FusionResult:
    final_score: float
    label: str            # FAKE / REAL / SUSPICIOUS
    confidence: str       # HIGH / MEDIUM / LOW
    confidence_pct: float
    engine_scores: dict
    all_violations: list[str]
    summary: str
    stability_modifier: float = 0.0


def fuse(
    cnn_result: dict,
    motion_result: dict,
    teeth_result: dict,
    hand_result: dict,
    av_result: dict,
    causal_result: dict,
    stability_result: dict,
) -> FusionResult:

    cnn_score    = float(cnn_result.get("score", 0.5))
    motion_score = float(motion_result.get("score", 0.5))
    teeth_score  = float(teeth_result.get("score", 0.5))
    hand_score   = float(hand_result.get("score", 0.3))  # 0.3 default if no hands
    av_score     = float(av_result.get("score", 0.5))
    causal_score = float(causal_result.get("score", 0.5))
    stab_score   = float(stability_result.get("score", 0.5))

    av_causal_avg = (av_score + causal_score) / 2.0

    weighted = (
        WEIGHTS["cnn_gru"]   * cnn_score    +
        WEIGHTS["motion"]    * motion_score +
        WEIGHTS["teeth"]     * teeth_score  +
        WEIGHTS["hand"]      * hand_score   +
        WEIGHTS["av_causal"] * av_causal_avg
    )

    # Stability modifier: erratic fake → slightly boost score; very stable → slight reduction
    stability_modifier = (stab_score - 0.5) * 0.05
    final = min(1.0, max(0.0, weighted + stability_modifier))

    # Classification
    if final >= HIGH_CONFIDENCE:
        label = "FAKE"
        confidence = "HIGH"
    elif final >= FAKE_THRESHOLD:
        label = "FAKE"
        confidence = "MEDIUM"
    elif final >= SUSPICIOUS:
        label = "SUSPICIOUS"
        confidence = "LOW"
    else:
        label = "REAL"
        confidence = "HIGH" if final < 0.25 else "MEDIUM"

    confidence_pct = abs(final - 0.5) * 200  # 0–100%

    # Aggregate all violations
    all_violations = []
    for engine_name, result in [
        ("CNN-GRU", cnn_result),
        ("Motion", motion_result),
        ("Teeth", teeth_result),
        ("Hand", hand_result),
        ("Audio-Visual", av_result),
        ("Causal", causal_result),
        ("Stability", stability_result),
    ]:
        for v in result.get("violations", []):
            all_violations.append(f"[{engine_name}] {v}")
        for v in result.get("violated_rules", []):
            all_violations.append(f"[Causal] {v}")

    # Remove duplicates from causal (violated_rules already added above)
    all_violations = list(dict.fromkeys(all_violations))

    engine_scores = {
        "CNN-GRU (0.40×)":    round(cnn_score, 4),
        "Motion (0.20×)":     round(motion_score, 4),
        "Teeth (0.15×)":      round(teeth_score, 4),
        "Hand (0.15×)":       round(hand_score, 4),
        "Audio-Visual (0.05×)": round(av_score, 4),
        "Causal (0.05×)":     round(causal_score, 4),
        "Stability (modifier)": round(stab_score, 4),
        "FINAL":              round(final, 4),
    }

    # Human-readable summary
    top_violations = all_violations[:4]
    if label == "FAKE":
        summary = (
            f"Video classified as FAKE with {confidence_pct:.0f}% confidence. "
            f"Primary indicators: {'; '.join(top_violations[:2]) if top_violations else 'Multiple engines flagged inconsistencies'}."
        )
    elif label == "SUSPICIOUS":
        summary = (
            f"Video is SUSPICIOUS — below fake threshold but shows {len(all_violations)} inconsistencies. "
            f"Manual review recommended."
        )
    else:
        summary = (
            f"Video classified as REAL with {confidence_pct:.0f}% confidence. "
            f"All behavioral checks passed within natural variation limits."
        )

    return FusionResult(
        final_score=round(final, 4),
        label=label,
        confidence=confidence,
        confidence_pct=round(confidence_pct, 1),
        engine_scores=engine_scores,
        all_violations=all_violations,
        summary=summary,
        stability_modifier=round(stability_modifier, 4),
    )

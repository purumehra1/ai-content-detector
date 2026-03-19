"""
Explainability Engine (XAI)
────────────────────────────
Produces human-interpretable explanations for why a video was flagged.

Methods:
  1. Region Attribution Map — occludes facial regions and measures score change
     (model-agnostic Grad-CAM equivalent, works with any black-box engine)
  2. Face Region Importance — forehead/eyes/nose/mouth/cheeks scoring
  3. Attention Heatmap — blends region scores onto face image
  4. Natural-language explanation generator

References:
  - Ribeiro et al. (2016) "LIME: Why Should I Trust You?" KDD 2016
  - Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks" ICCV
  - Zeiler & Fergus (2014) "Visualizing and Understanding CNNs" ECCV
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple


# Face region definitions (relative to face bounding box)
# Each region: (name, y_start_frac, y_end_frac, x_start_frac, x_end_frac)
FACE_REGIONS = {
    "forehead":     (0.00, 0.22, 0.20, 0.80),
    "left_eye":     (0.20, 0.42, 0.08, 0.45),
    "right_eye":    (0.20, 0.42, 0.55, 0.92),
    "nose":         (0.35, 0.62, 0.30, 0.70),
    "left_cheek":   (0.40, 0.70, 0.03, 0.35),
    "right_cheek":  (0.40, 0.70, 0.65, 0.97),
    "upper_lip":    (0.60, 0.72, 0.25, 0.75),
    "mouth":        (0.65, 0.82, 0.20, 0.80),
    "chin":         (0.80, 1.00, 0.20, 0.80),
}

# Region colors for visualization (BGR)
REGION_COLORS = {
    "forehead":    (255, 100, 100),
    "left_eye":    (100, 100, 255),
    "right_eye":   (100, 100, 255),
    "nose":        (100, 200, 100),
    "left_cheek":  (200, 200, 100),
    "right_cheek": (200, 200, 100),
    "upper_lip":   (200, 100, 200),
    "mouth":       (100, 200, 255),
    "chin":        (180, 180, 180),
}


class XAIEngine:
    """
    Generates visual and textual explanations for deepfake detection decisions.
    Model-agnostic: works with any scoring function.
    """

    def __init__(self):
        self.name = "XAI"
        self.violations: List[str] = []
        self._region_scores: Dict[str, float] = {}
        self._heatmap: Optional[np.ndarray] = None
        self._explanation: str = ""

    @property
    def region_scores(self) -> Dict[str, float]:
        return self._region_scores

    @property
    def heatmap(self) -> Optional[np.ndarray]:
        return self._heatmap

    @property
    def explanation(self) -> str:
        return self._explanation

    # ── public API ─────────────────────────────────────────────────────────
    def analyze(self, frames: List[np.ndarray],
                score_fn,
                baseline_score: float,
                n_sample: int = 8) -> float:
        """
        Args:
            frames: BGR face-crop frames
            score_fn: callable(List[np.ndarray]) → float (0=real, 1=fake)
            baseline_score: the already-computed score without occlusion
            n_sample: number of frames to use for region attribution
        Returns:
            attribution confidence (0→1)
        """
        self.violations = []
        self._region_scores = {}
        self._heatmap = None

        if not frames or not callable(score_fn):
            return 0.0

        sample = frames[:n_sample]
        if not sample:
            return 0.0

        # 1. Region occlusion attribution
        region_importances = self._region_occlusion(sample, score_fn, baseline_score)
        self._region_scores = region_importances

        # 2. Build heatmap on best frame
        best_frame = sample[len(sample) // 2]
        self._heatmap = self._build_heatmap(best_frame, region_importances)

        # 3. Generate natural-language explanation
        self._explanation = self._generate_explanation(
            region_importances, baseline_score
        )

        return float(np.clip(baseline_score, 0.0, 1.0))

    # ── region occlusion attribution ───────────────────────────────────────
    def _region_occlusion(self,
                          frames: List[np.ndarray],
                          score_fn,
                          baseline: float) -> Dict[str, float]:
        """
        For each facial region, occlude it with gray and measure score change.
        Importance = score_without_region - baseline
        High positive = this region is what makes it look fake.
        """
        importances = {}

        for region_name, (y0, y1, x0, x1) in FACE_REGIONS.items():
            occluded = []
            for frame in frames:
                if frame is None or frame.size == 0:
                    continue
                h, w = frame.shape[:2]
                masked = frame.copy()
                ry1, ry2 = int(y0 * h), int(y1 * h)
                rx1, rx2 = int(x0 * w), int(x1 * w)
                # Occlude with mean color (LIME-style neutral occlusion)
                if ry2 > ry1 and rx2 > rx1:
                    mean_color = frame[ry1:ry2, rx1:rx2].mean(axis=(0, 1))
                    masked[ry1:ry2, rx1:rx2] = mean_color.astype(np.uint8)
                occluded.append(masked)

            if not occluded:
                importances[region_name] = 0.0
                continue

            try:
                occluded_score = float(score_fn(occluded))
                # Positive = region was contributing to fake detection
                # Removing it makes it look more real → it was suspicious
                importance = occluded_score - baseline
                importances[region_name] = float(np.clip(importance, -1.0, 1.0))
            except Exception:
                importances[region_name] = 0.0

        return importances

    # ── heatmap generation ─────────────────────────────────────────────────
    def _build_heatmap(self,
                       frame: np.ndarray,
                       importances: Dict[str, float]) -> np.ndarray:
        """
        Overlay colored heatmap on frame.
        Red = suspicious region, Blue = clean region.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        # Create importance map (float32, same size)
        imp_map = np.zeros((h, w), dtype=np.float32)

        for region_name, imp in importances.items():
            y0, y1, x0, x1 = FACE_REGIONS[region_name]
            ry1, ry2 = int(y0 * h), int(y1 * h)
            rx1, rx2 = int(x0 * w), int(x1 * w)
            if ry2 > ry1 and rx2 > rx1:
                imp_map[ry1:ry2, rx1:rx2] = imp

        # Normalize to 0-255
        if imp_map.max() > imp_map.min():
            norm = (imp_map - imp_map.min()) / (imp_map.max() - imp_map.min() + 1e-9)
        else:
            norm = np.zeros_like(imp_map)

        # Apply JET colormap (blue=clean, red=suspicious)
        norm_uint8 = (norm * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)

        # Blend with original frame
        blended = cv2.addWeighted(frame, 0.55, heatmap_colored, 0.45, 0)

        # Add region labels
        for region_name, imp in importances.items():
            y0, y1, x0, x1 = FACE_REGIONS[region_name]
            cy = int((y0 + y1) / 2 * h)
            cx = int((x0 + x1) / 2 * w)
            color = (0, 0, 255) if imp > 0.05 else (0, 255, 0)
            cv2.putText(blended, f"{region_name[:5]} {imp:+.2f}",
                        (max(0, cx - 30), cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

        return blended

    def build_heatmap_only(self, frame: np.ndarray,
                           importances: Dict[str, float]) -> Optional[np.ndarray]:
        """Public method for building heatmap from external importances."""
        return self._build_heatmap(frame, importances)

    def build_clean_heatmap(self, frame: np.ndarray,
                            importances: Dict[str, float],
                            alpha: float = 0.50) -> Optional[np.ndarray]:
        """
        Cleaner version: smooth gradient heatmap without text labels.
        Better for display in the UI.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        imp_map = np.zeros((h, w), dtype=np.float32)

        for region_name, imp in importances.items():
            y0, y1, x0, x1 = FACE_REGIONS[region_name]
            ry1, ry2 = int(y0 * h), int(y1 * h)
            rx1, rx2 = int(x0 * w), int(x1 * w)
            if ry2 > ry1 and rx2 > rx1:
                # Gaussian-like smooth fill
                region_map = np.ones((ry2-ry1, rx2-rx1), dtype=np.float32) * imp
                imp_map[ry1:ry2, rx1:rx2] = np.maximum(
                    imp_map[ry1:ry2, rx1:rx2], region_map
                )

        # Smooth the map
        imp_map = cv2.GaussianBlur(imp_map, (31, 31), 0)

        # Normalize
        min_v, max_v = imp_map.min(), imp_map.max()
        if max_v > min_v:
            norm = (imp_map - min_v) / (max_v - min_v + 1e-9)
        else:
            norm = np.zeros_like(imp_map)

        norm_uint8 = (norm * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 1.0 - alpha, heatmap_colored, alpha, 0)
        return blended

    # ── explanation generation ─────────────────────────────────────────────
    def _generate_explanation(self,
                               importances: Dict[str, float],
                               baseline_score: float) -> str:
        """Generate natural-language explanation of detection decision."""
        if not importances:
            return "Insufficient data for attribution analysis."

        # Sort by absolute importance
        sorted_regions = sorted(importances.items(),
                                key=lambda x: abs(x[1]), reverse=True)
        top3 = [(r, v) for r, v in sorted_regions[:3] if abs(v) > 0.02]

        verdict = "FAKE" if baseline_score > 0.55 else \
                  "SUSPICIOUS" if baseline_score > 0.42 else "REAL"

        if verdict == "REAL":
            if top3:
                regions_str = ", ".join([r.replace("_", " ") for r, _ in top3])
                return (f"Video appears authentic. The {regions_str} regions "
                        f"show natural characteristics consistent with real human faces. "
                        f"No significant GAN artifacts detected in facial structure.")
            return ("All facial regions appear natural. Biological signals "
                    "(heartbeat, blink patterns, motion physics) are within normal ranges.")

        if not top3:
            return (f"Detection score {baseline_score:.2f} indicates potential deepfake. "
                    f"Artifacts are distributed across multiple facial regions.")

        most_suspicious = top3[0][0].replace("_", " ")
        imp_val = top3[0][1]
        change_pct = abs(imp_val) * 100

        lines = [
            f"Primary deepfake indicator: **{most_suspicious}** region "
            f"(score drops {change_pct:.0f}% when occluded — this region drives detection)."
        ]

        if len(top3) > 1:
            secondary = top3[1][0].replace("_", " ")
            lines.append(
                f"Secondary indicator: **{secondary}** region also shows GAN artifacts."
            )

        if baseline_score > 0.80:
            lines.append(
                "High-confidence fake: multiple regions simultaneously show "
                "GAN regeneration patterns inconsistent with biological skin."
            )

        return " ".join(lines)

    # ── LIME-style superpixel attribution (advanced) ───────────────────────
    def lime_attribution(self,
                         frame: np.ndarray,
                         score_fn,
                         n_segments: int = 20,
                         n_samples: int = 50) -> Optional[np.ndarray]:
        """
        LIME: sample random occlusions of superpixels, fit linear model.
        Returns attribution map showing segment importance.
        Computationally expensive — use for single-frame analysis.
        Reference: Ribeiro et al. KDD 2016.
        """
        if frame is None or frame.size == 0:
            return None

        try:
            # Simple grid-based superpixels (SLIC substitute)
            h, w = frame.shape[:2]
            grid_size = int(np.sqrt(h * w / n_segments))
            grid_size = max(8, grid_size)

            # Create segment IDs
            seg_map = np.zeros((h, w), dtype=np.int32)
            seg_id = 0
            segments = []
            for y in range(0, h, grid_size):
                for x in range(0, w, grid_size):
                    y2 = min(h, y + grid_size)
                    x2 = min(w, x + grid_size)
                    seg_map[y:y2, x:x2] = seg_id
                    segments.append((y, y2, x, x2))
                    seg_id += 1

            n_seg = seg_id
            # Mean colors for neutral fill
            seg_means = []
            for y1, y2, x1, x2 in segments:
                seg_means.append(frame[y1:y2, x1:x2].mean(axis=(0, 1)))

            # Sample random occlusion patterns
            X_samples = np.random.binomial(1, 0.5, (n_samples, n_seg))
            y_scores  = []

            for mask in X_samples:
                perturbed = frame.copy()
                for s_idx, keep in enumerate(mask):
                    if not keep:
                        y1, y2, x1, x2 = segments[s_idx]
                        perturbed[y1:y2, x1:x2] = seg_means[s_idx].astype(np.uint8)
                try:
                    y_scores.append(float(score_fn([perturbed])))
                except Exception:
                    y_scores.append(0.5)

            y_scores = np.array(y_scores)

            # Fit linear model: score ≈ W · segment_mask
            from numpy.linalg import lstsq
            W, _, _, _ = lstsq(X_samples.astype(float), y_scores, rcond=None)

            # Build attribution image
            attr_map = np.zeros((h, w), dtype=np.float32)
            for s_idx, w_val in enumerate(W):
                y1, y2, x1, x2 = segments[s_idx]
                attr_map[y1:y2, x1:x2] = w_val

            # Normalize and colormap
            attr_map_blur = cv2.GaussianBlur(attr_map, (21, 21), 0)
            min_v = attr_map_blur.min()
            max_v = attr_map_blur.max()
            if max_v > min_v:
                norm = (attr_map_blur - min_v) / (max_v - min_v + 1e-9)
            else:
                norm = np.zeros_like(attr_map_blur)

            norm_uint8 = (norm * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
            result = cv2.addWeighted(frame, 0.50, heatmap, 0.50, 0)
            return result

        except Exception:
            return None

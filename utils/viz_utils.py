"""
Visualization Utilities
────────────────────────
Helpers for generating heatmaps, annotated frames, and chart data.
"""
from __future__ import annotations
import numpy as np
import cv2
import base64
from typing import Optional, Dict, List
from io import BytesIO
from PIL import Image


def frame_to_b64(frame_bgr: np.ndarray, quality: int = 85) -> str:
    """Convert BGR numpy frame to base64 JPEG string for Streamlit HTML."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def frames_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy frame to PIL Image."""
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def annotate_frame(frame: np.ndarray,
                   label: str,
                   score: float,
                   color: tuple = (0, 0, 255)) -> np.ndarray:
    """Draw verdict label + score on a frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    # Background bar
    cv2.rectangle(out, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.rectangle(out, (0, 0), (int(w * score), 8), color, -1)
    text = f"{label}  {score:.2%}"
    cv2.putText(out, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_face_regions(frame: np.ndarray,
                      region_scores: Dict[str, float],
                      alpha: float = 0.35) -> np.ndarray:
    """Draw colored rectangles for each face region, colored by importance."""
    if frame is None or frame.size == 0:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()

    REGIONS = {
        "forehead":    (0.00, 0.22, 0.20, 0.80),
        "left_eye":    (0.20, 0.42, 0.08, 0.45),
        "right_eye":   (0.20, 0.42, 0.55, 0.92),
        "nose":        (0.35, 0.62, 0.30, 0.70),
        "left_cheek":  (0.40, 0.70, 0.03, 0.35),
        "right_cheek": (0.40, 0.70, 0.65, 0.97),
        "upper_lip":   (0.60, 0.72, 0.25, 0.75),
        "mouth":       (0.65, 0.82, 0.20, 0.80),
        "chin":        (0.80, 1.00, 0.20, 0.80),
    }

    for region, (y0, y1, x0, x1) in REGIONS.items():
        score = region_scores.get(region, 0.0)
        ry1, ry2 = int(y0 * h), int(y1 * h)
        rx1, rx2 = int(x0 * w), int(x1 * w)
        if ry2 > ry1 and rx2 > rx1:
            # Red for suspicious, green for clean
            intensity = abs(score)
            if score > 0.02:
                color = (0, int(50 * (1 - intensity)), int(255 * intensity))  # BGR red
            elif score < -0.02:
                color = (0, int(200 * intensity), 0)  # BGR green
            else:
                color = (80, 80, 80)
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), color, -1)
            # Label
            short = region[:5]
            cv2.putText(overlay, f"{short} {score:+.2f}",
                        (rx1 + 3, ry1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def make_comparison_strip(frames: List[np.ndarray],
                           heatmaps: List[Optional[np.ndarray]],
                           max_frames: int = 4,
                           size: tuple = (200, 200)) -> Optional[np.ndarray]:
    """Create a side-by-side strip: original frames | heatmap frames."""
    if not frames:
        return None
    n = min(max_frames, len(frames))
    step = max(1, len(frames) // n)
    selected = [frames[i * step] for i in range(n) if frames[i * step] is not None]
    selected_hm = [heatmaps[i * step] if heatmaps and i * step < len(heatmaps)
                   else None for i in range(n)]

    row1, row2 = [], []
    for f in selected:
        row1.append(cv2.resize(f, size))
    for hm, orig in zip(selected_hm, selected):
        row2.append(cv2.resize(hm if hm is not None else orig, size))

    if not row1:
        return None
    strip_top = np.hstack(row1)
    strip_bot = np.hstack(row2)
    divider = np.zeros((6, strip_top.shape[1], 3), dtype=np.uint8)
    return np.vstack([strip_top, divider, strip_bot])


def get_plotly_colors(scores: Dict[str, float]) -> List[str]:
    """Return plotly color list based on scores."""
    colors = []
    for v in scores.values():
        if float(v) > 0.60:
            colors.append("#ff4444")
        elif float(v) > 0.42:
            colors.append("#ffcc00")
        else:
            colors.append("#00e676")
    return colors

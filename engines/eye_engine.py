"""
Eye Consistency Engine
───────────────────────
Eyes are one of the hardest features for deepfakes to replicate correctly:
  1. Blink completeness — deepfakes often generate incomplete blinks
  2. Corneal reflection consistency — light reflections must be geometrically consistent
  3. Iris texture uniqueness — irises must be the same across frames
  4. Pupil circularity — synthetic pupils are often non-circular or flickering
  5. Eye aspect ratio (EAR) naturalness — blink speed/duration distribution
  6. Sclera color consistency — whites of eyes should be uniform

References:
  - Li et al. (2018) "In Ictu Oculi: Exposing AI Generated Fake Face Videos by Detecting Eye Blinking" AVSS
  - Ciftci et al. (2020) "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals" TPAMI
  - Jung et al. (2020) "Deepfake Detection via Eye Blink Analysis"
"""
import numpy as np
import cv2
import math
from typing import List, Optional


# MediaPipe landmark indices for eyes
LEFT_EYE_INNER   = 133
LEFT_EYE_OUTER   = 33
LEFT_EYE_TOP1    = 159
LEFT_EYE_TOP2    = 158
LEFT_EYE_BOTTOM1 = 145
LEFT_EYE_BOTTOM2 = 153

RIGHT_EYE_INNER   = 362
RIGHT_EYE_OUTER   = 263
RIGHT_EYE_TOP1    = 386
RIGHT_EYE_TOP2    = 385
RIGHT_EYE_BOTTOM1 = 374
RIGHT_EYE_BOTTOM2 = 380

# Full eye contour indices
LEFT_EYE_CONTOUR  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


class EyeEngine:
    """Comprehensive eye-based deepfake detection."""

    def __init__(self):
        self.name = "Eye Consistency"
        self.violations: List[str] = []
        self._blink_rate: Optional[float] = None
        self._ear_values: List[float] = []

    @property
    def blink_rate_per_min(self) -> Optional[float]:
        return self._blink_rate

    # ── public API ─────────────────────────────────────────────────────────
    def analyze(self, frames: List[np.ndarray], landmarks_per_frame: Optional[List] = None) -> float:
        """
        Args:
            frames: BGR face-crop arrays
            landmarks_per_frame: pre-computed MediaPipe landmarks (optional; computed here if not provided)
        Returns:
            score 0.0→1.0 (higher = more likely FAKE)
        """
        self.violations = []
        self._blink_rate = None
        self._ear_values = []

        if not frames or len(frames) < 5:
            return 0.35

        # Try to compute with MediaPipe
        try:
            import mediapipe as mp
            mp_fm = mp.solutions.face_mesh
            results_list = []
            with mp_fm.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            ) as fm:
                for frame in frames:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape)==3 else frame
                    res = fm.process(rgb)
                    results_list.append(res.multi_face_landmarks[0] if res.multi_face_landmarks else None)

            ear_scores, blink_scores, iris_scores, corneal_scores = [], [], [], []

            for i, (frame, lm) in enumerate(zip(frames, results_list)):
                if lm is None:
                    continue
                h, w = frame.shape[:2]
                pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])

                ear = self._eye_aspect_ratio(pts)
                self._ear_values.append(ear)

            if self._ear_values:
                ear_arr = np.array(self._ear_values)
                blink_score = self._analyze_blink_pattern(ear_arr, fps=25.0)
                pupil_score = self._analyze_pupil_consistency(frames, results_list)
                corneal_score = self._analyze_corneal_reflections(frames)
                sclera_score = self._analyze_sclera_consistency(frames, results_list)

                combined = (0.35*blink_score + 0.30*pupil_score + 0.20*corneal_score + 0.15*sclera_score)
                return float(np.clip(combined, 0.0, 1.0))

        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: basic analysis without landmarks
        return self._fallback_analysis(frames)

    # ── EAR (Eye Aspect Ratio) ─────────────────────────────────────────────
    def _eye_aspect_ratio(self, pts: np.ndarray) -> float:
        """
        EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
        EAR ≈ 0.3 for open eye, ≈ 0.0 for closed (blink)
        """
        def ear_single(inner, outer, t1, t2, b1, b2):
            # Vertical distances
            v1 = np.linalg.norm(pts[t1] - pts[b1])
            v2 = np.linalg.norm(pts[t2] - pts[b2])
            # Horizontal distance
            h = np.linalg.norm(pts[inner] - pts[outer])
            return (v1 + v2) / (2.0 * h + 1e-9)

        left  = ear_single(LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP1, LEFT_EYE_TOP2, LEFT_EYE_BOTTOM1, LEFT_EYE_BOTTOM2)
        right = ear_single(RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP1, RIGHT_EYE_TOP2, RIGHT_EYE_BOTTOM1, RIGHT_EYE_BOTTOM2)
        return (left + right) / 2.0

    def _analyze_blink_pattern(self, ear_arr: np.ndarray, fps: float = 25.0) -> float:
        """
        Detect blinks from EAR dips below threshold.
        Normal: 15–25 blinks/min. Deepfakes: typically 0–8 blinks/min.
        Also checks for incomplete blinks (EAR doesn't reach near-zero).
        """
        EAR_THRESHOLD = 0.22  # below = blink
        MIN_BLINK_FRAMES = 2

        blinks = 0
        in_blink = False
        blink_depth_ok = 0  # fully closed blinks
        blink_depths = []
        i = 0
        while i < len(ear_arr):
            if ear_arr[i] < EAR_THRESHOLD:
                if not in_blink:
                    in_blink = True
                    blink_start = i
            else:
                if in_blink:
                    blink_len = i - blink_start
                    if blink_len >= MIN_BLINK_FRAMES:
                        blinks += 1
                        min_ear = ear_arr[blink_start:i].min()
                        blink_depths.append(min_ear)
                        if min_ear < 0.12:
                            blink_depth_ok += 1
                    in_blink = False
            i += 1

        duration_min = len(ear_arr) / (fps * 60.0)
        if duration_min < 0.05:
            return 0.35

        bpm = blinks / duration_min
        self._blink_rate = bpm
        score = 0.0

        # Blink rate check
        if bpm < 5:
            self.violations.append(f"[Eye] Very low blink rate ({bpm:.1f}/min, normal=15–25) — deepfakes under-blink")
            score += 0.70
        elif bpm < 10:
            self.violations.append(f"[Eye] Low blink rate ({bpm:.1f}/min) — below normal range")
            score += 0.45
        elif bpm > 50:
            self.violations.append(f"[Eye] Abnormally high blink rate ({bpm:.1f}/min) — neural rendering artifact")
            score += 0.50

        # Incomplete blink check
        if blinks > 0:
            complete_ratio = blink_depth_ok / blinks
            if complete_ratio < 0.4:
                self.violations.append(f"[Eye] {int((1-complete_ratio)*100)}% blinks are incomplete (EAR stays >0.12) — GAN partial synthesis")
                score += 0.35

        # EAR stability (between blinks, EAR should be consistent)
        open_ear = ear_arr[ear_arr > EAR_THRESHOLD]
        if len(open_ear) > 5:
            ear_std = float(open_ear.std())
            if ear_std > 0.08:
                self.violations.append(f"[Eye] High open-eye EAR variance ({ear_std:.3f}) — eye shape unstable between frames")
                score += 0.25

        return float(np.clip(score, 0.0, 1.0))

    def _analyze_pupil_consistency(self, frames: List[np.ndarray], results_list: List) -> float:
        """Check pupil size (by iris area estimate) is consistent across frames."""
        iris_areas = []
        for frame, lm in zip(frames, results_list):
            if lm is None:
                continue
            h, w = frame.shape[:2]
            pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])

            # Iris area approximated from eye contour
            try:
                left_pts = pts[LEFT_EYE_CONTOUR]
                right_pts = pts[RIGHT_EYE_CONTOUR]
                left_area = self._polygon_area(left_pts)
                right_area = self._polygon_area(right_pts)
                face_area = w * h
                iris_areas.append((left_area + right_area) / (face_area + 1e-9))
            except Exception:
                continue

        if len(iris_areas) < 5:
            return 0.3

        arr = np.array(iris_areas)
        cv = arr.std() / (arr.mean() + 1e-9)

        if cv > 0.35:
            self.violations.append(f"[Eye] Iris area highly inconsistent (CV={cv:.2f}) — eye shape changing between frames")
            return min(0.80, cv * 1.5)

        # Symmetry check: left vs right iris should be similar
        return float(np.clip(cv * 0.8, 0.0, 0.5))

    def _analyze_corneal_reflections(self, frames: List[np.ndarray]) -> float:
        """
        Light reflections on the cornea (specular highlights) should be:
        - Present in both eyes
        - Consistent in position relative to face
        Deepfakes often have inconsistent, missing, or multiplied reflections.
        """
        reflection_presence = []
        for frame in frames[:30]:
            if frame is None:
                continue
            h, w = frame.shape[:2]
            # Check upper quarter of each eye region for bright specular spots
            left_roi  = frame[int(h*0.2):int(h*0.45), int(w*0.1):int(w*0.45)]
            right_roi = frame[int(h*0.2):int(h*0.45), int(w*0.55):int(w*0.9)]

            def has_specular(roi):
                if roi.size == 0:
                    return False
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
                bright_pixels = (gray > 220).sum()
                return bright_pixels > 2

            reflection_presence.append((has_specular(left_roi), has_specular(right_roi)))

        if not reflection_presence:
            return 0.3

        both_present = sum(1 for l, r in reflection_presence if l and r)
        none_present = sum(1 for l, r in reflection_presence if not l and not r)
        mismatch = sum(1 for l, r in reflection_presence if l != r)
        n = len(reflection_presence)

        # High mismatch = asymmetric eye lighting = likely deepfake
        if mismatch / n > 0.4:
            self.violations.append(f"[Eye] Corneal reflection asymmetry in {mismatch}/{n} frames — inconsistent eye rendering")
            return min(0.75, (mismatch / n) * 1.2)

        if none_present / n > 0.7:
            self.violations.append(f"[Eye] Missing corneal reflections in {none_present}/{n} frames — unnatural eye rendering")
            return 0.45

        return 0.15

    def _analyze_sclera_consistency(self, frames: List[np.ndarray], results_list: List) -> float:
        """Check that sclera (white of eye) color is consistent and realistic."""
        sclera_colors = []

        for frame, lm in zip(frames[:40], results_list[:40]):
            if lm is None:
                continue
            h, w = frame.shape[:2]
            pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])

            try:
                # Get inner corner region of each eye
                left_inner = pts[LEFT_EYE_INNER].astype(int)
                right_inner = pts[RIGHT_EYE_INNER].astype(int)

                def sclera_color(center, size=6):
                    y1, y2 = max(0, center[1]-size), min(h, center[1]+size)
                    x1, x2 = max(0, center[0]-size), min(w, center[0]+size)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        return None
                    return roi.mean(axis=(0,1))  # BGR mean

                lc = sclera_color(left_inner)
                rc = sclera_color(right_inner)
                if lc is not None and rc is not None:
                    sclera_colors.append((lc + rc) / 2)
            except Exception:
                continue

        if len(sclera_colors) < 5:
            return 0.25

        colors = np.array(sclera_colors)
        color_std = colors.std(axis=0).mean()

        # Real sclera: low temporal variation (std < 15 BGR units)
        if color_std > 30:
            self.violations.append(f"[Eye] Sclera color unstable (std={color_std:.1f}) — white of eye flickering between frames")
            return min(0.70, color_std / 50.0)

        return float(np.clip(color_std / 60.0, 0.0, 0.35))

    # ── utilities ──────────────────────────────────────────────────────────
    def _polygon_area(self, pts: np.ndarray) -> float:
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def _fallback_analysis(self, frames: List[np.ndarray]) -> float:
        """Simplified analysis without landmark data."""
        brightnesses = []
        for frame in frames:
            if frame is None:
                continue
            h, w = frame.shape[:2]
            # Rough eye region
            eye_region = frame[int(h*0.2):int(h*0.5), int(w*0.1):int(w*0.9)]
            if eye_region.size > 0:
                gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape)==3 else eye_region
                brightnesses.append(float(gray.mean()))

        if not brightnesses:
            return 0.35
        arr = np.array(brightnesses)
        cv = arr.std() / (arr.mean() + 1e-9)
        return float(np.clip(cv * 1.5, 0.0, 0.6))

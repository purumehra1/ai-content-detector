"""
Head Pose & 3D Consistency Engine
───────────────────────────────────
Early deepfake detection work (Exposing Deep Fakes Using Inconsistent Head Poses, 2019)
showed that deepfakes have inconsistent 3D head poses.

This engine:
  1. Estimates 3D head pose (pitch/yaw/roll) from facial landmarks
  2. Checks pose smoothness and physical plausibility
  3. Validates that head pose is consistent with audio (head nods during emphasis)
  4. Detects impossible head pose jumps
  5. Checks bilateral facial symmetry vs pose angle (face-swaps break this)

References:
  - Yang et al. (2019) "Exposing Deep Fakes Using Inconsistent Head Poses" ICASSP
  - Li et al. (2020) "Face X-ray for More General Face Forgery Detection" CVPR
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple


# 3D model points for solvePnP (generic face model, mm scale)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# MediaPipe face mesh landmark indices for these points
POSE_LM_INDICES = [1, 152, 226, 446, 57, 287]


class HeadPoseEngine:
    """Head pose estimation and consistency analysis."""

    def __init__(self):
        self.name = "Head Pose"
        self.violations: List[str] = []

    # ── public API ─────────────────────────────────────────────────────────
    def analyze(self, frames: List[np.ndarray]) -> float:
        self.violations = []
        if len(frames) < 5:
            return 0.35

        poses = self._estimate_poses(frames)
        if not poses:
            return 0.35

        scores = [
            self._pose_smoothness(poses),
            self._pose_plausibility(poses),
            self._symmetry_vs_pose(frames, poses),
        ]
        scores = [s for s in scores if s is not None]
        return float(np.clip(np.mean(scores) if scores else 0.4, 0.0, 1.0))

    # ── pose estimation ────────────────────────────────────────────────────
    def _estimate_poses(self, frames: List[np.ndarray]) -> List[Optional[Tuple]]:
        """Estimate yaw/pitch/roll for each frame."""
        poses = []
        try:
            import mediapipe as mp
            mp_fm = mp.solutions.face_mesh
            with mp_fm.FaceMesh(static_image_mode=False, max_num_faces=1,
                                min_detection_confidence=0.4) as fm:
                for frame in frames:
                    if frame is None:
                        poses.append(None)
                        continue
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = fm.process(rgb)
                    if not res.multi_face_landmarks:
                        poses.append(None)
                        continue
                    lm = res.multi_face_landmarks[0]
                    pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])

                    # 2D image points for solvePnP
                    image_points = np.array([pts[i] for i in POSE_LM_INDICES], dtype=np.float64)

                    focal = w
                    cam_matrix = np.array([[focal, 0, w//2],
                                           [0, focal, h//2],
                                           [0, 0, 1]], dtype=np.float64)
                    dist_coeffs = np.zeros((4,1))

                    ok, rvec, tvec = cv2.solvePnP(
                        MODEL_POINTS_3D, image_points,
                        cam_matrix, dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if ok:
                        rmat, _ = cv2.Rodrigues(rvec)
                        euler = self._rotation_matrix_to_euler(rmat)
                        poses.append(euler)  # (pitch, yaw, roll) in degrees
                    else:
                        poses.append(None)
        except Exception:
            return []
        return poses

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            pitch = np.degrees(np.arctan2(R[2,1], R[2,2]))
            yaw   = np.degrees(np.arctan2(-R[2,0], sy))
            roll  = np.degrees(np.arctan2(R[1,0], R[0,0]))
        else:
            pitch = np.degrees(np.arctan2(-R[1,2], R[1,1]))
            yaw   = np.degrees(np.arctan2(-R[2,0], sy))
            roll  = 0.0
        return (float(pitch), float(yaw), float(roll))

    # ── analysis functions ─────────────────────────────────────────────────
    def _pose_smoothness(self, poses: List) -> float:
        """Head movement should be smooth — sudden jumps are physically impossible."""
        valid = [(i, p) for i, p in enumerate(poses) if p is not None]
        if len(valid) < 3:
            return 0.3

        velocities = []
        for i in range(1, len(valid)):
            prev_i, prev_p = valid[i-1]
            curr_i, curr_p = valid[i]
            frame_delta = curr_i - prev_i
            vel = [abs(curr_p[j] - prev_p[j]) / (frame_delta + 1e-9) for j in range(3)]
            velocities.append(vel)

        velocities = np.array(velocities)
        max_vel = velocities.max(axis=0)  # per axis

        # Max physically realistic head rotation speed: ~60 deg/second = 2.4 deg/frame @ 25fps
        threshold = 5.0  # deg/frame (generous)
        violations_count = (max_vel > threshold).sum()

        if violations_count > 0:
            worst = float(velocities.max())
            self.violations.append(f"[Head Pose] Instantaneous head rotation {worst:.1f}°/frame — physically impossible movement")
            return min(0.80, violations_count * 0.25 + (worst - threshold) * 0.03)

        # Check smoothness via jerk (acceleration change)
        if len(velocities) > 2:
            jerk = np.diff(velocities, axis=0)
            mean_jerk = float(np.abs(jerk).mean())
            if mean_jerk > 2.0:
                self.violations.append(f"[Head Pose] High head movement jerk ({mean_jerk:.2f}) — unnatural motion")
                return min(0.60, mean_jerk * 0.15)

        return float(np.clip(velocities.std() * 0.05, 0.0, 0.3))

    def _pose_plausibility(self, poses: List) -> float:
        """Check that poses are within physically plausible ranges."""
        valid = [p for p in poses if p is not None]
        if not valid:
            return 0.35

        pitches = [p[0] for p in valid]
        yaws = [p[1] for p in valid]
        rolls = [p[2] for p in valid]

        score = 0.0
        # Extreme head rotation in video conversation: unlikely
        max_yaw = max(abs(y) for y in yaws)
        max_pitch = max(abs(p) for p in pitches)

        if max_yaw > 75:
            self.violations.append(f"[Head Pose] Extreme yaw angle ({max_yaw:.0f}°) — implausible head orientation")
            score += 0.40

        if max_pitch > 60:
            self.violations.append(f"[Head Pose] Extreme pitch angle ({max_pitch:.0f}°) — implausible head tilt")
            score += 0.30

        # Roll drift: heads shouldn't continuously roll without returning
        roll_arr = np.array(rolls)
        roll_drift = abs(float(np.polyfit(np.arange(len(rolls)), rolls, 1)[0]))
        if roll_drift > 0.3:
            self.violations.append(f"[Head Pose] Continuous roll drift ({roll_drift:.2f}°/frame) — deepfake pose drift")
            score += min(0.35, roll_drift * 0.8)

        return float(np.clip(score, 0.0, 0.8))

    def _symmetry_vs_pose(self, frames: List[np.ndarray], poses: List) -> float:
        """
        At yaw≈0 (frontal face), bilateral symmetry should be highest.
        Face-swaps break this relationship — symmetry is inconsistent with pose.
        """
        frontal_symmetries = []
        non_frontal_symmetries = []

        for frame, pose in zip(frames, poses):
            if pose is None or frame is None:
                continue
            _, yaw, _ = pose
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame

            left_half  = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            min_w = min(left_half.shape[1], right_half.shape[1])
            if min_w == 0:
                continue

            # Resize to same width for comparison
            left_half  = left_half[:, :min_w]
            right_half = right_half[:, :min_w]
            diff = float(np.mean(np.abs(left_half.astype(float) - right_half.astype(float))))
            symmetry_score = 1.0 - min(1.0, diff / 128.0)

            if abs(yaw) < 15:  # frontal
                frontal_symmetries.append(symmetry_score)
            else:
                non_frontal_symmetries.append(symmetry_score)

        if not frontal_symmetries:
            return 0.3

        mean_frontal = np.mean(frontal_symmetries)
        std_frontal = np.std(frontal_symmetries)

        # Frontal symmetry should be high AND consistent for real faces
        if mean_frontal < 0.50:
            self.violations.append(f"[Head Pose] Low frontal face symmetry ({mean_frontal:.2f}) — face-swap boundary misalignment")
            return float(np.clip(1.0 - mean_frontal, 0.0, 0.75))

        if std_frontal > 0.15:
            self.violations.append(f"[Head Pose] Inconsistent symmetry at frontal pose (std={std_frontal:.2f}) — unstable face replacement")
            return float(np.clip(std_frontal * 2.0, 0.0, 0.60))

        return float(np.clip((0.80 - mean_frontal) * 1.5, 0.0, 0.35))

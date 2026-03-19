"""
Hand / Finger Analysis Engine (Optional)
─────────────────────────────────────────
MediaPipe Hands provides 21 landmarks per hand.
Deepfakes (especially full-body or half-body videos) often hallucinate:
  • Wrong finger count (5 ≠ actual counted)
  • Fused fingers (distance between tips too small)
  • Unnatural finger motion (non-biological joint angles)
  • Missing hands when hands are visible
Output: hand_realism_score (0=real, 1=fake). Returns 0.3 if no hands detected.
"""
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

# Finger tip landmark indices (per MediaPipe 21-point model)
FINGERTIPS = [4, 8, 12, 16, 20]     # thumb, index, middle, ring, pinky
FINGER_MCP = [2, 5, 9, 13, 17]      # metacarpal joints

# Natural finger joint angle range (degrees)
MIN_JOINT_ANGLE = 10.0
MAX_JOINT_ANGLE = 175.0

# Min distance between adjacent fingertips (normalized by hand size)
MIN_TIP_SEPARATION = 0.04


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b, in degrees."""
    v1 = a - b
    v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def _hand_size(lm: np.ndarray) -> float:
    """Wrist to middle finger MCP distance as hand size reference."""
    return float(np.linalg.norm(lm[0] - lm[9]))


def _analyze_single_hand(lm: np.ndarray) -> dict:
    """Analyze one hand's 21 landmarks. Returns per-hand issues."""
    issues = []
    size = _hand_size(lm)

    # ── Finger separation check ───────────────────────────────────────────────
    tips = lm[FINGERTIPS, :2]  # (5, 2)
    for i in range(len(FINGERTIPS) - 1):
        dist = float(np.linalg.norm(tips[i] - tips[i + 1])) / max(size, 1)
        if dist < MIN_TIP_SEPARATION:
            issues.append(f"Fused fingers: tips {i+1} and {i+2} too close (dist={dist:.3f})")

    # ── Joint angle plausibility ──────────────────────────────────────────────
    # Check PIP joint angle for each finger (indices 6-7-8 etc.)
    finger_joints = [
        (5, 6, 7),   # index PIP
        (9, 10, 11), # middle PIP
        (13, 14, 15),# ring PIP
        (17, 18, 19),# pinky PIP
    ]
    for a_idx, b_idx, c_idx in finger_joints:
        angle = _angle(lm[a_idx], lm[b_idx], lm[c_idx])
        if not (MIN_JOINT_ANGLE <= angle <= MAX_JOINT_ANGLE):
            issues.append(f"Impossible joint angle: {angle:.1f}° at landmark {b_idx} (natural range {MIN_JOINT_ANGLE}°–{MAX_JOINT_ANGLE}°)")

    # ── Thumb anatomy check ───────────────────────────────────────────────────
    thumb_angle = _angle(lm[1], lm[2], lm[3])
    if thumb_angle > 160 or thumb_angle < 5:
        issues.append(f"Anatomically impossible thumb angle: {thumb_angle:.1f}°")

    return {"issues": issues, "hand_size_px": round(size, 1)}


class HandEngine:
    WEIGHT = 0.15

    def analyze(self, frames: list) -> dict:
        violations = []
        all_hand_issues = []
        hand_detected_frames = 0
        frame_hand_counts = []
        velocity_scores = []
        prev_lm = None

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if not result.multi_hand_landmarks:
                    frame_hand_counts.append(0)
                    prev_lm = None
                    continue

                hand_detected_frames += 1
                n_hands = len(result.multi_hand_landmarks)
                frame_hand_counts.append(n_hands)

                for hand_lm in result.multi_hand_landmarks:
                    h, w = frame.shape[:2]
                    lm = np.array([[l.x * w, l.y * h, l.z * w] for l in hand_lm.landmark])

                    # Anatomy analysis
                    hand_result = _analyze_single_hand(lm)
                    all_hand_issues.extend(hand_result["issues"])

                    # Motion velocity check
                    if prev_lm is not None and lm.shape == prev_lm.shape:
                        vel = np.mean(np.linalg.norm(lm[:, :2] - prev_lm[:, :2], axis=1))
                        velocity_scores.append(float(vel))
                    prev_lm = lm

        # ── No hands detected ─────────────────────────────────────────────────
        if hand_detected_frames == 0:
            return {
                "score": 0.3,
                "label": "INCONCLUSIVE",
                "violations": ["No hands detected in video — hand analysis skipped"],
                "confidence": 0.2,
                "hands_detected": False,
            }

        penalty = 0.0
        detection_rate = hand_detected_frames / len(frames)

        # ── 1. Anatomical violations ──────────────────────────────────────────
        anatomy_violations = len(all_hand_issues)
        if anatomy_violations > 0:
            top = list(dict.fromkeys(all_hand_issues))[:3]
            for v in top:
                violations.append(f"ANATOMY: {v}")
            penalty += min(0.5, anatomy_violations * 0.08)

        # ── 2. Unnatural velocity ─────────────────────────────────────────────
        if velocity_scores:
            mean_vel = float(np.mean(velocity_scores))
            vel_std = float(np.std(velocity_scores))
            if vel_std > mean_vel * 3:
                violations.append(f"HAND TELEPORT: Velocity std={vel_std:.1f} >> mean={mean_vel:.1f} — discontinuous hand motion")
                penalty += 0.25

        # ── 3. Inconsistent hand count ────────────────────────────────────────
        non_zero_counts = [c for c in frame_hand_counts if c > 0]
        if non_zero_counts and (max(non_zero_counts) - min(non_zero_counts)) > 1:
            violations.append(f"FINGER COUNT INSTABILITY: Hand count varies from {min(non_zero_counts)} to {max(non_zero_counts)} across frames")
            penalty += 0.20

        fake_score = min(1.0, penalty)
        return {
            "score": fake_score,
            "label": "FAKE" if fake_score > 0.5 else "REAL",
            "hands_detected": True,
            "hand_detection_rate": round(detection_rate, 3),
            "anatomy_violations_total": anatomy_violations,
            "mean_hand_velocity_px": round(float(np.mean(velocity_scores)) if velocity_scores else 0, 2),
            "violations": violations,
            "confidence": min(1.0, abs(fake_score - 0.5) * 2),
        }

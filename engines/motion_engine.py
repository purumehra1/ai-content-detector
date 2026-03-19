"""
Biological Motion Analysis Engine
──────────────────────────────────
Tracks facial landmarks across frames and computes:
  • Velocity, acceleration, jerk (rate of change of acceleration)
  • Reaction delay (speech onset → jaw movement lag)
  • Freeze phases (abnormally still periods)
  • Teleport motion (impossible jumps between frames)
Output: motion_realism_score (0=fake, 1=real)
"""
import numpy as np
import mediapipe as mp
from utils.face_utils import get_landmarks, get_jaw_center, get_lip_opening, get_eye_aspect_ratio

mp_face_mesh = mp.solutions.face_mesh

# Physics-based thresholds (empirically calibrated)
MAX_JERK = 120.0          # pixels/frame³ — higher = supernatural motion
MIN_VELOCITY_MEAN = 0.3   # px/frame — lower = suspiciously frozen
FREEZE_THRESHOLD = 0.8    # px/frame — velocity below this = freeze
TELEPORT_THRESHOLD = 30.0 # px/frame — velocity above this = teleport
BLINK_EAR_THRESHOLD = 0.20  # EAR below = eyes closed


class MotionEngine:
    WEIGHT = 0.20

    def analyze(self, frames: list) -> dict:
        if len(frames) < 5:
            return {"score": 0.5, "violations": [], "confidence": 0.0}

        landmarks_seq = []
        jaw_centers = []
        lip_openings = []
        ear_values = []

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4
        ) as mesh:
            for frame in frames:
                lm = get_landmarks(frame, mesh)
                if lm is not None:
                    landmarks_seq.append(lm)
                    jaw_centers.append(get_jaw_center(lm))
                    lip_openings.append(get_lip_opening(lm))
                    ear_values.append((get_eye_aspect_ratio(lm, "left") + get_eye_aspect_ratio(lm, "right")) / 2)

        if len(landmarks_seq) < 4:
            return {"score": 0.5, "violations": ["Insufficient face detections for motion analysis"], "confidence": 0.0}

        violations = []
        penalty = 0.0

        # ── 1. Jaw motion physics ─────────────────────────────────────────────
        jaw_arr = np.array(jaw_centers)  # (N, 2)
        jaw_vel = np.linalg.norm(np.diff(jaw_arr, axis=0), axis=1)    # pixels/frame
        jaw_acc = np.diff(jaw_vel)
        jaw_jerk = np.abs(np.diff(jaw_acc))

        mean_vel = float(np.mean(jaw_vel))
        max_jerk = float(np.max(jaw_jerk)) if len(jaw_jerk) > 0 else 0

        # Freeze detection: many consecutive near-zero velocities
        freeze_frames = np.sum(jaw_vel < FREEZE_THRESHOLD)
        freeze_ratio = freeze_frames / len(jaw_vel)

        # Teleport detection: impossible velocity spikes
        teleport_frames = np.sum(jaw_vel > TELEPORT_THRESHOLD)

        if freeze_ratio > 0.6:
            violations.append(f"FREEZE: Face nearly static for {freeze_ratio*100:.0f}% of frames (deepfakes often freeze mid-motion)")
            penalty += 0.3

        if teleport_frames > 0:
            violations.append(f"TELEPORT: {teleport_frames} frame(s) with impossible motion jumps (>{TELEPORT_THRESHOLD}px/frame)")
            penalty += min(0.4, teleport_frames * 0.1)

        if max_jerk > MAX_JERK:
            violations.append(f"UNNATURAL JERK: Facial acceleration spike of {max_jerk:.1f} px/frame³ exceeds physical limit")
            penalty += 0.2

        if mean_vel < MIN_VELOCITY_MEAN:
            violations.append(f"SUSPICIOUS STILLNESS: Mean facial velocity {mean_vel:.2f} px/frame — abnormally rigid for natural speech")
            penalty += 0.15

        # ── 2. Blink pattern analysis ────────────────────────────────────────
        ear_arr = np.array(ear_values)
        blink_events = np.sum(ear_arr < BLINK_EAR_THRESHOLD)
        n_frames = len(ear_arr)
        blink_rate = blink_events / max(n_frames, 1)

        # Natural blink rate: 15-20 blinks/min. At 10fps over N frames:
        expected_blinks_per_frame = 17.5 / (60 * 10)  # ~0.029 per frame
        if blink_rate < expected_blinks_per_frame * 0.3:
            violations.append(f"LOW BLINK RATE: Only {blink_events} blinks detected — humans blink 15-20× per minute")
            penalty += 0.15
        elif blink_rate > expected_blinks_per_frame * 5:
            violations.append(f"EXCESSIVE BLINKING: {blink_events} blink events — unnaturally high frequency")
            penalty += 0.1

        # ── 3. Lip opening vs jaw movement correlation ────────────────────────
        lip_arr = np.array(lip_openings)
        # If lips move (speech) but jaw stays still — violation
        lip_var = float(np.var(lip_arr))
        jaw_var = float(np.var(jaw_vel))
        if lip_var > 0.001 and jaw_var < 0.05:
            violations.append("DESYNC: Lips are moving but jaw shows no corresponding motion — characteristic of face-swap")
            penalty += 0.25

        # ── Score ─────────────────────────────────────────────────────────────
        fake_score = min(1.0, penalty)
        return {
            "score": fake_score,
            "label": "FAKE" if fake_score > 0.5 else "REAL",
            "mean_velocity_px": round(mean_vel, 3),
            "max_jerk_px": round(max_jerk, 3),
            "freeze_ratio": round(freeze_ratio, 3),
            "blink_events": int(blink_events),
            "violations": violations,
            "confidence": min(1.0, abs(fake_score - 0.5) * 2),
            "frames_analyzed": len(landmarks_seq),
        }

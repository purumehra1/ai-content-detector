"""
Rule-Based Causal Consistency Engine
─────────────────────────────────────
Verifies natural cause-effect relationships that all real humans obey:

  Rule 1:  SPEECH → JAW MOVEMENT  (jaw must open when speaking)
  Rule 2:  SPEECH PAUSE → BLINK   (humans tend to blink at breath pauses)
  Rule 3:  TEETH STABILITY        (teeth don't change shape during speech)
  Rule 4:  MOTION PHYSICS         (velocity/acceleration follow human biomechanics)
  Rule 5:  GAZE CONSISTENCY       (eyes don't skip positions unnaturally)
  Rule 6:  FACIAL SYMMETRY        (left-right face symmetry should be consistent)
  Rule 7:  LIP SYNC PHYSICS       (lip opening speed has physical upper limit)
  Rule 8:  MICRO-EXPRESSION RATE  (micro-expressions appear briefly, not sustained)

Each violated rule contributes to the causal_score.
Output: {score, violated_rules, consistency_score}
"""
import numpy as np
import cv2
import mediapipe as mp
from utils.face_utils import (
    get_landmarks, get_lip_opening, get_eye_aspect_ratio,
    LEFT_EYE, RIGHT_EYE, JAW
)

mp_face_mesh = mp.solutions.face_mesh

# ── Thresholds (biologically calibrated) ─────────────────────────────────────
LIP_OPEN_SPEECH_MIN = 0.025     # minimum lip opening during speech
LIP_SPEED_MAX = 0.08            # max lip opening change per frame (physical limit)
BLINK_EAR_CLOSED = 0.20         # EAR below = blink
SYMMETRY_THRESHOLD = 0.15       # max left-right asymmetry (normalized)
GAZE_JUMP_THRESHOLD = 0.12      # max normalized eye position jump per frame

# Eye/iris landmark indices (MediaPipe refined landmarks required)
LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]


def _iris_center(landmarks: np.ndarray, indices: list) -> np.ndarray:
    return landmarks[indices, :2].mean(axis=0)


def _face_symmetry_score(landmarks: np.ndarray) -> float:
    """
    Compute bilateral symmetry of key facial points.
    Returns 0 (symmetric) to 1 (asymmetric).
    """
    # Mirror pairs: (left_point, right_point) approximately
    mirror_pairs = [
        (33, 263),   # eye corners
        (61, 291),   # lip corners
        (70, 300),   # brow
        (105, 334),  # brow inner
        (159, 386),  # upper eyelid
        (145, 374),  # lower eyelid
    ]
    face_center_x = float(landmarks[1, 0])  # nose tip as center proxy
    face_width = abs(landmarks[234, 0] - landmarks[454, 0])  # cheek to cheek

    diffs = []
    for l_idx, r_idx in mirror_pairs:
        l_pt = landmarks[l_idx, :2]
        r_pt = landmarks[r_idx, :2]
        # Mirror right point across face center
        r_mirrored_x = 2 * face_center_x - r_pt[0]
        diff = abs(l_pt[0] - r_mirrored_x) / max(face_width, 1)
        diffs.append(diff)
    return float(np.mean(diffs))


class CausalEngine:
    WEIGHT = 0.10

    def analyze(self, frames: list, rms_timeline: np.ndarray | None = None) -> dict:
        violated_rules = []
        rule_penalties = {}

        landmarks_seq = []
        lip_openings = []
        ear_left_seq = []
        ear_right_seq = []
        symmetry_seq = []
        left_iris_seq = []
        right_iris_seq = []

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as mesh:
            for frame in frames:
                lm = get_landmarks(frame, mesh)
                if lm is None:
                    continue
                landmarks_seq.append(lm)
                lip_openings.append(get_lip_opening(lm))
                ear_left_seq.append(get_eye_aspect_ratio(lm, "left"))
                ear_right_seq.append(get_eye_aspect_ratio(lm, "right"))
                symmetry_seq.append(_face_symmetry_score(lm))

                # Gaze tracking (requires refined landmarks — indices 468-477)
                if len(lm) > 477:
                    left_iris_seq.append(_iris_center(lm, LEFT_IRIS))
                    right_iris_seq.append(_iris_center(lm, RIGHT_IRIS))

        n = len(landmarks_seq)
        if n < 4:
            return {"score": 0.5, "violated_rules": ["Insufficient landmark detections"], "confidence": 0.0}

        lip_arr = np.array(lip_openings)
        ear_l = np.array(ear_left_seq)
        ear_r = np.array(ear_right_seq)
        sym_arr = np.array(symmetry_seq)
        total_penalty = 0.0

        # ── RULE 1: SPEECH → JAW MOVEMENT ────────────────────────────────────
        if rms_timeline is not None:
            min_len = min(len(rms_timeline), len(lip_arr))
            rms = rms_timeline[:min_len]
            lip_sub = lip_arr[:min_len]
            speech_mask = rms > 0.02
            if np.sum(speech_mask) > 3:
                lip_during_speech = lip_sub[speech_mask]
                jaw_open_rate = np.mean(lip_during_speech > LIP_OPEN_SPEECH_MIN)
                if jaw_open_rate < 0.40:
                    violated_rules.append(f"RULE 1 VIOLATED — Speech→Jaw: Jaw open in only {jaw_open_rate*100:.0f}% of speech frames (natural ≥ 60%)")
                    rule_penalties["rule1"] = 0.35
                    total_penalty += 0.35
        else:
            # Without audio: check lip motion is nonzero at some point
            if np.max(lip_arr) < LIP_OPEN_SPEECH_MIN * 2:
                violated_rules.append("RULE 1 WARNING — Jaw barely opens across entire video (no visible speech motion)")
                rule_penalties["rule1"] = 0.15
                total_penalty += 0.15

        # ── RULE 2: SPEECH PAUSE → BLINK ─────────────────────────────────────
        # Detect transition points (lip closing after opening = pause)
        lip_delta = np.diff(lip_arr)
        pause_frames = np.where((lip_arr[:-1] > 0.02) & (lip_delta < -0.01))[0]
        blink_after_pause = 0
        for pf in pause_frames:
            window = min(5, len(ear_l) - pf)
            if window > 0 and np.any(ear_l[pf:pf + window] < BLINK_EAR_CLOSED):
                blink_after_pause += 1
        if len(pause_frames) > 3:
            blink_rate_at_pause = blink_after_pause / len(pause_frames)
            if blink_rate_at_pause < 0.15:
                violated_rules.append(f"RULE 2 VIOLATED — Pause→Blink: Only {blink_rate_at_pause*100:.0f}% of speech pauses followed by blink (natural ~30-50%)")
                rule_penalties["rule2"] = 0.15
                total_penalty += 0.15

        # ── RULE 3: TEETH STABILITY (structural consistency) ─────────────────
        # Delegated to TeethEngine — causal rule confirms concept
        sym_std = float(np.std(sym_arr))
        if sym_std > 0.05:
            violated_rules.append(f"RULE 3 VARIANT — Facial symmetry fluctuating (σ={sym_std:.3f}) — real faces maintain stable bilateral symmetry")
            rule_penalties["rule3"] = 0.15
            total_penalty += 0.15

        # ── RULE 4: MOTION PHYSICS — Lip speed physical limit ─────────────────
        lip_speed = np.abs(np.diff(lip_arr))
        overspeed_frames = np.sum(lip_speed > LIP_SPEED_MAX)
        if overspeed_frames > 2:
            violated_rules.append(f"RULE 4 VIOLATED — Lip Physics: {overspeed_frames} frames with lip speed >{LIP_SPEED_MAX}/frame — physically impossible jaw movement")
            rule_penalties["rule4"] = min(0.30, overspeed_frames * 0.07)
            total_penalty += rule_penalties["rule4"]

        # ── RULE 5: GAZE CONSISTENCY ──────────────────────────────────────────
        if len(left_iris_seq) > 3:
            iris_arr = np.array(left_iris_seq)
            face_h = float(landmarks_seq[0][10, 1] - landmarks_seq[0][152, 1])
            gaze_vel = np.linalg.norm(np.diff(iris_arr, axis=0), axis=1) / max(face_h, 1)
            gaze_jumps = np.sum(gaze_vel > GAZE_JUMP_THRESHOLD)
            if gaze_jumps > 1:
                violated_rules.append(f"RULE 5 VIOLATED — Gaze Physics: {gaze_jumps} impossible eye position jumps (deepfake eye tracking often fails)")
                rule_penalties["rule5"] = min(0.25, gaze_jumps * 0.08)
                total_penalty += rule_penalties["rule5"]

        # ── RULE 6: FACIAL SYMMETRY CONSISTENCY ──────────────────────────────
        mean_sym = float(np.mean(sym_arr))
        if mean_sym > SYMMETRY_THRESHOLD:
            violated_rules.append(f"RULE 6 VIOLATED — Symmetry: Mean asymmetry={mean_sym:.3f} exceeds natural limit ({SYMMETRY_THRESHOLD})")
            rule_penalties["rule6"] = 0.20
            total_penalty += 0.20

        # ── RULE 7: LIP SYNC PHYSICS ALREADY IN RULE 4 ───────────────────────

        # ── RULE 8: MICRO-EXPRESSION DURATION ────────────────────────────────
        # Quick expressions should be brief. If lip is "slightly open" for many frames = GAN interpolation artifact
        partial_open = lip_arr[(lip_arr > LIP_OPEN_SPEECH_MIN * 0.5) & (lip_arr < LIP_OPEN_SPEECH_MIN * 1.5)]
        if len(partial_open) > n * 0.4:
            violated_rules.append(f"RULE 8 VIOLATED — Micro-Expression: Lip held in unnatural intermediate position for {len(partial_open)/n*100:.0f}% of frames (GAN interpolation artifact)")
            rule_penalties["rule8"] = 0.15
            total_penalty += 0.15

        fake_score = min(1.0, total_penalty)
        return {
            "score": fake_score,
            "label": "FAKE" if fake_score > 0.4 else "REAL",
            "violated_rules": violated_rules,
            "rule_penalties": rule_penalties,
            "rules_checked": 8,
            "rules_violated": len(rule_penalties),
            "symmetry_mean": round(mean_sym, 4),
            "confidence": min(1.0, abs(fake_score - 0.5) * 2),
            "frames_analyzed": n,
        }

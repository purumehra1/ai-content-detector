from __future__ import annotations
"""Face detection and landmark extraction utilities."""
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# MediaPipe landmark indices for key facial regions
TEETH_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405,
                   314, 17, 84, 181, 91, 146, 78, 95, 88, 178, 87, 14, 317, 402,
                   318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 42, 183, 80]
LIP_UPPER = [61, 40, 37, 0, 267, 270, 409, 291]
LIP_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
JAW = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
       400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def detect_face_crop(frame: np.ndarray, padding: float = 0.2) -> np.ndarray | None:
    """Detect face, return cropped+padded face region."""
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as det:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = det.process(rgb)
        if not result.detections:
            return None
        d = result.detections[0].location_data.relative_bounding_box
        h, w = frame.shape[:2]
        x1 = max(0, int((d.xmin - padding * d.width) * w))
        y1 = max(0, int((d.ymin - padding * d.height) * h))
        x2 = min(w, int((d.xmin + d.width * (1 + padding)) * w))
        y2 = min(h, int((d.ymin + d.height * (1 + padding)) * h))
        return frame[y1:y2, x1:x2]


def get_landmarks(frame: np.ndarray, face_mesh) -> np.ndarray | None:
    """Get 478 MediaPipe Face Mesh landmarks as (N, 3) array in pixel coords."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]
    return np.array([[l.x * w, l.y * h, l.z * w] for l in lm])  # (478, 3)


def get_lip_opening(landmarks: np.ndarray) -> float:
    """Compute vertical lip opening distance (normalized)."""
    upper = landmarks[LIP_UPPER[0], 1]
    lower = landmarks[LIP_LOWER[5], 1]
    face_h = abs(landmarks[10, 1] - landmarks[152, 1])  # forehead to chin
    return abs(lower - upper) / max(face_h, 1)


def get_eye_aspect_ratio(landmarks: np.ndarray, eye: str = "right") -> float:
    """Eye Aspect Ratio (EAR) — low = closed."""
    pts = landmarks[RIGHT_EYE] if eye == "right" else landmarks[LEFT_EYE]
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def get_teeth_region(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
    """Crop the teeth region from frame using landmark polygon."""
    pts = landmarks[TEETH_LANDMARKS, :2].astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    if w < 5 or h < 5:
        return None
    pad = 4
    y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
    return frame[y1:y2, x1:x2]


def get_jaw_center(landmarks: np.ndarray) -> np.ndarray:
    """Return centroid of jaw landmarks."""
    return landmarks[JAW, :2].mean(axis=0)


def crop_faces(frames: list, padding: float = 0.20) -> list:
    """
    Detect and crop face region from each frame.
    Falls back to the original frame if no face is detected.

    Args:
        frames: list of BGR numpy arrays
        padding: fractional padding around detected face bounding box
    Returns:
        list of BGR numpy arrays (face crops or original frames)
    """
    if not frames:
        return []

    crops = []
    try:
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.45
        ) as detector:
            for frame in frames:
                if frame is None or frame.size == 0:
                    crops.append(frame)
                    continue
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = detector.process(rgb)
                    if result.detections:
                        d = result.detections[0].location_data.relative_bounding_box
                        h, w = frame.shape[:2]
                        x1 = max(0, int((d.xmin - padding * d.width) * w))
                        y1 = max(0, int((d.ymin - padding * d.height) * h))
                        x2 = min(w, int((d.xmin + d.width * (1 + padding)) * w))
                        y2 = min(h, int((d.ymin + d.height * (1 + padding)) * h))
                        crop = frame[y1:y2, x1:x2]
                        crops.append(crop if crop.size > 0 else frame)
                    else:
                        crops.append(frame)  # no face detected — use full frame
                except Exception:
                    crops.append(frame)
    except Exception:
        # MediaPipe unavailable — return original frames
        return frames

    return crops

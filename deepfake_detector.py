"""
DeepFake Guardian — Core Detection Pipeline
────────────────────────────────────────────
Orchestrates 10+ parallel detection engines and fuses results.
"""
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import numpy as np
import cv2

from engines.cnn_gru_engine      import CNNGRUEngine
from engines.motion_engine       import MotionEngine
from engines.teeth_engine        import TeethEngine
from engines.audio_visual_engine import AudioVisualEngine
from engines.hand_engine         import HandEngine
from engines.stability_engine    import StabilityEngine
from engines.causal_engine       import CausalEngine
from engines.frequency_engine    import FrequencyEngine
from engines.rppg_engine         import RPPGEngine
from engines.eye_engine          import EyeEngine
from engines.head_pose_engine    import HeadPoseEngine
from engines.skin_texture_engine import SkinTextureEngine
from fusion.weighted_fusion      import fuse, FusionResult
from utils.video_utils           import extract_frames, extract_audio_array, get_video_info
from utils.face_utils            import crop_faces


class DeepFakeDetector:
    """
    Main deepfake detection pipeline.

    Usage:
        detector = DeepFakeDetector(max_frames=60)
        result = detector.analyze("path/to/video.mp4")
        print(result.label, result.final_score)
    """

    def __init__(self,
                 max_frames: int = 60,
                 verbose: bool = True,
                 device: Optional[str] = None):
        self.max_frames = max_frames
        self.verbose    = verbose
        self.device     = device

        # Instantiate all engines
        self.cnn_gru   = CNNGRUEngine(device=device)
        self.motion    = MotionEngine()
        self.teeth     = TeethEngine()
        self.av        = AudioVisualEngine()
        self.hand      = HandEngine()
        self.stability = StabilityEngine(base_engine=self.cnn_gru)
        self.causal    = CausalEngine()
        self.frequency = FrequencyEngine()
        self.rppg      = RPPGEngine()
        self.eye       = EyeEngine()
        self.head_pose = HeadPoseEngine()
        self.skin      = SkinTextureEngine()

    def analyze(self, video_path: str) -> FusionResult:
        """Full analysis pipeline."""
        t0 = time.time()
        self._log(f"🔍 Analyzing: {video_path}")

        # 1. Extract frames and info
        video_info = get_video_info(video_path)
        frames = extract_frames(video_path, max_frames=self.max_frames)
        self._log(f"📹 {len(frames)} frames extracted ({video_info.get('width')}×{video_info.get('height')}, {video_info.get('fps', 0):.1f}fps)")

        # 2. Extract face crops
        face_crops = crop_faces(frames)
        self._log(f"👤 Face crops ready ({len(face_crops)} frames with faces)")

        # 3. Extract audio
        audio_array, sample_rate = extract_audio_array(video_path)
        self._log(f"🔊 Audio: {len(audio_array) if audio_array is not None else 0} samples @ {sample_rate}Hz")

        # 4. Run all engines in parallel
        engine_scores = {}
        all_violations = []
        stability_modifier = 0.0

        def run_engine(name, fn):
            try:
                return name, fn()
            except Exception as e:
                if self.verbose:
                    traceback.print_exc()
                return name, (0.5, [], 0.0)  # default on failure

        tasks = {
            "CNN-GRU":    lambda: self._run_cnn(face_crops or frames),
            "Frequency":  lambda: self._run_frequency(face_crops or frames),
            "Motion":     lambda: self._run_motion(frames),
            "Teeth":      lambda: self._run_teeth(face_crops or frames),
            "rPPG":       lambda: self._run_rppg(face_crops or frames),
            "Eye":        lambda: self._run_eye(face_crops or frames),
            "HeadPose":   lambda: self._run_head_pose(frames),
            "Hand":       lambda: self._run_hand(frames),
            "SkinTexture":lambda: self._run_skin(face_crops or frames),
            "AudioVisual":lambda: self._run_av(frames, audio_array, sample_rate),
            "Causal":     lambda: self._run_causal(frames),
        }

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(run_engine, name, fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name, result = future.result()
                score, viols = result[0], result[1]
                engine_scores[name] = float(score)
                all_violations.extend(viols)
                self._log(f"  ✓ {name}: {score:.4f} ({len(viols)} violations)")

        # 5. Stability engine (sequential, needs base engine)
        try:
            stab_result = self._run_stability(face_crops or frames)
            stability_modifier = stab_result[2] if len(stab_result) > 2 else 0.0
            all_violations.extend(stab_result[1])
            self._log(f"  ✓ Stability modifier: {stability_modifier:+.4f}")
        except Exception as e:
            self._log(f"  ⚠ Stability engine failed: {e}")

        # 6. Fuse results
        bpm       = getattr(self.rppg, '_bpm', None)
        blink_rate = getattr(self.eye, 'blink_rate_per_min', None)

        result = fuse(
            engine_scores,
            all_violations,
            stability_modifier=stability_modifier,
            elapsed=time.time() - t0,
            video_info=video_info,
            bpm=bpm,
            blink_rate=blink_rate,
        )
        self._log(f"⚡ VERDICT: {result.label} ({result.final_score:.4f}) in {result.elapsed_seconds:.1f}s")
        return result

    # ── engine runners (return score, violations[, modifier]) ─────────────
    def _run_cnn(self, frames):
        score = self.cnn_gru.analyze(frames)
        return score, self.cnn_gru.violations

    def _run_frequency(self, frames):
        score = self.frequency.analyze(frames)
        return score, self.frequency.violations

    def _run_motion(self, frames):
        score = self.motion.analyze(frames)
        return score, self.motion.violations

    def _run_teeth(self, frames):
        score = self.teeth.analyze(frames)
        return score, self.teeth.violations

    def _run_rppg(self, frames):
        score = self.rppg.analyze(frames)
        return score, self.rppg.violations

    def _run_eye(self, frames):
        score = self.eye.analyze(frames)
        return score, self.eye.violations

    def _run_head_pose(self, frames):
        score = self.head_pose.analyze(frames)
        return score, self.head_pose.violations

    def _run_hand(self, frames):
        score = self.hand.analyze(frames)
        return score, self.hand.violations

    def _run_skin(self, frames):
        score = self.skin.analyze(frames)
        return score, self.skin.violations

    def _run_av(self, frames, audio, sr):
        score = self.av.analyze(frames, audio, sr)
        return score, self.av.violations

    def _run_causal(self, frames):
        score = self.causal.analyze(frames)
        return score, self.causal.violations

    def _run_stability(self, frames):
        score, viols, modifier = self.stability.analyze(frames)
        return score, viols, modifier

    def _log(self, msg):
        if self.verbose:
            print(msg)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("Usage: python deepfake_detector.py <video_path> [max_frames]")
        sys.exit(1)

    path = sys.argv[1]
    max_f = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    detector = DeepFakeDetector(max_frames=max_f, verbose=True)
    result = detector.analyze(path)

    print("\n" + "="*60)
    print(f"  VERDICT  : {result.label}")
    print(f"  SCORE    : {result.final_score:.4f}")
    print(f"  CONFIDENCE: {result.confidence} ({result.confidence_pct:.0f}%)")
    print(f"  SUMMARY  : {result.summary}")
    print("="*60)
    print("ENGINE SCORES:")
    for k, v in result.engine_scores.items():
        print(f"  {k:<40} {v:.4f}")
    if result.all_violations:
        print("\nVIOLATIONS:")
        for v in result.all_violations:
            print(f"  ⚠ {v}")
    if result.bpm_detected:
        print(f"\n  Detected BPM: {result.bpm_detected:.0f}")
    if result.blink_rate:
        print(f"  Blink rate: {result.blink_rate:.1f}/min")
    print("="*60)

    # Export report
    report = {
        "verdict": result.label,
        "confidence": result.confidence,
        "confidence_pct": result.confidence_pct,
        "final_score": result.final_score,
        "engine_scores": result.engine_scores,
        "violations": result.all_violations,
        "summary": result.summary,
        "bpm": result.bpm_detected,
        "blink_rate": result.blink_rate,
        "elapsed_s": result.elapsed_seconds,
        "video_info": result.video_info,
    }
    out_path = path.rsplit(".", 1)[0] + "_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_path}")

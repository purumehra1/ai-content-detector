"""
DeepFake Guardian — Core Detection Pipeline v3.1
──────────────────────────────────────────────────
Orchestrates 11 parallel detection engines and fuses results.
All engines follow the same contract:
  • __init__(self)  — no required arguments
  • analyze(frames, ...) → float (0.0 = real, 1.0 = fake)
  • self.violations: List[str]  — populated after analyze()
"""
from __future__ import annotations
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
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
        result   = detector.analyze("video.mp4")
        print(result.label, result.final_score)
    """

    def __init__(self, max_frames: int = 60, verbose: bool = True,
                 device: Optional[str] = None):
        self.max_frames = max_frames
        self.verbose    = verbose

        # Instantiate all engines (no required args)
        self.cnn_gru   = CNNGRUEngine(device=device)
        self.motion    = MotionEngine()
        self.teeth     = TeethEngine()
        self.av        = AudioVisualEngine()
        self.hand      = HandEngine()
        self.stability = StabilityEngine()    # optional base_engine not required
        self.causal    = CausalEngine()
        self.frequency = FrequencyEngine()
        self.rppg      = RPPGEngine()
        self.eye       = EyeEngine()
        self.head_pose = HeadPoseEngine()
        self.skin      = SkinTextureEngine()

    # ── public API ─────────────────────────────────────────────────────────
    def analyze(self, video_path: str) -> FusionResult:
        t0 = time.time()
        self._log(f"🔍 Analyzing: {video_path}")

        # 1. Extract frames and metadata
        video_info = get_video_info(video_path)
        frames     = extract_frames(video_path, max_frames=self.max_frames)
        self._log(f"📹 {len(frames)} frames | "
                  f"{video_info.get('width')}×{video_info.get('height')} | "
                  f"{video_info.get('fps', 0):.1f} fps")

        if not frames:
            self._log("⚠️  No frames extracted — returning neutral result")
            return fuse({}, [], elapsed=time.time()-t0, video_info=video_info)

        # 2. Face crops (fallback to full frame if no face detected)
        face_crops = crop_faces(frames)
        self._log(f"👤 Face crops: {len([f for f in face_crops if f is not None])}/{len(frames)}")

        # 3. Audio
        audio_array, sample_rate = extract_audio_array(video_path)
        has_audio = audio_array is not None and len(audio_array) > 0
        self._log(f"🔊 Audio: {'OK' if has_audio else 'not found'}")

        # 4. Run engines in parallel
        engine_scores  : dict  = {}
        all_violations : List[str] = []
        stability_mod  : float = 0.0

        def _safe_run(name: str, fn) -> Tuple[str, float, List[str]]:
            try:
                score, viols = fn()
                return name, float(score), viols
            except Exception as exc:
                if self.verbose:
                    traceback.print_exc()
                return name, 0.5, [f"[{name}] Engine error: {exc}"]

        tasks = {
            "CNN-GRU":    lambda: (self.cnn_gru.analyze(face_crops or frames),
                                   self.cnn_gru.violations),
            "Frequency":  lambda: (self.frequency.analyze(face_crops or frames),
                                   self.frequency.violations),
            "Motion":     lambda: (self.motion.analyze(frames),
                                   self.motion.violations),
            "Teeth":      lambda: (self.teeth.analyze(face_crops or frames),
                                   self.teeth.violations),
            "rPPG":       lambda: (self.rppg.analyze(face_crops or frames),
                                   self.rppg.violations),
            "Eye":        lambda: (self.eye.analyze(face_crops or frames),
                                   self.eye.violations),
            "HeadPose":   lambda: (self.head_pose.analyze(frames),
                                   self.head_pose.violations),
            "Hand":       lambda: (self.hand.analyze(frames),
                                   self.hand.violations),
            "SkinTexture":lambda: (self.skin.analyze(face_crops or frames),
                                   self.skin.violations),
            "AudioVisual":lambda: (self.av.analyze(frames, audio_array, sample_rate),
                                   self.av.violations),
            "Causal":     lambda: (self.causal.analyze(frames),
                                   self.causal.violations),
        }

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_safe_run, name, fn): name
                       for name, fn in tasks.items()}
            for future in as_completed(futures):
                name, score, viols = future.result()
                engine_scores[name] = score
                all_violations.extend(viols)
                self._log(f"  ✓ {name:<14}: {score:.4f}  ({len(viols)} violations)")

        # 5. Stability engine (sequential — lightweight)
        try:
            stab_score, stab_viols, stability_mod = self.stability.analyze(
                face_crops or frames
            )
            all_violations.extend(stab_viols)
            self._log(f"  ✓ Stability:      modifier {stability_mod:+.4f}")
        except Exception as exc:
            if self.verbose:
                traceback.print_exc()
            self._log(f"  ⚠ Stability engine error: {exc}")

        # 6. Fuse
        result = fuse(
            engine_scores,
            all_violations,
            stability_modifier=stability_mod,
            elapsed=time.time() - t0,
            video_info=video_info,
            bpm=getattr(self.rppg, '_bpm', None),
            blink_rate=getattr(self.eye, 'blink_rate_per_min', None),
        )
        self._log(f"\n⚡ VERDICT: {result.label} | Score: {result.final_score:.4f} "
                  f"| {result.confidence} ({result.confidence_pct:.0f}%) "
                  f"| {result.elapsed_seconds:.1f}s")
        return result

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python deepfake_detector.py <video_path> [max_frames]")
        sys.exit(1)

    path  = sys.argv[1]
    max_f = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    detector = DeepFakeDetector(max_frames=max_f, verbose=True)
    result   = detector.analyze(path)

    print("\n" + "="*60)
    print(f"  VERDICT   : {result.label}")
    print(f"  SCORE     : {result.final_score:.4f}")
    print(f"  CONFIDENCE: {result.confidence} ({result.confidence_pct:.0f}%)")
    print(f"  SUMMARY   : {result.summary}")
    if result.bpm_detected:
        print(f"  BPM       : {result.bpm_detected:.0f}")
    if result.blink_rate:
        print(f"  BLINK RATE: {result.blink_rate:.1f}/min")
    print("="*60)
    print("ENGINE SCORES:")
    for k, v in result.engine_scores.items():
        print(f"  {k:<45} {v:.4f}")
    if result.all_violations:
        print(f"\nVIOLATIONS ({len(result.all_violations)}):")
        for v in result.all_violations:
            print(f"  ⚠ {v}")
    print("="*60)

    out = path.rsplit(".", 1)[0] + "_report.json"
    with open(out, "w") as f:
        json.dump({
            "verdict": result.label, "score": result.final_score,
            "confidence": result.confidence, "confidence_pct": result.confidence_pct,
            "engine_scores": result.engine_scores, "violations": result.all_violations,
            "summary": result.summary, "bpm": result.bpm_detected,
            "blink_rate": result.blink_rate, "elapsed_s": result.elapsed_seconds,
            "video_info": result.video_info,
        }, f, indent=2)
    print(f"Report saved: {out}")

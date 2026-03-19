"""
DeepFake Guardian — Core Detection Pipeline v3.2
──────────────────────────────────────────────────
11 detection engines + XAI (Explainability) engine.
"""
from __future__ import annotations
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
import numpy as np

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
from engines.xai_engine          import XAIEngine
from fusion.weighted_fusion      import fuse, FusionResult
from utils.video_utils           import extract_frames, extract_audio_array, get_video_info
from utils.face_utils            import crop_faces


class DeepFakeDetector:
    """Main deepfake detection pipeline with XAI."""

    def __init__(self, max_frames: int = 60, verbose: bool = True,
                 device: Optional[str] = None):
        self.max_frames = max_frames
        self.verbose    = verbose
        self.cnn_gru    = CNNGRUEngine(device=device)
        self.motion     = MotionEngine()
        self.teeth      = TeethEngine()
        self.av         = AudioVisualEngine()
        self.hand       = HandEngine()
        self.stability  = StabilityEngine()
        self.causal     = CausalEngine()
        self.frequency  = FrequencyEngine()
        self.rppg       = RPPGEngine()
        self.eye        = EyeEngine()
        self.head_pose  = HeadPoseEngine()
        self.skin       = SkinTextureEngine()
        self.xai        = XAIEngine()

    def analyze(self, video_path: str) -> FusionResult:
        t0 = time.time()
        self._log(f"🔍 Analyzing: {video_path}")

        video_info = get_video_info(video_path)
        frames     = extract_frames(video_path, max_frames=self.max_frames)
        self._log(f"📹 {len(frames)} frames | {video_info.get('width')}×"
                  f"{video_info.get('height')} | {video_info.get('fps', 0):.1f}fps")

        if not frames:
            return fuse({}, [], elapsed=time.time()-t0, video_info=video_info)

        face_crops   = crop_faces(frames)
        audio_array, sample_rate = extract_audio_array(video_path)
        has_audio    = audio_array is not None and len(audio_array) > 0
        self._log(f"👤 Faces: {len([f for f in face_crops if f is not None])}/{len(frames)} "
                  f"| Audio: {'✓' if has_audio else '✗'}")

        engine_scores : dict       = {}
        all_violations: List[str]  = []
        stability_mod : float      = 0.0

        def _safe(name: str, fn) -> Tuple[str, float, List[str]]:
            try:
                score, viols = fn()
                return name, float(score), list(viols)
            except Exception as exc:
                if self.verbose:
                    traceback.print_exc()
                return name, 0.5, [f"[{name}] Engine error: {exc}"]

        crops_or_frames = face_crops or frames
        tasks = {
            "CNN-GRU":    lambda: (self.cnn_gru.analyze(crops_or_frames),   self.cnn_gru.violations),
            "Frequency":  lambda: (self.frequency.analyze(crops_or_frames), self.frequency.violations),
            "Motion":     lambda: (self.motion.analyze(frames),             self.motion.violations),
            "Teeth":      lambda: (self.teeth.analyze(crops_or_frames),     self.teeth.violations),
            "rPPG":       lambda: (self.rppg.analyze(crops_or_frames),      self.rppg.violations),
            "Eye":        lambda: (self.eye.analyze(crops_or_frames),       self.eye.violations),
            "HeadPose":   lambda: (self.head_pose.analyze(frames),          self.head_pose.violations),
            "Hand":       lambda: (self.hand.analyze(frames),               self.hand.violations),
            "SkinTexture":lambda: (self.skin.analyze(crops_or_frames),      self.skin.violations),
            "AudioVisual":lambda: (self.av.analyze(frames, audio_array, sample_rate), self.av.violations),
            "Causal":     lambda: (self.causal.analyze(frames),             self.causal.violations),
        }

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_safe, n, fn): n for n, fn in tasks.items()}
            for future in as_completed(futures):
                name, score, viols = future.result()
                engine_scores[name] = score
                all_violations.extend(viols)
                self._log(f"  ✓ {name:<14}: {score:.4f} ({len(viols)} violations)")

        # Stability (sequential)
        try:
            stab_score, stab_viols, stability_mod = self.stability.analyze(crops_or_frames)
            all_violations.extend(stab_viols)
            self._log(f"  ✓ Stability     : modifier {stability_mod:+.4f}")
        except Exception as exc:
            self._log(f"  ⚠ Stability: {exc}")

        # XAI attribution (uses CNN-GRU as score function)
        xai_region_scores = {}
        xai_heatmap       = None
        xai_explanation   = ""
        try:
            baseline_cnn = engine_scores.get("CNN-GRU", 0.5)
            self.xai.analyze(
                crops_or_frames[:6],
                score_fn=lambda fs: self.cnn_gru.analyze(fs),
                baseline_score=baseline_cnn,
            )
            xai_region_scores = self.xai.region_scores
            xai_heatmap       = self.xai.heatmap
            xai_explanation   = self.xai.explanation
            self._log(f"  ✓ XAI           : {len(xai_region_scores)} regions attributed")
        except Exception as exc:
            self._log(f"  ⚠ XAI: {exc}")

        # Fuse
        result = fuse(
            engine_scores,
            all_violations,
            stability_modifier=stability_mod,
            elapsed=time.time() - t0,
            video_info=video_info,
            bpm=getattr(self.rppg, '_bpm', None),
            blink_rate=getattr(self.eye, 'blink_rate_per_min', None),
        )

        # Attach XAI data to result
        result.xai_region_scores = xai_region_scores
        result.xai_heatmap       = xai_heatmap
        result.xai_explanation   = xai_explanation
        result.face_crops        = crops_or_frames[:8]  # for display

        self._log(f"\n⚡ VERDICT: {result.label} | {result.final_score:.4f} "
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
        print("Usage: python deepfake_detector.py <video> [max_frames]")
        sys.exit(1)
    path  = sys.argv[1]
    max_f = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    r = DeepFakeDetector(max_frames=max_f, verbose=True).analyze(path)
    print(f"\nVERDICT: {r.label}  SCORE: {r.final_score:.4f}  "
          f"CONFIDENCE: {r.confidence} ({r.confidence_pct:.0f}%)")
    print(f"SUMMARY: {r.summary}")
    for v in r.all_violations[:10]:
        print(f"  ⚠ {v}")

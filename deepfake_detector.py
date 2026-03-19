"""
DeepFake Guardian — Main Detection Pipeline
────────────────────────────────────────────
Orchestrates all 7 detection engines in parallel using ThreadPoolExecutor
and fuses their outputs into a final classification.

Usage:
    detector = DeepFakeDetector()
    result = detector.analyze(video_path)
    print(result.label, result.confidence_pct)
"""
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.video_utils import extract_all_frames, extract_audio, get_video_info
from engines.cnn_gru_engine import CNNGRUEngine
from engines.motion_engine import MotionEngine
from engines.teeth_engine import TeethEngine
from engines.audio_visual_engine import AudioVisualEngine
from engines.hand_engine import HandEngine
from engines.stability_engine import StabilityEngine
from engines.causal_engine import CausalEngine
from fusion.weighted_fusion import fuse, FusionResult


class DeepFakeDetector:
    """
    Unified hybrid deepfake detection pipeline.
    All engines run in parallel; results fused via weighted scoring.
    """

    def __init__(self, max_frames: int = 60, verbose: bool = True):
        self.max_frames = max_frames
        self.verbose = verbose

        # Lazy-init engines (loaded on first use)
        self._cnn_gru    = CNNGRUEngine()
        self._motion     = MotionEngine()
        self._teeth      = TeethEngine()
        self._av         = AudioVisualEngine()
        self._hand       = HandEngine()
        self._stability  = StabilityEngine()
        self._causal     = CausalEngine()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[DeepFakeGuardian] {msg}")

    def analyze(self, video_path: str) -> FusionResult:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        t_start = time.perf_counter()
        self._log(f"Starting analysis: {os.path.basename(video_path)}")

        # ── Step 1: Extract frames and audio ──────────────────────────────────
        info = get_video_info(video_path)
        self._log(f"Video: {info['width']}×{info['height']} @ {info['fps']:.1f}fps, {info['duration']:.1f}s, {info['total_frames']} frames")

        frames, fps = extract_all_frames(video_path, max_frames=self.max_frames)
        self._log(f"Extracted {len(frames)} frames for analysis")

        audio_path = extract_audio(video_path)
        self._log(f"Audio: {'extracted to ' + audio_path if audio_path else 'not available'}")

        # Extract RMS for causal engine (shared computation)
        rms_timeline = None
        if audio_path:
            try:
                import librosa
                y, sr = librosa.load(audio_path, sr=16000, mono=True)
                hop_len = int(sr / fps)
                rms = librosa.feature.rms(y=y, hop_length=hop_len)[0]
                rms_timeline = np.array(rms[:len(frames)])
            except Exception:
                pass

        # ── Step 2: Run all engines in parallel ───────────────────────────────
        self._log("Running parallel detection engines...")
        results = {}

        def run_cnn():
            self._log("  → CNN+GRU engine starting...")
            r = self._cnn_gru.analyze(frames)
            self._log(f"  ✓ CNN+GRU: {r['score']:.3f}")
            return "cnn", r

        def run_motion():
            self._log("  → Motion engine starting...")
            r = self._motion.analyze(frames)
            self._log(f"  ✓ Motion: {r['score']:.3f}")
            return "motion", r

        def run_teeth():
            self._log("  → Teeth engine starting...")
            r = self._teeth.analyze(frames)
            self._log(f"  ✓ Teeth: {r['score']:.3f}")
            return "teeth", r

        def run_av():
            self._log("  → Audio-Visual engine starting...")
            r = self._av.analyze(frames, audio_path, fps)
            self._log(f"  ✓ Audio-Visual: {r['score']:.3f}")
            return "av", r

        def run_hand():
            self._log("  → Hand engine starting...")
            r = self._hand.analyze(frames)
            self._log(f"  ✓ Hand: {r['score']:.3f}")
            return "hand", r

        def run_stability():
            self._log("  → Stability engine starting...")
            r = self._stability.analyze(frames)
            self._log(f"  ✓ Stability: {r['score']:.3f}")
            return "stability", r

        def run_causal():
            self._log("  → Causal engine starting...")
            r = self._causal.analyze(frames, rms_timeline)
            self._log(f"  ✓ Causal: {r['score']:.3f}")
            return "causal", r

        tasks = [run_cnn, run_motion, run_teeth, run_av, run_hand, run_stability, run_causal]

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(fn): fn.__name__ for fn in tasks}
            for future in as_completed(futures):
                try:
                    key, result = future.result()
                    results[key] = result
                except Exception as e:
                    name = futures[future]
                    self._log(f"  ✗ Engine error in {name}: {e}")
                    results[name.replace("run_", "")] = {"score": 0.5, "violations": [f"Engine error: {e}"]}

        # ── Step 3: Fuse results ───────────────────────────────────────────────
        self._log("Fusing engine scores...")
        fusion_result = fuse(
            cnn_result=results.get("cnn", {"score": 0.5, "violations": []}),
            motion_result=results.get("motion", {"score": 0.5, "violations": []}),
            teeth_result=results.get("teeth", {"score": 0.5, "violations": []}),
            hand_result=results.get("hand", {"score": 0.3, "violations": []}),
            av_result=results.get("av", {"score": 0.5, "violations": []}),
            causal_result=results.get("causal", {"score": 0.5, "violated_rules": []}),
            stability_result=results.get("stability", {"score": 0.5, "violations": []}),
        )

        elapsed = time.perf_counter() - t_start
        self._log(f"Analysis complete in {elapsed:.1f}s → {fusion_result.label} ({fusion_result.confidence_pct:.0f}% confidence)")

        # Clean up temp audio
        if audio_path and os.path.isfile(audio_path):
            os.unlink(audio_path)

        # Attach raw engine results for debugging
        fusion_result.__dict__["_engine_results"] = results
        fusion_result.__dict__["elapsed_seconds"] = round(elapsed, 2)
        fusion_result.__dict__["video_info"] = info

        return fusion_result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python deepfake_detector.py <video_path>")
        sys.exit(1)
    detector = DeepFakeDetector()
    result = detector.analyze(sys.argv[1])
    print(f"\n{'='*50}")
    print(f"VERDICT: {result.label} ({result.confidence} confidence, {result.confidence_pct:.0f}%)")
    print(f"Final Score: {result.final_score:.4f}")
    print(f"\nEngine Breakdown:")
    for k, v in result.engine_scores.items():
        print(f"  {k}: {v}")
    print(f"\nViolations ({len(result.all_violations)}):")
    for v in result.all_violations:
        print(f"  ⚠ {v}")
    print(f"\nSummary: {result.summary}")
    print(f"{'='*50}")

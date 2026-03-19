"""
DeepFake Guardian — Detection Engines
10 specialized parallel detection engines.
"""
from .cnn_gru_engine      import CNNGRUEngine
from .motion_engine       import MotionEngine
from .teeth_engine        import TeethEngine
from .audio_visual_engine import AudioVisualEngine
from .hand_engine         import HandEngine
from .stability_engine    import StabilityEngine
from .causal_engine       import CausalEngine
from .frequency_engine    import FrequencyEngine
from .rppg_engine         import RPPGEngine
from .eye_engine          import EyeEngine
from .head_pose_engine    import HeadPoseEngine
from .skin_texture_engine import SkinTextureEngine

__all__ = [
    "CNNGRUEngine", "MotionEngine", "TeethEngine", "AudioVisualEngine",
    "HandEngine", "StabilityEngine", "CausalEngine", "FrequencyEngine",
    "RPPGEngine", "EyeEngine", "HeadPoseEngine", "SkinTextureEngine",
]

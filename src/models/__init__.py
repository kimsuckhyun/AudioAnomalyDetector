# AudioAnomalyDetector/src/models/__init__.py

"""
Model subpackage for AudioAnomalyDetector:
- despawn: Enhanced DeSpaWN wavelet model
- transformer: Enhanced masked audio transformer
- vqvae: VQ-VAE fusion anomaly detector
"""

from .despawn import EnhancedDeSpaWN, Kernel, LowPassWave, HighPassWave, LowPassTrans, HighPassTrans, HardThresholdAssym, ChannelAttention
from .transformer import EnhancedMaskedAudioTransformer, EnhancedTransformerBlock, TemporalConv
from .vqvae import AudioAnomalyVQVAE, VectorQuantizer, HierarchicalFusion

__all__ = [
    "EnhancedDeSpaWN", "Kernel", "LowPassWave", "HighPassWave", "LowPassTrans", "HighPassTrans", "HardThresholdAssym", "ChannelAttention",
    "EnhancedMaskedAudioTransformer", "EnhancedTransformerBlock", "TemporalConv",
    "AudioAnomalyVQVAE", "VectorQuantizer", "HierarchicalFusion",
]

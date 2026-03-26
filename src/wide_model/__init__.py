"""
Wide Encoder-Decoder model package (1st-place-scale).

Modules
-------
model    — WideEncoderDecoder architecture (2048-wide, LayerNorm, SVD output)
losses   — Multi-component loss: PearsonLoss, CombinedLoss
trainer  — WideTrainer with OneCycleLR and SVD reconstruction support
"""

from .model   import WideEncoderDecoder
from .losses  import PearsonLoss, CombinedLoss
from .trainer import WideTrainer, svd_predict_mrpc

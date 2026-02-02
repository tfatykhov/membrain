# Baseline implementations for benchmark comparison.

from bench.baselines.cosine import CosineBaseline

__all__ = ["CosineBaseline"]

# Optional imports — skip if dependencies not installed
try:
    from bench.baselines.faiss_flat import FAISSFlatBaseline, FAISSIVFBaseline
    __all__.extend(["FAISSFlatBaseline", "FAISSIVFBaseline"])
except ImportError:
    pass

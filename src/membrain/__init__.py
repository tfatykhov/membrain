"""
Membrain - Neuromorphic Memory Bridge for LLM Agents

A Spiking Neural Network (SNN) based memory system that provides
associative recall and continuous learning for AI agents.
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring numpy at import time
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "FlyHash":
        from membrain.encoder import FlyHash

        return FlyHash
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["FlyHash", "__version__"]

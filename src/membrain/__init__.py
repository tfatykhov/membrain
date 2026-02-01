"""
Membrain - Neuromorphic Memory Bridge for LLM Agents

A Spiking Neural Network (SNN) based memory system that provides
associative recall and continuous learning for AI agents.
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring heavy deps at import time
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "FlyHash":
        from membrain.encoder import FlyHash

        return FlyHash
    if name == "BiCameralMemory":
        from membrain.core import BiCameralMemory

        return BiCameralMemory
    if name == "MemoryEntry":
        from membrain.core import MemoryEntry

        return MemoryEntry
    if name == "RecallResult":
        from membrain.core import RecallResult

        return RecallResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FlyHash", "BiCameralMemory", "MemoryEntry", "RecallResult", "__version__"]

"""
TruthSim: Truth-Preserving Noisy Patient Simulator
===================================================

A controllable framework for realistic medical AI evaluation.
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

from .utils.config import load_config
from .utils.llm_client import LLMClient

__all__ = [
    "load_config",
    "LLMClient",
]

"""Command-line interface module."""

from .config import load_config
from .pipeline import TrainingPipeline

__all__ = ["load_config", "TrainingPipeline"]

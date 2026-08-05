"""Backward-compatible imports for the shared training workflow."""

from .workflows.training import main, run_pipeline

__all__ = ["main", "run_pipeline"]

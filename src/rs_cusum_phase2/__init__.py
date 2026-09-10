"""Phase-2 synthetic calibration layer; never imports OhioT1DM data."""

from .generator import SyntheticDataset, generate_dataset
from .protocol import load_phase2_config

__all__ = ["SyntheticDataset", "generate_dataset", "load_phase2_config"]

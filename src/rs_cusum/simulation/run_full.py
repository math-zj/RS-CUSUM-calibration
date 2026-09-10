"""Locked Phase-2 full Monte Carlo runner scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def require_phase2_full_authorization(config_path: str | Path) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["protocol"]["status"] != cfg["protocol"]["full_authorization_required"]:
        raise RuntimeError("full simulation is locked: Phase 2 authorization is absent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rs_cusum_simulation.yaml")
    args = parser.parse_args()
    require_phase2_full_authorization(args.config)
    raise RuntimeError("Phase 2 full execution has not been enabled in this Phase 1 implementation")


if __name__ == "__main__":
    main()

"""Locked Phase-2 pilot runner scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def require_phase2_pilot_authorization(config_path: str | Path) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["protocol"]["status"] != cfg["protocol"]["pilot_authorization_required"]:
        raise RuntimeError("pilot simulation is locked: Phase 2 authorization is absent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rs_cusum_simulation.yaml")
    args = parser.parse_args()
    require_phase2_pilot_authorization(args.config)
    raise RuntimeError("Phase 2 pilot execution has not been enabled in this Phase 1 implementation")


if __name__ == "__main__":
    main()

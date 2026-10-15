"""Locate an optional user-provided Hydra configuration directory."""

from pathlib import Path


def default_config_dir():
    path = Path(__file__).resolve().parents[2] / "conf"
    return str(path) if path.is_dir() else None

"""Shared pytest configuration."""

import sys
from pathlib import Path

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

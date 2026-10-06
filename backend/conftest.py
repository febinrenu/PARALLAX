"""Puts backend/ on sys.path so tests can import `medproof` before P4's pyproject lands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

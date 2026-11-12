"""Deprecated entry point retained for compatibility.

Use ``scripts/run_deterministic_eval.py`` for the CI gate or
``scripts/run_real_eval.py`` for real-model measurements.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_deterministic_eval import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

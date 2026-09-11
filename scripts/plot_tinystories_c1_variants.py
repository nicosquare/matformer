#!/usr/bin/env python3
"""Retain the original C1 plotting command; defaults to --family C1."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_tinystories_concat_variants import main


if __name__ == '__main__':
    main()

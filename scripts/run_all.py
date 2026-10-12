#!/usr/bin/env python
"""CPST-MalariaNet - command-line entry point.

Usage
-----
    python scripts/run_all.py --data data/raw/BdiDATASETS.xlsx --out outputs
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_full_pipeline
from src.data_loader import DEFAULT_FILEPATH

def main():
    ap = argparse.ArgumentParser(description="CPST-MalariaNet pipeline")
    ap.add_argument("--data", default=DEFAULT_FILEPATH,
                    help="Path to BdiDATASETS.xlsx")
    ap.add_argument("--out", default="outputs", help="Output directory")
    args = ap.parse_args()
    run_full_pipeline(filepath=args.data, out_dir=args.out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

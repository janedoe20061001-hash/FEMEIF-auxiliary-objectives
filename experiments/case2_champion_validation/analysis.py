"""Reuse the audited paired bootstrap implementation with unified labels."""
from pathlib import Path
import importlib.util

PATH = Path(__file__).resolve().parent.parent / "case2_extended_pairs_validation" / "analysis.py"
SPEC = importlib.util.spec_from_file_location("case2_champion_shared_analysis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
paired = MODULE.paired
analyze = MODULE.analyze


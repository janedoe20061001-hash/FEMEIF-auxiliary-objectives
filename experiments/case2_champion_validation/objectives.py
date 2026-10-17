"""Unified Case2 objectives for the independent champion validation.

The original, already-audited Chinese-PDF and English-paper implementations are
loaded without formula changes.  This adapter permits objectives from both
families to guide one NSGA-II run while keeping the two primary MSE columns
exactly once and in their established order.
"""
from pathlib import Path
import importlib.util
import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHINESE = _load("case2_champion_chinese_objectives",
                EXPERIMENTS / "case2_extended_pairs_validation" / "objectives.py")
PAPER = _load("case2_champion_paper_objectives",
              EXPERIMENTS / "case2_paper5_combinations" / "objectives.py")

CATALOG = list(CHINESE.CATALOG) + list(PAPER.CATALOG)
AUX = tuple(row[1] for row in CATALOG)
DEFAULTS = {"chinese": dict(CHINESE.DEFAULTS), "paper": dict(PAPER.DEFAULTS)}


class Objectives:
    def __init__(self, frequency, settings=None, aux=(), transform="loss",
                 epsilon=1e-12, target_z=None):
        settings = DEFAULTS if settings is None else settings
        if settings != DEFAULTS:
            raise ValueError("Champion protocol constants changed")
        if len(set(aux)) != len(aux) or any(name not in AUX for name in aux):
            raise ValueError("Unknown/repeated champion auxiliary")
        if transform != "loss":
            raise ValueError("Champion protocol uses raw minimization losses")
        self.aux = tuple(aux)
        self.chinese_aux = tuple(a for a in self.aux if a in CHINESE.AUX)
        self.paper_aux = tuple(a for a in self.aux if a in PAPER.AUX)
        self.chinese = CHINESE.Objectives(
            frequency, settings["chinese"], self.chinese_aux, transform,
            epsilon, target_z)
        self.paper = PAPER.Objectives(
            frequency, settings["paper"], self.paper_aux, transform,
            epsilon, target_z)
        self.f = self.chinese.f

    def target_summary(self):
        return {
            "chinese_pdf": self.chinese.target_summary(),
            "english_paper": self.paper.target_summary(),
            "unified_objective_order": list(self.aux),
        }

    def evaluate_spectra(self, z):
        chinese = self.chinese.evaluate_spectra(z)
        paper = self.paper.evaluate_spectra(z)
        np.testing.assert_allclose(chinese[:, :2], paper[:, :2], rtol=0, atol=0)
        values = {name: chinese[:, 2+i]
                  for i, name in enumerate(self.chinese_aux)}
        values.update({name: paper[:, 2+i]
                       for i, name in enumerate(self.paper_aux)})
        return np.column_stack([chinese[:, :2]] + [values[name] for name in self.aux])


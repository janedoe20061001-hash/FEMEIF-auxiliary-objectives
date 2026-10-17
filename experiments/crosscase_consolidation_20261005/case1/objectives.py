from global_metrics import AUX, CATALOG, DEFAULTS, GlobalObjectives
from legacy_objectives import TARGET, LOWER, UPPER, impedance
import numpy as np
class Objectives(GlobalObjectives):
    def __init__(self,frequency,settings=None,aux=(),transform="loss",epsilon=None,target_z=None):
        target=impedance(frequency,TARGET)[0] if target_z is None else target_z
        super().__init__(frequency,settings,aux,transform,epsilon,target)
    def evaluate(self,x):
        return self.evaluate_spectra(impedance(self.f,10**np.atleast_2d(x)))

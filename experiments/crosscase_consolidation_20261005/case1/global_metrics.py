"""Frozen common experiment: local pairs, cumulative length, residual variants."""
from pathlib import Path
import numpy as np
from legacy_objectives import Objectives as Legacy, DEFAULTS as LEGACY_DEFAULTS
from length_metrics import GlobalObjectives as Length
from trend_kernel import score, cumulative_same_coordinates

DIRECTION='reactance_crossing_direction_consistency'
SHIFT='characteristic_frequency_shift_error_decades'
SPEED='transition_speed_relative_error'
LENGTH='cumulative_length_logpolar'
TREND='all_interval_mean_residual_logpolar'
CUMULATIVE='bidirectional_residual_logpolar'
CATALOG=[('I-03',DIRECTION,'过零方向'),('II-01',SHIFT,'局部频移'),
         ('II-02',SPEED,'转变速度'),('R-05-C',LENGTH,'累计弧长'),
         ('T-01',TREND,'全区间平均残差'),('G-01-LP',CUMULATIVE,'同坐标双向累计残差')]
AUX=tuple(x[1] for x in CATALOG)
GROUPS=[(),(DIRECTION,SHIFT),(DIRECTION,SPEED),(LENGTH,),(TREND,),(CUMULATIVE,)]
DEFAULTS={}

class GlobalObjectives:
    def __init__(self,frequency,settings=None,aux=(),transform='loss',epsilon=None,target_z=None):
        if settings not in (None,{}) or transform!='loss' or tuple(aux) not in GROUPS:
            raise ValueError('Unfrozen objectives')
        self.aux=tuple(aux)
        self.geometry=Length(frequency,aux=(LENGTH,) if LENGTH in aux else (),target_z=target_z)
        self.f=self.geometry.f;self.z=self.geometry.z
        self.u=self.geometry.u;self.scale=self.geometry.scale
        self.local=Legacy(frequency,aux=aux,target_z=target_z) if DIRECTION in aux else None
        self.K=np.load(Path(__file__).with_name('trend_kernel.npy')) if TREND in aux else None

    def target_summary(self):
        return dict(**self.geometry.target_summary(),
                    local_constants=LEGACY_DEFAULTS,
                    local_target=Legacy(self.f,target_z=self.z).target_summary(),
                    protocol='crosscase_consolidation_20261005_v1')

    def evaluate_spectra(self,z):
        if self.local is not None:
            return self.local.evaluate_spectra(z)
        out=self.geometry.evaluate_spectra(z)
        if TREND in self.aux or CUMULATIVE in self.aux:
            z=np.atleast_2d(z)
            r=np.stack([20*np.log10(abs(z))-self.geometry.a,
                        np.angle(z)-self.geometry.phase],axis=-1)/self.scale
            value=score(r,self.K) if TREND in self.aux else cumulative_same_coordinates(r,self.u)
            out=np.column_stack([out,value])
        if not np.isfinite(out).all() or np.any(out<0):
            raise FloatingPointError('Invalid objectives')
        return out

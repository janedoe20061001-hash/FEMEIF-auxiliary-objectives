"""Post-evaluation selection only. This module never changes NSGA-II populations."""
import numpy as np


def select_wsmse(f,alpha=17.38):
    f=np.asarray(f,float)
    if f.ndim!=2 or len(f)==0 or f.shape[1]<2 or not np.all(np.isfinite(f[:,:2])):
        raise ValueError("Expected a nonempty finite population with two primary columns")
    if not np.isfinite(alpha) or alpha<=0:
        raise ValueError("WSMSE phase weight must be finite and positive")
    scores=f[:,0]+alpha*f[:,1]
    index=int(np.argmin(scores)) # Exact ties keep the first final-population row.
    return dict(wsmse_selected_index=index,wsmse_selected_alpha=float(alpha),
        wsmse_selected_magnitude_mse=float(f[index,0]),
        wsmse_selected_phase_mse=float(f[index,1]),
        wsmse_selected_value=float(scores[index]))

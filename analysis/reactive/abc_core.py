"""Established ABC distance and switch lookup from the October 19 corrected workflow.
Source: context/corrected_reactive_source/scripts/recompute_reactive_rules_5scan.py.
Only numerical helpers are reused. Policies and observation schedules are implemented separately.
"""
import numpy as np
def abc_noisy(ref: np.ndarray, nadirs: np.ndarray, queries: np.ndarray,
              k=25, slope_weight=2.0, epsilon=1e-8, batch=96):
    """Original reference-only ABC Euclidean-level/slope distance and inverse weights."""
    sd_level = float(np.std(ref.ravel())) or 1.0
    sd_slope = float(np.std(np.diff(ref,axis=1).ravel())) or 1.0
    rl, rs = ref/sd_level, np.diff(ref,axis=1)/sd_slope
    rl2 = np.sum(rl*rl,axis=1)
    rs2 = np.sum(rs*rs,axis=1)
    out = np.empty(len(queries),float)
    k=min(k,len(nadirs))
    for start in range(0,len(queries),batch):
        q = queries[start:start+batch]
        ql, qs = q/sd_level, np.diff(q,axis=1)/sd_slope
        dl=np.sqrt(np.maximum(np.sum(ql*ql,axis=1)[:,None]+rl2[None,:]-2*ql@rl.T,0)/4)
        ds=np.sqrt(np.maximum(np.sum(qs*qs,axis=1)[:,None]+rs2[None,:]-2*qs@rs.T,0)/3)
        distances=dl+slope_weight*ds
        ix=np.argpartition(distances,k-1,axis=1)[:,:k]
        w=1/(np.take_along_axis(distances,ix,axis=1)+epsilon)
        out[start:start+batch]=(w*nadirs[ix]).sum(axis=1)/w.sum(axis=1)
    return out



def score_at_switch(surface: np.ndarray, switches: np.ndarray) -> np.ndarray:
    """Switch after treatment census t uses saved 1-based lag t+1."""
    switches = np.asarray(switches, float)
    if switches.shape[-1] != surface.shape[0]:
        raise ValueError('Incorrect patient axis for lookup')
    mask = np.isfinite(switches)
    if np.any((switches[mask] < 0) | (switches[mask]+1 > surface.shape[1])):
        raise ValueError('Requested a switch outside simulated lag grid')
    if not np.all(np.isclose(switches[mask], np.rint(switches[mask]))):
        raise ValueError('Switch times must be integer simulation cycles')
    safe = np.where(mask, switches, 0).astype(int)
    indices = np.broadcast_to(np.arange(surface.shape[0]), switches.shape)
    selected = surface[indices, safe]
    return np.where(mask, selected, np.nan)


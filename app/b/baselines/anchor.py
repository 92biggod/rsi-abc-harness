"""Retain one frozen plain parent for random-label projections with few bearings."""
import numpy as np


def sparse_anchor(parent, labels, groups, group_count):
    if not 1 <= group_count <= 3:
        return None
    if parent.anchor is None:
        if 'rul_auxiliary' in parent.parent.meta or parent.stages is not None or parent.parent.meta.get('rul_harmonic') or parent.parent.meta.get('rul_aux_weight'):
            return None
    densities=[]
    for name in np.unique(groups):
        observed=labels[groups==name]
        if len(observed)<8 or np.max(abs(observed-np.rint(observed)))>1e-7:
            continue
        distinct=np.unique(np.rint(observed))
        span=distinct[-1]-distinct[0]+1
        if span>=24:
            densities.append(len(distinct)/span)
    if not densities or np.median(densities)>=.45:
        return None
    return parent.anchor if parent.anchor is not None else parent.parent

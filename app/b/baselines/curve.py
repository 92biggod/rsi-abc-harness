"""Cross-bearing degradation curves and terminal-life detector fitting."""

import json

from .model import Model

import numpy as np

def decreasing_levels(means, mass):
    blocks = []
    for i, (mean, count) in enumerate(zip(means, mass)):
        blocks.append((i, i+1, mean*count, count))
        while len(blocks) > 1 and blocks[-2][2]/blocks[-2][3] < blocks[-1][2]/blocks[-1][3]:
            a, b = blocks[-2:]
            blocks[-2:] = [(a[0], b[1], a[2]+b[2], a[3]+b[3])]
    levels = means.copy()
    for start, end, total, count in blocks:
        levels[start:end] = total / count
    return levels

def rank(values):
    _, index, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1)/2)[index]

def ranks_and_associations(x, y):
    ranks = np.column_stack([rank(v) for v in x.T])
    ranks -= ranks.mean(axis=0)
    ranks /= np.maximum(1e-12, np.linalg.norm(ranks, axis=0))
    target = rank(y)
    target -= target.mean()
    return ranks.T @ (target / max(np.linalg.norm(target), 1e-12)), ranks.T @ ranks

def fit(data, groups, keep, high_band_only=False, bearing_exponent=0.5):
    x, y, labels = data['x'][keep], data['y'][keep], groups[keep]
    names, counts = np.unique(labels, return_counts=True)
    masks = [labels == g for g in names]
    weights = counts.astype(float) ** bearing_exponent
    weights /= weights.sum()
    associations, duplicates = zip(*(ranks_and_associations(x[m], y[m]) for m in masks))
    corr = np.stack(associations)
    direction = np.sign(np.median(corr, axis=0))
    # Prefer features with a consistent direction even in weaker bearings.
    strength = np.quantile(corr * direction, .25, axis=0)
    redundancy = np.median(np.abs(duplicates), axis=0)
    selected = []
    for idx in np.argsort(-strength, kind='stable'):
        if high_band_only and json.loads(data['paths'][idx])[-1][1] != 'band3_energy_6k_12k':
            continue
        if strength[idx] > 0 and all(redundancy[idx, j] < .85 for j in selected):
            selected.append(idx)
        if len(selected) == (1 if high_band_only else 4):
            break
    prior = float(np.average([y[m].mean() for m in masks], weights=weights))
    names_short = ['.'.join(k[1] for k in json.loads(p)[1:]) for p in data['paths']]
    diagnostics = {name: dict(rho_per_bearing=dict(zip(names, corr[:, j].tolist())),
                               lower_quartile=float(strength[j]))
                   for j, name in enumerate(names_short)}
    units = []
    bias = prior
    for idx in selected:
        # Fit a monotone curve independently per training bearing, then average
        # the curves at shared feature values. No test-bearing statistics enter.
        ref = x[:, idx]
        knots = np.unique(np.quantile(ref, np.linspace(0, 1, 25)))
        if len(knots) < 2:
            continue
        curve = np.zeros_like(knots)
        for m, w in zip(masks, weights):
            sample = ref[m]
            cuts = np.unique(np.quantile(sample, np.arange(1,8)/8))
            bins = np.searchsorted(cuts, sample)
            occupied = np.unique(bins)
            centers = np.array([sample[bins == i].mean() for i in occupied])
            means = np.array([y[m][bins == i].mean() for i in occupied])
            mass = np.array([(bins == i).sum() for i in occupied])
            levels = -direction[idx] * decreasing_levels(-direction[idx]*means, mass)
            curve += w*(np.interp(knots, centers, levels) - y[m].mean())
        bias += (curve[0]+curve[-1])/(2*len(selected))
        for left, right, jump in zip(knots[:-1], knots[1:], np.diff(curve)):
            units.append((idx, (left+right)/2, max((right-left)/2,1e-10), jump/(2*len(selected))))
    model = Model('regression', data['paths'], [], x, y, hidden=max(1,len(units)), seed=42)
    model.mean.fill(0)
    model.scale = np.maximum(1., np.max(abs(x), axis=0))
    model.params['w1'].fill(0)
    model.params['b1'].fill(0)
    model.params['w2'].fill(0)
    for i,(idx, center, width, gain) in enumerate(units):
        model.params['w1'][idx,i] = model.scale[idx]/width
        model.params['b1'][i] = -center/width
        model.params['w2'][i,0] = gain/model.meta['target_scale']
    model.params['b2'][0] = (bias-model.meta['target_mean'])/model.meta['target_scale']
    model.validate()
    return model, prior, dict(selected=[names_short[i] for i in selected], associations=diagnostics)

def train_member(x, y, labels, horizon=60):
    scale=np.maximum(np.median(np.abs(x),axis=0),1e-8)
    tx=np.sign(x)*np.log1p(np.abs(x)/scale)
    q1,center,q3=np.quantile(tx,[.25,.5,.75],axis=0)
    spread=np.maximum(q3-q1,.05)
    design=np.column_stack([np.clip((tx-center)/spread,-6,6), np.ones(len(y))])
    _, idx, counts=np.unique(labels,return_inverse=True,return_counts=True)
    weight=1/np.sqrt(counts[idx])
    weight/=weight.sum()
    target=(y<horizon).astype(float)
    coef=np.zeros(design.shape[1])
    penalty=np.full(len(coef),.01)
    penalty[-1]=0.
    for _ in range(40):
        p=1/(1+np.exp(-np.clip(design@coef,-40,40)))
        gradient=design.T@(weight*(p-target))+penalty*coef
        hessian=(design.T*(weight*p*(1-p)))@design + np.diag(penalty)
        step=np.linalg.solve(hessian, gradient)
        coef-=step
        if np.linalg.norm(step)<1e-8:
            break
    return scale,center,spread,coef

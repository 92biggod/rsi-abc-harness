"""Compile a read-only one-snapshot forward path from a fitted RUL checkpoint."""
from math import tanh
import numpy as np
from .harmonics import adjustment

def network(model):
    weights=model.params['w1']
    intercept=model.params['b1']
    out=model.params['w2'][:,0]
    output_bias=model.params['b2'][0]
    center=model.mean
    scale=model.scale
    target_scale=model.meta['target_scale']
    target_mean=model.meta['target_mean']
    # The encoded monotone curve has one input per hidden unit (or a zero unit).
    column=None
    if np.all(np.count_nonzero(weights,axis=0)<=1):
        column=np.argmax(abs(weights),axis=0)
        gain=weights[column,np.arange(len(column))]
    def evaluate(x):
        normalized=(x-center)/scale
        np.clip(normalized,-10.,10.,out=normalized)
        h=normalized@weights if column is None else normalized[column]*gain
        h+=intercept
        np.tanh(h,out=h)
        return max(0.,float((h@out+output_bias)*target_scale+target_mean))
    return evaluate

def build(model):
    base_network=network(model.parent)
    anchor_network=network(model.anchor) if model.anchor is not None else None
    anchor_weight=model.parent.meta.get('rul_anchor_weight',0.)
    curve_network=network(model.curve) if model.curve is not None else None
    weight=model.parent.meta.get('rul_aux_weight',0.)
    prior=model.parent.meta.get('rul_prior',0.)
    cfg=model.parent.meta.get('rul_harmonic')
    stages=model.stages
    fast_terminal=stages is not None and stages.scale.shape[0]==1 and stages.horizons.tolist()==[60]
    if fast_terminal:
        scale,center,spread=stages.scale[0],stages.center[0],stages.spread[0]
        coefficients=stages.coef[0,:-1,0]
        bias=stages.coef[0,-1,0]
    def predict_one(values):
        x=np.asarray(values,dtype=np.float64)
        if x.ndim!=1 or len(x)!=len(model.parent.mean) or not np.isfinite(x).all():
            raise ValueError('Invalid B snapshot')
        base=base_network(x)
        if weight:
            aux=curve_network(x) if curve_network is not None else prior
            base=(1-weight)*base+weight*aux
        if fast_terminal:
            z=(np.sign(x)*np.log1p(abs(x)/scale)-center)/spread
            np.clip(z,-6.,6.,out=z)
            logit=float(z@coefficients+bias)
            if logit>0 and base>29.5:
                base-=tanh(min(logit,40.)*.5)*(base-29.5)
        elif stages is not None:
            base=float(stages.predict(x[None,:],np.array([base]))[0])
        if cfg:
            base*=adjustment(x[None,:],cfg)
        if anchor_network is not None:
            base=(1-anchor_weight)*base+anchor_weight*anchor_network(x)
        return base
    return predict_one

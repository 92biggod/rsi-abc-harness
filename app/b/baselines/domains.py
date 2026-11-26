"""Training-only vibration groups for projections with opaque sample identifiers."""
import json
import numpy as np


def vibration_groups(data,seed=42,restarts=3):
    x=data['x']
    if len(x)<90:
        return None
    columns=[j for j,p in enumerate(data['paths']) if json.loads(p)[-1][1] in
             ('shape_factor','spectral_centroid','spectral_kurtosis','dominant_freq')]
    if len(columns)<2:return None
    raw=x[:,columns]
    scale=np.maximum(np.median(abs(raw),axis=0),1e-6)
    tx=np.sign(raw)*np.log1p(abs(raw)/scale)
    lo,center,hi=np.quantile(tx,[.25,.5,.75],axis=0)
    features=np.clip((tx-center)/np.maximum(hi-lo,.02),-5,5)
    rng=np.random.default_rng(seed)
    sample=features[rng.choice(len(x),min(3000,len(x)),replace=False)]
    rng=np.random.RandomState(seed)
    tol=1e-4*np.var(sample,axis=0).mean()
    norm=(sample*sample).sum(axis=1)
    best=None
    best_loss=np.inf
    for _ in range(restarts):
        centers=[sample[int(rng.uniform()*len(sample))]]
        nearest=np.full(len(sample),np.inf)
        for _ in range(2):
            nearest=np.minimum(nearest,np.sum((sample-centers[-1])**2,axis=1))
            if nearest.sum()<=1e-12:
                return None
            choices=np.searchsorted(np.cumsum(nearest),rng.uniform(size=3)*nearest.sum()).clip(max=len(sample)-1)
            options=sample[choices]
            d=np.maximum(norm[:,None]+np.sum(options*options,axis=1)-2*sample@options.T,0)
            centers.append(options[np.argmin(np.minimum(nearest[:,None],d).sum(axis=0))])
        current=np.stack(centers)
        for _ in range(60):
            d=np.maximum(norm[:,None]+(current*current).sum(axis=1)-2*sample@current.T,0)
            labels=d.argmin(axis=1)
            if len(np.unique(labels))<3:break
            updated=np.stack([sample[labels==k].mean(axis=0) for k in range(3)])
            delta=np.sum((current-updated)**2)
            current=updated
            if delta<=tol:break
        d=np.maximum(norm[:,None]+(current*current).sum(axis=1)-2*sample@current.T,0)
        loss=d.min(axis=1).sum()
        if loss<best_loss:
            best_loss=loss
            best=current
    d=(features*features).sum(axis=1)[:,None]+(best*best).sum(axis=1)-2*features@best.T
    labels=d.argmin(axis=1)
    if np.bincount(labels,minlength=3).min()<4:return None
    return labels

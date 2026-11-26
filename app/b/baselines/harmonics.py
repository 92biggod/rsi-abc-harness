"""Adjust RUL when a snapshot has an unambiguous low-order rotational harmonic."""
import json
from bisect import bisect_right
import numpy as np

def votes(frequencies, rates):
    order = np.rint(frequencies[:,:,None]/rates)
    match = (order>=1) & (order<=6) & (np.abs(frequencies[:,:,None]-order*rates)<=.6+.35*order)
    match &= match.sum(axis=2,keepdims=True)==1
    result = match.sum(axis=1)
    result[result.max(axis=1)<result.sum(axis=1)] = 0
    return np.minimum(result,1)

def fit_harmonics(train, physical_groups):
    if physical_groups is None:
        return None
    columns=[j for j,p in enumerate(train['paths']) if json.loads(p)[-1][1]=='dominant_freq']
    if len(columns)!=2:
        return None
    freq=train['x'][:,columns]
    names,counts=np.unique(physical_groups,return_counts=True)
    observations=[]
    for name in names:
        f=freq[physical_groups==name].ravel()
        low=f[(f>=28)&(f<=45)]
        observations.append(float(np.median(low)) if len(low) else np.nan)
    clusters=[]
    for value in sorted(v for v in observations if np.isfinite(v)):
        if clusters and abs(value-np.median(clusters[-1]))<=1.25:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    rates=np.asarray([np.median(cluster) for cluster in clusters])
    if len(rates)<2:
        return None
    membership=np.zeros((len(names),len(rates)))
    for j,(name,obs) in enumerate(zip(names,observations)):
        if np.isfinite(obs):
            membership[j,np.argmin(abs(rates-obs))]=1.
        else:
            counts_by_rate=votes(freq[physical_groups==name],rates).sum(axis=0)
            if counts_by_rate.sum()>=6 and counts_by_rate.max()/counts_by_rate.sum()>=.75:
                membership[j,np.argmax(counts_by_rate)]=1.
    known=membership.sum(axis=1)>0
    weights=np.sqrt(counts[known])
    avg=np.asarray([train['y'][physical_groups==name].mean() for name in names])[known]
    prior=float(np.average(avg,weights=weights))
    shrink=.75*weights.mean()
    class_mean=((weights*avg)@membership[known]+shrink*prior)/(weights@membership[known]+shrink)
    factors=np.sqrt(np.clip(class_mean/max(prior,1.),.12,4.))
    # Frequency resolution follows from 25.6 kHz sampling and 32,768-point snapshots.
    grid = np.arange(16385)
    bandwidth = np.maximum(2, np.ceil(grid*.01)).astype(int)
    codes = np.where(known, membership.argmax(axis=1), -1)
    tables = []
    for channel in range(2):
        evidence = np.zeros((len(grid),len(rates)))
        supports = np.zeros_like(evidence)
        for name,code in zip(names,codes):
            if code < 0:
                continue
            values = freq[physical_groups==name,channel]
            ix = np.clip(np.rint(values/.78125),0,len(grid)-1).astype(int)
            cdf = np.r_[0,np.cumsum(np.bincount(ix,minlength=len(grid)))]
            hits = cdf[np.minimum(grid+bandwidth+1,len(grid))] - cdf[np.maximum(grid-bandwidth,0)]
            fraction = hits/len(ix)
            evidence[:,code] += np.minimum(fraction,.10)
            supports[:,code] += (hits>=2) & (fraction>=.003)
        per_class = np.bincount(codes[known],minlength=len(rates))
        likelihood = evidence/np.maximum(per_class,1)+.003
        likelihood /= likelihood.sum(axis=1,keepdims=True)
        winner = likelihood.argmax(axis=1)
        reliable = (likelihood[grid,winner]>=.85) & (supports[grid,winner]>=1)
        mapped = np.where(reliable,winner,-1)
        starts = np.r_[0,np.flatnonzero(np.diff(mapped))+1]
        tables.append([starts.tolist(),mapped[starts].tolist()])
    return dict(columns=columns,rates=rates.tolist(),factors=factors.tolist(),lookup=tables)

def adjustment(x, config):
    rates=config['rates']
    tables=config.get('lookup')
    if len(x)==1:
        direct=[]
        spectral=[]
        for ch,col in enumerate(config['columns']):
            frequency=float(x[0,col])
            hits=[]
            for i,rate in enumerate(rates):
                if not 0<frequency<=6.5*rate:
                    continue
                order=round(frequency/rate)
                if 1<=order<=6 and abs(frequency-order*rate)<=.6+.35*order:
                    hits.append(i)
            if len(hits)==1:
                direct.extend(hits)
            if tables:
                bounds,values=tables[ch]
                idx=round(min(max(frequency,0.),12800.)/.78125)
                winner=values[bisect_right(bounds,idx)-1]
                if winner>=0:
                    spectral.append(winner)
        chosen=direct if direct and min(direct)==max(direct) else spectral
        return config['factors'][chosen[0]] if chosen and min(chosen)==max(chosen) else 1.
    freq=x[:,config['columns']]
    matched=votes(freq,np.asarray(rates))
    direct=matched.sum(axis=1)>0
    best=np.where(direct,matched.argmax(axis=1),-1)
    if tables:
        inferred=[]
        for j,(bounds,values) in enumerate(tables):
            ix=np.rint(np.clip(freq[:,j],0.,12800.)/.78125).astype(int)
            inferred.append(np.asarray(values)[np.searchsorted(bounds,ix,side='right')-1])
        a,b=inferred
        agreed=(a<0)|(b<0)|(a==b)
        best=np.where(~direct & agreed,np.maximum(a,b),best)
    return np.where(best>=0,np.asarray(config['factors'])[best],1.)

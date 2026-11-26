"""Apply one or two horizon corrections, sharing the snapshot transformation."""
import numpy as np


class Stages:
    def __init__(self, horizons, scale, center, spread, coef):
        self.horizons = np.asarray(horizons)
        self.scale, self.center, self.spread, self.coef = scale, center, spread, coef
        if (self.horizons.tolist() not in ([60], [240, 60])
                or scale.ndim != 2 or scale.shape[0] not in (1, 7, 8, 9)
                or center.shape != scale.shape or spread.shape != scale.shape
                or coef.shape != (len(scale), scale.shape[1]+1, len(self.horizons))
                or np.any(scale <= 0) or np.any(spread <= 0)
                or not all(np.isfinite(a).all() for a in (scale, center, spread, coef))):
            raise ValueError('Invalid stage parameters')

    def predict(self, x, base):
        return self.correct(base, self.probabilities(x))

    def probabilities(self, x):
        # Shapes are [snapshots, members, features] and [members, features, horizons].
        values = x[:, None, :]
        features = np.sign(values)*np.log1p(np.abs(values)/self.scale)
        features = np.clip((features-self.center)/self.spread, -6., 6.)
        logits = np.einsum('smf,mfh->smh',features,self.coef[:,:-1,:])+self.coef[:,-1,:]
        if len(self.scale)==1:
            consensus = logits[:,0,:]
        else:
            # The nine-member deployed committee's lower quartile is an order statistic.
            if len(self.scale)==9:
                consensus = np.partition(logits,2,axis=1)[:,2,:]
            else:
                probs = 1/(1+np.exp(-np.clip(logits,-40,40)))
                return np.quantile(probs,.25,axis=1)
        return 1/(1+np.exp(-np.clip(consensus,-40,40)))

    def correct(self, base, scores):
        result = base.copy()
        for i,horizon in enumerate(self.horizons):
            weight = np.maximum(0.,2*scores[:,i]-1)
            result -= weight*np.maximum(result-(horizon-1)/2,0.)
        return result

    def save(self,path):
        np.savez(path,horizons=self.horizons,scale=self.scale,center=self.center,
                 spread=self.spread,coef=self.coef)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as arr:
            return cls(*(arr[k] for k in ('horizons','scale','center','spread','coef')))

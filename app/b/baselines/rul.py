"""Parent-preserving RUL predictor and bounded candidate training."""
import copy
import re
from pathlib import Path

from .model import Model
from .curve import fit, train_member
from .domains import vibration_groups
from .harmonics import adjustment, fit_harmonics
from .stages import Stages
import numpy as np


def grouped_ids(data):
    names=[]
    for value in data['ids']:
        match=re.fullmatch(r'xjtu_(Bearing\d+_\d+)_s\d+',value)
        if match is None:
            return None
        names.append(match.group(1))
    return np.array(names)


class Predictor:
    def __init__(self, parent, curve=None, stages=None, anchor=None):
        self.parent=parent
        self.curve=curve
        self.stages=stages
        self.anchor=anchor
        weight=parent.meta.get('rul_anchor_weight',0.)
        if not isinstance(weight,(int,float)) or not 0 <= weight <= 1 or bool(weight)!=(anchor is not None):
            raise ValueError('Invalid RUL anchor weight or missing checkpoint')
        if anchor is not None and (anchor.meta['task']!='regression' or anchor.meta['paths']!=parent.meta['paths']):
            raise ValueError('Incompatible RUL anchor')
        if parent.meta['task']!='regression':
            raise ValueError('B requires a regression checkpoint')
        if curve is not None and curve.meta['paths']!=parent.meta['paths']:
            raise ValueError('Incompatible auxiliary feature schema')
        if stages is not None and stages.scale.shape[1]!=len(parent.meta['paths']):
            raise ValueError('Incompatible terminal detector dimensions')

    @classmethod
    def load(cls, path, resolve=None):
        path=Path(path)
        parent=Model.load(path)
        if resolve is None:
            resolve=lambda name:path.parent/name
        curve=Model.load(resolve('curve.npz')) if parent.meta.get('rul_auxiliary')=='curve' else None
        stages=Stages.load(resolve('stages.npz')) if parent.meta.get('rul_stages',False) else None
        anchor=Model.load(resolve('anchor.npz')) if parent.meta.get('rul_anchor_weight') else None
        return cls(parent,curve,stages,anchor)

    def predict(self,x):
        p=self.parent.predict(x)
        alpha=self.parent.meta.get('rul_aux_weight',0.)
        if alpha:
            q=self.curve.predict(x) if self.curve is not None else self.parent.meta['rul_prior']
            p=(1-alpha)*p+alpha*q
        if self.stages is not None:
            p=self.stages.predict(x,p)
        harmonic = self.parent.meta.get('rul_harmonic')
        if harmonic:
            p = p * adjustment(x, harmonic)
        if self.anchor is not None:
            weight=self.parent.meta['rul_anchor_weight']
            p=(1-weight)*p+weight*self.anchor.predict(x)
        return p

    def validation_predictor(self, x):
        """Reuse fixed curve and terminal probabilities across Adam updates."""
        alpha = self.parent.meta.get('rul_aux_weight', 0.)
        auxiliary = None
        if alpha:
            auxiliary = self.curve.predict(x) if self.curve is not None else self.parent.meta['rul_prior']
        scores = self.stages.probabilities(x) if self.stages is not None else None

        harmonic = self.parent.meta.get('rul_harmonic')
        factor = adjustment(x, harmonic) if harmonic else None

        anchor_prediction=self.anchor.predict(x) if self.anchor is not None else None

        def current_prediction():
            base = self.parent.predict(x)
            if auxiliary is not None:
                base = (1-alpha)*base + alpha*auxiliary
            if scores is not None:
                base = self.stages.correct(base, scores)
            if factor is not None:
                base=base*factor
            if anchor_prediction is not None:
                weight=self.parent.meta['rul_anchor_weight']
                base=(1-weight)*base+weight*anchor_prediction
            return base

        return current_prediction

    def save(self, directory):
        files=['checkpoint.npz']
        self.parent.save(directory/files[0])
        if self.curve is not None:
            self.curve.save(directory/'curve.npz')
            files.append('curve.npz')
        if self.stages is not None:
            self.stages.save(directory/'stages.npz')
            files.append('stages.npz')
        if self.anchor is not None:
            self.anchor.save(directory/'anchor.npz')
            files.append('anchor.npz')
        return files


def metrics(pred,y):
    e=pred-y
    return dict(rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),bias=float(np.mean(e)))


def prepare(parent,tr,groups,weight):
    model=copy.deepcopy(parent.parent)
    model.meta.pop('rul_anchor_weight',None)
    y=tr['y']
    x=tr['x']
    curve=stages=None
    model.meta.update(rul_aux_weight=weight,rul_auxiliary='none',rul_stages=False,rul_harmonic=None,rul_curve_fit=None)
    if weight:
        prior=float((np.median(y)+y.mean())/2)
        if groups is not None:
            names,counts=np.unique(groups,return_counts=True)
            if len(names)>=3 and counts.min()>=4:
                curve,prior,_=fit(tr,groups,np.ones(len(y),bool))
        model.meta.update(rul_prior=prior,rul_auxiliary='curve' if curve is not None else 'prior')
        # Sparse, anonymous or single-class projections use only identifiable supervision.
        masks=[np.ones(len(y),bool)]
        if groups is not None and len(np.unique(groups))==9:
            masks=[groups!=g for g in np.unique(groups)]
        if all(np.any(y[m]<60) and np.any(y[m]>=60) for m in masks):
            members=[train_member(x[m],y[m],groups[m] if groups is not None else np.zeros(m.sum(),int))
                     for m in masks]
            scale,center,spread,coef=(np.stack(col) for col in zip(*members))
            stages=Stages([60],scale,center,spread,coef[:,:,None])
            model.meta['rul_stages']=True
    return Predictor(model,curve,stages)


def continue_parent(parent,tr,val,options):
    physical_groups=grouped_ids(tr)
    groups=physical_groups
    grouping='known_bearing' if groups is not None else 'none'
    if options['weight'] and (groups is None or options.get('grouping') == 'vibration'):
        inferred = vibration_groups(tr)
        if inferred is not None:
            groups = inferred
            grouping = 'vibration_clusters'
    # A curve depends on the training projection and grouping, independently of
    # Adam state, learning rate and mixture weight. Older checkpoints refit once.
    fingerprint = tr.get('sha256')
    signature = ['rul-curve-v3', fingerprint, grouping] if fingerprint else None
    reuse = bool(options['weight'] and signature and parent.parent.meta.get('rul_curve_fit') == signature)
    if reuse:
        updated = copy.deepcopy(parent.parent)
        updated.meta.pop('rul_anchor_weight',None)
        updated.meta['rul_aux_weight'] = options['weight']
        candidate = Predictor(updated, parent.curve, parent.stages)
    else:
        candidate = prepare(parent,tr,groups,options['weight'])
    candidate.parent.meta['rul_curve_fit'] = signature if options['weight'] else None
    model=candidate.parent
    harmonic = None
    if options.get('harmonic'):
        if reuse and parent.parent.meta.get('rul_harmonic'):
            harmonic = parent.parent.meta['rul_harmonic']
        else:
            harmonic = fit_harmonics(tr, physical_groups)
    model.meta['rul_harmonic'] = harmonic
    model.meta.update(learning_rate=options['lr'],batch_size=64)
    rng=np.random.default_rng(np.random.SeedSequence([model.meta['seed'],model.meta['steps']]))
    w=None
    if groups is not None and options['balance']:
        _,idx,counts=np.unique(groups,return_inverse=True,return_counts=True)
        w=1/np.sqrt(counts[idx])
        w/=w.sum()
    validation_prediction = candidate.validation_predictor(val['x'])
    fresh_curve = (not reuse and options['weight'] >= .95 and candidate.curve is not None
                   and (physical_groups is None or len(np.unique(physical_groups)) >= 7))
    stop_reason = None
    best_epoch=0
    best_score=float('inf')
    trace=[]
    best=None
    for epoch in range(1,options['epochs']+1):
        ix=rng.choice(len(w),len(w),p=w) if w is not None else slice(None)
        loss,=model.fit(tr['x'][ix],tr['y'][ix],1)
        score=metrics(validation_prediction(),val['y'])
        trace.append(dict(epoch=epoch,training_loss=loss,steps=model.meta['steps'],**score))
        if score['rmse']<best_score:
            best_score=score['rmse']
            best_epoch=epoch
            best=copy.deepcopy(model)
        if fresh_curve and epoch < options['epochs'] and epoch-best_epoch >= 2:
            parent_score = metrics(parent.predict(val['x']),val['y'])['rmse']
            if best_score <= .9 * parent_score:
                stop_reason = 'fresh_curve_gain_then_two_stale_epochs'
                break
        if epoch-best_epoch>=4:
            break
    result = dict(history=trace,selected_epoch=best_epoch,grouping=grouping,auxiliary_reused=reuse,
                     parent_steps=parent.parent.meta['steps'],selected_steps=best.meta['steps'],
                     trained_steps=model.meta['steps']-parent.parent.meta['steps'])
    if stop_reason:
        result['stop_reason'] = stop_reason
    return Predictor(best,candidate.curve,candidate.stages),result

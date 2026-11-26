"""Validate the previously deployed parent on the newly visible validation projection."""
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from .rul import Predictor,grouped_ids,metrics

def preceding_artifact(rnd, history, validation_hash, current_rmse):
    if len(rnd.artifacts)<2:
        return None
    for entry in reversed(history):
        cfg=entry.get('configuration') or {}
        if entry.get('selected_artifact_id')!=rnd.parent.artifact_id or not cfg.get('adopt'):
            continue
        previous_hash=(cfg.get('hashes') or {}).get('validation')
        accepted_score=cfg.get('host_candidate_rmse')
        if not previous_hash or previous_hash==validation_hash or type(accepted_score) not in (int,float):
            return None
        if not math.isfinite(accepted_score) or accepted_score<0 or current_rmse<=1.15*accepted_score:
            return None
        matches=[a for a in rnd.artifacts if isinstance(a,dict) and a.get('artifact_id')==cfg.get('parent') and a.get('artifact_sha256')==cfg.get('parent_digest')]
        if len(matches)!=1 or matches[0]['artifact_id']==rnd.parent.artifact_id:
            return None
        artifact=matches[0];spec=artifact.get('artifact') or {}
        files=spec.get('checkpoint_paths') or []
        allowed={'checkpoint.npz','stages.npz','curve.npz','anchor.npz'}
        if not isinstance(files,list) or not all(isinstance(f,str) for f in files) or spec.get('code_paths')!=[] or 'checkpoint.npz' not in files or not set(files)<=allowed:
            return None
        manifest=artifact.get('files') or []
        if not isinstance(manifest,list) or not all(isinstance(r,dict) for r in manifest) or len(manifest)!=len(files) or {r.get('path') for r in manifest}!=set(files):
            return None
        return artifact
    return None

def evaluate_ancestor(rnd,history,validation,parent_predictions,current_rmse):
    artifact=preceding_artifact(rnd,history,validation['sha256'],current_rmse)
    if artifact is None:return None
    from contest_harness.errors import HarnessError
    record=dict(artifact_id=artifact['artifact_id'],artifact_sha256=artifact['artifact_sha256'],
                parent_rmse=current_rmse,eligible=False)
    try:
        with tempfile.TemporaryDirectory(dir=os.environ.get('CONTEST_WORK','/workspace')) as temp:
            restored=rnd.restore_artifact(artifact['artifact_id'],Path(temp)/'previous-deployment')
            archived=Predictor.load(restored/'checkpoint.npz')
            if archived.parent.meta['paths']!=validation['paths']:return {**record,'error':'feature_schema'}
            prediction=archived.predict(validation['x'])
    except HarnessError as exc:
        if exc.code not in {'ARTIFACT_UNKNOWN','ARTIFACT_INVALID','ARTIFACT_LIMIT','ARTIFACT_READ_MISMATCH','ARTIFACT_RECEIPT_MISMATCH','ARTIFACT_DIGEST_MISMATCH'}:
            raise
        return {**record,'error':exc.code}
    except (ValueError,KeyError,OSError) as exc:
        return {**record,'error':type(exc).__name__}
    y=validation['y']
    score=metrics(prediction,y)['rmse']
    if not math.isfinite(score):return {**record,'error':'nonfinite'}
    record['restored_rmse']=score
    record['bytes']=sum(f['size_bytes'] for f in artifact['files'])
    acceptable=score<=.95*current_rmse and score<current_rmse
    groups=grouped_ids(validation)
    if groups is not None and len(np.unique(groups))>1:
        gains=[]
        for g in np.unique(groups):
            use=groups==g
            old=metrics(parent_predictions[use],y[use])['rmse']
            new=metrics(prediction[use],y[use])['rmse']
            gains.append((old-new)/max(old,1e-9))
        record['worst_bearing_gain']=float(min(gains))
        acceptable=acceptable and min(gains)>=-.05 and np.median(gains)>=.02
    record['eligible']=bool(acceptable)
    return record

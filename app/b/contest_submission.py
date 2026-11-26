"""B bearing RUL submission for the contest.harness.workspace.v2 protocol."""
import json
import math
import os
from pathlib import Path
import tempfile
import time
from functools import lru_cache

from baselines.data import read_dataset
from baselines.input_vector import vectorizer
from baselines.rul import Predictor, continue_parent, grouped_ids, metrics
from baselines.anchor import sparse_anchor
from baselines.rollback import evaluate_ancestor
from baselines.compact_decision import decision_menu,decode
from baselines.monitor import feature_shift
from baselines.single import build as snapshot_predictor
import numpy as np

TASKS=['regression']
PLANS={
    'curve80':dict(lr=.0005,epochs=4,weight=.8,balance=False),
    'curve95':dict(lr=.0005,epochs=4,weight=.95,balance=True),
    'curve50':dict(lr=.00025,epochs=4,weight=.5,balance=True),
    'vibration95':dict(lr=.0005,epochs=4,weight=.95,balance=True,grouping='vibration',harmonic=True),
    'plain':dict(lr=.0005,epochs=4,weight=0.,balance=False),
}


def rmse(summary):
    if isinstance(summary,dict):
        summary=summary.get('metrics',summary)
        if isinstance(summary,dict):
            value=summary.get('rmse')
            if type(value) in (int,float) and math.isfinite(value) and value>=0:
                return float(value)
    return None


def effective_options(name, history, parent_id, hashes):
    options = dict(PLANS[name])
    for record in reversed(history):
        cfg = record.get('configuration') or {}
        if record.get('selected_artifact_id') != parent_id or not cfg.get('adopt'):
            continue
        if cfg.get('plan') == name and cfg.get('hashes') == hashes:
            old, new = rmse({'rmse':cfg.get('parent_rmse')}), rmse({'rmse':cfg.get('host_candidate_rmse')})
            if old and new is not None and 0 <= (old-new)/old < .002:
                options['epochs'] = 1
        break
    return options


def resource_record(receipt, train_count, validation_count):
    usage = receipt.response.get('usage') if isinstance(receipt.response, dict) else None
    selected = {}
    if isinstance(usage,dict):
        for key in ('prompt_tokens','completion_tokens','total_tokens'):
            value = usage.get(key)
            if type(value) is int and value >= 0:
                selected[key] = value
    return dict(training_rows=train_count,validation_rows=validation_count,model_calls=1,
                host_reported_tokens=selected)


def group_guard(val, groups, old_prediction, candidate):
    names = grouped_ids(val)
    if names is None or groups is None:
        return dict(active=False, passed=True)
    count = len(np.unique(groups))
    if not 2 <= count <= 3 or len(np.unique(names))<2:
        return dict(active=False, passed=True)
    new_prediction = candidate.predict(val['x'])
    gains = []
    by_bearing = {}
    for name in np.unique(names):
        mask = names == name
        before = metrics(old_prediction[mask], val['y'][mask])['rmse']
        after = metrics(new_prediction[mask], val['y'][mask])['rmse']
        gain = (before-after)/max(before,1e-9)
        gains.append(gain)
        by_bearing[str(name)] = dict(n=int(mask.sum()),before=before,after=after)
    passed = float(np.median(gains)) >= .02 and min(gains) >= -.05
    return dict(active=True, passed=passed, bearings=by_bearing)


def available_plans(history, parent_id, hashes, options):
    attempts = {p:0 for p in options}
    scored = set()
    for rec in history:
        cfg = rec.get('configuration') or {}
        name = cfg.get('plan')
        if name in attempts and cfg.get('parent') == parent_id and cfg.get('hashes') == hashes:
            attempts[name] += 1
            if rmse({'rmse':cfg.get('host_candidate_rmse')}) is not None:
                scored.add(name)
    choices = [p for p in options if not attempts[p]]
    if not choices:
        choices = [p for p in options if p not in scored and attempts[p]<2]
    return choices


def recent_small_gains(history, parent_id, hashes):
    gains = []
    for rec in reversed(history):
        cfg = rec.get('configuration') or {}
        if rec.get('selected_artifact_id')!=parent_id or cfg.get('hashes')!=hashes:
            break
        if cfg.get('plan')=='hold':
            continue
        old = rmse({'rmse':cfg.get('parent_rmse')})
        new = rmse({'rmse':cfg.get('host_candidate_rmse')})
        if not old or new is None:
            break
        gains.append(max(0.,(old-new)/old) if cfg.get('adopt') else 0.)
        parent_id=cfg.get('parent')
        if len(gains)==2:
            return max(gains)<.002
    return False


def train(rnd):
    if rnd.dataset_key!='B' or rnd.task['task']!='regression':
        raise ValueError('Expected B bearing RUL')
    parent=Predictor.load(rnd.parent.path('checkpoint.npz'),rnd.parent.path)
    paths=parent.parent.meta['paths']
    encode = vectorizer(paths)
    tr=read_dataset(rnd.data.train_path,'regression',expected_paths=paths,encode_inputs=encode)
    val=read_dataset(rnd.data.validation_path,'regression',expected_paths=paths,encode_inputs=encode)
    history=rnd.history()
    hashes={'train':tr['sha256'],'validation':val['sha256']}
    validation_parent = parent.predict(val['x'])
    evidence = dict(train=metrics(parent.predict(tr['x']),tr['y']),
                    validation=metrics(validation_parent,val['y']))
    train_ref=evidence['train']['rmse']
    evidence['rmse_ratio']=evidence['validation']['rmse']/max(train_ref,1e-9)
    groups = grouped_ids(tr)
    count = len(np.unique(groups)) if groups is not None else 0
    anchor = sparse_anchor(parent,tr['y'],groups,count)
    options = [p for p in PLANS if p != 'vibration95']
    if count >= 7:
        options.insert(0,'vibration95')
    elif groups is None and len(tr['y'])>=90:
        # The anonymous 95% curve uses vibration clusters. Prefer the shorter
        # equivalent configuration before the 80% option when both are fresh.
        options.remove('curve95')
        options.insert(0,'curve95')
    choices = available_plans(history,rnd.parent.artifact_id,hashes,options)
    if not choices or evidence['validation']['rmse']==0:
        choices=['hold']
    elif recent_small_gains(history,rnd.parent.artifact_id,hashes):
        choices.append('hold')
    host_parent_score=rmse(rnd.frame.get('current_validation'))
    ancestor=None
    if host_parent_score is not None and math.isclose(host_parent_score,evidence['validation']['rmse'],rel_tol=1e-6,abs_tol=1e-6):
        ancestor=evaluate_ancestor(rnd,history,val,validation_parent,evidence['validation']['rmse'])
    rollback_available=ancestor is not None and ancestor['eligible']
    if rollback_available:
        choices.append('rollback')
    evidence['feature_shift'] = feature_shift(tr, val)
    previous = []
    for rec in history[-3:]:
        cfg = rec.get('configuration') or {}
        if cfg.get('plan'):
            old,new=cfg.get('parent_rmse'),cfg.get('host_candidate_rmse')
            gain=f'{old-new:+.1f}' if type(old) in (int,float) and type(new) in (int,float) else ''
            previous.append(str(cfg['plan']) + ('已回滚' if cfg.get('rollback_selected') else '采用' if cfg.get('adopt') else '未采用') + gain)
    notes = '、'.join(previous) or '首轮'
    if count >= 7:
        bias = evidence['validation']['bias']
        direction = '高估' if bias > 0 else '低估' if bias < 0 else '偏差'
        metrics_text = f"训/验RMSE {train_ref:.0f}/{evidence['validation']['rmse']:.0f}，验证平均{direction}{abs(bias):.0f}分钟。"
        descriptions = []
        for plan in choices:
            if plan == 'rollback':
                descriptions.append(f"rollback 回到前次模型，同批验证RMSE {ancestor['restored_rmse']:.1f}，零步")
            elif plan == 'hold':
                descriptions.append('hold(零步)')
            else:
                opt = PLANS[plan]
                detail = '振动分组及主频校正' if plan == 'vibration95' else ''
                descriptions.append(f"{plan}({opt['weight']:.0%}{detail})")
        drift = evidence['feature_shift'][0]
        coverage = f'{count}根训练轴承'
        prompt = (f'B轴承续训，降低验证RMSE，兼顾步数。{coverage}。{metrics_text}'
                  + f"振动漂移 {drift['feature']} {drift['median_shift_in_train_widths']:+g}。历史：{notes}。"
                  + '候选（括号内为曲线混合权重）：' + '、'.join(descriptions)
                  + '。JSON仅含plan（单个英文候选名）、hypothesis（20字内说明本轮依据）。')
    else:
        metrics_text = (f"训练/验证RMSE {train_ref:.1f}/{evidence['validation']['rmse']:.1f}，"
                        f"验证偏差{evidence['validation']['bias']:+.1f}。")
        descriptions = []
        for plan in choices:
            if plan == 'rollback':
                descriptions.append(f"rollback 回到前次模型，同批验证RMSE {ancestor['restored_rmse']:.1f}，零步")
            elif plan == 'hold':
                descriptions.append('hold 保留父模型，本轮零训练步')
            else:
                opt = PLANS[plan]
                group_note = ' 振动分组及主频修正' if plan == 'vibration95' else ''
                if plan=='curve95' and groups is None:
                    group_note = ' 振动分组'
                descriptions.append(f"{plan} 曲线{opt['weight']:.0%} 最多{opt['epochs']}轮{group_note}")
        drift = evidence['feature_shift'][0]
        coverage = f'{count}根训练轴承' if count else '匿名训练集'
        if 0 < count < 3:
            coverage += '，曲线退回寿命先验'
        if anchor is not None:
            coverage += '，标签稀疏，各续训方案预测与初始网络各占一半'
        prompt = ('B轴承续训，降低验证RMSE，兼顾训练步数。' + coverage + '。' + metrics_text
                  + f"振动中位数偏移较大项 {drift['feature']} {drift['median_shift_in_train_widths']:+g} 个训练分位宽。"
                  + '历史：'+notes+'。可选：'+'；'.join(descriptions)+'。仅答JSON，plan选其一，hypothesis写一句短理由。')
    numbered=rnd.max_output_tokens<=48
    if numbered:
        prompt,reasons=decision_menu(choices,PLANS,evidence,count,notes,anchor is not None,ancestor)
    if rollback_available:
        prompt+=' 已复测旧模型。选择续训时，只有通过逐根检验并优于旧模型的新候选才会上线，否则恢复旧模型。'
    receipt=rnd.model([{'role':'user','content':prompt}],temperature=0,max_tokens=min(96,rnd.max_output_tokens))
    try:
        reply=json.loads(receipt.text)
    except (ValueError,TypeError):
        reply=None
    if numbered:
        reply=decode(reply,choices,reasons)
    valid=isinstance(reply,dict) and reply.get('plan') in choices and isinstance(reply.get('hypothesis'),str) and reply['hypothesis'].strip()
    name=reply['plan'] if valid else choices[0]
    hypothesis=reply['hypothesis'][:400] if valid else '按未尝试方案顺序检验父模型续训与振动曲线组合。'
    resources = resource_record(receipt, len(tr['y']), len(val['y']))
    if numbered:
        resources['numbered_decision']=dict(choices=choices,evidence=reasons,
            choice_index=reply['choice_index'] if valid else None,
            evidence_index=reply['evidence_index'] if valid else None)
    if ancestor is not None:
        resources['ancestor_validation']=ancestor
    if name=='rollback':
        target=ancestor['artifact_id']
        config=dict(plan=name,hashes=hashes,parent=rnd.parent.artifact_id,parent_digest=rnd.parent.checkpoint_sha256,
                    decision_call=receipt.call_id,used_fallback=False,adopt=False,rollback_selected=True,
                    restored_artifact=target,parent_rmse=ancestor['parent_rmse'],restored_rmse=ancestor['restored_rmse'],
                    trained_steps=0,resources=resources)
        rnd.finish(selected_artifact_id=target,configuration=config,hypothesis=hypothesis,
                   reason=f"当前验证集上父模型 RMSE {ancestor['parent_rmse']:.2f}，恢复的旧模型 {ancestor['restored_rmse']:.2f}。",
                   change_summary='按模型决定回到先前部署的检查点，保留其优化器状态。',seed=parent.parent.meta['seed'])
        return
    if name == 'hold':
        config = dict(plan=name,hashes=hashes,parent=rnd.parent.artifact_id,parent_metrics=evidence,
                      decision_call=receipt.call_id,used_fallback=not bool(valid),adopt=False,trained_steps=0,resources=resources)
        rnd.finish(selected_artifact_id=rnd.parent.artifact_id,configuration=config,hypothesis=hypothesis,
                   reason='根据模型决定与同投影实验历史保留父模型，本轮优化步数为零。',
                   change_summary='沿用已验证的父模型。',seed=parent.parent.meta['seed'])
        return
    options = effective_options(name, history, rnd.parent.artifact_id, hashes)
    started = time.perf_counter()
    candidate,trace=continue_parent(parent,tr,val,options)
    if anchor is not None:
        candidate.parent.meta['rul_anchor_weight']=.5
        candidate=Predictor(candidate.parent,candidate.curve,candidate.stages,anchor)
        trace['initial_network_weight']=.5
    guard = group_guard(val, groups, validation_parent, candidate)
    resources.update(candidate_fit_seconds=time.perf_counter()-started,optimizer_steps=trace['trained_steps'])
    config=dict(plan=name,hashes=hashes,parent=rnd.parent.artifact_id,parent_digest=rnd.parent.checkpoint_sha256,
                parent_metrics=evidence,decision_call=receipt.call_id,used_fallback=not bool(valid),
                options=options,nominal_epoch_limit=PLANS[name]['epochs'],resources=resources,validation_group_guard=guard,training=trace,auxiliary=candidate.parent.meta['rul_auxiliary'])
    shared=dict(hypothesis=hypothesis,seed=candidate.parent.meta['seed'],
                change_summary=f'父网络延续 Adam 续训，选择第 {trace["selected_epoch"]} 个 epoch；采用 {name} 方案。')
    with tempfile.TemporaryDirectory(dir=Path(os.environ.get('CONTEST_WORK','/workspace'))) as tmp:
        dest=Path(tmp)
        checkpoints=candidate.save(dest)
        result=rnd.submit_candidate(dest,checkpoint_paths=checkpoints,code_paths=[],
                     predict_argv=['python3','/app/harness.py','predict'],configuration=config,**shared)
        before=rmse(rnd.frame.get('current_validation'))
        if before is None:
            before=evidence['validation']['rmse']
        after=rmse(result.get('summary'))
        adopt=after is not None and after<before and guard['passed']
        selected=result['artifact_id'] if adopt else rnd.parent.artifact_id
        decision='采用候选' if adopt else '保留父模型'
        if rollback_available and after is not None and (not adopt or ancestor['restored_rmse']<=after):
            selected=ancestor['artifact_id']
            config.update(rollback_selected=True,restored_artifact=selected,restored_rmse=ancestor['restored_rmse'],
                          candidate_passed_parent=adopt)
            adopt=False
            decision=f"恢复旧模型（当前验证 RMSE {ancestor['restored_rmse']:.4f}）"
            shared['change_summary']='已按所选方案完成续训，比较同批验证结果后恢复此前部署的完整模型与优化器状态。'
        config.update(parent_rmse=before,host_candidate_rmse=after,adopt=adopt)
        if not guard['passed']:
            decision+='，新候选未通过逐根改善条件'
        rnd.finish(selected_artifact_id=selected,configuration=config,
                   reason=f'父模型 RMSE {before:.4f}，候选 {after}，{decision}。',**shared)


@lru_cache(maxsize=1)
def load_model(model_dir):
    model=Predictor.load(Path(model_dir)/'checkpoint.npz')
    return snapshot_predictor(model),vectorizer(model.parent.meta['paths'])


def predict(sample,model_dir):
    if sample.task!='regression' or sample.dataset_key not in (None,'B'):
        raise ValueError('Expected a B regression request')
    fitted,encode=load_model(model_dir)
    values=encode(sample.inputs)
    return {'value':fitted(values)}

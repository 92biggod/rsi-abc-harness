"""A model-selected plan and evidence item fit in one short JSON object."""
def decision_menu(choices, plans, evidence, group_count, history, anchored, ancestor=None):
    validation=evidence['validation']
    shift=evidence['feature_shift'][0]
    bias=validation['bias']
    direction='高估' if bias>=0 else '低估'
    reasons=[
        f"训练/验证RMSE {evidence['train']['rmse']:.1f}/{validation['rmse']:.1f}",
        f"验证平均{direction}{abs(bias):.1f}分钟",
        f"{shift['feature']}中位数偏移{shift['median_shift_in_train_widths']:+g}个训练分位宽",
        f"历史{history}",
    ]
    if ancestor is not None and ancestor['eligible']:
        reasons.append(f"旧模型当前验证RMSE {ancestor['restored_rmse']:.1f}")
    descriptors=[]
    for i,name in enumerate(choices):
        if name=='rollback':
            detail=f"回到旧模型，当前验证RMSE {ancestor['restored_rmse']:.1f}，零步"
        elif name=='hold':
            detail='保留父模型，零步'
        else:
            detail=f"曲线混合权重{plans[name]['weight']:.0%}"
            if name=='vibration95':
                detail+='，振动分组和主频修正'
            elif name=='curve95' and not group_count:
                detail+='，振动分组'
        descriptors.append(f'{i} {name}（{detail}）')
    scope=f'{group_count}根训练轴承' if group_count else '匿名投影'
    if 0<group_count<3:scope+='，曲线项退回寿命先验'
    if anchored:scope+='，更新与初始网络各占一半'
    prompt=(f'B轴承续训，降低验证RMSE并节省步数。{scope}。方案编号：'+'；'.join(descriptors)
        +'。依据编号：'+'；'.join(f'{i} {r}' for i,r in enumerate(reasons))
        +'。只回一个单行JSON对象，唯一的键是你选的方案编号（字符串），值是你选的依据编号（整数）。')
    return prompt,reasons


def decode(answer, choices, reasons):
    if not isinstance(answer,dict) or len(answer)!=1:
        return None
    key,reason=next(iter(answer.items()))
    if key not in [str(i) for i in range(len(choices))] or type(reason) is not int or not 0<=reason<len(reasons):
        return None
    name=choices[int(key)]
    return dict(plan=name,hypothesis=f'依据{reasons[reason]}，模型选择 {name}。',evidence_index=reason,choice_index=int(key))

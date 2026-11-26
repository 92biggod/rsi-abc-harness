"""Gas recognition on the official A checkpoint, using the workspace v2 SDK."""
import json
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from baselines.model import Model
from baselines.data import read_dataset
from baselines.ethanol_ethylene import LOGISTIC_GATE, LOGISTIC_L2
from baselines.predict import predict as baseline_predict
from baselines.train import train_run
import numpy as np

TASKS = ["classification"]
DEFAULT_SHAPE_BOUNDARY = -0.25
# Each candidate starts from the checkpoint selected by the host for this round.
PLANS = {
    "standard": {"learning_rate": 0.003, "batch_size": 128, "epochs": 5},
    "ethanol_ethylene_review": {"learning_rate": 0.003, "batch_size": 128, "epochs": 5,
                               "ethanol_ethylene_gate": 0.2, "ethanol_ethylene_method": "centroid"},
    "ethanol_ethylene_logistic": {"learning_rate": 0.003, "batch_size": 128, "epochs": 5,
                                 "ethanol_ethylene_gate": 0.2, "ethanol_ethylene_method": "logistic"},
    "shape_direct": {"learning_rate": 0.003, "batch_size": 128, "epochs": 5, "shape_boundary": 0.0},
    "shape_cautious": {"learning_rate": 0.003, "batch_size": 128, "epochs": 5, "shape_margin": 0.1},
    "shape_strict": {"learning_rate": 0.003, "batch_size": 128, "epochs": 5, "shape_margin": 0.25},
    "gentle": {"learning_rate": 0.001, "batch_size": 128, "epochs": 5},
    "feature_noise": {"learning_rate": 0.003, "batch_size": 128, "epochs": 5, "feature_noise_std": 0.1},
    "short": {"learning_rate": 0.003, "batch_size": 128, "epochs": 2},
    "step_decay": {"learning_rate": 0.003, "batch_size": 128, "epochs": 5, "decay_after": 3, "end_lr": 0.001},
    "balanced": {"learning_rate": 0.003, "batch_size": 128, "epochs": 5, "balance_power": 1.0},
    "gain_jitter": {"learning_rate": 0.003, "batch_size": 128, "epochs": 5, "log_gain_std": 0.6},
    "balanced_gain": {"learning_rate": 0.003, "batch_size": 128, "epochs": 10, "balance_power": 0.7, "log_gain_std": 0.6},
    "long": {"learning_rate": 0.001, "batch_size": 128, "epochs": 20},
    "small_batch": {"learning_rate": 0.003, "batch_size": 32, "epochs": 5},
    "small_batch_gentle": {"learning_rate": 0.001, "batch_size": 32, "epochs": 1},
}
PLANS = {name: {"shape_boundary": DEFAULT_SHAPE_BOUNDARY, "shape_margin": 0.0, **options}
         for name, options in PLANS.items()}


def diagnostics(model, data):
    labels = model.meta["classes"]
    size = len(labels)
    predictions = model.predict(data["x"]).argmax(axis=1)
    matrix = np.bincount(size * data["y"] + predictions, minlength=size**2).reshape(size, size)
    counts = matrix.sum(axis=1)
    recall = {label: float(matrix[i, i] / counts[i]) if counts[i] else None
              for i, label in enumerate(labels)}
    return {"labels": labels, "confusion_true_by_pred": matrix.tolist(), "recall": recall}


def choose_plan(rnd, history, evidence, data_hashes):
    recent = []
    for record in history[-10:]:
        config = record.get("configuration") or {}
        recent.append({"id": record["id"], "plan": config.get("plan"),
                       "parent_id": config.get("parent_id"), "stage": config.get("stage"),
                       "selection": config.get("selection"), "reason": str(record.get("reason", ""))[:300]})
    tried = set()
    for record in history:
        config = record.get("configuration") or {}
        plan = config.get("plan")
        if (plan in PLANS and config.get("parent_id") == rnd.parent.artifact_id
                and config.get("select_best_epoch") is True
                and config.get("data_hashes") == data_hashes
                and all(config.get(k) == v for k, v in PLANS[plan].items())):
            tried.add(plan)
    available = {k: v for k, v in PLANS.items() if k not in tried} or PLANS
    if not history:
        available = {name: options for name, options in available.items()
                     if options["shape_boundary"] == DEFAULT_SHAPE_BOUNDARY and options["shape_margin"] == 0.0
                     and "ethanol_ethylene_gate" not in options}
    prompt = (
        '针对 A 的 Macro-F1 选择下一次受控续训实验。先看父模型的类别召回和以往结果。'
        '只从本轮 plans 中选择。默认形状边界 −0.25，用 standard 建立对照。'
        '已有对照时，若乙醇与乙烯有双向误判，优先检验 ethanol_ethylene_review，复核门槛 0.2。'
        '若窄门槛仍有较多乙烯误判为乙醇，可测试 ethanol_ethylene_logistic，门槛 1.0，L2 强度 0.1。'
        '小批量续训若出现分数波动或乙醛误报反弹，可试 small_batch_gentle，用较低学习率续训 1 个 epoch。'
        '若需要测量边界偏移的作用，可用 shape_direct 对照零边界；若误报多，可对照 shape_cautious 或 shape_strict。理由不超过80字。'
        '只输出 JSON 对象，形如 {"plan":"standard","hypothesis":"依据和可验证的预期"}。\n'
        + json.dumps({"round": rnd.round_index, "plans": available, "recent": recent,
                      "parent_id": rnd.parent.artifact_id, "stage": rnd.stage_id,
                      "parent_diagnostics": evidence}, ensure_ascii=False)
    )
    receipt = rnd.model([{"role": "user", "content": prompt}],
                        max_tokens=min(384, rnd.max_output_tokens), temperature=0)
    try:
        answer = json.loads(receipt.text)
    except ValueError:
        answer = None
    if isinstance(answer, dict):
        name, hypothesis = answer.get("plan"), answer.get("hypothesis")
        if isinstance(name, str) and name in available and isinstance(hypothesis, str) and hypothesis.strip():
            return name, hypothesis.strip()[:500], receipt.call_id, False
    fallback = next(iter(available))
    return fallback, f"模型未返回可用方案，按候选顺序检验 {fallback}。", receipt.call_id, True


def macro_f1(summary):
    if isinstance(summary, dict):
        metrics = summary.get("metrics", summary)
        if isinstance(metrics, dict):
            value = metrics.get("macro_f1")
            if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1:
                return float(value)
    return None


def train(rnd):
    if rnd.dataset_key != "A" or rnd.task["task"] != "classification":
        raise ValueError("Expected the A gas classification contract")
    paths = rnd.parent.descriptor["checkpoint_paths"]
    if len(paths) != 1:
        raise ValueError("A requires one NumPy checkpoint")
    parent_path = rnd.parent.path(paths[0])
    parent = Model.load(parent_path)
    if parent.meta.get("profile") != "A":
        raise ValueError("Expected an A parent checkpoint")
    history = rnd.history()
    splits = {"train": rnd.data.train_path, "validation": rnd.data.validation_path}
    def read_split(split):
        return read_dataset(splits[split], "classification", parent.meta["classes"], parent.meta["paths"])

    validation = read_split("validation")
    training = read_split("train")
    data_hashes = {"train": training["sha256"], "validation": validation["sha256"]}
    evidence = {"train": diagnostics(parent, training),
                "validation": diagnostics(parent, validation)}
    plan, hypothesis, call_id, fallback = choose_plan(rnd, history, evidence, data_hashes)
    options = PLANS[plan]
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    with tempfile.TemporaryDirectory(dir=work, prefix="gas-") as run_dir:
        candidate = Path(run_dir) / "candidate"
        report = train_run("A", splits["train"], splits["validation"], candidate, resume=parent_path,
                           stage_id=f"A-round-{rnd.round_index}", select_best_epoch=True, **options)
        model = Model.load(candidate / "checkpoint.npz")
        config = {"plan": plan, "data_hashes": data_hashes, "parent_id": rnd.parent.artifact_id, "stage": rnd.stage_id, **options, "used_fallback": fallback, "model_call_id": call_id,
                  "parent_diagnostics": evidence,
                  "candidate_validation": diagnostics(model, validation),
                  "select_best_epoch": True, "selected_epoch_offset": report["final_epochs"] - report["initial_epochs"],
                  "shape_margin_requested": report["shape_margin_requested"], "shape_boundary_requested": report["shape_boundary_requested"],
                  "shape_review_active": report["shape_review_active"],
                  "ethanol_ethylene_gate_requested": report["ethanol_ethylene_gate_requested"],
                  "ethanol_ethylene_method_requested": report["ethanol_ethylene_method_requested"],
                  "ethanol_ethylene_review_active": report["ethanol_ethylene_review_active"],
                  "local_before": report["validation_before"], "local_after": report["validation_after"],
                  "parent_checkpoint_sha256": report["parent_checkpoint_sha256"]}
        change = (f"从父检查点续训 {options['epochs']} 个 epoch，学习率 {options['learning_rate']}，"
                  f"批大小 {options['batch_size']}。")
        change += f" 取验证 Macro-F1 最高的第 {config['selected_epoch_offset']} 个 epoch。"
        if report["shape_review_active"]:
            change += (f" 用当轮训练集拟合两类形状中心，乙醛判定边界为 {report['shape_boundary_requested']}，"
                       f"改判所需余量为 {report['shape_margin_requested']}。")
        elif report["shape_margin_requested"] is not None:
            change += " 当轮训练集无法区分两个形状中心，沿用六分类输出。"
        if report["ethanol_ethylene_review_active"]:
            method = report["ethanol_ethylene_method_requested"]
            gate = report["ethanol_ethylene_gate_requested"]
            if method == "logistic":
                config.update(ethanol_ethylene_logistic_gate=LOGISTIC_GATE, ethanol_ethylene_logistic_l2=LOGISTIC_L2)
                change += (f" 用当轮训练集的响应形状和相对幅度拟合乙醇与乙烯逻辑回归，L2 强度为 {LOGISTIC_L2}。"
                           f" 前两名为这两类且相对分差小于 {LOGISTIC_GATE} 时优先采用逻辑回归，"
                           f"其余情况按原有门槛 {gate} 执行类中心复核。")
            else:
                change += (" 用当轮训练集拟合乙醇与乙烯的响应形状和相对幅度，"
                           f"仅复核前两名为这两类且相对分差小于 {gate} 的样本。")
        elif report["ethanol_ethylene_gate_requested"]:
            change += " 当轮训练集缺少乙醇或乙烯，跳过这组复核。"
        if "feature_noise_std" in options:
            change += f" 训练批次加入标准差为 {options['feature_noise_std']} 的标准化特征高斯噪声。"
        if "decay_after" in options:
            change += f" 第 {options['decay_after']} 个 epoch 后将学习率降至 {options['end_lr']}。"
        if "balance_power" in options:
            change += f" 按训练类别频数的负 {options['balance_power']} 次方抽样，保持每轮样本数。"
        if "log_gain_std" in options:
            change += f" 训练样本施加对数标准差为 {options['log_gain_std']} 的随机增益。"
        declaration = dict(hypothesis=hypothesis, change_summary=change, seed=report["seed"])
        submitted = rnd.submit_candidate(candidate, checkpoint_paths=["checkpoint.npz"], code_paths=[],
                                         predict_argv=["python3", "/app/harness.py", "predict"],
                                         configuration=config, **declaration)
        before = macro_f1(rnd.frame.get("current_validation"))
        parent_source = "平台"
        if before is None:
            before = report["validation_before"]["macro_f1"]
            parent_source = "当前公开验证集复测"
        after = macro_f1(submitted.get("summary"))
        adopt = after is not None and after > before
        selected = submitted["artifact_id"] if adopt else rnd.parent.artifact_id
        selection = "采用候选" if adopt else "保留父模型"
        measured = "缺少有效测量" if after is None else f"{after:.6f}"
        reason = f"{parent_source}父模型 Macro-F1={before:.6f}，平台候选={measured}，{selection}。"
        config.update(selection=selection, parent_score=before, parent_score_source=parent_source,
                      measured_candidate_score=after)
        rnd.finish(selected_artifact_id=selected, configuration=config, reason=reason, **declaration)


@lru_cache(maxsize=1)
def load_selected(path):
    model = Model.load(path)
    if model.meta.get("profile") != "A":
        raise ValueError("Expected an A prediction checkpoint")
    return model


def predict(sample, model_dir):
    if sample.task != "classification" or sample.dataset_key not in (None, "A"):
        raise ValueError("Expected an A classification sample")
    path = os.environ.get("CONTEST_ARTIFACT_PATH") or Path(model_dir) / "checkpoint.npz"
    scores = baseline_predict(load_selected(path), sample.inputs)["outputs"]
    # Match NumPy argmax on exact ties while satisfying the SDK's unique-top contract.
    top = max(item["score"] for item in scores)
    winners = [item for item in scores if item["score"] == top]
    for item in winners[1:]:
        item["score"] = math.nextafter(top, 0.0)
    return scores

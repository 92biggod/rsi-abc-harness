"""contest_submission —— C 组（C-MAPSS 涡扇发动机 RUL 回归）提交模块。

P0 协议闭环版（2026-11-23）：
- train(rnd)：从父检查点续训（上尾加权 cap1.5，与本地实测最优 C2-cap1.5 同一算法，
  RMSE 52.87 -> 50.91），调一次平台模型做 keep/rollback 决策，提交候选并 finish。
- predict(sample, model_dir)：读 CONTEST_ARTIFACT_PATH 指向的检查点，逐行输出
  {"outputs": {"value": 非负循环数}}；算完一行立刻 flush（等 EOF 会超时判失败）。
- 自包含：不依赖 packages/ 目录；检查点读写复刻官方 baselines.Model 的
  save/load 格式（zip + npy + metadata JSON），向后兼容官方检查点。
- dataset_key != "C" 或任务不是 regression 时走保守直通（原样提交父检查点），
  保证协议在任何路由下都不挂。

日志一律走 stderr；stdout 只允许协议 JSONL。
"""
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

TASKS = ("regression",)
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]

BETAS = (.9, .999)
EPS = 1e-8
CLIP_NORM = 5.0
FEATURE_CLIP = 10.0
WEIGHT_CAP = 1.5          # C2 实测最优权重上限
EPOCHS_HOT, LR_HOT = 40, 2e-3
EPOCHS_COLD, LR_COLD = 15, 2e-4
TIME_BUDGET_RATIO = 0.35  # 训练占用 wall_seconds 的比例上限

# C5：按工况分组标准化（3 维 settings 聚类后各组 z-score）+ 输出上截断。
# 本地 validation 实测 50.91 -> 47.15（RMSE）、NASA 均值 152540 -> 48915。
# C6：多 seed 集成（关键发现：从官方基线检查点起步显著优于从上尾加权检查点起步，
# 且 fresh-regime 下不用上尾加权、纯 MSE 更好）。本地实测 47.15 -> 46.41（4 seed）。
REGIME_K = 6
REGIME_EPOCHS_HOT, REGIME_EPOCHS_COLD = 170, 30   # 换输入空间需要更多轮次适应
PREDICT_CAP = 250.0                                 # 截断上界（240~260 为平台期）

ENSEMBLE_SEEDS = 4            # 集成成员数（受时间预算约束，至少保证 1 个完整 seed）
ENSEMBLE_HOT, ENSEMBLE_COLD = 360, 72   # 每 seed 的热/冷轮次（与离线最优 400+80 近似）
ENSEMBLE_LR_HOT, ENSEMBLE_LR_COLD = 3e-3, 2e-4
ENSEMBLE_SEED_TIME_RATIO = 0.6          # 单个 seed 预算占"剩余预算"的比例上限（留余量）

# R5 双轨选模（2026-11-24）：提交轨三门槛，全部在部署口径（含 cap 250）上判定；
# NASA 未截断值只作诊断记录（离线实测 38 万~千万级，字面对 160k 线不可行）。
GATE_RMSE = 50.30
GATE_NASA = 160000.0
GATE_OVER = 0.57

_LOG = sys.stderr


def log(message):
    print(message, file=_LOG, flush=True)


# ------------------------------------------------------------ SwanLab 实验记录（离线模式）
# 《实验记录规范》是提交物之一：每轮训练一个 offline run，日志落在可写的
# $CONTEST_WORK/swanlab-logs，赛后用 `swanlab sync` 回放到 swanlab.cn。
# swanlab 缺失或初始化失败绝不影响评测轮（降级为无记录继续跑）。


def sw_init(name, config):
    try:
        import swanlab
        log_dir = str(Path(os.environ.get("CONTEST_WORK", "/workspace")) / "swanlab-logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        swanlab.init(mode="offline", project="zhixun-c", name=name, log_dir=log_dir,
                     config=config)
        return True
    except Exception as exc:
        log(f"[swanlab] init skipped: {exc!r}")
        return False


def sw_log(payload):
    try:
        import swanlab
        swanlab.log(payload)
    except Exception:
        pass


def sw_finish():
    try:
        import swanlab
        swanlab.finish()
    except Exception:
        pass


# ---------------------------------------------------------------- 特征解析
_FORBIDDEN = {"truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots"}


def flatten(inputs):
    """与官方 baselines.data.flatten 同语义：展开 inputs 为 (paths, values)。"""
    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or "".join(ch for ch in key.lower() if ch.isalnum()) in _FORBIDDEN:
                    raise ValueError("target-derived input field is not allowed")
                visit(value[key], path + [("key", key)], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("expected 1..4096 numeric features")
    return paths, values


def collect_rows(rows):
    """把 {id, inputs, answer} 行流变成 (ids, x, y, paths)。"""
    ids, features, targets = [], [], []
    paths = None
    for row in rows:
        current, values = flatten(row["inputs"])
        if paths is None:
            paths = current
        elif current != paths:
            raise ValueError("feature schema changed mid-stream")
        answer = row["answer"]
        if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer) or answer < 0:
            raise ValueError("RUL answer must be finite nonnegative")
        ids.append(str(row["id"]))
        features.append(values)
        targets.append(float(answer))
    if not ids:
        raise ValueError("no training rows")
    return ids, np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64), paths


# ------------------------------------------------- 检查点（官方格式复刻）
PARAMETERS = ("w1", "b1", "w2", "b2")
FORMAT = "contest.cpu-mlp.v1"


class MemberDigestMismatch(ValueError):
    """The optional ensemble member failed an integrity check."""


def load_checkpoint(path, *, resolve_member=None, recover_digest=False, member_issues=None):
    """复刻 baselines.Model.load：zip + npy + metadata JSON；容忍 member_* 集成成员数组。"""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("checkpoint must be a regular file no larger than 16 MiB")
    expected = {*PARAMETERS, *("adam_first_" + k for k in PARAMETERS),
                *("adam_second_" + k for k in PARAMETERS), "mean", "scale", "metadata"}
    with zipfile.ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist()]
        members = [n[:-4] for n in names if n.startswith("member_") and n.endswith(".npy")]
        base = {name + ".npy" for name in expected}
        if not base.issubset({*names}) or any(not n.startswith("member_") and n not in base
                                              for n in names):
            raise ValueError("invalid checkpoint archive")
    with np.load(path, allow_pickle=False) as arrays:
        metadata = arrays["metadata"]
        if metadata.ndim != 0 or metadata.dtype.kind != "U":
            raise ValueError("invalid checkpoint metadata")
        state = {"meta": json.loads(str(metadata))}
        for key in (*PARAMETERS, "mean", "scale"):
            state[key] = arrays[key].copy()
        for key in PARAMETERS:
            state["adam_first_" + key] = arrays["adam_first_" + key].copy()
            state["adam_second_" + key] = arrays["adam_second_" + key].copy()
        for key in members:
            state[key] = arrays[key].copy()
    manifest = state["meta"].pop("ensemble_files", None)
    if manifest is not None:
        if members or state["meta"].get("ensemble_members", 1) != 1 or not isinstance(manifest, list) or not 1 <= len(manifest) < 32:
            raise ValueError("invalid ensemble manifest")
        sidecars, issues = {}, []
        for i, entry in enumerate(manifest, 1):
            name = f"member-{i}.npz"
            if not isinstance(entry, dict) or entry.get("path") != name:
                raise ValueError("invalid ensemble member name")
            expected_hash = entry.get("sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
                raise ValueError("invalid ensemble member digest: " + name)
            try:
                member_path = Path(resolve_member(name) if resolve_member else path.parent / name)
                stat = member_path.lstat()
                if member_path.is_symlink() or not member_path.is_file() or stat.st_size > 16 * 1024 * 1024:
                    raise ValueError("invalid ensemble member file: " + name)
                payload = member_path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != expected_hash:
                    raise MemberDigestMismatch("ensemble member digest mismatch: " + name)
            except FileNotFoundError:
                issues.append(name + " missing")
                continue
            except MemberDigestMismatch:
                if not recover_digest:
                    raise
                issues.append(name + " digest mismatch")
                continue
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                items = archive.infolist()
                if sorted(item.filename for item in items) != sorted(k + ".npy" for k in PARAMETERS) or sum(item.file_size for item in items) > 16 * 1024 * 1024:
                    raise ValueError("invalid ensemble member archive: " + name)
            with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                for k in PARAMETERS:
                    sidecars[f"member_{i}_{k}"] = arrays[k]
        if issues:
            if member_issues is not None:
                member_issues.extend(issues)
            log("[checkpoint] using primary single model: " + ", ".join(issues))
        else:
            state.update(sidecars)
            state["meta"]["ensemble_members"] = len(manifest) + 1
    _validate_state(state)
    return state


def _validate_state(state):
    meta = state["meta"]
    if meta.get("format") != FORMAT or meta.get("task") != "regression":
        raise ValueError("unsupported checkpoint (need regression cpu-mlp)")
    hidden = meta["hidden"]
    size = len(meta["paths"])
    shapes = {"w1": (size, hidden), "b1": (hidden,), "w2": (hidden, 1), "b2": (1,)}
    for name, shape in shapes.items():
        if state[name].shape != shape or not np.all(np.isfinite(state[name])):
            raise ValueError("checkpoint tensor invalid: " + name)
    members = meta.get("ensemble_members", 1)
    for i in range(1, members):
        for name, shape in shapes.items():
            key = f"member_{i}_{name}"
            if key not in state or state[key].shape != shape or not np.all(np.isfinite(state[key])):
                raise ValueError("ensemble member invalid: " + key)
    if state["mean"].shape != (size,) or state["scale"].shape != (size,) or np.any(state["scale"] <= 0):
        raise ValueError("checkpoint normalization invalid")
    if meta["target_scale"] <= 0:
        raise ValueError("invalid target_scale")


def save_checkpoint(path, state):
    """复刻 baselines.Model.save：固定 zip 元数据，保证同一状态字节级一致。"""
    meta = state["meta"]
    member_keys = sorted(k for k in state if k.startswith("member_"))
    arrays = {**{k: state[k] for k in PARAMETERS},
              **{k: state[k] for k in member_keys},
              **{"adam_first_" + k: state["adam_first_" + k] for k in PARAMETERS},
              **{"adam_second_" + k: state["adam_second_" + k] for k in PARAMETERS},
              "mean": state["mean"], "scale": state["scale"],
              "metadata": np.array(json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
    for name, value in arrays.items():
        if name != "metadata" and not np.all(np.isfinite(value)):
            raise ValueError("nonfinite model state: " + name)
    with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = __import__("io").BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


# ------------------------------------------------------------ 前向与训练
def save_artifact(directory, state):
    """Write one official primary checkpoint and content-verified ensemble members."""
    count = state["meta"].get("ensemble_members", 1)
    _validate_state(state)
    primary = {k: v for k, v in state.items() if not k.startswith("member_")}
    primary["meta"] = dict(state["meta"])
    primary["meta"].pop("ensemble_members", None)
    manifest = []
    for i in range(1, count):
        name = f"member-{i}.npz"
        path = directory / name
        with path.open("xb") as handle:
            np.savez(handle, **{k: state[f"member_{i}_{k}"] for k in PARAMETERS})
        manifest.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if manifest:
        primary["meta"]["ensemble_files"] = manifest
    save_checkpoint(directory / "checkpoint.npz", primary)
    return ["checkpoint.npz", *(entry["path"] for entry in manifest)]


def normalized(state, x):
    return np.clip((x - state["mean"]) / state["scale"], -FEATURE_CLIP, FEATURE_CLIP)


def settings_indices(paths):
    decoded = [json.loads(path) for path in paths]
    return [decoded.index([["key", "settings"], ["index", i]]) for i in range(3)]


def _kmeans(points, k, iters=25, seed=7):
    """极简 Lloyd 迭代；用来把连续带噪的 settings 聚成 K 个工况。"""
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), k, replace=False)].copy()
    assign = np.zeros(len(points), dtype=np.int64)
    for _ in range(iters):
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = dist.argmin(1)
        for j in range(k):
            member = assign == j
            if member.any():
                centers[j] = points[member].mean(0)
    dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return centers, dist.argmin(1)


def build_regime(x, paths, k=REGIME_K):
    """从训练数据拟合工况分组统计量（只用到 settings 与输入特征，不碰标签）。"""
    indices = settings_indices(paths)
    settings = x[:, indices]
    s_mean = settings.mean(0)
    s_std = settings.std(0) + 1e-6
    centers, assign = _kmeans((settings - s_mean) / s_std, k)
    group_mean = np.zeros((k, x.shape[1]))
    group_std = np.ones((k, x.shape[1]))
    for j in range(k):
        member = assign == j
        if member.sum() > 10:
            group_mean[j] = x[member].mean(0)
            group_std[j] = x[member].std(0) + 1e-6
    return {"k": int(k), "cap": float(PREDICT_CAP), "grouping_indices": indices,
            "settings_mean": s_mean.tolist(), "settings_std": s_std.tolist(),
            "centers": centers.tolist(),
            "group_mean": group_mean.tolist(), "group_std": group_std.tolist()}


def regime_assign(regime, x):
    settings = x[:, regime.get("grouping_indices", [0, 1, 2])]
    scaled = (settings - np.asarray(regime["settings_mean"])) / np.asarray(regime["settings_std"])
    centers = np.asarray(regime["centers"])
    return ((scaled[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)


def regime_normalize(regime, x):
    groups = regime_assign(regime, x)
    mean = np.asarray(regime["group_mean"])[groups]
    std = np.asarray(regime["group_std"])[groups]
    return np.clip((x - mean) / std, -FEATURE_CLIP, FEATURE_CLIP)


def inputs_for(state, x):
    """按 checkpoint 里是否带 regime 信息，选择输入标准化方式。"""
    regime = state["meta"].get("regime")
    return regime_normalize(regime, x) if regime else normalized(state, x)


def forward(state, xn):
    hidden = np.tanh(xn @ state["w1"] + state["b1"])
    return hidden, hidden @ state["w2"] + state["b2"]


def predict_values(state, x):
    return predict_values_norm(state, inputs_for(state, x))


def _member_logits(state, xn, member):
    """member=0 用规范键，其余用 member_{i}_ 键。"""
    if member == 0:
        hidden = np.tanh(xn @ state["w1"] + state["b1"])
        return hidden @ state["w2"] + state["b2"]
    hidden = np.tanh(xn @ state[f"member_{member}_w1"] + state[f"member_{member}_b1"])
    return hidden @ state[f"member_{member}_w2"] + state[f"member_{member}_b2"]


def raw_values_norm(state, xn):
    """未截断的原始预测（诊断口径：NASA 未截断值、cap 触及率都用它）。"""
    ts, tm = state["meta"]["target_scale"], state["meta"]["target_mean"]
    n_members = int(state["meta"].get("ensemble_members", 1))
    if n_members <= 1:
        return np.maximum(0.0, _member_logits(state, xn, 0)[:, 0] * ts + tm)
    return np.mean([np.maximum(0.0, _member_logits(state, xn, m)[:, 0] * ts + tm)
                    for m in range(n_members)], axis=0)


def predict_values_norm(state, xn):
    """xn 已经是标准化后的特征；带 regime 的检查点额外做输出上截断；
    带 ensemble_members 的检查点对成员预测取平均。"""
    values = raw_values_norm(state, xn)
    cap = (state["meta"].get("regime") or {}).get("cap")
    return np.minimum(values, cap) if cap else values


def train_one_epoch(state, xn, target, weight):
    meta = state["meta"]
    rng = np.random.default_rng(np.random.SeedSequence([meta["seed"], meta["epochs"]]))
    order = rng.permutation(len(xn))
    w2 = state["w2"]
    loss_sum, wsum = 0.0, float(weight.sum())
    for start in range(0, len(order), meta["batch_size"]):
        idx = order[start:start + meta["batch_size"]]
        batch, expected, weight_b = xn[idx], target[idx], weight[idx]
        hidden = np.tanh(batch @ state["w1"] + state["b1"])
        logits = hidden @ w2 + state["b2"]
        error = logits[:, 0] - expected
        loss_sum += float(np.sum(weight_b * error ** 2))
        delta = (2 * weight_b * error / len(idx))[:, None]
        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
        hidden_delta = (delta @ w2.T) * (1 - hidden ** 2)
        gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        if not math.isfinite(norm):
            raise ValueError("nonfinite gradient")
        meta["steps"] += 1
        scale = min(1.0, CLIP_NORM / max(norm, 1e-12))
        for name, gradient in gradients.items():
            gradient *= scale
            state["adam_first_" + name] = BETAS[0] * state["adam_first_" + name] + .1 * gradient
            state["adam_second_" + name] = BETAS[1] * state["adam_second_" + name] + .001 * gradient ** 2
            first = state["adam_first_" + name] / (1 - BETAS[0] ** meta["steps"])
            second = state["adam_second_" + name] / (1 - BETAS[1] ** meta["steps"])
            state[name] -= meta["learning_rate"] * first / (np.sqrt(second) + EPS)
        if not all(np.all(np.isfinite(state[name])) for name in PARAMETERS):
            raise ValueError("nonfinite model state")
    meta["epochs"] += 1
    return loss_sum / max(wsum, 1.0)


def weights_for(y, cap=WEIGHT_CAP):
    w = np.ones_like(y, dtype=np.float64)
    high = y > 100
    w[high] = 1.0 + (y[high] - 100.0) / 100.0
    return np.minimum(w, cap)


def nasa_mean(err):
    d = np.asarray(err, dtype=np.float64)
    penalty = np.expm1(np.where(d < 0, -d / 13.0, d / 10.0))
    return float(penalty.mean())


def evaluate(state, x, y):
    return evaluate_norm(state, inputs_for(state, x), y)


def evaluate_norm(state, xn, y):
    raw = raw_values_norm(state, xn)
    cap = (state["meta"].get("regime") or {}).get("cap")
    pred = np.minimum(raw, cap) if cap else raw
    err = pred - y
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "over_ratio": float((err > 0).mean()), "nasa_mean": nasa_mean(err),
            "nasa_uncapped": nasa_mean(raw - y),
            "cap_touch_ratio": float((raw > cap).mean()) if cap else 0.0,
            "tail_bias": float(err[y > 200].mean()) if (y > 200).any() else 0.0}


def gates_pass(metric):
    """提交轨三门槛（部署口径：rmse/nasa 均基于 cap 250 后的预测）。"""
    return (metric["rmse"] <= GATE_RMSE and metric["nasa_mean"] <= GATE_NASA
            and metric["over_ratio"] <= GATE_OVER)


# ------------------------------------------------------------ 平台模型调用
META_PROMPT = """你是 C 组（涡扇发动机剩余寿命预测）的训练决策器。
只输出一行 JSON，不要解释，不要代码块。

输出格式：
{{"action":"keep|rollback|tune","use_calibration":true|false,"extra_epochs":0|5|10,"note":"不超过40字"}}

本轮信息：
- 数据集 C，任务 regression，主指标 RMSE（越低越好），另报 NASA 非对称分（高估罚更重）
- 父模型：RMSE {parent_rmse}
- 本轮候选：RMSE {cand_rmse}，高估比例 {cand_over}
- 上轮动作：{last_action}

判断规则（按优先级）：
1. 候选 RMSE 比父模型低 1.0 以上 -> keep
2. 候选 RMSE 比父模型高 -> rollback
3. 差距在 1.0 以内：若高估比例 > 0.60 -> tune 压高估；否则 keep

信息不足时输出：{{"action":"keep","use_calibration":false,"extra_epochs":0,"note":"信息不足，保守保留"}}
"""


def decide(rnd, parent_metric, candidate_metric, last_action):
    prompt = META_PROMPT.format(
        parent_rmse=f"{parent_metric['rmse']:.2f}", cand_rmse=f"{candidate_metric['rmse']:.2f}",
        cand_over=f"{candidate_metric['over_ratio']:.3f}", last_action=last_action or "无")
    try:
        receipt = rnd.model([{"role": "user", "content": prompt}],
                            max_tokens=min(128, rnd.max_output_tokens), temperature=0)
    except Exception as exc:  # 模型调用失败不能毁掉整轮
        log(f"[decide] model call failed: {exc!r}")
        return "keep", None
    text = receipt.text.strip()
    action, parsed = "keep", None
    try:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end + 1])
            if parsed.get("action") in ("keep", "rollback", "tune"):
                action = parsed["action"]
    except (ValueError, AttributeError) as exc:
        log(f"[decide] unparseable model reply {text!r}: {exc!r}")
    return action, receipt


# ------------------------------------------------------------ train 主流程
def _parent_root_override(rnd):
    """本地测试用：Windows 没有 /previous 挂载点，可用 CONTEST_PARENT_ROOT 指向替身目录。
    生产镜像不设该变量，走 SDK 默认 /previous。"""
    override = os.environ.get("CONTEST_PARENT_ROOT")
    if override:
        rnd.parent.root = Path(override)


def _train_ensemble(state, xn, target, xn_val, y_val, meta, started, budget):
    """C6 多 seed 集成 + R5 双轨选模：每个成员从父权重出发独立训练（纯 MSE），
    每 epoch 用提交路径同口径评分。成员级记录两条轨：
      诊断轨  最低 RMSE（不限门槛，供分析）；
      提交轨  仅记录过三门槛（RMSE/NASA/高估比）的最低 RMSE 权重。
    组装顺序：合格成员集成 -> 全体诊断成员集成 -> 最佳合格单成员；都不过门槛则返回 None（保留父版本）。"""
    base_seed = int(meta.get("seed", 42))
    members, qualifying, done = [], [], 0
    for i in range(ENSEMBLE_SEEDS):
        if i > 0:
            elapsed = time.monotonic() - started
            per_seed = elapsed / max(done, 1) * (ENSEMBLE_HOT + ENSEMBLE_COLD)
            if budget - elapsed < per_seed * ENSEMBLE_SEED_TIME_RATIO:
                log(f"[train] ensemble stops at {len(members)} members (time budget)")
                break
        member = {k: v.copy() for k, v in state.items()
                  if not k.endswith("meta") and not k.startswith("member_")}
        m_meta = dict(meta)
        m_meta["seed"] = base_seed + i
        member["meta"] = m_meta
        weight = np.ones_like(target)          # fresh-regime 用纯 MSE（离线实测优于上尾加权）
        best_metric, best_params, sub_metric, sub_params, stop = None, None, None, None, False
        for epochs, lr in ((ENSEMBLE_HOT, ENSEMBLE_LR_HOT), (ENSEMBLE_COLD, ENSEMBLE_LR_COLD)):
            m_meta["learning_rate"] = lr
            for _ in range(epochs):
                loss = train_one_epoch(member, xn, target, weight)
                done += 1
                elapsed = time.monotonic() - started
                if elapsed > 3.0 and elapsed / done * (ENSEMBLE_HOT + ENSEMBLE_COLD) > budget:
                    log(f"[train] time budget reached inside member {i} (epoch {done})")
                    stop = True
                    break
                metric = evaluate_norm(member, xn_val, y_val)
                sw_log({"member": i, "epoch": done, "train/loss": loss,
                        "val/rmse": metric["rmse"], "val/mae": metric["mae"],
                        "val/over_ratio": metric["over_ratio"],
                        "val/nasa": metric["nasa_mean"], "val/nasa_uncapped": metric["nasa_uncapped"]})
                metric["epoch"] = m_meta["epochs"]
                snapshot = {k: v.copy() for k, v in member.items()}
                if best_metric is None or metric["rmse"] < best_metric["rmse"]:
                    best_metric, best_params = metric, snapshot
                if gates_pass(metric) and (sub_metric is None or metric["rmse"] < sub_metric["rmse"]):
                    sub_metric, sub_params = metric, snapshot
            if stop:
                break
        if best_params is not None:
            members.append((best_params, best_metric))
            if sub_params is not None:
                sub_metric = dict(sub_metric)
                qualifying.append((sub_params, sub_metric))
        if stop:
            break
    if not members:
        return None, None, done, 0

    def assemble(pool):
        pool = sorted(pool, key=lambda item: item[1]["rmse"])
        best_params, best_single = pool[0]
        assembled = dict(best_params)
        assembled["meta"] = dict(best_params["meta"])
        assembled["meta"]["ensemble_members"] = len(pool)
        for idx in range(1, len(pool)):
            for name in PARAMETERS:
                assembled[f"member_{idx}_{name}"] = pool[idx][0][name]
        _validate_state(assembled)
        metric = evaluate_norm(assembled, xn_val, y_val)
        metric["epoch"] = best_single.get("epoch")
        return assembled, metric, len(pool)

    chosen, chosen_metric, members_n = None, None, 0
    attempts = []
    if qualifying:
        attempts.append(("qualifying-ensemble", qualifying))
    attempts.append(("all-diag-ensemble", members))
    for label, pool in attempts:
        assembled, metric, n = assemble(pool)
        log(f"[train] {label}: members={n} rmse={metric['rmse']:.4f} over={metric['over_ratio']:.4f} "
            f"nasa={metric['nasa_mean']:.0f} gates={'PASS' if gates_pass(metric) else 'FAIL'}")
        if gates_pass(metric):
            chosen, chosen_metric, members_n = assembled, metric, n
            break
    if chosen is None and qualifying:
        sub_params, sub_metric = min(qualifying, key=lambda item: item[1]["rmse"])
        chosen = dict(sub_params)
        chosen["meta"] = dict(sub_params["meta"])
        chosen["meta"].pop("ensemble_members", None)
        chosen_metric, members_n = dict(sub_metric), 1
        log(f"[train] fallback to best qualifying single member: rmse={sub_metric['rmse']:.4f}")
    if chosen is not None:
        log(f"[train] selected: rmse={chosen_metric['rmse']:.4f} over={chosen_metric['over_ratio']:.4f} "
            f"nasa={chosen_metric['nasa_mean']:.0f} members={members_n} "
            f"(diag best {min(m['rmse'] for _, m in members):.4f})")
    return chosen, chosen_metric, done, members_n


def train(rnd):
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    candidate_dir = work / f"candidate-{rnd.round_index}"
    state_dir = work / f"state-{rnd.round_index}"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    state_dir.mkdir(parents=True, exist_ok=False)
    task = (rnd.task or {}).get("task")
    log(f"[train] dataset={rnd.dataset_key} round={rnd.round_index} task={task}")
    _parent_root_override(rnd)

    if rnd.dataset_key != "C" or task != "regression":
        _train_passthrough(rnd, candidate_dir, state_dir)
        return

    # 1) 父检查点（SDK 会核对 sha256）
    declared = rnd.parent.declared_paths()
    parent_name = "checkpoint.npz" if "checkpoint.npz" in declared else next((name for name in declared if name.endswith(".npz")), declared[0])
    parent_path = rnd.parent.path(parent_name)
    parent_files = [parent_name]
    parent_issues = []
    def resolve_parent_member(name):
        from contest_harness.errors import HarnessError
        member_name = (Path(parent_name).parent / name).as_posix()
        if member_name not in declared:
            raise FileNotFoundError(member_name)
        (parent_path.parent / name).lstat()
        try:
            member_path = rnd.parent.path(member_name)
        except HarnessError as exc:
            if exc.code == "PARENT_FILE_NOT_DECLARED":
                raise FileNotFoundError(member_name) from exc
            if exc.code == "PARENT_DIGEST_MISMATCH":
                raise MemberDigestMismatch(member_name) from exc
            raise
        parent_files.append(member_name)
        return member_path
    state = load_checkpoint(parent_path, resolve_member=resolve_parent_member,
                            recover_digest=True, member_issues=parent_issues)
    # Continuation mutates the live weights and Adam state. Keep the verified
    # single parent intact so a rollback can export a standalone checkpoint.
    fallback_parent = {k: v.copy() for k, v in state.items()} if parent_issues else None
    meta = state["meta"]
    meta_seed = int(meta.get("seed", 42))
    last_action = None
    for record in rnd.history():
        last_action = (record.get("configuration") or {}).get("last_action", last_action)

    # 2) 公开数据
    _, x_train, y_train, paths = collect_rows(rnd.data.training_rows())
    _, x_val, y_val, _ = collect_rows(rnd.data.validation_rows())
    if paths != meta["paths"]:
        raise ValueError("feature schema mismatch with parent checkpoint")
    log(f"[train] rows train={len(y_train)} val={len(y_val)} features={len(paths)} "
        f"parent_epochs={meta['epochs']} steps={meta['steps']}")

    parent_metric = evaluate(state, x_val, y_val)
    log(f"[train] parent validation: {json.dumps(parent_metric)}")

    # 3) 工况分组标准化（C5）：父检查点已带 regime 就沿用，否则用训练数据现拟合
    regime = meta.get("regime")
    fresh_regime = regime is None
    if fresh_regime:
        regime = build_regime(x_train, paths, REGIME_K)
        log(f"[train] fitted regime groups k={regime['k']}")
    meta["regime"] = regime
    xn = regime_normalize(regime, x_train)
    xn_val = regime_normalize(regime, x_val)

    # 4) 续训：fresh-regime 走多 seed 集成（C6，纯 MSE）；否则走单模型短续训（上尾加权）
    budget = TIME_BUDGET_RATIO * rnd.wall_seconds if rnd.wall_seconds else 240.0
    sw_init(f"round-{rnd.round_index}", {
        "round": rnd.round_index, "dataset": "C", "task": "regression",
        "parent_rmse": parent_metric["rmse"], "fresh_regime": fresh_regime,
        "ensemble_seeds": ENSEMBLE_SEEDS if fresh_regime else 1,
        "regime_k": regime["k"], "predict_cap": PREDICT_CAP,
        "weight_cap": WEIGHT_CAP, "time_budget_s": round(budget, 1)})
    target = (y_train - meta["target_mean"]) / meta["target_scale"]
    started = time.monotonic()
    members_n = 1
    if fresh_regime:
        best_state, candidate_metric, done, members_n = _train_ensemble(
            state, xn, target, xn_val, y_val, meta, started, budget)
    else:
        weight = weights_for(y_train)
        schedule = [(EPOCHS_HOT, LR_HOT), (EPOCHS_COLD, LR_COLD)]
        planned = sum(epochs for epochs, _ in schedule)
        meta.pop("ensemble_members", None)
        best_metric, best_state, sub_metric, sub_state, done = parent_metric, None, None, None, 0
        for epochs, lr in schedule:
            meta["learning_rate"] = lr
            for _ in range(epochs):
                loss = train_one_epoch(state, xn, target, weight)
                done += 1
                elapsed = time.monotonic() - started
                if elapsed > 3.0 and elapsed / done * planned > budget:
                    log(f"[train] time budget reached at epoch {done}")
                    break
                metric = evaluate_norm(state, xn_val, y_val)
                sw_log({"member": 0, "epoch": done, "train/loss": loss,
                        "val/rmse": metric["rmse"], "val/mae": metric["mae"],
                        "val/over_ratio": metric["over_ratio"],
                        "val/nasa": metric["nasa_mean"], "val/nasa_uncapped": metric["nasa_uncapped"]})
                metric["epoch"] = meta["epochs"]
                snapshot = {key: value.copy() for key, value in state.items()
                            if not key.startswith("member_")}
                if metric["rmse"] < best_metric["rmse"]:
                    best_metric, best_state = metric, snapshot
                if gates_pass(metric) and (sub_metric is None or metric["rmse"] < sub_metric["rmse"]):
                    sub_metric, sub_state = metric, snapshot
            else:
                continue
            break
        # 提交轨优先：只提交过三门槛的候选；没有合格 epoch 则保留父版本
        if sub_state is not None:
            best_state = dict(sub_state)
        else:
            best_state = None
        candidate_metric = sub_metric if sub_metric is not None else parent_metric
        if sub_state is None:
            log("[train] no gate-passing epoch in continuation -> keep parent")
    trained_seconds = time.monotonic() - started
    trained = best_state is not None
    candidate_metric = candidate_metric if trained else parent_metric
    log(f"[train] epochs_done={done} seconds={trained_seconds:.1f} "
        f"candidate: {json.dumps({k: v for k, v in candidate_metric.items() if k != 'epoch'})}")

    # 4) 平台模型决策（建议用途；晋级由脚本按指标决定）
    if trained:
        saved = state_dir / "selected.npz"
        save_checkpoint(saved, best_state)
        try:
            measured = evaluate(load_checkpoint(saved), x_val, y_val)
            if any(not math.isclose(v, candidate_metric[k], rel_tol=1e-12, abs_tol=1e-10)
                   for k, v in measured.items()):
                raise ValueError("selected checkpoint validation metrics changed on reload")
            candidate_metric = dict(candidate_metric, **measured)
            log(f"[train] reloaded candidate validation: {json.dumps(candidate_metric)}")
        except Exception:
            saved.unlink(missing_ok=True)
            raise
    action, receipt = decide(rnd, parent_metric, candidate_metric, last_action)
    better = candidate_metric["rmse"] < parent_metric["rmse"]
    submit_trained = (trained and better
                      and candidate_metric is not parent_metric and gates_pass(candidate_metric))
    log(f"[train] model_action={action}(advisory) trained={trained} better={better} "
        f"gates={'PASS' if trained and gates_pass(candidate_metric) else 'FAIL/NA'} "
        f"submit_trained={submit_trained}")

    # 5) 组装候选
    checkpoint_paths = ["checkpoint.npz"]
    if submit_trained:
        state = best_state
        if members_n > 1:
            hypothesis = ("按工况分组标准化 + 多 seed 集成（各成员从父权重独立续训后预测取平均），"
                          f"验证 RMSE {parent_metric['rmse']:.2f} -> {candidate_metric['rmse']:.2f}。")
            change_summary = (f"regime-normalized {members_n}-seed ensemble, {done} epochs total "
                              f"(k={regime['k']}, cap {PREDICT_CAP}, plain MSE), "
                              f"best member epoch {candidate_metric.get('epoch')}.")
        else:
            hypothesis = ("按工况分组标准化后在同一网络续训（上尾加权热冷两相学习率），"
                          f"验证 RMSE {parent_metric['rmse']:.2f} -> {candidate_metric['rmse']:.2f}。")
            change_summary = (f"regime-normalized continuation {done} epochs (k={regime['k']}, "
                              f"cap {PREDICT_CAP}, weight cap {WEIGHT_CAP}, hot {LR_HOT}/cold {LR_COLD}), "
                              f"best epoch {candidate_metric.get('epoch')}.")
        configuration = {"parent": parent_metric, "candidate": candidate_metric,
                         "epochs_done": done, "training_seconds": round(trained_seconds, 2),
                         "model_action": action, "last_action": last_action,
                         "weight_cap": WEIGHT_CAP, "regime_k": regime["k"],
                         "fresh_regime": fresh_regime, "predict_cap": PREDICT_CAP,
                         "ensemble_members": members_n}
        reason = (f"候选验证 RMSE 优于父模型（{candidate_metric['rmse']:.2f} vs "
                  f"{parent_metric['rmse']:.2f}），模型决策 {action}。")
    else:
        if fallback_parent is not None:
            save_checkpoint(candidate_dir / "checkpoint.npz", fallback_parent)
        else:
            shutil.copyfile(parent_path, candidate_dir / "checkpoint.npz")
            for member_name in parent_files[1:]:
                name = Path(member_name).name
                shutil.copyfile(rnd.parent.path(member_name), candidate_dir / name)
                checkpoint_paths.append(name)
        hypothesis = "平台模型判定 rollback 或训练无收益，保守保留父检查点。"
        change_summary = "pass-through parent checkpoint (no effective improvement)."
        configuration = {"parent": parent_metric, "candidate": candidate_metric,
                         "model_action": action, "rolled_back": True}
        reason = f"训练候选未优于父模型或模型决策 rollback（action={action}），提交父检查点副本。"

    if submit_trained:
        if members_n > 1:
            checkpoint_paths = save_artifact(candidate_dir, best_state)
            measured = evaluate(load_checkpoint(candidate_dir / "checkpoint.npz"), x_val, y_val)
            for key in ("rmse", "mae", "over_ratio", "nasa_mean"):
                if not math.isclose(measured[key], candidate_metric[key], rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError("exported ensemble metrics changed: " + key)
            saved.unlink()
        else:
            saved.replace(candidate_dir / "checkpoint.npz")
    elif trained:
        saved.unlink()

    if parent_issues:
        configuration["parent_fallback"] = {"members_used": 1, "issues": parent_issues}

    # 6) 状态与收尾
    (state_dir / "round-summary.json").write_text(json.dumps(
        {"dataset_key": rnd.dataset_key, "round_index": rnd.round_index,
         "hypothesis": hypothesis, "configuration": configuration},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rnd.save_state(state_dir)

    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis=hypothesis, change_summary=change_summary, configuration=configuration,
        seed=meta_seed,
        checkpoint_paths=checkpoint_paths, code_paths=[], predict_argv=PREDICT_ARGV,
    )
    history_ids = [record["id"] for record in rnd.history()]
    call_ids = [receipt.call_id] if receipt is not None else list(rnd.call_ids)
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis=hypothesis, change_summary=change_summary, reason=reason,
        configuration=configuration, seed=meta_seed,
        history_ids=history_ids, model_call_ids=call_ids,
    )
    sw_log({"final/parent_rmse": parent_metric["rmse"], "final/candidate_rmse": candidate_metric["rmse"],
            "final/candidate_mae": candidate_metric.get("mae"),
            "final/ensemble_members": members_n, "final/model_action": action})
    sw_finish()
    log("[train] round finished")


def _train_passthrough(rnd, candidate_dir, state_dir):
    """非 C 数据集的保守直通：原样提交父检查点，保证协议不挂。"""
    _parent_root_override(rnd)
    declared = rnd.parent.declared_paths()
    parent_name = next((name for name in declared if name.endswith(".npz")), declared[0])
    parent_path = rnd.parent.path(parent_name)
    shutil.copyfile(parent_path, candidate_dir / parent_name)
    receipt = rnd.model([{"role": "user", "content": "数据集 %s 第 %d 轮：本镜像仅实现 C 组回归，"
                        "本轮按协议保守直通父检查点，请输出 keep。" % (rnd.dataset_key, rnd.round_index)}],
                        max_tokens=min(32, rnd.max_output_tokens), temperature=0)
    configuration = {"passthrough": True, "dataset_key": rnd.dataset_key}
    rnd.save_state(state_dir)
    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis="本镜像当前只实现 C 组回归；对该数据集保守直通父检查点。",
        change_summary="passthrough parent checkpoint (submission implements task C only).",
        configuration=configuration, seed=1,
        checkpoint_paths=[parent_name], code_paths=[], predict_argv=PREDICT_ARGV,
    )
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis="本镜像当前只实现 C 组回归；对该数据集保守直通父检查点。",
        change_summary="passthrough parent checkpoint (submission implements task C only).",
        reason="保守直通：候选与父检查点一致，不引入回归风险。",
        configuration=configuration, seed=1,
        model_call_ids=[receipt.call_id],
    )
    log("[train] passthrough round finished")


# ------------------------------------------------------------ predict
_PREDICT_CACHE = {"key": None, "state": None}


def _predict_state(model_dir=None):
    path = os.environ.get("CONTEST_ARTIFACT_PATH")
    if not path:
        root = Path(model_dir or "/model")
        primary = root / "checkpoint.npz"
        candidates = [root] if root.is_file() else ([primary] if primary.is_file() else sorted(root.rglob("*.npz")))
        if not candidates:
            raise RuntimeError("no checkpoint: set CONTEST_ARTIFACT_PATH or mount /model")
        path = str(candidates[0])
    stat = os.stat(path)
    key = (path, stat.st_mtime_ns, stat.st_size)
    if _PREDICT_CACHE["key"] != key:
        _PREDICT_CACHE["state"] = load_checkpoint(path, recover_digest=True)
        _PREDICT_CACHE["key"] = key
        log(f"[predict] loaded checkpoint {path} "
            f"(epochs={_PREDICT_CACHE['state']['meta']['epochs']})")
    return _PREDICT_CACHE["state"]


def predict(sample, model_dir):
    if sample.task != "regression":
        raise RuntimeError("this submission implements regression only (task C)")
    state = _predict_state(model_dir)
    paths, values = flatten(sample.inputs)
    if paths != state["meta"]["paths"]:
        raise RuntimeError("feature schema mismatch with checkpoint")
    value = predict_values(state, np.asarray([values], dtype=np.float64))[0]
    return {"outputs": {"value": float(max(0.0, value))}}

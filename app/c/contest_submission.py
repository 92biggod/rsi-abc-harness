"""C-MAPSS RUL continuation and capped inference for the joint A/B/C harness."""
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

# Each training / prediction process serves one task. Set C's BLAS limit before NumPy loads.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"
import numpy as np

TASKS = ("regression",)
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]

BETAS = (.9, .999)
EPS = 1e-8
CLIP_NORM = 5.0
FEATURE_CLIP = 10.0
TIME_BUDGET_RATIO = 0.35
REGIME_K = 6
PREDICT_CAP = 250.0
# Cap the seed-42 anchor at 200 continuation epochs. Validation still selects the best epoch.
ANCHOR_SCHEDULE = ((200, .003),)
# Two predeclared continuations per parent and projection: (seed offset, epochs, LR).
BRANCHES = {"short": (1, 16, .0002), "medium": (2, 24, .0005)}

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


def choose_plan(rnd, history, key, fresh, parent_metric):
    used = {cfg.get("plan") for row in history
            if (cfg := row.get("configuration") or {}).get("experiment_key") == key}
    available = [p for p in (["anchor"] if fresh else BRANCHES) if p not in used] or ["hold"]
    prompt = ("Choose a C-MAPSS continuation plan from the available list. "
              "anchor reproduces the single-member baseline. short and medium train a companion "
              "from the mounted primary using different shuffles and learning rates. "
              "hold keeps the parent. Reply with JSON containing plan.\n"
              + json.dumps({"available": available, "parent": parent_metric,
                            "previously_tried": sorted(p for p in used if p)}))
    receipt = rnd.model([{"role": "user", "content": prompt}],
                        max_tokens=min(128, rnd.max_output_tokens), temperature=0)
    text = receipt.text.strip()
    try:
        plan = json.loads(text[text.index("{"):text.rindex("}") + 1]).get("plan")
    except (ValueError, AttributeError):
        plan = None
    valid = isinstance(plan, str) and plan in available
    return (plan if valid else available[0]), receipt, not valid


def projection_hash(x, y):
    digest = hashlib.sha256()
    for array in (x, y):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array, dtype="<f8").view(np.uint8))
    return digest.hexdigest()


def primary_copy(state):
    copy = {k: v.copy() for k, v in state.items() if not k.startswith("member_")}
    copy["meta"].pop("ensemble_members", None)
    return copy


def primary_digest(state):
    # A newly accepted companion changes the artifact ID, but not the next fit's starting point.
    primary = primary_copy(state)
    digest = hashlib.sha256(json.dumps(primary.pop("meta"), sort_keys=True,
                                       separators=(",", ":")).encode())
    for name in sorted(primary):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(primary[name], dtype="<f8").view(np.uint8))
    return digest.hexdigest()


def paired_state(parent, member):
    # Keep the mounted primary, its Adam moments, and its normalization intact.
    pair = {k: v for k, v in parent.items() if not k.startswith("member_")}
    pair["meta"] = dict(parent["meta"], ensemble_members=2)
    for name in PARAMETERS:
        pair["member_1_" + name] = member[name]
    return pair


def fit_candidate(parent, plan, xn, target, xv, yv, regime, before, budget):
    member = primary_copy(parent)
    meta = member["meta"]
    meta["regime"] = regime
    fresh = plan == "anchor"
    schedule = ANCHOR_SCHEDULE
    if not fresh:
        offset, epochs, lr = BRANCHES[plan]
        meta["seed"] += offset
        schedule = ((epochs, lr),)
    weights = np.ones_like(target)
    start = time.monotonic()
    steps = 0
    best, best_metrics, observed = None, None, {}
    for epochs, lr in schedule:
        meta["learning_rate"] = lr
        for _ in range(epochs):
            if time.monotonic() - start >= budget:
                break
            train_one_epoch(member, xn, target, weights)
            steps += 1
            single = evaluate_norm(member, xv, yv)
            candidate = member if fresh else paired_state(parent, member)
            score = single if fresh else evaluate_norm(candidate, xv, yv)
            if not observed or score["rmse"] < observed["candidate"]["rmse"]:
                observed = {"single": single, "candidate": score, "epoch": meta["epochs"]}
            improves = score["rmse"] < (best_metrics or before)["rmse"]
            if improves and gates_pass(score) and score["nasa_mean"] <= before["nasa_mean"]:
                best = {k: v.copy() for k, v in candidate.items()}
                best_metrics = dict(score, epoch=meta["epochs"], single_rmse=single["rmse"])
            sw_log({"epoch": steps, "val/rmse": score["rmse"], "val/nasa": score["nasa_mean"]})
    return best, best_metrics, {"epochs_done": steps, "training_seconds": round(time.monotonic()-start, 2),
                                "best_observed": observed, "seed": meta["seed"]}


def host_metrics(summary):
    if not isinstance(summary, dict):
        return {}
    metrics = summary.get("metrics", summary)
    return metrics if isinstance(metrics, dict) else {}
def _parent_root_override(rnd):
    """本地测试用：Windows 没有 /previous 挂载点，可用 CONTEST_PARENT_ROOT 指向替身目录。
    生产镜像不设该变量，走 SDK 默认 /previous。"""
    override = os.environ.get("CONTEST_PARENT_ROOT")
    if override:
        rnd.parent.root = Path(override)





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
        return member_path
    state = load_checkpoint(parent_path, resolve_member=resolve_parent_member,
                            recover_digest=True, member_issues=parent_issues)
    history = rnd.history()
    _, xt, yt, paths = collect_rows(rnd.data.training_rows())
    _, xv, yv, val_paths = collect_rows(rnd.data.validation_rows())
    meta = state["meta"]
    if paths != meta["paths"] or val_paths != paths:
        raise ValueError("feature schema mismatch")
    before = evaluate(state, xv, yv)
    fresh = meta.get("regime") is None
    train_hash, val_hash = projection_hash(xt, yt), projection_hash(xv, yv)
    key = "|".join(("anchor-pair-v1", primary_digest(state), train_hash, val_hash))
    plan, receipt, fallback = choose_plan(rnd, history, key, fresh, before)
    config = {"experiment_key": key, "plan": plan, "stage": rnd.stage_id,
              "parent_id": rnd.parent.artifact_id, "parent_digest": rnd.parent.checkpoint_sha256,
              "train_hash": train_hash, "validation_hash": val_hash,
              "training_rows": len(yt), "validation_rows": len(yv), "parent": before,
              "fallback_plan": fallback, "parent_issues": parent_issues}
    sw_ok = sw_init(f"round-{rnd.round_index}", config)
    # 0.3.0 的教训：swanlab 缺失时 sw_init 静默返回 False，20 轮零记录却无人发现。
    # 现在把结果写进回执，评审和本人都能在 configuration 里直接看到。
    config["swanlab_enabled"] = sw_ok
    if sw_ok:
        sw_log({"parent/rmse": before["rmse"], "parent/nasa": before["nasa_mean"],
                "parent/over_ratio": before["over_ratio"]})
    seed = int(meta["seed"])
    config.update(epochs_done=0, seed=seed)
    selected = rnd.parent.artifact_id
    reason = "Retain the mounted parent: no untried plan for this projection."
    hypothesis = "Preserve the primary model and evaluate a separately shuffled continuation."
    summary = "Retained the mounted parent."
    best, chosen = None, None
    if plan != "hold":
        regime = build_regime(xt, paths, REGIME_K) if fresh else meta["regime"]
        target = (yt-meta["target_mean"])/meta["target_scale"]
        budget = TIME_BUDGET_RATIO * rnd.wall_seconds if rnd.wall_seconds else 240.0
        best, chosen, trace = fit_candidate(state, plan, regime_normalize(regime, xt), target,
                                            regime_normalize(regime, xv), yv, regime, before, budget)
        config.update(trace, candidate=chosen)
        seed = trace["seed"]
        reason = "No epoch improved parent RMSE while passing the safety and NASA checks."

    if best is not None:
        files = save_artifact(candidate_dir, best)
        deployed = evaluate(load_checkpoint(candidate_dir / "checkpoint.npz"), xv, yv)
        if any(not math.isclose(v, chosen[k], rel_tol=1e-12, abs_tol=1e-10) for k, v in deployed.items()):
            raise ValueError("exported checkpoint metrics differ from selected epoch")
        summary = ("Trained the single-member anchor from the mounted baseline." if fresh else
                   "Trained a companion from the mounted primary; scored and exported a 50/50 pair.")
        submitted = rnd.submit_candidate(candidate_dir, checkpoint_paths=files, code_paths=[],
                    predict_argv=PREDICT_ARGV, hypothesis=hypothesis, change_summary=summary,
                    seed=seed, configuration=config)
        measured = host_metrics(submitted.get("summary"))
        host_before = host_metrics(rnd.frame.get("current_validation")).get("rmse", before["rmse"])
        rmse = measured.get("rmse")
        nasa = measured.get("nasa_score")
        consistent = (isinstance(rmse, (int, float)) and math.isfinite(rmse)
                      and math.isclose(rmse, deployed["rmse"], rel_tol=1e-7, abs_tol=5.1e-5))
        nasa_ok = (isinstance(nasa, (int, float)) and math.isfinite(nasa)
                   and math.isclose(nasa/len(yv), deployed["nasa_mean"], rel_tol=1e-7, abs_tol=1e-5)
                   and nasa/len(yv) <= min(GATE_NASA, before["nasa_mean"] + 1e-7))
        if consistent and nasa_ok and rmse <= GATE_RMSE and rmse < host_before:
            selected = submitted["artifact_id"]
            reason = "The reloaded artifact and host agree; RMSE improves and NASA passes."
        else:
            reason = "Retained the parent after checking the host measurement against local prediction."
        config.update(host_candidate=measured, host_parent_rmse=host_before,
                      proposed_members=len(files), deployed=deployed)
    config["rolled_back"] = selected == rnd.parent.artifact_id
    config["selected_id"] = selected
    config["selected_members"] = (int(meta.get("ensemble_members", 1)) if config["rolled_back"]
                                  else config["proposed_members"])
    (state_dir / "round-summary.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    rnd.save_state(state_dir)
    rnd.finish(selected_artifact_id=selected, hypothesis=hypothesis, change_summary=summary,
               reason=reason, configuration=config, seed=seed,
               history_ids=[row["id"] for row in history], model_call_ids=[receipt.call_id])
    sw_finish()


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

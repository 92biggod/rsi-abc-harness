"""Train or continue a CPU baseline from public train/validation JSONL only."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import tempfile
import time

try:
    from . import ethanol_ethylene
    from .model import Model, checkpoint_digest, _THREADS
    from .response_shape import fit_reference
    from .data import read_dataset, sha256
except ImportError:
    import ethanol_ethylene
    from model import Model, checkpoint_digest, _THREADS
    from response_shape import fit_reference
    from data import read_dataset, sha256
import numpy as np

ROOT = Path(__file__).resolve().parent


def metrics(model, data):
    predicted = model.predict(data["x"])
    expected = data["y"]
    if model.meta["task"] == "classification":
        labels = predicted.argmax(axis=1)
        observed = sorted(set(expected.tolist()) | set(labels.tolist()))
        f1 = [2 * np.sum((labels == label) & (expected == label)) /
              (np.sum(labels == label) + np.sum(expected == label)) for label in observed]
        result = {"accuracy": float(np.mean(labels == expected)), "macro_f1": float(np.mean(f1))}
        if "SevereFault" in model.meta["classes"]:
            severe = model.meta["classes"].index("SevereFault")
            count = int(np.sum(expected == severe))
            result["severe_miss_rate"] = float(np.sum((expected == severe) & (labels != severe)) / count) if count else None
        return result
    error = predicted - expected
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def train_run(profile, train_path, validation_path, output, *, resume=None, epochs=None,
              learning_rate=None, batch_size=None, seed=None, stage_id="baseline", label_provenance=None, feature_noise_std=0.0,
              decay_after=None, end_lr=None, balance_power=0.0, log_gain_std=0.0, select_best_epoch=False,
              shape_margin=None, shape_boundary=None, ethanol_ethylene_gate=None, ethanol_ethylene_method=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new directory; previous rounds are never overwritten")
    config_path = Path(profile) if str(profile).endswith(".json") else ROOT / "configs" / (str(profile) + ".json")
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    if config.get("profile") not in ("A", "B", "C") or config.get("task") not in ("classification", "regression"):
        raise ValueError("unsupported baseline profile")
    if not isinstance(stage_id, str) or not stage_id.strip() or len(stage_id) > 128:
        raise ValueError("stage ID must be nonempty and bounded")
    provenance = None
    if config["profile"] == "B":
        if label_provenance is None:
            raise ValueError("B requires explicit label provenance; do not silently derive official health labels from RUL")
        raw = Path(label_provenance).read_bytes()
        declared = json.loads(raw)
        if not isinstance(declared, dict) or declared.get("status") not in ("draft", "approved") or not isinstance(declared.get("method"), str) or not declared["method"].strip():
            raise ValueError("label provenance requires status=draft|approved and a nonempty method")
        provenance = {"sha256": sha256(raw), "declared_status": declared["status"], "method": declared["method"]}
    classes = config.get("classes", [])
    train = read_dataset(train_path, config["task"], classes)
    validation = read_dataset(validation_path, config["task"], classes, train["paths"])
    if set(train["ids"]) & set(validation["ids"]):
        raise ValueError("training and validation sample IDs overlap")
    parent = checkpoint_digest(resume) if resume else None
    model = Model.load(resume) if resume else Model(config["task"], train["paths"], classes,
        train["x"], train["y"], hidden=config["hidden"], seed=config["seed"] if seed is None else seed,
        learning_rate=config["learning_rate"], batch_size=config["batch_size"])
    if (model.meta["task"] != config["task"] or model.meta["paths"] != train["paths"] or model.meta["classes"] != classes
            or (resume and model.meta.get("profile") != config["profile"])):
        raise ValueError("checkpoint task, profile, feature schema or classes do not match")
    if resume and seed is not None and seed != model.meta["seed"]:
        raise ValueError("resume must preserve the checkpoint's random seed")
    if resume and config["hidden"] != model.meta["hidden"]:
        raise ValueError("resume hidden width must match checkpoint; changing the profile does not migrate weights")
    if resume and config["seed"] != model.meta["seed"]:
        raise ValueError("resume profile must preserve the checkpoint's random seed")
    model.meta["profile"] = config["profile"]
    if learning_rate is not None:
        model.meta["learning_rate"] = learning_rate
    if batch_size is not None:
        model.meta["batch_size"] = batch_size
    model.validate()
    count = config["epochs"] if epochs is None else epochs
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("epochs must be in 1..10000")
    initial_steps, initial_epochs = model.meta["steps"], model.meta["epochs"]
    validation_before = metrics(model, validation)
    started = time.monotonic()
    records = []
    initial_lr = model.meta["learning_rate"]
    if (decay_after is None) != (end_lr is None):
        raise ValueError("decay_after and end_lr must be set together")
    if decay_after is not None and not (0 < decay_after < count and 0 < end_lr <= 1):
        raise ValueError("invalid step decay")
    x, y = train["x"], train["y"]
    previous_reference = model.meta.pop("shape_reference", None)
    if shape_margin is None and previous_reference is not None:
        shape_margin = previous_reference["margin"]
    if shape_boundary is None and previous_reference is not None:
        shape_boundary = previous_reference.get("boundary")
    if shape_margin is not None or shape_boundary is not None:
        if config["profile"] != "A":
            raise ValueError("shape review requires the A gas profile")
        if shape_margin is None:
            shape_margin = 0.0
        reference = fit_reference(x, y, classes, shape_margin, boundary=shape_boundary)
        if reference is not None:
            model.meta["shape_reference"] = reference
    previous_pair = model.meta.pop("ethanol_ethylene_reference", None)
    if ethanol_ethylene_gate is None and previous_pair is not None:
        ethanol_ethylene_gate = previous_pair["gate"]
    if ethanol_ethylene_gate is not None:
        if config["profile"] != "A":
            raise ValueError("ethanol/ethylene review requires the A gas profile")
        if ethanol_ethylene_method is None:
            ethanol_ethylene_method = "logistic" if previous_pair and "logistic" in previous_pair else "centroid"
        reference = ethanol_ethylene.fit_reference(x, y, classes, ethanol_ethylene_gate,
                                                   method=ethanol_ethylene_method)
        if reference is not None:
            model.meta["ethanol_ethylene_reference"] = reference
    probabilities = None
    if balance_power:
        if not 0 < balance_power <= 1 or config["task"] != "classification":
            raise ValueError("balanced sampling requires classification and power in (0, 1]")
        probabilities = np.bincount(y)[y].astype(float) ** -balance_power
        probabilities /= probabilities.sum()
    if not 0 <= log_gain_std <= 1:
        raise ValueError("log_gain_std must be in [0, 1]")
    best_model = None
    best_score = -float("inf")
    selected_record = None
    for epoch_offset in range(count):
        rng = np.random.default_rng([model.meta["seed"], model.meta["epochs"], 17])
        epoch_x, epoch_y = x, y
        if probabilities is not None:
            indices = rng.choice(len(y), size=len(y), replace=True, p=probabilities)
            epoch_x, epoch_y = x[indices], y[indices]
        if log_gain_std:
            # Same gain across all channels of a training observation.
            epoch_x = epoch_x * np.exp(rng.normal(0, log_gain_std, size=(len(y), 1)))
        if epoch_offset == decay_after:
            model.meta["learning_rate"] = end_lr
        loss = model.fit(epoch_x, epoch_y, 1, feature_noise_std=feature_noise_std)[0]
        records.append({"epoch": model.meta["epochs"], "step": model.meta["steps"], "loss": loss,
                        "learning_rate": model.meta["learning_rate"],
                        "validation": metrics(model, validation)})
        if select_best_epoch:
            values = records[-1]["validation"]
            score = values["macro_f1"] if model.meta["task"] == "classification" else -values["rmse"]
            if score > best_score:
                best_model = copy.deepcopy(model)
                best_score = score
                selected_record = records[-1]
    if best_model is not None:
        model = best_model
    else:
        selected_record = records[-1]
    duration = time.monotonic() - started
    if not records:
        raise ValueError("at least one training epoch is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir(mode=0o700)
        checkpoint = staging / "checkpoint.npz"
        model.save(checkpoint)
        report = {"format": "contest.cpu-baseline-report.v1", "profile": config["profile"], "task": config["task"],
                  "purpose": "development_reference", "official_scoring_ready": False, "device": "cpu", "cpu_threads_requested": _THREADS,
                  "numpy_version": np.__version__, "python_version": platform.python_version(), "machine": platform.machine(), "stage_id": stage_id, "seed": model.meta["seed"],
                  "parameter_count": model.meta["parameter_count"], "input_features": len(model.meta["paths"]),
                  "train_rows": len(train["ids"]), "validation_rows": len(validation["ids"]),
                  "train_sha256": train["sha256"], "validation_sha256": validation["sha256"],
                  "profile_sha256": sha256(config_raw), "parent_checkpoint_sha256": parent,
                  "checkpoint_sha256": checkpoint_digest(checkpoint), "initial_steps": initial_steps,
                  "initial_epochs": initial_epochs, "final_steps": model.meta["steps"], "final_epochs": model.meta["epochs"],
                  "training_seconds": duration, "learning_rate": model.meta["learning_rate"], "batch_size": model.meta["batch_size"],
                  "validation_before": validation_before, "validation_after": selected_record["validation"],
                  "shape_margin_requested": shape_margin, "shape_boundary_requested": shape_boundary,
                  "shape_review_active": "shape_reference" in model.meta,
                  "ethanol_ethylene_gate_requested": ethanol_ethylene_gate,
                  "ethanol_ethylene_method_requested": ethanol_ethylene_method,
                  "ethanol_ethylene_review_active": "ethanol_ethylene_reference" in model.meta,
                  "last_evaluated_epoch": records[-1]["epoch"], "select_best_epoch": select_best_epoch,
                  "configuration": {"hidden": model.meta["hidden"], "activation": "tanh", "optimizer": "Adam",
                      "adam_beta1": .9, "adam_beta2": .999, "adam_epsilon": 1e-8, "gradient_clip_norm": 5.0,
                      "feature_clip": [-10.0, 10.0], "normalization": "initial_training_mean_std",
                      "learning_rate": model.meta["learning_rate"], "batch_size": model.meta["batch_size"],
                      "initial_learning_rate": initial_lr, "decay_after": decay_after, "end_lr": end_lr,
                      "balance_power": balance_power, "log_gain_std": log_gain_std, "requested_epochs": count, "feature_noise_std": feature_noise_std, "classes": model.meta["classes"]},
                  "source_sha256": {name: sha256((ROOT / name).read_bytes()) for name in
                                    ("model.py", "data.py", "train.py", "predict.py", "response_shape.py", "ethanol_ethylene.py")},
                  "label_provenance": provenance, "created_at": datetime.now(timezone.utc).isoformat()}
        (staging / "training.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        with (staging / "history.jsonl").open("w") as stream:
            for record in records:
                stream.write(json.dumps({"stage_id": stage_id, "seed": model.meta["seed"], **record}, allow_nan=False) + "\n")
        os.replace(staging, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="A, B, C or explicit profile JSON")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", help="Previous round checkpoint; preserve weights, Adam, normalization and epoch")
    continuation.add_argument("--from-baseline", help="Official/reference starting checkpoint; identical continuation semantics")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage-id", default="baseline")
    parser.add_argument("--label-provenance", help="Required for B: JSON with status and method; declared approval is not verified here")
    args = parser.parse_args()
    if args.epochs is not None and not 1 <= args.epochs <= 10000:
        parser.error("epochs must be in 1..10000")
    report = train_run(args.profile, args.train, args.validation, args.output, resume=args.resume or args.from_baseline,
        epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, seed=args.seed,
        stage_id=args.stage_id, label_provenance=args.label_provenance)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

"""Review close ethanol/ethylene decisions with two sensor response feature blocks."""
import math

import numpy as np

if __package__:
    from .response_shape import shapes, swap_winners, validate_centroids
else:
    from response_shape import shapes, swap_winners, validate_centroids


PAIR = ("Ethanol", "Ethylene")
BLOCK_WIDTHS = {"shape": 96, "amplitude": 32}
LOGISTIC_GATE = 1.0
LOGISTIC_L2 = 0.1


def feature_blocks(x):
    shape = shapes(x)
    amplitude = x.reshape(-1, 16, 8)[:, :, :2]
    maximum = np.abs(amplitude).max(axis=1, keepdims=True)
    scaled = np.divide(amplitude, maximum, out=np.zeros_like(amplitude), where=maximum > 0)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    relative = np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm > 0)
    return {"shape": shape, "amplitude": relative.reshape(-1, 32)}


def fit_reference(x, y, labels, gate, *, method="centroid"):
    if not valid_gate(gate):
        raise ValueError("ethanol/ethylene gate must be between zero and one")
    if method not in ("centroid", "logistic"):
        raise ValueError("invalid ethanol/ethylene review method")
    indices = [labels.index(label) for label in PAIR]
    pair_rows = np.isin(y, indices)
    targets = y[pair_rows]
    if gate == 0 or any(not np.any(targets == index) for index in indices):
        return None
    blocks = {}
    standardized_blocks = {}
    for name, features in feature_blocks(x[pair_rows]).items():
        mean, scale = features.mean(axis=0), features.std(axis=0)
        scale[scale < 1e-8] = 1
        standardized = (features - mean) / scale
        standardized_blocks[name] = standardized
        centers = [standardized[targets == index].mean(axis=0).tolist() for index in indices]
        blocks[name] = {"mean": mean.tolist(), "scale": scale.tolist(), "centers": centers}
    reference = {"version": 1, "gate": gate, "blocks": blocks}
    if method == "logistic":
        design = logistic_design(standardized_blocks)
        weights = fit_logistic(design, targets == indices[1], LOGISTIC_L2)
        reference["logistic"] = {"gate": LOGISTIC_GATE, "l2": LOGISTIC_L2, "weights": weights.tolist()}
    return reference


def logistic_design(standardized_blocks):
    parts = [standardized_blocks[name] / math.sqrt(width) for name, width in BLOCK_WIDTHS.items()]
    return np.column_stack((*parts, np.ones(len(parts[0]))))


def fit_logistic(x, targets, strength):
    weights = np.zeros(x.shape[1])
    ridge = np.full(x.shape[1], strength)
    ridge[-1] = 0.0  # The intercept is free.

    def loss(params):
        logits = x @ params
        return (np.logaddexp(0, logits) - targets * logits).mean() + np.dot(ridge * params, params) / 2

    for _ in range(60):
        probability = np.exp(-np.logaddexp(0, -(x @ weights)))
        gradient = x.T @ (probability - targets) / len(targets) + ridge * weights
        if np.linalg.norm(gradient, ord=np.inf) < 1e-8:
            return weights
        curvature = probability * (1 - probability)
        hessian = (x.T * curvature) @ x / len(targets) + np.diag(ridge)
        step = np.linalg.solve(hessian, gradient)
        current_loss = loss(weights)
        fraction = 1.0
        while loss(weights - fraction * step) >= current_loss:
            fraction /= 2
            if fraction < 1e-10:
                raise RuntimeError("logistic line search lost precision")
        weights -= fraction * step
    raise RuntimeError("logistic regression did not converge")


def valid_gate(gate):
    return type(gate) in (float, int) and math.isfinite(gate) and 0 <= gate <= 1


def validate_reference(ref, meta):
    if not isinstance(ref, dict) or ref.get("version") != 1 or meta.get("profile") != "A":
        raise ValueError("invalid ethanol/ethylene reference")
    if len(meta["paths"]) != 128 or not all(label in meta["classes"] for label in PAIR):
        raise ValueError("invalid ethanol/ethylene reference schema")
    if not valid_gate(ref.get("gate")) or ref["gate"] == 0:
        raise ValueError("invalid ethanol/ethylene gate")
    blocks = ref.get("blocks")
    if not isinstance(blocks, dict) or set(blocks) != set(BLOCK_WIDTHS):
        raise ValueError("invalid ethanol/ethylene feature blocks")
    for name, width in BLOCK_WIDTHS.items():
        if not isinstance(blocks[name], dict):
            raise ValueError("invalid ethanol/ethylene feature block")
        validate_centroids(blocks[name], width)
    if "logistic" in ref:
        logistic = ref["logistic"]
        if (not isinstance(logistic, dict) or not valid_gate(logistic.get("gate"))
                or logistic["gate"] != LOGISTIC_GATE or logistic.get("l2") != LOGISTIC_L2):
            raise ValueError("invalid ethanol/ethylene logistic configuration")
        weights = np.asarray(logistic.get("weights"))
        if (weights.shape != (sum(BLOCK_WIDTHS.values()) + 1,) or weights.dtype.kind not in "fi"
                or not np.isfinite(weights).all()):
            raise ValueError("invalid ethanol/ethylene logistic weights")


def review(scores, x, labels, ref):
    ethanol, ethylene = (labels.index(label) for label in PAIR)
    top_two = np.argsort(-scores, axis=1, kind="stable")[:, :2]
    rows = np.flatnonzero(np.isin(top_two, [ethanol, ethylene]).all(axis=1))
    first, second = top_two[rows].T
    high, low = scores[rows, first], scores[rows, second]
    logistic = ref.get("logistic")
    max_gate = max(ref["gate"], logistic["gate"]) if logistic else ref["gate"]
    close = (high - low) < max_gate * (high + low)
    rows, first, second = rows[close], first[close], second[close]
    if not len(rows):
        return scores

    contrast = np.zeros(len(rows))
    standardized_blocks = {}
    for name, features in feature_blocks(x[rows]).items():
        block = ref["blocks"][name]
        sample = (features - block["mean"]) / block["scale"]
        standardized_blocks[name] = sample
        ethanol_center, ethylene_center = np.asarray(block["centers"])
        midpoint = (ethanol_center + ethylene_center) / 2
        # Per-dimension distance advantage gives each feature block equal weight.
        contrast += 2 * ((sample - midpoint) @ (ethanol_center - ethylene_center)) / features.shape[1]
    if logistic:
        high, low = high[close], low[close]
        centroid_close = (high - low) < ref["gate"] * (high + low)
        logistic_close = (high - low) < logistic["gate"] * (high + low)
        logits = logistic_design(standardized_blocks) @ logistic["weights"]
        # Positive centroid contrast favors ethanol, while positive logits favor ethylene.
        contrast = np.where(logistic_close & (logits != 0), -logits,
                            np.where(centroid_close, contrast, 0.0))
    flip = ((first == ethanol) & (contrast < 0)) | ((first == ethylene) & (contrast > 0))
    return swap_winners(scores, rows[flip], first[flip], second[flip])

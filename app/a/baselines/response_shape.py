"""Nearest-centroid review of acetaldehyde and ethanol for the A gas profile."""
import math
import numpy as np

PAIR = ("Acetaldehyde", "Ethanol")


def shapes(x):
    if x.ndim != 2 or x.shape[1] != 128 or not np.isfinite(x).all():
        raise ValueError("expected 128 finite A features")
    triples = x.reshape(-1, 16, 8)[:, :, 2:].reshape(-1, 16, 2, 3)
    # Normalize each transient independently, also for zero or very large readings.
    maximum = np.abs(triples).max(axis=-1, keepdims=True)
    scaled = np.divide(triples, maximum, out=np.zeros_like(triples), where=maximum > 0)
    lengths = np.linalg.norm(scaled, axis=-1, keepdims=True)
    return np.divide(scaled, lengths, out=np.zeros_like(scaled), where=lengths > 0).reshape(-1, 96)


def fit_reference(x, y, labels, margin, *, boundary=None):
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("shape margin must be finite and nonnegative")
    if boundary is not None and not math.isfinite(boundary):
        raise ValueError("shape boundary must be finite")
    indices = [labels.index(label) for label in PAIR]
    if any(not np.any(y == idx) for idx in indices):
        return None
    t = shapes(x)
    mean, scale = t.mean(axis=0), t.std(axis=0)
    scale[scale < 1e-8] = 1
    standardized = (t - mean) / scale
    centers = np.stack([standardized[y == idx].mean(axis=0) for idx in indices])
    if np.sum((centers[0] - centers[1]) ** 2) < 1e-12:
        return None
    reference = {"version": 1, "margin": margin, "mean": mean.tolist(), "scale": scale.tolist(),
                 "centers": centers.tolist()}
    if boundary is not None:
        reference["boundary"] = boundary
    return reference


def validate_reference(ref, meta):
    if not isinstance(ref, dict) or ref.get("version") != 1 or meta.get("profile") != "A":
        raise ValueError("invalid A shape reference")
    if len(meta["paths"]) != 128 or not all(label in meta["classes"] for label in PAIR):
        raise ValueError("invalid shape reference schema")
    margin = ref.get("margin")
    if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
        raise ValueError("invalid shape margin")
    boundary = ref.get("boundary", 0.0)
    if type(boundary) not in (int, float) or not math.isfinite(boundary):
        raise ValueError("invalid shape boundary")
    validate_centroids(ref, 96)


def validate_centroids(ref, width):
    for key, dims in (("mean", (width,)), ("scale", (width,)), ("centers", (2, width))):
        arr = np.asarray(ref.get(key))
        if arr.shape != dims or arr.dtype.kind not in "fi" or not np.isfinite(arr).all():
            raise ValueError("invalid shape reference " + key)
    if np.any(np.asarray(ref["scale"]) <= 0):
        raise ValueError("invalid shape scale")


def review(scores, x, labels, ref):
    a, e = (labels.index(label) for label in PAIR)
    original = scores.argmax(axis=1)
    rows = np.flatnonzero((original == a) | (original == e))
    if not len(rows):
        return scores
    ca, ce = np.asarray(ref["centers"])
    z = (shapes(x[rows]) - ref["mean"]) / ref["scale"]
    # Difference of squared distances divided by squared inter-centroid distance;
    # equals +1 at the acetaldehyde centroid and -1 at the ethanol centroid.
    direction = ca - ce
    gap = 2 * ((z - (ca + ce) / 2) * direction).sum(axis=1)
    boundary = ref.get("boundary", 0.0)
    margin = ref["margin"]
    squared_distance = np.dot(direction, direction)
    toward_acetaldehyde = (original[rows] == e) & (gap > (boundary + margin) * squared_distance)
    toward_ethanol = (original[rows] == a) & (gap < (boundary - margin) * squared_distance)
    flip = toward_acetaldehyde | toward_ethanol
    changed = rows[flip]
    winner = np.where(original[changed] == e, a, e)
    return swap_winners(scores, changed, original[changed], winner)


def swap_winners(scores, rows, previous, winner):
    scores[rows, previous], scores[rows, winner] = scores[rows, winner], scores[rows, previous]
    # The SDK requires a unique winner, including the rare case of tied raw scores.
    top = scores[rows, winner, None]
    tied = (scores[rows] == top) & (np.arange(scores.shape[1]) != winner[:, None])
    scores[rows] = np.where(tied, np.nextafter(top, 0), scores[rows])
    return scores

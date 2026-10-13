"""Image-level Boundary F1 for binary masks.

Metric space: 224x224 evaluation masks. A boundary pixel matches when it is
within a Euclidean radius of two pixels from the opposing boundary.
"""

from __future__ import annotations

import numpy as np

BOUNDARY_TOLERANCE_PIXELS = 2


def _as_binary_2d(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D binary mask, got shape {array.shape}")
    return array.astype(bool, copy=False)


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract an 8-neighbour inner contour, treating outside-image as background."""
    binary = _as_binary_2d(mask)
    height, width = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    eroded = np.ones((height, width), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + height, dx : dx + width]
    return binary & ~eroded


def _dilate_euclidean(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate by an integer-radius Euclidean disk, not a square max-pool window."""
    binary = np.asarray(mask, dtype=bool)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return binary.copy()

    height, width = binary.shape
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    result = np.zeros((height, width), dtype=bool)
    radius_squared = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius_squared:
                continue
            y0 = radius + dy
            x0 = radius + dx
            result |= padded[y0 : y0 + height, x0 : x0 + width]
    return result


def boundary_f1_score(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance: int = BOUNDARY_TOLERANCE_PIXELS,
) -> float:
    """Compute symmetric, per-image Boundary F1 for already-binarized masks."""
    pred = _as_binary_2d(prediction)
    gt = _as_binary_2d(target)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/target shapes differ: {pred.shape} vs {gt.shape}")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    pred_boundary = inner_boundary(pred)
    gt_boundary = inner_boundary(gt)
    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())

    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    gt_neighbourhood = _dilate_euclidean(gt_boundary, tolerance)
    pred_neighbourhood = _dilate_euclidean(pred_boundary, tolerance)
    precision = float((pred_boundary & gt_neighbourhood).sum()) / pred_count
    recall = float((gt_boundary & pred_neighbourhood).sum()) / gt_count
    denominator = precision + recall
    if denominator == 0.0:
        return 0.0
    score = 2.0 * precision * recall / denominator
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Boundary F1 is outside [0, 1]: {score}")
    return score

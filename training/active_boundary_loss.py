"""Device-safe adaptation of Active Boundary Loss (ABL) for binary masks.

The direction-supervision formulation follows the authors' AAAI 2022 paper
and reference implementation:
https://github.com/wangchi95/active-boundary-loss
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class ActiveBoundaryLoss(nn.Module):
    """Align predicted boundaries by predicting directions toward GT edges.

    Args mirror the commonly used defaults in the reference implementation.
    ``target`` is an integer tensor shaped ``[B, H, W]``. ``logits`` is a
    multiclass tensor shaped ``[B, C, H, W]`` with at least two channels.
    """

    def __init__(
        self,
        isdetach: bool = True,
        max_n_ratio: float = 0.01,
        ignore_label: int = 255,
        label_smoothing: float = 0.2,
        max_clip_dist: float = 20.0,
    ) -> None:
        super().__init__()
        if not 0.0 < max_n_ratio <= 1.0:
            raise ValueError("max_n_ratio must be in (0, 1].")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if max_clip_dist <= 0.0:
            raise ValueError("max_clip_dist must be positive.")

        self.isdetach = bool(isdetach)
        self.max_n_ratio = float(max_n_ratio)
        self.ignore_label = int(ignore_label)
        self.label_smoothing = float(label_smoothing)
        self.max_clip_dist = float(max_clip_dist)

    @staticmethod
    def _kl_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        """Return elementwise KL(softmax(b) || softmax(a))."""
        return F.softmax(logits_b, dim=1) * (
            F.log_softmax(logits_b, dim=1) - F.log_softmax(logits_a, dim=1)
        )

    def _predicted_boundary(self, logits: torch.Tensor) -> torch.Tensor:
        _, _, height, width = logits.shape
        if height < 2 or width < 2:
            return torch.zeros(
                (logits.shape[0], height, width),
                dtype=torch.bool,
                device=logits.device,
            )

        kl_vertical = self._kl_divergence(
            logits[:, :, 1:, :], logits[:, :, :-1, :]
        ).sum(dim=1, keepdim=True)
        kl_horizontal = self._kl_divergence(
            logits[:, :, :, 1:], logits[:, :, :, :-1]
        ).sum(dim=1, keepdim=True)
        kl_vertical = F.pad(kl_vertical, (0, 0, 0, 1))
        kl_horizontal = F.pad(kl_horizontal, (0, 1, 0, 0))
        kl_map = kl_vertical + kl_horizontal

        # The reference implementation raises the KL threshold until only a
        # small fraction of the image remains active. The mask is intentionally
        # discrete; gradients are supplied by the direction KL below.
        threshold = 1e-5
        max_points = height * width * self.max_n_ratio
        active = kl_map > threshold
        for _ in range(256):
            if int(active.sum().item()) <= max_points:
                break
            threshold *= 1.2
            active = kl_map > threshold
        else:
            active = torch.zeros_like(active)

        # Dilate active boundary pixels to provide a narrow region for routing.
        active = F.max_pool2d(active.float(), kernel_size=3, stride=1, padding=1) > 0
        return active[:, 0]

    def _gt_boundary(self, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_label
        vertical = target[:, 1:, :] != target[:, :-1, :]
        horizontal = target[:, :, 1:] != target[:, :, :-1]
        vertical = F.pad(vertical, (0, 0, 0, 1))
        horizontal = F.pad(horizontal, (0, 1, 0, 0))
        return vertical | horizontal | ~valid

    @staticmethod
    def _distance_map(boundary: np.ndarray) -> np.ndarray:
        """Build the signed map used by ABL, returned as positive GT distance."""
        boundary = np.asarray(boundary, dtype=bool)
        non_boundary = ~boundary
        if not boundary.any() or not non_boundary.any():
            return np.zeros(boundary.shape, dtype=np.float32)

        # Equivalent to the reference one-hot signed-distance construction for
        # a single boundary class, followed by its positive-distance clamp.
        signed = np.zeros(boundary.shape, dtype=np.float32)
        signed[boundary] = distance_transform_edt(boundary)[boundary]
        signed[non_boundary] = -(
            distance_transform_edt(non_boundary)[non_boundary] - 1.0
        )
        return np.maximum(-signed, 0.0).astype(np.float32, copy=False)

    def _gt_distance_maps(self, target: torch.Tensor) -> torch.Tensor:
        boundary = self._gt_boundary(target).detach().cpu().numpy()
        maps = np.stack([self._distance_map(item) for item in boundary], axis=0)
        return torch.as_tensor(maps, device=target.device, dtype=torch.float32)

    @staticmethod
    def _label_smoothed_direction_targets(
        reference: torch.Tensor,
        direction_target: torch.Tensor,
        smoothing: float,
    ) -> torch.Tensor:
        """Build the target distribution used by the reference ABL LSSCE.

        This mirrors the reference LabelSmoothSoftmaxCEV1 implementation:
        initialize every class with ``smoothing / num_directions``, then
        overwrite the selected class with ``1 - smoothing``. In particular,
        do not add ``smoothing / num_directions`` to the selected class.
        """
        num_directions = reference.shape[1]
        smoothed_target = torch.full_like(
            reference, smoothing / num_directions
        )
        smoothed_target.scatter_(
            1,
            direction_target.unsqueeze(1),
            1.0 - smoothing,
        )
        return smoothed_target

    def _direction_loss(
        self,
        logits: torch.Tensor,
        pred_boundary: torch.Tensor,
        gt_distance: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        batch_ids, rows, cols = torch.nonzero(pred_boundary, as_tuple=True)
        if batch_ids.numel() == 0:
            return None

        height, width = logits.shape[-2:]
        distance_pad = F.pad(
            gt_distance.unsqueeze(1), (1, 1, 1, 1), value=1e5
        )[:, 0]
        logits_nhwc = logits.permute(0, 2, 3, 1)
        logits_pad = F.pad(
            logits, (1, 1, 1, 1), mode="replicate"
        ).permute(0, 2, 3, 1)

        # Eight neighbours followed by the center, matching the ABL direction
        # map. The center class is excluded from the supervised direction.
        offsets = (
            (1, 0),
            (-1, 0),
            (0, -1),
            (0, 1),
            (-1, 1),
            (1, 1),
            (-1, -1),
            (1, -1),
            (0, 0),
        )
        distances = torch.stack(
            [
                distance_pad[batch_ids, rows + dr + 1, cols + dc + 1]
                for dr, dc in offsets
            ],
            dim=1,
        )
        direction_target = distances.argmin(dim=1)
        valid_direction = direction_target != 8
        if not bool(valid_direction.any()):
            return None

        batch_ids = batch_ids[valid_direction]
        rows = rows[valid_direction]
        cols = cols[valid_direction]
        direction_target = direction_target[valid_direction]
        center_logits = logits_nhwc[batch_ids, rows, cols]

        direction_scores = []
        for dr, dc in offsets[:-1]:
            neighbor_logits = logits_pad[
                batch_ids, rows + dr + 1, cols + dc + 1
            ]
            if self.isdetach:
                neighbor_logits = neighbor_logits.detach()
            log_center = F.log_softmax(center_logits, dim=-1)
            log_neighbor = F.log_softmax(neighbor_logits, dim=-1)
            direction_kl = (
                F.softmax(neighbor_logits, dim=-1) * (log_neighbor - log_center)
            ).sum(dim=-1)
            direction_scores.append(direction_kl)

        direction_scores = torch.stack(direction_scores, dim=1)
        log_probs = F.log_softmax(direction_scores.float(), dim=1)
        smoothed_target = self._label_smoothed_direction_targets(
            log_probs,
            direction_target,
            self.label_smoothing,
        )
        ce = -(smoothed_target * log_probs).sum(dim=1)

        distances_at_boundary = gt_distance[batch_ids, rows, cols]
        distance_weight = distances_at_boundary.clamp(
            min=0.0, max=self.max_clip_dist
        ) / self.max_clip_dist
        return (ce * distance_weight).mean()

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError("Expected logits [B,C,H,W] and target [B,H,W].")
        if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
            raise ValueError("Logits and target batch/spatial shapes must match.")
        if logits.shape[1] < 2:
            raise ValueError("Active Boundary Loss requires at least two classes.")

        logits = logits.float()
        target = target.to(device=logits.device, dtype=torch.long)
        gt_distance = self._gt_distance_maps(target)
        pred_boundary = self._predicted_boundary(logits)
        return self._direction_loss(logits, pred_boundary, gt_distance)


def binary_logits_to_two_class(foreground_logits: torch.Tensor) -> torch.Tensor:
    """Represent a binary foreground logit as equivalent two-class logits."""
    if foreground_logits.ndim != 4 or foreground_logits.shape[1] != 1:
        raise ValueError("Expected foreground logits with shape [B,1,H,W].")
    return torch.cat((torch.zeros_like(foreground_logits), foreground_logits), dim=1)

import unittest
from unittest import skipUnless

import torch
import torch.nn as nn

from training.active_boundary_loss import (
    ActiveBoundaryLoss,
    binary_logits_to_two_class,
)
from training.medical_binary_segmentation import MedicalBinarySegmentation


class _DummyMaskNetwork(nn.Module):
    """Deterministic network with the same output container as EoSeg."""

    def forward(self, images):
        foreground = images[:, :1]
        mask_logits = torch.cat((foreground, foreground * 0.5), dim=1)
        class_logits = images.new_zeros((images.shape[0], 2, 3))
        return [mask_logits], [class_logits]


class ActiveBoundaryLossTests(unittest.TestCase):
    def test_label_smoothing_matches_reference_abl_convention(self):
        reference = torch.zeros((2, 8))
        direction = torch.tensor([0, 7])

        targets = ActiveBoundaryLoss._label_smoothed_direction_targets(
            reference, direction, smoothing=0.2
        )

        expected = torch.full((2, 8), 0.2 / 8)
        expected[0, 0] = 1.0 - 0.2
        expected[1, 7] = 1.0 - 0.2
        self.assertTrue(torch.equal(targets, expected))

    def test_binary_logits_bridge_preserves_foreground_probability(self):
        foreground_logits = torch.randn(2, 1, 12, 15)
        two_class_logits = binary_logits_to_two_class(foreground_logits)
        bridged_probability = torch.softmax(two_class_logits, dim=1)[:, 1:2]

        self.assertTrue(
            torch.allclose(
                bridged_probability,
                torch.sigmoid(foreground_logits),
                atol=1e-7,
                rtol=1e-6,
            )
        )

    def test_active_boundary_loss_is_finite_and_backpropagates(self):
        foreground_logits = torch.full((1, 1, 32, 32), -3.0)
        foreground_logits[:, :, 14:28, 14:28] = 3.0
        foreground_logits.requires_grad_()
        target = torch.zeros((1, 32, 32), dtype=torch.long)
        target[:, 8:22, 8:22] = 1

        loss = ActiveBoundaryLoss()(
            binary_logits_to_two_class(foreground_logits), target
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(foreground_logits.grad)
        self.assertTrue(torch.isfinite(foreground_logits.grad).all())
        self.assertGreater(float(foreground_logits.grad.abs().sum()), 0.0)

    def test_no_predicted_boundary_returns_none(self):
        logits = torch.zeros((1, 2, 12, 12), requires_grad=True)
        target = torch.zeros((1, 12, 12), dtype=torch.long)
        target[:, 3:9, 3:9] = 1

        self.assertIsNone(ActiveBoundaryLoss()(logits, target))

    def test_zero_abl_weight_matches_c0_foreground_path_and_loss(self):
        c0 = MedicalBinarySegmentation(
            network=_DummyMaskNetwork(),
            img_size=(12, 15),
            num_classes=2,
            mask_only_training_enabled=True,
        )
        c1_zero = MedicalBinarySegmentation(
            network=_DummyMaskNetwork(),
            img_size=(12, 15),
            num_classes=2,
            mask_only_training_enabled=True,
            active_boundary_loss_enabled=True,
            active_boundary_loss_weight=0.0,
        )
        images = torch.randn((2, 3, 12, 15))
        targets = torch.randint(0, 2, (2, 1, 12, 15)).float()

        c0_logits, c0_probs = c0._foreground_logits_and_probs(images)
        c1_logits, c1_probs = c1_zero._foreground_logits_and_probs(images)
        self.assertTrue(torch.equal(c0_logits, c1_logits))
        self.assertTrue(torch.equal(c0_probs, c1_probs))

        c0_base, _, c0_total = c0._compute_loss_components(
            c0_logits, c0_probs, targets, apply_abl=True
        )
        c1_base, c1_abl, c1_total = c1_zero._compute_loss_components(
            c1_logits, c1_probs, targets, apply_abl=True
        )

        self.assertTrue(torch.equal(c0_base, c1_base))
        self.assertEqual(float(c1_abl), 0.0)
        self.assertTrue(torch.equal(c0_total, c1_total))

    def test_single_channel_input_is_rejected_by_abl(self):
        with self.assertRaises(ValueError):
            ActiveBoundaryLoss()(
                torch.randn((1, 1, 8, 8)), torch.zeros((1, 8, 8), dtype=torch.long)
            )

    @skipUnless(torch.cuda.is_available(), "CUDA AMP check requires an NVIDIA GPU.")
    def test_cuda_amp_forward_backward_is_finite(self):
        foreground_logits = torch.full(
            (1, 1, 32, 32), -3.0, dtype=torch.float16, device="cuda", requires_grad=True
        )
        with torch.no_grad():
            foreground_logits[:, :, 14:28, 14:28] = 3.0
        target = torch.zeros((1, 32, 32), dtype=torch.long, device="cuda")
        target[:, 8:22, 8:22] = 1

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = ActiveBoundaryLoss()(
                binary_logits_to_two_class(foreground_logits), target
            )
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(foreground_logits.grad).all())


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from unittest import skipUnless

import torch
import torch.nn as nn
import yaml

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


class _EpochControlledSegmentation(MedicalBinarySegmentation):
    """Exercise the training step at a selected epoch without a full Trainer."""

    def __init__(self, **kwargs):
        super().__init__(network=_DummyMaskNetwork(), img_size=(32, 32), num_classes=2, **kwargs)
        self.simulated_epoch = 0
        self.logged_values = {}

    @property
    def current_epoch(self):
        return self.simulated_epoch

    def log(self, name, value, **kwargs):
        self.logged_values[name] = value


class ActiveBoundaryLossTests(unittest.TestCase):
    def test_late_schedule_changes_at_epoch_800(self):
        model = _EpochControlledSegmentation(
            mask_only_training_enabled=True,
            active_boundary_loss_enabled=True,
            active_boundary_loss_weight=0.10,
            abl_schedule="late",
            abl_start_epoch=800,
        )

        for phase in ("train", "val"):
            self.assertFalse(model._should_apply_abl(phase, 0))
            self.assertFalse(model._should_apply_abl(phase, 799))
            self.assertTrue(model._should_apply_abl(phase, 800))
            self.assertTrue(model._should_apply_abl(phase, 999))
        self.assertFalse(model._should_apply_abl("test", 999))

    def test_late_training_step_switches_loss_at_epoch_800(self):
        model = _EpochControlledSegmentation(
            mask_only_training_enabled=True,
            active_boundary_loss_enabled=True,
            active_boundary_loss_weight=0.10,
            abl_schedule="late",
            abl_start_epoch=800,
        )
        images = torch.full((1, 3, 32, 32), -4.0)
        images[:, :, 14:28, 14:28] = 4.0
        images.requires_grad_()
        mask = torch.zeros((1, 32, 32), dtype=torch.bool)
        mask[:, 8:22, 8:22] = True
        batch = images, [{"masks": mask, "labels": torch.tensor([1])}]

        model.simulated_epoch = 799
        before = model.training_step(batch, 0)
        self.assertTrue(torch.equal(before, model.logged_values["train/loss_base"]))
        self.assertEqual(float(model.logged_values["train/loss_abl"]), 0.0)
        self.assertEqual(float(model.logged_values["train/abl_active"]), 0.0)
        self.assertEqual(float(model.logged_values["train/abl_effective_weight"]), 0.0)

        model.simulated_epoch = 800
        after = model.training_step(batch, 0)
        after_base = model.logged_values["train/loss_base"]
        after_abl = model.logged_values["train/loss_abl"]
        self.assertTrue(torch.equal(before, after_base))
        self.assertGreater(float(after_abl.detach()), 0.0)
        self.assertTrue(torch.allclose(after, after_base + 0.10 * after_abl))
        self.assertEqual(float(model.logged_values["train/abl_active"]), 1.0)
        self.assertAlmostEqual(float(model.logged_values["train/abl_effective_weight"]), 0.10)
        self.assertTrue(torch.isfinite(after))
        after.backward()
        self.assertIsNotNone(images.grad)
        self.assertTrue(torch.isfinite(images.grad).all())

    def test_late_validation_and_test_loss_semantics(self):
        model = _EpochControlledSegmentation(
            mask_only_training_enabled=True,
            active_boundary_loss_enabled=True,
            active_boundary_loss_weight=0.10,
            abl_schedule="late",
            abl_start_epoch=800,
        )
        images = torch.full((1, 3, 32, 32), -4.0)
        images[:, :, 14:28, 14:28] = 4.0
        mask = torch.zeros((1, 32, 32), dtype=torch.bool)
        mask[:, 8:22, 8:22] = True
        batch = images, [{"masks": mask, "labels": torch.tensor([1])}]

        model.simulated_epoch = 799
        model.validation_step(batch)
        before_dice = model.logged_values["metrics/val_dice"]
        self.assertEqual(float(model.logged_values["val/loss_abl"]), 0.0)
        self.assertTrue(torch.equal(model.logged_values["val/loss"], model.logged_values["val/loss_base"]))

        model.simulated_epoch = 800
        model.validation_step(batch)
        self.assertTrue(torch.equal(before_dice, model.logged_values["metrics/val_dice"]))
        self.assertGreater(float(model.logged_values["val/loss_abl"]), 0.0)
        self.assertTrue(torch.allclose(
            model.logged_values["val/loss"],
            model.logged_values["val/loss_base"] + 0.10 * model.logged_values["val/loss_abl"],
        ))

        model.simulated_epoch = 999
        model.test_step(batch, 0)
        self.assertEqual(float(model.logged_values["test/loss_abl"]), 0.0)
        self.assertTrue(torch.equal(model.logged_values["test/loss"], model.logged_values["test/loss_base"]))

    def test_always_schedule_and_disabled_abl_keep_previous_behavior(self):
        always = _EpochControlledSegmentation(
            mask_only_training_enabled=True,
            active_boundary_loss_enabled=True,
            active_boundary_loss_weight=0.10,
        )
        disabled = _EpochControlledSegmentation(mask_only_training_enabled=True)
        for phase in ("train", "val", "test"):
            self.assertTrue(always._should_apply_abl(phase, 0))
            self.assertTrue(always._should_apply_abl(phase, 999))
            self.assertFalse(disabled._should_apply_abl(phase, 999))

    def test_late_yaml_freezes_budget_and_checkpoint_selection(self):
        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "configs/glas/vit_query_mul_scale_fusion_protocol_b_c1_late_abl_noes_1000.yaml"
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        trainer = config["trainer"]
        model = config["model"]["init_args"]
        callbacks = {entry["class_path"].rsplit(".", 1)[-1]: entry.get("init_args", {})
                     for entry in trainer["callbacks"]}

        self.assertEqual(trainer["max_epochs"], 1000)
        self.assertNotIn("EarlyStopping", callbacks)
        self.assertIn("MedicalValLogger", callbacks)
        self.assertEqual(callbacks["ModelCheckpoint"]["monitor"], "metrics/val_dice")
        self.assertEqual(callbacks["ModelCheckpoint"]["save_top_k"], 1)
        self.assertTrue(callbacks["ModelCheckpoint"]["save_last"])
        self.assertFalse(callbacks["ModelCheckpoint"]["save_weights_only"])
        self.assertEqual(model["abl_schedule"], "late")
        self.assertEqual(model["abl_start_epoch"], 800)
        self.assertEqual(model["active_boundary_loss_weight"], 0.10)
        self.assertEqual(model["metric_threshold"], 0.5)

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

    def test_pdb_sampling_invariant_to_batch_companion_and_order(self):
        # For identical logits, image A's selection must not depend on B.
        gen = torch.Generator().manual_seed(20261011)
        a = torch.randn(1, 1, 64, 64, generator=gen) * 4.0
        b = torch.randn(1, 1, 64, 64, generator=gen) * 2.0
        abl = ActiveBoundaryLoss(max_n_ratio=0.01)

        def pdb(foreground_logits):
            return abl._predicted_boundary(
                binary_logits_to_two_class(foreground_logits)
            )

        single_a = pdb(a)
        single_b = pdb(b)
        pair_ab = pdb(torch.cat((a, b), dim=0))
        pair_ba = pdb(torch.cat((b, a), dim=0))

        self.assertTrue(torch.equal(single_a[0], pair_ab[0]))
        self.assertTrue(torch.equal(single_b[0], pair_ab[1]))
        self.assertTrue(torch.equal(single_a[0], pair_ba[1]))
        self.assertTrue(torch.equal(single_b[0], pair_ba[0]))

    def test_constant_companion_does_not_change_other_image_pdb(self):
        gen = torch.Generator().manual_seed(20261012)
        active_image = torch.randn(1, 1, 64, 64, generator=gen) * 3.0
        blank = torch.zeros_like(active_image)
        abl = ActiveBoundaryLoss(max_n_ratio=0.01)

        single = abl._predicted_boundary(
            binary_logits_to_two_class(active_image)
        )
        paired = abl._predicted_boundary(
            binary_logits_to_two_class(torch.cat((blank, active_image), dim=0))
        )

        self.assertFalse(bool(paired[0].any()))
        self.assertTrue(torch.equal(single[0], paired[1]))

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

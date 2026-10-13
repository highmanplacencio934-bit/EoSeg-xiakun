import unittest

import numpy as np

from training.boundary_f1 import boundary_f1_score


class BoundaryF1Tests(unittest.TestCase):
    def setUp(self):
        self.target = np.zeros((32, 32), dtype=bool)
        self.target[8:24, 9:23] = True

    def test_identical_masks_score_one(self):
        self.assertEqual(boundary_f1_score(self.target, self.target), 1.0)

    def test_one_pixel_shift_is_within_tolerance(self):
        shifted = np.zeros_like(self.target)
        shifted[8:24, 10:24] = True
        self.assertGreater(boundary_f1_score(shifted, self.target), 0.9)

    def test_two_pixel_shift_is_within_inclusive_tolerance(self):
        shifted = np.zeros_like(self.target)
        shifted[8:24, 11:25] = True
        self.assertGreater(boundary_f1_score(shifted, self.target), 0.9)

    def test_five_pixel_shift_scores_lower_than_one_pixel_shift(self):
        shifted_one = np.zeros_like(self.target)
        shifted_one[8:24, 10:24] = True
        shifted_five = np.zeros_like(self.target)
        shifted_five[8:24, 14:28] = True
        score_one = boundary_f1_score(shifted_one, self.target)
        score_five = boundary_f1_score(shifted_five, self.target)
        self.assertLess(score_five, score_one)

    def test_empty_boundary_cases(self):
        empty = np.zeros_like(self.target)
        self.assertEqual(boundary_f1_score(empty, empty), 1.0)
        self.assertEqual(boundary_f1_score(empty, self.target), 0.0)
        self.assertEqual(boundary_f1_score(self.target, empty), 0.0)

    def test_border_touching_mask_is_finite(self):
        border_mask = np.zeros((8, 8), dtype=bool)
        border_mask[:4, :4] = True
        score = boundary_f1_score(border_mask, border_mask)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(score, 1.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            boundary_f1_score(np.zeros((3, 4)), np.zeros((4, 3)))

    def test_score_is_finite_and_bounded(self):
        shifted = np.zeros_like(self.target)
        shifted[8:24, 14:28] = True
        score = boundary_f1_score(shifted, self.target)
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

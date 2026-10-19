import unittest

import numpy as np


class ExposureBoundTest(unittest.TestCase):
    def test_bound_holds_for_predictions_inside_cap(self):
        rng = np.random.default_rng(20260724)
        for cap in [1.0, 10.0, 90.0, 125.0]:
            original = rng.uniform(0.0, 2.5 * cap, size=1000)
            prediction = rng.uniform(0.0, cap, size=1000)
            self_target = np.minimum(original, cap)
            hidden_mse = np.mean((prediction - original) ** 2) - np.mean(
                (prediction - self_target) ** 2
            )
            exposure_sq = np.mean(np.maximum(original - cap, 0.0) ** 2)
            self.assertGreaterEqual(hidden_mse + 1e-10, exposure_sq)

    def test_zero_cap_has_zero_self_scaled_loss(self):
        original = np.asarray([5.0, 10.0, 20.0])
        prediction = np.zeros_like(original)
        target = np.minimum(original, 0.0)
        self.assertEqual(float(np.mean((prediction - target) ** 2)), 0.0)
        self.assertGreater(float(np.mean((prediction - original) ** 2)), 0.0)

    def test_reference_scale_bound_holds_for_candidate_support(self):
        rng = np.random.default_rng(20260725)
        reference_cap = 125.0
        original = rng.uniform(0.0, 300.0, size=1000)
        reference_target = np.minimum(original, reference_cap)
        for candidate_cap in [1.0, 10.0, 90.0, 115.0, 125.0]:
            prediction = rng.uniform(0.0, candidate_cap, size=1000)
            candidate_target = np.minimum(original, candidate_cap)
            reference_gap = np.mean(
                (prediction - reference_target) ** 2
            ) - np.mean((prediction - candidate_target) ** 2)
            reference_exposure_sq = np.mean(
                np.maximum(
                    reference_target - candidate_cap, 0.0
                )
                ** 2
            )
            self.assertGreaterEqual(
                reference_gap + 1e-10, reference_exposure_sq
            )


if __name__ == "__main__":
    unittest.main()

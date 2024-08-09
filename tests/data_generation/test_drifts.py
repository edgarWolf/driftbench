import unittest
import numpy as np
from driftbench.data_generation.latent_information import LatentInformation
from driftbench.data_generation.drifts import (
    LinearDrift,
    DriftSequence
)

class TestLinearDrift(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.N = 100
        self.start = 50
        self.end = 80
        self.m = 2

    def _get_sample_input(self, shape):
        return self.rng.normal(size=shape)

    def test_start_greater_than_end(self):
        start = 50
        end = 30
        self.assertRaises(ValueError, LinearDrift, start, end, 0.3)

    def test_negative_drift_bounds(self):
        start = -2
        end = -1
        self.assertRaises(ValueError, LinearDrift, start, end, 0.3)

    def test_one_dimensional_drift(self):
        y = self._get_sample_input((self.N,))
        linear_drift = LinearDrift(self.start, self.end, self.m)
        drifted = linear_drift.transform(y)
        xs = np.arange(self.end - self.start + 1).reshape(-1, 1)
        expected = np.copy(y).reshape(-1, 1)
        expected[self.start:self.end + 1, :] += \
            self.m * xs
        expected[-(self.N - self.end) + 1:, :] += self.m * xs[-1]
        self.assertEqual(expected.shape, drifted.shape)
        np.testing.assert_array_almost_equal(drifted, expected)

    def test_n_dimensional_drift(self):
        dim = 3
        y = self._get_sample_input((self.N, dim))
        linear_drift = LinearDrift(self.start, self.end, self.m)
        drifted = linear_drift.transform(y)
        expected = np.copy(y)
        xs = np.arange(self.end - self.start + 1).reshape(-1, 1)
        expected[self.start:self.end + 1, :] += self.m * xs
        expected[-(self.N - self.end) + 1:, :] += self.m * xs[-1]
        self.assertEqual(expected.shape, drifted.shape)
        np.testing.assert_array_almost_equal(drifted, expected)


class TestDriftSequence(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.N = 100

    def _get_sample_input(self):
        latents = []
        for i in range(self.N):
            x0 = np.array([1, 5]) + self.rng.normal(size=2, scale=0.2)
            y0 = np.array([0, 4]) + self.rng.normal(size=2)

            x1 = np.array([2, 4]) + self.rng.normal(size=2, scale=0.2)
            y1 = np.array([1, -1]) + self.rng.normal(size=2)

            x2 = np.array([3.5]) + self.rng.normal(size=1, scale=0.2)
            y2 = np.array([-1]) + self.rng.normal(size=1)

            latent = LatentInformation(y0=y0, x0=x0, y1=y1, x1=x1, y2=y2, x2=x2)
            latents.append(latent)
        return latents

    def test_contained_drifts_in_same_feature_and_dimension(self):
        drifts = [
            LinearDrift(start=10, end=50, m=0.2, feature="y0", dimension=0),
            LinearDrift(start=20, end=30, m=0.3, feature="y0", dimension=0)
        ]
        self.assertRaises(ValueError, DriftSequence, drifts)

    def test_overlapping_drifts_in_same_feature_and_dimension(self):
        drifts = [
            LinearDrift(start=10, end=50, m=0.2, feature="y0", dimension=0),
            LinearDrift(start=40, end=70, m=0.3, feature="y0", dimension=0)
        ]
        self.assertRaises(ValueError, DriftSequence, drifts)

    def test_contained_drifts_in_different_features_and_same_dimension(self):
        drifts = [
            LinearDrift(start=10, end=50, m=0.2, feature="y0", dimension=0),
            LinearDrift(start=20, end=30, m=0.3, feature="x0", dimension=0)
        ]
        drift_sequence = DriftSequence(drifts)
        self.assertCountEqual(drift_sequence.drifts, drifts)
        self.assertTrue(all(drift in drifts for drift in drift_sequence.drifts))

    def test_overlapping_drifts_in_different_features_and_same_dimension(self):
        drifts = [
            LinearDrift(start=10, end=50, m=0.2, feature="y0", dimension=0),
            LinearDrift(start=40, end=70, m=0.3, feature="x0", dimension=0)
        ]
        drift_sequence = DriftSequence(drifts)
        self.assertCountEqual(drift_sequence.drifts, drifts)
        self.assertTrue(all(drift in drifts for drift in drift_sequence.drifts))

    def test_contained_drifts_in_same_feature_and_different_dimensions(self):
        drifts = [
            LinearDrift(start=10, end=50, m=0.2, feature="y0", dimension=0),
            LinearDrift(start=20, end=30, m=0.3, feature="y0", dimension=1)
        ]
        drift_sequence = DriftSequence(drifts)
        self.assertCountEqual(drift_sequence.drifts, drifts)
        self.assertTrue(all(drift in drifts for drift in drift_sequence.drifts))

    def test_overlapping_drifts_in_same_feature_and_different_dimensions(self):
        drifts = [
            LinearDrift(start=10, end=50, m=0.2, feature="y0", dimension=0),
            LinearDrift(start=20, end=30, m=0.3, feature="y0", dimension=1)
        ]
        drift_sequence = DriftSequence(drifts)
        self.assertCountEqual(drift_sequence.drifts, drifts)
        self.assertTrue(all(drift in drifts for drift in drift_sequence.drifts))

    def test_apply_drifts(self):
        y0_start, y0_end = 10, 50
        x0_start, x0_end = 20, 30
        y0_dimension = 0
        x0_dimension = 1
        drifts = [
            LinearDrift(start=y0_start, end=y0_end, m=0.2, feature="y0", dimension=y0_dimension),
            LinearDrift(start=x0_start, end=x0_end, m=0.3, feature="x0", dimension=x0_dimension)
        ]
        drift_sequence = DriftSequence(drifts)
        X = self._get_sample_input()
        drifted = drift_sequence.apply(X)

        # Extract drifted features
        Y0 = np.array([x.y0 for x in X])
        drifted_y0 = np.array([drift.y0 for drift in drifted])

        X0 = np.array([x.x0 for x in X])
        drifted_x0 = np.array([drift.x0 for drift in drifted])

        # Extract non drifted features
        Y1 = np.array([x.y1 for x in X])
        drifted_y1 = np.array([drift.y1 for drift in drifted])

        X1 = np.array([x.x1 for x in X])
        drifted_x1 = np.array([drift.x1 for drift in drifted])

        Y2 = np.array([x.y2 for x in X])
        drifted_y2 = np.array([drift.y2 for drift in drifted])

        X2 = np.array([x.x2 for x in X])
        drifted_x2 = np.array([drift.x2 for drift in drifted])

        # Assert non drifted features are still the same
        self.assertTrue(np.array_equal(Y1, drifted_y1))
        self.assertTrue(np.array_equal(X1, drifted_x1))
        self.assertTrue(np.array_equal(Y2, drifted_y2))
        self.assertTrue(np.array_equal(X2, drifted_x2))

        # Assert drifted features are not the same.
        self.assertFalse(np.array_equal(Y0, drifted_x0))
        self.assertFalse(np.array_equal(X0, drifted_y0))

        # Assert drifted features in non drifted dimensions are the same
        self.assertTrue(np.array_equal(Y0[:, 1], drifted_y0[:, 1]))
        self.assertTrue(np.array_equal(X0[:, 0], drifted_x0[:, 0]))

        # Assert the drift is applied in the corresponding interval
        self.assertFalse(
            np.array_equal(
                Y0[y0_start:y0_end + 1][:, y0_dimension],
                drifted_y0[y0_start:y0_end + 1][:, y0_dimension]
            )
        )
        self.assertFalse(
            np.array_equal(
                X0[x0_start:x0_end + 1][:, x0_dimension],
                drifted_x0[x0_start:x0_end + 1][:, x0_dimension]
            )
        )

        # Assert the consistency outside the drift bounds
        self.assertTrue(
            np.array_equal(
                Y0[:y0_start][:, y0_dimension],
                drifted_y0[:y0_start][:, y0_dimension]
            )
        )
        self.assertFalse(
            np.array_equal(
                Y0[y0_end + 1:][:, y0_dimension],
                drifted_y0[y0_end + 1:][:, y0_dimension]
            )
        )
        self.assertTrue(
            np.array_equal(
                X0[:x0_start][:, x0_dimension],
                drifted_x0[:x0_start][:, x0_dimension]
            ),
        )
        self.assertFalse(
            np.array_equal(
                X0[x0_end + 1:][:, x0_dimension],
                drifted_x0[x0_end + 1:][:, x0_dimension]
            )
        )

    def test_get_aggregated_drift_bounds(self):
        drifts = [
            LinearDrift(start=10, end=50, m=0.2, feature="y0", dimension=0),
            LinearDrift(start=20, end=30, m=0.3, feature="y0", dimension=1),
            LinearDrift(start=60, end=80, m=0.1, feature="x0", dimension=1)
        ]
        drift_sequence = DriftSequence(drifts)
        expected = (10, 80)
        self.assertEqual(drift_sequence.get_aggregated_drift_bounds(), expected)

    def test_get_individual_drift_bounds(self):
        drifts = [
            LinearDrift(start=10, end=50, m=0.2, feature="y0", dimension=0),
            LinearDrift(start=20, end=30, m=0.3, feature="y0", dimension=1),
            LinearDrift(start=60, end=80, m=0.1, feature="x0", dimension=1)
        ]
        drift_sequence = DriftSequence(drifts)
        expected = [(10, 50), (20, 30), (60, 80)]
        self.assertEqual(drift_sequence.get_individual_drift_bounds(), expected)

    def test_get_non_aggregated_drift_densities(self):
        drifts = [
            LinearDrift(start=10, end=50, m=0.2, feature="y0", dimension=0),
            LinearDrift(start=60, end=70, m=0.3, feature="x0", dimension=1)
        ]
        drift_sequence = DriftSequence(drifts)
        expected = {
            (10, 50): 1,
            (60, 70): 1,
        }
        self.assertEqual(drift_sequence.get_drift_intensities(), expected)

    def test_get_aggregated_drift_densities(self):
        drifts = [
            LinearDrift(start=10, end=50, m=0.2, feature="y0", dimension=0),
            LinearDrift(start=40, end=70, m=0.3, feature="x0", dimension=1),
        ]
        drift_sequence = DriftSequence(drifts)
        expected = {
            (10, 39): 1,
            (40, 50): 2,
            (51, 70): 1,
        }
        self.assertEqual(drift_sequence.get_drift_intensities(), expected)

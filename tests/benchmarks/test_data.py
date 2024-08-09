import unittest
import numpy as np

from driftbench.data_generation.loaders import load_dataset_specification_from_yaml
from driftbench.benchmarks.data import Dataset


class TestData(unittest.TestCase):

    def setUp(self):

        spec_str = """
        example:
          N: 10
          dimensions: 10
          latent_information:
            !LatentInformation
            y0: [0, 8, 64]
            x0: [0, 2, 4]
            y1: [3, 27]
            x1: [1, 3]
            y2: [12]
            x2: [2]
          drifts:
            !DriftSequence
              - !LinearDrift
                start: 3
                end: 5
                feature: x0
                dimension: 1
                m: 0.1
        """
        self.f = lambda w, x: w[0] * x ** 3 + w[1] * x ** 2 + w[2] * x + w[3]
        self.w0 = np.zeros(4)
        self.spec = load_dataset_specification_from_yaml(spec_str)
        self.Y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])

    def test_init(self):
        data = Dataset("example", self.spec["example"], f=self.f, w0=self.w0, n_variations=2)
        self.assertEqual(data.name, "example")
        self.assertEqual(data.n_variations, 2)
        self.assertEqual(data.spec, self.spec["example"])
        self.assertTrue(np.array_equal(data.w0, self.w0))
        self.assertEqual(data.f, self.f)
        self.assertTrue(np.array_equal(data.Y, self.Y))

    def test_iter(self):
        data = Dataset("example", self.spec["example"], f=self.f, w0=self.w0, n_variations=2)
        i_list, X_list, Y_list = [], [], []
        for i, X, Y in data:
            i_list.append(i)
            X_list.append(X)
            Y_list.append(Y)
        self.assertEqual(len(i_list), 2)
        self.assertEqual(len(X_list), 2)
        self.assertEqual(len(Y_list), 2)
        self.assertListEqual([0, 1], i_list)
        self.assertTrue(np.array_equal(Y_list[0], self.Y))
        self.assertFalse(np.array_equal(X_list[0], X_list[1]))
        self.assertTrue(np.array_equal(Y_list[0], Y_list[1]))

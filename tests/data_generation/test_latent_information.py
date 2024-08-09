import unittest
import numpy as np
from driftbench.data_generation.latent_information import LatentInformation


class TestLatentInformation(unittest.TestCase):
    def test_duplicate_x0(self):
        y0 = np.zeros(3)
        x0 = np.zeros(3)
        y1 = np.array([1., 2.])
        x1 = np.array([0., 1.])
        y2 = np.array([3.])
        x2 = np.array([1.])
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_duplicate_d_dp(self):
        y0 = np.zeros(3)
        x0 = np.arange(3)
        y1 = np.array([1., 2.])
        x1 = np.array([0., 0.])
        y2 = np.array([3.])
        x2 = np.array([1.])
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_duplicate_d2_dp(self):
        y0 = np.zeros(3)
        x0 = np.arange(3)
        y1 = np.array([1., 2.])
        x1 = np.arange(2)
        y2 = np.array([3., 4.])
        x2 = np.array([1., 1.])
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_cp_1d_array(self):
        y0 = np.arange(4).reshape((2, 2,))
        x0 = np.arange(4)
        y1 = np.array([1., 2.])
        x1 = np.arange(2)
        y2 = np.array([3., 4.])
        x2 = np.array([1., 1.])
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_dp_1d_array(self):
        y0 = np.arange(4)
        x0 = np.arange(4).reshape((2, 2))
        y1 = np.arange(4)
        x1 = np.arange(4)
        y2 = np.array([3., 4.])
        x2 = np.array([1., 1.])
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_d_cp_1d_array(self):
        y0 = np.arange(4)
        x0 = np.arange(4)
        y1 = np.arange(4).reshape((2, 2))
        x1 = np.arange(4)
        y2 = np.arange(4)
        x2 = np.arange(4)
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_d_dp_1d_array(self):
        y0 = np.arange(4)
        x0 = np.arange(4)
        y1 = np.arange(4)
        x1 = np.arange(4).reshape((2, 2))
        y2 = np.arange(4)
        x2 = np.arange(4)
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_d2_cp_1d_array(self):
        y0 = np.arange(4)
        x0 = np.arange(4)
        y1 = np.arange(4)
        x1 = np.arange(4)
        y2 = np.arange(4).reshape((2, 2))
        x2 = np.arange(4)
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_d2_dp_1d_array(self):
        y0 = np.arange(4)
        x0 = np.arange(4)
        y1 = np.arange(4)
        x1 = np.arange(4)
        y2 = np.arange(4)
        x2 = np.arange(4).reshape((2, 2))
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_cp_dp_shape_match(self):
        y0 = np.arange(3)
        x0 = np.arange(4)
        y1 = np.arange(4)
        x1 = np.arange(4)
        y2 = np.arange(4)
        x2 = np.arange(4)
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_d_cp_d_dp_shape_match(self):
        y0 = np.arange(4)
        x0 = np.arange(4)
        y1 = np.arange(3)
        x1 = np.arange(4)
        y2 = np.arange(4)
        x2 = np.arange(4)
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

    def test_d2_cp_d2_dp_shape_match(self):
        y0 = np.arange(4)
        x0 = np.arange(4)
        y1 = np.arange(4)
        x1 = np.arange(4)
        y2 = np.arange(3)
        x2 = np.arange(4)
        self.assertRaises(ValueError, LatentInformation, y0, x0, y1, x1, y2, x2)

import unittest
import numpy as np
from driftbench.data_generation.data_generator import CurveGenerator
from driftbench.data_generation.latent_information import LatentInformation


class TestCurveGenerator(unittest.TestCase):

    def setUp(self):
        self.p = lambda w, x: w[0] * x ** 3 + w[1] * x ** 2 + w[2] * x + w[3]
        x0 = np.array([0., 2., 4.])
        y0 = np.array([0., 8., 64.])
        x1 = np.array([1., 3.])
        y1 = np.array([3., 27.])
        x2 = np.array([2.])
        y2 = np.array([12.])
        self.latent_information = LatentInformation(y0, x0, y1, x1, y2, x2)

    def test_curve_generation(self):
        w0 = np.zeros(4)
        curve_generator = CurveGenerator(self.p, w0)
        solution = curve_generator.run([self.latent_information])
        expected = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.assertTupleEqual(solution.shape, (1, 4))
        self.assertTrue(np.allclose(expected, solution))

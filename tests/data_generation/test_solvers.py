import unittest

import jaxlib.xla_extension as jaxlib
import numpy as np
import jax.numpy as jnp
from driftbench.data_generation.solvers import JaxCurveGenerationSolver
from driftbench.data_generation.latent_information import LatentInformation


class TestJaxCurveGenerationSolver(unittest.TestCase):
    def setUp(self):
        self.p = lambda w, x: w[0] * x ** 3 + w[1] * x ** 2 + w[2] * x + w[3]
        x0 = np.array([0., 2., 4.])
        y0 = np.array([0., 8., 64.])
        x1 = np.array([1., 3.])
        y1 = np.array([3., 27.])
        x2 = np.array([2.])
        y2 = np.array([12.])
        self.latent_information = LatentInformation(y0, x0, y1, x1, y2, x2)

    def test_solve(self):
        w0 = jnp.zeros(4)
        solver = JaxCurveGenerationSolver(self.p, w0, max_fit_attemps=1, random_seed=10)
        coefficients = solver.solve([self.latent_information])
        expected = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.assertIs(type(coefficients), jaxlib.ArrayImpl)
        self.assertTupleEqual(coefficients.shape, (1, 4))
        self.assertTrue(np.allclose(expected, coefficients))

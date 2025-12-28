import unittest

import jaxlib.xla_extension as jaxlib
import numpy as np
import jax.numpy as jnp
from driftbench.data_generation.solvers import JaxCurveGenerationSolver
from driftbench.data_generation.latent_information import LatentInformation


class TestJaxCurveGenerationSolver(unittest.TestCase):
    def setUp(self):
        self.p = lambda w, x: w[0] * x**3 + w[1] * x**2 + w[2] * x + w[3]
        x0 = np.array([0.0, 2.0, 4.0])
        y0 = np.array([0.0, 8.0, 64.0])
        x1 = np.array([1.0, 3.0])
        y1 = np.array([3.0, 27.0])
        x2 = np.array([2.0])
        y2 = np.array([12.0])
        self.latent_information = LatentInformation(y0, x0, y1, x1, y2, x2)

    def test_solve_vectorized(self):
        w0 = jnp.zeros(4)
        solver = JaxCurveGenerationSolver(
            self.p, w0, max_fit_attemps=10, vectorize=True
        )
        coefficients = solver.solve([self.latent_information])
        expected = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.assertIs(type(coefficients), jaxlib.ArrayImpl)
        self.assertTupleEqual(coefficients.shape, (1, 4))
        self.assertTrue(np.allclose(expected, coefficients))

    def test_solve_sequentially(self):
        w0 = jnp.zeros(4)
        solver = JaxCurveGenerationSolver(
            self.p, w0, max_fit_attemps=10, vectorize=False
        )
        coefficients = solver.solve([self.latent_information])
        expected = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.assertIs(type(coefficients), jaxlib.ArrayImpl)
        self.assertTupleEqual(coefficients.shape, (1, 4))
        self.assertTrue(np.allclose(expected, coefficients))

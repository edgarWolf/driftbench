from abc import ABCMeta, abstractmethod
from driftbench.data_generation.solvers import JaxCurveGenerationSolver

class DataGenerator(metaclass=ABCMeta):
    """
    Represents a generator for high-dimensional data.
    """
    @abstractmethod
    def run(self, X):
        """
        Generates some high-dimensional data based on the input.
        Args:
            X (list-like): Input to sample from.

        Returns:

        """
        pass


class CurveGenerator(DataGenerator):
    """
    Generator for high-dimensional curves, based on a set of latent information.
    Based on a polynomial and an initial guess, the generator computes coefficients,
    which meet the constraints provided by the latent information.
    """
    def __init__(self, p, w0, max_fit_attempts=100, random_seed=42):
        """
        Args:
            p (func): The polynomial to fit.
            w0 (list-like): The initial guess.
            max_fit_attemps (int): The maxmium number of attempts to refit a curve, if optimization didn't succeed.
            random_seed (int): The random seed for the random number generator.
        """
        self.solver = JaxCurveGenerationSolver(p, w0, max_fit_attempts, random_seed)

    def run(self, X, callback=None):
        return self.solver.solve(X, callback=callback)

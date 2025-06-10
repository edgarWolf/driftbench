import numpy as np
import jax
from driftbench.data_generation.sample import sample_curves
from driftbench.drift_detection.helpers import transform_drift_segments_into_binary


class Dataset:
    """
    Represents a container class for a dataset specification for benchmarking purposes.
    """

    def __init__(self, name, spec, f=None, w0=None, n_variations=5):
        """
        Args:
            name (str): The name of the dataset specification.
            spec (dict): The yaml-specification of the dataset.
            f (Callable): The function to fit the curves.
            w0 (np.ndarray): The inital value for the internal parameters.
            n_variations (int): The number of variations each dataset is sampled.
            Each dataset is sampled as many times as `n_variations` is set, each time with a
            different random seed.
        """
        self.spec = spec
        self.name = name
        self.n_variations = n_variations
        self.w0 = w0
        self.f = f

        drift_bounds = self.spec["drifts"].get_individual_drift_bounds()
        self.Y = transform_drift_segments_into_binary(drift_bounds, self.spec["N"])

    def _generate(self, random_state):
        _, _, curves = sample_curves(
            dataset_specification=self.spec,
            f=self.f,
            w0=self.w0,
            random_state=random_state,
        )
        return curves

    def __iter__(self):
        for i in range(self.n_variations):
            X = self._generate(random_state=i)
            yield i, X, self.Y

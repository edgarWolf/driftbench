from dataclasses import dataclass
import numpy as np


@dataclass
class LatentInformation:
    """
    Represents the local latent information for high-dimensional object,
    which is used to generate such high-dimensional data. Currently, this
    structure is designed for creating curves meeting the conditions provided
    by the attributes defined in this class.

    Args:
        y0 (list-like): The y-values of a function.
        x0 (list-like): The x-values of a function. Hence, no duplicates are allowed.
        y1 (list-like): The y-values of the derivative of a function.
        x1 (list-like): The x-values of the derivative of a function.
        Hence, no duplicates are allowed.
        y2 (list-like): The y-values of the derivative of a function.
        x2 (list-like): The x-values of the second derivative of a function.
        Hence, no duplicates are allowed.

    """
    y0: np.ndarray
    x0: np.ndarray
    y1: np.ndarray
    x1: np.ndarray
    y2: np.ndarray
    x2: np.ndarray

    def __post_init__(self):
        self._validate_duplicates()
        self._validate_1d_array()
        self._validate_matching_shapes()

    def _validate_matching_shapes(self):
        if self.y0.shape != self.x0.shape:
            raise ValueError("Features y0 and x0 are not allowed to have different shape")
        if self.y1.shape != self.x1.shape:
            raise ValueError("Features y1 and x1 are not allowed to have different shape")
        if self.y2.shape != self.x2.shape:
            raise ValueError("Features y2 and x2 are not allowed to have different shape")

    def _validate_1d_array(self):

        if self.y0.ndim != 1:
            raise ValueError("Feature y0 has to be 1d-array.")
        if self.x0.ndim != 1:
            raise ValueError("Feature x0 has to be 1d-array.")
        if self.y1.ndim != 1:
            raise ValueError("Feature y1 has to be 1d-array.")
        if self.x1.ndim != 1:
            raise ValueError("Feature x1 has to be 1d-array.")
        if self.y2.ndim != 1:
            raise ValueError("Feature y2 has to be 1d-array.")
        if self.x2.ndim != 1:
            raise ValueError("Feature x2 has to be 1d-array.")

    def _validate_duplicates(self):
        _, x0_counts = np.unique(self.x0, return_counts=True)
        _, x1_counts = np.unique(self.x1, return_counts=True)
        _, x2_counts = np.unique(self.x2, return_counts=True)
        if np.any(x0_counts > 1):
            raise ValueError("Feature x0 is not allowed to contain duplicates.")
        if np.any(x1_counts > 1):
            raise ValueError("Feature x1 is not allowed to contain duplicates.")
        if np.any(x2_counts > 1):
            raise ValueError("Feature x2 is not allowed to contain duplicates.")

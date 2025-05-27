import numpy as np
import copy
from abc import ABCMeta, abstractmethod
from itertools import groupby, combinations

class Drift(metaclass=ABCMeta):
    """
    Represents a drift for 1d or 2d input.
    """
    def __init__(self, start, end, feature=None, dimension=0) -> None:
        """
        Args:
            start (int): The start index.
            end (int): The end index.
            feature (str): The feature the drift should be applied on.
            dimension (int): The dimension the drift should be applied on.
        """
        self._validate_drift_bounds(start, end)
        self.start = start
        self.end = end
        self.feature = feature
        self.dimension = dimension

    def _validate_drift_bounds(self, start, end):
        if start >= end:
            raise ValueError("End must be greater than start.")
        if start < 0 or end < 0:
            raise ValueError("Drift bounds are not allowed to be negative.")

    @abstractmethod
    def transform(self, X):
        """
        Applies the transformation specified by the drift object on the given input
        Args:
            X (numpy.ndarray): The 1d- or 2d-input data to be drifted.
        """
        pass


class DriftSequence:
    """
    Represents a sequence of drifts, which will be applied on a latent information object.
    """
    def __init__(self, drifts):
        """
        Args:
            drifts (list[Drift]): A list of drifts which are being used for the transformation.
        """
        self._validate_drifts(drifts)
        self.drifts = sorted(drifts, key=lambda drift: drift.start)

    def apply(self, X):
        """
        Applies the transformation by the given drifts on the latent information input.
        Args:
            X (list[LatentInformation]): The list of latent information the drifts are applied on.

        Returns (list): A list of drifted latent information according to the drift sequence.
        """
        drifted = copy.deepcopy(X)
        for drift in self.drifts:
            feature = np.array([getattr(x, drift.feature) for x in drifted])
            feature[:, drift.dimension] = drift.transform(feature[:, drift.dimension]).flatten()
            for i, x in enumerate(drifted):
                setattr(x, drift.feature, feature[i])
        return drifted

    def get_aggregated_drift_bounds(self):
        """
        Returns the aggregated drift bounds, i.e. the maximum range where drifts are applied.
        Returns:
            A tuple of (int, int), where the first value denotes the start index and the second value the
            end index of the aggregated drift bounds.
        """
        start = self.drifts[0].start
        end = self.drifts[-1].end
        return start, end

    def get_individual_drift_bounds(self):
        """
        Returns the drift bounds for each individual drift in the drift sequence.
        Returns:
            A list of tuples of (int, int), where the first value denotes the start of the drift,
            and the second value the end of the drift.
        """
        return [(drift.start, drift.end) for drift in self.drifts]

    def get_drift_intensities(self):
        """
        Returns the intensities for each range in the drift sequence. Each drift has a base intensity of 1,
        and when multiple drifts overlap, the intensity becomes the number of the drifts present in the given
        range.
        Returns:
            A dictionary with tuples as keys and ints as values.
            The keys indicate the range of the drift intensity, and the values indicate the intensity.
        """
        intensities = {}
        drift_intensities_array = np.zeros((len(self.drifts),
                                           np.max([drift.end for drift in self.drifts]) + 1))
        for i, drift in enumerate(self.drifts):
            drift_intensities_array[i, drift.start:drift.end + 1] = 1
        stacked_drift_intensities = np.sum(drift_intensities_array, axis=0)

        for intensity in range(1, np.max(stacked_drift_intensities).astype(int) + 1):
            indices = np.where(stacked_drift_intensities == intensity)[0]
            split_indices = np.where(np.diff(indices) > 1)[0] + 1
            bounds = np.split(indices, split_indices)
            for start, end in [(bound[0], bound[-1]) for bound in bounds]:
                intensities[(start, end)] = intensity
        return intensities

    def _validate_drifts(self, drifts):
        # Group drifts by their feature and their dimension they apply on.
        drifts_sorted = sorted(drifts, key=lambda drift: (drift.feature, drift.dimension))
        drifts_grouped = groupby(drifts_sorted, key=lambda drift: (drift.feature, drift.dimension))
        # Check within these groups if an overlap exists.
        for (feature, dimension), curr_drifts in drifts_grouped:
            curr_drifts = list(curr_drifts)
            for i, j in combinations(range(len(curr_drifts)), 2):
                drift1 = curr_drifts[i]
                drift2 = curr_drifts[j]
                if drift1.start <= drift2.end and drift2.start <= drift1.end:
                    raise ValueError(f"Drifts are not allowed to overlap. "
                               f"Overlapping drift at feature {feature} in dimension {dimension}")


class LinearDrift(Drift):
    """
    Represents a linear drift for a 1d or 2d-input, i.e. a drift
    where the input data is drifted in a linear fashion.
    """
    def __init__(self, start, end, m, feature=None, dimension=0):
        """
        Args:
            start (int): The start index.
            end (int): The end index.
            m (float): The slope of the linear drift. Usually in the range (-1, 1)
            feature (str): The feature the drift should be applied on.
            dimension (int): The dimension the drift should be applied on.
        """
        super().__init__(start, end, feature=feature, dimension=dimension)
        self.m = m

    def transform(self, X):
        drifted = np.copy(X).astype(float)
        if drifted.ndim == 1:
            drifted = drifted.reshape(-1, 1)
        # Use 0 based x indices for computing the slope at a given position
        xs = np.arange(self.end - self.start + 1).reshape(-1, 1)
        drifted[self.start:self.end + 1, :] += self.m * xs
        # Maintain data according to new data after drift happened.
        after_drift_idx = drifted.shape[0] - self.end
        drifted[-after_drift_idx + 1:, :] += self.m * xs[-1]
        return drifted

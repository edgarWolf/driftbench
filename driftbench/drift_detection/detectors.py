import numpy as np
import pandas as pd
import warnings
from abc import (
    ABCMeta,
    abstractmethod,
)
from driftbench.drift_detection.helpers import (
    binarize_scores,
)
from driftbench.drift_detection.metrics import Metric
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


class Detector(metaclass=ABCMeta):

    hparams = []

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, X, y, metric):
        if not isinstance(metric, Metric):
            raise ValueError("The metric has to be an instance of the metric class.")
        prediction = self.predict(X)
        return metric(prediction, y)

    @property
    def name(self):
        return self.__class__.__name__

    def get_hparams(self):
        return {
            hp: getattr(self, hp) for hp in self.hparams
        }


class RandomGuessDetector(Detector):
    def __init__(self, random_seed=42):
        self.rng = np.random.RandomState(random_seed)

    def predict(self, X):
        N = X.shape[0]
        scores = self.rng.normal(size=N)
        scores = np.cumsum(scores)
        return scores


class AlwaysGuessDriftDetector(Detector):
    def predict(self, X):
        return np.ones(X.shape[0])


class RollingMeanDifferenceDetector(Detector):

    hparams = ['window_size', 'center', 'fillna_strategy']

    def __init__(self, window_size, center=False, fillna_strategy=None):
        self.window_size = window_size
        self.center = center
        self.fillna_strategy = fillna_strategy

    def predict(self, X):
        series_data = pd.DataFrame(X)
        prediction = series_data.rolling(self.window_size, center=self.center).mean().max(axis=1).diff().abs()
        if self.fillna_strategy:
            fill_value = self.fillna_strategy(prediction)
            prediction = prediction.fillna(fill_value)
        return prediction.values


class RollingMeanStandardDeviationDetector(Detector):
    def __init__(self, window_size, center=False, fillna_strategy=None):
        self.window_size = window_size
        self.center = center
        self.fillna_strategy = fillna_strategy

    def predict(self, X):
        df_data = pd.DataFrame(X)
        prediction = df_data.rolling(self.window_size, center=self.center).mean().max(axis=1).rolling(
            self.window_size).std()
        if self.fillna_strategy:
            fill_value = self.fillna_strategy(prediction)
            prediction = prediction.fillna(fill_value)
        return prediction.values


class SlidingKSWINDetector(Detector):
    def __init__(self, window_size, stat_size, offset):
        self.window_size = window_size
        self.stat_size = stat_size
        self.offset = offset

    def predict(self, X):
        N = X.shape[0]
        prediction = np.ones((N,))
        stat_batches = np.array([X[i:i + self.stat_size] for i in range(N - self.stat_size + 1)])
        data_batches = np.array([X[i:i + self.window_size] for i in range(self.offset, N - self.window_size + 1)])
        num_batches = np.min([stat_batches.shape[0], data_batches.shape[0]])
        # Store last calculated score for the data batch containing not enough data.
        last_score = 1.
        for i in range(num_batches):
            stat_batch, data_batch = stat_batches[i], data_batches[i]
            _, p_value = ks_2samp(stat_batch, data_batch, method="auto")
            score = np.log1p(1.0 / p_value)
            prediction[i + self.window_size - 1] = score
            last_score = score
        prediction[:self.window_size] = last_score
        return prediction


class AggregateFeatureAlgorithm(Detector):
    def __init__(self, agg_feature_func, algorithm):
        self.algorithm = algorithm
        self.agg_feature_func = agg_feature_func

    def predict(self, X):
        input_with_feature = np.apply_along_axis(self.agg_feature_func, 1, X)
        return self.algorithm.predict(input_with_feature)

    @property
    def name(self):
        return self.algorithm.name


class ClusterDetector(Detector):
    supported_methods = ["kmeans", "gaussian mixture"]

    hparams = ['method', 'n_centers']

    def __init__(self, n_centers, method="kmeans", random_state=42):
        if not self._validate_cluster_method(method):
            raise ValueError(
                f"Unknown method {method}: Supported cluster methods are {ClusterDetector.supported_methods}.")
        self.n_centers = n_centers
        self.method = method
        self.random_state = random_state

    def _validate_cluster_method(self, method):
        return method in ClusterDetector.supported_methods

    def predict(self, X):
        warnings.simplefilter("ignore")
        if self.method == "kmeans":
            return np.min(KMeans(n_clusters=self.n_centers, random_state=self.random_state).fit_transform(X), axis=1)
        elif self.method == "gaussian mixture":
            gm = GaussianMixture(n_components=self.n_centers, random_state=self.random_state)
            gm.fit(X)
            return -1.0 * gm.score_samples(X)


class AutoencoderDetector(Detector, nn.Module):
    """

    Args:
        hidden_layers (list): List of number of neurons in each layer after input of encoder
        retrain (bool): If true, model is always retrained when predict is called.
    """
    _activation_functions = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }

    _optimizers = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
    }

    _device = "cuda:0" if torch.cuda.is_available() else "cpu"

    hparams = ['activation', 'lr', 'optim', 'num_epochs', 'hidden_layers']

    def __init__(self, hidden_layers, detector, activation='tanh', num_epochs=10,
                 batch_size=32, optim="adam", lr=0.001, retrain_always=False):
        Detector.__init__(self)
        nn.Module.__init__(self)

        self.lr = lr
        self.retrain = retrain_always
        self.activation = activation
        self.detector = detector
        self.optim = optim
        self.criterion = nn.MSELoss()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.losses = []
        self.hidden_layers = hidden_layers
        self.is_trained = False
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed_x = self.decoder(latent_space)
        return reconstructed_x

    def _build_model(self, input_size):
        encoder_layers = [input_size] + self.hidden_layers
        decoder_layers = self.hidden_layers[::-1] + [input_size]
        self.encoder = self._build_encoder(encoder_layers, self.activation)
        self.decoder = self._build_decoder(decoder_layers, self.activation)
        self.optimizer = self._optimizers[self.optim](self.parameters(), lr=self.lr)
        self.to(self._device)

    def _train(self, X):
        # Reset losses from possible previous training
        self.losses = []
        # Actual training loop
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            for i, inputs in enumerate(dataloader):
                self.optimizer.zero_grad()
                inputs = inputs[0]
                outputs = self(inputs)
                loss = self.criterion(outputs, inputs)
                self.losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
        self.is_trained = True

    def _build_encoder(self, layers, activation):
        encoder_layers = []
        for i in range(1, len(layers)):
            encoder_layers.append(nn.Linear(layers[i - 1], layers[i]))
            encoder_layers.append(self._activation_functions[activation])
        return nn.Sequential(*encoder_layers)

    def _build_decoder(self, layers, activation):
        decoder_layers = []
        for i in range(1, len(layers)):
            decoder_layers.append(nn.Linear(layers[i - 1], layers[i]))
            # Don't append activation function for output layer
            if i < len(layers) - 1:
                decoder_layers.append(self._activation_functions[activation])
        return nn.Sequential(*decoder_layers)

    def _prepare_data(self, X):
        """
        Scales (if necessary) the data and places it afterwards on device
        """
        if self.retrain or not self.is_trained:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        return torch.tensor(X, dtype=torch.float32).to(self._device)

    def predict(self, X):
        X = self._prepare_data(X)

        if self.retrain or not self.is_trained:
            self._build_model(X.shape[1])
            self._train(X)

        with torch.no_grad():
            latent_space = self.encoder(X).detach().cpu().numpy()
        return self.detector.predict(latent_space)

import os
default_n_threads = 20
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import logging
import numpy as np
import json

from driftbench.data_generation.loaders import load_dataset_specification_from_yaml

from driftbench.drift_detection.detectors import (
    RandomGuessDetector,
    RollingMeanDifferenceDetector,
    RollingMeanStandardDeviationDetector,
    AggregateFeatureAlgorithm,
    SlidingKSWINDetector,
    ClusterDetector,
    AutoencoderDetector,
)

from driftbench.drift_detection.metrics import (
    TemporalAUC,
    SoftTAUC,
    AUC,
)

from driftbench.benchmarks.data import Dataset

detectors = [
    RandomGuessDetector(),
    RollingMeanDifferenceDetector(window_size=20),
    RollingMeanDifferenceDetector(window_size=40),
    RollingMeanStandardDeviationDetector(window_size=20),
    AggregateFeatureAlgorithm(
        agg_feature_func=np.mean,
        algorithm=SlidingKSWINDetector(window_size=20, stat_size=20, offset=10),
    ),
    ClusterDetector(n_centers=10, method="kmeans"),
    ClusterDetector(n_centers=10, method="gaussian mixture"),
    ClusterDetector(n_centers=5, method="kmeans"),
    ClusterDetector(n_centers=5, method="gaussian mixture"),
    AutoencoderDetector(
        hidden_layers=[80, 50, 20, 10, 2],
        retrain_always=True,
        detector=AggregateFeatureAlgorithm(
            agg_feature_func=np.mean,
            algorithm=SlidingKSWINDetector(window_size=20, stat_size=20, offset=10),
        ),
        num_epochs=50,
        batch_size=200,
        lr=0.001,
    ),
    AutoencoderDetector(
        hidden_layers=[80, 50, 20, 10, 2],
        retrain_always=True,
        detector=AggregateFeatureAlgorithm(
            agg_feature_func=np.mean,
            algorithm=SlidingKSWINDetector(window_size=20, stat_size=20, offset=10),
        ),
        num_epochs=100,
        batch_size=200,
        lr=0.0001,
    ),
    AutoencoderDetector(
        hidden_layers=[80, 20, 4],
        retrain_always=True,
        detector=AggregateFeatureAlgorithm(
            agg_feature_func=np.mean,
            algorithm=SlidingKSWINDetector(window_size=20, stat_size=20, offset=10),
        ),
        num_epochs=10,
        batch_size=200,
        lr=0.0001,
    )
]

metrics = [
    TemporalAUC(rule='step'),
    TemporalAUC(rule='trapez'),
    SoftTAUC(rule='step'),
    AUC(),
]

logger = logging.getLogger("driftbench")


def make_datasets(data_spec, n_variations=5):

    datasets = [
        Dataset(
            name=name,
            spec=spec,
            n_variations=n_variations,
        ) for name, spec in data_spec.items()
    ]
    return datasets


def run_benchmarks(datasets):
    benchmarks = []

    for dataset in datasets:
        logger.info(f'Benchmark dataset {dataset.name}')
        for variation, X, Y in dataset:
            logger.info(f'Evaluate dataset {dataset.name}: {variation+1}/{dataset.n_variations}')
            for detector in detectors:
                logger.info(f'\tEvaluate detector {detector.name}')
                prediction = detector.predict(X)

                data = {
                    'dataset_name': dataset.name,
                    'variation': variation,
                    'detector_name': detector.name,
                    'hparams': detector.get_hparams(),
                    'prediction': prediction.tolist(),
                    'ground_truth': Y.tolist(),
                }

                for metric_fn in metrics:
                    data[metric_fn.name] = metric_fn(prediction, Y)

                benchmarks.append(data)

    return benchmarks


if __name__ == '__main__':

    path_to_dataset_specs = './data/paper_datasets.yaml'
    path_to_results = 'benchmarks.json'

    with open(path_to_dataset_specs, 'r') as f:
        data_spec = load_dataset_specification_from_yaml(f)

    datasets = make_datasets(data_spec)

    benchmarks = run_benchmarks(datasets)
    logger.info(f"Write results to {path_to_results}")
    with open(path_to_results, 'w') as f:
        json.dump(benchmarks, f)

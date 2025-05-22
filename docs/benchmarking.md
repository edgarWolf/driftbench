
# Metrics

# An example benchmark pipeline


First, we set up a list of detectors we would like to benchmark:


```python
from driftbench.drift_detection.detectors import (
    ClusterDetector,
    AutoencoderDetector,
    SlidingKSWINDetector,
    MMDDetector,
)

detectors = [
    ClusterDetector(n_centers=5, method="gaussian mixture"),
    AutoencoderDetector(
        hidden_layers=[80, 20, 4],
        retrain_always=True,
        detector=AggregateFeatureAlgorithm(
            algorithm=SlidingKSWINDetector(window_size=20, stat_size=20, offset=10),
        ),
        num_epochs=10,
        batch_size=200,
        lr=0.0001,
    ),
    AutoencoderDetector(
        hidden_layers=[80, 20, 4],
        retrain_always=True,
        detector=MMDDetector(window_size=20, stat_size=20, offset=10),
        num_epochs=10,
        batch_size=200,
        lr=0.0001,
    )
]
```


Next, we set up the dataset we would like to benchmark on.

```python
from driftbench.data_generation.loaders import load_dataset_specification_from_yaml
from driftbench.benchmarks.data import Dataset

with open("/path/to/your/spec.yml", 'r') as f:
    data_spec = load_dataset_specification_from_yaml(f)

dataset = Dataset(
    name="dataset-1",
    spec=data_spec,
    n_variations=5,
) 
```

Finally, we specify which metrics should be tested:


```python

from driftbench.drift_detection.metrics import TemporalAUC, AUC
metrics = [TemporalAUC(rule='step'), AUC()]
```

Finally, we test and evaluate all detectors on all datasets for all metrics:

```python

for dataset in datasets:
    for variation, X, Y in dataset:
        logger.info(f'Evaluate dataset {dataset.name}: {variation+1}/{dataset.n_variations}')
        for detector in detectors:
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
                score = metric_fn(prediction, Y)
                print(
                    f"Detector {detector.name} got {score} ({metrics_fn.name}) "
                    f"on dataset variation {variation}"
                )
```

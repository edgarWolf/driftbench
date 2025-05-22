# driftbench

This is the documentation of `driftbench`, a framwork to synthetically generate process curves
and to benchmark process drift detectors.

## Getting started

This is a minimal example to generate `N=10` curves from a cubic function:

```python
import numpy as np
from driftbench.data_generation.loaders import load_dataset_specification_from_yaml
from driftbench.data_generation.sample import sample_curves

input = """
example:
  N: 10
  dimensions: 10
  latent_information:
    !LatentInformation
    y0: [0, 8, 64]
    x0: [0, 2, 4]
    y1: [3, 27]
    x1: [1, 3]
    y2: [12]
    x2: [2]
  drifts:
    !DriftSequence
      - !LinearDrift
        start: 3
        end: 5
        feature: x0
        dimension: 1
        m: 0.1
"""

def f(w, x):
    return w[0] * x ** 3 + w[1] * x ** 2 + w[2] * x + w[3]

w0 = np.zeros(4)
dataset = load_dataset_specification_from_yaml(input)
coefficients, latent_information, curves = sample_curves(dataset["example"], w0=w0, f=f)

```


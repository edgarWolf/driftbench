# driftbench

Benchmarking framework for generating high-dimensional synthetic drifted data and evaluating
models.

The corresponding open-access paper, [Edgar Wolf and Tobias Windisch (2024), A method to benchmark high-dimensional 
process drift detection](https://arxiv.org/abs/2409.03669), describes the
method in detail.

To run the benchmarks, execute:

```python

python run_benchmarks.py
```


To visualize the model performance, run

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_benchmark(df):

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    sns.boxplot(data=df, x="TAUC", y="Detector", hue='Data',  native_scale=True, ax=axes[0])
    sns.boxplot(data=df, x="SoftTAUC", y="Detector", hue='Data', native_scale=True, ax=axes[1])
    sns.boxplot(data=df, x="AUC", y="Detector", hue='Data', native_scale=True, ax=axes[2])
    
    for ax in axes[1:]:
        ax.legend([])
        ax.set_yticklabels([])
    
    axes[0].set_xlabel('TAUC')
    axes[1].set_xlabel('sTAUC')
    axes[2].set_xlabel('AUC')
    for ax in axes:
        ax.grid()
        ax.set_ylabel('')
    fig.tight_layout()
    
    return fig

df = pd.read_json('benchmarks.json') 
fig = plot_benchmark(df)

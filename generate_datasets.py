import numpy as np

from driftbench.data_generation.loaders import load_dataset_specification_from_yaml
from driftbench.data_generation.sample import sample_curves

from datetime import datetime


if __name__ == '__main__':

    with open('./data/paper_datasets.yaml', 'r') as f:

        data_spec = load_dataset_specification_from_yaml(f)

        for dataset_name, spec in data_spec.items():
            print(datetime.now(), dataset_name)
            w, curves = sample_curves(spec, f=None, random_state=10)
            np.save(f'{dataset_name}.npy', curves)
            print(datetime.now())

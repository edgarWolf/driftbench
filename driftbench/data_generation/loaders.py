from abc import ABCMeta, abstractmethod

import numpy as np
import yaml
from driftbench.data_generation.drifts import (
    LinearDrift,
    DriftSequence,
)
from driftbench.data_generation.latent_information import LatentInformation


class Loader(metaclass=ABCMeta):
    @abstractmethod
    def get_loader(self):
        pass


class YamlConstructor(metaclass=ABCMeta):
    @abstractmethod
    def construct(self, loader, node):
        pass

    @property
    @abstractmethod
    def tag(self):
        pass


class LinearDriftYamlConstructor(YamlConstructor):
    def construct(self, loader, node):
        return LinearDrift(**loader.construct_mapping(node))

    @property
    def tag(self):
        return "!LinearDrift"


class DriftSequenceYamlConstructor(metaclass=ABCMeta):
    def construct(self, loader, node):
        return DriftSequence(loader.construct_sequence(node, deep=True))

    @property
    def tag(self):
        return "!DriftSequence"


class LatentInformationYamlConstructor(metaclass=ABCMeta):
    def _convert_lists_to_numpy(self, loader, node):
        latent_information_dict = {}
        for node_key, node_value in node.value:
            feature_name = loader.construct_object(node_key)
            feature_val = loader.construct_sequence(node_value)
            latent_information_dict[feature_name] = np.asarray(feature_val)
        return latent_information_dict

    def construct(self, loader, node):
        return LatentInformation(**self._convert_lists_to_numpy(loader, node))

    @property
    def tag(self):
        return "!LatentInformation"


class LinearDriftLoader(Loader):
    def get_loader(self):
        loader = yaml.SafeLoader
        constructor = LinearDriftYamlConstructor()
        loader.add_constructor(constructor.tag, constructor.construct)
        return loader


class DriftSequenceLoader(Loader):
    def get_loader(self):
        loader = yaml.SafeLoader
        linear_drift_constructor = LinearDriftYamlConstructor()
        drift_sequence_constructor = DriftSequenceYamlConstructor()
        loader.add_constructor(drift_sequence_constructor.tag, drift_sequence_constructor.construct)
        loader.add_constructor(linear_drift_constructor.tag, linear_drift_constructor.construct)
        return loader


class LatentInformationLoader(Loader):

    def get_loader(self):
        loader = yaml.SafeLoader
        latent_information_constructor = LatentInformationYamlConstructor()
        loader.add_constructor(latent_information_constructor.tag, latent_information_constructor.construct)
        return loader


class DatasetSpecificationLoader(Loader):
    def get_loader(self):
        loader = yaml.SafeLoader
        linear_drift_constructor = LinearDriftYamlConstructor()
        drift_sequence_constructor = DriftSequenceYamlConstructor()
        latent_information_constructor = LatentInformationYamlConstructor()
        loader.add_constructor(linear_drift_constructor.tag, linear_drift_constructor.construct)
        loader.add_constructor(drift_sequence_constructor.tag, drift_sequence_constructor.construct)
        loader.add_constructor(latent_information_constructor.tag, latent_information_constructor.construct)
        return loader


def load_dataset_specification_from_yaml(dataset_yaml):
    return yaml.load(dataset_yaml, Loader=DatasetSpecificationLoader().get_loader())

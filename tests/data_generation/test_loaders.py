import unittest
from unittest import mock

import yaml
import numpy as np

from driftbench.data_generation.drifts import LinearDrift, DriftSequence
from driftbench.data_generation.loaders import (
    LinearDriftLoader,
    DriftSequenceLoader,
    LatentInformationLoader,
    DatasetSpecificationLoader,
    load_dataset_specification_from_yaml
)
from driftbench.data_generation.latent_information import LatentInformation


class TestLinearDriftLoader(unittest.TestCase):
    def test_correct_format_input(self):
        loader = LinearDriftLoader()
        input = """
        !LinearDrift
        start: 50
        end: 70
        feature: y0
        dimension: 0
        m: 0.02
        """
        output = yaml.load(input, Loader=loader.get_loader())
        self.assertEqual(output.start, 50)
        self.assertEqual(output.end, 70)
        self.assertEqual(output.feature, "y0")
        self.assertEqual(output.dimension, 0)
        self.assertEqual(output.m, 0.02)
        self.assertIsInstance(output, LinearDrift)

    def test_incorrect_format_input(self):
        loader = LinearDriftLoader()
        input = """
        !LinearDrift
        start: 50
        end: 70
        feature: y0
        dimension: 0
        m: 0.02
        length: 42
        """
        with self.assertRaises(TypeError):
            yaml.load(input, Loader=loader.get_loader())


class TestDriftSequenceLoader(unittest.TestCase):
    def test_correct_format_input(self):
        loader = DriftSequenceLoader()
        input = """
        !DriftSequence
          - !LinearDrift
            start: 50
            end: 70
            feature: y0
            dimension: 0
            m: 0.02
          - !LinearDrift
            start: 100
            end: 120
            feature: y0
            dimension: 0
            m: 0.02
        """
        output = yaml.load(input, Loader=loader.get_loader())
        self.assertIsInstance(output, DriftSequence)
        self.assertEqual(len(output.drifts), 2)
        self.assertEqual(output.drifts[0].start, 50)
        self.assertEqual(output.drifts[0].end, 70)
        self.assertEqual(output.drifts[0].feature, "y0")
        self.assertEqual(output.drifts[0].dimension, 0)
        self.assertEqual(output.drifts[0].m, 0.02)
        self.assertIsInstance(output.drifts[0], LinearDrift)
        self.assertEqual(output.drifts[1].start, 100)
        self.assertEqual(output.drifts[1].end, 120)
        self.assertEqual(output.drifts[1].feature, "y0")
        self.assertEqual(output.drifts[1].dimension, 0)
        self.assertEqual(output.drifts[1].m, 0.02)
        self.assertIsInstance(output.drifts[1], LinearDrift)

    def test_incorrect_format_input(self):
        loader = DriftSequenceLoader()
        input = """
                !DriftSequence
              - !LinearDrift
                start: 50
                end: 70
                feature: y0
                dimension: 0
                m: 0.02
                name: test42
              - !LinearDrift
                start: 100
                end: 120
                feature: y0
                dimension: 0
                m: 0.02
                """
        with self.assertRaises(TypeError):
            yaml.load(input, Loader=loader.get_loader())


class TestLatentInformationLoader(unittest.TestCase):
    def test_correct_format_input(self):
        loader = LatentInformationLoader()
        input = """
                !LatentInformation
                y0: [4, 7, 5]
                x0: [0, 2, 4]
                y1: [2, -2]
                x1: [1, 3]
                y2: [-1]
                x2: [1]
                """
        output = yaml.load(input, Loader=loader.get_loader())
        self.assertIsInstance(output, LatentInformation)
        self.assertTrue(np.array_equal(output.y0, [4, 7, 5]))
        self.assertTrue(np.array_equal(output.x0, [0, 2, 4]))
        self.assertTrue(np.array_equal(output.y1, [2, -2]))
        self.assertTrue(np.array_equal(output.x1, [1, 3]))
        self.assertTrue(np.array_equal(output.y2, [-1]))
        self.assertTrue(np.array_equal(output.x2, [1]))

    def test_incorrect_format_input(self):
        loader = LatentInformationLoader()
        input = """
                   !LatentInformation
                   y0: [4, 7, 5]
                   x0: [0, 2, 4]
                   y1: [2, -2]
                   x1: [1, 3]
                   y2: [-1]
                   x2: [1]
                   x3: [42]
                   """
        with self.assertRaises(TypeError):
            yaml.load(input, Loader=loader.get_loader())


class TestDatasetSpecificationLoader(unittest.TestCase):
    def test_correct_format_input(self):
        loader = DatasetSpecificationLoader()
        input = """
          latent_information:
            !LatentInformation
              y0: [4, 7, 5]
              x0: [0, 2, 4]
              y1: [2, -2]
              x1: [1, 3]
              y2: [-1]
              x2: [1]
          drifts:
            !DriftSequence
              - !LinearDrift
                 start: 100
                 end: 120
                 feature: y0
                 dimension: 1
                 m: 0.02
                """
        output = yaml.load(input, Loader=loader.get_loader())
        self.assertIsInstance(output, dict)
        latent_information = output["latent_information"]
        self.assertIsInstance(latent_information, LatentInformation)
        self.assertTrue(np.array_equal(latent_information.y0, np.array([4, 7, 5])))
        self.assertTrue(np.array_equal(latent_information.x0, np.array([0, 2, 4])))
        self.assertTrue(np.array_equal(latent_information.y1, np.array([2, -2])))
        self.assertTrue(np.array_equal(latent_information.x1, np.array([1, 3])))
        self.assertTrue(np.array_equal(latent_information.y2, np.array([-1])))
        self.assertTrue(np.array_equal(latent_information.x2, np.array([1])))
        drifts = output["drifts"]
        self.assertIsInstance(drifts, DriftSequence)
        self.assertEqual(len(drifts.drifts), 1)
        self.assertIsInstance(drifts.drifts[0], LinearDrift)
        self.assertEqual(drifts.drifts[0].start, 100)
        self.assertEqual(drifts.drifts[0].end, 120)
        self.assertEqual(drifts.drifts[0].feature, "y0")
        self.assertEqual(drifts.drifts[0].m, 0.02)

    def test_incorrect_format_input(self):
        loader = DatasetSpecificationLoader()
        input = """
              latent_information:
                !LatentInformation
                  y0: [4, 7, 5]
                  x0: [0, 2, 4]
                  y1: [2, -2]
                  x1: [1, 3]
                  y2: [-1]
                  x2: [1]
              drifts:
                !DriftSequence
                  - !LinearDrift
                     start: 100
                     end: 120
                     feature: y0
                     dimension: 1
                     m: 0.02
                     some_key: 42
                    """
        with self.assertRaises(TypeError):
            yaml.load(input, Loader=loader.get_loader())


class TestLoadDatasetSpecificationFromYaml(unittest.TestCase):
    def test_correct_format_input(self):
        input = """
            example:
              N: 200
              dimensions: 10
              x_scale: 0.02
              y_scale: 1.0
              latent_information:
                !LatentInformation
                y0: [4, 7, 5]
                x0: [0, 2, 4]
                y1: [2, -2]
                x1: [1, 3]
                y2: [-1]
                x2: [1]
              drifts:
                !DriftSequence
                  - !LinearDrift
                    start: 100
                    end: 120
                    feature: y0
                    dimension: 1
                    m: 0.02
                    """
        dataset = load_dataset_specification_from_yaml(input)
        self.assertIsInstance(dataset, dict)
        self.assertTrue("example" in dataset)
        self.assertEqual(dataset["example"]["N"], 200)
        self.assertEqual(dataset["example"]["dimensions"], 10)
        self.assertEqual(dataset["example"]["x_scale"], 0.02)
        self.assertEqual(dataset["example"]["y_scale"], 1.0)
        latent_information = dataset["example"]["latent_information"]
        self.assertIsInstance(latent_information, LatentInformation)
        self.assertTrue(np.array_equal(latent_information.y0, np.array([4, 7, 5])))
        self.assertTrue(np.array_equal(latent_information.x0, np.array([0, 2, 4])))
        self.assertTrue(np.array_equal(latent_information.y1, np.array([2, -2])))
        self.assertTrue(np.array_equal(latent_information.x1, np.array([1, 3])))
        self.assertTrue(np.array_equal(latent_information.y2, np.array([-1])))
        self.assertTrue(np.array_equal(latent_information.x2, np.array([1])))
        drifts = dataset["example"]["drifts"]
        self.assertIsInstance(drifts, DriftSequence)
        self.assertEqual(len(drifts.drifts), 1)
        self.assertIsInstance(drifts.drifts[0], LinearDrift)
        self.assertEqual(drifts.drifts[0].start, 100)
        self.assertEqual(drifts.drifts[0].end, 120)
        self.assertEqual(drifts.drifts[0].feature, "y0")
        self.assertEqual(drifts.drifts[0].m, 0.02)

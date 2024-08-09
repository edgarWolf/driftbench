import unittest
import copy
import numpy as np
from driftbench.data_generation.loaders import load_dataset_specification_from_yaml
from driftbench.data_generation.sample import sample_curves


class TestSampleCurves(unittest.TestCase):
    def test_sample_curves(self):
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
        coefficients, curves = sample_curves(dataset["example"], w0=w0, f=f)
        self.assertTupleEqual(coefficients.shape, (10, 4))
        self.assertTupleEqual(curves.shape, (10, 10))

    def test_sample_curves_with_custom_scales(self):
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
        coefficients, curves = sample_curves(dataset["example"], w0=w0, f=f, random_state=42)
        altered_dataset = copy.deepcopy(dataset)
        altered_dataset["example"]["x_scale"] = 0.02
        altered_dataset["example"]["y_scale"] = 1.0
        altered_coefficients, altered_curves = sample_curves(altered_dataset["example"], w0=w0, f=f, random_state=42)
        self.assertTupleEqual(coefficients.shape, altered_coefficients.shape)
        self.assertTupleEqual(curves.shape, altered_curves.shape)
        self.assertFalse(np.allclose(coefficients, altered_coefficients))
        self.assertFalse(np.allclose(curves, altered_curves))

    def test_sample_curves_with_measurement_noise(self):
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
        coefficients, curves = sample_curves(dataset["example"], w0=w0, f=f, random_state=42)
        altered_coefficients, altered_curves = sample_curves(dataset["example"], w0=w0, f=f, random_state=42,
                                                             measurement_scale=0.1)
        self.assertTupleEqual(coefficients.shape, altered_coefficients.shape)
        self.assertTupleEqual(curves.shape, altered_curves.shape)
        self.assertTrue(np.allclose(coefficients, altered_coefficients))
        self.assertFalse(np.allclose(curves, altered_curves))

    def test_sample_curves_with_no_measurement_noise(self):
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
        coefficients, curves = sample_curves(dataset["example"], random_state=42, w0=w0, f=f, measurement_scale=0.)
        altered_coefficients, altered_curves = sample_curves(dataset["example"], w0=w0, f=f, random_state=42)
        self.assertTupleEqual(coefficients.shape, altered_coefficients.shape)
        self.assertTupleEqual(curves.shape, altered_curves.shape)
        self.assertTrue(np.allclose(coefficients, altered_coefficients))
        self.assertFalse(np.allclose(curves, altered_curves))

    def test_sample_curves_with_function_in_yaml(self):
        input = """
        example:
          N: 10
          dimensions: 10
          func: w[0] * x ** 3 + w[1] * x**2 + w[2] * x + w[3]
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
        w0 = np.zeros(4)
        dataset = load_dataset_specification_from_yaml(input)
        coefficients, curves = sample_curves(dataset["example"], w0=w0)

        def f(w, x):
            return w[0] * x ** 3 + w[1] * x ** 2 + w[2] * x + w[3]

        dataset_without_function = copy.deepcopy(dataset)
        del dataset_without_function["example"]["func"]
        coefficients2, curves2 = sample_curves(dataset_without_function["example"], w0=w0, f=f)
        self.assertTupleEqual(coefficients.shape, (10, 4))
        self.assertTupleEqual(curves.shape, (10, 10))
        self.assertTrue(np.allclose(coefficients, coefficients2))
        self.assertTrue(np.allclose(curves, curves2))

    def test_sample_curves_with_function_and_w_init_in_yaml(self):
        input = """
        example:
          N: 10
          dimensions: 10
          func: w[0] * x ** 3 + w[1] * x**2 + w[2] * x + w[3]
          w_init: [0, 0, 0, 0]
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
        dataset = load_dataset_specification_from_yaml(input)
        coefficients, curves = sample_curves(dataset["example"])
        dataset_without_function = copy.deepcopy(dataset)
        del dataset_without_function["example"]["w_init"]
        coefficients2, curves2 = sample_curves(dataset_without_function["example"], w0=np.zeros(4))
        self.assertTupleEqual(coefficients.shape, (10, 4))
        self.assertTupleEqual(curves.shape, (10, 10))
        self.assertTrue(np.allclose(coefficients, coefficients2))
        self.assertTrue(np.allclose(curves, curves2))

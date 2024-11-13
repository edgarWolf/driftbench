import numpy as np
import jax # noqa
from driftbench.data_generation.latent_information import LatentInformation
from driftbench.data_generation.data_generator import CurveGenerator


def sample_curves(dataset_specification, f=None, w0=None, random_state=2024, measurement_scale=None, callback=None):
    dimensions = dataset_specification["dimensions"]
    drifts = dataset_specification.get("drifts")
    x_scale = dataset_specification.get("x_scale", 0.02)
    y_scale = dataset_specification.get("y_scale", 0.1)
    func = _get_func(dataset_specification, f)
    w_init = _get_w_init(dataset_specification, w0)
    rng = np.random.RandomState(random_state)
    latent_information = _generate_latent_information(dataset_specification, rng, x_scale, y_scale)
    if drifts is not None:
        latent_information = drifts.apply(latent_information)
    data_generator = CurveGenerator(func, w_init)
    w = data_generator.run(latent_information, callback=callback)
    x_min = int(np.min(dataset_specification["latent_information"].x0))
    x_max = int(np.max(dataset_specification["latent_information"].x0))
    xs = np.linspace(x_min, x_max, dimensions)
    curves = np.array([func(w_i, xs) for w_i in w])
    # Apply a default noise of 5% of the mean of the sampled curves
    if measurement_scale is None:
        scale = 0.05 * np.mean(curves)
        curves = rng.normal(loc=curves, scale=scale)
    else:
        curves = rng.normal(loc=curves, scale=measurement_scale)
    return w, latent_information, curves


def _generate_latent_information(dataset_specification, rng, x_scale, y_scale):
    N = dataset_specification["N"]
    base_latent_information = dataset_specification["latent_information"]
    latent_information = []
    for i in range(N):
        # Apply some random noise on the base values
        x0 = base_latent_information.x0 + rng.normal(size=len(base_latent_information.x0), scale=x_scale)
        y0 = base_latent_information.y0 + rng.normal(size=len(base_latent_information.y0), scale=y_scale)

        x1 = base_latent_information.x1 + rng.normal(size=len(base_latent_information.x1), scale=x_scale)
        y1 = base_latent_information.y1 + rng.normal(size=len(base_latent_information.y1), scale=y_scale)

        x2 = base_latent_information.x2 + rng.normal(size=len(base_latent_information.x2), scale=x_scale)
        y2 = base_latent_information.y2 + rng.normal(size=len(base_latent_information.y2), scale=y_scale)
        latent_information.append(LatentInformation(y0, x0, y1, x1, y2, x2))
    return latent_information


def _get_func(dataset_specification, f):
    if f is not None:
        return f
    elif "func" in dataset_specification:
        func_expr = dataset_specification["func"]
        return eval(f"lambda w, x: {func_expr}")
    else:
        raise ValueError("""No function provided. Either specify function in yaml
                file, or provide a function as argument to the sample
                function.""")


def _get_w_init(dataset_specification, w0):
    if w0 is not None:
        return w0
    elif "w_init" in dataset_specification:
        w_init_expr = dataset_specification["w_init"]
        if isinstance(w_init_expr, str):
            w_init = eval(w_init_expr)
        else:
            w_init = w_init_expr
        return np.array(w_init, dtype=np.float64)
    else:
        raise ValueError("""No initial guess provided. Either specify initial guess in
                   yaml file, or provide an inital guess as argument to the sample function.""")

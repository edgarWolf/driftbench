import numpy as np
import jax  # noqa
from driftbench.data_generation.latent_information import LatentInformation
from driftbench.data_generation.data_generator import CurveGenerator


def sample_curves(
    dataset_specification,
    f=None,
    w0=None,
    random_state=2024,
    measurement_scale=None,
    callback=None,
):
    dimensions = dataset_specification["dimensions"]
    drifts = dataset_specification.get("drifts")
    x_scale = dataset_specification.get("x_scale", 0.02)
    y_scale = dataset_specification.get("y_scale", 0.1)
    func = _get_func(dataset_specification, f)
    w_init = _get_w_init(dataset_specification, w0)
    rng = np.random.RandomState(random_state)
    latent_information = _generate_latent_information(
        dataset_specification, rng, x_scale, y_scale
    )
    if drifts is not None:
        latent_information = drifts.apply(latent_information)
    data_generator = CurveGenerator(func, w_init)
    w = data_generator.run(latent_information, callback=callback)
    x_range = np.concatenate(
        (
            dataset_specification["latent_information"].x0,
            dataset_specification["latent_information"].x1,
            dataset_specification["latent_information"].x2,
        )
    )
    x_min, x_max = int(np.min(x_range)), int(np.max(x_range))
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
    xis = [f"x{i}" for i in range(3)]
    yis = [f"y{i}" for i in range(3)]
    for i in range(N):
        # Apply some random noise on the base values
        latent_dict = {}
        for xi in xis:
            latent_dict[xi] = getattr(base_latent_information, xi) + rng.normal(
                size=len(getattr(base_latent_information, xi)), scale=x_scale
            )
        for yi in yis:
            latent_dict[yi] = getattr(base_latent_information, yi) + rng.normal(
                size=len(getattr(base_latent_information, yi)), scale=y_scale
            )
        latent_information.append(LatentInformation(**latent_dict))
    return latent_information


def _get_func(dataset_specification, f):
    if f is not None:
        return f
    elif "func" in dataset_specification:
        func_expr = dataset_specification["func"]
        return eval(f"lambda w, x: {func_expr}")
    else:
        raise ValueError(
            """No function provided. Either specify function in yaml
                file, or provide a function as argument to the sample
                function."""
        )


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
        raise ValueError(
            """No initial guess provided. Either specify initial guess in
                   yaml file, or provide an inital guess as argument to the sample function."""
        )

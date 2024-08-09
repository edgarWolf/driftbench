import numpy as np
import matplotlib.pyplot as plt


def plot_curve_with_latent_information(coefficients, p, latent_information, title=None, ax=None, y_lim=None):
    """
    Plots the reconstructed wave with the given coefficients and a polynomial with the ground truth
    defined by the latent information.
    Args:
        coefficients (list-like): The coefficients for the polynomial.
        p (func): The polynomial.
        latent_information (LatentInformation): The latent information containing
        the ground truth for the polynomial and it's coefficients
        title (str): The title for the plot.
        ax (matplotlib.axes).: Extern axes if this function is used for external created figure.
        y_lim (tuple(int, int): The y-lim for the plot.

    Returns:

    """
    xs = [latent_information.x0, latent_information.x1, latent_information.x2]
    min_x, max_x = np.min(np.concatenate(xs)), max(np.concatenate(xs))
    x = np.linspace(min_x, max_x)

    if not ax:
        fig, ax = plt.subplots()

    ax.plot(x, p(coefficients, x))

    # Plot the given x-values
    for xx in latent_information.x0:
        ax.axvline(xx, linestyle='dashed', color='black')

    # Plot slope according to first derivative
    for slope, x_slope in zip(latent_information.y1, latent_information.x1):
        xxs = [x for x in range(int(x_slope - 1), int(x_slope + 3.))]
        dx_vals = np.array(
            [(slope * x) - (slope * x_slope - p(coefficients, x_slope)) for x in xxs])
        ax.scatter(x_slope, p(coefficients, x_slope), alpha=0.4, color="green")
        ax.plot(xxs, dx_vals, c="green")

    # Plot curvature
    for x_curvature, curvature in zip(latent_information.x2, latent_information.y2):
        label = "convex" if curvature > 0.0 else "concave"
        ax.axvline(x_curvature, linestyle='dashed', color='purple', label=label)

    # Mark the corresponding y-values
    for yy, xx in zip(latent_information.y0, latent_information.x0):
        ax.scatter(xx, yy, color="red")

    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend()
    if title:
        ax.set_title(title)


def plot_curves(curves, xs, title=None, cmap="coolwarm", ylim=None):
    fig, ax = plt.subplots()
    cmap_obj = plt.get_cmap(name=cmap)
    cycler = plt.cycler("color", cmap_obj(np.linspace(0, 1, curves.shape[0])))
    ax.set_prop_cycle(cycler)
    ax.plot(xs, curves.T)
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)

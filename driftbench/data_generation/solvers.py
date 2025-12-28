from abc import ABCMeta, abstractmethod
import jax
import jax.numpy as jnp
from jax import (
    grad,
    vmap,
    jit,
)
from jax.scipy.optimize import minimize
from functools import partial

jax.config.update("jax_enable_x64", True)


class Solver(metaclass=ABCMeta):
    """
    Represents a backend for solving an optimization problem.
    """

    @abstractmethod
    def solve(self, X):
        """
        Solves an optimization problem defined by the solver.

        Args:
            X (list-like): Input to optimize according to solver instance.

        Returns:
            (np.ndarray|jnp.ndarray): The parameters obtained by solving the optimzation problem.
        """
        pass


class JaxCurveGenerationSolver(Solver):
    """
    Fits latent information according to a given function.
    """

    def __init__(self, f, w0, max_fit_attemps, vectorize=True):
        """
        Args:
            f (Callable): The function.
            w0 (list-like): The initial guess for the solution.
            max_fit_attemps (int): The maxmium number of attempts to refit a curve, if optimization didn't succeed.
            vectorize (bool): Whether to vectorize the optimization over multiple latent information instances.
        """
        self.f = jit(vmap(partial(f), in_axes=(None, 0)))
        df_dx = grad(f, argnums=1)
        df_dx2 = grad(df_dx, argnums=1)
        self.df_dx = jit(vmap(partial(df_dx), in_axes=(None, 0)))
        self.df_dx2 = jit(vmap(partial(df_dx2), in_axes=(None, 0)))
        self.w0 = jnp.array(w0)
        self.max_fit_attempts = max_fit_attemps
        self.vectorize = vectorize
        self.min_func = _minimize

        # Compile function beforehand if vectorization is enabled for one compilation only.
        if self.vectorize:
            self.min_func = jit(
                vmap(
                    partial(_minimize),
                    in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0),
                ),
                static_argnums=(0, 1, 2),
            )

    def _latents_to_array(self, latents):
        l_x0_mat = jnp.array([latent.x0 for latent in latents])
        l_x1_mat = jnp.array([latent.x1 for latent in latents])
        l_x2_mat = jnp.array([latent.x2 for latent in latents])
        l_y0_mat = jnp.array([latent.y0 for latent in latents])
        l_y1_mat = jnp.array([latent.y1 for latent in latents])
        l_y2_mat = jnp.array([latent.y2 for latent in latents])
        return l_x0_mat, l_x1_mat, l_x2_mat, l_y0_mat, l_y1_mat, l_y2_mat

    def _solve_sequentially(self, X, callback=None):
        coefficients = []
        solution = self.w0
        for i, latent in enumerate(X):
            result = self.min_func(
                self.f,
                self.df_dx,
                self.df_dx2,
                solution,
                latent.y0,
                latent.x0,
                latent.y1,
                latent.x1,
                latent.y2,
                latent.x2,
            )
            if not result.success:
                result = self._refit(self.f, self.df_dx, self.df_dx2, latent)
            solution = result.x
            if callback:
                jax.debug.callback(callback, i, solution)
            coefficients.append(solution)
        return jnp.array(coefficients)

    def _solve_vectorized(self, X):
        solution = jnp.tile(self.w0, (len(X), 1))
        l_x0_mat, l_x1_mat, l_x2_mat, l_y0_mat, l_y1_mat, l_y2_mat = (
            self._latents_to_array(X)
        )
        result = self.min_func(
            self.f,
            self.df_dx,
            self.df_dx2,
            solution,
            l_y0_mat,
            l_x0_mat,
            l_y1_mat,
            l_x1_mat,
            l_y2_mat,
            l_x2_mat,
        )
        return result.x

    def solve(self, X, callback=None):
        if self.vectorize:
            return self._solve_vectorized(X)
        else:
            return self._solve_sequentially(X, callback)

    def _refit(self, f, df_dx, df_dx2, latent):
        # Restart with initial guess in order to be independent of previous solutions.
        solution = self.w0
        current_fit_attempts = 0
        success = False
        result = None
        # Fallback strategy: If fit is not successful, try again and use previous solution
        # for the same problem as starting point until convergence.
        while not success and current_fit_attempts < self.max_fit_attempts:
            current_fit_attempts += 1
            result = self.min_func(
                f,
                df_dx,
                df_dx2,
                solution,
                latent.y0,
                latent.x0,
                latent.y1,
                latent.x1,
                latent.y2,
                latent.x2,
            )
            solution = result.x
            success = result.success
        return result


@partial(jit, static_argnums=(0, 1, 2))
def _minimize(f, df_dx, df_dx2, w, y0, x0, y1, x1, y2, x2):
    return minimize(
        _solve,
        w,
        method="BFGS",
        args=(
            f,
            df_dx,
            df_dx2,
            jnp.array(y0),
            jnp.array(x0),
            jnp.array(y1),
            jnp.array(x1),
            jnp.array(y2),
            jnp.array(x2),
        ),
    )


@partial(jit, static_argnums=(1, 2, 3))
def _solve(w, f, df_dx, df_dx2, y0, x0, y1, x1, y2, x2):
    px = f(w, x0)
    dp_px = df_dx(w, x1)
    dp_px2 = df_dx2(w, x2)
    return _loss(y0, y1, y2, px, dp_px, dp_px2)


@jit
def _loss(y0, y1, y2, fx, df_fx, df_fx2):
    return (
        ((fx - y0) ** 2).sum() + ((df_fx - y1) ** 2).sum() + ((df_fx2 - y2) ** 2).sum()
    )

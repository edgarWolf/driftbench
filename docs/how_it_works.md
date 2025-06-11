# Mathematical model


# Technical implementation in `jax`
This package uses `jax` as its backend for generating synthetic curves.
In particular, `jax` is used for:

- Solving non-linear optimization problems.
- Automatic differentiation for calculating partial derivates of `f`.
- Just-in-time (JIT)-compilation with XLA for performance optimization.

For more detailed information regarding the XLA-compilation, please see the 
[offical `jax` documenation](https://docs.jax.dev/en/latest/index.html)
or [XLA-documentation](https://openxla.org/xla/tf2xla?hl=en).

These three points ensure an efficient generation of curves, while being
able to control the latent information used and the behaviour of drifts applied
on the curves.

The method used to solve optimization problems is the 
[LBFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) algorithm, which is supported by `jax`.
The corresponding functions in order to compute the error term, running a iteration of the
optimization solving problem, and computing the gradients is all done in functions which are 
compiled just-in-time and can be run on a GPU.
The procedure can be described as follows:

- Choose a function $f(w(t), x)$, which describes the shape of the curves to generate with 
inital internal parameters $w_0(t)$.
- Provide problem constraints encoded in
[`driftbench.data_generation.latent_information.LatentInformation`][driftbench.data_generation.latent_information.LatentInformation]
objects.
- Compute the $i$ partial derivates of $f(w(t), x)$ with respect to $x$.
- For each latent information object, compute $w(t)$ according to the LBFGS-algorithm.
- Return all computed solutions $w(t)$ for each curve.


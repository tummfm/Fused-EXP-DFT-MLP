from jax.experimental.optimizers import *
import jax.numpy as np
from jax import jit

# TODO jit and remove prints

def scale_tree_vector(vector, scale):
    normalize = lambda g: g * scale
    return tree_util.tree_map(normalize, vector)

def squared_euclidian_norm(tree):
  """Compute the squared euclidian norm of a pytree of arrays. Useful for norming gradients and adaptive step sizes."""
  leaves, _ = tree_flatten(tree)
  return sum(jnp.vdot(x, x) for x in leaves)

def polyak_step_size_gradient(gradient, loss, c=0.5, max_step=np.inf, f_min=0.):
    """Hacky implementation of stochastic gradient descent with stochastic Polyak step-size
    https://arxiv.org/abs/2002.10542. As we cannot give the optimizer update function more
    than the gradient, we apply the adaptive stepsize to the gradient a-prior and choose a
    step size of 1. for sgd.

    Args:
    c: positive scalar hyperparameter to be chosen based on properties of optimized function.
       For strongly convex functions the optimal value is 0.5
    max_step: the upper bound of the step size that can be taken, especially important when
            gradients are close to 0
    f_min: Minimum value the optimizer can achieve. In most machine learning problems where
           the model can interpolate the data, the optimal value is 0.

    Returns:
        Gradient to be used with sgd(step_size=1.) to achieve polyak adaptive stepsite algorithm.
    """
    step_size = (loss - f_min) / (c * squared_euclidian_norm(gradient))
    step_size = np.min(np.array([step_size, max_step]), dtype=np.float32)
    print('step_size=', step_size)
    return scale_tree_vector(gradient, step_size)

def polyak_step_size(c=0.5, max_step=np.inf, f_min=0.):
    """Conveniance wrapper implementing polyak adaptive step size using existing sgd implementation."""
    opt_init, sgd_update, get_params = sgd(1.)  # polyak adaptive step size: needs to be 1
    def update(step, gradient, opt_state, loss):
        transformed_grad = polyak_step_size_gradient(gradient, loss, c=c, f_min=f_min, max_step=max_step)
        return sgd_update(step, transformed_grad, opt_state)
    return opt_init, update, get_params

def adagrad_norm_gradient(gradient, b_2_state):
  """Optimizers in jax.experimental.optimizers act independently on all gradients in gradient-dict.
  We therefore cannot implement a norm over all parts of the gradient there. We therefore modify
  the gradient such that we can directly use sgd, where nu is the constant sgd step size."""
  b_2_state += squared_euclidian_norm(gradient)
  normed_grad = scale_tree_vector(gradient, 1./np.sqrt(b_2_state))
  print('Learning rate=', 0.1 / np.sqrt(b_2_state))
  print('normed grad norm:', l2_norm(normed_grad))
  return normed_grad, b_2_state

def adagrad_norm(nu=0.1, b_0=1.):
    """Conveniance wrapper implementing adagrad-norm using existing sgd implementation."""
    sgd_init, sgd_update, sgd_get_params = sgd(nu)  # polyak adaptive step size: needs to be 1
    def init(x):
        sgd_init_sate = sgd_init(x)
        return (sgd_init_sate, b_0)

    def update(step, gradient, opt_state):
        sgd_state, b = opt_state
        transformed_grad, b_new = adagrad_norm_gradient(gradient, b)
        new_sgd_state = sgd_update(step, transformed_grad, sgd_state)
        return (new_sgd_state, b_new)

    def get_params(opt_state):
        sgd_state, b = opt_state
        return sgd_get_params(sgd_state)

    return init, update, get_params

def linear_warmup_exponential_decay(step_size, warmup_steps, decay_rate, decay_steps):
    # cast to f32
    step_size = jnp.array(step_size, dtype=jnp.float32)
    warmup_steps = jnp.array(warmup_steps, dtype=jnp.float32)
    decay_rate = jnp.array(decay_rate, dtype=jnp.float32)
    decay_steps = jnp.array(decay_steps, dtype=jnp.float32)
    one = np.array([1.], dtype=jnp.float32)

    warmup_schedule = polynomial_decay(one / warmup_steps, warmup_steps, one)
    exp_decay_schedule = exponential_decay(step_size, decay_steps, decay_rate)
    def warmup_exponential_schedule(i):
        return warmup_schedule(i) * exp_decay_schedule(i)
    return warmup_exponential_schedule


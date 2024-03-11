import jax.numpy as jnp
import timeit
import jax
from typing import Callable
from jax.scipy.linalg import cholesky, solve_triangular,solve

def sqeuclidean_distance(x: jnp.array, y: jnp.array) -> float:
    return jnp.sum((x - y) ** 2)


def distmat(func: Callable, x: jnp.ndarray, y: jnp.ndarray)-> float :
    """distance matrix"""
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)
  


def pdist_squareform(x: jnp.ndarray, y: jnp.ndarray)-> float :
    """squared euclidean distance matrix

    Notes
    -----
    This is equivalent to the scipy commands

    >>> from scipy.spatial.distance import pdist, squareform
    >>> dists = squareform(pdist(X, metric='sqeuclidean')
    """
    return distmat(sqeuclidean_distance, x, y)
  

def rbf(x: jnp.ndarray, y: jnp.ndarray)-> float :

  return jnp.exp(- 0.5 * pdist_squareform(x,y))

def is_equal(x: jnp.ndarray, y: jnp.ndarray)-> float :

  f = lambda  x,y : jnp.allclose(x,y)* 1.

  return distmat(f,x,y)


def constrainer(a,b):

  return lambda x:  a + (b-a)* jax.lax.logistic(x)


def unconstrainer(a,b):

  return lambda y : jax.lax.log((y-a)/(b-y))  

def update_cholesky(K_XX,L_XX,K_XX_inv,K_XZ,K_ZZ):
    
    S11 = L_XX
    #S12 = L_XX @ (K_XX_inv @ K_XZ)
    S12 = L_XX @ solve(K_XX,K_XZ,assume_a="pos")
    S21 = jnp.zeros_like(S12).T
    S22 = jnp.linalg.cholesky(K_ZZ-S12.T@S12).T

    
    ###K_XZ_XZ = L.T@L
    L = jnp.vstack ([
                jnp.hstack([S11,S12]),
                jnp.hstack([S21,S22])
                ])
    
    
    
    
    return L

  
class CodeTimer:
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start)
        print('Code block' + self.name + ' took: ' + str(self.took) + ' s')



"""Utility functions to handle flattening policy parameters for the GP.

This file was taken from evosax and modified in order to simplify dependencies:

https://github.com/RobertTLange/evosax/blob/3c4f34828094e2b1e6798003190f10b67e2a4a9e/evosax/core/reshape.py#L23
"""
import jax
import jax.numpy as jnp
import chex
from typing import Union, Optional
from jax import vjp, flatten_util
from jax.tree_util import tree_flatten


def ravel_pytree(pytree):
    leaves, _ = tree_flatten(pytree)
    flat, _ = vjp(ravel_list, *leaves)
    return flat


def ravel_list(*lst):
    return (
        jnp.concatenate([jnp.ravel(elt) for elt in lst])
        if lst
        else jnp.array([])
    )


class ParameterReshaper(object):
    def __init__(
        self,
        placeholder_params: Union[chex.ArrayTree, chex.Array],
        n_devices: Optional[int] = None,
        verbose: bool = True,
    ):
        """Reshape flat parameters vectors into generation eval shape."""
        # Get network shape to reshape
        self.placeholder_params = placeholder_params

        # Set total parameters depending on type of placeholder params
        flat, self.unravel_pytree = flatten_util.ravel_pytree(
            placeholder_params
        )
        self.total_params = flat.shape[0]
        self.reshape_single = jax.jit(self.unravel_pytree)

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices
        if self.n_devices > 1 and verbose:
            print(
                f"ParameterReshaper: {self.n_devices} devices detected. Please"
                " make sure that the ES population size divides evenly across"
                " the number of devices to pmap/parallelize over."
            )

        if verbose:
            print(
                f"ParameterReshaper: {self.total_params} parameters detected"
                " for optimization."
            )

    def reshape(self, x: chex.Array) -> chex.ArrayTree:
        """Perform reshaping for a 2D matrix (pop_members, params)."""
        vmap_shape = jax.vmap(self.reshape_single)
        if self.n_devices > 1:
            x = self.split_params_for_pmap(x)
            map_shape = jax.pmap(vmap_shape)
        else:
            map_shape = vmap_shape
        return map_shape(x)

    def multi_reshape(self, x: chex.Array) -> chex.ArrayTree:
        """Reshape parameters lying already on different devices."""
        # No reshaping required!
        vmap_shape = jax.vmap(self.reshape_single)
        return jax.pmap(vmap_shape)(x)

    def flatten(self, x: chex.ArrayTree) -> chex.Array:
        """Reshaping pytree parameters (population) into flat array."""
        vmap_flat = jax.vmap(ravel_pytree)
        if self.n_devices > 1:
            # Flattening of pmap paramater trees to apply vmap flattening
            def map_flat(x):
                x_re = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), x)
                return vmap_flat(x_re)

        else:
            map_flat = vmap_flat
        flat = map_flat(x)
        # Out shape: (pop, params)
        return flat

    def flatten_single(self, x: chex.ArrayTree) -> chex.Array:
        """Reshaping pytree parameters (single) into flat array."""
        return ravel_pytree(x)

    def multi_flatten(self, x: chex.Array) -> chex.ArrayTree:
        """Flatten parameters lying remaining on different devices."""
        # No reshaping required!
        vmap_flat = jax.vmap(ravel_pytree)
        return jax.pmap(vmap_flat)(x)

    def split_params_for_pmap(self, param: chex.Array) -> chex.Array:
        """Helper reshapes param (bs, #params) into (#dev, bs/#dev, #params)."""
        return jnp.stack(jnp.split(param, self.n_devices))

    @property
    def vmap_dict(self) -> chex.ArrayTree:
        """Get a dictionary specifying axes to vmap over."""
        vmap_dict = jax.tree_map(lambda x: 0, self.placeholder_params)
        return vmap_dict
    





"""Utility functions to compute running statistics.

This file was taken from acme and modified in order to simplify dependencies:

https://github.com/deepmind/acme/blob/master/acme/jax/running_statistics.py
"""


import dataclasses
from typing import Tuple

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Array:
  """Describes a numpy array or scalar shape and dtype.

  Similar to dm_env.specs.Array.
  """
  shape: Tuple[int, ...]
  dtype: jnp.dtype

from typing import Any, Iterable, Mapping, Union


import jax.numpy as jnp

# Define types for nested arrays and tensors.
NestedArray = jnp.ndarray
NestedTensor = Any

# pytype: disable=not-supported-yet
NestedSpec = Union[
    Array,
    Iterable['NestedSpec'],
    Mapping[Any, 'NestedSpec'],
]
# pytype: enable=not-supported-yet

Nest = Union[NestedArray, NestedTensor, NestedSpec]

from typing import Any, Optional, Tuple


from flax import struct
import jax
import jax.numpy as jnp


def _zeros_like(nest: Nest, dtype=None) -> Nest:
  return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)


def _ones_like(nest: Nest, dtype=None) -> Nest:
  return jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, dtype or x.dtype), nest)


@struct.dataclass
class NestedMeanStd:
  """A container for running statistics (mean, std) of possibly nested data."""
  mean: Nest
  std: Nest


@struct.dataclass
class RunningStatisticsState(NestedMeanStd):
  """Full state of running statistics computation."""
  count: jnp.ndarray
  summed_variance: Nest


def init_state(nest: Nest) -> RunningStatisticsState:
  """Initializes the running statistics for the given nested structure."""
  dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

  return RunningStatisticsState(
      count=jnp.zeros((), dtype=dtype),
      mean=_zeros_like(nest, dtype=dtype),
      summed_variance=_zeros_like(nest, dtype=dtype),
      # Initialize with ones to make sure normalization works correctly
      # in the initial state.
      std=_ones_like(nest, dtype=dtype))


def _validate_batch_shapes(batch: NestedArray,
                           reference_sample: NestedArray,
                           batch_dims: Tuple[int, ...]) -> None:
  """Verifies shapes of the batch leaves against the reference sample.

  Checks that batch dimensions are the same in all leaves in the batch.
  Checks that non-batch dimensions for all leaves in the batch are the same
  as in the reference sample.

  Arguments:
    batch: the nested batch of data to be verified.
    reference_sample: the nested array to check non-batch dimensions.
    batch_dims: a Tuple of indices of batch dimensions in the batch shape.

  Returns:
    None.
  """
  def validate_node_shape(reference_sample: jnp.ndarray,
                          batch: jnp.ndarray) -> None:
    expected_shape = batch_dims + reference_sample.shape
    assert batch.shape == expected_shape, f'{batch.shape} != {expected_shape}'

  jax.tree_util.tree_map(validate_node_shape, reference_sample, batch)


def update(state: RunningStatisticsState,
           batch: Nest,
           *,
           weights: Optional[jnp.ndarray] = None,
           std_min_value: float = 1e-6,
           std_max_value: float = 1e6,
           pmap_axis_name: Optional[str] = None,
           validate_shapes: bool = True) -> RunningStatisticsState:

  # We require exactly the same structure to avoid issues when flattened
  # batch and state have different order of elements.
  assert jax.tree_util.tree_structure(batch) == jax.tree_util.tree_structure(state.mean)
  batch_shape = jax.tree_util.tree_leaves(batch)[0].shape
  # We assume the batch dimensions always go first.
  batch_dims = batch_shape[:len(batch_shape) -
                           jax.tree_util.tree_leaves(state.mean)[0].ndim]
  batch_axis = range(len(batch_dims))
  if weights is None:
    step_increment = jnp.prod(jnp.array(batch_dims))
  else:
    step_increment = jnp.sum(weights)
  if pmap_axis_name is not None:
    step_increment = jax.lax.psum(step_increment, axis_name=pmap_axis_name)
  count = state.count + step_increment

  # Validation is important. If the shapes don't match exactly, but are
  # compatible, arrays will be silently broadcasted resulting in incorrect
  # statistics.
  if validate_shapes:
    if weights is not None:
      if weights.shape != batch_dims:
        raise ValueError(f'{weights.shape} != {batch_dims}')
    _validate_batch_shapes(batch, state.mean, batch_dims)

  def _compute_node_statistics(
      mean: jnp.ndarray, summed_variance: jnp.ndarray,
      batch: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert isinstance(mean, jnp.ndarray), type(mean)
    assert isinstance(summed_variance, jnp.ndarray), type(summed_variance)
    # The mean and the sum of past variances are updated with Welford's
    # algorithm using batches (see https://stackoverflow.com/q/56402955).
    diff_to_old_mean = batch - mean
    if weights is not None:
      expanded_weights = jnp.reshape(
          weights,
          list(weights.shape) + [1] * (batch.ndim - weights.ndim))
      diff_to_old_mean = diff_to_old_mean * expanded_weights
    mean_update = jnp.sum(diff_to_old_mean, axis=batch_axis) / count
    if pmap_axis_name is not None:
      mean_update = jax.lax.psum(
          mean_update, axis_name=pmap_axis_name)
    mean = mean + mean_update

    diff_to_new_mean = batch - mean
    variance_update = diff_to_old_mean * diff_to_new_mean
    variance_update = jnp.sum(variance_update, axis=batch_axis)
    if pmap_axis_name is not None:
      variance_update = jax.lax.psum(variance_update, axis_name=pmap_axis_name)
    summed_variance = summed_variance + variance_update
    return mean, summed_variance

  updated_stats = jax.tree_util.tree_map(_compute_node_statistics, state.mean,
                                         state.summed_variance, batch)
  # Extract `mean` and `summed_variance` from `updated_stats` nest.
  mean = jax.tree_util.tree_map(lambda _, x: x[0], state.mean, updated_stats)
  summed_variance = jax.tree_util.tree_map(lambda _, x: x[1], state.mean,
                                           updated_stats)

  def compute_std(summed_variance: jnp.ndarray,
                  std: jnp.ndarray) -> jnp.ndarray:
    assert isinstance(summed_variance, jnp.ndarray)
    # Summed variance can get negative due to rounding errors.
    summed_variance = jnp.maximum(summed_variance, 0)
    std = jnp.sqrt(summed_variance / count)
    std = jnp.clip(std, std_min_value, std_max_value)
    return std

  std = jax.tree_util.tree_map(compute_std, summed_variance, state.std)

  return RunningStatisticsState(
      count=count, mean=mean, summed_variance=summed_variance, std=std)


def normalize(batch: NestedArray,
              mean_std: NestedMeanStd,
              max_abs_value: Optional[float] = None) -> NestedArray:
  """Normalizes data using running statistics."""

  def normalize_leaf(data: jnp.ndarray, mean: jnp.ndarray,
                     std: jnp.ndarray) -> jnp.ndarray:
    # Only normalize inexact
    if not jnp.issubdtype(data.dtype, jnp.inexact):
      return data
    data = (data - mean) / std
    if max_abs_value is not None:
      # TODO: remove pylint directive
      data = jnp.clip(data, -max_abs_value, +max_abs_value)
    return data

  return jax.tree_util.tree_map(normalize_leaf, batch, mean_std.mean, mean_std.std)


def denormalize(batch: NestedArray,
                mean_std: NestedMeanStd) -> NestedArray:
  """Denormalizes values in a nested structure using the given mean/std.

  Only values of inexact types are denormalized.
  See https://numpy.org/doc/stable/_images/dtype-hierarchy.png for Numpy type
  hierarchy.

  Args:
    batch: a nested structure containing batch of data.
    mean_std: mean and standard deviation used for denormalization.

  Returns:
    Nested structure with denormalized values.
  """

  def denormalize_leaf(data: jnp.ndarray, mean: jnp.ndarray,
                       std: jnp.ndarray) -> jnp.ndarray:
    # Only denormalize inexact
    if not jnp.issubdtype(data.dtype, jnp.inexact):
      return data
    return data * std + mean

  return jax.tree_util.tree_map(denormalize_leaf, batch, mean_std.mean, mean_std.std)
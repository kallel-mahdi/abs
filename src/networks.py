
import dataclasses
from typing import Any, Callable, Optional, Sequence, Tuple
import chex
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

import flax.linen as nn
import flax.linen as nn
import jax.numpy as jnp

from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]
default_init = nn.initializers.xavier_uniform

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:

        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size,
                             kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class StateActionValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, *args,
                 **kwargs) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)
        
        value = nn.tanh(value)

        return jnp.squeeze(value, -1)


class Ensemble(nn.Module):
    net_cls: nn.Module
    num: int = 2

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(self.net_cls,
                           variable_axes={'params': 0},
                           split_rngs={
                               'params': True,
                               'dropout': True
                           },
                           in_axes=None,
                           out_axes=0,
                           axis_size=self.num)
        return ensemble()(*args, **kwargs)



    


class PolicyNetwork(nn.Module):
    """Simple MLP Wrapper with flexible output head."""

    
    base_cls: nn.Module
    num_output_units : int
    output_activation : str = "tanh"
    use_bias : bool = False

    @nn.compact
    def __call__(
        self, x: chex.Array, *args,**kwargs) -> chex.Array:
        
        #x = self.base_cls()(x, *args, **kwargs)
        
        x = nn.Dense(features=self.num_output_units, kernel_init=default_init(),use_bias=self.use_bias)(x)
        
        if self.output_activation == "identity":
            return x
            
        elif self.output_activation == "tanh":
            return nn.tanh(x)

        elif self.output_activation == "cartpole":
            return 3*nn.tanh(x)
        
        else : 
            
            raise ValueError


class PendulumPolicyNetwork(nn.Module):
    """Simple MLP Wrapper with flexible output head."""

    
    base_cls: nn.Module
    num_output_units : int
    output_activation : str = "tanh"
    use_bias : bool = False

    @nn.compact
    def __call__(
        self, x: chex.Array, *args,**kwargs) -> chex.Array:


        #jax.debug.print(f'xxxxxxxxxx {x.shape}')
        angle = jax.lax.atan2(x[...,0],x[...,1])   
        z = jnp.vstack([angle,x[...,2]]).T
        z = nn.Dense(features=self.num_output_units,use_bias=self.use_bias)(z)
        
        
        z= 2*nn.tanh(z)
        
        if x.shape[0] in ([1,3]) :
            z = z.squeeze(axis=1)
        

        

        #jax.debug.print(f'zzzzzzzzz {z.shape}')

        return z



###################################""
@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]




@struct.dataclass
class ActorState: 
    actor_params: chex.Array
    obs_params : chex.Array


def make_normalized_network(
    network : nn.Module,
    preprocess_observations_fn,
    ) -> FeedForwardNetwork:
  """Creates a policy network."""
  
  
  def apply(policy_params,obs_params, obs):
    obs = preprocess_observations_fn(obs, obs_params)
    return network.apply(policy_params, obs)

  return FeedForwardNetwork(
      init=lambda rngs,inputs: network.init(rngs, inputs), apply=apply)
  


def make_normalized_qnetwork(
    network : nn.Module,
    preprocess_observations_fn,
    ) -> FeedForwardNetwork:
  """Creates a policy network."""
  
  
  
  def apply(params,observations,actions,training=False,rngs=None,obs_params=None):
    observations = preprocess_observations_fn(observations, obs_params)
    return network.apply(params,observations,actions,training,rngs=rngs)

  return FeedForwardNetwork(
      init=lambda rngs,obs,actions: network.init(rngs, obs,actions), apply=apply)

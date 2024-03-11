from functools import partial
from typing import Optional

import chex
import jax
import numpy as np
import jax.numpy as jnp
from flax import struct

from src.utils import *


class RBFKernel():
        
    
    def __init__(self,use_ard,ard_num_dims,lengthscale_prior,
                 *args,**kwargs):
        
        if use_ard == False :
            ard_num_dims = 1
        
        self.ard_num_dims = ard_num_dims
        
      

        self.state = None
        
        self.l_b = {
          
            "lengthscales":lengthscale_prior.a,
            "outputscale":0., ## placeholder
            "noise":0., ## placeholder
        }

        self.u_b = {
            "lengthscales":lengthscale_prior.b,
            "outputscale":0., ## placeholder
            "noise":0., ## placeholder
        }
        
        
        self.prior_mean =jax.tree_util.tree_map(lambda l_b,u_b : (l_b+u_b)/2,self.l_b,self.u_b)
        
        self.init_params= {
            "lengthscales" : self.prior_mean["lengthscales"]*jnp.ones(ard_num_dims),
            ### Just placeholder values they are modified at initial iteration
            "outputscale" : jnp.array(0.02),
            "noise" : jnp.array(0.01),
            }

    
    
    def _prepare_inputs(self,kernel_state,x1,x2):
        
        return x1,x2
    
    def _forward_inputs_hps(self,kernel_params,kernel_state,
                 x1,x2):
        
        
        n = x1.shape[0]
        
       
        x1 = jnp.divide(x1,kernel_params["lengthscales"]*jnp.sqrt(self.ard_num_dims))
        x2 = jnp.divide(x2,kernel_params["lengthscales"]*jnp.sqrt(self.ard_num_dims))
        
        rslt = (kernel_params["outputscale"]**2) * rbf(x1,x2)
        rslt += (kernel_params["noise"]**2) * is_equal(x1,x2)
        
        return rslt
        
    
    def __call__(self,kernel_params,kernel_state,
                 x1,x2):
        
        
        x1,x2 = self._prepare_inputs(kernel_state,x1,x2)
        rslt = self._forward_inputs_hps(kernel_params,kernel_state,x1,x2)
        
        
        return rslt




      


  
    

      


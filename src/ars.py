import jax
import optax

from src.kernels import *
from src.utils import CodeTimer
import time

class ARS:

  def __init__(self,
               n_info,n_dims,
               sigma,learning_rate,top_directions,
               num_rollouts,num_learning_steps,
              
               *args,**kwargs):
    

    # Initialize optimizer
    total_steps = 200
    optimizer = optax.sgd(learning_rate=learning_rate)
    opt_state = optimizer.init(jnp.zeros((n_dims)))

    num_rollouts_center = 2 ## (this is just for validation purpouses does not count in the plots)
    n_samples = 0

    self.__dict__.update(locals())
    
  
  def gradient_step(self,x,X,y,
                    optimizer,optimizer_state):
    
    
    top_directions = self.top_directions
    noise = self.noise
    eval_scores = y
    ###################################
    eval_scores = jnp.reshape(eval_scores, [-1])
    reward_plus, reward_minus = jnp.split(eval_scores, 2, axis=0)
    reward_max = jnp.maximum(reward_plus, reward_minus)
    reward_rank = jnp.argsort(jnp.argsort(-reward_max))
    reward_weight = jnp.where(reward_rank < top_directions, 1, 0)
    reward_weight_double = jnp.concatenate([reward_weight, reward_weight],
                                           axis=0)
    reward_std = jnp.std(eval_scores, where=reward_weight_double)
    ###################################
    grad = jnp.sum(
            jnp.transpose(
                jnp.transpose(noise) * reward_weight *
                (reward_plus - reward_minus)),
            axis=0)
    

    
    
    grad = grad / (top_directions * reward_std)
    
    ### When STD = 0 usually it means convergence :) 
    if reward_std.sum() == 0:
      grad = 0
    
  
    
    
    updates, optimizer_state = optimizer.update(-grad,optimizer_state)
    x = optax.apply_updates(x, updates)



 
    return x,optimizer_state
  
  def ask_neighbours(self,rng):
    
    """`ask` for new parameter candidates to evaluate next."""
    
    with CodeTimer("Gp ask neighbours"):
      
    
      x = self.x  
      
      pos_noise = jax.random.normal(rng, (self.n_info//2,self.n_dims))
      all_noise = jnp.vstack([pos_noise,-pos_noise])
      
      X_acq = x +  self.sigma * all_noise
      
      #### Placeholders
      X_acq_info = 0
      info_gain = 0  
      
      self.noise = pos_noise
      
      
      
    
    return X_acq,X_acq_info,info_gain
  

  def ask_local(self):

    x_old = self.x
    
    with CodeTimer("Gp ask local"):
      
      x_new,self.opt_state = self.gradient_step(x_old,
                                                self.X,self.y,self.optimizer,self.opt_state)
    
    return x_new,0
  

  def tell_local(self,rng,x_t,y_t,
                  *args,**kwargs):
    
    
    self.x = x_t
    self.y_t = y_t
    
   
      

  def tell_neighbours(self,rng,X_acq,y_acq,
                      *args,**kwargs):
    
    self.X = X_acq
    self.y = y_acq

  
        
          
      


      
    
  


    



